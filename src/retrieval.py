import json
from abc import ABC, abstractmethod
from typing import Dict, List
from tqdm import tqdm  # type: ignore[import-untyped]

import chromadb
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
import numpy as np

from .chunker import Chunk
from .models import MinimalSource
from .embedder import Embedder


class Retrieval(ABC):
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[MinimalSource]:
        pass  # This method should be implemented in subclasses


class BM25Retrieval(Retrieval):
    def __init__(self, chunks: List[Chunk]) -> None:
        super().__init__(chunks)

        self.tokenized_docs = [
            chunk.content.lower().split() for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, k: int = 3) -> List[MinimalSource]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        chunks: List[MinimalSource] = []

        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )[:k]

        progress = tqdm(
            desc="Retrieving BM25 Results", unit="chunk", total=k)
        for chunk, score in ranked:
            progress.update(1)
            chunks.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                    content=chunk.content,
                    file_type=chunk.file_type,
                    score=score,
                    )
                )
        progress.close()
        return chunks


class EmbeddingRetrieval(Retrieval):
    def __init__(self, chunks: List[Chunk]) -> None:
        super().__init__(chunks)
        self.client = chromadb.PersistentClient(path="chromadb")
        self.collection = self.client.get_or_create_collection("chunks")
        if self.collection.count() == 0:
            embedder = Embedder()
            embeddings = embedder.embed_batch(chunks)
            self.add_documents(chunks, embeddings)

    def add_documents(self,
                      documents: List[Chunk],
                      embeddings: List[np.ndarray]) -> None:
        progress = tqdm(
            desc="Adding Chunks to vector database",
            unit="chunk", total=len(documents))
        for i, document in enumerate(documents):
            progress.update(1)
            if document.file_type == "py" or document.file_type == "md":
                metadata_text = self.get_metadata(document.metadata)
                content = metadata_text + document.content
            else:
                content = document.content
            self.collection.add(
                documents=[content],
                embeddings=embeddings[i],
                ids=[f"chunk_{i}"],
                metadatas=[{
                    "file_path": document.file_path,
                    "first_character_index": document.first_character_index,
                    "last_character_index": document.last_character_index,
                    "content": document.content,
                    "file_type": document.file_type,
                }],
            )
        progress.close()

    def get_metadata(self, metadata: Dict[str, str]) -> str:
        metadata_text = "".join(
            [f"{key}: {value}" for key, value in metadata.items()])
        return metadata_text

    def retrieve(self, query: str, k: int = 3) -> List[MinimalSource]:
        chunks: List[MinimalSource] = []
        results = self.collection.query(query_texts=[query], n_results=k)
        metadatas_result = results.get("metadatas")
        distances_result = results.get("distances")

        if not metadatas_result or not distances_result:
            return chunks

        metadatas = metadatas_result[0]
        distances = distances_result[0]
        for metadata, distance in zip(metadatas, distances):
            chunk = MinimalSource(
                file_path=str(metadata["file_path"]),
                first_character_index=int(
                    str(metadata["first_character_index"])),
                last_character_index=int(
                    str(metadata["last_character_index"])),
                content=str(metadata["content"]),
                file_type=str(metadata["file_type"]),
                score=1.0 - float(distance),
            )
            chunks.append(chunk)
        return chunks


class RetrievalFactory:
    def create_retrieval(self, method: str,
                         chunks: List[Chunk],
                         embeddings: List[np.ndarray],
                         query: str, k: int = 3
                         ) -> List[MinimalSource]:

        if method == "bm25":
            if chunks is None:
                raise ValueError(
                    "Chunks must be provided for BM25 retrieval.")
            bm25 = BM25Retrieval(chunks)
            retrieved = bm25.retrieve(query, k)
            self.save_processed_data(
                retrieved, "data/processed/bm25_processed_data.json")
            return retrieved

        elif method == "embedding":
            embedding = EmbeddingRetrieval(chunks)
            retrieved = embedding.retrieve(query, k)
            self.save_processed_data(
                retrieved, "data/processed/embedding_processed_data.json")
            return retrieved
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

    def save_processed_data(self,
                            data: List[MinimalSource],
                            file_path: str
                            ) -> None:
        output_data = [item.model_dump() for item in data]
        with open(file_path, "w") as f:
            json.dump(output_data, f, indent=4)
