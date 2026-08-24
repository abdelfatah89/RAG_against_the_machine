from abc import ABC, abstractmethod
from .models import MinimalSource
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from typing import Dict, List
from .chunker import Chunk
import chromadb


class Retrieval(ABC):
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[MinimalSource]:
        pass  # This method should be implemented in subclasses


class BM25Retrieval:
    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks

        self.tokenized_docs = [
            chunk.content.lower().split() for chunk in self.chunks
            ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self,
                 query: str, k: int = 3) -> List[MinimalSource]:

        scores = self.bm25.get_scores(query.lower().split())
        top_k = self.bm25.get_top_n(
            query.lower().split(),
            self.chunks,
            n=k
            )

        return [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
                content=chunk.content,
                score=scores[self.chunks.index(chunk)],
                ) for chunk in top_k
            ]


class EmbeddingRetrieval:
    def __init__(self, chunks: List[Chunk]):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("policies")
        self.add_documents(chunks)

    def add_documents(self, documents: List[Chunk]):
        for i, document in enumerate(documents):
            if document.file_type == "py" or document.file_type == "md":
                metadata = self.get_metadata(document.metadata)
                content = metadata + document.content
            else:
                content = document.content
            self.collection.add(
                documents=[content],
                ids=[f"document_{i}"],
                metadatas={
                    "path": document.file_path,
                    "start": document.first_character_index,
                    "end": document.last_character_index,
                    "content": document.content
                    }
            )

    def get_metadata(self, metadata: Dict[str, str]) -> str:
        metadata_text = "".join(
            [f"{key}: {value}" for key, value in metadata.items()])
        return metadata_text

    def retrieve(self, query: str, k: int = 3) -> List[MinimalSource]:
        chunks: List[MinimalSource] = []
        results = self.collection.query(query_texts=[query], n_results=k)
        for r in results:
            print(r["metadatas"])
            chunk = MinimalSource(**r["metadatas"])
            chunk.score = 1 - r["distances"][0]
            chunks.append(chunk)
        return chunks


class RetrievalFactory:
    def create_retrieval(self, method: str, chunks: List[Chunk],
                         query: str, k: int = 3
                         ) -> List[MinimalSource]:

        chunks_text: List[str] = []
        if method == "bm25":
            if chunks is None:
                raise ValueError("Chunks must be provided for BM25 retrieval.")
            bm25 = BM25Retrieval(chunks)
            retrieved = bm25.retrieve(query, k)
            for r in retrieved:
                chunks_text.append(r.content)
            self.save_processed_data(
                retrieved, "bm25_processed_data.json")
            return retrieved

        elif method == "embedding":
            embedding = EmbeddingRetrieval(chunks)
            retrieved = embedding.retrieve(query, k)
            for r in retrieved:
                chunks_text.append(r.content)
            self.save_processed_data(
                retrieved, "embedding_processed_data.json")
            return retrieved
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

    def save_processed_data(self,
                            data: List[MinimalSource],
                            file_path: str
                            ) -> None:
        import json

        output_data = [item.model_dump_json() for item in data]
        with open(file_path, "w") as f:
            json.dump(output_data, f, indent=4)
