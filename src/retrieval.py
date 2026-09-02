from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import chromadb
from rank_bm25 import BM25Okapi

from .tools import save_processed_data


from .chunker import Chunk
from .models import MinimalSource
from .bm25_tokenizer import tokenize


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
            tokenize(chunk.content) for chunk in self.chunks
        ]
        self.bm25 = (BM25Okapi(self.tokenized_docs)
                     if self.tokenized_docs else None)

    @staticmethod
    def _rank_score(chunk: Chunk, score: float) -> float:
        """Prefer official docs when BM25 scores are otherwise close.

        Documentation questions often retrieve the correct markdown chunk in
        the top 10, but Python examples/tests can outrank it by a small margin
        because they repeat the same identifiers. This boost keeps lexical
        ranking dominant while moving close docs matches into the top 5.
        """
        if "/docs/" in chunk.file_path and chunk.file_type == "md":
            score += 2.0
        return score

    def retrieve(self, query: str, k: int = 3) -> List[MinimalSource]:
        if k <= 0 or self.bm25 is None:
            return []

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        chunks: List[MinimalSource] = []

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda pair: self._rank_score(pair[0], pair[1]),
            reverse=True,
        )[:k]

        for chunk, score in ranked:
            rank_score = self._rank_score(chunk, score)
            chunks.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                    content=chunk.content,
                    file_type=chunk.file_type,
                    score=rank_score,
                    )
                )
        return chunks


class EmbeddingRetrieval(Retrieval):
    def __init__(self, chunks: List[Chunk]) -> None:
        super().__init__(chunks)
        self.client = chromadb.PersistentClient(path="chromadb")
        self.collection = self.client.get_or_create_collection("chunks")

    def retrieve(self, query: str, k: int = 3) -> List[MinimalSource]:
        if k <= 0:
            return []

        from .embedder import Embedder

        embeddings = Embedder().embed(query)
        chunks: List[MinimalSource] = []
        results = self.collection.query(
            query_embeddings=embeddings, n_results=k)
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


class HybridRetrieval:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Retrieval(chunks)
        self.embedding = EmbeddingRetrieval(chunks)

        self.bm25_results: List[MinimalSource] = []
        self.embedding_results: List[MinimalSource] = []

    def generate_results(self, query: str, k: int = 3) -> None:
        self.bm25_results = self.bm25.retrieve(query, k)
        self.embedding_results = self.embedding.retrieve(query, k)

    @staticmethod
    def _key(result: MinimalSource) -> Tuple[str, int, int]:
        return (
            result.file_path,
            result.first_character_index,
            result.last_character_index,
        )

    @staticmethod
    def _normalize(results: List[MinimalSource]
                   ) -> Dict[Tuple[str, int, int], float]:
        if not results:
            return {}

        scored = {HybridRetrieval._key(r): r.score for r in results}
        lo = min(scored.values())
        hi = max(scored.values())
        span = (hi - lo) or 1.0

        return {key: (score - lo) / span for key, score in scored.items()}

    def retrieve(self, query: str, k: int = 3,
                 bm25_factor: float = 0.3,
                 embedding_factor: float = 0.7
                 ) -> List[MinimalSource]:

        self.generate_results(query, k)

        bm25_by_key = {self._key(r): r for r in self.bm25_results}
        embedding_by_key = {self._key(r): r for r in self.embedding_results}

        bm25_norm = self._normalize(self.bm25_results)
        embedding_norm = self._normalize(self.embedding_results)

        all_keys = set(bm25_by_key) | set(embedding_by_key)

        hybrid_results: List[MinimalSource] = []
        for key in all_keys:
            base = bm25_by_key.get(key) or embedding_by_key.get(key)
            assert base is not None

            bm25_score = bm25_norm.get(key, 0.0)
            embedding_score = embedding_norm.get(key, 0.0)
            hybrid_score = (
                (bm25_factor * bm25_score) +
                (embedding_factor * embedding_score)
            )

            hybrid_results.append(
                base.model_copy(update={"score": hybrid_score})
            )

        hybrid_results.sort(key=lambda r: r.score, reverse=True)
        hybrid_results = hybrid_results[:k]

        save_processed_data(
            hybrid_results, "data/processed/hybrid_processed_data.json")

        return hybrid_results
