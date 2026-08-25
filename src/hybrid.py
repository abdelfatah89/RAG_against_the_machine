from typing import Dict, List, Tuple

from .chunker import Chunk
from .models import MinimalSource
from .retrieval import BM25Retrieval, EmbeddingRetrieval, RetrievalFactory


class HybridSearch:
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

        scored = {HybridSearch._key(r): r.score for r in results}
        lo = min(scored.values())
        hi = max(scored.values())
        span = (hi - lo) or 1.0

        return {key: (score - lo) / span for key, score in scored.items()}

    def search(self, query: str, k: int = 3,
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

        self.save_processed_data(
            hybrid_results, "data/processed/hybrid_processed_data.json")

        return hybrid_results

    def save_processed_data(self,
                            data: List[MinimalSource],
                            file_path: str
                            ) -> None:
        RetrievalFactory().save_processed_data(data, file_path)
