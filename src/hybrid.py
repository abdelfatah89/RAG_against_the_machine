from .retrieval import RetrievalFactory
from .models import MinimalSource
from .chunker import Chunk
from typing import List


class HybridSearch:
    def __init__(self, chunks: List[Chunk]):
        self.retrieval_factory = RetrievalFactory()
        self.chunks = chunks
        self.bm25_results: List[MinimalSource] = []
        self.embedding_results: List[MinimalSource] = []

    def generate_results(self, query: str, k: int = 3) -> None:
        bm25_results = self.retrieval_factory.create_retrieval(
            "bm25", self.chunks, query, k)

        embedding_results = self.retrieval_factory.create_retrieval(
            "embedding", self.chunks, query, k)

        self.bm25_results = bm25_results
        self.embedding_results = embedding_results

        return

    def search(self, query: str, k: int = 3,
               bm25_factor: float = 0.3,
               embedding_factor: float = 0.7
               ) -> List[MinimalSource]:

        self.generate_results(query, k)
        hybrid_results = []
        for result in self.bm25_results:
            if result in self.embedding_results:
                index = self.embedding_results.index(result)
                hybrid_score = (
                    (bm25_factor * result.score) +
                    (embedding_factor * self.embedding_results[index].score)
                    )
                result.score = hybrid_score
                hybrid_results.append(result)
            else:
                hybrid_results.append(result)

        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        if len(hybrid_results) > k:
            hybrid_results = hybrid_results[:k]

        hybrid_json = [item.model_dump_json() for item in hybrid_results]
        with open("hybrid_processed_data.json", "w") as f:
            import json
            json.dump(hybrid_json, f, indent=4)

        return hybrid_results
