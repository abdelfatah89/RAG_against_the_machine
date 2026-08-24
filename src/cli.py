from .chunker import Chunk, ChunksFactory
from typing import List
from .hybrid import HybridSearch


class CLI:
    def __init__(self):
        self.chunks: List[Chunk] = []

    def index(self, max_chunk_size: int) -> None:
        Chunker = ChunksFactory(max_chunk_size=max_chunk_size)
        chunks = Chunker.get_chunks("data/raw")
        self.chunks = chunks

    def search(self, query: str, k: int):
        self.index(2000)  # Index the documents before searching
        self.hybrid_search = HybridSearch(self.chunks)
        return self.hybrid_search.search(query, k)

    def search_dataset(
        self,
        dataset_path: str, k: int,
        save_directory: str
            ):
        pass

    def answer(
        self,
        query: str, k: int
            ):
        pass

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
            ):
        pass

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
            ):
        pass
