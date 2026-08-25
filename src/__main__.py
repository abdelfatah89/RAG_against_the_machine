from fire import Fire  # type: ignore[import-untyped]
from typing import List

from .cli import CLI
from .chunker import ChunksFactory
from .hybrid import HybridSearch
from .retrieval import BM25Retrieval


def main() -> None:
    # cli = CLI()
    # Fire(cli)

    Chunker = ChunksFactory()
    chunks = Chunker.get_chunks("data/raw")

    retrieval = BM25Retrieval(chunks)
    results = retrieval.retrieve("How to use OpenAI APIs?", k=3)


if __name__ == "__main__":
    main()
