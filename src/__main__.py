from fire import Fire  # type: ignore[import-untyped]
from .cli import CLI
from .chunker import ChunksFactory
from .hybrid import HybridSearch


def main() -> None:
    cli = CLI()
    Fire(cli)

    Chunker = ChunksFactory()
    chunks = Chunker.get_chunks("data/raw")
    hybrid_search = HybridSearch(chunks)
    hybrid_search.search("What is the remote work policy?", k=3)


if __name__ == "__main__":
    main()
