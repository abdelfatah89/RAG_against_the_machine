from pathlib import Path
from typing import Set, List
import json

from .chunker import ChunksFactory, Chunk
from .file_manager import FileManager
from .vectordb import VectorDB


PROCESSED_CHUNKS_PATH = Path("data/processed/processed_chunks.json")


class Indexer:
    def __init__(self, data_dir: str = "data/raw",
                 max_chunk_size: int = 2000):
        self.file_manager = FileManager(data_dir)
        self.chunks_factory = ChunksFactory(
            data_dir=data_dir, max_chunk_size=max_chunk_size)
        # Share one FileManager so hash lookups aren't recomputed twice.
        self.chunks_factory.file_manager = self.file_manager
        self.vectordb: VectorDB | None = None

    def run(self, re: bool = False, embed: bool = False) -> List[Chunk]:
        if re:
            return self._full_index(re=True, embed=embed)
        if not PROCESSED_CHUNKS_PATH.is_file():
            return self._full_index(embed=embed)

        if not self.file_manager.modified_files_exist():
            print("No modified files detected. Nothing to index.")
            return self._load_chunks()

        return self._incremental_index(embed)

    def _full_index(self,
                    re: bool = False,
                    embed: bool = False) -> List[Chunk]:
        if re:
            print("Re-indexing all files...")
        else:
            print("No existing processed chunks found. Indexing all files...")
        chunks = self.chunks_factory.get_chunks()

        self._save_chunks(chunks)

        if embed:
            from .embedder import Embedder

            self.vectordb = VectorDB()
            embedder = Embedder()
            embeddings = embedder.embed_batch(chunks)
            self.vectordb.reset()
            self.vectordb.add_documents(chunks, embeddings)

        self._commit_hashes()
        return chunks

    def _incremental_index(self, embed: bool = False) -> List[Chunk]:
        modified_chunks = self.chunks_factory.get_modified_chunks()
        deleted_files = self.file_manager.get_deleted_files()
        stale_paths: Set[str] = (
            {c.file_path for c in modified_chunks} | set(deleted_files)
        )
        print(f" {len(stale_paths)} modified files detected."
              " Re-indexing modified files...")

        self._save_modified_chunks(modified_chunks, stale_paths)

        if stale_paths and embed:
            if self.vectordb is None:
                self.vectordb = VectorDB()
            assert self.vectordb is not None
            self.vectordb.delete_by_paths(list(stale_paths))
            print(f"Deleted {len(stale_paths)} stale file(s)"
                  " from vector database.")
        if modified_chunks and embed:
            from .embedder import Embedder

            if self.vectordb is None:
                self.vectordb = VectorDB()
            embedder = Embedder()
            embeddings = embedder.embed_batch(modified_chunks)
            self.vectordb.add_documents(modified_chunks, embeddings)
            print(f"{len(modified_chunks)} Added modified chunks"
                  " to vector database.")

        self._commit_hashes()
        return self._load_chunks()

    def _save_chunks(self, chunks: List[Chunk]) -> None:
        PROCESSED_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        output_data = [chunk.model_dump() for chunk in chunks]
        with open(PROCESSED_CHUNKS_PATH, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Indexed {len(chunks)} chunks under data/processed/")

    def _save_modified_chunks(self,
                              modified_chunks: List[Chunk],
                              stale_paths: Set[str]) -> None:
        if PROCESSED_CHUNKS_PATH.is_file():
            with open(PROCESSED_CHUNKS_PATH, "r") as f:
                output_data = json.load(f)
        else:
            output_data = []

        output_data = [
            c for c in output_data if c["file_path"] not in stale_paths
        ]
        output_data.extend(chunk.model_dump() for chunk in modified_chunks)

        with open(PROCESSED_CHUNKS_PATH, "w") as f:
            json.dump(output_data, f, indent=4)

    def _load_chunks(self) -> List[Chunk]:
        with open(PROCESSED_CHUNKS_PATH, "r") as f:
            data = json.load(f)
        chunks = [Chunk(**item) for item in data]
        print(f"Loaded {len(chunks)} chunks from"
              " data/processed/processed_chunks.json")
        return chunks

    def _commit_hashes(self) -> None:
        self.file_manager.save_hashes(self.file_manager.get_current_hashes())
