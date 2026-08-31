from typing import List
import chromadb
from tqdm import tqdm  # type: ignore[import-untyped]

from .chunker import Chunk


class VectorDB:
    """Thin wrapper around the Chroma collection. This is the ONLY class
    that writes to the vector database -- retrievers only query it."""

    def __init__(self,
                 path: str = "chromadb",
                 collection_name: str = "chunks"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(collection_name)

    @staticmethod
    def _make_id(document: Chunk) -> str:
        return (f"{document.file_path}::"
                f"{document.first_character_index}::"
                f"{document.last_character_index}")

    @staticmethod
    def _display_text(document: Chunk) -> str:
        if document.file_type in ("py", "md") and document.metadata:
            metadata_text = "".join(
                f"{key}: {value}" for key, value in document.metadata.items()
            )
            return metadata_text + document.content
        return document.content

    def add_documents(self, documents: List[Chunk], embeddings: List) -> None:
        for chunk, embedding in tqdm(
                list(zip(documents, embeddings)),
                desc="Adding chunks to vector database",
                unit="chunk"):
            self.collection.add(
                documents=[self._display_text(chunk)],
                embeddings=[embedding],
                ids=[self._make_id(chunk)],
                metadatas=[{
                    "file_path": chunk.file_path,
                    "first_character_index": chunk.first_character_index,
                    "last_character_index": chunk.last_character_index,
                    "content": chunk.content,
                    "file_type": chunk.file_type,
                }],
            )

    def delete_by_paths(self, paths: List[str]) -> None:
        for path in paths:
            self.collection.delete(where={"file_path": path})

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            self.collection_name)
