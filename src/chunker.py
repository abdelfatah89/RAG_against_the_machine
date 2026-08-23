from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter)
from .models import MinimalSource
from typing import List
from abc import ABC
from pathlib import Path
from typing import Tuple


SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}


class Chunk(MinimalSource):
    def __init__(self, file_path: str, first_character_index: int,
                 last_character_index: int, content: str,
                 file_type: str, metadata: dict = dict()):
        super().__init__(
            file_path=file_path,
            first_character_index=first_character_index,
            last_character_index=last_character_index,
            content=content,
            file_type=file_type,
            metadata=metadata
        )


class Chunker(ABC):
    def __init__(self):
        self.max_chunk_size = 2000

    def chunk(self, content: str) -> List[Chunk]:
        return []


class PythonChunker(Chunker):
    def chunk(self, path: str) -> List[Chunk]:
        search_from = 0
        with open(path, "r") as f:
            content = f.read()

        return []


class MarkdownChunker(Chunker):
    def chunk(self, path: str) -> List[Chunk]:
        search_from = 0
        chunk_objects = []
        with open(path, "r") as f:
            content = f.read()
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

        documents = splitter.split_text(content)
        for doc in documents:
            chunk_objects = TextChunker().get_chunks(
                path, doc.page_content, "md", search_from, doc.metadata)

        return chunk_objects


class TextChunker(Chunker):
    def chunk(self, path: str) -> List[Chunk]:
        search_from = 0
        with open(path, "r") as f:
            content = f.read()
        return self.get_chunks(path, content, "txt", search_from)

    def get_chunks(self, path: str, content: str, type_: str,
                   search_from: int, metadata: dict = dict()) -> List[Chunk]:
        chunk_objects = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.max_chunk_size // 10,
        )

        chunks = splitter.split_text(content)
        for chunk in chunks:
            start = content.find(chunk, search_from)
            end = start + len(chunk)
            chunk_obj = Chunk(path, start, end, chunk, type_, metadata)
            chunk_objects.append(chunk_obj)
        return chunk_objects


class ChunksFactory:
    def __init__(self):
        self.py_chunker = PythonChunker()
        self.md_chunker = MarkdownChunker()
        self.txt_chunker = TextChunker()

    def get_files(self, data_dir: str
                  ) -> Tuple[List[Path], List[Path], List[Path]]:
        root = Path(data_dir)

        py_files = []
        md_files = []
        txt_files = []
        for path in root.rglob("*"):
            if (path.is_file() and
                    path.suffix.lower() not in SUPPORTED_EXTENSIONS):
                continue
            if path.is_file() and path.suffix.lower() == ".py":
                py_files.append(path)
            elif path.is_file() and path.suffix.lower() == ".md":
                md_files.append(path)
            elif path.is_file() and path.suffix.lower() == ".txt":
                txt_files.append(path)

        return py_files, md_files, txt_files

    def get_chunks(self, data_dir: str
                   ) -> Tuple[List[Chunk], List[Chunk], List[Chunk]]:
        py_files, md_files, txt_files = self.get_files(data_dir)

        for file in py_files:
            py_chunks = self.py_chunker.chunk(file)
        for file in md_files:
            md_chunks = self.md_chunker.chunk(file)
        for file in txt_files:
            txt_chunks = self.txt_chunker.chunk(file)

        return py_chunks, md_chunks, txt_chunks
