from .models import MinimalSource
from typing import List
from abc import ABC
from pathlib import Path
from typing import Tuple
from .markdown_chunker import MarkdownChunker


SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}


class Chunk(MinimalSource):
    content: str
    file_type: str


class Chunker(ABC):
    def __init__(self):
        self.max_chunk_size = 2000

    def chunk(self, content: str) -> List[Chunk]:
        return []


class PythonChunker(Chunker):
    def chunk(self, content: str) -> List[Chunk]:
        return []


class ChunksFactory:
    def __init__(self):
        self.py_chunker = PythonChunker()
        self.md_chunker = MarkdownChunker()

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
            txt_chunks = self.md_chunker.chunk(file)

        return py_chunks, md_chunks, txt_chunks
