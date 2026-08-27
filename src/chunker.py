from typing import List, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import ast
from tqdm import tqdm  # type: ignore[import-untyped]

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language)

from .models import MinimalSource
from .file_manager import FileManager
from .tools import get_keywords_args


SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}


class Chunk(MinimalSource):
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str = ""
    file_type: str = ""
    metadata: dict = {}
    score: float = 0.0


class Chunker(ABC):
    def __init__(self):
        self.max_chunk_size = 2000
        self.overlap_size = self.max_chunk_size // 10

    @abstractmethod
    def chunk(self, path: str) -> List[Chunk]:
        pass  # This method should be implemented in subclasses

    def content_islonger(self, chunks: List[Chunk]) -> bool:
        return any(
            len(chunk.content) > self.max_chunk_size
            for chunk in chunks
            )


class PythonChunker(Chunker):
    def __init__(self, max_chunk_size: int = 2000):
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.overlap_size = self.max_chunk_size // 10

    def get_offsets(self, source: str, node: ast.stmt) -> tuple[int, int]:
        lines = source.splitlines(keepends=True)

        start_lineno = getattr(node, "lineno", 1)
        start_col = getattr(node, "col_offset", 0)
        end_lineno = getattr(node, "end_lineno", start_lineno)
        end_col = getattr(node, "end_col_offset", start_col)

        start = sum(len(line) for line in lines[:start_lineno - 1])
        start += start_col

        end = sum(len(line) for line in lines[:end_lineno - 1])
        end += end_col

        return start, end

    def split_chunk(self, chunk: Chunk) -> List[Chunk]:
        search_from = 0
        chunk_objects = []
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap_size,
        )

        chunks = splitter.split_text(chunk.content)
        for c in chunks:
            start = (chunk.first_character_index +
                     chunk.content.find(c, search_from))
            end = start + len(c)
            args = get_keywords_args(chunk.file_path, start, end, c, "py")
            chunk_obj = Chunk(**args)
            chunk_objects.append(chunk_obj)

        return chunk_objects

    def chunk(self, path: str) -> List[Chunk]:
        chunk_objects = []
        with open(path, "r") as f:
            content = f.read()

        tree = ast.parse(content)
        for node in tree.body:
            start, end = self.get_offsets(content, node)
            if isinstance(node, ast.ClassDef):
                code = ast.get_source_segment(content, node) or ""
                args = get_keywords_args(
                    path, start, end, code, "py",
                    metadata={"type": "class", "class_name": node.name})
                chunk_obj = Chunk(**args)
                chunk_objects.append(chunk_obj)

                for child in node.body:
                    start, end = self.get_offsets(content, child)
                    if isinstance(child,
                                  (ast.FunctionDef, ast.AsyncFunctionDef)):
                        code = ast.get_source_segment(content, child) or ""
                        args = get_keywords_args(
                            path, start, end, code, "py",
                            metadata={
                                "type": "method",
                                "class_name": node.name,
                                "function_name": child.name})
                        chunk_obj = Chunk(**args)
                        chunk_objects.append(chunk_obj)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                code = ast.get_source_segment(content, node) or ""
                args = get_keywords_args(
                    path, start, end, code, "py",
                    metadata={"type": "function",
                              "function_name": node.name})
                chunk_obj = Chunk(**args)
                chunk_objects.append(chunk_obj)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                code = ast.get_source_segment(content, node) or ""
                args = get_keywords_args(
                    path, start, end, code, "py",
                    metadata={"type": "import"})
                chunk_obj = Chunk(**args)
                if (chunk_objects and
                        chunk_objects[-1].metadata.get("type") == "import"):
                    chunk_objects[-1].content += "\n" + chunk_obj.content
                    chunk_objects[-1].last_character_index = end
                else:
                    chunk_objects.append(chunk_obj)
            else:
                code = ast.get_source_segment(content, node) or ""
                args = get_keywords_args(
                    path, start, end, code, "py",
                    metadata={"type": "module-level code"})
                chunk_obj = Chunk(**args)
                if (chunk_objects and
                        chunk_objects[-1].metadata.get(
                            "type") == "module-level code"):
                    chunk_objects[-1].content += "\n" + chunk_obj.content
                    chunk_objects[-1].last_character_index = end
                else:
                    chunk_objects.append(chunk_obj)

        while self.content_islonger(chunk_objects):
            for chunk in chunk_objects:
                if len(chunk.content) > self.max_chunk_size:
                    sub_chunks = self.split_chunk(chunk)
                    chunk_objects.remove(chunk)
                    chunk_objects.extend(sub_chunks)

        return chunk_objects


class MarkdownChunker(Chunker):
    def __init__(self, max_chunk_size: int = 2000):
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.overlap_size = self.max_chunk_size // 10

    def split_chunk(self, chunk: Chunk) -> List[Chunk]:
        search_from = 0
        chunk_objects = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap_size,
        )

        chunks = splitter.split_text(chunk.content)
        for c in chunks:
            start = (chunk.first_character_index +
                     chunk.content.find(c, search_from))
            end = start + len(c)
            args = get_keywords_args(chunk.file_path, start, end, c, "md")
            chunk_obj = Chunk(**args)
            chunk_objects.append(chunk_obj)

        return chunk_objects

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

        chunks = splitter.split_text(content)
        for chunk in chunks:
            start = content.find(f"{chunk.page_content}", search_from)
            end = start + len(chunk.page_content)
            args = get_keywords_args(
                path, start, end, chunk.page_content, "md", chunk.metadata)
            chunk_obj = Chunk(**args)
            search_from = end
            chunk_objects.append(chunk_obj)

        while self.content_islonger(chunk_objects):
            for chunk_obj in chunk_objects:
                if len(chunk_obj.content) > self.max_chunk_size:
                    sub_chunks = self.split_chunk(chunk_obj)
                    chunk_objects.remove(chunk_obj)
                    chunk_objects.extend(sub_chunks)

        return chunk_objects


class TextChunker(Chunker):
    def __init__(self, max_chunk_size: int = 2000):
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.overlap_size = self.max_chunk_size // 10

    def chunk(self, path: str) -> List[Chunk]:
        search_from = 0
        chunk_objects = []

        with open(path, "r") as f:
            content = f.read()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap_size,
        )

        chunks = splitter.split_text(content)
        for chunk in chunks:
            start = content.find(chunk, search_from)
            end = start + len(chunk)
            args = get_keywords_args(path, start, end, chunk, "txt")
            chunk_obj = Chunk(**args)
            chunk_objects.append(chunk_obj)
        return chunk_objects


class ChunksFactory:
    def __init__(self,
                 data_dir: str = "data/raw",
                 max_chunk_size: int = 2000):
        self.data_dir = data_dir
        self.file_manager = FileManager(data_dir)
        self.py_chunker = PythonChunker(max_chunk_size=max_chunk_size)
        self.md_chunker = MarkdownChunker(max_chunk_size=max_chunk_size)
        self.txt_chunker = TextChunker(max_chunk_size=max_chunk_size)

    def get_files(self) -> Tuple[List[str], List[str], List[str]]:
        # FileManager.get_files() already filters to SUPPORTED_EXTENSIONS,
        # no need to re-check suffixes against it here.
        files = self.file_manager.get_files()
        py_files: List[str] = []
        md_files: List[str] = []
        txt_files: List[str] = []
        for path in files:
            file_path = Path(path)
            suffix = file_path.suffix.lower()
            if suffix == ".py":
                py_files.append(str(file_path))
            elif suffix == ".md":
                md_files.append(str(file_path))
            elif suffix == ".txt":
                txt_files.append(str(file_path))

        return py_files, md_files, txt_files

    def get_chunks(self) -> List[Chunk]:
        py_files, md_files, txt_files = self.get_files()
        py_chunks: List[Chunk] = []
        md_chunks: List[Chunk] = []
        txt_chunks: List[Chunk] = []

        for file in py_files:
            py_chunks += self.py_chunker.chunk(file)
        for file in md_files:
            md_chunks += self.md_chunker.chunk(file)
        for file in txt_files:
            txt_chunks += self.txt_chunker.chunk(file)

        chunks = py_chunks + md_chunks + txt_chunks
        chunks.sort(key=lambda x: x.file_type)

        for _ in tqdm(range(len(chunks)), desc="Chunking", unit="chunk"):
            continue

        return chunks

    def get_modified_chunks(self) -> List[Chunk]:
        modified_files = self.file_manager.get_modified_files()
        modified_chunks: List[Chunk] = []
        for file in modified_files:
            file_path = Path(file)
            suffix = file_path.suffix.lower()
            if suffix == ".py":
                modified_chunks += self.py_chunker.chunk(file)
            elif suffix == ".md":
                modified_chunks += self.md_chunker.chunk(file)
            elif suffix == ".txt":
                modified_chunks += self.txt_chunker.chunk(file)

        return modified_chunks
