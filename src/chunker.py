from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language)
from .models import MinimalSource
from typing import List
from abc import ABC
from pathlib import Path
from typing import Tuple
import ast


SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}


class Chunk(MinimalSource):
    def __init__(self, file_path: str, first_character_index: int,
                 last_character_index: int, content: str, file_type: str,
                 metadata: dict = dict()):
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
        self.overlap_size = self.max_chunk_size // 10

    def chunk(self, content: str) -> List[Chunk]:
        return []


class PythonChunker(Chunker):
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
            chunk_obj = Chunk(chunk.file_path, start, end, c, "txt")
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
                chunk_obj = Chunk(path, start, end, code, "py",
                                  metadata={"type": "class",
                                            "class_name": node.name})
                chunk_objects.append(chunk_obj)

                for child in node.body:
                    start, end = self.get_offsets(content, child)
                    if isinstance(child,
                                  (ast.FunctionDef, ast.AsyncFunctionDef)):
                        code = ast.get_source_segment(content, child) or ""
                        chunk_obj = Chunk(path, start, end, code, "py",
                                          metadata={
                                              "type": "method",
                                              "class_name": node.name,
                                              "function_name": child.name})
                        chunk_objects.append(chunk_obj)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                code = ast.get_source_segment(content, node) or ""
                chunk_obj = Chunk(path, start, end, code, "py",
                                  metadata={"type": "function",
                                            "function_name": node.name})
                chunk_objects.append(chunk_obj)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                code = ast.get_source_segment(content, node) or ""
                chunk_obj = Chunk(path, start, end, code, "py",
                                  metadata={"type": "import"})
                if (chunk_objects and
                        chunk_objects[-1].metadata.get("type") == "import"):
                    chunk_objects[-1].content += "\n" + chunk_obj.content
                    chunk_objects[-1].last_character_index = end
                else:
                    chunk_objects.append(chunk_obj)
            else:
                code = ast.get_source_segment(content, node) or ""
                chunk_obj = Chunk(path, start, end, code, "py",
                                  metadata={"type": "module-level code"})
                if (chunk_objects and
                        chunk_objects[-1].metadata.get(
                            "type") == "module-level code"):
                    chunk_objects[-1].content += "\n" + chunk_obj.content
                    chunk_objects[-1].last_character_index = end
                else:
                    chunk_objects.append(chunk_obj)

        for chunk in chunk_objects:
            if len(chunk.content) > self.max_chunk_size:
                sub_chunks = self.split_chunk(chunk)
                chunk_objects.remove(chunk)
                chunk_objects.extend(sub_chunks)

        return chunk_objects


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

        chunks = splitter.split_text(content)
        for chunk in chunks:
            start = content.find(f"{chunk.page_content}", search_from)
            end = start + len(chunk.page_content)
            chunk_obj = Chunk(path, start, end,
                              chunk.page_content, "md", chunk.metadata)
            search_from = end
            chunk_objects.append(chunk_obj)

        return chunk_objects


class TextChunker(Chunker):
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
            chunk_obj = Chunk(path, start, end, chunk, "txt")
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
