from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import ast
import json
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
            args = get_keywords_args(chunk.file_path, start, end, c, "txt")
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
    def __init__(self, max_chunk_size: int = 2000):
        self.py_chunker = PythonChunker(max_chunk_size=max_chunk_size)
        self.md_chunker = MarkdownChunker(max_chunk_size=max_chunk_size)
        self.txt_chunker = TextChunker(max_chunk_size=max_chunk_size)

    def get_files(self, data_dir: str
                  ) -> Tuple[List[str], List[str], List[str]]:
        files = FileManager(data_dir).get_modified_files()
        py_files = []
        md_files = []
        txt_files = []
        for path in files:
            file_path = Path(path)
            if (file_path.is_file() and
                    file_path.suffix.lower() not in SUPPORTED_EXTENSIONS):
                continue
            if file_path.is_file() and file_path.suffix.lower() == ".py":
                py_files.append(str(file_path))
            elif file_path.is_file() and file_path.suffix.lower() == ".md":
                md_files.append(str(file_path))
            elif file_path.is_file() and file_path.suffix.lower() == ".txt":
                txt_files.append(str(file_path))

        return py_files, md_files, txt_files

    def get_chunks(self, data_dir: str) -> List[Chunk]:
        processed_data = Path("data/processed/processed_chunks.json")
        if processed_data.is_file():
            with open(processed_data, "r") as f:
                data = json.load(f)
            chunks = [Chunk(**item) for item in data]
            print(f"Loaded {len(chunks)} chunks from"
                  " data/processed/processed_chunks.json")
            return chunks

        py_files, md_files, txt_files = self.get_files(data_dir)
        py_chunks = []
        md_chunks = []
        txt_chunks = []

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

        output_data: List[Dict] = []
        progress = tqdm(
            desc="Tokenizing", unit="chunk", total=len(chunks))
        for chunk in chunks:
            my_dict = chunk.model_dump()
            output_data.append(my_dict)
            progress.update(1)
        progress.close()

        with open("data/processed/processed_chunks.json", "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Ingestion complete! Indexed {len(chunks)}"
              " chunks under data/processed/")
        return chunks
