from typing import List, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm  # type: ignore[import-untyped]

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


def _overlap_size(max_chunk_size: int) -> int:
    """A modest overlap so a semantic unit straddling a boundary is
    present in at least two chunks."""
    return max(1, max_chunk_size // 10)


def _split_text_exact(text: str,
                      max_chunk_size: int,
                      overlap: int) -> List[Tuple[int, int]]:
    """Split `text` into (start, end) slices that are exact byte-for-byte
    matches of the input, never exceed `max_chunk_size`, prefer breaking
    at newline boundaries, and overlap consecutive windows by `overlap`
    characters. This is the sliding-window strategy used by every
    chunker, regardless of file type."""
    n = len(text)
    if n <= max_chunk_size:
        return [(0, n)]
    if max_chunk_size <= 0:
        return [(0, n)]

    slices: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + max_chunk_size, n)
        if end < n:
            nl = text.rfind("\n", start + 1, end)
            if nl != -1 and nl - start > max_chunk_size // 2:
                end = nl + 1
        slices.append((start, end))
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0

    return slices


class Chunker(ABC):
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = _overlap_size(max_chunk_size)

    @abstractmethod
    def chunk(self, path: str) -> List[Chunk]:
        raise NotImplementedError

    def _sliding_window_chunk(self, path: str, file_type: str) -> List[Chunk]:
        """Primary chunking strategy for every file type: read the file
        and slide a fixed-size, overlapping window across its raw
        content. Each (start, end) slice is an exact substring of the
        source, never exceeds `max_chunk_size`, and overlaps the next
        slice by `overlap_size` characters."""
        with open(path, "r") as f:
            content = f.read()

        slices = _split_text_exact(content, self.max_chunk_size,
                                   self.overlap_size)
        out: List[Chunk] = []
        for start, end in slices:
            out.append(Chunk(**get_keywords_args(
                path, start, end, content[start:end], file_type)))
        return out


class PythonChunker(Chunker):
    __slots__ = ("max_chunk_size", "overlap_size")

    def chunk(self, path: str) -> List[Chunk]:
        return self._sliding_window_chunk(path, "py")


class MarkdownChunker(Chunker):
    __slots__ = ("max_chunk_size", "overlap_size")

    def chunk(self, path: str) -> List[Chunk]:
        return self._sliding_window_chunk(path, "md")


class TextChunker(Chunker):
    __slots__ = ("max_chunk_size", "overlap_size")

    def chunk(self, path: str) -> List[Chunk]:
        return self._sliding_window_chunk(path, "txt")


class ChunksFactory:
    def __init__(self,
                 data_dir: str = "data/raw",
                 max_chunk_size: int = 2000):
        self.data_dir = data_dir
        self.file_manager = FileManager(data_dir)
        self.max_chunk_size = max_chunk_size
        self.py_chunker = PythonChunker(max_chunk_size=max_chunk_size)
        self.md_chunker = MarkdownChunker(max_chunk_size=max_chunk_size)
        self.txt_chunker = TextChunker(max_chunk_size=max_chunk_size)

    def get_files(self) -> Tuple[List[str], List[str], List[str]]:
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

    def _chunk_files(self, files: List[str],
                     chunker: Chunker) -> List[Chunk]:
        chunks: List[Chunk] = []
        for file in tqdm(files, desc="Chunking", unit="file"):
            chunks.extend(chunker.chunk(file))
        return chunks

    def get_chunks(self) -> List[Chunk]:
        py_files, md_files, txt_files = self.get_files()
        chunks: List[Chunk] = []
        chunks.extend(self._chunk_files(py_files, self.py_chunker))
        chunks.extend(self._chunk_files(md_files, self.md_chunker))
        chunks.extend(self._chunk_files(txt_files, self.txt_chunker))
        chunks.sort(key=lambda x: x.file_type)
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
