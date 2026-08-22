from typing import List
from src.chunker import Chunk, Chunker
from dataclasses import dataclass


@dataclass
class MarkdownBlock:
    text: str
    start: int
    end: int
    type_: str


class MarkdownChunker(Chunker):
    def __init__(self) -> None:
        super().__init__()
        self.max_chunk_size = 2000
        self.in_codeblock = False
        self.in_paragraph = False
        self.blocks: List[MarkdownBlock] = []
        self.index = 0

    def add(self, text: str, start: int,
            end: int, type_: str
            ) -> None:
        self.blocks.append(MarkdownBlock(text, start, end, type_))

    def checker(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                end = len(line)
                if self.in_codeblock:
                    last_block = self.blocks[-1]
                    last_block.end += end
                    self.index += end
                elif line.startswith("```"):
                    self.in_codeblock = True
                    self.add(text=line, start=self.index,
                             end=end, type_="codeblock")
                    self.index += end
                elif line.startwith("#"):
                    self.add(text=line, start=self.index,
                             end=end, type_="header")
                    self.index += end
                elif line == "":
                    self.in_paragraph = False
                elif self.in_paragraph:
                    last_block = self.blocks[-1]
                    last_block.end += end
                    self.index += end
                else:
                    self.add(text=line, start=self.index,
                             end=end, type_="paragraph")

    def chunk(self, content: str) -> List[Chunk]:
        return []


if __name__ == "__main__":
    cunk = MarkdownChunker()
    cunk.checker("test.md")
    for b in cunk.blocks:
        print(b)
