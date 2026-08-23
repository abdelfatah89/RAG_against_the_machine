from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
import ast

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)

def get_offsets(source: str, node: ast.AST) -> tuple[int, int]:
    lines = source.splitlines(keepends=True)

    start = sum(len(line) for line in lines[:node.lineno - 1])
    start += node.col_offset

    end = sum(len(line) for line in lines[:node.end_lineno - 1])
    end += node.end_col_offset

    return start, end

with open("test.md", "r") as f:
    python_code = f.read()

print(python_code.find("class Chunk(MinimalSource):"))

tree = ast.parse(python_code)
for node in tree.body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
        start, end = get_offsets(python_code, node)

        print(node.name)
        print("start:", start)
        print("end:", end)
        # print("source:", repr(python_code[start:end]))

chunks = splitter.split_text(python_code)

# print(f"Number of chunks: {len(chunks)}")
# print("Chunks:")
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}:\n{chunk}\n{'-' * 40}")
