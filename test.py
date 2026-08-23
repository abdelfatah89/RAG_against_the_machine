from langchain_text_splitters import MarkdownHeaderTextSplitter

with open("test.md", "r") as f:
    markdown = f.read()

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
{m for m in doc.metadata.values()}
documents = splitter.split_text(markdown)
print(f"Number of documents: {len(documents)}")
for doc in documents:
    print("CONTENT: ___________________________________________________________")
    print(doc.page_content)
    print("METADATA:")
    print(doc.metadata)
    print("-----------------------------------------------------------------")