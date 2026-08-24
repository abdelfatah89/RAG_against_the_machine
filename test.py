from rank_bm25 import BM25Okapi

documents = [
    "Python is a programming language",
    "Java is used for enterprise applications",
    "Python can be used for machine learning",
]

tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

query = "Python machine learning"
scores = bm25.get_scores(query.lower().split())


doc_scores = list(zip(documents, scores.tolist()))
doc_scores.sort(key=lambda x: x[1], reverse=True)
for d in doc_scores:
    print(f"{d[0]} : {d[1]}")

print(documents)
print(scores.tolist())

top_k = bm25.get_top_n(
    query.lower().split(),
    documents,
    n=2,
)

# print(top_k)
