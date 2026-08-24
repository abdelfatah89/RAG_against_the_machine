# Chroma Vector Database Example
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


def main() -> None:
    # Setup
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.Client()
    collection = client.create_collection("policies")
    # Add 3 policy documents (Chroma handles embeddings automatically)
    policies = [
        "Dogs are allowed in the office on Fridays",
        "Pets can come to work on Furry Fridays",
        "Remote work policy allows 3 days from home"
    ]
    for i, policy in enumerate(policies):
        collection.add(
                documents=[policy],
                ids=[f"policy_{i}"]
                )
    # Query the system
    query = "Can I bring my dog to work?"
    policies.append(query)
    embeddings = model.encode(policies)
    sim_q1 = np.dot(embeddings[0], embeddings[3])
    sim_q2 = np.dot(embeddings[1], embeddings[3])
    results = collection.query(query_texts=[query], n_results=2)
    # Show results
    print(sim_q1)
    print(sim_q2)
    print(results["metadatas"][0])


if __name__ == "__main__":
    main()
