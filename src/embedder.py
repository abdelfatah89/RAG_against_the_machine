from typing import List
from tqdm import tqdm  # type: ignore[import-untyped]

from sentence_transformers import SentenceTransformer
import numpy as np

from .chunker import Chunk


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def embed_batch(self, chunks: List[Chunk]) -> List[np.ndarray]:
        progress = tqdm(
            desc="Embedding chunks", unit="chunk", total=len(chunks))
        embeddings: List[np.ndarray] = []
        for chunk in chunks:
            embed = self.embed(chunk.content)
            embeddings.append(embed)
            progress.update(1)

        progress.close()
        return embeddings
