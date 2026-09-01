from typing import List, cast
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.typing import NDArray

from .chunker import Chunk


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> NDArray[np.float32]:
        return cast(NDArray[np.float32], self.model.encode(text))

    def embed_batch(self, chunks: List[Chunk]) -> List[NDArray[np.float32]]:
        progress = tqdm(
            desc="Embedding chunks", unit="chunk", total=len(chunks))
        embeddings: List[NDArray[np.float32]] = []
        for chunk in chunks:
            embed = self.embed(chunk.content)
            embeddings.append(embed)
            progress.update(1)

        progress.close()
        return embeddings
