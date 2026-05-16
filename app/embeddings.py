from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Generate normalized embeddings for documents and queries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the embedding model once for reuse across calls."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """Apply safe L2 normalization and return float32 vectors."""
        if vectors.size == 0:
            return vectors.astype(np.float32, copy=False)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (vectors / norms).astype(np.float32, copy=False)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a list of document chunks as a 2D float32 numpy array."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return self._l2_normalize(embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string as a normalized float32 vector."""
        if not text or not text.strip():
            dimension = self.model.get_embedding_dimension()
            return np.zeros(dimension, dtype=np.float32)

        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        normalized = self._l2_normalize(embedding)
        return normalized[0]
