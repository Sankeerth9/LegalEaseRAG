from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.embeddings import Embedder
from config.settings import settings


class Retriever:
    """Custom FAISS retriever for high-precision legal context lookup."""

    def __init__(
        self,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.metadata_path = Path(metadata_path or settings.METADATA_PATH)
        self.embedder = embedder or Embedder(settings.EMBEDDING_MODEL)
        self.index = self._load_index()
        self.metadata = self._load_metadata()

    def _load_index(self) -> faiss.Index | None:
        """Load the FAISS index from disk if available."""
        if not self.index_path.exists():
            return None
        return faiss.read_index(str(self.index_path))

    def _load_metadata(self) -> list[dict[str, Any]]:
        """Load chunk metadata from disk if available."""
        if not self.metadata_path.exists():
            return []

        with self.metadata_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    @staticmethod
    def _deduplicate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate chunks using the first 100 characters as a key."""
        seen: set[str] = set()
        unique_results: list[dict[str, Any]] = []

        for result in results:
            text = result.get("text", "").strip()
            dedupe_key = text[:100].lower()

            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            unique_results.append(result)

        return unique_results

    def _similarity_threshold(self) -> float:
        """Use a looser threshold for small indexes and a stricter one for larger corpora."""
        if self.index is None:
            return 0.5
        return 0.6 if self.index.ntotal > 100 else 0.5

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Extract lowercase keyword tokens for simple lexical fallback."""
        return {
            token
            for token in re.findall(r"[a-zA-Z]+", text.lower())
            if len(token) > 2
        }

    def _lexical_fallback(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return a few keyword-overlap matches when vector filtering is too strict."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored_results: list[dict[str, Any]] = []
        for chunk in self.metadata:
            text = str(chunk.get("text", "")).strip()
            source = str(chunk.get("source", "")).strip()
            if not text:
                continue

            text_tokens = self._tokenize(text)
            overlap = query_tokens & text_tokens
            if not overlap:
                continue

            score = len(overlap) / max(len(query_tokens), 1)
            scored_results.append(
                {
                    "text": text,
                    "source": source,
                    "score": round(float(score), 4),
                }
            )

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return self._deduplicate(scored_results)[:top_k]

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Retrieve the highest-quality chunks for a query.

        Returns results in the format:
        [{"text": "...", "source": "...", "score": 0.87}]
        """
        if not query.strip() or self.index is None or not self.metadata:
            return []

        query_vector = self.embedder.embed_query(query)
        if query_vector.size == 0:
            return []

        search_k = max(top_k * 3, top_k)
        distances, indices = self.index.search(
            np.asarray([query_vector], dtype=np.float32),
            search_k,
        )
        threshold = self._similarity_threshold()

        results: list[dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            similarity = float(score)
            if similarity <= threshold:
                continue

            chunk = self.metadata[idx]
            text = str(chunk.get("text", "")).strip()
            source = str(chunk.get("source", "")).strip()

            if not text:
                continue

            results.append(
                {
                    "text": text,
                    "source": source,
                    "score": round(similarity, 4),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        unique_results = self._deduplicate(results)
        if unique_results:
            return unique_results[:top_k]

        return self._lexical_fallback(query, top_k)


def retrieve(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Convenience function for one-off retrieval calls."""
    retriever = Retriever()
    return retriever.retrieve(query=query, top_k=top_k)
