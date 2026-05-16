"""Tests for the Embedder module."""
from __future__ import annotations

import numpy as np
import pytest

from app.embeddings import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


class TestEmbedder:
    def test_embed_documents_shape(self, embedder: Embedder) -> None:
        texts = ["Security deposit rules.", "Notice period is 2 months."]
        result = embedder.embed_documents(texts)
        assert result.shape == (2, 384), "Expected 384-dim all-MiniLM-L6-v2 embeddings"

    def test_embed_documents_normalized(self, embedder: Embedder) -> None:
        texts = ["Rent is due on the 1st of each month."]
        result = embedder.embed_documents(texts)
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings should be L2-normalized"

    def test_embed_documents_empty(self, embedder: Embedder) -> None:
        result = embedder.embed_documents([])
        assert result.size == 0

    def test_embed_query_shape(self, embedder: Embedder) -> None:
        vec = embedder.embed_query("What is the notice period?")
        assert vec.ndim == 1
        assert vec.shape[0] == 384

    def test_embed_query_normalized(self, embedder: Embedder) -> None:
        vec = embedder.embed_query("How much is the security deposit?")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, "Query embedding should be L2-normalized"

    def test_embed_query_empty_string(self, embedder: Embedder) -> None:
        vec = embedder.embed_query("")
        assert np.all(vec == 0.0), "Empty query should return zero vector"

    def test_embed_query_whitespace(self, embedder: Embedder) -> None:
        vec = embedder.embed_query("   ")
        assert np.all(vec == 0.0), "Whitespace-only query should return zero vector"

    def test_dtype_is_float32(self, embedder: Embedder) -> None:
        result = embedder.embed_documents(["Test document."])
        assert result.dtype == np.float32
