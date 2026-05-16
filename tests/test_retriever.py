"""Tests for the Retriever module.

These tests load the real FAISS index from disk.
Run `python scripts/ingest.py` first if the index does not exist.
"""
from __future__ import annotations

import pytest

from app.embeddings import Embedder
from app.retriever import Retriever


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    embedder = Embedder()
    return Retriever(embedder=embedder)


class TestRetriever:
    def test_retrieve_returns_list(self, retriever: Retriever) -> None:
        results = retriever.retrieve("What is the notice period?")
        assert isinstance(results, list)

    def test_retrieve_result_structure(self, retriever: Retriever) -> None:
        results = retriever.retrieve("security deposit rules")
        for item in results:
            assert "text" in item
            assert "source" in item
            assert "score" in item

    def test_retrieve_relevant_chunk(self, retriever: Retriever) -> None:
        results = retriever.retrieve("How many months can a landlord charge as deposit?")
        assert len(results) > 0, "Expected at least one result for a relevant query"
        texts = " ".join(r["text"].lower() for r in results)
        assert "deposit" in texts or "month" in texts

    def test_retrieve_top_k_limit(self, retriever: Retriever) -> None:
        results = retriever.retrieve("rent increase rules", top_k=2)
        assert len(results) <= 2

    def test_retrieve_empty_query(self, retriever: Retriever) -> None:
        results = retriever.retrieve("")
        assert results == []

    def test_retrieve_whitespace_query(self, retriever: Retriever) -> None:
        results = retriever.retrieve("   ")
        assert results == []

    def test_retrieve_scores_are_float(self, retriever: Retriever) -> None:
        results = retriever.retrieve("illegal eviction")
        for item in results:
            assert isinstance(item["score"], float)

    def test_retrieve_no_duplicate_texts(self, retriever: Retriever) -> None:
        results = retriever.retrieve("notice period for tenant")
        texts = [r["text"][:100].lower() for r in results]
        assert len(texts) == len(set(texts)), "Retriever should deduplicate results"
