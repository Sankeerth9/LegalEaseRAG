"""Tests for the prompt_builder module."""
from __future__ import annotations

import pytest

from app.prompt_builder import build_prompt


SAMPLE_CONTEXTS = [
    {"text": "Security deposit is typically 2 to 3 months of rent.", "source": "security_deposit_rules.txt"},
    {"text": "Tenant must provide a 2-month notice before vacating.", "source": "sample_lease.txt"},
]


class TestBuildPrompt:
    def test_returns_string(self) -> None:
        result = build_prompt("What is the notice period?", SAMPLE_CONTEXTS)
        assert isinstance(result, str)

    def test_contains_question(self) -> None:
        question = "What is the security deposit limit?"
        result = build_prompt(question, SAMPLE_CONTEXTS)
        assert question in result

    def test_contains_context(self) -> None:
        result = build_prompt("Question?", SAMPLE_CONTEXTS)
        assert "Security deposit" in result

    def test_empty_contexts(self) -> None:
        result = build_prompt("How long is the notice period?", [])
        assert "How long is the notice period?" in result
        assert "Context:" in result

    def test_respects_max_context_length(self) -> None:
        # Build many large contexts to exceed max chars
        large_contexts = [
            {"text": "A" * 300, "source": "doc.txt"}
            for _ in range(20)
        ]
        result = build_prompt("test question", large_contexts)
        # Context section should not exceed 1200 raw chars
        context_start = result.find("Context:\n") + len("Context:\n")
        context_end = result.find("\n\nQuestion:")
        context_body = result[context_start:context_end]
        assert len(context_body) <= 1200 + 50  # small leeway for formatting

    def test_empty_question(self) -> None:
        result = build_prompt("", SAMPLE_CONTEXTS)
        assert isinstance(result, str)
