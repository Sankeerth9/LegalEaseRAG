"""Tests for the RAGEngine module.

LLM calls are mocked so these tests run without Ollama installed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.rag_engine import RAGEngine


@pytest.fixture(scope="module")
def engine() -> RAGEngine:
    return RAGEngine()


# ---------------------------------------------------------------------------
# Lease summarisation
# ---------------------------------------------------------------------------

class TestSummariseDocument:
    def test_empty_text_returns_not_found(self, engine: RAGEngine) -> None:
        result = engine.summarise_document("")
        assert result == {
            "rent": "Not found",
            "deposit": "Not found",
            "notice_period": "Not found",
            "maintenance": "Not found",
        }

    def test_llm_json_is_parsed(self, engine: RAGEngine) -> None:
        mock_response = (
            '{"rent": "₹15,000 per month", "deposit": "₹30,000", '
            '"notice_period": "2 months", "maintenance": "Landlord handles major repairs."}'
        )
        with patch.object(engine.llm, "generate_answer", return_value=mock_response):
            result = engine.summarise_document("Rent is ₹15,000. Deposit is ₹30,000.")
        assert result["rent"] == "₹15,000 per month"
        assert result["deposit"] == "₹30,000"
        assert result["notice_period"] == "2 months"

    def test_falls_back_to_regex_on_llm_error(self, engine: RAGEngine) -> None:
        with patch.object(engine.llm, "generate_answer", side_effect=RuntimeError("Ollama down")):
            result = engine.summarise_document(
                "The tenant shall pay a monthly rent of ₹12,000. "
                "A security deposit of ₹24,000 is required. "
                "A 2-month notice must be given. "
                "The landlord is responsible for maintenance of plumbing."
            )
        assert "12,000" in result["rent"]
        assert "24,000" in result["deposit"]
        assert "2-month notice" in result["notice_period"]
        assert "maintenance" in result["maintenance"].lower()

    def test_falls_back_to_regex_on_bad_json(self, engine: RAGEngine) -> None:
        with patch.object(engine.llm, "generate_answer", return_value="not valid json at all"):
            result = engine.summarise_document(
                "Monthly rent of ₹10,000. Security deposit of ₹20,000."
            )
        # regex fallback should handle this
        assert isinstance(result, dict)
        assert set(result.keys()) == {"rent", "deposit", "notice_period", "maintenance"}


# ---------------------------------------------------------------------------
# Clause flagging
# ---------------------------------------------------------------------------

class TestFlagClauses:
    def test_empty_text_returns_empty_list(self, engine: RAGEngine) -> None:
        result = engine.flag_clauses("")
        assert result == []

    def test_llm_json_array_is_parsed(self, engine: RAGEngine) -> None:
        mock_response = (
            '[{"clause": "Deposit of 6 months rent required.", "risk": "high", '
            '"reason": "Deposit exceeds 3 months."}]'
        )
        with patch.object(engine.llm, "generate_answer", return_value=mock_response):
            result = engine.flag_clauses("Deposit of 6 months rent required.")
        assert len(result) == 1
        assert result[0]["risk"] == "high"
        assert "Deposit" in result[0]["clause"]

    def test_falls_back_to_rules_on_llm_error(self, engine: RAGEngine) -> None:
        with patch.object(engine.llm, "generate_answer", side_effect=RuntimeError("Ollama down")):
            result = engine.flag_clauses(
                "Tenant must pay a deposit of 5 months rent. "
                "Any damage will result in an immediate eviction."
            )
        risks = [r["risk"] for r in result]
        assert "high" in risks

    def test_high_risk_for_excessive_deposit(self, engine: RAGEngine) -> None:
        with patch.object(engine.llm, "generate_answer", side_effect=RuntimeError("skip")):
            result = engine.flag_clauses("A deposit of 6 months is required.")
        assert any(r["risk"] == "high" for r in result)

    def test_medium_risk_for_penalty(self, engine: RAGEngine) -> None:
        with patch.object(engine.llm, "generate_answer", side_effect=RuntimeError("skip")):
            result = engine.flag_clauses("Tenant will pay a fine of ₹5,000 for late payment.")
        assert any(r["risk"] == "medium" for r in result)

    def test_missing_notice_flagged_as_medium(self, engine: RAGEngine) -> None:
        with patch.object(engine.llm, "generate_answer", side_effect=RuntimeError("skip")):
            result = engine.flag_clauses("Rent is ₹10,000 per month.")
        assert any(r["risk"] == "medium" for r in result)


# ---------------------------------------------------------------------------
# Q&A — answer_question
# ---------------------------------------------------------------------------

class TestAnswerQuestion:
    def test_returns_dict_with_expected_keys(self, engine: RAGEngine) -> None:
        with patch.object(engine.llm, "generate_answer", return_value="Mocked answer."):
            result = engine.answer_question("What is the notice period?")
        assert "answer" in result
        assert "sources" in result
        assert "retrieved_chunks" in result

    def test_answer_is_string(self, engine: RAGEngine) -> None:
        with patch.object(engine.llm, "generate_answer", return_value="Mocked answer."):
            result = engine.answer_question("What is the security deposit?")
        assert isinstance(result["answer"], str)

    def test_fallback_on_llm_error(self, engine: RAGEngine) -> None:
        with patch.object(engine.llm, "generate_answer", side_effect=RuntimeError("Ollama down")):
            result = engine.answer_question("What are the rules for subletting?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_empty_question(self, engine: RAGEngine) -> None:
        result = engine.answer_question("")
        assert result["answer"] == "No relevant legal information found."
        assert result["retrieved_chunks"] == []

    def test_sources_are_list(self, engine: RAGEngine) -> None:
        with patch.object(engine.llm, "generate_answer", return_value="Answer."):
            result = engine.answer_question("What happens on illegal eviction?")
        assert isinstance(result["sources"], list)
