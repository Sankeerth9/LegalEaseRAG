from __future__ import annotations

import json
import logging
import re

from app.embeddings import Embedder
from app.llm import LLM
from app.prompt_builder import build_prompt
from app.retriever import Retriever


logger = logging.getLogger(__name__)


class RAGEngine:
    """Coordinate retrieval, prompt construction, and answer generation."""

    def __init__(self) -> None:
        """Initialize shared RAG components."""
        self.embedder = Embedder()
        self.retriever = Retriever(embedder=self.embedder)
        self.llm = LLM()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace for downstream processing."""
        return " ".join(text.split()).strip()

    @staticmethod
    def _fallback_answer(question: str, retrieved_chunks: list[dict]) -> str:
        """Build a short extractive fallback answer from retrieved context."""
        query_terms = {
            token
            for token in re.findall(r"[a-zA-Z]+", question.lower())
            if len(token) > 2
        }

        for chunk in retrieved_chunks:
            text = chunk.get("text", "")
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sentence in sentences:
                lowered = sentence.lower()
                if any(term in lowered for term in query_terms):
                    return sentence.strip()

        return retrieved_chunks[0].get("text", "Not found")

    # ------------------------------------------------------------------
    # Lease Summarization — LLM-powered with regex fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _regex_summarise(text: str) -> dict:
        """Regex-based lease field extraction used as fallback."""
        rent_match = re.search(
            r"((?:monthly )?rent\s*(?:of|is)?\s*[₹Rs\.\s]*[\d,]+|[₹Rs\.\s]*[\d,]+\s*(?:per month|monthly rent))",
            text,
            re.IGNORECASE,
        )
        deposit_match = re.search(
            r"((?:interest-free\s+|refundable\s+|security\s+)*deposit\s*(?:of|is|amounting to)?\s*[₹Rs\.\s]*[\d,]+)",
            text,
            re.IGNORECASE,
        )
        notice_match = re.search(
            r"((?:\d+|one|two|three|four|five|six)\s*(?:\(\d+\))?\s*[- ]?(?:month|months|day|days)\s+notice|notice period of\s+(?:\d+|one|two|three|four|five|six)\s*(?:\(\d+\))?\s*[- ]?(?:month|months|day|days))",
            text,
            re.IGNORECASE,
        )

        maintenance_sentence = "Not found"
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            lowered = sentence.lower()
            if "maintenance" in lowered or "repair" in lowered or "damage" in lowered:
                maintenance_sentence = sentence.strip()
                break

        return {
            "rent": rent_match.group(0).strip().rstrip(".") if rent_match else "Not found",
            "deposit": deposit_match.group(0).strip().rstrip(".") if deposit_match else "Not found",
            "notice_period": notice_match.group(0).strip().rstrip(".") if notice_match else "Not found",
            "maintenance": maintenance_sentence,
        }

    def summarise_document(self, text: str) -> dict:
        """Extract key lease fields using the LLM, with regex as fallback."""
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return {
                "rent": "Not found",
                "deposit": "Not found",
                "notice_period": "Not found",
                "maintenance": "Not found",
            }

        prompt = (
            "You are a legal document assistant. Extract the following fields from the lease text below.\n"
            "Return ONLY a valid JSON object with these exact keys: rent, deposit, notice_period, maintenance.\n"
            "For each field, provide a short extracted value (e.g. '₹15,000 per month') or 'Not found' if absent.\n"
            "Do not include any explanation outside the JSON.\n\n"
            f"Lease Text:\n{cleaned_text[:2000]}\n\n"
            "JSON:"
        )

        try:
            raw = self.llm.generate_answer(prompt)
            # Remove markdown formatting if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            # Extract the first JSON object from the response
            json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return {
                    "rent": str(parsed.get("rent", "Not found")),
                    "deposit": str(parsed.get("deposit", "Not found")),
                    "notice_period": str(parsed.get("notice_period", "Not found")),
                    "maintenance": str(parsed.get("maintenance", "Not found")),
                }
            else:
                logger.warning("No JSON object found in LLM response.")
        except Exception as exc:
            logger.warning("LLM summarisation failed, using regex fallback: %s", exc)

        return self._regex_summarise(cleaned_text)

    # ------------------------------------------------------------------
    # Clause Risk Analysis — LLM-powered with rule-based fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_flag(text: str) -> list[dict]:
        """Pattern-based clause flagging used as fallback."""
        clauses = [
            clause.strip()
            for clause in re.split(r"(?<=[.!?])\s+", text)
            if clause.strip()
        ]

        findings: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for clause in clauses:
            lowered = clause.lower()

            deposit_match = re.search(r"(\d+)\s*[- ]?month", lowered)
            if "deposit" in lowered and deposit_match:
                months = int(deposit_match.group(1))
                if months > 3:
                    finding = {
                        "clause": clause,
                        "risk": "high",
                        "reason": "Security deposit exceeds 3 months, which may be unfair or legally excessive.",
                    }
                    key = (finding["clause"], finding["risk"])
                    if key not in seen:
                        findings.append(finding)
                        seen.add(key)

            for trigger in ("penalty", "fine", "immediate eviction", "no refund"):
                if trigger in lowered:
                    finding = {
                        "clause": clause,
                        "risk": "high" if trigger in {"immediate eviction", "no refund"} else "medium",
                        "reason": f"Contains potentially harsh wording: '{trigger}'.",
                    }
                    key = (finding["clause"], finding["risk"])
                    if key not in seen:
                        findings.append(finding)
                        seen.add(key)

        if "notice" not in text.lower():
            findings.append(
                {
                    "clause": "No clear notice period clause detected in the provided text.",
                    "risk": "medium",
                    "reason": "Lease text does not mention a notice period.",
                }
            )

        if not findings:
            findings.append(
                {
                    "clause": "No major risky clause detected in the provided text.",
                    "risk": "low",
                    "reason": "No deposit, notice, penalty, eviction, or refund red flags found.",
                }
            )

        return findings

    def flag_clauses(self, text: str) -> list[dict]:
        """Flag risky lease clauses using the LLM, with rule-based fallback."""
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return []

        prompt = (
            "You are a legal AI assistant specializing in Indian rental law.\n"
            "Analyze the lease text below and identify risky clauses.\n"
            "Return ONLY a valid JSON array. Each element must have exactly these keys:\n"
            "  - clause: the exact clause text (string)\n"
            "  - risk: one of 'high', 'medium', or 'low' (string)\n"
            "  - reason: a brief explanation of the risk (string)\n\n"
            "Rules for risk level:\n"
            "  - high: immediate eviction, no refund, deposit > 3 months, loss of all rights\n"
            "  - medium: penalties, fines, vague obligations, missing notice period\n"
            "  - low: minor issues or general concerns\n\n"
            "If no risky clauses are found, return a single low-risk entry explaining that.\n"
            "Do not include any text outside the JSON array.\n\n"
            f"Lease Text:\n{cleaned_text[:2500]}\n\n"
            "JSON Array:"
        )

        try:
            raw = self.llm.generate_answer(prompt)
            # Extract the first JSON array from the response
            json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list) and parsed:
                    findings = []
                    for item in parsed:
                        if isinstance(item, dict):
                            findings.append(
                                {
                                    "clause": str(item.get("clause", "Clause not available.")),
                                    "risk": str(item.get("risk", "low")).lower(),
                                    "reason": str(item.get("reason", "No reason provided.")),
                                }
                            )
                    if findings:
                        return findings
        except Exception as exc:
            logger.warning("LLM clause flagging failed, using rule-based fallback: %s", exc)

        return self._rule_based_flag(cleaned_text)

    # ------------------------------------------------------------------
    # Legal Q&A — Full RAG pipeline
    # ------------------------------------------------------------------

    def answer_question(self, question: str) -> dict:
        """Run the full RAG pipeline for a single question."""
        question = question.strip()[:300]
        retrieved_chunks = self.retriever.retrieve(question)

        if not retrieved_chunks:
            return {
                "answer": "No relevant legal information found.",
                "sources": [],
                "retrieved_chunks": [],
            }

        prompt = build_prompt(question, retrieved_chunks)

        try:
            answer = self.llm.generate_answer(prompt)
        except Exception as exc:
            logger.error("LLM error: %s", str(exc))
            sources = sorted({
                chunk["source"]
                for chunk in retrieved_chunks
                if chunk.get("source")
            })
            return {
                "answer": self._fallback_answer(question, retrieved_chunks),
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
            }

        if not answer or not answer.strip():
            sources = sorted({
                chunk["source"]
                for chunk in retrieved_chunks
                if chunk.get("source")
            })
            return {
                "answer": self._fallback_answer(question, retrieved_chunks),
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
            }

        context_length = sum(len(chunk["text"]) for chunk in retrieved_chunks)
        estimated_tokens = len(prompt) // 4

        logger.info("Chunks retrieved: %s", len(retrieved_chunks))
        logger.info("Context length (characters): %s", context_length)
        logger.info("Prompt length (characters): %s", len(prompt))
        logger.info("Estimated tokens: %s", estimated_tokens)

        sources = sorted({
            chunk["source"]
            for chunk in retrieved_chunks
            if chunk.get("source")
        })

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
        }
