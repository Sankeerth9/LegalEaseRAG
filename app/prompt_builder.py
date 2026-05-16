from __future__ import annotations


MAX_CONTEXT_CHARS = 1200
MAX_CHUNK_CHARS = 250


def _clean_text(text: str) -> str:
    """Normalize whitespace for compact prompt formatting."""
    return " ".join(text.split()).strip()


def build_prompt(question: str, contexts: list[dict]) -> str:
    """Build a compact legal QA prompt from the question and retrieved contexts."""
    cleaned_question = _clean_text(question)
    context_blocks: list[str] = []
    current_length = 0

    for index, item in enumerate(contexts, start=1):
        raw_text = str(item.get("text", ""))
        chunk = _clean_text(raw_text)[:MAX_CHUNK_CHARS]

        if not chunk:
            continue

        block = f"Source {index}:\n{chunk}"
        additional_length = len(block) + (2 if context_blocks else 0)

        if current_length + additional_length > MAX_CONTEXT_CHARS:
            break

        context_blocks.append(block)
        current_length += additional_length

    context = "\n\n".join(context_blocks)

    return (
        "You are a legal AI assistant.\n\n"
        "Use the provided context if relevant, but do NOT just repeat it.\n\n"
        "If the question requires explanation, reasoning, or advice, use your general legal knowledge.\n\n"
        "Provide a clear and helpful answer in 3-5 sentences.\n\n"
        "If the question is scenario-based, give practical steps or guidance.\n\n"
        "Avoid copying sentences directly from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{cleaned_question}"
    )
