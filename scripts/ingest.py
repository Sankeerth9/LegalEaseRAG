from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import faiss
import nltk
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.embeddings import Embedder
from config.settings import settings


def clean_text(text: str) -> str:
    """Collapse repeated whitespace and remove extra newlines."""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, downloading punkt on first use if needed."""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                continue
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            return [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", text)
                if sentence.strip()
            ]


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """Chunk text while preserving sentence boundaries where possible."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    sentences = split_sentences(cleaned)
    if not sentences:
        return [cleaned[:chunk_size]]

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = clean_text(sentence)
        if not sentence:
            continue

        candidate = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)

        if overlap > 0 and chunks:
            tail = current_chunk[-overlap:].strip()
            current_chunk = f"{tail} {sentence}".strip() if tail else sentence
            if len(current_chunk) > chunk_size:
                current_chunk = sentence
        else:
            current_chunk = sentence

        if len(current_chunk) > chunk_size:
            chunks.append(current_chunk[:chunk_size].strip())
            current_chunk = current_chunk[chunk_size - overlap :].strip() if overlap < chunk_size else ""

    if current_chunk:
        chunks.append(current_chunk)

    return [clean_text(chunk) for chunk in chunks if clean_text(chunk)]


def load_documents(data_dir: str | Path) -> list[dict[str, str]]:
    """Load plaintext legal documents from the data directory."""
    base_path = Path(data_dir)
    documents: list[dict[str, str]] = []

    for path in sorted(base_path.glob("*.txt")):
        text = clean_text(path.read_text(encoding="utf-8"))
        if text:
            documents.append({"text": text, "source": path.name})

    return documents


def build_chunks(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build chunk records in the expected retrieval format."""
    chunk_records: list[dict[str, str]] = []

    for document in documents:
        source = document["source"]
        for chunk in chunk_text(
            document["text"],
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        ):
            chunk_records.append(
                {
                    "text": clean_text(chunk),
                    "source": source,
                }
            )

    return chunk_records


def save_index(vectors: np.ndarray, chunks: list[dict[str, str]]) -> None:
    """Persist FAISS vectors and chunk metadata to disk."""
    data_dir = Path(settings.FAISS_INDEX_PATH).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    if vectors.size == 0:
        raise ValueError("No vectors generated. Add text files to the data directory first.")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    faiss.write_index(index, settings.FAISS_INDEX_PATH)

    with Path(settings.METADATA_PATH).open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)


def main() -> None:
    """Build the retrieval corpus and save the FAISS index."""
    documents = load_documents(settings.RAW_DATA_DIR)
    chunks = build_chunks(documents)

    if not chunks:
        raise ValueError("No .txt files with content found in the data directory.")

    embedder = Embedder(settings.EMBEDDING_MODEL)
    vectors = embedder.embed_documents([chunk["text"] for chunk in chunks])
    save_index(vectors, chunks)

    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents.")


if __name__ == "__main__":
    main()
