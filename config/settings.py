from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseModel):
    """Central application settings."""

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    # LLM — local Ollama only
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    # Paths
    FAISS_INDEX_PATH: str = str(DATA_DIR / "faiss_index.bin")
    METADATA_PATH: str = str(DATA_DIR / "chunks.json")
    RAW_DATA_DIR: str = str(DATA_DIR)
    # Chunking
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100
    # Retrieval
    TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.65


settings = Settings()
