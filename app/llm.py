from __future__ import annotations

import json
import urllib.error
import urllib.request

from config.settings import settings

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"


class LLM:
    """Local Ollama-backed LLM wrapper using the REST API."""

    def __init__(self) -> None:
        """Initialize with the Ollama model specified in settings."""
        self.model = settings.OLLAMA_MODEL

    def generate_answer(self, prompt: str) -> str:
        """Generate an answer by calling the Ollama REST API."""
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            OLLAMA_API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama server at {OLLAMA_API_URL}. "
                "Make sure 'ollama serve' is running."
            ) from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Unexpected response from Ollama: {body[:200]}") from exc

        answer = data.get("response", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")

        return answer
