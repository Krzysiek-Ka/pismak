from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaEmbedder:
    """
    Lokalny embedding przez natywne API Ollamy /api/embed.
    Domyślnie: bge-m3 (1024 wymiary).
    """

    def __init__(
        self,
        *,
        model: str = "bge-m3",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300.0,
        batch_size: int = 16,
    ) -> None:
        """Initialize the Ollama embedding client configuration."""
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings for one batch of texts from Ollama."""
        payload = json.dumps(
            {
                "model": self.model,
                "input": texts,
                "truncate": True,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama embeddings HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise RuntimeError(f"Nie udało się pobrać embeddingów z Ollamy: {exc}") from exc

        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError("Ollama zwróciła nieoczekiwany format embeddings.")
        return embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts in batches and preserve their input order."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            all_embeddings.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return all_embeddings
