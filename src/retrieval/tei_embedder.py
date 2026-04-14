from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

_E5_PREFIXES = ("intfloat/e5-", "intfloat/multilingual-e5-")
_QWEN_PREFIXES = ("Qwen/Qwen3-Embedding",)


def _detect_backend(model_name: str) -> str:
    lower = model_name.lower()
    if any(lower.startswith(prefix.lower()) for prefix in _E5_PREFIXES):
        return "e5"
    if any(lower.startswith(prefix.lower()) for prefix in _QWEN_PREFIXES):
        return "qwen3"
    return "qwen3"


class TEIEmbedder:
    """Embedding client for a long-running Hugging Face TEI server."""

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:8080",
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        dim: int = 1024,
        timeout_sec: float = 120.0,
        query_instruction: str | None = None,
        backend: str | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.dim = dim
        self.timeout_sec = timeout_sec
        self.query_instruction = (
            query_instruction
            or "Given a multilingual user query, retrieve relevant video transcript passages that answer the query."
        )
        self.backend = backend or _detect_backend(model_name)
        logger.info("Embedder: TEI model=%s backend=%s", model_name, self.backend)

    def _format_query(self, query: str) -> str:
        if self.backend == "e5":
            return f"query: {query}"
        return f"Instruct: {self.query_instruction}\nQuery:{query}"

    def _format_passage(self, text: str) -> str:
        if self.backend == "e5":
            return f"passage: {text}"
        return text

    @staticmethod
    def _batched(items: Iterable[str], batch_size: int) -> Iterable[list[str]]:
        batch: list[str] = []
        for item in items:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _post_embed(self, texts: list[str]) -> np.ndarray:
        payload = json.dumps({"inputs": texts}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError:
            raise RuntimeError(
                "TEI endpoint is unavailable. Check TEI_ENDPOINT in .env or configs/config.yaml."
            ) from None

        if not raw:
            return np.empty((0, self.dim), dtype=np.float32)
        if isinstance(raw[0], (int, float)):
            raw = [raw]

        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim != 2:
            raise RuntimeError(f"Unexpected TEI embedding response shape: {vectors.shape}")
        if vectors.shape[1] != self.dim:
            raise RuntimeError(
                f"TEI embedding dim mismatch: got {vectors.shape[1]}, expected {self.dim}. "
                "Update indexing.embedding_dim or run TEI with the configured embedding model."
            )
        return vectors

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        formatted = [self._format_passage(text) for text in texts]
        total_batches = (len(formatted) + batch_size - 1) // batch_size
        vectors = [
            self._post_embed(batch)
            for batch in tqdm(
                self._batched(formatted, batch_size),
                total=total_batches,
                desc="Embedding chunks via TEI",
            )
        ]
        return np.vstack(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        return self._post_embed([self._format_query(query)])[0]

    def embed_queries(self, queries: list[str], batch_size: int = 64) -> np.ndarray:
        if not queries:
            return np.empty((0, self.dim), dtype=np.float32)

        formatted = [self._format_query(query) for query in queries]
        total_batches = (len(formatted) + batch_size - 1) // batch_size
        vectors = [
            self._post_embed(batch)
            for batch in tqdm(
                self._batched(formatted, batch_size),
                total=total_batches,
                desc="Embedding queries via TEI",
            )
        ]
        return np.vstack(vectors)

    def close(self) -> None:
        return None
