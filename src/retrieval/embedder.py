from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.runtime import cleanup_torch_memory, detect_torch_device, resolve_torch_dtype

logger = logging.getLogger(__name__)

_E5_PREFIXES = ("intfloat/e5-", "intfloat/multilingual-e5-")
_QWEN_PREFIXES = ("Qwen/Qwen3-Embedding",)


def _detect_backend(model_name: str) -> str:
    """Detect pooling strategy from model name: 'e5' (mean pool) or 'qwen3' (last-token)."""
    lower = model_name.lower()
    if any(lower.startswith(p.lower()) for p in _E5_PREFIXES):
        return "e5"
    if any(lower.startswith(p.lower()) for p in _QWEN_PREFIXES):
        return "qwen3"
    return "qwen3"


class Embedder:
    """Compute dense embeddings. Auto-detects backend from model name.

    Supported backends:
      - 'qwen3': last-token pooling, Instruct/Query format (Qwen3-Embedding)
      - 'e5': mean pooling, passage:/query: prefix (multilingual-e5-*)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str | None = "auto",
        torch_dtype: str | None = None,
        max_length: int = 2048,
        query_instruction: str | None = None,
        output_dim: int | None = None,
        backend: str | None = None,
    ):
        self.model_name = model_name
        self.device = detect_torch_device(device)
        self.dtype = resolve_torch_dtype(torch_dtype, self.device)
        self.max_length = max_length
        self.query_instruction = (
            query_instruction
            or "Given a multilingual user query, retrieve relevant video transcript passages that answer the query."
        )
        self.output_dim = output_dim
        self.backend = backend or _detect_backend(model_name)

        logger.info("Embedder: загрузка '%s' (backend=%s) на %s...", model_name, self.backend, self.device)

        padding_side = "left" if self.backend == "qwen3" else "right"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side=padding_side,
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self._model.eval()

        self.dim = output_dim or int(getattr(self._model.config, "hidden_size"))
        logger.info("Embedder: модель готова (dim=%d)", self.dim)

    @staticmethod
    def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden_states[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    @staticmethod
    def _mean_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = (last_hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def _format_query(self, query: str) -> str:
        if self.backend == "e5":
            return f"query: {query}"
        return f"Instruct: {self.query_instruction}\nQuery:{query}"

    def _format_passage(self, text: str) -> str:
        if self.backend == "e5":
            return f"passage: {text}"
        return text

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self._model(**inputs)
            if self.backend == "e5":
                embeddings = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            else:
                embeddings = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            if self.output_dim:
                embeddings = embeddings[:, : self.output_dim]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        vectors = embeddings.float().cpu().numpy()

        del inputs, outputs, embeddings
        cleanup_torch_memory(self.device)
        return vectors

    def _batched(self, items: Iterable[str], batch_size: int) -> Iterable[list[str]]:
        batch: list[str] = []
        for item in items:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed document passages."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        formatted = [self._format_passage(t) for t in texts]
        total_batches = (len(formatted) + batch_size - 1) // batch_size
        vectors = [
            self._encode_batch(batch)
            for batch in tqdm(
                self._batched(formatted, batch_size),
                total=total_batches,
                desc="Embedding chunks",
            )
        ]
        return np.vstack(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query."""
        return self._encode_batch([self._format_query(query)])[0]

    def embed_queries(self, queries: list[str], batch_size: int = 64) -> np.ndarray:
        """Batch-embed multiple search queries on GPU."""
        if not queries:
            return np.empty((0, self.dim), dtype=np.float32)

        formatted = [self._format_query(q) for q in queries]
        total_batches = (len(formatted) + batch_size - 1) // batch_size
        vectors = [
            self._encode_batch(batch)
            for batch in tqdm(
                self._batched(formatted, batch_size),
                total=total_batches,
                desc="Embedding queries",
            )
        ]
        return np.vstack(vectors)

    def close(self) -> None:
        """Explicitly release model resources after retrieval is complete."""
        if hasattr(self, "_model"):
            del self._model
        cleanup_torch_memory(self.device)
