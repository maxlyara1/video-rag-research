from __future__ import annotations

from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import CandidateWindow
from src.runtime import cleanup_torch_memory, detect_torch_device, is_mps_device, resolve_torch_dtype


class Reranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        torch_dtype: str | None = "auto",
        max_length: int = 2048,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = detect_torch_device(device)
        self.dtype = resolve_torch_dtype(torch_dtype, self.device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.cleanup_interval = 8 if is_mps_device(self.device) else 0

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct '
            'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format_pair(self, query: str, document: str) -> str:
        document = document[:4000]
        return (
            "<Instruct>: Given the multilingual user query, retrieve the most relevant fused video context.\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _prepare_inputs(self, pairs: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        prompts = [self._format_pair(query, document) for query, document in pairs]
        encoded = self.tokenizer(
            prompts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for index, input_ids in enumerate(encoded["input_ids"]):
            encoded["input_ids"][index] = self.prefix_tokens + input_ids + self.suffix_tokens
        return self.tokenizer.pad(encoded, padding=True, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def _score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        inputs = self._prepare_inputs(pairs)
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        true_logits = logits[:, self.token_true_id]
        false_logits = logits[:, self.token_false_id]
        scores = torch.stack([false_logits, true_logits], dim=1)
        scores = torch.nn.functional.log_softmax(scores, dim=1)[:, 1].exp()
        values = scores.float().cpu().tolist()
        del inputs, outputs, logits, true_logits, false_logits, scores
        return values

    def rerank(self, query: str, candidates: list[CandidateWindow], top_k: int) -> list[CandidateWindow]:
        if not candidates:
            return []
        pairs = [(query, candidate.combined_text()) for candidate in candidates]
        scores: list[float] = []
        batch_index = 0
        for start in range(0, len(pairs), self.batch_size):
            batch_index += 1
            scores.extend(self._score_batch(pairs[start : start + self.batch_size]))
            if self.cleanup_interval and batch_index % self.cleanup_interval == 0:
                cleanup_torch_memory(self.device)
        cleanup_torch_memory(self.device)

        rescored = [
            replace(candidate, score=float(score))
            for candidate, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        cleanup_torch_memory(self.device)
