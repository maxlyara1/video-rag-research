from __future__ import annotations

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hf_models import chat_template_kwargs
from src.models import QueryDecomposition
from src.prompts import QUERY_DECOUPLE_SYSTEM_PROMPT, QUERY_DECOUPLE_USER_TEMPLATE
from src.runtime import cleanup_torch_memory, detect_torch_device, resolve_torch_dtype


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class QueryDecoupler:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        torch_dtype: str | None = "auto",
        max_new_tokens: int = 192,
    ) -> None:
        self.model_name = model_name
        self.device = detect_torch_device(device)
        self.dtype = resolve_torch_dtype(torch_dtype, self.device)
        self.max_new_tokens = max_new_tokens
        self.chat_kwargs = chat_template_kwargs(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _coerce_payload(query: str, payload: dict[str, object]) -> QueryDecomposition:
        asr_query = payload.get("asr_query")
        det_queries = payload.get("det_queries") or []
        det_mode = str(payload.get("det_mode") or "relation").lower()
        if det_mode not in {"location", "number", "relation"}:
            det_mode = "relation"
        normalized_det = []
        for item in det_queries:
            text = " ".join(str(item).split()).strip()
            if text and text not in normalized_det:
                normalized_det.append(text)
            if len(normalized_det) == 5:
                break
        normalized_asr = None
        if asr_query is not None:
            clean_asr = " ".join(str(asr_query).split()).strip()
            normalized_asr = clean_asr or None
        return QueryDecomposition(
            original_query=query,
            asr_query=normalized_asr,
            det_queries=normalized_det,
            det_mode=det_mode,
        )

    def decouple(self, query: str) -> QueryDecomposition:
        messages = [
            {"role": "system", "content": QUERY_DECOUPLE_SYSTEM_PROMPT},
            {"role": "user", "content": QUERY_DECOUPLE_USER_TEMPLATE.format(query=query)},
        ]
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **self.chat_kwargs,
            ).to(self.device)
            input_length = inputs.shape[-1]
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            decoded = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
            )
            json_match = _JSON_RE.search(decoded)
            if json_match:
                payload = json.loads(json_match.group(0))
                return self._coerce_payload(query, payload)
        except Exception:
            pass

        return QueryDecomposition(
            original_query=query,
            asr_query=query,
            det_queries=[],
            det_mode="relation",
        )

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        cleanup_torch_memory(self.device)
