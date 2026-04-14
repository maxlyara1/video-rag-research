from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModalityRecord:
    video_file: str
    modality: str
    start: float
    end: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        start_ms = int(round(self.start * 1000))
        end_ms = int(round(self.end * 1000))
        parts = [self.video_name, self.modality]
        det_type = self.metadata.get("det_type")
        if det_type:
            safe_type = "".join(ch if str(ch).isalnum() or ch in {"-", "_"} else "_" for ch in str(det_type))
            parts.append(safe_type)
        parts.extend([f"{start_ms:010d}", f"{end_ms:010d}"])
        return ":".join(parts)

    @property
    def video_name(self) -> str:
        name = self.video_file.rsplit("/", 1)[-1]
        return name.rsplit(".", 1)[0] if "." in name else name


@dataclass
class QueryDecomposition:
    original_query: str
    asr_query: str | None
    det_queries: list[str]
    det_mode: str


@dataclass
class SearchHit:
    video_file: str
    modality: str
    start: float
    end: float
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateWindow:
    video_file: str
    start: float
    end: float
    score: float
    hits: list[SearchHit] = field(default_factory=list)

    def combined_text(self) -> str:
        grouped: dict[str, list[str]] = {"asr": [], "ocr": [], "det": []}
        for hit in self.hits:
            if hit.text and hit.text not in grouped.setdefault(hit.modality, []):
                grouped[hit.modality].append(hit.text)

        parts: list[str] = []
        for modality in ("asr", "ocr", "det"):
            texts = grouped.get(modality) or []
            if texts:
                parts.append(f"[{modality.upper()}]\n" + "\n".join(texts))
        return "\n\n".join(parts)
