from __future__ import annotations

import logging
from pathlib import Path

import whisper

from src.models import ModalityRecord
from src.runtime import cleanup_torch_memory, detect_torch_device

logger = logging.getLogger(__name__)


class WhisperASRExtractor:
    def __init__(
        self,
        model_name: str = "small",
        *,
        device: str = "auto",
        language: str | None = None,
        no_speech_threshold: float = 0.6,
    ) -> None:
        self.model_name = model_name
        self.device = detect_torch_device(device)
        self.language = language
        self.no_speech_threshold = no_speech_threshold
        logger.info("ASR: загрузка Whisper '%s' на %s...", model_name, self.device)
        try:
            self.model = whisper.load_model(model_name, device=self.device)
        except Exception:
            if self.device != "cpu":
                self.device = "cpu"
                self.model = whisper.load_model(model_name, device=self.device)
            else:
                raise
        logger.info("ASR: модель готова")

    def extract(self, video_path: str | Path) -> list[ModalityRecord]:
        raw = self.model.transcribe(
            str(video_path),
            verbose=False,
            language=self.language,
            word_timestamps=False,
            condition_on_previous_text=True,
            no_speech_threshold=self.no_speech_threshold,
        )
        records: list[ModalityRecord] = []
        for segment in raw.get("segments", []):
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            records.append(
                ModalityRecord(
                    video_file=str(video_path),
                    modality="asr",
                    start=round(float(segment["start"]), 3),
                    end=round(float(segment["end"]), 3),
                    text=text,
                    metadata={
                        "language": raw.get("language"),
                        "no_speech_prob": round(float(segment.get("no_speech_prob", 0.0)), 4),
                    },
                )
            )
        return records

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        cleanup_torch_memory(self.device)

