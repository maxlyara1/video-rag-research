from __future__ import annotations

import logging
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch

from src.models import ModalityRecord
from src.runtime import detect_torch_device, is_cuda_device
from src.utils.video_frames import RobustVideoFrameSampler
from src.utils.video_metadata import probe_video_duration

logger = logging.getLogger(__name__)


def _preprocess_for_ocr(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


class EasyOCROnScreenExtractor:
    def __init__(
        self,
        languages: list[str],
        *,
        frame_step_sec: float,
        min_confidence: float = 0.3,
        device: str = "auto",
    ) -> None:
        self.languages = languages
        self.frame_step_sec = frame_step_sec
        self.min_confidence = min_confidence
        self.device = detect_torch_device(device)
        logger.info("OCR: загрузка EasyOCR %s на %s...", languages, self.device)
        self.reader = easyocr.Reader(
            languages,
            gpu=is_cuda_device(self.device) and torch.cuda.is_available(),
            verbose=False,
        )
        self.frame_sampler = RobustVideoFrameSampler(decoder_threads=1)
        logger.info("OCR: модель готова (шаг кадров %.1f сек)", self.frame_step_sec)

    def extract(self, video_path: str | Path) -> list[ModalityRecord]:
        duration = probe_video_duration(video_path)
        frames = self.frame_sampler.sample_regular_frames(
            video_path,
            max_end=duration,
            frame_step_sec=self.frame_step_sec,
        )
        results: list[ModalityRecord] = []
        for frame in frames:
            rgb = np.array(frame.image.convert("RGB"))
            processed = _preprocess_for_ocr(rgb)
            raw = self.reader.readtext(processed, detail=1, paragraph=False)
            blocks: list[tuple[float, str, float]] = []
            for bbox, text, confidence in raw:
                clean_text = str(text).strip()
                if not clean_text or float(confidence) < self.min_confidence:
                    continue
                top_y = float(bbox[0][1])
                blocks.append((top_y, clean_text, float(confidence)))
            blocks.sort(key=lambda item: item[0])
            if not blocks:
                continue

            full_text = " | ".join(text for _, text, _ in blocks)
            results.append(
                ModalityRecord(
                    video_file=str(video_path),
                    modality="ocr",
                    start=round(frame.timestamp, 3),
                    end=round(min(duration, frame.timestamp + self.frame_step_sec), 3),
                    text=full_text,
                    metadata={
                        "blocks": [
                            {"text": text, "confidence": round(confidence, 4)}
                            for _, text, confidence in blocks
                        ]
                    },
                )
            )
        return results

    def close(self) -> None:
        return None

