from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import whisper

from src.models import ModalityRecord
from src.runtime import cleanup_torch_memory, detect_torch_device

logger = logging.getLogger(__name__)

# Global inside child worker processes to keep model weights loaded in memory
_worker_model = None

def _init_worker(model_name: str, device: str) -> None:
    global _worker_model
    import whisper
    _worker_model = whisper.load_model(model_name, device=device)

def _transcribe_chunk(
    chunk_path: str,
    language: str | None,
    no_speech_threshold: float,
    initial_prompt: str | None,
    offset: float,
) -> list[dict[str, object]]:
    global _worker_model
    if _worker_model is None:
        raise RuntimeError("Whisper model is not initialized in worker process")
        
    raw = _worker_model.transcribe(
        str(chunk_path),
        verbose=False,
        language=language,
        word_timestamps=False,
        condition_on_previous_text=True,
        no_speech_threshold=no_speech_threshold,
        initial_prompt=initial_prompt,
    )
    
    segments = []
    for segment in raw.get("segments", []):
        segments.append({
            "start": segment["start"] + offset,
            "end": segment["end"] + offset,
            "text": segment["text"],
            "no_speech_prob": segment.get("no_speech_prob", 0.0),
            "language": raw.get("language")
        })
    return segments


class WhisperASRExtractor:
    def __init__(
        self,
        model_name: str = "small",
        *,
        device: str = "auto",
        language: str | None = None,
        no_speech_threshold: float = 0.6,
        initial_prompt: str | None = None,
        workers: int = 1,
    ) -> None:
        self.model_name = model_name
        self.device = detect_torch_device(device)
        self.language = language
        self.no_speech_threshold = no_speech_threshold
        self.initial_prompt = initial_prompt
        self.workers = int(workers)

        if self.workers <= 1:
            logger.info("ASR: загрузка Whisper '%s' на %s (последовательный режим)...", model_name, self.device)
            try:
                self.model = whisper.load_model(model_name, device=self.device)
            except Exception:
                if self.device != "cpu":
                    self.device = "cpu"
                    self.model = whisper.load_model(model_name, device=self.device)
                else:
                    raise
            logger.info("ASR: последовательная модель готова")
        else:
            logger.info("ASR: инициализация в параллельном режиме (воркеров: %d, девайс: %s)", self.workers, self.device)
            logger.info("ASR: Проверка/скачивание модели '%s' на CPU для безопасного кэширования...", model_name)
            try:
                whisper.load_model(model_name, device="cpu")
                logger.info("ASR: Модель '%s' успешно подготовлена и закэширована на диске.", model_name)
            except Exception as e:
                logger.warning("Предупреждение при предварительной подготовке модели: %s", e)

    def extract(self, video_path: str | Path) -> list[ModalityRecord]:
        if self.workers > 1:
            try:
                return self._extract_parallel(video_path)
            except Exception as e:
                logger.warning(
                    "Ошибка при параллельной транскрипции ASR: %s. Переходим на последовательный режим.", 
                    e, exc_info=True
                )
        return self._extract_sequential(video_path)

    def _extract_sequential(self, video_path: str | Path) -> list[ModalityRecord]:
        # Lazily load model if we skipped it in constructor
        if not hasattr(self, "model"):
            logger.info("ASR: ленивая загрузка Whisper '%s' на %s для фолбэка...", self.model_name, self.device)
            self.model = whisper.load_model(self.model_name, device=self.device)

        raw = self.model.transcribe(
            str(video_path),
            verbose=False,
            language=self.language,
            word_timestamps=False,
            condition_on_previous_text=True,
            no_speech_threshold=self.no_speech_threshold,
            initial_prompt=self.initial_prompt,
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

    def _extract_parallel(self, video_path: str | Path) -> list[ModalityRecord]:
        video_path = Path(video_path).resolve()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            full_audio_path = tmp_path / "extracted_audio.wav"
            
            # 1. Extract audio from video
            logger.info("  [ASR Parallel] Извлечение аудио из %s...", video_path.name)
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                str(full_audio_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # 2. Read wav duration
            with wave.open(str(full_audio_path), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duration = frames / float(rate)
                
            logger.info("  [ASR Parallel] Длительность аудиодорожки: %.2fс", duration)
            
            # 3. Slice audio into chunks
            chunk_size = duration / self.workers
            chunks = []
            for i in range(self.workers):
                start = i * chunk_size
                end = min(duration, (i + 1) * chunk_size)
                chunk_file = tmp_path / f"chunk_{i}.wav"
                
                slice_cmd = [
                    "ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-i", str(full_audio_path), str(chunk_file)
                ]
                subprocess.run(slice_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                chunks.append((chunk_file, start))
                
            logger.info("  [ASR Parallel] Нарезка на %d частей выполнена.", self.workers)
            
            # 4. Transcribe chunks concurrently
            logger.info("  [ASR Parallel] Запуск пула процессов (%d воркеров)...", self.workers)
            all_segments = []
            with ProcessPoolExecutor(
                max_workers=self.workers, 
                initializer=_init_worker, 
                initargs=(self.model_name, self.device)
            ) as executor:
                futures = []
                for chunk_file, offset in chunks:
                    futures.append(
                        executor.submit(
                            _transcribe_chunk, 
                            chunk_file, 
                            self.language, 
                            self.no_speech_threshold, 
                            self.initial_prompt, 
                            offset
                        )
                    )
                
                for fut in futures:
                    all_segments.extend(fut.result())
                    
            # 5. Sort segments chronologically
            all_segments.sort(key=lambda s: s["start"])
            
            # 6. Map to ModalityRecords
            records = []
            for seg in all_segments:
                text = str(seg["text"]).strip()
                if not text:
                    continue
                records.append(
                    ModalityRecord(
                        video_file=str(video_path),
                        modality="asr",
                        start=round(float(seg["start"]), 3),
                        end=round(float(seg["end"]), 3),
                        text=text,
                        metadata={
                            "language": seg.get("language"),
                            "no_speech_prob": round(float(seg.get("no_speech_prob", 0.0)), 4),
                        },
                    )
                )
            
            logger.info("  [ASR Parallel] Обработка завершена. Объединено %d реплик.", len(records))
            return records

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        cleanup_torch_memory(self.device)
