"""
ASR Module — Video Speech Recognizer
======================================
Извлекает аудио из видео и транскрибирует речь через OpenAI Whisper.
Результат — список сегментов с текстом и временными метками.

Зависимости:
    pip install openai-whisper
    apt install ffmpeg  (или brew install ffmpeg на Mac)

Запуск:
    python asr_module.py --video /path/to/video.mp4
    python asr_module.py --video /path/to/video.mp4 --model small
    python asr_module.py --video /path/to/video.mp4 --model tiny --output result.json
    python asr_module.py --video /path/to/video.mp4 --language ru

Доступные модели (по размеру/качеству):
    tiny   — ~75MB,  быстрее всего,  для коротких/чётких записей
    base   — ~145MB, чуть точнее
    small  — ~480MB, хороший баланс скорости и качества  ← рекомендуется
    medium — ~1.5GB, высокое качество, медленно на CPU
"""

import json
import argparse
import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import whisper


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start_sec: float        # начало сегмента
    end_sec: float          # конец сегмента
    text: str               # распознанный текст
    no_speech_prob: float   # вероятность тишины (чем выше — тем меньше доверия)


@dataclass
class ASRResult:
    video_path: str
    language: str           # определённый язык ("ru", "en", ...)
    duration_sec: float
    full_text: str          # весь транскрипт одной строкой
    segments: list[Segment]


# ---------------------------------------------------------------------------
# Извлечение аудио из видео через ffmpeg
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, tmp_dir: str) -> str:
    """
    Извлекает аудио из видео в WAV (16kHz mono) — формат, который ожидает Whisper.
    Возвращает путь к временному WAV-файлу.
    """
    audio_path = os.path.join(tmp_dir, "audio.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ar", "16000",      # 16kHz — стандарт для Whisper
        "-ac", "1",          # mono
        "-vn",               # без видео
        "-f", "wav",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg не смог извлечь аудио:\n{result.stderr}"
        )
    size_mb = os.path.getsize(audio_path) / 1024 / 1024
    print(f"  Аудио извлечено: {size_mb:.1f} MB  →  {audio_path}")
    return audio_path


# ---------------------------------------------------------------------------
# Форматирование времени
# ---------------------------------------------------------------------------

def fmt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Вывод результата
# ---------------------------------------------------------------------------

def print_result(result: ASRResult):
    sep = "─" * 60
    print(f"\n{'=' * 60}")
    print(f"  Язык       : {result.language}")
    print(f"  Длина      : {fmt_time(result.duration_sec)}")
    print(f"  Сегментов  : {len(result.segments)}")
    print(f"{'=' * 60}\n")

    print("СЕГМЕНТЫ:")
    print(sep)
    for seg in result.segments:
        # пропускаем сегменты где скорее всего тишина
        marker = "  " if seg.no_speech_prob < 0.6 else "?"
        print(
            f"{marker} [{fmt_time(seg.start_sec)} → {fmt_time(seg.end_sec)}]"
            f"  {seg.text.strip()}"
        )
    print(sep)

    print(f"\nПОЛНЫЙ ТЕКСТ:\n")
    # разбиваем на строки по 80 символов для читаемости
    words = result.full_text.split()
    line, lines = [], []
    for word in words:
        line.append(word)
        if sum(len(w) for w in line) + len(line) > 78:
            lines.append(" ".join(line))
            line = []
    if line:
        lines.append(" ".join(line))
    for l in lines:
        print(f"  {l}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_asr_pipeline(
    video_path: str,
    model_name: str = "tiny",
    language: Optional[str] = None,
    no_speech_threshold: float = 0.6,
    output_json: Optional[str] = None,
) -> ASRResult:

    print("=" * 60)
    print("  ASR Module  —  Video Speech Recognizer")
    print("=" * 60 + "\n")

    video_path = str(Path(video_path).resolve())
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Файл не найден: {video_path}")

    # 1. Извлекаем аудио
    print("Шаг 1. Извлечение аудио (ffmpeg) ...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = extract_audio(video_path, tmp_dir)

        # 2. Загружаем модель
        print(f"\nШаг 2. Загружаем Whisper '{model_name}' ...")
        print("  При первом запуске скачает веса и закеширует в ~/.cache/whisper/\n")
        model = whisper.load_model(model_name)

        # 3. Транскрибируем
        print("Шаг 3. Транскрибируем ...\n")
        transcribe_kwargs = dict(
            verbose=False,
            no_speech_threshold=no_speech_threshold,
            word_timestamps=False,
            condition_on_previous_text=True,
        )
        if language:
            transcribe_kwargs["language"] = language

        raw = model.transcribe(audio_path, **transcribe_kwargs)

    # 4. Разбираем результат
    detected_language = raw.get("language", "unknown")
    full_text = raw.get("text", "").strip()

    segments = []
    for seg in raw.get("segments", []):
        text = seg["text"].strip()
        if not text:
            continue
        segments.append(Segment(
            start_sec=round(seg["start"], 2),
            end_sec=round(seg["end"], 2),
            text=text,
            no_speech_prob=round(seg.get("no_speech_prob", 0.0), 3),
        ))

    # длительность — конец последнего сегмента
    duration = segments[-1].end_sec if segments else 0.0

    result = ASRResult(
        video_path=video_path,
        language=detected_language,
        duration_sec=duration,
        full_text=full_text,
        segments=segments,
    )

    # 5. Печатаем
    print_result(result)

    # 6. JSON
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        print(f"\n  Результаты сохранены → {output_json}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASR module: транскрибирует речь из видео через Whisper"
    )
    parser.add_argument(
        "--video", required=True,
        help="Путь к видеофайлу (mp4, avi, mov, mkv, ...)"
    )
    parser.add_argument(
        "--model", default="tiny",
        choices=["tiny", "base", "small", "medium", "large-v3-turbo"],
        help="Модель Whisper (default: tiny). Рекомендуется small для русского."
    )
    parser.add_argument(
        "--language", default=None,
        help="Язык аудио, например 'ru' или 'en'. "
             "Если не указан — Whisper определит автоматически."
    )
    parser.add_argument(
        "--no-speech-threshold", type=float, default=0.6,
        help="Порог вероятности тишины: сегменты выше помечаются '?' (default: 0.6)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Путь для сохранения результата в JSON (опционально)"
    )

    args = parser.parse_args()

    run_asr_pipeline(
        video_path=args.video,
        model_name=args.model,
        language=args.language,
        no_speech_threshold=args.no_speech_threshold,
        output_json=args.output,
    )
