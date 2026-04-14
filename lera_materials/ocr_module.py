"""
OCR Module — Video Text Extractor
==================================
Каждые N секунд берём кадр из видео и распознаём текст через EasyOCR.
Результат — список текстовых блоков с временными метками.

Зависимости:
    pip install easyocr opencv-python Pillow

Запуск:
    python ocr_module.py --video /path/to/video.mp4
    python ocr_module.py --video /path/to/video.mp4 --interval 3 --langs en ru
    python ocr_module.py --video /path/to/video.mp4 --interval 5 --output result.json --min-confidence 0.4
"""

import cv2
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import easyocr


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    text: str
    confidence: float


@dataclass
class FrameOCRResult:
    frame_idx: int
    timestamp_sec: float
    text_blocks: list[TextBlock]
    full_text: str          # все блоки склеены в одну строку через " | "


# ---------------------------------------------------------------------------
# Извлечение кадров
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, interval_sec: float) -> list[tuple[int, float, np.ndarray]]:
    """
    Возвращает список (frame_idx, timestamp_sec, bgr_frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    step = max(1, int(fps * interval_sec))

    print(f"  Видео    : {Path(video_path).name}")
    print(f"  FPS      : {fps:.1f}")
    print(f"  Длина    : {duration:.1f}с  ({total_frames} кадров)")
    print(f"  Интервал : {interval_sec}с → шаг {step} кадров")

    frames = []
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        if not ret:
            break
        timestamp = round(frame_idx / fps, 2)
        frames.append((frame_idx, timestamp, bgr))
        frame_idx += step

    cap.release()
    print(f"  Кадров к обработке: {len(frames)}\n")
    return frames


# ---------------------------------------------------------------------------
# Предобработка кадра для лучшего OCR
# ---------------------------------------------------------------------------

def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Лёгкое улучшение контраста — помогает EasyOCR на тёмных или
    засвеченных кадрах. Возвращает BGR (EasyOCR принимает BGR/RGB/путь).
    """
    # переводим в LAB, нормализуем яркость, возвращаем в BGR
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# OCR одного кадра
# ---------------------------------------------------------------------------

def run_ocr_on_frame(
    bgr: np.ndarray,
    reader: easyocr.Reader,
    min_confidence: float,
) -> list[TextBlock]:
    """
    Запускаем EasyOCR, фильтруем по порогу уверенности.
    Возвращает список TextBlock, отсортированный сверху вниз (по Y bbox).
    """
    processed = preprocess_for_ocr(bgr)

    # detail=1 → возвращает (bbox, text, confidence)
    raw_results = reader.readtext(processed, detail=1, paragraph=False)

    blocks = []
    for bbox, text, conf in raw_results:
        text = text.strip()
        if not text:
            continue
        if conf < min_confidence:
            continue
        blocks.append((bbox, TextBlock(text=text, confidence=round(conf, 3))))

    # сортируем по вертикали (верхний левый угол bbox[0][1])
    blocks.sort(key=lambda x: x[0][0][1])
    return [b for _, b in blocks]


# ---------------------------------------------------------------------------
# Форматирование вывода
# ---------------------------------------------------------------------------

def format_result(result: FrameOCRResult) -> str:
    sep = "─" * 55
    lines = [
        f"\n{sep}",
        f"  Frame #{result.frame_idx}  |  t = {result.timestamp_sec:.1f}s",
        sep,
    ]
    if result.text_blocks:
        for block in result.text_blocks:
            conf_bar = "█" * int(block.confidence * 10)
            lines.append(f"  [{conf_bar:<10}] {block.confidence:.2f}  \"{block.text}\"")
        lines.append(f"\n  Full text: {result.full_text}")
    else:
        lines.append("  — текст не обнаружен —")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ocr_pipeline(
    video_path: str,
    interval_sec: float = 5.0,
    langs: Optional[list[str]] = None,
    min_confidence: float = 0.3,
    output_json: Optional[str] = None,
) -> list[FrameOCRResult]:

    if langs is None:
        langs = ["ru", "en"]

    print("=" * 55)
    print("  OCR Module  —  Video Text Extractor")
    print("=" * 55 + "\n")

    # 1. Кадры
    frames = extract_frames(video_path, interval_sec)

    # 2. EasyOCR reader (загружается один раз)
    print(f"Загружаем EasyOCR (языки: {langs}) ...")
    print("  При первом запуске скачает модели (~100MB) и закеширует.\n")
    reader = easyocr.Reader(langs, gpu=False, verbose=False)
    print("  EasyOCR готов.\n")

    # 3. Основной цикл
    results: list[FrameOCRResult] = []
    total = len(frames)

    for i, (frame_idx, timestamp, bgr) in enumerate(frames):
        print(f"[{i+1}/{total}] frame={frame_idx}  t={timestamp:.1f}s ...", flush=True)

        blocks = run_ocr_on_frame(bgr, reader, min_confidence)

        full_text = " | ".join(b.text for b in blocks) if blocks else ""

        result = FrameOCRResult(
            frame_idx=frame_idx,
            timestamp_sec=timestamp,
            text_blocks=blocks,
            full_text=full_text,
        )
        results.append(result)
        print(format_result(result))

    # 4. Итоговый summary
    print("\n" + "=" * 55)
    print("  ИТОГО")
    print("=" * 55)
    frames_with_text = [r for r in results if r.text_blocks]
    print(f"  Кадров обработано  : {len(results)}")
    print(f"  Кадров с текстом   : {len(frames_with_text)}")
    all_texts = [b.text for r in results for b in r.text_blocks]
    print(f"  Текстовых блоков   : {len(all_texts)}")
    if all_texts:
        print(f"\n  Все найденные строки:")
        for r in results:
            if r.text_blocks:
                print(f"    t={r.timestamp_sec:.1f}s → {r.full_text}")

    # 5. JSON
    if output_json:
        data = [asdict(r) for r in results]
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n  Результаты сохранены → {output_json}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR module: извлекает текст из видео через EasyOCR"
    )
    parser.add_argument(
        "--video", required=True,
        help="Путь к видеофайлу (mp4, avi, mov, ...)"
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Интервал между кадрами в секундах (default: 5.0)"
    )
    parser.add_argument(
        "--langs", nargs="+", default=["ru", "en"],
        help="Языки для OCR, например: --langs ru en (default: ru en)"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.3,
        help="Минимальная уверенность для включения блока (default: 0.3)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Путь для сохранения результата в JSON (опционально)"
    )

    args = parser.parse_args()

    run_ocr_pipeline(
        video_path=args.video,
        interval_sec=args.interval,
        langs=args.langs,
        min_confidence=args.min_confidence,
        output_json=args.output,
    )
