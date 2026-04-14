"""
DET Module — Video Scene Graph Builder
======================================
Каждые N секунд берём кадр из видео, прогоняем через BLIP (image captioning),
из caption извлекаем объекты и отношения, строим scene graph как в Video-RAG.

Зависимости:
    pip install transformers torch torchvision opencv-python spacy Pillow
    python -m spacy download en_core_web_sm

Запуск:
    python det_module.py --video /path/to/video.mp4 --interval 5
    python det_module.py --video /path/to/video.mp4 --interval 3 --output results.json
"""

import cv2
import json
import argparse
import textwrap
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
import torch


# ---------------------------------------------------------------------------
# Dataclasses для scene graph
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    node_id: int
    category: str          # существительное из caption


@dataclass
class Relation:
    subject_id: int
    subject_cat: str
    predicate: str         # глагол или предлог
    object_id: int
    object_cat: str


@dataclass
class FrameSceneGraph:
    frame_idx: int
    timestamp_sec: float
    caption: str
    objects: list[DetectedObject]
    obj_counting: dict[str, int]          # {category: count}
    relations: list[Relation]
    auxiliary_texts: dict[str, str]        # финальные текстовые представления


# ---------------------------------------------------------------------------
# Извлечение кадров
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, interval_sec: float) -> list[tuple[int, float, Image.Image]]:
    """
    Возвращает список (frame_idx, timestamp_sec, PIL.Image)
    каждые interval_sec секунд видео.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    step = int(fps * interval_sec)

    print(f"  Видео: {Path(video_path).name}")
    print(f"  FPS: {fps:.1f}  |  Длительность: {duration:.1f}с  |  Всего кадров: {total_frames}")
    print(f"  Интервал: {interval_sec}с → шаг {step} кадров")

    frames = []
    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        if not ret:
            break
        timestamp = frame_idx / fps
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        frames.append((frame_idx, round(timestamp, 2), pil_img))
        frame_idx += step
        if frame_idx >= total_frames:
            break

    cap.release()
    print(f"  Извлечено кадров: {len(frames)}\n")
    return frames


# ---------------------------------------------------------------------------
# BLIP captioning
# ---------------------------------------------------------------------------

def load_blip(model_name: str = "Salesforce/blip-image-captioning-base"):
    print(f"Загружаем BLIP: {model_name} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  Устройство: {device}\n")
    return processor, model, device


def generate_caption(image: Image.Image, processor, model, device: str) -> str:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# ---------------------------------------------------------------------------
# Scene Graph из caption (spaCy)
# ---------------------------------------------------------------------------

def load_spacy(model: str = "en_core_web_sm"):
    try:
        return spacy.load(model)
    except OSError:
        raise OSError(
            f"Модель spaCy '{model}' не найдена.\n"
            f"Установите: python -m spacy download {model}"
        )


def extract_scene_graph(
    caption: str,
    frame_idx: int,
    timestamp_sec: float,
    nlp,
) -> FrameSceneGraph:
    """
    Из caption строим scene graph:
      - объекты: существительные (NOUN/PROPN), дедупликация по лемме
      - отношения: subj → verb/prep → obj из dependency tree
    """
    doc = nlp(caption)

    # --- 1. Объекты: все NOUN и PROPN, исключаем стоп-слова ---
    seen_lemmas: dict[str, int] = {}   # lemma → node_id
    objects: list[DetectedObject] = []

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            lemma = token.lemma_.lower()
            if lemma not in seen_lemmas:
                node_id = len(objects)
                seen_lemmas[lemma] = node_id
                objects.append(DetectedObject(node_id=node_id, category=lemma))

    # --- 2. Подсчёт объектов ---
    # Считаем сколько раз каждая лемма встречается в тексте
    obj_counting: dict[str, int] = {}
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            lemma = token.lemma_.lower()
            obj_counting[lemma] = obj_counting.get(lemma, 0) + 1

    # --- 3. Отношения из dependency tree ---
    relations: list[Relation] = []

    for token in doc:
        # Ищем глаголы с субъектом и объектом
        if token.pos_ == "VERB":
            subj_token = None
            obj_token = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass") and child.pos_ in ("NOUN", "PROPN"):
                    subj_token = child
                if child.dep_ in ("dobj", "obj", "attr") and child.pos_ in ("NOUN", "PROPN"):
                    obj_token = child
            if subj_token and obj_token:
                subj_lemma = subj_token.lemma_.lower()
                obj_lemma = obj_token.lemma_.lower()
                if subj_lemma in seen_lemmas and obj_lemma in seen_lemmas:
                    relations.append(Relation(
                        subject_id=seen_lemmas[subj_lemma],
                        subject_cat=subj_lemma,
                        predicate=token.lemma_.lower(),
                        object_id=seen_lemmas[obj_lemma],
                        object_cat=obj_lemma,
                    ))

        # Ищем предложные отношения (prep: X on/in/near Y)
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            for child in token.children:
                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj" and pobj.pos_ in ("NOUN", "PROPN"):
                            subj_lemma = token.lemma_.lower()
                            obj_lemma = pobj.lemma_.lower()
                            if subj_lemma in seen_lemmas and obj_lemma in seen_lemmas:
                                relations.append(Relation(
                                    subject_id=seen_lemmas[subj_lemma],
                                    subject_cat=subj_lemma,
                                    predicate=child.text.lower(),  # "on", "in", "near" ...
                                    object_id=seen_lemmas[obj_lemma],
                                    object_cat=obj_lemma,
                                ))

    # --- 4. Формируем auxiliary texts (как в статье) ---
    auxiliary_texts = build_auxiliary_texts(objects, obj_counting, relations)

    return FrameSceneGraph(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        caption=caption,
        objects=objects,
        obj_counting=obj_counting,
        relations=relations,
        auxiliary_texts=auxiliary_texts,
    )


# ---------------------------------------------------------------------------
# Auxiliary texts (текстовое представление как в Video-RAG)
# ---------------------------------------------------------------------------

def build_auxiliary_texts(
    objects: list[DetectedObject],
    obj_counting: dict[str, int],
    relations: list[Relation],
) -> dict[str, str]:
    """
    Строим три вида текстов из статьи:
      A_cnt  — Object counting
      A_loc  — здесь у нас нет bbox, поэтому пишем просто список объектов
      A_rel  — Relative relations between objects
    """
    texts = {}

    # A_cnt — подсчёт
    if obj_counting:
        lines = ["Object counting:"]
        for cat, cnt in obj_counting.items():
            lines.append(f"  - {cat}: {cnt}")
        texts["obj_counting"] = "\n".join(lines)

    # A_loc — объекты (без bbox, т.к. BLIP не даёт координат)
    if objects:
        lines = ["Detected objects:"]
        for obj in objects:
            lines.append(f"  - Object {obj.node_id} is a {obj.category}")
        texts["obj_list"] = "\n".join(lines)

    # A_rel — отношения
    if relations:
        lines = ["Object relations:"]
        for rel in relations:
            lines.append(
                f"  - Object {rel.subject_id} ({rel.subject_cat}) "
                f"[{rel.predicate}] "
                f"Object {rel.object_id} ({rel.object_cat})"
            )
        texts["obj_relations"] = "\n".join(lines)
    else:
        texts["obj_relations"] = "Object relations: NULL"

    return texts


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_scene_graph(sg: FrameSceneGraph):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Frame #{sg.frame_idx}  |  t = {sg.timestamp_sec:.1f}s")
    print(sep)
    print(f"  Caption : {sg.caption}")
    print()
    for key, text in sg.auxiliary_texts.items():
        print(textwrap.indent(text, "  "))
        print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_det_pipeline(
    video_path: str,
    interval_sec: float = 5.0,
    output_json: Optional[str] = None,
    blip_model: str = "Salesforce/blip-image-captioning-base",
    spacy_model: str = "en_core_web_sm",
):
    print("=" * 60)
    print("  DET Module  —  Video Scene Graph Builder")
    print("=" * 60 + "\n")

    # 1. Кадры
    frames = extract_frames(video_path, interval_sec)

    # 2. Модели
    processor, blip, device = load_blip(blip_model)
    nlp = load_spacy(spacy_model)

    # 3. Основной цикл
    scene_graphs: list[FrameSceneGraph] = []
    total = len(frames)

    for i, (frame_idx, timestamp, pil_img) in enumerate(frames):
        print(f"[{i+1}/{total}] frame={frame_idx}  t={timestamp:.1f}s", end="  →  ", flush=True)

        caption = generate_caption(pil_img, processor, blip, device)
        print(f'caption: "{caption}"')

        sg = extract_scene_graph(caption, frame_idx, timestamp, nlp)
        scene_graphs.append(sg)
        print_scene_graph(sg)

    # 4. Итоговый summary
    print("\n" + "=" * 60)
    print("  ИТОГО по видео")
    print("=" * 60)
    all_objects: dict[str, int] = {}
    for sg in scene_graphs:
        for cat, cnt in sg.obj_counting.items():
            all_objects[cat] = all_objects.get(cat, 0) + cnt

    print("\nВсе объекты (суммарно по всем кадрам):")
    for cat, cnt in sorted(all_objects.items(), key=lambda x: -x[1]):
        bar = "█" * min(cnt, 20)
        print(f"  {cat:<20} {bar}  ({cnt})")

    total_rels = sum(len(sg.relations) for sg in scene_graphs)
    print(f"\nВсего отношений найдено: {total_rels}")

    # 5. Сохраняем JSON
    if output_json:
        data = [asdict(sg) for sg in scene_graphs]
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены → {output_json}")

    return scene_graphs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DET module: строит scene graph для видео через BLIP captioning"
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
        "--output", type=str, default=None,
        help="Путь для сохранения результата в JSON (опционально)"
    )
    parser.add_argument(
        "--blip-model", type=str,
        default="Salesforce/blip-image-captioning-base",
        help="Название BLIP-модели с HuggingFace"
    )
    parser.add_argument(
        "--spacy-model", type=str, default="en_core_web_sm",
        help="Название spaCy-модели (default: en_core_web_sm)"
    )
    args = parser.parse_args()

    run_det_pipeline(
        video_path=args.video,
        interval_sec=args.interval,
        output_json=args.output,
        blip_model=args.blip_model,
        spacy_model=args.spacy_model,
    )