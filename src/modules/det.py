from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import spacy
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.models import ModalityRecord
from src.runtime import cleanup_torch_memory, detect_torch_device, resolve_torch_dtype
from src.utils.video_frames import RobustVideoFrameSampler
from src.utils.video_metadata import probe_video_duration

logger = logging.getLogger(__name__)


@dataclass
class _DetectedObject:
    node_id: int
    category: str


@dataclass
class _Relation:
    subject_id: int
    subject_cat: str
    predicate: str
    object_id: int
    object_cat: str


def _build_auxiliary_texts(
    objects: list[_DetectedObject],
    obj_counting: dict[str, int],
    relations: list[_Relation],
) -> dict[str, str]:
    texts: dict[str, str] = {}
    if obj_counting:
        lines = ["Object counting:"]
        for category, count in obj_counting.items():
            lines.append(f"- {category}: {count}")
        texts["obj_counting"] = "\n".join(lines)
    if objects:
        lines = ["Detected objects:"]
        for obj in objects:
            lines.append(f"- Object {obj.node_id} is a {obj.category}")
        texts["obj_list"] = "\n".join(lines)
    if relations:
        lines = ["Object relations:"]
        for relation in relations:
            lines.append(
                f"- {relation.subject_cat} ({relation.subject_id}) {relation.predicate} "
                f"{relation.object_cat} ({relation.object_id})"
            )
        texts["relations"] = "\n".join(lines)
    return texts


def _extract_scene_graph(caption: str, nlp: spacy.language.Language) -> tuple[list[_DetectedObject], dict[str, int], list[_Relation], dict[str, str]]:
    doc = nlp(caption)
    seen_lemmas: dict[str, int] = {}
    objects: list[_DetectedObject] = []
    obj_counting: dict[str, int] = {}
    relations: list[_Relation] = []

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            lemma = token.lemma_.lower()
            obj_counting[lemma] = obj_counting.get(lemma, 0) + 1
            if lemma not in seen_lemmas:
                seen_lemmas[lemma] = len(objects)
                objects.append(_DetectedObject(node_id=len(objects), category=lemma))

    for token in doc:
        if token.pos_ == "VERB":
            subject = None
            obj = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass") and child.pos_ in ("NOUN", "PROPN"):
                    subject = child
                if child.dep_ in ("dobj", "obj", "attr") and child.pos_ in ("NOUN", "PROPN"):
                    obj = child
            if subject and obj:
                subject_lemma = subject.lemma_.lower()
                object_lemma = obj.lemma_.lower()
                if subject_lemma in seen_lemmas and object_lemma in seen_lemmas:
                    relations.append(
                        _Relation(
                            subject_id=seen_lemmas[subject_lemma],
                            subject_cat=subject_lemma,
                            predicate=token.lemma_.lower(),
                            object_id=seen_lemmas[object_lemma],
                            object_cat=object_lemma,
                        )
                    )

    auxiliary_texts = _build_auxiliary_texts(objects, obj_counting, relations)
    return objects, obj_counting, relations, auxiliary_texts


class SceneGraphDETExtractor:
    def __init__(
        self,
        model_name: str,
        *,
        frame_step_sec: float,
        max_new_tokens: int,
        spacy_model: str,
        device: str = "auto",
        torch_dtype: str | None = "auto",
    ) -> None:
        self.model_name = model_name
        self.frame_step_sec = frame_step_sec
        self.max_new_tokens = max_new_tokens
        self.device = detect_torch_device(device)
        self.dtype = resolve_torch_dtype(torch_dtype, self.device)

        logger.info("DET: загрузка BLIP '%s' на %s...", model_name, self.device)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.frame_sampler = RobustVideoFrameSampler(decoder_threads=1)
        self.nlp = spacy.load(spacy_model)
        logger.info("DET: модель готова (шаг кадров %.1f сек)", self.frame_step_sec)

    def _generate_caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return " ".join(caption.split())

    def extract(self, video_path: str | Path) -> list[ModalityRecord]:
        duration = probe_video_duration(video_path)
        frames = self.frame_sampler.sample_regular_frames(
            video_path,
            max_end=duration,
            frame_step_sec=self.frame_step_sec,
        )
        results: list[ModalityRecord] = []
        for frame in frames:
            caption = self._generate_caption(frame.image.convert("RGB"))
            objects, obj_counting, relations, auxiliary_texts = _extract_scene_graph(caption, self.nlp)
            text_parts = [caption] + list(auxiliary_texts.values())
            text = "\n".join(part for part in text_parts if part)
            results.append(
                ModalityRecord(
                    video_file=str(video_path),
                    modality="det",
                    start=round(frame.timestamp, 3),
                    end=round(min(duration, frame.timestamp + self.frame_step_sec), 3),
                    text=text,
                    metadata={
                        "caption": caption,
                        "objects": [obj.category for obj in objects],
                        "counting": obj_counting,
                        "relations": [
                            {
                                "subject": rel.subject_cat,
                                "predicate": rel.predicate,
                                "object": rel.object_cat,
                            }
                            for rel in relations
                        ],
                    },
                )
            )
        return results

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        cleanup_torch_memory(self.device)

