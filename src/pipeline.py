from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import load_config
from src.logging_config import configure_logging
from src.models import CandidateWindow, ModalityRecord, QueryDecomposition, SearchHit
from src.runtime import apply_runtime_config
from src.utils.video_metadata import is_video_file

logger = logging.getLogger(__name__)


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}с"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}м {secs:.0f}с"

if TYPE_CHECKING:
    from src.generation import GeminiAnswerGenerator, GeminiSettings
    from src.modules.query_decoupler import QueryDecoupler
    from src.retrieval import Embedder, QdrantStore


def _safe_stem(path: Path) -> str:
    normalized = unicodedata.normalize("NFKD", path.stem)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in ascii_name).strip("_")
    return cleaned.lower() or "video"


class VideoRAGPipeline:
    def __init__(self, config_path: str | Path | None = None) -> None:
        configure_logging()
        self.cfg = load_config(config_path)
        runtime_cfg = self.cfg["runtime"]
        apply_runtime_config(runtime_cfg)
        data_cfg = self.cfg["data"]

        self.materials_dir = Path(data_cfg["materials_dir"]).resolve()
        self.prepared_videos_dir = Path(data_cfg["prepared_videos_dir"]).resolve()
        self.artifacts_dir = Path(data_cfg["artifacts_dir"]).resolve()
        self.prepared_videos_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._embedder: "Embedder | None" = None
        self._store: "QdrantStore | None" = None
        self._query_decoupler: "QueryDecoupler | object | None" = None
        self._answer_generator: "GeminiAnswerGenerator | None" = None
        self._extractors: dict[str, object] = {}

    def _get_embedder(self) -> "Embedder":
        if self._embedder is None:
            runtime_cfg = self.cfg["runtime"]
            indexing_cfg = self.cfg["indexing"]
            backend = indexing_cfg.get("embedding_backend", "local")
            if backend == "tei":
                from src.retrieval import TEIEmbedder

                self._embedder = TEIEmbedder(
                    endpoint=indexing_cfg.get("tei_endpoint", "http://127.0.0.1:8080"),
                    model_name=indexing_cfg["embedding_model"],
                    dim=indexing_cfg.get("embedding_dim", 1024),
                    timeout_sec=indexing_cfg.get("tei_timeout_sec", 120.0),
                    query_instruction=indexing_cfg.get("query_instruction"),
                )
            elif backend == "local":
                from src.retrieval import Embedder

                self._embedder = Embedder(
                    model_name=indexing_cfg["embedding_model"],
                    device=runtime_cfg["device"],
                    torch_dtype=runtime_cfg["torch_dtype"],
                    max_length=indexing_cfg.get("embedding_max_length", 2048),
                    query_instruction=indexing_cfg.get("query_instruction"),
                    output_dim=indexing_cfg.get("embedding_dim"),
                )
            else:
                raise ValueError(f"Unknown indexing.embedding_backend: {backend}")
        return self._embedder

    def _get_store(self) -> "QdrantStore":
        from src.retrieval import QdrantStore

        if self._store is None:
            indexing_cfg = self.cfg["indexing"]
            self._store = QdrantStore(
                path=indexing_cfg["qdrant_path"],
                collection_prefix=indexing_cfg["collection_prefix"],
                embedding_dim=self._get_embedder().dim,
            )
        return self._store

    def _get_query_decoupler(self) -> "QueryDecoupler | object | None":
        if self._query_decoupler is None and self.cfg["query_decoupler"].get("enabled", True):
            runtime_cfg = self.cfg["runtime"]
            decouple_cfg = self.cfg["query_decoupler"]
            backend = decouple_cfg.get("backend", "local")
            if backend == "gemini":
                from src.generation import GeminiQueryDecoupler

                self._query_decoupler = GeminiQueryDecoupler(
                    self._gemini_settings(model_names_key="query_model_names"),
                    max_output_tokens=decouple_cfg.get("max_new_tokens", 192),
                )
            elif backend == "local":
                from src.modules.query_decoupler import QueryDecoupler

                self._query_decoupler = QueryDecoupler(
                    decouple_cfg["model"],
                    device=runtime_cfg["device"],
                    torch_dtype=runtime_cfg["torch_dtype"],
                    max_new_tokens=decouple_cfg.get("max_new_tokens", 192),
                )
            else:
                raise ValueError(f"Unknown query_decoupler.backend: {backend}")
        return self._query_decoupler

    def _get_answer_generator(self) -> "GeminiAnswerGenerator":
        if self._answer_generator is None:
            answer_cfg = self.cfg.get("answering", {})
            if not answer_cfg.get("enabled", True):
                raise RuntimeError("Answer generation is disabled in answering.enabled")
            provider = answer_cfg.get("provider", "gemini")
            if provider != "gemini":
                raise ValueError(f"Unknown answering.provider: {provider}")

            from src.generation import GeminiAnswerGenerator

            self._answer_generator = GeminiAnswerGenerator(
                self._gemini_settings(model_names_key="model_names"),
                max_context_candidates=answer_cfg.get("max_context_candidates", 5),
                max_video_candidates=answer_cfg.get("max_video_candidates", 1),
                window_padding_sec=answer_cfg.get("window_padding_sec", 2.0),
                video_fps=answer_cfg.get("video_fps", 1.0),
                cleanup_uploaded_files=answer_cfg.get("cleanup_uploaded_files", True),
            )
        return self._answer_generator

    def _gemini_settings(self, *, model_names_key: str) -> "GeminiSettings":
        from src.generation import GeminiSettings

        gemini_cfg = self.cfg.get("gemini", {})
        return GeminiSettings(
            api_keys=_as_tuple(gemini_cfg.get("api_keys")),
            model_names=_as_tuple(gemini_cfg.get(model_names_key) or gemini_cfg.get("model_names")),
            temperature=float(gemini_cfg.get("temperature", 0.2)),
            max_output_tokens=int(gemini_cfg.get("max_output_tokens", 1024)),
            timeout_sec=float(gemini_cfg.get("timeout_sec", 120)),
            minimize_thinking=bool(gemini_cfg.get("minimize_thinking", True)),
            flash_thinking_level=str(gemini_cfg.get("flash_thinking_level", "minimal")),
            flash_25_thinking_budget=int(gemini_cfg.get("flash_25_thinking_budget", 0)),
        )

    def _get_extractors(self) -> dict[str, object]:
        from src.modules import (
            EasyOCROnScreenExtractor,
            SceneGraphDETExtractor,
            WhisperASRExtractor,
        )

        if self._extractors:
            return self._extractors
        runtime_cfg = self.cfg["runtime"]
        if self.cfg["asr"].get("enabled", True):
            self._extractors["asr"] = WhisperASRExtractor(
                self.cfg["asr"]["model"],
                device=runtime_cfg["device"],
                language=self.cfg["asr"].get("language"),
                no_speech_threshold=self.cfg["asr"].get("no_speech_threshold", 0.6),
            )
        if self.cfg["ocr"].get("enabled", True):
            self._extractors["ocr"] = EasyOCROnScreenExtractor(
                self.cfg["ocr"]["languages"],
                frame_step_sec=self.cfg["ocr"].get("frame_step_sec", 5.0),
                min_confidence=self.cfg["ocr"].get("min_confidence", 0.3),
                device=runtime_cfg["device"],
            )
        if self.cfg["det"].get("enabled", True):
            self._extractors["det"] = SceneGraphDETExtractor(
                self.cfg["det"]["model"],
                frame_step_sec=self.cfg["det"].get("frame_step_sec", 5.0),
                max_new_tokens=self.cfg["det"].get("max_new_tokens", 60),
                spacy_model=self.cfg["det"].get("spacy_model", "en_core_web_sm"),
                device=runtime_cfg["device"],
                torch_dtype=runtime_cfg["torch_dtype"],
            )
        return self._extractors

    def prepare_dataset(self, force: bool = False) -> list[Path]:
        prepared: list[Path] = []
        if force and self.prepared_videos_dir.exists():
            shutil.rmtree(self.prepared_videos_dir)
            self.prepared_videos_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(self.materials_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".zip":
                prepared.extend(self._extract_zip(path))
            elif is_video_file(path):
                prepared.append(self._copy_video(path))

        unique_paths: dict[str, Path] = {str(path): path for path in prepared}
        return sorted(unique_paths.values())

    def _extract_zip(self, archive_path: Path) -> list[Path]:
        extracted: list[Path] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(tmp_dir)
            for path in sorted(Path(tmp_dir).rglob("*")):
                if is_video_file(path):
                    extracted.append(self._copy_video(path))
        return extracted

    def _copy_video(self, source_path: Path) -> Path:
        stem = _safe_stem(source_path)
        suffix = source_path.suffix.lower()
        destination = self.prepared_videos_dir / f"{stem}{suffix}"
        counter = 2
        while destination.exists() and destination.stat().st_size != source_path.stat().st_size:
            destination = self.prepared_videos_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        if not destination.exists():
            shutil.copy2(source_path, destination)
        return destination

    def list_prepared_videos(self) -> list[Path]:
        if not self.prepared_videos_dir.exists():
            return []
        return sorted(path for path in self.prepared_videos_dir.iterdir() if is_video_file(path))

    def enabled_modalities(self) -> list[str]:
        modalities: list[str] = []
        for modality in ("asr", "ocr", "det"):
            if self.cfg.get(modality, {}).get("enabled", False):
                modalities.append(modality)
        return modalities

    def process_video(self, video_path: str | Path, force: bool = False) -> dict[str, list[ModalityRecord]]:
        video_path = Path(video_path).resolve()
        artifact_path = self.artifacts_dir / f"{video_path.stem}.json"
        if artifact_path.exists() and not force:
            return self._load_artifact(artifact_path)
        return self._extract_and_save(video_path)

    _MODALITY_UNITS: dict[str, str] = {
        "asr": "сегм. речи",
        "ocr": "кадров с текстом",
        "det": "кадров с описанием",
    }

    def _extract_and_save(self, video_path: Path) -> dict[str, list[ModalityRecord]]:
        artifact_path = self.artifacts_dir / f"{video_path.stem}.json"
        t_video = time.perf_counter()
        records_by_modality: dict[str, list[ModalityRecord]] = {}
        for modality, extractor in self._get_extractors().items():
            t_mod = time.perf_counter()
            records = extractor.extract(video_path)
            records_by_modality[modality] = records
            unit = self._MODALITY_UNITS.get(modality, "записей")
            logger.info(
                "  [%s] %d %s за %s",
                modality.upper(), len(records), unit,
                _fmt_elapsed(time.perf_counter() - t_mod),
            )
        self._save_artifact(artifact_path, records_by_modality)
        logger.info("  итого: %s", _fmt_elapsed(time.perf_counter() - t_video))
        return records_by_modality

    @staticmethod
    def _flush_cache_summary(
        cached_names: list[str], next_idx: int, total: int,
    ) -> None:
        if not cached_names:
            return
        count = len(cached_names)
        end = next_idx - 1
        start = end - count + 1
        if count == 1:
            logger.info("  Видео %d/%d: %s (кэш)", start, total, cached_names[0])
        else:
            logger.info("  Видео %d–%d/%d: из кэша (%d шт)", start, end, total, count)
        cached_names.clear()

    def build_indexes(self, recreate: bool = False, force: bool = False) -> dict[str, int]:
        videos = self.list_prepared_videos()
        if not videos:
            raise RuntimeError("Нет подготовленных видео. Сначала запусти prepare_dataset.")

        modalities = self.enabled_modalities()
        total = len(videos)
        logger.info(
            "=== Этап 1/2: извлечение [%s] из %d видео ===",
            ", ".join(m.upper() for m in modalities), total,
        )

        t_stage1 = time.perf_counter()
        all_records: dict[str, list[ModalityRecord]] = {m: [] for m in modalities}
        cached_names: list[str] = []

        for idx, video_path in enumerate(videos, 1):
            artifact_path = self.artifacts_dir / f"{video_path.stem}.json"
            has_cache = artifact_path.exists() and not force

            if has_cache:
                cached_names.append(video_path.name)
                artifact = self._load_artifact(artifact_path)
            else:
                self._flush_cache_summary(cached_names, idx, total)
                logger.info("Видео %d/%d: %s", idx, total, video_path.name)
                artifact = self._extract_and_save(video_path.resolve())

            for modality, records in artifact.items():
                all_records.setdefault(modality, []).extend(records)

        self._flush_cache_summary(cached_names, total + 1, total)
        self._close_extractors()

        all_records = {
            modality: self._records_for_index(modality, records)
            for modality, records in all_records.items()
        }
        total_records = sum(len(r) for r in all_records.values())
        logger.info(
            "Этап 1/2 завершён за %s — %d записей",
            _fmt_elapsed(time.perf_counter() - t_stage1), total_records,
        )

        logger.info("=== Этап 2/2: эмбеддинг и запись в Qdrant ===")
        t_stage2 = time.perf_counter()

        store = self._get_store()
        embedder = self._get_embedder()
        for modality, records in all_records.items():
            if recreate:
                store.recreate_collection(modality)
            if not records:
                logger.info("  [%s] 0 записей — пропуск", modality.upper())
                continue
            t_emb = time.perf_counter()
            logger.info("  [%s] эмбеддинг %d записей...", modality.upper(), len(records))
            embeddings = embedder.embed(
                [record.text for record in records],
                batch_size=self.cfg["indexing"].get("batch_size", 8),
            )
            store.upsert_records(modality, records, embeddings)
            logger.info(
                "  [%s] готово за %s",
                modality.upper(), _fmt_elapsed(time.perf_counter() - t_emb),
            )

        logger.info(
            "Этап 2/2 завершён за %s", _fmt_elapsed(time.perf_counter() - t_stage2),
        )
        return {modality: len(records) for modality, records in all_records.items()}

    def search(self, query: str) -> tuple[QueryDecomposition, list[CandidateWindow]]:
        query_decoupler = self._get_query_decoupler()
        decomposition = (
            query_decoupler.decouple(query)
            if query_decoupler is not None
            else QueryDecomposition(original_query=query, asr_query=query, det_queries=[], det_mode="relation")
        )

        modality_queries = {
            "asr": self._build_text_retrieval_query(query, decomposition.asr_query),
            "ocr": self._build_text_retrieval_query(query, decomposition.asr_query),
            "det": self._build_det_query(decomposition),
        }

        all_hits: list[SearchHit] = []
        top_k = self.cfg["search"].get("per_modality_top_k", 12)
        store = self._get_store()
        embedder = self._get_embedder()
        score_threshold = self.cfg["search"].get("score_threshold")
        for modality in self.enabled_modalities():
            modality_query = modality_queries.get(modality) or query
            if not modality_query.strip():
                continue
            query_vector = embedder.embed_query(modality_query)
            filter_payload = (
                {"det_type": self._det_type_for_mode(decomposition.det_mode)}
                if modality == "det"
                else None
            )
            hits = store.search(modality, query_vector, top_k=top_k, filter_payload=filter_payload)
            if score_threshold is not None:
                hits = [hit for hit in hits if hit.score >= float(score_threshold)]
            all_hits.extend(hits)

        candidates = self._merge_hits(all_hits)
        final_top_k = self.cfg["search"].get("final_top_k", 5)
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[:final_top_k]
        return decomposition, candidates

    def answer(self, query: str) -> tuple[QueryDecomposition, list[CandidateWindow], str, str | None, int | None]:
        decomposition, candidates = self.search(query)
        generator = self._get_answer_generator()
        answer, model_name, key_index = generator.generate(
            query=query,
            decomposition=decomposition,
            candidates=candidates,
        )
        return decomposition, candidates, answer, model_name, key_index

    def _build_det_query(self, decomposition: QueryDecomposition) -> str:
        if not decomposition.det_queries:
            return decomposition.original_query
        return ", ".join(part for part in decomposition.det_queries if part)

    @staticmethod
    def _det_type_for_mode(det_mode: str) -> str:
        if det_mode == "number":
            return "number"
        if det_mode == "location":
            return "location"
        return "relation"

    def _records_for_index(self, modality: str, records: list[ModalityRecord]) -> list[ModalityRecord]:
        if modality != "det":
            return records

        normalized: list[ModalityRecord] = []
        for record in records:
            if record.metadata.get("det_type"):
                normalized.append(record)
            else:
                normalized.extend(self._split_legacy_det_record(record))
        return normalized

    @staticmethod
    def _split_legacy_det_record(record: ModalityRecord) -> list[ModalityRecord]:
        metadata = record.metadata or {}
        texts: dict[str, str] = {}

        counting = metadata.get("counting") or {}
        if counting:
            lines = ["Object counting:"]
            for category, count in counting.items():
                lines.append(f"- {category}: {count}")
            texts["number"] = "\n".join(lines)

        objects = metadata.get("objects") or []
        if objects:
            lines = ["Detected object locations:"]
            for index, category in enumerate(objects):
                lines.append(f"- Object {index} is a {category} located in the sampled frame")
            texts["location"] = "\n".join(lines)

        relations = metadata.get("relations") or []
        if relations:
            lines = ["Object relations:"]
            for index, relation in enumerate(relations):
                subject = relation.get("subject", f"object_{index}")
                predicate = relation.get("predicate", "related_to")
                obj = relation.get("object", "object")
                lines.append(f"- {subject} {predicate} {obj}")
            texts["relation"] = "\n".join(lines)

        if not texts and record.text:
            texts["relation"] = record.text

        return [
            ModalityRecord(
                video_file=record.video_file,
                modality=record.modality,
                start=record.start,
                end=record.end,
                text=text,
                metadata={**metadata, "det_type": det_type},
            )
            for det_type, text in texts.items()
        ]

    @staticmethod
    def _build_text_retrieval_query(original_query: str, decoupled_query: str | None) -> str:
        if not decoupled_query:
            return original_query
        if decoupled_query.strip() == original_query.strip():
            return original_query
        return f"{original_query}\n{decoupled_query}"

    def _merge_hits(self, hits: list[SearchHit]) -> list[CandidateWindow]:
        gap = float(self.cfg["search"].get("merge_gap_sec", 6.0))
        weights = self.cfg["search"].get("modality_weights", {})
        candidates: list[CandidateWindow] = []

        for hit in sorted(hits, key=lambda item: (item.video_file, item.start, item.end, -item.score)):
            weight = float(weights.get(hit.modality, 1.0))
            merged = False
            for candidate in candidates:
                if candidate.video_file != hit.video_file:
                    continue
                if hit.start <= candidate.end + gap and hit.end >= candidate.start - gap:
                    candidate.start = min(candidate.start, hit.start)
                    candidate.end = max(candidate.end, hit.end)
                    candidate.score += hit.score * weight
                    candidate.hits.append(hit)
                    merged = True
                    break
            if not merged:
                candidates.append(
                    CandidateWindow(
                        video_file=hit.video_file,
                        start=hit.start,
                        end=hit.end,
                        score=hit.score * weight,
                        hits=[hit],
                    )
                )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return self._deduplicate_candidates(candidates)

    @staticmethod
    def _deduplicate_candidates(candidates: list[CandidateWindow], iou_threshold: float = 0.5) -> list[CandidateWindow]:
        deduped: list[CandidateWindow] = []
        for candidate in candidates:
            dominated = False
            for existing in deduped:
                if candidate.video_file != existing.video_file:
                    continue
                inter = max(0.0, min(candidate.end, existing.end) - max(candidate.start, existing.start))
                union = (candidate.end - candidate.start) + (existing.end - existing.start) - inter
                if union > 0 and inter / union >= iou_threshold:
                    dominated = True
                    break
            if not dominated:
                deduped.append(candidate)
        return deduped

    @staticmethod
    def _save_artifact(path: Path, records_by_modality: dict[str, list[ModalityRecord]]) -> None:
        payload = {
            modality: [
                {
                    "video_file": record.video_file,
                    "modality": record.modality,
                    "start": record.start,
                    "end": record.end,
                    "text": record.text,
                    "metadata": record.metadata,
                }
                for record in records
            ]
            for modality, records in records_by_modality.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _load_artifact(path: Path) -> dict[str, list[ModalityRecord]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded: dict[str, list[ModalityRecord]] = {}
        for modality, records in payload.items():
            loaded[modality] = [
                ModalityRecord(
                    video_file=record["video_file"],
                    modality=record["modality"],
                    start=float(record["start"]),
                    end=float(record["end"]),
                    text=record["text"],
                    metadata=record.get("metadata") or {},
                )
                for record in records
            ]
        return loaded

    def close(self) -> None:
        self._close_extractors()
        if self._query_decoupler is not None:
            self._query_decoupler.close()
            self._query_decoupler = None
        if self._embedder is not None:
            self._embedder.close()
            self._embedder = None
        if self._answer_generator is not None:
            self._answer_generator.close()
            self._answer_generator = None
        if self._store is not None:
            self._store.close()
            self._store = None

    def _close_extractors(self) -> None:
        for extractor in self._extractors.values():
            extractor.close()
        self._extractors = {}


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    try:
        return tuple(str(item).strip() for item in value if str(item).strip())  # type: ignore[operator]
    except TypeError:
        text = str(value).strip()
        return (text,) if text else ()
