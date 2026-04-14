from __future__ import annotations

import json
import logging
import mimetypes
import re
import time
import warnings
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.models import CandidateWindow, QueryDecomposition
from src.prompts import QUERY_DECOUPLE_SYSTEM_PROMPT, QUERY_DECOUPLE_USER_TEMPLATE

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"google\.genai\._api_client",
)

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    genai = None
    genai_errors = None
    genai_types = None


logger = logging.getLogger(__name__)

GOOGLE_MIN_HTTP_TIMEOUT_SECONDS = 10.0
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class GeminiSettings:
    api_keys: tuple[str, ...]
    model_names: tuple[str, ...]
    temperature: float = 0.2
    max_output_tokens: int = 1024
    timeout_sec: float = 120.0
    minimize_thinking: bool = True
    flash_thinking_level: str = "minimal"
    flash_25_thinking_budget: int = 0


def _extract_response_text(response: Any) -> str:
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = str(getattr(part, "text", "") or "").strip()
            if part_text:
                return part_text
    return str(getattr(response, "text", "") or "").strip()


def _thinking_config_for_model(settings: GeminiSettings, model_name: str) -> Any | None:
    if genai_types is None or not settings.minimize_thinking:
        return None

    normalized_name = model_name.strip().lower()
    if "flash" not in normalized_name:
        return None

    if "2.5" in normalized_name:
        return genai_types.ThinkingConfig(thinking_budget=settings.flash_25_thinking_budget)

    if settings.flash_thinking_level:
        return genai_types.ThinkingConfig(thinking_level=settings.flash_thinking_level)
    return None


def _client_http_options(timeout_sec: float | None) -> Any | None:
    if genai_types is None or not timeout_sec or timeout_sec < GOOGLE_MIN_HTTP_TIMEOUT_SECONDS:
        return None
    return genai_types.HttpOptions(timeout=max(1, int(timeout_sec * 1000)))


def _stage_deadline(timeout_sec: float | None) -> float | None:
    if not timeout_sec or timeout_sec <= 0:
        return None
    return time.perf_counter() + timeout_sec


def _remaining_timeout_sec(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.perf_counter())


def _require_google_genai(settings: GeminiSettings) -> None:
    if genai is None or genai_types is None or genai_errors is None:
        raise RuntimeError("Gemini provider requires `google-genai`. Install it with `pip install -r requirements.txt`.")
    if not settings.api_keys:
        raise RuntimeError("Gemini provider requires GOOGLE_API_KEYS or GOOGLE_API_KEY in .env.")
    if not settings.model_names:
        raise RuntimeError("Gemini provider requires at least one model name.")


class GeminiModelClient:
    def __init__(self, settings: GeminiSettings) -> None:
        _require_google_genai(settings)
        self.settings = settings

    def generate(
        self,
        *,
        build_contents: Callable[[Any], Any],
        system_instruction: str | None = None,
        response_mime_type: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, str | None, int | None]:
        last_error: Exception | None = None
        deadline = _stage_deadline(self.settings.timeout_sec)

        for model_name in self.settings.model_names:
            for key_index, api_key in enumerate(self.settings.api_keys):
                remaining_timeout_sec = _remaining_timeout_sec(deadline)
                if deadline is not None and (remaining_timeout_sec is None or remaining_timeout_sec <= 0):
                    if last_error is not None:
                        raise last_error
                    raise TimeoutError("Превышен общий лимит времени Gemini-вызова")

                client = genai.Client(
                    api_key=api_key,
                    http_options=_client_http_options(remaining_timeout_sec),
                )
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=build_contents(client),
                        config=genai_types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=self.settings.temperature if temperature is None else temperature,
                            max_output_tokens=max_output_tokens or self.settings.max_output_tokens,
                            response_mime_type=response_mime_type,
                            thinking_config=_thinking_config_for_model(self.settings, model_name),
                        ),
                    )
                    text = _extract_response_text(response)
                    if text:
                        return text, model_name, key_index
                    last_error = RuntimeError("Gemini returned an empty response")
                except TimeoutError as exc:
                    last_error = exc
                    continue
                except (genai_errors.ClientError, genai_errors.ServerError, genai_errors.APIError) as exc:
                    last_error = exc
                    continue
                except Exception as exc:  # noqa: BLE001 - try next key/model, then report the last root cause.
                    last_error = exc
                    continue
                finally:
                    with suppress(Exception):
                        client.close()

        if last_error is not None:
            raise last_error
        raise RuntimeError("Не удалось получить ответ Gemini")


class GeminiQueryDecoupler:
    def __init__(self, settings: GeminiSettings, *, max_output_tokens: int = 256) -> None:
        self.client = GeminiModelClient(settings)
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _coerce_payload(query: str, payload: dict[str, object]) -> QueryDecomposition:
        asr_query = payload.get("asr_query")
        det_queries = payload.get("det_queries") or payload.get("R_det") or []
        det_mode = str(payload.get("det_mode") or payload.get("R_type") or "relation").lower()
        if det_mode not in {"location", "number", "relation"}:
            det_mode = "relation"

        normalized_det: list[str] = []
        for item in det_queries if isinstance(det_queries, list) else [det_queries]:
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
        prompt = QUERY_DECOUPLE_USER_TEMPLATE.format(query=query)
        try:
            text, _, _ = self.client.generate(
                build_contents=lambda _client: prompt,
                system_instruction=QUERY_DECOUPLE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=self.max_output_tokens,
                temperature=0.0,
            )
            json_match = _JSON_RE.search(text)
            payload = json.loads(json_match.group(0) if json_match else text)
            return self._coerce_payload(query, payload)
        except Exception as exc:  # noqa: BLE001 - retrieval can still proceed with the original query.
            logger.warning("Gemini query decouple failed, using original query: %s", exc)
            return QueryDecomposition(
                original_query=query,
                asr_query=query,
                det_queries=[],
                det_mode="relation",
            )

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class _VisualWindow:
    video_path: Path
    start: float
    end: float


class GeminiAnswerGenerator:
    def __init__(
        self,
        settings: GeminiSettings,
        *,
        max_context_candidates: int = 5,
        max_video_candidates: int = 1,
        window_padding_sec: float = 2.0,
        video_fps: float | None = 1.0,
        cleanup_uploaded_files: bool = True,
    ) -> None:
        self.client = GeminiModelClient(settings)
        self.max_context_candidates = max_context_candidates
        self.max_video_candidates = max_video_candidates
        self.window_padding_sec = window_padding_sec
        self.video_fps = video_fps
        self.cleanup_uploaded_files = cleanup_uploaded_files

    def generate(
        self,
        *,
        query: str,
        decomposition: QueryDecomposition,
        candidates: list[CandidateWindow],
    ) -> tuple[str, str | None, int | None]:
        visual_windows = self._select_visual_windows(candidates)
        prompt = self._build_prompt(query, decomposition, candidates)

        def _build_contents(client: Any, uploaded_files: list[Any]) -> Any:
            parts: list[Any] = []
            for window in visual_windows:
                parts.append(self._upload_video_part(client, window, uploaded_files))
            parts.append(genai_types.Part(text=prompt))
            return genai_types.Content(parts=parts)

        text, model_name, key_index = self._generate_with_cleanup(_build_contents)
        return text, model_name, key_index

    def _generate_with_cleanup(self, build_contents: Callable[[Any, list[Any]], Any]) -> tuple[str, str | None, int | None]:
        last_error: Exception | None = None
        deadline = _stage_deadline(self.client.settings.timeout_sec)

        for model_name in self.client.settings.model_names:
            for key_index, api_key in enumerate(self.client.settings.api_keys):
                remaining_timeout_sec = _remaining_timeout_sec(deadline)
                if deadline is not None and (remaining_timeout_sec is None or remaining_timeout_sec <= 0):
                    if last_error is not None:
                        raise last_error
                    raise TimeoutError("Превышен общий лимит времени Gemini-вызова")

                client = genai.Client(
                    api_key=api_key,
                    http_options=_client_http_options(remaining_timeout_sec),
                )
                contents = None
                uploaded_files: list[Any] = []
                try:
                    contents = build_contents(client, uploaded_files)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=genai_types.GenerateContentConfig(
                            temperature=self.client.settings.temperature,
                            max_output_tokens=self.client.settings.max_output_tokens,
                            thinking_config=_thinking_config_for_model(self.client.settings, model_name),
                        ),
                    )
                    text = _extract_response_text(response)
                    if text:
                        return text, model_name, key_index
                    last_error = RuntimeError("Gemini returned an empty response")
                except TimeoutError as exc:
                    last_error = exc
                    continue
                except (genai_errors.ClientError, genai_errors.ServerError, genai_errors.APIError) as exc:
                    last_error = exc
                    continue
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue
                finally:
                    if contents is not None:
                        self._cleanup_uploaded_files(client, uploaded_files)
                    with suppress(Exception):
                        client.close()

        if last_error is not None:
            raise last_error
        raise RuntimeError("Не удалось получить ответ Gemini")

    def _select_visual_windows(self, candidates: list[CandidateWindow]) -> list[_VisualWindow]:
        windows: list[_VisualWindow] = []
        seen: set[tuple[str, int, int]] = set()
        for candidate in candidates[: self.max_video_candidates]:
            video_path = Path(candidate.video_file)
            if not video_path.exists():
                continue
            start = max(0.0, candidate.start - self.window_padding_sec)
            end = max(start + 0.1, candidate.end + self.window_padding_sec)
            key = (str(video_path), int(start * 1000), int(end * 1000))
            if key in seen:
                continue
            seen.add(key)
            windows.append(_VisualWindow(video_path=video_path, start=start, end=end))
        return windows

    def _upload_video_part(self, client: Any, window: _VisualWindow, uploaded_files: list[Any]) -> Any:
        uploaded = client.files.upload(file=str(window.video_path))
        uploaded = self._wait_for_file(client, uploaded)
        uploaded_files.append(uploaded)

        mime_type = (
            getattr(uploaded, "mime_type", None)
            or mimetypes.guess_type(str(window.video_path))[0]
            or "video/mp4"
        )
        video_metadata_kwargs: dict[str, Any] = {
            "start_offset": f"{window.start:.3f}s",
            "end_offset": f"{window.end:.3f}s",
        }
        if self.video_fps is not None and self.video_fps > 0:
            video_metadata_kwargs["fps"] = self.video_fps

        return genai_types.Part(
            file_data=genai_types.FileData(
                file_uri=getattr(uploaded, "uri"),
                mime_type=mime_type,
            ),
            video_metadata=genai_types.VideoMetadata(**video_metadata_kwargs),
        )

    @staticmethod
    def _wait_for_file(client: Any, uploaded: Any, poll_sec: float = 2.0, max_wait_sec: float = 120.0) -> Any:
        deadline = time.perf_counter() + max_wait_sec
        current = uploaded
        while time.perf_counter() < deadline:
            state = getattr(current, "state", None)
            state_name = str(getattr(state, "name", state) or "").upper()
            if not state_name or state_name in {"ACTIVE", "SUCCEEDED", "READY"}:
                return current
            if "FAIL" in state_name:
                raise RuntimeError(f"Gemini file processing failed for {getattr(current, 'name', '<unknown>')}")
            time.sleep(poll_sec)
            with suppress(Exception):
                current = client.files.get(name=getattr(current, "name"))
        raise TimeoutError(f"Gemini file processing timeout for {getattr(uploaded, 'name', '<unknown>')}")

    def _cleanup_uploaded_files(self, client: Any, uploaded_files: list[Any]) -> None:
        if not self.cleanup_uploaded_files:
            return
        for uploaded in uploaded_files:
            name = getattr(uploaded, "name", None)
            if not name:
                continue
            with suppress(Exception):
                client.files.delete(name=name)

    def _build_prompt(
        self,
        query: str,
        decomposition: QueryDecomposition,
        candidates: list[CandidateWindow],
    ) -> str:
        context_blocks = self._format_context_blocks(candidates[: self.max_context_candidates])
        return (
            "Ты финальный LVLM-блок пайплайна Video-RAG.\n"
            "Тебе переданы: исходный вопрос, декомпозиция запроса R, найденные вспомогательные тексты "
            "A_m из ASR/OCR/DET и, если доступно, видеоклип верхнего найденного интервала.\n"
            "Ответь на русском. Используй только переданные данные и визуальный контекст. "
            "Если данных недостаточно, прямо скажи, что точный ответ не найден.\n"
            "Обязательно укажи файл видео и интервал времени, если нашёл релевантный фрагмент.\n\n"
            f"Вопрос Q:\n{query}\n\n"
            "Декомпозиция R:\n"
            f"- R_asr: {decomposition.asr_query or 'null'}\n"
            f"- R_det: {decomposition.det_queries}\n"
            f"- R_type: {decomposition.det_mode}\n\n"
            "Найденные вспомогательные тексты A_m, отсортированные по релевантности и времени:\n"
            f"{context_blocks or 'Нет найденных фрагментов.'}\n\n"
            "Формат ответа:\n"
            "Ответ: <краткий ответ>\n"
            "Фрагмент: <имя файла>, <MM:SS-MM:SS>, <почему этот фрагмент релевантен>\n"
        )

    @staticmethod
    def _format_context_blocks(candidates: list[CandidateWindow]) -> str:
        blocks: list[str] = []
        for index, candidate in enumerate(candidates, start=1):
            lines = [
                f"### Кандидат {index}",
                f"Видео: {Path(candidate.video_file).name}",
                f"Интервал: {_format_ts(candidate.start)}-{_format_ts(candidate.end)}",
                f"Score: {candidate.score:.4f}",
            ]
            for hit in sorted(candidate.hits, key=lambda item: (item.start, item.modality, -item.score)):
                lines.append(
                    f"[{hit.modality.upper()} { _format_ts(hit.start)}-{_format_ts(hit.end)} "
                    f"score={hit.score:.4f}]\n{hit.text}"
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def close(self) -> None:
        return None


def _format_ts(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"
