from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    env_path = config_path.resolve().parent.parent / ".env"
    load_dotenv(env_path if env_path.exists() else None)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _apply_env_overrides(config)
    return config


def _split_env_list(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n,;]+", raw) if item.strip()]


def _env_flag(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _apply_env_overrides(config: dict[str, Any]) -> None:
    env_map = {
        "TORCH_DEVICE": ("runtime", "device"),
        "TORCH_DTYPE": ("runtime", "torch_dtype"),
        "QDRANT_PATH": ("indexing", "qdrant_path"),
        "EMBEDDING_BACKEND": ("indexing", "embedding_backend"),
        "EMBEDDING_MODEL": ("indexing", "embedding_model"),
        "EMBEDDING_DIM": ("indexing", "embedding_dim"),
        "TEI_ENDPOINT": ("indexing", "tei_endpoint"),
        "QUERY_DECOUPLER_MODEL": ("query_decoupler", "model"),
        "QUERY_DECOUPLER_BACKEND": ("query_decoupler", "backend"),
        "GOOGLE_FLASH_THINKING_LEVEL": ("gemini", "flash_thinking_level"),
    }
    for env_key, (section, key) in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            if key == "embedding_dim":
                value = int(value)
            config.setdefault(section, {})[key] = value

    list_env_map = {
        "GOOGLE_API_KEYS": ("gemini", "api_keys"),
        "GOOGLE_MODEL_NAMES": ("gemini", "model_names"),
        "GEMINI_MODEL_NAMES": ("gemini", "model_names"),
        "GOOGLE_QUERY_DECOUPLER_MODEL_NAMES": ("gemini", "query_model_names"),
        "GEMINI_QUERY_DECOUPLER_MODEL_NAMES": ("gemini", "query_model_names"),
    }
    if "GOOGLE_API_KEYS" not in os.environ and "GOOGLE_API_KEY" in os.environ:
        list_env_map["GOOGLE_API_KEY"] = ("gemini", "api_keys")
    if "GOOGLE_MODEL_NAMES" not in os.environ and "GOOGLE_MODEL_NAME" in os.environ:
        list_env_map["GOOGLE_MODEL_NAME"] = ("gemini", "model_names")

    for env_key, (section, key) in list_env_map.items():
        value = os.environ.get(env_key)
        if value:
            config.setdefault(section, {})[key] = _split_env_list(value)

    numeric_env_map = {
        "GOOGLE_TEMPERATURE": ("gemini", "temperature", float),
        "GOOGLE_MAX_OUTPUT_TOKENS": ("gemini", "max_output_tokens", int),
        "GOOGLE_GENERATION_TIMEOUT_SECONDS": ("gemini", "timeout_sec", float),
        "GOOGLE_25_FLASH_THINKING_BUDGET": ("gemini", "flash_25_thinking_budget", int),
        "ANSWER_MAX_CONTEXT_CANDIDATES": ("answering", "max_context_candidates", int),
        "ANSWER_MAX_VIDEO_CANDIDATES": ("answering", "max_video_candidates", int),
        "ANSWER_VIDEO_FPS": ("answering", "video_fps", float),
        "ANSWER_WINDOW_PADDING_SEC": ("answering", "window_padding_sec", float),
    }
    for env_key, (section, key, caster) in numeric_env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            config.setdefault(section, {})[key] = caster(value)

    bool_env_map = {
        "GOOGLE_MINIMIZE_THINKING": ("gemini", "minimize_thinking"),
        "ANSWER_GENERATION_ENABLED": ("answering", "enabled"),
        "ANSWER_CLEANUP_UPLOADED_FILES": ("answering", "cleanup_uploaded_files"),
    }
    for env_key, (section, key) in bool_env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            config.setdefault(section, {})[key] = _env_flag(value)
