from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: dict[str, Any]) -> None:
    env_map = {
        "TORCH_DEVICE": ("runtime", "device"),
        "TORCH_DTYPE": ("runtime", "torch_dtype"),
        "QDRANT_PATH": ("indexing", "qdrant_path"),
        "EMBEDDING_MODEL": ("indexing", "embedding_model"),
        "RERANKER_MODEL": ("reranker", "model"),
        "QUERY_DECOUPLER_MODEL": ("query_decoupler", "model"),
    }
    for env_key, (section, key) in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            config.setdefault(section, {})[key] = value

