"""Shared logging configuration for all pipeline scripts."""
from __future__ import annotations

import logging
import warnings

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Set up root logger and suppress noisy third-party loggers."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    logging.basicConfig(level=level, format=_LOG_FORMAT)

    for noisy_logger in (
        "httpx", "urllib3", "huggingface_hub", "huggingface_hub.utils._http",
        "safetensors", "easyocr", "google_genai.models", "google_genai.types",
        "google.genai.models", "google.genai.types", "google.genai._api_client",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)

    try:
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
    except (ImportError, AttributeError):
        pass

    try:
        from huggingface_hub import disable_progress_bars
        disable_progress_bars()
    except (ImportError, AttributeError):
        pass

    warnings.filterwarnings("ignore", message=".*resource_tracker.*")
    warnings.filterwarnings(
        "ignore",
        message=r".*unauthenticated requests to the HF Hub.*",
    )
