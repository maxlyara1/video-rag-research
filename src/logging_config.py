"""Shared logging configuration for all pipeline scripts."""
from __future__ import annotations

import logging
import warnings

import sys

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_CONFIGURED = False


class FlushingStreamHandler(logging.StreamHandler):
    """A standard stream handler that flushes after every log emit to prevent buffering."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def configure_logging(level: int = logging.INFO) -> None:
    """Set up root logger and suppress noisy third-party loggers."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)

    # Use FlushingStreamHandler to ensure logs write instantly
    handler = FlushingStreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)

    for noisy_logger in (
        "httpx", "urllib3", "safetensors", "easyocr", "google_genai.models", "google_genai.types",
        "google.genai.models", "google.genai.types", "google.genai._api_client",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)

    # Allow huggingface_hub to print info/warning logs
    logging.getLogger("huggingface_hub").setLevel(logging.INFO)

    try:
        import transformers
        transformers.logging.set_verbosity_info()
        transformers.utils.logging.enable_progress_bar()
    except (ImportError, AttributeError):
        pass

    try:
        from huggingface_hub import enable_progress_bars
        enable_progress_bars()
    except (ImportError, AttributeError):
        pass

    warnings.filterwarnings("ignore", message=".*resource_tracker.*")
    warnings.filterwarnings(
        "ignore",
        message=r".*unauthenticated requests to the HF Hub.*",
    )
