from __future__ import annotations

import logging
import resource
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


import threading
from typing import Callable, Any

_telemetry_lock = threading.Lock()
_telemetry_listeners: list[Callable[..., Any]] = []

def register_telemetry_listener(listener: Callable[..., Any]) -> None:
    with _telemetry_lock:
        if listener not in _telemetry_listeners:
            _telemetry_listeners.append(listener)

def unregister_telemetry_listener(listener: Callable[..., Any]) -> None:
    with _telemetry_lock:
        if listener in _telemetry_listeners:
            _telemetry_listeners.remove(listener)


def get_ram_rss_mb() -> float:
    """Returns the resident set size (RAM usage) of the current process in MB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # On macOS, ru_maxrss is in bytes
        return rss / (1024 * 1024)
    # On Linux, ru_maxrss is in kilobytes
    return rss / 1024


class ProgressTelemetry:
    """Progress tracker that logs structured telemetry according to modern standards.

    Avoids spamming stdout by rate-limiting logs to percentage steps or time intervals.
    """

    def __init__(
        self,
        stage_name: str,
        total_items: int,
        device: str = "cpu",
        log_interval_sec: float = 30.0,
        log_percent_step: int = 10,
    ) -> None:
        self.stage_name = stage_name
        self.total_items = total_items
        self.device = device
        self.log_interval_sec = log_interval_sec
        self.log_percent_step = log_percent_step

        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.last_log_percent = 0

        logger.info(
            "[%s] Started | total_items=%d device=%s rss_mb=%.1f",
            self.stage_name.upper(),
            self.total_items,
            self.device,
            get_ram_rss_mb(),
        )
        sys.stdout.flush()
        sys.stderr.flush()

    def update(self, current_idx: int) -> None:
        if self.total_items <= 0:
            return

        now = time.perf_counter()
        elapsed = now - self.start_time

        percent = int(current_idx * 100 / self.total_items)
        time_since_last_log = now - self.last_log_time
        percent_since_last_log = percent - self.last_log_percent

        is_first = current_idx == 1
        is_last = current_idx == self.total_items
        crossed_percent = percent_since_last_log >= self.log_percent_step
        timeout = time_since_last_log >= self.log_interval_sec

        if is_first or is_last or crossed_percent or (timeout and percent_since_last_log >= 1):
            fps = current_idx / elapsed if elapsed > 0 else 0.0
            eta = (self.total_items - current_idx) / fps if fps > 0 and current_idx < self.total_items else 0.0

            def format_duration(sec: float) -> str:
                if sec < 60:
                    return f"{sec:.1f}s"
                m, s = divmod(int(sec), 60)
                h, m = divmod(m, 60)
                if h > 0:
                    return f"{h}h {m}m {s}s"
                return f"{m}m {s}s"

            rss_mb = get_ram_rss_mb()

            logger.info(
                "[%s] Progress: %d%% (%d/%d) | elapsed=%s eta=%s speed=%.2f/s rss_mb=%.1f device=%s status=%s",
                self.stage_name.upper(),
                percent,
                current_idx,
                self.total_items,
                format_duration(elapsed),
                format_duration(eta) if current_idx < self.total_items else "0s",
                fps,
                rss_mb,
                self.device,
                "completed" if is_last else "running",
            )
            sys.stdout.flush()
            sys.stderr.flush()

            self.last_log_time = now
            self.last_log_percent = percent

            # Thread-safely dispatch events to telemetry listeners
            with _telemetry_lock:
                listeners_copy = list(_telemetry_listeners)
            for listener in listeners_copy:
                try:
                    listener(
                        stage=self.stage_name,
                        percent=percent,
                        current_idx=current_idx,
                        total_items=self.total_items,
                        elapsed=elapsed,
                        eta=eta,
                        speed=fps,
                        rss_mb=rss_mb,
                        device=self.device,
                        status="completed" if is_last else "running"
                    )
                except Exception:
                    pass

