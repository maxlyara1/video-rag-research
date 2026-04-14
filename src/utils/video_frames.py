from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


logger = logging.getLogger(__name__)
_SEEK_EPSILON_SEC = 1e-3

try:
    from torchcodec.decoders import VideoDecoder
except (ImportError, RuntimeError) as exc:
    VideoDecoder = None
    _TORCHCODEC_IMPORT_ERROR = exc
else:
    _TORCHCODEC_IMPORT_ERROR = None


@dataclass(frozen=True)
class SampledFrame:
    timestamp: float
    image: Image.Image


def build_regular_timestamps(
    *,
    max_end: float,
    frame_step_sec: float,
    video_duration: float | None,
) -> list[float]:
    duration = max_end
    if video_duration is not None and video_duration > 0:
        duration = min(duration, video_duration)
    if duration <= 0:
        return []

    timestamps: list[float] = []
    cursor = 0.0
    while cursor < duration:
        timestamps.append(round(cursor, 3))
        cursor += frame_step_sec

    tail_timestamp = max(0.0, duration - _SEEK_EPSILON_SEC)
    if not timestamps or tail_timestamp - timestamps[-1] >= frame_step_sec * 0.5:
        timestamps.append(round(tail_timestamp, 3))
    return timestamps


class _FFmpegFrameSampler:
    """Fallback: extract frames via ffmpeg subprocess when torchcodec is unavailable."""

    def __init__(self, decoder_threads: int = 1) -> None:
        self.decoder_threads = decoder_threads
        self._ffmpeg = shutil.which("ffmpeg")
        if not self._ffmpeg:
            raise RuntimeError(
                "FFmpeg binary is required for video captioning when torchcodec is unavailable. "
                "Install ffmpeg (e.g. apt install ffmpeg) or fix torchcodec/FFmpeg compatibility."
            )

    def sample_regular_frames(
        self,
        video_path: str | Path,
        *,
        max_end: float,
        frame_step_sec: float,
    ) -> list[SampledFrame]:
        if max_end <= 0 or frame_step_sec <= 0:
            return []

        timestamps = build_regular_timestamps(
            max_end=max_end,
            frame_step_sec=frame_step_sec,
            video_duration=None,
        )
        if not timestamps:
            return []

        resolved_path = Path(video_path)
        sampled_frames: list[SampledFrame] = []

        for timestamp in timestamps:
            try:
                result = subprocess.run(
                    [
                        self._ffmpeg,
                        "-ss", str(timestamp),
                        "-i", str(resolved_path),
                        "-vframes", "1",
                        "-f", "image2pipe",
                        "-vcodec", "png",
                        "-loglevel", "error",
                        "-",
                    ],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                if result.returncode != 0:
                    logger.debug(
                        "FFmpeg failed for %s at %.3fs: %s",
                        resolved_path, timestamp,
                        result.stderr.decode(errors="replace")[:200],
                    )
                    continue
                image = Image.open(BytesIO(result.stdout)).convert("RGB")
                sampled_frames.append(SampledFrame(timestamp=timestamp, image=image))
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.debug("FFmpeg frame extraction failed for %s at %.3fs: %s", resolved_path, timestamp, exc)
                continue

        return sampled_frames


class _TorchCodecImpl:
    """TorchCodec-based frame extraction (preferred when available)."""

    def __init__(self, decoder_threads: int) -> None:
        self.decoder_threads = decoder_threads

    def sample_regular_frames(
        self,
        video_path: str | Path,
        *,
        max_end: float,
        frame_step_sec: float,
    ) -> list[SampledFrame]:
        if max_end <= 0 or frame_step_sec <= 0:
            return []

        resolved_path = Path(video_path)
        try:
            decoder = VideoDecoder(
                str(resolved_path),
                device="cpu",
                seek_mode="exact",
                dimension_order="NHWC",
                num_ffmpeg_threads=self.decoder_threads,
            )
            metadata = decoder.metadata
            stream_start = self._metadata_value(metadata, "begin_stream_seconds", 0.0)
            stream_end = self._metadata_value(metadata, "end_stream_seconds")
            video_duration = None if stream_end is None else max(0.0, stream_end - stream_start)
            timestamps = build_regular_timestamps(
                max_end=max_end,
                frame_step_sec=frame_step_sec,
                video_duration=video_duration,
            )
            if not timestamps:
                return []

            frame_batch = decoder.get_frames_played_at(
                seconds=[stream_start + timestamp for timestamp in timestamps]
            )
        except Exception as exc:
            logger.warning("Failed to decode %s via TorchCodec: %s", resolved_path, exc)
            return []

        sampled_frames: list[SampledFrame] = []
        frames = getattr(frame_batch, "data", None)
        if frames is None:
            logger.warning("TorchCodec returned no frame tensor for %s", resolved_path)
            return []

        returned_count = int(frames.shape[0])
        if returned_count != len(timestamps):
            logger.warning(
                "TorchCodec returned %d/%d frames for %s",
                returned_count,
                len(timestamps),
                resolved_path,
            )

        for timestamp, frame_tensor in zip(timestamps, frames):
            try:
                image = Image.fromarray(frame_tensor.cpu().numpy())
            except Exception as exc:
                logger.debug("Failed to convert decoded frame to image for %s at %.3fs: %s", resolved_path, timestamp, exc)
                continue
            sampled_frames.append(SampledFrame(timestamp=timestamp, image=image))

        return sampled_frames

    @staticmethod
    def _metadata_value(metadata: Any, name: str, default: float | None = None) -> float | None:
        value = getattr(metadata, name, default)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


_FALLBACK_LOGGED = False


class RobustVideoFrameSampler:
    """Sample sparse video frames through TorchCodec or FFmpeg fallback."""

    def __init__(self, decoder_threads: int = 1) -> None:
        global _FALLBACK_LOGGED
        if decoder_threads <= 0:
            raise ValueError("decoder_threads must be positive")
        self.decoder_threads = decoder_threads
        if VideoDecoder is not None:
            self._impl: _TorchCodecImpl | _FFmpegFrameSampler = _TorchCodecImpl(decoder_threads)
        else:
            if not _FALLBACK_LOGGED:
                logger.debug("torchcodec unavailable, falling back to FFmpeg subprocess")
                _FALLBACK_LOGGED = True
            self._impl = _FFmpegFrameSampler(decoder_threads)

    def sample_regular_frames(
        self,
        video_path: str | Path,
        *,
        max_end: float,
        frame_step_sec: float,
    ) -> list[SampledFrame]:
        return self._impl.sample_regular_frames(
            video_path=video_path,
            max_end=max_end,
            frame_step_sec=frame_step_sec,
        )
