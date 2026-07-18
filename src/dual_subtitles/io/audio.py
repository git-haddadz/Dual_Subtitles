"""Audio extraction and normalization."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract a video's audio track with ffmpeg without importing MoviePy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() if exc.stderr else "unknown ffmpeg error"
        msg = f"Could not extract audio from {video_path}: {details}"
        raise RuntimeError(msg) from exc
    return output_path


def normalize_audio(
    input_path: Path,
    output_path: Path,
    *,
    frame_rate: int = 16_000,
    channels: int = 1,
) -> Path:
    """Normalize audio to mono WAV at the requested frame rate."""
    from pydub import AudioSegment

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(frame_rate).set_channels(channels)
    audio.export(output_path, format="wav")
    return output_path


def load_audio(path: Path) -> Any:
    """Load audio with pydub.

    The return type is intentionally left as the pydub runtime type so tests do
    not need pydub installed merely to import this module.
    """
    from pydub import AudioSegment

    return AudioSegment.from_file(path)


def detect_silence_boundaries(
    audio: Any,
    *,
    min_silence_duration: float,
    threshold_offset: float,
) -> list[float]:
    """Return silence midpoints in seconds for acoustic window boundaries."""
    from pydub.silence import detect_silence

    if len(audio) == 0:
        return []
    audio_level = float(audio.dBFS)
    silence_threshold = (
        audio_level - threshold_offset if math.isfinite(audio_level) else -50.0
    )
    ranges = detect_silence(
        audio,
        min_silence_len=max(1, round(min_silence_duration * 1000)),
        silence_thresh=silence_threshold,
        seek_step=10,
    )
    return [(start + end) / 2000 for start, end in ranges]
