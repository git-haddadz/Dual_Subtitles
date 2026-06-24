"""Audio extraction and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract a video's audio track into a WAV file."""
    from moviepy.editor import VideoFileClip

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with VideoFileClip(str(video_path)) as video:
        if video.audio is None:
            msg = f"No audio track found in {video_path}"
            raise ValueError(msg)
        video.audio.write_audiofile(str(output_path), verbose=False, logger=None)
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
