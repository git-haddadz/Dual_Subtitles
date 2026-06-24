"""Speech-to-text integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dual_subtitles.models.subtitle import Segment, SubtitleSegment


@dataclass(slots=True)
class WhisperTranscriber:
    """Thin wrapper around the Transformers Whisper pipeline."""

    model_name: str = "openai/whisper-large-v3"
    device: int | str | None = None
    _pipeline: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load the Transformers pipeline and choose a default device."""
        import torch
        from transformers import pipeline

        device = self.device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=device,
            return_timestamps=True,
        )

    def transcribe_file(
        self,
        audio_path: Path,
        *,
        language: str,
        chunk_length_seconds: int = 30,
        stride_length_seconds: int = 5,
    ) -> list[SubtitleSegment]:
        """Transcribe an audio file and return timestamped chunks."""
        result = self._pipeline(
            str(audio_path),
            chunk_length_s=chunk_length_seconds,
            stride_length_s=stride_length_seconds,
            return_timestamps=True,
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        return _chunks_to_segments(result, speaker="SPEAKER_00", offset=0.0)

    def transcribe_segment(
        self,
        audio_path: Path,
        segment: Segment,
        *,
        language: str,
        offset: float,
    ) -> list[SubtitleSegment]:
        """Transcribe an extracted segment and offset timestamps globally."""
        result = self._pipeline(
            str(audio_path),
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=True,
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        return _chunks_to_segments(result, speaker=segment.speaker, offset=offset)


def _chunks_to_segments(
    result: dict[str, Any],
    *,
    speaker: str,
    offset: float,
) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp", (None, None))
        if start is None or end is None:
            continue

        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        segments.append(
            SubtitleSegment(
                start=float(start) + offset,
                end=float(end) + offset,
                text=text,
                speaker=speaker,
            )
        )
    return segments
