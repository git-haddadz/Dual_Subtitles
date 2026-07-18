"""Speech-to-text integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dual_subtitles.models.subtitle import Segment, SubtitleSegment, TranscribedWord


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
        uses_cuda = torch.cuda.is_available() and device not in {-1, "cpu"}
        torch_dtype = torch.float16 if uses_cuda else torch.float32

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=device,
            torch_dtype=torch_dtype,
            model_kwargs={"low_cpu_mem_usage": True},
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
            return_timestamps="word",
            generate_kwargs={
                "language": language,
                "task": "transcribe",
                "num_beams": 5,
            },
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
        words = self.transcribe_window(
            audio_path,
            language=language,
            offset=offset,
        )
        return [
            SubtitleSegment(
                start=word.start,
                end=word.end,
                text=word.text,
                speaker=segment.speaker,
            )
            for word in words
        ]

    def transcribe_window(
        self,
        audio_path: Path,
        *,
        language: str,
        offset: float,
        num_beams: int = 5,
    ) -> list[TranscribedWord]:
        """Transcribe one continuous acoustic window into timestamped words."""
        result = self._pipeline(
            str(audio_path),
            chunk_length_s=30,
            stride_length_s=1,
            return_timestamps="word",
            generate_kwargs={
                "language": language,
                "task": "transcribe",
                "num_beams": num_beams,
            },
        )
        return _chunks_to_words(result, offset=offset)


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


def _chunks_to_words(
    result: dict[str, Any],
    *,
    offset: float,
) -> list[TranscribedWord]:
    """Convert Transformers word chunks to domain objects."""
    words: list[TranscribedWord] = []
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp", (None, None))
        if start is None or end is None:
            continue
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        raw_confidence = chunk.get("confidence", chunk.get("score"))
        confidence = float(raw_confidence) if raw_confidence is not None else None
        words.append(
            TranscribedWord(
                start=float(start) + offset,
                end=float(end) + offset,
                text=text,
                confidence=confidence,
            )
        )
    return words
