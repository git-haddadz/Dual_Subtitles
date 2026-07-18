"""Speech-to-text integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from dual_subtitles.models.subtitle import Segment, SubtitleSegment, TranscribedWord

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class WhisperTranscriber:
    """Thin wrapper around the Transformers Whisper pipeline."""

    model_name: str = "openai/whisper-large-v3"
    device: int | str | None = None
    _pipeline: Any = field(init=False, repr=False)
    _safe_word_timestamp_beams: int | None = field(
        init=False,
        repr=False,
        default=None,
    )
    _word_timestamps_supported: bool = field(
        init=False,
        repr=False,
        default=True,
    )

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
            model_kwargs={
                "low_cpu_mem_usage": True,
                "attn_implementation": "eager",
            },
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
        if not self._word_timestamps_supported:
            result = self._transcribe_with_segment_timestamps(
                audio_path,
                language=language,
                num_beams=1,
            )
            return _segment_chunks_to_words(result, offset=offset)

        effective_beams = self._safe_word_timestamp_beams or num_beams
        try:
            result = self._transcribe_with_word_timestamps(
                audio_path,
                language=language,
                num_beams=effective_beams,
            )
        except IndexError:
            result = self._retry_inconsistent_word_timestamps(
                audio_path,
                language=language,
                failed_beams=effective_beams,
            )
            if not self._word_timestamps_supported:
                return _segment_chunks_to_words(result, offset=offset)
        return _chunks_to_words(result, offset=offset)

    def _retry_inconsistent_word_timestamps(
        self,
        audio_path: Path,
        *,
        language: str,
        failed_beams: int,
    ) -> dict[str, Any]:
        """Retry broken word alignment, then use stable segment timestamps."""
        if failed_beams > 1:
            LOGGER.warning(
                "Whisper returned inconsistent word timestamps with %s beams; "
                "retrying with one beam",
                failed_beams,
            )
            self._safe_word_timestamp_beams = 1
            try:
                return self._transcribe_with_word_timestamps(
                    audio_path,
                    language=language,
                    num_beams=1,
                )
            except IndexError:
                pass

        LOGGER.warning(
            "Word timestamps remain inconsistent; using segment timestamps "
            "for this and subsequent windows"
        )
        self._safe_word_timestamp_beams = 1
        self._word_timestamps_supported = False
        return self._transcribe_with_segment_timestamps(
            audio_path,
            language=language,
            num_beams=1,
        )

    def _transcribe_with_word_timestamps(
        self,
        audio_path: Path,
        *,
        language: str,
        num_beams: int,
    ) -> dict[str, Any]:
        """Run one already-windowed audio file through Whisper."""
        return cast(
            dict[str, Any],
            self._pipeline(
                str(audio_path),
                return_timestamps="word",
                generate_kwargs={
                    "language": language,
                    "task": "transcribe",
                    "num_beams": num_beams,
                },
            ),
        )

    def _transcribe_with_segment_timestamps(
        self,
        audio_path: Path,
        *,
        language: str,
        num_beams: int,
    ) -> dict[str, Any]:
        """Use Whisper's more stable segment timestamps as a final fallback."""
        return cast(
            dict[str, Any],
            self._pipeline(
                str(audio_path),
                return_timestamps=True,
                generate_kwargs={
                    "language": language,
                    "task": "transcribe",
                    "num_beams": num_beams,
                },
            ),
        )


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


def _segment_chunks_to_words(
    result: dict[str, Any],
    *,
    offset: float,
) -> list[TranscribedWord]:
    """Approximate word timing inside stable segment-level timestamps."""
    words: list[TranscribedWord] = []
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp", (None, None))
        tokens = str(chunk.get("text", "")).split()
        if start is None or end is None or not tokens:
            continue
        start_value = float(start)
        end_value = float(end)
        token_duration = (end_value - start_value) / len(tokens)
        if token_duration <= 0:
            continue
        for index, token in enumerate(tokens):
            token_start = start_value + index * token_duration
            words.append(
                TranscribedWord(
                    start=token_start + offset,
                    end=token_start + token_duration + offset,
                    text=token,
                )
            )
    return words
