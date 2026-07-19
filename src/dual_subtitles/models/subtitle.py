"""Subtitle domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Segment:
    """A speech segment identified in an audio file."""

    start: float
    end: float
    speaker: str = "SPEAKER_00"

    @property
    def duration(self) -> float:
        """Return the segment duration in seconds."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class SubtitleSegment:
    """A transcript segment ready to be rendered as subtitles."""

    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    """Acoustic profile inferred from several clean turns of one speaker."""

    speaker: str
    perceived_voice_gender: str
    confidence: float
    median_f0_hz: float | None
    analyzed_duration: float
    analyzed_segments: int
    method: str = "librosa-pyin"


@dataclass(frozen=True, slots=True)
class TranscribedWord:
    """A word emitted by the speech recognizer with global timestamps."""

    start: float
    end: float
    text: str
    confidence: float | None = None
    speaker: str = "SPEAKER_00"
    average_log_probability: float | None = None
    compression_ratio: float | None = None
    no_speech_probability: float | None = None
    window_index: int | None = None
    acoustic_start: float | None = None
    acoustic_end: float | None = None


@dataclass(frozen=True, slots=True)
class SuspiciousPassage:
    """A transcript interval that may benefit from a targeted retry."""

    start: float
    end: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WordPair:
    """A source word and its literal translated gloss."""

    source: str
    translation: str


class InterlinearTranslator(Protocol):
    """Protocol for objects that build ASS interlinear text."""

    def interlinear(self, text: str) -> str:
        """Return subtitle text with translated text on a second ASS line."""

    def word_pairs(self, text: str) -> list[WordPair]:
        """Return source words paired with their literal translations."""
