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


class InterlinearTranslator(Protocol):
    """Protocol for objects that build ASS interlinear text."""

    def interlinear(self, text: str) -> str:
        """Return subtitle text with translated text on a second ASS line."""
