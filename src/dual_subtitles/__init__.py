"""Tools for generating dual-language subtitles from videos."""

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.models.subtitle import (
    Segment,
    SubtitleSegment,
    SuspiciousPassage,
    TranscribedWord,
    WordPair,
)

__all__ = [
    "ProcessingConfig",
    "Segment",
    "SubtitleSegment",
    "SuspiciousPassage",
    "TranscribedWord",
    "WordPair",
]
