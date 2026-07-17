"""Subtitle and linguistic annotation domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
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


class TokenKind(StrEnum):
    """Kinds emitted by the lossless tokenizer."""

    WORD = "word"
    PUNCTUATION = "punctuation"


class SpanKind(StrEnum):
    """Pedagogical display grouping."""

    TOKEN = "token"
    ENTITY = "entity"
    EXPRESSION = "expression"


class EntityType(StrEnum):
    """Named-entity categories supported by the Arabic NER backend."""

    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    MISC = "misc"


@dataclass(frozen=True, slots=True)
class ConfidenceBreakdown:
    """Independent confidence signals; missing models leave a value at zero."""

    morphology: float = 0.0
    diacritization: float = 0.0
    ner: float = 0.0
    alignment: float = 0.0
    gloss: float = 0.0
    overall: float = 0.0


@dataclass(slots=True)
class AnnotatedToken:
    """One exact token from a subtitle plus its linguistic annotations."""

    token_id: str
    subtitle_index: int
    token_index: int
    start_char: int
    end_char: int
    surface: str
    kind: TokenKind
    lemma: str | None = None
    root: str | None = None
    part_of_speech: str | None = None
    features: dict[str, str] = field(default_factory=dict)
    vocalized_surface: str | None = None
    entity_id: str | None = None
    entity_type: EntityType | None = None
    transliteration: str | None = None
    gloss: str | None = None
    target_indices: tuple[int, ...] = ()
    confidence: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)
    warnings: list[str] = field(default_factory=list)

    @property
    def display_surface(self) -> str:
        """Return a trusted vocalization or the exact transcript surface."""
        return self.vocalized_surface or self.surface


@dataclass(slots=True)
class AnnotatedSpan:
    """A contiguous token group displayed as one pedagogical pair."""

    span_id: str
    subtitle_index: int
    token_start: int
    token_end: int
    kind: SpanKind
    source_surface: str
    display_source: str
    gloss: str
    transliteration: str | None = None
    entity_id: str | None = None
    target_indices: tuple[int, ...] = ()
    confidence: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnnotatedSubtitle:
    """A subtitle enriched after the complete video transcript is available."""

    segment: SubtitleSegment
    subtitle_index: int
    tokens: list[AnnotatedToken]
    context_indices: tuple[int, ...] = ()
    natural_translation: str = ""
    target_tokens: list[str] = field(default_factory=list)
    alignments: dict[int, tuple[int, ...]] = field(default_factory=dict)
    spans: list[AnnotatedSpan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EntityRecord:
    """Canonical per-video representation of one detected entity."""

    entity_id: str
    entity_type: EntityType
    canonical_source: str
    canonical_target: str
    aliases: set[str] = field(default_factory=set)
    confidence: float = 0.0


@dataclass(slots=True)
class AnnotatedVideo:
    """Complete annotated transcript and its automatic entity memory."""

    subtitles: list[AnnotatedSubtitle]
    entities: dict[str, EntityRecord] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WordPair:
    """Compatibility view for callers using the former rendering contract."""

    source: str
    translation: str


class InterlinearTranslator(Protocol):
    """Compatibility protocol for the legacy ASS builder."""

    def interlinear(self, text: str) -> str:
        """Return subtitle text with translated text on a second ASS line."""

    def word_pairs(self, text: str) -> list[WordPair]:
        """Return source words paired with their literal translations."""
