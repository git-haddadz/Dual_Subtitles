"""Google-backed lexical translation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from dual_subtitles.core.segmentation import deduplicate_overlap
from dual_subtitles.models.subtitle import WordPair

LOGGER = logging.getLogger(__name__)


class TranslatorClient(Protocol):
    """Small compatibility protocol implemented by translation clients."""

    def translate(self, text: str) -> str:
        """Translate text and return the translated string."""


class LexicalTranslator(Protocol):
    """Protocol for the translator that supplies displayed word glosses."""

    def translate_words(self, words: list[str]) -> list[str]:
        """Translate isolated words in order."""


@dataclass(slots=True)
class InterlinearGoogleTranslator:
    """Google-backed word translator retained from the original pipeline."""

    source_language: str = "ar"
    target_language: str = "en"
    cache: dict[str, str] = field(default_factory=dict)
    client: TranslatorClient | None = None

    def __post_init__(self) -> None:
        """Create the account-free Google web translator lazily."""
        if self.client is not None:
            return
        from deep_translator import GoogleTranslator

        self.client = GoogleTranslator(
            source=self.source_language,
            target=self.target_language,
        )

    def translate_words(self, words: list[str]) -> list[str]:
        """Translate words independently with the same cache as the old path."""
        return [self._translate_word(word) for word in words]

    def interlinear(self, text: str) -> str:
        """Return the former two-line compatibility representation."""
        pairs = self.word_pairs(text)
        source = " ".join(pair.source for pair in pairs)
        translations = " ".join(pair.translation for pair in reversed(pairs))
        return source + r"\N" + translations

    def word_pairs(self, text: str) -> list[WordPair]:
        """Translate words through an explicitly injected legacy client."""
        return [
            WordPair(source=word, translation=self._translate_word(word))
            for word in text.split()
        ]

    def _translate_word(self, word: str) -> str:
        if word not in self.cache:
            self.cache[word] = self._safe_translate(word) or word
        return self.cache[word]

    def _safe_translate(self, text: str) -> str:
        if self.client is None:
            LOGGER.warning("No legacy translation client configured for %r", text)
            return ""
        try:
            return self.client.translate(text)
        except Exception as exc:  # noqa: BLE001 - compatibility clients vary.
            LOGGER.warning("Legacy translation failed for %r: %s", text, exc)
            return ""


__all__ = [
    "InterlinearGoogleTranslator",
    "LexicalTranslator",
    "TranslatorClient",
    "deduplicate_overlap",
]
