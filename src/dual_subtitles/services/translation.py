"""Translation helpers for ASS interlinear subtitles."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from dual_subtitles.core.segmentation import deduplicate_overlap
from dual_subtitles.models.subtitle import WordPair

LOGGER = logging.getLogger(__name__)


class TranslatorClient(Protocol):
    """Small protocol implemented by translation clients."""

    def translate(self, text: str) -> str:
        """Translate text and return the translated string."""


@dataclass(slots=True)
class InterlinearGoogleTranslator:
    """Build word-by-word interlinear ASS text using deep-translator."""

    source_language: str = "ar"
    target_language: str = "en"
    cache: dict[str, str] = field(default_factory=dict)
    client: TranslatorClient | None = None

    def __post_init__(self) -> None:
        """Create the Google translator lazily unless a client is injected."""
        if self.client is not None:
            return

        from deep_translator import GoogleTranslator

        self.client = GoogleTranslator(
            source=self.source_language,
            target=self.target_language,
        )

    def interlinear(self, text: str) -> str:
        """Return source text above literal translations in visual RTL order."""
        pairs = self.word_pairs(text)
        source = " ".join(pair.source for pair in pairs)
        translations = " ".join(pair.translation for pair in reversed(pairs))
        return source + r"\N" + translations

    def word_pairs(self, text: str) -> list[WordPair]:
        """Translate every source word independently and preserve pairing."""
        source_words = text.split()
        return [
            WordPair(source=word, translation=self._translate_word(word))
            for word in source_words
        ]

    def _translate_word(self, word: str) -> str:
        if word not in self.cache:
            self.cache[word] = self._safe_translate(word) or word
        return self.cache[word]

    def _safe_translate(self, text: str) -> str:
        if self.client is None:
            return ""

        try:
            return self.client.translate(text)
        except Exception as exc:  # noqa: BLE001 - third-party clients vary.
            LOGGER.warning("Translation failed for %r: %s", text, exc)
            return ""


__all__ = ["InterlinearGoogleTranslator", "TranslatorClient", "deduplicate_overlap"]
