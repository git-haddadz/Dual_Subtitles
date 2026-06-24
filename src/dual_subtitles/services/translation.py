"""Translation helpers for ASS interlinear subtitles."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

LOGGER = logging.getLogger(__name__)


class TranslatorClient(Protocol):
    """Small protocol implemented by translation clients."""

    def translate(self, text: str) -> str:
        """Translate text and return the translated string."""


@dataclass(slots=True)
class InterlinearGoogleTranslator:
    """Build word-aligned interlinear text using deep-translator."""

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
        """Return source text plus a translated second line for ASS."""
        words = text.split()
        translated_words = self._translate_words(text, words)
        return text + r"\N" + " ".join(translated_words)

    def _translate_words(self, text: str, words: list[str]) -> list[str]:
        full_translation = self._safe_translate(text)
        full_words = full_translation.split() if full_translation else []
        translated_words: list[str] = []

        for index, word in enumerate(words):
            if word in self.cache:
                translated_words.append(self.cache[word])
                continue

            if index < len(full_words):
                translated_word = full_words[index]
            else:
                translated_word = self._safe_translate(word) or word

            self.cache[word] = translated_word
            translated_words.append(translated_word)

        return translated_words

    def _safe_translate(self, text: str) -> str:
        if self.client is None:
            return ""

        try:
            return self.client.translate(text)
        except Exception as exc:  # noqa: BLE001 - third-party clients vary.
            LOGGER.warning("Translation failed for %r: %s", text, exc)
            return ""


def deduplicate_overlap(previous: str, new: str, *, max_overlap: int = 5) -> str:
    """Remove duplicated word overlap between two subtitle fragments."""
    previous_words = previous.split()
    new_words = new.split()
    overlap = min(len(previous_words), len(new_words), max_overlap)
    for size in range(overlap, 0, -1):
        if previous_words[-size:] == new_words[:size]:
            return " ".join(new_words[size:])
    return " ".join(new_words)
