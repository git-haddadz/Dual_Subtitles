"""Local translation helpers for natural anchors and lexical candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from dual_subtitles.core.segmentation import deduplicate_overlap
from dual_subtitles.models.subtitle import WordPair

LOGGER = logging.getLogger(__name__)


class TranslatorClient(Protocol):
    """Small compatibility protocol implemented by translation clients."""

    def translate(self, text: str) -> str:
        """Translate text and return the translated string."""


class NaturalTranslator(Protocol):
    """Protocol for an offline sentence translator."""

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate texts in order."""

    def translate_words(self, words: list[str]) -> list[str]:
        """Translate isolated lexical candidates with bounded decoding."""


@dataclass(slots=True)
class LocalMarianTranslator:
    """Lazy local Marian translator backed by Hugging Face Transformers."""

    model_name: str = "Helsinki-NLP/opus-mt-ar-en"
    device: int | str | None = None
    batch_size: int = 16
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _torch_device: str = field(default="cpu", init=False, repr=False)
    _sentence_cache: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _word_cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.device is None:
            self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(self.device, int):
            self._torch_device = f"cuda:{self.device}"
        else:
            self._torch_device = str(self.device)
        LOGGER.info("Loading local translation model %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(self._torch_device)
        self._model.eval()

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate complete sentences, caching exact source strings."""
        return self._translate(
            texts,
            cache=self._sentence_cache,
            max_new_tokens=256,
        )

    def translate_words(self, words: list[str]) -> list[str]:
        """Translate isolated words without allowing runaway generation."""
        return self._translate(
            words,
            cache=self._word_cache,
            max_new_tokens=16,
            lexical=True,
        )

    def _translate(
        self,
        texts: list[str],
        *,
        cache: dict[str, str],
        max_new_tokens: int,
        lexical: bool = False,
    ) -> list[str]:
        """Run one bounded generation mode and update its dedicated cache."""
        if not texts:
            return []
        missing = list(dict.fromkeys(text for text in texts if text not in cache))
        if missing:
            self._load()
            import torch

            assert self._model is not None
            assert self._tokenizer is not None
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self._torch_device)
                with torch.inference_mode():
                    generation_options: dict[str, Any] = {
                        "max_new_tokens": max_new_tokens,
                        "num_beams": 4,
                    }
                    if lexical:
                        generation_options.update(
                            no_repeat_ngram_size=2,
                            repetition_penalty=1.2,
                            early_stopping=True,
                        )
                    generated = self._model.generate(**encoded, **generation_options)
                decoded = self._tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
                cache.update(zip(batch, decoded, strict=True))
        return [cache.get(text, text) or text for text in texts]

    def release(self) -> None:
        """Release model memory before another large GPU stage starts."""
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return


@dataclass(slots=True)
class InterlinearGoogleTranslator:
    """Deprecated injected-client adapter retained for API compatibility."""

    source_language: str = "ar"
    target_language: str = "en"
    cache: dict[str, str] = field(default_factory=dict)
    client: TranslatorClient | None = None

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
    "LocalMarianTranslator",
    "NaturalTranslator",
    "TranslatorClient",
    "deduplicate_overlap",
]
