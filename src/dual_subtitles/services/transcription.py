"""Local Arabic speech-to-text integration."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from dual_subtitles.models.subtitle import Segment, SubtitleSegment

LOGGER = logging.getLogger(__name__)
MAX_REPEATED_CHARACTER_RUN = 12
MAX_REPEATED_TOKEN_RUN = 6
COLLAPSED_REPEATED_TOKEN_RUN = 3
MAX_CHARACTERS_PER_SECOND = 40
MAX_WORDS_PER_SECOND = 8
MIN_WORD_ALLOWANCE = 4
MAX_SINGLE_TOKEN_CHARACTERS = 48
NON_RETRYABLE_TRANSCRIPT_REASONS = frozenset(
    {
        "non-speech-marker",
        "unsupported-script",
    }
)
REPETITION_RETRY_REASONS = frozenset(
    {
        "repeated-character-run",
        "repeated-token-run",
    }
)
NON_SPEECH_MARKER_PATTERN = re.compile(r"@@@")


class SpeechTranscriber(Protocol):
    """Contract for a recognizer applied to one speaker/phrase unit."""

    def transcribe_segment(
        self,
        audio_path: Path,
        segment: Segment,
        *,
        language: str,
        retry: bool = False,
        retry_reasons: tuple[str, ...] = (),
    ) -> SubtitleSegment | None:
        """Transcribe one already-isolated speaker/phrase unit."""
        ...


def create_transcriber(
    *,
    model_name: str,
    device: int | str | None,
) -> SpeechTranscriber:
    """Create the branch's single local Arabic recognizer."""
    return CohereArabicTranscriber(model_name=model_name, device=device)


@dataclass(slots=True)
class CohereArabicTranscriber:
    """Run Cohere Transcribe Arabic without a competing ASR or aligner."""

    model_name: str = "CohereLabs/cohere-transcribe-arabic-07-2026"
    device: int | str | None = None
    max_new_tokens: int = 256
    _model: Any = field(init=False, repr=False, default=None)
    _processor: Any = field(init=False, repr=False, default=None)
    _torch_device: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the device while keeping model loading lazy."""
        self._torch_device = _resolve_torch_device(self.device)

    def _load_model(self) -> None:
        """Load Cohere only after diarization has produced phrase units."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        dtype = torch.float16 if self._torch_device.type == "cuda" else torch.float32
        LOGGER.info("Loading Cohere Arabic recognizer %s", self.model_name)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self._torch_device)
        self._model.eval()

    def transcribe_segment(
        self,
        audio_path: Path,
        segment: Segment,
        *,
        language: str,
        retry: bool = False,
        retry_reasons: tuple[str, ...] = (),
    ) -> SubtitleSegment | None:
        """Transcribe one unit and retain its acoustic boundaries unchanged."""
        import soundfile as sf
        import torch

        self._load_model()
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            return None
        inputs = self._processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            language=language,
        )
        inputs = inputs.to(self._torch_device, dtype=self._model.dtype)
        with torch.inference_mode():
            generation_options = _generation_options(
                max_new_tokens=self.max_new_tokens,
                retry_reasons=retry_reasons if retry else (),
            )
            output_ids = self._model.generate(
                **inputs,
                **generation_options,
            )
        decoded = self._processor.decode(
            output_ids,
            skip_special_tokens=True,
            language=language,
        )
        text = _decoded_text(decoded)
        if not text:
            return None
        return SubtitleSegment(
            start=segment.start,
            end=segment.end,
            text=text,
            speaker=segment.speaker,
        )


def _resolve_torch_device(device: int | str | None) -> Any:
    import torch

    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int):
        return torch.device("cpu" if device < 0 else f"cuda:{device}")
    return torch.device(device)


def _decoded_text(decoded: Any) -> str:
    """Normalize processor return shapes without modifying their text."""
    if isinstance(decoded, str):
        return decoded.strip()
    if isinstance(decoded, list):
        return str(decoded[0]).strip() if decoded else ""
    return str(decoded).strip() if decoded is not None else ""


def suspicious_transcript_reasons(
    text: str,
    *,
    duration: float,
) -> tuple[str, ...]:
    """Return generic reasons why an ASR result must not reach the SRT."""
    stripped = text.strip()
    reasons: list[str] = []
    if not stripped:
        return ("empty",)
    if NON_SPEECH_MARKER_PATTERN.search(stripped) or not _has_lexical_content(stripped):
        reasons.append("non-speech-marker")
    if "\ufffd" in stripped:
        reasons.append("replacement-character")
    if any(unicodedata.category(character) == "Cs" for character in stripped):
        reasons.append("invalid-unicode-surrogate")
    if _has_unsupported_script(stripped):
        reasons.append("unsupported-script")
    if re.search(rf"(.)\1{{{MAX_REPEATED_CHARACTER_RUN - 1},}}", stripped):
        reasons.append("repeated-character-run")
    if _has_repeated_token_run(stripped):
        reasons.append("repeated-token-run")
    if any(len(token) > MAX_SINGLE_TOKEN_CHARACTERS for token in stripped.split()):
        reasons.append("oversized-token")
    if _has_orphan_final_arabic_letter(stripped):
        reasons.append("orphan-final-arabic-letter")
    word_count = _spoken_word_count(stripped)
    maximum_words = max(
        MIN_WORD_ALLOWANCE,
        math.ceil(max(duration, 0.2) * MAX_WORDS_PER_SECOND),
    )
    if word_count > maximum_words:
        reasons.append("too-many-words-for-audio")
    maximum_characters = max(
        MAX_SINGLE_TOKEN_CHARACTERS,
        math.ceil(max(duration, 0.2) * MAX_CHARACTERS_PER_SECOND),
    )
    if len(stripped) > maximum_characters:
        reasons.append("text-too-long-for-audio")
    return tuple(reasons)


def should_retry_transcript(reasons: tuple[str, ...]) -> bool:
    """Return whether another Arabic decoding pass can plausibly help."""
    return not NON_RETRYABLE_TRANSCRIPT_REASONS.intersection(reasons)


def _has_unsupported_script(text: str) -> bool:
    """Reject letters outside the Arabic/Latin languages supported by the model."""
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        name = unicodedata.name(character, "")
        if "ARABIC" not in name and "LATIN" not in name:
            return True
    return False


def _has_lexical_content(text: str) -> bool:
    """Return whether the transcript contains at least one letter or number."""
    return any(unicodedata.category(character)[0] in {"L", "N"} for character in text)


def _has_repeated_token_run(text: str) -> bool:
    """Detect consecutive repeated words missed by character-run checks."""
    run_length = 0
    previous = ""
    for raw_token in text.split():
        token = _normalized_repetition_token(raw_token)
        if not token:
            previous = ""
            run_length = 0
            continue
        if token == previous:
            run_length += 1
        else:
            previous = token
            run_length = 1
        if run_length >= MAX_REPEATED_TOKEN_RUN:
            return True
    return False


def collapse_repeated_token_runs(
    text: str,
    *,
    trigger: int = MAX_REPEATED_TOKEN_RUN,
    keep: int = COLLAPSED_REPEATED_TOKEN_RUN,
) -> str:
    """Compress only pathological consecutive word runs after a failed retry."""
    if trigger <= keep or keep <= 0:
        msg = "Repeated-token collapse requires 0 < keep < trigger."
        raise ValueError(msg)

    tokens = text.split()
    collapsed: list[str] = []
    changed = False
    cursor = 0
    while cursor < len(tokens):
        normalized = _normalized_repetition_token(tokens[cursor])
        run_end = cursor + 1
        while (
            normalized
            and run_end < len(tokens)
            and _normalized_repetition_token(tokens[run_end]) == normalized
        ):
            run_end += 1

        run = tokens[cursor:run_end]
        if normalized and len(run) >= trigger:
            collapsed.extend([*run[: keep - 1], run[-1]])
            changed = True
        else:
            collapsed.extend(run)
        cursor = run_end
    return " ".join(collapsed) if changed else text


def _normalized_repetition_token(token: str) -> str:
    """Remove punctuation and diacritics when comparing repeated words."""
    return "".join(
        character.casefold()
        for character in token
        if unicodedata.category(character)[0] not in {"M", "P", "S"}
    )


def _spoken_word_count(text: str) -> int:
    """Count tokens containing at least one letter."""
    return sum(
        any(unicodedata.category(character).startswith("L") for character in token)
        for token in text.split()
    )


def _has_orphan_final_arabic_letter(text: str) -> bool:
    """Detect a likely mid-word cutoff at the end of an Arabic transcript."""
    if unicodedata.category(text[-1]).startswith("P"):
        return False
    final_token = text.rsplit(maxsplit=1)[-1]
    letters = [
        character
        for character in final_token
        if unicodedata.category(character).startswith("L")
    ]
    return (
        len(letters) == 1
        and "ARABIC" in unicodedata.name(letters[0], "")
        and all(
            character == letters[0] or unicodedata.category(character).startswith("M")
            for character in final_token
        )
    )


def _generation_options(
    *,
    max_new_tokens: int,
    retry_reasons: tuple[str, ...],
) -> dict[str, Any]:
    """Build the documented fixed budget and targeted retry constraints."""
    options: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if REPETITION_RETRY_REASONS.intersection(retry_reasons):
        options.update(
            {
                "no_repeat_ngram_size": 4,
                "repetition_penalty": 1.15,
                "renormalize_logits": True,
            }
        )
    return options
