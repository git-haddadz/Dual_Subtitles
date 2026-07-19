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
MAX_CHARACTERS_PER_SECOND = 40
MAX_SINGLE_TOKEN_CHARACTERS = 48
MIN_GENERATION_TOKENS = 48
GENERATION_TOKENS_PER_SECOND = 24
RETRY_TOKEN_MULTIPLIER = 1.5


class SpeechTranscriber(Protocol):
    """Contract for a recognizer applied to one speaker/phrase unit."""

    def transcribe_segment(
        self,
        audio_path: Path,
        segment: Segment,
        *,
        language: str,
        retry: bool = False,
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
            token_limit = _generation_token_limit(
                segment.duration,
                configured_limit=self.max_new_tokens,
                retry=retry,
            )
            generation_options: dict[str, Any] = {
                "max_new_tokens": token_limit,
            }
            if retry:
                generation_options.update(
                    {
                        "no_repeat_ngram_size": 4,
                        "repetition_penalty": 1.15,
                    }
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


def suspicious_transcript_reasons(text: str, *, duration: float) -> tuple[str, ...]:
    """Return generic reasons why an ASR result must not reach the SRT."""
    stripped = text.strip()
    reasons: list[str] = []
    if not stripped:
        return ("empty",)
    if "\ufffd" in stripped:
        reasons.append("replacement-character")
    if any(unicodedata.category(character) == "Cs" for character in stripped):
        reasons.append("invalid-unicode-surrogate")
    if re.search(rf"(.)\1{{{MAX_REPEATED_CHARACTER_RUN - 1},}}", stripped):
        reasons.append("repeated-character-run")
    if any(len(token) > MAX_SINGLE_TOKEN_CHARACTERS for token in stripped.split()):
        reasons.append("oversized-token")
    if _has_orphan_final_arabic_letter(stripped):
        reasons.append("orphan-final-arabic-letter")
    maximum_characters = max(
        MAX_SINGLE_TOKEN_CHARACTERS,
        math.ceil(max(duration, 0.2) * MAX_CHARACTERS_PER_SECOND),
    )
    if len(stripped) > maximum_characters:
        reasons.append("text-too-long-for-audio")
    return tuple(reasons)


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


def _generation_token_limit(
    duration: float,
    *,
    configured_limit: int,
    retry: bool,
) -> int:
    """Bound generation by acoustic duration to prevent runaway decoding."""
    estimated = max(
        MIN_GENERATION_TOKENS,
        math.ceil(max(duration, 0.2) * GENERATION_TOKENS_PER_SECOND),
    )
    if retry:
        estimated = math.ceil(estimated * RETRY_TOKEN_MULTIPLIER)
    return min(configured_limit, estimated)
