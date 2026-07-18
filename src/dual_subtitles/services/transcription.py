"""Local Arabic speech-to-text integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from dual_subtitles.models.subtitle import Segment, SubtitleSegment

LOGGER = logging.getLogger(__name__)


class SpeechTranscriber(Protocol):
    """Contract for a recognizer applied to one speaker/phrase unit."""

    def transcribe_segment(
        self,
        audio_path: Path,
        segment: Segment,
        *,
        language: str,
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
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
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
