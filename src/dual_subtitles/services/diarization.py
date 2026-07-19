"""Speaker diarization integration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dual_subtitles.models.subtitle import Segment


class MissingHuggingFaceTokenError(RuntimeError):
    """Raised when diarization is requested without a Hugging Face token."""


@dataclass(slots=True)
class PyannoteDiarizer:
    """Thin wrapper around pyannote speaker diarization."""

    model_name: str = "pyannote/speaker-diarization-community-1"
    token_env_var: str = "HUGGINGFACE_TOKEN"
    device: int | str | None = None
    _pipeline: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Load the pyannote pipeline after validating the token."""
        token = os.getenv(self.token_env_var)
        if not token:
            msg = (
                f"{self.token_env_var} is required when diarization is "
                "enabled. Set HF_TOKEN and HUGGINGFACE_TOKEN before running "
                "the pipeline."
            )
            raise MissingHuggingFaceTokenError(msg)

    def _load_pipeline(self) -> Any:
        """Load pyannote only when diarization actually starts."""
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            self.model_name,
            token=os.environ[self.token_env_var],
        )
        target_device = self._resolve_device(torch)
        if target_device.type == "cuda":
            pipeline.to(target_device)
        self._pipeline = pipeline
        return pipeline

    def _resolve_device(self, torch: Any) -> Any:
        """Resolve the configured pyannote execution device."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.device, int):
            if self.device < 0:
                return torch.device("cpu")
            return torch.device(f"cuda:{self.device}")
        return torch.device(self.device)

    def detect(self, audio_path: Path) -> list[Segment]:
        """Detect speaker turns in an audio file."""
        import soundfile as sf
        import torch

        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(audio.T.copy())
        result = self._load_pipeline()(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }
        )
        annotation = getattr(result, "speaker_diarization", result)
        segments: list[Segment] = []
        if hasattr(annotation, "itertracks"):
            tracks = (
                (turn, speaker)
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            )
        else:
            tracks = iter(annotation)
        for turn, speaker in tracks:
            segments.append(
                Segment(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=str(speaker),
                )
            )
        return segments


class SingleSpeakerDiarizer:
    """Fallback diarizer for workflows that skip pyannote."""

    def detect_duration(self, duration_seconds: float) -> list[Segment]:
        """Return a single segment covering the whole audio duration."""
        return [Segment(start=0.0, end=duration_seconds, speaker="SPEAKER_00")]
