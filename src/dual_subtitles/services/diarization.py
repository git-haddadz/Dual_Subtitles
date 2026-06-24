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

    model_name: str = "pyannote/speaker-diarization-3.1"
    token_env_var: str = "HUGGINGFACE_TOKEN"
    _pipeline: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load the pyannote pipeline after validating the token."""
        token = os.getenv(self.token_env_var)
        if not token:
            msg = (
                f"{self.token_env_var} is required when diarization is "
                "enabled. Revoke any token committed in notebooks and set a "
                "fresh token in the environment."
            )
            raise MissingHuggingFaceTokenError(msg)

        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(
            self.model_name,
            use_auth_token=token,
        )

    def detect(self, audio_path: Path) -> list[Segment]:
        """Detect speaker turns in an audio file."""
        result = self._pipeline(str(audio_path))
        segments: list[Segment] = []
        for turn, _, speaker in result.itertracks(yield_label=True):
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
