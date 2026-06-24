"""Configuration objects for subtitle generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Runtime configuration for the video-to-subtitles pipeline."""

    input_dir: Path
    output_dir: Path
    temp_dir: Path
    transcription_language: str = "ar"
    translation_source_language: str = "ar"
    translation_target_language: str = "en"
    whisper_model: str = "openai/whisper-large-v3"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    min_speech_duration: float = 0.7
    merge_gap: float = 0.6
    max_speech_duration: float = 10.0
    subtitle_gap_threshold: float = 0.25
    max_subtitle_duration: float = 2.5
    transcription_padding: float = 0.3
    max_words_per_subtitle: int = 8
    line_break_words: int = 6
    generate_srt: bool = True
    generate_ass: bool = True
    skip_existing: bool = True
    use_diarization: bool = True
    video_extension: str = ".mp4"
    device: int | str | None = None

    def normalized_extension(self) -> str:
        """Return the configured extension with a leading dot."""
        if self.video_extension.startswith("."):
            return self.video_extension.lower()
        return f".{self.video_extension.lower()}"
