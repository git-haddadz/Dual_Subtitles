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
    transcription_backend: str = "faster-whisper"
    faster_whisper_compute_type: str | None = None
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    min_speech_duration: float = 0.7
    merge_gap: float = 0.6
    max_speech_duration: float = 10.0
    subtitle_gap_threshold: float = 0.25
    min_subtitle_duration: float = 0.7
    max_subtitle_duration: float = 5.0
    transcription_padding: float = 0.3
    transcription_window_duration: float = 28.0
    transcription_max_window_duration: float = 30.0
    transcription_overlap: float = 1.0
    transcription_num_beams: int = 5
    transcription_retry_num_beams: int = 8
    transcription_context_words: int = 24
    transcription_context_reset_pause: float = 8.0
    silence_min_duration: float = 0.4
    silence_threshold_offset: float = 16.0
    suspicious_confidence_threshold: float = 0.45
    suspicious_log_probability_threshold: float = -1.0
    suspicious_compression_ratio_threshold: float = 2.4
    suspicious_no_speech_threshold: float = 0.6
    max_word_duration: float = 3.0
    retry_context: float = 8.0
    retry_min_improvement: float = 0.12
    enable_targeted_retry: bool = True
    max_words_per_subtitle: int = 8
    line_break_words: int = 6
    generate_srt: bool = True
    generate_ass: bool = True
    skip_existing: bool = True
    use_diarization: bool = True
    video_extension: str = ".mp4"
    device: int | str | None = None

    def __post_init__(self) -> None:  # noqa: PLR0912 - validates independent fields.
        """Reject configurations that cannot produce valid subtitles."""
        if not self.generate_srt and not self.generate_ass:
            msg = "At least one of SRT or ASS generation must be enabled."
            raise ValueError(msg)

        positive_values = {
            "min_speech_duration": self.min_speech_duration,
            "max_speech_duration": self.max_speech_duration,
            "min_subtitle_duration": self.min_subtitle_duration,
            "max_subtitle_duration": self.max_subtitle_duration,
            "max_words_per_subtitle": self.max_words_per_subtitle,
            "line_break_words": self.line_break_words,
            "transcription_window_duration": (self.transcription_window_duration),
            "transcription_max_window_duration": (
                self.transcription_max_window_duration
            ),
            "transcription_num_beams": self.transcription_num_beams,
            "transcription_retry_num_beams": self.transcription_retry_num_beams,
            "transcription_context_words": self.transcription_context_words,
            "transcription_context_reset_pause": (
                self.transcription_context_reset_pause
            ),
            "silence_min_duration": self.silence_min_duration,
            "suspicious_compression_ratio_threshold": (
                self.suspicious_compression_ratio_threshold
            ),
            "max_word_duration": self.max_word_duration,
        }
        for name, value in positive_values.items():
            if value <= 0:
                msg = f"{name} must be greater than zero."
                raise ValueError(msg)

        non_negative_values = {
            "merge_gap": self.merge_gap,
            "subtitle_gap_threshold": self.subtitle_gap_threshold,
            "transcription_padding": self.transcription_padding,
            "transcription_overlap": self.transcription_overlap,
            "silence_threshold_offset": self.silence_threshold_offset,
            "suspicious_confidence_threshold": (self.suspicious_confidence_threshold),
            "suspicious_no_speech_threshold": self.suspicious_no_speech_threshold,
            "retry_context": self.retry_context,
            "retry_min_improvement": self.retry_min_improvement,
        }
        for name, value in non_negative_values.items():
            if value < 0:
                msg = f"{name} must be zero or greater."
                raise ValueError(msg)

        if self.transcription_window_duration > self.transcription_max_window_duration:
            msg = (
                "transcription_window_duration must not exceed "
                "transcription_max_window_duration."
            )
            raise ValueError(msg)
        if self.transcription_overlap >= self.transcription_window_duration:
            msg = "transcription_overlap must be shorter than the window."
            raise ValueError(msg)
        if self.min_subtitle_duration > self.max_subtitle_duration:
            msg = "min_subtitle_duration must not exceed max_subtitle_duration."
            raise ValueError(msg)
        if self.transcription_backend not in {"faster-whisper", "transformers"}:
            msg = "transcription_backend must be 'faster-whisper' or 'transformers'."
            raise ValueError(msg)
        for name in (
            "suspicious_confidence_threshold",
            "suspicious_no_speech_threshold",
            "retry_min_improvement",
        ):
            value = getattr(self, name)
            if value > 1:
                msg = f"{name} must not exceed one."
                raise ValueError(msg)

        if self.suspicious_log_probability_threshold > 0:
            msg = "suspicious_log_probability_threshold must be zero or lower."
            raise ValueError(msg)

        if not self.normalized_extension().strip("."):
            msg = "video_extension must contain a file extension."
            raise ValueError(msg)

    def normalized_extension(self) -> str:
        """Return the configured extension with a leading dot."""
        if self.video_extension.startswith("."):
            return self.video_extension.lower()
        return f".{self.video_extension.lower()}"
