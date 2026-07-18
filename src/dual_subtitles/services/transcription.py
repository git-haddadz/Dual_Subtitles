"""Speech-to-text integration."""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from dual_subtitles.models.subtitle import Segment, SubtitleSegment, TranscribedWord

LOGGER = logging.getLogger(__name__)


class SpeechTranscriber(Protocol):
    """Common contract implemented by local Whisper backends."""

    def transcribe_window(  # noqa: PLR0913
        self,
        audio_path: Path,
        *,
        language: str,
        offset: float,
        num_beams: int = 5,
        prompt: str | None = None,
        is_retry: bool = False,
    ) -> list[TranscribedWord]:
        """Transcribe one continuous audio window."""
        ...


def create_transcriber(
    backend: str,
    *,
    model_name: str,
    device: int | str | None,
    compute_type: str | None = None,
) -> SpeechTranscriber:
    """Create the configured local speech-recognition backend."""
    if backend == "faster-whisper":
        return FasterWhisperTranscriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )
    if backend == "transformers":
        return WhisperTranscriber(model_name=model_name, device=device)
    msg = f"Unsupported transcription backend: {backend}"
    raise ValueError(msg)


@dataclass(slots=True)
class FasterWhisperTranscriber:
    """CTranslate2 Whisper backend with beam search and word diagnostics."""

    model_name: str = "openai/whisper-large-v3"
    device: int | str | None = None
    compute_type: str | None = None
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load the converted Whisper model on the selected device."""
        target_device, device_index = _resolve_faster_whisper_device(self.device)
        if target_device == "cuda":
            _preload_nvidia_libraries()

        from faster_whisper import WhisperModel

        compute_type = self.compute_type
        if compute_type is None:
            compute_type = "float16" if target_device == "cuda" else "int8"
        LOGGER.info(
            "Loading faster-whisper on %s:%s with %s",
            target_device,
            device_index,
            compute_type,
        )
        self._model = WhisperModel(
            _faster_whisper_model_name(self.model_name),
            device=target_device,
            device_index=device_index,
            compute_type=compute_type,
        )

    def transcribe_window(  # noqa: PLR0913
        self,
        audio_path: Path,
        *,
        language: str,
        offset: float,
        num_beams: int = 5,
        prompt: str | None = None,
        is_retry: bool = False,
    ) -> list[TranscribedWord]:
        """Transcribe a window with stable word timestamps and scores."""
        temperatures: float | tuple[float, ...]
        temperatures = (0.0, 0.2, 0.4, 0.6) if is_retry else 0.0
        segments, _ = self._model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            beam_size=num_beams,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            temperature=temperatures,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            vad_filter=False,
        )
        return _faster_segments_to_words(segments, offset=offset)


@dataclass(slots=True)
class WhisperTranscriber:
    """Thin wrapper around the Transformers Whisper pipeline."""

    model_name: str = "openai/whisper-large-v3"
    device: int | str | None = None
    _pipeline: Any = field(init=False, repr=False)
    _safe_word_timestamp_beams: int | None = field(
        init=False,
        repr=False,
        default=None,
    )
    _word_timestamps_supported: bool = field(
        init=False,
        repr=False,
        default=True,
    )

    def __post_init__(self) -> None:
        """Load the Transformers pipeline and choose a default device."""
        import torch
        from transformers import pipeline

        device = self.device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        uses_cuda = torch.cuda.is_available() and device not in {-1, "cpu"}
        torch_dtype = torch.float16 if uses_cuda else torch.float32

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=device,
            torch_dtype=torch_dtype,
            model_kwargs={
                "low_cpu_mem_usage": True,
                "attn_implementation": "eager",
            },
            return_timestamps=True,
        )

    def transcribe_file(
        self,
        audio_path: Path,
        *,
        language: str,
        chunk_length_seconds: int = 30,
        stride_length_seconds: int = 5,
    ) -> list[SubtitleSegment]:
        """Transcribe an audio file and return timestamped chunks."""
        result = self._pipeline(
            str(audio_path),
            chunk_length_s=chunk_length_seconds,
            stride_length_s=stride_length_seconds,
            return_timestamps="word",
            generate_kwargs={
                "language": language,
                "task": "transcribe",
                "num_beams": 5,
            },
        )
        return _chunks_to_segments(result, speaker="SPEAKER_00", offset=0.0)

    def transcribe_segment(
        self,
        audio_path: Path,
        segment: Segment,
        *,
        language: str,
        offset: float,
    ) -> list[SubtitleSegment]:
        """Transcribe an extracted segment and offset timestamps globally."""
        words = self.transcribe_window(
            audio_path,
            language=language,
            offset=offset,
        )
        return [
            SubtitleSegment(
                start=word.start,
                end=word.end,
                text=word.text,
                speaker=segment.speaker,
            )
            for word in words
        ]

    def transcribe_window(  # noqa: PLR0913
        self,
        audio_path: Path,
        *,
        language: str,
        offset: float,
        num_beams: int = 5,
        prompt: str | None = None,
        is_retry: bool = False,
    ) -> list[TranscribedWord]:
        """Transcribe one continuous acoustic window into timestamped words."""
        del prompt, is_retry
        if not self._word_timestamps_supported:
            result = self._transcribe_with_segment_timestamps(
                audio_path,
                language=language,
                num_beams=1,
            )
            return _segment_chunks_to_words(result, offset=offset)

        effective_beams = self._safe_word_timestamp_beams or num_beams
        try:
            result = self._transcribe_with_word_timestamps(
                audio_path,
                language=language,
                num_beams=effective_beams,
            )
        except IndexError:
            result = self._retry_inconsistent_word_timestamps(
                audio_path,
                language=language,
                failed_beams=effective_beams,
            )
            if not self._word_timestamps_supported:
                return _segment_chunks_to_words(result, offset=offset)
        return _chunks_to_words(result, offset=offset)

    def _retry_inconsistent_word_timestamps(
        self,
        audio_path: Path,
        *,
        language: str,
        failed_beams: int,
    ) -> dict[str, Any]:
        """Retry broken word alignment, then use stable segment timestamps."""
        if failed_beams > 1:
            LOGGER.warning(
                "Whisper returned inconsistent word timestamps with %s beams; "
                "retrying with one beam",
                failed_beams,
            )
            self._safe_word_timestamp_beams = 1
            try:
                return self._transcribe_with_word_timestamps(
                    audio_path,
                    language=language,
                    num_beams=1,
                )
            except IndexError:
                pass

        LOGGER.warning(
            "Word timestamps remain inconsistent; using segment timestamps "
            "for this and subsequent windows"
        )
        self._safe_word_timestamp_beams = 1
        self._word_timestamps_supported = False
        return self._transcribe_with_segment_timestamps(
            audio_path,
            language=language,
            num_beams=1,
        )

    def _transcribe_with_word_timestamps(
        self,
        audio_path: Path,
        *,
        language: str,
        num_beams: int,
    ) -> dict[str, Any]:
        """Run one already-windowed audio file through Whisper."""
        return cast(
            dict[str, Any],
            self._pipeline(
                str(audio_path),
                return_timestamps="word",
                generate_kwargs={
                    "language": language,
                    "task": "transcribe",
                    "num_beams": num_beams,
                },
            ),
        )

    def _transcribe_with_segment_timestamps(
        self,
        audio_path: Path,
        *,
        language: str,
        num_beams: int,
    ) -> dict[str, Any]:
        """Use Whisper's more stable segment timestamps as a final fallback."""
        return cast(
            dict[str, Any],
            self._pipeline(
                str(audio_path),
                return_timestamps=True,
                generate_kwargs={
                    "language": language,
                    "task": "transcribe",
                    "num_beams": num_beams,
                },
            ),
        )


def _chunks_to_segments(
    result: dict[str, Any],
    *,
    speaker: str,
    offset: float,
) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp", (None, None))
        if start is None or end is None:
            continue

        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        segments.append(
            SubtitleSegment(
                start=float(start) + offset,
                end=float(end) + offset,
                text=text,
                speaker=speaker,
            )
        )
    return segments


def _chunks_to_words(
    result: dict[str, Any],
    *,
    offset: float,
) -> list[TranscribedWord]:
    """Convert Transformers word chunks to domain objects."""
    words: list[TranscribedWord] = []
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp", (None, None))
        if start is None or end is None:
            continue
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        raw_confidence = chunk.get("confidence", chunk.get("score"))
        confidence = float(raw_confidence) if raw_confidence is not None else None
        words.append(
            TranscribedWord(
                start=float(start) + offset,
                end=float(end) + offset,
                text=text,
                confidence=confidence,
            )
        )
    return words


def _segment_chunks_to_words(
    result: dict[str, Any],
    *,
    offset: float,
) -> list[TranscribedWord]:
    """Approximate word timing inside stable segment-level timestamps."""
    words: list[TranscribedWord] = []
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp", (None, None))
        tokens = str(chunk.get("text", "")).split()
        if start is None or end is None or not tokens:
            continue
        start_value = float(start)
        end_value = float(end)
        token_duration = (end_value - start_value) / len(tokens)
        if token_duration <= 0:
            continue
        for index, token in enumerate(tokens):
            token_start = start_value + index * token_duration
            words.append(
                TranscribedWord(
                    start=token_start + offset,
                    end=token_start + token_duration + offset,
                    text=token,
                )
            )
    return words


def _faster_segments_to_words(
    segments: Any,
    *,
    offset: float,
) -> list[TranscribedWord]:
    """Convert faster-whisper segments and diagnostics to domain words."""
    words: list[TranscribedWord] = []
    for segment in segments:
        average_log_probability = _optional_float(getattr(segment, "avg_logprob", None))
        compression_ratio = _optional_float(getattr(segment, "compression_ratio", None))
        no_speech_probability = _optional_float(
            getattr(segment, "no_speech_prob", None)
        )
        segment_words = getattr(segment, "words", None)
        if segment_words:
            for word in segment_words:
                text = str(getattr(word, "word", "")).strip()
                start = _optional_float(getattr(word, "start", None))
                end = _optional_float(getattr(word, "end", None))
                if not text or start is None or end is None or end <= start:
                    continue
                words.append(
                    TranscribedWord(
                        start=start + offset,
                        end=end + offset,
                        text=text,
                        confidence=_optional_float(getattr(word, "probability", None)),
                        average_log_probability=average_log_probability,
                        compression_ratio=compression_ratio,
                        no_speech_probability=no_speech_probability,
                    )
                )
            continue
        words.extend(
            _approximate_faster_segment_words(
                segment,
                offset=offset,
                average_log_probability=average_log_probability,
                compression_ratio=compression_ratio,
                no_speech_probability=no_speech_probability,
            )
        )
    return words


def _approximate_faster_segment_words(
    segment: Any,
    *,
    offset: float,
    average_log_probability: float | None,
    compression_ratio: float | None,
    no_speech_probability: float | None,
) -> list[TranscribedWord]:
    """Distribute words when a backend segment lacks word alignment."""
    tokens = str(getattr(segment, "text", "")).split()
    start = _optional_float(getattr(segment, "start", None))
    end = _optional_float(getattr(segment, "end", None))
    if not tokens or start is None or end is None or end <= start:
        return []
    duration = (end - start) / len(tokens)
    return [
        TranscribedWord(
            start=start + index * duration + offset,
            end=start + (index + 1) * duration + offset,
            text=token,
            average_log_probability=average_log_probability,
            compression_ratio=compression_ratio,
            no_speech_probability=no_speech_probability,
        )
        for index, token in enumerate(tokens)
    ]


def _resolve_faster_whisper_device(
    device: int | str | None,
) -> tuple[str, int]:
    if device is None:
        import torch

        return ("cuda", 0) if torch.cuda.is_available() else ("cpu", 0)
    if isinstance(device, int):
        return ("cpu", 0) if device < 0 else ("cuda", device)
    normalized = device.lower()
    if normalized == "cpu":
        return "cpu", 0
    if normalized.startswith("cuda"):
        _, _, raw_index = normalized.partition(":")
        return "cuda", int(raw_index) if raw_index else 0
    msg = f"Unsupported faster-whisper device: {device}"
    raise ValueError(msg)


def _preload_nvidia_libraries() -> None:
    """Preload pip-provided CUDA libraries before CTranslate2 inference."""
    if sys.platform != "linux":
        return

    library_directories = _nvidia_library_directories()
    if not library_directories:
        LOGGER.warning(
            "No pip-provided NVIDIA library directory found; using the "
            "system CUDA runtime"
        )
        return

    existing_path = os.environ.get("LD_LIBRARY_PATH", "")
    directory_path = os.pathsep.join(str(path) for path in library_directories)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        part for part in (directory_path, existing_path) if part
    )

    library_names = (
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.8",
        "libcudnn_ops_infer.so.8",
        "libcudnn_cnn_infer.so.8",
        "libcudnn_adv_infer.so.8",
    )
    loaded: list[str] = []
    for library_name in library_names:
        library_path = next(
            (
                directory / library_name
                for directory in library_directories
                if (directory / library_name).is_file()
            ),
            None,
        )
        if library_path is None:
            continue
        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        loaded.append(library_name)

    LOGGER.info(
        "Preloaded %s CUDA libraries for CTranslate2: %s",
        len(loaded),
        ", ".join(loaded) if loaded else "none",
    )


def _nvidia_library_directories() -> list[Path]:
    directories: list[Path] = []
    for package_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            spec = importlib.util.find_spec(package_name)
        except ModuleNotFoundError:
            continue
        if spec is None:
            continue
        if spec.submodule_search_locations:
            candidates = [Path(path) for path in spec.submodule_search_locations]
        elif spec.origin:
            candidates = [Path(spec.origin).parent]
        else:
            continue
        for candidate in candidates:
            if candidate not in directories:
                directories.append(candidate)
    return directories


def _faster_whisper_model_name(model_name: str) -> str:
    prefix = "openai/whisper-"
    if model_name.startswith(prefix):
        return model_name.removeprefix(prefix)
    return model_name


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
