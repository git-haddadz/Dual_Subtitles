"""Pipeline orchestration for video subtitle generation."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.core.segmentation import (
    add_line_breaks,
    clean_transcribed_segments,
    merge_speech_segments,
    merge_subtitle_segments,
    split_long_segments,
)
from dual_subtitles.io.audio import extract_audio, load_audio, normalize_audio
from dual_subtitles.io.subtitle_files import parse_srt, write_ass, write_srt
from dual_subtitles.models.subtitle import Segment, SubtitleSegment
from dual_subtitles.services.diarization import PyannoteDiarizer, SingleSpeakerDiarizer
from dual_subtitles.services.transcription import WhisperTranscriber
from dual_subtitles.services.translation import InterlinearGoogleTranslator

LOGGER = logging.getLogger(__name__)
MIN_TRANSCRIBABLE_DURATION = 0.3


def discover_videos(input_dir: Path, extension: str) -> list[Path]:
    """Return videos in an input directory sorted by filename."""
    normalized = extension if extension.startswith(".") else f".{extension}"
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == normalized.lower()
    )


def process_directory(config: ProcessingConfig) -> list[Path]:
    """Process all matching videos in a directory."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(config.input_dir, config.normalized_extension())
    if not videos:
        LOGGER.warning(
            "No %s files found in %s",
            config.video_extension,
            config.input_dir,
        )
        return []

    transcriber = WhisperTranscriber(
        model_name=config.whisper_model,
        device=config.device,
    )
    diarizer = (
        PyannoteDiarizer(config.diarization_model) if config.use_diarization else None
    )
    translator = InterlinearGoogleTranslator(
        source_language=config.translation_source_language,
        target_language=config.translation_target_language,
    )

    generated_files: list[Path] = []
    for index, video_path in enumerate(videos, start=1):
        LOGGER.info("[%s/%s] Processing %s", index, len(videos), video_path.name)
        generated_files.extend(
            process_video(
                video_path,
                config=config,
                transcriber=transcriber,
                diarizer=diarizer,
                translator=translator,
            )
        )
    return generated_files


def process_video(
    video_path: Path,
    *,
    config: ProcessingConfig,
    transcriber: WhisperTranscriber,
    diarizer: PyannoteDiarizer | None,
    translator: InterlinearGoogleTranslator,
) -> list[Path]:
    """Process one video and return generated subtitle paths."""
    output_base = config.output_dir / video_path.with_suffix("").name
    srt_path = output_base.with_suffix(".srt")
    ass_path = output_base.with_suffix(".ass")
    if config.skip_existing and srt_path.exists() and srt_path.stat().st_size > 0:
        LOGGER.info("Skipping %s because %s already exists", video_path.name, srt_path)
        existing_outputs = [srt_path]
        if config.generate_ass and not ass_path.exists():
            write_ass(ass_path, parse_srt(srt_path), translator)
            existing_outputs.append(ass_path)
        return existing_outputs

    audio_path = config.temp_dir / f"{video_path.stem}.wav"
    extract_audio(video_path, audio_path)
    normalize_audio(audio_path, audio_path)
    audio = load_audio(audio_path)

    if config.use_diarization:
        if diarizer is None:
            msg = "Diarization is enabled but no diarizer was provided."
            raise ValueError(msg)
        speech_segments = diarizer.detect(audio_path)
    else:
        duration_seconds = len(audio) / 1000
        speech_segments = SingleSpeakerDiarizer().detect_duration(duration_seconds)

    speech_segments = prepare_speech_segments(speech_segments, config=config)
    LOGGER.info("Detected %s speech segments", len(speech_segments))
    if not speech_segments:
        return []

    transcribed = transcribe_segments(
        speech_segments,
        audio=audio,
        config=config,
        transcriber=transcriber,
    )
    subtitles = prepare_subtitles(transcribed, config=config)
    LOGGER.info("Final subtitle segments: %s", len(subtitles))

    generated_files: list[Path] = []
    if config.generate_srt:
        write_srt(srt_path, subtitles)
        generated_files.append(srt_path)
    if config.generate_ass:
        write_ass(ass_path, subtitles, translator)
        generated_files.append(ass_path)
    return generated_files


def prepare_speech_segments(
    segments: Iterable[Segment],
    *,
    config: ProcessingConfig,
) -> list[Segment]:
    """Apply filtering, merging, and splitting to speech segments."""
    merged = merge_speech_segments(
        segments,
        min_duration=config.min_speech_duration,
        merge_gap=config.merge_gap,
    )
    return split_long_segments(merged, max_duration=config.max_speech_duration)


def prepare_subtitles(
    segments: Iterable[SubtitleSegment],
    *,
    config: ProcessingConfig,
) -> list[SubtitleSegment]:
    """Clean and format transcribed subtitle segments."""
    cleaned = clean_transcribed_segments(segments)
    merged = merge_subtitle_segments(
        cleaned,
        gap_threshold=config.subtitle_gap_threshold,
        max_duration=config.max_subtitle_duration,
        max_words=config.max_words_per_subtitle,
    )
    return add_line_breaks(merged, line_break_words=config.line_break_words)


def transcribe_segments(
    segments: Iterable[Segment],
    *,
    audio: Any,
    config: ProcessingConfig,
    transcriber: WhisperTranscriber,
) -> list[SubtitleSegment]:
    """Transcribe prepared speech segments."""
    transcribed: list[SubtitleSegment] = []
    temp_chunk = config.temp_dir / "chunk.wav"
    for segment in segments:
        if segment.duration < MIN_TRANSCRIBABLE_DURATION:
            continue

        start_ms = max(0, int((segment.start - config.transcription_padding) * 1000))
        end_ms = int((segment.end + config.transcription_padding) * 1000)
        chunk = audio[start_ms:end_ms]
        chunk.export(temp_chunk, format="wav")
        offset = segment.start - config.transcription_padding
        transcribed.extend(
            transcriber.transcribe_segment(
                temp_chunk,
                segment,
                language=config.transcription_language,
                offset=offset,
            )
        )
    return transcribed
