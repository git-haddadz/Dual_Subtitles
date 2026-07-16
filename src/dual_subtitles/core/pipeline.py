"""Pipeline orchestration for video subtitle generation."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.core.segmentation import (
    add_line_breaks,
    clean_transcribed_segments,
    deduplicate_overlap,
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
    if not config.input_dir.is_dir():
        msg = f"Input directory does not exist: {config.input_dir}"
        raise FileNotFoundError(msg)

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

    transcriber: WhisperTranscriber | None = None
    diarizer: PyannoteDiarizer | None = None
    translator: InterlinearGoogleTranslator | None = None

    generated_files: list[Path] = []
    for index, video_path in enumerate(videos, start=1):
        LOGGER.info("[%s/%s] Processing %s", index, len(videos), video_path.name)
        try:
            if _needs_transcription(video_path, config):
                if transcriber is None:
                    transcriber = WhisperTranscriber(
                        model_name=config.whisper_model,
                        device=config.device,
                    )
                if config.use_diarization and diarizer is None:
                    diarizer = PyannoteDiarizer(config.diarization_model)

            if _needs_translation(video_path, config) and translator is None:
                translator = InterlinearGoogleTranslator(
                    source_language=config.translation_source_language,
                    target_language=config.translation_target_language,
                )

            generated_files.extend(
                process_video(
                    video_path,
                    config=config,
                    transcriber=transcriber,
                    diarizer=diarizer,
                    translator=translator,
                )
            )
        except Exception:  # noqa: BLE001 - one bad video must not stop the batch.
            LOGGER.exception("Failed to process %s", video_path)
    return generated_files


def _subtitle_paths(video_path: Path, output_dir: Path) -> tuple[Path, Path]:
    output_base = output_dir / video_path.stem
    return output_base.with_suffix(".srt"), output_base.with_suffix(".ass")


def _has_content(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _needs_transcription(video_path: Path, config: ProcessingConfig) -> bool:
    if not config.skip_existing:
        return True

    srt_path, ass_path = _subtitle_paths(video_path, config.output_dir)
    if config.generate_srt and not _has_content(srt_path):
        return True
    return (
        config.generate_ass
        and not _has_content(ass_path)
        and not _has_content(srt_path)
    )


def _needs_translation(video_path: Path, config: ProcessingConfig) -> bool:
    if not config.generate_ass:
        return False
    if not config.skip_existing:
        return True
    _, ass_path = _subtitle_paths(video_path, config.output_dir)
    return not _has_content(ass_path)


def process_video(  # noqa: PLR0912, PLR0915 - keep orchestration cleanup together.
    video_path: Path,
    *,
    config: ProcessingConfig,
    transcriber: WhisperTranscriber | None,
    diarizer: PyannoteDiarizer | None,
    translator: InterlinearGoogleTranslator | None,
) -> list[Path]:
    """Process one video and return generated subtitle paths."""
    srt_path, ass_path = _subtitle_paths(video_path, config.output_dir)
    requested_paths = [
        path
        for enabled, path in (
            (config.generate_srt, srt_path),
            (config.generate_ass, ass_path),
        )
        if enabled
    ]
    if config.skip_existing:
        existing_outputs = [path for path in requested_paths if _has_content(path)]
        if len(existing_outputs) == len(requested_paths):
            LOGGER.info("Skipping %s because requested outputs exist", video_path.name)
            return existing_outputs

        can_reuse_srt = (
            config.generate_ass
            and not _has_content(ass_path)
            and _has_content(srt_path)
        )
        if can_reuse_srt:
            if translator is None:
                msg = "ASS generation requires a translator."
                raise ValueError(msg)
            write_ass(ass_path, parse_srt(srt_path), translator)
            return [path for path in requested_paths if _has_content(path)]

    if transcriber is None:
        msg = "Transcription is required but no transcriber was provided."
        raise ValueError(msg)
    if config.generate_ass and translator is None:
        msg = "ASS generation requires a translator."
        raise ValueError(msg)

    audio_path = config.temp_dir / f"{video_path.stem}.wav"
    chunk_path = config.temp_dir / f"{video_path.stem}.chunk.wav"
    try:
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
            temp_chunk=chunk_path,
        )
        subtitles = prepare_subtitles(transcribed, config=config)
        LOGGER.info("Final subtitle segments: %s", len(subtitles))

        generated_files: list[Path] = []
        if config.generate_srt:
            write_srt(srt_path, subtitles)
            generated_files.append(srt_path)
        if config.generate_ass:
            assert translator is not None
            write_ass(ass_path, subtitles, translator)
            generated_files.append(ass_path)
        return generated_files
    finally:
        audio_path.unlink(missing_ok=True)
        chunk_path.unlink(missing_ok=True)


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
    temp_chunk: Path | None = None,
) -> list[SubtitleSegment]:
    """Transcribe prepared speech segments."""
    transcribed: list[SubtitleSegment] = []
    temp_chunk = temp_chunk or config.temp_dir / "chunk.wav"
    for segment in segments:
        if segment.duration < MIN_TRANSCRIBABLE_DURATION:
            continue

        start_ms = max(0, int((segment.start - config.transcription_padding) * 1000))
        end_ms = int((segment.end + config.transcription_padding) * 1000)
        chunk = audio[start_ms:end_ms]
        chunk.export(temp_chunk, format="wav")
        offset = start_ms / 1000
        new_segments = transcriber.transcribe_segment(
            temp_chunk,
            segment,
            language=config.transcription_language,
            offset=offset,
        )
        if transcribed and new_segments:
            deduplicated = deduplicate_overlap(
                transcribed[-1].text,
                new_segments[0].text,
            )
            if deduplicated:
                new_segments[0] = replace(new_segments[0], text=deduplicated)
            else:
                new_segments = new_segments[1:]
        transcribed.extend(new_segments)
    return transcribed
