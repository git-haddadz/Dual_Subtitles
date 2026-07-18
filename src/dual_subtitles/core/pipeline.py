"""Pipeline orchestration for video subtitle generation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.core.segmentation import (
    add_line_breaks,
    build_speaker_phrase_units,
    split_phrase_subtitle,
)
from dual_subtitles.io.audio import (
    detect_silence_boundaries,
    extract_audio,
    load_audio,
    normalize_audio,
)
from dual_subtitles.io.subtitle_files import parse_srt, write_ass, write_srt
from dual_subtitles.models.subtitle import Segment, SubtitleSegment
from dual_subtitles.services.diarization import PyannoteDiarizer, SingleSpeakerDiarizer
from dual_subtitles.services.transcription import SpeechTranscriber, create_transcriber
from dual_subtitles.services.translation import InterlinearGoogleTranslator

LOGGER = logging.getLogger(__name__)
MIN_TRANSCRIBABLE_DURATION = 0.2
VideoCompleteCallback = Callable[[Path, list[Path], Exception | None], None]


def discover_videos(input_dir: Path, extension: str) -> list[Path]:
    """Return videos in an input directory sorted by filename."""
    normalized = extension if extension.startswith(".") else f".{extension}"
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == normalized.lower()
    )


def process_directory(
    config: ProcessingConfig,
    *,
    video_limit: int | None = None,
    on_video_complete: VideoCompleteCallback | None = None,
) -> list[Path]:
    """Process all matching videos while isolating failures per file."""
    if not config.input_dir.is_dir():
        msg = f"Input directory does not exist: {config.input_dir}"
        raise FileNotFoundError(msg)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(config.input_dir, config.normalized_extension())
    if video_limit is not None:
        if video_limit <= 0:
            msg = "video_limit must be greater than zero or None."
            raise ValueError(msg)
        videos = videos[:video_limit]
    if not videos:
        LOGGER.warning(
            "No %s files found in %s",
            config.video_extension,
            config.input_dir,
        )
        return []

    transcriber: SpeechTranscriber | None = None
    diarizer: PyannoteDiarizer | None = None
    translator: InterlinearGoogleTranslator | None = None
    generated_files: list[Path] = []
    for index, video_path in enumerate(videos, start=1):
        LOGGER.info("[%s/%s] Processing %s", index, len(videos), video_path.name)
        video_outputs: list[Path] = []
        error: Exception | None = None
        try:
            if _needs_transcription(video_path, config):
                if transcriber is None:
                    transcriber = create_transcriber(
                        model_name=config.transcription_model,
                        device=config.device,
                    )
                if config.use_diarization and diarizer is None:
                    diarizer = PyannoteDiarizer(
                        config.diarization_model,
                        device=config.device,
                    )
            if _needs_translation(video_path, config) and translator is None:
                translator = InterlinearGoogleTranslator(
                    source_language=config.translation_source_language,
                    target_language=config.translation_target_language,
                )
            video_outputs = process_video(
                video_path,
                config=config,
                transcriber=transcriber,
                diarizer=diarizer,
                translator=translator,
            )
            generated_files.extend(video_outputs)
        except Exception as exc:  # noqa: BLE001 - isolate failures per video.
            error = exc
            LOGGER.exception("Failed to process %s", video_path)
        if on_video_complete is not None:
            on_video_complete(video_path, video_outputs, error)
    return generated_files


def process_video(  # noqa: PLR0912, PLR0915
    video_path: Path,
    *,
    config: ProcessingConfig,
    transcriber: SpeechTranscriber | None,
    diarizer: PyannoteDiarizer | None,
    translator: InterlinearGoogleTranslator | None,
) -> list[Path]:
    """Process one video with diarization before phrase-level recognition."""
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
        existing = [path for path in requested_paths if _has_content(path)]
        if len(existing) == len(requested_paths):
            LOGGER.info("Skipping %s because requested outputs exist", video_path.name)
            return existing
        if (
            config.generate_ass
            and _has_content(srt_path)
            and not _has_content(ass_path)
        ):
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
    chunk_path = config.temp_dir / f"{video_path.stem}.phrase.wav"
    try:
        LOGGER.info("Step 1/7 - Extracting audio from %s", video_path.name)
        extract_audio(video_path, audio_path)
        LOGGER.info("Step 2/7 - Normalizing audio to mono 16 kHz")
        normalize_audio(audio_path, audio_path)
        audio = load_audio(audio_path)
        duration_seconds = len(audio) / 1000

        if config.use_diarization:
            if diarizer is None:
                msg = "Diarization is enabled but no diarizer was provided."
                raise ValueError(msg)
            LOGGER.info("Step 3/7 - Detecting speaker turns before transcription")
            speaker_turns = diarizer.detect(audio_path)
        else:
            LOGGER.info("Step 3/7 - Using one speaker for the complete audio")
            speaker_turns = SingleSpeakerDiarizer().detect_duration(duration_seconds)

        LOGGER.info("Step 4/7 - Splitting speaker turns at acoustic pauses")
        silence_boundaries = detect_silence_boundaries(
            audio=audio,
            min_silence_duration=config.silence_min_duration,
            threshold_offset=config.silence_threshold_offset,
        )
        phrase_units = build_speaker_phrase_units(
            speaker_turns,
            silence_boundaries,
            min_duration=config.min_speech_duration,
            max_duration=config.max_speech_duration,
            merge_gap=config.merge_gap,
        )
        LOGGER.info(
            "Prepared %s speaker/phrase units from %s turns",
            len(phrase_units),
            len(speaker_turns),
        )

        LOGGER.info("Step 5/7 - Transcribing phrase units with Cohere Arabic")
        recognized = transcribe_phrase_units(
            phrase_units,
            audio=audio,
            language=config.transcription_language,
            transcriber=transcriber,
            temp_chunk=chunk_path,
        )
        subtitles = prepare_phrase_subtitles(recognized, config=config)
        if not subtitles:
            LOGGER.warning("Cohere returned no usable subtitle text")
            return []
        LOGGER.info("Step 6/7 - Prepared %s readable subtitles", len(subtitles))

        LOGGER.info("Step 7/7 - Writing subtitle files")
        generated: list[Path] = []
        if config.generate_srt:
            write_srt(srt_path, subtitles)
            generated.append(srt_path)
            LOGGER.info("SRT ready: %s", srt_path)
        if config.generate_ass:
            assert translator is not None
            write_ass(ass_path, subtitles, translator)
            generated.append(ass_path)
            LOGGER.info("ASS ready: %s", ass_path)
        return generated
    finally:
        audio_path.unlink(missing_ok=True)
        chunk_path.unlink(missing_ok=True)


def transcribe_phrase_units(
    units: Iterable[Segment],
    *,
    audio: Any,
    language: str,
    transcriber: SpeechTranscriber,
    temp_chunk: Path,
) -> list[SubtitleSegment]:
    """Transcribe each isolated unit without crossing a speaker boundary."""
    unit_list = list(units)
    recognized: list[SubtitleSegment] = []
    for index, unit in enumerate(unit_list, start=1):
        if unit.duration < MIN_TRANSCRIBABLE_DURATION:
            continue
        LOGGER.info(
            "Transcription %s/%s (%s%%) - %.1fs to %.1fs - %s",
            index,
            len(unit_list),
            round(index / len(unit_list) * 100),
            unit.start,
            unit.end,
            unit.speaker,
        )
        start_ms = max(0, round(unit.start * 1000))
        end_ms = min(len(audio), round(unit.end * 1000))
        audio[start_ms:end_ms].export(temp_chunk, format="wav")
        result = transcriber.transcribe_segment(
            temp_chunk,
            unit,
            language=language,
        )
        if result is not None:
            recognized.append(result)
    return recognized


def prepare_phrase_subtitles(
    subtitles: Iterable[SubtitleSegment],
    *,
    config: ProcessingConfig,
) -> list[SubtitleSegment]:
    """Split long text while preserving every acoustic speaker boundary."""
    split: list[SubtitleSegment] = []
    for subtitle in subtitles:
        if subtitle.end <= subtitle.start or not subtitle.text.strip():
            continue
        split.extend(
            split_phrase_subtitle(
                subtitle,
                max_words=config.max_words_per_subtitle,
            )
        )
    return add_line_breaks(split, line_break_words=config.line_break_words)


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
