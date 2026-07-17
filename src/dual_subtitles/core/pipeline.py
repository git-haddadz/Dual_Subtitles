"""Pipeline orchestration for video subtitle generation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
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
from dual_subtitles.io.subtitle_files import (
    parse_srt,
    provide_render_fonts,
    write_annotated_ass,
    write_srt,
)
from dual_subtitles.models.subtitle import Segment, SubtitleSegment
from dual_subtitles.services.diarization import PyannoteDiarizer, SingleSpeakerDiarizer
from dual_subtitles.services.linguistics import PedagogicalAnnotator
from dual_subtitles.services.transcription import WhisperTranscriber
from dual_subtitles.services.translation import (
    InterlinearGoogleTranslator,  # noqa: F401 - legacy monkeypatch surface.
    LocalMarianTranslator,
)

LOGGER = logging.getLogger(__name__)
MIN_TRANSCRIBABLE_DURATION = 0.3
VideoCompleteCallback = Callable[[Path, list[Path], Exception | None], None]
VideoProgressCallback = Callable[[Path, str, int, int], None]


def discover_videos(input_dir: Path, extension: str) -> list[Path]:
    """Return videos in an input directory sorted by filename."""
    normalized = extension if extension.startswith(".") else f".{extension}"
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == normalized.lower()
    )


def process_directory(  # noqa: PLR0912 - lazy service orchestration.
    config: ProcessingConfig,
    *,
    video_limit: int | None = None,
    on_video_complete: VideoCompleteCallback | None = None,
    on_progress: VideoProgressCallback | None = None,
) -> list[Path]:
    """Process all matching videos in a directory."""
    if not config.input_dir.is_dir():
        msg = f"Input directory does not exist: {config.input_dir}"
        raise FileNotFoundError(msg)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    if config.generate_ass:
        provide_render_fonts(config.output_dir)
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

    transcriber: WhisperTranscriber | None = None
    diarizer: PyannoteDiarizer | None = None
    translator: LocalMarianTranslator | None = None
    annotator: PedagogicalAnnotator | None = None

    generated_files: list[Path] = []
    for index, video_path in enumerate(videos, start=1):
        LOGGER.info("[%s/%s] Processing %s", index, len(videos), video_path.name)
        video_outputs: list[Path] = []
        error: Exception | None = None
        try:
            if _needs_transcription(video_path, config):
                if transcriber is None:
                    transcriber = WhisperTranscriber(
                        model_name=config.whisper_model,
                        device=config.device,
                    )
                if config.use_diarization and diarizer is None:
                    diarizer = PyannoteDiarizer(
                        config.diarization_model,
                        device=config.device,
                    )

            if _needs_translation(video_path, config):
                if translator is None:
                    translator = LocalMarianTranslator(
                        model_name=config.translation_model,
                        device=config.device,
                    )
                progress = None
                if on_progress is not None:

                    def progress(
                        stage: str,
                        current: int,
                        total: int,
                        path: Path = video_path,
                    ) -> None:
                        on_progress(path, stage, current, total)

                if annotator is None:
                    annotator = PedagogicalAnnotator(
                        config=config,
                        translator=translator,
                        progress=progress,
                    )
                else:
                    annotator.progress = progress

            video_outputs = process_video(
                video_path,
                config=config,
                transcriber=transcriber,
                diarizer=diarizer,
                annotator=annotator,
            )
            generated_files.extend(video_outputs)
        except Exception as exc:  # noqa: BLE001 - isolate failures per video.
            error = exc
            LOGGER.exception("Failed to process %s", video_path)
        if on_video_complete is not None:
            on_video_complete(video_path, video_outputs, error)
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
    annotator: PedagogicalAnnotator | None,
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
            if annotator is None:
                msg = "ASS generation requires a linguistic annotator."
                raise ValueError(msg)
            subtitles = parse_srt(srt_path)
            annotated = annotator.annotate(subtitles)
            write_annotated_ass(ass_path, annotated, config)
            annotator.report("ass", 1, 1)
            LOGGER.info("ASS ready: %s", ass_path)
            return [path for path in requested_paths if _has_content(path)]

    if transcriber is None:
        msg = "Transcription is required but no transcriber was provided."
        raise ValueError(msg)
    if config.generate_ass and annotator is None:
        msg = "ASS generation requires a linguistic annotator."
        raise ValueError(msg)

    audio_path = config.temp_dir / f"{video_path.stem}.wav"
    chunk_path = config.temp_dir / f"{video_path.stem}.chunk.wav"
    try:
        LOGGER.info("Step 1/6 - Extracting audio from %s", video_path.name)
        extract_audio(video_path, audio_path)
        LOGGER.info("Step 2/6 - Normalizing audio to mono 16 kHz")
        normalize_audio(audio_path, audio_path)
        audio = load_audio(audio_path)

        if config.use_diarization:
            if diarizer is None:
                msg = "Diarization is enabled but no diarizer was provided."
                raise ValueError(msg)
            LOGGER.info("Step 3/6 - Detecting speakers")
            speech_segments = diarizer.detect(audio_path)
        else:
            LOGGER.info("Step 3/6 - Using a single speaker segment")
            duration_seconds = len(audio) / 1000
            speech_segments = SingleSpeakerDiarizer().detect_duration(duration_seconds)

        speech_segments = prepare_speech_segments(speech_segments, config=config)
        LOGGER.info("Detected %s speech segments", len(speech_segments))
        if not speech_segments:
            return []

        LOGGER.info("Step 4/6 - Transcribing speech segments")
        transcribed = transcribe_segments(
            speech_segments,
            audio=audio,
            config=config,
            transcriber=transcriber,
            temp_chunk=chunk_path,
        )
        LOGGER.info("Step 5/6 - Preparing readable subtitles")
        subtitles = prepare_subtitles(transcribed, config=config)
        LOGGER.info("Final subtitle segments: %s", len(subtitles))

        LOGGER.info("Step 6/7 - Writing transcription output")
        generated_files: list[Path] = []
        if config.generate_srt:
            write_srt(srt_path, subtitles)
            LOGGER.info("SRT ready: %s", srt_path)
            generated_files.append(srt_path)
        if config.generate_ass:
            assert annotator is not None
            LOGGER.info("Step 7/7 - Annotating complete transcript")
            annotated = annotator.annotate(subtitles)
            write_annotated_ass(ass_path, annotated, config)
            annotator.report("ass", 1, 1)
            LOGGER.info("ASS ready: %s", ass_path)
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
    segment_list = list(segments)
    total_segments = len(segment_list)
    temp_chunk = temp_chunk or config.temp_dir / "chunk.wav"
    for index, segment in enumerate(segment_list, start=1):
        if segment.duration < MIN_TRANSCRIBABLE_DURATION:
            LOGGER.info(
                "Transcription %s/%s - skipped short segment",
                index,
                total_segments,
            )
            continue

        progress = round(index / total_segments * 100)
        LOGGER.info(
            "Transcription %s/%s (%s%%) - %.1fs to %.1fs - %s",
            index,
            total_segments,
            progress,
            segment.start,
            segment.end,
            segment.speaker,
        )

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
