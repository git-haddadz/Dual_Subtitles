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
from dual_subtitles.core.transcript import (
    assign_speakers,
    build_audio_windows,
    detect_suspicious_passages,
    merge_overlapping_words,
    replace_passage_if_better,
    words_to_subtitles,
)
from dual_subtitles.io.audio import (
    detect_silence_boundaries,
    extract_audio,
    load_audio,
    normalize_audio,
)
from dual_subtitles.io.subtitle_files import parse_srt, write_ass, write_srt
from dual_subtitles.models.subtitle import Segment, SubtitleSegment, TranscribedWord
from dual_subtitles.services.diarization import PyannoteDiarizer, SingleSpeakerDiarizer
from dual_subtitles.services.transcription import WhisperTranscriber
from dual_subtitles.services.translation import InterlinearGoogleTranslator

LOGGER = logging.getLogger(__name__)
MIN_TRANSCRIBABLE_DURATION = 0.3
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
    """Process all matching videos in a directory."""
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

    transcriber: WhisperTranscriber | None = None
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
                    transcriber = WhisperTranscriber(
                        model_name=config.whisper_model,
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
        LOGGER.info("Step 1/8 - Extracting audio from %s", video_path.name)
        extract_audio(video_path, audio_path)
        LOGGER.info("Step 2/8 - Normalizing audio to mono 16 kHz")
        normalize_audio(audio_path, audio_path)
        audio = load_audio(audio_path)

        duration_seconds = len(audio) / 1000
        LOGGER.info("Step 3/8 - Building continuous acoustic windows")
        silence_boundaries = detect_silence_boundaries(
            audio=audio,
            min_silence_duration=config.silence_min_duration,
            threshold_offset=config.silence_threshold_offset,
        )
        audio_windows = build_audio_windows(
            duration_seconds,
            silence_boundaries,
            target_duration=config.transcription_window_duration,
            max_duration=config.transcription_max_window_duration,
            overlap=config.transcription_overlap,
        )
        LOGGER.info(
            "Prepared %s continuous windows using %s silence boundaries",
            len(audio_windows),
            len(silence_boundaries),
        )

        LOGGER.info("Step 4/8 - Transcribing continuous audio before diarization")
        words = transcribe_audio_windows(
            audio_windows,
            audio=audio,
            config=config,
            transcriber=transcriber,
            temp_chunk=chunk_path,
        )
        if not words:
            LOGGER.warning("Whisper returned no timestamped words")
            return []

        LOGGER.info("Step 5/8 - Checking suspicious transcript passages")
        words = retry_suspicious_passages(
            words,
            audio=audio,
            config=config,
            transcriber=transcriber,
            temp_chunk=chunk_path,
        )

        if config.use_diarization:
            if diarizer is None:
                msg = "Diarization is enabled but no diarizer was provided."
                raise ValueError(msg)
            LOGGER.info("Step 6/8 - Detecting and assigning speakers")
            speaker_turns = diarizer.detect(audio_path)
        else:
            LOGGER.info("Step 6/8 - Assigning a single speaker")
            speaker_turns = SingleSpeakerDiarizer().detect_duration(duration_seconds)
        words = assign_speakers(words, speaker_turns)

        LOGGER.info("Step 7/8 - Reconstructing readable subtitles from words")
        subtitles = prepare_word_subtitles(words, config=config)
        LOGGER.info("Final subtitle segments: %s", len(subtitles))

        LOGGER.info("Step 8/8 - Writing subtitle files")
        generated_files: list[Path] = []
        if config.generate_srt:
            write_srt(srt_path, subtitles)
            LOGGER.info("SRT ready: %s", srt_path)
            generated_files.append(srt_path)
        if config.generate_ass:
            assert translator is not None
            write_ass(ass_path, subtitles, translator)
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


def prepare_word_subtitles(
    words: Iterable[TranscribedWord],
    *,
    config: ProcessingConfig,
) -> list[SubtitleSegment]:
    """Build final subtitle segments directly from Whisper word timestamps."""
    subtitles = words_to_subtitles(
        words,
        gap_threshold=config.subtitle_gap_threshold,
        max_duration=config.max_subtitle_duration,
        max_words=config.max_words_per_subtitle,
    )
    cleaned = clean_transcribed_segments(subtitles)
    return add_line_breaks(cleaned, line_break_words=config.line_break_words)


def transcribe_audio_windows(
    windows: Iterable[Segment],
    *,
    audio: Any,
    config: ProcessingConfig,
    transcriber: WhisperTranscriber,
    temp_chunk: Path | None = None,
) -> list[TranscribedWord]:
    """Transcribe overlapping continuous windows independently of speakers."""
    all_words: list[TranscribedWord] = []
    window_list = list(windows)
    temp_chunk = temp_chunk or config.temp_dir / "continuous-window.wav"
    for index, window in enumerate(window_list, start=1):
        progress = round(index / len(window_list) * 100)
        LOGGER.info(
            "Transcription %s/%s (%s%%) - %.1fs to %.1fs",
            index,
            len(window_list),
            progress,
            window.start,
            window.end,
        )
        start_ms = max(0, round(window.start * 1000))
        end_ms = min(len(audio), round(window.end * 1000))
        audio[start_ms:end_ms].export(temp_chunk, format="wav")
        all_words.extend(
            transcriber.transcribe_window(
                temp_chunk,
                language=config.transcription_language,
                offset=start_ms / 1000,
                num_beams=config.transcription_num_beams,
            )
        )
    merged = merge_overlapping_words(all_words)
    LOGGER.info(
        "Whisper emitted %s words; %s remain after overlap fusion",
        len(all_words),
        len(merged),
    )
    return merged


def retry_suspicious_passages(
    words: Iterable[TranscribedWord],
    *,
    audio: Any,
    config: ProcessingConfig,
    transcriber: WhisperTranscriber,
    temp_chunk: Path | None = None,
) -> list[TranscribedWord]:
    """Retry suspect intervals with more context and conservative selection."""
    selected = list(words)
    if not config.enable_targeted_retry:
        return selected
    passages = detect_suspicious_passages(
        selected,
        confidence_threshold=config.suspicious_confidence_threshold,
    )
    LOGGER.info("Detected %s suspicious passages", len(passages))
    if not passages:
        return selected

    temp_chunk = temp_chunk or config.temp_dir / "retry-window.wav"
    duration_seconds = len(audio) / 1000
    for index, passage in enumerate(passages, start=1):
        context_start = max(0.0, passage.start - config.retry_context)
        context_end = min(
            duration_seconds,
            passage.end + config.retry_context,
        )
        LOGGER.info(
            "Retry %s/%s - %.1fs to %.1fs (%s)",
            index,
            len(passages),
            context_start,
            context_end,
            ", ".join(passage.reasons),
        )
        start_ms = round(context_start * 1000)
        end_ms = round(context_end * 1000)
        audio[start_ms:end_ms].export(temp_chunk, format="wav")
        retry_words = transcriber.transcribe_window(
            temp_chunk,
            language=config.transcription_language,
            offset=context_start,
            num_beams=config.transcription_num_beams,
        )
        selected, replaced = replace_passage_if_better(
            selected,
            retry_words,
            passage,
            min_improvement=config.retry_min_improvement,
        )
        if replaced:
            LOGGER.info("Accepted retry %s/%s", index, len(passages))
        else:
            LOGGER.info(
                "Kept original transcript for retry %s/%s",
                index,
                len(passages),
            )
    return selected


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
