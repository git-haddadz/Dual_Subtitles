"""Command line interface for dual_subtitles."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.core.pipeline import process_directory


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="dual-subtitles",
        description="Generate SRT and ASS dual-language subtitles from videos.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process a video directory.")
    process.add_argument("--input-dir", required=True, type=Path)
    process.add_argument("--output-dir", required=True, type=Path)
    process.add_argument("--temp-dir", type=Path)
    process.add_argument("--extension", default=".mp4")
    process.add_argument("--transcription-language", default="ar")
    process.add_argument("--translation-source-language", default="ar")
    process.add_argument("--translation-target-language", default="en")
    process.add_argument("--whisper-model", default="openai/whisper-large-v3")
    process.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-3.1",
    )
    process.add_argument("--min-speech-duration", default=0.7, type=float)
    process.add_argument("--merge-gap", default=0.6, type=float)
    process.add_argument("--max-speech-duration", default=10.0, type=float)
    process.add_argument("--subtitle-gap-threshold", default=0.25, type=float)
    process.add_argument("--max-subtitle-duration", default=2.5, type=float)
    process.add_argument("--transcription-padding", default=0.3, type=float)
    process.add_argument("--max-words-per-subtitle", default=8, type=int)
    process.add_argument("--line-break-words", default=6, type=int)
    process.add_argument("--device")
    process.add_argument("--no-srt", action="store_true")
    process.add_argument("--no-ass", action="store_true")
    process.add_argument("--no-skip-existing", action="store_true")
    process.add_argument("--no-diarization", action="store_true")
    process.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "process":
        temp_dir = args.temp_dir or Path(tempfile.mkdtemp(prefix="dual-subtitles-"))
        config = ProcessingConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            temp_dir=temp_dir,
            transcription_language=args.transcription_language,
            translation_source_language=args.translation_source_language,
            translation_target_language=args.translation_target_language,
            whisper_model=args.whisper_model,
            diarization_model=args.diarization_model,
            min_speech_duration=args.min_speech_duration,
            merge_gap=args.merge_gap,
            max_speech_duration=args.max_speech_duration,
            subtitle_gap_threshold=args.subtitle_gap_threshold,
            max_subtitle_duration=args.max_subtitle_duration,
            transcription_padding=args.transcription_padding,
            max_words_per_subtitle=args.max_words_per_subtitle,
            line_break_words=args.line_break_words,
            generate_srt=not args.no_srt,
            generate_ass=not args.no_ass,
            skip_existing=not args.no_skip_existing,
            use_diarization=not args.no_diarization,
            video_extension=args.extension,
            device=_parse_device(args.device),
        )
        generated = process_directory(config)
        for path in generated:
            print(path)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
