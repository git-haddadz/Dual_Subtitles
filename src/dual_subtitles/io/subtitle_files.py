"""SRT and ASS file serialization helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from dual_subtitles.models.subtitle import InterlinearTranslator, SubtitleSegment
from dual_subtitles.utils.timestamps import (
    format_ass_timestamp,
    format_srt_timestamp,
    parse_srt_timestamp,
)

MIN_SRT_BLOCK_LINES = 3


def build_srt(segments: Iterable[SubtitleSegment]) -> str:
    """Build SRT content from subtitle segments."""
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = format_srt_timestamp(segment.start)
        end = format_srt_timestamp(segment.end)
        blocks.append(f"{index}\n{start} --> {end}\n{segment.text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(path: Path, segments: Iterable[SubtitleSegment]) -> None:
    """Write subtitle segments to an SRT file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_srt(segments), encoding="utf-8")


def parse_srt(path: Path) -> list[SubtitleSegment]:
    """Parse a simple SRT file into subtitle segments."""
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    segments: list[SubtitleSegment] = []
    for block in content.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < MIN_SRT_BLOCK_LINES or " --> " not in lines[1]:
            continue

        start_text, end_text = lines[1].split(" --> ", maxsplit=1)
        segments.append(
            SubtitleSegment(
                start=parse_srt_timestamp(start_text),
                end=parse_srt_timestamp(end_text),
                text=" ".join(lines[2:]),
            )
        )

    return segments


def build_ass(
    segments: Iterable[SubtitleSegment],
    translator: InterlinearTranslator,
) -> str:
    """Build ASS content with interlinear translated subtitle text."""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 640\n"
        "PlayResY: 360\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,Strikeout,ScaleX,"
        "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
        "MarginR,MarginV,Encoding\n"
        "Style: Interlinear,Courier New,22,&H00FFFFFF,&H000000FF,"
        "&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,30,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,"
        "Effect,Text\n"
    )

    events = []
    for segment in segments:
        if not segment.text.strip():
            continue
        start = format_ass_timestamp(segment.start)
        end = format_ass_timestamp(segment.end)
        text = translator.interlinear(segment.text)
        events.append(f"Dialogue: 0,{start},{end},Interlinear,,0,0,0,,{text}")

    return header + "\n".join(events) + ("\n" if events else "")


def write_ass(
    path: Path,
    segments: Iterable[SubtitleSegment],
    translator: InterlinearTranslator,
) -> None:
    """Write subtitle segments to an ASS file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_ass(segments, translator), encoding="utf-8")
