"""SRT and ASS file serialization helpers."""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from dual_subtitles.models.subtitle import (
    InterlinearTranslator,
    SubtitleSegment,
    WordPair,
)
from dual_subtitles.utils.timestamps import (
    format_ass_timestamp,
    format_srt_timestamp,
    parse_srt_timestamp,
)

MIN_SRT_BLOCK_LINES = 3
ASS_PLAY_RES_X = 1280
ASS_PLAY_RES_Y = 720
ASS_HORIZONTAL_MARGIN = 60
ASS_BOTTOM_MARGIN = 35
ASS_COLUMN_GAP = 18
ASS_COLUMN_PADDING = 14
ASS_ROW_HEIGHT = 90
ASS_PAIR_VERTICAL_GAP = 44
ASS_ARABIC_FONT_SIZE = 42
ASS_ENGLISH_FONT_SIZE = 30
LOGGER = logging.getLogger(__name__)


def _normalize_ass_text(text: str) -> str:
    """Keep an ASS event on one physical line and neutralize override tags."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", r"\N")
    return normalized.replace("{", r"\{").replace("}", r"\}")


def _estimated_text_width(text: str, font_size: int) -> float:
    """Estimate rendered width without depending on a platform font engine."""
    width_units = 0.0
    for character in text:
        if character.isspace():
            width_units += 0.33
        elif unicodedata.bidirectional(character) in {"AL", "R"}:
            width_units += 0.62
        elif unicodedata.east_asian_width(character) in {"F", "W"}:
            width_units += 1.0
        elif character.isupper():
            width_units += 0.65
        else:
            width_units += 0.54
    return width_units * font_size


def _pair_column_width(pair: WordPair) -> float:
    source_width = _estimated_text_width(pair.source, ASS_ARABIC_FONT_SIZE)
    translation_width = _estimated_text_width(
        pair.translation,
        ASS_ENGLISH_FONT_SIZE,
    )
    return max(source_width, translation_width) + ASS_COLUMN_PADDING * 2


def _split_word_pair_rows(
    pairs: list[WordPair],
) -> list[list[tuple[WordPair, float]]]:
    """Split pairs into right-to-left rows that fit inside the safe area."""
    maximum_width = ASS_PLAY_RES_X - ASS_HORIZONTAL_MARGIN * 2
    rows: list[list[tuple[WordPair, float]]] = []
    current_row: list[tuple[WordPair, float]] = []
    current_width = 0.0

    for pair in pairs:
        column_width = _pair_column_width(pair)
        required_width = column_width
        if current_row:
            required_width += ASS_COLUMN_GAP
        if current_row and current_width + required_width > maximum_width:
            rows.append(current_row)
            current_row = []
            current_width = 0.0
            required_width = column_width
        current_row.append((pair, column_width))
        current_width += required_width

    if current_row:
        rows.append(current_row)
    return rows


def _build_word_pair_events(
    segment: SubtitleSegment,
    pairs: list[WordPair],
) -> list[str]:
    """Position every source/gloss pair at a shared horizontal coordinate."""
    start = format_ass_timestamp(segment.start)
    end = format_ass_timestamp(segment.end)
    rows = _split_word_pair_rows(pairs)
    bottom_translation_y = (
        ASS_PLAY_RES_Y - ASS_BOTTOM_MARGIN - ASS_ENGLISH_FONT_SIZE // 2
    )
    events: list[str] = []

    for row_index, row in enumerate(rows):
        rows_below = len(rows) - row_index - 1
        translation_y = bottom_translation_y - rows_below * ASS_ROW_HEIGHT
        source_y = translation_y - ASS_PAIR_VERTICAL_GAP
        cursor_x = float(ASS_PLAY_RES_X - ASS_HORIZONTAL_MARGIN)

        for pair, column_width in row:
            center_x = round(cursor_x - column_width / 2)
            source = _normalize_ass_text(pair.source)
            translation = _normalize_ass_text(pair.translation)
            source_tag = rf"{{\an5\pos({center_x},{source_y})}}"
            translation_tag = rf"{{\an5\pos({center_x},{translation_y})}}"
            events.append(
                f"Dialogue: 0,{start},{end},ArabicWord,,0,0,0,,{source_tag}{source}"
            )
            events.append(
                f"Dialogue: 1,{start},{end},EnglishGloss,,0,0,0,,"
                f"{translation_tag}{translation}"
            )
            cursor_x -= column_width + ASS_COLUMN_GAP

    return events


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
        f"PlayResX: {ASS_PLAY_RES_X}\n"
        f"PlayResY: {ASS_PLAY_RES_Y}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,Strikeout,ScaleX,"
        "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
        "MarginR,MarginV,Encoding\n"
        "Style: ArabicWord,Arial,42,&H00FFFFFF,&H000000FF,"
        "&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,5,0,0,0,1\n"
        "Style: EnglishGloss,Arial,30,&H00D9FFFF,&H000000FF,"
        "&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,5,0,0,0,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,"
        "Effect,Text\n"
    )

    segment_list = list(segments)
    total_segments = len(segment_list)
    events = []
    for index, segment in enumerate(segment_list, start=1):
        if not segment.text.strip():
            continue
        progress = round(index / total_segments * 100)
        LOGGER.info(
            "ASS translation %s/%s (%s%%)",
            index,
            total_segments,
            progress,
        )
        pairs = translator.word_pairs(segment.text)
        events.extend(_build_word_pair_events(segment, pairs))

    return header + "\n".join(events) + ("\n" if events else "")


def write_ass(
    path: Path,
    segments: Iterable[SubtitleSegment],
    translator: InterlinearTranslator,
) -> None:
    """Write subtitle segments to an ASS file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_ass(segments, translator), encoding="utf-8")
