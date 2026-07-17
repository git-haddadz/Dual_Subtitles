"""SRT and ASS file serialization helpers."""

from __future__ import annotations

import logging
import shutil
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.models.subtitle import (
    AnnotatedSpan,
    AnnotatedSubtitle,
    AnnotatedVideo,
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
ASS_ROW_HEIGHT = 118
ASS_PAIR_VERTICAL_GAP = 68
ASS_ARABIC_FONT_SIZE = 64
ASS_ENGLISH_FONT_SIZE = 28
LOGGER = logging.getLogger(__name__)
FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
SOURCE_FONT_FILE = FONT_DIR / "NotoNaskhArabic-Regular.ttf"
TARGET_FONT_FILE = FONT_DIR / "NotoSans-Regular.ttf"


def _normalize_ass_text(text: str) -> str:
    """Keep an ASS event on one physical line and neutralize override tags."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", r"\N")
    return normalized.replace("{", r"\{").replace("}", r"\}")


def _estimated_text_width(text: str, font_size: int) -> float:
    """Measure rendered width, with a deterministic fallback for missing fonts."""
    font_path = SOURCE_FONT_FILE if _contains_rtl(text) else TARGET_FONT_FILE
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(str(font_path), font_size)
        return float(font.getlength(text))
    except (ImportError, OSError):
        return _fallback_text_width(text, font_size)


def _fallback_text_width(text: str, font_size: int) -> float:
    """Estimate width only when Pillow or the packaged font is unavailable."""
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


def _contains_rtl(text: str) -> bool:
    return any(
        unicodedata.bidirectional(character) in {"AL", "R"} for character in text
    )


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
        "Style: ArabicWord,Arial,48,&H00FFFFFF,&H000000FF,"
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


@dataclass(frozen=True, slots=True)
class AssLayout:
    """Rendering values for a pedagogical ASS canvas."""

    source_font_name: str = "Noto Naskh Arabic"
    target_font_name: str = "Noto Sans"
    source_font_size: int = ASS_ARABIC_FONT_SIZE
    target_font_size: int = ASS_ENGLISH_FONT_SIZE
    vertical_gap: int = ASS_PAIR_VERTICAL_GAP


def _annotated_ass_header(layout: AssLayout) -> str:
    """Build ASS styles whose source line remains visually dominant."""
    return (
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
        f"Style: SourceWord,{layout.source_font_name},{layout.source_font_size},"
        "&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,"
        "0,0,1,3,1,5,0,0,0,1\n"
        f"Style: TargetGloss,{layout.target_font_name},{layout.target_font_size},"
        "&H00D9FFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,"
        "0,0,1,2,1,5,0,0,0,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,"
        "Effect,Text\n"
    )


def _font_measure(text: str, font_path: Path, font_size: int) -> tuple[float, float]:
    """Return exact font width and line height where Pillow can load the font."""
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(str(font_path), font_size)
        left, top, right, bottom = font.getbbox(text or "Ag")
        return float(right - left), float(bottom - top)
    except (ImportError, OSError):
        return _fallback_text_width(text, font_size), float(font_size * 1.25)


def _wrap_gloss(gloss: str, maximum_width: float, font_size: int) -> str:
    """Wrap a long target gloss to at most two ASS lines."""
    width, _ = _font_measure(gloss, TARGET_FONT_FILE, font_size)
    if width <= maximum_width or " " not in gloss:
        return gloss
    words = gloss.split()
    best_index = 1
    best_difference = float("inf")
    for index in range(1, len(words)):
        left = " ".join(words[:index])
        right = " ".join(words[index:])
        left_width, _ = _font_measure(left, TARGET_FONT_FILE, font_size)
        right_width, _ = _font_measure(right, TARGET_FONT_FILE, font_size)
        difference = abs(left_width - right_width)
        if (
            max(left_width, right_width) <= maximum_width
            and difference < best_difference
        ):
            best_index = index
            best_difference = difference
    return " ".join(words[:best_index]) + r"\N" + " ".join(words[best_index:])


def _span_column(span: AnnotatedSpan, layout: AssLayout) -> tuple[str, float, float]:
    """Return wrapped gloss, column width and target line height."""
    safe_width = ASS_PLAY_RES_X - ASS_HORIZONTAL_MARGIN * 2
    source_width, _ = _font_measure(
        span.display_source,
        SOURCE_FONT_FILE,
        layout.source_font_size,
    )
    gloss = _wrap_gloss(span.gloss, safe_width, layout.target_font_size)
    gloss_lines = gloss.split(r"\N")
    measurements = [
        _font_measure(line, TARGET_FONT_FILE, layout.target_font_size)
        for line in gloss_lines
    ]
    target_width = max((width for width, _ in measurements), default=0.0)
    target_height = sum(height for _, height in measurements)
    if len(measurements) > 1:
        target_height += layout.target_font_size * 0.2
    width = min(
        safe_width,
        max(source_width, target_width) + ASS_COLUMN_PADDING * 2,
    )
    return gloss, width, target_height


def _annotated_rows(
    spans: list[AnnotatedSpan],
    layout: AssLayout,
) -> list[list[tuple[AnnotatedSpan, str, float, float]]]:
    """Pack complete source/gloss pairs into right-to-left rows."""
    maximum_width = ASS_PLAY_RES_X - ASS_HORIZONTAL_MARGIN * 2
    rows: list[list[tuple[AnnotatedSpan, str, float, float]]] = []
    row: list[tuple[AnnotatedSpan, str, float, float]] = []
    row_width = 0.0
    for span in spans:
        gloss, width, target_height = _span_column(span, layout)
        required = width + (ASS_COLUMN_GAP if row else 0)
        if row and row_width + required > maximum_width:
            rows.append(row)
            row = []
            row_width = 0.0
            required = width
        row.append((span, gloss, width, target_height))
        row_width += required
    if row:
        rows.append(row)
    return rows


def _annotated_events(
    subtitle: AnnotatedSubtitle,
    layout: AssLayout,
) -> list[str]:
    """Position each annotated span and its gloss on a shared x coordinate."""
    rows = _annotated_rows(subtitle.spans, layout)
    start = format_ass_timestamp(subtitle.segment.start)
    end = format_ass_timestamp(subtitle.segment.end)
    bottom_y = ASS_PLAY_RES_Y - ASS_BOTTOM_MARGIN - layout.target_font_size // 2
    row_heights = []
    for row in rows:
        target_height = max((item[3] for item in row), default=layout.target_font_size)
        row_heights.append(
            layout.vertical_gap + target_height + layout.source_font_size
        )
    events: list[str] = []
    consumed_height = 0.0
    for row_index in range(len(rows) - 1, -1, -1):
        row = rows[row_index]
        target_height = max((item[3] for item in row), default=layout.target_font_size)
        target_y = round(bottom_y - consumed_height - target_height / 2)
        source_y = round(target_y - target_height / 2 - layout.vertical_gap)
        cursor_x = float(ASS_PLAY_RES_X - ASS_HORIZONTAL_MARGIN)
        for span, gloss, width, _ in row:
            center_x = round(cursor_x - width / 2)
            source_tag = rf"{{\an5\pos({center_x},{source_y})}}"
            target_tag = rf"{{\an5\pos({center_x},{target_y})}}"
            events.append(
                f"Dialogue: 0,{start},{end},SourceWord,,0,0,0,,"
                f"{source_tag}{_normalize_ass_text(span.display_source)}"
            )
            normalized_gloss = _normalize_ass_text(gloss).replace(r"\\N", r"\N")
            events.append(
                f"Dialogue: 1,{start},{end},TargetGloss,,0,0,0,,"
                f"{target_tag}{normalized_gloss}"
            )
            cursor_x -= width + ASS_COLUMN_GAP
        consumed_height += row_heights[row_index]
    return events


def build_annotated_ass(
    video: AnnotatedVideo,
    config: ProcessingConfig | None = None,
) -> str:
    """Build ASS from validated pedagogical spans without translating."""
    layout = AssLayout(
        source_font_name=(config.source_font_name if config else "Noto Naskh Arabic"),
        target_font_name=(config.target_font_name if config else "Noto Sans"),
        source_font_size=(config.source_font_size if config else ASS_ARABIC_FONT_SIZE),
        target_font_size=(config.target_font_size if config else ASS_ENGLISH_FONT_SIZE),
        vertical_gap=(config.pair_vertical_gap if config else ASS_PAIR_VERTICAL_GAP),
    )
    events: list[str] = []
    for index, subtitle in enumerate(video.subtitles, start=1):
        if not subtitle.spans:
            continue
        LOGGER.info("ASS rendering %s/%s", index, len(video.subtitles))
        events.extend(_annotated_events(subtitle, layout))
    header = _annotated_ass_header(layout)
    return header + "\n".join(events) + ("\n" if events else "")


def write_annotated_ass(
    path: Path,
    video: AnnotatedVideo,
    config: ProcessingConfig | None = None,
) -> None:
    """Write a validated annotated video to an ASS file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_annotated_ass(video, config), encoding="utf-8")


def provide_render_fonts(output_dir: Path) -> Path:
    """Copy the exact measured fonts beside generated subtitle outputs."""
    destination = output_dir / "fonts"
    destination.mkdir(parents=True, exist_ok=True)
    for source in (SOURCE_FONT_FILE, TARGET_FONT_FILE, FONT_DIR / "README.txt"):
        target = destination / source.name
        if source.is_file() and not target.is_file():
            shutil.copy2(source, target)
    return destination


def write_ass(
    path: Path,
    segments: Iterable[SubtitleSegment],
    translator: InterlinearTranslator,
) -> None:
    """Write subtitle segments to an ASS file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_ass(segments, translator), encoding="utf-8")
