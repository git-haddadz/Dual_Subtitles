import re
from pathlib import Path

from dual_subtitles.io.subtitle_files import build_ass, build_srt, parse_srt
from dual_subtitles.models.subtitle import SubtitleSegment, WordPair
from dual_subtitles.utils.timestamps import (
    format_srt_timestamp,
)

SINGLE_PAIR_EVENT_COUNT = 2
TWO_PAIR_EVENT_COUNT = 4
SIX_PAIR_EVENT_COUNT = 12
MINIMUM_WRAPPED_Y_POSITIONS = 4


class StubTranslator:
    def interlinear(self, text: str) -> str:
        return text + r"\Ntranslated"

    def word_pairs(self, text: str) -> list[WordPair]:
        return [
            WordPair(source=word, translation=f"translated-{word}")
            for word in text.split()
        ]


def test_format_srt_timestamp() -> None:
    assert format_srt_timestamp(0) == "00:00:00,000"
    assert format_srt_timestamp(65.432) == "00:01:05,432"
    assert format_srt_timestamp(3661.007) == "01:01:01,007"


def test_build_srt_keeps_order_and_multiline_text() -> None:
    content = build_srt(
        [
            SubtitleSegment(start=0, end=1.2, text="hello\nworld"),
            SubtitleSegment(start=2.5, end=3, text="next"),
        ]
    )

    assert "1\n00:00:00,000 --> 00:00:01,200\nhello\nworld" in content
    assert "2\n00:00:02,500 --> 00:00:03,000\nnext" in content


def test_parse_srt_and_build_ass(tmp_path: Path) -> None:
    srt_path = tmp_path / "sample.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
        encoding="utf-8",
    )

    segments = parse_srt(srt_path)
    ass = build_ass(segments, StubTranslator())

    assert segments == [SubtitleSegment(start=0, end=1, text="hello")]
    dialogue_lines = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue_lines) == SINGLE_PAIR_EVENT_COUNT
    assert ",ArabicWord," in dialogue_lines[0]
    assert ",EnglishGloss," in dialogue_lines[1]
    positions = [re.search(r"\\pos\((\d+),(\d+)\)", line) for line in dialogue_lines]
    assert all(position is not None for position in positions)
    assert positions[0].group(1) == positions[1].group(1)  # type: ignore[union-attr]


def test_build_ass_keeps_events_on_one_line_and_escapes_tags() -> None:
    ass = build_ass(
        [SubtitleSegment(start=0, end=1, text="hello\n{world}")],
        StubTranslator(),
    )

    dialogue_lines = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue_lines) == TWO_PAIR_EVENT_COUNT
    assert all("\n" not in line for line in dialogue_lines)
    assert any(r"\{world\}" in line for line in dialogue_lines)


def test_build_ass_wraps_wide_pairs_without_breaking_alignment() -> None:
    ass = build_ass(
        [SubtitleSegment(start=0, end=1, text="one two three four five six")],
        StubTranslator(),
    )
    dialogue_lines = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    positions = [re.search(r"\\pos\((\d+),(\d+)\)", line) for line in dialogue_lines]
    coordinates = [
        (int(position.group(1)), int(position.group(2)))
        for position in positions
        if position is not None
    ]

    assert len(dialogue_lines) == SIX_PAIR_EVENT_COUNT
    assert len({y for _, y in coordinates}) >= MINIMUM_WRAPPED_Y_POSITIONS
    for index in range(0, len(coordinates), 2):
        assert coordinates[index][0] == coordinates[index + 1][0]
