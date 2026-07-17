from pathlib import Path

import pytest

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.core.pipeline import (
    prepare_speech_segments,
    prepare_subtitles,
    transcribe_segments,
)
from dual_subtitles.models.subtitle import Segment, SubtitleSegment
from dual_subtitles.services.diarization import (
    MissingHuggingFaceTokenError,
    PyannoteDiarizer,
)


def test_prepare_speech_segments_filters_merges_and_splits(tmp_path: Path) -> None:
    config = ProcessingConfig(
        input_dir=tmp_path,
        output_dir=tmp_path,
        temp_dir=tmp_path,
        min_speech_duration=0.7,
        merge_gap=0.6,
        max_speech_duration=2,
    )
    segments = [
        Segment(0, 0.2, "A"),
        Segment(0, 1, "A"),
        Segment(1.3, 4.3, "A"),
        Segment(5, 6, "B"),
    ]

    prepared = prepare_speech_segments(segments, config=config)

    assert prepared == [
        Segment(0, 2, "A"),
        Segment(2, 4, "A"),
        Segment(4, 4.3, "A"),
        Segment(5, 6, "B"),
    ]


def test_prepare_subtitles_cleans_merges_and_breaks_lines(tmp_path: Path) -> None:
    config = ProcessingConfig(
        input_dir=tmp_path,
        output_dir=tmp_path,
        temp_dir=tmp_path,
        line_break_words=4,
    )
    segments = [
        SubtitleSegment(0, 1, "one two", "A"),
        SubtitleSegment(0.9, 2, "three four five", "A"),
        SubtitleSegment(4, 5, "six", "A"),
    ]

    subtitles = prepare_subtitles(segments, config=config)

    assert subtitles == [
        SubtitleSegment(0, 2, "one two\nthree four five", "A"),
        SubtitleSegment(4, 5, "six", "A"),
    ]


def test_prepare_subtitles_enforces_projected_limits(tmp_path: Path) -> None:
    config = ProcessingConfig(
        input_dir=tmp_path,
        output_dir=tmp_path,
        temp_dir=tmp_path,
        max_subtitle_duration=2.5,
        max_words_per_subtitle=3,
        line_break_words=10,
    )

    subtitles = prepare_subtitles(
        [
            SubtitleSegment(0, 2, "one two", "A"),
            SubtitleSegment(2.1, 3, "three four", "A"),
        ],
        config=config,
    )

    assert subtitles == [
        SubtitleSegment(0, 2, "one two", "A"),
        SubtitleSegment(2.1, 3, "three four", "A"),
    ]


class FakeChunk:
    def export(self, path: Path, **kwargs: str) -> None:
        _ = (path, kwargs)


class FakeAudio:
    def __getitem__(self, interval: slice) -> FakeChunk:
        _ = interval
        return FakeChunk()


class FakeTranscriber:
    def __init__(self) -> None:
        self.offsets: list[float] = []

    def transcribe_segment(
        self,
        audio_path: Path,
        segment: Segment,
        *,
        language: str,
        offset: float,
    ) -> list[SubtitleSegment]:
        _ = (audio_path, language)
        self.offsets.append(offset)
        text = "one two" if len(self.offsets) == 1 else "two three"
        return [
            SubtitleSegment(offset, offset + segment.duration, text, segment.speaker)
        ]


def test_transcription_uses_clamped_offset_and_deduplicates(tmp_path: Path) -> None:
    config = ProcessingConfig(
        input_dir=tmp_path,
        output_dir=tmp_path,
        temp_dir=tmp_path,
        transcription_padding=0.3,
    )
    transcriber = FakeTranscriber()

    result = transcribe_segments(
        [Segment(0.1, 1, "A"), Segment(1.1, 2, "A")],
        audio=FakeAudio(),
        config=config,
        transcriber=transcriber,  # type: ignore[arg-type]
    )

    assert transcriber.offsets == [0.0, 0.8]
    assert [segment.text for segment in result] == ["one two", "three"]


def test_config_rejects_no_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one"):
        ProcessingConfig(
            input_dir=tmp_path,
            output_dir=tmp_path,
            temp_dir=tmp_path,
            generate_srt=False,
            generate_ass=False,
        )


def test_diarization_requires_huggingface_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    with pytest.raises(MissingHuggingFaceTokenError):
        PyannoteDiarizer()
