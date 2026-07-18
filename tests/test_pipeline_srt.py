from __future__ import annotations

from pathlib import Path
from typing import Any

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.core.pipeline import process_video
from dual_subtitles.io.subtitle_files import build_srt
from dual_subtitles.models.subtitle import Segment, SubtitleSegment


class FakeAudio:
    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, _key: slice) -> FakeAudio:
        return self

    def export(self, path: Path, *, format: str) -> None:  # noqa: A002
        del path, format


class RecordingTranscriber:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def transcribe_segment(
        self,
        _path: Path,
        segment: Segment,
        *,
        language: str,
    ) -> SubtitleSegment:
        del language
        self.calls.append("transcription")
        return SubtitleSegment(
            segment.start,
            segment.end,
            "مرحبا بكم.",
            segment.speaker,
        )


class RecordingDiarizer:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def detect(self, _audio_path: Path) -> list[Segment]:
        self.calls.append("diarization")
        return [Segment(0.0, 10.0, "SPEAKER_01")]


def test_process_video_diarizes_before_phrase_transcription(
    monkeypatch: Any,
) -> None:
    calls: list[str] = []
    written: list[SubtitleSegment] = []
    config = ProcessingConfig(
        input_dir=Path("input"),
        output_dir=Path("output"),
        temp_dir=Path("temp"),
        generate_ass=False,
    )
    monkeypatch.setattr(
        "dual_subtitles.core.pipeline.extract_audio",
        lambda _source, target: target,
    )
    monkeypatch.setattr(
        "dual_subtitles.core.pipeline.normalize_audio",
        lambda _source, target: target,
    )
    monkeypatch.setattr(
        "dual_subtitles.core.pipeline.load_audio",
        lambda _path: FakeAudio(),
    )
    monkeypatch.setattr(
        "dual_subtitles.core.pipeline.detect_silence_boundaries",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "dual_subtitles.core.pipeline.write_srt",
        lambda _path, segments: written.extend(segments),
    )

    outputs = process_video(
        Path("video.mp4"),
        config=config,
        transcriber=RecordingTranscriber(calls),  # type: ignore[arg-type]
        diarizer=RecordingDiarizer(calls),  # type: ignore[arg-type]
        translator=None,
    )

    assert calls == ["diarization", "transcription"]
    assert outputs == [Path("output/video.srt")]
    assert written == [SubtitleSegment(0.0, 10.0, "مرحبا بكم.", "SPEAKER_01")]


def test_srt_serialization_preserves_phrase_boundaries() -> None:
    content = build_srt(
        [
            SubtitleSegment(
                start=1.25,
                end=3.5,
                text="مرحبا بكم",
                speaker="SPEAKER_00",
            )
        ]
    )

    assert content == "1\n00:00:01,250 --> 00:00:03,500\nمرحبا بكم\n"
