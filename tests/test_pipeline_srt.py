from __future__ import annotations

from pathlib import Path
from typing import Any

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.core.pipeline import process_video, transcribe_phrase_units
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
        retry: bool = False,
    ) -> SubtitleSegment:
        del language, retry
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
        analyze_voice_profiles=False,
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


class RetryingTranscriber:
    def __init__(
        self,
        retry_text: str,
        *,
        initial_text: str = "�" * 60,
    ) -> None:
        self.retry_text = retry_text
        self.initial_text = initial_text
        self.attempts: list[bool] = []

    def transcribe_segment(
        self,
        _path: Path,
        segment: Segment,
        *,
        language: str,
        retry: bool = False,
    ) -> SubtitleSegment:
        del language
        self.attempts.append(retry)
        text = self.retry_text if retry else self.initial_text
        return SubtitleSegment(segment.start, segment.end, text, segment.speaker)


def test_suspicious_transcription_is_retried() -> None:
    transcriber = RetryingTranscriber("النص الصحيح")

    result = transcribe_phrase_units(
        [Segment(1.0, 3.0, "SPEAKER_02")],
        audio=FakeAudio(),
        language="ar",
        transcriber=transcriber,
        temp_chunk=Path("phrase.wav"),
    )

    assert transcriber.attempts == [False, True]
    assert result == [SubtitleSegment(1.0, 3.0, "النص الصحيح", "SPEAKER_02")]


def test_invalid_retry_is_discarded() -> None:
    transcriber = RetryingTranscriber("ه" * 100)

    result = transcribe_phrase_units(
        [Segment(1.0, 2.0, "SPEAKER_02")],
        audio=FakeAudio(),
        language="ar",
        transcriber=transcriber,
        temp_chunk=Path("phrase.wav"),
    )

    assert transcriber.attempts == [False, True]
    assert result == []


def test_truncated_arabic_ending_is_retried() -> None:
    transcriber = RetryingTranscriber(
        "لَنْ يَنْفَعَ إنْ لَمْ يَكُنْ مُطَهَّرًا.",
        initial_text="لَنْ يَنْفَعَ إنْ لَمْ يَكُنْ م",
    )

    result = transcribe_phrase_units(
        [Segment(6.13, 8.35, "SPEAKER_02")],
        audio=FakeAudio(),
        language="ar",
        transcriber=transcriber,
        temp_chunk=Path("phrase.wav"),
    )

    assert transcriber.attempts == [False, True]
    assert result == [
        SubtitleSegment(
            6.13,
            8.35,
            "لَنْ يَنْفَعَ إنْ لَمْ يَكُنْ مُطَهَّرًا.",
            "SPEAKER_02",
        )
    ]
