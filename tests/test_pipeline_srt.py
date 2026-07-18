from __future__ import annotations

from pathlib import Path
from typing import Any

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.core.pipeline import process_video, transcribe_audio_windows
from dual_subtitles.io.subtitle_files import build_srt
from dual_subtitles.models.subtitle import Segment, SubtitleSegment, TranscribedWord


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

    def transcribe_window(self, *_args: Any, **_kwargs: Any) -> list[TranscribedWord]:
        self.calls.append("transcription")
        return [
            TranscribedWord(0.2, 0.5, "مرحبا", confidence=0.9),
            TranscribedWord(0.6, 1.0, "بكم.", confidence=0.9),
        ]


class RecordingDiarizer:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def detect(self, _audio_path: Path) -> list[Segment]:
        self.calls.append("diarization")
        return [Segment(0.0, 10.0, "SPEAKER_01")]


def test_process_video_transcribes_before_diarization(
    monkeypatch: Any,
) -> None:
    calls: list[str] = []
    written: list[SubtitleSegment] = []
    video = Path("video.mp4")
    config = ProcessingConfig(
        input_dir=Path("input"),
        output_dir=Path("output"),
        temp_dir=Path("temp"),
        generate_ass=False,
        enable_targeted_retry=False,
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
        video,
        config=config,
        transcriber=RecordingTranscriber(calls),  # type: ignore[arg-type]
        diarizer=RecordingDiarizer(calls),  # type: ignore[arg-type]
        translator=None,
    )

    assert calls == ["transcription", "diarization"]
    assert outputs == [Path("output/video.srt")]
    assert [subtitle.text for subtitle in written] == ["مرحبا بكم."]


def test_srt_serialization_uses_reconstructed_word_boundaries() -> None:
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


class PromptRecordingTranscriber:
    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    def transcribe_window(
        self,
        _path: Path,
        **kwargs: Any,
    ) -> list[TranscribedWord]:
        self.prompts.append(kwargs.get("prompt"))
        offset = float(kwargs["offset"])
        return [TranscribedWord(offset + 0.2, offset + 0.5, "context")]


def test_continuous_windows_pass_bounded_text_context() -> None:
    config = ProcessingConfig(
        input_dir=Path("input"),
        output_dir=Path("output"),
        temp_dir=Path("temp"),
        generate_ass=False,
        transcription_context_words=4,
    )
    transcriber = PromptRecordingTranscriber()

    transcribe_audio_windows(
        [Segment(0.0, 5.0), Segment(0.4, 10.0)],
        audio=FakeAudio(),
        config=config,
        transcriber=transcriber,  # type: ignore[arg-type]
        temp_chunk=Path("window.wav"),
    )

    assert transcriber.prompts == [None, "context"]
