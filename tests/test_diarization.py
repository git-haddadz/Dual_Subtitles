from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dual_subtitles.models.subtitle import Segment
from dual_subtitles.services.diarization import PyannoteDiarizer


class FakeCommunityPipeline:
    def __init__(self) -> None:
        self.audio_input: dict[str, Any] | None = None

    def __call__(self, audio_input: dict[str, Any]) -> SimpleNamespace:
        self.audio_input = audio_input
        turns = [
            (SimpleNamespace(start=0.2, end=1.5), "SPEAKER_00"),
            (SimpleNamespace(start=1.8, end=3.9), "SPEAKER_01"),
        ]
        return SimpleNamespace(speaker_diarization=turns)


def test_community_pipeline_output_is_converted_to_segments() -> None:
    diarizer = object.__new__(PyannoteDiarizer)
    diarizer.model_name = "test"
    diarizer.token_env_var = "HUGGINGFACE_TOKEN"
    diarizer.device = -1
    pipeline = FakeCommunityPipeline()
    diarizer._pipeline = pipeline

    fake_audio = SimpleNamespace(T=SimpleNamespace(copy=lambda: [[0.1, 0.2]]))

    class FakeSoundFile:
        @staticmethod
        def read(*_args: Any, **_kwargs: Any) -> tuple[Any, int]:
            return fake_audio, 16_000

    class FakeTorch:
        @staticmethod
        def from_numpy(value: Any) -> Any:
            return value

    import sys

    original_soundfile = sys.modules.get("soundfile")
    original_torch = sys.modules.get("torch")
    sys.modules["soundfile"] = FakeSoundFile()  # type: ignore[assignment]
    sys.modules["torch"] = FakeTorch()  # type: ignore[assignment]
    try:
        segments = diarizer.detect(Path("audio.wav"))
    finally:
        if original_soundfile is None:
            del sys.modules["soundfile"]
        else:
            sys.modules["soundfile"] = original_soundfile
        if original_torch is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = original_torch

    assert segments == [
        Segment(0.2, 1.5, "SPEAKER_00"),
        Segment(1.8, 3.9, "SPEAKER_01"),
    ]
    assert pipeline.audio_input == {
        "waveform": [[0.1, 0.2]],
        "sample_rate": 16_000,
    }
