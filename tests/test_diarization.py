from pathlib import Path
from types import SimpleNamespace

from dual_subtitles.models.subtitle import Segment
from dual_subtitles.services.diarization import PyannoteDiarizer


class FakeCommunityPipeline:
    def __call__(self, _path: str) -> SimpleNamespace:
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
    diarizer._pipeline = FakeCommunityPipeline()

    assert diarizer.detect(Path("audio.wav")) == [
        Segment(0.2, 1.5, "SPEAKER_00"),
        Segment(1.8, 3.9, "SPEAKER_01"),
    ]
