from pathlib import Path
from typing import Any

from dual_subtitles.services.transcription import WhisperTranscriber, _chunks_to_words


def test_chunks_to_words_preserve_global_timestamps_and_confidence() -> None:
    words = _chunks_to_words(
        {
            "chunks": [
                {
                    "text": " في ",
                    "timestamp": (0.2, 0.6),
                    "confidence": 0.8,
                },
                {"text": "البيت", "timestamp": (0.7, 1.1)},
            ]
        },
        offset=10.0,
    )

    assert words[0].text == "في"
    assert words[0].start == 10.2
    assert words[0].end == 10.6
    assert words[0].confidence == 0.8
    assert words[1].confidence is None


class FailingBeamPipeline:
    def __init__(self) -> None:
        self.beam_calls: list[int] = []

    def __call__(self, _path: str, **kwargs: Any) -> dict[str, Any]:
        beams = int(kwargs["generate_kwargs"]["num_beams"])
        self.beam_calls.append(beams)
        if beams > 1:
            raise IndexError("inconsistent token timestamps")
        return {
            "chunks": [
                {"text": "مرحبا", "timestamp": (0.0, 0.4)},
            ]
        }


def test_word_timestamp_failure_falls_back_once_and_remembers_it() -> None:
    pipeline = FailingBeamPipeline()
    transcriber = object.__new__(WhisperTranscriber)
    transcriber.model_name = "test"
    transcriber.device = -1
    transcriber._pipeline = pipeline
    transcriber._safe_word_timestamp_beams = None
    transcriber._word_timestamps_supported = True

    first = transcriber.transcribe_window(
        Path("first.wav"),
        language="ar",
        offset=0.0,
        num_beams=5,
    )
    second = transcriber.transcribe_window(
        Path("second.wav"),
        language="ar",
        offset=10.0,
        num_beams=5,
    )

    assert pipeline.beam_calls == [5, 1, 1]
    assert first[0].text == "مرحبا"
    assert second[0].start == 10.0


class BrokenWordTimestampPipeline:
    def __init__(self) -> None:
        self.timestamp_modes: list[str] = []

    def __call__(self, _path: str, **kwargs: Any) -> dict[str, Any]:
        mode = kwargs["return_timestamps"]
        self.timestamp_modes.append(str(mode))
        if mode == "word":
            raise IndexError("inconsistent token timestamps")
        return {
            "chunks": [
                {"text": "مرحبا بكم", "timestamp": (0.0, 1.0)},
            ]
        }


def test_broken_word_timestamps_fall_back_to_segment_timing() -> None:
    pipeline = BrokenWordTimestampPipeline()
    transcriber = object.__new__(WhisperTranscriber)
    transcriber.model_name = "test"
    transcriber.device = -1
    transcriber._pipeline = pipeline
    transcriber._safe_word_timestamp_beams = None
    transcriber._word_timestamps_supported = True

    first = transcriber.transcribe_window(
        Path("first.wav"),
        language="ar",
        offset=2.0,
        num_beams=5,
    )
    second = transcriber.transcribe_window(
        Path("second.wav"),
        language="ar",
        offset=5.0,
        num_beams=5,
    )

    assert pipeline.timestamp_modes == ["word", "word", "True", "True"]
    assert [item.text for item in first] == ["مرحبا", "بكم"]
    assert first[0].start == 2.0
    assert first[1].end == 3.0
    assert second[0].start == 5.0
