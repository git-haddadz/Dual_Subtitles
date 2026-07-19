import json
from pathlib import Path

from dual_subtitles.io.speaker_metadata import write_speaker_metadata
from dual_subtitles.models.subtitle import Segment, SpeakerProfile, SubtitleSegment
from dual_subtitles.services.voice_profile import (
    _classify_f0,
    _non_overlapping_turns,
)


def test_voice_profile_classification_is_cautious() -> None:
    assert _classify_f0(120.0)[0] == "masculine"
    assert _classify_f0(170.0) == ("unknown", 0.5)
    assert _classify_f0(220.0)[0] == "feminine"


def test_voice_profile_rejects_a_median_with_a_wide_pitch_distribution() -> None:
    assert _classify_f0(220.0, q25_f0=140.0, q75_f0=260.0) == (
        "unknown",
        0.5,
    )


def test_voice_profile_uses_conservative_pitch_quartiles() -> None:
    assert _classify_f0(220.0, q25_f0=210.0, q75_f0=235.0)[0] == "feminine"
    assert _classify_f0(120.0, q25_f0=105.0, q75_f0=145.0)[0] == "masculine"


def test_overlapping_different_speakers_are_excluded() -> None:
    isolated = Segment(3.0, 4.0, "SPEAKER_00")

    result = _non_overlapping_turns(
        [
            Segment(0.0, 2.0, "SPEAKER_00"),
            Segment(1.0, 2.5, "SPEAKER_01"),
            isolated,
        ]
    )

    assert result == [isolated]


def test_speaker_metadata_preserves_subtitle_association() -> None:
    path = Path("speaker-metadata-test.json")
    profile = SpeakerProfile(
        speaker="SPEAKER_00",
        perceived_voice_gender="masculine",
        confidence=0.91,
        median_f0_hz=121.5,
        analyzed_duration=12.0,
        analyzed_segments=4,
    )

    try:
        write_speaker_metadata(
            path,
            profiles=[profile],
            turns=[Segment(1.0, 3.0, "SPEAKER_00")],
            subtitles=[SubtitleSegment(1.1, 2.9, "مرحبا", "SPEAKER_00")],
        )

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert (
            payload["profiles"]["SPEAKER_00"]["perceived_voice_gender"] == "masculine"
        )
        assert payload["subtitles"] == [
            {"start": 1.1, "end": 2.9, "speaker": "SPEAKER_00"}
        ]
        assert payload["schema_version"] == 2
    finally:
        path.unlink(missing_ok=True)
