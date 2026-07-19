"""Persistent speaker metadata used by later linguistic post-processing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from dual_subtitles.models.subtitle import Segment, SpeakerProfile, SubtitleSegment


def write_speaker_metadata(
    path: Path,
    *,
    profiles: list[SpeakerProfile],
    turns: list[Segment],
    subtitles: list[SubtitleSegment],
) -> None:
    """Write voice profiles and timestamp-to-speaker associations as JSON."""
    payload = {
        "schema_version": 1,
        "profiles": {
            profile.speaker: {
                key: value
                for key, value in asdict(profile).items()
                if key != "speaker"
            }
            for profile in profiles
        },
        "speaker_turns": [asdict(turn) for turn in turns],
        "subtitles": [
            {
                "start": subtitle.start,
                "end": subtitle.end,
                "speaker": subtitle.speaker,
            }
            for subtitle in subtitles
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
