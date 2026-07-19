"""Local, non-authoritative acoustic voice profiling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from dual_subtitles.models.subtitle import Segment, SpeakerProfile

MIN_PROFILE_TURN_SECONDS = 0.8
MIN_PROFILE_AUDIO_SECONDS = 2.0
MASCULINE_F0_MAX_HZ = 155.0
FEMININE_F0_MIN_HZ = 185.0
MIN_VOICED_PROBABILITY = 0.5
MIN_VOICED_FRAME_RATIO = 0.2


def analyze_voice_profiles(
    audio_path: Path,
    turns: Iterable[Segment],
    *,
    maximum_seconds: float,
    minimum_confidence: float,
) -> list[SpeakerProfile]:
    """Estimate a perceived voice profile from clean, non-overlapping turns."""
    import librosa
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    ordered = sorted(turns, key=lambda turn: (turn.start, turn.end))
    usable = _non_overlapping_turns(ordered)
    by_speaker: dict[str, list[Segment]] = defaultdict(list)
    for turn in usable:
        if turn.duration >= MIN_PROFILE_TURN_SECONDS:
            by_speaker[turn.speaker].append(turn)

    profiles: list[SpeakerProfile] = []
    for speaker, speaker_turns in sorted(by_speaker.items()):
        samples: list[object] = []
        selected_duration = 0.0
        selected_segments = 0
        for turn in sorted(speaker_turns, key=lambda item: item.duration, reverse=True):
            remaining = maximum_seconds - selected_duration
            if remaining <= 0:
                break
            duration = min(turn.duration, remaining)
            start = max(0, round(turn.start * sample_rate))
            end = min(len(audio), start + round(duration * sample_rate))
            if end <= start:
                continue
            samples.append(audio[start:end])
            selected_duration += (end - start) / sample_rate
            selected_segments += 1

        if not samples or selected_duration < MIN_PROFILE_AUDIO_SECONDS:
            profiles.append(_unknown_profile(speaker, selected_duration, 0))
            continue
        combined = np.concatenate(samples)
        f0, voiced, probabilities = librosa.pyin(
            combined,
            fmin=65.0,
            fmax=350.0,
            sr=sample_rate,
        )
        if probabilities is None:
            probabilities = np.ones_like(f0)
        reliable = f0[
            voiced & (probabilities >= MIN_VOICED_PROBABILITY) & np.isfinite(f0)
        ]
        if reliable.size == 0:
            profiles.append(
                _unknown_profile(speaker, selected_duration, selected_segments)
            )
            continue
        median_f0 = float(np.median(reliable))
        q25_f0, q75_f0 = (float(value) for value in np.percentile(reliable, [25, 75]))
        voiced_frame_ratio = float(reliable.size / max(f0.size, 1))
        label, confidence = _classify_f0(
            median_f0,
            q25_f0=q25_f0,
            q75_f0=q75_f0,
        )
        confidence *= min(1.0, voiced_frame_ratio / 0.5)
        if voiced_frame_ratio < MIN_VOICED_FRAME_RATIO:
            label = "unknown"
        if confidence < minimum_confidence:
            label = "unknown"
        profiles.append(
            SpeakerProfile(
                speaker=speaker,
                perceived_voice_gender=label,
                confidence=round(confidence, 4),
                median_f0_hz=round(median_f0, 2),
                analyzed_duration=round(selected_duration, 3),
                analyzed_segments=selected_segments,
                f0_q25_hz=round(q25_f0, 2),
                f0_q75_hz=round(q75_f0, 2),
                voiced_frame_ratio=round(voiced_frame_ratio, 4),
            )
        )
    return profiles


def _non_overlapping_turns(turns: list[Segment]) -> list[Segment]:
    """Keep turns that do not overlap a different speaker."""
    return [
        turn
        for turn in turns
        if not any(
            other.speaker != turn.speaker
            and other.start < turn.end
            and other.end > turn.start
            for other in turns
        )
    ]


def _classify_f0(
    median_f0: float,
    *,
    q25_f0: float | None = None,
    q75_f0: float | None = None,
) -> tuple[str, float]:
    """Classify only when the central pitch distribution clears a boundary."""
    lower_f0 = median_f0 if q25_f0 is None else q25_f0
    upper_f0 = median_f0 if q75_f0 is None else q75_f0
    if upper_f0 <= MASCULINE_F0_MAX_HZ:
        confidence = min(0.99, 0.7 + (MASCULINE_F0_MAX_HZ - upper_f0) / 150)
        return "masculine", confidence
    if lower_f0 >= FEMININE_F0_MIN_HZ:
        confidence = min(0.99, 0.7 + (lower_f0 - FEMININE_F0_MIN_HZ) / 200)
        return "feminine", confidence
    return "unknown", 0.5


def _unknown_profile(
    speaker: str,
    duration: float,
    segments: int,
) -> SpeakerProfile:
    return SpeakerProfile(
        speaker=speaker,
        perceived_voice_gender="unknown",
        confidence=0.0,
        median_f0_hz=None,
        analyzed_duration=round(duration, 3),
        analyzed_segments=segments,
    )
