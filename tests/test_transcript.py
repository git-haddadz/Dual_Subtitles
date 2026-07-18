from __future__ import annotations

from dual_subtitles.core.transcript import (
    assign_speakers,
    build_audio_windows,
    detect_suspicious_passages,
    merge_overlapping_words,
    replace_passage_if_better,
    words_to_subtitles,
)
from dual_subtitles.models.subtitle import Segment, TranscribedWord


def word(
    text: str,
    start: float,
    end: float,
    *,
    confidence: float | None = None,
) -> TranscribedWord:
    return TranscribedWord(
        start=start,
        end=end,
        text=text,
        confidence=confidence,
    )


def test_audio_windows_prefer_silence_and_keep_overlap() -> None:
    windows = build_audio_windows(
        70.0,
        [27.5, 56.0],
        target_duration=28.0,
        max_duration=30.0,
        overlap=1.0,
    )

    assert windows == [
        Segment(start=0.0, end=27.5),
        Segment(start=26.5, end=56.0),
        Segment(start=55.0, end=70.0),
    ]


def test_audio_windows_fall_back_to_target_duration() -> None:
    windows = build_audio_windows(
        65.0,
        [],
        target_duration=28.0,
        max_duration=30.0,
        overlap=1.0,
    )

    assert windows == [
        Segment(start=0.0, end=28.0),
        Segment(start=27.0, end=55.0),
        Segment(start=54.0, end=65.0),
    ]


def test_overlap_fusion_removes_only_same_timestamp_occurrence() -> None:
    words = merge_overlapping_words(
        [
            word("نعم", 4.0, 4.4, confidence=0.7),
            word("نعم", 4.05, 4.45, confidence=0.9),
            word("نعم", 5.2, 5.6, confidence=0.8),
        ]
    )

    assert len(words) == 2
    assert words[0].confidence == 0.9
    assert words[1].start == 5.2


def test_speaker_assignment_uses_largest_temporal_overlap() -> None:
    assigned = assign_speakers(
        [word("مرحبا", 1.8, 2.2), word("بكم", 2.3, 2.7)],
        [
            Segment(0.0, 2.0, "SPEAKER_00"),
            Segment(2.0, 4.0, "SPEAKER_01"),
        ],
    )

    assert assigned[0].speaker == "SPEAKER_00"
    assert assigned[1].speaker == "SPEAKER_01"


def test_subtitle_reconstruction_uses_punctuation_gap_and_speaker() -> None:
    words = [
        word("كيف", 0.0, 0.3),
        word("حالك؟", 0.35, 0.8),
        word("أنا", 0.9, 1.2),
        word("بخير", 1.25, 1.7),
        TranscribedWord(1.75, 2.0, "شكرا", speaker="SPEAKER_01"),
        TranscribedWord(3.0, 3.2, "جدا", speaker="SPEAKER_01"),
    ]

    subtitles = words_to_subtitles(
        words,
        gap_threshold=0.4,
        max_duration=5.0,
        max_words=8,
    )

    assert [subtitle.text for subtitle in subtitles] == [
        "كيف حالك؟",
        "أنا بخير",
        "شكرا",
        "جدا",
    ]


def test_subtitle_reconstruction_attaches_detached_punctuation() -> None:
    subtitles = words_to_subtitles(
        [
            word("مرحبا", 0.0, 0.3),
            word("،", 0.3, 0.4),
            word("كيف", 0.5, 0.8),
            word("حالك", 0.9, 1.2),
            word("؟", 1.2, 1.3),
        ],
        gap_threshold=0.4,
        max_duration=5.0,
        max_words=8,
    )

    assert [subtitle.text for subtitle in subtitles] == ["مرحبا،", "كيف حالك؟"]


def test_suspicion_detects_confidence_repetition_and_unknown_word() -> None:
    passages = detect_suspicious_passages(
        [
            word("مرحبا", 0.0, 0.3, confidence=0.9),
            word("العقري", 0.4, 0.8, confidence=0.2),
            word("لا", 1.0, 1.2),
            word("لا", 1.3, 1.5),
            word("لا", 1.6, 1.8),
        ],
        confidence_threshold=0.45,
        lexical_validator=lambda value: value != "العقري",
    )

    assert len(passages) == 1
    assert set(passages[0].reasons) == {
        "low_confidence",
        "repetition",
        "unknown_word",
    }


def test_retry_keeps_original_without_clear_improvement() -> None:
    original = [word("العقري", 1.0, 1.5, confidence=0.6)]
    passage = detect_suspicious_passages(
        original,
        confidence_threshold=0.7,
    )[0]

    selected, replaced = replace_passage_if_better(
        original,
        [word("العبقري", 1.0, 1.5, confidence=0.65)],
        passage,
        min_improvement=0.12,
    )

    assert not replaced
    assert selected == original


def test_retry_replaces_passage_when_acoustic_confidence_is_stronger() -> None:
    original = [word("العقري", 1.0, 1.5, confidence=0.4)]
    passage = detect_suspicious_passages(
        original,
        confidence_threshold=0.45,
    )[0]

    selected, replaced = replace_passage_if_better(
        original,
        [word("العبقري", 1.0, 1.5, confidence=0.9)],
        passage,
        min_improvement=0.12,
    )

    assert replaced
    assert [item.text for item in selected] == ["العبقري"]
