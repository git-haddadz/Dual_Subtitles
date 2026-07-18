from __future__ import annotations

from dual_subtitles.core.transcript import (
    assign_speakers,
    build_audio_windows,
    detect_suspicious_passages,
    merge_overlapping_words,
    merge_window_transcripts,
    repair_subtitle_timestamps,
    repair_word_timestamps,
    replace_passage_if_better,
    words_to_subtitles,
)
from dual_subtitles.models.subtitle import Segment, SubtitleSegment, TranscribedWord


def word(  # noqa: PLR0913 - compact fixture factory for independent signals.
    text: str,
    start: float,
    end: float,
    *,
    confidence: float | None = None,
    window_index: int | None = None,
    average_log_probability: float | None = None,
    compression_ratio: float | None = None,
    no_speech_probability: float | None = None,
) -> TranscribedWord:
    return TranscribedWord(
        start=start,
        end=end,
        text=text,
        confidence=confidence,
        window_index=window_index,
        average_log_probability=average_log_probability,
        compression_ratio=compression_ratio,
        no_speech_probability=no_speech_probability,
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


def test_window_fusion_uses_sequence_and_keeps_real_repetition() -> None:
    merged = merge_window_transcripts(
        [
            [
                word("one", 0.0, 0.3, confidence=0.8, window_index=0),
                word("two", 0.4, 0.7, confidence=0.7, window_index=0),
            ],
            [
                word("two", 0.42, 0.72, confidence=0.9, window_index=1),
                word("three", 0.8, 1.1, confidence=0.8, window_index=1),
                word("two", 2.0, 2.3, confidence=0.8, window_index=1),
            ],
        ]
    )

    assert [item.text for item in merged] == ["one", "two", "three", "two"]
    assert merged[1].confidence == 0.9


def test_word_timestamp_repair_clamps_outlier_and_preserves_raw_bounds() -> None:
    repaired = repair_word_timestamps(
        [word("one", 1.0, 8.0), word("two", 2.0, 2.01)],
        max_duration=3.0,
    )

    assert repaired[0].end == 2.0
    assert repaired[0].acoustic_end == 8.0
    assert repaired[1].end == 2.08


def test_short_subtitle_is_merged_without_exceeding_readability_limits() -> None:
    repaired = repair_subtitle_timestamps(
        [
            SubtitleSegment(0.0, 0.3, "one", "SPEAKER_00"),
            SubtitleSegment(0.35, 1.2, "two three", "SPEAKER_00"),
        ],
        min_duration=0.7,
        max_duration=5.0,
        max_words=8,
        merge_gap=0.25,
    )

    assert repaired == [SubtitleSegment(0.0, 1.2, "one two three", "SPEAKER_00")]


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


def test_suspicion_uses_backend_diagnostics() -> None:
    passages = detect_suspicious_passages(
        [
            word(
                "suspect",
                1.0,
                5.0,
                confidence=0.9,
                average_log_probability=-1.2,
                compression_ratio=2.8,
                no_speech_probability=0.8,
            )
        ],
        confidence_threshold=0.45,
    )

    assert set(passages[0].reasons) == {
        "abnormal_duration",
        "high_compression",
        "low_log_probability",
        "probable_silence",
    }


def test_distant_suspicious_words_create_separate_retries() -> None:
    passages = detect_suspicious_passages(
        [
            word("first", 1.0, 1.2, confidence=0.1),
            word("second", 20.0, 20.2, confidence=0.1),
        ],
        confidence_threshold=0.45,
    )

    assert [(passage.start, passage.end) for passage in passages] == [
        (1.0, 1.2),
        (20.0, 20.2),
    ]


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
