from dual_subtitles.core.segmentation import (
    build_speaker_phrase_units,
    split_phrase_subtitle,
)
from dual_subtitles.models.subtitle import Segment, SubtitleSegment


def test_phrase_units_never_cross_speaker_boundaries() -> None:
    units = build_speaker_phrase_units(
        [
            Segment(0.0, 4.0, "SPEAKER_00"),
            Segment(4.0, 8.0, "SPEAKER_01"),
        ],
        [2.0, 6.0],
        min_duration=0.5,
        max_duration=12.0,
        merge_gap=0.2,
    )

    assert units == [
        Segment(0.0, 2.0, "SPEAKER_00"),
        Segment(2.0, 4.0, "SPEAKER_00"),
        Segment(4.0, 6.0, "SPEAKER_01"),
        Segment(6.0, 8.0, "SPEAKER_01"),
    ]


def test_adjacent_turns_from_same_speaker_are_coalesced() -> None:
    units = build_speaker_phrase_units(
        [
            Segment(0.0, 2.0, "SPEAKER_00"),
            Segment(2.1, 4.0, "SPEAKER_00"),
        ],
        [],
        min_duration=0.5,
        max_duration=12.0,
        merge_gap=0.2,
    )

    assert units == [Segment(0.0, 4.0, "SPEAKER_00")]


def test_untranscribable_micro_turn_does_not_split_the_main_speaker() -> None:
    units = build_speaker_phrase_units(
        [
            Segment(0.0, 2.0, "SPEAKER_00"),
            Segment(2.0, 2.05, "SPEAKER_01"),
            Segment(2.05, 4.0, "SPEAKER_00"),
        ],
        [],
        min_duration=0.5,
        max_duration=12.0,
        merge_gap=0.2,
    )

    assert units == [Segment(0.0, 4.0, "SPEAKER_00")]


def test_short_but_transcribable_speaker_turn_is_preserved() -> None:
    units = build_speaker_phrase_units(
        [Segment(1.0, 1.25, "SPEAKER_01")],
        [],
        min_duration=0.5,
        max_duration=12.0,
        merge_gap=0.2,
    )

    assert units == [Segment(1.0, 1.25, "SPEAKER_01")]


def test_long_recognized_text_is_split_inside_its_speaker_turn() -> None:
    subtitles = split_phrase_subtitle(
        SubtitleSegment(
            10.0,
            14.0,
            "واحد اثنان ثلاثة أربعة خمسة ستة",
            "SPEAKER_02",
        ),
        max_words=3,
    )

    assert [subtitle.text for subtitle in subtitles] == [
        "واحد اثنان ثلاثة",
        "أربعة خمسة ستة",
    ]
    assert all(subtitle.speaker == "SPEAKER_02" for subtitle in subtitles)
    assert subtitles[0].start == 10.0
    assert subtitles[-1].end == 14.0


def test_phrase_split_balances_a_short_final_group() -> None:
    subtitles = split_phrase_subtitle(
        SubtitleSegment(
            10.0,
            11.5,
            "one two three four five six seven eight nine ten",
            "SPEAKER_02",
        ),
        max_words=8,
        min_duration=0.7,
    )

    assert [subtitle.text for subtitle in subtitles] == [
        "one two three four five",
        "six seven eight nine ten",
    ]
    assert all(subtitle.end - subtitle.start >= 0.7 for subtitle in subtitles)


def test_phrase_split_does_not_create_unreadable_micro_cues() -> None:
    subtitle = SubtitleSegment(
        10.0,
        10.22,
        "one two three four five six seven eight nine ten eleven twelve",
        "SPEAKER_02",
    )

    assert split_phrase_subtitle(
        subtitle,
        max_words=8,
        min_duration=0.7,
    ) == [subtitle]
