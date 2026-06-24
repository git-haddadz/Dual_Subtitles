"""Speech and subtitle segment processing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from dual_subtitles.models.subtitle import Segment, SubtitleSegment

SENTENCE_ENDINGS = ("\u061f", "?", "!", ".", "\u060c", ",")


def merge_speech_segments(
    segments: Iterable[Segment],
    *,
    min_duration: float,
    merge_gap: float,
) -> list[Segment]:
    """Filter short speech segments and merge adjacent segments per speaker."""
    speech_segments = [
        segment for segment in segments if segment.duration >= min_duration
    ]
    if not speech_segments:
        return []

    merged: list[Segment] = []
    current = speech_segments[0]
    for segment in speech_segments[1:]:
        same_speaker = segment.speaker == current.speaker
        close_enough = segment.start - current.end <= merge_gap
        if same_speaker and close_enough:
            current = replace(current, end=max(current.end, segment.end))
            continue

        merged.append(current)
        current = segment

    merged.append(current)
    return merged


def split_long_segments(
    segments: Iterable[Segment],
    *,
    max_duration: float,
) -> list[Segment]:
    """Split speech segments that exceed the configured maximum duration."""
    final_segments: list[Segment] = []
    for segment in segments:
        start = segment.start
        while segment.end - start > max_duration:
            end = start + max_duration
            final_segments.append(
                Segment(start=start, end=end, speaker=segment.speaker)
            )
            start = end
        final_segments.append(
            Segment(start=start, end=segment.end, speaker=segment.speaker)
        )
    return final_segments


def clean_transcribed_segments(
    segments: Iterable[SubtitleSegment],
) -> list[SubtitleSegment]:
    """Remove invalid transcripts and prevent overlapping timestamps."""
    cleaned: list[SubtitleSegment] = []
    previous_end = 0.0

    for segment in segments:
        start = max(float(segment.start), previous_end)
        end = float(segment.end)
        text = segment.text.strip()
        if end <= start or not text:
            continue

        cleaned.append(
            SubtitleSegment(
                start=start,
                end=end,
                text=text,
                speaker=segment.speaker,
            )
        )
        previous_end = end

    return cleaned


def merge_subtitle_segments(
    segments: Iterable[SubtitleSegment],
    *,
    gap_threshold: float,
    max_duration: float,
    max_words: int,
) -> list[SubtitleSegment]:
    """Merge transcript chunks into readable subtitle blocks."""
    final_subtitles: list[SubtitleSegment] = []
    current: SubtitleSegment | None = None

    for segment in segments:
        if current is None:
            current = segment
            continue

        gap = segment.start - current.end
        duration = current.end - current.start
        should_start_new = (
            gap > gap_threshold
            or duration > max_duration
            or current.speaker != segment.speaker
            or current.text.endswith(SENTENCE_ENDINGS)
            or len(current.text.split()) > max_words
        )

        if should_start_new:
            final_subtitles.append(current)
            current = segment
            continue

        current = SubtitleSegment(
            start=current.start,
            end=segment.end,
            text=f"{current.text} {segment.text}".strip(),
            speaker=current.speaker,
        )

    if current is not None:
        final_subtitles.append(current)

    return final_subtitles


def add_line_breaks(
    segments: Iterable[SubtitleSegment],
    *,
    line_break_words: int,
) -> list[SubtitleSegment]:
    """Split long subtitle text into two lines."""
    formatted: list[SubtitleSegment] = []
    for segment in segments:
        words = segment.text.split()
        if len(words) <= line_break_words:
            formatted.append(segment)
            continue

        midpoint = len(words) // 2
        text = " ".join(words[:midpoint]) + "\n" + " ".join(words[midpoint:])
        formatted.append(replace(segment, text=text))

    return formatted
