"""Speech and subtitle segment processing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from dual_subtitles.models.subtitle import Segment, SubtitleSegment

SENTENCE_ENDINGS = ("\u061f", "?", "!", ".", "\u060c", ",")
MIN_TRANSCRIBABLE_TURN_DURATION = 0.2


def build_speaker_phrase_units(
    turns: Iterable[Segment],
    silence_boundaries: Iterable[float],
    *,
    min_duration: float,
    max_duration: float,
    merge_gap: float,
) -> list[Segment]:
    """Split speaker turns at acoustic pauses into ASR-sized phrase units."""
    ordered = sorted(
        (turn for turn in turns if turn.duration >= MIN_TRANSCRIBABLE_TURN_DURATION),
        key=lambda turn: (turn.start, turn.end),
    )
    coalesced: list[Segment] = []
    for turn in ordered:
        if turn.duration <= 0:
            continue
        if (
            coalesced
            and turn.speaker == coalesced[-1].speaker
            and turn.start - coalesced[-1].end <= merge_gap
        ):
            previous = coalesced[-1]
            coalesced[-1] = Segment(
                start=previous.start,
                end=max(previous.end, turn.end),
                speaker=turn.speaker,
            )
        else:
            coalesced.append(turn)

    pauses = sorted(set(silence_boundaries))
    units: list[Segment] = []
    for turn in coalesced:
        internal_pauses = [
            pause
            for pause in pauses
            if turn.start + min_duration <= pause <= turn.end - min_duration
        ]
        boundaries = [turn.start, *internal_pauses, turn.end]
        pieces = [
            Segment(start=start, end=end, speaker=turn.speaker)
            for start, end in zip(boundaries, boundaries[1:], strict=False)
            if end > start
        ]
        pieces = _merge_short_phrase_units(pieces, min_duration=min_duration)
        for piece in pieces:
            units.extend(_split_phrase_unit(piece, max_duration=max_duration))
    return units


def split_phrase_subtitle(
    subtitle: SubtitleSegment,
    *,
    max_words: int,
) -> list[SubtitleSegment]:
    """Split a long recognized turn without crossing its speaker boundary."""
    words = subtitle.text.split()
    if len(words) <= max_words:
        return [subtitle]
    groups: list[list[str]] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if len(current) >= max_words or word.endswith(SENTENCE_ENDINGS):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    if len(groups) == 1:
        return [subtitle]

    total_weight = sum(len(group) for group in groups)
    duration = subtitle.end - subtitle.start
    cursor = subtitle.start
    result: list[SubtitleSegment] = []
    for index, group in enumerate(groups):
        if index == len(groups) - 1:
            end = subtitle.end
        else:
            end = cursor + duration * len(group) / total_weight
        result.append(
            SubtitleSegment(
                start=cursor,
                end=end,
                text=" ".join(group),
                speaker=subtitle.speaker,
            )
        )
        cursor = end
    return result


def _merge_short_phrase_units(
    units: list[Segment],
    *,
    min_duration: float,
) -> list[Segment]:
    """Keep short interjections by attaching only pause-created fragments."""
    merged: list[Segment] = []
    for unit in units:
        if merged and unit.duration < min_duration:
            previous = merged[-1]
            merged[-1] = Segment(
                start=previous.start,
                end=unit.end,
                speaker=unit.speaker,
            )
        else:
            merged.append(unit)
    if len(merged) > 1 and merged[0].duration < min_duration:
        first = merged.pop(0)
        next_unit = merged[0]
        merged[0] = Segment(
            start=first.start,
            end=next_unit.end,
            speaker=next_unit.speaker,
        )
    return merged


def _split_phrase_unit(unit: Segment, *, max_duration: float) -> list[Segment]:
    """Apply a hard safety limit only when no useful pause was detected."""
    result: list[Segment] = []
    start = unit.start
    while unit.end - start > max_duration:
        result.append(
            Segment(
                start=start,
                end=start + max_duration,
                speaker=unit.speaker,
            )
        )
        start += max_duration
    result.append(Segment(start=start, end=unit.end, speaker=unit.speaker))
    return result


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
        projected_duration = segment.end - current.start
        projected_words = len(f"{current.text} {segment.text}".split())
        should_start_new = (
            gap > gap_threshold
            or projected_duration > max_duration
            or current.speaker != segment.speaker
            or current.text.endswith(SENTENCE_ENDINGS)
            or projected_words > max_words
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


def deduplicate_overlap(previous: str, new: str, *, max_overlap: int = 5) -> str:
    """Remove duplicated word overlap between adjacent transcript fragments."""
    previous_words = previous.split()
    new_words = new.split()
    overlap = min(len(previous_words), len(new_words), max_overlap)
    for size in range(overlap, 0, -1):
        if previous_words[-size:] == new_words[:size]:
            return " ".join(new_words[size:])
    return " ".join(new_words)
