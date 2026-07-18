"""Continuous transcription and word-level subtitle reconstruction."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import replace
from statistics import fmean

from dual_subtitles.models.subtitle import (
    Segment,
    SubtitleSegment,
    SuspiciousPassage,
    TranscribedWord,
)

WORD_ENDINGS = ("؟", "?", "!", ".", "،", ",", ":", ";", "؛")
ARABIC_LETTER = re.compile(r"[\u0621-\u063A\u0641-\u064A]")
NORMALIZATION_MARKS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
NON_WORD_CHARACTERS = re.compile(r"[^\w\u0621-\u063A\u0641-\u064A]+")
DETACHED_PUNCTUATION = re.compile(r"^[.,!?،؛؟:;…]+$")
LexicalValidator = Callable[[str], bool]
MAX_REPETITION_INTERVAL = 5.0
SUSPICIOUS_GROUP_GAP = 0.6
OVERLAP_TIE_TOLERANCE = 1e-6
MIN_WORD_DURATION = 0.08
SEQUENCE_OVERLAP_LIMIT = 16
SEQUENCE_TIMESTAMP_TOLERANCE = 1.25
TEMPORAL_CONFLICT_IOU = 0.7


def build_audio_windows(
    duration: float,
    silence_boundaries: Iterable[float],
    *,
    target_duration: float,
    max_duration: float,
    overlap: float,
) -> list[Segment]:
    """Build overlapping windows, preferring silence near each target end."""
    if duration <= 0:
        return []

    boundaries = sorted(
        boundary for boundary in silence_boundaries if 0 < boundary < duration
    )
    minimum_duration = min(target_duration * 0.6, target_duration - overlap)
    windows: list[Segment] = []
    start = 0.0

    while start < duration:
        remaining = duration - start
        if remaining <= max_duration:
            end = duration
        else:
            target_end = start + target_duration
            latest_end = min(duration, start + max_duration)
            earliest_end = start + minimum_duration
            candidates = [
                boundary
                for boundary in boundaries
                if earliest_end <= boundary <= latest_end
            ]
            end = (
                min(candidates, key=lambda value: abs(value - target_end))
                if candidates
                else min(target_end, latest_end)
            )

        windows.append(Segment(start=start, end=end))
        if end >= duration:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return windows


def merge_overlapping_words(
    words: Iterable[TranscribedWord],
    *,
    timestamp_tolerance: float = 0.65,
) -> list[TranscribedWord]:
    """Remove duplicate words emitted by overlapping acoustic windows."""
    merged: list[TranscribedWord] = []
    ordered = sorted(words, key=lambda word: (word.start, word.end))
    for word in ordered:
        if word.end <= word.start or not word.text.strip():
            continue
        duplicate_index = _find_duplicate(
            merged,
            word,
            timestamp_tolerance=timestamp_tolerance,
        )
        if duplicate_index is None:
            merged.append(word)
            continue
        existing = merged[duplicate_index]
        if _confidence(word) > _confidence(existing):
            merged[duplicate_index] = word
    return sorted(merged, key=lambda word: (word.start, word.end))


def merge_window_transcripts(
    windows: Iterable[Iterable[TranscribedWord]],
) -> list[TranscribedWord]:
    """Fuse overlapping window sequences using text, time, and confidence."""
    merged: list[TranscribedWord] = []
    for raw_window in windows:
        window = sorted(raw_window, key=lambda word: (word.start, word.end))
        if not window:
            continue
        if not merged:
            merged.extend(window)
            continue

        overlap_size = _longest_sequence_overlap(merged, window)
        if overlap_size:
            first_merged = len(merged) - overlap_size
            for offset in range(overlap_size):
                existing_index = first_merged + offset
                merged[existing_index] = _preferred_word(
                    merged[existing_index],
                    window[offset],
                )
            window = window[overlap_size:]

        for candidate in window:
            conflict = _find_temporal_conflict(merged, candidate)
            if conflict is None:
                merged.append(candidate)
            else:
                merged[conflict] = _preferred_word(merged[conflict], candidate)
        merged.sort(key=lambda word: (word.start, word.end))
    return merge_overlapping_words(merged)


def repair_word_timestamps(
    words: Iterable[TranscribedWord],
    *,
    max_duration: float,
    min_duration: float = MIN_WORD_DURATION,
) -> list[TranscribedWord]:
    """Repair invalid word bounds while preserving their acoustic origin."""
    ordered = sorted(words, key=lambda word: (word.start, word.end))
    repaired: list[TranscribedWord] = []
    for index, word in enumerate(ordered):
        acoustic_start = (
            word.acoustic_start if word.acoustic_start is not None else word.start
        )
        acoustic_end = word.acoustic_end if word.acoustic_end is not None else word.end
        start = max(0.0, word.start)
        end = word.end
        next_start = ordered[index + 1].start if index + 1 < len(ordered) else None

        if end - start > max_duration:
            end = start + max_duration
        if next_start is not None and next_start > start:
            end = min(end, next_start)
        if end - start < min_duration:
            end = start + min_duration
            if next_start is not None and next_start > start:
                end = min(end, next_start)
        if end <= start:
            continue
        repaired.append(
            replace(
                word,
                start=start,
                end=end,
                acoustic_start=acoustic_start,
                acoustic_end=acoustic_end,
            )
        )
    return repaired


def assign_speakers(
    words: Iterable[TranscribedWord],
    speaker_turns: Iterable[Segment],
) -> list[TranscribedWord]:
    """Assign each transcribed word to the best-overlapping speaker turn."""
    turns = sorted(speaker_turns, key=lambda turn: (turn.start, turn.end))
    if not turns:
        return list(words)

    assigned: list[TranscribedWord] = []
    previous_speaker = turns[0].speaker
    for word in words:
        overlaps = [
            (max(0.0, min(word.end, turn.end) - max(word.start, turn.start)), turn)
            for turn in turns
        ]
        overlap = max(item[0] for item in overlaps)
        tied_turns = [
            turn
            for value, turn in overlaps
            if abs(value - overlap) <= OVERLAP_TIE_TOLERANCE
        ]
        best_turn = next(
            (turn for turn in tied_turns if turn.speaker == previous_speaker),
            tied_turns[0],
        )
        if overlap <= 0:
            midpoint = (word.start + word.end) / 2
            best_turn = min(
                turns,
                key=lambda turn: _distance_to_segment(midpoint, turn),
            )
            if _distance_to_segment(midpoint, best_turn) > 1.0:
                assigned.append(replace(word, speaker=previous_speaker))
                continue
        previous_speaker = best_turn.speaker
        assigned.append(replace(word, speaker=best_turn.speaker))
    return assigned


def words_to_subtitles(
    words: Iterable[TranscribedWord],
    *,
    gap_threshold: float,
    max_duration: float,
    max_words: int,
) -> list[SubtitleSegment]:
    """Reconstruct readable subtitle blocks from timestamped words."""
    ordered = [
        word
        for word in sorted(words, key=lambda item: (item.start, item.end))
        if word.end > word.start and word.text.strip()
    ]
    if not ordered:
        return []

    groups: list[list[TranscribedWord]] = []
    current: list[TranscribedWord] = []
    for word in ordered:
        if current and _starts_new_subtitle(
            current,
            word,
            gap_threshold=gap_threshold,
            max_duration=max_duration,
            max_words=max_words,
        ):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)

    return [
        SubtitleSegment(
            start=group[0].start,
            end=max(word.end for word in group),
            text=_join_word_texts(group),
            speaker=group[0].speaker,
        )
        for group in groups
    ]


def repair_subtitle_timestamps(
    segments: Iterable[SubtitleSegment],
    *,
    min_duration: float,
    max_duration: float,
    max_words: int,
    merge_gap: float,
) -> list[SubtitleSegment]:
    """Merge flashing fragments and enforce safe display durations."""
    merged = _merge_short_subtitles(
        list(segments),
        min_duration=min_duration,
        max_duration=max_duration,
        max_words=max_words,
        merge_gap=merge_gap,
    )
    repaired: list[SubtitleSegment] = []
    for index, segment in enumerate(merged):
        start = max(0.0, segment.start)
        end = min(segment.end, start + max_duration)
        next_start = merged[index + 1].start if index + 1 < len(merged) else None
        if end - start < min_duration:
            end = start + min_duration
            if next_start is not None and next_start > start:
                end = min(end, max(start + MIN_WORD_DURATION, next_start - 0.02))
        if end <= start:
            continue
        repaired.append(replace(segment, start=start, end=end))
    return repaired


def detect_suspicious_passages(  # noqa: PLR0912, PLR0913
    words: Iterable[TranscribedWord],
    *,
    confidence_threshold: float,
    log_probability_threshold: float = -1.0,
    compression_ratio_threshold: float = 2.4,
    no_speech_threshold: float = 0.6,
    max_word_duration: float = 3.0,
    lexical_validator: LexicalValidator | None = None,
) -> list[SuspiciousPassage]:
    """Find low-confidence, repeated, or linguistically suspect intervals."""
    ordered = list(words)
    reasons_by_index: dict[int, set[str]] = {}

    for index, word in enumerate(ordered):
        reasons: set[str] = set()
        if word.confidence is not None and word.confidence < confidence_threshold:
            reasons.add("low_confidence")
        if (
            word.average_log_probability is not None
            and word.average_log_probability < log_probability_threshold
        ):
            reasons.add("low_log_probability")
        if (
            word.compression_ratio is not None
            and word.compression_ratio > compression_ratio_threshold
        ):
            reasons.add("high_compression")
        if (
            word.no_speech_probability is not None
            and word.no_speech_probability > no_speech_threshold
        ):
            reasons.add("probable_silence")
        if word.end - word.start > max_word_duration:
            reasons.add("abnormal_duration")
        if word.end <= word.start:
            reasons.add("invalid_timestamp")
        if (
            lexical_validator is not None
            and ARABIC_LETTER.search(word.text)
            and not lexical_validator(word.text)
        ):
            reasons.add("unknown_word")
        if reasons:
            reasons_by_index[index] = reasons

    normalized = [_normalize_word(word.text) for word in ordered]
    for index in range(2, len(ordered)):
        repeated = (
            normalized[index] and len(set(normalized[index - 2 : index + 1])) == 1
        )
        short_interval = (
            ordered[index].end - ordered[index - 2].start <= MAX_REPETITION_INTERVAL
        )
        if repeated and short_interval:
            for repeated_index in range(index - 2, index + 1):
                reasons_by_index.setdefault(repeated_index, set()).add("repetition")

    if not reasons_by_index:
        return []
    return _group_suspicious_words(ordered, reasons_by_index)


def transcript_quality(
    words: Iterable[TranscribedWord],
    *,
    lexical_validator: LexicalValidator | None = None,
) -> float:
    """Return a conservative score used only to compare retry candidates."""
    items = list(words)
    if not items:
        return 0.0
    confidences = [word.confidence for word in items if word.confidence is not None]
    score = fmean(confidences) if confidences else 0.5
    log_probabilities = [
        word.average_log_probability
        for word in items
        if word.average_log_probability is not None
    ]
    if log_probabilities:
        normalized_log_probability = max(
            0.0,
            min(1.0, (fmean(log_probabilities) + 2.0) / 2.0),
        )
        score = (score + normalized_log_probability) / 2

    normalized = [_normalize_word(word.text) for word in items]
    repeated_triples = sum(
        bool(normalized[index]) and len(set(normalized[index - 2 : index + 1])) == 1
        for index in range(2, len(items))
    )
    score -= min(0.3, repeated_triples * 0.12)
    score -= sum(word.end <= word.start for word in items) * 0.2
    score -= min(
        0.25,
        sum(word.no_speech_probability or 0.0 for word in items) / len(items) * 0.25,
    )
    if lexical_validator is not None:
        unknown = sum(
            bool(ARABIC_LETTER.search(word.text)) and not lexical_validator(word.text)
            for word in items
        )
        score -= min(0.25, unknown / len(items) * 0.25)
    return max(0.0, min(1.0, score))


def replace_passage_if_better(
    words: Iterable[TranscribedWord],
    retry_words: Iterable[TranscribedWord],
    passage: SuspiciousPassage,
    *,
    min_improvement: float,
    lexical_validator: LexicalValidator | None = None,
) -> tuple[list[TranscribedWord], bool]:
    """Replace one suspect interval only when its retry is clearly stronger."""
    original = list(words)
    selected_original = _words_in_passage(original, passage)
    selected_retry = _words_in_passage(list(retry_words), passage, tolerance=0.35)
    if not selected_retry:
        return original, False
    old_score = transcript_quality(
        selected_original,
        lexical_validator=lexical_validator,
    )
    new_score = transcript_quality(
        selected_retry,
        lexical_validator=lexical_validator,
    )
    if new_score < old_score + min_improvement:
        return original, False

    retained = [word for word in original if word not in selected_original]
    return merge_overlapping_words([*retained, *selected_retry]), True


def _find_duplicate(
    merged: list[TranscribedWord],
    candidate: TranscribedWord,
    *,
    timestamp_tolerance: float,
) -> int | None:
    normalized_candidate = _normalize_word(candidate.text)
    if not normalized_candidate:
        return None
    for index in range(len(merged) - 1, max(-1, len(merged) - 8), -1):
        existing = merged[index]
        if candidate.start - existing.end > timestamp_tolerance:
            break
        same_text = _normalize_word(existing.text) == normalized_candidate
        same_time = (
            abs(existing.start - candidate.start) <= timestamp_tolerance
            and abs(existing.end - candidate.end) <= timestamp_tolerance
        )
        overlaps = min(existing.end, candidate.end) > max(
            existing.start,
            candidate.start,
        )
        if same_text and (same_time or overlaps):
            return index
    return None


def _longest_sequence_overlap(
    merged: list[TranscribedWord],
    window: list[TranscribedWord],
) -> int:
    """Find the longest matching suffix/prefix near a window boundary."""
    maximum = min(len(merged), len(window), SEQUENCE_OVERLAP_LIMIT)
    for size in range(maximum, 0, -1):
        previous = merged[-size:]
        current = window[:size]
        same_words = all(
            _normalize_word(left.text) == _normalize_word(right.text)
            and bool(_normalize_word(left.text))
            for left, right in zip(previous, current, strict=True)
        )
        if not same_words:
            continue
        centers_are_close = all(
            abs(_word_center(left) - _word_center(right))
            <= SEQUENCE_TIMESTAMP_TOLERANCE
            for left, right in zip(previous, current, strict=True)
        )
        if centers_are_close:
            return size
    return 0


def _find_temporal_conflict(
    merged: list[TranscribedWord],
    candidate: TranscribedWord,
) -> int | None:
    """Locate a competing boundary hypothesis from another audio window."""
    for index in range(len(merged) - 1, max(-1, len(merged) - 12), -1):
        existing = merged[index]
        if candidate.start - existing.end > SEQUENCE_TIMESTAMP_TOLERANCE:
            break
        if (
            candidate.window_index is not None
            and existing.window_index == candidate.window_index
        ):
            continue
        if _normalize_word(existing.text) == _normalize_word(candidate.text):
            continue
        if _temporal_iou(existing, candidate) >= TEMPORAL_CONFLICT_IOU:
            return index
    return None


def _preferred_word(
    left: TranscribedWord,
    right: TranscribedWord,
) -> TranscribedWord:
    """Choose the acoustically stronger duplicate or competing hypothesis."""
    return right if _word_score(right) > _word_score(left) else left


def _word_score(word: TranscribedWord) -> float:
    confidence = word.confidence if word.confidence is not None else 0.5
    log_probability = word.average_log_probability
    normalized_log_probability = (
        max(0.0, min(1.0, (log_probability + 2.0) / 2.0))
        if log_probability is not None
        else 0.5
    )
    no_speech = word.no_speech_probability or 0.0
    compression_penalty = max(0.0, (word.compression_ratio or 0.0) - 2.4)
    return (
        confidence * 0.6
        + normalized_log_probability * 0.4
        - no_speech * 0.25
        - min(0.25, compression_penalty * 0.1)
    )


def _temporal_iou(left: TranscribedWord, right: TranscribedWord) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union > 0 else 0.0


def _word_center(word: TranscribedWord) -> float:
    return (word.start + word.end) / 2


def _merge_short_subtitles(
    segments: list[SubtitleSegment],
    *,
    min_duration: float,
    max_duration: float,
    max_words: int,
    merge_gap: float,
) -> list[SubtitleSegment]:
    """Merge adjacent fragments when either would flash too briefly."""
    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end))
    if not ordered:
        return []

    merged: list[SubtitleSegment] = []
    for segment in ordered:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        combined_words = len(previous.text.split()) + len(segment.text.split())
        can_merge = (
            previous.speaker == segment.speaker
            and segment.start - previous.end <= merge_gap
            and max(previous.end, segment.end) - previous.start <= max_duration
            and combined_words <= max_words
        )
        needs_merge = (
            previous.end - previous.start < min_duration
            or segment.end - segment.start < min_duration
        )
        if can_merge and needs_merge:
            merged[-1] = SubtitleSegment(
                start=previous.start,
                end=max(previous.end, segment.end),
                text=f"{previous.text.rstrip()} {segment.text.lstrip()}",
                speaker=previous.speaker,
            )
        else:
            merged.append(segment)
    return merged


def _starts_new_subtitle(
    current: list[TranscribedWord],
    word: TranscribedWord,
    *,
    gap_threshold: float,
    max_duration: float,
    max_words: int,
) -> bool:
    previous = current[-1]
    return (
        word.speaker != current[0].speaker
        or word.start - previous.end > gap_threshold
        or word.end - current[0].start > max_duration
        or len(current) >= max_words
        or previous.text.rstrip().endswith(WORD_ENDINGS)
    )


def _group_suspicious_words(
    words: list[TranscribedWord],
    reasons_by_index: dict[int, set[str]],
) -> list[SuspiciousPassage]:
    passages: list[SuspiciousPassage] = []
    indexes = sorted(reasons_by_index)
    group = [indexes[0]]
    for index in indexes[1:]:
        previous = group[-1]
        close_to_previous = (
            words[index].start - words[previous].end <= SUSPICIOUS_GROUP_GAP
        )
        if close_to_previous:
            group.append(index)
            continue
        passages.append(_passage_from_indexes(words, group, reasons_by_index))
        group = [index]
    passages.append(_passage_from_indexes(words, group, reasons_by_index))
    return passages


def _passage_from_indexes(
    words: list[TranscribedWord],
    indexes: list[int],
    reasons_by_index: dict[int, set[str]],
) -> SuspiciousPassage:
    reasons = sorted(reason for index in indexes for reason in reasons_by_index[index])
    return SuspiciousPassage(
        start=words[indexes[0]].start,
        end=words[indexes[-1]].end,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _words_in_passage(
    words: list[TranscribedWord],
    passage: SuspiciousPassage,
    *,
    tolerance: float = 0.0,
) -> list[TranscribedWord]:
    start = passage.start - tolerance
    end = passage.end + tolerance
    return [word for word in words if min(word.end, end) > max(word.start, start)]


def _normalize_word(text: str) -> str:
    value = NORMALIZATION_MARKS.sub("", text.casefold())
    return NON_WORD_CHARACTERS.sub("", value)


def _join_word_texts(words: Iterable[TranscribedWord]) -> str:
    text = ""
    for word in words:
        token = word.text.strip()
        if not text or DETACHED_PUNCTUATION.fullmatch(token):
            text += token
        else:
            text += f" {token}"
    return text


def _confidence(word: TranscribedWord) -> float:
    return word.confidence if word.confidence is not None else -1.0


def _distance_to_segment(point: float, segment: Segment) -> float:
    if segment.start <= point <= segment.end:
        return 0.0
    return min(abs(point - segment.start), abs(point - segment.end))
