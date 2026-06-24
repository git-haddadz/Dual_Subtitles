"""Timestamp formatting and parsing helpers."""

from __future__ import annotations

from datetime import timedelta


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""
    safe_seconds = max(0.0, seconds)
    td = timedelta(seconds=safe_seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds as an ASS timestamp."""
    srt = format_srt_timestamp(seconds)
    hours = int(srt[:2])
    return f"{hours}:{srt[3:5]}:{srt[6:8]}.{srt[9:11]}"


def parse_srt_timestamp(timestamp: str) -> float:
    """Parse an SRT timestamp into seconds."""
    time_part, milliseconds = timestamp.strip().split(",", maxsplit=1)
    hours, minutes, seconds = time_part.split(":", maxsplit=2)
    return (
        int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    )
