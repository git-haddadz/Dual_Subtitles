from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from dual_subtitles.io.audio import extract_audio


def test_extract_audio_uses_ffmpeg_without_moviepy(
    monkeypatch: Any,
) -> None:
    commands: list[list[str]] = []

    def record_command(command: list[str], **kwargs: Any) -> None:
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
        }
        commands.append(command)

    monkeypatch.setattr(subprocess, "run", record_command)
    video_path = Path("video.mp4")
    output_path = Path("audio.wav")

    result = extract_audio(video_path, output_path)

    assert result == output_path
    assert commands == [
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
    ]


def test_extract_audio_reports_ffmpeg_failure(
    monkeypatch: Any,
) -> None:
    def fail(command: list[str], **_kwargs: Any) -> None:
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="Stream map '0:a:0' matches no streams",
        )

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="matches no streams"):
        extract_audio(Path("silent.mp4"), Path("silent.wav"))
