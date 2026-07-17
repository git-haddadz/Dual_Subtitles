from pathlib import Path

from dual_subtitles.core import pipeline
from dual_subtitles.core.config import ProcessingConfig


def test_existing_outputs_do_not_load_ml_services(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "subtitles"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "sample.mp4").touch()
    (input_dir / "second.mp4").touch()
    srt_path = output_dir / "sample.srt"
    ass_path = output_dir / "sample.ass"
    srt_path.write_text("existing", encoding="utf-8")
    ass_path.write_text("existing", encoding="utf-8")
    (output_dir / "second.srt").write_text("existing", encoding="utf-8")
    (output_dir / "second.ass").write_text("existing", encoding="utf-8")

    def fail(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("A service was loaded for an already completed video")

    monkeypatch.setattr(pipeline, "WhisperTranscriber", fail)
    monkeypatch.setattr(pipeline, "PyannoteDiarizer", fail)
    monkeypatch.setattr(pipeline, "InterlinearGoogleTranslator", fail)

    config = ProcessingConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        temp_dir=tmp_path / "temp",
    )
    completed: list[tuple[Path, list[Path], Exception | None]] = []

    generated = pipeline.process_directory(
        config,
        video_limit=1,
        on_video_complete=lambda video, outputs, error: completed.append(
            (video, outputs, error)
        ),
    )

    assert generated == [srt_path, ass_path]
    assert completed == [
        (input_dir / "sample.mp4", [srt_path, ass_path], None),
    ]
