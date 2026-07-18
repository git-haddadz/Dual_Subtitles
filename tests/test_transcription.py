from dual_subtitles.services.transcription import _decoded_text


def test_decoded_text_accepts_processor_string() -> None:
    assert _decoded_text("  مرحبا بكم  ") == "مرحبا بكم"


def test_decoded_text_accepts_processor_list() -> None:
    assert _decoded_text(["مرحبا بكم"]) == "مرحبا بكم"


def test_decoded_text_rejects_empty_result() -> None:
    assert _decoded_text([]) == ""
