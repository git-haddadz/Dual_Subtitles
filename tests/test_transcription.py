from dual_subtitles.services.transcription import (
    _decoded_text,
    _generation_token_limit,
    suspicious_transcript_reasons,
)


def test_decoded_text_accepts_processor_string() -> None:
    assert _decoded_text("  مرحبا بكم  ") == "مرحبا بكم"


def test_decoded_text_accepts_processor_list() -> None:
    assert _decoded_text(["مرحبا بكم"]) == "مرحبا بكم"


def test_decoded_text_rejects_empty_result() -> None:
    assert _decoded_text([]) == ""


def test_replacement_character_is_suspicious() -> None:
    reasons = suspicious_transcript_reasons("مرحبا � بكم", duration=2.0)

    assert "replacement-character" in reasons


def test_runaway_laughter_is_suspicious() -> None:
    reasons = suspicious_transcript_reasons("ه" * 80, duration=2.0)

    assert "repeated-character-run" in reasons


def test_normal_arabic_phrase_is_not_suspicious() -> None:
    reasons = suspicious_transcript_reasons(
        "فيها الروحانية العبقرية",
        duration=2.0,
    )

    assert reasons == ()


def test_single_arabic_letter_at_end_is_suspicious() -> None:
    reasons = suspicious_transcript_reasons(
        "لَنْ يَنْفَعَ إنْ لَمْ يَكُنْ م",
        duration=2.2,
    )

    assert "orphan-final-arabic-letter" in reasons


def test_single_diacritized_arabic_letter_at_end_is_suspicious() -> None:
    reasons = suspicious_transcript_reasons("تَ", duration=0.5)

    assert "orphan-final-arabic-letter" in reasons


def test_complete_short_arabic_interjection_is_not_suspicious() -> None:
    reasons = suspicious_transcript_reasons("آه", duration=0.5)

    assert reasons == ()


def test_generation_limit_depends_on_audio_duration() -> None:
    assert _generation_token_limit(0.5, configured_limit=256, retry=False) == 48
    assert _generation_token_limit(2.0, configured_limit=256, retry=False) == 48
    assert _generation_token_limit(2.0, configured_limit=256, retry=True) == 72
    assert _generation_token_limit(30.0, configured_limit=256, retry=False) == 256
