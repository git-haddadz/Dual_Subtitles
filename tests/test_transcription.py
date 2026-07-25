from dual_subtitles.services.transcription import (
    _decoded_text,
    _generation_options,
    collapse_repeated_token_runs,
    should_retry_transcript,
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


def test_non_speech_markers_are_terminal() -> None:
    for text in ("@@@فراغ", "@@@موسيقى", "@@@ضحك", "♪ ♪ ♪", "..."):
        reasons = suspicious_transcript_reasons(text, duration=1.0)

        assert "non-speech-marker" in reasons
        assert not should_retry_transcript(reasons)


def test_japanese_script_is_rejected_for_the_arabic_english_model() -> None:
    reasons = suspicious_transcript_reasons(
        "涙をこえて光のシャンソーへ",
        duration=3.0,
    )

    assert "unsupported-script" in reasons
    assert not should_retry_transcript(reasons)


def test_arabic_english_code_switch_is_supported() -> None:
    reasons = suspicious_transcript_reasons(
        "هل الإجابة Yes or No؟",
        duration=2.0,
    )

    assert reasons == ()
    assert suspicious_transcript_reasons("OH", duration=0.5) == ()


def test_arabic_lyrics_with_music_marks_are_not_discarded() -> None:
    reasons = suspicious_transcript_reasons(
        "♪ هذه كلمات عربية ♪",
        duration=2.0,
    )

    assert reasons == ()


def test_repeated_word_run_is_suspicious() -> None:
    reasons = suspicious_transcript_reasons(
        "بس بس بس بس بس بس بس بس",
        duration=0.14,
    )

    assert "repeated-token-run" in reasons
    assert "too-many-words-for-audio" in reasons
    assert should_retry_transcript(reasons)


def test_short_natural_repetition_is_preserved() -> None:
    reasons = suspicious_transcript_reasons(
        "لا لا لا لا لا",
        duration=2.0,
    )

    assert reasons == ()


def test_pathological_token_run_is_collapsed_without_losing_final_punctuation() -> None:
    assert (
        collapse_repeated_token_runs("ها ها ها ها ها ها ها ها. ثم انتهى")
        == "ها ها ها. ثم انتهى"
    )


def test_short_natural_repetition_is_not_collapsed() -> None:
    text = "لا  لا لا لا لا"

    assert collapse_repeated_token_runs(text) == text


def test_word_density_allows_short_interjections_but_rejects_runaway_text() -> None:
    assert suspicious_transcript_reasons("تمام!", duration=0.2) == ()

    reasons = suspicious_transcript_reasons(
        "واحد اثنان ثلاثة أربعة خمسة ستة",
        duration=0.2,
    )

    assert "too-many-words-for-audio" in reasons


def test_generation_uses_the_documented_fixed_budget() -> None:
    assert _generation_options(max_new_tokens=256, retry_reasons=()) == {
        "max_new_tokens": 256
    }


def test_generation_penalties_are_limited_to_repetition_retries() -> None:
    assert _generation_options(
        max_new_tokens=256,
        retry_reasons=("orphan-final-arabic-letter",),
    ) == {"max_new_tokens": 256}
    assert _generation_options(
        max_new_tokens=256,
        retry_reasons=("repeated-token-run",),
    ) == {
        "max_new_tokens": 256,
        "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.15,
        "renormalize_logits": True,
    }
