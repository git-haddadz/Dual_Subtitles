from dual_subtitles.services.transcription import _chunks_to_words


def test_chunks_to_words_preserve_global_timestamps_and_confidence() -> None:
    words = _chunks_to_words(
        {
            "chunks": [
                {
                    "text": " في ",
                    "timestamp": (0.2, 0.6),
                    "confidence": 0.8,
                },
                {"text": "البيت", "timestamp": (0.7, 1.1)},
            ]
        },
        offset=10.0,
    )

    assert words[0].text == "في"
    assert words[0].start == 10.2
    assert words[0].end == 10.6
    assert words[0].confidence == 0.8
    assert words[1].confidence is None
