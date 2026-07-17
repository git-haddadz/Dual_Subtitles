from dual_subtitles.services.translation import (
    InterlinearGoogleTranslator,
    deduplicate_overlap,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        return f"translated-{text}"


class FailingClient:
    def translate(self, text: str) -> str:
        _ = text
        raise RuntimeError("boom")


def test_interlinear_translation_uses_literal_words_rtl_order_and_cache() -> None:
    client = FakeClient()
    translator = InterlinearGoogleTranslator(client=client)

    expected = r"hello friend\Ntranslated-friend translated-hello"
    assert translator.interlinear("hello friend") == expected
    assert translator.interlinear("hello friend") == expected
    assert client.calls == ["hello", "friend"]
    assert [
        (pair.source, pair.translation)
        for pair in translator.word_pairs("hello friend")
    ] == [
        ("hello", "translated-hello"),
        ("friend", "translated-friend"),
    ]


def test_interlinear_translation_falls_back_to_source_word() -> None:
    translator = InterlinearGoogleTranslator(client=FailingClient())

    assert translator.interlinear("hello friend") == r"hello friend\Nfriend hello"


def test_deduplicate_overlap() -> None:
    assert deduplicate_overlap("one two three", "two three four") == "four"
    assert deduplicate_overlap("one two", "three four") == "three four"
