from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dual_subtitles.core import pipeline
from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.io.subtitle_files import build_annotated_ass
from dual_subtitles.models.subtitle import (
    AnnotatedVideo,
    EntityType,
    Segment,
    SpanKind,
    SubtitleSegment,
)
from dual_subtitles.services.arabic import (
    lexical_tokens,
    reconstruct_from_offsets,
    tokenize_losslessly,
)
from dual_subtitles.services.linguistics import PedagogicalAnnotator

EXPECTED_EVENT_COUNT = 6
MULTI_TOKEN_ENTITY_LENGTH = 2


class FakeTranslator:
    translations = {
        "أنا أحب نيويورك.": "I love New York.",
        "زرت نيويورك أمس.": "I visited New York yesterday.",
        "أنا": "I",
        "أحب": "love",
        "نيويورك": "New York",
        "زرت": "visited",
        "أمس": "yesterday",
    }

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [self.translations.get(text, f"translated-{text}") for text in texts]


class FakeMorphology:
    def analyze(self, words: list[str]) -> list[dict[str, object]]:
        analyses = []
        vocalized = {
            "أنا": "أَنَا",
            "أحب": "أُحِبّ",
            "نيويورك": "نِيُويُورْك",
            "زرت": "زُرْت",
            "أمس": "أَمْس",
        }
        for word in words:
            analyses.append(
                {
                    "lex": word,
                    "root": "فعل",
                    "pos": "noun",
                    "diac": vocalized.get(word, word),
                    "_score": 0.9,
                }
            )
        return analyses


class FakeDiacritizer:
    def diacritize(self, text: str) -> tuple[str, float]:
        replacements = {
            "أنا أحب نيويورك.": "أَنَا أُحِبّ نِيُويُورْك.",
            "زرت نيويورك أمس.": "زُرْت نِيُويُورْك أَمْس.",
        }
        return replacements[text], 0.9


class FakeNer:
    def predict(self, words: list[str]) -> list[str]:
        return ["B-LOC" if word == "نيويورك" else "O" for word in words]


class FakeAligner:
    def align(self, source: list[str], target: list[str]) -> dict[int, tuple[int, ...]]:
        result: dict[int, tuple[int, ...]] = {}
        lowered = [word.lower() for word in target]
        mapping = {
            "أنا": ("i",),
            "أحب": ("love",),
            "زرت": ("visited",),
            "أمس": ("yesterday",),
            "نيويورك": ("new", "york"),
        }
        for source_index, word in enumerate(source):
            wanted = mapping.get(word, ())
            indices = tuple(
                index
                for index, target_word in enumerate(lowered)
                if target_word in wanted
            )
            if indices:
                result[source_index] = indices
        return result


def make_config(tmp_path: Path) -> ProcessingConfig:
    return ProcessingConfig(
        input_dir=tmp_path,
        output_dir=tmp_path,
        temp_dir=tmp_path,
    )


def test_lossless_tokenization_preserves_arabic_text_and_offsets() -> None:
    text = "  أَنَا،  لا أحبُّ 42!\n"
    tokens = tokenize_losslessly(text, 0)

    assert reconstruct_from_offsets(text, tokens) == text
    assert [token.surface for token in lexical_tokens(tokens)] == [
        "أَنَا",
        "لا",
        "أحبُّ",
        "42",
    ]


def test_complete_annotation_uses_context_entities_and_safe_diacritics(
    tmp_path: Path,
) -> None:
    annotator = PedagogicalAnnotator(
        config=make_config(tmp_path),
        translator=FakeTranslator(),
        morphology=FakeMorphology(),
        diacritizer=FakeDiacritizer(),
        ner=FakeNer(),
        aligner=FakeAligner(),
    )
    video = annotator.annotate(
        [
            SubtitleSegment(0, 2, "أنا أحب نيويورك."),
            SubtitleSegment(2.1, 4, "زرت نيويورك أمس."),
        ]
    )

    assert video.subtitles[0].context_indices == (0, 1)
    assert video.subtitles[1].context_indices == (0, 1)
    assert video.subtitles[0].natural_translation == "I love New York."
    assert video.subtitles[0].spans[-1].kind is SpanKind.ENTITY
    assert video.subtitles[0].spans[-1].gloss == "New York"
    first_entity = video.subtitles[0].spans[-1].entity_id
    assert video.subtitles[1].spans[1].entity_id == first_entity
    assert video.subtitles[0].spans[0].display_source == "أَنَا"


def test_diacritization_falls_back_for_changed_base_letters(tmp_path: Path) -> None:
    class InvalidDiacritizer:
        def diacritize(self, text: str) -> tuple[str, float]:
            _ = text
            return "إِنْتَ", 0.99

    annotator = PedagogicalAnnotator(
        config=make_config(tmp_path),
        translator=FakeTranslator(),
        morphology=FakeMorphology(),
        diacritizer=InvalidDiacritizer(),
        ner=FakeNer(),
        aligner=FakeAligner(),
    )
    video = annotator.annotate([SubtitleSegment(0, 1, "أنا")])

    assert video.subtitles[0].spans[0].display_source == "أنا"


def test_annotated_ass_keeps_source_and_gloss_centered(tmp_path: Path) -> None:
    annotator = PedagogicalAnnotator(
        config=make_config(tmp_path),
        translator=FakeTranslator(),
        morphology=FakeMorphology(),
        diacritizer=FakeDiacritizer(),
        ner=FakeNer(),
        aligner=FakeAligner(),
    )
    video = annotator.annotate([SubtitleSegment(0, 2, "أنا أحب نيويورك.")])
    ass = build_annotated_ass(video, make_config(tmp_path))
    lines = [line for line in ass.splitlines() if line.startswith("Dialogue:")]

    assert "Style: SourceWord,Noto Naskh Arabic,64" in ass
    assert "Style: TargetGloss,Noto Sans,28" in ass
    assert len(lines) == EXPECTED_EVENT_COUNT
    positions = [re.search(r"\\pos\((\d+),(\d+)\)", line) for line in lines]
    for index in range(0, len(positions), 2):
        assert positions[index] is not None
        assert positions[index + 1] is not None
        assert positions[index].group(1) == positions[index + 1].group(1)


def test_only_ner_ranges_create_multi_token_entity_spans(tmp_path: Path) -> None:
    class MultiTokenNer:
        def predict(self, _words: list[str]) -> list[str]:
            return ["O", "O", "B-PER", "I-PER"]

    class FourWordTranslator:
        def translate_batch(self, texts: list[str]) -> list[str]:
            return ["I met John Smith" for _ in texts]

    class NoopDiacritizer:
        def diacritize(self, text: str) -> tuple[str, float]:
            return text, 0.1

    annotator = PedagogicalAnnotator(
        config=make_config(tmp_path),
        translator=FourWordTranslator(),
        morphology=FakeMorphology(),
        diacritizer=NoopDiacritizer(),
        ner=MultiTokenNer(),
        aligner=FakeAligner(),
    )
    video = annotator.annotate([SubtitleSegment(0, 1, "أنا قابلت جون سميث")])

    spans = video.subtitles[0].spans
    assert [span.kind for span in spans] == [
        SpanKind.TOKEN,
        SpanKind.TOKEN,
        SpanKind.ENTITY,
    ]
    assert spans[-1].token_end - spans[-1].token_start == MULTI_TOKEN_ENTITY_LENGTH
    assert spans[-1].confidence.ner > 0
    assert video.entities[spans[-1].entity_id or ""].entity_type is EntityType.PERSON


def test_video_finishes_transcription_before_annotation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[str] = []

    class FakeAudio:
        def __len__(self) -> int:
            return 5000

        def __getitem__(self, _key: object) -> FakeAudio:
            return self

        def export(self, _path: Path, **kwargs: str) -> None:
            assert kwargs["format"] == "wav"

    class FakeDiarizer:
        def detect(self, _path: Path) -> list[Segment]:
            return [Segment(0, 1), Segment(2, 3)]

    class FakeTranscriber:
        def transcribe_segment(
            self,
            _path: Path,
            segment: Segment,
            *,
            language: str,
            offset: float,
        ) -> list[SubtitleSegment]:
            _ = (language, offset)
            calls.append("transcribe")
            return [
                SubtitleSegment(
                    segment.start,
                    segment.end,
                    f"word-{segment.start}",
                )
            ]

    class FakeAnnotator:
        def annotate(self, segments: list[SubtitleSegment]) -> AnnotatedVideo:
            assert len(calls) == MULTI_TOKEN_ENTITY_LENGTH
            assert len(segments) == MULTI_TOKEN_ENTITY_LENGTH
            calls.append("annotate")
            return AnnotatedVideo(subtitles=[])

        def report(self, stage: str, current: int, total: int) -> None:
            assert (stage, current, total) == ("ass", 1, 1)

    monkeypatch.setattr(pipeline, "extract_audio", lambda *_args: None)
    monkeypatch.setattr(pipeline, "normalize_audio", lambda *_args: None)
    monkeypatch.setattr(pipeline, "load_audio", lambda *_args: FakeAudio())
    monkeypatch.setattr(pipeline, "write_annotated_ass", lambda *_args: None)

    video = tmp_path / "sample.mp4"
    video.touch()
    config = ProcessingConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
        skip_existing=False,
        use_diarization=True,
    )
    config.output_dir.mkdir()
    config.temp_dir.mkdir()
    pipeline.process_video(
        video,
        config=config,
        transcriber=FakeTranscriber(),  # type: ignore[arg-type]
        diarizer=FakeDiarizer(),  # type: ignore[arg-type]
        annotator=FakeAnnotator(),  # type: ignore[arg-type]
    )

    assert calls == ["transcribe", "transcribe", "annotate"]
