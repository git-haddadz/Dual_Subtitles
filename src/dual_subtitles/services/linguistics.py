"""Offline Arabic annotation pipeline used before pedagogical ASS rendering."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any, Protocol

from dual_subtitles.core.config import ProcessingConfig
from dual_subtitles.models.subtitle import (
    AnnotatedSpan,
    AnnotatedSubtitle,
    AnnotatedToken,
    AnnotatedVideo,
    ConfidenceBreakdown,
    EntityRecord,
    EntityType,
    SpanKind,
    SubtitleSegment,
)
from dual_subtitles.services.arabic import (
    consonantal_key,
    contains_arabic,
    lexical_tokens,
    reconstruct_from_offsets,
    strip_diacritics,
    tokenize_losslessly,
    transliterate_arabic,
)
from dual_subtitles.services.translation import NaturalTranslator

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int], None]
ENTITY_SIMILARITY_THRESHOLD = 0.88

TARGET_TOKEN_PATTERN = re.compile(r"\w+(?:['’-]\w+)*", re.UNICODE)
MAX_GLOSS_CHARACTERS = 40
MAX_GLOSS_WORDS = 4
MIN_REPETITION_CHECK_WORDS = 3
MIN_GLOSS_UNIQUENESS_RATIO = 0.6
ENTITY_TARGET_STOPWORDS = {
    "a",
    "an",
    "he",
    "her",
    "him",
    "i",
    "it",
    "she",
    "the",
    "they",
    "we",
    "you",
}


class MorphologyBackend(Protocol):
    """Sentence-level morphology provider."""

    def analyze(self, words: list[str]) -> list[dict[str, Any]]:
        """Return one best analysis for every word."""


class NerBackend(Protocol):
    """BIO named-entity provider."""

    def predict(self, words: list[str]) -> list[str]:
        """Return one BIO tag per word."""


class DiacritizationBackend(Protocol):
    """Sentence diacritization provider."""

    def diacritize(self, text: str) -> tuple[str, float]:
        """Return vocalized text and a model confidence."""


class AlignmentBackend(Protocol):
    """Contextual source-target word aligner."""

    def align(self, source: list[str], target: list[str]) -> dict[int, tuple[int, ...]]:
        """Map source positions to target positions."""


@dataclass(slots=True)
class CamelMorphologyBackend:
    """Lazy CAMeL MLE morphology disambiguator."""

    model_name: str = "auto"
    _disambiguators: list[Any] = field(default_factory=list, init=False, repr=False)

    def _load(self) -> None:
        if self._disambiguators:
            return
        from camel_tools.disambig.mle import MLEDisambiguator

        model_names = (
            ["calima-msa-r13", "calima-egy-r13"]
            if self.model_name == "auto"
            else [self.model_name]
        )
        for model_name in model_names:
            try:
                self._disambiguators.append(MLEDisambiguator.pretrained(model_name))
            except Exception as exc:  # noqa: BLE001 - optional dialect data.
                LOGGER.warning(
                    "CAMeL morphology model %s is unavailable: %s",
                    model_name,
                    exc,
                )
        if not self._disambiguators:
            msg = "No configured CAMeL morphology model is available."
            raise RuntimeError(msg)

    def analyze(self, words: list[str]) -> list[dict[str, Any]]:
        """Return CAMeL analyses without modifying source words."""
        if not words:
            return []
        self._load()
        candidates = [
            disambiguator.disambiguate(words) for disambiguator in self._disambiguators
        ]
        results = max(candidates, key=_sentence_analysis_score)
        analyses: list[dict[str, Any]] = []
        for result in results:
            if not result.analyses:
                analyses.append({})
                continue
            best = result.analyses[0]
            analysis = dict(best.analysis)
            analysis["_score"] = float(getattr(best, "score", 0.0))
            analyses.append(analysis)
        return analyses


@dataclass(slots=True)
class CamelNerBackend:
    """Lazy AraBERT NER model distributed with CAMeL Tools."""

    model_name: str = "arabert"
    use_gpu: bool = False
    _recognizer: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._recognizer is not None:
            return
        from camel_tools.ner import NERecognizer

        self._recognizer = NERecognizer.pretrained(
            model_name=self.model_name,
            use_gpu=self.use_gpu,
        )

    def predict(self, words: list[str]) -> list[str]:
        """Return BIO labels for a tokenized sentence."""
        if not words:
            return []
        self._load()
        return list(self._recognizer.predict_sentence(words))

    def release(self) -> None:
        """Release the NER model before another GPU stage."""
        self._recognizer = None
        _empty_cuda_cache()


@dataclass(slots=True)
class CattDiacritizationBackend:
    """Lazy local CATT encoder-only diacritizer."""

    model_name: str = "TigreGotico/catt-diacritizer"
    _model: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        from catt_tashkeel import CATTEncoderOnly

        self._model = CATTEncoderOnly()

    def diacritize(self, text: str) -> tuple[str, float]:
        """Diacritize with CATT and return a conservative model confidence."""
        if not contains_arabic(text):
            return text, 1.0
        self._load()
        result = self._model.do_tashkeel(text, verbose=False)
        # CATT's public wrapper does not expose logits. Confidence remains
        # conservative and is increased only when morphology agrees later.
        return str(result), 0.72

    def release(self) -> None:
        """Release the ONNX session before another GPU stage."""
        self._model = None


@dataclass(slots=True)
class ContextualWordAligner:
    """Mutual-nearest contextual aligner using multilingual BERT embeddings."""

    model_name: str = "bert-base-multilingual-cased"
    device: int | str | None = None
    similarity_threshold: float = 0.40
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _torch_device: str = field(default="cpu", init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(self.device, int):
            self._torch_device = f"cuda:{self.device}"
        else:
            self._torch_device = str(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self._torch_device)
        self._model.eval()

    def _word_vectors(self, words: list[str]) -> Any:
        import torch

        assert self._model is not None
        assert self._tokenizer is not None
        encoded = self._tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        word_ids = encoded.word_ids()
        inputs = {key: value.to(self._torch_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self._model(**inputs, output_hidden_states=True)
        hidden = torch.stack(output.hidden_states[-4:]).mean(dim=0)[0]
        vectors = []
        for word_index in range(len(words)):
            positions = [
                index for index, value in enumerate(word_ids) if value == word_index
            ]
            if positions:
                vectors.append(hidden[positions].mean(dim=0))
            else:
                vectors.append(torch.zeros(hidden.shape[-1], device=hidden.device))
        return torch.stack(vectors)

    def align(self, source: list[str], target: list[str]) -> dict[int, tuple[int, ...]]:
        """Align only high-confidence mutual-nearest lexical words."""
        if not source or not target:
            return {}
        self._load()
        import torch

        source_vectors = torch.nn.functional.normalize(
            self._word_vectors(source),
            dim=1,
        )
        target_vectors = torch.nn.functional.normalize(
            self._word_vectors(target),
            dim=1,
        )
        similarities = source_vectors @ target_vectors.T
        source_best = similarities.argmax(dim=1)
        target_best = similarities.argmax(dim=0)
        alignments: dict[int, list[int]] = {}
        for source_index, target_index_tensor in enumerate(source_best):
            target_index = int(target_index_tensor)
            score = float(similarities[source_index, target_index])
            if score < self.similarity_threshold:
                continue
            if int(target_best[target_index]) == source_index:
                alignments[source_index] = [target_index]
        return {index: tuple(values) for index, values in alignments.items()}

    def release(self) -> None:
        """Release mBERT after all subtitle alignments are available."""
        self._model = None
        self._tokenizer = None
        _empty_cuda_cache()


@dataclass(slots=True)
class PedagogicalAnnotator:
    """Annotate a complete transcript before any ASS rendering starts."""

    config: ProcessingConfig
    translator: NaturalTranslator
    morphology: MorphologyBackend | None = None
    ner: NerBackend | None = None
    diacritizer: DiacritizationBackend | None = None
    aligner: AlignmentBackend | None = None
    progress: ProgressCallback | None = None

    def __post_init__(self) -> None:
        """Create default local backends only when callers did not inject fakes."""
        if self.morphology is None:
            self.morphology = CamelMorphologyBackend(self.config.morphology_model)
        if self.ner is None:
            ner_model = (
                "arabert"
                if self.config.ner_model == "camel_tools"
                else (self.config.ner_model)
            )
            self.ner = CamelNerBackend(
                model_name=ner_model,
                use_gpu=self._gpu_enabled(),
            )
        if self.diacritizer is None:
            self.diacritizer = CattDiacritizationBackend(
                self.config.diacritization_model
            )
        if self.aligner is None:
            self.aligner = ContextualWordAligner(
                self.config.alignment_model,
                self.config.device,
            )

    def _gpu_enabled(self) -> bool:
        if self.config.device in {"cpu", -1}:
            return False
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def _report(self, stage: str, current: int, total: int) -> None:
        LOGGER.info("%s %s/%s", stage, current, total)
        if self.progress is not None:
            self.progress(stage, current, total)

    def report(self, stage: str, current: int, total: int) -> None:
        """Publish an orchestration stage through the configured callback."""
        self._report(stage, current, total)

    def annotate(self, segments: list[SubtitleSegment]) -> AnnotatedVideo:
        """Run all local stages over a complete video's subtitles."""
        subtitles = [
            AnnotatedSubtitle(
                segment=segment,
                subtitle_index=index,
                tokens=tokenize_losslessly(segment.text, index),
            )
            for index, segment in enumerate(segments)
        ]
        self._build_contexts(subtitles)
        self._translate(subtitles)
        _release_backend(self.translator)
        self._annotate_morphology(subtitles)
        self._diacritize(subtitles)
        _release_backend(self.diacritizer)
        entity_ranges = self._recognize_entities(subtitles)
        _release_backend(self.ner)
        self._align_subtitles(subtitles)
        _release_backend(self.aligner)
        entity_ranges = self._validate_entity_ranges(subtitles, entity_ranges)
        video = AnnotatedVideo(subtitles=subtitles)
        self._build_entity_memory(video, entity_ranges)
        self._build_glosses(video, entity_ranges)
        _release_backend(self.translator)
        self._validate(video)
        return video

    def _build_contexts(self, subtitles: list[AnnotatedSubtitle]) -> None:
        for index, subtitle in enumerate(subtitles):
            candidates = range(
                max(0, index - self.config.context_before),
                min(len(subtitles), index + self.config.context_after + 1),
            )
            accepted = [candidate for candidate in candidates if candidate == index]
            budget = len(lexical_tokens(subtitle.tokens))
            for candidate in sorted(candidates, key=lambda item: abs(item - index)):
                if candidate == index:
                    continue
                left = min(candidate, index)
                right = max(candidate, index)
                gap = subtitles[right].segment.start - subtitles[left].segment.end
                candidate_size = len(lexical_tokens(subtitles[candidate].tokens))
                if gap > self.config.context_max_gap:
                    continue
                if budget + candidate_size > self.config.context_max_tokens:
                    continue
                accepted.append(candidate)
                budget += candidate_size
            subtitle.context_indices = tuple(sorted(accepted))

    def _translate(self, subtitles: list[AnnotatedSubtitle]) -> None:
        texts = [subtitle.segment.text.replace("\n", " ") for subtitle in subtitles]
        translations = self.translator.translate_batch(texts)
        for index, (subtitle, translation) in enumerate(
            zip(subtitles, translations, strict=True),
            start=1,
        ):
            subtitle.natural_translation = translation
            subtitle.target_tokens = TARGET_TOKEN_PATTERN.findall(translation)
            self._report("translation", index, len(subtitles))

    def _annotate_morphology(self, subtitles: list[AnnotatedSubtitle]) -> None:
        assert self.morphology is not None
        for index, subtitle in enumerate(subtitles, start=1):
            words = lexical_tokens(subtitle.tokens)
            context_words, current_start = _source_context(subtitles, subtitle)
            try:
                context_analyses = self.morphology.analyze(context_words)
                analyses = context_analyses[current_start : current_start + len(words)]
            except Exception as exc:  # noqa: BLE001 - optional model data can fail.
                LOGGER.warning("Morphology unavailable: %s", exc)
                analyses = [{} for _ in words]
                subtitle.warnings.append("morphology_unavailable")
            for token, analysis in zip(words, analyses, strict=False):
                score = _bounded_score(float(analysis.get("_score", 0.0)))
                token.lemma = _clean_feature(analysis.get("lex"))
                token.root = _clean_feature(analysis.get("root"))
                token.part_of_speech = _clean_feature(analysis.get("pos"))
                token.features = {
                    key: str(value)
                    for key, value in analysis.items()
                    if key
                    in {
                        "asp",
                        "cas",
                        "enc0",
                        "form_gen",
                        "form_num",
                        "mod",
                        "per",
                        "prc0",
                        "prc1",
                        "prc2",
                        "prc3",
                        "stt",
                        "vox",
                    }
                    and value not in {None, "na", "0"}
                }
                token.confidence = replace(token.confidence, morphology=score)
                diac = _clean_feature(analysis.get("diac"))
                if diac and strip_diacritics(diac) == strip_diacritics(token.surface):
                    token.features["morphological_diacritization"] = diac
            self._report("morphology", index, len(subtitles))

    def _diacritize(self, subtitles: list[AnnotatedSubtitle]) -> None:
        assert self.diacritizer is not None
        for index, subtitle in enumerate(subtitles, start=1):
            try:
                vocalized, sentence_confidence = self.diacritizer.diacritize(
                    subtitle.segment.text.replace("\n", " ")
                )
                predicted = tokenize_losslessly(vocalized, subtitle.subtitle_index)
                predicted_words = lexical_tokens(predicted)
            except Exception as exc:  # noqa: BLE001 - preserve undiacritized text.
                LOGGER.warning("Diacritization unavailable: %s", exc)
                predicted_words = []
                sentence_confidence = 0.0
                subtitle.warnings.append("diacritization_unavailable")
            words = lexical_tokens(subtitle.tokens)
            for word_index, token in enumerate(words):
                candidate = (
                    predicted_words[word_index].surface
                    if word_index < len(predicted_words)
                    else ""
                )
                morphology_candidate = token.features.get(
                    "morphological_diacritization",
                    "",
                )
                valid = strip_diacritics(candidate) == strip_diacritics(token.surface)
                agreement = bool(
                    morphology_candidate and morphology_candidate == candidate
                )
                confidence = min(
                    1.0,
                    sentence_confidence + (0.12 if agreement else 0.0),
                )
                token.confidence = replace(
                    token.confidence,
                    diacritization=confidence if valid else 0.0,
                )
                if valid and confidence >= self.config.diacritization_min_confidence:
                    token.vocalized_surface = _remove_uncertain_case_ending(
                        candidate,
                        token.features,
                    )
                elif candidate:
                    token.warnings.append("diacritization_below_threshold")
            self._report("diacritization", index, len(subtitles))

    def _recognize_entities(
        self,
        subtitles: list[AnnotatedSubtitle],
    ) -> dict[int, list[tuple[int, int, EntityType]]]:
        assert self.ner is not None
        ranges: dict[int, list[tuple[int, int, EntityType]]] = {}
        for index, subtitle in enumerate(subtitles, start=1):
            words = lexical_tokens(subtitle.tokens)
            context_words, current_start = _source_context(subtitles, subtitle)
            try:
                context_tags = self.ner.predict(context_words)
                tags = context_tags[current_start : current_start + len(words)]
            except Exception as exc:  # noqa: BLE001 - NER must not block output.
                LOGGER.warning("NER unavailable: %s", exc)
                tags = ["O"] * len(words)
                subtitle.warnings.append("ner_unavailable")
            ranges[subtitle.subtitle_index] = _bio_ranges(words, tags)
            for start, end, entity_type in ranges[subtitle.subtitle_index]:
                for token in words[start:end]:
                    token.entity_type = entity_type
                    token.confidence = replace(token.confidence, ner=0.75)
            self._report("ner", index, len(subtitles))
        return ranges

    def _build_entity_memory(
        self,
        video: AnnotatedVideo,
        entity_ranges: dict[int, list[tuple[int, int, EntityType]]],
    ) -> None:
        records: list[EntityRecord] = []
        for subtitle in video.subtitles:
            words = lexical_tokens(subtitle.tokens)
            for start, end, entity_type in entity_ranges[subtitle.subtitle_index]:
                entity_tokens = words[start:end]
                source = " ".join(token.surface for token in entity_tokens)
                key = consonantal_key(source)
                record = _matching_entity(records, key, entity_type)
                if record is None:
                    entity_id = f"entity:{len(records)}"
                    target_indices = sorted(
                        {
                            target_index
                            for word_index in range(start, end)
                            for target_index in subtitle.alignments.get(
                                word_index,
                                (),
                            )
                        }
                    )
                    aligned_name = " ".join(
                        subtitle.target_tokens[target_index]
                        for target_index in target_indices
                    )
                    target_name = (
                        aligned_name
                        if _is_plausible_entity_target(aligned_name)
                        else ""
                    ) or transliterate_arabic(
                        " ".join(token.display_surface for token in entity_tokens)
                    )
                    record = EntityRecord(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        canonical_source=source,
                        canonical_target=target_name or source,
                        aliases={key},
                        confidence=0.75,
                    )
                    records.append(record)
                    video.entities[entity_id] = record
                else:
                    record.aliases.add(key)
                for token in entity_tokens:
                    token.entity_id = record.entity_id
                    token.transliteration = record.canonical_target

    def _build_glosses(
        self,
        video: AnnotatedVideo,
        entity_ranges: dict[int, list[tuple[int, int, EntityType]]],
    ) -> None:
        subtitles = video.subtitles
        for position, subtitle in enumerate(subtitles, start=1):
            words = lexical_tokens(subtitle.tokens)
            alignment = subtitle.alignments

            fallback_candidates: dict[int, list[str]] = {}
            lexical_inputs: list[str] = []
            for word_index, token in enumerate(words):
                if token.entity_id is not None:
                    continue
                candidates = list(
                    dict.fromkeys(
                        candidate
                        for candidate in (token.surface, token.lemma)
                        if candidate
                    )
                )
                fallback_candidates[word_index] = candidates
                lexical_inputs.extend(candidates)
            lexical_outputs = _translate_lexical_batch(
                self.translator,
                lexical_inputs,
            )
            translated_candidates: dict[str, str] = dict(
                zip(lexical_inputs, lexical_outputs, strict=True)
            )
            for word_index, token in enumerate(words):
                target_indices = alignment.get(word_index, ())
                token.target_indices = target_indices
                if token.entity_id is not None:
                    entity = video.entities[token.entity_id]
                    token.gloss = entity.canonical_target
                    gloss_confidence = entity.confidence
                elif target_indices:
                    aligned_gloss = _sanitize_gloss(
                        " ".join(
                            subtitle.target_tokens[index] for index in target_indices
                        )
                    )
                    if aligned_gloss:
                        token.gloss = aligned_gloss
                        gloss_confidence = 0.78
                    else:
                        token.target_indices = ()
                        token.gloss = _fallback_gloss(
                            token,
                            fallback_candidates.get(word_index, []),
                            translated_candidates,
                        )
                        gloss_confidence = 0.48
                else:
                    token.gloss = _fallback_gloss(
                        token,
                        fallback_candidates.get(word_index, []),
                        translated_candidates,
                    )
                    gloss_confidence = 0.48
                confidence_with_alignment = replace(
                    token.confidence,
                    alignment=0.75 if target_indices else 0.25,
                    gloss=gloss_confidence,
                )
                token.confidence = replace(
                    confidence_with_alignment,
                    overall=_overall_confidence(
                        confidence_with_alignment,
                        gloss_confidence,
                    ),
                )
            subtitle.spans = _build_spans(
                subtitle,
                entity_ranges[subtitle.subtitle_index],
                video.entities,
            )
            self._report("glosses", position, len(subtitles))

    def _align_subtitles(self, subtitles: list[AnnotatedSubtitle]) -> None:
        assert self.aligner is not None
        for position, subtitle in enumerate(subtitles, start=1):
            words = lexical_tokens(subtitle.tokens)
            try:
                raw_alignment = self.aligner.align(
                    [token.surface for token in words],
                    subtitle.target_tokens,
                )
                used_targets: set[int] = set()
                for source_index in range(len(words)):
                    targets = tuple(
                        dict.fromkeys(
                            target_index
                            for target_index in raw_alignment.get(source_index, ())
                            if 0 <= target_index < len(subtitle.target_tokens)
                            and target_index not in used_targets
                        )
                    )
                    if targets:
                        subtitle.alignments[source_index] = targets
                        used_targets.update(targets)
            except Exception as exc:  # noqa: BLE001 - lexical MT remains available.
                LOGGER.warning("Contextual alignment unavailable: %s", exc)
                subtitle.warnings.append("alignment_unavailable")
            self._report("alignment", position, len(subtitles))

    def _validate_entity_ranges(
        self,
        subtitles: list[AnnotatedSubtitle],
        entity_ranges: dict[int, list[tuple[int, int, EntityType]]],
    ) -> dict[int, list[tuple[int, int, EntityType]]]:
        """Reject isolated NER spans that have no credible proper-name anchor."""
        frequencies: dict[tuple[str, EntityType], int] = {}
        for subtitle in subtitles:
            words = lexical_tokens(subtitle.tokens)
            for start, end, entity_type in entity_ranges[subtitle.subtitle_index]:
                key = consonantal_key(
                    " ".join(token.surface for token in words[start:end])
                )
                frequency_key = (key, entity_type)
                frequencies[frequency_key] = frequencies.get(frequency_key, 0) + 1

        validated: dict[int, list[tuple[int, int, EntityType]]] = {}
        for subtitle in subtitles:
            words = lexical_tokens(subtitle.tokens)
            accepted: list[tuple[int, int, EntityType]] = []
            for entity_range in entity_ranges[subtitle.subtitle_index]:
                start, end, entity_type = entity_range
                target_indices = sorted(
                    {
                        target_index
                        for word_index in range(start, end)
                        for target_index in subtitle.alignments.get(word_index, ())
                    }
                )
                aligned_name = " ".join(
                    subtitle.target_tokens[index] for index in target_indices
                )
                key = consonantal_key(
                    " ".join(token.surface for token in words[start:end])
                )
                corroborated = frequencies.get((key, entity_type), 0) > 1
                if _is_plausible_entity_target(aligned_name) or corroborated:
                    accepted.append(entity_range)
                    continue
                for token in words[start:end]:
                    token.entity_type = None
                    token.confidence = replace(token.confidence, ner=0.0)
                    token.warnings.append("uncorroborated_entity")
                subtitle.warnings.append("uncorroborated_entity")
            validated[subtitle.subtitle_index] = accepted
        return validated

    def _validate(self, video: AnnotatedVideo) -> None:
        for position, subtitle in enumerate(video.subtitles, start=1):
            if reconstruct_from_offsets(subtitle.segment.text, subtitle.tokens) != (
                subtitle.segment.text
            ):
                msg = (
                    "Lossless tokenization failed for subtitle "
                    f"{subtitle.subtitle_index}"
                )
                raise ValueError(msg)
            lexical_count = len(lexical_tokens(subtitle.tokens))
            covered: set[int] = set()
            previous_end = 0
            for span in subtitle.spans:
                if (
                    span.token_start < previous_end
                    or span.token_end <= span.token_start
                ):
                    msg = f"Invalid overlapping span {span.span_id}"
                    raise ValueError(msg)
                previous_end = span.token_end
                covered.update(range(span.token_start, span.token_end))
                if not span.gloss.strip():
                    msg = f"Missing gloss for {span.span_id}"
                    raise ValueError(msg)
            if covered != set(range(lexical_count)):
                msg = (
                    "Incomplete lexical coverage for subtitle "
                    f"{subtitle.subtitle_index}"
                )
                raise ValueError(msg)
            self._report("validation", position, len(video.subtitles))


def _clean_feature(value: Any) -> str | None:
    if value in {None, "", "NOAN", "na"}:
        return None
    return str(value).split("_")[0]


def _sentence_analysis_score(results: list[Any]) -> float:
    scores = [float(result.analyses[0].score) for result in results if result.analyses]
    if not scores:
        return float("-inf")
    return sum(scores) / len(scores)


def _release_backend(backend: object) -> None:
    release = getattr(backend, "release", None)
    if callable(release):
        release()


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _source_context(
    subtitles: list[AnnotatedSubtitle],
    current: AnnotatedSubtitle,
) -> tuple[list[str], int]:
    words: list[str] = []
    current_start = 0
    for subtitle_index in current.context_indices:
        if subtitle_index == current.subtitle_index:
            current_start = len(words)
        words.extend(
            token.surface for token in lexical_tokens(subtitles[subtitle_index].tokens)
        )
    return words, current_start


def _target_context(
    subtitles: list[AnnotatedSubtitle],
    current: AnnotatedSubtitle,
) -> tuple[list[str], int]:
    words: list[str] = []
    current_start = 0
    for subtitle_index in current.context_indices:
        if subtitle_index == current.subtitle_index:
            current_start = len(words)
        words.extend(subtitles[subtitle_index].target_tokens)
    return words, current_start


def _bounded_score(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if 0 <= value <= 1:
        return value
    return 1 / (1 + math.exp(-value))


def _remove_uncertain_case_ending(text: str, features: dict[str, str]) -> str:
    case = features.get("cas")
    if not case or case not in {"u", "a", "i", "n", "g"}:
        return text
    # Only marks on the final base character are removed. Internal lexical
    # vowels, shadda and sukun remain untouched.
    return re.sub(r"([\u0621-\u064a])[\u064b-\u0650]+$", r"\1", text)


def _bio_ranges(
    words: list[AnnotatedToken],
    tags: list[str],
) -> list[tuple[int, int, EntityType]]:
    del words
    ranges: list[tuple[int, int, EntityType]] = []
    current_start: int | None = None
    current_type: EntityType | None = None
    for index, raw_tag in enumerate([*tags, "O"]):
        prefix, _, label = raw_tag.partition("-")
        entity_type = _entity_type(label) if prefix in {"B", "I"} else None
        starts_new = prefix == "B" or (prefix == "I" and entity_type != current_type)
        if current_start is not None and (entity_type != current_type or starts_new):
            assert current_type is not None
            ranges.append((current_start, index, current_type))
            current_start = None
            current_type = None
        if entity_type is not None and current_start is None:
            current_start = index
            current_type = entity_type
    return ranges


def _entity_type(label: str) -> EntityType:
    normalized = label.upper()
    if normalized in {"PER", "PERSON"}:
        return EntityType.PERSON
    if normalized in {"LOC", "LOCATION", "GPE"}:
        return EntityType.LOCATION
    if normalized in {"ORG", "ORGANIZATION"}:
        return EntityType.ORGANIZATION
    return EntityType.MISC


def _matching_entity(
    records: list[EntityRecord],
    key: str,
    entity_type: EntityType,
) -> EntityRecord | None:
    for record in records:
        if record.entity_type is not entity_type:
            continue
        if key in record.aliases:
            return record
        if any(
            SequenceMatcher(None, key, alias).ratio() >= ENTITY_SIMILARITY_THRESHOLD
            for alias in record.aliases
        ):
            return record
    return None


def _is_plausible_entity_target(text: str) -> bool:
    """Accept aligned Latin proper names, not generic labels or stopwords."""
    if not text or "_" in text:
        return False
    words = TARGET_TOKEN_PATTERN.findall(text)
    if not words or any(not re.search(r"[A-Za-z]", word) for word in words):
        return False
    if all(word.casefold() in ENTITY_TARGET_STOPWORDS for word in words):
        return False
    significant = [
        word for word in words if word.casefold() not in ENTITY_TARGET_STOPWORDS
    ]
    return bool(significant) and all(word[0].isupper() for word in significant)


def _translate_lexical_batch(
    translator: NaturalTranslator,
    words: list[str],
) -> list[str]:
    """Use bounded lexical decoding while retaining injected-client support."""
    translate_words = getattr(translator, "translate_words", None)
    if callable(translate_words):
        return list(translate_words(words))
    return translator.translate_batch(words)


def _sanitize_gloss(value: str, *, max_words: int = MAX_GLOSS_WORDS) -> str:
    """Keep a short lexical gloss and reject generation artefacts."""
    if not value or r"\N" in value or "\n" in value or "\r" in value:
        return ""
    cleaned = re.sub(r"\s+", " ", value).strip()
    cleaned = cleaned.strip(".,;:!?…،؛؟ ")
    cleaned = re.sub(r"([.!?…])(?:\s*\1)+$", r"\1", cleaned).strip()
    invalid_shape = (
        not cleaned
        or "_" in cleaned
        or len(cleaned) > MAX_GLOSS_CHARACTERS
        or not re.search(r"\w", cleaned, re.UNICODE)
        or bool(re.search(r"(.)\1{4,}", cleaned, re.IGNORECASE))
    )
    if invalid_shape:
        return ""
    words = cleaned.split()
    if len(words) > max_words:
        return ""
    normalized = [word.casefold().strip(".,;:!?…") for word in words]
    if (
        len(normalized) >= MIN_REPETITION_CHECK_WORDS
        and len(set(normalized)) / len(normalized) < MIN_GLOSS_UNIQUENESS_RATIO
    ):
        return ""
    return cleaned


def _fallback_gloss(
    token: AnnotatedToken,
    candidates: list[str],
    translations: dict[str, str],
) -> str:
    """Choose the first safe lexical candidate or preserve the source word."""
    for candidate in candidates:
        gloss = _sanitize_gloss(translations.get(candidate, ""), max_words=3)
        if gloss:
            return gloss
    token.warnings.append("unsafe_lexical_translation")
    return token.surface


def _display_words_with_punctuation(subtitle: AnnotatedSubtitle) -> list[str]:
    """Attach source punctuation to neighbouring words without glossing it."""
    words = lexical_tokens(subtitle.tokens)
    displays = [token.display_surface for token in words]
    if not words:
        return displays
    lexical_positions = {token.token_index: index for index, token in enumerate(words)}
    for token_position, token in enumerate(subtitle.tokens):
        if token.token_index in lexical_positions:
            continue
        previous = next(
            (
                candidate
                for candidate in reversed(subtitle.tokens[:token_position])
                if candidate.token_index in lexical_positions
            ),
            None,
        )
        following = next(
            (
                candidate
                for candidate in subtitle.tokens[token_position + 1 :]
                if candidate.token_index in lexical_positions
            ),
            None,
        )
        if previous is None and following is not None:
            displays[lexical_positions[following.token_index]] = (
                token.surface + displays[lexical_positions[following.token_index]]
            )
        elif previous is not None:
            displays[lexical_positions[previous.token_index]] += token.surface
    return displays


def _overall_confidence(
    confidence: ConfidenceBreakdown,
    gloss_confidence: float,
) -> float:
    values = [
        (confidence.morphology, 0.15),
        (confidence.diacritization, 0.15),
        (confidence.ner, 0.10),
        (confidence.alignment, 0.35),
        (gloss_confidence, 0.25),
    ]
    present = [(value, weight) for value, weight in values if value > 0]
    if not present:
        return 0.0
    return sum(value * weight for value, weight in present) / sum(
        weight for _, weight in present
    )


def _build_spans(
    subtitle: AnnotatedSubtitle,
    entity_ranges: list[tuple[int, int, EntityType]],
    entities: dict[str, EntityRecord],
) -> list[AnnotatedSpan]:
    words = lexical_tokens(subtitle.tokens)
    display_words = _display_words_with_punctuation(subtitle)
    entity_by_start = {
        start: (end, entity_type) for start, end, entity_type in entity_ranges
    }
    spans: list[AnnotatedSpan] = []
    index = 0
    while index < len(words):
        entity = entity_by_start.get(index)
        if entity is not None:
            end, _ = entity
            grouped = words[index:end]
            record = entities[grouped[0].entity_id or ""]
            spans.append(
                AnnotatedSpan(
                    span_id=f"s{subtitle.subtitle_index}:span{len(spans)}",
                    subtitle_index=subtitle.subtitle_index,
                    token_start=index,
                    token_end=end,
                    kind=SpanKind.ENTITY,
                    source_surface=" ".join(
                        display_words[word_index] for word_index in range(index, end)
                    ),
                    display_source=" ".join(
                        display_words[word_index] for word_index in range(index, end)
                    ),
                    gloss=record.canonical_target,
                    transliteration=record.canonical_target,
                    entity_id=record.entity_id,
                    confidence=ConfidenceBreakdown(
                        ner=record.confidence,
                        gloss=record.confidence,
                        overall=record.confidence,
                    ),
                )
            )
            index = end
            continue
        token = words[index]
        spans.append(
            AnnotatedSpan(
                span_id=f"s{subtitle.subtitle_index}:span{len(spans)}",
                subtitle_index=subtitle.subtitle_index,
                token_start=index,
                token_end=index + 1,
                kind=SpanKind.TOKEN,
                source_surface=display_words[index],
                display_source=display_words[index],
                gloss=token.gloss or token.surface,
                transliteration=token.transliteration,
                target_indices=token.target_indices,
                confidence=token.confidence,
                warnings=list(token.warnings),
            )
        )
        index += 1
    return spans


__all__ = [
    "AlignmentBackend",
    "CamelMorphologyBackend",
    "CamelNerBackend",
    "CattDiacritizationBackend",
    "ContextualWordAligner",
    "DiacritizationBackend",
    "MorphologyBackend",
    "NerBackend",
    "PedagogicalAnnotator",
    "ProgressCallback",
]
