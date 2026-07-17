"""Lossless Arabic tokenization and script helpers."""

from __future__ import annotations

import re
import unicodedata

from dual_subtitles.models.subtitle import AnnotatedToken, TokenKind

TOKEN_PATTERN = re.compile(
    r"\w+(?:[\u064b-\u065f\u0670\u06d6-\u06ed]+\w*)*|[^\w\s]",
    re.UNICODE,
)
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
ARABIC_LETTER = re.compile(r"[\u0621-\u063a\u0641-\u064a]")

TRANSLITERATION_MAP = {
    "ء": "ʾ",
    "آ": "ʾā",
    "أ": "ʾa",
    "ؤ": "ʾu",
    "إ": "ʾi",
    "ئ": "ʾ",
    "ا": "ā",
    "ب": "b",
    "ة": "a",
    "ت": "t",
    "ث": "th",
    "ج": "j",
    "ح": "ḥ",
    "خ": "kh",
    "د": "d",
    "ذ": "dh",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "ṣ",
    "ض": "ḍ",
    "ط": "ṭ",
    "ظ": "ẓ",
    "ع": "ʿ",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",
    "ى": "ā",
    "ي": "y",
    "َ": "a",
    "ُ": "u",
    "ِ": "i",
    "ً": "an",
    "ٌ": "un",
    "ٍ": "in",
    "ْ": "",
    "ّ": "",
    "ـ": "",
}


def tokenize_losslessly(text: str, subtitle_index: int) -> list[AnnotatedToken]:
    """Tokenize text while retaining exact source offsets and surfaces."""
    tokens: list[AnnotatedToken] = []
    for token_index, match in enumerate(TOKEN_PATTERN.finditer(text)):
        surface = match.group(0)
        kind = (
            TokenKind.WORD
            if any(character.isalnum() for character in surface)
            else TokenKind.PUNCTUATION
        )
        tokens.append(
            AnnotatedToken(
                token_id=f"s{subtitle_index}:t{token_index}",
                subtitle_index=subtitle_index,
                token_index=token_index,
                start_char=match.start(),
                end_char=match.end(),
                surface=surface,
                kind=kind,
            )
        )
    return tokens


def reconstruct_from_offsets(text: str, tokens: list[AnnotatedToken]) -> str:
    """Reconstruct text from token offsets and untouched gaps."""
    if not tokens:
        return text
    pieces: list[str] = []
    cursor = 0
    for token in tokens:
        pieces.append(text[cursor : token.start_char])
        pieces.append(token.surface)
        cursor = token.end_char
    pieces.append(text[cursor:])
    return "".join(pieces)


def strip_diacritics(text: str) -> str:
    """Remove Arabic combining marks without changing base letters."""
    return ARABIC_DIACRITICS.sub("", unicodedata.normalize("NFC", text))


def consonantal_key(text: str) -> str:
    """Return a conservative matching key without changing display text."""
    stripped = strip_diacritics(text)
    normalized = stripped.translate(str.maketrans("أإآىؤئ", "ااايوي"))
    return "".join(character for character in normalized if character.isalnum())


def contains_arabic(text: str) -> bool:
    """Return whether text contains at least one Arabic base letter."""
    return ARABIC_LETTER.search(text) is not None


def transliterate_arabic(text: str) -> str:
    """Produce a readable deterministic Latin transliteration."""
    output: list[str] = []
    previous_base = ""
    for character in unicodedata.normalize("NFC", text):
        if character == "ّ" and previous_base:
            output.append(TRANSLITERATION_MAP.get(previous_base, ""))
            continue
        mapped = TRANSLITERATION_MAP.get(character)
        if mapped is not None:
            output.append(mapped)
            if not unicodedata.combining(character):
                previous_base = character
        elif character.isspace() or character in "-'":
            output.append(character)
    return re.sub(r"\s+", " ", "".join(output)).strip()


def lexical_tokens(tokens: list[AnnotatedToken]) -> list[AnnotatedToken]:
    """Return only words, preserving their original token indices."""
    return [token for token in tokens if token.kind is TokenKind.WORD]
