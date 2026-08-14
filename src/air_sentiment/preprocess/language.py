"""Language identification without a model or a wheel to download.

Two signals, in order of reliability. Script ranges settle Cyrillic, CJK, Arabic
and Devanagari outright — a text written in Cyrillic is not English regardless
of what any n-gram model thinks. Latin-script text falls back to a stop-word hit
rate, because function words are the highest-frequency, most language-specific
tokens available and matching them costs one set lookup per token.

The reported confidence is the load-bearing part. A wrong-but-confident answer
routes traffic to a tier that cannot read the input, so short text returns
``und`` rather than a guess, and weak evidence returns a low number that lets
the escalation policy decide to spend a rung on it.
"""

from __future__ import annotations

import re
from typing import Final

from air_sentiment.schemas.common import LanguageInfo

#: Below this many characters, stop-word evidence is noise: a single shared
#: token ("no", "es") would decide the answer. Undetermined is the honest reply.
MIN_DETECTION_CHARS: Final[int] = 12

#: Share of letters a non-Latin script needs before it decides the answer. Well
#: below half, because quoted product names and URLs dilute the native script.
_SCRIPT_SHARE_FLOOR: Final[float] = 0.30

#: Latin-script text with no functional-word evidence at all. English is the
#: bulk of such traffic (short ASCII product praise), but the confidence stays
#: deliberately low so the policy can still escalate on it.
_LATIN_FALLBACK_CODE: Final[str] = "en"
_LATIN_FALLBACK_CONFIDENCE: Final[float] = 0.40

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[^\W\d_]+")


def _words(listing: str) -> frozenset[str]:
    return frozenset(listing.split())


#: Function words only. Content words are deliberately absent: they are what the
#: lexicon scores, and overlapping the two would make a strongly worded review
#: look like a different language.
STOPWORDS: Final[dict[str, frozenset[str]]] = {
    "en": _words(
        "the and is was are were of to in it this that for with on at as but not have has "
        "had they we you your my our their from there which when would could should been "
        "will just about after before very really then than them these those here into over"
    ),
    "es": _words(
        "el la los las un una unos unas de del que y en es son era fue por para con sin "
        "pero muy como lo al su sus me mi te se ya todo toda cuando porque esta este esto "
        "estos hay ha han más también nada algo otro otra desde hasta tiene"
    ),
    "fr": _words(
        "le la les un une des du de et est sont était que qui dans pour avec sans mais très "
        "pas ne sur au aux ce cet cette ces je vous nous ils elle son sa ses leur plus moins "
        "aussi comme encore chez être avoir fait bien toujours jamais"
    ),
    "de": _words(
        "der die das den dem des ein eine einen einem eines und ist sind war waren nicht mit "
        "für auf aus bei nach von zu zum zur auch aber sehr schon noch wie wenn weil man ich "
        "wir sie ihr mein sein kein keine wurde haben hatte immer wieder leider"
    ),
    "pt": _words(
        "o os as um uma uns umas do da dos das de que e em no na nos nas por para com sem "
        "mas muito mais como não sim ao aos seu sua meu minha ele ela eles isso isto está "
        "estão foi são ter tem também já porque quando"
    ),
    "it": _words(
        "il lo gli un uno una del della dei delle di che e in nel nella per con senza ma "
        "molto più come non sono era erano al alla ai alle suo sua mio mia questo questa "
        "quello anche però perché quando ancora sempre mai"
    ),
}

#: Diacritics and inverted punctuation, counted without tokenising. Each is a
#: weak signal, so an occurrence contributes a small bonus and the total is
#: capped — one "café" in an English sentence must not flip the answer.
CHAR_HINTS: Final[dict[str, frozenset[str]]] = {
    "es": frozenset("ñ¿¡"),
    "fr": frozenset("çœàèùêâî"),
    "de": frozenset("äöüß"),
    "pt": frozenset("ãõçâê"),
    "it": frozenset("ìòàèù"),
}
_CHAR_HINT_UNIT: Final[float] = 0.04
_CHAR_HINT_CAP: Final[float] = 0.20

#: Contiguous code-point ranges, checked in order. Latin is the implicit default
#: because it is the broadest and the least informative.
_SCRIPT_RANGES: Final[tuple[tuple[str, int, int], ...]] = (
    ("cyrillic", 0x0400, 0x052F),
    ("greek", 0x0370, 0x03FF),
    ("hebrew", 0x0590, 0x05FF),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("arabic", 0xFB50, 0xFDFF),
    ("arabic", 0xFE70, 0xFEFF),
    ("devanagari", 0x0900, 0x097F),
    ("thai", 0x0E00, 0x0E7F),
    ("hangul", 0x1100, 0x11FF),
    ("hangul", 0x3130, 0x318F),
    ("hangul", 0xAC00, 0xD7AF),
    ("kana", 0x3040, 0x30FF),
    ("han", 0x3400, 0x4DBF),
    ("han", 0x4E00, 0x9FFF),
    ("han", 0xF900, 0xFAFF),
)

_SCRIPT_LANGUAGE: Final[dict[str, str]] = {
    "cyrillic": "ru",
    "greek": "el",
    "hebrew": "he",
    "arabic": "ar",
    "devanagari": "hi",
    "thai": "th",
    "hangul": "ko",
    "kana": "ja",
    "han": "zh",
}


def detect_language(text: str) -> LanguageInfo:
    """Best-effort language of `text`.

    Returns ``und`` with zero confidence whenever the evidence does not support
    a judgement — that is a valid answer here, not a failure.
    """
    stripped = text.strip()
    if len(stripped) < MIN_DETECTION_CHARS:
        return LanguageInfo.undetermined()

    counts = _script_counts(stripped)
    total = sum(counts.values())
    if total == 0:
        return LanguageInfo.undetermined()

    non_latin = {script: n for script, n in counts.items() if script != "latin"}
    if non_latin:
        script = max(non_latin, key=lambda key: non_latin[key])
        share = non_latin[script] / total
        if share >= _SCRIPT_SHARE_FLOOR:
            return LanguageInfo(
                code=_script_language(script, counts),
                confidence=min(0.99, 0.55 + 0.44 * share),
            )

    if counts.get("latin", 0) == 0:
        return LanguageInfo.undetermined()
    return _detect_latin(stripped)


def _script_counts(text: str) -> dict[str, int]:
    """Letters per script. Digits and punctuation are ignored entirely."""
    counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        script = _script_of(char)
        counts[script] = counts.get(script, 0) + 1
    return counts


def _script_of(char: str) -> str:
    point = ord(char)
    for name, low, high in _SCRIPT_RANGES:
        if low <= point <= high:
            return name
    return "latin"


def _script_language(script: str, counts: dict[str, int]) -> str:
    """Map a script to a language code.

    Han is the one genuinely ambiguous case: Japanese is written with Han
    characters too, and the presence of any kana is what settles it.
    """
    if script == "han" and counts.get("kana", 0) > 0:
        return "ja"
    return _SCRIPT_LANGUAGE.get(script, "und")


def _detect_latin(text: str) -> LanguageInfo:
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    if not tokens:
        return LanguageInfo.undetermined()

    scores = {
        code: _language_score(code, words, tokens, lowered) for code, words in STOPWORDS.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_code, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    if best <= 0.0:
        return LanguageInfo(code=_LATIN_FALLBACK_CODE, confidence=_LATIN_FALLBACK_CONFIDENCE)

    # Absolute evidence and the gap to the runner-up both count: a high hit rate
    # for two languages at once (Spanish and Portuguese) is not confidence.
    confidence = min(0.99, 0.45 + 1.1 * best + 0.8 * (best - runner_up))
    return LanguageInfo(code=best_code, confidence=confidence)


def _language_score(code: str, words: frozenset[str], tokens: list[str], lowered: str) -> float:
    hits = sum(1 for token in tokens if token in words)
    hint_chars = CHAR_HINTS.get(code)
    hint = 0.0
    if hint_chars:
        hint = min(_CHAR_HINT_CAP, _CHAR_HINT_UNIT * sum(1 for ch in lowered if ch in hint_chars))
    return hits / len(tokens) + hint
