"""Text normalisation — the first thing that touches caller input.

Every downstream consumer sees the output of this module rather than the raw
request body: the lexicon, the language detector, the cache key, the tokenizer,
and the prompt. That makes normalisation part of the cache contract — two
requests that differ only in zero-width joiners or trailing whitespace must
produce the same key, or the cache silently stops working.

What is deliberately *preserved* matters as much as what is stripped. Emoji and
sentence punctuation carry sentiment and complexity signal, so they survive
untouched; only characters that are invisible or actively hostile to tokenisers
are removed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

#: C0/C1 control characters, minus tab/newline/carriage-return which are
#: whitespace and are handled by collapsing. Replaced by a space rather than
#: deleted so that ``word\x00word`` does not become one glued token.
_CONTROL_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

#: Soft hyphen, zero-width space/joiners, bidirectional overrides and the BOM.
#: These are invisible, so a human reviewer cannot tell two "identical" inputs
#: apart — and they are the standard way to split a keyword past a filter or to
#: make two cache keys out of one sentence.
_ZERO_WIDTH_RE: Final[re.Pattern[str]] = re.compile(
    "[\xad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]"
)

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

#: A truncation only moves back to a word boundary when that boundary is at
#: least this far into the budget. Without the floor, a single very long token
#: (a URL, a base64 blob) would collapse the text to almost nothing.
_WORD_BOUNDARY_FLOOR: Final[float] = 0.75


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """Cleaned text plus what had to be done to get it there.

    ``original_length`` is the length of the caller's text *before* any
    cleaning, because that is what the response reports and what a truncation
    warning is measured against.
    """

    text: str
    original_length: int
    truncated: bool


def normalize(text: str, *, max_chars: int) -> NormalizedText:
    """Fold `text` to a canonical form and cap it at `max_chars`.

    NFKC is chosen over NFC so that full-width Latin and ligature forms (``ﬁ``)
    fold onto their ASCII equivalents — the lexicon carries one spelling of each
    word and should not need to carry the others.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    original_length = len(text)

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    truncated = len(cleaned) > max_chars
    if truncated:
        cleaned = _truncate_on_word_boundary(cleaned, max_chars)

    return NormalizedText(text=cleaned, original_length=original_length, truncated=truncated)


def _truncate_on_word_boundary(text: str, max_chars: int) -> str:
    """Cut at the last whitespace inside the budget, falling back to a hard cut.

    Mid-word truncation is not cosmetic damage: it invents tokens the lexicon
    and the classifier have never seen, and a half-word is a worse input than a
    slightly shorter sentence.
    """
    window = text[:max_chars]
    boundary = window.rfind(" ")
    if boundary >= max_chars * _WORD_BOUNDARY_FLOOR:
        return window[:boundary].rstrip()
    return window.rstrip()
