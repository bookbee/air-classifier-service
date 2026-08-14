"""Complexity feature extraction — the input to every routing decision.

The policy in :mod:`air_sentiment.pipeline.policy` never reads the text; it
reads this. That makes the module's cost budget absolute: it runs on every
request, on the hot path, before anything else has happened, so it stays
synchronous, regex-light and free of any allocation proportional to anything
but the input.

Nothing here classifies. These are signals about *who should be asked* — a
contrastive conjunction says "two judgements in one sentence, a lexicon will
mangle this", not "the sentiment is mixed".
"""

from __future__ import annotations

import re
from typing import Final

from air_sentiment.preprocess.lexicon import LexiconScore, score_text
from air_sentiment.schemas.common import ComplexityFeatures

#: Word-ish tokens, plus one token per CJK ideograph or kana. Without the second
#: half, a Japanese sentence would count as a single token and would never trip
#: the length-based triggers.
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]|[^\W_]+(?:['\u2019][^\W_]+)*"
)

_EXCLAMATIONS: Final[tuple[str, ...]] = ("!", "\uff01")

_URL_RE: Final[re.Pattern[str]] = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)

#: Fenced blocks, markup tags, assignment/comparison operators and the keywords
#: that open a declaration. Code in a review is usually a bug report, and it
#: derails a lexicon completely.
_CODE_RE: Final[re.Pattern[str]] = re.compile(
    r"```|</?[A-Za-z][^>]*>|=>|&&|\|\||[{}]"
    r"|\b(?:def|class|function|import|const|return|SELECT|INSERT|UPDATE|DELETE)\s",
)

#: Terminators that make a text a statement rather than a pure question.
_STATEMENT_RE: Final[re.Pattern[str]] = re.compile(r"[.!\uff01\u3002]")


def extract_features(text: str, *, lexicon_score: LexiconScore | None = None) -> ComplexityFeatures:
    """Derive the routing signals for `text`.

    `lexicon_score` is accepted so a caller that has already scored the text —
    Tier 0 does, on its way to a shortcut — does not pay for it twice.
    """
    score = lexicon_score if lexicon_score is not None else score_text(text)
    stripped = text.strip()

    return ComplexityFeatures(
        token_count=len(_TOKEN_RE.findall(text)),
        char_count=len(text),
        negation_count=score.negation_count,
        contrastive_count=score.contrastive_count,
        intensifier_count=score.intensifier_count,
        exclamation_count=sum(text.count(mark) for mark in _EXCLAMATIONS),
        positive_hits=score.positive_hits,
        negative_hits=score.negative_hits,
        emoji_polarity=_clamp(score.emoji_polarity),
        lexical_polarity=_clamp(score.polarity),
        uppercase_ratio=_uppercase_ratio(text),
        question_only=_is_question_only(stripped),
        has_url_or_code=bool(_URL_RE.search(text) or _CODE_RE.search(text)),
    )


def _uppercase_ratio(text: str) -> float:
    """Share of letters that are upper case; 0.0 when there are no letters.

    Cased letters only — counting the whole string would report every CJK or
    Arabic text as 0% shouting for the wrong reason.
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if char.isupper()) / len(letters)


def _is_question_only(stripped: str) -> bool:
    """A text made entirely of questions, with no assertion anywhere in it.

    Worth isolating because "Is the battery any good?" contains a positive
    lexicon hit while asserting nothing at all.
    """
    return stripped.endswith("?") and not _STATEMENT_RE.search(stripped)


def _clamp(value: float) -> float:
    """Guard the model's bounds. A caller-supplied score is not to be trusted."""
    return max(-1.0, min(1.0, value))
