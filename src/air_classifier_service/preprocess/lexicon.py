"""Hand-built polarity resources and the scorer that uses them.

Everything here is inline and versioned with the code: no downloads, no model
artifacts, no import-time I/O. Tier 0 answers from this module alone and Tier 1
blends it with the neural score, so it must be deterministic and it must be
fast — one dictionary lookup per token, no regexes inside the scoring loop.

The single most important behaviour is **negation**. "not great" is negative and
"never had a problem" is positive; a scorer that reads the term and ignores the
word in front of it is worse than no scorer at all, because it is confidently
wrong on exactly the inputs a human would call obvious. Negation therefore flips
the sign of any term inside a short window after the negator, and the window
does not cross a clause boundary — "Not good. Amazing service." must not turn
the praise into criticism.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

# ── Tables ────────────────────────────────────────────────────────────────────

# Weights are the term's polarity in isolation, on -1..+1. They are ordinal
# rather than measured: what matters is that "superb" outranks "fine" and that
# nothing saturates, so a single strong word cannot pin the whole text at ±1.

# fmt: off
POSITIVE_TERMS: Final[dict[str, float]] = {
    "excellent": 0.90, "amazing": 0.90, "amazingly": 0.85, "awesome": 0.90,
    "superb": 0.95, "superbly": 0.85, "outstanding": 0.90, "fantastic": 0.90,
    "wonderful": 0.85, "brilliant": 0.85, "flawless": 0.95, "flawlessly": 0.90,
    "perfect": 0.95, "perfectly": 0.85, "great": 0.80, "good": 0.60,
    "nice": 0.55, "lovely": 0.70, "delightful": 0.80, "pleasant": 0.60,
    "enjoyable": 0.70, "satisfying": 0.70, "satisfied": 0.70, "happy": 0.75,
    "thrilled": 0.90, "delighted": 0.85, "impressed": 0.75, "impressive": 0.80,
    "exceptional": 0.90, "remarkable": 0.80, "terrific": 0.85, "fabulous": 0.85,
    "marvellous": 0.85, "marvelous": 0.85, "stellar": 0.85, "premium": 0.60,
    "polished": 0.70, "intuitive": 0.70, "seamless": 0.80, "smooth": 0.70,
    "smoothly": 0.65, "fast": 0.60, "quick": 0.50, "quickly": 0.50,
    "responsive": 0.70, "reliable": 0.75, "dependable": 0.70, "sturdy": 0.60,
    "durable": 0.70, "solid": 0.60, "robust": 0.70, "efficient": 0.70,
    "helpful": 0.70, "friendly": 0.70, "courteous": 0.70, "polite": 0.60,
    "professional": 0.60, "knowledgeable": 0.70, "attentive": 0.65,
    "prompt": 0.60, "punctual": 0.60, "affordable": 0.60, "worthwhile": 0.65,
    "valuable": 0.70, "bargain": 0.70, "generous": 0.65, "refreshing": 0.65,
    "comfortable": 0.65, "cosy": 0.60, "cozy": 0.60, "spacious": 0.55,
    "clean": 0.60, "spotless": 0.80, "tidy": 0.55, "beautiful": 0.80,
    "gorgeous": 0.85, "elegant": 0.75, "stylish": 0.70, "sleek": 0.70,
    "vibrant": 0.65, "crisp": 0.60, "bright": 0.50, "sharp": 0.50,
    "accurate": 0.70, "precise": 0.65, "consistent": 0.60, "stable": 0.65,
    "secure": 0.60, "easy": 0.60, "effortless": 0.75, "convenient": 0.65,
    "straightforward": 0.60, "simple": 0.50, "informative": 0.60,
    "useful": 0.65, "practical": 0.60, "versatile": 0.65, "powerful": 0.70,
    "capable": 0.60, "innovative": 0.70, "improved": 0.60, "better": 0.50,
    "best": 0.85, "favourite": 0.80, "favorite": 0.80, "recommend": 0.75,
    "recommended": 0.75, "love": 0.85, "loved": 0.85, "loves": 0.80,
    "adore": 0.85, "enjoy": 0.70, "enjoyed": 0.70, "appreciate": 0.65,
    "appreciated": 0.65, "grateful": 0.75, "thankful": 0.70, "thanks": 0.50,
    "kudos": 0.75, "praise": 0.70, "applaud": 0.70, "worth": 0.50,
    "works": 0.45, "resolved": 0.70, "fixed": 0.60, "exceeded": 0.80,
    "exceeds": 0.75, "wow": 0.80, "bravo": 0.75, "gem": 0.70,
    "winner": 0.70, "success": 0.70, "successful": 0.70, "lifesaver": 0.85,
    "godsend": 0.85, "charming": 0.70, "cheerful": 0.70, "fun": 0.65,
    "exciting": 0.70, "excited": 0.70, "pleased": 0.70, "snappy": 0.65,
    "lightweight": 0.50, "quiet": 0.50, "relaxing": 0.70, "delicious": 0.85,
    "tasty": 0.75, "fresh": 0.60, "flavourful": 0.75, "flavorful": 0.75,
    "attractive": 0.65, "pristine": 0.80, "immaculate": 0.85, "painless": 0.70,
    "fine": 0.45, "decent": 0.50, "trustworthy": 0.75, "honest": 0.70,
    "transparent": 0.60, "welcoming": 0.70, "treasure": 0.70, "glad": 0.65,
    "joy": 0.75, "joyful": 0.80, "ecstatic": 0.90, "recommendable": 0.70,
}

NEGATIVE_TERMS: Final[dict[str, float]] = {
    "terrible": -0.90, "awful": -0.90, "horrible": -0.90, "atrocious": -0.95,
    "abysmal": -0.95, "appalling": -0.90, "dreadful": -0.90,
    "horrendous": -0.90, "disgusting": -0.90, "disgusted": -0.85,
    "revolting": -0.90, "unacceptable": -0.85, "unusable": -0.85,
    "useless": -0.80, "worthless": -0.85, "garbage": -0.85, "trash": -0.80,
    "junk": -0.75, "rubbish": -0.80, "crap": -0.80, "lousy": -0.75,
    "poor": -0.70, "poorly": -0.70, "bad": -0.70, "badly": -0.65,
    "worse": -0.70, "worst": -0.90, "mediocre": -0.50, "subpar": -0.65,
    "disappointing": -0.75, "disappointed": -0.75, "disappointment": -0.75,
    "letdown": -0.75, "underwhelming": -0.60, "lacklustre": -0.60,
    "lackluster": -0.60, "shoddy": -0.75, "janky": -0.65,
    "lacking": -0.60, "lacks": -0.60, "flawed": -0.65,
    "faulty": -0.75, "defective": -0.80, "defect": -0.70, "defects": -0.70,
    "broken": -0.80, "breaks": -0.65, "cracked": -0.70, "damaged": -0.75,
    "dented": -0.60, "scratched": -0.55, "malfunction": -0.80,
    "malfunctioning": -0.80, "buggy": -0.75, "bug": -0.50, "bugs": -0.55,
    "glitch": -0.60, "glitchy": -0.70, "crash": -0.75, "crashes": -0.75,
    "crashed": -0.75, "freezes": -0.65, "frozen": -0.50, "laggy": -0.70,
    "lag": -0.60, "slow": -0.60, "sluggish": -0.65, "unresponsive": -0.75,
    "unstable": -0.70, "unreliable": -0.75, "inconsistent": -0.55,
    "clunky": -0.65, "confusing": -0.60, "complicated": -0.50,
    "cumbersome": -0.60, "unintuitive": -0.60, "awkward": -0.50, "ugly": -0.65,
    "flimsy": -0.70, "fragile": -0.50, "brittle": -0.55, "leaks": -0.60,
    "leaky": -0.60, "noisy": -0.55, "smelly": -0.70, "dirty": -0.70,
    "filthy": -0.85, "stained": -0.60, "mouldy": -0.85, "moldy": -0.85,
    "stale": -0.65, "bland": -0.60, "tasteless": -0.70, "inedible": -0.85,
    "overpriced": -0.75, "expensive": -0.45, "pricey": -0.40, "ripoff": -0.90,
    "scam": -0.95, "fraud": -0.95, "fraudulent": -0.90, "misleading": -0.75,
    "deceptive": -0.80, "dishonest": -0.85, "lied": -0.80, "liar": -0.85,
    "rude": -0.85, "impolite": -0.70, "disrespectful": -0.80,
    "unprofessional": -0.75, "unhelpful": -0.70, "incompetent": -0.85,
    "careless": -0.70, "negligent": -0.80, "ignored": -0.70,
    "dismissive": -0.70, "arrogant": -0.70, "hostile": -0.80, "abusive": -0.90,
    "threatening": -0.80, "angry": -0.75, "furious": -0.85, "frustrated": -0.75,
    "frustrating": -0.75, "annoyed": -0.65, "annoying": -0.65,
    "irritating": -0.65, "upset": -0.70, "unhappy": -0.70, "miserable": -0.80,
    "regret": -0.75, "regretted": -0.75, "avoid": -0.70, "complaint": -0.60,
    "complain": -0.60, "refund": -0.55, "refunds": -0.55, "chargeback": -0.70,
    "cancelled": -0.50, "canceled": -0.50, "delayed": -0.60, "delay": -0.55,
    "late": -0.50, "missing": -0.60, "nightmare": -0.85, "disaster": -0.85,
    "fiasco": -0.80, "mess": -0.65, "hassle": -0.65, "headache": -0.60,
    "painful": -0.70, "struggle": -0.60, "difficult": -0.50,
    "impossible": -0.70, "failed": -0.75, "failure": -0.75, "fails": -0.70,
    "fail": -0.70, "error": -0.55, "errors": -0.55, "overheats": -0.70,
    "overheating": -0.70, "stuck": -0.60, "died": -0.75, "dead": -0.70,
    "wasted": -0.75, "waste": -0.75, "meh": -0.40, "spam": -0.70,
    "creepy": -0.70, "sketchy": -0.70, "shady": -0.75, "insulting": -0.80,
    "embarrassing": -0.65, "issue": -0.45, "issues": -0.50, "problem": -0.55,
    "problems": -0.60, "wrong": -0.60, "unsafe": -0.80,
}

#: Multipliers applied to the next sentiment term. Above 1 amplifies, and the
#: values are ordinal in the same way the term weights are.
INTENSIFIERS: Final[dict[str, float]] = {
    "very": 1.50, "extremely": 1.80, "absolutely": 1.70, "incredibly": 1.70,
    "insanely": 1.70, "super": 1.40, "really": 1.40, "totally": 1.50,
    "completely": 1.60, "utterly": 1.70, "highly": 1.40, "hugely": 1.50,
    "immensely": 1.60, "exceptionally": 1.70, "remarkably": 1.50,
    "particularly": 1.30, "especially": 1.30, "thoroughly": 1.50,
    "deeply": 1.50, "seriously": 1.40, "terribly": 1.50, "awfully": 1.50,
    "extraordinarily": 1.80, "unbelievably": 1.60, "ridiculously": 1.60,
    "truly": 1.50, "genuinely": 1.40, "definitely": 1.30, "certainly": 1.30,
    "so": 1.30, "such": 1.20, "too": 1.30, "way": 1.40, "much": 1.20,
}

#: The mirror image: multipliers below 1 that soften the term they modify.
DIMINISHERS: Final[dict[str, float]] = {
    "slightly": 0.50, "somewhat": 0.60, "marginally": 0.50, "mildly": 0.60,
    "moderately": 0.60, "partially": 0.60, "fairly": 0.70, "rather": 0.75,
    "occasionally": 0.70, "sometimes": 0.75, "kinda": 0.60, "sorta": 0.60,
    "bit": 0.60, "little": 0.70, "almost": 0.80, "nearly": 0.80,
    "pretty": 0.85, "relatively": 0.70,
}
# fmt: on

#: Words that invert the polarity of what follows them. Contractions ending in
#: ``n't`` are recognised by suffix instead of being enumerated.
# fmt: off
NEGATIONS: Final[frozenset[str]] = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "without", "cannot", "cant", "hardly", "barely", "scarcely", "rarely",
    "seldom", "lack",
})
# fmt: on

#: Clause-level contrast. Their presence means the text probably holds two
#: opposing judgements, which is a routing signal rather than a polarity one:
#: the lexicon cannot say which side the author landed on, so it escalates.
# fmt: off
CONTRASTIVE_MARKERS: Final[frozenset[str]] = frozenset({
    "but", "however", "although", "though", "yet", "nevertheless",
    "nonetheless", "whereas", "albeit", "conversely",
})
# fmt: on

#: Emoji polarity, keyed on the bare code point. One entry covers every
#: rendering of the same face: lookup is per code point, so variation selectors,
#: skin-tone modifiers and zero-width joiners never match and never interfere.
# fmt: off
EMOJI_POLARITY: Final[dict[str, float]] = {
    "😀": 0.80, "😃": 0.80, "😄": 0.85, "😁": 0.80, "😆": 0.75, "😊": 0.85,
    "🙂": 0.50, "😉": 0.50, "😍": 0.95, "🥰": 0.95, "😘": 0.85, "🤩": 0.90,
    "😎": 0.70, "🥳": 0.90, "😂": 0.70, "🤣": 0.75, "👍": 0.80, "👏": 0.80,
    "🙌": 0.80, "💪": 0.70, "❤": 0.90, "🧡": 0.85, "💛": 0.85, "💚": 0.85,
    "💙": 0.85, "💜": 0.85, "💖": 0.90, "💕": 0.85, "💯": 0.85, "🔥": 0.70,
    "⭐": 0.70, "🌟": 0.75, "✨": 0.60, "✅": 0.60, "🎉": 0.85, "🎊": 0.80,
    "😌": 0.60, "🤗": 0.80, "🙏": 0.50, "😇": 0.70, "🥇": 0.80, "😻": 0.85,
    "😞": -0.75, "😔": -0.70, "😢": -0.80, "😭": -0.85, "😡": -0.90,
    "🤬": -0.95, "😠": -0.85, "😤": -0.60, "👎": -0.85, "💔": -0.85,
    "😩": -0.75, "😫": -0.75, "😖": -0.70, "😣": -0.65, "🙄": -0.60,
    "😒": -0.65, "😕": -0.50, "🙁": -0.55, "☹": -0.60, "😰": -0.70,
    "😨": -0.70, "😱": -0.70, "🤮": -0.90, "🤢": -0.85, "💩": -0.85,
    "❌": -0.60, "⛔": -0.60, "🚫": -0.60, "😐": -0.20, "😑": -0.30,
    "🤨": -0.35, "😬": -0.40, "⚠": -0.40, "😴": -0.30, "🥱": -0.35,
}
# fmt: on

#: Merged view used by the scorer. Built once at import; the two source tables
#: stay separate because they are maintained (and tested) as separate lists.
TERM_POLARITY: Final[dict[str, float]] = {**POSITIVE_TERMS, **NEGATIVE_TERMS}

# ── Scoring parameters ────────────────────────────────────────────────────────

#: How many tokens after a negator remain inverted. Three covers "never had a
#: problem" and "not a great experience" without reaching into the next clause.
NEGATION_WINDOW: Final[int] = 3

#: A negated term is not the exact opposite of itself: "not great" is mildly
#: negative, nowhere near as negative as "terrible". Damping encodes that.
NEGATION_DAMPING: Final[float] = 0.75

#: How far back a term looks for an intensifier or diminisher.
MODIFIER_WINDOW: Final[int] = 2

#: Softening constant of the ``x / sqrt(x * x + alpha)`` squash. Larger values
#: make a single strong term less decisive, demanding more agreement to score high.
NORMALISER_ALPHA: Final[float] = 2.0

#: Weight of emoji when a caller asks for one blended number. Emoji qualify
#: prose, they do not overrule it.
EMOJI_BLEND_WEIGHT: Final[float] = 0.50

#: Matched terms are diagnostics, not data. Cap the list so a pathological input
#: cannot make a log line or a trace attribute enormous.
MAX_MATCHED_TERMS: Final[int] = 32

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z]+(?:'[a-z]+)?")
_CLAUSE_BREAK_RE: Final[re.Pattern[str]] = re.compile(r"[.!?;:,()\[\]\n]|--")
#: Typographic apostrophes, folded so that a curly-quoted contraction negates
#: exactly like its ASCII spelling.
_APOSTROPHES: Final[dict[int, str]] = str.maketrans({"\u2019": "'", "\u02bc": "'"})

# ── Result ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LexiconScore:
    """What the lexicon saw in one piece of text.

    ``polarity`` covers the *prose only*. Emoji are reported separately and are
    deliberately not folded in, because the disagreement between the two is
    itself a signal — blending them first would average away the very conflict
    that :attr:`ComplexityFeatures.emoji_text_conflict` exists to detect.
    """

    polarity: float = 0.0
    positive_hits: int = 0
    negative_hits: int = 0
    negation_count: int = 0
    intensifier_count: int = 0
    contrastive_count: int = 0
    emoji_polarity: float = 0.0
    matched_terms: tuple[str, ...] = ()

    @property
    def overall_polarity(self) -> float:
        """Prose and emoji combined, for callers that want one number.

        Emoji-only text has no prose to qualify, so there the emoji *are* the
        judgement — which is exactly the Tier-0 emoji shortcut.
        """
        if not self.matched_terms:
            return self.emoji_polarity
        return _clamp(self.polarity + EMOJI_BLEND_WEIGHT * self.emoji_polarity)


# ── Scorer ────────────────────────────────────────────────────────────────────


def score_text(text: str) -> LexiconScore:
    """Score `text` against the tables above.

    Pure and allocation-light: it runs on every request at Tier 0, and again
    inside Tier 1's ensemble.
    """
    tokens = _tokenise(text.translate(_APOSTROPHES).lower())

    total = 0.0
    positive_hits = 0
    negative_hits = 0
    negation_count = 0
    intensifier_count = 0
    contrastive_count = 0
    matched: list[str] = []

    for index, token in enumerate(tokens):
        word = token.text

        if word in NEGATIONS or word.endswith("n't"):
            negation_count += 1
            continue
        if word in INTENSIFIERS:
            intensifier_count += 1
            continue
        if word in CONTRASTIVE_MARKERS:
            contrastive_count += 1
            continue

        weight = TERM_POLARITY.get(word)
        if weight is None:
            continue

        weight *= _modifier_multiplier(tokens, index)
        if _is_negated(tokens, index):
            weight = -weight * NEGATION_DAMPING

        total += weight
        if len(matched) < MAX_MATCHED_TERMS:
            matched.append(word)
        # Hits are counted on the *post-negation* sign: "never had a problem"
        # is a positive statement, and the polarity-flip trigger downstream
        # would fire spuriously if it were filed as a negative hit.
        if weight > 0:
            positive_hits += 1
        elif weight < 0:
            negative_hits += 1

    return LexiconScore(
        polarity=_squash(total),
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        negation_count=negation_count,
        intensifier_count=intensifier_count,
        contrastive_count=contrastive_count,
        emoji_polarity=score_emoji(text),
        matched_terms=tuple(matched),
    )


def score_emoji(text: str) -> float:
    """Mean polarity of the recognised emoji in `text`, or 0.0 if there are none."""
    # Scanning per code point means variation selectors, skin-tone modifiers
    # and zero-width joiners fall out on their own: they are never keys, so
    # "\u2764\ufe0f" is found by its base heart and the selector is ignored.
    weights = [EMOJI_POLARITY[char] for char in text if char in EMOJI_POLARITY]
    if not weights:
        return 0.0
    return _clamp(sum(weights) / len(weights))


# ── Internals ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    #: True when punctuation or a line break separates this token from the
    #: previous one. Negation and modifier windows refuse to cross it.
    starts_clause: bool


def _tokenise(lowered: str) -> list[_Token]:
    tokens: list[_Token] = []
    previous_end = 0
    for match in _WORD_RE.finditer(lowered):
        gap = lowered[previous_end : match.start()]
        starts_clause = not tokens or bool(_CLAUSE_BREAK_RE.search(gap))
        tokens.append(_Token(text=match.group(), starts_clause=starts_clause))
        previous_end = match.end()
    return tokens


def _is_negated(tokens: list[_Token], index: int) -> bool:
    for offset in range(1, NEGATION_WINDOW + 1):
        back = index - offset
        if back < 0 or tokens[back + 1].starts_clause:
            return False
        word = tokens[back].text
        if word in NEGATIONS or word.endswith("n't"):
            return True
    return False


def _modifier_multiplier(tokens: list[_Token], index: int) -> float:
    """Multiplier contributed by the nearest preceding intensifier/diminisher."""
    for offset in range(1, MODIFIER_WINDOW + 1):
        back = index - offset
        if back < 0 or tokens[back + 1].starts_clause:
            return 1.0
        word = tokens[back].text
        multiplier = INTENSIFIERS.get(word) or DIMINISHERS.get(word)
        if multiplier is not None:
            return multiplier
    return 1.0


def _squash(total: float) -> float:
    """Map an unbounded weight sum onto -1..+1 without a hard clip.

    A clip would make every emphatic review look identical; this keeps the
    ordering between "good" and "superb, flawless, perfect" intact.
    """
    return total / math.sqrt(total * total + NORMALISER_ALPHA)


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))
