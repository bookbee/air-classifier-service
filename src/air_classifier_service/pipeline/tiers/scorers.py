"""Pluggable scoring backends for Tier 1.

The tier owns the policy-facing behaviour; this module owns the arithmetic that
turns text into a distribution over :data:`SCORED_LABELS`. Three backends
implement :class:`ScorerBackend`:

* :class:`ZeroShotOnnxScorer` — a quantised DeBERTa-v3 NLI head run on CPU,
  scoring all four labels directly by entailment against a hypothesis per
  label. The default when an ONNX artifact is present; see the class
  docstring for why this replaced a plain sequence-classification head.
* :class:`OnnxScorer` — a two- or three-class sequence-classification head
  (the SST-2 family). Kept working, not the default: a deployment that already
  baked in that artifact loses nothing by staying there, and the escalation
  ladder does not care which of the two produced a verdict.
* :class:`LexiconScorer` — pure Python, no artifact, no wheel, always available.

Both ONNX backends are optional: the model artifact is baked into the Docker
image and the runtime lives behind the ``onnx`` extra, so neither is present in
a plain checkout. Which one is live is a deployment fact, not a code path the
rest of the service knows about: :func:`build_scorer` picks the best one that
actually loaded and matches ``classifier.scorer_kind``, and the tier degrades
in accuracy rather than disappearing when the artifact is missing. That is the
whole reason the backend is a protocol.

Scoring here is deliberately synchronous. It is CPU work, and the caller — not
the callee — decides how to get it off the event loop.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from air_classifier_service.config import Settings
from air_classifier_service.constants import Label
from air_classifier_service.observability.logging import get_logger
from air_classifier_service.preprocess.lexicon import LexiconScore, score_text

__all__ = [
    "LEXICON_SCORER_NAME",
    "SCORED_LABELS",
    "LexiconScorer",
    "OnnxScorer",
    "ScoreResult",
    "ScorerBackend",
    "ScorerUnavailableError",
    "ZeroShotOnnxScorer",
    "apply_temperature",
    "blend",
    "build_scorer",
    "lexicon_distribution",
    "normalise",
    "softmax",
]

_log = get_logger(__name__)

#: The classes a scorer ranks. ``UNKNOWN`` is deliberately absent: it means "no
#: judgement was made", which is a tier-level outcome rather than a class with a
#: probability attached to it.
SCORED_LABELS: Final[tuple[Label, ...]] = (
    Label.POSITIVE,
    Label.NEGATIVE,
    Label.NEUTRAL,
    Label.MIXED,
)

#: Tie-break order. Two labels at identical probability must resolve the same
#: way on every call, or the same text yields different answers run to run.
_LABEL_RANK: Final[dict[Label, int]] = {label: index for index, label in enumerate(SCORED_LABELS)}

#: Floor applied before taking a logarithm. ``log(0)`` is ``-inf``, and one
#: infinite logit poisons every re-softmax it takes part in.
_MIN_PROB: Final[float] = 1e-9


class ScorerUnavailableError(RuntimeError):
    """A backend was asked to score without having loaded."""


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """One backend's answer for one piece of text.

    ``margin`` is top-1 minus top-2 and is not decoration: the escalation policy
    compares it against ``tiers.t1_classifier.margin_floor`` to catch the case a
    confidence number hides — a 0.45/0.44 photo finish and a 0.45/0.05 rout both
    report 0.45, and only the first one needs a better model to look at it.
    """

    probabilities: dict[Label, float]
    top_label: Label
    confidence: float
    margin: float

    @classmethod
    def from_probabilities(cls, probabilities: Mapping[Label, float]) -> ScoreResult:
        """Normalise a mapping and derive the ranking from it."""
        normalised = normalise(probabilities)
        ranked = sorted(normalised.items(), key=lambda item: (-item[1], _LABEL_RANK[item[0]]))
        top_label, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        # Clamped because these land in a Pydantic model with 0..1 bounds and
        # float division does not guarantee the sum is exactly one.
        return cls(
            probabilities=normalised,
            top_label=top_label,
            confidence=min(1.0, max(0.0, top)),
            margin=min(1.0, max(0.0, top - runner_up)),
        )


# ── Distribution helpers ──────────────────────────────────────────────────────


def normalise(probabilities: Mapping[Label, float]) -> dict[Label, float]:
    """Project any mapping onto a proper distribution over :data:`SCORED_LABELS`.

    Missing labels are zero and negatives are clipped, so a backend that ranks
    only two classes composes with one that ranks four without either of them
    knowing about the other.
    """
    clipped = {label: max(0.0, probabilities.get(label, 0.0)) for label in SCORED_LABELS}
    total = sum(clipped.values())
    if total <= 0.0:
        return dict.fromkeys(SCORED_LABELS, 1.0 / len(SCORED_LABELS))
    return {label: value / total for label, value in clipped.items()}


def softmax(logits: Mapping[Label, float]) -> dict[Label, float]:
    """Softmax over whichever labels the mapping carries."""
    # The shift is not cosmetic: an unshifted exp() of a large logit overflows to
    # inf and turns the whole distribution into nan.
    highest = max(logits.values())
    exponentiated = {label: math.exp(value - highest) for label, value in logits.items()}
    total = sum(exponentiated.values())
    return {label: value / total for label, value in exponentiated.items()}


def blend(
    primary: Mapping[Label, float], lexicon: Mapping[Label, float], lexicon_weight: float
) -> dict[Label, float]:
    """Convex combination of a neural distribution and a lexicon one.

    Mixing probabilities rather than logits keeps the result a distribution
    whatever the two backends disagree about, and keeps ``lexicon_weight``
    readable as "how much of the answer the lexicon owns".
    """
    weight = min(1.0, max(0.0, lexicon_weight))
    left, right = normalise(primary), normalise(lexicon)
    return {label: (1.0 - weight) * left[label] + weight * right[label] for label in SCORED_LABELS}


# ── Calibration ───────────────────────────────────────────────────────────────
#
# READ THIS BEFORE TRUSTING ANY CONFIDENCE NUMBER THIS SERVICE REPORTS.
#
# Every threshold in the escalation policy — the 0.80 floor at T1, 0.72 at T2,
# the 0.65 arbitration trigger — is a claim about *probability*: "when this tier
# says 0.80, it is right about 80% of the time". A raw softmax does not make
# that claim true. Neural classification heads are systematically overconfident,
# so an uncalibrated model reports 0.97 on inputs it gets wrong one time in
# five, sails over every floor above it, and the ladder stops climbing exactly
# where it should have kept going.
#
# Temperature scaling is the one-parameter fix: fit a single scalar T on a
# held-out set (minimise NLL), then divide the logits by it before the softmax.
# T > 1 flattens the distribution, T < 1 sharpens it, and T = 1.0 is the
# identity — which means the **default of 1.0 in `ClassifierSettings` ships
# UNCALIBRATED**. Baking a model artifact into the image without also fitting
# and setting `classifier.calibration_temperature` is not a missing nicety: it
# makes every threshold in docs/02-lld.md §5 meaningless, while the dashboards
# continue to look healthy because the tier distribution still moves.


def apply_temperature(
    probabilities: Mapping[Label, float], temperature: float
) -> dict[Label, float]:
    """Temperature-scale a distribution: log, divide by T, re-softmax.

    Operating on probabilities rather than raw logits costs one logarithm and
    buys a backend-agnostic interface — the ONNX head, the lexicon and the
    blended ensemble all calibrate through the same call.
    """
    if temperature <= 0.0:
        raise ValueError(f"calibration temperature must be positive, got {temperature}")

    normalised = normalise(probabilities)
    if temperature == 1.0:
        return normalised
    logits = {
        label: math.log(max(value, _MIN_PROB)) / temperature for label, value in normalised.items()
    }
    return softmax(logits)


# ── Backend protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class ScorerBackend(Protocol):
    """A Tier-1 scoring model.

    ``score`` is synchronous on purpose: it is CPU-bound, and hiding a thread
    hop inside every implementation would make each backend responsible for a
    concurrency decision that belongs to the tier.
    """

    #: Reported as the tier's ``model_version``; must change when behaviour does.
    name: str
    #: False when the backend's artifact or runtime did not load.
    available: bool

    def score(self, text: str) -> ScoreResult: ...


# ── Lexicon backend ───────────────────────────────────────────────────────────

#: Bump whenever the tables in `preprocess.lexicon` or the mapping below change.
#: It travels in the verdict as `model_version`, and an accuracy comparison that
#: silently mixes two different scorers under one version is worse than no
#: comparison at all.
LEXICON_SCORER_NAME: Final[str] = "lexicon-1.0"

#: Softmax gain on polarity. Sets how decisive a unanimous, strongly worded text
#: is allowed to be: at 4.0 a saturated ±1 polarity reaches ~0.94 confidence,
#: which is about as far as a bag of words has earned the right to go.
_POLARITY_GAIN: Final[float] = 4.0

#: Pull toward NEUTRAL, strongest when nothing at all matched. Without it,
#: evidence-free text would land wherever rounding noise pointed.
_NEUTRAL_GAIN: Final[float] = 1.0

#: Pull toward MIXED, scaled by how evenly the hits are split.
_MIXED_GAIN: Final[float] = 2.0

#: Hits needed for the evidence factor to reach one half. Small on purpose: two
#: matched terms is real evidence, ten is not five times as much.
_EVIDENCE_HALF_LIFE: Final[float] = 2.0

#: Emoji strength that counts as one lexical hit. Below it the emoji is a
#: garnish rather than a judgement.
_EMOJI_EVIDENCE_FLOOR: Final[float] = 0.30

#: How much a negation and a contrastive marker each dampen the polarity signal.
#: These are the two constructions a bag-of-words scorer mis-scopes, so their
#: presence is a statement about *this scorer's* reliability, not about the text.
_NEGATION_PENALTY: Final[float] = 0.25
_CONTRASTIVE_PENALTY: Final[float] = 0.60


def lexicon_distribution(score: LexiconScore) -> dict[Label, float]:
    """Map a :class:`LexiconScore` onto a four-way distribution.

    The label this produces matters less than the confidence attached to it. The
    confidence is what the escalation policy compares against
    ``tiers.t1_classifier.confidence_floor``, which makes an honest number the
    single most valuable thing this function can return: a scorer that answers
    0.9 to everything does not merely lose accuracy, it disables the ladder,
    because nothing ever escalates and the expensive tiers never see the inputs
    they exist for.

    Three signals push it down, all of them derived from what the lexicon
    actually saw. ``evidence`` is thin when few terms matched. ``conflict`` is
    high when positive and negative hits are evenly split — the "great screen,
    terrible battery" shape. ``reliability`` falls when negation or contrast is
    present, because those are precisely the constructions this kind of scorer
    gets wrong.
    """
    positive = score.positive_hits + (1 if score.emoji_polarity >= _EMOJI_EVIDENCE_FLOOR else 0)
    negative = score.negative_hits + (1 if score.emoji_polarity <= -_EMOJI_EVIDENCE_FLOOR else 0)
    hits = positive + negative

    evidence = hits / (hits + _EVIDENCE_HALF_LIFE) if hits else 0.0
    conflict = 1.0 - abs(positive - negative) / hits if hits else 0.0
    reliability = 1.0 / (
        1.0
        + _NEGATION_PENALTY * score.negation_count
        + _CONTRASTIVE_PENALTY * score.contrastive_count
    )

    polar = _POLARITY_GAIN * score.overall_polarity * reliability
    return softmax(
        {
            Label.POSITIVE: polar,
            Label.NEGATIVE: -polar,
            Label.NEUTRAL: _NEUTRAL_GAIN * (1.0 - evidence),
            Label.MIXED: _MIXED_GAIN * conflict * evidence,
        }
    )


class LexiconScorer:
    """Dependency-free fallback backend.

    Always available, which is what lets Tier 1 promise it can serve on a bare
    Python install. It is also the ensemble partner the ONNX head is blended
    with, so the two roles share one implementation.
    """

    name: str
    available: bool

    def __init__(self) -> None:
        self.name = LEXICON_SCORER_NAME
        self.available = True

    def score(self, text: str) -> ScoreResult:
        return ScoreResult.from_probabilities(lexicon_distribution(score_text(text)))


# ── ONNX backends ─────────────────────────────────────────────────────────────

#: Bump alongside a change to the inference path, independently of the artifact.
ONNX_SCORER_VERSION: Final[str] = "1.0"

#: How a sequence-classification model's output columns map onto labels, keyed
#: by how many there are. Two-class heads (the SST-2 family) say nothing about
#: NEUTRAL or MIXED, and this leaves them at zero rather than inventing mass
#: for them — the lexicon blend in the tier is where those two classes come
#: from. Not used by :class:`ZeroShotOnnxScorer`, whose 2-class head means
#: something else entirely (entailment/not-entailment) — see its docstring.
_ONNX_LABEL_ORDERS: Final[dict[int, tuple[Label, ...]]] = {
    2: (Label.NEGATIVE, Label.POSITIVE),
    3: (Label.NEGATIVE, Label.NEUTRAL, Label.POSITIVE),
}

#: BERT-family positional embeddings stop here; a longer sequence is a crash,
#: not a slower inference.
_ONNX_MAX_TOKENS: Final[int] = 512

_DEFAULT_TOKENIZER_FILE: Final[str] = "tokenizer.json"


def _import_optional(name: str) -> Any:
    """Import a module from an optional extra at call time.

    Resolved dynamically rather than with a top-level ``import`` so that this
    module imports, and this file type-checks, identically whether or not the
    ``onnx`` extra is installed — which is the whole point of the extra.
    """
    return importlib.import_module(name)


@dataclass(frozen=True, slots=True)
class _OnnxSession:
    """A loaded model + tokenizer, ready for either ONNX backend to score with."""

    session: Any
    tokenizer: Any
    numpy: Any
    input_names: frozenset[str]


def _load_onnx_session(
    model_path: Path | None, tokenizer_path: Path | None
) -> _OnnxSession | None:
    """Load the artifact both ONNX backends need, or ``None`` on any failure.

    Shared rather than duplicated in each backend's ``__init__``: the loading
    and error-handling is identical regardless of what the model's output
    means, which is the one thing that differs between them.
    """
    if model_path is None or not model_path.is_file():
        _log.info("classifier.onnx_artifact_absent", model_path=str(model_path))
        return None

    resolved_tokenizer_path = tokenizer_path or model_path.parent / _DEFAULT_TOKENIZER_FILE
    if not resolved_tokenizer_path.is_file():
        _log.warning(
            "classifier.onnx_tokenizer_absent", tokenizer_path=str(resolved_tokenizer_path)
        )
        return None

    try:
        onnxruntime = _import_optional("onnxruntime")
        tokenizers = _import_optional("tokenizers")
        numpy = _import_optional("numpy")
    except ImportError as exc:
        _log.info("classifier.onnx_runtime_absent", error=str(exc))
        return None

    try:
        options = onnxruntime.SessionOptions()
        # One thread per inference: the service already runs many scorings
        # concurrently through the tier's worker pool, so intra-op parallelism
        # here buys nothing and contends with everything.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        tokenizer = tokenizers.Tokenizer.from_file(str(resolved_tokenizer_path))
    except Exception as exc:  # onnxruntime raises its own unrelated hierarchy
        _log.warning("classifier.onnx_load_failed", error=str(exc), model_path=str(model_path))
        return None

    input_names = frozenset(spec.name for spec in session.get_inputs())
    return _OnnxSession(session=session, tokenizer=tokenizer, numpy=numpy, input_names=input_names)


class OnnxScorer:
    """Quantised sequence-classification head executed on CPU through onnxruntime.

    Construction is the only place that can fail: everything optional is
    imported and loaded here, and any failure leaves :attr:`available` false
    instead of raising. A missing model must degrade the service, never stop it
    from starting.
    """

    name: str
    available: bool

    def __init__(self, settings: Settings) -> None:
        config = settings.classifier
        self.name = _onnx_name(config.model_path)
        self.available = False
        self._loaded: _OnnxSession | None = None

        loaded = _load_onnx_session(config.model_path, config.tokenizer_path)
        if loaded is None:
            return
        self._loaded = loaded
        self.available = True
        _log.info("classifier.onnx_loaded", backend=self.name, model_path=str(config.model_path))

    def score(self, text: str) -> ScoreResult:
        if self._loaded is None:
            raise ScorerUnavailableError(f"{self.name} did not load")
        numpy = self._loaded.numpy
        tokenizer = self._loaded.tokenizer
        input_names = self._loaded.input_names

        encoding = tokenizer.encode(text)
        ids = encoding.ids[:_ONNX_MAX_TOKENS]
        mask = encoding.attention_mask[:_ONNX_MAX_TOKENS]

        feeds: dict[str, Any] = {
            "input_ids": numpy.asarray([ids], dtype=numpy.int64),
            "attention_mask": numpy.asarray([mask], dtype=numpy.int64),
        }
        # Single-segment classification, so the type ids are all zero — but a
        # BERT export still declares the input and refuses to run without it.
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = numpy.zeros((1, len(ids)), dtype=numpy.int64)

        declared = {name: value for name, value in feeds.items() if name in input_names}
        outputs = self._loaded.session.run(None, declared)
        logits = [float(value) for value in outputs[0][0]]

        order = _ONNX_LABEL_ORDERS.get(len(logits))
        if order is None:
            raise ScorerUnavailableError(
                f"{self.name} produced {len(logits)} classes; expected one of "
                f"{sorted(_ONNX_LABEL_ORDERS)}"
            )
        return ScoreResult.from_probabilities(softmax(dict(zip(order, logits, strict=True))))


#: What each label means for the purpose of an entailment check. Written as a
#: claim about the text, matching the "This example is {}." style templates
#: the model was trained to score — deviating from that phrasing measurably
#: hurts a zero-shot NLI model's calibration, so this is not free-form.
_HYPOTHESIS_TEMPLATES: Final[dict[Label, str]] = {
    Label.POSITIVE: "This text expresses a positive sentiment.",
    Label.NEGATIVE: "This text expresses a negative sentiment.",
    Label.NEUTRAL: "This text expresses a neutral sentiment, with no positive or negative opinion.",
    Label.MIXED: "This text expresses mixed sentiment, both positive and negative.",
}

#: Row order fixed once so the batch built in `score` and the labels zipped
#: onto its output can never drift apart.
_HYPOTHESIS_ORDER: Final[tuple[Label, ...]] = tuple(_HYPOTHESIS_TEMPLATES)

#: `MoritzLaurer/deberta-v3-base-zeroshot-v1`'s id2label — entailment first.
#: Fixed by the checkpoint, not configurable: this is what its two output
#: columns mean, not a convention this module chose.
_ENTAILMENT_COLUMN: Final[int] = 0


class ZeroShotOnnxScorer:
    """Zero-shot sentiment via NLI entailment, executed on CPU through onnxruntime.

    The default Tier-1 backend. Where :class:`OnnxScorer` needs a head
    fine-tuned for exactly this label set, this one needs a head fine-tuned
    for *entailment* — "does the premise entail the hypothesis" — and answers
    an arbitrary classification question by templating each candidate label
    into a hypothesis and checking whether the text entails it. That is the
    standard technique behind HF's ``zero-shot-classification`` pipeline
    (Yin et al., 2019; see also arxiv.org/abs/2312.17543 for why it composes
    well with an encoder this size), and it earns its place here for a
    concrete reason a fine-tuned head does not have: it can score `MIXED`
    directly, rather than needing a lexicon blend to manufacture it from a
    two- or three-class head that was never trained to express it.

    One inference call scores all four labels — the batch dimension is the
    label, not separate texts — because they share nothing else about the
    request and this is CPU work inside Tier 1's tight timeout budget.
    """

    name: str
    available: bool

    def __init__(self, settings: Settings) -> None:
        config = settings.classifier
        self.name = f"{_onnx_name(config.model_path)}-zeroshot"
        self.available = False
        self._loaded: _OnnxSession | None = None

        loaded = _load_onnx_session(config.model_path, config.tokenizer_path)
        if loaded is None:
            return
        self._loaded = loaded
        self.available = True
        _log.info("classifier.onnx_loaded", backend=self.name, model_path=str(config.model_path))

    def score(self, text: str) -> ScoreResult:
        if self._loaded is None:
            raise ScorerUnavailableError(f"{self.name} did not load")
        numpy = self._loaded.numpy
        tokenizer = self._loaded.tokenizer
        input_names = self._loaded.input_names

        # Each row is (premise, hypothesis) for one candidate label. Encoded
        # individually rather than via `encode_batch` + `enable_padding`,
        # which would require knowing this tokenizer's pad token id; padding
        # to 0 by hand and masking it out via `attention_mask` is correct
        # regardless of what token id 0 happens to mean, because a masked
        # position never enters the attention computation.
        encodings = [
            tokenizer.encode(text, _HYPOTHESIS_TEMPLATES[label]) for label in _HYPOTHESIS_ORDER
        ]
        width = min(_ONNX_MAX_TOKENS, max(len(e.ids) for e in encodings))

        ids = numpy.zeros((len(encodings), width), dtype=numpy.int64)
        mask = numpy.zeros((len(encodings), width), dtype=numpy.int64)
        type_ids = numpy.zeros((len(encodings), width), dtype=numpy.int64)
        for row, encoding in enumerate(encodings):
            length = min(width, len(encoding.ids))
            ids[row, :length] = encoding.ids[:length]
            mask[row, :length] = encoding.attention_mask[:length]
            type_ids[row, :length] = encoding.type_ids[:length]

        feeds: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = type_ids

        declared = {name: value for name, value in feeds.items() if name in input_names}
        outputs = self._loaded.session.run(None, declared)
        rows = outputs[0]
        if rows.shape[-1] != 2:
            raise ScorerUnavailableError(
                f"{self.name} produced a {rows.shape[-1]}-class head; "
                "expected an entailment/not-entailment NLI head with 2"
            )

        entailment_logits = {
            label: float(rows[row, _ENTAILMENT_COLUMN])
            for row, label in enumerate(_HYPOTHESIS_ORDER)
        }
        # Softmax *across* labels' entailment logits, not within each pair's
        # own two columns — the within-pair softmax says "is this hypothesis
        # true at all," the across-label one says "which hypothesis is truest
        # relative to the others," and it is the second question a single-
        # label classifier needs answered. Same procedure HF's zero-shot
        # pipeline uses in single-label mode.
        return ScoreResult.from_probabilities(softmax(entailment_logits))


def _onnx_name(model_path: Path | None) -> str:
    """Version string identifying the artifact, not just the code path."""
    if model_path is None:
        return f"onnx-{ONNX_SCORER_VERSION}"
    return f"{model_path.stem}-onnx-{ONNX_SCORER_VERSION}"


# ── Selection ─────────────────────────────────────────────────────────────────

#: `ClassifierSettings.scorer_kind` values, mapped onto the backend they build.
_ONNX_BACKENDS: Final[dict[str, type[OnnxScorer | ZeroShotOnnxScorer]]] = {
    "zero_shot_nli": ZeroShotOnnxScorer,
    "sequence_classification": OnnxScorer,
}


def build_scorer(settings: Settings) -> ScorerBackend:
    """Return the best Tier-1 backend that actually loaded.

    ONNX wins when both its artifact and its runtime are present; otherwise the
    lexicon takes over. Which ONNX backend is even attempted is
    ``classifier.scorer_kind`` — the two are not interchangeable at the same
    artifact, so this is not autodetected. The fallback is logged at warning
    level because it is a quiet, permanent accuracy loss that is invisible
    from the outside: the service keeps answering, the tier keeps reporting
    verdicts, and only the confidence numbers get worse.
    """
    backend_cls = _ONNX_BACKENDS[settings.classifier.scorer_kind]
    onnx = backend_cls(settings)
    if onnx.available:
        _log.info("classifier.backend_selected", backend=onnx.name)
        return onnx

    _log.warning(
        "classifier.backend_fallback",
        backend=LEXICON_SCORER_NAME,
        wanted=onnx.name,
        model_path=str(settings.classifier.model_path),
        hint="install air-classifier-service[onnx] and bake the model artifact into the image",
    )
    return LexiconScorer()
