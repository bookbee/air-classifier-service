"""The internal currency of the ladder.

``TierVerdict`` is the single shape every tier produces. Because it is uniform,
the orchestrator and the escalation policy never name a concrete tier, and
adding a fifth rung is a registration rather than a refactor.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from air_classifier_service.constants import (
    SILENT_STOP_REASONS,
    EscalationReason,
    Intent,
    Label,
    Tier,
    Tone,
    Topic,
)
from air_classifier_service.schemas.common import (
    Aspect,
    Emotion,
    LabeledScore,
    SentimentResult,
    TierUsage,
)

# ── What a tier returns ───────────────────────────────────────────────────────


class TierVerdict(BaseModel):
    """One rung's answer, successful or not."""

    tier: Tier
    model_version: str = "unknown"
    result: SentimentResult
    aspects: list[Aspect] = Field(default_factory=list)
    emotions: list[Emotion] = Field(default_factory=list)
    #: `/v1/classify` fields. `None`/empty on every T0/T1 verdict and on a T2/T3
    #: verdict for any other resource — those tiers structurally cannot produce
    #: them (no lexicon or classifier head for tone/topic/intent), and the
    #: other resources' prompts never ask for them. `urgency` is the one
    #: exception: it *can* come from a lower tier, via the heuristic in
    #: `pipeline/service.py`, which runs after the ladder rather than inside it
    #: — so it is never set on a `TierVerdict` directly, only on the response.
    tone: LabeledScore[Tone] | None = None
    topics: list[LabeledScore[Topic]] = Field(default_factory=list)
    intent: LabeledScore[Intent] | None = None
    label_margin: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Top-1 minus top-2 probability."
    )
    language_code: str | None = None
    sarcasm_detected: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    succeeded: bool = True
    error: str | None = None
    usage: TierUsage = Field(default_factory=TierUsage)
    #: Free-form, never used for control flow — diagnostics only.
    detail: dict[str, str] = Field(default_factory=dict)
    #: The rung answered, but from a fallback it only uses because a dependency
    #: is missing (Tier 1 on its lexicon when the ONNX artifact is absent). If
    #: this verdict is the one returned, the response is `degraded` — the same
    #: "a dependency being down is why the answer is worse" the ladder already
    #: reports for failed and unavailable rungs (SIT CL-005).
    fallback: bool = False

    @classmethod
    def failed(
        cls,
        tier: Tier,
        error: str,
        *,
        model_version: str = "unknown",
        latency_ms: float = 0.0,
    ) -> TierVerdict:
        """A rung that timed out, raised, or returned unusable output.

        Confidence is zero so that this verdict can never win
        :meth:`select_best`, and the policy always escalates past it.
        """
        return cls(
            tier=tier,
            model_version=model_version,
            result=SentimentResult(label=Label.UNKNOWN, polarity=0.0, confidence=0.0),
            latency_ms=latency_ms,
            succeeded=False,
            error=error[:500],
        )

    @classmethod
    def abstained(
        cls, tier: Tier, *, model_version: str = "unknown", latency_ms: float = 0.0
    ) -> TierVerdict:
        """A rung that ran fine but declined to judge (e.g. no T0 shortcut matched)."""
        return cls(
            tier=tier,
            model_version=model_version,
            result=SentimentResult(label=Label.UNKNOWN, polarity=0.0, confidence=0.0),
            latency_ms=latency_ms,
            succeeded=True,
        )

    @property
    def is_usable(self) -> bool:
        """Whether this verdict could be returned to a caller."""
        return self.succeeded and self.result.label is not Label.UNKNOWN


# ── What the policy decides ───────────────────────────────────────────────────


class EscalationDecision(BaseModel):
    """The policy's answer to 'should we climb?'."""

    escalate: bool
    reasons: list[EscalationReason] = Field(default_factory=list)

    @property
    def primary_reason(self) -> EscalationReason | None:
        return self.reasons[0] if self.reasons else None

    def as_label(self) -> str:
        """Comma-joined reason codes, stable enough for a metric label."""
        return ",".join(r.value for r in self.reasons)

    @classmethod
    def stop(cls, *reasons: EscalationReason) -> EscalationDecision:
        chosen = list(reasons) or [EscalationReason.CONFIDENT]
        return cls(escalate=False, reasons=chosen)

    @classmethod
    def climb(cls, *reasons: EscalationReason) -> EscalationDecision:
        return cls(escalate=True, reasons=list(reasons))


# ── What the caller sees ──────────────────────────────────────────────────────


class EscalationStep(BaseModel):
    """One row of the trace returned to the caller."""

    tier: Tier
    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    succeeded: bool = True
    escalated_because: str | None = Field(
        default=None, description="Comma-joined reason codes; null means this rung was terminal."
    )

    @classmethod
    def build(cls, verdict: TierVerdict, decision: EscalationDecision) -> EscalationStep:
        # A terminal step carries null rather than a stop-reason code: the caller
        # asked "why did you keep going", and "you didn't" is best said with null.
        # `NEXT_TIER_UNAVAILABLE` is the one stop reason that does not get this
        # silence — it means the ladder wanted to keep going and a dependency
        # was not there, which is worth a caller (or DevOps, reading the same
        # response) actually seeing rather than reading as a confident answer.
        because: str | None = None
        if decision.escalate or decision.primary_reason not in SILENT_STOP_REASONS:
            because = decision.as_label()
        return cls(
            tier=verdict.tier,
            label=verdict.result.label,
            confidence=verdict.result.confidence,
            latency_ms=round(verdict.latency_ms, 2),
            succeeded=verdict.succeeded,
            escalated_because=because,
        )


class OrchestrationResult(BaseModel):
    """Everything the ladder produced, before it is shaped per resource."""

    best: TierVerdict
    verdicts: list[TierVerdict]
    steps: list[EscalationStep]
    degraded: bool = False
    #: The deciding verdict met its own rung's floor: the ladder stopped because
    #: it was confident, not because it was capped or had nowhere to climb, and
    #: the answer is not a fallback backend's. A result can be `degraded` *and*
    #: settled — a dead T2 hopped on the way to a confident T3 — and such an
    #: answer is as good as one reached with every rung up, so it is safe to
    #: cache (SIT CL-006i regression: while T2 was down, nothing was cached).
    settled: bool = False
    cached: bool = False

    @property
    def total_usage(self) -> TierUsage:
        total = TierUsage()
        for verdict in self.verdicts:
            total = total + verdict.usage
        return total


# ── What we ask an LLM for ────────────────────────────────────────────────────


class LLMSentimentOutput(BaseModel):
    """Structured-output contract for Tiers 2 and 3.

    Deliberately flatter and stricter than the API response: the model's job is
    to fill fields, not to design an envelope.

    ``extra="forbid"`` becomes ``additionalProperties: false`` in the emitted
    JSON Schema, which providers require. Numeric bounds (``ge``/``le``) are
    *not* expressible in that schema dialect — providers ignore them, so Pydantic
    enforces them on the way back in. An out-of-range value is a validation
    failure, which counts as a tier failure and triggers one bounded repair
    retry before escalation.
    """

    model_config = ConfigDict(extra="forbid")

    label: Label = Field(description="Overall sentiment of the text.")
    polarity: float = Field(
        ge=-1.0, le=1.0, description="Valence from -1.0 (fully negative) to 1.0 (fully positive)."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="How certain you are, 0.0 to 1.0. Be honest, not generous."
    )
    rationale: str = Field(max_length=300, description="One or two sentences justifying the label.")
    emotions: list[Emotion] = Field(
        default_factory=list, max_length=5, description="Salient emotions, strongest first."
    )
    aspects: list[Aspect] = Field(
        default_factory=list,
        max_length=10,
        description="Per-facet sentiment when the text discusses distinct facets.",
    )
    sarcasm_detected: bool = Field(
        default=False, description="True when the literal reading inverts the intended one."
    )
    language_code: str = Field(
        default="und", max_length=8, description="ISO 639-1 code of the input text."
    )
    #: `/v1/classify` only. Optional rather than a second output schema: every
    #: resource shares one structured-output contract, and the user turn's
    #: resource framing (`providers/prompts/loader.py`) is what tells the model
    #: whether to fill these or leave them at their default. A model that was
    #: never asked pays nothing for fields it did not populate.
    tone: LabeledScore[Tone] | None = Field(
        default=None, description="Emotional register — fill only when the task asks for it."
    )
    topics: list[LabeledScore[Topic]] = Field(
        default_factory=list,
        max_length=6,
        description="What the text discusses — fill only when the task asks for it.",
    )
    intent: LabeledScore[Intent] | None = Field(
        default=None,
        description="What the author wants done — fill only when the task asks for it.",
    )

    def to_verdict(
        self,
        tier: Tier,
        *,
        model_version: str,
        latency_ms: float,
        usage: TierUsage,
    ) -> TierVerdict:
        return TierVerdict(
            tier=tier,
            model_version=model_version,
            result=SentimentResult(
                label=self.label,
                polarity=self.polarity,
                confidence=self.confidence,
                rationale=self.rationale or None,
            ),
            aspects=list(self.aspects),
            emotions=list(self.emotions),
            tone=self.tone,
            # A topic the model scored at zero is one it did not find: kept, it
            # reads as a finding and is counted in a summary's `topic_counts`
            # (SIT CL-006e). Dropped here rather than rejected by the schema,
            # where one noise entry would fail the whole structured answer.
            topics=[topic for topic in self.topics if topic.confidence > 0.0],
            intent=self.intent,
            # An LLM gives no runner-up distribution, so margin is derived from
            # confidence: a 0.9-confident answer implies a wide gap to second place.
            label_margin=max(0.0, min(1.0, (self.confidence - 0.5) * 2.0)),
            language_code=self.language_code or None,
            sarcasm_detected=self.sarcasm_detected,
            latency_ms=latency_ms,
            succeeded=True,
            usage=usage,
        )
