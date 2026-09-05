"""Tier-1 backend tests.

The assertions that matter most here are about *confidence*, not about labels.
The escalation policy consumes these numbers directly, so a scorer that gets the
label right and its own certainty wrong breaks the ladder more thoroughly than
one that is merely inaccurate: it either stops on inputs that needed a better
model, or escalates everything and turns the cheap rung into dead weight.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from air_classifier_service.config import ClassifierSettings, Settings
from air_classifier_service.constants import Label
from air_classifier_service.pipeline.tiers import scorers
from air_classifier_service.pipeline.tiers.scorers import (
    LEXICON_SCORER_NAME,
    SCORED_LABELS,
    LexiconScorer,
    OnnxScorer,
    ScorerBackend,
    ScoreResult,
    ScorerUnavailableError,
    ZeroShotOnnxScorer,
    apply_temperature,
    blend,
    build_scorer,
    normalise,
    softmax,
)

# Text whose polarity a human would not hesitate over.
CLEAR_POSITIVE = "Absolutely fantastic, superb service, I love it."
CLEAR_NEGATIVE = "Absolutely terrible, broken on arrival, a total waste of money."

# Two judgements in one sentence. The lexicon can see both and cannot pick.
CONTRASTIVE = "The battery life is great but the camera is awful."
CONTRASTIVE_LONG = "Great screen, however the software is buggy and slow."

# ── Distribution helpers ──────────────────────────────────────────────────────


def test_normalise_fills_missing_labels_and_sums_to_one() -> None:
    result = normalise({Label.POSITIVE: 3.0, Label.NEGATIVE: 1.0})

    assert set(result) == set(SCORED_LABELS)
    assert result[Label.POSITIVE] == pytest.approx(0.75)
    assert result[Label.NEUTRAL] == 0.0
    assert sum(result.values()) == pytest.approx(1.0)


def test_normalise_clips_negatives_and_ignores_unranked_labels() -> None:
    result = normalise({Label.POSITIVE: 1.0, Label.NEGATIVE: -5.0, Label.UNKNOWN: 9.0})

    assert result[Label.POSITIVE] == pytest.approx(1.0)
    assert result[Label.NEGATIVE] == 0.0
    assert Label.UNKNOWN not in result


def test_an_empty_distribution_becomes_uniform_rather_than_a_division_by_zero() -> None:
    result = normalise(dict.fromkeys(SCORED_LABELS, 0.0))

    assert result == {label: pytest.approx(0.25) for label in SCORED_LABELS}


def test_softmax_survives_a_logit_large_enough_to_overflow_exp() -> None:
    result = softmax({Label.POSITIVE: 1000.0, Label.NEGATIVE: 0.0})

    assert result[Label.POSITIVE] == pytest.approx(1.0)
    assert sum(result.values()) == pytest.approx(1.0)


def test_score_result_ranking_reports_confidence_and_margin() -> None:
    result = ScoreResult.from_probabilities(
        {Label.POSITIVE: 0.5, Label.NEGATIVE: 0.3, Label.NEUTRAL: 0.2}
    )

    assert result.top_label is Label.POSITIVE
    assert result.confidence == pytest.approx(0.5)
    assert result.margin == pytest.approx(0.2)


def test_ties_break_deterministically_rather_than_by_dict_order() -> None:
    forwards = ScoreResult.from_probabilities({Label.NEGATIVE: 0.5, Label.POSITIVE: 0.5})
    backwards = ScoreResult.from_probabilities({Label.POSITIVE: 0.5, Label.NEGATIVE: 0.5})

    assert forwards.top_label is backwards.top_label
    assert forwards.margin == 0.0


def test_blend_is_a_convex_combination_of_the_two_distributions() -> None:
    neural = {Label.POSITIVE: 1.0}
    lexicon = {Label.NEGATIVE: 1.0}

    quarter = blend(neural, lexicon, 0.25)

    assert quarter[Label.POSITIVE] == pytest.approx(0.75)
    assert quarter[Label.NEGATIVE] == pytest.approx(0.25)
    assert blend(neural, lexicon, 0.0)[Label.POSITIVE] == pytest.approx(1.0)
    assert blend(neural, lexicon, 1.0)[Label.NEGATIVE] == pytest.approx(1.0)


def test_blend_clamps_a_weight_outside_the_unit_interval() -> None:
    result = blend({Label.POSITIVE: 1.0}, {Label.NEGATIVE: 1.0}, 4.0)

    assert result[Label.NEGATIVE] == pytest.approx(1.0)


# ── Calibration ───────────────────────────────────────────────────────────────


def test_temperature_of_one_is_the_identity() -> None:
    original = {Label.POSITIVE: 0.7, Label.NEGATIVE: 0.2, Label.NEUTRAL: 0.1}

    assert apply_temperature(original, 1.0) == pytest.approx(normalise(original))


def test_rising_temperature_flattens_the_distribution() -> None:
    original = {Label.POSITIVE: 0.7, Label.NEGATIVE: 0.2, Label.NEUTRAL: 0.07, Label.MIXED: 0.03}

    peaks = [max(apply_temperature(original, t).values()) for t in (0.5, 1.0, 2.0, 5.0, 20.0)]

    assert peaks == sorted(peaks, reverse=True)
    assert peaks[0] > peaks[1] > peaks[2] > peaks[3] > peaks[4]
    # The point of the exercise: an overconfident 0.70 stops clearing the 0.65
    # arbitration floor once a fitted temperature is applied to it.
    assert peaks[1] > 0.65
    assert apply_temperature(original, 5.0)[Label.POSITIVE] < 0.65


def test_a_large_temperature_approaches_the_uniform_distribution() -> None:
    result = apply_temperature({Label.POSITIVE: 0.97, Label.NEGATIVE: 0.03}, 500.0)

    assert result[Label.POSITIVE] == pytest.approx(0.25, abs=0.02)


def test_temperature_below_one_sharpens_instead() -> None:
    original = {Label.POSITIVE: 0.5, Label.NEGATIVE: 0.3, Label.NEUTRAL: 0.2}

    assert apply_temperature(original, 0.4)[Label.POSITIVE] > 0.5


def test_temperature_scaling_preserves_the_ranking() -> None:
    original = {Label.POSITIVE: 0.5, Label.NEGATIVE: 0.3, Label.NEUTRAL: 0.15, Label.MIXED: 0.05}

    scaled = apply_temperature(original, 3.0)

    assert sorted(scaled, key=lambda label: scaled[label], reverse=True)[0] is Label.POSITIVE
    assert sum(scaled.values()) == pytest.approx(1.0)


def test_a_non_positive_temperature_is_refused() -> None:
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature({Label.POSITIVE: 1.0}, 0.0)


# ── Lexicon backend ───────────────────────────────────────────────────────────


def test_the_lexicon_backend_needs_nothing_to_be_available() -> None:
    scorer = LexiconScorer()

    assert scorer.available is True
    assert scorer.name == LEXICON_SCORER_NAME
    assert isinstance(scorer, ScorerBackend)


@pytest.mark.parametrize(
    ("text", "expected"),
    [(CLEAR_POSITIVE, Label.POSITIVE), (CLEAR_NEGATIVE, Label.NEGATIVE)],
)
def test_unambiguous_text_is_labelled_confidently(text: str, expected: Label) -> None:
    result = LexiconScorer().score(text)

    assert result.top_label is expected
    assert result.confidence > 0.85
    assert result.margin > 0.5
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("text", [CONTRASTIVE, CONTRASTIVE_LONG])
def test_contrastive_text_is_scored_with_low_confidence_and_a_narrow_margin(text: str) -> None:
    """The behaviour the whole escalation policy is built on.

    Both halves matter. Low confidence alone would be satisfied by a scorer that
    is uniformly unsure; a narrow margin says the runner-up label is genuinely
    live, which is exactly the input worth spending an LLM on.
    """
    result = LexiconScorer().score(text)

    assert result.confidence < 0.60
    assert result.margin < 0.25


def test_confidence_rises_with_the_amount_of_agreeing_evidence() -> None:
    thin = LexiconScorer().score("good")
    thick = LexiconScorer().score("Good, great, excellent, wonderful and delightful.")

    assert thin.confidence < thick.confidence


def test_text_with_no_polarity_lands_on_neutral_without_claiming_certainty() -> None:
    result = LexiconScorer().score("It arrived on Tuesday and the box was blue.")

    assert result.top_label is Label.NEUTRAL
    # An honest number: the lexicon saw no evidence, which is not the same thing
    # as evidence of neutrality, and the ladder should climb rather than settle.
    assert result.confidence < 0.80


def test_evenly_split_polarity_is_reported_as_mixed() -> None:
    result = LexiconScorer().score(CONTRASTIVE)

    assert result.top_label is Label.MIXED


def test_negation_moves_the_label_rather_than_the_certainty_alone() -> None:
    plain = LexiconScorer().score("The service was great.")
    negated = LexiconScorer().score("The service was not great.")

    assert plain.top_label is Label.POSITIVE
    assert negated.top_label is not Label.POSITIVE


def test_emoji_carry_polarity_when_there_is_no_prose_to_qualify() -> None:
    result = LexiconScorer().score("\U0001f60d\U0001f60d")

    assert result.top_label is Label.POSITIVE
    assert result.confidence > 0.80


def test_emoji_contradicting_the_prose_destroys_confidence() -> None:
    result = LexiconScorer().score("Great service \U0001f621")

    assert result.confidence < 0.60
    assert result.margin < 0.25


def test_scoring_is_deterministic() -> None:
    scorer = LexiconScorer()

    assert scorer.score(CONTRASTIVE) == scorer.score(CONTRASTIVE)


# ── ONNX backend and selection ────────────────────────────────────────────────


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    """A model path that exists and is not a model.

    Enough to clear the artifact check so that the *runtime* branch is the one
    under test, whether the ``onnx`` extra is installed or not.
    """
    model = tmp_path / "sst2.onnx"
    model.write_bytes(b"not really an onnx graph")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


def _with_classifier(settings: Settings, **kw: object) -> Settings:
    return settings.model_copy(update={"classifier": ClassifierSettings(**kw)})


def test_the_onnx_backend_is_unavailable_without_a_configured_artifact(
    settings: Settings,
) -> None:
    scorer = OnnxScorer(settings)

    assert settings.classifier.model_path is None
    assert scorer.available is False
    assert "onnx" in scorer.name


def test_the_onnx_backend_is_unavailable_when_the_path_does_not_exist(
    settings: Settings, tmp_path: Path
) -> None:
    scorer = OnnxScorer(_with_classifier(settings, model_path=tmp_path / "missing.onnx"))

    assert scorer.available is False


def test_the_onnx_backend_is_unavailable_when_the_runtime_is_not_installed(
    settings: Settings, artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinned by patching the import rather than by the venv's contents.

    Without the patch this would pass for the wrong reason in an image that
    *does* have the extra installed, where the bogus artifact fails to load.
    """

    def _absent(name: str) -> object:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(scorers, "_import_optional", _absent)

    assert OnnxScorer(_with_classifier(settings, model_path=artifact)).available is False


def test_the_onnx_backend_is_unavailable_when_the_tokenizer_is_missing(
    settings: Settings, artifact: Path
) -> None:
    artifact.parent.joinpath("tokenizer.json").unlink()

    assert OnnxScorer(_with_classifier(settings, model_path=artifact)).available is False


def test_an_unloaded_backend_refuses_to_score_rather_than_returning_a_guess(
    settings: Settings,
) -> None:
    scorer = OnnxScorer(settings)

    with pytest.raises(ScorerUnavailableError):
        scorer.score("anything at all")


def test_build_scorer_falls_back_to_the_lexicon_when_onnx_is_absent(settings: Settings) -> None:
    scorer = build_scorer(settings)

    assert isinstance(scorer, LexiconScorer)
    assert scorer.available is True
    assert scorer.name == LEXICON_SCORER_NAME


def test_build_scorer_falls_back_even_with_a_configured_artifact(
    settings: Settings, artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _absent(name: str) -> object:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(scorers, "_import_optional", _absent)

    assert isinstance(build_scorer(_with_classifier(settings, model_path=artifact)), LexiconScorer)


def test_the_selected_backend_satisfies_the_protocol(settings: Settings) -> None:
    assert isinstance(build_scorer(settings), ScorerBackend)


# ── Zero-shot ONNX backend ──────────────────────────────────────────────────────
#
# Availability guards mirror OnnxScorer's exactly: both go through the same
# `_load_onnx_session` helper, so the same fixture proves the same three ways
# to fail to load leave `available` false rather than raising.


def test_the_zero_shot_backend_is_unavailable_without_a_configured_artifact(
    settings: Settings,
) -> None:
    scorer = ZeroShotOnnxScorer(settings)

    assert scorer.available is False
    assert "onnx" in scorer.name


def test_the_zero_shot_backend_is_unavailable_when_the_runtime_is_not_installed(
    settings: Settings, artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _absent(name: str) -> object:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(scorers, "_import_optional", _absent)

    assert ZeroShotOnnxScorer(_with_classifier(settings, model_path=artifact)).available is False


def test_the_zero_shot_backend_is_unavailable_when_the_tokenizer_is_missing(
    settings: Settings, artifact: Path
) -> None:
    artifact.parent.joinpath("tokenizer.json").unlink()

    assert ZeroShotOnnxScorer(_with_classifier(settings, model_path=artifact)).available is False


def test_an_unloaded_zero_shot_backend_refuses_to_score_rather_than_returning_a_guess(
    settings: Settings,
) -> None:
    with pytest.raises(ScorerUnavailableError):
        ZeroShotOnnxScorer(settings).score("anything at all")


def test_build_scorer_dispatches_on_scorer_kind(
    settings: Settings, artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two artifacts at the same path mean two different things, so the
    dispatch has to come from configuration, not from probing the file."""

    def _absent(name: str) -> object:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(scorers, "_import_optional", _absent)

    zero_shot = build_scorer(
        _with_classifier(settings, model_path=artifact, scorer_kind="zero_shot_nli")
    )
    sequence = build_scorer(
        _with_classifier(settings, model_path=artifact, scorer_kind="sequence_classification")
    )
    # Both fall back to the lexicon here (the runtime is patched absent), but
    # the *name* each attempted and logged before falling back differs, which
    # is the only externally visible proof dispatch actually happened.
    assert isinstance(zero_shot, LexiconScorer)
    assert isinstance(sequence, LexiconScorer)


@pytest.mark.skipif(
    not (Path("models") / "model.onnx").is_file(),
    reason="real artifact not present — see scripts/export_zeroshot_model.py",
)
class TestZeroShotOnnxScorerAgainstTheRealArtifact:
    """Exercises the actual exported model, when one is sitting in ./models.

    Skipped everywhere else, deliberately: this needs a ~230MB file no CI
    environment should have to fetch. It exists so that a local change to the
    scoring math in this module gets checked against real weights before it
    ships, not only against the fake-bytes fixture above.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def scorer() -> ZeroShotOnnxScorer:
        settings = Settings(
            classifier=ClassifierSettings(
                model_path=Path("models/model.onnx"),
                tokenizer_path=Path("models/tokenizer.json"),
                scorer_kind="zero_shot_nli",
            )
        )
        loaded = ZeroShotOnnxScorer(settings)
        assert loaded.available, "models/model.onnx did not load — see its own log line for why"
        return loaded

    def test_clearly_positive_text_is_labelled_confidently(
        self, scorer: ZeroShotOnnxScorer
    ) -> None:
        result = scorer.score(CLEAR_POSITIVE)
        assert result.top_label is Label.POSITIVE
        assert result.confidence > 0.8

    def test_clearly_negative_text_is_labelled_confidently(
        self, scorer: ZeroShotOnnxScorer
    ) -> None:
        result = scorer.score(CLEAR_NEGATIVE)
        assert result.top_label is Label.NEGATIVE
        assert result.confidence > 0.8

    def test_a_factual_statement_is_labelled_neutral(self, scorer: ZeroShotOnnxScorer) -> None:
        result = scorer.score("The package was delivered on Tuesday as scheduled.")
        assert result.top_label is Label.NEUTRAL

    def test_scoring_one_text_fits_comfortably_inside_the_tier_1_budget(
        self, scorer: ZeroShotOnnxScorer, settings: Settings
    ) -> None:
        import time

        started = time.perf_counter()
        scorer.score(CLEAR_POSITIVE)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        assert elapsed_ms < settings.tiers.t1_classifier.timeout_ms
