# ruff: noqa: RUF001 — non-ASCII look-alike characters are the subject matter here.
"""Language detection tests.

The asymmetry to keep in mind while reading these: a missed detection costs one
tier hop, while a confident wrong detection routes the request to a rung that
cannot read it. Every assertion below is written from that side of the trade.
"""

from __future__ import annotations

import pytest

from air_sentiment.preprocess.language import MIN_DETECTION_CHARS, detect_language

LATIN_CASES: list[tuple[str, str]] = [
    ("The battery life is great but the camera is awful.", "en"),
    ("I would not recommend this to anyone in my team.", "en"),
    ("El producto es excelente pero la cámara es muy mala.", "es"),
    ("No me gusta el servicio, tampoco la calidad de las piezas.", "es"),
    ("Le produit est excellent mais la caméra est très mauvaise.", "fr"),
    ("Je ne suis pas content de la livraison, elle est en retard.", "fr"),
    ("Das Produkt ist sehr gut, aber die Kamera ist nicht schön.", "de"),
    ("Ich habe das Paket nicht bekommen und der Support war leider schlecht.", "de"),
    ("O produto é ótimo mas a câmera é muito ruim.", "pt"),
    ("Não gostei do atendimento e a entrega foi muito demorada.", "pt"),
    ("Il prodotto è ottimo ma la fotocamera è molto scarsa.", "it"),
    ("Non sono soddisfatto della consegna, è arrivata in ritardo.", "it"),
]

SCRIPT_CASES: list[tuple[str, str]] = [
    ("Этот товар очень хороший, я доволен покупкой", "ru"),
    ("这个产品非常好，我很喜欢它的质量", "zh"),
    ("هذا المنتج رائع جدا وانا سعيد بالشراء", "ar"),
    ("यह उत्पाद बहुत अच्छा है और मैं खुश हूं", "hi"),
    ("この製品はとても良いです、買ってよかった", "ja"),
]


@pytest.mark.parametrize(("text", "expected"), LATIN_CASES)
def test_latin_languages_are_identified_from_stopwords(text: str, expected: str) -> None:
    info = detect_language(text)

    assert info.code == expected
    assert info.confidence > 0.5


@pytest.mark.parametrize(("text", "expected"), SCRIPT_CASES)
def test_non_latin_scripts_are_identified_from_code_points(text: str, expected: str) -> None:
    info = detect_language(text)

    assert info.code == expected
    assert info.confidence > 0.5


@pytest.mark.parametrize("text", ["", "   ", "bad", "so slow", "\U0001f60d\U0001f44d", "!!!"])
def test_text_too_short_to_judge_is_undetermined(text: str) -> None:
    info = detect_language(text)

    assert info.code == "und"
    assert info.confidence == 0.0


def test_the_short_text_threshold_is_the_documented_one() -> None:
    just_under = "a" * (MIN_DETECTION_CHARS - 1)
    just_over = "the product " + "a" * (MIN_DETECTION_CHARS - 1)

    assert detect_language(just_under).code == "und"
    assert detect_language(just_over).code != "und"


def test_ascii_latin_without_stopword_evidence_falls_back_to_english_weakly() -> None:
    info = detect_language("Excellent product, works perfectly")

    assert info.code == "en"
    # Low enough that the policy can still decide to spend a rung on it.
    assert info.confidence < 0.6


def test_a_single_borrowed_word_does_not_flip_the_answer() -> None:
    info = detect_language("The café downstairs is great and the staff are friendly")

    assert info.code == "en"


def test_confidence_stays_inside_the_schema_bounds() -> None:
    samples = [text for text, _ in LATIN_CASES + SCRIPT_CASES]
    samples += ["", "hi", "1234567890123456", "\U0001f600" * 20]

    for text in samples:
        info = detect_language(text)
        assert 0.0 <= info.confidence <= 1.0


def test_digits_and_punctuation_alone_are_undetermined() -> None:
    info = detect_language("1234 5678 9012 3456 !!! ...")

    assert info.code == "und"
    assert info.confidence == 0.0
