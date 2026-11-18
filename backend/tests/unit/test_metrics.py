import pytest

from ledger_agent.answer_format import extract_final_answer, is_abstention
from ledger_agent.evaluation.metrics import (
    citation_precision,
    is_numeric_answer,
    numeric_correct,
    page_hit,
)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("$1577.00", True),
        ("14.9%", True),
        ("0.66", True),
        ("-0.02", True),
        ("$8.70 billion", True),
        ("Data Center", False),
        ("The consumer segment shrunk by 0.9% organically.", False),
        ("Yes", False),
    ],
)
def test_numeric_answer_detection(answer: str, expected: bool) -> None:
    assert is_numeric_answer(answer) is expected


@pytest.mark.parametrize(
    ("gold", "prediction"),
    [
        ("$1577.00", "Capex was $1,577 million [1].\nFinal answer: $1,577 million"),
        ("$1577.00", "Final answer: $1.577 billion"),
        ("14.9%", "Final answer: 14.86%"),
        ("24.26", "Final answer: 24.3"),
        ("0.66", "Final answer: 0.6612"),
        # Une référence négative accepte un signe moins, des parenthèses ou un mot de baisse
        ("-0.02", "Final answer: -0.02"),
        ("-0.02", "Final answer: (0.02)"),
        ("-4.1%", "Final answer: a decline of 4.1%"),
        # Une référence positive accepte une formulation en baisse (« de combien a-t-il baissé »)
        ("1.9%", "Final answer: margin decreased by 1.9%"),
    ],
)
def test_numeric_matches(gold: str, prediction: str) -> None:
    assert numeric_correct(gold, prediction)


@pytest.mark.parametrize(
    ("gold", "prediction"),
    [
        ("$1577.00", "Final answer: $1,373 million"),
        ("14.9%", "Final answer: 16.2%"),
        # Le bon chiffre n'apparaît que dans le raisonnement, pas dans la réponse finale
        ("$1577.00", "In 2018 capex was 1,577 but in 2017 it was 1,373.\nFinal answer: $1,373"),
        # Signe inversé : une hausse ne répond pas à une baisse
        ("-4.1%", "Final answer: an increase of 4.1%"),
        # « 3M » est une entreprise, « FY2018 » une année : ni 3 ni 2018
        ("3", "Final answer: 3M did not disclose it"),
        ("2018", "Final answer: see FY2018"),
        # Un ratio de 1.5 ne vaut pas 1.5 milliard
        ("1.5", "Final answer: 1.5 billion"),
    ],
)
def test_numeric_mismatches(gold: str, prediction: str) -> None:
    assert not numeric_correct(gold, prediction)


def test_final_answer_takes_the_last_one_and_ignores_markdown() -> None:
    assert extract_final_answer("just 42") == "just 42"
    assert extract_final_answer("a\nFinal answer: 1\nFinal Answer: 2") == "2"
    assert extract_final_answer("**Final answer:** Yes") == "Yes"


def test_abstention_detection() -> None:
    assert is_abstention("INSUFFICIENT_CONTEXT")
    assert is_abstention("Final answer: The information cannot be determined.")
    assert not is_abstention("Final answer: $1,577 million")


def test_real_answer_mentioning_missing_context_is_not_an_abstention() -> None:
    # Cas réel (financebench_id_01107) : une vraie réponse, compté à tort comme abstention
    text = (
        "**Final answer:** CVS Health reported a materially important ongoing legal battle "
        "in 2022 (opioid litigation charges), but the context does not show any such "
        "disclosures for 2021 or 2020."
    )
    assert not is_abstention(text)


def test_page_hit_and_citation_precision() -> None:
    gold = [("3M_2018_10K", 59)]

    assert page_hit([("3M_2018_10K", 12), ("3M_2018_10K", 59)], gold)
    assert not page_hit([("3M_2018_10K", 58)], gold)
    assert citation_precision([("3M_2018_10K", 59), ("3M_2018_10K", 1)], gold) == 0.5
    assert citation_precision([], gold) is None
