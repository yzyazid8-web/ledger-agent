import pytest

from ledger_agent.evaluation.metrics import (
    citation_precision,
    extract_final_answer,
    is_abstention,
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
        ("-0.02", "Final answer: a decrease of 0.02"),
        ("0.66", "Final answer: 0.6612"),
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
    ],
)
def test_numeric_mismatches(gold: str, prediction: str) -> None:
    assert not numeric_correct(gold, prediction)


def test_final_answer_falls_back_to_whole_text() -> None:
    assert extract_final_answer("just 42") == "just 42"
    assert extract_final_answer("a\nFinal answer: 1\nFinal Answer: 2") == "2"


def test_abstention_detection() -> None:
    assert is_abstention("INSUFFICIENT_CONTEXT")
    assert is_abstention("The passages do not contain the 2019 figure.")
    assert not is_abstention("Final answer: $1,577 million")


def test_page_hit_and_citation_precision() -> None:
    gold = [("3M_2018_10K", 59)]

    assert page_hit([("3M_2018_10K", 12), ("3M_2018_10K", 59)], gold)
    assert not page_hit([("3M_2018_10K", 58)], gold)
    assert citation_precision([("3M_2018_10K", 59), ("3M_2018_10K", 1)], gold) == 0.5
    assert citation_precision([], gold) is None
