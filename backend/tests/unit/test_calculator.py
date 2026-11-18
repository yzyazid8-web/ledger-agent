import pytest

from ledger_agent.tools.calculator import CalculationError, calculate


def test_growth_rate_with_variables() -> None:
    result = calculate("(c18 - c17) / c17 * 100", {"c18": 1577, "c17": 1373})

    assert round(result.value, 2) == 14.86
    assert result.source_id.startswith("calc:")


def test_functions_and_unary_minus() -> None:
    assert calculate("round(-abs(-2.345), 2)").value == -2.35
    assert calculate("max(1, 2, 3) - min(4, 5)").value == -1
    assert calculate("2 ** 3").value == 8


def test_same_inputs_give_same_source_id() -> None:
    first = calculate("a / b", {"a": 1, "b": 2})
    second = calculate(" a / b ", {"b": 2, "a": 1})

    assert first.source_id == second.source_id


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('dir')",
        "(1).__class__",
        "open('x')",
        "'abc' * 2",
        "True + 1",
        "unknown + 1",
        "1 / 0",
        "10 ** 100",
        "1 +",
        "[1, 2]",
        "x if y else z",
    ],
)
def test_rejected_expressions(expression: str) -> None:
    with pytest.raises(CalculationError):
        calculate(expression)


def test_expression_length_is_bounded() -> None:
    with pytest.raises(CalculationError, match="too long"):
        calculate("1+" * 300 + "1")
