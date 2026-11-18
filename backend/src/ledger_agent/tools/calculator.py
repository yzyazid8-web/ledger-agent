"""Calculatrice sûre : l'expression est analysée en arbre syntaxique, jamais exécutée."""

import ast
import hashlib
import json
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

MAX_EXPRESSION_LENGTH = 500
MAX_EXPONENT = 10

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


class CalculationError(ValueError):
    """Expression refusée ou impossible à calculer."""


@dataclass(frozen=True, slots=True)
class CalculationResult:
    expression: str
    variables: dict[str, float] = field(default_factory=dict)
    value: float = 0.0

    @property
    def source_id(self) -> str:
        # Identifiant stable : la même formule avec les mêmes entrées donne le même id
        raw = json.dumps([self.expression, self.variables], sort_keys=True)
        return "calc:" + hashlib.sha256(raw.encode()).hexdigest()[:10]


def _evaluate(node: ast.AST, variables: Mapping[str, float]) -> float:
    match node:
        case ast.Expression(body=body):
            return _evaluate(body, variables)
        case ast.Constant(value=value) if isinstance(value, int | float) and not isinstance(
            value, bool
        ):
            return float(value)
        case ast.Name(id=name):
            if name not in variables:
                raise CalculationError(f"Unknown variable: {name}")
            return float(variables[name])
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY:
            return _UNARY[type(op)](_evaluate(operand, variables))
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY:
            a, b = _evaluate(left, variables), _evaluate(right, variables)
            if isinstance(op, ast.Pow) and abs(b) > MAX_EXPONENT:
                raise CalculationError(f"Exponent too large: {b}")
            if isinstance(op, ast.Div) and b == 0:
                raise CalculationError("Division by zero")
            return _BINARY[type(op)](a, b)
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _FUNCTIONS:
            values = [_evaluate(arg, variables) for arg in args]
            if name == "round":
                # round(x) ou round(x, n), avec n entier
                return float(round(values[0], int(values[1]) if len(values) > 1 else 0))
            return float(_FUNCTIONS[name](*values))
    raise CalculationError(f"Unsupported syntax: {type(node).__name__}")


def calculate(expression: str, variables: Mapping[str, float] | None = None) -> CalculationResult:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculationError("Expression too long")
    variables = dict(variables or {})
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"Invalid expression: {exc.msg}") from exc
    value = _evaluate(tree, variables)
    if not math.isfinite(value):
        raise CalculationError("Result is not a finite number")
    return CalculationResult(expression=expression.strip(), variables=variables, value=value)
