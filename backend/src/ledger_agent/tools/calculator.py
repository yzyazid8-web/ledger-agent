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
# Au-delà, un nombre n'a plus de sens financier et risque le dépassement de capacité
MAX_ABS_VALUE = 1e18

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
# Nombre d'arguments accepté par fonction : (minimum, maximum)
_ARITY: dict[str, tuple[int, int]] = {
    "abs": (1, 1),
    "round": (1, 2),
    "min": (2, 20),
    "max": (2, 20),
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


def _checked(value: float) -> float:
    if isinstance(value, complex) or not math.isfinite(value) or abs(value) > MAX_ABS_VALUE:
        raise CalculationError(f"Résultat hors limites : {value}")
    return float(value)


def _call(name: str, values: list[float]) -> float:
    low, high = _ARITY[name]
    if not low <= len(values) <= high:
        raise CalculationError(f"{name} attend entre {low} et {high} arguments")
    if name == "abs":
        return abs(values[0])
    if name == "round":
        digits = int(values[1]) if len(values) > 1 else 0
        if not 0 <= digits <= 10:
            raise CalculationError("round accepte entre 0 et 10 décimales")
        return float(round(values[0], digits))
    return float(min(values) if name == "min" else max(values))


def _evaluate(node: ast.AST, variables: Mapping[str, float]) -> float:
    match node:
        case ast.Expression(body=body):
            return _evaluate(body, variables)
        case ast.Constant(value=value) if isinstance(value, int | float) and not isinstance(
            value, bool
        ):
            return _checked(float(value))
        case ast.Name(id=name):
            if name not in variables:
                raise CalculationError(f"Variable inconnue : {name}")
            return _checked(float(variables[name]))
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY:
            return _UNARY[type(op)](_evaluate(operand, variables))
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY:
            a, b = _evaluate(left, variables), _evaluate(right, variables)
            if isinstance(op, ast.Pow) and abs(b) > MAX_EXPONENT:
                raise CalculationError(f"Exposant trop grand : {b}")
            # 1 / 0 et 0 ** -1 sont deux divisions par zéro
            if (isinstance(op, ast.Div) and b == 0) or (
                isinstance(op, ast.Pow) and a == 0 and b < 0
            ):
                raise CalculationError("Division par zéro")
            return _checked(_BINARY[type(op)](a, b))
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _ARITY:
            return _checked(_call(name, [_evaluate(arg, variables) for arg in args]))
    raise CalculationError(f"Syntaxe non autorisée : {type(node).__name__}")


def calculate(expression: str, variables: Mapping[str, float] | None = None) -> CalculationResult:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculationError("Expression trop longue")
    variables = dict(variables or {})
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _evaluate(tree, variables)
    except CalculationError:
        raise
    except SyntaxError as exc:
        raise CalculationError(f"Expression invalide : {exc.msg}") from exc
    except (ArithmeticError, TypeError, ValueError, IndexError) as exc:
        # Filet de sécurité : toute autre erreur respecte le contrat CalculationError
        raise CalculationError(f"Calcul impossible : {type(exc).__name__}") from exc
    return CalculationResult(expression=expression.strip(), variables=variables, value=value)
