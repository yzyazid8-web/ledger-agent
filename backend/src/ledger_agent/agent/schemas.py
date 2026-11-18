"""Sorties JSON attendues des LLM de l'agent, validées par pydantic."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Period = Literal["FY", "Q1", "Q2", "Q3", "Q4"]


class _Lenient(BaseModel):
    # Les LLM ajoutent parfois des champs : on les ignore plutôt que d'échouer
    model_config = ConfigDict(extra="ignore")


class MetricRequest(_Lenient):
    label: str = Field(min_length=2, max_length=80)
    fiscal_year: int | None = Field(default=None, ge=1990, le=2100)
    fiscal_period: Period = "FY"


class Plan(_Lenient):
    ticker: str | None = None
    fiscal_year: int | None = Field(default=None, ge=1990, le=2100)
    answer_type: Literal["numeric", "text"] = "text"
    metrics: list[MetricRequest] = Field(default_factory=list, max_length=6)
    search_queries: list[str] = Field(default_factory=list, max_length=3)


class Grade(_Lenient):
    sufficient: bool
    missing: str = ""
    new_queries: list[str] = Field(default_factory=list, max_length=2)


class CalcInput(_Lenient):
    evidence: str = Field(pattern=r"^E\d+$")
    value: float


class Calculation(_Lenient):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,30}$")
    expression: str = Field(max_length=300)
    inputs: dict[str, CalcInput] = Field(default_factory=dict)


class Synthesis(_Lenient):
    sufficient: bool = True
    calculations: list[Calculation] = Field(default_factory=list, max_length=6)
    answer: str = ""
    final_answer: str = ""
    cited: list[str] = Field(default_factory=list)
