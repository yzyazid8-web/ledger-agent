"""Questions FinanceBench et découpage figé en jeux dev et test."""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Split = Literal["dev", "test", "all"]

DEFAULT_DEV_SIZE = 50
DEFAULT_SEED = 13


@dataclass(frozen=True, slots=True)
class EvidencePage:
    doc_name: str
    page_index: int


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    answer: str
    company: str
    doc_name: str
    question_type: str
    evidence: tuple[EvidencePage, ...]


def to_question(row: dict[str, Any]) -> Question:
    return Question(
        id=row["financebench_id"],
        question=row["question"],
        answer=row["answer"],
        company=row["company"],
        doc_name=row["doc_name"],
        question_type=row["question_type"],
        evidence=tuple(
            EvidencePage(e["doc_name"], int(e["evidence_page_num"])) for e in row["evidence"]
        ),
    )


def make_split(
    questions: list[Question], dev_size: int = DEFAULT_DEV_SIZE, seed: int = DEFAULT_SEED
) -> dict[str, list[str]]:
    """Tirage stratifié par type de question, reproductible grâce à la graine."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q.id)

    rng = random.Random(seed)
    dev: list[str] = []
    types = sorted(by_type)
    for i, question_type in enumerate(types):
        ids = sorted(by_type[question_type])
        rng.shuffle(ids)
        # Répartition proportionnelle, le reste allant aux premiers types
        quota = dev_size // len(types) + (1 if i < dev_size % len(types) else 0)
        dev.extend(ids[:quota])

    dev_set = set(dev)
    test = [q.id for q in questions if q.id not in dev_set]
    return {"dev": sorted(dev), "test": sorted(test)}


def load_or_create_split(path: Path, questions: list[Question]) -> dict[str, list[str]]:
    # Le découpage est écrit une fois puis versionné : il ne doit jamais changer en douce
    if path.exists():
        split: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
        return split
    split = make_split(questions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return split


def select_split(
    questions: list[Question], split: Split, splits: dict[str, list[str]]
) -> list[Question]:
    if split == "all":
        return questions
    wanted = set(splits[split])
    return [q for q in questions if q.id in wanted]
