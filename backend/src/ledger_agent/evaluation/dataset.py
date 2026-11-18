"""Questions FinanceBench et découpage figé en jeux dev et test, groupé par document."""

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Split = Literal["dev", "test"]

SPLIT_STRATEGY = "doc_name"
DEFAULT_DEV_SIZE = 50
DEFAULT_SEED = 13
# Écart toléré par type de question par rapport à la répartition proportionnelle
TYPE_TOLERANCE = 2


class SplitError(ValueError):
    """Fichier de découpage incohérent avec les questions."""


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
) -> dict[str, Any]:
    """Un document est entièrement en dev ou entièrement en test, types de questions équilibrés."""
    by_doc: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_doc[q.doc_name].append(q)
    type_totals = Counter(q.question_type for q in questions)
    quotas = {t: n * dev_size / len(questions) for t, n in type_totals.items()}

    docs = sorted(by_doc)
    random.Random(seed).shuffle(docs)
    dev_docs: list[str] = []
    dev_types: Counter[str] = Counter()
    for doc in docs:
        doc_types = Counter(q.question_type for q in by_doc[doc])
        size = sum(dev_types.values())
        fits_size = size + len(by_doc[doc]) <= dev_size
        fits_types = all(
            dev_types[t] + n <= quotas[t] + TYPE_TOLERANCE for t, n in doc_types.items()
        )
        if fits_size and fits_types:
            dev_docs.append(doc)
            dev_types += doc_types
        if sum(dev_types.values()) >= dev_size:
            break

    dev_set = set(dev_docs)
    return {
        "strategy": SPLIT_STRATEGY,
        "seed": seed,
        "dev_documents": sorted(dev_set),
        "dev": sorted(q.id for q in questions if q.doc_name in dev_set),
        "test": sorted(q.id for q in questions if q.doc_name not in dev_set),
    }


def validate_split(split: dict[str, Any], questions: list[Question]) -> None:
    if split.get("strategy") != SPLIT_STRATEGY:
        raise SplitError("Découpage ancien ou inconnu : supprime splits.json pour le régénérer")
    dev, test = set(split["dev"]), set(split["test"])
    known = {q.id for q in questions}
    if dev & test:
        raise SplitError("Des questions sont à la fois en dev et en test")
    if (dev | test) != known:
        raise SplitError("splits.json ne couvre pas exactement les questions FinanceBench")
    doc_of = {q.id: q.doc_name for q in questions}
    if {doc_of[i] for i in dev} & {doc_of[i] for i in test}:
        raise SplitError("Un même document apparaît en dev et en test")


def load_or_create_split(path: Path, questions: list[Question]) -> dict[str, Any]:
    # Le découpage est écrit une fois puis versionné : il ne doit jamais changer en douce
    if path.exists():
        split: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    else:
        split = make_split(questions)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    validate_split(split, questions)
    return split


def select_split(questions: list[Question], split: Split, splits: dict[str, Any]) -> list[Question]:
    wanted = set(splits[split])
    selected = [q for q in questions if q.id in wanted]
    if not selected:
        raise SplitError(f"Aucune question dans le jeu {split}")
    return selected
