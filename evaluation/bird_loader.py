from __future__ import annotations

import json
from pathlib import Path


class InferenceCase:
    def __init__(self, case_id: str, database_id: str, question: str, evidence: str = "") -> None:
        self.case_id = case_id
        self.database_id = database_id
        self.question = question
        self.evidence = evidence


class EvaluationLabel:
    def __init__(self, case_id: str, sql: str) -> None:
        self.case_id = case_id
        self.sql = sql


def load_bird_json(path: str | Path) -> tuple[list[InferenceCase], list[EvaluationLabel]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[InferenceCase] = []
    labels: list[EvaluationLabel] = []
    for index, row in enumerate(rows):
        case_id = str(row.get("question_id") or row.get("id") or index)
        cases.append(
            InferenceCase(
                case_id,
                str(row["db_id"]),
                str(row["question"]),
                str(row.get("evidence") or ""),
            )
        )
        labels.append(EvaluationLabel(case_id, str(row["SQL"])))
    return cases, labels
