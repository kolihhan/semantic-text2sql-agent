from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.bird_loader import load_bird_json
from evaluation.run_bird import _official_ex, _resolve_database
from semantic_sql.contracts import SQLCandidate
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider
from semantic_sql.zero_row import build_zero_row_evidence


def _load_frozen_cases(artifact_dir: Path) -> list[dict]:
    paths = sorted(glob.glob(str(artifact_dir / "**" / "*.json"), recursive=True))
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    payloads = [payload for payload in payloads if isinstance(payload, dict) and isinstance(payload.get("cases"), list)]
    cases = [case for payload in payloads for case in payload["cases"]]
    if len(cases) != 100 or len({str(case["case_id"]) for case in cases}) != 100:
        raise ValueError("expected exactly 100 unique frozen 9B cases")
    baseline_correct = sum(bool(case["guarded"]["official_ex"]) for case in cases)
    if baseline_correct != 39:
        raise ValueError(f"expected frozen guarded baseline 39/100, got {baseline_correct}/100")
    return cases


def _is_zero_row_case(case: dict) -> bool:
    guarded = case["guarded"]
    if guarded.get("status") != "ok" or guarded.get("official_ex") is True:
        return False
    return any(
        isinstance(stage, dict)
        and stage.get("name") == "execute"
        and stage.get("summary") == "rows=0"
        for stage in guarded.get("stages", [])
    )


def run_experiment(
    *, source: Path, database_root: Path, artifact_dir: Path, model: str,
    base_url: str, timeout_s: float, output: Path,
) -> dict:
    frozen_cases = _load_frozen_cases(artifact_dir)
    targeted = [case for case in frozen_cases if _is_zero_row_case(case)]
    if len(targeted) != 12:
        raise ValueError(f"expected 12 frozen zero-row cases, got {len(targeted)}")

    cases, labels = load_bird_json(source)
    cases_by_id = {case.case_id: case for case in cases}
    labels_by_id = {label.case_id: label for label in labels}
    provider = OllamaProvider(model=model, base_url=base_url, timeout_s=timeout_s)
    results = []

    for frozen in targeted:
        case_id = str(frozen["case_id"])
        case = cases_by_id[case_id]
        label = labels_by_id[case_id]
        if case.database_id != str(frozen["database_id"]) or case.question != str(frozen["question"]):
            raise ValueError(f"source identity mismatch for {case_id}")
        database = _resolve_database(database_root, case.database_id)
        baseline_sql = str(frozen["guarded"]["final_sql"])
        zero_row_evidence = build_zero_row_evidence(
            database, question=case.question, sql=baseline_sql,
        )
        if not zero_row_evidence:
            results.append({
                "case_id": case_id,
                "database_id": case.database_id,
                "question": case.question,
                "baseline_sql": baseline_sql,
                "baseline_ex": False,
                "repair_attempted": False,
                "treatment_sql": baseline_sql,
                "treatment_ex": False,
                "evidence": "",
                "status": "no_bounded_evidence",
            })
            continue

        result = run_guarded(
            database=database,
            provider=provider,
            question=case.question,
            schema_context=str(frozen["schema_context"]),
            evidence=case.evidence.strip() or None,
            initial_candidate=SQLCandidate(sql=baseline_sql, attempt=0),
            max_repairs=0,
            max_rows=50_000,
            zero_row_repair=True,
            zero_row_evidence=zero_row_evidence,
        )
        treatment_sql = result.candidate.sql if result.candidate is not None else None
        treatment_ex = _official_ex(database, treatment_sql, label.sql, max_rows=50_000)
        results.append({
            "case_id": case_id,
            "database_id": case.database_id,
            "question": case.question,
            "baseline_sql": baseline_sql,
            "baseline_ex": False,
            "repair_attempted": True,
            "treatment_sql": treatment_sql,
            "treatment_ex": treatment_ex,
            "evidence": zero_row_evidence,
            "status": result.status,
            "stages": [{"name": stage.name, "summary": stage.summary} for stage in result.stages],
        })

    rescued = [row["case_id"] for row in results if row["treatment_ex"]]
    attempted = sum(bool(row["repair_attempted"]) for row in results)
    payload = {
        "experiment": "qwen9b_zero_row_conditional_repair_v1",
        "scope": "posthoc diagnostic on 12 frozen guarded zero-row failures; untouched 88 cases remain frozen",
        "model": model,
        "baseline_ex_correct": 39,
        "baseline_ex_rate": 0.39,
        "targeted_zero_row_cases": len(targeted),
        "repair_attempted": attempted,
        "rescued_cases": rescued,
        "rescued_count": len(rescued),
        "correct_to_wrong": 0,
        "treatment_ex_correct_if_spliced": 39 + len(rescued),
        "treatment_ex_rate_if_spliced": (39 + len(rescued)) / 100,
        "promotion_status": "diagnostic_only_same_eval_set",
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_experiment(
        source=args.source,
        database_root=args.database_root,
        artifact_dir=args.artifact_dir,
        model=args.model,
        base_url=args.base_url,
        timeout_s=args.timeout,
        output=args.output,
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
