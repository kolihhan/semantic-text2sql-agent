"""Paired DEV evaluation for the Direct and Guarded runtimes."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from statistics import median
import tempfile
import time
from typing import Callable
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.direct import generate_direct_sql
from semantic_sql.execution import ExecutionResult, SQLExecutionError, execute_readonly
from semantic_sql.inference import GUARDED_TREATMENT_ID, run_guarded
from semantic_sql.providers import ModelProvider, OllamaProvider

try:
    from .bird_loader import load_bird_json
    from .metrics import paired_transition_counts
    from .protocol import selected_case_ids, validate_source_hash
    from .schema_context import build_schema_context, select_schema_context
except ImportError:  # direct script execution
    from bird_loader import load_bird_json
    from metrics import paired_transition_counts
    from protocol import selected_case_ids, validate_source_hash
    from schema_context import build_schema_context, select_schema_context


OFFICIAL_BIRD_EVALUATOR_URL = "https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py"
OFFICIAL_BIRD_EVALUATOR_REVISION = "483554eae102996f5ec1f4feab4e78ef29c2a394"


def _ollama_model_digest(base_url: str, model: str) -> str:
    try:
        with urlopen(base_url.rstrip("/") + "/api/tags", timeout=10) as response:
            payload = json.loads(response.read())
    except Exception as exc:
        raise RuntimeError("Ollama model identity preflight failed") from exc
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Ollama model identity response is invalid")
    row = next((item for item in rows if isinstance(item, dict) and str(item.get("name") or item.get("model")) == model), None)
    if row is None:
        raise RuntimeError(f"Ollama model is missing: {model}")
    digest = str(row.get("digest") or "").casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("Ollama model digest is invalid")
    return digest


class CountingProvider:
    def __init__(self, inner: ModelProvider) -> None:
        self.inner = inner
        self.text_calls = 0

    @property
    def total_calls(self) -> int:
        return self.text_calls

    def complete_text(self, *, system: str, user: str) -> str:
        self.text_calls += 1
        return self.inner.complete_text(system=system, user=user)


def _resolve_database(database_root: Path, db_id: str) -> Path:
    direct = database_root / db_id / f"{db_id}.sqlite"
    if direct.is_file():
        return direct
    candidates = [path for path in (database_root / db_id).glob("*.sqlite") if path.is_file()]
    if len(candidates) == 1:
        return candidates[0]
    candidates = [path for path in database_root.rglob("*.sqlite") if path.stem == db_id]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"could not resolve SQLite database for {db_id!r} under {database_root}")


def _official_ex(database: Path, predicted_sql: str | None, gold_sql: str, *, max_rows: int) -> bool:
    if not predicted_sql:
        return False
    try:
        predicted = execute_readonly(database, predicted_sql, max_rows=max_rows)
        gold = execute_readonly(database, gold_sql, max_rows=max_rows)
    except SQLExecutionError:
        return False
    if predicted.truncated or gold.truncated:
        return False
    return set(predicted.rows) == set(gold.rows)


def _official_ex_results(predicted, gold) -> bool:
    if predicted is None or gold is None or predicted.truncated or gold.truncated:
        return False
    return set(predicted.rows) == set(gold.rows)


def _first_stage(result, name: str) -> str | None:
    for stage in result.stages:
        if stage.name == name:
            return stage.summary
    return None


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[max(0, int(0.95 * len(values) + 0.999999) - 1)]


def _arm_metrics(records: list[dict], arm: str) -> dict[str, int | float]:
    rows = [record[arm] for record in records]
    latencies = [float(row["latency_s"]) for row in rows]
    return {
        "official_bird_ex": (sum(bool(row["official_ex"]) for row in rows) / len(rows)) if rows else 0.0,
        "execution_success": (sum(bool(row["execution_success"]) for row in rows) / len(rows)) if rows else 0.0,
        "model_calls": sum(int(row["model_calls"]) for row in rows),
        "latency_median_s": median(latencies) if latencies else 0.0,
        "latency_p95_s": _p95(latencies),
    }


def _load_resume_payload(path: Path, expected: dict, selected_ids: list[str]) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"resume artifact is unreadable: {path}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "incomplete":
        raise ValueError("resume requires an incomplete artifact")
    identity_keys = (
        "project", "protocol_version", "split", "model", "model_digest", "think",
        "source_sha256", "split_manifest_sha256", "sample_size", "runtime",
        "official_bird_evaluator",
    )
    mismatches = [key for key in identity_keys if payload.get(key) != expected[key]]
    if mismatches:
        raise ValueError(f"resume artifact identity/config mismatch: {', '.join(mismatches)}")
    if payload.get("metrics") is not None:
        raise ValueError("incomplete resume artifact must not contain final metrics")
    completed = payload.get("cases")
    if not isinstance(completed, list) or any(not isinstance(case, dict) for case in completed):
        raise ValueError("resume artifact cases are invalid")
    completed_ids = [case.get("case_id") for case in completed]
    if completed_ids != selected_ids[:len(completed_ids)]:
        raise ValueError("completed case IDs are not the exact manifest prefix")
    return payload


def run_paired_dev(
    *,
    source_path: Path,
    database_root: Path,
    manifest_path: Path,
    output_path: Path,
    model: str = "qwen3.5:4b",
    provider_factory: Callable[[], ModelProvider] | None = None,
    base_url: str = "http://localhost:11434",
    timeout_s: float = 180.0,
    max_repairs: int = 2,
    max_rows: int = 50_000,
    resume: bool = False,
) -> dict:
    if output_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if resume and not output_path.exists():
        raise FileNotFoundError(f"resume output does not exist: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_source_hash(source_path, str(manifest["source_sha256"]))
    selected_ids = selected_case_ids(manifest, "dev")
    cases, labels = load_bird_json(source_path)
    cases_by_id = {case.case_id: case for case in cases}
    labels_by_id = {label.case_id: label for label in labels}
    missing = [case_id for case_id in selected_ids if case_id not in cases_by_id or case_id not in labels_by_id]
    if missing:
        raise RuntimeError(f"manifest case IDs missing from source: {missing[:5]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_digest = _ollama_model_digest(base_url, model) if provider_factory is None else None
    expected_payload: dict = {
        "project": "P1 Benchmark-First Text-to-SQL",
        "protocol_version": 2,
        "status": "incomplete",
        "split": "dev",
        "model": model,
        "model_digest": model_digest,
        "think": False,
        "source_sha256": str(manifest["source_sha256"]),
        "split_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "sample_size": len(selected_ids),
        "runtime": {
            "model": model, "model_digest": model_digest, "base_url": base_url, "timeout_s": timeout_s,
            "think": False, "max_repairs": max_repairs, "max_rows": max_rows,
            "guarded_treatment": GUARDED_TREATMENT_ID,
            "schema_grounding": "deterministic_lexical_v1",
        },
        "official_bird_evaluator": {"url": OFFICIAL_BIRD_EVALUATOR_URL, "revision": OFFICIAL_BIRD_EVALUATOR_REVISION},
        "metrics": None,
        "cases": [],
    }
    if resume:
        payload = _load_resume_payload(output_path, expected_payload, selected_ids)
    else:
        payload = expected_payload
        _write(output_path, payload)

    cache: dict[str, tuple[Path, DatabaseCatalog]] = {}
    invalid = any(case.get("initial_sql_identical") is not True for case in payload["cases"])
    completed_count = len(payload["cases"])
    try:
        for index, case_id in enumerate(selected_ids[completed_count:], start=completed_count + 1):
            case = cases_by_id[case_id]
            label = labels_by_id[case_id]
            if case.database_id not in cache:
                database = _resolve_database(database_root, case.database_id)
                catalog = DatabaseCatalog.from_sqlite(database)
                cache[case.database_id] = (database, catalog)
            database, catalog = cache[case.database_id]
            schema_context, schema_diagnostics = select_schema_context(catalog, database_root, case.database_id, case.question)
            provider = CountingProvider(provider_factory() if provider_factory else OllamaProvider(model=model, base_url=base_url, timeout_s=timeout_s))
            evidence = case.evidence.strip() or None

            shared_started = time.perf_counter()
            shared_candidate = generate_direct_sql(case.question, schema_context, provider, external_evidence=evidence)
            shared_generation_latency = time.perf_counter() - shared_started
            shared_calls = provider.total_calls

            direct_started = time.perf_counter()
            direct_error = None
            try:
                direct_execution = execute_readonly(database, shared_candidate.sql, max_rows=max_rows)
                direct_status = "ok"
                direct_success = True
            except SQLExecutionError as exc:
                direct_execution = None
                direct_status = "execution_failed"
                direct_success = False
                direct_error = str(exc)
            direct_latency = shared_generation_latency + (time.perf_counter() - direct_started)

            guarded_started = time.perf_counter()
            guarded = run_guarded(
                database=database,
                provider=provider,
                question=case.question,
                schema_context=schema_context,
                evidence=evidence,
                initial_candidate=shared_candidate,
                max_repairs=max_repairs,
                max_rows=max_rows,
            )
            guarded_latency = shared_generation_latency + (time.perf_counter() - guarded_started)
            guarded_initial_sql = _first_stage(guarded, "sql")
            initial_identical = guarded_initial_sql == shared_candidate.sql
            if not initial_identical:
                invalid = True

            try:
                gold_execution = execute_readonly(database, label.sql, max_rows=max_rows)
            except SQLExecutionError:
                gold_execution = None
                invalid = True
            direct_ex = _official_ex_results(direct_execution, gold_execution)
            guarded_execution = None
            if guarded.status == "ok":
                guarded_execution = ExecutionResult(columns=guarded.columns, rows=guarded.rows, truncated=guarded.truncated)
            guarded_ex = _official_ex_results(guarded_execution, gold_execution)
            if (
                (direct_execution is not None and direct_execution.truncated)
                or (guarded_execution is not None and guarded_execution.truncated)
                or (gold_execution is not None and gold_execution.truncated)
            ):
                invalid = True
            direct_record = {
                "initial_sql": shared_candidate.sql,
                "final_sql": shared_candidate.sql,
                "status": direct_status,
                "execution_success": direct_success,
                "official_ex": direct_ex,
                "model_calls": shared_calls,
                "latency_s": direct_latency,
                "repair_attempts": 0,
                "error": direct_error,
            }
            guarded_record = {
                "initial_sql": guarded_initial_sql,
                "final_sql": guarded.candidate.sql if guarded.candidate else None,
                "status": guarded.status,
                "execution_success": guarded.status == "ok",
                "official_ex": guarded_ex,
                "model_calls": provider.total_calls,
                "latency_s": guarded_latency,
                "repair_attempts": guarded.candidate.attempt if guarded.candidate else 0,
                "error": guarded.message or None,
                "stages": [{"name": stage.name, "summary": stage.summary} for stage in guarded.stages],
            }
            payload["cases"].append({
                "case_id": case.case_id,
                "database_id": case.database_id,
                "question": case.question,
                "evidence_chars": len(case.evidence),
                "shared_initial_sql": shared_candidate.sql,
                "shared_generation_latency_s": shared_generation_latency,
                "schema_context": schema_diagnostics,
                "initial_sql_identical": initial_identical,
                "direct": direct_record,
                "guarded": guarded_record,
            })
            _write(output_path, payload)
            print(f"[paired-dev {index:03d}/{len(selected_ids)}] {case.case_id} direct={direct_status} guarded={guarded.status}", flush=True)
    except Exception:
        payload["status"] = "incomplete"
        _write(output_path, payload)
        raise

    payload["status"] = "invalid" if invalid else "complete"
    payload["metrics"] = {
        "direct": _arm_metrics(payload["cases"], "direct"),
        "guarded": _arm_metrics(payload["cases"], "guarded"),
        "transitions": paired_transition_counts(payload["cases"]),
    }
    _write(output_path, payload)
    return payload


def _cmd_paired_dev(args: argparse.Namespace) -> int:
    payload = run_paired_dev(
        source_path=Path(args.source), database_root=Path(args.database_root),
        manifest_path=Path(args.manifest), output_path=Path(args.output),
        model=args.model, base_url=args.base_url, timeout_s=args.timeout,
        max_repairs=2, max_rows=args.max_rows, resume=args.resume,
    )
    print(json.dumps({"status": payload["status"], "metrics": payload["metrics"]}, indent=2))
    return 0 if payload["status"] == "complete" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired BIRD DEV evaluation for Direct and Guarded")
    sub = parser.add_subparsers(dest="command", required=True)
    paired = sub.add_parser("paired-dev", help="run the one paired Direct/Guarded DEV protocol")
    paired.add_argument("--source", required=True)
    paired.add_argument("--database-root", required=True)
    paired.add_argument("--manifest", required=True)
    paired.add_argument("--model", default="qwen3.5:4b")
    paired.add_argument("--base-url", default="http://localhost:11434")
    paired.add_argument("--timeout", type=float, default=180.0)
    paired.add_argument("--max-rows", type=int, default=50_000)
    paired.add_argument("--output", required=True)
    paired.add_argument("--resume", action="store_true")
    paired.set_defaults(func=_cmd_paired_dev)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
