"""Prospective, paired Qwen3.5-9B single-pass SQL intent prompt vs unmodified baseline.

Qwen-only, no gold SQL in prompts, BIRD DEV500-v2 IDs, pre-registered 32-case
screening subset (not the reserved HOLDOUT300). This is DEV evidence only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from evaluation.bird_loader import load_bird_json
from evaluation.protocol import validate_source_hash
from evaluation.run_bird import _ollama_model_digest, _resolve_database, _official_ex_results
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.direct import _strip_fence, generate_direct_sql
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider


SEED = "qwen9b-onepass-structural-v2dev32-20261116"
N = 32
MODEL = "qwen3.5:9b-q4_K_M"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"

TREATMENT_SYSTEM = """Generate exactly one read-only SQLite SELECT query from the provided schema.
Resolve the user intent and relational structure internally, without outputting analysis:
- Identify the answer entity and the required row granularity. Aggregate once per
  requested entity; avoid multiplying rows through one-to-many joins.
- Use valid schema columns and foreign-key relationships; do not guess join keys.
- Preserve every asked filter, threshold, date boundary, exact value, ranking and
  requested output column. Distinguish entity attributes from event attributes.
- For top/bottom queries, rank the requested aggregate, not arbitrary source rows.
Use only the schema and evidence supplied. Return SQL only, no commentary."""

def select_ids(manifest: dict) -> list[str]:
    ids = [str(v) for v in manifest["dev_case_ids"]]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("Expected immutable DEV500 v2 manifest")
    if not manifest.get("seed", "").endswith("v2"):
        raise ValueError("Expected known-exposure-cleaned V2 split")
    return sorted(ids, key=lambda s: (hashlib.sha256((SEED + ":" + s).encode()).hexdigest(), s))[:N]


def treatment_sql(question: str, context: str, evidence: str | None, provider: OllamaProvider) -> SQLCandidate:
    user = "Question: " + question + "\nSchema context:\n" + context
    if evidence:
        user += "\nExternal evidence:\n" + evidence
    return SQLCandidate(sql=_strip_fence(provider.complete_text(system=TREATMENT_SYSTEM, user=user)), attempt=0)


def run_arm(candidate: SQLCandidate, *, question: str, schema: str, evidence: str | None,
            db: Path, provider: OllamaProvider, gold_execution) -> dict:
    t0 = perf_counter()
    result = run_guarded(database=db, provider=provider, question=question,
                         schema_context=schema, evidence=evidence, initial_candidate=candidate,
                         max_repairs=2, max_rows=50_000)
    predicted = None
    if result.status == "ok":
        predicted = execute_readonly(db, result.candidate.sql, max_rows=50_000)
    return {
        "correct": _official_ex_results(predicted, gold_execution),
        "execution_success": bool(result.status == "ok" and predicted is not None and not predicted.truncated),
        "initial_sql": candidate.sql,
        "final_sql": result.candidate.sql if result.candidate else None,
        "status": result.status,
        "repairs": result.candidate.attempt if result.candidate else None,
        "latency_s": round(perf_counter() - t0, 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--database-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-count", type=int, default=8)
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()
    if a.model != MODEL or a.shard_count != 8 or not 0 <= a.shard_index < a.shard_count:
        raise ValueError("This protocol is pinned to Qwen3.5-9B and eight shards")
    manifest_path = Path(a.manifest)
    manifest = json.loads(manifest_path.read_text())
    validate_source_hash(a.source, manifest["source_sha256"])
    model_digest = _ollama_model_digest("http://127.0.0.1:11434", a.model)
    if model_digest != MODEL_DIGEST:
        raise RuntimeError("Model digest mismatch: " + model_digest)
    all_ids = select_ids(manifest)
    selected = all_ids[a.shard_index::a.shard_count]
    if len(selected) != 4:
        raise RuntimeError("Expected exactly four frozen cases in each shard")
    cases, labels = load_bird_json(Path(a.source))
    cmap = {str(c.case_id): c for c in cases}
    lmap = {str(l.case_id): l for l in labels}
    if not set(selected).issubset(cmap.keys() & lmap.keys()):
        raise RuntimeError("Selected cases missing from pinned BIRD")
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Refuse to overwrite an existing result")
    rows: list[dict] = []
    cache: dict[str, tuple[Path, DatabaseCatalog]] = {}
    for cid in selected:
        c, gold = cmap[cid], lmap[cid]
        if c.database_id not in cache:
            db = _resolve_database(Path(a.database_root), c.database_id)
            cache[c.database_id] = (db, DatabaseCatalog.from_sqlite(db))
        db, catalog = cache[c.database_id]
        schema, _ = select_schema_context(catalog, Path(a.database_root), c.database_id, c.question)
        evidence = c.evidence.strip() or None
        gold_execution = execute_readonly(db, gold.sql, max_rows=50_000)
        if gold_execution.truncated:
            raise RuntimeError("Unscorable reference query: " + cid)
        baseline_model = OllamaProvider(model=a.model, timeout_s=600)
        treatment_model = OllamaProvider(model=a.model, timeout_s=600)
        base_initial = generate_direct_sql(c.question, schema, baseline_model, external_evidence=evidence)
        treated_initial = treatment_sql(c.question, schema, evidence, treatment_model)
        base = run_arm(base_initial, question=c.question, schema=schema, evidence=evidence,
                       db=db, provider=baseline_model, gold_execution=gold_execution)
        treated = run_arm(treated_initial, question=c.question, schema=schema, evidence=evidence,
                          db=db, provider=treatment_model, gold_execution=gold_execution)
        rows.append({"case_id": cid, "database_id": c.database_id, "baseline": base, "treatment": treated})
        print("case=" + cid + " baseline=" + str(int(base["correct"])) +
              " treatment=" + str(int(treated["correct"])), flush=True)
        output.write_text(json.dumps({"status": "incomplete", "cases": rows}, indent=2), encoding="utf-8")
    result = {
        "protocol": "Qwen9B single-pass structural intent vs unmodified direct+guarded",
        "status": "complete", "population": "BIRD DEV500-v2 prospective subset, NOT independent holdout",
        "sample_size": len(rows), "model": a.model, "model_digest": model_digest,
        "seed": SEED, "all32_ids_sha256": hashlib.sha256("\n".join(all_ids).encode()).hexdigest(),
        "source_sha256": manifest["source_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "shard_index": a.shard_index, "shard_count": a.shard_count,
        "cases": rows,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
