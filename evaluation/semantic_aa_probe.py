"""Gold-blind controlled Qwen9B semantic evidence pilot.

Pre-register 20 NEW BIRD DEV question identities: ten with a physically verified
quoted value and ten with no quoted values (inert-treatment controls).
Three independent generations per case: identical baseline A/A and evidence.
Explicit Ollama temperature=0, seed=42. Freeze ALL outputs before using gold.
This is a stability/causality diagnostic, NOT a production improvement claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

from evaluation.bird_loader import load_bird_json
from evaluation.schema_context import select_schema_context
from evaluation.literal_grounding_paired import load_excluded
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.literal_evidence import (
    append_literal_evidence, exact_literal_evidence, quoted_literals,
)
from semantic_sql.providers import OllamaProvider

SEED = "qwen9b-untouched-semantic-aa-20261113-v1"
MODEL = "qwen3.5:9b-q4_K_M"
TEMP = 0
OLLAMA_SEED = 42


@dataclass
class PinnedOllamaProvider(OllamaProvider):
    """Local experiment-only provider; no change to the default API/provider."""

    def _chat(self, *, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"temperature": TEMP, "seed": OLLAMA_SEED},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            document = json.loads(response.read().decode("utf-8"))
        return str(document["message"]["content"])


def excluded_440(paths: list[Path]) -> set[str]:
    if len(paths) != 5:
        raise ValueError("must exclude original frozen100, three independent100, and recovered literal40")
    previous = load_excluded(paths[:4])
    literal = json.loads(paths[4].read_text(encoding="utf-8"))
    if literal.get("sample_size") != 40 or not isinstance(literal.get("cases"), list):
        raise ValueError("last source must be the completed, recovered literal40")
    identifiers = [str(x["case_id"]) for x in literal["cases"]]
    if len(identifiers) != 40 or len(set(identifiers)) != 40:
        raise ValueError("literal40 identities are incomplete or duplicated")
    if set(identifiers) & previous:
        raise ValueError("previous BIRD identity cohorts overlap")
    return previous | set(identifiers)


def select_fresh20(cases, *, excluded: set[str], root: Path) -> dict:
    if len(excluded) != 440:
        raise ValueError("require exact frozen 440 question exclusion")
    if len({str(c.case_id) for c in cases}) != len(cases):
        raise ValueError("duplicate BIRD case identities")
    eligible = [c for c in cases if str(c.case_id) not in excluded]
    eligible.sort(key=lambda c: (
        sha256(f"{SEED}:{c.case_id}".encode()).hexdigest(), str(c.case_id)
    ))
    chosen: dict[str, list[dict]] = {"value_hit": [], "no_quoted_literal": []}
    used: dict[str, Counter] = {key: Counter() for key in chosen}
    cache: dict[str, DatabaseCatalog] = {}
    for c in eligible:
        value_group = bool(quoted_literals(c.question))
        group = "value_hit" if value_group else "no_quoted_literal"
        if len(chosen[group]) >= 10 or used[group][c.database_id] >= 3:
            continue
        if value_group:
            if c.database_id not in cache:
                from evaluation.run_bird import _resolve_database
                db = _resolve_database(root, c.database_id)
                cache[c.database_id] = DatabaseCatalog.from_sqlite(db)
            if not exact_literal_evidence(cache[c.database_id], c.question):
                continue
        chosen[group].append({
            "case_id": str(c.case_id),
            "database_id": str(c.database_id),
            "group": group,
        })
        used[group][c.database_id] += 1
        if all(len(v) == 10 for v in chosen.values()):
            break
    if any(len(v) != 10 for v in chosen.values()):
        raise ValueError(
            f"insufficient new gold-blind source-verified cases: "
            f"{ {k: len(v) for k,v in chosen.items()} }"
        )
    if any(len(used[group]) < 4 for group in chosen):
        raise ValueError("too few independent database domains in either stratum")
    result = [value for group in ("value_hit", "no_quoted_literal")
              for value in chosen[group]]
    return {
        "schema": "qwen9b-semantic-aa-manifest/v1",
        "selected": result,
        "excluded_count": len(excluded),
        "selection_seed": SEED,
        "labels_used": False,
        "source_domains": {g: len(used[g]) for g in chosen},
        "group_counts": {g: len(chosen[g]) for g in chosen},
    }


def _score(db: Path, sql: str, gold: str) -> dict[str, bool]:
    try:
        predicted = execute_readonly(db, sql, max_rows=50_000)
        expected = execute_readonly(db, gold, max_rows=50_000)
        if predicted.truncated or expected.truncated:
            return {"set": False, "bag": False}
    except (SQLExecutionError, ValueError):
        return {"set": False, "bag": False}
    p = [tuple(r) for r in predicted.rows]
    g = [tuple(r) for r in expected.rows]
    return {"set": set(p) == set(g), "bag": Counter(p) == Counter(g)}


def run_shard(*, source: Path, root: Path, manifest: Path,
              shard: int, model: str = MODEL) -> dict:
    if shard not in range(4):
        raise ValueError("must use one of four frozen CPU shards")
    cohort = json.loads(manifest.read_text(encoding="utf-8"))
    selected = cohort.get("selected", [])
    if (cohort.get("schema") != "qwen9b-semantic-aa-manifest/v1"
        or cohort.get("excluded_count") != 440
        or cohort.get("labels_used") is not False
        or len(selected) != 20
        or len({str(c["case_id"]) for c in selected}) != 20):
        raise ValueError("invalid preregistered independent cohort")
    questions, labels = load_bird_json(source)
    by_id = {str(x.case_id): x for x in questions}
    provider = PinnedOllamaProvider(model=model, timeout_s=900)
    cache: dict[str, tuple[Path, DatabaseCatalog]] = {}
    frozen = []
    for n, row in enumerate(selected[shard::4], 1):
        case = by_id[str(row["case_id"])]
        if str(case.database_id) != str(row["database_id"]):
            raise ValueError("frozen database identity drift")
        if case.database_id not in cache:
            from evaluation.run_bird import _resolve_database
            db = _resolve_database(root, case.database_id)
            cache[case.database_id] = (db, DatabaseCatalog.from_sqlite(db))
        db, catalog = cache[case.database_id]
        schema, _ = select_schema_context(catalog, root, case.database_id, case.question)
        evidence_schema = append_literal_evidence(
            catalog, case.question, schema_context=schema
        )
        should_change = row["group"] == "value_hit"
        if (evidence_schema != schema) is not should_change:
            raise ValueError("preregistered treatment is not physically source-grounded")
        configs = {"baseline_a": schema, "baseline_b": schema,
                   "grounded": evidence_schema}
        # Counterbalance grounded generation relative to the second identical
        # baseline while keeping the baseline_a reference predeclared.
        if int(sha256(str(case.case_id).encode()).hexdigest(), 16) % 2 == 0:
            order = ("baseline_a", "grounded", "baseline_b")
        else:
            order = ("baseline_a", "baseline_b", "grounded")
        predictions = {}
        for label in order:
            reply = run_guarded(
                database=db, provider=provider, question=case.question,
                schema_context=configs[label],
                evidence=case.evidence.strip() or None,
                max_repairs=2, max_rows=50_000, semantic_revision=False,
            )
            predictions[label] = {
                "sql": reply.candidate.sql if reply.candidate else "",
                "status": reply.status,
                "repairs": reply.candidate.attempt if reply.candidate else 0,
            }
        frozen.append({
            "case_id": str(case.case_id), "database_id": str(case.database_id),
            "group": row["group"], "order": list(order),
            "treatment_context_changed": evidence_schema != schema,
            "predictions": predictions,
        })
        print(f"SEMANTIC_AA_SHARD {shard} {n}/5 case={case.case_id}", flush=True)
    # Only AFTER all 15 model generations for this shard: access reference SQL.
    gold = {str(c.case_id): c.sql for c in labels}
    for row in frozen:
        db = cache[row["database_id"]][0]
        row["ex"] = {
            label: _score(db, pred["sql"], gold[row["case_id"]])
            for label, pred in row["predictions"].items()
        }
    return {
        "schema": "qwen9b-semantic-aa-shard/v1", "shard": shard,
        "model": model, "temperature": TEMP, "seed": OLLAMA_SEED,
        "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "gold_used_during_generation": False, "cases": frozen,
    }


def _metrics(rows: list[dict], metric: str) -> dict:
    a = lambda x: x["ex"]["baseline_a"][metric]
    b = lambda x: x["ex"]["baseline_b"][metric]
    g = lambda x: x["ex"]["grounded"][metric]
    stable = [r for r in rows if a(r) == b(r)]
    return {
        "baseline_a_correct": sum(a(r) for r in rows),
        "baseline_b_correct": sum(b(r) for r in rows),
        "grounded_correct": sum(g(r) for r in rows),
        "aa_correctness_disagreement": sum(a(r) != b(r) for r in rows),
        "aa_sql_disagreement": sum(
            r["predictions"]["baseline_a"]["sql"] != r["predictions"]["baseline_b"]["sql"]
            for r in rows
        ),
        "grounded_vs_a_wrong_to_correct": sum(not a(r) and g(r) for r in rows),
        "grounded_vs_a_correct_to_wrong": sum(a(r) and not g(r) for r in rows),
        "grounded_vs_b_wrong_to_correct": sum(not b(r) and g(r) for r in rows),
        "grounded_vs_b_correct_to_wrong": sum(b(r) and not g(r) for r in rows),
        "stable_baseline_cases": len(stable),
        "stable_wrong_rescued": sum(not a(r) and g(r) for r in stable),
        "stable_correct_regressed": sum(a(r) and not g(r) for r in stable),
    }


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != 4 or sorted(s.get("shard") for s in shards) != list(range(4)):
        raise ValueError("four distinct shards required")
    if any(s.get("schema") != "qwen9b-semantic-aa-shard/v1"
           or s.get("gold_used_during_generation") is not False
           or s.get("temperature") != TEMP or s.get("seed") != OLLAMA_SEED
           for s in shards):
        raise ValueError("invalid or unpinned runtime protocol")
    if len({s["manifest_sha256"] for s in shards}) != 1 or len({s["model"] for s in shards}) != 1:
        raise ValueError("manifest/model mismatch")
    rows = [r for s in shards for r in s["cases"]]
    if len(rows) != 20 or len({r["case_id"] for r in rows}) != 20:
        raise ValueError("incomplete/duplicate predeclared paired cohort")
    groups = {"value_hit": [r for r in rows if r["group"] == "value_hit"],
              "no_quoted_literal": [r for r in rows if r["group"] == "no_quoted_literal"]}
    if any(len(x) != 10 for x in groups.values()):
        raise ValueError("stratum counts changed")
    for r in rows:
        if (r["treatment_context_changed"] is not (r["group"] == "value_hit")
            or set(r["predictions"]) != {"baseline_a","baseline_b","grounded"}
            or set(r["ex"]) != {"baseline_a","baseline_b","grounded"}):
            raise ValueError("unmatched treatment/response")
    result = {
        "schema": "qwen9b-semantic-aa-aggregate/v1",
        "count": len(rows), "model": shards[0]["model"],
        "temperature": TEMP, "seed": OLLAMA_SEED,
        "manifest_sha256": shards[0]["manifest_sha256"],
        "set": {"overall": _metrics(rows, "set"),
                **{key: _metrics(group, "set") for key, group in groups.items()}},
        "bag": {"overall": _metrics(rows, "bag"),
                **{key: _metrics(group, "bag") for key, group in groups.items()}},
        "promote": False,
        "claim_boundary": "20 BIRD DEV questions after excluding 440 prior identities; "
        "database overlap and model pretraining still possible. Fixed seed does "
        "not guarantee deterministic kernels. A/A instability prevents confident "
        "attribution of small treatment differences. No production change.",
        "cases": sorted(rows, key=lambda x: int(x["case_id"])),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--database-root", type=Path)
    p.add_argument("--exclude", nargs="+", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("never overwrite a frozen semantic measurement")
    if a.inputs:
        result = aggregate([json.loads(path.read_text()) for path in a.inputs])
    elif a.exclude:
        if a.source is None or a.database_root is None:
            p.error("preregister requires source and database-root")
        cases, _ = load_bird_json(a.source)
        result = select_fresh20(
            cases, excluded=excluded_440(a.exclude), root=a.database_root
        )
    else:
        if any(v is None for v in (a.source, a.database_root, a.manifest, a.shard)):
            p.error("model shards require source, database-root, manifest and shard")
        result = run_shard(
            source=a.source, root=a.database_root, manifest=a.manifest,
            shard=a.shard, model=a.model,
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("SEMANTIC_AA_RESULT", json.dumps(
        {k:v for k,v in result.items() if k not in ("cases","selected")},
        sort_keys=True,
    ), flush=True)


if __name__ == "__main__":
    main()
