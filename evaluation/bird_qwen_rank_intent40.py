"""Question-conditioned ranking/grain contract: frozen Qwen9B Guarded paired study.

Research only. 40 question-disjoint BIRD TRAIN cases from physical databases not
used by BIRD DEV or the two previous Train40 cohorts (if enough exist).
20 ranking-intent cases get one concise, gold-blind guidance; 20 nonranking
cases get an exact A/A rerun to measure generation drift. Same Qwen model,
same guarded verifier, same schema context and no SQL training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from urllib.parse import quote
from urllib.request import urlopen

from evaluation.bird_generator_transfer import (
    MIRROR, DATASET, REVISION, TRAIN_SHA, check_database_bytes,
    load_checked, remote_manifest, select_cases,
)
from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from evaluation.xiyan_qwen_factorial_fresh40 import QWEN_DIGEST, MODEL_IDS, SOURCE_SHA256
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

SHARDS = 10
PER_GROUP = 20
SEED = "qwen9b-ranking-obligations-unseen-train-dbs-20261114"
RANK_CUES = re.compile(
    r"\b(top|bottom|most|least|highest|lowest|largest|smallest|best|worst|"
    r"earliest|latest|newest|oldest|fewest|maximum|minimum|first|last|rank|"
    r"ranking|popular|common|sorted|sort|ordered|order)\b", re.I
)
OBLIGATIONS = (
    "Before generating the single SQL answer, check these question-specific "
    "ranking obligations internally. Identify the exact entity to RETURN "
    "and the quantitative or temporal measure to rank it by. Distinguish "
    "MAX/MIN of one row from highest aggregate per GROUP, and select GROUP BY "
    "at the entity grain requested. Apply relevant filters before aggregation, "
    "avoid multiplying entities with joins, and preserve the question's "
    "requested ASC/DESC direction and number of results (including top N). "
    "For a ranking expression, ORDER BY the actual measure, not a display "
    "label or an unrelated field. Do not invent criteria absent from the "
    "question; return exactly one read-only SQLite SQL query."
)


def choose_cases(train: list[dict], dev: list[dict], db_manifest: dict) -> list[dict]:
    """Select by question text only; deny DB and question overlap with old cohorts."""
    original = select_cases(train, dev, db_manifest, guarded=False)
    guarded = select_cases(train, dev, db_manifest, guarded=True)
    prior_ids = {r["case_id"] for r in original + guarded}
    dbs_seen = ({r["db_id"] for r in dev}
                | {r["database_id"] for r in original + guarded})
    normalized_seen = {" ".join(r["question"].casefold().split()) for r in dev}
    normalized_seen |= {" ".join(r["question"].casefold().split()) for r in original + guarded}
    choices = {"rank": [], "control": []}
    for i, item in enumerate(train):
        db = item["db_id"]
        q = str(item["question"])
        norm = " ".join(q.casefold().split())
        if (i in prior_ids or db in dbs_seen or db not in db_manifest or
                db_manifest[db]["size"] > 128 * 1024**2 or
                not norm or norm in normalized_seen):
            continue
        group = "rank" if RANK_CUES.search(q) else "control"
        choices[group].append({
            "case_id": i, "database_id": db, "group": group,
            "question": q, "evidence": item.get("evidence") or "",
        })
        normalized_seen.add(norm)
    selected = []
    for group in ("rank", "control"):
        pool = sorted(choices[group], key=lambda row: (
            sha256(f"{SEED}|{row['case_id']}".encode()).hexdigest(),
            row["case_id"],
        ))
        domain_count = Counter()
        kept = []
        for item in pool:
            if domain_count[item["database_id"]] >= 5:
                continue
            kept.append(item)
            domain_count[item["database_id"]] += 1
            if len(kept) == PER_GROUP:
                break
        if len(kept) != PER_GROUP or len(domain_count) < 5:
            raise ValueError(f"not enough independent {group} questions across unseen train DBs: {len(kept)}; domains={len(domain_count)}")
        selected.extend(kept)
    if len(selected) != 40 or len({x["case_id"] for x in selected}) != 40:
        raise ValueError("not 40 unique questions")
    return selected


def inference_shard(*, source_train: Path, source_dev: Path,
                    shard: int, work: Path) -> dict:
    if shard not in range(SHARDS):
        raise ValueError("must use one of ten shards")
    train = load_checked(source_train, TRAIN_SHA)
    dev = load_checked(source_dev, SOURCE_SHA256)
    manifest = remote_manifest()
    cohort = choose_cases(train, dev, manifest)
    selected = cohort[4*shard:4*(shard+1)]
    if len(selected) != 4:
        raise ValueError("incomplete 4-case shard")
    model = MODEL_IDS["qwen"]
    digest = _ollama_model_digest("http://127.0.0.1:11434", model)
    if digest != QWEN_DIGEST:
        raise ValueError("pinned Qwen9B model digest changed")
    dbs = {}
    for dbid in sorted({r["database_id"] for r in selected}):
        path = work / dbid / f"{dbid}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{MIRROR}/datasets/{DATASET}/resolve/{REVISION}/raw/bird/train_databases"
        with urlopen(f"{url}/{quote(dbid)}/{quote(dbid)}.sqlite", timeout=120) as resp:
            data = resp.read(manifest[dbid]["size"] + 1)
        if not check_database_bytes(data, manifest[dbid]):
            raise ValueError("BIRD database bytes not matched to frozen source")
        path.write_bytes(data)
        dbs[dbid] = (path, DatabaseCatalog.from_sqlite(path))
    provider = OllamaProvider(
        model=model, timeout_s=900, temperature=0, seed=42, num_predict=1024
    )
    records = []
    for row in selected:
        db, catalog = dbs[row["database_id"]]
        schema, _ = select_schema_context(catalog, work, row["database_id"], row["question"])
        contexts = {
            "original": schema,
            "rank_contract": (schema + "\n\nGENERATION GUIDANCE (not database schema):\n"
                              + OBLIGATIONS if row["group"] == "rank" else schema),
        }
        arms = {}
        start = perf_counter()
        for name, context in contexts.items():
            res = run_guarded(
                database=db, provider=provider, question=row["question"],
                schema_context=context, evidence=row["evidence"] or None,
                max_repairs=2, max_rows=50_000,
            )
            ok = res.status == "ok" and not res.truncated
            arms[name] = {
                "sql": res.candidate.sql if res.candidate else "",
                "rows": res.rows if ok else (),
                "execution_success": ok,
                "repairs": max(0, res.candidate.attempt) if res.candidate else 0,
            }
        records.append({
            "case_id": row["case_id"], "database_id": row["database_id"],
            "group": row["group"], "arms": arms,
            "wall_s": round(perf_counter() - start, 3),
        })
        print("QWEN_RANK_FROZEN", shard, row["case_id"], row["group"], flush=True)
    # Only inspect ground-truth after both arms finish generation.
    for rec in records:
        db, _ = dbs[rec["database_id"]]
        expected = execute_readonly(db, train[rec["case_id"]]["SQL"], max_rows=50_000)
        if expected.truncated:
            raise ValueError("gold truncated")
        for arm in rec["arms"].values():
            rows = arm.pop("rows")
            arm["ex"] = bool(arm["execution_success"] and set(rows) == set(expected.rows))
            arm["bag"] = bool(arm["execution_success"] and Counter(rows) == Counter(expected.rows))
    return {
        "shard": shard, "model_digest": digest,
        "cohort_sha": sha256(json.dumps([
            [x["case_id"], x["database_id"], x["group"]] for x in cohort
        ], separators=(",", ":")).encode()).hexdigest(),
        "rows": records,
    }


def aggregate(parts: list[dict]) -> dict:
    if len(parts) != SHARDS or sorted(x["shard"] for x in parts) != list(range(SHARDS)):
        raise ValueError("ten shards mandatory")
    if len({x["model_digest"] for x in parts}) != 1 or len({x["cohort_sha"] for x in parts}) != 1:
        raise ValueError("model or source changed")
    records = [r for part in parts for r in part["rows"]]
    if len(records) != 40 or len({r["case_id"] for r in records}) != 40:
        raise ValueError("40 unique rows required")
    group_counts = Counter(r["group"] for r in records)
    if group_counts != {"rank": 20, "control": 20}:
        raise ValueError("failed rank/control balance")
    outcome = {}
    for group in ("rank", "control", "all"):
        subset = [r for r in records if group == "all" or r["group"] == group]
        outcome[group] = {
            "n": len(subset),
            "baseline_ex": sum(r["arms"]["original"]["ex"] for r in subset),
            "treatment_ex": sum(r["arms"]["rank_contract"]["ex"] for r in subset),
            "rescues": sum(not r["arms"]["original"]["ex"] and r["arms"]["rank_contract"]["ex"] for r in subset),
            "regressions": sum(r["arms"]["original"]["ex"] and not r["arms"]["rank_contract"]["ex"] for r in subset),
            "changed_sql": sum(r["arms"]["original"]["sql"] != r["arms"]["rank_contract"]["sql"] for r in subset),
            "avg_pair_wall_s": round(sum(r["wall_s"] for r in subset)/len(subset), 2),
        }
    return {
        "n": len(records), "model_digest": parts[0]["model_digest"],
        "cohort_sha": parts[0]["cohort_sha"],
        "groups": outcome,
        "claim_boundary": "New question+DB-disjoint BIRD Train40 within 51 accessible DBs; public data and not untouched competition test. Neutral arm is identical-prompt A/A."
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path)
    p.add_argument("--dev", type=Path)
    p.add_argument("--work", type=Path, default=Path("rank-work"))
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.inputs:
        out = aggregate([json.loads(path.read_text()) for path in a.inputs])
        print("QWEN_RANK_FINAL", json.dumps(out, sort_keys=True), flush=True)
    else:
        if a.train is None or a.dev is None or a.shard is None:
            p.error("train/dev/shard required")
        out = inference_shard(
            source_train=a.train, source_dev=a.dev, shard=a.shard, work=a.work
        )
    a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
