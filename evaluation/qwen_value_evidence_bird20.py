"""Qwen3.5-9B: original Guarded vs question-matched physical SQLite value evidence.

Test a new *generation input* rather than another candidate selector. No training,
gold-blind selection, database-disjoint from prior two BIRD DEV20/100 studies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

from evaluation.run_bird import _ollama_model_digest, _resolve_database
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.grounding import build_value_index, match_indexed_values
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
PROTOCOL = "qwen9b-bird20-db-disjoint-question-grounded-values-v1"
SEED = "value-candidate-20261115-source-only-v1"
N_SHARDS = 5
MAX_ROWS = 50000


def checked_source(path: Path) -> list[dict]:
    if sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("pinned BIRD DEV source SHA changed")
    rows = json.loads(path.read_text(encoding="utf8"))
    if len(rows) != 1534:
        raise ValueError("pinned BIRD DEV row count changed")
    return rows


def case_id(row: dict, i: int) -> str:
    return str(row.get("question_id") or row.get("id") or i)


def _key(s: str) -> str:
    return sha256((SEED + ":" + s).encode()).hexdigest()


def _index(catalog: DatabaseCatalog):
    return build_value_index(catalog, max_values_per_column=64, max_value_chars=80)


def _hits(index, question: str, evidence: str = "") -> list[dict]:
    raw = match_indexed_values(index, question=question, evidence=evidence, max_hits=10)
    selected = []
    for hit in raw:
        value = hit.value.strip()
        if (len(value) < 4 or len(value) > 80 or "\n" in value or
                value.casefold() in {"true", "false", "none", "null", "unknown"}):
            continue
        selected.append({"table": hit.table, "column": hit.column, "value": value})
        if len(selected) == 6:
            break
    return selected


def value_hits(catalog: DatabaseCatalog, question: str, evidence: str = "") -> list[dict]:
    """Question/evidence and real DB only; does not inspect gold SQL or outcomes."""
    return _hits(_index(catalog), question, evidence)


def enhanced_context(context: str, hits: list[dict]) -> str:
    if not hits:
        raise ValueError("non-eligible question has no grounded values")
    lines = [
        "Matched physical SQLite values (data only; never copy a value unless the question asks for it):"
    ]
    lines.extend(
        f"- {h['table']}.{h['column']} = {json.dumps(h['value'], ensure_ascii=False)}"
        for h in hits
    )
    return context + "\n" + "\n".join(lines)


def make_manifest(dev: list[dict], old_reference: dict, previous_cohort: dict,
                  database_root: Path) -> dict:
    old_cases = old_reference.get("cases", [])
    previous = previous_cohort.get("selected", [])
    if len(old_cases) != 100 or len(previous) != 20:
        raise ValueError("historical source-disjoint exclusions missing")
    forbidden_db = {str(row["database_id"]) for row in old_cases} | {
        str(row["db_id"]) for row in previous
    }
    forbidden_id = {str(row["case_id"]) for row in old_cases + previous}
    by_db = defaultdict(list)
    cats, indexed = {}, {}
    for i, row in enumerate(dev):
        db, cid = str(row["db_id"]), case_id(row, i)
        if db in forbidden_db or cid in forbidden_id:
            continue
        if db not in cats:
            cats[db] = DatabaseCatalog.from_sqlite(_resolve_database(database_root, db))
            indexed[db] = _index(cats[db])
        hits = _hits(indexed[db], row["question"], str(row.get("evidence") or ""))
        if hits:
            by_db[db].append({
                "case_id": cid, "db_id": db, "question": row["question"],
                "evidence": str(row.get("evidence") or ""), "value_hits": hits,
            })
    eligible = sorted(
        (db for db, rows in by_db.items() if len(rows) >= 2),
        key=lambda db: _key("db:" + db),
    )
    if len(eligible) < 10:
        raise ValueError(
            f"Only {len(eligible)} never-seen DBs have >=2 eligible questions; stop without inference"
        )
    selected = []
    for db in eligible[:10]:
        selected.extend(
            sorted(by_db[db], key=lambda r: _key("case:" + r["case_id"]))[:2]
        )
    if len(selected) != 20 or len({row["db_id"] for row in selected}) != 10:
        raise AssertionError("cohort selection drift")
    cohort = sha256(json.dumps(
        [(row["case_id"], row["db_id"], row["value_hits"]) for row in selected],
        sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()
    return {
        "protocol": PROTOCOL, "model": MODEL, "model_digest": DIGEST,
        "source_sha256": SOURCE_SHA, "cohort_sha256": cohort,
        "cases": selected, "count": 20, "db_count": 10,
        "excluded_db_count": len(forbidden_db),
        "eligibility": "bounded exact first-party SQLite text value in question/evidence",
        "selection": "hash-by-DB then hash-by-question; before gold/outcome",
        "no_training": True, "no_gold_for_retrieval": True,
        "not_full_BIRD_claim": True,
    }


def _evaluate(db: Path, question: str, schema: str, evidence: str) -> dict:
    provider = OllamaProvider(
        model=MODEL, base_url="http://127.0.0.1:11434",
        timeout_s=900, temperature=0, seed=42, num_predict=1024,
    )
    start = monotonic()
    result = run_guarded(
        database=db, provider=provider, question=question,
        schema_context=schema, evidence=evidence or None,
        max_repairs=2, max_rows=MAX_ROWS,
    )
    valid = result.status == "ok" and not result.truncated
    return {
        "sql": result.candidate.sql if result.candidate else "",
        "status": result.status, "executable": valid,
        "rows": result.rows if valid else (),
        "latency_s": round(monotonic() - start, 2),
    }


def run_shard(source: Path, manifest: Path, root: Path, shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("bad shard")
    cfg = json.loads(manifest.read_text())
    if (cfg.get("protocol") != PROTOCOL or cfg.get("source_sha256") != SOURCE_SHA
        or cfg.get("count") != 20 or cfg.get("db_count") != 10
        or len(cfg.get("cases", [])) != 20):
        raise ValueError("manifest drift")
    digest = _ollama_model_digest("http://127.0.0.1:11434", MODEL)
    if digest != DIGEST:
        raise ValueError("Qwen9B physical digest drift")
    rows = checked_source(source)
    by_id = {case_id(row, i): row for i, row in enumerate(rows)}
    selected = cfg["cases"][shard*4:(shard+1)*4]
    out = []
    for target in selected:
        original = by_id[target["case_id"]]
        if (original["db_id"] != target["db_id"]
                or original["question"] != target["question"]):
            raise ValueError("source/manifest question mismatch")
        db = _resolve_database(root, target["db_id"])
        catalog = DatabaseCatalog.from_sqlite(db)
        hits = value_hits(catalog, target["question"], target["evidence"])
        if hits != target["value_hits"]:
            raise ValueError("physical value evidence drift from preregistered manifest")
        context, _ = select_schema_context(
            catalog, root, target["db_id"], target["question"]
        )
        base = _evaluate(db, target["question"], context, target["evidence"])
        grounded = _evaluate(
            db, target["question"], enhanced_context(context, hits),
            target["evidence"],
        )
        out.append({
            "case_id": target["case_id"], "db_id": target["db_id"],
            "baseline": base, "evidence": grounded, "hit_count": len(hits),
        })
        print("QWEN_VALUE_GROUNDING_FROZEN", shard, target["case_id"], flush=True)

    # No reference SQL touched until BOTH arms have frozen candidates for the shard.
    for outcome in out:
        row = by_id[outcome["case_id"]]
        db = _resolve_database(root, outcome["db_id"])
        try:
            gold = execute_readonly(db, str(row["SQL"]), max_rows=MAX_ROWS)
            scorable = not gold.truncated
        except SQLExecutionError:
            scorable, gold = False, None
        outcome["reference_scorable"] = scorable
        for arm in ("baseline", "evidence"):
            pred = outcome[arm]
            predicted = pred.pop("rows")
            pred["ex"] = (
                bool(pred["executable"] and set(map(tuple, predicted))
                     == set(map(tuple, gold.rows)))
                if scorable else None
            )
            pred["bag"] = (
                bool(pred["executable"] and Counter(map(tuple, predicted))
                     == Counter(map(tuple, gold.rows)))
                if scorable else None
            )
    return {
        "protocol": PROTOCOL, "shard": shard,
        "cohort_sha256": cfg["cohort_sha256"],
        "model_digest": digest, "cases": out,
    }


def aggregate(shards: list[dict], *, expected_digest: str = DIGEST) -> dict:
    if len(shards) != N_SHARDS or sorted(
        shard.get("shard") for shard in shards
    ) != list(range(N_SHARDS)):
        raise ValueError("five shards required")
    if (len({shard.get("cohort_sha256") for shard in shards}) != 1
        or any(shard.get("protocol") != PROTOCOL
               or shard.get("model_digest") != expected_digest
               or len(shard.get("cases", [])) != 4 for shard in shards)):
        raise ValueError("model/cohort drift")
    rows = [row for shard in sorted(shards, key=lambda x:x["shard"])
            for row in shard["cases"]]
    if (len(rows) != 20 or len({row["case_id"] for row in rows}) != 20
        or len({row["db_id"] for row in rows}) != 10):
        raise ValueError("duplicate/incomplete source split")
    valid = [row for row in rows if row["reference_scorable"]]
    old = sum(row["baseline"]["ex"] is True for row in valid)
    new = sum(row["evidence"]["ex"] is True for row in valid)
    rescued = [row for row in valid if row["baseline"]["ex"] is False
               and row["evidence"]["ex"] is True]
    regressed = [row for row in valid if row["baseline"]["ex"] is True
                 and row["evidence"]["ex"] is False]
    if new-old != len(rescued)-len(regressed):
        raise AssertionError("paired EX accounting drift")
    bag_old = sum(row["baseline"]["bag"] is True for row in valid)
    bag_new = sum(row["evidence"]["bag"] is True for row in valid)
    db_gain = len({row["db_id"] for row in rescued})
    return {
        "protocol": PROTOCOL, "model_digest": expected_digest,
        "count": len(rows), "scorable": len(valid), "dbs": 10,
        "baseline_ex": old, "evidence_ex": new,
        "baseline_bag": bag_old, "evidence_bag": bag_new,
        "rescues": len(rescued), "regressions": len(regressed),
        "improved_db_count": db_gain,
        "latency_base_s": round(sum(row["baseline"]["latency_s"] for row in rows),1),
        "latency_evidence_s": round(sum(row["evidence"]["latency_s"] for row in rows),1),
        "positive_screen_only": len(valid) == 20 and new-old >= 3
                                and len(regressed) <= 1 and db_gain >= 3
                                and bag_new >= bag_old,
        "promotion_allowed": False, "no_training": True,
        "caveat": "Gold-blind value-eligible DEV20 on 10 DBs disjoint from frozen100 and ICL20; conditional pilot, not full BIRD/OOD proof",
        "cases": rows,
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("source", "reference", "old-icl", "db-root", "manifest", "output"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    args = p.parse_args()
    if args.output is None or args.output.exists():
        raise FileExistsError("unique output path required; refuse overwrite")
    if args.inputs:
        result = aggregate([json.loads(path.read_text()) for path in args.inputs])
        print("QWEN_VALUE_EVIDENCE_BIRD20_FINAL", json.dumps(
            {k:v for k,v in result.items() if k!="cases"},sort_keys=True), flush=True)
    elif args.manifest:
        if args.source is None or args.db_root is None or args.shard is None:
            raise ValueError("shard needs source, db-root, manifest, shard")
        result = run_shard(args.source, args.manifest, args.db_root, args.shard)
    else:
        if any(x is None for x in (args.source, args.reference, args.old_icl, args.db_root)):
            raise ValueError("preflight needs source, db-root and both prior cohorts")
        result = make_manifest(
            checked_source(args.source),
            json.loads(args.reference.read_text()),
            json.loads(args.old_icl.read_text()), args.db_root)
        print("QWEN_VALUE_EVIDENCE_COHORT_FROZEN", json.dumps(
            {k:v for k,v in result.items() if k!="cases"},sort_keys=True), flush=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
