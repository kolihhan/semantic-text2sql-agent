"""BIRD DEV24: Qwen-only real-DB facts vs original Guarded SQL generation.

The challenger sees bounded physically observed PK/FK and exact-entity evidence,
not a free-form semantic review. Freeze 24 question IDs using question/schema/
database probes ONLY; score gold SQL after both Qwen arms are complete.
This is exploratory public BIRD DEV and must not be treated as hidden OOD.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

from evaluation.physical_db_evidence import verified_db_facts
from evaluation.run_bird import _resolve_database, _ollama_model_digest
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly, SQLExecutionError
from semantic_sql.inference import run_guarded

SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
QWEN = "qwen3.5:9b-q4_K_M"
QWEN_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SEED = "qwen-real-physical-db-evidence-24-20261114-v1"
COHORT_COUNTS = {"value": 4, "relation": 12, "control": 8}
SHARDS, ITEMS_PER_SHARD = 6, 4
ARMS = ("baseline", "physical_facts")
# Previous strict Qwen schema-token pilot, excluded to avoid repeating known cases.
PREVIOUS_QWEN16 = {
    "632", "706", "1477", "629", "635", "1140", "1097", "1112",
    "1156", "195", "721", "1422", "391", "155", "821", "851",
}


def _source(source: Path) -> list[dict]:
    if sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("pinned BIRD DEV content changed")
    data = json.loads(source.read_text(encoding="utf-8"))
    if len(data) != 1534:
        raise ValueError("unexpected BIRD DEV row count")
    return data


def _case_id(row: dict, i: int) -> str:
    return str(row.get("question_id") or row.get("id") or i)


def _schema_and_facts(
    dbpath: Path, catalog: DatabaseCatalog, root: Path, row: dict,
):
    context, diag = select_schema_context(
        catalog, root, str(row["db_id"]), str(row["question"])
    )
    ranked = sorted(
        diag["selected_tables"],
        key=lambda table: (-int(diag["table_scores"].get(table, 0)), table),
    )[:8]
    facts = verified_db_facts(
        dbpath, catalog, set(ranked), str(row["question"]), max_tables=8,
        max_scans=24,
    )
    return context, diag, facts


def build_manifest(source: Path, dbroot: Path, baseline: Path) -> dict:
    rows = _source(source)
    prior = json.loads(baseline.read_text(encoding="utf-8"))
    if prior.get("sample_size") != 100 or prior.get("status") != "complete":
        raise ValueError("missing exact 100 frozen reference exclusions")
    frozen_ids = {str(x["case_id"]) for x in prior["cases"]}
    if len(frozen_ids) != 100 or len(PREVIOUS_QWEN16) != 16:
        raise ValueError("invalid exclusion list")
    excluded = frozen_ids | PREVIOUS_QWEN16
    ordered = sorted(
        ((_case_id(row, i), row) for i, row in enumerate(rows)
         if _case_id(row, i) not in excluded),
        key=lambda pair: (sha256(f"{SEED}:{pair[0]}".encode()).hexdigest(), pair[0]),
    )
    selected = {k: [] for k in COHORT_COUNTS}
    perdb = {k: Counter() for k in COHORT_COUNTS}
    dbcache = {}
    inspected = 0
    for cid, row in ordered:
        dbid = str(row["db_id"])
        if dbid not in dbcache:
            path = _resolve_database(dbroot, dbid)
            dbcache[dbid] = (path, DatabaseCatalog.from_sqlite(path))
        dbpath, catalog = dbcache[dbid]
        ctx, diag, facts = _schema_and_facts(dbpath, catalog, dbroot, row)
        inspected += 1
        positive = facts.value_matches > 0
        if (positive and len(selected["value"]) < COHORT_COUNTS["value"]
            and perdb["value"][dbid] < 3):
            group = "value"
        elif ((facts.pk_lines or facts.fk_lines)
              and len(selected["relation"]) < COHORT_COUNTS["relation"]
              and perdb["relation"][dbid] < 3):
            group = "relation"
        elif (len(selected["control"]) < COHORT_COUNTS["control"]
              and perdb["control"][dbid] < 3):
            group = "control"
        else:
            continue
        selected[group].append({
            "case_id": cid, "db_id": dbid, "question": str(row["question"]),
            "evidence": str(row.get("evidence") or ""), "group": group,
            "schema_sha256": diag["schema_context_sha256"],
            "facts_sha256": sha256(facts.text.encode()).hexdigest(),
            "value_matches": facts.value_matches, "fk_lines": facts.fk_lines,
            "pk_lines": facts.pk_lines, "probes": facts.probes,
        })
        perdb[group][dbid] += 1
        if all(len(selected[g]) == count for g, count in COHORT_COUNTS.items()):
            break
    if any(len(selected[k]) != n for k, n in COHORT_COUNTS.items()):
        raise ValueError(f"preregister missing strata { {k:len(v) for k,v in selected.items()} }")
    flat = [case for k in COHORT_COUNTS for case in selected[k]]
    ids = [r["case_id"] for r in flat]
    if len(flat) != 24 or len(set(ids)) != 24:
        raise ValueError("case overlap or incomplete registration")
    key = sha256(json.dumps(
        [(r["case_id"], r["db_id"], r["group"], r["schema_sha256"], r["facts_sha256"]) for r in flat],
        separators=(",", ":"),
    ).encode()).hexdigest()
    return {
        "protocol": "qwen-physical-db-facts24-v1", "cohort_sha256": key,
        "source_sha256": SOURCE_SHA, "model": QWEN, "model_digest": QWEN_DIGEST,
        "seed": SEED, "counts": COHORT_COUNTS,
        "selection_scanned_without_gold": inspected,
        "excluded_prior_ids": len(excluded),
        "gold_used_for_selection": False, "gold_used_for_generation": False,
        "selected": flat,
        "limits": "Public BIRD DEV, excludes 100 frozen +16 Qwen pilot only; "
                  "remaining questions/DBs may overlap other prior experiments or model training.",
    }


class PinnedQwen:
    def complete_text(self, *, system: str, user: str) -> str:
        request = Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps({
                "model": QWEN, "stream": False, "think": False,
                "options": {"temperature": 0, "seed": 42, "num_predict": 1024},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=900) as response:
            obj = json.load(response)
        return str(obj["message"]["content"])


def run_shard(source: Path, dbroot: Path, manifest: Path, shard: int) -> dict:
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    rows = _source(source)
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    if (cfg.get("protocol") != "qwen-physical-db-facts24-v1"
        or cfg.get("source_sha256") != SOURCE_SHA
        or cfg.get("model") != QWEN or cfg.get("model_digest") != QWEN_DIGEST
        or cfg.get("counts") != COHORT_COUNTS
        or cfg.get("gold_used_for_selection") is not False
        or cfg.get("gold_used_for_generation") is not False
        or len(cfg.get("selected", [])) != 24):
        raise ValueError("unrecognized frozen cohort identity")
    digest = _ollama_model_digest("http://127.0.0.1:11434", QWEN)
    if digest != QWEN_DIGEST:
        raise ValueError("Qwen model drift")
    chosen = cfg["selected"][shard * ITEMS_PER_SHARD:(shard + 1) * ITEMS_PER_SHARD]
    if len(chosen) != ITEMS_PER_SHARD:
        raise ValueError("incomplete shard")
    indexed = {_case_id(row, i): row for i,row in enumerate(rows)}
    cache = {}
    work = []
    for case in chosen:
        cid, dbid = case["case_id"], case["db_id"]
        raw = indexed[cid]
        if str(raw["db_id"]) != dbid or raw["question"] != case["question"]:
            raise ValueError("BIRD source identity drift")
        if dbid not in cache:
            path = _resolve_database(dbroot, dbid)
            cache[dbid] = (path, DatabaseCatalog.from_sqlite(path))
        path, catalog = cache[dbid]
        schema, diag, facts = _schema_and_facts(path,catalog,dbroot,case)
        if (diag["schema_context_sha256"] != case["schema_sha256"]
            or sha256(facts.text.encode()).hexdigest() != case["facts_sha256"]):
            raise ValueError("physical evidence drift")
        challenger_schema = schema
        if case["group"] != "control" and facts.text:
            challenger_schema = (
                schema
                + "\nPhysical SQLite facts, independently verified read-only (not SQL answers). "
                "Keys/relationships do not imply a particular user-intent interpretation:\n"
                + facts.text
            )
        work.append({**case, "_schemas": (schema, challenger_schema), "arms": {}})
    provider = PinnedQwen()
    for rec in work:
        path, _catalog = cache[rec["db_id"]]
        for arm, schema in zip(ARMS, rec["_schemas"]):
            start = perf_counter()
            outcome = run_guarded(
                database=path, provider=provider,
                question=rec["question"], schema_context=schema,
                evidence=rec["evidence"].strip() or None, max_repairs=2,
                max_rows=50_000,
            )
            success = outcome.status == "ok" and not outcome.truncated
            rec["arms"][arm] = {
                "sql": outcome.candidate.sql if outcome.candidate else "",
                "execution_success": success,
                "_rows": outcome.rows if success else (),
                "status": outcome.status,
                "repairs": outcome.candidate.attempt if outcome.candidate else -1,
                "latency_s": round(perf_counter()-start,3),
            }
            print("QWEN_PHYSICAL_CANDIDATE", shard, rec["case_id"], arm, flush=True)
    # Gold SQL is not loaded until ALL candidate generations have finished.
    for rec in work:
        rec.pop("_schemas")
        path, _catalog = cache[rec["db_id"]]
        try:
            gold = execute_readonly(
                path, indexed[rec["case_id"]]["SQL"], max_rows=50_000,
            )
            if gold.truncated:
                raise ValueError("truncated gold result")
        except (SQLExecutionError, ValueError) as exc:
            rec.update(reference_scorable=False, reference_error=type(exc).__name__)
            for arm in ARMS:
                rec["arms"][arm].pop("_rows")
                rec["arms"][arm].update(ex=None, bag=None)
            continue
        rec.update(reference_scorable=True, reference_error=None)
        for arm in ARMS:
            output = rec["arms"][arm]
            rows = output.pop("_rows")
            ok = output["execution_success"]
            output["ex"] = bool(ok and set(rows)==set(gold.rows))
            output["bag"] = bool(ok and Counter(rows)==Counter(gold.rows))
    return {
        "protocol": cfg["protocol"], "shard": shard, "model_digest": digest,
        "cohort_sha256": cfg["cohort_sha256"], "cases": work,
    }


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != SHARDS or sorted(s["shard"] for s in shards) != list(range(SHARDS)):
        raise ValueError("six unique complete shards required")
    first = shards[0]
    if any(
        s["protocol"] != "qwen-physical-db-facts24-v1"
        or s["cohort_sha256"] != first["cohort_sha256"]
        or s["model_digest"] != QWEN_DIGEST
        or len(s["cases"]) != ITEMS_PER_SHARD for s in shards
    ):
        raise ValueError("cohort, model or protocol drift")
    cases = [case for shard in shards for case in shard["cases"]]
    if len(cases)!=24 or len({x["case_id"] for x in cases})!=24:
        raise ValueError("missing or duplicate cases")
    def measure(group: list[dict]) -> dict:
        scorable=[r for r in group if r["reference_scorable"]]
        return {
            "n":len(group), "scorable":len(scorable), "invalid_gold":len(group)-len(scorable),
            "baseline_ex":sum(bool(r["arms"]["baseline"]["ex"]) for r in scorable),
            "challenger_ex":sum(bool(r["arms"]["physical_facts"]["ex"]) for r in scorable),
            "baseline_bag":sum(bool(r["arms"]["baseline"]["bag"]) for r in scorable),
            "challenger_bag":sum(bool(r["arms"]["physical_facts"]["bag"]) for r in scorable),
            "rescues":sum(not r["arms"]["baseline"]["ex"] and r["arms"]["physical_facts"]["ex"] for r in scorable),
            "regressions":sum(r["arms"]["baseline"]["ex"] and not r["arms"]["physical_facts"]["ex"] for r in scorable),
            "oracle_ex":sum(bool(r["arms"]["baseline"]["ex"] or r["arms"]["physical_facts"]["ex"]) for r in scorable),
            "changed_sql":sum(r["arms"]["baseline"]["sql"]!=r["arms"]["physical_facts"]["sql"] for r in group),
            "value_matches":sum(r["value_matches"] for r in group),
        }
    return {
        "protocol":first["protocol"], "model":QWEN, "model_digest":first["model_digest"],
        "cohort_sha256":first["cohort_sha256"], "overall":measure(cases),
        "groups":{name:measure([r for r in cases if r["group"]==name]) for name in COHORT_COUNTS},
        "cases":cases,"no_training":True,"gold_blind_generation":True,
        "oracle_is_hindsight_only":True,
        "not_a_hidden_or_cross_database_generalization_claim":True,
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--database-root",type=Path)
    parser.add_argument("--reference",type=Path,
                        default=Path("runs/bird-paired-dev-v5/paired.json"))
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--shard",type=int)
    parser.add_argument("--inputs",nargs="+",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse overwrite")
    if args.inputs:
        report=aggregate([json.loads(p.read_text(encoding="utf-8")) for p in args.inputs])
        print("QWEN_PHYSICAL_FINAL",json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)
    elif args.shard is not None:
        if args.source is None or args.database_root is None or args.manifest is None:
            parser.error("paired mode requires source, database-root, manifest")
        report=run_shard(args.source,args.database_root,args.manifest,args.shard)
    else:
        if args.source is None or args.database_root is None:
            parser.error("preregister requires source and database-root")
        report=build_manifest(args.source,args.database_root,args.reference)
        print("QWEN_PHYSICAL_PREREGISTERED",json.dumps({k:v for k,v in report.items() if k!="selected"},sort_keys=True),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
