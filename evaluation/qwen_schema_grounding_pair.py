"""Qwen-only, paired schema-tokenization mechanism test on BIRD DEV.

Pre-register by question and schema context, never gold. Each arm is the same
pinned Qwen9B Guarded, with a different deterministic schema tokenizer only.
Unchanged-context controls quantify re-generation noise. The two-arm oracle is
a hindsight diagnostic, NOT a usable runtime selector.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from urllib.request import Request, urlopen

from evaluation.run_bird import _ollama_model_digest, _resolve_database
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly, SQLExecutionError
from semantic_sql.inference import run_guarded

SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
MODEL = "qwen3.5:9b-q4_K_M"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SEED = "qwen-schema-lexical-v2-pre-registered-20261114"
GROUP_SIZE = 8
N_SHARDS = 4
ARMS = ("legacy", "split_identifiers")


def legacy_tokens(value: str) -> set[str]:
    # Byte-for-byte logical equivalent of evaluation/schema_context.py before fix.
    words = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", value.casefold()))
    stopwords = {
        "a", "an", "and", "are", "be", "by", "for", "from", "has",
        "have", "how", "in", "is", "of", "on", "or", "the", "to",
        "what", "which", "who", "with",
    }
    return {w.rstrip("s") for w in words if w not in stopwords and len(w.rstrip("s")) > 1}


def checked_source(source: Path) -> list[dict]:
    if sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("BIRD source bytes changed")
    rows = json.loads(source.read_text(encoding="utf-8"))
    if len(rows) != 1534:
        raise ValueError("BIRD source row count changed")
    ids = [str(r.get("question_id") or r.get("id") or i) for i, r in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate BIRD IDs")
    return rows


def case_id(row: dict, index: int) -> str:
    return str(row.get("question_id") or row.get("id") or index)


def contexts(catalog: DatabaseCatalog, root: Path, db: str, question: str):
    original, old = select_schema_context(catalog, root, db, question, tokenizer=legacy_tokens)
    revised, new = select_schema_context(catalog, root, db, question)
    return (original, old), (revised, new)


def build_manifest(source: Path, root: Path, frozen_reference: Path) -> dict:
    rows = checked_source(source)
    reference = json.loads(frozen_reference.read_text(encoding="utf-8"))
    if reference.get("status") != "complete" or reference.get("sample_size") != 100:
        raise ValueError("frozen historical exclusion set invalid")
    excluded = {str(row["case_id"]) for row in reference["cases"]}
    if len(excluded) != 100:
        raise ValueError("duplicate historical IDs")
    indexed = sorted(
        [(case_id(row, i), row) for i, row in enumerate(rows)
         if case_id(row, i) not in excluded],
        key=lambda item: (sha256(f"{SEED}:{item[0]}".encode()).hexdigest(), item[0]),
    )
    chosen = {"changed": [], "control": []}
    db_counts = {"changed": Counter(), "control": Counter()}
    databases = {}
    observed_changed = 0
    observed_control = 0
    for cid, item in indexed:
        dbid = str(item["db_id"])
        if dbid not in databases:
            path = _resolve_database(root, dbid)
            databases[dbid] = DatabaseCatalog.from_sqlite(path)
        old, new = contexts(databases[dbid], root, dbid, str(item["question"]))
        delta = old[0] != new[0]
        observed_changed += int(delta)
        observed_control += int(not delta)
        kind = "changed" if delta else "control"
        if len(chosen[kind]) >= GROUP_SIZE or db_counts[kind][dbid] >= 4:
            continue
        chosen[kind].append({
            "case_id": cid, "database_id": dbid, "question": str(item["question"]),
            "evidence": str(item.get("evidence") or ""),
            "group": kind,
            "legacy_sha256": old[1]["schema_context_sha256"],
            "split_sha256": new[1]["schema_context_sha256"],
            "legacy_tables": old[1]["selected_tables"],
            "split_tables": new[1]["selected_tables"],
        })
        db_counts[kind][dbid] += 1
        # Always scan the full source to measure actual schema-context prevalence.
    if any(len(a) != GROUP_SIZE for a in chosen.values()):
        raise ValueError(
            f"insufficient preregistered changed/control cases: "
            f"{ {k: len(v) for k, v in chosen.items()} }"
        )
    selected = chosen["changed"] + chosen["control"]
    identity = [(r["case_id"], r["database_id"], r["group"]) for r in selected]
    cohort_hash = sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
    return {
        "protocol": "qwen-schema-tokenizer-paired16-v1",
        "source_sha256": SOURCE_SHA,
        "excluded_frozen100": 100, "selection_seed": SEED,
        "count": len(selected), "cohort_sha256": cohort_hash,
        "model_id": MODEL, "model_digest": MODEL_DIGEST,
        "preregistered_without_reference_sql": True,
        "selected": selected,
        "observed_full_source": {"changed": observed_changed, "control": observed_control},
        "limits": "Public BIRD DEV; question-disjoint from prior frozen100 only. "
                  "May overlap other prior experiments and model pretraining.",
    }


class PinnedQwen:
    def complete_text(self, *, system: str, user: str) -> str:
        req = Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps({
                "model": MODEL, "stream": False, "think": False,
                "options": {"temperature": 0, "seed": 42, "num_predict": 1024},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=900) as response:
            obj = json.load(response)
        return str(obj["message"]["content"])


def run_shard(source: Path, manifest_path: Path, root: Path, shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("shard out of range")
    rows = checked_source(source)
    cfg = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (cfg.get("protocol") != "qwen-schema-tokenizer-paired16-v1"
        or cfg.get("source_sha256") != SOURCE_SHA or cfg.get("model_digest") != MODEL_DIGEST
        or cfg.get("count") != 16 or len(cfg.get("selected", [])) != 16
        or cfg.get("preregistered_without_reference_sql") is not True):
        raise ValueError("unrecognized preregistered cohort")
    model_hash = _ollama_model_digest("http://127.0.0.1:11434", MODEL)
    if model_hash != MODEL_DIGEST:
        raise ValueError("pinned Qwen9B Ollama digest changed")
    picked = cfg["selected"][4 * shard:4 * (shard + 1)]
    if len(picked) != 4:
        raise ValueError("incomplete shard")
    by_id = {case_id(r, i): r for i, r in enumerate(rows)}
    dbs = {}
    work = []
    for row in picked:
        cid, dbid = row["case_id"], row["database_id"]
        raw = by_id[cid]
        if raw["db_id"] != dbid or raw["question"] != row["question"]:
            raise ValueError("question or database drift")
        if dbid not in dbs:
            path = _resolve_database(root, dbid)
            dbs[dbid] = (path, DatabaseCatalog.from_sqlite(path))
        (old_ctx, old), (new_ctx, new) = contexts(dbs[dbid][1], root, dbid, row["question"])
        if (old["schema_context_sha256"] != row["legacy_sha256"]
            or new["schema_context_sha256"] != row["split_sha256"]):
            raise ValueError(f"schema grounding context drift {cid}")
        work.append({**row, "_schemas": (old_ctx, new_ctx), "arms": {}})
    provider = PinnedQwen()
    # Freeze both model candidates before accessing ANY reference SQL for scoring.
    for item in work:
        db, _ = dbs[item["database_id"]]
        for arm, ctx in zip(ARMS, item["_schemas"]):
            start = perf_counter()
            result = run_guarded(
                database=db, provider=provider,
                question=item["question"], schema_context=ctx,
                evidence=item["evidence"].strip() or None,
                max_repairs=2, max_rows=50_000,
            )
            good = result.status == "ok" and not result.truncated
            item["arms"][arm] = {
                "sql": result.candidate.sql if result.candidate else "",
                "_rows": result.rows if good else (),
                "execution_success": good,
                "status": result.status,
                "repairs": result.candidate.attempt if result.candidate else -1,
                "latency_s": round(perf_counter()-start, 3),
            }
            print("QWEN_SCHEMA_PAIR_GENERATED", shard, item["case_id"], arm, flush=True)
    for item in work:
        item.pop("_schemas")
        db, _ = dbs[item["database_id"]]
        try:
            gold = execute_readonly(db, by_id[item["case_id"]]["SQL"], max_rows=50_000)
            if gold.truncated:
                raise ValueError("gold rows truncated")
        except (SQLExecutionError, ValueError) as exc:
            item["reference_scorable"] = False
            item["reference_error"] = type(exc).__name__
            for arm in ARMS:
                item["arms"][arm].pop("_rows")
                item["arms"][arm].update(ex=None, bag=None, both_empty=None)
            continue
        item["reference_scorable"] = True
        item["reference_error"] = None
        for arm in ARMS:
            result = item["arms"][arm]
            predicted = result.pop("_rows")
            is_ok = result["execution_success"]
            result.update(
                ex=bool(is_ok and set(predicted) == set(gold.rows)),
                bag=bool(is_ok and Counter(predicted) == Counter(gold.rows)),
                both_empty=bool(is_ok and not gold.rows and not predicted),
            )
    return {
        "protocol": cfg["protocol"], "shard": shard,
        "cohort_sha256": cfg["cohort_sha256"], "model_digest": model_hash,
        "cases": work,
    }


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != N_SHARDS or sorted(s["shard"] for s in shards) != list(range(N_SHARDS)):
        raise ValueError("need four unique shards")
    first = shards[0]
    if any(
        s["protocol"] != "qwen-schema-tokenizer-paired16-v1"
        or s["cohort_sha256"] != first["cohort_sha256"]
        or s["model_digest"] != MODEL_DIGEST
        or len(s["cases"]) != 4 for s in shards
    ):
        raise ValueError("model, cohort or protocol mismatch")
    cases = [case for s in shards for case in s["cases"]]
    if len({c["case_id"] for c in cases}) != 16:
        raise ValueError("duplicate cases")
    def stats(group: list[dict]):
        val = [c for c in group if c["reference_scorable"]]
        return {
            "n": len(group), "scorable": len(val), "unscorable": len(group)-len(val),
            "original_ex": sum(bool(c["arms"]["legacy"]["ex"]) for c in val),
            "fixed_ex": sum(bool(c["arms"]["split_identifiers"]["ex"]) for c in val),
            "original_bag": sum(bool(c["arms"]["legacy"]["bag"]) for c in val),
            "fixed_bag": sum(bool(c["arms"]["split_identifiers"]["bag"]) for c in val),
            "rescues": sum(not c["arms"]["legacy"]["ex"] and c["arms"]["split_identifiers"]["ex"] for c in val),
            "regressions": sum(c["arms"]["legacy"]["ex"] and not c["arms"]["split_identifiers"]["ex"] for c in val),
            "oracle_ex": sum(c["arms"]["legacy"]["ex"] or c["arms"]["split_identifiers"]["ex"] for c in val),
            "sql_changed": sum(c["arms"]["legacy"]["sql"] != c["arms"]["split_identifiers"]["sql"] for c in group),
            "both_empty": sum(any(bool(c["arms"][a]["both_empty"]) for a in ARMS) for c in val),
        }
    return {
        "n": 16, "cohort_sha256": first["cohort_sha256"],
        "model_id": MODEL, "model_digest": MODEL_DIGEST,
        "groups": {key: stats([c for c in cases if c["group"] == key])
                   for key in ("changed", "control")},
        "overall": stats(cases), "cases": cases,
        "no_training": True, "gold_blind_generation": True,
        "oracle_not_a_runtime_selector": True,
        "decision": "not promoted; require separate database/source-shift confirmation",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--reference", type=Path, default=Path("runs/bird-paired-dev-v5/paired.json"))
    p.add_argument("--database-root", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("refusing to overwrite result")
    if a.inputs:
        result = aggregate([json.loads(path.read_text()) for path in a.inputs])
        print("QWEN_SCHEMA_PAIR_FINAL", json.dumps({k:v for k,v in result.items() if k!="cases"}, sort_keys=True))
    elif a.shard is not None:
        if a.source is None or a.database_root is None or a.manifest is None:
            p.error("paired mode requires --source, --database-root, --manifest")
        result = run_shard(a.source, a.manifest, a.database_root, a.shard)
    else:
        if a.source is None or a.database_root is None:
            p.error("manifest mode requires --source and --database-root")
        result = build_manifest(a.source, a.database_root, a.reference)
        print("QWEN_SCHEMA_PAIR_PREREGISTERED",
              json.dumps({k:v for k,v in result.items() if k!="selected"}, sort_keys=True), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
