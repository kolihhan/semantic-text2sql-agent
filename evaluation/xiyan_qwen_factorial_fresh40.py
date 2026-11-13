"""Pre-registered zero-training 2x2: XiYanSQL 7B vs Qwen3.5 9B, two fixed prompts.

One question-disjoint BIRD DEV slice for all arms. Same physical SQLite DBs,
lexical schema context, model seed, temperature, disabled thinking and scorer.
Count intent/control strata; no gold SQL available for cohort selection or
inference. No hidden test, no cross-database-disjoint claim, no promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import time
import urllib.request

from evaluation.bird_loader import load_bird_json
from evaluation.run_bird import _ollama_model_digest, _resolve_database
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _SYSTEM, _strip_fence
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.verifier import verify_sql_preflight
from semantic_sql.semantic_shape_v3 import _COUNT_INTENT


MODEL_IDS = {
    "qwen": "qwen3.5:9b-q4_K_M",
    "xiyan": "hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M",
}
QWEN_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SOURCE_SHA256 = "630272f2b1c44d8cef2c3b246f623355cf0bbc1e832c81061df895530dfc2f06"
SELECT_SEED = "sql-xiyan-qwen-factorial-2x2-fresh40-20261113"
MODEL_SEED = 42
COHORT_COUNTS = (100, 100, 100, 100, 40, 20, 40, 40)
ARM_KEYS = ("qwen:generic", "qwen:xiyan", "xiyan:generic", "xiyan:xiyan")
N_SHARDS = 8


def source_cohort_ids(paths: list[Path]) -> set[str]:
    if len(paths) != len(COHORT_COUNTS):
        raise ValueError("eight source cohorts required: 4x100, 40,20,40,40")
    seen: set[str] = set()
    for path, count in zip(paths, COHORT_COUNTS, strict=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("cases")
        if not isinstance(rows, list):
            rows = payload.get("selected")
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"invalid original cohort {path}: expected {count}")
        ids = [str(row["case_id"]) for row in rows]
        if len(set(ids)) != count or set(ids).intersection(seen):
            raise ValueError("overlap in previously evaluated DEV question IDs")
        seen.update(ids)
    if len(seen) != 540:
        raise AssertionError("frozen exclusion count is not 540")
    return seen


def manifest(*, source: Path, excluded: list[Path]) -> dict:
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("unrecognized source BIRD DEV content")
    seen = source_cohort_ids(excluded)
    cases, _unused_reference = load_bird_json(source)
    by_id = {str(c.case_id): c for c in cases}
    if len(by_id) != len(cases):
        raise ValueError("source case identities are not unique")
    selected: dict[str, list[dict]] = {"count_intent": [], "control": []}
    databases: dict[str, Counter] = {k: Counter() for k in selected}
    ordered = sorted(
        (c for c in cases if str(c.case_id) not in seen),
        key=lambda c: (sha256(f"{SELECT_SEED}:{c.case_id}".encode()).hexdigest(), str(c.case_id)),
    )
    for item in ordered:
        group = "count_intent" if _COUNT_INTENT.match(item.question) else "control"
        if len(selected[group]) == 20 or databases[group][item.database_id] >= 3:
            continue
        selected[group].append({
            "case_id": str(item.case_id),
            "database_id": item.database_id,
            "group": group,
        })
        databases[group][item.database_id] += 1
        if all(len(items) == 20 for items in selected.values()):
            break
    if any(len(items) != 20 or len(databases[group]) < 7 for group, items in selected.items()):
        raise ValueError("not enough eligible database-diverse cases in both strata")
    chosen = selected["count_intent"] + selected["control"]
    if len({row["case_id"] for row in chosen}) != 40:
        raise AssertionError("selection is not question unique")
    return {
        "schema": "bird-2x2-xiyan-qwen-preregistered/v1",
        "selection_seed": SELECT_SEED,
        "source_sha256": SOURCE_SHA256,
        "excluded_identity_count": 540,
        "strata": {"count_intent": 20, "control": 20},
        "distinct_databases_by_stratum": {k: len(v) for k, v in databases.items()},
        "model_ids": MODEL_IDS,
        "prompts": ["generic", "xiyan"],
        "temperature": 0,
        "thinking": False,
        "model_seed": MODEL_SEED,
        "max_output_tokens": 1024,
        "gold_used_for_selection": False,
        "gold_used_for_generation": False,
        "selected": chosen,
    }


def prompt_for(*, condition: str, question: str, schema_context: str,
               evidence: str | None) -> tuple[str, str]:
    if condition == "generic":
        prompt = f"Question: {question}\nSchema context:\n{schema_context}"
        if evidence is not None:
            prompt += f"\nExternal evidence:\n{evidence}"
        return _SYSTEM, prompt
    if condition == "xiyan":
        # The actual baseline prompt from experiments/xiyan_dgsl_frozen100.py.
        # Its opening SQL fence is intentional and preserved across models.
        prompt = (
            "你是一名SQLite专家，现在需要阅读并理解下面的【数据库schema】描述，"
            "以及可能用到的【参考信息】，并运用SQLite知识生成sql语句回答【用户问题】。\n"
            f"【用户问题】\n{question}\n\n"
            f"【数据库schema】\n{schema_context}\n\n"
            f"【参考信息】\n{evidence or ''}\n\n"
            f"【用户问题】\n{question}\n\n"
            "只返回一条只读SQLite SQL，不要解释。\n"
            + chr(96) * 3 + "sql"
        )
        return "", prompt
    raise ValueError("unknown fixed prompt condition")


def generate(*, model: str, system: str, user: str, timeout: float = 900) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": MODEL_SEED, "num_predict": 1024},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result.get("message"), dict):
        raise RuntimeError("Ollama response lacks message")
    return _strip_fence(str(result["message"]["content"]).strip())


def _ex(db: Path, sql: str, gold_rows: set[tuple]) -> dict:
    checked = verify_sql_preflight(db, sql)
    if not checked.ok:
        return {"ex": False, "execution_success": False, "preflight": False,
                "reason": ",".join(issue.code for issue in checked.issues)}
    try:
        got = execute_readonly(db, sql, max_rows=50_000)
    except (SQLExecutionError, ValueError) as exc:
        return {"ex": False, "execution_success": False, "preflight": True,
                "reason": "execution_failed:" + type(exc).__name__}
    return {
        "ex": not got.truncated and set(got.rows) == gold_rows,
        "execution_success": not got.truncated,
        "preflight": True,
        "reason": "truncated" if got.truncated else "executed",
    }


def _verify_manifest(cfg: dict, source: Path) -> None:
    if (cfg.get("schema") != "bird-2x2-xiyan-qwen-preregistered/v1"
        or cfg.get("source_sha256") != SOURCE_SHA256
        or sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256
        or cfg.get("selection_seed") != SELECT_SEED
        or cfg.get("excluded_identity_count") != 540
        or cfg.get("strata") != {"count_intent": 20, "control": 20}
        or cfg.get("model_ids") != MODEL_IDS
        or cfg.get("model_seed") != MODEL_SEED
        or cfg.get("gold_used_for_selection") is not False
        or cfg.get("gold_used_for_generation") is not False
        or len(cfg.get("selected", [])) != 40
        or len({str(row["case_id"]) for row in cfg["selected"]}) != 40):
        raise ValueError("unrecognized or changed preregistered benchmark")


def shard(*, source: Path, database_root: Path, manifest_path: Path,
          shard_id: int) -> dict:
    if shard_id not in range(N_SHARDS):
        raise ValueError("shard index must be in 0..7")
    cfg = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_manifest(cfg, source)
    expected = cfg["selected"][shard_id::N_SHARDS]
    if len(expected) != 5:
        raise ValueError("five matched cases per shard required")
    digests = {
        key: _ollama_model_digest("http://127.0.0.1:11434", name)
        for key, name in MODEL_IDS.items()
    }
    if digests["qwen"] != QWEN_DIGEST:
        raise ValueError("Qwen9B model digest differs from frozen original")
    if any(len(value) != 64 for value in digests.values()):
        raise ValueError("model digest malformed")
    cases, _reference_unused = load_bird_json(source)
    by_id = {str(row.case_id): row for row in cases}
    cache: dict[str, tuple[Path, str, str]] = {}
    records = []
    for item in expected:
        original = by_id.get(item["case_id"])
        if original is None or original.database_id != item["database_id"]:
            raise ValueError("manifest question/database identity mismatch")
        if original.database_id not in cache:
            db = _resolve_database(database_root, original.database_id)
            cat = DatabaseCatalog.from_sqlite(db)
            context, _diagnostics = select_schema_context(
                cat, database_root, original.database_id, original.question,
            )
            cache[original.database_id] = (db, context, sha256(context.encode()).hexdigest())
        # Context must be per question, not just per database.
        db = cache[original.database_id][0]
        cat = DatabaseCatalog.from_sqlite(db)
        context, _diagnostics = select_schema_context(
            cat, database_root, original.database_id, original.question,
        )
        records.append({
            **item,
            "question": original.question,
            "evidence": original.evidence.strip() or None,
            "schema_sha256": sha256(context.encode()).hexdigest(),
            "_schema": context,
            "_db": db,
            "arms": {},
        })
    # Batch per model to reduce CPU-only model swapping. Every arm still
    # sees exactly the same fixed question, evidence and schema.
    for model_key, prompt_key in [tuple(key.split(":")) for key in ARM_KEYS]:
        for row in records:
            system, user = prompt_for(
                condition=prompt_key,
                question=row["question"], schema_context=row["_schema"],
                evidence=row["evidence"],
            )
            start = time.perf_counter()
            sql = generate(model=MODEL_IDS[model_key], system=system, user=user)
            latency = round(time.perf_counter() - start, 3)
            row["arms"][f"{model_key}:{prompt_key}"] = {
                "sql": sql, "latency_s": latency,
            }
            print("FACTORIAL_GENERATED", shard_id, row["case_id"],
                  model_key, prompt_key, "seconds", latency, flush=True)
    # Gold SQL is referenced only after all four generation arms are frozen.
    _unread, labels = load_bird_json(source)
    gold_by_id = {str(item.case_id): item.sql for item in labels}
    for row in records:
        gold_sql = gold_by_id[row["case_id"]]
        db = row.pop("_db")
        row.pop("_schema")
        try:
            gold_result = execute_readonly(db, gold_sql, max_rows=50_000)
        except SQLExecutionError as exc:
            raise RuntimeError(f"gold SQL failed {row['case_id']}") from exc
        if gold_result.truncated:
            raise ValueError("gold execution truncated; do not score unreliable case")
        gold_rows = set(gold_result.rows)
        for name in ARM_KEYS:
            row["arms"][name].update(_ex(db, row["arms"][name]["sql"], gold_rows))
    return {
        "schema": "bird-2x2-xiyan-qwen-shard/v1",
        "shard": shard_id,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "model_digests": digests,
        "model_ids": MODEL_IDS,
        "model_seed": MODEL_SEED,
        "gold_used_for_generation": False,
        "cases": records,
    }


def _transitions(rows: list[dict], challenger: str, reference: str) -> dict:
    rescue = sum(bool(r["arms"][challenger]["ex"]) and
                 not bool(r["arms"][reference]["ex"]) for r in rows)
    regress = sum(bool(r["arms"][reference]["ex"]) and
                  not bool(r["arms"][challenger]["ex"]) for r in rows)
    return {"from": reference, "to": challenger, "rescues": rescue,
            "regressions": regress, "net": rescue - regress}


def aggregate(inputs: list[dict]) -> dict:
    if len(inputs) != 8 or sorted(x.get("shard") for x in inputs) != list(range(8)):
        raise ValueError("eight unique successful shards required")
    if any(x.get("schema") != "bird-2x2-xiyan-qwen-shard/v1"
           or x.get("gold_used_for_generation") is not False
           or x.get("model_ids") != MODEL_IDS
           or x.get("model_seed") != MODEL_SEED for x in inputs):
        raise ValueError("shard configuration or gold separation inconsistent")
    if len({x["manifest_sha256"] for x in inputs}) != 1:
        raise ValueError("manifest mismatch across shards")
    if len({json.dumps(x["model_digests"], sort_keys=True) for x in inputs}) != 1:
        raise ValueError("model digest drift across shards")
    if inputs[0]["model_digests"]["qwen"] != QWEN_DIGEST:
        raise ValueError("pinned Qwen identity drift")
    rows = [row for item in inputs for row in item["cases"]]
    if (len(rows) != 40 or len({x["case_id"] for x in rows}) != 40
        or Counter(x["group"] for x in rows) != {"count_intent": 20, "control": 20}
        or any(set(x["arms"]) != set(ARM_KEYS) for x in rows)):
        raise ValueError("incomplete or duplicated 2x2 matched design")
    by_group = {
        group: [r for r in rows if r["group"] == group]
        for group in ("count_intent", "control")
    }

    def stats(records: list[dict]) -> dict:
        counts = {
            arm: {
                "ex_correct": sum(bool(r["arms"][arm]["ex"]) for r in records),
                "execution_success": sum(bool(r["arms"][arm]["execution_success"]) for r in records),
                "preflight_pass": sum(bool(r["arms"][arm]["preflight"]) for r in records),
            }
            for arm in ARM_KEYS
        }
        return {
            "n": len(records),
            "arms": counts,
            "model_effect_under_generic": _transitions(records, "xiyan:generic", "qwen:generic"),
            "model_effect_under_xiyan_prompt": _transitions(records, "xiyan:xiyan", "qwen:xiyan"),
            "prompt_effect_qwen": _transitions(records, "qwen:xiyan", "qwen:generic"),
            "prompt_effect_xiyan": _transitions(records, "xiyan:xiyan", "xiyan:generic"),
        }

    return {
        "schema": "bird-2x2-xiyan-qwen-aggregate/v1",
        "n": 40, "manifest_sha256": inputs[0]["manifest_sha256"],
        "model_ids": MODEL_IDS, "model_digests": inputs[0]["model_digests"],
        "seed": MODEL_SEED, "temperature": 0, "think": False,
        "max_output_tokens": 1024, "no_training": True,
        "same_source_schema_database_scorer": True,
        "gold_used_for_generation": False,
        "overall": stats(rows),
        "by_group": {k: stats(v) for k, v in by_group.items()},
        "promotion": "NO; inspect failures and independently replicate first",
        "claim_boundary": "40 question-disjoint public BIRD DEV cases, not DB-disjoint nor hidden; direct generation only, no repair.",
        "cases": sorted(rows, key=lambda row: int(row["case_id"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--exclude", type=Path, nargs="+")
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite frozen artifact")
    if args.inputs:
        result = aggregate([json.loads(p.read_text()) for p in args.inputs])
    elif args.exclude:
        if args.source is None:
            parser.error("source required when preregistering")
        result = manifest(source=args.source, excluded=args.exclude)
    else:
        if args.source is None or args.database_root is None or args.manifest is None or args.shard is None:
            parser.error("source, database-root, manifest and shard required")
        result = shard(source=args.source, database_root=args.database_root,
                       manifest_path=args.manifest, shard_id=args.shard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print("XIYAN_QWEN_2X2_STAGE", json.dumps({
        k: v for k, v in result.items() if k not in ("selected", "cases"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
