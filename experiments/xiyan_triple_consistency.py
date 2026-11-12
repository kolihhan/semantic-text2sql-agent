"""Gold-blind XiYan triple candidate voting over observed SQLite executions.

Not a DPC implementation: a deliberately cheap three-candidate control for
testing whether execution agreement improves the existing 58/59 frozen arms.
Correctness labels are accessed only AFTER selection to score this ablation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

from experiments.xiyan_dgsl_frozen100 import (
    load_baseline_sql, resolve_database, score_query,
)
from experiments.xiyan_mschema_frozen100 import build_mschema
from evaluation.bird_loader import load_bird_json
from evaluation.schema_context import _description_rows, select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.providers import OllamaProvider


def _fingerprint(database: Path, sql: str, max_rows: int):
    if not isinstance(sql, str) or not sql.strip():
        return None
    try:
        result = execute_readonly(database, sql, max_rows=max_rows)
    except (SQLExecutionError, ValueError, TypeError):
        return None
    # Empty outputs lead to spurious agreement on this benchmark; they are not
    # affirmative semantic evidence. Truncated results cannot be compared.
    if result.truncated or not result.rows:
        return None
    try:
        return frozenset(tuple(row) for row in result.rows)
    except TypeError:
        return None


def consensus_choice(
    database: Path, *, lexical_sql: str, mschema_sql: str,
    independent_sql: str, max_rows: int = 50_000,
) -> dict:
    """No user question, gold SQL, reference output or EX score is accepted."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    choices = {
        "lexical": lexical_sql,
        "mschema": mschema_sql,
        "independent": independent_sql,
    }
    fingerprints = {
        arm: _fingerprint(database, sql, max_rows)
        for arm, sql in choices.items()
    }
    counts = Counter(x for x in fingerprints.values() if x is not None)
    valid = sum(x is not None for x in fingerprints.values())
    agreeing = {k for k, v in counts.items() if v >= 2}
    if len(agreeing) == 1:
        winning = next(iter(agreeing))
        for preference in ("mschema", "lexical", "independent"):
            if fingerprints[preference] == winning:
                return {
                    "chosen_arm": preference,
                    "chosen_sql": choices[preference],
                    "reason": "execution_agreement",
                    "valid_candidates": valid,
                    "agreement_size": counts[winning],
                }
    for fallback in ("mschema", "lexical", "independent"):
        if fingerprints[fallback] is not None:
            return {
                "chosen_arm": fallback,
                "chosen_sql": choices[fallback],
                "reason": "no_agreement_mschema_fallback" if fallback == "mschema" else "no_agreement_valid_fallback",
                "valid_candidates": valid,
                "agreement_size": 0,
            }
    return {
        "chosen_arm": "mschema", "chosen_sql": mschema_sql,
        "reason": "all_candidates_invalid", "valid_candidates": 0,
        "agreement_size": 0,
    }


def generate_independent_sql(
    *, provider: OllamaProvider, question: str,
    schema_context: str, evidence: str | None,
) -> str:
    # Independent third formulation. Does NOT receive either previous SQL,
    # the gold SQL, or execution outcomes; no LLM judging.
    prompt = f"""你是一名准确且谨慎的SQLite查询工程师。请独立将下面问题写成一条可执行的SELECT SQL。

请在生成之前仔细检查：
1. 哪些实体决定主表和JOIN路径？JOIN条件必须来自数据库schema。
2. 问题中的比较、日期、实体值是否准确落到WHERE/HAVING？只用可验证的实际列。
3. 计算的是行、去重实体还是分组？只有需要时才用DISTINCT/GROUP BY。
4. 对最高/最低/前N名，明确ORDER BY方向与LIMIT。
5. 不要添加用户问题没有要求的条件或列；没有可靠依据时保持简单。

【问题】
{question}

【数据库真实SQLite schema】
{schema_context}

【提供的领域解释】
{evidence or ''}

只返回一条SQL语句，不要Markdown，不要解释。
SQL:"""
    return _strip_fence(provider.complete_text(system="", user=prompt))


def run_shard(
    *, source: Path, database_root: Path, frozen: Path,
    lexical_baseline: Path, mschema_previous: Path,
    model: str, shard_index: int, num_shards: int,
) -> dict:
    if not 0 <= shard_index < num_shards or num_shards != 4:
        raise ValueError("exactly four frozen shards required")
    ordered = [str(row["case_id"]) for row in json.loads(frozen.read_text(encoding="utf-8"))["cases"]]
    if len(ordered) != 100 or len(set(ordered)) != 100:
        raise ValueError("frozen100 source does not have 100 distinct cases")
    selected = [case_id for idx, case_id in enumerate(ordered) if idx % num_shards == shard_index]
    original = load_baseline_sql(lexical_baseline)
    previous_payload = json.loads(mschema_previous.read_text(encoding="utf-8"))
    if previous_payload.get("schema") != "xiyan-mschema-frozen100-v1" or previous_payload.get("shard_index") != shard_index:
        raise ValueError("M-Schema shard identity mismatch")
    previous = {str(row["case_id"]): row for row in previous_payload["cases"]}
    if set(previous) != set(selected) or any(case_id not in original for case_id in selected):
        raise ValueError("missing/extra original frozen SQL cases")
    cases, gold = load_bird_json(source)
    case_by_id = {str(row.case_id): row for row in cases}
    gold_by_id = {str(row.case_id): row for row in gold}
    provider = OllamaProvider(model=model, timeout_s=900)
    catalogs = {}
    outputs = []
    for position, case_id in enumerate(selected, 1):
        case = case_by_id[case_id]
        if case.database_id not in catalogs:
            path = resolve_database(database_root, case.database_id)
            catalog = DatabaseCatalog.from_sqlite(path)
            catalogs[case.database_id] = (path, catalog)
        path, catalog = catalogs[case.database_id]
        lexical_schema, _ = select_schema_context(catalog, database_root, case.database_id, case.question)
        start = perf_counter()
        independent_sql = generate_independent_sql(
            provider=provider, question=case.question,
            schema_context=lexical_schema, evidence=case.evidence.strip() or None,
        )
        latency = perf_counter() - start
        original_sql = original[case_id]
        prior_sql = str(previous[case_id]["candidate_sql"])
        choice = consensus_choice(
            path, lexical_sql=original_sql, mschema_sql=prior_sql,
            independent_sql=independent_sql,
        )
        # No access to expected SQL until all candidates and choice are frozen.
        gold_sql = gold_by_id[case_id].sql
        lexical_ex = score_query(path, original_sql, gold_sql)["frozen_execution_match"]
        mschema_ex = score_query(path, prior_sql, gold_sql)["frozen_execution_match"]
        third_ex = score_query(path, independent_sql, gold_sql)["frozen_execution_match"]
        selected_ex = score_query(path, choice["chosen_sql"], gold_sql)["frozen_execution_match"]
        if bool(previous[case_id]["baseline_ex"]) != lexical_ex or bool(previous[case_id]["mschema_ex"]) != mschema_ex:
            raise ValueError(f"prior scorer drift on case {case_id}")
        outputs.append({
            "case_id": case_id, "database_id": case.database_id,
            "lexical_ex": lexical_ex, "mschema_ex": mschema_ex,
            "independent_ex": third_ex, "consensus_ex": selected_ex,
            "independent_sql": independent_sql,
            "chosen_arm": choice["chosen_arm"], "reason": choice["reason"],
            "valid_candidates": choice["valid_candidates"],
            "agreement_size": choice["agreement_size"],
            "third_generation_s": latency,
        })
        print(
            f"[triple {shard_index} {position}/25] "
            f"lexical={lexical_ex} mschema={mschema_ex} independent={third_ex} "
            f"consensus={selected_ex} chosen={choice['chosen_arm']}",
            flush=True,
        )
    return {
        "schema": "xiyan-triple-v1",
        "model": model,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "cases": outputs,
        "gold_prompt_access": False,
        "candidate_selection": "nonempty nontruncated SQLite execution consensus, fallback mschema",
        "source": "original frozen XiYan lexical and M-Schema JSON shards, third new independent prompt",
    }


def aggregate_triple_shards(shards: list[dict], *, min_correct: int = 61) -> dict:
    if len(shards) != 4 or sorted(x.get("shard_index") for x in shards) != list(range(4)):
        raise ValueError("exactly four distinct shards required")
    if any(x.get("schema") != "xiyan-triple-v1" for x in shards):
        raise ValueError("shard schema mismatch")
    models = {str(x.get("model")) for x in shards}
    if len(models) != 1:
        raise ValueError("model mismatch")
    rows = [r for shard in shards for r in shard.get("cases", [])]
    ids = [str(r["case_id"]) for r in rows]
    if len(rows) != 100 or len(set(ids)) != 100:
        raise ValueError("expected 100 unique cases")
    lexical = sum(bool(r["lexical_ex"]) for r in rows)
    mschema = sum(bool(r["mschema_ex"]) for r in rows)
    independent = sum(bool(r["independent_ex"]) for r in rows)
    consensus = sum(bool(r["consensus_ex"]) for r in rows)
    if lexical != 58 or mschema != 59:
        raise ValueError(f"frozen arm drift: lexical={lexical}, mschema={mschema}")
    return {
        "schema": "xiyan-triple-aggregate-v1",
        "sample_size": len(rows),
        "model": next(iter(models)),
        "lexical_ex": lexical,
        "mschema_ex": mschema,
        "independent_ex": independent,
        "consensus_ex": consensus,
        "net_vs_mschema": consensus - mschema,
        "wrong_to_correct_vs_mschema": sum(not r["mschema_ex"] and r["consensus_ex"] for r in rows),
        "correct_to_wrong_vs_mschema": sum(r["mschema_ex"] and not r["consensus_ex"] for r in rows),
        "selected_arm_counts": dict(Counter(r["chosen_arm"] for r in rows)),
        "selection_reason_counts": dict(Counter(r["reason"] for r in rows)),
        "gate": min_correct,
        "gate_met": consensus >= min_correct,
        "limitations": "Same previously exposed frozen100; no independent holdout. Project set-of-rows match, not official BIRD submission.",
        "cases": sorted(rows, key=lambda r: int(r["case_id"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--lexical-baseline", type=Path)
    parser.add_argument("--mschema-previous", type=Path)
    parser.add_argument("--model", type=str, default="hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-correct", type=int, default=61)
    args = parser.parse_args()
    if args.inputs:
        result = aggregate_triple_shards(
            [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs],
            min_correct=args.min_correct,
        )
        summary = {k: v for k, v in result.items() if k != "cases"}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        if any(value is None for value in (args.source, args.database_root, args.frozen, args.lexical_baseline, args.mschema_previous, args.shard_index)):
            parser.error("shard run requires all input paths and shard-index")
        result = run_shard(
            source=args.source, database_root=args.database_root,
            frozen=args.frozen, lexical_baseline=args.lexical_baseline,
            mschema_previous=args.mschema_previous, model=args.model,
            shard_index=args.shard_index, num_shards=args.num_shards,
        )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
