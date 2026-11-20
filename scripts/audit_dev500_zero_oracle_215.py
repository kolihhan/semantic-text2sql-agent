"""BIRD DEV500: exhaustive diagnostic of 215 original Qwen9B K8 zero-oracle cases.

Pure CPU. Reuses exactly the 32 original sealed K8 SQL caches and eight
original uniformly 8s bounded physical-score worker artifacts.
The official 2025-11-06 dev source is SHA-locked and used ONLY for the
500 previously frozen development questions; HOLDOUT300 is never selected.
Gold reference SQL is used POSTHOC for structural comparison only, never
for generating/picking a candidate. Structural differences are NOT causal
failure diagnoses and are not official BIRD EX.

Outputs:
- 215 distinct question IDs and evidence-based per-case flags (JSON)
- aggregate machine-readable counts / breakdown (JSON)
- compact readable Markdown with verified facts and caution.
No Qwen calls, no data training, no changes to model/runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from sqlglot import parse_one, exp

SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
CACHE_PROTOCOL = "FROZEN-BIRD-DEV500-QWEN35-9B-KCACHE-V1"
WORKER_PROTOCOL = "BIRD-DEV500-SAME-ORIGINAL-QWEN9B-K8-CACHE-UNIFORM-8S-SQLITE-PREFIX-FRONTIER-V1"
STUDY = "ORIGINAL-QWEN9B-BIRD-DEV500-ZERO-ORACLE-215-FORENSIC-CPU-V1"
FLAGGED_OPS = ("join", "group", "where", "order", "limit", "count", "sum", "avg",
               "distinct", "division", "exists", "union", "intersect", "except")
OPERATORS: dict[str, type] = {
    "join": exp.Join, "group": exp.Group, "where": exp.Where,
    "order": exp.Order, "limit": exp.Limit, "count": exp.Count,
    "sum": exp.Sum, "avg": exp.Avg, "distinct": exp.Distinct,
    "division": exp.Div, "exists": exp.Exists, "union": exp.Union,
    "intersect": exp.Intersect, "except": exp.Except,
}


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def features(sql: str) -> dict[str, Any] | None:
    try:
        node = parse_one(sql, read="sqlite", error_level="RAISE")
        if not node:
            return None
        select = node if isinstance(node, exp.Select) else node.find(exp.Select)
        if select is None:
            return None
        ops = {k for k, v in OPERATORS.items() if node.find(v) is not None}
        tables = sorted({t.name.casefold() for t in node.find_all(exp.Table)
                         if t.name})
        projection_width = len(select.expressions)
        if projection_width <= 0:
            return None
        shape = [projection_width, sorted(ops), tables]
        return {"tables": tables, "projection_width": projection_width,
                "ops": sorted(ops),
                "coarse_shape_sha256": digest(json.dumps(shape).encode())[:16]}
    except (Exception,):
        # SQLGlot may reject SQL that SQLite accepts, or parse questionable
        # candidate syntax. This is not a physical execution verdict.
        return None


def read_json(path: Path) -> dict:
    out = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(out, dict):
        raise ValueError(f"Expected object in {path}")
    return out


def load_original(cache_dir: Path, workers_dir: Path) -> tuple[dict, dict]:
    cache_files = list(cache_dir.glob("k8-dev500-candidate-cache-*.json"))
    worker_files = list(workers_dir.glob("prefix-frontier-worker-*.json"))
    if len(cache_files) != 32 or len(worker_files) != 8:
        raise ValueError(f"Need original 32 K8 caches and 8 physical workers, "
                         f"got {len(cache_files)}, {len(worker_files)}")
    caches: dict[str, dict] = {}
    workers: dict[str, dict] = {}
    cache_indices: set[int] = set()
    worker_indices: set[int] = set()

    for file in cache_files:
        d = read_json(file)
        shard = d.get("shard")
        if (not isinstance(shard, int) or shard in cache_indices
            or shard not in range(32)
            or d.get("protocol") != CACHE_PROTOCOL
            or d.get("status") != "CANDIDATE_CACHE_COMPLETE"
            or d.get("source_sha256") != SOURCE_SHA
            or d.get("model_digest") != MODEL_DIGEST
            or d.get("no_gold_sql_in_cache") is not True
            or d.get("k") != 8):
            raise ValueError(f"Wrong original K8 cache identity: {file}")
        cache_indices.add(shard)
        for c in d["cases"]:
            cid = str(c["case_id"])
            if cid in caches or len(c["samples"]) != 8:
                raise ValueError(f"Duplicate or incomplete candidate case {cid}")
            if len({s["index"] for s in c["samples"]}) != 8:
                raise ValueError(f"Missing K8 sample index for {cid}")
            caches[cid] = c

    for file in worker_files:
        d = read_json(file)
        ix = d.get("worker")
        if (not isinstance(ix, int) or ix in worker_indices or ix not in range(8)
            or d.get("protocol") != WORKER_PROTOCOL
            or d.get("status") != "SOURCE_LOCKED_K_PREFIX_WORKER_COMPLETE"
            or d.get("source_sha256") != SOURCE_SHA
            or d.get("model_digest") != MODEL_DIGEST
            or d.get("model_calls") != 0):
            raise ValueError(f"Wrong original physical worker identity: {file}")
        worker_indices.add(ix)
        for c in d["cases"]:
            cid = str(c["case_id"])
            if cid in workers:
                raise ValueError(f"Duplicate physical score {cid}")
            workers[cid] = c

    if (cache_indices != set(range(32)) or worker_indices != set(range(8))
        or len(caches) != 500 or len(workers) != 500
        or set(caches) != set(workers)
        or sum(len(x["samples"]) for x in caches.values()) != 4000):
        raise ValueError("Original source/score 500 × K8 union incomplete")
    for cid in caches:
        if caches[cid]["db_id"] != workers[cid]["db_id"]:
            raise ValueError(f"Database mismatch {cid}")
    if (sum(bool(x["k8_set_ex"]) for x in workers.values()) != 233 or
        sum(bool(x["pass8_oracle_set"]) for x in workers.values()) != 285 or
        sum(bool(x["gold_scorable"]) for x in workers.values()) != 499):
        raise ValueError("Cannot reproduce original certified DEV500 frontier")
    return caches, workers


def load_dev(source: Path, manifest: Path, case_ids: set[str]) -> dict[str, dict]:
    data = source.read_bytes()
    if digest(data) != SOURCE_SHA:
        raise ValueError("Original official dev JSON SHA256 drift")
    m = read_json(manifest)
    dev = set(str(x) for x in m["dev_case_ids"])
    sealed_holdout = set(str(x) for x in m["holdout_case_ids"])
    if (len(dev) != 500 or len(sealed_holdout) != 300
        or dev & sealed_holdout or dev != case_ids or
        m.get("source_sha256") != SOURCE_SHA):
        raise ValueError("Original frozen 500 source IDs or sealed 300 drift")
    rows = json.loads(data)
    if len(rows) != 1534:
        raise ValueError("Original BIRD dev source not 1534 rows")
    selected: dict[str, dict] = {}
    for r in rows:
        cid = str(r["question_id"])
        if cid in dev:
            if cid in selected:
                raise ValueError(f"Duplicate source ID {cid}")
            selected[cid] = r
    if set(selected) != dev:
        raise ValueError("Cannot map all frozen IDs to exact original source")
    return selected


def simple_intents(question: str) -> list[str]:
    q = question.casefold()
    signals = {
        "count": r"\b(how many|number of|count|total number)\b",
        "ratio_percent": r"\b(percent(?:age)?|ratio|proportion|fraction|growth rate)\b",
        "ranking": r"\b(top\s*\d*|highest|lowest|largest|smallest|most|least|rank|first|last)\b",
        "aggregation": r"\b(sum|average|mean|total amount|maximum|minimum)\b",
        "relationship": r"\b(related|connected|belong|owns?|parent|child|joined|associated)\b",
        "time_range": r"\b(between|before|after|during|from)\b.{0,50}\b(?:19|20)\d{2}\b",
    }
    return [name for name, pattern in signals.items() if re.search(pattern, q)]


def audit(caches: dict, scores: dict, rows: dict) -> tuple[dict, dict, str]:
    failures = []
    by_db: dict[str, Counter] = defaultdict(Counter)
    by_difficulty: dict[str, Counter] = defaultdict(Counter)
    total = Counter()
    ref_parse_fail = 0
    examples: dict[str, list] = defaultdict(list)
    for cid in sorted(caches, key=int):
        source = rows[cid]
        original = caches[cid]
        scored = scores[cid]
        question = str(source["question"])
        if (original["db_id"] != source["db_id"]
            or original["question_sha256"] != digest(question.encode())):
            raise ValueError(f"Original SQL prompt ↔ dev source SHA/DB mismatch {cid}")
        db = original["db_id"]
        level = str(source.get("difficulty") or "unlabeled")
        by_db[db]["all"] += 1
        by_difficulty[level]["all"] += 1
        if scored["pass8_oracle_set"]:
            by_db[db]["has_gold_matched_candidate"] += 1
            by_difficulty[level]["has_gold_matched_candidate"] += 1
            continue

        # The one scored publisher timeout #1126 is certified all-invalid;
        # do not claim reference execution success or an actual gold result.
        executed = int(scored["candidate_executed"])
        if executed < 0 or executed > 8:
            raise ValueError("Bad physical executed count")
        execution_class = ("none_executable" if executed == 0 else
                           "all_executable_but_wrong" if executed == 8 else
                           "some_executable_none_correct")
        candidate_sql = [str(s["sql"]) for s in sorted(
            original["samples"], key=lambda x: x["index"])]
        candidate_features = [features(sql) for sql in candidate_sql]
        reference = features(str(source["SQL"]))
        if reference is None:
            ref_parse_fail += 1
        parseable = [v for v in candidate_features if v is not None]
        uniq_sql = len({re.sub(r"\s+", " ", sql.strip()).casefold()
                        for sql in candidate_sql})
        uniq_shapes = len({f["coarse_shape_sha256"] for f in parseable})
        flags: list[str] = []
        if reference and parseable:
            gold_tables = set(reference["tables"])
            if (gold_tables and all(
                not gold_tables <= set(v["tables"]) for v in parseable
            )):
                flags.append("no_parsed_candidate_has_all_reference_tables")
            if all(v["projection_width"] != reference["projection_width"]
                   for v in parseable):
                flags.append("no_parsed_candidate_matches_reference_output_width")
            for op in FLAGGED_OPS:
                if op in reference["ops"] and all(
                    op not in v["ops"] for v in parseable
                ):
                    flags.append("no_parsed_candidate_has_reference_" + op)
        elif reference is None:
            flags.append("reference_sql_ast_unparsed")
        else:
            flags.append("no_candidate_sql_ast_parsed")
        if uniq_shapes == 1 and len(parseable) == 8:
            flags.append("all_8_candidates_same_coarse_ast_shape")
        if uniq_sql == 1:
            flags.append("all_8_candidates_same_normalized_sql")
        if any(s.get("done_reason") == "length" for s in original["samples"]):
            flags.append("at_least_one_model_generation_hit_token_limit")

        out = {
            "case_id": cid, "db_id": db, "difficulty": level,
            "question_sha256": original["question_sha256"],
            "question_excerpt": " ".join(question.split())[:140],
            "question_intent_keywords": simple_intents(question),
            "physical_8s_reference_scorable": bool(scored["gold_scorable"]),
            "gold_reference_timeout_candidate_failure_certified":
                bool(not scored["gold_scorable"] and executed == 0),
            "original_k8_pass_any_gold_sql_result": False,
            "original_k8_selected_set_ex": bool(scored["k8_set_ex"]),
            "physical_executable_candidates": executed,
            "physical_timed_out_candidates": int(scored["candidate_timeouts"]),
            "execution_class": execution_class,
            "distinct_normalized_sql": uniq_sql,
            "parsed_candidate_sql_count": len(parseable),
            "distinct_coarse_ast_structures": uniq_shapes,
            "publisher_reference_sql_ast_parseable": reference is not None,
            "candidate_end_length_count": sum(
                s.get("done_reason") == "length" for s in original["samples"]),
            "reference_ast_compared_to_candidate_ast_flags": flags,
        }
        failures.append(out)
        total["failures"] += 1
        total["exec_" + execution_class] += 1
        total["fully_distinct_sql"] += int(uniq_sql == 8)
        total["all_same_sql"] += int(uniq_sql == 1)
        total["at_least_four_distinct_sql"] += int(uniq_sql >= 4)
        total["all_eight_candidate_ast_parsed"] += int(len(parseable) == 8)
        total["reference_ast_unparsed"] += int(reference is None)
        total["model_length_stop_cases"] += int(
            out["candidate_end_length_count"] > 0)
        total["model_length_stop_candidate_slots"] += out["candidate_end_length_count"]
        total["all_executable_correct_reference_unmatched"] += int(executed == 8)
        total["publisher_unscorable_certified"] += int(not scored["gold_scorable"])
        for flag in flags:
            total["flag_" + flag] += 1
            if len(examples[flag]) < 6:
                examples[flag].append(cid)
        by_db[db]["failures"] += 1
        by_db[db]["exec_" + execution_class] += 1
        by_difficulty[level]["failures"] += 1

    if (len(failures) != 215
        or total["exec_all_executable_but_wrong"] != 97
        or total["exec_some_executable_none_correct"] != 90
        or total["exec_none_executable"] != 28
        or total["publisher_unscorable_certified"] != 1):
        raise ValueError("Cannot reproduce the certified original 215-case cohort")
    if sum(x["all"] for x in by_db.values()) != 500:
        raise ValueError("DB population count mismatch")
    if ref_parse_fail != total["reference_ast_unparsed"]:
        raise ValueError("Reference AST denominator mismatch")

    ledger = {
        "protocol": STUDY,
        "status": "COMPLETE_ORIGINAL_QWEN9B_DEV500_215_ZERO_ORACLE_CASE_LEDGER",
        "source_sha256": SOURCE_SHA, "model_digest": MODEL_DIGEST,
        "n_cases_source_frozen": 500, "n_original_k8_sql": 4000,
        "n_zero_gold_matching_sql": 215, "n_recorded": len(failures),
        "metric_type": "physical_8s_bounded_set_EX_no_correct_candidate_at_K8",
        "source_public_DEV500_previously_exposed": True,
        "HOLDOUT300_cases_never_selected_or_scored": True,
        "no_new_Qwen_calls": True, "no_new_training": True,
        "no_change_to_default_pipeline": True,
        "gold_SQL_used_only_retrospectively_for_AST_flags": True,
        "structural_AST_flags_are_not_validated_root_causes": True,
        "cases": failures,
    }
    breakdown = {
        "protocol": STUDY, "status": "COMPLETE_FROZEN_DEV500_FORENSIC_NOT_EXPERIMENT",
        "source_sha256": SOURCE_SHA, "model_digest": MODEL_DIGEST,
        "counts": dict(total), "by_database": {
            k: dict(v) for k, v in sorted(by_db.items())},
        "by_publisher_difficulty": {
            k: dict(v) for k, v in sorted(by_difficulty.items())},
        "sample_case_ids_by_noncausal_ast_flag": dict(examples),
        "population": {
            "total": 500, "candidate_sql": 4000,
            "original_K8_oracle_upper_bound_correct": 285,
            "original_K8_goldblind_selected_correct": 233,
            "original_K8_no_correct_candidate": 215,
            "physical_db_count": len(by_db), "publisher_reference_scorable": 499,
            "reference_timeout_certified_all_invalid": 1,
        },
        "methodology": {
            "physical_execution_class_is_observed": True,
            "ast_structural_flags_are_posthoc_and_may_overlap": True,
            "AST_parsing_is_different_from_SQLite_execution": True,
            "reference_SQL_structural_disagreement_does_not_imply_wrong_semantics": True,
            "no_SQL_generative_or_selection_intervention": True,
            "no_model_calls": True, "HOLDOUT300_untouched": True,
        },
    }
    lines = [
        "# BIRD DEV500 — all 215 Qwen3.5-9B K8 zero-correct-candidate failures",
        "",
        "**Frozen original data only. No inference, training, tuning, hidden holdout, or production-code changes.**",
        "",
        "- Original Qwen9B model SHA: \`" + MODEL_DIGEST + "\`",
        "- Source SHA256: \`" + SOURCE_SHA + "\`",
        "- Physical scoring: 8-second read-only SQLite time limit, not a new official unbounded BIRD result",
        "- 32 immutable cache shards, 500 unique questions, 4,000 original model-generated SQLs",
        "- 8 original physical result-scoring worker artifacts; 499 reference SQLs scored; #1126 certified all-eight invalid and reference timeout",
        "",
        "## Observed operational failure classes (mutually exclusive)",
        "",
        "| Physical situation | Questions | What this proves |",
        "|---|---:|---|",
        "| All 8 candidates executed, **none matched reference** | 97 | Pure compiler repair cannot directly solve these observed mismatches |",
        "| Some executed, but none matched | 90 | Mixed reliability + result mismatch; semantic cause unknown |",
        "| 0/8 executed | 28 | Execution/reliability bottleneck; includes one all-invalid reference-timeout-certified question |",
        "| **All K8 no correct candidate** | **215** | Correct SQL not reachable in the eight original samples |",
        "",
        "## Mechanical candidate diversity, not meaning",
        "",
        f"- {total['fully_distinct_sql']}/215 have eight distinct normalized SQL texts; "
        "distinct strings do not imply correct or different semantic strategies.",
        f"- {total['at_least_four_distinct_sql']}/215 have at least four distinct SQL texts.",
        f"- {total['all_same_sql']}/215 generated the same normalized SQL eight times.",
        f"- {total['model_length_stop_cases']}/215 had at least one model response ending at the "
        "384-token generation cap; this is not necessarily the cause of the wrong answer.",
        "",
        "## Retrospective SQL AST comparison — evidence gaps, **not root causes**",
        "",
        "| Non-exclusive AST disagreement indicator | Number of cases |",
        "|---|---:|",
    ]
    for key, n in sorted(
        ((k.removeprefix("flag_"), int(v)) for k,v in total.items()
         if k.startswith("flag_")), key=lambda row:(-row[1],row[0])):
        lines.append(f"| \`{key}\` | {n} |")
    lines += [
        "",
        "These labels compare the *existing* gold SQL's AST with all eight already-generated candidate SQL ASTs. "
        "Alternate equivalent SQL can use different tables, projection widths, operators or values. "
        "A parser rejection is not a database execution failure. Counts overlap and must never be summed to 215.",
        "",
        "## Where to put effort, using evidence rather than candidate-count sweeps",
        "",
        "1. **97 fully executable zero-oracle cases:** priority for human blind review of relation roles, "
        "output grain, predicates and aggregation. Compare question + schema to one candidate **before** "
        "consulting gold SQL; then check the publisher reference and adjudicate ambiguity.",
        "2. **90 mixed cases:** split compiling/timeout problems from successful-but-wrong candidates, "
        "then investigate incorrect semantics on the executed subset.",
        "3. **28 zero-executable cases:** review identifier quoting, invented physical columns, "
        "unsupported dialect, pathological queries and generated token limits using actual compiler errors "
        "(the K8 cache itself does not contain individual compile-error reasons).",
        "4. This census alone does **not** justify a new QLoRA run, an optimizer, a new plan module, "
        "or a claim of more EX. Freeze a causal error mechanism and independent evaluation first.",
        "",
        "## Auditable source and restrictions",
        "",
        "- Original generation Actions #44086713331 and #44093103960.",
        "- Original complete physical frontier run #44222560767; independent CPU aggregate #44223938142.",
        "- CSV/JSON-style per-case source-ID ledger is in \`evidence/dev500-zero-oracle-215-ledger.json\`. "
        "It includes 215 case IDs, DB/difficulty, question excerpt, real execution counts, SQL-diversity stats "
        "and non-causal structural flags; no original 4,000 SQLs or reference gold SQL text copied into git.",
        "- Frozen BIRD DEV500 and HOLDOUT300 manifest: "
        "\`manifests/bird-dev500-holdout300-20261116-v2.json\`. "
        "The HOLDOUT300 case IDs were validated as disjoint but never scored/selected.",
        "- Dataset attribution: BIRD (Li et al., NeurIPS 2023); source mirrored by "
        "\`birdsql/bird_sql_dev_20251106\`, pinned revision \`3c11fb193e5439b338e23677fa0aae11e8b85db9\`.",
        "- The 500 DEV question identities and 11 physical databases are already exposed; "
        "neither this report nor subsequent tuning on these 215 proves independent generalization.",
        "",
    ]
    return ledger, breakdown, "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir",type=Path,required=True)
    p.add_argument("--workers-dir",type=Path,required=True)
    p.add_argument("--original-dev",type=Path,required=True)
    p.add_argument("--frozen-manifest",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise FileExistsError("Refuse to overwrite research evidence")
    caches, scores = load_original(a.cache_dir,a.workers_dir)
    questions = load_dev(a.original_dev,a.frozen_manifest,set(caches))
    ledger, breakdown, report = audit(caches,scores,questions)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    (a.out_dir/"dev500-zero-oracle-215-ledger.json").write_text(
        json.dumps(ledger,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    (a.out_dir/"dev500-zero-oracle-215-breakdown.json").write_text(
        json.dumps(breakdown,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    (a.out_dir/"dev500-zero-oracle-215-audit.md").write_text(report)
    print("FINAL_ORIGINAL_DEV500_215_FAILURES_CPU",
          json.dumps({"n":len(ledger["cases"]),"counts":breakdown["counts"],
                      "by_database":breakdown["by_database"]},sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
