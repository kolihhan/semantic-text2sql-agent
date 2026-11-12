"""A0/A1/A2/A3 XiYan semantic-layer ablation on question-disjoint BIRD DEV.

This module intentionally has no access to reference SQL while making candidate
or routing decisions. A1 is passive; A2 changes only provably noncompiling SQL;
A3 adds one grounded, bounded semantic repair if the question and SQL conflict.

Database overlap with prior BIRD DEV studies is possible; this is NOT an
independent cross-database or hidden-test result. Never tune on this same cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

from sqlglot import exp

from evaluation.bird_loader import load_bird_json
from evaluation.schema_context import _description_rows, select_schema_context
from experiments.xiyan_dgsl_frozen100 import generate_xiyan, resolve_database, score_query
from experiments.xiyan_mschema_frozen100 import build_mschema
from experiments.xiyan_semantic_shards import _parse_readonly_sql
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.providers import OllamaProvider
from semantic_sql.verifier import verify_sql_preflight

SEED = "semantic-sql-abc-question-disjoint-20261112-v1"
MODEL = "hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M"
_ARMS = ("a0", "a1", "a2", "a3")


def choose_disjoint_ids(all_ids: list[str], previous_ids: set[str], *,
                        count: int = 100, seed: str = SEED) -> list[str]:
    if count < 1 or not all_ids or len(set(all_ids)) != len(all_ids):
        raise ValueError("invalid question IDs")
    available = [x for x in all_ids if x not in previous_ids]
    if len(available) < count:
        raise ValueError("insufficient question-disjoint heldout cases")
    return sorted(available, key=lambda x: (sha256(f"{seed}:{x}".encode()).hexdigest(), x))[:count]


def contract_from_question(question: str, evidence: str = "") -> dict:
    """An intentionally *partial* typed contract: no guesses about columns.

    Only unambiguous output operators are actionable. Evidence and literals are
    retained as provenance rather than pretending to prove schema bindings.
    """
    text = " ".join(question.strip().split())
    lower = text.casefold()
    operation = None
    evidence_span = None
    patterns = (
        ("count", r"\bhow many\b|\bnumber of\b|\bcount of\b"),
        ("average", r"\baverage\b|\bmean\b"),
    )
    for name, pattern in patterns:
        found = re.search(pattern, lower)
        if found:
            operation = name
            evidence_span = text[found.start():found.end()]
            break
    numbers = tuple(dict.fromkeys(
        re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", question)
    ))
    return {
        "version": "partial-intent-v1",
        "operation": operation,
        "operation_source_span": evidence_span,
        "literal_evidence": list(numbers),
        "grain": "unknown",
        "schema_bindings": "unknown",
        "ranking_scope": "unknown",
        "evidence_available": bool(evidence.strip()),
    }


def _direct_projection_is_entity(select: exp.Select) -> bool:
    """Only flag an output when every projection is a plain field or star."""
    if not select.expressions:
        return False
    for projection in select.expressions:
        target = projection.this if isinstance(projection, exp.Alias) else projection
        if not isinstance(target, (exp.Column, exp.Star)):
            return False
    return True


def observe_semantics(question: str, evidence: str, sql: str) -> dict:
    contract = contract_from_question(question, evidence)
    ast = _parse_readonly_sql(sql)
    if ast is None:
        return {"verdict": "UNKNOWN", "reason": "sql_unparseable",
                "contract": contract, "provenance": []}
    if not isinstance(ast, exp.Select):
        return {"verdict": "UNKNOWN", "reason": "nonselect_root_scope",
                "contract": contract, "provenance": []}
    op = contract["operation"]
    plain = _direct_projection_is_entity(ast)
    if plain and op == "count":
        return {"verdict": "PROVEN_MISMATCH", "reason": "count_output_missing",
                "contract": contract, "provenance": [contract["operation_source_span"]]}
    if plain and op == "average":
        return {"verdict": "PROVEN_MISMATCH", "reason": "average_output_missing",
                "contract": contract, "provenance": [contract["operation_source_span"]]}
    return {"verdict": "UNKNOWN", "reason": "no_provable_mismatch",
            "contract": contract, "provenance": []}


def _checked_sql(database: Path, sql: str) -> tuple[bool, list[str]]:
    if _parse_readonly_sql(sql) is None:
        return False, ["parse_or_safety_error"]
    checked = verify_sql_preflight(database, sql)
    return checked.ok, [f"{i.code}: {i.message}" for i in checked.issues]


def _literal_lock(question: str, before: str, after: str) -> bool:
    """A repair cannot change a number directly stated by the user.

    This is a protective invariant for literals present in BOTH the original
    SQL and the question, not an assertion that all user numbers are literals.
    """
    original = _parse_readonly_sql(before)
    revised = _parse_readonly_sql(after)
    if original is None or revised is None:
        return False
    q_numbers = set(contract_from_question(question)["literal_evidence"])
    a_numbers = {v.this for v in original.find_all(exp.Literal) if not v.is_string}
    b_numbers = {v.this for v in revised.find_all(exp.Literal) if not v.is_string}
    return (q_numbers & a_numbers).issubset(b_numbers)


def _ast_clause(ast: exp.Expression, field: str) -> str:
    part = ast.args.get(field)
    if isinstance(part, list):
        return " ".join(x.sql(dialect="sqlite", normalize=True) for x in part)
    return part.sql(dialect="sqlite", normalize=True) if part is not None else ""


def _guard_semantic_repair(question: str, before: str, after: str) -> bool:
    """For output-only problems, *all* query population predicates are frozen."""
    a = _parse_readonly_sql(before)
    b = _parse_readonly_sql(after)
    if not isinstance(a, exp.Select) or not isinstance(b, exp.Select):
        return False
    if not _literal_lock(question, before, after):
        return False
    return all(_ast_clause(a, part) == _ast_clause(b, part)
               for part in ("from_", "joins", "where", "having"))


def _repair_compiler(provider: OllamaProvider, *, question: str, evidence: str,
                     schema: str, original: str, issues: list[str]) -> str:
    prompt = (
        "Repair ONE noncompiling read-only SQLite SELECT using the actual schema. "
        "Fix only invalid physical table/column names, aliases or syntax. "
        "Preserve the question's answer grain, aggregation, Boolean connectors, "
        "conditions, literals, and ordering. Never invent new constraints. "
        "Return SQL only.\n"
        f"QUESTION:\n{question}\nEVIDENCE:\n{evidence}\n"
        f"SCHEMA:\n{schema}\nSQL:\n{original}\nCOMPILER ISSUES:\n"
        + "\n".join(issues)
        + "\nREPAIRED SQL:"
    )
    return _strip_fence(provider.complete_text(system="", user=prompt))


def _repair_mismatch(provider: OllamaProvider, *, question: str, evidence: str,
                     schema: str, original: str, mismatch: dict) -> str:
    prompt = (
        "Repair ONLY this PROVEN answer-output semantic mismatch in one SQLite "
        "SELECT. Preserve FROM, JOIN, WHERE, HAVING, explicit numeric constants, "
        "Boolean connectors and filters exactly. Adjust output projection and "
        "necessary GROUP BY only. If impossible without changing the population, "
        "return the original SQL unchanged. Return read-only SQL only.\n"
        f"QUESTION:\n{question}\nEVIDENCE:\n{evidence}\n"
        f"SCHEMA:\n{schema}\nCURRENT SQL:\n{original}\n"
        f"MISMATCH: {mismatch['reason']}\n"
        f"EVIDENCE SPAN: {mismatch['provenance']}\nREPAIRED SQL:"
    )
    return _strip_fence(provider.complete_text(system="", user=prompt))


def run_one(database: Path, *, question: str, evidence: str, schema: str,
            baseline_sql: str, provider: OllamaProvider) -> dict:
    """Freeze A0/A1/A2/A3 without ever reading gold SQL."""
    observation = observe_semantics(question, evidence, baseline_sql)
    base_valid, issues = _checked_sql(database, baseline_sql)
    # A1 must be exactly observational, independent of model nondeterminism.
    a2_sql = baseline_sql
    a2_reason = "unchanged"
    model_calls = 0
    if not base_valid:
        proposal = _repair_compiler(
            provider, question=question, evidence=evidence,
            schema=schema, original=baseline_sql, issues=issues)
        model_calls += 1
        repaired_valid, _ = _checked_sql(database, proposal)
        if repaired_valid and (
            _parse_readonly_sql(baseline_sql) is None
            or _literal_lock(question, baseline_sql, proposal)
        ):
            a2_sql = proposal
            a2_reason = "compiler_repair_accepted"
        else:
            a2_reason = "compiler_repair_rejected"
    a3_sql = a2_sql
    a3_reason = "unchanged"
    candidate_obs = observe_semantics(question, evidence, a2_sql)
    if candidate_obs["verdict"] == "PROVEN_MISMATCH":
        proposal = _repair_mismatch(
            provider, question=question, evidence=evidence, schema=schema,
            original=a2_sql, mismatch=candidate_obs)
        model_calls += 1
        ok, _ = _checked_sql(database, proposal)
        next_obs = observe_semantics(question, evidence, proposal)
        if (ok and proposal.strip() != a2_sql.strip()
            and next_obs["verdict"] != "PROVEN_MISMATCH"
            and _guard_semantic_repair(question, a2_sql, proposal)):
            a3_sql = proposal
            a3_reason = "semantic_output_repair_accepted"
        else:
            a3_reason = "semantic_repair_rejected"
    return {
        "sql": {"a0": baseline_sql, "a1": baseline_sql, "a2": a2_sql, "a3": a3_sql},
        "baseline_valid": base_valid, "a1_observation": observation,
        "a2_reason": a2_reason, "a3_reason": a3_reason,
        "extra_model_calls": model_calls, "gold_used_for_routing": False,
    }


def run_shard(*, source: Path, database_root: Path, frozen: Path, previous: Path,
              shard_index: int, num_shards: int = 4, count: int = 100,
              model: str = MODEL) -> dict:
    if num_shards != 4 or count != 100 or shard_index not in range(4):
        raise ValueError("frozen protocol requires 4 shards x 25 cases")
    prior = json.loads(frozen.read_text(encoding="utf-8"))
    former = json.loads(previous.read_text(encoding="utf-8"))
    old_ids = {str(item["case_id"]) for item in prior["cases"]}
    second_ids = {str(item["case_id"]) for item in former["cases"]}
    if len(old_ids) != 100 or len(second_ids) != 100 or old_ids & second_ids:
        raise ValueError("old explored cohorts must be 100+100 non-overlapping IDs")
    cases, labels = load_bird_json(source)
    ids = choose_disjoint_ids([str(c.case_id) for c in cases],
                              old_ids | second_ids, count=count)
    cohort_sha = sha256("\n".join(ids).encode()).hexdigest()
    chosen = ids[shard_index::4]
    if len(chosen) != 25:
        raise ValueError("unexpected shard size")
    lookup = {str(c.case_id): c for c in cases}
    provider = OllamaProvider(model=model, timeout_s=900.0)
    cache = {}
    frozen_proposals = []
    for index, cid in enumerate(chosen, 1):
        case = lookup[cid]
        if case.database_id not in cache:
            db_path = resolve_database(database_root, case.database_id)
            catalog = DatabaseCatalog.from_sqlite(db_path)
            desc = {(table, col): d for table, col, d
                    in _description_rows(database_root, case.database_id)}
            cache[case.database_id] = (db_path, catalog, desc)
        db, catalog, descriptions = cache[case.database_id]
        _, lexical = select_schema_context(
            catalog, database_root, case.database_id, case.question)
        schema = build_mschema(
            catalog=catalog, database_id=case.database_id,
            selected_tables=lexical["selected_tables"],
            descriptions=descriptions)
        started = perf_counter()
        base = generate_xiyan(
            question=case.question, schema_context=schema,
            evidence=case.evidence or None, provider=provider)
        stage = run_one(
            db, question=case.question, evidence=case.evidence,
            schema=schema, baseline_sql=base, provider=provider)
        stage["case_id"] = cid
        stage["database_id"] = case.database_id
        stage["latency_s"] = round(perf_counter() - started, 3)
        frozen_proposals.append(stage)
        print(f"ABC_GENERATED [{shard_index}:{index}/25] id={cid} "
              f"a2={stage['a2_reason']} a3={stage['a3_reason']}", flush=True)

    # Post-selection gold evaluation: NO label is consulted in run_one,
    # contract extraction, model generation or routing.
    by_label = {str(label.case_id): label.sql for label in labels}
    results = []
    for item in frozen_proposals:
        cid = item["case_id"]
        db = cache[item["database_id"]][0]
        scores = {
            key: bool(score_query(db, sql, by_label[cid])["frozen_execution_match"])
            for key, sql in item["sql"].items()
        }
        if scores["a0"] != scores["a1"]:
            raise AssertionError("passive A1 must match baseline exactly")
        results.append({**item, "ex": scores})
    return {
        "schema": "semantic-abc-shard-v1", "shard": shard_index,
        "model": model, "cohort_sha256": cohort_sha, "count": len(results),
        "excluded_explored_ids": len(old_ids | second_ids),
        "gold_used_for_routing": False,
        "limits": "question-disjoint BIRD DEV, same databases possible; not hidden-test",
        "cases": results,
    }


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != 4 or sorted(s.get("shard") for s in shards) != [0,1,2,3]:
        raise ValueError("need exactly four distinct shards")
    if any(s.get("schema") != "semantic-abc-shard-v1"
           or s.get("gold_used_for_routing") is not False for s in shards):
        raise ValueError("invalid schema or gold leakage marker")
    if len({s["cohort_sha256"] for s in shards}) != 1 or len({s["model"] for s in shards}) != 1:
        raise ValueError("model/cohort mismatch")
    cases = [x for s in shards for x in s["cases"]]
    ids = [x["case_id"] for x in cases]
    if len(cases) != 100 or len(set(ids)) != 100:
        raise ValueError("cohort not 100 unique cases")
    scores = {arm: sum(bool(x["ex"][arm]) for x in cases) for arm in _ARMS}
    paired = {}
    for arm in ("a1","a2","a3"):
        gains = sum(not x["ex"]["a0"] and x["ex"][arm] for x in cases)
        losses = sum(x["ex"]["a0"] and not x["ex"][arm] for x in cases)
        paired[arm] = {"wrong_to_correct": gains, "correct_to_wrong": losses,
                       "net_gain": gains - losses}
        if gains - losses != scores[arm] - scores["a0"]:
            raise AssertionError("invalid paired arithmetic")
    if scores["a0"] != scores["a1"]:
        raise AssertionError("A1 mutated baseline")
    actions = Counter(x["a2_reason"] for x in cases)
    actions.update(x["a3_reason"] for x in cases)
    result = {
        "schema": "semantic-abc-aggregate-v1",
        "sample_size": 100, "cohort_sha256": shards[0]["cohort_sha256"],
        "model": shards[0]["model"], "scores": scores, "paired_vs_a0": paired,
        "actions": dict(sorted(actions.items())),
        "extra_model_calls": sum(x["extra_model_calls"] for x in cases),
        "mean_latency_s": round(sum(x["latency_s"] for x in cases)/100, 3),
        "passive_a1_invariant": True,
        "dataset": "BIRD DEV, questions disjoint from two previously explored 100-case cohorts",
        "caveat": "Not database-disjoint from prior BIRD DEV; project-local set-of-tuples EX. No tuning after labels.",
        "cases": sorted(cases, key=lambda row: int(row["case_id"])),
    }
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--database-root",type=Path)
    parser.add_argument("--frozen",type=Path)
    parser.add_argument("--previous",type=Path)
    parser.add_argument("--model",default=MODEL)
    parser.add_argument("--shard-index",type=int)
    parser.add_argument("--num-shards",type=int,default=4)
    parser.add_argument("--count",type=int,default=100)
    parser.add_argument("--inputs",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite old evidence")
    if args.inputs:
        result=aggregate([json.loads(x.read_text(encoding="utf-8")) for x in args.inputs])
        print("SEMANTIC_ABC_RESULT",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if any(x is None for x in (args.source,args.database_root,args.frozen,args.previous,args.shard_index)):
            parser.error("missing shard arguments")
        result=run_shard(source=args.source,database_root=args.database_root,
                         frozen=args.frozen,previous=args.previous,
                         shard_index=args.shard_index,num_shards=args.num_shards,
                         count=args.count,model=args.model)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()
