"""Independent BIRD semantic reranking with four Ollama/CPU shards.

Research question: can a semantic-audit revision recover correct SQL meaning
without blindly replacing executable M-Schema answers?

PREDECLARED, gold-blind protocol:
1. Choose new deterministic SHA256 holdout100 from BIRD DEV excluding the old
   frozen100. No correct SQL or previous labels influence selection.
2. Generate *paired* M-Schema baseline and a second semantic audit/revision
   using the identical XiYan 7B model, exact schema and relevant evidence.
3. Pick revision ONLY for (a) invalid baseline and valid revision, or
   (b) strictly fewer explicit question-intent violations; otherwise keep base.
4. Only AFTER selection compare against reference SQL via the project's
   frozen SQLite set-of-tuples scorer; audit losses/gains and oracle headroom.

Important: CPU shards increase throughput, NOT model intelligence. This
experiment tests whether targeted semantic repair beats brute consensus voting.
Unseen heldout performance, not a claim of official BIRD leaderboard score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

from experiments.xiyan_dgsl_frozen100 import generate_xiyan, resolve_database, score_query
from experiments.xiyan_mschema_frozen100 import build_mschema
from evaluation.bird_loader import load_bird_json
from evaluation.schema_context import _description_rows, select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.providers import OllamaProvider

SEED = "semantic-sql-shards-independent-dev-v1"
MODEL = "hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M"


def build_holdout_ids(all_case_ids: list[str], explored_case_ids: set[str], *,
                      count: int = 100, seed: str = SEED) -> list[str]:
    """Stable independent cohort selection BEFORE consulting BIRD reference SQL."""
    case_ids = [str(x) for x in all_case_ids]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("invalid or duplicated BIRD dev case IDs")
    eligible = [x for x in case_ids if x not in explored_case_ids]
    if count < 1 or len(eligible) < count:
        raise ValueError("insufficient independent holdout after frozen-case exclusion")
    return sorted(eligible,
                  key=lambda cid: (sha256((seed + ":" + cid).encode()).hexdigest(),cid))[:count]


def _required_intents(question: str) -> list[str]:
    lowered = " ".join(question.lower().split())
    required=[]
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount of\b",lowered):
        required.append("count")
    if re.search(r"\baverage\b|\bmean\b",lowered):
        required.append("average")
    if re.search(r"\bsum of\b|\bcombined total\b",lowered):
        required.append("sum")
    if re.search(r"\bhighest\b|\blargest\b|\bmaximum\b|\btop one\b|\bmost\b",lowered):
        required.append("maximum")
    if re.search(r"\blowest\b|\bsmallest\b|\bminimum\b|\bfewest\b|\bleast\b",lowered):
        required.append("minimum")
    if re.search(r"\bnumber of distinct\b|\bhow many distinct\b|\bhow many different\b|\bnumber of unique\b",lowered):
        required.append("distinct_count")
    return required


def _parse_readonly_sql(sql: str):
    import sqlglot
    from sqlglot import exp
    if not isinstance(sql,str) or not sql.strip():
        return None
    try:
        queries=sqlglot.parse(sql,read="sqlite")
    except (ValueError,TypeError,sqlglot.errors.ParseError,sqlglot.errors.TokenError):
        return None
    if len(queries)!=1 or queries[0] is None:
        return None
    parsed=queries[0]
    if not isinstance(parsed,(exp.Select,exp.Union,exp.Intersect,exp.Except)):
        return None
    forbidden=(exp.Insert,exp.Delete,exp.Update,exp.Create,exp.Drop,exp.Alter,exp.Command)
    if any(isinstance(node,forbidden) for node in parsed.walk()):
        return None
    return parsed


def check_question_constraints(question: str, sql: str) -> dict:
    """Deterministic, imperfect surface checks, not an oracle or SQL semantic proof."""
    from sqlglot import exp
    required=_required_intents(question)
    parsed=_parse_readonly_sql(sql)
    if parsed is None:
        return {"required":required,"missing":required,"parseable_readonly":False}
    has_count=parsed.find(exp.Count) is not None
    has_avg=parsed.find(exp.Avg) is not None
    has_sum=parsed.find(exp.Sum) is not None
    has_distinct_count=any(
        isinstance(count.this,exp.Distinct) for count in parsed.find_all(exp.Count)
    )
    ordering=list(parsed.find_all(exp.Order))
    limits=list(parsed.find_all(exp.Limit))
    has_order_desc=any(
        node.args.get("desc") is True
        for order in ordering for node in order.expressions
    )
    has_order_asc=any(
        node.args.get("desc") is not True
        for order in ordering for node in order.expressions
    )
    has_max=(parsed.find(exp.Max) is not None) or bool(has_order_desc and limits)
    has_min=(parsed.find(exp.Min) is not None) or bool(has_order_asc and limits)
    conditions={
        "count":has_count,
        "average":has_avg,
        "sum":has_sum,
        "distinct_count":has_distinct_count,
        "maximum":has_max,
        "minimum":has_min,
    }
    return {
        "required":required,
        "missing":[name for name in required if not conditions[name]],
        "parseable_readonly":True,
    }


def _execution_status(database: Path, sql: str, max_rows: int) -> dict:
    if _parse_readonly_sql(sql) is None:
        return {"valid":False,"why":"unsafe_or_not_parseable"}
    try:
        result=execute_readonly(database,sql,max_rows=max_rows)
    except (SQLExecutionError,ValueError,TypeError):
        return {"valid":False,"why":"failed_sql_execution"}
    if result.truncated:
        return {"valid":False,"why":"truncated_execution"}
    # An EMPTY SQL result is not an error: empty answer may be semantically correct.
    return {"valid":True,"why":"read_only_execution_succeeded"}


def choose_semantic_sql(
    database: Path, *, question: str, baseline_sql: str, revised_sql: str,
    max_rows: int = 50_000,
) -> dict:
    """Gold-blind selector accepts NO ground truth, score, evaluation or labels.

    Select revision only if it is valid AND (baseline invalid OR strict
    improvement on an *explicit* question clause). Independent reviser alone
    is not enough reason to replace the M-Schema baseline.
    """
    if max_rows<1:
        raise ValueError("max_rows must be positive")
    baseline_status=_execution_status(database,baseline_sql,max_rows)
    revision_status=_execution_status(database,revised_sql,max_rows)
    baseline_contract=check_question_constraints(question,baseline_sql)
    revision_contract=check_question_constraints(question,revised_sql)
    use_revision=False
    reason="no_gold_blind_semantic_evidence"
    if revision_status["valid"]:
        if not baseline_status["valid"]:
            use_revision=True
            reason="baseline_non_executable"
        elif (
            baseline_contract["required"]
            and len(revision_contract["missing"]) < len(baseline_contract["missing"])
            and set(revision_contract["missing"]).issubset(set(baseline_contract["missing"]))
        ):
            use_revision=True
            reason="explicit_question_constraint_recovered"
    return {
        "chosen_arm":"semantic_revision" if use_revision else "mschema",
        "chosen_sql":revised_sql if use_revision else baseline_sql,
        "reason":reason,
        "baseline_valid":bool(baseline_status["valid"]),
        "revision_valid":bool(revision_status["valid"]),
        "baseline_violations":baseline_contract["missing"],
        "revised_violations":revision_contract["missing"],
        "chosen_violations":(
            revision_contract["missing"] if use_revision else baseline_contract["missing"]
        ),
        "semantic_requirements":baseline_contract["required"],
        "gold_used_for_selection":False,
    }


def revise_with_semantic_audit(
    *,provider:OllamaProvider,question:str,schema_context:str,
    evidence:str|None,baseline_sql:str,
) -> str:
    """One revision model call; schema and baseline are the only auxiliary inputs."""
    requirements=_required_intents(question)
    prompt=f"""You are a careful SQLite semantic auditor. You are given a question,
the physical M-Schema, provided domain evidence, and an INITIAL generated SQL.
Independently audit whether the SQL answers the QUESTION, not merely whether
it runs. Write exactly ONE final read-only SQLite SELECT query.

Review, in order:
- What is the exact requested output/granularity? COUNT vs SUM vs AVG vs
  value vs distinct entity. Do not swap aggregation operators.
- Which table and FK-justified joins identify those entities?
- Which filters are explicitly required, including numeric/date/range and
  entity values? Never add constraints not specified.
- Does the question ask for highest/lowest, most/fewest, top-N, ordering or
  tie behavior? Check ORDER BY direction and LIMIT.
- Does GROUP BY match the entity being reported, not an unrelated field?
- Compare the INITIAL SQL against these requirements; fix only a semantic
  mismatch you can identify. Otherwise return INITIAL SQL unchanged.
- Use only real physical columns shown below, no imaginary columns.

Question-intent hints from deterministic text rules (may be incomplete):
{json.dumps(requirements,ensure_ascii=False)}

QUESTION:
{question}

EXACT SQLITE M-SCHEMA:
{schema_context}

SUPPLIED DOMAIN EVIDENCE:
{evidence or ''}

INITIAL SQL:
{baseline_sql}

Return exactly one SQLite SQL SELECT query; NO markdown, JSON or commentary.
SQL:"""
    return _strip_fence(provider.complete_text(system="",user=prompt))


def run_shard(
    *,source:Path,database_root:Path,frozen:Path,
    model:str,shard_index:int,num_shards:int,
    count:int=100,
) -> dict:
    if num_shards!=4 or not 0<=shard_index<num_shards or count!=100:
        raise ValueError("experiment preregistered for exactly 4 shards x 25 cases")
    prior=json.loads(frozen.read_text(encoding="utf-8"))
    explored={str(r["case_id"]) for r in prior["cases"]}
    if len(explored)!=100:
        raise ValueError("historical frozen cohort must contain 100 unique IDs")
    # Ground truth labels are not accessed until AFTER both candidates are frozen.
    cases,labels=load_bird_json(source)
    ids=build_holdout_ids([str(c.case_id) for c in cases],explored,count=count)
    cohort_sha=sha256("\n".join(ids).encode()).hexdigest()
    selected=ids[shard_index::num_shards]
    assert len(selected)==25
    by_id={str(c.case_id):c for c in cases}
    labels_by_id={str(l.case_id):l for l in labels}
    provider=OllamaProvider(model=model,timeout_s=900.0)
    catalog_cache={}
    output=[]
    for idx,case_id in enumerate(selected,1):
        case=by_id[case_id]
        if case.database_id not in catalog_cache:
            path=resolve_database(database_root,case.database_id)
            catalog=DatabaseCatalog.from_sqlite(path)
            descriptions={(t,c):d for t,c,d in _description_rows(database_root,case.database_id)}
            catalog_cache[case.database_id]=(path,catalog,descriptions)
        db,catalog,descriptions=catalog_cache[case.database_id]
        _,lexical=select_schema_context(catalog,database_root,case.database_id,case.question)
        schema=build_mschema(
            catalog=catalog,database_id=case.database_id,
            selected_tables=lexical["selected_tables"],descriptions=descriptions,
        )
        started=perf_counter()
        mschema_sql=generate_xiyan(
            question=case.question,schema_context=schema,
            evidence=case.evidence.strip() or None,provider=provider,
        )
        base_latency=perf_counter()-started
        started=perf_counter()
        revised_sql=revise_with_semantic_audit(
            provider=provider,question=case.question,schema_context=schema,
            evidence=case.evidence.strip() or None,baseline_sql=mschema_sql,
        )
        review_latency=perf_counter()-started
        selection=choose_semantic_sql(
            db,question=case.question,baseline_sql=mschema_sql,
            revised_sql=revised_sql,
        )
        # SEPARATE post-selection offline scoring; never feed labels to
        # generator, auditor, schema selector or candidate choice.
        gold_sql=labels_by_id[case_id].sql
        base_ex=bool(score_query(db,mschema_sql,gold_sql)["frozen_execution_match"])
        revised_ex=bool(score_query(db,revised_sql,gold_sql)["frozen_execution_match"])
        chosen_ex=bool(score_query(db,selection["chosen_sql"],gold_sql)["frozen_execution_match"])
        output.append({
            "case_id":case_id,"database_id":case.database_id,
            "baseline_sql":mschema_sql,"revised_sql":revised_sql,
            "chosen_arm":selection["chosen_arm"],"reason":selection["reason"],
            "baseline_valid":selection["baseline_valid"],
            "revision_valid":selection["revision_valid"],
            "baseline_violations":selection["baseline_violations"],
            "revised_violations":selection["revised_violations"],
            "semantic_requirements":selection["semantic_requirements"],
            "changed":mschema_sql.strip()!=revised_sql.strip(),
            "baseline_ex":base_ex,"revision_ex":revised_ex,
            "selected_ex":chosen_ex,
            "baseline_latency_s":round(base_latency,3),
            "revision_latency_s":round(review_latency,3),
        })
        print(
            f"SEMANTIC_SHARD [{shard_index}:{idx}/25] "
            f"case={case_id} baseline_ex={base_ex} revised_ex={revised_ex} "
            f"selected_ex={chosen_ex} arm={selection['chosen_arm']} "
            f"why={selection['reason']}",flush=True,
        )
    return {
        "schema":"semantic-shard-independent-v1",
        "model":model,
        "cohort_sha256":cohort_sha,
        "cohort_selection_seed":SEED,
        "historical_frozen100_excluded":True,
        "num_shards":num_shards,
        "shard_index":shard_index,
        "count":count,
        "gold_visible_to_generation":False,
        "metric":"project-local read-only SQLite set-of-tuples execution match",
        "cases":output,
    }


def aggregate_semantic_shards(shards:list[dict],*,min_gain:int=3)->dict:
    if len(shards)!=4 or sorted(r.get("shard_index") for r in shards)!=list(range(4)):
        raise ValueError("require four distinct shards [0, 1, 2, 3]")
    if any(x.get("schema")!="semantic-shard-independent-v1" or
           x.get("gold_visible_to_generation") is not False for x in shards):
        raise ValueError("unexpected shard schema or gold exposure")
    models={str(x.get("model")) for x in shards}
    cohorts={str(x.get("cohort_sha256")) for x in shards}
    if len(models)!=1 or len(cohorts)!=1:
        raise ValueError("model or cohort mismatch")
    if min_gain<1:
        raise ValueError("meaningful positive gain gate required")
    all_rows=[row for shard in shards for row in shard.get("cases",[])]
    ids=[str(row["case_id"]) for row in all_rows]
    if len(all_rows)!=100 or len(set(ids))!=100:
        raise ValueError("need exactly 100 unique independent cases")
    baseline=sum(bool(r["baseline_ex"]) for r in all_rows)
    revision=sum(bool(r["revision_ex"]) for r in all_rows)
    selected=sum(bool(r["selected_ex"]) for r in all_rows)
    oracle=sum(bool(r["baseline_ex"] or r["revision_ex"]) for r in all_rows)
    gains=sum(not r["baseline_ex"] and r["selected_ex"] for r in all_rows)
    losses=sum(r["baseline_ex"] and not r["selected_ex"] for r in all_rows)
    if gains-losses!=selected-baseline:
        raise ValueError("paired label arithmetic inconsistency")
    return {
        "schema":"semantic-shard-independent-aggregate-v1",
        "sample_size":100,"model":next(iter(models)),
        "cohort_sha256":next(iter(cohorts)),
        "baseline_correct":baseline,
        "revision_correct":revision,
        "selected_correct":selected,
        "net_gain":selected-baseline,
        "wrong_to_correct":gains,
        "correct_to_wrong":losses,
        "oracle_correct":oracle,
        "oracle_headroom":oracle-baseline,
        "unrealized_oracle_gap":oracle-selected,
        "changed":sum(bool(r["changed"]) for r in all_rows),
        "selection_counts":dict(Counter(str(r["chosen_arm"]) for r in all_rows)),
        "selection_reasons":dict(Counter(str(r["reason"]) for r in all_rows)),
        "min_gain":min_gain,
        "gate_met":selected-baseline>=min_gain,
        "metric":"project-local read-only SQLite set-of-tuples execution match",
        "limitations":"Independent BIRD DEV cohort from same public BIRD release, not official BIRD hidden-test leaderboard; study-selection and original dataset bias remain.",
        "cases":sorted(all_rows,key=lambda r:int(r["case_id"])),
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--database-root",type=Path)
    parser.add_argument("--frozen",type=Path)
    parser.add_argument("--model",default=MODEL)
    parser.add_argument("--shard-index",type=int)
    parser.add_argument("--num-shards",type=int,default=4)
    parser.add_argument("--count",type=int,default=100)
    parser.add_argument("--inputs",type=Path,nargs="+")
    parser.add_argument("--min-gain",type=int,default=3)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite previously reported shard results")
    if args.inputs:
        result=aggregate_semantic_shards([
            json.loads(p.read_text(encoding="utf-8")) for p in args.inputs
        ],min_gain=args.min_gain)
        print("SEMANTIC_AGGREGATE",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if any(x is None for x in (args.source,args.database_root,args.frozen,args.shard_index)):
            parser.error("shard requires source, database-root, frozen, shard-index")
        result=run_shard(source=args.source,database_root=args.database_root,
                         frozen=args.frozen,model=args.model,shard_index=args.shard_index,
                         num_shards=args.num_shards,count=args.count)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__=="__main__":
    main()
