"""Independent BIRD DEV generation ablation: FK-bridge + answer-grain SQL.

Frozen A0: XiYanSQL-7B with original selected M-Schema.
A1: same model/questions/evidence, independent directly generated SQL using
only physically verified FK bridge tables and explicit question-grain advice.
No A5 free-text plan or baseline SQL enters A1. Evaluate candidate oracle
headroom, not post-hoc selector. Question-disjoint from three earlier
exploratory BIRD DEV100 cohorts, but still a public benchmark and possibly
database-overlapping. Gold SQL is scored only after a whole shard is generated.
"""
from __future__ import annotations

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from evaluation.bird_loader import load_bird_json
from evaluation.schema_context import _description_rows, select_schema_context
from experiments.xiyan_dgsl_frozen100 import generate_xiyan, resolve_database, score_query
from experiments.xiyan_mschema_frozen100 import build_mschema
from experiments.xiyan_semantic_shards import build_holdout_ids, MODEL
from experiments.semantic_layer_abc import choose_disjoint_ids
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.providers import OllamaProvider

SEED = "bird-fk-bridge-independent-candidates-20261113-v1"
SECOND_SHA = "54d1cdf613d4db736a45cfd57b8659b2fa6b468a709e3ad4ef77be4d57bbc266"
THIRD_SHA = "3676427aa9d480e830f9632fd00e2c3e84032f103234c3d3ed9d226f2dea65ed"


def select_new100(all_ids: list[str], frozen_ids: set[str]) -> list[str]:
    if len(frozen_ids) != 100:
        raise ValueError("frozen100 required")
    second = build_holdout_ids(all_ids, frozen_ids, count=100)
    third = choose_disjoint_ids(all_ids, frozen_ids | set(second), count=100)
    if len(frozen_ids | set(second) | set(third)) != 300:
        raise ValueError("overlapping old cohorts")
    # Pin previously published SHA-selected cohorts, without looking at labels.
    if sha256("\n".join(second).encode()).hexdigest() != SECOND_SHA:
        raise ValueError("second cohort diverged from frozen run")
    if sha256("\n".join(third).encode()).hexdigest() != THIRD_SHA:
        raise ValueError("third cohort diverged from frozen run")
    return choose_disjoint_ids(all_ids, frozen_ids | set(second) | set(third),
                               count=100, seed=SEED)


def expand_fk_bridges(selected: list[str], foreign_keys: list[tuple],
                      *, max_added: int = 2) -> list[str]:
    """Add shortest intermediate join tables when physical FK declarations exist."""
    if len(set(selected)) != len(selected) or max_added < 0:
        raise ValueError("duplicate anchors / bad budget")
    graph: dict[str, set[str]] = {}
    for src, _, dst, _ in foreign_keys:
        if src == dst:
            continue
        graph.setdefault(src, set()).add(dst)
        graph.setdefault(dst, set()).add(src)
    added: list[str] = []
    for i, src in enumerate(selected):
        for dst in selected[i+1:]:
            queue = deque([(src, (src,))])
            visited = {src}
            shortest = None
            while queue:
                node, path = queue.popleft()
                if node == dst:
                    shortest = path
                    break
                if len(path) > max_added + 1:
                    continue
                for nxt in sorted(graph.get(node, ())):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, path+(nxt,)))
            if shortest is None:
                continue
            need = [node for node in shortest[1:-1]
                    if node not in selected and node not in added]
            if len(need) + len(added) <= max_added:
                added.extend(need)
    return selected + added


def generate_question_grounded(provider, *, question: str,
                               evidence: str, physical_schema: str) -> str:
    # Direct independent generation: do not make a separate A5 plan or trust
    # a guessed intermediate intention that can corrupt a correct baseline.
    prompt = f"""Write exactly one safe read-only SQLite SELECT query from the original question.
Use only declared physical tables and columns shown below. A foreign key
can justify a join path, but does not prove actual referential integrity or
row uniqueness. Derive the answer entity and grouping grain from the wording;
do not count fan-out join rows when unique source entities are requested.
Respect directional relations (A to B is not B to A), requested output
columns, dates, comparison operators, explicit values, ranking and ties.
Never guess an absent predicate, identifier, or synonym.
Original question:
{question}
Physical SQLite M-Schema and verified FK bridge paths:
{physical_schema}
User-provided evidence (no reference answers):
{evidence}
Return one SQL SELECT statement ONLY; no explanation or markdown.
SQL:"""
    return _strip_fence(provider.complete_text(system="", user=prompt))


def run_shard(*, source: Path, root: Path, frozen: Path,
              shard: int, model: str = MODEL) -> dict:
    if shard not in range(4):
        raise ValueError("4x25 shard protocol")
    old = json.loads(frozen.read_text(encoding="utf-8"))
    frozen_ids = {str(x["case_id"]) for x in old["cases"]}
    cases, labels = load_bird_json(source)
    ids = select_new100([str(x.case_id) for x in cases], frozen_ids)
    cohort_sha = sha256("\n".join(ids).encode()).hexdigest()
    chosen = ids[shard::4]
    if len(chosen) != 25:
        raise ValueError("25 cases per shard required")
    by_id = {str(x.case_id):x for x in cases}
    provider = OllamaProvider(model=model, timeout_s=900)
    cache = {}
    generated = []
    for n, case_id in enumerate(chosen, 1):
        case = by_id[case_id]
        if case.database_id not in cache:
            db = resolve_database(root, case.database_id)
            catalog = DatabaseCatalog.from_sqlite(db)
            desc = {(t,c):d for t,c,d in _description_rows(root,case.database_id)}
            cache[case.database_id] = (db,catalog,desc)
        db,catalog,descs = cache[case.database_id]
        _, selection = select_schema_context(catalog,root,case.database_id,case.question)
        selected = selection["selected_tables"]
        bridged = expand_fk_bridges(selected, catalog.foreign_keys)
        if not set(bridged).issubset(catalog.tables):
            raise ValueError("nonexistent inferred bridge table")
        a0_schema = build_mschema(catalog=catalog,database_id=case.database_id,
                                  selected_tables=selected,descriptions=descs)
        a1_schema = build_mschema(catalog=catalog,database_id=case.database_id,
                                  selected_tables=bridged,descriptions=descs)
        start=perf_counter()
        a0 = generate_xiyan(question=case.question,schema_context=a0_schema,
                            evidence=case.evidence.strip() or None,provider=provider)
        second=perf_counter()
        a1 = generate_question_grounded(provider,question=case.question,
                                       physical_schema=a1_schema,
                                       evidence=case.evidence.strip())
        generated.append({
            "case_id":case_id,"database_id":case.database_id,
            "sql":{"a0":a0,"a1":a1},
            "new_bridge_tables":bridged[len(selected):],
            "changed":a0.strip()!=a1.strip(),
            "latency_a0_s":round(second-start,2),
            "latency_a1_s":round(perf_counter()-second,2),
        })
        print(f"BRIDGE_GENERATED shard={shard} {n}/25",flush=True)
    # Do not inspect any gold label until all 25 cases have two frozen SQLs.
    gold={str(x.case_id):x.sql for x in labels}
    scored=[]
    for row in generated:
        db=cache[row["database_id"]][0]
        scores={arm:bool(score_query(db,sql,gold[row["case_id"]])["frozen_execution_match"])
                for arm,sql in row["sql"].items()}
        scored.append({**row,"ex":scores})
    return {"schema":"bird-fk-bridge-shard-v1","shard":shard,
            "model":model,"cohort_sha256":cohort_sha,
            "gold_used_for_generation":False,"cases":scored}


def aggregate(shards: list[dict]) -> dict:
    if len(shards)!=4 or sorted(x.get("shard") for x in shards)!=list(range(4)):
        raise ValueError("4 distinct shards")
    if any(x.get("schema")!="bird-fk-bridge-shard-v1" or
           x.get("gold_used_for_generation") is not False for x in shards):
        raise ValueError("shard policy mismatch")
    if len({x["cohort_sha256"] for x in shards})!=1 or len({x["model"] for x in shards})!=1:
        raise ValueError("model/cohort drift")
    rows=[r for x in shards for r in x["cases"]]
    if len(rows)!=100 or len({x["case_id"] for x in rows})!=100:
        raise ValueError("100 unique questions required")
    a0=sum(r["ex"]["a0"] for r in rows)
    a1=sum(r["ex"]["a1"] for r in rows)
    gains=sum(not r["ex"]["a0"] and r["ex"]["a1"] for r in rows)
    losses=sum(r["ex"]["a0"] and not r["ex"]["a1"] for r in rows)
    if a1-a0!=gains-losses:
        raise AssertionError("paired arithmetic")
    return {"schema":"bird-fk-bridge-aggregate-v1",
            "count":100,"model":shards[0]["model"],
            "cohort_sha256":shards[0]["cohort_sha256"],
            "baseline_correct":a0,"question_grounded_correct":a1,
            "wrong_to_correct":gains,"correct_to_wrong":losses,
            "candidate_oracle_correct":a0+gains,
            "candidate_oracle_headroom":gains,
            "changed":sum(r["changed"] for r in rows),
            "bridge_used":sum(bool(r["new_bridge_tables"]) for r in rows),
            "promote":False,
            "caveat":"Question-disjoint BIRD DEV, databases may overlap; public development data and project-local set-of-tuples EX; no selector tested.",
            "cases":sorted(rows,key=lambda r:int(r["case_id"]))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--database-root",type=Path)
    p.add_argument("--frozen",type=Path)
    p.add_argument("--shard-index",type=int)
    p.add_argument("--model",default=MODEL)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwriting a result")
    if a.inputs:
        result=aggregate([json.loads(x.read_text()) for x in a.inputs])
        print("BIRD_BRIDGE_FINAL",json.dumps({k:v for k,v in result.items() if k!="cases"},
                                             sort_keys=True),flush=True)
    else:
        if any(x is None for x in (a.source,a.database_root,a.frozen,a.shard_index)):
            p.error("missing shard args")
        result=run_shard(source=a.source,root=a.database_root,
                         frozen=a.frozen,shard=a.shard_index,model=a.model)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    main()
