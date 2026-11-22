"""Real physical Spider 1.0 paired proof: Qwen3.5-9B base vs joint repair.

Independent source family from the BIRD DEV tuning loop, NOT a secret
benchmark, NOT guaranteed absent from Qwen pretraining. Freeze selects questions
without gold SQL. Inference never reads Spider labels; scoring happens only
after all inference shards are sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import time
import zipfile

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.joint_candidate_selector import JointObligation, select_joint_repair
from semantic_sql.providers import OllamaProvider
from semantic_sql.sql_semantics import ColumnRef, NormalizedJoin
from semantic_sql.execution import SQLExecutionError, execute_readonly

MODEL = "qwen3.5:9b-q4_K_M"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
ARCHIVE_REV = "4a01bbac6520cd35b216db9e1724e5e1ada60aa4"
PROTOCOL = "SPIDER-V1-12-DB-JOINT-REPAIR-PHYSICAL-PAIRED-V1"
SEED = "20261122-source-only-spider-12-db-no-candidate-or-label-selection"
N = 12
SHARDS = 3


def _hash(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()


def _rows(archive: Path) -> list[dict]:
    with zipfile.ZipFile(archive) as z:
        paths = [n for n in z.namelist() if n.endswith("/dev.json")]
        if len(paths) != 1:
            raise ValueError("Spider dev.json missing or ambiguous")
        rows = json.loads(z.read(paths[0]))
    if not isinstance(rows, list):
        raise ValueError("Spider source rows malformed")
    return rows


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)


def freeze_source(archive: Path) -> dict:
    """Gold-blind count-question cohort: one frozen question per physical DB."""
    rows = _rows(archive)
    eligible: dict[str, list[dict]] = {}
    for i, row in enumerate(rows):
        db, q = row.get("db_id"), row.get("question")
        if not isinstance(db, str) or not isinstance(q, str):
            continue
        if not re.search(r"\b(how many|number of|count of)\b", q.casefold()):
            continue
        eligible.setdefault(db, []).append({
            "source_index": i, "db_id": db, "question": q,
            "question_sha256": _hash(q),
        })
    if len(eligible) < N:
        raise ValueError(f"insufficient distinct physical Spider DBs: {len(eligible)}")
    chosen_db = sorted(eligible, key=lambda d: _hash(SEED + ":db:" + d))[:N]
    selected = [
        min(eligible[d], key=lambda c:_hash(SEED+":q:"+str(c["source_index"])))
        for d in chosen_db
    ]
    if len(selected)!=N or len({x["db_id"] for x in selected})!=N:
        raise ValueError("source DB cohort not independent")
    return {
        "protocol":PROTOCOL, "source":"Spider 1.0 original public validation",
        "archive_revision":ARCHIVE_REV,
        "archive_sha256":sha256(archive.read_bytes()).hexdigest(),
        "source_count":len(rows), "count_eligible_db_count":len(eligible),
        "sample_size":N, "db_count":N, "cases":selected,
        "gold_exposed_to_inference":False,
        "selection_uses_question_and_db_only":True,
        "selection_seed":SEED,
        "public_dataset_pretraining_exposure_unknown":True,
    }


def _db_path(root: Path, dbid: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_]+", dbid):
        raise ValueError("unsafe database identifier")
    matches = [p for p in root.rglob(dbid + ".sqlite")
               if p.parent.name == dbid]
    if len(matches)!=1:
        raise ValueError("physical database missing or ambiguous: "+dbid)
    return matches[0]


def _schema_fk(annotations: list[dict], dbid: str) -> set[frozenset[ColumnRef]]:
    matches = [d for d in annotations if d.get("db_id")==dbid]
    if len(matches)!=1:
        raise ValueError("Spider FK annotation missing or ambiguous")
    obj=matches[0]
    tables=obj["table_names_original"]
    cols=obj["column_names_original"]
    edges=set()
    for i,j in obj["foreign_keys"]:
        ti, ci = cols[i]
        tj, cj = cols[j]
        if ti < 0 or tj < 0:continue
        edges.add(frozenset((
            ColumnRef(tables[ti], ci),
            ColumnRef(tables[tj], cj)
        )))
    return edges


def _ref(s: str, catalog:DatabaseCatalog) -> ColumnRef | None:
    if not isinstance(s,str) or s.count(".")!=1:
        return None
    t,c=s.split(".")
    actual=next((v for k,v in catalog.tables.items() if k.casefold()==t.casefold()),None)
    if actual is None or c.casefold() not in {v.casefold() for v in actual.columns}:
        return None
    return ColumnRef(actual.name,c)


def _obligation(text: str, catalog:DatabaseCatalog,
                declared_edges:set[frozenset[ColumnRef]]) -> JointObligation | None:
    try:
        raw=json.loads(text.strip())
    except (ValueError,TypeError):
        return None
    if not isinstance(raw,dict) or raw.get("confidence")!="high" or type(raw.get("distinct")) is not bool:
        return None
    join=raw.get("join_edge")
    if not isinstance(join,list) or len(join)!=2:return None
    refs=[_ref(raw.get("entity_column"),catalog),
          _ref(join[0],catalog),_ref(join[1],catalog),
          _ref(raw.get("count_grain"),catalog)]
    if any(x is None for x in refs) or frozenset(refs[1:3]) not in declared_edges:
        return None
    if refs[1]==refs[2] or refs[0]==refs[3]:
        return None
    return JointObligation(entity_column=refs[0],
                           join_edge=NormalizedJoin(refs[1],refs[2]),
                           count_grain=refs[3],distinct=raw["distinct"],
                           confidence="high")


CONTRACT_SYSTEM = """Question/schema-only typed SQL intent for a real SQLite database.
Independently identify an explicit requested output entity column, exact
schema-annotated FK relationship (both physical join columns), COUNT entity
grain and whether DISTINCT is required. Never guess. If not all four are
unambiguously entailed by the question/schema, return confidence=low.
JSON only:
{"confidence":"high|low","entity_column":"table.column",
"join_edge":["table.column","table.column"],"count_grain":"table.column",
"distinct":true}
Never write SQL and do not inspect candidate SQL or any reference answer."""

REPAIR_SYSTEM = """Repair only a concrete output-entity, FK join-edge or COUNT grain
contradiction in this SQLite SELECT. Keep source tables, WHERE, HAVING, GROUP BY,
ORDER BY and LIMIT exactly unchanged. Reply ONE read-only SQLite SQL query;
no prose, explanations or markdown. Abstain by returning original SQL if unsure."""


def _provider():
    from evaluation.run_bird import CountingProvider
    return CountingProvider(OllamaProvider(
        model=MODEL,temperature=0.0,seed=42,num_predict=768,timeout_s=600,
    ))


def _model_digest() -> str:
    import urllib.request
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=30) as resp:
        d=json.load(resp)
    matches=[m["digest"] for m in d["models"]
             if m.get("name")==MODEL or m.get("model")==MODEL]
    if len(matches)!=1:return ""
    return matches[0]


def infer_one(case:dict, db:Path, annotations:list[dict]) ->dict:
    from semantic_sql.inference import run_guarded
    from semantic_sql.verifier import verify_sql_preflight

    catalog=DatabaseCatalog.from_sqlite(db)
    edges=_schema_fk(annotations,case["db_id"])
    fk_context="\n".join("JOIN "+(" = ".join(sorted(
        f"{r.table}.{r.column}" for r in e))) for e in edges)
    schema=catalog.schema_text()+"\n"+fk_context
    q=case["question"]
    start=time.perf_counter()
    # Question-only intent must be frozen BEFORE any baseline SQL is seen.
    treatment=_provider()
    try:
        extracted=treatment.complete_text(
            system=CONTRACT_SYSTEM,
            user=f"Question:\n{q}\n\nPhysical schema and annotated FK:\n{schema}",
        )
        obligation=_obligation(extracted,catalog,edges)
    except Exception:
        obligation=None
    baseline=_provider()
    base_sql=""
    base_status="generation_failed"
    try:
        out=run_guarded(database=db,provider=baseline,question=q,
                        evidence=None,schema_context=schema,
                        max_repairs=2,max_rows=50000)
        base_sql=out.candidate.sql if out.candidate else ""
        base_status=out.status
    except Exception:
        pass
    chosen=base_sql
    status="ABSTAIN"
    reason="no_independent_high_confidence_contract"
    if obligation and base_sql:
        try:
            # Do not call Qwen if the original already satisfies the joint check.
            original=select_joint_repair(base_sql,[],obligation)
            if original.status=="KEEP_ORIGINAL":
                status="KEEP_ORIGINAL"
                reason="already_joint_consistent"
            elif original.reason=="zero_or_multiple_joint_compatible_candidates":
                candidate=_strip_fence(treatment.complete_text(
                    system=REPAIR_SYSTEM,
                    user=f"Question:\n{q}\n\nSchema:\n{schema}\n\n"
                         f"Required output: {obligation.entity_column}\n"
                         f"Required join: {obligation.join_edge}\n"
                         f"COUNT grain: {obligation.count_grain} DISTINCT={obligation.distinct}\n"
                         f"Original SQL:\n{base_sql}",
                ))
                decision=select_joint_repair(base_sql,[candidate],obligation)
                if (decision.status=="SELECT_REPAIR"
                    and verify_sql_preflight(db,decision.sql).ok):
                    chosen=decision.sql
                    status="SELECT_REPAIR"
                    reason=decision.reason
                else:
                    reason="repair_not_unique_and_executable"
            else:
                reason=original.reason
        except Exception:
            reason="repair_execution_or_parsing_failed"
    return {
        "source_index":case["source_index"],"db_id":case["db_id"],
        "question_sha256":case["question_sha256"],
        "baseline_sql":base_sql,"treated_sql":chosen,
        "baseline_status":base_status,
        "decision":status,"decision_reason":reason,
        "contract_grounded":obligation is not None,
        "baseline_model_calls":baseline.total_calls,
        "incremental_model_calls":treatment.total_calls,
        "elapsed_s":round(time.perf_counter()-start,3),
    }


def strict_execution_compare(db:Path,predicted:str,reference:str)->tuple[bool,bool] | None:
    try:
        gold=execute_readonly(db,reference,max_rows=50000)
        if gold.truncated:return None
    except (SQLExecutionError,ValueError,TypeError):
        return None
    try:
        observed=execute_readonly(db,predicted,max_rows=50000) if predicted else None
        if observed is None or observed.truncated:return (False,False)
        return set(observed.rows)==set(gold.rows), Counter(observed.rows)==Counter(gold.rows)
    except (SQLExecutionError,ValueError,TypeError):
        return (False,False)


def do_infer(frozen:Path,root:Path,annotations:Path,shard:int,out:Path)->dict:
    f=json.loads(frozen.read_text())
    if f.get("protocol")!=PROTOCOL or f.get("sample_size")!=N or f.get("gold_exposed_to_inference") is not False:
        raise ValueError("not the frozen gold-free source")
    if shard not in range(SHARDS) or _model_digest()!=MODEL_DIGEST:
        raise ValueError("wrong shard or Qwen3.5-9B digest")
    anno=json.loads(annotations.read_text())
    cases=[c for i,c in enumerate(f["cases"]) if i%SHARDS==shard]
    rows=[infer_one(c,_db_path(root,c["db_id"]),anno) for c in cases]
    obj={
        "protocol":PROTOCOL,"status":"GOLD_BLIND_COMPLETE",
        "shard":shard,"model_digest":MODEL_DIGEST,
        "archive_sha256":f["archive_sha256"],
        "reference_sql_accessed":False,"cases":rows,
    }
    _write(out,obj)
    print("SPIDER_PHYSICAL_QWEN_BLIND_SHARD",shard,"cases",len(rows),
          "selected_repairs",sum(c["decision"]=="SELECT_REPAIR" for c in rows),
          "grounded",sum(c["contract_grounded"] for c in rows),flush=True)
    return obj


def do_score(frozen:Path,archive:Path,root:Path,shard_files:list[Path],out:Path)->dict:
    f=json.loads(frozen.read_text())
    if f["protocol"]!=PROTOCOL or sha256(archive.read_bytes()).hexdigest()!=f["archive_sha256"]:
        raise ValueError("source drift")
    if len(shard_files)!=SHARDS:
        raise ValueError("incomplete blinded shards")
    blind=[json.loads(p.read_text()) for p in shard_files]
    for i,obj in enumerate(blind):
        if (obj.get("protocol")!=PROTOCOL or obj.get("status")!="GOLD_BLIND_COMPLETE"
            or obj.get("shard")!=i or obj.get("reference_sql_accessed") is not False
            or obj.get("model_digest")!=MODEL_DIGEST
            or obj.get("archive_sha256")!=f["archive_sha256"]):
            raise ValueError("unsealed, mismatched or incomplete inference")
    combined={c["source_index"]:c for obj in blind for c in obj["cases"]}
    if len(combined)!=N or set(combined)!={c["source_index"] for c in f["cases"]}:
        raise ValueError("missing or duplicated cases")
    source=_rows(archive)
    result=[]
    for c in f["cases"]:
        item=combined[c["source_index"]]
        original=source[c["source_index"]]
        if (c["db_id"]!=original["db_id"]
            or c["question"]!=original["question"]
            or item["question_sha256"]!=c["question_sha256"]
            or item["db_id"]!=c["db_id"]):
            raise ValueError("source and blind SQL mismatch")
        db=_db_path(root,c["db_id"])
        gold=original["query"]
        base=strict_execution_compare(db,item["baseline_sql"],gold)
        treat=strict_execution_compare(db,item["treated_sql"],gold)
        result.append({
            "db_id":c["db_id"],"source_index":c["source_index"],
            "scorable":base is not None and treat is not None,
            "baseline":base,"treatment":treat,
            "selected":item["decision"]=="SELECT_REPAIR",
            "grounded":item["contract_grounded"],
            "baseline_calls":item["baseline_model_calls"],
            "incremental_calls":item["incremental_model_calls"],
        })
    sc=[x for x in result if x["scorable"]]
    summary={
        "protocol":PROTOCOL, "status":"FINISHED_PHYSICAL_SET_BAG_EX",
        "n_selected":N,"n_scorable":len(sc),
        "unique_db_count":len({x["db_id"] for x in result}),
        "baseline_set":sum(x["baseline"][0] for x in sc),
        "treatment_set":sum(x["treatment"][0] for x in sc),
        "baseline_bag":sum(x["baseline"][1] for x in sc),
        "treatment_bag":sum(x["treatment"][1] for x in sc),
        "rescues":sum(not x["baseline"][0] and x["treatment"][0] for x in sc),
        "regressions":sum(x["baseline"][0] and not x["treatment"][0] for x in sc),
        "eligible_contracts":sum(x["grounded"] for x in result),
        "actual_selected_repairs":sum(x["selected"] for x in result),
        "total_baseline_calls":sum(x["baseline_calls"] for x in result),
        "total_incremental_calls":sum(x["incremental_calls"] for x in result),
        "model_digest":MODEL_DIGEST,
        "source_sha256":f["archive_sha256"],
        "source_shift":"Spider validation; separate physical database families from BIRD DEV",
        "caveats":"Public Spider not guaranteed unseen in foundation-model pretraining; custom physical SET/BAG matching, not official leaderboard",
        "cases":result,
    }
    _write(out,summary)
    print("SPIDER_PAIRED_PHYSICAL_STRICT",json.dumps({k:v for k,v in summary.items() if k!="cases"}),flush=True)
    return summary


def main() -> None:
    p=argparse.ArgumentParser()
    modes=p.add_subparsers(dest="action",required=True)
    fr=modes.add_parser("freeze");fr.add_argument("--archive",type=Path,required=True);fr.add_argument("--out",type=Path,required=True)
    inf=modes.add_parser("infer");inf.add_argument("--frozen",type=Path,required=True);inf.add_argument("--db-root",type=Path,required=True);inf.add_argument("--annotations",type=Path,required=True);inf.add_argument("--shard",type=int,required=True);inf.add_argument("--out",type=Path,required=True)
    sc=modes.add_parser("score");sc.add_argument("--frozen",type=Path,required=True);sc.add_argument("--archive",type=Path,required=True);sc.add_argument("--db-root",type=Path,required=True);sc.add_argument("--shards",type=Path,nargs=SHARDS,required=True);sc.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.action=="freeze":
        obj=freeze_source(a.archive);_write(a.out,obj)
        print("SPIDER_FROZEN_BEFORE_QWEN",json.dumps({k:v for k,v in obj.items() if k!="cases"}))
    elif a.action=="infer":
        do_infer(a.frozen,a.db_root,a.annotations,a.shard,a.out)
    else:
        do_score(a.frozen,a.archive,a.db_root,a.shards,a.out)


if __name__=="__main__":
    main()
