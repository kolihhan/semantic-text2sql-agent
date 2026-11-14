"""CPU-only BIRD DEV source gate for real entity column grounding; no model/gold.

Pre-register a question-disjoint seeded sample of quoted-entity questions, query
only the physical SQLite databases, and measure exact positive entity-location
evidence **outside the original lexical schema**. Do not infer absent evidence
from negative or interrupted scans. A value match alone does not prove user
intent, but it anchors a candidate table/column for future Qwen generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evaluation.run_bird import _resolve_database
from evaluation.schema_context import select_schema_context
from evaluation.source_first_value_anchor import _literals, observed_value_anchors
from semantic_sql.catalog import DatabaseCatalog

SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
SEED = "qwen-source-first-entity-gate-v1-20261114"
PREVIOUS_16 = {
    "632", "706", "1477", "629", "635", "1140", "1097", "1112",
    "1156", "195", "721", "1422", "391", "155", "821", "851",
}
PREVIOUS_24 = {
    "1465", "576", "1348", "539", "156", "525", "1469", "1413",
    "113", "628", "499", "1324", "1000", "981", "307", "769",
    "244", "1174", "489", "141", "1186", "748", "529", "954",
}
PREVIOUS_NATIVE16 = {
    "317","29","827","1205","1253","749","1349","871",
    "1018","498","197","176","1201","781","609","870",
}


def audit(source: Path, root: Path, frozen: Path, n: int = 40) -> dict:
    if sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("BIRD DEV pin mismatch")
    data = json.loads(source.read_text(encoding="utf-8"))
    if len(data) != 1534:
        raise ValueError("unexpected BIRD size")
    old = json.loads(frozen.read_text(encoding="utf-8"))
    if old.get("sample_size") != 100 or old.get("status") != "complete":
        raise ValueError("unknown frozen 100")
    excluded = {str(v["case_id"]) for v in old["cases"]}
    excluded |= PREVIOUS_16 | PREVIOUS_24 | PREVIOUS_NATIVE16
    if len(excluded) != 156:
        raise ValueError(f"prior cohort overlap/drift {len(excluded)}")
    chosen = []
    counts = Counter()
    cached = {}
    eligible = sorted(
        ((str(r.get("question_id") or r.get("id") or i),r)
         for i,r in enumerate(data) if (str(r.get("question_id") or r.get("id") or i) not in excluded
                                      and _literals(str(r["question"])))),
        key=lambda pair:(sha256(f"{SEED}:{pair[0]}".encode()).hexdigest(), pair[0]),
    )
    for cid,row in eligible:
        dbid = str(row["db_id"])
        if counts[dbid] >= 3:
            continue
        chosen.append((cid,row))
        counts[dbid] += 1
        if len(chosen)==n:
            break
    # Record the attainable denominator rather than forcing a 40-case claim
    # when the pinned public source has fewer quote-entity cases after caps.
    if not chosen:
        raise ValueError("no eligible quoted-entity cases under the frozen exclusions")
    records = []
    for i,(cid,row) in enumerate(chosen):
        dbid,question=str(row["db_id"]),str(row["question"])
        if dbid not in cached:
            db = _resolve_database(root,dbid)
            cached[dbid]=(db,DatabaseCatalog.from_sqlite(db))
        path,catalog=cached[dbid]
        _context,info=select_schema_context(catalog,root,dbid,question)
        probe=observed_value_anchors(
            path,catalog,question,
            selected_tables=set(info["selected_tables"]),
            max_tables=40,max_probes=100,
        )
        records.append({
            "case_id":cid,"db_id":dbid,"question":question,
            "schema_sha256":info["schema_context_sha256"],
            "literals":list(_literals(question)),
            "selected_tables":info["selected_tables"],
            "hits":[{"table":v.table,"column":v.column,"value":v.value} for v in probe.hits],
            "outside_schema":list(probe.outside_schema),
            "ambiguous":probe.ambiguous,
            "probe_count":probe.probes,
            "probe_interrupted":probe.interrupted,
        })
        print(f"SOURCE_FIRST_ENTITY_CASE {i+1}/{n} id={cid} "
              f"positive_hits={len(probe.hits)} "
              f"outside_selected={len(probe.outside_schema)}",flush=True)
    report = {
        "protocol":"source-first-entity-positive-observation-v1",
        "source_sha256":SOURCE_SHA,
        "seed":SEED,"sample_size":len(chosen),"requested_sample_size":n,
        "prior_case_ids_excluded":len(excluded),
        "db_count":len(counts),
        "query_and_database_only":True,"no_model_inference":True,"no_gold_used":True,
        "positive_value_hit_questions":sum(bool(r["hits"]) for r in records),
        "outside_lexical_schema_questions":sum(bool(r["outside_schema"]) for r in records),
        "ambiguous_question_hits":sum(r["ambiguous"] for r in records),
        "total_probes":sum(r["probe_count"] for r in records),
        "interrupted_probes":sum(r["probe_interrupted"] for r in records),
        "cohort_sha256":sha256(json.dumps(
            [(r["case_id"],r["db_id"],r["schema_sha256"]) for r in records],
            separators=(",",":"),
        ).encode()).hexdigest(),
        "cases":records,
        "caveat":"Sample of quotes only; many real entity names unquoted. Probe is capped "
                 "and positive-only, not a claim of all missing anchors. "
                 "No reference SQL or correctness measurement.",
    }
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--database-root",type=Path,required=True)
    p.add_argument("--frozen",type=Path,default=Path("runs/bird-paired-dev-v5/paired.json"))
    p.add_argument("--n",type=int,default=40)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwrite source gate")
    report=audit(a.source,a.database_root,a.frozen,n=a.n)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("SOURCE_FIRST_ENTITY_FINAL",json.dumps({
        k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
