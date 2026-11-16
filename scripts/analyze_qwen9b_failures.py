#!/usr/bin/env python3
"""Post-hoc SQL difference screen. These are NOT human causal error labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
RAW_SHA="ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
HIST_SHA="0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"

def features(sql):
    clean=re.sub(r"'(?:''|[^'])*'", "'?'", sql or "").casefold()
    strings=sorted(set(v.strip().casefold() for v in re.findall(r"'((?:''|[^'])*)'", sql or "") if v.strip()))
    return {
      "tables": sorted(set(re.findall(r"\b(?:from|join)\s+([a-z_][a-z_0-9.]*)",clean))),
      "join_count":len(re.findall(r"\bjoin\b",clean)),
      "aggregates": sorted(set(re.findall(r"\b(?:count|sum|avg|min|max)\s*\(",clean))),
      "group":bool(re.search(r"\bgroup\s+by\b",clean)),
      "having":bool(re.search(r"\bhaving\b",clean)),
      "set_ops":sorted(set(re.findall(r"\b(?:union|intersect|except)\b",clean))),
      "strings":strings,
      "select_commas":projection_commas(clean),
    }

def projection_commas(clean):
    m=re.search(r"\bselect\s+(.*?)\s+\bfrom\b",clean,flags=re.S)
    if not m: return None
    depth=0; commas=0
    for ch in m.group(1):
        if ch=="(":depth+=1
        elif ch==")":depth=max(0,depth-1)
        elif ch=="," and depth==0:commas+=1
    return commas

def differences(pred,gold):
    p,g=features(pred),features(gold)
    return {
      "schema_linking": p["tables"]!=g["tables"],
      "value_matching": p["strings"]!=g["strings"],
      "evidence_or_external_knowledge":None,
      "join_or_aggregation":any(p[k]!=g[k] for k in ("join_count","aggregates","group","having","set_ops")),
      "result_shape":p["select_commas"]!=g["select_commas"] if p["select_commas"] is not None and g["select_commas"] is not None else None,
      "execution_failure":False,
    }

def analyze(source,shards):
    source_path=Path(source)
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=RAW_SHA:
        raise ValueError("pinned BIRD DEV source SHA mismatch")
    data=json.loads(source_path.read_text(encoding="utf-8"))
    if len(data)!=1534:raise ValueError("source must contain 1534 rows")
    index={str(r.get("question_id") or r.get("id") or i):r for i,r in enumerate(data)}
    cases=[];provenance={};invalid=[]
    for path in shards:
        path=Path(path)
        shard=json.loads(path.read_text(encoding="utf-8"))
        if shard.get("model_digest")!=DIGEST or shard.get("source_sha256")!=HIST_SHA or len(shard.get("cases",[]))!=25:
            raise ValueError("historical shard provenance mismatch")
        provenance[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if shard.get("status")!="complete":invalid.append(path.name)
        cases.extend(shard["cases"])
    if len(cases)!=100 or len({str(c["case_id"]) for c in cases})!=100:
        raise ValueError("expected 100 unique original cases")
    counters=Counter();dbs=defaultdict(Counter);candidate=Counter();review=[]
    for case in cases:
        cid=str(case["case_id"]);ref=index[cid];output=case["guarded"]
        if ref["db_id"]!=case["database_id"] or ref["question"]!=case["question"]:
            raise ValueError("historical case identity mismatch")
        db=case["database_id"]
        counters["total"]+=1
        counters["direct_correct"]+=int(bool(case["direct"]["official_ex"]))
        counters["guarded_correct"]+=int(bool(output["official_ex"]))
        counters["guarded_executable"]+=int(bool(output["execution_success"]))
        dbs[db]["total"]+=1
        if output["official_ex"]:
            dbs[db]["correct"]+=1
            continue
        if not output["execution_success"]:
            counters["execution_failed"]+=1
            dbs[db]["execution_failed"]+=1
            f={"execution_failure":True}
        else:
            counters["executed_but_ex_wrong"]+=1
            dbs[db]["executed_but_ex_wrong"]+=1
            f=differences(output.get("final_sql") or "",ref["SQL"])
            for key,value in f.items():
                if value is True:candidate[key]+=1
                if value is None:candidate[key+"_unknown"]+=1
            if not any(value is True for value in f.values()):
                candidate["no_structural_difference_detected"]+=1
        review.append({
          "case_id":cid,"db_id":db,"candidate_difference_flags":f,
          "requires_manual_primary_label":True,
          "primary_error_type":"UNREVIEWED",
          "human_agreement":"NOT_MEASURED",
        })
    expected=(100,33,39,91,52,9)
    actual=tuple(counters[k] for k in ("total","direct_correct","guarded_correct","guarded_executable","executed_but_ex_wrong","execution_failed"))
    if actual!=expected:raise ValueError(f"historical evidence mismatch {actual} vs {expected}")
    return {
      "status":"DIAGNOSTIC_ONLY_NOT_NEW_SCORE",
      "original_model_digest":DIGEST,
      "source_sha256":RAW_SHA,
      "shard_sha256":provenance,
      "invalid_historical_shards":invalid,
      "counts":dict(counters),
      "candidate_noncausal_difference_counts_on_52":dict(candidate),
      "by_database":{k:dict(v) for k,v in sorted(dbs.items())},
      "review_queue":sorted(review,key=lambda x:int(x["case_id"])),
      "causal_error_breakdown":"NOT_ESTABLISHED",
      "human_label_agreement":"NOT_MEASURED",
      "new_model_calls":0,
      "new_holdout_uses":0,
      "warning":"SQL string/shallow-structure differences are not causal labels; equivalent SQL and label errors can confound them. Preserve original EX labels."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--shard",action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("refusing overwrite")
    result=analyze(a.source,a.shard)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="review_queue"},sort_keys=True))
if __name__=="__main__":
    main()
