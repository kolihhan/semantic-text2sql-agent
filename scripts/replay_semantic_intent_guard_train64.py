"""Two-stage historical replay; candidate decisions frozen without EX labels."""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from scripts.semantic_intent_invariant_gate import freeze_decisions

def load(path):
    return json.loads(Path(path).read_text("utf-8"))

def save(path,data):
    p=Path(path)
    p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+"\n")

def freeze(unlabeled,frozen,out):
    u,f=load(unlabeled),load(frozen)
    if u.get("status")!="FROZEN_TRAIN64_QWEN_SQL_NO_GOLD" or f.get("status")!="FROZEN_NO_GOLD" or f.get("no_gold") is not True:
        raise ValueError("wrong no-gold sources")
    packet=freeze_decisions(u,f)
    orig={str(r["source_index"]):r for r in u["cases"]}
    for row in f["cases"]:
        idx=str(row["source_index"])
        if sha256(orig[idx]["sql"].encode()).hexdigest()!=row["original_sha"]:
            raise ValueError("original SQL tampered")
    if sum(d["action"] in {"ACCEPT_CANDIDATE","REJECT_CANDIDATE"} for d in packet["decisions"])!=5:
        raise ValueError("original 5 Qwen proposals changed")
    packet["protocol"]="TRAIN64-INTENT-INVARIANT-LOCKED-V1"
    packet["original_source_digest"]=f.get("source_digest")
    save(out,packet)
    print("SEMANTIC_INTENT_GATE_FROZEN",json.dumps({
        "n":64,"accepted":sum(x["action"]=="ACCEPT_CANDIDATE" for x in packet["decisions"]),
        "rejected":sum(x["action"]=="REJECT_CANDIDATE" for x in packet["decisions"]),
    }),flush=True)

def score(decisions,original,out):
    d,s=load(decisions),load(original)
    if d.get("status")!="FROZEN_INTENT_INVARIANT_DECISIONS_NO_GOLD" or d.get("no_gold_accessed") is not True or d.get("n")!=64 or d.get("protocol")!="TRAIN64-INTENT-INVARIANT-LOCKED-V1":
        raise ValueError("gate not sealed before scoring")
    if s.get("n")!=64 or s.get("status")!="EXPOSED_DB_DISJOINT_TRAIN64_TRANSFER_DIAGNOSTIC_ONLY" or s.get("scores",{}).get("base_set")!=31:
        raise ValueError("historical gold score source mismatch")
    byid={str(r["source_index"]):r for r in d["decisions"]}
    cases=s["changed_case_details"]
    if len(byid)!=64 or len(cases)!=5 or len({str(x["source_index"]) for x in cases})!=5:
        raise ValueError("changed original candidate count mismatch")
    counts=Counter()
    details=[]
    for row in cases:
        idx=str(row["source_index"])
        dec=byid[idx]
        if dec["action"] not in ("ACCEPT_CANDIDATE","REJECT_CANDIDATE") or dec["db_id"]!=row["db_id"]:
            raise ValueError("frozen decision identity mismatched scored candidate")
        accepted=dec["action"]=="ACCEPT_CANDIDATE"
        before=bool(row["before_set"])
        after=bool(row["after_set"]) if accepted else before
        bbag=bool(row["before_bag"])
        abag=bool(row["after_bag"]) if accepted else bbag
        counts["rescues"]+=int(not before and after)
        counts["regressions"]+=int(before and not after)
        counts["bag_rescues"]+=int(not bbag and abag)
        counts["bag_regressions"]+=int(bbag and not abag)
        counts["accepted"]+=int(accepted)
        counts["rejected"]+=int(not accepted)
        details.append({
            "case_id":idx,"db_id":row["db_id"],"action":dec["action"],
            "flags":dec["flags"],"before_set":before,"after_set":after,
        })
    output={
        "status":"HISTORICALLY_EXPOSED_POSTHOC_GATE_DIAGNOSTIC_NOT_PROSPECTIVE","n":64,
        "baseline_set":31,"result_set":31+counts["rescues"]-counts["regressions"],
        "baseline_bag":28,"result_bag":28+counts["bag_rescues"]-counts["bag_regressions"],
        "scores":dict(counts),"changed_cases":details,"no_new_model_calls":True,
        "promotion_allowed":False,
        "warning":"Mechanism conceived after original TRAIN64 labels were reviewed. Not independent. Not a new model improvement.",
    }
    save(out,output)
    print("SEMANTIC_INTENT_GATE_RETROSPECTIVE",json.dumps({
        "set":output["result_set"],"bag":output["result_bag"],**dict(counts),
    }),flush=True)

def main():
    p=argparse.ArgumentParser()
    s=p.add_subparsers(dest="mode",required=True)
    a=s.add_parser("freeze")
    a.add_argument("--unlabeled",required=True)
    a.add_argument("--frozen",required=True)
    a.add_argument("--output",required=True)
    b=s.add_parser("score")
    b.add_argument("--decisions",required=True)
    b.add_argument("--original-score",required=True)
    b.add_argument("--output",required=True)
    args=p.parse_args()
    if args.mode=="freeze":
        freeze(args.unlabeled,args.frozen,args.output)
    else:
        score(args.decisions,args.original_score,args.output)

if __name__=="__main__":
    main()
