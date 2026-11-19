"""CPU-only retrospective frozen original-Qwen TRAIN96 conservative candidate gate.

A = first direct SQL. B = choose first role/grain SQL only if it passes
a question/evidence-grounded constraint missing in A. Source and blind raw
candidate SQLs are sealed BEFORE physical reference-score artifacts read.
This is a posthoc diagnostic on exposed TRAIN96, NOT a promotion trial.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

PROTOCOL="ORIGINAL-QWEN9B-TRAIN96-GOLDBLIND-CONSERVATIVE-CANDIDATE-SELECTOR-CPU-DIAGNOSTIC"
N=96
GATE={"min_net_correct":3,"max_regressions":1,"require_scorable":96}

def evidence_literals(evidence: str) -> list[str]:
    # Only explicit original publisher evidence: physical column = 'literal'.
    return list(dict.fromkeys(
        v.casefold() for v in re.findall(
            r"(?i)\b[A-Za-z_][A-Za-z_0-9]*\s*(?:=|!=|<>|LIKE)\s*'([^']{2,50})'",
            evidence or "")
    ))

def projection_width(sql: str) -> int | None:
    m=re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s", sql.strip())
    if not m or "(select" in m.group(1).casefold():
        return None
    level=count=0
    for ch in m.group(1):
        if ch=="(":level+=1
        elif ch==")":level-=1
        elif ch=="," and level==0:count+=1
    return count+1 if level==0 else None

def suspicious_invalid(sql: str) -> bool:
    return bool(re.search(r"\.[Nn]one\b",sql)) or bool(
        re.search(r"\bWHERE\b[^;]+\bJOIN\b",sql,flags=re.I|re.S))

def choose(question: str, evidence: str, direct: str, constrained: str) -> dict:
    # This function has no access to answer/reference/EX/outcome or DB labels.
    signals=[]
    for lit in evidence_literals(evidence):
        a=lit in direct.casefold()
        b=lit in constrained.casefold()
        if a!=b:
            signals.append({"rule":"explicit_evidence_literal","supports":"constrained" if b else "direct"})
    q=question.strip().casefold()
    singular=bool(re.match(
        r"^(?:what is the (?:name|title|height|weight)|which (?:year|country|city|state|film)|in which year|who is the (?:player|person))\b",
        q))
    if singular and not re.search(r"\b(and|list|give|all|how many)\b",q[:135]):
        a,b=projection_width(direct),projection_width(constrained)
        if a and b and a!=b and min(a,b)==1:
            signals.append({"rule":"singular_projection","supports":"constrained" if b==1 else "direct"})
    x,y=suspicious_invalid(direct),suspicious_invalid(constrained)
    if x!=y:
        signals.append({"rule":"obvious_invalid_syntax","supports":"direct" if y else "constrained"})
    count=Counter(x["supports"] for x in signals)
    selected="constrained" if count["constrained"]>count["direct"] else "direct"
    return {"selected":selected,"signals":signals}

def sealed_proposals(source_path:Path,generated_paths:list[Path]) -> dict:
    original=[json.loads(x) for x in source_path.read_text().splitlines() if x.strip()]
    generated={}
    if len(generated_paths)!=12:
        raise ValueError("exact 12 original immutable physical generated shards required")
    sha=None
    for p in generated_paths:
        shard=json.loads(p.read_text())
        if (shard.get("status")!="BOTH_ARM_REAL_SQL_GENERATIONS_SEALED_BEFORE_REFERENCE"
            or shard.get("model_digest")!="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"):
            raise ValueError("not original fixed model sources")
        if sha is None: sha=shard["source_sha256"]
        if shard["source_sha256"]!=sha or shard["shard"] not in range(12):
            raise ValueError("source SHA drift")
        for r in shard["cases"]:
            idx=r["source_index"]
            if idx in generated:raise ValueError("duplicate generated case")
            generated[idx]=r
    if len(original)!=N or len(generated)!=N or set(generated)!={r["source_index"] for r in original}:
        raise ValueError("original 96 complete unique cases required")
    cases=[]
    for row in original:
        gen=generated[row["source_index"]]
        if (gen["db_id"]!=row["db_id"] or gen["question_sha256"]!=row["question_sha256"]
            or sha256(row["question"].encode()).hexdigest()!=row["question_sha256"]):
            raise ValueError("source ID/question digest mismatch")
        decision=choose(row["question"],row.get("evidence",""),
                        gen["baseline_first_SQL"],gen["treatment_final_SQL"])
        cases.append({"source_index":row["source_index"],"db_id":row["db_id"],**decision})
    return {"protocol":PROTOCOL,"status":"SOURCE_MATCHED_GOLD_FREE_SELECTION_SEALED",
            "source_sha":sha,"n":N,"gate":GATE,
            "no_gold_scoring_artifacts_read":True,
            "no_model_calls":True,"cases":cases}

def score_sealed(seal:dict,score_paths:list[Path]) -> dict:
    if (seal.get("protocol")!=PROTOCOL or seal.get("status")!="SOURCE_MATCHED_GOLD_FREE_SELECTION_SEALED"
        or seal.get("n")!=N or seal.get("gate")!=GATE or len(seal.get("cases",[]))!=N
        or len(score_paths)!=12):
        raise ValueError("unverified gold-blind seal")
    matched={}
    for p in score_paths:
        obj=json.loads(p.read_text())
        if (obj.get("status")!="ORIGINAL_QWEN9B_PAIRED_PHYSICAL_STRICT_EX_AFTER_GOLD_BLIND_SQL"
            or obj.get("source_sha256")!=seal["source_sha"]
            or obj.get("model_digest")!="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"):
            raise ValueError("wrong original source/score")
        for row in obj["cases"]:
            if row["source_index"] in matched:raise ValueError("duplicate physical score")
            matched[row["source_index"]]=row
    if len(matched)!=N or set(matched)!={x["source_index"] for x in seal["cases"]}:
        raise ValueError("incomplete physical scores")
    c=Counter()
    for decision in seal["cases"]:
        scored=matched[decision["source_index"]]
        if scored["db_id"]!=decision["db_id"]:raise ValueError("DB mismatch")
        if not scored["reference_scorable"]:
            c["unscorable"]+=1
            continue
        a=bool(scored["baseline_selected_set_EX"])
        b=bool(scored["treatment_selected_set_EX"])
        pick= b if decision["selected"]=="constrained" else a
        c["scorable"]+=1
        c["original_correct"]+=int(a)
        c["selector_correct"]+=int(pick)
        c["changed"]+=int(decision["selected"]=="constrained")
        c["rescues"]+=int(not a and pick)
        c["regressions"]+=int(a and not pick)
    net=c["rescues"]-c["regressions"]
    return {"protocol":PROTOCOL,"status":"COMPLETE_RETROSPECTIVE_CPU_NO_PROMOTION",
            "source_sha":seal["source_sha"],"n":N,"counts":dict(c),
            "net":net,"gate":GATE,"positive_screen_only":(
                net>=GATE["min_net_correct"]
                and c["regressions"]<=GATE["max_regressions"]
                and c["scorable"]==GATE["require_scorable"]),
            "no_model_calls":True,"no_holdout300":True,"training":False,
            "promotion_allowed":False,
            "limitation":"Posthoc heuristic on same already-scored 96 original BIRD TRAIN questions. No prospective score, no independent DB generalization."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("seal","score"))
    p.add_argument("--source",type=Path)
    p.add_argument("--generated",nargs="+",type=Path)
    p.add_argument("--sealed",type=Path)
    p.add_argument("--scores",nargs="+",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable file")
    o=(sealed_proposals(a.source,a.generated) if a.mode=="seal"
       else score_sealed(json.loads(a.sealed.read_text()),a.scores))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    print("QWEN9B_96_CPU_CONSERVATIVE_SELECTOR",json.dumps(
        {k:v for k,v in o.items() if k!="cases"},sort_keys=True),flush=True)
if __name__=="__main__":
    main()
