"""Retrospective CPU replay of source-only semantic contracts on preserved Qwen SQL.

Entirely label-free detector; correctness labels are accessed only for
after-the-fact reporting. Historical DEV100 includes one invalid original
shard; 48 TRAIN questions are previously exposed, not independent.
No model calls, no publisher SQL fetched, no QLoRA or new SQL generation.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
from semantic_sql.semantic_contracts import detect_semantic_concerns

PROTOCOL="ORIGINAL-QWEN9B-SEMANTIC-CONTRACT-CPU-RETROSPECTIVE-V1"
MODEL_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
DEV100_SOURCE="0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
TRAIN48_SOURCE="006773fbd40894ba202c9d31aa54e0af95bcbd7f50e82d65871b36b7e2b89fc0"
SCREEN={"max_known_correct_blocking_flagged_dev100":0,
        "min_dev100_wrong_blocking_flagged_for_new_model_trial":5,
        "min_train48_wrong_blocking_flagged_for_new_model_trial":3}


def _summarize(flags:list[dict],n:int,metric:str)->dict:
    counts=Counter()
    cases=[]
    for r in flags:
        if r["correct"] is None:
            counts["unscorable"]+=1
        else:
            counts["scorable"]+=1
            counts["correct"]+=int(r["correct"])
            counts["wrong"]+=int(not r["correct"])
        if not r["blocking"] and not r["advisory"]:
            continue
        counts["flagged"]+=1
        counts["blocking_flagged"]+=int(bool(r["blocking"]))
        counts["advisory_flagged"]+=int(bool(r["advisory"]))
        if r["correct"] is not None:
            counts["known_correct_blocking_flagged"]+=int(bool(r["blocking"]) and r["correct"])
            counts["wrong_blocking_flagged"]+=int(bool(r["blocking"]) and not r["correct"])
            counts["known_correct_advisory_flagged"]+=int(bool(r["advisory"]) and r["correct"])
            counts["wrong_advisory_flagged"]+=int(bool(r["advisory"]) and not r["correct"])
        for code in r["blocking"]:
            counts["blocking_"+code]+=1
        for code in r["advisory"]:
            counts["advisory_"+code]+=1
        cases.append({k:r[k] for k in ("id","db","correct","blocking","advisory")})
    if counts["scorable"]+counts["unscorable"]!=n:
        raise ValueError("dataset must be complete")
    return {"n":n,"metric":metric,"counts":dict(counts),"flagged_cases":cases}


def _scan(q:str,sql:str)->tuple[list[str],list[str]]:
    signals=detect_semantic_concerns(q,sql)
    return ([x.code for x in signals if x.blocking],
            [x.code for x in signals if not x.blocking])


def replay_original_dev100(folder:Path)->dict:
    files=sorted(folder.glob("qwen9b-shard-*.json"))
    if len(files)!=4:raise ValueError("require four original physical DEV100 Qwen shards")
    ids=set()
    results=[]
    status=Counter()
    for f in files:
        data=json.loads(f.read_text())
        if (data.get("source_sha256")!=DEV100_SOURCE or
            data.get("model_digest")!=MODEL_DIGEST or
            data.get("sample_size")!=25):
            raise ValueError("original Qwen9B DEV100 source drift")
        status[data["status"]]+=1
        for c in data["cases"]:
            id=c["case_id"]
            if id in ids:raise ValueError("duplicate historical DEV100 case")
            ids.add(id)
            ans=c["guarded"]
            sql=ans["final_sql"] or ""
            b,a=_scan(c["question"],sql)
            results.append({"id":id,"db":c["database_id"],
                            "correct":bool(ans["official_ex"]),
                            "blocking":b,"advisory":a})
    if (len(ids)!=100 or status["complete"]!=3 or status["invalid"]!=1
        or sum(r["correct"] for r in results)!=39):
        raise ValueError("historical failed workflow or known score changed")
    return {**_summarize(results,100,"historical_guarded_bird_dev100_source_ex"),
            "statuses":dict(status),
            "original_workflow_not_end_to_end_valid":True,
            "historical_ex_39_of_100_is_not_new_score":True}


def replay_train48(source:Path,generated_folder:Path,scored_folder:Path)->dict:
    if sha256(source.read_bytes()).hexdigest()!=TRAIN48_SOURCE:
        raise ValueError("original 48 source identity drift")
    rows=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    if len(rows)!=48 or len({r["source_index"] for r in rows})!=48:
        raise ValueError("wrong original 48 unique questions")
    questions={x["source_index"]:x for x in rows}
    generated={}
    scores={}
    genfiles=sorted(generated_folder.glob("generated-*.json"))
    scorefiles=sorted(scored_folder.glob("scored-*.json"))
    if len(genfiles)!=8 or len(scorefiles)!=8:
        raise ValueError("all 8 original physical DB shards required")
    for file in genfiles:
        obj=json.loads(file.read_text())
        if (obj["source_sha256"]!=TRAIN48_SOURCE or
            obj["model_digest"]!=MODEL_DIGEST or
            obj["status"]!="ORIGINAL_QWEN9B_CANDIDATES_AND_CHOICES_SEALED_PRE_GOLD"):
            raise ValueError("wrong Qwen model or pre-gold SQL source")
        for c in obj["cases"]:
            idx=c["source_index"]
            if idx in generated:raise ValueError("duplicate sealed SQL case")
            generated[idx]=c
    for file in scorefiles:
        obj=json.loads(file.read_text())
        if (obj["source_sha256"]!=TRAIN48_SOURCE or
            obj["model_digest"]!=MODEL_DIGEST or
            obj["status"]!="PUBLISHER_STRICT_PHYSICAL_SET_BAG_EX_SCORED_POST_SEAL"):
            raise ValueError("wrong original publisher score")
        for c in obj["cases"]:
            idx=c["source_index"]
            if idx in scores:raise ValueError("duplicate original physical score")
            scores[idx]=c
    if set(questions)!=set(generated) or set(questions)!=set(scores):
        raise ValueError("some original physical data missing")
    arms={}
    for arm in ("control","treatment"):
        checked=[]
        for idx,q in questions.items():
            gen=generated[idx]
            score=scores[idx]
            if (q["db_id"]!=gen["db_id"] or q["db_id"]!=score["db_id"] or
                q["question_sha256"]!=gen["question_sha256"]):
                raise ValueError("case DB, SQL and original question identity mismatch")
            ci=gen[arm+"_choice"]["index"]
            if ci not in range(4):raise ValueError("missing blind choice")
            sql=gen[arm+"_sql"][ci]
            b,a=_scan(q["question"],sql)
            correct=(bool(score[arm+"_selected_set_EX"])
                     if score["reference_scorable"] else None)
            checked.append({"id":str(idx),"db":q["db_id"],
                            "correct":correct,"blocking":b,"advisory":a})
        arms[arm]=_summarize(checked,48,"original_train48_selected_set_ex")
    if (arms["control"]["counts"]["scorable"]!=46 or
        arms["control"]["counts"]["correct"]!=26 or
        arms["treatment"]["counts"]["correct"]!=25):
        raise ValueError("physical K4 source score mismatch")
    return {"n":48,"source_question_ids_past_exposed":True,"arms":arms}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw100",type=Path,required=True)
    p.add_argument("--source48",type=Path,required=True)
    p.add_argument("--generated48",type=Path,required=True)
    p.add_argument("--scored48",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("immutable output")
    original=replay_original_dev100(args.raw100)
    reused=replay_train48(args.source48,args.generated48,args.scored48)
    # This is ONLY a safety/coverage screen, and cannot establish causal EX.
    screen=(
        original["counts"].get("known_correct_blocking_flagged",0)
        <=SCREEN["max_known_correct_blocking_flagged_dev100"] and
        original["counts"].get("wrong_blocking_flagged",0)
        >=SCREEN["min_dev100_wrong_blocking_flagged_for_new_model_trial"] and
        reused["arms"]["control"]["counts"].get("wrong_blocking_flagged",0)
        >=SCREEN["min_train48_wrong_blocking_flagged_for_new_model_trial"]
    )
    report={
        "protocol":PROTOCOL,
        "status":"COMPLETE_RETROSPECTIVE_ORIGINAL_QWEN9B_SEMANTIC_SIGNAL_SCREEN",
        "model_digest":MODEL_DIGEST,"source_dev100":DEV100_SOURCE,
        "source_train48":TRAIN48_SOURCE,"historical_dev100":original,
        "previously_exposed_train48":reused,
        "screen_gate":SCREEN,"positive_screen_only":bool(screen),
        "promotion_allowed":False,"new_model_calls":0,
        "no_training":True,"no_new_execution_match":True,
        "DEV500_or_HOLDOUT300_not_opened":True,
        "limitation":"All questions and original candidate EX outcomes previously exposed; historical DEV100 has one invalid original shard. Signal-vs-label associations are retrospective, not repair gains, independent test, or confirmed semantic correctness."
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    brief=lambda r:{k:v for k,v in r.items() if k not in ("flagged_cases","cases")}
    print("ORIGINAL_SEMANTIC_CONTRACT_CPU_FINAL",json.dumps({
        "historical_dev100":brief(original),
        "train48_control":brief(reused["arms"]["control"]),
        "train48_treatment":brief(reused["arms"]["treatment"]),
        "positive_screen_only":bool(screen),
        "promotion_allowed":False
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
