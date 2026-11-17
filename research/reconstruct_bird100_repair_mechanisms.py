"""Reconstruct historical BIRD100 repair mechanisms from four original raw Actions artifacts.

CPU-only and zero model calls. This is a label/repair audit, not full upstream
BIRD scorer parity, a clean inference rerun, or semantic-revision evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from hashlib import sha256
from pathlib import Path

MODEL="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SOURCE="0310b7ca2dfb6247234e2073f93e7a75106d2ef68efe2ec"
UPSTREAM="483554eae102996f5ec1f4feab4e78ef29c2a394"
EXPECTED_SHARD_STATUSES=["complete","complete","complete","invalid"]
N=100

def audit(shards):
    if len(shards)!=4:
        raise ValueError("need four original physical JSON shards")
    rows,manifests=[],set()
    for i,obj in enumerate(shards):
        if (obj.get("status")!=EXPECTED_SHARD_STATUSES[i] or
            obj.get("sample_size")!=25 or len(obj.get("cases",[]))!=25 or
            obj.get("model_digest")!=MODEL or
            obj.get("source_sha256")!=SOURCE or
            (obj.get("official_bird_evaluator") or {}).get("revision")!=UPSTREAM):
            raise ValueError(f"original shard {i} status/source/model/protocol drift")
        manifest=obj.get("split_manifest_sha256")
        if not isinstance(manifest,str) or len(manifest)!=64 or manifest in manifests:
            raise ValueError(f"invalid or duplicate shard-{i} selection manifest SHA")
        manifests.add(manifest)
        rows.extend(obj["cases"])
    ids,dbs=set(),set()
    counts=Counter()
    perdb=defaultdict(lambda:Counter())
    repaired=[]
    for case in rows:
        if not isinstance(case,dict):
            raise ValueError("case must be dict")
        cid,db=str(case.get("case_id") or ""),str(case.get("database_id") or "")
        if not cid or not db or cid in ids:
            raise ValueError("duplicate/missing case ID or physical database ID")
        ids.add(cid);dbs.add(db)
        a,b=case.get("direct"),case.get("guarded")
        if (case.get("initial_sql_identical") is not True or
            not isinstance(a,dict) or not isinstance(b,dict) or
            a.get("final_sql")!=case.get("shared_initial_sql") or
            b.get("initial_sql")!=case.get("shared_initial_sql")):
            raise ValueError(f"paired original SQL not identical at {cid}")
        for arm in (a,b):
            if (type(arm.get("official_ex")) is not bool or
                type(arm.get("execution_success")) is not bool or
                type(arm.get("model_calls")) is not int):
                raise ValueError(f"invalid scored arm at {cid}")
        repair=b.get("repair_attempts")
        if type(repair) is not int or repair not in (0,1,2):
            raise ValueError(f"unexpected repair count at {cid}")
        direct_ok=a["official_ex"];guard_ok=b["official_ex"]
        direct_exe=a["execution_success"];guard_exe=b["execution_success"]
        counts["direct_correct"]+=direct_ok
        counts["guarded_correct"]+=guard_ok
        counts["direct_executable"]+=direct_exe
        counts["guarded_executable"]+=guard_exe
        counts["newly_executable"]+=not direct_exe and guard_exe
        counts["previously_executable_now_failed"]+=direct_exe and not guard_exe
        counts["rescued"]+=not direct_ok and guard_ok
        counts["regressed"]+=direct_ok and not guard_ok
        counts["rescued_after_direct_execution_failure"]+=(not direct_exe and not direct_ok and guard_ok)
        counts["new_executable_wrong"]+=(not direct_exe and guard_exe and not guard_ok)
        counts["already_executable_wrong"]+=(direct_exe and not direct_ok)
        counts[f"repair_{repair}"]+=1
        counts["direct_calls"]+=a["model_calls"]
        counts["guarded_calls"]+=b["model_calls"]
        perdb[db]["n"]+=1
        perdb[db]["direct_ex"]+=direct_ok
        perdb[db]["guarded_ex"]+=guard_ok
        if not direct_ok and guard_ok:
            repaired.append({"case_id":cid,"database_id":db,
                             "direct_error":a.get("error"),
                             "repair_attempts":repair,
                             "sql_changed":a.get("final_sql")!=b.get("final_sql")})
    expected={
        "direct_correct":33,"guarded_correct":39,"rescued":6,"regressed":0,
        "direct_executable":75,"guarded_executable":91,
        "newly_executable":16,"new_executable_wrong":10,
        "rescued_after_direct_execution_failure":6,
        "already_executable_wrong":42,
        "repair_0":75,"repair_1":15,"repair_2":10,
        "direct_calls":100,"guarded_calls":135,
    }
    if len(ids)!=N or len(dbs)!=11:
        raise ValueError(f"expected 100 unique cases / 11 DBs; got {len(ids)}/{len(dbs)}")
    if any(counts[key]!=value for key,value in expected.items()):
        raise ValueError(f"historical 100-case metrics drift: {dict(counts)}")
    if not all(r["repair_attempts"]==1 and r["sql_changed"] and r["direct_error"]
               for r in repaired):
        raise ValueError("some correctness rescue not from initial SQL error repaired once")
    if counts["previously_executable_now_failed"] or counts["regressed"]:
        raise ValueError("unexpected executable/EX regression")
    return {
        "status":"RAW_LABELS_AND_REPAIR_MECHANISMS_RECONSTRUCTED_WITH_INVALID_SHARD",
        "original_run":37307108955,"n":N,"physical_databases":11,
        "original_shard_statuses":EXPECTED_SHARD_STATUSES,
        "counts":dict(counts),
        "per_database":{k:dict(v) for k,v in sorted(perdb.items())},
        "six_rescued_cases":sorted(repaired,key=lambda r:int(r["case_id"])),
        "all_six_rescues_one_repair_of_initial_execution_error":True,
        "upstream_full_scorer_parity":False,
        "new_inference":False,
        "independent_generalization":False,"promotion_allowed":False,
        "claim_boundary":"This audit proves the historical 6 EX rescues coincide with initial nonexecuting SQL, not that a semantic reviewer fixed executable-but-wrong SQL.",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shards",nargs=4,required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("never overwrite prior result")
    x=audit([json.loads(z.read_text(encoding="utf-8")) for z in args.shards])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(x,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    print("BIRD100_REPAIR_ROOT_CAUSE_RECONSTRUCTED",json.dumps({k:v for k,v in x.items() if k not in ("six_rescued_cases","per_database")},sort_keys=True))

if __name__=="__main__":
    main()
