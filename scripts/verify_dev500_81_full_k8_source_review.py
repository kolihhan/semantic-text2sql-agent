"""No-model reproducibility check for assistant source-first 81×8 survey.

Inputs: 32 original source-only Qwen cache shards; original 81-case physical
source-only ledger; assistant review markdown. No publisher reference SQL or
BIRD evaluator file is used or downloaded. Fails closed on SHA mismatch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

SOURCE_SHA="ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
MODEL_SHA="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
CACHE_PROTO="FROZEN-BIRD-DEV500-QWEN35-9B-KCACHE-V1"
PROTOCOL="QWEN9B-BIRD-DEV500-81-FULL-K8-SOURCE-ANNOTATION-VERIFY-V1"
KNOWN_PILOT16={199,235,317,474,957,852,567,477,1497,172,227,344,1229,1093,1251,978}
STATUS={"SOURCE_TEXT_CONTRADICTION","PROVISIONAL","ABSTAIN"}


def digest(s):
    if isinstance(s,str):s=s.encode()
    return sha256(s).hexdigest()


def run(source_file,cache_dir,ledger_file):
    j=json.loads(source_file.read_text())
    assert j["remaining_count"]==81 and j["n_prior_pilot"]==16
    assert j["n_fully_executable_wrong"]==97
    assert j["no_gold_reference_consulted"] and j["no_new_model_calls"]
    cases={str(x["case_id"]):x for x in j["source_first_cases"]}
    assert len(cases)==81 and not set(map(str,KNOWN_PILOT16))&set(cases)
    cache_files=sorted(cache_dir.glob("k8-dev500-candidate-cache-*.json"))
    assert len(cache_files)==32
    cache={}
    shards=set()
    for file in cache_files:
        d=json.loads(file.read_text())
        assert d["protocol"]==CACHE_PROTO and d["status"]=="CANDIDATE_CACHE_COMPLETE"
        assert d["source_sha256"]==SOURCE_SHA and d["model_digest"]==MODEL_SHA
        assert d["no_gold_sql_in_cache"] is True and d["k"]==8
        assert d["shard"] in range(32) and d["shard"] not in shards
        shards.add(d["shard"])
        for case in d["cases"]:
            cid=str(case["case_id"])
            assert cid not in cache
            cache[cid]=case
    assert shards==set(range(32)) and len(cache)==500
    for cid,row in cases.items():
        c=cache[cid]
        assert c["db_id"]==row["db_id"]
        assert digest(row["question"])==c["question_sha256"]
        sqls=[x["sql"] for x in sorted(c["samples"],key=lambda x:x["index"])]
        assert len(sqls)==8 and [x["index"] for x in sorted(c["samples"],key=lambda x:x["index"])]==list(range(8))
        assert sqls[0]==row["first_original_qwen_sql"]
        assert digest(json.dumps(sqls,sort_keys=True))==row["original_eight_sql_sha256"]
    ledger_rows={}
    for line in ledger_file.read_text().splitlines():
        match=re.fullmatch(r"\\|\\s*(\\d+)\\s*\\|\\s*([\\w_]+)\\s*\\|\\s*(SOURCE_TEXT_CONTRADICTION|PROVISIONAL|ABSTAIN)\\s*\\|\\s*([A-Z_]+)\\s*\\|",line)
        if match:
            cid,db,status,family=match.groups()
            assert cid not in ledger_rows
            ledger_rows[cid]=(db,status,family)
    assert set(ledger_rows)==set(cases)
    counts=Counter(r[1] for r in ledger_rows.values())
    assert counts=={"SOURCE_TEXT_CONTRADICTION":9,"PROVISIONAL":53,"ABSTAIN":19}
    for cid,(db,stat,fam) in ledger_rows.items():
        assert db==cases[cid]["db_id"] and stat in STATUS
        assert not fam.startswith("GOLD")
    return {
        "protocol":PROTOCOL,"n_source_cases":81,"n_original_sqls":648,
        "n_all_original_cache_cases":500,"n_cache_shards":32,
        "n_source_db_families":len({r["db_id"] for r in cases.values()}),
        "source_and_all_eight_sql_sha_checks_passed":True,
        "assistant_annotations_complete":len(ledger_rows)==81,
        "assistant_annotations_not_human_expert":True,
        "known_source_dev_outcome_conditioned":True,
        "annotation_categories":dict(sorted(counts.items())),
        "no_gold_reference_sql_opened":True,
        "no_Qwen_calls":True,"no_train":True,"holdout300_untouched":True,
        "no_accuracy_claim":True,
    }


def main():
    p=argparse.ArgumentParser()
    for n in ("source_file","cache_dir","ledger_file","out"):
        p.add_argument("--"+n.replace("_","-"),required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable output already exists")
    out=run(a.source_file,a.cache_dir,a.ledger_file)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    print("ORIGINAL_81_ALL_EIGHT_SQLS_VERIFIED",json.dumps(out,sort_keys=True),flush=True)

if __name__=="__main__":main()
