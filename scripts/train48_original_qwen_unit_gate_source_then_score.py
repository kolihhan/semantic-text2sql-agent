"""Pure CPU historical TRAIN48 source-unit gate and post-seal outcome assessment.

Source step receives original Qwen K4 SQL + previously frozen *questions* and
SHA-pinned physical SQLite only, NO publisher reference SQL / outcome labels.
Score step is separate, after source-gate artifact is sealed, consumes original
historical Boolean SET EX; never re-generates, rewrites or executes gold SQL.

TRAIN48 was previously used for a DIFFERENT counterhypothesis A/B, so this is
a reused retrospective *mixed* screening set, not untouched promotion data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from urllib.request import urlopen
from urllib.parse import quote

GATE_PROTO="BIRD-QUESTION-SCHEMA-SQL-UNIT-GATE-CAUTIOUS-V1"

SOURCE_PROTO="BIRD-TRAIN48-ORIGINAL-QWEN-K4-TYPED-UNIT-PRE-GOLD-RECHECK-V1"
SCORE_PROTO="BIRD-TRAIN48-ORIGINAL-QWEN-K4-TYPED-UNIT-REUSED-LABEL-SCORE-V1"
OLD_PROTO="QWEN35-9B-TRAIN48-MATCHED-K4-STRUCTURAL-COUNTERHYPOTHESIS-V1"
MODEL_SHA="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
TRAIN_SHA="cd5884489dffd7c76d73509b396bd919fbe8b486efbc4ba2f48f89460de7d69f"
DATASET="Emulated-Inc/text-to-sql-with-evidence-training-pool"
REVISION="cd2c4ea321075aa53ce7dfe52deaaffa505d046a"
MIRROR="https://huggingface.co"


def source_rows(source:Path,freeze:Path):
    manifest=json.loads(freeze.read_text())
    data=source.read_bytes()
    assert manifest["source_sha256"]==sha256(data).hexdigest()
    assert manifest["protocol"]==OLD_PROTO
    assert manifest["model_digest"]==MODEL_SHA
    assert manifest["train_sha256"]==TRAIN_SHA
    assert manifest["n"]==48 and manifest["n_dbs"]==8 and manifest["per_db"]==6
    assert manifest["source_contains_no_gold_SQL"] and manifest["DEV500_unused"] and manifest["HOLDOUT300_unused"]
    assert manifest["excluded_prior_source_indices"]==711
    rows=[json.loads(line) for line in data.splitlines() if line.strip()]
    assert len(rows)==48 and len({r["source_index"] for r in rows})==48
    for row in rows:
        assert row["question_sha256"]==sha256(row["question"].encode()).hexdigest()
        assert "SQL" not in row and "gold" not in row
    return rows,manifest


def _original_shards(directory:Path, rows:list[dict], digest:str):
    ids={r["source_index"]:r for r in rows}
    result={}
    for shard in range(8):
        p=directory/f"generated-{shard}.json"
        obj=json.loads(p.read_text())
        assert obj["protocol"]==OLD_PROTO
        assert obj["model_digest"]==MODEL_SHA
        assert obj["source_sha256"]==digest
        assert obj["status"]=="ORIGINAL_QWEN9B_CANDIDATES_AND_CHOICES_SEALED_PRE_GOLD"
        assert obj["n"]==6 and obj["shard"]==shard
        assert obj["no_gold_in_model_or_selector"] is True
        for row in obj["cases"]:
            cid=row["source_index"]
            assert cid in ids and cid not in result
            assert row["db_id"]==ids[cid]["db_id"]
            assert row["question_sha256"]==ids[cid]["question_sha256"]
            sqls=row["control_sql"]
            assert len(sqls)==4 and row["control_choice"]["index"] in range(4)
            result[cid]=sqls[row["control_choice"]["index"]]
    assert set(result)==set(ids)
    return result


def _manifest():
    url=f"{MIRROR}/api/datasets/{DATASET}/tree/{REVISION}/raw/bird/train_databases?recursive=true&limit=1000"
    with urlopen(url,timeout=60) as r:
        docs=json.load(r)
    result={Path(x["path"]).stem:x for x in docs if str(x.get("path","")).endswith(".sqlite")}
    assert len(result)==51
    return result


def _fetch_db(db:str,record:dict,root:Path):
    """SHA-check LFS or original Git blob; never fetch any SQL labels."""
    from hashlib import sha1
    size=record["size"]
    assert size>0 and size<100_000_000
    url=(f"{MIRROR}/datasets/{DATASET}/resolve/{REVISION}/raw/bird/"
         f"train_databases/{quote(db)}/{quote(db)}.sqlite")
    with urlopen(url,timeout=180) as r:
        blob=r.read(size+1)
    assert len(blob)==size
    if "lfs" in record:
        assert sha256(blob).hexdigest()==record["lfs"]["oid"]
    else:
        assert sha1(f"blob {len(blob)}\0".encode()+blob).hexdigest()==record["oid"]
    path=root/db/(db+".sqlite")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(blob)
    return path


def source_stage(source:Path,freeze:Path,generated_dir:Path,db_cache:Path)->dict:
    from semantic_sql.question_schema_sql_unit_gate import decide
    rows,manifest=source_rows(source,freeze)
    original_sql=_original_shards(generated_dir,rows,manifest["source_sha256"])
    dbmap=_manifest()
    by_db=defaultdict(list)
    for row in rows:
        by_db[row["db_id"]].append(row)
    assert set(by_db)==set(manifest["dbs"])
    decisions=[]
    for db in sorted(by_db):
        path=_fetch_db(db,dbmap[db],db_cache)
        with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as con:
            con.execute("PRAGMA query_only=ON")
            for row in by_db[db]:
                cid=row["source_index"]
                test=decide(row["question"],original_sql[cid],con)
                assert test["protocol"]==GATE_PROTO
                decisions.append({
                    "source_index":cid,
                    "db_id":db,
                    "question_sha256":row["question_sha256"],
                    "selected_original_Qwen_SQL_sha256":sha256(original_sql[cid].encode()).hexdigest(),
                    "unit_decision":test["decision"],
                    "unit_reason":test["reason"],
                    "whole_query_correctness_not_inferred":True
                })
    assert len(decisions)==48
    return {
        "protocol":SOURCE_PROTO,
        "old_original_provenance_protocol":OLD_PROTO,
        "model_digest":MODEL_SHA,"frozen_question_sha256":manifest["source_sha256"],
        "frozen_train_source_sha256":manifest["train_sha256"],
        "n":48,"n_databases":8,
        "decision_histogram":dict(sorted(Counter(x["unit_decision"] for x in decisions).items())),
        "decisions":sorted(decisions,key=lambda r:r["source_index"]),
        "no_publisher_gold_SQL_or_preexisting_correctness_labels_loaded":True,
        "original_Qwen_K4_SQL_only":True,
        "no_new_Qwen_generation":True,"no_training":True,
        "no_SQL_rewrite_or_Guarded_default_change":True,
        "no_DEV500_or_HOLDOUT300":True,
        "preselected_historical_TRAIN48_not_independent_promotion_data":True,
    }


def score_stage(frozen:Path,scored_dir:Path)->dict:
    a=json.loads(frozen.read_text())
    assert a["protocol"]==SOURCE_PROTO and a["n"]==48 and a["n_databases"]==8
    assert a["model_digest"]==MODEL_SHA and a["no_publisher_gold_SQL_or_preexisting_correctness_labels_loaded"]
    prior={r["source_index"]:r for r in a["decisions"]}
    assert len(prior)==48
    labels={}
    for shard in range(8):
        p=scored_dir/f"scored-{shard}.json"
        d=json.loads(p.read_text())
        assert d["protocol"]==OLD_PROTO and d["model_digest"]==MODEL_SHA
        assert d["source_sha256"]==a["frozen_question_sha256"]
        assert d["status"]=="PUBLISHER_STRICT_PHYSICAL_SET_BAG_EX_SCORED_POST_SEAL"
        assert d["shard"]==shard and d["n"]==6
        for row in d["cases"]:
            cid=row["source_index"]
            assert cid in prior and cid not in labels
            assert row["db_id"]==prior[cid]["db_id"]
            labels[cid]=row["control_selected_set_EX"] if row["reference_scorable"] else None
    assert len(labels)==48
    matrix=defaultdict(Counter)
    units_by_db=defaultdict(Counter)
    for cid,decision in prior.items():
        outcome=labels[cid]
        label="correct" if outcome is True else "wrong" if outcome is False else "censored"
        matrix[decision["unit_decision"]][label]+=1
        units_by_db[decision["db_id"]][decision["unit_decision"]]+=1
    flat={t:{"correct":matrix[t]["correct"],
             "wrong":matrix[t]["wrong"],"censored":matrix[t]["censored"]}
          for t in ("CONSISTENT","CONFLICT","ABSTAIN")}
    assert sum(x["correct"] for x in flat.values())==26
    assert sum(x["wrong"] for x in flat.values())==20
    assert sum(x["censored"] for x in flat.values())==2
    conflict=flat["CONFLICT"]
    false_alarms=conflict["correct"]
    true_proposals=conflict["wrong"]
    promoted=(true_proposals>=3 and false_alarms==0
              and len({d["db_id"] for d in a["decisions"]
                       if d["unit_decision"]=="CONFLICT" and labels[d["source_index"]] is False})>=2)
    return {
        "protocol":SCORE_PROTO,"n":48,"n_scorable":46,
        "original_control_selected_correct":26,
        "original_control_selected_wrong":20,
        "original_gold_censored":2,
        "three_state_outcome_cross_tab":flat,
        "three_state_by_database":{db:dict(x) for db,x in sorted(units_by_db.items())},
        "false_conflicts_on_already_correct_original_SQL":false_alarms,
        "true_conflicts_on_wrong_original_SQL":true_proposals,
        "exploratory_positive_gate":promoted,
        "promote_or_launch_new_Qwen_from_this_alone":False,
        "original_train48_reused_after_other_Qwen_experiment":True,
        "new_Qwen_calls":0,"no_gold_SQL_loaded":True,
        "no_train_or_qlora":True,"holdout300_untouched":True,
        "decision":"Retrospective unit mismatch screening only; any future training-source-disjoint confirmation must be frozen pre-use.",
    }


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="mode",required=True)
    src=sub.add_parser("source")
    for name in ("source","freeze","generated_dir","db_cache","out"):
        src.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    sc=sub.add_parser("score")
    for name in ("frozen","scored_dir","out"):
        sc.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("original sealed result already exists")
    result=(source_stage(args.source,args.freeze,args.generated_dir,args.db_cache)
            if args.mode=="source" else score_stage(args.frozen,args.scored_dir))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("TRAIN48_UNIT_GATE_"+args.mode.upper(),json.dumps({
        "n":result["n"],
        "summary":result.get("three_state_outcome_cross_tab",result.get("decision_histogram")),
        "new_Qwen_calls":0,
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
