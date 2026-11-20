"""Retrospective frozen original Qwen TRAIN48 unit-gate ABSTAIN reason census.

Only reads immutable original pre-gold source, K4 sealed SQL, unit decisions,
and already-scored BOOLEAN correctness labels. No model, publisher gold SQL,
database, reference execution, training or HOLDOUT300 access.

TRAIN48 is historically exposed and was previously reused. Descriptive only.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.train48_original_qwen_unit_gate_source_then_score import (
    SOURCE_PROTO,OLD_PROTO,MODEL_SHA,source_rows,_original_shards,
)

PROTO="BIRD-TRAIN48-UNIT-GATE-ABSTENTION-ROOTCAUSE-SOURCE-FROZEN-V1"
# This is explicitly a lexical indicator, NOT a semantic truth label.
COUNT_QUESTION_LEXICAL=re.compile(
    r"\b(?:how\s+many|number\s+of|count\s+of|number|count|total\s+number)\b",re.I
)
SQL_COUNT_OPERATOR=re.compile(r"\bCOUNT\s*\(",re.I)


def audit(frozen:Path,source:Path,freeze:Path,generated_dir:Path,scored_dir:Path)->dict:
    decision_data=json.loads(frozen.read_text())
    assert decision_data["protocol"]==SOURCE_PROTO
    assert decision_data["model_digest"]==MODEL_SHA
    assert decision_data["no_publisher_gold_SQL_or_preexisting_correctness_labels_loaded"]
    assert decision_data["no_new_Qwen_generation"] and decision_data["no_DEV500_or_HOLDOUT300"]
    rows,source_info=source_rows(source,freeze)
    assert decision_data["frozen_question_sha256"]==source_info["source_sha256"]
    question_map={r["source_index"]:r["question"] for r in rows}
    sql_map=_original_shards(generated_dir,rows,source_info["source_sha256"])
    reasons={x["source_index"]:x for x in decision_data["decisions"]}
    assert len(reasons)==48 and set(reasons)==set(sql_map)==set(question_map)
    labels={}
    for shard in range(8):
        d=json.loads((scored_dir/f"scored-{shard}.json").read_text())
        assert d["protocol"]==OLD_PROTO and d["model_digest"]==MODEL_SHA
        assert d["source_sha256"]==source_info["source_sha256"]
        assert d["status"]=="PUBLISHER_STRICT_PHYSICAL_SET_BAG_EX_SCORED_POST_SEAL"
        assert d["shard"]==shard and d["n"]==6
        for x in d["cases"]:
            cid=x["source_index"]
            assert cid in reasons and cid not in labels
            assert x["db_id"]==reasons[cid]["db_id"]
            labels[cid]=("CENSORED" if not x["reference_scorable"] else
                         "CORRECT" if x["control_selected_set_EX"] is True
                         else "WRONG")
    assert len(labels)==48
    assert Counter(labels.values())=={"CORRECT":26,"WRONG":20,"CENSORED":2}
    matrix=defaultdict(Counter)
    decisions=Counter()
    bydb=defaultdict(Counter)
    no_count=Counter()
    for cid,r in reasons.items():
        reason=r["unit_reason"]
        outcome=labels[cid]
        decisions[r["unit_decision"]]+=1
        matrix[reason][outcome]+=1
        bydb[r["db_id"]][reason]+=1
        sql=sql_map[cid]
        q=question_map[cid]
        assert sha256(sql.encode()).hexdigest()==r["selected_original_Qwen_SQL_sha256"]
        if reason=="NOT_SINGLE_SIMPLE_COUNT":
            if not SQL_COUNT_OPERATOR.search(sql):
                no_count["selected_SQL_has_no_COUNT_operator"]+=1
                if not COUNT_QUESTION_LEXICAL.search(q):
                    no_count["question_lacks_simple_count_lexical_marker_AND_SQL_has_no_COUNT"]+=1
            else:
                no_count["selected_SQL_contains_COUNT_but_not_supported_simple_structure"]+=1
    assert decisions=={"ABSTAIN":45,"CONSISTENT":3}
    assert sum(v["CORRECT"] for v in matrix.values())==26
    assert sum(v["WRONG"] for v in matrix.values())==20
    assert sum(v["CENSORED"] for v in matrix.values())==2
    assert no_count=={
        "selected_SQL_has_no_COUNT_operator":31,
        "question_lacks_simple_count_lexical_marker_AND_SQL_has_no_COUNT":27,
        "selected_SQL_contains_COUNT_but_not_supported_simple_structure":1,
    }
    assert matrix["NOT_SINGLE_SIMPLE_COUNT"]=={"CORRECT":15,"WRONG":15,"CENSORED":2}
    assert matrix["COMPLEX_OR_UNSAFE_SQL"]=={"CORRECT":4,"WRONG":3}
    return {
        "protocol":PROTO,
        "n":48,"original_scorable":46,"n_db":8,
        "three_state_histogram":dict(sorted(decisions.items())),
        "first_abstain_reason_counts":dict(sorted(
            (reason,sum(v.values())) for reason,v in matrix.items()
            if reason not in ("COUNT_DIRECTLY_FROM_TARGET_TABLE",
                              "COUNT_CHILD_ROWS_JOINED_ONLY_TO_PK_PARENTS"))),
        "reason_by_original_selected_SQL_SET_EX":{
            k:{"correct":v["CORRECT"],"wrong":v["WRONG"],"censored":v["CENSORED"]}
            for k,v in sorted(matrix.items())},
        "reason_by_db":{k:dict(sorted(v.items())) for k,v in sorted(bydb.items())},
        "NOT_SINGLE_SIMPLE_COUNT_subtypes_exact_regex":{
            k:no_count[k] for k in sorted(no_count)},
        "question_lexical_marker_is_not_semantic_ground_truth":True,
        "complex_or_unsafe_reason_is_structure_unsupported_not_proof_SQL_unsafe":True,
        "abstain_cause_not_equivalent_to_model_semantic_failure":True,
        "sample_was_not_filtered_for_count_unit_mechanism":True,
        "sample_known_exposed_TRAIN48_not_fresh_holdout":True,
        "no_new_Qwen_or_training":True,
        "no_publisher_gold_SQL_read":True,
        "no_new_SQL_execution":True,
        "no_DEV500_or_HOLDOUT300":True,
        "decision":"STOP COUNT-ONLY GENERATOR A_B; SOURCE-DEFINE COUNT-ELIGIBLE TASKS BEFORE NEW EXPERIMENT",
    }


def main():
    p=argparse.ArgumentParser()
    for n in ("frozen","source","freeze","generated_dir","scored_dir","out"):
        p.add_argument("--"+n.replace("_","-"),required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable evidence")
    data=audit(a.frozen,a.source,a.freeze,a.generated_dir,a.scored_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    print("TRAIN48_ABSTAIN_CAUSE_CONFIRMED",json.dumps({
        "reason_counts":data["first_abstain_reason_counts"],
        "count_sql_operators":data["NOT_SINGLE_SIMPLE_COUNT_subtypes_exact_regex"],
        "Qwen_calls":0,
    },sort_keys=True))


if __name__=="__main__":main()
