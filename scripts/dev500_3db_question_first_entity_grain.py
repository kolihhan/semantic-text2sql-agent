"""Source-first→postblind physical probe for three *different* DB semantics.

Uses original frozen Qwen3.5-9B candidate SQL and its physical keys, with no new
model inference. Proposals are determined entirely by question + schema + first
original SQL before loading publisher reference. Outcome-conditioned public DEV:
this is a diagnostic only, not Guarded or HOLDOUT EX.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from scripts.postblind_bird_dev500_count_probe5 import run_query, locate, SOURCE_SHA

TARGETS = {
    "219": ("toxicology", "percent of carcinogenic MOLECULES having triple bonds"),
    "571": ("codebase_community", "ratio of user's POSTS to their independent VOTES"),
    "1080": ("european_football_2", "number of distinct PLAYERS rather than snapshots"),
}
PROTOCOL = "QWEN35-BIRD-DEV500-THREE-SOURCE-FIRST-ENTITY-GRAIN-V1"


def digest(s):
    if isinstance(s, str):
        s = s.encode()
    return sha256(s).hexdigest()


def proposal(cid, sql, schema):
    if cid == "219":
        assert "bond" in schema and "molecule" in schema
        assert schema["molecule"]["primary_key"] == ["molecule_id"]
        assert schema["bond"]["primary_key"] == ["bond_id"]
        assert "SUM(CASE WHEN b.bond_type = '#'" in sql
        return (
            "SELECT 100.0 * COUNT(DISTINCT CASE WHEN m.label = '+' "
            "THEN m.molecule_id END) / NULLIF(COUNT(DISTINCT m.molecule_id), 0) "
            "FROM molecule m JOIN bond b ON b.molecule_id = m.molecule_id "
            "WHERE b.bond_type = '#';"
        )
    if cid == "571":
        assert "posts" in schema and "votes" in schema
        assert schema["posts"]["primary_key"] == ["Id"]
        assert schema["votes"]["primary_key"] == ["Id"]
        assert "LEFT JOIN votes ON posts.Id = votes.PostId" in sql
        return (
            "SELECT 1.0 * (SELECT COUNT(*) FROM posts WHERE OwnerUserId = 24) "
            "/ NULLIF((SELECT COUNT(*) FROM votes WHERE UserId = 24), 0);"
        )
    if cid == "1080":
        assert "Player_Attributes" in schema
        assert "player_api_id" in {
            c["name"] for c in schema["Player_Attributes"]["columns"]
        }
        assert sql.count("COUNT(*)") == 1
        return sql.replace("COUNT(*)", "COUNT(DISTINCT player_api_id)", 1)
    raise ValueError(cid)


def source_first(original_source, db_root):
    x = json.loads(original_source.read_text())
    assert x["remaining_count"] == 81 and x["database_count"] == 11
    assert x["no_gold_reference_consulted"] and x["no_new_model_calls"]
    byid = {r["case_id"]: r for r in x["source_first_cases"]}
    rows = []
    for cid, (db, meaning) in sorted(TARGETS.items(), key=lambda kv:int(kv[0])):
        r = byid[cid]
        assert r["db_id"] == db and r["publisher_reference_sql_read"] is False
        q, old = r["question"], r["first_original_qwen_sql"]
        new = proposal(cid, old, r["source_physical_schema_and_declared_keys"])
        assert new != old
        physical = locate(db_root, db)
        base, changed = run_query(physical, old), run_query(physical, new)
        rows.append({
            "case_id":cid, "db":db, "question":q, "question_sha":digest(q),
            "hypothesized_question_target":meaning,
            "original_sql":old, "proposed_sql":new,
            "original_sql_sha":digest(old), "proposal_sha":digest(new),
            "baseline_physical":base, "proposed_physical":changed,
            "physical_result_changed": (
                base["status"]=="ok" and changed["status"]=="ok"
                and base["set_sha256"] != changed["set_sha256"]
            ), "reference_sql_not_seen":True,
        })
    return {
        "protocol":PROTOCOL,"phase":"source","n":3,"n_source_dbs":3,
        "cases":rows,
        "original_model_digest":"56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90",
        "source_previously_outcome_conditioned":True,
        "source_first_81_case_run":"https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44555827205",
        "no_publisher_SQL_read":True,"new_model_calls":0,
        "no_Guarded_change":True,"no_training":True,
        "HOLDOUT300_untouched":True,"not_prospective_Guarded_EX":True,
    }


def postblind(sealed_source, official_dev, db_root):
    frozen = json.loads(sealed_source.read_text())
    assert frozen["protocol"]==PROTOCOL and frozen["phase"]=="source"
    assert frozen["no_publisher_SQL_read"] and frozen["n"]==3
    assert set(r["case_id"] for r in frozen["cases"])==set(TARGETS)
    data = official_dev.read_bytes()
    assert digest(data)==SOURCE_SHA
    byid = {str(r["question_id"]):r for r in json.loads(data)}
    out=[]
    for row in frozen["cases"]:
        cid=row["case_id"]
        gold=byid[cid]
        assert gold["db_id"]==row["db"] and gold["question"]==row["question_sha"]*0+row["question"]
        assert digest(row["question"])==row["question_sha"]
        assert digest(row["original_sql"])==row["original_sql_sha"]
        assert digest(row["proposed_sql"])==row["proposal_sha"]
        phys=locate(db_root,row["db"])
        gt=run_query(phys,gold["SQL"])
        base=run_query(phys,row["original_sql"])
        treated=run_query(phys,row["proposed_sql"])
        assert base==row["baseline_physical"] and treated==row["proposed_physical"]
        before=(gt["set_sha256"]==base["set_sha256"]
                if gt["status"]=="ok" and base["status"]=="ok" else None)
        after=(gt["set_sha256"]==treated["set_sha256"]
               if gt["status"]=="ok" and treated["status"]=="ok" else None)
        out.append({
            "case_id":cid,"db":row["db"],
            "reference_scorable":gt["status"]=="ok",
            "baseline_set_match":before,"proposed_set_match":after,
            "rescue": before is False and after is True,
            "regression":before is True and after is False,
            "physical_result_changed":row["physical_result_changed"],
        })
    paired=[r for r in out if r["baseline_set_match"] is not None
            and r["proposed_set_match"] is not None]
    return {
        "protocol":PROTOCOL,"phase":"postblind","n":3,
        "paired_scorable":len(paired),"baseline_matches":sum(r["baseline_set_match"] for r in paired),
        "proposed_matches":sum(r["proposed_set_match"] for r in paired),
        "rescues":sum(r["rescue"] for r in paired),
        "regressions":sum(r["regression"] for r in paired),
        "case_results":out,
        "source_frozen_before_reference":True,
        "new_model_calls":0,"no_Guarded_change":True,"no_training":True,
        "HOLDOUT300_untouched":True,"not_prospective_Guarded_EX":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["source","score"])
    p.add_argument("--original-81-source-first",type=Path)
    p.add_argument("--sealed",type=Path)
    p.add_argument("--official-dev",type=Path)
    p.add_argument("--db-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("immutable output already present")
    if a.phase=="source":
        assert a.original_81_source_first and not a.official_dev
        result=source_first(a.original_81_source_first,a.db_root)
    else:
        assert a.sealed and a.official_dev
        result=postblind(a.sealed,a.official_dev,a.db_root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("SOURCE_FIRST_ENTITY_GRAIN_3_"+a.phase.upper(),json.dumps({
        "n":result["n"],
        "result_changes":([[r["case_id"],r["physical_result_changed"]]
                           for r in result["cases"]] if a.phase=="source" else
                          [[r["case_id"],r["baseline_set_match"],r["proposed_set_match"]]
                           for r in result["case_results"]]),
        "model_calls":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
