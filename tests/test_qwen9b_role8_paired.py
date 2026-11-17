"""CPU-only strict Qwen9B role8 paired aggregate/provenance tests."""
from experiments.qwen9b_role8_paired import (
    SHARDS, MODEL_DIGEST, COHORT_SHA, PROTOCOL, aggregate,
)


def score(ex):
    return {"ex":ex,"bag":ex,"execution_ok":True}


def arm(ex):
    return {"direct":score(ex), "guarded":score(ex), "elapsed_s":1.,
            "initial_sql":"SELECT 1","final_sql":"SELECT 1",
            "repair_attempts":0,"guarded_status":"success"}


def cases():
    out=[]
    for n in range(8):
        out.append({
            "source_index":9000+n,"db_id":f"db{n//2}","order":["original","grounded"],
            "arms":{"original":arm(n>=2),"grounded":arm(True if n in (0,2,3,4,5,6,7) else False)},
        })
    # 1 rescue on db0, no other changes, so strict independent DB gate fails.
    return out


def shards(records):
    return [{"status":"complete","protocol":PROTOCOL,
             "cohort_sha256":COHORT_SHA,"model_digest":MODEL_DIGEST,
             "source_only_role_scoped_value_lookup":True,
             "shard":i,"n":2,"db_id":f"db{i}",
             "cases":records[2*i:2*i+2]} for i in range(SHARDS)]


def test_exact_source_and_paired_fails_nonpromotional_screen():
    assert SHARDS==4
    rows=cases()
    result=aggregate([__import__("pathlib").Path("unused")]) if False else None
    # Use temporary per-shard immutable JSON records like the actual aggregator.
    import json,tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as root:
        files=[]
        for i,r in enumerate(shards(rows)):
            p=Path(root)/f"{i}.json"
            p.write_text(json.dumps(r))
            files.append(p)
        result=aggregate(files)
        assert result["n"]==8
        assert result["arms"]["original"]["strict_set_ex"]==6
        assert result["arms"]["grounded"]["strict_set_ex"]==6
        assert result["rescues"]==1
        assert result["regressions"]==1
        assert result["positive_screen_only"] is False
        assert result["promotion_allowed"] is False


def test_duplicate_source_ids_are_fail_closed(tmp_path):
    import json,pytest
    rows=cases()
    rows[1]["source_index"]=rows[0]["source_index"]
    files=[]
    for i,item in enumerate(shards(rows)):
        p=tmp_path/f"{i}.json"
        p.write_text(json.dumps(item))
        files.append(p)
    with pytest.raises(ValueError,match="duplicate"):
        aggregate(files)
