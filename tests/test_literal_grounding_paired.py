import pytest
from evaluation.literal_grounding_paired import load_excluded, aggregate


def test_exclude_requires_four_disjoint_100(tmp_path):
    paths=[]
    for i in range(4):
        p=tmp_path / f"{i}.json"
        p.write_text(__import__("json").dumps({"cases":[{"case_id":str(x)} for x in range(i*100,(i+1)*100)]}))
        paths.append(p)
    assert len(load_excluded(paths))==400
    paths[2].write_text(paths[1].read_text())
    with pytest.raises(ValueError,match="overlapping"):
        load_excluded(paths)


def test_paired_grounded_math():
    shards=[]
    for i in range(4):
        rows=[]
        for j in range(10):
            k=i*10+j
            base=k<20
            grounded=base or k in {20,21}
            if k==1:grounded=False
            rows.append({"case_id":str(k),"ex":{"baseline":base,"grounded":grounded},
                         "changed_sql":True})
        shards.append({"schema":"qwen9b-exact-value-grounding-shard/v1",
                       "shard":i,"model":"test","manifest_sha256":"pin",
                       "gold_used_during_generation":False,"cases":rows})
    out=aggregate(shards)
    assert out["baseline_correct"]==20
    assert out["grounded_correct"]==21
    assert out["wrong_to_correct"]==2 and out["correct_to_wrong"]==1
    assert out["net_gain"]==1
    assert out["promote"] is False
