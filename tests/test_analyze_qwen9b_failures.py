from scripts.analyze_qwen9b_failures import differences, features

def test_differences_are_flags_not_causal_labels():
    pred="SELECT a FROM people WHERE city='London'"
    gold="SELECT count(*) FROM people JOIN city ON people.city=city.id WHERE city.name='Paris' GROUP BY city.id"
    f=differences(pred,gold)
    assert f["schema_linking"]
    assert f["value_matching"]
    assert f["join_or_aggregation"]
    assert f["evidence_or_external_knowledge"] is None

def test_same_query_has_no_difference():
    sql="SELECT a, COALESCE(b,c) FROM t"
    f=differences(sql,sql)
    assert not any(value is True for value in f.values())
    assert features(sql)["select_commas"]==1
