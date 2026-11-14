"""The counterfactual must be evidence-backed and avoid changing other predicates."""
from evaluation.qwen_casefold_counterfactual import rewrite_verified_casefold


def test_rewrites_repeated_same_physical_column_predicates():
    sql = (
        "SELECT x FROM schools s WHERE s.FundingType = 'Locally Funded' "
        "AND id IN (SELECT id FROM schools s2 WHERE s2.FundingType = 'Locally Funded')"
    )
    evidence=[{
        "table":"schools","column":"FundingType",
        "value":"Locally Funded","replacement":"Locally funded",
        "status":"casefold_match",
    }]
    output=rewrite_verified_casefold(sql,evidence)
    assert output is not None
    assert output.count("= 'Locally funded'")==2
    assert "Locally Funded" not in output


def test_avoids_ambiguous_cross_column_rewrite():
    sql="SELECT * FROM t WHERE t.type='Commander' AND t.note='Commander'"
    evidence=[{
        "table":"t","column":"type","value":"Commander",
        "replacement":"commander","status":"casefold_match",
    }]
    assert rewrite_verified_casefold(sql,evidence) is None


def test_only_casefold_not_semantic_value_guess():
    sql="SELECT * FROM t WHERE t.type = 'Commander'"
    evidence=[{"table":"t","column":"type","value":"Commander",
                "replacement":"Captain","status":"casefold_match"}]
    assert rewrite_verified_casefold(sql,evidence) is None
    evidence[0]["replacement"]="commander"
    assert rewrite_verified_casefold(sql,evidence)=="SELECT * FROM t WHERE t.type = 'commander'"


def test_unknown_probe_and_casefold_multiplicity_fail_closed():
    sql="SELECT * FROM t WHERE t.type = 'Commander'"
    miss=[{"column":"type","value":"Commander","replacement":"commander","status":"probe_interrupted"}]
    assert rewrite_verified_casefold(sql,miss) is None
    yes={**miss[0],"status":"casefold_match"}
    assert rewrite_verified_casefold(sql,[yes,yes]) is None
