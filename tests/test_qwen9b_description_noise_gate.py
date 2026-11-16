import pytest

from evaluation.qwen9b_description_noise_gate import prune_descriptions

HEAD=(
    "Physical SQLite schema:\n"
    "orders(order_id, buyer_id, item_total, status)\n"
    "buyers(buyer_id, city_name, age)\n"
    "FK orders.buyer_id -> buyers.buyer_id"
)


def test_removes_irrelevant_descriptions_only_preserves_physical_schema():
    context=HEAD+(
        "\nColumn descriptions:\n"
        "- orders.item_total: total spending of each order in dollars\n"
        "- buyers.city_name: city of residence of the buyer\n"
        "- buyers.age: age in years\n"
    ).rstrip()
    out,meta=prune_descriptions(context,"What is total spending per order?")
    assert out.startswith(HEAD)
    assert "- orders.item_total:" in out
    assert "- buyers.age:" not in out
    assert meta["original_descriptions"]==3
    assert meta["retained_descriptions"]<3


def test_no_descriptions_returns_exact_same_schema_bytes():
    assert prune_descriptions(HEAD,"How many orders?")[0]==HEAD


def test_zero_hits_keeps_all_physical_columns_and_fks():
    original=HEAD+"\nColumn descriptions:\n- orders.item_total: value of the item"
    out,counts=prune_descriptions(original,"Find zzzfoobar.")
    assert out==HEAD
    assert counts["retained_descriptions"]==0


def test_unsupported_description_format_is_fail_closed():
    with pytest.raises(ValueError,match="format"):
        prune_descriptions(HEAD+"\nColumn descriptions:\nwrong delimiter","What are orders?")


def test_caps_do_not_drop_table_columns():
    original=HEAD+"\nColumn descriptions:\n"+"\n".join(
        f"- buyers.age: age {i}" for i in range(16))
    out,counts=prune_descriptions(original,"What is the age?")
    assert out.startswith(HEAD)
    assert counts["retained_descriptions"]==4
    assert counts["original_descriptions"]==16


def test_multiline_column_description_is_one_atomic_row():
    raw=HEAD+(
        "\nColumn descriptions:\n"
        "- orders.item_total: total spend in USD\n"
        "and may contain a newline inside source CSV field\n"
        "- buyers.age: age of buyer\n"
        "including a second line of detail"
    )
    out,count=prune_descriptions(raw,"What is total spend?")
    assert out.startswith(HEAD)
    assert "including a second line" not in out
    assert "and may contain a newline" in out
    assert count=={"original_descriptions":2,"retained_descriptions":1}
