from evaluation.qwen_native_think_gate import MODEL, DIGEST, payload


def test_same_pinned_qwen_both_arms():
    assert MODEL == "qwen3.5:9b-q4_K_M"
    assert len(DIGEST) == 64
    direct = payload(think=False)
    reasoned = payload(think=True)
    assert direct["model"] == reasoned["model"] == MODEL
    assert direct["think"] is False
    assert reasoned["think"] is True
    assert direct["messages"] == reasoned["messages"]
    assert reasoned["options"]["num_predict"] > direct["options"]["num_predict"]
    assert reasoned["options"]["temperature"] == direct["options"]["temperature"] == 0
