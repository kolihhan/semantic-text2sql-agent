"""Offline protocol tests for the gold-blind Qwen9B semantic A/A pilot."""
import io
import json
from types import SimpleNamespace

import pytest

from evaluation import semantic_aa_probe as probe


def test_seeded_provider_pins_runtime_without_mutating_default(monkeypatch):
    captured = []
    def fake_urlopen(request, timeout):
        captured.append((json.loads(request.data), timeout))
        return io.BytesIO(b'{"message":{"content":"SELECT 1"}}')
    monkeypatch.setattr(probe.urllib.request, "urlopen", fake_urlopen)
    provider = probe.PinnedOllamaProvider(model="test", timeout_s=13)
    assert provider.complete_text(system="s", user="u") == "SELECT 1"
    assert captured[0][0]["options"] == {"temperature": 0, "seed": 42}
    assert captured[0][0]["think"] is False
    assert captured[0][1] == 13


def test_exclusion_of_original_440_is_strict(tmp_path):
    paths = []
    for i in range(4):
        path = tmp_path / f"hundred{i}.json"
        path.write_text(json.dumps({"cases": [{"case_id": str(k)}
                              for k in range(i * 100, i * 100 + 100)]}))
        paths.append(path)
    recovered = tmp_path / "recovered40.json"
    recovered.write_text(json.dumps({
        "sample_size": 40,
        "cases": [{"case_id": str(x)} for x in range(400, 440)]
    }))
    assert len(probe.excluded_440([*paths, recovered])) == 440
    with pytest.raises(ValueError, match="exact frozen"):
        probe.excluded_440(paths)
    recovered.write_text(json.dumps({
        "sample_size": 40,
        "cases": [{"case_id": str(x)} for x in range(399, 439)]
    }))
    with pytest.raises(ValueError, match="overlap"):
        probe.excluded_440([*paths, recovered])


def test_manifest_stratifies_new_questions_without_gold(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.DatabaseCatalog, "from_sqlite",
                        lambda db: db)
    monkeypatch.setattr(probe, "exact_literal_evidence",
                        lambda db, question: [{"literal": "good"}])
    from evaluation import run_bird
    monkeypatch.setattr(run_bird, "_resolve_database",
                        lambda root, db_id: tmp_path / db_id)
    cases = []
    for idx in range(32):
        cases.append(SimpleNamespace(
            case_id=500+idx,
            question='"known"' if idx < 16 else "list records",
            database_id=f"db{idx%8}",
        ))
    result = probe.select_fresh20(cases, excluded=set(map(str, range(440))),
                                  root=tmp_path)
    assert result["labels_used"] is False
    assert result["excluded_count"] == 440
    assert result["group_counts"] == {"value_hit": 10, "no_quoted_literal": 10}
    assert len({x["case_id"] for x in result["selected"]}) == 20
    assert all(v >= 4 for v in result["source_domains"].values())


def fake_shards():
    rows = []
    for i in range(20):
        group = "value_hit" if i < 10 else "no_quoted_literal"
        a = i < 8
        b = (not a) if i == 0 else a
        treatment = False if i == 1 else True if i == 8 else a
        rows.append({
            "case_id": str(i), "group": group, "database_id": f"db{i%5}",
            "treatment_context_changed": group == "value_hit",
            "predictions": {
                "baseline_a": {"sql": "SELECT 1"},
                "baseline_b": {"sql": "SELECT 2" if i == 0 else "SELECT 1"},
                "grounded": {"sql": "SELECT 3" if group == "value_hit" else "SELECT 1"},
            },
            "ex": {label: {"set": correct, "bag": correct}
                   for label, correct in
                   (("baseline_a", a), ("baseline_b", b), ("grounded", treatment))},
        })
    return [{
        "schema": "qwen9b-semantic-aa-shard/v1", "shard": j,
        "model": probe.MODEL, "temperature": 0, "seed": 42,
        "manifest_sha256": "same", "gold_used_during_generation": False,
        "cases": rows[j::4],
    } for j in range(4)]


def test_aggregate_reports_aa_instability_and_regression_not_success():
    result = probe.aggregate(fake_shards())
    assert result["count"] == 20
    assert result["promote"] is False
    metrics = result["set"]["overall"]
    assert metrics["baseline_a_correct"] == 8
    assert metrics["baseline_b_correct"] == 7
    assert metrics["grounded_correct"] == 8
    assert metrics["aa_correctness_disagreement"] == 1
    assert metrics["aa_sql_disagreement"] == 1
    assert metrics["grounded_vs_a_wrong_to_correct"] == 1
    assert metrics["grounded_vs_a_correct_to_wrong"] == 1
    assert metrics["stable_baseline_cases"] == 19
    assert result["bag"]["overall"] == metrics
    assert result["set"]["no_quoted_literal"]["aa_correctness_disagreement"] == 0


def test_aggregate_fails_closed_for_missing_or_unpinned_shards():
    runs = fake_shards()
    with pytest.raises(ValueError, match="four distinct"):
        probe.aggregate(runs[:3])
    runs = fake_shards()
    runs[1]["temperature"] = 0.7
    with pytest.raises(ValueError, match="unpinned"):
        probe.aggregate(runs)
    runs = fake_shards()
    runs[1]["cases"][0]["case_id"] = runs[0]["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="duplicate"):
        probe.aggregate(runs)
