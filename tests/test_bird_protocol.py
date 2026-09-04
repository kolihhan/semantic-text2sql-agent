from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from evaluation.metrics import paired_transition_counts
from evaluation.run_bird import _official_ex, main, run_paired_dev


def _fixture(tmp_path: Path, *, sql: str = "SELECT COUNT(*) FROM items", case_count: int = 1) -> tuple[Path, Path, Path]:
    database_root = tmp_path / "dbs"
    db_dir = database_root / "db1"
    db_dir.mkdir(parents=True)
    database = db_dir / "db1.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, label TEXT)")
        connection.executemany("INSERT INTO items(label) VALUES (?)", [("a",), ("b",)])
    source = tmp_path / "bird.json"
    source.write_text(json.dumps([{
        "question_id": f"q{index}", "db_id": "db1", "question": "How many items are there?",
        "evidence": "Count rows in items.", "SQL": sql,
    } for index in range(1, case_count + 1)]), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "protocol_version": 1,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": "test", "dev_case_ids": [f"q{index}" for index in range(1, case_count + 1)], "final_case_ids": [],
    }), encoding="utf-8")
    return source, database_root, manifest


class Provider:
    def __init__(self, *responses: str) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def complete_text(self, *, system: str, user: str) -> str:
        self.prompts.append(system + "\n" + user)
        return next(self.responses)


def test_paired_runner_shares_initial_candidate_and_counts_repairs(tmp_path: Path) -> None:
    source, database_root, manifest = _fixture(tmp_path)
    provider = Provider("SELECT missing FROM items", "SELECT COUNT(*) FROM items")
    payload = run_paired_dev(
        source_path=source, database_root=database_root, manifest_path=manifest,
        output_path=tmp_path / "paired.json", model="fake", provider_factory=lambda: provider,
        max_rows=10,
    )
    case = payload["cases"][0]
    assert case["direct"]["initial_sql"] == case["guarded"]["initial_sql"]
    assert case["direct"]["initial_sql"] == "SELECT missing FROM items"
    assert case["direct"]["model_calls"] == 1
    assert case["guarded"]["model_calls"] == 2
    assert case["guarded"]["repair_attempts"] == 1
    assert case["direct"]["latency_s"] >= case["shared_generation_latency_s"]
    assert case["guarded"]["latency_s"] >= case["shared_generation_latency_s"]
    assert len(provider.prompts) == 2
    assert all("SELECT COUNT(*) FROM items" not in prompt for prompt in provider.prompts)


def test_paired_latencies_include_shared_generation(monkeypatch, tmp_path: Path) -> None:
    source, database_root, manifest = _fixture(tmp_path)
    provider = Provider("SELECT COUNT(*) FROM items")
    ticks = iter((0.0, 2.0, 3.0, 5.0, 6.0, 10.0))
    monkeypatch.setattr("evaluation.run_bird.time.perf_counter", lambda: next(ticks))
    payload = run_paired_dev(
        source_path=source, database_root=database_root, manifest_path=manifest,
        output_path=tmp_path / "timed.json", model="fake", provider_factory=lambda: provider,
    )
    case = payload["cases"][0]
    assert case["shared_generation_latency_s"] == 2.0
    assert case["direct"]["latency_s"] == 4.0
    assert case["guarded"]["latency_s"] == 6.0


def test_official_ex_uses_sets_and_fails_closed_on_truncation(tmp_path: Path) -> None:
    source, database_root, _ = _fixture(tmp_path, sql="SELECT label FROM items")
    database = database_root / "db1" / "db1.sqlite"
    assert _official_ex(database, "SELECT label FROM items ORDER BY label DESC", "SELECT label FROM items", max_rows=10) is True
    assert _official_ex(database, "SELECT label FROM items", "SELECT label FROM items", max_rows=1) is False
    assert _official_ex(database, "SELECT 1 UNION ALL SELECT 1", "SELECT 1", max_rows=10) is True
    provider = Provider("SELECT COUNT(*) FROM items")
    payload = run_paired_dev(
        source_path=source, database_root=database_root, manifest_path=tmp_path / "manifest.json",
        output_path=tmp_path / "truncated.json", model="fake", provider_factory=lambda: provider, max_rows=1,
    )
    assert payload["status"] == "invalid"


def test_paired_metrics_are_rates(tmp_path: Path) -> None:
    source, database_root, manifest = _fixture(tmp_path)
    provider = Provider("SELECT missing FROM items", "SELECT COUNT(*) FROM items")
    payload = run_paired_dev(
        source_path=source, database_root=database_root, manifest_path=manifest,
        output_path=tmp_path / "metrics.json", model="fake", provider_factory=lambda: provider,
        max_rows=10,
    )
    assert payload["metrics"]["direct"]["official_bird_ex"] == 0.0
    assert payload["metrics"]["direct"]["execution_success"] == 0.0
    assert payload["metrics"]["guarded"]["official_bird_ex"] == 1.0
    assert payload["metrics"]["guarded"]["execution_success"] == 1.0
    assert payload["runtime"]["max_repairs"] == 2
    assert payload["runtime"]["max_rows"] == 10
    assert payload["runtime"]["guarded_treatment"] == "langgraph_preflight_plus_internal_cot_repair_v1"


def test_paired_transition_counts_cover_all_four_transitions() -> None:
    records = [
        {"direct": {"official_ex": False, "execution_success": True}, "guarded": {"official_ex": True, "execution_success": True}},
        {"direct": {"official_ex": True, "execution_success": True}, "guarded": {"official_ex": False, "execution_success": True}},
        {"direct": {"official_ex": False, "execution_success": False}, "guarded": {"official_ex": True, "execution_success": True}},
        {"direct": {"official_ex": True, "execution_success": True}, "guarded": {"official_ex": False, "execution_success": False}},
    ]
    assert paired_transition_counts(records) == {
        "direct_wrong_to_guarded_correct": 2,
        "direct_correct_to_guarded_wrong": 2,
        "direct_execution_fail_to_guarded_success": 1,
        "direct_execution_success_to_guarded_fail": 1,
    }


def test_output_overwrite_is_refused_and_crash_is_incomplete(tmp_path: Path) -> None:
    source, database_root, manifest = _fixture(tmp_path)
    output = tmp_path / "paired.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_paired_dev(source_path=source, database_root=database_root, manifest_path=manifest, output_path=output, model="fake", provider_factory=lambda: Provider())

    output.unlink()
    with pytest.raises(StopIteration):
        run_paired_dev(source_path=source, database_root=database_root, manifest_path=manifest, output_path=output, model="fake", provider_factory=lambda: Provider())
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "incomplete"


def test_resume_validates_and_continues_only_the_exact_manifest_suffix(tmp_path: Path) -> None:
    source, database_root, manifest = _fixture(tmp_path, case_count=2)
    output = tmp_path / "paired.json"
    interrupted_provider = Provider("SELECT COUNT(*) FROM items")
    with pytest.raises(StopIteration):
        run_paired_dev(
            source_path=source, database_root=database_root, manifest_path=manifest,
            output_path=output, model="fake", provider_factory=lambda: interrupted_provider,
            max_rows=10,
        )
    before_bytes = output.read_bytes()
    before_case = json.loads(before_bytes)["cases"][0]

    with pytest.raises(ValueError, match="identity/config mismatch"):
        run_paired_dev(
            source_path=source, database_root=database_root, manifest_path=manifest,
            output_path=output, model="fake", provider_factory=lambda: Provider(),
            max_rows=11, resume=True,
        )
    assert output.read_bytes() == before_bytes

    malformed = json.loads(before_bytes)
    malformed["cases"][0]["case_id"] = "q2"
    output.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="exact manifest prefix"):
        run_paired_dev(
            source_path=source, database_root=database_root, manifest_path=manifest,
            output_path=output, model="fake", provider_factory=lambda: Provider(),
            max_rows=10, resume=True,
        )

    output.write_bytes(before_bytes)
    resumed = run_paired_dev(
        source_path=source, database_root=database_root, manifest_path=manifest,
        output_path=output, model="fake", provider_factory=lambda: Provider("SELECT COUNT(*) FROM items"),
        max_rows=10, resume=True,
    )
    assert resumed["status"] == "complete"
    assert [case["case_id"] for case in resumed["cases"]] == ["q1", "q2"]
    assert resumed["cases"][0] == before_case


def test_cli_rejects_split_arm_and_final_selector() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--source", "x", "--database-root", "x", "--manifest", "x", "--split", "dev", "--arm", "structured", "--output", "x"])
    with pytest.raises(SystemExit):
        main(["paired-dev", "--source", "x", "--database-root", "x", "--manifest", "x", "--final", "--output", "x"])


def test_ollama_model_digest_preflight_records_exact_identity(monkeypatch):
    import json
    from evaluation import run_bird

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"models": [{"name": "qwen3.5:4b", "digest": "a" * 64}]}).encode()

    monkeypatch.setattr(run_bird, "urlopen", lambda *args, **kwargs: Response())
    assert run_bird._ollama_model_digest("http://localhost:11434", "qwen3.5:4b") == "a" * 64
