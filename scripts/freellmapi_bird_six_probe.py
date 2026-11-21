"""Bounded, preregistered, already-exposed BIRD DEV diagnostic for anonymous OVH Qwen.

NOT a held-out benchmark, not model-equivalent to local Qwen 9B Q4_K_M.
Use only with the frozen existing BIRD v5 100-case reference.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from evaluation.run_bird import run_paired_dev
from semantic_sql.providers import FreeLLMAPIProvider

SOURCE_SHA = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
REFERENCE_SHA = "b3b4ec070e63a7106d023917e0f0cc61ea75e2be2f2888b8c7bdaf2064fb3ae1"
# First six frozen v5 cases in their *original* order, chosen without looking at correctness.
CASE_IDS = ("38", "952", "943", "622", "1366", "1139")
MAX_HTTP_ATTEMPTS = 22
MIN_GAP_S = 35.0  # anonymous OVH listed at ~2 requests/min per model/IP
MAX_REPAIRS = 2


def request_json(url: str, payload: dict | None = None, *, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, headers=headers, data=data,
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=25) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("unexpected gateway reply type")
    return value


class CallBudget:
    def __init__(self) -> None:
        self.last_start = 0.0
        self.http_attempts = 0
        self.calls: list[dict] = []

    def wait_turn(self) -> None:
        if self.http_attempts >= MAX_HTTP_ATTEMPTS:
            raise RuntimeError("anonymous HTTP attempt budget exhausted")
        sleep = MIN_GAP_S - (time.monotonic() - self.last_start)
        if sleep > 0:
            print(f"ANONYMOUS_PACING sleep_s={sleep:.1f}", flush=True)
            time.sleep(sleep)
        self.last_start = time.monotonic()
        self.http_attempts += 1


class MeteredProvider:
    def __init__(self, api_key: str, budget: CallBudget) -> None:
        self.delegate = FreeLLMAPIProvider(
            base_url="http://127.0.0.1:3001/v1",
            api_key=api_key, model="qwen3.5-9b",
            expected_response_model="Qwen3.5-9B",
            expected_response_platform="ovh",
            max_tokens=512, timeout_s=95,
        )
        self.budget = budget

    def complete_text(self, *, system: str, user: str) -> str:
        # Only one retry on transient quota/server failure; never change model.
        for retry in range(2):
            self.budget.wait_turn()
            started = time.perf_counter()
            try:
                content = self.delegate.complete_text(system=system, user=user)
            except urllib.error.HTTPError as exc:
                self.budget.calls.append({
                    "http_attempt": self.budget.http_attempts, "error": "HTTPError",
                    "http_status": exc.code, "latency_s": round(time.perf_counter()-started, 3),
                })
                if exc.code in (429, 502, 503) and retry == 0:
                    print(f"ANONYMOUS_RETRY status={exc.code}", flush=True)
                    continue
                raise
            except Exception as exc:
                self.budget.calls.append({
                    "http_attempt": self.budget.http_attempts, "error": type(exc).__name__,
                    "latency_s": round(time.perf_counter()-started, 3),
                })
                raise
            self.budget.calls.append({
                "http_attempt": self.budget.http_attempts,
                "latency_s": round(time.perf_counter()-started, 3),
                "status": "completed_validated_ovh_identity",
                "usage": self.delegate.last_usage,
            })
            return content
        raise RuntimeError("retry limit exceeded")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    reference_path = Path("runs/bird-paired-dev-v5/paired.json")
    source_hash = sha256(source.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA:
        raise ValueError(f"pinned source mismatch: {source_hash}")
    original = json.loads(reference_path.read_text("utf-8"))
    if original.get("status") != "complete" or original.get("source_sha256") != SOURCE_SHA:
        raise ValueError("reference source/status mismatch")
    if original.get("split_manifest_sha256") != REFERENCE_SHA or len(original["cases"]) != 100:
        raise ValueError("v5 split identity/count mismatch")
    selected = original["cases"][:len(CASE_IDS)]
    if tuple(str(row["case_id"]) for row in selected) != CASE_IDS:
        raise ValueError("frozen first-six case IDs have drifted")
    rows = json.loads(source.read_text("utf-8"))
    by_id = {str(row.get("question_id") or row.get("id") or i): row for i, row in enumerate(rows)}
    for old in selected:
        x = by_id[str(old["case_id"])]
        if x["question"] != old["question"] or str(x["db_id"]) != str(old["database_id"]):
            raise ValueError(f"case identity drift: {old['case_id']}")

    manifest = {
        "protocol_version": 1, "source_sha256": SOURCE_SHA,
        "seed": "first-six-frozen-v5-preregistered-20261122",
        "dev_case_ids": list(CASE_IDS), "final_case_ids": [],
    }
    manifest_path = output_dir / "frozen-six-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("FROZEN_PRECHECK_PASS cases=" + ",".join(CASE_IDS), flush=True)

    base = "http://127.0.0.1:3001"
    session = request_json(base + "/api/auth/setup", {
        "email": "evaluation@example.invalid", "password": "temporary-eval-only-credential",
    })
    api_key = request_json(base + "/api/settings/api-key", token=session["token"])["apiKey"]
    if not isinstance(api_key, str) or not api_key.startswith("freellmapi-"):
        raise ValueError("FreeLLMAPI unified local key unavailable")
    qwen_ready = False
    for check in range(25):
        ready = request_json(base + "/v1/models?execution_status=ready", token=api_key)
        qwen_ready = any(str(x.get("id", "")).casefold() == "qwen3.5-9b" for x in ready.get("data", []))
        if qwen_ready:
            break
        if check % 5 == 0:
            print(f"WAIT_ANONYMOUS_CATALOG check={check + 1} ready_entries={len(ready.get('data', []))}", flush=True)
        time.sleep(3)
    if not qwen_ready:
        raise ValueError("Qwen3.5-9B absent after 75-second anonymous catalog synchronization window")

    budget = CallBudget()
    output_file = output_dir / "paired-six.json"
    evaluation_status = "not_started"
    failure = None
    try:
        result = run_paired_dev(
            source_path=source, database_root=Path(args.database_root),
            manifest_path=manifest_path, output_path=output_file,
            model="qwen3.5-9b@ovh-anonymous", base_url=base + "/v1",
            provider_factory=lambda: MeteredProvider(api_key, budget),
            timeout_s=95, max_repairs=MAX_REPAIRS, max_rows=50_000,
        )
        evaluation_status = result["status"]
    except Exception as exc:
        evaluation_status = "incomplete"
        failure = type(exc).__name__  # no untrusted error messages or credentials
        print("PAIRED_PILOT_INCOMPLETE exception_type=" + failure, flush=True)

    measured = {}
    if output_file.exists():
        current = json.loads(output_file.read_text("utf-8"))
        recorded = current.get("cases", [])
        for case in recorded:
            old = next(row for row in selected if row["case_id"] == case["case_id"])
            measured[case["case_id"]] = {
                "database_id": case["database_id"],
                "ovh_direct_ex": case["direct"]["official_ex"],
                "ovh_guarded_ex": case["guarded"]["official_ex"],
                "ovh_direct_exec": case["direct"]["execution_success"],
                "ovh_guarded_exec": case["guarded"]["execution_success"],
                "ovh_calls": case["guarded"]["model_calls"],
                "ovh_guarded_latency_s": round(case["guarded"]["latency_s"], 3),
                "historical_4b_direct_ex": old["direct"]["official_ex"],
                "historical_4b_guarded_ex": old["guarded"]["official_ex"],
            }
        if current.get("status") != evaluation_status:
            evaluation_status = current.get("status", "incomplete")

    valid_usage = [r["usage"] for r in budget.calls if isinstance(r.get("usage"), dict)]
    report = {
        "study": "first-six-historically-exposed-v5-BIRD-DEV-diagnostic",
        "status": evaluation_status,
        "not_holding_out": True,
        "source_sha256": SOURCE_SHA,
        "reference_sha256": REFERENCE_SHA,
        "case_ids_preregistered": list(CASE_IDS),
        "completed_cases": len(measured),
        "backend": "FreeLLMAPI pinned SHA 7ad9097; anonymous OVH Qwen3.5-9B; reasoning_effort none",
        "original_4b_history": "context-only, different model; not causal comparator",
        "original_9b_100_summary": "not paired here; quantization identity unverified",
        "calls_attempted": budget.http_attempts,
        "calls_succeeded": sum(r.get("status") == "completed_validated_ovh_identity" for r in budget.calls),
        "provider_usage_reports": len(valid_usage),
        "provider_reported_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in valid_usage) if valid_usage else None,
        "provider_reported_completion_tokens": sum(r.get("completion_tokens", 0) for r in valid_usage) if valid_usage else None,
        "calls": budget.calls,
        "cases": measured,
        "metrics": current.get("metrics") if output_file.exists() and evaluation_status == "complete" else None,
        "failure_type": failure,
    }
    (output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PILOT_SUMMARY " + json.dumps({
        k: report[k] for k in ("status", "completed_cases", "calls_attempted",
                              "calls_succeeded", "provider_usage_reports", "metrics", "failure_type")
    }, ensure_ascii=False), flush=True)
    return 0 if evaluation_status == "complete" and len(measured) == len(CASE_IDS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
