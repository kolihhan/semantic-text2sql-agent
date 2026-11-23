"""5-minute coordinator: BIRD-only Qwen3.5-9B, preregistered experiments only.

Does not train, infer, use benchmark gold, or automatically promote research
results. A future approved worker must itself enforce the declared call budget.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

PROTOCOL = "BIRD-LANGGRAPH-HYBRID-SCHEDULER-V1"
MODEL = "qwen3.5:9b-q4_K_M"
PREFIX = ".github/workflows/bird-research-approved-"
RUN_PREFIX = "BIRD-APPROVED:"
PAGE_SIZE = 100


class Blocked(RuntimeError):
    pass


class GitHub:
    def __init__(self, repo: str, token: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise Blocked("invalid repository")
        self.repo, self.token = repo, token

    def request(self, endpoint: str, method: str = "GET", payload: dict | None = None):
        url = "https://api.github.com/repos/" + self.repo + "/" + endpoint
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28",
                   "Authorization": "Bearer " + self.token}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            with urlopen(Request(url, data=body, headers=headers, method=method), timeout=12) as response:
                raw = response.read()
                return json.loads(raw.decode("utf-8")) if raw else None
        except HTTPError as exc:
            raise Blocked(f"GitHub API HTTP {exc.code}, no dispatch") from exc
        except (URLError, TimeoutError, ValueError) as exc:
            raise Blocked("GitHub API unavailable or malformed, no dispatch") from exc


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("protocol") != PROTOCOL or plan.get("benchmark") != "BIRD"
            or plan.get("fixed_model") != MODEL
            or plan.get("protected_data", {}).get("reserved_BIRD_HOLDOUT300") is not True
            or plan.get("protected_data", {}).get("remaining_two_TRAIN_DBs") is not True):
        raise Blocked("invalid BIRD/model/protected-data contract")
    jobs = plan.get("experiments")
    if not isinstance(jobs, list) or any(not isinstance(j, dict) for j in jobs):
        raise Blocked("invalid experiment list")
    ids = [j.get("id") for j in jobs]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise Blocked("duplicate/missing experiment id")
    return plan


def verify_ready(job: dict) -> None:
    if job.get("status") != "approved":
        raise Blocked("not approved")
    if job.get("benchmark") != "BIRD" or job.get("fixed_model") != MODEL:
        raise Blocked("not BIRD or original fixed Qwen3.5-9B")
    if job.get("ref") != "main" or job.get("requires_protected_holdout") is not False:
        raise Blocked("unreviewed ref or protected holdout")
    identifier = job.get("id")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", identifier or "") is None:
        raise Blocked("invalid experiment id")
    workflow = job.get("workflow")
    if (not isinstance(workflow, str) or not workflow.startswith(PREFIX)
            or not workflow.endswith(".yml") or "/" in workflow[len(PREFIX):]):
        raise Blocked("worker is not on reviewed BIRD-only allowlist")
    calls, mins = job.get("max_model_calls"), job.get("max_wall_minutes")
    if type(calls) is not int or not 0 < calls <= 128 or type(mins) is not int or not 0 < mins <= 240:
        raise Blocked("Qwen call or runtime budget missing/excessive")
    if re.fullmatch(r"[0-9a-f]{64}", job.get("frozen_source_sha256", "")) is None:
        raise Blocked("missing frozen gold-blind BIRD source SHA")
    gate = job.get("metric_gate")
    if (not isinstance(gate, dict) or gate.get("metric") != "strict_set_ex"
            or type(gate.get("min_net_correct")) is not int or gate["min_net_correct"] < 1
            or type(gate.get("max_regressions")) is not int or gate["max_regressions"] < 0
            or job.get("no_gold_in_model_or_training") is not True):
        raise Blocked("strict EX gate or label-leakage protection missing")


def validate_worker(client: GitHub, name: str) -> None:
    doc = client.request("contents/" + quote(name, safe="/") + "?ref=main")
    try:
        content = base64.b64decode(doc["content"]).decode("utf-8")
    except (TypeError, KeyError, ValueError) as exc:
        raise Blocked("reviewed worker cannot be read") from exc
    for marker in ("# BIRD_APPROVED_EXP_HANDLER_V1", "workflow_dispatch:",
                   "run-name:", RUN_PREFIX, "inputs.experiment_id",
                   "inputs.max_model_calls", "inputs.max_wall_minutes",
                   "inputs.frozen_source_sha256", MODEL, "BIRD"):
        if marker not in content:
            raise Blocked("worker is missing mandatory BIRD/budget/identity contract: " + marker)


def find_previous_dispatch(client: GitHub, worker: str, identifier: str) -> str | None:
    filename = quote(worker.rsplit("/", 1)[-1])
    for page in range(1, 6):
        result = client.request(
            f"actions/workflows/{filename}/runs?event=workflow_dispatch&per_page={PAGE_SIZE}&page={page}")
        runs = result.get("workflow_runs", [])
        for run in runs:
            if str(run.get("display_title", "")).startswith(RUN_PREFIX + identifier):
                return f"ALREADY_STARTED:{run['id']}:{run.get('status')}"
        if len(runs) < PAGE_SIZE:
            return None
    raise Blocked("workflow dispatch history too large to deduplicate safely")


def other_qwen_active(client: GitHub) -> bool:
    for state in ("queued", "in_progress"):
        data = client.request(f"actions/runs?status={state}&per_page=100")
        for run in data.get("workflow_runs", []):
            name = str(run.get("name", "")).lower()
            if "bird" in name and any(k in name for k in ("qwen", "langgraph", "bird train", "bird dev")):
                return True
    return False


def execute(plan: dict, client: GitHub, dry_run: bool = False) -> list[dict]:
    """Allow at most one approved research experiment, ever launched once."""
    jobs = plan["experiments"]
    if sum(x.get("status") == "approved" for x in jobs) > 1:
        raise Blocked("parallel Qwen experiments forbidden")
    out = []
    for job in jobs:
        identifier = job["id"]
        if job.get("status") == "running_research":
            # This is an already dispatched user-authorized research branch run.
            # Monitor it instead of incorrectly calling all running work "idle".
            run_id = job.get("github_run_id")
            if type(run_id) is not int or run_id < 1:
                out.append({"id": identifier, "state": "BLOCKED_INVALID_RUNNING_RUN_ID"})
                continue
            try:
                run = client.request(f"actions/runs/{run_id}")
                if (run.get("id") != run_id
                        or run.get("head_branch") != job.get("ref")
                        or "BIRD" not in str(run.get("name","")).upper()):
                    raise Blocked("active run identity / branch mismatch")
                status, conclusion = run.get("status"), run.get("conclusion")
                state = ("RUNNING_REAL_EXPERIMENT" if status in ("queued","in_progress")
                         else "COMPLETED_REQUIRES_STRICT_EX_SCORE_REVIEW" if status=="completed" and conclusion=="success"
                         else "FAILED_REQUIRES_TOOLING_DIAGNOSIS")
                out.append({"id":identifier, "state":state,
                            "run_id":run_id,"status":status,"conclusion":conclusion})
            except Blocked as exc:
                out.append({"id":identifier, "state":"BLOCKED", "reason":str(exc)})
            continue
        if job.get("status") != "approved":
            out.append({"id": identifier, "state": "NEEDS_RESEARCH", "reason": job.get("reason","not approved")})
            continue
        try:
            verify_ready(job)
            validate_worker(client, job["workflow"])
            previous = find_previous_dispatch(client, job["workflow"], identifier)
            if previous:
                out.append({"id": identifier, "state": previous})
            elif other_qwen_active(client):
                out.append({"id": identifier, "state": "BLOCKED_ACTIVE_QWEN"})
            elif dry_run:
                out.append({"id": identifier, "state": "DRY_RUN_READY"})
            else:
                client.request("actions/workflows/" + quote(job["workflow"].rsplit("/", 1)[-1]) +
                               "/dispatches", method="POST", payload={
                    "ref": "main",
                    "inputs": {
                        "experiment_id": identifier,
                        "frozen_source_sha256": job["frozen_source_sha256"],
                        "max_model_calls": str(job["max_model_calls"]),
                        "max_wall_minutes": str(job["max_wall_minutes"]),
                    }})
                out.append({"id": identifier, "state": "DISPATCHED_ONCE"})
        except Blocked as exc:
            out.append({"id": identifier, "state": "BLOCKED", "reason": str(exc)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("research/bird-hybrid-plan.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        plan = load_plan(args.plan)
        token = os.environ.get("GH_TOKEN", "")
        if not token and any(x.get("status") == "approved" for x in plan["experiments"]):
            raise Blocked("token required to inspect approved job")
        client = GitHub(os.environ.get("GITHUB_REPOSITORY", "kolihhan/semantic-text2sql-agent"), token)
        result = {"protocol": PROTOCOL, "model": MODEL, "benchmark": "BIRD",
                  "results": execute(plan, client, args.dry_run)}
    except (Blocked, ValueError, TypeError, KeyError) as exc:
        result = {"protocol": PROTOCOL, "state": "FAIL_CLOSED", "reason": str(exc)}
    print("BIRD_HYBRID_SCHEDULER " + json.dumps(result, sort_keys=True))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("BIRD-only research scheduler status\n\n")
            f.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 1 if result.get("state") == "FAIL_CLOSED" else 0


if __name__ == "__main__":
    sys.exit(main())
