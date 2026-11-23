"""Coordinator invariants: safe default, exact Qwen/BIRD, dedupe, budgets."""
from __future__ import annotations

import base64
import json

import pytest

from scripts import bird_hybrid_scheduler as s


def backlog():
    return {"protocol": s.PROTOCOL, "benchmark": "BIRD", "fixed_model": s.MODEL,
            "protected_data": {"reserved_BIRD_HOLDOUT300": True,
                               "remaining_two_TRAIN_DBs": True},
            "experiments": [{
                "id":"physical-evidence-causal-ablation-v1",
                "status":"design_required", "reason":"Needs independent proof"
            }]}


def ready_job():
    return {
        "id":"bird-physical-null-a-b-001",
        "status":"approved",
        "benchmark":"BIRD", "fixed_model":s.MODEL,
        "ref":"main", "requires_protected_holdout":False,
        "workflow":".github/workflows/bird-research-approved-physical-ablation.yml",
        "max_model_calls":64, "max_wall_minutes":120,
        "frozen_source_sha256":"a"*64,
        "metric_gate":{"metric":"strict_set_ex","min_net_correct":2,"max_regressions":1},
        "no_gold_in_model_or_training":True
    }


class FakeGithub:
    def __init__(self, active=None, seen=None, worker_ok=True):
        self.active=active or []
        self.seen=seen or []
        self.worker_ok=worker_ok
        self.calls=[]

    def request(self, endpoint, method="GET", payload=None):
        self.calls.append((endpoint, method, payload))
        if endpoint.startswith("contents/"):
            source=(
                "# BIRD_APPROVED_EXP_HANDLER_V1\n"
                "workflow_dispatch:\n"
                "run-name: BIRD-APPROVED:${{ inputs.experiment_id }}\n"
                "inputs.experiment_id inputs.max_model_calls\n"
                "inputs.max_wall_minutes inputs.frozen_source_sha256\n"
                "qwen3.5:9b-q4_K_M BIRD\n"
            ) if self.worker_ok else "workflow_dispatch:\n"
            return {"content":base64.b64encode(source.encode()).decode()}
        if "/runs?event=workflow_dispatch" in endpoint:
            return {"workflow_runs":self.seen}
        if endpoint.startswith("actions/runs?status="):
            return {"workflow_runs":self.active}
        if endpoint.endswith("/dispatches") and method=="POST":
            return None
        raise AssertionError(f"unexpected GitHub API call: {endpoint}")


def test_inert_without_preapproved_research(tmp_path):
    file=tmp_path/"plan.json"
    file.write_text(json.dumps(backlog()))
    item=s.load_plan(file)
    assert s.execute(item, FakeGithub()) == [{
        "id":"physical-evidence-causal-ablation-v1",
        "state":"NEEDS_RESEARCH", "reason":"Needs independent proof"
    }]


@pytest.mark.parametrize("override", [
    {"fixed_model":"qwen2.5:7b"}, {"benchmark":"Spider"},
    {"protected_data":{"reserved_BIRD_HOLDOUT300":False,
                       "remaining_two_TRAIN_DBs":True}}
])
def test_root_contract_is_fail_closed(tmp_path,override):
    p=backlog();p.update(override)
    file=tmp_path/"plan.json";file.write_text(json.dumps(p))
    with pytest.raises(s.Blocked):
        s.load_plan(file)


@pytest.mark.parametrize("change", [
    {"fixed_model":"some-teacher"}, {"benchmark":"Spider"},
    {"requires_protected_holdout":True}, {"ref":"research/unsafe"},
    {"workflow":".github/workflows/qwen9b-paired-frozen100.yml"},
    {"max_model_calls":129}, {"max_wall_minutes":241},
    {"frozen_source_sha256":"not-sha"},
    {"metric_gate":{"metric":"execution_success","min_net_correct":2,"max_regressions":1}},
    {"metric_gate":{"metric":"strict_set_ex","min_net_correct":0,"max_regressions":0}},
    {"no_gold_in_model_or_training":False}
])
def test_ready_experiment_budget_and_semantic_gates(change):
    d=ready_job();d.update(change)
    with pytest.raises(s.Blocked):
        s.verify_ready(d)


def test_blocked_if_worker_contract_unavailable():
    plan=backlog();plan["experiments"]=[ready_job()]
    fake=FakeGithub(worker_ok=False)
    r=s.execute(plan,fake)
    assert r[0]["state"]=="BLOCKED"
    assert not any(method=="POST" for _,method,_ in fake.calls)


def test_one_approved_job_starts_exactly_once():
    plan=backlog();plan["experiments"]=[ready_job()]
    fake=FakeGithub()
    results=s.execute(plan,fake)
    assert results==[{"id":ready_job()["id"],"state":"DISPATCHED_ONCE"}]
    posts=[(url,args) for url,meth,args in fake.calls if meth=="POST"]
    assert len(posts)==1
    assert posts[0][1]["ref"]=="main"
    assert posts[0][1]["inputs"]["max_model_calls"]=="64"
    assert posts[0][1]["inputs"]["frozen_source_sha256"]=="a"*64


def test_previously_completed_run_is_not_dispatched_again():
    plan=backlog();plan["experiments"]=[ready_job()]
    fake=FakeGithub(seen=[{"id":8001,"status":"completed",
         "display_title":s.RUN_PREFIX+ready_job()["id"]}])
    out=s.execute(plan,fake)
    assert out[0]["state"]=="ALREADY_STARTED:8001:completed"
    assert not any(m=="POST" for _,m,_ in fake.calls)


def test_running_qwen_blocks_new_experiment():
    plan=backlog();plan["experiments"]=[ready_job()]
    fake=FakeGithub(active=[{"name":"PROOF BIRD TRAIN Qwen9B paired", "id":100}])
    assert s.execute(plan,fake)[0]["state"]=="BLOCKED_ACTIVE_QWEN"
    assert not any(m=="POST" for _,m,_ in fake.calls)


def test_dry_run_never_dispatches():
    plan=backlog();plan["experiments"]=[ready_job()]
    fake=FakeGithub()
    assert s.execute(plan,fake,dry_run=True)[0]["state"]=="DRY_RUN_READY"
    assert not any(m=="POST" for _,m,_ in fake.calls)


def test_rejects_parallel_approved_runs():
    plan=backlog()
    other=ready_job();other["id"]="bird-physical-null-a-b-002"
    plan["experiments"]=[ready_job(),other]
    with pytest.raises(s.Blocked):
        s.execute(plan,FakeGithub())


def test_duplicate_ids_are_forbidden(tmp_path):
    p=backlog();p["experiments"].append(p["experiments"][0].copy())
    file=tmp_path/"plan.json";file.write_text(json.dumps(p))
    with pytest.raises(s.Blocked):
        s.load_plan(file)


class RunningResearchGitHub(FakeGithub):
    def __init__(self, *, status="in_progress", conclusion=None, branch=None):
        super().__init__()
        self.run = {
            "id": 45100836322,
            "name": "BIRD TRAIN19 original Qwen9B COUNT-grain LangGraph paired Strict EX",
            "head_branch": branch or "research/bird-count-grain-witness-prototype-20261123",
            "status": status,
            "conclusion": conclusion,
        }

    def request(self, endpoint, method="GET", payload=None):
        if endpoint=="actions/runs/45100836322":
            self.calls.append((endpoint, method, payload))
            return self.run
        return super().request(endpoint, method, payload)


def test_existing_real_bird_work_is_monitored_not_misreported_as_idle():
    plan=backlog()
    plan["experiments"]=[{
        "id":"count-grain-witness-train19-v1",
        "status":"running_research",
        "ref":"research/bird-count-grain-witness-prototype-20261123",
        "github_run_id":45100836322,
    }]
    fake=RunningResearchGitHub()
    status=s.execute(plan,fake)
    assert status[0]["state"]=="RUNNING_REAL_EXPERIMENT"
    assert status[0]["run_id"]==45100836322
    assert not any(method=="POST" for _,method,_ in fake.calls)
    completed=RunningResearchGitHub(status="completed",conclusion="success")
    assert s.execute(plan,completed)[0]["state"]=="COMPLETED_REQUIRES_STRICT_EX_SCORE_REVIEW"
    failed=RunningResearchGitHub(status="completed",conclusion="failure")
    assert s.execute(plan,failed)[0]["state"]=="FAILED_REQUIRES_TOOLING_DIAGNOSIS"


def test_running_bird_identity_mismatch_fails_closed():
    plan=backlog()
    plan["experiments"]=[{
        "id":"count-grain-witness-train19-v1",
        "status":"running_research",
        "ref":"research/bird-count-grain-witness-prototype-20261123",
        "github_run_id":45100836322,
    }]
    bad=RunningResearchGitHub(branch="main")
    assert s.execute(plan,bad)[0]["state"]=="BLOCKED"
    plan["experiments"][0]["github_run_id"]=None
    assert s.execute(plan,bad)[0]["state"]=="BLOCKED_INVALID_RUNNING_RUN_ID"
