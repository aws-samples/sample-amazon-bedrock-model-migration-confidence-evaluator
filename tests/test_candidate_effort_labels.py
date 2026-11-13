"""Candidate identity = model + reasoning effort (spec Decision 6 / task 2.9)."""
import time

import pytest

from modelshift import config as cfg
from modelshift.orchestrator import CandidateConfig
from modelshift.schemas import ReasoningEffort


def test_label_and_base_model():
    c = CandidateConfig(model="gpt-5.6-luna", reasoning_effort=ReasoningEffort.LOW)
    assert c.label == "gpt-5.6-luna@low"
    assert cfg.base_model(c.label) == "gpt-5.6-luna"
    assert cfg.base_model("gpt-5.6-luna") == "gpt-5.6-luna"   # old runs: unchanged
    assert c.matches("gpt-5.6-luna@low") and c.matches("gpt-5.6-luna")
    assert not c.matches("gpt-5.6-luna@high")


def test_label_prices_like_the_base_model():
    assert cfg.estimate_cost("gpt-5.6-sol@high", 1000, 1000) == pytest.approx(0.0044 + 0.022)


def _client():
    from fastapi.testclient import TestClient
    from modelshift import api as api_mod
    return TestClient(api_mod.app)


def _ingest(c, run_id, n=3):
    import json
    rows = [{"request_id": f"r{i}", "model": "gpt-4.1", "messages": {},
             "proxy_server_request": {"model": "gpt-4.1",
                                      "messages": [{"role": "user", "content": f"q{i}"}]},
             "response": {"choices": [{"message": {"role": "assistant", "content": f"a{i}"}}]}}
            for i in range(n)]
    up = c.post("/api/v1/uploads", files={"file": ("r.json", json.dumps({"data": rows}).encode(),
                                                   "application/json")}).json()
    c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})


def _wait(c, run_id):
    for _ in range(100):
        st = c.get(f"/api/v1/runs/{run_id}").json()
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_same_model_two_efforts_are_two_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "efforts"}).json()["run_id"]
    _ingest(c, run_id, n=3)
    plan = c.post(f"/api/v1/runs/{run_id}/plan", json={"candidates": [
        {"model": "gpt-5.6-luna", "reasoning_effort": "low"},
        {"model": "gpt-5.6-luna", "reasoning_effort": "high"},
        {"model": "gpt-5.6-luna", "reasoning_effort": "high"},   # exact duplicate -> dropped
    ]}).json()
    assert plan["estimated_calls"] == 6
    assert sorted(p["model"] for p in plan["per_candidate"]) == ["gpt-5.6-luna@high", "gpt-5.6-luna@low"]
    c.post(f"/api/v1/runs/{run_id}/launch", json={"candidates": [
        {"model": "gpt-5.6-luna", "reasoning_effort": "low"},
        {"model": "gpt-5.6-luna", "reasoning_effort": "high"}]})
    _wait(c, run_id)
    from modelshift import api as api_mod
    run = api_mod._RUNS[run_id]
    assert sorted(v.model for v in run.verdicts) == ["gpt-5.6-luna@high", "gpt-5.6-luna@low"]
    for v in run.verdicts:
        assert v.scored == 3                         # not merged: 3 each, not 6 in one
    assert sorted({r.model for r in run.results}) == ["gpt-5.6-luna@high", "gpt-5.6-luna@low"]
    by_label = {r.model: r.settings.reasoning_effort.value for r in run.results}
    assert by_label == {"gpt-5.6-luna@low": "low", "gpt-5.6-luna@high": "high"}
    # each candidate's own evaluations
    s_low = c.get(f"/api/v1/runs/{run_id}/samples?candidate=gpt-5.6-luna@low").json()
    assert s_low["total"] == 3
    # a bare model name is ambiguous here -> matches nothing rather than mixing efforts
    assert c.get(f"/api/v1/runs/{run_id}/samples?candidate=gpt-5.6-luna").json()["total"] == 0


def test_bare_model_name_resolves_when_unambiguous(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "single"}).json()["run_id"]
    _ingest(c, run_id, n=2)
    c.post(f"/api/v1/runs/{run_id}/launch", json={"candidates": [{"model": "gpt-5.6-terra"}]})
    _wait(c, run_id)
    assert c.get(f"/api/v1/runs/{run_id}/samples?candidate=gpt-5.6-terra").json()["total"] == 2
    assert c.get(f"/api/v1/runs/{run_id}/samples?candidate=gpt-5.6-terra@medium").json()["total"] == 2


def test_old_run_with_bare_names_still_resolves():
    from modelshift import api as api_mod
    from modelshift.orchestrator import Run, RunConfig
    from modelshift.schemas import CandidateResult
    run = Run(run_id="old", name="old")
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")])
    run.results = [CandidateResult(sample_id="s", model="gpt-5.6-luna")]  # pre-label run
    assert api_mod._resolve_candidate(run, "gpt-5.6-luna") == "gpt-5.6-luna"
    assert api_mod._candidate_effort(run, "gpt-5.6-luna") == ReasoningEffort.MEDIUM
