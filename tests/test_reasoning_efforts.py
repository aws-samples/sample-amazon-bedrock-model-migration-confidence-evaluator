"""Per-model reasoning efforts (probed live against Bedrock, 2026-11-12)."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from modelshift import api as api_mod
from modelshift.config import MODEL_CATALOG, supported_efforts
from modelshift.remediate import lower_effort, validate_change
from modelshift.schemas import ReasoningEffort, RemediationChange


def test_catalog_efforts_match_bedrock_probes():
    for m in ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"):
        assert supported_efforts(m) == ["none", "low", "medium", "high", "xhigh"]
    assert supported_efforts("gpt-5.4") == ["none", "minimal", "low", "medium", "high", "xhigh"]
    assert supported_efforts("gpt-5.6-luna@low") == supported_efforts("gpt-5.6-luna")
    assert supported_efforts("not-in-catalog") is None
    assert MODEL_CATALOG["claude-sonnet-4.5"].effort_applies is False


def test_enum_has_none_and_xhigh():
    assert ReasoningEffort("none") and ReasoningEffort("xhigh")


def test_step_down_respects_model_and_never_picks_none():
    luna = supported_efforts("gpt-5.6-luna")
    assert lower_effort(ReasoningEffort.MEDIUM, luna) == ReasoningEffort.LOW
    assert lower_effort(ReasoningEffort.LOW, luna) is None          # skips minimal (unsupported) and none
    g54 = supported_efforts("gpt-5.4")
    assert lower_effort(ReasoningEffort.LOW, g54) == ReasoningEffort.MINIMAL
    assert lower_effort(ReasoningEffort.XHIGH, luna) == ReasoningEffort.HIGH


def test_validate_change_rejects_effort_model_does_not_accept():
    with pytest.raises(ValueError, match="for gpt-5.6-luna@medium"):
        validate_change(RemediationChange(type="reasoning_effort", after="minimal"), "gpt-5.6-luna@medium")
    assert validate_change(RemediationChange(type="reasoning_effort", after="minimal"), "gpt-5.4@medium")


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    api_mod._RUNS.clear()
    api_mod.configure_transports(candidate_transport=lambda p: {"output_text": "q", "usage": {}}, judge_transport=None)
    c = TestClient(api_mod.app)
    rid = c.post("/api/v1/runs", json={"name": "eff"}).json()["run_id"]
    rows = [{"request_id": "r", "model": "gpt-4.1", "messages": {},
             "proxy_server_request": {"model": "gpt-4.1", "messages": [{"role": "user", "content": "q"}]},
             "response": {"choices": [{"message": {"role": "assistant", "content": "q"}}]}}]
    payload = json.dumps({"data": rows}).encode()
    up = c.post("/api/v1/uploads", files={"file": ("r.json", payload, "application/json")}).json()
    c.post(f"/api/v1/runs/{rid}/ingest", json={"upload_id": up["upload_id"]})
    return c, rid


def test_plan_and_launch_reject_unsupported_effort_before_any_call(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    body = {"candidates": [{"model": "gpt-5.6-luna", "reasoning_effort": "minimal"}]}
    for path in ("plan", "launch"):
        r = c.post(f"/api/v1/runs/{rid}/{path}", json=body)
        assert r.status_code == 400, r.text
        err = r.json()["detail"]["error"]
        assert err["code"] == "unsupported_reasoning_effort"
        assert "none, low, medium, high, xhigh" in err["message"]


def test_new_efforts_run_end_to_end(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    # GPT-5.4 'minimal' (an effort GPT-5.6 rejects) now runs end to end on runtime too.
    p = c.post(f"/api/v1/runs/{rid}/plan", json={"candidates": [{"model": "gpt-5.4", "reasoning_effort": "minimal"}]})
    assert p.status_code == 200 and p.json()["per_candidate"][0]["model"] == "gpt-5.4@minimal"
    body = {"candidates": [{"model": "gpt-5.6-luna", "reasoning_effort": "none"},
                           {"model": "gpt-5.6-luna", "reasoning_effort": "xhigh"},
                           {"model": "gpt-5.4", "reasoning_effort": "minimal"}], "use_judge": False}
    assert c.post(f"/api/v1/runs/{rid}/launch", json=body).status_code == 202
    for _ in range(80):
        if c.get(f"/api/v1/runs/{rid}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    models = sorted(v["model"] for v in c.get(f"/api/v1/runs/{rid}/results").json()["candidates"])
    assert models == ["gpt-5.4@minimal", "gpt-5.6-luna@none", "gpt-5.6-luna@xhigh"]


def test_models_endpoint_exposes_efforts(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    m = c.get("/api/v1/models").json()["models"]
    assert m["gpt-5.6-luna"]["reasoning_efforts"] == ["none", "low", "medium", "high", "xhigh"]
    assert m["claude-sonnet-4.5"]["effort_applies"] is False
