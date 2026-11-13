"""Live run-progress events that drive the Running view (stages, per-call detail, retries)."""
import json
import time


from modelshift import api as api_mod
from modelshift.adapters import BedrockResponsesAdapter
from modelshift.schemas import CandidateSettings
from test_phase1_cost_latency import _sample
from test_reasoning_efforts import _client


def test_adapter_reports_each_retry_before_backing_off():
    calls = {"n": 0}

    def transport(payload):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("ThrottlingException: rate exceeded")
        return {"output_text": "4", "usage": {"input_tokens": 5, "output_tokens": 1}}

    seen = []
    a = BedrockResponsesAdapter(transport, sleep=lambda s: None, max_retries=3)
    a.on_retry = lambda attempt, mx, why, err: seen.append((attempt, mx, why))
    r = a.generate(_sample(), "gpt-5.6-luna@low", CandidateSettings())
    assert r.status == "ok" and r.attempts == 3
    assert seen == [(2, 4, "throttled"), (3, 4, "throttled")]


def test_broken_retry_observer_never_breaks_the_call():
    n = {"i": 0}

    def transport(payload):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("429 Too Many Requests")
        return {"output_text": "4", "usage": {}}

    a = BedrockResponsesAdapter(transport, sleep=lambda s: None)
    a.on_retry = lambda *a_: 1 / 0
    assert a.generate(_sample(), "gpt-5.6-luna@low", CandidateSettings()).status == "ok"


def _events(c, rid):
    for _ in range(100):
        if c.get(f"/api/v1/runs/{rid}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    body = c.get(f"/api/v1/runs/{rid}/stream").text
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(ln.split(": ", 1) for ln in block.split("\n") if ": " in ln)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_run_emits_stages_call_starts_and_rich_progress(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    api_mod.configure_transports(
        candidate_transport=lambda p: {"output_text": "q", "usage": {"input_tokens": 12, "output_tokens": 3}},
        judge_transport=None)
    body = {"candidates": [{"model": "gpt-5.6-luna", "reasoning_effort": "low"}], "use_judge": False}
    assert c.post(f"/api/v1/runs/{rid}/launch", json=body).status_code == 202
    ev = _events(c, rid)
    types = [t for t, _ in ev]
    assert types.index("plan") < types.index("stage") < types.index("call_start") < types.index("progress")
    assert [d["stage"] for t, d in ev if t == "stage"] == ["replay", "aggregate"]
    assert all("ts" in d for t, d in ev if t != "end")  # timestamps for elapsed/pace
    start = next(d for t, d in ev if t == "call_start")
    prog = next(d for t, d in ev if t == "progress")
    assert start["index"] == prog["index"] == 0 and start["sample_id"] == prog["sample_id"]
    for k in ("latency_ms", "attempts", "from_cache", "cost", "input_tokens", "output_tokens", "reasoning_tokens"):
        assert k in prog, k
    assert prog["attempts"] == 1 and prog["from_cache"] is False
    assert prog["input_tokens"] == 12 and prog["output_tokens"] == 3
    assert prog["cost"] is not None and prog["cost"] > 0   # Luna has a public list price
