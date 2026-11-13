"""Runs list summary: date, evaluations, Best overall, spend, errors, live progress."""
import time

from modelshift import api as api_mod
from modelshift.orchestrator import ProgressEvent, Run, RunStatus
from modelshift.store import _run_from_dict, _run_to_dict
from test_reasoning_efforts import _client


def test_created_at_persists_and_old_runs_fall_back():
    r = Run(run_id="run_x", name="x")
    assert abs(r.created_at - time.time()) < 5
    assert _run_from_dict(_run_to_dict(r)).created_at == r.created_at
    old = _run_to_dict(r)
    old.pop("created_at")
    old["events"] = [{"type": "plan", "data": {}, "ts": 1234.5}]
    assert _run_from_dict(old).created_at == 1234.5           # first event
    old["events"] = []
    assert _run_from_dict(old, created_fallback=99.0).created_at == 99.0   # file mtime


def test_runs_list_summarises_a_finished_run(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    api_mod.configure_transports(
        candidate_transport=lambda p: {"output_text": "q", "usage": {"input_tokens": 10, "output_tokens": 2}},
        judge_transport=None)
    body = {"candidates": [{"model": "gpt-5.6-luna", "reasoning_effort": "low"}], "use_judge": False}
    assert c.post(f"/api/v1/runs/{rid}/launch", json=body).status_code == 202
    for _ in range(100):
        if c.get(f"/api/v1/runs/{rid}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    row = next(x for x in c.get("/api/v1/runs").json() if x["run_id"] == rid)
    assert row["status"] == "done" and row["evaluations"] == 1 and row["candidates_planned"] == 1
    assert row["created_at"] > 0 and row["progress"] is None and row["error"] is None
    assert row["spend"] is not None and row["spend"] > 0 and row["spend_unknown"] is False
    v = row["verdict_summary"][0]
    assert v["model"] == "gpt-5.6-luna@low" and "eligible" in v
    assert row["none_eligible"] == (row["best_overall"] is None)


def test_runs_list_shows_progress_for_a_live_run(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    r = api_mod._RUNS[rid]
    r.status = RunStatus.REPLAYING
    r.events = [ProgressEvent("plan", {"estimated_calls": 10}, ts=100.0),
                ProgressEvent("progress", {"done": 4, "total": 10}, ts=120.0)]
    row = next(x for x in c.get("/api/v1/runs").json() if x["run_id"] == rid)
    p = row["progress"]
    assert {k: p[k] for k in ("done", "total", "started_ts", "last_ts", "stage")} == \
        {"done": 4, "total": 10, "started_ts": 100.0, "last_ts": 120.0, "stage": "replaying"}
    assert p["inflight"] == 0 and p["throttled"] is False and p["parallelism"] >= 1
    assert row["best_overall"] is None and row["spend"] is None


def test_runs_list_survives_a_run_whose_recommendation_breaks(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    r = api_mod._RUNS[rid]
    r.status = RunStatus.DONE
    r.verdicts = ["not a verdict"]  # summarising this run raises
    resp = c.get("/api/v1/runs")
    assert resp.status_code == 200
    row = next(x for x in resp.json() if x["run_id"] == rid)
    assert row["verdict_summary"] == [] and row["error"] == "This run's summary could not be read."
