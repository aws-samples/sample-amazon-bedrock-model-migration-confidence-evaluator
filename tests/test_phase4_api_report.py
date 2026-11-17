"""Phase 4 (multi-objective recommendation): API, persistence and JSON report."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from modelshift import api as api_mod
from modelshift import config as cfg


class _CountingTransport:
    """Echo transport that counts calls, with usage so costs and latency exist."""

    def __init__(self):
        self.calls = 0

    def __call__(self, payload):
        self.calls += 1
        text = ""
        for m in payload.get("input", []) or payload.get("messages", []) or []:
            c = m.get("content")
            text = c if isinstance(c, str) else (c[0].get("text", "") if c else "")
        return {"output_text": text, "usage": {"input_tokens": 100, "output_tokens": 20}}


def _client(tmp_path, monkeypatch, transport=None):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    api_mod._RUNS.clear()
    api_mod._UPLOADS.clear()
    api_mod.configure_transports(candidate_transport=transport or _CountingTransport(), judge_transport=None)
    return TestClient(api_mod.app)


def _finished_run(c, candidates, n=6, **launch):
    run_id = c.post("/api/v1/runs", json={"name": "p4"}).json()["run_id"]
    rows = [{"request_id": f"r{i}", "model": "gpt-4.1", "messages": {},
             "proxy_server_request": {"model": "gpt-4.1", "messages": [{"role": "user", "content": f"q{i}"}]},
             "response": {"choices": [{"message": {"role": "assistant", "content": f"q{i}"}}]}}
            for i in range(n)]
    up = c.post("/api/v1/uploads", files={"file": ("r.json", json.dumps({"data": rows}).encode(),
                                                   "application/json")}).json()
    c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})
    r = c.post(f"/api/v1/runs/{run_id}/launch", json={"candidates": candidates, "use_judge": False, **launch})
    assert r.status_code == 202, r.text
    for _ in range(100):
        if c.get(f"/api/v1/runs/{run_id}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert c.get(f"/api/v1/runs/{run_id}").json()["status"] == "done"
    return run_id


CANDS = [{"model": "gpt-5.6-luna"}, {"model": "gpt-5.6-terra"}]


def test_results_and_run_carry_recommendation(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    rid = _finished_run(c, CANDS)
    res = c.get(f"/api/v1/runs/{rid}/results").json()
    rec = res["recommendation"]
    assert rec and [p["kind"] for p in rec["profiles"]] == [
        "best_overall", "best_quality", "most_economical", "lowest_latency"]
    assert res["weights"]["preset"] == "balanced"
    assert "indicative only" in rec["latency_disclaimer"]
    # echo transport -> both exact matches; Luna is cheaper
    assert next(p for p in rec["profiles"] if p["kind"] == "most_economical")["model"] == "gpt-5.6-luna@medium"
    # new verdict metrics present on results candidates
    v0 = res["candidates"][0]
    assert "cost_per_request" in v0 and "eligible" in v0
    assert c.get(f"/api/v1/runs/{rid}").json()["recommendation"]["profiles"]


def test_changing_weights_never_calls_a_model(tmp_path, monkeypatch):
    t = _CountingTransport()
    c = _client(tmp_path, monkeypatch, t)
    rid = _finished_run(c, CANDS)
    calls = t.calls
    r = c.put(f"/api/v1/runs/{rid}/weights", json={"preset": "cost_first"})
    assert r.status_code == 200
    assert r.json()["weights"]["preset"] == "cost_first"
    assert r.json()["recommendation"]["weights"]["cost"] == pytest.approx(0.5)
    c.put(f"/api/v1/runs/{rid}/weights", json={"preset": "custom", "quality": 1, "cost": 1, "latency": 2})
    c.get(f"/api/v1/runs/{rid}/recommendation?preset=latency_first")
    assert t.calls == calls  # no model calls from any weight change or preview


def test_custom_weights_normalised_and_persisted(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    rid = _finished_run(c, CANDS)
    w = c.put(f"/api/v1/runs/{rid}/weights",
              json={"preset": "custom", "quality": 2, "cost": 1, "latency": 1}).json()["weights"]
    assert (w["quality"], w["cost"], w["latency"]) == (0.5, 0.25, 0.25)
    from modelshift.store import load_all_runs
    assert load_all_runs()[rid].weights.preset.value == "custom"


def test_preview_does_not_save(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    rid = _finished_run(c, CANDS)
    prev = c.get(f"/api/v1/runs/{rid}/recommendation?preset=latency_first").json()
    assert prev["weights"]["preset"] == "latency_first"
    assert c.get(f"/api/v1/runs/{rid}/results").json()["weights"]["preset"] == "balanced"


def test_weights_errors(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    rid = c.post("/api/v1/runs", json={"name": "not-done"}).json()["run_id"]
    assert c.put(f"/api/v1/runs/{rid}/weights", json={"preset": "balanced"}).status_code == 409
    assert c.get(f"/api/v1/runs/{rid}/recommendation").status_code == 409
    done = _finished_run(c, CANDS)
    r = c.put(f"/api/v1/runs/{done}/weights", json={"preset": "nope"})
    assert r.status_code == 400 and r.json()["detail"]["error"]["code"] == "invalid_weights"


def test_launch_with_weights(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    rid = _finished_run(c, CANDS, weights={"preset": "quality_first"})
    assert c.get(f"/api/v1/runs/{rid}/results").json()["weights"]["preset"] == "quality_first"


def test_named_preset_follows_settings_edits(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    rid = _finished_run(c, CANDS)
    old = dict(cfg.DEFAULT_SETTINGS.weight_presets["balanced"])
    try:
        cfg.DEFAULT_SETTINGS.weight_presets["balanced"] = {"quality": 0.6, "cost": 0.2, "latency": 0.2}
        w = c.get(f"/api/v1/runs/{rid}/results").json()["recommendation"]["weights"]
        assert w["quality"] == pytest.approx(0.6)
    finally:
        cfg.DEFAULT_SETTINGS.weight_presets["balanced"] = old


def test_threshold_change_applies_to_existing_run(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    rid = _finished_run(c, CANDS)
    old = cfg.DEFAULT_SETTINGS.min_confidence
    try:
        cfg.DEFAULT_SETTINGS.min_confidence = 101.0  # nothing can pass
        rec = c.get(f"/api/v1/runs/{rid}/results").json()["recommendation"]
        assert rec["none_eligible"] is True and rec["closest_by_fit"]
    finally:
        cfg.DEFAULT_SETTINGS.min_confidence = old


def test_old_run_json_loads_and_is_repriced(tmp_path, monkeypatch):
    """A run saved by an older build: bare model names, no new verdict fields, stale costs
    (old Sol price + reasoning billed twice), no weights, no metrics_version."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(runs))
    samples = [{"id": f"s{i}", "source": {"shape": "chat_completions"}, "legacy_model": "gpt-4.1",
                "request": {"input_messages": [{"role": "user", "content": "q"}]},
                "golden": {"kind": "text", "text": "a"}, "evaluable": True} for i in range(6)]
    results = [{"sample_id": f"s{i}", "model": "gpt-5.6-sol", "cost_estimate": 9.99, "latency_ms": 1000,
                "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "reasoning_tokens": 400}}
               for i in range(6)]
    scores = [{"sample_id": f"s{i}", "model": "gpt-5.6-sol", "label": "compatible"} for i in range(6)]
    old = {"run_id": "old1", "name": "legacy", "status": "done",
           "config": {"candidates": [{"model": "gpt-5.6-sol", "reasoning_effort": "medium"}]},
           "samples": samples, "results": results, "pair_scores": scores,
           "verdicts": [{"model": "gpt-5.6-sol", "migration_confidence": 100.0, "scored": 6}]}
    (runs / "old1.json").write_text(json.dumps(old))
    api_mod._RUNS.clear()
    api_mod._load_persisted_runs()
    c = TestClient(api_mod.app)
    res = c.get("/api/v1/runs/old1/results").json()
    v = res["candidates"][0]
    # 1000 in * $4.40/1M + 1000 out (reasoning included) * $22/1M = 0.0044 + 0.022
    assert v["cost_per_request"] == pytest.approx(0.0264)
    assert v["eligible"] is True
    rec = res["recommendation"]
    assert rec["profiles"][0]["model"] == "gpt-5.6-sol"
    assert any("older build" in n for n in rec["notes"])
    assert res["weights"]["preset"] == "balanced"
    # stored results untouched
    assert api_mod._RUNS["old1"].results[0].cost_estimate == 9.99


def test_report_keys_superset_and_best_is_fit(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    rid = _finished_run(c, CANDS)
    from modelshift.report import build_run_report
    rep = build_run_report(api_mod._RUNS[rid])
    legacy_keys = {"run_id", "name", "description", "status", "error", "source", "candidates", "best",
                   "remediations", "top_failure_reasons_overall", "caveats"}
    assert legacy_keys <= set(rep)
    assert {"recommendation", "weights", "latency_disclaimer"} <= set(rep)
    assert rep["best"]["model"] == rep["candidates"][0]["model"]  # best = best by fit
    assert any("no prompt caching" in x for x in rep["caveats"])
    assert any("indicative only" in x for x in rep["caveats"])
    legacy_cand = {"model", "endpoint", "migration_confidence", "verdict_band", "distribution",
                   "dimension_averages", "top_failure_reasons", "hard_break_rate", "cost_total",
                   "latency_avg_ms", "latency_p50_ms", "latency_p95_ms", "scored", "errored",
                   "errors", "evaluations"}
    assert legacy_cand <= set(rep["candidates"][0])
    assert {"cost_per_request", "eligible", "ms_per_output_token_p50"} <= set(rep["candidates"][0])
    # PDF still renders with the new payload
    pdf = c.get(f"/api/v1/runs/{rid}/report.pdf").content
    assert pdf[:4] == b"%PDF"


def test_weights_roundtrip_in_store():
    from modelshift.orchestrator import Run
    from modelshift.schemas import WeightPreset, Weights
    from modelshift.store import _run_from_dict, _run_to_dict
    r = Run(run_id="w", name="w")
    r.weights = Weights(preset=WeightPreset.COST_FIRST, quality=0.35, cost=0.5, latency=0.15)
    back = _run_from_dict(_run_to_dict(r))
    assert back.weights.preset == WeightPreset.COST_FIRST and back.weights.cost == 0.5
    d = _run_to_dict(r)
    d.pop("weights")
    assert _run_from_dict(d).weights.preset == WeightPreset.BALANCED
