"""Request replays end to end: plan x N, real separate calls, normal scoring, D7, gates, re-tests."""
import itertools
import time

import pytest

from modelshift import api as api_mod
from modelshift.aggregate import aggregate, retest
from modelshift.config import DEFAULT_SETTINGS
from modelshift.schemas import (CompatLabel, PairScore, ReasoningEffort, Remediation,
                                RemediationChange, RemediationTarget)
from test_reasoning_efforts import _client

BODY = {"candidates": [{"model": "gpt-5.6-luna", "reasoning_effort": "low"}], "use_judge": False}


def _varying(texts):
    calls = {"n": 0}
    cyc = itertools.cycle(texts)

    def transport(payload):
        calls["n"] += 1
        return {"output_text": next(cyc), "usage": {"input_tokens": 10, "output_tokens": 3}}
    return transport, calls


def _run(c, rid, body):
    assert c.post(f"/api/v1/runs/{rid}/launch", json=body).status_code == 202
    for _ in range(100):
        if c.get(f"/api/v1/runs/{rid}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    return c.get(f"/api/v1/runs/{rid}/results").json()


def test_replays_validation_and_plan_multiplier(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    for bad in (0, 6):
        r = c.post(f"/api/v1/runs/{rid}/plan", json={**BODY, "replays": bad})
        assert r.status_code == 400 and r.json()["detail"]["error"]["code"] == "invalid_replays"
    one = c.post(f"/api/v1/runs/{rid}/plan", json=BODY).json()
    three = c.post(f"/api/v1/runs/{rid}/plan", json={**BODY, "replays": 3}).json()
    assert one["replays"] == 1 and three["replays"] == 3
    assert three["estimated_calls"] == 3 * one["estimated_calls"] == 3 * three["calls_without_replays"]
    assert three["est_cost"] == pytest.approx(3 * one["est_cost"], abs=1e-4)
    assert three["over_ceiling"] is False


def test_each_replay_is_a_real_call_scored_normally(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    transport, calls = _varying(["q", "q", "something completely different"])
    api_mod.configure_transports(candidate_transport=transport, judge_transport=None)
    res = _run(c, rid, {**BODY, "replays": 3})
    assert calls["n"] == 3                                     # not served from the cache
    v = res["candidates"][0]
    assert res["replays"] == 3 and v["replays"] == 3
    assert v["scored"] == 3 and v["requests_scored"] == 1      # answers are normal calls
    assert sum(v["distribution"].values()) == 3
    assert v["consistency"] is not None and v["consistency"] < 1.0
    assert "d7_consistency" in v["dimension_averages"]
    run = api_mod._RUNS[rid]
    assert sorted(r.replay_index for r in run.results) == [0, 1, 2]
    assert sorted(p.replay_index for p in run.pair_scores) == [0, 1, 2]
    # Migration Confidence is the plain formula over every answer.
    assert v["migration_confidence"] == aggregate(v["model"], run.pair_scores).migration_confidence
    # Events carry the replay number; progress total counts every replay.
    prog = [e.data for e in run.events if e.type == "progress"]
    assert [p["replay"] for p in prog] == [0, 1, 2] and prog[-1]["total"] == 3
    # Drill-downs: list carries D7, detail lists every answer, least-consistent endpoint.
    sid = run.samples[0].id
    items = c.get(f"/api/v1/runs/{rid}/samples", params={"candidate": v["model"]}).json()["items"]
    assert len(items) == 3 and items[0]["consistency"]["d7"] == v["consistency"]
    d = c.get(f"/api/v1/runs/{rid}/samples/{sid}", params={"candidate": v["model"], "replay": 2}).json()
    assert d["replay"] == 2 and len(d["answers"]) == 3 and d["consistency"]["answers"] == 3
    lc = c.get(f"/api/v1/runs/{rid}/candidates/{v['model']}/consistency").json()
    assert lc["measured"] == 1 and lc["items"][0]["sample_id"] == sid
    assert len(c.get(f"/api/v1/runs/{rid}/transcript", params={"candidate": v["model"]}).json()["items"]) == 3


def test_replays_off_is_unchanged(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    transport, calls = _varying(["q"])
    api_mod.configure_transports(candidate_transport=transport, judge_transport=None)
    v = _run(c, rid, BODY)["candidates"][0]
    assert calls["n"] == 1 and v["replays"] == 1 and v["consistency"] is None
    assert "d7_consistency" not in v["dimension_averages"]
    items = c.get(f"/api/v1/runs/{rid}/samples").json()["items"]
    assert items[0]["consistency"] is None


def test_consistency_gates_and_override_per_replay(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    transport, _ = _varying(["q", "totally unrelated words here"])
    api_mod.configure_transports(candidate_transport=transport, judge_transport=None)
    _run(c, rid, {**BODY, "replays": 2})
    run = api_mod._RUNS[rid]
    m = run.verdicts[0].model
    s = DEFAULT_SETTINGS.model_copy()
    s.min_consistency = 0.99
    v = aggregate(m, run.pair_scores, run.results, settings=s)
    assert any(r.startswith("consistency") for r in v.ineligible_reasons)
    s.min_consistency, s.max_flip_rate = None, None
    reasons = aggregate(m, run.pair_scores, run.results, settings=s).ineligible_reasons
    assert not any("consistency" in r for r in reasons)
    # Override relabels exactly one replay's answer.
    sid = run.samples[0].id
    r = c.patch(f"/api/v1/runs/{rid}/samples/{sid}", json={"candidate": m, "label": "incompatible", "replay_index": 1})
    assert r.status_code == 200
    flagged = [p.replay_index for p in run.pair_scores if p.overridden]
    assert flagged == [1]


def test_retest_compares_replay_by_replay():
    def ps(sid, k, label):
        return PairScore(sample_id=sid, model="m", label=label, replay_index=k)
    before = [ps("a", 0, CompatLabel.PARTIAL), ps("a", 1, CompatLabel.COMPATIBLE)]
    rem = Remediation(remediation_id="r1", run_id="x", candidate="m",
                      target=RemediationTarget(sample_ids=["a"]), change=RemediationChange(type="prompt_edit"))

    class S:  # minimal sample
        id = "a"
    out = retest("x", "m", rem, [S()], before,
                 lambda s, e, ch: [ps("a", 0, CompatLabel.COMPATIBLE), ps("a", 1, CompatLabel.INCOMPATIBLE)],
                 reasoning_effort=ReasoningEffort.LOW)
    assert out.result.moved["to_compatible"] == 1 and out.result.moved["regressions"] == 1
    # A single PairScore (old callers) is still accepted.
    out2 = retest("x", "m", rem, [S()], before, lambda s, e, ch: ps("a", 0, CompatLabel.COMPATIBLE),
                  reasoning_effort=ReasoningEffort.LOW)
    assert out2.result.moved["to_compatible"] == 1


def test_old_saved_config_loads_as_replays_off():
    from modelshift.store import _config_from_dict
    assert _config_from_dict({"candidates": []}).replays == 1
    assert _config_from_dict({"candidates": [], "replays": 3}).replays == 3


def test_replay_settings_apply_clamp_and_switch_off():
    from modelshift.config import _apply_recommendation_settings
    s = DEFAULT_SETTINGS.model_copy()
    _apply_recommendation_settings(s, {"default_replays": 9, "instability_threshold": 1.5,
                                       "min_consistency": 0.8, "max_flip_rate": 0.1, "replay_call_ceiling": 0})
    assert (s.default_replays, s.instability_threshold, s.min_consistency, s.max_flip_rate, s.replay_call_ceiling) \
        == (5, 1.0, 0.8, 0.1, 1)
    _apply_recommendation_settings(s, {"min_consistency": None, "max_flip_rate": None})
    assert s.min_consistency is None and s.max_flip_rate is None
