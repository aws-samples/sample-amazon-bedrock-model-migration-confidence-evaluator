"""Phase 2 (multi-objective recommendation): per-candidate metrics and the quality gate."""
import json

import pytest

from modelshift import config as cfg
from modelshift.aggregate import aggregate, eligibility
from modelshift.schemas import (
    CandidateResult,
    CandidateVerdict,
    CompatLabel,
    Golden,
    GoldenKind,
    Message,
    NormalizedSample,
    PairScore,
    SampleRequest,
    SampleSource,
    Shape,
    Usage,
    VerdictBand,
)

M = "gpt-5.6-luna"


def _sample(sid, legacy="gpt-5.4", prompt=1000, completion=100, spend=None):
    return NormalizedSample(
        id=sid, source=SampleSource(shape=Shape.CHAT_COMPLETIONS), legacy_model=legacy,
        request=SampleRequest(input_messages=[Message(role="user", content="q")]),
        golden=Golden(kind=GoldenKind.TEXT, text="a"),
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion, spend=spend),
        evaluable=True,
    )


def _res(sid, cost=0.01, latency=1000, out=200, reasoning=50, status="ok", cached=False,
         known=True, model=M):
    return CandidateResult(sample_id=sid, model=model, cost_estimate=cost, cost_known=known,
                           latency_ms=latency, status=status, from_cache=cached,
                           usage=Usage(prompt_tokens=1000, completion_tokens=out,
                                       reasoning_tokens=reasoning))


def _ps(sid, label=CompatLabel.COMPATIBLE, hard=False, model=M):
    return PairScore(sample_id=sid, model=model, label=label, hard_break=hard)


def _settings(**kw):
    s = cfg.Settings()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# ------------------------------- cost metrics ------------------------------ #
def test_cost_per_request_excludes_errored_calls():
    results = [_res("a", 0.01), _res("b", 0.03), _res("c", 0.0, status="error")]
    v = aggregate(M, [_ps("a"), _ps("b")], results)
    assert v.ok_calls == 2
    assert v.cost_per_request == pytest.approx(0.02)
    assert v.cost_per_1k_requests == pytest.approx(20.0)
    assert v.error_rate == pytest.approx(1 / 3, abs=1e-4)


def test_unknown_price_gives_no_cost_metrics():
    v = aggregate(M, [_ps("a")], [_res("a", 0.0, known=False)], monthly_volume=1000)
    assert v.cost_known is False
    assert v.cost_per_request is None and v.cost_per_1k_requests is None
    assert v.projected_monthly_cost is None and v.cost_vs_current_pct is None


def test_projected_monthly_cost_only_when_volume_set():
    v = aggregate(M, [_ps("a")], [_res("a", 0.002)])
    assert v.projected_monthly_cost is None
    v = aggregate(M, [_ps("a")], [_res("a", 0.002)], monthly_volume=500_000)
    assert v.projected_monthly_cost == pytest.approx(1000.0)


def test_cost_vs_current_uses_logged_spend():
    samples = [_sample("a", spend=0.04), _sample("b", spend=0.06)]
    v = aggregate(M, [_ps("a"), _ps("b")], [_res("a", 0.01), _res("b", 0.04)], samples=samples)
    # 0.05 vs 0.10 -> -50%
    assert v.cost_vs_current_pct == pytest.approx(-50.0)


def test_cost_vs_current_uses_legacy_list_price_when_no_spend():
    # legacy gpt-5.4: 1000 in * 0.00275/1K + 100 out * 0.0165/1K = 0.00275 + 0.00165 = 0.0044
    samples = [_sample("a", legacy="gpt-5.4")]
    v = aggregate(M, [_ps("a")], [_res("a", 0.0022)], samples=samples)
    assert v.cost_vs_current_pct == pytest.approx(-50.0)


def test_cost_vs_current_none_when_any_baseline_unknown():
    samples = [_sample("a", legacy="gpt-4.1"), _sample("b", spend=0.05)]  # gpt-4.1 not priced
    v = aggregate(M, [_ps("a"), _ps("b")], [_res("a"), _res("b")], samples=samples)
    assert v.cost_vs_current_pct is None


def test_cost_vs_current_mixed_baselines_per_sample():
    samples = [_sample("a", legacy="gpt-5.4"), _sample("b", spend=0.0044)]
    v = aggregate(M, [_ps("a"), _ps("b")], [_res("a", 0.0044), _res("b", 0.0044)], samples=samples)
    assert v.cost_vs_current_pct == pytest.approx(0.0)


# ------------------------------- token metrics ----------------------------- #
def test_token_averages_and_output_ratio():
    samples = [_sample("a", completion=100), _sample("b", completion=200)]
    results = [_res("a", out=200, reasoning=50), _res("b", out=200, reasoning=20)]
    v = aggregate(M, [_ps("a"), _ps("b")], results, samples=samples)
    assert v.avg_output_tokens == pytest.approx(200.0)       # reasoning is inside output
    assert v.avg_reasoning_tokens == pytest.approx(35.0)
    assert v.output_token_ratio == pytest.approx((2.0 + 1.0) / 2)


def test_output_ratio_none_without_golden_usage():
    s = _sample("a")
    s.usage.completion_tokens = None
    v = aggregate(M, [_ps("a")], [_res("a")], samples=[s])
    assert v.output_token_ratio is None


# ------------------------------- latency metrics --------------------------- #
def test_ms_per_output_token_median_and_samples():
    results = [_res("a", latency=1000, out=100), _res("b", latency=4000, out=200),
               _res("c", latency=9000, out=300),
               _res("d", latency=1, out=1, cached=True),          # cached: excluded
               _res("e", latency=99999, out=10, status="error")]  # failed: excluded
    v = aggregate(M, [_ps(x) for x in "abcd"], results, settings=_settings(min_latency_samples=3))
    # per-token: 10, 20, 30 -> median 20
    assert v.ms_per_output_token_p50 == pytest.approx(20.0)
    assert v.latency_samples == 3
    assert v.latency_low_confidence is False
    assert v.latency_avg_ms == pytest.approx(14000 / 3, abs=0.1)


def test_latency_low_confidence_below_threshold():
    v = aggregate(M, [_ps("a")], [_res("a")], settings=_settings(min_latency_samples=5))
    assert v.latency_samples == 1 and v.latency_low_confidence is True


# --------------------------------- eligibility ----------------------------- #
def _verdict(**kw):
    base = dict(model=M, migration_confidence=90.0, verdict_band=VerdictBand.SAFE_DROP_IN,
                hard_break_rate=0.0, error_rate=0.0, scored=10)
    base.update(kw)
    return CandidateVerdict(**base)


def test_eligible_when_all_gates_pass():
    assert eligibility(_verdict()) == (True, [])


@pytest.mark.parametrize("kw,reason", [
    (dict(verdict_band=VerdictBand.NOT_A_DROP_IN), "Not a drop-in"),
    (dict(hard_break_rate=0.10), "hard-break rate 10% > 5%"),
    (dict(migration_confidence=60.0), "confidence 60 < 65"),
    (dict(error_rate=0.20), "error rate 20% > 5%"),
])
def test_each_gate_fails_independently(kw, reason):
    ok, reasons = eligibility(_verdict(**kw))
    assert ok is False
    assert len(reasons) == 1 and reason in reasons[0]


def test_no_scored_evaluations_ineligible():
    v = aggregate(M, [], [_res("a", status="error")])
    assert v.eligible is False and v.ineligible_reasons == ["no scored evaluations"]
    assert v.error_rate == 1.0


def test_thresholds_come_from_settings():
    ok, _ = eligibility(_verdict(migration_confidence=60.0), _settings(min_confidence=50.0))
    assert ok is True


def test_aggregate_sets_eligibility():
    good = aggregate(M, [_ps(str(i)) for i in range(10)], [_res(str(i)) for i in range(10)])
    assert good.eligible is True
    bad = aggregate(M, [_ps(str(i), hard=i < 3) for i in range(10)],
                    [_res(str(i)) for i in range(10)])
    assert bad.eligible is False and any("hard-break" in r for r in bad.ineligible_reasons)


# ----------------------------------- settings ------------------------------ #
def test_default_presets_are_data_and_sum_to_one():
    s = cfg.Settings()
    assert set(s.weight_presets) == {"balanced", "quality_first", "cost_first", "latency_first"}
    for w in s.weight_presets.values():
        assert sum(w.values()) == pytest.approx(1.0)
    assert s.weight_presets["balanced"] == {"quality": 0.5, "cost": 0.3, "latency": 0.2}
    assert "indicative only" in s.latency_disclaimer


def test_recommendation_settings_persist_and_validate(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_SETTINGS_PATH", str(tmp_path / "s.json"))
    s = cfg.Settings()
    s.min_confidence = 70.0
    s.quality_tolerance = 5.0
    s.latency_metric = "p50"
    s.weight_presets["balanced"] = {"quality": 0.6, "cost": 0.2, "latency": 0.2}
    cfg.save_settings(s)
    s2 = cfg.load_settings(cfg.Settings())
    assert s2.min_confidence == 70.0 and s2.quality_tolerance == 5.0 and s2.latency_metric == "p50"
    assert s2.weight_presets["balanced"]["quality"] == 0.6
    # garbage is ignored
    (tmp_path / "s.json").write_text(json.dumps({"latency_metric": "bogus", "min_confidence": "x",
                                                 "min_latency_samples": True}))
    s3 = cfg.load_settings(cfg.Settings())
    assert s3.latency_metric == "per_token" and s3.min_confidence == 65.0
    assert s3.min_latency_samples == 5


def test_api_settings_roundtrip_recommendation(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from modelshift import api as api_mod
    monkeypatch.setenv("MODELSHIFT_SETTINGS_PATH", str(tmp_path / "s.json"))
    c = TestClient(api_mod.app)
    before = c.get("/api/v1/settings").json()
    assert before["recommendation"]["min_confidence"] == cfg.DEFAULT_SETTINGS.min_confidence
    assert "indicative only" in before["latency_disclaimer"]
    old = cfg.DEFAULT_SETTINGS.max_hard_break_rate
    try:
        r = c.put("/api/v1/settings", json={"recommendation": {"max_hard_break_rate": 0.1}})
        assert r.status_code == 200
        assert c.get("/api/v1/settings").json()["recommendation"]["max_hard_break_rate"] == 0.1
    finally:
        cfg.DEFAULT_SETTINGS.max_hard_break_rate = old


def test_monthly_volume_roundtrips_in_run_config():
    from modelshift.orchestrator import CandidateConfig, RunConfig
    from modelshift.store import _config_from_dict, _config_to_dict
    rc = RunConfig(candidates=[CandidateConfig(model=M)], monthly_volume=250_000)
    assert _config_from_dict(_config_to_dict(rc)).monthly_volume == 250_000
    d = _config_to_dict(rc)
    d.pop("monthly_volume")
    assert _config_from_dict(d).monthly_volume is None
