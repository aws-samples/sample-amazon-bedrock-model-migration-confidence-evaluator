"""Phase 1 (multi-objective recommendation): trustworthy cost and latency."""
import json

import pytest

from modelshift import adapters as adapters_mod
from modelshift import config as cfg
from modelshift.adapters import BedrockResponsesAdapter
from modelshift.aggregate import aggregate
from modelshift.schemas import (
    CandidateResult,
    CompatLabel,
    Golden,
    GoldenKind,
    Message,
    NormalizedSample,
    PairScore,
    SampleRequest,
    SampleSource,
    Shape,
)


def _sample(sid="s1"):
    return NormalizedSample(
        id=sid, source=SampleSource(shape=Shape.CHAT_COMPLETIONS), legacy_model="gpt-4.1",
        request=SampleRequest(input_messages=[Message(role="user", content="What is 2+2?")]),
        golden=Golden(kind=GoldenKind.TEXT, text="4"), evaluable=True,
    )


class _FakeClock:
    """time.time() replacement; sleeps and transport calls advance it."""

    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


# ------------------------------ 1.1 latency ------------------------------ #
def test_latency_excludes_retry_backoff_and_counts_attempts(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(adapters_mod.time, "time", clock.time)
    calls = {"n": 0}

    def _t(payload):
        calls["n"] += 1
        clock.t += 0.5  # every attempt takes 500 ms
        if calls["n"] <= 2:
            raise RuntimeError("ThrottlingException: rate exceeded")
        return {"output_text": "4", "usage": {"input_tokens": 10, "output_tokens": 5}}

    a = BedrockResponsesAdapter(transport=_t, max_retries=3, sleep=clock.sleep)
    r = a.generate(_sample(), "gpt-5.6-luna")
    assert r.status == "ok"
    assert r.attempts == 3
    # 2 failed attempts (1 s) + back-off sleeps 1 s + 2 s = 4 s of waiting excluded.
    assert r.latency_ms == 500


def test_first_try_success_is_one_attempt():
    r = BedrockResponsesAdapter(
        transport=lambda p: {"output_text": "4", "usage": {"input_tokens": 1, "output_tokens": 1}}
    ).generate(_sample(), "gpt-5.6-luna")
    assert r.attempts == 1 and r.from_cache is False


# ------------------------------ 1.2 cache flag ---------------------------- #
def test_cache_hit_is_flagged_and_original_is_not():
    a = BedrockResponsesAdapter(transport=lambda p: {"output_text": "4", "usage": {}})
    first = a.generate(_sample(), "gpt-5.6-luna")
    second = a.generate(_sample(), "gpt-5.6-luna")
    assert first.from_cache is False
    assert second.from_cache is True
    assert a.generate(_sample(), "gpt-5.6-luna").from_cache is True
    # the stored entry itself is never mutated
    assert first.from_cache is False


def test_aggregate_excludes_cached_results_from_latency():
    ps = [PairScore(sample_id=f"s{i}", model="m", label=CompatLabel.COMPATIBLE) for i in range(3)]
    results = [
        CandidateResult(sample_id="s0", model="m", latency_ms=100),
        CandidateResult(sample_id="s1", model="m", latency_ms=300),
        CandidateResult(sample_id="s2", model="m", latency_ms=5, from_cache=True),
    ]
    v = aggregate("m", ps, results)
    assert v.latency_avg_ms == 200.0


# ---------------------------- 1.3 unknown price --------------------------- #
def test_unknown_price_is_none_not_zero():
    assert cfg.estimate_cost("some-unpriced-model", 1000, 1000) is None


def test_unknown_price_result_and_verdict_flagged():
    r = BedrockResponsesAdapter(
        transport=lambda p: {"output_text": "4", "usage": {"input_tokens": 100, "output_tokens": 50}}
    ).generate(_sample(), "some-unpriced-model")
    assert r.status == "ok" and r.cost_known is False and r.cost_estimate == 0.0
    v = aggregate("some-unpriced-model",
                  [PairScore(sample_id="s1", model="some-unpriced-model", label=CompatLabel.COMPATIBLE)],
                  [r])
    assert v.cost_known is False


def test_known_price_result_flagged_known():
    r = BedrockResponsesAdapter(
        transport=lambda p: {"output_text": "4", "usage": {"input_tokens": 1000, "output_tokens": 1000}}
    ).generate(_sample(), "gpt-5.5")
    assert r.cost_known is True
    assert abs(r.cost_estimate - (0.0055 + 0.033)) < 1e-9


@pytest.mark.parametrize("ref", ["gpt-5.5", "us.openai.gpt-5.5", "openai.gpt-5.5",
                                 "bedrock/converse/us.openai.gpt-5.5", "global.openai.gpt-5.5"])
def test_price_key_resolves_aliases(ref):
    assert cfg.price_key_for(ref) == "gpt-5.5"
    assert cfg.estimate_cost(ref, 1000, 0) == pytest.approx(0.0055)


def test_price_key_resolves_catalog_ids():
    assert cfg.price_key_for("us.openai.gpt-5.6-sol") == "gpt-5.6-sol"
    assert cfg.price_key_for("us.anthropic.claude-sonnet-4-5-20250929-v1:0") == "claude-sonnet-4.5"


# ---------------- 1.4b Settings price table is the live source ------------- #
def test_settings_price_edit_changes_cost(monkeypatch):
    s = cfg.Settings()
    s.price_table["gpt-5.6-luna"] = cfg.ModelPrice(input_per_1k=1.0, output_per_1k=1.0)
    monkeypatch.setattr(cfg, "DEFAULT_SETTINGS", s)
    assert cfg.estimate_cost("gpt-5.6-luna", 1000, 0) == pytest.approx(1.0)


def _load(tmp_path, monkeypatch, data):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(data))
    monkeypatch.setenv("MODELSHIFT_SETTINGS_PATH", str(p))
    return cfg.load_settings(cfg.Settings())


def test_legacy_full_price_table_does_not_pin_stale_defaults(tmp_path, monkeypatch):
    legacy = {"price_table": {
        "gpt-5.6-sol": {"input_per_1k": 0.005, "output_per_1k": 0.03},       # old default -> drop
        "gpt-5.6-luna": {"input_per_1k": 0.00022, "output_per_1k": 0.00132},  # same as now -> drop
        "gpt-5.4": {"input_per_1k": 0.009, "output_per_1k": 0.009},           # user edit -> keep
    }}
    s = _load(tmp_path, monkeypatch, legacy)
    assert s.price_table["gpt-5.6-sol"].input_per_1k == pytest.approx(0.0044)
    assert s.price_table["gpt-5.6-sol"].output_per_1k == pytest.approx(0.022)
    assert s.price_table["gpt-5.4"].input_per_1k == pytest.approx(0.009)
    assert s.price_table["gpt-5.4"].cache_read_per_1k == pytest.approx(0.000275)  # kept from public
    assert "gpt-5.5" in s.price_table


def test_save_writes_only_overrides(tmp_path, monkeypatch):
    p = tmp_path / "settings.json"
    monkeypatch.setenv("MODELSHIFT_SETTINGS_PATH", str(p))
    s = cfg.Settings()
    s.price_table["gpt-5.4"] = cfg.ModelPrice(input_per_1k=0.009, output_per_1k=0.009)
    cfg.save_settings(s)
    data = json.loads(p.read_text())
    assert "price_table" not in data
    assert list(data["price_overrides"]) == ["gpt-5.4"]
    s2 = cfg.load_settings(cfg.Settings())
    assert s2.price_table["gpt-5.4"].input_per_1k == pytest.approx(0.009)
    assert s2.price_table["gpt-5.6-sol"].input_per_1k == pytest.approx(0.0044)


# ------------------------------ 1.5 versioning ----------------------------- #
def test_metrics_version_new_runs_2_old_runs_1():
    from modelshift.orchestrator import Run
    from modelshift.store import _run_from_dict, _run_to_dict
    run = Run(run_id="r1", name="n")
    assert run.metrics_version == 2
    d = _run_to_dict(run)
    assert d["metrics_version"] == 2
    d.pop("metrics_version")
    assert _run_from_dict(d).metrics_version == 1


def test_old_result_json_without_new_fields_loads():
    r = CandidateResult(**{"sample_id": "s", "model": "m", "latency_ms": 10, "cost_estimate": 0.1})
    assert r.attempts == 1 and r.from_cache is False and r.cost_known is True
