"""Data-driven 'why this pick' explanations (at most two lines per pick)."""
from modelshift.recommend import build_recommendation
from modelshift.schemas import CandidateVerdict, CompatLabel, VerdictBand


def _v(model, mc, cpk, p50, mstok, hb=0.0, ratio=1.0, ok=20, n=22, elig=True, reasons=None):
    return CandidateVerdict(
        model=model, migration_confidence=mc, verdict_band=VerdictBand.SAFE_DROP_IN, hard_break_rate=hb,
        scored=n, ok_calls=n, distribution={CompatLabel.COMPATIBLE: ok, CompatLabel.COMPATIBLE_WITH_DRIFT: n - ok},
        cost_per_request=cpk / 1000, cost_per_1k_requests=cpk, latency_p50_ms=p50, ms_per_output_token_p50=mstok,
        latency_samples=n, latency_low_confidence=False, output_token_ratio=ratio,
        eligible=elig, ineligible_reasons=reasons or [])


def test_explanations_are_data_driven_and_short():
    vs = [_v("luna", 94.5, 0.73, 1314, 6.3, hb=0.0155, ratio=0.97),
          _v("sol", 93.9, 13.19, 1540, 10.3),
          _v("fast", 92.9, 1.04, 2240, 5.5, ratio=1.85)]
    rec = build_recommendation(vs)
    ex = rec.explanations
    assert all(1 <= len(lines) <= 2 for lines in ex.values())
    bo = " ".join(ex["best_overall"])
    assert "highest confidence (94.5)" in bo and "lowest cost ($0.73" in bo and "median 1.3 s" in bo
    assert "18× cheaper than sol" in " ".join(ex["most_economical"])
    lat = " ".join(ex["lowest_latency"])
    assert "5.5 ms/token" in lat and "slower because answers are 1.85× longer" in lat   # per-request caveat


def test_no_slower_caveat_when_not_true_and_tie_wording():
    vs = [_v("a", 90.0, 1.0, 1000, 8.0), _v("b", 90.0, 2.0, 900, 7.0)]
    ex = build_recommendation(vs).explanations
    assert "slower" not in " ".join(ex.get("lowest_latency", []))
    assert "tied with" in " ".join(ex["best_quality"])


def test_none_eligible_explains_closest():
    vs = [_v("x", 85.5, 1.0, 1000, 8.0, hb=0.09, elig=False, reasons=["hard-break rate 9% > 5%"])]
    rec = build_recommendation(vs)
    assert rec.none_eligible and "Closest by fit is x" in rec.explanations["none"][0]


def test_broken_data_never_breaks_recommendation(monkeypatch):
    import modelshift.explain as e
    monkeypatch.setattr(e, "explain", lambda *a, **k: 1 / 0)
    rec = build_recommendation([_v("a", 90.0, 1.0, 1000, 8.0)])
    assert rec.explanations == {} and rec.profiles[0].model == "a"
