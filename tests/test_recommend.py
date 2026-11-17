"""Phase 3 (multi-objective recommendation): the recommendation engine."""
import pytest

from modelshift import config as cfg
from modelshift.recommend import build_recommendation, weights_for
from modelshift.schemas import (
    CandidateVerdict,
    ProfileKind,
    VerdictBand,
    WeightPreset,
    Weights,
)


def V(model, conf=90.0, cpr=0.01, ms_tok=10.0, eligible=True, hb=0.0, low_lat=False, p50=1000.0,
      reasons=None):
    return CandidateVerdict(
        model=model, migration_confidence=conf, verdict_band=VerdictBand.SAFE_DROP_IN,
        hard_break_rate=hb, scored=20, cost_per_request=cpr, cost_known=cpr is not None,
        ms_per_output_token_p50=ms_tok, latency_p50_ms=p50, latency_low_confidence=low_lat,
        eligible=eligible, ineligible_reasons=reasons or ([] if eligible else ["confidence 50 < 65"]))


def P(rec, kind):
    return rec.profile(kind)


# ------------------------------- weights ----------------------------------- #
def test_presets_resolve_and_normalise():
    w = weights_for("balanced")
    assert (w.quality, w.cost, w.latency) == (0.5, 0.3, 0.2) and w.preset == WeightPreset.BALANCED
    assert weights_for("cost_first").cost == 0.5
    assert weights_for("nonsense").preset == WeightPreset.BALANCED


def test_custom_weights_normalise_and_quality_floor():
    w = weights_for("custom", custom=Weights(quality=0.0, cost=3.0, latency=1.0))
    assert w.preset == WeightPreset.CUSTOM
    assert w.quality == pytest.approx(0.2)
    assert w.cost + w.latency == pytest.approx(0.8)
    assert w.cost == pytest.approx(0.6)
    z = Weights(quality=0, cost=0, latency=0).normalised()
    assert (z.quality, z.cost, z.latency) == (0.5, 0.3, 0.2)


# ------------------------------- axis scores ------------------------------- #
def test_axis_scores_are_ratio_to_best():
    rec = build_recommendation([V("a", cpr=0.01, ms_tok=10), V("b", cpr=0.02, ms_tok=40)])
    ax = {a.model: a for a in rec.axes}
    assert ax["a"].cost == 100.0 and ax["b"].cost == 50.0
    assert ax["a"].latency == 100.0 and ax["b"].latency == 25.0
    assert ax["a"].quality == 90.0


def test_composite_uses_weights():
    rec = build_recommendation([V("a", conf=90, cpr=0.01, ms_tok=10)])
    a = rec.axes[0]
    assert a.quality == 90.0 and a.quality_score == 100.0   # best confidence -> 100
    assert a.composite == pytest.approx(100.0)


# --------------------------------- profiles -------------------------------- #
def _three():
    # sol: best quality, expensive, slow; terra: balanced; luna: cheap, fast, 6 pts lower
    return [V("sol", conf=95, cpr=0.040, ms_tok=30),
            V("terra", conf=90, cpr=0.012, ms_tok=12),
            V("luna", conf=86, cpr=0.002, ms_tok=6)]


def test_four_profiles_in_order():
    rec = build_recommendation(_three())
    assert [p.kind for p in rec.profiles] == [ProfileKind.BEST_OVERALL, ProfileKind.BEST_QUALITY,
                                              ProfileKind.MOST_ECONOMICAL, ProfileKind.LOWEST_LATENCY]
    assert P(rec, ProfileKind.BEST_QUALITY).model == "sol"
    assert P(rec, ProfileKind.MOST_ECONOMICAL).model == "luna"
    assert P(rec, ProfileKind.LOWEST_LATENCY).model == "luna"


def test_best_overall_follows_preset():
    vs = _three()
    assert P(build_recommendation(vs, weights=weights_for("quality_first")), ProfileKind.BEST_OVERALL).model == "sol"
    assert P(build_recommendation(vs, weights=weights_for("cost_first")), ProfileKind.BEST_OVERALL).model == "luna"
    assert P(build_recommendation(vs, weights=weights_for("latency_first")), ProfileKind.BEST_OVERALL).model == "luna"


def test_quality_first_is_not_outweighed_by_a_huge_price_gap():
    # Regression: with raw confidence in the composite, a 20x cheaper model 9 points lower won
    # even Quality-first. Quality is now scaled over the tolerance window.
    vs = [V("sol", conf=95, cpr=0.040, ms_tok=30), V("luna", conf=86, cpr=0.002, ms_tok=6)]
    assert P(build_recommendation(vs, weights=weights_for("quality_first")), ProfileKind.BEST_OVERALL).model == "sol"
    ax = {a.model: a for a in build_recommendation(vs).axes}
    assert ax["luna"].quality == 86.0 and ax["luna"].quality_score == pytest.approx(10.0)


def test_best_overall_stays_inside_quality_window():
    # luna is 11 points below sol: outside the 10-point window, so it can't be Best overall
    # even with cost-first weights -- consistent with it not qualifying as Most economical.
    vs = [V("sol", conf=95, cpr=0.040, ms_tok=30), V("luna", conf=84, cpr=0.0001, ms_tok=1)]
    rec = build_recommendation(vs, weights=weights_for("cost_first"))
    assert P(rec, ProfileKind.BEST_OVERALL).model == "sol"
    assert P(rec, ProfileKind.MOST_ECONOMICAL).model == "sol"


def test_quality_tolerance_excludes_cheap_but_far_worse():
    vs = [V("sol", conf=95, cpr=0.04), V("cheap", conf=80, cpr=0.001)]  # 15 pts below
    rec = build_recommendation(vs)
    me = P(rec, ProfileKind.MOST_ECONOMICAL)
    assert me.model == "sol"
    s = cfg.Settings()
    s.quality_tolerance = 20
    assert P(build_recommendation(vs, settings=s), ProfileKind.MOST_ECONOMICAL).model == "cheap"


def test_economical_none_when_no_prices():
    rec = build_recommendation([V("a", cpr=None), V("b", cpr=None)])
    me = P(rec, ProfileKind.MOST_ECONOMICAL)
    assert me.model is None and "configured price" in me.reason
    assert any("price not configured" in n for n in rec.notes)


def test_lowest_latency_skips_low_confidence_latency():
    rec = build_recommendation([V("a", ms_tok=50), V("b", ms_tok=5, low_lat=True)])
    assert P(rec, ProfileKind.LOWEST_LATENCY).model == "a"
    ax = {a.model: a for a in rec.axes}
    assert ax["b"].latency is None and ax["b"].complete is False


def test_lowest_latency_none_when_nothing_timed():
    rec = build_recommendation([V("a", low_lat=True), V("b", low_lat=True)])
    ll = P(rec, ProfileKind.LOWEST_LATENCY)
    assert ll.model is None and "timed calls" in ll.reason


def test_p50_latency_metric_option():
    s = cfg.Settings()
    s.latency_metric = "p50"
    rec = build_recommendation([V("a", ms_tok=5, p50=4000), V("b", ms_tok=50, p50=1000)], settings=s)
    assert P(rec, ProfileKind.LOWEST_LATENCY).model == "b"
    assert rec.latency_metric == "p50"


# --------------------------- missing-axis margin ---------------------------- #
def test_unknown_price_cannot_win_on_missing_axis():
    # 'free' has no price: its composite renormalises over quality+latency and looks higher,
    # but within the margin a complete candidate wins Best overall.
    vs = [V("free", conf=91, cpr=None, ms_tok=10), V("priced", conf=90, cpr=0.02, ms_tok=10)]
    rec = build_recommendation(vs)
    assert P(rec, ProfileKind.BEST_OVERALL).model == "priced"


def test_incomplete_candidate_wins_if_far_ahead():
    s = cfg.Settings()
    s.missing_axis_margin = 0.0
    vs = [V("free", conf=99, cpr=None, ms_tok=10), V("priced", conf=70, cpr=0.02, ms_tok=10)]
    assert P(build_recommendation(vs, settings=s), ProfileKind.BEST_OVERALL).model == "free"


def test_cost_weight_zero_lets_unpriced_win():
    vs = [V("free", conf=95, cpr=None), V("priced", conf=90, cpr=0.02)]
    w = weights_for("custom", custom=Weights(quality=0.8, cost=0.0, latency=0.2))
    assert P(build_recommendation(vs, weights=w), ProfileKind.BEST_OVERALL).model == "free"


# ---------------------------------- pareto --------------------------------- #
def test_pareto_marks_dominated():
    vs = [V("good", conf=90, cpr=0.01, ms_tok=10), V("worse", conf=85, cpr=0.02, ms_tok=20),
          V("tradeoff", conf=95, cpr=0.05, ms_tok=30)]
    ax = {a.model: a for a in build_recommendation(vs).axes}
    assert ax["good"].pareto_efficient and ax["tradeoff"].pareto_efficient
    assert ax["worse"].pareto_efficient is False and ax["worse"].dominated_by == "good"


# --------------------------------- trade-offs ------------------------------ #
def test_tradeoff_text_and_deltas():
    vs = [V("terra", conf=84, cpr=0.010, ms_tok=10), V("luna", conf=78, cpr=0.0038, ms_tok=10.2)]
    rec = build_recommendation(vs, weights=weights_for("quality_first"))
    assert P(rec, ProfileKind.BEST_OVERALL).model == "terra"
    me = P(rec, ProfileKind.MOST_ECONOMICAL)
    assert me.model == "luna"
    assert me.tradeoff == ("62% cheaper per request than terra, for 6 points less confidence "
                           "(78 vs 84) and similar latency.")
    assert me.deltas["quality_pts"] == -6.0
    assert me.deltas["cost_pct"] == pytest.approx(-0.62)


def test_tradeoff_multiplier_wording():
    vs = [V("terra", conf=90, cpr=0.01, ms_tok=10), V("sol", conf=94, cpr=0.023, ms_tok=10)]
    rec = build_recommendation(vs, weights=weights_for("cost_first"))
    bq = P(rec, ProfileKind.BEST_QUALITY)
    assert bq.model == "sol"
    assert bq.tradeoff == ("4 points more confidence (94 vs 90) than terra, at 2.3x the cost per "
                           "request and similar latency.")


def test_same_as_best_overall():
    rec = build_recommendation([V("only", conf=90)])
    for kind in (ProfileKind.BEST_QUALITY, ProfileKind.MOST_ECONOMICAL, ProfileKind.LOWEST_LATENCY):
        assert P(rec, kind).model == "only"
        assert P(rec, kind).tradeoff == "Same as Best overall."
    assert rec.notes[0] == "Only one candidate passed the quality gate."
    a = rec.axes[0]
    assert a.cost == 100.0 and a.latency == 100.0


# ------------------------------- gate handling ------------------------------ #
def test_none_eligible_names_no_model_and_explains():
    vs = [V("a", conf=60, eligible=False, reasons=["confidence 60 < 65"]),
          V("b", conf=55, eligible=False, reasons=["hard-break rate 12% > 5%"])]
    rec = build_recommendation(vs)
    assert rec.none_eligible is True and rec.closest_by_fit == "a"
    assert all(p.model is None for p in rec.profiles)
    assert "confidence 60 < 65" in rec.profiles[0].reason
    assert rec.ineligible == {"a": ["confidence 60 < 65"], "b": ["hard-break rate 12% > 5%"]}


def test_ineligible_candidates_never_recommended():
    vs = [V("cheap_but_broken", conf=60, cpr=0.0001, ms_tok=1, eligible=False), V("ok", conf=88)]
    rec = build_recommendation(vs)
    assert rec.eligible_models == ["ok"]
    assert all(p.model in (None, "ok") for p in rec.profiles)
    assert "cheap_but_broken" in rec.ineligible


def test_effort_variants_are_separate_candidates():
    vs = [V("gpt-5.6-luna@low", conf=82, cpr=0.001, ms_tok=4),
          V("gpt-5.6-luna@high", conf=91, cpr=0.004, ms_tok=12)]
    rec = build_recommendation(vs)
    assert P(rec, ProfileKind.BEST_QUALITY).model == "gpt-5.6-luna@high"
    assert P(rec, ProfileKind.MOST_ECONOMICAL).model == "gpt-5.6-luna@low"


def test_deterministic_tie_break():
    vs = [V("b", conf=90, cpr=0.01, ms_tok=10), V("a", conf=90, cpr=0.01, ms_tok=10)]
    rec = build_recommendation(vs)
    assert all(p.model == "a" for p in rec.profiles)


def test_old_build_note_and_disclaimer():
    rec = build_recommendation([V("a")], metrics_version=1)
    assert any("older build" in n for n in rec.notes)
    assert "indicative only" in rec.latency_disclaimer


def test_pure_function_does_not_mutate_verdicts():
    vs = _three()
    before = [v.model_dump() for v in vs]
    build_recommendation(vs)
    assert [v.model_dump() for v in vs] == before


def test_one_point_is_singular():
    vs = [V("terra", conf=89, cpr=0.006, ms_tok=16), V("fast", conf=88, cpr=0.011, ms_tok=11)]
    rec = build_recommendation(vs)
    ll = P(rec, ProfileKind.LOWEST_LATENCY)
    assert ll.model == "fast" and "1 point less confidence" in ll.tradeoff
