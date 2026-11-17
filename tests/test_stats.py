"""Confidence intervals and statistical ties (modelshift/stats.py + recommend)."""
from modelshift.config import DEFAULT_RUBRIC, Settings
from modelshift.recommend import _attach_intervals, build_recommendation
from modelshift.schemas import CandidateVerdict, CompatLabel, PairScore
from modelshift.stats import quality_stats

C, D, X = CompatLabel.COMPATIBLE, CompatLabel.COMPATIBLE_WITH_DRIFT, CompatLabel.INCOMPATIBLE


def _scores(model, labels, replay=0):
    return [PairScore(sample_id=f"r{i}", model=model, label=lbl, replay_index=replay)
            for i, lbl in enumerate(labels)]


def _mc(labels):
    return round(100.0 * sum(DEFAULT_RUBRIC.confidence_weight(x) for x in labels) / len(labels), 1)


def test_interval_contains_estimate_and_narrows_with_more_requests():
    small = [C] * 8 + [D, X]
    large = small * 10
    s1 = quality_stats(_scores("a", small), ["a"], DEFAULT_RUBRIC)
    s2 = quality_stats(_scores("a", large), ["a"], DEFAULT_RUBRIC)
    lo1, hi1 = s1.ci["a"]
    lo2, hi2 = s2.ci["a"]
    assert lo1 <= _mc(small) <= hi1
    assert lo2 <= _mc(large) <= hi2
    assert (hi2 - lo2) < (hi1 - lo1)          # more requests -> tighter interval


def test_deterministic():
    ps = _scores("a", [C, D, C, X, C, C]) + _scores("b", [C, C, C, D, D, C])
    a = quality_stats(ps, ["a", "b"], DEFAULT_RUBRIC)
    # Bypass the cache: a fresh computation must give the same numbers (seeded by request ids).
    from modelshift import stats
    stats._CACHE.clear()
    b = quality_stats(ps, ["a", "b"], DEFAULT_RUBRIC)
    assert a.ci == b.ci and a.diff == b.diff


def test_identical_answers_are_tied_and_large_gap_is_not():
    base = [C] * 20
    ps = (_scores("lead", base) + _scores("same", base)
          + _scores("worse", [C] * 5 + [X] * 15))
    st = quality_stats(ps, ["lead", "same", "worse"], DEFAULT_RUBRIC)
    assert st.diff_ci("same", "lead") == (0.0, 0.0)
    assert st.tied("same", "lead")
    assert not st.tied("worse", "lead")


def test_paired_test_is_tighter_than_overlapping_intervals():
    # Both miss the same 4 requests; b also misses 6 more. Their separate intervals overlap,
    # but every discordant request goes the same way, so the paired difference excludes 0.
    lead = [X] * 4 + [C] * 56
    worse = [X] * 10 + [C] * 50
    st = quality_stats(_scores("a", lead) + _scores("b", worse), ["a", "b"], DEFAULT_RUBRIC)
    assert st.ci["b"][1] >= st.ci["a"][0]   # the two intervals overlap...
    assert not st.tied("b", "a")            # ...but the paired test separates them


def test_replays_resampled_together_per_request():
    labels = [C, D, X, C]
    ps = _scores("a", labels, 0) + _scores("a", labels, 1)
    st = quality_stats(ps, ["a"], DEFAULT_RUBRIC)
    assert st.requests == 4                 # 4 requests, not 8 independent answers


def test_too_few_requests_returns_none():
    assert quality_stats(_scores("a", [C]), ["a"], DEFAULT_RUBRIC) is None


def _verdict(model, mc, cost):
    return CandidateVerdict(model=model, migration_confidence=mc, eligible=True, scored=20,
                            cost_per_request=cost, latency_low_confidence=True)


def _with_intervals(verdicts, labels_by_model):
    ps = [p for m, labels in labels_by_model.items() for p in _scores(m, labels)]
    _attach_intervals(verdicts, ps, Settings())
    return verdicts


def test_tied_cheaper_candidate_gets_full_quality_score_and_wins():
    lead = [C] * 19 + [D]
    near = [C] * 18 + [D, D]       # one answer worse: statistically tied
    vs = _with_intervals([_verdict("pricey", _mc(lead), 0.010), _verdict("cheap", _mc(near), 0.001)],
                         {"pricey": lead, "cheap": near})
    rec = build_recommendation(vs, Settings())
    axes = {a.model: a for a in rec.axes}
    assert rec.quality_ties == ["pricey", "cheap"]
    assert axes["cheap"].quality_score == 100.0 and axes["cheap"].quality_tied
    assert axes["cheap"].quality_ci is not None
    assert rec.profile(rec.profiles[0].kind).model == "cheap"
    assert rec.profiles[1].model == "pricey"               # Best quality is still the top estimate
    assert "Statistically tied" in rec.profiles[1].reason
    assert any("Statistically tied on quality" in n for n in rec.notes)
    assert any("statistically tied" in line for line in rec.explanations["best_quality"])


def test_ties_off_restores_linear_quality_score():
    lead = [C] * 19 + [D]
    near = [C] * 18 + [D, D]
    vs = _with_intervals([_verdict("pricey", _mc(lead), 0.010), _verdict("cheap", _mc(near), 0.001)],
                         {"pricey": lead, "cheap": near})
    rec = build_recommendation(vs, Settings(statistical_ties=False))
    axes = {a.model: a for a in rec.axes}
    assert rec.quality_ties == [] and not rec.statistical_ties
    assert axes["cheap"].quality_score < 100.0


def test_tie_never_reaches_outside_quality_window():
    lead = [C] * 4
    far = [C, X, X, C]            # 4 requests: wide intervals, but 50 points behind
    vs = _with_intervals([_verdict("lead", _mc(lead), 0.01), _verdict("far", _mc(far), 0.001)],
                         {"lead": lead, "far": far})
    rec = build_recommendation(vs, Settings())
    assert "far" not in rec.quality_ties


def test_verdicts_without_intervals_behave_as_before():
    vs = [_verdict("a", 90.0, 0.01), _verdict("b", 88.0, 0.001)]
    rec = build_recommendation(vs, Settings())
    assert rec.quality_ties == []
    assert {a.model: a.quality_score for a in rec.axes}["b"] == 80.0
