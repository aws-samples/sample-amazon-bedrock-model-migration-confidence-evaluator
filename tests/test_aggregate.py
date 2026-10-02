"""Stage 4 tests: judge, aggregation, remediation."""
from modelshift.aggregate import aggregate, recommend, retest, side_by_side
from modelshift.judge import Judge
from modelshift.scoring import score_pair
from modelshift.schemas import (
    CandidateOutput,
    CandidateResult,
    CompatLabel,
    DimensionScore,
    Golden,
    GoldenKind,
    Message,
    NormalizedSample,
    PairScore,
    SampleRequest,
    SampleSource,
    Shape,
    VerdictBand,
)


def _sample(sid="s1", prompt="Q?", golden="A", instructions=None):
    return NormalizedSample(
        id=sid, source=SampleSource(shape=Shape.CHAT_COMPLETIONS),
        legacy_model="gpt-4.1",
        request=SampleRequest(instructions=instructions,
                              input_messages=[Message(role="user", content=prompt)]),
        golden=Golden(kind=GoldenKind.TEXT, text=golden), evaluable=True,
    )


def _ps(sid, model, label, hard=False, reason=None, d1=0.9):
    return PairScore(
        sample_id=sid, model=model, label=label, hard_break=hard,
        d1_semantic=DimensionScore(score=d1), d2_format=DimensionScore(score=1.0),
        d3_factual=DimensionScore(score=1.0), d4_verbosity=DimensionScore(score=1.0),
        d5_instruction=DimensionScore(score=1.0), judge_reason=reason,
    )


# --------------------------------- judge ----------------------------------- #
def test_judge_steering_injected_into_prompt():
    from modelshift.judge import judge_prompt, Judge
    s = _sample(golden="x")
    out = CandidateOutput(text="totally different content here")
    p = judge_prompt(s, out, steering="Dosage changes are always Incompatible.")
    assert p["evaluation_steering"] == "Dosage changes are always Incompatible."
    # steering forwarded to the transport
    seen = {}

    def _t(prompt):
        seen["steer"] = prompt.get("evaluation_steering")
        return {"semantic": 0.5, "instruction": 1.0, "reason": "missing_content"}

    Judge(transport=_t, steering="follow my rubric").refine(s, out, score_pair(s, out, "m"))
    assert seen["steer"] == "follow my rubric"


def test_judge_captures_rationale():
    s = _sample(golden="The capital of France is Paris.")
    out = CandidateOutput(text="It's Paris, the French capital.")
    ps = score_pair(s, out, "m")
    j = Judge(transport=lambda p: {"semantic": 0.95, "instruction": 1.0,
                                   "reason": "equivalent", "rationale": "Both name Paris as the capital."})
    refined = j.refine(s, out, ps)
    assert refined.judge_rationale == "Both name Paris as the capital."


def test_judge_does_not_override_hard_break():
    s = _sample(golden="Call 1-800-555-0142")
    ps = score_pair(s, CandidateOutput(text="Call 1-866-606-3700"), "m")
    assert ps.hard_break
    j = Judge(transport=lambda p: {"semantic": 0.99, "instruction": 1.0, "reason": "equivalent"})
    out = j.refine(s, CandidateOutput(text="Call 1-866-606-3700"), ps)
    assert out.label == CompatLabel.INCOMPATIBLE  # judge cannot rescue a hard break


def test_judge_upgrades_semantic():
    s = _sample(golden="The capital of France is Paris.")
    out = CandidateOutput(text="It's Paris, the French capital, of course.")
    ps = score_pair(s, out, "m")
    # low lexical overlap deterministically; judge lifts semantic
    j = Judge(transport=lambda p: {"semantic": 0.95, "instruction": 1.0, "reason": "equivalent"})
    refined = j.refine(s, out, ps)
    assert refined.d1_semantic.score == 0.95
    assert refined.label in (CompatLabel.COMPATIBLE, CompatLabel.COMPATIBLE_WITH_DRIFT)


def test_judge_instruction_violation_downgrades():
    s = _sample(golden="ok", instructions="Answer in one word.")
    out = CandidateOutput(text="Sure, absolutely, here is a long-winded reply.")
    ps = score_pair(s, out, "m")
    j = Judge(transport=lambda p: {"semantic": 0.9, "instruction": 0.2, "reason": "instruction_violation"})
    refined = j.refine(s, out, ps)
    assert refined.d5_instruction.score == 0.2
    assert refined.label != CompatLabel.COMPATIBLE


def test_judge_caches():
    calls = {"n": 0}

    def _t(p):
        calls["n"] += 1
        return {"semantic": 0.7, "instruction": 1.0, "reason": "missing_content"}

    s = _sample(golden="a long golden answer about topic X")
    out = CandidateOutput(text="different words entirely here now")
    j = Judge(transport=_t)
    j.refine(s, out, score_pair(s, out, "m"))
    j.refine(s, out, score_pair(s, out, "m"))
    assert calls["n"] == 1


# ------------------------------ aggregation -------------------------------- #
def test_aggregate_confidence_and_band():
    scores = [
        _ps("1", "gpt-5.6-luna", CompatLabel.COMPATIBLE),
        _ps("2", "gpt-5.6-luna", CompatLabel.COMPATIBLE),
        _ps("3", "gpt-5.6-luna", CompatLabel.COMPATIBLE_WITH_DRIFT),
        _ps("4", "gpt-5.6-luna", CompatLabel.PARTIAL),
    ]
    v = aggregate("gpt-5.6-luna", scores)
    # weighted = 1 + 1 + 0.7 + 0.3 = 3.0 / 4 = 75
    assert v.migration_confidence == 75.0
    assert v.verdict_band == VerdictBand.DROP_IN_WITH_TUNING
    assert v.distribution[CompatLabel.COMPATIBLE] == 2
    assert v.scored == 4


def test_aggregate_high_hard_break_rate_caps_band():
    scores = [_ps(str(i), "m", CompatLabel.INCOMPATIBLE, hard=True) for i in range(5)]
    scores += [_ps(str(i), "m", CompatLabel.COMPATIBLE) for i in range(5, 10)]
    v = aggregate("m", scores)
    # 50% hard-break -> cannot be Safe
    assert v.verdict_band in (VerdictBand.NEEDS_WORK, VerdictBand.NOT_A_DROP_IN)
    assert v.hard_break_rate == 0.5


def test_aggregate_counts_errored_results():
    scores = [_ps("1", "m", CompatLabel.COMPATIBLE)]
    results = [
        CandidateResult(sample_id="1", model="m", status="ok", cost_estimate=0.001, latency_ms=100),
        CandidateResult(sample_id="2", model="m", status="error", error="throttled"),
    ]
    v = aggregate("m", scores, results)
    assert v.errored == 1
    assert v.cost_total == 0.001


def test_aggregate_latency_percentiles():
    scores = [_ps(str(i), "m", CompatLabel.COMPATIBLE) for i in range(5)]
    results = [CandidateResult(sample_id=str(i), model="m", status="ok",
                               latency_ms=100 * (i + 1)) for i in range(5)]  # 100..500
    v = aggregate("m", scores, results)
    assert v.latency_avg_ms == 300.0
    assert v.latency_p50_ms == 300.0
    assert v.latency_p95_ms == 500.0


def test_side_by_side_ranks_by_confidence():
    v = side_by_side([
        aggregate("terra", [_ps("1", "terra", CompatLabel.PARTIAL)]),
        aggregate("luna", [_ps("1", "luna", CompatLabel.COMPATIBLE)]),
    ])
    assert v[0].model == "luna"  # higher confidence ranks first


# ------------------------------ remediation -------------------------------- #
def test_recommend_clusters_by_reason():
    scores = [
        _ps("1", "m", CompatLabel.COMPATIBLE_WITH_DRIFT, reason="added_verbosity"),
        _ps("2", "m", CompatLabel.PARTIAL, reason="added_verbosity"),
        _ps("3", "m", CompatLabel.INCOMPATIBLE, hard=True, reason="format_change"),
    ]
    recs = recommend("run1", "m", scores)
    # largest cluster (added_verbosity, 2) first
    assert recs[0].target.reason == "added_verbosity"
    assert recs[0].evidence_count == 2
    assert recs[0].change.type == "reasoning_effort"
    reasons = {r.target.reason for r in recs}
    assert "format_change" in reasons


def test_retest_before_after_delta():
    samples = [_sample("1", golden="Yes."), _sample("2", golden="No.")]
    before = [
        _ps("1", "m", CompatLabel.COMPATIBLE_WITH_DRIFT, reason="added_verbosity", d1=0.9),
        _ps("2", "m", CompatLabel.PARTIAL, reason="added_verbosity", d1=0.6),
    ]
    recs = recommend("run1", "m", before)
    rem = recs[0]

    # rescore that "fixes" everything to Compatible
    def _rescore(sample, effort, change):
        return _ps(sample.id, "m", CompatLabel.COMPATIBLE)

    out = retest("run1", "m", rem, samples, before, _rescore)
    assert out.applied
    assert out.result.confidence_after > out.result.confidence_before
    assert out.result.moved["to_compatible"] == 2
    assert out.result.moved["regressions"] == 0
