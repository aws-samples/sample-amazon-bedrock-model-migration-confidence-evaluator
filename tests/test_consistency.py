"""D7 Consistency: answer agreement + verdict agreement across replays."""
from modelshift.consistency import candidate_consistency, pair_agreement, request_consistency
from modelshift.schemas import CandidateOutput, CandidateResult, CompatLabel, PairScore, ToolCall

C, D, P, INC = (CompatLabel.COMPATIBLE, CompatLabel.COMPATIBLE_WITH_DRIFT,
                CompatLabel.PARTIAL, CompatLabel.INCOMPATIBLE)


def _r(text, k, sid="s1", m="m", tools=None):
    return CandidateResult(sample_id=sid, model=m, replay_index=k,
                           output=CandidateOutput(text=text, tool_calls=tools))


def _p(label, k, sid="s1", m="m", hb=False):
    return PairScore(sample_id=sid, model=m, label=label, replay_index=k, hard_break=hb)


def test_pair_agreement_rules():
    a = CandidateOutput(text="The refund is 42 dollars, call 555-123-4567")
    assert pair_agreement(a, a) == 1.0
    assert pair_agreement(a, CandidateOutput(text="- a list\n- of items")) == 0.0          # format class differs
    assert pair_agreement(a, CandidateOutput(text="The refund is 42 dollars, call 555-999-0000")) == 0.0  # fact clash
    j1, j2 = CandidateOutput(text='{"a": 1, "b": 2}'), CandidateOutput(text='{"b": 2, "a": 1}')
    assert pair_agreement(j1, j2) == 1.0                                                   # key order ignored
    assert pair_agreement(j1, CandidateOutput(text='{"a": 1}')) == 0.0                     # json keys differ
    t1 = CandidateOutput(text="", tool_calls=[ToolCall(name="lookup", arguments={"id": 1})])
    t2 = CandidateOutput(text="", tool_calls=[ToolCall(name="refund", arguments={"id": 1})])
    assert pair_agreement(t1, t1) == 1.0 and pair_agreement(t1, t2) == 0.0
    assert pair_agreement(t1, CandidateOutput(text="no tool")) == 0.0


def test_identical_answers_score_one():
    rc = request_consistency("s1", "m", [_r("same answer", k) for k in range(3)], [_p(C, k) for k in range(3)])
    assert rc.d7 == 1.0 and not rc.unstable and not rc.flip and rc.labels == [C, C, C]


def test_two_of_three_labels_and_flip():
    res = [_r("alpha beta gamma", 0), _r("alpha beta gamma", 1), _r("alpha beta gamma delta", 2)]
    rc = request_consistency("s1", "m", res, [_p(C, 0), _p(C, 1), _p(INC, 2, hb=True)])
    assert rc.verdict_agreement == round(2 / 3, 4)
    assert rc.flip and rc.unstable                     # Compatible and Incompatible together
    assert rc.most_different_pair in ([0, 2], [1, 2])


def test_overrides_count_and_single_answer_not_measured():
    p = _p(INC, 1)
    p.overridden, p.override_label = True, C
    rc = request_consistency("s1", "m", [_r("x y", 0), _r("x y", 1)], [_p(C, 0), p])
    assert rc.labels == [C, C] and rc.d7 == 1.0
    one = request_consistency("s1", "m", [_r("x", 0)], [_p(C, 0)])
    assert one.d7 is None and one.answers == 1


def test_candidate_rollup_and_low_confidence():
    results = [_r("a b", k, sid) for sid in ("s1", "s2") for k in range(2)]
    scores = [_p(C, k, sid) for sid in ("s1", "s2") for k in range(2)]
    s = candidate_consistency("m", results, scores, min_scored=1)
    assert s["consistency"] == 1.0 and s["consistency_measured"] == 2 and s["unstable_requests"] == 0
    assert s["flip_rate"] == 0.0 and s["consistency_low_confidence"] is False
    assert candidate_consistency("m", results, scores, min_scored=5)["consistency_low_confidence"] is True
    # replays off: one answer per request -> not measured
    s1 = candidate_consistency("m", [_r("a", 0, "s1")], [_p(C, 0, "s1")])
    assert s1["consistency"] is None and s1["consistency_measured"] == 0
