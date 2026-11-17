"""D7 Consistency: how alike a candidate's replayed answers to the same request are.

Replays are ordinary calls, scored against the golden like any other (see the
replay-consistency spec). This module adds a separate dimension on top:

    D7 = 0.5 * answer agreement + 0.5 * verdict agreement            (per request, 0..1)

* Answer agreement: mean, over every pair of answers, of the same token-overlap measure D1
  uses against the golden. A pair counts as 0 if the two answers differ in format class
  (or JSON keys), contradict on concrete facts, or call different tools / tool arguments.
  Token overlap ignores order, so JSON key order makes no difference.
* Verdict agreement: share of answers whose label equals the most common label.

D7 never changes a label or Migration Confidence; it only feeds the optional
consistency gates and is reported alongside D1-D6.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from .schemas import CandidateOutput, CandidateResult, CompatLabel, PairScore
from .scoring import (
    _cand_text,
    _format_class,
    score_d1_semantic,
    score_d2_format,
    score_d3_factual,
    score_d6_tool_call,
)

DEFAULT_INSTABILITY_THRESHOLD = 0.70


class RequestConsistency(BaseModel):
    sample_id: str
    model: str
    answers: int = 0
    d7: Optional[float] = None               # None when fewer than 2 answers succeeded
    answer_agreement: Optional[float] = None
    verdict_agreement: Optional[float] = None
    labels: List[CompatLabel] = Field(default_factory=list)   # per replay, override-aware
    replay_indices: List[int] = Field(default_factory=list)
    flip: bool = False                       # some answers hard-break, some don't
    unstable: bool = False
    most_different_pair: Optional[List[int]] = None          # replay indices, for the diff view


def _label(p: PairScore) -> CompatLabel:
    return p.override_label if p.overridden and p.override_label else p.label


def pair_agreement(a: CandidateOutput, b: CandidateOutput) -> float:
    """0..1 similarity of two answers to the same request."""
    ta, tb = _cand_text(a), _cand_text(b)
    if a.tool_calls or b.tool_calls:
        if not a.tool_calls or not b.tool_calls:
            return 0.0
        d6, broke = score_d6_tool_call(a.tool_calls, b.tool_calls)
        if broke or score_d6_tool_call(b.tool_calls, a.tool_calls)[1]:
            return 0.0
        if not ta and not tb:
            return d6.score
    ca, cb = _format_class(ta), _format_class(tb)
    if ca != cb:
        return 0.0
    if ca == "json" and (score_d2_format(ta, tb)[1] or score_d2_format(tb, ta)[1]):
        return 0.0                           # different JSON keys
    if score_d3_factual(ta, tb)[1] or score_d3_factual(tb, ta)[1]:
        return 0.0                           # contradicting phones / emails / key numbers
    return score_d1_semantic(ta, tb).score


def request_consistency(sample_id: str, model: str,
                        results: Sequence[CandidateResult],
                        scores: Sequence[PairScore],
                        instability_threshold: float = DEFAULT_INSTABILITY_THRESHOLD) -> RequestConsistency:
    """D7 for one request, from its successful answers and their per-answer scores."""
    by_rep = {p.replay_index: p for p in scores}
    ok = sorted((r for r in results if r.status == "ok" and r.replay_index in by_rep),
                key=lambda r: r.replay_index)
    rc = RequestConsistency(sample_id=sample_id, model=model, answers=len(ok),
                            labels=[_label(by_rep[r.replay_index]) for r in ok],
                            replay_indices=[r.replay_index for r in ok])
    if len(ok) < 2:
        return rc
    pairs = [((x.replay_index, y.replay_index), pair_agreement(x.output, y.output))
             for x, y in combinations(ok, 2)]
    answer = sum(v for _, v in pairs) / len(pairs)
    modal = Counter(rc.labels).most_common(1)[0][1]
    verdict = modal / len(rc.labels)
    rc.answer_agreement = round(answer, 4)
    rc.verdict_agreement = round(verdict, 4)
    rc.d7 = round(0.5 * answer + 0.5 * verdict, 4)
    hb = [by_rep[r.replay_index].hard_break for r in ok]
    rc.flip = any(hb) and not all(hb)
    rc.unstable = rc.d7 < instability_threshold or (
        CompatLabel.COMPATIBLE in rc.labels and CompatLabel.INCOMPATIBLE in rc.labels)
    rc.most_different_pair = list(min(pairs, key=lambda kv: kv[1])[0])
    return rc


def requests_for(model: str, results: Sequence[CandidateResult], scores: Sequence[PairScore],
                 instability_threshold: float = DEFAULT_INSTABILITY_THRESHOLD) -> List[RequestConsistency]:
    res_by: Dict[str, List[CandidateResult]] = defaultdict(list)
    sc_by: Dict[str, List[PairScore]] = defaultdict(list)
    for r in results:
        if r.model == model:
            res_by[r.sample_id].append(r)
    for p in scores:
        if p.model == model:
            sc_by[p.sample_id].append(p)
    return [request_consistency(sid, model, res_by[sid], sc_by.get(sid, []), instability_threshold)
            for sid in res_by]


def candidate_consistency(model: str, results: Sequence[CandidateResult], scores: Sequence[PairScore],
                          instability_threshold: float = DEFAULT_INSTABILITY_THRESHOLD,
                          min_scored: int = 1) -> Dict[str, Any]:
    """Per-candidate D7 roll-up: mean D7, unstable count, flip rate, latency spread."""
    reqs = requests_for(model, results, scores, instability_threshold)
    measured = [r for r in reqs if r.d7 is not None]
    out: Dict[str, Any] = {"consistency": None, "consistency_measured": len(measured),
                           "consistency_low_confidence": False, "unstable_requests": 0,
                           "flip_rate": None, "latency_spread_p50": None}
    if measured:
        out["consistency"] = round(sum(r.d7 for r in measured) / len(measured), 4)
        out["unstable_requests"] = sum(1 for r in measured if r.unstable)
        out["flip_rate"] = round(sum(1 for r in measured if r.flip) / len(measured), 4)
        out["consistency_low_confidence"] = len(measured) < max(1, min_scored)
    # Latency spread: per request with >= 2 fresh timed answers, max/min; then the median.
    spreads = []
    timed: Dict[str, List[int]] = defaultdict(list)
    for r in results:
        if r.model == model and r.status == "ok" and r.latency_ms and not r.from_cache:
            timed[r.sample_id].append(r.latency_ms)
    for vals in timed.values():
        if len(vals) >= 2 and min(vals) > 0:
            spreads.append(max(vals) / min(vals))
    if spreads:
        out["latency_spread_p50"] = round(median(spreads), 2)
    return out
