"""Aggregation + remediation recommender (Doc 3 §4.3, §6)."""
from __future__ import annotations

import statistics
import uuid
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from .config import (
    DEFAULT_RUBRIC,
    DEFAULT_SETTINGS,
    Rubric,
    Settings,
    billed_output_tokens,
    estimate_cost,
)
from .schemas import (
    CandidateResult,
    CandidateVerdict,
    CompatLabel,
    NormalizedSample,
    PairScore,
    ReasoningEffort,
    Remediation,
    RemediationChange,
    RemediationResult,
    RemediationTarget,
    VerdictBand,
)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(model: str,
              pair_scores: List[PairScore],
              results: Optional[List[CandidateResult]] = None,
              rubric: Rubric = DEFAULT_RUBRIC,
              endpoint: str = "",
              samples: Optional[List[NormalizedSample]] = None,
              settings: Optional[Settings] = None,
              monthly_volume: Optional[int] = None) -> CandidateVerdict:
    settings = settings or DEFAULT_SETTINGS
    scored = [p for p in pair_scores if p.model == model]
    results = [r for r in (results or []) if r.model == model]
    errored = sum(1 for r in results if r.status == "error")

    distribution: Dict[CompatLabel, int] = {lbl: 0 for lbl in CompatLabel}
    for p in scored:
        lbl = p.override_label if p.overridden and p.override_label else p.label
        distribution[lbl] += 1

    n = len(scored)
    if n == 0:
        v = CandidateVerdict(model=model, scored=0, errored=errored, endpoint=endpoint,
                             error_rate=round(errored / len(results), 4) if results else 0.0)
        v.eligible, v.ineligible_reasons = eligibility(v, settings)
        return v

    # Migration Confidence 0..100 (weighted share).
    weighted = sum(rubric.confidence_weight(p.override_label if p.overridden and p.override_label else p.label)
                   for p in scored)
    confidence = round(100.0 * weighted / n, 1)

    hard_breaks = sum(1 for p in scored if p.hard_break)
    hard_break_rate = round(hard_breaks / n, 4)

    # Per-dimension averages.
    def _avg(attr: str) -> float:
        vals = [getattr(p, attr).score for p in scored if getattr(p, attr) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    dim_avgs = {
        "d1_semantic": _avg("d1_semantic"),
        "d2_format": _avg("d2_format"),
        "d3_factual": _avg("d3_factual"),
        "d4_verbosity": _avg("d4_verbosity"),
        "d5_instruction": _avg("d5_instruction"),
    }

    # Top failure reasons (from judge_reason on non-compatible samples).
    reasons = Counter(
        p.judge_reason for p in scored
        if p.judge_reason and p.label != CompatLabel.COMPATIBLE
    )
    top = [{"reason": r, "count": c} for r, c in reasons.most_common(5)]

    ok_results = [r for r in results if r.status == "ok"]
    cost_known = all(r.cost_known for r in ok_results)
    cost_total = round(sum(r.cost_estimate for r in results), 6)
    # Latency: fresh (non-cached) SUCCESSFUL calls only -- cached results were timed by an
    # earlier call, and a failed call's duration says nothing about the model's speed.
    timed = [r for r in ok_results if r.latency_ms is not None and not r.from_cache]
    latencies = sorted(r.latency_ms for r in timed)
    latency_avg = round(sum(latencies) / len(latencies), 1) if latencies else None

    def _pct(vals, q):
        if not vals:
            return None
        idx = min(len(vals) - 1, int(round(q * (len(vals) - 1))))
        return float(vals[idx])

    latency_p50 = _pct(latencies, 0.50)
    latency_p95 = _pct(latencies, 0.95)

    v = CandidateVerdict(
        model=model,
        migration_confidence=confidence,
        verdict_band=rubric.band_for(confidence, hard_break_rate),
        endpoint=endpoint,
        distribution=distribution,
        dimension_averages=dim_avgs,
        top_failure_reasons=top,
        hard_break_rate=hard_break_rate,
        cost_total=cost_total,
        cost_known=cost_known,
        latency_avg_ms=latency_avg,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        scored=n,
        errored=errored,
    )

    # ---- Multi-objective recommendation metrics ---- #
    ok = len(ok_results)
    v.ok_calls = ok
    v.error_rate = round(errored / len(results), 4) if results else 0.0
    if ok and cost_known:
        ok_cost = sum(r.cost_estimate for r in ok_results)
        v.cost_per_request = round(ok_cost / ok, 8)
        v.cost_per_1k_requests = round(ok_cost / ok * 1000, 4)
        if monthly_volume:
            v.projected_monthly_cost = round(ok_cost / ok * monthly_volume, 2)

    outs = [billed_output_tokens(r.usage.completion_tokens or 0, r.usage.reasoning_tokens or 0)
            for r in ok_results if r.usage.completion_tokens is not None]
    if outs:
        v.avg_output_tokens = round(sum(outs) / len(outs), 1)
    reas = [r.usage.reasoning_tokens for r in ok_results if r.usage.reasoning_tokens is not None]
    if reas:
        v.avg_reasoning_tokens = round(sum(reas) / len(reas), 1)

    by_id = {s.id: s for s in (samples or [])}
    v.output_token_ratio = _output_token_ratio(ok_results, by_id)
    v.cost_vs_current_pct = _cost_vs_current_pct(ok_results, by_id) if cost_known else None

    per_tok = [r.latency_ms / max(1, billed_output_tokens(r.usage.completion_tokens or 0,
                                                          r.usage.reasoning_tokens or 0))
               for r in timed]
    v.ms_per_output_token_p50 = round(statistics.median(per_tok), 3) if per_tok else None
    v.latency_samples = len(timed)
    v.latency_low_confidence = len(timed) < settings.min_latency_samples

    # ---- Replays / D7 Consistency: replayed answers above were scored as normal calls;
    # D7 is a separate dimension and never changes labels or Migration Confidence. ----
    v.replays = max([r.replay_index for r in results] + [p.replay_index for p in scored] + [0]) + 1
    v.requests_scored = len({p.sample_id for p in scored})
    if v.replays > 1:
        from .consistency import candidate_consistency
        cc = candidate_consistency(model, results, scored, settings.instability_threshold,
                                   settings.min_scored)
        for k, val in cc.items():
            setattr(v, k, val)
        if v.consistency is not None:
            v.dimension_averages["d7_consistency"] = v.consistency
            v.unstable = (not v.consistency_low_confidence) and v.consistency < settings.instability_threshold

    v.eligible, v.ineligible_reasons = eligibility(v, settings)
    return v


def _output_token_ratio(ok_results: List[CandidateResult],
                        by_id: Dict[str, NormalizedSample]) -> Optional[float]:
    """Mean of candidate billed output / current-model output over samples where both are known."""
    ratios = []
    for r in ok_results:
        s = by_id.get(r.sample_id)
        golden = s.usage.completion_tokens if s else None
        if not golden or r.usage.completion_tokens is None:
            continue
        cand = billed_output_tokens(r.usage.completion_tokens, r.usage.reasoning_tokens or 0)
        ratios.append(cand / golden)
    return round(sum(ratios) / len(ratios), 3) if ratios else None


def _current_cost(s: NormalizedSample) -> Optional[float]:
    """What the current production model cost for this logged call: logged spend if present,
    else list price of the legacy model (if priced). None when neither is available."""
    if s.usage.spend is not None:
        return float(s.usage.spend)
    if s.usage.prompt_tokens is None or s.usage.completion_tokens is None:
        return None
    return estimate_cost(s.legacy_model or "", s.usage.prompt_tokens, s.usage.completion_tokens,
                         s.usage.reasoning_tokens or 0)


def _cost_vs_current_pct(ok_results: List[CandidateResult],
                         by_id: Dict[str, NormalizedSample]) -> Optional[float]:
    """Candidate cost vs the current model on the SAME samples, as a percentage.

    Computed per sample against that sample's own legacy model, so mixed-baseline runs work.
    Returns None unless every successful sample has a known current cost: a partial baseline
    would silently compare different request sets.
    """
    if not ok_results:
        return None
    cand = base = 0.0
    for r in ok_results:
        s = by_id.get(r.sample_id)
        c = _current_cost(s) if s else None
        if c is None:
            return None
        base += c
        cand += r.cost_estimate
    if base <= 0:
        return None
    return round((cand - base) / base * 100.0, 1)


def eligibility(v: CandidateVerdict, settings: Optional[Settings] = None) -> Tuple[bool, List[str]]:
    """Quality gate for recommendation (spec R2.1). Returns (eligible, human-readable reasons)."""
    s = settings or DEFAULT_SETTINGS
    reasons: List[str] = []
    if v.scored < s.min_scored or v.scored == 0:
        reasons.append("no scored evaluations")
        return False, reasons
    if v.verdict_band == VerdictBand.NOT_A_DROP_IN:
        reasons.append("verdict is 'Not a drop-in'")
    if v.hard_break_rate > s.max_hard_break_rate:
        reasons.append(f"hard-break rate {v.hard_break_rate:.0%} > {s.max_hard_break_rate:.0%}")
    if v.migration_confidence < s.min_confidence:
        reasons.append(f"confidence {v.migration_confidence:.0f} < {s.min_confidence:.0f}")
    if v.error_rate > s.max_error_rate:
        reasons.append(f"error rate {v.error_rate:.0%} > {s.max_error_rate:.0%}")
    # Optional D7 gates (off by default); skipped when too few requests were measured.
    if v.consistency is not None and not v.consistency_low_confidence:
        if s.min_consistency is not None and v.consistency < s.min_consistency:
            reasons.append(f"consistency {v.consistency:.2f} < {s.min_consistency:.2f}")
        if s.max_flip_rate is not None and v.flip_rate is not None and v.flip_rate > s.max_flip_rate:
            reasons.append(f"flip rate {v.flip_rate:.0%} > {s.max_flip_rate:.0%}")
    return (not reasons), reasons


def side_by_side(verdicts: List[CandidateVerdict]) -> List[CandidateVerdict]:
    """Rank candidates: highest confidence, then lowest hard-break rate, then cost."""
    return sorted(verdicts,
                  key=lambda v: (-v.migration_confidence, v.hard_break_rate, v.cost_total))


# --------------------------------------------------------------------------- #
# Remediation recommender (Doc 3 §6)
# --------------------------------------------------------------------------- #
_REASON_TO_REMEDIATION = {
    "added_verbosity": ("reasoning_effort",
                        "Add a concise-output instruction or lower reasoning effort (medium->low)."),
    "format_change": ("format_schema",
                      "Add an explicit output-format instruction / pin a structured-output schema."),
    "missing_content": ("prompt_edit",
                        "Restore the context the newer model no longer assumes; make the instruction explicit."),
    "instruction_violation": ("prompt_edit",
                              "Simplify/clarify the prompt; remove contradictory constraints."),
    "factual_divergence": ("prompt_edit",
                           "Pin the concrete fact in the prompt / provide it as context; do not rely on memory."),
}


def recommend(run_id: str, model: str, pair_scores: List[PairScore],
              samples: Optional[List[NormalizedSample]] = None,
              reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM) -> List[Remediation]:
    """Cluster non-compatible samples by judge reason and emit suggestions.

    When ``samples`` is given each suggestion is made concrete (see remediate.py):
    an effort step-down, or an instruction to add to the system prompt.
    """
    from .remediate import concretize
    scored = [p for p in pair_scores if p.model == model
              and p.label != CompatLabel.COMPATIBLE]
    clusters: Dict[str, List[str]] = {}
    for p in scored:
        reason = p.judge_reason or ("factual_divergence" if p.hard_break else "missing_content")
        ids = clusters.setdefault(reason, [])
        if p.sample_id not in ids:  # with replays a request can fail on several answers
            ids.append(p.sample_id)

    suggestions: List[Remediation] = []
    for reason, sample_ids in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        change_type, effect = _REASON_TO_REMEDIATION.get(
            reason, ("prompt_edit", "Review and clarify the prompt for this cluster.")
        )
        suggestions.append(Remediation(
            remediation_id=f"rem_{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            candidate=model,
            target=RemediationTarget(scope="cluster", reason=reason, sample_ids=sample_ids),
            change=RemediationChange(type=change_type),
            evidence_count=len(sample_ids),
            expected_effect=effect,
            applied=False,
        ))
        if samples is not None:
            concretize(suggestions[-1], samples, reasoning_effort)
    # If capability-limited (low semantic broadly), suggest the next candidate up.
    return suggestions


def retest(run_id: str,
           model: str,
           remediation: Remediation,
           samples: List[NormalizedSample],
           before_scores: List[PairScore],
           rescore: Callable[[NormalizedSample, ReasoningEffort, RemediationChange], Optional[PairScore]],
           reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
           rubric: Rubric = DEFAULT_RUBRIC,
           guard_ids: Optional[List[str]] = None) -> Remediation:
    """Re-run the targeted subset with the change applied; compute before/after delta.

    `rescore` is injected (calls the adapter+scoring with the change applied) so this
    stays testable without live models. It returns None when the candidate call
    failed; those evaluations are counted as errors and left out of the "after"
    confidence (the same way errored calls are excluded from a normal run).

    `guard_ids` are evaluations that were already Compatible. They are re-run with
    the same change purely to catch regressions (a fix for one cluster that breaks
    answers which used to be fine):
      regressions = Compatible -> Partial or Incompatible
      drifted     = Compatible -> Compatible w/ drift
    """
    target_ids = set(remediation.target.sample_ids)
    guard_set = set(guard_ids or []) - target_ids
    subset = [s for s in samples if s.id in target_ids]
    guard = [s for s in samples if s.id in guard_set]
    # With replays a request has one score per replay; compare answers replay by replay.
    before_map: Dict[str, Dict[int, PairScore]] = {}
    for p in before_scores:
        if p.model == model:
            before_map.setdefault(p.sample_id, {})[p.replay_index] = p

    def _answers(out) -> List[PairScore]:
        # rescore may return None (call failed), one PairScore, or a list (one per replay).
        if out is None:
            return []
        return [x for x in (out if isinstance(out, list) else [out]) if x is not None]

    confidence_before = aggregate(model, [p for s in subset for p in before_map.get(s.id, {}).values()],
                                  rubric=rubric).migration_confidence

    after_scores: List[PairScore] = []
    after_by_id: Dict[str, List[PairScore]] = {}
    moved_to_compatible = 0
    regressions = 0
    drifted = 0
    errors = 0
    for s in subset:
        new = _answers(rescore(s, reasoning_effort, remediation.change))
        if not new:
            errors += 1
            continue
        after_scores.extend(new)
        after_by_id[s.id] = new
        old_reps = before_map.get(s.id, {})
        for new_ps in new:
            old = old_reps.get(new_ps.replay_index)
            if old is not None:
                if old.label != CompatLabel.COMPATIBLE and new_ps.label == CompatLabel.COMPATIBLE:
                    moved_to_compatible += 1
                if old.label == CompatLabel.COMPATIBLE and new_ps.label == CompatLabel.INCOMPATIBLE:
                    regressions += 1

    guard_checked = 0
    for s in guard:
        new = _answers(rescore(s, reasoning_effort, remediation.change))
        if not new:
            errors += 1
            continue
        guard_checked += 1
        after_by_id[s.id] = new
        for new_ps in new:
            if new_ps.label in (CompatLabel.PARTIAL, CompatLabel.INCOMPATIBLE):
                regressions += 1
            elif new_ps.label == CompatLabel.COMPATIBLE_WITH_DRIFT:
                drifted += 1

    confidence_after = aggregate(model, after_scores, rubric=rubric).migration_confidence
    # Projection for the whole candidate: re-tested evaluations take their new scores,
    # everything not re-tested keeps its original score.
    model_scores = [p for p in before_scores if p.model == model]
    projected = ([p for p in model_scores if p.sample_id not in after_by_id]
                 + [p for new in after_by_id.values() for p in new])

    remediation.applied = True
    result = remediation.result or RemediationResult()
    result.confidence_before = confidence_before
    result.confidence_after = confidence_after
    result.confidence_run_before = aggregate(model, model_scores, rubric=rubric).migration_confidence
    result.confidence_run_after = aggregate(model, projected, rubric=rubric).migration_confidence
    result.regression_checked = guard_checked
    result.moved = {"to_compatible": moved_to_compatible, "regressions": regressions,
                    "drifted": drifted, "errors": errors}
    remediation.result = result
    return remediation
