"""Aggregation + remediation recommender (Doc 3 §4.3, §6)."""
from __future__ import annotations

import uuid
from collections import Counter
from typing import Callable, Dict, List, Optional

from .config import DEFAULT_RUBRIC, Rubric
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
)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(model: str,
              pair_scores: List[PairScore],
              results: Optional[List[CandidateResult]] = None,
              rubric: Rubric = DEFAULT_RUBRIC,
              endpoint: str = "") -> CandidateVerdict:
    scored = [p for p in pair_scores if p.model == model]
    results = [r for r in (results or []) if r.model == model]
    errored = sum(1 for r in results if r.status == "error")

    distribution: Dict[CompatLabel, int] = {lbl: 0 for lbl in CompatLabel}
    for p in scored:
        lbl = p.override_label if p.overridden and p.override_label else p.label
        distribution[lbl] += 1

    n = len(scored)
    if n == 0:
        return CandidateVerdict(model=model, scored=0, errored=errored, endpoint=endpoint)

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

    cost_total = round(sum(r.cost_estimate for r in results), 6)
    latencies = sorted(r.latency_ms for r in results if r.latency_ms is not None)
    latency_avg = round(sum(latencies) / len(latencies), 1) if latencies else None

    def _pct(vals, q):
        if not vals:
            return None
        idx = min(len(vals) - 1, int(round(q * (len(vals) - 1))))
        return float(vals[idx])

    latency_p50 = _pct(latencies, 0.50)
    latency_p95 = _pct(latencies, 0.95)

    return CandidateVerdict(
        model=model,
        migration_confidence=confidence,
        verdict_band=rubric.band_for(confidence, hard_break_rate),
        endpoint=endpoint,
        distribution=distribution,
        dimension_averages=dim_avgs,
        top_failure_reasons=top,
        hard_break_rate=hard_break_rate,
        cost_total=cost_total,
        latency_avg_ms=latency_avg,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        scored=n,
        errored=errored,
    )


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
        clusters.setdefault(reason, []).append(p.sample_id)

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
    before_map = {p.sample_id: p for p in before_scores if p.model == model}

    confidence_before = aggregate(model, [before_map[s.id] for s in subset if s.id in before_map],
                                  rubric=rubric).migration_confidence

    after_scores: List[PairScore] = []
    after_by_id: Dict[str, PairScore] = {}
    moved_to_compatible = 0
    regressions = 0
    drifted = 0
    errors = 0
    for s in subset:
        new_ps = rescore(s, reasoning_effort, remediation.change)
        if new_ps is None:
            errors += 1
            continue
        after_scores.append(new_ps)
        after_by_id[s.id] = new_ps
        old = before_map.get(s.id)
        if old is not None:
            if old.label != CompatLabel.COMPATIBLE and new_ps.label == CompatLabel.COMPATIBLE:
                moved_to_compatible += 1
            if old.label == CompatLabel.COMPATIBLE and new_ps.label == CompatLabel.INCOMPATIBLE:
                regressions += 1

    guard_checked = 0
    for s in guard:
        new_ps = rescore(s, reasoning_effort, remediation.change)
        if new_ps is None:
            errors += 1
            continue
        guard_checked += 1
        after_by_id[s.id] = new_ps
        if new_ps.label in (CompatLabel.PARTIAL, CompatLabel.INCOMPATIBLE):
            regressions += 1
        elif new_ps.label == CompatLabel.COMPATIBLE_WITH_DRIFT:
            drifted += 1

    confidence_after = aggregate(model, after_scores, rubric=rubric).migration_confidence
    # Projection for the whole candidate: re-tested evaluations take their new scores,
    # everything not re-tested keeps its original score.
    model_scores = [p for p in before_scores if p.model == model]
    projected = [after_by_id.get(p.sample_id, p) for p in model_scores]

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
