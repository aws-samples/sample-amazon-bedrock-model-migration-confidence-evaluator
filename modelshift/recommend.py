"""Multi-objective recommendation (spec: multi-objective-recommendation, design §5).

A PURE function of (verdicts, settings, weights): no model calls, no run mutation. That is
what lets the Results page re-rank instantly when the user changes the weights.

Quality = Migration Confidence (unchanged, not blended -- Decision 1). Cost and latency are
scored relative to the best eligible candidate (ratio-to-best, so a 2% price difference is
not a 0-vs-100 gap). Only candidates that pass the quality gate (aggregate.eligibility) can
be recommended.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import DEFAULT_SETTINGS, Settings
from .schemas import (
    CandidateAxes,
    CandidateVerdict,
    Profile,
    ProfileKind,
    Recommendation,
    WeightPreset,
    Weights,
)

SIMILAR = 0.05          # relative differences under 5% are worded "similar"
_AXES = ("quality", "cost", "latency")
_PROFILE_ORDER = (ProfileKind.BEST_OVERALL, ProfileKind.BEST_QUALITY,
                  ProfileKind.MOST_ECONOMICAL, ProfileKind.LOWEST_LATENCY)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def weights_for(preset: str = "balanced", settings: Optional[Settings] = None,
                custom: Optional[Weights] = None) -> Weights:
    """Resolve a preset name (from settings.weight_presets) or custom weights, normalised."""
    s = settings or DEFAULT_SETTINGS
    if custom is not None and (preset == WeightPreset.CUSTOM.value or preset is None):
        return Weights(preset=WeightPreset.CUSTOM, quality=custom.quality, cost=custom.cost,
                       latency=custom.latency).normalised()
    vals = s.weight_presets.get(preset) or s.weight_presets.get("balanced") or {}
    try:
        p = WeightPreset(preset)
    except ValueError:
        p = WeightPreset.BALANCED
    return Weights(preset=p, **{a: vals.get(a, 0.0) for a in _AXES}).normalised()


# --------------------------------------------------------------------------- #
# Axis scores
# --------------------------------------------------------------------------- #
def _latency_metric(v: CandidateVerdict, metric: str) -> Optional[float]:
    """Latency value used for ranking (lower is better), or None if not usable."""
    if v.latency_low_confidence:
        return None
    val = v.latency_p50_ms if metric == "p50" else v.ms_per_output_token_p50
    return val if (val is not None and val > 0) else None


def _ratio_scores(values: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Lower-is-better values -> 0..100 ratio-to-best (best = 100). None stays None."""
    known = {k: x for k, x in values.items() if x is not None and x >= 0}
    if not known:
        return {k: None for k in values}
    best = min(known.values())
    out: Dict[str, Optional[float]] = {}
    for k, x in values.items():
        if x is None or x < 0:
            out[k] = None
        elif x == 0:
            out[k] = 100.0
        elif best == 0:
            out[k] = 0.0
        else:
            out[k] = round(100.0 * best / x, 2)
    return out


def _composite(ax: CandidateAxes, w: Weights) -> Tuple[Optional[float], bool]:
    """Weighted score renormalised over the axes this candidate HAS. Returns (score, complete),
    where complete = it has every axis whose weight > 0. Quality enters as quality_score."""
    parts = [(w.quality, ax.quality_score), (w.cost, ax.cost), (w.latency, ax.latency)]
    complete = all(score is not None for wt, score in parts if wt > 0)
    present = [(wt, score) for wt, score in parts if wt > 0 and score is not None]
    total = sum(wt for wt, _ in present)
    if total <= 0:
        return None, complete
    return round(sum(wt * score for wt, score in present) / total, 2), complete


def _dominates(a: CandidateAxes, b: CandidateAxes) -> bool:
    """a dominates b: >= on every axis both have, > on at least one."""
    shared = [ax for ax in _AXES if getattr(a, ax) is not None and getattr(b, ax) is not None]
    if not shared:
        return False
    ge = all(getattr(a, ax) >= getattr(b, ax) for ax in shared)
    gt = any(getattr(a, ax) > getattr(b, ax) for ax in shared)
    return ge and gt


# --------------------------------------------------------------------------- #
# Trade-off wording (deterministic templates, never an LLM)
# --------------------------------------------------------------------------- #
def _cost_phrase(r: Optional[float]) -> Tuple[str, bool]:
    """(phrase, comparative) for cost ratio r = this / best_overall."""
    if r is None:
        return "unknown cost", False
    if abs(r - 1.0) < SIMILAR:
        return "similar cost per request", False
    if r < 1.0:
        return f"{(1 - r) * 100:.0f}% cheaper per request", True
    if r >= 2.0:
        return f"{r:.1f}x the cost per request", False
    return f"{(r - 1) * 100:.0f}% more per request", True


def _latency_phrase(r: Optional[float]) -> Tuple[str, bool]:
    if r is None:
        return "latency not comparable", False
    if abs(r - 1.0) < SIMILAR:
        return "similar latency", False
    if r < 1.0:
        return f"{(1 - r) * 100:.0f}% faster", True
    if r >= 2.0:
        return f"{r:.1f}x slower", False
    return f"{(r - 1) * 100:.0f}% slower", True


def _quality_phrase(q: float, q_bo: float) -> Tuple[str, bool]:
    d = q - q_bo
    if abs(d) < 0.5:
        return f"the same confidence ({q:.0f})", False
    word = "more" if d > 0 else "less"
    n = f"{abs(d):.0f}"
    unit = "point" if n == "1" else "points"
    return f"{n} {unit} {word} confidence ({q:.0f} vs {q_bo:.0f})", True


def _tradeoff(kind: ProfileKind, v: CandidateVerdict, bo: CandidateVerdict,
              lat: Optional[float], lat_bo: Optional[float]) -> Tuple[str, Dict[str, Optional[float]]]:
    cost_r = (v.cost_per_request / bo.cost_per_request
              if v.cost_per_request is not None and bo.cost_per_request else None)
    lat_r = (lat / lat_bo if lat is not None and lat_bo else None)
    deltas = {
        "quality_pts": round(v.migration_confidence - bo.migration_confidence, 1),
        "cost_pct": round(cost_r - 1.0, 4) if cost_r is not None else None,
        "latency_pct": round(lat_r - 1.0, 4) if lat_r is not None else None,
    }
    phrases = {
        "quality": _quality_phrase(v.migration_confidence, bo.migration_confidence),
        "cost": _cost_phrase(cost_r),
        "latency": _latency_phrase(lat_r),
    }
    lead_axis = {ProfileKind.MOST_ECONOMICAL: "cost", ProfileKind.LOWEST_LATENCY: "latency",
                 ProfileKind.BEST_QUALITY: "quality"}[kind]
    lead, comparative = phrases[lead_axis]
    others = [phrases[a][0] for a in _AXES if a != lead_axis]
    joiner = "at" if lead_axis == "quality" else "for"
    vs = "than" if comparative else "to"
    text = f"{lead} {vs} {bo.model}, {joiner} {others[0]} and {others[1]}."
    # Replays: mention consistency when it separates the two (descriptive only; R5.5).
    if (v.consistency is not None and bo.consistency is not None
            and not v.consistency_low_confidence and not bo.consistency_low_confidence
            and abs(v.consistency - bo.consistency) >= 0.05):
        word = "more" if v.consistency > bo.consistency else "less"
        text = text[:-1] + f"; {word} consistent ({v.consistency:.2f} vs {bo.consistency:.2f})."
        deltas["consistency"] = round(v.consistency - bo.consistency, 4)
    return text, deltas


# --------------------------------------------------------------------------- #
# Selection helpers (deterministic tie-breaks)
# --------------------------------------------------------------------------- #
def _cpr_key(v: CandidateVerdict) -> float:
    return v.cost_per_request if v.cost_per_request is not None else float("inf")


def _pick_best_quality(e: List[CandidateVerdict]) -> CandidateVerdict:
    return sorted(e, key=lambda v: (-v.migration_confidence, v.hard_break_rate, _cpr_key(v), v.model))[0]


def _metric_desc(metric: str) -> str:
    return "median end-to-end latency" if metric == "p50" else "median ms per output token"


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def build_recommendation(verdicts: List[CandidateVerdict],
                         settings: Optional[Settings] = None,
                         weights: Optional[Weights] = None,
                         metrics_version: int = 2) -> Recommendation:
    s = settings or DEFAULT_SETTINGS
    w = (weights or weights_for("balanced", s)).normalised()
    rec = Recommendation(weights=w, latency_metric=s.latency_metric,
                         quality_tolerance=s.quality_tolerance,
                         latency_disclaimer=s.latency_disclaimer)
    rec.ineligible = {v.model: list(v.ineligible_reasons) for v in verdicts if not v.eligible}
    if metrics_version < 2:
        rec.notes.append("Measured by an older build; latency may include retry wait time.")

    elig = [v for v in verdicts if v.eligible]
    rec.eligible_models = [v.model for v in elig]

    # ---- 1. nobody passed the quality gate ---- #
    if not elig:
        rec.none_eligible = True
        if verdicts:
            from .aggregate import side_by_side
            rec.closest_by_fit = side_by_side(list(verdicts))[0].model
        why = "; ".join(f"{m}: {', '.join(r) or 'not eligible'}" for m, r in rec.ineligible.items())
        reason = "No candidate passed the quality gate" + (f" ({why})." if why else ".")
        rec.profiles = [Profile(kind=k, reason=reason) for k in _PROFILE_ORDER]
        _add_explanations(rec, verdicts, s)
        return rec

    # ---- 2. axis scores ---- #
    q_best = max(v.migration_confidence for v in elig)
    tol_pts = s.quality_tolerance

    def _quality_score(q: float) -> float:
        if tol_pts <= 0:
            return 100.0 if q >= q_best else 0.0
        return round(min(100.0, max(0.0, 100.0 * (1.0 - (q_best - q) / tol_pts))), 2)

    lat_vals = {v.model: _latency_metric(v, s.latency_metric) for v in elig}
    cost_scores = _ratio_scores({v.model: v.cost_per_request for v in elig})
    lat_scores = _ratio_scores(lat_vals)
    axes: Dict[str, CandidateAxes] = {}
    for v in elig:
        ax = CandidateAxes(model=v.model, quality=v.migration_confidence,
                           quality_score=_quality_score(v.migration_confidence),
                           cost=cost_scores[v.model], latency=lat_scores[v.model],
                           consistency=v.consistency)
        ax.composite, ax.complete = _composite(ax, w)
        axes[v.model] = ax
        if ax.cost is None and w.cost > 0:
            rec.notes.append(f"Cost not included for {v.model}: price not configured.")
        if ax.latency is None and w.latency > 0:
            rec.notes.append(f"Latency not included for {v.model}: fewer than "
                             f"{s.min_latency_samples} timed calls.")

    # ---- 3. Pareto ---- #
    ranked = sorted(axes.values(), key=lambda a: (-(a.composite or 0.0), a.model))
    for a in ranked:
        dom = next((b for b in ranked if b.model != a.model and _dominates(b, a)), None)
        a.pareto_efficient = dom is None
        a.dominated_by = dom.model if dom else None
    rec.axes = ranked

    by_model = {v.model: v for v in elig}

    def _bo_key(m: str):
        v = by_model[m]
        return (-(axes[m].composite or 0.0), -v.migration_confidence, _cpr_key(v), m)

    # ---- 4. Best overall: within the quality-tolerance window (same pool as the economical and
    #         latency picks, so the profiles never contradict each other), and an incomplete
    #         candidate can't win on a missing axis ---- #
    in_window = [m for m, v in by_model.items()
                 if v.migration_confidence >= q_best - s.quality_tolerance - 1e-9]
    order = sorted(in_window, key=_bo_key)
    best_any = order[0]
    complete = [m for m in order if axes[m].complete]
    bo_model = best_any
    if not axes[best_any].complete and complete:
        lead = (axes[best_any].composite or 0.0) - (axes[complete[0]].composite or 0.0)
        if lead <= s.missing_axis_margin:
            bo_model = complete[0]
    bo = by_model[bo_model]
    preset_name = w.preset.value.replace("_", "-")
    profiles: Dict[ProfileKind, Profile] = {
        ProfileKind.BEST_OVERALL: Profile(
            kind=ProfileKind.BEST_OVERALL, model=bo.model,
            reason=(f"Highest weighted score ({axes[bo.model].composite:.1f}) with {preset_name} weights "
                    f"(quality {w.quality:.0%}, cost {w.cost:.0%}, latency {w.latency:.0%})."),
            deltas={"quality_pts": 0.0, "cost_pct": 0.0 if bo.cost_per_request else None,
                    "latency_pct": 0.0 if lat_vals[bo.model] else None}),
    }

    # ---- 5. Best quality ---- #
    bq = _pick_best_quality(elig)
    profiles[ProfileKind.BEST_QUALITY] = Profile(
        kind=ProfileKind.BEST_QUALITY, model=bq.model,
        reason=f"Highest Migration Confidence ({bq.migration_confidence:.1f}).")

    # ---- 6. Most economical / lowest latency within the quality tolerance ---- #
    floor = bq.migration_confidence - s.quality_tolerance
    pool = [v for v in elig if v.migration_confidence >= floor - 1e-9]
    tol = f"{s.quality_tolerance:.0f}"

    priced = [v for v in pool if v.cost_per_request is not None]
    if priced:
        me = sorted(priced, key=lambda v: (v.cost_per_request, -v.migration_confidence, v.model))[0]
        profiles[ProfileKind.MOST_ECONOMICAL] = Profile(
            kind=ProfileKind.MOST_ECONOMICAL, model=me.model,
            reason=(f"Lowest cost per request (${me.cost_per_request:.5f}) among candidates within "
                    f"{tol} points of the best quality ({bq.migration_confidence:.0f})."))
    else:
        any_priced = any(v.cost_per_request is not None for v in elig)
        profiles[ProfileKind.MOST_ECONOMICAL] = Profile(
            kind=ProfileKind.MOST_ECONOMICAL,
            reason=(f"No candidate with a known price is within {tol} points of the best quality "
                    f"({bq.migration_confidence:.0f})." if any_priced
                    else "No eligible candidate has a configured price."))

    timed = [v for v in pool if lat_vals[v.model] is not None]
    if timed:
        ll = sorted(timed, key=lambda v: (lat_vals[v.model], -v.migration_confidence, v.model))[0]
        profiles[ProfileKind.LOWEST_LATENCY] = Profile(
            kind=ProfileKind.LOWEST_LATENCY, model=ll.model,
            reason=(f"Fastest by {_metric_desc(s.latency_metric)} among candidates within {tol} points "
                    f"of the best quality. Latency is indicative only (see note)."))
    else:
        any_timed = any(lat_vals[v.model] is not None for v in elig)
        profiles[ProfileKind.LOWEST_LATENCY] = Profile(
            kind=ProfileKind.LOWEST_LATENCY,
            reason=(f"No candidate with enough timed calls is within {tol} points of the best quality."
                    if any_timed else
                    f"No eligible candidate has at least {s.min_latency_samples} timed calls."))

    # ---- 7. Trade-offs vs Best overall ---- #
    for kind in (ProfileKind.BEST_QUALITY, ProfileKind.MOST_ECONOMICAL, ProfileKind.LOWEST_LATENCY):
        p = profiles[kind]
        if p.model is None:
            continue
        if p.model == bo.model:
            p.tradeoff = "Same as Best overall."
            p.deltas = {"quality_pts": 0.0, "cost_pct": 0.0, "latency_pct": 0.0}
            continue
        v = by_model[p.model]
        p.tradeoff, p.deltas = _tradeoff(kind, v, bo, lat_vals[v.model], lat_vals[bo.model])

    if len(elig) == 1:
        rec.notes.insert(0, "Only one candidate passed the quality gate.")
    rec.profiles = [profiles[k] for k in _PROFILE_ORDER]
    _add_explanations(rec, verdicts, s)
    return rec


def _add_explanations(rec, verdicts, s) -> None:
    from .explain import explain
    try:
        rec.explanations = explain(rec, list(verdicts), s.max_hard_break_rate)
    except Exception:  # noqa: BLE001 — explanatory text must never break a recommendation
        rec.explanations = {}


# --------------------------------------------------------------------------- #
# Run-level helpers (Phase 4): recompute, never trust stale stored verdicts
# --------------------------------------------------------------------------- #
def _repriced(results):
    """Copies of the run's results priced at the CURRENT public list prices (from the stored
    token counts), so old runs rank on today's prices and on the corrected reasoning billing.
    The stored results themselves are never modified."""
    from .config import estimate_cost
    out = []
    for r in results:
        u = r.usage
        if r.status == "ok" and (u.prompt_tokens is not None or u.completion_tokens is not None):
            c = estimate_cost(r.model, u.prompt_tokens or 0, u.completion_tokens or 0,
                              u.reasoning_tokens or 0)
            r = r.model_copy(update={"cost_estimate": c or 0.0, "cost_known": c is not None})
        out.append(r)
    return out


def refresh_verdicts(run, settings: Optional[Settings] = None) -> List[CandidateVerdict]:
    """Recompute a finished run's verdicts from its stored results and scores.

    Picks up label overrides, the current Settings thresholds, current public prices and the
    multi-objective metrics -- so runs saved by older builds become comparable too. Returns
    the verdicts (also assigned to run.verdicts); a run without verdicts is returned as-is.
    """
    from .aggregate import aggregate, side_by_side
    s = settings or DEFAULT_SETTINGS
    if not run.verdicts:
        return run.verdicts
    results = _repriced(run.results)
    volume = getattr(run.config, "monthly_volume", None) if run.config else None
    fresh = [aggregate(v.model, run.pair_scores, results, s.rubric, endpoint=v.endpoint,
                       samples=run.samples, settings=s, monthly_volume=volume)
             for v in run.verdicts]
    run.verdicts = side_by_side(fresh)
    return run.verdicts


def run_recommendation(run, settings: Optional[Settings] = None,
                       weights: Optional[Weights] = None) -> Optional[Recommendation]:
    """Fresh recommendation for a run (None if it has no verdicts). Uses the run's saved
    weights unless `weights` is given (e.g. a preview of another preset)."""
    s = settings or DEFAULT_SETTINGS
    verdicts = refresh_verdicts(run, s)
    if not verdicts:
        return None
    w = weights or getattr(run, "weights", None)
    if w is None:
        w = weights_for("balanced", s)
    elif w.preset != WeightPreset.CUSTOM:
        # A named preset follows the current Settings values (preset edits apply to old runs);
        # only custom weights are fixed numbers.
        w = weights_for(w.preset.value, s)
    return build_recommendation(verdicts, s, w, metrics_version=getattr(run, "metrics_version", 2))


NO_CACHING_CAVEAT = "Cost assumes no prompt caching; real cost may be lower for cache-heavy apps."
REPRICED_CAVEAT = ("Costs are list prices recomputed from each call's token counts at the current "
                   "public Bedrock rates (Standard tier, short context).")
