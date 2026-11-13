"""Short, data-driven "why this pick" text for the recommendation (about two lines per pick).

Every sentence is built from the run's own numbers and only appears when the data supports it
(e.g. the "slower per request" caveat for Lowest latency only when that is actually true).
Shared by the Results page, the PDF and the JSON report via ``Recommendation.explanations``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .schemas import CandidateVerdict, CompatLabel, ProfileKind, Recommendation


def _money1k(v: CandidateVerdict) -> str:
    x = v.cost_per_1k_requests
    return "—" if x is None else (f"${x:.2f}" if x >= 0.1 else f"${x:.3f}")


def _times(a: float, b: float) -> str:
    """How much cheaper/faster a is than b, as '30%' or '18×'."""
    if not a or not b:
        return ""
    r = b / a
    return f"{r:.0f}×" if r >= 2 else f"{(1 - a / b) * 100:.0f}%"


def _secs(ms: Optional[float]) -> str:
    return "—" if ms is None else (f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms")


def _answers(v: CandidateVerdict) -> str:
    n = v.scored or sum(v.distribution.values())
    ok = v.distribution.get(CompatLabel.COMPATIBLE, 0)
    hb = round(v.hard_break_rate * n)
    plural = "" if hb == 1 else "s"
    return (f"{ok} of {n} answers matched your current model; "
            f"{hb} hard break{plural} ({v.hard_break_rate:.1%})")


def explain(rec: Recommendation, verdicts: List[CandidateVerdict],
            max_hard_break_rate: float = 0.05) -> Dict[str, List[str]]:
    by = {v.model: v for v in verdicts}
    out: Dict[str, List[str]] = {}

    if rec.none_eligible:
        c = by.get(rec.closest_by_fit or "")
        if c:
            out["none"] = [f"No candidate passed the quality gate. Closest by fit is {c.model} at "
                           f"{c.migration_confidence:.1f} confidence; it failed on "
                           f"{', '.join(c.ineligible_reasons) or 'the gate'}.",
                           _answers(c) + "."]
        return out

    elig = [by[m] for m in rec.eligible_models if m in by]
    axes = {a.model: a for a in rec.axes}
    prof = {p.kind: p for p in rec.profiles}
    bo = by.get((prof.get(ProfileKind.BEST_OVERALL) or object()).__dict__.get("model") or "")
    if not bo or not elig:
        return out
    others = [v for v in elig if v.model != bo.model]

    def best(key, lower=True):
        vals = [v for v in elig if key(v) is not None]
        return (min if lower else max)(vals, key=key) if vals else None

    # ---- Best overall ---- #
    leads = []
    if best(lambda v: v.migration_confidence, lower=False) is bo:
        leads.append(f"the highest confidence ({bo.migration_confidence:.1f})")
    if best(lambda v: v.cost_per_request) is bo:
        leads.append(f"the lowest cost ({_money1k(bo)} per 1K requests)")
    if best(lambda v: v.latency_p50_ms) is bo:
        leads.append(f"the fastest requests (median {_secs(bo.latency_p50_ms)})")
    ax = axes.get(bo.model)
    ranked = sorted((a for a in rec.axes if a.composite is not None), key=lambda a: -a.composite)
    runner = next((a for a in ranked if a.model != bo.model), None)
    first = f"Top weighted score ({ax.composite:.1f})" if ax and ax.composite is not None else "Top weighted score"
    if runner and ax and ax.composite is not None:
        first += f", {ax.composite - runner.composite:.1f} points ahead of {runner.model}"
    first += (", with " + _join(leads) if leads else "") + "."
    second = _answers(bo) + f", gate {max_hard_break_rate:.0%}."
    if bo.consistency is not None and bo.replays > 1:
        second += (f" Consistent across {bo.replays} replays (D7 {bo.consistency:.2f}, "
                   f"{bo.unstable_requests} of {bo.consistency_measured} requests unstable).")
    else:
        dominated = sum(1 for a in rec.axes if a.dominated_by == bo.model)
        if dominated:
            second += f" Beats or ties {dominated} of {len(others)} others on quality, cost and latency together."
    out[ProfileKind.BEST_OVERALL.value] = [first, second]

    # ---- Best quality ---- #
    p = prof.get(ProfileKind.BEST_QUALITY)
    q = by.get(p.model) if p and p.model else None
    if q:
        rest = sorted((v for v in elig if v.model != q.model), key=lambda v: -v.migration_confidence)
        spread = q.migration_confidence - min(v.migration_confidence for v in elig)
        if rest and q.migration_confidence - rest[0].migration_confidence < 0.05:
            o = rest[0]
            why = ("fewer hard breaks" if q.hard_break_rate < o.hard_break_rate
                   else "lower cost" if (q.cost_per_request or 9e9) < (o.cost_per_request or 9e9) else "")
            l1 = f"Confidence {q.migration_confidence:.1f}, tied with {o.model}" + (f"; wins on {why}." if why else ".")
        elif rest:
            gap = q.migration_confidence - rest[0].migration_confidence
            l1 = (f"Confidence {q.migration_confidence:.1f}, {gap:.1f} "
                  f"points ahead of {rest[0].model}.")
        else:
            l1 = f"Confidence {q.migration_confidence:.1f}; the only candidate that passed the gate."
        if q is bo:
            l2 = (f"All {len(elig)} eligible candidates are within {spread:.1f} points, "
                  "so quality barely separates them."
                  if len(elig) > 1 and spread < 5 else "")
        else:
            both = bo.cost_per_request and q.cost_per_request
            cx = _times(bo.cost_per_request or 0, q.cost_per_request or 0) if both else ""
            dearer = bool(cx) and q.cost_per_request > bo.cost_per_request
            unit = "the cost" if cx.endswith("×") else "more cost"
            l2 = (f"{q.migration_confidence - bo.migration_confidence:+.1f} points vs Best overall"
                  + (f", at {cx} {unit}." if dearer else "."))
        out[ProfileKind.BEST_QUALITY.value] = [x for x in (l1, l2) if x]

    # ---- Most economical ---- #
    p = prof.get(ProfileKind.MOST_ECONOMICAL)
    e = by.get(p.model) if p and p.model else None
    if e and e.cost_per_request:
        dear = sorted((v for v in elig if v.model != e.model and v.cost_per_request), key=lambda v: v.cost_per_request)
        l1 = f"{_money1k(e)} per 1K requests"
        if dear:
            nxt = dear[0]
            l1 += (f", {_times(e.cost_per_request, nxt.cost_per_request)} below the next cheapest "
                   f"({nxt.model}, {_money1k(nxt)})")
            qb = max((v for v in elig if v.model != e.model), key=lambda v: v.migration_confidence)
            if qb.cost_per_request and qb.model != dear[0].model:
                l1 += (f" and {_times(e.cost_per_request, qb.cost_per_request)} cheaper than {qb.model}, "
                       "the closest on quality")
        l1 += "."
        l2 = (f"Answers are {e.output_token_ratio:.2f}× the length of your current model's, so no hidden output cost."
              if e.output_token_ratio is not None and e.output_token_ratio <= 1.1
              else (f"Answers are {e.output_token_ratio:.2f}× longer than your current model's."
                    if e.output_token_ratio else ""))
        out[ProfileKind.MOST_ECONOMICAL.value] = [x for x in (l1, l2) if x]

    # ---- Lowest latency ---- #
    p = prof.get(ProfileKind.LOWEST_LATENCY)
    f = by.get(p.model) if p and p.model else None
    if f:
        if rec.latency_metric == "per_token" and f.ms_per_output_token_p50 is not None:
            l1 = f"Fastest per word-piece ({f.ms_per_output_token_p50:.1f} ms/token)"
            if f is not bo and bo.ms_per_output_token_p50:
                l1 += f", {_times(f.ms_per_output_token_p50, bo.ms_per_output_token_p50)} faster than Best overall"
            l1 += "."
        else:
            l1 = f"Fastest median request ({_secs(f.latency_p50_ms)})."
        l2 = ""
        if (f is not bo and f.latency_p50_ms and bo.latency_p50_ms and f.latency_p50_ms > bo.latency_p50_ms):
            longer = (f" because answers are {f.output_token_ratio:.2f}× longer"
                      if (f.output_token_ratio or 0) > 1.2 else "")
            l2 = (f"But a whole request is slower{longer}: median {_secs(f.latency_p50_ms)} vs "
                  f"{_secs(bo.latency_p50_ms)} for {bo.model}.")
        elif f is not bo:
            l2 = f"Median request {_secs(f.latency_p50_ms)} vs {_secs(bo.latency_p50_ms)} for Best overall."
        out[ProfileKind.LOWEST_LATENCY.value] = [x for x in (l1, l2) if x]
    return out


def _join(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
