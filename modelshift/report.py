"""Run report assembly (Stage 1 of the PDF-report feature).

build_run_report(run) is a PURE function: it reads only the already-computed
run state (verdicts, pair_scores, results, ingestion, samples, remediations)
and returns a structured dict. No I/O, no re-running. The PDF renderer and the
API layer consume this dict, so the data shape is tested independently of
rendering.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .config import DEFAULT_SETTINGS
from .recommend import NO_CACHING_CAVEAT, REPRICED_CAVEAT

# Per-candidate multi-objective metrics copied into the report (spec R7.5).
_MULTI_OBJECTIVE_FIELDS = (
    "cost_known", "ok_calls", "cost_per_request", "cost_per_1k_requests", "cost_vs_current_pct",
    "projected_monthly_cost", "avg_output_tokens", "avg_reasoning_tokens", "output_token_ratio",
    "ms_per_output_token_p50", "latency_samples", "latency_low_confidence", "error_rate",
    "eligible", "ineligible_reasons",
)
# Request replays / D7 Consistency (replay-consistency spec).
_CONSISTENCY_FIELDS = (
    "replays", "requests_scored", "consistency", "consistency_measured", "consistency_low_confidence",
    "unstable_requests", "flip_rate", "unstable", "latency_spread_p50",
)


def _exec_summary(stats):
    from .concurrency import exec_summary
    return exec_summary(stats)


def _latency_disclaimer(run) -> str:
    n = run.config.parallelism if getattr(run, "config", None) else 1
    base = DEFAULT_SETTINGS.latency_disclaimer
    if n > 1 and getattr(run, "exec_stats", None):
        return base.rstrip() + f" Measured with {n} parallel calls, which can add some latency per call."
    return base


def _least_consistent(run, model: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Lowest-D7 requests for a candidate (empty when replays were off)."""
    if not run.config or (getattr(run.config, "replays", 1) or 1) < 2:
        return []
    from .consistency import requests_for
    rows = [rc for rc in requests_for(model, run.results, run.pair_scores,
                                      DEFAULT_SETTINGS.instability_threshold) if rc.d7 is not None]
    rows.sort(key=lambda rc: (rc.d7, rc.sample_id))
    return [{"sample_id": rc.sample_id, "d7": rc.d7, "labels": [lb.value for lb in rc.labels],
             "flip": rc.flip, "unstable": rc.unstable} for rc in rows[:limit]]


def _pct(vals: List[float], q: float):
    if not vals:
        return None
    s = sorted(vals)
    return float(s[min(len(s) - 1, int(round(q * (len(s) - 1))))])


def _source_summary(run) -> Dict[str, Any]:
    ing = run.ingestion
    sources = sorted({s.source.object_uri for s in run.samples if s.source and s.source.object_uri})
    lat = sorted(s.original_latency_ms for s in run.samples if s.original_latency_ms is not None)
    prompt_toks = [s.usage.prompt_tokens for s in run.samples if s.usage and s.usage.prompt_tokens]
    total_toks = [s.usage.total_tokens for s in run.samples if s.usage and s.usage.total_tokens]

    def _avg(v):
        return round(sum(v) / len(v), 1) if v else None

    return {
        "sources": sources,
        "detected_models": [dm.model_dump() for dm in ing.detected_models] if ing else [],
        "found": ing.found if ing else 0,
        "evaluable": ing.evaluable if ing else len(run.samples),
        "dropped": {k.value: v for k, v in ing.dropped.items()} if ing else {},
        "redaction_rate": ing.redaction_rate if ing else 0.0,
        "avg_prompt_tokens": _avg(prompt_toks),
        "avg_total_tokens": _avg(total_toks),
        "original_latency_ms": {
            "count": len(lat), "avg": _avg(lat),
            "p50": _pct(lat, 0.50), "p95": _pct(lat, 0.95),
            "min": (lat[0] if lat else None), "max": (lat[-1] if lat else None),
        },
    }


def _caveats(run) -> List[str]:
    out: List[str] = []
    s = DEFAULT_SETTINGS
    ing = run.ingestion
    if ing:
        if ing.evaluable < s.small_sample_threshold:
            out.append(f"Small sample size ({ing.evaluable} evaluable) — low statistical confidence.")
        if ing.redaction_rate > s.high_redaction_threshold:
            out.append(f"High redaction rate ({ing.redaction_rate:.0%}) — limited signal.")
    return out


# Max per-candidate evaluations embedded in the PDF appendix. The full set is
# always available in the app's click-to-drill Evaluations view.
APPENDIX_LIMIT = 50

# Dimension field -> human label (mirrors the UI scores card).
_DIM_LABELS = [
    ("d1_semantic", "D1 Semantic"),
    ("d2_format", "D2 Format"),
    ("d3_factual", "D3 Factual"),
    ("d4_verbosity", "D4 Verbosity"),
    ("d5_instruction", "D5 Instruction"),
    ("d6_tool_call", "D6 Tool call"),
]


def _prompt_text(sample) -> str:
    msgs = sample.request.input_messages
    return msgs[-1].content if msgs else ""


def _response_text(text, tool_calls) -> str:
    """Mirror the UI: tool-call responses render as 'tool_call → [...]'."""
    if tool_calls:
        import json as _json
        try:
            return "tool_call → " + _json.dumps([t.model_dump() for t in tool_calls], indent=1)
        except Exception:
            return "tool_call → " + str(tool_calls)
    return text or ""


def _evaluations_for(run, model: str) -> Dict[str, Any]:
    """Per-evaluation rows for one candidate (first APPENDIX_LIMIT), mirroring the
    UI drill-down: request, golden, candidate response, per-dimension scores +
    reasons, judge assessment, hard-break, label, latency, cost."""
    results_by_id = {(r.sample_id, r.replay_index): r for r in run.results if r.model == model}
    scores_by_id = {(p.sample_id, p.replay_index): p for p in run.pair_scores if p.model == model}
    reps = sorted({k[1] for k in results_by_id} | {k[1] for k in scores_by_id}) or [0]
    d7 = {}
    if len(reps) > 1:
        from .consistency import requests_for
        d7 = {rc.sample_id: rc for rc in requests_for(model, run.results, run.pair_scores,
                                                      DEFAULT_SETTINGS.instability_threshold)}
    rows: List[Dict[str, Any]] = []
    for s, rep in ((s, rep) for s in run.samples for rep in reps):
        res = results_by_id.get((s.id, rep))
        score = scores_by_id.get((s.id, rep))
        if res is None and score is None:
            continue
        dims = []
        if score is not None:
            for field, label in _DIM_LABELS:
                ds = getattr(score, field, None)
                if ds is not None:
                    dims.append({"key": field, "label": label, "score": ds.score, "reason": ds.reason})
        label = "—"
        if score is not None:
            lab = score.override_label if (score.overridden and score.override_label) else score.label
            label = lab.value
        elif res is not None:
            label = res.status
        rows.append({
            "sample_id": s.id,
            "replay_index": rep,
            "replays": len(reps),
            "consistency": (d7[s.id].d7 if s.id in d7 else None),
            "unstable": (d7[s.id].unstable if s.id in d7 else False),
            "label": label,
            "hard_break": bool(score.hard_break) if score else False,
            "overridden": bool(score.overridden) if score else False,
            "prompt": _prompt_text(s),
            "instructions": s.request.instructions,
            "messages": [{"role": m.role, "content": str(m.content or "")} for m in s.request.input_messages],
            "original_model": s.legacy_model,
            "original_latency_ms": s.original_latency_ms,
            "original_tokens": (s.usage.total_tokens if s.usage else None),
            "golden": _response_text(s.golden.text, s.golden.tool_calls),
            "response": (_response_text(res.output.text, res.output.tool_calls) if (res and res.output) else ""),
            "reasoning_effort": (res.settings.reasoning_effort.value if res else None),
            "tokens": (res.usage.total_tokens if (res and res.usage) else None),
            "status": (res.status if res else "—"),
            "error": (res.error if res else None),
            "latency_ms": (res.latency_ms if res else None),
            "cost_estimate": (res.cost_estimate if res else None),
            "dimensions": dims,
            "judge_reason": (score.judge_reason if score else None),
            "judge_rationale": (score.judge_rationale if score else None),
        })
    total = len(rows)
    return {"rows": rows[:APPENDIX_LIMIT], "total": total,
            "truncated": total > APPENDIX_LIMIT, "limit": APPENDIX_LIMIT}


def build_run_report(run) -> Dict[str, Any]:
    """Assemble the full report payload for a run from existing computed state."""
    from .recommend import run_recommendation
    rec = run_recommendation(run, DEFAULT_SETTINGS) if run.verdicts else None  # refreshes verdicts
    candidates: List[Dict[str, Any]] = []
    for v in run.verdicts:
        # Distinct per-call errors for this candidate (deduped by message).
        errs = []
        seen = set()
        for r in run.results:
            if r.model == v.model and r.status == "error" and r.error and r.error not in seen:
                seen.add(r.error)
                errs.append(r.error)
        candidates.append({
            "model": v.model,
            "endpoint": v.endpoint,
            "migration_confidence": v.migration_confidence,
            "verdict_band": v.verdict_band.value,
            "distribution": {k.value: n for k, n in (v.distribution or {}).items()},
            "dimension_averages": v.dimension_averages,
            "top_failure_reasons": v.top_failure_reasons,
            "hard_break_rate": v.hard_break_rate,
            "cost_total": v.cost_total,
            "latency_avg_ms": v.latency_avg_ms,
            "latency_p50_ms": v.latency_p50_ms,
            "latency_p95_ms": v.latency_p95_ms,
            "scored": v.scored,
            "errored": v.errored,
            **{k: getattr(v, k) for k in _MULTI_OBJECTIVE_FIELDS},
            **{k: getattr(v, k) for k in _CONSISTENCY_FIELDS},
            "least_consistent": _least_consistent(run, v.model),
            "errors": errs,
            "evaluations": _evaluations_for(run, v.model),
        })
    # Rank best-first (verdicts are already side_by_side ranked, but be explicit).

    def _retest_summary(x):
        r = x.result
        if r is None:
            return None
        return {"status": r.status, "confidence_before": r.confidence_before,
                "confidence_after": r.confidence_after,
                "confidence_run_before": r.confidence_run_before,
                "confidence_run_after": r.confidence_run_after,
                "moved": dict(r.moved or {}), "regression_checked": r.regression_checked,
                "replays": r.replays, "consistency_before": r.consistency_before,
                "consistency_after": r.consistency_after,
                "cost": r.cost, "finished_at": r.finished_at,
                "change": r.change.model_dump() if r.change else None}

    remediations = [
        {"candidate": x.candidate, "type": x.change.type,
         "reason": x.target.reason, "evidence_count": x.evidence_count,
         "expected_effect": x.expected_effect,
         "before": x.change.before, "after": x.change.after, "placement": x.change.placement,
         "retest": _retest_summary(x)}
        for x in run.remediations
    ]
    reason_totals = Counter()
    for c in candidates:
        for fr in c["top_failure_reasons"]:
            if isinstance(fr, dict) and fr.get("reason"):
                reason_totals[fr["reason"]] += fr.get("count", 0)
    return {
        "run_id": run.run_id,
        "name": run.name or "Untitled run",
        "description": run.description or "",
        "status": run.status.value,
        "error": run.error,
        "replays": (max(1, int(getattr(run.config, "replays", 1) or 1)) if run.config else 1),
        "parallelism": (run.config.parallelism if run.config else 1),
        "exec_stats": getattr(run, "exec_stats", None),
        "exec_summary": _exec_summary(getattr(run, "exec_stats", None)),
        "source": _source_summary(run),
        "candidates": candidates,
        # "best" keeps its original meaning: best by FIT (Migration Confidence). The
        # all-things-considered pick lives in recommendation.profiles[best_overall].
        "best": candidates[0] if candidates else None,
        "recommendation": rec.model_dump(mode="json") if rec else None,
        "weights": run.weights.model_dump(mode="json") if getattr(run, "weights", None) else None,
        "latency_disclaimer": _latency_disclaimer(run),
        "remediations": remediations,
        "top_failure_reasons_overall": [{"reason": r, "count": n} for r, n in reason_totals.most_common(6)],
        "caveats": _caveats(run) + ([NO_CACHING_CAVEAT, REPRICED_CAVEAT, DEFAULT_SETTINGS.latency_disclaimer]
                                    if candidates else []),
    }


# --------------------------------------------------------------------------
# PDF rendering lives in report_pdf.py (UI-faithful design, dark/light themes).
# --------------------------------------------------------------------------

def render_run_pdf(report: Dict[str, Any], theme: str = "dark") -> bytes:
    """Render the assembled report dict to PDF bytes."""
    from .report_pdf import render_pdf
    return render_pdf(report, theme=theme)
