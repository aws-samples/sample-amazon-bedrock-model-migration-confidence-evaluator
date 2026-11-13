"""Phase 6 (multi-objective recommendation): PDF report."""
import fitz  # pymupdf
import pytest

from modelshift.report_pdf import render_pdf


def _cand(model, conf=90.0, eligible=True, cpr=0.006, reasons=None, rows=None):
    return {"model": model, "endpoint": "bedrock-runtime", "migration_confidence": conf,
            "verdict_band": "safe_drop_in", "distribution": {"compatible": 10}, "dimension_averages": {},
            "top_failure_reasons": [], "hard_break_rate": 0.0, "cost_total": 0.1, "cost_known": cpr is not None,
            "cost_per_request": cpr, "cost_per_1k_requests": None if cpr is None else cpr * 1000,
            "cost_vs_current_pct": -40.0, "ms_per_output_token_p50": 12.0, "latency_low_confidence": False,
            "latency_avg_ms": 900, "latency_p50_ms": 800, "latency_p95_ms": 2000, "scored": 10, "errored": 0,
            "error_rate": 0.0, "eligible": eligible, "ineligible_reasons": reasons or [], "errors": [],
            "evaluations": {"rows": rows or [], "total": len(rows or []), "truncated": False, "limit": 50}}


def _profile(kind, model, tradeoff=None, reason="r"):
    return {"kind": kind, "model": model, "tradeoff": tradeoff, "reason": reason, "deltas": {}}


def _rec(profiles, axes, none=False, closest=None, inel=None):
    return {"weights": {"preset": "balanced", "quality": 0.5, "cost": 0.3, "latency": 0.2},
            "latency_metric": "per_token", "quality_tolerance": 10.0, "eligible_models": [a["model"] for a in axes],
            "ineligible": inel or {}, "axes": axes, "profiles": profiles, "none_eligible": none,
            "closest_by_fit": closest, "notes": [], "latency_disclaimer": "Latency numbers are indicative only. XYZ."}


def _report(cands, rec):
    return {"run_id": "r1", "name": "pdf-test", "description": "", "status": "done", "error": None,
            "source": {"evaluable": 10, "found": 10, "sources": ["upload:x"], "detected_models": []},
            "candidates": cands, "best": cands[0] if cands else None, "recommendation": rec,
            "weights": rec["weights"] if rec else None,
            "latency_disclaimer": "Latency numbers are indicative only. XYZ.",
            "remediations": [], "top_failure_reasons_overall": [],
            "caveats": ["Cost assumes no prompt caching; real cost may be lower for cache-heavy apps.",
                        "Latency numbers are indicative only. XYZ."]}


def _text(pdf: bytes) -> str:
    """Page text with whitespace collapsed (narrow cells wrap words across lines)."""
    doc = fitz.open(stream=pdf, filetype="pdf")
    return " ".join(" ".join(p.get_text().split()) for p in doc)


def _ax(model, comp=90.0):
    return {"model": model, "quality": 90.0, "quality_score": 100.0, "cost": 100.0, "latency": 100.0,
            "composite": comp, "complete": True, "pareto_efficient": True}


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_multi_candidate_recommendation_page(theme):
    cands = [_cand("gpt-5.6-terra@medium"), _cand("gpt-5.4@medium", conf=87), _cand("luna@low", 84, False,
             reasons=["hard-break rate 9% > 5%"])]
    rec = _rec([_profile("best_overall", "gpt-5.6-terra@medium"),
                _profile("best_quality", "gpt-5.6-terra@medium", "Same as Best overall."),
                _profile("most_economical", "gpt-5.6-terra@medium", "Same as Best overall."),
                _profile("lowest_latency", "gpt-5.4@medium", "28% faster than gpt-5.6-terra@medium.")],
               [_ax("gpt-5.6-terra@medium", 94), _ax("gpt-5.4@medium", 80)],
               inel={"luna@low": ["hard-break rate 9% > 5%"]})
    pdf = render_pdf(_report(cands, rec), theme=theme)
    assert pdf[:4] == b"%PDF"
    txt = _text(pdf)
    for s in ("Recommendation", "BEST OVERALL", "LOWEST LATENCY", "Ranked with:", "Balanced",
              "Best fit (Migration Confidence alone)", "Quality gate", "Failed", "hard-break rate 9% > 5%",
              "Cost / 1K req", "28% faster"):
        assert s in txt, s
    assert txt.count("Latency numbers are indicative only") == 1   # full disclaimer once, not repeated


def test_none_eligible_page():
    cands = [_cand("a", 60, False, reasons=["confidence 60 < 65"])]
    rec = _rec([_profile(k, None, reason="No candidate passed") for k in
                ("best_overall", "best_quality", "most_economical", "lowest_latency")], [],
               none=True, closest="a", inel={"a": ["confidence 60 < 65"]})
    txt = _text(render_pdf(_report(cands, rec)))
    assert "No candidate is ready to recommend" in txt and "Closest by fit" in txt
    assert "BEST OVERALL" not in txt


def test_single_candidate_and_unknown_price():
    cands = [_cand("only", cpr=None)]
    rec = _rec([_profile("best_overall", "only")] + [_profile(k, "only", "Same as Best overall.") for k in
               ("best_quality", "most_economical", "lowest_latency")],
               [dict(_ax("only"), cost=None, complete=False)])
    txt = _text(render_pdf(_report(cands, rec)))
    assert "Price not configured" in txt
    assert "$0.00" not in txt


def test_report_without_recommendation_keeps_old_hero():
    cands = [_cand("m")]
    rep = _report(cands, None)
    txt = _text(render_pdf(rep))
    assert "RECOMMENDED CANDIDATE" in txt and "Quality gate" not in txt


def test_oversized_evaluation_does_not_break_the_pdf():
    # An evaluation whose original/candidate answers are each far longer than a page used to
    # raise reportlab LayoutError and fail the whole report.
    long = ("lorem ipsum dolor sit amet " * 1200).strip()
    row = {"sample_id": "s1", "label": "compatible", "hard_break": False, "overridden": False,
           "prompt": "q", "instructions": None, "messages": [{"role": "user", "content": "q"}],
           "original_response": long, "candidate_response": long, "original_model": "gpt-4.1",
           "dimensions": [], "judge_reason": None, "judge_rationale": None,
           "original_latency_ms": 100, "latency_ms": 100, "cost_estimate": 0.01}
    cands = [_cand("m", rows=[row])]
    rec = _rec([_profile("best_overall", "m")] + [_profile(k, "m", "Same as Best overall.") for k in
               ("best_quality", "most_economical", "lowest_latency")], [_ax("m")])
    pdf = render_pdf(_report(cands, rec))
    assert pdf[:4] == b"%PDF"
