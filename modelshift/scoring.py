"""Deterministic compatibility scoring (Doc 3 §4).

Six dimensions scored deterministically (the semantic D1 base is a lexical
similarity; the LLM judge in Stage 4 augments D1/D5 and adjudicates D3):
  D1 semantic     — token-overlap (Jaccard) as a cheap base signal
  D2 format       — structure match (json validity/schema, format class)
  D3 factual      — atomic-fact agreement (numbers, phones, emails, entities)
  D4 verbosity    — length ratio candidate/golden
  D5 instruction  — placeholder base 1.0 (judge refines in Stage 4)
  D6 tool_call    — name + argument match (only for tool-call goldens)

Labeling applies the hard-break rule (D2/D3/D6 break caps at Incompatible)
and the exact-match short-circuit (identical output -> Compatible).
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Set, Tuple

from .config import DEFAULT_RUBRIC, Rubric
from .schemas import (
    CandidateOutput,
    CompatLabel,
    DimensionScore,
    Golden,
    GoldenKind,
    NormalizedSample,
    PairScore,
    ToolCall,
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_PHONE_RE = re.compile(r"\b(?:\d[\s.-]?){7,}\d\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def _tokens(text: str) -> Set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _format_class(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "empty"
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            json.loads(s)
            return "json"
        except json.JSONDecodeError:
            pass
    if re.search(r"^\s*[-*]\s+", s, re.MULTILINE) or re.search(r"^\s*\d+\.\s+", s, re.MULTILINE):
        return "list"
    if "```" in s or re.search(r"^\s{4}", s, re.MULTILINE):
        return "code"
    return "prose"


def _facts(text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    t = text or ""
    nums = {n.replace(",", "") for n in _NUM_RE.findall(t)}
    phones = {re.sub(r"[\s.-]", "", p) for p in _PHONE_RE.findall(t)}
    emails = {e.lower() for e in _EMAIL_RE.findall(t)}
    # phones are also captured as numbers; prefer phone bucket
    nums = {n for n in nums if not any(n in p for p in phones)}
    return nums, phones, emails


# --------------------------------------------------------------------------- #
# Dimension scorers
# --------------------------------------------------------------------------- #
def score_d1_semantic(golden_text: str, cand_text: str) -> DimensionScore:
    j = _jaccard(golden_text, cand_text)
    return DimensionScore(score=round(j, 4), reason=f"token-overlap={j:.2f}")


def score_d2_format(golden_text: str, cand_text: str) -> Tuple[DimensionScore, bool]:
    gclass, cclass = _format_class(golden_text), _format_class(cand_text)
    if gclass == "json":
        # hard requirement: candidate must be valid json with same top-level keys
        try:
            gk = set(json.loads(golden_text).keys()) if isinstance(json.loads(golden_text), dict) else set()
        except (json.JSONDecodeError, AttributeError):
            gk = set()
        try:
            cj = json.loads(cand_text)
            ck = set(cj.keys()) if isinstance(cj, dict) else set()
            if gk and gk != ck:
                return DimensionScore(score=0.3, reason=f"json key drift {gk}!={ck}"), True
            return DimensionScore(score=1.0, reason="json schema match"), False
        except json.JSONDecodeError:
            return DimensionScore(score=0.0, reason="candidate not valid json (golden was json)"), True
    if gclass == cclass:
        return DimensionScore(score=1.0, reason=f"format match ({gclass})"), False
    return DimensionScore(score=0.5, reason=f"format drift {gclass}->{cclass}"), False


def score_d3_factual(golden_text: str, cand_text: str) -> Tuple[DimensionScore, bool]:
    gn, gp, ge = _facts(golden_text)
    cn, cp, ce = _facts(cand_text)
    # A concrete fact present in golden but contradicted (different value) in candidate is a break.
    phone_break = bool(gp) and bool(cp) and not (gp & cp)
    email_break = bool(ge) and bool(ce) and not (ge & ce)
    if phone_break or email_break:
        return DimensionScore(score=0.0, reason="factual divergence (phone/email)"), True
    # number agreement (soft): if golden has numbers, how many appear in candidate
    if gn:
        agree = len(gn & cn) / len(gn)
        if agree < 0.5 and cn:
            return DimensionScore(score=round(agree, 4), reason="numeric divergence"), agree == 0.0 and bool(gp)
        return DimensionScore(score=round(max(agree, 0.5), 4), reason="numbers largely agree"), False
    return DimensionScore(score=1.0, reason="no concrete facts to diverge"), False


def score_d4_verbosity(golden_text: str, cand_text: str, rubric: Rubric) -> DimensionScore:
    gl = max(len(golden_text or ""), 1)
    cl = len(cand_text or "")
    ratio = cl / gl
    if ratio > rubric.verbosity_drift_ratio or ratio < rubric.verbosity_drift_ratio_low:
        return DimensionScore(score=0.5, reason=f"verbosity drift ratio={ratio:.2f}")
    return DimensionScore(score=1.0, reason=f"comparable length ratio={ratio:.2f}")


def score_d6_tool_call(golden: List[ToolCall], cand: Optional[List[ToolCall]]) -> Tuple[DimensionScore, bool]:
    if not cand:
        return DimensionScore(score=0.0, reason="golden was a tool call; candidate produced none"), True
    g0, c0 = golden[0], cand[0]
    if g0.name != c0.name:
        return DimensionScore(score=0.0, reason=f"tool name mismatch {g0.name}!={c0.name}"), True
    ga = _as_dict(g0.arguments)
    ca = _as_dict(c0.arguments)
    if ga and set(ga.keys()) != set(ca.keys()):
        return DimensionScore(score=0.4, reason="tool argument schema drift"), True
    if ga == ca:
        return DimensionScore(score=1.0, reason="tool call matches"), False
    return DimensionScore(score=0.7, reason="tool name + keys match, values differ"), False


def _as_dict(args):
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            v = json.loads(args)
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


# --------------------------------------------------------------------------- #
# Labeling
# --------------------------------------------------------------------------- #
def _golden_text(golden: Golden) -> str:
    return (golden.text or "").strip()


def _cand_text(output: CandidateOutput) -> str:
    return (output.text or "").strip()


def score_pair(sample: NormalizedSample,
               output: CandidateOutput,
               model: str,
               rubric: Rubric = DEFAULT_RUBRIC) -> PairScore:
    golden = sample.golden
    gtext = _golden_text(golden)
    ctext = _cand_text(output)

    ps = PairScore(sample_id=sample.id, model=model, label=CompatLabel.PARTIAL)

    # Tool-call golden path (D6)
    if golden.kind == GoldenKind.TOOL_CALL:
        d6, hard = score_d6_tool_call(golden.tool_calls or [], output.tool_calls)
        ps.d6_tool_call = d6
        ps.hard_break = hard
        if hard:
            ps.label = CompatLabel.INCOMPATIBLE
        elif d6.score >= 0.99:
            ps.label = CompatLabel.COMPATIBLE
        elif d6.score >= 0.6:
            ps.label = CompatLabel.COMPATIBLE_WITH_DRIFT
        else:
            ps.label = CompatLabel.PARTIAL
        return ps

    # Exact / near-exact short-circuit (Doc 3 §4.2 note)
    if rubric.exact_match_short_circuit and gtext and gtext.strip().lower() == ctext.strip().lower():
        ps.d1_semantic = DimensionScore(score=1.0, reason="exact match")
        ps.d2_format = DimensionScore(score=1.0, reason="exact match")
        ps.d3_factual = DimensionScore(score=1.0, reason="exact match")
        ps.d4_verbosity = DimensionScore(score=1.0, reason="exact match")
        ps.d5_instruction = DimensionScore(score=1.0, reason="exact match")
        ps.label = CompatLabel.COMPATIBLE
        return ps

    ps.d1_semantic = score_d1_semantic(gtext, ctext)
    ps.d2_format, hard2 = score_d2_format(gtext, ctext)
    ps.d3_factual, hard3 = score_d3_factual(gtext, ctext)
    ps.d4_verbosity = score_d4_verbosity(gtext, ctext, rubric)
    ps.d5_instruction = DimensionScore(score=1.0, reason="deterministic base (judge refines)")
    ps.hard_break = hard2 or hard3
    ps.label = _label_from(ps, rubric, semantic=ps.d1_semantic.score)
    return ps


def _label_from(ps: PairScore, rubric: Rubric, semantic: float) -> CompatLabel:
    # Hard break caps at Incompatible regardless of eloquence.
    if ps.hard_break:
        return CompatLabel.INCOMPATIBLE
    high = semantic >= rubric.semantic_high
    moderate = semantic >= rubric.semantic_moderate
    format_ok = ps.d2_format.score >= 0.99
    drift = ps.d4_verbosity.score < 0.99 or ps.d2_format.score < 0.99
    if high and format_ok and not drift:
        return CompatLabel.COMPATIBLE
    if high and not drift:
        return CompatLabel.COMPATIBLE
    if high or (moderate and format_ok):
        return CompatLabel.COMPATIBLE_WITH_DRIFT
    if moderate:
        return CompatLabel.PARTIAL
    return CompatLabel.INCOMPATIBLE
