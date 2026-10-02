"""LLM-as-judge (Doc 3 §5).

Augments the deterministic PairScore: adjudicates semantic equivalence (D1) and
instruction-following (D5), and resolves ambiguous factual cases (D3), emitting a
categorical reason. Deterministic hard-breaks are NEVER overridden by the judge
(a broken schema / diverged fact / tool mismatch stays Incompatible).

The model call is isolated in a JudgeTransport so tests inject a fake judge.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, Optional, Protocol

from .config import DEFAULT_RUBRIC, Rubric
from .scoring import _cand_text, _golden_text, _label_from
from .schemas import (
    CandidateOutput,
    CompatLabel,
    DimensionScore,
    NormalizedSample,
    PairScore,
)

# Categorical reasons the judge may return (Doc 3 §5).
JUDGE_REASONS = {
    "equivalent",
    "missing_content",
    "factual_divergence",
    "format_change",
    "added_verbosity",
    "instruction_violation",
}


class JudgeTransport(Protocol):
    def __call__(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Return {semantic: 0..1, instruction: 0..1, factual_agree: bool, reason: str, rationale: str}."""
        ...


def judge_prompt(sample: NormalizedSample, output: CandidateOutput,
                 steering: Optional[str] = None) -> Dict[str, Any]:
    guidance = (
        "Judge COMPATIBILITY with the golden, not absolute quality. "
        "Do not treat the golden as ground truth for world facts; if they factually "
        "disagree and you cannot verify, return reason=factual_divergence."
    )
    prompt = {
        "instructions": sample.request.instructions or "",
        "request": [m.model_dump() for m in sample.request.input_messages],
        "golden": _golden_text(sample.golden),
        "candidate": _cand_text(output),
        "guidance": guidance,
    }
    if steering and steering.strip():
        # Custom evaluation steering / skill provided by the user in the run config.
        prompt["evaluation_steering"] = steering.strip()
    return prompt


def _judge_cache_key(sample_id: str, model: str, output_text: str, steering: str = "") -> str:
    raw = f"{sample_id}|{model}|judge|{output_text}|{steering}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Judge:
    def __init__(self, transport: JudgeTransport, rubric: Rubric = DEFAULT_RUBRIC,
                 steering: Optional[str] = None):
        self._transport = transport
        self._rubric = rubric
        self._steering = steering
        self._cache: Dict[str, Dict[str, Any]] = {}

    def refine(self, sample: NormalizedSample, output: CandidateOutput, ps: PairScore) -> PairScore:
        """Apply judge adjudication to a deterministic PairScore (in place-ish, returns it)."""
        # Never call the judge on a hard-break — the verdict is already settled.
        if ps.hard_break:
            ps.judge_reason = ps.judge_reason or "factual_divergence"
            ps.judge_rationale = ps.judge_rationale or "Deterministic hard break — verdict settled without the judge."
            return ps
        # Exact match already Compatible — no judge needed.
        if ps.label == CompatLabel.COMPATIBLE and ps.d1_semantic.score >= 0.999:
            ps.judge_reason = "equivalent"
            ps.judge_rationale = "Candidate output is an exact match of the original."
            return ps

        key = _judge_cache_key(sample.id, ps.model, _cand_text(output), self._steering or "")
        verdict = self._cache.get(key)
        if verdict is None:
            verdict = self._transport(judge_prompt(sample, output, self._steering))
            self._cache[key] = verdict

        sem = float(verdict.get("semantic", ps.d1_semantic.score))
        instr = float(verdict.get("instruction", 1.0))
        reason = verdict.get("reason") or "equivalent"

        # Judge upgrades/downgrades D1 (semantic) and sets D5 (instruction).
        ps.d1_semantic = DimensionScore(score=round(sem, 4), reason=f"judge:{reason}")
        ps.d5_instruction = DimensionScore(score=round(instr, 4), reason="judge instruction-following")
        ps.judge_reason = reason
        ps.judge_rationale = (verdict.get("rationale") or "").strip() or None

        # Instruction violation is a soft-fail that pulls the label down but is not a hard break.
        instruction_fail = instr < 0.5
        label = _label_from(ps, self._rubric, semantic=sem)
        if instruction_fail and label == CompatLabel.COMPATIBLE:
            label = CompatLabel.COMPATIBLE_WITH_DRIFT
        ps.label = label
        return ps


def make_bedrock_judge_transport(model_id: str, region: str = "us-east-1") -> JudgeTransport:
    """Real Bedrock judge via Converse (lazy import). Expects a JSON verdict back."""
    import boto3  # lazy

    client = boto3.client("bedrock-runtime", region_name=region)

    def _invoke(prompt: Dict[str, Any]) -> Dict[str, Any]:
        instruction = _judge_instruction(prompt)
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": instruction}]}],
            inferenceConfig={"maxTokens": 512},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        return _parse_judge_json(text)

    return _invoke


def _judge_instruction(prompt: Dict[str, Any]) -> str:
    """The shared strict-judge instruction used by both the Bedrock and LiteLLM
    judge transports (kept identical so scores don't drift by transport)."""
    return (
        "You are a strict migration-compatibility judge. Compare the CANDIDATE to the "
        "GOLDEN and return ONLY a JSON object with keys: semantic (0..1), instruction (0..1), "
        "factual_agree (true/false), reason (one of: equivalent, missing_content, "
        "factual_divergence, format_change, added_verbosity, instruction_violation), rationale. "
        "If the payload includes an 'evaluation_steering' field, follow those custom evaluation "
        "instructions in addition to the default guidance.\n\n"
        + json.dumps(prompt)
    )


def _parse_judge_json(text: str) -> Dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1]) if start >= 0 else {"semantic": 0.5, "reason": "equivalent"}


def make_litellm_judge_transport(base_url: str, api_key: str, model: str) -> JudgeTransport:
    """Judge via the LiteLLM proxy (/v1/chat/completions) so judge traffic is also
    proxied through the gateway when a run's call path is LiteLLM. `model` is the
    LiteLLM model-group alias for the judge (e.g. 'claude-sonnet-4.5')."""
    import httpx  # lazy

    def _invoke(prompt: Dict[str, Any]) -> Dict[str, Any]:
        body = {"model": model,
                "messages": [{"role": "user", "content": _judge_instruction(prompt)}],
                "max_tokens": 512}
        with httpx.Client(timeout=120) as client:
            r = client.post(f"{base_url.rstrip('/')}/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"}, json=body)
            if r.status_code >= 400:
                raise RuntimeError(f"{r.status_code}: {r.text[:400]}")
            data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        return _parse_judge_json(text)

    return _invoke


def deterministic_judge(_sleep: Optional[Callable[[float], None]] = None) -> JudgeTransport:
    """A no-LLM judge that echoes the deterministic signal (Doc 3 OQ-E1 fast pass)."""
    def _invoke(prompt: Dict[str, Any]) -> Dict[str, Any]:
        from .scoring import _jaccard
        sem = _jaccard(prompt.get("golden", ""), prompt.get("candidate", ""))
        equivalent = sem >= 0.85
        return {"semantic": sem, "instruction": 1.0, "factual_agree": True,
                "reason": "equivalent" if equivalent else "missing_content",
                "rationale": (f"Deterministic fast-pass: token overlap {sem:.2f} — "
                              + ("meaning preserved." if equivalent
                                 else "candidate appears to drop content vs the original."))}
    return _invoke
