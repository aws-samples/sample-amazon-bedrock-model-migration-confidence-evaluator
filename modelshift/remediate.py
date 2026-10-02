"""Concrete, applyable remediation changes (Doc 3 §6).

The recommender clusters failing evaluations by judge reason. This module turns
each cluster's suggestion into a change that can actually be applied and re-tested:

  reasoning_effort        before = current effort, after = one step lower
                          (no prompt change).
  prompt_edit /
  format_schema           after = an instruction added to the system prompt
                          (``placement`` append|prepend). ``before`` is None
                          because every request keeps its own system prompt; the
                          instruction is added to each one. Templates are
                          pre-filled per reason and can be edited before applying.

Pure functions only: no model calls, no run mutation beyond the remediation passed
in. ``apply_change`` defines exactly what "apply" means so the re-test path and
the UI preview agree.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from typing import List, Optional, Tuple

from .schemas import (
    CompatLabel,
    GoldenKind,
    NormalizedSample,
    PairScore,
    ReasoningEffort,
    Remediation,
    RemediationChange,
)

CHANGE_TYPES = ("prompt_edit", "format_schema", "reasoning_effort")
PLACEMENTS = ("append", "prepend")
MAX_INSTRUCTION_CHARS = 4000
DEFAULT_REGRESSION_SAMPLE = 20  # Compatible evaluations re-run per re-test (-1 = all, 0 = off)


def regression_guard_ids(pair_scores: List[PairScore], model: str, exclude_ids: List[str],
                         available_ids: List[str], n: int, seed: str) -> List[str]:
    """Pick up to ``n`` of the candidate's currently Compatible evaluations (all if n < 0).

    The pick is seeded (by the remediation id) so the estimate and the actual re-test
    use the same evaluations, and re-running a re-test checks the same ones again.
    """
    if n == 0:
        return []
    excl, avail = set(exclude_ids), set(available_ids)
    pool = sorted({p.sample_id for p in pair_scores
                   if p.model == model and p.sample_id in avail and p.sample_id not in excl
                   and (p.override_label if p.overridden and p.override_label else p.label)
                   == CompatLabel.COMPATIBLE})
    if n < 0 or n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n))


_EFFORT_ORDER = [ReasoningEffort.MINIMAL, ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH]

# Per-reason instruction templates. Written as instructions to the model, in the
# imperative, so they can be pasted into a production system prompt unchanged.
PROMPT_TEMPLATES = {
    "missing_content": (
        "Answer completely. Include every item, detail and caveat the request asks for; "
        "do not summarize, shorten or omit parts of the answer."
    ),
    "factual_divergence": (
        "Use only facts stated in the conversation or the provided context. Copy numbers, "
        "dates, amounts, names, phone numbers and email addresses exactly as given. If a fact "
        "is not provided, say so instead of guessing."
    ),
    "instruction_violation": (
        "Follow every instruction in this system prompt exactly, including any constraints on "
        "scope, tone, language and output format. If instructions conflict, follow the most "
        "specific one."
    ),
    "added_verbosity": (
        "Be concise. Answer in the fewest words that fully address the request. Do not add "
        "preambles, restate the question, or include extra explanation unless asked."
    ),
    "format_change": (
        "Keep the output format exactly as specified. Do not wrap the answer in markdown or "
        "code fences, and do not add text before or after it."
    ),
}
_DEFAULT_TEMPLATE = (
    "Follow the instructions in this system prompt exactly and keep the same answer content "
    "and format as previously expected."
)


def lower_effort(effort: ReasoningEffort) -> Optional[ReasoningEffort]:
    """One step lower, or None if already at the lowest setting."""
    i = _EFFORT_ORDER.index(effort)
    return _EFFORT_ORDER[i - 1] if i > 0 else None


def _golden_json_keys(samples: List[NormalizedSample]) -> Optional[List[str]]:
    """Top-level keys the goldens use, when most of them are JSON objects."""
    objs = []
    for s in samples:
        txt = (s.golden.text or "").strip()
        if s.golden.kind == GoldenKind.JSON or txt.startswith("{"):
            try:
                v = json.loads(txt)
            except (ValueError, TypeError):
                continue
            if isinstance(v, dict):
                objs.append(v)
    if not samples or len(objs) * 2 < len(samples):
        return None
    counts = Counter(k for o in objs for k in o.keys())
    # keys present in at least half of the JSON goldens, most common first
    keys = [k for k, n in counts.most_common() if n * 2 >= len(objs)]
    return keys[:12] or None


def instruction_template(reason: Optional[str], samples: List[NormalizedSample]) -> str:
    """Pre-filled instruction for a cluster; format clusters cite the expected keys."""
    base = PROMPT_TEMPLATES.get(reason or "", _DEFAULT_TEMPLATE)
    if reason == "format_change":
        keys = _golden_json_keys(samples)
        if keys:
            return ("Respond with only a single valid JSON object - no prose, markdown or code "
                    "fences - using exactly these top-level keys: " + ", ".join(keys) + ".")
    return base


def concretize(rem: Remediation,
               samples: List[NormalizedSample],
               current_effort: ReasoningEffort = ReasoningEffort.MEDIUM) -> Remediation:
    """Fill rem.change.before/after (and placement) if the suggestion is still abstract.

    Idempotent: a change that already has ``after`` (e.g. user-edited) is left alone.
    """
    ch = rem.change
    if ch.after:
        return rem
    target = [s for s in samples if s.id in set(rem.target.sample_ids)]
    if ch.type == "reasoning_effort":
        lower = lower_effort(current_effort)
        if lower is not None:
            ch.before, ch.after, ch.placement = current_effort.value, lower.value, None
            rem.expected_effect = (f"Lower reasoning effort {current_effort.value} -> {lower.value} "
                                   "for shorter, more direct answers; no prompt change.")
            return rem
        # already at the lowest effort: fall back to a concise-output instruction
        ch.type = "prompt_edit"
    ch.before = None
    ch.after = instruction_template(rem.target.reason, target)
    ch.placement = "append"
    return rem


def validate_change(ch: RemediationChange) -> RemediationChange:
    """Reject changes that can't be applied. Raises ValueError with a user-facing message."""
    if ch.type not in CHANGE_TYPES:
        raise ValueError(f"change type must be one of {', '.join(CHANGE_TYPES)}")
    if ch.type == "reasoning_effort":
        valid = [e.value for e in ReasoningEffort]
        if ch.after not in valid:
            raise ValueError(f"reasoning effort must be one of {', '.join(valid)}")
        ch.placement = None
        return ch
    text = (ch.after or "").strip()
    if not text:
        raise ValueError("instruction text is empty")
    if len(text) > MAX_INSTRUCTION_CHARS:
        raise ValueError(f"instruction is longer than {MAX_INSTRUCTION_CHARS} characters")
    ch.after = text
    ch.placement = ch.placement or "append"
    if ch.placement not in PLACEMENTS:
        raise ValueError("placement must be 'append' or 'prepend'")
    return ch


def apply_change(sample: NormalizedSample,
                 change: RemediationChange,
                 effort: ReasoningEffort) -> Tuple[NormalizedSample, ReasoningEffort]:
    """Return (patched copy of the sample, effort to use). The original is untouched."""
    if change.type == "reasoning_effort":
        return sample, ReasoningEffort(change.after or effort.value)
    text = (change.after or "").strip()
    if not text:
        return sample, effort
    existing = (sample.request.instructions or "").strip()
    if not existing:
        new = text
    elif change.placement == "prepend":
        new = f"{text}\n\n{existing}"
    else:
        new = f"{existing}\n\n{text}"
    patched = sample.model_copy(deep=True)
    patched.request.instructions = new
    return patched, effort
