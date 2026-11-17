"""Confidence intervals and statistical ties for Migration Confidence.

Migration Confidence is a weighted share of answers (compatible = 1.0, drift = 0.7, ...).
With a few dozen requests one answer moves it by several points, so two candidates a point
apart are usually indistinguishable. This module puts a number on that:

* Per-candidate 95% interval: a request-level bootstrap. Requests are resampled with
  replacement and every answer of a drawn request comes along (so replays of one request
  stay together -- they are not independent evidence).
* Paired differences: every candidate answered the SAME requests, so the bootstrap reuses
  the same draws for all candidates and takes the interval of (A - B) per draw. That is much
  tighter than comparing two separate intervals, and is the right test for "is A better
  than B on these requests?".

Deterministic: the random seed is derived from the request ids, so a run always shows the
same intervals. Pure Python (no numpy dependency); the draw count shrinks for very large
runs so a refresh stays fast, and results are cached by a digest of the labels.
"""
from __future__ import annotations

import hashlib
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from .schemas import PairScore

LEVEL = 0.95
MAX_DRAWS = 1000
MIN_DRAWS = 200
WORK_BUDGET = 2_000_000      # draws x requests x candidates per computation
MIN_REQUESTS = 2             # below this an interval means nothing

Interval = Tuple[float, float]


class QualityStats:
    """Bootstrap intervals for each candidate and for each ordered pair (a - b)."""

    def __init__(self, ci: Dict[str, Interval], diff: Dict[Tuple[str, str], Interval],
                 requests: int, draws: int, level: float = LEVEL):
        self.ci = ci
        self.diff = diff
        self.requests = requests
        self.draws = draws
        self.level = level

    def diff_ci(self, a: str, b: str) -> Optional[Interval]:
        """Interval of MC(a) - MC(b), in confidence points."""
        return self.diff.get((a, b))

    def tied(self, a: str, b: str) -> bool:
        """True when the paired interval of a - b contains 0 (the data can't separate them)."""
        d = self.diff_ci(a, b)
        return d is not None and d[0] <= 0.0 <= d[1]


def _label_of(p: PairScore):
    return p.override_label if p.overridden and p.override_label else p.label


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _interval(vals: List[float], level: float) -> Interval:
    s = sorted(vals)
    a = (1.0 - level) / 2.0
    return round(_percentile(s, a), 1), round(_percentile(s, 1.0 - a), 1)


_CACHE: "OrderedDict[str, QualityStats]" = OrderedDict()
_CACHE_MAX = 64


def quality_stats(pair_scores: Sequence[PairScore], models: Sequence[str], rubric,
                  level: float = LEVEL) -> Optional[QualityStats]:
    """Bootstrap intervals for the given candidates, or None if there are too few requests."""
    models = list(dict.fromkeys(models))
    by_model = {m: [p for p in pair_scores if p.model == m] for m in models}
    requests = sorted({p.sample_id for ps in by_model.values() for p in ps})
    if len(requests) < MIN_REQUESTS or not models:
        return None

    digest = hashlib.sha256(repr((level, sorted(
        (p.model, p.sample_id, getattr(p, "replay_index", 0), _label_of(p).value)
        for ps in by_model.values() for p in ps))).encode("utf-8")).hexdigest()
    if digest in _CACHE:
        _CACHE.move_to_end(digest)
        return _CACHE[digest]

    idx = {sid: i for i, sid in enumerate(requests)}
    n = len(requests)
    # Per candidate, per request: summed answer weight and answer count.
    sums: Dict[str, List[float]] = {}
    counts: Dict[str, List[int]] = {}
    for m, ps in by_model.items():
        sw, sc = [0.0] * n, [0] * n
        for p in ps:
            i = idx[p.sample_id]
            sw[i] += rubric.confidence_weight(_label_of(p))
            sc[i] += 1
        sums[m], counts[m] = sw, sc

    draws = max(MIN_DRAWS, min(MAX_DRAWS, WORK_BUDGET // max(1, n * len(models))))
    rng = random.Random(int(hashlib.sha256("|".join(requests).encode("utf-8")).hexdigest()[:16], 16))
    rows = range(n)
    per_draw: Dict[str, List[Optional[float]]] = {m: [] for m in models}
    for _ in range(draws):
        pick = rng.choices(rows, k=n)
        for m in models:
            sw, sc = sums[m], counts[m]
            c = sum(map(sc.__getitem__, pick))
            per_draw[m].append(100.0 * sum(map(sw.__getitem__, pick)) / c if c else None)

    ci = {m: _interval([x for x in per_draw[m] if x is not None], level)
          for m in models if any(x is not None for x in per_draw[m])}
    diff: Dict[Tuple[str, str], Interval] = {}
    for a in models:
        for b in models:
            if a == b or a not in ci or b not in ci:
                continue
            d = [x - y for x, y in zip(per_draw[a], per_draw[b]) if x is not None and y is not None]
            if d:
                diff[(a, b)] = _interval(d, level)

    st = QualityStats(ci, diff, requests=n, draws=draws, level=level)
    _CACHE[digest] = st
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return st
