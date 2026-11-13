"""Concurrent model calls (parallel-calls spec).

Pieces, all thread-based (calls are network-bound):

* ``SingleFlight``   -- a cache wrapper so two workers asking for the same key make ONE call.
* ``AdaptiveLimiter`` -- per-endpoint parallel-call limit: halves on a throttle, +1 after a
                        streak of clean completions, never above the configured N.
* ``server_slots()``  -- process-wide cap on calls in flight across all runs and re-tests.
* ``ExecStats``       -- counters the UI and reports show (throttles, time slowed, wall time).
* ``FailFast``        -- stops a run when most recent calls fail with the same non-throttle error.
* ``CallPool``        -- runs work items with at most N in flight. Workers never touch run
                        state: results are handed to ``on_done`` on the CALLER's thread, and
                        worker events go through a queue drained on the caller's thread.

With ``limit == 1`` the pool runs items inline, in order, on the caller's thread, so a
sequential run behaves exactly as before.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Deque, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")
V = TypeVar("V")

MAX_PARALLELISM = 16
DEFAULT_PARALLELISM = 4
RECOVERY_STREAK = 20          # clean completions before the limit goes back up by 1
THROTTLE_COOLDOWN_S = 2.0     # throttles inside this window after a cut don't cut again


class RunAbort(Exception):
    """Raised by CallPool.run when on_done asks the run to stop (fail fast)."""


def clamp_parallelism(n: Any, hi: int = MAX_PARALLELISM) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("parallelism must be a whole number")
    return max(1, min(hi, n))


# --------------------------------------------------------------------------- #
# Single-flight cache
# --------------------------------------------------------------------------- #
class SingleFlight(Generic[V]):
    """Dict-backed cache where concurrent misses on the same key wait for the first caller.

    ``get_or_compute(key, compute)`` returns ``(value, cached)``; ``cached`` is True when the
    value came from the cache or from another thread's in-progress computation.
    """

    def __init__(self, store: Optional[Dict[str, V]] = None):
        self.store: Dict[str, V] = store if store is not None else {}
        self._lock = threading.Lock()
        self._pending: Dict[str, threading.Event] = {}

    def get(self, key: str) -> Optional[V]:
        with self._lock:
            return self.store.get(key)

    def get_or_compute(self, key: str, compute: Callable[[], V]) -> Tuple[V, bool]:
        while True:
            with self._lock:
                if key in self.store:
                    return self.store[key], True
                ev = self._pending.get(key)
                if ev is None:
                    ev = self._pending[key] = threading.Event()
                    owner = True
                else:
                    owner = False
            if not owner:
                ev.wait()
                continue  # re-check: the owner stored a value (or failed and we retry)
            try:
                value = compute()
                with self._lock:
                    self.store[key] = value
                return value, False
            finally:
                with self._lock:
                    self._pending.pop(key, None)
                ev.set()


# --------------------------------------------------------------------------- #
# Adaptive per-endpoint limit
# --------------------------------------------------------------------------- #
class AdaptiveLimiter:
    """AIMD limit per endpoint key. Thread-safe. ``on_change(endpoint, limit, effective, reason)``
    fires (from whichever thread caused it) whenever an endpoint's effective limit moves."""

    def __init__(self, limit: int, enabled: bool = True,
                 on_change: Optional[Callable[[str, int, int, str], None]] = None,
                 clock: Callable[[], float] = time.time):
        self.limit = max(1, limit)
        self.enabled = enabled
        self.on_change = on_change
        self._clock = clock
        self._lock = threading.Lock()
        self._eff: Dict[str, int] = {}
        self._streak: Dict[str, int] = {}
        self._last_cut: Dict[str, float] = {}
        self._below_since: Optional[float] = None
        self.slowed_seconds = 0.0
        self.min_effective = self.limit

    def effective(self, endpoint: str) -> int:
        with self._lock:
            return self._eff.get(endpoint, self.limit)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._eff)

    def _update_slowed(self, now: float) -> None:
        below = any(v < self.limit for v in self._eff.values())
        if below and self._below_since is None:
            self._below_since = now
        elif not below and self._below_since is not None:
            self.slowed_seconds += now - self._below_since
            self._below_since = None

    def slowed_total(self) -> float:
        with self._lock:
            extra = (self._clock() - self._below_since) if self._below_since is not None else 0.0
            return round(self.slowed_seconds + extra, 1)

    def on_throttle(self, endpoint: str) -> None:
        if not self.enabled:
            return
        change = None
        with self._lock:
            now = self._clock()
            self._streak[endpoint] = 0
            cur = self._eff.get(endpoint, self.limit)
            if cur > 1 and now - self._last_cut.get(endpoint, -1e9) >= THROTTLE_COOLDOWN_S:
                new = max(1, cur // 2)
                self._eff[endpoint] = new
                self._last_cut[endpoint] = now
                self.min_effective = min(self.min_effective, new)
                self._update_slowed(now)
                change = (endpoint, self.limit, new, "throttle")
        if change and self.on_change:
            self.on_change(*change)

    def on_success(self, endpoint: str) -> None:
        if not self.enabled:
            return
        change = None
        with self._lock:
            cur = self._eff.get(endpoint, self.limit)
            if cur >= self.limit:
                return
            s = self._streak.get(endpoint, 0) + 1
            if s >= RECOVERY_STREAK:
                new = cur + 1
                self._eff[endpoint] = new
                self._streak[endpoint] = 0
                self._update_slowed(self._clock())
                change = (endpoint, self.limit, new, "recovered")
            else:
                self._streak[endpoint] = s
        if change and self.on_change:
            self.on_change(*change)


# --------------------------------------------------------------------------- #
# Server-wide cap
# --------------------------------------------------------------------------- #
class ServerSlots:
    """Counting cap on calls in flight across the whole process. Resizable."""

    def __init__(self, cap: int):
        self._cond = threading.Condition()
        self.cap = max(1, cap)
        self.in_use = 0

    def resize(self, cap: int) -> None:
        with self._cond:
            self.cap = max(1, cap)
            self._cond.notify_all()

    def try_acquire(self) -> bool:
        with self._cond:
            if self.in_use < self.cap:
                self.in_use += 1
                return True
            return False

    def release(self) -> None:
        with self._cond:
            self.in_use = max(0, self.in_use - 1)
            self._cond.notify_all()


_SERVER: Optional[ServerSlots] = None
_SERVER_LOCK = threading.Lock()
# run_id -> live view of that run's pool (for GET /api/v1/concurrency)
_POOLS: Dict[str, "CallPool"] = {}


def server_slots(cap: Optional[int] = None) -> ServerSlots:
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is None:
            _SERVER = ServerSlots(cap or 16)
        elif cap is not None and cap != _SERVER.cap:
            _SERVER.resize(cap)
        return _SERVER


def concurrency_snapshot() -> Dict[str, Any]:
    s = server_slots()
    with _SERVER_LOCK:
        pools = list(_POOLS.items())
    return {"server_inflight": s.in_use, "server_cap": s.cap,
            "runs": [{"run_id": rid, "inflight": p.inflight, "limit": p.limit,
                      "effective": p.limiter.snapshot(), "waiting": p.waiting}
                     for rid, p in pools]}


# --------------------------------------------------------------------------- #
# Run counters
# --------------------------------------------------------------------------- #
class ExecStats:
    """Thread-safe counters for one run (or re-test). ``to_dict`` is what the API/UI/report show."""

    def __init__(self, parallelism: int):
        self._lock = threading.Lock()
        self.parallelism = parallelism
        self.started = time.time()
        self.finished: Optional[float] = None
        self.throttles = 0
        self.throttle_retries = 0
        self.throttle_failures = 0
        self.by_endpoint: Counter = Counter()
        self.by_candidate: Counter = Counter()
        self.last_throttle: Dict[str, float] = {}
        self.max_inflight = 0

    def throttled(self, endpoint: str, candidate: str, retried: bool) -> None:
        with self._lock:
            self.throttles += 1
            if retried:
                self.throttle_retries += 1
            self.by_endpoint[endpoint] += 1
            self.by_candidate[candidate] += 1
            self.last_throttle[endpoint] = time.time()

    def throttle_failed(self) -> None:
        with self._lock:
            self.throttle_failures += 1

    def saw_inflight(self, n: int) -> None:
        with self._lock:
            self.max_inflight = max(self.max_inflight, n)

    def to_dict(self, limiter: Optional[AdaptiveLimiter] = None, inflight: int = 0) -> Dict[str, Any]:
        with self._lock:
            end = self.finished or time.time()
            return {
                "parallelism": self.parallelism, "inflight": inflight, "max_inflight": self.max_inflight,
                "effective": limiter.snapshot() if limiter else {},
                "min_effective": limiter.min_effective if limiter else self.parallelism,
                "slowed_seconds": limiter.slowed_total() if limiter else 0.0,
                "throttles": self.throttles, "throttle_retries": self.throttle_retries,
                "throttle_failures": self.throttle_failures,
                "throttles_by_endpoint": dict(self.by_endpoint),
                "throttles_by_candidate": dict(self.by_candidate),
                "last_throttle": dict(self.last_throttle),
                "wall_seconds": round(end - self.started, 1),
            }


def exec_summary(stats: Optional[Dict[str, Any]]) -> Optional[str]:
    """One line for the results page / reports, e.g. 'Sent 4 at a time · 7 throttles · ...'."""
    if not stats:
        return None

    def dur(s: float) -> str:
        s = int(round(s or 0))
        return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"
    n = stats.get("parallelism", 1)
    parts = ["Sent one call at a time" if n == 1 else f"Sent {n} at a time"]
    t = stats.get("throttles", 0)
    parts.append(f"{t} throttle{'' if t == 1 else 's'}" if t else "no throttling")
    if stats.get("slowed_seconds") and stats.get("min_effective", n) < n:
        parts.append(f"slowed to {stats['min_effective']} for {dur(stats['slowed_seconds'])}")
    if stats.get("throttle_failures"):
        parts.append(f"{stats['throttle_failures']} failed after retries")
    parts.append(f"wall time {dur(stats.get('wall_seconds', 0))}")
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# Fail fast
# --------------------------------------------------------------------------- #
class FailFast:
    """Window of recent outcomes; ``record`` returns an abort message when too many recent calls
    failed with the same (non-throttle) error."""

    def __init__(self, share: Optional[float] = 0.5, window: int = 20):
        self.share = share
        self.window: Deque[Optional[str]] = deque(maxlen=window)

    @staticmethod
    def _norm(err: str) -> str:
        return " ".join((err or "").split())[:80]

    def record(self, error: Optional[str], throttle: bool = False) -> Optional[str]:
        if self.share is None:
            return None
        self.window.append(None if (error is None or throttle) else self._norm(error))
        if len(self.window) < self.window.maxlen:
            return None
        errs = Counter(e for e in self.window if e)
        if not errs:
            return None
        msg, n = errs.most_common(1)[0]
        if n / len(self.window) >= self.share:
            return f"Stopped early: {n} of the last {len(self.window)} calls failed with: {msg}"
        return None


# --------------------------------------------------------------------------- #
# Pool
# --------------------------------------------------------------------------- #
class CallContext:
    """Handed to the worker function: report events and throttles from the worker thread."""

    def __init__(self, pool: "CallPool", item: Any, endpoint: str):
        self._pool, self.item, self.endpoint = pool, item, endpoint
        self.throttled = False

    def emit(self, etype: str, **data: Any) -> None:
        if self._pool.limit == 1:   # inline: we ARE the caller's thread, deliver now (live view)
            self._pool._drain()
            self._pool.on_event(etype, data)
        else:
            self._pool._events.put((etype, data))

    def throttle(self) -> None:
        self.throttled = True
        self._pool.limiter.on_throttle(self.endpoint)


class CallPool:
    def __init__(self, limit: int,
                 limiter: Optional[AdaptiveLimiter] = None,
                 server: Optional[ServerSlots] = None,
                 cancel: Optional[Callable[[], bool]] = None,
                 on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 stats: Optional[ExecStats] = None,
                 run_id: Optional[str] = None,
                 poll_s: float = 0.05):
        self.limit = max(1, limit)
        self.limiter = limiter or AdaptiveLimiter(self.limit, enabled=False)
        self.server = server
        self.cancel = cancel or (lambda: False)
        self.on_event = on_event or (lambda t, d: None)
        self.stats = stats or ExecStats(self.limit)
        self.run_id = run_id
        self._poll = poll_s
        self._events: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
        self.inflight = 0
        self.waiting = False
        if self.limiter.on_change is None:
            self.limiter.on_change = lambda ep, lim, eff, why: self._events.put(
                ("concurrency", {"endpoint": ep, "limit": lim, "effective": eff, "reason": why}))

    def _drain(self) -> None:
        while True:
            try:
                etype, data = self._events.get_nowait()
            except queue.Empty:
                return
            self.on_event(etype, data)

    def run(self, items: Sequence[T],
            fn: Callable[[T, CallContext], Any],
            on_done: Callable[[T, Any, CallContext], Optional[str]],
            endpoint: Callable[[T], str] = lambda _i: "default") -> str:
        """Run every item. Returns 'done' or 'cancelled'; raises RunAbort when on_done returns
        an abort message. on_done runs on this (the caller's) thread, in completion order."""
        if self.run_id:
            with _SERVER_LOCK:
                _POOLS[self.run_id] = self
        try:
            if self.limit == 1:
                return self._run_inline(items, fn, on_done, endpoint)
            return self._run_threads(items, fn, on_done, endpoint)
        finally:
            self.inflight = 0
            self.waiting = False
            if self.run_id:
                with _SERVER_LOCK:
                    if _POOLS.get(self.run_id) is self:
                        _POOLS.pop(self.run_id, None)

    def _finish_one(self, item, value, ctx, on_done) -> None:
        if not ctx.throttled:
            self.limiter.on_success(ctx.endpoint)
        self._drain()
        msg = on_done(item, value, ctx)
        self._drain()
        if msg:
            raise RunAbort(msg)

    def _run_inline(self, items, fn, on_done, endpoint) -> str:
        for item in items:
            if self.cancel():
                return "cancelled"
            ctx = CallContext(self, item, endpoint(item))
            got = self.server.try_acquire() if self.server else True
            while not got:  # sequential run still respects the server-wide cap
                self._set_waiting(True)
                time.sleep(self._poll)
                if self.cancel():
                    return "cancelled"
                got = self.server.try_acquire()
            self._set_waiting(False)
            self.inflight = 1
            self.stats.saw_inflight(1)
            try:
                value = fn(item, ctx)
            finally:
                self.inflight = 0
                if self.server:
                    self.server.release()
            self._finish_one(item, value, ctx, on_done)
        return "done"

    def _set_waiting(self, w: bool) -> None:
        if w != self.waiting:
            self.waiting = w
            s = self.server
            self._events.put(("waiting_capacity", {"run_waiting": w,
                                                   "server_inflight": s.in_use if s else 0,
                                                   "server_cap": s.cap if s else 0}))
            self._drain()

    def _run_threads(self, items, fn, on_done, endpoint) -> str:
        pending: List[T] = list(items)
        futs: Dict[Future, Tuple[T, CallContext]] = {}
        by_ep: Counter = Counter()
        ex = ThreadPoolExecutor(max_workers=self.limit, thread_name_prefix="modelshift-call")
        cancelled = False
        try:
            while pending or futs:
                if not cancelled and self.cancel():
                    cancelled = True
                if cancelled:
                    # Stop now: in-flight calls finish in the background, their results are dropped.
                    return "cancelled"
                # Submit while there is room: run limit, per-endpoint limit, server-wide cap.
                blocked_by_server = False
                i = 0
                while i < len(pending) and len(futs) < self.limit:
                    it = pending[i]
                    ep = endpoint(it)
                    if by_ep[ep] >= self.limiter.effective(ep):
                        i += 1
                        continue
                    if self.server and not self.server.try_acquire():
                        blocked_by_server = True
                        break
                    pending.pop(i)
                    ctx = CallContext(self, it, ep)
                    f = ex.submit(fn, it, ctx)
                    if self.server:
                        f.add_done_callback(lambda _f, s=self.server: s.release())
                    futs[f] = (it, ctx)
                    by_ep[ep] += 1
                self.inflight = len(futs)
                self.stats.saw_inflight(self.inflight)
                self._set_waiting(blocked_by_server and not futs)
                if not futs:
                    time.sleep(self._poll)  # everything is waiting on capacity
                    self._drain()
                    continue
                done, _ = wait(list(futs), timeout=self._poll, return_when=FIRST_COMPLETED)
                self._drain()
                for f in done:
                    it, ctx = futs.pop(f)
                    by_ep[ctx.endpoint] -= 1
                    self.inflight = len(futs)
                    try:
                        self._finish_one(it, f.result(), ctx, on_done)
                    except RunAbort:
                        pending.clear()
                        for g in list(futs):  # let the others finish, then stop
                            g.result()
                        self._drain()
                        raise
            return "done"
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            self._drain()
