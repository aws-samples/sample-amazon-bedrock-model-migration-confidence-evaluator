"""Retry and timeout policy shared by every model call (candidates and the judge).

Why this exists: a call that hangs must not hold a run hostage. The rules, in order:

* Every attempt has a hard time limit (CALL_READ_TIMEOUT_S, set on the HTTP and boto clients).
* Only transient failures are retried. A 4xx such as a validation or permission error would
  fail the same way again, so it fails at once instead of burning three more attempts.
* Back-off is exponential with jitter (half fixed, half random), capped, so many parallel
  workers that hit a throttle together do not retry in lockstep.
* A timeout is retried once, not three times: a call that hung for the full limit is usually
  stuck, and repeating it just multiplies the wait.
* A per-call deadline bounds the total time of all attempts plus back-off.

Pure functions only (no model calls), so the policy is easy to test.
"""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, ""))
        return v if v > 0 else default
    except ValueError:
        return default


# Per-attempt limits, applied to the HTTP and boto clients (see adapters.py).
CONNECT_TIMEOUT_S = _env_float("MODELSHIFT_CONNECT_TIMEOUT_S", 10.0)
CALL_READ_TIMEOUT_S = _env_float("MODELSHIFT_CALL_TIMEOUT_S", 120.0)
# Upper bound for ONE call across all its attempts and back-offs.
CALL_DEADLINE_S = _env_float("MODELSHIFT_CALL_DEADLINE_S", 300.0)

MAX_RETRIES = 3          # retries after the first attempt (throttles, 5xx, connection errors)
TIMEOUT_RETRIES = 1      # a timeout is retried once
BACKOFF_BASE_S = 1.0
THROTTLE_BASE_S = 2.0    # throttles back off harder: the service asked us to slow down
BACKOFF_CAP_S = 20.0

THROTTLED, TIMEOUT, SERVER, CONNECTION, FATAL, UNKNOWN = (
    "throttled", "timeout", "server", "connection", "fatal", "unknown")

_STATUS = re.compile(r"(?<![\d.])([1-5]\d\d)(?![\d.])")
_FATAL_NAMES = ("validationexception", "accessdenied", "unrecognizedclient", "resourcenotfound",
                "expiredtoken", "invalidsignature", "invalid_request_error", "validation_error",
                "unauthorized", "forbidden")
_SERVER_NAMES = ("internalserver", "serviceunavailable", "modelerror", "bad gateway", "overloaded")
_CONN_NAMES = ("connection reset", "connection refused", "connection aborted", "remote end closed",
               "remoteprotocolerror", "endpointconnectionerror", "connecterror", "broken pipe",
               "name resolution", "network is unreachable", "incompleteread")


def http_status(exc: BaseException) -> Optional[int]:
    """HTTP status carried by the exception, from its attributes or the first status-like
    number in its message (transports raise e.g. '503 from https://...')."""
    for attr in ("status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        v = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        if isinstance(v, int):
            return v
    m = _STATUS.search(str(exc)[:160])
    return int(m.group(1)) if m else None


def is_throttle(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("throttl" in msg or "toomanyrequests" in msg or "rate exceeded" in msg
            or "too many requests" in msg or re.search(r"(?<!\d)429(?!\d)", msg) is not None)


def is_timeout(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "timeout" in name or "timed out" in msg or "timeout" in msg or http_status(exc) in (408, 504)


def classify(exc: BaseException) -> str:
    """throttled | timeout | server | connection | fatal | unknown (unknown is retried)."""
    if is_throttle(exc):          # checked first: Bedrock reports throttling with HTTP 400
        return THROTTLED
    if is_timeout(exc):
        return TIMEOUT
    msg = str(exc).lower()
    status = http_status(exc)
    if (status is not None and status >= 500) or any(n in msg for n in _SERVER_NAMES):
        return SERVER
    if any(n in msg for n in _CONN_NAMES):
        return CONNECTION
    if (status is not None and 400 <= status < 500 and status not in (408, 409, 429)) \
            or any(n in msg for n in _FATAL_NAMES):
        return FATAL
    return UNKNOWN


def backoff_s(attempt: int, kind: str = SERVER, rng: Optional[random.Random] = None) -> float:
    """Wait before retry number attempt+1: exponential, capped, half fixed + half jitter."""
    base = THROTTLE_BASE_S if kind == THROTTLED else BACKOFF_BASE_S
    ceiling = min(BACKOFF_CAP_S, base * (2 ** attempt))
    return round(ceiling / 2.0 + (rng or random).uniform(0.0, ceiling / 2.0), 2)


@dataclass
class Retry:
    reason: str          # label shown in the UI: "throttled" | "timeout" | "error"
    wait_s: float
    max_attempts: int    # total attempts allowed for this kind of failure (first try included)
    kind: str


def decide(exc: BaseException, attempt: int, max_retries: int = MAX_RETRIES, elapsed_s: float = 0.0,
           deadline_s: Optional[float] = None, rng: Optional[random.Random] = None) -> Optional[Retry]:
    """Retry plan after a failed attempt (attempt counts from 0), or None to give up."""
    kind = classify(exc)
    if kind == FATAL:
        return None
    limit = min(TIMEOUT_RETRIES, max_retries) if kind == TIMEOUT else max_retries
    if attempt >= limit:
        return None
    wait = backoff_s(attempt, kind, rng)
    budget = CALL_DEADLINE_S if deadline_s is None else deadline_s
    # Do not start an attempt that cannot finish inside the deadline.
    if elapsed_s + wait + CALL_READ_TIMEOUT_S > budget:
        return None
    reason = {THROTTLED: "throttled", TIMEOUT: "timeout"}.get(kind, "error")
    return Retry(reason=reason, wait_s=wait, max_attempts=limit + 1, kind=kind)


def call_with_retries(fn: Callable[[], T], max_retries: int = MAX_RETRIES,
                      sleep: Callable[[float], None] = time.sleep,
                      deadline_s: Optional[float] = None) -> T:
    """Run fn(), retrying transient failures by the policy above. Raises the last error."""
    start = time.time()
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- classified below
            plan = decide(exc, attempt, max_retries, time.time() - start, deadline_s)
            if plan is None:
                raise
            sleep(plan.wait_s)
            attempt += 1
