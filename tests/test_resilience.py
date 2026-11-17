"""Retry and timeout policy (modelshift/resilience.py) and how the adapters and judge use it."""
import random

import httpx
import pytest

from modelshift import resilience as R
from modelshift.adapters import BedrockResponsesAdapter, _boto_client, _shared_http_client
from modelshift.judge import Judge
from modelshift.schemas import CandidateOutput
from modelshift.scoring import score_pair
from test_aggregate import _sample as judge_sample
from test_engine import _sample


class _HttpErr(Exception):
    def __init__(self, status, msg="boom"):
        super().__init__(msg)
        self.response = {"ResponseMetadata": {"HTTPStatusCode": status}}


@pytest.mark.parametrize("exc,kind", [
    (RuntimeError("ThrottlingException: Rate exceeded"), R.THROTTLED),
    (RuntimeError("429 Too Many Requests"), R.THROTTLED),
    (_HttpErr(400, "ThrottlingException"), R.THROTTLED),           # Bedrock throttles with HTTP 400
    (httpx.ReadTimeout("slow"), R.TIMEOUT),
    (RuntimeError('Read timeout on endpoint URL: "https://bedrock-runtime..."'), R.TIMEOUT),
    (_HttpErr(504), R.TIMEOUT),
    (RuntimeError("503 from https://x/y: unavailable"), R.SERVER),
    (_HttpErr(500), R.SERVER),
    (RuntimeError("ServiceUnavailableException: try later"), R.SERVER),
    (ConnectionResetError("Connection reset by peer"), R.CONNECTION),
    (RuntimeError("400 from https://x/y: Unsupported value"), R.FATAL),
    (RuntimeError("An error occurred (ValidationException) when calling the Converse operation"), R.FATAL),
    (RuntimeError("403: access denied"), R.FATAL),
    (_HttpErr(404), R.FATAL),
    (RuntimeError("something odd happened"), R.UNKNOWN),
])
def test_classify(exc, kind):
    assert R.classify(exc) == kind


def test_url_digits_are_not_mistaken_for_a_status():
    assert R.http_status(RuntimeError("from https://bedrock-runtime.us-east-1.amazonaws.com:8443/x")) is None
    assert R.http_status(RuntimeError("502 from https://x")) == 502


def test_backoff_is_exponential_capped_and_jittered():
    rng = random.Random(7)
    for attempt in range(0, 12):
        w = R.backoff_s(attempt, R.SERVER, rng)
        ceil = min(R.BACKOFF_CAP_S, R.BACKOFF_BASE_S * 2 ** attempt)
        assert ceil / 2 <= w <= ceil
    assert R.backoff_s(30, R.SERVER, rng) <= R.BACKOFF_CAP_S
    # throttles start higher than other errors
    assert min(R.backoff_s(0, R.THROTTLED, random.Random(i)) for i in range(50)) >= R.THROTTLE_BASE_S / 2
    assert len({R.backoff_s(3, R.SERVER, random.Random(i)) for i in range(20)}) > 1   # jitter varies


def test_decide_rules():
    assert R.decide(RuntimeError("400 from x"), 0) is None                       # fatal: no retry
    t0 = R.decide(httpx.ReadTimeout("t"), 0)
    assert t0.reason == "timeout" and t0.max_attempts == 2
    assert R.decide(httpx.ReadTimeout("t"), 1) is None                           # a timeout retries once
    s = [R.decide(RuntimeError("503 from x"), a, max_retries=3) for a in range(4)]
    assert [bool(x) for x in s] == [True, True, True, False] and s[0].max_attempts == 4
    assert R.decide(RuntimeError("429"), 0).reason == "throttled"
    assert R.decide(RuntimeError("odd"), 0).reason == "error"
    # deadline: no attempt that cannot finish inside it
    assert R.decide(RuntimeError("503 from x"), 1, elapsed_s=R.CALL_DEADLINE_S - 10) is None
    assert R.decide(RuntimeError("503 from x"), 1, elapsed_s=5) is not None


def test_call_with_retries_succeeds_after_transient_failures():
    n, sleeps = {"i": 0}, []

    def fn():
        n["i"] += 1
        if n["i"] < 3:
            raise RuntimeError("503 from x")
        return "ok"
    assert R.call_with_retries(fn, sleep=sleeps.append) == "ok"
    assert n["i"] == 3 and len(sleeps) == 2 and sleeps[1] > 0.5


def test_call_with_retries_gives_up_on_fatal_immediately():
    n = {"i": 0}

    def fn():
        n["i"] += 1
        raise RuntimeError("403 forbidden")
    with pytest.raises(RuntimeError):
        R.call_with_retries(fn, sleep=lambda s: pytest.fail("must not sleep"))
    assert n["i"] == 1


# ---------------------------------------------------------------- adapter ---- #
def _adapter(transport, sleeps=None):
    sleep = sleeps.append if sleeps is not None else (lambda s: None)
    return BedrockResponsesAdapter(transport=transport, sleep=sleep)


def test_adapter_does_not_retry_a_client_error():
    n = {"i": 0}

    def t(payload):
        n["i"] += 1
        raise RuntimeError("400 from https://x: model does not accept this request")
    r = _adapter(t).generate(_sample(), "gpt-5.6-luna")
    assert r.status == "error" and n["i"] == 1 and r.attempts == 1


def test_adapter_retries_a_timeout_once_and_reports_two_attempts():
    n, seen = {"i": 0}, []

    def t(payload):
        n["i"] += 1
        raise httpx.ReadTimeout("read timed out")
    a = _adapter(t)
    r = a.generate(_sample(), "gpt-5.6-luna", on_retry=lambda *args: seen.append(args))
    assert r.status == "error" and n["i"] == 2 and r.attempts == 2
    assert len(seen) == 1 and seen[0][1] == 2 and seen[0][2] == "timeout"        # "attempt 2 of 2"


def test_adapter_server_error_backs_off_with_growing_jittered_waits():
    n, sleeps = {"i": 0}, []

    def t(payload):
        n["i"] += 1
        if n["i"] < 4:
            raise RuntimeError("503 from https://x")
        return {"output_text": "4", "usage": {}}
    r = _adapter(t, sleeps).generate(_sample(), "gpt-5.6-luna")
    assert r.status == "ok" and r.attempts == 4 and len(sleeps) == 3
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0 and 2.0 <= sleeps[2] <= 4.0
    assert all(s <= R.BACKOFF_CAP_S for s in sleeps)


def test_clients_carry_explicit_time_limits_and_no_hidden_retries():
    b = _boto_client("us-east-1").meta.config
    assert b.read_timeout == R.CALL_READ_TIMEOUT_S and b.connect_timeout == R.CONNECT_TIMEOUT_S
    assert b.retries.get("total_max_attempts") == 1
    t = _shared_http_client().timeout
    assert t.read == R.CALL_READ_TIMEOUT_S and t.connect == R.CONNECT_TIMEOUT_S


# ------------------------------------------------------------------ judge ---- #
def _scored():
    s = judge_sample(golden="The capital of France is Paris.")
    out = CandidateOutput(text="It's Paris, the French capital.")
    return s, out, score_pair(s, out, "m")


def test_judge_retries_a_transient_failure_then_uses_the_verdict():
    s, out, ps = _scored()
    n = {"i": 0}

    def t(prompt):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("503 from https://x")
        return {"semantic": 0.95, "instruction": 1.0, "reason": "equivalent", "rationale": "same"}
    refined = Judge(transport=t, sleep=lambda s_: None).refine(s, out, ps)
    assert n["i"] == 2 and refined.judge_rationale == "same"


def test_judge_that_stays_down_keeps_the_deterministic_score_instead_of_aborting():
    s, out, ps = _scored()
    before = ps.label

    def t(prompt):
        raise httpx.ReadTimeout("read timed out")
    refined = Judge(transport=t, sleep=lambda s_: None).refine(s, out, ps)
    assert refined.label == before and refined.judge_reason is None
    assert "Judge unavailable" in refined.judge_rationale


def test_judge_configuration_error_still_stops_the_run():
    s, out, ps = _scored()

    def t(prompt):
        raise RuntimeError("403 from https://x: AccessDeniedException")
    with pytest.raises(RuntimeError):
        Judge(transport=t, sleep=lambda s_: None).refine(s, out, ps)
