"""Parallel calls, Phases 1-2: thread-safe caches, per-call retry observer, settings, and the
CallPool / AdaptiveLimiter / FailFast / ServerSlots engine (not yet wired into runs)."""
import threading
import time

import pytest

from modelshift import api as api_mod
from modelshift.adapters import BedrockResponsesAdapter
from modelshift.concurrency import (AdaptiveLimiter, CallPool, ExecStats, FailFast, RunAbort,
                                    ServerSlots, SingleFlight, exec_summary)
from modelshift.config import DEFAULT_SETTINGS
from modelshift.judge import Judge
from modelshift.schemas import CandidateOutput, CandidateSettings, CompatLabel, DimensionScore, PairScore
from test_phase1_cost_latency import _sample
from test_reasoning_efforts import _client


# ----------------------------- Phase 1: shared state ----------------------------- #
def test_single_flight_computes_once_under_contention():
    sf, calls = SingleFlight(), {"n": 0}
    gate = threading.Event()

    def compute():
        calls["n"] += 1
        gate.wait(1)
        return "v"
    out = []
    ts = [threading.Thread(target=lambda: out.append(sf.get_or_compute("k", compute))) for _ in range(6)]
    [t.start() for t in ts]
    time.sleep(0.05)
    gate.set()
    [t.join() for t in ts]
    assert calls["n"] == 1 and sorted(c for _, c in out) == [False] + [True] * 5


def test_single_flight_retries_after_owner_failure():
    sf = SingleFlight()
    with pytest.raises(RuntimeError):
        sf.get_or_compute("k", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert sf.get_or_compute("k", lambda: 7) == (7, False)


def test_adapter_sends_identical_concurrent_calls_once():
    n = {"calls": 0}

    def transport(p):
        n["calls"] += 1
        time.sleep(0.1)
        return {"output_text": "4", "usage": {}}
    a = BedrockResponsesAdapter(transport)
    res = []
    ts = [threading.Thread(target=lambda: res.append(a.generate(_sample(), "gpt-5.6-luna@low", CandidateSettings())))
          for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert n["calls"] == 1 and sum(r.from_cache for r in res) == 3


def test_per_call_retry_observer_gets_wait_and_wins_over_attribute():
    k = {"i": 0}

    def transport(p):
        k["i"] += 1
        if k["i"] < 3:
            raise RuntimeError("ThrottlingException: rate exceeded")
        return {"output_text": "4", "usage": {}}
    a = BedrockResponsesAdapter(transport, sleep=lambda s: None)
    attr, per_call = [], []
    a.on_retry = lambda *x: attr.append(x)
    r = a.generate(_sample(), "gpt-5.6-luna@low", CandidateSettings(), on_retry=lambda *x: per_call.append(x))
    assert r.status == "ok" and attr == []
    assert [(x[0], x[2]) for x in per_call] == [(2, "throttled"), (3, "throttled")]
    # jittered exponential back-off: throttles start at 2 s (half fixed + half random), doubling each retry
    assert 1.0 <= per_call[0][4] <= 2.0 and 2.0 <= per_call[1][4] <= 4.0


def test_judge_cache_is_single_flight():
    n = {"calls": 0}

    def judge_t(prompt):
        n["calls"] += 1
        time.sleep(0.05)
        return {"semantic": 0.9, "instruction": 1.0, "reason": "equivalent"}
    j = Judge(judge_t)

    def ps():
        return PairScore(sample_id="s1", model="m", label=CompatLabel.COMPATIBLE_WITH_DRIFT,
                         d1_semantic=DimensionScore(score=0.5))
    ts = [threading.Thread(target=lambda: j.refine(_sample(), CandidateOutput(text="x"), ps())) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert n["calls"] == 1


# ----------------------------- Phase 1: settings / config ----------------------------- #
def test_settings_validation_and_snapshot(tmp_path, monkeypatch):
    c, rid = _client(tmp_path, monkeypatch)
    keep = (DEFAULT_SETTINGS.default_parallelism, DEFAULT_SETTINGS.fail_fast_error_share)
    try:
        assert c.put("/api/v1/settings", json={"parallelism": "x"}).status_code == 400
        assert c.put("/api/v1/settings", json={"parallelism": True}).status_code == 400
        assert c.put("/api/v1/settings", json={"parallelism": 40}).status_code == 200
        assert DEFAULT_SETTINGS.default_parallelism == 16
        assert c.put("/api/v1/settings", json={"parallelism": 3, "fail_fast_error_share": None}).status_code == 200
        g = c.get("/api/v1/settings").json()
        assert g["parallelism"] == 3 and g["concurrency"]["fail_fast_error_share"] is None
        body = {"candidates": [{"model": "gpt-5.6-luna", "reasoning_effort": "low"}]}
        c.post(f"/api/v1/runs/{rid}/plan", json=body)
        assert api_mod._RUNS[rid].config.parallelism == 3            # snapshot of Settings
        c.post(f"/api/v1/runs/{rid}/plan", json={**body, "parallelism": 8})
        assert api_mod._RUNS[rid].config.parallelism == 8            # API override
        for bad in (0, 17, "4"):
            r = c.post(f"/api/v1/runs/{rid}/plan", json={**body, "parallelism": bad})
            assert r.status_code == 400 and r.json()["detail"]["error"]["code"] == "invalid_parallelism"
    finally:
        DEFAULT_SETTINGS.default_parallelism, DEFAULT_SETTINGS.fail_fast_error_share = keep


def test_old_runs_load_as_sequential_and_concurrency_endpoint(tmp_path, monkeypatch):
    from modelshift.store import _config_from_dict
    assert _config_from_dict({"candidates": []}).parallelism == 1
    c, _ = _client(tmp_path, monkeypatch)
    snap = c.get("/api/v1/concurrency").json()
    assert set(snap) == {"server_inflight", "server_cap", "runs"} and snap["server_cap"] >= 1


# ----------------------------- Phase 2: engine ----------------------------- #
def _sleepy(d=0.1, track=None):
    lock, cur = threading.Lock(), {"n": 0}

    def fn(item, ctx):
        with lock:
            cur["n"] += 1
            if track is not None:
                track.append(cur["n"])
        time.sleep(d)
        with lock:
            cur["n"] -= 1
        return item * 10
    return fn


def test_pool_runs_at_most_n_and_scales():
    peaks, done, threads = [], [], set()
    t0 = time.time()
    status = CallPool(4).run(list(range(12)), _sleepy(0.1, peaks),
                             lambda it, v, ctx: (done.append(v), threads.add(threading.get_ident()))[0] and None)
    assert status == "done" and max(peaks) == 4 and sorted(done) == [i * 10 for i in range(12)]
    assert time.time() - t0 < 0.6                    # ~3 waves x 0.1s, not 1.2s sequential
    assert threads == {threading.get_ident()}        # on_done always on the caller's thread


def test_limit_one_runs_inline_in_order():
    order = []
    CallPool(1).run([3, 1, 2], lambda it, ctx: it, lambda it, v, ctx: order.append(v))
    assert order == [3, 1, 2]


def test_per_endpoint_limit_from_limiter():
    lim = AdaptiveLimiter(4)
    lim._eff["slow"] = 1
    lock, cur, peak = threading.Lock(), {"slow": 0}, {"slow": 0}

    def fn(item, ctx):
        with lock:
            cur[ctx.endpoint] = cur.get(ctx.endpoint, 0) + 1
            peak[ctx.endpoint] = max(peak.get(ctx.endpoint, 0), cur[ctx.endpoint])
        time.sleep(0.05)
        with lock:
            cur[ctx.endpoint] -= 1
    items = [("slow" if i % 2 else "fast", i) for i in range(10)]
    CallPool(4, limiter=lim).run(items, fn, lambda *a: None, endpoint=lambda it: it[0])
    assert peak["slow"] == 1 and peak["fast"] > 1


def test_limiter_halves_with_cooldown_and_recovers():
    t = {"now": 100.0}
    events = []
    lim = AdaptiveLimiter(8, on_change=lambda *e: events.append(e), clock=lambda: t["now"])
    lim.on_throttle("rt")
    lim.on_throttle("rt")              # inside cooldown: no second cut
    assert lim.effective("rt") == 4
    t["now"] += 3
    lim.on_throttle("rt")
    assert lim.effective("rt") == 2 and lim.min_effective == 2
    t["now"] += 10
    for _ in range(20):
        lim.on_success("rt")
    assert lim.effective("rt") == 3
    assert [e[3] for e in events] == ["throttle", "throttle", "recovered"]
    assert lim.slowed_total() == pytest.approx(13.0)
    off = AdaptiveLimiter(4, enabled=False)
    off.on_throttle("rt")
    assert off.effective("rt") == 4


def test_worker_throttle_slows_pool_and_events_reach_caller_thread():
    got, threads = [], set()

    def fn(item, ctx):
        if item == 0:
            ctx.throttle()
            ctx.emit("retry", item=item)
        time.sleep(0.02)

    def on_event(t, d):
        got.append(t)
        threads.add(threading.get_ident())
    pool = CallPool(4, limiter=AdaptiveLimiter(4), on_event=on_event)
    pool.run(list(range(8)), fn, lambda *a: None)
    assert "retry" in got and "concurrency" in got and threads == {threading.get_ident()}
    assert pool.limiter.effective("default") == 2


def test_server_cap_shared_by_two_pools():
    slots = ServerSlots(3)
    peak, lock, cur = [0], threading.Lock(), [0]

    def fn(item, ctx):
        with lock:
            cur[0] += 1
            peak[0] = max(peak[0], cur[0])
        time.sleep(0.05)
        with lock:
            cur[0] -= 1
    ts = [threading.Thread(target=lambda: CallPool(3, server=slots).run(list(range(6)), fn, lambda *a: None))
          for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert peak[0] <= 3 and slots.in_use == 0


def test_cancel_stops_submitting_quickly():
    flag = {"stop": False}
    seen = []

    def on_done(it, v, ctx):
        seen.append(it)
        if len(seen) == 2:
            flag["stop"] = True
    t0 = time.time()
    status = CallPool(2, cancel=lambda: flag["stop"]).run(list(range(50)), _sleepy(0.05), on_done)
    assert status == "cancelled" and len(seen) < 10 and time.time() - t0 < 1.0


def test_fail_fast_aborts_and_ignores_throttles():
    ff = FailFast(0.5)
    for _ in range(19):
        assert ff.record("403 AccessDenied: not authorized") is None
    assert ff.record("403 AccessDenied: not authorized").startswith("Stopped early: 20 of the last 20")
    ff2 = FailFast(0.5)
    assert all(ff2.record("429 throttled", throttle=True) is None for _ in range(30))
    assert FailFast(None).record("x") is None

    ff3 = FailFast(0.5)
    with pytest.raises(RunAbort, match="Stopped early"):
        CallPool(4).run(list(range(40)), lambda it, ctx: "boom",
                        lambda it, v, ctx: ff3.record("500 boom"))


def test_exec_stats_and_summary():
    st = ExecStats(4)
    st.throttled("bedrock-runtime", "luna@low", retried=True)
    st.throttled("bedrock-runtime", "terra@medium", retried=True)
    lim = AdaptiveLimiter(4)
    lim.min_effective, lim.slowed_seconds = 2, 95
    d = st.to_dict(lim)
    assert d["throttles"] == 2 and d["throttles_by_endpoint"] == {"bedrock-runtime": 2}
    d["wall_seconds"] = 372
    assert exec_summary(d) == "Sent 4 at a time · 2 throttles · slowed to 2 for 1m 35s · wall time 6m 12s"
    assert exec_summary({"parallelism": 1, "wall_seconds": 30}) == \
        "Sent one call at a time · no throttling · wall time 30s"
