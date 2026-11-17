"""Parallel calls, Phases 3 and 5: runs and re-tests go through CallPool; results match a
sequential run; throttles are counted and slow the run down; reports carry the summary."""
import threading
import time

from modelshift import api as api_mod
from modelshift.concurrency import concurrency_snapshot
from modelshift.report import build_run_report
from test_reasoning_efforts import _client

CANDS = [{"model": "gpt-5.6-luna", "reasoning_effort": "low"},
         {"model": "gpt-5.6-terra", "reasoning_effort": "medium"}]


def _rows(n):
    return [{"request_id": f"r{i}", "model": "gpt-4.1", "messages": {},
             "proxy_server_request": {"model": "gpt-4.1", "messages": [{"role": "user", "content": f"q{i}"}]},
             "response": {"choices": [{"message": {"role": "assistant", "content": f"answer {i}"}}]}}
            for i in range(n)]


def _client_n(tmp_path, monkeypatch, n=6):
    import json
    keep = api_mod._ORCH            # the transport the test configured; _client() resets it
    c, rid = _client(tmp_path, monkeypatch)
    api_mod._ORCH = keep
    up = c.post("/api/v1/uploads", files={"file": ("r.json", json.dumps({"data": _rows(n)}).encode(),
                                                   "application/json")}).json()
    rid = c.post("/api/v1/runs", json={"name": "par"}).json()["run_id"]
    c.post(f"/api/v1/runs/{rid}/ingest", json={"upload_id": up["upload_id"]})
    return c, rid


def _launch(c, rid, **kw):
    body = {"candidates": CANDS, "use_judge": False, **kw}
    assert c.post(f"/api/v1/runs/{rid}/launch", json=body).status_code == 202
    for _ in range(400):
        if c.get(f"/api/v1/runs/{rid}").json()["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.02)
    return api_mod._RUNS[rid]


def _echo(delay=0.0, track=None):
    lock, cur = threading.Lock(), [0]

    def transport(payload):
        with lock:
            cur[0] += 1
            if track is not None:
                track.append(cur[0])
        time.sleep(delay)
        with lock:
            cur[0] -= 1
        text = str(payload.get("input") or payload.get("messages") or "")
        return {"output_text": "answer " + text[-6:-3], "usage": {"input_tokens": 10, "output_tokens": 3}}
    return transport


def _sig(run):
    return ([(r.model, r.sample_id, r.replay_index, r.status) for r in run.results],
            [(p.model, p.sample_id, p.replay_index, p.label.value) for p in run.pair_scores],
            [(v.model, v.migration_confidence, v.verdict_band.value) for v in run.verdicts])


def test_parallel_run_matches_sequential(tmp_path, monkeypatch):
    api_mod.configure_transports(candidate_transport=_echo(), judge_transport=None)
    c1, r1 = _client_n(tmp_path / "a", monkeypatch)
    seq = _launch(c1, r1, parallelism=1, replays=2)
    api_mod.configure_transports(candidate_transport=_echo(0.01), judge_transport=None)
    c2, r2 = _client_n(tmp_path / "b", monkeypatch)
    par = _launch(c2, r2, parallelism=4, replays=2)
    assert seq.status.value == par.status.value == "done"
    assert _sig(seq) == _sig(par)                       # same results, order, verdicts
    assert seq.exec_stats["parallelism"] == 1 and par.exec_stats["parallelism"] == 4


def test_parallel_run_is_faster_and_respects_limit(tmp_path, monkeypatch):
    peaks = []
    api_mod.configure_transports(candidate_transport=_echo(0.1, peaks), judge_transport=None)
    c, rid = _client_n(tmp_path, monkeypatch, n=8)      # 16 calls
    t0 = time.time()
    run = _launch(c, rid, parallelism=4)
    assert run.status.value == "done" and max(peaks) == 4
    assert time.time() - t0 < 1.2                        # ~4 waves x 0.1s, not 1.6s
    assert run.exec_stats["max_inflight"] == 4


def test_replay_passes_overlap_so_a_slow_call_holds_only_its_own_slot(tmp_path, monkeypatch):
    seen, lock = {"slow": False}, threading.Lock()

    def transport(payload):
        text = str(payload.get("input") or "")
        with lock:   # exactly one slow call: request q0 on the luna candidate, first time only
            slow = (not seen["slow"]) and "q0" in text and "luna" in payload["model"]
            if slow:
                seen["slow"] = True
        time.sleep(0.6 if slow else 0.01)
        return {"output_text": "answer", "usage": {"input_tokens": 10, "output_tokens": 3}}
    api_mod.configure_transports(candidate_transport=transport, judge_transport=None)
    c, rid = _client_n(tmp_path, monkeypatch, n=5)
    run = _launch(c, rid, parallelism=4, replays=2)
    ev = run.events
    slow_done = next(i for i, e in enumerate(ev) if e.type == "progress" and e.data["replay"] == 0
                     and e.data["index"] == 0 and e.data["model"].startswith("gpt-5.6-luna"))
    first_pass2 = next(i for i, e in enumerate(ev) if e.type == "call_start" and e.data["replay"] == 1)
    assert first_pass2 < slow_done                 # pass 2 started while the slow pass-1 call was still running
    assert [e.data["replay"] for e in ev if e.type == "pass"] == [0, 1]
    assert run.status.value == "done" and len(run.results) == 20


def test_throttles_are_counted_and_slow_the_run(tmp_path, monkeypatch):
    from modelshift.adapters import CandidateAdapter
    monkeypatch.setattr(CandidateAdapter, "__init__", _no_sleep_init(CandidateAdapter.__init__))
    n = {"i": 0}
    lock = threading.Lock()

    def transport(p):
        with lock:
            n["i"] += 1
            k = n["i"]
        if k <= 3:
            raise RuntimeError("ThrottlingException: rate exceeded")
        return {"output_text": "x", "usage": {}}
    api_mod.configure_transports(candidate_transport=transport, judge_transport=None)
    c, rid = _client_n(tmp_path, monkeypatch, n=4)
    run = _launch(c, rid, parallelism=4)
    es = run.exec_stats
    assert run.status.value == "done" and es["throttles"] == 3 and es["throttle_retries"] == 3
    assert es["min_effective"] < 4
    assert any(e.type == "concurrency" and e.data["reason"] == "throttle" for e in run.events)
    retries = [e.data for e in run.events if e.type == "retry"]
    assert retries and all("endpoint" in d and "wait_s" in d for d in retries)
    summary = c.get(f"/api/v1/runs/{rid}/results").json()["exec_summary"]
    assert summary.startswith("Sent 4 at a time · 3 throttles")


def _no_sleep_init(orig):
    def init(self, transport, max_retries=3, sleep=None):
        orig(self, transport, max_retries=max_retries, sleep=lambda s: None)
    return init


def test_fail_fast_stops_a_failing_run(tmp_path, monkeypatch):
    from modelshift.adapters import CandidateAdapter
    monkeypatch.setattr(CandidateAdapter, "__init__", _no_sleep_init(CandidateAdapter.__init__))

    def transport(p):
        raise RuntimeError("403 AccessDeniedException: not authorized to invoke model")
    api_mod.configure_transports(candidate_transport=transport, judge_transport=None)
    c, rid = _client_n(tmp_path, monkeypatch, n=20)     # 40 calls
    run = _launch(c, rid, parallelism=4)
    assert run.status.value == "failed" and run.error.startswith("Stopped early")
    assert len(run.results) < 40


def test_cancel_stops_parallel_run(tmp_path, monkeypatch):
    api_mod.configure_transports(candidate_transport=_echo(0.05), judge_transport=None)
    c, rid = _client_n(tmp_path, monkeypatch, n=20)
    c.post(f"/api/v1/runs/{rid}/launch", json={"candidates": CANDS, "use_judge": False, "parallelism": 2})
    time.sleep(0.15)
    assert concurrency_snapshot()["runs"]                # live run is visible server-wide
    c.post(f"/api/v1/runs/{rid}/cancel")
    for _ in range(100):
        if c.get(f"/api/v1/runs/{rid}").json()["status"] == "cancelled":
            break
        time.sleep(0.02)
    run = api_mod._RUNS[rid]
    assert run.status.value == "cancelled" and len(run.results) < 40


def test_report_carries_execution_summary(tmp_path, monkeypatch):
    api_mod.configure_transports(candidate_transport=_echo(), judge_transport=None)
    c, rid = _client_n(tmp_path, monkeypatch, n=3)
    run = _launch(c, rid, parallelism=4)
    rep = build_run_report(run)
    assert rep["parallelism"] == 4 and rep["exec_summary"].startswith("Sent 4 at a time")
    assert "Measured with 4 parallel calls" in rep["latency_disclaimer"]
    import io
    from pypdf import PdfReader
    from modelshift.report_pdf import render_pdf
    text = " ".join(" ".join((p.extract_text() or "").split()) for p in PdfReader(io.BytesIO(render_pdf(rep))).pages)
    assert "Execution: Sent 4 at a time" in text


def test_plan_reports_parallelism_and_duration(tmp_path, monkeypatch):
    c, rid = _client_n(tmp_path, monkeypatch, n=8)
    p1 = c.post(f"/api/v1/runs/{rid}/plan", json={"candidates": CANDS, "use_judge": False, "parallelism": 1}).json()
    p4 = c.post(f"/api/v1/runs/{rid}/plan", json={"candidates": CANDS, "use_judge": False, "parallelism": 4}).json()
    assert p4["parallelism"] == 4 and p4["est_seconds"] * 4 == p1["est_seconds"]


def test_finished_old_runs_load_as_sequential():
    from modelshift.orchestrator import Run, RunConfig, RunStatus
    from modelshift.store import _run_from_dict, _run_to_dict
    r = Run(run_id="old", name="old")
    r.config = RunConfig(candidates=[], parallelism=6)   # stale, never-used value from old builds
    r.status = RunStatus.DONE
    d = _run_to_dict(r)
    d.pop("exec_stats")
    assert _run_from_dict(d).config.parallelism == 1
    r.exec_stats = {"parallelism": 6}
    assert _run_from_dict(_run_to_dict(r)).config.parallelism == 6
