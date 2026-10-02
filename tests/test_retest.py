"""Remediation re-test: real rescorer through the run's adapter/scoring, endpoints, guards."""
import time

from fastapi.testclient import TestClient

from modelshift import api as api_mod
from modelshift import store
from modelshift.orchestrator import CandidateConfig, Orchestrator, Run, RunConfig, RunStatus
from modelshift.schemas import (
    CallPath, Golden, GoldenKind, Message, NormalizedSample, RemediationChange,
    RemediationResult, SampleRequest, SampleSource, Shape,
)

GOLDEN = {"q1": "Paris is the capital of France.",
          "q2": "The invoice total is 42 USD, due on 3 March.",
          "q3": "Yes."}
FIX = "FIXIT: answer completely."


def _sample(sid):
    return NormalizedSample(
        id=sid, source=SampleSource(shape=Shape.CHAT_COMPLETIONS, object_uri="upload:t"),
        legacy_model="gpt-4.1",
        request=SampleRequest(instructions="You are helpful.", input_messages=[Message(role="user", content=sid)]),
        golden=Golden(kind=GoldenKind.TEXT, text=GOLDEN[sid]), evaluable=True,
    )


def _last(p):
    t = ""
    for it in p.get("input", []):
        for part in it.get("content", []):
            t = part.get("text", t)
    return t


def _transport(fail_on=None):
    """Answers q2 badly unless the fix instruction is in the system prompt."""
    seen = []

    def t(p):
        q = _last(p)
        seen.append({"q": q, "instructions": p.get("instructions", ""), "effort": (p.get("reasoning") or {})})
        if fail_on and q == fail_on and FIX in (p.get("instructions") or ""):
            raise RuntimeError("ThrottlingException: slow down")
        if q == "q2" and FIX not in (p.get("instructions") or ""):
            return {"output_text": "Sorry.", "usage": {}}
        return {"output_text": GOLDEN[q], "usage": {}}
    t.seen = seen
    return t


def _done_run(orch, run_id="rt1"):
    run = Run(run_id=run_id, name="retest")
    run.samples = [_sample("q1"), _sample("q2"), _sample("q3")]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], call_path=CallPath.BEDROCK)
    orch.run(run)
    assert run.status == RunStatus.DONE
    rem = next(x for x in run.remediations if "q2" in x.target.sample_ids)
    rem.change = RemediationChange(type="prompt_edit", after=FIX, placement="append")
    return run, rem


def test_rescorer_applies_change_through_the_run_adapter():
    t = _transport()
    orch = Orchestrator(candidate_transport=t, judge_transport=None)
    run, rem = _done_run(orch)
    n_before = len(t.seen)
    orch.retest_remediation(run, rem)
    res = rem.result
    assert res.status == "done" and rem.applied
    targets = len(rem.target.sample_ids)
    assert res.done == res.total == targets + res.regression_checked
    assert res.regression_checked == 2  # q1 + q3 were Compatible
    assert res.moved["regressions"] == 0
    assert res.confidence_after > res.confidence_before
    assert res.moved["to_compatible"] >= 1 and res.moved["errors"] == 0
    # the call really went through the adapter with the instruction appended
    retest_calls = t.seen[n_before:]
    assert retest_calls and all(c["instructions"].endswith(FIX) for c in retest_calls)
    assert all(c["instructions"].startswith("You are helpful.") for c in retest_calls)
    item = next(i for i in res.items if i["sample_id"] == "q2")
    assert item["before_label"] != "compatible" and item["after_label"] == "compatible"
    assert res.change.after == FIX and res.started_at and res.finished_at
    # the run's own scores are untouched (re-test is an experiment, not an overwrite)
    assert any(p.sample_id == "q2" and p.label.value != "compatible" for p in run.pair_scores)


def test_effort_change_sets_the_effort_not_the_prompt():
    t = _transport()
    orch = Orchestrator(candidate_transport=t, judge_transport=None)
    run, rem = _done_run(orch)
    rem.change = RemediationChange(type="reasoning_effort", before="medium", after="low")
    n_before = len(t.seen)
    orch.retest_remediation(run, rem)
    calls = t.seen[n_before:]
    assert calls and all(c["instructions"] == "You are helpful." for c in calls)
    assert all(c["effort"].get("effort") == "low" for c in calls)


def test_failed_calls_are_counted_not_fatal():
    t = _transport(fail_on="q2")
    orch = Orchestrator(candidate_transport=t, judge_transport=None)
    run, rem = _done_run(orch)
    orch.retest_remediation(run, rem)
    assert rem.result.status == "done"
    assert rem.result.moved["errors"] == 1
    assert any(i["status"] == "error" and "Throttling" in (i["error"] or "") for i in rem.result.items)


def test_cancel_stops_the_retest():
    orch = Orchestrator(candidate_transport=_transport(), judge_transport=None)
    run, rem = _done_run(orch)
    orch.retest_remediation(run, rem, cancel=lambda: True)
    assert rem.result.status == "cancelled" and not rem.applied and rem.result.done == 0


def test_plan_retest_estimates_without_calls():
    t = _transport()
    orch = Orchestrator(candidate_transport=t, judge_transport=None)
    run, rem = _done_run(orch)
    n = len(t.seen)
    plan = orch.plan_retest(run, rem)
    assert len(t.seen) == n  # no model calls
    assert plan["target_calls"] == len(rem.target.sample_ids)
    assert plan["regression_calls"] == plan["compatible_available"] == 2
    assert plan["calls"] == plan["target_calls"] + plan["regression_calls"]
    assert plan["est_cost"] >= 0 and plan["est_seconds"] > 0
    assert plan["change"]["after"] == FIX


def test_retest_api_launch_poll_and_guards():
    api_mod.configure_transports(candidate_transport=_transport(), judge_transport=None)
    c = TestClient(api_mod.app)
    run, rem = _done_run(api_mod._ORCH, run_id="rt_api")
    api_mod._RUNS[run.run_id] = run
    base = f"/api/v1/runs/{run.run_id}/remediations/{rem.remediation_id}"
    try:
        plan = c.post(base + "/retest/plan", json={"after": FIX}).json()
        assert plan["target_calls"] == len(rem.target.sample_ids)
        assert c.post(base + "/retest/plan", json={"regression_sample": 0}).json()["regression_calls"] == 0
        assert c.post(base + "/retest/plan", json={"regression_sample": -2}).status_code == 400
        assert c.post(base + "/retest", json={"type": "prompt_edit", "after": ""}).status_code == 400
        assert c.post(base + "/retest/cancel").status_code == 409  # nothing running

        # busy guard
        api_mod._RETEST_ACTIVE["active"] = "other/rem_x"
        assert c.post(base + "/retest").status_code == 409
        api_mod._RETEST_ACTIVE.pop("active", None)

        resp = c.post(base + "/retest", json={"after": FIX + " "})
        assert resp.status_code == 202
        for _ in range(100):
            got = c.get(base).json()
            if got["result"] and got["result"]["status"] != "running":
                break
            time.sleep(0.05)
        assert got["result"]["status"] == "done"
        assert got["applied"] is True
        assert got["change"]["after"] == FIX  # validated (trimmed) change was saved
        assert got["result"]["confidence_after"] > got["result"]["confidence_before"]
        assert "active" not in api_mod._RETEST_ACTIVE  # guard released

        # not-ready guard
        run.status = RunStatus.REPLAYING
        assert c.post(base + "/retest").status_code == 409
        run.status = RunStatus.DONE
        assert c.post(f"/api/v1/runs/{run.run_id}/remediations/nope/retest").status_code == 404
    finally:
        api_mod._RUNS.pop(run.run_id, None)
        api_mod._RETEST_ACTIVE.pop("active", None)


def test_store_marks_running_retest_interrupted():
    orch = Orchestrator(candidate_transport=_transport(), judge_transport=None)
    run, rem = _done_run(orch, run_id="rt_store")
    rem.result = RemediationResult(status="running", total=1)
    store.save_run(run)
    loaded = store.load_all_runs()["rt_store"]
    lrem = next(x for x in loaded.remediations if x.remediation_id == rem.remediation_id)
    assert lrem.result.status == "interrupted" and "restart" in lrem.result.error
    store.delete_run("rt_store")


# ------------------------------ regression check ------------------------------ #
def _breaking_transport():
    """The fix repairs q2 but breaks q1 (which used to be Compatible)."""
    def t(p):
        q, ins = _last(p), (p.get("instructions") or "")
        if FIX in ins and q == "q1":
            return {"output_text": "Sorry.", "usage": {}}
        if q == "q2" and FIX not in ins:
            return {"output_text": "Sorry.", "usage": {}}
        return {"output_text": GOLDEN[q], "usage": {}}
    return t


def test_regression_is_detected_on_compatible_evaluations():
    orch = Orchestrator(candidate_transport=_breaking_transport(), judge_transport=None)
    run, rem = _done_run(orch)
    orch.retest_remediation(run, rem)
    res = rem.result
    assert res.regression_checked == 2
    assert res.moved["to_compatible"] >= 1
    assert res.moved["regressions"] == 1
    roles = {i["sample_id"]: i["role"] for i in res.items}
    assert roles["q2"] == "target" and roles["q1"] == roles["q3"] == "regression_check"
    q1 = next(i for i in res.items if i["sample_id"] == "q1")
    assert q1["before_label"] == "compatible" and q1["after_label"] in ("partial", "incompatible")
    # the whole-candidate projection includes the regression
    assert res.confidence_run_before is not None and res.confidence_run_after is not None


def test_regression_sample_sizes():
    for n, expected in ((0, 0), (1, 1), (-1, 2), (50, 2)):
        orch = Orchestrator(candidate_transport=_transport(), judge_transport=None)
        run, rem = _done_run(orch, run_id=f"rt_n{n}")
        plan = orch.plan_retest(run, rem, regression_sample=n)
        orch.retest_remediation(run, rem, regression_sample=n)
        assert plan["regression_calls"] == rem.result.regression_checked == expected
        assert rem.result.regression_sample == n
        # the estimate and the actual re-test pick the same evaluations
        checked = sorted(i["sample_id"] for i in rem.result.items if i["role"] == "regression_check")
        assert len(checked) == expected


def test_regression_guard_ids_selection():
    from modelshift.remediate import regression_guard_ids
    from modelshift.schemas import CompatLabel, PairScore
    scores = [PairScore(sample_id=f"s{i}", model="m",
                        label=CompatLabel.COMPATIBLE if i % 2 == 0 else CompatLabel.PARTIAL)
              for i in range(10)]
    scores.append(PairScore(sample_id="s0", model="other", label=CompatLabel.COMPATIBLE))
    avail = [f"s{i}" for i in range(10)]
    allc = regression_guard_ids(scores, "m", ["s2"], avail, -1, "seed")
    assert allc == ["s0", "s4", "s6", "s8"]  # Compatible only, target excluded
    a = regression_guard_ids(scores, "m", ["s2"], avail, 2, "seed")
    assert a == regression_guard_ids(scores, "m", ["s2"], avail, 2, "seed")  # deterministic
    assert len(a) == 2 and set(a) <= set(allc)
    assert regression_guard_ids(scores, "m", [], avail, 0, "seed") == []
    assert regression_guard_ids(scores, "m", [], ["s0"], -1, "seed") == ["s0"]  # only available samples
