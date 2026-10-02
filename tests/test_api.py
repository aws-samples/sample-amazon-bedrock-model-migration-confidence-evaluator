"""Stage 5 tests: orchestrator pipeline + FastAPI API (offline transports)."""
import time
from pathlib import Path

from fastapi.testclient import TestClient

from modelshift import api as api_mod
from modelshift.orchestrator import (
    CandidateConfig,
    Orchestrator,
    Run,
    RunConfig,
    RunStatus,
    default_candidates_for,
)
from modelshift.schemas import (
    CallPath,
    Golden,
    GoldenKind,
    Message,
    NormalizedSample,
    ReasoningEffort,
    SampleRequest,
    SampleSource,
    Shape,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _echo_transport(text_map=None):
    """Candidate transport that echoes the golden-ish answer (compatible) by default."""
    def _t(payload):
        # return the instructions-free echo of the last user input
        last = ""
        for item in payload.get("input", []):
            for part in item.get("content", []):
                last = part.get("text", last)
        out = (text_map or {}).get(last, last)
        return {"output_text": out, "usage": {"input_tokens": 10, "output_tokens": 8}}
    return _t


def _sample(sid, prompt, golden):
    return NormalizedSample(
        id=sid, source=SampleSource(shape=Shape.CHAT_COMPLETIONS), legacy_model="gpt-4.1",
        request=SampleRequest(input_messages=[Message(role="user", content=prompt)]),
        golden=Golden(kind=GoldenKind.TEXT, text=golden), evaluable=True,
    )


# ------------------------------ orchestrator ------------------------------- #
def test_orchestrator_mantle_without_transport_fails_clearly():
    run = Run(run_id="rm", name="t")
    run.samples = [_sample("1", "hi", "hi")]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.4")],
                           call_path=CallPath.BEDROCK, bedrock_endpoint="mantle")
    # candidate_transport set (runtime), but NO mantle transport
    orch = Orchestrator(candidate_transport=_echo_transport())
    orch.run(run)
    assert run.status == RunStatus.FAILED
    assert "mantle" in (run.error or "").lower()
    assert "bedrock-key" in (run.error or "").lower() or "api key" in (run.error or "").lower()


def test_run_persistence_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    from modelshift import store
    run = Run(run_id="run_persist1", name="persisted")
    run.samples = [_sample("1", "Q", "A")]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")])
    from modelshift.orchestrator import Orchestrator as _O
    _O(candidate_transport=_echo_transport({"Q": "A"})).run(run)
    assert run.status.value == "done"
    store.save_run(run)
    # reload from disk
    loaded = store.load_all_runs()
    assert "run_persist1" in loaded
    r2 = loaded["run_persist1"]
    assert r2.name == "persisted"
    assert r2.status.value == "done"
    assert len(r2.samples) == 1
    assert len(r2.verdicts) == 1
    assert r2.verdicts[0].model == "gpt-5.6-luna"
    assert r2.config.candidates[0].model == "gpt-5.6-luna"


def test_orchestrator_claude_uses_converse_transport():
    seen = {}

    def _converse(payload):
        seen["model"] = payload["model"]
        return {"output_text": "Paris", "usage": {"input_tokens": 5, "output_tokens": 2}}

    run = Run(run_id="rc", name="claude")
    run.samples = [_sample("1", "capital of France?", "Paris")]
    run.config = RunConfig(candidates=[CandidateConfig(model="claude-sonnet-4.5")])
    Orchestrator(candidate_transport=lambda p: {"output_text": "x", "usage": {}},
                 converse_transport=_converse).run(run)
    assert run.status.value == "done"
    assert seen["model"] == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert run.verdicts[0].model == "claude-sonnet-4.5"


def test_default_candidates_from_matrix():
    cands = default_candidates_for("gpt-4.1")
    assert [c.model for c in cands] == ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
    assert all(c.reasoning_effort == ReasoningEffort.MEDIUM for c in cands)


def test_orchestrator_dry_run_no_calls():
    calls = {"n": 0}

    def _t(p):
        calls["n"] += 1
        return {"output_text": "x", "usage": {}}

    run = Run(run_id="r1", name="t")
    run.samples = [_sample("1", "Q", "A")]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], dry_run=True)
    orch = Orchestrator(candidate_transport=_t)
    orch.run(run)
    assert run.status == RunStatus.DONE
    assert calls["n"] == 0  # dry-run never calls the model
    plan_event = next(e for e in run.events if e.type == "plan")
    assert plan_event.data["estimated_calls"] == 1


def test_orchestrator_full_pipeline_compatible():
    run = Run(run_id="r2", name="t")
    run.samples = [_sample("1", "capital of France?", "Paris"),
                   _sample("2", "2+2?", "4")]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], use_judge=True)
    # echo transport returns the prompt text, not the golden -> not exact match;
    # map to golden so it's compatible
    orch = Orchestrator(candidate_transport=_echo_transport({"capital of France?": "Paris", "2+2?": "4"}))
    orch.run(run)
    assert run.status == RunStatus.DONE
    assert len(run.verdicts) == 1
    v = run.verdicts[0]
    assert v.model == "gpt-5.6-luna"
    assert v.scored == 2
    assert v.migration_confidence == 100.0  # both exact matches -> compatible


def test_orchestrator_litellm_call_path():
    seen = {}

    def _t(payload):
        seen["model"] = payload["model"]
        return {"output_text": "4", "usage": {}}

    run = Run(run_id="r3", name="t")
    run.samples = [_sample("1", "2+2?", "4")]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-terra")], call_path=CallPath.LITELLM)
    Orchestrator(candidate_transport=_t).run(run)
    assert seen["model"] == "gpt-5.6-terra"  # litellm model_group


# --------------------------------- API ------------------------------------- #
def _client():
    api_mod._RUNS.clear()
    api_mod._UPLOADS.clear()
    api_mod.configure_transports(
        candidate_transport=_echo_transport(),  # echoes prompt -> exact-match with golden when golden==prompt
        judge_transport=None,
    )
    return TestClient(api_mod.app)


def test_api_models_endpoint():
    c = _client()
    r = c.get("/api/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["default_reasoning_effort"] == "medium"
    assert any(e["legacy_model"] == "gpt-4.1" for e in body["matrix"])


def test_api_end_to_end_upload_ingest_plan_launch():
    c = _client()
    # create
    run_id = c.post("/api/v1/runs", json={"name": "demo"}).json()["run_id"]
    # upload the real spendlogs fixture
    data = (FIXTURES / "litellm_spendlogs.json").read_bytes()
    up = c.post("/api/v1/uploads", files={"file": ("sl.json", data, "application/json")}).json()
    # ingest
    ing = c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]}).json()
    assert ing["ingestion"]["found"] == 3
    assert ing["ingestion"]["evaluable"] == 2
    # plan (dry-run cost preview)
    plan = c.post(f"/api/v1/runs/{run_id}/plan",
                  json={"candidates": [{"model": "gpt-5.6-luna"}]}).json()
    assert plan["estimated_calls"] == 2
    # launch (echo transport => candidate text == prompt, goldens differ, so scored)
    lr = c.post(f"/api/v1/runs/{run_id}/launch",
                json={"candidates": [{"model": "gpt-5.6-luna"}], "use_judge": True})
    assert lr.status_code == 202
    # wait for the background thread to finish
    for _ in range(50):
        if c.get(f"/api/v1/runs/{run_id}").json()["status"] == "done":
            break
        time.sleep(0.05)
    results = c.get(f"/api/v1/runs/{run_id}/results").json()
    assert results["status"] == "done"
    assert len(results["candidates"]) == 1
    assert results["candidates"][0]["scored"] == 2


def test_api_dry_run_launch_no_spend():
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "dry"}).json()["run_id"]
    data = (FIXTURES / "litellm_spendlogs.json").read_bytes()
    up = c.post("/api/v1/uploads", files={"file": ("sl.json", data, "application/json")}).json()
    c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})
    lr = c.post(f"/api/v1/runs/{run_id}/launch",
                json={"candidates": [{"model": "gpt-5.6-luna"}], "dry_run": True})
    assert lr.status_code == 202
    for _ in range(50):
        if c.get(f"/api/v1/runs/{run_id}").json()["status"] == "done":
            break
        time.sleep(0.05)
    # dry run => no verdicts produced
    assert c.get(f"/api/v1/runs/{run_id}/results").json()["candidates"] == []


def test_api_ingest_missing_source_400():
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "x"}).json()["run_id"]
    r = c.post(f"/api/v1/runs/{run_id}/ingest", json={})
    assert r.status_code == 400


def test_api_run_not_found_404():
    c = _client()
    assert c.get("/api/v1/runs/nope").status_code == 404


def test_caveat_threshold_not_tripped_by_15_rows():
    from modelshift.api import _caveats
    from modelshift.orchestrator import Run
    from modelshift.schemas import IngestionReport
    r15 = Run(run_id="c1", name="x")
    r15.ingestion = IngestionReport(found=15, evaluable=15)
    assert not any("Small sample" in c for c in _caveats(r15))  # 15 >= default threshold 10
    r5 = Run(run_id="c2", name="x")
    r5.ingestion = IngestionReport(found=5, evaluable=5)
    assert any("Small sample" in c for c in _caveats(r5))       # 5 < 10 still warns


def test_api_unmapped_legacy_model_falls_back_and_runs():
    # A sample logged under a NON-legacy model (e.g. gpt-5.6-terra) is unmapped
    # in the matrix; the run must still proceed with a fallback candidate.
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "unmapped"}).json()["run_id"]
    row = {"request_id": "u1", "model": "gpt-5.6-terra", "messages": {},
           "proxy_server_request": {"model": "gpt-5.6-terra",
                                    "messages": [{"role": "user", "content": "hi"}]},
           "response": {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}}
    import json as _j
    data = _j.dumps({"data": [row]}).encode()
    up = c.post("/api/v1/uploads", files={"file": ("u.json", data, "application/json")}).json()
    ing = c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]}).json()
    assert ing["ingestion"]["detected_models"][0]["model"] == "gpt-5.6-terra"
    # plan with no candidates specified -> fallback, not a 400
    plan = c.post(f"/api/v1/runs/{run_id}/plan", json={})
    assert plan.status_code == 200
    assert plan.json()["estimated_calls"] >= 1


def test_api_gpt54_forces_mantle_endpoint():
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "m"}).json()["run_id"]
    row = {"request_id": "u1", "model": "gpt-5.4", "messages": {},
           "proxy_server_request": {"model": "gpt-5.4",
                                    "messages": [{"role": "user", "content": "hi"}]},
           "response": {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}}
    import json as _j
    data = _j.dumps({"data": [row]}).encode()
    up = c.post("/api/v1/uploads", files={"file": ("u.json", data, "application/json")}).json()
    c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})
    # request runtime, but GPT-5.4 is mantle-only -> resolver forces mantle
    c.post(f"/api/v1/runs/{run_id}/plan",
           json={"candidates": [{"model": "gpt-5.4"}], "bedrock_endpoint": "runtime"})
    assert api_mod._RUNS[run_id].config.bedrock_endpoint == "mantle"


def test_settings_persist_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from modelshift import config
    config.DEFAULT_SETTINGS.default_reasoning_effort = config.ReasoningEffort.LOW
    config.DEFAULT_SETTINGS.small_sample_threshold = 7
    config.save_settings()
    assert (tmp_path / "settings.json").exists()
    # reset in memory, then reload from disk
    config.DEFAULT_SETTINGS.default_reasoning_effort = config.ReasoningEffort.MEDIUM
    config.DEFAULT_SETTINGS.small_sample_threshold = 10
    config.load_settings()
    assert config.DEFAULT_SETTINGS.default_reasoning_effort == config.ReasoningEffort.LOW
    assert config.DEFAULT_SETTINGS.small_sample_threshold == 7
    # restore
    config.DEFAULT_SETTINGS.default_reasoning_effort = config.ReasoningEffort.MEDIUM
    config.DEFAULT_SETTINGS.small_sample_threshold = 10


def test_api_update_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_SETTINGS_PATH", str(tmp_path / "s.json"))
    c = _client()
    r = c.put("/api/v1/settings", json={
        "default_reasoning_effort": "low",
        "parallelism": 4,
        "small_sample_threshold": 5,
        "price_table": {"gpt-5.6-luna": {"input_per_1k": 0.001, "output_per_1k": 0.002}},
    })
    assert r.status_code == 200
    got = r.json()
    assert got["default_reasoning_effort"] == "low"
    assert got["parallelism"] == 4
    assert got["small_sample_threshold"] == 5
    assert got["price_table"]["gpt-5.6-luna"]["input_per_1k"] == 0.001
    # restore defaults so other tests are unaffected
    c.put("/api/v1/settings", json={
        "default_reasoning_effort": "medium", "parallelism": 6, "small_sample_threshold": 10,
        "price_table": {"gpt-5.6-luna": {"input_per_1k": 0.00022, "output_per_1k": 0.00132}}})


def test_litellm_settings_editable_key_masked(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_SETTINGS_PATH", str(tmp_path / "s.json"))
    from modelshift import api as api_mod
    c = _client()
    r = c.put("/api/v1/settings", json={
        "litellm_base": "http://127.0.0.1:9999",
        "litellm_key": "sk-secret-xyz",
    })
    assert r.status_code == 200
    got = r.json()
    # base is returned; key is NOT echoed, only a set-indicator
    assert got["litellm_base"] == "http://127.0.0.1:9999"
    assert got["litellm_key_set"] is True
    assert "litellm_key" not in got
    # the live LiteLLM transport was rebuilt so changes take effect without restart
    assert api_mod._ORCH._litellm_transport is not None
    # judge_litellm_model is editable and rebuilds the LiteLLM judge transport
    r3 = c.put("/api/v1/settings", json={"judge_litellm_model": "claude-sonnet-4.5"})
    assert r3.json()["judge_litellm_model"] == "claude-sonnet-4.5"
    assert api_mod._ORCH._litellm_judge_transport is not None
    # omitting the key keeps the existing one (still set)
    r2 = c.put("/api/v1/settings", json={"litellm_base": "http://127.0.0.1:4000"})
    assert r2.json()["litellm_key_set"] is True
    # restore defaults
    c.put("/api/v1/settings", json={"litellm_base": "http://127.0.0.1:4000", "litellm_key": ""})


def test_ingest_lifts_original_latency():
    row = {"request_id": "L1", "model": "gpt-4.1", "messages": {},
           "request_duration_ms": 1624,
           "proxy_server_request": {"model": "gpt-4.1",
                                    "messages": [{"role": "user", "content": "hi"}]},
           "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}}
    from modelshift.ingest import normalize_row, RawRow
    sample, drop, shape = normalize_row(RawRow(data=row, object_uri="u", row_index=0))
    assert sample.original_latency_ms == 1624


def test_verdict_records_effective_endpoint():
    # A run records which endpoint each candidate actually used, on the verdict.
    from modelshift.orchestrator import _effective_endpoint
    from modelshift.adapters import LiteLLMResponsesAdapter, BedrockResponsesAdapter
    t = lambda p: {"output_text": "x", "usage": {}}   # noqa: E731
    assert _effective_endpoint(LiteLLMResponsesAdapter(transport=t)) == "litellm"
    assert _effective_endpoint(BedrockResponsesAdapter(transport=t, endpoint="runtime")) == "bedrock-runtime"
    assert _effective_endpoint(BedrockResponsesAdapter(transport=t, endpoint="mantle")) == "bedrock-mantle"


def test_judge_routes_through_litellm_when_call_path_litellm():
    # When call_path is LiteLLM, _judge must use the litellm_judge_transport;
    # otherwise it uses the default judge_transport.
    bedrock_judge = lambda p: {"semantic": 1.0, "instruction": 1.0, "reason": "equivalent"}  # noqa: E731
    lite_judge = lambda p: {"semantic": 0.9, "instruction": 1.0, "reason": "equivalent"}  # noqa: E731
    orch = Orchestrator(judge_transport=bedrock_judge, litellm_judge_transport=lite_judge)
    j_lite = orch._judge(call_path=CallPath.LITELLM)
    assert j_lite._transport is lite_judge
    j_bed = orch._judge(call_path=CallPath.BEDROCK)
    assert j_bed._transport is bedrock_judge
    # Falls back to the default judge if no LiteLLM judge transport is wired
    orch2 = Orchestrator(judge_transport=bedrock_judge)
    assert orch2._judge(call_path=CallPath.LITELLM)._transport is bedrock_judge


def test_report_pdf_endpoint_done_and_not_ready():
    c = _client()
    # Build a completed run through the orchestrator, register it in _RUNS.
    run = Run(run_id="rep1", name="Report / Run 1")
    run.samples = [_sample("1", "Paris", "Paris")]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], call_path=CallPath.BEDROCK)
    Orchestrator(candidate_transport=_echo_transport()).run(run)
    assert run.status == RunStatus.DONE
    api_mod._RUNS[run.run_id] = run

    resp = c.get(f"/api/v1/runs/{run.run_id}/report.pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment" in resp.headers["content-disposition"]
    assert ".pdf" in resp.headers["content-disposition"]
    assert resp.content[:5] == b"%PDF-"
    assert len(resp.content) > 1500

    light = c.get(f"/api/v1/runs/{run.run_id}/report.pdf?theme=light")
    assert light.status_code == 200
    assert light.content[:5] == b"%PDF-"
    assert "-print.pdf" in light.headers["content-disposition"]
    assert c.get(f"/api/v1/runs/{run.run_id}/report.pdf?theme=neon").status_code == 400

    # An in-progress run -> 409.
    pending = Run(run_id="rep2", name="pending")
    pending.status = RunStatus.REPLAYING
    api_mod._RUNS[pending.run_id] = pending
    r2 = c.get(f"/api/v1/runs/{pending.run_id}/report.pdf")
    assert r2.status_code == 409

    # Unknown run -> 404.
    assert c.get("/api/v1/runs/nope/report.pdf").status_code == 404


def test_run_level_crash_captures_error_detail():
    # Force a crash in the run loop (not a caught per-call error) and assert the
    # full traceback lands in run.error_detail with status FAILED.
    from modelshift.orchestrator import Orchestrator, Run, RunConfig, CandidateConfig, RunStatus, new_run_id
    from modelshift.schemas import CallPath
    orch = Orchestrator(candidate_transport=lambda p: {"output_text": "x", "usage": {}}, judge_transport=None)
    run = Run(run_id=new_run_id(), name="crash")
    run.samples = [_sample("s1", "hi", "hi")]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], call_path=CallPath.BEDROCK)
    # Poison _sample_subset to raise inside run() (a genuine run-level failure).
    import modelshift.orchestrator as orch_mod
    orig = orch_mod._sample_subset
    orch_mod._sample_subset = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("subset boom"))
    try:
        orch.run(run)
    finally:
        orch_mod._sample_subset = orig
    assert run.status == RunStatus.FAILED
    assert run.error and "subset boom" in run.error
    assert run.error_detail and "Traceback" in run.error_detail
    assert "subset boom" in run.error_detail


def test_run_failure_captures_traceback():
    # A transport that raises should fail the run and capture a full traceback,
    # surfaced as error_detail (+ per-call sample_errors) via /results.
    from fastapi.testclient import TestClient
    from modelshift import api as api_mod

    def _boom(payload):
        raise RuntimeError("kaboom from transport")
    api_mod.configure_transports(candidate_transport=_boom, judge_transport=None)
    c = TestClient(api_mod.app)
    rid = c.post("/api/v1/runs", json={"name": "boom"}).json()["run_id"]
    r = api_mod._RUNS[rid]
    r.samples = [_sample("s1", "hi", "hi")]
    r.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], call_path=CallPath.BEDROCK)
    # per-call errors are caught inside generate() -> status=error (not a run-level crash),
    # so assert the results endpoint surfaces the per-call error with its message.
    api_mod._ORCH.run(r)
    res = c.get(f"/api/v1/runs/{rid}/results").json()
    errs = res.get("sample_errors") or []
    assert errs, "expected at least one per-call error surfaced"
    assert any("kaboom" in e["error"] for e in errs)


def test_litellm_call_path_uses_litellm_transport():
    # When the run's call_path is LITELLM, the adapter must use the dedicated
    # litellm_transport (proxy), NOT the direct-Bedrock candidate transport.
    from modelshift.adapters import LiteLLMResponsesAdapter
    cand = lambda p: {"output_text": "direct-bedrock", "usage": {}}   # noqa: E731
    lite = lambda p: {"choices": [{"message": {"content": "via-proxy"}}], "usage": {}}  # noqa: E731
    orch = Orchestrator(candidate_transport=cand, litellm_transport=lite)
    cfg = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], call_path=CallPath.LITELLM)
    a = orch._adapter_for(cfg, "gpt-5.6-luna")
    assert isinstance(a, LiteLLMResponsesAdapter)
    assert a._transport is lite  # the proxy transport, not the Bedrock one


def test_runtime_model_uses_runtime_even_in_mantle_run():
    # A runtime-capable model (Luna) must NOT be routed to mantle even if the
    # run's endpoint was forced to mantle by a mantle-only sibling (GPT-5.4).
    from modelshift.orchestrator import Orchestrator, RunConfig, CandidateConfig
    from modelshift.adapters import BedrockResponsesAdapter
    orch = Orchestrator(candidate_transport=lambda p: {"output_text": "x", "usage": {}},
                        mantle_transport=lambda p: {"output_text": "m", "usage": {}})
    cfg = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], bedrock_endpoint="mantle")
    a = orch._adapter_for(cfg, "gpt-5.6-luna")
    assert isinstance(a, BedrockResponsesAdapter)
    assert a._endpoint == "runtime"  # Luna is runtime_supported → runtime, not mantle
    # GPT-5.4 (mantle-only) still goes to mantle
    a54 = orch._adapter_for(cfg, "gpt-5.4")
    assert a54._endpoint == "mantle"


def test_sampling_first_n_limits_subset():
    from modelshift.orchestrator import Orchestrator, Run, RunConfig, CandidateConfig
    run = Run(run_id="rl", name="limit")
    run.samples = [_sample(str(i), f"q{i}", f"a{i}") for i in range(20)]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")],
                           sampling_mode="first_n", sampling_n=5)
    Orchestrator(candidate_transport=lambda p: {"output_text": "x", "usage": {}}).run(run)
    assert run.verdicts[0].scored == 5


def test_ingest_tallies_teams_and_filter_targets_subset():
    from modelshift.ingest import ingest
    from modelshift.orchestrator import Orchestrator, Run, RunConfig, CandidateConfig
    rows = []
    for i in range(6):
        team = "member-services" if i < 4 else "claims"
        rows.append({"request_id": f"t{i}", "model": "gpt-4.1", "messages": {},
                     "metadata": {"user_api_key_team_alias": team},
                     "proxy_server_request": {"model": "gpt-4.1",
                                              "messages": [{"role": "user", "content": f"q{i}"}]},
                     "response": {"choices": [{"message": {"role": "assistant", "content": f"a{i}"}}]}})
    import json as _j
    res = ingest([(_j.dumps({"data": rows}).encode(), "u")])
    assert res.report.teams == {"claims": 2, "member-services": 4}
    # a run filtered to 'claims' only evaluates its 2 rows
    run = Run(run_id="rf", name="f")
    run.samples = res.samples
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], filter_teams=["claims"])
    Orchestrator(candidate_transport=lambda p: {"output_text": "x", "usage": {}}).run(run)
    assert run.verdicts[0].scored == 2


def test_api_delete_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    from modelshift import store
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "to delete"}).json()["run_id"]
    assert run_id in store.load_all_runs()
    d = c.delete(f"/api/v1/runs/{run_id}")
    assert d.status_code == 200 and d.json()["deleted"] == run_id
    assert run_id not in store.load_all_runs()      # gone from disk
    assert c.get(f"/api/v1/runs/{run_id}").status_code == 404  # gone from memory


def test_api_source_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "src"}).json()["run_id"]
    row = {"request_id": "s1", "model": "gpt-4.1", "messages": {}, "request_duration_ms": 900,
           "proxy_server_request": {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
           "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}}
    import json as _j
    data = _j.dumps({"data": [row]}).encode()
    up = c.post("/api/v1/uploads", files={"file": ("s.json", data, "application/json")}).json()
    c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})
    s = c.get(f"/api/v1/runs/{run_id}/source").json()
    assert s["evaluable"] == 1
    assert s["found"] == 1
    assert any(dm["model"] == "gpt-4.1" for dm in s["detected_models"])
    assert s["original_latency_ms"]["count"] == 1
    assert s["original_latency_ms"]["avg"] == 900.0
    assert s["sources"]  # source uri captured


def test_api_run_name_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    from modelshift import store
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "my eval run"}).json()["run_id"]
    # persisted on disk at create
    loaded = store.load_all_runs()
    assert loaded[run_id].name == "my eval run"
    # a name changed at launch time is applied + saved
    row = {"request_id": "n1", "model": "gpt-4.1", "messages": {},
           "proxy_server_request": {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
           "response": {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}}
    import json as _j
    data = _j.dumps({"data": [row]}).encode()
    up = c.post("/api/v1/uploads", files={"file": ("n.json", data, "application/json")}).json()
    c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})
    c.post(f"/api/v1/runs/{run_id}/launch",
           json={"candidates": [{"model": "gpt-5.6-luna"}], "name": "renamed at launch"})
    for _ in range(60):
        if c.get(f"/api/v1/runs/{run_id}").json()["status"] == "done":
            break
        time.sleep(0.05)
    assert store.load_all_runs()[run_id].name == "renamed at launch"
    # rename endpoint
    c.patch(f"/api/v1/runs/{run_id}", json={"name": "final name"})
    assert store.load_all_runs()[run_id].name == "final name"


def test_api_transcript_paginated(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "tx"}).json()["run_id"]
    rows = [{"request_id": f"t{i}", "model": "gpt-4.1", "messages": {},
             "proxy_server_request": {"model": "gpt-4.1",
                                      "messages": [{"role": "user", "content": f"q{i}"}]},
             "response": {"choices": [{"message": {"role": "assistant", "content": f"a{i}"}}]}}
            for i in range(30)]
    import json as _j
    data = _j.dumps({"data": rows}).encode()
    up = c.post("/api/v1/uploads", files={"file": ("t.json", data, "application/json")}).json()
    c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})
    c.post(f"/api/v1/runs/{run_id}/launch", json={"candidates": [{"model": "gpt-5.6-luna"}]})
    for _ in range(80):
        if c.get(f"/api/v1/runs/{run_id}").json()["status"] == "done":
            break
        time.sleep(0.05)
    page1 = c.get(f"/api/v1/runs/{run_id}/transcript?candidate=gpt-5.6-luna&limit=25&offset=0").json()
    assert page1["total"] == 30
    assert len(page1["items"]) == 25
    it = page1["items"][0]
    assert it["request"]["messages"][0]["content"].startswith("q")
    assert it["original_response"]["text"].startswith("a")
    assert it["candidate_model"] == "gpt-5.6-luna"
    page2 = c.get(f"/api/v1/runs/{run_id}/transcript?candidate=gpt-5.6-luna&limit=25&offset=25").json()
    assert len(page2["items"]) == 5


def test_api_sample_detail_join(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "detail"}).json()["run_id"]
    row = {"request_id": "d1", "model": "gpt-4.1", "messages": {}, "request_duration_ms": 900,
           "proxy_server_request": {"model": "gpt-4.1",
                                    "messages": [{"role": "user", "content": "capital of France?"}]},
           "response": {"choices": [{"message": {"role": "assistant", "content": "Paris"}}]}}
    import json as _j
    data = _j.dumps({"data": [row]}).encode()
    up = c.post("/api/v1/uploads", files={"file": ("d.json", data, "application/json")}).json()
    c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})
    c.post(f"/api/v1/runs/{run_id}/launch", json={"candidates": [{"model": "gpt-5.6-luna"}]})
    for _ in range(60):
        if c.get(f"/api/v1/runs/{run_id}").json()["status"] == "done":
            break
        time.sleep(0.05)
    det = c.get(f"/api/v1/runs/{run_id}/samples/d1?candidate=gpt-5.6-luna").json()
    assert det["original"]["model"] == "gpt-4.1"
    assert det["original"]["latency_ms"] == 900
    assert det["original"]["messages"][0]["content"] == "capital of France?"
    assert det["original"]["response"]["text"] == "Paris"
    assert det["candidate_result"]["model"] == "gpt-5.6-luna"
    assert det["candidate_result"]["latency_ms"] is not None
    assert det["score"] is not None


def test_api_version_and_build_id():
    c = _client()
    v = c.get("/api/v1/version").json()
    assert v["build_id"] and "+" in v["build_id"]
    assert v["source_hash"]
    assert v["started_at"]
    h = c.get("/api/v1/health").json()
    assert h["build_id"] == v["build_id"]


def test_api_stream_emits_end():
    c = _client()
    run_id = c.post("/api/v1/runs", json={"name": "s"}).json()["run_id"]
    data = (FIXTURES / "litellm_spendlogs.json").read_bytes()
    up = c.post("/api/v1/uploads", files={"file": ("sl.json", data, "application/json")}).json()
    c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})
    c.post(f"/api/v1/runs/{run_id}/launch", json={"candidates": [{"model": "gpt-5.6-luna"}]})
    for _ in range(50):
        if c.get(f"/api/v1/runs/{run_id}").json()["status"] == "done":
            break
        time.sleep(0.05)
    body = c.get(f"/api/v1/runs/{run_id}/stream").text
    assert "event: end" in body
    assert "event: progress" in body or "event: done" in body


# --------------------------------------------------------------------------- #
# Auth middleware (H1)
# --------------------------------------------------------------------------- #
import base64 as _b64  # noqa: E402


def _auth_client():
    from fastapi.testclient import TestClient
    from modelshift import api as api_mod
    api_mod.configure_transports(candidate_transport=_echo_transport(), judge_transport=None)
    return TestClient(api_mod.app)


def test_auth_none_is_open(monkeypatch):
    monkeypatch.delenv("MODELSHIFT_AUTH_MODE", raising=False)
    c = _auth_client()
    assert c.get("/api/v1/models").status_code == 200


def test_auth_shared_secret(monkeypatch):
    monkeypatch.setenv("MODELSHIFT_AUTH_MODE", "shared_secret")
    monkeypatch.setenv("MODELSHIFT_AUTH_USER", "admin")
    monkeypatch.setenv("MODELSHIFT_AUTH_PASSWORD", "s3cret")
    c = _auth_client()
    # health stays open
    assert c.get("/api/v1/health").status_code == 200
    # no creds -> 401
    assert c.get("/api/v1/models").status_code == 401
    # wrong creds -> 401
    bad = _b64.b64encode(b"admin:nope").decode()
    assert c.get("/api/v1/models", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    # right creds -> 200
    good = _b64.b64encode(b"admin:s3cret").decode()
    assert c.get("/api/v1/models", headers={"Authorization": f"Basic {good}"}).status_code == 200


def test_auth_shared_secret_unconfigured_fails_closed(monkeypatch):
    monkeypatch.setenv("MODELSHIFT_AUTH_MODE", "shared_secret")
    monkeypatch.delenv("MODELSHIFT_AUTH_USER", raising=False)
    monkeypatch.delenv("MODELSHIFT_AUTH_PASSWORD", raising=False)
    c = _auth_client()
    assert c.get("/api/v1/models").status_code == 401  # no creds configured -> deny


def test_auth_cognito_alb(monkeypatch):
    monkeypatch.setenv("MODELSHIFT_AUTH_MODE", "cognito_alb")
    c = _auth_client()
    assert c.get("/api/v1/health").status_code == 200          # open
    assert c.get("/api/v1/models").status_code == 401          # no ALB identity header
    ok = c.get("/api/v1/models", headers={"x-amzn-oidc-identity": "user@example.com"})
    assert ok.status_code == 200                               # ALB-injected header present


def test_auth_unknown_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("MODELSHIFT_AUTH_MODE", "bogus")
    c = _auth_client()
    assert c.get("/api/v1/models").status_code == 401


# --------------------------------------------------------------------------- #
# DoS guards (M1 upload cap/eviction, M3 SSE cap)
# --------------------------------------------------------------------------- #
def test_upload_size_cap(monkeypatch):
    from modelshift import api as api_mod
    monkeypatch.delenv("MODELSHIFT_AUTH_MODE", raising=False)
    monkeypatch.setattr(api_mod, "_MAX_UPLOAD_BYTES", 1024)  # 1 KiB cap for the test
    c = _auth_client()
    big = b"x" * 4096
    r = c.post("/api/v1/uploads", files={"file": ("big.json", big, "application/json")})
    assert r.status_code == 413
    small = b'{"data":[]}'
    assert c.post("/api/v1/uploads", files={"file": ("ok.json", small, "application/json")}).status_code == 201


def test_upload_eviction(monkeypatch):
    from modelshift import api as api_mod
    monkeypatch.delenv("MODELSHIFT_AUTH_MODE", raising=False)
    monkeypatch.setattr(api_mod, "_MAX_UPLOADS_RETAINED", 3)
    api_mod._UPLOADS.clear()
    c = _auth_client()
    for i in range(5):
        c.post("/api/v1/uploads", files={"file": (f"f{i}.json", b"{}", "application/json")})
    assert len(api_mod._UPLOADS) <= 3  # oldest evicted


def test_sse_stream_cap(monkeypatch):
    from modelshift import api as api_mod
    monkeypatch.delenv("MODELSHIFT_AUTH_MODE", raising=False)
    c = _auth_client()
    # create a run so /stream resolves
    rid = c.post("/api/v1/runs", json={"name": "s"}).json()["run_id"]
    # simulate being at the stream cap
    monkeypatch.setattr(api_mod, "_MAX_SSE_STREAMS", 0)
    r = c.get(f"/api/v1/runs/{rid}/stream")
    assert r.status_code == 429
