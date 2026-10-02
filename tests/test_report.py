from modelshift.orchestrator import Orchestrator, Run, RunConfig, CandidateConfig, new_run_id
from modelshift.report import build_run_report
from modelshift.schemas import (
    CallPath, Golden, GoldenKind, Message, NormalizedSample, SampleRequest, SampleSource, Shape,
)


def _sample(sid, prompt, golden):
    return NormalizedSample(
        id=sid, source=SampleSource(shape=Shape.CHAT_COMPLETIONS, object_uri="upload:t"),
        legacy_model="gpt-4.1",
        request=SampleRequest(input_messages=[Message(role="user", content=prompt)]),
        golden=Golden(kind=GoldenKind.TEXT, text=golden),
    )


def _completed_run():
    # echo transport -> exact match -> compatible/scored
    orch = Orchestrator(candidate_transport=lambda p: {"output_text": _last(p), "usage": {}},
                        judge_transport=None)
    run = Run(run_id=new_run_id(), name="Report Test")
    run.samples = [_sample("s1", "PONG", "PONG"), _sample("s2", "Paris", "Paris")]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], call_path=CallPath.BEDROCK)
    orch.run(run)
    return run


def _last(payload):
    last = ""
    for item in payload.get("input", []):
        for part in item.get("content", []):
            last = part.get("text", last)
    return last


def test_build_run_report_structure():
    run = _completed_run()
    rep = build_run_report(run)
    assert rep["run_id"] == run.run_id
    assert rep["name"] == "Report Test"
    assert rep["status"] == "done"
    # source summary
    assert rep["source"]["evaluable"] >= 1
    assert "upload:t" in rep["source"]["sources"]
    # candidates + best
    assert rep["candidates"], "expected at least one candidate verdict"
    c0 = rep["candidates"][0]
    assert c0["model"] == "gpt-5.6-luna"
    assert c0["endpoint"] in ("bedrock-runtime", "bedrock-mantle", "litellm")
    assert 0 <= c0["migration_confidence"] <= 100
    assert "dimension_averages" in c0 and "distribution" in c0
    assert rep["best"]["model"] == "gpt-5.6-luna"
    # collections present (possibly empty)
    assert isinstance(rep["remediations"], list)
    assert isinstance(rep["caveats"], list)
    assert isinstance(rep["top_failure_reasons_overall"], list)
    # appendix: per-candidate evaluations mirror the UI drill-down
    ev = rep["candidates"][0]["evaluations"]
    assert ev["total"] >= 1
    assert ev["limit"] == 50
    assert isinstance(ev["truncated"], bool)
    row = ev["rows"][0]
    for k in ("sample_id", "label", "prompt", "golden", "response", "dimensions"):
        assert k in row
    assert any(d["label"].startswith("D1") for d in row["dimensions"])


def test_evaluations_appendix_truncates_at_50():
    # 60 evaluable samples -> appendix caps at 50 and flags truncation.
    orch = Orchestrator(candidate_transport=lambda p: {"output_text": _last(p), "usage": {}},
                        judge_transport=None)
    run = Run(run_id=new_run_id(), name="Big")
    run.samples = [_sample(f"s{i}", f"q{i}", f"q{i}") for i in range(60)]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], call_path=CallPath.BEDROCK)
    orch.run(run)
    rep = build_run_report(run)
    ev = rep["candidates"][0]["evaluations"]
    assert ev["total"] == 60
    assert len(ev["rows"]) == 50
    assert ev["truncated"] is True


def test_render_run_pdf_valid():
    from modelshift.report import render_run_pdf
    rep = build_run_report(_completed_run())
    pdf = render_run_pdf(rep)
    assert isinstance(pdf, (bytes, bytearray))
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 1500  # non-trivial document


def test_render_run_pdf_failed_run():
    from modelshift.report import render_run_pdf
    rep = {
        "run_id": "r1", "name": "Boom", "description": "", "status": "failed",
        "error": "RuntimeError: kaboom", "source": {}, "candidates": [],
        "best": None, "remediations": [], "top_failure_reasons_overall": [], "caveats": [],
    }
    pdf = render_run_pdf(rep)
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 800


def test_render_run_pdf_light_theme():
    from modelshift.report import render_run_pdf
    pdf = render_run_pdf(build_run_report(_completed_run()), theme="light")
    assert pdf[:5] == b"%PDF-"


def test_render_run_pdf_long_content_paginates():
    # Multi-line request/response far taller than one page must split cleanly
    # (no LayoutError) and be truncated with a pointer to the app.
    from modelshift.report import render_run_pdf
    rep = build_run_report(_completed_run())
    row = rep["candidates"][0]["evaluations"]["rows"][0]
    long_text = "\n".join(f"line {i}: " + "lorem ipsum dolor sit amet " * 4 for i in range(400))
    row["messages"] = [{"role": "user", "content": long_text}]
    row["golden"] = long_text
    row["response"] = "x" * 5000  # one unbroken token
    pdf = render_run_pdf(rep)
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 4000


def test_report_includes_remediation_change_and_retest():
    from modelshift.report import render_run_pdf
    from modelshift.schemas import RemediationChange, RemediationResult
    run = Run(run_id=new_run_id(), name="Rem report")
    run.samples = [_sample("s1", "PONG", "PONG"), _sample("s2", "Paris", "London")]
    run.config = RunConfig(candidates=[CandidateConfig(model="gpt-5.6-luna")], call_path=CallPath.BEDROCK)
    Orchestrator(candidate_transport=lambda p: {"output_text": _last(p), "usage": {}}, judge_transport=None).run(run)
    assert run.remediations
    rem = run.remediations[0]
    rem.change = RemediationChange(type="prompt_edit", after="Answer completely.", placement="append")
    rem.result = RemediationResult(status="done", confidence_before=0, confidence_after=100,
                                   confidence_run_before=50, confidence_run_after=100,
                                   moved={"to_compatible": 1, "regressions": 0, "errors": 0},
                                   regression_checked=1, change=rem.change)
    rep = build_run_report(run)
    r0 = rep["remediations"][0]
    assert r0["after"] == "Answer completely." and r0["placement"] == "append"
    assert r0["retest"]["confidence_after"] == 100 and r0["retest"]["regression_checked"] == 1
    for theme in ("dark", "light"):
        assert render_run_pdf(rep, theme=theme)[:5] == b"%PDF-"
