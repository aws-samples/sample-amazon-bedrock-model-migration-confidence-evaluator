"""Concrete remediation changes: before/after per type, apply semantics, edit API."""
import pytest
from fastapi.testclient import TestClient

from modelshift import api as api_mod
from modelshift.aggregate import recommend
from modelshift.orchestrator import Run
from modelshift.remediate import (
    PROMPT_TEMPLATES, apply_change, concretize, lower_effort, validate_change,
)
from modelshift.schemas import (
    CompatLabel, DimensionScore, Golden, GoldenKind, Message, NormalizedSample, PairScore,
    ReasoningEffort, RemediationChange, SampleRequest, SampleSource, Shape,
)


def _sample(sid, golden="Paris.", instructions=None, kind=GoldenKind.TEXT):
    return NormalizedSample(
        id=sid, source=SampleSource(shape=Shape.CHAT_COMPLETIONS, object_uri="upload:t"),
        legacy_model="gpt-4.1",
        request=SampleRequest(instructions=instructions, input_messages=[Message(role="user", content="q")]),
        golden=Golden(kind=kind, text=golden), evaluable=True,
    )


def _ps(sid, reason, label=CompatLabel.PARTIAL, hard=False):
    return PairScore(sample_id=sid, model="m", label=label, judge_reason=reason, hard_break=hard,
                     d1_semantic=DimensionScore(score=0.5))


def test_each_reason_gets_a_concrete_change():
    samples = [_sample(str(i)) for i in range(5)]
    scores = [_ps("0", "added_verbosity"), _ps("1", "missing_content"), _ps("2", "factual_divergence"),
              _ps("3", "instruction_violation"), _ps("4", "format_change")]
    recs = recommend("run1", "m", scores, samples=samples, reasoning_effort=ReasoningEffort.MEDIUM)
    recs = {r.target.reason: r for r in recs}
    eff = recs["added_verbosity"].change
    assert (eff.type, eff.before, eff.after, eff.placement) == ("reasoning_effort", "medium", "low", None)
    for reason in ("missing_content", "factual_divergence", "instruction_violation"):
        ch = recs[reason].change
        assert ch.type == "prompt_edit" and ch.placement == "append" and ch.before is None
        assert ch.after == PROMPT_TEMPLATES[reason]
    assert recs["format_change"].change.type == "format_schema"
    assert recs["format_change"].change.after  # text template when goldens are prose


def test_format_template_cites_golden_json_keys():
    samples = [_sample("1", '{"total": 42, "currency": "USD"}', kind=GoldenKind.JSON),
               _sample("2", '{"total": 7, "currency": "EUR", "note": "x"}', kind=GoldenKind.JSON)]
    rec = recommend("run1", "m", [_ps("1", "format_change"), _ps("2", "format_change")], samples=samples)[0]
    assert "valid JSON" in rec.change.after
    assert "total" in rec.change.after and "currency" in rec.change.after
    assert "note" in rec.change.after  # present in half of the goldens -> kept


def test_effort_at_minimal_falls_back_to_concise_instruction():
    rec = recommend("run1", "m", [_ps("1", "added_verbosity")], samples=[_sample("1")],
                    reasoning_effort=ReasoningEffort.MINIMAL)[0]
    assert rec.change.type == "prompt_edit"
    assert rec.change.after == PROMPT_TEMPLATES["added_verbosity"]
    assert lower_effort(ReasoningEffort.MINIMAL) is None


def test_concretize_is_idempotent_and_keeps_user_edits():
    rec = recommend("run1", "m", [_ps("1", "missing_content")])[0]  # no samples -> still abstract
    assert rec.change.after is None
    concretize(rec, [_sample("1")])
    assert rec.change.after == PROMPT_TEMPLATES["missing_content"]
    rec.change.after = "my edit"
    concretize(rec, [_sample("1")])
    assert rec.change.after == "my edit"


def test_apply_change_semantics():
    s = _sample("1", instructions="You are a claims assistant.")
    app_ch = RemediationChange(type="prompt_edit", after="Be complete.", placement="append")
    out, eff = apply_change(s, app_ch, ReasoningEffort.MEDIUM)
    assert out.request.instructions == "You are a claims assistant.\n\nBe complete."
    assert s.request.instructions == "You are a claims assistant."  # original untouched
    assert eff == ReasoningEffort.MEDIUM
    pre, _ = apply_change(s, RemediationChange(type="prompt_edit", after="Be complete.", placement="prepend"),
                          ReasoningEffort.MEDIUM)
    assert pre.request.instructions.startswith("Be complete.\n\n")
    none, _ = apply_change(_sample("2"), app_ch, ReasoningEffort.MEDIUM)
    assert none.request.instructions == "Be complete."  # no system prompt -> becomes the system prompt
    same, eff2 = apply_change(s, RemediationChange(type="reasoning_effort", before="medium", after="low"),
                              ReasoningEffort.MEDIUM)
    assert same is s and eff2 == ReasoningEffort.LOW


@pytest.mark.parametrize("ch, msg", [
    (RemediationChange(type="swap_candidate", after="x"), "change type"),
    (RemediationChange(type="prompt_edit", after="   "), "empty"),
    (RemediationChange(type="prompt_edit", after="x" * 5000), "longer than"),
    (RemediationChange(type="prompt_edit", after="ok", placement="middle"), "placement"),
    (RemediationChange(type="reasoning_effort", after="extreme"), "reasoning effort"),
])
def test_validate_change_rejects(ch, msg):
    with pytest.raises(ValueError, match=msg):
        validate_change(ch)


def test_api_backfills_and_edits_remediations():
    c = TestClient(api_mod.app)
    run = Run(run_id="rem_api_1", name="rem")
    run.samples = [_sample("1"), _sample("2")]
    # recorded the old way: abstract suggestions with no before/after
    run.remediations = recommend("rem_api_1", "m", [_ps("1", "missing_content"), _ps("2", "added_verbosity")])
    api_mod._RUNS[run.run_id] = run
    try:
        rems = c.get(f"/api/v1/runs/{run.run_id}/remediations").json()
        by = {x["target"]["reason"]: x for x in rems}
        assert by["missing_content"]["change"]["after"] == PROMPT_TEMPLATES["missing_content"]
        assert by["added_verbosity"]["change"]["after"] == "low"

        rid = by["missing_content"]["remediation_id"]
        url = f"/api/v1/runs/{run.run_id}/remediations/{rid}"
        ok = c.patch(url, json={"after": "  Always list every item.  ", "placement": "prepend"})
        assert ok.status_code == 200
        assert ok.json()["change"]["after"] == "Always list every item."
        assert ok.json()["change"]["placement"] == "prepend"
        sw = c.patch(url, json={"type": "reasoning_effort", "after": "high"}).json()["change"]
        assert (sw["before"], sw["after"], sw["placement"]) == ("medium", "high", None)
        assert c.patch(url, json={"type": "prompt_edit", "after": ""}).status_code == 400
        assert c.patch(f"/api/v1/runs/{run.run_id}/remediations/nope", json={"after": "x"}).status_code == 404
    finally:
        api_mod._RUNS.pop(run.run_id, None)
