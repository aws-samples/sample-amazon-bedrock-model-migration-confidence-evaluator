"""Stage 3 tests: candidate adapters + deterministic scoring."""
from modelshift.adapters import (
    BedrockResponsesAdapter,
    LiteLLMResponsesAdapter,
    cache_key,
    render_responses_payload,
)
from modelshift.scoring import score_pair
from modelshift.schemas import (
    CandidateOutput,
    CandidateSettings,
    CompatLabel,
    Golden,
    GoldenKind,
    Message,
    NormalizedSample,
    ReasoningEffort,
    SampleRequest,
    SampleSource,
    Shape,
    ToolCall,
)


def _sample(prompt="What is 2+2?", golden_text="4", golden_kind=GoldenKind.TEXT,
            instructions=None, tool_calls=None):
    return NormalizedSample(
        id="s1",
        source=SampleSource(shape=Shape.CHAT_COMPLETIONS),
        legacy_model="gpt-4.1",
        request=SampleRequest(
            instructions=instructions,
            input_messages=[Message(role="user", content=prompt)],
        ),
        golden=Golden(kind=golden_kind, text=golden_text, tool_calls=tool_calls),
        evaluable=True,
    )


# ------------------------------- adapters ---------------------------------- #
def _fake_transport(text):
    def _t(payload):
        return {"output_text": text, "usage": {"input_tokens": 10, "output_tokens": 5}}
    return _t


def test_bedrock_adapter_mantle_uses_plain_model_id():
    seen = {}

    def _t(payload):
        seen["model"] = payload["model"]
        return {"output_text": "ok", "usage": {}}

    BedrockResponsesAdapter(transport=_t, endpoint="mantle").generate(_sample(), "gpt-5.4")
    assert seen["model"] == "openai.gpt-5.4"  # mantle id, no us. prefix


def test_bedrock_adapter_runtime_uses_profile_id():
    seen = {}

    def _t(payload):
        seen["model"] = payload["model"]
        return {"output_text": "ok", "usage": {}}

    BedrockResponsesAdapter(transport=_t, endpoint="runtime").generate(_sample(), "gpt-5.6-terra")
    assert seen["model"] == "us.openai.gpt-5.6-terra"


def test_render_payload_injects_reasoning_and_instructions():
    s = _sample(instructions="Be brief.")
    payload = render_responses_payload(s, "us.openai.gpt-5.6-luna",
                                       CandidateSettings(reasoning_effort=ReasoningEffort.MEDIUM))
    assert payload["reasoning"]["effort"] == "medium"
    assert payload["instructions"] == "Be brief."
    assert payload["input"][0]["content"][0]["type"] == "input_text"


def test_render_payload_never_emits_blank_block():
    s = _sample(prompt="   ")  # whitespace-only
    payload = render_responses_payload(s, "m", CandidateSettings())
    assert payload["input"] == []


def test_bedrock_adapter_maps_model_and_returns_output():
    a = BedrockResponsesAdapter(transport=_fake_transport("4"))
    r = a.generate(_sample(), "gpt-5.6-luna")
    assert r.status == "ok"
    assert r.output.text == "4"
    assert r.cost_estimate > 0
    assert r.model == "gpt-5.6-luna"


def test_litellm_adapter_uses_model_group():
    seen = {}

    def _t(payload):
        seen["model"] = payload["model"]
        return {"output_text": "ok", "usage": {}}

    a = LiteLLMResponsesAdapter(transport=_t)
    a.generate(_sample(), "gpt-5.6-terra")
    assert seen["model"] == "gpt-5.6-terra"  # litellm model_group alias


def test_bedrock_adapter_uses_inference_profile_id():
    seen = {}

    def _t(payload):
        seen["model"] = payload["model"]
        return {"output_text": "ok", "usage": {}}

    BedrockResponsesAdapter(transport=_t).generate(_sample(), "gpt-5.6-terra")
    assert seen["model"] == "us.openai.gpt-5.6-terra"


def test_adapter_caches_by_key():
    calls = {"n": 0}

    def _t(payload):
        calls["n"] += 1
        return {"output_text": "4", "usage": {}}

    a = BedrockResponsesAdapter(transport=_t)
    s = _sample()
    a.generate(s, "gpt-5.6-luna")
    a.generate(s, "gpt-5.6-luna")
    assert calls["n"] == 1  # second call served from cache


def test_adapter_strips_unsupported_param_proactively():
    attempts = {"n": 0}

    def _t(payload):
        attempts["n"] += 1
        # temperature must never reach the transport (proactively stripped)
        assert "temperature" not in payload
        return {"output_text": "4", "usage": {}}

    a = BedrockResponsesAdapter(transport=_t, sleep=lambda s: None)
    r = a.generate(_sample(), "gpt-5.6-luna", extra_params={"temperature": 0.7})
    assert r.status == "ok"
    assert "temperature" in r.settings.dropped_params
    assert attempts["n"] == 1  # no wasted failing call — dropped before sending


def test_adapter_backoff_then_error_on_persistent_throttle():
    def _t(payload):
        raise RuntimeError("ThrottlingException: Rate exceeded")

    a = BedrockResponsesAdapter(transport=_t, max_retries=2, sleep=lambda s: None)
    r = a.generate(_sample(), "gpt-5.6-luna")
    assert r.status == "error"
    assert "throttl" in r.error.lower()


def test_cache_key_changes_with_reasoning_effort():
    s = _sample()
    k1 = cache_key(s, "m", CandidateSettings(reasoning_effort=ReasoningEffort.MEDIUM))
    k2 = cache_key(s, "m", CandidateSettings(reasoning_effort=ReasoningEffort.LOW))
    assert k1 != k2


# ------------------------------- scoring ----------------------------------- #
def test_exact_match_short_circuits_to_compatible():
    s = _sample(golden_text="Tokyo")
    ps = score_pair(s, CandidateOutput(kind=GoldenKind.TEXT, text="Tokyo"), "gpt-5.6-luna")
    assert ps.label == CompatLabel.COMPATIBLE
    assert not ps.hard_break


def test_semantically_equivalent_scores_high():
    s = _sample(golden_text="The capital of France is Paris.")
    ps = score_pair(s, CandidateOutput(text="Paris is the capital of France."), "m")
    assert ps.label in (CompatLabel.COMPATIBLE, CompatLabel.COMPATIBLE_WITH_DRIFT)


def test_phone_divergence_is_hard_break_incompatible():
    s = _sample(golden_text="Call the nurse line at 1-800-555-0142.")
    ps = score_pair(s, CandidateOutput(text="Call the nurse line at 1-866-606-3700."), "m")
    assert ps.hard_break
    assert ps.label == CompatLabel.INCOMPATIBLE


def test_broken_json_is_hard_break_incompatible():
    s = _sample(golden_text='{"name": "Ada", "age": 36}', golden_kind=GoldenKind.JSON)
    ps = score_pair(s, CandidateOutput(text="Sure, here is Ada who is 36 years old."), "m")
    assert ps.hard_break
    assert ps.label == CompatLabel.INCOMPATIBLE


def test_json_key_match_not_broken():
    s = _sample(golden_text='{"name": "Ada", "age": 36}', golden_kind=GoldenKind.JSON)
    ps = score_pair(s, CandidateOutput(text='{"name": "Ada", "age": 36}'), "m")
    # exact match short-circuit -> compatible
    assert ps.label == CompatLabel.COMPATIBLE


def test_tool_call_name_mismatch_hard_break():
    s = _sample(golden_kind=GoldenKind.TOOL_CALL,
                tool_calls=[ToolCall(name="get_weather", arguments={"loc": "NYC"})])
    out = CandidateOutput(kind=GoldenKind.TOOL_CALL,
                          tool_calls=[ToolCall(name="lookup_forecast", arguments={"loc": "NYC"})])
    ps = score_pair(s, out, "m")
    assert ps.hard_break
    assert ps.label == CompatLabel.INCOMPATIBLE


def test_tool_call_match_compatible():
    s = _sample(golden_kind=GoldenKind.TOOL_CALL,
                tool_calls=[ToolCall(name="get_weather", arguments={"loc": "NYC"})])
    out = CandidateOutput(kind=GoldenKind.TOOL_CALL,
                          tool_calls=[ToolCall(name="get_weather", arguments={"loc": "NYC"})])
    ps = score_pair(s, out, "m")
    assert not ps.hard_break
    assert ps.label == CompatLabel.COMPATIBLE


def test_verbosity_drift_flagged():
    s = _sample(golden_text="Yes.")
    long = "Yes. " + ("This is a very long elaboration. " * 20)
    ps = score_pair(s, CandidateOutput(text=long), "m")
    assert ps.d4_verbosity.score < 0.99
