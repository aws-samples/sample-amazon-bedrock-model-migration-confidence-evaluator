"""Stage 2 tests: ingestion & normalization against the REAL LiteLLM fixtures."""
from pathlib import Path

from modelshift.ingest import (
    detect_container,
    ingest,
    normalize_row,
    RawRow,
)
from modelshift.schemas import Container, DropReason, GoldenKind, Shape

FIXTURES = Path(__file__).parent / "fixtures"


def _bytes(name):
    return (FIXTURES / name).read_bytes()


# --------------------------- container detection --------------------------- #
def test_detect_wrapped_data():
    assert detect_container(_bytes("litellm_spendlogs.json").decode()) == Container.WRAPPED_DATA
    assert detect_container(_bytes("litellm_wrapped.json").decode()) == Container.WRAPPED_DATA


def test_detect_jsonl():
    assert detect_container(_bytes("litellm_logs.jsonl").decode()) == Container.JSONL


# ------------------------------- SpendLogs -------------------------------- #
def test_spendlogs_ingest_lifts_proxy_request_and_response():
    res = ingest([(_bytes("litellm_spendlogs.json"), "upload:spendlogs")])
    # 3 rows: 2 evaluable, 1 redacted/empty (chatcmpl-sl-003-redacted has empty response)
    assert res.report.found == 3
    assert res.report.evaluable == 2
    s = next(x for x in res.samples if x.id == "chatcmpl-sl-001")
    assert s.source.shape == Shape.SPENDLOGS
    assert s.request.instructions and "CareConcierge" in s.request.instructions
    assert s.request.input_messages[0].content == "What's my primary care copay?"
    assert "$25" in s.golden.text
    assert s.legacy_model == "gpt-4.1"


def test_spendlogs_third_row_dropped():
    # chatcmpl-sl-003-redacted has no proxy_server_request and empty response.
    res = ingest([(_bytes("litellm_spendlogs.json"), "upload:spendlogs")])
    reasons = {d.eval_block_reason for d in res.dropped}
    assert reasons & {DropReason.NO_PROMPT, DropReason.EMPTY_RESPONSE}
    assert res.report.reconciles()


# ------------------------------ wrapped {data:[]} -------------------------- #
def test_wrapped_container_mixed_shapes():
    res = ingest([(_bytes("litellm_wrapped.json"), "upload:wrapped")])
    assert res.report.found == 2
    assert res.report.evaluable == 2  # one chat_completions (ping/pong), one responses (hello/hi)
    kinds = {s.source.shape for s in res.samples}
    assert Shape.CHAT_COMPLETIONS in kinds
    assert Shape.RESPONSES in kinds


# --------------------- Chat Completions & Responses JSONL ------------------ #
def test_jsonl_all_shapes_and_drops():
    res = ingest([(_bytes("litellm_logs.jsonl"), "upload:jsonl")])
    assert res.report.found == 12
    assert res.report.reconciles()
    reasons = {d.eval_block_reason for d in res.dropped}
    # multimodal image row -> unsupported_modality
    assert DropReason.UNSUPPORTED_MODALITY in reasons
    # REDACTED row -> redacted
    assert DropReason.REDACTED in reasons
    # stream chunk -> stream_fragment
    assert DropReason.STREAM_FRAGMENT in reasons
    # embedding -> non_eval_call_type
    assert DropReason.NON_EVAL_CALL_TYPE in reasons


def test_multiturn_context_preserved():
    res = ingest([(_bytes("litellm_logs.jsonl"), "upload:jsonl")])
    s = next(x for x in res.samples if x.id == "chatcmpl-2")
    # user / assistant / user preserved (3 turns), golden "42"
    assert len(s.request.input_messages) == 3
    assert s.golden.text == "42"


def test_tool_call_golden():
    # Tool-call goldens are dropped for now (skip tool calls): there is no text
    # response to compare a replayed candidate answer against.
    from modelshift.schemas import DropReason
    res = ingest([(_bytes("litellm_logs.jsonl"), "upload:jsonl")])
    s = next((x for x in res.samples if x.id == "chatcmpl-3"), None)
    assert s is None
    d = next((x for x in res.dropped if x.id == "chatcmpl-3"), None)
    assert d is not None
    assert d.eval_block_reason == DropReason.TOOL_CALLS_UNSUPPORTED


def test_json_golden_detected():
    res = ingest([(_bytes("litellm_logs.jsonl"), "upload:jsonl")])
    s = next(x for x in res.samples if x.id == "chatcmpl-6")
    assert s.golden.kind == GoldenKind.JSON


def test_responses_string_input():
    res = ingest([(_bytes("litellm_logs.jsonl"), "upload:jsonl")])
    s = next(x for x in res.samples if x.id == "resp-1")
    assert s.source.shape == Shape.RESPONSES
    assert "bedtime story" in s.request.input_messages[0].content
    assert "unicorn" in s.golden.text.lower()


def test_responses_structured_input_and_output_text():
    res = ingest([(_bytes("litellm_logs.jsonl"), "upload:jsonl")])
    s = next(x for x in res.samples if x.id == "resp-2")
    assert s.request.instructions == "Be terse."
    assert s.request.input_messages[0].content == "2+2?"
    assert s.golden.text == "4"


# ------------------------------ ChatCompletions→Responses ------------------ #
def test_tool_call_rows_dropped_as_unsupported():
    # A conversation with a tool-call round-trip is dropped (not replayed) for
    # now: replaying only the text turns would score the candidate on a
    # truncated prompt vs a golden produced with the tool result in context.
    from modelshift.schemas import DropReason
    row = {
        "id": "tc", "model": "gpt-4.1",
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": "{\"city\":\"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "18C sunny"},
            {"role": "user", "content": "Thanks, and in London?"},
        ],
        "response": {"choices": [{"message": {"content": "London is 15C."}}]},
    }
    sample, drop, shape = normalize_row(RawRow(data=row, object_uri="u", row_index=0))
    assert shape == Shape.CHAT_COMPLETIONS
    assert sample is None
    assert drop.eval_block_reason == DropReason.TOOL_CALLS_UNSUPPORTED


def test_plain_text_row_not_dropped_as_tool():
    # Regression guard: a normal conversation with no tool calls is unaffected.
    row = {
        "id": "plain", "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "response": {"choices": [{"message": {"content": "Hello"}}]},
    }
    sample, drop, shape = normalize_row(RawRow(data=row, object_uri="u", row_index=0))
    assert sample is not None
    assert sample.request.had_tool_calls is False


def test_system_message_becomes_instructions():
    row = {
        "id": "x", "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        "response": {"choices": [{"message": {"content": "Hello"}}]},
    }
    sample, drop, shape = normalize_row(RawRow(data=row, object_uri="u", row_index=0))
    assert shape == Shape.CHAT_COMPLETIONS
    assert sample.request.instructions == "Be brief."
    assert [m.role for m in sample.request.input_messages] == ["user"]


# ------------------------------ evaluability ------------------------------- #
def test_golden_without_prompt_not_evaluable():
    row = {
        "id": "np", "model": "gpt-4.1",
        "messages": [],
        "response": {"choices": [{"message": {"content": "answer with no question"}}]},
    }
    sample, drop, shape = normalize_row(RawRow(data=row, object_uri="u", row_index=0))
    assert sample is None
    assert drop.eval_block_reason == DropReason.NO_PROMPT


def test_reconciliation_across_all_fixtures():
    res = ingest([
        (_bytes("litellm_spendlogs.json"), "u1"),
        (_bytes("litellm_wrapped.json"), "u2"),
        (_bytes("litellm_logs.jsonl"), "u3"),
    ])
    assert res.report.reconciles()
    # detected models surfaced
    assert any(dm.rows > 0 for dm in res.report.detected_models)


# ------------------------- Prompt capture (flat) --------------------------- #
import json  # noqa: E402


def _capture(**overrides):
    row = json.loads(_bytes("prompt_capture_single.json"))
    row.update(overrides)
    return row


def test_detect_single_pretty_printed_object():
    # One pretty-printed object must be ONE row, not 15 line-split parse errors.
    assert detect_container(_bytes("prompt_capture_single.json").decode()) == Container.SINGLE_OBJECT


def test_prompt_capture_single_file_ingests_one_evaluable_sample():
    res = ingest([(_bytes("prompt_capture_single.json"), "cap1")])
    assert res.report.found == 1
    assert res.report.evaluable == 1
    assert res.report.reconciles()
    s = res.samples[0]
    assert s.id == "CAP-0001"
    assert s.source.shape == Shape.PROMPT_CAPTURE
    assert s.legacy_model == "gpt41-deployment:analyze"
    assert s.request.instructions == "You are a helpful AI assistant."
    assert [m.role for m in s.request.input_messages] == ["user"]
    assert s.request.input_messages[0].content.startswith("Compare the two extractions")
    assert s.golden.kind == GoldenKind.JSON  # response string is a JSON list
    assert s.usage.prompt_tokens == 120 and s.usage.completion_tokens == 30
    assert s.usage.total_tokens == 150


def test_prompt_capture_in_jsonl_and_array_containers():
    rows = [_capture(request_id="c1"), _capture(request_id="c2", response="plain text answer")]
    jsonl = "\n".join(json.dumps(r) for r in rows).encode()
    arr = json.dumps(rows).encode()
    for blob in (jsonl, arr):
        res = ingest([(blob, "multi")])
        assert res.report.evaluable == 2
        assert {s.golden.kind for s in res.samples} == {GoldenKind.JSON, GoldenKind.TEXT}


def test_prompt_capture_with_list_field_is_not_treated_as_wrapper():
    blob = json.dumps(_capture(tags=["a", "b"]), indent=2)
    assert detect_container(blob) == Container.SINGLE_OBJECT
    assert ingest([(blob.encode(), "x")]).report.evaluable == 1


def test_prompt_capture_non_2xx_dropped():
    s, drop, shape = normalize_row(RawRow(data=_capture(status_code=429, response="rate limited"),
                                          object_uri="x", row_index=0))
    assert shape == Shape.PROMPT_CAPTURE
    assert s is None and drop.eval_block_reason == DropReason.EMPTY_RESPONSE


def test_prompt_capture_empty_prompt_dropped():
    _, drop, _ = normalize_row(RawRow(data=_capture(user_prompt="   "), object_uri="x", row_index=0))
    assert drop.eval_block_reason == DropReason.NO_PROMPT


def test_prompt_capture_redacted_dropped():
    _, drop, _ = normalize_row(RawRow(data=_capture(response="[REDACTED]"), object_uri="x", row_index=0))
    assert drop.eval_block_reason == DropReason.REDACTED


# ------------------- Conversation array (Semantic Kernel) ------------------- #
def _sk(role, text):
    return {"Content": text, "Role": {"Label": role}}


def _conv_bytes(msgs):
    return json.dumps(msgs, indent=2).encode()


def test_conversation_array_is_one_row_not_one_per_message():
    msgs = [_sk("user", "Classify this appeal: code 99213, remark CO-50."),
            _sk("assistant", '{"decision": "Uphold"}')]
    res = ingest([(_conv_bytes(msgs), "upload:z!case_1.json")])
    assert res.report.found == 1 and res.report.evaluable == 1
    assert res.report.reconciles()
    s = res.samples[0]
    assert s.source.shape == Shape.CONVERSATION
    assert s.id == "upload:z!case_1.json#0"  # no request_id in the format -> file-based id
    assert [m.role for m in s.request.input_messages] == ["user"]
    assert s.request.instructions is None
    assert s.golden.kind == GoldenKind.JSON and s.golden.text == '{"decision": "Uphold"}'


def test_conversation_leading_assistant_becomes_instructions():
    msgs = [_sk("assistant", "You are an expert appeals processor."),
            _sk("user", "Process this appeal."),
            _sk("assistant", "Decision: Overturn")]
    s = ingest([(_conv_bytes(msgs), "u")]).samples[0]
    assert s.request.instructions == "You are an expert appeals processor."
    assert [m.role for m in s.request.input_messages] == ["user"]
    assert s.golden.kind == GoldenKind.TEXT and s.golden.text == "Decision: Overturn"


def test_conversation_multiturn_and_system_role_and_string_roles():
    msgs = [{"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]
    s = ingest([(_conv_bytes(msgs), "u")]).samples[0]
    assert s.request.instructions == "Be terse."
    assert [(m.role, m.content) for m in s.request.input_messages] == [
        ("user", "Q1"), ("assistant", "A1"), ("user", "Q2")]
    assert s.golden.text == "A2"


def test_conversation_semantic_kernel_items_text():
    msgs = [{"Role": {"Label": "user"}, "Items": [{"$type": "TextContent", "Text": "hello"}]},
            {"Role": {"Label": "assistant"}, "Items": [{"$type": "TextContent", "Text": "hi"}]}]
    s = ingest([(_conv_bytes(msgs), "u")]).samples[0]
    assert s.request.input_messages[0].content == "hello" and s.golden.text == "hi"


def test_conversation_not_ending_on_assistant_dropped():
    _, drop, shape = normalize_row(RawRow(data={"__conversation__": [_sk("user", "only a prompt")]},
                                          object_uri="x", row_index=0))
    assert shape == Shape.CONVERSATION and drop.eval_block_reason == DropReason.EMPTY_RESPONSE


def test_conversation_tool_turn_dropped_as_unsupported():
    msgs = [_sk("user", "look it up"), _sk("tool", "result"), _sk("assistant", "done")]
    res = ingest([(_conv_bytes(msgs), "u")])
    assert res.report.evaluable == 0
    assert res.dropped[0].eval_block_reason == DropReason.TOOL_CALLS_UNSUPPORTED


def test_array_of_log_rows_is_not_mistaken_for_a_conversation():
    rows = [{"request_id": "r1", "role": "user", "content": "x",
             "messages": [{"role": "user", "content": "x"}],
             "response": {"choices": [{"message": {"content": "y"}}]}}]
    res = ingest([(json.dumps(rows).encode(), "u")])
    assert res.samples[0].source.shape == Shape.CHAT_COMPLETIONS


def test_zip_of_conversation_files_consolidates():
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("case_1_step_a.json", _conv_bytes([_sk("user", "a"), _sk("assistant", "A")]))
        zf.writestr("case_1_step_b.json",
                    _conv_bytes([_sk("assistant", "persona"), _sk("user", "b"), _sk("assistant", "B")]))
    from modelshift.sources import load_zip_bytes
    objects, skipped = load_zip_bytes(buf.getvalue(), "upload:z")
    res = ingest(objects, skipped_objects=skipped)
    assert res.report.found == 2 and res.report.evaluable == 2 and res.report.reconciles()
    assert len({s.id for s in res.samples}) == 2
