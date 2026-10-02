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
