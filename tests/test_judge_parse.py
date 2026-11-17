"""Judge reply parsing must never raise (a bad reply aborted a whole run)."""
from modelshift.judge import _converse_text, _parse_judge_json


def test_clean_json():
    v = _parse_judge_json('{"semantic": 0.9, "instruction": 1, "reason": "equivalent"}')
    assert v["semantic"] == 0.9 and v["reason"] == "equivalent"


def test_fenced_json_with_prose():
    v = _parse_judge_json('Here you go:\n```json\n{"semantic": 0.4, "reason": "missing_content"}\n```')
    assert v["semantic"] == 0.4


def test_brace_in_prose_before_object():
    v = _parse_judge_json('Note {not json}. {"semantic": 0.7, "reason": "format_change"}')
    assert v["reason"] == "format_change"


def test_truncated_object_salvaged():
    # The failing case: opening brace, cut off at max tokens, no closing brace.
    v = _parse_judge_json('{"semantic": 0.82, "instruction": 1.0, "factual_agree": true, '
                          '"reason": "added_verbosity", "rationale": "The candidate adds')
    assert v["semantic"] == 0.82 and v["instruction"] == 1.0
    assert v["reason"] == "added_verbosity" and v["factual_agree"] is True


def test_garbage_returns_empty_verdict():
    for text in ("", "{", "no json here", "{ broken"):
        v = _parse_judge_json(text)
        assert "semantic" not in v and "reason" not in v


def test_converse_text_skips_reasoning_block():
    resp = {"output": {"message": {"content": [{"reasoningContent": {}}, {"text": '{"semantic": 1}'}]}}}
    assert _converse_text(resp) == '{"semantic": 1}'
    assert _converse_text({}) == ""
