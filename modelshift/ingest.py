"""Ingestion & normalization (Doc 2).

Turns heterogeneous LiteLLM logs (file or S3) into a uniform set of
NormalizedSample records with full drop-reason accounting.

Container detection:      JSONL | JSON array | {data:[...]} / single-array-key wrapper | single object
Recognized row shapes:    SpendLogs (A) | wrapped SDK (B) | Chat Completions (C) | Responses (D)
                          | Prompt capture (flat system_prompt/user_prompt/response)
                          | Conversation (bare JSON array of messages, e.g. Semantic Kernel ChatHistory)
                          + Shape E (unknown / non-eval)
Canonical request:        system/developer -> instructions; turns -> input_messages (multi-turn kept)
Evaluability (E1-E4):     non-empty prompt AND usable golden AND text modality AND shape A-D
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .schemas import (
    Container,
    DetectedModel,
    DropReason,
    DroppedRow,
    Golden,
    GoldenKind,
    IngestionReport,
    Message,
    Modality,
    NormalizedSample,
    SampleRequest,
    SampleSource,
    Shape,
    ToolCall,
    Usage,
)

REDACTED_TOKENS = {"REDACTED", "[REDACTED]", "***REDACTED***"}
NON_EVAL_CALL_TYPES = {"embedding", "embeddings", "moderation", "image_generation", "transcription"}


# --------------------------------------------------------------------------- #
# Raw row + source loading
# --------------------------------------------------------------------------- #
@dataclass
class RawRow:
    data: Dict[str, Any]
    object_uri: str
    row_index: int


def _maybe_gunzip(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


def _is_prompt_capture(row: Any) -> bool:
    """Flat app-side capture: user_prompt string + response, no chat `messages`."""
    return (
        isinstance(row, dict)
        and isinstance(row.get("user_prompt"), str)
        and "response" in row
        and "messages" not in row
        and "proxy_server_request" not in row
    )


_ROW_MARKERS = ("messages", "proxy_server_request", "request_id", "response", "user_prompt")


def _msg_role(m: Dict[str, Any]) -> Optional[str]:
    """Role of a transcript message: 'role'/'Role', as a string or Semantic Kernel {'Label': ...}."""
    r = m.get("role", m.get("Role"))
    if isinstance(r, dict):
        r = r.get("Label", r.get("label"))
    return r.strip().lower() if isinstance(r, str) and r.strip() else None


def _msg_text(m: Dict[str, Any]) -> Tuple[Optional[str], Modality]:
    """Text of a transcript message: 'content'/'Content', else Semantic Kernel 'Items'[].Text."""
    content = m.get("content", m.get("Content"))
    if content is not None:
        return _text_of_content(content)
    parts: List[str] = []
    modality = Modality.TEXT
    for item in m.get("Items") or m.get("items") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("$type", "")).lower()
        if "image" in kind:
            modality = Modality.CONTAINS_IMAGE
        elif "audio" in kind:
            modality = Modality.CONTAINS_AUDIO
        text = item.get("Text", item.get("text"))
        if isinstance(text, str) and text:
            parts.append(text)
    return ("\n".join(parts) if parts else None), modality


def _is_message_list(obj: Any) -> bool:
    """A non-empty JSON array whose every element is a chat message (has a role and
    content/Items) and none looks like a log ROW -- i.e. one bare transcript."""
    if not isinstance(obj, list) or not obj:
        return False
    for m in obj:
        if not isinstance(m, dict) or _msg_role(m) is None:
            return False
        if not any(k in m for k in ("content", "Content", "Items", "items")):
            return False
        if any(k in m for k in _ROW_MARKERS):
            return False
    return True


def detect_container(text: str) -> Container:
    """Sniff container: {data:[...]}/single-array-key -> array -> single object -> JSONL (Doc 2 §2.2)."""
    stripped = text.lstrip("\ufeff \t\r\n")
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                # A capture record is a row in its own right, even if it carries a list field.
                if _is_prompt_capture(obj):
                    return Container.SINGLE_OBJECT
                array_keys = [k for k, v in obj.items() if isinstance(v, list)]
                if array_keys:
                    return Container.WRAPPED_DATA
                # The whole text parsed as ONE object (e.g. pretty-printed): it is a single row,
                # not JSONL. Splitting it by line would turn every line into a parse error.
                return Container.SINGLE_OBJECT
        except json.JSONDecodeError:
            pass
        # object-per-line JSONL also starts with '{'
        return Container.JSONL
    if stripped.startswith("["):
        try:
            json.loads(stripped)
            return Container.JSON_ARRAY
        except json.JSONDecodeError:
            return Container.JSONL
    return Container.JSONL


def _rows_from_wrapped(obj: Dict[str, Any]) -> List[Any]:
    if "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    for _k, v in obj.items():
        if isinstance(v, list):
            return v
    return []


def iter_rows(raw_bytes: bytes, object_uri: str) -> Iterator[RawRow]:
    """Yield RawRow items from a single log object, streaming JSONL where possible."""
    text = _maybe_gunzip(raw_bytes).decode("utf-8", errors="replace")
    container = detect_container(text)
    if container == Container.SINGLE_OBJECT:
        yield RawRow(data=json.loads(text.lstrip("\ufeff \t\r\n")), object_uri=object_uri, row_index=0)
        return
    if container in (Container.WRAPPED_DATA, Container.JSON_ARRAY):
        obj = json.loads(text.lstrip("\ufeff \t\r\n"))
        if _is_message_list(obj):
            # The whole array is ONE transcript (one LLM call), not one row per message.
            yield RawRow(data={"__conversation__": obj}, object_uri=object_uri, row_index=0)
            return
        rows = _rows_from_wrapped(obj) if isinstance(obj, dict) else obj
        for i, row in enumerate(rows):
            if isinstance(row, dict):
                yield RawRow(data=row, object_uri=object_uri, row_index=i)
            else:
                yield RawRow(data={"__parse_error__": True}, object_uri=object_uri, row_index=i)
        return
    # JSONL
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            yield RawRow(data=row, object_uri=object_uri, row_index=i)
        except json.JSONDecodeError:
            yield RawRow(data={"__parse_error__": True}, object_uri=object_uri, row_index=i)


# --------------------------------------------------------------------------- #
# Shape classification
# --------------------------------------------------------------------------- #
def classify(row: Dict[str, Any]) -> Shape:
    if row.get("__parse_error__"):
        return Shape.UNKNOWN
    if isinstance(row.get("__conversation__"), list):
        return Shape.CONVERSATION
    call_type = row.get("call_type")
    resp = row.get("response") or {}
    resp_obj = resp.get("object") if isinstance(resp, dict) else None

    # Non-eval call types / stream fragments / embeddings -> unknown (Shape E)
    if call_type in NON_EVAL_CALL_TYPES:
        return Shape.UNKNOWN
    if resp_obj in {"list", "chat.completion.chunk"}:
        return Shape.UNKNOWN

    # Prompt capture — flat system_prompt / user_prompt / response strings
    if _is_prompt_capture(row):
        return Shape.PROMPT_CAPTURE

    # Shape A — SpendLogs: proxy_server_request present, or spend+request_id with empty top messages
    top_messages = row.get("messages")
    if "proxy_server_request" in row:
        return Shape.SPENDLOGS
    if "spend" in row and "request_id" in row and (top_messages in (None, {}, [])):
        return Shape.SPENDLOGS

    # Shape D — Responses
    if call_type == "responses" or resp_obj == "response" or (isinstance(resp, dict) and "output" in resp):
        return Shape.RESPONSES

    # Shape B — wrapped SDK: rich metadata (usage_object / model_group / cost_breakdown)
    meta = row.get("metadata") or {}
    if isinstance(meta, dict) and (
        "usage_object" in meta or "cost_breakdown" in meta
    ) or "model_group" in row:
        if isinstance(top_messages, list):
            return Shape.WRAPPED

    # Shape C — Chat Completions
    if isinstance(top_messages, list) and isinstance(resp, dict) and "choices" in resp:
        return Shape.CHAT_COMPLETIONS
    if isinstance(top_messages, list):
        return Shape.CHAT_COMPLETIONS

    return Shape.UNKNOWN


# --------------------------------------------------------------------------- #
# Helpers for reading messages / responses
# --------------------------------------------------------------------------- #
def _text_of_content(content: Any) -> Tuple[Optional[str], Modality]:
    """Extract plain text + modality from a Chat/Responses content field."""
    if content is None:
        return None, Modality.TEXT
    if isinstance(content, str):
        return content, Modality.TEXT
    if isinstance(content, list):
        parts: List[str] = []
        modality = Modality.TEXT
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("text", "input_text", "output_text"):
                if part.get("text"):
                    parts.append(part["text"])
            elif ptype in ("image_url", "input_image", "image"):
                modality = Modality.CONTAINS_IMAGE
            elif ptype in ("input_audio", "audio"):
                modality = Modality.CONTAINS_AUDIO
        return ("\n".join(parts) if parts else None), modality
    return None, Modality.TEXT


def _canonical_request_from_chat(messages: List[Dict[str, Any]],
                                 tools: Optional[Any] = None,
                                 response_format: Optional[Any] = None) -> SampleRequest:
    """Chat Completions messages -> canonical request (Doc 2 §4)."""
    instructions_parts: List[str] = []
    input_messages: List[Message] = []
    modality = Modality.TEXT
    had_tool_calls = False
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        # Detect tool/function-call turns. For now these rows are dropped as
        # non-evaluable (see evaluability()): a `tool`/`function` result message
        # references a preceding assistant `tool_calls` stub, and replaying only
        # the text turns would score the candidate on a truncated prompt vs a
        # golden that had the tool result in context — not apples-to-apples.
        if role in ("tool", "function"):
            had_tool_calls = True
            continue
        text, mod = _text_of_content(m.get("content"))
        if mod != Modality.TEXT:
            modality = mod
        if role == "assistant" and m.get("tool_calls"):
            had_tool_calls = True
            if not text:
                continue
        if role in ("system", "developer"):
            if text:
                instructions_parts.append(text)
        else:
            if text:
                input_messages.append(Message(role=role, content=text))
    return SampleRequest(
        instructions="\n".join(instructions_parts) if instructions_parts else None,
        input_messages=input_messages,
        tools=tools if isinstance(tools, list) else None,
        response_format=response_format if isinstance(response_format, dict) else None,
        modality=modality,
        had_tool_calls=had_tool_calls,
    )


def _golden_from_chat_response(resp: Dict[str, Any]) -> Golden:
    choices = resp.get("choices") or []
    if not choices:
        return Golden(kind=GoldenKind.EMPTY, raw=resp)
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    tool_calls = msg.get("tool_calls")
    refusal = msg.get("refusal")
    if content:
        kind = GoldenKind.JSON if _looks_json(content) else GoldenKind.TEXT
        return Golden(kind=kind, text=content, raw=resp)
    if tool_calls:
        tcs = [
            ToolCall(name=(tc.get("function") or {}).get("name", ""),
                     arguments=(tc.get("function") or {}).get("arguments"))
            for tc in tool_calls if isinstance(tc, dict)
        ]
        return Golden(kind=GoldenKind.TOOL_CALL, tool_calls=tcs, raw=resp)
    if refusal:
        return Golden(kind=GoldenKind.REFUSAL, text=refusal, raw=resp)
    return Golden(kind=GoldenKind.EMPTY, raw=resp)


def _golden_from_responses(resp: Dict[str, Any]) -> Golden:
    if resp.get("output_text"):
        txt = resp["output_text"]
        return Golden(kind=GoldenKind.JSON if _looks_json(txt) else GoldenKind.TEXT, text=txt, raw=resp)
    outputs = resp.get("output") or []
    texts: List[str] = []
    tool_calls: List[ToolCall] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            t, _ = _text_of_content(item.get("content"))
            if t:
                texts.append(t)
        elif itype == "function_call":
            tool_calls.append(ToolCall(name=item.get("name", ""), arguments=item.get("arguments")))
    if texts:
        joined = "\n".join(texts)
        return Golden(kind=GoldenKind.JSON if _looks_json(joined) else GoldenKind.TEXT, text=joined, raw=resp)
    if tool_calls:
        return Golden(kind=GoldenKind.TOOL_CALL, tool_calls=tool_calls, raw=resp)
    return Golden(kind=GoldenKind.EMPTY, raw=resp)


def _canonical_request_from_responses(row: Dict[str, Any]) -> SampleRequest:
    """Responses-shaped request -> canonical request."""
    instructions_parts: List[str] = []
    input_messages: List[Message] = []
    modality = Modality.TEXT

    if row.get("instructions"):
        instructions_parts.append(row["instructions"])

    messages = row.get("messages")
    if isinstance(messages, str):
        input_messages.append(Message(role="user", content=messages))
    elif isinstance(messages, dict):
        if messages.get("instructions"):
            instructions_parts.append(messages["instructions"])
        for item in messages.get("input", []) or []:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "user")
            text, mod = _text_of_content(item.get("content"))
            if mod != Modality.TEXT:
                modality = mod
            if role in ("system", "developer"):
                if text:
                    instructions_parts.append(text)
            elif text:
                input_messages.append(Message(role=role, content=text))
    return SampleRequest(
        instructions="\n".join(instructions_parts) if instructions_parts else None,
        input_messages=input_messages,
        modality=modality,
    )


def _looks_json(text: str) -> bool:
    s = (text or "").strip()
    if not (s.startswith("{") and s.endswith("}")) and not (s.startswith("[") and s.endswith("]")):
        return False
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def _is_redacted(text: Optional[str]) -> bool:
    return bool(text) and text.strip() in REDACTED_TOKENS


def _original_latency_of(row: Dict[str, Any]) -> Optional[int]:
    """Original request latency from the legacy log, if present (wrapped SDK shape)."""
    dur = row.get("request_duration_ms")
    if isinstance(dur, (int, float)):
        return int(dur)
    # derive from startTime/endTime ISO timestamps if available
    start, end = row.get("startTime"), row.get("endTime")
    if isinstance(start, str) and isinstance(end, str):
        try:
            from datetime import datetime
            s = datetime.fromisoformat(start.replace("Z", "+00:00"))
            e = datetime.fromisoformat(end.replace("Z", "+00:00"))
            return int((e - s).total_seconds() * 1000)
        except ValueError:
            return None
    return None


def _usage_of(row: Dict[str, Any]) -> Usage:
    meta = row.get("metadata") or {}
    uobj = meta.get("usage_object") if isinstance(meta, dict) else None
    reasoning = None
    if isinstance(uobj, dict):
        ctd = uobj.get("completion_tokens_details") or {}
        reasoning = ctd.get("reasoning_tokens")
    # Capture-style nested usage: {"usage": {"input_tokens", "output_tokens", ...}}
    nested = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    prompt_tokens = row.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = nested.get("prompt_tokens", nested.get("input_tokens"))
    completion_tokens = row.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = nested.get("completion_tokens", nested.get("output_tokens"))
    total_tokens = row.get("total_tokens")
    if total_tokens is None:
        total_tokens = nested.get("total_tokens")
        if total_tokens is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning,
        spend=row.get("spend"),
    )


def _tags_of(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = row.get("metadata") or {}
    tags: Dict[str, Any] = {}
    if isinstance(meta, dict):
        team = meta.get("user_api_key_team_alias")
        if team:
            tags["team"] = team
        user = meta.get("user_api_key_alias") or meta.get("user_api_key_user_id")
        if user:
            tags["user"] = user
    # top-level end-user id (SpendLogs / wrapped shapes)
    if not tags.get("user") and row.get("user") and row.get("user") != "default_user_id":
        tags["user"] = row["user"]
    return tags


# --------------------------------------------------------------------------- #
# Per-shape adapters
# --------------------------------------------------------------------------- #
def adapt_spendlogs(row: Dict[str, Any]) -> Tuple[SampleRequest, Golden]:
    psr = row.get("proxy_server_request") or {}
    messages = psr.get("messages") or []
    req = _canonical_request_from_chat(messages,
                                       tools=psr.get("tools"),
                                       response_format=psr.get("response_format"))
    golden = _golden_from_chat_response(row.get("response") or {})
    return req, golden


def adapt_wrapped(row: Dict[str, Any]) -> Tuple[SampleRequest, Golden]:
    messages = row.get("messages") or []
    req = _canonical_request_from_chat(messages)
    resp = row.get("response") or {}
    if resp.get("object") == "response" or "output" in resp:
        golden = _golden_from_responses(resp)
    else:
        golden = _golden_from_chat_response(resp)
    return req, golden


def adapt_chat_completions(row: Dict[str, Any]) -> Tuple[SampleRequest, Golden]:
    messages = row.get("messages") or []
    req = _canonical_request_from_chat(messages,
                                       tools=row.get("tools"),
                                       response_format=row.get("response_format"))
    golden = _golden_from_chat_response(row.get("response") or {})
    return req, golden


def adapt_responses(row: Dict[str, Any]) -> Tuple[SampleRequest, Golden]:
    req = _canonical_request_from_responses(row)
    golden = _golden_from_responses(row.get("response") or {})
    return req, golden


def adapt_prompt_capture(row: Dict[str, Any]) -> Tuple[SampleRequest, Golden]:
    """Flat capture record -> canonical request + golden.

    system_prompt -> instructions, user_prompt -> single user turn,
    response (string, or a Chat Completions object) -> golden.
    A non-2xx status_code means the response is an error body, not a baseline answer.
    """
    system = row.get("system_prompt")
    user = row.get("user_prompt")
    req = SampleRequest(
        instructions=system if isinstance(system, str) and system.strip() else None,
        input_messages=[Message(role="user", content=user)] if isinstance(user, str) and user.strip() else [],
    )
    status = row.get("status_code")
    resp = row.get("response")
    if isinstance(status, int) and not 200 <= status < 300:
        return req, Golden(kind=GoldenKind.EMPTY, raw={"status_code": status, "response": resp})
    if isinstance(resp, dict):
        golden = _golden_from_chat_response(resp) if "choices" in resp else _golden_from_responses(resp)
    elif isinstance(resp, str) and resp.strip():
        golden = Golden(kind=GoldenKind.JSON if _looks_json(resp) else GoldenKind.TEXT,
                        text=resp, raw={"response": resp})
    else:
        golden = Golden(kind=GoldenKind.EMPTY, raw={"response": resp})
    return req, golden


def adapt_conversation(row: Dict[str, Any]) -> Tuple[SampleRequest, Golden]:
    """Bare transcript -> canonical request + golden.

    - The final message, if it is an assistant turn, is the golden; everything before it
      is the replayed request. A transcript that does not end on an assistant turn has no
      baseline answer (EMPTY golden -> dropped as empty_response).
    - system/developer messages, and any assistant messages BEFORE the first user turn,
      become instructions: an assistant turn can't precede the first prompt, so those are
      persona/system text the app sent under the wrong role.
    - Later assistant turns stay in context (multi-turn); tool/function turns mark the row
      as a tool round-trip (unsupported for replay, same rule as the other shapes).
    """
    msgs = [m for m in row.get("__conversation__") or [] if isinstance(m, dict)]
    golden = Golden(kind=GoldenKind.EMPTY, raw={"messages": len(msgs)})
    if msgs and _msg_role(msgs[-1]) == "assistant":
        text, _ = _msg_text(msgs[-1])
        if text and text.strip():
            golden = Golden(kind=GoldenKind.JSON if _looks_json(text) else GoldenKind.TEXT,
                            text=text, raw={"messages": len(msgs)})
        msgs = msgs[:-1]

    instructions: List[str] = []
    turns: List[Message] = []
    modality = Modality.TEXT
    had_tools = False
    seen_user = False
    for m in msgs:
        role = _msg_role(m) or "user"
        text, mod = _msg_text(m)
        if mod != Modality.TEXT:
            modality = mod
        if role in ("tool", "function"):
            had_tools = True
            continue
        if role in ("system", "developer") or (role == "assistant" and not seen_user):
            if text and text.strip():
                instructions.append(text)
            continue
        if role == "user":
            seen_user = True
        if text and text.strip():
            turns.append(Message(role=role if role in ("user", "assistant") else "user", content=text))
    req = SampleRequest(
        instructions="\n".join(instructions) if instructions else None,
        input_messages=turns,
        modality=modality,
        had_tool_calls=had_tools,
    )
    return req, golden


ADAPTERS = {
    Shape.CONVERSATION: adapt_conversation,
    Shape.SPENDLOGS: adapt_spendlogs,
    Shape.WRAPPED: adapt_wrapped,
    Shape.CHAT_COMPLETIONS: adapt_chat_completions,
    Shape.RESPONSES: adapt_responses,
    Shape.PROMPT_CAPTURE: adapt_prompt_capture,
}


# --------------------------------------------------------------------------- #
# Evaluability (E1-E4, Doc 2 §6.1) + drop reason
# --------------------------------------------------------------------------- #
def _prompt_text(req: SampleRequest) -> str:
    parts = [m.content for m in req.input_messages if m.content]
    return "\n".join(parts).strip()


def evaluability(req: SampleRequest, golden: Golden) -> Optional[DropReason]:
    """Return None if evaluable, else the DropReason (E1-E4)."""
    prompt = _prompt_text(req)
    golden_text = (golden.text or "").strip() if golden.text else ""

    # redaction detection first
    if _is_redacted(golden.text) or any(_is_redacted(m.content) for m in req.input_messages):
        return DropReason.REDACTED

    # Tool/function calls not supported for replay yet — skip these rows.
    # Either the request had a tool round-trip, or the golden was itself a tool
    # call (no text to compare against a replayed text response).
    if req.had_tool_calls or golden.kind == GoldenKind.TOOL_CALL:
        return DropReason.TOOL_CALLS_UNSUPPORTED

    # E3 modality
    if req.modality != Modality.TEXT:
        return DropReason.UNSUPPORTED_MODALITY

    # E1 non-empty prompt
    if not prompt:
        return DropReason.NO_PROMPT

    # E2 usable golden
    if golden.kind == GoldenKind.EMPTY:
        return DropReason.EMPTY_RESPONSE
    if golden.kind in (GoldenKind.TEXT, GoldenKind.JSON, GoldenKind.REFUSAL) and not golden_text:
        return DropReason.EMPTY_RESPONSE
    if golden.kind == GoldenKind.TOOL_CALL and not golden.tool_calls:
        return DropReason.TOOL_ONLY_NO_TEXT

    return None


# --------------------------------------------------------------------------- #
# Normalize a single row
# --------------------------------------------------------------------------- #
def normalize_row(raw: RawRow) -> Tuple[Optional[NormalizedSample], Optional[DroppedRow], Shape]:
    row = raw.data
    shape = classify(row)
    sample_id = str(row.get("request_id") or row.get("id") or f"{raw.object_uri}#{raw.row_index}")
    source = SampleSource(shape=shape, object_uri=raw.object_uri, row_index=raw.row_index)

    if row.get("__parse_error__"):
        return None, DroppedRow(id=sample_id, source=source, eval_block_reason=DropReason.PARSE_ERROR), shape

    if shape == Shape.UNKNOWN:
        resp = row.get("response") or {}
        resp_obj = resp.get("object") if isinstance(resp, dict) else None
        if resp_obj == "chat.completion.chunk":
            reason = DropReason.STREAM_FRAGMENT
        elif row.get("call_type") in NON_EVAL_CALL_TYPES or resp_obj == "list":
            reason = DropReason.NON_EVAL_CALL_TYPE
        else:
            reason = DropReason.UNKNOWN_SHAPE
        return None, DroppedRow(id=sample_id, source=source, eval_block_reason=reason), shape

    req, golden = ADAPTERS[shape](row)
    legacy_model = row.get("model_group") or row.get("model")
    reason = evaluability(req, golden)

    sample = NormalizedSample(
        id=sample_id,
        source=source,
        legacy_model=legacy_model,
        request=req,
        golden=golden,
        usage=_usage_of(row),
        original_latency_ms=_original_latency_of(row),
        tags=_tags_of(row),
        evaluable=reason is None,
        eval_block_reason=reason,
    )
    if reason is not None:
        return None, DroppedRow(id=sample_id, source=source, eval_block_reason=reason), shape
    return sample, None, shape


# --------------------------------------------------------------------------- #
# Full ingestion over one or more objects
# --------------------------------------------------------------------------- #
@dataclass
class IngestionResult:
    samples: List[NormalizedSample] = field(default_factory=list)
    dropped: List[DroppedRow] = field(default_factory=list)
    report: IngestionReport = field(default_factory=IngestionReport)


def ingest(objects: Iterable[Tuple[bytes, str]], skipped_objects: int = 0) -> IngestionResult:
    """Ingest an iterable of (raw_bytes, object_uri) into samples + report.

    `skipped_objects` counts non-log objects the caller already filtered out
    (e.g. from an S3 prefix listing), for the reconciliation identity.
    """
    result = IngestionResult()
    dropped_counts: Dict[DropReason, int] = {}
    model_counts: Dict[str, int] = {}
    shape_counts: Dict[Shape, int] = {}
    team_counts: Dict[str, int] = {}
    user_counts: Dict[str, int] = {}
    found = 0
    redacted = 0
    # Sample ids key results, scores, caches and reports, so they MUST be unique within a run.
    # Several log files can share one request_id (e.g. one end-user request that fanned out
    # into many LLM calls, one capture file per call): later repeats get '#2', '#3', ...
    seen_ids: Dict[str, int] = {}

    def _unique(base: str) -> str:
        n = seen_ids.get(base, 0) + 1
        seen_ids[base] = n
        return base if n == 1 else f"{base}#{n}"

    for raw_bytes, uri in objects:
        for raw in iter_rows(raw_bytes, uri):
            found += 1
            sample, drop, shape = normalize_row(raw)
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            if sample is not None:
                uid = _unique(sample.id)
                if uid != sample.id:
                    sample = sample.model_copy(update={
                        "id": uid, "tags": {**sample.tags, "request_id": sample.id}})
                result.samples.append(sample)
                m = sample.legacy_model or "unknown"
                model_counts[m] = model_counts.get(m, 0) + 1
                t = sample.tags.get("team")
                if t:
                    team_counts[str(t)] = team_counts.get(str(t), 0) + 1
                u = sample.tags.get("user")
                if u:
                    user_counts[str(u)] = user_counts.get(str(u), 0) + 1
            else:
                uid = _unique(drop.id)
                if uid != drop.id:
                    drop = drop.model_copy(update={"id": uid})
                result.dropped.append(drop)
                dropped_counts[drop.eval_block_reason] = dropped_counts.get(drop.eval_block_reason, 0) + 1
                if drop.eval_block_reason == DropReason.REDACTED:
                    redacted += 1

    # Skipped non-log objects are part of what was "found" (Doc 2 §6.3 identity:
    # found = evaluable + Σ dropped + skipped_objects).
    rows_seen = found
    found += skipped_objects
    total_rows = max(rows_seen, 1)
    result.report = IngestionReport(
        found=found,
        evaluable=len(result.samples),
        dropped=dropped_counts,
        skipped_objects=skipped_objects,
        detected_models=[DetectedModel(model=m, rows=c) for m, c in sorted(model_counts.items())],
        shape_mix={s: round(c / total_rows, 4) for s, c in shape_counts.items()},
        redaction_rate=round(redacted / total_rows, 4),
        teams=dict(sorted(team_counts.items())),
        users=dict(sorted(user_counts.items())),
    )
    return result
