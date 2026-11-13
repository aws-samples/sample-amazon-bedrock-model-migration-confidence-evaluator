"""Candidate model adapters (Doc 3 §3).

Two call paths behind one interface (CallPath.BEDROCK / CallPath.LITELLM), both
rendering the Responses API shape from a NormalizedSample and returning a
uniform CandidateResult. Features:
  - reasoning-effort injection (default medium; NOT taken from the legacy log)
  - unsupported-param strip + single retry (GPT-5.x reject Converse temperature)
  - retry/backoff on throttling/transient errors
  - output caching keyed by (sample_id | model | settings | prompt)

The actual network calls are isolated in `_invoke_bedrock` / `_invoke_litellm`
so unit tests can inject a fake transport without AWS/LiteLLM.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from .config import MODEL_CATALOG, base_model, estimate_cost
from .schemas import (
    CallPath,
    CandidateOutput,
    CandidateResult,
    CandidateSettings,
    GoldenKind,
    NormalizedSample,
    ToolCall,
    Usage,
)

# Params that reasoning-family models reject and must be stripped (Doc 3 §3.2).
UNSUPPORTED_PARAMS = {"temperature", "top_p", "presence_penalty", "frequency_penalty"}


def cache_key(sample: NormalizedSample, model: str, settings: CandidateSettings,
              replay_index: int = 0) -> str:
    prompt = json.dumps(
        {
            "instructions": sample.request.instructions,
            "input": [m.model_dump() for m in sample.request.input_messages],
        },
        sort_keys=True,
    )
    raw = f"{sample.id}|{model}|{settings.reasoning_effort.value}|{prompt}"
    if replay_index > 0:
        # Each replay is a real, separate call: never answer it from another replay's cache.
        # Replay 0 keeps the original key so existing caches and results are unchanged.
        raw += f"|r{replay_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def render_responses_payload(sample: NormalizedSample,
                             model: str,
                             settings: CandidateSettings,
                             extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render a NormalizedSample into a Responses API request payload.

    Never emits a blank content block (skips empty turns) — prior lesson.
    Proactively strips known reasoning-unsupported params (e.g. temperature)
    and records them under the private "__dropped_params__" key.
    """
    input_items: List[Dict[str, Any]] = []
    for m in sample.request.input_messages:
        if not (m.content and m.content.strip()):
            continue  # never emit a blank block
        part_type = "output_text" if m.role == "assistant" else "input_text"
        input_items.append({
            "type": "message",
            "role": m.role,
            "content": [{"type": part_type, "text": m.content}],
        })

    payload: Dict[str, Any] = {
        "model": model,
        "input": input_items,
        "reasoning": {"effort": settings.reasoning_effort.value},
    }
    if sample.request.instructions:
        payload["instructions"] = sample.request.instructions
    if sample.request.tools:
        payload["tools"] = sample.request.tools
    if sample.request.response_format:
        payload["text"] = {"format": sample.request.response_format}
    dropped: List[str] = []
    if extra_params:
        for k, v in extra_params.items():
            if k in UNSUPPORTED_PARAMS:
                dropped.append(k)
            else:
                payload[k] = v
    payload["__dropped_params__"] = dropped
    return payload


def _has_usable_input(payload: Dict[str, Any]) -> bool:
    return bool(payload.get("input")) or bool(payload.get("instructions"))


def to_chat_completions_body(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the internal canonical payload to the Chat Completions body that
    Bedrock OpenAI models (and LiteLLM /chat/completions) actually accept.

    Validated live: GPT-5.6 on Bedrock invoke_model wants `messages`,
    top-level `reasoning_effort`, and `max_completion_tokens` (NOT the Responses
    `input`/`reasoning.effort` shape, which raises 'missing field messages').
    """
    messages: List[Dict[str, Any]] = []
    if payload.get("instructions"):
        messages.append({"role": "system", "content": payload["instructions"]})
    for item in payload.get("input", []) or []:
        role = item.get("role", "user")
        text = ""
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("text"):
                text += part["text"]
        if text:
            messages.append({"role": role, "content": text})
    body: Dict[str, Any] = {"messages": messages}
    eff = (payload.get("reasoning") or {}).get("effort")
    if eff:
        body["reasoning_effort"] = eff
    if payload.get("tools"):
        body["tools"] = payload["tools"]
    if payload.get("text", {}).get("format"):
        body["response_format"] = payload["text"]["format"]
    # carry any non-private extra params (unsupported ones already stripped)
    for k, v in payload.items():
        if k not in ("model", "input", "reasoning", "instructions", "tools", "text") \
                and not k.startswith("__"):
            body[k] = v
    return body


# --------------------------------------------------------------------------- #
# Transport protocol (fake-able in tests)
# --------------------------------------------------------------------------- #
class Transport(Protocol):
    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a Responses-API-shaped dict, or raise on error."""
        ...


def _parse_responses_output(resp: Dict[str, Any]) -> CandidateOutput:
    # Chat Completions shape (real Bedrock/LiteLLM response for these models)
    if resp.get("choices"):
        msg = (resp["choices"][0] or {}).get("message") or {}
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        if content:
            return CandidateOutput(kind=GoldenKind.TEXT, text=content)
        if tool_calls:
            tcs = [ToolCall(name=(tc.get("function") or {}).get("name", ""),
                            arguments=(tc.get("function") or {}).get("arguments"))
                   for tc in tool_calls if isinstance(tc, dict)]
            return CandidateOutput(kind=GoldenKind.TOOL_CALL, tool_calls=tcs)
        return CandidateOutput(kind=GoldenKind.EMPTY)
    # Responses API shape (fallback / future)
    if resp.get("output_text"):
        return CandidateOutput(kind=GoldenKind.TEXT, text=resp["output_text"])
    texts: List[str] = []
    tool_calls2: List[ToolCall] = []
    for item in resp.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text") and part.get("text"):
                    texts.append(part["text"])
        elif item.get("type") == "function_call":
            tool_calls2.append(ToolCall(name=item.get("name", ""), arguments=item.get("arguments")))
    if texts:
        return CandidateOutput(kind=GoldenKind.TEXT, text="\n".join(texts))
    if tool_calls2:
        return CandidateOutput(kind=GoldenKind.TOOL_CALL, tool_calls=tool_calls2)
    return CandidateOutput(kind=GoldenKind.EMPTY)


def _usage_from_responses(resp: Dict[str, Any]) -> Usage:
    u = resp.get("usage") or {}
    ctd = u.get("completion_tokens_details") or {}
    return Usage(
        prompt_tokens=u.get("input_tokens") or u.get("prompt_tokens"),
        completion_tokens=u.get("output_tokens") or u.get("completion_tokens"),
        total_tokens=u.get("total_tokens"),
        reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens")
        or ctd.get("reasoning_tokens"),
    )


def _is_unsupported_param_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in UNSUPPORTED_PARAMS) and (
        "not support" in msg or "unsupported" in msg or "unexpected" in msg or "blank" not in msg
    )


def _is_throttle_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "throttl" in msg or "toomanyrequests" in msg or "rate exceeded" in msg or "429" in msg


# --------------------------------------------------------------------------- #
# Base adapter
# --------------------------------------------------------------------------- #
class CandidateAdapter:
    """Common interface. Subclasses provide a transport."""

    call_path: CallPath

    def __init__(self, transport: Transport, max_retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        from .concurrency import SingleFlight
        self._transport = transport
        self._max_retries = max_retries
        self._sleep = sleep
        # Thread-safe result cache: concurrent workers asking for the same call wait for the
        # first one instead of sending it twice (parallel-calls R2.2).
        self._flight: "SingleFlight[CandidateResult]" = SingleFlight()
        self._cache: Dict[str, CandidateResult] = self._flight.store
        # Fallback observer, called as on_retry(next_attempt, max_attempts, reason, error) before
        # each back-off sleep. Prefer the per-call ``on_retry`` argument of generate(): this shared
        # attribute is only safe when one call runs at a time.
        self.on_retry: Optional[Callable[[int, int, str, str], None]] = None

    def _notify_retry(self, attempt: int, reason: str, exc: Exception,
                      wait_s: float = 0.0, cb: Optional[Callable[..., None]] = None) -> None:
        try:
            if cb is not None:
                cb(attempt + 2, self._max_retries + 1, reason, str(exc)[:200], wait_s)
            elif self.on_retry is not None:
                self.on_retry(attempt + 2, self._max_retries + 1, reason, str(exc)[:200])
        except Exception:  # noqa: BLE001 — an observer must never break a call
            pass

    def _model_ref(self, model: str) -> str:
        raise NotImplementedError

    def generate(self,
                 sample: NormalizedSample,
                 model: str,
                 settings: Optional[CandidateSettings] = None,
                 extra_params: Optional[Dict[str, Any]] = None,
                 replay_index: int = 0,
                 on_retry: Optional[Callable[..., None]] = None) -> CandidateResult:
        """``on_retry(next_attempt, max_attempts, reason, error, wait_s)`` is called before each
        back-off sleep of THIS call (safe with concurrent calls)."""
        settings = settings or CandidateSettings()
        key = cache_key(sample, model, settings, replay_index)
        result, cached = self._flight.get_or_compute(
            key, lambda: self._generate_uncached(sample, model, settings, extra_params,
                                                 replay_index, key, on_retry))
        if cached:
            # A copy flagged from_cache: its latency was measured by an earlier call, so
            # aggregation must not count it as a fresh timing.
            return result.model_copy(update={"from_cache": True, "replay_index": replay_index})
        return result

    def _generate_uncached(self, sample: NormalizedSample, model: str, settings: CandidateSettings,
                           extra_params: Optional[Dict[str, Any]], replay_index: int, key: str,
                           on_retry: Optional[Callable[..., None]]) -> CandidateResult:
        model_ref = self._model_ref(model)
        payload = render_responses_payload(sample, model_ref, settings, extra_params)
        dropped_params: List[str] = list(payload.pop("__dropped_params__", []))
        if not _has_usable_input(payload):
            return CandidateResult(
                sample_id=sample.id, model=model, settings=settings,
                status="error", error="no usable prompt text", cache_key=key, replay_index=replay_index,
            )

        attempt = 0
        while True:
            # Time each attempt on its own: back-off sleeps between retries happen OUTSIDE
            # this window, so latency_ms is the duration of the call that produced the result.
            started = time.time()
            try:
                resp = self._transport(payload)
                latency_ms = int((time.time() - started) * 1000)
                output = _parse_responses_output(resp)
                usage = _usage_from_responses(resp)
                cost = estimate_cost(
                    model,
                    usage.prompt_tokens or 0,
                    usage.completion_tokens or 0,
                    usage.reasoning_tokens or 0,
                )
                return CandidateResult(
                    sample_id=sample.id, model=model,
                    settings=CandidateSettings(reasoning_effort=settings.reasoning_effort,
                                               dropped_params=dropped_params),
                    output=output, usage=usage,
                    cost_estimate=cost or 0.0, cost_known=cost is not None,
                    latency_ms=latency_ms, attempts=attempt + 1,
                    status="ok", cache_key=key, replay_index=replay_index,
                )
            except Exception as exc:  # noqa: BLE001 — deliberate: classify & retry
                failed_ms = int((time.time() - started) * 1000)
                if _is_throttle_error(exc) and attempt < self._max_retries:
                    wait_s = min(2 ** attempt, 8)
                    self._notify_retry(attempt, "throttled", exc, wait_s, on_retry)
                    self._sleep(wait_s)
                    attempt += 1
                    continue
                if attempt < self._max_retries and not _is_unsupported_param_error(exc):
                    wait_s = min(2 ** attempt, 8)
                    self._notify_retry(attempt, "error", exc, wait_s, on_retry)
                    self._sleep(wait_s)
                    attempt += 1
                    continue
                return CandidateResult(
                    sample_id=sample.id, model=model,
                    settings=CandidateSettings(reasoning_effort=settings.reasoning_effort,
                                               dropped_params=dropped_params),
                    status="error", error=str(exc),
                    latency_ms=failed_ms, attempts=attempt + 1, cache_key=key, replay_index=replay_index,
                )


class BedrockResponsesAdapter(CandidateAdapter):
    call_path = CallPath.BEDROCK

    def __init__(self, transport, endpoint="runtime", **kw):
        super().__init__(transport, **kw)
        self._endpoint = endpoint  # "runtime" | "mantle"

    def _model_ref(self, model: str) -> str:
        entry = MODEL_CATALOG.get(base_model(model))
        if not entry:
            return base_model(model)
        return entry.mantle_model_id if self._endpoint == "mantle" else entry.bedrock_model_id


class LiteLLMResponsesAdapter(CandidateAdapter):
    call_path = CallPath.LITELLM

    def _model_ref(self, model: str) -> str:
        entry = MODEL_CATALOG.get(base_model(model))
        return entry.litellm_model_group if entry else base_model(model)


def make_bedrock_converse_transport(region: str = "us-east-1") -> Transport:
    """Bedrock Converse transport for Anthropic (Claude) candidates.

    Converts the canonical payload (instructions + input[]) into a Converse
    request, then maps the Converse response back into the OpenAI-ish
    {choices:[{message:{content}}], usage:{...}} shape the parser understands.
    """
    client = _boto_client(region)

    def _invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
        system = [{"text": payload["instructions"]}] if payload.get("instructions") else []
        messages = []
        for item in payload.get("input", []) or []:
            role = "assistant" if item.get("role") == "assistant" else "user"
            text = "".join(p.get("text", "") for p in (item.get("content") or []) if isinstance(p, dict))
            if text:
                messages.append({"role": role, "content": [{"text": text}]})
        if not messages:
            messages = [{"role": "user", "content": [{"text": " "}]}]
        kwargs: Dict[str, Any] = {"modelId": payload["model"], "messages": messages,
                                  "inferenceConfig": {"maxTokens": 2048}}
        if system:
            kwargs["system"] = system
        resp = client.converse(**kwargs)
        msg = resp.get("output", {}).get("message", {})
        text = "".join(b.get("text", "") for b in msg.get("content", []) if isinstance(b, dict))
        u = resp.get("usage", {})
        return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": u.get("inputTokens"), "completion_tokens": u.get("outputTokens"),
                          "total_tokens": u.get("totalTokens")}}

    return _invoke


def make_bedrock_sigv4_transport(region: str = "us-east-1", endpoint: str = "runtime") -> Transport:
    """Bedrock OpenAI-compatible /openai/v1 transport signed with SigV4 using the
    machine's local AWS credentials (default provider chain) — NO API key needed.

    endpoint="runtime": https://bedrock-runtime.{region}.amazonaws.com/openai/v1
                        (Luna/Terra/Sol/GPT-5.4 via us.openai.* profile ids)
    endpoint="mantle":  https://bedrock-mantle.{region}.api.aws/openai/v1
                        (openai.* ids, e.g. openai.gpt-5.4)
    """
    import boto3  # lazy
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    if endpoint == "mantle":
        base = f"https://bedrock-mantle.{region}.api.aws/openai/v1"
    else:
        base = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"
    session = boto3.Session()
    creds_src = session.get_credentials()
    creds_lock = threading.Lock()
    client = _shared_http_client()

    def _invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
        body = to_chat_completions_body(payload)
        body["model"] = payload["model"]
        data = json.dumps(body)
        url = f"{base}/chat/completions"
        with creds_lock:  # boto3 Session isn't documented as thread-safe; refresh one at a time
            creds = creds_src.get_frozen_credentials()
        aws_req = AWSRequest(method="POST", url=url, data=data,
                             headers={"Content-Type": "application/json"})
        SigV4Auth(creds, "bedrock", region).add_auth(aws_req)
        r = client.post(url, content=data, headers=dict(aws_req.headers))
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} from {url}: {r.text[:400]}")
        return r.json()

    return _invoke


def make_bedrock_transport(region: str = "us-east-1") -> Transport:
    """Real Bedrock transport via boto3 invoke_model (lazy import).

    Bedrock's GPT-5.6 invoke_model uses the Chat Completions body shape
    (validated live). We convert the internal canonical payload accordingly.
    Note: the AWS-recommended path is the /openai/v1 endpoint — see
    make_bedrock_openai_transport.
    """
    client = _boto_client(region)

    def _invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
        body = to_chat_completions_body(payload)
        resp = client.invoke_model(
            modelId=payload["model"],
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(resp["body"].read())

    return _invoke


def make_bedrock_openai_transport(api_key: str, region: str = "us-east-1",
                                  endpoint: str = "runtime") -> Transport:
    """Bedrock OpenAI-compatible /openai/v1 Chat Completions transport (Bearer key).

    endpoint="runtime": https://bedrock-runtime.{region}.amazonaws.com/openai/v1
                        (Luna/Terra/Sol/GPT-5.4 via us.openai.* profile ids)
    endpoint="mantle":  https://bedrock-mantle.{region}.api.aws/openai/v1
                        (openai.* ids, e.g. openai.gpt-5.4)
    """
    if endpoint == "mantle":
        base = f"https://bedrock-mantle.{region}.api.aws/openai/v1"
    else:
        base = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"

    client = _shared_http_client()

    def _invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
        body = to_chat_completions_body(payload)
        body["model"] = payload["model"]
        r = client.post(f"{base}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"}, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:400]}")
        return r.json()

    return _invoke


def make_litellm_transport(base_url: str, api_key: str) -> Transport:
    """Real LiteLLM transport via httpx (lazy import) to /v1/chat/completions."""
    client = _shared_http_client()

    def _invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
        body = to_chat_completions_body(payload)
        body["model"] = payload["model"]  # LiteLLM routes by model in the body
        r = client.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:400]}")
        return r.json()

    return _invoke


_HTTP_CLIENT = None
_HTTP_LOCK = threading.Lock()
HTTP_POOL_SIZE = 64


def _shared_http_client():
    """One thread-safe httpx client for all transports: reuses connections so concurrent calls
    don't pay TLS setup each time (keeps measured latency honest; parallel-calls R8.3)."""
    global _HTTP_CLIENT
    import httpx  # lazy
    with _HTTP_LOCK:
        if _HTTP_CLIENT is None:
            _HTTP_CLIENT = httpx.Client(timeout=120, limits=httpx.Limits(
                max_connections=HTTP_POOL_SIZE, max_keepalive_connections=HTTP_POOL_SIZE))
        return _HTTP_CLIENT


def _boto_client(region: str):
    """bedrock-runtime client with a connection pool sized for concurrent calls
    (botocore's default is 10). Low-level boto3 clients are thread-safe."""
    import boto3  # lazy
    from botocore.config import Config
    return boto3.client("bedrock-runtime", region_name=region,
                        config=Config(max_pool_connections=HTTP_POOL_SIZE))


def discover_litellm_models(base_url: str, api_key: str, timeout: float = 8.0) -> List[str]:
    """Query the LiteLLM proxy's /v1/models and return the list of model-group
    aliases it serves. Raises RuntimeError on any failure so the caller can
    surface a reason (the API endpoint catches it and degrades gracefully)."""
    import httpx  # lazy

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=timeout) as client:
        r = client.get(f"{base_url.rstrip('/')}/v1/models", headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
        data = r.json()
    # OpenAI-compatible shape: {"data": [{"id": "<alias>", ...}, ...]}
    items = data.get("data") if isinstance(data, dict) else data
    ids = [m.get("id") for m in (items or []) if isinstance(m, dict) and m.get("id")]
    # Stable, de-duplicated order.
    seen: Dict[str, None] = {}
    for i in ids:
        seen.setdefault(i, None)
    return list(seen.keys())
