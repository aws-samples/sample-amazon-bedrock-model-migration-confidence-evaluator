"""Dev server entrypoint.

Usage:
  # offline demo (no AWS) — seeds a sample run so the UI has data:
  python -m modelshift.server --demo --port 8971

  # live: uses real Bedrock/LiteLLM transports (needs AWS creds / proxy)
  python -m modelshift.server --live --region us-east-1 --port 8971
"""
from __future__ import annotations

import argparse

import uvicorn

from . import api as api_mod
from .config import DEFAULT_SETTINGS
from .ingest import ingest
from .orchestrator import CandidateConfig, Run, RunConfig, new_run_id
from .schemas import CallPath


def _demo_candidate_transport():
    """Echoes a plausible candidate answer keyed to the prompt, with a couple of
    deliberate incompatibilities so the UI shows a mixed verdict."""
    canned = {
        "What's my primary care copay?": "Your in-network primary care copay is $25.",
        "What is the nurse line number?": "The nurse line is 1-866-606-3700.",  # diverges -> incompatible
    }

    def _t(payload):
        last = ""
        for item in payload.get("input", []):
            for part in item.get("content", []):
                last = part.get("text", last)
        out = canned.get(last, last)  # default: echo (exact match -> compatible)
        return {"output_text": out, "usage": {"input_tokens": 20, "output_tokens": 12}}

    return _t


def seed_demo_run():
    sl_rows = [
        {"request_id": "sl-1", "model": "gpt-4.1", "messages": {},
         "proxy_server_request": {"model": "gpt-4.1", "messages": [
             {"role": "system", "content": "You are CareConcierge. Be concise."},
             {"role": "user", "content": "What's my primary care copay?"}]},
         "response": {"choices": [{"message": {
             "role": "assistant",
             "content": "Your in-network primary care copay is $25."},
             "finish_reason": "stop"}]}},
        {"request_id": "sl-2", "model": "gpt-4.1", "messages": {},
         "proxy_server_request": {"model": "gpt-4.1", "messages": [
             {"role": "user", "content": "What is the nurse line number?"}]},
         "response": {"choices": [{"message": {
             "role": "assistant",
             "content": "The nurse line is 1-800-555-0142."},
             "finish_reason": "stop"}]}},
    ]
    import json as _json
    sl = _json.dumps({"data": sl_rows}).encode("utf-8")
    res = ingest([(sl, "upload:demo")])
    run = Run(run_id=new_run_id(), name="Demo — member-services (GPT-4.1)")
    run.samples = res.samples
    run.ingestion = res.report
    run.config = RunConfig(
        candidates=[CandidateConfig(model="gpt-5.6-luna"), CandidateConfig(model="gpt-5.6-terra")],
        call_path=CallPath.BEDROCK,
    )
    api_mod._RUNS[run.run_id] = run
    api_mod._ORCH.run(run)
    return run.run_id


def _env(name, default=None):
    import os
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def main():
    # Every option falls back to an env var so a container can be configured
    # entirely via task-definition environment (no command override needed).
    # Precedence for each: CLI flag > env var > built-in default.
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="offline demo transports + a seeded run")
    ap.add_argument("--live", action="store_true", help="use real Bedrock/LiteLLM transports")
    ap.add_argument("--region", default=_env("MODELSHIFT_REGION", "us-east-1"))
    ap.add_argument("--litellm-base", default=_env("MODELSHIFT_LITELLM_BASE"))
    ap.add_argument("--litellm-key", default=_env("MODELSHIFT_LITELLM_KEY"))
    ap.add_argument("--bedrock-key", default=_env("MODELSHIFT_BEDROCK_KEY"),
                    help="Bedrock API key to use the /openai/v1 endpoint instead of invoke_model")
    ap.add_argument("--host", default=_env("MODELSHIFT_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(_env("MODELSHIFT_PORT", "8971")))
    args = ap.parse_args()

    # Mode: explicit --demo/--live flag wins; else MODELSHIFT_MODE=demo|live; else live.
    mode = "demo" if args.demo else ("live" if args.live else _env("MODELSHIFT_MODE", "live").lower())
    live = mode != "demo"

    if live:
        from .adapters import (make_bedrock_converse_transport, make_bedrock_openai_transport,
                               make_bedrock_sigv4_transport, make_litellm_transport)
        mantle_t = None
        if args.litellm_base:
            cand = make_litellm_transport(args.litellm_base, args.litellm_key)
        elif args.bedrock_key:
            # explicit Bedrock API key (Bearer)
            cand = make_bedrock_openai_transport(args.bedrock_key, args.region, endpoint="runtime")
            mantle_t = make_bedrock_openai_transport(args.bedrock_key, args.region, endpoint="mantle")
        else:
            # DEFAULT: sign /openai/v1 with the machine's local AWS credentials (no key).
            cand = make_bedrock_sigv4_transport(args.region, endpoint="runtime")
            mantle_t = make_bedrock_sigv4_transport(args.region, endpoint="mantle")
        converse_t = make_bedrock_converse_transport(args.region)  # for Anthropic/Claude candidates
        # LiteLLM call path (UI "LiteLLM" endpoint choice): route through the local
        # LiteLLM proxy so calls are proxied to Bedrock via the gateway, not sent
        # to Bedrock directly. Precedence: CLI flag > persisted Settings (editable in
        # the UI) > local dev proxy default.
        litellm_base = args.litellm_base or DEFAULT_SETTINGS.litellm_base or "http://127.0.0.1:4000"
        from .config import resolve_litellm_key
        # No built-in key: a proxy with a master key needs one via --litellm-key,
        # MODELSHIFT_LITELLM_KEY / Secrets Manager, or Settings in the UI.
        litellm_key = args.litellm_key or resolve_litellm_key(DEFAULT_SETTINGS) or ""
        litellm_t = make_litellm_transport(litellm_base, litellm_key)
        from .judge import make_bedrock_judge_transport, make_litellm_judge_transport
        judge_t = make_bedrock_judge_transport(DEFAULT_SETTINGS.judge_model, args.region)
        # Judge over the proxy — used only when a run's call path is LiteLLM, so the
        # judge traffic is proxied through the gateway too (same base/key as candidates).
        litellm_judge_t = make_litellm_judge_transport(
            litellm_base, litellm_key, DEFAULT_SETTINGS.judge_litellm_model)
        api_mod.configure_transports(candidate_transport=cand, judge_transport=judge_t,
                                     mantle_transport=mantle_t, converse_transport=converse_t,
                                     litellm_transport=litellm_t,
                                     litellm_judge_transport=litellm_judge_t)
    else:
        api_mod.configure_transports(candidate_transport=_demo_candidate_transport(), judge_transport=None)
        rid = seed_demo_run()
        print(f"[demo] seeded run {rid}")

    uvicorn.run(api_mod.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
