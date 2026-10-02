# ModelShift

**LiteLLM → Bedrock GPT migration evaluation tool.**

ModelShift takes a customer's existing **LiteLLM logs** (legacy GPT models on Azure), **replays each captured request** against newer candidate GPT models on **Amazon Bedrock** using the **Responses API**, and **scores how compatible** the new responses are with the legacy "golden" responses — giving an evidence-backed *migration confidence* verdict, plus prompt / reasoning-effort / settings remediation suggestions.

Built from the spec set in [`../gpt-migration-eval-spec/`](../gpt-migration-eval-spec/).

## Demo

**Overall flow** — load logs → pick candidates → replay → score → migration-confidence verdict:

![ModelShift overall flow](assets/overall-flow.png)

**Start a run** — upload or point at logs, narrow the slice, and choose candidate models:

![New run page](assets/new-run-page.png)

**Read the results** — side-by-side candidate cards, per-dimension scores, and the migration-confidence verdict:

![Results page](assets/results-page.png)

---

## Why compatibility, not quality

A newer model that is *more accurate* but changes the output format, drops a fact, or bloats verbosity will still **break downstream consumers**. ModelShift optimizes for interchangeability with the legacy output — a broken JSON schema, a diverged phone number, or a mismatched tool call is a **hard break** (capped at *Incompatible*) regardless of how eloquent the new answer is.

## What it does

1. **Load logs** — upload a LiteLLM log file or point at an `s3://bucket/prefix`.
2. **Auto-detect & normalize** — parses SpendLogs (`proxy_server_request`), wrapped-SDK, Chat Completions, and Responses shapes; converts legacy Chat Completions to the Responses API shape; drops non-evaluable rows with full accounting.
3. **Pick candidates** — pre-filled from the migration matrix (GPT-4.1 → Luna/Terra/Sol, etc.), evaluated **side-by-side**, at **medium reasoning effort** by default.
4. **Choose call path** — **Bedrock (direct)** or **LiteLLM proxy**.
5. **Replay & score** — six dimensions (semantic, format/schema, factual, verbosity, instruction-following, tool-call) → four labels → **Migration Confidence 0–100** and a verdict band.
6. **Remediate** — cluster failures by reason, suggest changes, re-test a subset, see a before/after delta.

## Architecture

```
web/ (SPA)  ──REST/SSE──▶  FastAPI (modelshift/api.py)
                              ├─ ingest.py      (shape adapters, evaluability, drop accounting)
                              ├─ orchestrator.py(run lifecycle: ingest→replay→score→judge→aggregate→recommend)
                              ├─ adapters.py    (Bedrock / LiteLLM Responses-API candidate adapters)
                              ├─ scoring.py     (deterministic D1–D6 + hard-break + labels)
                              ├─ judge.py       (LLM-as-judge, never overrides a hard-break)
                              ├─ aggregate.py   (Migration Confidence, side-by-side, remediation, retest)
                              ├─ config.py      (matrix, model catalog, price table, rubric — all data-driven)
                              └─ schemas.py     (Pydantic data contracts)
```

## Module map

| Module | Responsibility |
|---|---|
| `modelshift/schemas.py` | NormalizedSample, CandidateResult, PairScore, CandidateVerdict, Remediation, IngestionReport |
| `modelshift/config.py` | Migration matrix, model catalog, price table, scoring rubric (config-driven — add models without code) |
| `modelshift/ingest.py` | Container detection, 4 shape adapters, ChatCompletions→Responses conversion, evaluability + reconciliation |
| `modelshift/sources.py` | File upload + S3 prefix loaders |
| `modelshift/adapters.py` | Bedrock / LiteLLM candidate adapters (reasoning inject, param strip, backoff, caching) |
| `modelshift/scoring.py` | Deterministic D1–D6, hard-break rule, exact-match short-circuit, 4-label classifier |
| `modelshift/judge.py` | LLM-as-judge (Bedrock Converse) + deterministic fast-pass |
| `modelshift/aggregate.py` | Aggregation, side-by-side ranking, remediation recommender, subset re-test |
| `modelshift/orchestrator.py` | Run lifecycle state machine + pipeline wiring + dry-run cost preview |
| `modelshift/api.py` | FastAPI `/api/v1` + SSE + static SPA mount |
| `modelshift/server.py` | Dev entrypoint (`--demo` / `--live`) |
| `web/` | Single-page UI (design tokens, dark theme, semantic label pills) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e ".[dev]"
```

## Run

**Offline demo** (no AWS — seeds a sample run so the UI has data):
```bash
python -m modelshift.server --demo --port 8971
# open http://127.0.0.1:8971
```

**Live** (real candidate calls — needs AWS credentials / a LiteLLM proxy):
```bash
# Bedrock direct:
python -m modelshift.server --live --region us-east-1 --port 8971
# via LiteLLM proxy:
python -m modelshift.server --live --litellm-base http://127.0.0.1:4000 --litellm-key sk-... --port 8971
```

Configure AWS creds in your terminal first (`aws configure` / `aws sso login`), then the default provider chain is used.

## Test

```bash
python -m pytest tests/ -q      # 62 tests
python -m flake8 modelshift tests
```

## Key API endpoints (`/api/v1`)

`POST /runs` · `POST /uploads` · `POST /runs/{id}/ingest` · `GET /runs/{id}/ingestion` · `POST /runs/{id}/plan` (dry-run cost preview) · `POST /runs/{id}/launch` · `GET /runs/{id}/stream` (SSE) · `GET /runs/{id}/results` · `GET /runs/{id}/samples` · `PATCH /runs/{id}/samples/{sid}` (human override) · `GET /runs/{id}/remediations` · `GET /models` · `GET /settings`

## Verification status

**Verified (offline, this build):**
- ✅ 62/62 unit + API tests pass; flake8 clean.
- ✅ Ingestion parses all four real LiteLLM fixture shapes; reconciliation identity holds (17 found = 12 evaluable + 5 dropped across the fixtures).
- ✅ Deterministic scoring: exact-match → Compatible; phone/email divergence, broken JSON, and tool-call mismatch → hard-break Incompatible.
- ✅ Judge never overrides a hard-break; aggregation produces Migration Confidence + verdict bands + side-by-side ranking.
- ✅ **Both call paths** exercised end-to-end: Bedrock adapter uses the `us.openai.gpt-5.6-*` inference-profile id; LiteLLM adapter uses the model-group alias.
- ✅ Full API flow: upload → ingest → dry-run plan (no spend) → launch → results with a scored verdict; SSE stream emits progress + end.
- ✅ Remediation cluster → subset re-test → before/after confidence delta.
- ✅ Dry-run makes **zero** candidate calls.

**NOT verified (requires the user's environment):**
- ⛔ **Live Bedrock / LiteLLM calls** — needs AWS credentials with `bedrock:InvokeModel` on the GPT-5.6/GPT-5.4 inference profiles (and, for the LiteLLM path, a reachable proxy). The exact Responses-API request/response envelope should be validated against the live service (per prior experience, some constraints only surface on the real call).
- ⛔ **Browser screenshot of the SPA** — the build host has no headless browser and Browser Mode was off, so the UI was verified through its data endpoints (assets 200, every endpoint returns coherent data), not a pixel capture. Enable Browser Mode to capture screenshots.
- ⛔ **Persistence** — v1 uses an in-process run store (spec-sanctioned). A SQLite/Postgres + S3 object-store layer is the documented next step.

## Not-yet-built (documented in the spec, deferred for v1)

- SQLite/Postgres persistence + object store; SQS worker fan-out for large corpora.
- Export report (HTML/PDF).
- Embedding-based D1 (currently token-overlap; the LLM judge augments semantic scoring).
- Auth for a deployed instance.
