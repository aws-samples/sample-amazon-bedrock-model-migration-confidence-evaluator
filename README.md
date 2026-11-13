# amazon-bedrock-model-migration-confidence-evaluator

**Bedrock GPT migration evaluation tool.**

ModelShift takes a customer's existing **LiteLLM logs**, **replays each captured request** against newer candidate GPT models on **Amazon Bedrock** using the **Responses API**, and **scores how compatible** the new responses are with the legacy "golden" responses — giving an evidence-backed *migration confidence* verdict, plus prompt / reasoning-effort / settings remediation suggestions.

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

1. **Load logs — or enter prompts** — upload a LiteLLM log file (or a **.zip** of them, nested folders included), point at an `s3://bucket/prefix`, **or type one or more prompt / golden-response pairs directly in the UI** when you don't have a log export. App-side prompt captures and saved chat conversations are recognized too. Hand-entered pairs become the dataset and flow through the exact same pipeline as logged requests.
2. **Auto-detect & normalize** — parses SpendLogs (`proxy_server_request`), wrapped-SDK, Chat Completions, and Responses shapes; converts legacy Chat Completions to the Responses API shape; drops non-evaluable rows with full accounting.
3. **Pick candidates** — pre-filled from the migration matrix (GPT-4.1 → Luna/Terra/Sol, etc.), evaluated **side-by-side**. Pick one or more **reasoning efforts**; each `model@effort` runs as its own candidate, and only the efforts each model accepts on Bedrock are offered.
4. **Choose call path** — **Bedrock (direct)** or **LiteLLM proxy**.
5. **Replay & score** — six dimensions (semantic, format/schema, factual, verbosity, instruction-following, tool-call) → four labels → **Migration Confidence 0–100** and a verdict band. Optionally **replay each request 2–5 times** to measure consistency (D7). Calls are sent **several at a time**, slowing down automatically if Bedrock throttles.
6. **Recommend** — name the best model **all things considered** (quality, cost, latency), plus Best quality, Most economical and Lowest latency, each with a short data-driven explanation.
7. **Remediate** — cluster failures by reason, suggest changes, re-test a subset, see a before/after delta.

## Tune the judge to your use case (Skills / steering docs)

Compatibility isn't the same for every workload — a dosage number, an order ID, or a specific disclaimer may be make-or-break for your consumers, while a greeting style or extra pleasantry may not matter at all. On the **New Run → Configuration** step you can hand the LLM-as-judge your own **evaluation steering** — either typed inline or uploaded as a **`.md` / `.txt`** "Skill" / steering document — to encode those use-case-specific nuances.

The steering text is injected into the judge's prompt as explicit guidance, so the judge applies *your* rules when it scores each candidate answer against the golden. For example:

> *"Treat any change to a dosage number or a member ID as Incompatible. A shorter answer that keeps every fact is Compatible. Ignore differences in greeting style or sign-off."*

Notes:
- **Affects the LLM-judge dimensions only.** Deterministic **hard breaks** (broken JSON schema, a diverged fact like a phone/email, a mismatched tool call) still cap a result at *Incompatible* regardless of steering — steering can tighten or relax the judgment calls, not override a structural break.
- **Re-judged on change.** The steering text is part of the judge cache key, so editing it re-scores affected evaluations rather than serving a stale verdict.
- **Reusable.** Keep your steering docs in version control and upload the right one per workload (claims triage, pharmacy, member chat, …) so each run is judged by the standard that use case actually cares about.

## Reading the results

**The recommendation.** Fit (Migration Confidence) decides *whether* a model can be recommended; cost and latency help decide *which*.

- **Quality gate first.** Only candidates with enough confidence (default ≥ 65), few hard breaks (≤ 5%) and few errors can be recommended. If none pass, the page says so and names the closest by fit instead of forcing a pick.
- **Best overall** is the highest weighted score of quality, cost and latency (*Balanced* = 50 / 30 / 20; also *Quality*, *Cost*, *Latency* or custom weights, changeable after the run without re-running anything). Quality is always pure Migration Confidence; nothing else is blended into it.
- **Best quality, Most economical, Lowest latency** are named alongside, each with a plain trade-off against Best overall. Cheaper and faster picks must stay within 10 confidence points of the best quality.
- **Why this pick** gives two short lines per pick, built from the run's own numbers, e.g. *"Top weighted score (97.7), 14.6 points ahead of gpt-5.6-luna@high, with the highest confidence (94.5) and the lowest cost ($0.73 per 1K requests)"*. A caveat only appears when it's true, such as a model that's fastest per output token but slower per request because its answers are longer.
- **Cost** uses public Amazon Bedrock list prices and assumes no prompt caching. **Latency** is measured from the machine running the evaluation, so it's indicative, not a production benchmark.

**Request replays and consistency (D7).** Turn on *Replay requests* on the Configuration step to send each request 2–5 times (multiplying calls and cost). Every replayed answer is scored like a normal call. A separate **D7 Consistency** score (0–1) measures how alike the answers to the same request are and how often they got the same label; it never changes a label or Migration Confidence. Models with unstable answers get an **Unstable** badge, the results list the least consistent requests, and optional gates in Settings can require a minimum consistency or a maximum *flip rate* (requests where some answers hard-break and others don't).

**Parallel calls.** Calls are sent several at a time (**Settings → Parallel calls**, default 4, max 16, with a server-wide cap across runs). If Bedrock throttles, ModelShift halves that endpoint's parallel calls and speeds back up after clean calls; throttled calls are retried, so no answers are lost. The Running view shows every call in flight, throttles and any slow-down, and the results record an execution summary such as *"Sent 4 at a time · 3 throttles · slowed to 2 for 1m 35s"*. Results are stored in the same order as a one-at-a-time run, so verdicts don't depend on concurrency.

The same recommendation, explanations, consistency and execution summary appear in the **PDF report** and the **JSON report**.

## Architecture

```
web/ (SPA)  ──REST/SSE──▶  FastAPI (modelshift/api.py)
                              ├─ ingest.py      (shape adapters, evaluability, drop accounting)
                              ├─ orchestrator.py(run lifecycle: ingest→replay→score→judge→aggregate→recommend)
                              ├─ adapters.py    (Bedrock / LiteLLM Responses-API candidate adapters)
                              ├─ scoring.py     (deterministic D1–D6 + hard-break + labels)
                              ├─ judge.py       (LLM-as-judge, never overrides a hard-break)
                              ├─ aggregate.py   (Migration Confidence, side-by-side, remediation, retest)
                              ├─ recommend.py   (quality gate, weighted best overall, named alternatives)
                              ├─ explain.py     ("why this pick" text from the run's numbers)
                              ├─ consistency.py (D7 consistency across replays)
                              ├─ concurrency.py (parallel calls, adaptive throttle slow-down, server cap)
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
| `modelshift/recommend.py` | Quality gate, axis scores, weighted Best overall, Best quality / Most economical / Lowest latency |
| `modelshift/explain.py` | Short data-driven "why this pick" text for each recommendation |
| `modelshift/consistency.py` | D7 Consistency across replayed answers (agreement, unstable requests, flip rate) |
| `modelshift/concurrency.py` | Parallel call pool, per-endpoint adaptive throttling, server-wide cap, fail-fast, execution stats |
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
python -m pytest tests/ -q      # 332 tests
python -m flake8 modelshift tests
```

