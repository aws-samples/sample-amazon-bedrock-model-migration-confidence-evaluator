"""FastAPI backend (Doc 5 §3).

In-process run store + orchestrator for v1 (clean seam to add SQLite/Postgres
and SQS later). Candidate/judge transports default to live factories but can be
overridden via `configure_transports()` for tests and offline runs.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import DEFAULT_SETTINGS, MIGRATION_MATRIX, MODEL_CATALOG, RECOMMENDATION_SETTING_KEYS
from .ingest import ingest
from .orchestrator import (
    CandidateConfig,
    Orchestrator,
    Run,
    RunConfig,
    RunStatus,
    default_candidates_for,
    new_run_id,
    run_async,
)
from .schemas import CallPath, CompatLabel, ReasoningEffort
from .version import BUILD_ID, version_info
from .auth import require_auth

app = FastAPI(title="ModelShift API", version=BUILD_ID)


# Enforce the configured auth mode on EVERY request (including the static SPA
# mount, which isn't a normal route so a route-level dependency wouldn't cover it).
# require_auth() leaves OPEN_PATHS (health/version) unauthenticated and is a no-op
# when MODELSHIFT_AUTH_MODE is unset/none.
@app.middleware("http")
async def _auth_middleware(request, call_next):
    from fastapi import HTTPException as _HTTPException
    from fastapi.responses import JSONResponse
    try:
        require_auth(request)
    except _HTTPException as e:
        return JSONResponse(status_code=e.status_code, content=e.detail, headers=e.headers)
    return await call_next(request)

# ---- in-process state ----------------------------------------------------- #
_RUNS: Dict[str, Run] = {}
_CANCEL: Dict[str, bool] = {}
_ORCH: Orchestrator = Orchestrator()  # replaced by configure_transports for offline/tests


def _load_persisted_runs() -> None:
    try:
        from .store import load_all_runs
        _RUNS.update(load_all_runs())
    except Exception:  # noqa: BLE001 — never let a bad run file block startup
        pass


def _save_run(run: Run) -> None:
    try:
        from .store import save_run
        save_run(run)
    except Exception:  # noqa: BLE001 — persistence must never break a request
        pass


_load_persisted_runs()


def configure_transports(candidate_transport=None, judge_transport=None, mantle_transport=None,
                         converse_transport=None, litellm_transport=None,
                         litellm_judge_transport=None) -> None:
    global _ORCH
    _ORCH = Orchestrator(candidate_transport=candidate_transport,
                         judge_transport=judge_transport,
                         mantle_transport=mantle_transport,
                         converse_transport=converse_transport,
                         litellm_transport=litellm_transport,
                         litellm_judge_transport=litellm_judge_transport,
                         settings=DEFAULT_SETTINGS)


def _get_run(run_id: str) -> Run:
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "not_found",
                                                               "message": f"run {run_id} not found"}})
    return run


# ---- request models ------------------------------------------------------- #
class CreateRunBody(BaseModel):
    name: str
    description: str = ""


class ManualPrompt(BaseModel):
    """A single hand-entered golden request: the prompt that was sent and the
    response the current model gave. Optional system/developer instructions and a
    legacy-model label. These are synthesized into Chat Completions-shaped rows and
    run through the exact same ingest() pipeline as uploaded logs."""
    prompt: str
    golden: str
    instructions: Optional[str] = None
    model: Optional[str] = None


class IngestBody(BaseModel):
    s3_uri: Optional[str] = None  # if omitted, use a prior /uploads upload_id
    upload_id: Optional[str] = None
    manual_prompts: Optional[List[ManualPrompt]] = None  # UI-entered golden pairs


class CandidateBody(BaseModel):
    model: str
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM


class RunConfigBody(BaseModel):
    candidates: Optional[List[CandidateBody]] = None
    name: Optional[str] = None
    call_path: CallPath = CallPath.BEDROCK
    bedrock_endpoint: str = "runtime"   # runtime | mantle
    sampling_mode: str = "all"
    sampling_n: int = 200
    sampling_seed: int = 42
    use_judge: bool = True
    judge_steering: Optional[str] = None
    filter_teams: Optional[List[str]] = None
    filter_users: Optional[List[str]] = None
    dry_run: bool = False
    monthly_volume: Optional[int] = None  # expected production requests/month (optional)
    weights: Optional["WeightsBody"] = None  # recommendation weights preset (or custom)
    replays: int = 1  # send each request N times (1 = off; 2-5 when replays are on)
    parallelism: Optional[Any] = None  # calls at once for this run; default = Settings (1-16)


class WeightsBody(BaseModel):
    """A preset name, or preset='custom' with quality/cost/latency (normalised server-side)."""
    preset: str = "balanced"
    quality: Optional[float] = None
    cost: Optional[float] = None
    latency: Optional[float] = None


def _weights_from(body: Optional[WeightsBody]):
    """Preset name -> that preset's weights; 'custom' -> the given numbers (normalised)."""
    from .recommend import weights_for
    from .schemas import Weights
    if body is None:
        return weights_for("balanced", DEFAULT_SETTINGS)
    if body.preset == "custom":
        return weights_for("custom", DEFAULT_SETTINGS, custom=Weights(
            quality=body.quality or 0.0, cost=body.cost or 0.0, latency=body.latency or 0.0))
    if body.preset not in DEFAULT_SETTINGS.weight_presets:
        raise HTTPException(400, detail={"error": {"code": "invalid_weights",
                            "message": (f"unknown preset {body.preset!r}; use one of "
                                        f"{sorted(DEFAULT_SETTINGS.weight_presets)} or 'custom'")}})
    return weights_for(body.preset, DEFAULT_SETTINGS)


RunConfigBody.model_rebuild()


def _rec_payload(r: Run, weights=None) -> Optional[Dict[str, Any]]:
    """Fresh recommendation for a finished run, or None."""
    if r.status != RunStatus.DONE or not r.verdicts:
        return None
    from .recommend import run_recommendation
    rec = run_recommendation(r, DEFAULT_SETTINGS, weights)
    return rec.model_dump(mode="json") if rec else None


class OverrideBody(BaseModel):
    candidate: str
    label: CompatLabel
    note: str = ""
    replay_index: int = 0   # which replay's answer to relabel (0 when replays are off)


_UPLOADS: Dict[str, bytes] = {}
# DoS guards: cap a single upload's size and the number retained in memory.
# Overridable via env; defaults suit typical LiteLLM log exports.
_MAX_UPLOAD_BYTES = int(os.environ.get("MODELSHIFT_MAX_UPLOAD_MB", "100")) * 1024 * 1024
_MAX_UPLOADS_RETAINED = int(os.environ.get("MODELSHIFT_MAX_UPLOADS", "20"))
# Cap concurrent SSE streams so many open /stream connections can't exhaust workers.
_MAX_SSE_STREAMS = int(os.environ.get("MODELSHIFT_MAX_SSE_STREAMS", "50"))
_sse_lock = threading.Lock()
_sse_active = {"n": 0}


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/v1/models")
def list_models():
    return {
        "matrix": [e.model_dump() for e in MIGRATION_MATRIX],
        "models": {k: v.model_dump() for k, v in MODEL_CATALOG.items()},
        "default_reasoning_effort": DEFAULT_SETTINGS.default_reasoning_effort.value,
    }


def _region() -> str:
    return os.environ.get("MODELSHIFT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"


@app.get("/api/v1/models/available")
def available_models(endpoint: str = "bedrock_runtime", legacy: Optional[str] = None, refresh: bool = False):
    """Models the selected endpoint offers right now (live list; catalog models matched first).
    Never 500s on an unreachable endpoint: returns the catalog plus a `reason`."""
    from .discovery import available
    try:
        return available(endpoint, _region(), legacy_model=legacy, refresh=refresh)
    except ValueError as e:
        raise HTTPException(400, detail={"error": {"code": "invalid_endpoint", "message": str(e)}})


class ProbeBody(BaseModel):
    name: str


@app.post("/api/v1/models/probe-efforts")
def probe_model_efforts(body: ProbeBody):
    """Check which reasoning efforts a listed (non-catalog) model accepts: one tiny call per
    effort. Results are saved, so each model is checked once."""
    from .discovery import get, probe_efforts
    dm = get(body.name)
    if dm is None:
        raise HTTPException(404, detail={"error": {"code": "not_found",
                                                   "message": f"{body.name} is not a listed model"}})
    if _ORCH is None:
        raise HTTPException(409, detail={"error": {"code": "conflict", "message": "server has no transports"}})
    transport = {"litellm": _ORCH._litellm_transport, "mantle": _ORCH._mantle_transport}.get(
        dm.endpoint, _ORCH._candidate_transport)
    if transport is None:
        raise HTTPException(409, detail={"error": {"code": "conflict",
                                                   "message": f"no transport for the {dm.endpoint} endpoint"}})
    return probe_efforts(body.name, transport)


@app.post("/api/v1/uploads", status_code=201)
async def upload_file(file: UploadFile):
    # Read in chunks and enforce a hard size cap (DoS guard) instead of an
    # unbounded file.read() into memory.
    chunks: List[bytes] = []
    size = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MiB
        if not chunk:
            break
        size += len(chunk)
        if size > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail={"error": {"code": "payload_too_large",
                                "message": f"upload exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"}})
        chunks.append(chunk)
    data = b"".join(chunks)
    # Bound retained uploads: evict oldest (FIFO) so the in-memory store can't grow forever.
    while len(_UPLOADS) >= _MAX_UPLOADS_RETAINED:
        _UPLOADS.pop(next(iter(_UPLOADS)))
    upload_id = f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
    _UPLOADS[upload_id] = data
    return {"upload_id": upload_id, "filename": file.filename, "bytes": len(data)}


@app.post("/api/v1/runs", status_code=201)
def create_run(body: CreateRunBody):
    run_id = new_run_id()
    run = Run(run_id=run_id, name=body.name, description=body.description)
    _RUNS[run_id] = run
    _save_run(run)
    return {"run_id": run_id, "status": RunStatus.CREATED.value}


@app.patch("/api/v1/runs/{run_id}")
def rename_run(run_id: str, body: CreateRunBody):
    r = _get_run(run_id)
    if body.name:
        r.name = body.name
    if body.description:
        r.description = body.description
    _save_run(r)
    return {"run_id": run_id, "name": r.name}


@app.get("/api/v1/runs")
def list_runs():
    out = []
    for r in _RUNS.values():
        try:
            out.append(_run_summary(r))
        except Exception:  # noqa: BLE001 — one unreadable run must not hide all the others
            out.append({"run_id": r.run_id, "name": r.name, "status": r.status.value,
                        "created_at": r.created_at, "evaluations": None, "candidates_planned": 0,
                        "verdict_summary": [], "best_overall": None, "none_eligible": False,
                        "spend": None, "spend_unknown": False, "progress": None,
                        "error": "This run's summary could not be read."})
    return out


def _run_summary(r: Run) -> Dict[str, Any]:
    """One row of the Runs list: enough to show status, recommendation, spread and spend."""
    best, none_eligible = None, False
    if r.status == RunStatus.DONE and r.verdicts:
        try:
            rec = _rec_payload(r) or {}
            none_eligible = bool(rec.get("none_eligible"))
            if not none_eligible:
                best = next((p.get("model") for p in rec.get("profiles", [])
                             if p.get("kind") == "best_overall"), None)
        except Exception:  # noqa: BLE001 — a bad run must not break the whole list
            best = None
    known = [x.cost_estimate for x in r.results if x.cost_known]
    progress = None
    if r.status in (RunStatus.PLANNING, RunStatus.REPLAYING, RunStatus.SCORING, RunStatus.AGGREGATING):
        plan = next((e for e in r.events if e.type == "plan"), None)
        last = next((e for e in reversed(r.events) if e.type == "progress"), None)
        progress = {"done": last.data.get("done", 0) if last else 0,
                    "total": ((last.data.get("total") if last else None)
                              or (plan.data.get("estimated_calls") if plan else 0)),
                    "started_ts": plan.ts if plan else None,
                    "last_ts": last.ts if last else None,
                    "stage": r.status.value}
        es = r.exec_stats or {}
        par = r.config.parallelism if r.config else 1
        eff = list((es.get("effective") or {}).values())
        progress.update({"parallelism": par, "inflight": es.get("inflight", 0),
                         "throttles": es.get("throttles", 0),
                         "throttled": bool(eff) and min(eff) < par})
    return {
        "run_id": r.run_id, "name": r.name, "status": r.status.value,
        "created_at": r.created_at,
        "evaluations": r.ingestion.evaluable if r.ingestion else len(r.samples),
        "candidates_planned": len(r.config.candidates) if r.config else 0,
        "replays": _run_replays(r),
        "verdict_summary": [{"model": v.model, "confidence": v.migration_confidence,
                             "band": v.verdict_band.value, "eligible": v.eligible} for v in r.verdicts],
        "best_overall": best,
        "none_eligible": none_eligible,
        "spend": round(sum(known), 6) if r.results else None,
        "spend_unknown": any(not x.cost_known for x in r.results if x.status == "ok"),
        "error": (r.error or "")[:300] or None,
        "progress": progress,
    }


@app.get("/api/v1/runs/{run_id}")
def get_run(run_id: str):
    r = _get_run(run_id)
    rec = _rec_payload(r)  # also refreshes r.verdicts
    return {
        "run_id": r.run_id, "name": r.name, "status": r.status.value,
        "ingestion": r.ingestion.model_dump() if r.ingestion else None,
        "candidates": [v.model for v in r.verdicts],
        "error": r.error,
        "error_detail": r.error_detail,
        "weights": r.weights.model_dump(mode="json"),
        "recommendation": rec,
        "replays": _run_replays(r),
        "parallelism": r.config.parallelism if r.config else 1,
        "exec_stats": r.exec_stats,
    }


@app.get("/api/v1/concurrency")
def get_concurrency():
    """Server-wide calls in flight vs the cap, and each live run's in-flight / limit."""
    from .concurrency import concurrency_snapshot, server_slots
    server_slots(DEFAULT_SETTINGS.max_inflight_calls)
    return concurrency_snapshot()


def _rows_from_manual_prompts(items: List["ManualPrompt"]) -> List[Dict[str, Any]]:
    """Turn UI-entered prompt/golden pairs into Chat Completions-shaped rows.

    Each row is the same shape a LiteLLM Chat Completions log line would have, so
    the existing ingest() pipeline normalizes, evaluates, and scores them with no
    special-casing downstream. Rows with an empty prompt or empty golden are
    skipped here; ingest() would otherwise drop them as NO_PROMPT/EMPTY_RESPONSE.
    """
    rows: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        prompt = (it.prompt or "").strip()
        golden = (it.golden or "").strip()
        if not prompt or not golden:
            continue
        messages: List[Dict[str, Any]] = []
        if it.instructions and it.instructions.strip():
            messages.append({"role": "system", "content": it.instructions.strip()})
        messages.append({"role": "user", "content": prompt})
        rows.append({
            "request_id": f"manual-{i}",
            "model": it.model or "manual-input",
            "messages": messages,
            "response": {
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": golden}}],
            },
        })
    return rows


@app.post("/api/v1/runs/{run_id}/ingest", status_code=202)
def ingest_run(run_id: str, body: IngestBody):
    r = _get_run(run_id)
    r.status = RunStatus.INGESTING
    if body.manual_prompts:
        rows = _rows_from_manual_prompts(body.manual_prompts)
        if not rows:
            raise HTTPException(400, detail={"error": {"code": "conflict",
                                "message": "provide at least one prompt with a golden response"}})
        data = json.dumps(rows).encode("utf-8")
        res = ingest([(data, "manual:ui")])
    elif body.upload_id:
        data = _UPLOADS.get(body.upload_id)
        if data is None:
            raise HTTPException(400, detail={"error": {"code": "not_found", "message": "upload not found"}})
        from .sources import is_zip, load_zip_bytes, ZipArchiveError
        if is_zip(data):
            # A zip of (nested) folders of log files -> one consolidated run.
            try:
                objects, skipped = load_zip_bytes(data, f"upload:{body.upload_id}")
            except ZipArchiveError as e:
                raise HTTPException(400, detail={"error": {"code": "invalid_zip", "message": str(e)}})
            res = ingest(objects, skipped_objects=skipped)
        else:
            res = ingest([(data, f"upload:{body.upload_id}")])
    elif body.s3_uri:
        from .sources import load_s3, S3UriError, ZipArchiveError
        try:
            objects, skipped = load_s3(body.s3_uri)
        except S3UriError as e:
            raise HTTPException(400, detail={"error": {"code": "invalid_s3_uri", "message": str(e)}})
        except ZipArchiveError as e:
            raise HTTPException(400, detail={"error": {"code": "invalid_zip", "message": str(e)}})
        res = ingest(objects, skipped_objects=skipped)
    else:
        raise HTTPException(400, detail={"error": {"code": "conflict",
                                                   "message": "provide manual_prompts, upload_id, or s3_uri"}})
    r.samples = res.samples
    r.ingestion = res.report
    r.status = RunStatus.INGESTED
    _save_run(r)
    return {"status": r.status.value, "ingestion": r.ingestion.model_dump()}


@app.get("/api/v1/runs/{run_id}/ingestion")
def get_ingestion(run_id: str):
    r = _get_run(run_id)
    if r.ingestion is None:
        raise HTTPException(409, detail={"error": {"code": "conflict", "message": "not ingested yet"}})
    return r.ingestion.model_dump()


@app.get("/api/v1/runs/{run_id}/source")
def get_source_summary(run_id: str):
    """Golden / existing-baseline summary: what was ingested (source, models, counts,
    redaction) plus the original (golden) latency + token stats from the source logs."""
    r = _get_run(run_id)
    ing = r.ingestion
    # Source URI: samples carry the object they came from (upload:<file> or s3://...).
    sources = sorted({s.source.object_uri for s in r.samples if s.source and s.source.object_uri})
    lat = sorted(s.original_latency_ms for s in r.samples if s.original_latency_ms is not None)

    def _avg(vals):
        return round(sum(vals) / len(vals), 1) if vals else None

    def _pct(vals, q):
        if not vals:
            return None
        return float(vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))])

    prompt_toks = [s.usage.prompt_tokens for s in r.samples if s.usage and s.usage.prompt_tokens]
    total_toks = [s.usage.total_tokens for s in r.samples if s.usage and s.usage.total_tokens]
    return {
        "sources": sources,
        "detected_models": [dm.model_dump() for dm in ing.detected_models] if ing else [],
        "found": ing.found if ing else 0,
        "evaluable": ing.evaluable if ing else len(r.samples),
        "dropped": {k.value: v for k, v in ing.dropped.items()} if ing else {},
        "redaction_rate": ing.redaction_rate if ing else 0.0,
        "shape_mix": {k.value: v for k, v in ing.shape_mix.items()} if ing else {},
        "original_latency_ms": {
            "count": len(lat), "avg": _avg(lat),
            "p50": _pct(lat, 0.50), "p95": _pct(lat, 0.95),
            "min": (lat[0] if lat else None), "max": (lat[-1] if lat else None),
        },
        "avg_prompt_tokens": _avg(prompt_toks),
        "avg_total_tokens": _avg(total_toks),
    }


def _resolve_config(r: Run, body: RunConfigBody) -> RunConfig:
    if body.candidates:
        cands = []
        seen = set()
        for c in body.candidates:
            cc = CandidateConfig(model=c.model, reasoning_effort=c.reasoning_effort)
            if cc.label not in seen:  # same model + same effort twice = one candidate
                seen.add(cc.label)
                cands.append(cc)
    else:
        legacy = r.ingestion.detected_models[0].model if (r.ingestion and r.ingestion.detected_models) else None
        cands = default_candidates_for(legacy)
        if not cands:
            # Unmapped / newer source model: don't hard-fail — fall back to the
            # first catalog model so the run still proceeds (user can override).
            first = next(iter(MODEL_CATALOG), None)
            if first:
                cands = [CandidateConfig(model=first)]
    if not cands:
        raise HTTPException(400, detail={"error": {"code": "conflict",
                                                   "message": "no candidate models available"}})
    from .config import supported_efforts
    bad = []
    for c in cands:
        ok = supported_efforts(c.model)
        if ok is not None and c.reasoning_effort.value not in ok:
            bad.append(f"{c.model} does not support reasoning effort '{c.reasoning_effort.value}' "
                       f"(supported: {', '.join(ok)})")
    if bad:
        # Fail before replaying anything: Bedrock would reject every call with a 400.
        raise HTTPException(400, detail={"error": {"code": "unsupported_reasoning_effort",
                                                   "message": "; ".join(bad)}})
    endpoint = body.bedrock_endpoint
    # A runtime-unsupported (mantle-only) model forces the mantle endpoint.
    if body.call_path == CallPath.BEDROCK and any(
        (MODEL_CATALOG.get(c.model) and not MODEL_CATALOG[c.model].runtime_supported) for c in cands
    ):
        endpoint = "mantle"
    return RunConfig(
        candidates=cands, call_path=body.call_path, bedrock_endpoint=endpoint,
        sampling_mode=body.sampling_mode, sampling_n=body.sampling_n,
        sampling_seed=body.sampling_seed, use_judge=body.use_judge, dry_run=body.dry_run,
        judge_steering=body.judge_steering,
        filter_teams=body.filter_teams, filter_users=body.filter_users,
        monthly_volume=body.monthly_volume if (body.monthly_volume or 0) > 0 else None,
        replays=_valid_replays(body.replays),
        parallelism=_valid_parallelism(body.parallelism),
    )


def _valid_parallelism(n: Any) -> int:
    """Run's parallel calls: the API override if given (1-16), else the current Settings value."""
    if n is None:
        return DEFAULT_SETTINGS.default_parallelism
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 16:
        raise HTTPException(400, detail={"error": {
            "code": "invalid_parallelism",
            "message": "parallelism must be a whole number between 1 and 16"}})
    return n


def _valid_replays(n: Any) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 5:
        raise HTTPException(400, detail={"error": {
            "code": "invalid_replays",
            "message": "replays must be between 1 and 5 (1 = off; 2-5 sends each request that many times)"}})
    return n


@app.post("/api/v1/runs/{run_id}/plan")
def plan_run(run_id: str, body: RunConfigBody):
    r = _get_run(run_id)
    if not r.samples:
        raise HTTPException(409, detail={"error": {"code": "conflict", "message": "ingest first"}})
    r.config = _resolve_config(r, body)
    if body.weights is not None:
        r.weights = _weights_from(body.weights)
    return _ORCH.plan(r)


@app.post("/api/v1/runs/{run_id}/launch", status_code=202)
def launch_run(run_id: str, body: RunConfigBody):
    r = _get_run(run_id)
    if not r.samples:
        raise HTTPException(409, detail={"error": {"code": "conflict", "message": "ingest first"}})
    if body.name:
        r.name = body.name
    r.config = _resolve_config(r, body)
    if body.weights is not None:
        r.weights = _weights_from(body.weights)
    _CANCEL[run_id] = False
    run_async(_ORCH, r, _CANCEL, on_complete=_save_run)
    return {"status": RunStatus.REPLAYING.value}


@app.post("/api/v1/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    _get_run(run_id)
    _CANCEL[run_id] = True
    return {"status": RunStatus.CANCELLED.value}


@app.delete("/api/v1/runs/{run_id}")
def delete_run(run_id: str):
    _get_run(run_id)  # 404 if missing
    _CANCEL[run_id] = True  # stop it if a run is in flight
    _RUNS.pop(run_id, None)
    try:
        from .store import delete_run as _store_delete
        _store_delete(run_id)
    except Exception:  # noqa: BLE001 — deletion must not error the request
        pass
    return {"deleted": run_id}


@app.get("/api/v1/runs/{run_id}/results")
def get_results(run_id: str):
    r = _get_run(run_id)
    rec = _rec_payload(r)  # refreshes r.verdicts from stored results first
    # All distinct per-call errors (model + message), not just the first, so the
    # UI can show every failing candidate/reason.
    sample_errors = []
    seen_err = set()
    for res in r.results:
        if res.status == "error" and res.error:
            key = (res.model, res.error)
            if key not in seen_err:
                seen_err.add(key)
                sample_errors.append({"model": res.model, "sample_id": res.sample_id, "error": res.error})
    return {"status": r.status.value,
            "candidates": [v.model_dump() for v in r.verdicts],
            "caveats": _caveats(r),
            "error": r.error,
            "error_detail": r.error_detail,   # full traceback for a run-level failure
            "sample_error": (sample_errors[0]["error"] if sample_errors else None),  # back-compat
            "sample_errors": sample_errors,
            "weights": r.weights.model_dump(mode="json"),
            "recommendation": rec,
            "replays": _run_replays(r),
            "parallelism": r.config.parallelism if r.config else 1,
            "exec_stats": r.exec_stats,
            "exec_summary": _exec_summary(r.exec_stats)}


def _exec_summary(stats):
    from .concurrency import exec_summary
    return exec_summary(stats)


@app.get("/api/v1/runs/{run_id}/candidates/{candidate}/consistency")
def least_consistent(run_id: str, candidate: str, limit: int = 10):
    """Least consistent requests for a candidate (lowest D7 first). Empty when replays are off."""
    r = _get_run(run_id)
    cand = _resolve_candidate(r, candidate)
    rows = [v for (m, _), v in _request_consistency_map(r, {cand}).items() if m == cand and v.get("d7") is not None]
    rows.sort(key=lambda v: (v["d7"], v["sample_id"]))
    prev = {s.id: _prompt_preview(s)[:140] for s in r.samples}
    for v in rows:
        v["prompt_preview"] = prev.get(v["sample_id"], "")
    return {"candidate": cand, "replays": _run_replays(r), "items": rows[:max(1, min(limit, 100))],
            "measured": len(rows)}


@app.put("/api/v1/runs/{run_id}/weights")
def put_run_weights(run_id: str, body: WeightsBody):
    """Change the recommendation weights for a finished run. Re-ranks from stored results:
    never calls a model."""
    r = _get_run(run_id)
    if r.status != RunStatus.DONE or not r.verdicts:
        raise HTTPException(409, detail={"error": {"code": "not_ready",
                            "message": f"run is {r.status.value}; weights apply once the run is done"}})
    r.weights = _weights_from(body)
    _save_run(r)
    return {"weights": r.weights.model_dump(mode="json"), "recommendation": _rec_payload(r)}


@app.get("/api/v1/runs/{run_id}/recommendation")
def get_run_recommendation(run_id: str, preset: Optional[str] = None):
    """The run's recommendation, or a preview under another preset (not saved)."""
    r = _get_run(run_id)
    if r.status != RunStatus.DONE or not r.verdicts:
        raise HTTPException(409, detail={"error": {"code": "not_ready",
                            "message": f"run is {r.status.value}; recommendation available once the run is done"}})
    w = _weights_from(WeightsBody(preset=preset)) if preset else None
    return _rec_payload(r, w)


@app.get("/api/v1/runs/{run_id}/report.pdf")
def get_run_report_pdf(run_id: str, theme: str = "dark"):
    import io
    import re
    from .report import build_run_report, render_run_pdf
    r = _get_run(run_id)
    if theme not in ("dark", "light"):
        raise HTTPException(status_code=400, detail={"error": {"code": "invalid_theme",
                            "message": "theme must be 'dark' or 'light'"}})
    # Report is meaningful only for a finished run (done carries verdicts; failed
    # carries the error/traceback). Block in-progress states.
    if r.status not in (RunStatus.DONE, RunStatus.FAILED):
        raise HTTPException(status_code=409, detail={"error": {"code": "not_ready",
                            "message": f"run is {r.status.value}; report available once the run is done"}})
    pdf = render_run_pdf(build_run_report(r), theme=theme)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (r.name or "run")).strip("-") or "run"
    suffix = "-print" if theme == "light" else ""
    fname = f"modelshift-{slug}-{r.run_id[:8]}{suffix}.pdf"
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


def _caveats(r: Run) -> List[str]:
    out: List[str] = []
    s = DEFAULT_SETTINGS
    if r.ingestion:
        if r.ingestion.evaluable < s.small_sample_threshold:
            out.append(f"Small sample size ({r.ingestion.evaluable} evaluable) — low statistical confidence.")
        if r.ingestion.redaction_rate > s.high_redaction_threshold:
            out.append(f"High redaction rate ({r.ingestion.redaction_rate:.0%}) — limited signal.")
    return out


@app.get("/api/v1/runs/{run_id}/samples")
def get_samples(run_id: str, label: Optional[str] = None, candidate: Optional[str] = None,
                limit: int = 100, offset: int = 0, unstable: bool = False):
    r = _get_run(run_id)
    items = r.pair_scores
    if candidate:
        candidate = _resolve_candidate(r, candidate)
        items = [p for p in items if p.model == candidate]
    if label:
        items = [p for p in items if p.label.value == label]
    # Replays: attach each request's D7 so the list can show / filter unstable requests.
    rc = _request_consistency_map(r, {p.model for p in items})
    if unstable:
        items = [p for p in items if (rc.get((p.model, p.sample_id)) or {}).get("unstable")]
    total = len(items)
    page = items[offset: offset + limit]
    out = []
    for p in page:
        d = p.model_dump()
        c = rc.get((p.model, p.sample_id))
        d["replays"] = _run_replays(r)
        d["consistency"] = c
        out.append(d)
    return {"items": out, "total": total}


def _run_replays(r: Run) -> int:
    return max(1, int(getattr(r.config, "replays", 1) or 1)) if r.config else 1


def _request_consistency_map(r: Run, models) -> Dict[tuple, Dict[str, Any]]:
    """(model, sample_id) -> D7 summary for runs with replays; {} when replays are off."""
    if _run_replays(r) < 2:
        return {}
    from .consistency import requests_for
    out: Dict[tuple, Dict[str, Any]] = {}
    for m in models:
        for rc in requests_for(m, r.results, r.pair_scores, DEFAULT_SETTINGS.instability_threshold):
            out[(m, rc.sample_id)] = rc.model_dump(mode="json")
    return out


@app.get("/api/v1/runs/{run_id}/samples/{sample_id}")
def get_sample_detail(run_id: str, sample_id: str, candidate: str, replay: int = 0):
    """Full drill-down for one (sample, candidate): original request/response/model/
    latency + the candidate's rendered request, response, and latency + the scores.
    With replays, `replay` picks which answer is shown; `answers` lists all of them."""
    r = _get_run(run_id)
    sample = next((s for s in r.samples if s.id == sample_id), None)
    if sample is None:
        raise HTTPException(404, detail={"error": {"code": "not_found", "message": "sample not found"}})
    candidate = _resolve_candidate(r, candidate)
    mine = [x for x in r.results if x.sample_id == sample_id and x.model == candidate]
    my_scores = [p for p in r.pair_scores if p.sample_id == sample_id and p.model == candidate]
    result = next((x for x in mine if x.replay_index == replay), None) or (mine[0] if mine else None)
    shown = result.replay_index if result is not None else replay
    score = next((p for p in my_scores if p.replay_index == shown), None)

    # The candidate saw the same request as the golden (that's the point of a replay).
    original = {
        "model": sample.legacy_model,
        "instructions": sample.request.instructions,
        "messages": [m.model_dump() for m in sample.request.input_messages],
        "response": {"kind": sample.golden.kind.value,
                     "text": sample.golden.text,
                     "tool_calls": [tc.model_dump() for tc in (sample.golden.tool_calls or [])]},
        "latency_ms": sample.original_latency_ms,
        "usage": sample.usage.model_dump(),
    }
    cand = None
    if result is not None:
        cand = {
            "model": result.model,
            "reasoning_effort": result.settings.reasoning_effort.value,
            "dropped_params": result.settings.dropped_params,
            "response": {"kind": result.output.kind.value,
                         "text": result.output.text,
                         "tool_calls": [tc.model_dump() for tc in (result.output.tool_calls or [])]},
            "latency_ms": result.latency_ms,
            "usage": result.usage.model_dump(),
            "cost_estimate": result.cost_estimate,
            "status": result.status,
            "error": result.error,
        }
    answers = []
    if _run_replays(r) > 1:
        sc_by = {p.replay_index: p for p in my_scores}
        for x in sorted(mine, key=lambda x: x.replay_index):
            p = sc_by.get(x.replay_index)
            answers.append({
                "replay_index": x.replay_index, "status": x.status, "error": x.error,
                "text": x.output.text if x.output else None,
                "tool_calls": [tc.model_dump() for tc in ((x.output.tool_calls if x.output else None) or [])],
                "latency_ms": x.latency_ms, "cost_estimate": x.cost_estimate,
                "label": ((p.override_label if p.overridden and p.override_label else p.label).value
                          if p else None),
                "hard_break": p.hard_break if p else None,
            })
    consistency = _request_consistency_map(r, {candidate}).get((candidate, sample_id))
    return {"sample_id": sample_id, "candidate": candidate,
            "original": original, "candidate_result": cand,
            "score": score.model_dump() if score else None,
            "replay": shown, "replays": _run_replays(r),
            "answers": answers, "consistency": consistency}


def _prompt_preview(sample) -> str:
    msgs = sample.request.input_messages
    return msgs[-1].content if msgs else ""


@app.get("/api/v1/runs/{run_id}/transcript")
def get_transcript(run_id: str, candidate: str, limit: int = 25, offset: int = 0):
    """Full request/response data for EVERY replayed row of a candidate model,
    inline and paginated — original request + original response + candidate
    response (including errored candidate calls), with the per-row label/latency."""
    r = _get_run(run_id)
    candidate = _resolve_candidate(r, candidate)
    results_by_id = {(res.sample_id, res.replay_index): res for res in r.results if res.model == candidate}
    scores_by_id = {(p.sample_id, p.replay_index): p for p in r.pair_scores if p.model == candidate}
    # Every evaluable sample that was replayed against this candidate, one row per replay.
    keys = set(results_by_id) | set(scores_by_id)
    order = {s.id: i for i, s in enumerate(r.samples)}
    ids = sorted((k for k in keys if k[0] in order), key=lambda k: (order[k[0]], k[1]))
    total = len(ids)
    page_ids = ids[offset: offset + limit]
    samples_by_id = {s.id: s for s in r.samples}

    items = []
    for key in page_ids:
        sid, rep = key
        s = samples_by_id[sid]
        res = results_by_id.get(key)
        sc = scores_by_id.get(key)
        items.append({
            "sample_id": sid,
            "replay_index": rep,
            "original_model": s.legacy_model,
            "request": {"instructions": s.request.instructions,
                        "messages": [m.model_dump() for m in s.request.input_messages]},
            "original_response": {"kind": s.golden.kind.value, "text": s.golden.text,
                                  "tool_calls": [tc.model_dump() for tc in (s.golden.tool_calls or [])]},
            "original_latency_ms": s.original_latency_ms,
            "candidate_model": candidate,
            "candidate_response": (
                {"kind": res.output.kind.value, "text": res.output.text,
                 "tool_calls": [tc.model_dump() for tc in (res.output.tool_calls or [])]}
                if res and res.status == "ok" else None),
            "candidate_latency_ms": res.latency_ms if res else None,
            "candidate_status": res.status if res else "missing",
            "candidate_error": res.error if res else None,
            "label": (sc.override_label.value if (sc and sc.overridden and sc.override_label)
                      else (sc.label.value if sc else None)),
        })
    return {"items": items, "total": total, "limit": limit, "offset": offset, "candidate": candidate}


@app.patch("/api/v1/runs/{run_id}/samples/{sample_id}")
def override_sample(run_id: str, sample_id: str, body: OverrideBody):
    r = _get_run(run_id)
    cand = _resolve_candidate(r, body.candidate)
    for p in r.pair_scores:
        if p.sample_id == sample_id and p.model == cand and p.replay_index == body.replay_index:
            p.overridden = True
            p.override_label = body.label
            p.override_note = body.note
            _save_run(r)
            return {"updated": True}
    raise HTTPException(404, detail={"error": {"code": "not_found", "message": "pair not found"}})


def _resolve_candidate(r: Run, name: Optional[str]) -> Optional[str]:
    """Map a ?candidate= value to the run's candidate key.

    Candidates are keyed 'model@effort'. A bare model name ('gpt-5.6-luna') still works when
    exactly one candidate in the run uses that model, so existing links and scripts keep working.
    """
    if not name:
        return name
    keys = {x.model for x in r.results} | {p.model for p in r.pair_scores} | {v.model for v in r.verdicts}
    if name in keys:
        return name
    matches = [c.label for c in (r.config.candidates if r.config else []) if c.model == name]
    return matches[0] if len(matches) == 1 else name


def _candidate_effort(r: Run, model: str) -> ReasoningEffort:
    for c in (r.config.candidates if r.config else []):
        if c.matches(model):
            return c.reasoning_effort
    return ReasoningEffort.MEDIUM


@app.get("/api/v1/runs/{run_id}/remediations")
def get_remediations(run_id: str, candidate: Optional[str] = None):
    from .remediate import concretize
    r = _get_run(run_id)
    # Runs recorded before suggestions were made concrete have no before/after yet:
    # fill them in once from the run's own samples and persist.
    changed = False
    for x in r.remediations:
        if not x.change.after and x.change.type in ("prompt_edit", "format_schema", "reasoning_effort"):
            concretize(x, r.samples, _candidate_effort(r, x.candidate))
            changed = True
    if changed:
        _save_run(r)
    rems = r.remediations
    if candidate:
        candidate = _resolve_candidate(r, candidate)
        rems = [x for x in rems if x.candidate == candidate]
    return [x.model_dump() for x in rems]


class RemediationEditBody(BaseModel):
    type: Optional[str] = None
    after: Optional[str] = None
    placement: Optional[str] = None


@app.patch("/api/v1/runs/{run_id}/remediations/{remediation_id}")
def edit_remediation(run_id: str, remediation_id: str, body: RemediationEditBody):
    """Edit a suggestion's change before applying it (instruction text, placement,
    target effort, or switch between instruction and effort changes)."""
    from .remediate import validate_change
    r = _get_run(run_id)
    rem = next((x for x in r.remediations if x.remediation_id == remediation_id), None)
    if rem is None:
        raise HTTPException(404, detail={"error": {"code": "not_found", "message": "remediation not found"}})
    ch = rem.change.model_copy()
    if body.type is not None and body.type != ch.type:
        ch.type = body.type
        if ch.type == "reasoning_effort":
            ch.before = _candidate_effort(r, rem.candidate).value
        else:
            ch.before = None
    if body.after is not None:
        ch.after = body.after
    if body.placement is not None:
        ch.placement = body.placement
    try:
        rem.change = validate_change(ch, rem.candidate)
    except ValueError as e:
        raise HTTPException(400, detail={"error": {"code": "invalid_change", "message": str(e)}})
    # An edited change invalidates any earlier re-test result.
    rem.applied, rem.result = False, None
    _save_run(r)
    return rem.model_dump()


# ---- remediation re-test ------------------------------------------------- #
# One re-test at a time per server: a re-test makes real model calls and shares
# the run's transports, so overlapping ones would only race for the same quota.
_RETEST_LOCK = threading.Lock()
_RETEST_ACTIVE: Dict[str, str] = {}   # "active" -> "run_id/remediation_id"
_RETEST_CANCEL: Dict[str, bool] = {}  # remediation_id -> cancel requested


def _get_remediation(r: Run, remediation_id: str):
    rem = next((x for x in r.remediations if x.remediation_id == remediation_id), None)
    if rem is None:
        raise HTTPException(404, detail={"error": {"code": "not_found", "message": "remediation not found"}})
    return rem


def _retest_change(r: Run, rem, body: Optional["RemediationEditBody"]):
    """The change to apply: the saved one, optionally overridden by the request body."""
    from .remediate import concretize, validate_change
    if not rem.change.after:
        concretize(rem, r.samples, _candidate_effort(r, rem.candidate))
    ch = rem.change.model_copy()
    if body is not None:
        if body.type is not None and body.type != ch.type:
            ch.type = body.type
            ch.before = _candidate_effort(r, rem.candidate).value if ch.type == "reasoning_effort" else None
        if body.after is not None:
            ch.after = body.after
        if body.placement is not None:
            ch.placement = body.placement
    try:
        return validate_change(ch, rem.candidate)
    except ValueError as e:
        raise HTTPException(400, detail={"error": {"code": "invalid_change", "message": str(e)}})


class RetestBody(RemediationEditBody):
    # Compatible evaluations to re-run as a regression check: default 20, 0 = off, -1 = all.
    regression_sample: Optional[int] = None


def _regression_sample(body: Optional[RetestBody]) -> int:
    from .remediate import DEFAULT_REGRESSION_SAMPLE
    n = body.regression_sample if (body is not None and body.regression_sample is not None) \
        else DEFAULT_REGRESSION_SAMPLE
    if n < -1:
        raise HTTPException(400, detail={"error": {"code": "invalid_regression_sample",
                            "message": "regression_sample must be -1 (all), 0 (off) or a positive count"}})
    return n


@app.post("/api/v1/runs/{run_id}/remediations/{remediation_id}/retest/plan")
def plan_retest(run_id: str, remediation_id: str, body: Optional[RetestBody] = None):
    """Estimate (calls, cost, duration) before launching a re-test. No model calls."""
    r = _get_run(run_id)
    rem = _get_remediation(r, remediation_id)
    return _ORCH.plan_retest(r, rem, _retest_change(r, rem, body), regression_sample=_regression_sample(body))


@app.post("/api/v1/runs/{run_id}/remediations/{remediation_id}/retest", status_code=202)
def launch_retest(run_id: str, remediation_id: str, body: Optional[RetestBody] = None):
    """Apply the (optionally edited) change and re-test the cluster in the background,
    plus a regression check on Compatible evaluations.
    Poll GET .../remediations/{rid} for progress (result.done / result.total)."""
    r = _get_run(run_id)
    rem = _get_remediation(r, remediation_id)
    if r.status != RunStatus.DONE:
        raise HTTPException(409, detail={"error": {"code": "not_ready",
                            "message": f"run is {r.status.value}; re-test once the run is done"}})
    change = _retest_change(r, rem, body)
    n_guard = _regression_sample(body)
    key = f"{run_id}/{remediation_id}"
    with _RETEST_LOCK:
        busy = _RETEST_ACTIVE.get("active")
        if busy:
            raise HTTPException(409, detail={"error": {"code": "retest_in_progress",
                                "message": f"another re-test is running ({busy}); wait for it or cancel it"}})
        _RETEST_ACTIVE["active"] = key
    rem.change = change  # the applied change becomes the saved one
    _RETEST_CANCEL[remediation_id] = False

    def _work():
        try:
            _ORCH.retest_remediation(r, rem, cancel=lambda: _RETEST_CANCEL.get(remediation_id, False),
                                     regression_sample=n_guard)
        finally:
            _save_run(r)
            with _RETEST_LOCK:
                if _RETEST_ACTIVE.get("active") == key:
                    _RETEST_ACTIVE.pop("active", None)
            _RETEST_CANCEL.pop(remediation_id, None)

    plan = _ORCH.plan_retest(r, rem, change, regression_sample=n_guard)
    threading.Thread(target=_work, daemon=True).start()
    return {"status": "running", "remediation_id": remediation_id, "total": plan["calls"],
            "target_calls": plan["target_calls"], "regression_calls": plan["regression_calls"]}


@app.get("/api/v1/runs/{run_id}/remediations/{remediation_id}")
def get_remediation(run_id: str, remediation_id: str):
    r = _get_run(run_id)
    return _get_remediation(r, remediation_id).model_dump()


@app.post("/api/v1/runs/{run_id}/remediations/{remediation_id}/retest/cancel")
def cancel_retest(run_id: str, remediation_id: str):
    r = _get_run(run_id)
    rem = _get_remediation(r, remediation_id)
    if not (rem.result and rem.result.status == "running"):
        raise HTTPException(409, detail={"error": {"code": "not_running", "message": "no re-test is running"}})
    _RETEST_CANCEL[remediation_id] = True
    return {"status": "cancelling"}


@app.get("/api/v1/runs/{run_id}/stream")
def stream_progress(run_id: str):
    r = _get_run(run_id)

    with _sse_lock:
        if _sse_active["n"] >= _MAX_SSE_STREAMS:
            raise HTTPException(429, detail={"error": {"code": "too_many_streams",
                                "message": "too many concurrent progress streams; retry shortly"}})
        _sse_active["n"] += 1

    def _gen():
        # Replay any buffered events, then stream new ones until terminal.
        idx = 0
        terminal = {RunStatus.DONE, RunStatus.FAILED, RunStatus.CANCELLED}
        try:
            while True:
                while idx < len(r.events):
                    ev = r.events[idx]
                    idx += 1
                    yield f"event: {ev.type}\ndata: {json.dumps({**ev.data, 'ts': ev.ts})}\n\n"
                if r.status in terminal and idx >= len(r.events):
                    yield f"event: end\ndata: {json.dumps({'status': r.status.value})}\n\n"
                    return
                import time as _t
                _t.sleep(0.1)
        finally:
            with _sse_lock:
                _sse_active["n"] = max(0, _sse_active["n"] - 1)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/api/v1/settings")
def get_settings():
    s = DEFAULT_SETTINGS
    # Never return secret values — only the tunable, non-secret surface.
    return {"default_reasoning_effort": s.default_reasoning_effort.value,
            "judge_model": s.judge_model,
            "judge_litellm_model": s.judge_litellm_model,
            "price_table": {k: v.model_dump() for k, v in s.price_table.items()},
            "parallelism": s.default_parallelism,
            "concurrency": {"parallelism": s.default_parallelism, "max_parallelism": 16,
                            "adaptive_concurrency": s.adaptive_concurrency,
                            "max_inflight_calls": s.max_inflight_calls,
                            "fail_fast_error_share": s.fail_fast_error_share},
            "small_sample_threshold": s.small_sample_threshold,
            "high_redaction_threshold": s.high_redaction_threshold,
            "litellm_base": s.litellm_base,
            "recommendation": {k: getattr(s, k) for k in RECOMMENDATION_SETTING_KEYS},
            "latency_disclaimer": s.latency_disclaimer,
            # Never return the key itself — only whether one is available. Cheap
            # presence check (no live Secrets Manager fetch on every settings read):
            # env var set, OR an SM secret ARN configured, OR a persisted key.
            "litellm_key_set": bool(
                os.environ.get("MODELSHIFT_LITELLM_KEY")
                or os.environ.get("MODELSHIFT_LITELLM_KEY_SECRET_ARN")
                or s.litellm_key)}


class PriceBody(BaseModel):
    input_per_1k: float
    output_per_1k: float


class SettingsBody(BaseModel):
    default_reasoning_effort: Optional[ReasoningEffort] = None
    judge_model: Optional[str] = None
    judge_litellm_model: Optional[str] = None
    parallelism: Optional[Any] = None     # validated strictly in update_settings (400 on non-integers)
    adaptive_concurrency: Optional[bool] = None
    max_inflight_calls: Optional[Any] = None
    fail_fast_error_share: Optional[Any] = None   # null switches fail-fast off (when sent explicitly)
    small_sample_threshold: Optional[int] = None
    high_redaction_threshold: Optional[float] = None
    price_table: Optional[Dict[str, PriceBody]] = None
    litellm_base: Optional[str] = None
    # Recommendation thresholds / presets (any subset; validated in update_settings).
    recommendation: Optional[Dict[str, Any]] = None
    # Secret: send a new value to set/replace it; omit (null) to keep the existing
    # key; send an empty string "" to explicitly clear it.
    litellm_key: Optional[str] = None


@app.put("/api/v1/settings")
def update_settings(body: SettingsBody):
    """Update the live tunable settings. Persists for the process. The LiteLLM
    key is a secret: it is persisted and applied to the live transport but never
    returned by GET /settings."""
    s = DEFAULT_SETTINGS
    if body.default_reasoning_effort is not None:
        s.default_reasoning_effort = body.default_reasoning_effort
    if body.judge_model is not None:
        s.judge_model = body.judge_model
    litellm_changed = False
    if body.judge_litellm_model is not None:
        s.judge_litellm_model = body.judge_litellm_model.strip()
        litellm_changed = True
    conc: Dict[str, Any] = {}
    for k in ("parallelism", "max_inflight_calls"):
        v = getattr(body, k)
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, int):
                raise HTTPException(400, detail={"error": {
                    "code": "invalid_parallelism", "message": f"{k} must be a whole number"}})
            conc["default_parallelism" if k == "parallelism" else k] = v
    if body.adaptive_concurrency is not None:
        conc["adaptive_concurrency"] = body.adaptive_concurrency
    if "fail_fast_error_share" in body.model_fields_set:
        v = body.fail_fast_error_share
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise HTTPException(400, detail={"error": {
                "code": "invalid_setting",
                "message": "fail_fast_error_share must be a number between 0 and 1, or null"}})
        conc["fail_fast_error_share"] = v
    if conc:
        from .config import apply_concurrency_settings
        from .concurrency import server_slots
        apply_concurrency_settings(s, conc)
        server_slots(s.max_inflight_calls)  # resize the live cap
    if body.small_sample_threshold is not None:
        s.small_sample_threshold = max(0, body.small_sample_threshold)
    if body.high_redaction_threshold is not None:
        s.high_redaction_threshold = min(max(body.high_redaction_threshold, 0.0), 1.0)
    if body.price_table is not None:
        from .config import ModelPrice
        for model, p in body.price_table.items():
            prev = s.price_table.get(model)
            s.price_table[model] = ModelPrice(input_per_1k=p.input_per_1k, output_per_1k=p.output_per_1k,
                                              cache_read_per_1k=prev.cache_read_per_1k if prev else None)
    if body.litellm_base is not None:
        s.litellm_base = body.litellm_base.strip()
        litellm_changed = True
    if body.recommendation:
        from .config import _apply_recommendation_settings
        _apply_recommendation_settings(s, body.recommendation)
    if body.litellm_key is not None:  # explicit set (may be "" to clear)
        s.litellm_key = body.litellm_key
        litellm_changed = True
    from .config import save_settings
    save_settings(s)
    # Rebuild the live LiteLLM transport so UI changes take effect without a restart.
    if litellm_changed and _ORCH is not None and s.litellm_base:
        try:
            from .adapters import make_litellm_transport
            from .judge import make_litellm_judge_transport
            from .config import resolve_litellm_key
            key = resolve_litellm_key(s)
            _ORCH._litellm_transport = make_litellm_transport(s.litellm_base, key)
            _ORCH._litellm_judge_transport = make_litellm_judge_transport(
                s.litellm_base, key, s.judge_litellm_model)
        except Exception:  # noqa: BLE001 — never let a rebuild failure break the settings write
            pass
    return get_settings()


@app.get("/api/v1/litellm/models")
def litellm_models():
    """Discover the model-group aliases the configured LiteLLM proxy serves, for
    dynamic candidate selection on the LiteLLM path. Degrades gracefully: returns
    an empty list + a reason string if the proxy is unreachable/misconfigured."""
    s = DEFAULT_SETTINGS
    base = s.litellm_base
    if not base:
        return {"base": "", "models": [], "reason": "no LiteLLM base URL configured"}
    from .adapters import discover_litellm_models
    from .config import resolve_litellm_key
    try:
        key = resolve_litellm_key(s)
        models = discover_litellm_models(base, key)
        return {"base": base, "models": models, "reason": None}
    except Exception as e:  # noqa: BLE001 — never 500 the settings/config page
        return {"base": base, "models": [], "reason": f"could not reach proxy: {e}"}


@app.get("/api/v1/health")
@app.get("/api/v1/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "runs": len(_RUNS), "build_id": BUILD_ID}


@app.get("/api/v1/version")
def get_version() -> Dict[str, Any]:
    return version_info()


# ---- static SPA ----------------------------------------------------------- #
from fastapi.staticfiles import StaticFiles  # noqa: E402

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.isdir(_WEB_DIR):
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
