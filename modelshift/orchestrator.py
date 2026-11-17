"""Run orchestrator + job lifecycle (Doc 5 §2, §3.1).

Wires ingestion -> replay -> score -> judge -> aggregate -> remediation.
Runs are async jobs with a status state machine and progress events. The
candidate/judge transports are injected so the orchestrator is testable
offline (no live Bedrock/LiteLLM).
"""
from __future__ import annotations

import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .adapters import (BedrockResponsesAdapter, CandidateAdapter, LiteLLMResponsesAdapter, Transport,
                       _is_throttle_error)
from .concurrency import AdaptiveLimiter, CallPool, ExecStats, FailFast, RunAbort, server_slots
from .aggregate import aggregate, recommend, retest, side_by_side
from .config import (DEFAULT_SETTINGS, MODEL_CATALOG, Settings, base_model, candidate_label,
                     candidates_for, estimate_cost)
from .judge import Judge, JudgeTransport, deterministic_judge
from .remediate import DEFAULT_REGRESSION_SAMPLE, apply_change, regression_guard_ids
from .scoring import score_pair
from .schemas import (
    CallPath,
    CandidateResult,
    CandidateSettings,
    CandidateVerdict,
    IngestionReport,
    NormalizedSample,
    PairScore,
    ReasoningEffort,
    Remediation,
    RemediationChange,
    RemediationResult,
    Weights,
)


def _limiter_key(model: str, adapter: CandidateAdapter) -> str:
    """Throttling is tracked per endpoint: Claude goes through Converse, GPT through /openai/v1."""
    entry = MODEL_CATALOG.get(base_model(model))
    if entry and entry.family == "anthropic":
        return "bedrock-converse"
    if not entry and adapter.call_path != CallPath.LITELLM:
        from .discovery import get as _discovered
        dm = _discovered(model)
        if dm is not None and dm.family == "converse":
            return "bedrock-converse"
    return _effective_endpoint(adapter)


def _effective_endpoint(adapter: CandidateAdapter) -> str:
    """Human-readable label of the endpoint an adapter actually calls, for the UI/results.
    "litellm" | "bedrock-runtime" | "bedrock-mantle"."""
    if isinstance(adapter, LiteLLMResponsesAdapter):
        return "litellm"
    if isinstance(adapter, BedrockResponsesAdapter):
        return "bedrock-mantle" if getattr(adapter, "_endpoint", "runtime") == "mantle" else "bedrock-runtime"
    return "bedrock-runtime"


class RunStatus(str, Enum):
    CREATED = "created"
    INGESTING = "ingesting"
    INGESTED = "ingested"
    PLANNING = "planning"
    REPLAYING = "replaying"
    SCORING = "scoring"
    AGGREGATING = "aggregating"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CandidateConfig:
    model: str
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM

    @property
    def label(self) -> str:
        """Run-level identity: 'gpt-5.6-luna@low'. Same model at two efforts = two candidates."""
        return candidate_label(self.model, self.reasoning_effort)

    def matches(self, name: str) -> bool:
        """True for this candidate's label, or its bare model name (runs saved before labels)."""
        return name == self.label or name == self.model


@dataclass
class RunConfig:
    candidates: List[CandidateConfig]
    call_path: CallPath = CallPath.BEDROCK
    bedrock_endpoint: str = "runtime"   # runtime | mantle (BedrockEndpoint value)
    sampling_mode: str = "all"       # all | first_n | random_n
    sampling_n: int = 200
    sampling_seed: int = 42
    use_judge: bool = True
    judge_steering: Optional[str] = None   # custom evaluation guidance for the LLM judge
    filter_teams: Optional[List[str]] = None  # only evaluate rows tagged with these teams
    filter_users: Optional[List[str]] = None  # only evaluate rows tagged with these users
    # Calls sent at once. Copied from Settings at launch (parallel-calls R1.3); 1 = sequential.
    # Default 1 so runs saved before concurrency existed read as what they were.
    parallelism: int = 1
    dry_run: bool = False
    # Optional expected production requests/month -> projected monthly cost per candidate.
    monthly_volume: Optional[int] = None
    # Request replays: how many times each request is sent to each candidate (1 = off; 2-5).
    # Every replayed answer is scored like a normal call; D7 Consistency compares them.
    replays: int = 1


@dataclass
class ProgressEvent:
    type: str
    data: Dict[str, Any]
    ts: float = field(default_factory=time.time)


@dataclass
class Run:
    run_id: str
    name: str
    description: str = ""
    status: RunStatus = RunStatus.CREATED
    config: Optional[RunConfig] = None
    samples: List[NormalizedSample] = field(default_factory=list)
    ingestion: Optional[IngestionReport] = None
    results: List[CandidateResult] = field(default_factory=list)
    pair_scores: List[PairScore] = field(default_factory=list)
    verdicts: List[CandidateVerdict] = field(default_factory=list)
    remediations: List[Remediation] = field(default_factory=list)
    events: List[ProgressEvent] = field(default_factory=list)
    error: Optional[str] = None
    error_detail: Optional[str] = None  # full traceback for a run-level failure
    # 2 = latency excludes retry back-off, cache hits flagged, unknown prices tracked.
    # Runs saved by older builds load as 1 (their latency may include retry wait time).
    metrics_version: int = 2
    # Recommendation weights chosen for this run (changeable after the run; no model calls).
    weights: Weights = field(default_factory=Weights)
    # When the run was created (epoch seconds). Older saved runs fall back to their first
    # event, or the saved file's modification time (see store._run_from_dict).
    created_at: float = field(default_factory=time.time)
    # Execution counters (parallel calls, throttles, time slowed, wall time); None for old runs.
    exec_stats: Optional[Dict[str, Any]] = None

    def emit(self, etype: str, **data: Any) -> None:
        self.events.append(ProgressEvent(type=etype, data=data))


def default_candidates_for(legacy_model: Optional[str],
                           effort: ReasoningEffort = ReasoningEffort.MEDIUM) -> List[CandidateConfig]:
    entry = candidates_for(legacy_model)
    if not entry:
        return []
    return [CandidateConfig(model=m, reasoning_effort=effort) for m in entry.candidates]


def _sample_subset(samples: List[NormalizedSample], cfg: RunConfig) -> List[NormalizedSample]:
    # Team/user targeting: keep only rows tagged with a selected team or user.
    teams = set(cfg.filter_teams or [])
    users = set(cfg.filter_users or [])
    if teams or users:
        samples = [s for s in samples
                   if (teams and str(s.tags.get("team")) in teams)
                   or (users and str(s.tags.get("user")) in users)]
    if cfg.sampling_mode == "first_n":
        return samples[: cfg.sampling_n]
    if cfg.sampling_mode == "random_n":
        rng = random.Random(cfg.sampling_seed)
        return rng.sample(samples, min(cfg.sampling_n, len(samples)))
    return samples


class Orchestrator:
    """Runs the pipeline for a Run. Transports injected for testability."""

    def __init__(self,
                 candidate_transport: Optional[Transport] = None,
                 judge_transport: Optional[JudgeTransport] = None,
                 mantle_transport: Optional[Transport] = None,
                 converse_transport: Optional[Transport] = None,
                 litellm_transport: Optional[Transport] = None,
                 litellm_judge_transport: Optional[JudgeTransport] = None,
                 settings: Settings = DEFAULT_SETTINGS):
        self._candidate_transport = candidate_transport
        self._judge_transport = judge_transport
        self._mantle_transport = mantle_transport
        self._converse_transport = converse_transport
        self._litellm_transport = litellm_transport
        self._litellm_judge_transport = litellm_judge_transport
        self._settings = settings

    # ---- planning / dry-run cost preview (Doc 5 §2.2) --------------------- #
    def plan(self, run: Run) -> Dict[str, Any]:
        cfg = run.config
        subset = _sample_subset(run.samples, cfg)
        per_candidate = []
        total = 0.0
        for c in cfg.candidates:
            calls = len(subset)
            # estimate output tokens ~ prompt tokens as a rough preview
            est = 0.0
            price_known = True
            for s in subset:
                pt = s.usage.prompt_tokens or _approx_tokens(s)
                c_est = estimate_cost(c.model, pt, pt, 0)
                if c_est is None:
                    price_known = False
                    continue
                est += c_est
            per_candidate.append({"model": c.label, "calls": calls, "est_cost": round(est, 4),
                                  "price_known": price_known})
            total += est
        # Replays: every request is sent N times, so calls and cost scale by N.
        n_rep = max(1, int(getattr(cfg, "replays", 1) or 1))
        base_calls = len(subset) * len(cfg.candidates)
        for pc in per_candidate:
            pc["calls"] *= n_rep
            pc["est_cost"] = round(pc["est_cost"] * n_rep, 4)
        calls = base_calls * n_rep
        ceiling = getattr(self._settings, "replay_call_ceiling", 2000)
        return {"estimated_calls": calls,
                "est_cost": round(total * n_rep, 4),
                "per_candidate": per_candidate,
                "replays": n_rep,
                "requests": len(subset),
                "calls_without_replays": base_calls,
                "est_cost_without_replays": round(total, 4),
                "judge_calls": calls if cfg.use_judge else 0,
                "call_ceiling": ceiling,
                "over_ceiling": calls > ceiling,
                # Rough duration: ~4 s per call (+1.5 s judge), N calls at a time.
                "parallelism": max(1, int(getattr(cfg, "parallelism", 1) or 1)),
                "est_seconds": round(-(-calls // max(1, int(getattr(cfg, "parallelism", 1) or 1)))
                                     * (4.0 + (1.5 if cfg.use_judge else 0.0)))}

    def _adapter_for(self, cfg: RunConfig, model: str) -> CandidateAdapter:
        entry = MODEL_CATALOG.get(base_model(model))
        from .discovery import get as _discovered
        dm = None if entry else _discovered(model)
        family = entry.family if entry else (dm.family if dm else "openai")
        # Anthropic (Claude) and every other non-OpenAI provider listed from Bedrock use the
        # Converse transport (on the LiteLLM path the proxy handles the provider instead).
        if family == "anthropic" or (family == "converse" and cfg.call_path != CallPath.LITELLM):
            transport = self._converse_transport
            if transport is None:
                raise RuntimeError(
                    f"{model} is an Anthropic (Converse) model but no Converse transport is "
                    "configured for this run. It uses Bedrock Converse, not the OpenAI /openai/v1 path."
                )
            return BedrockResponsesAdapter(transport=transport, endpoint="runtime")
        if cfg.call_path == CallPath.LITELLM:
            transport = self._litellm_transport or self._candidate_transport
            if transport is None:
                raise RuntimeError(
                    "No LiteLLM transport configured for a live run. Start the server with "
                    "--litellm-base <url> (defaults to the local proxy in live mode)."
                )
            return LiteLLMResponsesAdapter(transport=transport)
        # Endpoint is per-MODEL: a runtime-capable model (Luna/Terra/Sol) always uses
        # runtime, even if the run also contains a mantle-only model. Only a
        # runtime-unsupported model — or an explicit run-wide mantle choice for THIS
        # model — routes to mantle.
        runtime_ok = entry.runtime_supported if entry else True
        use_mantle = (not runtime_ok) or (cfg.bedrock_endpoint == "mantle" and not runtime_ok)
        if dm is not None:
            # A model listed live goes to the endpoint it was listed from.
            use_mantle = dm.endpoint == "mantle"
        if use_mantle:
            transport = self._mantle_transport
            if transport is None:
                raise RuntimeError(
                    f"{model} requires the Bedrock Mantle /openai/v1 endpoint, which needs a "
                    "Bedrock API key. Start the server with --bedrock-key <key> (or pick a "
                    "runtime model like GPT-5.6 Luna/Terra/Sol)."
                )
            return BedrockResponsesAdapter(transport=transport, endpoint="mantle")
        if self._candidate_transport is None:
            raise RuntimeError("No candidate transport configured for a live run.")
        return BedrockResponsesAdapter(transport=self._candidate_transport, endpoint="runtime")

    def _judge(self, steering: Optional[str] = None,
               call_path: CallPath = CallPath.BEDROCK) -> Judge:
        # When the run routes candidates through LiteLLM, route the judge through
        # the same proxy too (if a LiteLLM judge transport is wired), so ALL model
        # traffic for the run goes through the gateway. Otherwise judge direct.
        if call_path == CallPath.LITELLM and self._litellm_judge_transport is not None:
            transport: JudgeTransport = self._litellm_judge_transport
        else:
            transport = self._judge_transport or deterministic_judge()
        return Judge(transport=transport, rubric=self._settings.rubric, steering=steering)

    # ---- remediation re-test (Doc 3 §6.3) ----------------------------------- #
    def _retest_targets(self, run: Run, rem: Remediation) -> List[NormalizedSample]:
        ids = set(rem.target.sample_ids)
        return [s for s in run.samples if s.id in ids]

    def _retest_guard(self, run: Run, rem: Remediation, regression_sample: int) -> List[NormalizedSample]:
        ids = set(regression_guard_ids(run.pair_scores, rem.candidate, rem.target.sample_ids,
                                       [s.id for s in run.samples], regression_sample,
                                       seed=rem.remediation_id))
        return [s for s in run.samples if s.id in ids]

    def _candidate_cfg(self, run: Run, model: str) -> "CandidateConfig":
        for c in (run.config.candidates if run.config else []):
            if c.matches(model):
                return c
        return CandidateConfig(model=base_model(model))

    def plan_retest(self, run: Run, rem: Remediation,
                    change: Optional[RemediationChange] = None,
                    regression_sample: int = DEFAULT_REGRESSION_SAMPLE) -> Dict[str, Any]:
        """Calls, estimated cost and duration for re-testing one remediation cluster,
        including the Compatible evaluations re-run to check for regressions."""
        change = change or rem.change
        targets = self._retest_targets(run, rem)
        guard = self._retest_guard(run, rem, regression_sample)
        cand = self._candidate_cfg(run, rem.candidate)
        est = 0.0
        for s in targets + guard:
            patched, _ = apply_change(s, change, cand.reasoning_effort)
            pt = s.usage.prompt_tokens or _approx_tokens(patched)
            if s.usage.prompt_tokens and patched is not s:  # account for the added instruction
                pt += max(_approx_tokens(patched) - _approx_tokens(s), 0)
            est += estimate_cost(rem.candidate, pt, pt, 0) or 0.0
        # duration: this candidate's observed average latency in the run (sequential calls),
        # plus a rough per-call judge allowance when the LLM judge is on.
        n_rep = max(1, int(getattr(run.config, "replays", 1) or 1)) if run.config else 1
        est *= n_rep   # re-tests use the run's replays so before/after are comparable
        calls = (len(targets) + len(guard)) * n_rep
        lat = [r.latency_ms for r in run.results if r.model == rem.candidate and r.latency_ms]
        avg_ms = (sum(lat) / len(lat)) if lat else 3000.0
        judge_ms = 1500.0 if (run.config and run.config.use_judge) else 0.0
        compatible_pool = len(regression_guard_ids(run.pair_scores, rem.candidate, rem.target.sample_ids,
                                                   [s.id for s in run.samples], -1, rem.remediation_id))
        return {"remediation_id": rem.remediation_id, "candidate": rem.candidate,
                "calls": calls, "target_calls": len(targets) * n_rep, "regression_calls": len(guard) * n_rep,
                "replays": n_rep,
                "compatible_available": compatible_pool, "regression_sample": regression_sample,
                "judge_calls": calls if judge_ms else 0,
                "est_cost": round(est, 4),
                "est_seconds": round(calls * (avg_ms + judge_ms) / 1000.0, 1),
                "latency_basis_ms": round(avg_ms), "change": change.model_dump()}

    def retest_remediation(self, run: Run, rem: Remediation,
                           cancel: Optional[Callable[[], bool]] = None,
                           on_progress: Optional[Callable[[Remediation], None]] = None,
                           regression_sample: int = DEFAULT_REGRESSION_SAMPLE) -> Remediation:
        """Re-run the remediation's cluster with its change applied, through the same
        adapter / scoring / judge the run used, and record before/after. Also re-runs
        a sample of currently Compatible evaluations to detect regressions."""
        from datetime import datetime, timezone
        now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731
        cancel = cancel or (lambda: False)
        cfg = run.config
        cand = self._candidate_cfg(run, rem.candidate)
        targets = self._retest_targets(run, rem)
        guard = self._retest_guard(run, rem, regression_sample)
        guard_ids = {s.id for s in guard}
        before = {p.sample_id: p for p in run.pair_scores if p.model == rem.candidate}
        n_rep = max(1, int(getattr(cfg, "replays", 1) or 1)) if cfg else 1
        res = RemediationResult(status="running", total=(len(targets) + len(guard)) * n_rep,
                                change=rem.change.model_copy(), started_at=now(),
                                regression_sample=regression_sample, replays=n_rep)
        after_results: List[CandidateResult] = []
        after_scores: List[PairScore] = []
        rem.result, rem.applied = res, False
        if on_progress:
            on_progress(rem)

        class _Cancelled(Exception):
            pass

        try:
            if cfg is None:
                raise RuntimeError("run has no configuration to re-test with")
            adapter = self._adapter_for(cfg, rem.candidate)
            judge = self._judge(cfg.judge_steering, cfg.call_path) if cfg.use_judge else None
            par = max(1, int(getattr(cfg, "parallelism", 1) or 1))
            effort = cand.reasoning_effort
            order: Dict[Any, int] = {}
            work = []
            for s in targets + guard:
                patched, eff = apply_change(s, rem.change, effort)
                for rep_i in range(n_rep):
                    order[(s.id, rep_i)] = len(work)
                    work.append((s, patched, eff, rep_i))
            answers: Dict[Any, Optional[PairScore]] = {}

            def execute(item, ctx):
                s, patched, eff, rep_i = item

                def on_retry(a, m, why, err, wait_s=0.0):
                    if why == "throttled":
                        ctx.throttle()
                result = adapter.generate(patched, rem.candidate, CandidateSettings(reasoning_effort=eff),
                                          replay_index=rep_i, on_retry=on_retry)
                ps = None
                if result.status == "ok":
                    ps = score_pair(patched, result.output, rem.candidate, self._settings.rubric)
                    if judge is not None:
                        ps = judge.refine(patched, result.output, ps)
                    ps.replay_index = rep_i
                return result, ps

            def on_done(item, value, ctx):
                s, patched, eff, rep_i = item
                result, ps = value
                res.cost += result.cost_estimate or 0.0
                old = before.get(s.id)
                rec: Dict[str, Any] = {
                    "sample_id": s.id, "replay_index": rep_i,
                    "role": "regression_check" if s.id in guard_ids else "target",
                    "before_label": (old.label.value if old else None),
                    "status": result.status, "latency_ms": result.latency_ms,
                    "response": result.output.text if result.output else None,
                    "error": result.error,
                }
                if ps is not None:
                    rec.update(after_label=ps.label.value, hard_break=ps.hard_break,
                               judge_reason=ps.judge_reason, score=ps.model_dump(mode="json"))
                    after_scores.append(ps)
                after_results.append(result)
                answers[(s.id, rep_i)] = ps
                res.items.append(rec)
                res.done += 1
                if on_progress:
                    on_progress(rem)
                return None

            pool = CallPool(par, limiter=AdaptiveLimiter(par, enabled=self._settings.adaptive_concurrency),
                            server=server_slots(self._settings.max_inflight_calls), cancel=cancel)
            if pool.run(work, execute, on_done,
                        endpoint=lambda it: _limiter_key(rem.candidate, adapter)) == "cancelled":
                raise _Cancelled()
            res.items.sort(key=lambda r: order.get((r["sample_id"], r["replay_index"]), 0))

            def rescore(sample: NormalizedSample, effort: ReasoningEffort,
                        change: RemediationChange) -> List[PairScore]:
                # Answers were computed above (concurrently); hand them over in replay order.
                return [p for p in (answers.get((sample.id, r)) for r in range(n_rep)) if p is not None]

            retest(run.run_id, rem.candidate, rem, run.samples, run.pair_scores, rescore,
                   reasoning_effort=cand.reasoning_effort, rubric=self._settings.rubric,
                   guard_ids=sorted(guard_ids))
            if n_rep > 1:
                from .consistency import candidate_consistency
                tids = {s.id for s in targets}
                th = self._settings.instability_threshold
                b = candidate_consistency(rem.candidate,
                                          [x for x in run.results if x.sample_id in tids],
                                          [p for p in run.pair_scores if p.sample_id in tids], th)
                a = candidate_consistency(rem.candidate,
                                          [x for x in after_results if x.sample_id in tids],
                                          [p for p in after_scores if p.sample_id in tids], th)
                res.consistency_before, res.consistency_after = b["consistency"], a["consistency"]
            res.status = "done"
        except _Cancelled:
            res.status = "cancelled"
            rem.applied = False
        except Exception as exc:  # noqa: BLE001 — surface any failure on the result
            res.status, res.error = "failed", str(exc)
            rem.applied = False
        res.cost = round(res.cost, 6)
        res.finished_at = now()
        rem.result = res
        if on_progress:
            on_progress(rem)
        return rem

    # ---- run (Doc 5 §2.3) ------------------------------------------------- #
    def run(self, run: Run, cancel: Optional[Callable[[], bool]] = None) -> Run:
        cfg = run.config
        cancel = cancel or (lambda: False)
        try:
            subset = _sample_subset(run.samples, cfg)
            run.status = RunStatus.PLANNING
            run.emit("plan", **self.plan(run))

            if cfg.dry_run:
                run.status = RunStatus.DONE
                run.emit("done", dry_run=True)
                return run

            judge = self._judge(cfg.judge_steering, cfg.call_path) if cfg.use_judge else None

            run.status = RunStatus.REPLAYING
            n_rep = max(1, int(getattr(cfg, "replays", 1) or 1))
            par = max(1, int(getattr(cfg, "parallelism", 1) or 1))
            run.emit("stage", stage="replay", replays=n_rep, parallelism=par)
            total = len(subset) * len(cfg.candidates) * n_rep
            endpoints: Dict[str, str] = {}
            adapters: Dict[str, Any] = {}
            lim_keys: Dict[str, str] = {}
            for c in cfg.candidates:
                adapters[c.label] = self._adapter_for(cfg, c.model)
                endpoints[c.label] = _effective_endpoint(adapters[c.label])
                lim_keys[c.label] = _limiter_key(c.model, adapters[c.label])
            cand_idx = {c.label: i for i, c in enumerate(cfg.candidates)}
            sample_idx = {s.id: i for i, s in enumerate(subset)}

            stats = ExecStats(par)
            limiter = AdaptiveLimiter(par, enabled=self._settings.adaptive_concurrency)
            ff = FailFast(self._settings.fail_fast_error_share)
            state = {"done": 0}
            pool = CallPool(par, limiter=limiter, server=server_slots(self._settings.max_inflight_calls),
                            cancel=cancel, stats=stats, run_id=run.run_id,
                            on_event=lambda t, d: run.emit(t, **d))

            passes_started: set = set()
            pass_lock = threading.Lock()

            def execute(item, ctx):
                rep_i, c, idx, s = item
                cid = c.label

                def on_retry(a, m, why, err, wait_s=0.0):
                    if why == "throttled":
                        ctx.throttle()
                        stats.throttled(ctx.endpoint, cid, retried=True)
                    ctx.emit("retry", model=cid, sample_id=s.id, index=idx, replay=rep_i,
                             attempt=a, max_attempts=m, reason=why, error=err,
                             wait_s=wait_s, endpoint=ctx.endpoint)
                if n_rep > 1:
                    with pass_lock:
                        first_of_pass = rep_i not in passes_started
                        passes_started.add(rep_i)
                    if first_of_pass:
                        ctx.emit("pass", replay=rep_i, replays=n_rep)
                ctx.emit("call_start", model=cid, sample_id=s.id, index=idx, replay=rep_i,
                         replays=n_rep, endpoint=ctx.endpoint)
                result = adapters[cid].generate(s, cid, CandidateSettings(reasoning_effort=c.reasoning_effort),
                                                replay_index=rep_i, on_retry=on_retry)
                ps = None
                if result.status == "ok":
                    ps = score_pair(s, result.output, cid, self._settings.rubric)
                    if judge is not None:
                        ps = judge.refine(s, result.output, ps)
                    ps.replay_index = rep_i
                return result, ps

            def on_done(item, value, ctx):
                rep_i, c, idx, s = item
                result, ps = value
                run.results.append(result)
                if ps is not None:
                    run.pair_scores.append(ps)
                throttle_fail = result.status == "error" and _is_throttle_error(RuntimeError(result.error or ""))
                if throttle_fail:
                    stats.throttled(ctx.endpoint, c.label, retried=False)
                    stats.throttle_failed()
                state["done"] += 1
                snap = stats.to_dict(limiter, inflight=pool.inflight)
                run.exec_stats = snap
                u = result.usage
                run.emit("progress", done=state["done"], total=total, model=c.label,
                         sample_id=s.id, index=idx, replay=rep_i, replays=n_rep,
                         status=result.status,
                         error=(result.error if result.status == "error" else None),
                         latency_ms=result.latency_ms, attempts=result.attempts,
                         from_cache=result.from_cache,
                         cost=(round(result.cost_estimate, 6) if result.cost_known else None),
                         input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
                         reasoning_tokens=u.reasoning_tokens,
                         endpoint=ctx.endpoint, inflight=pool.inflight,
                         throttles=snap["throttles"], throttle_failures=snap["throttle_failures"])
                return ff.record(result.error if result.status == "error" else None, throttle=throttle_fail)

            def _finish_stats():
                stats.finished = time.time()
                run.exec_stats = stats.to_dict(limiter, inflight=0)
                # Completion order varies with concurrency; store in the sequential order.
                key = lambda x: (x.replay_index, cand_idx.get(x.model, 1 << 30),  # noqa: E731
                                 sample_idx.get(x.sample_id, 1 << 30))
                run.results.sort(key=key)
                run.pair_scores.sort(key=key)

            # Replays are queued pass by pass (pass 1's calls first), but passes are NOT barriers:
            # a free slot takes the next queued call even from the next pass, so one slow or
            # retrying call holds only its own slot. Every replay is a real call. A run cancelled
            # part-way can therefore hold a partly-sent later pass; consistency (D7) is computed
            # from whichever answers a request has.
            try:
                items: List[Any] = []
                for rep_i in range(n_rep):
                    if par == 1:   # today's order: candidate by candidate
                        items += [(rep_i, c, idx, s) for c in cfg.candidates for idx, s in enumerate(subset)]
                    else:          # interleave candidates so every model progresses together
                        items += [(rep_i, c, idx, s) for idx, s in enumerate(subset) for c in cfg.candidates]
                status = pool.run(items, execute, on_done, endpoint=lambda it: lim_keys[it[1].label])
                if status == "cancelled":
                    _finish_stats()
                    run.status = RunStatus.CANCELLED
                    run.emit("cancelled")
                    return run
            except RunAbort as abort:
                _finish_stats()
                raise RuntimeError(str(abort))
            _finish_stats()

            run.status = RunStatus.AGGREGATING
            run.emit("stage", stage="aggregate")
            verdicts = [aggregate(c.label, run.pair_scores, run.results, self._settings.rubric,
                                  endpoint=endpoints.get(c.label, ""), samples=subset,
                                  settings=self._settings, monthly_volume=cfg.monthly_volume)
                        for c in cfg.candidates]
            run.verdicts = side_by_side(verdicts)
            for c in cfg.candidates:
                run.remediations.extend(recommend(run.run_id, c.label, run.pair_scores,
                                                  samples=subset, reasoning_effort=c.reasoning_effort))

            run.status = RunStatus.DONE
            run.emit("done", candidates=[v.model for v in run.verdicts])
            return run
        except Exception as exc:  # noqa: BLE001
            import traceback as _tb
            run.status = RunStatus.FAILED
            run.error = str(exc)
            run.error_detail = _tb.format_exc()
            run.emit("error", error=str(exc), detail=run.error_detail)
            return run


def _approx_tokens(sample: NormalizedSample) -> int:
    chars = len(sample.request.instructions or "")
    chars += sum(len(m.content) for m in sample.request.input_messages)
    return max(chars // 4, 1)  # ~4 chars/token


# Convenience: run in a background thread (used by the API).
def run_async(orch: Orchestrator, run: Run,
              cancel_flag: Dict[str, bool],
              on_complete: Optional[Callable[[Run], None]] = None) -> threading.Thread:
    def _target():
        orch.run(run, cancel=lambda: cancel_flag.get(run.run_id, False))
        if on_complete:
            try:
                on_complete(run)
            except Exception:  # noqa: BLE001 — persistence must never crash the worker
                pass
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:10]}"


# Silence unused-import checkers for the live-only names that the API imports.
__all__ = [
    "Orchestrator", "Run", "RunConfig", "RunStatus", "CandidateConfig",
    "default_candidates_for", "run_async", "new_run_id", "MODEL_CATALOG",
]
