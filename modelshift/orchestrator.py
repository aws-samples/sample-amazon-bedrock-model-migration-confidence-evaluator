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

from .adapters import BedrockResponsesAdapter, CandidateAdapter, LiteLLMResponsesAdapter, Transport
from .aggregate import aggregate, recommend, retest, side_by_side
from .config import DEFAULT_SETTINGS, MODEL_CATALOG, Settings, candidates_for, estimate_cost
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
)


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
    parallelism: int = 6
    dry_run: bool = False


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
            for s in subset:
                pt = s.usage.prompt_tokens or _approx_tokens(s)
                est += estimate_cost(c.model, pt, pt, 0)
            per_candidate.append({"model": c.model, "calls": calls, "est_cost": round(est, 4)})
            total += est
        return {"estimated_calls": len(subset) * len(cfg.candidates),
                "est_cost": round(total, 4),
                "per_candidate": per_candidate}

    def _adapter_for(self, cfg: RunConfig, model: str) -> CandidateAdapter:
        entry = MODEL_CATALOG.get(model)
        family = entry.family if entry else "openai"
        # Anthropic (Claude) candidates use the Bedrock Converse transport regardless of call_path.
        if family == "anthropic":
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
        # runtime, even if the run also contains a mantle-only model (GPT-5.4). Only a
        # runtime-unsupported model — or an explicit run-wide mantle choice for THIS
        # model — routes to mantle.
        runtime_ok = entry.runtime_supported if entry else True
        use_mantle = (not runtime_ok) or (cfg.bedrock_endpoint == "mantle" and not runtime_ok)
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
            if c.model == model:
                return c
        return CandidateConfig(model=model)

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
            est += estimate_cost(rem.candidate, pt, pt, 0)
        # duration: this candidate's observed average latency in the run (sequential calls),
        # plus a rough per-call judge allowance when the LLM judge is on.
        calls = len(targets) + len(guard)
        lat = [r.latency_ms for r in run.results if r.model == rem.candidate and r.latency_ms]
        avg_ms = (sum(lat) / len(lat)) if lat else 3000.0
        judge_ms = 1500.0 if (run.config and run.config.use_judge) else 0.0
        compatible_pool = len(regression_guard_ids(run.pair_scores, rem.candidate, rem.target.sample_ids,
                                                   [s.id for s in run.samples], -1, rem.remediation_id))
        return {"remediation_id": rem.remediation_id, "candidate": rem.candidate,
                "calls": calls, "target_calls": len(targets), "regression_calls": len(guard),
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
        res = RemediationResult(status="running", total=len(targets) + len(guard),
                                change=rem.change.model_copy(), started_at=now(),
                                regression_sample=regression_sample)
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

            def rescore(sample: NormalizedSample, effort: ReasoningEffort,
                        change: RemediationChange) -> Optional[PairScore]:
                if cancel():
                    raise _Cancelled()
                patched, eff = apply_change(sample, change, effort)
                result = adapter.generate(patched, rem.candidate, CandidateSettings(reasoning_effort=eff))
                res.cost += result.cost_estimate or 0.0
                old = before.get(sample.id)
                item: Dict[str, Any] = {
                    "sample_id": sample.id,
                    "role": "regression_check" if sample.id in guard_ids else "target",
                    "before_label": (old.label.value if old else None),
                    "status": result.status, "latency_ms": result.latency_ms,
                    "response": result.output.text if result.output else None,
                    "error": result.error,
                }
                ps: Optional[PairScore] = None
                if result.status == "ok":
                    ps = score_pair(patched, result.output, rem.candidate, self._settings.rubric)
                    if judge is not None:
                        ps = judge.refine(patched, result.output, ps)
                    item.update(after_label=ps.label.value, hard_break=ps.hard_break,
                                judge_reason=ps.judge_reason, score=ps.model_dump(mode="json"))
                res.items.append(item)
                res.done += 1
                if on_progress:
                    on_progress(rem)
                return ps

            retest(run.run_id, rem.candidate, rem, run.samples, run.pair_scores, rescore,
                   reasoning_effort=cand.reasoning_effort, rubric=self._settings.rubric,
                   guard_ids=sorted(guard_ids))
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
            total = len(subset) * len(cfg.candidates)
            done = 0
            endpoints: Dict[str, str] = {}
            for c in cfg.candidates:
                adapter = self._adapter_for(cfg, c.model)
                endpoints[c.model] = _effective_endpoint(adapter)
                settings = CandidateSettings(reasoning_effort=c.reasoning_effort)
                for s in subset:
                    if cancel():
                        run.status = RunStatus.CANCELLED
                        run.emit("cancelled")
                        return run
                    result = adapter.generate(s, c.model, settings)
                    run.results.append(result)
                    if result.status == "ok":
                        ps = score_pair(s, result.output, c.model, self._settings.rubric)
                        if judge is not None:
                            ps = judge.refine(s, result.output, ps)
                        run.pair_scores.append(ps)
                    done += 1
                    run.emit("progress", done=done, total=total, model=c.model,
                             sample_id=s.id, status=result.status,
                             error=(result.error if result.status == "error" else None))

            run.status = RunStatus.AGGREGATING
            verdicts = [aggregate(c.model, run.pair_scores, run.results, self._settings.rubric,
                                  endpoint=endpoints.get(c.model, ""))
                        for c in cfg.candidates]
            run.verdicts = side_by_side(verdicts)
            for c in cfg.candidates:
                run.remediations.extend(recommend(run.run_id, c.model, run.pair_scores,
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
