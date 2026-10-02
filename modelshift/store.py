"""File-backed run store — persists runs so they survive server restarts.

Location: ~/.modelshift/runs/<run_id>.json (override dir via MODELSHIFT_RUNS_DIR).
Runs are saved at lifecycle checkpoints (ingested / done / failed / cancelled)
and loaded on boot. Writes are atomic (.tmp + replace).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .orchestrator import (
    CandidateConfig,
    ProgressEvent,
    Run,
    RunConfig,
    RunStatus,
)
from .schemas import (
    CandidateResult,
    CandidateVerdict,
    IngestionReport,
    NormalizedSample,
    PairScore,
    Remediation,
)


def runs_dir() -> Path:
    env = os.environ.get("MODELSHIFT_RUNS_DIR")
    if env:
        p = Path(env)
    else:
        home = Path(os.environ.get("MODELSHIFT_HOME") or (Path.home() / ".modelshift"))
        p = home / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run_to_dict(run: Run) -> Dict:
    return {
        "run_id": run.run_id,
        "name": run.name,
        "description": run.description,
        "status": run.status.value,
        "config": _config_to_dict(run.config) if run.config else None,
        "samples": [s.model_dump(mode="json") for s in run.samples],
        "ingestion": run.ingestion.model_dump(mode="json") if run.ingestion else None,
        "results": [r.model_dump(mode="json") for r in run.results],
        "pair_scores": [p.model_dump(mode="json") for p in run.pair_scores],
        "verdicts": [v.model_dump(mode="json") for v in run.verdicts],
        "remediations": [x.model_dump(mode="json") for x in run.remediations],
        "events": [asdict(e) for e in run.events],
        "error": run.error,
        "error_detail": run.error_detail,
    }


def _config_to_dict(cfg: RunConfig) -> Dict:
    d = asdict(cfg)
    d["candidates"] = [{"model": c.model, "reasoning_effort": c.reasoning_effort.value}
                       for c in cfg.candidates]
    d["call_path"] = cfg.call_path.value
    return d


def _config_from_dict(d: Dict) -> RunConfig:
    from .schemas import CallPath, ReasoningEffort
    return RunConfig(
        candidates=[CandidateConfig(model=c["model"],
                                    reasoning_effort=ReasoningEffort(c["reasoning_effort"]))
                    for c in d.get("candidates", [])],
        call_path=CallPath(d.get("call_path", "bedrock")),
        bedrock_endpoint=d.get("bedrock_endpoint", "runtime"),
        sampling_mode=d.get("sampling_mode", "all"),
        sampling_n=d.get("sampling_n", 200),
        sampling_seed=d.get("sampling_seed", 42),
        use_judge=d.get("use_judge", True),
        parallelism=d.get("parallelism", 6),
        dry_run=d.get("dry_run", False),
    )


def _run_from_dict(d: Dict) -> Run:
    run = Run(run_id=d["run_id"], name=d.get("name", ""), description=d.get("description", ""))
    run.status = RunStatus(d.get("status", "created"))
    run.config = _config_from_dict(d["config"]) if d.get("config") else None
    run.samples = [NormalizedSample(**s) for s in d.get("samples", [])]
    run.ingestion = IngestionReport(**d["ingestion"]) if d.get("ingestion") else None
    run.results = [CandidateResult(**r) for r in d.get("results", [])]
    run.pair_scores = [PairScore(**p) for p in d.get("pair_scores", [])]
    run.verdicts = [CandidateVerdict(**v) for v in d.get("verdicts", [])]
    run.remediations = [Remediation(**x) for x in d.get("remediations", [])]
    # A re-test that was running when the server stopped can never finish.
    for rem in run.remediations:
        if rem.result is not None and rem.result.status == "running":
            rem.result.status = "interrupted"
            rem.result.error = "server restarted during the re-test; launch it again"
            rem.applied = False
    run.events = [ProgressEvent(type=e["type"], data=e.get("data", {}), ts=e.get("ts", 0.0))
                  for e in d.get("events", [])]
    run.error = d.get("error")
    run.error_detail = d.get("error_detail")
    return run


def save_run(run: Run) -> None:
    p = runs_dir() / f"{run.run_id}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_run_to_dict(run)))
    tmp.replace(p)


def load_all_runs() -> Dict[str, Run]:
    out: Dict[str, Run] = {}
    for f in sorted(runs_dir().glob("*.json")):
        try:
            out[f.stem] = _run_from_dict(json.loads(f.read_text()))
        except (ValueError, OSError, KeyError, TypeError):
            continue  # skip a corrupt/incompatible run file, don't crash boot
    return out


def delete_run(run_id: str) -> None:
    f = runs_dir() / f"{run_id}.json"
    if f.exists():
        f.unlink()


def list_run_ids() -> List[str]:
    return [f.stem for f in sorted(runs_dir().glob("*.json"))]
