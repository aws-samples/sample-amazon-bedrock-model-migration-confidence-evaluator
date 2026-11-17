"""Config-driven policy: migration matrix, reasoning defaults, price table, rubric.

Everything here is data (NFR-EXT-1): new models / mappings / prices / thresholds
are added by editing these structures (or an override YAML), never code.

Sources:
  - Migration matrix + reasoning policy: Doc 1 §6
  - Price table: Doc 1 NFR-COST-1, Doc 5 §5
  - Rubric (labels, thresholds, hard-break, exact-match short-circuit): Doc 3 §4
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .schemas import CompatLabel, ReasoningEffort, VerdictBand


# --------------------------------------------------------------------------- #
# Migration matrix (Doc 1 §6)
# --------------------------------------------------------------------------- #
class MatrixEntry(BaseModel):
    legacy_model: str
    candidates: List[str]           # side-by-side candidates, in preference order
    default_start: str              # the recommended starting candidate
    guidance: str                   # one-liner shown in the Config UI


# GPT-5.6 family + GPT-5.4 like-for-like.
MIGRATION_MATRIX: List[MatrixEntry] = [
    MatrixEntry(
        legacy_model="gpt-4.1",
        candidates=["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
        default_start="gpt-5.6-luna",
        guidance="Luna for well-defined, cost-sensitive interactions; Terra for nuanced "
                 "conversation, complex tool use, stronger reasoning; Sol for particularly complex workloads.",
    ),
    MatrixEntry(
        legacy_model="gpt-4.1-mini",
        candidates=["gpt-5.6-luna"],
        default_start="gpt-5.6-luna",
        guidance="Recommended starting point — migrating up from a smaller model.",
    ),
    MatrixEntry(
        legacy_model="gpt-5",
        candidates=["gpt-5.6-luna", "gpt-5.6-terra"],
        default_start="gpt-5.6-luna",
        guidance="Luna for well-defined/cost-sensitive; Terra for nuanced/tool-heavy/reasoning.",
    ),
    MatrixEntry(
        legacy_model="gpt-5-mini",
        candidates=["gpt-5.6-luna"],
        default_start="gpt-5.6-luna",
        guidance="Natural initial candidate for smaller, cost-efficient workloads.",
    ),
    MatrixEntry(
        legacy_model="gpt-5.2",
        candidates=["gpt-5.6-terra", "gpt-5.6-sol"],
        default_start="gpt-5.6-terra",
        guidance="Start with Terra; consider Sol where extra capability could improve results.",
    ),
    MatrixEntry(
        legacy_model="gpt-5.4",
        candidates=["gpt-5.4", "gpt-5.6-sol"],
        default_start="gpt-5.4",
        guidance="GPT-5.4 available as a like-for-like peer; evaluate Sol for complex workflows.",
    ),
    MatrixEntry(
        legacy_model="gpt-5.1",
        candidates=["gpt-5.6-luna", "gpt-5.6-terra"],
        default_start="gpt-5.6-luna",
        guidance="Luna for routine interactions; Terra for complex customer-facing / tool-heavy workflows.",
    ),
]


def _norm(model: Optional[str]) -> str:
    """Normalize a logged model id to a matrix key (strip provider prefixes)."""
    if not model:
        return ""
    m = model.strip().lower()
    for prefix in ("azure/", "openai/", "bedrock/converse/", "bedrock/", "us.openai.", "litellm_proxy/"):
        if m.startswith(prefix):
            m = m[len(prefix):]
    return m


def candidates_for(legacy_model: Optional[str]) -> Optional[MatrixEntry]:
    """Return the matrix entry for a detected legacy model, or None if unmapped."""
    key = _norm(legacy_model)
    for entry in MIGRATION_MATRIX:
        if _norm(entry.legacy_model) == key:
            return entry
    return None


# --------------------------------------------------------------------------- #
# Reasoning-effort policy (Doc 1 §6.1)
# --------------------------------------------------------------------------- #
DEFAULT_REASONING_EFFORT = ReasoningEffort.MEDIUM

# Models available as candidates, with their Bedrock inference-profile ids and
# LiteLLM model-group aliases. GPT-5.6 models are inference-profile-only and need
# the bedrock/converse/ prefix in LiteLLM (known gotcha).


class ModelCatalogEntry(BaseModel):
    name: str
    bedrock_model_id: str            # runtime (us.openai.*) id — the default adapter ref
    mantle_model_id: str             # mantle (openai.*) id for the /openai/v1 mantle endpoint
    litellm_model_group: str         # alias used by the LiteLLM adapter
    family: str = "openai"           # "openai" (/openai/v1 or invoke_model) | "anthropic" (Converse)
    like_for_like: bool = False      # GPT-5.4 peer (no reasoning remap by default)
    runtime_supported: bool = True   # False => mantle-only (no catalog model today)
    structured_outputs: bool = False  # per AWS model card (Luna yes, Terra no)
    # Reasoning efforts this model accepts, lowest -> highest. Probed live against Bedrock
    # (us-east-1, 2026-11-12): GPT-5.6 rejects 'minimal' ("Supported values are: 'none', 'low',
    # 'medium', 'high', and 'xhigh'"); GPT-5.4 accepts 'minimal' but rejects 'max'.
    reasoning_efforts: List[str] = Field(default_factory=lambda: ["none", "low", "medium", "high", "xhigh"])
    # False when the call path ignores reasoning effort (e.g. Claude via Converse): extra efforts
    # would just repeat the same request, so only one is offered.
    effort_applies: bool = True


def supported_efforts(model: Optional[str]) -> Optional[List[str]]:
    """Efforts a model accepts: catalog list, probed list for a discovered model, or None
    when unknown (not validated)."""
    entry = MODEL_CATALOG.get(base_model(model)) if model else None
    if entry:
        return list(entry.reasoning_efforts)
    from .discovery import supported_efforts as _discovered_efforts
    return _discovered_efforts(model)


MODEL_CATALOG: Dict[str, ModelCatalogEntry] = {
    "gpt-5.6-luna": ModelCatalogEntry(
        name="gpt-5.6-luna",
        bedrock_model_id="us.openai.gpt-5.6-luna",
        mantle_model_id="openai.gpt-5.6-luna",
        litellm_model_group="gpt-5.6-luna",
        structured_outputs=True,
    ),
    "gpt-5.6-terra": ModelCatalogEntry(
        name="gpt-5.6-terra",
        bedrock_model_id="us.openai.gpt-5.6-terra",
        mantle_model_id="openai.gpt-5.6-terra",
        litellm_model_group="gpt-5.6-terra",
        structured_outputs=False,
    ),
    "gpt-5.6-sol": ModelCatalogEntry(
        name="gpt-5.6-sol",
        bedrock_model_id="us.openai.gpt-5.6-sol",
        mantle_model_id="openai.gpt-5.6-sol",
        litellm_model_group="gpt-5.6-sol",
    ),
    "gpt-5.4": ModelCatalogEntry(
        name="gpt-5.4",
        # GPT-5.4 is on BOTH endpoints: bedrock-runtime via the us.openai.gpt-5.4 inference
        # profile (probed live us-east-1 2026-11-13; the bare openai.gpt-5.4 id is refused on
        # runtime with "on-demand throughput isn't supported") and bedrock-mantle via openai.gpt-5.4.
        bedrock_model_id="us.openai.gpt-5.4",
        mantle_model_id="openai.gpt-5.4",
        litellm_model_group="gpt-5.4",
        like_for_like=True,
        reasoning_efforts=["none", "minimal", "low", "medium", "high", "xhigh"],
    ),
    "claude-sonnet-4.5": ModelCatalogEntry(
        name="claude-sonnet-4.5",
        # Anthropic on Bedrock — uses the Converse API, not the OpenAI /openai/v1 path.
        bedrock_model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        mantle_model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        litellm_model_group="claude-sonnet-4.5",
        family="anthropic",
        structured_outputs=True,
        # The Converse transport does not send a reasoning effort for Claude.
        reasoning_efforts=["medium"],
        effort_applies=False,
    ),
}


# --------------------------------------------------------------------------- #
# Price table — PUBLIC Amazon Bedrock list prices, Standard tier, short context
# (<=272K input tokens), USD per 1K tokens. Source: the Bedrock model cards
# (docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-*.html), checked
# 2026-11-12. For the models below In-Region and US Geo CRIS (the us.* profiles
# ModelShift calls) are the same price. Reasoning tokens are billed at the output rate.
# cache_read_per_1k is the published cache-read rate; not yet used by estimate_cost.
# Always use public pricing here -- no negotiated/enterprise rates.
# --------------------------------------------------------------------------- #
class ModelPrice(BaseModel):
    input_per_1k: float
    output_per_1k: float
    cache_read_per_1k: Optional[float] = None


PRICE_TABLE: Dict[str, ModelPrice] = {
    # GPT-5.6 Luna: $0.22 in / $0.022 cache read / $1.32 out per 1M
    "gpt-5.6-luna": ModelPrice(input_per_1k=0.00022, output_per_1k=0.00132, cache_read_per_1k=0.000022),
    # GPT-5.6 Terra: $2.20 / $0.22 / $13.20 per 1M
    "gpt-5.6-terra": ModelPrice(input_per_1k=0.00220, output_per_1k=0.01320, cache_read_per_1k=0.00022),
    # GPT-5.6 Sol: $4.40 / $0.44 / $22.00 per 1M (was $5.00/$30.00 -- not the public rate)
    "gpt-5.6-sol": ModelPrice(input_per_1k=0.00440, output_per_1k=0.02200, cache_read_per_1k=0.00044),
    # GPT-5.5: $5.50 / $0.55 / $33.00 per 1M (was missing -> costed as $0)
    "gpt-5.5": ModelPrice(input_per_1k=0.00550, output_per_1k=0.03300, cache_read_per_1k=0.00055),
    # GPT-5.4: $2.75 / $0.275 / $16.50 per 1M
    "gpt-5.4": ModelPrice(input_per_1k=0.00275, output_per_1k=0.01650, cache_read_per_1k=0.000275),
    # Claude Sonnet 4.5 (judge): rate NOT re-verified -- the model card defers to the
    # Bedrock pricing page, whose tables render client-side. Kept at $3.00 / $15.00.
    "claude-sonnet-4.5": ModelPrice(input_per_1k=0.003, output_per_1k=0.015),
}


CANDIDATE_EFFORT_SEP = "@"


def candidate_label(model: str, effort) -> str:
    """Candidate identity = model + reasoning effort, e.g. 'gpt-5.6-luna@low'.

    The same model at two efforts is two candidates (spec Decision 6), so every run-level key
    (results, pair scores, verdicts, remediations, progress events) uses this label.
    """
    eff = getattr(effort, "value", effort)
    return f"{base_model(model)}{CANDIDATE_EFFORT_SEP}{eff}" if eff else base_model(model)


def base_model(name: Optional[str]) -> str:
    """The real model behind a candidate label ('gpt-5.6-luna@low' -> 'gpt-5.6-luna').
    Plain model names (and runs saved before labels existed) pass through unchanged."""
    return (name or "").split(CANDIDATE_EFFORT_SEP, 1)[0]


def price_key_for(model: Optional[str]) -> str:
    """Map a candidate/model reference to a price-table key.

    Accepts the catalog name ('gpt-5.5'), a candidate label ('gpt-5.5@high'), a Bedrock id
    ('us.openai.gpt-5.5', 'openai.gpt-5.4'), or a LiteLLM model group, so every way ModelShift
    names a model hits the same price row.
    """
    if not model:
        return ""
    raw = base_model(model).strip()
    if raw in PRICE_TABLE:
        return raw
    for name, entry in MODEL_CATALOG.items():
        if raw in (entry.bedrock_model_id, entry.mantle_model_id, entry.litellm_model_group):
            return name
    m = _norm(raw)
    for prefix in ("global.openai.", "openai.", "us.", "global."):
        if m.startswith(prefix):
            m = m[len(prefix):]
    return m


def price_for(model: Optional[str]) -> Optional[ModelPrice]:
    """Current price for a model: the live Settings table (Settings-page edits apply), else None."""
    key = price_key_for(model)
    table = DEFAULT_SETTINGS.price_table if "DEFAULT_SETTINGS" in globals() else PRICE_TABLE
    return table.get(key) or table.get(model or "")


def billed_output_tokens(output_tokens: int, reasoning_tokens: int = 0) -> int:
    """Output tokens actually billed.

    Bedrock (like the OpenAI API) reports reasoning tokens as a SUBSET of the output count
    (`output_tokens_details.reasoning_tokens` <= `output_tokens`); the output total is what is
    billed. Only when a source reports reasoning larger than output (i.e. counted separately)
    are the two added.
    """
    output_tokens = output_tokens or 0
    reasoning_tokens = reasoning_tokens or 0
    return output_tokens if output_tokens >= reasoning_tokens else output_tokens + reasoning_tokens


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  reasoning_tokens: int = 0) -> Optional[float]:
    """List-price cost of one call, or None when the model has no configured price.

    None means UNKNOWN, not free: callers must not treat it as $0 when comparing models.
    Reasoning is billed at the output rate and is not double-counted (see billed_output_tokens).
    """
    price = price_for(model)
    if price is None:
        return None
    return (
        input_tokens / 1000.0 * price.input_per_1k
        + billed_output_tokens(output_tokens, reasoning_tokens) / 1000.0 * price.output_per_1k
    )


# --------------------------------------------------------------------------- #
# Scoring rubric (Doc 3 §4) — thresholds & weights, all tunable.
# --------------------------------------------------------------------------- #
class Rubric(BaseModel):
    # Dimension thresholds (0..1)
    semantic_high: float = 0.85         # D1 >= this counts as "high"
    semantic_moderate: float = 0.60     # D1 >= this counts as "moderate"
    verbosity_drift_ratio: float = 1.75  # candidate/golden length ratio flagged as drift
    verbosity_drift_ratio_low: float = 0.5

    # Migration Confidence weighting (Doc 3 §4.3)
    weight_compatible: float = 1.0
    weight_with_drift: float = 0.7
    weight_partial: float = 0.3
    weight_incompatible: float = 0.0

    # Verdict bands by Migration Confidence (0..100), applied top-down.
    band_safe_min: float = 85.0
    band_tuning_min: float = 65.0
    band_needs_work_min: float = 40.0

    # Exact/near-exact match short-circuits to Compatible (Doc 3 §4.2 note).
    exact_match_short_circuit: bool = True

    def band_for(self, confidence: float, hard_break_rate: float) -> VerdictBand:
        if hard_break_rate > 0.20:
            # a high hard-break rate can't be "safe" regardless of confidence
            if confidence >= self.band_needs_work_min:
                return VerdictBand.NEEDS_WORK
            return VerdictBand.NOT_A_DROP_IN
        if confidence >= self.band_safe_min:
            return VerdictBand.SAFE_DROP_IN
        if confidence >= self.band_tuning_min:
            return VerdictBand.DROP_IN_WITH_TUNING
        if confidence >= self.band_needs_work_min:
            return VerdictBand.NEEDS_WORK
        return VerdictBand.NOT_A_DROP_IN

    def confidence_weight(self, label: CompatLabel) -> float:
        return {
            CompatLabel.COMPATIBLE: self.weight_compatible,
            CompatLabel.COMPATIBLE_WITH_DRIFT: self.weight_with_drift,
            CompatLabel.PARTIAL: self.weight_partial,
            CompatLabel.INCOMPATIBLE: self.weight_incompatible,
        }[label]


DEFAULT_RUBRIC = Rubric()


# --------------------------------------------------------------------------- #
# Recommendation defaults (spec: multi-objective-recommendation R4.4, R6.1).
# --------------------------------------------------------------------------- #
WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "balanced": {"quality": 0.50, "cost": 0.30, "latency": 0.20},
    "quality_first": {"quality": 0.70, "cost": 0.15, "latency": 0.15},
    "cost_first": {"quality": 0.35, "cost": 0.50, "latency": 0.15},
    "latency_first": {"quality": 0.35, "cost": 0.15, "latency": 0.50},
}

LATENCY_DISCLAIMER = (
    "Latency numbers are indicative only. They were measured from the machine running this "
    "evaluation (for example a laptop or a dev account), over its own network path, with the "
    "evaluation's own concurrency and account quotas. Production latency will differ with "
    "deployment region, network path (VPC endpoints, gateway hops such as LiteLLM), "
    "concurrency, quota and throughput tier, and prompt caching. Use these numbers to compare "
    "candidates relative to each other within this run, not as production SLAs. Validate "
    "latency in your target environment before relying on it."
)


# --------------------------------------------------------------------------- #
# Aggregated settings object (Doc 5 §5) — the single tunable surface.
# --------------------------------------------------------------------------- #
class Settings(BaseModel):
    default_reasoning_effort: ReasoningEffort = DEFAULT_REASONING_EFFORT
    price_table: Dict[str, ModelPrice] = Field(default_factory=lambda: dict(PRICE_TABLE))
    rubric: Rubric = Field(default_factory=Rubric)
    judge_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    # LiteLLM model-group alias for the judge when a run routes through the proxy
    # (the proxy registers Claude as this group name, not the raw inference-profile id).
    judge_litellm_model: str = "claude-sonnet-4.5"
    # ---- Concurrent model calls (parallel-calls spec) ----
    default_parallelism: int = 4          # calls a run sends at once (1-16); copied into each run
    adaptive_concurrency: bool = True     # halve on throttling, recover after clean calls
    max_inflight_calls: int = 16          # server-wide cap across all runs and re-tests
    fail_fast_error_share: Optional[float] = 0.5   # stop when this share of the last 20 calls fail alike
    # Low-signal caveat thresholds (tunable, NFR-EXT-1). A caveat is advisory only
    # — it never blocks a run. Default 10 so a 15-row smoke test does not trip it.
    small_sample_threshold: int = 10
    high_redaction_threshold: float = 0.5
    # LiteLLM proxy connection (used when a run's call path is LiteLLM). The base
    # URL is non-secret; the key is a secret (persisted but never returned by the API).
    litellm_base: str = "http://127.0.0.1:4000"
    litellm_key: str = ""

    # ---- Multi-objective recommendation (spec: multi-objective-recommendation) ---- #
    # Quality gate: a candidate is only recommended if it passes all of these.
    min_confidence: float = 65.0        # Migration Confidence floor (= "drop-in with tuning")
    max_hard_break_rate: float = 0.05
    max_error_rate: float = 0.05
    min_scored: int = 1
    # Economical / lowest-latency picks must be within this many confidence points of the
    # best-quality candidate.
    quality_tolerance: float = 10.0
    # Fewer timed (fresh, successful) calls than this -> latency is "low confidence".
    min_latency_samples: int = 5
    latency_metric: str = "per_token"   # "per_token" (median ms / output token) | "p50"
    missing_axis_margin: float = 5.0
    # Statistical ties: candidates whose Migration Confidence can't be separated from the
    # best-quality pick (paired 95% bootstrap interval of the difference contains 0) get the
    # full quality score, so the pick between them rests on cost and latency.
    statistical_ties: bool = True
    # Weight presets as data (quality, cost, latency). Normalised to sum to 1 when used.
    weight_presets: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        k: dict(v) for k, v in WEIGHT_PRESETS.items()})
    latency_disclaimer: str = LATENCY_DISCLAIMER
    # ---- Request replays / D7 Consistency (replay-consistency spec) ----
    default_replays: int = 3              # N preselected when replays are switched on (2-5)
    replay_call_ceiling: int = 2000       # warn before launch above this many calls
    instability_threshold: float = 0.70   # D7 below this marks a request / candidate unstable
    max_flip_rate: Optional[float] = None    # optional gate (off by default)
    min_consistency: Optional[float] = None  # optional gate on mean D7 (off by default)


DEFAULT_SETTINGS = Settings()


# --------------------------------------------------------------------------- #
# File-backed persistence (survives server restarts). Location overridable via
# the MODELSHIFT_SETTINGS_PATH env var; defaults to ~/.modelshift/settings.json.
# --------------------------------------------------------------------------- #
import json as _json  # noqa: E402
import os as _os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


def settings_path() -> _Path:
    env = _os.environ.get("MODELSHIFT_SETTINGS_PATH")
    if env:
        return _Path(env)
    home = _Path(_os.environ.get("MODELSHIFT_HOME") or (_Path.home() / ".modelshift"))
    return home / "settings.json"


def save_settings(s: Settings = DEFAULT_SETTINGS) -> None:
    """Persist the tunable (non-secret) settings to disk."""
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "default_reasoning_effort": s.default_reasoning_effort.value,
        "judge_model": s.judge_model,
        "judge_litellm_model": s.judge_litellm_model,
        "default_parallelism": s.default_parallelism,
        "adaptive_concurrency": s.adaptive_concurrency,
        "max_inflight_calls": s.max_inflight_calls,
        "fail_fast_error_share": s.fail_fast_error_share,
        "small_sample_threshold": s.small_sample_threshold,
        "high_redaction_threshold": s.high_redaction_threshold,
        "litellm_base": s.litellm_base,
        "litellm_key": s.litellm_key,
        # Only prices the user changed are saved, so newer public defaults in code are
        # never pinned to stale values by an old settings file.
        "price_overrides": price_overrides(s),
        **{k: getattr(s, k) for k in RECOMMENDATION_SETTING_KEYS},
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(data, indent=2))
    try:
        _os.chmod(tmp, 0o600)  # settings.json holds the litellm_key — owner-only
    except OSError:
        pass
    tmp.replace(p)  # atomic
    try:
        _os.chmod(p, 0o600)
    except OSError:
        pass


def resolve_litellm_key(settings: "Settings" = None) -> str:
    """Resolve the LiteLLM proxy key in priority order:
      1. env MODELSHIFT_LITELLM_KEY (highest — 12-factor / container secret env)
      2. AWS Secrets Manager (MODELSHIFT_LITELLM_KEY_SECRET_ARN) — fetched lazily
      3. persisted settings.litellm_key (the UI-saved value on disk)
    Returns "" if none is available. Never logs the value.
    """
    s = settings if settings is not None else DEFAULT_SETTINGS
    env_key = _os.environ.get("MODELSHIFT_LITELLM_KEY")
    if env_key:
        return env_key
    secret_arn = _os.environ.get("MODELSHIFT_LITELLM_KEY_SECRET_ARN")
    if secret_arn:
        try:
            import boto3  # lazy — keeps the module importable without AWS deps
            region = _os.environ.get("MODELSHIFT_REGION") or _os.environ.get("AWS_REGION") or "us-east-1"
            sm = boto3.client("secretsmanager", region_name=region)
            val = sm.get_secret_value(SecretId=secret_arn).get("SecretString") or ""
            if val:
                return val
        except Exception:  # noqa: BLE001 — fall through to persisted settings on any SM error
            pass
    return s.litellm_key or ""


def load_settings(into: Settings = DEFAULT_SETTINGS) -> Settings:
    """Load persisted overrides into the given Settings singleton (if the file exists)."""
    p = settings_path()
    if not p.exists():
        return into
    try:
        data = _json.loads(p.read_text())
    except (ValueError, OSError):
        return into
    if data.get("default_reasoning_effort"):
        try:
            into.default_reasoning_effort = ReasoningEffort(data["default_reasoning_effort"])
        except ValueError:
            pass
    if data.get("judge_model"):
        into.judge_model = data["judge_model"]
    if data.get("judge_litellm_model"):
        into.judge_litellm_model = data["judge_litellm_model"]
    apply_concurrency_settings(into, data)
    if isinstance(data.get("small_sample_threshold"), int):
        into.small_sample_threshold = data["small_sample_threshold"]
    if isinstance(data.get("high_redaction_threshold"), (int, float)):
        into.high_redaction_threshold = float(data["high_redaction_threshold"])
    if isinstance(data.get("litellm_base"), str) and data["litellm_base"]:
        into.litellm_base = data["litellm_base"]
    if isinstance(data.get("litellm_key"), str):
        into.litellm_key = data["litellm_key"]
    _apply_saved_prices(into, data)
    _apply_recommendation_settings(into, data)
    return into


# Tunable recommendation settings persisted in settings.json (the disclaimer text is not:
# it stays code-owned so wording fixes ship with the build).
RECOMMENDATION_SETTING_KEYS = (
    "min_confidence", "max_hard_break_rate", "max_error_rate", "min_scored",
    "quality_tolerance", "min_latency_samples", "latency_metric", "missing_axis_margin",
    "weight_presets", "statistical_ties", "replay_call_ceiling", "instability_threshold",
    "max_flip_rate", "min_consistency",
)


def apply_concurrency_settings(into: "Settings", data: Dict) -> None:
    """Validate and apply concurrency settings from a saved file or PUT /settings body.
    Bad types are ignored (load) — the API layer rejects them first with a 400."""
    _int = lambda x: isinstance(x, int) and not isinstance(x, bool)  # noqa: E731
    if _int(data.get("default_parallelism")):
        into.default_parallelism = max(1, min(16, data["default_parallelism"]))
    if isinstance(data.get("adaptive_concurrency"), bool):
        into.adaptive_concurrency = data["adaptive_concurrency"]
    if _int(data.get("max_inflight_calls")):
        into.max_inflight_calls = max(1, min(64, data["max_inflight_calls"]))
    if "fail_fast_error_share" in data:
        v = data["fail_fast_error_share"]
        if v is None:
            into.fail_fast_error_share = None
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            into.fail_fast_error_share = max(0.05, min(1.0, float(v)))


def _apply_recommendation_settings(into: Settings, data: Dict) -> None:
    _num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)  # noqa: E731
    if _num(data.get("default_replays")):
        into.default_replays = max(2, min(5, int(data["default_replays"])))
    if _num(data.get("replay_call_ceiling")):
        into.replay_call_ceiling = max(1, int(data["replay_call_ceiling"]))
    if _num(data.get("instability_threshold")):
        into.instability_threshold = max(0.0, min(1.0, float(data["instability_threshold"])))
    for k in ("max_flip_rate", "min_consistency"):  # optional gates: null switches them off
        if k in data and data[k] is None:
            setattr(into, k, None)
        elif _num(data.get(k)):
            setattr(into, k, max(0.0, min(1.0, float(data[k]))))
    for k in ("min_confidence", "max_hard_break_rate", "max_error_rate",
              "quality_tolerance", "missing_axis_margin"):
        if isinstance(data.get(k), (int, float)) and not isinstance(data.get(k), bool):
            setattr(into, k, float(data[k]))
    for k in ("min_scored", "min_latency_samples"):
        if isinstance(data.get(k), int) and not isinstance(data.get(k), bool):
            setattr(into, k, max(0, data[k]))
    if data.get("latency_metric") in ("per_token", "p50"):
        into.latency_metric = data["latency_metric"]
    if isinstance(data.get("statistical_ties"), bool):
        into.statistical_ties = data["statistical_ties"]
    presets = data.get("weight_presets")
    if isinstance(presets, dict):
        for name, w in presets.items():
            if isinstance(w, dict) and all(isinstance(w.get(a), (int, float)) for a in ("quality", "cost", "latency")):
                into.weight_presets[name] = {a: float(w[a]) for a in ("quality", "cost", "latency")}


# The pre-2026-11-12 code defaults. Older settings files saved the WHOLE price table, so an
# entry equal to one of these is a stale copy of an old default, not a user edit -- drop it
# so the current public price wins. Anything else in a legacy file is a real user override.
_LEGACY_DEFAULT_PRICES_V1: Dict[str, tuple] = {
    "gpt-5.6-luna": (0.00022, 0.00132),
    "gpt-5.6-terra": (0.00220, 0.01320),
    "gpt-5.6-sol": (0.00500, 0.03000),
    "gpt-5.4": (0.00275, 0.01650),
    "claude-sonnet-4.5": (0.003, 0.015),
}


def _same_price(p: ModelPrice, inp: float, out: float) -> bool:
    return abs(p.input_per_1k - inp) < 1e-12 and abs(p.output_per_1k - out) < 1e-12


def price_overrides(s: Settings) -> Dict[str, Dict[str, Optional[float]]]:
    """Entries in the live table that differ from the public defaults (or aren't in them)."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for model, p in s.price_table.items():
        d = PRICE_TABLE.get(model)
        if d is None or not _same_price(p, d.input_per_1k, d.output_per_1k):
            out[model] = p.model_dump()
    return out


def _apply_saved_prices(into: Settings, data: Dict) -> None:
    def _put(model: str, price: Dict) -> None:
        try:
            base = PRICE_TABLE.get(model)
            p = ModelPrice(**price)
            if p.cache_read_per_1k is None and base is not None:
                p.cache_read_per_1k = base.cache_read_per_1k
            into.price_table[model] = p
        except (TypeError, ValueError):
            pass

    for model, price in (data.get("price_overrides") or {}).items():
        _put(model, price)
    # Legacy files (full table under "price_table"): keep only genuine user edits.
    for model, price in (data.get("price_table") or {}).items():
        if model in (data.get("price_overrides") or {}):
            continue
        legacy = _LEGACY_DEFAULT_PRICES_V1.get(model)
        try:
            p = ModelPrice(**price)
        except (TypeError, ValueError):
            continue
        if legacy is not None and _same_price(p, *legacy):
            continue  # stale copy of an old default
        cur = PRICE_TABLE.get(model)
        if cur is not None and _same_price(p, cur.input_per_1k, cur.output_per_1k):
            continue  # identical to the current default
        _put(model, price)


# Load any persisted overrides at import time so every entrypoint sees them.
load_settings()
