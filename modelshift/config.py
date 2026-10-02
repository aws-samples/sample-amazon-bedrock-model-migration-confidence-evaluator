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
    runtime_supported: bool = True   # False => mantle-only (e.g. GPT-5.4)
    structured_outputs: bool = False  # per AWS model card (Luna yes, Terra no)


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
        # GPT-5.4 is bedrock-mantle ONLY (no us. prefix, not on bedrock-runtime).
        bedrock_model_id="openai.gpt-5.4",
        mantle_model_id="openai.gpt-5.4",
        litellm_model_group="gpt-5.4",
        like_for_like=True,
        runtime_supported=False,
    ),
    "claude-sonnet-4.5": ModelCatalogEntry(
        name="claude-sonnet-4.5",
        # Anthropic on Bedrock — uses the Converse API, not the OpenAI /openai/v1 path.
        bedrock_model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        mantle_model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        litellm_model_group="claude-sonnet-4.5",
        family="anthropic",
        structured_outputs=True,
    ),
}


# --------------------------------------------------------------------------- #
# Price table — real AWS Bedrock in-region short-context rates (USD/1K tokens),
# from the OpenAI model cards. Editable in Settings. Reasoning billed at output.
# --------------------------------------------------------------------------- #
class ModelPrice(BaseModel):
    input_per_1k: float
    output_per_1k: float


PRICE_TABLE: Dict[str, ModelPrice] = {
    "gpt-5.6-luna": ModelPrice(input_per_1k=0.00022, output_per_1k=0.00132),
    "gpt-5.6-terra": ModelPrice(input_per_1k=0.00220, output_per_1k=0.01320),
    "gpt-5.6-sol": ModelPrice(input_per_1k=0.00500, output_per_1k=0.03000),
    "gpt-5.4": ModelPrice(input_per_1k=0.00275, output_per_1k=0.01650),
    "claude-sonnet-4.5": ModelPrice(input_per_1k=0.003, output_per_1k=0.015),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> float:
    price = PRICE_TABLE.get(model)
    if price is None:
        return 0.0
    return (
        input_tokens / 1000.0 * price.input_per_1k
        + (output_tokens + reasoning_tokens) / 1000.0 * price.output_per_1k
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
    default_parallelism: int = 6
    # Low-signal caveat thresholds (tunable, NFR-EXT-1). A caveat is advisory only
    # — it never blocks a run. Default 10 so a 15-row smoke test does not trip it.
    small_sample_threshold: int = 10
    high_redaction_threshold: float = 0.5
    # LiteLLM proxy connection (used when a run's call path is LiteLLM). The base
    # URL is non-secret; the key is a secret (persisted but never returned by the API).
    litellm_base: str = "http://127.0.0.1:4000"
    litellm_key: str = ""


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
        "small_sample_threshold": s.small_sample_threshold,
        "high_redaction_threshold": s.high_redaction_threshold,
        "litellm_base": s.litellm_base,
        "litellm_key": s.litellm_key,
        "price_table": {k: v.model_dump() for k, v in s.price_table.items()},
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
    if isinstance(data.get("default_parallelism"), int):
        into.default_parallelism = data["default_parallelism"]
    if isinstance(data.get("small_sample_threshold"), int):
        into.small_sample_threshold = data["small_sample_threshold"]
    if isinstance(data.get("high_redaction_threshold"), (int, float)):
        into.high_redaction_threshold = float(data["high_redaction_threshold"])
    if isinstance(data.get("litellm_base"), str) and data["litellm_base"]:
        into.litellm_base = data["litellm_base"]
    if isinstance(data.get("litellm_key"), str):
        into.litellm_key = data["litellm_key"]
    for model, price in (data.get("price_table") or {}).items():
        try:
            into.price_table[model] = ModelPrice(**price)
        except (TypeError, ValueError):
            pass
    return into


# Load any persisted overrides at import time so every entrypoint sees them.
load_settings()
