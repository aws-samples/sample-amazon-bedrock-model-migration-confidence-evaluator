"""Core data contracts for ModelShift.

Mirrors the spec set:
  - NormalizedSample        (Doc 2 §5)
  - CandidateResult         (Doc 3 §3.3)
  - PairScore               (Doc 3 §4)
  - CandidateVerdict        (Doc 3 §4.3)
  - Remediation             (Doc 3 §6.4)
  - IngestionReport         (Doc 2 §6.3 / Doc 5 §4.2)

These are the internal contracts AND the API payload shapes (Doc 5 §3).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums / vocabularies
# --------------------------------------------------------------------------- #
class Shape(str, Enum):
    """Recognized LiteLLM row shapes (Doc 2 §3)."""

    SPENDLOGS = "spendlogs"
    WRAPPED = "wrapped"
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    # App-side capture record: flat system_prompt / user_prompt / response strings
    # (one call per object, e.g. a custom Azure OpenAI request logger).
    PROMPT_CAPTURE = "prompt_capture"
    # Bare chat transcript: a JSON array of messages (e.g. a serialized Semantic Kernel
    # ChatHistory: [{"Content": ..., "Role": {"Label": "user"}}, ...]). One array = one call;
    # the final assistant message is the golden.
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"  # Shape E — never evaluable


class Container(str, Enum):
    """Container formats the parser auto-detects (Doc 2 §2.2)."""

    JSONL = "jsonl"
    JSON_ARRAY = "json_array"
    WRAPPED_DATA = "wrapped_data"
    SINGLE_OBJECT = "single_object"  # one (possibly pretty-printed) JSON object = one row


class GoldenKind(str, Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    REFUSAL = "refusal"
    JSON = "json"
    EMPTY = "empty"


class Modality(str, Enum):
    TEXT = "text"
    CONTAINS_IMAGE = "contains_image"
    CONTAINS_AUDIO = "contains_audio"


class DropReason(str, Enum):
    """Mutually-exclusive drop/block reasons (Doc 2 §6.2)."""

    REDACTED = "redacted"
    EMPTY_RESPONSE = "empty_response"
    NO_PROMPT = "no_prompt"
    TOOL_ONLY_NO_TEXT = "tool_only_no_text"
    TOOL_CALLS_UNSUPPORTED = "tool_calls_unsupported"
    UNSUPPORTED_MODALITY = "unsupported_modality"
    STREAM_FRAGMENT = "stream_fragment"
    NON_EVAL_CALL_TYPE = "non_eval_call_type"
    PARSE_ERROR = "parse_error"
    UNKNOWN_SHAPE = "unknown_shape"


class ReasoningEffort(str, Enum):
    # Ordered lowest -> highest. Not every model accepts every value: see
    # ModelCatalogEntry.reasoning_efforts (probed live against Bedrock).
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class CompatLabel(str, Enum):
    """Per-sample compatibility label (Doc 3 §4.2)."""

    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_DRIFT = "compatible_with_drift"
    PARTIAL = "partial"
    INCOMPATIBLE = "incompatible"


class VerdictBand(str, Enum):
    """Aggregate per-candidate verdict band (Doc 3 §4.3)."""

    SAFE_DROP_IN = "safe_drop_in"
    DROP_IN_WITH_TUNING = "drop_in_with_prompt_tuning"
    NEEDS_WORK = "needs_work"
    NOT_A_DROP_IN = "not_a_drop_in"


class CallPath(str, Enum):
    BEDROCK = "bedrock"
    LITELLM = "litellm"


class BedrockEndpoint(str, Enum):
    """Which Bedrock OpenAI-compatible endpoint to call (AWS model cards).

    RUNTIME: https://bedrock-runtime.{region}.amazonaws.com/openai/v1 with a
             us.openai.* cross-Region profile (Luna/Terra/Sol).
    MANTLE:  https://bedrock-mantle.{region}.api.aws/openai/v1 with the plain
             openai.* id — required only for mantle-only models.
    """

    RUNTIME = "runtime"
    MANTLE = "mantle"


# --------------------------------------------------------------------------- #
# NormalizedSample (Doc 2 §5)
# --------------------------------------------------------------------------- #
class SampleSource(BaseModel):
    shape: Shape
    container: Optional[Container] = None
    object_uri: str = "upload:unknown"
    row_index: int = 0


class Message(BaseModel):
    role: str
    content: str


class ToolCall(BaseModel):
    name: str
    arguments: Any = None


class SampleRequest(BaseModel):
    instructions: Optional[str] = None
    input_messages: List[Message] = Field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    modality: Modality = Modality.TEXT
    had_tool_calls: bool = False


class Golden(BaseModel):
    kind: GoldenKind = GoldenKind.EMPTY
    text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    raw: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    spend: Optional[float] = None


class NormalizedSample(BaseModel):
    id: str
    source: SampleSource
    legacy_model: Optional[str] = None
    request: SampleRequest
    golden: Golden
    usage: Usage = Field(default_factory=Usage)
    original_latency_ms: Optional[int] = None
    tags: Dict[str, Any] = Field(default_factory=dict)
    evaluable: bool = False
    eval_block_reason: Optional[DropReason] = None
    flags: List[str] = Field(default_factory=list)


class DroppedRow(BaseModel):
    """A row that failed evaluability — kept for accounting (Doc 2 §6.3)."""

    id: str
    source: SampleSource
    eval_block_reason: DropReason


# --------------------------------------------------------------------------- #
# CandidateResult (Doc 3 §3.3)
# --------------------------------------------------------------------------- #
class CandidateOutput(BaseModel):
    kind: GoldenKind = GoldenKind.TEXT
    text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class CandidateSettings(BaseModel):
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    dropped_params: List[str] = Field(default_factory=list)


class CandidateResult(BaseModel):
    sample_id: str
    model: str
    settings: CandidateSettings = Field(default_factory=CandidateSettings)
    output: CandidateOutput = Field(default_factory=CandidateOutput)
    usage: Usage = Field(default_factory=Usage)
    cost_estimate: float = 0.0
    # False when the model has no configured price: cost_estimate is then 0.0 for
    # compatibility but means UNKNOWN, never "free".
    cost_known: bool = True
    # Duration of the SUCCESSFUL attempt only (retry back-off is excluded).
    latency_ms: Optional[int] = None
    attempts: int = 1           # 1 = the first try succeeded
    from_cache: bool = False    # served from the in-process output cache, not timed in this call
    status: str = "ok"  # ok | error
    error: Optional[str] = None
    cache_key: Optional[str] = None
    replay_index: int = 0       # which replay of the request (0..N-1); 0 when replays are off


# --------------------------------------------------------------------------- #
# PairScore (Doc 3 §4)
# --------------------------------------------------------------------------- #
class DimensionScore(BaseModel):
    score: float = 0.0  # 0..1
    reason: str = ""


class PairScore(BaseModel):
    sample_id: str
    model: str
    label: CompatLabel
    d1_semantic: DimensionScore = Field(default_factory=DimensionScore)
    d2_format: DimensionScore = Field(default_factory=DimensionScore)
    d3_factual: DimensionScore = Field(default_factory=DimensionScore)
    d4_verbosity: DimensionScore = Field(default_factory=DimensionScore)
    d5_instruction: DimensionScore = Field(default_factory=DimensionScore)
    d6_tool_call: Optional[DimensionScore] = None
    judge_reason: Optional[str] = None
    judge_rationale: Optional[str] = None
    hard_break: bool = False
    overridden: bool = False
    override_label: Optional[CompatLabel] = None
    override_note: Optional[str] = None
    replay_index: int = 0       # which replay's answer this scores (0 when replays are off)


# --------------------------------------------------------------------------- #
# CandidateVerdict (Doc 3 §4.3)
# --------------------------------------------------------------------------- #
class CandidateVerdict(BaseModel):
    model: str
    migration_confidence: float = 0.0  # 0..100
    verdict_band: VerdictBand = VerdictBand.NOT_A_DROP_IN
    # Which transport/endpoint this candidate actually used for the run:
    # "bedrock-runtime" | "bedrock-mantle" | "litellm" | "" (unknown/not run).
    endpoint: str = ""
    distribution: Dict[CompatLabel, int] = Field(default_factory=dict)
    dimension_averages: Dict[str, float] = Field(default_factory=dict)
    top_failure_reasons: List[Dict[str, Any]] = Field(default_factory=list)
    hard_break_rate: float = 0.0
    cost_total: float = 0.0
    cost_known: bool = True     # False if any successful call had no configured price
    latency_avg_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    scored: int = 0
    errored: int = 0
    # ---- Multi-objective recommendation metrics (all additive; None = not computable) ---- #
    ok_calls: int = 0
    cost_per_request: Optional[float] = None          # cost_total / ok_calls (None if price unknown)
    cost_per_1k_requests: Optional[float] = None
    cost_vs_current_pct: Optional[float] = None       # -48.0 = 48% cheaper than the current model
    projected_monthly_cost: Optional[float] = None    # only when a monthly volume is configured
    avg_output_tokens: Optional[float] = None         # billed output tokens per ok call (incl. reasoning)
    avg_reasoning_tokens: Optional[float] = None
    output_token_ratio: Optional[float] = None        # candidate output / current-model output
    ms_per_output_token_p50: Optional[float] = None
    latency_samples: int = 0                          # fresh (non-cached) timed successful calls
    latency_low_confidence: bool = True
    error_rate: float = 0.0
    eligible: bool = False
    ineligible_reasons: List[str] = Field(default_factory=list)
    # ---- Request replays / D7 Consistency (replay-consistency spec; additive) ---- #
    replays: int = 1                                  # answers per request (1 = replays off)
    requests_scored: int = 0                          # distinct requests with >= 1 scored answer
    consistency: Optional[float] = None               # mean D7 (0..1); None = not measured
    consistency_measured: int = 0                     # requests with a D7 value
    consistency_low_confidence: bool = False          # too few measured requests to rely on
    unstable_requests: int = 0
    flip_rate: Optional[float] = None                 # share of requests that both break and don't
    unstable: bool = False                            # mean D7 below the instability threshold
    latency_spread_p50: Optional[float] = None        # median per-request max/min latency ratio


# --------------------------------------------------------------------------- #
# Remediation (Doc 3 §6.4)
# --------------------------------------------------------------------------- #
class RemediationChange(BaseModel):
    type: str  # prompt_edit | reasoning_effort | format_schema | swap_candidate
    # reasoning_effort: before/after are effort values (e.g. "medium" -> "low").
    # prompt_edit / format_schema: after is the instruction added to each request's
    # system prompt (before is None - every request keeps its own system prompt).
    before: Optional[str] = None
    after: Optional[str] = None
    placement: Optional[str] = None  # append | prepend (instruction changes only)


class RemediationTarget(BaseModel):
    scope: str = "cluster"  # cluster | run
    reason: Optional[str] = None
    sample_ids: List[str] = Field(default_factory=list)


class RemediationResult(BaseModel):
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    moved: Dict[str, int] = Field(default_factory=dict)
    # Whole-candidate view: the run's confidence for this candidate, and the projection
    # with re-tested evaluations replaced by their new scores.
    confidence_run_before: Optional[float] = None
    confidence_run_after: Optional[float] = None
    regression_checked: int = 0  # Compatible evaluations re-run to catch regressions
    regression_sample: int = 0   # how many were requested (-1 = all)
    # Replays: re-tests send each request the run's N times; D7 over the re-tested requests.
    replays: int = 1
    consistency_before: Optional[float] = None
    consistency_after: Optional[float] = None
    # Re-test bookkeeping (set by Orchestrator.retest_remediation):
    status: str = "done"  # running | done | failed | cancelled | interrupted
    done: int = 0
    total: int = 0
    change: Optional[RemediationChange] = None  # exactly what was applied
    items: List[Dict[str, Any]] = Field(default_factory=list)  # per-evaluation before/after
    cost: float = 0.0
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class Remediation(BaseModel):
    remediation_id: str
    run_id: str
    candidate: str
    target: RemediationTarget
    change: RemediationChange
    result: Optional[RemediationResult] = None
    # Recommender-only fields (a suggestion not yet applied):
    evidence_count: int = 0
    expected_effect: Optional[str] = None
    applied: bool = False


# --------------------------------------------------------------------------- #
# IngestionReport (Doc 2 §6.3 / Doc 5 §4.2)
# --------------------------------------------------------------------------- #
class DetectedModel(BaseModel):
    model: str
    rows: int


class IngestionReport(BaseModel):
    found: int = 0
    evaluable: int = 0
    dropped: Dict[DropReason, int] = Field(default_factory=dict)
    skipped_objects: int = 0
    detected_models: List[DetectedModel] = Field(default_factory=list)
    shape_mix: Dict[Shape, float] = Field(default_factory=dict)
    redaction_rate: float = 0.0
    # Evaluable-row counts per team / per user (for targeting a subset).
    teams: Dict[str, int] = Field(default_factory=dict)
    users: Dict[str, int] = Field(default_factory=dict)

    def reconciles(self) -> bool:
        """found = evaluable + Σ dropped + skipped_objects (Doc 2 §6.3)."""
        return self.found == self.evaluable + sum(self.dropped.values()) + self.skipped_objects


# --------------------------------------------------------------------------- #
# Multi-objective recommendation (spec: multi-objective-recommendation, design §2.3)
# --------------------------------------------------------------------------- #
QUALITY_WEIGHT_FLOOR = 0.2


class WeightPreset(str, Enum):
    BALANCED = "balanced"
    QUALITY_FIRST = "quality_first"
    COST_FIRST = "cost_first"
    LATENCY_FIRST = "latency_first"
    CUSTOM = "custom"


class Weights(BaseModel):
    preset: WeightPreset = WeightPreset.BALANCED
    quality: float = 0.50
    cost: float = 0.30
    latency: float = 0.20

    def normalised(self) -> "Weights":
        """Drop negatives, sum to 1 (ratios preserved), and keep quality >= QUALITY_WEIGHT_FLOOR
        so 'best overall' is never chosen without regard to fit. All-zero input -> Balanced."""
        q, c, lat = (max(float(x or 0.0), 0.0) for x in (self.quality, self.cost, self.latency))
        total = q + c + lat
        if total <= 0:
            q, c, lat, total = 0.50, 0.30, 0.20, 1.0
        q, c, lat = q / total, c / total, lat / total
        if q < QUALITY_WEIGHT_FLOOR:
            rest = c + lat
            share = 1.0 - QUALITY_WEIGHT_FLOOR
            c, lat = ((c / rest * share, lat / rest * share) if rest > 0 else (share / 2, share / 2))
            q = QUALITY_WEIGHT_FLOOR
        return Weights(preset=self.preset, quality=round(q, 6), cost=round(c, 6), latency=round(lat, 6))


class ProfileKind(str, Enum):
    BEST_OVERALL = "best_overall"
    BEST_QUALITY = "best_quality"
    MOST_ECONOMICAL = "most_economical"
    LOWEST_LATENCY = "lowest_latency"


class CandidateAxes(BaseModel):
    model: str
    quality: float                    # 0..100 (= Migration Confidence, shown as-is)
    # 0..100 quality used in the composite: best confidence = 100, the edge of the quality
    # tolerance window = 0. Puts quality on the same spread as cost/latency so a 10-point fit
    # gap isn't outweighed by any price difference. Still pure fit (no dimension blending).
    quality_score: Optional[float] = None
    cost: Optional[float] = None      # 0..100, ratio to the cheapest; None if price unknown
    latency: Optional[float] = None   # 0..100, ratio to the fastest; None if too few timed calls
    composite: Optional[float] = None
    complete: bool = True             # has every axis whose weight > 0
    pareto_efficient: bool = False
    dominated_by: Optional[str] = None
    consistency: Optional[float] = None   # mean D7 when replays were on; display only, not in composite


class Profile(BaseModel):
    kind: ProfileKind
    model: Optional[str] = None       # None = no candidate qualifies
    reason: str = ""                  # why chosen, or why none qualifies
    tradeoff: Optional[str] = None    # vs Best overall; None for Best overall itself
    # Relative to Best overall: quality_pts (confidence points), cost_pct / latency_pct
    # (fractional change, e.g. -0.62 = 62% cheaper / faster).
    deltas: Dict[str, Optional[float]] = Field(default_factory=dict)


class Recommendation(BaseModel):
    weights: Weights
    latency_metric: str = "per_token"
    quality_tolerance: float = 10.0
    eligible_models: List[str] = Field(default_factory=list)
    ineligible: Dict[str, List[str]] = Field(default_factory=dict)   # model -> gate reasons
    axes: List[CandidateAxes] = Field(default_factory=list)
    profiles: List[Profile] = Field(default_factory=list)           # always 4, ProfileKind order
    none_eligible: bool = False
    closest_by_fit: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    latency_disclaimer: str = ""
    # Short "why this pick" lines per profile kind (and "none" when nobody passed the gate).
    explanations: Dict[str, List[str]] = Field(default_factory=dict)

    def profile(self, kind: ProfileKind) -> Optional[Profile]:
        return next((p for p in self.profiles if p.kind == kind), None)
