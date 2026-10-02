"""Stage 1 tests: data contracts + config-driven policy."""
from modelshift.config import (
    DEFAULT_RUBRIC,
    MODEL_CATALOG,
    candidates_for,
    estimate_cost,
)
from modelshift.schemas import (
    CompatLabel,
    DropReason,
    IngestionReport,
    NormalizedSample,
    SampleRequest,
    SampleSource,
    Shape,
    VerdictBand,
    Golden,
    GoldenKind,
    Message,
)


def test_normalized_sample_roundtrip():
    s = NormalizedSample(
        id="chatcmpl-1",
        source=SampleSource(shape=Shape.SPENDLOGS, object_uri="upload:x"),
        legacy_model="gpt-4.1",
        request=SampleRequest(input_messages=[Message(role="user", content="hi")]),
        golden=Golden(kind=GoldenKind.TEXT, text="hello"),
        evaluable=True,
    )
    assert s.model_dump()["golden"]["text"] == "hello"


def test_ingestion_report_reconciles():
    r = IngestionReport(
        found=10,
        evaluable=6,
        dropped={DropReason.REDACTED: 3, DropReason.NO_PROMPT: 1},
        skipped_objects=0,
    )
    assert r.reconciles()


def test_ingestion_report_does_not_reconcile_when_counts_wrong():
    r = IngestionReport(found=10, evaluable=6, dropped={DropReason.REDACTED: 1})
    assert not r.reconciles()


def test_matrix_lookup_gpt41():
    entry = candidates_for("gpt-4.1")
    assert entry is not None
    assert entry.default_start == "gpt-5.6-luna"
    assert "gpt-5.6-terra" in entry.candidates
    assert "gpt-5.6-sol" in entry.candidates


def test_matrix_lookup_strips_provider_prefix():
    # Azure-logged and Bedrock-addressed ids both normalize to the matrix key.
    assert candidates_for("azure/gpt-5.4") is not None
    assert candidates_for("bedrock/converse/us.openai.gpt-5.4") is not None


def test_matrix_gpt54_like_for_like():
    entry = candidates_for("gpt-5.4")
    assert entry.default_start == "gpt-5.4"
    assert MODEL_CATALOG["gpt-5.4"].like_for_like is True


def test_matrix_unmapped_returns_none():
    assert candidates_for("gpt-3.5-turbo") is None


def test_estimate_cost_uses_output_rate_for_reasoning():
    # 1000 input + 1000 output + 1000 reasoning tokens for luna
    c = estimate_cost("gpt-5.6-luna", 1000, 1000, 1000)
    # input 0.00022 + (output+reasoning 2000/1000 * 0.00132) = 0.00022 + 0.00264
    assert abs(c - 0.00286) < 1e-9


def test_gpt54_is_mantle_only_model_id():
    # GPT-5.4 is bedrock-mantle only: model id has NO us. prefix.
    assert MODEL_CATALOG["gpt-5.4"].bedrock_model_id == "openai.gpt-5.4"


def test_rubric_bands():
    r = DEFAULT_RUBRIC
    assert r.band_for(90, 0.0) == VerdictBand.SAFE_DROP_IN
    assert r.band_for(70, 0.0) == VerdictBand.DROP_IN_WITH_TUNING
    assert r.band_for(50, 0.0) == VerdictBand.NEEDS_WORK
    assert r.band_for(10, 0.0) == VerdictBand.NOT_A_DROP_IN


def test_rubric_high_hard_break_caps_band():
    # Even high confidence can't be "safe" with a high hard-break rate.
    assert DEFAULT_RUBRIC.band_for(95, 0.5) == VerdictBand.NEEDS_WORK


def test_rubric_confidence_weights_ordered():
    r = DEFAULT_RUBRIC
    assert (
        r.confidence_weight(CompatLabel.COMPATIBLE)
        > r.confidence_weight(CompatLabel.COMPATIBLE_WITH_DRIFT)
        > r.confidence_weight(CompatLabel.PARTIAL)
        >= r.confidence_weight(CompatLabel.INCOMPATIBLE)
    )
