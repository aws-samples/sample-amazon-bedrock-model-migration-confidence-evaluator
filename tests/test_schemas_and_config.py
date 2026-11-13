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
    # Bedrock reports reasoning as a SUBSET of output tokens: 1000 output (incl. 400 reasoning)
    # is billed as 1000 output tokens, not 1400.
    c = estimate_cost("gpt-5.6-luna", 1000, 1000, 400)
    assert abs(c - (0.00022 + 1.0 * 0.00132)) < 1e-9
    # If a source counts reasoning separately (reasoning > output), both are billed.
    c2 = estimate_cost("gpt-5.6-luna", 1000, 100, 900)
    assert abs(c2 - (0.00022 + 1.0 * 0.00132)) < 1e-9


def test_price_table_matches_public_bedrock_rates():
    # Public Bedrock model-card list prices, USD per 1M tokens (input, cache read, output).
    expected = {
        "gpt-5.6-luna": (0.22, 0.022, 1.32),
        "gpt-5.6-terra": (2.20, 0.22, 13.20),
        "gpt-5.6-sol": (4.40, 0.44, 22.00),
        "gpt-5.5": (5.50, 0.55, 33.00),
        "gpt-5.4": (2.75, 0.275, 16.50),
    }
    from modelshift.config import PRICE_TABLE
    for model, (inp, cached, out) in expected.items():
        p = PRICE_TABLE[model]
        assert abs(p.input_per_1k * 1000 - inp) < 1e-9, model
        assert abs(p.cache_read_per_1k * 1000 - cached) < 1e-9, model
        assert abs(p.output_per_1k * 1000 - out) < 1e-9, model
    # GPT-5.5 is no longer costed as $0
    assert estimate_cost("gpt-5.5", 1000, 1000) > 0


def test_gpt54_model_ids_runtime_and_mantle():
    # GPT-5.4 is on bedrock-runtime via the us. inference profile, and on mantle via openai.*.
    e = MODEL_CATALOG["gpt-5.4"]
    assert e.bedrock_model_id == "us.openai.gpt-5.4"
    assert e.mantle_model_id == "openai.gpt-5.4"
    assert e.runtime_supported is True


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
