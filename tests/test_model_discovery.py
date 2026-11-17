"""Live model discovery for the candidate picker (modelshift/discovery.py)."""
import pytest

from modelshift import discovery
from modelshift.adapters import BedrockResponsesAdapter, LiteLLMResponsesAdapter, _chat_url
from modelshift.config import supported_efforts
from modelshift.discovery import DiscoveredModel


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(discovery, "_REGISTRY", {})
    monkeypatch.setattr(discovery, "_LOADED", False)
    monkeypatch.setattr(discovery, "_CACHE", {})
    yield


def _runtime():
    return [
        DiscoveredModel(name="gpt-5.6-luna", model_id="us.openai.gpt-5.6-luna", endpoint="runtime",
                        provider="OpenAI", family="openai"),                      # catalog
        DiscoveredModel(name="gpt-6-luna", model_id="us.openai.gpt-6-luna", endpoint="runtime",
                        provider="OpenAI", family="openai"),
        DiscoveredModel(name="nova-lite-v1:0", model_id="amazon.nova-lite-v1:0", endpoint="runtime",
                        provider="Amazon", family="converse"),
    ]


def test_short_names():
    assert discovery._short_name("us.openai.gpt-6-luna") == "gpt-6-luna"
    assert discovery._short_name("global.anthropic.claude-x-v1:0") == "claude-x-v1:0"
    assert discovery._short_name("amazon.nova-lite-v1:0") == "nova-lite-v1:0"


def test_available_matches_catalog_first_and_registers_extras():
    r = discovery.available("bedrock_runtime", legacy_model="gpt-4.1", runtime_lister=_runtime)
    names = [m["name"] for m in r["models"]]
    assert r["reason"] is None and r["listed_count"] == 3
    luna = next(m for m in r["models"] if m["name"] == "gpt-5.6-luna")
    assert luna["catalog"] and luna["suggested"] and luna["price"] and luna["efforts_checked"]
    new = next(m for m in r["models"] if m["name"] == "gpt-6-luna")
    assert not new["catalog"] and new["price"] is None and new["efforts"] is None and not new["efforts_checked"]
    nova = next(m for m in r["models"] if m["name"] == "nova-lite-v1:0")
    assert nova["effort_applies"] is False and nova["efforts"] == ["medium"]
    assert names.index("gpt-5.6-luna") < names.index("gpt-6-luna")      # catalog first
    assert discovery.get("gpt-6-luna").model_id == "us.openai.gpt-6-luna"
    assert discovery.get("gpt-5.6-luna") is None                          # catalog names never shadowed


def test_unreachable_endpoint_returns_catalog_with_reason():
    def boom():
        raise RuntimeError("AccessDenied")
    r = discovery.available("bedrock_runtime", runtime_lister=boom)
    assert "AccessDenied" in r["reason"]
    assert {m["name"] for m in r["models"]} >= {"gpt-5.6-luna", "claude-sonnet-4.5"}


def test_mantle_names_are_suffixed_and_called_as_chat_completions():
    r = discovery.available("bedrock_mantle", mantle_lister=lambda: ["openai.gpt-5.6-luna", "zai.glm-5",
                                                                     "mistral.voxtral-mini-3b-2507"])
    names = {m["name"] for m in r["models"]}
    assert "glm-5-mantle" in names and not any("voxtral" in n for n in names)
    dm = discovery.get("glm-5-mantle")
    assert dm.endpoint == "mantle" and dm.family == "openai" and dm.provider == "Z.AI"


def test_mantle_chat_url_by_model_id():
    base = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    assert _chat_url(base, "mantle", "openai.gpt-5.4") == base + "/chat/completions"
    assert _chat_url(base, "mantle", "zai.glm-5") == "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
    rt = "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
    assert _chat_url(rt, "runtime", "zai.glm-5") == rt + "/chat/completions"


def test_adapters_resolve_discovered_ids():
    discovery.available("bedrock_runtime", runtime_lister=_runtime)
    a = BedrockResponsesAdapter(transport=lambda p: {}, endpoint="runtime")
    assert a._model_ref("gpt-6-luna@low") == "us.openai.gpt-6-luna"
    assert a._model_ref("gpt-5.6-luna") == "us.openai.gpt-5.6-luna"
    discovery.register([DiscoveredModel(name="team-alias", model_id="team/alias@v2", endpoint="litellm")])
    assert LiteLLMResponsesAdapter(transport=lambda p: {})._model_ref("team-alias") == "team/alias@v2"


def test_orchestrator_routes_discovered_models():
    from modelshift.orchestrator import CandidateConfig, Orchestrator, RunConfig
    discovery.available("bedrock_runtime", runtime_lister=_runtime)
    discovery.available("bedrock_mantle", mantle_lister=lambda: ["zai.glm-5"])
    conv, rt, mant = (lambda p: {}), (lambda p: {}), (lambda p: {})
    orch = Orchestrator(candidate_transport=rt, mantle_transport=mant, converse_transport=conv)
    cfg = RunConfig(candidates=[CandidateConfig(model="x")])
    assert orch._adapter_for(cfg, "nova-lite-v1:0")._transport is conv        # non-OpenAI -> Converse
    assert orch._adapter_for(cfg, "gpt-6-luna@high")._transport is rt         # OpenAI -> /openai/v1 runtime
    a = orch._adapter_for(cfg, "glm-5-mantle")
    assert a._transport is mant and a._endpoint == "mantle"                   # listed on Mantle -> Mantle


def test_probe_records_accepted_efforts_and_validation_uses_them():
    discovery.available("bedrock_runtime", runtime_lister=_runtime)
    sent = []

    def transport(payload):
        sent.append(payload["reasoning"]["effort"])
        if payload["reasoning"]["effort"] in ("minimal", "xhigh"):
            raise RuntimeError("400 from x: Unsupported value: 'minimal' is not supported with this model.")
        return {"choices": [{"message": {"content": "OK"}}]}

    r = discovery.probe_efforts("gpt-6-luna", transport)
    assert sent == discovery.ALL_EFFORTS
    assert r["efforts"] == ["none", "low", "medium", "high"] and r["errors"] == {}
    assert supported_efforts("gpt-6-luna@low") == ["none", "low", "medium", "high"]
    # Saved: a fresh process (registry reloaded from disk) still knows the result.
    discovery._REGISTRY.clear()
    discovery._LOADED = False
    assert discovery.get("gpt-6-luna").efforts == ["none", "low", "medium", "high"]


def test_probe_reports_errors_that_are_not_rejections():
    discovery.available("bedrock_runtime", runtime_lister=_runtime)

    def down(payload):
        raise RuntimeError("503 Service Unavailable")
    r = discovery.probe_efforts("gpt-6-luna", down)
    assert r["efforts"] is None and "503" in r["errors"]["medium"]
    assert supported_efforts("gpt-6-luna") is None                           # still unknown, not validated


def test_converse_models_skip_the_probe():
    discovery.available("bedrock_runtime", runtime_lister=_runtime)
    r = discovery.probe_efforts("nova-lite-v1:0", lambda p: pytest.fail("must not call the model"))
    assert r["effort_applies"] is False and supported_efforts("nova-lite-v1:0") == ["medium"]


def test_api_available_and_probe(monkeypatch):
    from fastapi.testclient import TestClient
    from modelshift import api as api_mod
    monkeypatch.setattr(discovery, "_list_runtime", lambda region: _runtime())
    api_mod.configure_transports(candidate_transport=lambda p: {"choices": [{"message": {"content": "OK"}}]})
    c = TestClient(api_mod.app)
    r = c.get("/api/v1/models/available", params={"endpoint": "bedrock_runtime", "legacy": "gpt-4.1"})
    assert r.status_code == 200 and any(m["name"] == "gpt-6-luna" for m in r.json()["models"])
    assert c.get("/api/v1/models/available", params={"endpoint": "nope"}).status_code == 400
    p = c.post("/api/v1/models/probe-efforts", json={"name": "gpt-6-luna"})
    assert p.status_code == 200 and p.json()["efforts"] == discovery.ALL_EFFORTS
    assert c.post("/api/v1/models/probe-efforts", json={"name": "unknown-x"}).status_code == 404
