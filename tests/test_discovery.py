import json

from modelshift.adapters import discover_litellm_models


class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, headers=None):
        return self._resp


def _patch_httpx(monkeypatch, resp):
    import httpx
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: _FakeClient(resp))


def test_discover_parses_openai_models_shape(monkeypatch):
    payload = {"data": [{"id": "gpt-5.6-luna"}, {"id": "acme-terra-v2"},
                        {"id": "claude-sonnet-4.5"}, {"id": "gpt-5.6-luna"}]}  # dup
    _patch_httpx(monkeypatch, _FakeResp(200, payload))
    models = discover_litellm_models("http://127.0.0.1:4000", "sk-x")
    assert models == ["gpt-5.6-luna", "acme-terra-v2", "claude-sonnet-4.5"]  # de-duped, ordered


def test_discover_raises_on_http_error(monkeypatch):
    _patch_httpx(monkeypatch, _FakeResp(403, {"error": "forbidden"}))
    try:
        discover_litellm_models("http://127.0.0.1:4000", "sk-x")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "403" in str(e)


def test_litellm_models_endpoint_degrades_gracefully(monkeypatch):
    # Point at a dead port so the real httpx call fails -> endpoint returns [] + reason.
    from fastapi.testclient import TestClient
    from modelshift import api as api_mod, config
    monkeypatch.delenv("MODELSHIFT_AUTH_MODE", raising=False)
    orig_base = config.DEFAULT_SETTINGS.litellm_base
    config.DEFAULT_SETTINGS.litellm_base = "http://127.0.0.1:1"  # unreachable
    try:
        api_mod.configure_transports(candidate_transport=lambda p: {"output_text": "x", "usage": {}},
                                     judge_transport=None)
        c = TestClient(api_mod.app)
        r = c.get("/api/v1/litellm/models")
        assert r.status_code == 200
        body = r.json()
        assert body["models"] == []
        assert body["reason"]  # a non-empty reason string
    finally:
        config.DEFAULT_SETTINGS.litellm_base = orig_base  # never leak to other tests
