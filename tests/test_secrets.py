import os
import stat

from modelshift import config


def test_resolve_key_env_wins(monkeypatch):
    monkeypatch.setenv("MODELSHIFT_LITELLM_KEY", "env-key")
    config.DEFAULT_SETTINGS.litellm_key = "persisted-key"
    try:
        assert config.resolve_litellm_key() == "env-key"
    finally:
        config.DEFAULT_SETTINGS.litellm_key = ""


def test_resolve_key_falls_back_to_persisted(monkeypatch):
    monkeypatch.delenv("MODELSHIFT_LITELLM_KEY", raising=False)
    monkeypatch.delenv("MODELSHIFT_LITELLM_KEY_SECRET_ARN", raising=False)
    config.DEFAULT_SETTINGS.litellm_key = "persisted-key"
    try:
        assert config.resolve_litellm_key() == "persisted-key"
    finally:
        config.DEFAULT_SETTINGS.litellm_key = ""


def test_resolve_key_empty_when_none(monkeypatch):
    monkeypatch.delenv("MODELSHIFT_LITELLM_KEY", raising=False)
    monkeypatch.delenv("MODELSHIFT_LITELLM_KEY_SECRET_ARN", raising=False)
    config.DEFAULT_SETTINGS.litellm_key = ""
    assert config.resolve_litellm_key() == ""


def test_settings_file_is_chmod_600(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSHIFT_SETTINGS_PATH", str(tmp_path / "settings.json"))
    config.DEFAULT_SETTINGS.litellm_key = "secret-on-disk"
    try:
        config.save_settings()
        p = tmp_path / "settings.json"
        assert p.exists()
        mode = stat.S_IMODE(os.stat(p).st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
    finally:
        config.DEFAULT_SETTINGS.litellm_key = ""
