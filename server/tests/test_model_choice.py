"""Choosing what makes the pictures: settings and the server's routing."""
import json
import os
import sys

import pytest

from server import config, main, models


def test_defaults_are_the_local_klein_model(tmp_path):
    s = config.load_model_settings(tmp_path / "none.json", env={})
    assert s["provider"] == "local"
    assert s["openrouter_model"] == config.DEFAULT_OPENROUTER_MODEL
    assert s["openrouter_base_url"] == "https://openrouter.ai/api/v1"
    assert s["api_key"] == "" and s["api_key_source"] is None


def test_saved_choices_round_trip_and_merge(tmp_path):
    path = tmp_path / "config.json"
    config.save_port(9001, path)
    config.save_model_settings({"provider": "openrouter",
                                "openrouter_model": " openai/gpt-image ",
                                "openrouter_api_key": "sk-or-abc12345"}, path)
    s = config.load_model_settings(path, env={})
    assert s["provider"] == "openrouter"
    assert s["openrouter_model"] == "openai/gpt-image"
    assert s["api_key"] == "sk-or-abc12345" and s["api_key_source"] == "saved"
    assert config.load_port(path) == 9001  # merged, not replaced


def test_empty_values_reset_and_bad_provider_is_ignored(tmp_path):
    path = tmp_path / "config.json"
    config.save_model_settings({"openrouter_model": "me/my-model",
                                "openrouter_api_key": "sk-or-1"}, path)
    config.save_model_settings({"openrouter_model": "",
                                "openrouter_api_key": "",
                                "provider": "nonsense"}, path)
    s = config.load_model_settings(path, env={})
    assert s["openrouter_model"] == config.DEFAULT_OPENROUTER_MODEL
    assert s["api_key"] == ""
    assert s["provider"] == "local"


def test_the_environment_key_wins(tmp_path):
    path = tmp_path / "config.json"
    config.save_model_settings({"openrouter_api_key": "sk-saved"}, path)
    s = config.load_model_settings(path, env={"OPENROUTER_API_KEY": "sk-env"})
    assert s["api_key"] == "sk-env" and s["api_key_source"] == "env"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_a_saved_key_is_owner_only(tmp_path):
    path = tmp_path / "config.json"
    config.save_model_settings({"openrouter_api_key": "sk-or-secret"}, path)
    assert os.stat(path).st_mode & 0o077 == 0


def test_mask_key_never_shows_the_key():
    assert config.mask_key("sk-or-v1-0123456789abcd") == "…abcd"
    assert config.mask_key("short") == "set"
    assert config.mask_key("") == "" and config.mask_key(None) == ""


def test_config_env_override_points_at_a_file(tmp_path):
    target = tmp_path / "mine.json"
    assert config.config_path(env={"PIXAGON_CONFIG": str(target)}) == target


# ---- routing in the server

@pytest.fixture
def registry():
    models.reset()
    yield
    models.reset()


def use(settings: dict):
    config.save_model_settings(settings)


def test_active_model_follows_the_provider(registry):
    models.register("klein", object)
    models.register("openrouter", object)
    assert main.active_model() == "klein"
    use({"provider": "openrouter"})
    assert main.active_model() == "openrouter"


def test_active_model_falls_back_when_the_cloud_is_not_registered(registry):
    models.register("klein", object)
    use({"provider": "openrouter"})
    assert main.active_model() == "klein"


def test_backend_label_names_the_model():
    assert main.backend_label() == "FLUX.2-klein-4B"
    use({"provider": "openrouter", "openrouter_model": "google/x-image"})
    assert main.backend_label() == "OpenRouter · google/x-image"


def test_default_registration_offers_both(registry):
    main._register_default_models()
    assert {"klein", "openrouter"} <= set(models._factories)


def test_the_openrouter_factory_reads_the_settings(registry, monkeypatch):
    use({"provider": "openrouter", "openrouter_model": "a/b",
         "openrouter_api_key": "sk-or-xyz"})
    main._register_default_models()
    pipe = models._factories["openrouter"]()
    assert pipe.model == "a/b" and pipe.api_key == "sk-or-xyz"
    assert pipe.base_url == config.DEFAULT_OPENROUTER_URL


def test_the_openrouter_factory_fails_clearly_without_a_key(registry):
    use({"provider": "openrouter"})
    main._register_default_models()
    with pytest.raises(RuntimeError) as err:
        models._factories["openrouter"]()
    assert "API key" in str(err.value)


def test_ping_reports_ready_and_the_backend(registry):
    import asyncio

    class FakePipe:
        pass
    models.register("klein", FakePipe)
    models.register("openrouter", FakePipe)
    use({"provider": "openrouter", "openrouter_model": "g/img"})
    models.get("openrouter")

    class WS:
        def __init__(self):
            self.sent = []

        def __aiter__(self):
            async def gen():
                yield json.dumps({"type": "ping"})
            return gen()

        async def send(self, msg):
            self.sent.append(json.loads(msg))

    ws = WS()
    asyncio.run(main._serve_connection(ws))
    assert ws.sent == [{"type": "pong", "model": "ready",
                        "backend": "OpenRouter · g/img"}]
