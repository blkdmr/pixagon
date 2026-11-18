"""Every test gets a private, empty settings file: none of them may read or
change the real one (provider, API key, port...) of whoever runs them."""
import pytest


@pytest.fixture(autouse=True)
def _private_settings(tmp_path_factory, monkeypatch):
    folder = tmp_path_factory.mktemp("settings")
    monkeypatch.setenv("PIXAGON_CONFIG", str(folder / "config.json"))
    monkeypatch.setenv("PIXAGON_DATA", str(folder / "data"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
