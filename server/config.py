# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

"""Settings shared by the launcher and the server, in one json file."""
import json
import os
import pathlib
import sys

DEFAULT_PORT = 8765
HOST = "127.0.0.1"

VRAM_MODES = ("auto", "bf16", "offload")  # fp8 retired; a saved "fp8" -> auto
DEFAULT_VRAM_MODE = "auto"

SETTING_KEYS = ("port", "vram_mode", "root", "python", "aseprite_dir",
                "models_dir")
DEFAULT_MODELS_DIR = "models"


def user_config_root(system: str | None = None,
                     env=None) -> pathlib.Path:
    """The per-user settings folder each OS expects (APPDATA, Application
    Support, XDG), with a home-folder fallback when the variable is unset."""
    system = system or sys.platform
    env = os.environ if env is None else env
    home = pathlib.Path(env.get("HOME") or pathlib.Path.home())
    if system == "win32":
        base = env.get("APPDATA")
        return pathlib.Path(base) if base else home
    if system == "darwin":
        return home / "Library" / "Application Support"
    xdg = env.get("XDG_CONFIG_HOME")
    return pathlib.Path(xdg) if xdg else home / ".config"


CONFIG_ENV = "PIXAGON_CONFIG"  # a settings file of your own (and tests')
DATA_ENV = "PIXAGON_DATA"      # a data folder of your own (dev, tests)


def app_folder(system: str | None = None, env=None) -> pathlib.Path:
    """Pixagon's per-user folder: settings, .venv, uv, models, logs."""
    system = system or sys.platform
    # Linux folders are lowercase by convention; elsewhere keep the brand name
    folder = "pixagon" if system.startswith("linux") else "Pixagon"
    return user_config_root(system, env) / folder


def config_path(system: str | None = None, env=None) -> pathlib.Path:
    override = (os.environ if env is None else env).get(CONFIG_ENV)
    if override:
        return pathlib.Path(override)
    return app_folder(system, env) / "config.json"


def data_dir(system: str | None = None, env=None) -> pathlib.Path:
    """Where everything writable lives. Never inside the Aseprite extension:
    Aseprite replaces that folder on every update, venv and models included."""
    override = (os.environ if env is None else env).get(DATA_ENV)
    if override:
        return pathlib.Path(override)
    return app_folder(system, env)


def load_port(path: pathlib.Path | None = None) -> int:
    target = path or config_path()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))["port"]
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_PORT
    if isinstance(value, int) and not isinstance(value, bool) \
            and 1024 <= value <= 65535:
        return value
    return DEFAULT_PORT


def load_vram_mode(path: pathlib.Path | None = None) -> str:
    target = path or config_path()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))["vram_mode"]
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_VRAM_MODE
    if value in VRAM_MODES:
        return value
    return DEFAULT_VRAM_MODE


def _read(target: pathlib.Path) -> dict:
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_settings(path: pathlib.Path | None = None) -> dict:
    target = path or config_path()
    data = _read(target)
    out = {key: data.get(key) for key in SETTING_KEYS}
    out["port"] = load_port(target)
    out["vram_mode"] = load_vram_mode(target)
    return out


def save_settings(values: dict, path: pathlib.Path | None = None) -> None:
    # merge, never replace: the file is shared with settings we do not own
    target = path or config_path()
    data = _read(target)
    data.update(values)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if data.get(API_KEY_SETTING) and sys.platform != "win32":
        # an API key is in there: owner-only, like ~/.ssh files
        try:
            target.chmod(0o600)
        except OSError:
            pass


# ---- which model makes the pictures

PROVIDERS = ("local", "openrouter")
DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_MODEL = "black-forest-labs/FLUX.2-klein-4B"
# image in + image out on OpenRouter; any model with image output works
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash-image"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
API_KEY_SETTING = "openrouter_api_key"
API_KEY_ENV = "OPENROUTER_API_KEY"  # wins over the saved key when set
MODEL_KEYS = ("provider", "local_model", "openrouter_model",
              API_KEY_SETTING, "openrouter_base_url")


def _text(value, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() \
        else default


def load_model_settings(path: pathlib.Path | None = None,
                        env=None) -> dict:
    """The provider and model choice, every value filled in. 'api_key_source'
    says where the key came from: 'env', 'saved' or None."""
    data = _read(path or config_path())
    env = os.environ if env is None else env
    provider = data.get("provider")
    saved_key = _text(data.get(API_KEY_SETTING), "")
    env_key = _text(env.get(API_KEY_ENV), "")
    return {
        "provider": provider if provider in PROVIDERS else DEFAULT_PROVIDER,
        "local_model": _text(data.get("local_model"), DEFAULT_LOCAL_MODEL),
        "openrouter_model": _text(data.get("openrouter_model"),
                                  DEFAULT_OPENROUTER_MODEL),
        "openrouter_base_url": _text(data.get("openrouter_base_url"),
                                     DEFAULT_OPENROUTER_URL).rstrip("/"),
        "api_key": env_key or saved_key,
        "api_key_source": "env" if env_key else ("saved" if saved_key
                                                  else None),
    }


def save_model_settings(values: dict, path: pathlib.Path | None = None) -> None:
    """Saves only known keys; an empty string resets a model to its default
    (or forgets the key), an invalid provider is ignored."""
    clean = {}
    for key in MODEL_KEYS:
        if key not in values:
            continue
        value = values[key]
        if key == "provider":
            if value in PROVIDERS:
                clean[key] = value
            continue
        clean[key] = value.strip() if isinstance(value, str) and \
            value.strip() else None
    if clean:
        save_settings(clean, path)


def mask_key(key: str | None) -> str:
    """Enough of a key to recognise it, never enough to use it."""
    if not key:
        return ""
    return "…" + key[-4:] if len(key) > 8 else "set"


def load_models_dir(path: pathlib.Path | None = None) -> str:
    return load_settings(path).get("models_dir") or DEFAULT_MODELS_DIR


def save_port(port: int, path: pathlib.Path | None = None) -> None:
    save_settings({"port": port}, path)


def save_vram_mode(mode: str, path: pathlib.Path | None = None) -> None:
    if mode not in VRAM_MODES:
        return
    save_settings({"vram_mode": mode}, path)
