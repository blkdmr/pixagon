# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

"""Where things live: detected first, overridden by the settings file."""
import dataclasses
import pathlib
import subprocess

from launcher import platforms
from launcher.platforms import NO_WINDOW
from server.config import data_dir, load_settings

MIN_PYTHON = (3, 11)
PROBE = "import sys; print(sys.executable + '|%d.%d.%d' % sys.version_info[:3])"


def app_root() -> pathlib.Path:
    """Where the code lives: server/, launcher/ and the plugin's Lua. In a
    release that is the Aseprite extension folder; in a checkout, the repo."""
    return pathlib.Path(__file__).resolve().parent.parent


def run_command(cmd: list[str]) -> str | None:
    """Stdout of a short probe, or None on failure; setup_checks shares it."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                              creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def probe_python(command, run=None):
    """(path, version) for a working 3.11+ interpreter, else None."""
    out = (run or run_command)(list(command) + ["-c", PROBE])
    if not out or "|" not in out:
        return None
    exe, _, raw = out.rpartition("|")
    try:
        version = tuple(int(part) for part in raw.split("."))
    except ValueError:
        return None
    if len(version) != 3 or version < MIN_PYTHON:
        return None
    return pathlib.Path(exe), version


@dataclasses.dataclass
class Paths:
    root: pathlib.Path
    python: pathlib.Path | None
    python_version: tuple | None
    models_dir: pathlib.Path
    # uv installs everything; a Python set in the settings is only a hint
    # for `uv venv`
    uv: pathlib.Path | None = None
    # where .venv, .tools and models go (server.config.data_dir)
    data: pathlib.Path | None = None

    def __post_init__(self):
        if self.data is None:
            self.data = self.root


def resolve(settings: dict | None = None, run=None) -> Paths:
    values = settings if settings is not None else load_settings()
    root = pathlib.Path(values["root"]) if values.get("root") else app_root()
    # no auto-detection: run under the supervisor, `python3` would be uv's
    # throwaway environment, and a .venv built on it breaks once uv prunes it
    python, version = None, None
    if values.get("python"):
        python = pathlib.Path(values["python"])
        probed = probe_python([str(python)], run=run)
        version = probed[1] if probed else None
    data = data_dir()
    models = (pathlib.Path(values["models_dir"]) if values.get("models_dir")
              else data / "models")
    return Paths(root=root, python=python, python_version=version,
                 models_dir=models,
                 uv=platforms.find_uv(data), data=data)
