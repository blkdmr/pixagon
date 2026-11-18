# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

"""What is already in place. Detection only, never a side effect."""
import shutil

from launcher import platforms
from launcher.paths import run_command
from launcher.server_proc import venv_python
from server.config import (DEFAULT_LOCAL_MODEL, load_model_settings,
                           mask_key)

OK = "ok"
MISSING = "missing"
BLOCKED = "blocked"

MODEL_FOLDER = "models--black-forest-labs--FLUX.2-klein-4B"  # the default
# a snapshot dir appears from the first file downloaded, long before it's done
MODEL_MARKER = ".complete"
MODEL_GB = 15


def model_folder(model_id: str) -> str:
    """The Hugging Face cache folder of a repo id (as server.instruct names it)."""
    return "models--" + model_id.replace("/", "--")


def is_local(settings) -> bool:
    return settings["provider"] == "local"


SERVER_NEEDS = ("venv", "deps", "torch", "model")  # the local model
CLOUD_NEEDS = ("venv", "deps", "apikey")           # OpenRouter: no GPU stack


def server_needs(settings=None) -> tuple:
    """The rows that must be OK before Start, for the chosen provider."""
    settings = settings or load_model_settings()
    return SERVER_NEEDS if is_local(settings) else CLOUD_NEEDS


def backend_label(settings=None) -> str:
    """What makes the pictures, for the launcher's main screen."""
    settings = settings or load_model_settings()
    if settings["provider"] == "openrouter":
        return "OpenRouter · " + settings["openrouter_model"]
    return settings["local_model"].rsplit("/", 1)[-1] + " · local"


# metadata only, not a real import -- importing torch/diffusers costs seconds
def deps_probe(system=None, local=True) -> str:
    packages = (platforms.server_packages(system) if local
                else platforms.cloud_packages())
    names = ",".join(repr(n) for n in packages)
    return ("import importlib.metadata as m; "
            f"[m.version(d) for d in ({names},)]; print('ok')")


DEPS_PROBE = deps_probe()
TORCH_PROBE = "import importlib.metadata as m; print(m.version('torch'))"


def _cuda_tag(version):
    # torch's local version tag carries the CUDA build, e.g. 2.4.1+cu124 -> 12.4
    if not version or "+cu" not in version:
        return None
    digits = "".join(c for c in version.split("+cu", 1)[1] if c.isdigit())
    return f"{digits[:-1]}.{digits[-1]}" if len(digits) >= 2 else None


def torch_detail(version, system=None):
    """'CUDA 12.8' / 'Metal' for a usable GPU build of torch, else None."""
    if not version:
        return None
    if platforms.gpu_backend(system) == "mps":
        # macOS wheels carry no local tag; every arm64 build has Metal
        return "Metal"
    cuda = _cuda_tag(version)
    return f"CUDA {cuda}" if cuda else None


def _item(item_id, label, state, detail, required=True, needs=()):
    return {"id": item_id, "label": label, "state": state, "detail": detail,
            "required": required, "needs": list(needs)}


def _free_gb(folder) -> int:
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free // 2**30
    except OSError:
        return 0


def model_label(model_id: str) -> str:
    if model_id == DEFAULT_LOCAL_MODEL:
        return f"Model, {MODEL_GB} GB"
    return "Model, " + model_id.rsplit("/", 1)[-1]


def _model_item(paths, settings=None) -> dict:
    settings = settings or load_model_settings()
    model_id = settings["local_model"]
    local = is_local(settings)
    marker = paths.models_dir / model_folder(model_id) / MODEL_MARKER
    if marker.is_file():
        return _item("model", model_label(model_id), OK, "downloaded",
                     required=local, needs=["deps"])
    detail = (f"{_free_gb(paths.models_dir)} GB free" if local
              else "only for the local model")
    return _item("model", model_label(model_id), MISSING, detail,
                 required=local, needs=["deps"])


def _apikey_item(settings) -> dict:
    """Not installable: the key is pasted in the launcher's Model step."""
    if settings["api_key"]:
        detail = ("from OPENROUTER_API_KEY" if settings["api_key_source"]
                  == "env" else mask_key(settings["api_key"]))
        return _item("apikey", "OpenRouter API key", OK, detail)
    return _item("apikey", "OpenRouter API key", BLOCKED, "paste it below")


# Setup walks Backend, Install, Model, Run. Backend and Run own no check
# rows; the other two each own these
STEP_ITEMS = {"env": ("uv", "venv", "torch", "deps"),
              "model": ("model", "apikey")}


def step_status(rows: list[dict]) -> dict:
    """Each step's badge in the stepper: running/failed while an install
    touches it, done once its required rows are OK, todo otherwise. Rows may
    carry the runner's step_state on top of the check state."""
    out = {"backend": "done"}  # a provider is always chosen (local by default)
    for step, ids in STEP_ITEMS.items():
        mine = [r for r in rows if r["id"] in ids]
        runs = {r.get("step_state") for r in mine}
        if "running" in runs:
            out[step] = "running"
        elif "failed" in runs:
            out[step] = "failed"
        elif all(r["state"] == OK for r in mine if r["required"]):
            out[step] = "done"
        else:
            out[step] = "todo"
    return out


def first_step(steps: dict) -> str:
    """Where Setup opens: the provider choice on a fresh machine, the model
    step while it is unfinished, Run once everything is in place."""
    if steps["env"] != "done":
        return "backend"
    if steps["model"] != "done":
        return "model"
    return "run"


def refresh_live(items: list[dict], paths) -> list[dict]:
    # filesystem-only, no subprocess -- cheap enough to run on every poll
    fresh = {"model": _model_item}
    return [fresh[it["id"]](paths) if it["id"] in fresh else it
            for it in items]


def uv_version(output: str | None) -> str | None:
    """'0.9.5' from `uv --version` ('uv 0.9.5 (abc 2025-10-01)')."""
    parts = (output or "").split()
    if len(parts) >= 2 and parts[0] == "uv":
        return parts[1]
    return None


def _uv_item(paths, runner) -> dict:
    uv = getattr(paths, "uv", None)
    version = uv_version(runner([str(uv), "--version"])) if uv else None
    if version:
        return _item("uv", "uv package manager", OK, version)
    return _item("uv", "uv package manager", MISSING,
                 "not found" if not uv else "does not run")


def check_all(paths, run=None, settings=None) -> list[dict]:
    runner = run or run_command
    settings = settings or load_model_settings()
    local = is_local(settings)
    items = []

    items.append(_uv_item(paths, runner))

    # uv brings its own Python when none is installed, and needs no ensurepip,
    # so the environment is never blocked on the system's Python
    interpreter = venv_python(paths.data)
    if interpreter:
        items.append(_item("venv", "Virtual environment", OK, ".venv",
                           needs=["uv"]))
    else:
        items.append(_item("venv", "Virtual environment", MISSING, "missing",
                           needs=["uv"]))

    # deps alone would let bitsandbytes pull a plain CPU torch in; needing
    # torch here (not just venv) keeps the CUDA build ahead of that pull.
    # A cloud provider needs neither torch nor the local-model packages.
    deps_needs = ["venv", "torch"] if local else ["venv"]
    torch_label = platforms.torch_label()
    if not interpreter:
        items.append(_item("torch", torch_label, MISSING,
                           "needs the environment" if local
                           else "only for the local model",
                           required=local, needs=["venv"]))
        items.append(_item("deps", "Server dependencies", MISSING,
                           "needs the environment", needs=deps_needs))
    else:
        got = runner([str(interpreter), "-c", TORCH_PROBE])
        detail = torch_detail(got)
        if detail:
            items.append(_item("torch", torch_label, OK, detail,
                               required=local, needs=["venv"]))
        else:
            missing = ("no CUDA" if got else "missing") if local \
                else "only for the local model"
            items.append(_item("torch", torch_label, MISSING, missing,
                               required=local, needs=["venv"]))
        got = runner([str(interpreter), "-c", deps_probe(local=local)])
        items.append(_item("deps", "Server dependencies",
                           OK if got else MISSING,
                           "installed" if got else "missing",
                           needs=deps_needs))

    if not local:
        items.append(_apikey_item(settings))
    items.append(_model_item(paths, settings))
    # the page puts everything after the first optional row under "Optional"
    return sorted(items, key=lambda it: not it["required"])
