"""Everything that differs between Windows, macOS and Linux, in one place.

Each helper takes the platform (sys.platform style) and the environment as
optional arguments, so the tests can ask about any OS from any OS.
"""
import os
import pathlib
import shutil
import subprocess
import sys


WINDOWS = "win32"
MAC = "darwin"

# no console window when a windowed exe spawns a child; 0 means nothing off Windows
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

APP_NAME = "Pixagon"
CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def current(system: str | None = None) -> str:
    return system or sys.platform


def is_windows(system: str | None = None) -> bool:
    return current(system) == WINDOWS


def is_mac(system: str | None = None) -> bool:
    return current(system) == MAC


def is_linux(system: str | None = None) -> bool:
    return current(system).startswith("linux")


def _env(env):
    return os.environ if env is None else env


def _home(env) -> pathlib.Path:
    return pathlib.Path(_env(env).get("HOME") or pathlib.Path.home())


# ---- Python and the virtual environment

def venv_python_rel(system: str | None = None) -> pathlib.PurePath:
    if is_windows(system):
        return pathlib.PurePath("Scripts", "python.exe")
    return pathlib.PurePath("bin", "python")


VENV_LAYOUTS = (pathlib.PurePath("Scripts", "python.exe"),
                pathlib.PurePath("bin", "python"),
                pathlib.PurePath("bin", "python3"))


def find_venv_python(venv: pathlib.Path,
                     system: str | None = None) -> pathlib.Path | None:
    """The interpreter inside a venv; this OS's layout first, then any other."""
    preferred = venv_python_rel(system)
    for rel in (preferred, *[r for r in VENV_LAYOUTS if r != preferred]):
        candidate = pathlib.Path(venv) / rel
        if candidate.exists():
            return candidate
    return None


# ---- uv, which builds the environment and installs every package

UV_INSTALLER_SH = "https://astral.sh/uv/install.sh"
UV_INSTALLER_PS = "https://astral.sh/uv/install.ps1"
TOOLS_DIR = ".tools"  # Setup's own uv lives here, inside the install folder


def uv_name(system: str | None = None) -> str:
    return "uv.exe" if is_windows(system) else "uv"


def uv_candidates(root, system: str | None = None, env=None,
                  which=shutil.which) -> list[pathlib.Path]:
    """Where uv may be: Setup's copy, PATH, then the installer's defaults.
    A GUI app often has a shorter PATH than a terminal, hence the defaults."""
    name = uv_name(system)
    tools = pathlib.Path(root) / TOOLS_DIR
    home = _home(env)
    out = [tools / name, tools / "bin" / name]
    on_path = which("uv")
    if on_path:
        out.append(pathlib.Path(on_path))
    out += [home / ".local" / "bin" / name, home / ".cargo" / "bin" / name]
    if is_mac(system):
        out += [pathlib.Path("/opt/homebrew/bin/uv"),
                pathlib.Path("/usr/local/bin/uv")]
    return out


def find_uv(root, system: str | None = None, env=None,
            which=shutil.which) -> pathlib.Path | None:
    for candidate in uv_candidates(root, system, env, which):
        if candidate.is_file():
            return candidate
    return None


def uv_install_command(root, system: str | None = None) -> tuple[list, dict]:
    """(command, extra env) running Astral's official uv installer, aimed at
    the install folder's .tools/ and leaving the user's shell profile alone."""
    extra = {"UV_INSTALL_DIR": str(pathlib.Path(root) / TOOLS_DIR),
             "UV_NO_MODIFY_PATH": "1"}
    if is_windows(system):
        return (["powershell", "-NoProfile", "-NonInteractive",
                 "-ExecutionPolicy", "ByPass", "-Command",
                 f"irm {UV_INSTALLER_PS} | iex"], extra)
    script = (f"if command -v curl >/dev/null 2>&1; then "
              f"curl -LsSf {UV_INSTALLER_SH} | sh; "
              f"elif command -v wget >/dev/null 2>&1; then "
              f"wget -qO- {UV_INSTALLER_SH} | sh; "
              f"else echo 'needs curl or wget to download uv'; exit 1; fi")
    return (["/bin/sh", "-c", script], extra)


# ---- the GPU and PyTorch

def gpu_backend(system: str | None = None) -> str:
    """Which accelerator this OS can offer: 'cuda' (NVIDIA) or 'mps' (Apple)."""
    if is_mac(system):
        return "mps"
    return "cuda"


def gpu_memory_gb(run, system: str | None = None) -> float | None:
    """Total memory of the first NVIDIA card in GB, from nvidia-smi, so the
    model menu can tell which load mode fits. None on a Mac (one shared pool)
    or when there is no card or driver. `run` returns a command's stdout."""
    if gpu_backend(system) == "mps":
        return None
    out = run(["nvidia-smi", "--query-gpu=memory.total",
               "--format=csv,noheader,nounits"])
    try:
        return round(int(out.splitlines()[0].strip()) / 1024, 1)
    except (AttributeError, IndexError, ValueError):
        return None


def torch_label(system: str | None = None) -> str:
    if gpu_backend(system) == "mps":
        return "PyTorch with Metal"
    return "PyTorch with CUDA"


def torch_install_args(system: str | None = None) -> list[str]:
    """`uv pip install` arguments that fetch a GPU build of torch for this OS."""
    if gpu_backend(system) == "cuda":
        # a bare "torch" reads as satisfied by any installed build, a CPU one
        # included; reinstalling from the CUDA-only index replaces it
        return ["torch", "--reinstall-package", "torch",
                "--index-url", CUDA_TORCH_INDEX]
    # macOS: the default PyPI wheel already carries Metal (MPS) support
    return ["torch", "--upgrade-package", "torch"]


def server_packages(system: str | None = None) -> tuple:
    """Distributions the server needs, as named on PyPI."""
    base = ("websockets", "diffusers", "transformers", "accelerate", "peft",
            "Pillow", "numpy", "scipy")
    # bitsandbytes is CUDA-only; requirements.txt skips it on macOS too
    return base if is_mac(system) else base + ("bitsandbytes",)


def cloud_packages() -> tuple:
    """What the server needs with a cloud provider: requirements-base.txt."""
    return ("websockets", "Pillow", "numpy", "scipy")


