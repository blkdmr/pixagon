# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

"""Installing what is missing, one step at a time, in a fixed order.

uv does the heavy lifting: it creates the environment (downloading a Python
when the machine has none), and installs PyTorch and the server packages.
"""
import os
import pathlib
import subprocess
import threading

from launcher import platforms, setup_checks
from launcher.server_proc import TreeGuard, clean_line
from server.config import DEFAULT_LOCAL_MODEL, load_model_settings

# uv first, everything else runs through it. torch before deps:
# bitsandbytes pulls torch>=2.3,<3 on its own, and if nothing satisfies that
# yet, a plain CPU build lands here that the torch step would then replace
ORDER = ("uv", "venv", "torch", "deps", "model")
LABELS = {"uv": "Installing uv",
          "venv": "Creating virtual environment",
          "deps": "Installing server dependencies",
          "torch": "Installing " + platforms.torch_label(),
          "model": "Downloading the model"}
TORCH_INDEX = platforms.CUDA_TORCH_INDEX
MODEL_ID = DEFAULT_LOCAL_MODEL  # Setup > Model can pick another
# asked of uv when no usable Python was found; uv downloads it if needed
DEFAULT_PYTHON = "3.12"


def plan(selected) -> list[str]:
    chosen = set(selected)
    return [step for step in ORDER if step in chosen]


def venv_interpreter(paths) -> pathlib.Path:
    return paths.data / ".venv" / platforms.venv_python_rel()


def uv_executable(paths) -> str:
    # looked up at run time: the uv step may have just put it in .tools/
    found = platforms.find_uv(paths.data)
    return str(found) if found else "uv"


def command_env(step_id: str, paths) -> dict | None:
    """Extra environment variables a step needs, or None."""
    if step_id == "uv":
        return platforms.uv_install_command(paths.data)[1]
    return None


def command(step_id: str, paths) -> list[str] | None:
    if step_id == "uv":
        return platforms.uv_install_command(paths.data)[0]
    uv = uv_executable(paths)
    if step_id == "venv":
        # --seed keeps pip inside, for anyone following the by-hand steps
        python = str(paths.python) if paths.python else DEFAULT_PYTHON
        return [uv, "venv", "--seed", "--allow-existing", "--python", python,
                str(paths.data / ".venv")]
    interpreter = str(venv_interpreter(paths))
    settings = load_model_settings()
    if step_id == "deps":
        # a cloud provider skips the multi-GB local-model stack
        name = ("requirements.txt" if setup_checks.is_local(settings)
                else "requirements-base.txt")
        return [uv, "pip", "install", "--python", interpreter, "-r",
                str(paths.root / "server" / name)]
    if step_id == "torch":
        return [uv, "pip", "install", "--python", interpreter,
                *platforms.torch_install_args()]
    if step_id == "model":
        # the local model chosen in Setup (FLUX.2 Klein 4B unless changed)
        model_id = settings["local_model"]
        folder = setup_checks.model_folder(model_id)
        # byte-total bar (not file count); Xet's progress lags real transfer,
        # so it's disabled; the default repo's standalone ~8 GB checkpoint is
        # unused (the pattern matches nothing in other repos)
        code = (
            "import os\n"
            "os.environ['HF_HUB_DISABLE_XET'] = '1'\n"
            "from huggingface_hub import snapshot_download\n"
            "from tqdm.auto import tqdm as _base_tqdm\n"
            "class _P(_base_tqdm):\n"
            "    _last = None\n"
            "    def display(self, msg=None, pos=None):\n"
            "        if self.total < 1_000_000 or 'incomplete total' not"
            " in (self.desc or ''):\n"
            "            return\n"
            "        pct = min(int(self.n * 100 / self.total), 100)\n"
            "        line = (f'Downloading model files... {pct}% '\n"
            "                f'({self.n/1e6:.0f}/{self.total/1e6:.0f} MB)')\n"
            "        if line != _P._last:\n"
            "            print(line, flush=True)\n"
            "            _P._last = line\n"
            f"snapshot_download({model_id!r}, cache_dir=r'{paths.models_dir}', "
            "ignore_patterns=['flux-2-klein-4b.safetensors'], tqdm_class=_P)\n"
            "import pathlib\n"
            # marker written only once the download actually finishes
            f"(pathlib.Path(r'{paths.models_dir}') / {folder!r} / "
            f"{setup_checks.MODEL_MARKER!r}).write_text('ok')\n"
        )
        return [interpreter, "-u", "-c", code]
    return None


class Runner:
    def __init__(self, paths, on_event, on_log):
        self.paths = paths
        self.on_event = on_event
        self.on_log = on_log
        self.commands = {}
        self.proc = None
        self.guard = None
        self.cancelled = False
        self.thread = None

    def is_running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def start(self, ids) -> None:
        if self.is_running():
            return
        self.cancelled = False
        self.thread = threading.Thread(target=self._run, args=(plan(ids),),
                                       daemon=True)
        self.thread.start()

    def wait(self, timeout=None) -> None:
        if self.thread:
            self.thread.join(timeout)

    def cancel(self) -> None:
        self.cancelled = True
        guard, self.guard = self.guard, None
        if guard:
            guard.kill()

    def _run(self, steps) -> None:
        halted = False
        for step in steps:
            if self.cancelled:
                self.on_event(step, "skipped", "cancelled")
                continue
            if halted:
                self.on_event(step, "skipped", "a previous step failed")
                continue
            self.on_log(f"===== {LABELS[step]} =====")
            self.on_event(step, "running", "")
            ok = self._one(step)
            if self.cancelled:
                self.on_event(step, "cancelled", "stopped")
                halted = True
                continue
            self.on_event(step, "done" if ok else "failed",
                          "" if ok else "see the log")
            if not ok:
                halted = True

    def _one(self, step) -> bool:
        cmd = self.commands.get(step) or command(step, self.paths)
        return self._spawn(cmd, command_env(step, self.paths))

    def _spawn(self, cmd, extra_env=None) -> bool:
        env = dict(os.environ)
        env.update(extra_env or {})
        # uv only draws bars on a terminal; keep its log plain and colourless
        env.setdefault("NO_COLOR", "1")
        try:
            # same guard as the server: cancelling must take uv's children too
            self.proc, self.guard = TreeGuard.popen(
                cmd, cwd=str(self.paths.data), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", env=env)
        except OSError as e:
            self.on_log(f"could not start: {e}")
            return False
        # cancel may have fired before the guard existed; catch that here
        if self.cancelled:
            guard, self.guard = self.guard, None
            if guard:
                guard.kill()
            return False
        noted = False
        for raw in self.proc.stdout:
            line = clean_line(raw)
            if not line:
                continue
            if line.startswith("Downloading model files..."):
                self.on_log(line, replace=True)
                continue
            self.on_log(line)
            # uv names each big wheel once, then is silent while it downloads
            if not noted and line.startswith("Downloading ") and "(" in line:
                self.on_log("(large downloads can take a few minutes with "
                            "no progress shown)")
                noted = True
        code = self.proc.wait()
        guard, self.guard = self.guard, None
        if guard:
            guard.kill()
        return code == 0
