# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

import pathlib
import shutil
import subprocess
import sys
import threading

import pytest

from launcher import paths, platforms, setup_checks, setup_steps


def make_paths(tmp_path):
    return paths.Paths(root=tmp_path, python=pathlib.Path(sys.executable),
                       python_version=(3, 14, 0),
                       models_dir=tmp_path / "models")


def test_plan_keeps_the_fixed_order():
    assert setup_steps.plan(["model", "venv", "torch", "uv"]) == \
        ["uv", "venv", "torch", "model"]


def test_uv_is_installed_before_anything_uses_it():
    assert setup_steps.ORDER[0] == "uv"


def test_plan_drops_unknown_ids():
    assert setup_steps.plan(["venv", "nonsense"]) == ["venv"]


def fake_uv(monkeypatch, path="/tools/uv"):
    monkeypatch.setattr(platforms, "find_uv",
                        lambda root, *a, **k: pathlib.Path(path))
    return str(pathlib.Path(path))


def test_command_for_venv_runs_uv(tmp_path, monkeypatch):
    uv = fake_uv(monkeypatch)
    cmd = setup_steps.command("venv", make_paths(tmp_path))
    assert cmd[:2] == [uv, "venv"]
    assert "--seed" in cmd  # pip stays available for the by-hand steps
    assert cmd[cmd.index("--python") + 1] == sys.executable
    assert cmd[-1] == str(tmp_path / ".venv")


def test_venv_without_any_python_asks_uv_for_one(tmp_path, monkeypatch):
    fake_uv(monkeypatch)
    p = make_paths(tmp_path)
    p = paths.Paths(root=p.root, python=None, python_version=None,
                    models_dir=p.models_dir)
    cmd = setup_steps.command("venv", p)
    assert cmd[cmd.index("--python") + 1] == setup_steps.DEFAULT_PYTHON


def test_commands_fall_back_to_uv_on_path(tmp_path, monkeypatch):
    monkeypatch.setattr(platforms, "find_uv", lambda root, *a, **k: None)
    assert setup_steps.command("deps", make_paths(tmp_path))[0] == "uv"


def test_uv_step_runs_the_official_installer_into_tools(tmp_path):
    p = make_paths(tmp_path)
    cmd = setup_steps.command("uv", p)
    env = setup_steps.command_env("uv", p)
    assert env["UV_INSTALL_DIR"] == str(tmp_path / platforms.TOOLS_DIR)
    assert env["UV_NO_MODIFY_PATH"] == "1"
    assert "astral.sh/uv/install" in " ".join(cmd)
    assert setup_steps.command_env("deps", p) is None


def real_uv():
    found = shutil.which("uv")
    if not found:
        home = pathlib.Path.home() / ".local" / "bin" / platforms.uv_name()
        found = str(home) if home.is_file() else None
    return found


@pytest.mark.skipif(real_uv() is None, reason="uv is not installed here")
def test_venv_step_actually_creates_a_working_venv_with_uv(tmp_path,
                                                          monkeypatch):
    uv = real_uv()
    monkeypatch.setattr(platforms, "find_uv",
                        lambda root, *a, **k: pathlib.Path(uv))
    logs = []
    runner = setup_steps.Runner(make_paths(tmp_path),
                                on_event=lambda *a: None,
                                on_log=lambda line, replace=False: logs.append(line))
    runner.start(["venv"])
    runner.wait(timeout=120)
    exe = tmp_path / ".venv" / platforms.venv_python_rel()
    assert exe.exists(), logs
    # --seed: pip is in there too
    done = subprocess.run([str(exe), "-m", "pip", "--version"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_command_for_torch_uses_the_cuda_index(tmp_path, monkeypatch):
    monkeypatch.setattr(platforms, "gpu_backend", lambda system=None: "cuda")
    uv = fake_uv(monkeypatch)
    p = make_paths(tmp_path)
    cmd = setup_steps.command("torch", p)
    assert cmd[:3] == [uv, "pip", "install"]
    assert cmd[cmd.index("--python") + 1] == \
        str(setup_steps.venv_interpreter(p))
    assert "--index-url" in cmd
    assert cmd[-1].endswith("cu128")
    # a bare "torch" is satisfied by any build, a CPU one included
    assert cmd[cmd.index("--reinstall-package") + 1] == "torch"


def test_torch_args_per_platform():
    for system in ("win32", "linux"):
        args = platforms.torch_install_args(system)
        assert args[-1].endswith("cu128") and "--reinstall-package" in args
    # the default PyPI wheel already has Metal on macOS
    assert platforms.torch_install_args("darwin") == \
        ["torch", "--upgrade-package", "torch"]


def test_venv_interpreter_follows_the_os_layout(tmp_path):
    expected = tmp_path / ".venv" / platforms.venv_python_rel()
    assert setup_steps.venv_interpreter(make_paths(tmp_path)) == expected


def test_torch_step_runs_before_deps():
    assert setup_steps.ORDER.index("torch") < setup_steps.ORDER.index("deps")


def test_deps_install_into_the_venv_through_uv(tmp_path, monkeypatch):
    uv = fake_uv(monkeypatch)
    p = make_paths(tmp_path)
    cmd = setup_steps.command("deps", p)
    assert cmd[:3] == [uv, "pip", "install"]
    assert cmd[cmd.index("--python") + 1] == \
        str(setup_steps.venv_interpreter(p))
    assert cmd[-1] == str(tmp_path / "server" / "requirements.txt")
    # no raw pip anywhere in the install steps
    for step in ("venv", "torch", "deps"):
        assert "pip" not in pathlib.Path(setup_steps.command(step, p)[0]).name


def test_model_command_reports_the_shared_byte_total(tmp_path):
    # file-count progress hides multi-minute stalls on a single 7+ GB file;
    # the aggregate byte bar moves continuously regardless of file size
    code = setup_steps.command("model", make_paths(tmp_path))[-1]
    assert "incomplete total" in code
    assert "tqdm_class=_P" in code


def test_model_command_skips_xet_and_the_redundant_checkpoint(tmp_path):
    code = setup_steps.command("model", make_paths(tmp_path))[-1]
    assert "HF_HUB_DISABLE_XET" in code
    assert "flux-2-klein-4b.safetensors" in code


def test_model_command_writes_the_completion_marker(tmp_path):
    # setup_checks treats the snapshot folder's mere existence as insufficient
    # (it appears from the first file downloaded); only this marker means done
    code = setup_steps.command("model", make_paths(tmp_path))[-1]
    assert setup_checks.MODEL_FOLDER in code
    assert setup_checks.MODEL_MARKER in code






def test_failure_skips_dependents(tmp_path):
    events = []
    runner = setup_steps.Runner(make_paths(tmp_path),
                                on_event=lambda *a: events.append(a),
                                on_log=lambda line: None)
    runner.commands = {"venv": [sys.executable, "-c", "raise SystemExit(1)"],
                       "deps": [sys.executable, "-c", "print('never')"]}
    runner.start(["venv", "deps", "model"])
    runner.wait(timeout=30)
    states = {step: state for step, state, _ in events}
    assert states["venv"] == "failed"
    assert states["deps"] == "skipped"
    assert states["model"] == "skipped"  # later step, halted all the same


def test_missing_interpreter_fails_cleanly_instead_of_hanging(tmp_path):
    events = []
    runner = setup_steps.Runner(make_paths(tmp_path),
                                on_event=lambda *a: events.append(a),
                                on_log=lambda line: None)
    runner.commands = {"torch": [str(tmp_path / "no-such-python.exe"), "-V"]}
    runner.start(["torch"])
    runner.wait(timeout=30)
    states = {step: state for step, state, _ in events}
    assert states["torch"] == "failed"


def test_success_runs_every_step(tmp_path):
    events = []
    runner = setup_steps.Runner(make_paths(tmp_path),
                                on_event=lambda *a: events.append(a),
                                on_log=lambda line: None)
    runner.commands = {"venv": [sys.executable, "-c", "print('made')"],
                       "deps": [sys.executable, "-c", "print('installed')"]}
    runner.start(["venv", "deps"])
    runner.wait(timeout=30)
    states = {step: state for step, state, _ in events}
    assert states["venv"] == "done"
    assert states["deps"] == "done"




def test_big_downloads_get_one_note_about_the_silence(tmp_path):
    logs = []
    runner = setup_steps.Runner(make_paths(tmp_path),
                                on_event=lambda *a: None,
                                on_log=lambda line: logs.append(line))
    script = ("print('Resolved 30 packages in 1.2s')\n"
              "print('Downloading torch (846.0MiB)')\n"
              "print('Downloading nvidia-cudnn-cu12 (664.8MiB)')\n"
              "print('Installed 30 packages in 3.1s')\n")
    runner.commands = {"deps": [sys.executable, "-c", script]}
    runner.start(["deps"])
    runner.wait(timeout=30)
    notes = [l for l in logs if "no progress shown" in l]
    assert len(notes) == 1
    assert "Installed 30 packages in 3.1s" in logs


def test_the_uv_step_gets_its_environment(tmp_path):
    logs = []
    runner = setup_steps.Runner(make_paths(tmp_path),
                                on_event=lambda *a: None,
                                on_log=lambda line: logs.append(line))
    script = "import os; print('dir=' + os.environ['UV_INSTALL_DIR'])"
    runner.commands = {"uv": [sys.executable, "-c", script]}
    runner.start(["uv"])
    runner.wait(timeout=30)
    assert f"dir={tmp_path / platforms.TOOLS_DIR}" in logs


def test_cancel_stops_the_run(tmp_path):
    events = []
    runner = setup_steps.Runner(make_paths(tmp_path),
                                on_event=lambda *a: events.append(a),
                                on_log=lambda line: None)
    runner.commands = {"venv": [sys.executable, "-c",
                                "import time; time.sleep(60)"],
                       "deps": [sys.executable, "-c", "print('never')"]}
    runner.start(["venv", "deps"])
    for _ in range(100):
        if runner.is_running():
            break
        threading.Event().wait(0.05)
    runner.cancel()
    runner.wait(timeout=30)
    states = {step: state for step, state, _ in events}
    assert states["venv"] == "cancelled"
    assert states.get("deps") in ("skipped", "cancelled")
    assert not runner.is_running()
