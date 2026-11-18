# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

import pathlib

import pytest

from launcher import paths, platforms, setup_checks


@pytest.fixture
def nvidia(monkeypatch):
    """Ask the CUDA questions even on a Mac, whose torch check is Metal."""
    monkeypatch.setattr(platforms, "gpu_backend", lambda system=None: "cuda")


def make_paths(tmp_path, python=True, venv=False, uv=None):
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    if venv:
        exe = root / ".venv" / platforms.venv_python_rel()
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("", encoding="utf-8")
    return paths.Paths(
        root=root,
        python=pathlib.Path(r"C:\P\python.exe") if python else None,
        python_version=(3, 14, 0) if python else None,
        models_dir=root / "models",
        uv=uv)


def by_id(items):
    return {item["id"]: item for item in items}


def test_uv_missing(tmp_path):
    items = by_id(setup_checks.check_all(make_paths(tmp_path),
                                         run=lambda cmd: None))
    assert items["uv"]["state"] == setup_checks.MISSING
    assert items["uv"]["required"] is True
    assert "python" not in items  # uv brings its own Python


def test_uv_ok_shows_version(tmp_path):
    def run(cmd):
        return "uv 0.9.5 (d5f39331a 2025-10-21)" if cmd[-1] == "--version" \
            else None
    items = by_id(setup_checks.check_all(
        make_paths(tmp_path, uv=pathlib.Path("/x/uv")), run=run))
    assert items["uv"]["state"] == setup_checks.OK
    assert items["uv"]["detail"] == "0.9.5"


def test_uv_that_does_not_run_counts_as_missing(tmp_path):
    items = by_id(setup_checks.check_all(
        make_paths(tmp_path, uv=pathlib.Path("/x/uv")), run=lambda cmd: None))
    assert items["uv"]["state"] == setup_checks.MISSING


def test_uv_version_parsing():
    assert setup_checks.uv_version("uv 0.9.5 (abc 2025-10-21)") == "0.9.5"
    assert setup_checks.uv_version("uv 0.4.0") == "0.4.0"
    assert setup_checks.uv_version("garbage") is None
    assert setup_checks.uv_version(None) is None


def test_venv_missing_still_lets_deps_and_torch_be_selected(tmp_path):
    # the runner creates the venv first regardless, so these stay selectable
    items = by_id(setup_checks.check_all(make_paths(tmp_path),
                                         run=lambda cmd: None))
    assert items["venv"]["state"] == setup_checks.MISSING
    assert items["deps"]["state"] == setup_checks.MISSING
    assert items["torch"]["state"] == setup_checks.MISSING


def test_deps_needs_torch_too(tmp_path):
    # picking deps without torch would still let bitsandbytes pull a plain
    # CPU torch in, with no later step to upgrade it to the CUDA build
    no_venv = by_id(setup_checks.check_all(make_paths(tmp_path),
                                           run=lambda cmd: None))
    assert set(no_venv["deps"]["needs"]) == {"venv", "torch"}
    with_venv = by_id(setup_checks.check_all(make_paths(tmp_path, venv=True),
                                             run=lambda cmd: None))
    assert set(with_venv["deps"]["needs"]) == {"venv", "torch"}


def test_deps_ok_when_imports_succeed(tmp_path):
    items = by_id(setup_checks.check_all(make_paths(tmp_path, venv=True),
                                         run=lambda cmd: "ok"))
    assert items["venv"]["state"] == setup_checks.OK
    assert items["deps"]["state"] == setup_checks.OK


def test_deps_missing_when_import_fails(tmp_path):
    items = by_id(setup_checks.check_all(make_paths(tmp_path, venv=True),
                                         run=lambda cmd: None))
    assert items["deps"]["state"] == setup_checks.MISSING


def test_venv_needs_only_uv_even_without_any_python(tmp_path):
    # no system Python and no python3-venv: uv downloads one, so the
    # environment is still installable, never blocked
    items = by_id(setup_checks.check_all(make_paths(tmp_path, python=False),
                                         run=lambda cmd: None))
    assert items["venv"]["state"] == setup_checks.MISSING
    assert items["venv"]["needs"] == ["uv"]


def test_deps_probe_skips_bitsandbytes_on_macos():
    assert "bitsandbytes" in setup_checks.deps_probe("linux")
    assert "bitsandbytes" in setup_checks.deps_probe("win32")
    assert "bitsandbytes" not in setup_checks.deps_probe("darwin")


def test_torch_detail_per_platform():
    assert setup_checks.torch_detail("2.8.0+cu128", "linux") == "CUDA 12.8"
    assert setup_checks.torch_detail("2.8.0+cpu", "linux") is None
    assert setup_checks.torch_detail("2.8.0", "darwin") == "Metal"
    assert setup_checks.torch_detail(None, "darwin") is None


def test_torch_missing_without_cuda(tmp_path, nvidia):
    def run(cmd):
        return "2.8.0+cpu" if "torch" in cmd[-1] else "ok"
    items = by_id(setup_checks.check_all(make_paths(tmp_path, venv=True),
                                         run=run))
    assert items["torch"]["state"] == setup_checks.MISSING


def test_torch_ok_with_cuda(tmp_path, nvidia):
    def run(cmd):
        return "2.8.0+cu128" if "torch" in cmd[-1] else "ok"
    items = by_id(setup_checks.check_all(make_paths(tmp_path, venv=True),
                                         run=run))
    assert items["torch"]["state"] == setup_checks.OK
    assert "12.8" in items["torch"]["detail"]


def test_model_missing_reports_free_space(tmp_path):
    items = by_id(setup_checks.check_all(make_paths(tmp_path),
                                         run=lambda cmd: None))
    assert items["model"]["state"] == setup_checks.MISSING
    assert "free" in items["model"]["detail"]


def test_model_not_ok_from_a_partial_snapshot(tmp_path):
    # a snapshot folder appears from the very first file downloaded; treating
    # its mere existence as "done" reported a stalled/interrupted download
    # as complete, so only the post-download marker counts
    p = make_paths(tmp_path)
    snap = p.models_dir / setup_checks.MODEL_FOLDER / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_text("x", encoding="utf-8")
    items = by_id(setup_checks.check_all(p, run=lambda cmd: None))
    assert items["model"]["state"] == setup_checks.MISSING


def test_model_ok_once_the_marker_is_written(tmp_path):
    p = make_paths(tmp_path)
    folder = p.models_dir / setup_checks.MODEL_FOLDER
    folder.mkdir(parents=True)
    (folder / setup_checks.MODEL_MARKER).write_text("ok", encoding="utf-8")
    items = by_id(setup_checks.check_all(p, run=lambda cmd: None))
    assert items["model"]["state"] == setup_checks.OK


def test_refresh_live_notices_a_finished_download_without_a_full_recheck(
        tmp_path):
    p = make_paths(tmp_path)
    items = setup_checks.check_all(p, run=lambda cmd: None)
    assert by_id(items)["model"]["state"] == setup_checks.MISSING
    folder = p.models_dir / setup_checks.MODEL_FOLDER
    folder.mkdir(parents=True)
    (folder / setup_checks.MODEL_MARKER).write_text("ok", encoding="utf-8")
    fresh = by_id(setup_checks.refresh_live(items, p))
    assert fresh["model"]["state"] == setup_checks.OK


def test_model_is_required(tmp_path):
    # the server won't start without it either; see SERVER_NEEDS in api.py
    items = by_id(setup_checks.check_all(make_paths(tmp_path),
                                         run=lambda cmd: None))
    assert items["model"]["required"] is True
    assert items["venv"]["required"] is True


def _row(item_id, state, required=True, step_state=None):
    return {"id": item_id, "state": state, "required": required,
            "step_state": step_state}


def test_a_step_is_done_when_its_required_rows_are_ok():
    rows = [_row("uv", "ok"), _row("venv", "ok"), _row("deps", "ok"),
            _row("torch", "missing", required=False),
            _row("model", "missing")]
    steps = setup_checks.step_status(rows)
    assert steps == {"backend": "done", "env": "done", "model": "todo"}


def test_a_running_or_failed_install_shows_on_its_step():
    rows = [_row("uv", "ok", step_state="done"),
            _row("venv", "missing", step_state="running"),
            _row("model", "missing", step_state="failed")]
    steps = setup_checks.step_status(rows)
    assert steps["env"] == "running" and steps["model"] == "failed"


def test_setup_opens_on_the_first_unfinished_step():
    base = {"backend": "done", "env": "done", "model": "done"}
    assert setup_checks.first_step(base) == "run"
    assert setup_checks.first_step({**base, "model": "failed"}) == "model"
    # a fresh machine starts at the provider choice, not at Install
    assert setup_checks.first_step({**base, "env": "todo"}) == "backend"
