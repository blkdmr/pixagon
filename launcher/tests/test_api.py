"""The one Start/Stop button: Api.start_server installs what is missing,
then starts the server. Checks, the installer and the server are fakes."""
import pathlib

import pytest

from launcher import api as api_mod
from launcher import paths, setup_checks
from server import config

OK, MISSING, BLOCKED = setup_checks.OK, setup_checks.MISSING, setup_checks.BLOCKED


class FakeRunner:
    def __init__(self, paths_, on_event, on_log):
        self.on_log = on_log
        self.started = None
        self.running = False
        self.cancelled = False

    def start(self, ids):
        self.started = list(ids)
        self.running = True

    def is_running(self):
        return self.running

    def cancel(self):
        self.cancelled = True
        self.running = False

    def finish(self, log_line="done"):
        self.on_log(log_line)
        self.running = False


class FakeProc:
    def __init__(self, *a, **k):
        self.alive = False
        self.proc = None
        self.lines = []

    def is_alive(self):
        return self.alive

    def start(self):
        self.alive = True
        self.proc = object()

    def stop(self):
        self.alive = False


class SyncThread:
    """Runs the background check at once, so tests need no waiting."""

    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()


ROWS = {}  # row id -> state, what the next check reports


def row(item_id, state, required=True):
    return {"id": item_id, "label": item_id, "state": state, "detail": "",
            "required": required, "needs": []}


@pytest.fixture
def api(monkeypatch, tmp_path):
    ROWS.clear()
    ROWS.update(uv=OK, venv=OK, torch=OK, deps=OK, model=OK)
    monkeypatch.setattr(api_mod.Api, "_poll", lambda self: None)
    monkeypatch.setattr(api_mod.threading, "Thread", SyncThread)
    monkeypatch.setattr(api_mod.setup_steps, "Runner", FakeRunner)
    monkeypatch.setattr(api_mod.server_proc, "ServerProcess", FakeProc)
    monkeypatch.setattr(api_mod.server_proc, "pick_port", lambda p: p)
    monkeypatch.setattr(api_mod.platforms, "gpu_memory_gb", lambda run: None)
    monkeypatch.setattr(paths, "resolve", lambda *a, **k: paths.Paths(
        root=tmp_path, python=None, python_version=None,
        models_dir=tmp_path / "models", data=tmp_path))

    def check_all(*a, **k):
        local = setup_checks.is_local(config.load_model_settings())
        out = [row(i, s) for i, s in ROWS.items()
               if local or i not in ("torch", "model")]
        if not local:
            out.append(row("apikey", ROWS.get("apikey", BLOCKED)))
        return out

    monkeypatch.setattr(setup_checks, "check_all", check_all)
    return api_mod.Api(root=pathlib.Path(tmp_path))


def test_start_with_everything_in_place_launches_at_once(api):
    snap = api.start_server()
    assert api.proc.is_alive() and snap["running"]
    assert api.runner.started is None


def test_start_installs_what_is_missing_then_launches(api):
    ROWS.update(venv=MISSING, deps=MISSING, model=MISSING)
    api._start_check()
    snap = api.start_server()
    assert api.runner.started == ["venv", "deps", "model"]
    assert snap["installing"] and snap["pending"] and not snap["running"]
    assert snap["label"].startswith("Setting up")
    # the install ends; the next poll re-checks and the server starts
    ROWS.update(venv=OK, deps=OK, model=OK)
    api.runner.finish()
    api._settle()
    assert api.proc.is_alive()
    assert not api.start_pending


def test_a_failed_install_is_reported_not_retried(api):
    ROWS.update(torch=MISSING)
    api._start_check()
    api.start_server()
    first = api.runner
    first.finish("error: no matching torch wheel")
    api._settle()  # torch still missing after the check
    assert not api.proc.is_alive()
    assert api.runner is first  # no second install
    assert "no matching torch wheel" in api._snapshot()["hint"]
    assert not api._snapshot()["pending"]


def test_start_during_the_first_check_waits_for_it(api):
    api.checking = True
    snap = api.start_server()
    assert snap["pending"] and not api.proc.is_alive()
    api.checking = False
    api._start_check()  # the check ends, the pending Start goes on
    assert api.proc.is_alive()


def test_cloud_without_a_key_asks_for_it_and_installs_nothing(api):
    config.save_model_settings({"provider": "openrouter"})
    ROWS.update(venv=MISSING, deps=MISSING)
    api._start_check()
    snap = api.start_server()
    assert api.runner.started is None
    assert snap["hint"] == api_mod.NEED_KEY


def test_cloud_with_a_key_installs_only_the_light_stack(api):
    config.save_model_settings({"provider": "openrouter",
                                "openrouter_api_key": "sk-test"})
    ROWS.update(venv=MISSING, deps=MISSING, apikey=OK)
    api._start_check()
    api.start_server()
    assert api.runner.started == ["venv", "deps"]


def test_stop_cancels_a_starts_install(api):
    ROWS.update(model=MISSING)
    api._start_check()
    api.start_server()
    snap = api.stop_server()
    assert api.runner.cancelled
    assert not snap["pending"]
    api._settle()
    assert not api.proc.is_alive()  # the cancelled Start does not come back


def test_stop_stops_a_running_server(api):
    api.start_server()
    assert not api.stop_server()["running"]


def test_setup_dialog_defaults_match_the_server():
    # plugin/setup.lua fills its fields before the supervisor answers
    lua = (pathlib.Path(__file__).resolve().parents[2] / "plugin" /
           "setup.lua").read_text(encoding="utf-8")
    assert f'"{config.DEFAULT_LOCAL_MODEL}"' in lua
    assert f'"{config.DEFAULT_OPENROUTER_MODEL}"' in lua
    assert f'provider = "{config.DEFAULT_PROVIDER}"' in lua
