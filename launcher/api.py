# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

"""Setup and the server process, glued: what the Aseprite plugin drives
through the supervisor (launcher.daemon). No window, no GUI toolkit."""
import pathlib
import threading

from launcher import paths, platforms, server_proc, setup_checks, setup_steps
from server import catalog
from launcher.setup_checks import model_folder, MODEL_MARKER
from launcher.paths import app_root
from launcher.version import VERSION
from server.config import (DEFAULT_LOCAL_MODEL, DEFAULT_OPENROUTER_MODEL,
                           HOST, VRAM_MODES, load_model_settings, load_port,
                           load_settings, load_vram_mode, mask_key, save_port,
                           save_model_settings, save_settings, save_vram_mode)

SERVER_NEEDS = setup_checks.SERVER_NEEDS
SERVER_NEEDS_LABELS = {"venv": "virtual env", "deps": "dependencies",
                       "torch": "PyTorch", "model": "model",
                       "apikey": "an OpenRouter API key"}
server_needs = setup_checks.server_needs
backend_label = setup_checks.backend_label
POLL_SECONDS = 1.5
OFFLINE = {"state": "offline", "progress": 0.0, "stage": None}
# paths the plugin may override; the code root is wherever Aseprite put us
PATH_KINDS = ("python", "models_dir")
NEED_KEY = "Add an OpenRouter API key, then press Start."


class Api:
    def __init__(self, root: pathlib.Path | None = None):
        self.root = pathlib.Path(root) if root else app_root()
        self.port = load_port()
        self.paths = paths.resolve()
        self.proc = self._new_proc()
        self.health = dict(OFFLINE)
        self.stopped_by_user = False
        self.hint = ""
        self.hint_bad = False
        self.gpu_gb = None  # NVIDIA card memory, filled by the first check
        self.items: list[dict] = []
        self.checking = False
        self.setup_log: list[str] = []
        self.log_live = False
        self.step_state: dict[str, tuple] = {}
        self.was_running = False
        # Start asked while setup was incomplete: finish setup, then start
        self.start_pending = False
        # this Start already ran an install; never loop on a failing one
        self.installed_for_start = False
        self.runner = setup_steps.Runner(self.paths, self._step_event,
                                         self._setup_log)
        threading.Thread(target=self._poll, daemon=True).start()
        self._start_check()

    def _new_proc(self):
        return server_proc.ServerProcess(self.root, self.port,
                                         on_log=lambda line: None,
                                         data=self.paths.data)

    def _say(self, text: str, bad: bool = False) -> None:
        self.hint = text
        self.hint_bad = bad

    def _poll(self) -> None:
        # health off the request path, so an RPC never waits on a socket timeout
        while True:
            self.health = (server_proc.probe(self.port, timeout=1.0)
                           if self.proc.is_alive() else dict(OFFLINE))
            # here too, not only on view: Start must go on with no one polling
            self._settle()
            threading.Event().wait(POLL_SECONDS)

    def _settle(self) -> None:
        """An install that just ended: re-check, so the rows show reality
        (and a pending Start can go on once the check is done)."""
        if self.was_running and not self.runner.is_running():
            self.was_running = False
            self._start_check()

    # ---- RPC methods (launcher.daemon.METHODS)

    def view(self) -> dict:
        """Everything Setup paints, in one round trip: the server for the
        Run step, the checks and settings for the steps before it."""
        setup = self.setup_state()  # refreshes the live checks first
        return {"server": self._snapshot(), "setup": setup}

    def _server_ready(self) -> bool:
        state = {it["id"]: it["state"] for it in self.items}
        return bool(self.items) and all(
            state.get(k) == setup_checks.OK for k in server_needs())

    def _missing(self) -> list[str]:
        """What Start still needs, by row id, for the chosen provider."""
        state = {it["id"]: it["state"] for it in self.items}
        return [k for k in server_needs() if state.get(k) != setup_checks.OK]

    def _install_ids(self) -> list[str]:
        """Required rows that Setup can install and are not OK yet."""
        return [it["id"] for it in self.items
                if it["required"] and it["state"] != setup_checks.OK
                and it["id"] in setup_steps.ORDER]

    def start_server(self) -> dict:
        """The one Start button: install whatever is missing, then start the
        model server. A no-op while it already runs."""
        if self.proc.is_alive():
            return self._snapshot()
        if self.checking or self.runner.is_running():
            # picked up when the check (or install) ends, see _start_check
            self.start_pending = True
            return self._snapshot()
        if self._server_ready():
            self.start_pending = self.installed_for_start = False
            return self._launch()
        if "apikey" in self._missing():
            # no point installing for a cloud backend we cannot call yet
            self.start_pending = self.installed_for_start = False
            self._say(NEED_KEY, bad=True)
            return self._snapshot()
        todo = self._install_ids()
        if todo and not self.installed_for_start:
            self.start_pending = self.installed_for_start = True
            self._say("")
            self.install(todo)
            return self._snapshot()
        # nothing left to install (a missing API key), or the install failed
        self.start_pending = self.installed_for_start = False
        last = self.setup_log[-1] if self.setup_log else ""
        if todo and last:
            self._say(f"Setup did not finish: {last}", bad=True)
        return self._snapshot()

    def _launch(self) -> dict:
        """Start the model server process, on a free port."""
        free = server_proc.pick_port(self.port)
        if free != self.port:
            # the plugin reads the port from server.json, so nothing to redo
            self._say(f"Port {self.port} was busy, moved to {free}.")
            self.port = free
            save_port(free)
        else:
            self._say("")
        self.stopped_by_user = False
        self.proc = self._new_proc()
        self.proc.start()
        return self._snapshot()

    def stop_server(self) -> dict:
        """The one Stop button: cancel a Start's install, or stop the
        model server; a no-op if neither is going on."""
        self.start_pending = self.installed_for_start = False
        if self.runner.is_running():
            self.cancel_install()
        if self.proc.is_alive():
            self.stopped_by_user = True
            self.proc.stop()
            self.proc.lines.append("-- stopped --")
            self.health = dict(OFFLINE)
            self._say("")
        return self._snapshot()

    def toggle_server(self) -> dict:
        if self.proc.is_alive():
            return self.stop_server()
        return self.start_server()

    def shutdown(self) -> None:
        """Everything this process started goes down with it."""
        self.runner.cancel()
        self.proc.stop()

    def set_model(self, values: dict) -> dict:
        """Provider, model, VRAM mode and API key from Setup. The checks
        rerun: the cloud needs a different set of pieces than the local model."""
        if not isinstance(values, dict):
            return self.setup_state()
        # the model menu offers each model once per VRAM mode, so a pick
        # carries both; the mode lives in its own setting
        mode = values.get("vram_mode")
        if mode in VRAM_MODES:
            save_vram_mode(mode)
        save_model_settings(values)
        self._say("Model settings saved. Restart the server to use them."
                  if self.proc.is_alive() else "")
        self.step_state = {}
        self._start_check()
        return self.setup_state()

    def _model_view(self) -> dict:
        s = load_model_settings()
        cat = catalog.catalog_for_ui()
        models_dir = self.paths.models_dir
        for entry in cat:
            marker = models_dir / model_folder(entry["id"]) / MODEL_MARKER
            entry["downloaded"] = marker.is_file()
        return {"provider": s["provider"],
                "local_model": s["local_model"],
                "openrouter_model": s["openrouter_model"],
                "key": mask_key(s["api_key"]),
                "key_source": s["api_key_source"],
                "defaults": {"local_model": DEFAULT_LOCAL_MODEL,
                             "openrouter_model": DEFAULT_OPENROUTER_MODEL},
                "catalog": cat}

    # ---- setup

    def _setup_log(self, line: str, replace: bool = False) -> None:
        if not replace:
            # daemon.log keeps the full story; download bars only live here
            print(line, flush=True)
        if replace and self.setup_log and self.log_live:
            self.setup_log[-1] = line
        else:
            self.setup_log.append(line)
            del self.setup_log[:-200]
        self.log_live = replace

    def _step_event(self, step_id: str, state: str, detail: str) -> None:
        self.step_state[step_id] = (state, detail)

    def _start_check(self) -> None:
        # deps and torch probes import torch, seconds each; keep them off RPCs
        if self.checking:
            return
        self.checking = True

        def work():
            resolved = paths.resolve()
            items = setup_checks.check_all(resolved)
            self.gpu_gb = platforms.gpu_memory_gb(paths.run_command)
            self.paths = resolved
            self.items = items
            self.step_state = {}
            self.checking = False
            if self.start_pending:
                self.start_pending = False
                self.start_server()

        threading.Thread(target=work, daemon=True).start()

    def _refresh_live(self) -> None:
        # the model marker is filesystem-only, cheap to re-check every poll
        if self.items and not self.runner.is_running() and not self.checking:
            self.items = setup_checks.refresh_live(self.items, self.paths)

    def recheck(self) -> dict:
        self._start_check()
        return self.setup_state()

    def setup_state(self) -> dict:
        self._settle()
        running = self.runner.is_running()
        self._refresh_live()
        rows = []
        for item in self.items:
            state, detail = self.step_state.get(item["id"], (None, ""))
            rows.append({**item, "step_state": state, "step_detail": detail})
        steps = setup_checks.step_status(rows)
        saved = load_settings()
        overrides = [k for k in PATH_KINDS if saved.get(k)]
        return {"items": rows, "paths": self._paths_view(), "running": running,
                "checking": self.checking, "steps": steps,
                "first_step": setup_checks.first_step(steps),
                "step_items": setup_checks.STEP_ITEMS,
                "overrides": overrides, "vram_mode": load_vram_mode(),
                "gpu": platforms.gpu_backend(),
                "gpu_gb": self.gpu_gb,
                "model": self._model_view(),
                "log": "\n".join(self.setup_log[-200:])}

    def _paths_view(self) -> dict:
        p = self.paths
        return {"root": str(p.root), "data": str(p.data),
                "python": str(p.python) if p.python else "",
                "models_dir": str(p.models_dir)}

    def install(self, ids) -> dict:
        if self.runner.is_running():
            return self.setup_state()
        self.step_state = {}
        self.setup_log = []
        self.runner = setup_steps.Runner(self.paths, self._step_event,
                                         self._setup_log)
        self.runner.start(list(ids))
        self.was_running = True
        return self.setup_state()

    def cancel_install(self) -> dict:
        self.runner.cancel()
        self._setup_log("-- cancelled --")
        return self.setup_state()

    def set_path(self, kind: str, value: str) -> dict:
        """A path picked in the plugin's own file dialog."""
        if kind in PATH_KINDS and value:
            save_settings({kind: str(value)})
            self.paths = paths.resolve()
            self.step_state = {}
        return self.recheck()

    def reset_path(self, kind: str) -> dict:
        # drop the manual override so this path auto-detects again
        if kind in PATH_KINDS:
            save_settings({kind: None})
            self.paths = paths.resolve()
            self.step_state = {}
        return self.recheck()

    # ---- state assembly

    def _running_step(self) -> str | None:
        for step_id, (state, _) in self.step_state.items():
            if state == "running":
                return step_id
        return None

    def _server_view(self) -> tuple:
        if self.runner.is_running():
            step = self._running_step()
            label = setup_steps.LABELS.get(step, "Installing")
            return "busy", f"Setting up: {label}", 0.0
        if self.start_pending:
            return "busy", "Checking setup", 0.0
        if self.proc.is_alive():
            health = self.health
            if health["state"] == "ready":
                return "ready", "Ready", 0.0
            if health["state"] == "loading":
                stage = health["stage"] or "model"
                return "busy", f"Loading: {stage}", health["progress"]
            return "busy", "Starting", 0.0
        if self.proc.proc is not None and not self.stopped_by_user:
            return "error", "Server crashed", 0.0
        if self.proc.proc is None:
            # while the background check is still running self.items is
            # empty, which would otherwise read as "not ready" for a split
            # second even on a machine that's perfectly set up
            if self.items and not self._server_ready():
                return "", "Need to configure", 0.0
            return "", "Not started", 0.0
        return "", "Server stopped", 0.0

    def _snapshot(self) -> dict:
        tone, label, progress = self._server_view()
        not_ready = self.items and not self.proc.is_alive() \
            and not self._server_ready() and not self.start_pending \
            and not self.hint
        hint = self.hint
        if not_ready:
            missing = self._missing()
            listed = [SERVER_NEEDS_LABELS[k] for k in missing]
            listed = (listed[0] if len(listed) < 2 else
                      ", ".join(listed[:-1]) + " and " + listed[-1])
            # a key is pasted, not installed: ask for it before anything else
            hint = (NEED_KEY if "apikey" in missing
                    else f"Start installs {listed} first.")
        hint_bad = True if not_ready else self.hint_bad
        installing = self.runner.is_running()
        if installing and self.setup_log:
            # the live line: which package, or how far the model download is
            hint, hint_bad = self.setup_log[-1], False
        return {
            "version": VERSION,
            "url": f"ws://{HOST}:{self.port}",
            "port": self.port,
            "backend": backend_label(),
            "tone": tone,
            "label": label,
            "progress": progress,
            "running": self.proc.is_alive(),
            "installing": installing,
            "pending": self.start_pending,
            "can_start": self._server_ready(),
            "hint": hint,
            "hint_bad": hint_bad,
            "vram_mode": load_vram_mode(),
            "log": "\n".join(self.proc.lines[-200:]),
        }
