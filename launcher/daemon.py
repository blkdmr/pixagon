"""The supervisor the Aseprite plugin starts: Setup and the model server,
driven over a local WebSocket. One per user; it exits when the plugin stops
talking to it (Aseprite closed or crashed) and takes the server along.

    uv run --no-project --with websockets python -m launcher.daemon --data DIR

Wire format, one JSON object per message:
    request  {"id": 1, "method": "view", "params": {...}}
    reply    {"id": 1, "type": "reply", "result": ...}
    error    {"id": 1, "type": "error", "message": "..."}
"""
import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

import websockets

from launcher.server_proc import pick_port
from launcher.version import VERSION
from server.config import DATA_ENV, HOST

# well clear of the model server's 8765 and pick_port's 20-port search above it
CONTROL_PORT = 8790
STATE_FILE = "server.json"
LOG_FILE = "daemon.log"
# the plugin pings every 60 s while Aseprite runs; three misses mean it's gone
IDLE_SECONDS = 180

# RPC name -> (Api method, parameter names in call order)
METHODS = {
    "view": ("view", ()),
    "recheck": ("recheck", ()),
    "start_server": ("start_server", ()),
    "stop_server": ("stop_server", ()),
    "toggle_server": ("toggle_server", ()),
    "set_model": ("set_model", ("values",)),
    "install": ("install", ("ids",)),
    "cancel_install": ("cancel_install", ()),
    "set_path": ("set_path", ("kind", "value")),
    "reset_path": ("reset_path", ("kind",)),
}


def log(text: str) -> None:
    # stdout goes to daemon.log (the plugin redirects it there)
    print(f"[daemon] {text}", flush=True)


class Supervisor:
    def __init__(self, api, data: pathlib.Path, control_port: int,
                 idle: float = IDLE_SECONDS, clock=time.monotonic):
        self.api = api
        self.data = pathlib.Path(data)
        self.control_port = control_port
        self.idle_after = idle
        self.clock = clock
        self.last_seen = clock()
        self.shutdown_requested = False

    @property
    def state_path(self) -> pathlib.Path:
        return self.data / STATE_FILE

    def handle(self, raw: str) -> dict:
        """One request in, one reply out. Never raises: a bad request or a
        failing method must not take the supervisor (and the server) down."""
        self.last_seen = self.clock()
        try:
            msg = json.loads(raw)
        except ValueError:
            return {"id": None, "type": "error", "message": "not JSON"}
        if not isinstance(msg, dict):
            return {"id": None, "type": "error", "message": "not an object"}
        mid, method = msg.get("id"), msg.get("method")

        def error(text):
            return {"id": mid, "type": "error", "message": text}

        if method == "ping":
            return {"id": mid, "type": "reply",
                    "result": {"pong": True, "version": VERSION}}
        if method == "shutdown":
            self.shutdown_requested = True
            return {"id": mid, "type": "reply", "result": None}
        if method not in METHODS:
            return error(f"unknown method: {method}")
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            return error("params must be an object")
        attr, names = METHODS[method]
        missing = [n for n in names if n not in params]
        if missing:
            return error(f"missing params: {', '.join(missing)}")
        try:
            result = getattr(self.api, attr)(*(params[n] for n in names))
        except Exception as e:  # reported to the plugin, logged here
            log(f"{method} failed: {e!r}")
            return error(f"{method} failed: {e}")
        # start_server may have moved the model server to another port
        self.write_state()
        return {"id": mid, "type": "reply", "result": result}

    def write_state(self) -> None:
        """Ports for the plugin: where to send control and generation."""
        self.data.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"control": self.control_port,
                                   "port": self.api.port,
                                   "pid": os.getpid()}), encoding="utf-8")
        # replace, so the plugin never reads a half-written file
        os.replace(tmp, self.state_path)

    def clear_state(self) -> None:
        """Remove server.json if it is still ours (not a newer supervisor's)."""
        try:
            if json.loads(self.state_path.read_text(encoding="utf-8")).get(
                    "pid") == os.getpid():
                self.state_path.unlink()
        except (OSError, ValueError, AttributeError):
            pass

    def is_idle(self) -> bool:
        return self.clock() - self.last_seen > self.idle_after

    async def serve(self) -> None:
        """Answer requests until shutdown or idle, then stop everything."""
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        async def handler(ws):
            async for raw in ws:
                # Api calls touch files and threads; keep them off the loop
                reply = await loop.run_in_executor(None, self.handle, raw)
                await ws.send(json.dumps(reply))
                if self.shutdown_requested:
                    stop.set()

        async with websockets.serve(handler, HOST, self.control_port):
            self.write_state()
            log(f"listening on {HOST}:{self.control_port}")
            while not stop.is_set() and not self.is_idle():
                try:
                    await asyncio.wait_for(stop.wait(), 1.0)
                except asyncio.TimeoutError:
                    pass
        log("shutdown requested" if self.shutdown_requested else
            f"no word from the plugin for {self.idle_after:.0f} s")
        self.api.shutdown()
        self.clear_state()


def running_control_port(data: pathlib.Path, timeout: float = 2.0) -> int | None:
    """The control port of a supervisor that already answers, else None."""
    try:
        port = json.loads((pathlib.Path(data) / STATE_FILE).read_text(
            encoding="utf-8"))["control"]
    except (OSError, ValueError, KeyError, TypeError):
        return None

    async def ask():
        async with websockets.connect(f"ws://{HOST}:{port}",
                                      open_timeout=timeout) as ws:
            await ws.send(json.dumps({"id": 0, "method": "ping"}))
            return json.loads(await asyncio.wait_for(ws.recv(), timeout))

    try:
        reply = asyncio.run(ask())
    except Exception:
        return None
    return port if reply.get("type") == "reply" else None


def log_to_file(data: pathlib.Path) -> None:
    """Send stdout and stderr to <data>/daemon.log. Started windowless on
    Windows there is no console to write to (stdout may even be None), and
    the plugin cannot redirect a `start`ed process, so we do it ourselves."""
    pathlib.Path(data).mkdir(parents=True, exist_ok=True)
    out = open(pathlib.Path(data) / LOG_FILE, "a", encoding="utf-8",
               buffering=1)  # line-buffered: the log is readable while live
    sys.stdout = sys.stderr = out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m launcher.daemon")
    parser.add_argument("--data", help="data folder (default: the per-user one)")
    parser.add_argument("--port", type=int, default=CONTROL_PORT)
    parser.add_argument("--idle", type=float, default=IDLE_SECONDS)
    args = parser.parse_args(argv)
    if args.data:
        # before the Api resolves its paths: everything reads this variable
        os.environ[DATA_ENV] = args.data
    from server.config import data_dir
    data = data_dir()
    log_to_file(data)
    existing = running_control_port(data)
    if existing:
        log(f"already running on port {existing}")
        return
    # imported late: building the Api starts its checks right away
    from launcher.api import Api
    supervisor = Supervisor(Api(), data, pick_port(args.port), idle=args.idle)
    asyncio.run(supervisor.serve())


if __name__ == "__main__":
    main()
