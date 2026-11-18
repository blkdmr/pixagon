# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

"""Running the server subprocess and reading its state."""
import asyncio
import ctypes
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import threading

import websockets

from launcher import platforms
from launcher.platforms import NO_WINDOW
from server.config import HOST

if sys.platform == "win32":
    from ctypes import wintypes

MAX_LINES = 500
BAR_WIDTH = 16
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# a tqdm bar (desc, pct, n, total, remaining); text-mode splits its \r redraws
_BAR = re.compile(
    r"^(.+?):\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)(?:\s*\[[^<\]]*<([^,\]]+))?")
# a bar we already formatted; captures its description for in-place updates
_FMT_BAR = re.compile(r"^(.+?):\s*\d+% \[")
# no per-item data past text_encoder; the count/percent here is misleading
_NO_BAR_DESCS = {"Loading pipeline components..."}


def format_bar(m) -> str:
    """A matched tqdm bar redrawn clean: fixed width, percent, count, and ETA."""
    if m.group(1) in _NO_BAR_DESCS:
        return m.group(1)
    pct = int(m.group(2))
    filled = round(pct / 100 * BAR_WIDTH)
    bar = "#" * filled + " " * (BAR_WIDTH - filled)
    out = f"{m.group(1)}: {pct:>3}% [{bar}] {m.group(3)}/{m.group(4)}"
    eta = m.group(5)
    if pct < 100 and eta and eta.strip() != "?":
        out += f" ETA {eta.strip()}"
    return out


_KILL_ON_JOB_CLOSE = 0x2000
_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


if sys.platform == "win32":
    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOperationCount", "WriteOperationCount",
                     "OtherOperationCount", "ReadTransferCount",
                     "WriteTransferCount", "OtherTransferCount")]


    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]


    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits),
                    ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _kernel32():
    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    # the default int restype truncates handles on 64 bit
    lib.CreateJobObjectW.restype = wintypes.HANDLE
    lib.OpenProcess.restype = wintypes.HANDLE
    return lib


def make_kill_on_close_job():
    """A job whose processes all die when the last handle to it closes.

    That covers both a clean stop and a launcher that was killed outright,
    and it takes grandchildren with it, which terminate() cannot do.
    """
    if sys.platform != "win32":
        return None
    lib = _kernel32()
    job = lib.CreateJobObjectW(None, None)
    if not job:
        return None
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
    if not lib.SetInformationJobObject(job, _EXTENDED_LIMIT_INFORMATION,
                                       ctypes.byref(limits),
                                       ctypes.sizeof(limits)):
        lib.CloseHandle(job)
        return None
    return job


def close_job(job) -> None:
    """Closing the last handle is what kills the tree, so it is worth a name."""
    if job:
        _kernel32().CloseHandle(job)


def assign_to_job(job, pid: int) -> bool:
    if not job or sys.platform != "win32":
        return False
    lib = _kernel32()
    handle = lib.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE,
                             False, pid)
    if not handle:
        return False
    try:
        return bool(lib.AssignProcessToJobObject(job, handle))
    finally:
        lib.CloseHandle(handle)


# POSIX stand-in for the job object, in plain sh so it needs no Python (uv
# may not have fetched one yet). It starts the real command in this (new)
# process group, and a watcher holds the read end of a pipe whose write end
# only the launcher has. However the launcher dies, the OS closes that end,
# the read returns, and the watcher takes the whole group down.
LIFELINE = """\
exec 3<&0 0</dev/null
"$@" 3<&- &
kid=$!
( trap '' TERM
  while read -r _ <&3; do :; done
  kill -TERM 0 2>/dev/null; sleep 3; kill -KILL 0 2>/dev/null
) >/dev/null 2>&1 &
watcher=$!
exec 3<&-
# set after the spawn: an ignored signal would survive into the child's exec
trap '' TERM
wait "$kid"
code=$?
kill -TERM 0 2>/dev/null
kill -KILL "$watcher" 2>/dev/null
exit "$code"
"""


class TreeGuard:
    """Ties a child and everything it spawns to the launcher's lifetime.

    Windows: a kill-on-close job object. macOS/Linux: a new process group
    led by the LIFELINE wrapper. Either way kill() takes the whole tree,
    grandchildren included, which terminate() alone cannot do.
    """

    def __init__(self):
        self.job = None
        self.proc = None

    @classmethod
    def popen(cls, cmd, **kwargs):
        """(Popen, guard) for cmd, tied to the launcher's lifetime."""
        guard = cls()
        cmd = [str(part) for part in cmd]
        if sys.platform == "win32":
            proc = subprocess.Popen(cmd, creationflags=NO_WINDOW, **kwargs)
            guard.job = make_kill_on_close_job()
            assign_to_job(guard.job, proc.pid)
        else:
            proc = subprocess.Popen(
                ["/bin/sh", "-c", LIFELINE, "pixagon-guard", *cmd],
                stdin=subprocess.PIPE, start_new_session=True, **kwargs)
        guard.proc = proc
        return proc, guard

    def terminate(self) -> None:
        """Ask nicely: the direct child on Windows, the whole group elsewhere."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        if sys.platform == "win32":
            proc.terminate()
        else:
            _signal_group(proc.pid, signal.SIGTERM)

    def kill(self) -> None:
        """Take the whole tree down now."""
        if sys.platform == "win32":
            job, self.job = self.job, None
            close_job(job)
            return
        proc = self.proc
        if proc is None:
            return
        _signal_group(proc.pid, signal.SIGKILL)
        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass


def _signal_group(pgid: int, sig) -> None:
    try:
        os.killpg(pgid, sig)
    except (OSError, AttributeError):
        pass


def clean_line(raw: str) -> str:
    # tqdm redraws its bar with \r; the log keeps only the final state
    return _ANSI.sub("", raw.split("\r")[-1]).rstrip()


def venv_python(root: pathlib.Path) -> pathlib.Path | None:
    return platforms.find_venv_python(pathlib.Path(root) / ".venv")


def port_is_free(port: int, host: str = HOST) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) != 0


def pick_port(preferred: int) -> int:
    for candidate in range(preferred, preferred + 20):
        if port_is_free(candidate):
            return candidate
    return preferred


def probe(port: int, timeout: float = 2.0) -> dict:
    """The same ping the panel sends, so both agree on the state."""
    async def ask():
        url = f"ws://{HOST}:{port}"
        async with websockets.connect(url, open_timeout=timeout,
                                      ping_interval=None) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            raw = await asyncio.wait_for(ws.recv(), timeout)
            return json.loads(raw)

    offline = {"state": "offline", "progress": 0.0, "stage": None}
    try:
        pong = asyncio.run(ask())
    except Exception:
        return offline
    if not isinstance(pong, dict) or pong.get("type") != "pong":
        return offline
    ready = pong.get("model", "ready") == "ready"
    return {"state": "ready" if ready else "loading",
            "progress": float(pong.get("progress") or 0.0),
            "stage": pong.get("stage")}


class ServerProcess:
    def __init__(self, root, port: int, on_log, data=None):
        self.root = pathlib.Path(root)
        # the writable folder (.venv, models, output); the code root unless
        # the code sits in a macOS .app, see paths.data_root
        self.data = pathlib.Path(data) if data else self.root
        self.port = port
        self.on_log = on_log
        self.lines: list[str] = []
        self.proc: subprocess.Popen | None = None
        # held for the launcher's lifetime: dropping it kills the server tree
        self.guard = None
        found = venv_python(self.data)
        self.python = str(found) if found else sys.executable
        self.args = ["-u", "-m", "server.main"]

    def start(self) -> None:
        if self.is_alive():
            return
        # run from the data folder so the server's relative models/ and
        # output/ land there; PYTHONPATH still finds the server package
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(self.root), env.get("PYTHONPATH")) if p)
        self.data.mkdir(parents=True, exist_ok=True)
        self.proc, self.guard = TreeGuard.popen(
            [self.python, *self.args], cwd=str(self.data),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env)
        threading.Thread(target=self._pump, daemon=True).start()

    def _bar_line_index(self, desc: str):
        """Index of the same-named bar in the trailing run of bar lines, if any."""
        for i in range(len(self.lines) - 1, -1, -1):
            line = self.lines[i]
            if line in _NO_BAR_DESCS:
                cur = line
            else:
                m = _FMT_BAR.match(line)
                if not m:
                    return None
                cur = m.group(1)
            if cur == desc:
                return i
        return None

    def _pump(self) -> None:
        proc = self.proc
        if not proc or not proc.stdout:
            return
        for raw in proc.stdout:
            line = clean_line(raw)
            if not line:
                continue
            bar = _BAR.match(line)
            if bar:
                formatted = format_bar(bar)
                idx = self._bar_line_index(bar.group(1))
                if idx is not None:
                    self.lines[idx] = formatted  # same bar redrawing in place
                else:
                    self.lines.append(formatted)
                    del self.lines[:-MAX_LINES]
                self.on_log(formatted)
                continue
            self.lines.append(line)
            del self.lines[:-MAX_LINES]
            self.on_log(line)

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def wait_for_exit(self, timeout: float = 10) -> None:
        if self.proc:
            self.proc.wait(timeout=timeout)

    def stop(self) -> None:
        guard, self.guard = self.guard, None
        if not self.is_alive():
            if guard:
                guard.kill()
            return
        proc = self.proc
        assert proc
        if guard:
            guard.terminate()
        else:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if guard:
                guard.kill()
            proc.kill()
            proc.wait(timeout=5)
        # terminate() can spare grandchildren; the guard sweeps them up
        if guard:
            guard.kill()
