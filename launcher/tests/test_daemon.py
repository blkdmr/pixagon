import asyncio
import json
import os

import websockets

from launcher import daemon
from launcher.server_proc import pick_port


class FakeApi:
    """Records calls instead of running Setup or a server."""

    def __init__(self):
        self.port = 8765
        self.calls = []
        self.shut = False

    def view(self):
        self.calls.append(("view",))
        return {"server": {}, "setup": {}}

    def set_model(self, values):
        self.calls.append(("set_model", values))
        return {"saved": values}

    def set_path(self, kind, value):
        self.calls.append(("set_path", kind, value))
        return {}

    def start_server(self):
        # a busy port moves the server; server.json must follow
        self.port = 8766
        return {"running": True}

    def install(self, ids):
        raise RuntimeError("disk full")

    def shutdown(self):
        self.shut = True


def make(tmp_path, clock=None):
    api = FakeApi()
    sup = daemon.Supervisor(api, tmp_path, 8790, idle=10,
                            clock=clock or (lambda: 0.0))
    return api, sup


def ask(sup, method, params=None, mid=1):
    msg = {"id": mid, "method": method}
    if params is not None:
        msg["params"] = params
    return sup.handle(json.dumps(msg))


def test_ping_answers_without_touching_the_api(tmp_path):
    api, sup = make(tmp_path)
    reply = ask(sup, "ping", mid=7)
    assert reply["id"] == 7 and reply["type"] == "reply"
    assert reply["result"]["pong"] is True
    assert api.calls == []


def test_methods_reach_the_api_with_their_params(tmp_path):
    api, sup = make(tmp_path)
    assert ask(sup, "view")["result"] == {"server": {}, "setup": {}}
    assert ask(sup, "set_model", {"values": {"provider": "local"}})["result"] \
        == {"saved": {"provider": "local"}}
    ask(sup, "set_path", {"kind": "models_dir", "value": "/big"})
    assert api.calls == [("view",), ("set_model", {"provider": "local"}),
                         ("set_path", "models_dir", "/big")]


def test_bad_requests_get_errors_not_crashes(tmp_path):
    _, sup = make(tmp_path)
    assert sup.handle("{nope")["type"] == "error"
    assert sup.handle("[1, 2]")["type"] == "error"
    assert "unknown method" in ask(sup, "rm_rf")["message"]
    assert "missing params: values" in ask(sup, "set_model", {})["message"]
    assert ask(sup, "view", params=[1])["type"] == "error"


def test_a_failing_method_is_reported_and_the_supervisor_lives_on(tmp_path):
    _, sup = make(tmp_path)
    reply = ask(sup, "install", {"ids": ["venv"]})
    assert reply["type"] == "error" and "disk full" in reply["message"]
    assert ask(sup, "ping")["type"] == "reply"


def test_server_json_tracks_both_ports(tmp_path):
    _, sup = make(tmp_path)
    sup.write_state()
    state = json.loads((tmp_path / daemon.STATE_FILE).read_text())
    assert state == {"control": 8790, "port": 8765, "pid": os.getpid()}
    ask(sup, "start_server")
    state = json.loads((tmp_path / daemon.STATE_FILE).read_text())
    assert state["port"] == 8766


def test_clear_state_leaves_another_supervisors_file_alone(tmp_path):
    _, sup = make(tmp_path)
    (tmp_path / daemon.STATE_FILE).write_text(json.dumps({"pid": -1}))
    sup.clear_state()
    assert (tmp_path / daemon.STATE_FILE).exists()
    sup.write_state()
    sup.clear_state()
    assert not (tmp_path / daemon.STATE_FILE).exists()


def test_idle_counts_from_the_last_request(tmp_path):
    now = [0.0]
    _, sup = make(tmp_path, clock=lambda: now[0])
    now[0] = 9.0
    assert not sup.is_idle()
    ask(sup, "ping")
    now[0] = 18.0
    assert not sup.is_idle()
    now[0] = 20.5
    assert sup.is_idle()


def test_shutdown_over_the_socket_stops_everything(tmp_path):
    api = FakeApi()
    port = pick_port(daemon.CONTROL_PORT + 100)
    sup = daemon.Supervisor(api, tmp_path, port, idle=30)

    async def scenario():
        serving = asyncio.create_task(sup.serve())
        for _ in range(50):  # wait for the listener
            if (tmp_path / daemon.STATE_FILE).exists():
                break
            await asyncio.sleep(0.05)
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"id": 1, "method": "view"}))
            assert json.loads(await ws.recv())["type"] == "reply"
            await ws.send(json.dumps({"id": 2, "method": "shutdown"}))
            assert json.loads(await ws.recv())["id"] == 2
        await asyncio.wait_for(serving, 10)

    asyncio.run(scenario())
    assert api.shut
    assert not (tmp_path / daemon.STATE_FILE).exists()


def test_no_running_supervisor_without_a_live_one(tmp_path):
    assert daemon.running_control_port(tmp_path) is None
    (tmp_path / daemon.STATE_FILE).write_text(
        json.dumps({"control": pick_port(daemon.CONTROL_PORT + 150)}))
    assert daemon.running_control_port(tmp_path, timeout=0.5) is None


def test_version_matches_the_extension_manifest():
    import pathlib
    from launcher.version import VERSION
    manifest = pathlib.Path(__file__).resolve().parents[2] / "plugin" / \
        "package.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["version"] \
        == VERSION
