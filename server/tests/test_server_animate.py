"""Integration test: animation over WebSocket with FakeKlein."""
import asyncio
import json
import threading
import pytest
import websockets
from PIL import Image

import server.main as srv
from server.protocol import image_to_b64, image_from_raw

HOST, PORT = "127.0.0.1", 8797


class FakeKlein:
    """Minimal pipeline for animation integration tests."""
    from server.instruct import KleinPipeline
    variant_seeds = staticmethod(KleinPipeline.variant_seeds)

    def txt2img(self, prompt, target_size, variants=1, on_progress=None,
                seeds=None):
        if on_progress:
            on_progress(1.0)
        return [Image.new("RGBA", (512, 512), (255, 0, 0, 255))
                for _ in range(variants)]

    def edit_by_instruction(self, instruction, image, variants=1,
                            on_progress=None, seeds=None):
        return [Image.new("RGBA", image.size, (0, 255, 0, 255))
                for _ in range(variants)]


@pytest.fixture()
def server_thread(monkeypatch):
    from server import models
    models.reset()
    models.register("klein", FakeKlein)
    monkeypatch.setattr(srv, "DEBUG_SAVE", False)
    loop = asyncio.new_event_loop()
    stop = loop.create_future()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(srv.serve(HOST, PORT, stop, on_ready=ready.set))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert ready.wait(5)
    yield
    loop.call_soon_threadsafe(stop.set_result, None)
    t.join(timeout=5)
    models.reset()


def test_animate_returns_anim_result(server_thread):
    async def go():
        async with websockets.connect(f"ws://{HOST}:{PORT}",
                                      max_size=64 * 2**20) as ws:
            await ws.send(json.dumps({
                "id": "anim1", "mode": "animate",
                "prompt": "knight walking right",
                "target_size": [32, 32],
                "animation": {
                    "frame_count": 4, "strength": 0.55,
                    "loop": True, "fps": 8,
                    "preset": "walk_cycle",
                },
            }))
            msgs = []
            while True:
                msg = json.loads(await ws.recv())
                msgs.append(msg)
                if msg["type"] in ("anim_result", "error"):
                    return msgs

    msgs = asyncio.run(go())
    result = msgs[-1]
    assert result["type"] == "anim_result"
    assert len(result["frame_images"]) == 4
    assert len(result["frame_durations"]) == 4
    assert result["frame_durations"] == [125, 125, 125, 125]  # 1000/8 = 125
    assert result["loop"] is True
    assert len(result["seeds"]) == 4

    # progress messages precede the result
    progress = [m for m in msgs if m["type"] == "progress"]
    assert len(progress) > 0

    # frames decode to valid images at target size
    for fi in result["frame_images"]:
        img = image_from_raw(fi)
        assert img.size == (32, 32)


def test_animate_with_starting_frame(server_thread):
    src = Image.new("RGBA", (32, 32), (100, 100, 200, 255))
    async def go():
        async with websockets.connect(f"ws://{HOST}:{PORT}",
                                      max_size=64 * 2**20) as ws:
            await ws.send(json.dumps({
                "id": "anim2", "mode": "animate",
                "prompt": "knight idle",
                "target_size": [32, 32],
                "frames": [{"image": image_to_b64(src)}],
                "animation": {
                    "frame_count": 3, "strength": 0.35,
                    "loop": True, "fps": 8,
                    "preset": "idle",
                },
            }))
            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] in ("anim_result", "error"):
                    return msg

    result = asyncio.run(go())
    assert result["type"] == "anim_result"
    assert len(result["frame_images"]) == 3


def test_animate_without_preset(server_thread):
    async def go():
        async with websockets.connect(f"ws://{HOST}:{PORT}",
                                      max_size=64 * 2**20) as ws:
            await ws.send(json.dumps({
                "id": "anim3", "mode": "animate",
                "prompt": "dragon breathing fire",
                "target_size": [64, 64],
                "animation": {
                    "frame_count": 6, "strength": 0.60,
                    "loop": False, "fps": 12,
                    "preset": None,
                },
            }))
            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] in ("anim_result", "error"):
                    return msg

    result = asyncio.run(go())
    assert result["type"] == "anim_result"
    assert len(result["frame_images"]) == 6
    assert result["loop"] is False
    # 1000/12 ≈ 83
    assert result["frame_durations"][0] == 83
