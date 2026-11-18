"""The whole cloud path: panel request -> server -> HTTP -> postprocess ->
result, against a local stand-in for OpenRouter's chat-completions API."""
import asyncio
import base64
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import websockets
from PIL import Image

import server.main as srv
from server import config, models
from server.protocol import image_from_raw, image_to_b64

HOST = "127.0.0.1"


def free_port():
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def picture(size=(512, 512)):
    """A red square on white, like a sprite on a plain background."""
    img = Image.new("RGB", size, (255, 255, 255))
    for x in range(size[0] // 4, 3 * size[0] // 4):
        for y in range(size[1] // 4, 3 * size[1] // 4):
            img.putpixel((x, y), (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def images_response(size=(512, 512)):
    """Dedicated /api/v1/images-style response with base64 image data."""
    buf = io.BytesIO()
    Image.new("RGBA", size, (200, 30, 30, 255)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"created": 1700000000, "data": [{"b64_json": b64}]}


class FakeOpenRouter(BaseHTTPRequestHandler):
    seen = []
    answer = None
    images_answer = None  # separate answer for /api/v1/images
    status = 200
    images_status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenRouter.seen.append({"path": self.path, "body": body,
                                    "auth": self.headers.get("Authorization"),
                                    "title": self.headers.get("X-Title")})
        if self.headers.get("Authorization") != "Bearer sk-or-good":
            self._send(401, {"error": {"message": "No auth credentials"}})
            return
        if self.path == "/api/v1/images":
            if FakeOpenRouter.images_answer is not None:
                self._send(FakeOpenRouter.images_status,
                           FakeOpenRouter.images_answer)
            else:
                self._send(200, images_response())
            return
        self._send(FakeOpenRouter.status, FakeOpenRouter.answer or {
            "choices": [{"message": {
                "role": "assistant", "content": "",
                "images": [{"type": "image_url",
                            "image_url": {"url": picture()}}]}}]})

    def _send(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture()
def cloud_server(monkeypatch):
    FakeOpenRouter.seen = []
    FakeOpenRouter.answer = None
    FakeOpenRouter.images_answer = None
    FakeOpenRouter.status = 200
    FakeOpenRouter.images_status = 200
    api = ThreadingHTTPServer((HOST, 0), FakeOpenRouter)
    threading.Thread(target=api.serve_forever, daemon=True).start()
    config.save_model_settings({
        "provider": "openrouter", "openrouter_model": "test/image-model",
        "openrouter_api_key": "sk-or-good",
        "openrouter_base_url": f"http://{HOST}:{api.server_address[1]}/api/v1"})
    models.reset()
    monkeypatch.setattr(srv, "DEBUG_SAVE", False)
    port = free_port()
    loop = asyncio.new_event_loop()
    stop = loop.create_future()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        # no fakes registered: serve() registers the real pipelines
        loop.run_until_complete(srv.serve(HOST, port, stop,
                                          on_ready=ready.set))
    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert ready.wait(5)
    yield port
    loop.call_soon_threadsafe(stop.set_result, None)
    t.join(timeout=5)
    api.shutdown()
    models.reset()


def ask(port, request):
    async def go():
        async with websockets.connect(f"ws://{HOST}:{port}",
                                      max_size=64 * 2**20) as ws:
            await ws.send(json.dumps(request))
            msgs = []
            while True:
                msg = json.loads(await ws.recv())
                msgs.append(msg)
                if msg["type"] in ("result", "error"):
                    return msgs
    return asyncio.run(go())


def test_generate_through_openrouter(cloud_server):
    msgs = ask(cloud_server, {"id": "g1", "mode": "generate",
                              "prompt": "red block", "target_size": [32, 32],
                              "variants": 2, "frames": []})
    assert msgs[-1]["type"] == "result", msgs[-1]
    images = [image_from_raw(i) for i in msgs[-1]["images"]]
    assert len(images) == 2
    assert all(img.size == (32, 32) for img in images)  # fit to the canvas
    # the white background was stripped, the red subject kept
    corner = images[0].getpixel((0, 0))
    assert corner[3] == 0
    assert any(img.getpixel((16, 16))[3] == 255 for img in images)
    stages = [m.get("stage") for m in msgs if m["type"] == "progress"]
    assert "Waiting for OpenRouter · test/image-model" in stages
    assert "Decoding images" not in stages  # that's the local model's step
    call = FakeOpenRouter.seen[0]
    assert call["path"] == "/api/v1/chat/completions"
    assert call["auth"] == "Bearer sk-or-good"
    assert call["title"] == "Pixagon"
    assert call["body"]["model"] == "test/image-model"


def test_edit_and_inpaint_send_the_sprite(cloud_server):
    sprite = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for x in range(4, 12):
        for y in range(4, 12):
            sprite.putpixel((x, y), (20, 160, 40, 255))
    mask = Image.new("L", (16, 16), 0)
    for x in range(4, 8):
        for y in range(4, 8):
            mask.putpixel((x, y), 255)
    edit = ask(cloud_server, {"id": "e1", "mode": "edit", "prompt": "redder",
                              "target_size": [16, 16], "variants": 1,
                              "frames": [{"image": image_to_b64(sprite),
                                          "mask": None}]})
    assert edit[-1]["type"] == "result", edit[-1]
    parts = FakeOpenRouter.seen[-1]["body"]["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]
    inpaint = ask(cloud_server, {"id": "i1", "mode": "inpaint",
                                 "prompt": "a red patch",
                                 "target_size": [16, 16], "variants": 1,
                                 "frames": [{"image": image_to_b64(sprite),
                                             "mask": image_to_b64(mask)}]})
    assert inpaint[-1]["type"] == "result", inpaint[-1]
    parts = FakeOpenRouter.seen[-1]["body"]["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    out = image_from_raw(inpaint[-1]["images"][0])
    assert out.getpixel((15, 15))[3] == 0  # outside the selection: untouched


def test_a_wrong_key_reaches_the_panel_as_an_error(cloud_server):
    config.save_model_settings({"openrouter_api_key": "sk-or-bad"})
    models.reset()  # the server reads the key when it loads the backend
    srv._register_default_models()
    msgs = ask(cloud_server, {"id": "g2", "mode": "generate", "prompt": "x",
                              "target_size": [16, 16], "variants": 1,
                              "frames": []})
    assert msgs[-1]["type"] == "error"
    assert msgs[-1]["message"].startswith("OpenRouter refused the API key")
    assert "No auth credentials" in msgs[-1]["message"]


def test_unsupported_image_output_reaches_the_panel(cloud_server):
    FakeOpenRouter.status = 404
    FakeOpenRouter.answer = {"error": {"message": (
        "No endpoints found that support the requested output "
        "modalities: image, text")}}
    # the images endpoint also rejects this model
    FakeOpenRouter.images_status = 404
    FakeOpenRouter.images_answer = {"error": {"message": "Model not found"}}
    msgs = ask(cloud_server, {"id": "g3", "mode": "generate", "prompt": "x",
                              "target_size": [16, 16], "variants": 1,
                              "frames": []})
    assert msgs[-1]["type"] == "error"
    message = msgs[-1]["message"]
    assert "test/image-model" in message
    assert "image output" in message
    assert "Backend step" in message
    assert config.DEFAULT_OPENROUTER_MODEL in message
    assert "restart" in message
    assert len(FakeOpenRouter.seen) == 2  # tried both endpoints


def test_image_gen_model_falls_back_to_images_endpoint(cloud_server):
    """When chat/completions rejects the model, the server switches to the
    dedicated /api/v1/images endpoint automatically."""
    FakeOpenRouter.status = 404
    FakeOpenRouter.answer = {"error": {"message": (
        "test/image-model is an image generation model and cannot be used "
        "with the chat/completions endpoint. "
        "Use the /api/v1/images endpoint instead.")}}
    msgs = ask(cloud_server, {"id": "g4", "mode": "generate",
                              "prompt": "red block", "target_size": [32, 32],
                              "variants": 1, "frames": []})
    assert msgs[-1]["type"] == "result", msgs[-1]
    images = [image_from_raw(i) for i in msgs[-1]["images"]]
    assert len(images) == 1
    # first call hit chat/completions (rejected), second hit images
    paths = [c["path"] for c in FakeOpenRouter.seen]
    assert "/api/v1/chat/completions" in paths
    assert "/api/v1/images" in paths
    # the images call used the right body shape
    img_call = next(c for c in FakeOpenRouter.seen
                    if c["path"] == "/api/v1/images")
    assert img_call["body"]["model"] == "test/image-model"
    assert "prompt" in img_call["body"]
    assert "messages" not in img_call["body"]
