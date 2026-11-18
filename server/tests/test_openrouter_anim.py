"""Tests for OpenRouter animation parity with mocked HTTP."""
import base64
import io
import json
from PIL import Image

from server.openrouter import OpenRouterPipeline
from server.protocol import Frame, AnimationParams, Request


def _fake_image_response():
    """A chat-completions response containing one inline image."""
    buf = io.BytesIO()
    Image.new("RGBA", (64, 64), (100, 200, 50, 255)).save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return {
        "choices": [{
            "message": {
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": data_url}}
                ]
            }
        }]
    }


def _make_post(calls):
    """Return a fake post function that records calls and returns images."""
    def post(url, body, api_key, timeout=180):
        calls.append(body)
        return _fake_image_response()
    return post


def _make_request(frame_count=3, preset="walk_cycle"):
    return Request(
        id="or_anim", mode="animate", prompt="knight walking",
        target_size=(64, 64),
        animation=AnimationParams(
            frame_count=frame_count, strength=0.5, loop=True,
            fps=8, preset=preset, prompt_sequence=None),
    )


def test_openrouter_animate_sequential():
    calls = []
    pipe = OpenRouterPipeline("sk-test", "test/model",
                              "https://openrouter.ai/api/v1",
                              post=_make_post(calls), sleep=lambda _: None)
    pipe.load()
    req = _make_request(frame_count=3)
    frames, seeds = pipe.animate_sequential(pipe, req)
    assert len(frames) == 3
    assert len(seeds) == 3
    # txt2img for frame 0, then 2 edit calls
    # (the AnimationGenerator uses pipe.txt2img and pipe.edit_by_instruction)
    assert len(calls) >= 3


def test_openrouter_animate_progress_callback():
    calls = []
    pipe = OpenRouterPipeline("sk-test", "test/model",
                              "https://openrouter.ai/api/v1",
                              post=_make_post(calls), sleep=lambda _: None)
    pipe.load()
    req = _make_request(frame_count=4)
    values = []
    pipe.animate_sequential(pipe, req, on_progress=values.append)
    assert len(values) == 4
    assert values[-1] == 1.0
