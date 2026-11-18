"""Tests for the animate mode protocol extension."""
import json
import pytest
from PIL import Image
from server.protocol import (
    ProtocolError, parse_request, progress_msg, anim_result_msg,
    image_to_b64, image_from_raw,
)


def _red_16():
    return Image.new("RGBA", (16, 16), (255, 0, 0, 255))


def _anim_payload(**overrides):
    base = {
        "id": "a1", "mode": "animate", "prompt": "knight walking right",
        "target_size": [64, 64],
        "animation": {
            "frame_count": 8, "strength": 0.55, "loop": True,
            "fps": 8, "preset": "walk_cycle",
        },
    }
    base.update(overrides)
    return json.dumps(base)


def test_parse_animate_request():
    req = parse_request(_anim_payload())
    assert req.mode == "animate"
    assert req.animation is not None
    assert req.animation.frame_count == 8
    assert req.animation.strength == 0.55
    assert req.animation.loop is True
    assert req.animation.fps == 8
    assert req.animation.preset == "walk_cycle"
    assert req.animation.prompt_sequence is None


def test_animate_forces_variants_to_1():
    req = parse_request(_anim_payload(variants=4))
    assert req.variants == 1


def test_animate_requires_animation_block():
    with pytest.raises(ProtocolError, match="animation"):
        parse_request(json.dumps({
            "id": "a2", "mode": "animate", "prompt": "walk",
            "target_size": [64, 64],
        }))


def test_animate_clamps_frame_count():
    # below min
    req = parse_request(_anim_payload(
        animation={"frame_count": 0, "strength": 0.5}))
    assert req.animation.frame_count == 2
    # above max
    req = parse_request(_anim_payload(
        animation={"frame_count": 100, "strength": 0.5}))
    assert req.animation.frame_count == 32


def test_animate_clamps_strength():
    req = parse_request(_anim_payload(
        animation={"frame_count": 4, "strength": 0.0}))
    assert req.animation.strength == 0.10
    req = parse_request(_anim_payload(
        animation={"frame_count": 4, "strength": 1.5}))
    assert req.animation.strength == 0.95


def test_animate_clamps_fps():
    req = parse_request(_anim_payload(
        animation={"frame_count": 4, "strength": 0.5, "fps": 0}))
    assert req.animation.fps == 1
    req = parse_request(_anim_payload(
        animation={"frame_count": 4, "strength": 0.5, "fps": 999}))
    assert req.animation.fps == 60


def test_animate_rejects_unknown_preset():
    with pytest.raises(ProtocolError, match="preset"):
        parse_request(_anim_payload(
            animation={"frame_count": 4, "strength": 0.5,
                        "preset": "moonwalk"}))


def test_animate_accepts_null_preset():
    req = parse_request(_anim_payload(
        animation={"frame_count": 4, "strength": 0.5, "preset": None}))
    assert req.animation.preset is None


def test_animate_accepts_prompt_sequence():
    seq = ["pose A", "pose B", "pose C", "pose D"]
    req = parse_request(_anim_payload(
        animation={"frame_count": 4, "strength": 0.5,
                    "prompt_sequence": seq}))
    assert req.animation.prompt_sequence == seq


def test_animate_accepts_starting_frame():
    b64 = image_to_b64(_red_16())
    req = parse_request(_anim_payload(
        frames=[{"image": b64}]))
    assert req.frames[0].image.size == (16, 16)


def test_animation_ignored_on_non_animate_modes():
    req = parse_request(json.dumps({
        "id": "g1", "mode": "generate", "prompt": "sword",
        "target_size": [64, 64],
        "animation": {"frame_count": 8, "strength": 0.5},
    }))
    assert req.animation is None


def test_progress_msg_frame_fields():
    msg = json.loads(progress_msg("r", 0.5, frame_index=2, frame_total=8))
    assert msg["frame_index"] == 2
    assert msg["frame_total"] == 8
    # old fields still there
    assert msg["value"] == 0.5
    assert msg["type"] == "progress"


def test_progress_msg_frame_fields_absent_by_default():
    msg = json.loads(progress_msg("r", 0.5))
    assert "frame_index" not in msg
    assert "frame_total" not in msg


def test_anim_result_msg():
    imgs = [_red_16(), _red_16()]
    msg = json.loads(anim_result_msg("a1", imgs, [125, 125], [1, 2], True))
    assert msg["type"] == "anim_result"
    assert len(msg["frame_images"]) == 2
    assert msg["frame_durations"] == [125, 125]
    assert msg["seeds"] == [1, 2]
    assert msg["loop"] is True
    # images decode back
    decoded = image_from_raw(msg["frame_images"][0])
    assert decoded.size == (16, 16)
    assert decoded.getpixel((0, 0)) == (255, 0, 0, 255)
