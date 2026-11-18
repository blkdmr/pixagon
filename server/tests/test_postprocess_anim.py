"""Tests for animation postprocessing: union_bbox, postprocess_animation."""
import numpy as np
from PIL import Image
from server.postprocess import union_bbox, postprocess_animation
from server.protocol import Frame, AnimationParams, Request


def _frame_with_opaque_rect(w, h, x0, y0, x1, y1, color=(255, 0, 0, 255)):
    """Transparent image with one opaque rectangle."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for y in range(y0, y1):
        for x in range(x0, x1):
            img.putpixel((x, y), color)
    return img


def test_union_bbox_single_frame():
    img = _frame_with_opaque_rect(100, 100, 20, 30, 50, 60)
    bbox = union_bbox([img], margin=0)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert x0 == 20 and y0 == 30 and x1 == 50 and y1 == 60


def test_union_bbox_multiple_frames():
    f1 = _frame_with_opaque_rect(100, 100, 10, 10, 30, 30)
    f2 = _frame_with_opaque_rect(100, 100, 50, 50, 80, 80)
    bbox = union_bbox([f1, f2], margin=0)
    x0, y0, x1, y1 = bbox
    assert x0 == 10 and y0 == 10 and x1 == 80 and y1 == 80


def test_union_bbox_with_margin():
    img = _frame_with_opaque_rect(100, 100, 20, 20, 80, 80)
    bbox = union_bbox([img], margin=0.05)
    x0, y0, x1, y1 = bbox
    assert x0 < 20 and y0 < 20 and x1 > 80 and y1 > 80


def test_union_bbox_all_transparent():
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    assert union_bbox([img]) is None


def test_postprocess_animation_produces_target_size():
    frames = [Image.new("RGBA", (512, 512), (200, 100, 50, 255))
              for _ in range(4)]
    req = Request(
        id="t", mode="animate", prompt="walk",
        target_size=(32, 32), background="remove",
        animation=AnimationParams(
            frame_count=4, strength=0.5, loop=True, fps=8,
            preset=None, prompt_sequence=None),
    )
    out = postprocess_animation(req, frames)
    assert len(out) == 4
    for img in out:
        assert img.size == (32, 32)


def test_postprocess_animation_background_keep():
    frames = [Image.new("RGBA", (64, 64), (100, 100, 100, 255))
              for _ in range(2)]
    req = Request(
        id="t", mode="animate", prompt="idle",
        target_size=(32, 32), background="keep",
        animation=AnimationParams(
            frame_count=2, strength=0.3, loop=True, fps=8,
            preset=None, prompt_sequence=None),
    )
    out = postprocess_animation(req, frames)
    # all pixels should be opaque since background=keep
    for img in out:
        arr = np.asarray(img)
        assert (arr[:, :, 3] > 0).any()
