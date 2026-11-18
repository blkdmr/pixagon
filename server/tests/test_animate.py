"""Tests for AnimationGenerator with a mocked pipeline."""
import pytest
from PIL import Image
from dataclasses import dataclass, field
from server.animate import AnimationGenerator, LOOP_STRENGTH_REDUCTION
from server.protocol import Frame, AnimationParams, Request


class MockPipeline:
    """Records calls and returns solid-color frames."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def variant_seeds(seed, count):
        if seed is None:
            return list(range(count))
        return [(seed + i) % 2**32 for i in range(count)]

    def txt2img(self, prompt, target_size, variants=1, on_progress=None,
                seeds=None):
        self.calls.append(("txt2img", prompt, target_size, seeds))
        return [Image.new("RGBA", target_size, (255, 0, 0, 255))
                for _ in range(variants)]

    def edit_by_instruction(self, instruction, image, variants=1,
                            on_progress=None, seeds=None):
        self.calls.append(("edit", instruction, image.size, seeds))
        return [Image.new("RGBA", image.size, (0, 255, 0, 255))
                for _ in range(variants)]


def _make_request(frame_count=4, strength=0.5, loop=False, preset=None,
                  starting_frame=None):
    frames = []
    if starting_frame is not None:
        frames = [Frame(image=starting_frame)]
    return Request(
        id="test", mode="animate", prompt="knight walking",
        target_size=(64, 64),
        animation=AnimationParams(
            frame_count=frame_count, strength=strength, loop=loop,
            fps=8, preset=preset, prompt_sequence=None),
        frames=frames,
    )


def test_generates_correct_number_of_frames():
    pipe = MockPipeline()
    gen = AnimationGenerator()
    req = _make_request(frame_count=4)
    frames, seeds = gen.generate_sequential(pipe, req)
    assert len(frames) == 4
    assert len(seeds) == 4


def test_first_frame_via_txt2img_when_no_starting_frame():
    pipe = MockPipeline()
    gen = AnimationGenerator()
    req = _make_request(frame_count=3)
    frames, _ = gen.generate_sequential(pipe, req)
    assert pipe.calls[0][0] == "txt2img"
    # remaining frames use edit
    assert all(c[0] == "edit" for c in pipe.calls[1:])


def test_first_frame_from_provided_image():
    pipe = MockPipeline()
    gen = AnimationGenerator()
    starting = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
    req = _make_request(frame_count=3, starting_frame=starting)
    frames, _ = gen.generate_sequential(pipe, req)
    # frame 0 is the starting image, no txt2img call
    assert frames[0] is starting
    assert all(c[0] == "edit" for c in pipe.calls)
    assert len(pipe.calls) == 2  # only frames 1 and 2


def test_progress_callback_called():
    pipe = MockPipeline()
    gen = AnimationGenerator()
    req = _make_request(frame_count=4)
    values = []
    frames, _ = gen.generate_sequential(pipe, req, on_progress=values.append)
    assert len(values) == 4
    assert values[-1] == pytest.approx(1.0)
    # monotonically increasing
    for i in range(1, len(values)):
        assert values[i] > values[i - 1]


def test_preset_phases_in_prompts():
    pipe = MockPipeline()
    gen = AnimationGenerator()
    req = _make_request(frame_count=4, preset="walk_cycle")
    gen.generate_sequential(pipe, req)
    # edit calls should contain preset phase descriptions
    from server.anim_presets import PRESETS
    phases = PRESETS["walk_cycle"]["phases"]
    for i, call in enumerate(pipe.calls):
        if call[0] == "edit":
            assert any(p in call[1] for p in phases)


def test_loop_closure_prompt():
    pipe = MockPipeline()
    gen = AnimationGenerator()
    req = _make_request(frame_count=4, loop=True, preset="walk_cycle")
    gen.generate_sequential(pipe, req)
    # last edit call should have loop closure
    last_edit = [c for c in pipe.calls if c[0] == "edit"][-1]
    assert "returning to starting pose" in last_edit[1]


def test_seeds_are_per_frame():
    pipe = MockPipeline()
    gen = AnimationGenerator()
    req = _make_request(frame_count=3)
    _, seeds = gen.generate_sequential(pipe, req)
    assert len(set(seeds)) == 3  # all distinct
