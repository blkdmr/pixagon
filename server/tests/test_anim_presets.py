"""Tests for animation presets: phase cycling, prompt expansion."""
from server.anim_presets import (
    PRESETS, VALID_PRESETS, phase_for_frame, build_frame_prompt,
)


def test_all_presets_have_required_keys():
    for name, preset in PRESETS.items():
        assert "phases" in preset and len(preset["phases"]) > 0
        assert "default_frames" in preset
        assert "loop" in preset
        assert "strength" in preset


def test_valid_presets_matches_keys():
    assert VALID_PRESETS == set(PRESETS)


def test_phase_cycles_modulo():
    phases = PRESETS["walk_cycle"]["phases"]
    # frame 0 gets phase 0
    assert phase_for_frame("walk_cycle", 0, 8) == phases[0]
    # frame 4 wraps back to phase 0 (4 phases, frame 4 % 4 = 0)
    assert phase_for_frame("walk_cycle", 4, 8) == phases[0]
    # frame 5 wraps to phase 1
    assert phase_for_frame("walk_cycle", 5, 8) == phases[1]


def test_build_frame_prompt_no_preset():
    prompt = build_frame_prompt("knight", None, 0, 4)
    assert prompt == "knight"
    prompt = build_frame_prompt("knight", None, 3, 4)
    assert prompt == "knight"


def test_build_frame_prompt_with_preset():
    prompt = build_frame_prompt("knight", "walk_cycle", 0, 8)
    assert "knight" in prompt
    assert PRESETS["walk_cycle"]["phases"][0] in prompt


def test_build_frame_prompt_loop_closure():
    # last frame of a looped animation
    prompt = build_frame_prompt("knight", "walk_cycle", 7, 8, loop=True)
    assert "returning to starting pose" in prompt
    # non-last frame: no closure
    prompt = build_frame_prompt("knight", "walk_cycle", 3, 8, loop=True)
    assert "returning to starting pose" not in prompt
    # loop=False: no closure even on last frame
    prompt = build_frame_prompt("knight", "attack", 5, 6, loop=False)
    assert "returning to starting pose" not in prompt


def test_build_frame_prompt_no_preset_with_loop():
    prompt = build_frame_prompt("knight", None, 3, 4, loop=True)
    assert "returning to starting pose" in prompt
