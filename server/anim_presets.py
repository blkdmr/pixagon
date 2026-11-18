"""Animation preset definitions: phase descriptions, defaults, and cycling."""

PRESETS = {
    "walk_cycle": {
        "phases": [
            "left leg forward, right arm forward, mid-stride",
            "both feet on ground, passing position",
            "right leg forward, left arm forward, mid-stride",
            "both feet on ground, passing position",
        ],
        "default_frames": 8,
        "loop": True,
        "strength": 0.50,
    },
    "idle": {
        "phases": [
            "standing still, slight exhale",
            "standing still, subtle weight shift",
            "standing still, slight inhale",
            "standing still, subtle weight shift back",
        ],
        "default_frames": 4,
        "loop": True,
        "strength": 0.35,
    },
    "attack": {
        "phases": [
            "winding up for attack, pulling back",
            "mid-swing, weapon extended forward",
            "follow through, full extension",
            "recovering to idle stance",
        ],
        "default_frames": 6,
        "loop": False,
        "strength": 0.60,
    },
}

VALID_PRESETS = set(PRESETS)


def phase_for_frame(preset_name: str, frame_index: int,
                    frame_count: int) -> str:
    """Phase description for a frame, cycling modulo the preset's phases."""
    phases = PRESETS[preset_name]["phases"]
    return phases[frame_index % len(phases)]


def build_frame_prompt(base_prompt: str, preset: str | None,
                       frame_index: int, frame_count: int,
                       loop: bool = False) -> str:
    """Per-frame prompt: base + preset phase (if any) + loop closure hint."""
    if preset is None:
        text = base_prompt
    else:
        phase = phase_for_frame(preset, frame_index, frame_count)
        text = f"{base_prompt}, {phase}"
    if loop and frame_index == frame_count - 1:
        text += ", returning to starting pose"
    return text
