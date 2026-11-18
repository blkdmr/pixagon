<!-- Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
     Licensed under the Apache License, Version 2.0
     Modifications Copyright 2026 Alessio Russo (Pixagon) -->

# Contributing

Thanks for taking a look. Pixagon is a Python WebSocket server plus an
Aseprite Lua extension.

## Setup

See the [README](README.md) install steps. You need an NVIDIA GPU with 12+ GB
of VRAM (Windows or Linux) or an Apple Silicon Mac to run the model, but most
of the code (protocol, postprocess, prompt assembly, plugin UI, launcher) is
testable **without** a GPU, on any of the three systems.

## Before opening a PR

Run the same checks CI runs (on macOS/Linux use `.venv/bin/python`):

```
# server + launcher
.venv\Scripts\python -m pytest server/tests/ launcher/tests/ --ignore=server/tests/smoke.py

# plugin (Windows: scoop install lua luacheck; see the README for macOS/Linux)
luacheck plugin/*.lua plugin/tests/*.lua
lua plugin/tests/test_prompt.lua
lua plugin/tests/test_panel.lua
```

Anything that differs by OS goes in `launcher/platforms.py` (or, for the
server, next to `memory_status()` in `server/instruct.py`), with a test that
asks about every OS by passing the platform in.

## Conventions

- Keep comments short — one line, and only when they carry a constraint the
  code can't show.
- Prompt assembly (`plugin/prompt.lua`) and postprocess are pure and
  unit-tested; add a test when you touch them.
- The plugin never modifies the user's existing pixels. Keep it that way.
