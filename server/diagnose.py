# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

"""Loader self-check for when the server crashes on start.

Run: .venv\\Scripts\\python -m server.diagnose   (Windows)
     .venv/bin/python -m server.diagnose       (macOS, Linux)
"""
import importlib.metadata as md
import sys

from server.config import LOCAL_MODEL, load_model_settings, load_models_dir
from server.instruct import (memory_status, page_file_hint, pick_device,
                             safetensors_expected_size, snapshot_dir)

PACKAGES = ("torch", "safetensors", "diffusers", "transformers",
            "accelerate", "bitsandbytes")  # bitsandbytes: not on macOS


def gb(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


def show_versions() -> None:
    print("== versions ==")
    print(f"  python:       {sys.version.split()[0]}")
    for name in PACKAGES:
        try:
            print(f"  {name + ':':13} {md.version(name)}")
        except md.PackageNotFoundError:
            print(f"  {name + ':':13} (not installed)")


def show_memory() -> None:
    print("== memory ==")
    mem = memory_status()
    if mem is None:
        print("  (this system doesn't report memory; skipped)")
        return
    print(f"  RAM total:   {gb(mem['total_ram'])}")
    print(f"  RAM free:    {gb(mem['free_ram'])}")
    print(f"  commit max:  {gb(mem['total_commit'])}  (RAM + page file/swap)")
    print(f"  commit free: {gb(mem['free_commit'])}")


def show_gpu() -> None:
    print("== gpu ==")
    try:
        import torch
    except Exception as e:
        print(f"  torch does not import: {e}")
        return
    device = pick_device(torch)
    if device == "cuda":
        print(f"  CUDA: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("  Metal (Apple Silicon)")
    else:
        print("  NONE: no CUDA and no Metal device -> reinstall PyTorch in the launcher's Install step")


def show_files() -> object | None:
    """Print each weight file against its header size; return the transformer if complete."""
    models_dir = load_models_dir()
    settings = load_model_settings()
    if settings["provider"] != "local":
        print("== model files ==")
        print(f"  skipped: the launcher uses {settings['provider']} "
              f"({settings['openrouter_model']}), no local model")
        return None
    model_id = LOCAL_MODEL
    print(f"== model files ({model_id}, models_dir = {models_dir!r}) ==")
    snap = snapshot_dir(models_dir, model_id)
    if snap is None:
        print("  not downloaded; use the launcher's Model step")
        return None
    transformer = None
    ok = True
    for path in sorted(snap.rglob("*.safetensors")):
        actual = path.stat().st_size
        try:
            expected = safetensors_expected_size(path)
        except (OSError, ValueError):
            print(f"  UNREADABLE  {path.name}")
            ok = False
            continue
        rel = path.relative_to(snap).as_posix()
        tag = "ok" if actual >= expected else f"TRUNCATED (want {expected})"
        print(f"  {tag:28} {actual:>13} bytes  {rel}")
        ok = ok and actual >= expected
        if path.parent.name == "transformer":
            transformer = path
    return transformer if ok else None


def load_transformer_alone(path) -> None:
    print("== isolated load ==")
    if path is None:
        print("  skipped (files missing or truncated -> re-download the model)")
        return
    print("  loading the transformer alone...")
    from safetensors.torch import load_file
    load_file(str(path))
    print("  transformer loaded with no crash -> files are fine")
    print("  a crash in the full run is then memory: close apps or "
          f"{page_file_hint()}")


def main() -> None:
    show_versions()
    show_memory()
    show_gpu()
    transformer = show_files()
    load_transformer_alone(transformer)


if __name__ == "__main__":
    main()
