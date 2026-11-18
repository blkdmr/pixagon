"""The parts of the server that differ by OS and GPU, without a GPU."""
import pathlib
import sys
import types

import pytest

from server import config, models
from server.instruct import (NO_GPU, KleinPipeline, chunk_size,
                             generator_device, page_file_hint, parse_meminfo,
                             parse_vm_stat, pick_device, resolve_mode)

GIB = 1024 ** 3


def fake_torch(cuda=False, mps=False):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)))


def test_pick_device_prefers_cuda_then_metal():
    assert pick_device(fake_torch(cuda=True, mps=True)) == "cuda"
    assert pick_device(fake_torch(mps=True)) == "mps"
    assert pick_device(fake_torch()) is None


def test_pick_device_copes_with_a_torch_without_mps():
    old = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False))
    assert pick_device(old) is None


def test_apple_silicon_always_loads_resident():
    # offload buys nothing on unified memory, whatever the saved VRAM mode
    for mode in ("auto", "bf16", "offload"):
        assert resolve_mode(mode, 0, device="mps") == "mps"
    assert chunk_size("mps", "t2i") == 2
    assert chunk_size("mps", "edit") == 1


def test_nvidia_modes_are_unchanged_by_the_device_argument():
    assert resolve_mode("auto", 16 * GIB, device="cuda") == "bf16"
    assert resolve_mode("auto", 6 * GIB, device="cuda") == "offload"


def test_seeds_draw_cpu_noise_on_metal():
    assert generator_device("cuda") == "cuda"
    assert generator_device("mps") == "cpu"


def test_load_without_any_gpu_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    diffusers = types.ModuleType("diffusers")
    diffusers.Flux2KleinPipeline = object
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    try:
        with pytest.raises(RuntimeError) as err:
            KleinPipeline().load()
    finally:
        models.set_load_progress(None)  # load() is normally run by models.get
    assert str(err.value) == NO_GPU
    assert "Apple Silicon" in NO_GPU and "NVIDIA" in NO_GPU


MEMINFO = """\
MemTotal:       32768000 kB
MemFree:         1000000 kB
MemAvailable:   20000000 kB
SwapTotal:       8000000 kB
SwapFree:        6000000 kB
"""


def test_parse_meminfo_counts_swap_as_committable():
    mem = parse_meminfo(MEMINFO)
    assert mem["total_ram"] == 32768000 * 1024
    assert mem["free_ram"] == 20000000 * 1024
    assert mem["free_commit"] == (20000000 + 6000000) * 1024
    assert mem["total_commit"] == (32768000 + 8000000) * 1024
    assert mem["commit_limited"] is True


def test_parse_meminfo_without_swap_or_available():
    mem = parse_meminfo("MemTotal: 1000 kB\nMemFree: 400 kB\n")
    assert mem["free_ram"] == 400 * 1024
    assert mem["free_commit"] == 400 * 1024
    assert parse_meminfo("garbage") is None


VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10000.
Pages active:                            200000.
Pages inactive:                           20000.
Pages speculative:                         5000.
Pages purgeable:                           1000.
"""


def test_parse_vm_stat_never_hard_limits():
    mem = parse_vm_stat(VM_STAT, 32 * GIB)
    assert mem["total_ram"] == 32 * GIB
    assert mem["free_ram"] == 16384 * (10000 + 20000 + 5000 + 1000)
    # macOS grows swap by itself: a low figure must only warn
    assert mem["commit_limited"] is False
    assert parse_vm_stat("", 32 * GIB) is None
    assert parse_vm_stat(VM_STAT, 0) is None


def test_page_file_hint_names_each_os_fix():
    assert "Windows page file" in page_file_hint("win32")
    assert "swap" in page_file_hint("linux")
    assert "macOS" in page_file_hint("darwin")


def test_config_lives_where_each_os_expects():
    win = config.config_path("win32", {"APPDATA": r"C:\A", "HOME": "/h"})
    assert win == pathlib.Path(r"C:\A") / "Pixagon" / "config.json"
    mac = config.config_path("darwin", {"HOME": "/Users/ada"})
    assert mac == pathlib.Path("/Users/ada/Library/Application Support/"
                               "Pixagon/config.json")
    linux = config.config_path("linux", {"HOME": "/home/ada"})
    assert linux == pathlib.Path("/home/ada/.config/pixagon/config.json")
    xdg = config.config_path("linux", {"HOME": "/home/ada",
                                       "XDG_CONFIG_HOME": "/cfg"})
    assert xdg == pathlib.Path("/cfg/pixagon/config.json")


def test_config_without_appdata_falls_back_to_home():
    got = config.config_path("win32", {"HOME": "/h"})
    assert got == pathlib.Path("/h") / "Pixagon" / "config.json"
