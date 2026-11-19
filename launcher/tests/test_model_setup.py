"""Setup with a different local model, or OpenRouter instead of the GPU."""
import pathlib
import sys

from launcher import paths, platforms, setup_checks, setup_steps
from server import config


def make_paths(tmp_path, venv=True):
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    if venv:
        exe = root / ".venv" / platforms.venv_python_rel()
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("", encoding="utf-8")
    return paths.Paths(root=root, python=pathlib.Path(sys.executable),
                       python_version=(3, 12, 0),
                       models_dir=root / "models")


def by_id(items):
    return {item["id"]: item for item in items}


def cloud(key=None, **extra):
    values = {"provider": "openrouter", **extra}
    if key:
        values["openrouter_api_key"] = key
    config.save_model_settings(values)


def test_local_is_the_default_and_needs_the_gpu_stack(tmp_path):
    items = by_id(setup_checks.check_all(make_paths(tmp_path),
                                         run=lambda cmd: None))
    assert items["torch"]["required"] and items["model"]["required"]
    assert set(items["deps"]["needs"]) == {"venv", "torch"}
    assert "apikey" not in items


def test_openrouter_needs_no_torch_and_no_download(tmp_path):
    cloud(key="sk-or-0123456789")
    items = by_id(setup_checks.check_all(make_paths(tmp_path),
                                         run=lambda cmd: None))
    assert not items["torch"]["required"]
    assert not items["model"]["required"]
    assert items["model"]["detail"] == "only for the local model"
    assert items["deps"]["needs"] == ["venv"]
    assert items["apikey"]["state"] == setup_checks.OK
    assert items["apikey"]["detail"] == "…6789"  # never the whole key


def test_openrouter_without_a_key_is_blocked_until_pasted(tmp_path):
    cloud()
    items = by_id(setup_checks.check_all(make_paths(tmp_path),
                                         run=lambda cmd: None))
    assert items["apikey"]["state"] == setup_checks.BLOCKED
    assert items["apikey"]["required"]


def test_openrouter_key_from_the_environment(tmp_path, monkeypatch):
    cloud()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-env")
    items = by_id(setup_checks.check_all(make_paths(tmp_path),
                                         run=lambda cmd: None))
    assert items["apikey"]["state"] == setup_checks.OK
    assert "OPENROUTER_API_KEY" in items["apikey"]["detail"]


def test_optional_rows_come_after_required_ones(tmp_path):
    # the page starts the "Optional" section at the first optional row
    cloud(key="sk-or-0123456789")
    items = setup_checks.check_all(make_paths(tmp_path), run=lambda cmd: None)
    flags = [it["required"] for it in items]
    assert flags == sorted(flags, reverse=True)


def test_cloud_deps_probe_checks_only_the_light_packages():
    probe = setup_checks.deps_probe(local=False)
    assert "websockets" in probe and "diffusers" not in probe
    assert "diffusers" in setup_checks.deps_probe(local=True)


def test_cloud_deps_step_installs_the_base_requirements(tmp_path):
    cloud(key="k")
    cmd = setup_steps.command("deps", make_paths(tmp_path))
    assert cmd[-1].endswith("requirements-base.txt")
    config.save_model_settings({"provider": "local"})
    cmd = setup_steps.command("deps", make_paths(tmp_path))
    assert cmd[-1].endswith("requirements.txt")
    assert not cmd[-1].endswith("requirements-base.txt")


def test_base_requirements_exist_and_are_part_of_the_full_list():
    server = pathlib.Path(__file__).resolve().parents[2] / "server"
    base = (server / "requirements-base.txt").read_text()
    full = (server / "requirements.txt").read_text()
    assert "websockets" in base and "diffusers" not in base
    assert "-r requirements-base.txt" in full and "diffusers" in full
    for name in platforms.cloud_packages():
        assert name.lower() in base.lower()


def test_model_folder_matches_the_server():
    from server.instruct import model_folder
    assert setup_checks.MODEL_FOLDER == model_folder(config.LOCAL_MODEL)


def test_what_start_needs_per_provider():
    local = setup_checks.server_needs({"provider": "local"})
    assert local == ("venv", "deps", "torch", "model")
    remote = setup_checks.server_needs({"provider": "openrouter"})
    assert "apikey" in remote and "torch" not in remote
    assert "model" not in remote


def test_backend_label_for_the_main_screen():
    assert setup_checks.backend_label() == "FLUX.2-klein-4B · local"
    cloud(openrouter_model="google/gemini-2.5-flash-image")
    assert setup_checks.backend_label() == \
        "OpenRouter · google/gemini-2.5-flash-image"
