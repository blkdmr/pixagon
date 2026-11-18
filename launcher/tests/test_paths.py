# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

import pathlib

from launcher import paths
from server import config


def fake_run(answers):
    def run(cmd):
        return answers.get(cmd[0])
    return run


def test_probe_python_accepts_new_enough():
    run = fake_run({"py": r"C:\Python314\python.exe|3.14.0"})
    got = paths.probe_python(["py", "-3"], run=run)
    assert got == (pathlib.Path(r"C:\Python314\python.exe"), (3, 14, 0))


def test_probe_python_rejects_old():
    run = fake_run({"py": r"C:\Python39\python.exe|3.9.7"})
    assert paths.probe_python(["py", "-3"], run=run) is None


def test_probe_python_handles_no_interpreter():
    assert paths.probe_python(["nope"], run=fake_run({})) is None


def test_probe_python_handles_garbage():
    assert paths.probe_python(["py"], run=fake_run({"py": "hello"})) is None


def test_resolve_prefers_overrides(tmp_path):
    settings = {"root": str(tmp_path), "python": r"C:\custom\python.exe",
                "models_dir": str(tmp_path / "big")}
    got = paths.resolve(settings, run=fake_run({}))
    assert got.root == tmp_path
    assert got.python == pathlib.Path(r"C:\custom\python.exe")
    assert got.models_dir == tmp_path / "big"


def test_resolve_leaves_python_to_uv_unless_overridden():
    # a python3 on PATH must not become the venv's base (see paths.resolve)
    run = fake_run({"python3": "/usr/bin/python3|3.12.3",
                    "py": r"C:\P\python.exe|3.13.0"})
    got = paths.resolve({}, run=run)
    assert got.python is None and got.python_version is None
    assert got.root == paths.app_root()


def test_code_root_holds_the_server_package():
    assert (paths.app_root() / "server" / "main.py").is_file()


def test_resolve_puts_models_and_venv_under_the_data_folder(tmp_path,
                                                            monkeypatch):
    # an extension update replaces the code folder; data must live elsewhere
    data = tmp_path / "support"
    monkeypatch.setenv(config.DATA_ENV, str(data))
    got = paths.resolve({}, run=lambda cmd: None)
    assert got.data == data and got.root == paths.app_root()
    assert got.models_dir == data / "models"


def test_data_folder_per_os():
    env = {"HOME": "/home/ada", "APPDATA": r"C:\Users\ada\AppData\Roaming"}
    assert config.data_dir("linux", env) == \
        pathlib.Path("/home/ada/.config/pixagon")
    assert config.data_dir("darwin", env) == \
        pathlib.Path("/home/ada/Library/Application Support/Pixagon")
    assert config.data_dir("win32", env) == \
        pathlib.Path(r"C:\Users\ada\AppData\Roaming") / "Pixagon"


def test_data_folder_follows_xdg_and_the_override():
    env = {"HOME": "/home/ada", "XDG_CONFIG_HOME": "/cfg"}
    assert config.data_dir("linux", env) == pathlib.Path("/cfg/pixagon")
    env[config.DATA_ENV] = "/elsewhere"
    assert config.data_dir("linux", env) == pathlib.Path("/elsewhere")


def test_settings_file_sits_in_the_data_folder_by_default():
    env = {"HOME": "/home/ada"}
    assert config.config_path("linux", env).parent == config.data_dir("linux", env)
