import pathlib

from launcher import platforms


def test_venv_layout_per_os():
    assert platforms.venv_python_rel("win32") == \
        pathlib.PurePath("Scripts", "python.exe")
    assert platforms.venv_python_rel("linux") == pathlib.PurePath("bin", "python")
    assert platforms.venv_python_rel("darwin") == pathlib.PurePath("bin", "python")


def test_find_venv_python_prefers_this_os_then_any(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").write_text("")
    assert platforms.find_venv_python(tmp_path, "win32") == \
        tmp_path / "bin" / "python"
    assert platforms.find_venv_python(tmp_path / "nope", "linux") is None


def test_gpu_backend_and_labels():
    assert platforms.gpu_backend("win32") == "cuda"
    assert platforms.gpu_backend("linux") == "cuda"
    assert platforms.gpu_backend("darwin") == "mps"
    assert "CUDA" in platforms.torch_label("linux")
    assert "Metal" in platforms.torch_label("darwin")


def test_server_packages_drop_bitsandbytes_only_on_macos():
    assert "bitsandbytes" in platforms.server_packages("linux")
    assert "bitsandbytes" not in platforms.server_packages("darwin")


def test_uv_candidates_put_setups_copy_first(tmp_path):
    env = {"HOME": "/home/ada"}
    got = platforms.uv_candidates(tmp_path, "linux", env,
                                  which=lambda name: "/usr/bin/uv")
    assert got[0] == tmp_path / ".tools" / "uv"
    assert pathlib.Path("/usr/bin/uv") in got
    assert pathlib.Path("/home/ada/.local/bin/uv") in got
    win = platforms.uv_candidates(tmp_path, "win32", env, which=lambda n: None)
    assert win[0] == tmp_path / ".tools" / "uv.exe"
    mac = platforms.uv_candidates(tmp_path, "darwin", env, which=lambda n: None)
    assert pathlib.Path("/opt/homebrew/bin/uv") in mac


def test_find_uv_returns_the_first_that_exists(tmp_path):
    env = {"HOME": str(tmp_path / "home")}
    none = lambda name: None  # noqa: E731
    assert platforms.find_uv(tmp_path, "linux", env, which=none) is None
    user = tmp_path / "home" / ".local" / "bin" / "uv"
    user.parent.mkdir(parents=True)
    user.write_text("")
    assert platforms.find_uv(tmp_path, "linux", env, which=none) == user
    mine = tmp_path / ".tools" / "uv"
    mine.parent.mkdir()
    mine.write_text("")
    assert platforms.find_uv(tmp_path, "linux", env, which=none) == mine


def test_uv_installer_per_os(tmp_path):
    cmd, env = platforms.uv_install_command(tmp_path, "win32")
    assert cmd[0] == "powershell" and "install.ps1" in cmd[-1]
    cmd, env = platforms.uv_install_command(tmp_path, "linux")
    assert cmd[:2] == ["/bin/sh", "-c"] and "install.sh" in cmd[-1]
    assert "curl" in cmd[-1] and "wget" in cmd[-1]
    assert env == {"UV_INSTALL_DIR": str(tmp_path / ".tools"),
                   "UV_NO_MODIFY_PATH": "1"}


def test_gpu_memory_reads_nvidia_smi_in_gigabytes():
    calls = []

    def run(cmd):
        calls.append(cmd)
        return "16376\n8192"  # two cards: the first one counts

    assert platforms.gpu_memory_gb(run, "linux") == 16.0
    assert calls[0][0] == "nvidia-smi"


def test_gpu_memory_is_unknown_without_a_card_or_on_a_mac():
    assert platforms.gpu_memory_gb(lambda cmd: None, "win32") is None
    assert platforms.gpu_memory_gb(lambda cmd: "N/A", "linux") is None
    assert platforms.gpu_memory_gb(lambda cmd: "16376", "darwin") is None
