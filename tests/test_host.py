"""The host fingerprint and the thermal reader, with an injected `run`."""
import platform
import subprocess
import sys
from types import SimpleNamespace

from callo.host import host_info, therm_limit


def _runner(table):
    def run(argv, **kw):
        key = " ".join(argv)
        for prefix, (rc, out) in table.items():
            if key.startswith(prefix):
                return SimpleNamespace(returncode=rc, stdout=out)
        raise FileNotFoundError(argv[0])
    return run


def test_macos_host_info_reads_sysctl_and_fingerprints_chip_memory_and_gpu():
    run = _runner({"sysctl -n machdep.cpu.brand_string": (0, "Apple M4 Max\n"), "sysctl -n hw.memsize": (0, "68719476736\n")})
    h = host_info(run=run, system="Darwin")
    assert h["chip"] == "Apple M4 Max" and h["memory_bytes"] == 68719476736 and h["gpu"] is None
    assert h["os"].startswith(("macOS-", "Darwin-")) and h["fingerprint"].startswith("sha256:") and len(h["fingerprint"]) == 71
    assert host_info(run=run, system="Darwin")["fingerprint"] == h["fingerprint"]
    other = host_info(run=_runner({"sysctl -n machdep.cpu.brand_string": (0, "Apple M2\n"),
                                   "sysctl -n hw.memsize": (0, "68719476736\n")}), system="Darwin")
    assert other["fingerprint"] != h["fingerprint"]


def test_linux_host_info_reads_cpuinfo_and_nvidia_smi_and_tolerates_their_absence(tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor : 0\nmodel name\t: AMD EPYC 7763\nflags : x\n")
    run = _runner({"nvidia-smi": (0, "NVIDIA H100, 81559 MiB\n")})
    h = host_info(run=run, system="Linux", cpuinfo=str(cpuinfo))
    assert h["chip"] == "AMD EPYC 7763" and h["gpu"] == "NVIDIA H100, 81559 MiB" and h["memory_bytes"]
    bare = host_info(run=_runner({}), system="Linux", cpuinfo=str(tmp_path / "missing"))
    assert bare["chip"] is None and bare["gpu"] is None and bare["fingerprint"] is None   # RAM alone: unknown host


def test_a_failing_or_timing_out_command_never_raises():
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 2)

    h = host_info(run=boom, system="Darwin")
    assert h["chip"] is None and h["memory_bytes"] is None
    assert host_info(run=_runner({"sysctl": (1, "")}), system="Darwin")["chip"] is None
    assert host_info(run=_runner({}), system="Windows")["chip"] is None


def test_therm_limit_reads_cpu_speed_limit_only_on_macos_and_null_otherwise():
    run = _runner({"pmset -g therm": (0, "Note: No thermal warning level has been recorded\nCPU_Speed_Limit \t= 100\n")})
    assert therm_limit(run=run, system="Darwin") == 100
    assert therm_limit(run=_runner({"pmset -g therm": (0, "CPU_Speed_Limit = 57")}), system="Darwin") == 57
    assert therm_limit(run=_runner({"pmset -g therm": (0, "Note: nothing recorded")}), system="Darwin") is None
    assert therm_limit(run=_runner({}), system="Darwin") is None
    assert therm_limit(run=run, system="Linux") is None


def test_an_unknown_host_has_no_fingerprint_rather_than_a_shared_one():
    """Hashing "None|None|None" would make every unprobeable box "the same machine"."""
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 2)

    assert host_info(run=boom, system="Darwin")["fingerprint"] is None
    assert host_info(run=_runner({}), system="Windows")["fingerprint"] is None
    chip_only = host_info(run=_runner({"sysctl -n machdep.cpu.brand_string": (0, "Apple M2\n")}), system="Darwin")
    assert chip_only["memory_bytes"] is None and chip_only["fingerprint"].startswith("sha256:")


def test_undecodable_output_and_a_colonless_model_name_line_never_raise(tmp_path):
    """A non-UTF-8 byte from a real child process is replaced, not raised."""
    def real(argv, **kw):
        return subprocess.run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'Apple \\xff M9')"], **kw)

    h = host_info(run=real, system="Darwin")
    assert h["chip"] == "Apple \ufffd M9" and h["memory_bytes"] is None
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("model name\nmodel name : ARM Neoverse\n")
    assert host_info(run=_runner({}), system="Linux", cpuinfo=str(cpuinfo))["chip"] == "ARM Neoverse"


def test_every_command_is_an_argv_list_with_a_bounded_timeout_and_no_shell():
    """The hang protection is pinned, on every probe of every platform."""
    calls = []

    def record(argv, **kw):
        calls.append((argv, kw))
        return SimpleNamespace(returncode=0, stdout="1\n")

    for system in ("Darwin", "Linux", "Windows"):
        host_info(run=record, system=system)
        therm_limit(run=record, system=system)
    assert {argv[0] for argv, _ in calls} == {"sysctl", "nvidia-smi", "pmset"}
    for argv, kw in calls:
        assert isinstance(argv, list) and not kw.get("shell")
        assert 0 < kw["timeout"] <= 5 and kw["check"] is False
        assert kw["encoding"] == "utf-8" and kw["errors"] == "replace" and kw["capture_output"] is True


def test_os_is_macos_and_its_version_on_darwin_and_system_release_elsewhere(monkeypatch):
    """The OS as "macOS-26.0-arm64", not the Darwin kernel version."""
    monkeypatch.setattr(platform, "mac_ver", lambda: ("26.0", ("", "", ""), "arm64"))
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(platform, "release", lambda: "25.0.0")
    assert host_info(run=_runner({}), system="Darwin")["os"] == "macOS-26.0-arm64"
    monkeypatch.setattr(platform, "mac_ver", lambda: ("", ("", "", ""), ""))
    assert host_info(run=_runner({}), system="Darwin")["os"] == "Darwin-25.0.0-arm64"
    assert host_info(run=_runner({}), system="Linux", cpuinfo="/nonexistent")["os"] == "Linux-25.0.0-arm64"
