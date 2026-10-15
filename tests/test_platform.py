"""Windows: the `callo` console script and `python -m callo` refuse in one line (exit 2) before importing anything
POSIX-only; on Linux and macOS both are a byte-for-byte pass-through to `callo.cli.main`.

Windows is simulated in a fresh interpreter: `sys.platform` set to "win32" and `os.killpg` deleted (Windows' `os` has
none, so importing `callo.menu`, which binds it as a default argument, would die with AttributeError)."""
from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import entry_points

import pytest

import callo.__main__ as entry
from callo import cli

REFUSAL = "Callo runs on Linux and macOS. Windows support is coming; for now, run it under WSL2.\n"
TIMEOUT = 120   # a cold interpreter importing numpy and gguf on a contended runner

# Runs in a fresh interpreter as Windows would: the entry point named by argv[1], then the callo modules it imported.
# The stdlib is imported before the patch (importlib.metadata pulls in shutil, which would look for _winapi).
WINDOWS = """
import json, os, runpy, sys
from importlib.metadata import entry_points
(ep,) = entry_points(group="console_scripts", name="callo")
sys.platform = "win32"
del os.killpg
how = sys.argv.pop(1)
try:
    if how == "console-script":   # what the generated `callo` script does
        sys.exit(ep.load()())
    runpy.run_module("callo", run_name="__main__", alter_sys=True)   # python -m callo
except SystemExit as e:
    code = e.code
print(json.dumps(sorted(m for m in sys.modules if m == "callo" or m.startswith("callo."))))
sys.exit(code)
"""


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=TIMEOUT)


def test_the_console_script_is_the_guarded_entry():
    (ep,) = entry_points(group="console_scripts", name="callo")
    assert ep.value == "callo.__main__:main"


@pytest.mark.parametrize("how", ["console-script", "python -m callo"])
@pytest.mark.parametrize("argv", [[], ["serve", "--model", "m.gguf"], ["version"], ["--help"]])
def test_windows_refuses_in_one_line_before_importing_anything_posix_only(how, argv):
    p = run(["-c", WINDOWS, how, *argv])
    assert p.returncode == 2
    assert p.stderr == REFUSAL
    assert set(json.loads(p.stdout)) <= {"callo", "callo.__main__"}   # no callo.cli, callo.menu, callo.upstream


def test_windows_refusal_in_process(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    assert entry.main(["version"]) == 2
    assert capsys.readouterr() == ("", REFUSAL)
    assert entry.WINDOWS_REFUSAL + "\n" == REFUSAL


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("argv", [["version"], ["classify", "--options", ""]])
def test_linux_and_macos_pass_through_to_cli(monkeypatch, capsys, platform, argv):
    monkeypatch.setattr(sys, "platform", platform)
    want = cli.main(argv)
    want_out = capsys.readouterr()
    assert entry.main(argv) == want
    assert capsys.readouterr() == want_out


@pytest.mark.skipif(sys.platform == "win32", reason="the pass-through is POSIX's")
@pytest.mark.parametrize("argv", [["version"], ["--help"], ["serve"], ["no-such-command"]])
def test_python_m_callo_matches_python_m_callo_cli_byte_for_byte(argv):
    got, want = run(["-m", "callo", *argv]), run(["-m", "callo.cli", *argv])
    assert (got.returncode, got.stdout, got.stderr) == (want.returncode, want.stdout, want.stderr)
    assert got.stdout or got.stderr
