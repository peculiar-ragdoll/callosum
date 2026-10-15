"""tests/conftest.py's `_no_real_llama_server` guard: a real llama-server (a PATH lookup, or an existing binary outside
pytest's basetemp) is refused AND recorded, and a recorded attempt fails the test at teardown even when the code under
test swallowed the refusal. Nothing here launches anything: every start below is refused before Popen, or runs a path
that does not exist."""
import os
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from conftest import NO_NETWORK, REAL_HOME, REFUSED, refused_binary

from callo.bias import BuiltinSetError, builtin_set_bytes, builtin_set_url
from callo.upstream import LlamaServerProcess


def test_refused_binary_classifies_path_lookups_foreign_paths_fakes_and_missing_paths(tmp_path, tmp_path_factory):
    base = str(tmp_path_factory.getbasetemp())
    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\n")
    assert refused_binary("llama-server", base) and refused_binary("llama-cli", base)      # resolved through PATH
    assert refused_binary("/bin/sh", base)                                                 # exists, outside basetemp
    assert not refused_binary(str(fake), base)                                             # a fake under tmp_path
    assert not refused_binary(str(tmp_path / "no-such-llama-server"), base)                # "not found" stays callo's
    assert not refused_binary("/nonexistent/llama-server", base)


def test_a_bare_llama_server_is_refused_and_recorded(tmp_path, _no_real_llama_server):
    attempts = _no_real_llama_server
    for prog in ("llama-server", "/bin/sh"):
        p = LlamaServerProcess([prog, "-m", "/x.gguf", "--port", "0"], cache_dir=tmp_path / "c", startup_timeout=1)
        with pytest.raises(FileNotFoundError, match=f"{REFUSED}: {prog}"):
            p.start()
        assert p.proc is None                                                  # refused before Popen: nothing ran
    assert attempts == ["llama-server", "/bin/sh"]                             # what would fail this test's teardown
    attempts.clear()                                                           # (proved separately below)
    missing = LlamaServerProcess([str(tmp_path / "no-such-llama-server")], cache_dir=tmp_path / "c", startup_timeout=1)
    with pytest.raises(FileNotFoundError) as e:                                # passed through: Popen's own error
        missing.start()
    assert REFUSED not in str(e.value) and attempts == []


def test_a_swallowed_refusal_still_fails_the_test_at_teardown(tmp_path):
    """An inner pytest session (a Python subprocess; nothing else is started) with this conftest: its test swallows
    the refusal, as serve's thread and the menu driver do, and passes its body, yet errors at teardown."""
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "conftest.py").write_text((Path(__file__).parent / "conftest.py").read_text())
    (inner / "test_swallowed.py").write_text(textwrap.dedent("""
        import contextlib
        from callo.upstream import LlamaServerProcess

        def test_swallows_the_refusal(tmp_path):
            p = LlamaServerProcess(["llama-server", "-m", "/x.gguf"], cache_dir=tmp_path / "c", startup_timeout=1)
            with contextlib.suppress(FileNotFoundError):
                p.start()
            assert p.proc is None
    """))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(inner),
                        "--basetemp", str(tmp_path / "inner-tmp"), str(inner)],
                       capture_output=True, text=True, timeout=120, check=False, cwd=inner)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "1 passed, 1 error" in r.stdout and "ERROR at teardown of test_swallows_the_refusal" in r.stdout
    assert f"{REFUSED}: llama-server (tests/conftest.py refused the launch)" in r.stdout


# a volume this machine really has, listed at import (before any test hides /Volumes/), so hiding it proves something
_VOLUMES = sorted(os.listdir("/Volumes")) if os.path.isdir("/Volumes") else []
VOLUME = f"/Volumes/{_VOLUMES[0]}" if _VOLUMES else "/Volumes/Data"


@pytest.mark.skipif(os.environ.get("CALLO_TEST_HIDE_VOLUMES") != "1", reason="CALLO_TEST_HIDE_VOLUMES is not set")
def test_hidden_volumes_read_as_missing_everywhere(tmp_path):
    """With CALLO_TEST_HIDE_VOLUMES=1 the dev box looks like a CI runner: nothing under /Volumes/ exists or opens."""
    hidden = Path(VOLUME, "calibration.jsonl")
    assert not hidden.exists() and not hidden.is_file() and not os.path.exists(VOLUME)
    with pytest.raises(FileNotFoundError):
        hidden.read_text()
    with pytest.raises(FileNotFoundError):
        open(hidden, "rb")
    assert not os.access(VOLUME, os.R_OK)
    (tmp_path / "x").write_text("ok")                                   # everything else is untouched
    assert (tmp_path / "x").read_text() == "ok" and os.listdir(tmp_path) == ["x"]


def test_a_real_fetch_is_refused_and_recorded_but_loopback_passes(tmp_path, monkeypatch, _no_network_fetch):
    """`_no_network_fetch`: the built-in bias set's download (any non-loopback host) is refused and recorded, which
    fails the test at teardown even though callo turned the refusal into its own error; loopback is untouched."""
    attempts = _no_network_fetch
    monkeypatch.delenv("CALLO_BIAS_SET", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(BuiltinSetError, match=NO_NETWORK):
        builtin_set_bytes()
    assert attempts == [builtin_set_url()]
    attempts.clear()                                                           # else this test fails at teardown
    with pytest.raises(urllib.error.URLError) as e:                            # loopback reaches the real urlopen:
        urllib.request.urlopen("http://127.0.0.1:1/", timeout=5)               # refused by the OS, not the guard
    assert NO_NETWORK not in str(e.value) and attempts == []


def test_home_and_the_xdg_cache_are_a_per_test_directory(tmp_path_factory):
    """`_isolated_home`: a real `~/.cache/callo` (a cached built-in set, a vocabulary cache) can never make a test pass
    here that would fail on a clean CI runner."""
    base = os.path.realpath(tmp_path_factory.getbasetemp())
    home, xdg = os.environ["HOME"], os.environ["XDG_CACHE_HOME"]
    assert os.path.commonpath([os.path.realpath(home), base]) == base and home != REAL_HOME
    assert xdg == os.path.join(home, ".cache") and not os.listdir(home)
    assert str(Path.home()) == home
