"""`make release-check` (bench/release_check.py): the manual, network-using gate run before publishing. It refuses the
placeholder revision, and otherwise fetches builtin_set_url() into a throwaway cache and verifies the sha256. Here the
fetch is a fake; the conftest guard refuses any real one."""
import hashlib
import io
import os
import re
import urllib.request
from pathlib import Path

import pytest
from conftest import BUILTIN_SET_FILE as REPO_SET

import callo.bias as bias
from bench import release_check as rc

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def real_bytes():
    if not REPO_SET.is_file():
        pytest.skip(f"{REPO_SET} is absent (an sdist ships no data)")
    return REPO_SET.read_bytes()


def _serve(monkeypatch, body: bytes) -> list[str]:
    urls: list[str] = []

    def urlopen(url, *a, **kw):
        urls.append(str(url))
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return urls


def test_the_placeholder_revision_fails_the_gate_without_any_fetch(monkeypatch, capsys):
    monkeypatch.setattr(bias, "BUILTIN_SET_REVISION", "0" * 40)
    urls = _serve(monkeypatch, b"")
    assert rc.main() == 1
    err = capsys.readouterr().err
    assert "BUILTIN_SET_REVISION" in err and "placeholder" in err and urls == []


def test_a_pinned_revision_is_fetched_into_a_temp_cache_and_verified(monkeypatch, capsys, real_bytes):
    monkeypatch.setattr(bias, "BUILTIN_SET_REVISION", "a" * 40)
    monkeypatch.setenv("CALLO_BIAS_SET", "/nonexistent/ignored.jsonl")         # the gate always fetches
    home = os.environ["HOME"]
    urls = _serve(monkeypatch, real_bytes)
    assert rc.main() == 0
    out = capsys.readouterr().out
    assert urls == [bias.builtin_set_url()] and "a" * 40 in urls[0]
    assert bias.BUILTIN_SET_SHA256 in out and "OK" in out
    assert os.environ["HOME"] == home and os.environ["CALLO_BIAS_SET"] == "/nonexistent/ignored.jsonl"   # restored
    assert not (Path(home) / ".cache").exists()                                # the user's cache is never touched


def test_wrong_bytes_at_the_pinned_revision_fail_the_gate(monkeypatch, capsys, real_bytes):
    monkeypatch.setattr(bias, "BUILTIN_SET_REVISION", "b" * 40)
    _serve(monkeypatch, real_bytes[:-1] + b"X")
    assert rc.main() == 1
    err = capsys.readouterr().err
    assert hashlib.sha256(real_bytes[:-1] + b"X").hexdigest() in err and "FAILED" in err


def test_make_release_check_runs_the_gate_and_make_check_does_not():
    text = (ROOT / "Makefile").read_text()
    assert re.search(r"^release-check:\n\tuv run python -m bench\.release_check$", text, re.M)
    check = re.search(r"^check:(.*)$", text, re.M).group(1)
    assert "release" not in check and "release-check" in re.search(r"^\.PHONY:(.*)$", text, re.M).group(1)
    readme = (ROOT / "README.md").read_text()
    assert "make release-check" in readme.split("## Development")[1]
