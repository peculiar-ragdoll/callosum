"""The built-in bias set leaves the wheel: `builtin-v1` is pinned by repo, revision and sha256,
resolved from `CALLO_BIAS_SET`, then callo's cache, then one download; its identity is a constant that never touches
disk or network; and only calibrating with the built-in set may fetch it."""
import hashlib
import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from conftest import BUILTIN_SET_FILE as REPO_SET
from fake_callo import CalloStub
from fake_upstream import FakeUpstream, PlantedBias
from helpers import Running, post_json, tiny_gguf
from test_profile import profile_dict

import callo.bias as bias
from callo.bias import (
    BUILTIN_SET_FILE,
    BUILTIN_SET_REPO,
    BUILTIN_SET_REVISION,
    BUILTIN_SET_SHA256,
    BUILTIN_SET_SIZE,
    BuiltinSetError,
    builtin_set_bytes,
    builtin_set_cache_path,
    builtin_set_sha256,
    builtin_set_url,
)
from callo.cli import main
from callo.modelkey import load_model_key

SHA = "f73ff674470f75993076e4e00cc306c88fa1f28bde1e85b49dcf03738a9811a1"


@pytest.fixture
def real_bytes():
    if not REPO_SET.is_file():
        pytest.skip(f"{REPO_SET} is absent (an sdist ships no data)")
    return REPO_SET.read_bytes()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """No override and an empty cache: callo's cache dir lives under this HOME."""
    monkeypatch.delenv("CALLO_BIAS_SET", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    return h


class FakeNet:
    """A stand-in for `urllib.request.urlopen` that serves `body` (or raises `error`) and counts the calls. Loopback
    requests (a stub callo serve, a fake upstream) pass through to the real `urlopen` and are not counted."""

    def __init__(self, body: bytes = b"", error: Exception | None = None):
        self.body, self.error, self.calls = body, error, []
        self.real = urllib.request.urlopen

    def __call__(self, url, *a, **kw):
        full = url.full_url if isinstance(url, urllib.request.Request) else str(url)
        if full.startswith("http://127.0.0.1:"):
            return self.real(url, *a, **kw)
        self.calls.append((full, kw.get("timeout")))
        if self.error is not None:
            raise self.error
        return io.BytesIO(self.body)


def _no_net(monkeypatch) -> FakeNet:
    net = FakeNet(error=urllib.error.URLError("network is unreachable (test)"))
    monkeypatch.setattr(urllib.request, "urlopen", net)
    return net


# ---- the pinned source ---------------------------------------------------------------------------------------

def test_the_source_is_pinned_by_repo_revision_file_and_sha256():
    assert BUILTIN_SET_REPO == "peculiar-ragdoll/callo-bias-set" and BUILTIN_SET_FILE == "bias_set.jsonl"
    assert len(BUILTIN_SET_REVISION) == 40 and all(c in "0123456789abcdef" for c in BUILTIN_SET_REVISION)
    assert BUILTIN_SET_REVISION == "05abe05d908f34325b0cf1e0ab45f420ab0e285e"     # the published upload
    assert BUILTIN_SET_SHA256 == SHA
    assert builtin_set_url() == (f"https://huggingface.co/datasets/peculiar-ragdoll/callo-bias-set/resolve/"
                                 f"{BUILTIN_SET_REVISION}/bias_set.jsonl")


def test_the_repo_copy_is_the_pinned_set(real_bytes):
    """bench/data/bias_set.jsonl is what gets uploaded: its bytes are builtin-v1's."""
    assert hashlib.sha256(real_bytes).hexdigest() == SHA and len(real_bytes) == 246_712


def test_the_cache_path_follows_callos_cache_dir(home):
    assert builtin_set_cache_path() == home / ".cache" / "callo" / "bias_set" / f"builtin-v1-{SHA[:12]}.jsonl"


# ---- identity: a constant, no disk, no network ------------------------------------------------------------

def test_the_identity_check_needs_no_file_and_no_network(home, monkeypatch):
    net = _no_net(monkeypatch)
    monkeypatch.setattr(bias, "_read_verified", lambda *a, **k: pytest.fail("builtin_set_sha256 read a file"))
    assert builtin_set_sha256() == SHA
    assert net.calls == [] and not (home / ".cache").exists()


# ---- (a) CALLO_BIAS_SET -------------------------------------------------------------------------------------

def test_the_env_override_is_read_and_verified(home, monkeypatch, real_bytes, tmp_path):
    net = _no_net(monkeypatch)
    f = tmp_path / "mine.jsonl"
    f.write_bytes(real_bytes)
    monkeypatch.setenv("CALLO_BIAS_SET", str(f))
    assert builtin_set_bytes() == real_bytes
    assert net.calls == [] and not (home / ".cache").exists()          # the override never fills the cache


def test_an_env_override_with_the_wrong_bytes_is_a_clear_error(home, monkeypatch, tmp_path):
    net = _no_net(monkeypatch)
    f = tmp_path / "other.jsonl"
    f.write_bytes(b'{"state": "x"}\n')
    monkeypatch.setenv("CALLO_BIAS_SET", str(f))
    with pytest.raises(BuiltinSetError) as e:
        builtin_set_bytes()
    msg = str(e.value)
    assert str(f) in msg and SHA in msg and hashlib.sha256(b'{"state": "x"}\n').hexdigest() in msg
    assert "CALLO_BIAS_SET" in msg and net.calls == []


def test_an_env_override_naming_a_missing_file_is_a_clear_error(home, monkeypatch, tmp_path):
    _no_net(monkeypatch)
    monkeypatch.setenv("CALLO_BIAS_SET", str(tmp_path / "nope.jsonl"))
    with pytest.raises(BuiltinSetError, match="nope.jsonl"):
        builtin_set_bytes()


# ---- (b) the cache, (c) the fetch ---------------------------------------------------------------------------

def test_a_verified_cache_is_used_without_the_network(home, monkeypatch, real_bytes):
    net = _no_net(monkeypatch)
    cache = builtin_set_cache_path()
    cache.parent.mkdir(parents=True)
    cache.write_bytes(real_bytes)
    assert builtin_set_bytes() == real_bytes and net.calls == []


def test_a_fetch_verifies_writes_the_cache_atomically_and_says_so(home, monkeypatch, real_bytes, capsys):
    net = FakeNet(real_bytes)
    monkeypatch.setattr(urllib.request, "urlopen", net)
    replaced = []
    real_replace = os.replace

    def replace(src, dst):
        replaced.append((Path(src), Path(dst), Path(src).read_bytes() == real_bytes))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    assert builtin_set_bytes() == real_bytes
    cache = builtin_set_cache_path()
    assert [c[0] for c in net.calls] == [builtin_set_url()] and 10 <= net.calls[0][1] <= 60      # a bounded timeout
    ((src, dst, complete),) = replaced
    assert dst == cache and src.parent == cache.parent and src != cache and complete   # temp in the same dir
    assert cache.read_bytes() == real_bytes and sorted(cache.parent.iterdir()) == [cache]   # no temp left behind
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "fetching the built-in bias set (247 KB) from " + builtin_set_url() in err
    assert builtin_set_bytes() == real_bytes and len(net.calls) == 1          # the second call reads the cache


def test_a_corrupt_cache_is_deleted_and_fetched_again_once(home, monkeypatch, real_bytes):
    cache = builtin_set_cache_path()
    cache.parent.mkdir(parents=True)
    cache.write_bytes(real_bytes[:-10])
    net = FakeNet(real_bytes)
    monkeypatch.setattr(urllib.request, "urlopen", net)
    assert builtin_set_bytes() == real_bytes and len(net.calls) == 1 and cache.read_bytes() == real_bytes


def test_a_corrupt_cache_and_a_bad_download_fail_after_one_fetch(home, monkeypatch):
    cache = builtin_set_cache_path()
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"truncated")
    net = FakeNet(b"not the set")
    monkeypatch.setattr(urllib.request, "urlopen", net)
    with pytest.raises(BuiltinSetError) as e:
        builtin_set_bytes()
    msg = str(e.value)
    assert len(net.calls) == 1 and SHA in msg and hashlib.sha256(b"not the set").hexdigest() in msg
    assert builtin_set_url() in msg and not cache.exists()               # nothing unverified is kept


@pytest.mark.parametrize("error", [urllib.error.URLError("nodename nor servname provided"),
                                   urllib.error.HTTPError("u", 404, "Not Found", {}, None),   # type: ignore[arg-type]
                                   TimeoutError("timed out")])
def test_a_failed_fetch_names_the_url_the_offline_route_and_data(home, monkeypatch, error):
    monkeypatch.setattr(urllib.request, "urlopen", FakeNet(error=error))
    with pytest.raises(BuiltinSetError) as e:
        builtin_set_bytes()
    msg = str(e.value)
    assert builtin_set_url() in msg and "CALLO_BIAS_SET" in msg and "--data" in msg
    assert not builtin_set_cache_path().exists()


def test_an_unwritable_cache_still_returns_the_verified_bytes(home, monkeypatch, real_bytes, capsys):
    monkeypatch.setattr(urllib.request, "urlopen", FakeNet(real_bytes))
    (home / ".cache").write_text("a file where the cache dir should be")
    assert builtin_set_bytes() == real_bytes
    assert "cannot cache" in capsys.readouterr().err


class Body(io.BytesIO):
    """A response body that records the size each read() asked for."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.asked: list[int | None] = []

    def read(self, size=-1):
        self.asked.append(size)
        return super().read(size)


def test_the_download_is_bounded_and_a_longer_body_is_an_error(home, monkeypatch, real_bytes):
    """At most BUILTIN_SET_SIZE + 1 bytes are read: a body longer than the set is refused before any hashing."""
    assert BUILTIN_SET_SIZE == 246_712 == len(real_bytes)
    bodies = []

    def urlopen(url, *a, **kw):
        bodies.append(Body(real_bytes + b"\n" * 10_000))
        return bodies[-1]

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(BuiltinSetError) as e:
        builtin_set_bytes()
    assert bodies[0].asked == [BUILTIN_SET_SIZE + 1]
    assert "longer than" in str(e.value) and "246712" in str(e.value).replace(",", "")
    assert builtin_set_url() in str(e.value) and not builtin_set_cache_path().exists()
    bodies.clear()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: bodies.append(Body(real_bytes)) or bodies[-1])
    assert builtin_set_bytes() == real_bytes and bodies[0].asked == [BUILTIN_SET_SIZE + 1]


@pytest.mark.parametrize("fail", ["replace", "write"])
def test_a_failure_mid_write_returns_the_verified_bytes_and_leaves_no_temp_file(home, monkeypatch, real_bytes,
                                                                                capsys, fail):
    monkeypatch.setattr(urllib.request, "urlopen", FakeNet(real_bytes))
    if fail == "replace":
        def boom(*a, **k):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(os, "replace", boom)
    else:
        real_fdopen = os.fdopen

        class Full:
            def __init__(self, f):
                self.f = f

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.f.close()

            def write(self, data):
                self.f.write(data[:100])
                raise OSError(28, "No space left on device")

        monkeypatch.setattr(os, "fdopen", lambda *a, **k: Full(real_fdopen(*a, **k)))
    assert builtin_set_bytes() == real_bytes
    cache = builtin_set_cache_path()
    assert cache.parent.is_dir() and list(cache.parent.iterdir()) == []        # no temp file, no partial cache
    assert "cannot cache" in capsys.readouterr().err


def test_the_messages_go_to_the_callers_err_stream(home, monkeypatch, real_bytes, capsys):
    monkeypatch.setattr(urllib.request, "urlopen", FakeNet(real_bytes))
    (home / ".cache").write_text("a file where the cache dir should be")
    err = io.StringIO()
    assert builtin_set_bytes(err=err) == real_bytes
    lines = err.getvalue().splitlines()
    assert len(lines) == 2 and lines[0].startswith("callo: fetching the built-in bias set (247 KB) from ")
    assert "cannot cache" in lines[1] and capsys.readouterr().err == ""


def test_calibrate_bias_writes_the_download_line_to_its_err(home, monkeypatch, real_bytes):
    """`calibrate_bias.run(args, out=, err=)` threads its err stream into the fetch."""
    import argparse

    from callo import calibrate_bias

    monkeypatch.setattr(urllib.request, "urlopen", FakeNet(real_bytes + b"x"))     # fetched, then refused
    err = io.StringIO()
    args = argparse.Namespace(data=None, out=None, raw=None, resume=False, seed=None, min_questions=None,
                              max_per_n=None, allow_busy=False, url="http://127.0.0.1:1")
    assert calibrate_bias.run(args, out=io.StringIO(), err=err) == 1
    text = err.getvalue()
    assert text.startswith("callo: fetching the built-in bias set") and "longer than" in text


# ---- the CLI: a clean error, and the paths that never fetch -----------------------------------------------

def test_calibrate_bias_on_a_failed_fetch_exits_1_with_one_clean_line(home, monkeypatch, capsys):
    net = _no_net(monkeypatch)
    stub = CalloStub(PlantedBias())
    try:
        assert main(["calibrate", "--bias", "--url", stub.url]) == 1
    finally:
        stub.stop()
    err = capsys.readouterr().err
    assert "Traceback" not in err and err.startswith("callo: ") and builtin_set_url() in err
    assert "CALLO_BIAS_SET" in err and "--data" in err
    assert [c[0] for c in net.calls] == [builtin_set_url()] and stub.posts == []


def test_out_without_data_is_refused_before_any_fetch(home, monkeypatch, tmp_path, capsys):
    net = _no_net(monkeypatch)
    stub = CalloStub(PlantedBias())
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--out", str(tmp_path / "b.json")]) == 2
    finally:
        stub.stop()
    assert "--out needs --data" in capsys.readouterr().err and net.calls == [] and stub.posts == []


def test_calibrate_bias_with_data_never_fetches(home, monkeypatch, tmp_path, capsys):
    """A user's own set is compared with the built-in one by the pinned sha256: no fetch."""
    net = _no_net(monkeypatch)
    lines = []
    for i in range(24):
        crit = {f"o{i}_{k}": None for k in range(3)}
        lines.append(json.dumps({"state": f"s{i}", "questions": {"q": {"type": "choice", "instructions": "pick",
                                                                       "criteria": crit}}}))
    data = tmp_path / "mine.jsonl"
    data.write_text("\n".join(lines) + "\n")
    monkeypatch.chdir(tmp_path)
    stub = CalloStub(PlantedBias())
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data)]) == 0
    finally:
        stub.stop()
    assert json.loads(capsys.readouterr().out)["set"]["builtin"] is False
    assert net.calls == [] and not (home / ".cache" / "callo" / "bias_set").exists()


def test_serve_with_a_bias_profile_never_fetches(home, monkeypatch, tmp_path):
    """Loopback requests (the fake upstream, the client) pass; anything else would be a fetch and raises."""
    net = _no_net(monkeypatch)
    up = FakeUpstream().start()
    try:
        g = tiny_gguf(tmp_path / "m.gguf")
        up.props["model_path"] = str(g)
        prof = tmp_path / "bias.json"
        prof.write_text(json.dumps(profile_dict(key=load_model_key(g, None), entries={3: [0.6, 0.3, 0.1]})))
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"),
                     "--position-bias", str(prof)])
        try:
            body = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "x", "b": "y", "c": "z"}}}}
            status, _, _ = post_json(f"{r.url}/v1/systemone", body)
            assert status == 200
        finally:
            assert r.finish() == 0
    finally:
        up.stop()
    assert net.calls == [] and not (home / ".cache" / "callo" / "bias_set").exists()


def test_imports_and_help_never_touch_the_network(home, tmp_path):
    """A fresh interpreter with `urlopen` and socket connects raising: every callo module imports, `callo --help` and
    `callo calibrate --help` print, and the identity check answers."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import pkgutil, importlib, socket, urllib.request, sys\n"
        "def boom(*a, **k):\n"
        "    raise RuntimeError('network touched')\n"
        "urllib.request.urlopen = boom\n"
        "socket.socket.connect = boom\n"
        "socket.create_connection = boom\n"
        "import callo\n"
        "for m in pkgutil.walk_packages(callo.__path__, 'callo.'):\n"
        "    importlib.import_module(m.name)\n"
        "from callo.cli import main\n"
        "from callo.bias import builtin_set_sha256\n"
        "for argv in (['--help'], ['calibrate', '--help'], ['serve', '--help']):\n"
        "    try:\n"
        "        main(argv)\n"
        "    except SystemExit as e:\n"
        "        assert e.code == 0, e.code\n"
        f"assert builtin_set_sha256() == {SHA!r}\n"
        "print('ok')\n")
    env = {k: v for k, v in os.environ.items() if k != "CALLO_BIAS_SET"}
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env={**env, "HOME": str(home)},
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.rstrip().endswith("ok") and not (home / ".cache").exists()
