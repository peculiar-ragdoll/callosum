"""Suite-wide guards.

`_no_real_llama_server`: no test may launch a real llama-server. Every launch in callo goes through one spawn point,
`callo.upstream.LlamaServerProcess.start`; this autouse fixture wraps it and refuses a binary resolved through PATH
(a bare name such as the default `llama-server`) or an existing path outside pytest's basetemp. Fakes written under
`tmp_path` run as before, and a path that does not exist reaches Popen, so callo's own "not found" handling is still
exercised. A refusal raises FileNotFoundError (callo then reports it as a missing binary) and is RECORDED: serve's
worker thread and the menu driver swallow exceptions as a model's failure, so the test fails at teardown instead.
`live` tests (a real GPU run, deselected by default) are left alone.

`_hide_local_volumes`: with `CALLO_TEST_HIDE_VOLUMES=1`, every path under `/Volumes/` reads as missing (os.stat,
os.lstat, os.open, open, os.scandir, os.listdir, os.access raise FileNotFoundError / answer False), so a run on the
dev box sees what a CI runner sees: no local volumes. A test must never depend on local data; this proves it locally.

`_no_network_fetch`: no test may fetch over `urllib`. `urllib.request.urlopen` (what `callo.bias` fetches the
built-in bias set with) refuses any host that is not loopback; the fake upstreams on 127.0.0.1 pass. A refusal raises
URLError (with a gaierror reason, as a failed DNS lookup), which callo reports as a failed fetch, and is RECORDED, so
the test fails at teardown even if that error was handled (an RFC 2606 `.invalid` host, which never resolves, is
refused without being recorded). A test that monkeypatches `urlopen` itself (a fake serving the set) replaces the
guard for that test. SCOPE: `urllib.request.urlopen` only; `http.client`, raw sockets and subprocesses are not
covered.

`_isolated_home`: `HOME` and `XDG_CACHE_HOME` point at a fresh per-test directory (outside `tmp_path`), so a real
`~/.cache/callo` (a cached built-in set, a vocabulary cache) can never make a test pass locally that would fail on a
clean CI runner. A test that sets `HOME` itself wins. `REAL_HOME` / `real_env()` restore the user's for a subprocess
that needs its own caches (`uv build`).

`builtin_set`: the built-in bias set for tests that read its content, from the repo's `bench/data/bias_set.jsonl`
through `CALLO_BIAS_SET` (never the network or the user's cache); skipped where that file is absent (an sdist).

`BENCH_ENV`: the acceptance benches take their data, model, output and llama-server paths from these variables
(bench/lib.py, bench/README.md) and refuse to start when one is unset. The unit tests run them on fakes, so each unset
one is given a placeholder that exists nowhere (set before any bench module is imported); the tests that read the
user's real data or models (`helpers.bench_path`) then skip, as they do when a file is missing. An exported but empty
variable counts as unset. `CALLO_BENCH_OUT` is always the placeholder, even when the user's is set: no test reads real
output, so none can ever write into the real results tree.

`collect_ignore`: an sdist ships tests/ but not the repo-root `bench` package, so without bench/ the test modules that
import it, directly or through another module in tests/ (`bench_importers`), are not collected.
"""
from __future__ import annotations

import ast
import builtins
import io
import os
import pathlib
import socket
import urllib.error
import urllib.parse
import urllib.request

import pytest

from callo import upstream

REFUSED = "tests may not launch a real llama-server"
NO_NETWORK = "tests may not reach the network"
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
BUILTIN_SET_FILE = pathlib.Path(__file__).resolve().parents[1] / "bench" / "data" / "bias_set.jsonl"
REAL_HOME = os.environ.get("HOME") or str(pathlib.Path.home())
BENCH_ENV = ("CALLO_BENCH_DATA", "CALLO_BENCH_MODELS", "CALLO_BENCH_OUT", "CALLO_BENCH_LLAMA_SERVER")
for _var in BENCH_ENV:
    if not os.environ.get(_var):
        os.environ[_var] = f"/callo-bench-unset/{_var}"
os.environ["CALLO_BENCH_OUT"] = "/callo-bench-unset/CALLO_BENCH_OUT"
REAL_XDG_CACHE_HOME = os.environ.get("XDG_CACHE_HOME")


def bench_importers(tests: pathlib.Path) -> list[str]:
    """The test_*.py files in `tests` that import the repo-root `bench` package, directly or through another module in
    `tests` (any import statement, a function-local one included)."""
    imports = {}
    for path in tests.glob("*.py"):
        names = set()
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        imports[path.stem] = names
    needs = {"bench"}
    while True:
        more = {mod for mod, names in imports.items() if names & needs} - needs
        if not more:
            break
        needs |= more
    return sorted(f"{mod}.py" for mod in needs if mod.startswith("test_"))


if not (pathlib.Path(__file__).resolve().parents[1] / "bench").is_dir():   # an sdist: no bench/ package
    collect_ignore = bench_importers(pathlib.Path(__file__).resolve().parent)


def real_env() -> dict[str, str]:
    """os.environ with the user's own HOME and XDG_CACHE_HOME (for a subprocess such as `uv build`, whose cache lives
    there; `_isolated_home` replaced both for the test)."""
    env = {**os.environ, "HOME": REAL_HOME}
    env.pop("XDG_CACHE_HOME", None)
    if REAL_XDG_CACHE_HOME is not None:
        env["XDG_CACHE_HOME"] = REAL_XDG_CACHE_HOME
    return env


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    yield home


def refused_binary(prog: str, basetemp: str) -> bool:
    """A bare name (a PATH lookup), or an existing file outside the session's basetemp."""
    if os.sep not in prog:
        return True
    if not os.path.exists(prog):
        return False
    base = os.path.realpath(basetemp)
    return os.path.commonpath([os.path.realpath(prog), base]) != base


@pytest.fixture(autouse=True)
def _no_real_llama_server(request, monkeypatch, tmp_path_factory):
    if request.node.get_closest_marker("live"):
        yield
        return
    basetemp = str(tmp_path_factory.getbasetemp())
    attempts: list[str] = []
    real_start = upstream.LlamaServerProcess.start

    def start(self):
        prog = str(self.argv[0]) if self.argv else ""
        if refused_binary(prog, basetemp):
            attempts.append(prog)
            raise FileNotFoundError(2, f"{REFUSED}: {prog}", prog)
        return real_start(self)

    monkeypatch.setattr(upstream.LlamaServerProcess, "start", start)
    yield attempts
    if attempts:
        pytest.fail(f"{REFUSED}: {', '.join(attempts)} (tests/conftest.py refused the launch)", pytrace=False)


HIDDEN = "/Volumes/"


def _hidden(path) -> bool:
    try:
        p = os.fspath(path)
    except TypeError:                         # a file descriptor
        return False
    if isinstance(p, bytes):
        p = p.decode(errors="replace")
    return isinstance(p, str) and os.path.abspath(p).startswith(HIDDEN)


@pytest.fixture(autouse=True)
def _hide_local_volumes(monkeypatch):
    if os.environ.get("CALLO_TEST_HIDE_VOLUMES") != "1":
        yield
        return

    def missing(fn):
        def wrapper(*a, **kw):
            path = a[0] if a else kw.get("path", kw.get("file"))
            if path is not None and _hidden(path):
                raise FileNotFoundError(2, f"hidden from tests (CALLO_TEST_HIDE_VOLUMES): {os.fspath(path)}",
                                        os.fspath(path))
            return fn(*a, **kw)
        return wrapper

    for name in ("stat", "lstat", "open", "scandir", "listdir"):
        monkeypatch.setattr(os, name, missing(getattr(os, name)))
    monkeypatch.setattr(builtins, "open", missing(builtins.open))
    monkeypatch.setattr(io, "open", missing(io.open))
    accessor = getattr(pathlib, "_NormalAccessor", None)    # Python 3.10: Path calls os.stat etc. bound at import
    if accessor is not None:
        for name in ("stat", "lstat", "open", "scandir", "listdir"):
            if hasattr(accessor, name):
                monkeypatch.setattr(accessor, name, staticmethod(missing(getattr(accessor, name))))
    real_access = os.access
    monkeypatch.setattr(os, "access", lambda path, *a, **kw: False if _hidden(path) else real_access(path, *a, **kw))
    yield


def _host(url) -> str | None:
    full = url.full_url if isinstance(url, urllib.request.Request) else str(url)
    return urllib.parse.urlsplit(full).hostname


@pytest.fixture(autouse=True)
def _no_network_fetch(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        yield []
        return
    attempts: list[str] = []
    real_urlopen = urllib.request.urlopen

    def urlopen(url, *a, **kw):
        host = _host(url)
        if host not in LOOPBACK:
            full = url.full_url if isinstance(url, urllib.request.Request) else str(url)
            if not (host or "").endswith(".invalid"):     # RFC 2606: never resolves, so a test's "nothing answers"
                attempts.append(full)                     # case; refused (as DNS would) but not a fetch
            # a DNS failure's shape (callo's "nothing answers" pre-check keys on gaierror), so a refusal fails fast
            raise urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, f"{NO_NETWORK}: {full}"))
        return real_urlopen(url, *a, **kw)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    yield attempts
    if attempts:
        pytest.fail(f"{NO_NETWORK}: {', '.join(attempts)} (tests/conftest.py refused the request)", pytrace=False)


@pytest.fixture
def builtin_set(monkeypatch) -> pathlib.Path:
    if not BUILTIN_SET_FILE.is_file():
        pytest.skip(f"{BUILTIN_SET_FILE} is absent (an sdist ships no data)")
    monkeypatch.setenv("CALLO_BIAS_SET", str(BUILTIN_SET_FILE))
    return BUILTIN_SET_FILE
