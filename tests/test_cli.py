"""The CLI, end to end against the fake upstream (attach mode) and the fake llama-server executable (launch mode).
(v0.3: the audio and video self-tests and stats, the audio-only projector.)"""
import argparse
import base64
import io
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fake_upstream import FakeUpstream
from helpers import STARTUP_TIMEOUT, Running, get_json, post_json, tiny_gguf
from test_launch import FAKE
from test_profile import profile_dict

import callo.cli as cli_module
from callo.av import RED_URL, TONE_URL
from callo.cli import DEFAULT_PORT, build_parser, main, serve, split_passthrough
from callo.errors import NO_VIDEO_DETAIL, VIDEO_OFF_DETAIL, InvalidInput
from callo.modelkey import load_model_key
from callo.profile import prompt_sha256

BODY = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "one", "b": "two"}}}}


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "callo 0.3.1"


def test_split_passthrough_and_defaults():
    assert split_passthrough(["serve", "--model", "x", "--", "-np", "8"]) == (["serve", "--model", "x"], ["-np", "8"])
    assert split_passthrough(["serve", "--upstream", "u"]) == (["serve", "--upstream", "u"], [])
    a = build_parser().parse_args(["serve", "--upstream", "http://127.0.0.1:8199"])
    assert (a.port, a.s1_slots, a.temperature, a.cache, a.s1_timeout, a.parallel, a.host) == \
        (DEFAULT_PORT, 1, None, True, 900.0, 4, "127.0.0.1")        # temperature None: resolved by serve
    # --cache is on by default since 0.3.0; --no-cache turns it off, --cache still parses
    assert build_parser().parse_args(["serve", "--upstream", "u", "--no-cache"]).cache is False
    assert build_parser().parse_args(["serve", "--upstream", "u", "--cache"]).cache is True
    assert (a.think_budget, a.think_max, a.think_profile, a.allow_think_probe) == ("0", "4096", None, False)
    # strings on purpose: serve parses them in phase 1, so a non-integer is a refusal (exit 1), not a usage error
    assert DEFAULT_PORT == 8180
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve"])                    # --upstream or --model is required


def test_serve_attach_mode_end_to_end(tmp_path):
    up = FakeUpstream().start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        try:
            assert r.port
            _, _, stats = get_json(f"{r.url}/callo/stats")
            assert stats["probe"]["clears_idle"] is False and stats["router"]["pinning"] is True
            assert stats["router"]["reserved"] == [0] and stats["temperature"] == 1.0
            status, headers, out = post_json(f"{r.url}/v1/systemone", BODY)
            assert status == 200 and headers["X-Callo-Slot"] == "0" and out["answers"]["q"]["choice"] == "b"
            assert out["model"] == "fake-9b"
            status, headers, _ = post_json(f"{r.url}/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
            assert headers["X-Callo-Slot"] == "1"
            assert list((tmp_path / "c").glob("vocab-*.json"))
            comps = [q.json for q in up.requests if q.path == "/completion"]
            assert comps[0]["id_slot"] == 3 and comps[1]["id_slot"] == 0     # the probe ran first (N-1 then 0)
            assert all(c["cache_prompt"] is True for c in comps[2:])          # engine passes: --cache on (default)
        finally:
            assert r.finish() == 0
    finally:
        up.stop()


@pytest.mark.parametrize("cmd", ["serve", "calibrate"])
def test_the_cache_help_states_its_default_in_words_only(capsys, cmd):
    """Python 3.10-3.12's BooleanOptionalAction appends " (default: True)" to --cache's help, which
    already says the default in words; callo's OnOff action leaves the help as written, with no warning under -W
    error (3.14 deprecates some of the action's parameters), and parses as before."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([cmd, "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--cache, --no-cache" in out and "(default: True)" not in out and "(default: None)" not in out
    base = ["serve", "--upstream", "u"] if cmd == "serve" else ["calibrate"]
    default = True if cmd == "serve" else None
    assert [parser.parse_args([*base, *f]).cache for f in ([], ["--cache"], ["--no-cache"])] == [default, True, False]


@pytest.mark.parametrize("flags, on", [([], True), (["--cache"], True), (["--no-cache"], False)])
def test_serve_cache_is_on_by_default_and_no_cache_turns_it_off(tmp_path, flags, on):
    up = FakeUpstream().start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        r = Running(["serve", "--upstream", up.url, "--port", "0", *flags, "--cache-dir", str(tmp_path / "c")])
        try:
            post_json(f"{r.url}/v1/systemone", BODY)
            comps = [q.json for q in up.requests if q.path == "/completion"]
            assert comps[-1]["cache_prompt"] is on
            _, _, stats = get_json(f"{r.url}/callo/stats")
            assert stats["cache_prompt"]["plain"] is on and stats["provenance"]["cache_prompt"] is on
        finally:
            r.finish()
    finally:
        up.stop()


def test_serve_disables_agent_pinning_when_the_upstream_clears_idle_slots(tmp_path):
    up = FakeUpstream(clears_idle=True).start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        try:
            _, _, stats = get_json(f"{r.url}/callo/stats")
            assert stats["probe"]["clears_idle"] is True and stats["router"]["pinning"] is False
            _, headers, _ = post_json(f"{r.url}/v1/chat/completions", {"messages": []})
            assert headers["X-Callo-Slot"] == "-"
            _, headers, _ = post_json(f"{r.url}/v1/systemone", BODY)
            assert headers["X-Callo-Slot"] == "0"                              # S1 stays pinned
        finally:
            r.finish()
    finally:
        up.stop()


def test_serve_native_decision_model_passes_systemone_through(tmp_path):
    up = FakeUpstream().start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "kev.gguf", decision="kev"))
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        try:
            _, headers, out = post_json(f"{r.url}/v1/systemone", BODY)
            assert "echo" in out and headers["X-Callo-Slot"] == "-"
            assert get_json(f"{r.url}/callo/stats")[2]["temperature"] is None          # engine off: no S1 temperature
            assert not [q for q in up.requests if q.path == "/apply-template"]   # no engine, no self-test
        finally:
            r.finish()
    finally:
        up.stop()


def test_serve_refuses_a_router_upstream_and_a_bad_s1_slots(tmp_path):
    up = FakeUpstream().start()
    try:
        up.props["role"] = "router"
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        assert r.finish() == 1 and r.port is None
        del up.props["role"]
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--s1-slots", "4", "--cache-dir", str(tmp_path / "c")])
        assert r.finish() == 1
    finally:
        up.stop()


def test_serve_launch_mode_exits_3_when_the_child_dies(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_text(FAKE.replace("{python}", sys.executable))
    exe.chmod(0o755)
    model = tiny_gguf(tmp_path / "m.gguf")
    monkeypatch.setenv("FAKE_ARGV_FILE", str(tmp_path / "argv.json"))
    monkeypatch.setenv("FAKE_MODEL", str(model))
    monkeypatch.setenv("FAKE_MODE", "die-later")
    monkeypatch.setenv("FAKE_DIE_AFTER", "5")      # long enough for serve to reach `ready` even on a slow CI box
    r = Running(["serve", "--model", str(model), "--llama-server", str(exe), "--port", "0", "--ctx", "8192",
                 "--cache-dir", str(tmp_path / "c"), "--", "--no-mmap"])
    assert r.port
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv == ["-m", str(model), "--port", "0", "-c", "8192", "-np", "4", "-kvu", "--no-cache-idle-slots",
                    "-b", "512", "-ub", "512", "--jinja", "--slot-save-path", str(tmp_path / "c" / "slots"),
                    "--no-mmap"]
    r.thread.join(timeout=STARTUP_TIMEOUT)                              # the fake dies 5 s after listening
    assert r.rc == 3 and not list((tmp_path / "c").glob("llama-server-*.pid"))


def test_serve_launch_mode_rejects_multi_host_passthrough(tmp_path):
    r = Running(["serve", "--model", "/x.gguf", "--llama-server", "/nonexistent", "--port", "0",
                 "--cache-dir", str(tmp_path / "c"), "--", "--host", "a,b"])
    assert r.finish() == 1


class _S1(BaseHTTPRequestHandler):
    answers = {"answers": {"q": {"type": "choice", "choice": "b", "confidence": 0.5,
                                 "probabilities": {"a": 0.25, "b": 0.75}}}, "model": "m",
               "usage": {"input_tokens": 1, "output_tokens": 0}}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        data = json.dumps(self.answers).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_classify_prints_a_table_or_json(capsys, monkeypatch):
    _S1.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _S1)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        monkeypatch.setattr(sys, "stdin", io.StringIO("the state text"))
        assert main(["classify", "--url", url, "--options", "a, b", "--instructions", "pick"]) == 0
        out = capsys.readouterr().out
        assert "0.7500  b" in out and "choice: b" in out
        assert _S1.seen[-1] == {"state": "the state text", "questions": {"q": {"type": "choice", "instructions": "pick",
                                                                               "criteria": {"a": None, "b": None}}}}
        monkeypatch.setattr(sys, "stdin", io.StringIO("x"))
        assert main(["classify", "--url", url, "--options", "a,b", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["answers"]["q"]["choice"] == "b"
    finally:
        srv.shutdown()
        srv.server_close()
    assert main(["classify", "--url", "http://127.0.0.1:1", "--options", "a,b"]) == 1


class _Tempered(BaseHTTPRequestHandler):
    """Answers with softmax(z) where z is encoded in the record's state: the served-at-T=1 probabilities.
    `GET /callo/stats` reports `temperature` (class attribute, 1.0 unless a test changes it)."""

    temperature = 1.0
    posts = 0
    stats_extra: dict = {}

    def do_GET(self):
        data = json.dumps({"temperature": type(self).temperature, **type(self).stats_extra}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        type(self).posts += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        z = json.loads(body["state"])
        top = max(z)
        e = [2.718281828459045 ** (v - top) for v in z]
        p = [x / sum(e) for x in e]
        out = {"answers": {"q": {"type": "choice", "choice": "a", "confidence": 0.0,
                                 "probabilities": dict(zip("abcd", p, strict=True))}}, "model": "m",
               "usage": {"input_tokens": 1, "output_tokens": 0}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_calibrate_recovers_a_known_temperature_from_a_labelled_jsonl(tmp_path, capsys):
    import math
    import random
    rng = random.Random(0)
    lines = []
    for i in range(1500):
        z = [rng.gauss(0, 2) for _ in range(4)]
        w = [math.exp(v / 2.0) for v in z]
        label = "abcd"[rng.choices(range(4), weights=w)[0]]
        lines.append(json.dumps({"state": json.dumps(z), "questions": {"q": {"type": "choice", "criteria": {k: None for k in "abcd"},
                                                                            "label": label, "src": "syn"}},
                                 "_meta": {"id": f"syn/{i}", "variant": "clean"}}))
    lines.append(json.dumps({"state": "[0,0,0,0]", "questions": {"q": {"type": "choice", "criteria": {k: None for k in "abcd"}, "label": "a"}},
                             "_meta": {"id": "syn/x", "variant": "paraphrase"}}))     # non-clean: skipped
    data = tmp_path / "labeled.jsonl"
    data.write_text("\n".join(lines) + "\n")
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out_file = tmp_path / "T.json"
        rc = main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data), "--out", str(out_file)])
    finally:
        srv.shutdown()
        srv.server_close()
    printed = capsys.readouterr().out
    assert rc == 0 and "T=" in printed and "NLL" in printed and "ECE" in printed
    result = json.loads(out_file.read_text())
    assert 1.8 <= result["temperature"] <= 2.2 and result["n"] == 1500
    assert result["nll_after"] < result["nll_before"]
    assert result["limit_note"] is None and "search limit" not in printed


def test_calibrate_notes_a_temperature_at_the_search_limit(tmp_path, capsys):
    """Labels drawn near-uniformly from confident readouts: the best T is past the grid's top, so the fit lands on
    GRID[-1] and says so, on stdout and in the --out JSON; the interior fit above records no note."""
    import math
    import random

    from callo.calibrate import GRID
    rng = random.Random(1)
    lines = []
    for _row in range(600):
        z = [rng.gauss(0, 2) for _ in range(4)]
        label = "abcd"[rng.choices(range(4), weights=[math.exp(v / 200.0) for v in z])[0]]
        lines.append(json.dumps({"state": json.dumps(z), "questions": {"q": {"type": "choice", "label": label,
                                                                            "criteria": {k: None for k in "abcd"}}}}))
    data = tmp_path / "labeled.jsonl"
    data.write_text("\n".join(lines) + "\n")
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out_file = tmp_path / "T.json"
        rc = main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data), "--out", str(out_file)])
    finally:
        srv.shutdown()
        srv.server_close()
    printed = capsys.readouterr().out
    note = "T is at the search limit (16): the probabilities are still overconfident after calibration"
    result = json.loads(out_file.read_text())
    assert rc == 0 and result["temperature"] == GRID[-1] and note in printed and result["limit_note"] == note
    assert result["grid"] == "181-point log grid on 0.25..16"


def test_calibrate_on_a_flat_loss_keeps_t_and_notes_nothing(tmp_path, capsys):
    """Every readout uniform: the NLL is the same at every T, so T is the tie-break GRID[0] and no note is made."""
    from callo.calibrate import GRID
    data = tmp_path / "labeled.jsonl"
    data.write_text("".join(json.dumps({"state": "[0,0,0,0]", "questions": {"q": {
        "type": "choice", "label": k, "criteria": {c: None for c in "abcd"}}}}) + "\n" for k in "abcd"))
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out_file = tmp_path / "T.json"
        rc = main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data), "--out", str(out_file)])
    finally:
        srv.shutdown()
        srv.server_close()
    result = json.loads(out_file.read_text())
    assert rc == 0 and result["temperature"] == GRID[0] and result["limit_note"] is None
    assert "search limit" not in capsys.readouterr().out


def test_calibrate_refuses_unless_the_server_runs_at_temperature_one(tmp_path, capsys):
    data = tmp_path / "labeled.jsonl"
    data.write_text(json.dumps({"state": "[0,0,0,0]", "questions": {"q": {"type": "choice", "label": "a",
                                                                          "criteria": {k: None for k in "abcd"}}}}) + "\n")
    _Tempered.temperature, _Tempered.posts = 1.91, 0
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rc = main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data)])
    finally:
        srv.shutdown()
        srv.server_close()
        _Tempered.temperature = 1.0
    err = capsys.readouterr().err
    assert rc == 2 and "temperature" in err and "1.91" in err
    assert _Tempered.posts == 0                     # refused before a single classification request


def test_serve_logs_orphans_including_one_still_loading(tmp_path, caplog):
    cache = tmp_path / "c"
    cache.mkdir()
    sleepers = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) for _ in range(2)]
    try:
        (cache / "llama-server-starting-4242.pid").write_text(f"{sleepers[0].pid}\n")   # a child killed mid-load
        (cache / "llama-server-9999.pid").write_text(f"{sleepers[1].pid}\n")
        with caplog.at_level(logging.WARNING, logger="callo"):
            r = Running(["serve", "--model", "/x.gguf", "--llama-server", str(tmp_path / "missing-llama-server"),
                         "--port", "0", "--cache-dir", str(cache)])
            assert r.finish() == 1                     # the missing binary is a CalloError, not a traceback
        text = caplog.text
        assert f"pid {sleepers[0].pid} (still loading)" in text and f"kill {sleepers[0].pid}" in text
        assert f"pid {sleepers[1].pid} (port 9999)" in text
        assert "could not launch" in text
    finally:
        for s in sleepers:
            s.terminate()
            s.wait(timeout=5)


def test_serve_attach_mode_with_a_single_slot_upstream(tmp_path):
    up = FakeUpstream(total_slots=1).start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        try:
            assert r.port
            _, _, stats = get_json(f"{r.url}/callo/stats")
            assert stats["router"]["reserved"] == [] and stats["router"]["pinning"] is False
            assert stats["probe"]["clears_idle"] is None
            status, headers, out = post_json(f"{r.url}/v1/systemone", BODY)
            assert status == 200 and headers["X-Callo-Slot"] == "-" and out["answers"]["q"]["choice"] == "b"
            assert all("id_slot" not in q.json for q in up.requests if q.path == "/completion")
        finally:
            assert r.finish() == 0
        up.props["total_slots"] = 0
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        assert r.finish() == 1 and r.port is None      # a CalloError, not a ValueError traceback
    finally:
        up.stop()


@pytest.fixture(params=[signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def sentinel(request):
    """(signal, calls): a harmless handler for that signal, so a serve that failed to install its own cannot kill
    or interrupt pytest; the original is restored after."""
    sig, seen = request.param, []
    prev = signal.signal(sig, lambda *_: seen.append("sentinel"))
    try:
        yield sig, seen
    finally:
        signal.signal(sig, prev)


def _assert_previous_handler_restored(sig, seen):
    assert seen == []                                                # serve's own handler took the signal
    os.kill(os.getpid(), sig)
    deadline = time.monotonic() + 5
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen == ["sentinel"]


def test_signal_during_launch_startup_stops_the_loading_child(tmp_path, monkeypatch, sentinel):
    sig, seen = sentinel
    exe = tmp_path / "llama-server"
    exe.write_text(FAKE.replace("{python}", sys.executable))
    exe.chmod(0o755)
    model = tiny_gguf(tmp_path / "m.gguf")
    cache = tmp_path / "c"
    monkeypatch.setenv("FAKE_ARGV_FILE", str(tmp_path / "argv.json"))
    monkeypatch.setenv("FAKE_MODEL", str(model))
    monkeypatch.setenv("FAKE_STARTUP_DELAY", "60")     # still "loading the model" when the signal lands
    for k in ("FAKE_MODE", "FAKE_DIE_AFTER", "FAKE_JSONL", "FAKE_CHILD_PID_FILE"):
        monkeypatch.delenv(k, raising=False)
    args, extra = split_passthrough(["serve", "--model", str(model), "--llama-server", str(exe), "--port", "0",
                                     "--cache-dir", str(cache)])
    ns = build_parser().parse_args(args)
    ns.extra = extra
    starting = cache / f"llama-server-starting-{os.getpid()}.pid"
    child: list[int] = []
    bound: list[int] = []
    stop = threading.Event()
    backstop = threading.Timer(45, stop.set)           # a broken handler must fail the test, not hang it

    def fire():
        deadline = time.monotonic() + 20
        while not starting.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        child.append(int(starting.read_text()))
        os.kill(os.getpid(), sig)

    t0 = time.monotonic()
    threading.Thread(target=fire, daemon=True).start()
    backstop.start()
    try:
        rc = serve(ns, ready=lambda p: pytest.fail("startup must not complete"), stop=stop,   # main thread
                   listening=bound.append)
    finally:
        backstop.cancel()
    assert rc == 0 and time.monotonic() - t0 < 30 and not stop.is_set()   # returned well before the 60 s load
    with pytest.raises(ConnectionRefusedError):                          # callo's loading server was shut down
        socket.create_connection(("127.0.0.1", bound[0]), timeout=2).close()
    with pytest.raises(ProcessLookupError):
        os.kill(child[0], 0)                                         # the loading llama-server was stopped (and reaped)
    assert not list(cache.glob("llama-server-*.pid"))
    _assert_previous_handler_restored(sig, seen)


def test_signal_after_startup_is_a_clean_stop(tmp_path, sentinel):
    sig, seen = sentinel
    up = FakeUpstream().start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        args, _ = split_passthrough(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        ns = build_parser().parse_args(args)
        ns.extra = []
        stop = threading.Event()
        backstop = threading.Timer(45, stop.set)
        backstop.start()
        t0 = time.monotonic()
        try:
            rc = serve(ns, ready=lambda p: os.kill(os.getpid(), sig), stop=stop)   # main thread
        finally:
            backstop.cancel()
        assert rc == 0 and time.monotonic() - t0 < 30                # the handler set `stop`, not the backstop
        _assert_previous_handler_restored(sig, seen)
    finally:
        up.stop()


class _SignalOnError(logging.Handler):
    """Sends `sig` to this process while serve is logging its CalloError: the second-signal-on-the-error-path race."""

    def __init__(self, sig):
        super().__init__(logging.ERROR)
        self.sig = sig

    def emit(self, record):
        os.kill(os.getpid(), self.sig)


def test_a_signal_while_serve_logs_a_startup_error_is_not_a_traceback(tmp_path, sentinel):
    sig, seen = sentinel
    up = FakeUpstream().start()
    up.props["role"] = "router"                                      # discover raises CalloError
    handler = _SignalOnError(sig)
    logging.getLogger("callo").addHandler(handler)
    try:
        args, _ = split_passthrough(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        ns = build_parser().parse_args(args)
        ns.extra = []
        assert serve(ns, stop=threading.Event()) == 1                # main thread: serve's handlers are live
    finally:
        logging.getLogger("callo").removeHandler(handler)
        up.stop()
    _assert_previous_handler_restored(sig, seen)


def _fake_exe(tmp_path, monkeypatch, **env):
    exe = tmp_path / "llama-server"
    exe.write_text(FAKE.replace("{python}", sys.executable))
    exe.chmod(0o755)
    model = tiny_gguf(tmp_path / "m.gguf")
    monkeypatch.setenv("FAKE_ARGV_FILE", str(tmp_path / "argv.json"))
    monkeypatch.setenv("FAKE_MODEL", str(model))
    for k in ("FAKE_MODE", "FAKE_DIE_AFTER", "FAKE_JSONL", "FAKE_CHILD_PID_FILE", "FAKE_STARTUP_DELAY",
              "FAKE_ERASE_STATUS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return exe, model


@pytest.mark.parametrize("flags", [["--temperature", "0"], ["--temperature", "-1"], ["--temperature", "nan"],
                                   ["--s1-slots", "0"]])
def test_serve_rejects_a_bad_temperature_or_s1_slots_before_launching(tmp_path, monkeypatch, caplog, flags):
    exe, model = _fake_exe(tmp_path, monkeypatch)
    with caplog.at_level(logging.ERROR, logger="callo"):
        r = Running(["serve", "--model", str(model), "--llama-server", str(exe), "--port", "0",
                     "--cache-dir", str(tmp_path / "c"), *flags])
        assert r.finish() == 1
    assert not (tmp_path / "argv.json").exists()                     # refused before any llama-server ran
    assert flags[0] in caplog.text


def test_serve_busy_port_is_a_clean_error_before_launching(tmp_path, monkeypatch, caplog):
    exe, model = _fake_exe(tmp_path, monkeypatch)
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    port = busy.getsockname()[1]
    try:
        with caplog.at_level(logging.ERROR, logger="callo"):
            r = Running(["serve", "--model", str(model), "--llama-server", str(exe), "--port", str(port),
                         "--cache-dir", str(tmp_path / "c")])
            assert r.finish() == 1 and r.port is None
    finally:
        busy.close()
    assert not (tmp_path / "argv.json").exists()                     # bound (and failed) before the model load
    assert f"127.0.0.1:{port}" in caplog.text and "--port" in caplog.text


def test_serve_launch_mode_fails_promptly_when_the_child_dies_right_after_listening(tmp_path, monkeypatch, caplog):
    exe, model = _fake_exe(tmp_path, monkeypatch, FAKE_MODE="exit-after-listen")
    t0 = time.monotonic()
    with caplog.at_level(logging.ERROR, logger="callo"):
        r = Running(["serve", "--model", str(model), "--llama-server", str(exe), "--port", "0",
                     "--cache-dir", str(tmp_path / "c")])
        r.thread.join(timeout=STARTUP_TIMEOUT)
    # Once "listening on" is seen, the child's death is detected at once (wait_health's `alive` check), so serve
    # returns in ~time-to-listen, never the 600 s /health ceiling. The bound tracks the startup budget, not 20 s,
    # so a slow-but-correct runner that takes a while just to print "listening on" is not failed for it.
    assert r.rc == 1 and time.monotonic() - t0 < STARTUP_TIMEOUT    # not wait_health's 600 s
    assert "exited" in caplog.text and not list((tmp_path / "c").glob("llama-server-*.pid"))


def test_calibrate_against_a_native_decision_model_says_the_engine_is_off(tmp_path, capsys):
    data = tmp_path / "labeled.jsonl"
    data.write_text(json.dumps({"state": "[0,0,0,0]", "questions": {"q": {"type": "choice", "label": "a",
                                                                          "criteria": {k: None for k in "abcd"}}}}) + "\n")
    _Tempered.temperature, _Tempered.posts = None, 0
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rc = main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data)])
    finally:
        srv.shutdown()
        srv.server_close()
        _Tempered.temperature = 1.0
    err = capsys.readouterr().err
    assert rc == 2 and "engine is off" in err and "native decision model" in err
    assert "restart" not in err and "--temperature" not in err and _Tempered.posts == 0


class _Messy(_Tempered):
    """`_Tempered`, except a state of "noanswers" gets a 200 with no `answers`."""

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        raw = self.rfile.read(n)
        if json.loads(raw)["state"] == "noanswers":
            data = json.dumps({"model": "m"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.rfile = io.BytesIO(raw)
        super().do_POST()


def test_calibrate_skips_and_counts_malformed_replies(tmp_path, capsys):
    crit = {k: None for k in "abcd"}
    lines = [json.dumps({"state": json.dumps([i % 4, 0, 1, 2]), "questions": {"q": {"type": "choice", "criteria": crit,
                                                                                   "label": "abcd"[i % 4]}}})
             for i in range(30)]
    lines.append(json.dumps({"state": "noanswers", "questions": {"q": {"type": "choice", "criteria": crit, "label": "a"}}}))
    lines.append(json.dumps({"state": "[0,0,0,0]", "questions": {"q": {"type": "choice", "criteria": crit, "label": "z"}}}))
    data = tmp_path / "labeled.jsonl"
    data.write_text("\n".join(lines) + "\n")
    _Tempered.temperature = 1.0
    srv = HTTPServer(("127.0.0.1", 0), _Messy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out_file = tmp_path / "T.json"
        rc = main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data),
                   "--out", str(out_file)])
    finally:
        srv.shutdown()
        srv.server_close()
    err = capsys.readouterr().err
    result = json.loads(out_file.read_text())
    assert rc == 0 and result["n"] == 30
    assert result["skipped_records"] == 1 and result["skipped_answers"] == 1
    assert "skipped 1 record(s) and 1 answer(s)" in err


LOADING = {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}


def _raw(method, url, body=None):
    """(status, headers, parsed body) without raising on 503."""
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


def test_serve_answers_503_loading_model_during_the_load_and_200_after(tmp_path, monkeypatch):
    exe, model = _fake_exe(tmp_path, monkeypatch, FAKE_STARTUP_DELAY="4")
    args, extra = split_passthrough(["serve", "--model", str(model), "--llama-server", str(exe), "--port", "0",
                                     "--cache-dir", str(tmp_path / "c")])
    ns = build_parser().parse_args(args)
    ns.extra = extra
    stop, ready, listening = threading.Event(), threading.Event(), []
    rc = []
    th = threading.Thread(target=lambda: rc.append(serve(ns, ready=lambda p: ready.set(), stop=stop,
                                                         listening=listening.append)), daemon=True)
    th.start()
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while not listening and time.monotonic() < deadline:
            time.sleep(0.01)
        url = f"http://127.0.0.1:{listening[0]}"
        for method, path, body in (("GET", "/health", None), ("POST", "/v1/chat/completions", {"messages": []}),
                                   ("POST", "/v1/systemone", BODY)):
            t0 = time.monotonic()
            status, headers, out = _raw(method, url + path, body)
            assert time.monotonic() - t0 < 2                         # a fast 503, not a hang
            assert status == 503 and out == LOADING and headers["X-Callo-Slot"] == "-"
        assert not ready.is_set()                                    # still loading: the fake waits 4 s
        assert ready.wait(timeout=STARTUP_TIMEOUT)
        status, _, out = _raw("GET", url + "/health")
        assert status == 200 and out == {"status": "ok"}
    finally:
        stop.set()
        th.join(timeout=STARTUP_TIMEOUT)
    assert rc == [0] and not list((tmp_path / "c").glob("llama-server-*.pid"))


# ---- v0.2: --mmproj, vision detection, the delimiter, the vision self-test -----------------------------------

DELIM = "<|im_end|>\n<|im_start|>user"
IMG = "data:image/png;base64,iVBORw0KGgo="
IMG_BODY = {"state": "", "images": [IMG], "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}}


def test_serve_launch_mode_with_mmproj_detects_vision_and_runs_both_self_tests(tmp_path, monkeypatch, caplog):
    exe, model = _fake_exe(tmp_path, monkeypatch)
    with caplog.at_level(logging.INFO, logger="callo"):
        r = Running(["serve", "--model", str(model), "--mmproj", "/m/mmproj.gguf", "--llama-server", str(exe),
                     "--port", "0", "--cache-dir", str(tmp_path / "c"), "--", "--no-mmap"])
        try:
            assert r.port
            argv = json.loads((tmp_path / "argv.json").read_text())
            assert argv[argv.index("--jinja") + 1:] == ["--slot-save-path", str(tmp_path / "c" / "slots"),
                                                        "--mmproj", "/m/mmproj.gguf", "--no-mmap"]
            _, _, stats = get_json(f"{r.url}/callo/stats")
            assert stats["vision"] is True and stats["media_marker"] == "<__media_fake__>"
            assert stats["user_delimiter"] == DELIM
            assert stats["cache_prompt"] == {"plain": True, "image": True, "context": True, "audio": True,
                                             "video": True, "mixed": True}                  # --cache: on by default
            status, headers, out = post_json(f"{r.url}/v1/systemone", IMG_BODY)
            assert status == 200 and headers["X-Callo-Slot"] == "0" and out["answers"]["q"]["choice"] == "b"
        finally:
            assert r.finish() == 0
    text = caplog.text
    assert "vision: on, media marker <__media_fake__>, mmproj /m/mmproj.gguf" in text
    assert "vision: on (" not in text                                   # no "vision: on (vision: on)" stutter
    assert "vision self-test ok" in text and "self-test ok: 2 + 2" in text
    assert "validated on a one-image S1 prompt" in text


def test_mmproj_passed_after_the_double_dash_also_gives_vision(tmp_path, monkeypatch):
    exe, model = _fake_exe(tmp_path, monkeypatch)
    r = Running(["serve", "--model", str(model), "--llama-server", str(exe), "--port", "0",
                 "--cache-dir", str(tmp_path / "c"), "--", "--mmproj", "/m/mmproj.gguf"])
    try:
        assert r.port and get_json(f"{r.url}/callo/stats")[2]["vision"] is True
        assert json.loads((tmp_path / "argv.json").read_text())[-2:] == ["--mmproj", "/m/mmproj.gguf"]
    finally:
        assert r.finish() == 0


def test_serve_exits_non_zero_and_stops_the_child_when_mmproj_gives_no_vision(tmp_path, monkeypatch, caplog):
    exe, model = _fake_exe(tmp_path, monkeypatch, FAKE_NO_VISION="1")
    with caplog.at_level(logging.ERROR, logger="callo"):
        r = Running(["serve", "--model", str(model), "--mmproj", "/m/mmproj.gguf", "--llama-server", str(exe),
                     "--port", "0", "--cache-dir", str(tmp_path / "c")])
        r.thread.join(timeout=STARTUP_TIMEOUT)
    assert r.rc == 1 and r.port is None
    # the detection note says what /props showed
    assert ("--mmproj was given but the upstream reports no media modality (vision, audio or video): "
            "upstream reports no media modality" in caplog.text)
    assert not list((tmp_path / "c").glob("llama-server-*.pid"))      # the child was stopped


def test_serve_refuses_mmproj_in_attach_mode(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="callo"):
        r = Running(["serve", "--upstream", "http://127.0.0.1:1", "--mmproj", "/m/mm.gguf", "--port", "0",
                     "--cache-dir", str(tmp_path / "c")])
        assert r.finish() == 1
    assert "--mmproj" in caplog.text and "launch mode" in caplog.text


# ---- v0.2: --no-kvu (per-slot KV, deterministic readouts) ------------------------------------------------

UNIFIED_LINE = ("kv: unified (-kvu): readouts on MoE models can shift with other slots' contents; "
                "--no-kvu removes that on text (see README caveats)")


def test_serve_launch_mode_default_is_unified_kv_and_says_so(tmp_path, monkeypatch, caplog):
    exe, model = _fake_exe(tmp_path, monkeypatch)
    with caplog.at_level(logging.INFO, logger="callo"):
        r = Running(["serve", "--model", str(model), "--ctx", "32768", "--llama-server", str(exe), "--port", "0",
                     "--cache-dir", str(tmp_path / "c")])
        try:
            argv = json.loads((tmp_path / "argv.json").read_text())
            assert "-kvu" in argv and argv[argv.index("-c") + 1] == "32768"
            assert get_json(f"{r.url}/callo/stats")[2]["kv"] == "unified"
        finally:
            assert r.finish() == 0
    assert UNIFIED_LINE in caplog.text and "per-slot" not in caplog.text


def test_serve_no_kvu_launches_with_c_times_parallel_and_no_kvu(tmp_path, monkeypatch, caplog):
    exe, model = _fake_exe(tmp_path, monkeypatch)
    with caplog.at_level(logging.INFO, logger="callo"):
        r = Running(["serve", "--model", str(model), "--no-kvu", "--ctx", "32768", "--parallel", "3",
                     "--llama-server", str(exe), "--port", "0", "--cache-dir", str(tmp_path / "c"), "--", "--no-mmap"])
        try:
            argv = json.loads((tmp_path / "argv.json").read_text())
            assert "-kvu" not in argv
            assert argv[argv.index("-c") + 1] == str(32768 * 3) and argv[argv.index("-np") + 1] == "3"
            assert argv[argv.index("--no-cache-idle-slots"):] == ["--no-cache-idle-slots", "-b", "512", "-ub", "512",
                                                                   "--jinja", "--slot-save-path",
                                                                   str(tmp_path / "c" / "slots"), "--no-mmap"]
            assert get_json(f"{r.url}/callo/stats")[2]["kv"] == "per_slot"
        finally:
            assert r.finish() == 0
    assert ("kv: per-slot (no -kvu), -c 98304 = 32768 per slot x 3; text readouts independent of other slots "
            "(see README caveats)" in caplog.text)
    assert "kv: unified" not in caplog.text


def test_no_kvu_without_ctx_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--model", "/m/x.gguf", "--no-kvu", "--llama-server", str(tmp_path / "no-such-binary"),
              "--port", "0", "--cache-dir", str(tmp_path / "c")])
    assert exc.value.code == 2
    assert "--no-kvu needs --ctx: the per-slot context to reserve" in capsys.readouterr().err


@pytest.mark.parametrize("ctx", ["0", "-1"])
def test_no_kvu_with_a_non_positive_ctx_is_a_usage_error(ctx, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--model", "/m/x.gguf", "--no-kvu", "--ctx", ctx, "--llama-server",
              str(tmp_path / "no-such-binary"), "--port", "0", "--cache-dir", str(tmp_path / "c")])
    assert exc.value.code == 2
    assert "--no-kvu needs --ctx >= 1" in capsys.readouterr().err


def test_no_kvu_in_attach_mode_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--upstream", "http://127.0.0.1:1", "--no-kvu", "--ctx", "8192", "--port", "0",
              "--cache-dir", str(tmp_path / "c")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "attach mode does not start llama-server" in err and "without -kvu" in err


def test_stats_kv_is_null_in_attach_mode(tmp_path, caplog):
    up = FakeUpstream().start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        with caplog.at_level(logging.INFO, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
            try:
                assert get_json(f"{r.url}/callo/stats")[2]["kv"] is None
            finally:
                assert r.finish() == 0
        assert "kv: " not in caplog.text
    finally:
        up.stop()


def test_serve_attach_mode_derives_the_delimiter_on_a_text_only_upstream(tmp_path, caplog):
    up = FakeUpstream().start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        with caplog.at_level(logging.INFO, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache", "--cache-dir", str(tmp_path / "c")])
            try:
                _, _, stats = get_json(f"{r.url}/callo/stats")
                assert stats["vision"] is False and stats["user_delimiter"] == DELIM
                assert stats["cache_prompt"] == {"plain": True, "image": True, "context": True, "audio": True,
                                                 "video": True, "mixed": True}
                try:
                    post_json(f"{r.url}/v1/systemone", IMG_BODY)
                    raise AssertionError("expected 501")
                except urllib.error.HTTPError as e:
                    assert e.code == 501
            finally:
                r.finish()
        assert "vision: off" in caplog.text and "text-only context" in caplog.text
        assert "vision self-test" not in caplog.text
        assert not any(isinstance(q.json.get("prompt"), dict) for q in up.requests if q.path == "/completion")
    finally:
        up.stop()


def test_serve_attach_mode_with_a_vision_upstream_and_a_pinned_delimiter(tmp_path, caplog):
    up = FakeUpstream(vision=True).start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        with caplog.at_level(logging.INFO, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--user-delimiter", "<|im_start|>user",
                         "--cache-dir", str(tmp_path / "c")])
            try:
                _, _, stats = get_json(f"{r.url}/callo/stats")
                assert stats["vision"] is True and stats["user_delimiter"] == "<|im_start|>user"
                status, _, _ = post_json(f"{r.url}/v1/systemone", IMG_BODY)
                comp = up.last("/completion").json
                assert status == 200 and comp["message_delimiters"] == [{"role": "user", "delimiter": "<|im_start|>user"}]
            finally:
                r.finish()
        assert "pinned (--user-delimiter)" in caplog.text and "vision self-test ok" in caplog.text
        selftest = [q.json for q in up.requests if q.path == "/completion" and isinstance(q.json.get("prompt"), dict)]
        assert len(selftest) >= 2 and selftest[0]["id_slot"] == 0       # the vision self-test ran twice on slot 0
    finally:
        up.stop()


@pytest.mark.parametrize("flag", [["--user-delimiter", ""], ["--user-delimiter="]])
def test_an_explicitly_empty_user_delimiter_is_a_usage_error_not_a_silent_derivation(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--upstream", "http://127.0.0.1:1", "--port", "0", *flag])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--user-delimiter" in err and "empty" in err


def test_a_failed_vision_self_test_warns_keeps_vision_on_and_callo_still_starts(tmp_path, caplog):
    up = FakeUpstream(vision=True, image_blind=True).start()          # the same answer for every image
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        with caplog.at_level(logging.INFO, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
            try:
                assert r.port and get_json(f"{r.url}/callo/stats")[2]["vision"] is True
            finally:
                assert r.finish() == 0
        assert "vision self-test FAILED" in caplog.text
        assert any(rec.levelno == logging.WARNING and "same answer" in rec.getMessage() for rec in caplog.records)
    finally:
        up.stop()


@pytest.mark.parametrize("flag", [["--mmproj", ""], ["--mmproj="]])
def test_an_explicitly_empty_mmproj_is_a_usage_error(flag, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:                  # a missing binary: nothing can ever be launched here
        main(["serve", "--model", "/m/x.gguf", "--llama-server", str(tmp_path / "no-such-binary"), "--port", "0",
              "--cache-dir", str(tmp_path / "c"), *flag])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--mmproj" in err and "empty" in err


def test_classify_sends_images_as_data_urls_by_extension_and_the_context(tmp_path, capsys, monkeypatch):
    png, jpg = tmp_path / "a.PNG", tmp_path / "b.jpeg"
    png.write_bytes(b"\x89PNG-bytes")
    jpg.write_bytes(b"\xff\xd8jpeg")
    _S1.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _S1)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        assert main(["classify", "--url", url, "--options", "left,right", "--image", str(png), "--image", str(jpg),
                     "--context", "orders: go left"]) == 0
        body = _S1.seen[-1]
        assert body["images"] == ["data:image/png;base64," + base64.b64encode(b"\x89PNG-bytes").decode(),
                                  "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()]
        assert body["context"] == "orders: go left" and body["state"] == ""
        monkeypatch.setattr(sys, "stdin", io.StringIO("s"))
        assert main(["classify", "--url", url, "--options", "a,b"]) == 0
        assert "images" not in _S1.seen[-1] and "context" not in _S1.seen[-1]      # a v0.1-shaped request
    finally:
        srv.shutdown()
        srv.server_close()


def test_classify_refuses_a_bad_image_argument_with_exit_2_before_any_request(tmp_path, capsys, monkeypatch):
    tif = tmp_path / "a.tiff"
    tif.write_bytes(b"x")
    png = tmp_path / "a.png"
    png.write_bytes(b"x")
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    unreachable = "http://127.0.0.1:1"
    assert main(["classify", "--url", unreachable, "--options", "a,b", "--image", str(tif)]) == 2
    assert ".tiff" in capsys.readouterr().err
    assert main(["classify", "--url", unreachable, "--options", "a,b", "--image", str(tmp_path / "missing.png")]) == 2
    assert "missing.png" in capsys.readouterr().err
    assert main(["classify", "--url", unreachable, "--options", "a,b", *(["--image", str(png)] * 9)]) == 2
    assert "8" in capsys.readouterr().err


# ---- v0.3: --position-bias, --slot-save-path, model_key and provenance ----------------------------------------

CHOICE3 = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "x", "b": "y", "c": "z"}}}}


def _profile_file(tmp_path, **kw):
    f = tmp_path / "bias.json"
    f.write_text(json.dumps(profile_dict(**kw)))
    return f


@pytest.mark.parametrize("content", ["{not json", json.dumps({**profile_dict(), "callo_position_bias": 1}),
                                     json.dumps({**profile_dict(), "entries": {"3": {"prior": [0.9, 0.9, 0.9]}}}),
                                     json.dumps({**profile_dict(), "set": {**profile_dict()["set"], "builtin": True}})])
def test_a_bad_profile_is_refused_before_any_llama_server_starts(tmp_path, monkeypatch, caplog, content):
    """The last case: a profile fitted on the built-in set, which is a diagnostic only, is
    refused by `callo serve` itself, not only by load_profile."""
    exe, model = _fake_exe(tmp_path, monkeypatch)
    bad = tmp_path / "bad.json"
    bad.write_text(content)
    with caplog.at_level(logging.ERROR, logger="callo"):
        r = Running(["serve", "--model", str(model), "--llama-server", str(exe), "--port", "0",
                     "--cache-dir", str(tmp_path / "c"), "--position-bias", str(bad)])
        assert r.finish() == 1
    assert not (tmp_path / "argv.json").exists() and "--position-bias" in caplog.text     # no child was spawned
    if '"builtin": true' in content:
        assert "built-in set, which is a diagnostic only" in caplog.text


def test_a_missing_profile_file_is_exit_1(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="callo"):
        r = Running(["serve", "--upstream", "http://127.0.0.1:1", "--port", "0", "--cache-dir", str(tmp_path / "c"),
                     "--position-bias", str(tmp_path / "nope.json")])
        assert r.finish() == 1
    assert "cannot read" in caplog.text


def test_a_profile_from_another_model_is_refused_after_the_gguf_is_read(tmp_path, caplog):
    up = FakeUpstream().start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        with caplog.at_level(logging.ERROR, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"),
                         "--position-bias", str(_profile_file(tmp_path, name="Other-9B.gguf"))])
            assert r.finish() == 1
        assert "this profile was measured on Other-9B.gguf" in caplog.text
        # refused right after the key is computed, before the idle-clear probe or the delimiter derivation
        # touch the upstream: no probe completion, no slot action, no tokenize/apply-template call
        touched = [r.path for r in up.requests if r.path.split("?")[0] not in ("/health", "/props")]
        assert touched == [] and up.erased == []
    finally:
        up.stop()


def test_a_profile_on_a_native_decision_model_is_refused(tmp_path, caplog):
    up = FakeUpstream(slot_save=True).start()
    try:
        g = tiny_gguf(tmp_path / "kev.gguf", decision="kev")
        up.props["model_path"] = str(g)
        with caplog.at_level(logging.ERROR, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"),
                         "--position-bias", str(_profile_file(tmp_path, key=load_model_key(g, None)))])
            assert r.finish() == 1
        assert "native decision model" in caplog.text
        # like a wrong-model profile, refused before the startup probe touches a slot
        touched = [r.path for r in up.requests if r.path.split("?")[0] not in ("/health", "/props")]
        assert touched == [] and up.erased == []
    finally:
        up.stop()


def test_a_matching_profile_is_applied_counted_and_published(tmp_path, caplog):
    up = FakeUpstream().start()
    try:
        g = tiny_gguf(tmp_path / "m.gguf")
        up.props["model_path"] = str(g)
        prof = _profile_file(tmp_path, key=load_model_key(g, None), entries={3: [0.6, 0.3, 0.1]})
        with caplog.at_level(logging.INFO, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"),
                         "--position-bias", str(prof)])
            try:
                status, _, out = post_json(f"{r.url}/v1/systemone", CHOICE3)
                assert status == 200 and set(out["answers"]["q"]["probabilities"]) == {"a", "b", "c"}
                post_json(f"{r.url}/v1/systemone", BODY)                         # n = 2: no entry
                _, _, stats = get_json(f"{r.url}/callo/stats")
            finally:
                assert r.finish() == 0
        pb = stats["position_bias"]
        assert pb["profile"] == str(prof) and pb["entries"] == [3] and pb["kinds"] == ["plain"]
        assert pb["applied"] == 1 and pb["skipped"]["no_entry"] == 1 and pb["state"] == "clean"
        assert stats["model_key"] == load_model_key(g, None) and stats["model_file"] == "m.gguf"
        assert stats["provenance"]["prompt_sha256"] == prompt_sha256()
        assert stats["provenance"]["build_info"] == "b1-926862e" and stats["provenance"]["cache_prompt"] is True
        assert stats["provenance"]["server_args"] is None and stats["model_key_error"] is None    # attach mode
        assert stats["s1"]["readouts"] == 2
        assert "position bias:" in caplog.text and "self-test" in caplog.text
        assert "user_delimiter differs" in caplog.text                        # profile_dict's provenance has None
    finally:
        up.stop()


def test_serve_without_a_profile_publishes_null_and_the_model_key(tmp_path):
    up = FakeUpstream().start()
    try:
        g = tiny_gguf(tmp_path / "m.gguf")
        up.props["model_path"] = str(g)
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        try:
            _, _, stats = get_json(f"{r.url}/callo/stats")
        finally:
            assert r.finish() == 0
        assert stats["position_bias"] is None and stats["model_key"] == load_model_key(g, None)
        assert stats["probe"]["erase"] == "unsupported" and stats["probe"]["residue"] == [3, 0]
        assert list((tmp_path / "c").glob("modelkey-*.json"))                  # cached next to the vocabulary
    finally:
        up.stop()


def test_attach_mode_with_slot_save_erases_the_probe_residue(tmp_path, caplog):
    up = FakeUpstream(slot_save=True).start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        with caplog.at_level(logging.WARNING, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
            try:
                _, _, stats = get_json(f"{r.url}/callo/stats")
            finally:
                assert r.finish() == 0
        assert stats["probe"]["erase"] == "ok" and stats["probe"]["residue"] == [] and up.erased == [3, 0]
        assert "residue" not in caplog.text
    finally:
        up.stop()


def test_attach_mode_without_slot_save_warns_about_the_residue(tmp_path, caplog):
    up = FakeUpstream().start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        with caplog.at_level(logging.WARNING, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
            assert r.finish() == 0
        (w,) = [m for m in caplog.messages if "startup-probe residue" in m]
        assert "Restart llama-server with --slot-save-path DIR" in w
        assert "passthrough" not in w
    finally:
        up.stop()


@pytest.mark.parametrize(("status", "erase", "says"), [
    ("501", "unsupported", "a passthrough flag after -- overrode it"),
    ("500", "failed", "The erase itself failed"),
])
def test_launch_mode_residue_warning_names_why_the_erase_did_not_happen(tmp_path, monkeypatch, caplog, status, erase,
                                                                        says):
    """Launch mode already passed --slot-save-path, so advising it would not help."""
    exe, model = _fake_exe(tmp_path, monkeypatch, FAKE_ERASE_STATUS=status)
    with caplog.at_level(logging.WARNING, logger="callo"):
        r = Running(["serve", "--model", str(model), "--llama-server", str(exe), "--port", "0",
                     "--cache-dir", str(tmp_path / "c")])
        try:
            stats = get_json(f"{r.url}/callo/stats")[2]
        finally:
            assert r.finish() == 0
    assert stats["probe"]["erase"] == erase and stats["probe"]["residue"] == [3, 0]
    (w,) = [m for m in caplog.messages if "startup-probe residue" in m]
    assert says in w and "restart llama-server with --slot-save-path" not in w.lower()


def test_attach_mode_residue_after_a_failed_erase_does_not_advise_a_flag_it_already_has():
    from callo.upstream import ProbeResult
    w = cli_module.residue_warning(ProbeResult(False, "d", 16, (3,), "failed"), launch=False)
    assert "the erase itself failed" in w.lower() and "restart llama-server" in w and "--slot-save-path" not in w


def test_launch_mode_creates_the_slot_save_directory_erases_and_records_the_passthrough(tmp_path, monkeypatch):
    exe, model = _fake_exe(tmp_path, monkeypatch)
    r = Running(["serve", "--model", str(model), "--llama-server", str(exe), "--port", "0",
                 "--cache-dir", str(tmp_path / "c"), "--", "--no-mmap"])
    try:
        assert r.port and (tmp_path / "c" / "slots").is_dir()
        stats = get_json(f"{r.url}/callo/stats")[2]
        assert stats["probe"]["erase"] == "ok" and stats["provenance"]["server_args"] == ["--no-mmap"]
    finally:
        assert r.finish() == 0


@pytest.mark.parametrize("exc", [InvalidInput, OSError, ValueError, TypeError])
def test_a_model_key_failure_without_a_profile_warns_and_with_one_exits_1(tmp_path, caplog, monkeypatch, exc):
    """Computing model_key must not become a new way for a v0.2 user to fail at startup, and
    that holds for a stray ValueError/TypeError too, not only for CalloError/OSError."""
    def boom(path, cache_dir):
        raise exc("cannot hash this GGUF")

    monkeypatch.setattr(cli_module, "load_model_key", boom)
    up = FakeUpstream().start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        with caplog.at_level(logging.WARNING, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
            try:
                _, _, stats = get_json(f"{r.url}/callo/stats")
            finally:
                assert r.finish() == 0
        assert stats["model_key"] is None and "cannot hash" in stats["model_key_error"]
        assert "cannot compute model_key" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"),
                         "--position-bias", str(_profile_file(tmp_path))])
            assert r.finish() == 1
        assert "cannot compute the served model's model_key" in caplog.text and "cannot hash" in caplog.text
    finally:
        up.stop()


# ---- v0.3: temperature mode next to --bias --------------------------------------------------------------------

def test_temperature_mode_without_data_or_with_a_bias_only_flag_is_a_usage_error(capsys):
    assert main(["calibrate", "--url", "http://127.0.0.1:1"]) == 2
    assert "--data" in capsys.readouterr().err
    for flag in (["--resume"], ["--allow-busy"], ["--raw", "r.jsonl"], ["--seed", "1"], ["--max-per-n", "3"],
                 ["--min-questions", "3"], ["--seed", "0"], ["--max-per-n", "0"], ["--min-questions", "0"]):
        assert main(["calibrate", "--url", "http://127.0.0.1:1", "--data", "x.jsonl", *flag]) == 2
        assert "only with --bias" in capsys.readouterr().err


def test_temperature_mode_records_the_loaded_profiles_sha256(tmp_path, capsys):
    data = tmp_path / "labeled.jsonl"
    data.write_text(json.dumps({"state": "[0,2,0,0]", "questions": {"q": {"type": "choice", "label": "b",
                                                                          "criteria": {k: None for k in "abcd"}}}})
                    + "\n")
    _Tempered.stats_extra = {"position_bias": {"sha256": "ab" * 32}}
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "T.json"
        assert main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data),
                     "--out", str(out)]) == 0
        assert json.loads(out.read_text())["position_bias_sha256"] == "ab" * 32
        assert "a refitted profile invalidates this T" in capsys.readouterr().out
        _Tempered.stats_extra = {}
        assert main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data),
                     "--out", str(out)]) == 0
        assert json.loads(out.read_text())["position_bias_sha256"] is None
    finally:
        srv.shutdown()
        srv.server_close()
        _Tempered.stats_extra = {}


def test_temperature_mode_names_a_malformed_line(tmp_path, capsys):
    data = tmp_path / "labeled.jsonl"
    data.write_text(json.dumps({"state": "[0,0,0,0]", "questions": {"q": {"type": "choice", "label": "a",
                                                                          "criteria": {k: None for k in "abcd"}}}})
                    + "\n\n[1, 2]\n")
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data)]) == 1
    finally:
        srv.shutdown()
        srv.server_close()
    assert "--data line 3: not a JSON object" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [{"state": "s"}, {"state": "s", "questions": ["q"]},
                                 {"state": "s", "questions": {"q": "choice"}}, {"questions": {}}])
def test_temperature_mode_names_a_line_that_is_not_a_system_one_request(tmp_path, capsys, bad):
    """A JSON object that is not a request is exit 1 with its line number, not a
    traceback."""
    data = tmp_path / "labeled.jsonl"
    data.write_text(json.dumps({"state": "[0,0,0,0]", "questions": {"q": {"type": "choice", "label": "a",
                                                                          "criteria": {k: None for k in "abcd"}}}})
                    + "\n" + json.dumps(bad) + "\n")
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data)]) == 1
    finally:
        srv.shutdown()
        srv.server_close()
    err = capsys.readouterr().err
    assert "--data line 2: not a System One request" in err and "Traceback" not in err


@pytest.mark.parametrize("meta", ["x", None, [1]])
def test_temperature_mode_tolerates_a_non_dict_meta_as_it_does_for_the_variant(tmp_path, capsys, meta):
    """A non-dict _meta reads as a clean record with no id (as the variant check treats it), never a
    traceback after the POST."""
    data = tmp_path / "labeled.jsonl"
    data.write_text(json.dumps({"state": "[0,2,0,0]", "_meta": meta,
                                "questions": {"q": {"type": "choice", "label": "b",
                                                    "criteria": {k: None for k in "abcd"}}}}) + "\n")
    _Tempered.posts = 0
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data)]) == 0
    finally:
        srv.shutdown()
        srv.server_close()
    assert _Tempered.posts == 1 and "Traceback" not in capsys.readouterr().err


# ---- temperature mode: errors, line splitting, non-clean variants ----------------------------------------------

@pytest.mark.parametrize("content", [None, b"\xff\xfe not utf-8\n"])
def test_temperature_mode_names_a_missing_or_non_utf8_data_without_a_traceback(tmp_path, capsys, content):
    """Exit 1 with a `callo:` line, worded as --bias mode words it."""
    data = tmp_path / "labeled.jsonl"
    if content is not None:
        data.write_bytes(content)
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data)]) == 1
    finally:
        srv.shutdown()
        srv.server_close()
    err = capsys.readouterr().err
    want = f"callo: --data {data}: " if content is None else "callo: --data labeled.jsonl: not UTF-8 ("
    assert want in err and "Traceback" not in err


def test_temperature_mode_splits_data_lines_on_newline_only(tmp_path, capsys):
    """A JSON string may hold a raw U+2028, U+2029 or U+0085; the reader
    splits on "\n" only, so such a record is one line and the line numbers stay right."""
    data = tmp_path / "labeled.jsonl"
    data.write_text(json.dumps({"state": "a\u2028b\u2029c\x85d", "questions": {}, "_meta": {"variant": "para"}},
                               ensure_ascii=False) + "\n{not json\n", encoding="utf-8")
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data)]) == 1
    finally:
        srv.shutdown()
        srv.server_close()
    err = capsys.readouterr().err
    assert "callo: --data line 2: not valid JSON" in err and "--data line 1" not in err


def test_temperature_mode_skips_a_non_clean_record_before_building_its_request(tmp_path, capsys):
    """v0.2 skipped a non-clean variant before anything else, so a structurally broken
    non-clean record is skipped silently, never a line-numbered exit 1."""
    data = tmp_path / "labeled.jsonl"
    good = {"state": "[0,2,0,0]", "questions": {"q": {"type": "choice", "label": "b",
                                                      "criteria": {k: None for k in "abcd"}}}}
    data.write_text(json.dumps({"state": "s", "_meta": {"variant": "noisy"}}) + "\n" + json.dumps(good) + "\n")
    _Tempered.posts = 0
    srv = HTTPServer(("127.0.0.1", 0), _Tempered)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data)]) == 0
    finally:
        srv.shutdown()
        srv.server_close()
    assert _Tempered.posts == 1 and "not a System One request" not in capsys.readouterr().err


# ---- v0.3: audio and video at startup ----------------------------------------------------------------------------

AV_UP = {"vision": True, "audio": True, "video": True}
AUDIO_REQ = {"state": "", "audio": [TONE_URL], "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}}
VIDEO_REQ = {"state": "", "videos": [RED_URL], "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}}


def _serve_av(tmp_path, caplog, **fake):
    up = FakeUpstream(**fake).start()
    up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
    with caplog.at_level(logging.INFO, logger="callo"):
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
    return up, r


def test_serve_runs_the_self_tests_in_order_and_publishes_them(tmp_path, caplog):
    up, r = _serve_av(tmp_path, caplog, **AV_UP)
    try:
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert (stats["vision"], stats["audio"], stats["video"]) == (True, True, True)
        assert stats["self_test"] == {"text": True, "vision": True, "audio": True, "video": True, "think": None}
        status, headers, out = post_json(f"{r.url}/v1/systemone", AUDIO_REQ)
        assert status == 200 and json.loads(headers["X-Callo-Timings"])[0]["kind"] == "audio"
        assert post_json(f"{r.url}/v1/systemone", VIDEO_REQ)[0] == 200
    finally:
        assert r.finish() == 0
        up.stop()
    text = caplog.text
    order = [text.index(s) for s in ("vision self-test ok", "self-test ok: 2 + 2", "audio self-test ok",
                                     "video self-test ok")]
    assert order == sorted(order)                       # v0.2's vision-then-text, audio and video appended
    assert "audio: on, video: on" in text and "Video is OFF" not in text


def test_a_video_self_test_that_cannot_load_its_clip_turns_video_off_and_logs_an_error(tmp_path, caplog):
    up, r = _serve_av(tmp_path, caplog, video_fails=True, **AV_UP)        # e.g. no ffmpeg on llama-server's PATH
    try:
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["video"] is False and stats["audio"] is True
        assert stats["self_test"] == {"text": True, "vision": True, "audio": True, "video": False, "think": None}
        # video is off
        assert stats["cache_prompt"]["video"] is None and stats["cache_prompt"]["audio"] is True
        with pytest.raises(urllib.error.HTTPError) as e:
            post_json(f"{r.url}/v1/systemone", VIDEO_REQ)
        detail = json.loads(e.value.read())["detail"]
        e.value.close()
        # a client can tell "callo turned video off at startup" from "this build has no video"
        assert e.value.code == 501 and detail == VIDEO_OFF_DETAIL and detail != NO_VIDEO_DETAIL
        assert "turned off at startup" in detail and "failed to load" in detail
        assert post_json(f"{r.url}/v1/systemone", AUDIO_REQ)[0] == 200
    finally:
        assert r.finish() == 0
        up.stop()
    errors = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.ERROR]
    assert any("could not load the built-in AVI" in m and "Video is OFF" in m for m in errors)


def test_an_audio_self_test_that_cannot_load_its_clip_only_warns(tmp_path, caplog):
    """Only the VIDEO self-test's load_failed turns a modality off. An audio "Failed
    to load" (the fake's `audio_fails`, end to end) is a warning: audio stays on, so an audio request still
    reaches the upstream and gets its 422, never a 501; video is untouched."""
    up, r = _serve_av(tmp_path, caplog, audio_fails=True, **AV_UP)
    try:
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert (stats["audio"], stats["video"]) == (True, True)
        assert stats["self_test"] == {"text": True, "vision": True, "audio": False, "video": True, "think": None}
        assert stats["cache_prompt"]["audio"] is True
        with pytest.raises(urllib.error.HTTPError) as e:
            post_json(f"{r.url}/v1/systemone", AUDIO_REQ)
        e.value.close()
        assert e.value.code == 422
        assert post_json(f"{r.url}/v1/systemone", VIDEO_REQ)[0] == 200
    finally:
        assert r.finish() == 0
        up.stop()
    assert "audio self-test could not run: MediaLoadFailed" in caplog.text
    assert "OFF" not in caplog.text and not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


@pytest.mark.parametrize("fake", [{"image_blind": True}, {}])
def test_a_wrong_answer_or_an_unavailable_upstream_warns_and_keeps_audio_and_video_on(tmp_path, caplog, fake):
    """Only "Failed to load" turns video off; a wrong answer (a media-blind model) or a 503 only warns."""
    up = FakeUpstream(**AV_UP, **fake)
    if not fake:
        up.completion_error = (503, "Loading model")
    up.start()
    up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
    with caplog.at_level(logging.INFO, logger="callo"):
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
    try:
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["audio"] is True and stats["video"] is True
        assert stats["self_test"]["audio"] is False and stats["self_test"]["video"] is False
    finally:
        assert r.finish() == 0
        up.stop()
    assert "Video is OFF" not in caplog.text
    if fake:
        assert "audio self-test FAILED" in caplog.text and "video self-test FAILED" in caplog.text
    else:
        assert "video self-test could not run: UpstreamUnavailable" in caplog.text


def test_a_text_only_upstream_runs_only_the_text_self_test(tmp_path, caplog):
    up, r = _serve_av(tmp_path, caplog)
    try:
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["self_test"] == {"text": True, "vision": None, "audio": None, "video": None, "think": None}
        assert (stats["audio"], stats["video"]) == (False, False)
    finally:
        assert r.finish() == 0
        up.stop()
    assert "audio self-test" not in caplog.text and "video self-test" not in caplog.text


def test_launch_mode_accepts_an_audio_only_mmproj(tmp_path, monkeypatch, caplog):
    """An audio-only projector (Ultravox, Voxtral, Qwen3-ASR) reports no vision; launch mode must not exit 1."""
    exe, model = _fake_exe(tmp_path, monkeypatch, FAKE_AUDIO_ONLY="1")
    with caplog.at_level(logging.INFO, logger="callo"):
        r = Running(["serve", "--model", str(model), "--mmproj", "/m/voxtral-mmproj.gguf", "--llama-server", str(exe),
                     "--port", "0", "--cache-dir", str(tmp_path / "c")])
        try:
            assert r.port
            _, _, stats = get_json(f"{r.url}/callo/stats")
            assert (stats["vision"], stats["audio"], stats["video"]) == (False, True, False)
            assert stats["self_test"]["audio"] is not None and stats["self_test"]["vision"] is None
            status, headers, _ = post_json(f"{r.url}/v1/systemone", AUDIO_REQ)          # it serves audio
            assert status == 200 and json.loads(headers["X-Callo-Timings"])[0]["kind"] == "audio"
        finally:
            assert r.finish() == 0
    assert "audio: on, video: off" in caplog.text and "text-only context" in caplog.text   # text validation


def test_the_native_model_startup_line_says_audio_and_video_are_refused(tmp_path, caplog):
    """Since v0.3 a native decision model's /v1/systemone is not a plain passthrough: audio and video get 422."""
    up = FakeUpstream().start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "kev.gguf", decision="kev"))
        with caplog.at_level(logging.INFO, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
            assert r.finish() == 0
    finally:
        up.stop()
    line = next(rec.getMessage() for rec in caplog.records if "native decision model" in rec.getMessage())
    assert "refuses audio and video with 422" in line and "straight through" not in line



# ---- v0.3: callo classify --audio / --video -----------------------------------------------------------------------

def _s1_server():
    _S1.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _S1)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_the_extension_tables_are_fixed():
    assert cli_module.AUDIO_TYPES == {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}
    assert cli_module.VIDEO_TYPES == {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
                                      ".mov": "video/quicktime", ".avi": "video/x-msvideo"}
    assert cli_module.CLASSIFY_MAX_BODY == 64 * 1024 * 1024


def test_classify_sends_audio_and_video_as_data_urls_by_extension(tmp_path, monkeypatch):
    wav, mkv, avi, png = tmp_path / "a.WAV", tmp_path / "b.mkv", tmp_path / "c.avi", tmp_path / "d.png"
    for f, data in ((wav, b"RIFF-wav"), (mkv, b"\x1a\x45\xdf\xa3"), (avi, b"RIFF-avi"), (png, b"\x89PNG")):
        f.write_bytes(data)
    srv, url = _s1_server()
    try:
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        assert main(["classify", "--url", url, "--options", "a,b", "--video", str(mkv), "--audio", str(wav),
                     "--video", str(avi), "--image", str(png)]) == 0
        body = _S1.seen[-1]
        assert body["audio"] == ["data:audio/wav;base64," + base64.b64encode(b"RIFF-wav").decode()]
        assert body["videos"] == ["data:video/x-matroska;base64," + base64.b64encode(b"\x1a\x45\xdf\xa3").decode(),
                                  "data:video/x-msvideo;base64," + base64.b64encode(b"RIFF-avi").decode()]
        assert body["images"] == ["data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()]
        monkeypatch.setattr(sys, "stdin", io.StringIO("s"))
        assert main(["classify", "--url", url, "--options", "a,b", "--audio", str(wav)]) == 0
        assert "videos" not in _S1.seen[-1] and "images" not in _S1.seen[-1]     # only the fields given
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize(("flags", "said"), [
    (["--audio", "a.ogg"], ".ogg; use one of .wav, .mp3, .flac"),
    (["--video", "a.gif"], ".gif; use one of .mp4, .webm, .mkv, .mov, .avi"),
    (["--audio", "missing.wav"], "--audio missing.wav"),
    (["--audio", "a.wav"] * 5 + ["--video", "a.mp4"] * 4, "at most 8 media files (--image, --audio and --video "
                                                         "together), got 9"),
])
def test_classify_refuses_a_bad_audio_or_video_argument_with_exit_2_before_any_request(tmp_path, capsys, monkeypatch,
                                                                                       flags, said):
    for name in ("a.ogg", "a.gif", "a.wav", "a.mp4"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["classify", "--url", "http://127.0.0.1:1", "--options", "a,b", *flags]) == 2
    assert said in capsys.readouterr().err


def test_classify_pre_checks_the_encoded_size_before_reading_anything(tmp_path, capsys, monkeypatch):
    big = tmp_path / "big.mp4"
    big.write_bytes(b"\x00" * 3000)                       # 4,000 base64 characters plus the prefix
    monkeypatch.setattr(cli_module, "CLASSIFY_MAX_BODY", 4000)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail(f"read {self} before the size check"))
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["classify", "--url", "http://127.0.0.1:1", "--options", "a,b", "--video", str(big)]) == 2
    err = capsys.readouterr().err
    assert "the media would encode to 4022 bytes, over the 4000-byte (64 MiB) body limit" in err


def test_classify_refuses_a_body_that_the_state_pushes_over_the_limit(tmp_path, capsys, monkeypatch):
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"RIFF")
    monkeypatch.setattr(cli_module, "CLASSIFY_MAX_BODY", 500)
    monkeypatch.setattr(sys, "stdin", io.StringIO("s" * 600))
    assert main(["classify", "--url", "http://127.0.0.1:1", "--options", "a,b", "--audio", str(clip)]) == 2
    err = capsys.readouterr().err
    assert "the request body would be" in err and "the state from stdin" not in err      # no cause assumed
    # the fix hint
    assert "send fewer or smaller media files, or a shorter state (stdin) or --context" in err


def test_classify_accepts_exactly_8_media_files(tmp_path, monkeypatch):
    for name in ("a.png", "b.wav", "c.mp4"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    srv, url = _s1_server()
    try:
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        flags = ["--image", "a.png"] * 3 + ["--audio", "b.wav"] * 3 + ["--video", "c.mp4"] * 2
        assert main(["classify", "--url", url, "--options", "a,b", *flags]) == 0
        body = _S1.seen[-1]
        assert (len(body["images"]), len(body["audio"]), len(body["videos"])) == (3, 3, 2)
    finally:
        srv.shutdown()
        srv.server_close()


def test_classify_accepts_media_and_a_body_of_exactly_the_limit(tmp_path, monkeypatch):
    """Both checks refuse only ABOVE the cap: media that encode to exactly it pass the pre-check, and a body
    of exactly it is sent."""
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"\x00" * 3000)                     # 22 + 4,000 = 4,022 encoded characters
    args = build_parser().parse_args(["classify", "--options", "a", "--video", str(clip)])
    monkeypatch.setattr(cli_module, "CLASSIFY_MAX_BODY", 4022)
    assert isinstance(cli_module._media_fields(args), dict)
    monkeypatch.setattr(cli_module, "CLASSIFY_MAX_BODY", 4021)
    assert isinstance(cli_module._media_fields(args), str)
    srv, url = _s1_server()
    try:
        monkeypatch.setattr(cli_module, "CLASSIFY_MAX_BODY", 10**9)
        monkeypatch.setattr(sys, "stdin", io.StringIO("s"))
        assert main(["classify", "--url", url, "--options", "a,b", "--video", str(clip)]) == 0
        size = len(json.dumps(_S1.seen[-1]).encode())
        for limit, rc in ((size, 0), (size - 1, 2)):
            monkeypatch.setattr(cli_module, "CLASSIFY_MAX_BODY", limit)
            monkeypatch.setattr(sys, "stdin", io.StringIO("s"))
            assert main(["classify", "--url", url, "--options", "a,b", "--video", str(clip)]) == rc
    finally:
        srv.shutdown()
        srv.server_close()


def test_every_classify_media_flag_names_the_8_media_cap():
    sub = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    flags = ("image", "audio", "video")
    helps = {a.dest: a.help for a in sub.choices["classify"]._actions if a.dest in flags}
    assert len(helps) == 3 and all("at most 8 media files in all" in h for h in helps.values())
