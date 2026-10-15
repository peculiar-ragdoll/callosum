"""The GPU-free parts of the v0.2 acceptance bench, plus a smoke run of every main-instance row against callo in attach
mode on the fake vision upstream (short paced windows), so a bench bug shows here and not on the GPU."""
import hashlib
import json
import logging
import math
import os
import signal
import string
import struct
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

import pytest
from fake_upstream import FakeUpstream

from bench import acceptance_v02 as v02
from bench import lib
from bench.frames import DIRS, GRID, PAC, PAC_TILE, fresh_frames, ghost_tile, maze_frame, spike_frames
from bench.report_v02 import render
from callo.cli import build_parser, serve, split_passthrough

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
TOKENS = list(string.ascii_uppercase) + ["Ġ" + c for c in string.ascii_uppercase] + ["**", "Ġthe", "x"]


def _rgb(png: bytes) -> tuple[int, int, bytes]:
    w, h = struct.unpack(">II", png[16:24])
    i, idat = 8, b""
    while i < len(png):
        n = struct.unpack(">I", png[i:i + 4])[0]
        if png[i + 4:i + 8] == b"IDAT":
            idat += png[i + 8:i + 8 + n]
        i += 12 + n
    raw = zlib.decompress(idat)
    return w, h, b"".join(raw[y * (1 + 3 * w) + 1:(y + 1) * (1 + 3 * w)] for y in range(h))


def _at(png: bytes, x: int, y: int) -> tuple[int, ...]:
    w, _, rgb = _rgb(png)
    return tuple(rgb[(y * w + x) * 3:(y * w + x) * 3 + 3])


def test_maze_frames_are_deterministic_distinct_and_draw_pacman_and_the_ghost():
    a, b = maze_frame(224, "up", 3, salt=5), maze_frame(224, "up", 3, salt=5)
    assert a == b and _rgb(a)[:2] == (224, 224)
    frames = fresh_frames(40)
    assert len({hashlib.sha256(f).hexdigest() for f in frames}) == 40
    tile = 224 // GRID
    pad = (224 - tile * GRID) // 2
    centre = (pad + PAC_TILE[0] * tile + tile // 2, pad + PAC_TILE[1] * tile + tile // 2)
    assert _at(a, *centre) == PAC
    gx, gy = ghost_tile("up", 3)
    assert _at(a, pad + gx * tile + tile // 2, pad + gy * tile + tile * 3 // 4) == (255, 0, 0)
    assert _rgb(maze_frame(448, "left", 2))[:2] == (448, 448) and DIRS == ("left", "right", "up", "down")


def test_the_spike_frames_are_committed_with_their_checksums():
    frames = spike_frames()
    assert len(frames) == 36
    assert sorted({(size, sum(1 for f in frames if f[2] == size)) for _, _, size, _ in frames}) == \
        [(224, 12), (336, 12), (448, 12)]
    sums = dict(line.split()[::-1] for line in (Path(frames[0][3]).parent / "SHA256SUMS").read_text().splitlines())
    for gold, dist, _, path in frames:
        assert gold in DIRS and dist in (2, 3, 4)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sums[path.name]


def test_gpu_server_pids_sees_llama_cpp_omlx_and_mlx_servers_only():
    ps = ("  101 /opt/llama.cpp/build/bin/llama-server -m x.gguf --port 0\n"
          "  102 llama-server -m y.gguf\n"
          "  103 /usr/bin/python3 /tmp/llama-server-wrapper.py\n"
          "  104 grep llama-server\n"
          "  105 /opt/homebrew/bin/llama-server-old --help\n"
          "  106 /Users/me/.venv/bin/python3 /Users/me/.venv/bin/omlx start\n"
          "  107 python -m mlx_lm.server --model m\n"
          "  108 /opt/homebrew/bin/llama-cli -m z.gguf\n"
          "  109 grep -E [o]mlx|[m]lx_lm\n"
          "  110 uv run python -m bench.acceptance_v02 --out x\n"
          "  111 /usr/bin/python3 harness/omlx_ctl.py status\n"
          "  112 tail -f /Users/me/omlx.log\n"
          "  113 /x/build/bin/llama-imatrix -m m.gguf -f corpus.txt\n"
          "  114 vim omlx_notes.md\n"
          "  115 python3.13 -m mlx_vlm.server --model v\n"
          "garbage line\n")
    assert [pid for pid, _ in lib.gpu_server_pids(ps, resident_bytes=lambda pid: 3 * GIB)] == \
        [101, 102, 106, 107, 108, 113, 115]


def test_omlx_mlx_and_ollama_block_only_while_they_hold_a_model():
    """Real `ps -axo pid=,args=` lines from this box: the oMLX menu-bar app and an idle ollama are always running."""
    idle = (" 5922 /Applications/oMLX.app/Contents/MacOS/oMLX\n"
            "  791 /opt/homebrew/opt/ollama/bin/ollama serve\n")
    resident = {5922: 88160 * 1024, 791: 71136 * 1024}             # their RSS on 2026-10-03
    assert lib.gpu_server_pids(idle, resident_bytes=resident.get) == []
    loaded = idle + (" 201 /Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3 -m omlx.cli serve "
                     "--port 8124\n"
                     "  902 /opt/homebrew/opt/ollama/bin/ollama runner --model /x/blobs/sha256-ab --port 50123\n")
    resident.update({201: 21 * GIB, 902: 19 * GIB})
    busy = lib.gpu_server_pids(loaded, resident_bytes=resident.get)
    assert [pid for pid, _ in busy] == [201, 902] and busy[0][1].endswith("[resident 21.0 GiB]")
    resident[5922] = 2 * GIB - 1                                     # the threshold: 2 GiB blocks, one byte less not
    assert [pid for pid, _ in lib.gpu_server_pids(idle, resident_bytes=resident.get)] == []
    resident[5922] = 2 * GIB
    assert [pid for pid, _ in lib.gpu_server_pids(idle, resident_bytes=resident.get)] == [5922]
    assert [pid for pid, _ in lib.gpu_server_pids(idle, resident_bytes=lambda pid: None)] == [5922, 791]  # unknown
    measured = []
    llama = " 7 /x/build/bin/llama-server -m m.gguf --port 0\n"
    assert lib.gpu_server_pids(llama, resident_bytes=lambda pid: measured.append(pid) or 0) == \
        [(7, "/x/build/bin/llama-server -m m.gguf --port 0")] and measured == []   # llama.cpp always blocks


@pytest.mark.parametrize("cmd", [
    "/opt/homebrew/Cellar/python@3.13/3.13.1/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/"
    "MacOS/Python -m mlx_lm.server --model m",
    "python3 -u -m mlx_lm.server --model m",
    "python3 -X utf8 -m mlx_lm.server --model m",
    "python3 -Xfrozen_modules=off -m mlx_vlm.server --model v",
    "python3 -O -B -m mlx_vlm.server --model v",
    "python3.12 -m mlx_lm server --model m",
    "python -W ignore -m mlx_vlm server --model v",
    "/Users/me/.venv/bin/python3 /Users/me/.venv/bin/mlx_lm.server --model m",
    "/Users/me/.venv/bin/mlx_lm.server --model m",
    "/opt/homebrew/bin/omlx serve",
    "/bin/sh /opt/homebrew/bin/omlx start",
    "/Applications/oMLX.app/Contents/MacOS/omlx-cli serve --port 8124",
    "/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3 -m omlx.cli serve --port 8124",
    "/Applications/oMLX.app/Contents/MacOS/oMLX",
    "/opt/homebrew/opt/ollama/bin/ollama serve",
    "/opt/homebrew/opt/ollama/bin/ollama runner --model /x/blobs/sha256-ab --port 50123",
])
def test_every_omlx_mlx_and_ollama_launch_form_is_seen(cmd):
    assert lib.gpu_class(cmd.split()) == "resident"
    assert lib.gpu_server_pids(f" 9 {cmd}\n", resident_bytes=lambda pid: 3 * GIB) == [(9, f"{cmd} [resident 3.0 GiB]")]


@pytest.mark.parametrize("cmd", ["ollama run qwen3", "ollama list", "python3 -c import mlx_lm", "python3 -m pytest -q",
                                 "python3 harness/omlx_ctl.py status", "tail -f /Users/me/omlx.log",
                                 "/bin/zsh -c omlx start", "uv run python -m mlx_lm.server", "vim mlx_lm.server",
                                 "python3 -m bench.acceptance_v02 --out x --rows mlx_lm"])
def test_clients_files_and_wrappers_of_other_programs_are_not_gpu_servers(cmd):
    assert lib.gpu_class(cmd.split()) is None


def test_a_llama_cpp_named_script_run_by_python_is_a_test_fake_not_a_gpu_server():
    """The suite's fake llama-server is a Python script: `make check` on the box must not abort a live GPU run's
    watchdog. Real llama.cpp programs are native binaries; a python-run MLX server still counts."""
    fake = ("/Users/me/callosum/.venv/bin/python /private/var/folders/xy/T/pytest-of-me/pytest-12/"
            "test_x0/llama-server -m x --port 0")
    assert lib.gpu_class(fake.split()) is None
    assert lib.gpu_class("python3.13 -u /tmp/llama-cli -m x".split()) is None
    assert lib.gpu_class("python -m llama-server".split()) is None
    assert lib.gpu_server_pids(f" 9 {fake}\n", resident_bytes=lambda pid: 3 * GIB) == []
    assert lib.gpu_class("/opt/homebrew/bin/llama-server -m X".split()) == "always"
    assert lib.gpu_class(["/opt/llama.cpp/build/bin/llama-server"]) == "always"
    assert lib.gpu_class("python -m mlx_lm.server --model m".split()) == "resident"
    assert lib.gpu_class("/Users/me/.venv/bin/python3 /Users/me/.venv/bin/mlx_lm.server --model m".split()) == \
        "resident"


def test_residency_comes_from_the_pids_own_footprint_line_then_ps(monkeypatch):
    text = ("======================================================================\n"
            "oMLX [5922]: 64-bit    Footprint: 135 MB (16384 bytes per page)\n"
            "======================================================================\n"
            "  Dirty      Clean  Reclaimable    Regions    Category\n"
            "other [77]: 64-bit    Footprint: 9 GB (16384 bytes per page)\n")
    assert lib.footprint_of_pid(text, 5922) == 135 * 1024**2 and lib.footprint_of_pid(text, 1) is None
    assert lib.process_resident_bytes(os.getpid()) > 10 * 1024**2          # the real tools, on this test process
    calls = []

    def run(argv, **kw):
        calls.append(argv[0])
        if argv[0] == "footprint":
            raise OSError("no footprint")
        return subprocess.CompletedProcess(argv, 0, stdout=" 2048\n", stderr="")

    monkeypatch.setattr(lib.subprocess, "run", run)
    assert lib.process_resident_bytes(5922) == 2048 * 1024 and calls == ["footprint", "ps"]


def test_paced_ticks_keep_the_grid_and_never_build_a_backlog():
    # paced_ticks is DESIGNED to drop ticks under load (never a backlog), so on a contended runner the fired indices
    # are legitimately sparse, e.g. [0, 2, 3, 5, ...]. Assert the real guarantees, not a contiguous [0, 1, 2, ...].
    grid = math.ceil(20 * 0.5)                                 # ceil(hz * duration_s): the number of grid slots

    def check_grid(rows):
        ks = [r["k"] for r in rows]
        assert ks, "a live scheduler must fire at least one tick"          # catches a dead scheduler (zero ticks)
        assert ks == sorted(set(ks)), f"k must strictly increase (no backlog, no rewind): {ks}"
        assert all(0 <= k < grid + 2 for k in ks), f"every tick is on the grid (+margin for jitter): {ks}"
        assert all(r["t_start"] >= r["t_due"] - 0.01 for r in rows)        # a tick never starts before it is due

    # A fast send would fire every tick on an idle box, but a contended runner legitimately drops some. Either way
    # the count stays within the grid (never more than it allows) and the indices obey the invariants above.
    fast = lib.paced_ticks(lambda k: {"ok": True}, hz=20, duration_s=0.5)
    check_grid(fast)
    assert 1 <= len(fast) <= grid + 1

    # A send slower than one interval MUST force paced_ticks to skip ticks (that is the anti-backlog behaviour).
    def slow(k):
        time.sleep(0.12)
        return {"ok": True}

    rows = lib.paced_ticks(slow, hz=20, duration_s=0.5)
    ks = [r["k"] for r in rows]
    check_grid(rows)
    assert len(rows) <= 5 and any(b - a > 1 for a, b in zip(ks, ks[1:], strict=False))   # ticks were genuinely skipped
    assert all(r["t_start"] - r["t_due"] < 0.05 + 0.01 for r in rows[1:])   # each send is for a tick due just now


def test_window_overlap_and_margin_helpers():
    assert lib.tokens_in_window([1.0, 1.5, 2.0, 3.5], 1.0, 3.0) == 1.5 and lib.tokens_in_window([], 2.0, 2.0) is None
    rows = [{"t_start": 0.0, "t_end": 1.0}, {"t_start": 2.0, "t_end": 3.0}, {"t_start": 5.0, "t_end": 6.0}]
    hit, rest = lib.split_by_overlap(rows, [(0.5, 2.5)])
    assert hit == rows[:2] and rest == rows[2:]
    assert lib.top2_margin({"a": 0.6, "b": 0.3, "c": 0.1}) == pytest.approx(0.3) and lib.top2_margin({"a": 1.0}) == 1.0


def test_first_pass_skips_the_truncation_marker_of_a_capped_timings_header():
    assert v02.first_pass({"timings": [{"q": "a", "cache_n": 5}, {"truncated": 2}]}) == {"q": "a", "cache_n": 5}
    assert v02.first_pass({"timings": [{"truncated": 1}]}) == {} and v02.first_pass({"timings": []}) == {}


def _rec(pair, probs, prompt="P<m1>Q"):
    return {"pair": pair, "media_marker": "<m1>", "prompt_string": prompt, "body": {"pair": pair}, "probs": probs}


def test_g0_parity_is_a_decision_equivalence_gate():
    g0 = {"up": 0.8, "down": 0.2}
    recs = [_rec(i, dict(g0)) for i in range(106)]

    # argmax 106/106 and a readout-method |Δp| (3e-5, well within the 1e-3 tolerance): PASS, not bit-exact
    answers = {i: dict(g0) for i in range(106)}
    answers[42] = {"up": 0.8 + 3e-5, "down": 0.2 - 3e-5}
    out = lib.g0_parity(recs, render=lambda body: "P<m2>Q", answer=lambda body: answers[body["pair"]], marker="<m2>")
    assert out["n"] == 106 and out["prompt_equal"] == 106 and out["agree"] == 106 and out["dp_tol"] == 1e-3
    assert out["max_dp"] == pytest.approx(3e-5) and out["max_dp_pair"] == 42 and out["over_tol"] == []
    assert out["pass"] is True

    # one argmax flip: hard FAIL (the decision differs), and it is listed as a disagreement
    flip = {i: dict(g0) for i in range(106)}
    flip[7] = {"up": 0.2, "down": 0.8}
    out = lib.g0_parity(recs, render=lambda body: "P<m2>Q", answer=lambda body: flip[body["pair"]], marker="<m2>")
    assert out["agree"] == 105 and [d["pair"] for d in out["disagreements"]] == [7] and out["pass"] is False

    # argmax unchanged but |Δp| 2e-3 over the tolerance: FAIL (cannot mask a flip, but a real divergence still fails)
    over = {i: dict(g0) for i in range(106)}
    over[3] = {"up": 0.798, "down": 0.202}
    out = lib.g0_parity(recs, render=lambda body: "P<m2>Q", answer=lambda body: over[body["pair"]], marker="<m2>")
    assert out["agree"] == 106 and out["max_dp"] == pytest.approx(2e-3) and [d["pair"] for d in out["over_tol"]] == [3]
    assert out["pass"] is False

    # fewer than n_expected, or a byte-different prompt, also fails
    assert lib.g0_parity(recs[:105], render=lambda body: "P<m2>Q", answer=lambda body: dict(g0),
                         marker="<m2>")["pass"] is False
    bad_prompt = lib.g0_parity(recs, render=lambda body: "P<m2>Q!", answer=lambda body: dict(g0), marker="<m2>")
    assert bad_prompt["prompt_equal"] == 0 and bad_prompt["pass"] is False


def test_g0_parity_aggregate_recomputes_from_stored_rows():
    rows = [{"pair": i, "prompt_equal": True, "max_dp": 3e-5, "callo": "up", "g0": "up"} for i in range(106)]
    assert lib.g0_parity_aggregate(rows)["pass"] is True
    rows[7]["callo"] = "down"                                       # a flip in the stored rows -> FAIL
    agg = lib.g0_parity_aggregate(rows)
    assert agg["agree"] == 105 and agg["pass"] is False and [d["pair"] for d in agg["disagreements"]] == [7]


def test_top2_logodds_shift_tracks_a_confident_item_that_barely_moves_in_probability():
    assert lib.top2_logodds_shift({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.0)
    assert lib.top2_logodds_shift({"a": 1.0}, {"a": 1.0}) == 0.0            # one option: no pair
    # a confident item: |Δp| is tiny but the log-odds moves a lot (ln(0.999/0.001) - ln(0.99/0.01))
    shift = lib.top2_logodds_shift({"a": 0.999, "b": 0.001}, {"a": 0.99, "b": 0.01})
    assert shift == pytest.approx(math.log(0.999 / 0.001) - math.log(0.99 / 0.01))
    assert shift > 2.0


def test_kv_sensitivity_summary_math_on_fixed_inputs():
    pairs = [
        ({"a": 0.9, "b": 0.1}, {"a": 0.9, "b": 0.1}),       # unchanged
        ({"a": 0.7, "b": 0.3}, {"a": 0.5, "b": 0.5}),       # |Δp| 0.2, argmax still a
        ({"a": 0.6, "b": 0.4}, {"a": 0.3, "b": 0.7}),       # |Δp| 0.3, argmax flips
    ]
    out = lib.kv_sensitivity_summary(pairs)
    assert out["n"] == 3 and out["argmax_agree"] == 2
    assert out["max_dp"] == pytest.approx(0.3) and out["median_dp"] == pytest.approx(0.2)
    assert out["max_logodds_shift"] == pytest.approx(lib.top2_logodds_shift(*pairs[2]))
    assert lib.kv_sensitivity_summary([])["n"] == 0 and lib.kv_sensitivity_summary([])["max_dp"] is None


def test_check_g0_records_enforces_the_contract():
    good = {"pair": 0, "media_marker": "<m>", "prompt_string": "p", "body": {"state": ""}, "probs": {"up": 1.0}}
    assert lib.check_g0_records([good]) is None
    assert "lacks ['probs']" in lib.check_g0_records([{k: v for k, v in good.items() if k != "probs"}])
    assert "wrong type" in lib.check_g0_records([{**good, "media_marker": ""}])
    assert "non-integer pair" in lib.check_g0_records([{**good, "pair": "0"}])
    assert "non-integer pair" in lib.check_g0_records([{**good, "pair": True}])
    assert lib.check_g0_records([good, {**good, "pair": 1}]) is None
    assert lib.check_g0_records([good, {**good, "pair": 1}, {**good, "pair": 0}]) == "duplicate pairs [0]"


class _Scorer:
    """Characters are tokens (id 1000 + ord); `vocab` adds multi-character pieces."""

    def __init__(self, vocab, detok):
        self.vocab, self.detok = vocab, detok

    def tokenize(self, text, *, add_special=True, parse_special=None):
        out, i = [], 0
        while i < len(text):
            hit = next((v for v in sorted(self.vocab, key=len, reverse=True) if text.startswith(v, i)), None)
            out.append(self.vocab[hit] if hit else 1000 + ord(text[i]))
            i += len(hit) if hit else 1
        return out

    def detokenize(self, tokens):
        return "".join(self.detok[t] for t in tokens)


def test_round_trip_token_skips_a_token_that_merges_with_the_tail():
    class Eng:
        media_marker = "<M>"
        scorer = _Scorer({"\n\n": 7, "**": 6}, {5: "\n", 6: "**", 9: "\ufffd"})

    prompt = "sys<M>question\n"                       # a "\n" step merges with the tail's "\n" into token 7
    assert v02._round_trip_token(Eng, prompt, {5: -0.1, 9: -0.5, 6: -1.0}) == (6, "**")
    assert v02._round_trip_token(Eng, prompt, {5: -0.1}) is None


def test_chat_stream_records_deltas_and_cancels_at_the_restart():
    up = FakeUpstream().start()
    up.stream_gap_s = 0.15                      # each chunk 150 ms apart: a 0.2 s cycle cancels mid-stream
    try:
        s = lib.ChatStream(f"{up.url}/v1/chat/completions", lambda: {"stream": True, "messages": []},
                           headers={"X-Callo-Conversation": "s2"}, restart_every=0.2)
        s.start()
        assert s.first_delta.wait(5)
        time.sleep(0.7)
        s.stop()
        assert len(s.requests) >= 2 and s.requests[0]["status"] == 200 and s.requests[0]["cancelled"]
        assert s.deltas and all(r["error"] is None for r in s.requests)
        assert up.last("/v1/chat/completions").headers["x-callo-conversation"] == "s2"
    finally:
        up.stop()


def test_chat_stream_records_a_non_200_as_an_error_and_backs_off():
    up = FakeUpstream().start()
    up.routes["/v1/chat/completions"] = lambda up, rec, h: h._reply(503, {"error": {"message": "Loading model"}})
    try:
        s = lib.ChatStream(f"{up.url}/v1/chat/completions", lambda: {"stream": True, "messages": []})
        s.start()
        time.sleep(1.4)
        assert s.stop() is True and s.stuck is False
        assert 1 <= len(s.requests) <= 2
        assert all(r["status"] == 503 and r["error"].startswith("HTTP 503") for r in s.requests)
        assert all(b["t_send"] - a["t_send"] >= 0.5 for a, b in zip(s.requests, s.requests[1:], strict=False))
        assert "System 2 request(s) errored, first: HTTP 503" in s.problems()
    finally:
        up.stop()


def test_chat_stream_stop_reports_a_thread_that_did_not_exit():
    release = threading.Event()

    def payload():
        release.wait(10)
        raise RuntimeError("released")

    s = lib.ChatStream("http://127.0.0.1:1/v1/chat/completions", payload)
    s.start()
    time.sleep(0.1)
    assert s.stop(timeout=0.2) is False and s.stuck is True
    assert "still running after stop()" in s.problems()
    release.set()
    s.join(5)
    assert not s.is_alive()


class AttachedCallo:
    """`callo serve` in attach mode on a fake upstream, with the attributes the bench rows use."""

    def __init__(self, up: FakeUpstream, tmp, *flags):
        self.lines: list[str] = []
        h = logging.Handler()
        h.emit = lambda rec: self.lines.append(rec.getMessage())
        logging.getLogger("callo").addHandler(h)
        logging.getLogger("callo").setLevel(logging.INFO)
        self._h = h
        args, extra = split_passthrough(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp / "c"),
                                         *flags])
        self.args = build_parser().parse_args(args)
        self.args.extra = extra
        self.stop, ready, self.port = threading.Event(), threading.Event(), None
        threading.Thread(target=lambda: serve(self.args, ready=lambda p: (setattr(self, "port", p), ready.set()),
                                              stop=self.stop), daemon=True).start()
        assert ready.wait(30)
        self.url, self.upstream, self.pid, self.build_info = f"http://127.0.0.1:{self.port}", up.url, None, "fake"

    def log_text(self):
        return "\n".join(self.lines)

    def stats(self):
        return lib.get(f"{self.url}/callo/stats")

    def finish(self):
        self.stop.set()
        logging.getLogger("callo").removeHandler(self._h)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    gguf = pytest.importorskip("gguf")
    monkeypatch.setenv("HOME", str(tmp_path))                      # local_engine's vocab cache
    g = tmp_path / "m.gguf"
    w = gguf.GGUFWriter(str(g), "qwen35")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(TOKENS)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    up = FakeUpstream(vision=True).start()
    up.props["model_path"] = str(g)
    up.stream_gap_s = 0.02
    c = AttachedCallo(up, tmp_path, "--s1-slots", "2")
    try:
        yield c, v02.V02Ctx(tmp_path / "out", hz_seconds=1.5, gguf=str(g)), up
    finally:
        c.finish()
        up.stop()


def test_every_main_row_runs_against_a_fake_vision_upstream(bench, tmp_path, monkeypatch):
    c, ctx, up = bench
    monkeypatch.setattr(v02, "_round_trip_token", lambda eng, prompt, lp: (1, "B"))  # the fake's ids are positional
    warm = v02.bind_lanes(c, ["s1", "aux"])
    assert [w["slot"] for w in warm] == ["0", "1"]
    assert "g0_parity" not in v02.MAIN_ROWS and "g0_parity" in v02.STANDALONE   # moved to its own instance
    summary = {"rows": {}, "started": "t0", "finished": "t1", "model": "m", "mmproj": "p", "ctx": 32768,
               "build_info": "b"}
    v02.run_main_rows(c, ctx, v02.MAIN_ROWS, lambda name, res: summary["rows"].__setitem__(name, res))
    rows = summary["rows"]
    assert not [n for n, r in rows.items() if r.get("error")], {n: r.get("error") for n, r in rows.items()}
    assert rows["selftests"]["pass"] is True
    assert rows["readout"]["n"] == 36 and rows["readout"]["invalid"] is None and rows["readout"]["correct"] == 9
    walk = rows["walk_reuse"]
    assert walk["slot"] == 1 and walk["step_token"] == [1, "B"] and walk["pass2"]["prompt_n"] == 3
    assert walk["identical_resend"]["prompt_n"] == 3 and walk["pass"] is False      # the fake reuses nothing
    assert up.last("/completion").json["prompt"]["prompt_string"].endswith("\n\n")   # the re-send, no "B"
    assert "Pac-Man is the yellow" not in rows["readout"]["rendered_prompt_example"]
    assert "The yellow circle is Pac-Man" in rows["readout"]["rendered_prompt_example"]
    assert rows["prefix_reuse"]["system_turn_tokens"] > 0 and rows["prefix_reuse"]["invalid"] is None
    assert rows["context_reuse"]["max_dp_cold_vs_warm"] == 0.0 and rows["context_reuse"]["context_tokens"] > 100
    assert rows["cache_drift"]["max_dp"] == 0.0 and rows["cache_drift"]["cold_was_cold"] is True
    assert rows["cache_drift"]["invalid"] is None                 # the fake's cold pass reports cache_n 0
    hz = rows["hz2_stream"]
    assert hz["invalid"] is None and hz["s1"]["slots"] == {"0": hz["s1"]["ok"]} and set(hz["s2_slots"]) <= {"2", "3"}
    assert hz["s2_tok_s"] > 0 and 2 <= hz["s1"]["n"] <= 4
    assert rows["hz2_prefill"]["s2_text_tokens"] >= 430 and rows["hz2_prefill"]["s2_cycles"] == 1
    md = render(summary)
    for check in ("vision readout in the demo layout", "2 Hz while System 2 streams", "G0 parity",
                  "KV-layout sensitivity", "walk pass reuse", "system-prefix reuse", "`context`-prefix reuse",
                  "demo slot layout", "cache drift", "no vision", "text-path regression", "self-tests", "footprint"):
        assert check in md
    assert "| NOT RUN |" in md and "9/36" in md                       # standalone rows were not run here


def _tiny_gguf(tmp_path, monkeypatch) -> Path:
    gguf = pytest.importorskip("gguf")
    monkeypatch.setenv("HOME", str(tmp_path))                      # local_engine's vocab cache
    g = tmp_path / "m.gguf"
    w = gguf.GGUFWriter(str(g), "qwen35")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(TOKENS)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return g


class Launched(AttachedCallo):
    def finish(self):
        super().finish()
        self.up.stop()


def _fake_launch(tmp_path, g, ups, *, force_vision=None):
    """`_launch` swapped for callo in attach mode on a fake upstream: vision iff `--mmproj` (or `force_vision`)."""

    def launch(name, out, *, model, flags):
        vision = "--mmproj" in flags if force_vision is None else force_vision
        rest = [f for i, f in enumerate(flags) if f != "--mmproj" and (i == 0 or flags[i - 1] != "--mmproj")]
        up = FakeUpstream(vision=vision).start()
        up.props["model_path"] = str(g)
        up.stream_gap_s = 0.02
        ups.append(up)
        c = Launched(up, tmp_path / name, *rest)
        c.up = up
        return c
    return launch


def test_the_standalone_rows_run_through_the_launch_seam(tmp_path, monkeypatch):
    """`demo_slots` and `no_vision` against callo on a fake upstream, through the same `_instance` path the GPU uses."""
    g = _tiny_gguf(tmp_path, monkeypatch)
    ups = []
    launch = _fake_launch(tmp_path, g, ups)
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(v02, "_launch", launch)
    ctx = v02.V02Ctx(tmp_path / "out", hz_seconds=1.5, gguf=str(g), instinct_every=0.4)
    demo = v02.row_demo_slots(ctx, tmp_path)
    assert demo["invalid"] is None, demo["invalid"]
    assert demo["lanes"] == {"s1": 0, "instinct_text": 1, "instinct_view": 2} and demo["s2_slots"] == ["3"]
    assert {i["lane"] for i in demo["instincts"]} <= {"instinct_text", "instinct_view"} and demo["instincts"]
    nv = v02.row_no_vision(ctx, tmp_path)
    assert nv["pass"] is True and nv["with_images_status"] == 501 and nv["without_images_status"] == 200
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [(7, "/x/omlx start")])
    with pytest.raises(RuntimeError, match="already running"):
        v02.row_no_vision(ctx, tmp_path)
    assert len(ups) == 2                                         # the refused start launched nothing


def test_main_refuses_to_start_while_a_model_server_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [(4242, "/x/llama-server -m y.gguf")])
    assert v02.main(["--out", str(tmp_path / "o")]) == 2
    assert "refusing to start" in capsys.readouterr().out and not (tmp_path / "o").exists()
    assert v02.main(["--out", str(tmp_path / "o"), "--rows", "nope"]) == 2


def test_no_vision_fails_when_the_server_answers_images(tmp_path, monkeypatch):
    g = _tiny_gguf(tmp_path, monkeypatch)
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(v02, "_launch", _fake_launch(tmp_path, g, [], force_vision=True))
    nv = v02.row_no_vision(v02.V02Ctx(tmp_path / "out", gguf=str(g)), tmp_path)
    assert nv["with_images_status"] == 200 and nv["vision"] is True and nv["pass"] is False


def test_main_stamps_each_row_restores_the_handlers_and_removes_run_pid(tmp_path, monkeypatch):
    g = _tiny_gguf(tmp_path, monkeypatch)
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(v02, "_launch", _fake_launch(tmp_path, g, []))
    monkeypatch.setattr(v02, "TIEL", str(g))
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    out = tmp_path / "o"
    assert v02.main(["--out", str(out), "--rows", "no_vision", "--min-free-gb", "0", "--hz-seconds", "1.5"]) == 0
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before
    assert not (out / "run.pid").exists()
    summary = json.loads((out / "summary.json").read_text())
    row = summary["rows"]["no_vision"]
    assert row["pass"] is True and row["build_info"] == "fake" and row["hz_seconds"] == 1.5
    assert row["finished_at"].endswith("+00:00")
    assert row["launch_flags"] == []                 # the instance's launch flags are stamped per row (I-3)
    assert f"| no_vision | {row['finished_at']} | fake | 1.5 | base launch only |" in render(summary)


def test_report_flags_abbreviates_absolute_paths_and_handles_missing():
    from bench.report_v02 import _flags
    assert _flags(None) == "n/a" and _flags([]) == "base launch only"   # old summary vs base launch
    out = _flags(["--mmproj", "/Volumes/Data/models/x/mmproj-BF16.gguf", "--s1-slots", "2",
                  "--", "--cache-ram", "0", "--slot-save-path", "/Users/me/results/main/slots/"])
    assert "--cache-ram 0" in out and "--s1-slots 2" in out
    assert "…/x/mmproj-BF16.gguf" in out and "…/main/slots/" in out    # abbreviated, trailing slash kept
    assert "/Volumes/Data" not in out


_SIGTERM_BENCH = """
import subprocess, sys, time
from pathlib import Path
from bench import acceptance_v02 as v02, lib

tmp = Path(sys.argv[1])
lib.running_gpu_servers = lambda: []


class FakeServer:
    \"\"\"Stands in for callo + llama-server: a child in its own session, as callo launches llama-server.\"\"\"
    build_info, lines = "fake", []

    def __init__(self):
        self.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)
        (tmp / "child.pid").write_text(str(self.proc.pid))

    def finish(self):
        self.proc.terminate()
        self.proc.wait(30)
        (tmp / "finished").write_text("1")


def block(c, ctx):
    (tmp / "ready").write_text("1")
    time.sleep(300)


v02._launch = lambda name, out, *, model, flags: FakeServer()
v02.bind_lanes = lambda c, lanes: [{"ok": True, "status": 200, "slot": str(i)} for i in range(len(lanes))]
v02._MAIN["selftests"] = block
sys.exit(v02.main(["--out", str(tmp / "out"), "--rows", "selftests", "--min-free-gb", "0"]))
"""


def test_sigterm_to_run_pid_stops_the_server_and_removes_run_pid(tmp_path):
    script = tmp_path / "bench_under_test.py"
    script.write_text(_SIGTERM_BENCH)
    p = subprocess.Popen([sys.executable, str(script), str(tmp_path)], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    child = None
    try:
        deadline = time.time() + 60
        while not (tmp_path / "ready").exists():
            assert p.poll() is None, p.stdout.read()
            assert time.time() < deadline
            time.sleep(0.05)
        child = int((tmp_path / "child.pid").read_text())
        assert int((tmp_path / "out" / "run.pid").read_text()) == p.pid
        p.send_signal(signal.SIGTERM)                                # what `kill $(cat run.pid)` does
        assert p.wait(60) == 128 + signal.SIGTERM
        assert (tmp_path / "finished").exists() and not (tmp_path / "out" / "run.pid").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
        child = None
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        p.stdout.close()
        if child is not None:                                        # a pid this test started, never by name
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
