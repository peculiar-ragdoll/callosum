"""The GPU-free parts of the acceptance bench plumbing."""
import pytest

from bench.lib import (
    assess_validity,
    build_long_text,
    delta_p,
    overlap_fraction,
    parse_footprint,
    percentile,
    s1_subset,
    summarize,
)


def test_percentile_nearest_rank():
    assert percentile([3, 1, 2], 0.5) == 2 and percentile([1, 2, 3, 4], 0.95) == 4
    assert percentile([5.0] * 100, 0.07) == 5.0


def test_parse_footprint():
    assert parse_footprint("phys_footprint ... Footprint: 8.31 GB\n") == int(8.31 * 1024**3)
    assert parse_footprint("Footprint: 512 MB") == 512 * 1024**2 and parse_footprint("nothing") is None


def test_s1_subset_is_evenly_spaced_and_deterministic():
    recs = list(range(656))
    sub = s1_subset(recs, 100)
    assert len(sub) == 100 and sub[0] == 0 and sub == [i * 656 // 100 for i in range(100)]
    assert s1_subset(recs, 1000) == recs


def test_overlap_fraction_and_validity_gate():
    bg = [{"t_start": 0.0, "t_end": 10.0}, {"t_start": 10.2, "t_end": 20.0}, {"t_start": 30.0, "t_end": 31.0, "error": "x"}]
    s1 = [{"ok": True, "t_start": 1.0, "t_end": 2.0}, {"ok": True, "t_start": 9.9, "t_end": 10.1},
          {"ok": True, "t_start": 25.0, "t_end": 26.0}]
    assert abs(overlap_fraction(s1, bg) - 2 / 3) < 1e-12          # the 0.2 s gap merges; 25..26 is outside
    v = assess_validity(s1, bg, has_workload=True)
    assert not v["valid"] and "errored" in v["reason"]
    v = assess_validity(s1, bg[:2], has_workload=True)
    assert not v["valid"] and "overlapped" in v["reason"]
    v = assess_validity(s1[:2], bg[:2], has_workload=True)
    assert v["valid"] and v["overlap_frac"] == 1.0
    assert assess_validity(s1, [], has_workload=False)["valid"]


def test_delta_p_and_summarize():
    a = {"r/q": [0.2, 0.8], "r/o": [0.5, 0.5]}
    b = {"r/q": [0.25, 0.75], "r/z": [1.0]}
    d = delta_p(a, b)
    assert d["n"] == 1 and d["max"] == pytest.approx(0.05) and d["mean"] == pytest.approx(0.05)   # 0.8 - 0.75 != 0.05 exactly
    s1 = [{"ok": True, "latency_ms": 10.0, "t_start": 0, "t_end": 0.01}, {"ok": False, "t_start": 1, "t_end": 2}]
    bg = [{"t_start": 0, "t_end": 5, "predicted_per_second": 40.0, "prompt_per_second": 800.0, "prompt_n": 32000, "cache_n": 0}]
    out = summarize(s1, bg, [{"footprint": 100, "rss": 90}, {"footprint": 120, "rss": 80}])
    assert out["s1"] == {"n": 1, "errors": 1, "p50_ms": 10.0, "p95_ms": 10.0, "max_ms": 10.0}
    assert out["background"]["decode_tok_s_mean"] == 40.0 and out["background"]["prompt_n_min"] == 32000
    assert out["memory"] == {"footprint_max": 120, "rss_max": 90}


def test_build_long_text_hits_the_token_target_within_three_percent():
    corpus = "alpha beta gamma delta epsilon zeta eta theta iota kappa\n" * 50

    def count(text):
        return len(text.split())

    text, measured = build_long_text(count, 3000, corpus)
    assert 3000 <= measured <= 3090 and count(text) == measured


def test_report_renders_without_a_full_summary():
    from bench.report import render
    text = render({"rows": {"selftest": {"pass": True, "lines": ["self-test ok"]}, "drop_in": {"pass": False, "error": "boom"}},
                   "model": "m", "started": "t0", "finished": "t1", "master_build_info": "b"})
    assert "| self-test + answer-slot check |" in text and "PASS" in text and "FAIL" in text and "- drop_in: boom" in text


def test_drop_in_norm_ignores_only_the_random_tool_call_id():
    from bench.acceptance import _norm

    def resp(call_id, args):
        return {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": args}, "id": call_id}]}}]}

    assert _norm(resp("q4Ah", '{"city":"Paris"}')) == _norm(resp("96R0", '{"city":"Paris"}'))
    assert _norm(resp("q4Ah", '{"city":"Paris"}')) != _norm(resp("q4Ah", '{"city":"Lyon"}'))
    plain = {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}
    assert _norm(plain) == {"content": "hi", "reasoning_content": None, "tool_calls": None, "finish_reason": "stop"}


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  2337884.
Pages active:                                 562743.
Pages inactive:                               836164.
Pages speculative:                            132527.
Pages throttled:                                   0.
"""


def test_parse_vm_stat_counts_free_inactive_speculative():
    from bench.lib import parse_vm_stat
    assert parse_vm_stat(VM_STAT) == (2337884 + 836164 + 132527) * 16384
    assert parse_vm_stat("Pages free: 5.\n") is None and parse_vm_stat("") is None


def test_s1_overlap_stats_use_only_requests_under_load():
    from bench.lib import overlapping_s1
    bg = [{"t_start": 0.0, "t_end": 10.0}]
    s1 = [{"ok": True, "latency_ms": 900.0, "t_start": 1.0, "t_end": 2.0},
          {"ok": True, "latency_ms": 100.0, "t_start": 11.0, "t_end": 11.1},      # after the load: not under it
          {"ok": True, "latency_ms": 100.0, "t_start": 12.0, "t_end": 12.1}]
    assert [r["latency_ms"] for r in overlapping_s1(s1, bg)] == [900.0]
    out = summarize(s1, bg, [])
    assert out["s1"]["p50_ms"] == 100.0 and out["s1_overlap"] == {"n": 1, "p50_ms": 900.0, "p95_ms": 900.0, "max_ms": 900.0}


def test_inside_s1_window_and_slot_conflicts():
    from bench.lib import inside_s1_window, slot_conflicts
    s1 = [{"ok": True, "latency_ms": 1.0, "t_start": 10.0, "t_end": 11.0}, {"ok": True, "latency_ms": 1.0, "t_start": 40.0, "t_end": 50.0}]
    bg = [{"t_start": 5.0, "t_end": 15.0, "slot": "1"},                 # starts before the window: diluted, excluded
          {"t_start": 12.0, "t_end": 20.0, "slot": "1", "predicted_per_second": 10.0},
          {"t_start": 13.0, "t_end": 49.0, "slot": "2", "predicted_per_second": 20.0},
          {"t_start": 45.0, "t_end": 60.0, "slot": "3"},                # ends after it
          {"t_start": 20.0, "t_end": 30.0, "slot": "4", "error": "x"}]
    assert [r["slot"] for r in inside_s1_window(bg, s1)] == ["1", "2"]
    b = summarize(s1, bg, [])["background"]
    assert b["n_inside_s1"] == 2 and b["decode_tok_s_mean_inside_s1"] == 15.0 and b["slots_inside_s1"] == ["1", "2"]
    assert slot_conflicts(bg[:4]) == 1                                    # rows 0 and 1 overlap 3 s on slot 1
    assert slot_conflicts(bg[1:4]) == 0
    handoff = [{"t_start": 0.0, "t_end": 4.5, "slot": "1"}, {"t_start": 4.499, "t_end": 9.0, "slot": "1"}]
    assert slot_conflicts(handoff) == 0                                   # a 1 ms client-side handoff is not queueing


def test_run_s1_cycles_records_until_the_minimum_window(monkeypatch):
    from bench import lib
    sent = []

    def fake_post(url, payload, **kw):
        sent.append(payload["state"])
        return 200, {"x-callo-slot": "0"}, {"answers": {"q": {"noul": 0.5}}}

    monkeypatch.setattr(lib, "post", fake_post)
    recs = [{"state": i, "questions": {"q": {"type": "noul"}}, "_meta": {"id": f"r{i}"}} for i in range(3)]
    rows, probs = lib.run_s1("http://x", recs)
    assert sent == [0, 1, 2] and len(rows) == 3 and rows[0]["slot"] == "0" and set(probs) == {"r0/q", "r1/q", "r2/q"}
    sent.clear()
    rows, _ = lib.run_s1("http://x", recs, min_duration_s=0.05)
    assert sent[:3] == [0, 1, 2] and len(sent) > 3 and sent[3:6] == [0, 1, 2][: len(sent[3:6])]


def test_response_timings_reads_timings_or_anthropic_usage():
    from bench.lib import response_timings
    assert response_timings({"timings": {"prompt_n": 5, "cache_n": 2, "predicted_n": 1}})["cache_n"] == 2
    u = response_timings({"type": "message", "usage": {"input_tokens": 32000, "cache_read_input_tokens": 0, "output_tokens": 1}})
    assert u == {"prompt_n": 32000, "cache_n": 0, "predicted_n": 1}
    assert response_timings({"usage": {"input_tokens": 10, "cache_read_input_tokens": 90}})["prompt_n"] == 100
    assert response_timings("not json") == {} and response_timings({}) == {}


def test_background_records_the_callo_slot_and_timings():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from bench.lib import Background

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            data = json.dumps({"timings": {"prompt_n": 7, "cache_n": 0, "predicted_per_second": 3.0}}).encode()
            self.send_response(200)
            self.send_header("X-Callo-Slot", "2")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        bg = Background(f"http://127.0.0.1:{srv.server_address[1]}/x", dict, n=2, max_requests=2).start()
        bg.join(timeout=10)
    finally:
        srv.shutdown()
        srv.server_close()
    assert len(bg.rows) == 4 and all(r["slot"] == "2" and r["prompt_n"] == 7 and "error" not in r for r in bg.rows)


def test_memory_sampler_stops_and_joins():
    import os
    import time

    from bench.lib import MemorySampler

    m = MemorySampler(os.getpid(), interval=0.05)
    m.start()
    deadline = time.monotonic() + 60    # a contended CI runner may not have run the thread yet: wait for a first row
    while not m.rows and m.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    m.stop()
    m.join(timeout=60)                  # with `_stop` shadowing Thread._stop this raised TypeError on py3.10-3.12
    assert not m.is_alive() and m.rows and m.rows[0]["rss"]


def test_report_verdicts_never_show_invalid_or_missing_as_pass():
    from bench.report import _verdict, render
    assert _verdict(None) == "NOT RUN" and _verdict({}) == "NOT RUN"
    assert _verdict({"pass": True, "invalid": "only 40% overlapped | x"}) == "INVALID: only 40% overlapped / x"
    assert _verdict({"pass": None, "invalid": "prefill not proven"}).startswith("INVALID")
    assert _verdict({"pass": None}) == "measured" and _verdict({"pass": False}) == "FAIL"
    text = render({"rows": {"s1_prefill_128k": {"pass": None, "invalid": "no background requests completed"}}})
    assert "INVALID: no background requests completed" in text and "| NOT RUN |" in text and "measured |" not in text


def test_main_refuses_unknown_rows_and_low_memory_before_launching(monkeypatch, tmp_path):
    from bench import acceptance, lib
    launched = []
    monkeypatch.setattr(acceptance, "Callo", lambda *a, **k: launched.append(1))
    assert acceptance.main(["--out", str(tmp_path / "o"), "--rows", "selftest,nope"]) == 2
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 10 * 1024**3)
    assert acceptance.main(["--out", str(tmp_path / "o"), "--rows", "selftest"]) == 2
    assert not launched and not (tmp_path / "o").exists()


def test_main_survives_a_failed_callo_start_and_exits_nonzero(monkeypatch, tmp_path):
    from bench import acceptance, lib

    def boom(*a, **k):
        raise RuntimeError("callo did not start (rc=1)")

    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(acceptance, "Callo", boom)
    suite = tmp_path / "suite.jsonl"
    suite.write_text('{"state": {}, "questions": {}, "_meta": {"id": "r0", "variant": "clean"}}\n')
    rc = acceptance.main(["--out", str(tmp_path / "o"), "--rows", "selftest,brew", "--suite-file", str(suite)])
    import json
    rows = json.loads((tmp_path / "o" / "summary.json").read_text())["rows"]
    assert rc == 1 and set(rows) == {"selftest", "brew"} and all("RuntimeError" in r["error"] for r in rows.values())


def test_callo_failed_start_raises_runtimeerror_and_reaps_its_child(monkeypatch, tmp_path):
    """serve dies during startup leaving its llama-server behind: Callo must raise RuntimeError (not SystemExit),
    detach its log handler and stop the child whose pid file its own launch wrote."""
    import logging
    import os
    import subprocess

    from bench import acceptance

    children = []

    def fake_serve(args, *, ready=None, stop=None, listening=None):
        d = acceptance.Path(args.cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(["sleep", "60"], start_new_session=True)
        children.append(p)
        (d / f"llama-server-starting-{os.getpid()}.pid").write_text(f"{p.pid}\n")
        return 1

    monkeypatch.setattr(acceptance, "serve", fake_serve)
    # the child is `sleep`, standing in for llama-server: reap_cache_dir's args check sees it as one
    monkeypatch.setattr(acceptance, "_pid_args", lambda pid: f"/x/llama-server -m m.gguf (pid {pid})")
    before = list(logging.getLogger("callo").handlers)
    with pytest.raises(RuntimeError, match="did not start"):
        acceptance.Callo(tmp_path)
    assert children and children[0].wait(timeout=20) is not None
    assert logging.getLogger("callo").handlers == before


def test_report_number_formatting():
    from bench.report import _dp, _gib, _kv_events, _ms, _pct, _s, _tps, render
    assert _tps(681.0653958710403) == "681 tok/s" and _s(42.13364803791046) == "42.13 s" and _ms(2150.4) == "2.15 s"
    assert _pct(0.7054430362203452) == "70.5%" and _gib(20401094656) == "19.0 GiB" and _dp(0.0) == "0.0000"
    assert _tps(None) == _s(None) == _pct(None) == _gib(None) == _dp(None) == "n/a"
    ev = [{"conv": "A-turn2", "status": 200, "slot": "1", "timings": {"prompt_n": 141806, "cache_n": 0, "prompt_ms": 299900.0}}]
    assert _kv_events(ev) == "A-turn2: HTTP 200 slot 1, prompt_n 141806 cache_n 0 (prefill 299.90 s)"
    text = render({"rows": {"s1_decode": {"pass": None, "solo_decode_tok_s": 30.744083078339923, "decode_hit_frac": 0.7054430362203452,
                                          "background": {"decode_tok_s_mean_inside_s1": 9.055883765745271, "n_inside_s1": 35}}}})
    assert "31 tok/s -> 9 tok/s over 35 requests (hit 70.5%)" in text and "30.744" not in text


def test_kv_log_lines_scan_the_whole_log(tmp_path):
    from bench.acceptance import kv_log_lines
    log = tmp_path / "llama-server-1.log"
    log.write_text("8.52 W srv  try_clear_id: purging slot 1 with 141812 tokens\n" + "noise line\n" * 5000
                   + "13.54 W srv  try_clear_id: purging slot 2 with 141812 tokens\n")
    assert [ln.split()[0] for ln in kv_log_lines(log)] == ["8.52", "13.54"]     # a 20 kB tail would miss the first
    assert kv_log_lines(tmp_path / "missing.log") == []


def test_callo_finish_reaps_its_own_child_even_when_an_interrupt_lands_during_the_join(monkeypatch, tmp_path):
    """A Ctrl-C (or the stop handlers' SystemExit) inside `thread.join` still runs `_reap`, which signals
    only the llama-server pid files THIS instance's launch wrote (never one that was there before it). No real
    process is signalled: `os.killpg` and the liveness check are injected."""
    import logging
    import threading

    from bench import acceptance

    class Interrupted:
        def join(self, timeout=None):
            raise KeyboardInterrupt

    c = acceptance.Callo.__new__(acceptance.Callo)
    c.stop, c.thread, c.rc, c.cache_dir = threading.Event(), Interrupted(), None, tmp_path / "cache"
    c._handler = logging.Handler()
    logging.getLogger("callo").addHandler(c._handler)
    c._preexisting = {111}                                             # a pid file from before this launch: not ours
    monkeypatch.setattr(acceptance, "find_orphans", lambda cache_dir: [(111, 9001), (222, 9002)])
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(acceptance.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(acceptance, "_alive", lambda pid: False)
    monkeypatch.setattr(acceptance, "_pid_args", lambda pid: "/x/llama-server -m m.gguf --port 9002")
    with pytest.raises(KeyboardInterrupt):
        c.finish()
    assert c.stop.is_set() and sent == [(222, acceptance.signal.SIGTERM)]
    assert c._handler not in logging.getLogger("callo").handlers


def test_reap_cache_dir_signals_only_a_pid_whose_args_name_llama_server(monkeypatch, tmp_path, caplog):
    """A pid file's pid can be reused once our llama-server died; `reap_cache_dir`
    signals it only when `ps -o args=` names llama-server, else logs and skips it (an unreadable one too). No real
    process is signalled: `os.killpg`, the liveness check and `ps` are injected."""
    from bench import acceptance
    monkeypatch.setattr(acceptance, "find_orphans", lambda cache_dir: [(301, 9001), (302, 9002), (303, 9003)])
    args = {301: "/usr/bin/vim notes.txt", 302: None, 303: "/opt/llama.cpp/build/bin/llama-server -m m.gguf"}
    monkeypatch.setattr(acceptance, "_pid_args", args.get)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(acceptance.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(acceptance, "_alive", lambda pid: False)
    with caplog.at_level("WARNING"):
        acceptance.reap_cache_dir(tmp_path, set())
    assert sent == [(303, acceptance.signal.SIGTERM)]
    assert "not reaping pid 301" in caplog.text and "not reaping pid 302" in caplog.text


def test_is_bench_run_is_the_p3_rule_with_signal_0_and_ps_as_seams():
    """lib.is_bench_run (shared by the think and transcription benches): a live pid whose command line names the
    bench's module. Another user's process (PermissionError on signal 0), a dead pid, or an unreadable command line
    never is. Nothing is signalled: `kill` and `pid_args` are injected."""
    from bench.lib import is_bench_run
    sent, asked = [], []

    def alive(pid, sig):
        sent.append((pid, sig))

    def args(pid):
        asked.append(pid)
        return "/usr/bin/python3 -m bench.acceptance_v03_asr --set s --out o"

    def raises(e):
        def kill(pid, sig):
            raise e
        return kill

    assert is_bench_run(4242, "acceptance_v03_asr", pid_args=args, kill=alive) is True
    assert is_bench_run(4242, "acceptance_v03_think", pid_args=args, kill=alive) is False
    assert is_bench_run(4242, "acceptance_v03_asr", pid_args=lambda pid: None, kill=alive) is False
    assert sent == [(4242, 0)] * 3 and asked == [4242, 4242]
    for e in (PermissionError(1, "denied"), ProcessLookupError(3, "gone"), OverflowError("pid")):
        assert is_bench_run(4242, "acceptance_v03_asr", pid_args=args, kill=raises(e)) is False
    assert asked == [4242, 4242]                       # ps is never asked about a pid signal 0 cannot reach


def test_callo_serve_argv_says_the_cache_explicitly():
    """--cache is on by default since 0.3.0; the bench's launches measure with it OFF unless asked (their recorded
    results and resumes are cache-off), so the argv always says which, before any `--` passthrough in `flags`."""
    from pathlib import Path

    from bench import acceptance
    from callo.cli import build_parser, split_passthrough

    off = acceptance.callo_serve_argv(model="/m.gguf", llama_server="ls", ctx=8192, parallel=4, cache_dir=Path("/c"),
                                      flags=["--s1-slots", "1", "--", "--cache-ram", "0"])
    on = acceptance.callo_serve_argv(model="/m.gguf", llama_server="ls", ctx=8192, parallel=4, cache_dir=Path("/c"),
                                     flags=["--", "--cache-ram", "0"], cache=True)
    assert "--no-cache" in off and "--cache" not in off and off.index("--no-cache") < off.index("--")
    assert "--cache" in on and "--no-cache" not in on and on.index("--cache") < on.index("--")
    for argv, want in ((off, False), (on, True)):
        args, extra = split_passthrough(argv)
        assert build_parser().parse_args(args).cache is want and extra == ["--cache-ram", "0"]
    # the --cache multi-question row compares OFF with ON, both explicit
    assert acceptance.CACHE_MULTI_MODES == (("default", False), ("cache", True))


# ---- --model, the solo rows and |Δp| against them -------------------------------------------------------

# A shared fixture and the reference implementation's output on it, copied here as literals (checked once by hand
# against that implementation): per question the max |Δp| over options, then the max and mean over the keys both
# runs answered. Keys only one run answered are ignored. Every value is dyadic, so the sums are exact on every Python
# (3.12 made float sum() compensated: 0.134 on 3.13 is 0.13400000000000004 on 3.10).
DP_SOLO = {"r0/q": [0.25, 0.75], "r0/n": [0.625, 0.375], "r1/q": [0.125, 0.25, 0.625], "r2/q": [0.5, 0.5],
           "r3/q": [0.0625, 0.1875, 0.75], "only_solo/q": [1.0, 0.0]}
DP_OTHER = {"r0/q": [0.375, 0.625], "r0/n": [0.25, 0.75], "r1/q": [0.125, 0.625, 0.25], "r2/q": [0.5, 0.5],
            "r3/q": [0.125, 0.125, 0.75], "only_other/q": [0.0, 1.0]}
DP_REFERENCE = {"n": 5, "max": 0.375, "mean": 0.1875}


def test_delta_p_matches_the_reference_on_a_shared_fixture_and_counts_argmax_flips():
    d = delta_p(DP_SOLO, DP_OTHER)
    assert {k: d[k] for k in ("n", "max", "mean")} == DP_REFERENCE      # exact, not approx: the same arithmetic
    # r0/n (0 -> 1) and r1/q (2 -> 1) flip; r0/q and r3/q keep their argmax; r2/q is a tie both times (first index)
    assert d["flips"] == 2
    assert delta_p({}, DP_OTHER) == {"n": 0, "max": None, "mean": None, "flips": 0}


SOLO_OPTIONS = [["one", "two", "three"], ["red", "green", "blue", "cyan"], ["yes", "no"]]


def _solo_suite(path, n=6):
    import json
    lines = []
    for i in range(n):
        crit = {f"o{j}": t for j, t in enumerate(SOLO_OPTIONS[i % len(SOLO_OPTIONS)])}
        lines.append(json.dumps({"state": f"state {i}", "questions": {"q": {"type": "choice", "instructions": f"pick {i}",
                                                                           "criteria": crit}},
                                 "_meta": {"id": f"r{i}", "variant": "clean"}}))
    lines.append(json.dumps({"state": "dirty", "questions": {"q": {"type": "choice", "instructions": "x",
                                                                   "criteria": {"o0": "a", "o1": "b"}}},
                             "_meta": {"id": "dirty", "variant": "noisy"}}))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def stub_callo(monkeypatch):
    """`acceptance.Callo` swapped for a stand-in served by the callo stub (POST /v1/systemone, GET /callo/stats):
    every construction's (out dir, kwargs) is recorded in `made`, and a background workload is refused outright."""
    from fake_callo import CalloStub
    from fake_upstream import PlantedBias

    from bench import acceptance, lib
    made, stubs = [], []

    class StubCallo:
        def __init__(self, out, **kw):
            made.append((out.name, kw))
            self.stub = CalloStub(PlantedBias())
            stubs.append(self.stub)
            self.url, self.upstream, self.pid, self.build_info, self.lines = self.stub.url, None, None, "fake", []

        def stats(self):
            return lib.get(f"{self.url}/callo/stats")

        def log_text(self):
            return ""

        def finish(self):
            self.stub.stop()
            return 0

    def no_background(*a, **k):
        raise AssertionError("a solo row started a background workload")

    monkeypatch.setattr(acceptance, "Callo", StubCallo)
    monkeypatch.setattr(lib, "Background", no_background)
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 1024**3)
    yield made
    for s in stubs:
        try:
            s.stop()
        except Exception:  # noqa: BLE001 -- already stopped by finish()
            pass


def test_the_solo_rows_run_first_on_the_main_callo_with_no_background_and_write_three_files(stub_callo, monkeypatch,
                                                                                           tmp_path):
    import json

    from bench import acceptance
    order = []

    def fake_prefill(c, ctx, n_tokens, name, pass_p50_s):
        order.append(name)
        ctx.save(name, "probs.json", {"r0/q": [0.9, 0.05, 0.05], "r1/q": [0.25, 0.25, 0.25, 0.25]})
        return {"pass": None}

    real_solo = acceptance.row_s1_solo

    def spy_solo(c, ctx, name):
        order.append(name)
        return real_solo(c, ctx, name)

    monkeypatch.setattr(acceptance, "row_s1_prefill", fake_prefill)
    monkeypatch.setattr(acceptance, "row_s1_solo", spy_solo)
    suite = _solo_suite(tmp_path / "suite.jsonl")
    out = tmp_path / "o"
    rc = acceptance.main(["--out", str(out), "--rows", "s1_prefill_32k,s1_solo_repeat,s1_solo", "--suite-file", str(suite),
                          "--s1-limit", "4"])
    assert rc == 0, json.loads((out / "summary.json").read_text())
    assert order == ["s1_solo", "s1_solo_repeat", "s1_prefill_32k"]
    assert [m[0] for m in stub_callo] == ["callo_main"]                    # one launch: every main row shares it
    for row in ("s1_solo", "s1_solo_repeat"):
        assert sorted(p.name for p in (out / row).iterdir()) == ["probs.json", "s1.json", "summary.json"]
        s1 = json.loads((out / row / "s1.json").read_text())
        assert [r["id"] for r in s1] == ["r0", "r1", "r3", "r4"]          # --s1-limit 4 of the 6 clean records
        probs = json.loads((out / row / "probs.json").read_text())
        assert sorted(probs) == ["r0/q", "r1/q", "r3/q", "r4/q"]
        summ = json.loads((out / row / "summary.json").read_text())
        assert summ["s1"]["n"] == 4 and summ["s1"]["errors"] == 0
        assert {"p50_ms", "p95_ms", "max_ms"} <= set(summ["s1"]) and summ["slots"] == {"0": 4}
        assert summ["pass"] is None and summ["invalid"] is None
    summary = json.loads((out / "summary.json").read_text())
    # the stub answers deterministically, so the two solo passes agree exactly: the floor is 0
    assert summary["delta_p_floor"] == {"n": 4, "max": 0.0, "mean": 0.0, "flips": 0}
    assert set(summary["delta_p_vs_solo"]) == {"s1_solo_repeat", "s1_prefill_32k"}
    assert summary["delta_p_vs_solo"]["s1_solo_repeat"] == summary["delta_p_floor"]
    assert summary["delta_p_vs_solo"]["s1_prefill_32k"]["n"] == 2


def test_the_floor_is_absent_without_both_solo_rows(stub_callo, tmp_path):
    import json

    from bench import acceptance
    suite = _solo_suite(tmp_path / "suite.jsonl")
    out = tmp_path / "o"
    assert acceptance.main(["--out", str(out), "--rows", "s1_solo", "--suite-file", str(suite), "--s1-limit", "3"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert "delta_p_floor" not in summary and summary["delta_p_vs_solo"] == {}
    assert (out / "s1_solo" / "probs.json").exists() and not (out / "s1_solo_repeat").exists()


def test_model_reaches_every_callo_launch_and_the_summary(monkeypatch, tmp_path):
    """--model is passed to every `Callo(...)` the rows construct (the brew row keeps its own model by design) and
    recorded as the summary's model; the default stays the series' own model."""
    import json

    from bench import acceptance, lib
    made = []

    class Launched(Exception):
        pass

    def spy_init(self, out, **kw):
        made.append((out.name, kw.get("model")))
        raise Launched(out.name)

    monkeypatch.setattr(acceptance.Callo, "__init__", spy_init)
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 1024**3)
    suite = _solo_suite(tmp_path / "suite.jsonl")
    tiel = "/m/tiel.gguf"
    out = tmp_path / "o"
    rc = acceptance.main(["--out", str(out), "--model", tiel, "--suite-file", str(suite),
                          "--rows", "s1_solo,s1_decode,kv_capacity,s1_fallback,cache_multi"])
    assert rc == 1 and json.loads((out / "summary.json").read_text())["model"] == tiel
    # a loop row stops at its first failed launch; each remaining launch site is driven directly
    ctx = acceptance.Ctx(out, str(suite), 3, model=tiel)
    with pytest.raises(Launched):
        acceptance._fallback_mode(ctx, out, "off")
    monkeypatch.setattr(acceptance, "CACHE_MULTI_MODES", (("cache", True),))
    with pytest.raises(Launched):
        acceptance.row_cache_multi(ctx, out)
    assert sorted(made) == sorted([("callo_main", tiel), ("callo_decode", tiel), ("callo_kv", tiel),
                                   ("callo_fallback_on", tiel), ("callo_default", tiel), ("callo_fallback_off", tiel),
                                   ("callo_cache", tiel)])
    made.clear()
    assert acceptance.main(["--out", str(tmp_path / "d"), "--suite-file", str(suite), "--rows", "s1_solo"]) == 1
    assert made == [("callo_main", acceptance.MODEL)]
    assert json.loads((tmp_path / "d" / "summary.json").read_text())["model"] == acceptance.MODEL


def test_the_32k_pass_gate_applies_only_to_the_default_model(stub_callo, monkeypatch, tmp_path):
    from bench import acceptance
    gates = []
    monkeypatch.setattr(acceptance, "row_s1_prefill",
                        lambda c, ctx, n_tokens, name, pass_p50_s: gates.append((name, pass_p50_s)) or {"pass": None})
    suite = _solo_suite(tmp_path / "suite.jsonl")
    for model in (acceptance.MODEL, "/m/tiel.gguf"):
        assert acceptance.main(["--out", str(tmp_path / model.replace("/", "_")), "--model", model, "--suite-file",
                                str(suite), "--rows", "s1_prefill_32k,s1_prefill_128k"]) == 0
    assert gates == [("s1_prefill_32k", acceptance.STAGE1_PINNED_B512_32K_S), ("s1_prefill_128k", None),
                     ("s1_prefill_32k", None), ("s1_prefill_128k", None)]


@pytest.mark.parametrize("flag, want, never", [(None, "--no-cache", "--cache"), ("on", "--cache", "--no-cache"),
                                               ("off", "--no-cache", "--cache")])
def test_cache_flag_reaches_every_row_launch_and_the_summary(monkeypatch, tmp_path, flag, want, never):
    """--cache defaults to off, as every recorded run measured; `on` is the shipped default of `callo serve` (asked
    for explicitly, so the argv says which). cache_multi keeps its own off/on pair and brew its own launch."""
    import json
    import logging

    from bench import acceptance, lib
    argvs = []

    class Launched(Exception):
        pass

    real = acceptance.callo_serve_argv

    def spy(**kw):
        argvs.append((kw["cache_dir"].parent.name, real(**kw)))
        raise Launched

    before = list(logging.getLogger("callo").handlers)
    monkeypatch.setattr(acceptance, "callo_serve_argv", spy)
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 1024**3)
    suite = _solo_suite(tmp_path / "suite.jsonl")
    out = tmp_path / "o"
    try:
        acceptance.main(["--out", str(out), "--suite-file", str(suite),
                         "--rows", "s1_solo,s1_decode,kv_capacity,s1_fallback", *(["--cache", flag] if flag else [])])
        ctx = acceptance.Ctx(out, str(suite), 3, cache=(flag == "on"))
        with pytest.raises(Launched):
            acceptance._fallback_mode(ctx, out, "off")
    finally:
        for h in logging.getLogger("callo").handlers:
            if h not in before:
                logging.getLogger("callo").removeHandler(h)
    assert sorted(n for n, _ in argvs) == ["callo_decode", "callo_fallback_off", "callo_fallback_on", "callo_kv",
                                           "callo_main"]
    for _, argv in argvs:
        assert want in argv and never not in argv
    assert json.loads((out / "summary.json").read_text())["cache"] == (flag or "off")


def _probs(out, row, obj):
    import json
    (out / row).mkdir(parents=True, exist_ok=True)
    (out / row / "probs.json").write_text(obj if isinstance(obj, str) else json.dumps(obj))


def test_delta_p_report_reads_each_row_against_s1_solo_and_the_floor_against_the_repeat(tmp_path):
    """Three runs whose answers differ: the floor (solo vs repeat) and the prefill row's |Δp| are distinct and
    non-zero. Values are dyadic, so the means are exact."""
    from bench.acceptance import ROWS, delta_p_report
    _probs(tmp_path, "s1_solo", {"a/q": [0.75, 0.25], "b/q": [0.5, 0.25, 0.25]})
    _probs(tmp_path, "s1_solo_repeat", {"a/q": [0.625, 0.375], "b/q": [0.5, 0.25, 0.25]})
    _probs(tmp_path, "s1_prefill_32k", {"a/q": [0.25, 0.75], "b/q": [0.25, 0.5, 0.25]})
    ok = {"pass": None, "invalid": None}
    results = {"s1_solo": ok, "s1_solo_repeat": ok, "s1_prefill_32k": ok}
    vs, floor = delta_p_report(tmp_path, ROWS, results)
    assert floor == {"n": 2, "max": 0.125, "mean": 0.0625, "flips": 0}
    assert vs == {"s1_solo_repeat": floor, "s1_prefill_32k": {"n": 2, "max": 0.5, "mean": 0.375, "flips": 2}}


def test_delta_p_report_takes_only_rows_recorded_ok_and_needs_s1_solo(tmp_path):
    from bench.acceptance import ROWS, delta_p_report
    for row in ("s1_solo", "s1_solo_repeat", "s1_prefill_32k", "s1_prefill_128k", "s1_decode"):
        _probs(tmp_path, row, {"a/q": [0.75, 0.25]})
    ok = {"pass": None, "invalid": None}
    results = {"s1_solo": ok, "s1_solo_repeat": ok, "s1_prefill_32k": {"pass": False},
               "s1_prefill_128k": {"pass": None, "invalid": "prefill not proven"}}       # s1_decode: not recorded
    vs, floor = delta_p_report(tmp_path, ROWS, results)
    assert set(vs) == {"s1_solo_repeat"} and floor == {"n": 1, "max": 0.0, "mean": 0.0, "flips": 0}
    # the repeat without s1_solo (missing, failed, or its file gone): no comparison and no floor
    assert delta_p_report(tmp_path, ROWS, {"s1_solo_repeat": ok, "s1_prefill_32k": ok}) == ({}, None)
    assert delta_p_report(tmp_path, ROWS, {"s1_solo": {"pass": False, "error": "x"}, "s1_solo_repeat": ok}) == ({}, None)
    (tmp_path / "s1_solo" / "probs.json").unlink()
    assert delta_p_report(tmp_path, ROWS, results) == ({}, None)


def test_delta_p_report_records_a_bad_file_as_that_rows_error(tmp_path):
    from bench.acceptance import ROWS, delta_p_report
    _probs(tmp_path, "s1_solo", {"a/q": [0.75, 0.25]})
    _probs(tmp_path, "s1_solo_repeat", '{"a/q": [0.75, ')                              # truncated
    _probs(tmp_path, "s1_prefill_32k", {"a/q": [0.5, 0.25, 0.25]})                    # a length mismatch
    _probs(tmp_path, "s1_decode", {"a/q": [0.5, 0.5]})
    ok = {"pass": None, "invalid": None}
    vs, floor = delta_p_report(tmp_path, ROWS, dict.fromkeys(("s1_solo", "s1_solo_repeat", "s1_prefill_32k",
                                                              "s1_decode"), ok))
    assert "JSONDecodeError" in vs["s1_solo_repeat"]["error"] and floor == vs["s1_solo_repeat"]
    assert "ValueError" in vs["s1_prefill_32k"]["error"]
    assert vs["s1_decode"] == {"n": 1, "max": 0.25, "mean": 0.25, "flips": 0}
    _probs(tmp_path, "s1_solo", "not json")
    vs, floor = delta_p_report(tmp_path, ROWS, {"s1_solo": ok, "s1_decode": ok})
    assert "JSONDecodeError" in vs["s1_solo"]["error"] and set(vs) == {"s1_solo"} and floor is None


def test_a_bad_probs_file_does_not_stop_the_run_finishing(stub_callo, monkeypatch, tmp_path):
    import json

    from bench import acceptance
    monkeypatch.setattr(acceptance, "row_s1_prefill", lambda c, ctx, n_tokens, name, pass_p50_s:
                        _probs(ctx.out, name, "{trunc") or {"pass": None})
    suite = _solo_suite(tmp_path / "suite.jsonl")
    out = tmp_path / "o"
    assert acceptance.main(["--out", str(out), "--rows", "s1_solo,s1_prefill_32k", "--suite-file", str(suite),
                            "--s1-limit", "2"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert "finished" in summary and "error" in summary["delta_p_vs_solo"]["s1_prefill_32k"]


@pytest.mark.parametrize("first, second", [(["--model", "/m/tiel.gguf"], []), ([], ["--cache", "on"]),
                                           (["--cache", "on"], [])])
def test_a_resumed_run_refuses_a_different_model_or_cache(stub_callo, tmp_path, capsys, first, second):
    import json

    from bench import acceptance
    suite = _solo_suite(tmp_path / "suite.jsonl")
    out = tmp_path / "o"
    base = ["--out", str(out), "--rows", "s1_solo", "--suite-file", str(suite), "--s1-limit", "2"]
    assert acceptance.main([*base, *first]) == 0
    before = (out / "summary.json").read_text()
    stub_callo.clear()
    assert acceptance.main([*base, *second]) == 2
    assert stub_callo == [] and (out / "summary.json").read_text() == before
    assert "refusing to resume" in capsys.readouterr().out
    assert acceptance.main([*base, *first]) == 0                                    # the same settings resume
    assert json.loads((out / "summary.json").read_text())["rows"]["s1_solo"]["invalid"] is None


def test_an_older_summary_without_cache_or_model_resumes_as_the_defaults(stub_callo, tmp_path):
    import json

    from bench import acceptance
    out = tmp_path / "o"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"rows": {}}))
    suite = _solo_suite(tmp_path / "suite.jsonl")
    base = ["--out", str(out), "--rows", "s1_solo", "--suite-file", str(suite), "--s1-limit", "2"]
    assert acceptance.main([*base, "--cache", "on"]) == 2
    assert acceptance.main(base) == 0


@pytest.mark.parametrize("cache, cache_n, valid", [
    (True, 179, True),            # --cache on: a small shared prefix reused, the prefill still real
    (True, 664, True),            # at the 2 % bound (0.02 x 33219 = 664.38)
    (True, 665, False),
    (True, 20000, False),         # a large cached prefix: the prefill was not measured
    (False, 0, True),
    (False, 1, False),            # --cache off: any cached token is a failed proof, as before
    (False, 179, False)])
def test_the_foreign_prefill_proof_allows_a_small_cached_prefix_only_with_the_cache_on(cache, cache_n, valid):
    from bench import acceptance
    f = {"status": 200, "prompt_n": 33441, "cache_n": cache_n}
    why = acceptance.foreign_prefill_problem(f, 33219, cache=cache)
    assert (why is None) is valid
    if not valid:
        assert f"prompt_n 33441 vs 33219, cache_n {cache_n}" in why


@pytest.mark.parametrize("cache", [True, False])
def test_the_foreign_prefill_proof_still_needs_status_200_and_the_whole_prompt(cache):
    from bench import acceptance
    assert acceptance.foreign_prefill_problem({"status": 200, "prompt_n": 31000, "cache_n": 0}, 33219, cache=cache)
    assert acceptance.foreign_prefill_problem({"status": 500, "prompt_n": 33441, "cache_n": 0}, 33219, cache=cache)
    assert acceptance.foreign_prefill_problem({}, 33219, cache=cache)
    assert acceptance.foreign_prefill_problem({"status": 200, "prompt_n": 31559, "cache_n": None}, 33219,
                                              cache=cache) is None             # 0.95 x 33219 = 31558.05


def test_the_summary_records_the_callo_version(stub_callo, tmp_path):
    import json

    import callo
    from bench import acceptance
    suite = _solo_suite(tmp_path / "suite.jsonl")
    out = tmp_path / "o"
    assert acceptance.main(["--out", str(out), "--rows", "s1_solo", "--suite-file", str(suite), "--s1-limit", "2"]) == 0
    assert json.loads((out / "summary.json").read_text())["callo_version"] == callo.__version__


def test_callo_reads_its_version_from_the_serving_line():
    from bench import acceptance
    assert acceptance.version_from_log("x\ncallo 0.3.0 serving on http://127.0.0.1:8180 -> http://u (reserved ...)") \
        == "0.3.0"
    assert acceptance.version_from_log("no such line") is None


def test_a_resume_under_another_callo_version_is_refused_and_an_unversioned_summary_is_adopted(stub_callo, tmp_path,
                                                                                                capsys):
    import json

    import callo
    from bench import acceptance
    suite = _solo_suite(tmp_path / "suite.jsonl")
    out = tmp_path / "o"
    out.mkdir()
    base = ["--out", str(out), "--rows", "s1_solo", "--suite-file", str(suite), "--s1-limit", "2"]
    (out / "summary.json").write_text(json.dumps({"rows": {}, "callo_version": "0.2.9"}))
    before = (out / "summary.json").read_text()
    assert acceptance.main(base) == 2
    assert stub_callo == [] and (out / "summary.json").read_text() == before
    assert "refusing to resume" in capsys.readouterr().out and "0.2.9" in before
    (out / "summary.json").write_text(json.dumps({"rows": {}}))              # before callo_version was recorded
    assert acceptance.main(base) == 0
    assert json.loads((out / "summary.json").read_text())["callo_version"] == callo.__version__
    assert acceptance.main(base) == 0                                          # the same version resumes


# ---- the user's own paths: env-var defaults and the startup refusal ------------------------------------------------

def test_env_path_joins_its_variable_and_is_none_when_it_is_unset(monkeypatch):
    from bench import lib
    monkeypatch.delenv("CALLO_BENCH_TEST_ROOT", raising=False)
    assert lib.env_path("CALLO_BENCH_TEST_ROOT", "a.gguf") is None
    monkeypatch.setenv("CALLO_BENCH_TEST_ROOT", "")
    assert lib.env_path("CALLO_BENCH_TEST_ROOT") is None
    monkeypatch.setenv("CALLO_BENCH_TEST_ROOT", "/r")
    assert lib.env_path("CALLO_BENCH_TEST_ROOT") == "/r" and lib.env_path("CALLO_BENCH_TEST_ROOT", "m", "a.gguf") == "/r/m/a.gguf"


def test_unset_paths_names_each_flag_and_variable_once():
    from bench import lib
    assert lib.unset_paths(("--data", "/d.jsonl", lib.DATA_ENV), ("", "/m.gguf", lib.MODELS_ENV)) is None
    assert lib.unset_paths(("--data", None, lib.DATA_ENV), ("", None, lib.MODELS_ENV), ("", "", lib.MODELS_ENV),
                           ("", "/bin/llama-server", lib.SERVER_ENV)) == (
        "refusing to start: a required path is not set (pass --data or set CALLO_BENCH_DATA; set CALLO_BENCH_MODELS; "
        "see bench/README.md)")


def _refusal_cases(tmp_path):
    from bench import acceptance, acceptance_v02, check_cache_delimiters, wide_check
    from bench import acceptance_v03_asr as asr
    from bench import acceptance_v03_bias as bias
    from bench import acceptance_v03_deadline as deadline
    from bench import acceptance_v03_early_exit as early_exit
    from bench import acceptance_v03_think as think
    out = str(tmp_path / "out")
    return [
        (acceptance, {"SUITE": None}, ["--out", out, "--rows", "selftest"], "pass --suite-file or set CALLO_BENCH_DATA"),
        (acceptance_v02, {"TIEL": None}, ["--out", out], "set CALLO_BENCH_MODELS"),
        (asr, {"GEMMA": None}, ["--set", str(tmp_path / "set"), "--out", out], "pass --model or set CALLO_BENCH_MODELS"),
        (bias, {"FIT_DATA": None}, ["--out", out, "--expect-head", "abc1234"], "set CALLO_BENCH_DATA"),
        (think, {"OUT_DEFAULT": None}, [], "pass --out or set CALLO_BENCH_OUT"),
        (think, {"LLAMA_SERVER": None}, ["--out", out], "set CALLO_BENCH_LLAMA_SERVER"),
        (deadline, {"OUT_DEFAULT": None}, ["--parts", "gates"], "pass --out or set CALLO_BENCH_OUT"),
        (deadline, {"TRANSFER_TEST": None}, ["--out", out, "--parts", "calib"], "pass --data or set CALLO_BENCH_DATA"),
        (early_exit, {"OUT_DEFAULT": None}, ["--parts", "gates"], "pass --out or set CALLO_BENCH_OUT"),
        (early_exit, {"DEV": None}, ["--out", out, "--parts", "calib"], "pass --dev or set CALLO_BENCH_DATA"),
        (wide_check, {"OUT_DEFAULT": None}, [], "pass --out or set CALLO_BENCH_OUT"),
        (check_cache_delimiters, {"MASTER": None}, ["--model", "m.gguf", "--bodies", "b.jsonl", "--out", out],
         "pass --llama-server or set CALLO_BENCH_LLAMA_SERVER"),
    ]


@pytest.mark.parametrize("case", range(12))
def test_each_bench_refuses_at_startup_when_a_path_it_needs_is_unset(case, monkeypatch, capsys, tmp_path):
    """No private default: an unset path is a refusal (exit 2) naming its flag and variable, before any GPU check or
    write (nothing under --out)."""
    module, unset, argv, says = _refusal_cases(tmp_path)[case]
    for name, value in unset.items():
        monkeypatch.setattr(module, name, value)
    assert module.main(argv) == 2
    printed = capsys.readouterr()
    assert "refusing to start: a required path is not set (" in printed.out + printed.err
    assert says in printed.out + printed.err and not (tmp_path / "out").exists()
