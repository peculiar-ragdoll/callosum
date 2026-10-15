"""Think deadlines' serve surface: `/callo/stats.think.deadline`, the request log, the serve
flags with their refusals and startup line, and calibrate's refusal of a server that thinks under a default deadline."""
import hashlib
import json
import logging
import socket
import time

import pytest
from fake_callo import default_stats
from fake_upstream import FakeUpstream
from helpers import get_json, post_json, tiny_gguf
from test_deadline_serve import live, post, stats
from test_think import profile_dict
from test_think_cli import _profile, _serve
from test_think_serve import BODY

from callo import calibrate_bias
from callo.cli import deadline_startup_line
from callo.deadline import Estimators
from callo.proxy import RequestLog
from callo.think import load_think_profile, think_default_refusal

TWO = {**BODY, "questions": {**BODY["questions"], "r": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}}


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=10_000).start()
    u.think_real_time, u.think_ms_per_token, u.think_prompt_ms_per_token = True, 2.0, 0.05
    yield u
    u.stop()


def test_stats_think_deadline_reports_the_estimators_closes_and_windows(up):
    srv, _, st, svc = live(up, reserve_ms=40.0)
    st.deadline_default_ms = 0
    try:
        post(srv, {**BODY, "think_deadline_ms": 900})                      # a deadline cut
        post(srv, {**TWO, "think_deadline_ms": 1200})                      # two deliberate questions
        post(srv, {**BODY, "think_deadline_ms": 40})                       # the budget0 fallback
        post(srv, {**BODY, "think_budget": 0, "think_deadline_ms": 900})   # timed, not deliberate
        up.think_multi_tag = True
        svc.engine.think = None                                          # a model that cannot chunk
        post(srv, {**BODY, "think_deadline_ms": 900})
        d = stats(srv)["think"]["deadline"]
        assert (d["default_ms"], d["reserve_ms"], d["reserve_source"]) == (0, 40.0, "flag")
        assert d["decode_ms_per_token"]["p50"] is not None and set(d["stall_ms"]) == {"p50", "p95"}
        assert (d["requests"], d["questions"], d["questions_by_count"]) == (3, 4, {"1": 2, "2": 2, "3+": 0})
        assert d["closes"]["deadline"] == 3 and d["closes"]["budget0"] == 1 and d["closes"]["natural"] == 0
        assert d["met"] + d["missed"] == 6 and d["not_chunkable"] == 1 and d["rerouted"] == 0
        assert d["cont_reprefill"] == 0 and d["over_ms"]["n"] == 3 and d["chunks_mean"] > 1
        assert set(d["elapsed_ms"]) == {"p50", "p95", "p99", "p99.5", "max"} and d["elapsed_ms"]["max"] > 0
        assert d["think_tokens"]["by_count"]["2"]["p50"] > 0 and d["think_tokens"]["by_count"]["3+"]["p50"] is None
    finally:
        srv.shutdown()


def _abandon(srv, raw: bytes, after_s):
    s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10)
    s.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
              + str(len(raw)).encode() + b"\r\n\r\n" + raw)
    time.sleep(after_s)
    s.close()


def test_the_request_log_writes_one_line_per_request_with_the_body_hash_and_never_its_text(tmp_path, up):
    path = tmp_path / "req.jsonl"
    log = RequestLog(str(path))
    srv, *_ = live(up, request_log=log)
    try:
        body = {**TWO, "think_deadline_ms": 1200, "state": "a secret state"}
        status, headers, out = post(srv, body)
        post(srv, {**BODY, "think_deadline_ms": "bad"})                    # a 422 is logged too
        raw = json.dumps({**BODY, "think_deadline_ms": 5000}).encode()
        _abandon(srv, raw, 0.3)                                             # the client leaves mid-trace
        deadline = time.monotonic() + 5
        while len(path.read_text().splitlines()) < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        srv.shutdown()
        log.close()
    text = path.read_text()
    assert "secret" not in text and "state" not in text and "the state" not in text and "pick" not in text
    first, bad, gone = (json.loads(x) for x in text.splitlines())
    sent = json.dumps(body).encode()                                      # urllib sends exactly these bytes
    assert first["body_sha256"] == hashlib.sha256(sent).hexdigest() and first["seq"] == 1
    assert (first["status"], first["slot"], first["deadline_ms"], first["budget"]) == (200, 0, 1200, 0)
    assert all(list(q) == ["qid", "answer", "think_close", "k", "elapsed_ms", "prompt_tokens"]
               for q in first["questions"])
    assert [q["qid"] for q in first["questions"]] == ["q", "r"]
    assert [q["answer"] for q in first["questions"]] == [out["answers"]["q"]["choice"], out["answers"]["r"]["choice"]]
    assert all(q["think_close"] == "deadline" and q["k"] > 0 and q["prompt_tokens"] > 0 for q in first["questions"])
    assert first["served_ms"] >= max(q["elapsed_ms"] for q in first["questions"])
    assert first["t_written_unix_ms"] >= first["t_arr_unix_ms"] + first["served_ms"] - 5
    assert list(first) == ["seq", "body_sha256", "t_arr_unix_ms", "t_written_unix_ms", "served_ms", "status", "slot",
                           "deadline_ms", "budget", "questions"]
    assert bad["status"] == 422 and bad["questions"] == [] and bad["seq"] == 2
    assert gone["status"] == "cancelled" and gone["body_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("flags", [["--think-deadline", "-1"], ["--think-deadline", "abc"],
                                   ["--think-deadline", "600001"], ["--think-deadline-reserve", "1.5"],
                                   ["--think-deadline-reserve", "-3"], ["--request-log", "/nonexistent/dir/r.jsonl"]])
def test_phase_1_refusals_of_the_deadline_flags(tmp_path, up, caplog, flags):
    r = _serve(tmp_path, up, *flags)
    assert r.finish() == 1 and r.port is None and not up.requests and "ERROR" in caplog.text
    assert flags[0] in caplog.text
    if flags[0] == "--request-log":
        assert "cannot open it for appending" in caplog.text


def test_a_failing_log_fill_never_fails_the_served_answer(tmp_path, up, monkeypatch, caplog):
    import callo.proxy as proxy
    monkeypatch.setattr(proxy, "_log_questions", lambda *a: 1 / 0)
    log = RequestLog(str(tmp_path / "r.jsonl"))
    srv, *_ = live(up, request_log=log)
    try:
        status, _, out = post(srv, {**BODY, "think_deadline_ms": 900})
        assert status == 200 and "q" in out["answers"]
        deadline = time.monotonic() + 5
        while not (tmp_path / "r.jsonl").read_text() and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        srv.shutdown()
        log.close()
    (rec,) = (json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines())
    assert rec["status"] == 200 and rec["questions"] == []
    assert "cannot describe this request's questions" in caplog.text


def test_request_log_with_a_native_upstream_warns_that_nothing_is_logged(tmp_path, caplog):
    native = tiny_gguf(tmp_path / "kev.gguf", decision="kev")
    plain = FakeUpstream().start()
    try:
        r = _serve(tmp_path, plain, "--request-log", str(tmp_path / "n.jsonl"), gguf=native)
        try:
            assert "--request-log only covers /v1/systemone requests callo serves" in caplog.text
        finally:
            r.finish()
    finally:
        plain.stop()


def test_phase_2_refusals_native_a_model_that_cannot_think_and_a_multi_token_tag(tmp_path, up, caplog):
    native = tiny_gguf(tmp_path / "kev.gguf", decision="kev")
    r = _serve(tmp_path, up, "--think-deadline", "2000", gguf=native)
    assert r.finish() == 1 and "--think-deadline: the upstream is a native decision model" in caplog.text
    plain = FakeUpstream().start()
    try:
        caplog.clear()
        r = _serve(tmp_path, plain, "--think-deadline", "2000")
        assert r.finish() == 1 and "no thinking mode callo can read" in caplog.text
    finally:
        plain.stop()
    up.think_multi_tag = True
    caplog.clear()
    r = _serve(tmp_path, up, "--think-deadline", "2000")
    assert r.finish() == 1 and "think deadlines need a single-token close tag; this model's is 2 tokens" in caplog.text


def test_serve_names_the_deadlines_reach_and_publishes_it(tmp_path, up, caplog):
    caplog.set_level(logging.INFO, logger="callo")
    log = tmp_path / "r.jsonl"
    r = _serve(tmp_path, up, "--think-deadline", "1800", "--request-log", str(log))
    try:
        assert ("think deadline 1800 ms (all clients; think_budget 0 opts out): cap 4096 (--think-max), reserve "
                "learned (seed 100 ms from the default)") in caplog.text
        _, _, st = get_json(f"{r.url}/callo/stats")
        assert st["think"]["deadline"]["default_ms"] == 1800 and st["think"]["deadline"]["reserve_source"] == "default"
        assert think_default_refusal(st) is not None and "--think-deadline 1800" in think_default_refusal(st)
    finally:
        assert r.finish() == 0
    assert log.exists()
    caplog.clear()
    r = _serve(tmp_path, up, "--think-deadline", "1800", "--think-deadline-reserve", "60")
    try:
        assert "cap 4096 (--think-max), reserve 60 ms (--think-deadline-reserve)" in caplog.text
    finally:
        assert r.finish() == 0


def test_temperature_and_bias_calibrate_refuse_a_server_with_a_default_deadline():
    st = {**default_stats(), "think": {"default_budget": 0, "deadline": {"default_ms": 1500}}}
    assert "thinks by default under a deadline (--think-deadline 1500)" in think_default_refusal(st)
    with pytest.raises(calibrate_bias.Refusal, match="under a deadline"):
        calibrate_bias.check_preconditions(st, "http://x")
    assert think_default_refusal({"think": {"default_budget": 0, "deadline": {"default_ms": 0}}}) is None


def test_the_startup_line_names_the_bound_that_won_the_cap():
    """B* is the profile's largest budget but never above --think-max."""
    tp = load_think_profile(profile_dict())                           # budgets 0, 256, 1024
    est = Estimators()
    assert "cap 1024 (the profile's largest budget)" in deadline_startup_line(1500, tp, 4096, est)
    assert "cap 512 (--think-max)" in deadline_startup_line(1500, tp, 512, est)      # profile max 1024 > --think-max
    assert "cap 1024 (the profile's largest budget)" in deadline_startup_line(1500, tp, 1024, est)
    assert "cap 4096 (--think-max)" in deadline_startup_line(1500, None, 4096, est)


def test_a_deadline_request_through_the_real_serve_wiring_moves_the_shared_estimators(tmp_path, up):
    """One estimators object, built by Estimators.from_profile in cli.serve, is seeded from the profile, fed by the engine's
    chunks and published by stats: the seed shows before a request, the windows and counters move after one."""
    gguf = tiny_gguf(tmp_path / "m.gguf")
    prof, _ = _profile(tmp_path, up, gguf, latency={"read_forced_ms": {"p50": 20.0, "p95": 40.0, "p99": 60.0}})
    r = _serve(tmp_path, up, "--think-profile", str(prof), "--think-deadline", "900", gguf=gguf)
    try:
        d = get_json(f"{r.url}/callo/stats")[2]["think"]["deadline"]
        assert (d["reserve_ms"], d["reserve_source"]) == (70.0, "profile")           # from_profile's seed, not the default 100
        assert d["decode_ms_per_token"]["p50"] is None and d["closes"]["deadline"] == 0
        status, headers, out = post_json(f"{r.url}/v1/systemone", BODY)             # the serve-wide default deadline applies
        assert status == 200 and out["answers"]["q"]["type"] == "choice"
        (e,) = json.loads(headers["X-Callo-Timings"])
        assert e["think_close"] == "deadline" and e["think_deadline_ms"] == 900 and e["chunks"] >= 2
        d = get_json(f"{r.url}/callo/stats")[2]["think"]["deadline"]
        assert d["decode_ms_per_token"]["p50"] is not None                           # the engine fed the published estimators
        assert d["closes"]["deadline"] == 1 and d["requests"] == 1 and d["chunks_mean"] >= 2
    finally:
        assert r.finish() == 0


def test_the_docs_document_every_deadline_flag_and_serve_s_close_words():
    """The "Think deadlines" section (docs/guide/think-budgets.md; the README links it) names every new flag, the
    resolution table's rows, the opt-outs and serve's close words (`budget`, never the older `forced`, in the X-Callo-Timings
    list); the deadline X-Callo-Timings fields with the three `deadline_fallback` values,
    `compare --deadline`, B* never above `--think-max`, and a link (not a copy) to wide-choice's latency note on the
    shared c0 estimator (docs/guide/api.md). The pilot recipe for a client with a hard timeout names the request log
    and no private harness."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    guide = root / "docs" / "guide"
    readme = (root / "README.md").read_text()
    think = (guide / "think-budgets.md").read_text()
    api = (guide / "api.md").read_text()
    calibration = (guide / "calibration.md").read_text()
    compare = (guide / "compare.md").read_text()
    docs = "\n".join([readme, think, api, calibration, compare])
    assert "](docs/guide/think-budgets.md)" in readme and "think_deadline_ms" in readme
    section = think[think.index("## Think deadlines"):]
    for flag in ("--think-deadline MS", "--think-deadline-reserve MS", "--request-log FILE", "\"think_budget\": 0",
                 "\"think_deadline_ms\": 0", "deadline_fallback: \"budget0\"", "--s1-slots", "-b 512 -ub 512",
                 "callo compare P.think.json --deadline", "--converge-probe K"):
        assert flag in section, flag
    assert "--think-deadline D --request-log FILE" in section and "fetch_log.mjs" not in docs
    assert "| explicit 0 | D > 0 |" in section and "| B > 0 | D > 0 |" in section
    assert "never above `--think-max`" in section                       # a deadline never raises a cap
    assert "B* never exceeds `--think-max`" in section
    for field in ("`think_deadline_ms`", "`elapsed_ms`", "`deadline_met`", "`chunks`", "`deadline_fallback`"):
        assert field in section, field
    assert "`budget0` | `not_chunkable` | `media`" in section            # every fallback value, each explained
    assert "](api.md#wide-latency)" in section and "Estimators.c0" in section  # the wide note is linked ...
    assert section.count("770 ms") == 0                                  # ... not duplicated
    assert api.count('<a id="wide-latency"></a>') == 1 and docs.count('<a id="wide-latency"></a>') == 1
    tag = '<a id="wide-latency"></a>'
    assert api[api.index(tag) + len(tag):].startswith("**Latency and deadlines.**")
    assert "770 ms" in api[api.index(tag):]
    assert "would put the model on your accuracy/latency front, net of probe cost" in " ".join(section.split())
    for flag in ("--deadlines MS[,MS...]", "--converge-probe [K]", "--converge-cost N"):
        assert flag in calibration, flag
    for flag in ("[--deadline MS[,MS...] [--budget B]", "[--latency mean|p50|p95] [--limit STAT=X ...] [--price R]"):
        assert flag in compare, flag
    assert "[--latency mean|p50|p95] [--limit STAT=X ...] [--price R]" in calibration
    flat = " ".join(docs.split())
    assert "`think_close` (`natural` | `forced` | `eos`)" not in flat
    # the timed entries, the fallback's scope, the chunk seeds and c0's seed chain, as the code has them
    flat = " ".join(think.split())
    assert "every entry timed under a request deadline gains `think_deadline_ms`, `elapsed_ms` and `deadline_met`" in flat
    assert "an explicit `think_budget: 0` under a deadline carries no `deadline_fallback`" in flat
    assert "only when the deadline's thinking fell back" in flat and "only when it got no thinking" not in flat
    assert "`think_seed(digest, qid, i)`" in flat and "where the chunks end" in flat
    assert "each chunk draws its own seed" not in " ".join(docs.split())
    assert "the window's p95 once 8 readouts exist, else the profile's budget-0 p95, else 300 ms" in flat
