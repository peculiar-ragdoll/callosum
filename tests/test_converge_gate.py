"""The early-exit gate (opt-in, experimental): one budget-0 question, asked before thinking, decides whether early exit
may act on a request. `think_converge_gate` (true, false or null) and `callo serve --think-converge-gate`; the gate runs
only when early exit runs; a yes thinks to the cap with no probes, a no early-exits as before, a failed read skips early
exit and says why; the seeds ignore the field; X-Callo-Timings, /callo/stats and the request log report it; with the
gate off or absent everything is as before."""
import json
import math
import re
import threading
import time

import pytest
from fake_upstream import _TEXT_ID, FakeUpstream
from test_converge_serve import CONVERGE_KEYS, TWO, profile
from test_deadline_serve import CHUNKED_KEYS, entries, gens, live, post, stats
from test_profile import profile_dict as bias_profile
from test_think_serve import BODY, THINK_KEYS, V02_KEYS

from callo.errors import UpstreamUnavailable
from callo.profile import load_profile
from callo.proxy import (
    GATE_BUSY,
    GATE_CRITERIA,
    GATE_ERROR_MAX,
    GATE_INSTRUCTIONS,
    GATE_QID,
    GATE_SKIP_P,
    NO_REPLY,
    RequestLog,
    gate_p,
)
from callo.systemone import THINK_CONVERGE_GATE_ERROR, InvalidRequest, native_think_refusal, parse_request

BODY_DIGEST = "82342c1785847349a4f8d51c9d4a6f04b090e1c4cd3e16e89249c68004ce48b3"     # test_converge_request's
GATE_TEXT = ("Before answering, would you need to apply several stated rules or conditions step by step to reach the "
             "answer?")


# ---- the field ------------------------------------------------------------------------------------------------------

def test_the_gate_question_is_the_one_the_spike_validated():
    assert GATE_INSTRUCTIONS == GATE_TEXT and GATE_CRITERIA == {"a": "yes", "b": "no"} and GATE_SKIP_P == 0.5
    assert THINK_CONVERGE_GATE_ERROR == "think_converge_gate must be true, false or null"


@pytest.mark.parametrize("bad", [0, 1, 0.5, "true", "yes", [True], {}])
def test_a_value_other_than_true_false_or_null_is_422(bad):
    with pytest.raises(InvalidRequest, match=re.escape(THINK_CONVERGE_GATE_ERROR)):
        parse_request({**BODY, "think_converge_gate": bad})


def test_null_and_absent_take_the_server_default():
    assert parse_request(BODY).think_converge_gate is None
    assert parse_request({**BODY, "think_converge_gate": None}).think_converge_gate is None
    assert parse_request({**BODY, "think_converge_gate": True}).think_converge_gate is True
    assert parse_request({**BODY, "think_converge_gate": False}).think_converge_gate is False


def test_the_field_is_in_the_digest_but_never_in_the_seeds():
    assert parse_request(BODY).seed_digest == BODY_DIGEST
    for v in (True, False, None):
        r = parse_request({**BODY, "think_converge_gate": v})
        assert r.digest != BODY_DIGEST and r.seed_digest == BODY_DIGEST
        assert parse_request({**BODY, "think_converge": 0.95, "think_converge_gate": v}).seed_digest == BODY_DIGEST
    assert parse_request({**BODY, "think_converge_gate": True, "think_budget": 64}).seed_digest != BODY_DIGEST


def test_a_native_upstream_refuses_only_true():
    assert "think_converge_gate must be false or absent" in native_think_refusal({**BODY, "think_converge_gate": True})
    assert native_think_refusal({**BODY, "think_converge_gate": False}) is None
    assert native_think_refusal({**BODY, "think_converge_gate": None}) is None
    assert native_think_refusal({**BODY, "think_converge_gate": 1}) == THINK_CONVERGE_GATE_ERROR


# ---- the service ----------------------------------------------------------------------------------------------------

@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=10_000).start()
    yield u
    u.stop()


def says(up, yes: bool):
    """The fake answers the gate question yes (P(yes) near 1) or no (near 0); every other question as before."""
    up.bias.content.update({"yes": 4.0 if yes else -4.0, "no": -4.0 if yes else 4.0})


def gated(up, *, default=False, **kw):
    srv, router, st, svc = live(up, profile=profile(), think_converge_gate=default, **kw)
    st.converge_gate_default = default                  # serve sets it from --think-converge-gate
    return srv, router, st, svc


def gate_reads(up):
    """The upstream readouts of the gate question (its instructions are in the rendered prompt)."""
    return [r for r in up.requests if r.path == "/completion" and (r.json or {}).get("n_probs")
            and isinstance(r.json.get("prompt"), list) and r.json["prompt"] and r.json["prompt"][0] >= _TEXT_ID
            and GATE_TEXT in up._texts[r.json["prompt"][0] - _TEXT_ID]]


def one(srv, body):
    status, headers, out = post(srv, body)
    assert status == 200, out
    (e,) = entries(headers)
    return e


def test_a_no_early_exits_as_before_and_says_so(up):
    says(up, False)
    srv, *_ = gated(up)
    try:
        e = one(srv, {**BODY, "think_converge": 0.9, "think_converge_gate": True})
        assert list(e) == [*THINK_KEYS, *CONVERGE_KEYS, "converge_gate"] and e["think_close"] == "converged"
        g = e["converge_gate"]
        assert list(g) == ["p", "ms", "skip"] and g["skip"] is False and g["p"] < 0.5 and g["ms"] >= 0
        (read,) = gate_reads(up)
        assert all(x["id_slot"] == read.json["id_slot"] for x in gens(up))       # the slot the request thinks on
        assert up.requests.index(read) < up.requests.index(next(r for r in up.requests if r.json in gens(up)))
        assert [x["n_predict"] for x in gens(up)] == [32] * e["probes"]
    finally:
        srv.shutdown()


def test_a_yes_thinks_to_the_cap_with_no_probes(up):
    says(up, True)
    srv, *_ = gated(up)
    try:
        e = one(srv, {**BODY, "think_converge": 0.9, "think_converge_gate": True})
        assert list(e) == [*THINK_KEYS, "converge_gate"] and e["think_close"] == "budget"
        assert (e["think_budget"], e["think_tokens"]) == (1024, 1024)               # B*, the cap early exit had
        g = e["converge_gate"]
        assert g["skip"] is True and g["p"] >= 0.5 and len(gate_reads(up)) == 1
        reads = [r for r in up.requests if r.path == "/completion" and (r.json or {}).get("n_probs")]
        assert len(reads) == 2                                     # the gate and the answer's readout: no probe
        assert sum(x["n_predict"] for x in gens(up)) == 1024 and all(x["n_predict"] > 32 for x in gens(up))
    finally:
        srv.shutdown()


def test_the_server_default_gates_every_early_exit_request_and_false_opts_out(up):
    says(up, True)
    srv, *_ = gated(up, default=True, think_converge=0.9)
    try:
        assert one(srv, BODY)["converge_gate"]["skip"] is True
        up.requests.clear()
        e = one(srv, {**BODY, "think_converge_gate": False})
        assert e["think_close"] == "converged" and "converge_gate" not in e and not gate_reads(up)
        up.requests.clear()
        assert one(srv, {**BODY, "think_converge_gate": None})["converge_gate"]["skip"] is True     # null: the default
    finally:
        srv.shutdown()


def test_the_gate_runs_only_when_early_exit_does(up):
    says(up, True)
    srv, *_ = gated(up, default=True)
    try:
        for body, keys in (({**BODY}, V02_KEYS),                                         # early exit off
                           ({**BODY, "think_converge_gate": True}, V02_KEYS),
                           ({**BODY, "think_converge": 0.97, "think_converge_gate": True}, [*V02_KEYS, "converge"]),
                           ({**BODY, "think_converge": 0.9, "think_budget": 0}, V02_KEYS),   # opts out of thinking
                           ({**BODY, "think_budget": 256, "think_converge_gate": True}, THINK_KEYS)):
            up.requests.clear()
            e = one(srv, body)
            assert list(e) == keys and not gate_reads(up), body
        assert stats(srv)["think"]["converge"]["gate"]["asked"] == 0
    finally:
        srv.shutdown()


def test_one_gate_read_per_request_on_every_think_entry_and_under_a_deadline(up):
    says(up, False)
    srv, *_ = gated(up, default=True)
    try:
        status, headers, _ = post(srv, {**TWO, "think_converge": 0.9, "think_deadline_ms": 600_000})
        assert status == 200 and len(gate_reads(up)) == 1
        a, b = entries(headers)
        assert list(a) == list(b) == [*CHUNKED_KEYS, *CONVERGE_KEYS, "converge_gate"]
        assert a["converge_gate"] == b["converge_gate"] and a["converge_gate"]["skip"] is False
        up.requests.clear()
        says(up, True)
        status, headers, _ = post(srv, {**TWO, "think_converge": 0.9, "think_deadline_ms": 600_000})
        assert status == 200 and len(gate_reads(up)) == 1
        assert all(list(e) == [*CHUNKED_KEYS, "converge_gate"] and e["think_close"] == "budget"
                   for e in entries(headers))
        single = {**BODY, "questions": {"s": {"type": "choice", "criteria": {"a": "only"}}}}
        up.requests.clear()
        e = one(srv, {**single, "think_converge": 0.9})          # nothing to think: no gate read, no gate field
        assert "converge_gate" not in e and not gate_reads(up)
    finally:
        srv.shutdown()


def _failing(svc, message):
    real = svc.engine.answer

    def answer(state, q, **kw):
        if q.qid == GATE_QID:
            raise UpstreamUnavailable(message)
        return real(state, q, **kw)

    svc.engine.answer = answer


def test_a_failed_read_skips_early_exit_and_records_why(up):
    srv, _, _, svc = gated(up, default=True)
    _failing(svc, "llama-server unavailable: boom")
    try:
        e = one(srv, {**BODY, "think_converge": 0.9})
        g = e["converge_gate"]
        assert list(g) == ["p", "ms", "skip", "error"] and g["p"] is None and g["skip"] is True
        assert g["error"] == "llama-server unavailable: boom" and e["think_close"] == "budget" and "probes" not in e
        gb = stats(srv)["think"]["converge"]["gate"]
        assert (gb["asked"], gb["skipped"], gb["errors"]) == (1, 1, 1) and gb["read_ms"] == {"p50": None, "p95": None}
        _failing(svc, "x" * 500)
        assert one(srv, {**BODY, "think_converge": 0.9})["converge_gate"]["error"] == "x" * GATE_ERROR_MAX == "x" * 200
    finally:
        srv.shutdown()


def test_under_a_deadline_a_busy_slot_bounds_the_gate_s_wait_by_the_share(up):
    """The gate waits for the slot no longer than the first question's T_q - c_0; then it skips early exit, counts no
    section timeout, and the question's own bounded wait takes over."""
    srv, router, st, svc = gated(up, default=True)
    slot = list(router.reserved)[0]
    section = router.section(slot)
    assert section.acquire(exclusive=True, timeout=1.0)          # a trace of another request holds the slot
    threading.Timer(2.0, lambda: section.release(exclusive=True)).start()
    try:
        t0 = time.monotonic()
        status, headers, out = post(srv, {**BODY, "think_converge": 0.9, "think_deadline_ms": 800})
        assert status == 200, out
        (e,) = entries(headers)
        g = e["converge_gate"]
        assert (g["skip"], g["p"], g["error"]) == (True, None, GATE_BUSY) and g["ms"] < 1000
        assert GATE_BUSY == "the slot stayed busy past the deadline share" and not gate_reads(up)
        assert e["deadline_fallback"] == "budget0" and "converge" not in e and time.monotonic() - t0 < 4.5
        s = stats(srv)
        assert s["s1"]["section_timeouts"] == 0 and s["think"]["converge"]["gate"]["errors"] == 1
    finally:
        srv.shutdown()


def test_a_client_that_leaves_during_the_gate_s_wait_ends_the_request(up):
    srv, router, st, svc = gated(up, default=True)
    section = router.section(list(router.reserved)[0])
    assert section.acquire(exclusive=True, timeout=1.0)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    try:
        code, out, _ = svc.respond({**BODY, "think_converge": 0.9}, cancel=cancel)
        assert (code, out) == (NO_REPLY, {}) and not gate_reads(up) and not gens(up)
        assert st.snapshot(router, None)["think"]["converge"]["gate"]["asked"] == 0
    finally:
        section.release(exclusive=True)
        srv.shutdown()


def test_the_gate_reads_the_model_s_own_p_yes_never_debiased(up):
    """A position-bias profile whose 2-option entry would flip the gate's answer is not applied to it."""
    says(up, False)
    srv, _, _, svc = gated(up, default=True)
    svc.engine.position_bias = load_profile(bias_profile({2: [0.0001, 0.9999]}))
    try:
        e = one(srv, {**BODY, "think_converge": 0.9})
        g = e["converge_gate"]
        assert g["skip"] is False and g["p"] < 0.5 and e["think_close"] == "converged"
        two = {**BODY, "questions": {"g": {"type": "choice", "instructions": GATE_TEXT, "criteria": dict(GATE_CRITERIA)}}}
        status, _, out = post(srv, two)                           # the same question served: debiased, a yes
        assert status == 200 and out["answers"]["g"]["probabilities"]["a"] > 0.5
    finally:
        srv.shutdown()


def test_a_no_serves_the_same_answer_and_trace_as_no_gate_field(up):
    """Apart from the gate's read and keys, a gated request answered no is the request without the field."""
    def run(body):
        u = FakeUpstream(think="preopened", think_len=10_000).start()
        says(u, False)
        srv, *_ = gated(u)
        try:
            status, headers, out = post(srv, body)
            (e,) = entries(headers)
            reads = [r.json for r in u.requests if r.path == "/completion" and (r.json or {}).get("n_probs")
                     and r not in gate_reads(u)]
            keep = ("think_budget", "think_tokens", "think_close", "temperature", "calibrated", *CONVERGE_KEYS)
            # the fake numbers each text it tokenizes in order, so the gate's text shifts the ids: compare the text
            trace = [{**g, "prompt": [u._texts[g["prompt"][0] - _TEXT_ID], *g["prompt"][1:]]} for g in gens(u)]
            return status, out, trace, len(reads), {k: e.get(k) for k in keep}, e.get("converge_gate")
        finally:
            srv.shutdown()
            u.stop()

    ee = {**BODY, "think_converge": 0.9}
    *plain, none = run(ee)
    *gated_no, g = run({**ee, "think_converge_gate": True})
    assert none is None and g["skip"] is False and plain == gated_no


def test_missing_probabilities_are_a_failed_read():
    assert gate_p([]) is None
    assert gate_p([0.7, 0.3]) == 0.7
    for bad in ([math.nan, 0.5], [math.inf, 0.0], [1.5, -0.5], ["x", 0.1], [True, False], [None, 1.0]):
        assert gate_p(bad) is None, bad


def test_the_stats_count_asks_skips_errors_and_the_read_window(up):
    srv, _, _, svc = gated(up, default=True, think_converge=0.9)
    try:
        gb = stats(srv)["think"]["converge"]["gate"]
        assert gb == {"default": True, "asked": 0, "skipped": 0, "errors": 0, "read_ms": {"p50": None, "p95": None}}
        says(up, True)
        one(srv, BODY)
        says(up, False)
        one(srv, BODY)
        cv = stats(srv)["think"]["converge"]
        gb = cv["gate"]
        assert (gb["asked"], gb["skipped"], gb["errors"]) == (2, 1, 0) and 0 < gb["read_ms"]["p50"] <= gb["read_ms"]["p95"]
        assert (cv["requests"], cv["questions"]) == (1, 1)            # the skipped request ran no early-exit rule
    finally:
        srv.shutdown()


def test_the_request_log_names_the_gate_next_to_converge(tmp_path, up):
    says(up, True)
    log = RequestLog(str(tmp_path / "r.jsonl"))
    srv, *_ = gated(up, request_log=log)
    try:
        one(srv, {**BODY, "think_converge": 0.9, "think_converge_gate": True})
        one(srv, {**BODY, "think_converge": 0.9})
        deadline = time.monotonic() + 5
        while len((tmp_path / "r.jsonl").read_text().splitlines()) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        srv.shutdown()
        log.close()
    a, b = (json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines())
    assert list(a)[-3:] == ["converge", "converge_gate", "questions"] and a["converge"] == 0.9
    assert a["converge_gate"]["skip"] is True and a["questions"][0]["think_close"] == "budget"
    assert list(b)[-2:] == ["converge", "questions"]


def _masked(headers):
    return re.sub(r'"(ms|gen_ms|elapsed_ms)":-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', r'"\1":#', headers["X-Callo-Timings"])


def test_with_the_gate_off_or_absent_early_exit_is_byte_identical(up):
    """`think_converge_gate` false (or a gate-on server's request with early exit off) changes nothing: the same
    upstream calls, the same X-Callo-Timings (wall clocks masked), the same reply."""
    def run(body, default):
        u = FakeUpstream(think="preopened", think_len=10_000).start()
        srv, *_ = gated(u, default=default)
        try:
            status, headers, out = post(srv, body)
            return status, _masked(headers), out, [[r.path, r.body.decode()] for r in u.requests]
        finally:
            srv.shutdown()
            u.stop()

    ee = {**BODY, "think_converge": 0.9}
    base = run(ee, False)
    assert base == run({**ee, "think_converge_gate": False}, False) == run({**ee, "think_converge_gate": False}, True)
    plain = run({**BODY, "think_budget": 256}, False)
    assert plain == run({**BODY, "think_budget": 256}, True)


# ---- callo serve --think-converge-gate ------------------------------------------------------------------------------

def test_the_flag_is_off_by_default():
    from callo.cli import build_parser

    argv = ["serve", "--upstream", "http://127.0.0.1:8199"]
    assert build_parser().parse_args(argv).think_converge_gate is False
    assert build_parser().parse_args([*argv, "--think-converge-gate"]).think_converge_gate is True


def _post_url(url, body):
    import urllib.request

    req = urllib.request.Request(f"{url}/v1/systemone", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, dict(r.headers), json.loads(r.read())


@pytest.fixture
def small():
    u = FakeUpstream(think="preopened", think_len=5).start()
    yield u
    u.stop()


def test_serve_says_the_gate_is_on_publishes_it_and_gates_early_exit(tmp_path, small, caplog):
    import logging

    from helpers import get_json, tiny_gguf
    from test_converge_serve import block
    from test_think_cli import _profile, _serve

    caplog.set_level(logging.INFO, logger="callo")
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, small, gguf, converge=block())
    r = _serve(tmp_path, small, "--think-profile", str(p), "--think-converge-gate", gguf=gguf)
    try:
        assert ("think converge gate on (experimental; think_converge_gate false opts a request out): before an "
                "early-exit request thinks, one budget-0 question asks whether it needs several stated rules applied "
                "step by step, and a yes thinks to the cap without early exit") in caplog.text
        _, _, st = get_json(f"{r.url}/callo/stats")
        assert st["think"]["converge"]["gate"]["default"] is True
        status, headers, _ = _post_url(r.url, {**BODY, "think_converge": 0.95})
        assert status == 200 and "converge_gate" in entries(headers)[0]
    finally:
        assert r.finish() == 0
    caplog.clear()
    r = _serve(tmp_path, small, "--think-profile", str(p), gguf=gguf)
    try:
        assert "think converge gate" not in caplog.text
        _, _, st = get_json(f"{r.url}/callo/stats")
        assert st["think"]["converge"]["gate"]["default"] is False
    finally:
        assert r.finish() == 0


def test_without_early_exit_to_gate_serve_warns_and_starts(tmp_path, small, caplog):
    from test_think_cli import _serve

    r = _serve(tmp_path, small, "--think-converge-gate")
    try:
        assert r.port and ("WARNING: --think-converge-gate: early exit cannot run on this server (no think profile with "
                           "a usable convergence block), so the gate never acts") in caplog.text
    finally:
        assert r.finish() == 0


def test_a_model_that_cannot_chunk_warns_that_the_gate_never_acts(tmp_path, small, caplog):
    from helpers import tiny_gguf
    from test_converge_serve import block
    from test_think_cli import _profile, _serve

    from callo.cli import gate_startup_line
    from callo.converge import serve_rules

    rules = serve_rules(block())[0]
    assert gate_startup_line(rules).startswith("think converge gate on")
    assert gate_startup_line(rules, chunkable=False) == (
        "WARNING: --think-converge-gate: early exit cannot run on this model (no thinking mode callo can read, or a "
        "close tag of several tokens), so the gate never acts")
    small.think_multi_tag = True
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, small, gguf, converge=block())
    r = _serve(tmp_path, small, "--think-profile", str(p), "--think-converge-gate", gguf=gguf)
    try:
        assert r.port and "early exit cannot run on this model" in caplog.text
    finally:
        assert r.finish() == 0


def test_a_native_upstream_refuses_the_flag(tmp_path, small, caplog):
    from helpers import tiny_gguf
    from test_think_cli import _serve

    r = _serve(tmp_path, small, "--think-converge-gate", gguf=tiny_gguf(tmp_path / "kev.gguf", decision="kev"))
    assert r.finish() == 1 and "--think-converge-gate: the upstream is a native decision model" in caplog.text


# ---- the public docs ------------------------------------------------------------------------------------------------

def test_the_docs_document_the_gate_and_carry_no_placeholder():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    guide = root / "docs" / "guide"
    think = (guide / "think-budgets.md").read_text(encoding="utf-8")
    section = think[think.index("### Early-exit gate (experimental)"):think.index("### What it does not promise")]
    flat = " ".join(section.split())
    assert think.index("## Early exit") < think.index("### Early-exit gate (experimental)")
    for text in (f"> {GATE_TEXT}", "P(yes) of at least 0.5", "about 150 ms", "`--think-converge-gate`",
                 "`think_converge_gate`", "The gate acts only when early exit runs for the request",
                 "`converge_gate`", '"skip": true | false', '"error": "<reason>"', "/callo/stats.think.converge.gate",
                 "`asked`", "`skipped`", "`errors`", "`read_ms`", "The startup log",
                 "the question was chosen after seeing which kind of question early exit fails on",
                 "validated on one model (Tiel-Coder-35B-A3B) and one question mix"):
        assert text in flat or text in section, text
    assert "`think_converge_gate`" in (guide / "api.md").read_text(encoding="utf-8")
    assert "| `--think-converge-gate` | off |" in (guide / "serving.md").read_text(encoding="utf-8")
    assert "`think.converge.gate`" in (guide / "stats.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "--think-converge-gate" in readme
    for path in (root / "README.md", *guide.glob("*.md")):
        assert "measured result below" not in path.read_text(encoding="utf-8"), path.name
