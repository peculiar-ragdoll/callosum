"""The service layer of think budgets through a real proxy against the thinking fake: the effective
budget and temperature, the S1 section, cancellation, the statuses, X-Callo-Timings and /callo/stats.think."""
import base64
import http.client
import json
import socket
import string
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest
from fake_upstream import FakeUpstream
from test_think import profile_dict

from callo.engine import Engine, SurfaceSets
from callo.errors import Cancelled, InvalidInput, ThinkTimeout, ThinkTooLarge, UpstreamUnavailable
from callo.prompt import TemplateClient, TemplateError
from callo.proxy import NO_REPLY, ProxyConfig, Stats, SystemOneService, make_server
from callo.scorer import LlamaCppForwardScorer, make_http_post
from callo.slots import SlotRouter
from callo.think import frame_from_template, load_think_profile, parse_sampling, resolve_sampling
from callo.vision import encode_png

TOKENS = list(string.ascii_uppercase) + ["Ġ" + c for c in string.ascii_uppercase] + ["**", "Ġthe", "x"]
DELIM = "<|im_start|>user"
SAMPLING = resolve_sampling(parse_sampling("fixed"))
BODY = {"state": "the state", "questions": {"q": {"type": "choice", "instructions": "pick",
                                                  "criteria": {"a": "one", "b": "two", "c": "three"}}}}
TWO = {**BODY, "questions": {**BODY["questions"], "r": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}}
PNG = "data:image/png;base64," + base64.b64encode(encode_png(1, 1, b"\x00\x00\x00")).decode()
IMG = {**BODY, "images": [PNG]}
V02_KEYS = ["q", "kind", "cache_n", "prompt_n", "prompt_ms", "tokens", "stop", "ms", "miss", "passes"]
THINK_KEYS = [*V02_KEYS, "think_budget", "think_tokens", "think_close", "gen_ms", "gen_prompt_n", "read_prompt_n",
              "temperature", "calibrated"]


def _engine(up, *, temperature=1.0, delimiter=DELIM, s1_timeout=5.0):
    post = make_http_post(timeout=10)
    e = Engine(LlamaCppForwardScorer(up.url, http_post=post), TemplateClient(up.url, http_post=post),
               SurfaceSets(TOKENS, "gpt2"), user_delimiter=delimiter, temperature=temperature, vision=up.vision,
               media_marker=up.props.get("media_marker") if up.vision else None, s1_timeout=s1_timeout)
    e.think, e.think_reason = frame_from_template(
        lambda thinking: e.template.render([{"role": "user", "content": "x"}], thinking=thinking), e.scorer.tokenize)
    e.n_ctx = up.props["default_generation_settings"]["n_ctx"]
    e.think_sampling = SAMPLING.pinned
    return e


def _live(up, *, s1_slots=1, think_default=0, think_max=4096, profile=None, s1_timeout=5.0, temperature=1.0,
          budget_keys=(256,)):
    """A real proxy + service + engine against the fake; returns (server, router, stats, service)."""
    engine = _engine(up, temperature=temperature, s1_timeout=s1_timeout)
    router, stats = SlotRouter(4, s1_slots), Stats()
    stats.think = {"capable": engine.think is not None, "reason": engine.think_reason, "default_budget": think_default,
                   "max": think_max, "profile": profile.info() if profile else None}
    stats.think_budget_keys = set(budget_keys) | ({think_default} - {0})
    svc = SystemOneService(engine, router, stats, model_name="fake-9b", think_default=think_default,
                           think_max=think_max, think_profile=profile, s1_timeout=s1_timeout)
    srv = make_server(ProxyConfig(upstream=up.url, router=router, stats=stats, systemone=svc.respond,
                                  systemone_cancel=True), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, router, stats, svc


def _post(srv, body, headers=None, path="/v1/systemone"):
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}{path}",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


def _timings(headers):
    return json.loads(headers["X-Callo-Timings"])


def _gens(up):
    return [r.json for r in up.requests if r.path == "/completion" and r.json.get("n_predict", 1) > 1]


def _stats(srv):
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/callo/stats", timeout=10) as r:
        return json.loads(r.read())


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=5).start()
    yield u
    u.stop()


def test_the_request_field_turns_a_deliberate_readout_on_and_its_timings_carry_the_think_fields(up):
    srv, _, _, _ = _live(up)
    try:
        status, headers, out = _post(srv, {**BODY, "think_budget": 256})
        assert status == 200 and headers["X-Callo-Slot"] == "0" and out["answers"]["q"]["type"] == "choice"
        (g,) = _gens(up)
        assert g["n_predict"] == 256 and g["id_slot"] == 0 and g["temperature"] == 0.6
        (entry,) = _timings(headers)
        assert list(entry) == THINK_KEYS
        assert (entry["think_budget"], entry["think_tokens"], entry["think_close"]) == (256, 6, "natural")
        assert entry["read_prompt_n"] == 2 and entry["temperature"] == 1.0 and entry["calibrated"] is False
        st = _stats(srv)["think"]
        assert (st["requests"], st["natural"], st["budget"], st["uncalibrated"]) == (1, 1, 0, 1)
        assert st["questions_by_budget"] == {"256": 1, "other": 0} and st["think_tokens_mean"] == 6
        assert st["gen_p50_ms"] is not None and st["capable"] is True
    finally:
        srv.shutdown()


def test_budget_0_entries_are_byte_identical_to_v03s_and_the_default_applies_only_without_the_field(up):
    srv, _, _, _ = _live(up, think_default=256)
    try:
        _, headers, _ = _post(srv, {**BODY, "think_budget": 0})
        (entry,) = _timings(headers)
        assert list(entry) == V02_KEYS and not _gens(up)                     # 0 overrides the default
        _, headers, _ = _post(srv, BODY)
        assert list(_timings(headers)[0]) == THINK_KEYS and len(_gens(up)) == 1
        assert _stats(srv)["think"]["questions_by_budget"] == {"256": 1, "other": 0}
    finally:
        srv.shutdown()


def test_think_max_a_media_request_and_a_model_without_a_thinking_mode_are_422(up):
    srv, _, _, _ = _live(up, think_max=512)
    try:
        status, _, out = _post(srv, {**BODY, "think_budget": 1024})
        assert status == 422 and out["detail"] == "think_budget 1024 exceeds this server's --think-max 512"
        status, _, out = _post(srv, {**BODY, "think_budget": "x"})
        assert status == 422 and out["detail"].startswith("think_budget must be an integer")
    finally:
        srv.shutdown()
    vup = FakeUpstream(think="preopened", think_len=5, vision=True).start()
    try:
        srv, _, _, _ = _live(vup, think_default=256)
        try:
            status, _, out = _post(srv, {**IMG, "think_budget": 64})
            assert status == 422 and out["detail"] == "deliberate readouts are text-only in this version"
            status, headers, _ = _post(srv, IMG)                           # inherits the default: served at 0
            assert status == 200 and _timings(headers)[0]["kind"] == "image" and not _gens(vup)
            assert _stats(srv)["think"]["skipped_media"] == 1
        finally:
            srv.shutdown()
    finally:
        vup.stop()
    plain = FakeUpstream().start()                                           # the default fake: not capable
    try:
        srv, _, _, _ = _live(plain)
        try:
            status, _, out = _post(srv, {**BODY, "think_budget": 64})
            assert status == 422 and "no thinking mode (not_preopened)" in out["detail"]
            assert _post(srv, BODY)[0] == 200
            assert _stats(srv)["think"]["capable"] is False
        finally:
            srv.shutdown()
    finally:
        plain.stop()


def test_the_profile_supplies_t_b_and_an_uncalibrated_budget_takes_the_nearest_lower_one(up):
    tp = load_think_profile(profile_dict())
    srv, _, _, _ = _live(up, profile=tp, budget_keys=(256, 1024))
    try:
        (e,) = _timings(_post(srv, {**BODY, "think_budget": 256})[1])
        assert (e["temperature"], e["calibrated"]) == (1.1, True)
        (e,) = _timings(_post(srv, {**BODY, "think_budget": 512})[1])
        assert (e["temperature"], e["calibrated"]) == (1.1, False)
        (e,) = _timings(_post(srv, {**BODY, "think_budget": 2048})[1])
        assert (e["temperature"], e["calibrated"]) == (0.9, False)
        st = _stats(srv)["think"]
        assert st["uncalibrated"] == 2 and st["questions_by_budget"] == {"256": 1, "1024": 0, "other": 2}
        assert st["profile"]["budgets"] == [0, 256, 1024]
    finally:
        srv.shutdown()


def test_a_profile_with_t_past_the_old_grid_ceiling_loads_and_serves(up):
    """The fit grid reaches 16 since 0.3.0: a profile with T = 12 loads, and serves at T = 12."""
    tp = load_think_profile(profile_dict(budgets={"0": {"T": 12.0}, "256": {"T": 12.0}}))
    assert tp.t0 == 12.0 and tp.temperature_for(256) == (12.0, True)
    srv, _, _, svc = _live(up, profile=tp, temperature=tp.t0)
    try:
        status, hdrs, _ = _post(srv, {**BODY, "think_budget": 256})
        (e,) = _timings(hdrs)
        assert status == 200 and (e["temperature"], e["calibrated"]) == (12.0, True)
        assert _post(srv, BODY)[0] == 200 and svc.engine.temperature == 12.0             # budget 0 at T_0
    finally:
        srv.shutdown()


def test_too_large_is_413_and_a_slow_generation_is_504(up):
    srv, _, _, svc = _live(up, s1_timeout=0.2)
    try:
        svc.engine.n_ctx = 10
        status, _, out = _post(srv, {**BODY, "think_budget": 64})
        assert status == 413 and "exceeds the slot context" in out["detail"]
        svc.engine.n_ctx = None
        up.think_delay_s, up.think_len = 0.2, 40                            # ~8 s of generation
        status, _, out = _post(srv, {**BODY, "think_budget": 8})           # timeout 0.2 + 0.8 s
        assert status == 504 and "exceeded its 1 s timeout" in out["detail"]
        st = _stats(srv)["think"]
        assert st["too_large"] == 1 and st["timeouts"] == 1
    finally:
        srv.shutdown()


def test_the_section_holds_a_fast_request_until_the_deliberate_readout_is_done(up):
    up.think_delay_s, up.think_len = 0.04, 20                                 # ~0.8 s of generation
    srv, _, _, _ = _live(up)
    try:
        results = {}

        def deliberate():
            results["d"] = _post(srv, {**BODY, "think_budget": 64})

        t = threading.Thread(target=deliberate)
        t.start()
        time.sleep(0.25)                                                    # the generation is in flight
        t0 = time.monotonic()
        results["f"] = _post(srv, TWO)
        waited = time.monotonic() - t0
        t.join(timeout=10)
        assert results["d"][0] == 200 and results["f"][0] == 200 and waited > 0.3
        comps = [r.json for r in up.requests if r.path == "/completion"]
        read_after_trace = next(i for i, c in enumerate(comps) if c.get("n_predict", 1) == 1
                                and any(5000 <= t < 5100 for t in c["prompt"]))
        gen = next(i for i, c in enumerate(comps) if c.get("n_predict", 1) > 1)
        assert gen < read_after_trace and all(c.get("n_predict", 1) == 1 for c in comps[gen + 1:read_after_trace + 1])
        assert all(any(5000 <= t < 5100 for t in c["prompt"]) for c in comps[gen + 1:read_after_trace + 1])
        assert _stats(srv)["s1"]["section_waits"] >= 1
    finally:
        srv.shutdown()


def test_without_deliberate_traffic_two_budget_0_requests_interleave_without_waiting(up):
    srv, _, _, _ = _live(up)
    try:
        out = {}
        ts = [threading.Thread(target=lambda k=k: out.__setitem__(k, _post(srv, TWO)[0])) for k in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=10)
        assert out == {0: 200, 1: 200}
        st = _stats(srv)
        assert st["s1"]["section_waits"] == 0 and st["s1"]["section_timeouts"] == 0 and not _gens(up)
    finally:
        srv.shutdown()


def test_a_wait_past_s1_timeout_is_a_503_counted_as_a_section_timeout(up):
    up.think_delay_s, up.think_len = 0.05, 40                                 # ~2 s of generation
    srv, _, _, _ = _live(up, s1_timeout=0.3)
    try:
        t = threading.Thread(target=lambda: _post(srv, {**BODY, "think_budget": 64}))
        t.start()
        time.sleep(0.2)
        status, _, out = _post(srv, BODY)
        assert status == 503 and "S1 section stayed busy for 0.3 s" in out["detail"]
        t.join(timeout=10)
        assert _stats(srv)["s1"]["section_timeouts"] == 1
    finally:
        srv.shutdown()


def test_a_two_question_deliberate_request_releases_the_section_between_questions(up):
    up.think_delay_s, up.think_len = 0.03, 20                                 # ~0.6 s per question
    srv, _, _, _ = _live(up)
    try:
        res = {}
        t = threading.Thread(target=lambda: res.__setitem__("d", _post(srv, {**TWO, "think_budget": 64})))
        t.start()
        time.sleep(0.2)
        res["f"] = _post(srv, BODY)
        t.join(timeout=15)
        assert res["d"][0] == 200 and res["f"][0] == 200
        comps = [r.json for r in up.requests if r.path == "/completion"]
        gens = [i for i, c in enumerate(comps) if c.get("n_predict", 1) > 1]
        plain = [i for i, c in enumerate(comps) if c.get("n_predict", 1) == 1
                 and not any(5000 <= t < 5100 for t in c["prompt"])]
        assert len(gens) == 2 and plain and gens[0] < min(plain) < gens[1]   # the fast request ran between them
    finally:
        srv.shutdown()


def test_a_client_that_leaves_cancels_the_generation_frees_the_slot_and_the_section(up):
    up.think_delay_s, up.think_len = 0.05, 100                                # ~5 s of generation
    srv, router, _, _ = _live(up)
    try:
        u = urllib.parse.urlsplit(f"http://127.0.0.1:{srv.server_address[1]}")
        s = socket.create_connection((u.hostname, u.port))
        body = json.dumps({**BODY, "think_budget": 100}).encode()
        s.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\n\r\n" % len(body) + body)
        time.sleep(0.4)
        s.close()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and (not up.aborted or router.inflight(0)):
            time.sleep(0.02)
        assert up.aborted and router.inflight(0) == 0
        t0 = time.monotonic()
        assert _post(srv, BODY)[0] == 200 and time.monotonic() - t0 < 2      # the section is free again
        st = _stats(srv)
        assert st["think"]["cancelled"] == 1 and st["think"]["requests"] == 0
        assert router.snapshot()["deliberate"] == {"0": 0}
    finally:
        srv.shutdown()


def test_cancel_before_the_slot_is_taken_ends_the_request_with_no_reply(up):
    srv, _, _, svc = _live(up)
    try:
        cancel = threading.Event()
        cancel.set()
        code, _, extra = svc.respond({**BODY, "think_budget": 64}, {}, cancel=cancel, watch=lambda: None)
        assert code == NO_REPLY and not _gens(up) and _stats(srv)["think"]["cancelled"] == 1
        code, _, _ = svc.respond(BODY, {}, cancel=cancel, watch=lambda: None)          # budget 0 ignores cancel
        assert code == 200
    finally:
        srv.shutdown()


def test_a_native_decision_model_refuses_a_positive_budget_with_422():
    plain = FakeUpstream().start()
    try:
        srv = make_server(ProxyConfig(upstream=plain.url, router=SlotRouter(4, 1), stats=Stats(), systemone=None),
                          port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            status, _, out = _post(srv, {**BODY, "think_budget": 64})
            assert status == 422 and "native decision model" in out["detail"]
            assert _post(srv, {**BODY, "think_budget": 0})[0] == 200          # forwarded
            status, _, out = _post(srv, {**BODY, "think_deadline_ms": 1500})
            assert status == 422 and out["detail"].endswith("think_deadline_ms must be 0 or absent")
            assert _post(srv, {**BODY, "think_deadline_ms": 0})[0] == 200
            assert _stats(srv)["counters"]["native_think_refused"] == 2
        finally:
            srv.shutdown()
    finally:
        plain.stop()


def test_stats_publish_the_think_block_and_the_comparability_keys(up):
    srv, _, stats, _ = _live(up)
    try:
        stats.host, stats.model_bytes, stats.n_ctx, stats.total_slots = {"chip": "M4"}, 123, 4096, 4
        stats.temperature_source = "default"
        st = _stats(srv)
        assert st["host"] == {"chip": "M4"} and st["model_bytes"] == 123 and st["mmproj_bytes"] is None
        assert st["n_ctx"] == 4096 and st["total_slots"] == 4 and st["temperature_source"] == "default"
        think = st["think"]
        for k in ("requests", "questions", "natural", "budget", "eos", "budget0", "cancelled", "timeouts",
                  "skipped_media", "uncalibrated", "too_large", "readout_reprefill", "text_fallback"):
            assert think[k] == 0, k
        assert think["gen_p50_ms"] is None and think["think_tokens_mean"] is None
        assert Stats().snapshot(SlotRouter(4, 1), None)["think"] is None     # never set: a plain Stats
    finally:
        srv.shutdown()


# ---- Review Focus 3-5 -----------------------------------------------------------------------------------------

def test_the_same_request_gets_the_same_trace_and_another_budget_is_another_seed(up):
    """A retried request repeats its trace (best-effort), and the budget is part of the digest, so
    the served trace at 256 is NOT the served trace at 512 cut short (calibrate's cuts are; served budgets are
    independent draws, which is what the profile's expected accuracy describes)."""
    srv, _, _, _ = _live(up)
    try:
        a = _post(srv, {**BODY, "think_budget": 256})[2]
        b = _post(srv, {**BODY, "think_budget": 256})[2]
        _post(srv, {**BODY, "think_budget": 512})
        gens = _gens(up)
        assert gens[0]["seed"] == gens[1]["seed"] != gens[2]["seed"]
        assert a == b
    finally:
        srv.shutdown()


def test_a_single_slot_upstream_serves_deliberate_readouts_with_no_reserved_slot_and_no_section(up):
    """llama-server -np 1: no reserved slots (X-Callo-Slot "-"), no section, still served."""
    engine = _engine(up)
    router, stats = SlotRouter(1, 1), Stats()
    stats.think = {"capable": True}
    svc = SystemOneService(engine, router, stats, model_name="fake-9b", s1_timeout=5)
    srv = make_server(ProxyConfig(upstream=up.url, router=router, stats=stats, systemone=svc.respond,
                                  systemone_cancel=True), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        status, headers, _ = _post(srv, {**BODY, "think_budget": 64})
        assert status == 200 and headers["X-Callo-Slot"] == "-" and _timings(headers)[0]["think_budget"] == 64
        assert "id_slot" not in _gens(up)[0] and _stats(srv)["s1"]["section_waits"] == 0
    finally:
        srv.shutdown()


def test_a_deliberate_request_on_a_lane_stays_on_its_lane_slot(up):
    """A lane request is never moved to an idle slot, deliberate or not."""
    srv, router, _, _ = _live(up, s1_slots=2)
    try:
        _, h1, _ = _post(srv, BODY, {"X-Callo-Conversation": "L"})
        _, h2, _ = _post(srv, {**BODY, "think_budget": 64}, {"X-Callo-Conversation": "L"})
        assert h1["X-Callo-Slot"] == h2["X-Callo-Slot"] and _timings(h2)[0]["think_budget"] == 64
        assert router.counters["s1_deliberate_idle"] == 0 and router.snapshot()["lanes"] == {"L": int(h1["X-Callo-Slot"])}
    finally:
        srv.shutdown()


# ---- every exit releases the deliberate count, a finite section wait, no section -----------------------------

def _settled(router):
    """The router is back to rest: no deliberate question, nothing in flight, every section free."""
    snap = router.snapshot()
    assert set(snap["deliberate"].values()) == {0}, snap["deliberate"]
    assert set(snap["inflight"].values()) == {0}, snap["inflight"]
    for sec in snap["sections"].values():
        assert (sec["readers"], sec["writer"], sec["queued"]) == (0, False, 0), sec


@pytest.mark.parametrize("s1_slots", [1, 2])
def test_a_served_deliberate_request_leaves_no_deliberate_count(up, s1_slots):
    srv, router, _, _ = _live(up, s1_slots=s1_slots)
    try:
        for _ in range(3):
            assert _post(srv, {**TWO, "think_budget": 64})[0] == 200
        _settled(router)
    finally:
        srv.shutdown()


class _Boom(BaseException):
    """Not an Exception: escapes every handler, so only a `finally` can release the slot."""


@pytest.mark.parametrize(("raised", "status"), [
    (Cancelled("left"), NO_REPLY), (ThinkTimeout("slow"), 504), (ThinkTooLarge("big"), 413),
    (UpstreamUnavailable("down"), 503), (ConnectionRefusedError("refused"), 503), (RuntimeError("bug"), 500)])
def test_every_failed_exit_of_a_deliberate_request_releases_its_deliberate_count(up, raised, status):
    srv, router, _, svc = _live(up, s1_slots=2)
    try:
        def fail(*_a, **_k):
            raise raised
        svc.engine.answer_parsed = fail
        code, _, extra = svc.respond({**BODY, "think_budget": 64}, {})
        assert code == status and extra["X-Callo-Slot"] == "0"
        _settled(router)
    finally:
        srv.shutdown()


def test_an_exception_that_escapes_the_service_still_releases_the_deliberate_count(up):
    srv, router, _, svc = _live(up)
    try:
        def boom(*_a, **_k):
            raise _Boom
        svc.engine.answer_parsed = boom
        with pytest.raises(_Boom):
            svc.respond({**BODY, "think_budget": 64}, {})
        _settled(router)
    finally:
        srv.shutdown()


def test_an_upstream_error_mid_generation_releases_the_deliberate_count(up):
    srv, router, _, _ = _live(up)
    try:
        up.completion_error = (500, "boom")
        status, _, _ = _post(srv, {**BODY, "think_budget": 64})
        assert status == 503                                                # an upstream 5xx is a 503
        _settled(router)
        up.completion_error = None
        assert _post(srv, {**BODY, "think_budget": 64})[0] == 200
        _settled(router)
    finally:
        srv.shutdown()


def test_too_large_and_a_generation_timeout_release_the_deliberate_count(up):
    srv, router, _, svc = _live(up, s1_timeout=0.2)
    try:
        svc.engine.n_ctx = 10
        assert _post(srv, {**BODY, "think_budget": 64})[0] == 413
        _settled(router)
        svc.engine.n_ctx = None
        up.think_delay_s, up.think_len = 0.2, 40
        assert _post(srv, {**BODY, "think_budget": 8})[0] == 504
        _settled(router)
    finally:
        srv.shutdown()


def test_a_deliberate_section_timeout_is_a_503_and_releases_the_deliberate_count(up):
    srv, router, _, _ = _live(up, s1_timeout=0.3)
    section = router.section(0)
    try:
        assert section.acquire(exclusive=True, timeout=1)                    # another writer holds slot 0's section
        t0 = time.monotonic()
        status, _, out = _post(srv, {**BODY, "think_budget": 64})
        assert status == 503 and "S1 section stayed busy for 0.3 s" in out["detail"]
        assert 0.25 < time.monotonic() - t0 < 3 and not _gens(up)
        section.release(exclusive=True)
        _settled(router)
        assert _stats(srv)["s1"]["section_timeouts"] == 1
    finally:
        srv.shutdown()


def test_a_cancel_while_waiting_for_the_section_is_no_reply_and_releases_the_deliberate_count(up):
    srv, router, _, svc = _live(up, s1_timeout=5)
    section = router.section(0)
    try:
        assert section.acquire(exclusive=True, timeout=1)
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        t0 = time.monotonic()
        code, _, _ = svc.respond({**BODY, "think_budget": 64}, {}, cancel=cancel, watch=lambda: None)
        assert code == NO_REPLY and time.monotonic() - t0 < 2 and not _gens(up)  # left the queue, no 5 s wait
        section.release(exclusive=True)
        _settled(router)
        st = _stats(srv)
        assert st["think"]["cancelled"] == 1 and st["s1"]["section_timeouts"] == 0
    finally:
        srv.shutdown()


def test_a_cancel_before_the_first_question_releases_the_deliberate_count(up):
    srv, router, _, svc = _live(up)
    try:
        cancel = threading.Event()
        cancel.set()
        assert svc.respond({**BODY, "think_budget": 64}, {}, cancel=cancel, watch=lambda: None)[0] == NO_REPLY
        _settled(router)
    finally:
        srv.shutdown()


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 0, -1.0, True, None])
def test_the_service_refuses_an_s1_timeout_the_section_cannot_wait_on(up, bad):
    with pytest.raises(InvalidInput, match="--s1-timeout must be a finite number of seconds above 0"):
        SystemOneService(_engine(up), SlotRouter(4, 1), Stats(), model_name="fake-9b", s1_timeout=bad)


def test_a_deliberate_request_the_fallback_moves_to_an_agent_slot_runs_without_a_section(up):
    """The S1 fallback (unchanged) moved the request off reserved slot 0, which a foreign task holds, onto idle
    agent slot 1, which has no section: served there without one (never refused), and slot 0's section is not
    touched, so a writer holding it does not delay the request."""
    engine = _engine(up)
    router, stats = SlotRouter(4, 1), Stats()
    stats.think = {"capable": True}
    svc = SystemOneService(engine, router, stats, model_name="fake-9b", s1_timeout=0.5,
                           slots_snapshot=lambda: ([{"id": 0, "is_processing": True}], time.monotonic()))
    section = router.section(0)
    assert router.section(1) is None
    try:
        assert section.acquire(exclusive=True, timeout=1)
        code, out, extra = svc.respond({**BODY, "think_budget": 64}, {})
        assert code == 200 and extra["X-Callo-Slot"] == "1" and out["answers"]["q"]["type"] == "choice"
        (g,) = _gens(up)
        assert g["id_slot"] == 1 and _timings(extra)[0]["think_budget"] == 64
        assert stats.counters["foreign_in_reserved"] == 1 and stats.section_timeouts == 0
        section.release(exclusive=True)
        _settled(router)
        assert section.snapshot()["waits"] == 0
    finally:
        if section.snapshot()["writer"]:
            section.release(exclusive=True)


# ---- POST /callo/think-probe ---------------------------------------------------------------------------------

import callo.proxy as proxy_module  # noqa: E402
from callo.guard import EXEMPT_ROUTES  # noqa: E402
from callo.proxy import ThinkProbe  # noqa: E402


def _probe_live(up, *, temperature=1.0, allow=False, engine_off=False):
    engine = _engine(up, temperature=temperature)
    router, stats = SlotRouter(4, 1), Stats()
    stats.think = {"capable": engine.think is not None}
    probe = ThinkProbe(engine, router, stats, s1_timeout=5)
    svc = SystemOneService(engine, router, stats, model_name="fake-9b", s1_timeout=5)
    srv = make_server(ProxyConfig(upstream=up.url, router=router, stats=stats, systemone=svc.respond,
                                  systemone_cancel=True, think_probe=None if engine_off else probe.respond,
                                  allow_think_probe=allow), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, router, stats, engine


def test_an_eos_trace_s_n_gen_minus_1_cut_is_one_readout_in_the_probe_and_its_stats(up):
    up.think_len, up.think_eos = 5, True
    srv, router, stats, engine = _probe_live(up)
    try:
        status, _, out = _post(srv, {"request": BODY, "budgets": [5, 64], "seed": 5}, path="/callo/think-probe")
        q = out["questions"]["q"]
        assert status == 200 and q["trace"]["close"] == "eos" and q["trace"]["n_gen"] == 6
        assert (q["budgets"]["5"]["k"], q["budgets"]["64"]["k"]) == (5, 6)
        assert q["budgets"]["5"]["probs"] == q["budgets"]["64"]["probs"]
        assert out["readouts"] == 1 and _stats(srv)["think_probe"]["readouts"] == 1 and _stats(srv)["s1"]["readouts"] == 1
    finally:
        srv.shutdown()


def test_plan_mode_renders_and_tokenizes_without_a_slot_a_section_or_a_readout(up):
    srv, router, stats, engine = _probe_live(up)
    try:
        status, headers, out = _post(srv, {"request": BODY, "budgets": [0, 256], "plan": True}, path="/callo/think-probe")
        assert status == 200 and headers["X-Callo-Slot"] == "-"
        n = engine.plan_deliberate("the state", REQ_Q.questions[0])
        assert out == {"readouts": 0, "questions": {"q": {"on_tokens": n, "n_ctx": 32768}}}
        assert not [r for r in up.requests if r.path == "/completion"]
        assert _stats(srv)["s1"]["readouts"] == 0 and router.inflight(0) == 0
    finally:
        srv.shutdown()


REQ_Q = __import__("callo.systemone", fromlist=["parse_request"]).parse_request(BODY)


def test_the_probe_reads_every_budget_once_at_t_1_and_counts_its_readouts(up):
    up.think_len = 40
    srv, router, stats, engine = _probe_live(up, temperature=1.9)
    try:
        status, headers, out = _post(srv, {"request": BODY, "budgets": [0, 8, 64, 256], "seed": 5},
                                     path="/callo/think-probe")
        assert status == 200 and headers["X-Callo-Slot"] == "0" and out["readouts"] == 3   # 0, the cut at 8, n_gen
        q = out["questions"]["q"]
        assert q["on_tokens"] == engine.plan_deliberate("the state", REQ_Q.questions[0]) and q["n_ctx"] == 32768
        assert q["trace"]["n_gen"] == 41 and q["trace"]["close"] == "natural" and len(q["trace"]["tokens"]) == 41
        assert set(q["budgets"]) == {"0", "8", "64", "256"}
        assert q["budgets"]["64"] == q["budgets"]["256"] and q["budgets"]["64"]["k"] == 41
        assert q["budgets"]["8"]["k"] == 8 and q["budgets"]["64"]["read_prompt_n"] == 2
        assert q["budgets"]["8"]["read_prompt_n"] == 3        # a cut on a pure-attention cache: the 3 close tokens
        assert q["budgets"]["0"]["bias"] is None and q["budgets"]["8"]["bias"] == "deliberate"
        probs0 = q["budgets"]["0"]["probs"]
        assert abs(sum(probs0) - 1) < 1e-9
        served = _post(srv, BODY)[2]["answers"]["q"]["probabilities"]          # served at 1.9: flatter than T = 1
        assert max(served.values()) < max(probs0)
        (g,) = _gens(up)
        assert g["seed"] == 5 and g["n_predict"] == 256
        st = _stats(srv)
        assert st["s1"]["readouts"] == 3 + 1 and router.inflight(0) == 0        # condition 4 sees every readout
        # the probe stays out of the served windows (the brief expected it in think.*); the served
        # budget-0 request is the only entry there, and the probe has its own block
        assert st["s1"]["n"] == 1 and st["think"]["requests"] == 0 and st["think"]["questions"] == 0
        assert st["think"]["budget"] == 0 and st["think"]["natural"] == 0 and st["think"]["gen_p95_ms"] is None
        assert st["think_probe"] == {"requests": 1, "readouts": 3, "errors": 0, "cancelled": 0, "timeouts": 0,
                                     "too_large": 0}
    finally:
        srv.shutdown()


def test_the_probe_refusals_404_403_and_422(up, monkeypatch):
    srv, _, _, _ = _probe_live(up, engine_off=True)
    try:
        assert _post(srv, {"request": BODY, "budgets": [0]}, path="/callo/think-probe")[0] == 404
    finally:
        srv.shutdown()
    srv, _, _, _ = _probe_live(up)
    try:
        monkeypatch.setattr(proxy_module, "_is_loopback", lambda addr: False)
        status, _, out = _post(srv, {"request": BODY, "budgets": [0]}, path="/callo/think-probe")
        assert status == 403 and "--allow-think-probe" in out["detail"]
        assert _stats(srv)["counters"]["think_probe_refused"] == 1
        monkeypatch.setattr(proxy_module, "_is_loopback", lambda addr: True)
        for bad, why in ([{"request": BODY, "budgets": []}, "budgets must be"],
                         [{"request": BODY, "budgets": [256, 0]}, "ascending"],
                         [{"request": BODY, "budgets": [0, 40000]}, "0 to 32768"],
                         [{"request": BODY, "budgets": [0, 0]}, "distinct"],
                         [{"request": TWO, "budgets": [0]}, "exactly one question"],
                         [{"request": IMG, "budgets": [0]}, "no media"],
                         [{"request": BODY, "budgets": [0], "seed": -1}, "seed must be"],
                         [{"request": BODY, "budgets": [0], "sampling": 3}, "sampling must be an object"],
                         [{"request": BODY, "budgets": [0], "plan": "yes"}, "plan must be"],
                         [[], "must be an object"]):
            status, _, out = _post(srv, bad, path="/callo/think-probe")
            assert status == 422 and why in out["detail"], (bad, out)
    finally:
        srv.shutdown()


def test_the_probe_takes_only_a_full_pinned_sampler_dict_and_null_keeps_the_engine_s(up, monkeypatch):
    from callo.think import parse_sampling, resolve_sampling
    pinned = resolve_sampling(parse_sampling("fixed")).pinned
    srv, _, _, engine = _probe_live(up)
    try:
        for bad, why in ([{"temperature": 0.6}, "lacks"],                               # partial: would unpin the rest
                         [{**pinned, "n_predict": 3}, "not a sampler field"],         # would cut the trace at 3
                         [{**pinned, "top_k": 20.0}, "top_k must be an int"],
                         [{**pinned, "top_p": "0.9"}, "must be a number"],
                         [{**pinned, "min_p": float("inf")}, "must be finite"]):
            status, _, out = _post(srv, {"request": BODY, "budgets": [8, 64], "sampling": bad}, path="/callo/think-probe")
            assert status == 422 and why in out["detail"], (bad, out)
        assert not _gens(up)                                                            # refused before any call
        status, _, out = _post(srv, {"request": BODY, "budgets": [8, 64], "sampling": None}, path="/callo/think-probe")
        assert status == 200 and out["questions"]["q"]["trace"]["n_gen"] == 6
        (g,) = _gens(up)
        assert g["n_predict"] == 64 and g["temperature"] == engine.think_sampling["temperature"]
        hot = {**pinned, "temperature": 1.7}
        status, _, _ = _post(srv, {"request": BODY, "budgets": [64], "sampling": hot}, path="/callo/think-probe")
        assert status == 200 and _gens(up)[-1]["temperature"] == 1.7 and _gens(up)[-1]["n_predict"] == 64
    finally:
        srv.shutdown()
    srv, _, _, _ = _probe_live(up, allow=True)
    try:
        monkeypatch.setattr(proxy_module, "_is_loopback", lambda addr: False)
        assert _post(srv, {"request": BODY, "budgets": [0]}, path="/callo/think-probe")[0] == 200
    finally:
        srv.shutdown()
    monkeypatch.setattr(proxy_module, "_is_loopback", lambda addr: True)
    plain = FakeUpstream().start()
    try:
        srv, _, _, _ = _probe_live(plain)
        try:
            status, _, out = _post(srv, {"request": BODY, "budgets": [0, 256]}, path="/callo/think-probe")
            assert status == 422 and "no thinking mode" in out["detail"]
            assert _post(srv, {"request": BODY, "budgets": [0]}, path="/callo/think-probe")[0] == 200
            planned = _post(srv, {"request": BODY, "budgets": [0], "plan": True}, path="/callo/think-probe")[2]
            assert planned["readouts"] == 0 and planned["questions"]["q"]["on_tokens"] > 0   # the thinking-off prompt
        finally:
            srv.shutdown()
    finally:
        plain.stop()


def test_a_mixed_budget_probe_takes_the_section_once_and_the_route_is_exempt_from_condition_5(up):
    up.think_delay_s, up.think_len = 0.02, 20                                 # ~0.4 s of generation
    srv, router, _, _ = _probe_live(up)
    try:
        res = {}
        t = threading.Thread(target=lambda: res.__setitem__("p", _post(srv, {"request": BODY, "budgets": [0, 64]},
                                                                      path="/callo/think-probe")))
        t.start()
        time.sleep(0.15)
        res["f"] = _post(srv, BODY)
        t.join(timeout=10)
        assert res["p"][0] == 200 and res["f"][0] == 200 and router.inflight(0) == 0
        # once: the probe's budget-0 readout, its generation and its trace readouts run back to back, and the fast
        # request (no trace tokens in its prompt) waited for all of them (one section wait)
        comps = [r.json for r in up.requests if r.path == "/completion"]
        traced = [any(5000 <= t < 5100 for t in c["prompt"]) for c in comps]
        gen = next(i for i, c in enumerate(comps) if c.get("n_probs", 0) == 0)
        assert gen > 0 and not any(traced[:gen]) and traced[gen + 1]
        fast = next(i for i in range(gen + 1, len(comps)) if not traced[i])
        assert all(traced[gen + 1:fast]) and not any(traced[fast:])
        st = _stats(srv)
        assert st["s1"]["section_waits"] == 1
        assert "/callo/think-probe" in EXEMPT_ROUTES
        assert st["routes"]["/callo/think-probe"] == 1
    finally:
        srv.shutdown()


# ---- the probe: the default pinned sampler, one release on every exit, a finite section wait ------------------

def _probe(up, *, s1_timeout=5.0):
    """A ThinkProbe called directly (no HTTP), for the exit paths; returns (probe, router, stats)."""
    router, stats = SlotRouter(4, 2), Stats()
    stats.think = {"capable": True}
    return ThinkProbe(_engine(up), router, stats, s1_timeout=s1_timeout), router, stats


def test_a_null_sampling_on_an_engine_without_think_sampling_sends_the_default_pinned_dict(up):
    """The engine's own sampler is its pinned dict, else the default policy's, never an empty dict (which would
    leave every sampler field to llama-server's flags)."""
    from callo.engine import default_trace_sampling
    srv, _, _, engine = _probe_live(up)
    try:
        engine.think_sampling = None
        status, _, _ = _post(srv, {"request": BODY, "budgets": [64], "sampling": None}, path="/callo/think-probe")
        assert status == 200
        (g,) = _gens(up)
        default = default_trace_sampling()
        assert default and {k: g[k] for k in default} == default and g["temperature"] == 0.6
    finally:
        srv.shutdown()


@pytest.mark.parametrize("budgets", [[0], [0, 64], [64]])
@pytest.mark.parametrize(("raised", "status"), [
    (Cancelled("left"), NO_REPLY), (ThinkTimeout("slow"), 504), (ThinkTooLarge("big"), 413),
    (UpstreamUnavailable("down"), 503), (ConnectionRefusedError("refused"), 503),
    (TemplateError("the render failed"), 500), (RuntimeError("bug"), 500)])
def test_every_failed_exit_of_a_probe_releases_its_slot_and_section_once(up, budgets, raised, status):
    probe, router, _ = _probe(up)

    def fail(*_a, **_k):
        raise raised
    probe.engine.answer = fail
    probe.engine.answer_deliberate = fail
    code, out, extra = probe.respond({"request": BODY, "budgets": budgets}, {})
    assert code == status and extra["X-Callo-Slot"] == "0"
    if isinstance(raised, TemplateError):
        assert out["detail"] == "the render failed"             # a ReadoutError: 500 with its detail
    _settled(router)
    snap = probe.stats.snapshot(router, None)
    counter = {Cancelled: "cancelled", ThinkTimeout: "timeouts", ThinkTooLarge: "too_large"}.get(type(raised))
    assert snap["think_probe"] == {"requests": 0, "readouts": 0, "errors": 1, "cancelled": int(counter == "cancelled"),
                                   "timeouts": int(counter == "timeouts"), "too_large": int(counter == "too_large")}
    assert all(snap["think"][k] == 0 for k in ("cancelled", "timeouts", "too_large", "requests"))   # serve's stay 0


def test_an_exception_that_escapes_the_probe_still_releases_its_slot_and_section(up):
    probe, router, _ = _probe(up)

    def boom(*_a, **_k):
        raise _Boom
    probe.engine.answer_deliberate = boom
    with pytest.raises(_Boom):
        probe.respond({"request": BODY, "budgets": [0, 64]}, {})
    _settled(router)


def test_a_probe_section_timeout_is_a_503_naming_its_own_s1_timeout_and_a_queued_cancel_is_no_reply(up):
    probe, router, stats = _probe(up, s1_timeout=0.3)
    section = router.section(0)
    assert section.acquire(exclusive=True, timeout=1)                        # another writer holds slot 0's section
    try:
        code, out, _ = probe.respond({"request": BODY, "budgets": [0, 64]}, {})
        assert code == 503 and "S1 section stayed busy for 0.3 s" in out["detail"]
        assert stats.section_timeouts == 1 and not _gens(up)
        probe._svc.s1_timeout = 5.0
        assert probe.s1_timeout == 5.0                                       # one source of truth
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        t0 = time.monotonic()
        code, _, _ = probe.respond({"request": BODY, "budgets": [64]}, {}, cancel=cancel, watch=lambda: None)
        assert code == NO_REPLY and time.monotonic() - t0 < 2 and not _gens(up)
    finally:
        section.release(exclusive=True)
    _settled(router)
    assert stats.section_timeouts == 1


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 0, -1.0, True, None])
def test_the_probe_refuses_an_s1_timeout_the_section_cannot_wait_on(up, bad):
    with pytest.raises(InvalidInput, match="--s1-timeout must be a finite number of seconds above 0"):
        ThinkProbe(_engine(up), SlotRouter(4, 1), Stats(), s1_timeout=bad)


@pytest.mark.parametrize(("raised", "status", "detail"), [
    (TemplateError("the render failed"), 500, "the render failed"), (RuntimeError("bug"), 500, "internal error"),
    (ConnectionRefusedError("refused"), 503, "llama-server unavailable")])
def test_plan_mode_failures_are_statuses_and_touch_no_slot(up, raised, status, detail):
    probe, router, _ = _probe(up)

    def fail(*_a, **_k):
        raise raised
    probe.engine.plan_deliberate = fail
    code, out, extra = probe.respond({"request": BODY, "budgets": [0, 64], "plan": True}, {})
    assert code == status and detail in out["detail"] and extra["X-Callo-Slot"] == "-"
    _settled(router)


# ---- the shared frame, the probe's own stats, the drained refusals, single options, loopback --------------------

ONE = {"state": "the state", "questions": {"q": {"type": "choice", "criteria": {"a": "only"}}}}


def test_a_single_option_question_skips_the_trace_as_serve_does(up):
    srv, router, _, _ = _probe_live(up)
    try:
        assert _post(srv, {**ONE, "think_budget": 64})[0] == 200 and not _gens(up)    # serve: the shortcut, no trace
        status, _, out = _post(srv, {"request": ONE, "budgets": [0, 64, 256]}, path="/callo/think-probe")
        assert status == 200 and out["readouts"] == 1 and not _gens(up)
        q = out["questions"]["q"]
        assert q["trace"] is None and q["on_tokens"] is None and q["content"] is None
        assert all(b["probs"] == [1.0] and b["stop"] == "single_option" for b in q["budgets"].values())
        assert q["budgets"]["64"]["k"] == 0 and q["budgets"]["256"]["read_prompt_n"] is None
        assert _stats(srv)["think_probe"]["readouts"] == 1 and router.inflight(0) == 0
    finally:
        srv.shutdown()


@pytest.mark.parametrize("engine_off", [True, False])
def test_a_refused_probe_drains_its_body_so_the_client_reads_the_refusal(up, monkeypatch, engine_off):
    srv, _, _, _ = _probe_live(up, engine_off=engine_off)
    try:
        monkeypatch.setattr(proxy_module, "_is_loopback", lambda addr: False)
        body = json.dumps({"request": BODY, "budgets": [0], "pad": "x" * (8 << 20)}).encode()
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=15)
        try:
            conn.request("POST", "/callo/think-probe", body=body, headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            assert r.status == (404 if engine_off else 403) and json.loads(r.read())["detail"]
        finally:
            conn.close()
    finally:
        srv.shutdown()


@pytest.mark.parametrize(("address", "loopback"), [
    ("127.0.0.1", True), ("127.1.2.3", True), ("::1", True), ("::ffff:127.0.0.1", True),
    ("10.0.0.1", False), ("192.168.1.20", False), ("::ffff:10.0.0.1", False), ("0.0.0.0", False), ("::", False),
    ("fe80::1", False), ("128.0.0.1", False)])
def test_is_loopback_on_real_addresses(address, loopback):
    assert proxy_module._is_loopback(address) is loopback


def test_a_gguf_that_recommends_temp_0_is_honoured_end_to_end(up):
    """llama.cpp reads temp <= 0 as greedy; a GGUF's temp 0 is pinned as 0 (per-GGUF sampling by default),
    the probe takes that dict, and the generation carries it, from the request and from the engine's own."""
    from callo.think import check_pinned
    s = resolve_sampling(parse_sampling("model"), {"temp": 0.0, "top_k": 20, "top_p": 0.95})
    assert s.pinned["temperature"] == 0.0 and s.origin["temperature"] == "gguf" and check_pinned(s.pinned) is None
    srv, _, _, engine = _probe_live(up)
    try:
        status, _, out = _post(srv, {"request": BODY, "budgets": [64], "sampling": s.pinned}, path="/callo/think-probe")
        assert status == 200, out
        engine.think_sampling = s.pinned
        assert _post(srv, {"request": BODY, "budgets": [64], "sampling": None}, path="/callo/think-probe")[0] == 200
        assert _post(srv, {**BODY, "think_budget": 64})[0] == 200                 # serve sends the same dict
        assert [g["temperature"] for g in _gens(up)] == [0.0, 0.0, 0.0]
    finally:
        srv.shutdown()


# ---- cancellation, the hard cap, strict-JSON stats, a client that leaves ------------------------------------

def test_cancelled_has_no_status_of_its_own_only_the_service_frame_answers_no_reply():
    from callo.proxy import _status

    assert _status(Cancelled("left")) == 500                          # never a 499 on the wire from another caller


def test_the_hard_cap_has_one_source():
    from callo import systemone
    from callo.think import MAX_BUDGET

    assert proxy_module.MAX_PROBE_BUDGET is MAX_BUDGET and systemone.MAX_THINK_BUDGET is MAX_BUDGET
    assert systemone.THINK_BUDGET_ERROR == "think_budget must be an integer from 0 to 32768"


def test_stats_are_strict_json_when_the_gguf_holds_nan_or_inf():
    stats = Stats()
    stats.think = {"capable": True, "gguf_sampling": {"temp": float("nan"), "top_p": float("inf")}, "gguf_error": None}
    snap = stats.snapshot(SlotRouter(4, 1), None)
    assert snap["think"]["gguf_sampling"] == {"temp": "nan", "top_p": "inf"}
    json.dumps(snap, allow_nan=False)                                   # raises on a bare NaN


def test_a_client_that_leaves_is_logged(up, caplog):
    import logging  # noqa: PLC0415

    up.think_delay_s, up.think_len = 0.05, 100
    srv, router, _, _ = _live(up)
    try:
        with caplog.at_level(logging.INFO, logger="callo"):
            u = urllib.parse.urlsplit(f"http://127.0.0.1:{srv.server_address[1]}")
            s = socket.create_connection((u.hostname, u.port))
            body = json.dumps({**BODY, "think_budget": 100}).encode()
            s.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                      b"Content-Length: %d\r\n\r\n" % len(body) + body)
            time.sleep(0.4)
            s.close()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and "the client left" not in caplog.text:
                time.sleep(0.02)
        assert "POST /v1/systemone from 127.0.0.1: the client left during a deliberate readout; no reply" in caplog.text
    finally:
        srv.shutdown()


def test_the_probe_s_budget_0_pass_honours_cancel(up):
    engine = _engine(up)
    probe = ThinkProbe(engine, SlotRouter(4, 1), Stats(), s1_timeout=5)
    req = REQ_Q
    cancel = threading.Event()
    cancel.set()
    up.requests.clear()
    with pytest.raises(Cancelled):
        probe._read_budgets(req, req.questions[0], [0, 64], 1, None, 0, [], cancel)
    assert not [r for r in up.requests if r.path == "/completion"]
