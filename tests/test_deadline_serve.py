"""Think deadlines through a real proxy and service against the thinking fake in real time: the resolution
table, the chunked serve path, the budget-0 fallback, the readout temperature, the
X-Callo-Timings fields and the `think` counters by close."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest
from fake_upstream import FakeUpstream
from test_think import profile_dict
from test_think_serve import BODY, IMG, THINK_KEYS, V02_KEYS, _engine

from callo.deadline import Estimators
from callo.proxy import ProxyConfig, Stats, SystemOneService, make_server
from callo.slots import SlotRouter
from callo.systemone import parse_request
from callo.think import load_think_profile

DEADLINE_KEYS = ["think_deadline_ms", "elapsed_ms", "deadline_met"]
CHUNKED_KEYS = [*THINK_KEYS, *DEADLINE_KEYS, "chunks"]
FALLBACK_KEYS = [*V02_KEYS, *DEADLINE_KEYS, "chunks", "deadline_fallback"]


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=10_000).start()
    u.think_real_time, u.think_ms_per_token, u.think_prompt_ms_per_token = True, 2.0, 0.05
    yield u
    u.stop()


def live(up, *, think_deadline=0, think_default=0, think_max=4096, profile=None, s1_slots=1, reserve_ms=30.0,
         temperature=1.0, slots_snapshot=None, s1_timeout=5.0, request_log=None, think_converge=0.0,
         think_converge_gate=False):
    """A real proxy + service + engine against the fake; returns (server, router, stats, service)."""
    engine = _engine(up, temperature=temperature)
    router, stats = SlotRouter(4, s1_slots), Stats()
    stats.think = {"capable": engine.think is not None, "reason": engine.think_reason, "default_budget": think_default,
                   "max": think_max, "profile": None}
    svc = SystemOneService(engine, router, stats, model_name="fake-9b", think_default=think_default,
                           think_max=think_max, think_profile=profile, s1_timeout=s1_timeout, think_deadline=think_deadline,
                           estimators=Estimators(reserve_ms=reserve_ms), slots_snapshot=slots_snapshot,
                           think_converge=think_converge, think_converge_gate=think_converge_gate)
    srv = make_server(ProxyConfig(upstream=up.url, router=router, stats=stats, systemone=svc.respond,
                                  systemone_cancel=True, request_log=request_log), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, router, stats, svc


def post(srv, body, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/v1/systemone", data=json.dumps(body).encode(),
                                 method="POST", headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


def entries(headers):
    return json.loads(headers["X-Callo-Timings"])


def gens(up):
    return [r.json for r in up.requests if r.path == "/completion" and not r.json.get("n_probs")]


def stats(srv):
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/callo/stats", timeout=10) as r:
        return json.loads(r.read())


def test_the_request_field_is_parsed_and_a_bad_value_is_422(up):
    srv, *_ = live(up)
    try:
        for bad in (-1, 600_001, 1.5, True, "100"):
            status, _, out = post(srv, {**BODY, "think_deadline_ms": bad})
            assert status == 422 and out["detail"] == "think_deadline_ms must be an integer from 0 to 600000"
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": None})
        assert status == 200 and list(entries(headers)[0]) == V02_KEYS and not gens(up)    # null: the default (off)
    finally:
        srv.shutdown()


def test_a_deadline_only_request_is_chunked_under_the_cap_b_star_and_reported(up):
    srv, *_ = live(up)
    try:
        status, headers, out = post(srv, {**BODY, "think_deadline_ms": 900})
        assert status == 200 and out["answers"]["q"]["type"] == "choice"
        (e,) = entries(headers)
        assert list(e) == CHUNKED_KEYS and e["think_close"] == "deadline" and e["think_budget"] == 4096
        assert e["chunks"] == len(gens(up)) >= 2 and e["think_deadline_ms"] == 900 and e["read_prompt_n"] == 4
        assert e["elapsed_ms"] <= 900 + 50 and e["deadline_met"] is (e["elapsed_ms"] <= 900)
        assert all(x["prompt_n"] == 1 for x in up.gen_log[1:])
        st = stats(srv)["think"]
        assert (st["requests"], st["questions"], st["natural"], st["budget"]) == (1, 1, 0, 0)
        assert st["deadline"]["closes"]["deadline"] == 1
    finally:
        srv.shutdown()


def test_both_limits_close_on_the_first_to_bind(up):
    srv, *_ = live(up)
    try:
        _, headers, _ = post(srv, {**BODY, "think_budget": 40, "think_deadline_ms": 5000})
        (e,) = entries(headers)
        assert (e["think_close"], e["think_tokens"], e["think_budget"]) == ("budget", 40, 40)
        assert all(g["n_predict"] <= 40 for g in gens(up))
        up.think_len = 25
        _, headers, _ = post(srv, {**BODY, "think_budget": 1024, "think_deadline_ms": 5000})
        assert (entries(headers)[0]["think_close"], entries(headers)[0]["think_tokens"]) == ("natural", 26)
        st = stats(srv)["think"]
        assert (st["budget"], st["natural"], st["questions"]) == (1, 1, 2)
    finally:
        srv.shutdown()


def test_an_explicit_budget_0_with_a_deadline_is_a_timed_budget_0_readout(up):
    srv, router, *_ = live(up, think_deadline=2000)
    try:
        _, headers, _ = post(srv, {**BODY, "think_budget": 0})
        (e,) = entries(headers)
        assert list(e) == [*V02_KEYS, *DEADLINE_KEYS] and e["think_deadline_ms"] == 2000 and e["deadline_met"] is True
        assert not gens(up) and stats(srv)["think"]["questions"] == 0
        section = router.section(0)                       # the elapsed time counts the section wait (stamped after)
        assert section.acquire(exclusive=True, timeout=1.0)
        threading.Timer(0.3, lambda: section.release(exclusive=True)).start()
        _, headers, _ = post(srv, {**BODY, "think_budget": 0})
        assert entries(headers)[0]["elapsed_ms"] >= 280.0
    finally:
        srv.shutdown()


def test_under_a_serve_wide_deadline_a_budget_only_request_is_chunked_unless_it_sends_deadline_0(up):
    """Only the `neither` and `budget only` rows are the plain budget path's upstream calls."""
    up.think_len = 25
    srv, *_ = live(up, think_deadline=5000)
    try:
        _, headers, _ = post(srv, {**BODY, "think_budget": 256})
        assert list(entries(headers)[0]) == CHUNKED_KEYS
        del up.requests[:]
        _, headers, _ = post(srv, {**BODY, "think_budget": 256, "think_deadline_ms": 0})
        assert list(entries(headers)[0]) == THINK_KEYS and [g["n_predict"] for g in gens(up)] == [256]
        _, headers, _ = post(srv, BODY)
        assert list(entries(headers)[0]) == CHUNKED_KEYS                     # no field: the default thinks
    finally:
        srv.shutdown()


def test_the_tb_rows_keep_tbs_upstream_calls_bit_for_bit(up):
    up.think_len = 25
    calls = []
    for think_deadline in (0, 7000):
        srv, *_ = live(up, think_deadline=think_deadline)
        try:
            del up.requests[:]
            post(srv, {**BODY, "think_budget": 256, "think_deadline_ms": 0})
            post(srv, {**BODY, "think_budget": 0})
            calls.append([(r.path, r.json) for r in up.requests])
        finally:
            srv.shutdown()
    assert calls[0] == calls[1]


def test_think_max_still_caps_the_budget(up):
    srv, *_ = live(up, think_max=512)
    try:
        status, _, out = post(srv, {**BODY, "think_budget": 1024, "think_deadline_ms": 2000})
        assert status == 422 and "--think-max 512" in out["detail"]
    finally:
        srv.shutdown()


def test_a_media_request_with_a_deadline_is_a_timed_budget_0_readout_and_one_asking_for_a_budget_is_422():
    vup = FakeUpstream(think="preopened", think_len=25, vision=True).start()
    srv, *_ = live(vup, think_deadline=2000)
    try:
        status, headers, _ = post(srv, IMG)
        (e,) = entries(headers)
        assert status == 200 and list(e) == [*V02_KEYS, *DEADLINE_KEYS, "deadline_fallback"] and not gens(vup)
        status, _, out = post(srv, {**IMG, "think_budget": 64})
        assert status == 422 and out["detail"] == "deliberate readouts are text-only in this version"
    finally:
        srv.shutdown()
        vup.stop()


def test_a_model_whose_close_tag_is_several_tokens_answers_a_request_deadline_at_budget_0(up):
    up.think_multi_tag = True
    srv, *_ = live(up)
    try:
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": 2000})
        (e,) = entries(headers)
        assert status == 200 and list(e) == [*V02_KEYS, *DEADLINE_KEYS, "deadline_fallback"] and not gens(up)
    finally:
        srv.shutdown()


def test_a_share_too_short_to_think_is_read_at_budget_0_at_t0_and_flagged(up):
    """The fallback is the budget-0 readout as served today, thinking-off prompt and T_0."""
    srv, *_ = live(up, temperature=1.9, reserve_ms=30.0)
    try:
        _, _, plain = post(srv, {**BODY, "think_budget": 0})
        status, headers, out = post(srv, {**BODY, "think_deadline_ms": 40})
        (e,) = entries(headers)
        assert status == 200 and list(e) == FALLBACK_KEYS and e["deadline_fallback"] == "budget0" and e["chunks"] == 0
        assert out["answers"] == plain["answers"] and not gens(up)                  # T_0, the same readout
        st = stats(srv)["think"]
        assert (st["budget0"], st["questions"], st["requests"]) == (1, 1, 1)
    finally:
        srv.shutdown()


def _one(body):
    """(state, the first question) of a /v1/systemone body, as the engine plans it."""
    req = parse_request(body)
    return req.state, req.questions[0]


def test_the_readout_temperature_follows_the_close_and_the_deadline_point(up):
    tp = load_think_profile(profile_dict(deadlines={"3000": {"T": 0.77, "cap": 1024}}))
    srv, _, _, svc = live(up, profile=tp)
    try:
        _, headers, _ = post(srv, {**BODY, "think_deadline_ms": 3000})
        e = entries(headers)[0]
        assert (e["temperature"], e["calibrated"], e["think_budget"]) == (0.77, True, 1024)     # B* = the profile's max
        _, headers, _ = post(srv, {**BODY, "think_deadline_ms": 900})
        e = entries(headers)[0]
        assert e["think_close"] == "deadline" and (e["temperature"], e["calibrated"]) == (1.1, False)
        up.think_len = 25
        _, headers, _ = post(srv, {**BODY, "think_deadline_ms": 2500})
        e = entries(headers)[0]
        assert e["think_close"] == "natural" and (e["temperature"], e["calibrated"]) == (0.9, True)
        assert stats(srv)["think"]["uncalibrated"] == 1
        up.think_len = 10_000                             # the CLIPPED cap: a context below the point's cap
        svc.engine.n_ctx = svc.engine.plan_deliberate(*_one(BODY)) + 16 + 300
        _, headers, _ = post(srv, {**BODY, "think_deadline_ms": 3000})
        e = entries(headers)[0]
        assert e["think_budget"] == 300 and e["think_close"] == "budget" and e["temperature"] != 0.77
    finally:
        srv.shutdown()


def _median_wall(srv, up, body, n=3):
    """Median wall of n runs; each run must be one generation that closed naturally in one chunk (structural)."""
    walls = []
    for _ in range(n):
        before = len(gens(up))
        t0 = time.monotonic()
        status, headers, _ = post(srv, body)
        walls.append(1000 * (time.monotonic() - t0))
        assert status == 200
        (e,) = entries(headers)
        assert len(gens(up)) - before == 1 and e["think_close"] == "natural", (body, e)
        assert e.get("chunks", 1) == 1, (body, e)
    return sorted(walls)[n // 2]


def test_no_padding_a_natural_close_costs_what_it_costs_under_any_limits(up):
    """A natural close at L gives the same time for (B = L + 1, no D), (B = 1024, D = 10 s), (no B, D = 10 s)."""
    up.think_len = 30
    srv, *_ = live(up)
    try:
        times = [_median_wall(srv, up, {**BODY, **limits}) for limits in
                 ({"think_budget": 31}, {"think_budget": 1024, "think_deadline_ms": 10_000}, {"think_deadline_ms": 10_000})]
        assert max(times) - min(times) <= 100.0, times
        assert max(times) < 1000.0
    finally:
        srv.shutdown()


def test_a_multi_token_close_tag_deadline_says_not_chunkable(up):
    up.think_multi_tag = True
    srv, *_ = live(up)
    try:
        status, headers, out = post(srv, {**BODY, "think_deadline_ms": 2000})
        (e,) = entries(headers)
        assert status == 200 and out["answers"]["q"]["type"] == "choice" and not gens(up)
        assert list(e) == [*V02_KEYS, *DEADLINE_KEYS, "deadline_fallback"] and e["deadline_fallback"] == "not_chunkable"
        assert "think_budget" not in e and "think_close" not in e
    finally:
        srv.shutdown()


def test_a_model_with_no_think_frame_deadline_says_not_chunkable():
    nup = FakeUpstream().start()
    srv, *_ = live(nup)
    try:
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": 2000})
        (e,) = entries(headers)
        assert status == 200 and e["deadline_fallback"] == "not_chunkable" and "think_budget" not in e
    finally:
        srv.shutdown()
        nup.stop()


def test_a_media_request_with_a_deadline_says_media():
    vup = FakeUpstream(think="preopened", think_len=25, vision=True).start()
    srv, *_ = live(vup, think_deadline=2000)
    try:
        status, headers, _ = post(srv, IMG)
        (e,) = entries(headers)
        assert status == 200 and e["deadline_fallback"] == "media" and "think_budget" not in e
    finally:
        srv.shutdown()
        vup.stop()


def test_a_met_chunked_deadline_and_a_no_deadline_request_carry_no_fallback(up):
    srv, *_ = live(up)
    try:
        _, headers, _ = post(srv, {**BODY, "think_deadline_ms": 5000, "think_budget": 8})
        (e,) = entries(headers)
        assert "deadline_fallback" not in e and e["think_deadline_ms"] == 5000
        _, headers, _ = post(srv, BODY)
        (e,) = entries(headers)
        assert list(e) == V02_KEYS
        _, headers, _ = post(srv, {**BODY, "think_deadline_ms": 2000, "think_budget": 0})
        assert "deadline_fallback" not in entries(headers)[0]
    finally:
        srv.shutdown()
