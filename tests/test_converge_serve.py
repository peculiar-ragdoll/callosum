"""Early exit through a real proxy and service against the thinking fake: the resolution (a third limit beside the
budget and the deadline), the z -> rule mapping, `unreachable`, the opt-outs and fallbacks, the converged readout's
temperature, early exit under a deadline (two questions), the serve default's reach, and the request log."""
import json
import threading
import time

import pytest
from fake_upstream import FakeUpstream
from test_deadline_serve import CHUNKED_KEYS, FALLBACK_KEYS, _one, entries, gens, live, post, stats
from test_think import profile_dict
from test_think_engine import REQ, engine_for
from test_think_serve import BODY, IMG, THINK_KEYS, V02_KEYS

from callo.deadline import ConvergeRun, Estimators
from callo.engine import DeadlineRun
from callo.proxy import RequestLog
from callo.think import load_think_profile

TWO = {**BODY, "questions": {**BODY["questions"], "r": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}}
CONVERGE_KEYS = ["converge", "converge_rule", "probes"]


def block(**rules):
    base = {"0.9": {"cell": [1, 0.6], "T": 2.0}, "0.95": {"cell": [2, 0.6], "T": 1.5}, "0.99": None}
    return {"K": 32, "rules": {**base, **rules}, "cost": {"c_ms": 5.0}}


def profile(**rules):
    return load_think_profile(profile_dict(converge=block(**rules)))


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=10_000).start()
    yield u
    u.stop()


def one(srv, body):
    status, headers, out = post(srv, body)
    assert status == 200, out
    (e,) = entries(headers)
    return e


def test_think_converge_alone_thinks_to_b_star_in_k_chunks_and_stops_converged(up):
    srv, *_ = live(up, profile=profile())
    try:
        e = one(srv, {**BODY, "think_converge": 0.9})
        assert e["think_close"] == "converged" and list(e) == [*THINK_KEYS, *CONVERGE_KEYS]
        assert (e["converge"], e["converge_rule"], e["think_budget"]) == (0.9, [1, 0.6], 1024)    # B*: the profile's
        assert e["think_tokens"] == 32 * e["probes"] and (e["temperature"], e["calibrated"]) == (2.0, True)
        assert [g["n_predict"] for g in gens(up)] == [32] * e["probes"]
    finally:
        srv.shutdown()


@pytest.mark.parametrize("z, target, rule", [(0.92, 0.95, [2, 0.6]), (0.9000000000000001, 0.9, [1, 0.6]),
                                             (0.5, 0.9, [1, 0.6])])
def test_a_z_takes_the_next_target_at_or_above_it(up, z, target, rule):
    srv, *_ = live(up, profile=profile())
    try:
        e = one(srv, {**BODY, "think_converge": z})
        assert (e["converge"], e["converge_rule"]) == (target, rule)
    finally:
        srv.shutdown()


def test_no_rule_for_z_is_unreachable_and_the_request_is_served_as_if_it_had_not_asked(up):
    for prof in (profile(), load_think_profile(profile_dict()), None):     # .99 has no rule; no block; no profile
        up.requests.clear()
        srv, *_ = live(up, profile=prof)
        try:
            e = one(srv, {**BODY, "think_converge": 0.97})
            assert list(e) == [*V02_KEYS, "converge"] and e["converge"] == "unreachable" and not gens(up)
            e = one(srv, {**BODY, "think_converge": 0.97, "think_budget": 256})
            assert e["converge"] == "unreachable" and "converge_rule" not in e
            assert [g["n_predict"] for g in gens(up)] == [256]                # one generation, as without early exit
        finally:
            srv.shutdown()


def test_an_explicit_budget_0_opts_out_and_media_and_unchunkable_models_fall_back(up):
    srv, *_ = live(up, profile=profile())
    try:
        e = one(srv, {**BODY, "think_converge": 0.9, "think_budget": 0})
        assert list(e) == V02_KEYS and not gens(up)
    finally:
        srv.shutdown()
    vup = FakeUpstream(think="preopened", think_len=25, vision=True).start()
    srv, *_ = live(vup, profile=profile())
    try:
        e = one(srv, {**IMG, "think_converge": 0.9})
        assert e["converge"] == "media" and "think_budget" not in e and not gens(vup)
    finally:
        srv.shutdown()
        vup.stop()
    multi = FakeUpstream(think="preopened", think_len=10_000).start()
    multi.think_multi_tag = True
    srv, *_ = live(multi, profile=profile())
    try:
        assert one(srv, {**BODY, "think_converge": 0.9})["converge"] == "not_chunkable"
    finally:
        srv.shutdown()
        multi.stop()


def test_a_rule_without_t_serves_the_nearest_budget_t_uncalibrated(up):
    tp = profile(**{"0.9": {"cell": [1, 0.6], "T": None}})
    srv, *_ = live(up, profile=tp)
    try:
        e = one(srv, {**BODY, "think_converge": 0.9})
        assert e["think_close"] == "converged"
        assert (e["temperature"], e["calibrated"]) == (tp.temperature_for(e["think_tokens"])[0], False)
    finally:
        srv.shutdown()


def test_early_exit_under_a_deadline_two_questions_each_probe_their_own_trace(up):
    up.think_real_time, up.think_ms_per_token, up.think_prompt_ms_per_token = True, 2.0, 0.05
    srv, *_ = live(up, profile=profile())
    try:
        status, headers, _ = post(srv, {**TWO, "think_converge": 0.9, "think_deadline_ms": 20_000})
        assert status == 200
        es = entries(headers)
        assert len(es) == 2 and all(list(e) == [*CHUNKED_KEYS, *CONVERGE_KEYS] for e in es)
        assert all(e["converge"] == 0.9 and e["think_tokens"] % 32 == 0 for e in es if e["think_close"] == "converged")
        for e in es:                                  # every chunk of a question ends on its own multiple of K
            assert e["chunks"] >= e["probes"]
        assert all(g["n_predict"] <= 32 for g in gens(up))
    finally:
        srv.shutdown()


def test_the_server_default_reaches_every_client_and_0_opts_out(up):
    srv, *_ = live(up, profile=profile(), think_converge=0.9)
    try:
        assert one(srv, BODY)["think_close"] == "converged"
        up.requests.clear()
        assert list(one(srv, {**BODY, "think_converge": 0})) == V02_KEYS and not gens(up)
        e = one(srv, {**BODY, "think_converge": 0, "think_budget": 256})
        assert "converge" not in e and [g["n_predict"] for g in gens(up)] == [256]
    finally:
        srv.shutdown()


def test_the_request_log_names_early_exit_only_when_asked(tmp_path, up):
    log = RequestLog(str(tmp_path / "r.jsonl"))
    srv, *_ = live(up, profile=profile(), request_log=log)
    try:
        one(srv, {**BODY, "think_converge": 0.9})
        one(srv, BODY)
        deadline = time.monotonic() + 5
        while len((tmp_path / "r.jsonl").read_text().splitlines()) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        srv.shutdown()
        log.close()
    a, b = (json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines())
    assert a["converge"] == 0.9 and a["questions"][0]["think_close"] == "converged" and "converge" not in b
    assert list(a)[-2:] == ["converge", "questions"] and list(b)[-2:] == ["budget", "questions"]   # old lines as before


def test_a_deadline_too_short_to_think_still_reports_the_early_exit_it_asked_for(up):
    srv, *_ = live(up, profile=profile())
    try:
        e = one(srv, {**BODY, "think_converge": 0.9, "think_deadline_ms": 1})          # the budget0 fallback
        assert list(e) == [*FALLBACK_KEYS, *CONVERGE_KEYS] and e["deadline_fallback"] == "budget0"   # no think_close
        assert (e["converge"], e["converge_rule"], e["probes"]) == (0.9, [1, 0.6], 0)
    finally:
        srv.shutdown()


def test_early_exit_with_no_context_left_to_think_in_says_it_fell_back_to_budget_0(up):
    """No deadline: the cap clips below one token, the question is read at budget 0, and X-Callo-Timings says so as the
    request log and /callo/stats do."""
    srv, *_, svc = live(up, profile=profile())
    try:
        svc.engine.n_ctx = svc.engine.plan_deliberate(*_one(BODY)) + 16
        e = one(srv, {**BODY, "think_converge": 0.9})
        assert list(e) == [*V02_KEYS, "deadline_fallback", *CONVERGE_KEYS] and e["deadline_fallback"] == "budget0"
        assert (e["converge"], e["converge_rule"], e["probes"]) == (0.9, [1, 0.6], 0) and gens(up) == []
        assert stats(srv)["think"]["budget0"] == 1
    finally:
        srv.shutdown()


class _Reads(Estimators):
    """Estimators that record what reaches the reserve's readout window."""

    def __init__(self):
        super().__init__()
        self.reads: list[float] = []

    def add_readout(self, ms: float) -> None:
        self.reads.append(ms)
        super().add_readout(ms)


def test_a_reused_probe_answer_never_feeds_the_reserve_window_and_a_read_after_the_close_does(up):
    e = engine_for(up)
    q = REQ.questions[0]
    conv = ConvergeRun(0.95, 32, 4, 0.99, 0.0)                    # never stops early on the fake

    def run(est, *, cap, explicit, share_end, clock):
        dl = DeadlineRun(1000, cap, explicit, share_end, None, 0.0, est, lambda c, k, cap: (1.0, False), clock, conv)
        return e._answer_deadline(REQ, q, dl, id_slot=0, cancel=None)[1]

    now = [0.0]
    up.think_gen_hook = lambda body: now.__setitem__(0, 1e6 - 1.0)            # chunk 0 uses up the share
    est = _Reads()
    extra = run(est, cap=1024, explicit=False, share_end=1e6, clock=lambda: now[0])
    assert (extra["think_close"], extra["think_k"], extra["probes"]) == ("deadline", 32, 1)    # the probe on the cut
    assert est.reads == []                                                    # reused: read cost 0, not a readout
    up.think_gen_hook = None
    est = _Reads()
    extra = run(est, cap=20, explicit=True, share_end=1e9, clock=lambda: 0.0)
    assert (extra["think_close"], extra["probes"]) == ("budget", 0) and len(est.reads) == 1   # read after the close


@pytest.mark.parametrize("asks", [True, False])
def test_a_question_rerouted_to_budget_0_reports_the_early_exit_it_asked_for(up, asks):
    """A deadline question queued behind another trace and rerouted to a quiet reserved slot at budget 0 carries
    converge, converge_rule and probes 0 when it asked for early exit, and exactly the fallback keys when it did not."""
    up.think_real_time, up.think_ms_per_token, up.think_prompt_ms_per_token = True, 2.0, 0.05
    srv, *_ = live(up, s1_slots=2, profile=profile())
    lane = {"X-Callo-Conversation": "L"}
    held: dict = {}
    try:
        assert post(srv, {**BODY, "think_budget": 0}, lane)[0] == 200
        t = threading.Thread(target=lambda: held.__setitem__("r", post(srv, {**BODY, "think_deadline_ms": 3000},
                                                                       lane)))
        t.start()                                                       # holds the lane slot's section
        time.sleep(0.15)
        body = {**BODY, "think_deadline_ms": 700, **({"think_converge": 0.9} if asks else {})}
        status, headers, _ = post(srv, body, lane)
        (e,) = entries(headers)
        assert status == 200 and e["deadline_fallback"] == "budget0"
        assert any(kind == "readout" and slot == 1 for kind, slot, _ in up.arrivals)     # read on the other slot
        if asks:
            assert list(e) == [*FALLBACK_KEYS, *CONVERGE_KEYS]
            assert (e["converge"], e["converge_rule"], e["probes"]) == (0.9, [1, 0.6], 0)
        else:
            assert list(e) == FALLBACK_KEYS
        t.join()
        assert held["r"][0] == 200
    finally:
        srv.shutdown()
