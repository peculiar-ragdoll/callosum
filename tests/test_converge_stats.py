"""/callo/stats under early exit: `converged` beside the other closes (also in think.deadline.closes), and
think.converge's counters, stop positions, the tokens-saved bound, the probe windows and the probe cost; temperature and
bias calibration refuse a server whose default thinks with early exit."""
import threading
import time

import pytest
from fake_callo import default_stats
from fake_upstream import FakeUpstream
from test_converge_serve import BODY, one, profile
from test_deadline_serve import entries, live, post, stats
from test_think_serve import IMG, TWO

from callo import calibrate_bias
from callo.think import think_default_refusal

CLOSES = ("natural", "budget", "eos", "converged", "budget0")


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=10_000).start()
    yield u
    u.stop()


def closes_sum(think):
    """The top-level closes plus `deadline`, which is published in think.deadline.closes only."""
    return sum(think[c] for c in CLOSES) + think["deadline"]["closes"]["deadline"]


def test_a_server_without_early_exit_publishes_an_idle_block(up):
    srv, *_ = live(up)
    try:
        one(srv, BODY)
        s = stats(srv)["think"]
        assert s["converged"] == 0 and s["deadline"]["closes"]["converged"] == 0
        cv = s["converge"]
        assert (cv["default_z"], cv["K"], cv["rules"], cv["questions"], cv["tokens_saved_mean"]) == (0.0, None, None, 0, None)
        assert cv["after_probe"] == {"n": 0, "restored": 0, "share": None} and cv["c_ms"] is None
    finally:
        srv.shutdown()


def test_think_converge_counts_questions_stops_probes_and_the_bound_on_tokens_saved(up):
    srv, _, st, _ = live(up, profile=profile())
    st.converge_static = {"default_z": 0.0, "K": 32, "rules": {"0.9": {"n": 1, "y": 0.6, "T": 2.0}}}
    try:
        a = one(srv, {**BODY, "think_converge": 0.9})
        b = one(srv, {**BODY, "think_converge": 0.9, "think_deadline_ms": 600_000})
        one(srv, {**BODY, "think_converge": 0.97})                        # unreachable
        one(srv, BODY)                                                     # not asked
        s = stats(srv)["think"]
        assert s["converged"] == 2 and s["deadline"]["closes"]["converged"] == 1
        cv = s["converge"]
        assert (cv["requests"], cv["questions"], cv["converged"], cv["unreachable"]) == (2, 2, 2, 1)
        assert cv["probes"] == a["probes"] + b["probes"] and cv["probes_per_question"] == cv["probes"] / 2
        assert cv["tokens_saved"] == (1024 - a["think_tokens"]) + (1024 - b["think_tokens"])
        assert cv["stop_k"]["p50"] in (a["think_tokens"], b["think_tokens"]) and cv["probe_read_ms"]["p50"] > 0
        assert cv["K"] == 32 and (cv["c_ms"], cv["c_source"]) == (None, None)    # no profile seed, no chunk after
        assert sum(s["deadline"]["closes"].values()) == s["deadline"]["questions"]
        assert closes_sum(s) == s["questions"] == 2
    finally:
        srv.shutdown()


def test_a_budget_0_fallback_that_asked_counts_as_a_question_with_no_probes(up):
    srv, *_ = live(up, profile=profile())
    try:
        e = one(srv, {**BODY, "think_converge": 0.9, "think_deadline_ms": 1})
        assert e["deadline_fallback"] == "budget0"
        s = stats(srv)["think"]
        cv = s["converge"]
        assert (cv["requests"], cv["questions"], cv["probes"], cv["probes_per_question"]) == (1, 1, 0, 0.0)
        assert (cv["converged"], cv["stop_k"], cv["tokens_saved"], cv["tokens_saved_mean"]) == (
            0, {"p50": None, "p95": None}, 0, None)
        assert s["budget0"] == 1 and s["deadline"]["closes"]["budget0"] == 1
        assert closes_sum(s) == s["questions"] == 1
    finally:
        srv.shutdown()


def test_a_rerouted_question_that_asked_counts_as_a_question_with_no_probes(up):
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
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": 700, "think_converge": 0.9}, lane)
        (e,) = entries(headers)
        assert status == 200 and e["deadline_fallback"] == "budget0"
        t.join()
        assert held["r"][0] == 200
        s = stats(srv)["think"]
        cv = s["converge"]
        assert s["deadline"]["rerouted"] == 1
        assert (cv["requests"], cv["questions"], cv["probes"], cv["converged"]) == (1, 1, 0, 0)  # the held one never asked
        assert closes_sum(s) == s["questions"] == 2
    finally:
        srv.shutdown()


def test_media_and_not_chunkable_count_once_per_question_that_could_think():
    vup = FakeUpstream(think="preopened", think_len=25, vision=True).start()
    srv, *_ = live(vup, profile=profile())
    try:
        one(srv, {**IMG, "think_converge": 0.9})
        cv = stats(srv)["think"]["converge"]
        assert (cv["media"], cv["not_chunkable"], cv["unreachable"], cv["requests"], cv["questions"]) == (1, 0, 0, 0, 0)
    finally:
        srv.shutdown()
        vup.stop()
    multi = FakeUpstream(think="preopened", think_len=10_000).start()
    multi.think_multi_tag = True
    srv, *_ = live(multi, profile=profile())
    try:
        assert post(srv, {**TWO, "think_converge": 0.9})[0] == 200
        cv = stats(srv)["think"]["converge"]
        assert (cv["not_chunkable"], cv["media"], cv["requests"], cv["questions"]) == (2, 0, 0, 0)
    finally:
        srv.shutdown()


def test_temperature_and_bias_calibrate_refuse_a_server_with_an_early_exit_default():
    st = {**default_stats(), "think": {"default_budget": 0, "deadline": {"default_ms": 0},
                                       "converge": {"default_z": 0.95}}}
    assert "thinks by default with early exit (--think-converge 0.95)" in think_default_refusal(st)
    with pytest.raises(calibrate_bias.Refusal, match="early exit"):
        calibrate_bias.check_preconditions(st, "http://x")
    assert think_default_refusal({"think": {"default_budget": 0, "converge": {"default_z": 0.0}}}) is None
    assert think_default_refusal({"think": {"default_budget": 0}}) is None             # older than early exit


@pytest.mark.parametrize("cv, what", [(None, "think.converge is None, not an object"),
                                      ([0.9], "think.converge is [0.9], not an object"),
                                      ({"default_z": "0.9"}, "default_z is '0.9', not a finite number"),
                                      ({"default_z": True}, "default_z is True, not a finite number"),
                                      ({"default_z": None}, "default_z is None, not a finite number"),
                                      ({"default_z": float("nan")}, "default_z is nan, not a finite number"),
                                      ({"default_z": float("inf")}, "default_z is inf, not a finite number")])
def test_a_converge_block_callo_cannot_read_is_a_refusal(cv, what):
    st = {**default_stats(), "think": {"default_budget": 0, "converge": cv}}
    refusal = think_default_refusal(st)
    assert refusal.startswith("cannot tell whether the server thinks by default with early exit") and what in refusal
    with pytest.raises(calibrate_bias.Refusal, match="cannot tell"):
        calibrate_bias.check_preconditions(st, "http://x")
