"""Several questions and several requests under think deadlines: the even split with no
minimum, the leftover passed on, no phantom stall, the section wait bounded by the share, the budget-0 reroute and
the draining slot after a cancel."""
import copy
import json
import socket
import threading
import time

import pytest
from fake_upstream import FakeUpstream
from test_deadline_serve import entries, live, post, stats
from test_think_serve import BODY

from callo.systemone import parse_request

SLACK_MS = 30                     # measured: these ended 39-60 ms EARLY in 100 runs under 8 busy loops

TWO = {**BODY, "questions": {**BODY["questions"], "r": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}}


def q_closes_at(natural_q: int):
    """The planted length: question q (its options start "A. one") closes at `natural_q` tokens, r thinks on and on."""
    return lambda text: natural_q if "\nA. one" in text else 10_000


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=10_000).start()
    u.think_real_time, u.think_ms_per_token, u.think_prompt_ms_per_token = True, 2.0, 0.05
    yield u
    u.stop()


def test_two_questions_split_the_time_evenly_and_the_second_ends_inside_the_deadline(up):
    srv, *_ = live(up)
    try:
        status, headers, _ = post(srv, {**TWO, "think_deadline_ms": 1200})
        q, r = entries(headers)
        assert status == 200 and q["think_close"] == r["think_close"] == "deadline"
        assert 300 <= q["elapsed_ms"] <= 600 + 40 and 900 <= r["elapsed_ms"] <= 1200 + 40
    finally:
        srv.shutdown()


def test_time_the_first_question_leaves_passes_to_the_second(up):
    up.think_len = q_closes_at(10)
    srv, *_ = live(up)
    try:
        _, headers, _ = post(srv, {**TWO, "think_deadline_ms": 1200})
        q, r = entries(headers)
        assert q["think_close"] == "natural" and q["elapsed_ms"] < 300
        assert r["think_close"] == "deadline" and r["elapsed_ms"] >= 0.8 * 1200      # not just its half
    finally:
        srv.shutdown()


def test_no_minimum_a_half_share_without_room_for_one_token_falls_back_and_the_second_gets_the_rest(up):
    """The only rule that sends a question to budget 0 is "one token does not fit its share"."""
    srv, _, _, svc = live(up)
    try:
        on = svc.engine.plan_deliberate("the state", parse_request(BODY).questions[0])
        e = svc.estimators.estimate(i=0, slot=0, question_steps=[], uncached=on)
        need = e.prefill_ms + e.overhead_ms + e.stall_ms + e.step_ms + svc.estimators.reserve()[0]
        _, headers, _ = post(srv, {**TWO, "think_deadline_ms": int(2 * need - 20)})
        q, r = entries(headers)
        assert q.get("deadline_fallback") == "budget0" and q["chunks"] == 0
        assert "deadline_fallback" not in r and r["chunks"] >= 1
    finally:
        srv.shutdown()


def test_no_phantom_stall_second_questions_on_a_quiet_server_never_fall_back(up):
    """Every chunk 0 carries a prefill, and none of it reaches the stall allowance."""
    srv, _, _, svc = live(up)
    try:
        for _ in range(5):
            _, headers, _ = post(srv, {**TWO, "think_deadline_ms": 1500})
            assert all("deadline_fallback" not in e for e in entries(headers))
        assert svc.estimators.resid95() < 50.0
    finally:
        srv.shutdown()


def test_no_phantom_stall_budget_0_and_deadline_questions_alternating_on_the_same_slot(up):
    """Across the two paths: a budget-0 readout (no note_sent) before each deadline question on the same slot
    leaves the estimators' per-slot token array stale, which must not read as a stall, a fallback or a late answer."""
    srv, _, _, svc = live(up)
    try:
        for _ in range(4):
            assert post(srv, {**TWO, "think_budget": 0})[0] == 200
            status, headers, _ = post(srv, {**TWO, "think_deadline_ms": 1500})
            assert status == 200
            assert all("deadline_fallback" not in e and e["deadline_met"] is True for e in entries(headers))
        assert svc.estimators.resid95() < 50.0
    finally:
        srv.shutdown()


def _later(fn, delay):
    out = {}

    def run():
        time.sleep(delay)
        out["r"] = fn()
    t = threading.Thread(target=run)
    t.start()
    return t, out


def test_a_question_queued_behind_another_trace_is_rerouted_to_a_quiet_reserved_slot_at_budget_0(up):
    srv, *_ = live(up, s1_slots=2)
    lane = {"X-Callo-Conversation": "L"}
    try:
        assert post(srv, {**BODY, "think_budget": 0}, lane)[0] == 200                  # lane L is slot 0's
        a, a_out = _later(lambda: post(srv, {**BODY, "think_deadline_ms": 3000}, lane), 0.0)
        time.sleep(0.15)
        t0 = time.monotonic()
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": 700}, lane)
        took = 1000 * (time.monotonic() - t0)
        (c,) = entries(headers)
        assert status == 200 and c["deadline_fallback"] == "budget0" and c["deadline_met"] is True and took < 700
        assert any(kind == "readout" and slot == 1 for kind, slot, _ in up.arrivals)   # read on the quiet slot
        a.join()
        assert entries(a_out["r"][1])[0]["think_close"] == "deadline"
    finally:
        srv.shutdown()


def test_with_no_quiet_slot_it_waits_on_its_own_and_is_answered_late_never_a_5xx(up):
    srv, *_ = live(up)
    try:
        a, _ = _later(lambda: post(srv, {**BODY, "think_deadline_ms": 2000}), 0.0)
        time.sleep(0.15)
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": 700})
        (c,) = entries(headers)
        assert status == 200 and c["deadline_met"] is False and c["elapsed_ms"] > 700
        assert c["deadline_fallback"] == "budget0"
        a.join()
    finally:
        srv.shutdown()


def test_a_late_question_keeps_its_place_in_the_queue_so_the_one_queued_first_enters_first(up):
    """The bounded wait only decides the reroute; a question that keeps waiting keeps its FIFO place, so B
    (queued first, longer deadline) enters before C (queued 50 ms later, shorter deadline, so it passes its bound
    first and, with a fresh ticket, would have jumped B)."""
    srv, *_ = live(up)
    done: dict = {}

    def run(name, body):
        def go():
            r = post(srv, body)
            done[name] = (time.monotonic(), r)
        return go

    try:
        a, _ = _later(run("A", {**BODY, "think_deadline_ms": 2500}), 0.0)
        time.sleep(0.15)
        b, _ = _later(run("B", {**BODY, "think_deadline_ms": 1200}), 0.0)
        time.sleep(0.05)
        c, _ = _later(run("C", {**BODY, "think_deadline_ms": 500}), 0.0)
        for t in (a, b, c):
            t.join()
        assert all(done[k][1][0] == 200 for k in "ABC")
        assert done["A"][0] < done["B"][0] < done["C"][0]
    finally:
        srv.shutdown()


def test_a_deadline_longer_than_s1_timeout_still_tries_the_reroute_before_the_section_times_out(up):
    """The bound is min(T_q - c_0, s1_timeout - a little): a question whose share outlasts --s1-timeout is rerouted to a
    quiet slot near the timeout rather than 503ing."""
    srv, *_ = live(up, s1_slots=2, s1_timeout=1.0)
    lane = {"X-Callo-Conversation": "L"}
    try:
        assert post(srv, {**BODY, "think_budget": 0}, lane)[0] == 200
        a, _ = _later(lambda: post(srv, {**BODY, "think_deadline_ms": 4000}, lane), 0.0)
        time.sleep(0.15)
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": 3000}, lane)
        assert status == 200 and entries(headers)[0]["deadline_fallback"] == "budget0"
        a.join()
    finally:
        srv.shutdown()


def _abandon(srv, body, after_s):
    s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10)
    data = json.dumps(body).encode()
    s.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
              + str(len(data)).encode() + b"\r\n\r\n" + data)
    time.sleep(after_s)
    s.close()


def test_after_a_cancelled_deliberate_question_route_s1_avoids_the_slot_until_slots_shows_it_idle(up):
    def snapshot():
        with up.lock:
            return copy.deepcopy(up.slots), time.monotonic()

    srv, router, *_ = live(up, s1_slots=2, slots_snapshot=snapshot)
    try:
        _abandon(srv, {**BODY, "think_deadline_ms": 5000}, 0.3)
        deadline = time.monotonic() + 3
        while not up.aborted and time.monotonic() < deadline:
            time.sleep(0.02)
        while router.inflight(0) and time.monotonic() < deadline:
            time.sleep(0.02)
        up.set_slot(0, is_processing=True)                       # the zombie: llama-server has not noticed yet
        up.think_len = 10
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": 2000})
        assert status == 200 and headers["X-Callo-Slot"] == "1" and router.counters["s1_draining"] == 1
        up.set_slot(0, is_processing=False)
        assert post(srv, {**BODY, "think_deadline_ms": 2000})[1]["X-Callo-Slot"] == "0"
    finally:
        srv.shutdown()


# ---- adherence under load ------------------------------------------------------------------------------------------

def test_a_decode_twice_as_slow_mid_trace_still_ends_inside_the_deadline(up):
    """The planner sizes each chunk on the time left (α = 0.5), so a 2x slowdown after the trace started is absorbed."""
    srv, *_ = live(up)
    try:
        threading.Timer(0.3, lambda: setattr(up, "think_ms_per_token", 4.0)).start()
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": 1500})
        (e,) = entries(headers)
        assert status == 200 and e["think_close"] == "deadline" and e["elapsed_ms"] <= 1500 + SLACK_MS
    finally:
        srv.shutdown()


def test_a_prefill_dispatched_on_another_slot_before_the_chunk_is_allowed_for(up):
    """S_known: slot 1's prefill (1,100 ms) is dispatched while chunk 1 runs, so chunk 2 is planned around what is
    left of it and the 600 ms stall it causes in chunk 2 still ends inside the deadline. Without S_known chunk 2 is
    sized as if nothing were coming (153 tokens rather than about 60) and the answer is some 200 ms late."""
    srv, _, _, svc = live(up, s1_slots=2)
    seen: list = []

    def hook(body):                                          # the fake calls it as each generation arrives
        seen.append(body)
        if len(seen) == 2:                                   # chunk 1 arrives: slot 1's prefill is dispatched
            seen.append(svc.estimators.dispatch_start(1, 1100.0))
        if len(seen) == 4:                                   # chunk 2 arrives: that prefill stalls its next step
            up.stall_next(600.0)

    up.think_gen_hook = hook
    try:
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": 1500})
        (e,) = entries(headers)
        assert status == 200 and e["think_close"] == "deadline" and e["elapsed_ms"] <= 1500 + SLACK_MS
        assert up.gen_log[2]["n_predict"] < 100                 # chunk 2 shrank for the known prefill
    finally:
        srv.shutdown()


def test_a_large_stall_after_the_chunk_started_is_answered_late_never_a_5xx(up):
    """An unpredicted stall inside a chunk (a prefill that starts mid-chunk) cannot be planned around: the answer comes
    late, `deadline_met` false, with status 200."""
    srv, *_ = live(up)
    try:
        threading.Timer(0.4, lambda: up.stall_next(1200.0)).start()
        status, headers, _ = post(srv, {**BODY, "think_deadline_ms": 1000})
        (e,) = entries(headers)
        assert status == 200 and e["deadline_met"] is False and e["elapsed_ms"] > 1000
        block = stats(srv)["think"]["deadline"]                  # the deadline block counts it
        assert block["missed"] == 1 and block["met"] == 0
    finally:
        srv.shutdown()


def test_three_deadline_requests_on_three_reserved_slots_all_meet_the_deadline(up):
    """--s1-slots 3: three concurrent deadline requests think on three slots at once, and all end inside D."""
    srv, *_ = live(up, s1_slots=3)
    try:
        runs = [_later(lambda: post(srv, {**BODY, "think_deadline_ms": 1500}), 0.0) for _ in range(3)]
        for t, _ in runs:
            t.join()
        outs = [out["r"] for _, out in runs]
        assert all(status == 200 for status, _, _ in outs)
        assert len({headers["X-Callo-Slot"] for _, headers, _ in outs}) == 3
        assert all(entries(headers)[0]["elapsed_ms"] <= 1500 + SLACK_MS for _, headers, _ in outs)
    finally:
        srv.shutdown()


def test_the_section_is_held_across_every_chunk_of_a_trace(up):
    """The slot's S1 section is held from chunk 0 to the readout, so a budget-0 readout queued on the same slot
    reaches llama-server only after the trace's last chunk, never between two chunks."""
    srv, *_ = live(up)
    try:
        a, a_out = _later(lambda: post(srv, {**BODY, "think_deadline_ms": 1200}), 0.0)
        time.sleep(0.2)
        assert post(srv, {**BODY, "think_budget": 0, "think_deadline_ms": 0})[0] == 200
        a.join()
        assert entries(a_out["r"][1])[0]["chunks"] >= 2
        gens_at = [t for kind, slot, t in up.arrivals if kind == "generation" and slot == 0]
        reads_at = sorted(t for kind, slot, t in up.arrivals if kind == "readout" and slot == 0)
        # the trace's own readout comes first, right after its last chunk; the queued budget-0 readout after it
        assert len(reads_at) >= 2 and reads_at[-1] > max(gens_at) and reads_at[-2] > max(gens_at)
    finally:
        srv.shutdown()
