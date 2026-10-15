"""Early exit's planner pieces: a probe that closes the trace, the probe cost charged like the per-chunk overhead when
a probe is due, and the server-wide probe windows /callo/stats and E1 read."""
from dataclasses import replace

import pytest
from test_think import profile_dict

from callo.deadline import Chunk, Estimate, Estimators, chunk_size, feasible, run_chunks, with_probe_cost
from callo.think import load_think_profile


def _chunk(n, *, wall=10.0, prompt_n=1, predicted=None, stop="limit"):
    return Chunk(tuple(range(n)), stop, wall, 0.1, float(n) if predicted is None else predicted, prompt_n, None, n)


def test_after_may_close_the_trace_and_none_keeps_today_s_closes():
    seen = []

    def after(so_far):
        seen.append(len(so_far))
        return "converged" if len(so_far) == 64 else None

    run = run_chunks(lambda i, n, so_far, e: _chunk(min(n, 32)), lambda i, c: Estimate(0.0, 1.0, 0.0), cap=1024,
                     share_end_ms=1e9, reserve_ms=0.0, clock=lambda: 0.0, after=after)
    assert (run.close, len(run.tokens), len(run.chunks), seen) == ("converged", 64, 2, [32, 64])
    run = run_chunks(lambda i, n, so_far, e: _chunk(min(n, 32)), lambda i, c: Estimate(0.0, 1.0, 0.0), cap=96,
                     share_end_ms=1e9, reserve_ms=0.0, clock=lambda: 0.0, after=lambda so_far: None)
    assert (run.close, len(run.tokens)) == ("budget", 96)


E = Estimate(0.0, 10.0, 5.0)


def test_a_due_probe_is_charged_its_cost_beyond_the_chunk_overhead_it_already_pays():
    # c = 30 ms holds the next chunk's ô (5 ms), which the planner charges every chunk anyway: 25 ms more, never 30
    assert with_probe_cost(E, i=1, k=32, cap=1024, left_ms=None, every=32, c_ms=30.0) == replace(E, probe_ms=25.0)
    assert with_probe_cost(E, i=1, k=32, cap=1024, left_ms=10_000.0, every=32, c_ms=30.0).probe_ms == 25.0
    assert with_probe_cost(E, i=1, k=32, cap=1024, left_ms=None, every=32, c_ms=3.0).probe_ms == 0.0
    # the planner's chunk (14 tokens at 295 ms of room) stops short of the boundary: no probe is due
    assert with_probe_cost(E, i=1, k=32, cap=1024, left_ms=300.0, every=32, c_ms=30.0) == E
    # the next boundary is at the cap: the trace closes with budget there and reads no probe
    assert with_probe_cost(E, i=1, k=1000, cap=1024, left_ms=None, every=32, c_ms=30.0) == E


def test_the_probe_charge_shrinks_the_chunk_but_never_the_feasibility_of_one_token():
    charged = replace(E, probe_ms=1000.0)
    assert chunk_size(1, 32, 1024, 2000.0, charged) == 49 < chunk_size(1, 32, 1024, 2000.0, E) == 99
    assert feasible(40.0, charged) == feasible(40.0, E) is True        # one token needs no probe: no budget-0 push


def test_the_probe_cost_is_the_profile_s_until_8_chunks_followed_a_probe_then_live():
    assert Estimators().probe_cost() == (None, None)
    est = Estimators(probe_seed_ms=40.0)
    assert est.probe_cost() == (40.0, "profile")
    for _ in range(8):
        est.add_chunk(1, _chunk(32, wall=330.0, predicted=320.0))      # a step of (0.1 + 320) / 32 ms per token
        est.add_probe_read(25.0)
        est.add_after_probe(_chunk(32, wall=340.0, prompt_n=1))
    c, source = est.probe_cost()
    assert source == "live" and c == pytest.approx(25.0 + 340.0 - (0.1 + 320.0))
    est.add_after_probe(_chunk(32, wall=340.0, prompt_n=40))           # a re-prefill
    s = est.converge_stats()
    assert s["after_probe"] == {"n": 9, "restored": 8, "share": 8 / 9}
    assert s["probe_read_ms"] == {"p50": 25.0, "p95": 25.0} and s["c_source"] == "live"


def test_from_profile_seeds_the_probe_cost_with_the_measured_c():
    tp = load_think_profile(profile_dict(converge={"K": 32, "cost": {"c_ms": 31.5}}))
    assert Estimators.from_profile(tp).probe_cost() == (31.5, "profile")
    assert Estimators.from_profile(load_think_profile(profile_dict())).probe_cost() == (None, None)


def test_with_probe_cost_refuses_a_probe_interval_below_1():
    for every in (0, -32):
        with pytest.raises(ValueError, match="every must be >= 1"):
            with_probe_cost(E, i=1, k=32, cap=1024, left_ms=None, every=every, c_ms=30.0)


def _view(est):
    e = est.estimate(i=1, slot=0, question_steps=[10.0], uncached=0)
    s = est.stats()
    return (est.overhead(), est.resid95(), e.overhead_ms, e.stall_ms, s["chunk_overhead_ms"], s["stall_ms"])


def test_a_chunk_after_a_probe_feeds_neither_the_residual_stall_nor_the_chunk_overhead():
    est, plain = Estimators(), Estimators()
    for _ in range(8):
        for x in (est, plain):
            x.add_chunk(1, _chunk(32, wall=340.0, predicted=320.0))
    before = _view(est)
    for _ in range(40):                                                 # each carries a 2 s probe excess
        est.add_chunk(1, _chunk(32, wall=2340.0, predicted=320.0), after_probe=True)
        plain.add_chunk(1, _chunk(32, wall=2340.0, predicted=320.0))
    assert _view(est) == before
    assert _view(plain) != before                                       # without the flag they would have moved


def test_the_live_probe_excess_is_net_of_the_known_stall_g_already_charges():
    est = Estimators()
    for _ in range(8):
        est.add_chunk(1, _chunk(32, wall=330.0, predicted=320.0))
        est.add_probe_read(25.0)
        est.add_after_probe(replace(_chunk(32, wall=340.0), known_ms=15.0))
    c, source = est.probe_cost()
    assert source == "live" and c == pytest.approx(25.0 + 340.0 - (0.1 + 320.0) - 15.0)
    floor = Estimators()
    for _ in range(8):
        floor.add_chunk(1, _chunk(32, wall=330.0, predicted=320.0))
        floor.add_probe_read(25.0)
        floor.add_after_probe(replace(_chunk(32, wall=340.0), known_ms=500.0))
    assert floor.probe_cost() == (25.0, "live")                         # the excess never goes below 0


LATENCY = {"decode_ms_per_token": {"p50": 12.0, "p95": 15.0, "p99": 20.0},
           "prefill_ms_per_token": {"p50": 1.0, "p95": 1.5, "p99": 2.0},
           "read_forced_ms": {"p50": 40.0, "p95": 60.0, "p99": 80.0}}


def test_a_converge_block_changes_no_estimate_on_a_server_without_early_exit():
    def trail(converge):
        over = {"latency": LATENCY} | ({"converge": converge} if converge is not None else {})
        est = Estimators.from_profile(load_think_profile(profile_dict(**over)))
        seen = []
        for r in range(12):
            seen.append((est.estimate(i=0, slot=0, question_steps=[], uncached=100),
                         est.estimate(i=1, slot=0, question_steps=[10.0 + r], uncached=0), est.reserve(), est.c0()))
            est.add_chunk(0, _chunk(32, wall=400.0 + r, prompt_n=100, predicted=310.0))
            est.add_chunk(1, _chunk(32, wall=340.0 + 3 * r, predicted=320.0))
            est.add_readout(50.0 + r)
            est.add_budget0(200.0 + 5 * r)
        return seen, est.stats()

    assert trail({"K": 32, "cost": {"c_ms": 31.5}}) == trail(None)


def test_a_chunk_after_a_probe_that_re_prefilled_feeds_d_hat_its_decode_steps_only():
    # a checkpoint restore that missed re-evaluates the trace: 4000 tokens at 1 ms in its prompt phase. Its step is
    # predicted_ms / (n - 1), as chunk 0's, never (prompt_ms + predicted_ms) / n, which would put the prefill in d̂
    est, plain = Estimators(), Estimators()
    for _ in range(8):
        for x in (est, plain):
            x.add_chunk(1, _chunk(32, wall=330.0, predicted=320.0))
    missed = Chunk(tuple(range(32)), "limit", 4400.0, 4000.0, 31 * 10.0, 4000, None, 32)
    for _ in range(10):
        assert est.add_chunk(1, missed, after_probe=True) == pytest.approx(10.0)
        plain.add_chunk(1, _chunk(32, wall=330.0, predicted=320.0))
    assert est.stats()["decode_ms_per_token"] == pytest.approx(plain.stats()["decode_ms_per_token"], abs=0.01)
    assert est.add_chunk(1, replace(missed, tokens=(0,), predicted_ms=0.0), after_probe=True) is None
    # a restored one (prompt_n <= 2) evaluated only its first step in the prompt phase: (prompt_ms + predicted_ms) / n
    restored = replace(missed, wall_ms=330.0, prompt_ms=10.0, prompt_n=2)
    assert est.add_chunk(1, restored, after_probe=True) == pytest.approx((10.0 + 310.0) / 32)
