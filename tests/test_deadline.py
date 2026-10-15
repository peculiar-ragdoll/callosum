"""Think deadlines, the pure half: the planner every deadline trace runs, calibrate's
simulation of it, the even split, the deadline-only cap and the readout temperature per close."""
import pytest
from test_think import profile_dict

from callo.deadline import (
    ALPHA,
    C_MAX,
    CONTEXT_ROOM,
    MAX_DEADLINE_MS,
    N_MIN,
    SIM_OVERHEAD_MS,
    Chunk,
    Estimate,
    check_deadline_points,
    chunk_size,
    clip_cap,
    deadline_cap,
    deadline_temperature,
    feasible,
    run_chunks,
    share_end,
    simulate,
)
from callo.errors import InvalidInput
from callo.think import load_think_profile


class Clock:
    def __init__(self):
        self.ms = 0.0

    def __call__(self):
        return self.ms


def fake_gen(clock, *, step_ms, natural_at=None, eos=False, prefill_ms=0.0, overhead_ms=0.0, log=None):
    """A generate callback that advances `clock` as a chunk would: chunk 0 pays `prefill_ms`, each chunk `overhead_ms`,
    each token `step_ms`; the trace closes on its own at `natural_at` tokens (a word stop, or eos)."""
    def gen(i, n, so_far, e):
        k = len(so_far)
        take = n if natural_at is None else min(n, natural_at - k)
        clock.ms += (prefill_ms if i == 0 else 0.0) + overhead_ms + take * step_ms
        if log is not None:
            log.append((i, n, e))
        stop = ("eos" if eos else "word") if natural_at is not None and k + take >= natural_at else "limit"
        return Chunk(tuple(range(k, k + take)), stop, take * step_ms, 0.0, take * step_ms, 1, k, n)
    return gen


def test_the_constants_are_the_designed_values():
    assert (ALPHA, N_MIN, C_MAX) == (0.5, 4, 256)
    assert (SIM_OVERHEAD_MS, MAX_DEADLINE_MS, CONTEXT_ROOM) == (5.0, 600_000, 16)


def test_chunks_halve_with_the_time_left_and_the_trace_closes_with_deadline_below_n_min():
    """R = 1600 ms at 10 ms/token: 80, 40, 20, 10, 5 tokens, then floor(0.5 * 50 / 10) = 2 < n_min: `deadline`."""
    clock, log = Clock(), []
    run = run_chunks(fake_gen(clock, step_ms=10.0, log=log), lambda i, c: Estimate(0.0, 10.0, 0.0), cap=1024,
                     share_end_ms=1600.0, reserve_ms=0.0, clock=clock)
    assert [n for _, n, _ in log] == [80, 40, 20, 10, 5]
    assert run.close == "deadline" and len(run.tokens) == 155 and len(run.chunks) == 5 and not run.fallback
    assert clock.ms == 1550.0                                    # the tail given up: < n_min * d / alpha (80 ms)


def test_chunk_0_pays_the_prefill_and_is_at_least_one_token():
    e = Estimate(prefill_ms=300.0, step_ms=10.0, overhead_ms=5.0)
    assert chunk_size(0, 0, 1024, 1000.0, e) == 34                # floor(0.5 * (1000 - 300 - 5) / 10)
    assert chunk_size(1, 34, 1024, 1000.0, e) == 49               # later chunks do not pay P̂ again
    assert chunk_size(0, 0, 1024, 316.0, e) == 1                  # step 2 passed: chunk 0 gets one token whatever
    assert chunk_size(1, 1, 1024, 40.0, e) == 0                   # floor(0.5 * 35 / 10) = 1 < n_min
    assert chunk_size(0, 0, 1024, 10**9, e) == C_MAX


def test_the_cap_bounds_a_chunk_but_never_closes_with_deadline():
    e = Estimate(0.0, 10.0, 0.0)
    assert chunk_size(1, 1022, 1024, 10_000.0, e) == 2            # two tokens left under the cap, time to spare
    clock = Clock()
    run = run_chunks(fake_gen(clock, step_ms=1.0), lambda i, c: e, cap=6, share_end_ms=10_000.0, reserve_ms=0.0,
                     clock=clock)
    assert run.close == "budget" and len(run.tokens) == 6


def test_feasibility_needs_exactly_one_token_of_room_and_otherwise_falls_back():
    """R >= P̂ + ô + G_0 + d̂ thinks; a millisecond less is the budget-0 fallback, with no chunk sent."""
    e = Estimate(prefill_ms=100.0, step_ms=10.0, overhead_ms=5.0, stall_ms=20.0)
    assert feasible(135.0, e) and not feasible(134.999, e)
    for reserve, think in ((865.0, True), (865.001, False)):     # R = 1000 - reserve
        clock, calls = Clock(), []
        run = run_chunks(fake_gen(clock, step_ms=10.0, prefill_ms=100.0, log=calls), lambda i, c: e, cap=64,
                         share_end_ms=1000.0, reserve_ms=reserve, clock=clock)
        assert run.fallback is (not think) and bool(calls) is think
        assert run.close == (None if not think else "deadline") and (run.tokens == ()) is (not think)


def test_every_close_reason():
    e = Estimate(0.0, 1.0, 0.0)
    for natural_at, eos, cap, close, k in ((30, False, 1024, "natural", 30), (30, True, 1024, "eos", 30),
                                           (None, False, 40, "budget", 40), (None, False, 10_000, "deadline", None)):
        clock = Clock()
        run = run_chunks(fake_gen(clock, step_ms=1.0, natural_at=natural_at, eos=eos), lambda i, c: e, cap=cap,
                         share_end_ms=500.0, reserve_ms=0.0, clock=clock)
        assert run.close == close and (k is None or len(run.tokens) == k)


def test_a_decode_twice_as_slow_as_planned_still_ends_inside_the_share():
    """The bound: alpha = 0.5 tolerates a 2x slower decode, so the trace never runs past T_q - r."""
    clock = Clock()
    run_chunks(fake_gen(clock, step_ms=20.0), lambda i, c: Estimate(0.0, 10.0, 0.0), cap=4096, share_end_ms=2000.0,
               reserve_ms=50.0, clock=clock)
    assert clock.ms <= 1950.0


def test_the_estimate_is_asked_per_chunk_and_the_check_runs_before_each():
    clock, asked, checks = Clock(), [], []

    def estimate(i, chunks):
        asked.append((i, len(chunks)))
        return Estimate(0.0, 10.0, 0.0, stall_ms=10.0 * i, known_ms=5.0 * i)

    run_chunks(fake_gen(clock, step_ms=10.0), estimate, cap=1024, share_end_ms=1000.0, reserve_ms=0.0, clock=clock,
               check=lambda: checks.append(clock.ms))
    assert asked[0] == (0, 0) and all(i == n for i, n in asked[1:]) and len(checks) == len(asked) - 1


def test_the_simulation_matches_a_hand_count():
    """a = 400 ms, d = 10 ms/token, D = 2000, r = 50: chunk 0 floor(0.5 * (1950 - 400 - 5) / 10) = 77, then the
    continuation chunks at ô = 5 until the next one is below n_min."""
    s = simulate(a_ms=400.0, d_ms=10.0, n_gen=900, close="forced", deadline_ms=2000.0, reserve_ms=50.0, cap=1024)
    assert s.close == "deadline" and s.chunks == 5 and s.k == 77 + 38 + 19 + 9 + 5
    assert s.elapsed_ms == pytest.approx(400 + 4 * 5 + 148 * 10.0)
    nat = simulate(a_ms=400.0, d_ms=10.0, n_gen=60, close="natural", deadline_ms=2000.0, reserve_ms=50.0, cap=1024)
    assert (nat.close, nat.k, nat.chunks) == ("natural", 60, 1) and nat.elapsed_ms == pytest.approx(1000.0)
    eos = simulate(a_ms=400.0, d_ms=10.0, n_gen=100, close="eos", deadline_ms=2000.0, reserve_ms=50.0, cap=1024)
    assert (eos.close, eos.k) == ("eos", 100)
    capped = simulate(a_ms=400.0, d_ms=10.0, n_gen=64, close="forced", deadline_ms=60_000.0, reserve_ms=50.0, cap=64)
    assert (capped.close, capped.k) == ("budget", 64)
    late = simulate(a_ms=400.0, d_ms=10.0, n_gen=900, close="forced", deadline_ms=460.0, reserve_ms=50.0, cap=1024)
    assert (late.close, late.k, late.chunks, late.elapsed_ms) == (None, 0, 0, 0.0)          # a + ô + d > D - r


def test_the_even_split_has_no_minimum():
    """Each remaining deliberate question gets an equal share of the time left."""
    assert share_end(100.0, 2100.0, 2) == 1100.0 and share_end(1100.0, 2100.0, 1) == 2100.0
    assert share_end(500.0, 2100.0, 1) == 2100.0                  # the first ended at 500 ms, 600 ms early: it all passes on
    assert share_end(500.0, 2100.0, 2) == 1300.0                  # a request that starts 500 ms late splits what is left
    assert share_end(0.0, 900.0, 0) == 900.0


def test_the_deadline_only_cap_is_the_profiles_largest_budget_else_think_max_clipped_to_the_context():
    assert deadline_cap({0, 256, 1024}, 4096) == 1024 and deadline_cap(None, 4096) == 4096
    assert deadline_cap({0}, 2048) == 2048
    assert deadline_cap({0, 256, 1024}, 512) == 512 and deadline_cap({256}, 4096) == 256      # a deadline never raises a cap
    assert clip_cap(1024, 8192, 7500) == 676 and clip_cap(1024, None, 7500) == 1024 and clip_cap(1024, 8192, 100) == 1024


def _tp(**over):
    return load_think_profile(profile_dict(**over))


def test_the_readout_temperature_per_close():
    """A deadline point whose cap is reached wins; a natural or budget close takes its cap's T; a deadline cut its
    k's nearest lower budget, never calibrated; no profile is T = 1, uncalibrated."""
    tp = _tp(deadlines={"2000": {"T": 0.77, "cap": 1024}})
    assert deadline_temperature("deadline", 90, cap=1024, deadline_ms=2000, profile=tp) == (0.77, True)
    assert deadline_temperature("natural", 40, cap=1024, deadline_ms=2000, profile=tp) == (0.77, True)
    assert deadline_temperature("natural", 40, cap=512, deadline_ms=2000, profile=tp) == (1.1, False)   # cap 512 < 1024
    assert deadline_temperature("budget", 256, cap=256, deadline_ms=1500, profile=tp) == (1.1, True)
    assert deadline_temperature("deadline", 300, cap=1024, deadline_ms=1500, profile=tp) == (1.1, False)
    assert deadline_temperature("deadline", 100, cap=1024, deadline_ms=1500, profile=tp) == (1.1, False)   # below 256
    assert deadline_temperature("eos", 50, cap=1024, deadline_ms=1500, profile=tp) == (0.9, True)
    assert deadline_temperature("deadline", 100, cap=1024, deadline_ms=2000, profile=None) == (1.0, False)


def test_a_profiles_deadline_points_load_and_a_malformed_block_is_refused():
    tp = _tp(deadlines={"2000": {"T": 0.8, "cap": 1024, "n": 3}})
    assert tp.deadline_point(2000) == {"T": 0.8, "cap": 1024, "n": 3} and tp.deadline_point(1000) is None
    assert _tp().deadlines == {} and _tp().latency == {}
    assert _tp(latency={"read_forced_ms": {"p50": 1, "p95": 2, "p99": 3}}).latency["read_forced_ms"]["p95"] == 2
    for bad, why in (({"x": {"T": 1, "cap": 1}}, "deadline key"), ({"0": {"T": 1, "cap": 1}}, "deadline key"),
                     ({"10": {"T": 0, "cap": 1}}, "finite positive T"), ({"10": {"T": 1, "cap": True}}, "cap >= 1"),
                     ([1], "must be an object"), ({"10": {"T": 10**400, "cap": 1}}, "finite positive T"),
                     ({"1" * 5000: {"T": 1, "cap": 1}}, "deadline key"),
                     ({"02000": {"T": 1, "cap": 1}, "2000": {"T": 1, "cap": 1}}, "deadline key")):
        assert check_deadline_points(bad)[0] is None
        with pytest.raises(InvalidInput, match=why):
            _tp(deadlines=bad)


def test_a_malformed_latency_block_is_refused_and_a_null_read_forced_is_valid():
    ok = {"p50": 1, "p95": 2, "p99": 3}
    assert _tp(latency={"read_forced_ms": None, "frame_ms": ok}).latency["read_forced_ms"] is None
    for bad, why in (([1], "latency must be an object"), ({"frame_ms": 5}, "frame_ms must be null or an object"),
                     ({"frame_ms": {"p50": 1, "p95": 2}}, r"frame_ms\.p99"),
                     ({"read_forced_ms": {**ok, "p95": -1}}, r"read_forced_ms\.p95"),
                     ({"on_tokens": {**ok, "p50": float("inf")}}, r"on_tokens\.p50"),
                     ({"fixed_ms": {**ok, "p99": 10**400}}, r"fixed_ms\.p99"),
                     ({"fixed_ms": {**ok, "p99": True}}, r"fixed_ms\.p99")):
        with pytest.raises(InvalidInput, match=why):
            _tp(latency=bad)


def test_estimators_from_profile_seeds_latency_and_the_budget_0_p95_in_one_place(tmp_path):
    import json

    from callo.deadline import Estimators

    d = profile_dict()
    d["budgets"]["0"]["lat_p95_ms"] = [250.0, 260.0]
    p = tmp_path / "p.json"
    p.write_text(json.dumps(d))
    tp = load_think_profile(str(p))
    e = Estimators.from_profile(tp, reserve_ms=60)
    assert e._b0_seed == 250.0 and e._latency == dict(tp.latency) and e._reserve_flag == 60
    bare = Estimators.from_profile(None)
    assert bare._b0_seed is None and bare._latency == {} and bare._reserve_flag is None
    d["budgets"]["0"]["lat_p95_ms"] = [True]
    p.write_text(json.dumps(d))
    assert Estimators.from_profile(load_think_profile(str(p)))._b0_seed is None
