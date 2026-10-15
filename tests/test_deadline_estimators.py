"""Think deadlines' server-wide estimators: the readout reserve, d̂, P̂ (with its pure-attention test),
ô, the stall allowance G = S_known + s95_resid and c_0, each with its seeds."""
import pytest

from callo.deadline import (
    DEFAULT_C0_MS,
    DEFAULT_OVERHEAD_MS,
    DEFAULT_PREFILL_MS,
    DEFAULT_RESERVE_MS,
    DEFAULT_STEP_MS,
    Chunk,
    Estimate,
    Estimators,
    feasible,
)

LATENCY = {"read_forced_ms": {"p50": 26.9, "p95": 27.4, "p99": 28.5}, "decode_ms_per_token": {"p50": 10.7, "p95": 10.9},
           "prefill_ms_per_token": {"p50": 1.2, "p95": 1.6}}


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def chunk(n, *, wall, prompt_ms=0.0, predicted_ms=None, prompt_n=1, known=0.0):
    predicted = 10.0 * n if predicted_ms is None else predicted_ms
    return Chunk(tuple(range(n)), "limit", wall, prompt_ms, predicted, prompt_n, 0, n, known_ms=known)


def test_the_reserve_seeds_learns_floors_and_takes_the_flag():
    assert Estimators().reserve() == (DEFAULT_RESERVE_MS, "default")
    assert Estimators(latency=LATENCY).reserve() == pytest.approx((1.5 * 27.4 + 10, "profile"))
    assert Estimators(latency=LATENCY, reserve_ms=40).reserve() == (40.0, "flag")
    e = Estimators(latency=LATENCY)
    for ms in (20.0,) * 7:
        e.add_readout(ms)
    assert e.reserve()[1] == "profile"                          # 7 live readouts: still the seed
    e.add_readout(40.0)
    assert e.reserve() == (1.5 * 40.0 + 10.0, "learned")        # p95 of 8 readouts: the largest
    tiny = Estimators()
    for _ in range(8):
        tiny.add_readout(1.0)
    assert tiny.reserve() == (20.0, "learned")                  # the 20 ms floor


def test_c0_is_the_budget0_window_p95_else_the_profiles_else_300():
    assert Estimators().c0() == DEFAULT_C0_MS and Estimators(budget0_p95_ms=280.0).c0() == 280.0
    e = Estimators(budget0_p95_ms=280.0)
    for ms in range(100, 900, 100):
        e.add_budget0(float(ms))
    assert e.c0() == 800.0


def test_the_step_seed_is_the_recent_window_else_the_profile_else_30():
    clock = Clock()
    e = Estimators(latency=LATENCY, clock=clock)
    assert e.step_seed() == 10.7 and Estimators().step_seed() == DEFAULT_STEP_MS
    e.add_chunk(0, chunk(10, wall=200.0, predicted_ms=108.0))   # 12 ms per token (9 decode steps), finished now
    assert e.step_seed() == 12.0
    clock.t += 2.5                                              # older than 2 s: the profile again
    assert e.step_seed() == 10.7


def test_a_questions_step_counts_a_continuations_prompt_phase_and_is_capped_at_3x_the_window_median():
    """The llama-server's predicted_ms leaves out the first sampled token, which comes from the prompt phase. A
    continuation's 1-token prompt phase is that step, so (prompt_ms + predicted_ms) / n; chunk 0's prompt phase is the
    prefill, so predicted_ms / (n - 1), and a 1-token chunk 0 measures no step."""
    e = Estimators()
    assert e.add_chunk(0, chunk(50, wall=900.0, prompt_ms=400.0, predicted_ms=490.0, prompt_n=300)) == 10.0
    assert e.add_chunk(0, chunk(1, wall=410.0, prompt_ms=400.0, predicted_ms=0.0, prompt_n=300)) is None
    assert e.stats()["decode_ms_per_token"]["p50"] == 10.0                # the 1-token chunk 0 left no step behind
    assert e.add_chunk(1, chunk(20, wall=230.0, prompt_ms=10.0, predicted_ms=190.0)) == 10.0
    assert e.step_for([10.0, 11.0, 30.0, 12.0]) == 12.0         # the median of the last 3
    assert e.step_for([100.0]) == 30.0                          # capped: 3 x the window median (10)
    assert e.step_for([]) == e.step_seed()


def test_the_prefill_rate_learns_from_chunk_0s_over_32_tokens_after_8():
    e = Estimators(latency=LATENCY)
    assert e.prefill_rate() == 1.6 and Estimators().prefill_rate() == DEFAULT_PREFILL_MS
    for _ in range(8):
        e.add_chunk(0, chunk(4, wall=10.0, prompt_ms=10.0, prompt_n=20))          # 20 tokens: not a prefill sample
    assert e.prefill_rate() == 1.6
    for _ in range(8):
        e.add_chunk(0, chunk(4, wall=400.0, prompt_ms=300.0, prompt_n=300))
    assert e.prefill_rate() == pytest.approx(1.0)


def test_the_uncached_tokens_are_the_whole_prompt_until_the_cache_behaves_like_pure_attention():
    """The LCP candidate is used only while >= 90 % of the last 32 chunk 0s matched it within 8 tokens."""
    e = Estimators()
    prompt = list(range(100))
    e.note_sent(0, list(range(60)) + [999])
    assert e.uncached(0, prompt) == 100                          # no evidence yet: the whole prompt
    for _ in range(8):
        e.add_chunk0_prompt(0, prompt, 44)                       # candidate 40, measured 44: within 8
    assert e.uncached(0, prompt) == 40 and e.uncached(1, prompt) == 100     # slot 1: nothing sent there
    for _ in range(2):
        e.add_chunk0_prompt(0, prompt, 100)                      # a hybrid-like rollback: 8 of 10 < 90 %
    assert e.uncached(0, prompt) == 100


def test_the_overhead_is_an_ewma_after_8_chunks():
    e = Estimators()
    assert e.overhead() == DEFAULT_OVERHEAD_MS
    for _ in range(8):
        e.add_chunk(1, chunk(10, wall=112.0, prompt_ms=0.0, predicted_ms=100.0))
    assert e.overhead() == pytest.approx(12.0)


def test_s_known_counts_other_slots_dispatched_prefill_minus_its_elapsed_time():
    clock = Clock()
    e = Estimators(clock=clock)
    a = e.dispatch_start(1, 300.0)
    e.dispatch_start(2, 100.0)
    e.dispatch_start(0, 500.0)                                   # the asking slot's own: never counted
    assert e.known(0) == 400.0
    clock.t += 0.15
    assert e.known(0) == pytest.approx(150.0)                    # 300 - 150 and 100 - 150 -> 0
    e.dispatch_end(a)
    assert e.known(0) == 0.0 and e.known(None) == pytest.approx(350.0)


def test_no_phantom_stall_chunk_0s_never_feed_the_residual_but_an_unpredicted_continuation_stall_does():
    """Every chunk 0 carries a whole prefill; were it a stall, one chunk 0 in five would put a prefill into every
    chunk's allowance and a request's second question would fall back for a stall that does not exist."""
    e = Estimators()
    for _ in range(20):
        e.add_chunk(0, chunk(40, wall=900.0, prompt_ms=480.0, predicted_ms=390.0, prompt_n=400))
        e.add_chunk(1, chunk(20, wall=205.0, prompt_ms=10.0, predicted_ms=190.0))
    assert e.resid95() == 0.0
    est = e.estimate(i=0, slot=0, question_steps=[], uncached=400)
    assert est.stall_ms == 0.0 and feasible(est.prefill_ms + est.overhead_ms + est.step_ms + 0.0, est)
    for _ in range(8):
        e.add_chunk(1, chunk(20, wall=705.0, prompt_ms=10.0, predicted_ms=690.0))     # +500 ms in its decode steps
    assert e.resid95() == pytest.approx(500.0, abs=10.0)                          # that nobody dispatched


def test_a_stall_s_known_predicted_is_not_learned_twice():
    e = Estimators()
    for _ in range(10):
        e.add_chunk(1, chunk(20, wall=705.0, prompt_ms=10.0, predicted_ms=190.0, known=500.0))
    assert e.resid95() == 0.0


def test_the_estimate_composes_the_terms():
    clock = Clock()
    e = Estimators(latency=LATENCY, clock=clock)
    e.dispatch_start(1, 200.0)
    est = e.estimate(i=0, slot=0, question_steps=[], uncached=100)
    assert est == Estimate(100 * 1.6, 10.7, DEFAULT_OVERHEAD_MS, 200.0, 200.0)
    later = e.estimate(i=2, slot=0, question_steps=[11.0, 12.0], uncached=100)
    assert later.prefill_ms == 0.0 and later.step_ms == 11.5 and later.stall_ms == 200.0


def test_the_stats_report_each_windows_p50_and_p95():
    e = Estimators()
    assert e.stats() == {k: {"p50": None, "p95": None} for k in
                         ("decode_ms_per_token", "prefill_ms_per_token", "chunk_overhead_ms", "stall_ms")}
    e.add_chunk(0, chunk(10, wall=390.0, prompt_ms=200.0, predicted_ms=90.0, prompt_n=100))
    e.add_chunk(1, chunk(10, wall=120.0, prompt_ms=10.0, predicted_ms=100.0))
    st = e.stats()
    assert st["decode_ms_per_token"] == {"p50": 10.0, "p95": 11.0} and st["prefill_ms_per_token"]["p50"] == 2.0
    assert st["chunk_overhead_ms"]["p95"] == 100.0 and st["stall_ms"]["p50"] is not None
