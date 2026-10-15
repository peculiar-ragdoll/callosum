"""The chunked trace against the thinking fake in real time: continuations that evaluate one token,
the readout at the cut, every close, the fallback, seeds and timeouts per chunk, cancellation, the probe's live mode, the
estimators it feeds, and the live loop equal to calibrate's simulation under matching timings."""
import threading
import time

import pytest
from fake_upstream import FakeUpstream
from test_think_engine import REQ, engine_for

from callo.deadline import Estimate, Estimators, simulate
from callo.engine import ChunkedReadout
from callo.errors import Cancelled, ThinkTooLarge
from callo.think import think_seed

Q = REQ.questions[0]
SEED = (REQ.digest, Q.qid)
T1 = lambda close, k, cap: (1.0, False)     # noqa: E731 -- the readout temperature rule, as the service passes it


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=10_000).start()
    u.think_real_time, u.think_ms_per_token, u.think_prompt_ms_per_token = True, 2.0, 0.05
    yield u
    u.stop()


def ms():
    return 1000.0 * time.monotonic()


def gens(up):
    return [r.json for r in up.requests if r.path == "/completion" and not r.json.get("n_probs")]


def run(e, *, budget_ms, cap=1024, explicit=False, est=None, **kw):
    now = ms()
    return e.answer_chunked(REQ.state, Q, cap=cap, explicit_cap=explicit, seed=SEED, sampling=e.trace_sampling(),
                            temperature_of=kw.pop("temperature_of", T1), share_end_ms=now + budget_ms,
                            request_end_ms=now + budget_ms, reserve_ms=kw.pop("reserve_ms", 20.0),
                            estimators=est if est is not None else Estimators(), id_slot=0, **kw)


def test_a_deadline_trace_is_a_chain_of_continuations_each_evaluating_one_token(up):
    e = engine_for(up)
    r = run(e, budget_ms=700)
    assert isinstance(r, ChunkedReadout) and r.close == "deadline" and r.trace.close == "forced"
    sent = gens(up)
    assert len(sent) == len(r.chunks) >= 3 and r.trace.chunks == len(sent)
    sizes = [g["n_predict"] for g in sent[1:]]           # chunk 0 is sized on the 30 ms seed; then the measured d̂
    assert sizes == sorted(sizes, reverse=True) and sizes[0] > sent[0]["n_predict"]
    log = up.gen_log
    assert log[0]["prompt_n"] == r.trace.prompt_tokens and all(x["prompt_n"] == 1 for x in log[1:])   # no rollback
    assert len(sent[1]["prompt"]) == r.trace.prompt_tokens + len(r.chunks[0].tokens)                 # on + gen so far
    assert r.readout.read_prompt_n == 4 and r.readout.k == r.trace.n_gen == sum(len(c.tokens) for c in r.chunks)
    assert r.trace.gen_wall_ms == pytest.approx(sum(c.wall_ms for c in r.chunks))
    assert r.trace.cont_prompt_n == len(sent) - 1 and r.trace.gen_prompt_n == r.trace.prompt_tokens
    assert r.trace.cont_reprefill == 0
    assert [g["seed"] for g in sent] == [think_seed(*SEED)] + [think_seed(*SEED, str(i)) for i in range(1, len(sent))]
    assert len(up.checkpoints) == len(sent) - 1 and not up.restores


def test_a_natural_close_returns_at_once_and_reads_the_two_token_first_pass(up):
    up.think_len = 30
    e = engine_for(up)
    t0 = ms()
    r = run(e, budget_ms=10_000)
    assert (r.close, r.trace.n_gen, r.readout.read_prompt_n, r.trace.chunks) == ("natural", 31, 2, 1)
    assert ms() - t0 < 2000                                       # no padding: nothing waits out the deadline


def test_the_cap_closes_with_budget_and_eos_with_eos(up):
    e = engine_for(up)
    r = run(e, budget_ms=10_000, cap=40, explicit=True)
    assert (r.close, r.trace.n_gen, r.readout.read_prompt_n) == ("budget", 40, 4)
    up.think_len, up.think_eos = 20, True
    r = run(e, budget_ms=10_000)
    assert r.close == "eos" and r.trace.close == "eos"


def test_a_share_too_short_for_one_token_is_the_fallback_and_sends_nothing(up):
    e = engine_for(up)
    assert run(e, budget_ms=10, reserve_ms=20.0) is None and not gens(up)


def test_an_explicit_budget_keeps_the_413_and_the_deadline_only_cap_is_clipped(up):
    e = engine_for(up)
    on = e.plan_deliberate(REQ.state, Q)
    e.n_ctx = on + 16 + 50
    with pytest.raises(ThinkTooLarge):
        run(e, budget_ms=10_000, cap=1024, explicit=True)
    seen = []
    r = run(e, budget_ms=10_000, cap=1024, temperature_of=lambda close, k, cap: (seen.append(cap), (1.0, False))[1])
    assert (r.close, r.trace.n_gen, r.cap) == ("budget", 50, 50) and seen == [50]   # the rule sees the clipped cap
    e.n_ctx = on + 16
    assert run(e, budget_ms=10_000, cap=1024) is None             # nothing left to think in: the fallback


def test_each_chunk_s_timeout_is_capped_5_s_past_the_request_deadline(up):
    e = engine_for(up)
    seen = []
    real = e.scorer.generate

    def spy(*a, **kw):
        seen.append((kw["n_predict"], kw["timeout"]))
        return real(*a, **kw)

    e.scorer.generate = spy
    e.s1_timeout = 900.0
    run(e, budget_ms=600)
    n0, t0 = seen[0]
    assert 5.4 < t0 <= 5.6 and all(t <= 5.6 for _, t in seen)    # min(900 + 0.1 n, 0.6 s left + 5 s)
    e.s1_timeout = 1.0
    seen.clear()
    run(e, budget_ms=600)
    assert seen[0][1] == pytest.approx(1.0 + 0.1 * seen[0][0])


def test_cancel_before_a_chunk_and_during_one(up):
    e = engine_for(up)
    cancel = threading.Event()
    up.think_gen_hook = lambda body: cancel.set()                # set during chunk 0: chunk 1 never starts
    with pytest.raises(Cancelled):
        run(e, budget_ms=2000, cancel=cancel)
    assert len(gens(up)) == 1
    up.think_gen_hook = None
    cancel = threading.Event()
    up.think_ms_per_token = 50.0
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        run(e, budget_ms=20_000, cancel=cancel)
    assert time.monotonic() - t0 < 2.0


def test_the_probe_s_live_mode_aligns_chunks_to_k_and_reads_a_forced_probe_at_each_boundary(up):
    up.think_len = 100
    e = engine_for(up)
    r = e.answer_chunked(REQ.state, Q, cap=1024, explicit_cap=False, seed=SEED, sampling=e.trace_sampling(),
                         temperature_of=T1, id_slot=0, probe_every=32)
    assert [g["n_predict"] for g in gens(up)] == [32, 32, 32, 32]
    assert r.close == "natural" and r.trace.n_gen == 101 and [p.k for p in r.probes] == [32, 64, 96]
    assert all(p.read_prompt_n == 4 for p in r.probes)
    assert [x["prompt_n"] for x in up.gen_log[1:]] == [1, 1, 1] and len(up.restores) == 3   # each resumes a checkpoint


def test_a_run_feeds_the_estimators_and_a_budget_0_first_pass_is_recorded_while_it_runs(up):
    e = engine_for(up)
    est = Estimators()
    run(e, budget_ms=600, est=est)
    st = est.stats()
    assert st["decode_ms_per_token"]["p50"] is not None and st["chunk_overhead_ms"]["p50"] is not None

    class Spy(Estimators):
        def __init__(self):
            super().__init__()
            self.calls = []

        def dispatch_start(self, slot, est_ms):
            self.calls.append(("start", slot, est_ms))
            return super().dispatch_start(slot, est_ms)

        def dispatch_end(self, token):
            self.calls.append(("end", token))
            super().dispatch_end(token)

    e.estimators = spy = Spy()
    e.answer(REQ.state, Q, id_slot=1)
    (start, slot, est_ms), end = spy.calls
    assert (start, slot, end[0]) == ("start", 1, "end") and est_ms > 0 and spy.known(0) == 0.0


def test_the_live_loop_equals_the_simulation_under_matching_timings(up):
    """One planner serves both. A manual clock advances as calibrate's simulation models a chunk (chunk 0: a + n·d,
    later ones: ô + n·d), and the estimates are the simulation's (P̂ = a, d̂ = d, ô = 5, G = 0)."""
    a, d = 300.0, 4.0

    class Fixed(Estimators):
        def estimate(self, *, i, slot, question_steps, uncached):
            return Estimate(a, d, 5.0)

    for think_len, D in ((10_000, 1500.0), (10_000, 900.0), (60, 2000.0)):
        up.think_len = think_len
        e = engine_for(up)
        clock = [0.0]
        on = e.plan_deliberate(REQ.state, Q)

        def timed(*args, _real=e.scorer.generate, _clock=clock, _on=on, **kw):
            g = _real(*args, **kw)
            _clock[0] += (a if len(args[0]) == _on else 5.0) + len(g.tokens) * d
            return g

        e.scorer.generate = timed
        r = e.answer_chunked(REQ.state, Q, cap=1024, explicit_cap=False, seed=SEED, sampling=e.trace_sampling(),
                             temperature_of=T1, share_end_ms=D, request_end_ms=D, reserve_ms=50.0, estimators=Fixed(),
                             id_slot=0, clock=lambda c=clock: c[0])
        s = simulate(a_ms=a, d_ms=d, n_gen=think_len + 1, close="natural", deadline_ms=D, reserve_ms=50.0, cap=1024)
        assert (r.close, r.trace.n_gen, r.trace.chunks) == (s.close, s.k, s.chunks)
        assert clock[0] == pytest.approx(s.elapsed_ms)


def test_the_fake_and_the_estimators_split_a_chunk_as_llama_server_does(up):
    """llama-server's predicted_ms leaves out the first sampled token (its logits come from the prompt phase): the
    fake reports it that way, and the estimators read chunk 0's step as predicted_ms / (n - 1)."""
    up.think_ms_per_token = 20.0
    e = engine_for(up)
    est = Estimators()
    r = run(e, budget_ms=10_000, cap=4, explicit=True, est=est)
    c0 = r.chunks[0]
    n = len(c0.tokens)
    assert n == 4 and c0.predicted_ms == pytest.approx(20.0 * (n - 1), rel=0.2)   # 60 ms (48-72), not 80: n - 1 steps
    assert c0.prompt_ms >= 5.0                                     # the prefill plus the first token's step
    assert est.stats()["decode_ms_per_token"]["p50"] == pytest.approx(20.0, rel=0.3)


def test_a_stall_planted_in_a_generation_s_last_step_is_not_charged_to_the_next(up):
    import json
    import urllib.request

    up.think_ms_per_token = 20.0

    def post(n):
        req = urllib.request.Request(up.url + "/completion", method="POST",
                                     data=json.dumps({"prompt": [1, 2, 3], "n_predict": n, "id_slot": 0}).encode(),
                                     headers={"Content-Type": "application/json"})
        t = time.monotonic()
        urllib.request.urlopen(req, timeout=10).read()
        return time.monotonic() - t

    def plant():
        with up.lock:
            up._stalls[0] = 400.0

    threading.Timer(0.05, plant).start()                          # during the 3rd (last) step of a 3-token generation
    post(3)
    assert post(1) < 0.25                                         # the leftover 400 ms is gone


def test_the_probe_reads_only_on_k_boundaries_even_with_deadline_sized_chunks(up):
    e = engine_for(up)
    r = run(e, budget_ms=500, probe_every=64)
    assert r.trace.chunks >= 2 and r.probes and all(p.k % 64 == 0 for p in r.probes)
