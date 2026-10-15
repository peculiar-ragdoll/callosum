"""Early exit in the chunk loop against the thinking fake: the stop at the first probe the rule accepts (the live
probe mode's own probes say where), the stop probe as the answer re-tempered to the rule's T, a probe on the deadline
cut doubling as the answer's read, a cap below or off K, a client that leaves during a probe, and the probe windows."""
import math
import threading

import pytest
from fake_upstream import FakeUpstream
from test_think_engine import REQ, _gen_calls, _reads, engine_for

from callo.converge import Probe, stop_index
from callo.deadline import ALPHA, ConvergeRun, Estimators
from callo.engine import NEG_INF, Readout, retemper, wide_bounded_readout
from callo.errors import Cancelled

Q = REQ.questions[0]
SEED = (REQ.digest, Q.qid)


def T(close, k, cap):
    return (2.0, True) if close == "converged" else (1.0, False)


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=10_000).start()
    yield u
    u.stop()


def chunked(e, *, cap=1024, explicit=False, **kw):
    return e.answer_chunked(REQ.state, Q, cap=cap, explicit_cap=explicit, seed=SEED, sampling=e.trace_sampling(),
                            temperature_of=kw.pop("temperature_of", T), id_slot=0, **kw)


def test_retemper_equals_a_read_at_that_temperature_bit_for_bit(up):
    e = engine_for(up)
    a = e.answer_deliberate(REQ.state, Q, [64], seed=1, sampling=e.trace_sampling(), temperature=1.0).readouts[64]
    b = e.answer_deliberate(REQ.state, Q, [64], seed=1, sampling=e.trace_sampling(), temperature=2.3).readouts[64]
    assert retemper(a.readout, 2.3).probs == b.readout.probs and retemper(a.readout, 1.0).probs == a.readout.probs


def test_retemper_equals_a_wide_readout_at_that_temperature_bit_for_bit():
    """The wide (two-letter) path: floored direct values, split credit added, a clamp at 0, a miss."""
    lp = math.log
    labels = ("AA", "AB", "BA", "BB")
    direct = {"AA": lp(0.5), "AB": lp(0.2), "BA": NEG_INF, "BB": NEG_INF}
    credit = {"AA": lp(0.6), "AB": NEG_INF, "BA": lp(0.0002), "BB": NEG_INF}
    dump = {0: lp(0.5), 1: lp(0.2), 2: lp(0.001)}
    p1, lms, miss = wide_bounded_readout(direct, credit, dump, labels, 1.0)
    assert lms[0] == 0.0 and miss == 1                                       # the clamp and the miss are exercised
    r = Readout(p1, lms, 1.0, "word", 1, miss, 10, wide=True)
    for T in (0.4, 1.0, 1.7, 3.1):
        assert retemper(r, T).probs == wide_bounded_readout(direct, credit, dump, labels, T)[0]


def test_early_exit_stops_at_the_first_probe_the_rule_accepts(up):
    e = engine_for(up)
    every = [Probe(p.k, p.readout.probs) for p in chunked(e, probe_every=32).probes]   # the live mode: no stop
    j = stop_index(every, 1, 0.6)
    assert j is not None
    up.requests.clear()
    r = chunked(e, converge=ConvergeRun(0.9, 32, 1, 0.6, 0.0))
    assert r.close == "converged" and r.trace.n_gen == r.readout.k == every[j].k == 32 * (j + 1)
    assert [p.k for p in r.probes] == [p.k for p in every[: j + 1]] and r.trace.close == "forced"
    assert r.readout.readout.probs == retemper(r.probes[-1].readout, 2.0).probs
    assert (r.temperature, r.calibrated, r.reused) == (2.0, True, True)
    assert [g["n_predict"] for g in _gen_calls(up)] == [32] * (j + 1)        # no chunk after the stop ...
    assert len(_reads(up)) == sum(p.readout.passes for p in r.probes)        # ... and no read beyond the probes


def test_a_probe_on_the_deadline_cut_is_the_answer_s_read(up):
    e = engine_for(up)
    now = [0.0]
    up.think_gen_hook = lambda body: now.__setitem__(0, 1e6 - 1.0)            # chunk 0 uses up the share
    r = chunked(e, converge=ConvergeRun(0.95, 32, 4, 0.99, 0.0), share_end_ms=1e6, request_end_ms=None,
                reserve_ms=0.0, estimators=Estimators(), clock=lambda: now[0])
    assert r.close == "deadline" and r.trace.n_gen == r.readout.k == 32 and len(r.probes) == 1
    assert len(_reads(up)) == r.probes[0].readout.passes                     # no second read at the cut
    assert r.readout.readout.probs == retemper(r.probes[0].readout, 1.0).probs and r.reused


@pytest.mark.parametrize("cap, ks", [(20, []), (80, [32, 64])])
def test_a_cap_below_or_off_k_closes_budget_with_the_probes_it_passed(up, cap, ks):
    e = engine_for(up)
    r = chunked(e, cap=cap, explicit=True, converge=ConvergeRun(0.95, 32, 4, 0.99, 0.0))
    assert r.close == "budget" and r.trace.n_gen == r.readout.k == cap and [p.k for p in r.probes] == ks
    assert len(_reads(up)) == sum(p.readout.passes for p in r.probes) + r.readout.readout.passes   # one final read
    assert not r.reused


def test_a_deadline_cut_off_a_probe_boundary_is_read_and_not_reused(up):
    e = engine_for(up)
    now = [0.0]
    up.think_gen_hook = lambda body: now.__setitem__(0, 1.0)                 # chunk 0 uses up the share
    # no estimators: d̂ = 1e-3 ms, so a share of 0.02 / ALPHA ms sizes chunk 0 at 20 tokens, off the K = 32 boundary
    r = chunked(e, converge=ConvergeRun(0.95, 32, 4, 0.99, 0.0), share_end_ms=0.0201 / ALPHA, request_end_ms=None,
                reserve_ms=0.0, clock=lambda: now[0])
    assert r.close == "deadline" and r.trace.n_gen == r.readout.k == 20 and r.probes == () and not r.reused
    assert len(_reads(up)) == r.readout.readout.passes                       # the read after the close


def test_a_client_that_leaves_during_a_probe_gets_nothing_from_the_partial_trace(up):
    e = engine_for(up)
    cancel = threading.Event()
    up.readout_top = lambda prompt, top: cancel.set()                        # set while the first probe reads
    with pytest.raises(Cancelled):
        chunked(e, converge=ConvergeRun(0.95, 32, 4, 0.99, 0.0), cancel=cancel)
    assert len(_gen_calls(up)) == 1                                          # no chunk after the probe


def test_early_exit_feeds_the_probe_windows_and_the_live_mode_does_not(up):
    e = engine_for(up)
    est = Estimators()
    chunked(e, probe_every=32, estimators=est)
    assert est.converge_stats()["after_probe"]["n"] == 0 and est.converge_stats()["probe_read_ms"]["p50"] is None
    r = chunked(e, converge=ConvergeRun(0.9, 32, 1, 0.6, 0.0), estimators=est)
    s = est.converge_stats()
    assert s["after_probe"]["n"] == len(r.probes) - 1 and s["after_probe"]["restored"] == s["after_probe"]["n"]
    assert s["probe_read_ms"]["p50"] > 0


def test_probe_every_and_converge_together_are_a_programming_error(up):
    with pytest.raises(ValueError, match="not both"):
        chunked(engine_for(up), probe_every=32, converge=ConvergeRun(0.9, 32, 1, 0.6, 0.0))


def test_an_early_exit_trace_is_the_live_mode_trace_cut_at_the_stop(up):
    """The same chunk layout and seeds: a stopped trace is the live mode's prefix (E2 and E3 pair on it). A single
    generation is NOT: chunk i >= 1 is reseeded with think_seed(digest, qid, i), so the two part at token K."""
    e = engine_for(up)
    live = chunked(e, probe_every=32)
    r = chunked(e, converge=ConvergeRun(0.9, 32, 1, 0.6, 0.0))
    assert r.close == "converged" and r.trace.gen_tokens == live.trace.gen_tokens[: r.trace.n_gen]
    assert [p.readout.probs for p in r.probes] == [p.readout.probs for p in live.probes[: len(r.probes)]]


def test_a_restored_chunk_after_a_probe_is_counted_by_after_probe_and_never_as_a_reprefill(up):
    # prompt_n 2 is a restore (deadline.RESTORE_PROMPT_N) after a probe and a re-prefill (> 1) anywhere else: the
    # chunks after a probe are counted once, in after_probe, and cont_reprefill keeps the others
    e = engine_for(up)
    up.think_gen_prompt_n = 2
    est = Estimators()
    r = chunked(e, cap=128, explicit=True, converge=ConvergeRun(0.95, 32, 4, 0.99, 0.0), share_end_ms=1e9,
                request_end_ms=None, reserve_ms=0.0, estimators=est, clock=lambda: 0.0)
    assert r.close == "budget" and r.trace.chunks == 4 and r.trace.cont_prompt_n == 3 * 2
    assert est.converge_stats()["after_probe"] == {"n": 3, "restored": 3, "share": 1.0}
    assert r.trace.cont_reprefill == 0
    live = chunked(e, cap=128, explicit=True, probe_every=32, share_end_ms=1e9, request_end_ms=None, reserve_ms=0.0,
                   estimators=Estimators(), clock=lambda: 0.0)
    assert live.trace.chunks == 4 and live.trace.cont_reprefill == 3        # no after_probe count in the live mode
