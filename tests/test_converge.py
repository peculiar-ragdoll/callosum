"""The convergence measurement's pure half: the stop rule, the fractional-Beta
Clopper-Pearson bound with the cluster design effect, the fixed sequence with count-only skips, the recorded monotone
fit, the cross-fitted rule, the flip statistics, the early-exit latency, the gain test (every early-exit point on the
frontier) and the rho map, on hand-made traces."""
import math

import pytest

import callo.converge as cvg
from callo.converge import (
    RHOS,
    Probe,
    QTrace,
    betainc,
    cell_stats,
    conv_lower,
    ee_point,
    fixed_sequence,
    flip_stats,
    gain,
    gains,
    icc,
    latency,
    monotone_fit,
    net_saving,
    outcome,
    p_value,
    pairing,
    point_over,
    rho_map,
    rule_for,
    sequence_order,
    stop_index,
    zero_flip_minimum,
)
from callo.think import latency_stats


def probs(arg: int, p: float, n: int = 3) -> tuple[float, ...]:
    rest = (1 - p) / (n - 1)
    return tuple(p if i == arg else rest for i in range(n))


def trace(uid, *, label=0, final=0, n_gen=300, steps=(), k=32, cluster=None, a=200.0, d=10.0):
    """`steps`: (argmax, pmax) per probe at k, 2k, ..."""
    ps = tuple(Probe(k * (i + 1), probs(arg, p)) for i, (arg, p) in enumerate(steps))
    return QTrace(uid, cluster or uid, label, n_gen, probs(final, 0.95), ps, a, d, a + d * n_gen + 30.0)


def test_the_stop_rule_needs_n_probes_in_a_row_on_one_argmax_each_at_y():
    ps = [Probe(32, probs(1, 0.95)), Probe(64, probs(0, 0.95)), Probe(96, probs(0, 0.85)), Probe(128, probs(0, 0.99))]
    assert stop_index(ps, 1, 0.9) == 0 and stop_index(ps, 2, 0.9) is None and stop_index(ps, 2, 0.8) == 2
    assert stop_index(ps, 3, 0.8) == 3 and stop_index(ps, 4, 0.6) is None and stop_index([], 1, 0.6) is None


def test_the_incomplete_beta_is_the_binomial_tail_for_integer_counts():
    assert betainc(0.3, 2.0, 1.0) == pytest.approx(0.09) and betainc(0.3, 1.0, 3.0) == pytest.approx(1 - 0.7 ** 3)
    for n, x, z in ((40, 37, 0.9), (100, 97, 0.95), (12, 3, 0.4)):
        tail = sum(math.comb(n, i) * z ** i * (1 - z) ** (n - i) for i in range(x, n + 1))
        assert p_value(x, n, z) == pytest.approx(tail, rel=1e-9)
    assert betainc(0.5, 2.5, 2.5) == pytest.approx(0.5)


def test_a_flip_free_cell_reaches_z_only_with_ln_alpha_over_ln_z_effective_stops():
    assert [zero_flip_minimum(z) for z in (0.9, 0.95, 0.99)] == [29, 59, 299]
    for z, need in ((0.9, 29), (0.95, 59), (0.99, 299)):
        assert conv_lower(need, need) >= z and p_value(need, need, z) <= 0.05
        assert conv_lower(need - 1, need - 1) < z and p_value(need - 1, need - 1, z) > 0.05
    assert conv_lower(0, 50) == 0.0 and p_value(0, 50, 0.9) == 1.0
    assert conv_lower(29.5, 29.5) > conv_lower(29, 29)                    # the fractional form


def test_icc_is_zero_without_variance_one_cluster_or_singletons_and_high_when_clustered():
    assert icc([[1, 1], [1, 1, 1]]) == 0.0 and icc([[1, 0, 1]]) == 0.0 and icc([[1], [0], [1]]) == 0.0
    assert icc([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]) == pytest.approx(1.0)
    assert icc([[1, 0], [1, 0], [0, 1], [1, 0]]) == 0.0                   # negative estimate, floored


def _flip_free(n_q, *, stops, cluster_size=1):
    qs = []
    for i in range(n_q):
        steps = [(0, 0.99)] * 4 if i < stops else [(1, 0.5), (2, 0.5), (1, 0.5), (2, 0.5)]
        qs.append(trace(f"q{i}", steps=steps, cluster=f"c{i // cluster_size}"))
    return qs


def test_the_fixed_sequence_runs_least_to_most_permissive_and_stops_at_the_first_failure():
    order = sequence_order()
    assert order[:5] == [(4, 0.99), (3, 0.99), (2, 0.99), (1, 0.99), (4, 0.95)] and order[-1] == (1, 0.6)
    assert len(order) == 24
    qs = _flip_free(100, stops=60)                                           # 60 flip-free stops of 100
    cells = {c: cell_stats(qs, c, rho=None, seed=0) for c in order}
    assert all(v["verdict"] == "pass" for v in fixed_sequence(cells, 0.95, len(qs)))     # 60 >= 59 at every cell
    assert all(v["verdict"] == "skip" for v in fixed_sequence(cells, 0.99, len(qs)))     # 60 < 299: skipped
    flip = [trace(f"f{i}", steps=[(0, 0.99)] * 4, final=1 if i % 4 == 0 else 0) for i in range(60)]
    cells = {c: cell_stats(flip, c, rho=None, seed=0) for c in order}
    seq = fixed_sequence(cells, 0.9, len(flip))
    assert seq[0]["verdict"] == "fail" and {v["verdict"] for v in seq[1:]} == {"untested"}


def test_skips_read_counts_only_raw_stops_share_and_stops_over_m_bar():
    """A skip never consults agreements; deff, which does, enters only the test."""
    few = _flip_free(400, stops=35)                                         # 35 stops, but < 10 % of 400
    cells = {c: cell_stats(few, c, rho=None, seed=0) for c in sequence_order()}
    assert fixed_sequence(cells, 0.9, len(few))[0]["verdict"] == "skip"
    lumpy = _flip_free(100, stops=60, cluster_size=2)                        # 60 stops in 30 clusters: 60 / 2 = 30
    cells = {c: cell_stats(lumpy, c, rho=None, seed=0) for c in sequence_order()}
    assert cells[(4, 0.99)]["m_bar"] == 2 and cells[(4, 0.99)]["deff"] == 1.0         # no variance: deff 1
    assert fixed_sequence(cells, 0.95, 100)[0]["verdict"] == "skip"                  # 30 < 59, whatever deff says
    assert fixed_sequence(cells, 0.9, 100)[0]["verdict"] == "pass"                   # 30 >= 29


def test_a_flip_heavy_clustered_cell_is_failed_not_skipped():
    qs = []
    for i in range(120):
        wrong = (i // 4) % 3 == 0                                            # whole clusters flip together
        qs.append(trace(f"q{i}", steps=[(0, 0.99)] * 4, final=1 if wrong else 0, cluster=f"c{i // 4}"))
    cells = {c: cell_stats(qs, c, rho=None, seed=0) for c in sequence_order()}
    c = cells[(4, 0.99)]
    assert c["icc"] > 0.9 and c["deff"] > 3 and c["conv"] == pytest.approx(80 / 120)
    v = fixed_sequence(cells, 0.9, len(qs))[0]
    assert v["verdict"] == "fail" and v["p"] > 0.05


def test_the_monotone_fit_is_the_greatest_monotone_minorant_and_never_above_conv():
    """Recorded, not required: non-decreasing in n and in y over the nested grid, at most each cell's conv."""
    qs = []
    for i in range(80):                       # a quarter turn confidently away from F after an unsure agreeing probe
        steps = [(0, 0.7), (1, 0.99), (1, 0.99), (1, 0.99)] if i % 4 == 0 else [(0, 0.99)] * 4
        qs.append(trace(f"q{i}", steps=steps))
    cells = {c: cell_stats(qs, c, rho=None, seed=0) for c in sequence_order()}
    mono = monotone_fit(cells)
    for (n, y), v in mono.items():
        if v is None:
            continue
        assert v <= cells[(n, y)]["conv"] + 1e-12
        for (n2, y2), w in mono.items():
            if n2 >= n and y2 >= y and w is not None:
                assert w >= v - 1e-12
    assert cells[(1, 0.6)]["conv"] == 1.0 and cells[(1, 0.99)]["conv"] == 0.75
    assert mono[(1, 0.6)] == 0.75                                            # smoothed down to the stricter cell's


def test_net_saving_charges_the_probes_before_the_answer_at_rho_decode_steps():
    """As latency() charges them, the stop's own read is the forced read every question pays,
    so only the probes read before it cost rho; a question that never stops read every probe before its end read."""
    q = trace("a", n_gen=300, steps=[(0, 0.99), (0, 0.99), (0, 0.99)])
    assert net_saving([q], (2, 0.9), 0.0) == 300 - 64 and net_saving([q], (2, 0.9), 4.0) == 300 - 64 - 1 * 4
    never = trace("b", n_gen=300, steps=[(1, 0.5)] * 3)
    assert net_saving([never], (1, 0.9), 2.0) == -6.0                       # 3 probes before, nothing saved
    c = cell_stats([q], (2, 0.9), rho=4.0, seed=0)
    assert c["probes"] == 1 and c["net_saving"] == 300 - 64 - 1 * 4
    assert cell_stats([never], (1, 0.9), rho=2.0, seed=0)["probes"] == 3


def test_the_ee_latency_charges_c_per_probe_before_the_stop_and_a_forced_read():
    q = trace("a", n_gen=300, steps=[(1, 0.5), (0, 0.99), (0, 0.99)], a=200.0, d=10.0)
    o = outcome(q, (2, 0.9))
    assert (o.stopped, o.k, o.before) == (True, 96, 2)
    assert latency(q, o, 30.0, 25.0) == 200 + 10 * 96 + 2 * 30 + 25
    o = outcome(q, (4, 0.9))
    assert not o.stopped and latency(q, o, 30.0, 25.0) == q.end_lat_ms + 3 * 30
    one = QTrace("s", "s", 0, 0, probs(0, 0.9), (), 0.0, 0.0, 250.0)          # one option: no trace, never stops
    assert latency(one, outcome(one, (1, 0.6)), 30.0, 25.0) == 250.0


def test_the_rule_is_the_max_saving_passed_cell_and_its_numbers_are_cross_fitted():
    qs = _flip_free(100, stops=70)
    cells = {c: cell_stats(qs, c, rho=1.0, seed=0) for c in sequence_order()}
    seq = fixed_sequence(cells, 0.9, len(qs))
    r = rule_for(qs, cells, seq, 0.9, 1.0, 0)
    assert r is not None and tuple(r.summary["cell"]) == (1, 0.99)          # earliest stop, fewest probes
    assert r.summary["conv"] == 1.0 and r.summary["stops"] == 70 and r.summary["acc"] == 1.0
    assert set(r.assigned) == {q.uid for q in qs}
    assert rule_for(qs, cells, [{"cell": [4, 0.99], "verdict": "fail", "p": 0.5}], 0.9, 1.0, 0) is None


def test_the_flip_statistics():
    qs = [trace("a", label=0, final=0, steps=[(1, 0.95), (0, 0.6), (0, 0.9)]),
          trace("b", label=1, final=1, steps=[(1, 0.5), (1, 0.97)]),
          trace("c", label=0, final=1, steps=[(0, 0.92)])]
    st = flip_stats(qs, 32)
    assert st["flip_rate"] == {"64": 0.5, "96": 0.0} and st["flip_n"] == {"64": 2, "96": 1}
    assert (st["confident"], st["changed_after_confident"], st["right_to_wrong"], st["wrong_to_right"]) == (3, 2, 1, 1)
    assert st["acc_at"]["32"] == pytest.approx(2 / 3) and st["acc_n"]["96"] == 1


def _qs(n=10):
    return [trace(f"q{i}") for i in range(n)]


def _pt(qs, kind, value, correct, lat):
    return point_over("m", qs, [(c, lat * (0.9 + 0.02 * i)) for i, c in enumerate(correct)], kind=kind, value=value,
                      seed=0, cap=1024 if kind == "deadline" else None)


def test_points_are_keyword_built_with_pp_s_five_latency_statistics():
    qs = _qs()
    p = _pt(qs, "deadline", 2000, [1.0] * 10, 1500.0).point
    lats = [1500.0 * (0.9 + 0.02 * i) for i in range(10)]
    assert (p.kind, p.value, p.cap, p.label, p.budget) == ("deadline", 2000, 1024, "m@2000ms", None)
    assert p.stats() == latency_stats(lats, [q.cluster for q in qs], seed=0)
    b = _pt(qs, "budget", 512, [1.0] * 10, 900.0).point
    assert (b.kind, b.value, b.budget, b.label) == ("budget", 512, 512, "m@512")     # value derived from the budget


def test_the_gain_test_on_front_within_noise_is_no_gain_a_speed_gain_and_an_accuracy_gain():
    qs = _qs()
    same = [1.0] * 8 + [0.0] * 2
    ee, slow = _pt(qs, "ee", 0.9, same, 500.0), _pt(qs, "budget", 1024, same, 2000.0)
    g = gain(ee, [slow], pairing([ee, slow], qs, 0))
    assert g["gains"] and g["p50"]["speed"] and g["p50"]["front"]
    near = _pt(qs, "budget", 512, same, 510.0)                               # overlapping latency, same accuracy
    g = gain(ee, [near], pairing([ee, near], qs, 0))
    assert g["p50"]["front"] and not g["gains"]
    acc_ee, fast = _pt(qs, "ee", 0.9, [1.0] * 10, 900.0), _pt(qs, "budget", 0, [1.0] * 2 + [0.0] * 8, 300.0)
    g = gain(acc_ee, [fast], pairing([acc_ee, fast], qs, 0))
    assert g["p50"]["accuracy"] and g["gains"]
    assert not gain(ee, [], pairing([ee], qs, 0))["gains"]                  # nothing to gain against


def test_the_gain_tests_frontier_holds_every_early_exit_point():
    """The frontier is built from the model's budget, deadline and early-exit points: an early-exit point that
    another one dominates is off the front, so it gains nothing, though it beats every budget point."""
    qs = _qs()
    same = [1.0] * 8 + [0.0] * 2
    slow = _pt(qs, "budget", 1024, same, 2000.0)
    a, b = _pt(qs, "ee", 0.9, same, 500.0), _pt(qs, "ee", 0.95, same, 900.0)
    paired = pairing([slow, a, b], qs, 0)
    assert gain(b, [slow], paired)["gains"]                                   # alone, .95 gains on speed
    assert not gain(b, [slow], paired, peers=[a])["gains"] and gain(a, [slow], paired, peers=[b])["gains"]
    _, verdicts = gains("m", [], {}, [], c_ms=1.0, read_ms=1.0, seed=0)
    assert verdicts == {}


def test_the_rho_map_holds_rho_3_and_finds_the_largest_probe_cost_at_which_early_exit_still_gains():
    assert RHOS == (0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)
    qs = _flip_free(100, stops=70)                                           # end latency 200 + 10 x 300 + 30
    cells = {c: cell_stats(qs, c, rho=1.0, seed=0) for c in sequence_order()}
    seqs = {0.9: fixed_sequence(cells, 0.9, len(qs))}
    bmax = point_over("m", qs, [(1.0, q.end_lat_ms) for q in qs], kind="budget", value=1024, seed=0)
    m = rho_map("m", qs, cells, seqs, [bmax], d_ms=10.0, read_ms=25.0, seed=0)
    assert set(m["by_rho"]) == {"0.5", "1", "2", "3", "4", "8", "16", "32"}
    assert m["rho_max"] is not None and m["by_rho"]["3"] == [0.9]
    assert m["rho_prefix"] is not None and m["rho_prefix"] <= m["rho_max"]
    assert all(m["by_rho"][f"{r:g}"] for r in RHOS if r <= m["rho_prefix"])
    unsure = [trace(f"u{i}", steps=[(1, 0.5), (2, 0.5)] * 2) for i in range(100)]          # no cell ever stops
    ucells = {c: cell_stats(unsure, c, rho=1.0, seed=0) for c in sequence_order()}
    none = rho_map("m", unsure, ucells, {0.9: fixed_sequence(ucells, 0.9, 100)}, [bmax], d_ms=10.0, read_ms=25.0, seed=0)
    assert none["rho_max"] is None and set(map(len, none["by_rho"].values())) == {0}
    assert none["rho_prefix"] is None
    r = rule_for(qs, cells, seqs[0.9], 0.9, 1.0, 0)
    e = ee_point("m", qs, r, c_ms=10.0, read_ms=25.0, seed=0)
    assert e.point.label == "m@ee.9" and e.point.kind == "ee" and e.point.acc == 1.0 and e.point.lat_p50[0] < 3230.0


@pytest.mark.parametrize(("gaining", "prefix", "top"), [(RHOS, 32.0, 32.0), ((0.5, 2.0, 3.0, 4.0), 0.5, 4.0),
                                                        ((2.0, 3.0), None, 3.0), ((), None, None)])
def test_the_rho_map_stores_the_prefix_of_grid_rhos_that_gain_from_the_cheapest_up(monkeypatch, gaining, prefix, top):
    """`rho_prefix` is the largest grid rho such that some z gains at every
    grid rho from the cheapest up to it; rho_max, the largest gaining rho, may sit above a gap (each rho re-selects its
    rules), so "pays while a probe costs <= rho_max" would overstate."""
    def gains(name, qs, rules, others, *, c_ms, **kw):
        return {}, {"0.9": {"gains": c_ms / 10.0 in gaining}}

    monkeypatch.setattr(cvg, "gains", gains)
    qs = _flip_free(100, stops=70)
    cells = {c: cell_stats(qs, c, rho=1.0, seed=0) for c in sequence_order()}
    m = rho_map("m", qs, cells, {0.9: fixed_sequence(cells, 0.9, len(qs))}, [], d_ms=10.0, read_ms=25.0, seed=0)
    assert (m["rho_prefix"], m["rho_max"]) == (prefix, top)
