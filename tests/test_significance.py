"""callo.significance: the omnibus cyclic-shift permutation test, Holm across n, display intervals."""
import itertools
import math
import tracemalloc

import numpy as np
import pytest

import callo.significance as sig
from callo.significance import bootstrap_intervals, decide, holm, permutation_test


def _null(rng, q, n):
    """Per-question priors with no position effect: softmax of iid N(0, 0.3²)."""
    z = rng.normal(0, 0.3, size=(q, n))
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def _upper(trials, p, conf=0.99):
    """The smallest k with P(Binomial(trials, p) <= k) >= conf: a one-sided 99% bound."""
    acc = 0.0
    for k in range(trials + 1):
        acc += math.comb(trials, k) * p**k * (1 - p) ** (trials - k)
        if acc >= conf:
            return k
    return trials


def test_family_wise_error_rate_stays_under_alpha_on_null_data():
    rng = np.random.default_rng(12345)
    trials, any_kept = 300, 0
    for t in range(trials):
        fits = {n: (_null(rng, 48, n), [f"q{i}" for i in range(48)]) for n in range(2, 9)}
        any_kept += any(d.kept for d in decide(fits, draws=300, seed=t).values())
    assert any_kept <= _upper(trials, 0.05), any_kept


def test_holm_on_p_values_where_holm_and_bonferroni_differ(monkeypatch):
    p = {2: 0.01, 3: 0.02, 4: 0.04}
    assert holm(p) == pytest.approx({2: 0.03, 3: 0.04, 4: 0.04})            # Bonferroni: 0.03, 0.06, 0.12
    assert holm({5: 0.5, 6: 0.001}) == pytest.approx({6: 0.002, 5: 0.5})
    assert holm({2: 0.01, 3: 0.03, 4: 0.04}) == pytest.approx({2: 0.03, 3: 0.06, 4: 0.06})   # step-down: 4 keeps 3's 0.06
    assert holm({2: 0.6, 3: 0.7}) == {2: 1.0, 3: 1.0}                                        # 1.2 capped at 1
    monkeypatch.setattr(sig, "permutation_test", lambda priors, clusters, **kw: (0.1, p[priors.shape[1]]))
    fits = {n: (np.full((30, n), 1.0 / n), [f"q{i}" for i in range(30)]) for n in p}
    d = decide(fits)
    assert [d[n].kept for n in (2, 3, 4)] == [True, True, True]              # Bonferroni would keep only n = 2
    assert d[4].p_holm == pytest.approx(0.04) and d[4].reason is None


def test_power_a_planted_bias_of_0_08_at_n4_is_kept_inside_a_seven_n_family():
    """n = 4 carries the bias; n = 2, 3, 5..8 are null, so Holm's family has 7 members as on the built-in set."""
    rng = np.random.default_rng(7)
    base = np.log([0.33, 0.25, 0.25, 0.17])                                   # max |p - 1/4| = 0.08
    kept = 0
    for t in range(100):
        z = base + rng.normal(0, 0.3, size=(48, 4))
        fits = {n: (_null(rng, 48, n), [f"q{i}" for i in range(48)]) for n in (2, 3, 5, 6, 7, 8)}
        fits[4] = (np.exp(z) / np.exp(z).sum(axis=1, keepdims=True), [f"q{i}" for i in range(48)])
        kept += decide(fits, draws=300, seed=t)[4].kept
    assert kept >= 95


def test_the_monte_carlo_p_matches_exact_enumeration():
    """3 single-question clusters at n = 3: all 27 shift combinations enumerated give the exact null probability."""
    priors = np.array([[0.6, 0.3, 0.1], [0.5, 0.2, 0.3], [0.2, 0.5, 0.3]])
    observed = sig.bias_stat(priors.mean(axis=0))
    hits = sum(sig.bias_stat(np.mean([np.roll(priors[i], -s) for i, s in enumerate(shifts)], axis=0))
               >= observed - 1e-12 for shifts in itertools.product(range(3), repeat=3))
    _, p = permutation_test(priors, ["a", "b", "c"], draws=20000, seed=0)
    assert abs(p - hits / 27) <= 0.01


def test_the_test_and_the_bootstrap_are_deterministic_by_seed_and_exact_on_uniform_priors():
    rng = np.random.default_rng(3)
    p = _null(rng, 48, 4)
    clusters = [f"c{i // 2}" for i in range(48)]                              # pairs share a cluster
    assert permutation_test(p, clusters, draws=500, seed=1) == permutation_test(p, clusters, draws=500, seed=1)
    assert bootstrap_intervals(p, clusters, seed=1) == bootstrap_intervals(p, clusters, seed=1)
    assert bootstrap_intervals(p, clusters, seed=1) != bootstrap_intervals(p, clusters, seed=2)
    flat = np.full((30, 3), 1.0 / 3)
    stat, pv = permutation_test(flat, [f"q{i}" for i in range(30)])
    assert stat == pytest.approx(0.0, abs=1e-15) and pv == 1.0                # every draw ties: never significant
    planted = np.tile([0.5, 0.3, 0.2], (30, 1))
    assert permutation_test(planted, [f"q{i}" for i in range(30)])[1] == pytest.approx(1 / 2001)   # the minimum p


def test_clusters_move_together_under_the_null_draw():
    """One cluster of 40 identical rows can only be shifted as a whole: the draw then reproduces the observed
    statistic in every shift that is a rotation of it, so a single cluster is never significant at n = 4."""
    planted = np.tile([0.5, 0.3, 0.15, 0.05], (40, 1))
    assert permutation_test(planted, ["one"] * 40, draws=400, seed=0)[1] == 1.0


def test_bootstrap_intervals_cover_the_mean_per_position():
    rng = np.random.default_rng(5)
    p = _null(rng, 60, 5)
    iv = bootstrap_intervals(p, [f"q{i}" for i in range(60)])
    mean = p.mean(axis=0)
    assert len(iv) == 5 and all(lo <= m <= hi for (lo, hi), m in zip(iv, mean, strict=True))


def test_the_bootstrap_resamples_clusters_not_rows():
    """30 pairs of identical rows: resampling the 30 clusters must give wider intervals than treating the 60 rows as
    independent (by about sqrt(2))."""
    rng = np.random.default_rng(6)
    half = _null(rng, 30, 4)
    p = np.repeat(half, 2, axis=0)
    paired = bootstrap_intervals(p, [f"c{i // 2}" for i in range(60)])
    rows = bootstrap_intervals(p, [f"r{i}" for i in range(60)])
    assert all((hi - lo) > 1.2 * (h2 - l2) for (lo, hi), (l2, h2) in zip(paired, rows, strict=True))


def test_decide_reasons_and_what_is_tested():
    rng = np.random.default_rng(11)
    planted = np.exp(np.log([0.5, 0.3, 0.2]) + rng.normal(0, 0.1, size=(30, 3)))
    planted /= planted.sum(axis=1, keepdims=True)
    fits = {3: (planted, [f"q{i}" for i in range(30)]),
            5: (_null(rng, 10, 5), [f"q{i}" for i in range(10)]),
            7: (np.full((40, 7), 1 / 7), [f"q{i}" for i in range(40)])}
    d = decide(fits, min_questions=24)
    assert d[3].kept and d[3].reason is None and d[3].p_holm <= 0.05 and d[3].bias > 0.1
    assert not d[5].kept and d[5].reason == "fewer than 24 questions read under every rotation (10)"
    assert d[5].p is None and d[5].p_holm is None                             # not tested, not in Holm's family
    assert not d[7].kept and d[7].reason == "not significant (p_holm 1)"


def test_a_large_set_is_processed_in_bounded_chunks(monkeypatch):
    """2,000 questions at n = 26 must not materialise draws x questions x n at once. With a small chunk
    the traced peak stays far below the ~250 MB an unchunked gather of 300 draws needs."""
    monkeypatch.setattr(sig, "CHUNK_ELEMENTS", 400_000)
    rng = np.random.default_rng(0)
    p = _null(rng, 2000, 26)
    clusters = [f"q{i}" for i in range(2000)]
    tracemalloc.start()
    try:
        _, pv = permutation_test(p, clusters, draws=300)
        iv = bootstrap_intervals(p, clusters, resamples=300)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert 0 < pv <= 1 and len(iv) == 26 and peak < 40e6, peak
