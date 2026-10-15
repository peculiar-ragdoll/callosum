"""Pick preferences: the paired Δacc over one fixed row set (the pick's noise test, unchanged) and the reported
Δscore: 100 × Δacc at R = 0, the per-question scores on mean, the joint bootstrap on p50 / p95, and the identities that
tie them together."""
import random

import numpy as np
import pytest

from callo.think import (
    Paired,
    Point,
    _bootstrap_weights,
    joint_dscore,
    nearest_rank,
    paired_bootstrap,
    point_latencies,
    score_draws,
)

N = 120
RNG = random.Random(5)
CLUSTERS = tuple(f"g{i // 3}" for i in range(N))
STRATA = tuple("s0" if i % 6 < 3 else "s1" for i in range(N))
CA = [float(RNG.random() < 0.80) for _ in range(N)]
CB = [c if RNG.random() < 0.9 else 1.0 - c for c in CA]
LA = [RNG.uniform(200, 400) for _ in range(N)]
LB = [x + RNG.uniform(0, 3000) * (RNG.random() < 0.6) for x in LA]


def pt(kind: str, value) -> Point:
    s = {"lat_mean_ms": [1.0] * 3, "lat_p50_ms": [1.0] * 3, "lat_p95_ms": [1.0] * 3, "lat_p99_ms": [1.0] * 3,
         "lat_max_ms": 1.0}
    return Point(name="m", kind=kind, value=value, budget=value if kind == "budget" else None, acc=0.5, n=N,
                 cap=512 if kind == "deadline" else None, **point_latencies(s))


A, B = pt("budget", 0), pt("deadline", 2000)
ROWS = {("budget", 0): (CA, LA), ("deadline", 2000): (CB, LB)}


def paired(strata=None, clusters=CLUSTERS) -> Paired:
    return Paired(lambda p: ROWS[(p.kind, p.value)], clusters, 0, strata)


@pytest.mark.parametrize("strata", [None, STRATA])
def test_dacc_is_the_paired_bootstrap_over_the_getters_rows(strata):
    assert paired(strata).dacc(A, B) == paired_bootstrap(CA, CB, CLUSTERS, seed=0, strata=strata)


@pytest.mark.parametrize("stat", ["mean", "p50", "p95"])
def test_at_price_0_dscore_is_100_times_dacc_and_the_formulas_agree(stat):
    p = paired(STRATA)
    d = p.dacc(A, B)
    assert p.dscore(A, B, stat=stat, price=0.0) == (100 * d[0], 100 * d[1], 100 * d[2])     # the delegation itself
    assert joint_dscore(CA, LA, CB, LB, CLUSTERS, stat=stat, price=0.0, seed=0, strata=STRATA) == \
        pytest.approx(tuple(100 * x for x in d), rel=1e-9, abs=1e-9)                          # at R = 0


@pytest.mark.parametrize("price", [1.0, 5.0])
def test_on_mean_the_joint_formula_matches_the_per_question_scores(price):
    sa = [100 * c - price * x / 1000 for c, x in zip(CA, LA, strict=True)]
    sb = [100 * c - price * x / 1000 for c, x in zip(CB, LB, strict=True)]
    want = paired_bootstrap(sa, sb, CLUSTERS, seed=0, strata=STRATA)
    assert paired(STRATA).dscore(A, B, stat="mean", price=price) == want
    assert joint_dscore(CA, LA, CB, LB, CLUSTERS, stat="mean", price=price, seed=0, strata=STRATA) == \
        pytest.approx(want, rel=1e-9, abs=1e-9)                                               # the same estimator


def test_identical_latency_rows_give_a_zero_latency_term_on_every_resample():
    for stat in ("p50", "p95", "mean"):
        _, priced = score_draws(CA, LA, CB, LA, CLUSTERS, stat=stat, price=7.0, seed=0, strata=STRATA)
        _, free = score_draws(CA, LA, CB, LA, CLUSTERS, stat=stat, price=0.0, seed=0, strata=STRATA)
        assert np.array_equal(priced, free)


@pytest.mark.parametrize("stat, q", [("p50", 0.5), ("p95", 0.95)])
def test_the_quantile_point_estimate_is_unweighted_over_the_fixed_rows(stat, q):
    r = nearest_rank(N, q) - 1
    want = 100 * (sum(CB) - sum(CA)) / N - 3.0 * (sorted(LB)[r] - sorted(LA)[r]) / 1000
    point, lo, hi = paired(STRATA).dscore(A, B, stat=stat, price=3.0)
    assert point == pytest.approx(want) and lo <= hi
    _, draws = score_draws(CA, LA, CB, LB, CLUSTERS, stat=stat, price=3.0, seed=0, strata=STRATA)
    assert (lo, hi) == tuple(float(x) for x in np.percentile(draws, [2.5, 97.5])) and len(draws) == 2000


def test_no_paired_evidence_is_zero_within_noise():
    empty = Paired(lambda p: ([], []), (), 0)
    assert empty.dacc(A, B) == (0.0, 0.0, 0.0)
    assert empty.dscore(A, B, stat="p50", price=4.0) == (0.0, 0.0, 0.0)


def test_the_draws_are_every_daccs_resamples():
    """One seed, one set of clusters and strata: the joint draws at R = 0 are 100 × paired_bootstrap's own resamples
    (the same percentiles), so a Δscore and a Δacc never resample different questions."""
    _, draws = score_draws(CA, LA, CB, LB, CLUSTERS, stat="p95", price=0.0, seed=3, strata=STRATA)
    _, lo, hi = paired_bootstrap(CA, CB, CLUSTERS, seed=3, strata=STRATA)
    assert tuple(np.percentile(draws, [2.5, 97.5])) == pytest.approx((100 * lo, 100 * hi), rel=1e-9, abs=1e-9)


def test_score_draws_refuses_mismatched_empty_and_unknown_stat():
    for bad in (dict(lb=LB[:100]), dict(clusters=CLUSTERS[:100])):
        kw = dict(ca=CA, la=LA, cb=CB, lb=LB, clusters=CLUSTERS) | bad
        with pytest.raises(ValueError, match="one length"):
            score_draws(kw["ca"], kw["la"], kw["cb"], kw["lb"], kw["clusters"], stat="p50", price=3.0, seed=0)
    with pytest.raises(ValueError, match="one length"):
        joint_dscore([], [], [], [], (), stat="mean", price=3.0, seed=0)
    for stat in ("p99", "max"):
        with pytest.raises(ValueError, match="stat must be"):
            score_draws(CA, LA, CB, LB, CLUSTERS, stat=stat, price=3.0, seed=0)


@pytest.mark.parametrize("stat, q", [("p50", 0.5), ("p95", 0.95)])
def test_the_joint_draws_are_the_expanded_resamples_computed_directly(stat, q):
    _, draws = score_draws(CA, LA, CB, LB, CLUSTERS, stat=stat, price=3.0, seed=7, strata=STRATA, resamples=50)
    idx, w = _bootstrap_weights(CLUSTERS, STRATA, 7, 50)
    for r in (0, 1, 17, 49):
        rep = w[r, idx].astype(int)
        ca, cb = np.repeat(CA, rep), np.repeat(CB, rep)
        la, lb = np.sort(np.repeat(LA, rep)), np.sort(np.repeat(LB, rep))
        k = nearest_rank(len(ca), q) - 1
        want = 100 * (cb.mean() - ca.mean()) - 3.0 * (lb[k] - la[k]) / 1000
        assert draws[r] == pytest.approx(want, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("price", [0.0, 2.0])
def test_dscore_refuses_an_unknown_statistic_even_at_price_0_or_without_rows(price):
    with pytest.raises(ValueError, match="latency axis"):
        paired(STRATA).dscore(A, B, stat="p99", price=price)
    with pytest.raises(ValueError, match="latency axis"):
        Paired(lambda p: ([], []), (), 0).dscore(A, B, stat="max", price=price)
