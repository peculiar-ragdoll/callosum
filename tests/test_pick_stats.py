"""Pick preferences: the five latency statistics,
their intervals from the one set of resamples, the tail counts, and the recomputation of an absent statistic from a
profile's rows."""
import random

import numpy as np
import pytest

from callo.think import (
    QUANTILES,
    STAT_FIELDS,
    _bootstrap_weights,
    bootstrap_mean,
    bootstrap_percentiles,
    latency_stats,
    nearest_rank,
    paired_bootstrap,
    point_stats,
    row_latencies,
    tail_counts,
)


def test_nearest_rank_and_the_tail_counts_at_376_questions():
    assert [nearest_rank(376, q) for q in (0.5, 0.95, 0.99)] == [188, 358, 373]
    assert tail_counts(376) == {"p50": 189, "p95": 19, "p99": 4, "max": 1}
    assert nearest_rank(1, 0.5) == 1 and nearest_rank(0, 0.5) == 1          # never below rank 1


@pytest.mark.parametrize("n", [1, 8, 48, 99])
def test_below_100_questions_p99_is_the_maximum(n):
    assert nearest_rank(n, 0.99) == n and tail_counts(n)["p99"] == 1
    values = [float(v) for v in random.Random(n).sample(range(1, 10_000), n)]
    clusters = [f"c{i}" for i in range(n)]
    assert latency_stats(values, clusters, seed=0)["lat_p99_ms"][0] == max(values)


def test_at_100_questions_p99_is_the_99th_value():
    assert nearest_rank(100, 0.99) == 99 and tail_counts(100)["p99"] == 2


def test_bootstrap_mean_draws_the_same_resamples_as_paired_bootstrap():
    rng = random.Random(7)
    values = [rng.uniform(100, 5000) for _ in range(60)]
    clusters = [f"g{i // 3}" for i in range(60)]
    strata = ["a" if i < 30 else "b" for i in range(60)]
    zeros = [0.0] * 60
    for st in (None, strata):
        for seed in (0, 3):
            assert bootstrap_mean(values, clusters, seed=seed, strata=st) == \
                paired_bootstrap(zeros, values, clusters, seed=seed, strata=st)
    point, lo, hi = bootstrap_mean(values, clusters, seed=0)
    assert point == pytest.approx(sum(values) / 60) and lo < point < hi
    with pytest.raises(ValueError, match="one length"):
        bootstrap_mean(values, clusters[:-1], seed=0)


def _old_percentiles(values, clusters, q, *, seed, strata=None, resamples=2000):
    """`bootstrap_percentiles` as it was before the weighted walk was factored out (callo bd716f0)."""
    xs = np.asarray(values, dtype=float)
    order = np.argsort(xs, kind="stable")
    xs_sorted = xs[order]
    idx, w = _bootstrap_weights(clusters, strata, seed, resamples)
    per_value = w[:, idx[order]]
    cum = np.cumsum(per_value, axis=1)
    rank = np.maximum(np.ceil(round(q * 10**6) * cum[:, -1] / 10**6), 1.0)
    pos = np.minimum((cum < rank[:, None]).sum(axis=1), len(xs) - 1)
    lo, hi = np.percentile(xs_sorted[pos], [2.5, 97.5])
    return float(xs_sorted[max(1, -(-round(q * 10**6) * len(xs) // 10**6)) - 1]), float(lo), float(hi)


@pytest.mark.parametrize("q", [0.5, 0.95, 0.99])
def test_bootstrap_percentiles_stay_bit_identical_after_the_walk_is_factored_out(q):
    rng = random.Random(11)
    values = [round(rng.uniform(200, 12_000), 3) for _ in range(120)] + [5000.0] * 6     # ties included
    clusters = [f"g{i // 2}" for i in range(126)]
    strata = ["s0" if i % 4 < 2 else "s1" for i in range(126)]
    for st in (None, strata):
        assert bootstrap_percentiles(values, clusters, q, seed=0, strata=st) == \
            _old_percentiles(values, clusters, q, seed=0, strata=st)


def test_latency_stats_are_the_five_profile_fields_on_a_hand_row_set():
    values = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 10_000.0]
    clusters = [f"c{i}" for i in range(10)]
    got = latency_stats(values, clusters, seed=0)
    assert list(got) == ["lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "lat_max_ms"]
    assert list(got) == list(STAT_FIELDS.values())
    assert got["lat_mean_ms"] == list(bootstrap_mean(values, clusters, seed=0)) and got["lat_mean_ms"][0] == 1450.0
    for stat, q in QUANTILES.items():
        assert got[STAT_FIELDS[stat]] == list(bootstrap_percentiles(values, clusters, q, seed=0))
    assert got["lat_p50_ms"][0] == 500.0 and got["lat_p99_ms"][0] == 10_000.0 and got["lat_max_ms"] == 10_000.0
    assert isinstance(got["lat_max_ms"], float)                        # a number, no interval


def _profile(strata="source", sources=("x", "x", "y", "y", "x", "x")):
    rows = [{"q": f"q{i}", "cluster": f"g{i // 2}", "source": s,
             "b": {"0": [True, 0.1, 0.9, 250.0 + 10 * i, 0, None],
                   **({"256": [i % 2 == 0, 0.2, 0.8, 1000.0 + 300 * i, 120, "natural"]} if i != 3 else {})},
             "e": {"0.95": [i % 3 == 0, 800.0 + 50 * i]}}
            for i, s in enumerate(sources)]
    return {"comparable": {"seed": 4, "data": {"strata": strata}},
            "row_fields": ["correct", "nll1", "pmax", "lat_ms", "k", "close"], "rows": rows}


def test_row_latencies_read_the_rows_that_hold_the_point_in_row_order():
    p = _profile()
    lats, clusters, strata = row_latencies(p, "b", "256")
    assert lats == [1000.0, 1300.0, 1600.0, 2200.0, 2500.0]                # row 3 holds no budget 256
    assert clusters == ["g0", "g0", "g1", "g2", "g2"] and strata == ["x", "x", "y", "x", "x"]
    assert row_latencies(p, "e", "0.95")[0] == [800.0 + 50 * i for i in range(6)]   # an EE pair: lat_ms at index 1
    assert row_latencies(_profile(strata=None), "b", "0")[2] is None                   # unstratified profile
    spans = _profile(sources=("x", "y", "y", "y", "x", "x"))                          # g0 spans two sources
    assert row_latencies(spans, "b", "0")[2] is None


def test_point_stats_keep_a_stored_field_and_recompute_an_absent_one_bit_for_bit():
    p = _profile()
    lats, clusters, strata = row_latencies(p, "b", "256")
    want = latency_stats(lats, clusters, seed=4, strata=strata)
    assert point_stats(p, {}, "b", "256") == want                        # every field absent: all recomputed
    stored = {**want, "lat_mean_ms": [1.0, 0.5, 1.5]}
    assert point_stats(p, stored, "b", "256") == stored                  # a stored field wins
    partial = {k: v for k, v in want.items() if k != "lat_p99_ms"}
    assert point_stats(p, partial, "b", "256") == want
    ee = point_stats(p, {}, "e", "0.95")
    assert ee == latency_stats(*row_latencies(p, "e", "0.95")[:2], seed=4, strata=row_latencies(p, "e", "0.95")[2])
