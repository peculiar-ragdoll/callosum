"""Is a position prior real?

One OMNIBUS test per option count n. The null: the distribution of the per-question rows (each question's
mean served probability per position over its n rotations) is invariant to cyclic shifts of positions (weaker than
full exchangeability; with no position effect every row is exactly uniform). A randomisation test over cyclic shifts
is then exact: each null draw shifts every CLUSTER's vectors by one independent uniform offset in 0..n-1 (so
within-cluster dependence is kept) and recomputes the statistic `bias_n = max_k |mean row[k] - 1/n|`;
p = (1 + #{draw >= observed}) / (draws + 1). Holm then runs across the n tested at family-wise alpha = 0.05. The
display intervals are a plain cluster bootstrap, per position, not simultaneous, and never stored in the profile.
Vectorised with numpy, in chunks of at most CHUNK_ELEMENTS gathered elements, so a large user set cannot
exhaust memory.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

ALPHA = 0.05
DRAWS = 2000
RESAMPLES = 2000
CHUNK_ELEMENTS = 4_000_000          # draws x questions x n gathered per chunk: about 32 MB of float64
_TIE = 1e-12                        # a draw equal to the observed statistic up to rounding counts as >=


def bias_stat(prior: np.ndarray) -> float:
    n = prior.shape[-1]
    return float(np.abs(prior - 1.0 / n).max())


def _cluster_index(clusters: Sequence[str]) -> tuple[np.ndarray, int]:
    ids: dict[str, int] = {}
    idx = np.array([ids.setdefault(c, len(ids)) for c in clusters], dtype=np.intp)
    return idx, len(ids)


def permutation_test(priors: np.ndarray, clusters: Sequence[str], *, draws: int = DRAWS,
                     seed: int = 0) -> tuple[float, float]:
    """(observed bias_n, p) for one n. `priors` is (questions, n), the question rows; `clusters` names each row's
    cluster."""
    q, n = priors.shape
    observed = bias_stat(priors.mean(axis=0))
    idx, n_clusters = _cluster_index(clusters)
    rng = np.random.default_rng([seed, n])
    pos = np.arange(n)
    per_chunk = max(1, CHUNK_ELEMENTS // (q * n))
    at_least, done = 0, 0
    while done < draws:
        d = min(per_chunk, draws - done)
        shifts = rng.integers(0, n, size=(d, n_clusters))[:, idx]                 # (d, q): one offset per cluster
        cols = (pos[None, None, :] + shifts[:, :, None]) % n                       # (d, q, n)
        shifted = np.take_along_axis(np.broadcast_to(priors, (d, q, n)), cols, axis=2)
        stats = np.abs(shifted.mean(axis=1) - 1.0 / n).max(axis=1)
        at_least += int(np.count_nonzero(stats >= observed - _TIE))
        done += d
    return observed, (1 + at_least) / (draws + 1)


def holm(pvalues: Mapping[int, float]) -> dict[int, float]:
    """Holm-adjusted p-values: sorted ascending, p_(i) * (m - i + 1), made monotone, capped at 1."""
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def bootstrap_intervals(priors: np.ndarray, clusters: Sequence[str], *, resamples: int = RESAMPLES,
                        seed: int = 0) -> list[tuple[float, float]]:
    """95% percentile intervals of the mean row per position from a cluster bootstrap (display only)."""
    q, n = priors.shape
    idx, c = _cluster_index(clusters)
    sums = np.zeros((c, n))
    np.add.at(sums, idx, priors)
    counts = np.bincount(idx, minlength=c).astype(float)
    rng = np.random.default_rng([seed, n, 1])
    per_chunk = max(1, CHUNK_ELEMENTS // max(1, c))
    est, done = [], 0
    while done < resamples:
        r = min(per_chunk, resamples - done)
        w = rng.multinomial(c, np.full(c, 1.0 / c), size=r).astype(float)        # (r, c) cluster multiplicities
        est.append((w @ sums) / (w @ counts)[:, None])
        done += r
    lo, hi = np.percentile(np.concatenate(est), [2.5, 97.5], axis=0)
    return [(float(a), float(b)) for a, b in zip(lo, hi, strict=True)]


@dataclass(frozen=True)
class Decision:
    n: int
    kept: bool
    bias: float | None          # the observed bias_n (None when the n was not tested)
    p: float | None
    p_holm: float | None
    reason: str | None          # why it was dropped; None when kept


def decide(fits: Mapping[int, tuple[np.ndarray, Sequence[str]]], *, min_questions: int = 24, alpha: float = ALPHA,
           draws: int = DRAWS, seed: int = 0) -> dict[int, Decision]:
    """An n is KEPT when it has at least `min_questions` questions read under every rotation AND its
    Holm-adjusted test rejects. An n below the minimum is dropped untested and is not part of Holm's family."""
    out: dict[int, Decision] = {}
    tested: dict[int, tuple[float, float]] = {}
    for n, (priors, clusters) in sorted(fits.items()):
        if priors.shape[0] < min_questions:
            out[n] = Decision(n, False, None, None, None,
                              f"fewer than {min_questions} questions read under every rotation ({priors.shape[0]})")
        else:
            tested[n] = permutation_test(priors, clusters, draws=draws, seed=seed)
    adjusted = holm({n: p for n, (_, p) in tested.items()})
    for n, (stat, p) in tested.items():
        keep = adjusted[n] <= alpha
        out[n] = Decision(n, keep, stat, p, adjusted[n], None if keep else f"not significant (p_holm {adjusted[n]:.2g})")
    return dict(sorted(out.items()))
