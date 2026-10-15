"""The convergence measurement, the pure half: the user's "n measurements in
a row at y certainty means z probability of convergence", fitted on probe readouts every K tokens along calibrate's
Bmax traces.

- The (n, y) stop rule over a 4 x 6 grid, each cell's convergence `conv` (the stop's argmax equals the end answer F)
  and its one-sided exact Clopper-Pearson lower bound in the fractional-Beta form, deflated by the cluster design
  effect, and the grid's greatest monotone minorant, recorded beside it and never used.
- For each target z: the fixed-sequence test from the least to the most permissive cell at full alpha, stopping at
  the first failure, with count-only skips; the rule is the passed cell with the largest net saving, and its
  reported numbers are cross-fitted over 2 seeded cluster folds.
- The flip statistics, the early-exit points, the gain test on the model's frontier of budget,
  deadline and EVERY early-exit point, and the break-even map over the probe cost in decode steps,
  rho = c / d.

Points are the pick's keyword-built `think.Point`s (an early-exit point is `kind="ee"`, `value=z`) with the
five latency statistics (`think.latency_stats`); every paired interval is a `think.Paired` over the convergence
questions in one order. Calibrate (`calibrate_think.convergence_block`) builds the inputs from its raw file.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .think import (
    EE_KINDS,
    Paired,
    Point,
    ee_key,
    finite_number,
    finite_positive,
    fit_T,
    frontier,
    latency_stats,
    paired_bootstrap,
    point_latencies,
)

GRID_N = (1, 2, 3, 4)
GRID_Y = (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
TARGETS = (0.9, 0.95, 0.99)
ALPHA = 0.05
MIN_STOPS = 30                  # a cell with fewer raw stops is skipped ...
MIN_STOP_SHARE = 0.10           # ... or with stops on fewer than 10 % of the questions
RHOS = (0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)   # 3 is the pure-attention edge
CONFIDENT = 0.9                 # "changed its mind after a confident probe"
PROBE_KS = (32, 64, 128)        # --converge-probe's K
# early exit's gate (experimental): one budget-0 question, read before thinking on the request's own state, whose yes
# serves the request without early exit. Two options, so the threshold is the argmax at any temperature. Serve reads it
# (`proxy.GATE_QUESTION`), so does calibrate's think probe, and the fit applies the same rule (`gate_skips`).
GATE_QID = "callo_converge_gate"
GATE_INSTRUCTIONS = ("Before answering, would you need to apply several stated rules or conditions step by step to "
                     "reach the answer?")
GATE_CRITERIA = {"a": "yes", "b": "no"}
GATE_SKIP_P = 0.5
# the gate's identity: every gate read calibrate measures and the profile's `converge.gate` carry it, so a gated point
# names the question it was measured with
GATE_SHA256 = hashlib.sha256(json.dumps({"instructions": GATE_INSTRUCTIONS, "criteria": GATE_CRITERIA},
                                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()

Cell = tuple[int, float]


def gate_skips(p: float | None) -> bool:
    """The gate's rule, serve's and the fit's: a yes (P(yes) >= GATE_SKIP_P) or a failed read (None) serves the
    request without early exit."""
    return p is None or p >= GATE_SKIP_P


@dataclass(frozen=True)
class Probe:
    """One probe readout at k tokens, read at T = 1 (y is defined at T = 1)."""
    k: int
    probs: tuple[float, ...]

    @property
    def argmax(self) -> int:
        return max(range(len(self.probs)), key=self.probs.__getitem__)

    @property
    def pmax(self) -> float:
        return max(self.probs)


@dataclass(frozen=True)
class QTrace:
    """One question's Bmax trace as the fit reads it: its probes, the final readout F (at n_gen), its label and
    cluster, its latency terms (`deadline.trace_terms`' a and d) and the latency of its end read (the Bmax point's
    lat_ms). A question without a trace (one option) has n_gen 0 and no probes: it never stops."""
    uid: str
    cluster: str
    label: int
    n_gen: int
    final: tuple[float, ...]
    probes: tuple[Probe, ...]
    a_ms: float
    d_ms: float
    end_lat_ms: float
    source: str | None = None

    @property
    def final_argmax(self) -> int:
        return max(range(len(self.final)), key=self.final.__getitem__)


def stop_index(probes: Sequence[Probe], n: int, y: float) -> int | None:
    """The first probe j whose last n probes (j - n + 1 .. j) share one argmax, each with max p >= y; None."""
    for j in range(n - 1, len(probes)):
        run = probes[j - n + 1: j + 1]
        if len({p.argmax for p in run}) == 1 and all(p.pmax >= y for p in run):
            return j
    return None


# ---- the regularized incomplete beta, its inverse, the bound and the p-value --------------------------------

def _betacf(x: float, a: float, b: float) -> float:
    """The continued fraction of I_x(a, b) (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(x: float, a: float, b: float) -> float:
    """The regularized incomplete beta I_x(a, b) for a, b > 0 (real, so fractional counts work)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """x with I_x(a, b) = q, by bisection (I is increasing in x)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if betainc(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def conv_lower(x_e: float, n_e: float, alpha: float = ALPHA) -> float:
    """Beta^-1(alpha; x_e, n_e - x_e + 1), 0 when x_e = 0: the one-sided exact Clopper-Pearson lower bound."""
    if x_e <= 0 or n_e <= 0:
        return 0.0
    return beta_ppf(alpha, x_e, n_e - x_e + 1.0)


def p_value(x_e: float, n_e: float, z: float) -> float:
    """The one-sided p for H0 conv <= z: I_z(x_e, n_e - x_e + 1), P(Bin(n, z) >= x) for integer counts."""
    if x_e <= 0 or n_e <= 0:
        return 1.0
    return betainc(z, x_e, n_e - x_e + 1.0)


def zero_flip_minimum(z: float, alpha: float = ALPHA) -> int:
    """The effective stops a flip-free cell needs to reach z: ceil(ln alpha / ln z) (29, 59, 299 at .9, .95, .99)."""
    return math.ceil(math.log(alpha) / math.log(z) - 1e-12)


def icc(groups: Sequence[Sequence[float]]) -> float:
    """The one-way ANOVA estimator of the intraclass correlation of a 0/1 indicator over clusters, floored at 0;
    0 when the indicator has no variance (a flip-free cell gives 0/0), with fewer than 2 clusters, or with no cluster
    holding 2 values."""
    groups = [list(g) for g in groups if g]
    n = sum(len(g) for g in groups)
    k = len(groups)
    if k < 2 or n <= k:
        return 0.0
    ys = [y for g in groups for y in g]
    mean = sum(ys) / n
    if all(y == ys[0] for y in ys):
        return 0.0
    ssb = sum(len(g) * (sum(g) / len(g) - mean) ** 2 for g in groups)
    ssw = sum((y - sum(g) / len(g)) ** 2 for g in groups for y in g)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    den = msb + (m0 - 1) * msw
    return max(0.0, (msb - msw) / den) if den > 0 else 0.0


# ---- one question under a cell, and a cell's statistics -----------------------------------------------------------

@dataclass(frozen=True)
class Outcome:
    """One question answered under a rule: the stop probe's readout, else F."""
    stopped: bool
    probs: tuple[float, ...]
    k: int                      # the tokens thought: the stop position, else n_gen
    before: int                 # probes read before the answer (the stop probe itself is the answer's read)
    agrees: bool                # its argmax equals F's

    def correct(self, label: int) -> bool:
        return max(range(len(self.probs)), key=self.probs.__getitem__) == label


def outcome(q: QTrace, cell: Cell | None) -> Outcome:
    j = stop_index(q.probes, *cell) if cell is not None else None
    if j is None:
        return Outcome(False, q.final, q.n_gen, len(q.probes), True)
    p = q.probes[j]
    return Outcome(True, p.probs, p.k, j, p.argmax == q.final_argmax)


def latency(q: QTrace, o: Outcome, c_ms: float, read_ms: float) -> float:
    """A stop costs a + d·k_stop + (probes before it)·c + a forced read; a question that never stops pays its
    end read's latency plus every probe's c."""
    if o.stopped:
        return q.a_ms + q.d_ms * o.k + o.before * c_ms + read_ms
    return q.end_lat_ms + o.before * c_ms


def net_saving(qs: Sequence[QTrace], cell: Cell, rho: float) -> float:
    """tokens_saved - probes x c/d, in decode steps, per question: tokens saved by stopping early, minus every
    probe read before the answer (the answer's own read is the forced read every question pays) at rho steps
    each (as `latency` charges c)."""
    if not qs:
        return 0.0
    outs = [outcome(q, cell) for q in qs]
    saved = sum(q.n_gen - o.k for q, o in zip(qs, outs, strict=True)) / len(qs)
    probes = sum(o.before for o in outs) / len(qs)
    return saved - probes * rho


def cell_stats(qs: Sequence[QTrace], cell: Cell, *, rho: float | None, seed: int,
               strata: Sequence[str] | None = None) -> dict:
    """One cell in sample: stops (count, share), conv, its bound's inputs (m̄, ICC, deff, the effective counts)
    and conv_lo, tokens saved, probes read before the answer (as the latency charges them), acc and the paired
    dacc against F [CI], T fitted on the stop readouts, and the net saving at rho (None without a probe cost)."""
    outs = [outcome(q, cell) for q in qs]
    stops = [(q, o) for q, o in zip(qs, outs, strict=True) if o.stopped]
    x = sum(o.agrees for _, o in stops)
    by_cluster: dict[str, list[float]] = {}
    for q, o in stops:
        by_cluster.setdefault(q.cluster, []).append(1.0 if o.agrees else 0.0)
    m_bar = len(stops) / len(by_cluster) if by_cluster else 1.0
    rho_icc = icc(list(by_cluster.values()))
    deff = max(1.0, 1.0 + (m_bar - 1.0) * rho_icc)
    n_e, x_e = len(stops) / deff, x / deff
    final = [1.0 if q.final_argmax == q.label else 0.0 for q in qs]
    answered = [1.0 if o.correct(q.label) else 0.0 for q, o in zip(qs, outs, strict=True)]
    dacc = list(paired_bootstrap(final, answered, [q.cluster for q in qs], seed=seed, strata=strata)) if qs else None
    stop_rows = [(list(o.probs), q.label) for q, o in stops]
    return {"n": cell[0], "y": cell[1], "stops": len(stops), "share": len(stops) / len(qs) if qs else 0.0,
            "conv": x / len(stops) if stops else None, "m_bar": m_bar, "icc": rho_icc, "deff": deff,
            "n_e": n_e, "x_e": x_e, "conv_lo": conv_lower(x_e, n_e),
            "tokens_saved": sum(q.n_gen - o.k for q, o in zip(qs, outs, strict=True)) / len(qs) if qs else 0.0,
            "probes": sum(o.before for o in outs) / len(qs) if qs else 0.0,
            "acc": sum(answered) / len(qs) if qs else None, "dacc": dacc,
            "T": fit_T(stop_rows) if len(stop_rows) >= 2 else None,
            "net_saving": net_saving(qs, cell, rho) if rho is not None else None}


def monotone_fit(cells: Mapping[Cell, Mapping]) -> dict[Cell, float | None]:
    """The optional smoother, recorded and never used: the greatest monotone minorant of conv over the nested grid,
    conv_mono(n, y) = the smallest conv among the cells at least as strict (n' >= n and y' >= y). It is non-decreasing
    in n and in y and never above the cell's own conv; None where no such cell stopped."""
    out: dict[Cell, float | None] = {}
    for n, y in cells:
        stricter = [c["conv"] for (n2, y2), c in cells.items() if n2 >= n and y2 >= y and c.get("conv") is not None]
        out[(n, y)] = min(stricter) if stricter else None
    return out


def sequence_order() -> list[Cell]:
    """Least to most permissive: y from .99 down to .6, within each y n from 4 down to 1."""
    return [(n, y) for y in sorted(GRID_Y, reverse=True) for n in sorted(GRID_N, reverse=True)]


def fixed_sequence(cells: Mapping[Cell, Mapping], z: float, n_questions: int, alpha: float = ALPHA) -> list[dict]:
    """Each cell in `sequence_order`, tested at full alpha until the first failure. A cell is skipped,
    spending no alpha, when its counts alone make it untestable: fewer than 30 raw stops or than 10 % of the questions,
    or stops / m̄ (the effective count at the worst-case design effect) below the zero-flip minimum for z. The skip
    never consults agreements; deff enters only the test. Verdicts: pass | fail | skip | untested (after a failure)."""
    out, failed = [], False
    need = zero_flip_minimum(z, alpha)
    for cell in sequence_order():
        c = cells[cell]
        if failed:
            out.append({"cell": list(cell), "verdict": "untested", "p": None})
            continue
        if c["stops"] < MIN_STOPS or c["stops"] < MIN_STOP_SHARE * n_questions or c["stops"] / c["m_bar"] < need:
            out.append({"cell": list(cell), "verdict": "skip", "p": None})
            continue
        p = p_value(c["x_e"], c["n_e"], z)
        ok = p <= alpha
        failed = not ok
        out.append({"cell": list(cell), "verdict": "pass" if ok else "fail", "p": p})
    return out


def passed(sequence: Sequence[Mapping]) -> list[Cell]:
    return [(int(v["cell"][0]), float(v["cell"][1])) for v in sequence if v["verdict"] == "pass"]


def folds(qs: Sequence[QTrace], seed: int) -> tuple[list[QTrace], list[QTrace]]:
    """Two seeded cluster folds (as think budgets' cross-fit)."""
    names = sorted({q.cluster for q in qs})
    random.Random(f"{seed}:converge").shuffle(names)
    half = set(names[: len(names) // 2])
    return [q for q in qs if q.cluster in half], [q for q in qs if q.cluster not in half]


def crossfit_cells(qs: Sequence[QTrace], candidates: Sequence[Cell], rho: float, seed: int) -> dict[str, Cell | None]:
    """Each question's cell, chosen on the OTHER fold: the candidate with the largest net saving there (from the
    full-data passed set, without rerunning the sequence at half n). {uid: cell}; None when there is no candidate."""
    if not candidates:
        return {q.uid: None for q in qs}
    a, b = folds(qs, seed)
    if not a or not b:                               # one cluster: no fold to choose on, so choose in sample
        best = max(candidates, key=lambda c: net_saving(qs, c, rho))
        return {q.uid: best for q in qs}
    on_a = max(candidates, key=lambda c: net_saving(a, c, rho))
    on_b = max(candidates, key=lambda c: net_saving(b, c, rho))
    return {**{q.uid: on_a for q in b}, **{q.uid: on_b for q in a}}


@dataclass(frozen=True)
class Rule:
    """One target z's rule: its full-data cell, the cross-fitted per-question outcomes and their summary."""
    z: float
    cell: Cell
    assigned: dict[str, Cell | None]
    outcomes: dict[str, Outcome]
    summary: dict


def rule_for(qs: Sequence[QTrace], cells: Mapping[Cell, Mapping], sequence: Sequence[Mapping], z: float, rho: float,
             seed: int, strata: Sequence[str] | None = None) -> Rule | None:
    """The passed cell with the largest net saving (in sample), and the cross-fitted numbers the summary prints:
    conv, acc, dacc [CI], the probes read before the answer and the net saving, evaluated on each question under the
    cell chosen on the other fold. None: no cell passed for z ("no rule reaches z", not a failure)."""
    cands = passed(sequence)
    if not cands:
        return None
    best = max(cands, key=lambda c: net_saving(qs, c, rho))
    assigned = crossfit_cells(qs, cands, rho, seed)
    outs = {q.uid: outcome(q, assigned[q.uid]) for q in qs}
    stops = [outs[q.uid] for q in qs if outs[q.uid].stopped]
    final = [1.0 if q.final_argmax == q.label else 0.0 for q in qs]
    answered = [1.0 if outs[q.uid].correct(q.label) else 0.0 for q in qs]
    saved = sum(q.n_gen - outs[q.uid].k for q in qs) / len(qs)
    probes = sum(outs[q.uid].before for q in qs) / len(qs)              # read before the answer
    stop_rows = [(list(outs[q.uid].probs), q.label) for q in qs if outs[q.uid].stopped]
    summary = {"cell": list(best), "conv_lo": cells[best]["conv_lo"],
               "conv": sum(o.agrees for o in stops) / len(stops) if stops else None,
               "stops": len(stops), "acc": sum(answered) / len(qs),
               "dacc": list(paired_bootstrap(final, answered, [q.cluster for q in qs], seed=seed, strata=strata)),
               "tokens_saved": saved, "probes": probes, "net_saving": saved - probes * rho,
               "T": fit_T(stop_rows) if len(stop_rows) >= 2 else None}
    return Rule(z, best, assigned, outs, summary)



# ---- the serve side: the profile's rules and the stop test after each probe ------------------------------------

@dataclass(frozen=True)
class ServeRule:
    """One target's served rule: the target z, its full-data cell (n, y) and T fitted on its stop readouts (None when
    the fit had fewer than 2 stops)."""
    z: float
    n: int
    y: float
    T: float | None


@dataclass(frozen=True)
class ServeRules:
    """A think profile's convergence block as serve reads it: K (used as is), each target's rule (None: no rule
    reaches it) and the measured probe cost (None: not measured)."""
    K: int
    rules: dict[float, ServeRule | None]
    c_ms: float | None

    def for_z(self, z: float) -> ServeRule | None:
        """The rule serving a requested z in (0, 1): the smallest target at or above z (a z between targets takes the
        next higher target's rule; z <= .9 takes .9's), None when that target has no rule or z is above every target.
        A z within 1e-9 of a target is that target (float noise in a client's 0.9)."""
        above = [t for t in sorted(self.rules) if t >= z - 1e-9]
        return self.rules[above[0]] if above else None

    def as_dict(self) -> dict:
        """`/callo/stats.think.converge`'s static part: K and each target's n, y and T (None: no rule)."""
        return {"K": self.K, "rules": {ee_key(t): (None if r is None else {"n": r.n, "y": r.y, "T": r.T})
                                       for t, r in sorted(self.rules.items())}}


def serve_rules(block: Any) -> tuple[ServeRules | None, str | None]:
    """(ServeRules, None), or (None, why) when a profile cannot serve early exit: no block, a block that is not an
    object, a K other than 32, 64 or 128, rules not keyed by exactly the targets (an empty rules object is no rule at
    any target, not a fault), a cell off the grid, a T that is not null or finite and positive, a `cells` that is
    present (not null) but not a list or holds an entry that is not an object with an integer n, a finite y and a T
    that is null or finite and positive (every entry is checked, whether or not a rule uses it), a `cost` that is
    present (not null) but not an object, a cost.c_ms that is not null or a finite non-negative number. A rule's
    served T is its full-data cell's (`cells`, fitted on that cell's stop readouts over every calibration question:
    the cell serve runs), else the rule summary's cross-fitted T."""
    if block is None:
        return None, "it has no convergence block"
    if not isinstance(block, Mapping):
        return None, "its convergence block is not an object"
    k = block.get("K")
    if isinstance(k, bool) or k not in PROBE_KS:
        return None, f"its convergence block's K must be one of {', '.join(map(str, PROBE_KS))}, got {k!r}"
    rules = block.get("rules")
    if isinstance(rules, Mapping) and not rules:     # calibrate's empty fit (no question had a trace): no rule at all
        rules = {ee_key(z): None for z in TARGETS}
    if not isinstance(rules, Mapping) or set(rules) != {ee_key(z) for z in TARGETS}:
        return None, "its convergence block's rules must be keyed by the targets 0.9, 0.95 and 0.99"
    cell_t: dict[tuple[int, float], Any] = {}       # the full-data cells' T (cell_stats: fitted on all their stops)
    cells = block.get("cells")
    if cells is not None and not isinstance(cells, list):
        return None, f"its convergence block's cells must be a list, got {type(cells).__name__}"
    for c in cells or []:
        if not (isinstance(c, Mapping) and isinstance(c.get("n"), int) and not isinstance(c.get("n"), bool)
                and finite_number(c.get("y"))):
            return None, f"its convergence block's cells must be objects with an integer n and a finite y, got {c!r}"
        t = c.get("T")
        if t is not None and not finite_positive(t):
            return None, f"its convergence block's cells' T must be null or finite and positive, got {t!r}"
        cell_t[(int(c["n"]), float(c["y"]))] = t
    out: dict[float, ServeRule | None] = {}
    for z in TARGETS:
        r = rules[ee_key(z)]
        if r is None:
            out[z] = None
            continue
        cell = r.get("cell") if isinstance(r, Mapping) else None
        if not (isinstance(cell, list) and len(cell) == 2 and isinstance(cell[0], int) and not isinstance(cell[0], bool)
                and cell[0] in GRID_N and finite_number(cell[1]) and float(cell[1]) in GRID_Y):
            return None, f"rule {ee_key(z)}'s cell must be [n, y] on the grid, got {cell!r}"
        t = r.get("T")
        if t is not None and not finite_positive(t):
            return None, f"rule {ee_key(z)}'s T must be null or finite and positive, got {t!r}"
        key = (int(cell[0]), float(cell[1]))
        t = cell_t[key] if key in cell_t else t       # served T: the full-data cell's, else the summary's (older block)
        out[z] = ServeRule(z, int(cell[0]), float(cell[1]), None if t is None else float(t))
    cost = block.get("cost")
    if cost is not None and not isinstance(cost, Mapping):
        return None, f"its convergence block's cost must be an object, got {type(cost).__name__}"
    c = (cost or {}).get("c_ms")
    if c is not None and not (finite_number(c) and c >= 0):
        return None, f"its convergence block's cost.c_ms must be null or a finite non-negative number, got {c!r}"
    return ServeRules(int(k), out, None if c is None else float(c)), None


UNSERVED = "callo serve cannot run it"       # the pick's exclusion reason for an early-exit point serve cannot run


def unservable(points: Sequence[Point], block: Any) -> dict[str, str]:
    """{label: why} for each early-exit point among `points` (one model's, plain or gated) that `callo serve
    --think-converge z` cannot run as measured, from that model's profile `converge` block: every one when serve refuses
    the block (`serve_rules`), else each whose z `for_z` finds no rule for (serve would run it without early exit). A
    gated point shares its z's verdict: the gate only decides per request whether that rule runs. The pick excludes them
    first (`think.pick`'s `unservable`); budget and deadline points are never named."""
    ee = [p for p in points if p.kind in EE_KINDS]
    if not ee:
        return {}
    rules, why = serve_rules(block)
    out = {}
    for p in ee:
        z = float(p.value or 0.0)
        if rules is None:
            out[p.label] = f"{UNSERVED}: --think-converge refuses this profile ({why})"
        elif rules.for_z(z) is None:
            out[p.label] = (f"{UNSERVED}: no early-exit rule reaches z {ee_key(z).removeprefix('0')} (serve would run "
                            f"without early exit)")
    return out


def converged(probes: Sequence[Probe], n: int, y: float) -> bool:
    """The serve-side stop test, run after each probe: the last n probes share one argmax, each with max p >= y (read
    at T = 1). Checked after every probe, it first holds at `stop_index`'s probe."""
    run = list(probes[-n:]) if n <= len(probes) else []
    return len(run) == n and len({p.argmax for p in run}) == 1 and all(p.pmax >= y for p in run)

# ---- the flip statistics ---------------------------------------------------------------------------------

def flip_stats(qs: Sequence[QTrace], k: int) -> dict:
    """The argmax flip rate between consecutive probes by position (the later probe's k), the traces whose F
    differs from their first probe with max p >= .9 with right->wrong and wrong->right counts between that probe and F,
    and the accuracy at each probe position j over the questions still thinking there (the data for a useful
    minimum)."""
    flips: dict[int, list[int]] = {}
    acc: dict[int, list[int]] = {}
    confident = changed = right_wrong = wrong_right = 0
    for q in qs:
        for j, p in enumerate(q.probes):
            acc.setdefault(p.k, []).append(1 if p.argmax == q.label else 0)
            if j:
                flips.setdefault(p.k, []).append(1 if p.argmax != q.probes[j - 1].argmax else 0)
        first = next((p for p in q.probes if p.pmax >= CONFIDENT), None)
        if first is not None:
            confident += 1
            if first.argmax != q.final_argmax:
                changed += 1
                right_wrong += first.argmax == q.label
                wrong_right += q.final_argmax == q.label
    return {"K": k, "flip_rate": {str(pos): sum(v) / len(v) for pos, v in sorted(flips.items())},
            "flip_n": {str(pos): len(v) for pos, v in sorted(flips.items())},
            "confident": confident, "changed_after_confident": changed,
            "changed_share": changed / confident if confident else None,
            "right_to_wrong": right_wrong, "wrong_to_right": wrong_right,
            "acc_at": {str(pos): sum(v) / len(v) for pos, v in sorted(acc.items())},
            "acc_n": {str(pos): len(v) for pos, v in sorted(acc.items())}}


# ---- early-exit points, the gain test and the rho map -------------------------------------------------

@dataclass(frozen=True)
class Scored:
    """A frontier point with its per-question (correct, lat_ms) over the convergence questions, in their one order
    (the pairing contract: `paired` reads these rows)."""
    point: Point
    correct: tuple[float, ...]
    lat: tuple[float, ...]


def scored(point: Point, rows: Sequence[tuple[float, float]]) -> Scored:
    return Scored(point, tuple(float(c) for c, _ in rows), tuple(float(x) for _, x in rows))


def point_over(name: str, qs: Sequence[QTrace], rows: Sequence[tuple[float, float]], *, kind: str, value: float,
               seed: int, strata: Sequence[str] | None = None, cap: int | None = None,
               think_mean: float | None = None) -> Scored:
    """A budget, deadline or early-exit point (keyword-built; a budget's value is derived from its budget)
    over the convergence questions: its accuracy and the five latency statistics (`think.latency_stats`) from
    `rows`, (correct, lat_ms) per question in `qs` order."""
    lats = [float(x) for _, x in rows]
    stats = latency_stats(lats, [q.cluster for q in qs], seed=seed, strata=strata)
    common = {"name": name, "acc": sum(float(c) for c, _ in rows) / len(rows), "n": len(rows), "think_mean": think_mean,
              **point_latencies(stats)}
    if kind == "budget":
        p = Point(kind="budget", budget=int(value), **common)
    else:
        p = Point(kind=kind, value=value, cap=cap, **common)
    return scored(p, rows)


def ee_rows(qs: Sequence[QTrace], rule: Rule, *, c_ms: float, read_ms: float) -> list[tuple[float, float]]:
    """Each question's (correct, lat_ms) under its cross-fitted cell: the stop readout, else F, at
    `latency`'s cost."""
    return [(1.0 if rule.outcomes[q.uid].correct(q.label) else 0.0, latency(q, rule.outcomes[q.uid], c_ms, read_ms))
            for q in qs]


def ee_point(name: str, qs: Sequence[QTrace], rule: Rule, *, c_ms: float, read_ms: float, seed: int,
             strata: Sequence[str] | None = None, cap: int | None = None) -> Scored:
    """`<model>@ee<z>` (`kind="ee"`, `value=z`, `cap` the Bmax cap it runs to): every question answered and
    timed under its cross-fitted cell."""
    ks = [rule.outcomes[q.uid].k for q in qs]
    return point_over(name, qs, ee_rows(qs, rule, c_ms=c_ms, read_ms=read_ms), kind="ee", value=rule.z, seed=seed,
                      strata=strata, cap=cap, think_mean=sum(ks) / len(ks))


def pairing(points: Sequence[Scored], qs: Sequence[QTrace], seed: int, strata: Sequence[str] | None = None) -> Paired:
    """The `Paired` over the convergence questions, each point's rows keyed on its kind and value."""
    table = {(s.point.kind, s.point.value): (s.correct, s.lat) for s in points}
    return Paired(lambda p: table[(p.kind, p.value)], tuple(q.cluster for q in qs), seed,
                  tuple(strata) if strata is not None else None)


def gain(ee: Scored, others: Sequence[Scored], paired: Paired, *, peers: Sequence[Scored] = ()) -> dict:
    """On each latency axis (p50, p95), E gains when it is on the frontier of the model's budget, deadline
    and early-exit points (`others` and E's `peers`, every other early-exit point) and either (speed) its latency
    interval lies wholly below that of every non-EE point whose accuracy is within noise of or above E's (that point
    minus E has a CI reaching 0 or above), there being at least one such point, or (accuracy) its paired Δacc against
    every non-EE point no slower than E has a CI above 0, there being at least one. Within noise of a neighbour is no
    gain. `paired.dacc(a, b)` is b - a."""
    out: dict = {}
    above = {o.point.label: paired.dacc(ee.point, o.point) for o in others}       # o - E (the same on either axis)
    below = {o.point.label: paired.dacc(o.point, ee.point) for o in others}       # E - o
    for axis in ("p50", "p95"):
        ranked = frontier([ee.point, *(p.point for p in peers), *(o.point for o in others)], latency=axis)
        on_front = next(p for p in ranked if p.label == ee.point.label).dominated_by is None
        e_lat = ee.point.latency(axis)
        level = [o for o in others if above[o.point.label][2] >= 0]          # o within noise of E, or above it
        speed = bool(level) and all(e_lat[2] < o.point.latency(axis)[1] for o in level)
        faster = [o for o in others if o.point.latency(axis)[0] <= e_lat[0]]
        accuracy = bool(faster) and all(below[o.point.label][1] > 0 for o in faster)
        out[axis] = {"front": on_front, "speed": speed, "accuracy": accuracy, "gains": on_front and (speed or accuracy)}
    out["gains"] = out["p50"]["gains"] or out["p95"]["gains"]
    return out


def gains(name: str, qs: Sequence[QTrace], rules: Mapping[float, Rule | None], others: Sequence[Scored], *,
          c_ms: float, read_ms: float, seed: int, strata: Sequence[str] | None = None,
          cap: int | None = None) -> tuple[dict, dict]:
    """Every rule's early-exit point at probe cost `c_ms` and its gain verdict against `others`, with ALL the
    early-exit points on the frontier together. Returns ({ee_key(z): Scored}, {ee_key(z): verdict})."""
    ees = {ee_key(z): ee_point(name, qs, r, c_ms=c_ms, read_ms=read_ms, seed=seed, strata=strata, cap=cap)
           for z, r in rules.items() if r is not None}
    paired = pairing([*others, *ees.values()], qs, seed, strata)
    verdicts = {z: gain(e, others, paired, peers=[x for k, x in ees.items() if k != z]) for z, e in ees.items()}
    return ees, verdicts


def rho_map(name: str, qs: Sequence[QTrace], cells: Mapping[Cell, Mapping], sequences: Mapping[float, Sequence[Mapping]],
            others: Sequence[Scored], *, d_ms: float, read_ms: float, seed: int, strata: Sequence[str] | None = None,
            rhos: Sequence[float] = RHOS, cap: int | None = None) -> dict:
    """The gain test re-evaluated at each probe cost rho = c / d (in decode steps): each z's rule (its choice
    moves with rho; the passed sets do not) and its early-exit point at c = rho x d, all of them on the frontier
    together. `by_rho` maps each rho to the targets that gain there; rho_max is the largest rho at which some z
    gains; rho_prefix the largest rho such that some z gains at EVERY grid rho up to it, None when
    the cheapest does not gain. Each rho re-selects its rules, so a rho below rho_max may not gain: early exit "pays
    while a probe costs <= rho_prefix", never "<= rho_max"."""
    by_rho: dict[str, list[float]] = {}
    for rho in rhos:
        rules = {z: rule_for(qs, cells, seq, z, rho, seed, strata) for z, seq in sequences.items()}
        _, verdicts = gains(name, qs, rules, others, c_ms=rho * d_ms, read_ms=read_ms, seed=seed, strata=strata,
                            cap=cap)
        by_rho[f"{rho:g}"] = [float(z) for z, v in verdicts.items() if v["gains"]]
    rho_max = max((rho for rho in rhos if by_rho[f"{rho:g}"]), default=None)
    prefix = None                           # gains at every grid rho from the cheapest up
    for rho in sorted(rhos):
        if not by_rho[f"{rho:g}"]:
            break
        prefix = rho
    return {"by_rho": by_rho, "rho_max": rho_max, "rho_prefix": prefix}
