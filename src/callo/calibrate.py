"""One-scalar temperature fit: minimum mean NLL over a 181-point log grid on 0.25..16, every question weighted
equally: Kev's procedure, with the grid extended to 16 (0.3.0). The extension keeps the old 121-point
grid on 0.25..4 as its bitwise prefix (the same step, the same float operations), so a fit inside 0.25..4 comes out
exactly as before. A T on the grid's first or last point is a real signal, not an artefact of the grid's size
(`limit_note`).

callo serves probabilities renormalised over the options at `--temperature` T from the log-marginals:
p_i(T) = softmax(lm / T). Since log p_i(1) = lm_i - logsumexp(lm), dividing the LOG-PROBABILITIES returned at
T = 1 by T and renormalising gives exactly softmax(lm / T) (the per-row constant cancels). So a fit on the
served probabilities recovers the same T a fit on the raw log-marginals would. Stdlib only.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from .errors import InvalidInput

# The step is the old grid's (121 points on 0.25..4), kept as its numerator and denominator: `i * SPAN / STEPS` and
# not `i * (SPAN / STEPS)`, which rounds differently, so GRID[:121] is bitwise the old grid.
STEP_SPAN, STEP_STEPS = math.log(4.0) - math.log(0.25), 120
GRID_POINTS, GRID_LO = 181, 0.25
GRID = [math.exp(math.log(GRID_LO) + i * STEP_SPAN / STEP_STEPS) for i in range(GRID_POINTS)]
GRID_HI = GRID[-1]              # 15.999999999999991, not exactly 16.0
GRID_TEXT = f"{GRID_POINTS}-point log grid on {GRID_LO:g}..{GRID_HI:g}"
EPS = 1e-12
Row = tuple[list[float], int]   # (probabilities in key order, served at T = 1; label index)


def temper(probs: Sequence[float], temperature: float) -> list[float]:
    z = [math.log(max(float(p), EPS)) / temperature for p in probs]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    s = math.fsum(e)
    return [x / s for x in e]


def nll(rows: Sequence[Row], temperature: float) -> float:
    return -math.fsum(math.log(max(temper(p, temperature)[y], EPS)) for p, y in rows) / len(rows)


def ece(rows: Sequence[Row], temperature: float, bins: int = 10) -> float:
    """Expected calibration error with `bins` equal-width bins on p_max (Kev's ECE10)."""
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, y in rows:
        q = temper(p, temperature)
        pm = max(q)
        buckets[min(bins - 1, int(pm * bins))].append((pm, q.index(pm) == y))
    total = 0.0
    for b in buckets:
        if b:
            conf = math.fsum(c for c, _ in b) / len(b)
            acc = sum(1 for _, ok in b if ok) / len(b)
            total += len(b) / len(rows) * abs(acc - conf)
    return total


def _losses(rows: Sequence[Row]) -> list[float]:
    if not rows:
        raise InvalidInput("cannot fit a temperature without labelled rows")
    return [nll(rows, t) for t in GRID]


def fit_temperature(rows: Sequence[Row]) -> float:
    losses = _losses(rows)
    return GRID[min(range(len(GRID)), key=losses.__getitem__)]   # first minimum on a tie


def flat_fit(rows: Sequence[Row]) -> bool:
    """The NLL is the same at every grid T to within EPS (relative), e.g. every row uniform: `fit_temperature`'s
    T is then its tie-break (GRID[0]), not a fit, and `limit_note` says nothing about it."""
    losses = _losses(rows)
    return max(losses) - min(losses) <= EPS * max(1.0, abs(min(losses)))


def limit_note(temperature: float, at: str = "", *, flat: bool = False) -> str | None:
    """The note for a T on the grid's first or last point (`at` names the points, e.g. " at tiel@0"), else None,
    and None for a flat fit (`flat_fit`), whose T is a tie-break. At the top the best T may lie beyond the grid, so
    the probabilities stay overconfident after calibration. The bottom is worded without that claim: a perfectly
    separated (or 100%-accurate) set drives the fit there however far the grid reaches."""
    if flat:
        return None
    if temperature == GRID[-1]:
        return f"T is at the search limit ({GRID_HI:g}){at}: the probabilities are still overconfident after calibration"
    if temperature == GRID[0]:
        return (f"T is at the search limit ({GRID_LO:g}){at}: the fit wants sharper probabilities than the grid allows "
                f"(common on a small or perfectly separated set)")
    return None


def row_from_answer(question: dict, answer: dict) -> Row:
    """(probabilities in key order, label index) from a labelled question (with `label`) and callo's answer."""
    label = question["label"]
    try:
        if question["type"] == "noul":
            p = float(answer["noul"])
            return [1.0 - p, p], ["false", "true"].index(str(label).lower())
        probs = answer["probabilities"]
        keys = list(probs)
        return [float(probs[k]) for k in keys], keys.index(str(label))
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        raise InvalidInput(f"label {label!r} does not fit the answer {answer!r} "
                           f"({type(e).__name__}: {e})") from e


def request_from_record(record: dict) -> dict:
    """The System One request for a labelled record: `label`, `src` and `_meta` stripped. `images` and
    `context` are kept, so labelled image records calibrate exactly as they are served; `audio` and `videos`
    too."""
    req = {"state": record["state"], "model": record.get("model", "kev-latest"),
           "questions": {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
                         for qid, q in record["questions"].items()}}
    req.update({k: record[k] for k in ("images", "audio", "videos", "context") if k in record})
    return req
