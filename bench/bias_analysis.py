"""The pure half of the v0.3 position-bias acceptance bench: the gate
arithmetic on recorded readouts, the descriptives, the offline judgement of one model's raw files (the fit set and
the pre-registered eval set) and the pre-registered verdict. No
server and no GPU: everything here is unit-tested on fixed readouts (tests/test_bias_analysis.py).
`bench/acceptance_v03_bias.py` runs the live half.

Offline judgement is exact: debiasing a readout served at T = 1 is what serving with the profile gives
(`callo.bias.debias`), so B1 and B2 need only one held-out read per model. Raw files are parsed only by
`callo.calibrate_bias.read_raw`, and B3's CPU half times callo's own serving path (`Engine.bias_for`,
`bounded_readout`).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from bench.bias_refit import builtin_bytes
from callo.bias import (
    Item,
    builtin_set_sha256,
    complete_cells,
    consistent,
    debias,
    labelled_rows,
    offsets_from_prior,
    read_set,
)
from callo.calibrate import ece, fit_temperature, nll
from callo.calibrate_bias import read_raw
from callo.engine import Engine, bounded_readout
from callo.profile import load_profile
from callo.significance import DRAWS, Decision, _cluster_index, bias_stat, decide
from callo.systemone import parse_request

B1_MIN_QUESTIONS = 48
B2_TOLERANCE = 0.01
B3_MAX_MS = 1.0
B3_BODIES = 50
B5_MAX_ACC_DROP = 0.005                       # pre-registered: debiased accuracy may fall by at most 0.5 pp ...
B5_MAX_NLL_RISE = 0.002                       # ... and NLL rise by at most 0.002 (point estimates, T = 1)
# A bound attained exactly passes (the pre-registered ">= -1 pp", "<= +0.002"). Accuracy is compared exactly, in hits;
# the two quantities that are not ratios of counts (B2's interpolated percentile, a mean NLL) get this tolerance,
# far below any real difference (one question in 958 is 1e-3).
BOUND_EPS = 1e-12
RESAMPLES = 2000
B12_SETS = ("eval",)                          # the held-out read B1, B2 and B5 judge (public-pool-v6 test)
# The gates the pre-registered verdict needs per model: B0, B3 and B4 run on TielCoder only.
REQUIRED = {"tiel_kvu": ("b0", "b1", "b2", "b3", "b4", "b5"), "qwen9b": ("b1", "b2", "b5")}


def load_raw(path: str | Path) -> tuple[dict, dict, list[dict]]:
    """(header, readings, checks) of a calibrate --bias raw file, through `read_raw` (the one reader of the format).
    A missing file is empty: a run that refused before its first readout writes none."""
    if not Path(path).exists():
        return {}, {}, []
    raw = read_raw(path)
    return raw.header, raw.readings, raw.checks


def raw_readouts(path: str | Path) -> int:
    return len(load_raw(path)[1])


def items_of(path: str | Path | None) -> list[Item]:
    return read_set(builtin_bytes().decode() if path is None else Path(path).read_text()).items


def offsets_of(profile: dict | None) -> dict[int, tuple[float, ...]]:
    return {int(n): offsets_from_prior(e["prior"]) for n, e in ((profile or {}).get("entries") or {}).items()}


def _paired(a: np.ndarray, b: np.ndarray, clusters: Sequence[str], stat: Callable[[np.ndarray], Any], *,
            resamples: int, seed: int) -> np.ndarray:
    """stat(b) - stat(a) under a cluster bootstrap that draws the SAME clusters for both. Needs at least one cluster:
    the gates check for an empty sample first and report it instead."""
    idx, count = _cluster_index(clusters)
    if not count:
        raise ValueError("a paired bootstrap needs at least one question")
    members = [np.flatnonzero(idx == j) for j in range(count)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(resamples):
        pick = np.concatenate([members[j] for j in rng.integers(0, count, size=count)])
        out.append(np.asarray(stat(b[pick])) - np.asarray(stat(a[pick])))
    return np.array(out)


def _bias(rows: np.ndarray) -> float:
    """max_k |meanP_n[k] - 1/n| over question rows: the significance test's statistic on their mean."""
    return bias_stat(rows.mean(axis=0))


def _question_meanp(items: Sequence[Item], readings: dict, offsets: Sequence[float] | None,
                    n: int) -> tuple[np.ndarray, list[str]]:
    rows, clusters = [], []
    for it in items:
        cells = complete_cells(it, readings) if it.n == n else None
        if cells is not None:
            rows.append(np.mean([debias(p, offsets) for p, _ in cells], axis=0))
            clusters.append(it.cluster)
    return np.array(rows, dtype=float).reshape(len(rows), n), clusters


def held_out_detection(rows_by_n: dict[int, tuple[np.ndarray, list[str]]], *, draws: int = DRAWS,
                       seed: int = 0) -> dict[int, Decision]:
    """Is there a held-out bias to remove at all? The cyclic-shift permutation test (`decide`) on the held-out
    RAW per-question meanP rows, with Holm across the n given (alpha 0.05); `kept` means detectable."""
    return decide(rows_by_n, min_questions=1, draws=draws, seed=seed)


def fitted_direction(offsets: Sequence[float]) -> np.ndarray:
    """The unit vector of (prior_n - 1/n) for the FITTED prior, recovered from its centred-log offsets. It is
    fixed by the fit before the eval set is read. Zero for a uniform prior, which moves nothing and cannot pass."""
    z = np.asarray(offsets, dtype=float)
    p = np.exp(z - z.max())
    u = p / p.sum() - 1.0 / len(p)
    norm = float(np.linalg.norm(u))
    return u / norm if norm > 1e-12 else np.zeros_like(u)


def gate_b1(items: Sequence[Item], readings: dict, offsets_by_n: dict, *, resamples: int = RESAMPLES,
            draws: int = DRAWS, seed: int = 0) -> dict:
    """B1, per n with an entry and at least B1_MIN_QUESTIONS held-out questions.

    - TESTED when the held-out raw meanP bias is detectable (`held_out_detection`, Holm across those n). The gate is
      the SIGNED held-out deviation along the fitted prior's direction, s(X) = (mean row of X - 1/n) . u: a tested n
      PASSES when the paired cluster-bootstrap 95% CI of s(debiased) - s(raw) lies wholly below 0 and |s(debiased)| <
      |s(raw)| (a wrong-direction prior or an overshoot fails). max_k |meanP_n[k] - 1/n| raw and debiased is
      reported as a point check, not gated (its |.| folds resamples at 0, which fails good corrections).
    - The HARM leg: on every other such n, the same detection test runs on the DEBIASED rows (Holm across those n).
      A rejection, with the debiased max-|.| above the raw one, means the correction created a detectable bias where
      none was, and fails B1 (the smaller Holm family alone could flag an n the correction improved).

    `pass`: False on any tested FAIL or any harm; True when at least one n was tested and none failed; else None."""
    per_n: dict[str, dict] = {}
    raw_rows: dict[int, tuple[np.ndarray, list[str]]] = {}
    for n in sorted(offsets_by_n):
        raw, clusters = _question_meanp(items, readings, None, n)
        if raw.shape[0] < max(B1_MIN_QUESTIONS, 1):
            per_n[str(n)] = {"tested": False, "questions": raw.shape[0], "note": "not tested held-out"}
        else:
            raw_rows[n] = (raw, clusters)
    detection = held_out_detection(raw_rows, draws=draws, seed=seed)
    harm_rows: dict[int, tuple[np.ndarray, list[str]]] = {}
    harm_raw: dict[int, np.ndarray] = {}
    for n, (raw, clusters) in raw_rows.items():
        found = detection[n]
        deb, _ = _question_meanp(items, readings, offsets_by_n[n], n)
        if not found.kept:
            harm_rows[n], harm_raw[n] = (deb, clusters), raw
            per_n[str(n)] = {"tested": False, "questions": raw.shape[0], "detect_p_holm": found.p_holm,
                             "note": f"no detectable held-out bias (p_holm {found.p_holm:.2g})"}
            continue
        u = fitted_direction(offsets_by_n[n])

        def signed(x: np.ndarray, u: np.ndarray = u, n: int = n) -> float:
            return float((x.mean(axis=0) - 1.0 / n) @ u)

        lo, hi = np.percentile(_paired(raw, deb, clusters, signed, resamples=resamples, seed=seed), [2.5, 97.5])
        s_raw, s_deb = signed(raw), signed(deb)
        per_n[str(n)] = {"tested": True, "questions": raw.shape[0], "detect_p_holm": found.p_holm,
                         "signed": [s_raw, s_deb], "signed_diff_ci": [float(lo), float(hi)],
                         "raw": _bias(raw), "debiased": _bias(deb),
                         "pass": bool(hi < 0 and abs(s_deb) < abs(s_raw))}
    for n, h in held_out_detection(harm_rows, draws=draws, seed=seed).items():
        before, after = _bias(harm_raw[n]), _bias(harm_rows[n][0])
        per_n[str(n)].update(harm=bool(h.kept and after > before), harm_p_holm=h.p_holm,
                             harm_max_abs=[before, after])
    per_n = dict(sorted(per_n.items(), key=lambda kv: int(kv[0])))
    tested = [v for v in per_n.values() if v["tested"]]
    harm = [n for n, v in per_n.items() if v.get("harm")]
    verdict = False if harm or any(not v["pass"] for v in tested) else (True if tested else None)
    return {"per_n": per_n, "harm": harm, "pass": verdict}


def _min_margin(cells: Sequence[tuple[Sequence[float], int]], offsets: Sequence[float] | None) -> float:
    """The smallest top-2 gap over a question's rotations: how close its nearest tie is."""
    gaps = []
    for p, _ in cells:
        top = sorted(debias(p, offsets))
        gaps.append(top[-1] - top[-2])
    return float(min(gaps))


def gate_b2(items: Sequence[Item], readings: dict, offsets_by_n: dict, *, resamples: int = RESAMPLES,
            seed: int = 0) -> dict:
    """B2: held-out rotation consistency over every question, debiased vs raw; PASS when the lower end of the paired
    CI of (debiased - raw) is at least -1 pp. `flips` (descriptive, never gated) lists every question the correction
    made inconsistent ("broke") or consistent ("mended") with its smallest top-2 margin raw and debiased, so a FAIL
    reads as "k near-ties broke" or as a broad drop."""
    raw, deb, clusters, flips = [], [], [], []
    for it in items:
        cells = complete_cells(it, readings)
        if cells is not None:
            offs = offsets_by_n.get(it.n)
            before, after = consistent(cells, None), consistent(cells, offs)
            raw.append(float(before))
            deb.append(float(after))
            clusters.append(it.cluster)
            if before != after:
                flips.append({"q": it.uid, "n": it.n, "source": it.source, "flip": "broke" if before else "mended",
                              "margin": [_min_margin(cells, None), _min_margin(cells, offs)]})
    if not raw:
        return {"questions": 0, "pass": None, "error": "no held-out question was read under every rotation"}
    a, b = np.array(raw)[:, None], np.array(deb)[:, None]
    lo, hi = np.percentile(_paired(a, b, clusters, lambda x: float(x.mean()), resamples=resamples, seed=seed),
                           [2.5, 97.5])
    return {"questions": len(raw), "raw": float(a.mean()), "debiased": float(b.mean()),
            "diff_ci": [float(lo), float(hi)], "pass": bool(lo >= -B2_TOLERANCE - BOUND_EPS), "flips": flips}


def floored_rates(items: Sequence[Item], readings: dict) -> dict:
    """The floored-option rate per (source, n): questions with a floored option in any rotation (descriptive; they are
    fitted, not excluded, hence `floored`, not an earlier `excluded`)."""
    total: Counter = Counter()
    floored: Counter = Counter()
    for it in items:
        cells = complete_cells(it, readings)
        if cells is None:
            continue
        total[(it.source, it.n)] += 1
        floored[(it.source, it.n)] += any(m for _, m in cells)
    out: dict[str, dict] = {}
    for (src, n), k in sorted(total.items()):
        out.setdefault(src, {})[str(n)] = {"questions": k, "floored": floored[(src, n)], "rate": floored[(src, n)] / k}
    return out


def _metrics(rows: list, t: float) -> dict:
    hits = sum(max(range(len(p)), key=p.__getitem__) == y for p, y in rows)
    return {"accuracy": hits / len(rows), "hits": hits, "nll": nll(rows, t), "ece10": ece(rows, t),
            "rows": len(rows)}


def labelled(items: Sequence[Item], readings: dict, offsets_by_n: dict, *, dev_items: Sequence[Item],
             dev_readings: dict) -> dict:
    """Held-out accuracy / NLL / ECE10, raw vs debiased, at T = 1 and at a T refitted on dev (descriptive)."""
    out: dict[str, dict] = {"T1": {}, "Tdev": {}}
    for name, offs in (("raw", {}), ("debiased", offsets_by_n)):
        rows = labelled_rows(items, readings, offs)
        dev = labelled_rows(dev_items, dev_readings, offs)
        if not rows:
            continue
        out["T1"][name] = _metrics(rows, 1.0)
        if dev:
            t = fit_temperature(dev)
            out["Tdev"][name] = {**_metrics(rows, t), "T": t}
    return out


def gate_b5(lab: dict) -> dict:
    """B5 (no harm): on the labelled held-out rows the correction touches (n with an entry; the caller
    restricts them) at T = 1, debiased accuracy is at most B5_MAX_ACC_DROP below raw and debiased NLL at most
    B5_MAX_NLL_RISE above it (point estimates). A bound attained exactly passes: the accuracy leg is compared in
    hits as exact rationals (as floats, -4/800 is below -0.005), the NLL leg with BOUND_EPS."""
    t1 = lab.get("T1") or {}
    if "raw" not in t1 or "debiased" not in t1:
        return {"pass": None, "error": "no labelled held-out rows"}
    raw, deb = t1["raw"], t1["debiased"]
    d_acc = deb["accuracy"] - raw["accuracy"]
    d_nll = deb["nll"] - raw["nll"]
    exact = Fraction(deb["hits"], deb["rows"]) - Fraction(raw["hits"], raw["rows"])
    return {"d_accuracy": d_acc, "d_nll": d_nll, "rows": raw["rows"],
            "pass": bool(exact >= -Fraction(str(B5_MAX_ACC_DROP)) and d_nll <= B5_MAX_NLL_RISE + BOUND_EPS)}


VACUOUS_OK = ("b1", "b5")    # pass None is judged only here: B1 tested no n, B5 had no corrected n


def prereg_verdict(models: dict) -> dict:
    """The pre-registered decision, in this order:

    - FALSIFIED: any required gate FAILs, in any model, also one that later errored (a measured FAIL is never lost;
      B1's harm leg is part of B1);
    - ERROR: a required gate could not be judged (missing, an `error`, or `pass` None outside VACUOUS_OK, so a B3
      "not run" is ERROR), or a model did not complete;
    - CONFIRMED: at least one (model, n) was B1-tested and passed, with every required gate on every model judged
      and none failed;
    - INCONCLUSIVE: otherwise (no falsifier, no error, no tested n).

    B1 is read from its `pass` alone, not re-derived from `per_n` and `harm`: `gate_b1` always
    writes them consistently (`pass` False on any tested FAIL or harm), so only a hand-edited summary.json could
    disagree, and that is a deviation `--override-final` records, not a case this function second-guesses."""
    reasons: list[str] = []
    tested: list[str] = []
    falsified = errored = False
    for name, gates in REQUIRED.items():
        m = models.get(name)
        if m is None:
            errored = True
            reasons.append(f"{name}: not run")
            continue
        if "error" in m:
            errored = True
            reasons.append(f"{name}: {m['error']}")
        for g in gates:
            v = m.get(g)
            if v is not None and v.get("pass") is False:
                falsified = True
                reasons.append(f"{name} {g.upper()} FAIL")
            elif "error" in m:
                continue                # its unjudged gates are the model's error, already counted
            elif v is None or v.get("error") or (v.get("pass") is None and g not in VACUOUS_OK):
                errored = True
                reasons.append(f"{name} {g.upper()}: {(v or {}).get('error') or (v or {}).get('note') or 'not run'}")
        tested += [f"{name} n={n}" for n, x in ((m.get("b1") or {}).get("per_n") or {}).items() if x.get("tested")]
    verdict = ("FALSIFIED" if falsified else "ERROR" if errored else "INCONCLUSIVE" if not tested else "CONFIRMED")
    return {"verdict": verdict, "b1_tested": tested, "reasons": reasons}


def b3_cpu(reps: int = 20000) -> dict:
    """B3's CPU half: the added cost per readout of serving with a profile, timed on callo's own serving path: the
    eligibility lookup `Engine.bias_for` and `bounded_readout` with its offsets, against the same two calls on an
    engine without a profile. `bias_for` reads only the profile, so the engine needs no upstream."""
    marg = {"A": -0.5, "B": -1.2, "C": -2.0, "D": float("-inf")}
    top = {0: -0.5, 1: -1.2, 2: -2.0, 9: -6.0}
    pb = load_profile({"callo_position_bias": 2, "model_key": "sha256:" + "0" * 64, "set": {"kinds": ["plain"]},
                       "entries": {"4": {"prior": [0.4, 0.3, 0.2, 0.1]}}})
    q = parse_request({"state": "s", "questions": {"q": {"type": "choice", "criteria": {
        "w1": None, "x1": None, "y1": None, "z1": None}}}}).questions[0]          # canonical: A, B, C, D
    bare = Engine(None, None, None)                                                # type: ignore[arg-type]
    served = Engine(None, None, None, position_bias=pb)                           # type: ignore[arg-type]

    def timed(engine: Engine) -> float:
        t0 = time.perf_counter()
        for _ in range(reps):
            offsets, _ = engine.bias_for(q, "plain")
            bounded_readout(marg, top, q.labels, 1.0, offsets)
        return time.perf_counter() - t0

    base = timed(bare)
    added = max(0.0, (timed(served) - base) / reps * 1000)
    return {"reps": reps, "added_ms": added, "pass": added < B3_MAX_MS}


def b3_bodies(items: Sequence[Item], entries: Iterable[str | int], k: int = B3_BODIES) -> list[dict]:
    """B3's request bodies (rotation 0), spread over the n the profile kept: ceil(k / len(kept)) per kept n, in file
    order (the held-out file is sorted by n, so its first k rows are all n = 2). Without entries
    they spread over every n; the bench then runs B3 with a synthetic profile that has an entry for each of them
    (`acceptance_v03_bias.synthetic_profile`)."""
    kept = sorted({int(n) for n in entries}) or sorted({it.n for it in items})
    per = math.ceil(k / len(kept)) if kept else 0
    out: list[dict] = []
    for n in kept:
        out += [it.body(0) for it in items if it.n == n][:per]
    return out


def set_error(raw_path: Path, data: str | Path | None) -> str | None:
    """Why the raw file at `raw_path` cannot be read against the set `data` (None: the built-in set): its header names
    another set by sha256 (as `bench.bias_refit.refit` checks). Readings are keyed by line-numbered uids, so readouts
    of another set would pair with the wrong questions silently. None for a missing file or a match."""
    if not raw_path.exists():
        return None
    got = (load_raw(raw_path)[0].get("set") or {}).get("sha256")
    want = builtin_set_sha256() if data is None else hashlib.sha256(Path(data).read_bytes()).hexdigest()
    return None if got == want else f"{raw_path.name} was measured on set sha256 {got}, not {want}"


def reads_error(d: Path, reads: dict, sets: dict[str, str | Path | None] | None = None) -> str | None:
    """Why B1, B2 and B5 cannot be judged: a held-out read that did not run, exited non-zero after its
    one resume, or left no raw file. Judging partial data would silently drop the unread questions, and with them
    perhaps a whole n ("not tested held-out" with no flag). With `sets`, also a raw file measured on another set."""
    bad, other = [], []
    for name in B12_SETS:
        r = reads.get(name)
        if r is None:
            bad.append(f"{name} not run")
        elif r.get("rc") != 0:
            bad.append(f"{name} exited {r.get('rc')} with {r.get('readouts')} readouts")
        elif not (d / f"{name}.raw.jsonl").exists():
            bad.append(f"{name} left no raw file")
        elif sets is not None and (e := set_error(d / f"{name}.raw.jsonl", sets[name])):
            other.append(e)
    why = (["not judged on partial held-out data: " + "; ".join(bad)] if bad else []) + (
        ["not judged on another set's readouts: " + "; ".join(other)] if other else [])
    return "; ".join(why) or None


def analyse(d: Path, sets: dict[str, str | Path | None], reads: dict) -> dict:
    """Offline judgement of one model's raw files in `d`: B1, B2, B5 and the descriptives. `sets` maps "fit" and "eval"
    to their --data; `reads` is the bench's {"rc", "readouts", "attempts"} per held-out run ("eval")."""
    prof = json.loads((d / "fit.json").read_text()) if (d / "fit.json").exists() else None
    offs = offsets_of(prof)
    out: dict[str, Any] = {"entries": sorted((prof or {}).get("entries") or {}, key=int),
                           "dropped": (prof or {}).get("dropped"), "floored_rates": {}}
    fit_err = set_error(d / "fit.raw.jsonl", sets["fit"])
    if fit_err:
        out["fit_error"] = "fit descriptives not computed: " + fit_err
    else:
        out["floored_rates"]["fit"] = floored_rates(items_of(sets["fit"]), load_raw(d / "fit.raw.jsonl")[1])
    err = reads_error(d, reads, sets)
    if err is not None:
        out["b1"], out["b2"], out["b5"] = ({"pass": None, "error": err} for _ in range(3))
        return out
    items, readings = items_of(sets["eval"]), load_raw(d / "eval.raw.jsonl")[1]
    corrected = [it for it in items if it.n in offs]          # B5 judges the rows the correction touches
    out.update(b1=gate_b1(items, readings, offs), b2=gate_b2(items, readings, offs),
               labelled=labelled(items, readings, offs, dev_items=[], dev_readings={}),
               labelled_corrected=labelled(corrected, readings, offs, dev_items=[], dev_readings={}))
    out["b5"] = (gate_b5(out["labelled_corrected"]) if offs else
                 {"pass": None, "note": "no n has an entry: nothing is corrected"})
    out["floored_rates"]["eval"] = floored_rates(items, readings)
    return out
