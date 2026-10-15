"""The v0.3 think-deadline acceptance bench: the gate calibration runs (Tiel, Tiel greedy, qwen9b), the served
sessions on Tiel, and the gates G1 (adherence, single-stream), G3 (the cut readout equals the cut), G4 (the chunk cost),
G5 (the static mapping predicts serving, greedy), G6 (the deadline point against serving, descriptive, with the served
latency beside the point's) and C1 (the convergence measurement and the early-exit build decision, by a
pre-registered rule, `BUILD_RULE`). G2 is an agent pilot's: it is computed outside this bench from the pilot's
request log and recorded here by `--g2 FILE`.

callo is measured as shipped: the calibrations run with calibrate's default `--cache` (on)
and the served sessions launch with `cache=True`, so G5 and G6 compare a calibration and a serve with one cache setting.
Two documented deviations from the gates' original wording (the doc's header names them): each served
session first sends WARMUP_N discarded warm-up requests, and the served sessions repeat the calibration launch's
llama-server passthrough (`--cache-ram 0`), so G5 and G6 compare one launch shape.

GPU, llama.cpp-master @ 4ebdf2c, one model server at a time (the bench refuses to start while one holds the GPU, and
re-checks before each GPU part). Raw output goes to `--out` (OUT_DEFAULT), the doc to
`docs/bench/v0.3-deadline.md` by default (`--doc`; `bench/report_v03_deadline.py` renders it). The gate functions and the session drivers are unit-tested on fakes; `main` runs the
parts in order (`calib`, `serve`, `gates`), each rerunnable, and is stopped only through `kill -TERM $(cat DIR/run.pid)`.
A gate never PASSes on missing evidence: it is ERROR, naming what is missing.
"""
from __future__ import annotations

import argparse
import http.client
import json
import logging
import math
import os
import signal
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench import acceptance_v03_think as v03t
from bench import lib
from bench.acceptance import MODEL as QWEN9B
from bench.acceptance_v02 import TIEL, _install_stop_handlers
from bench.acceptance_v03_think import (
    BUDGETS,
    CTX,
    EXPECTED_BUILD,
    LLAMA_SERVER,
    PARALLEL,
    SEED,
    TRANSFER_TEST,
    _calibrate,
    _foreign_servers,
    _join_flags,
    _pid_in,
    _start,
    pid_owner,
    profile_launch,
    select_records,
)
from bench.report_v03_deadline import REGIMES, render, verdict_lines
from bench.report_v03_think import version_line
from callo.calibrate_think import Item, load_data
from callo.deadline import percentile
from callo.think import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "bench" / "v0.3-deadline.md"
OUT_DEFAULT = lib.env_path(lib.OUT_ENV, "callo_v0.3", "deadline")
SUBSET_N, G3_N = 200, 48                        # the greedy subset (G1, G3, G5, G6) and G3's questions
WARMUP_N = 3                                    # each served session's first requests, discarded (a cold first request
#                                                 measures llama-server's warm-up, not adherence)
G1_DEADLINES, G3_DEADLINE, G5_DEADLINE = (1000, 2000), 1500, 2000
G1_MIN, G1_OVER_P99 = 60, 50.0                  # >= 60 deadline/budget closes per D; p99(elapsed - D) <= 50 ms
G3_DP, G3_SHARE = 1e-3, 0.99                    # the think bench's T1 bounds; prompt_n bounds on >= 99 %
G3_MIN = 24                                     # usable pairs G3 needs (the think bench's T1_MIN_CUTS); fewer is not testable
G4_OVERHEAD, G4_RATE = 15.0, 0.90               # ô p50 <= 15 ms; think tokens/s >= 90 % of the budget-only rate
G5_NATURAL, G5_K = 0.03, 0.05                   # natural share within 3 points; k p50 within 5 %
TIEL_CONVERGE_K, QWEN_CONVERGE_K = 32, 64       # Tiel's n_gen p50 is 110, qwen9b thinks long
CUT_CLOSES = ("deadline", "budget")
CALIB = {                                       # the gate calibration runs: (model, data, extra calibrate flags)
    "tiel": (TIEL, "test", ["--think-budgets", BUDGETS, "--deadlines", "1000,2000", "--converge-probe",
                            str(TIEL_CONVERGE_K), "--audit", "24"]),
    "tiel-greedy": (TIEL, "subset", ["--think-budgets", BUDGETS, "--think-sampling", "temp=0", "--deadlines", "2000",
                                     "--audit", "0"]),
    "qwen9b": (QWEN9B, "test", ["--think-budgets", BUDGETS, "--converge-probe", str(QWEN_CONVERGE_K), "--audit", "0"]),
}
PARTS = ("calib", "serve", "gates")
log = logging.getLogger("bench")


# ---- served answers and their correctness ------------------------------------------------------------------------

def served_correct(item: Item, answer: Mapping[str, Any] | None) -> bool | None:
    """Whether a served answer's argmax is the item's label, a tie going to the first option as calibrate's accuracy
    (None for a missing or unreadable answer)."""
    if not isinstance(answer, Mapping):
        return None
    if answer.get("type") == "noul" and isinstance(answer.get("noul"), (int, float)):
        return (1 if float(answer["noul"]) > 0.5 else 0) == item.label
    keys = list(item.question.keys)
    probs = answer.get("probabilities")
    if not isinstance(probs, Mapping) or not all(k in probs for k in keys):
        return None
    return max(range(len(keys)), key=lambda i: float(probs[keys[i]])) == item.label


def _post(url: str, body: dict) -> tuple[int, dict, Any]:
    """`lib.post`; (-1, {}, None) when the request itself failed (a dead server is the gate's missing evidence)."""
    try:
        return lib.post(url, body)
    except (OSError, http.client.HTTPException, ValueError) as e:
        log.warning("request failed: %s: %s", type(e).__name__, e)
        return -1, {}, None


def serve_items(url: str, items: Sequence[Item], *, deadline_ms: int | None, budget: int | None = None) -> list[dict]:
    """One /v1/systemone request per item, sequentially (single-stream), with `think_deadline_ms` / `think_budget`
    when given (None: the field is not sent); per item its status, its X-Callo-Timings entry, whether the served
    answer is right and the client's wall time (G1's elapsed for a 504, which carries no X-Callo-Timings)."""
    rows = []
    for it in items:
        body = dict(it.body)
        if deadline_ms is not None:
            body["think_deadline_ms"] = deadline_ms
        if budget is not None:
            body["think_budget"] = budget
        t0 = time.monotonic()
        status, headers, out = _post(f"{url}/v1/systemone", body)
        wall_ms = 1000.0 * (time.monotonic() - t0)
        try:
            entries = json.loads(headers.get("x-callo-timings") or "[]") if status == 200 else []
        except ValueError:
            entries = []
        answer = (out.get("answers") or {}).get(it.qid) if status == 200 and isinstance(out, dict) else None
        rows.append({"q": it.uid, "status": status, "entry": entries[0] if entries and isinstance(entries[0], dict)
                     else None, "correct": served_correct(it, answer), "wall_ms": round(wall_ms, 1)})
    return rows


def g3_pairs(url: str, items: Sequence[Item], *, cap: int, deadline_ms: int = G3_DEADLINE) -> list[dict]:
    """G3's two probes per item on the greedy profile's session (the probe traces with the served profile's
    sampling, temp=0): its live mode at the deadline, then its budgets mode at B = the live cut k. Per item the live
    trace, chunks and readout, and the budget trace (first k tokens) and readout."""
    out = []
    for it in items:
        st, _, live = _post(f"{url}/callo/think-probe", {"request": it.body, "budgets": [cap], "seed": SEED,
                                                         "live": {"deadline_ms": deadline_ms, "probe_every": None}})
        lv = ((live or {}).get("questions") or {}).get(it.qid, {}).get("live") if st == 200 else None
        if not lv or lv.get("close") == "budget0" or int(lv.get("n_gen") or 0) < 1:
            out.append({"q": it.uid, "live": lv, "budget": None})
            continue
        k = int(lv["n_gen"])
        st, _, bud = _post(f"{url}/callo/think-probe", {"request": it.body, "budgets": [k], "seed": SEED})
        q = ((bud or {}).get("questions") or {}).get(it.qid) if st == 200 else None
        ok = isinstance(q, dict) and isinstance(q.get("trace"), dict) and str(k) in (q.get("budgets") or {})
        out.append({"q": it.uid, "live": lv, "budget": {"tokens": q["trace"]["tokens"][:k], "readout": q["budgets"][str(k)]}
                    if ok else None})
    return out


# ---- the gates ----------------------------------------------------------------------------------------------------

def _cut(rows: Sequence[Mapping]) -> list[Mapping]:
    return [r["entry"] for r in rows if r.get("entry") and r["entry"].get("think_close") in CUT_CLOSES]


def _failed(rows: Sequence[Mapping], *, but: tuple[int, ...] = ()) -> int:
    """The requests that got no 200 (-1: the request itself failed), the statuses in `but` excepted."""
    return sum(1 for r in rows if r.get("status") != 200 and r.get("status") not in but)


def g1(sessions: Mapping[int, Sequence[Mapping]]) -> dict:
    """G1: per D, over the `deadline` / `budget` closes only (>= 60), p99(elapsed_ms - D) <= 50 ms, the max reported;
    no budget-0 fallback at 2,000 ms. A 504 (ThinkTimeout: a chunk still running 5 s past D) is a cut over D, at the
    client's measured wall time, else +inf, so a timed-out tail can never pass; any other failed request is the D's
    ERROR (`failed`). A measured FAIL at either D stands over the other's ERROR."""
    per: dict[str, dict] = {}
    errors = []
    for d in G1_DEADLINES:
        rows = sessions.get(d)
        if rows is None:
            errors.append(f"no session at {d} ms")
            continue
        over = [e["elapsed_ms"] - d for e in _cut(rows) if e.get("elapsed_ms") is not None]
        timeouts = [r for r in rows if r.get("status") == 504]
        for r in timeouts:
            el = (r.get("entry") or {}).get("elapsed_ms") or r.get("wall_ms")
            over.append(float(el) - d if el is not None else math.inf)
        failed = _failed(rows, but=(504,))
        fallbacks = sum(1 for r in rows if (r.get("entry") or {}).get("deadline_fallback") == "budget0")
        entry: dict[str, Any] = {"n_cut": len(over), "fallbacks": fallbacks, "n": len(rows), "timeouts": len(timeouts),
                                 "failed": failed, "p99_over_ms": percentile(over, 0.99) if over else None,
                                 "max_over_ms": max(over) if over else None}
        if len(over) < G1_MIN:
            entry["error"] = f"not testable: {len(over)} deadline/budget closes (< {G1_MIN})"
        else:
            entry["pass"] = entry["p99_over_ms"] <= G1_OVER_P99 and (d != 2000 or fallbacks == 0)
        if failed:
            entry["error"] = "; ".join(x for x in (entry.get("error"), f"{failed} requests failed") if x)
        per[str(d)] = entry
    passes = [e.get("pass") for e in per.values()]
    ok = False if False in passes else None if errors or any("error" in e for e in per.values()) else True
    return {"pass": ok, **({"error": "; ".join(errors)} if errors else {}), "deadlines": per}


def g3(pairs: Sequence[Mapping]) -> dict:
    """G3: token arrays equal on >= n - 2; on the m identical, argmax agreement >= m - 1 and median |Δp| <= 1e-3 (the
    largest per question); the live readout's first-pass prompt_n <= 4 on >= 99 % and every continuation's
    prompt_n = 1 on >= 99 %. A failed probe (no live reply, or a live trace that thought and has no budget readout) is
    the gate's ERROR; fewer than G3_MIN usable pairs is not testable."""
    usable = [p for p in pairs if p.get("live") and p.get("budget")]
    broken = sum(1 for p in pairs if not p.get("live") or (p["live"].get("close") != "budget0" and not p.get("budget")))
    if len(usable) < G3_MIN:
        return {"pass": None, "n": len(usable), "error": f"not testable: {len(usable)} questions gave a live cut and "
                f"its budget readout (< {G3_MIN})" + (f"; {broken} probe(s) failed" if broken else "")}
    same = [p for p in usable if list(p["live"]["tokens"]) == list(p["budget"]["tokens"])]
    agree, dps = 0, []
    for p in same:
        a, b = p["live"]["readout"]["probs"], p["budget"]["readout"]["probs"]
        agree += max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
        dps.append(max(abs(x - y) for x, y in zip(a, b, strict=True)))
    first = [p["live"]["readout"].get("read_prompt_n") for p in usable]
    conts = [c.get("prompt_n") for p in usable for c in (p["live"].get("chunks") or [])[1:]]
    first_ok = sum(1 for x in first if x is not None and x <= 4) / len(first)
    cont_ok = sum(1 for x in conts if x == 1) / len(conts) if conts else 1.0
    n, m = len(usable), len(same)
    dp_median = statistics.median(dps) if dps else None
    ok = (m >= n - 2 and agree >= m - 1 and dp_median is not None and dp_median <= G3_DP and first_ok >= G3_SHARE
          and cont_ok >= G3_SHARE)
    return {"pass": ok, "n": n, "identical": m, "argmax_agree": agree, "dp_median": dp_median,
            "first_pass_ok": first_ok, "continuation_ok": cont_ok, "continuations": len(conts), "broken": broken,
            **({"error": f"{broken} probe(s) failed"} if broken else {})}


def g4(stats_deadline: Mapping[str, Any] | None, deadline_rows: Sequence[Mapping], budget_rows: Sequence[Mapping],
       prefill_ms_per_token: float | None) -> dict:
    """G4: ô p50 <= 15 ms (`/callo/stats.think.deadline.chunk_overhead_ms`), and the think tokens per second of think
    time (gen_ms less chunk 0's prefill, gen_prompt_n x the profile's prefill rate) in deadline-cut traces >= 90 % of
    the budget-only session's. A measured overhead above 15 ms FAILs even when a rate is missing."""
    def rate(entries: Sequence[Mapping]) -> float | None:
        if not entries or prefill_ms_per_token is None:
            return None
        toks = sum(e["think_tokens"] for e in entries)
        ms = sum(max(1e-9, e["gen_ms"] - (e.get("gen_prompt_n") or 0) * prefill_ms_per_token) for e in entries)
        return 1000.0 * toks / ms if ms > 0 else None

    overhead = ((stats_deadline or {}).get("chunk_overhead_ms") or {}).get("p50")
    cut = [e for e in _cut(deadline_rows) if e.get("think_close") == "deadline"]
    base = [r["entry"] for r in budget_rows if r.get("entry") and r["entry"].get("think_tokens")]
    r_cut, r_base = rate(cut), rate(base)
    if overhead is None or r_cut is None or r_base is None:
        missing = [n for n, v in (("chunk_overhead_ms", overhead), ("the profile's prefill rate", prefill_ms_per_token),
                                  ("deadline cuts", cut), ("budget-only traces", base)) if not v and v != 0.0]
        return {"pass": False if overhead is not None and overhead > G4_OVERHEAD else None, "overhead_p50_ms": overhead,
                "error": "missing: " + ", ".join(missing or ["a think rate"])}
    return {"pass": overhead <= G4_OVERHEAD and r_cut >= G4_RATE * r_base, "overhead_p50_ms": overhead,
            "rate_cut_tok_s": r_cut, "rate_budget_tok_s": r_base, "ratio": r_cut / r_base, "n_cut": len(cut),
            "n_budget": len(base)}


def _served_close(e: Mapping[str, Any]) -> str:
    """A served entry's close as a deadline point counts it (over every question): a fallback is `budget0`,
    a question read without a trace (one option) `natural`."""
    if e.get("deadline_fallback"):
        return "budget0"
    return str(e.get("think_close") or "natural")


def g5(profile: Mapping[str, Any], rows: Sequence[Mapping], deadline_ms: int = G5_DEADLINE) -> dict:
    """G5 (greedy): the greedy calibration's exact deadline point at D against the same questions served greedy at D:
    the natural-close share (natural + eos, over every question, as the point counts it) within 3 points and the cut
    k p50 (nearest rank, as the point's) within 5 %. Any served request that failed is the gate's ERROR."""
    point = (profile.get("deadlines") or {}).get(str(deadline_ms))
    failed = _failed(rows)
    if failed:
        return {"pass": None, "error": f"{failed} requests failed", "failed": failed}
    served = [r["entry"] for r in rows if r.get("status") == 200 and r.get("entry")]
    if not point or not served:
        return {"pass": None, "error": f"no deadline point at {deadline_ms} ms" if not point
                else "no served readouts"}
    closes = [_served_close(e) for e in served]
    nat_served = sum(1 for c in closes if c in ("natural", "eos")) / len(served)
    pc = point.get("closes") or {}
    nat_point = float(pc.get("natural", 0.0)) + float(pc.get("eos", 0.0))
    ks = [e["think_tokens"] for e in served if e.get("think_close") == "deadline"]
    k_served, k_point = (percentile(ks, 0.5) if ks else None), point.get("k_p50")
    k_ok = (k_served is None and k_point is None) or (
        k_served is not None and bool(k_point) and abs(k_served - k_point) <= G5_K * k_point)
    return {"pass": abs(nat_served - nat_point) <= G5_NATURAL and bool(k_ok), "natural_served": nat_served,
            "natural_point": nat_point, "k_p50_served": k_served, "k_p50_point": k_point, "n": len(served),
            "closes_served": {c: closes.count(c) for c in sorted(set(closes))}}


def g6(profile: Mapping[str, Any], rows: Sequence[Mapping], deadline_ms: int = G5_DEADLINE) -> dict:
    """G6 (descriptive): the deadline point's per-question correctness (`rows[].d["<D>"]`, the profile's own
    `row_fields`) against the same questions served at D with the model's sampler: the paired Δacc (served - point)
    with its cluster-bootstrap CI. Beside it the served `elapsed_ms` p50/p95 against the point's
    `lat_ms` p50/p95 over the same questions and the point's own `lat_p50_ms`/`lat_p95_ms`: calibrate keeps the deadline
    rows on the budgets' clock, said to understate deadline latency by 1-4 %; this line checks it. Descriptive only."""
    fields = list(profile.get("row_fields") or ["correct"])
    at, lat_at = fields.index("correct"), (fields.index("lat_ms") if "lat_ms" in fields else None)
    point = {r["q"]: r for r in profile.get("rows") or [] if str(deadline_ms) in (r.get("d") or {})}
    used = [(point[r["q"]], r) for r in rows if r.get("correct") is not None and r["q"] in point]
    if not used:
        return {"pass": None, "error": f"no question has both a {deadline_ms} ms deadline row and a served answer"}
    d = paired_bootstrap([float(bool(p["d"][str(deadline_ms)][at])) for p, _ in used],
                         [float(r["correct"]) for _, r in used],
                         [str(p.get("cluster") or p["q"]) for p, _ in used], seed=SEED)
    timed = [(p, r) for p, r in used if (r.get("entry") or {}).get("elapsed_ms") is not None]
    served = [float(r["entry"]["elapsed_ms"]) for _, r in timed]
    same = [float(p["d"][str(deadline_ms)][lat_at]) for p, _ in timed] if lat_at is not None else []
    dp = (profile.get("deadlines") or {}).get(str(deadline_ms)) or {}

    def head(v: Any) -> float | None:
        return float(v[0]) if isinstance(v, (list, tuple)) and v else float(v) if isinstance(v, (int, float)) else None

    lat: dict[str, Any] = {"n": len(timed), "point_p50_ms": head(dp.get("lat_p50_ms")),
                           "point_p95_ms": head(dp.get("lat_p95_ms"))}
    for q, name in ((0.5, "p50"), (0.95, "p95")):
        lat[f"served_{name}_ms"] = percentile(served, q) if served else None
        lat[f"point_same_{name}_ms"] = percentile(same, q) if same else None
        s, p = lat[f"served_{name}_ms"], lat[f"point_same_{name}_ms"]
        lat[f"served_vs_point_{name}"] = s / p - 1 if s is not None and p else None
    return {"pass": None, "descriptive": True, "n": len(used), "dacc": list(d), "latency": lat}


# The early-exit build rule, PRE-REGISTERED before any data. A
# measured model's own gain decides it; otherwise early exit must stay on the front at every grid probe cost from the
# cheap end up (monotone, `rho_prefix`), slow decode (rho 0.5) counting: an isolated gain at a higher rho is not enough.
BUILD_RULE = ("build = early exit gains on >= 1 measured model at its measured probe cost, OR its gains hold "
              "consistently at every grid ρ from 0.5 up to some ρ on >= 1 model (rho_prefix >= 0.5: monotone from the "
              "cheap end; slow decode, ρ 0.5, counts) (pre-registered on 2026-10-09, before any data)")


def rho_terms(rho: Mapping[str, Any] | None) -> dict:
    """A converge block's rho map read for the build rule: `by_rho` as (rho, gains) sorted NUMERICALLY (its keys are
    ':g' strings, and "16" sorts before "2" as text), `rho_max`, and `rho_prefix` (the largest grid rho such that some
    z gains at every grid rho from the cheapest up to it): the profile's own when it has the field (newer
    profiles), else derived from `by_rho`, else None. `consistent` is the rule's second half (None: no rho map)."""
    if not isinstance(rho, Mapping):
        return {"grid": [], "rho_max": None, "rho_prefix": None, "prefix_source": None, "consistent": None}
    grid = sorted(((float(r), bool(zs)) for r, zs in (rho.get("by_rho") or {}).items()), key=lambda t: t[0])
    rho_max = rho.get("rho_max")
    if rho_max is None and grid:
        rho_max = max((r for r, ok in grid if ok), default=None)
    if "rho_prefix" in rho:
        prefix, source = rho["rho_prefix"], "profile"
    elif grid:
        prefix, source = None, "derived"
        for r, ok in grid:
            if not ok:
                break
            prefix = r
    else:
        return {"grid": grid, "rho_max": rho_max, "rho_prefix": None, "prefix_source": None, "consistent": None}
    cheapest_ok = not grid or grid[0][0] <= REGIMES[0][1]              # the walk starts at slow decode or cheaper
    return {"grid": grid, "rho_max": None if rho_max is None else float(rho_max),
            "rho_prefix": None if prefix is None else float(prefix), "prefix_source": source,
            "consistent": prefix is not None and cheapest_ok}


def _regime(terms: Mapping[str, Any], r: float) -> str:
    """One regime cell: `yes` (within the prefix: gains at every grid rho up to it), `isolated` (gains at r, but some
    cheaper rho does not), `no`, `n/a` (no rho map)."""
    if terms["prefix_source"] is None:
        return "n/a"
    if terms["rho_prefix"] is not None and r <= terms["rho_prefix"]:
        return "yes"
    return "isolated" if any(g == r and ok for g, ok in terms["grid"]) else "no"


def c1(profiles: Mapping[str, Mapping[str, Any]]) -> dict:
    """C1 (descriptive) and the early-exit build decision by `BUILD_RULE`: per model the convergence block's headline (flips,
    the sequence verdicts per z, the cross-fitted rules, c with restore_share, the EE points' gain verdicts at the
    measured c, rho_max, rho_prefix and K) and the regime table (`REGIMES`). `build` is True as soon as either half
    holds on any model; False only when every model was measured and neither does; None (with `error`) otherwise."""
    out: dict[str, Any] = {}
    halves: list[bool | None] = []
    for name, p in profiles.items():
        cv = p.get("converge") if isinstance(p, Mapping) else None
        if not isinstance(cv, Mapping):
            out[name] = {"error": "no converge block (calibrate without --converge-probe?)"}
            continue
        measured = (cv.get("cost") or {}).get("c_ms") is not None
        gains = any(bool((v or {}).get("gains")) for v in (cv.get("gain") or {}).values()) if measured else None
        terms = rho_terms(cv.get("rho"))
        halves += [gains, terms["consistent"]]
        out[name] = {"K": cv.get("K"), "n": cv.get("n"), "cost": cv.get("cost"), "flips": cv.get("flips"),
                     "sequence": {z: [v["verdict"] for v in seq] for z, seq in (cv.get("sequence") or {}).items()},
                     "rules": cv.get("rules"), "gain": cv.get("gain") or {}, "ee": p.get("ee") or {},
                     "gains": gains, "rho_max": terms["rho_max"], "rho_prefix": terms["rho_prefix"],
                     "prefix_source": terms["prefix_source"], "consistent": terms["consistent"],
                     "rho_max_per_token": cv.get("rho_max_per_token"),
                     "regime_floor": terms["grid"][0][0] if terms["grid"] else REGIMES[0][1],
                     "regimes": {label: _regime(terms, r) for label, r in REGIMES}}
    basis = [*(f"{n}: gains at its measured probe cost" for n, v in out.items() if v.get("gains")),
             *(f"{n}: gains at every grid ρ from {v['regime_floor']:g} up to {v['rho_prefix']:g}" for n, v in out.items()
               if v.get("consistent"))]
    missing = [n for n, v in out.items() if "error" in v]
    unknown = [n for n, v in out.items() if "error" not in v and (v["gains"] is None or v["consistent"] is None)]
    build = True if True in halves else None if missing or unknown or not out else False
    why = [*(f"{n} (no converge block)" for n in missing),
           *(f"{n} ({'no measured probe cost' if out[n]['gains'] is None else 'no rho map'})" for n in unknown)]
    return {"pass": None, "descriptive": True, "models": out, "rule": BUILD_RULE, "build": build, "basis": basis,
            **({"error": "missing: " + ", ".join(why)} if why else {})}


# ---- the parts -----------------------------------------------------------------------------------------------------

def session_provenance(out: Path, parts: Sequence[str] = ()) -> dict:
    """What ran, recorded at start (the think bench's seams): the code's commit and whether its tracked files were
    dirty, the llama-server binary and its `--version`, the free disk; `warnings` for a dirty or unreadable checkout and
    a build that is not EXPECTED_BUILD."""
    session = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "parts": list(parts),
               "code": v03t.code_state(ROOT),
               "llama_server": LLAMA_SERVER, "version": v03t._llama_server_version(LLAMA_SERVER),
               "disk": v03t.disk_free_gib([str(out), "/"])}
    code, warnings = session["code"], []
    if code.get("commit") is None:
        warnings.append(f"the code's commit could not be read ({code.get('error')})")
    elif code.get("dirty"):
        warnings.append("the code that ran is dirty (tracked files differ from " + str(code["commit"])[:12] + "): "
                        + "; ".join(code.get("status") or [])[:300])
    if EXPECTED_BUILD not in str(session["version"]):
        warnings.append(f"llama-server --version does not name {EXPECTED_BUILD}: {version_line(session['version'])!r}")
    session["warnings"] = warnings
    return session


def greedy_flags(path: Path, profile: Mapping[str, Any]) -> list[str]:
    """G5's greedy session: its profile, and the served reserve fixed at the 2,000 ms deadline point's r_cal (rounded
    up to whole ms, as serve takes it), so the served cut and the simulated one share their reserve."""
    r = ((profile.get("deadlines") or {}).get(str(G5_DEADLINE)) or {}).get("reserve_ms")
    return ["--think-profile", str(path), *(["--think-deadline-reserve", str(math.ceil(float(r)))] if r is not None else [])]


def calib_argv(name: str, out: Path, data: Path) -> list[str]:
    """`callo calibrate` for one gate run in launch mode (one model server at a time; calibrate's launch passes
    llama-server `--cache-ram 0`, recorded in the profile's `server_args`). No `--cache` flag: calibrate's default (on)
    is what serve ships. A DIR holding a run.json is resumed."""
    model, _, flags = CALIB[name]
    d = out / "calib" / name
    return ["calibrate", *flags, "--data", str(data), "--model", model, "--llama-server", LLAMA_SERVER, "--ctx", str(CTX),
            "--parallel", str(PARALLEL), "--seed", str(SEED), "--out", str(d),
            *(["--resume"] if (d / "run.json").exists() else [])]


def _profile_path(out: Path, name: str) -> Path | None:
    paths = sorted((out / "calib" / name).glob("*.think.json"))
    return paths[0] if paths else None


def _profile(out: Path, name: str) -> dict | None:
    p = _profile_path(out, name)
    return json.loads(p.read_text()) if p is not None else None


def run_calib(out: Path, data: Mapping[str, Path]) -> dict:
    res = {}
    for name in CALIB:
        rc, err = _calibrate(calib_argv(name, out, data[CALIB[name][1]]))
        res[name] = {"rc": rc, "tail": err.strip().splitlines()[-1] if err.strip() else ""}
        if rc == 130:
            raise KeyboardInterrupt(f"calibrate {name} was interrupted")
    return res


def _session(name: str, out: Path, path: Path, profile: Mapping[str, Any], flags: Sequence[str]) -> Any:
    """A served Tiel session: callo as shipped (`cache=True`), the profile's llama-server passthrough repeated
    (`profile_launch`: its `--cache-ram 0`), so `--think-profile` finds nothing that differs."""
    return _start(name, out, model=TIEL, flags=_join_flags(flags, profile_launch(dict(profile))["server_args"]), ctx=CTX,
                  cache=True)


def run_serve(out: Path, subset: Sequence[Item], res: dict) -> dict:
    """The served sessions on Tiel, one launch per profile, filling `res` as they go (a failed session keeps what the
    earlier ones measured): the Tiel run's profile for G1 (1,000 and 2,000 ms; the 2,000 ms rows are G6's too) and
    G4's budget-only baseline; the greedy run's for G5 (2,000 ms, its pinned greedy sampler, the point's reserve) and
    G3. Each session first sends WARMUP_N discarded requests."""
    tiel_path, greedy_path = _profile_path(out, "tiel"), _profile_path(out, "tiel-greedy")
    if tiel_path is None or greedy_path is None:
        raise RuntimeError("serve needs the tiel and tiel-greedy profiles (run the calib part first)")
    tiel, greedy = json.loads(tiel_path.read_text()), json.loads(greedy_path.read_text())
    res["cache"], res["warmup"] = True, WARMUP_N
    c = _session("serve-tiel", out, tiel_path, tiel, ["--think-profile", str(tiel_path)])
    try:
        serve_items(c.url, subset[:WARMUP_N], deadline_ms=G1_DEADLINES[0])
        res["g1"] = {}
        for d in G1_DEADLINES:
            res["g1"][str(d)] = serve_items(c.url, subset, deadline_ms=d)
        bmax = max(int(b) for b in tiel["budgets"])
        res["g4_budget"] = serve_items(c.url, subset[:G3_N], deadline_ms=0, budget=bmax)    # unchunked: one call
        res["stats_deadline"] = ((c.stats() or {}).get("think") or {}).get("deadline")
    finally:
        c.finish()
    c = _session("serve-greedy", out, greedy_path, greedy, greedy_flags(greedy_path, greedy))
    try:
        serve_items(c.url, subset[:WARMUP_N], deadline_ms=G5_DEADLINE)
        res["g5"] = serve_items(c.url, subset, deadline_ms=G5_DEADLINE)
        served = ((c.stats() or {}).get("think") or {}).get("deadline") or {}
        res["g5_reserve"] = {"point": ((greedy.get("deadlines") or {}).get(str(G5_DEADLINE)) or {}).get("reserve_ms"),
                             "served": served.get("reserve_ms"), "source": served.get("reserve_source")}
        res["g3"] = g3_pairs(c.url, subset[:G3_N], cap=max(int(b) for b in greedy["budgets"]))
    finally:
        c.finish()
    return res


def gates(out: Path, serve: Mapping[str, Any], g2: Mapping[str, Any] | None) -> dict:
    tiel, greedy, qwen = _profile(out, "tiel"), _profile(out, "tiel-greedy"), _profile(out, "qwen9b")
    g1_rows = {int(d): rows for d, rows in (serve.get("g1") or {}).items()}
    pre = (((tiel or {}).get("latency") or {}).get("prefill_ms_per_token") or {}).get("p50")
    return {"g1": g1(g1_rows) if g1_rows else {"pass": None, "error": "no G1 sessions"},
            "g2": dict(g2) if g2 is not None else {"pass": None, "error": "the game-client agent pilot has not run (--g2)"},
            "g3": g3(serve.get("g3") or []),
            "g4": g4(serve.get("stats_deadline"), g1_rows.get(1000, []), serve.get("g4_budget") or [],
                     None if pre is None else float(pre)),
            "g5": {**g5(greedy or {}, serve.get("g5") or []), "reserve_ms": serve.get("g5_reserve")},
            "g6": g6(tiel or {}, g1_rows.get(2000, [])),
            "c1": c1({"tiel": tiel or {}, "qwen9b": qwen or {}})}


def main(argv: list[str] | None = None) -> int:
    """0 when every requested part ran (a FAIL is the feature's and is in the doc), 1 when a part errored, 2 when the
    bench refused to start, 3 when it was interrupted."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--data", default=TRANSFER_TEST)
    ap.add_argument("--parts", default=",".join(PARTS), help=f"which parts to run, in this order: {','.join(PARTS)}")
    ap.add_argument("--g2", default=None, help="the game-client agent pilot's gate-2 JSON")
    ap.add_argument("--doc", default=str(DOC))
    ap.add_argument("--min-free-gb", type=float, default=30.0, help="free RAM (vm_stat) the GPU parts need to start")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parts = [p for p in a.parts.split(",") if p]
    if any(p not in PARTS for p in parts):
        print(f"refusing to start: unknown part(s) in {parts}; choose from {','.join(PARTS)}")
        return 2
    gpu = bool({"calib", "serve"} & set(parts))
    gpu_paths = [("--data", a.data, lib.DATA_ENV), *(("", model, lib.MODELS_ENV) for model, _, _ in CALIB.values()),
                 ("", LLAMA_SERVER, lib.SERVER_ENV)] if gpu else []
    unset = lib.unset_paths(("--out", a.out, lib.OUT_ENV), *gpu_paths)
    if unset:
        print(unset)
        return 2
    if gpu:
        try:
            busy = lib.running_gpu_servers()
        except RuntimeError as e:                   # a ps that fails is "cannot tell", never "no servers"
            print(f"refusing to start: {e}: cannot tell whether a model server is running")
            return 2
        if busy:
            print(f"refusing to start: a model server is already running (one model server at a time; wait for an "
                  f"agreed window): {busy}")
            return 2
        if not Path(a.data).is_file():
            print(f"refusing to start: no data file {a.data}")
            return 2
        free = lib.free_memory_bytes()
        if a.min_free_gb > 0 and (free is None or free < a.min_free_gb * 1024**3):
            print(f"refusing to start: {free} bytes free (vm_stat) < --min-free-gb {a.min_free_gb}")
            return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pid_file = out / "run.pid"
    owner = pid_owner(pid_file)
    if owner is not None:
        print(f"refusing to start: {pid_file} names live pid {owner} (another run? check it; never signal it blindly)")
        return 2
    previous = _install_stop_handlers()
    me = os.getpid()
    summary_path = out / "summary.json"
    s = json.loads(summary_path.read_text()) if summary_path.exists() else {"parts": {}, "errors": {}}
    if "session" in s:                              # a summary from before `sessions`: its one record goes first
        s.setdefault("sessions", []).insert(0, s.pop("session"))
    record = session_provenance(out, parts)
    s.setdefault("sessions", []).append(record)     # every run's, so a later gates-only run keeps the GPU run's
    for w in record["warnings"]:
        print(f"WARNING: {w}")
        log.warning("%s", w)
    rc = 0
    try:
        pid_file.write_text(f"{me}\n")
        test, subset_path = out / "data" / "test.jsonl", out / "data" / "subset.jsonl"
        for part in parts:
            try:
                if part in ("calib", "serve"):
                    busy = _foreign_servers()          # every server this bench started is stopped between parts
                    if busy:
                        raise RuntimeError(f"a model server is running before the {part} part: {busy}")
                    if not test.exists():
                        select_records(a.data, test)
                    if not subset_path.exists():
                        select_records(a.data, subset_path, limit=SUBSET_N, seed=SEED)
                if part == "calib":
                    s["parts"]["calib"] = run_calib(out, {"test": test, "subset": subset_path})
                elif part == "serve":
                    s["parts"]["serve"] = {}
                    run_serve(out, load_data(subset_path)[0], s["parts"]["serve"])
                else:
                    g2 = s.get("gates", {}).get("g2")
                    if a.g2:
                        g2 = {**json.loads(Path(a.g2).read_text()), "source": str(Path(a.g2).resolve())}
                    s["gates"] = gates(out, s["parts"].get("serve") or {}, g2 if g2 and g2.get("source") else None)
                s["errors"].pop(part, None)
            except (KeyboardInterrupt, SystemExit):     # Ctrl-C, or `kill -TERM $(cat run.pid)` (the stop handlers)
                s["errors"][part] = "interrupted"
                rc = 3
                break
            except Exception as e:  # noqa: BLE001 -- a part's failure is recorded; the next part still runs
                log.exception("part %s failed", part)
                s["errors"][part] = f"{type(e).__name__}: {e}"
                rc = 1
            finally:
                s["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                summary_path.write_text(json.dumps(s, indent=1, default=str))
        Path(a.doc).parent.mkdir(parents=True, exist_ok=True)
        Path(a.doc).write_text(render(s), encoding="utf-8")
        for line in verdict_lines(s):
            print(line)
        print(f"deadline bench: done, rc {rc} ({a.doc})")
        return rc
    finally:
        if _pid_in(pid_file) == me:
            pid_file.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
