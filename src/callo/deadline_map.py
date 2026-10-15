"""Think deadlines' static mapping: for one think profile and a setting (a deadline D, optionally with a
budget B and another readout reserve), the closes the planner would make, the cut k, the static budgets that meet D,
and the accuracy: exact when the profile measured the setting (a deadline point at D simulated at this reserve, and a
calibrated B when one is given), bracketed otherwise. Offline and pure: `callo compare --deadline` and calibrate's
summary print it.

Each question's token side is calibrate's own simulation (`deadline.simulate`) on its `rows[].t` terms
(`deadline.trace_terms`); a profile from before the deadline fields falls back to the derivation from the decode
median and the Bmax latency, and says so. Every figure is single-stream and in-callo, on the budgets' clock.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .deadline import DEFAULT_RESERVE_MS, simulate
from .think import bootstrap_mean, nearest_rank, row_latencies

COVERAGE = (0.95, 0.99)
Terms = tuple[dict, int, float, float, "str | None"]   # (row, n_gen, a, d, the Bmax trace's close; None: one option)


def _pct(xs: Sequence[float], q: float) -> float:
    """Nearest rank (`think.nearest_rank`, as /callo/stats and calibrate's `percentile`)."""
    s = sorted(xs)
    return s[min(len(s), nearest_rank(len(s), q)) - 1]


def bmax_of(profile: Mapping[str, Any]) -> int:
    """The profile's largest calibrated budget (0 for a budget-0-only profile)."""
    return max((int(b) for b in (profile.get("budgets") or {}) if int(b) > 0), default=0)


def point_cap(profile: Mapping[str, Any], deadline_ms: int) -> int:
    """The cap a `--think-deadline D` request runs under, for the serve line (one source for compare and the summary):
    the deadline point's own `cap` when the profile holds one at D, else the profile's Bmax (the B*)."""
    point = (profile.get("deadlines") or {}).get(str(deadline_ms))
    cap = point.get("cap") if isinstance(point, Mapping) else None
    return cap if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0 else bmax_of(profile)


def profile_reserve(profile: Mapping[str, Any], deadline_ms: int) -> tuple[float, str]:
    """The reserve a mapping simulates with, and its source: the deadline point's r_cal (`deadline point`), else the
    profile's served-like forced read as serve seeds r (1.5 x p95 + 10 ms, `profile`), else 100 ms (`default`)."""
    point = (profile.get("deadlines") or {}).get(str(deadline_ms))
    r = point.get("reserve_ms") if isinstance(point, Mapping) else None
    if isinstance(r, (int, float)) and not isinstance(r, bool):
        return float(r), "deadline point"
    read = ((profile.get("latency") or {}).get("read_forced_ms") or {}).get("p95")
    if isinstance(read, (int, float)) and not isinstance(read, bool):
        return 1.5 * float(read) + 10.0, "profile"
    return DEFAULT_RESERVE_MS, "default"


def terms(profile: Mapping[str, Any]) -> tuple[list[Terms], bool]:
    """Every row holding Bmax with (n_gen, a, d, the Bmax trace's close), in row order, and whether the terms are
    approximate: the derivation for a profile from before `rows[].t` (d = 1000 / decode_tok_s_median, a = lat(Bmax)
    - d x k(Bmax), n_gen = k(Bmax)). A question without a trace (one option: k 0, no close) thinks 0 tokens and has
    close None: `mapping` counts it natural without simulating it, as serve answers it at once under any deadline,
    and it never makes the terms approximate."""
    bmax = bmax_of(profile)
    fields = list(profile.get("row_fields") or ["correct", "nll1", "pmax", "lat_ms", "k", "close"])
    at_lat, at_k, at_close = fields.index("lat_ms"), fields.index("k"), fields.index("close")
    decode = ((profile.get("environment") or {}).get("run") or {}).get("decode_tok_s_median")
    out: list[Terms] = []
    approx = False
    for r in profile.get("rows") or []:
        top = (r.get("b") or {}).get(str(bmax))
        if top is None:
            continue
        close = top[at_close] if top[at_close] in ("natural", "eos", "forced") else "natural"
        t = r.get("t")
        if top[at_k] == 0 and top[at_close] not in ("natural", "eos", "forced"):         # one option: no trace
            out.append((r, 0, 0.0, 0.0, None))
        elif isinstance(t, list) and len(t) == 3:
            out.append((r, int(t[0]), float(t[1]), float(t[2]), close))
        elif isinstance(decode, (int, float)) and decode > 0:
            d = 1000.0 / decode
            approx = True
            out.append((r, int(top[at_k]), float(top[at_lat]) - d * int(top[at_k]), d, close))
    return out, approx


def mapping(profile: Mapping[str, Any], deadline_ms: int, *, budget: int | None = None,
            reserve_ms: float | None = None, seed: int = 0) -> dict:
    """For one profile and setting. Per question, the first limit to bind: natural when the trace closes and
    n_gen <= min(B, k_D); else `budget` at B when B <= k_D; else `deadline` at k_D; the budget-0 fallback when no token
    fits. k_D is the deadline point's own row when the profile holds one at D and the reserve is the point's, else the
    planner simulated on the row's terms at this reserve (cap Bmax: k_D does not depend on the cap while the cap is
    above it). A single-option question (no trace) is natural, never simulated. Under a budget
    B above Bmax, a question whose Bmax trace reached the cap before the deadline thinks on past every measured trace:
    it is re-simulated under B (its length past Bmax unmeasured, so it never closes on its own) and has no readout.
    The shares, the cut k percentiles, the static budgets meeting D on 95 % / 99 % of the questions, and the
    accuracy: exact (a cluster-bootstrap CI, the profile's clusters, strata and seed) when every question's binding
    limit has a readout in the profile, else the bracket over the questions without one."""
    bmax = bmax_of(profile)
    cap = budget if budget is not None else bmax
    r, source = (float(reserve_ms), "--reserve") if reserve_ms is not None else profile_reserve(profile, deadline_ms)
    point = (profile.get("deadlines") or {}).get(str(deadline_ms))
    exact_point = isinstance(point, Mapping) and source == "deadline point"
    calibrated = {int(b) for b in (profile.get("budgets") or {})}
    fields = list(profile.get("row_fields") or ["correct", "nll1", "pmax", "lat_ms", "k", "close"])
    at_correct, at_k, at_close = fields.index("correct"), fields.index("k"), fields.index("close")
    rows, approx = terms(profile)
    closes = {"natural": 0, "budget": 0, "deadline": 0, "budget0": 0}
    cut: list[int] = []
    known: list[float | None] = []
    beyond = 0
    for row, n_gen, a, d, close in rows:
        b_rows = row.get("b") or {}
        if close is None:                       # one option: answered at once under any limit (finding 3)
            closes["natural"] += 1
            res = b_rows.get(str(bmax))
            known.append(float(bool(res[at_correct])) if res is not None else None)
            continue
        dl = (row.get("d") or {}).get(str(deadline_ms)) if exact_point else None
        if dl is not None:
            k_d, c_d = int(dl[at_k]), str(dl[at_close] or "natural")
        else:
            s = simulate(a_ms=a, d_ms=d, n_gen=n_gen, close=close, deadline_ms=deadline_ms, reserve_ms=r, cap=bmax)
            k_d, c_d = s.k, s.close or "budget0"
        past = budget is not None and budget > bmax and close == "forced" and c_d == "budget"
        if past:                                # finding 4: under B > Bmax the trace thinks on past Bmax, unmeasured
            s = simulate(a_ms=a, d_ms=d, n_gen=n_gen, close=close, deadline_ms=deadline_ms, reserve_ms=r, cap=budget)
            k_d, c_d = s.k, s.close or "budget0"
            beyond += 1
        if c_d == "budget0":
            which, res = "budget0", dl if dl is not None else b_rows.get("0")
        elif close in ("natural", "eos") and n_gen <= min(cap, k_d):
            which, res = "natural", b_rows.get(str(bmax))
        elif cap <= k_d:
            which, res = "budget", b_rows.get(str(cap)) if cap in calibrated else None
        else:
            which, res = "deadline", dl
            cut.append(k_d)
        if past:
            res = None
        closes[which] += 1
        known.append(float(bool(res[at_correct])) if res is not None else None)
    n = len(known)
    out: dict[str, Any] = {"deadline_ms": deadline_ms, "budget": budget, "cap": cap, "bmax": bmax,
                           "questions": len(profile.get("rows") or []), "reserve_ms": r,
                           "reserve_source": source, "n": n, "closes": {c: (v / n if n else 0.0) for c, v in closes.items()},
                           "cut_k": [_pct(cut, q) for q in (0.05, 0.5, 0.95)] if cut else None,
                           "static": static_budgets(rows, deadline_ms, r, bmax), "approx_terms": approx}
    unknown = sum(v is None for v in known)
    if n and not unknown:
        _, clusters, strata = row_latencies(profile, "b", str(bmax))    # the profile's own clusters and strata
        if len(clusters) != n:                                          # a hand-cut profile: plain clusters
            clusters, strata = [str(row.get("cluster") or row.get("q")) for row, *_ in rows], None
        acc, lo, hi = bootstrap_mean([float(v or 0.0) for v in known], clusters, seed=seed, strata=strata)
        out["accuracy"] = {"exact": True, "acc": acc, "ci": [lo, hi]}
    elif n:
        sure = sum(v for v in known if v is not None)
        out["accuracy"] = {"exact": False, "lo": sure / n, "hi": (sure + unknown) / n, "unknown": unknown / n,
                           "smallest_budget": min((b for b in calibrated if b > 0), default=None),
                           "beyond_bmax": beyond / n}
    else:
        out["accuracy"] = None
    return out


def static_budgets(rows: Sequence[Terms], deadline_ms: float, reserve_ms: float, bmax: int) -> dict[str, int | None]:
    """The static budget meeting D on p: for p in .95 and .99, the largest B (1..Bmax) for which the share of the
    questions with a + d·min(B, n_gen) + r <= D is at least p; None when not even B = 1 meets it."""
    out: dict[str, int | None] = {}
    for p in COVERAGE:
        best = None
        for b in range(1, bmax + 1):
            ok = sum(a + d * min(b, n_gen) + reserve_ms <= deadline_ms for _, n_gen, a, d, _ in rows)
            if rows and ok >= math.ceil(p * len(rows) - 1e-9):
                best = b
            else:
                break
        out[f"{p:g}"] = best
    return out


def mapping_lines(name: str, m: Mapping[str, Any], serve: str | None = None) -> list[str]:
    """The block for one profile and setting; `serve` is the `callo serve ... --think-deadline D` command."""
    pct = {k: f"{100 * v:.0f} %" for k, v in m["closes"].items()}
    cut, st = m.get("cut_k"), m["static"]
    lines = [f"{name}  deadline {m['deadline_ms']} ms, cap {m['cap']} (single-stream, in-callo: add your client's hop "
             f"and any queueing)"]
    total = m.get("questions")
    if not m["n"]:                                       # nothing to simulate on
        lines.append("  no question has deadline terms (rows[].t) or a decode median to derive them")
        if serve:
            lines.append(f"  serve: {serve}")
        return lines
    lines.append(f"  closes: natural {pct['natural']}  budget {pct['budget']}  deadline {pct['deadline']}  budget0 "
                 f"{pct['budget0']}" + (f"   cut k p5/p50/p95 {cut[0]}/{cut[1]}/{cut[2]}" if cut else "")
                 + (f"   (over {m['n']} of {total} questions)" if isinstance(total, int) and total != m["n"] else ""))
    if st.get("0.95") is not None:
        lines.append(f"  static: --think-budget {st['0.95']} meets {m['deadline_ms']} ms on 95 % of questions"
                     + (f", {st['0.99']} on 99 %" if st.get("0.99") is not None else ", none on 99 %"))
    else:
        lines.append(f"  static: no budget meets {m['deadline_ms']} ms on 95 % of questions")
    acc = m.get("accuracy")
    note = "  approx (profile predates deadline fields)" if m.get("approx_terms") else ""
    if acc and acc["exact"] and m.get("approx_terms"):   # every readout measured, the closes derived
        lines.append(f"  accuracy: {acc['acc']:.2f} [{acc['ci'][0]:.2f}, {acc['ci'][1]:.2f}] (reserve "
                     f"{m['reserve_ms']:.0f} ms, {m['reserve_source']}, readouts measured; closes approx: profile "
                     f"predates deadline fields)")
    elif acc and acc["exact"]:
        lines.append(f"  accuracy: {acc['acc']:.2f} [{acc['ci'][0]:.2f}, {acc['ci'][1]:.2f}] (exact, reserve "
                     f"{m['reserve_ms']:.0f} ms, {m['reserve_source']})")
    elif acc:
        small = acc.get("smallest_budget")
        below = acc["unknown"] > acc.get("beyond_bmax", 0.0) + 1e-12  # some unknowns are cuts below the budgets
        lines.append(f"  accuracy: approx [{acc['lo']:.2f}, {acc['hi']:.2f}]: {100 * acc['unknown']:.0f} % of questions "
                     f"cut where the profile holds no readout" + (f" (the smallest calibrated budget is {small})"
                                                                   if small and below else "")
                     + (f"; {100 * acc['beyond_bmax']:.0f} % think on above Bmax {m.get('bmax')}, past every measured "
                        f"trace" if acc.get("beyond_bmax") else "") + note)
    if serve:                                            # a combined setting already names its budget
        static = (st.get("0.99") or st.get("0.95")) if m.get("budget") is None else None
        lines.append(f"  serve: {serve}" + (f"   (static: --think-budget {static})" if static is not None else ""))
    return lines
