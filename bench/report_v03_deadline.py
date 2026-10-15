"""Render docs/bench/v0.3-deadline.md from the v0.3 think-deadline bench's summary.json. Numbers come from the file,
never by hand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bench.report_v02 import _f  # one "n/a"-aware number format for every bench report
from bench.report_v03_think import _verdict, version_line

# C1's regime table (the report's columns): rho_max/rho_prefix read against the common probe costs, in decode steps;
# each label's column header
REGIMES = (("slow decode", 0.5), ("pure attention", 1.0), ("ρ 2", 2.0), ("ρ 3", 3.0))
REGIME_HEADERS = {"slow decode": "ρ 0.5 (slow decode)", "pure attention": "ρ 1 (pure attention ≈ 1–3)", "ρ 2": "ρ 2",
                  "ρ 3": "ρ 3"}

GATES = [
    ("G1", "g1", "adherence, single-stream: Tiel, the 200 questions at D = 1,000 and 2,000; over the deadline/budget "
                 "closes only (>= 60 per D), p99(elapsed_ms - D) <= 50 ms, max reported; no budget0 fallback at 2,000"),
    ("G2", "g2", "adherence under the game-client agent pilot's load: the pilot at D* (>= 1,200 calls); over the deadline/budget "
                 "closes (>= 200), p99(elapsed_ms - D*) <= 100 ms, p99.5 and max reported, with the think tokens and "
                 "the close mix beside it"),
    ("G3", "g3", "the cut readout equals the cut: greedy, 48 questions, live at D = 1,500 then budgets at B = k; tokens "
                 "equal on >= n - 2; on the m identical argmax agrees on >= m - 1, median \\|Δp\\| <= 1e-3; first-pass "
                 "prompt_n <= 4 and continuation prompt_n = 1 on >= 99 %"),
    ("G4", "g4", "chunk cost: ô p50 <= 15 ms; think tokens/s in deadline-cut traces >= 90 % of the budget-only rate"),
    ("G5", "g5", "the static mapping predicts serving (greedy, D = 2,000): natural share within 3 points, k p50 within "
                 "5 %"),
    ("G6", "g6", "deadline point vs served (descriptive): paired Δacc (served - point) with its CI; served elapsed "
                 "p50/p95 beside the point's latency"),
    ("C1", "c1", "convergence measurement (descriptive): flips, the fixed-sequence verdicts, the rules, c, the gain "
                 "verdicts, rho_max and rho_prefix per model; the early-exit build decision by the pre-registered rule"),
]


def _g(v: Any) -> str:
    return "n/a" if v is None else format(float(v), "g")


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{100 * v:+.1f} %"


def _g6_latency(lat: dict | None) -> str:
    """G6: the served elapsed p50/p95 beside the deadline point's, on the same questions, then the point's
    own statistics over all its questions (the deadline rows run on the budgets' clock)."""
    if not lat:
        return ""
    return (f"; served elapsed p50/p95 {_f(lat.get('served_p50_ms'), '.0f')}/{_f(lat.get('served_p95_ms'), '.0f')} ms "
            f"vs the point's {_f(lat.get('point_same_p50_ms'), '.0f')}/{_f(lat.get('point_same_p95_ms'), '.0f')} ms "
            f"on the same {lat.get('n')} questions ({_pct(lat.get('served_vs_point_p50'))}/"
            f"{_pct(lat.get('served_vs_point_p95'))}); the point's lat p50/p95 over all its questions "
            f"{_f(lat.get('point_p50_ms'), '.0f')}/{_f(lat.get('point_p95_ms'), '.0f')} ms")


def _result(key: str, g: Any) -> str:
    if not isinstance(g, dict):
        return "not run"
    if g.get("error") and key not in ("g1", "c1"):
        return str(g["error"])
    if key == "g1":
        per = "; ".join(f"{d} ms: n_cut {e['n_cut']}, p99 {_f(e['p99_over_ms'], '.1f')} ms, max "
                        f"{_f(e['max_over_ms'], '.1f')} ms, budget0 {e['fallbacks']}, 504 {e.get('timeouts', 0)}, "
                        f"failed {e.get('failed', 0)}" + (f" ({e['error']})" if "error" in e else "")
                        for d, e in (g.get("deadlines") or {}).items())
        return "; ".join(x for x in (per, g.get("error")) if x) or "n/a"
    if key == "g2":
        closes = ", ".join(f"{c} {n}" for c, n in (g.get("closes") or {}).items())
        tokens = "; ".join(f"{n} q: {_g(t.get('p5'))}/{_g(t.get('p50'))}/{_g(t.get('p95'))}"
                           for n, t in (g.get("think_tokens") or {}).items())
        return (f"calls {g.get('n_calls')}, n_cut {g.get('n_cut')}, p99 {_f(g.get('p99_over_ms'), '.1f')} ms, p99.5 "
                f"{_f(g.get('p995_over_ms'), '.1f')} ms, max {_f(g.get('max_over_ms'), '.1f')} ms"
                + (f"; closes {closes}" if closes else "") + (f"; think tokens p5/p50/p95 {tokens}" if tokens else "")
                + (f"; rerouted {g.get('rerouted')}, draining {g.get('draining')}" if "rerouted" in g else ""))
    if key == "g3":
        return (f"n {g['n']}, identical {g['identical']}, argmax {g['argmax_agree']}, median \\|Δp\\| "
                f"{_f(g['dp_median'], '.1e')}, first pass {_f(g['first_pass_ok'], '.3f')}, continuation "
                f"{_f(g['continuation_ok'], '.3f')}")
    if key == "g4":
        return (f"ô p50 {_f(g['overhead_p50_ms'], '.1f')} ms; {_f(g['rate_cut_tok_s'], '.1f')} vs "
                f"{_f(g['rate_budget_tok_s'], '.1f')} tok/s ({_f(100 * g['ratio'], '.0f')} %)")
    if key == "g5":
        r = g.get("reserve_ms") or {}
        return (f"natural {_f(100 * g['natural_served'], '.1f')} % served vs {_f(100 * g['natural_point'], '.1f')} % "
                f"point; k p50 {g['k_p50_served']} vs {g['k_p50_point']} (n {g['n']}); reserve served "
                f"{_f(r.get('served'), '.0f')} ms ({r.get('source')}) vs point {_f(r.get('point'), '.1f')} ms")
    if key == "g6":
        d = g["dacc"]
        return (f"Δacc {_f(d[0], '+.3f')} [{_f(d[1], '+.3f')}, {_f(d[2], '+.3f')}] (n {g['n']})"
                + _g6_latency(g.get("latency")))
    build = g.get("build")
    if build and g.get("basis"):
        return "build the early-exit serve side (" + "; ".join(g["basis"]) + ")" + (
            f"; {g['error']}" if g.get("error") else "")
    return ("build the early-exit serve side" if build else "do not build it" if build is False else
            f"no decision ({g.get('error')})") + (f"; {g['error']}" if build is not None and g.get("error") else "")


def _converge(c1: dict) -> list[str]:
    """The C1 table, then the regime table (does early exit still gain at that probe cost, consistently from the cheap
    end), then the pre-registered build rule and the decision."""
    out = ["| model | K | n | c ms (restore) | changed after confident | gains at the measured c | rho_max | rho_prefix |",
           "|---|---|---|---|---|---|---|---|"]
    models = c1.get("models") or {}
    for name, m in models.items():
        if "error" in m:
            out.append(f"| {name} | {m['error']} | | | | | | |")
            continue
        cost = m.get("cost") or {}
        flips = m.get("flips") or {}
        gains = ("not measured" if m.get("gains") is None else
                 ", ".join(f"{z}: {'yes' if (v or {}).get('gains') else 'no'}" for z, v in (m.get("gain") or {}).items())
                 or "no rule")
        prefix = ("n/a" if m.get("prefix_source") is None else "none" if m.get("rho_prefix") is None else
                  _g(m["rho_prefix"]) + (" (derived)" if m["prefix_source"] == "derived" else ""))
        out.append(f"| {name} | {m.get('K')} | {m.get('n')} | {_f(cost.get('c_ms'), '.1f')} "
                   f"({_f(cost.get('restore_share'), '.2f')}) | {flips.get('changed_after_confident', 'n/a')} | {gains} | "
                   f"{_g(m.get('rho_max'))} | {prefix} |")
    labels = list(next((m["regimes"] for m in models.values() if "regimes" in m), {}))
    if labels:
        heads = [REGIME_HEADERS.get(x, x) for x in labels]
        out += ["", "Early exit on the front at a probe cost of ρ decode steps (yes: at every grid ρ from 0.5 up to it; "
                    "isolated: there, but not at some cheaper ρ):", "",
                "| model | " + " | ".join(heads) + " |", "|---|" + "---|" * len(labels)]
        for name, m in models.items():
            if "regimes" in m:
                out.append(f"| {name} | " + " | ".join(m["regimes"][x] for x in labels) + " |")
    lines = [ln for name, m in models.items() if "error" not in m for ln in _z_lines(name, m)]
    if lines:
        out += ["", "Per target z: the cross-fitted rule, its early-exit point and its gain verdict at the measured c:",
                "", *lines]
    if c1.get("rule"):
        out += ["", f"Build rule: {c1['rule']}.", "", f"Decision: {_result('c1', c1)}."]
    return out


def _axis(v: Any) -> str:
    """One frontier axis of a gain verdict: yes / no, with what held (front, speed, accuracy)."""
    if not isinstance(v, dict):
        return "n/a"
    held = [k for k in ("front", "speed", "accuracy") if v.get(k)]
    return ("yes" if v.get("gains") else "no") + (f" ({', '.join(held)})" if v.get("gains") and held else "")


def _z_lines(name: str, m: dict) -> list[str]:
    """One line per z: its rule (cell, acc with Δ against the full trace, tokens saved), its EE point (acc, p50, p95)
    and its gain verdict on each axis; or that no rule reaches z."""
    out = []
    zs = sorted({*(m.get("rules") or {}), *(m.get("gain") or {})}, key=float)
    for z in zs:
        rule = (m.get("rules") or {}).get(z)
        if not rule:
            out.append(f"- {name} z {z}: no rule reaches z")
            continue
        cell, d = rule.get("cell") or [None, None], rule.get("dacc") or [None, None, None]
        line = (f"- {name} z {z}: rule (n {cell[0]}, y {cell[1]}), acc {_f(rule.get('acc'))} "
                f"(Δ vs full think {_f(d[0], '+.3f')} [{_f(d[1], '+.3f')}, {_f(d[2], '+.3f')}]), saves {_f(rule.get('tokens_saved'), '.1f')} tokens "
                f"({_f(rule.get('net_saving'), '.1f')} net)")
        ee = (m.get("ee") or {}).get(z)
        if ee:
            p50, p95 = ee.get("lat_p50_ms"), ee.get("lat_p95_ms")
            line += (f"; EE point acc {_f(ee.get('acc'))}, p50 {_f(p50[0] if isinstance(p50, list) else p50, '.0f')} "
                     f"ms, p95 {_f(p95[0] if isinstance(p95, list) else p95, '.0f')} ms")
        else:
            line += "; no EE point (no measured probe cost)"
        v = (m.get("gain") or {}).get(z)
        line += f"; gains: p50 {_axis((v or {}).get('p50'))}, p95 {_axis((v or {}).get('p95'))}" if v else "; gains: n/a"
        out.append(line)
    return out


def _session_line(label: str, session: dict) -> str:
    code = session.get("code") or {}
    commit = str(code.get("commit") or "unknown")[:12] + (" (dirty)" if code.get("dirty") else "")
    disk = ", ".join(f"{p} {_f(g, '.1f')} GiB" for p, g in (session.get("disk") or {}).items())
    parts = session.get("parts")
    return (f"{label}: commit {commit}, llama-server {version_line(session.get('version'))}, disk free {disk} "
            f"(started {session.get('started')}" + (f"; parts {', '.join(parts)}" if parts else "") + ").")


def _session_lines(sessions: Any) -> list[str]:
    """The latest run's header line (commit, llama-server build, free disk) and, when a gates-only run came later,
    the first GPU run's (the one that measured: calib or serve), then their start-up warnings (a dirty checkout,
    another build)."""
    if not isinstance(sessions, list) or not sessions:
        return []
    latest = sessions[-1]
    gpu = next((x for x in sessions if isinstance(x, dict) and {"calib", "serve"} & set(x.get("parts") or ())), None)
    shown = [("Run", latest), *([("GPU run", gpu)] if gpu is not None and gpu is not latest else [])]
    warnings = [w for _, x in shown for w in x.get("warnings") or []]
    return [*(_session_line(label, x) for label, x in shown), "",
            *(f"WARNING: {w}." for w in dict.fromkeys(warnings)), *([""] if warnings else [])]


def verdict(key: str, g: Any) -> str:
    """G6 is information ("info") unless the bench could not compute it, C1 whenever it decided the build (a missing
    model is then noted in its row); the gates are `_verdict`'s."""
    if key == "c1" and isinstance(g, dict) and g.get("build") is not None:
        return "info"
    return "info" if key == "g6" and isinstance(g, dict) and not g.get("error") else _verdict(g)


def verdict_lines(s: dict) -> list[str]:
    """One `<gate>: <verdict>` line per gate, the bench's terminal lines."""
    gates = s.get("gates") or {}
    return [f"{name}: {verdict(key, gates.get(key))}" for name, key, _ in GATES]


def render(s: dict) -> str:
    gates = s.get("gates") or {}
    warm = ((s.get("parts") or {}).get("serve") or {}).get("warmup")
    out = ["# callo v0.3: think deadlines acceptance bench", "",
           f"Updated {s.get('updated')}. The gates are `GATES` in `bench/report_v03_deadline.py`, judged by "
           "`bench/acceptance_v03_deadline.py`; their pre-registration is in the development history. "
           "G1-G5 failing means the feature is wrong, G6 and C1 are information. ERROR means the bench could not judge "
           "the gate (missing evidence names itself). callo is measured as shipped: the calibrations and the served "
           "sessions both run with `--cache` on (its default).", "",
           "## Deviations from the gates' original wording", "",
           f"- **Warm-up.** Each served session first sends {warm or 3} discarded warm-up requests at its deadline "
           "(`WARMUP_N`, as the other v0.3 benches warm up): below 100 cuts G1's nearest-rank p99 is the maximum, so a "
           "single cold-start late answer would otherwise decide G1.",
           "- **Launch shape.** The served sessions repeat the calibration launch's llama-server passthrough "
           "(the profile's `server_args`: `--cache-ram 0`, the host prompt cache off), so G5 and G6 compare serving "
           "with the calibration on one launch shape.", ""]
    out += _session_lines(s.get("sessions"))
    calib = (s.get("parts") or {}).get("calib") or {}
    if calib:
        out += ["| calibration run | exit | last line |", "|---|---|---|",
                *(f"| {n} | {r.get('rc')} | {r.get('tail') or ''} |" for n, r in calib.items()), ""]
    out += ["| gate | pass when | result | verdict |", "|---|---|---|---|"]
    for name, key, cond in GATES:
        g = gates.get(key)
        out.append(f"| {name} | {cond} | {_result(key, g)} | {verdict(key, g)} |")
    out.append("")
    if isinstance(gates.get("c1"), dict) and gates["c1"].get("models"):
        out += ["## Convergence (C1)", "", *_converge(gates["c1"]), ""]
    if s.get("errors"):
        out += ["## Errors", "", *(f"- {k}: {v}" for k, v in s["errors"].items()), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "bench" / "v0.3-deadline.md"))
    a = ap.parse_args(argv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(render(json.loads(Path(a.summary).read_text())))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
