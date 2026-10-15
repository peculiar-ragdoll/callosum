"""The early-exit bench's report: the calibration runs, then per model E1-E3 with their numbers and verdicts. A gate
with missing evidence is ERROR, never PASS. A `--gate` run's report is the early-exit gate's result instead, with its
gated point's profile vs served line."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

GATES = (("e1", "E1", "probe cost: restore >= 95 % of hybrid probes; probe read p50 <= 1.5 x read_forced p50"),
         ("e2", "E2", "greedy, one trace: stops whose answer equals the end answer >= z - 0.03"),
         ("e3", "E3", "served Δacc CI holds the profile's Δacc; the profile's p50 saving lies in the served CI"))


def _cell(v: Any) -> str:
    """A value as one Markdown table cell: a "|" in it (a calibrate error's last line) would split the row."""
    return str(v).replace("|", "\\|").replace("\n", " ")


def _verdict(g: Mapping[str, Any]) -> str:
    return "ERROR" if g.get("pass") is None else "PASS" if g["pass"] else "FAIL"


def _result(key: str, g: Mapping[str, Any]) -> str:
    if g.get("pass") is None:
        return str(g.get("error") or "no evidence")
    if key == "e1":
        restore = f"restored {100 * g['restored_share']:.1f} % of {g['n']}" + ("" if g.get("hybrid") else " (not hybrid)")
        n = f" (n {g['read_forced_n']})" if g.get("read_forced_n") is not None else ""
        return (f"{restore}; probe read p50 {g['read_p50_ms']:.1f} ms vs read_forced p50 "
                f"{g['read_forced_p50_ms']:.1f} ms{n}")
    if key == "e2":
        lo, hi = g["ci"]
        n, y = g["cell"]
        traces = f"; {g['traces']} traces, {g.get('dropped', 0)} dropped" if g.get("traces") is not None else ""
        return (f"{g['share']:.3f} [{lo:.3f}, {hi:.3f}] of {g['n']} stops, rule ({n}, {y}) (bound {g['bound']:.2f})"
                f"{traces}")
    d, s = g["dacc"], g["saving_ms"]
    conv = (g.get("closes") or {}).get("converged")
    return (f"Δacc {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] vs profile {g['profile_dacc']:+.3f}; p50 saving "
            f"{s[0]:.0f} [{s[1]:.0f}, {s[2]:.0f}] ms vs profile {g['profile_saving_ms']:.0f} ms (n {g['n']}"
            + (f", converged {conv} of {g['n']}" if conv is not None else "") + ")")


def _gate_verdict(g: Mapping[str, Any]) -> str:
    return "ERROR" if g.get("pass") is None else str(g.get("verdict"))


def _ms(v: Any) -> str:
    return "n/a" if v is None else f"{v:.0f}"


def _profile_line(pv: Mapping[str, Any]) -> str:
    """`--gate`'s profile vs served line for the gated point, as E3's."""
    head = f"Profile vs served (the gated point, as E3): **{_verdict(pv)}**"
    if pv.get("pass") is None:
        return f"{head}: {pv.get('error') or 'no evidence'}."
    return (f"{head}: the served Δacc CI {'holds' if pv['acc_pass'] else 'misses'} the profile's "
            f"{pv['profile_dacc']:+.3f}; the profile's p50 saving {pv['profile_saving_ms']:.0f} ms is "
            f"{'in' if pv['lat_pass'] else 'outside'} the served CI.")


def render_gate(s: Mapping[str, Any]) -> str:
    """The `--gate` run: Tiel, early exit with the gate against the end reading, on the held-out file."""
    g = s.get("gate") or {}
    label = f" ({s['label']})" if s.get("label") else ""
    lines = [f"# callo v0.3: the early-exit gate{label}", "",
             f"Updated {s.get('updated', '?')}. z = {s.get('z')}. Tiel, served on the transfer set's development "
             "split (held out): "
             "early exit with think_converge_gate true against the end reading (budget = the cap), alternating per "
             "question. Pre-declared: the gate holds iff the Δacc CI's upper end is >= 0 and the p50 saving CI's "
             "lower end is > 0.", "",
             f"**Verdict: {_gate_verdict(g)}**", ""]
    for run, r in ((s.get("parts") or {}).get("calib") or {}).items():       # the gated calibration
        lines += [f"Calibration {run}: exit {r.get('rc')}" + (", from a copy of the unlabelled run's raw file (the gate "
                  "reads added)" if r.get("seeded") else "") + (f": {r['error']}" if r.get("error") else "")
                  + (f"; last line: {_cell(r['tail'])}" if r.get("tail") else ""), ""]
    if g.get("pass") is None:
        lines.append(f"Error: {g.get('error') or 'no evidence'}")
    else:
        d, sv, lat, acc, rd = g["dacc"], g["saving_ms"], g["lat"], g["acc"], g["read_ms"]
        lines += ["| arm | accuracy | p50 ms | mean ms |", "|---|---|---|---|",
                  f"| early exit + gate | {acc['ee']:.3f} | {_ms(lat['ee']['p50'])} | {_ms(lat['ee']['mean'])} |",
                  f"| end reading | {acc['end']:.3f} | {_ms(lat['end']['p50'])} | {_ms(lat['end']['mean'])} |", "",
                  f"Δacc {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]; p50 saving {sv[0]:.0f} [{sv[1]:.0f}, {sv[2]:.0f}] ms "
                  f"(n {g['n']}).", "",
                  *([_profile_line(g["profile"]), ""] if isinstance(g.get("profile"), Mapping) else []),
                  f"Gate: skipped {100 * g['skip_share']:.1f} % of {g['n']} requests ({g['gate_errors']} failed reads); "
                  f"read p50 {_ms(rd['p50'])} ms, p95 {_ms(rd['p95'])} ms.", "",
                  "Closes (early-exit arm): " + ", ".join(f"{k} {v}" for k, v in sorted((g.get("closes") or {}).items()))
                  + ".", "", "| source | n | early exit + gate | end reading |", "|---|---|---|---|"]
        lines += [f"| {_cell(src)} | {b['n']} | {b['ee']:.3f} | {b['end']:.3f} |"
                  for src, b in (g.get("by_source") or {}).items()]
    if s.get("errors"):
        lines += ["", "Errors: " + "; ".join(f"{k}: {v}" for k, v in s["errors"].items())]
    return "\n".join(lines) + "\n"


def render(s: Mapping[str, Any]) -> str:
    if "gate" in s:
        return render_gate(s)
    label = f" ({s['label']})" if s.get("label") else ""
    lines = [f"# callo v0.3: early-exit acceptance bench{label}", "",
             f"Updated {s.get('updated', '?')}. z = {s.get('z')}. Calibrated on the transfer set's test split, "
             "served on its development split (held out). ERROR means the bench could not judge the gate.", "",
             "| calibration run | exit | last line |", "|---|---|---|"]
    for run, r in ((s.get("parts") or {}).get("calib") or {}).items():
        lines.append(f"| {run} | {r.get('rc')} | {_cell(r.get('tail', ''))} |")
    lines += ["", "| model | gate | pass when | result | verdict |", "|---|---|---|---|---|"]
    for model, gs in (s.get("gates") or {}).items():
        for key, name, when in GATES:
            g = gs.get(key) or {"pass": None, "error": "not computed"}
            lines.append(f"| {model} | {name} | {when} | {_cell(_result(key, g))} | {_verdict(g)} |")
    if s.get("errors"):
        lines += ["", "Errors: " + "; ".join(f"{k}: {v}" for k, v in s["errors"].items())]
    return "\n".join(lines) + "\n"


def verdict_lines(s: Mapping[str, Any]) -> list[str]:
    if "gate" in s:
        g = s.get("gate") or {}
        pv = g.get("profile")
        return [f"tiel gate: {_gate_verdict(g)}",
                *([f"tiel gate profile vs served: {_verdict(pv)}"] if isinstance(pv, Mapping) else [])]
    return [f"{model} {name}: {_verdict(gs.get(key) or {'pass': None})}"
            for model, gs in (s.get("gates") or {}).items() for key, name, _ in GATES]
