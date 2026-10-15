"""Render docs/bench/v0.3-bias.md from the v0.3 position-bias bench's summary.json. Numbers come from the file,
never by hand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.report_v02 import _f  # one "n/a"-aware number format for both reports

GATES = [("B0", "b0", "calibrate passes the guard on a fresh default launch (TielCoder, -kvu, probe erase active), "
                      "no --allow-busy, profile clean"),
         ("B1", "b1", "for every n with an entry, >= 48 held-out questions and a detectable held-out bias "
                      "(permutation test on the raw rows, Holm): the signed held-out deviation along the fitted "
                      "prior's direction falls (paired cluster-bootstrap CI of debiased - raw wholly below 0, and "
                      "|debiased| < |raw|); max-|.| is a reported point check. Harm leg: on the other such n the "
                      "debiased rows do not show both a detectable bias and a debiased max-|.| above the raw one"),
         ("B2", "b2", "held-out rotation consistency: debiased >= raw - 1 pp at the lower bound of the paired CI"),
         ("B3", "b3", "identical upstream call sequence with and without the profile; added CPU < 1 ms per readout"),
         ("B4", "b4", "the guard refuses (exit 2, for the load's reason) under a resident 300-token prompt and under a "
                      "continuous decode, and passes (exit 0) once the load is removed and the slots erased"),
         ("B5", "b5", "no harm: on the held-out rows of n with an entry (T = 1), accuracy falls by at most 0.5 pp "
                      "and NLL rises by at most 0.002")]


def _verdict(g: dict) -> str:
    if g.get("error"):
        return "ERROR"
    return {True: "PASS", False: "FAIL"}.get(g.get("pass"), "NOT RUN")


def _result(key: str, g: dict) -> str:
    if g.get("error"):
        return g["error"]
    if g.get("note"):
        return g["note"]
    if key == "b1":
        cells = []
        for n, v in g["per_n"].items():
            if v["tested"]:
                (sr, sd), (lo, hi) = v["signed"], v["signed_diff_ci"]
                cells.append(f"n={n}: signed {sr:+.4f} -> {sd:+.4f}, CI [{lo:+.4f}, {hi:+.4f}]; max-|.| "
                             f"{_f(v['raw'])} -> {_f(v['debiased'])}")
            else:
                harm = ("" if "harm" not in v else
                        f"; HARM: the correction created a detectable bias (p_holm {v['harm_p_holm']:.2g})"
                        if v["harm"] else f"; harm leg clear (p_holm {v['harm_p_holm']:.2g})")
                cells.append(f"n={n}: {v['note']} ({v['questions']} questions){harm}")
        return "; ".join(cells) or "no entry"
    if key == "b2":
        flips = g.get("flips") or []
        broke = sum(f["flip"] == "broke" for f in flips)
        return (f"{_f(g['raw'])} -> {_f(g['debiased'])}, CI [{_f(g['diff_ci'][0])}, {_f(g['diff_ci'][1])}]; "
                f"{broke} questions broke, {len(flips) - broke} mended")
    if key == "b3":
        return (f"{g.get('calls')} calls, identical {g.get('identical')}, {g.get('applied')} readouts applied, "
                f"added {_f(g.get('added_ms'), '.4f')} ms")
    if key == "b5":
        return f"accuracy {g['d_accuracy']:+.4f}, NLL {g['d_nll']:+.4f} ({g['rows']} rows)"
    if key == "b4":
        parts = []
        for x in "abc":
            why = g.get(x + "_failed")
            parts.append(f"({x}) {g[x + '_rc']}" + (f" [{', '.join(why)}]" if why else ""))
        return ", ".join(parts)
    return f"exit {g.get('rc')}, state {g.get('state')}"


def render(s: dict) -> str:
    out = ["# callo v0.3: position-bias profile acceptance bench", "",
           f"Run {s.get('started')} to {s.get('finished')} on `{s.get('llama_server')}`. The gates are `GATES` in "
           "`bench/report_v03_bias.py`, judged by `bench/acceptance_v03_bias.py` and its pre-registered verdict by "
           "`bench/bias_analysis.py` (`prereg_verdict`); the pre-registration is in the development history. A FAIL "
           "means the feature is wrong, an ERROR that the bench could not judge it.", ""]
    if s.get("aborted"):
        out[2:2] = [f"**ABORTED:** {s['aborted']}. The models below are the ones that finished.", ""]
    pre = s.get("prereg") or {}
    if pre:
        out += [f"**Pre-registered verdict: {pre.get('verdict', 'n/a')}** (the pre-registration's sha256 "
                f"`{(pre.get('sha256') or '')[:12]}`, as committed in C, before its `Pinned commit C:` line was "
                f"appended). B1 tested at: "
                f"{', '.join(pre.get('b1_tested') or []) or 'none'}."
                + (f" Reasons: {'; '.join(pre['reasons'])}." if pre.get("reasons") else ""), ""]
    for role, v in (s.get("data") or {}).items():
        out.append(f"- {role} set: `{v['path']}` (sha256 `{v['sha256'][:12]}`)")
    if s.get("data"):
        out.append("")
    for name, m in s.get("models", {}).items():
        out += [f"## {name}", ""]
        if "error" in m:
            out += [f"Did not complete: `{m['error']}`", ""]
            continue
        out += [f"`{m.get('model')}` {' '.join(m.get('flags') or []) or '(default launch)'}, build "
                f"`{m.get('build_info')}`, kv `{m.get('kv')}`. Entries kept: "
                f"{', '.join(m.get('entries') or []) or 'none'}; dropped: {m.get('dropped') or 'none'}.", ""]
        if m.get("aborted"):
            out += [f"Stopped early: {m['aborted']}.", ""]
        out += ["| gate | pass condition | result | verdict |", "|---|---|---|---|"]
        for label, key, cond in GATES:
            g = m.get(key)
            if g is None:
                out.append(f"| {label} | {cond} | not run on this model | NOT RUN |")
            else:
                out.append(f"| {label} | {cond} | {_result(key, g)} | {_verdict(g)} |")
        out.append("")
        flips = (m.get("b2") or {}).get("flips") or []
        if flips:
            out += ["B2's discordant questions (descriptive; margin = smallest top-2 gap over the rotations, raw -> "
                    "debiased):", ""]
            out += [f"- `{f['q']}` (n={f['n']}, {f['source']}): {f['flip']}, margin {f['margin'][0]:.3f} -> "
                    f"{f['margin'][1]:.3f}" for f in flips]
            out.append("")
        if m.get("reads"):
            out.append("Held-out reads: " + "; ".join(
                f"{run} exit {r['rc']}, {r['readouts']} readouts"
                + (" (resumed once)" if r.get("attempts", 1) > 1 else "") for run, r in m["reads"].items()) + ".")
            out.append("")
        lab = m.get("labelled") or {}
        for t, label in (("T1", "T = 1"), ("Tdev", "T refitted on public-pool-v6 calibration")):
            for kind, v in (lab.get(t) or {}).items():
                out.append(f"- held-out {kind}, {label}: accuracy {_f(v['accuracy'])}, NLL {_f(v['nll'], '.4f')}, "
                           f"ECE10 {_f(v['ece10'], '.4f')} ({v['rows']} rows)"
                           + (f", T {v['T']:.3f}" if "T" in v else ""))
        if m.get("fit_error"):
            out.append(f"- {m['fit_error']}")
        for where, by_source in (m.get("floored_rates") or {}).items():         # {"fit"|"eval": {source: {n: ..}}}
            for src, per in by_source.items():
                out.append(f"- floored-option rate, {where} {src}: " + ", ".join(
                    f"n={n} {v['floored']}/{v['questions']} ({v['rate']:.0%})" for n, v in per.items()))
        out.append("")
    g0 = s.get("g0_frames") or {}
    out += ["## G0 frames, raw vs debiased", "", f"{_verdict(g0)}: {g0.get('note', 'not run')}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default="docs/bench/v0.3-bias.md")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)       # a fresh clone has no docs/bench/
    out.write_text(render(json.loads(Path(a.summary).read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
