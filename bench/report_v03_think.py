"""Render docs/bench/v0.3-think.md from the v0.3 think-budgets bench's summary.json. Numbers come from the file, never
by hand; the joint `callo compare` report is embedded as the bench wrote it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bench.report_v02 import _f  # one "n/a"-aware number format for every bench report
from callo.calibrate_think import tails_line
from callo.think import nearest_rank

GATES = [
    ("T1", "t1", "cut = generation: per (thinking model, budget < 1024) with >= 24 cut questions (n_cut) and every one "
                 "of min(n_cut, 48) audited, the real generation's tokens equal the 1024 run's cut on >= n - 2; on the "
                 "identical ones argmax agrees on >= m - 1 and median |Δp| <= 1e-3. Required: qwen9b@256, qwen9b@512 and "
                 "tiel@256 (tiel with < 24 cuts: not testable, verified on one hybrid only). MiniCPM5: max |Δp| against "
                 "1e-6, descriptive, never gated. Amended post-hoc on 2026-10-08 (a deviation from the pre-registered gate): on a "
                 "MoE hybrid (GGUF expert_count > 0 with SSM layers) a pair is judged on trace identity, argmax >= m - 1 "
                 "and median |Δp| <= 1e-2, the original 1e-3 outcome still reported; dense-FFN models keep the original "
                 "criterion"),
    ("T2", "t2", "the latency model: on the T1-gated pairs, over identical-trace cuts, median |relative error| <= 10 % "
                 "and the p50s within 10 %"),
    ("T3", "t3", "temperature: for every budget > 0, cross-fitted NLL at T_B <= NLL at T = 1 + 0.002"),
    ("T4", "t4", "serve contract: 422 (think_budget \"x\", 40000, over --think-max, with an image, on gemma-e4b), 413 (a "
                 "budget past n_ctx on a -c 4096 launch), 403 (the probe from a non-loopback address), 200 otherwise; "
                 "X-Callo-Timings think fields present only at B > 0; a "
                 "client disconnect 1 s into a 1024 generation leaves no processing slot within 2 s; the think self-test "
                 "passes"),
    ("T5", "t5", "the guard: calibrate --think-budgets refuses (exit 2) under a resident 300-token prompt in a "
                 "non-reserved slot under -kvu and under a continuous agent decode; exit 0 once the load is removed and "
                 "the slots erased"),
    ("T6", "t6", "budget-0-only model: gemma-e4b yields a budget-0-only profile (unsupported_think_format), its own run "
                 "done in run.json, the joint report notes \"thinks in a format callo cannot read\" and lists its point"),
    ("T7", "t7", "menu isolation: Run 1 into a new DIR with qwen9b's path missing: tiel and gemma-e4b get profiles, "
                 "run.json names the failure, exit 1; --resume with the path fixed runs only qwen9b"),
    ("T8", "t8", "compare: the joint report joins Run 1's three profiles (exit 0); adding Run 2's profile (another data "
                 "sha) is refused (exit 2, \"different data\")"),
    ("T9", "t9", "re-prefill: >= 99 % of the deliberate readouts read at n_gen (calibrate: from the raw phase-B rows; served: "
                 "the natural closes' X-Callo-Timings read_prompt_n, none: not testable) have a first-pass prompt_n <= 4; "
                 "text_fallback = 0 on every thinking model"),
    ("T10", "t10", "served vs profile latency: the first 48 measured questions through /v1/systemone at the pick budget: "
                   "served p50 (engine time) within 15 % of the profile's p50 over the same questions (fewer than 48 "
                   "shared: not testable)"),
]


def _passes(g: Any) -> list[bool | None]:
    """Every `pass` under a gate entry: its own, or one per model when the entry is a per-model dict."""
    if not isinstance(g, dict):
        return []
    if "pass" in g:
        return [g["pass"]]
    return [v.get("pass") for v in g.values() if isinstance(v, dict) and "pass" in v]


def _has_error(g: Any) -> bool:
    """An `error` anywhere in a gate entry (the gate's own, a model's, a budget's; t9's under `calibrate` / `served`),
    the dense model's descriptive T1 entries excepted (they never count)."""
    if not isinstance(g, dict):
        return False
    return bool(g.get("error")) or any(_has_error(v) for k, v in g.items() if k != "descriptive")


def _verdict(g: Any) -> str:
    """NOT RUN (no entry), FAIL (a measured failure stands), ERROR (the bench could not judge the gate, or a model or
    pair of it), n/a (no model met the precondition), else PASS."""
    if not isinstance(g, dict):
        return "NOT RUN"
    passes = _passes(g)
    if any(p is False for p in passes):
        return "FAIL"
    if _has_error(g):
        return "ERROR"
    if not passes or all(p is None for p in passes):
        return "n/a"
    return "PASS"


def _t1_rows(g: dict, field: str) -> str:
    """T1 / T2 from `t1_t2_gates`' entry: the required pairs, each model's budgets, the notes, the dense model's
    descriptive line (T1 only). The gate's own `error` (the required pairs' problems, printed first by `_result`)
    already says why a pair is not gated, so a reason it holds is not repeated here: each message appears once."""
    top = str(g.get("error") or "")
    required = list(g.get("required") or {})

    def said(pair: str, msg: str) -> bool:
        """The gate's error already gives `msg` as the reason the required pair `pair` is not gated."""
        return bool(top) and f"required pair {pair} not gated: {msg}" in top

    cells = []
    if required:
        cells.append("required: " + ", ".join(f"{k} not gated" if v.startswith("not gated: ") and said(k, v[len("not gated: "):])
                                              else f"{k} {v}" for k, v in g["required"].items()))
    models = g.get("models") or {}
    for model, r in [*models.items(), *((g.get("descriptive") or {}).items() if field == "t1" else ())]:
        if r.get("error") and not any(said(k, r["error"]) for k in required if k.startswith(f"{model}@")):
            cells.append(f"{model}: {r['error']}")
        cells += [f"{model}: {n}" for n in r.get("notes") or []]
        for b, e in (r.get("budgets") or {}).items():
            if e.get("error"):
                if not said(f"{model}@{b}", e["error"]):
                    cells.append(f"{model}@{b}: {e['error']}")
            elif e.get("note") and e.get("t1") is None:
                if not said(f"{model}@{b}", e["note"]):
                    cells.append(f"{model}@{b}: {e['note']}")
            elif field == "t1":
                cells.append(f"{model}@{b}: identical {e['identical']}/{e['n']}, argmax {e['argmax_agree']}/{e['identical']}, "
                             f"|Δp| median {_f(e.get('dp_median'), '.2e')} max {_f(e.get('dp_max'), '.2e')}"
                             + (f": {e['amended']}" if e.get("amended") else ""))
            else:
                cells.append(f"{model}@{b}: median error {_f(e.get('lat_err_median'), '.3f')}, p50 model "
                             f"{_f(e.get('p50_model_ms'), '.0f')} vs real {_f(e.get('p50_real_ms'), '.0f')} ms")
    cells += list(g.get("notes") or [])
    return "; ".join(cells) or "no audited budget"


def _body(key: str, g: dict) -> str | None:
    """The gate's own detail, or None when the entry lacks the keys it is built from (an entry that is only an
    `error`, say)."""
    models = {m: v for m, v in g.items() if isinstance(v, dict)}       # the per-model gates (T3, T10)
    if key in ("t1", "t2"):
        return _t1_rows(g, key) if "models" in g or "required" in g else None
    if key == "t3":
        return "; ".join(f"{m}: {r['error']}" if r.get("error") else
                         "; ".join(f"{m}@{b}: NLL@1 {_f(v['nll_1'], '.4f')} -> NLL@T {_f(v['nll_T'], '.4f')} (T "
                                   f"{_f(v['T'], '.2f')})" for b, v in (r.get("budgets") or {}).items()) or f"{m}: no budget > 0"
                         for m, r in models.items()) or None
    if key == "t4":
        if not isinstance(g.get("cases"), dict):
            return None
        cases = ", ".join(f"{k} {v['got']}" + ("" if v["pass"] else f" (want {v['expected']})") for k, v in g["cases"].items())
        return (f"{cases or 'no status measured'}; X-Callo-Timings think fields only at B > 0: {g.get('timings_ok')}; "
                f"disconnect clears slot: {g.get('disconnect_clear')}; think self-test: {g.get('self_test')}"
                + (f"; not testable: {', '.join(g['not_testable'])}" if g.get("not_testable") and not g.get("error")
                   else ""))       # t4's error already lists what was not testable
    if key == "t5":
        if not isinstance(g.get("rcs"), dict):
            return None
        failed = g.get("failed") or {}
        return ", ".join(f"({x}) exit {g['rcs'].get(x)}" + (f" [{', '.join(failed.get(x, []))}]" if failed.get(x) else "")
                         for x in "abc")
    if key == "t6":
        return (f"capable {g.get('capable')}, reason {g.get('reason')}, budget 0 only {g.get('budget_0_only')}, its run "
                f"{g.get('status')} (Run 1 exit {g.get('rc')}), report note {g.get('report_note')}, on the joint report's "
                f"points {g.get('on_frontier')}") if "capable" in g else None
    if key == "t7":
        return (f"broken run: {g.get('broken')} (exit {g.get('broken_rc')}); resumed: {g.get('resumed')} (exit "
                f"{g.get('resumed_rc')}); only the failed model reran: {g.get('only_the_failed_model_reran')}"
                + (f"; problems: {'; '.join(g['problems'])}" if g.get("problems") else "")) if "broken" in g else None
    if key == "t8":
        return (f"join exit {g.get('join_rc')}, refusal exit {g.get('refuse_rc')}"
                + (f" ({g['refuse_reason']})" if g.get("refuse_reason") else "")) if "join_rc" in g else None
    if key == "t9":
        if "calibrate" not in g and "served" not in g:
            return None
        cal = ", ".join(f"{m} ERROR ({v['error']})" if v.get("error") else
                        f"{m} {v['at_n_gen_ok']}/{v['n']} readouts at n_gen with prompt_n <= 4, {v['text_fallback']} fallbacks"
                        + (f" ({v['note']})" if v.get("note") else "")
                        for m, v in (g.get("calibrate") or {}).items())
        srv = ", ".join((f"{m} ERROR ({v['error']})" if v.get("error") else f"{m} {v['at_n_gen_ok']}/{v['n']}")
                        + (f" ({v['note']})" if v.get("note") else "")
                        for m, v in (g.get("served") or {}).items())
        return f"calibrate: {cal or 'none'}; served: {srv or 'none'}"
    if key == "t10":
        def at(m: str, v: dict) -> str:                                # why the session served that budget
            why = [v["budget_note"]] if v.get("budget_note") else []
            why += [f"picked on {v['pick_latency']}"] if v.get("pick_latency") else []   # under which rule
            return f"{m}@{v.get('budget')}" + (f" ({', '.join(why)})" if why else "")

        def beside(v: dict) -> str:                                    # reported, never gated
            if "served_mean_ms" not in v:
                return ""
            return (f"; beside it, not gated: mean {_f(v.get('served_mean_ms'), '.0f')} vs "
                    f"{_f(v.get('profile_mean_ms'), '.0f')} ms (ratio {_f(v.get('mean_ratio'), '.3f')}), p95 "
                    f"{_f(v.get('served_p95_ms'), '.0f')} vs {_f(v.get('profile_p95_ms'), '.0f')} ms (from "
                    f"{v.get('p95_from')} of {v.get('n')}), max {_f(v.get('served_max_ms'), '.0f')} vs "
                    f"{_f(v.get('profile_max_ms'), '.0f')} ms")
        return "; ".join(f"{at(m, v)}: {v['error']}" if v.get("error") and v.get("budget") is not None else
                         f"{m}: {v['error']}" if v.get("error") else
                         f"{at(m, v)}: served p50 {_f(v.get('served_p50_ms'), '.0f')} vs profile "
                         f"{_f(v.get('profile_p50_ms'), '.0f')} ms over {v.get('n')} questions (error "
                         f"{_f(v.get('rel_err'), '.3f')}){beside(v)}" for m, v in models.items()) or None
    return json.dumps(g, default=str)[:300]


def _result(key: str, g: Any) -> str:
    """The gate's result cell: its top-level `error` FIRST (an entry that is only an error renders its message for
    every gate), then the gate's own detail when the entry has it."""
    if not isinstance(g, dict):
        return "not run"
    body = _body(key, g)
    if g.get("error"):
        return str(g["error"]) + (f"; {body}" if body else "")
    return body or {"t1": "no audited budget", "t2": "no audited budget", "t3": "no budget"}.get(key, "not run")


def _descriptive(d: dict) -> list[str]:
    out = ["| model | sampler | acc per budget | Δacc(B − 0) [95 % CI] | forced | think mean | n_cut | pick | overhead | drift | decode |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in d.items():
        acc = ", ".join(f"{b}: {_f(a, '.3f')}" for b, a in v["acc"].items())
        dacc = ", ".join(f"{b}: {_f(ci[0], '+.3f')} [{_f(ci[1], '+.3f')}, {_f(ci[2], '+.3f')}]" if ci else f"{b}: n/a"
                         for b, ci in v["dacc"].items())
        forced = ", ".join(f"{b}: {_f(x, '.0%')}" for b, x in v["forced"].items())
        think = ", ".join(f"{b}: {_f(x, '.0f')}" for b, x in v["think_mean"].items())
        ncut = ", ".join(f"{b}: {x}" for b, x in v["n_cut"].items())
        drift = v.get("drift") or {}
        therm = ", ".join(f"{t.get('at')} {t.get('limit') if t.get('limit') is not None else 'n/a'}" for t in drift.get("therm") or [])
        out.append(f"| {name} | {v.get('sampler')} | {acc} | {dacc or 'n/a'} | {forced or 'n/a'} | {think or 'n/a'} | {ncut or 'n/a'} | "
                   f"{v.get('pick')} | {_f(v.get('overhead_s'), '.1f')} s | ratio {_f(drift.get('ratio'), '.2f')}, "
                   f"{'FLAGGED' if drift.get('flagged') else 'none'}{'; therm ' + therm if therm else ''} | "
                   f"{_f(v.get('decode_tok_s'), '.0f')} tok/s |")
    return out


def _sec(ms: Any) -> str:
    return _f(None if ms is None else ms / 1000, ".2f") + ("s" if ms is not None else "")


def _latencies(d: dict) -> list[str]:
    """Per (model, budget) the five latency statistics, p99 `= max` where its rank is the last question, then
    what the tails rest on."""
    out = ["| model | budget | n | mean | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|---|---|"]
    ns: list[int] = []
    for name, v in d.items():
        for b, e in (v.get("lat") or {}).items():
            if e.get("unreadable"):
                out.append(f"| {name} | {b} | {e.get('n') if e.get('n') is not None else 'n/a'} | n/a | n/a | n/a | n/a | n/a |")
                continue
            n = int(e["n"])
            p99 = "= max" if nearest_rank(n, 0.99) == n else _sec(e["p99"])
            out.append(f"| {name} | {b} | {n} | {_sec(e['mean'])} | {_sec(e['p50'])} | {_sec(e['p95'])} | {p99} | "
                       f"{_sec(e['max'])} |")
            if n not in ns:
                ns.append(n)
    return [*out, "", *(tails_line(n) for n in ns)]


def _pct(v: Any) -> str:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return "n/a"
    return f"{100 * v[0]:+.1f}pp [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"


def _repicks(d: dict) -> list[str]:
    """The pick each profile recorded (the p50 rule of the run) beside `callo compare`'s re-pick under the mean
    default, each with its target and the paired Δacc against it (pick − target); the discordant counts behind the
    re-pick's Δacc (the percentile bootstrap is anti-conservative at a handful of discordant questions)."""
    out = ["The profiles were written under the p50 rule, and the bench lists the default pick as descriptive: no "
           "gate depends on it. The re-pick is `callo compare PROFILE` under the mean default. Δacc is the "
           "pick minus its target; discordant counts the questions the pick answers right and its target wrong, then "
           "the reverse.", "",
           "| model | recorded pick (statistic) | its target | Δacc vs it | mean re-pick | its target | Δacc vs it | "
           "discordant |", "|---|---|---|---|---|---|---|---|"]
    for name, v in d.items():
        r = v.get("repick")
        if not isinstance(r, dict):
            continue
        disc = r.get("discordant")
        out.append(f"| {name} | {r.get('recorded')} ({r.get('recorded_on') or 'n/a'}) | {r.get('recorded_best')} | "
                   f"{_pct(r.get('recorded_dacc_best'))} | {r.get('mean')} | {r.get('mean_best')} | "
                   f"{_pct(r.get('mean_dacc_best'))} | {f'{disc[0]} / {disc[1]}' if disc else 'n/a'} |")
    return out


def _by_group(d: dict) -> list[str]:
    """Δacc(B − 0) per source group (composition / date-score / control), one row per model."""
    out = ["| model | group | n | Δacc(B − 0) [95 % CI] per budget |", "|---|---|---|---|"]
    for name, v in d.items():
        for g, per in (v.get("dacc_by_group") or {}).items():
            cells = ", ".join(f"{b}: {_f(ci[0], '+.3f')} [{_f(ci[1], '+.3f')}, {_f(ci[2], '+.3f')}]"
                              for b, ci in per.items() if b != "n")
            out.append(f"| {name} | {g} | {per.get('n')} | {cells or 'n/a'} |")
    return out


def _pairs(d: dict) -> list[str]:
    """The paired Δacc(first − second) per budget between two profiles of the same questions (tiel at its GGUF's 1.0 vs
    tiel-fixed at 0.6)."""
    out = ["Δacc is the first model named minus the second (point − reference): `tiel vs tiel-fixed` reads "
           "Δacc(tiel − tiel-fixed), its GGUF's sampler minus `fixed`.", "",
           "| pair | n | Δacc(first − second) [95 % CI] per budget |", "|---|---|---|"]
    for name, per in d.items():
        cells = ", ".join(f"{b}: {_f(ci[0], '+.3f')} [{_f(ci[1], '+.3f')}, {_f(ci[2], '+.3f')}]" for b, ci in per.items() if b != "n")
        out.append(f"| {name} | {per.get('n')} | {cells or 'n/a'} |")
    return out


def version_line(version: str) -> str:
    """The line of `llama-server --version` that names the build (`version: N (sha)`), else its first line."""
    lines = [ln.strip() for ln in str(version or "").splitlines() if ln.strip()]
    return next((ln for ln in lines if ln.lower().startswith("version")), lines[0] if lines else "")


def _provenance(s: dict) -> list[str]:
    """Per part: the callo commit (12 characters, `dirty` when the tracked tree differed) and the llama-server version
    line that produced it; a flag when the parts come from different commits; the data files' sha256."""
    prov = s.get("provenance") or {}
    out: list[str] = []
    if prov:
        out += ["| part | callo commit | llama-server |", "|---|---|---|"]
        for part, p in prov.items():
            commit = (str(p.get("commit"))[:12] if p.get("commit") else "unknown") + (" dirty" if p.get("dirty") else "")
            build = version_line(p.get("llama_server_version") or "").replace("|", "\\|") or "unknown"
            out.append(f"| {part} | {commit} | {build} |")
        commits = sorted({str(p.get("commit")) for p in prov.values()})
        if len(commits) > 1:
            out += ["", f"**Mixed commits:** the parts ran on {len(commits)} different commits "
                        f"({', '.join(c[:12] for c in commits)})."]
        out.append("")
    d = s.get("data") or {}
    out.append(f"Data sha256: test `{d.get('sha256')}` (selected from `{d.get('source_sha256')}`), Run 2 subset "
               f"`{d.get('run2_sha256')}`.")
    rr = s.get("rerendered")
    if rr:
        sha = str(rr.get("commit") or "unknown")[:12] if isinstance(rr, dict) else str(rr)
        dirty = " with uncommitted changes" if isinstance(rr, dict) and rr.get("dirty") else ""
        out += ["", f"Re-rendered offline by `--rerender` at callo {sha}{dirty}; T10 (its p50 verdict and the "
                    "served-vs-profile mean/p95/max), the latency table, the mean re-pick table and the descriptive tables "
                    "are recomputed by that code from the recorded run's summary.json and profiles; every other gate "
                    "verdict renders as the run recorded it."]
    return out


def render(s: dict) -> str:
    data = s.get("data") or {}
    out = ["# callo v0.3: think budgets acceptance bench", "",
           f"Run {s.get('started')} to {s.get('finished')} on `{s.get('llama_server')}`. Data: `{data.get('path')}` "
           f"({data.get('questions')} questions from sources {', '.join(data.get('sources') or [])}; Run 2 on a "
           f"{data.get('run2_questions')}-question seeded subset). The gates are `GATES` in `bench/report_v03_think.py`, "
           "judged by `bench/acceptance_v03_think.py`; their pre-registration is in the development history. A FAIL "
           "means the feature is wrong, an ERROR that the "
           "bench could not judge it, n/a that the gate's precondition (for example 24 audited cuts) was not met.", ""]
    if s.get("aborted"):
        out[2:2] = [f"**ABORTED:** {s['aborted']}. The parts below are the ones that finished.", ""]
    runs = s.get("runs") or {}
    if runs:
        out += ["| run | exit | models |", "|---|---|---|"]
        for name, r in runs.items():
            models = ", ".join(f"{m} {rec.get('status')}" + (f" ({rec.get('reason')})" if rec.get("reason") else "")
                               for m, rec in ((r.get("run_json") or {}).get("models") or {}).items())
            out.append(f"| {name} | {r.get('rc')} | {models or 'n/a'} |")
        out.append("")
    gates = s.get("gates") or {}
    out += ["| gate | pass when | result | verdict |", "|---|---|---|---|"]
    for name, key, cond in GATES:
        g = gates.get(key)
        out.append(f"| {name} | {cond} | {_result(key, g)} | {_verdict(g)} |")
    out.append("")
    if s.get("descriptive"):
        out += ["## Descriptive results (no gate)", "", *_descriptive(s["descriptive"]), ""]
        if any(v.get("lat") for v in s["descriptive"].values()):
            out += ["### Latency per budget: mean, p50, p95, p99, max", "", *_latencies(s["descriptive"]), ""]
        if any(isinstance(v.get("repick"), dict) for v in s["descriptive"].values()):
            out += ["### The pick: as recorded (p50) and the mean re-pick", "", *_repicks(s["descriptive"]), ""]
        if any(v.get("dacc_by_group") for v in s["descriptive"].values()):
            out += ["### Δacc(B − 0) per source group", "", *_by_group(s["descriptive"]), ""]
    if s.get("descriptive_pairs"):
        out += ["### The same model under two samplers (paired, same questions)", "", *_pairs(s["descriptive_pairs"]), ""]
    if s.get("joint_md"):
        out += ["## The joint frontier (`callo compare`, Runs 1 and 3)", "", s["joint_md"].rstrip("\n"), ""]
    if s.get("errors"):
        out += ["## Errors", "", *(f"- {k}: {v}" for k, v in s["errors"].items()), ""]
    if s.get("warnings"):
        out += ["## Warnings (at the last start)", "", *(f"- {w}" for w in s["warnings"]), ""]
    if s.get("provenance") or s.get("data") or s.get("rerendered"):
        out += ["## Provenance", "", *_provenance(s), ""]
    out += ["## Files", "", f"Raw output: `{s.get('out')}` (per run `run.json`, `<name>.think.json`, `<name>.raw.jsonl`, "
            f"`<name>.serve.log`, `report.md`, `report.json`; `summary.json`; the serve-path sessions' callo logs).", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "bench" / "v0.3-think.md"))
    a = ap.parse_args(argv)
    text = render(json.loads(Path(a.summary).read_text()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
