"""The v0.3 early-exit acceptance bench: the serve side's live gates on held-out questions.

- E1, the probe cost: on a hybrid model the chunk after a probe evaluates at most 2 tokens on >= 95 % of probes; the
  probe readout's p50 is <= 1.5 x the profile's `latency.read_forced_ms` p50 (recorded by `--deadlines`' pre-pass).
- E2, convergence holds out of sample: greedy, z = .95; one trace per question through /callo/think-probe's live mode
  serves both readings, and the greedy profile's served rule (stop_index over its probes) stops it; the share of stops
  whose answer equals the end answer on the same trace is >= z - 0.03, with its Clopper-Pearson interval. A served
  early-exit trace is the K-aligned live-mode trace of the same body cut at the stop, so the live trace is the served
  one up to the stop and the end reading is that same trace's.
- E3, the trade holds when served: the model's own sampler, early exit and the end reading alternating per item; the
  served Δacc has a CI containing the profile's cross-fitted Δacc, and the profile's p50 latency saving lies in the
  served saving's CI. The two arms think DIFFERENT traces of one question (the early-exit trace is chunked and each
  sampled chunk is reseeded; the end arm is one plain budget request): the comparison is paired by question, never by
  trace, and the question-level cluster bootstrap carries both traces' noise in each question's difference.

Calibration reads the transfer set's TEST split (the S1 selection, 376 questions); serving reads its DEVELOPMENT
split (the same selection, 336 questions), which no calibration reads (`overlap` refuses a shared text). Models:
Tiel (MoE, hybrid, K 32) and MiniCPM5 (K 64), each calibrated twice (its own sampler; greedy). GPU, one model server
at a time. Parts: calib, serve, gates; `--models` and `--label` make a labelled subset run (the z .9 run);
stop with `kill -TERM $(cat OUT/run.pid)`. A part run for a subset of the models replaces only those models' entries.
A labelled run reuses the unlabelled run's greedy sessions (same profile file): E2's live rows do not depend on z.

Reading E3: the held-out development split has no date-score group (test, the calibration file, has 40 of its 376),
so the question mix differs from the profile's; an E3 latency miss on its own may reflect that mix.

`--gate`, the early-exit gate's run: a labelled run (label `gate`, the only one it takes) on Tiel only. Its calib part
is Tiel's own-sampler calibration with the gate measured (`--converge-probe` reads it) into its own directory,
`calib/tiel-gate`, so the unlabelled run's is never overwritten; when that directory holds no raw file yet it starts
from a copy of the unlabelled run's Tiel raw file, and the resume adds only the gate reads (`--gate-fresh` calibrates
from scratch instead). Its serve part serves that profile: early exit at z with `think_converge_gate: true` and the
end reading alternate per item as in E3; no greedy session. Its gates part reports each arm's accuracy, Δacc vs the end
reading (the cluster bootstrap), each arm's p50 and mean latency, the p50 saving with its CI, the gate's skip share and
read ms, and per-source accuracy. The pre-declared verdict: "holds" iff the Δacc CI's upper end is >= 0 AND the p50
saving CI's lower end is > 0; else "does not hold". Beside it, profile vs served for the gated point, as E3: the served
Δacc CI holds the profile's gated Δacc (`ee_gated[z].dacc_cap`), and the profile's gated p50 saving (the cap budget's
p50 minus the gated point's) lies in the served saving's CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import signal
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench import lib
from bench.acceptance_v02 import TIEL, _install_stop_handlers
from bench.acceptance_v03_deadline import _post, served_correct, session_provenance
from bench.acceptance_v03_think import (
    BUDGETS,
    CTX,
    LLAMA_SERVER,
    MINICPM,
    PARALLEL,
    SEED,
    TRANSFER_TEST,
    _calibrate,
    _foreign_servers,
    _join_flags,
    _pid_in,
    _start,
    gguf_traits,
    pid_owner,
    profile_launch,
    select_records,
)
from bench.report_v03_early_exit import render, verdict_lines
from callo.calibrate_think import Item, load_data, read_raw, reserve_reads
from callo.converge import Probe, beta_ppf, serve_rules, stop_index
from callo.deadline import percentile
from callo.errors import InvalidInput
from callo.think import ee_key, paired_bootstrap

DEV = lib.env_path(lib.DATA_ENV, "v4", "transfer-v4", "development.jsonl")   # the transfer set's development split
OUT_DEFAULT = lib.env_path(lib.OUT_ENV, "callo_v0.3", "early_exit")
Z = 0.95
E1_RESTORE, E1_READ, E2_SLACK = 0.95, 1.5, 0.03
MIN_PROBES = MIN_STOPS = 30
MIN_PAIRED = 0.90
MIN_TRACES = 200                # E2: usable live-mode traces (of 336 held-out questions), as E3's MIN_PAIRED
MIN_FORCED = 5                  # E1: forced reads behind the profile's read_forced_ms p50
WARMUP_N = 3
WARMUP_FIELDS = {"think_budget": 64, "think_converge": 0}     # warms the server, leaves E1's probe windows untouched
CALIB_DEADLINE = "2000"                                      # its reserve pre-pass records latency.read_forced_ms
CALIB_WARMUP = "9"              # the pre-pass reads warm-up questions 2..W: 8 forced reads (calibrate's default 3 gives
#                                 2); the warm-up questions are still measured in phases A and B
MODELS: dict[str, tuple[str, int]] = {"tiel": (TIEL, 32), "minicpm": (MINICPM, 64)}
PARTS = ("calib", "serve", "gates")
GATE_MODEL, GATE_LABEL, GATE_PARTS = "tiel", "gate", PARTS      # --gate: Tiel, its own label
GATE_RUN = "tiel-gate"          # --gate's calibration (Tiel, own sampler, the gate measured): its own directory
GATE_MAX_ERRORS = 0.10          # --gate: more failed gate reads than this share is no evidence for the gate
log = logging.getLogger("bench")


def runs(models: Sequence[str] | None = None) -> dict[str, tuple[str, int, bool]]:
    """The calibration runs of `models` (all by default): name -> (model, K, greedy)."""
    out: dict[str, tuple[str, int, bool]] = {}
    for name, (model, k) in MODELS.items():
        if models is None or name in models:
            out[name], out[f"{name}-greedy"] = (model, k, False), (model, k, True)
    return out


def spec(run: str) -> tuple[str, int, bool]:
    """A calibration run's (model, K, greedy): one of `runs()`, or `--gate`'s GATE_RUN (Tiel's own sampler)."""
    return {**runs(), GATE_RUN: runs()[GATE_MODEL]}[run]


def calib_argv(run: str, out: Path, data: Path) -> list[str]:
    """`callo calibrate` for one run in launch mode, callo as shipped (calibrate's default cache); a DIR holding a
    run.json, or (GATE_RUN) the raw file `seed_gate_calib` copied in, is resumed. `--converge-probe` measures early
    exit's gate too (callo after the gate's calibration step)."""
    model, k, greedy = spec(run)
    d = out / "calib" / run
    resume = (d / "run.json").exists() or (run == GATE_RUN and any(d.glob("*.raw.jsonl")))
    return ["calibrate", "--think-budgets", BUDGETS, "--converge-probe", str(k), "--deadlines", CALIB_DEADLINE,
            "--warmup", CALIB_WARMUP, "--audit", "0", *(["--think-sampling", "temp=0"] if greedy else []),
            "--data", str(data), "--model", model,
            "--llama-server", LLAMA_SERVER, "--ctx", str(CTX), "--parallel", str(PARALLEL), "--seed", str(SEED),
            "--out", str(d), *(["--resume"] if resume else [])]


def seed_gate_calib(out: Path) -> Path | None:
    """`--gate`'s calibration, unless `--gate-fresh`: when its directory holds no raw file, the unlabelled run's Tiel
    raw file (same data, settings and model) is copied in under its own name, so the resume reads only the gate (a raw
    file from before the gate has every other cell). Returns the copy, or None (nothing to copy, or a raw file is
    already there: a resume of its own). Several raw files to copy from is a RuntimeError (the part fails): the
    copy must be the unlabelled Tiel run's alone."""
    d = out / "calib" / GATE_RUN
    if any(d.glob("*.raw.jsonl")):
        return None
    found = sorted((out / "calib" / GATE_MODEL).glob("*.raw.jsonl"))
    if not found:
        return None
    if len(found) > 1:
        raise RuntimeError(f"cannot seed {d}: {found[0].parent} holds {len(found)} raw files "
                           f"({', '.join(p.name for p in found)}); which one is Tiel's is not clear")
    d.mkdir(parents=True, exist_ok=True)
    log.info("calib %s: copying %s into %s", GATE_RUN, found[0], d)
    return Path(shutil.copyfile(found[0], d / found[0].name))


def paths(out: Path, label: str | None) -> tuple[Path, Path]:
    """(the summary, the report) of a run; a labelled run (the z .9 one) writes its own pair."""
    tag = f"-{label}" if label else ""
    return out / f"summary{tag}.json", out / f"v0.3-early-exit{tag}.md"


def _profile_path(out: Path, run: str) -> Path | None:
    found = sorted((out / "calib" / run).glob("*.think.json"))
    return found[0] if found else None


def _profile(out: Path, run: str) -> dict:
    p = _profile_path(out, run)
    return json.loads(p.read_text()) if p is not None else {}


def k_problem(run: str, profile: Mapping[str, Any]) -> str | None:
    """Why a run's profile cannot stand for it: its `converge.K` is not the K the bench calibrates that model with
    (`MODELS`; a profile from another run left in its directory), else None."""
    k = spec(run)[1]
    got = (profile.get("converge") or {}).get("K") if isinstance(profile.get("converge"), Mapping) else None
    if got != k:
        return f"{run}'s profile has converge.K {got!r}, not the bench's K {k} for this model (another run's profile?)"
    return None


def read_forced_n(out: Path, run: str) -> int | None:
    """How many reads the run's `latency.read_forced_ms` rests on, counted from its raw file as `latency_block` takes
    them (the audit's forced reads, the reserve pre-pass's warm-up questions 2..W); None without a raw file."""
    found = sorted((out / "calib" / run).glob("*.raw.jsonl"))
    if not found:
        return None
    try:
        raw = read_raw(found[0])
    except InvalidInput:
        return None
    audit = sum(1 for r in raw.rows if r.get("phase") == "audit" and isinstance(r.get("read_ms"), (int, float))
                and (r.get("trace") or {}).get("close") == "forced")
    return audit + len(reserve_reads((raw.reserve or {}).get("reads") or []))


def overlap(a: Path, b: Path) -> int:
    """Records whose `_meta.text_sha256` is in both files: the held-out file must share none with calibration's."""
    def shas(p: Path) -> set:
        return {(json.loads(x).get("_meta") or {}).get("text_sha256") for x in p.read_text().splitlines()
                if x.strip()} - {None}
    return len(shas(a) & shas(b))


def chosen(item: Item, answer: Mapping[str, Any] | None) -> str | None:
    """The served answer's chosen option key (a noul's "1" / "0" at 0.5), None when unreadable."""
    if not isinstance(answer, Mapping):
        return None
    if answer.get("type") == "noul" and isinstance(answer.get("noul"), (int, float)):
        return "1" if float(answer["noul"]) > 0.5 else "0"
    keys, probs = list(item.question.keys), answer.get("probabilities")
    if not isinstance(probs, Mapping) or not all(k in probs for k in keys):
        return None
    return keys[max(range(len(keys)), key=lambda i: float(probs[keys[i]]))]


def serve_rows(url: str, items: Sequence[Item], fields: Mapping[str, Any]) -> list[dict]:
    """One /v1/systemone request per item, sequentially, with `fields` added; per item its status, X-Callo-Timings
    entry, chosen key, correctness, in-callo question ms (`ms`: the question's whole wall time, trace and readout),
    cluster and source (the bootstrap's strata)."""
    rows = []
    for it in items:
        status, headers, out = _post(f"{url}/v1/systemone", {**it.body, **fields})
        try:
            entries = json.loads(headers.get("x-callo-timings") or "[]") if status == 200 else []
        except ValueError:
            entries = []
        answer = (out.get("answers") or {}).get(it.qid) if status == 200 and isinstance(out, dict) else None
        entry = entries[0] if entries and isinstance(entries[0], dict) else None
        rows.append({"q": it.uid, "cluster": it.cluster, "source": it.source, "status": status, "entry": entry,
                     "choice": chosen(it, answer), "correct": served_correct(it, answer),
                     "ms": entry.get("ms") if entry else None})
    return rows


def serve_pairs(url: str, items: Sequence[Any], ee_fields: Mapping[str, Any],
                end_fields: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """E3's two runs, one request each per item, alternating which goes first (even items early exit first, odd items
    the end reading first), so a warm prefix cache or a drifting box favours neither. (early-exit rows, end rows)."""
    ee: list[dict] = []
    end: list[dict] = []
    for i, it in enumerate(items):
        order = ((ee, ee_fields), (end, end_fields)) if i % 2 == 0 else ((end, end_fields), (ee, ee_fields))
        for rows, fields in order:
            rows.extend(serve_rows(url, [it], fields))
    return ee, end


def probe_live_rows(url: str, items: Sequence[Item], *, cap: int, k: int, seed: int = SEED) -> list[dict]:
    """E2's evidence: one trace per item through /callo/think-probe's live mode (probe every K, no deadline, capped at
    `cap`; the served profile's sampler, greedy): every probe's k and probabilities and the final readout's, all at
    T = 1, so the stop and the end are read off the same trace."""
    rows = []
    for it in items:
        st, _, out = _post(f"{url}/callo/think-probe", {"request": it.body, "budgets": [cap], "seed": seed,
                                                         "live": {"deadline_ms": None, "probe_every": k}})
        lv = (((out or {}).get("questions") or {}).get(it.qid) or {}).get("live") \
            if st == 200 and isinstance(out, dict) else None
        if not isinstance(lv, dict) or not isinstance(lv.get("readout"), dict):
            rows.append({"q": it.uid, "status": st, "probes": None, "final": None})
            continue
        rows.append({"q": it.uid, "status": st, "close": lv.get("close"),
                     "probes": [[int(p["k"]), list(p["probs"])] for p in lv.get("probes") or []],
                     "final": list(lv["readout"]["probs"])})
    return rows


def run_calib(out: Path, test: Path, models: Sequence[str] | None, res: dict, *, gate: bool = False,
              fresh: bool = False) -> dict:
    """The calibration runs of `models`, filling `res` (the other models' entries are kept). As the think bench's
    `_run_calibrate`: a 130 is this bench's KeyboardInterrupt and a 2 (a refusal: the GPU rule, a bad flag) ends the
    part at once. Any other non-zero exit, or a run that wrote no profile or one of another K, is recorded and fails
    the part (RuntimeError, so `main` returns 1) once the remaining runs have had their turn.

    `gate` (`--gate`): GATE_RUN alone, seeded from the unlabelled run's raw file (`seed_gate_calib`) unless `fresh`;
    a profile without the gate measured (`converge.gate`: no gate read, an older callo) fails the part."""
    failed = []
    for run in ([GATE_RUN] if gate else runs(models)):
        seeded = seed_gate_calib(out) if gate and not fresh else None
        if seeded is not None:
            log.info("calib %s: starting from a copy of the unlabelled run's raw file (%s): the resume reads the gate "
                     "alone", run, seeded.name)
        rc, err = _calibrate(calib_argv(run, out, test))
        tail = err.strip().splitlines()[-1] if err.strip() else ""
        res[run] = {"rc": rc, "tail": tail, **({"seeded": str(seeded)} if seeded is not None else {})}
        if rc == 130:
            raise KeyboardInterrupt(f"calibrate {run} was interrupted")
        if rc == 2:
            raise RuntimeError(f"calibrate {run} refused (exit 2): {tail or 'no stderr'}")
        if rc != 0:
            failed.append(f"{run} (exit {rc}: {tail or 'no stderr'})")
            continue
        why = (f"{run} wrote no profile" if _profile_path(out, run) is None
               else k_problem(run, _profile(out, run)))
        if why is None and gate and not isinstance((_profile(out, run).get("converge") or {}).get("gate"), Mapping):
            why = f"{run}'s profile has no gate measured (converge.gate): calibrated by a callo before the gate?"
        if why:
            res[run]["error"] = why
            failed.append(why)
    if failed:
        raise RuntimeError("calibrate failed: " + "; ".join(failed))
    return res


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_serve(out: Path, dev: Sequence[Item], res: dict, *, z: float, models: Sequence[str] | None,
              reuse: Mapping[str, Any] | None = None, gate: bool = False) -> dict:
    """One served session per calibration run (callo as shipped, the profile's llama-server passthrough repeated),
    after WARMUP_N warm-up requests without early exit. The own-sampler session: early exit at z and the end reading
    (the profile's largest budget, think_converge 0), alternating per item, then its think.converge stats (E1, E3).
    The greedy session: the probe's live mode on every item (E2). Fills `res` as it goes: a run's old entry is dropped
    when its turn comes, the other runs' entries are kept.

    `reuse` (a labelled run: the unlabelled summary's serve part): a greedy session there, on this same profile file
    (path and sha256), is copied instead of served again; E2's live rows do not depend on z. A run with no profile, or
    one whose converge.K is not the bench's, is recorded and fails the part (RuntimeError) after the remaining runs.

    `gate` (`--gate`): the own-sampler sessions only, on GATE_RUN's profile (the gate measured), their early-exit arm
    with `think_converge_gate: true`."""
    failed = []
    for run, (model, k, greedy) in runs(models).items():
        if gate and greedy:                         # the gate run has no greedy session
            continue
        res.pop(run, None)
        path = _profile_path(out, GATE_RUN if gate else run)
        if path is None:
            res[run] = {"error": "no profile (run the calib part first)"}
            failed.append(f"{run}: no profile (run the calib part first)")
            continue
        profile = json.loads(path.read_text())
        why = k_problem(GATE_RUN if gate else run, profile)
        if why:
            res[run] = {"error": why}
            failed.append(why)
            continue
        sha = _sha256(path)
        old = (reuse or {}).get(run) or {}
        if greedy and old.get("live") and old.get("profile") == str(path) and old.get("profile_sha256") == sha:
            res[run] = {**old, "reused": True}
            log.info("serve %s: the unlabelled run's greedy session is reused (its live rows do not depend on z)", run)
            continue
        bmax = max(int(b) for b in profile["budgets"])
        c = _start(f"serve-{run}", out, model=model, ctx=CTX, cache=True,
                   flags=_join_flags(["--think-profile", str(path)], profile_launch(profile)["server_args"]))
        try:
            serve_rows(c.url, dev[:WARMUP_N], WARMUP_FIELDS)
            r: dict = {"profile": str(path), "profile_sha256": sha, "bmax": bmax,
                       "hybrid": bool(gguf_traits(model)["hybrid"])}
            if greedy:
                r["live"] = probe_live_rows(c.url, dev, cap=bmax, k=k)
            else:
                r["z"] = z
                ee_fields = {"think_converge": z, **({"think_converge_gate": True} if gate else {})}
                r["ee"], r["end"] = serve_pairs(c.url, dev, ee_fields, {"think_budget": bmax, "think_converge": 0})
                if gate:
                    r["gate"] = True
                r["stats"] = ((c.stats() or {}).get("think") or {}).get("converge")
            res[run] = r
        finally:
            c.finish()
    if failed:
        raise RuntimeError("serve: " + "; ".join(failed))
    return res


# ---- the gates ------------------------------------------------------------------------------------------------------

def clopper_pearson(x: int, n: int) -> tuple[float, float]:
    """The exact two-sided 95 % interval of x successes in n."""
    lo = 0.0 if x == 0 else beta_ppf(0.025, x, n - x + 1)
    hi = 1.0 if x == n else beta_ppf(0.975, x + 1, n - x)
    return lo, hi


def p50_saving(end_ms: Sequence[float], ee_ms: Sequence[float], clusters: Sequence[str], *,
               strata: Sequence[str] | None = None, seed: int = SEED, resamples: int = 2000) -> tuple[float, float, float]:
    """p50(end) - p50(early exit) over one question set (index i is one question in both arms; the arms' traces are
    independent), with a 95 % percentile interval from a cluster bootstrap that resamples whole clusters of
    questions, within each of `strata` (the rows' sources, as the Δacc bootstrap) when given, and takes both arms' p50
    over the same draw."""
    layers: dict[str, dict[str, list[int]]] = {}
    for i, c in enumerate(clusters):
        layers.setdefault(strata[i] if strata is not None else "", {}).setdefault(c, []).append(i)
    plan = [(sorted(groups), groups) for _, groups in sorted(layers.items())]
    rng = random.Random(seed)

    def diff(idx: Sequence[int]) -> float:
        return percentile([end_ms[i] for i in idx], 0.5) - percentile([ee_ms[i] for i in idx], 0.5)

    def draw() -> list[int]:
        return [i for names, groups in plan for _ in names for i in groups[rng.choice(names)]]

    draws = sorted(diff(draw()) for _ in range(resamples))
    return diff(range(len(end_ms))), draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]


def e1(cv: Mapping | None, profile: Mapping, *, hybrid: bool, forced_n: int | None = None) -> dict:
    """E1 on the own-sampler session's think.converge stats. `forced_n`: the reads behind the profile's
    read_forced_ms (`read_forced_n`); fewer than MIN_FORCED, or unknown, is ERROR: a p50 of 2 cold reads is no
    baseline."""
    forced = ((profile.get("latency") or {}).get("read_forced_ms") or {}).get("p50")
    if forced is None:
        return {"pass": None, "error": "the profile has no latency.read_forced_ms (calibrated without --deadlines or "
                                       "--audit, so no served-like forced read was timed)"}
    if forced_n is None or forced_n < MIN_FORCED:
        return {"pass": None, "read_forced_n": forced_n,
                "error": f"the profile's latency.read_forced_ms rests on "
                         f"{'an unknown number of' if forced_n is None else forced_n} forced reads (>= {MIN_FORCED} "
                         f"needed; the bench's --warmup {CALIB_WARMUP} gives {int(CALIB_WARMUP) - 1})"}
    if not cv:
        return {"pass": None, "error": "no think.converge stats from the served session"}
    after = cv.get("after_probe") or {}
    read = (cv.get("probe_read_ms") or {}).get("p50")
    n = int(after.get("n") or 0)
    if n < MIN_PROBES or read is None:
        return {"pass": None, "error": f"too little evidence: {n} chunks after a probe (>= {MIN_PROBES}), probe read "
                                       f"p50 {read}"}
    share = int(after.get("restored") or 0) / n
    restore_ok = share >= E1_RESTORE if hybrid else None
    read_ok = read <= E1_READ * float(forced)
    return {"pass": read_ok and restore_ok is not False, "hybrid": hybrid, "n": n, "restored_share": share,
            "restore_pass": restore_ok, "read_p50_ms": read, "read_forced_p50_ms": forced, "read_forced_n": forced_n,
            "read_pass": read_ok}


def _no_rule(z: float, why: str | None) -> str:
    """The one phrase E2 and E3 use for a missing rule, the runbook's trigger for the labelled z .9 run."""
    return f"no rule that reaches z {z:g}" + (f" ({why})" if why else "")


def e2(rows: Sequence[Mapping], profile: Mapping, *, z: float = Z) -> dict:
    rules, why = serve_rules(profile.get("converge"))
    rule = rules.for_z(z) if rules is not None else None
    if rule is None:
        return {"pass": None, "error": f"the greedy profile has {_no_rule(z, why)}; for MiniCPM5, run the labelled "
                                       f"z .9 pass (--models minicpm --z 0.9 --label z0.9)"}
    traces = [r for r in rows if r.get("probes") is not None and r.get("final") is not None]
    dropped = len(rows) - len(traces)
    if len(traces) < MIN_TRACES:
        return {"pass": None, "traces": len(traces), "dropped": dropped,
                "error": f"{len(traces)} live-mode traces of {len(rows)} questions ({dropped} probe requests failed; "
                         f">= {MIN_TRACES} needed)"}
    agree = []
    for r in traces:
        probes = [Probe(int(k), tuple(float(x) for x in p)) for k, p in r["probes"]]
        j = stop_index(probes, rule.n, rule.y)
        if j is not None:
            final = r["final"]
            agree.append(probes[j].argmax == max(range(len(final)), key=final.__getitem__))
    if len(agree) < MIN_STOPS:
        return {"pass": None, "traces": len(traces), "dropped": dropped,
                "error": f"{len(agree)} stops on {len(traces)} traces (>= {MIN_STOPS} needed)"}
    x = sum(agree)
    share = x / len(agree)
    return {"pass": share >= z - E2_SLACK, "n": len(agree), "agree": x, "share": share,
            "ci": list(clopper_pearson(x, len(agree))), "bound": z - E2_SLACK, "cell": [rule.n, rule.y],
            "traces": len(traces), "dropped": dropped}


def e3(ee: Sequence[Mapping], end: Sequence[Mapping], profile: Mapping, *, z: float = Z, seed: int = SEED) -> dict:
    """E3 on the own-sampler session's two arms, paired by question (never by trace: the arms think independent
    traces, see the module doc). The rule is the one serve runs for z (`serve_rules(...).for_z(z)`); its target's
    cross-fitted Δacc and early-exit point are the profile's side. ERROR when there is no such rule, point or end
    latency, when fewer than MIN_PAIRED of the questions were answered in both arms, or when a 200 early-exit answer
    carries no target z in its `converge` (served without early exit: not this gate's evidence)."""
    rules, why = serve_rules(profile.get("converge"))
    rule = rules.for_z(z) if rules is not None else None
    if rule is None:
        return {"pass": None, "error": f"the profile has {_no_rule(z, why)}"}
    key = ee_key(rule.z)
    summary = ((profile.get("converge") or {}).get("rules") or {}).get(key) or {}
    point = (profile.get("ee") or {}).get(key) or {}
    budgets = profile.get("budgets") or {}
    bmax = max((int(b) for b in budgets), default=0)
    budget = budgets.get(str(bmax)) or {}
    if not isinstance(summary.get("dacc"), list) or not summary["dacc"]:
        return {"pass": None, "error": f"the profile's rule {key} has no cross-fitted dacc"}
    if not point.get("lat_p50_ms"):
        return {"pass": None, "error": f"the profile has no early-exit point at {key} (no measured probe cost?)"}
    if not budget.get("lat_p50_ms"):
        return {"pass": None, "error": f"the profile has no budget-{bmax} latency"}
    off = [a for a in ee if a.get("status") == 200 and isinstance(a.get("entry"), Mapping)
           and not (isinstance(a["entry"].get("converge"), (int, float)) and not isinstance(a["entry"]["converge"], bool))]
    if off:
        whys = sorted({str(a["entry"].get("converge")) for a in off})
        return {"pass": None, "error": f"{len(off)} early-exit answers were served without early exit "
                                       f"(converge {', '.join(whys)})"}
    by_q = {r["q"]: r for r in end}
    ok = [(a, by_q[a["q"]]) for a in ee if a["q"] in by_q
          and None not in (a.get("correct"), by_q[a["q"]].get("correct"), a.get("ms"), by_q[a["q"]].get("ms"))]
    if not ok or len(ok) < MIN_PAIRED * len(ee):
        return {"pass": None, "error": f"{len(ok)} of {len(ee)} questions answered in both runs "
                                       f"(>= {MIN_PAIRED:.0%} needed)"}
    clusters = [str(a["cluster"]) for a, _ in ok]
    sources = [a.get("source") for a, _ in ok]
    strata = [str(s) for s in sources] if all(s is not None for s in sources) else None   # as the profile's Δacc
    dacc = [float(v) for v in paired_bootstrap([float(b["correct"]) for _, b in ok],
                                               [float(a["correct"]) for a, _ in ok], clusters, seed=seed,
                                               strata=strata)]
    want = float(summary["dacc"][0])
    saving = p50_saving([float(b["ms"]) for _, b in ok], [float(a["ms"]) for a, _ in ok], clusters, strata=strata,
                        seed=seed)
    want_s = float(budget["lat_p50_ms"][0]) - float(point["lat_p50_ms"][0])
    acc_ok, lat_ok = dacc[1] <= want <= dacc[2], saving[1] <= want_s <= saving[2]
    closes: dict[str, int] = {}
    for a, _ in ok:
        c = str((a.get("entry") or {}).get("think_close"))
        closes[c] = closes.get(c, 0) + 1
    return {"pass": acc_ok and lat_ok, "n": len(ok), "target": rule.z, "cell": [rule.n, rule.y], "dacc": dacc,
            "profile_dacc": want, "acc_pass": acc_ok, "saving_ms": list(saving), "profile_saving_ms": want_s,
            "lat_pass": lat_ok, "closes": closes}


def gates(out: Path, serve: Mapping[str, Any], *, z: float = Z, models: Sequence[str] | None = None) -> dict:
    """E1-E3 per model of `models` (all by default). A profile whose converge.K is not the bench's makes all three
    ERROR; E3 is judged only at the z its session served."""
    res = {}
    for name in MODELS:
        if models is not None and name not in models:
            continue
        own_p, greedy_p = _profile(out, name), _profile(out, f"{name}-greedy")
        bad = [k_problem(run, p) for run, p in ((name, own_p), (f"{name}-greedy", greedy_p)) if p]
        bad = [w for w in bad if w]
        if bad:
            res[name] = {g: {"pass": None, "error": "; ".join(bad)} for g in ("e1", "e2", "e3")}
            continue
        own, greedy = serve.get(name) or {}, serve.get(f"{name}-greedy") or {}
        missing = {"pass": None, "error": f"no served session for {name}"}
        served_z = own.get("z")
        if not own.get("ee"):
            g3 = missing
        elif served_z is not None and abs(float(served_z) - z) > 1e-9:
            g3 = {"pass": None, "error": f"served at z {float(served_z):g}, judged at z {z:g}: rerun the serve part "
                                         f"at z {z:g} (a labelled run for another z)"}
        else:
            g3 = e3(own.get("ee") or [], own.get("end") or [], own_p, z=z)
        res[name] = {
            "e1": e1(own.get("stats"), own_p, hybrid=bool(own.get("hybrid")), forced_n=read_forced_n(out, name))
            if "hybrid" in own else missing,
            "e2": e2(greedy.get("live") or [], greedy_p, z=z) if greedy.get("live")
            else {"pass": None, "error": f"no greedy session for {name}"},
            "e3": g3}
    return res


def _gate_entry(row: Mapping) -> Mapping | None:
    g = (row.get("entry") or {}).get("converge_gate") if isinstance(row.get("entry"), Mapping) else None
    return g if isinstance(g, Mapping) else None


def gated_vs_profile(dacc: Sequence[float], saving: Sequence[float], profile: Mapping[str, Any] | None, *,
                     z: float = Z) -> dict:
    """Profile vs served for the gated point, as E3: the rule serve runs for z (`serve_rules(...).for_z(z)`) and its
    gated point (`ee_gated`); PASS iff the served Δacc CI holds the point's cross-fitted Δacc against the cap
    (`dacc_cap`) and the profile's gated p50 saving (the cap budget's p50 minus the gated point's) lies in the served
    saving's CI. ERROR when there is no profile, rule, gated point or cap latency."""
    if not profile:
        return {"pass": None, "error": "no gated profile (run the calib part)"}
    rules, why = serve_rules(profile.get("converge"))
    rule = rules.for_z(z) if rules is not None else None
    if rule is None:
        return {"pass": None, "error": f"the profile has {_no_rule(z, why)}"}
    key = ee_key(rule.z)
    point = (profile.get("ee_gated") or {}).get(key) or {}
    budgets = profile.get("budgets") or {}
    bmax = max((int(b) for b in budgets), default=0)
    budget = budgets.get(str(bmax)) or {}
    if not isinstance(point.get("dacc_cap"), list) or not point["dacc_cap"] or not point.get("lat_p50_ms"):
        return {"pass": None, "error": f"the profile has no gated point at {key} (the gate not measured, or no "
                                       f"measured probe cost)"}
    if not budget.get("lat_p50_ms"):
        return {"pass": None, "error": f"the profile has no budget-{bmax} latency"}
    want = float(point["dacc_cap"][0])
    want_s = float(budget["lat_p50_ms"][0]) - float(point["lat_p50_ms"][0])
    acc_ok, lat_ok = dacc[1] <= want <= dacc[2], saving[1] <= want_s <= saving[2]
    return {"pass": acc_ok and lat_ok, "target": rule.z, "profile_dacc": want, "acc_pass": acc_ok,
            "profile_saving_ms": want_s, "lat_pass": lat_ok, "flagged": (point.get("gate") or {}).get("flagged")}


def gate_result(own: Mapping[str, Any], *, profile: Mapping[str, Any] | None = None, z: float = Z,
                seed: int = SEED) -> dict:
    """The `--gate` run's result on its own-sampler session: the gated early-exit arm against the end reading, paired
    by question as E3 (never by trace). ERROR (`pass` None) when there is no session, when a 200 early-exit answer
    carries no `converge_gate` (served without the gate: early exit off, or unreachable), when fewer than
    MIN_PAIRED of the questions were answered in both arms, or when more than GATE_MAX_ERRORS of the paired
    questions' gate reads failed (a failed read skips early exit, so that run measures the failure, not the gate). The
    verdict, pre-declared: "holds" iff the Δacc CI's upper end is >= 0 and the p50 saving CI's lower end is > 0.
    `profile` (the gated calibration's): its gated point against the served result (`gated_vs_profile`), in
    `profile`, which never changes the verdict."""
    ee, end = own.get("ee") or [], own.get("end") or []
    if not ee:
        return {"pass": None, "verdict": None, "error": f"no served session for {GATE_MODEL}"}
    off = [a for a in ee if a.get("status") == 200 and _gate_entry(a) is None]
    if off:
        return {"pass": None, "verdict": None,
                "error": f"{len(off)} early-exit answers were served without the gate (converge "
                         f"{', '.join(sorted({str((a.get('entry') or {}).get('converge')) for a in off}))})"}
    by_q = {r["q"]: r for r in end}
    ok = [(a, by_q[a["q"]]) for a in ee if a["q"] in by_q
          and None not in (a.get("correct"), by_q[a["q"]].get("correct"), a.get("ms"), by_q[a["q"]].get("ms"))]
    if not ok or len(ok) < MIN_PAIRED * len(ee):
        return {"pass": None, "verdict": None, "error": f"{len(ok)} of {len(ee)} questions answered in both runs "
                                                        f"(>= {MIN_PAIRED:.0%} needed)"}
    n = len(ok)
    clusters = [str(a["cluster"]) for a, _ in ok]
    sources = [a.get("source") for a, _ in ok]
    strata = [str(x) for x in sources] if all(x is not None for x in sources) else None
    ee_c, end_c = [float(a["correct"]) for a, _ in ok], [float(b["correct"]) for _, b in ok]
    ee_ms, end_ms = [float(a["ms"]) for a, _ in ok], [float(b["ms"]) for _, b in ok]
    dacc = [float(v) for v in paired_bootstrap(end_c, ee_c, clusters, seed=seed, strata=strata)]
    saving = list(p50_saving(end_ms, ee_ms, clusters, strata=strata, seed=seed))
    holds = dacc[2] >= 0 and saving[1] > 0
    gates = [_gate_entry(a) or {} for a, _ in ok]
    failed = sum(1 for g in gates if g.get("error") is not None)
    if failed > GATE_MAX_ERRORS * n:
        return {"pass": None, "verdict": None, "gate_errors": failed, "n": n,
                "error": f"{failed} of {n} gate reads failed (> {GATE_MAX_ERRORS:.0%}): "
                         f"{sorted({str(g['error']) for g in gates if g.get('error') is not None})[:3]}"}
    reads = [float(g["ms"]) for g in gates if g.get("error") is None and isinstance(g.get("ms"), (int, float))]
    by_source: dict[str, dict] = {}
    for (a, b), src in zip(ok, sources, strict=True):
        d = by_source.setdefault(str(src), {"n": 0, "ee": 0.0, "end": 0.0})
        d["n"] += 1
        d["ee"] += float(a["correct"])
        d["end"] += float(b["correct"])
    for d in by_source.values():
        d["ee"], d["end"] = d["ee"] / d["n"], d["end"] / d["n"]
    closes: dict[str, int] = {}
    for a, _ in ok:
        c = str((a.get("entry") or {}).get("think_close"))
        closes[c] = closes.get(c, 0) + 1
    return {"pass": holds, "verdict": "holds" if holds else "does not hold", "n": n,
            "acc": {"ee": sum(ee_c) / n, "end": sum(end_c) / n}, "dacc": dacc,
            "lat": {"ee": {"p50": percentile(ee_ms, 0.5), "mean": sum(ee_ms) / n},
                    "end": {"p50": percentile(end_ms, 0.5), "mean": sum(end_ms) / n}},
            "saving_ms": saving, "skip_share": sum(1 for g in gates if g.get("skip") is True) / n,
            "gate_errors": failed,
            "read_ms": {"p50": percentile(reads, 0.5) if reads else None,
                        "p95": percentile(reads, 0.95) if reads else None},
            "by_source": dict(sorted(by_source.items())), "closes": closes, "stats": (own.get("stats") or {}).get("gate"),
            "profile": gated_vs_profile(dacc, saving, profile, z=z)}


def served_profile(out: Path, own: Mapping[str, Any]) -> dict:
    """The profile the `--gate` session served (its recorded path), else GATE_RUN's; {} without one."""
    p = Path(str(own.get("profile"))) if own.get("profile") else _profile_path(out, GATE_RUN)
    return json.loads(p.read_text()) if p is not None and p.is_file() else {}


def unlabelled_serve(out: Path) -> dict:
    """The unlabelled run's serve part (its summary.json), the greedy sessions a labelled run reuses; {} without one."""
    p = paths(out, None)[0]
    if not p.exists():
        return {}
    return (json.loads(p.read_text()).get("parts") or {}).get("serve") or {}


def main(argv: list[str] | None = None) -> int:
    """0 when every requested part ran (a FAIL is the feature's and is in the report), 1 when a part errored, 2 when
    the bench refused to start, 3 when it was interrupted."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--data", default=TRANSFER_TEST, help="the calibration file (the transfer set's test split)")
    ap.add_argument("--dev", default=DEV, help="the held-out file (the transfer set's development split); never "
                                                        "calibrated on")
    ap.add_argument("--parts", default=None, help=f"which parts to run, in this order: {','.join(PARTS)} (default all; "
                                                  f"{','.join(GATE_PARTS)} with --gate)")
    ap.add_argument("--models", default=None, help=f"a comma list of {', '.join(MODELS)} (default all)")
    ap.add_argument("--z", type=float, default=Z, help="the served target (the gates' z); .95 by default")
    ap.add_argument("--label", default=None, help="a labelled run's own summary and report (e.g. z0.9)")
    ap.add_argument("--min-free-gb", type=float, default=30.0, help="free RAM (vm_stat) the GPU parts need to start")
    ap.add_argument("--gate", action="store_true",
                    help="the early-exit gate's run: Tiel only, a calibration with the gate measured (calib/tiel-gate), "
                         "then early exit with think_converge_gate true against the end reading, no greedy session; "
                         "its own summary and report (label gate)")
    ap.add_argument("--gate-fresh", action="store_true",
                    help="--gate's calibration from scratch, not from a copy of the unlabelled run's raw file; refused "
                         "while calib/tiel-gate holds a raw file (delete that directory first)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    unset = lib.unset_paths(("--out", a.out, lib.OUT_ENV))
    if unset:
        print(unset)
        return 2
    parts = [p for p in (a.parts if a.parts is not None else ",".join(GATE_PARTS if a.gate else PARTS)).split(",")
             if p]
    if any(p not in PARTS for p in parts):
        print(f"refusing to start: unknown part(s) in {parts}; choose from {','.join(PARTS)}")
        return 2
    models = [m for m in a.models.split(",") if m] if a.models else None
    if models is not None and (not models or any(m not in MODELS for m in models)):
        print(f"refusing to start: unknown model(s) in {models}; choose from {', '.join(MODELS)}")
        return 2
    if a.gate_fresh and not a.gate:
        print("refusing to start: --gate-fresh shapes --gate's calibration: it needs --gate")
        return 2
    held = sorted((Path(a.out) / "calib" / GATE_RUN).glob("*.raw.jsonl"))
    if a.gate_fresh and held:
        print(f"refusing to start: --gate-fresh calibrates from scratch, but {held[0].parent} already holds a raw file "
              f"({held[0].name}), which a fresh run would resume: delete {held[0].parent} first")
        return 2
    if a.gate:
        if models not in (None, [GATE_MODEL]) or a.label not in (None, GATE_LABEL):
            print(f"refusing to start: --gate is a run on {GATE_MODEL} alone with its own calibration (no --models "
                  f"other than {GATE_MODEL}, no --label other than {GATE_LABEL})")
            return 2
        models = [GATE_MODEL]
        a.label = a.label or GATE_LABEL
    if not 0 < a.z < 1:
        print(f"refusing to start: --z {a.z} is not in (0, 1)")
        return 2
    if a.label is not None and (not a.label or "/" in a.label or a.label.startswith(".")):
        print(f"refusing to start: --label {a.label!r} must be a plain name (it names the summary and report files)")
        return 2
    if {"calib", "serve"} & set(parts):
        unset = lib.unset_paths(("--data", a.data, lib.DATA_ENV), ("--dev", a.dev, lib.DATA_ENV),
                                *(("", MODELS[m][0], lib.MODELS_ENV) for m in models or MODELS),
                                ("", LLAMA_SERVER, lib.SERVER_ENV))
        if unset:
            print(unset)
            return 2
        try:
            busy = lib.running_gpu_servers()
        except RuntimeError as e:                   # a ps that fails is "cannot tell", never "no servers"
            print(f"refusing to start: {e}: cannot tell whether a model server is running")
            return 2
        if busy:
            print(f"refusing to start: a model server is already running (one model server at a time): {busy}")
            return 2
        for f in (a.data, a.dev):
            if not Path(f).is_file():
                print(f"refusing to start: no data file {f}")
                return 2
        if Path(a.data).resolve() == Path(a.dev).resolve() or overlap(Path(a.data), Path(a.dev)):
            print("refusing to start: the held-out file shares records with the calibration file")
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
    summary_path, doc = paths(out, a.label)
    s = json.loads(summary_path.read_text()) if summary_path.exists() else {"parts": {}, "errors": {}}
    s.update(z=a.z, label=a.label, models=models)
    if a.gate:                                      # the gate's report until its gates part has judged it
        s.setdefault("gate", {"pass": None, "verdict": None, "error": "not computed (run the gates part)"})
    record = session_provenance(out, parts)
    s.setdefault("sessions", []).append(record)     # every run's, so a later gates-only run keeps the GPU run's
    for w in record["warnings"]:
        print(f"WARNING: {w}")
        log.warning("%s", w)
    rc = 0
    try:
        pid_file.write_text(f"{me}\n")
        test, dev = out / "data" / "test.jsonl", out / "data" / "dev.jsonl"
        for part in parts:
            try:
                if part in ("calib", "serve"):
                    busy = _foreign_servers()          # every server this bench started is stopped between parts
                    if busy:
                        raise RuntimeError(f"a model server is running before the {part} part: {busy}")
                    if not test.exists():
                        select_records(a.data, test)
                    if not dev.exists():
                        select_records(a.dev, dev)
                    if overlap(test, dev):
                        raise RuntimeError(f"{dev} shares records with {test}: the held-out set is not held out")
                if part == "calib":
                    run_calib(out, test, models, s["parts"].setdefault("calib", {}), gate=a.gate, fresh=a.gate_fresh)
                elif part == "serve":
                    run_serve(out, load_data(dev)[0], s["parts"].setdefault("serve", {}), z=a.z, models=models,
                              reuse=unlabelled_serve(out) if a.label and not a.gate else None, gate=a.gate)
                elif a.gate:
                    own = (s["parts"].get("serve") or {}).get(GATE_MODEL) or {}
                    s["gate"] = gate_result(own, profile=served_profile(out, own), z=a.z)
                else:
                    s.setdefault("gates", {}).update(gates(out, s["parts"].get("serve") or {}, z=a.z, models=models))
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
        doc.write_text(render(s), encoding="utf-8")
        for line in verdict_lines(s):
            print(line)
        print(f"early-exit bench: done, rc {rc} ({doc})")
        return rc
    finally:
        if _pid_in(pid_file) == me:
            pid_file.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
