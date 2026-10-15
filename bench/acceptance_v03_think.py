"""The v0.3 think-budgets acceptance bench: Run 1 (the main menu), Run 2 (the dense model), Run 3 (TielCoder
under `fixed`), the joint `callo compare`, the T7 menu-isolation rerun, a serve-path session per thinking model, the T5
guard probe, and the gates T1-T10. GPU, llama.cpp-master @ 4ebdf2c, one model server at a time on an otherwise idle
machine. Raw output goes to `--out` (OUT_DEFAULT), the result doc to
`docs/bench/v0.3-think.md` (`bench/report_v03_think.py`).

The pure half (this section) is unit-tested on fakes: the data selection, the gate functions, which read profiles,
raw files, run.json files and the served sessions' records. `main` (the orchestration) is smoke-tested through its
seams: `_calibrate` (how a calibrate command line runs), `_launch` (how a serve-path session starts) and the model
table, which the smoke test points at the fake llama-server executable and tiny GGUFs.

The gates never PASS on missing or incomplete evidence. Only a profile its run's `run.json` marks `done` reaches a
gate (`run_profile`): a model that run failed (another model server appeared, its stamp failed) is missing everywhere,
an ERROR naming why. Each gate entry carries `pass` (True / False / None) and, when
the bench could not judge it, an `error` (on the gate, a model or a budget), which the report renders as ERROR; a
measured FAIL stands over an ERROR. The orchestration builds `summary["gates"]["t1"]` and `["t2"]` ONLY through
`t1_t2_gates` (one `t1_t2` result per model in, the two entries out, every `error` carried and the required pairs
`T1_REQUIRED` judged), never by reshaping a `t1_t2` result by hand, so an ERROR cannot turn into n/a on the way.
Likewise `["t3"]` through `t3_gate` and `["t10"]` through `t10_gate`, and `t9` is given every raw file and served
session it has: each holds the gate to its expected models (EXPECTED_PROFILES, T9_RAW_MODELS, SERVED_MODELS), so a
model or a half the orchestration lost is an ERROR naming it.

The orchestration (`main`, the last section) runs the requested `--parts` in order (run1, run2, run3, t7, t8, serve,
t5), each isolated: a part's exception is its `errors` entry and the next part runs; a Ctrl-C or a stop signal aborts
(exit 3) with what finished kept, the offline gates recomputed from the files even then. A part that runs again first
drops the verdicts an earlier session left for it, so
a part that fails now leaves its gates missing (NOT RUN, or ERROR where the gate names what it expected), never an old
PASS. `summary.json` keeps every part that finished and the bench is rerunnable: a menu run whose DIR holds a
`run.json` resumes it, T7's broken run and T5's probes always start in a fresh DIR.

Live-run hygiene: `summary.json` records the code's commit and whether its tree is dirty, the
llama-server binary and its `--version`, and the data files' sha256. The bench refuses to start (exit 2) while another
model server runs unless `--allow-gpu-busy` (recorded, and passed to every menu run, whose profiles are then marked
gpu_busy), checks the disks and that no other model server appeared before every part, and never binds or calls
ports 8080 or 8130 (`FORBIDDEN_PORTS`, checked on every serve-path launch; every server it starts binds an ephemeral
port). Its pid is in `DIR/run.pid` for the whole run: a `run.pid` naming a live process refuses the start (exit 2:
another bench run owns that --out when the pid's command line names this module, else a pid to check and remove by
hand, never to signal), a dead one is replaced, and at exit the file is removed only while it still holds this
run's pid. A `ps` that fails is "cannot tell" (callo's GPU rule, `lib.running_gpu_servers`): a refusal at the start
and before a part, never "no servers". Each part's commit, dirty flag and llama-server version are kept per part (`provenance`, shown
in the doc), and the start warns (stderr and `warnings`) about what would leave T4 untestable, a build other than
EXPECTED_BUILD and the free disk of --out and /. The live run is stopped ONLY through run.pid:
`kill -TERM $(cat DIR/run.pid)` (the stop handlers turn SIGTERM into a clean exit that stops every server the bench
started); never by process name, and never by signalling a llama-server directly. Rerun the same command to continue.
"""
from __future__ import annotations

import argparse
import contextlib
import http.client
import io
import json
import logging
import os
import random
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from bench import lib
from bench.acceptance import MASTER, Callo
from bench.acceptance import MODEL as QWEN9B
from bench.acceptance_v02 import QWEN9B_MMPROJ, TIEL, _install_stop_handlers
from bench.acceptance_v03_bias import (
    DISK_PATHS,
    MIN_FREE_GIB,
    _erase_all,
    _fill,
    _seen,
    _sha256,
    _Tee,
    disk_problems,
    failed_conditions,
    start_decode,
)
from bench.report_v03_think import render, version_line
from callo import cli
from callo.calibrate_think import (
    _latest,
    _phase_b,
    audit_pool,
    check_rows,
    load_data,
    outcomes,
    read_raw,
    shuffled,
)
from callo.compare import correct_at, joint, load_profile, malformed, rows_at
from callo.errors import CalloError, InvalidInput
from callo.menu import RunFile
from callo.think import finite_number, nearest_rank, paired_bootstrap, point_stats, tail_counts

ROOT = Path(__file__).resolve().parents[1]
# The transfer test set the System One sources are selected from (the bias bench reads another pool, so the path lives here).
TRANSFER_TEST = lib.env_path(lib.DATA_ENV, "v4", "transfer-v4", "test.jsonl")
DOC = ROOT / "docs" / "bench" / "v0.3-think.md"
OUT_DEFAULT = lib.env_path(lib.OUT_ENV, "callo_v0.3", "think")
DATA_SRC = TRANSFER_TEST            # the test split: the earlier System One study read dev, so test is fresh here
GEMMA = lib.env_path(lib.MODELS_ENV, "unsloth", "Gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-UD-Q4_K_XL.gguf")
MINICPM = lib.env_path(lib.MODELS_ENV, "local", "MiniCPM5-2B-UD-Q6_K_XL.gguf")
# The data: the System One selection rule on the transfer TEST split (`group_of`): the composition set (96 clean
# records), the date-score group (the 40 `contrastive` records whose question type is `score`; the 40 `contrastive`
# `noul` records are another family and are not selected), and the MMLU/QNLI/PAWS control (240): 376 questions from
# the file's 764 records. Clean variants only (calibrate skips the others anyway).
S1_SOURCES = ("composition_holdout", "mmlu", "qnli", "paws")
S1_SCORE_SOURCES = ("contrastive",)          # kept only when every question of the record is a `score` question
GROUPS = {"composition_holdout": "composition", "contrastive": "date-score", "mmlu": "control", "qnli": "control",
          "paws": "control"}
BUDGETS, AUDIT, CTX, PARALLEL, SEED = "0,256,512,1024", 48, 65536, 4, 0
RUN2_N = 120                     # Run 2: a seeded subset of the same test file (its own data sha)
T7_N, T7_BUDGETS = 8, "0,256"    # T7 tests menu isolation, not measurement: the first 8 questions, two budgets, no audit
SERVE_N = 48                     # T10: the first 48 measured questions through /v1/systemone
SKIP_MAX = 48                    # T10: stop after this many non-200 replies (persistent 504s, each a full timeout)
LLAMA_SERVER = MASTER
MODELS: dict[str, dict] = {"tiel": {"model": TIEL}, "qwen9b": {"model": QWEN9B}, "gemma-e4b": {"model": GEMMA}}
RUN2_MODELS: dict[str, dict] = {"minicpm": {"model": MINICPM}}
RUN3_MODELS: dict[str, dict] = {"tiel-fixed": {"model": TIEL, "think_sampling": "fixed"}}
THINKING = ("tiel", "qwen9b")    # the serve-path sessions (T4, T9, T10) and T5's guard probe (on tiel)
# The models each gate must hear from (a missing one is that gate's ERROR, never a PASS on the rest): T3 every profile
# of Runs 1-3; T9's calibrate half the thinking models' raw files (Run 1's and MiniCPM5's); T9's served half and T10
# every serve-path session.
EXPECTED_PROFILES = (*MODELS, *RUN2_MODELS, *RUN3_MODELS)
T9_RAW_MODELS = (*THINKING, *RUN2_MODELS)
SERVED_MODELS = THINKING
T1_MIN_CUTS, T1_DP, T1_DP_DENSE, T2_ERR, T3_SLACK = 24, 1e-3, 1e-6, 0.10, 0.002
# T1's post-hoc amendment (T1_AMENDED): on a MoE hybrid (GGUF metadata: an
# `<arch>.expert_count` above 0 and `<arch>.ssm.*` keys; never the model's name) T1 is judged on trace identity, argmax
# agreement (>= m - 1 of the m identical) and a looser median |Δp| bound, T1_DP_MOE; the original 1e-3 outcome is still reported.
T1_DP_MOE, T1_AMENDED = 1e-2, "2026-10-08"
T1_AMENDED_NOTE = f"amended post-hoc on {T1_AMENDED}, a deviation from the pre-registered gate"
T9_REPREFILL, T10_ERR, T4_PROMPT_N = 0.01, 0.15, 4
FILL_TOKENS, SMALL_CTX, SMALL_BUDGET = 300, 4096, 4090      # T4's 413: any prompt plus 4090 overflows a 4096 context
log = logging.getLogger("bench")


# ---- data --------------------------------------------------------------------------------------------------------

def select_records(src: str | Path, dst: Path, *, sources: Sequence[str] = S1_SOURCES,
                   score_sources: Sequence[str] = S1_SCORE_SOURCES, limit: int | None = None, seed: int = SEED) -> int:
    """Write the bench's data file: the clean records of `sources`, plus the clean records of `score_sources` whose
    every question is a `score` question (the date-score group), in file order, or a seeded subset of `limit` of them
    (Run 2). Returns the count. A record without `_meta.source` is kept only when both source lists are empty."""
    kept = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            meta = r.get("_meta") or {}
            if meta.get("variant", "clean") != "clean":
                continue
            source = meta.get("source")
            types = {(q or {}).get("type") for q in (r.get("questions") or {}).values()}
            if sources or score_sources:
                if not (source in sources or (source in score_sources and types == {"score"})):
                    continue
            kept.append(line.rstrip("\n"))
    if limit is not None and limit < len(kept):
        kept = random.Random(seed).sample(kept, limit)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept)


def menu_doc(models: dict[str, dict], *, ctx: int = CTX, parallel: int = PARALLEL, llama_server: str | None = None) -> dict:
    """A `--menu` document for `models` (name -> entry keys); the launch shape of Run 1 (`-c 65536 -np 4 -kvu -b 512
    -ub 512`: -kvu, -b and -ub are callo's own launch flags)."""
    entries = []
    for name, extra in models.items():
        e = {"name": name, "ctx": ctx, "parallel": parallel, **extra}
        if llama_server and "llama_server" not in e:
            e["llama_server"] = llama_server
        entries.append(e)
    return {"models": entries}


# ---- the gates ---------------------------------------------------------------------------------------------------

# T1/T2's required (model, budget) pairs: each must be gated and pass. The flag says the pair may instead be
# `not testable` when its pool holds fewer than T1_MIN_CUTS cut questions (tiel@256 only: TielCoder now thinks at its
# GGUF's 1.0, so its cut count on test may differ from the earlier dev reading); the doc then says ONE_HYBRID.
T1_REQUIRED: tuple[tuple[str, int, bool], ...] = (("qwen9b", 256, False), ("qwen9b", 512, False), ("tiel", 256, True))
ONE_HYBRID = "the cut model was verified on one hybrid only"
PARTIAL = "partial last line dropped"


def _median(xs: Sequence[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _all3(values: Sequence[bool | None]) -> bool | None:
    """One verdict from several: False when any is False, None when there are none or any is None, else True."""
    if any(v is False for v in values):
        return False
    return None if not values or any(v is None for v in values) else True


def _dense_note(e: dict) -> str:
    if e["dp_max"] is None:
        return f"dense, descriptive (no gate): no identical trace ({e['n']} audited, n_cut {e['n_cut']})"
    side = "within" if e["dp_max"] <= T1_DP_DENSE else "above"
    return (f"dense, descriptive (no gate): |Δp| max {e['dp_max']:.2e} against {T1_DP_DENSE:.0e} ({side}), argmax "
            f"{e['argmax_agree']}/{e['identical']} on {e['identical']}/{e['n']} identical traces (n_cut {e['n_cut']})")


def gguf_traits(path: str | Path) -> dict:
    """The T1 amendment's model class, from the GGUF's header metadata (`ggufkv.open_kv` memory-maps the file and
    parses its key-value fields and tensor index; no tensor view is built and no tensor data is read): `arch`,
    `expert_count` (its `<arch>.expert_count`, 0 when absent), `moe` (expert_count > 0) and `hybrid` (any
    `<arch>.ssm.*` key: recurrent layers, so a cut below n_gen is read from a re-prefilled state).
    InvalidInput when it cannot be read."""
    try:
        # the gates import without gguf; only this reader needs it (missing: InvalidInput, the original criterion)
        from callo import ggufkv  # noqa: PLC0415

        reader = ggufkv.open_kv(path)
        arch_field = reader.get_field("general.architecture")
        if arch_field is None:
            raise InvalidInput(f"{path}: no general.architecture in its header")
        arch = str(arch_field.contents())
        experts = reader.get_field(f"{arch}.expert_count")
        count = int(experts.contents()) if experts is not None else 0
        hybrid = any(k.startswith(f"{arch}.ssm.") for k in reader.fields)
    except InvalidInput:
        raise
    except Exception as e:  # noqa: BLE001 -- gguf raises assorted errors on a missing or garbled file
        raise InvalidInput(f"{path}: not a readable GGUF ({type(e).__name__}: {e})") from e
    return {"arch": arch, "expert_count": count, "moe": count > 0, "hybrid": hybrid}


def model_traits(profile: dict) -> tuple[dict | None, str | None]:
    """(gguf_traits of the GGUF the profile was measured on (`environment.model.path`), None), or (None, why). The file
    at that path must still be the one measured: its size must equal the profile's `environment.model.bytes`; a
    different size, or none recorded, is (None, why), so the caller keeps the original criterion."""
    model = (profile.get("environment") or {}).get("model") or {}
    path, recorded = model.get("path"), model.get("bytes")
    if not path:
        return None, "the profile records no model path (environment.model.path)"
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return None, f"{path}: cannot read it ({e.strerror or e})"
    if type(recorded) is not int:
        return None, f"the profile records no model size (environment.model.bytes) to confirm {path} is the GGUF it measured"
    if size != recorded:
        return None, f"{path} is {size} bytes, the profile measured a GGUF of {recorded}: not the same file"
    try:
        return gguf_traits(path), None
    except InvalidInput as e:
        return None, str(e)


def _amendment(entry: dict, traits: dict, *, ident_ok: bool, argmax_ok: bool, moe_ok: bool) -> str:
    """The amended T1 verdict, said in full: the gate that was applied and the original criterion's outcome."""
    med = entry["dp_median"]
    dp = f"{med:.2e}" if med is not None else "n/a"
    orig = "met" if med is not None and med <= T1_DP else "FAILS"
    moe = "met" if moe_ok else "exceeded"
    verdict = "PASS" if ident_ok and argmax_ok and moe_ok else "FAIL"
    return (f"{verdict} ({T1_AMENDED_NOTE}; for MoE hybrids, {traits.get('arch')}.expert_count "
            f"{traits.get('expert_count')} with SSM layers: judged on trace identity {entry['identical']}/{entry['n']}, "
            f"argmax >= m - 1 and median |Δp| <= {T1_DP_MOE:.0e}; original |Δp| criterion {T1_DP:.0e} {orig}, median {dp}; MoE bound {T1_DP_MOE:.0e} "
            f"{moe}; argmax {entry['argmax_agree']}/{entry['identical']} {'meets' if argmax_ok else 'misses'} >= m - 1)")


def t1_t2(profile: dict, raw_path: str | Path, data_path: str | Path, *, dense: bool = False,
          traits: dict | None = None) -> dict:
    """T1 (cut = generation) and T2 (the latency model) per (model, budget < Bmax) from the raw file's audit rows: the
    real B-token generation against the Bmax run's cut. Gated when the budget's pool holds >= 24 cut questions
    (`n_cut`): identical traces on >= n - 2 of the n audited; on the m identical ones argmax agreement on >= m - 1 and
    median |Δp| <= 1e-3 (max reported); T2 on the identical ones: median |relative latency error| <= 10 % and the p50s
    within 10 %. The thresholds are the bench's, never the profile's: a run with `--audit` below 48 (0 included) is the
    model's `error` ("the bench requires --audit 48"). Fewer than 24 cuts is `audit n/a` (no wider pool exists:
    --audit 48 audits every cut up to 48); 24 or more with fewer than min(n_cut, 48) audited (so fewer than 24 at the
    least) is an incomplete audit, the entry's `error`. `dense` (MiniCPM5,
    pure attention): no gate at all; the entry is `descriptive`, its note the max |Δp| against 1e-6 and the argmax
    agreement.

    `traits` (`model_traits`, from the GGUF header) selects the post-hoc amendment (T1_AMENDED):
    a MoE hybrid's gated pair is judged on trace identity, argmax agreement and median |Δp| <= T1_DP_MOE (1e-2), the
    original 1e-3 outcome still said in the entry's `amended` verdict text; every other model, and one whose traits are unknown (None),
    keeps the original criterion. `criterion` names which applied.

    The fit's own definitions (calibrate's): Bmax and the budgets are the run's (`comparable.budgets`); `n_cut` is the
    size of `audit_pool` (the measured, non-warm-up questions cut at B, in the run order of the comparable seed and
    warm-up); the phase-B rows are `_phase_b`'s; `n` counts the audited questions (the latest audit row of each) with
    a phase-B trace to compare against (a single-option question's `trace: null` has none). An `n_cut` the profile
    records differently, an incomplete row (`check_rows`), unreadable data or a profile without its comparable
    settings is the model's `error`. A partial last line `read_raw` dropped is a note, not a failure."""
    fail: dict[str, Any] = {"budgets": {}, "gated": 0, "t1_pass": None, "t2_pass": None, "notes": []}
    try:
        raw = read_raw(raw_path)
        check_rows(raw, raw_path)                        # read_raw is lenient; an acceptance gate is not
        items, _ = load_data(data_path)
    except InvalidInput as e:
        return {**fail, "error": str(e)}
    notes = [PARTIAL] if raw.partial else []
    comp = profile.get("comparable") or raw.header.get("comparable") or {}
    seed, warmup, run_budgets, audit = comp.get("seed"), comp.get("warmup"), comp.get("budgets"), comp.get("audit")
    if (type(seed) is not int or type(warmup) is not int or not isinstance(run_budgets, list) or not run_budgets
            or not all(type(b) is int for b in run_budgets)):
        return {**fail, "notes": notes, "error": "neither the profile nor the raw header records the run's seed, warm-up "
                                                 "and budgets (comparable)"}
    if not dense and (type(audit) is not int or audit < AUDIT):
        return {**fail, "notes": notes, "error": f"the bench requires --audit {AUDIT}, the profile has {audit}"}
    moe_hybrid = not dense and bool(traits and traits.get("moe") and traits.get("hybrid"))
    budgets = sorted(run_budgets)
    outs = outcomes(items, raw, budgets)
    b_rows = _phase_b(raw)
    order = shuffled(items, seed)
    uids = {it.uid for it in items}
    audit_entries = ((profile.get("audit") or {}).get("budgets") or {})
    per: dict[str, dict] = {}
    mismatch: list[str] = []
    for b in [b for b in budgets if 0 < b < max(budgets)]:
        latest: dict[str, dict] = {}                     # the latest audit row per question: a resumed audit never
        for r in raw.rows:                               # counts a question twice
            if r.get("phase") == "audit" and r.get("budget") == b and r["q"] in uids:
                latest[r["q"]] = r
        compared = [r for r in latest.values() if isinstance((b_rows.get(r["q"]) or {}).get("trace"), dict)
                    and b in outs.get(r["q"], {})]
        n_cut = sum(1 for it in audit_pool(order, b_rows, warmup, b) if b in outs.get(it.uid, {}))
        recorded_cut = (audit_entries.get(str(b)) or {}).get("n_cut")
        if recorded_cut is not None and recorded_cut != n_cut:
            mismatch.append(f"budget {b}: the bench counts n_cut {n_cut}, the profile records {recorded_cut}")
        identical = agree = 0
        dps: list[float] = []
        errs: list[float] = []
        real: list[float] = []
        model: list[float] = []
        for r in compared:
            full, o = b_rows[r["q"]], outs[r["q"]][b]
            same = list(r["trace"]["tokens"]) == list(full["trace"]["tokens"])[:o.k]
            identical += same
            if not same:
                continue
            agree += max(range(len(r["probs"])), key=r["probs"].__getitem__) == max(range(len(o.probs)), key=o.probs.__getitem__)
            dps.append(max(abs(float(x) - float(y)) for x, y in zip(r["probs"], o.probs, strict=True)))
            lat = float(r["lat_ms"])
            if lat > 0:
                errs.append(abs(o.lat_ms - lat) / lat)
                real.append(lat)
                model.append(o.lat_ms)
        n = len(compared)
        entry: dict[str, Any] = {"n_cut": n_cut, "n": n, "identical": identical, "argmax_agree": agree,
                                 "dp_median": _median(dps), "dp_max": max(dps) if dps else None,
                                 "lat_err_median": _median(errs), "p50_real_ms": _median(real), "p50_model_ms": _median(model)}
        recorded = audit_entries.get(str(b)) or {}       # calibrate records |Δp| and the agreement over identical traces
        for key in ("dp_median", "dp_max", "argmax_agree"):   # itself (deduplicated); the profile's numbers win
            if isinstance(recorded.get(key), (int, float)) and not isinstance(recorded.get(key), bool):
                entry[key] = recorded[key]
        agree = entry["argmax_agree"]
        need = min(n_cut, AUDIT)                         # n_cut >= T1_MIN_CUTS (24) makes need >= 24 too
        if dense:
            entry.update({"t1": None, "t2": None, "descriptive": True, "note": _dense_note(entry)})
        elif n_cut < T1_MIN_CUTS:
            entry.update({"t1": None, "t2": None, "note": f"audit n/a: n_cut {n_cut}, the gate needs {T1_MIN_CUTS} cut "
                                                           f"questions ({n} audited)"})
        elif n < need:
            entry.update({"t1": None, "t2": None, "error": f"incomplete audit at {b}: {n} audited of the {need} the run "
                                                           f"regenerates (min of n_cut {n_cut} and {AUDIT})"})
        else:
            dp_ok = entry["dp_median"] is not None and entry["dp_median"] <= T1_DP
            ident_ok, argmax_ok = identical >= n - 2, agree >= identical - 1
            if moe_hybrid:                               # the post-hoc amendment: the looser |Δp| bound, still gated
                moe_ok = entry["dp_median"] is not None and entry["dp_median"] <= T1_DP_MOE
                entry["t1"] = ident_ok and argmax_ok and moe_ok
                entry["t1_original"] = ident_ok and argmax_ok and dp_ok      # what the unamended gate says, kept as data
                entry["criterion"] = f"moe-hybrid ({T1_AMENDED_NOTE})"
                entry["amended"] = _amendment(entry, traits or {}, ident_ok=ident_ok, argmax_ok=argmax_ok, moe_ok=moe_ok)
            else:
                entry["t1"] = ident_ok and argmax_ok and dp_ok
                entry["criterion"] = "original"
            p50_ok = (entry["p50_real_ms"] and entry["p50_model_ms"] is not None
                      and abs(entry["p50_model_ms"] - entry["p50_real_ms"]) / entry["p50_real_ms"] <= T2_ERR)
            entry["t2"] = bool(entry["lat_err_median"] is not None and entry["lat_err_median"] <= T2_ERR and p50_ok)
        per[str(b)] = entry
    gated = [e for e in per.values() if e["t1"] is not None]
    errored = mismatch or any(e.get("error") for e in per.values())
    out: dict[str, Any] = {"budgets": per, "gated": len(gated),
                           "t1_pass": _all3([e["t1"] for e in gated] + ([None] if errored else [])),
                           "t2_pass": _all3([e["t2"] for e in gated] + ([None] if errored else [])), "notes": notes}
    if mismatch:
        out["error"] = "; ".join(mismatch)
    return out


def _errored(r: Any) -> bool:
    """An `error` anywhere in a gate entry (the model's own, a budget's)."""
    if isinstance(r, dict):
        return bool(r.get("error")) or any(_errored(v) for v in r.values())
    return False


def t1_t2_gates(results: Mapping[str, dict], *, dense: Collection[str] = (),
                required: Sequence[tuple[str, int, bool]] = T1_REQUIRED) -> tuple[dict, dict]:
    """The summary's T1 and T2 gate entries from each model's `t1_t2` result: the ONE way the orchestration builds
    them, so a model's `error` (and a budget's) always reaches the report as ERROR, never n/a. Each entry is
    {"models": {name: {"budgets", "pass", "error"?, "notes"?}}, "required": {"m@B": status}, "pass", "error"?,
    "notes"?, "descriptive"?}: `dense` models are kept under `descriptive` and never count. Every required pair must
    be gated; one that is missing, `audit n/a` or errored makes the gate's `error` ("required pair m@B not gated: ...")
    unless its flag allows `not testable` (fewer than T1_MIN_CUTS cuts), which adds the ONE_HYBRID note. `pass`:
    False when any gated pair fails, else None on any error, else True (None when nothing was gated)."""
    gates = []
    for field in ("t1", "t2"):
        models: dict[str, dict] = {}
        desc: dict[str, dict] = {}
        for name, r in results.items():
            m: dict[str, Any] = {"budgets": r.get("budgets") or {}}
            if r.get("error"):
                m["error"] = r["error"]
            if r.get("notes"):
                m["notes"] = list(r["notes"])
            if name in dense:
                desc[name] = m
            else:
                models[name] = {**m, "pass": r.get(f"{field}_pass")}
        req: dict[str, str] = {}
        problems: list[str] = []
        notes: list[str] = []
        for name, b, may_skip in required:
            label = f"{name}@{b}"
            r = models.get(name)
            e = (r or {}).get("budgets", {}).get(str(b))
            if r is None:
                why = "no result for the model"
            elif e is None:
                why = r.get("error") or "no audit entry at this budget"
            elif e.get("error"):
                why = e["error"]
            elif e.get(field) is None and may_skip and isinstance(e.get("n_cut"), int) and e["n_cut"] < T1_MIN_CUTS:
                req[label] = f"not testable (n_cut {e['n_cut']} < {T1_MIN_CUTS})"
                notes.append(f"{label} not testable (n_cut {e['n_cut']} < {T1_MIN_CUTS}): {ONE_HYBRID}")
                continue
            elif e.get(field) is None:
                why = e.get("note") or "not gated"
            else:
                req[label] = ("PASS" if e[field] else "FAIL") + (
                    f" ({T1_AMENDED_NOTE})" if field == "t1" and e.get("amended") else "")
                continue
            req[label] = f"not gated: {why}"
            problems.append(f"required pair {label} not gated: {why}")
        values = [e[field] for m in models.values() for e in m["budgets"].values() if e.get(field) is not None]
        broken = bool(problems) or any(_errored(m) for m in models.values())
        gate: dict[str, Any] = {"models": models, "required": req, "pass": _all3(values + ([None] if broken else []))}
        if problems:
            gate["error"] = "; ".join(problems)
        if notes:
            gate["notes"] = notes
        if desc:
            gate["descriptive"] = desc
        gates.append(gate)
    return gates[0], gates[1]


def t3_gate(profiles: Mapping[str, dict], *, expected: Collection[str] = EXPECTED_PROFILES) -> dict:
    """The summary's T3 entry: `t3` per profile, and an ERROR entry for every expected profile that is missing (or is
    not a profile at all)."""
    out: dict[str, dict] = {}
    for name in [*expected, *(n for n in profiles if n not in expected)]:
        p = profiles.get(name)
        out[name] = t3(p) if isinstance(p, dict) else {"error": "no profile (expected for T3)", "pass": None}
    return out


def t3(profile: dict) -> dict:
    """T3: for every budget > 0, cross-fitted NLL at T_B <= NLL at T = 1 + 0.002. A thinking model (any profile whose
    `thinking.capable` is not false) must carry, for EVERY budget > 0 of its run (`comparable.budgets`), an entry with
    finite `nll` and `nll_T`: one missing, None or non-finite is the model's `error`, naming the budget, never a
    silent skip. A budget-0-only profile (`thinking.capable` false) has nothing to judge and is no error."""
    budgets = profile.get("budgets") or {}
    capable = (profile.get("thinking") or {}).get("capable") is not False
    run = (profile.get("comparable") or {}).get("budgets")
    problems: list[str] = []
    if capable and not (isinstance(run, list) and all(type(b) is int for b in run)):
        problems.append("the profile records no run budgets (comparable.budgets)")
        run = []
    wanted = sorted({b for b in run or [] if b > 0} if capable else set())
    present = sorted({int(b) for b in budgets if int(b) > 0} | set(wanted))
    per = {}
    for b in present:
        e = budgets.get(str(b))
        if not isinstance(e, dict):
            problems.append(f"budget {b}: no entry in the profile")
            continue
        bad = [k for k in ("nll", "nll_T") if not finite_number(e.get(k))]
        if bad:
            if b in wanted or "nll_T" in e:
                problems.append(f"budget {b}: {' and '.join(f'{k} {e.get(k)!r}' for k in bad)}, not a finite number")
            continue
        per[str(b)] = {"nll_1": e["nll"], "nll_T": e["nll_T"], "T": e.get("T"), "pass": e["nll_T"] <= e["nll"] + T3_SLACK}
    out: dict[str, Any] = {"budgets": per, "pass": _all3([v["pass"] for v in per.values()] + ([None] if problems else []))
                           if per or problems else None}
    if problems:
        out["error"] = "; ".join(problems)
    return out


T4_EXPECTED = {"think_budget_x": 422, "think_budget_40000": 422, "over_think_max": 422, "image_with_budget": 422,
               "gemma_budget": 422, "past_n_ctx": 413, "probe_non_loopback": 403, "ok": 200}


def t4(cases: Mapping[str, int | None], *, disconnect_clear: bool | None, self_test: bool | None,
       timings_ok: bool | None, case_errors: Mapping[str, str] | None = None) -> dict:
    """T4 (the serve contract): every status of T4_EXPECTED, X-Callo-Timings' think fields present only at B > 0
    (`timings_ok`), a client disconnect leaving no processing slot within 2 s, the think self-test. Any of them not
    measured (a case absent or None, a check None) is listed in `not_testable` and makes T4 incomplete (`error`): T4
    never passes on part of its contract. A measured failure is a FAIL either way. `case_errors` says why a case the
    bench sent was not measured (`image_case`: the request did not carry its image), in the error beside its name."""
    per = {k: {"expected": v, "got": cases.get(k), "pass": cases.get(k) == v} for k, v in T4_EXPECTED.items()
           if cases.get(k) is not None}
    checks = {"timings_ok": timings_ok, "disconnect_clear": disconnect_clear, "self_test": self_test}
    missing = [k for k in T4_EXPECTED if cases.get(k) is None] + [k for k, v in checks.items() if v is None]
    why = {k: v for k, v in (case_errors or {}).items() if k in missing and v}
    failed = any(not v["pass"] for v in per.values()) or any(v is False for v in checks.values())
    out: dict[str, Any] = {"cases": per, "not_testable": missing, **checks,
                           "pass": False if failed else None if missing else True}
    if why:
        out["case_errors"] = why
    if missing:
        out["error"] = "incomplete: not testable: " + ", ".join(f"{k} ({why[k]})" if k in why else k for k in missing)
    return out


def t5(rcs: dict[str, int], failed: dict[str, list[str]]) -> dict:
    """T5 (the guard): (a) a resident 300-token prompt in a non-reserved slot under -kvu: exit 2 for `foreign_tokens`;
    (b) a continuous agent decode: exit 2 for `foreign_processing` or `foreign_tokens`; (c) the load removed and the
    slots erased: exit 0."""
    ok = (rcs.get("a") == 2 and "foreign_tokens" in failed.get("a", []) and rcs.get("b") == 2
          and bool({"foreign_processing", "foreign_tokens"} & set(failed.get("b", []))) and rcs.get("c") == 0)
    return {"rcs": rcs, "failed": failed, "pass": ok}


def t6(profile: dict, report: dict | None, *, status: str | None, rc: int | None = None) -> dict:
    """T6: the budget-0-only model: `unsupported_think_format`, budget 0 only, its OWN calibration done (`status`, its
    run.json record: another model's failure in the same menu run is not this feature's), the joint report's note,
    and its point among the joint report's `points` (it appears on the joint frontier). Run 1's exit (`rc`) is
    recorded, descriptive only."""
    thinking = profile.get("thinking") or {}
    name = profile.get("model_name")
    note = f"{name}: budget 0 only: thinks in a format callo cannot read"
    in_report = bool(report) and note in (report.get("notes") or [])
    points = (report or {}).get("points") or []
    on_frontier = any(isinstance(pt, dict) and pt.get("model") == name for pt in points)
    only0 = sorted(profile.get("budgets") or {}) == ["0"]
    return {"capable": thinking.get("capable"), "reason": thinking.get("reason"), "budget_0_only": only0, "status": status,
            "rc": rc, "report_note": in_report, "on_frontier": on_frontier,
            "pass": thinking.get("capable") is False and thinking.get("reason") == "unsupported_think_format" and only0
            and in_report and on_frontier and status == "done"}


def t7(broken: dict, resumed: dict, *, failed_name: str = "qwen9b") -> dict:
    """T7 (menu isolation): the broken rerun ends with at least one other model, every other model done, and
    `failed_name` failed with a reason in run.json (exit 1); the resume completes every model (exit 0) and runs only
    `failed_name` (the others keep their `started`). Each shortfall is a named problem, never a KeyError."""
    bm, rm = broken.get("models") or {}, resumed.get("models") or {}
    problems: list[str] = []
    f = bm.get(failed_name)
    if f is None:
        problems.append(f"the broken run.json has no {failed_name}")
    elif f.get("status") != "failed" or not f.get("reason"):
        problems.append(f"the broken run.json has {failed_name} {f.get('status')} with reason {f.get('reason')!r}, "
                        f"not failed with a reason")
    others = [n for n in bm if n != failed_name]
    if not others:
        problems.append(f"the broken run ran no model besides {failed_name}")
    problems += [f"the broken run's {n} is {bm[n].get('status')}, not done" for n in others if bm[n].get("status") != "done"]
    if broken.get("rc") != 1:
        problems.append(f"the broken run exited {broken.get('rc')}, not 1")
    problems += [f"the resumed run.json has no {n}" for n in bm if n not in rm]
    problems += [f"the resumed run's {n} is {m.get('status')}, not done" for n, m in rm.items() if m.get("status") != "done"]
    if resumed.get("rc") != 0:
        problems.append(f"the resumed run exited {resumed.get('rc')}, not 0")
    only_new = all(n in rm and rm[n].get("started") == bm[n].get("started") for n in others)
    if others and not only_new:
        problems.append(f"the resume reran a model other than {failed_name}")
    return {"broken": {n: m.get("status") for n, m in bm.items()}, "broken_rc": broken.get("rc"),
            "resumed": {n: m.get("status") for n, m in rm.items()}, "resumed_rc": resumed.get("rc"),
            "only_the_failed_model_reran": only_new, "problems": problems, "pass": not problems}


T8_REFUSAL = "different data"     # compare's own reason (`... was measured on different data (...)`)


def t8(rc_join: int, rc_refuse: int, refuse_err: str) -> dict:
    """T8: the join exits 0; the refusal exits 2 FOR compare's reason, "different data" in its stderr (a malformed
    profile's exit 2 is no PASS)."""
    lines = [ln for ln in (refuse_err or "").splitlines() if ln.strip()]
    reason = next((ln for ln in lines if T8_REFUSAL in ln), lines[-1] if lines else "")
    return {"join_rc": rc_join, "refuse_rc": rc_refuse, "refuse_reason": reason[:300],
            "pass": rc_join == 0 and rc_refuse == 2 and T8_REFUSAL in (refuse_err or "")}


CARRIED = "the serve part's own verdict, carried: its readouts predate summary.json keeping them"


def t9(raw_paths: Mapping[str, str | Path], served: Mapping[str, dict], *,
       expected_raw: Collection[str] = T9_RAW_MODELS, expected_served: Collection[str] = SERVED_MODELS,
       judged: Mapping[str, dict] | None = None) -> dict:
    """T9 (re-prefill): calibrate: of the deliberate readouts read AT n_gen (per question, the phase-B readout
    whose k >= n_gen; a cut re-prefills from the frame on hybrid and SWA models by design and is not a readout at
    n_gen) >= 99 % have a first-pass `read_prompt_n` <= 4; served: the same share over the served readouts whose
    trace closed on its own (`think_close` natural or eos; forced, budget, deadline and budget0 are cuts), from X-Callo-Timings' `read_prompt_n`; a session whose
    readouts lack it, or that has no natural close, is that half's ERROR (not testable), never judged on the server's
    `readout_reprefill` counter (every readout, cuts included, above 8: not this gate's criterion; a
    text fallback still FAILs it). `text_fallback` = 0 everywhere. A single-option question's row
    (`trace: null`, no generation) has no readout at n_gen and is skipped. An incomplete raw row (`check_rows`) is that
    model's `error` and the gate's `pass` None (an ERROR verdict) unless another entry fails; a partial last line
    `read_raw` dropped is a `note`. Both halves are required: a model of `expected_raw` without a raw file, or of
    `expected_served` without a served session, is that model's `error`. `judged`: served-half entries an earlier serve
    part already judged and whose readouts summary.json did not keep (a summary written before it kept them); each is
    used as it stands, with the CARRIED note, for a model `served` lacks."""
    judged = {n: v for n, v in (judged or {}).items() if n not in served and isinstance(v, dict) and "pass" in v}
    cal: dict[str, dict] = {n: {"error": "no raw file (expected for T9's calibrate half)", "pass": None}
                            for n in expected_raw if n not in raw_paths}
    for name, path in raw_paths.items():
        if not isinstance(path, (str, Path)):
            cal[name] = {"error": "no raw file (expected for T9's calibrate half)", "pass": None}
            continue
        try:
            raw = read_raw(path)
            check_rows(raw, path)                        # read_raw is lenient; an acceptance gate is not
        except InvalidInput as e:
            cal[name] = {"error": str(e), "pass": None}
            continue
        n = ok = fallback = 0
        for r in _latest(raw.rows, "B").values():
            if not isinstance(r.get("trace"), dict):
                continue
            n_gen = int(r["trace"]["n_gen"])
            reads = r.get("budgets") or {}
            fallback += sum(1 for x in reads.values() if x.get("text_fallback"))
            at = next((x for x in reads.values() if int(x["k"]) >= n_gen), None)
            if at is None:
                continue
            n += 1
            pn = at.get("read_prompt_n")
            ok += int(pn is not None and int(pn) <= T4_PROMPT_N)
        cal[name] = {"n": n, "at_n_gen_ok": ok, "text_fallback": fallback,
                     "source": "raw phase-B rows: the readout at n_gen, read_prompt_n <= 4",
                     "pass": n > 0 and ok / n >= 1 - T9_REPREFILL and fallback == 0}
        if raw.partial:
            cal[name]["note"] = PARTIAL
    srv: dict[str, dict] = {n: {"error": "no serve-path session (expected for T9's served half)", "pass": None}
                            for n in expected_served if n not in served and n not in judged}
    srv.update({n: {**v, "note": CARRIED} for n, v in judged.items()})
    for name, s in served.items():
        if not isinstance(s, dict):
            srv[name] = {"error": "no serve-path session (expected for T9's served half)", "pass": None}
            continue
        think = s.get("think") or {}
        fallback = int(think.get("text_fallback", 0) or 0)
        readouts = s.get("readouts") or []
        natural = [x for x in readouts if x.get("think_close") in ("natural", "eos")]   # allow-list: not forced/budget/deadline/budget0
        if natural and all(x.get("read_prompt_n") is not None for x in natural):
            ok = sum(1 for x in natural if int(x["read_prompt_n"]) <= T4_PROMPT_N)
            srv[name] = {"n": len(natural), "at_n_gen_ok": ok, "text_fallback": fallback,
                         "source": "X-Callo-Timings read_prompt_n <= 4 on the natural closes",
                         "pass": ok / len(natural) >= 1 - T9_REPREFILL and fallback == 0}
        else:                                            # never judged on the readout_reprefill counter (a proxy)
            why = ("not testable: no read_prompt_n in X-Callo-Timings" if natural else
                   f"not testable: no natural close among {len(readouts)} served readouts")
            srv[name] = {"error": why, "pass": False if fallback else None, "text_fallback": fallback}
    passes = [v["pass"] for v in (*cal.values(), *srv.values())]
    return {"calibrate": cal, "served": srv, "pass": _all3(passes)}


def t10_gate(entries: Mapping[str, dict], *, expected: Collection[str] = SERVED_MODELS) -> dict:
    """The summary's T10 entry: each serve session's `t10` result, and an ERROR entry for every expected session
    without one."""
    out: dict[str, dict] = {}
    for name in [*expected, *(n for n in entries if n not in expected)]:
        e = entries.get(name)
        out[name] = e if isinstance(e, dict) and "pass" in e else {"error": "no serve-path session result (expected for "
                                                                            "T10)", "pass": None}
    return out


def t10(served_ms: dict[str, float], profile: dict, budget: int, *, stopped: str | None = None) -> dict:
    """T10 (served vs profile latency): over the same question ids, at least SERVE_N (48) of them, the served p50
    (engine time from X-Callo-Timings) within 15 % of the profile's p50 at the pick budget. Different seeds, the same
    trace-length distribution. Fewer shared questions is not testable: the entry's `error`, never a PASS on a few;
    `stopped` (`served_latencies` gave up on too many non-200 replies) is the error itself."""
    if stopped:
        return {"n": len(served_ms), "budget": budget, "pass": None, "error": stopped}
    rows = {r["q"]: r for r in profile.get("rows") or []}
    fields = list(profile.get("row_fields") or [])
    i = fields.index("lat_ms") if "lat_ms" in fields else 3
    pairs = [(ms, rows[q]["b"][str(budget)][i]) for q, ms in served_ms.items()
             if q in rows and str(budget) in (rows[q].get("b") or {})]
    if len(pairs) < SERVE_N:
        return {"n": len(pairs), "budget": budget, "pass": None,
                "error": f"not testable: {len(pairs)} shared questions < {SERVE_N}"}
    served_p50, profile_p50 = statistics.median(p[0] for p in pairs), statistics.median(p[1] for p in pairs)
    err = abs(served_p50 - profile_p50) / profile_p50 if profile_p50 else None
    # the verdict stays on p50, as pre-registered; beside it, never gated, the mean (`callo compare`'s default pick
    # statistic), p95 (nearest rank, from few questions) and max over the same questions; p99 is max at n = 48
    served, prof = sorted(p[0] for p in pairs), sorted(p[1] for p in pairs)
    r95 = nearest_rank(len(pairs), 0.95) - 1
    served_mean, profile_mean = statistics.fmean(served), statistics.fmean(prof)
    return {"n": len(pairs), "budget": budget, "served_p50_ms": served_p50, "profile_p50_ms": profile_p50, "rel_err": err,
            "pass": err is not None and err <= T10_ERR,
            "served_mean_ms": served_mean, "profile_mean_ms": profile_mean,
            "mean_ratio": served_mean / profile_mean if profile_mean else None,
            "served_p95_ms": served[r95], "profile_p95_ms": prof[r95], "p95_from": tail_counts(len(pairs))["p95"],
            "served_max_ms": served[-1], "profile_max_ms": prof[-1],
            "pick_latency": (profile.get("pick") or {}).get("latency")}     # the rule that picked the budget


def groups_of(data_path: str | Path) -> dict[str, str]:
    """{item id (`_meta.id`): group} from the bench's data file: composition / date-score / control (GROUPS), else the
    record's own source."""
    out: dict[str, str] = {}
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            meta = r.get("_meta") or {}
            if meta.get("id") is not None:
                out[str(meta["id"])] = GROUPS.get(str(meta.get("source")), str(meta.get("source")))
    return out


def _seed(p: dict) -> int:
    return int(((p.get("comparable") or {}).get("seed")) or 0)


def _strata(p: dict, uids: Sequence[str]) -> list[str] | None:
    """The rows' sources, the strata calibrate resampled its own intervals in, when every row carries
    one; else None (plain clusters)."""
    of = {r["q"]: r.get("source") for r in p.get("rows") or []}
    sources = [of.get(u) for u in uids]
    return [str(s) for s in sources] if all(s is not None for s in sources) else None


def dacc_by_group(p: dict, groups: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Δacc(B − 0) per source group of the profile's rows (the "TielCoder composition positive at 256" check):
    {group: {"n": shared, budget: [mean, lo, hi]}}, paired cluster bootstraps (point − reference: B − 0) over ONE
    fixed row set per group (the group's questions read at 0 and every positive budget, so `n` holds for every
    budget), within the rows' sources when they carry them."""
    rows = correct_at(p)
    clusters = {r["q"]: str(r.get("cluster")) for r in p.get("rows") or []}
    budgets = sorted(int(b) for b in (p.get("budgets") or {}) if int(b) > 0)
    by_group: dict[str, list[str]] = {}
    for uid in rows:
        by_group.setdefault(groups.get(uid.rsplit("/", 1)[0], "other"), []).append(uid)
    out: dict[str, dict[str, Any]] = {}
    for g, uids in sorted(by_group.items()):
        shared = [u for u in uids if all(b in rows[u] for b in (0, *budgets))]
        if not shared or not budgets:
            continue
        entry: dict[str, Any] = {"n": len(shared)}
        for b in budgets:
            entry[str(b)] = list(paired_bootstrap([rows[u][0] for u in shared], [rows[u][b] for u in shared],
                                                  [clusters.get(u, u) for u in shared], seed=_seed(p),
                                                  strata=_strata(p, shared)))
        out[g] = entry
    return out


def descriptive_pairs(profiles: dict[str, dict], pairs: Sequence[tuple[str, str]]) -> dict[str, dict]:
    """The paired Δacc per budget between two profiles of the SAME questions (Run 1's `tiel` at its
    GGUF's 1.0 against Run 3's `tiel-fixed` at 0.6): {"a vs b": {"n": shared, budget: [mean(a − b), lo, hi]}}, the
    point − reference sign of every Δacc (the first named model is the point: tiel − tiel-fixed), over ONE fixed row
    set (the questions both read at every shared budget)."""
    out: dict[str, dict] = {}
    for a, b in pairs:
        pa, pb = profiles.get(a), profiles.get(b)
        if pa is None or pb is None:
            continue
        ra, rb = correct_at(pa), correct_at(pb)
        clusters = {r["q"]: str(r.get("cluster")) for r in pa.get("rows") or []}
        common = sorted({int(k) for k in (pa.get("budgets") or {})} & {int(k) for k in (pb.get("budgets") or {})})
        shared = [u for u in ra if u in rb and all(bud in ra[u] and bud in rb[u] for bud in common)]
        if not shared or not common:
            continue
        entry: dict[str, Any] = {"n": len(shared)}
        for bud in common:
            entry[str(bud)] = list(paired_bootstrap([rb[u][bud] for u in shared], [ra[u][bud] for u in shared],
                                                    [clusters.get(u, u) for u in shared], seed=_seed(pa),
                                                    strata=_strata(pa, shared)))
        out[f"{a} vs {b}"] = entry
    return out


def latencies(p: dict) -> dict[str, dict]:
    """Per budget its n and the point estimates (ms) of the five latency statistics, an older profile's absent
    ones recomputed from its rows (`think.point_stats`, bit for bit what calibrate stores); a budget whose latencies
    cannot be read is an `unreadable` entry, shown as an n/a row."""
    out: dict[str, dict] = {}
    budgets = p.get("budgets") if isinstance(p.get("budgets"), dict) else {}
    for b, e in sorted(budgets.items(), key=lambda kv: (not str(kv[0]).isdecimal(), int(kv[0]) if str(kv[0]).isdecimal() else 0,
                                                       str(kv[0]))):
        if not str(b).isdecimal() or not isinstance(e, dict):
            out[str(b)] = {"n": e.get("n") if isinstance(e, dict) and isinstance(e.get("n"), int) else None,
                           "unreadable": True}
            continue
        try:
            st = point_stats(p, e, "b", b)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            out[b] = {"n": e.get("n") if isinstance(e, dict) else None, "unreadable": True}
            continue
        if any(v is None for v in st.values()) or not isinstance(e.get("n"), int):
            out[b] = {"n": e.get("n") if isinstance(e.get("n"), int) else None, "unreadable": True}
            continue
        out[b] = {"n": e["n"], "mean": st["lat_mean_ms"][0], "p50": st["lat_p50_ms"][0], "p95": st["lat_p95_ms"][0],
                  "p99": st["lat_p99_ms"][0], "max": float(st["lat_max_ms"])}
    return out


def repick(p: dict) -> dict | None:
    """The profile's recorded pick (under the statistic it was made on) beside `callo compare`'s re-pick under
    the default preference (mean), each with its target and its paired Δacc against it, and for the re-pick the
    discordant questions behind that Δacc (the pick right and its target wrong, then the reverse: a percentile bootstrap is
    anti-conservative on a handful of them). None for a profile compare cannot read."""
    if not isinstance(p, dict) or malformed(p) is not None:
        return None
    try:
        j = joint([p])
    except CalloError:
        return None
    rec = p.get("pick") if isinstance(p.get("pick"), dict) else {}
    c = j.chosen
    out = {"recorded": rec.get("label"), "recorded_on": rec.get("latency"), "recorded_best": rec.get("best"),
           "recorded_dacc_best": rec.get("dacc_best"), "mean": c.point.label if c.point else None,
           "mean_best": c.best.label if c.best else None,
           "mean_dacc_best": list(c.dacc_best) if c.dacc_best is not None else None, "discordant": None}
    if c.point is not None and c.best is not None:
        rows = rows_at(p)
        kp, kb = (c.point.kind, c.point.value), (c.best.kind, c.best.value)
        pairs = [(rows[u][kp][0], rows[u][kb][0]) for u in j.fixed]
        out["discordant"] = [sum(x > y for x, y in pairs), sum(y > x for x, y in pairs)]
    return out


def descriptive(profiles: dict[str, dict], *, data_path: str | Path | None = None) -> dict:
    """The descriptive results: Δacc(B − 0) with CIs (overall and, with the data file, per source group), the pick,
    the overhead, the forced share, the drift, n_cut; and the five latency statistics per budget and the recorded
    pick beside the mean re-pick."""
    groups = groups_of(data_path) if data_path else {}
    out = {}
    for name, p in profiles.items():
        budgets = p.get("budgets") or {}
        out[name] = {"dacc": {b: e.get("dacc") for b, e in budgets.items() if int(b) > 0},
                     "dacc_by_group": dacc_by_group(p, groups) if groups and p.get("rows") else {},
                     "acc": {b: e.get("acc") for b, e in budgets.items()},
                     "forced": {b: e.get("forced") for b, e in budgets.items() if int(b) > 0},
                     "think_mean": {b: e.get("think_mean") for b, e in budgets.items() if int(b) > 0},
                     "n_cut": {b: a.get("n_cut") for b, a in ((p.get("audit") or {}).get("budgets") or {}).items()},
                     "pick": (p.get("pick") or {}).get("label"), "overhead_s": p.get("calibration_overhead_s"),
                     "drift": p.get("drift"), "sampler": ((p.get("comparable") or {}).get("sampling") or {}).get("policy"),
                     "decode_tok_s": ((p.get("environment") or {}).get("run") or {}).get("decode_tok_s_median"),
                     "lat": latencies(p), "repick": repick(p)}
    return out


# ---- the orchestration: Runs 1-3, T7, T8, the serve-path sessions (T4, T9, T10), T5 ------------------------------

# T4's image case needs a projector. It runs on the small qwen9b launch (`small_session`), which measures no latency
# and carries no --think-profile: the thinking models' serve sessions launch exactly as their profiles were measured
# (no mmproj in Run 1's menu), so no `mmproj differs from the profile's` warning and T10 compares like with like.
SERVE_MMPROJ: dict[str, str] = {"qwen9b": QWEN9B_MMPROJ}
IMAGE_REFUSAL = "text-only"      # proxy.think_plan's 422 for media at B > 0: "deliberate readouts are text-only in ..."
PARTS = ("run1", "run2", "run3", "t7", "t8", "serve", "t5")
GPU_PARTS = ("run1", "run2", "run3", "t7", "serve", "t5")   # t8 only compares profiles already written
BUDGET0_MODEL = "gemma-e4b"      # T6's budget-0-only model and T4's gemma_budget case
SETTLE_S: float | None = None    # the menu's --settle between models; None: callo's own default (30 s)
DISCONNECT_AFTER_S, DISCONNECT_CLEAR_S = 1.0, 2.0
PORT_FREE_S = 30.0               # the same-URL restart waits this long for the stopped callo to release its port
GPU_RECHECK = (3, 5.0)           # before a part: a model server seen is re-checked this often, this far apart
EXPECTED_BUILD = "4ebdf2c"       # llama.cpp-master @ 4ebdf2c; a mismatch is a start-up warning
FORBIDDEN_PORTS = (8080, 8130)   # other sessions' servers: the bench never binds or calls them
THINK_TIMING_FIELDS = ("think_budget", "think_tokens", "think_close", "gen_ms", "read_prompt_n")   # X-Callo-Timings, B > 0
_STOP_CODES = {128 + signal.SIGTERM, 128 + signal.SIGINT}    # acceptance_v02's stop handlers raise SystemExit(128 + n)
PNG_1x1 = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAA"
           "SUVORK5CYII=")


def _calibrate(argv: list[str]) -> tuple[int, str]:
    """Seam: a `callo calibrate` / `callo compare` command line in this process; (exit code, its stderr), the stderr
    tee'd to this process's own (a 3-hour run still shows its progress; the guard's `(failed conditions: ...)` line is
    read from the copy). An argparse usage error inside cli.main is its exit code; a stop signal propagates."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(_Tee(buf, sys.stderr)):
        try:
            rc = cli.main(argv)
        except SystemExit as e:
            if e.code in _STOP_CODES:
                raise
            rc = e.code if isinstance(e.code, int) else 2
    return rc, buf.getvalue()


def t5_probe_argv(url: str, data: Path, out: Path) -> list[str]:
    """One T5 probe: a short `calibrate --think-budgets --url` against a session `Callo` launched cache off (the
    bench's `--no-cache`), so the probe says `--no-cache` too (serve's default is on since 0.3.0)."""
    return ["calibrate", "--think-budgets", "0,256", "--url", url, "--data", str(data), "--out", str(out),
            "--warmup", "0", "--audit", "0", "--no-cache"]


def _launch(name: str, out: Path, *, model: str, flags: Sequence[str], ctx: int, cache: bool = False) -> Any:
    """Seam: a serve-path session in launch mode (the smoke test swaps in callo on the thinking fake). `Callo` passes
    `--port 0` (an ephemeral port; a later `--port` in `flags` wins, the same-URL restart's) and `--no-cache`, as
    this bench's profiles were measured, unless `cache` (the deadline bench measures callo as shipped, cache on)."""
    return Callo(out / f"callo_{name}", llama_server=LLAMA_SERVER, model=model, ctx=ctx, parallel=PARALLEL,
                 flags=list(flags), cache=cache)


def _port_of(url: str | None) -> int | None:
    try:
        return urlsplit(url).port if url else None
    except ValueError:
        return None


def port_problems(urls: Sequence[str | None]) -> list[str]:
    """Every URL of the bench's own servers that is on another session's port (FORBIDDEN_PORTS)."""
    return [f"{u} is on port {_port_of(u)}, another session's server: the bench never binds or calls 8080 or 8130"
            for u in urls if _port_of(u) in FORBIDDEN_PORTS]


def _start(name: str, out: Path, *, model: str, flags: Sequence[str], ctx: int, cache: bool = False) -> Any:
    """Every serve-path launch: refused before it starts when `flags` ask for a forbidden port, and stopped and refused
    when it came up on one (callo's URL or its llama-server's). `cache=True` (the deadline bench's sessions) launches
    callo with `--cache`; `_launch` hears of it only then, so its stand-ins keep their signature."""
    asked = [flags[i + 1] for i, f in enumerate(flags[:-1]) if f == "--port"]
    if asked and asked[-1] in {str(p) for p in FORBIDDEN_PORTS}:
        raise RuntimeError(f"refusing to launch {name} on port {asked[-1]}: another session's server (never "
                           f"{' or '.join(map(str, FORBIDDEN_PORTS))})")
    c = _launch(name, out, model=model, flags=flags, ctx=ctx, **({"cache": True} if cache else {}))
    problems = port_problems([c.url, getattr(c, "upstream", None)])
    if problems:
        c.finish()
        raise RuntimeError(f"stopped {name}: " + "; ".join(problems))
    return c


def _fresh_dir(base: Path) -> Path:
    """`base` when it does not exist or is empty, else the first of `base-2`, `base-3`, ... that does not: a run that
    must start fresh (T7's broken run, T5's probes) never resumes or refuses on an earlier session's files."""
    d, k = base, 1
    while d.exists() and any(d.iterdir()):
        k += 1
        d = base.with_name(f"{base.name}-{k}")
    return d


def _run_calibrate(out_dir: Path, data: Path, models: dict[str, dict], *extra: str, budgets: str | None = None,
                   audit: int | None = None, allow_gpu_busy: bool = False) -> tuple[int, dict | None, str]:
    """One menu run: the menu file next to `out_dir`, then `callo calibrate --think-budgets --menu`; (rc, run.json,
    stderr). The menu's launch passes `--cache-ram 0` to every llama-server. A DIR that already holds
    a run.json is resumed (`--resume`: a rerun of the bench continues where it stopped). A 130 (Ctrl-C) is this bench's
    KeyboardInterrupt (`main` records `aborted`, exit 3) and a 2 (a refusal, the GPU rule say) is an error naming the
    cause, so the later parts never run on missing profiles."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    menu_path = out_dir.parent / f"{out_dir.name}.menu.json"
    menu_path.write_text(json.dumps(menu_doc(models, llama_server=LLAMA_SERVER), indent=1) + "\n")
    resume = ["--resume"] if (out_dir / "run.json").exists() and "--resume" not in extra else []
    rc, err = _calibrate(["calibrate", "--think-budgets", budgets or BUDGETS, "--data", str(data), "--menu", str(menu_path),
                          "--out", str(out_dir), "--audit", str(AUDIT if audit is None else audit), "--seed", str(SEED),
                          "--no-cache",          # the bench's runs are cache off; serve's default is on since 0.3.0
                          *(["--settle", f"{SETTLE_S:g}"] if SETTLE_S is not None else []),
                          *(["--allow-gpu-busy"] if allow_gpu_busy else []), *extra, *resume])
    if rc == 130:
        raise KeyboardInterrupt(f"calibrate --menu {menu_path.name} was interrupted")
    if rc == 2:
        tail = err.strip().splitlines()[-1] if err.strip() else "no stderr"
        raise RuntimeError(f"calibrate --menu {menu_path.name} refused (exit 2): {tail}")
    run_path = out_dir / "run.json"
    run_json = json.loads(run_path.read_text()) if run_path.exists() else None
    return rc, run_json, err


def _post_full(url: str, body: dict) -> tuple[int, dict, Any]:
    """(status, lower-cased headers, reply body); -1 (and no headers or body) when the request itself failed."""
    try:
        return lib.post(url, body)
    except (OSError, http.client.HTTPException, ValueError) as e:
        log.warning("request failed: %s: %s", type(e).__name__, e)
        return -1, {}, None


def _post(url: str, body: dict) -> tuple[int, dict]:
    """(status, lower-cased headers); -1 when the request itself failed."""
    status, headers, _ = _post_full(url, body)
    return status, headers


def _status(url: str, body: dict) -> int:
    return _post(url, body)[0]


def image_request(base: dict) -> dict:
    """T4's image case: `base` with one image in the top-level `images` field (the contract's media field, which
    `parse_request` reads; the shape of tests/test_think_serve.py's IMG) and a budget > 0."""
    return {**base, "images": [PNG_1x1], "think_budget": 256}


def image_case(status: int, headers: Mapping[str, str], body: Any) -> tuple[int | None, str | None]:
    """(the status T4 judges, None), or (None, why) when the reply shows the request did not reach the rule it tests:
    a 422 counts only with the media refusal's own reason (IMAGE_REFUSAL), a 200 only when X-Callo-Timings says the
    image was read (`kind` image: the server served a media request it must refuse, the feature's FAIL). Anything
    else (a 422 for another reason, a 200 read as text: the image never attached; a 501: no projector; a failed
    request) is the bench's ERROR, never a pass or fail of the product."""
    detail = body.get("detail") if isinstance(body, dict) else body
    if status == 422:
        if IMAGE_REFUSAL in str(detail):
            return 422, None
        return None, f"a 422 for another reason ({str(detail)[:200]!r}): the request was malformed"
    if status == 200:
        try:
            entries = json.loads(headers.get("x-callo-timings") or "[]")
        except ValueError:
            entries = []
        kinds = [e.get("kind") for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
        if "image" in kinds:
            return 200, None
        return None, f"a 200 read as {kinds or 'no X-Callo-Timings'}: the request carried no image"
    return None, f"status {status}" + (f" ({str(detail)[:200]!r})" if detail else "") + ": the image was not judged"


def _split_flags(flags: Sequence[str]) -> tuple[list[str], list[str]]:
    """callo's flags and llama-server's passthrough (everything after `--`)."""
    if "--" in flags:
        i = list(flags).index("--")
        return list(flags[:i]), list(flags[i + 1:])
    return list(flags), []


def _join_flags(callo: Sequence[str], server_args: Sequence[str]) -> list[str]:
    return [*callo, *(["--", *server_args] if server_args else [])]


def profile_launch(profile: dict) -> dict:
    """What a serve launch repeats of the profile's own launch, so `--think-profile` finds nothing that differs: the
    llama-server passthrough it was measured with (`provenance.server_args`: Run 1's menu passes `--cache-ram 0`) and
    its projector (`environment.model.mmproj_path`: None, Run 1's menu loads none)."""
    prov = profile.get("provenance") or {}
    return {"server_args": [str(a) for a in prov.get("server_args") or []],
            "mmproj": ((profile.get("environment") or {}).get("model") or {}).get("mmproj_path")}


def _think_fields_ok(status: int, headers: Mapping[str, str], *, positive: bool) -> bool | None:
    """T4's X-Callo-Timings rule on one /v1/systemone reply: at B > 0 (`positive`) every entry carries the think fields,
    at B = 0 none does. None when it cannot be told (not a 200, no header, a truncated header)."""
    if status != 200:
        return None
    try:
        entries = json.loads(headers.get("x-callo-timings") or "")
    except ValueError:
        return None
    if not isinstance(entries, list) or not entries or not all(isinstance(e, dict) and "q" in e for e in entries):
        return None
    if positive:
        return all(all(k in e for k in THINK_TIMING_FIELDS) for e in entries)
    return not any(k in e for e in entries for k in THINK_TIMING_FIELDS)


def _merge_cases(into: dict[str, int | None], new: Mapping[str, int | None]) -> None:
    """T4's statuses from several sessions: a measured status fills an unmeasured one, and a status other than the
    expected one replaces a passing one (a failure in any session stands)."""
    for k, v in new.items():
        if into.get(k) is None or (v is not None and v != T4_EXPECTED.get(k)):
            into[k] = v


def _lan_ip() -> str | None:
    """A non-loopback address of this box, for the 403 case; None when it has none."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        return None
    return None if ip.startswith("127.") else ip


def _disconnect_clears(c: Any, body: dict) -> bool | None:
    """T4: a client that disconnects 1 s into a 1024-token generation leaves no processing slot within 2 s."""
    u = urlsplit(c.url)
    payload = json.dumps({**body, "think_budget": 1024}).encode()
    try:
        with socket.create_connection((u.hostname, u.port), timeout=10) as s:
            s.sendall(f"POST /v1/systemone HTTP/1.1\r\nHost: {u.hostname}\r\nContent-Type: application/json\r\n"
                      f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
            time.sleep(DISCONNECT_AFTER_S)
    except OSError as e:
        log.warning("disconnect probe failed: %s", e)
        return None
    time.sleep(DISCONNECT_CLEAR_S)
    try:
        slots = c.stats().get("slots") or []
    except (OSError, http.client.HTTPException, ValueError):
        return None
    return not any(s.get("is_processing") for s in slots if isinstance(s, dict))


def serve_session(name: str, out: Path, profile_path: Path, data: Path, *, guard: bool = False) -> dict:
    """A serve-path session on `name` with its profile and pick budget: the T4 statuses (with the X-Callo-Timings rule,
    the disconnect and the self-test), the T10 latencies over the first SERVE_N measured questions, the served
    readouts for T9, and (on `guard`) the T5 guard probe, whose failure is T5's error, never the session's. One launch,
    shaped as the profile's own (`profile_launch`: its llama-server args, its projector or none), so T10 compares like
    with like and `--think-profile` warns of no difference; the image case runs here only when that launch has a
    projector (else on the small launch, IMAGE_MODEL)."""
    profile = load_profile(profile_path)
    positive = [int(b) for b in profile["budgets"] if int(b) > 0]
    pick = (profile.get("pick") or {}).get("budget")
    if pick:                                             # the profile's pick, a deliberate budget: T10 as defined
        budget, note = int(pick), "the profile's pick"
    elif positive:                                       # a budget-0 pick (or none): the session still has to exercise
        budget, note = max(positive), f"the profile picked {pick!r}; served at Bmax so T4 and T10 see a deliberate readout"
    else:
        budget, note = 0, "budget 0 only"
    launch = profile_launch(profile)
    mmproj = launch["mmproj"]
    flags = _join_flags(["--think-profile", str(profile_path), "--think-budget", str(budget),
                         *(["--mmproj", mmproj] if mmproj else [])], launch["server_args"])
    c = _start(name, out, model=MODELS[name]["model"], flags=flags, ctx=CTX)
    res: dict[str, Any] = {"pick": pick, "budget": budget, "budget_note": note, "flags": flags, "url": c.url}
    try:
        items, _ = load_data(data)
        by_uid = {it.uid: it for it in items}
        base = items[0].body
        url = f"{c.url}/v1/systemone"
        s0, h0 = _post(url, {**base, "think_budget": 0})
        s_b, h_b = _post(url, {**base, "think_budget": budget})
        res["timings_ok"] = _all3([_think_fields_ok(s0, h0, positive=False),
                                   _think_fields_ok(s_b, h_b, positive=True) if budget > 0 else None])
        res["cases"] = {
            "think_budget_x": _status(url, {**base, "think_budget": "x"}),
            "think_budget_40000": _status(url, {**base, "think_budget": 40000}),
            "over_think_max": _status(url, {**base, "think_budget": 4097}),
            "ok": s_b}
        if mmproj:
            res["cases"]["image_with_budget"], why = image_case(*_post_full(url, image_request(base)))
            if why:
                res["case_errors"] = {"image_with_budget": f"{name}: {why}"}
        res["disconnect_clear"] = _disconnect_clears(c, base) if positive else None
        res["self_test"] = (c.stats().get("self_test") or {}).get("think")
        served_ms, readouts, res["served_skipped"], stopped = served_latencies(
            lambda body: _post(url, body), profile.get("rows") or [], by_uid, budget)
        res["served_ms"] = served_ms
        res["served_readouts"] = readouts
        res["t10"] = {**t10(served_ms, profile, budget, stopped=stopped), "budget_note": note}     # why that budget
        res["t10_stopped"] = stopped                     # with served_ms and budget: T10 recomputed by _offline_gates
        res["stats_think"] = c.stats().get("think") or {}
        if guard:
            try:
                res["t5"] = guard_probe(c, _fresh_dir(out / "t5"), data)
            except Exception as e:  # noqa: BLE001 -- T5's setup failing is T5's ERROR, not the session's
                log.exception("T5 on %s failed", name)
                res["t5"] = {"error": f"T5 could not be judged: {type(e).__name__}: {e}", "pass": None}
    finally:
        c.finish()
    return res


def served_latencies(post: Any, rows: Sequence[dict], by_uid: Mapping[str, Any], budget: int,
                     n: int | None = None) -> tuple[dict[str, float], list[dict], dict[str, int], str | None]:
    """T10 / T9's served half: the profile's questions, in its row order, sent through /v1/systemone at `budget`
    (`post(body) -> (status, headers)`) until `n` (SERVE_N) of them are MEASURED: a 200 whose X-Callo-Timings carries
    the question's entry. A non-200 reply (a transient 503 or 504) or an unusable header is skipped and counted, never
    one of the 48 ("the first 48 measured questions"); fewer than `n` measured over every row is T10's ERROR. After
    SKIP_MAX non-200 replies the loop stops (persistent 504s each cost a full --s1-timeout) and says why.
    Returns ({q: engine ms}, [per readout: q, think_close, read_prompt_n, think_tokens, ms, gen_ms], {status or "no
    timings": count}, the stop reason or None). The engine ms is X-Callo-Timings' `ms` alone: it is the question's whole
    wall time in the engine, the generation included (`answer_parsed` starts its clock before `answer_deliberate`);
    `gen_ms` is the generation's share of it, kept for the record and never added to it."""
    n = SERVE_N if n is None else n
    served_ms: dict[str, float] = {}
    readouts: list[dict] = []                            # per served readout, for T9: the close and the first pass
    skipped: dict[str, int] = {}
    non_200 = 0
    for row in rows:
        if len(served_ms) >= n:
            break
        it = by_uid.get(row["q"])
        if it is None:
            continue
        status, headers = post({**it.body, "think_budget": budget})
        if status != 200:
            skipped[str(status)] = skipped.get(str(status), 0) + 1
            non_200 += 1
            if non_200 >= SKIP_MAX:
                counts = ", ".join(f"{k}: {v}" for k, v in skipped.items() if k != "no timings")
                return served_ms, readouts, skipped, (f"too many non-200 replies: {non_200} ({counts}) before {n} "
                                                       f"measured questions")
            continue
        try:
            timings = json.loads(headers.get("x-callo-timings") or "[]")
        except ValueError:
            timings = None
        if not (isinstance(timings, list) and timings and isinstance(timings[0], dict) and "ms" in timings[0]):
            skipped["no timings"] = skipped.get("no timings", 0) + 1
            continue
        # the engine time of a deliberate question is `ms` alone: the generation (`gen_ms`) is inside it
        served_ms[row["q"]] = float(timings[0]["ms"])
        readouts.append({"q": row["q"], "think_close": timings[0].get("think_close"),
                         "read_prompt_n": timings[0].get("read_prompt_n"), "think_tokens": timings[0].get("think_tokens"),
                         "ms": timings[0]["ms"], "gen_ms": timings[0].get("gen_ms")})
    return served_ms, readouts, skipped, None


def _wait_port_free(url: str, timeout: float) -> bool:
    """True once nothing accepts on `url`'s port (the stopped callo released it), within `timeout` seconds."""
    u = urlsplit(url)
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((u.hostname, u.port), timeout=1):
                pass
        except OSError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)


def same_url_restart(name: str, out: Path, *, model: str, flags: Sequence[str], ctx: int, url: str,
                     old_instance: str | None, body: dict) -> dict:
    """`callo serve` restarted on the URL a client already holds: wait for the stopped one's port to be free,
    launch again with `--port` that port, then, through the OLD URL (the client's view), check that /callo/stats
    answers with a NEW instance id (a fresh startup, not the old process still up) and that a /v1/systemone request
    is served (200). Not a gate: its `pass`, and on any shortfall its `error`, which the serve part reports as
    `errors["serve"]` (exit 1, in the doc). Restarted on loopback only: no `--host` from `flags` is reused (the 403
    case is the only LAN bind). llama-server's passthrough (after `--`) is kept as it was, `--port` going before it."""
    port = _port_of(url)
    res: dict[str, Any] = {"url": url, "port": port, "old_instance": old_instance}
    own, server_args = _split_flags(flags)
    loopback = [f for i, f in enumerate(own) if f != "--host" and (i == 0 or own[i - 1] != "--host")]
    again = _join_flags([*loopback, "--port", str(port)], server_args)
    problems: list[str] = []
    try:
        if not _wait_port_free(url, PORT_FREE_S):
            return {**res, "pass": False, "error": f"port {port} still taken {PORT_FREE_S:g} s after the stop"}
        c = _start(name, out, model=model, flags=again, ctx=ctx)
        try:
            res["flags"] = again
            res["same_url"] = _port_of(c.url) == port
            try:
                res["new_instance"] = lib.get(f"{url}/callo/stats").get("instance")
            except (OSError, http.client.HTTPException, ValueError) as e:
                res["new_instance"] = None
                problems.append(f"/callo/stats on {url} did not answer ({type(e).__name__}: {e})")
            res["served"] = _status(f"{url}/v1/systemone", {**body, "think_budget": 0})
        finally:
            c.finish()
    except Exception as e:  # noqa: BLE001 -- the restart's failure is reported, the session's results kept
        log.exception("same-URL restart of %s failed", name)
        return {**res, "pass": False, "error": f"{type(e).__name__}: {e}"}
    if not res["same_url"]:
        problems.append(f"the restart came up on port {_port_of(c.url)}, not {port}")
    if res["new_instance"] is not None and (not res["new_instance"] or res["new_instance"] == old_instance):
        problems.append(f"/callo/stats on {url} reports instance {res['new_instance']!r}, not a new startup "
                        f"(was {old_instance!r})")
    if res["served"] != 200:
        problems.append(f"/v1/systemone on {url} answered {res['served']}, not 200")
    res["pass"] = not problems
    if problems:
        res["error"] = "; ".join(problems)
    return res


def small_session(out: Path, profile_path: Path, data: Path) -> dict:
    """The -c 4096 launch of T4: a budget past n_ctx is 413, an image at a budget > 0 is 422 (with qwen9b's projector,
    SERVE_MMPROJ; `image_case` checks the image really attached), and (with a LAN address: the one `--host 0.0.0.0`
    bind) the probe from it is 403. Then the same-URL restart on this cheap launch, on loopback. It passes the
    profile's llama-server args (`profile_launch`) but no --think-profile: it measures neither latency nor temperature,
    and a profile measured without a projector would only warn that the mmproj differs."""
    lan = _lan_ip()
    mmproj = SERVE_MMPROJ.get("qwen9b")
    case_errors: dict[str, str] = {}
    if not mmproj:
        case_errors["image_with_budget"] = "qwen9b-small: no mmproj configured: the image was not sent"
    elif not Path(mmproj).is_file():                     # launch without it: only the image case drops out
        case_errors["image_with_budget"] = f"qwen9b-small: the mmproj {mmproj} is missing: the image was not sent"
        mmproj = None
    flags = _join_flags(["--think-budget", "0", *(["--mmproj", mmproj] if mmproj else []),
                         *(["--host", "0.0.0.0"] if lan else [])], profile_launch(load_profile(profile_path))["server_args"])
    model = MODELS["qwen9b"]["model"]
    items, _ = load_data(data)
    c = _start("qwen9b-small", out, model=model, flags=flags, ctx=SMALL_CTX)
    try:
        cases: dict[str, int | None] = {"past_n_ctx": _status(f"{c.url}/v1/systemone", {**items[0].body, "think_budget": SMALL_BUDGET})}
        if mmproj:
            cases["image_with_budget"], why = image_case(*_post_full(f"{c.url}/v1/systemone", image_request(items[0].body)))
            if why:
                case_errors["image_with_budget"] = f"qwen9b-small: {why}"
        if lan:
            port = c.url.rsplit(":", 1)[1]
            cases["probe_non_loopback"] = _status(f"http://{lan}:{port}/callo/think-probe",
                                                  {"request": items[0].body, "budgets": [0], "plan": True})
        else:
            cases["probe_non_loopback"] = None
        try:
            old_instance = c.stats().get("instance")
        except (OSError, http.client.HTTPException, ValueError):
            old_instance = None
    finally:
        c.finish()
    restart = same_url_restart("qwen9b-small", out, model=model, flags=flags, ctx=SMALL_CTX, url=c.url,
                               old_instance=old_instance, body=items[0].body)
    return {"cases": cases, "lan": lan, "flags": flags, "restart": restart,
            **({"case_errors": case_errors} if case_errors else {})}


def gemma_session(out: Path, profile_path: Path, data: Path) -> dict:
    """T4's `gemma-e4b` case: a budget on the budget-0-only model is 422; T6's profile check happens offline. Launched
    with the profile's llama-server args (`profile_launch`)."""
    launch = profile_launch(load_profile(profile_path))
    c = _start(BUDGET0_MODEL, out, model=MODELS[BUDGET0_MODEL]["model"],
               flags=_join_flags(["--think-profile", str(profile_path), "--think-budget", "0",
                                  *(["--mmproj", launch["mmproj"]] if launch["mmproj"] else [])], launch["server_args"]),
               ctx=CTX)
    try:
        items, _ = load_data(data)
        return {"cases": {"gemma_budget": _status(f"{c.url}/v1/systemone", {**items[0].body, "think_budget": 256})}}
    finally:
        c.finish()


def guard_probe(c: Any, out: Path, data: Path) -> dict:
    """T5 on a running session (`-kvu`, 4 slots): (a) a resident 300-token prompt in the last (non-reserved) slot, (b) a
    continuous decode, (c) the load removed and every slot erased; each a short `calibrate --think-budgets --url`
    into its own fresh DIR under `out`."""
    small = out / "t5.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    small.write_text("".join(json.dumps(r) + "\n" for r in _first_records(data, 4)))
    last = c.parallel - 1

    def probe(tag: str) -> tuple[int, list[str]]:
        rc, err = _calibrate(t5_probe_argv(c.url, small, out / tag))
        if rc == 130:                                    # a 2 is T5's measured outcome (the guard's refusal), never an error
            raise KeyboardInterrupt(f"the T5 probe {tag} was interrupted")
        return rc, failed_conditions(err)

    _fill(c, last, FILL_TOKENS)
    _seen(c, lambda slots, reserved: any(s.get("id") == last and last not in reserved and (s.get("n_prompt_tokens") or 0) > 0
                                         for s in slots), f"{FILL_TOKENS} tokens resident in non-reserved slot {last}")
    rc_a, failed_a = probe("a")
    _erase_all(c)
    stop = start_decode(c)
    try:
        _seen(c, lambda slots, reserved: any(s.get("is_processing") and s.get("id") not in reserved for s in slots),
              "a decode in a non-reserved slot")
        rc_b, failed_b = probe("b")
    finally:
        stop()
    _erase_all(c)
    _seen(c, lambda slots, _: not any(s.get("is_processing") or (s.get("n_prompt_tokens") or 0) > 0 for s in slots),
          "every slot idle and empty")
    rc_c, _ = probe("c")
    return {**t5({"a": rc_a, "b": rc_b, "c": rc_c}, {"a": failed_a, "b": failed_b}), "dir": str(out)}


def _first_records(data: Path, n: int) -> list[dict]:
    out = []
    with open(data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def run_profile(out: Path, run: str, name: str) -> tuple[Path | None, str | None]:
    """A menu run's profile the gates may use: (`<out>/<run>/<name>.think.json`, None) only when that run's
    run.json records the model `done` with that very profile; (None, None) when no profile file exists (the gates
    name it missing); (None, why) when the file exists but run.json does not vouch for it (the model failed: another
    model server appeared, its stamp failed, or it is still running; a different path; no readable run.json), so a
    profile the run itself rejected never passes a gate."""
    path = out / run / f"{name}.think.json"
    if not path.exists():
        return None, None
    try:
        rf = RunFile.load(out / run / "run.json")
    except InvalidInput as e:
        return None, f"{run}/run.json cannot vouch for {name}'s profile: {e}"
    rec = rf.data["models"].get(name)
    if rec is None:
        return None, f"{run}/run.json has no {name}"
    if rec.get("status") != "done":
        return None, f"{run}/run.json says {name} {rec.get('status')}" + (f": {rec['reason']}" if rec.get("reason") else "")
    recorded = rf.profile_path(name)
    if recorded is None or os.path.realpath(recorded) != os.path.realpath(path):
        return None, f"{run}/run.json records {name}'s profile as {recorded}, not {path}"
    return path, None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_state(root: Path = ROOT) -> dict:
    """The commit of the code that runs and whether its tracked files differ from it (`git status --porcelain
    --untracked-files=no`, its first lines kept)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    try:
        status = git("status", "--porcelain", "--untracked-files=no")
        return {"root": str(root), "commit": git("rev-parse", "HEAD"), "dirty": bool(status), "status": status.splitlines()[:50]}
    except (OSError, subprocess.SubprocessError) as e:
        return {"root": str(root), "commit": None, "dirty": None, "error": f"{type(e).__name__}: {e}"}


def _llama_server_version(binary: str) -> str:
    """Seam: `llama-server --version`'s output, recorded beside the binary's path. The tests stub it: under
    pytest no real llama-server runs, not even for its version."""
    if not binary:
        return f"unavailable: no llama-server set ({lib.SERVER_ENV})"
    try:
        p = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return f"unavailable: {type(e).__name__}: {e}"
    return (p.stdout + p.stderr).strip()[-2000:]


def disk_free_gib(paths: Sequence[str]) -> dict[str, float | None]:
    """Free GiB per path (None when it cannot be read), recorded at start."""
    out: dict[str, float | None] = {}
    for p in paths:
        try:
            out[p] = round(shutil.disk_usage(p).free / 1024**3, 1)
        except OSError:
            out[p] = None
    return out


def startup_warnings(parts: Sequence[str], version: str, disk: Mapping[str, float | None]) -> list[str]:
    """What will make a gate end ERROR, said before a 3-hour run rather than after it: no LAN address (T4's 403 case
    not testable), an mmproj missing or not configured (T4's image 422 not testable), a llama-server that is not
    EXPECTED_BUILD, and a disk near MIN_FREE_GIB."""
    out: list[str] = []
    if "serve" in parts:
        if _lan_ip() is None:
            out.append("no LAN address: T4's 403 case (probe_non_loopback) will be not testable, so T4 can only end ERROR")
        mmproj = SERVE_MMPROJ.get("qwen9b")                  # the small launch's: T4's only image case
        if mmproj is None:
            out.append("no mmproj configured for qwen9b (the small launch): T4's image 422 (image_with_budget) will be "
                       "not testable, so T4 can only end ERROR")
        elif not Path(mmproj).is_file():
            out.append(f"the mmproj for qwen9b (the small launch) is missing ({mmproj}): T4's image 422 "
                       f"(image_with_budget) will be not testable, so T4 can only end ERROR")
    if EXPECTED_BUILD not in version:
        out.append(f"llama-server --version does not name the expected build {EXPECTED_BUILD}: {version_line(version)!r}")
    for p, g in disk.items():
        if g is None or g < 2 * MIN_FREE_GIB:
            out.append(f"{p}: {'unknown' if g is None else f'{g:.1f} GiB'} free, near the {MIN_FREE_GIB:g} GiB the "
                       f"bench stops at between parts")
    return out


def _foreign_servers() -> list[tuple[int, str]]:
    """The model servers running now (between parts every server this bench started is stopped), re-checked
    GPU_RECHECK times so one that is still exiting is not mistaken for another session's. A ps that fails raises
    (`lib.running_gpu_servers` fails closed)."""
    tries, wait = GPU_RECHECK
    for i in range(tries):
        busy = lib.running_gpu_servers()
        if not busy or i == tries - 1:
            return busy
        time.sleep(wait)
    return []


def rerender(out: Path, doc: Path) -> int:
    """The doc re-rendered from what the bench persisted, READ-ONLY on `out`. It reads `summary.json` and the
    Run 1-3 profiles their run.json files mark done (`run_profile`, `load_profile`), recomputes IN MEMORY (they are
    recomputed, not recorded) the descriptive results (the five statistics, the mean re-pick) and T10 (`t10_now`: the unchanged p50 verdict with its
    mean, p95 and max beside it), and writes ONLY the doc. No model-server check, no `llama-server --version`, no
    data selection, no run.pid, no `_offline_gates`, no summary.json write: every other verdict renders as the run
    recorded it. A profile its run.json no longer vouches for is that profile's error in the doc, as in the run.
    0, or 2 without a usable `summary.json` (missing, unreadable, or not a JSON object)."""
    path = out / "summary.json"
    if not path.is_file():
        print(f"cannot re-render: no {path}")
        return 2
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"cannot re-render: {path} is unreadable ({type(e).__name__}: {e})")
        return 2
    if not isinstance(s, dict):
        print(f"cannot re-render: {path} is not a JSON object")
        return 2
    for key in ("errors", "gates"):
        if not isinstance(s.setdefault(key, {}), dict):
            print(f"cannot re-render: {path} has a non-object {key!r}")
            return 2
    errors = s["errors"]

    def profiles_of(run: str, names: Sequence[str]) -> dict[str, dict]:
        found = {}
        for m in names:
            p, why = run_profile(out, run, m)
            if why:
                errors[f"profile:{run}/{m}"] = why
            if p is None:
                continue
            try:
                found[m] = load_profile(p)
            except CalloError as e:
                errors[f"profile:{run}/{m}"] = str(e)
        return found

    run1 = profiles_of("run1", list(MODELS))
    every = {**run1, **profiles_of("run2", list(RUN2_MODELS)), **profiles_of("run3", list(RUN3_MODELS))}
    data = out / "data" / "test-s1.jsonl"
    s["descriptive"] = descriptive(every, data_path=data if data.is_file() else None)
    gates = s.setdefault("gates", {})
    if "t10" in gates:
        gates["t10"] = t10_now(s, run1, previous=gates["t10"])
    st = code_state()                                # in memory only: the doc says which code computed the new numbers
    s["rerendered"] = {"commit": st.get("commit"), "dirty": st.get("dirty")}
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render(s))
    print(f"wrote {doc}; {out} read only")
    return 0


def main(argv: list[str] | None = None) -> int:
    """0 when every requested part ran (the gates' verdicts are in the doc, a FAIL there is the feature's), 1 when a
    part errored, 2 when the bench refused to start, 3 when it was interrupted. Rerunnable: `summary.json` keeps the
    parts that finished, `--parts` picks which to (re)run. Stop the live run ONLY with `kill -TERM $(cat DIR/run.pid)`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--data", default=DATA_SRC, help="the transfer test file; the System One sources are selected from it")
    ap.add_argument("--parts", default=",".join(PARTS), help=f"which parts to run, in this order: {','.join(PARTS)}")
    ap.add_argument("--doc", default=str(DOC))
    ap.add_argument("--min-free-gb", type=float, default=30.0, help="free RAM (vm_stat) needed to start")
    ap.add_argument("--allow-gpu-busy", action="store_true",
                    help="start although another model server is running: recorded in summary.json and passed to every "
                         "menu run (whose profiles are then marked gpu_busy); a deviation from the one-server rule")
    ap.add_argument("--rerender", action="store_true",
                    help="re-render --doc from --out's summary.json and profiles, read-only on --out: no part, no "
                         "server check, no summary.json write")
    a = ap.parse_args(argv)
    unset = lib.unset_paths(("--out", a.out, lib.OUT_ENV))
    if unset:
        print(unset)
        return 2
    if a.rerender:                                   # before every check: it measures nothing and writes only the doc
        return rerender(Path(a.out), Path(a.doc))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parts = [p for p in a.parts.split(",") if p]
    if any(p not in PARTS for p in parts):
        print(f"unknown part(s) in {parts}; choose from {','.join(PARTS)}")
        return 2
    models = [*MODELS.values(), *RUN2_MODELS.values(), *RUN3_MODELS.values()] if set(parts) & set(GPU_PARTS) else []
    unset = lib.unset_paths(("--data", a.data, lib.DATA_ENV), *(("", m["model"], lib.MODELS_ENV) for m in models),
                            ("", LLAMA_SERVER, lib.SERVER_ENV))
    if unset:
        print(unset)
        return 2
    try:
        busy = lib.running_gpu_servers()
    except RuntimeError as e:                        # a ps that fails is "cannot tell", never "no servers"
        if not a.allow_gpu_busy:
            print(f"refusing to start: {e}: cannot tell whether a model server is running (pass --allow-gpu-busy to "
                  f"start anyway)")
            return 2
        busy = [(-1, str(e))]
    if busy and not a.allow_gpu_busy:
        print(f"refusing to start: a model server is already running (one model server at a time; wait for an agreed "
              f"window): {busy}")
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
    problems = disk_problems([*DISK_PATHS, str(out)], MIN_FREE_GIB)
    if problems:
        print("refusing to start: " + "; ".join(problems))
        return 2
    pid_file = out / "run.pid"                       # the live run is stopped ONLY by `kill -TERM $(cat run.pid)`
    owner = pid_owner(pid_file)
    if owner is not None:
        if is_bench_run(owner):
            print(f"refusing to start: another bench run (pid {owner}) owns this --out ({pid_file}); stop it through "
                  f"its run.pid or wait for it to finish")
        else:                                        # never suggest signalling a pid that may be anything
            print(f"refusing to start: {pid_file} names pid {owner}, which is not a bench run: check it and remove "
                  f"run.pid by hand")
        return 2
    previous = _install_stop_handlers()
    me = os.getpid()
    try:
        pid_file.write_text(f"{me}\n")              # a dead pid's stale file is replaced
        return _run(a, out, parts, free, busy)
    finally:
        if _pid_in(pid_file) == me:                  # never remove a file another run wrote over ours
            pid_file.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _pid_in(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


BENCH_MODULE = "acceptance_v03_think"


_pid_args = lib.pid_args                             # the seam the tests stub: `ps -o args= -p PID`


def is_bench_run(pid: int) -> bool:
    """The live pid is another run of this bench (`lib.is_bench_run`, shared with the transcription bench)."""
    return lib.is_bench_run(pid, BENCH_MODULE, pid_args=_pid_args)


def pid_owner(pid_file: Path) -> int | None:
    """The live process `run.pid` names, or None (no file, unreadable, or a dead pid: a stale file to replace).
    A liveness check only (signal 0 is never delivered); nothing is signalled."""
    pid = _pid_in(pid_file)
    if pid is None or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except PermissionError:
        return pid                                   # alive, under another user
    except (ProcessLookupError, OverflowError):
        return None
    except OSError:
        return None
    return pid


# What each part produces in summary.json: a part that runs again first drops these, so a part that fails now leaves
# its gates missing (NOT RUN / ERROR) instead of an earlier session's verdict.
PRODUCES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {   # part: (top-level keys, gates keys)
    "run1": ((), ()), "run2": ((), ()), "run3": ((), ()), "t7": ((), ("t7",)), "t8": (("joint_md",), ("t8",)),
    "serve": (("serve",), ("t4", "t9", "t10")), "t5": ((), ())}


def _forget(s: dict, part: str, parts: Sequence[str]) -> None:
    if part == "t5" and "serve" in parts:            # serve forgot T5 already and may have written why it was not
        return                                       # judged this session: that reason must survive
    keys, gates = PRODUCES[part]
    if part in s["runs"]:
        del s["runs"][part]
    for k in keys:
        s.pop(k, None)
    for g in (*gates, *(("t5",) if part == "serve" and "t5" in parts else ())):
        s["gates"].pop(g, None)
    for k in [k for k in s["errors"] if k == part or k.startswith(f"{part}:") or (part == "serve" and "t5" in parts and k == "t5")]:
        del s["errors"][k]


def _run(a: argparse.Namespace, out: Path, parts: list[str], free: int | None, busy: list) -> int:
    path = out / "summary.json"
    s: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    for k in ("aborted", "finished"):
        s.pop(k, None)
    s.update(started=s.get("started") or _now(), llama_server=LLAMA_SERVER, out=str(out), free_memory_at_start=free)
    for k in ("runs", "gates", "errors"):
        s.setdefault(k, {})
    data = out / "data" / "test-s1.jsonl"
    data2 = out / "data" / "test-s1-subset.jsonl"
    n = select_records(a.data, data)
    n2 = select_records(a.data, data2, limit=RUN2_N)
    s["data"] = {"path": str(data), "source": str(a.data), "questions": n, "sources": list(S1_SOURCES), "run2_questions": n2,
                 "sha256": _sha256(data), "run2_sha256": _sha256(data2), "source_sha256": _sha256(a.data)}
    if n == 0:
        print(f"refusing to start: no clean records of {S1_SOURCES} in {a.data}")
        return 2
    version = _llama_server_version(LLAMA_SERVER)
    disk = disk_free_gib([str(out), "/"])
    print("disk at start: " + "; ".join(f"{p}: {'unknown' if g is None else f'{g:.1f} GiB'} free" for p, g in disk.items()),
          file=sys.stderr)
    warnings = startup_warnings(parts, version, disk)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    session = {"started": _now(), "parts": parts, "code": code_state(), "llama_server": LLAMA_SERVER,
               "llama_server_version": version, "allow_gpu_busy": a.allow_gpu_busy, "gpu_busy_at_start": busy,
               "disk_free_gib": disk, "warnings": warnings}
    s.update(code=session["code"], llama_server_version=version, disk_free_gib_at_start=disk, warnings=warnings)
    s.setdefault("sessions", []).append(session)
    s.setdefault("provenance", {})

    def stamp(part: str) -> None:
        """The code and the binary that produced `part`, kept per part: a later session that runs other parts
        overwrites the top-level `code`, never this."""
        code = session["code"]
        s["provenance"][part] = {"commit": code.get("commit"), "dirty": code.get("dirty"), "llama_server": LLAMA_SERVER,
                                 "llama_server_version": version, "at": _now(), "session": len(s["sessions"])}

    def save() -> None:
        path.write_text(json.dumps(s, indent=1, default=str))

    def profiles_of(run: str, names: Sequence[str]) -> dict[str, dict]:
        """The run's profiles that its run.json marks done (`run_profile`); one it does not vouch for, or that cannot
        be read, is the `profile:` error and missing for the gates."""
        found = {}
        for m in names:
            p, why = run_profile(out, run, m)
            if why:
                s["errors"][f"profile:{run}/{m}"] = why
            if p is None:
                continue
            try:
                found[m] = load_profile(p)
            except CalloError as e:
                s["errors"][f"profile:{run}/{m}"] = str(e)
        return found

    errored = False

    def fail(key: str, e: BaseException) -> None:
        nonlocal errored
        log.error("%s failed: %s: %s", key, type(e).__name__, e, exc_info=e)
        s["errors"][key] = f"{type(e).__name__}: {e}"
        errored = True

    try:
        for part in parts:
            log.info("part %s", part)
            problems = disk_problems([*DISK_PATHS, str(out)], MIN_FREE_GIB)   # between parts too: a multi-hour run
            try:                                                               # must not start a part on a full disk
                others = [] if a.allow_gpu_busy else _foreign_servers()       # or beside another session's server
                cannot_tell = None
            except RuntimeError as e:                                          # fails closed
                others, cannot_tell = [], str(e)
            if problems or others or cannot_tell:
                s["errors"][part] = "not started: " + "; ".join(
                    [*(f"disk: {p}" for p in problems), *([f"a model server is running: {others}"] if others else []),
                     *([f"cannot tell whether a model server is running: {cannot_tell}"] if cannot_tell else [])])
                errored = True
                save()
                break
            _forget(s, part, parts)                    # it starts: an earlier session's verdict for it is void now
            stamp(part)
            try:
                if part in ("run1", "run2", "run3"):
                    models, d = {"run1": (MODELS, data), "run2": (RUN2_MODELS, data2), "run3": (RUN3_MODELS, data)}[part]
                    r, rj, _ = _run_calibrate(out / part, d, models, allow_gpu_busy=a.allow_gpu_busy)
                    s["runs"][part] = {"rc": r, "dir": str(out / part), "run_json": rj}
                elif part == "t7":                     # isolation, not measurement: a subset, two budgets, no audit
                    data_t7 = out / "data" / "test-s1-t7.jsonl"
                    data_t7.write_text("".join(json.dumps(r) + "\n" for r in _first_records(data, T7_N)))
                    t7_dir = _fresh_dir(out / "t7")    # the broken run must be a new run, never a resume
                    broken = {**MODELS, "qwen9b": {**MODELS["qwen9b"], "model": MODELS["qwen9b"]["model"] + ".missing"}}
                    r1, rj1, _ = _run_calibrate(t7_dir, data_t7, broken, budgets=T7_BUDGETS, audit=0,
                                                allow_gpu_busy=a.allow_gpu_busy)
                    r2, rj2, _ = _run_calibrate(t7_dir, data_t7, MODELS, "--resume", budgets=T7_BUDGETS, audit=0,
                                                allow_gpu_busy=a.allow_gpu_busy)
                    s["gates"]["t7"] = {**t7({**(rj1 or {"models": {}}), "rc": r1}, {**(rj2 or {"models": {}}), "rc": r2}),
                                        "dir": str(t7_dir)}
                elif part == "t8":
                    _part_t8(s, out)
                elif part == "serve":
                    errored |= _part_serve(s, out, data, parts, fail)
                elif part == "t5" and "serve" not in parts:
                    s["errors"]["t5"] = "T5 runs inside the tiel serve-path session: add `serve` to --parts"
                    errored = True
            except Exception as e:  # noqa: BLE001 -- one part's failure must not lose the others
                fail(part, e)
            save()
        try:
            stamp("gates")                             # T1-T3, T6 and the descriptive results: recomputed here
            _offline_gates(s, out, data, data2, profiles_of)
            errored |= any(k.startswith("profile:") for k in s["errors"])
        except Exception as e:  # noqa: BLE001 -- the doc still renders what the parts measured
            fail("gates", e)
        s["finished"] = _now()
    except (KeyboardInterrupt, SystemExit) as e:
        s["aborted"] = f"interrupted ({type(e).__name__})"
        try:                                           # they only read files, so an earlier session's T1-T3 and
            stamp("gates")                             # T6 never render under ABORTED
            _offline_gates(s, out, data, data2, profiles_of)
        except BaseException as e2:  # noqa: BLE001 -- a second Ctrl-C here, or a fault: the doc still renders
            s["errors"]["gates"] = f"{type(e2).__name__}: {e2}"
    save()
    doc = Path(a.doc)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render(s))
    log.info("wrote %s and %s", path, doc)
    return 3 if s.get("aborted") else 1 if errored else 0


def _part_t8(s: dict, out: Path) -> None:
    """T8: `callo compare` joins Run 1's three profiles (and Run 3's: the joint frontier) and refuses Run 2's beside
    Run 1's (another data sha). Only profiles their run.json marks done count (`run_profile`); without Run 1's and Run
    2's T8 cannot be judged: its `error`, never a FAIL."""
    found = {(run, m): run_profile(out, run, m) for run, names in (("run1", MODELS), ("run2", RUN2_MODELS),
                                                                   ("run3", RUN3_MODELS)) for m in names}
    run1 = [p for (run, _), (p, _) in found.items() if run == "run1" and p is not None]
    run2 = [p for (run, _), (p, _) in found.items() if run == "run2" and p is not None]
    run3 = [p for (run, _), (p, _) in found.items() if run == "run3" and p is not None]     # the joint frontier's extra
    missing = [f"{run}/{m}.think.json" + (f" ({why})" if why else "") for (run, m), (p, why) in found.items()
               if run != "run3" and p is None]
    if missing:
        s["gates"]["t8"] = {"error": f"T8 needs Run 1's and Run 2's profiles, done in their run.json; missing: "
                                     f"{', '.join(missing)}", "pass": None}
        return
    r_join, _ = _calibrate(["compare", *map(str, run1), *map(str, run3), "--out", str(out / "joint")])
    r_refuse, refuse_err = _calibrate(["compare", *map(str, run1), *map(str, run2)])
    s["gates"]["t8"] = t8(r_join, r_refuse, refuse_err)
    if (out / "joint" / "report.md").exists():
        s["joint_md"] = (out / "joint" / "report.md").read_text()


def _part_serve(s: dict, out: Path, data: Path, parts: Sequence[str], fail: Any) -> bool:
    """The serve-path sessions, each isolated (one's failure is its `serve:<name>` error, the others run; each one's
    record is `summary["serve"][name]`: its budget and why, its T4 statuses, its T10 entry): one per
    thinking model (T4, T9's served half, T10; T5 on tiel when `t5` is in --parts), the budget-0-only model (T4's
    422) and the small launch (T4's 413, 403 and image 422, the same-URL restart). Then T4, T9 and T10 from whatever
    ran: a session that did not run is missing there, so its gates say ERROR. Each record keeps what the gates are
    computed from (`served_ms`, `t10_stopped`, `served_readouts`, `stats_think`), so `_offline_gates` recomputes T9
    and T10 later from the run.json files and profiles as they are then. True when a session errored."""
    errored = False
    cases: dict[str, int | None] = {}
    case_errors: dict[str, str] = {}
    t10s: dict[str, dict] = {}
    disconnect: list[bool | None] = []
    self_tests: list[bool | None] = []
    timings: list[bool | None] = []
    sessions: list[tuple[str, Any]] = [
        *((name, lambda name=name, prof=out / "run1" / f"{name}.think.json":
           serve_session(name, out / "serve", prof, data, guard=name == "tiel" and "t5" in parts)) for name in THINKING),
        (BUDGET0_MODEL, lambda: gemma_session(out / "serve", out / "run1" / f"{BUDGET0_MODEL}.think.json", data)),
        ("qwen9b-small", lambda: small_session(out / "serve", out / "run1" / "qwen9b.think.json", data))]
    needs = {"qwen9b-small": "qwen9b"}
    for key, run in sessions:
        prof, why = run_profile(out, "run1", needs.get(key, key))
        if prof is None:                                 # only a profile Run 1's run.json marks done
            s["errors"][f"serve:{key}"] = why or "no Run 1 profile"
            errored = True
            continue
        try:
            r = run()
        except Exception as e:  # noqa: BLE001 -- one session's failure must not lose the others
            fail(f"serve:{key}", e)
            errored = True
            continue
        s.setdefault("serve", {})[key] = dict(r)
        _merge_cases(cases, r["cases"])
        for k, why in (r.get("case_errors") or {}).items():
            case_errors.setdefault(k, why)
        restart = r.get("restart")
        if isinstance(restart, dict) and not restart.get("pass"):
            s["errors"]["serve"] = f"same-URL restart of {key} failed: {restart.get('error') or 'not verified'}"
            errored = True
        if key in THINKING:
            t10s[key] = r["t10"]
            disconnect.append(r["disconnect_clear"])
            self_tests.append(r["self_test"])
            timings.append(r["timings_ok"])
            if "t5" in r:
                s["gates"]["t5"] = r["t5"]
    if "t5" in parts and "t5" not in s["gates"]:
        s["errors"]["t5"] = "T5 was not judged: the tiel serve-path session did not run"
        errored = True
    # every thinking session that ran must tell: one None makes that check not testable (T4 ERROR), never a PASS
    # on the others
    s["gates"]["t4"] = t4(cases, disconnect_clear=_all3(disconnect), self_test=_all3(self_tests), timings_ok=_all3(timings),
                          case_errors=case_errors)
    s["gates"]["t9"] = t9_now(s, out)
    s["gates"]["t10"] = t10_gate(t10s)
    return errored


def t9_now(s: dict, out: Path, *, previous: dict | None = None) -> dict:
    """T9 from the files as they are now: the calibrate half from every raw file whose run's run.json marks its model
    done at this moment, so a model that finished after the serve part (Run 2 after serve, say) is judged, not
    left a stale ERROR; the served half from the serve records (`summary["serve"][name]`: `served_readouts`,
    `stats_think`). A record from before summary.json kept its readouts carries its served-half verdict from
    `previous` (the T9 entry the serve part wrote, CARRIED note); none of either is that half's ERROR."""
    raws = {m: out / run / f"{m}.raw.jsonl" for run, names in (("run1", THINKING), ("run2", RUN2_MODELS)) for m in names
            if (out / run / f"{m}.raw.jsonl").exists() and run_profile(out, run, m)[0] is not None}   # done models only
    served: dict[str, dict] = {}
    judged: dict[str, dict] = {}
    carried = (previous or {}).get("served") or {}
    for name in THINKING:
        rec = (s.get("serve") or {}).get(name)
        if not isinstance(rec, dict):
            continue
        if "served_readouts" in rec:
            served[name] = {"think": rec.get("stats_think") or {}, "readouts": rec["served_readouts"]}
        elif isinstance(carried.get(name), dict):
            judged[name] = carried[name]
    return t9(raws, served, judged=judged)


def _offline_gates(s: dict, out: Path, data: Path, data2: Path, profiles_of: Any) -> None:
    """T1, T2, T3, T6 and the descriptive results from the profiles each run's run.json marks done (a missing one is
    the gates' ERROR). Their earlier verdicts are dropped first, so one that cannot be recomputed now is missing
    (NOT RUN / ERROR), never an earlier session's PASS. T1 reads each thinking model's class from its GGUF header
    (`model_traits`: the post-hoc MoE-hybrid amendment); a header it cannot read keeps the original criterion, with a
    note. T9 and T10, once a serve part has judged them, are recomputed from the run.json files as they are now
    (`t9_now`, `t10_now`)."""
    for k in [k for k in s["errors"] if k.startswith("profile:")]:
        del s["errors"][k]
    for g in ("t1", "t2", "t3", "t6"):
        s["gates"].pop(g, None)
    s.pop("descriptive", None)
    s.pop("descriptive_pairs", None)
    run1 = profiles_of("run1", list(MODELS))
    run2 = profiles_of("run2", list(RUN2_MODELS))
    run3 = profiles_of("run3", list(RUN3_MODELS))
    results = {}
    for m, p in run1.items():
        if m not in THINKING:
            continue
        traits, why = model_traits(p)
        results[m] = t1_t2(p, out / "run1" / f"{m}.raw.jsonl", data, traits=traits)
        if why:
            results[m]["notes"] = [*results[m].get("notes", []), f"model class unknown ({why}): the original T1 criterion "
                                                                   f"applies"]
    results.update({m: t1_t2(p, out / "run2" / f"{m}.raw.jsonl", data2, dense=True) for m, p in run2.items()})
    s["gates"]["t1"], s["gates"]["t2"] = t1_t2_gates(results, dense=set(RUN2_MODELS))
    every = {**run1, **run2, **run3}
    s["gates"]["t3"] = t3_gate(every)
    gemma = run1.get(BUDGET0_MODEL)
    run1_rec = s["runs"].get("run1") or {}
    run1_rc = run1_rec.get("rc")
    if gemma is None:
        s["gates"]["t6"] = {"error": f"no Run 1 profile for {BUDGET0_MODEL}" + (
            f" ({s['errors'][f'profile:run1/{BUDGET0_MODEL}']})" if f"profile:run1/{BUDGET0_MODEL}" in s["errors"] else ""),
            "pass": None}
    elif run1_rc is None:                        # a Run 1 rerun that failed: its exit is unknown, never a FAIL
        s["gates"]["t6"] = {"error": "Run 1 did not finish in this session (no exit code recorded): T6 cannot be "
                                     "judged", "pass": None}
    else:                                        # the joint report's note; Run 1's own report until T8 wrote it
        report = next((p for p in (out / "joint" / "report.json", out / "run1" / "report.json") if p.exists()), None)
        status = (((run1_rec.get("run_json") or {}).get("models") or {}).get(BUDGET0_MODEL) or {}).get("status")
        s["gates"]["t6"] = t6(gemma, json.loads(report.read_text()) if report else None, status=status, rc=run1_rc)
    s["descriptive"] = descriptive(every, data_path=data)
    s["descriptive_pairs"] = descriptive_pairs(every, [("tiel", "tiel-fixed")])
    if "t9" in s["gates"]:                       # a serve part judged it once: the calibrate half is recomputed now
        s["gates"]["t9"] = t9_now(s, out, previous=s["gates"]["t9"])
    if "t10" in s["gates"]:
        s["gates"]["t10"] = t10_now(s, run1, previous=s["gates"]["t10"])


def t10_now(s: dict, run1: Mapping[str, dict], *, previous: Mapping[str, Any] | None = None) -> dict:
    """T10 from the kept serve records (`served_ms`, `budget`, `t10_stopped`, `budget_note`) against each model's Run 1
    profile as it is now (one its run.json no longer marks done is that model's ERROR). A record from before
    summary.json kept `served_ms` keeps the serve part's own entry from `previous`."""
    entries: dict[str, dict] = {}
    for name in SERVED_MODELS:
        rec = (s.get("serve") or {}).get(name)
        if not isinstance(rec, dict):
            continue
        if not isinstance(rec.get("served_ms"), dict) or type(rec.get("budget")) is not int:
            old = (previous or {}).get(name)
            if isinstance(old, dict):
                entries[name] = old
            continue
        note = {"budget_note": rec["budget_note"]} if rec.get("budget_note") else {}
        if name not in run1:
            entries[name] = {"n": len(rec["served_ms"]), "budget": rec["budget"], "pass": None, **note,
                             "error": "no Run 1 profile its run.json marks done: T10 cannot be recomputed"}
            continue
        entries[name] = {**t10(rec["served_ms"], run1[name], rec["budget"], stopped=rec.get("t10_stopped")), **note}
    return t10_gate(entries)


if __name__ == "__main__":
    raise SystemExit(main())
