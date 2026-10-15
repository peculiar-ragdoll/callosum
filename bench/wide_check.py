"""The wide-choice GPU validation: the 40 items of `bench/data/wide_items.jsonl` (20 BANKING77,
77 options; 20 CLINC150, 151 options) and their 10-option controls, read on one model through `callo serve` and
in-process, against the gates below. Written test-first on fakes; the live run needs the GPU free.

    nohup .venv/bin/python -m bench.wide_check --model tiel --out OUT > OUT/tiel.log 2>&1 &
    nohup .venv/bin/python -m bench.wide_check --model gemma --out OUT > OUT/gemma.log 2>&1 &
    .venv/bin/python -m bench.wide_check --recompute --model tiel --out OUT          # offline, no server
    nohup .venv/bin/python -m bench.wide_check --only-resend --model tiel --out OUT > OUT/tiel-resend.log 2>&1 &

Per model (`--model tiel`: all 40 items, accuracy gated; `--model gemma`: Gemma-4-E4B, the first 10 BANKING77 items,
accuracy reported only: the accuracy gate applies to the primary model; this run covers
sliding-window checkpoints):
1. `callo serve` in launch mode with `--cache`, one pinned slot (`--s1-slots 1`) and `-- -v` (so the llama-server log
   carries `created context checkpoint` / `restored context checkpoint`); refuse unless `/callo/stats` reports a
   `wide_coverage` that covers every item.
2. SERVE: each item, wide then control, at budget 0 through `/v1/systemone`; from X-Callo-Timings `passes`, `split`,
   `unread`, `ms`, `prompt_n`; from the llama-server log segment of the request (offsets around it), the readout
   slot's checkpoint lines and its prompt-eval lines (a split pass's own `prompt_n` is the last `split` of them).
3. The n_probs reply size: one raw `/completion` at n_probs 1000 and one at 100.
4. IN-PROCESS: each wide item again through `local_engine` on the same llama-server (cache off): `option_mass`,
   `unread_mass`, `split_passes`, `partial_by_letter`, and every scoring pass's token prefix. The 5 items with the
   largest partial mass are read again with the forced-split diagnostic (`split_read_above=0`,
   `split_max_passes=None`, `split_min_partial=1e-3`); total variation against the capped read, next to the noise
   baseline (the same 5 items read capped once more). Forced reads that expand nothing make that gate N/A.
5. RE-SEND: for 3 items with two first-letter tokens t1, t2 in the slot's dump (serve splits first, then the largest
   split mass), the wide request again through callo, then by hand to the readout slot (`cache_prompt`) the slot
   prefix s + t1 (split pass 1: one token past the cache) and s + t2 (split pass 2: it diverges at len(s), the
   checkpoint restore), reading each `timings.prompt_n`. The split-prompt_n gate takes the max over both and the
   serve readouts' own split passes in the `-v` log, and needs s + t2 measured.
6. The gates (`gates`), the report (`report`) and a 10-line verdict (`verdict_lines`), into OUT/summary.json under
   `models.<model>` and OUT/verdict-<model>.txt; the rows into OUT/<model>/rows.jsonl as they are measured;
   `gpu_after`, the model servers other than ours at the end.

`--recompute` (offline, no server) evaluates OUT/<model>/rows.jsonl again with the current gates (the serve rows carry
their parsed log segments) and rewrites summary.json's entry and the verdict file. `--only-resend` (GPU, a few
minutes) launches the model, runs only step 5 (in-process reads for the slots, then the re-sends) into
OUT/<model>/rows-resend.jsonl, records `resend_supplement` and recomputes: its re-sends replace the full run's.

A gate is PASS, FAIL, ERROR (missing evidence: a gate never passes on rows it does not have), N/A (nothing to judge,
with the reason) or REPORTED (accuracy on Gemma). The model's verdict is FAIL if any gate FAILs, else ERROR if any
gate is ERROR, else PASS; N/A and REPORTED do not fail it, so the verdict's first line counts the gates by status.
The prompt_n line counts its values by source (s+t1, s+t2, the -v log). Exit 0 on PASS, 1 otherwise, 2 on a refusal.

Hygiene, as the other v0.3 benches. Run it from a frozen worktree of the commit under test (`git worktree add
--detach ../callosum-wide-run <sha>`, `uv sync --frozen` there) with that worktree's interpreter and no `uv` wrapper:
- refuses to start (exit 2) while a model server holds the GPU (`lib.running_gpu_servers`, fails closed), on a live
  `OUT/run.pid`, on uncommitted changes under src/ or bench/ (`--allow-dirty` records them instead), on a volume with
  under 20 GiB free, or on an items file that does not validate;
- its pid is in `OUT/run.pid` for the whole run, removed at exit only while it still names this process; stop it ONLY
  with `kill -TERM $(cat OUT/run.pid)` (the stop handlers raise SystemExit, so `finally` stops callo and reaps the
  llama-server it launched; never by process name);
- every server binds an ephemeral port; a port in `FORBIDDEN_PORTS` is an ERROR before anything is sent;
- summary.json records the commit and dirty state, the items file's sha256, the launch flags and the llama-server
  build.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import signal
import statistics
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench import lib
from bench.acceptance import MASTER, Callo
from bench.acceptance_v02 import TIEL, _install_stop_handlers
from bench.acceptance_v03_bias import DISK_PATHS, MIN_FREE_GIB, code_state, disk_problems
from bench.acceptance_v03_think import GEMMA, _pid_in, pid_owner
from bench.check_cache_delimiters import (
    llama_log_path,
    parse_prompt_evals,
    parse_timings,
    port_problems,
)
from callo.engine import WIDE_TOP_K
from callo.errors import CalloError
from callo.marginalize import WIDE_MAX_SPLIT_PASSES, WIDE_SPLIT_READ, _mass_tol
from callo.systemone import parse_request

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "bench" / "data" / "wide_items.jsonl"
OUT_DEFAULT = lib.env_path(lib.OUT_ENV, "callo_v0.3", "wide")
BENCH_MODULE = "wide_check"
MODELS = {"tiel": {"model": TIEL, "sources": ("banking77", "clinc150"), "per_source": 20, "accuracy": "gated"},
          "gemma": {"model": GEMMA, "sources": ("banking77",), "per_source": 10, "accuracy": "reported"}}
CTX, PARALLEL = 32768, 2
SERVE_FLAGS = ("--s1-slots", "1", "--", "-v")
INSTRUCTIONS = {"banking77": "Which intent does this customer message express?",
                "clinc150": "Which intent does this request express?"}
QID = "intent"
# the gates
MASS_MEDIAN_MIN = 0.9
UNREAD_MAX, UNREAD_SHARE_MIN = WIDE_SPLIT_READ, 0.95
SPLIT0_SHARE_MIN, SPLIT_MAX = 0.90, WIDE_MAX_SPLIT_PASSES
SPLIT_PROMPT_N_MAX = 16
TV_MAX = 0.01
ACC_WIDE_MIN, ACC_CONTROL_SLACK = 0.5, 0.25
SPLIT_MASS_TRIGGER = 0.05            # a median split mass above this calls for a wide system variant
FORCED_N, RESEND_N = 5, 3
FORCED = {"split_read_above": 0.0, "split_max_passes": None, "split_min_partial": 1e-3}
TRACKED = ("src", "bench")

_CHECKPOINT = re.compile(r"slot\s+\S+: id\s+(\d+) \| task (-?\d+) \| (created|restored) context checkpoint"
                         r"[^\n]*?n_tokens = (\d+)")

Post = Callable[..., tuple[int, dict, Any]]
_POST: Post = lib.post                          # the seam the tests swap: every HTTP request of the run


class RunError(RuntimeError):
    """A launch that would measure the wrong thing: the run stops as an ERROR, its rows so far kept."""


class LogTail:
    """The llama-server log read from a byte offset (a `-v` log grows by every request and reply)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def since(self, offset: int) -> str:
        try:
            with self.path.open("rb") as f:
                f.seek(offset)
                return f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""


# ---- items and bodies (pure) ------------------------------------------------------------------------------------

def load_items(path: str | Path = ITEMS, *, model: str = "tiel") -> tuple[dict, list[dict]]:
    """(the `_meta` row, the model's items): its sources, the first `per_source` of each, in file order. ValueError on
    a file that does not have the shape `bench/wide_items.py` writes."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or set(rows[0]) != {"_meta"}:
        raise ValueError(f"{path}: the first row must be the _meta row")
    spec = MODELS[model]
    items = []
    for src in spec["sources"]:
        mine = [r for r in rows[1:] if r.get("source") == src][: spec["per_source"]]
        if len(mine) < spec["per_source"]:
            raise ValueError(f"{path}: {len(mine)} {src} items, {model} needs {spec['per_source']}")
        items += mine
    for it in items:
        labels, control = it.get("labels"), it.get("control")
        if (not isinstance(labels, list) or len(labels) <= 26 or len(set(labels)) != len(labels)
                or it.get("gold") not in labels or not isinstance(control, list) or len(set(control)) != 10
                or it["gold"] not in control or not set(control) <= set(labels) or not str(it.get("text", "")).strip()):
            raise ValueError(f"{path}: item {it.get('id')!r} is malformed")
    return rows[0], items


def _body(item: dict, options: Sequence[str]) -> dict:
    return {"state": item["text"], "think_budget": 0,
            "questions": {QID: {"type": "choice", "instructions": INSTRUCTIONS[item["source"]],
                                "criteria": {o: None for o in options}}}}


def wide_body(item: dict) -> dict:
    """The item against its dataset's every intent (77 or 151 options, two-letter labels), at budget 0."""
    return _body(item, item["labels"])


def control_body(item: dict) -> dict:
    """The 10-option control (letters A-J): the gold and its 9 seeded distractors."""
    return _body(item, item["control"])


def parse_checkpoints(log_text: str, slot: int | None) -> dict[str, list[int]]:
    """The `created` / `restored context checkpoint` lines of `slot` (never another slot's; every slot's when None),
    as the n_tokens of each, in order."""
    out: dict[str, list[int]] = {"created": [], "restored": []}
    for m in _CHECKPOINT.finditer(log_text):
        if slot is None or int(m.group(1)) == slot:
            out[m.group(3)].append(int(m.group(4)))
    return out


def tv(a: Sequence[float], b: Sequence[float]) -> float:
    """Total variation distance between two probability vectors over the same options."""
    if len(a) != len(b):
        raise ValueError("total variation needs vectors over the same options")
    return 0.5 * math.fsum(abs(x - y) for x, y in zip(a, b, strict=True))


def mass_tolerance(passes: int, top_k: int = WIDE_TOP_K) -> float:
    """`_mass_tol` over every term the readout can have summed: at most `top_k` finite entries per scoring pass (the
    engine checks the exact count inside; this re-check is the bound)."""
    return _mass_tol(top_k * max(passes, 1))


# ---- the serve run (seams: post, log) ---------------------------------------------------------------------------

def _argmax(probs: Mapping[str, float]) -> str | None:
    return max(probs, key=lambda k: probs[k]) if probs else None


def serve_readout(url: str, item: dict, arm: str, *, post: Post, log: Any, timeout: float = 900,
                  log_wait: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> dict:
    """One `/v1/systemone` readout of `item` (`arm` "wide" or "control"): X-Callo-Timings, the answer against the
    gold, and the readout slot's checkpoint and prompt-eval lines from the llama-server log it added (re-read for up to
    `log_wait` seconds until there is one prompt-eval line per pass). A non-200 reply is a row with `error`."""
    body = wide_body(item) if arm == "wide" else control_body(item)
    row: dict[str, Any] = {"id": item["id"], "source": item["source"], "arm": arm,
                           "n_options": len(body["questions"][QID]["criteria"]), "gold": item["gold"]}
    before = log.size()
    t0 = time.perf_counter()
    status, headers, out = post(f"{url}/v1/systemone", body, timeout=timeout)
    row["wall_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    answer = (out.get("answers") or {}).get(QID) if isinstance(out, dict) else None
    if status != 200 or not isinstance(answer, dict) or not isinstance(answer.get("probabilities"), dict):
        row.update(status=status, error=f"/v1/systemone: HTTP {status}: {str(out)[:300]}")
        return row
    slot_h = headers.get("x-callo-slot")
    slot = int(slot_h) if slot_h is not None and str(slot_h).isdigit() else None
    entry = next((e for e in parse_timings(headers.get("x-callo-timings")) if e.get("q") == QID), {})
    passes = entry.get("passes") if isinstance(entry.get("passes"), int) else None
    deadline = time.monotonic() + log_wait
    while True:
        seg = log.since(before)
        evals = parse_prompt_evals(seg, slot) if slot is not None else []
        if passes is None or len(evals) >= passes or time.monotonic() >= deadline:
            break
        sleep(0.05)
    cps = parse_checkpoints(seg, slot)
    split = entry.get("split")
    probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
    top = _argmax(probs)
    row.update(status=status, slot=slot, passes=passes, split=split, unread=entry.get("unread"), ms=entry.get("ms"),
               prompt_n=entry.get("prompt_n"), cache_n=entry.get("cache_n"), tokens=entry.get("tokens"),
               stop=entry.get("stop"), miss=entry.get("miss"), argmax=top, p_gold=probs.get(item["gold"]),
               correct=top == item["gold"], created=len(cps["created"]), restored=len(cps["restored"]),
               checkpoints=cps, prompt_evals=[e["prompt_n"] for e in evals],
               split_prompt_n_log=([e["prompt_n"] for e in evals[-split:]]
                                   if isinstance(split, int) and split > 0 and passes is not None
                                   and len(evals) == passes else None))
    if slot is None:
        row["error"] = f"no integer X-Callo-Slot ({slot_h!r}): the readout's log lines are unknown"
    elif arm == "wide" and (not isinstance(split, int) or not isinstance(entry.get("unread"), (int, float))):
        row["error"] = "a wide readout's X-Callo-Timings entry has no split / unread"
    return row


def reply_sizes(upstream: str, *, post: Post, raw_post: Callable[[str, dict], tuple[int, bytes]]) -> dict:
    """The bytes of one raw `/completion` reply at n_probs 1000 (a wide pass) and at 100 (a narrow one)."""
    status, _, tok = post(f"{upstream}/tokenize", {"content": "Which intent is this?"}, timeout=60)
    tokens = tok.get("tokens") if status == 200 and isinstance(tok, dict) else None
    if not tokens:
        return {"error": f"/tokenize: HTTP {status}"}
    out: dict[str, Any] = {}
    for k in (WIDE_TOP_K, 100):
        s, raw = raw_post(f"{upstream}/completion", {"prompt": tokens, "n_predict": 1, "n_probs": k,
                                                      "cache_prompt": False})
        out[f"n_probs_{k}"] = len(raw) if s == 200 else None
    return out


def _raw_post(url: str, payload: dict, timeout: float = 300) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 -- local servers only
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ---- the in-process run (seam: the engine) ----------------------------------------------------------------------

class Calls(list):
    """The plain scoring passes of the current readout: each token prefix (the list) and its top-k dump (`dumps`)."""

    def __init__(self) -> None:
        super().__init__()
        self.dumps: list[Mapping[int, float]] = []

    def clear(self) -> None:
        super().clear()
        self.dumps.clear()


def record_calls(engine: Any) -> Calls:
    """Every plain scoring pass's token prefix and dump, in order (the engine's scorer is wrapped in place)."""
    calls = Calls()
    inner = engine.scorer.next_logprobs

    def next_logprobs(tokens, *a, **kw):
        out = inner(tokens, *a, **kw)
        calls.append(tuple(tokens))
        calls.dumps.append(out)
        return out

    engine.scorer.next_logprobs = next_logprobs
    return calls


@contextlib.contextmanager
def forced_split(engine: Any) -> Iterator[None]:
    """The forced-split diagnostic's settings (`FORCED`), restored afterwards."""
    saved = {k: getattr(engine, k) for k in FORCED}
    try:
        for k, v in FORCED.items():
            setattr(engine, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(engine, k, v)


def inprocess_read(engine: Any, item: dict, calls: Calls, *, mode: str = "capped"
                   ) -> tuple[dict, list[list[int]], dict]:
    """(the readout of `item`'s wide question as a row, the full token prefix of each of its split passes, the slot:
    its token prefix `s` and `cands`, the question's first-letter tokens in that slot's dump as [id, p], likeliest
    first (ties by id), the order split attribution expands them in)."""
    req = parse_request(wide_body(item))
    q = req.questions[0]
    calls.clear()
    t0 = time.perf_counter()
    r = engine.answer(req.state, q)
    ms = 1000 * (time.perf_counter() - t0)
    base = len(calls[0]) if calls else 0
    splits = [list(c) for c in calls[len(calls) - r.split_passes:]] if r.split_passes else []
    at = len(calls) - 1 - r.split_passes                       # the slot: the walk's last pass, before any split pass
    slot: dict[str, Any] = {"prefix": list(calls[at]) if 0 <= at < len(calls) else [], "cands": []}
    if 0 <= at < len(calls):
        partial = frozenset().union(*engine.sets.partial(tuple(sorted({lab[0] for lab in q.labels}))).values())
        dump = calls.dumps[at]
        slot["cands"] = [[t, math.exp(lp)] for t, lp in sorted(((t, lp) for t, lp in dump.items() if t in partial),
                                                                key=lambda x: (-x[1], x[0]))]
    top = max(range(len(r.probs)), key=r.probs.__getitem__)
    tol = mass_tolerance(r.passes)
    row = {"id": item["id"], "source": item["source"], "mode": mode, "option_mass": r.option_mass,
           "unread_mass": r.unread_mass, "split_passes": r.split_passes, "passes": r.passes, "depth": r.depth,
           "stop": r.stop, "miss": r.topk_miss, "wide": r.wide,
           "partial_by_letter": [[f, m] for f, m in r.partial_by_letter],
           "split_mass": math.fsum(m for _, m in r.partial_by_letter), "probs": list(r.probs),
           "argmax": q.keys[top], "correct": q.keys[top] == item["gold"], "ms": round(ms, 1),
           "prompt_tokens": r.prompt_tokens, "split_tails": [c[base:] for c in splits],
           "slot_len": len(slot["prefix"]), "slot_cands": slot["cands"][:4],
           "mass_plus_unread": r.option_mass + r.unread_mass, "mass_tol": tol,
           "mass_ok": r.option_mass + r.unread_mass <= 1.0 + tol}
    return row, splits, slot


def _completion_prompt_n(upstream: str, prefix: Sequence[int], slot: int, *, post: Post, delimiter: str | None,
                         timeout: float) -> tuple[int | None, int | None, str | None]:
    payload: dict = {"prompt": list(prefix), "n_predict": 1, "n_probs": 1, "temperature": 1.0, "cache_prompt": True,
                     "id_slot": slot}
    if delimiter:
        payload["message_delimiters"] = [{"role": "user", "delimiter": delimiter}]
    s, _, rep = post(f"{upstream}/completion", payload, timeout=timeout)
    timings = rep.get("timings") if s == 200 and isinstance(rep, dict) else None
    if not isinstance(timings, dict) or not isinstance(timings.get("prompt_n"), int):
        return None, None, f"/completion: HTTP {s}: {str(rep)[:200]}"
    return timings["prompt_n"], timings.get("cache_n"), None


def resend(url: str, upstream: str, item: dict, prefix: Sequence[int], t1: int, t2: int | None, *, post: Post,
           delimiter: str | None, timeout: float = 900) -> dict:
    """The wide readout of `item` through callo, then by hand to the readout slot, as callo's cached plain pass sends
    them (`cache_prompt`, the user-turn `message_delimiters`; n_probs 1, which moves no cache): `s + t1` (split pass 1:
    it extends the readout's cache by one token and checkpoints at len(s)), then `s + t2` (split pass 2: it diverges at
    len(s), so a hybrid or SWA model restores that checkpoint). Their `timings.prompt_n`: `prompt_n`, `prompt_n_2`."""
    row: dict[str, Any] = {"id": item["id"], "prefix_len": len(prefix), "t1": t1, "t2": t2}
    status, headers, out = post(f"{url}/v1/systemone", wide_body(item), timeout=timeout)
    slot_h = headers.get("x-callo-slot") if status == 200 else None
    if slot_h is None or not str(slot_h).isdigit():
        row["error"] = f"/v1/systemone: HTTP {status}, X-Callo-Slot {slot_h!r}"
        return row
    entry = next((e for e in parse_timings(headers.get("x-callo-timings")) if e.get("q") == QID), {})
    row.update(slot=int(slot_h), readout_split=entry.get("split"))
    for tok, sfx in ((t1, ""), (t2, "_2")):
        if tok is None:
            continue
        pn, cn, err = _completion_prompt_n(upstream, [*prefix, tok], int(slot_h), post=post, delimiter=delimiter,
                                           timeout=timeout)
        if err is not None:
            row["error"] = err
            return row
        row.update({f"prompt_n{sfx}": pn, f"cache_n{sfx}": cn})
    return row


def resend_order(serve_rows: Sequence[dict], inproc_rows: Sequence[dict], slots: Mapping[str, dict]) -> list[tuple]:
    """(item id, slot) for the re-sends, at most RESEND_N: items with two first-letter candidates at the slot, those
    whose serve readout split first, then by in-process split mass, largest first (ties by id)."""
    served = {r["id"] for r in serve_rows if r.get("arm") == "wide" and (r.get("split") or 0) > 0}
    mass = {r["id"]: r.get("split_mass") or 0.0 for r in inproc_rows if r.get("mode") == "capped" and "error" not in r}
    ok = [i for i, s in slots.items() if len(s.get("cands") or []) >= 2]
    ok.sort(key=lambda i: (i not in served, -mass.get(i, 0.0), i))
    return [(i, slots[i]) for i in ok[:RESEND_N]]


# ---- gates, report, verdict (pure) ------------------------------------------------------------------------------

def _gate(status: str, value: Any = None, threshold: Any = None, n: int | None = None, note: str | None = None) -> dict:
    return {"status": status, "value": value, "threshold": threshold, "n": n, "note": note}


def _judge(ok: bool, value: Any, threshold: Any, n: int, note: str | None = None) -> dict:
    return _gate("PASS" if ok else "FAIL", value, threshold, n, note)


def _missing(rows: Sequence[dict], expected: int, what: str) -> str | None:
    bad = [r["id"] for r in rows if "error" in r]
    if len(rows) - len(bad) < expected:
        return f"{len(rows) - len(bad)} of {expected} {what} measured" + (f" (failed: {bad[:5]})" if bad else "")
    return None


MASS_REFUSAL = "InvalidInput: option mass is"       # the engine's refusal of an option mass above 1 + tol, as a row error


def _noise(capped_ok: Sequence[dict], baseline: Sequence[dict]) -> dict[str, float]:
    """TV between each item's capped read and its second capped read (the re-read noise floor of the forced TV)."""
    by_id = {c["id"]: c for c in capped_ok}
    return {b["id"]: tv(by_id[b["id"]]["probs"], b["probs"]) for b in baseline
            if "error" not in b and b["id"] in by_id}


def gates(serve_rows: Sequence[dict], inproc_rows: Sequence[dict], forced_rows: Sequence[dict],
          resend_rows: Sequence[dict], *, n_items: int, accuracy: str = "gated",
          baseline: Sequence[dict] = ()) -> dict[str, dict]:
    """The gates over one model's rows. `n_items`: the items the run should have measured (each gate needs all of
    its rows); `accuracy`: "gated" or "reported" (any model but the primary, Tiel);
    `baseline`: the second capped reads of the forced items (noise, reported next to the forced TV)."""
    wide = [r for r in serve_rows if r.get("arm") == "wide"]
    control = [r for r in serve_rows if r.get("arm") == "control"]
    capped = [r for r in inproc_rows if r.get("mode") == "capped"]
    capped_ok = [r for r in capped if "error" not in r]
    out: dict[str, dict] = {}
    why_capped = _missing(capped, n_items, "in-process wide readouts")
    if why_capped:
        out["mass_median"] = _gate("ERROR", note=why_capped)
    else:
        med = statistics.median(r["option_mass"] for r in capped_ok)
        out["mass_median"] = _judge(med >= MASS_MEDIAN_MIN, med, f">= {MASS_MEDIAN_MIN}", len(capped_ok))
    why_serve = _missing(wide, n_items, "serve wide readouts")
    ok_wide = [r for r in wide if "error" not in r]
    if why_serve:
        for g in ("unread_share", "split0_share"):
            out[g] = _gate("ERROR", note=why_serve)
    else:
        share = sum(r["unread"] <= UNREAD_MAX for r in ok_wide) / len(ok_wide)
        out["unread_share"] = _judge(share >= UNREAD_SHARE_MIN, share, f"unread <= {UNREAD_MAX} on >= "
                                     f"{UNREAD_SHARE_MIN:.0%}", len(ok_wide))
        share = sum(r["split"] == 0 for r in ok_wide) / len(ok_wide)
        out["split0_share"] = _judge(share >= SPLIT0_SHARE_MIN, share, f"split == 0 on >= {SPLIT0_SHARE_MIN:.0%}",
                                     len(ok_wide))
    splits = [r["split"] for r in ok_wide] + [r["split_passes"] for r in capped_ok]
    if why_serve or not splits:
        out["split_max"] = _gate("ERROR", note=why_serve or "no readout")
    else:
        out["split_max"] = _judge(max(splits) <= SPLIT_MAX, max(splits), f"<= {SPLIT_MAX}", len(splits))
    out["split_prompt_n"] = _split_prompt_n_gate(ok_wide, resend_rows)
    want = min(FORCED_N, n_items)
    pairs = [(r, next((c for c in capped_ok if c["id"] == r["id"]), None)) for r in forced_rows if "error" not in r]
    pairs = [(f, c) for f, c in pairs if c is not None]
    noise = _noise(capped_ok, baseline)
    noise_note = f"noise baseline max {max(noise.values()):.3g}" if noise else "no noise baseline in this record"
    if len(pairs) < want:
        out["forced_tv"] = _gate("ERROR", note=f"{len(pairs)} of {want} forced-split reads measured")
    else:
        worst = max(tv(c["probs"], f["probs"]) for f, c in pairs)
        if sum(f["split_passes"] for f, _ in pairs) == 0:
            top = max(c["split_mass"] for _, c in pairs)
            out["forced_tv"] = _gate("N/A", worst, f"<= {TV_MAX}", len(pairs),
                                     f"forced reads expanded nothing (max split mass {top:.3g} < 1e-3)")
        else:
            out["forced_tv"] = _judge(worst <= TV_MAX, worst, f"<= {TV_MAX}", len(pairs), noise_note)
    refused = [f"{r['id']} ({r.get('mode')})" for r in [*capped, *forced_rows, *baseline]
               if str(r.get("error", "")).startswith(MASS_REFUSAL)]
    every = [r for r in [*capped_ok, *forced_rows, *baseline] if "error" not in r]
    why_mass = why_capped or _missing(forced_rows, want, "forced-split reads")
    if refused:
        out["mass_bound"] = _gate("FAIL", None, "<= 1 + mass tol", len(refused),
                                  f"the engine refused the option mass of: {refused}")
    elif why_mass or not every:
        out["mass_bound"] = _gate("ERROR", note=why_mass or "no in-process readout")
    else:
        bad = [f"{r['id']} ({r['mode']})" for r in every if not r["mass_ok"]]
        out["mass_bound"] = _judge(not bad, max(r["mass_plus_unread"] for r in every), "<= 1 + mass tol", len(every),
                                   f"over: {bad}" if bad else None)
    out["accuracy"] = accuracy_gate(ok_wide, [r for r in control if "error" not in r], n_items=n_items, mode=accuracy)
    return out


def _split_prompt_n_gate(ok_wide: Sequence[dict], resend_rows: Sequence[dict]) -> dict:
    """Max split-pass prompt_n over every source: the re-sends' s + t1 and s + t2, and the serve readouts' own split
    passes from the `-v` log. Split pass 2 (s + t2, the restore path) must have been measured."""
    sent = [r for r in resend_rows if "error" not in r]
    t1 = [r["prompt_n"] for r in sent if isinstance(r.get("prompt_n"), int)]
    t2 = [r["prompt_n_2"] for r in sent if isinstance(r.get("prompt_n_2"), int)]
    log = [v for r in ok_wide for v in (r.get("split_prompt_n_log") or [])]
    gate = _split_prompt_n_status(ok_wide, resend_rows, sent, t1, t2, log)
    gate["sources"] = {"s+t1": len(t1), "s+t2": len(t2), "-v log": len(log)}   # where `n` came from, for the verdict
    return gate


def _split_prompt_n_status(ok_wide: Sequence[dict], resend_rows: Sequence[dict], sent: Sequence[dict],
                           t1: Sequence[int], t2: Sequence[int], log: Sequence[int]) -> dict:
    values = [*t1, *t2, *log]
    threshold = f"<= {SPLIT_PROMPT_N_MAX} (expected ~1; s+t1, s+t2 and the -v log)"
    if values and max(values) > SPLIT_PROMPT_N_MAX:
        return _gate("FAIL", max(values), threshold, len(values))
    if not resend_rows:
        if any((r.get("split") or 0) > 0 for r in ok_wide):
            return _gate("ERROR", max(values) if values else None, threshold, len(values),
                         "a serve readout split but nothing was re-sent")
        return _gate("N/A", note="no split pass in any readout and no re-send in this record")
    if not sent:
        return _gate("ERROR", note=f"every re-send failed: {resend_rows[0].get('error')}")
    if not t2:
        return _gate("ERROR", max(values), threshold, len(values),
                     "split pass 2 (s + t2, the checkpoint restore) not measured in this record: run --only-resend")
    return _judge(True, max(values), threshold, len(values))


def accuracy_gate(wide: Sequence[dict], control: Sequence[dict], *, n_items: int, mode: str) -> dict:
    """Wide accuracy >= 0.5 per dataset AND pooled wide >= pooled control - 0.25 (a label-mapping bug reads near
    chance); "reported" (Gemma) states the values without judging them."""
    if len(wide) < n_items or len(control) < n_items:
        return _gate("ERROR", note=f"{len(wide)} wide and {len(control)} control readouts of {n_items}")
    per = {s: sum(r["correct"] for r in wide if r["source"] == s) / sum(r["source"] == s for r in wide)
           for s in sorted({r["source"] for r in wide})}
    pooled_w = sum(r["correct"] for r in wide) / len(wide)
    pooled_c = sum(r["correct"] for r in control) / len(control)
    value = {"wide": per, "wide_pooled": pooled_w, "control_pooled": pooled_c}
    threshold = f"wide >= {ACC_WIDE_MIN} per dataset and >= control - {ACC_CONTROL_SLACK} pooled"
    if mode == "reported":
        return _gate("REPORTED", value, threshold, len(wide), "accuracy reported only on this model")
    ok = all(v >= ACC_WIDE_MIN for v in per.values()) and pooled_w >= pooled_c - ACC_CONTROL_SLACK
    return _judge(ok, value, threshold, len(wide))


def overall(g: Mapping[str, dict]) -> str:
    states = {v["status"] for v in g.values()}
    return "FAIL" if "FAIL" in states else "ERROR" if "ERROR" in states else "PASS"


def _pct(xs: Sequence[float], q: float) -> float | None:
    """The nearest-rank q-quantile, as /callo/stats computes it."""
    if not xs:
        return None
    s = sorted(xs)
    return s[max(1, -(-round(q * 10**6) * len(s) // 10**6)) - 1]


def report(serve_rows: Sequence[dict], inproc_rows: Sequence[dict], forced_rows: Sequence[dict],
           resend_rows: Sequence[dict], sizes: Mapping[str, Any], *, baseline: Sequence[dict] = ()) -> dict:
    """What the validation logs without gating: wide vs control readout ms (callo's per-question time, X-Callo-Timings `ms`) p50 and
    p95, the median split mass (the SPLIT_MASS_TRIGGER at 5 %), `partial_by_letter` per item, the served `unread` next to
    the in-process `unread_mass`, checkpoint lines per readout, the split passes' own prompt_n from the log, the
    re-sends, each forced read's TV with its noise baseline, and the n_probs reply sizes."""
    ok = [r for r in serve_rows if "error" not in r]
    capped = [r for r in inproc_rows if r.get("mode") == "capped" and "error" not in r]
    by_id = {c["id"]: c for c in capped}
    noise = _noise(capped, baseline)

    def ms(arm: str, key: str = "ms") -> dict:
        xs = [r[key] for r in ok if r["arm"] == arm and isinstance(r.get(key), (int, float))]
        return {"p50": _pct(xs, 0.5), "p95": _pct(xs, 0.95), "n": len(xs)}

    def cp(arm: str) -> dict:
        rows = [r for r in ok if r["arm"] == arm]
        return {"created_mean": statistics.fmean(r["created"] for r in rows) if rows else None,
                "restored_mean": statistics.fmean(r["restored"] for r in rows) if rows else None,
                "created_max": max((r["created"] for r in rows), default=None),
                "restored_max": max((r["restored"] for r in rows), default=None)}

    split_masses = [r["split_mass"] for r in capped]
    median_split = statistics.median(split_masses) if split_masses else None
    return {"ms": {"wide": ms("wide"), "control": ms("control")},
            "wall_ms": {"wide": ms("wide", "wall_ms"), "control": ms("control", "wall_ms")},
            "split_mass_median": median_split,
            "split_mass_trigger": None if median_split is None else median_split > SPLIT_MASS_TRIGGER,
            "partial_by_letter": {r["id"]: r["partial_by_letter"] for r in capped},
            "checkpoints": {"wide": cp("wide"), "control": cp("control"),
                            "per_readout": [[r["id"], r["arm"], r["created"], r["restored"]] for r in ok]},
            "split_prompt_n_log": {r["id"]: r["split_prompt_n_log"] for r in ok if r.get("split_prompt_n_log")},
            "resend": list(resend_rows),
            "unread": [[r["id"], r.get("unread"), (by_id.get(r["id"]) or {}).get("unread_mass")]
                       for r in ok if r["arm"] == "wide"],
            "forced": [{"id": f["id"], "split_passes": f["split_passes"], "unread_mass": f["unread_mass"],
                        "tv": tv(by_id[f["id"]]["probs"], f["probs"]) if f["id"] in by_id else None,
                        "noise_tv": noise.get(f["id"])}
                       for f in forced_rows if "error" not in f],
            "noise_tv_max": max(noise.values()) if noise else None,
            "inprocess_split_passes": sum(r["split_passes"] for r in capped),
            "reply_bytes": dict(sizes)}


def _fmt(v: Any, spec: str = ".4g") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, int) and not isinstance(v, bool) and spec == ".4g":
        return str(v)                                           # counts and bytes in full
    return format(v, spec) if isinstance(v, (int, float)) else str(v)


def verdict_lines(model: str, g: Mapping[str, dict], rep: Mapping[str, Any], counts: Mapping[str, int]) -> list[str]:
    """The 10-line verdict."""
    def st(k: str) -> str:
        return g[k]["status"] + (f" ({g[k]['note']})" if g[k]["status"] in ("ERROR", "N/A") and g[k]["note"] else "")

    acc = g["accuracy"]["value"] or {}
    cps, size = rep["checkpoints"]["wide"], rep["reply_bytes"]
    tally = Counter(v["status"] for v in g.values())
    gated = ", ".join(f"{tally[s]} {s}" for s in ("PASS", "FAIL", "ERROR", "N/A", "REPORTED") if tally[s])
    src = g["split_prompt_n"].get("sources") or {}
    pn_src = ", ".join(f"{k} {src[k]}" for k in ("s+t1", "s+t2", "-v log") if k in src) or "s+t1, s+t2, -v log"
    return [
        f"wide_check {model}: {overall(g)} (gates: {gated}; "
        f"{counts['serve']} serve readouts, {counts['inprocess']} in-process, "
        f"{counts['forced']} forced-split, {counts.get('baseline', 0)} noise re-reads, {counts['resend']} re-sends)",
        f"median option_mass {_fmt(g['mass_median']['value'])} (>= {MASS_MEDIAN_MIN}): {st('mass_median')}",
        f"unread <= {UNREAD_MAX} on {_fmt(g['unread_share']['value'], '.0%')} (>= {UNREAD_SHARE_MIN:.0%}): "
        f"{st('unread_share')}",
        f"split == 0 on {_fmt(g['split0_share']['value'], '.0%')} (>= {SPLIT0_SHARE_MIN:.0%}): {st('split0_share')}; "
        f"max split {_fmt(g['split_max']['value'])} (<= {SPLIT_MAX}): {st('split_max')}",
        f"split-pass prompt_n max {_fmt(g['split_prompt_n']['value'])} over {_fmt(g['split_prompt_n']['n'])} "
        f"({pn_src}; <= {SPLIT_PROMPT_N_MAX}): {st('split_prompt_n')}",
        f"forced-split TV max {_fmt(g['forced_tv']['value'])} (<= {TV_MAX}; noise {_fmt(rep.get('noise_tv_max'))}): "
        f"{st('forced_tv')}; mass + unread <= 1 + tol: {st('mass_bound')}",
        f"accuracy wide {', '.join(f'{k} {v:.2f}' for k, v in (acc.get('wide') or {}).items()) or 'n/a'}, pooled "
        f"{_fmt(acc.get('wide_pooled'), '.2f')} vs control {_fmt(acc.get('control_pooled'), '.2f')}: {st('accuracy')}",
        f"ms p50/p95 wide {_fmt(rep['ms']['wide']['p50'])}/{_fmt(rep['ms']['wide']['p95'])} vs control "
        f"{_fmt(rep['ms']['control']['p50'])}/{_fmt(rep['ms']['control']['p95'])}",
        f"median split mass {_fmt(rep['split_mass_median'])} (trigger {SPLIT_MASS_TRIGGER:.0%}: "
        f"{'REACHED' if rep['split_mass_trigger'] else 'not reached' if rep['split_mass_trigger'] is False else 'n/a'})",
        f"checkpoints per wide readout: created {_fmt(cps['created_mean'], '.2f')}, restored "
        f"{_fmt(cps['restored_mean'], '.2f')}; reply bytes n_probs 1000 {_fmt(size.get(f'n_probs_{WIDE_TOP_K}'))} "
        f"vs 100 {_fmt(size.get('n_probs_100'))}",
    ]


# ---- the live run -----------------------------------------------------------------------------------------------

def _launch(a: argparse.Namespace, model_path: str) -> Callo:
    return Callo(a.out / a.model, llama_server=a.llama_server, ctx=a.ctx, parallel=a.parallel, model=model_path,
                 cache=True, flags=list(SERVE_FLAGS))


_log_tail = LogTail                             # seam


def _engine_for(upstream: str, gguf: str) -> Any:
    from callo.engine import local_engine  # noqa: PLC0415
    return local_engine(upstream, gguf=gguf)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evaluate(rows: Sequence[dict], *, model: str, n_items: int, sizes: Mapping[str, Any]) -> dict:
    """Gates, report, counts and the verdict from one model's rows (the `kind` each row was written with): the live
    run and `--recompute` both end here."""
    by: dict[str, list[dict]] = {k: [] for k in ("serve", "inprocess", "baseline", "forced", "resend")}
    for r in rows:
        if r.get("kind") in by:
            by[r["kind"]].append(r)
    g = gates(by["serve"], by["inprocess"], by["forced"], by["resend"], n_items=n_items,
              accuracy=MODELS[model]["accuracy"], baseline=by["baseline"])
    rep = report(by["serve"], by["inprocess"], by["forced"], by["resend"], sizes, baseline=by["baseline"])
    counts = {k: len(v) for k, v in by.items()}
    return {"gates": g, "report": rep, "counts": counts, "verdict": overall(g),
            "verdict_lines": verdict_lines(model, g, rep, counts)}


def _failed(item: dict, e: Exception, **extra: Any) -> dict:
    """One item's failure is its row, not the run's end."""
    return {"id": item["id"], "source": item["source"], **extra, "error": f"{type(e).__name__}: {e}"}


def _keeper(rows_file: Any, rows: list[dict]) -> Callable[[str, dict], dict]:
    def keep(kind: str, row: dict) -> dict:
        row = {"kind": kind, **row}
        rows_file.write(json.dumps(row, default=str) + "\n")
        rows_file.flush()
        rows.append(row)
        return row
    return keep


def _prepare(c: Any, items: Sequence[dict], m: dict) -> Any:
    """The launch's checks before anything is sent (ports, coverage, the llama-server log), recorded in `m`."""
    bad = port_problems([c.url, c.upstream])
    if bad:
        raise RunError("; ".join(bad))
    stats = c.stats()
    cov, need = stats.get("wide_coverage"), max(len(it["labels"]) for it in items)
    m.update(build_info=getattr(c, "build_info", None), wide_coverage=cov, user_delimiter=stats.get("user_delimiter"),
             wide_contract=(stats.get("provenance") or {}).get("wide_contract"),
             llama_log=llama_log_path(c.log_text()))
    if not isinstance(cov, int) or isinstance(cov, bool) or cov < need:
        raise RunError(f"the server's wide_coverage is {cov!r}; the items need {need}")
    if m["llama_log"] is None:
        raise RunError("callo's log names no llama-server log, so no checkpoint line can be read")
    return _log_tail(m["llama_log"])


_READ_ERRORS = (CalloError, OSError, ValueError)        # per item (InvalidInput is a ValueError and a CalloError)


def _read(engine: Any, item: dict, calls: Calls, mode: str) -> tuple[dict, list, dict]:
    try:
        return inprocess_read(engine, item, calls, mode=mode)
    except _READ_ERRORS as e:
        return _failed(item, e, mode=mode), [], {"prefix": [], "cands": []}


def _resend_all(c: Any, picks: Sequence[tuple], by_id: Mapping[str, dict], delimiter: str | None,
                keep: Callable[[str, dict], dict]) -> None:
    for item_id, slot in picks:
        t1, t2 = slot["cands"][0][0], slot["cands"][1][0]
        try:
            row = resend(c.url, c.upstream, by_id[item_id], slot["prefix"], t1, t2, post=_POST, delimiter=delimiter)
        except (OSError, ValueError) as e:
            row = _failed(by_id[item_id], e)
        keep("resend", row)


def _measure(a: argparse.Namespace, c: Any, items: list[dict], m: dict, rows_file: Any) -> dict:
    log = _prepare(c, items, m)
    rows: list[dict] = []
    keep = _keeper(rows_file, rows)
    for it in items:
        for arm in ("wide", "control"):
            try:
                r = serve_readout(c.url, it, arm, post=_POST, log=log)
            except (OSError, ValueError) as e:                     # a timeout, a dropped connection, a bad reply
                r = _failed(it, e, arm=arm)
            keep("serve", r)
            print(f"serve {arm:7} {r['id']}: split {r.get('split')} unread {r.get('unread')} ms {r.get('ms')} "
                  f"correct {r.get('correct')} created {r.get('created')} restored {r.get('restored')}"
                  + (f" ERROR {r['error']}" if "error" in r else ""), flush=True)
    m["reply_bytes"] = sizes = reply_sizes(c.upstream, post=_POST, raw_post=_raw_post)
    engine = _engine_for(c.upstream, m["model_path"])
    calls = record_calls(engine)
    slots: dict[str, dict] = {}
    for it in items:
        row, _, slots[it["id"]] = _read(engine, it, calls, "capped")
        keep("inprocess", row)
        print(f"in-process {row['id']}: " + (f"ERROR {row['error']}" if "error" in row else
              f"mass {row['option_mass']:.4f} unread {row['unread_mass']:.4f} split {row['split_passes']} split "
              f"mass {row['split_mass']:.4f}"), flush=True)
    by_id = {it["id"]: it for it in items}
    capped = [r for r in rows if r["kind"] == "inprocess" and "error" not in r]
    top = sorted(capped, key=lambda r: (-r["split_mass"], r["id"]))[:FORCED_N]
    for r in top:                                              # the re-read noise of the same items, capped
        keep("baseline", _read(engine, by_id[r["id"]], calls, "baseline")[0])
    with forced_split(engine):
        for r in top:
            keep("forced", _read(engine, by_id[r["id"]], calls, "forced")[0])
    serve = [r for r in rows if r["kind"] == "serve"]
    _resend_all(c, resend_order(serve, capped, slots), by_id, m["user_delimiter"], keep)
    return evaluate(rows, model=a.model, n_items=len(items), sizes=sizes)


def _resend_only(a: argparse.Namespace, c: Any, items: list[dict], sup: dict, rows_file: Any,
                 prior: Sequence[dict]) -> None:
    """The re-send supplement: in-process reads (for each item's slot and its first-letter candidates), then the s + t1 and
    s + t2 re-sends; items in the order the full run's rows rank them (serve split first, then split mass)."""
    _prepare(c, items, sup)
    rows: list[dict] = []
    keep = _keeper(rows_file, rows)
    served = {r["id"] for r in prior if r.get("kind") == "serve" and r.get("arm") == "wide" and (r.get("split") or 0)}
    mass = {r["id"]: r.get("split_mass") or 0.0 for r in prior if r.get("kind") == "inprocess" and "error" not in r}
    order = sorted(items, key=lambda it: (it["id"] not in served, -mass.get(it["id"], 0.0), it["id"]))
    engine = _engine_for(c.upstream, sup["model_path"])
    calls = record_calls(engine)
    slots: dict[str, dict] = {}
    for it in order:
        row, _, slot = _read(engine, it, calls, "capped")
        keep("resend_read", row)
        if len(slot["cands"]) >= 2:
            slots[it["id"]] = slot
        if len(slots) >= RESEND_N:
            break
    picks = [(i, slots[i]) for i in [it["id"] for it in order] if i in slots][:RESEND_N]
    _resend_all(c, picks, {it["id"]: it for it in items}, sup["user_delimiter"], keep)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recompute(out: Path, model: str) -> str:
    """Offline (no server): the gates, report and verdict of `model` again from OUT/<model>/rows.jsonl (whose serve
    rows carry their parsed log segments), with the re-sends of OUT/<model>/rows-resend.jsonl, when it has any,
    replacing the full run's. Rewrites summary.json's `models.<model>` and OUT/verdict-<model>.txt; the verdict."""
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text())
    m = summary["models"][model]
    rows = read_rows(out / model / "rows.jsonl")
    sup_path = out / model / "rows-resend.jsonl"
    extra = [r for r in read_rows(sup_path) if r.get("kind") == "resend"] if sup_path.exists() else []
    if extra:
        rows = [r for r in rows if r.get("kind") != "resend"] + extra
    n_items = len(m.get("items") or {r["id"] for r in rows if r.get("kind") == "serve"})
    sizes = m.get("reply_bytes") or (m.get("report") or {}).get("reply_bytes") or {}
    res = evaluate(rows, model=model, n_items=n_items, sizes=sizes)
    m.update({k: res[k] for k in ("gates", "report", "counts", "verdict", "verdict_lines")})
    m["recomputed"] = {"at": _now(), "code": code_state().get("head"), "resend_rows": "rows-resend.jsonl" if extra
                       else "rows.jsonl"}
    summary_path.write_text(json.dumps(summary, indent=1, default=str))
    (out / f"verdict-{model}.txt").write_text("\n".join(res["verdict_lines"]) + "\n")
    print("\n".join(res["verdict_lines"]), flush=True)
    return res["verdict"]


def _gpu_after(c: Any) -> Any:
    """The model servers running at the end of the run other than our own llama-server (or why unknown)."""
    try:
        return [[pid, args] for pid, args in lib.running_gpu_servers() if pid != getattr(c, "pid", None)]
    except RuntimeError as e:
        return f"unknown: {e}"


def _run(a: argparse.Namespace, items: list[dict], meta: dict) -> int:
    summary_path = a.out / "summary.json"
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        summary = {}
    summary.setdefault("models", {})
    only = getattr(a, "only_resend", False)
    entry: dict[str, Any] = {"started": _now(), "model_path": MODELS[a.model]["model"], "code": code_state(),
                             "llama_server": a.llama_server, "ctx": a.ctx, "parallel": a.parallel,
                             "flags": list(SERVE_FLAGS), "cache": True, "allow_dirty": a.allow_dirty}
    if only:
        prior_path = a.out / a.model / "rows.jsonl"
        if a.model not in summary["models"] or not prior_path.exists():
            print(f"refusing: --only-resend needs a full run of {a.model} in {a.out} (summary and rows.jsonl)",
                  file=sys.stderr)
            return 2
        prior = read_rows(prior_path)
        m = entry
        summary["models"][a.model]["resend_supplement"] = m
    else:
        m = {**entry, "items": [it["id"] for it in items], "items_file": str(a.items),
             "items_sha256": hashlib.sha256(Path(a.items).read_bytes()).hexdigest(),
             "sources": meta["_meta"].get("sources")}
        summary["models"][a.model] = m

    def save() -> None:
        summary_path.write_text(json.dumps(summary, indent=1, default=str))

    def finish(verdict: str, lines: list[str] | None = None, errors: list[str] | None = None) -> None:
        m.update(verdict=verdict, finished=_now(), **({"errors": errors} if errors else {}))
        if lines:
            m["verdict_lines"] = lines
        save()
        if only and not errors:
            return
        text = "\n".join(lines or [f"wide_check {a.model}: {verdict}: {'; '.join(errors or [])}"]) + "\n"
        (a.out / f"verdict-{a.model}{'-resend' if only else ''}.txt").write_text(text)
        print(text, end="", flush=True)

    save()
    (a.out / a.model).mkdir(parents=True, exist_ok=True)
    c = None
    try:
        try:
            c = _launch(a, m["model_path"])
            name = "rows-resend.jsonl" if only else "rows.jsonl"
            with (a.out / a.model / name).open("w") as rows_file:
                res = _resend_only(a, c, items, m, rows_file, prior) if only else _measure(a, c, items, m, rows_file)
            m["gpu_after"] = _gpu_after(c)
        except (RunError, OSError) as e:
            finish("ERROR", errors=[str(e)])
            return 1
        except Exception as e:                                    # a bug or a surprise: recorded before the stop
            m["traceback"] = traceback.format_exc()
            finish("ERROR", errors=[f"unexpected {type(e).__name__}: {e}"])
            return 1
        except BaseException as e:                                # a stop signal (SystemExit) or Ctrl-C
            finish("STOPPED", errors=[f"{type(e).__name__}: {e}"])
            raise
        if only:
            finish("DONE")
            return 0 if recompute(a.out, a.model) == "PASS" else 1
        m.update({k: res[k] for k in ("gates", "report", "counts")})
        finish(res["verdict"], res["verdict_lines"])
        return 0 if res["verdict"] == "PASS" else 1
    finally:
        if c is not None:
            c.finish()


def code_refusals(state: Mapping[str, Any]) -> list[str]:
    """Why the code under test is not a committed tree imported from this checkout."""
    if state.get("error"):
        return [f"cannot read the checkout's git state ({state['error']}): run from a git worktree"]
    out = []
    dirty = [ln for ln in (state.get("dirty") or "").splitlines()       # an untracked non-.py file (.DS_Store) is not code
             if any(ln[3:].startswith(t) for t in TRACKED) and not (ln.startswith("??") and not ln.endswith(".py"))]
    if dirty:
        out.append(f"uncommitted changes under {', '.join(TRACKED)}: {dirty[:5]} (run from a frozen worktree)")
    root = Path(state.get("root") or ".")
    for name, sub in (("callo", "src"), ("bench", "bench")):
        where = (state.get("sources") or {}).get(name)
        if where is None or not Path(where).resolve().is_relative_to((root / sub).resolve()):
            out.append(f"{name} is imported from {where}, not from {root / sub}: use this checkout's own environment")
    return out


def _pid_refusal(pid_file: Path) -> bool:
    """A live `run.pid` refuses the start (and a --recompute, which rewrites the same summary)."""
    owner = pid_owner(pid_file)
    if owner is None:
        return False
    if lib.is_bench_run(owner, BENCH_MODULE):
        print(f"refusing to start: another run (pid {owner}) owns this --out; stop it through its run.pid",
              file=sys.stderr)
    else:
        print(f"refusing to start: run.pid names pid {owner}, which is not a run of this check: check it and "
              f"remove run.pid by hand", file=sys.stderr)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=BENCH_MODULE, description="the wide-choice GPU validation. GPU: one "
                                 "model server. Stop it ONLY with kill -TERM $(cat OUT/run.pid).")
    ap.add_argument("--model", choices=sorted(MODELS), default="tiel")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--items", type=Path, default=ITEMS)
    ap.add_argument("--llama-server", default=MASTER)
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--parallel", type=int, default=PARALLEL)
    ap.add_argument("--allow-dirty", action="store_true", help="record uncommitted changes instead of refusing")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--recompute", action="store_true", help="offline, no server: the gates, report and verdict "
                      "of --model again from OUT/<model>/rows.jsonl (and rows-resend.jsonl)")
    mode.add_argument("--only-resend", action="store_true", help="GPU, a few minutes: only the s + t1 / s + t2 "
                      "re-sends (OUT/<model>/rows-resend.jsonl), then --recompute")
    a = ap.parse_args(argv)
    unset = lib.unset_paths(("--out", a.out, lib.OUT_ENV), *([] if a.recompute else [
        ("", MODELS[a.model]["model"], lib.MODELS_ENV), ("--llama-server", a.llama_server, lib.SERVER_ENV)]))
    if unset:
        print(unset, file=sys.stderr)
        return 2
    try:
        meta, items = load_items(a.items, model=a.model)
    except (OSError, ValueError) as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return 2
    if a.recompute:
        if _pid_refusal(a.out / "run.pid"):
            return 2
        rows = a.out / a.model / "rows.jsonl"
        try:
            has_model = a.model in json.loads((a.out / "summary.json").read_text()).get("models", {})
        except (OSError, ValueError):
            has_model = False
        if not rows.exists() or not has_model:
            print(f"refusing to recompute: no {rows} or no models.{a.model} in {a.out / 'summary.json'}",
                  file=sys.stderr)
            return 2
        return 0 if recompute(a.out, a.model) == "PASS" else 1
    problems = code_refusals(code_state())
    if problems and not a.allow_dirty:
        print("refusing to start: " + "; ".join(problems), file=sys.stderr)
        return 2
    try:
        busy = lib.running_gpu_servers()
    except RuntimeError as e:
        print(f"refusing to start: {e}: cannot tell whether a model server holds the GPU", file=sys.stderr)
        return 2
    if busy:
        print(f"refusing to start: a model server holds the GPU (one at a time): {busy}", file=sys.stderr)
        return 2
    bad = disk_problems([*DISK_PATHS], MIN_FREE_GIB)
    if bad:
        print("refusing to start: " + "; ".join(bad), file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    pid_file = a.out / "run.pid"
    if _pid_refusal(pid_file):
        return 2
    previous = _install_stop_handlers()
    me = os.getpid()
    try:
        pid_file.write_text(f"{me}\n")
        return _run(a, items, meta)
    finally:
        if _pid_in(pid_file) == me:
            pid_file.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
