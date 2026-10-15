"""Live check of `--cache` with the user-turn delimiter on plain `/v1/systemone` readouts (written, not yet run).

Two launches of the same GGUF, one at a time, both with llama-server's host prompt cache off (`-- --cache-ram 0`):

1. CACHED: `callo serve --model GGUF --cache --s1-slots 1`. For each request body, two readouts on the reserved slot:
   - COLD: every slot erased (`POST /slots/N?action=erase`; launch mode passes `--slot-save-path`), then the body;
   - WARM: every slot erased, a DIFFERENT body sent first (the next one in the list, as a game-client agent sends one
     state after another), then the body.
2. OFF (the drift reference, what calibration measures): the first server stopped and its llama-server pid confirmed
   gone, then `callo serve` again with `--no-cache` (explicit: serve's default is on since 0.3.0); each body
   read once after an erase.

Reported per body and question:
- max |Δ logprob| over the options (the log of the served probabilities, at the server's temperature: no
  `--think-profile`, so T = 1) for warm vs cold, warm vs off and cold vs off, and whether the argmax agrees (not for
  `score` answers; a `noul` answer is read as the two-way distribution (1 - noul, noul));
- the first upstream pass's `prompt_n`, `cache_n` and prompt time (`log_ms`) of each arm, and each readout's wall time.
  X-Callo-Timings carries those only for media and `context` readouts (the plain pass returns logprobs alone), so for
  plain ones they come from the llama-server log: the `prompt eval time = MS ms / N tokens` line of each pass on the
  readout's slot gives `prompt_n` and `log_ms`, and `cache_n` is the readout's `tokens` (X-Callo-Timings) minus it;
- attribution: each question's user-turn start, the token index where the delimiter's tokens first occur in its
  tokenized prompt (rendered with callo's own `messages` through the upstream's `/apply-template`, tokenized by its
  `/tokenize` as the engine does). That is where llama.cpp puts the user-turn checkpoint (common/chat.cpp `split`:
  a span starts at the delimiter's first token), so a warm first pass whose `cache_n` is within `--attr-tol` of it
  reused exactly the system head.

Verdicts: DONE, or ERROR when the run would measure nothing or something else: no user-turn delimiter or `--cache`
not reaching the cached server (its provenance has no `cache_delimiter`), `--cache` on in the OFF server, no
llama-server log path in callo's log, no warm count read from that log, a cold / primer / warm readout that landed on
different slots, a failed request or erase, the first llama-server still alive after the stop, or any unexpected
exception (summary.json records it before the server is stopped). A stop signal records STOPPED.

Request bodies. A game-client agent recording (`game-NN.json`) does NOT hold them: it stores the frame step deltas,
the (frame, decision key, choice) tuples and the panel events (probabilities and latency, not the state or the
questions). Rebuilding a body means replaying the game through the game client's own TypeScript (`src/replay.ts`
drives the sim, `src/brain.ts` `buildRequest` makes the body), which this script does not do. So `--bodies` takes files
of bodies: `.jsonl` (one `{"state": ..., "questions": {...}}` per line) or `.json` (one body, or a list of them).
Capture them by pointing the game client's `--endpoint` at a logging proxy, or dump `buildRequest`'s output from a
replay. A recording passed by mistake is refused with this explanation, as is a body with media or `context` (those are always cached).

Run it from a frozen worktree of the commit under test (`git worktree add ../callosum-cachecheck <sha>`, then
`uv sync --frozen` there), with that worktree's interpreter and no `uv` wrapper, under nohup:

    nohup .venv/bin/python -m bench.check_cache_delimiters --model GGUF --bodies bodies.jsonl --out OUT \\
        > OUT.log 2>&1 &

Hygiene, as the other v0.3 benches:
- refuses to start (exit 2) while a model server holds the GPU (`lib.running_gpu_servers`, fails closed), on a live
  `OUT/run.pid`, or with fewer than two bodies;
- its pid is in `OUT/run.pid` for the whole run, removed at exit only while it still names this process; stop it ONLY
  with `kill -TERM $(cat OUT/run.pid)`: the stop handlers raise SystemExit, so `finally` runs `Callo.finish()`, which
  stops callo and reaps the llama-server it launched (never by process name);
- every server binds an ephemeral port (callo `--port 0`, launch mode picks llama-server's). `Callo` itself reads
  `/props` through callo while it starts; after that, a port in `FORBIDDEN_PORTS` (other sessions' servers) is an
  ERROR before this check sends anything;
- `OUT/summary.json` records the commit and dirty state, each launch's stats, the verdict and the summary; the rows
  go to `OUT/rows.jsonl` as they are measured (rewritten with the OFF arm at the end).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from bench import lib
from bench.acceptance import MASTER, Callo
from bench.acceptance import _alive as _proc_alive
from bench.acceptance_v02 import _install_stop_handlers
from bench.acceptance_v03_bias import code_state
from bench.acceptance_v03_think import _pid_in, pid_owner
from callo.prompt import TemplateClient, messages
from callo.scorer import LlamaCppForwardScorer, make_http_post
from callo.systemone import parse_request

FORBIDDEN_PORTS = (8080, 8130, 8190)      # other sessions' servers (8190: the game-client agent runs): never bound or called
BENCH_MODULE = "check_cache_delimiters"
CTX = 65536
PARALLEL = 4
ATTR_TOL = 4                              # tokens: warm cache_n within this of the user-turn start counts as attributed
GONE_WAIT_S = 120.0                       # how long the first llama-server gets to exit before the second launch
NOT_PLAIN = ("context", "images", "audio", "videos")

_PROMPT_EVAL = re.compile(r"slot\s+\S+: id\s+(\d+) \| task (-?\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
_LOG_PATH = re.compile(r"launched llama-server pid \d+ on \S+ \(log (.+)\)\s*$", re.M)

Post = Callable[..., tuple[int, dict, Any]]
_POST: Post = lib.post                    # the seam the tests swap: every request of the live run goes through it


# ---- the pure half (unit-tested on fakes) -----------------------------------------------------------------------------

def _check_body(b: Any, where: str) -> dict:
    if not isinstance(b, dict):
        raise ValueError(f"{where}: a body is a JSON object, got {type(b).__name__}")
    if "decisions" in b and "frames" in b:
        raise ValueError(f"{where}: this is a game-client agent recording, which holds no request bodies (frames, decisions and "
                         f"events only); see this module's docstring for how to capture bodies")
    if "state" not in b or not isinstance(b.get("questions"), dict) or not b["questions"]:
        raise ValueError(f"{where}: a body needs `state` and a non-empty `questions` object")
    extra = [k for k in NOT_PLAIN if b.get(k)]
    if extra:
        raise ValueError(f"{where}: not a plain body ({', '.join(extra)}): media and context requests are always "
                         f"cached, so they do not test --cache")
    return b


def load_bodies(paths: Sequence[str | Path]) -> list[dict]:
    """Every body in `paths`, in order: a `.jsonl` file holds one per line (blank lines skipped), a `.json` file one
    body or a list of them. ValueError naming the file (and line) on anything else."""
    out: list[dict] = []
    for p in map(Path, paths):
        text = p.read_text()
        if p.suffix == ".jsonl":
            for n, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    try:
                        out.append(_check_body(json.loads(line), f"{p}:{n}"))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{p}:{n}: not JSON: {e}") from None
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}: not JSON: {e}") from None
            items = data if isinstance(data, list) else [data]
            out.extend(_check_body(b, f"{p}[{i}]" if isinstance(data, list) else str(p)) for i, b in enumerate(items))
    return out


def primer_index(i: int, n: int) -> int:
    """The body the warm readout of body `i` follows: the next one, cyclically (never itself)."""
    if n < 2:
        raise ValueError("the warm readout needs at least two bodies")
    return (i + 1) % n


def _log(p: float) -> float:
    return math.log(p) if p > 0 else -math.inf


def _dlog(a: float, b: float) -> float:
    la, lb = _log(a), _log(b)
    if la == lb:                            # both -inf included
        return 0.0
    return abs(la - lb)


def answer_probs(a: Any) -> tuple[dict[str, float] | None, str | None]:
    """(the option distribution of one served answer, why there is none). `choice` and `score` answers carry
    `probabilities`; a `noul` answer carries only `noul` = P(option 1), read as {"0": 1 - noul, "1": noul}."""
    if not isinstance(a, dict):
        return None, "no answer"
    if a.get("type") == "noul":
        v = a.get("noul")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None, "a noul answer without a numeric `noul`"
        return {"0": 1.0 - float(v), "1": float(v)}, None
    p = a.get("probabilities")
    if not isinstance(p, dict) or not p:
        return None, f"a {a.get('type') or 'typeless'} answer without `probabilities`"
    return {str(k): float(v) for k, v in p.items()}, None


def compare_answers(ref: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, dict]:
    """Per question of `ref`: `dlogprob` per option (|log p_other - log p_ref|, inf when only one is 0),
    `max_dlogprob`, and `argmax_agree` (the most probable option; None for a `score` answer, whose argmax is not its
    reading). An answer with no distribution, or with other options, is an `error` entry saying why."""
    out: dict[str, dict] = {}
    for q, ra in ref.items():
        rp, why = answer_probs(ra)
        op, why_o = answer_probs(other.get(q))
        if rp is None or op is None:
            out[q] = {"error": f"cannot compare: {why or why_o}"}
            continue
        if set(rp) != set(op):
            out[q] = {"error": "the two answers do not have the same options"}
            continue
        d = {k: _dlog(op[k], rp[k]) for k in rp}
        score = isinstance(ra, dict) and ra.get("type") == "score"
        top_r, top_o = max(rp, key=rp.__getitem__), max(op, key=op.__getitem__)
        out[q] = {"dlogprob": d, "max_dlogprob": max(d.values()), "argmax_agree": None if score else top_r == top_o,
                  "argmax_ref": None if score else top_r, "argmax_other": None if score else top_o}
    return out


def parse_timings(header: str | None) -> list[dict]:
    """X-Callo-Timings as a list of entries ([] when absent or unreadable; a `truncated` tail entry is dropped)."""
    if not header:
        return []
    try:
        v = json.loads(header)
    except ValueError:
        return []
    return [e for e in v if isinstance(e, dict) and "q" in e] if isinstance(v, list) else []


def parse_prompt_evals(log_text: str, slot: int) -> list[dict]:
    """The llama-server `prompt eval time = MS ms / N tokens` lines of `slot` (never another slot's), one per finished
    task, in order: {slot, task, ms, prompt_n}."""
    out = []
    for m in _PROMPT_EVAL.finditer(log_text):
        s = int(m.group(1))
        if s == slot:
            out.append({"slot": s, "task": int(m.group(2)), "ms": float(m.group(3)), "prompt_n": int(m.group(4))})
    return out


def llama_log_path(callo_log: str) -> str | None:
    """The llama-server log path from callo's `launched llama-server pid N on URL (log PATH)` line."""
    m = _LOG_PATH.search(callo_log)
    return m.group(1).strip() if m else None


def first_pass_counts(timings: Sequence[dict], evals: Sequence[dict]) -> dict[str, dict]:
    """Per question: the first upstream pass's `prompt_n` / `cache_n` / `log_ms` and where they came from.
    X-Callo-Timings' own numbers when it has them (media and context readouts); otherwise the log's prompt-eval line of
    that pass (questions run in order, each `passes` long, so question k's first pass is the sum of the earlier
    `passes`) and `cache_n = tokens - prompt_n`. None where neither source says. A question with no pass (`passes` not
    a positive integer: a single-option question is answered without the upstream) has None counts and no source, and
    takes no log line, so it never borrows the next question's."""
    out: dict[str, dict] = {}
    at = 0
    for e in timings:
        passes = e.get("passes") if isinstance(e.get("passes"), int) and not isinstance(e.get("passes"), bool) else None
        if passes is None or passes <= 0:
            out[e["q"]] = {"prompt_n": None, "cache_n": None, "log_ms": None, "tokens": e.get("tokens"),
                           "passes": passes, "source": None, "ms": e.get("ms")}
            continue
        pn, cn, ms, src = e.get("prompt_n"), e.get("cache_n"), e.get("prompt_ms"), "timings"
        if pn is None:
            ev = evals[at] if at < len(evals) else None
            pn, ms, src = (ev["prompt_n"], ev["ms"], "log") if ev else (None, None, None)
            tokens = e.get("tokens")
            cn = tokens - pn if isinstance(tokens, int) and pn is not None else None
        out[e["q"]] = {"prompt_n": pn, "cache_n": cn, "log_ms": ms, "tokens": e.get("tokens"), "passes": passes,
                       "source": src, "ms": e.get("ms")}
        at += passes
    return out


def first_question(counts: Mapping[str, dict]) -> str | None:
    """The first question of a readout that made an upstream pass (`passes` > 0), or None."""
    return next((q for q, c in counts.items() if isinstance(c.get("passes"), int) and c["passes"] > 0), None)


def token_index(seq: Sequence[int], sub: Sequence[int]) -> int | None:
    """Where `sub` first occurs in `seq` (llama.cpp matches the delimiter token-wise), or None."""
    n = len(sub)
    if not n:
        return None
    return next((i for i in range(len(seq) - n + 1) if list(seq[i:i + n]) == list(sub)), None)


def user_starts(body: dict, delimiter: str, *, render: Callable[[list[dict]], str],
                tokenize: Callable[..., list[int]]) -> dict[str, int | None]:
    """Per question of a plain body: the token index of its user-turn start (the delimiter's first token) in the
    prompt callo's plain readout sends: `messages(state, q, "main")` through `render` (the upstream's
    /apply-template), tokenized with add_special as the engine does (the server parses specials by default)."""
    req = parse_request(body)
    d = tokenize(delimiter, add_special=False, parse_special=True)
    return {q.qid: token_index(tokenize(render(messages(req.state, q, "main")), add_special=True), d)
            for q in req.questions}


def attribution(row: dict, tol: int = ATTR_TOL) -> dict:
    """The warm readout's first question with an upstream pass (`first_question`): its first-pass cache_n against its
    user-turn start. `attributed` is True when cache_n is within `tol` tokens of it, None when either number is
    missing."""
    warm = row["warm"]["counts"]
    q = first_question(warm)
    if q is None:
        return {"q": None, "cache_n": None, "user_start": None, "attributed": None}
    cn, start = warm[q].get("cache_n"), (row.get("user_start") or {}).get(q)
    ok = None if not isinstance(cn, int) or not isinstance(start, int) else abs(cn - start) <= tol
    return {"q": q, "cache_n": cn, "user_start": start, "attributed": ok}


def port_problems(urls: Sequence[str | None]) -> list[str]:
    """Each URL whose port is a forbidden one (or that has none)."""
    out = []
    for u in urls:
        port = urlsplit(u).port if u else None
        if port is None or port in FORBIDDEN_PORTS:
            out.append(f"{u}: port {port} is forbidden or unknown (FORBIDDEN_PORTS {FORBIDDEN_PORTS})")
    return out


def stats_problems(stats: Mapping[str, Any], *, cached: bool) -> list[str]:
    """What makes a launch measure the wrong thing. CACHED: a user-turn delimiter, `cache_prompt.plain` true and
    `provenance.cache_delimiter` (the plain readouts send the delimiter). OFF: `cache_prompt.plain` false."""
    plain = (stats.get("cache_prompt") or {}).get("plain")
    if not cached:
        return [] if plain is False else [f"the OFF server has cache_prompt.plain {plain!r}, not false"]
    out = []
    if not stats.get("user_delimiter"):
        out.append("callo derived no user-turn delimiter: the plain readouts send none, so there is nothing to check")
    if plain is not True:
        out.append("cache_prompt.plain is not true: --cache did not reach the server")
    if not (stats.get("provenance") or {}).get("cache_delimiter"):
        out.append("provenance.cache_delimiter is not set: this callo's plain readouts do not send the delimiter")
    return out


def run_problems(rows: Sequence[dict]) -> list[str]:
    """What voids measured rows: readouts of one body on different slots, or no warm count from the llama-server log."""
    out = []
    moved = [r["body"] for r in rows if not r.get("same_slot")]
    if moved:
        out.append(f"bodies {moved}: the cold, primer and warm readouts did not all land on one slot")
    if rows and not any(c.get("source") == "log" for r in rows for c in r["warm"]["counts"].values()):
        out.append("no warm first-pass count came from the llama-server log (no prompt-eval lines on the readout slot)")
    return out


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _dstats(rows: Sequence[dict], key: str) -> dict:
    entries = [q for r in rows for q in (r.get(key) or {}).values()]
    qs = [q for q in entries if "error" not in q]
    finite = [q["max_dlogprob"] for q in qs if math.isfinite(q["max_dlogprob"])]
    judged = [q["argmax_agree"] for q in qs if q["argmax_agree"] is not None]
    return {"questions": len(qs), "errors": len(entries) - len(qs),
            "max_dlogprob": max((q["max_dlogprob"] for q in qs), default=None), "mean_max_dlogprob": _mean(finite),
            "infinite": len(qs) - len(finite), "argmax_agree": sum(judged), "argmax_judged": len(judged)}


def summarize(rows: Sequence[dict]) -> dict:
    """Across every question: the three comparisons (warm vs cold, warm vs off, cold vs off), each arm's mean
    first-pass prompt_n / cache_n / log_ms and mean readout wall_ms, and the attribution count."""
    def counts(arm: str, key: str) -> list[float]:
        return [c[key] for r in rows if r.get(arm) for c in r[arm]["counts"].values()
                if isinstance(c.get(key), (int, float)) and not isinstance(c.get(key), bool)]

    def first(arm: str, key: str) -> list[float]:
        out = []
        for r in rows:
            c = r[arm]["counts"] if r.get(arm) else {}
            q = first_question(c)
            v = c[q].get(key) if q is not None else None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(v)
        return out

    # every question mixes in questions 2+, which restore question 1's checkpoint; `first_question` is each readout's
    # first question with a pass only, the one the cold/warm contrast is about (its `ms` is callo's per-question time)
    arms = {arm: {"prompt_n_mean": _mean(counts(arm, "prompt_n")), "cache_n_mean": _mean(counts(arm, "cache_n")),
                  "first_pass_log_ms_mean": _mean(counts(arm, "log_ms")),
                  "wall_ms_mean": _mean([r[arm]["wall_ms"] for r in rows if r.get(arm)]),
                  "first_question": {"prompt_n_mean": _mean(first(arm, "prompt_n")),
                                     "cache_n_mean": _mean(first(arm, "cache_n")),
                                     "log_ms_mean": _mean(first(arm, "log_ms")), "ms_mean": _mean(first(arm, "ms"))}}
            for arm in ("cold", "warm", "off")}
    attr = [(r.get("attribution") or {}).get("attributed") for r in rows]
    return {"bodies": len(rows), "warm_vs_cold": _dstats(rows, "compare"),
            "warm_vs_off": _dstats(rows, "warm_vs_off"), "cold_vs_off": _dstats(rows, "cold_vs_off"),
            "arms": arms, "attributed": sum(a is True for a in attr), "attribution_judged": sum(a is not None for a in attr)}


def progress_line(row: dict) -> str:
    """One line per body: the largest warm-vs-cold |Δ logprob|, argmax agreement, each arm's first-pass prompt_n
    (warm: with cache_n), and the warm first question's user-turn start."""
    s = row["compare"]
    worst = max((q.get("max_dlogprob", math.nan) for q in s.values()), default=math.nan)
    agree = sum(q.get("argmax_agree") is True for q in s.values())
    cold = [v["prompt_n"] for v in row["cold"]["counts"].values()]
    warm = [(v["prompt_n"], v["cache_n"]) for v in row["warm"]["counts"].values()]
    start = (row.get("attribution") or {}).get("user_start")
    return f"body {row['body']}: max|dlogp| {worst:.4g}, argmax {agree}/{len(s)}, cold prompt_n {cold}, warm " \
           f"(prompt_n, cache_n) {warm}, user-turn start {start}"


# ---- the measurement (seams: `post`, `read_log`, `starts`) ------------------------------------------------------------

class RunError(RuntimeError):
    """A readout or an erase that failed: the run stops as an ERROR, its rows so far kept."""


def _erase_all(upstream: str, parallel: int, post: Post) -> None:
    for s in range(parallel):
        status, _, body = post(f"{upstream}/slots/{s}?action=erase", {}, timeout=120)
        if status != 200:
            raise RunError(f"erasing slot {s}: HTTP {status}: {str(body)[:300]}")


def readout(url: str, body: dict, *, post: Post, read_log: Callable[[], str], timeout: float = 900,
            log_wait: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> dict:
    """One /v1/systemone request: the answers, the slot, X-Callo-Timings, the wall time, and the llama-server log
    lines it added (read before and after), turned into first-pass counts on the readout's slot. llama-server may
    write a task's timing line just after its reply, so the log is re-read for up to `log_wait` seconds until it
    holds one line per upstream pass (`passes`)."""
    before = len(read_log())
    t0 = time.perf_counter()
    status, headers, out = post(f"{url}/v1/systemone", body, timeout=timeout)
    wall_ms = 1000 * (time.perf_counter() - t0)
    if status != 200 or not isinstance(out, dict) or not isinstance(out.get("answers"), dict):
        raise RunError(f"/v1/systemone: HTTP {status}: {str(out)[:300]}")
    slot_h = headers.get("x-callo-slot")
    if slot_h is None or not str(slot_h).isdigit():
        raise RunError(f"/v1/systemone answered without an integer X-Callo-Slot ({slot_h!r}): the readout's slot, and "
                       f"so its llama-server log lines, are unknown")
    slot = int(slot_h)
    timings = parse_timings(headers.get("x-callo-timings"))
    want = sum(e["passes"] for e in timings if isinstance(e.get("passes"), int))
    deadline = time.monotonic() + log_wait
    while True:
        evals = parse_prompt_evals(read_log()[before:], slot)
        if len(evals) >= want or time.monotonic() >= deadline:
            break
        sleep(0.05)
    return {"answers": out["answers"], "slot": slot, "timings": timings, "evals": evals, "wall_ms": round(wall_ms, 1),
            "counts": first_pass_counts(timings, evals)}


def measure_cached(url: str, upstream: str, parallel: int, bodies: Sequence[dict], *, read_log: Callable[[], str],
                   starts: Callable[[dict], dict[str, int | None]] = lambda b: {}, post: Post = lib.post,
                   tol: int = ATTR_TOL, on_row: Callable[[dict], None] = lambda r: None) -> list[dict]:
    """The CACHED launch. For each body: erase, COLD readout; erase, the primer (`primer_index`), WARM readout;
    compare, and attribute the warm first pass against `starts(body)` (the user-turn starts)."""
    rows = []
    for i, body in enumerate(bodies):
        j = primer_index(i, len(bodies))
        _erase_all(upstream, parallel, post)
        cold = readout(url, body, post=post, read_log=read_log)
        _erase_all(upstream, parallel, post)
        primer = readout(url, bodies[j], post=post, read_log=read_log)
        warm = readout(url, body, post=post, read_log=read_log)
        row = {"body": i, "primer": j, "user_start": starts(body), "cold": cold,
               "primer_readout": {"slot": primer["slot"], "counts": primer["counts"], "wall_ms": primer["wall_ms"]},
               "warm": warm, "compare": compare_answers(cold["answers"], warm["answers"]),
               "same_slot": cold["slot"] == primer["slot"] == warm["slot"]}
        row["attribution"] = attribution(row, tol)
        rows.append(row)
        on_row(row)
    return rows


def measure_off(url: str, upstream: str, parallel: int, bodies: Sequence[dict], rows: list[dict], *,
                read_log: Callable[[], str], post: Post = lib.post) -> list[dict]:
    """The OFF launch: each body read once after an erase; warm-vs-off and cold-vs-off added to its row."""
    for row in rows:
        _erase_all(upstream, parallel, post)
        off = readout(url, bodies[row["body"]], post=post, read_log=read_log)
        row.update(off=off, warm_vs_off=compare_answers(off["answers"], row["warm"]["answers"]),
                   cold_vs_off=compare_answers(off["answers"], row["cold"]["answers"]))
    return rows


def wait_gone(pid: int | None, *, timeout: float = GONE_WAIT_S, alive: Callable[[int], bool] = _proc_alive,
              sleep: Callable[[float], None] = time.sleep) -> bool:
    """Whether `pid` (the first launch's llama-server) is gone within `timeout`: one model server at a time. `alive` is
    bench.acceptance's check (a zombie counts as gone). An unknown pid is never "gone"."""
    if pid is None:
        return False
    deadline = time.monotonic() + timeout
    while alive(pid):
        if time.monotonic() >= deadline:
            return False
        sleep(0.5)
    return True


_wait_gone = wait_gone                    # seam


# ---- the live run ----------------------------------------------------------------------------------------------------

def _read_file(path: str | None) -> Callable[[], str]:
    def read() -> str:
        if path is None:
            return ""
        try:
            return Path(path).read_text(errors="replace")
        except OSError:
            return ""
    return read


def _starts_via(upstream: str, delimiter: str) -> Callable[[dict], dict[str, int | None]]:
    post = make_http_post(timeout=120)
    template, scorer = TemplateClient(upstream, http_post=post), LlamaCppForwardScorer(upstream, http_post=post)
    return lambda body: user_starts(body, delimiter, render=template.render, tokenize=scorer.tokenize)


def _launch(a: argparse.Namespace, name: str, *, cache: bool) -> Callo:
    return Callo(a.out / name, llama_server=a.llama_server, ctx=a.ctx, parallel=a.parallel, model=a.model,
                 cache=cache, flags=["--s1-slots", "1", "--", "--cache-ram", "0"])   # --cache or --no-cache, explicit


def _checked(c: Callo, launch: dict, *, cached: bool) -> list[str]:
    """The port check BEFORE this check's first request (nothing else is sent when it fails), then the stats and the
    llama-server log path, recorded in `launch`; the problems."""
    bad = port_problems([c.url, c.upstream])
    if bad:
        return bad
    stats = c.stats()
    launch.update(build_info=c.build_info, llama_pid=getattr(c, "pid", None), llama_log=llama_log_path(c.log_text()),
                  stats={k: stats.get(k) for k in ("user_delimiter", "cache_prompt", "provenance", "kv", "n_ctx",
                                                   "total_slots", "temperature")})
    bad = stats_problems(stats, cached=cached)
    if launch["llama_log"] is None:
        bad.append("callo's log names no llama-server log, so no prompt_n / cache_n can be read")
    return bad


def _run(a: argparse.Namespace, bodies: list[dict]) -> int:
    summary: dict[str, Any] = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "model": a.model, "llama_server": a.llama_server, "ctx": a.ctx, "parallel": a.parallel,
                               "bodies": [str(p) for p in a.bodies], "n_bodies": len(bodies), "attr_tol": a.attr_tol,
                               "code": code_state(), "launches": {"cached": {}, "off": {}}}
    summary_path = a.out / "summary.json"
    rows: list[dict] = []

    def save() -> None:
        summary_path.write_text(json.dumps(summary, indent=1, default=str))

    def error(errors: list[str]) -> int:
        summary.update(verdict="ERROR", errors=errors, summary=summarize(rows) if rows else None,
                       finished=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        save()
        print("ERROR: " + "; ".join(errors), file=sys.stderr)
        return 1

    c = None
    try:
        try:
            c = _launch(a, "cached", cache=True)
            cached = summary["launches"]["cached"]
            bad = _checked(c, cached, cached=True)
            if bad:
                return error(bad)
            with (a.out / "rows.jsonl").open("w") as f:
                def on_row(row: dict) -> None:
                    f.write(json.dumps(row, default=str) + "\n")
                    f.flush()
                    print(progress_line(row), flush=True)
                rows = measure_cached(c.url, c.upstream, a.parallel, bodies, read_log=_read_file(cached["llama_log"]),
                                      starts=_starts_via(c.upstream, cached["stats"]["user_delimiter"]),
                                      post=_POST, tol=a.attr_tol, on_row=on_row)
            bad = run_problems(rows)
            if bad:
                return error(bad)
            first_pid = cached["llama_pid"]
            c.finish()
            c = None
            if not _wait_gone(first_pid):
                return error([f"the cached launch's llama-server (pid {first_pid}) is not confirmed gone after the "
                              f"stop: the OFF launch would make two model servers"])
            c = _launch(a, "off", cache=False)
            off = summary["launches"]["off"]
            bad = _checked(c, off, cached=False)
            if bad:
                return error(bad)
            measure_off(c.url, c.upstream, a.parallel, bodies, rows, read_log=_read_file(off["llama_log"]), post=_POST)
            with (a.out / "rows.jsonl").open("w") as f:          # the rows again, now with the OFF arm
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")
        except (RunError, OSError) as e:
            return error([str(e)])
        except Exception as e:                                    # a bug or a surprise: recorded before the stop
            summary["traceback"] = traceback.format_exc()
            return error([f"unexpected {type(e).__name__}: {e}"])
        except BaseException as e:                                # a stop signal (SystemExit) or Ctrl-C
            summary.update(verdict="STOPPED", errors=[f"{type(e).__name__}: {e}"],
                           summary=summarize(rows) if rows else None)
            save()
            raise
        summary.update(verdict="DONE", summary=summarize(rows),
                       finished=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        save()
        print(json.dumps(summary["summary"], indent=1))
        return 0
    finally:
        if c is not None:
            c.finish()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=BENCH_MODULE, description="cold vs warm plain readouts under --cache with the "
                                 "user-turn delimiter, against a cache-off reference. GPU: two launches, one at a "
                                 "time. Stop it ONLY with kill -TERM $(cat OUT/run.pid).")
    ap.add_argument("--model", required=True, help="the GGUF to launch")
    ap.add_argument("--bodies", nargs="+", required=True, help=".jsonl (one body per line) or .json files of bodies")
    ap.add_argument("--out", type=Path, required=True, help="run.pid, rows.jsonl, summary.json, cached/, off/")
    ap.add_argument("--limit", type=int, default=None, help="use the first N bodies (at least 2)")
    ap.add_argument("--llama-server", default=MASTER)
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--parallel", type=int, default=PARALLEL)
    ap.add_argument("--attr-tol", type=int, default=ATTR_TOL,
                    help="tokens: a warm cache_n this close to the user-turn start is attributed to its checkpoint")
    a = ap.parse_args(argv)
    unset = lib.unset_paths(("--llama-server", a.llama_server, lib.SERVER_ENV))
    if unset:
        print(unset, file=sys.stderr)
        return 2
    try:
        bodies = load_bodies(a.bodies)
    except (OSError, ValueError) as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return 2
    bodies = bodies[:a.limit] if a.limit else bodies
    if len(bodies) < 2:
        print("refusing to start: the warm readout needs at least two bodies", file=sys.stderr)
        return 2
    try:
        busy = lib.running_gpu_servers()
    except RuntimeError as e:
        print(f"refusing to start: {e}: cannot tell whether a model server holds the GPU", file=sys.stderr)
        return 2
    if busy:
        print(f"refusing to start: a model server holds the GPU (one at a time): {busy}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    pid_file = a.out / "run.pid"
    owner = pid_owner(pid_file)
    if owner is not None:
        if lib.is_bench_run(owner, BENCH_MODULE):
            print(f"refusing to start: another run (pid {owner}) owns this --out; stop it through its run.pid",
                  file=sys.stderr)
        else:
            print(f"refusing to start: run.pid names pid {owner}, which is not a run of this check: check it and "
                  f"remove run.pid by hand", file=sys.stderr)
        return 2
    previous = _install_stop_handlers()
    me = os.getpid()
    try:
        pid_file.write_text(f"{me}\n")
        return _run(a, bodies)
    finally:
        if _pid_in(pid_file) == me:
            pid_file.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
