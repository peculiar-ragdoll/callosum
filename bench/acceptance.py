"""callo v0.1 acceptance bench. Every request goes through callo; callo runs in launch mode on the
master build, in this process, so its log (probe, self-test, answer-slot check) is captured.

    uv run python -m bench.acceptance --out DIR [--rows a,b,c]

Rows and pass conditions are rendered into docs/bench/v0.1.md by bench/report.py. Raw per-row JSON goes under --out.

A row that measured something other than what it claims carries `invalid: <reason>`; the report renders it as
INVALID, never as PASS or "measured", and the run exits non-zero.
"""
from __future__ import annotations

import argparse
import http.client
import itertools
import json
import logging
import os
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import callo
from bench import lib
from callo.calibrate import request_from_record
from callo.cli import build_parser, serve, split_passthrough
from callo.upstream import find_orphans

MASTER = lib.env_path(lib.SERVER_ENV)                                       # the llama.cpp master build
BREW = "/opt/homebrew/bin/llama-server"
MODEL = lib.env_path(lib.MODELS_ENV, "local", "system1", "Qwen3.5-9B-Q8_0.gguf")
SUITE = lib.env_path(lib.DATA_ENV, "v4", "transfer-v4", "test.jsonl")       # Kev's transfer test split
CORPUS = "\n\n".join(p.read_text() for p in sorted((Path(__file__).resolve().parents[1] / "src" / "callo").glob("*.py")))
STAGE1_PINNED_B512_32K_S = 2.48          # the earlier System One study on b9890, raw /completion, pinned + -b 512 -ub 512 at 32k, p50
GEN_PROMPTS = [
    "Write a complete, well-commented Python implementation of a thread-safe LRU cache with TTL expiry, plus pytest tests.",
    "Explain, with code, how to implement a rate limiter using the token bucket algorithm in Go, including concurrency safety.",
    "Design a REST API for a library lending system: list the endpoints, request/response schemas, and error cases, then implement two of them in FastAPI.",
    "Write a detailed step-by-step tutorial for building a small Rust CLI that parses CSV files and prints summary statistics.",
]
DECODE_MAX_TOKENS = 128       # short generations, so several complete ENTIRELY inside the S1 window (the earlier study's method)
DECODE_MIN_INSIDE = 10        # background rows that must lie fully inside the S1 window for the hit to be measured
DECODE_S1_MIN_S = 120.0       # S1 records are cycled until the window is this long (a contended generation takes ~15 s)
FALLBACK_BLOCK_TOKENS = 4096  # long enough to still be running at dispatch; they are cancelled once the task lands
AGENT_SLOTS = {"1", "2", "3"}
FOREIGN_CACHED_MAX = 0.02     # --cache on: the share of the measured prompt the foreign prefill may take from cache
log = logging.getLogger("bench")
# The --cache multi-question row's two launches: (result key, --cache). "default" is the key earlier summaries and
# report.py read; it was serve's default then (cache off) and is now asked for explicitly with --no-cache.
CACHE_MULTI_MODES = (("default", False), ("cache", True))


def callo_serve_argv(*, model: str, llama_server: str, ctx: int, parallel: int, cache_dir: Path, flags=(),
                     cache: bool = False) -> list[str]:
    """The `callo serve` launch-mode argv every bench session runs. `cache` is always explicit (`--cache` or
    `--no-cache`, before `flags` and so before any `--` passthrough in them): serve's default went on in 0.3.0,
    and the benches measure with it OFF unless they ask, as every recorded result and resumable run was."""
    return ["serve", "--model", model, "--llama-server", llama_server, "--port", "0", "--ctx", str(ctx),
            "--parallel", str(parallel), "--cache-dir", str(cache_dir), "--cache" if cache else "--no-cache", *flags]


class Callo:
    """`callo serve` in a thread (launch mode), log captured, upstream URL and llama-server pid parsed from it.

    A failed start never leaks: everything after the thread starts is guarded, and `finish()` (also run on a
    failed start) reaps any llama-server whose pid file this instance's launch wrote into its cache dir."""

    def __init__(self, out: Path, *, llama_server: str = MASTER, ctx: int = 262144, parallel: int = 4, flags=(),
                 model: str = MODEL, cache: bool = False):
        self.lines: list[str] = []
        self.parallel = parallel
        self.cache_dir = out / "cache"
        self._preexisting = {pid for pid, _ in find_orphans(self.cache_dir)}   # not ours: never reaped
        h = logging.Handler()
        h.emit = lambda rec: self.lines.append(rec.getMessage())  # type: ignore[method-assign]
        logging.getLogger("callo").addHandler(h)
        logging.getLogger("callo").setLevel(logging.INFO)
        self._handler = h
        argv, extra = split_passthrough(callo_serve_argv(model=model, llama_server=llama_server, ctx=ctx,
                                                         parallel=parallel, cache_dir=self.cache_dir, flags=flags,
                                                         cache=cache))
        self.args = build_parser().parse_args(argv)
        self.args.extra = extra
        self.stop, self.port, self.rc = threading.Event(), None, None
        ready = threading.Event()

        def run():
            self.rc = serve(self.args, ready=lambda p: (setattr(self, "port", p), ready.set()), stop=self.stop)
            ready.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        try:
            ready.wait(timeout=1800)
            if not self.port:
                raise RuntimeError(f"callo did not start (rc={self.rc}):\n" + "\n".join(self.lines[-20:]))
            m = re.search(r"launched llama-server pid (\d+) on (http://\S+)", self.log_text())
            if m is None:
                raise RuntimeError("callo started but logged no 'launched llama-server pid' line")
            self.pid, self.upstream = int(m.group(1)), m.group(2)
            self.url = f"http://127.0.0.1:{self.port}"
            self.build_info = lib.get(f"{self.url}/props")["build_info"]
        except BaseException:
            self.finish()
            raise

    def log_text(self) -> str:
        return "\n".join(self.lines)

    def stats(self) -> dict:
        return lib.get(f"{self.url}/callo/stats")

    def finish(self) -> int | None:
        """Stop serve and reap this instance's own llama-server. The reap runs in a `finally`: a signal (the stop
        handlers' SystemExit) or a Ctrl-C landing during the join must not leave the child running."""
        self.stop.set()
        try:
            self.thread.join(timeout=120)
        finally:
            logging.getLogger("callo").removeHandler(self._handler)
            self._reap()
        return self.rc

    def _reap(self) -> None:
        """Normally a no-op: serve()'s `finally` stops its child. If serve is still stuck in startup (stop is only
        honoured after it), stop the process group of every llama-server pid file THIS instance's launch wrote."""
        reap_cache_dir(self.cache_dir, self._preexisting)


_pid_args = lib.pid_args          # the seam the tests stub: `ps -o args= -p PID`


def reap_cache_dir(cache_dir: Path, preexisting: set[int]) -> None:
    """`Callo._reap`'s rule, also for a launch whose `Callo` never reached its caller (a stop signal right after it
    returned): stop the process group of every live llama-server pid file in `cache_dir` that was not in `preexisting`
    (the pids `find_orphans` listed before the launch: never ours), and only while `ps -o args=` for the pid names
    llama-server (a pid reused after our llama-server died is logged and skipped,
    as is one whose command line cannot be read)."""
    for pid, port in find_orphans(cache_dir):
        if pid in preexisting:
            continue
        args = _pid_args(pid)
        if "llama-server" not in (args or ""):
            log.warning("not reaping pid %d (port %s): its pid file is ours but its command line is %s, not "
                        "llama-server", pid, port, repr(args) if args is not None else "unreadable")
            continue
        log.warning("reaping llama-server pid %d (port %s) left by this bench's callo", pid, port)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            if _wait_until(lambda pid=pid: not _alive(pid), timeout=15, interval=0.2):
                break


def _alive(pid: int) -> bool:
    """Running (a zombie, exited but not yet waited for by its Popen in this process, counts as gone)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(stat) and not stat.startswith("Z")


def _wait_until(pred: Callable[[], bool], timeout: float, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


class Ctx:
    def __init__(self, out: Path, suite: str, s1_limit: int, fallback_cycles: int = 8, model: str = MODEL,
                 cache: bool = False):
        self.out = out
        self.model = model          # the GGUF every row's callo launches (the brew row keeps MODEL)
        self.cache = cache          # --cache for those launches (cache_multi sets its own; brew stays off)
        clean = [r for r in map(json.loads, Path(suite).read_text().splitlines()) if r["_meta"]["variant"] == "clean"]
        self.records = lib.s1_subset(clean, s1_limit)
        self.all_records = clean
        self.fallback_cycles = fallback_cycles

    def tokenizer(self, c: Callo):
        return lambda text: len(lib.post(f"{c.url}/tokenize", {"content": text, "add_special": False})[2]["tokens"])

    def save(self, row: str, name: str, obj) -> None:
        d = self.out / row
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(json.dumps(obj, indent=1))


NO_THINKING = {"enable_thinking": False}   # Qwen3.5 thinks by default: short max_tokens would leave `content` empty


def _chat(content: str, **kw) -> dict:
    return {"model": "x", "messages": [{"role": "user", "content": content}], "chat_template_kwargs": NO_THINKING, **kw}


def _norm(resp: dict) -> dict:
    ch = resp["choices"][0]
    # llama-server gives every tool call a fresh random `id`, so two identical generations never share one: compare
    # the calls without it (type, name, arguments and order still must match).
    calls = ch["message"].get("tool_calls")
    calls = [{k: v for k, v in tc.items() if k != "id"} for tc in calls] if calls else calls
    return {"content": ch["message"].get("content"), "reasoning_content": ch["message"].get("reasoning_content"),
            "tool_calls": calls, "finish_reason": ch.get("finish_reason")}


def _stream_concat(events: list[dict]) -> dict:
    deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
    content = "".join(d.get("content") or "" for d in deltas)
    reasoning = "".join(d.get("reasoning_content") or "" for d in deltas)
    finish = [e["choices"][0].get("finish_reason") for e in events if e.get("choices") and e["choices"][0].get("finish_reason")]
    return {"content": content, "reasoning_content": reasoning, "finish_reason": finish[-1] if finish else None}


def _post_ok(url: str, payload: dict, **kw) -> tuple[dict, dict]:
    status, headers, body = lib.post(url, payload, **kw)
    if status != 200:
        raise RuntimeError(f"POST {url}: HTTP {status}: {str(body)[:300]}")
    return headers, body


def _slots_brief(c: Callo) -> list[dict]:
    """/slots without the (detokenised, possibly 140k-token) prompt and generated text."""
    return [{**{k: s.get(k) for k in ("id", "is_processing", "id_task", "n_prompt_tokens", "n_ctx")},
             "n_decoded": (s.get("next_token") or [{}])[0].get("n_decoded")} for s in lib.get(f"{c.upstream}/slots")]


def _busy(c: Callo) -> dict[int, bool]:
    return {s["id"]: bool(s["is_processing"]) for s in _slots_brief(c)}


def _invalid(reasons: list[str | None]) -> str | None:
    return "; ".join(r for r in reasons if r) or None


# ---- rows ---------------------------------------------------------------------------------------------

def row_selftest(c: Callo, ctx: Ctx, name: str = "selftest") -> dict:
    text = c.log_text()
    ok = "self-test ok: 2 + 2 = ? -> answer=4" in text and "answer slot check: answer slot confirmed" in text
    lines = [ln for ln in c.lines if "self-test" in ln or "answer slot" in ln or "probe" in ln]
    ctx.save(name, "log.json", lines)
    return {"pass": ok, "lines": lines}


def row_drop_in(c: Callo, ctx: Ctx, name: str = "drop_in") -> dict:
    body = _chat("Write a limerick about a proxy server.", max_tokens=64, temperature=0, seed=42, cache_prompt=False,
                 id_slot=1)
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                               "required": ["city"]}}}]
    tool_body = {**_chat("What is the weather in Paris right now? Use the tool.", max_tokens=96, temperature=0, seed=42,
                         cache_prompt=False, id_slot=1), "tools": tools}
    out = {}
    for label, b in (("plain", body), ("tools", tool_body)):
        h_via, via = _post_ok(f"{c.url}/v1/chat/completions", b)
        _, direct = _post_ok(f"{c.upstream}/v1/chat/completions", b)
        nv = _norm(via)
        # Non-vacuous: equal empty answers prove nothing. The tools case may legitimately answer with a tool call only.
        answered = bool(nv["content"]) or (label == "tools" and bool(nv["tool_calls"]))
        out[label] = {"via": nv, "direct": _norm(direct), "slot_header": h_via.get("x-callo-slot"),
                      "equal": nv == _norm(direct), "non_empty": answered}
    sv = _stream_concat(lib.sse(f"{c.url}/v1/chat/completions", {**body, "stream": True}))
    sd = _stream_concat(lib.sse(f"{c.upstream}/v1/chat/completions", {**body, "stream": True}))
    out["stream"] = {"via": sv, "direct": sd, "equal": sv == sd, "non_empty": bool(sv["content"])}
    out["pass"] = (all(out[k]["equal"] and out[k]["non_empty"] for k in ("plain", "tools", "stream"))
                   and out["plain"]["slot_header"] == "1")
    ctx.save(name, "drop_in.json", out)
    return out


def row_agent_cache(c: Callo, ctx: Ctx) -> dict:
    text, _ = lib.build_long_text(ctx.tokenizer(c), 1500, CORPUS)
    hdr = {"X-Callo-Conversation": "bench-cache"}
    msgs = [{"role": "user", "content": "Here is some code:\n" + text + "\nAcknowledge in one word."}]
    turns, s1 = [], []
    for i in range(3):
        h, resp = _post_ok(f"{c.url}/v1/chat/completions", {"model": "x", "messages": msgs, "max_tokens": 32,
                                                            "temperature": 0, "cache_prompt": True,
                                                            "chat_template_kwargs": NO_THINKING}, headers=hdr)
        t = resp["timings"]
        turns.append({"slot": h.get("x-callo-slot"), "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n"),
                      "predicted_n": t["predicted_n"]})
        msgs = msgs + [{"role": "assistant", "content": resp["choices"][0]["message"]["content"]},
                       {"role": "user", "content": f"Turn {i + 2}: name one function in the code, in one word."}]
        rows, _ = lib.run_s1(c.url, ctx.records[:1])         # a classification between every turn
        s1.append(rows[0])
    checks = []
    for prev, cur in zip(turns, turns[1:], strict=False):
        prev_total = prev["prompt_n"] + prev["predicted_n"] + (prev["cache_n"] or 0)
        checks.append({"cache_n_ok": (cur["cache_n"] or 0) >= 0.9 * prev_total,
                       "prompt_n_ok": cur["prompt_n"] <= 0.2 * prev_total, "prev_total": prev_total})
    slots = {t["slot"] for t in turns}
    out = {"turns": turns, "s1_between": s1, "checks": checks, "s1_between_ok": all(r["ok"] for r in s1),
           "pass": (all(ch["cache_n_ok"] and ch["prompt_n_ok"] for ch in checks) and len(slots) == 1
                    and slots <= AGENT_SLOTS                    # a missing X-Callo-Slot (None) fails here
                    and all(r["ok"] for r in s1))}
    ctx.save("agent_cache", "turns.json", out)
    return out


def row_endpoints(c: Callo, ctx: Ctx) -> dict:
    st_resp, h_resp, _ = lib.post(f"{c.url}/v1/responses", {"model": "x", "input": "Say hi.", "max_output_tokens": 8})
    slot = h_resp.get("x-callo-slot")
    held = next((s["n_prompt_tokens"] for s in _slots_brief(c) if str(s["id"]) == slot), None)
    st_msg, h_msg, body = lib.post(f"{c.url}/v1/messages", {"model": "x", "max_tokens": 8,
                                                            "messages": [{"role": "user", "content": "Say hi."}]})
    out = {"responses_status": st_resp, "responses_slot": slot, "responses_slot_held_tokens": held,
           "messages_status": st_msg, "messages_slot": h_msg.get("x-callo-slot"),
           "messages_status_type": body.get("type") if isinstance(body, dict) else None,
           "pass": (st_resp == 200 and st_msg == 200 and slot in AGENT_SLOTS and (held or 0) > 0
                    and h_msg.get("x-callo-slot") == "-")}
    ctx.save("endpoints", "endpoints.json", out)
    return out


def _prefill_payload(text: str):
    return lambda: _chat(f"nonce {uuid.uuid4()}\n{text}", max_tokens=1, cache_prompt=False)


def _under_load(c: Callo, ctx: Ctx, name: str, bg: lib.Background, *, warm_s: float, min_s1_s: float = 0.0) -> dict:
    mem = lib.MemorySampler(c.pid)
    mem.start()
    s1_rows: list[dict] = []
    probs: dict = {}
    try:
        bg.start()
        bg.ready.wait(timeout=120)
        time.sleep(warm_s)
        s1_rows, probs = lib.run_s1(c.url, ctx.records, min_duration_s=min_s1_s)
    finally:
        bg.stop()
        mem.stop()
        mem.join(timeout=30)
    summary = lib.summarize(s1_rows, bg.rows, mem.rows)
    summary["validity"] = lib.assess_validity(s1_rows, bg.rows, has_workload=True)
    summary["stats"] = c.stats()
    for fname, obj in (("s1.json", s1_rows), ("background.json", bg.rows), ("memory.json", mem.rows), ("probs.json", probs)):
        ctx.save(name, fname, obj)
    return summary


def row_s1_prefill(c: Callo, ctx: Ctx, n_tokens: int, name: str, pass_p50_s: float | None) -> dict:
    text, measured = lib.build_long_text(ctx.tokenizer(c), n_tokens, CORPUS)
    bg = lib.Background(f"{c.url}/v1/chat/completions", _prefill_payload(text), n=1)
    out = _under_load(c, ctx, name, bg, warm_s=2.0)
    out["prompt_tokens_target"], out["prompt_tokens_measured"] = n_tokens, measured
    bgs = out["background"]
    really = (bgs["prompt_n_min"] or 0) >= 0.95 * measured and (bgs["cache_n_max"] or 0) == 0
    out["really_prefills"] = really
    out["invalid"] = _invalid([out["validity"]["reason"],
                               None if really else f"prefill not proven: prompt_n_min {bgs['prompt_n_min']} vs "
                                                   f"{measured} measured tokens, cache_n_max {bgs['cache_n_max']}"])
    # every S1 request answered, and the gated p50 is over the S1 requests that really ran under the prefill
    out["s1_complete"] = out["s1"]["errors"] == 0 and out["s1"]["n"] == len(ctx.records)
    p50 = out["s1_overlap"].get("p50_ms")
    out["pass"] = (out["invalid"] is None and out["s1_complete"] and p50 is not None
                   and p50 <= 1000 * pass_p50_s * 1.25) if pass_p50_s else None
    out["pass_condition"] = (f"all {len(ctx.records)} S1 requests ok; p50 of those under the prefill <= "
                             f"{1.25 * pass_p50_s:.2f} s (1.25 x the earlier System One study's {pass_p50_s} s, on b9890, raw /completion)"
                             if pass_p50_s else "measured and reported")
    ctx.save(name, "summary.json", out)
    return out


def row_s1_solo(c: Callo, ctx: Ctx, name: str) -> dict:
    """The classification records alone on the main callo, nothing else running: the reference the rows under load
    are read against (their |Δp| from it, in the run summary). Run twice (`s1_solo`, `s1_solo_repeat`), the two
    passes' own |Δp| is the run-to-run floor."""
    s1_rows, probs = lib.run_s1(c.url, ctx.records)
    out = {"s1": lib.summarize(s1_rows, [], [])["s1"],
           "slots": dict(Counter(str(r.get("slot")) for r in s1_rows if r["ok"])), "stats": c.stats(),
           "n_records": len(ctx.records), "pass": None}
    errors = out["s1"]["errors"]
    out["invalid"] = f"{errors} S1 request(s) failed" if errors else None
    for fname, obj in (("s1.json", s1_rows), ("probs.json", probs), ("summary.json", out)):
        ctx.save(name, fname, obj)
    return out


def row_s1_decode(ctx: Ctx, out_dir: Path) -> dict:
    """>= 5 overlapping agent requests. Its own callo with --parallel 6 (1 reserved + 5 agent slots), so
    all 5 really decode at once; on the main --parallel 4 run 2 of them would sit deferred upstream. The hit uses
    only background rows that ran ENTIRELY inside the S1 window (the earlier study's method), so uncontended time before or
    after the S1 traffic cannot dilute it; generations are short so enough of them fit."""
    i = itertools.count()

    def payload():
        return _chat(GEN_PROMPTS[next(i) % len(GEN_PROMPTS)], max_tokens=DECODE_MAX_TOKENS, ignore_eos=True,
                     temperature=0.6, top_p=0.95, top_k=20, cache_prompt=False)

    c = None
    try:
        c = Callo(out_dir / "callo_decode", parallel=6, model=ctx.model, cache=ctx.cache)
        solo = lib.Background(f"{c.url}/v1/chat/completions", payload, n=5).start()
        try:
            time.sleep(150)
        finally:
            solo.stop()
        bg = lib.Background(f"{c.url}/v1/chat/completions", payload, n=5)
        out = _under_load(c, ctx, "s1_decode", bg, warm_s=5.0, min_s1_s=DECODE_S1_MIN_S)
    finally:
        if c is not None:
            c.finish()
    solo_ok = [r for r in solo.rows if not r.get("error")]
    out["solo_decode_tok_s"] = lib.summarize([], solo.rows, [])["background"]["decode_tok_s_mean"]
    with_s1 = out["background"]["decode_tok_s_mean_inside_s1"]
    out["decode_hit_frac"] = (1 - with_s1 / out["solo_decode_tok_s"]) if with_s1 and out["solo_decode_tok_s"] else None
    bg_ok = [r for r in bg.rows if not r.get("error")]
    want = [str(s) for s in range(1, 6)]
    out["slots_solo"] = sorted({str(r.get("slot")) for r in solo_ok})
    out["slots_with_s1"] = sorted({str(r.get("slot")) for r in bg_ok})
    out["slot_conflicts"] = lib.slot_conflicts(solo_ok) + lib.slot_conflicts(bg_ok)
    out["router_deferred"] = out["stats"]["router"]["counters"].get("deferred", 0)
    n_inside = out["background"]["n_inside_s1"]
    out["invalid"] = _invalid([
        out["validity"]["reason"],
        f"{len(solo.rows) - len(solo_ok)} solo request(s) errored" if len(solo_ok) != len(solo.rows) else None,
        f"only {n_inside} background rows ran entirely inside the S1 window (need >= {DECODE_MIN_INSIDE})"
        if n_inside < DECODE_MIN_INSIDE else None,
        f"the 5 requests did not sit in 5 distinct agent slots (solo {out['slots_solo']}, with S1 "
        f"{out['slots_with_s1']}, {out['slot_conflicts']} same-slot overlaps, {out['router_deferred']} deferred routes)"
        if (out["slot_conflicts"] or out["router_deferred"] or out["slots_solo"] != want
            or out["slots_with_s1"] != want) else None,
        f"{out['s1']['errors']} S1 request(s) failed" if out["s1"]["errors"] else None])
    out["concurrent_requests"], out["parallel"], out["max_tokens"] = 5, 6, DECODE_MAX_TOKENS
    out["s1_window_min_s"] = DECODE_S1_MIN_S
    out["pass"] = None
    ctx.save("s1_decode", "solo_background.json", solo.rows)
    ctx.save("s1_decode", "summary.json", out)
    return out


def _open_blocker(c: Callo, slot: int) -> http.client.HTTPConnection:
    """A long STREAMING generation pinned to `slot` through callo, left pending: closing the connection cancels it
    (callo's DisconnectWatcher shuts the upstream socket down; llama-server aborts when its next token write fails).
    Streaming on purpose: a NON-streaming task notices a closed connection only after 1 s with no result traffic
    on the whole server (server-queue.cpp:450-470 restarts its 1 s wait on every notification), which concurrent
    traffic, /slots polling included, can postpone indefinitely."""
    u = urlsplit(c.url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
    body = json.dumps(_chat(f"nonce {uuid.uuid4()}\nCount from 1 to 5000, one number per line.", id_slot=slot, stream=True,
                            max_tokens=FALLBACK_BLOCK_TOKENS, ignore_eos=True, temperature=0, cache_prompt=False)).encode()
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
    return conn


def _cancel(conn: http.client.HTTPConnection) -> None:
    try:
        if conn.sock is not None:
            conn.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    conn.close()


def _fallback_cycle(c: Callo, text: str, recs) -> dict:
    """One forced foreign-task episode. (1) Pin long streaming generations onto every agent slot 1..N-1 through
    callo and wait until each is DECODING (a slot still in prompt processing gets no batch share while the 32k
    prefill runs, so it would neither emit a token nor notice its cancellation). (2) While they run, send the unpinned 32k /v1/messages prefill: slot 0 is then llama-server's only idle slot
    (get_available_slot, server-context.cpp:1602-1690 at 4ebdf2c: LCP similarity > 0.1 among non-processing slots,
    else LRU among non-processing slots), so the foreign task lands in callo's reserved slot. (3) Cancel the
    blockers so fallback targets exist: they cannot simply be left to finish, because the 32k prefill takes nearly
    every 512-token batch and starves their decode until it is done. (4) Send classifications while /slots shows
    slot 0 still busy with the foreign prefill."""
    agent = list(range(1, c.parallel))
    conns = [_open_blocker(c, s) for s in agent]
    foreign: dict = {}
    ft = None
    try:
        def decoding() -> bool:
            snap = {s["id"]: s for s in _slots_brief(c)}
            return all(snap[s]["is_processing"] and (snap[s]["n_decoded"] or 0) >= 1 for s in agent)

        all_busy = _wait_until(decoding, timeout=30)

        def send_foreign() -> None:
            foreign["t_start"] = time.time()
            st, h, body = lib.post(f"{c.url}/v1/messages", {"model": "x", "max_tokens": 1, "messages": [
                {"role": "user", "content": f"nonce {uuid.uuid4()}\n{text}"}]})
            foreign.update({"t_end": time.time(), "status": st, "slot_header": h.get("x-callo-slot"),
                            **lib.response_timings(body)})

        ft = threading.Thread(target=send_foreign, daemon=True)
        idle_at_send = [s for s, b in _busy(c).items() if not b]
        ft.start()
        landed = _wait_until(lambda: bool(_busy(c).get(0)), timeout=15)
    finally:
        for conn in conns:
            _cancel(conn)
    freed = _wait_until(lambda: not any(_busy(c).get(s) for s in agent), timeout=30)
    time.sleep(0.3)                                  # callo's /slots snapshot may be up to 250 ms old
    samples, slot0_tokens = [], []
    while landed and freed and ft.is_alive():
        snap = {s["id"]: s for s in _slots_brief(c)}
        if not snap[0]["is_processing"]:
            break
        slot0_tokens.append(snap[0].get("n_prompt_tokens"))
        rec = next(recs)
        t_start, t0 = time.time(), time.perf_counter()
        st, h, resp = lib.post(f"{c.url}/v1/systemone", request_from_record(rec), timeout=600)
        samples.append({"id": rec["_meta"]["id"], "ok": st == 200, "status": st, "t_start": t_start, "t_end": time.time(),
                        "latency_ms": 1000 * (time.perf_counter() - t0), "slot": h.get("x-callo-slot"),
                        "foreign_at_send": True, **({} if st == 200 else {"error": str(resp)[:200]})})
    ft.join(timeout=3600)
    return {"blockers_all_busy": all_busy, "idle_at_foreign_send": idle_at_send, "foreign_landed_slot0": landed,
            "agent_slots_freed": freed, "foreign": foreign, "slot0_n_prompt_tokens_seen": slot0_tokens,
            "samples": samples}


def version_from_log(text: str) -> str | None:
    """The version callo's `callo <version> serving on ...` line names, or None."""
    m = re.search(r"^callo (\S+) serving on ", text, re.MULTILINE)
    return m.group(1) if m else None


def foreign_prefill_problem(f: dict, measured: int, *, cache: bool) -> str | None:
    """Why the foreign /v1/messages reply does not prove a real ~32k prefill in slot 0, or None. It needs status 200
    and prompt_n >= 95 % of the measured tokens. With --cache off no token may come from cache. With --cache on a
    shared prefix of up to FOREIGN_CACHED_MAX of the measured tokens may (each cycle's nonce follows a short
    template prefix the cache keeps, about 179 tokens on Tiel); a larger one means the prefill was not measured."""
    cached = f.get("cache_n") or 0
    allowed = FOREIGN_CACHED_MAX * measured if cache else 0
    if f.get("status") == 200 and (f.get("prompt_n") or 0) >= 0.95 * measured and cached <= allowed:
        return None
    return (f"foreign prefill not proven (status {f.get('status')}, prompt_n {f.get('prompt_n')} vs {measured}, "
            f"cache_n {f.get('cache_n')}" + (f", at most {allowed:.0f} cached with --cache on)" if cache else ")"))


def _fallback_mode(ctx: Ctx, out_dir: Path, mode: str) -> dict:
    c = None
    try:
        c = Callo(out_dir / f"callo_fallback_{mode}", flags=["--s1-fallback", mode], model=ctx.model,
                  cache=ctx.cache)
        text, measured = lib.build_long_text(ctx.tokenizer(c), 32768, CORPUS)
        recs = itertools.cycle(ctx.records)
        cycles = [_fallback_cycle(c, text, recs) for _ in range(ctx.fallback_cycles)]
        stats = c.stats()
    finally:
        if c is not None:
            c.finish()
    samples = [s for cy in cycles for s in cy["samples"]]
    lat = [s["latency_ms"] for s in samples if s["ok"]]
    durs = [cy["foreign"]["t_end"] - cy["foreign"]["t_start"] for cy in cycles if "t_end" in cy["foreign"]]
    r = {"cycles": len(cycles), "prompt_tokens_measured": measured, "n_samples": len(samples),
         "errors": sum(not s["ok"] for s in samples),
         "p50_ms": lib.percentile(lat, 0.5) if lat else None, "p95_ms": lib.percentile(lat, 0.95) if lat else None,
         "max_ms": max(lat) if lat else None, "foreign_prefill_s_median": statistics.median(durs) if durs else None,
         "sample_slots": dict(Counter(str(s["slot"]) for s in samples)),
         "s1_fallback_count": stats["router"]["counters"].get("s1_fallback", 0),
         "foreign_in_reserved": stats["counters"].get("foreign_in_reserved", 0),
         "foreign_observed": sum(1 for s in samples if s["foreign_at_send"]), "stats": stats}
    reasons = []
    for k, cy in enumerate(cycles):
        f = cy["foreign"]
        if not cy["blockers_all_busy"]:
            reasons.append(f"cycle {k}: the blockers did not hold every agent slot")
        if not cy["agent_slots_freed"]:
            reasons.append(f"cycle {k}: the cancelled blockers did not free the agent slots")
        if not cy["foreign_landed_slot0"]:
            reasons.append(f"cycle {k}: the /v1/messages prefill never showed up in slot 0")
        if (why := foreign_prefill_problem(f, measured, cache=ctx.cache)) is not None:
            reasons.append(f"cycle {k}: {why}")
        if not cy["samples"]:
            reasons.append(f"cycle {k}: no classification was sent while slot 0 held the foreign prefill")
        elif mode == "off" and cy["samples"][0]["slot"] != "0":
            reasons.append(f"cycle {k}: with fallback off the classification ran on slot {cy['samples'][0]['slot']}")
    if r["errors"]:
        reasons.append(f"{r['errors']} classification(s) failed")
    if mode == "on" and not (r["foreign_in_reserved"] > 0 and r["s1_fallback_count"] > 0
                             and any(s["slot"] not in ("0", None) for s in samples)):
        reasons.append(f"fallback never happened (foreign_in_reserved {r['foreign_in_reserved']}, s1_fallback "
                       f"{r['s1_fallback_count']}, sample slots {r['sample_slots']})")
    r["invalid"] = _invalid(reasons)
    ctx.save(f"s1_fallback_{mode}", "cycles.json", cycles)
    return r


def row_s1_fallback(ctx: Ctx, out_dir: Path) -> dict:
    """S1 latency while a foreign /v1/messages prefill holds the reserved slot, with --s1-fallback on vs off.
    The foreign task is FORCED into slot 0 every cycle (see `_fallback_cycle`). With fallback off callo never reads
    /slots, so its foreign_in_reserved / s1_fallback counters stay 0 by design; there the bench's own /slots check
    (`foreign_observed`) plus X-Callo-Slot 0 on every first sample prove the condition instead."""
    results = {mode: _fallback_mode(ctx, out_dir, mode) for mode in ("on", "off")}
    out = {"on": results["on"], "off": results["off"], "pass": None,
           "invalid": _invalid([f"{m}: {results[m]['invalid']}" if results[m]["invalid"] else None for m in ("on", "off")]),
           "p50_ms_on": results["on"]["p50_ms"], "p50_ms_off": results["off"]["p50_ms"]}
    ctx.save("s1_fallback", "summary.json", out)
    return out


def row_cache_multi(ctx: Ctx, out_dir: Path) -> dict:
    rec = ctx.records[0]
    extra = {"e1": {"type": "noul", "instructions": "Is the input about a scientific topic?"},
             "e2": {"type": "noul", "instructions": "Is the input written in English?"},
             "e3": {"type": "noul", "instructions": "Does the input mention a number?"},
             "e4": {"type": "noul", "instructions": "Is the input phrased as a question?"}}
    base = {"state": rec["state"], "questions": {**{k: {kk: v for kk, v in q.items() if kk in ("type", "instructions", "criteria")}
                                                     for k, q in rec["questions"].items()}, **extra}}
    results = {}
    for mode, cache in CACHE_MULTI_MODES:
        c = None
        try:
            c = Callo(out_dir / f"callo_{mode}", cache=cache, model=ctx.model)
            multi_lat, single_lat, multi_probs, single_probs = [], [], {}, {}
            for _ in range(3):
                t0 = time.perf_counter()
                _, resp = _post_ok(f"{c.url}/v1/systemone", base)
                multi_lat.append(time.perf_counter() - t0)
                multi_probs = {q: (a.get("probabilities") or {"noul": a["noul"]}) for q, a in resp["answers"].items()}
                t0 = time.perf_counter()
                for qid, q in base["questions"].items():
                    _, r1 = _post_ok(f"{c.url}/v1/systemone", {"state": base["state"], "questions": {qid: q}})
                    a = r1["answers"][qid]
                    single_probs[qid] = a.get("probabilities") or {"noul": a["noul"]}
                single_lat.append(time.perf_counter() - t0)
            results[mode] = {"multi_s_median": statistics.median(multi_lat), "five_single_s_median": statistics.median(single_lat),
                             "multi_probs": multi_probs, "single_probs": single_probs}
        finally:
            if c is not None:
                c.finish()

    def maxdp(a, b):
        return max(abs(a[q][k] - b[q][k]) for q in a for k in a[q])

    out = {**results, "max_dp_multi_vs_single_default": maxdp(results["default"]["multi_probs"], results["default"]["single_probs"]),
           "max_dp_multi_vs_single_cache": maxdp(results["cache"]["multi_probs"], results["cache"]["single_probs"]),
           "max_dp_cache_vs_default_multi": maxdp(results["cache"]["multi_probs"], results["default"]["multi_probs"]),
           "pass": None}
    ctx.save("cache_multi", "summary.json", out)
    return out


KV_LOG_RE = re.compile(r"exceed|purg|idle|halv|n_batch", re.I)


def kv_log_lines(log_path: Path, limit: int = 80) -> list[str]:
    """The purge / batch-retry / exceeded lines from the WHOLE llama-server log (a tail can miss the first purges,
    which happen minutes before the last request)."""
    if not log_path.exists():
        return []
    return [ln for ln in log_path.read_text(errors="replace").splitlines() if KV_LOG_RE.search(ln)][:limit]


def row_kv_capacity(ctx: Ctx, out_dir: Path) -> dict:
    """On its own fresh callo, so no earlier row's KV is resident: two conversations whose KV together exceeds -c."""
    c = None
    try:
        c = Callo(out_dir / "callo_kv", model=ctx.model, cache=ctx.cache)
        before = _slots_brief(c)
        text, measured = lib.build_long_text(ctx.tokenizer(c), 140000, CORPUS)
        out = {"per_conversation_tokens": measured, "ctx": 262144, "slots_before": before, "events": []}
        for conv, suffix in (("A", "Say ok."), ("B", "Say ok."), ("A", "Say ok again.")):
            st, h, body = lib.post(f"{c.url}/v1/chat/completions",
                                   _chat(f"conversation {conv}\n{text}\n{suffix}", max_tokens=8, cache_prompt=True),
                                   headers={"X-Callo-Conversation": conv})
            out["events"].append({"conv": conv if len(out["events"]) < 2 else "A-turn2", "status": st,
                                  "slot": h.get("x-callo-slot"),
                                  "timings": body.get("timings") if isinstance(body, dict) else None,
                                  "error": body if st != 200 else None, "slots_after": _slots_brief(c)})
        out["server_log_lines"] = kv_log_lines(Path(c.args.cache_dir) / f"llama-server-{c.upstream.rsplit(':', 1)[1]}.log")
        out["slots_after"] = _slots_brief(c)
    finally:
        if c is not None:
            c.finish()
    resident = sum(s.get("n_prompt_tokens") or 0 for s in before)
    out["resident_before"] = resident
    out["invalid"] = _invalid([
        f"{resident} tokens already resident before the first request" if resident > 2048 else None,
        f"2 x {measured} tokens does not exceed -c 262144" if 2 * measured <= 262144 else None])
    out["pass"] = None
    ctx.save("kv_capacity", "summary.json", out)
    return out


def row_brew(ctx: Ctx, out_dir: Path) -> dict:
    c = None
    try:
        c = Callo(out_dir / "callo_brew", llama_server=BREW, ctx=32768)
        out = {"build_info": c.build_info, "drop_in": row_drop_in(c, ctx, name="brew_drop_in"),
               "selftest": row_selftest(c, ctx, name="brew_selftest")}
        rows, _ = lib.run_s1(c.url, ctx.records[:10])
        lat = [r["latency_ms"] for r in rows if r["ok"]]
        out["s1_short"] = {"n_ok": len(lat), "p50_ms": lib.percentile(lat, 0.5) if lat else None}
        out["pass"] = out["drop_in"]["pass"] and out["selftest"]["pass"] and out["s1_short"]["n_ok"] == 10
    finally:
        if c is not None:
            c.finish()
    ctx.save("brew", "summary.json", out)
    return out


SOLO_ROWS = ["s1_solo", "s1_solo_repeat"]      # always first on the main callo, before anything has loaded it
MAIN_ROWS = [*SOLO_ROWS, "selftest", "drop_in", "endpoints", "agent_cache", "s1_prefill_32k", "s1_prefill_128k"]
STANDALONE = {"s1_decode": row_s1_decode, "kv_capacity": row_kv_capacity, "s1_fallback": row_s1_fallback,
              "cache_multi": row_cache_multi, "brew": row_brew}
ROWS = [*MAIN_ROWS, *STANDALONE]


def _main_row(name: str, c: Callo, ctx: Ctx) -> dict:
    if name in SOLO_ROWS:
        return row_s1_solo(c, ctx, name)
    if name == "selftest":
        return row_selftest(c, ctx)
    if name == "drop_in":
        return row_drop_in(c, ctx)
    if name == "endpoints":
        return row_endpoints(c, ctx)
    if name == "agent_cache":
        return row_agent_cache(c, ctx)
    if name == "s1_prefill_32k":
        # the 2.48 s gate was measured on MODEL; any other model's 32k row is measured and reported, ungated
        return row_s1_prefill(c, ctx, 32768, "s1_prefill_32k", STAGE1_PINNED_B512_32K_S if ctx.model == MODEL else None)
    return row_s1_prefill(c, ctx, 131072, "s1_prefill_128k", None)


def delta_p_report(out: Path, rows: list[str], results: dict[str, dict]) -> tuple[dict, dict | None]:
    """(|Δp| of each row's probs.json from s1_solo's, the s1_solo vs s1_solo_repeat floor or None). A row counts
    only when `results` (the summary's rows) holds it and it did not fail, s1_solo included: a failed or invalid
    row's answers are not a reading. A probs.json that cannot be read or compared is that row's `{"error": ...}`,
    never a crash, so the run still finishes. Empty when s1_solo has no usable result."""
    def usable(name: str) -> bool:
        return name in results and not row_failed(results[name]) and (out / name / "probs.json").exists()

    def load(name: str):
        return json.loads((out / name / "probs.json").read_text())

    if not usable("s1_solo"):
        return {}, None
    try:
        solo = load("s1_solo")
    except Exception as e:  # noqa: BLE001 -- a bad file is reported, not raised
        return {"s1_solo": {"error": f"{type(e).__name__}: {e}"}}, None
    vs: dict[str, dict] = {}
    for name in rows:
        if name == "s1_solo" or not usable(name):
            continue
        try:
            vs[name] = lib.delta_p(solo, load(name))
        except Exception as e:  # noqa: BLE001 -- e.g. truncated JSON, or a question with a different option count
            vs[name] = {"error": f"{type(e).__name__}: {e}"}
    return vs, vs.get("s1_solo_repeat")


def row_failed(res: dict) -> bool:
    return res.get("pass") is False or bool(res.get("error")) or bool(res.get("invalid"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(ROWS))
    ap.add_argument("--suite-file", default=SUITE)
    ap.add_argument("--model", default=MODEL, help="the GGUF every row's callo launches (the brew row keeps the default)")
    ap.add_argument("--cache", choices=("on", "off"), default="off",
                    help="callo serve's --cache for the row launches: off (the default) as every recorded run "
                         "measured; on is serve's own default")
    ap.add_argument("--s1-limit", type=int, default=100)
    ap.add_argument("--fallback-cycles", type=int, default=8, help="forced foreign-prefill episodes per fallback mode")
    ap.add_argument("--min-free-gb", type=float, default=30.0,
                    help="refuse to start below this much free memory (vm_stat free + inactive + speculative)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rows = [r for r in a.rows.split(",") if r]
    unknown = [r for r in rows if r not in ROWS]
    if unknown or not rows:
        print(f"unknown row(s) {unknown}; choose from {','.join(ROWS)}")
        return 2
    unset = lib.unset_paths(("--suite-file", a.suite_file, lib.DATA_ENV), ("--model", a.model, lib.MODELS_ENV),
                            ("", MASTER, lib.SERVER_ENV), *([("", MODEL, lib.MODELS_ENV)] if "brew" in rows else []))
    if unset:
        print(unset)
        return 2
    free = lib.free_memory_bytes()
    if a.min_free_gb > 0 and (free is None or free < a.min_free_gb * 1024**3):
        have = "unknown" if free is None else f"{free / 1024**3:.1f} GB"
        print(f"refusing to start: {have} free memory (vm_stat free + inactive + speculative) < --min-free-gb {a.min_free_gb}")
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(out, a.suite_file, a.s1_limit, a.fallback_cycles, model=a.model, cache=a.cache == "on")
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"rows": {}}
    # a resumed --out must keep its settings: rows of two models or cache modes are not one run, and |Δp| would
    # compare them (a summary from before these flags existed was the default model, cache off)
    was = {"model": summary.get("model", MODEL), "cache": summary.get("cache", "off")}
    if summary_path.exists() and was != {"model": a.model, "cache": a.cache}:
        print(f"refusing to resume {out}: it was run with --model {was['model']} --cache {was['cache']}, this run "
              f"asks for --model {a.model} --cache {a.cache}; use another --out")
        return 2
    # nor its callo: a summary from before callo_version was recorded is adopted, and records this version below
    if summary.get("callo_version", callo.__version__) != callo.__version__:
        print(f"refusing to resume {out}: it was run with callo {summary['callo_version']}, this is callo "
              f"{callo.__version__}; use another --out")
        return 2
    summary["model"], summary["started"] = a.model, summary.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["free_memory_at_start"], summary["cache"] = free, a.cache
    # the main callo's serving line when it runs (below); otherwise the in-process callo every row's launch runs
    summary["callo_version"] = callo.__version__
    ran: dict[str, dict] = {}

    def record(name: str, res: dict) -> None:
        ran[name] = summary["rows"][name] = res
        summary_path.write_text(json.dumps(summary, indent=1, default=str))

    main_rows = [r for r in SOLO_ROWS if r in rows] + [r for r in rows if r in MAIN_ROWS and r not in SOLO_ROWS]
    if main_rows:
        c = None
        try:
            c = Callo(out / "callo_main", model=ctx.model, cache=ctx.cache)
            summary["master_build_info"] = c.build_info
            summary["callo_version"] = version_from_log(c.log_text()) or callo.__version__
            for name in main_rows:
                log.info("row %s", name)
                try:
                    res = _main_row(name, c, ctx)
                except Exception as e:  # noqa: BLE001 -- one row's failure must not lose the others
                    log.exception("row %s failed", name)
                    res = {"pass": False, "error": f"{type(e).__name__}: {e}"}
                record(name, res)
        except Exception as e:  # noqa: BLE001 -- the main callo did not start: its rows fail, the others still run
            log.exception("main callo failed")
            for name in main_rows:
                if name not in ran:
                    record(name, {"pass": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            if c is not None:
                summary["callo_main_log"] = c.lines
                c.finish()
    for name, fn in STANDALONE.items():
        if name in rows:
            log.info("row %s", name)
            try:
                res = fn(ctx, out)
            except Exception as e:  # noqa: BLE001
                log.exception("row %s failed", name)
                res = {"pass": False, "error": f"{type(e).__name__}: {e}"}
            record(name, res)
    summary["delta_p_vs_solo"], floor = delta_p_report(out, ROWS, summary["rows"])
    summary.pop("delta_p_floor", None)
    if floor is not None:
        summary["delta_p_floor"] = floor
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary_path.write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: ("INVALID" if v.get("invalid") else v.get("pass")) for k, v in ran.items()}, indent=1))
    return 1 if any(row_failed(v) for v in ran.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
