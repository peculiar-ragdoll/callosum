# Adapted from the author's benchmark harness (harness/system1/concurrency.py @ f886c50f).
# Edits: only parse_footprint and the _FOOT/_UNITS constants are copied (unit powers reformatted) and s1_subset condensed (docstring and the limit < 1 check dropped); everything else is re-implemented from that module's ideas; percentile is imported from callo.proxy. The v0.2 section at the end is new: g0_parity as a decision-equivalence gate (g0_parity + g0_parity_aggregate), plus top2_logodds_shift and kv_sensitivity_summary for the KV-layout sensitivity row (parity diagnostic). v0.3: gpu_class no longer counts a llama.cpp-named script run by a Python interpreter (the test suite's fake llama-server) as a GPU server (_is_python split out of _program); the GPU rule (gpu_class, the footprint line, the residency probe, the ps listing) is callo.menu's, and running_gpu_servers fails closed (a ps that fails raises); is_bench_run and pid_args (the run.pid rule the think and transcription benches share).
"""Plumbing for the acceptance bench. Re-implemented from the ideas in that harness's concurrency.py (workload
loop, overlap + validity gate, memory sampler); nothing is imported from it.

Every request goes through callo. A condition is VALID only if its background workload completed without errors
and >= 90% of the S1 requests overlapped a background request; otherwise its numbers measured something else.
"""
from __future__ import annotations

import http.client
import itertools
import json
import math
import os
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from callo import menu as _menu
from callo.calibrate import request_from_record
from callo.proxy import percentile  # one nearest-rank percentile for callo's /callo/stats and the bench

# ---- where the user's own data, models and results live (bench/README.md) ---------------------------------------
# The acceptance benches ship no data and no models. Their default paths come from these variables; a bench whose
# path is neither set here nor given as a flag refuses at startup (`unset_paths`).
DATA_ENV = "CALLO_BENCH_DATA"                 # a checkout of Kev's `evals/` directory (the transfer set, the pools)
MODELS_ENV = "CALLO_BENCH_MODELS"             # the GGUF root; each bench names its files relative to it
OUT_ENV = "CALLO_BENCH_OUT"                   # the raw-results root; each bench writes under its own subdirectory
SERVER_ENV = "CALLO_BENCH_LLAMA_SERVER"       # the llama-server binary the benches launch


def env_path(var: str, *parts: str) -> str | None:
    """`$var` joined with `parts`, or None when `var` is unset or empty (never a guessed location)."""
    root = os.environ.get(var, "")
    return str(Path(root, *parts)) if root else None


def unset_paths(*needed: tuple[str, str | None, str]) -> str | None:
    """The startup refusal for required paths nobody set, or None. Each of `needed` is (flag, value, env var); the
    flag is "" for a path the bench takes only from its environment variable."""
    missing = [f"pass {flag} or set {var}" if flag else f"set {var}" for flag, value, var in needed if not value]
    if not missing:
        return None
    return f"refusing to start: a required path is not set ({'; '.join(dict.fromkeys(missing))}; see bench/README.md)"

_FOOT = re.compile(r"Footprint:\s+([\d.]+)\s+(B|KB|MB|GB)\b")
_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
_VM_PAGE = re.compile(r"page size of (\d+) bytes")
_VM_ROW = re.compile(r"^Pages (free|inactive|speculative):\s+(\d+)\.?\s*$", re.M)


def parse_footprint(text: str) -> int | None:
    m = _FOOT.search(text)
    return int(float(m.group(1)) * _UNITS[m.group(2)]) if m else None


def parse_vm_stat(text: str) -> int | None:
    """Reclaimable memory in bytes from macOS `vm_stat`: (free + inactive + speculative) pages x page size."""
    page = _VM_PAGE.search(text)
    rows = dict(_VM_ROW.findall(text))
    if not page or set(rows) != {"free", "inactive", "speculative"}:
        return None
    return sum(int(v) for v in rows.values()) * int(page.group(1))


def free_memory_bytes() -> int | None:
    try:
        return parse_vm_stat(subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout)
    except (OSError, subprocess.SubprocessError):
        return None


def s1_subset(records: Sequence[Any], limit: int) -> list[Any]:
    n = len(records)
    return list(records) if limit >= n else [records[i * n // limit] for i in range(limit)]


def post(url: str, payload: dict, *, timeout: float = 3600, headers: dict | None = None) -> tuple[int, dict, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 -- local servers only
            return r.status, {k.lower(): v for k, v in r.headers.items()}, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            body = json.loads(raw)
        except ValueError:
            body = raw.decode(errors="replace")
        return e.code, {k.lower(): v for k, v in e.headers.items()}, body


def get(url: str, timeout: float = 60) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
        return json.loads(r.read())


def sse(url: str, payload: dict, *, headers: dict | None = None, timeout: float = 3600) -> list[dict]:
    """POST a streaming request and return the parsed `data:` events (DONE excluded)."""
    u = urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    body = json.dumps(payload).encode()
    conn.request("POST", u.path, body=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body)),
                                                     **(headers or {})})
    resp = conn.getresponse()
    events = []
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                break
            events.append(json.loads(data))
    conn.close()
    return events


def probs_of(record: dict, answers: dict) -> dict[str, list[float]]:
    out = {}
    for qid, q in record["questions"].items():
        a = answers[qid]
        out[f"{record['_meta']['id']}/{qid}"] = ([1 - a["noul"], a["noul"]] if q["type"] == "noul"
                                                  else list(a["probabilities"].values()))
    return out


def _argmax(p: Sequence[float]) -> int:
    return max(range(len(p)), key=p.__getitem__)       # the first index on a tie


def delta_p(a: dict[str, list[float]], b: dict[str, list[float]]) -> dict:
    """How far run `b`'s readouts moved from run `a`'s, over the questions (keys) both answered: per question the
    largest |Δp| over its options, then the max and mean of those (`n` questions), and `flips`, how many questions
    changed their most likely option. Keys only one run answered are ignored."""
    common = [k for k in a if k in b]
    ds = [max(abs(x - y) for x, y in zip(a[k], b[k], strict=True)) for k in common]
    return {"n": len(ds), "max": max(ds) if ds else None, "mean": sum(ds) / len(ds) if ds else None,
            "flips": sum(_argmax(a[k]) != _argmax(b[k]) for k in common)}


def _ok_bg(bg: list[dict]) -> list[dict]:
    return [r for r in bg if not r.get("error") and r.get("t_start") is not None and r.get("t_end") is not None]


def _ok_s1(s1: list[dict]) -> list[dict]:
    return [r for r in s1 if r.get("ok") and r.get("t_start") is not None and r.get("t_end") is not None]


def _busy_intervals(bg: list[dict], gap_s: float) -> list[list[float]]:
    """Successful background requests merged into busy intervals (a re-issue within `gap_s` continues one)."""
    merged: list[list[float]] = []
    for r in sorted(_ok_bg(bg), key=lambda r: r["t_start"]):
        if merged and r["t_start"] - merged[-1][1] <= gap_s:
            merged[-1][1] = max(merged[-1][1], r["t_end"])
        else:
            merged.append([r["t_start"], r["t_end"]])
    return merged


def overlapping_s1(s1: list[dict], bg: list[dict], gap_s: float = 0.5) -> list[dict]:
    """The OK S1 rows that ran entirely inside a background busy interval: the only ones measured under load."""
    merged = _busy_intervals(bg, gap_s)
    return [q for q in _ok_s1(s1) if any(lo <= q["t_start"] and q["t_end"] <= hi for lo, hi in merged)]


def overlap_fraction(s1: list[dict], bg: list[dict], gap_s: float = 0.5) -> float | None:
    reqs = _ok_s1(s1)
    if not reqs:
        return None
    return len(overlapping_s1(s1, bg, gap_s)) / len(reqs)


def s1_window(s1: list[dict]) -> tuple[float, float] | None:
    reqs = _ok_s1(s1)
    return (min(r["t_start"] for r in reqs), max(r["t_end"] for r in reqs)) if reqs else None


def inside_s1_window(bg: list[dict], s1: list[dict]) -> list[dict]:
    """Successful background rows that ran ENTIRELY inside the S1 window (first S1 start .. last S1 end): their
    tok/s is purely "with S1 traffic present", undiluted by uncontended time before or after it."""
    w = s1_window(s1)
    return [] if w is None else [r for r in _ok_bg(bg) if w[0] <= r["t_start"] and r["t_end"] <= w[1]]


def slot_conflicts(rows: list[dict], min_overlap_s: float = 0.5) -> int:
    """Pairs of rows on the same `slot` that overlap by more than `min_overlap_s`: > 0 means requests shared (queued
    on) a slot. The tolerance ignores the ~1 ms handoff where a worker's next request starts while the client-side
    end of the previous one on that slot is still being recorded."""
    rs = sorted((r for r in rows if r.get("slot") is not None), key=lambda r: r["t_start"])
    return sum(1 for i, a in enumerate(rs) for b in rs[i + 1:]
               if a["slot"] == b["slot"] and min(a["t_end"], b["t_end"]) - b["t_start"] > min_overlap_s)


def assess_validity(s1: list[dict], bg: list[dict], *, has_workload: bool, min_overlap: float = 0.9) -> dict:
    errors = sum(1 for r in bg if r.get("error"))
    ov = overlap_fraction(s1, bg) if has_workload else None
    reason = None
    if has_workload and not bg:
        reason = "no background requests completed"
    elif has_workload and errors:
        reason = f"{errors} background request(s) errored"
    elif s1 and not any(r.get("ok") for r in s1):
        reason = "all S1 requests failed"
    elif has_workload and s1 and (ov is None or ov < min_overlap):
        reason = f"only {0.0 if ov is None else ov:.0%} of S1 requests overlapped a background request (need {min_overlap:.0%})"
    return {"valid": reason is None, "reason": reason, "bg_errors": errors, "overlap_frac": ov}


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summarize(s1: list[dict], bg: list[dict], mem: list[dict]) -> dict:
    lat = [r["latency_ms"] for r in s1 if r.get("ok")]
    s1_out: dict[str, Any] = {"n": len(lat), "errors": sum(not r.get("ok") for r in s1)}
    if lat:
        s1_out |= {"p50_ms": percentile(lat, 0.5), "p95_ms": percentile(lat, 0.95), "max_ms": max(lat)}
    ov = [r["latency_ms"] for r in overlapping_s1(s1, bg)]
    s1_overlap: dict[str, Any] = {"n": len(ov)}
    if ov:
        s1_overlap |= {"p50_ms": percentile(ov, 0.5), "p95_ms": percentile(ov, 0.95), "max_ms": max(ov)}
    ok = _ok_bg(bg)
    win = inside_s1_window(bg, s1)
    background = {"n": len(bg), "errors": len(bg) - len(ok),
                  "decode_tok_s_mean": _mean([r["predicted_per_second"] for r in ok if r.get("predicted_per_second")]),
                  "prefill_tok_s_mean": _mean([r["prompt_per_second"] for r in ok if r.get("prompt_per_second")]),
                  "prompt_n_min": min((r["prompt_n"] for r in ok if r.get("prompt_n") is not None), default=None),
                  "cache_n_max": max((r["cache_n"] for r in ok if r.get("cache_n") is not None), default=None),
                  "n_inside_s1": len(win),
                  "decode_tok_s_mean_inside_s1": _mean([r["predicted_per_second"] for r in win
                                                        if r.get("predicted_per_second")]),
                  "slots_inside_s1": sorted({str(r["slot"]) for r in win if r.get("slot") is not None})}
    fp = [m["footprint"] for m in mem if m.get("footprint")]
    rss = [m["rss"] for m in mem if m.get("rss")]
    return {"s1": s1_out, "s1_overlap": s1_overlap, "background": background,
            "memory": {"footprint_max": max(fp) if fp else None, "rss_max": max(rss) if rss else None}}


class MemorySampler(threading.Thread):
    """`footprint -p PID` (macOS) and `ps rss` every `interval` seconds until stopped."""

    def __init__(self, pid: int, interval: float = 2.0):
        super().__init__(daemon=True)
        # `_halt`, not `_stop`: Thread has its own `_stop()` on py3.10-3.12, and shadowing it breaks `join()`
        self.pid, self.interval, self.rows, self._halt = pid, interval, [], threading.Event()

    @staticmethod
    def _run(cmd: list[str]) -> str:
        # `footprint` is macOS-only; on a host without it (Linux CI) subprocess raises FileNotFoundError,
        # which would crash the sampler thread. Tolerate a missing binary and fall back to the other reading.
        try:
            return subprocess.run(cmd, capture_output=True, text=True).stdout
        except (FileNotFoundError, OSError):
            return ""

    def run(self) -> None:
        while not self._halt.is_set():
            fp = self._run(["footprint", "-p", str(self.pid)])
            rss = self._run(["ps", "-o", "rss=", "-p", str(self.pid)]).strip()
            self.rows.append({"t": time.time(), "footprint": parse_footprint(fp), "rss": int(rss) * 1024 if rss else None})
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()


class Background:
    """`n` workers POSTing `payload_fn()` to `url` back to back until `stop()` (or `max_requests` each). Each row
    records t_start/t_end, the `X-Callo-Slot` it ran on and the response `timings` (prompt_n, cache_n,
    predicted_per_second, prompt_per_second) so the raw log PROVES a prefill really happened. An Anthropic
    `/v1/messages` reply has no `timings`; its `usage` gives prompt_n (input + cache_read) and cache_n (cache_read).
    `ready` is set just before the first POST."""

    def __init__(self, url: str, payload_fn: Callable[[], dict], *, n: int = 1, max_requests: int | None = None,
                 headers: dict | None = None):
        self.url, self.payload_fn, self.n, self.max_requests, self.headers = url, payload_fn, n, max_requests, headers
        self.rows: list[dict] = []
        self.ready, self._stop = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._threads = [threading.Thread(target=self._loop, daemon=True) for _ in range(n)]

    def _loop(self) -> None:
        i = 0
        while not self._stop.is_set() and (self.max_requests is None or i < self.max_requests):
            i += 1
            t0 = time.time()
            self.ready.set()
            try:
                status, hdrs, body = post(self.url, self.payload_fn(), headers=self.headers)
                row = {"t_start": t0, "t_end": time.time(), "status": status, "slot": hdrs.get("x-callo-slot"),
                       **response_timings(body)}
                if status != 200:
                    row["error"] = f"HTTP {status}: {str(body)[:200]}"
            except Exception as e:  # noqa: BLE001 -- recorded; the validity gate counts it
                row = {"t_start": t0, "t_end": time.time(), "error": str(e)}
                self._stop.wait(1.0)
            with self._lock:
                self.rows.append(row)

    def start(self) -> Background:
        for t in self._threads:
            t.start()
        return self

    def stop(self, timeout: float = 7200) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)

    def join(self, timeout: float = 7200) -> None:
        for t in self._threads:
            t.join(timeout=timeout)


def response_timings(body: Any) -> dict:
    """prompt_n / cache_n / predicted_n / tok/s from a llama-server reply: `timings` (OpenAI-style and native
    endpoints) or, for `/v1/messages`, the Anthropic `usage` block (which carries no rates)."""
    if not isinstance(body, dict):
        return {}
    t = body.get("timings")
    if isinstance(t, dict):
        return {"prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "predicted_n": t.get("predicted_n"),
                "predicted_per_second": t.get("predicted_per_second"), "prompt_per_second": t.get("prompt_per_second")}
    u = body.get("usage")
    if isinstance(u, dict) and "input_tokens" in u:
        cached = u.get("cache_read_input_tokens") or 0
        return {"prompt_n": u["input_tokens"] + cached, "cache_n": cached, "predicted_n": u.get("output_tokens")}
    return {}


def build_long_text(count_tokens: Callable[[str], int], n_tokens: int, corpus: str) -> tuple[str, int]:
    """Text of `n_tokens`..`n_tokens*1.03` tokens by the given tokenizer: repeat the corpus, measure, trim, re-measure."""
    base = count_tokens(corpus)
    reps = n_tokens // base + 2
    text = (corpus + "\n\n") * reps
    chars_per_token = len(text) / count_tokens(text)
    for _ in range(6):
        measured = count_tokens(text)
        if n_tokens <= measured <= int(n_tokens * 1.03):
            return text, measured
        target_chars = int(len(text) * (n_tokens * 1.015) / measured) if measured > n_tokens * 1.03 else len(text) + int(chars_per_token * (n_tokens - measured + 8))
        text = (text if target_chars <= len(text) else (corpus + "\n\n") * (reps + 2))[:target_chars]
    return text, count_tokens(text)


def run_s1(url: str, records: Sequence[dict], *, timeout: float = 600,
           min_duration_s: float = 0.0) -> tuple[list[dict], dict[str, list[float]]]:
    """Each record once, in order; then, if `min_duration_s` has not elapsed yet, the records again (cycled) until it
    has, so a workload can be sampled over a window longer than one pass."""
    rows, probs = [], {}
    deadline = time.monotonic() + min_duration_s
    passes = itertools.chain(records, itertools.takewhile(lambda _: time.monotonic() < deadline, itertools.cycle(records)))
    for rec in passes:
        t_start, t0 = time.time(), time.perf_counter()
        try:
            status, hdrs, resp = post(f"{url}/v1/systemone", request_from_record(rec), timeout=timeout)
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {str(resp)[:200]}")
            rows.append({"id": rec["_meta"]["id"], "t_start": t_start, "t_end": time.time(),
                         "latency_ms": 1000 * (time.perf_counter() - t0), "ok": True, "slot": hdrs.get("x-callo-slot")})
            probs.update(probs_of(rec, resp["answers"]))
        except Exception as e:  # noqa: BLE001
            rows.append({"id": rec["_meta"]["id"], "t_start": t_start, "t_end": time.time(), "ok": False, "error": str(e)})
    return rows, probs


# ---- v0.2 acceptance bench plumbing ----------------------------------------------------------------

# The GPU rule is callo's own (`callo.menu`: one classifier for the menu's one-server rule and the benches' preflights;
# the bench may import callo, callo never imports the bench). llama.cpp's servers and tools ALWAYS block. oMLX, MLX and
# ollama block only while they hold a model, resident >= RESIDENT_MIN_BYTES: the oMLX menu-bar app (about 90 MB) and an
# idle `ollama serve` (about 70 MB) are always running here, and a plain name match would refuse every run. A
# llama.cpp-named script run by a Python interpreter (the test suite's fake llama-server) is never a GPU server.
LLAMA_PROGRAMS, RESIDENT_ROOTS, OLLAMA_SERVERS = _menu.LLAMA_PROGRAMS, _menu.RESIDENT_ROOTS, _menu.OLLAMA_SERVERS
RESIDENT_MIN_BYTES = _menu.RESIDENT_MIN_BYTES
gpu_class = _menu.gpu_class
footprint_of_pid = _menu.footprint_bytes


def process_resident_bytes(pid: int) -> int | None:
    """macOS `footprint -p PID` (its own line), falling back to RSS from `ps -o rss= -p PID`; None when neither
    answers (callo.menu's, run through this module's `subprocess.run`)."""
    return _menu.process_resident_bytes(pid, run=subprocess.run)


def gpu_server_pids(ps_text: str, *, resident_bytes: Callable[[int], int | None] | None = None,
                    min_resident: int = RESIDENT_MIN_BYTES) -> list[tuple[int, str]]:
    """(pid, command line) of every process from `ps -axo pid=,args=` that holds or may hold the GPU: every llama.cpp
    program, and every oMLX / MLX / ollama-server process whose residency (`resident_bytes(pid)`, by default
    `process_resident_bytes`) is >= `min_resident` or cannot be measured (unknown counts as busy). Detection only: the
    bench never signals a process it did not start."""
    measure = resident_bytes or process_resident_bytes
    out = []
    for line in ps_text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, kind = int(parts[0]), gpu_class(parts[1].split())
        if kind == "always":
            out.append((pid, parts[1]))
        elif kind == "resident":
            b = measure(pid)
            if b is None or b >= min_resident:
                out.append((pid, parts[1] if b is None else f"{parts[1]} [resident {b / 1024**3:.1f} GiB]"))
    return out


def running_gpu_servers() -> list[tuple[int, str]]:
    """The model servers on this box (`gpu_server_pids` over `callo.menu.ps_listing`). Fails CLOSED: a ps that is
    missing, errors, exits non-zero, times out or lists nothing is a RuntimeError ("could not list processes ..."),
    never an empty list, as the menu's own rule refuses."""
    text, why = _menu.ps_listing()
    if text is None:
        raise RuntimeError(_menu.ps_failed_reason(why))
    return gpu_server_pids(text)


def pid_args(pid: int) -> str | None:
    """Seam: `ps -o args= -p PID` (None when it cannot be read)."""
    try:
        p = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def is_bench_run(pid: int, module: str, *, pid_args: Callable[[int], str | None] = pid_args,
                 kill: Callable[[int, int], None] = os.kill) -> bool:
    """The think bench's run.pid rule: the live pid is another run of the bench `module`, because its command line
    names it. Another user's process (a PermissionError on signal 0) never is, nor one whose command line cannot be
    read, so a refusal never suggests signalling an unknown pid. `kill` sends only signal 0 (a liveness check, never
    delivered); `pid_args` and `kill` are seams."""
    try:
        kill(pid, 0)
    except (OSError, OverflowError):
        return False
    return module in (pid_args(pid) or "")


def paced_ticks(send: Callable[[int], dict], *, hz: float, duration_s: float) -> list[dict]:
    """bench2-style pacing: tick k is due at t0 + k/hz, at most one request is in flight, and a request that returns
    late is followed by the LATEST tick already due (no backlog). Each row is `send(k)` plus `k`, `t_due`, `t_start`,
    `t_end` (wall clock) and `latency_ms`. Nothing is sent once `duration_s` has passed."""
    interval = 1.0 / hz
    t0, wall0 = time.monotonic(), time.time()
    rows: list[dict] = []
    k = 0
    while True:
        now = time.monotonic()
        if now - t0 >= duration_s:
            return rows
        due = int((now - t0) / interval)
        if due < k:
            time.sleep(max(0.0, min(t0 + k * interval, t0 + duration_s) - now))
            continue
        k = due
        t_start, p0 = time.time(), time.perf_counter()
        row = dict(send(k))
        row.update(k=k, t_due=wall0 + k * interval, t_start=t_start, t_end=time.time(),
                   latency_ms=1000 * (time.perf_counter() - p0))
        rows.append(row)
        k += 1


def _join_reasons(reasons: Sequence[str | None]) -> str | None:
    return "; ".join(r for r in reasons if r) or None


class ChatStream(threading.Thread):
    """System 2's stand-in: streamed chat completions through callo, one at a time. With `restart_every` a new request
    starts every that many seconds (the running one is cancelled by closing its connection); without it a request
    runs until it ends or `stop()`. Records each request's send time, first-delta time, status and slot, and the
    arrival time of every streamed content or reasoning delta (one per token on llama-server). A non-200 response or
    a failed request is recorded as `error` and followed by RETRY_BACKOFF_S of backoff."""

    RETRY_BACKOFF_S = 1.0

    def __init__(self, url: str, payload_fn: Callable[[], dict], *, headers: dict | None = None,
                 restart_every: float | None = None):
        super().__init__(daemon=True)
        self.url, self.payload_fn, self.headers, self.restart_every = url, payload_fn, headers or {}, restart_every
        self.requests: list[dict] = []
        self.deltas: list[float] = []
        self.first_delta = threading.Event()
        self.stuck = False
        self._halt = threading.Event()

    def _one(self, rec: dict, deadline: float | None) -> None:
        u = urlsplit(self.url)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
        try:
            body = json.dumps(self.payload_fn()).encode()
            conn.request("POST", u.path, body=body, headers={"Content-Type": "application/json",
                                                             "Content-Length": str(len(body)), **self.headers})
            resp = conn.getresponse()
            rec["status"], rec["slot"] = resp.status, resp.getheader("X-Callo-Slot")
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {resp.read(300).decode(errors='replace')}"
                return
            for raw in resp:
                if self._halt.is_set() or (deadline is not None and time.time() >= deadline):
                    rec["cancelled"] = True
                    return
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                ev = json.loads(data)
                delta = (ev["choices"][0].get("delta") or {}) if ev.get("choices") else {}
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.time()
                    rec["t_first"] = rec["t_first"] or now
                    rec["n"] += 1
                    self.deltas.append(now)
                    self.first_delta.set()
        finally:
            if conn.sock is not None:
                try:
                    conn.sock.shutdown(socket.SHUT_RDWR)   # callo's watcher then aborts the upstream generation
                except OSError:
                    pass
            conn.close()

    def run(self) -> None:
        while not self._halt.is_set():
            rec = {"t_send": time.time(), "t_first": None, "n": 0, "status": None, "slot": None, "error": None,
                   "cancelled": False}
            self.requests.append(rec)
            deadline = rec["t_send"] + self.restart_every if self.restart_every else None
            try:
                self._one(rec, deadline)
            except Exception as e:  # noqa: BLE001 -- recorded; the row's validity check counts it
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["t_end"] = time.time()
            if rec["error"]:
                self._halt.wait(self.RETRY_BACKOFF_S)          # a failing System 2 must not spin
            if deadline is not None:
                self._halt.wait(max(0.0, deadline - time.time()))

    def stop(self, timeout: float = 60) -> bool:
        """Stop and wait for the thread. False (and `stuck` set) when it is still running after `timeout`: a row must
        then be invalid, since System 2 keeps loading the server into whatever runs next."""
        self._halt.set()
        self.join(timeout=timeout)
        self.stuck = self.is_alive()
        return not self.stuck

    def problems(self) -> str | None:
        """Why this stream's load was not what the row claims, or None: an errored request, or a thread that did not
        stop."""
        bad = [r for r in self.requests if r["error"]]
        return _join_reasons([f"{len(bad)} System 2 request(s) errored, first: {bad[0]['error']}" if bad else None,
                              "System 2's stream was still running after stop()" if self.stuck else None])


def tokens_in_window(times: Sequence[float], w0: float, w1: float) -> float | None:
    """Deltas per second inside [w0, w1]: a stream's decode rate while the S1 ticks ran."""
    return sum(1 for t in times if w0 <= t <= w1) / (w1 - w0) if w1 > w0 else None


def split_by_overlap(rows: Sequence[dict], intervals: Sequence[tuple[float, float]]) -> tuple[list[dict], list[dict]]:
    """(rows whose [t_start, t_end] overlaps any interval, the others)."""
    hit, rest = [], []
    for r in rows:
        (hit if any(r["t_start"] < b and r["t_end"] > a for a, b in intervals) else rest).append(r)
    return hit, rest


def top2_margin(probs: dict[str, float]) -> float:
    p = sorted(probs.values(), reverse=True)
    return p[0] - (p[1] if len(p) > 1 else 0.0)


def g0_parity_aggregate(rows: Sequence[dict], *, n_expected: int = 106, dp_tol: float = 1e-3) -> dict:
    """The G0-parity verdict over per-record rows (each with `pair`, `prompt_equal`, `max_dp`, `callo`, `g0`). Split
    out so the gate can run live (`g0_parity`) or recompute from a stored run's rows without the GPU. PASS iff every
    record is counted (`n_expected`), every prompt is byte-identical, every argmax agrees and `max |Δp| < dp_tol`."""
    n_equal = sum(r["prompt_equal"] for r in rows)
    agree = sum(r["callo"] == r["g0"] for r in rows)
    worst = max(rows, key=lambda r: r["max_dp"], default=None)

    def pick(r: dict) -> dict:
        return {k: r[k] for k in ("pair", "max_dp", "callo", "g0")}

    return {"n": len(rows), "prompt_equal": n_equal, "agree": agree, "dp_tol": dp_tol,
            "max_dp": worst["max_dp"] if worst else None, "max_dp_pair": worst["pair"] if worst else None,
            "disagreements": [pick(r) for r in rows if r["callo"] != r["g0"]],
            "over_tol": [pick(r) for r in rows if r["max_dp"] >= dp_tol],
            "pass": (len(rows) == n_expected and n_equal == len(rows) and agree == len(rows)
                     and worst is not None and worst["max_dp"] < dp_tol)}


def g0_parity(records: Sequence[dict], *, render: Callable[[dict], str],
              answer: Callable[[dict], dict[str, float]], marker: str, n_expected: int = 106,
              dp_tol: float = 1e-3) -> dict:
    """The "G0 parity" row, re-specified after the parity diagnostic as a DECISION-equivalence
    test, run on G0's own server state (pinned delimiter, slot 0, every other slot empty), so the two sides run the
    same forward pass. Each record is one G0 first decision (from the game-client demo's gate harness, which is not
    part of this repository): `pair`, `media_marker` (of the server G0 ran on; random per process), `prompt_string`
    (what G0 sent to /completion), `body` (the same request as a /v1/systemone body) and `probs` (G0's option
    probabilities). `render(body)` is callo's prompt and
    `answer(body)` its served probabilities. PASS iff, on every one of `n_expected` records: the prompt is byte-identical
    (media marker normalised), the argmax agrees, and `max |Δp| < dp_tol`. It is NOT a bit-exact test: G0 records the
    demo's `letter_mass` readout and callo serves the full `bounded_readout`, two different formulas over the same
    logprobs that differ by ~1e-5 by construction (diagnostic test 3 = 1.4e-5), irreducible even with identical state.
    `dp_tol` (1e-3) is a readout-method tolerance far below any decision threshold, so it cannot mask a flipped argmax;
    the ACTUAL observed max |Δp| is reported alongside the gate."""
    rows = []
    for rec in records:
        probs = answer(rec["body"])
        g0 = rec["probs"]
        want = rec["prompt_string"].replace(rec["media_marker"], marker)
        rows.append({"pair": rec["pair"], "prompt_equal": render(rec["body"]) == want,
                     "max_dp": max(abs(probs[k] - g0[k]) for k in g0),
                     "callo": max(probs, key=probs.get), "g0": max(g0, key=g0.get)})
    out = g0_parity_aggregate(rows, n_expected=n_expected, dp_tol=dp_tol)
    out["rows"] = rows
    return out


def top2_logodds_shift(a: dict[str, float], b: dict[str, float]) -> float:
    """|Δ(log-odds)| from state `a` to state `b` for the two options with the highest probability IN `a` (a fixed
    pair, so the shift is well defined). ln(p_hi/p_lo) in each state, probabilities floored at 1e-12 so a saturated
    option stays finite. 0.0 when there are fewer than two options. A scale-free companion to |Δp|: on a confident
    item a visible log-odds move barely dents the probabilities."""
    top2 = sorted(a, key=a.get, reverse=True)[:2]
    if len(top2) < 2:
        return 0.0
    hi, lo = top2

    def lodds(p: dict[str, float]) -> float:
        return math.log(max(p[hi], 1e-12)) - math.log(max(p[lo], 1e-12))

    return abs(lodds(a) - lodds(b))


def kv_sensitivity_summary(pairs: Sequence[tuple[dict[str, float], dict[str, float]]]) -> dict:
    """From [(probs_s0, probs_s2), ...] (each pair the SAME prompt read in two server states, same option keys):
    median/max |Δp| over options, median/max top-2 log-odds shift (nats), and argmax agreement out of n. Ungated: the
    row reports these for `-kvu` and for `--no-kvu` (which is expected bit-exact on text)."""
    dps = [max(abs(s0[k] - s2[k]) for k in s0) for s0, s2 in pairs]
    shifts = [top2_logodds_shift(s0, s2) for s0, s2 in pairs]
    agree = sum(max(s0, key=s0.get) == max(s2, key=s2.get) for s0, s2 in pairs)
    return {"n": len(pairs), "median_dp": statistics.median(dps) if dps else None, "max_dp": max(dps) if dps else None,
            "median_logodds_shift": statistics.median(shifts) if shifts else None,
            "max_logodds_shift": max(shifts) if shifts else None, "argmax_agree": agree}


G0_RECORD_KEYS = ("pair", "media_marker", "prompt_string", "body", "probs")


def check_g0_records(records: Sequence[dict]) -> str | None:
    """None when every record has the five keys of the G0 record contract and the right types, and the
    `pair` values are distinct integers (106 lines with a repeated pair are not 106 decisions)."""
    for i, rec in enumerate(records):
        missing = [k for k in G0_RECORD_KEYS if k not in rec]
        if missing:
            return f"record {i} lacks {missing}"
        if not (isinstance(rec["prompt_string"], str) and isinstance(rec["media_marker"], str) and rec["media_marker"]
                and isinstance(rec["body"], dict) and isinstance(rec["probs"], dict) and rec["probs"]):
            return f"record {i} has a field of the wrong type"
        if not isinstance(rec["pair"], int) or isinstance(rec["pair"], bool):
            return f"record {i} has a non-integer pair {rec['pair']!r}"
    dup = sorted(p for p, n in Counter(rec["pair"] for rec in records).items() if n > 1)
    return f"duplicate pairs {dup[:10]}" if dup else None
