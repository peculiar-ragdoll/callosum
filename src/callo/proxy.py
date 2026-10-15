"""The front door: one `ThreadingHTTPServer` that relays everything to one llama-server.

- Routed endpoints (`slots.PINNABLE`) get exactly one key added to their JSON body, `id_slot`, chosen by the
  `SlotRouter`; everything else is byte-identical passthrough, including OPTIONS/CORS, except that /v1/systemone
  on a native decision model refuses audio and video with 422.
- Responses are HTTP/1.0 (the stdlib default): no keep-alive, streamed bodies delimited by connection close.
  Requests to the upstream use `http.client` (HTTP/1.1), which de-chunks, so `Transfer-Encoding` is never
  forwarded. Everything else the upstream sends, SSE comment pings included, is relayed verbatim as it arrives.
- A `DisconnectWatcher` `select()`s the client socket while a request is pending upstream; EOF shuts the
  upstream socket down, which trips llama-server's own abort. Requests carrying `X-Conversation-Id` are
  resumable streams on master and are deliberately not cancelled.
- Request bodies are read fully; chunked request bodies get 411; bodies over `max_routed_body` (64 MiB) are
  streamed through unrouted. `Expect: 100-continue` is answered at once (curl sends it for bodies > 1 KiB).
"""
from __future__ import annotations

import dataclasses
import hashlib
import http.client
import json
import logging
import math
import select
import socket
import threading
import time
import traceback
import uuid
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from . import transcribe
from .converge import GATE_CRITERIA as GATE_CRITERIA
from .converge import GATE_INSTRUCTIONS as GATE_INSTRUCTIONS
from .converge import GATE_QID as GATE_QID
from .converge import GATE_SHA256 as GATE_SHA256
from .converge import GATE_SKIP_P as GATE_SKIP_P
from .converge import ServeRule, ServeRules, gate_skips, serve_rules
from .deadline import (
    MAX_DEADLINE_MS,
    ConvergeRun,
    Estimators,
    deadline_cap,
    deadline_temperature,
    share_end,
    simulate,
    trace_terms,
)
from .engine import _BACKEND_DOWN, DeadlineRun, Engine, QuestionReadout, Readout
from .errors import (
    CalloError,
    Cancelled,
    InvalidInput,
    MediaRejected,
    MediaTooLarge,
    NoMedia,
    ReadoutError,
    ThinkTimeout,
    ThinkTooLarge,
    TooManyOptions,
    UpstreamUnavailable,
)
from .marginalize import WIDE_SPLIT_READ
from .profile import SKIPPED_STATES
from .prompt import TemplateError, messages
from .slots import PINNABLE, SlotRouter, canonical_text, conversation_hint
from .systemone import (
    InvalidRequest,
    Request,
    _question,
    native_media_refusal,
    native_think_refusal,
    parse_request,
)
from .think import MAX_BUDGET, ThinkProfile, check_pinned, json_safe
from .transcribe import TRANSCRIPTION_PATHS, Transcription

HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
                        "transfer-encoding", "upgrade"})
DEFAULT_MAX_ROUTED_BODY = 64 * 1024 * 1024
TIMINGS_MAX_BYTES = 4096   # X-Callo-Timings cap: under aiohttp's 8 KiB header line, room for the rest
# (request body, request headers) -> (status, body, reply headers); the reply headers always carry X-Callo-Slot.
# With ProxyConfig.systemone_cancel the handler also passes cancel= (a threading.Event) and watch= (a callable that
# starts the client-disconnect watcher), t_arr= (the monotonic arrival time) and log= (the request-log record dict),
# and a status of NO_REPLY means the client left: nothing is sent.
SystemOneFn = Callable[..., tuple[int, dict, dict[str, str]]]
NO_REPLY = 499
# the closes are `natural`, `budget` (a think profile's `forced`), `eos`, `converged` (early exit), `budget0` (a
# question whose share could not hold one token of thinking) and `deadline`, which is published as
# `think.deadline.closes.deadline` only: `think.deadline` is the deadline block's own name. The six sum to
# `questions`.
THINK_COUNTERS = ("requests", "questions", "natural", "budget", "eos", "converged", "budget0", "cancelled",
                  "timeouts", "skipped_media", "uncalibrated", "too_large", "readout_reprefill", "text_fallback")
# why a question that asked for early exit was served without it (X-Callo-Timings' `converge` word)
CONVERGE_OFF = ("unreachable", "not_chunkable", "media")
# early exit's gate (experimental): one budget-0 question, read before thinking on the request's own state, whose yes
# serves the request without early exit. Its text, its identity and its 0.5 rule are converge's (`GATE_INSTRUCTIONS`,
# `GATE_SHA256`, `gate_skips`), which calibrate's think probe and fit share.
GATE_ERROR_MAX = 200           # a failed read's `error`, in characters
GATE_BUSY = "the slot stayed busy past the deadline share"
GATE_QUESTION = _question(GATE_QID, {"type": "choice", "instructions": GATE_INSTRUCTIONS, "criteria": GATE_CRITERIA})
# POST /callo/think-probe: its own block, so calibration traffic stays out of serve's windows; only
# `s1.readouts` (the guard's condition 4) counts both
THINK_PROBE_COUNTERS = ("requests", "readouts", "errors", "cancelled", "timeouts", "too_large")
REPREFILL_PROMPT_N = 8          # a deliberate readout whose first pass evaluated more than this re-prefilled
log = logging.getLogger("callo")
SlotsFn = Callable[[], "list[dict] | None"]
StampedSlotsFn = Callable[[], "tuple[list[dict] | None, float]"]   # (snapshot, monotonic time it was fetched)


def _encode(obj: Any) -> bytes:
    """A routed body re-serialised for the upstream: UTF-8 JSON, or ASCII escapes when it holds a lone
    surrogate ("\\ud800" is valid JSON, but UTF-8 cannot encode it), so such a body is forwarded instead of
    dropping the connection. A RecursionError (too deep) propagates to the caller."""
    try:
        return json.dumps(obj, ensure_ascii=False).encode()
    except UnicodeEncodeError:
        return json.dumps(obj).encode()


def think_close_of(qr: QuestionReadout) -> str:
    """The serve-facing close of a deliberate readout: its own `think_close` under a deadline, else the
    trace's close with a profile's `forced` read as `budget`; a readout cut short of its trace (the probe's cuts) is
    `budget` whatever the trace did. Profiles and raw files keep calibration's `forced`."""
    if qr.think_close is not None:
        return qr.think_close
    assert qr.think is not None
    if qr.think_k < qr.think.n_gen or qr.think.close == "forced":
        return "budget"
    return qr.think.close


def gate_p(probs: Sequence[Any]) -> float | None:
    """P(yes) of a gate readout (option `a`'s probability), or None when it is missing: no readout, or a value that is
    not a finite number in [0, 1]."""
    if not probs:
        return None
    p = probs[0]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
        return None
    return float(p)


@dataclass(frozen=True)
class GateRead:
    """One request's gate: P(yes) (None when the read failed), the read's wall ms (the section wait included),
    whether early exit is skipped (a yes, or a failed read), and why the read failed."""
    p: float | None
    ms: float
    skip: bool
    error: str | None = None

    def fields(self) -> dict:
        """X-Callo-Timings' and the request log's `converge_gate`."""
        return {"p": None if self.p is None else round(self.p, 6), "ms": round(self.ms, 1), "skip": self.skip,
                **({"error": self.error} if self.error is not None else {})}


def percentile(xs: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the ceil(q*n)-th smallest value. q is rounded to 1e-6 so float noise cannot bump
    the rank."""
    s = sorted(xs)
    rank = -(-round(q * 10**6) * len(s) // 10**6)
    return s[max(0, min(len(s) - 1, rank - 1))]


class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.routes: Counter[str] = Counter()
        self.counters: Counter[str] = Counter()
        self._s1_ms: deque[float] = deque(maxlen=2000)
        self.low_mass = 0
        self.probe: dict | None = None
        self.temperature: float | None = None   # effective S1 temperature (serve sets it); None when the engine is off
        self.vision: bool | None = None         # all four set by serve
        self.audio: bool | None = None          # serve sets these two next to vision; video turns false
        self.video: bool | None = None          #   when the video self-test cannot load its clip
        self.media_marker: str | None = None
        self.user_delimiter: str | None = None
        self.cache_prompt: dict | None = None   # {"plain": --cache, "image" / "context" / "audio" / "video" / "mixed": True}
        # cut 1: each startup self-test's verdict (true | false), null when it did not run. A verdict reads
        # the readout's answer, which is debiased when a position-bias profile is loaded; its option mass is raw.
        self.self_test: dict[str, bool | None] = {"text": None, "vision": None, "audio": None, "video": None,
                                                  "think": None}      # the think self-test
        self._kind_ms: dict[str, deque[float]] = {k: deque(maxlen=2000) for k in ("image", "audio", "video")}
        self._kinds: Counter[str] = Counter()   # requests per kind: Request.kind
        self._passes = {k: [0, 0, 0] for k in ("image", "context", "audio", "video")}   # first passes with numbers,
                                                                                        # sum cache_n, sum prompt_n
        self.retok_stops = 0
        self.eog_stops = 0                      # readouts read AT a dominant end-of-generation token (walk stop "eog")
        self.reasks = 0                         # extra requests re-asking passes whose probabilities were withheld
        self.kv: str | None = None              # "unified" | "per_slot" in launch mode (serve sets it); None: attached, unknown
        # s1.n is a 2,000-wide latency window, so foreign-traffic checks need a monotonic count.
        self.s1_readouts = 0                    # question readouts answered, ever
        self.s1_topk_miss = 0                   # floored options served, ever
        # readouts of questions above 26 options, their split passes, those whose unread
        # first-letter mass exceeded WIDE_SPLIT_READ, and those of a question past wide_coverage (labels that are not
        # single tokens, read in two passes); serve sets wide_coverage and wide_readable (None: the engine is off)
        self.wide_readouts = self.wide_split_passes = self.wide_unread_over = self.wide_uncovered_readouts = 0
        self.wide_coverage: int | None = None
        self.wide_readable: int | None = None
        self.instance = uuid.uuid4().hex        # one id per callo startup: calibrate's guard tells a restart by it
        self.model_key: str | None = None       # serve sets these four
        self.model_key_error: str | None = None  # why model_key is null (no profile loaded, so startup went on)
        self.model_file: str | None = None
        self.provenance: dict | None = None
        self.position_bias: dict | None = None  # PositionBias.info() when serve loaded a profile
        self._bias: Counter[str] = Counter()    # Readout.bias per question readout
        # think budgets: the static block serve sets (capable, reason, close, synthesized_open, default_budget,
        # max, profile, sampling, gguf_sampling), its counters, the questions per budget, the generation window
        self.think: dict | None = None
        self._think: Counter[str] = Counter()
        self._think_probe: Counter[str] = Counter()  # THINK_PROBE_COUNTERS
        self._think_budgets: Counter[str] = Counter()
        self._deadline: Counter[str] = Counter()     # think.deadline's counters
        # the deadline windows (elapsed per timed question, overshoot per deadline/budget close, think tokens
        # per deliberate question by the request's deliberate-question count) and serve's estimators and default
        self._dl_elapsed: deque[float] = deque(maxlen=2000)
        self._dl_over: deque[float] = deque(maxlen=2000)
        self._dl_tokens: dict[str, deque[float]] = {k: deque(maxlen=2000) for k in ("all", "1", "2", "3+")}
        self._dl_by_count: Counter[str] = Counter()
        self._dl_chunks = [0, 0]                         # chunks summed over chunked questions, their count
        self.estimators: Estimators | None = None
        self.deadline_default_ms = 0
        # early exit: the static block serve sets (default_z, K, rules), the counters by question, the stop positions
        self.converge_static: dict = {"default_z": 0.0, "K": None, "rules": None}
        self._cv: Counter[str] = Counter()
        self._cv_stop_k: deque[float] = deque(maxlen=2000)
        # early exit's gate: serve's default (--think-converge-gate), its counters and the read window
        self.converge_gate_default = False
        self._gate: Counter[str] = Counter()
        self._gate_ms: deque[float] = deque(maxlen=2000)
        self.think_budget_keys: set[int] = set()     # the budgets counted by name; everything else is "other"
        self._gen_ms: deque[float] = deque(maxlen=2000)
        self._think_tokens = [0, 0]                   # sum, n
        self.host: dict | None = None                 # serve sets it
        self.temperature_source: str | None = None    # flag | profile | default
        self.model_bytes: int | None = None
        self.mmproj_bytes: int | None = None
        self.n_ctx: int | None = None
        self.total_slots: int | None = None
        self.upstream_local: bool | None = None       # llama-server on this box (loopback upstream); the signal
        self.section_timeouts = 0
        self.asr_preset: str | None = None            # "default" | "lfm2"; None without audio
        self.transcription: dict | None = None        # {"translate", "reason"}; serve sets it

    def count_route(self, path: str) -> None:
        with self._lock:
            self.routes[path] += 1

    def bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.counters[key] += n

    def bump_think(self, key: str, n: int = 1) -> None:
        """One of THINK_COUNTERS the service sees before or instead of a readout (cancelled, timeouts, skipped_media,
        uncalibrated, too_large). `skipped_media` and `uncalibrated` are counted when the request is PLANNED
        (`think_plan`), not when it is served: a request that then 413s, 504s or is cancelled still counts, so
        `uncalibrated` can exceed `requests`."""
        with self._lock:
            self._think[key] += n

    def bump_deadline(self, key: str, n: int = 1) -> None:
        """A `think.deadline` counter the service sees outside a readout: `rerouted`, `not_chunkable`."""
        with self._lock:
            self._deadline[key] += n

    def bump_probe(self, key: str, n: int = 1) -> None:
        """One of THINK_PROBE_COUNTERS: a probe answered (`requests`) or not (`errors`), and why it was not."""
        with self._lock:
            self._think_probe[key] += n

    def add_probe(self, readouts: Sequence[QuestionReadout]) -> None:
        """One answered think probe's readouts: counted in the monotonic `s1.readouts` (the guard's condition 4 must
        see every readout on a slot) and in `think_probe.readouts`, and in nothing else: the probe's cut readouts,
        Bmax generations and T = 1 budget-0 readouts would skew every window serve reports."""
        with self._lock:
            self.s1_readouts += len(readouts)
            self._think_probe["readouts"] += len(readouts)

    def add_gate(self, g: GateRead) -> None:
        """One request's gate read: asked, whether it skipped early exit, whether it failed; a read that answered
        enters the read window."""
        with self._lock:
            self._gate["asked"] += 1
            self._gate["skipped"] += g.skip
            if g.error is not None:
                self._gate["errors"] += 1
            else:
                self._gate_ms.append(g.ms)

    def bump_section_timeout(self) -> None:
        with self._lock:
            self.section_timeouts += 1

    def add_s1(self, ms: float, option_mass: float) -> None:
        """The per-question record. Bumps the monotonic `s1.readouts` too, so `s1.n <= s1.readouts` always holds."""
        with self._lock:
            self._add_s1_locked(ms, option_mass)
            self.s1_readouts += 1

    def _add_s1_locked(self, ms: float, option_mass: float) -> None:
        self._s1_ms.append(ms)
        if option_mass < 0.5:
            self.low_mass += 1

    def add_request(self, kind: str, ms: float, readouts: Sequence[QuestionReadout]) -> None:
        """One answered /v1/systemone request: per-question latency and mass (the `s1`), plus the per-kind
        counts, the image / audio / video request latency and the first-pass cache numbers."""
        with self._lock:
            self._kinds[kind] += 1
            if kind in self._kind_ms:
                self._kind_ms[kind].append(ms)
            deliberate_seen: set[int] = set()          # the traces counted in this request
            counted = False                             # `requests` once per request with a deliberate question
            for qr in readouts:
                r = qr.readout
                self._add_s1_locked(qr.ms, r.option_mass)
                self.s1_readouts += 1
                self.s1_topk_miss += r.topk_miss
                if r.wide:
                    self.wide_readouts += 1
                    self.wide_split_passes += r.split_passes
                    self.wide_unread_over += r.unread_mass > WIDE_SPLIT_READ
                    self.wide_uncovered_readouts += r.uncovered > 0
                if r.bias is not None:
                    self._bias[r.bias] += 1
                if r.stop == "retok":
                    self.retok_stops += 1
                elif r.stop == "eog":
                    self.eog_stops += 1
                self.reasks += r.reasks
                if kind in self._passes and qr.think is None and r.cache_n is not None and r.prompt_n is not None:
                    acc = self._passes[kind]                  # deliberate readouts stay out of the cache stats
                    acc[0], acc[1], acc[2] = acc[0] + 1, acc[1] + r.cache_n, acc[2] + r.prompt_n
                if qr.think is not None or qr.think_close == "budget0":    # a fallback is deliberate too
                    if not counted:
                        self._think["requests"] += 1
                        counted = True
                    self._think["questions"] += 1
                    self._think[think_close_of(qr)] += 1
                if qr.think is not None:                  # the trace's own numbers
                    self._think_budgets[str(qr.think_budget) if qr.think_budget in self.think_budget_keys
                                        else "other"] += 1
                    if id(qr.think) not in deliberate_seen:   # one generation, however many readouts cut it
                        self._gen_ms.append(qr.think.gen_wall_ms)
                    deliberate_seen.add(id(qr.think))
                    self._think_tokens[0] += qr.think_k
                    self._think_tokens[1] += 1
                    if qr.read_prompt_n is not None and qr.read_prompt_n > REPREFILL_PROMPT_N:
                        self._think["readout_reprefill"] += 1
                    if qr.text_fallback:
                        self._think["text_fallback"] += 1
            self._add_deadline_locked(readouts)
            self._add_converge_locked(readouts)

    def _add_deadline_locked(self, readouts: Sequence[QuestionReadout]) -> None:
        """One request's questions served under a deadline. `met` / `missed` and `elapsed_ms` count every timed
        question; `questions`, the closes, the think tokens and `chunks_mean` its deliberate ones (fallbacks
        included); `over_ms` the `deadline` and `budget` closes; `cont_reprefill` continuation chunks that evaluated
        more than one token, the chunks after an early-exit probe aside (`converge.after_probe` counts those)."""
        timed = [qr for qr in readouts if qr.deadline_ms is not None]
        if not timed:
            return
        for qr in timed:
            if qr.elapsed_ms is not None:
                self._dl_elapsed.append(qr.elapsed_ms)
                self._deadline["met" if qr.elapsed_ms <= qr.deadline_ms else "missed"] += 1
        deliberate = [qr for qr in timed if qr.think is not None or qr.think_close == "budget0"]
        if not deliberate:
            return
        self._deadline["requests"] += 1
        key = str(len(deliberate)) if len(deliberate) < 3 else "3+"
        self._dl_by_count[key] += len(deliberate)
        for qr in deliberate:
            close = think_close_of(qr)
            self._deadline["questions"] += 1
            self._deadline[f"close_{close}"] += 1
            k = qr.think_k if qr.think is not None else 0
            self._dl_tokens["all"].append(k)
            self._dl_tokens[key].append(k)
            if close in ("deadline", "budget") and qr.elapsed_ms is not None:
                self._dl_over.append(qr.elapsed_ms - qr.deadline_ms)
            if qr.chunks:
                self._dl_chunks[0] += qr.chunks
                self._dl_chunks[1] += 1
            if qr.think is not None:
                self._deadline["cont_reprefill"] += qr.think.cont_reprefill

    def _deadline_block_locked(self) -> dict:
        """/callo/stats `think.deadline`."""
        def q(xs: Sequence[float], keys: Sequence[tuple[str, float]]) -> dict:
            return {name: (percentile(xs, at) if xs else None) for name, at in keys}

        reserve, source = self.estimators.reserve() if self.estimators is not None else (None, None)
        est = self.estimators.stats() if self.estimators is not None else {}
        el, over = list(self._dl_elapsed), list(self._dl_over)
        tok = {k: list(v) for k, v in self._dl_tokens.items()}
        return {"default_ms": self.deadline_default_ms, "reserve_ms": reserve, "reserve_source": source, **est,
                "requests": self._deadline["requests"], "questions": self._deadline["questions"],
                "questions_by_count": {k: self._dl_by_count[k] for k in ("1", "2", "3+")},
                "closes": {c: self._deadline[f"close_{c}"]
                           for c in ("natural", "budget", "deadline", "eos", "converged", "budget0")},
                **{k: self._deadline[k] for k in ("met", "missed", "rerouted", "not_chunkable", "cont_reprefill")},
                "elapsed_ms": q(el, (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("p99.5", 0.995), ("max", 1.0))),
                "over_ms": {**q(over, (("p95", 0.95), ("p99", 0.99), ("p99.5", 0.995), ("max", 1.0))), "n": len(over)},
                "think_tokens": {**q(tok["all"], (("p50", 0.5), ("p95", 0.95))),
                                 "by_count": {k: q(tok[k], (("p50", 0.5), ("p95", 0.95))) for k in ("1", "2", "3+")}},
                "chunks_mean": self._dl_chunks[0] / self._dl_chunks[1] if self._dl_chunks[1] else None}

    def _add_converge_locked(self, readouts: Sequence[QuestionReadout]) -> None:
        """One request's questions that asked for early exit: when it ran (`converge` a target z), its probes, its stop
        and the cap it stopped short of (the tokens-saved bound); else why not (CONVERGE_OFF)."""
        ran = [qr for qr in readouts if isinstance(qr.converge, float)]
        for qr in readouts:
            if isinstance(qr.converge, str):
                self._cv[qr.converge] += 1
        if not ran:
            return
        self._cv["requests"] += 1
        for qr in ran:
            self._cv["questions"] += 1
            self._cv["probes"] += qr.probes or 0
            if qr.think_close == "converged":
                self._cv["converged"] += 1
                self._cv_stop_k.append(qr.think_k)
                self._cv["tokens_saved"] += max(0, qr.think_budget - qr.think_k)

    def _converge_block_locked(self) -> dict:
        """/callo/stats `think.converge`, counted by question:

        - `requests` and `questions`: the requests, and the deliberate questions in them, served with an early-exit
          rule. A question that fell back to budget 0 (its share of the deadline could not hold one token of thinking,
          or it was rerouted to a quiet slot) counts too, with 0 probes.
        - `probes_per_question`: `probes` over `questions`, fallbacks included.
        - `converged` and `stop_k`: the questions that stopped early, and the think tokens each had when it stopped.
        - `unreachable`, `not_chunkable` and `media`: questions that asked for early exit and were served without it,
          one per question that could think (a question with a single option never counts).
        - `tokens_saved`: the cap minus the stop position, summed over the converged questions, and
          `tokens_saved_mean` its mean per converged question. Both are upper bounds: the rest of a stopped trace is
          never generated, so the real saving cannot be observed.
        - `gate` (experimental), by request: serve's `default`, the gate reads `asked`, the requests they `skipped`
          (served without early exit: a yes, or a failed read), the failed reads (`errors`) and the answered reads'
          wall `read_ms`. A request the gate skipped runs no rule, so it is not in `requests` or `questions`."""
        est = self.estimators.converge_stats() if self.estimators is not None else {
            "probe_read_ms": {"p50": None, "p95": None}, "after_probe": {"n": 0, "restored": 0, "share": None},
            "c_ms": None, "c_source": None}
        stop, q, conv, saved = list(self._cv_stop_k), self._cv["questions"], self._cv["converged"], self._cv["tokens_saved"]
        gms = list(self._gate_ms)
        return {**self.converge_static, "requests": self._cv["requests"], "questions": q, "converged": conv,
                **{k: self._cv[k] for k in CONVERGE_OFF}, "probes": self._cv["probes"],
                "probes_per_question": self._cv["probes"] / q if q else None,
                "stop_k": {"p50": percentile(stop, 0.5) if stop else None,
                           "p95": percentile(stop, 0.95) if stop else None},
                "tokens_saved": saved, "tokens_saved_mean": saved / conv if conv else None, **est,
                "gate": {"default": self.converge_gate_default,
                         **{k: self._gate[k] for k in ("asked", "skipped", "errors")},
                         "read_ms": {"p50": percentile(gms, 0.5) if gms else None,
                                     "p95": percentile(gms, 0.95) if gms else None}}}

    def _mean(self, kind: str, i: int) -> float | None:
        n = self._passes[kind][0]
        return self._passes[kind][i] / n if n else None

    def _per_kind(self, kind: str) -> dict:
        """s1.<kind>_requests, _p50_ms, _p95_ms, _cache_n_mean, _prompt_n_mean (image, audio, video)."""
        ms = list(self._kind_ms[kind])
        return {f"{kind}_requests": self._kinds[kind], f"{kind}_p50_ms": percentile(ms, 0.5) if ms else None,
                f"{kind}_p95_ms": percentile(ms, 0.95) if ms else None, f"{kind}_cache_n_mean": self._mean(kind, 1),
                f"{kind}_prompt_n_mean": self._mean(kind, 2)}

    def snapshot(self, router: SlotRouter, slots: list[dict] | None) -> dict:
        with self._lock:
            lat = list(self._s1_ms)
            s1 = {"n": len(lat), "p50_ms": percentile(lat, 0.5) if lat else None,
                  "p95_ms": percentile(lat, 0.95) if lat else None, "low_mass": self.low_mass,
                  **self._per_kind("image"),
                  "context_requests": self._kinds["context"], "context_cache_n_mean": self._mean("context", 1),
                  "context_prompt_n_mean": self._mean("context", 2), "retok_stops": self.retok_stops,
                  "eog_stops": self.eog_stops, "reasks": self.reasks,
                  "readouts": self.s1_readouts, "topk_miss": self.s1_topk_miss,
                  **self._per_kind("audio"), **self._per_kind("video"), "mixed_requests": self._kinds["mixed"],
                  "wide_readouts": self.wide_readouts, "wide_split_passes": self.wide_split_passes,
                  "wide_unread_over": self.wide_unread_over, "wide_uncovered_readouts": self.wide_uncovered_readouts}
            bias = None if self.position_bias is None else {
                **self.position_bias, "applied": self._bias["applied"],
                "skipped": {k: self._bias[k] for k in SKIPPED_STATES}}
            rs = router.snapshot()
            s1["section_waits"] = sum(sec["waits"] for sec in rs.get("sections", {}).values())
            s1["section_timeouts"] = self.section_timeouts
            s1["draining"] = rs.get("counters", {}).get("s1_draining", 0)
            # callo:asr's home (null while homeless) in the transcription block, and its moves as the
            # counter transcription_rehomed (absent until first counted, as every counter)
            asr = rs.get("sticky", {}).get(transcribe.ASR_HINT, {})
            counters = dict(self.counters)
            if asr.get("rehomed"):
                counters["transcription_rehomed"] = asr["rehomed"]
            transcription = None if self.transcription is None else {
                **self.transcription, "asr_home": asr.get("home"), "asr_home_idle_s": router.asr_home_idle}
            gen = list(self._gen_ms)
            think = None if self.think is None else json_safe({     # a GGUF's nan / inf: strict JSON all the same
                **self.think, **{k: self._think[k] for k in THINK_COUNTERS},
                "questions_by_budget": {**{str(b): self._think_budgets[str(b)] for b in sorted(self.think_budget_keys)},
                                        "other": self._think_budgets["other"]},
                "gen_p50_ms": percentile(gen, 0.5) if gen else None,
                "gen_p95_ms": percentile(gen, 0.95) if gen else None,
                "think_tokens_mean": self._think_tokens[0] / self._think_tokens[1] if self._think_tokens[1] else None,
                "deadline": self._deadline_block_locked(), "converge": self._converge_block_locked()})
            return {"uptime_s": time.time() - self.started, "routes": dict(self.routes),
                    "counters": counters, "router": rs, "s1": s1,
                    "probe": self.probe, "slots": slots, "temperature": self.temperature, "vision": self.vision,
                    "audio": self.audio, "video": self.video, "self_test": dict(self.self_test),
                    "media_marker": self.media_marker, "user_delimiter": self.user_delimiter,
                    "cache_prompt": self.cache_prompt, "kv": self.kv, "instance": self.instance,
                    "model_key": self.model_key, "model_key_error": self.model_key_error,
                    "model_file": self.model_file, "provenance": self.provenance, "position_bias": bias,
                    "think": think, "think_probe": {k: self._think_probe[k] for k in THINK_PROBE_COUNTERS},
                    "host": self.host, "temperature_source": self.temperature_source,
                    "model_bytes": self.model_bytes, "mmproj_bytes": self.mmproj_bytes, "n_ctx": self.n_ctx,
                    "total_slots": self.total_slots, "upstream_local": self.upstream_local,
                    "asr_preset": self.asr_preset, "transcription": transcription,
                    "wide_coverage": self.wide_coverage, "wide_readable": self.wide_readable}


@dataclass
class ProxyConfig:
    upstream: str
    router: SlotRouter
    systemone: SystemOneFn | None = None      # None: /v1/systemone goes to a native decision model (_systemone_native)
    slots_snapshot: SlotsFn | None = None     # a GET /slots at most 250 ms old, when the upstream exposes it
    stats: Stats = field(default_factory=Stats)
    # Passthrough bounds only the CONNECT. Reads have no timeout: a long non-streaming generation is one blocking read
    # that ends when llama-server is done (it has no cap of its own), and timing it out would 502 the agent and abort
    # the generation upstream. A vanished client is the DisconnectWatcher's job; a dead upstream reads as EOF.
    connect_timeout: float = 10.0
    max_routed_body: int = DEFAULT_MAX_ROUTED_BODY
    # think budgets: the handler passes cancel= and watch= to `systemone` (SystemOneService.respond takes them; a
    # plain 2-argument callable does not), and drops the reply when the status is NO_REPLY
    systemone_cancel: bool = False
    # POST /callo/think-probe (calibrate's internal endpoint); None answers 404. It is answered
    # from loopback peers only unless `allow_think_probe` (serve's --allow-think-probe).
    think_probe: SystemOneFn | None = None
    allow_think_probe: bool = False
    # POST /v1/audio/transcriptions translation; None (no audio upstream) passes every transcription
    # through, counted as `transcription_unrouted_no_audio`. serve sets it from /props and the startup render check.
    transcription: Transcription | None = None
    # `callo serve --request-log FILE`, one JSON line per /v1/systemone request; None: off
    request_log: RequestLog | None = None


class RequestLog:
    """`--request-log FILE`, opened for appending, line-buffered. One JSON line per /v1/systemone
    request, written after the reply's last byte (or when the client left): `seq`, `body_sha256` (of the raw body
    bytes as received), `t_arr_unix_ms`, `t_written_unix_ms`, `served_ms` (from `t_arr` to the last byte written, the
    socket write included), `status` (or "cancelled"), `slot`, `deadline_ms`, `budget`, `converge` (only when the
    request asked for early exit: its target z or why it was off), `converge_gate` (only when early exit's gate was
    read: X-Callo-Timings' object) and, per question in request order, `qid`, `answer`
    (the chosen label), `think_close`, `k`, `elapsed_ms`, `prompt_tokens`. Never request text.
    The constructor raises OSError for a file it cannot open (serve's phase-1 refusal)."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "a", buffering=1, encoding="utf-8")      # open for the server's lifetime
        self._lock = threading.Lock()
        self._seq = 0

    def write(self, rec: Mapping[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            line = {"seq": self._seq, **{k: rec.get(k) for k in ("body_sha256", "t_arr_unix_ms", "t_written_unix_ms",
                                                                   "served_ms", "status", "slot", "deadline_ms",
                                                                   "budget")}}
            if "converge" in rec:                       # early exit asked for: its target z or why it was off
                line["converge"] = rec["converge"]
            if "converge_gate" in rec:                  # early exit's gate was read
                line["converge_gate"] = rec["converge_gate"]
            line["questions"] = list(rec.get("questions") or [])
            self._f.write(json.dumps(line, separators=(",", ":")) + "\n")

    def close(self) -> None:
        with self._lock:
            self._f.close()


class _LimitedReader:
    """`rfile` capped at Content-Length, so http.client can stream a huge body without reading past it."""

    def __init__(self, raw, n: int):
        self.raw, self.left = raw, n

    def read(self, size: int = -1) -> bytes:
        if self.left <= 0:
            return b""
        if size is None or size < 0 or size > self.left:
            size = self.left
        data = self.raw.read(size)
        self.left -= len(data)
        return data


class DisconnectWatcher(threading.Thread):
    """While a request is pending upstream, watch the client socket; on EOF shut the upstream socket down.

    Only `shutdown()`, never `conn.close()`: the handler thread may be inside `resp.read1()` on that connection,
    and closing it from here races the reader. The shutdown wakes the reader with EOF; the handler's `finally`
    closes `conn`. Think budgets: `on_eof` generalises it: a callback run on EOF (the deliberate
    request's `cancel.set`), with `conn` None when there is no single upstream socket to shut."""

    def __init__(self, client: socket.socket, conn: http.client.HTTPConnection | None, *,
                 on_eof: Callable[[], None] | None = None):
        super().__init__(daemon=True)
        self.client, self.conn, self.on_eof = client, conn, on_eof
        self.tripped = False
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([self.client], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not r:
                continue
            try:
                data = self.client.recv(1, socket.MSG_PEEK)
            except OSError:
                data = b""
            if data:
                return  # bytes queued by the client are not a disconnect; stop polling and leave the request alone
            self.tripped = True
            sock = self.conn.sock if self.conn is not None else None
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if self.on_eof is not None:
                self.on_eof()
            return

    def stop(self) -> None:
        self._stop.set()


def _is_loopback(address: str) -> bool:
    """The peer address of a connection, as BaseHTTPRequestHandler reports it (IPv4 or IPv6, no port). A numeric
    predicate: a host NAME such as "localhost" is cli.upstream_is_local's."""
    return address in ("127.0.0.1", "::1", "::ffff:127.0.0.1") or address.startswith("127.")


def s1_timeout_problem(t: Any) -> str | None:
    """Why `t` cannot be --s1-timeout, or None: the S1 section wait is bounded by it, and
    Section.acquire needs a finite number of seconds above 0 (inf overflows the wait). The one check: cli's usage error
    and SystemOneService's refusal both say this."""
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
        return f"--s1-timeout must be a finite number of seconds above 0, got {t!r}"
    return None


# llama-server's own reply to every request while it loads (tools/server/server-http.cpp:307-326, README /health).
LOADING_BODY = json.dumps({"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}).encode()
LOADING_CTYPE = "application/json; charset=utf-8"


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "callo"
    sys_version = ""
    cfg: ProxyConfig            # per server class attribute; None while `serve` is still starting up

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 -- http.server's naming
        pass

    # ---- entry ------------------------------------------------------------------------------------

    def _dispatch(self) -> None:
        self.t_arr = time.monotonic()       # a deadline's clock starts here, before the body is read
        self.t_arr_unix = time.time()       # the same instant on the wall clock, for the request log
        cfg = self.cfg = type(self).cfg     # ONE read: the whole request sees the same config (set_config swaps it)
        if cfg is None:
            try:
                self._loading()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        path = self.path.split("?", 1)[0]
        cfg.stats.count_route(path)
        try:
            if path == "/callo/stats" and self.command == "GET":
                slots = cfg.slots_snapshot() if cfg.slots_snapshot is not None else None
                self._send_json(200, cfg.stats.snapshot(cfg.router, slots))
            elif path == "/health" and self.command == "GET":
                self._health()
            elif path == "/v1/systemone" and self.command == "POST" and cfg.systemone is not None:
                self._systemone()
            elif path == "/v1/systemone" and self.command == "POST":    # a native decision model
                self._systemone_native()
            elif path == "/callo/think-probe" and self.command == "POST":
                self._think_probe()
            elif path in TRANSCRIPTION_PATHS and self.command == "POST":
                self._transcription(path)
            elif path in PINNABLE and self.command == "POST":
                self._routed(path)
            else:
                self._passthrough()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client left; there is nobody to tell

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = _dispatch

    def _loading(self) -> None:
        """llama-server's loading semantics: every path, /health included, gets its 503. The body is drained
        first (when the client is not waiting for 100 Continue) so closing the socket cannot reset the reply."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if 0 < n <= DEFAULT_MAX_ROUTED_BODY and self.headers.get("Expect", "").lower() != "100-continue":
            while n > 0:
                chunk = self.rfile.read(min(n, 65536))
                if not chunk:
                    break
                n -= len(chunk)
        self._send_raw(503, LOADING_BODY, LOADING_CTYPE)

    # ---- request body -------------------------------------------------------------------------------

    def _body_length(self) -> int | None:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            self._send_json(411, {"error": {"message": "chunked request bodies are not supported; send Content-Length",
                                            "type": "callo"}})
            return None
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0:   # a negative length would sail past the size cap and make rfile.read(n) read to EOF
            self._send_json(400, {"error": {"message": "bad Content-Length", "type": "callo"}})
            return None
        return n

    def _expect_continue(self) -> None:
        # RFC 7231 section 5.1.1: a server must not send 100 Continue to an HTTP/1.0 client
        if self.request_version >= "HTTP/1.1" and self.headers.get("Expect", "").lower() == "100-continue":
            self.wfile.write(b"HTTP/1.1 100 Continue\r\n\r\n")

    def _drain_body(self, n: int, *, limit: int = DEFAULT_MAX_ROUTED_BODY) -> None:
        """Read and discard up to `limit` bytes of the request body, so closing the socket does not reset the
        connection before the client has read the reply we already sent. Best-effort and bounded: a short read
        timeout means a client that declared a body it is not sending cannot stall us, and a client waiting for
        100 Continue (which it will never now get) is left alone."""
        if n <= 0 or self.headers.get("Expect", "").lower() == "100-continue":
            return
        remaining = min(n, limit)
        try:
            old = self.connection.gettimeout()
            self.connection.settimeout(1.0)
        except OSError:
            return
        try:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.connection.settimeout(old)
            except OSError:
                pass

    # ---- routes -------------------------------------------------------------------------------------

    def _routed(self, path: str) -> None:
        cfg = self.cfg
        n = self._body_length()
        if n is None:
            return
        self._expect_continue()
        if n > cfg.max_routed_body:
            cfg.stats.bump("unrouted_large")
            self._passthrough(stream_len=n)
            return
        body = self.rfile.read(n)
        obj = None
        if not self.headers.get("Content-Encoding"):
            try:
                obj = json.loads(body)
            except (ValueError, RecursionError):   # RecursionError: a deeply nested body is not routable JSON
                obj = None
        if not isinstance(obj, dict):
            cfg.stats.bump("unrouted_body")
            self._passthrough(body=body)
            return
        router = cfg.router
        try:
            text, digests = canonical_text(path, obj)     # media digested in a copy; obj is unchanged
        except RecursionError:                            # parsed, but too deep to walk again: route it unpinned
            self._unpinned(obj, body, count="unrouted_body")
            return
        hint = conversation_hint(dict(self.headers.items()), obj)
        d = router.route_agent(text, hint=hint, client_slot=obj.get("id_slot"), digests=digests)   # choose + acquire
        if d.slot is None:
            self._unpinned(obj, body)
            return
        obj["id_slot"] = d.slot
        try:
            self._passthrough(body=_encode(obj), slot=str(d.slot))
        finally:
            router.release(d.slot)

    # ---- POST /v1/audio/transcriptions ---------------------------------------------------------

    def _transcription(self, path: str) -> None:
        """A well-formed multipart request becomes the `/v1/chat/completions` request
        llama-server builds itself, pinned to an agent slot and reshaped; anything else passes through untouched,
        counted as `transcription_unrouted_<reason>`. `no_audio`, `no_pinning` and `render` are decided before the body
        is read, so those requests take the streaming passthrough unchanged."""
        cfg = self.cfg
        asr = cfg.transcription
        early = ("no_audio" if asr is None else "no_pinning" if not cfg.router.pinning
                 else "render" if asr.off is not None else None)
        if asr is None or early is not None:
            cfg.stats.bump(transcribe.unrouted_counter(early or "no_audio"))     # early is set whenever asr is None
            self._passthrough()
            return
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            cfg.stats.bump(transcribe.unrouted_counter("chunked"))   # _body_length answers it with the 411
        n = self._body_length()
        if n is None:
            return
        self._expect_continue()
        if n > cfg.max_routed_body:
            self._asr_unrouted("large", stream_len=n)
            return
        body = self.rfile.read(n)
        if self.headers.get("Content-Encoding"):
            self._asr_unrouted("encoded", body=body)
            return
        t = transcribe.plan(self.headers.get_all("Content-Type") or [], body, lfm2=asr.lfm2)
        if isinstance(t, str):
            self._asr_unrouted(t, body=body)
            return
        fields = t.fields
        data = _encode(t.body)                     # flat strings and two short lists: never too deep to serialise
        del t                                      # memory: the base64 string goes with it
        if len(data) + transcribe.ID_SLOT_ROOM >= transcribe.UPSTREAM_MAX_BODY:
            self._asr_unrouted("large", body=body)  # llama-server would refuse the JSON but takes the multipart
            return
        del body
        router = cfg.router
        # the client's hint (which moves on a fallback, as for any agent chat), else the
        # shared callo:asr, which is sticky (a busy home sends this one request to the normal pick but stays home; the
        # home is a slot no agent hint is bound to, and an agent hint claiming it re-homes callo:asr); the empty routing
        # text never wins the LCP test, and a form id_slot is ignored
        client = conversation_hint(dict(self.headers.items()), fields)
        d = router.route_agent("", hint=client or transcribe.ASR_HINT, sticky=client is None)
        # route_agent answers None only with agent pinning off, and the `no_pinning` row has already passed that through
        assert d.slot is not None
        head = b'{"id_slot":%d,' % d.slot
        try:
            self._asr_forward(path, [head, memoryview(data)[1:]], len(head) + len(data) - 1, d.slot)
        finally:
            router.release(d.slot)

    def _asr_unrouted(self, reason: str, *, body: bytes | None = None, stream_len: int | None = None) -> None:
        """Pass an already-read (or streamed) transcription through untouched, counted; a large one also counts
        as the generic `unrouted_large`."""
        self.cfg.stats.bump(transcribe.unrouted_counter(reason))
        if reason == "large":
            self.cfg.stats.bump("unrouted_large")
        self._passthrough(body=body, stream_len=stream_len)

    def _asr_forward(self, path: str, chunks: list, length: int, slot: int) -> None:
        """The translated request, buffered: `_passthrough`'s header filter plus the body-describing headers, the
        client's Authorization / X-Api-Key / Origin unchanged and never callo's own key; the DisconnectWatcher
        always (a client that left gets nothing); a 2xx reshaped with the upstream's headers (CORS) kept, a non-2xx
        relayed unchanged, a dead upstream a 502 `callo_upstream`."""
        cfg = self.cfg
        label = str(slot)
        query = self.path.split("?", 1)[1] if "?" in self.path else None
        target = ("/v1/chat/completions" if path.startswith("/v1/") else "/chat/completions") + (
            f"?{query}" if query is not None else "")
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_BY_HOP and k.lower() not in transcribe.REQUEST_DROP}
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(length)
        conn = self._open_upstream(label)
        if conn is None:
            return
        watcher = DisconnectWatcher(self.connection, conn)
        watcher.start()
        try:
            try:
                conn.request("POST", target, body=chunks, headers=headers)
                resp = conn.getresponse()
                payload = resp.read()
            except (OSError, http.client.HTTPException) as e:
                if watcher.tripped:
                    return                   # the client left; closing conn below aborts the generation
                self._upstream_502(e, label)
                return
            if watcher.tripped:
                return
            if not 200 <= resp.status < 300:
                self._relay(resp, payload, label)
                return
            out = transcribe.reshape(payload)
            if out is None:
                cfg.stats.bump("transcription_bad_reply")
                self._send_json(502, {"detail": "llama-server's reply to the translated transcription has no "
                                                "choices[0].message.content string"}, extra={"X-Callo-Slot": label})
                return
            cfg.stats.bump("transcription_translated")
            data = _encode(out)              # ensure_ascii=False: raw UTF-8, as upstream
            self.send_response(200)
            self._relay_headers(resp, transcribe.REPLY_DROP)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Callo-Slot", label)
            self.end_headers()
            self.wfile.write(data)
        finally:
            watcher.stop()
            conn.close()

    def _relay(self, resp: http.client.HTTPResponse, payload: bytes, label: str) -> None:
        """A buffered upstream reply relayed as `_passthrough` relays one: status, headers (minus hop-by-hop and the
        upstream's Server / Date) and body unchanged, plus X-Callo-Slot."""
        self.send_response(resp.status, resp.reason)
        self._relay_headers(resp)
        self.send_header("X-Callo-Slot", label)
        self.end_headers()
        self.wfile.write(payload)

    def _unpinned(self, obj: dict, body: bytes, *, count: str | None = None) -> None:
        """Forward a parsed routed body unpinned: untouched, except a client id_slot that lands on an S1-reserved slot
        is dropped, so an agent can never pin onto the System One slot. A body too deep to serialise again after
        that strip is refused with 400 rather than forwarded with the reserved id_slot, and
        counted as `unrouted_refused` only: `count` and `id_slot_rerouted` are bumped once the body is really
        forwarded."""
        router = self.cfg.router
        rerouted = router.normalise_id_slot(obj.get("id_slot")) in router.reserved
        if rerouted:
            del obj["id_slot"]
            try:
                body = _encode(obj)
            except RecursionError:
                self.cfg.stats.bump("unrouted_refused")
                self._send_json(400, {"error": {"message": "request body is nested too deeply to route; send it "
                                                           "without id_slot", "type": "callo"}})
                return
            self.cfg.stats.bump("id_slot_rerouted")
        if count is not None:
            self.cfg.stats.bump(count)
        self._passthrough(body=body)

    def _systemone(self) -> None:
        self._json_service(self.cfg.systemone, oversize="unrouted_large",  # type: ignore[arg-type]
                           request_log=self.cfg.request_log)

    def _think_probe(self) -> None:
        """POST /callo/think-probe: 404 when the engine is off, 403 from a non-loopback peer
        without --allow-think-probe, then the probe service with /v1/systemone's body handling and cancellation."""
        cfg = self.cfg
        if cfg.think_probe is None:
            self._refuse(404, "no think probe: callo's engine is off (a native decision model)")
            return
        if not cfg.allow_think_probe and not _is_loopback(self.client_address[0]):
            cfg.stats.bump("think_probe_refused")
            self._refuse(403, "/callo/think-probe answers loopback peers only; start callo with --allow-think-probe "
                              "to open it")
            return
        self._json_service(cfg.think_probe)

    def _json_service(self, fn: SystemOneFn, *, oversize: str | None = None,
                      request_log: RequestLog | None = None) -> None:
        """The body handling /v1/systemone and the think probe share: a Content-Length body (411 / 400 otherwise),
        413 over the routed-body cap (`oversize` names its counter), 100 Continue, 422 when it is not JSON, then the
        service with `cancel` / `watch`, and no reply when it says NO_REPLY (the client left). With
        `request_log` every request with a body is logged once its reply is written, or when its client left;
        the service fills in its slot, limits and answers."""
        n = self._body_length()
        if n is None:
            return
        rec: dict | None = {} if request_log is not None else None
        if n > self.cfg.max_routed_body:
            if oversize is not None:
                self.cfg.stats.bump(oversize)
            self._send_json(413, {"detail": f"body exceeds {self.cfg.max_routed_body} bytes"})
            self._drain_body(n)   # discard the (bounded) oversized body so the 413 is not lost to a reset
            self._log_request(request_log, rec, 413)
            return
        self._expect_continue()
        raw = self.rfile.read(n)
        if rec is not None:
            rec["body_sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            obj = json.loads(raw)
        except (ValueError, RecursionError):  # JSONDecodeError/UnicodeDecodeError are ValueErrors; deep nesting recurses
            self._send_json(422, {"detail": "body is not JSON"})
            self._log_request(request_log, rec, 422)
            return
        code, body, extra = self._call_cancellable(fn, obj, log=rec)
        if code == NO_REPLY:            # the client left during a deliberate request: nobody to answer
            log.info("POST %s from %s: the client left during a deliberate readout; no reply (counted as cancelled)",
                     self.path.split("?", 1)[0], self.client_address[0])
            self._log_request(request_log, rec, "cancelled")
            return
        try:
            self._send_json(code, body, extra=extra)
        finally:                        # a reply the client no longer reads is still logged, with its status
            self._log_request(request_log, rec, code)

    def _log_request(self, request_log: RequestLog | None, rec: dict | None, status: int | str) -> None:
        """One line, after the reply's last byte. A failed write is logged, never the request's fault."""
        if request_log is None or rec is None:
            return
        now = time.time()
        try:
            request_log.write({**rec, "status": status, "t_arr_unix_ms": round(1000 * self.t_arr_unix, 3),
                               "t_written_unix_ms": round(1000 * now, 3),
                               "served_ms": round(1000 * (time.monotonic() - self.t_arr), 3)})
        except (OSError, ValueError) as e:
            log.warning("--request-log %s: cannot write: %s", request_log.path, e)

    def _refuse(self, code: int, detail: str) -> None:
        """A refusal sent before the body is read, which is then drained (bounded, `_drain_body`) so closing the
        socket does not reset the connection before the client reads the refusal. A chunked or unreadable
        length is left alone: nothing to drain by length."""
        self._send_json(code, {"detail": detail})
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        self._drain_body(n)

    def _call_cancellable(self, fn: SystemOneFn, obj: Any, *, log: dict | None = None) -> tuple[int, dict, dict[str, str]]:
        """Call a service with a `cancel` event and a lazy `watch()` that starts the client-disconnect watcher: the
        service calls `watch()` only for a deliberate request, so budget-0 requests keep the plain
        behaviour (no watcher). Without `systemone_cancel` the service is called with no cancel event. `t_arr` always
        goes along; `log`, the request-log record the service fills in, only when the log is on."""
        headers = dict(self.headers.items())
        if not self.cfg.systemone_cancel:
            return fn(obj, headers)
        cancel = threading.Event()
        watchers: list[DisconnectWatcher] = []

        def watch() -> None:
            w = DisconnectWatcher(self.connection, None, on_eof=cancel.set)
            w.start()
            watchers.append(w)

        try:
            if log is not None:
                return fn(obj, headers, cancel=cancel, watch=watch, t_arr=self.t_arr, log=log)
            return fn(obj, headers, cancel=cancel, watch=watch, t_arr=self.t_arr)
        finally:
            for w in watchers:
                w.stop()

    def _systemone_native(self) -> None:
        """/v1/systemone for a NATIVE decision model. Native takes images only; it would
        drop top-level audio/videos and answer without them, and render audio/video state parts into its prompt as
        text. A body callo can read is checked (`native_media_refusal`) and refused with 422, nothing sent upstream;
        anything else goes through as the ORIGINAL bytes: a body that is not JSON, and (unscanned, counted as
        `native_unscanned`) one over the routed-body cap or with a Content-Encoding. The dispatch runs it whenever
        `cfg.systemone` is None; `serve` leaves it None exactly for a native decision model, and for any other
        `ProxyConfig` without an engine the check is harmless."""
        cfg = self.cfg
        n = self._body_length()
        if n is None:
            return
        self._expect_continue()
        if n > cfg.max_routed_body:
            cfg.stats.bump("native_unscanned")
            self._passthrough(stream_len=n)
            return
        body = self.rfile.read(n)
        if self.headers.get("Content-Encoding"):
            cfg.stats.bump("native_unscanned")
            self._passthrough(body=body)
            return
        try:
            obj = json.loads(body)
        except (ValueError, RecursionError):    # UnicodeDecodeError and JSONDecodeError are ValueErrors
            obj = None
        refusal = native_media_refusal(obj)
        if refusal is not None:
            cfg.stats.bump("native_media_refused")
            self._send_json(422, {"detail": refusal})
            return
        refusal = native_think_refusal(obj)          # native has no deliberate readout
        if refusal is not None:
            cfg.stats.bump("native_think_refused")
            self._send_json(422, {"detail": refusal})
            return
        self._passthrough(body=body)

    def _health(self) -> None:
        u = urlsplit(self.cfg.upstream)
        conn = http.client.HTTPConnection(u.hostname or "127.0.0.1", u.port, timeout=5)
        try:
            conn.request("GET", "/health")
            resp = conn.getresponse()
            data, status, ctype = resp.read(), resp.status, resp.getheader("Content-Type") or "application/json"
        except (OSError, http.client.HTTPException) as e:
            self._send_json(503, {"status": "error", "detail": f"upstream unreachable: {e}"})
            return
        finally:
            conn.close()
        self._send_raw(status, data, ctype)

    # ---- the relay ----------------------------------------------------------------------------------

    def _passthrough(self, body: bytes | None = None, *, stream_len: int | None = None, slot: str = "-") -> None:
        cfg = self.cfg
        if body is None and stream_len is None:
            n = self._body_length()
            if n is None:
                return
            self._expect_continue()
            if n > cfg.max_routed_body:
                stream_len = n
            else:
                body = self.rfile.read(n)
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_BY_HOP and k.lower() not in ("host", "content-length", "expect")}
        send_body: Any
        if stream_len is not None:
            headers["Content-Length"] = str(stream_len)
            send_body = _LimitedReader(self.rfile, stream_len)
        elif body:
            headers["Content-Length"] = str(len(body))
            send_body = body
        else:
            send_body = None
        conn = self._open_upstream()     # connect BEFORE the watcher starts, so conn.sock exists for it to shut down
        if conn is None:
            return
        watcher = None
        if stream_len is None and not self.headers.get("X-Conversation-Id"):
            watcher = DisconnectWatcher(self.connection, conn)
            watcher.start()
        try:
            try:
                conn.request(self.command, self.path, body=send_body, headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                if watcher is not None and watcher.tripped:
                    return
                self._upstream_502(e)
                return
            if watcher is not None and watcher.tripped:
                return
            self.send_response(resp.status, resp.reason)
            self._relay_headers(resp)
            self.send_header("X-Callo-Slot", slot)
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)   # wfile is unbuffered: each upstream chunk reaches the client at once
            except (OSError, http.client.HTTPException):
                # The client left (a failed write, or the watcher shut the upstream socket down mid-chunk) or the
                # upstream died mid-body. Either way closing conn below frees the slot (llama-server aborts on close).
                cfg.stats.bump("relay_aborted")
        finally:
            if watcher is not None:
                watcher.stop()
            conn.close()

    # ---- the upstream connection (shared by _passthrough and the transcription handler) ---------

    def _open_upstream(self, label: str = "-") -> http.client.HTTPConnection | None:
        """A connection to the upstream, connected within `connect_timeout` and then with no read timeout (see
        ProxyConfig.connect_timeout); None after answering the 502 `_upstream_502` sends when it cannot connect."""
        u = urlsplit(self.cfg.upstream)
        conn = http.client.HTTPConnection(u.hostname or "127.0.0.1", u.port, timeout=self.cfg.connect_timeout)
        try:
            conn.connect()
            conn.sock.settimeout(None)
        except OSError as e:
            self._upstream_502(e, label)
            return None
        return conn

    def _upstream_502(self, e: BaseException, label: str = "-") -> None:
        """callo's one 502 for an upstream that cannot be reached or fails during the request, counted as
        `upstream_errors`."""
        self.cfg.stats.bump("upstream_errors")
        self._send_json(502, {"error": {"message": f"upstream unreachable: {e}", "type": "callo_upstream"}},
                        extra={"X-Callo-Slot": label})

    def _relay_headers(self, resp: http.client.HTTPResponse, drop: frozenset[str] | tuple[str, ...] = ("server", "date")
                       ) -> None:
        """The upstream reply's headers, minus hop-by-hop and `drop`: by default its Server and Date, which
        send_response already wrote as callo's own (relaying them would duplicate them)."""
        for k, v in resp.getheaders():
            if k.lower() not in HOP_BY_HOP and k.lower() not in drop:
                self.send_header(k, v)

    # ---- replies ------------------------------------------------------------------------------------

    def _send_raw(self, code: int, data: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in {"X-Callo-Slot": "-", **(extra or {})}.items():   # every callo reply carries the header
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, code: int, obj: dict, extra: dict[str, str] | None = None) -> None:
        self._send_raw(code, json.dumps(obj).encode(), "application/json", extra)


def make_server(cfg: ProxyConfig | None, *, host: str = "127.0.0.1", port: int = 8180) -> ThreadingHTTPServer:
    """A bound, listening server (a busy port raises OSError here). With `cfg` None it answers every request with
    llama-server's loading 503 until `set_config` installs the real config, so `serve` can claim its port and
    start serving before a model load that can take minutes."""
    handler = type("CalloHandler", (ProxyHandler,), {"cfg": cfg})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def set_config(srv: ThreadingHTTPServer, cfg: ProxyConfig) -> None:
    """Swap in the real config: a single attribute store, so a request sees either the loading 503 or `cfg`."""
    srv.RequestHandlerClass.cfg = cfg  # type: ignore[attr-defined]


# ---- POST /v1/systemone ----------------------------------------------------------------------

_STATUS: tuple[tuple[type[CalloError], int], ...] = ((MediaRejected, 422), (NoMedia, 501), (MediaTooLarge, 413),
                                                      (ThinkTooLarge, 413), (ThinkTimeout, 504),
                                                      (UpstreamUnavailable, 503), (TooManyOptions, 422))


def _status(e: CalloError) -> int:
    """422 for media the upstream could not use, 501 for a media kind it lacks, 413 for a media
    request that exceeds the context, 503 for an upstream that is down or failing; 413 for a
    budget that does not fit the slot context, 504 for a generation that timed out; 422 for a question
    with more options than the model's two-letter labels cover (`TooManyOptions`); every other deliberate error (a
    readout that could not be made, an upstream 4xx) is a 500. A client that left (`Cancelled`) is not a status: only
    `SystemOneService._on_slot` catches it, before this, and answers NO_REPLY (nothing is sent), so no other caller
    can put a 499 on the wire."""
    return next((code for kind, code in _STATUS if isinstance(e, kind)), 500)


def _lane(headers: Mapping[str, str] | None) -> str | None:
    """X-Callo-Conversation, read on /v1/systemone too. Only that header names a lane; its value is
    stripped, and an empty or blank one names none."""
    for k, v in (headers or {}).items():
        if k.lower() == "x-callo-conversation" and v.strip():
            return v.strip()
    return None


def _think_fields(qr: QuestionReadout) -> dict:
    """The X-Callo-Timings fields of a question read at a budget > 0, appended after `passes`,
    with `think_close` in serve's words (`natural` | `budget` | `deadline` | `eos`); then, for a question
    served under a deadline, `think_deadline_ms`, `elapsed_ms`, `deadline_met`, and for one planned to think
    `chunks` and `deadline_fallback` (`budget0`, only when it fell back); a budget-0 question served under a request
    deadline because the model cannot chunk or the request has media carries `deadline_fallback` `not_chunkable` /
    `media`, and an early-exit question without a deadline whose cap left no context to think in carries
    `deadline_fallback` `budget0` alone. {} for a budget-0 question without a deadline, whose entry stays
    byte-identical to a plain readout's.
    Then, for a question served with early exit, `converge` (the target z), `converge_rule` ([n, y]) and `probes`; one
    that asked and was refused early exit carries `converge` alone (`unreachable` | `not_chunkable` | `media`). Last,
    on every deliberate question of a request early exit's gate was read for, `converge_gate` (`GateRead.fields`); a
    question the gate skipped carries no `converge` key."""
    out: dict = {}
    if qr.think is not None:
        out = {"think_budget": qr.think_budget, "think_tokens": qr.think_k, "think_close": think_close_of(qr),
               "gen_ms": round(qr.think.gen_wall_ms, 1), "gen_prompt_n": qr.think.gen_prompt_n,
               "read_prompt_n": qr.read_prompt_n, "temperature": qr.think_T, "calibrated": qr.calibrated}
    if qr.deadline_ms is not None:
        elapsed = qr.elapsed_ms
        out.update({"think_deadline_ms": qr.deadline_ms, "elapsed_ms": None if elapsed is None else round(elapsed, 1),
                    "deadline_met": elapsed is not None and elapsed <= qr.deadline_ms})
        if qr.chunks is not None:
            out["chunks"] = qr.chunks
    if qr.deadline_fallback is not None:      # also early exit's budget-0 fallback without a deadline
        out["deadline_fallback"] = qr.deadline_fallback
    if qr.converge is not None:
        out["converge"] = qr.converge
        if qr.converge_rule is not None:
            out["converge_rule"] = list(qr.converge_rule)
            out["probes"] = qr.probes
    if qr.converge_gate is not None:
        out["converge_gate"] = qr.converge_gate
    return out


def _wide_entry(r: Readout) -> dict:
    """X-Callo-Timings' wide-choice fields, after `passes`: the unread first-letter mass (6 decimals) and the split
    passes. {} for a question of 26 or fewer options, whose entry stays byte-identical."""
    return {"unread": round(r.unread_mass, 6), "split": r.split_passes} if r.wide else {}


def timings_header(readouts: Sequence[QuestionReadout]) -> str:
    """X-Callo-Timings: what llama-server reported for each question's first pass, so a client can log `cache_n` /
    `prompt_n` per request, `miss`, how many options were floored at the top-k bound, which
    `callo calibrate --bias` needs to exclude such questions from the fit, and `passes`, the upstream
    scoring calls the readout made (a video is decoded again on every one); a question above 26 options adds `unread`
    and `split` after it. ASCII JSON (`ensure_ascii`), so any
    question id is a legal header value.

    Never longer than `TIMINGS_MAX_BYTES`: a request may carry any number of questions and any qid, and an
    oversized header turns a 200 into a response the client cannot parse. The leading entries are kept, in request
    order, while they fit; when any are dropped the last element is `{"truncated": k}`, k the questions omitted
    (`[{"truncated": n}]` when not even the first entry fits)."""
    entries = [json.dumps({"q": qr.qid, "kind": qr.readout.kind, "cache_n": qr.readout.cache_n,
                           "prompt_n": qr.readout.prompt_n, "prompt_ms": qr.readout.prompt_ms,
                           "tokens": qr.readout.prompt_tokens, "stop": qr.readout.stop, "ms": round(qr.ms, 1),
                           "miss": qr.readout.topk_miss, "passes": qr.readout.passes,   # miss, then passes
                           **_wide_entry(qr.readout),                                   # wide only
                           **_think_fields(qr)},                                        # B > 0 only
                          separators=(",", ":")) for qr in readouts]
    full = "[" + ",".join(entries) + "]"
    if len(full) <= TIMINGS_MAX_BYTES:      # ASCII: characters are bytes
        return full
    kept, used = 0, 1                       # used: "[" plus each kept entry and its trailing comma
    while kept < len(entries) and (used + len(entries[kept]) + 1
                                   + len(f'{{"truncated":{len(entries) - kept - 1}}}]') <= TIMINGS_MAX_BYTES):
        used += len(entries[kept]) + 1
        kept += 1
    return "[" + "".join(e + "," for e in entries[:kept]) + f'{{"truncated":{len(entries) - kept}}}]'


@dataclass(frozen=True)
class ServePlan:
    """One request's resolution: each limit resolves on its own, the request's field
    winning when present (an explicit 0 turns that limit off), else the serve default.

    | budget          | deadline | early exit z         | serve                                             |
    |-----------------|----------|----------------------|---------------------------------------------------|
    | 0, not explicit | 0        | off                  | the readout                                       |
    | B > 0           | 0        | off                  | one generation (`think_T`, `calibrated`)          |
    | 0, not explicit | D > 0    | any                  | chunked, cap B* (profile's largest budget, else --think-max; <= --think-max) |
    | B > 0           | D > 0    | any                  | chunked, cap B                                    |
    | explicit 0      | D > 0    | any                  | the readout, timed against D                      |
    | 0, not explicit | 0        | a rule reaches z     | chunked every K, cap B*                           |
    | B > 0           | 0        | a rule reaches z     | chunked every K, cap B                            |
    | explicit 0      | 0        | any                  | the readout                                       |

    "off": z is 0, or early exit is refused for the request (`converge_status`: `unreachable` | `not_chunkable` |
    `media`). With a deadline, early exit rides on the chunked rows (probing every K when a rule reaches z); an
    explicit budget 0 never thinks.

    A media request with a deadline, and one on a model that cannot chunk (no frame, or a close tag of several tokens,
    `not_chunkable`), is served at budget 0 and timed.

    `gate` (experimental): early exit runs (`converge`) and its gate is on for the request (`think_converge_gate`, else
    --think-converge-gate), so one gate read precedes the request's thinking, and a yes drops `converge` for it."""
    budget: int
    deadline_ms: int
    chunked: bool
    cap: int
    think_T: float | None = None
    calibrated: bool = True
    fallback_reason: str | None = None    # why a deadline request got no thinking: `not_chunkable` | `media`
    converge: ServeRule | None = None     # early exit's rule, when it runs
    converge_status: str | None = None    # why early exit, asked for, is off: `unreachable` | `not_chunkable` | `media`
    gate: bool = False                    # early exit's gate is read before thinking (only with `converge`)

    @property
    def not_chunkable(self) -> bool:
        return self.fallback_reason == "not_chunkable"

    @property
    def deliberate(self) -> bool:
        return self.chunked or self.budget > 0


def _log_questions(req: Request, readouts: Sequence[QuestionReadout]) -> list[dict]:
    """Each question's request-log entry, in request order: its chosen label, close, think tokens, elapsed
    time and prompt tokens; never its text."""
    by_qid = {q.qid: q for q in req.questions}
    out = []
    for qr in readouts:
        q = by_qid.get(qr.qid)
        p = qr.readout.probs
        deliberate = qr.think is not None or qr.think_close == "budget0"
        out.append({"qid": qr.qid, "answer": q.keys[max(range(len(p)), key=p.__getitem__)] if q is not None else None,
                    "think_close": think_close_of(qr) if deliberate else None, "k": qr.think_k,
                    "elapsed_ms": None if qr.elapsed_ms is None else round(qr.elapsed_ms, 3),
                    "prompt_tokens": qr.readout.prompt_tokens})
    return out


class SystemOneService:
    """One request -> the questions answered sequentially on ONE reserved slot (so prefix reuse, when enabled,
    stays in one place); separate requests round-robin across the reserved slots. Before dispatch, the S1
    fallback checks the router's own accounting and a /slots snapshot (<= 250 ms old): a reserved slot busy
    with a task callo did not send (a passthrough /v1/messages, an embedding, anything when agent pinning is
    off) is skipped for the lowest-id idle slot, so a classification never waits for that task's whole decode.

    Status codes follow Kev's RemotePredictor, which drops 4xx records and retries 5xx: 422 only for an invalid
    request; 503 when the upstream is unreachable, and on the text path for any upstream HTTP error too. A media
    request's (image, audio, video or mixed) upstream HTTP error maps so: 422 for media the
    upstream could not load, 413 for a request that exceeds the context, 501 for a kind the upstream lacks (also
    `check_request`'s refusal, before a slot is taken), 500 for another 4xx, 503 for another 5xx. Kev drops a 413 or
    422 record, but a 501 is a 5xx, so Kev retries it like a 503: the distinct status names a configuration fault
    (the upstream's --mmproj) that a retry cannot fix, for an operator or a status-aware client. 500 with detail for
    a readout failure and for ANY other exception, never an escaped traceback.

    Think deadlines: a request with a deadline in effect (its `think_deadline_ms`, else `think_deadline`) is
    planned by `think_plan` (`ServePlan`), and its deliberate questions are chunked traces, each in an equal share of
    the time left, sized by the server-wide `estimators`, read at the temperature, and answered at
    budget 0 when the share cannot hold one token of thinking. The clock starts at `t_arr`, when the handler
    received the request.

    The body comes from `Engine.answer_parsed`, the code `callo.engine.local_engine` users run too;
    media errors map as above (`_status`); a request with X-Callo-Conversation uses its lane's slot.

    Think budgets: each request's effective budget is its `think_budget`, else `think_default`; a
    deliberate request (budget > 0) is routed with `deliberate=True`, takes the slot's S1 section exclusively per
    question, is read at the profile's T_B (nearest lower budget when uncalibrated; T = 1 without a profile) and is
    watched for a client disconnect for its whole duration (the handler's `watch()`), which cancels the generation
    upstream and ends the request with no reply. Budget-0 requests take the section shared and keep the behaviour:
    no watcher (a client that gives up frees nothing upstream)."""

    def __init__(self, engine: Engine, router: SlotRouter, stats: Stats, *, model_name: str,
                 slots_snapshot: StampedSlotsFn | None = None, fallback: bool = True, think_default: int = 0,
                 think_max: int = 4096, think_profile: ThinkProfile | None = None, s1_timeout: float = 900.0,
                 think_deadline: int = 0, estimators: Estimators | None = None, think_converge: float = 0.0,
                 converge_rules: ServeRules | None = None, think_converge_gate: bool = False):
        self.engine, self.router, self.stats = engine, router, stats
        self.model_name, self._slots, self.fallback = model_name, slots_snapshot, fallback
        self.think_default, self.think_max, self.think_profile = think_default, think_max, think_profile
        # the serve-wide deadline, the deadline-only cap B* and the estimators (shared with the engine,
        # which records its budget-0 first passes in them; the profile's latency block and T_0 p95 seed them)
        self.think_deadline = think_deadline
        self.deadline_cap = deadline_cap(think_profile.budgets if think_profile is not None else None, think_max)
        # early exit: the serve default z (0: off) and the profile's rules (None: no usable convergence block, so every
        # request that asks is `unreachable`; serve refuses --think-converge on such a profile before it starts).
        # `callo serve` passes the rules it parsed once; without them the service parses the profile's block itself
        self.think_converge = think_converge
        self.think_converge_gate = think_converge_gate      # early exit's gate, the serve default (experimental)
        self.converge_rules: ServeRules | None = (
            converge_rules if converge_rules is not None else
            serve_rules(think_profile.data.get("converge"))[0] if think_profile is not None else None)
        if estimators is None:
            estimators = (getattr(engine, "estimators", None)        # a test's stand-in engine has none
                          or Estimators.from_profile(think_profile))
        self.estimators = estimators
        engine.estimators = estimators
        if stats.estimators is None:                # /callo/stats think.deadline reads its windows
            stats.estimators = estimators
        # the S1 section wait is bounded by it: Section.acquire needs a finite timeout (inf overflows the wait)
        problem = s1_timeout_problem(s1_timeout)
        if problem is not None:
            raise InvalidInput(problem)
        self.s1_timeout = float(s1_timeout)

    def _processing(self) -> tuple[Callable[[int], bool] | None, float | None]:
        """A PURE predicate over a `/slots` snapshot fetched NOW, before the router lock is taken (the router calls
        it under its lock, so it must not do I/O or call back into the router), and the time that snapshot was
        fetched, so the router can ignore what it shows for a slot callo released since. None: no fallback check.
        The check is advisory, so it fails open: a broken snapshot is counted and never fails a classification."""
        if not self.fallback or self._slots is None:
            return None, None
        try:
            snap, as_of = self._slots()
            if not snap:
                return None, None
            busy = frozenset(s.get("id") for s in snap if isinstance(s, dict) and s.get("is_processing"))
        except Exception:  # noqa: BLE001 -- advisory input; any failure means "no fallback", not a dropped request
            self.stats.bump("slots_snapshot_error")
            return None, None
        return busy.__contains__, as_of

    def __call__(self, body: Any, headers: Mapping[str, str] | None = None) -> tuple[int, dict, str]:
        """The shape: (status, body, X-Callo-Slot label)."""
        code, out, extra = self.respond(body, headers)
        return code, out, extra["X-Callo-Slot"]

    def think_plan(self, req: Request) -> ServePlan | str:
        """The request's `ServePlan`, or the 422 detail. A media request inherits budget 0 from the
        default (counted) and is refused when it asks; under a deadline it is served at budget 0 and timed."""
        budget = req.think_budget if req.think_budget is not None else self.think_default
        deadline = req.think_deadline_ms if req.think_deadline_ms is not None else self.think_deadline
        z = req.think_converge if req.think_converge is not None else self.think_converge
        if req.think_budget is not None and req.think_budget > self.think_max:
            return f"think_budget {req.think_budget} exceeds this server's --think-max {self.think_max}"
        if req.media and budget > 0:
            if req.think_budget:
                return "deliberate readouts are text-only in this version"
            self.stats.bump_think("skipped_media")
            budget = 0
        if budget > 0 and self.engine.think is None:
            return f"this model's chat template has no thinking mode ({self.engine.think_reason or 'unknown'}); it " \
                   f"is served at budget 0 only"
        rule, status = self._converge(req, z)
        gate = rule is not None and (req.think_converge_gate if req.think_converge_gate is not None
                                     else self.think_converge_gate)
        if deadline > 0 and not req.media and req.think_budget != 0:
            frame = self.engine.think
            if frame is None or not frame.tag_single:           # a deadline is a limit, not a demand
                self.stats.bump_deadline("not_chunkable")
                return ServePlan(0, deadline, False, 0, fallback_reason="not_chunkable", converge_status=status)
            return ServePlan(budget, deadline, True, budget if budget > 0 else self.deadline_cap, converge=rule,
                             converge_status=status, gate=gate)
        if rule is not None:                                     # early exit without a deadline: chunked every K
            return ServePlan(budget, 0, True, budget if budget > 0 else self.deadline_cap, converge=rule, gate=gate)
        if budget == 0:
            media_fb = "media" if deadline > 0 and req.media and req.think_budget != 0 else None
            return ServePlan(0, deadline, False, 0, fallback_reason=media_fb, converge_status=status)
        if self.think_profile is not None:
            temperature, calibrated = self.think_profile.temperature_for(budget)
        else:
            temperature, calibrated = 1.0, False
        if not calibrated:
            self.stats.bump_think("uncalibrated")
        return ServePlan(budget, 0, False, budget, temperature, calibrated, converge_status=status)

    def _converge(self, req: Request, z: float) -> tuple[ServeRule | None, str | None]:
        """The rule serving z for this request, or why early exit is off: (None, None) when z is 0 or the request opts
        out of thinking (an explicit think_budget 0); `media`; `not_chunkable` (no frame, or a close tag of several
        tokens); `unreachable` (no usable convergence block, or no rule reaches z)."""
        if not z or req.think_budget == 0:
            return None, None
        if req.media:
            return None, "media"
        frame = self.engine.think
        if frame is None or not frame.tag_single:
            return None, "not_chunkable"
        rule = self.converge_rules.for_z(z) if self.converge_rules is not None else None
        return (rule, None) if rule is not None else (None, "unreachable")

    @staticmethod
    def _stamp_converge(readouts: list[QuestionReadout], before: int, status: str | None) -> None:
        """Early exit asked for and off for this request: `converge` = why, on each question that could think."""
        if status is None:
            return
        for j in range(before, len(readouts)):
            if readouts[j].readout.stop != "single_option" and readouts[j].converge is None:
                readouts[j] = dataclasses.replace(readouts[j], converge=status)

    def respond(self, body: Any, headers: Mapping[str, str] | None = None, *, cancel: threading.Event | None = None,
                watch: Callable[[], None] | None = None, t_arr: float | None = None,
                log: dict | None = None) -> tuple[int, dict, dict[str, str]]:
        """The proxy's entry point (`SystemOneFn`): (status, body, reply headers). `cancel` / `watch` come from the
        handler: `watch()` starts the client-disconnect watcher, called only for a deliberate request; a set
        `cancel` ends the request with NO_REPLY. `t_arr`: `time.monotonic()` when the handler received the request,
        before reading its body; None is now. `log`: the request-log record, filled in with the slot, the resolved
        limits and each question's answer."""
        t_arr = time.monotonic() if t_arr is None else t_arr
        extra = {"X-Callo-Slot": "-"}
        try:
            req = parse_request(body)
        except InvalidRequest as e:
            return 422, {"detail": str(e)}, extra
        except Exception:  # noqa: BLE001 -- a parser bug is a server fault: retryable 500, never an escape
            traceback.print_exc()   # the real exception is logged server-side; the body stays generic
            return 500, {"detail": "internal error"}, extra
        try:
            self.engine.check_request(req)          # 501 / 422 before a slot is taken or a lane bound
        except CalloError as e:
            return _status(e), {"detail": str(e)}, extra
        except Exception:  # noqa: BLE001 -- a check bug is a retryable 500, never a dropped connection
            traceback.print_exc()
            return 500, {"detail": "internal error"}, extra
        plan = self.think_plan(req)
        if isinstance(plan, str):
            return 422, {"detail": plan}, extra
        if plan.deliberate and watch is not None:
            watch()
        if log is not None:
            log.update(deadline_ms=plan.deadline_ms, budget=plan.budget)
            if plan.converge is not None or plan.converge_status is not None:
                log["converge"] = plan.converge.z if plan.converge is not None else plan.converge_status

        gates: list[GateRead] = []                  # the request's gate read, when early exit's gate ran

        def record(ms: float, readouts: list[QuestionReadout]) -> None:
            self.stats.add_request(req.kind, ms, readouts)
            for qr in readouts:                     # c_0 is the budget-0 readouts' p95
                if qr.think is None and qr.readout.stop != "single_option":
                    self.estimators.add_budget0(qr.ms)
            extra["X-Callo-Timings"] = timings_header(readouts)
            if log is not None:                     # logging never fails a served answer
                try:
                    log["questions"] = _log_questions(req, readouts)
                except Exception as e:  # noqa: BLE001
                    logging.getLogger("callo").warning("--request-log: cannot describe this request's questions: %r", e)
                    log["questions"] = []

        out = self._on_slot(headers, extra, deliberate=plan.deliberate, count=self.stats.bump_think, record=record,
                            work=lambda slot, readouts: self._answer(req, slot, readouts, plan=plan, cancel=cancel,
                                                                     t_arr=t_arr, gates=gates))
        if log is not None:
            label = out[2].get("X-Callo-Slot", "-")
            log["slot"] = int(label) if label.isdigit() else None
            if gates:
                log["converge_gate"] = gates[0].fields()
        return out

    def _on_slot(self, headers: Mapping[str, str] | None, extra: dict[str, str], *, deliberate: bool,
                 work: Callable[[int | None, list[QuestionReadout]], dict], count: Callable[[str], None],
                 record: Callable[[float, list[QuestionReadout]], None]) -> tuple[int, dict, dict[str, str]]:
        """The ONE service-layer frame /v1/systemone and POST /callo/think-probe share (calibrate measures what
        serve does): choose and acquire an S1 slot (the /slots fallback check, the lane, `deliberate`), run
        `work(slot, readouts)`, map what it raises to the think statuses, release the slot exactly once, then
        `record(ms, readouts)`. `count` takes the think counters (`cancelled`, `timeouts`, `too_large`): serve's
        `think` block, or the probe's own."""
        readouts: list[QuestionReadout] = []
        processing, as_of = self._processing()
        slot, fell_back = self.router.route_s1(processing, as_of=as_of, lane=_lane(headers),
                                               deliberate=deliberate)                     # choose + acquire
        # Nothing may run between route_s1 and this try: its finally is the ONE release, with the same `deliberate`,
        # on every exit (a reply, a section timeout, a cancel, an upstream error, a bug); a missed deliberate release
        # leaves the slot's deliberate count at 1 and every lane-less request avoids that slot for good.
        try:
            extra["X-Callo-Slot"] = str(slot) if slot is not None else "-"
            t0 = time.perf_counter()
            if fell_back:
                self.stats.bump("foreign_in_reserved")   # the reserved slot held a task callo did not send
            out = work(slot, readouts)
        except Cancelled:
            count("cancelled")
            if deliberate:                          # its upstream task outlives the cancel
                self.router.mark_draining(slot)
            return NO_REPLY, {}, extra
        except CalloError as e:
            if isinstance(e, ThinkTimeout):
                count("timeouts")
            elif isinstance(e, ThinkTooLarge):
                count("too_large")
            return _status(e), {"detail": str(e)}, extra
        except _BACKEND_DOWN as e:                  # an engine that does not wrap its upstream errors
            return 503, {"detail": f"llama-server unavailable: {e}"}, extra
        except Exception:  # noqa: BLE001 -- any bug must be a retryable 500, never a 4xx the runner drops
            traceback.print_exc()
            return 500, {"detail": "internal error"}, extra
        finally:
            if slot is not None:
                self.router.release_s1(slot, deliberate=deliberate)
        try:                                        # the slot is already released: a bug here is a plain 500
            record(1000 * (time.perf_counter() - t0), readouts)
        except Exception:  # noqa: BLE001 -- never an escaped traceback
            traceback.print_exc()
            extra.pop("X-Callo-Timings", None)
            return 500, {"detail": "internal error"}, extra
        return 200, out, extra

    def _in_section(self, slot: int | None, fn: Callable[[], dict], *, exclusive: bool,
                    cancel: threading.Event | None = None) -> dict:
        """`fn()` under the slot's S1 section, shared or exclusive; the section released exactly once. A wait
        past `s1_timeout` is the 503 `_section_timeout`; a `cancel` set while waiting is `Cancelled`. None (no
        section): the upstream has no reserved slot (-np 1), or the S1 fallback moved the request off a reserved slot
        a foreign task held, onto an idle agent slot. Both are served without one (the fallback is unchanged),
        never refused: the request's in-flight count keeps every other S1 fallback off that slot, and an agent task
        that still lands there shows up in think.readout_reprefill."""
        section = self.router.section(slot)
        if section is not None and not section.acquire(exclusive=exclusive, timeout=self.s1_timeout, cancel=cancel):
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client left while waiting for the slot")
            raise self._section_timeout()
        try:
            return fn()
        finally:
            if section is not None:
                section.release(exclusive=exclusive)

    def _answer(self, req: Request, slot: int | None, readouts: list[QuestionReadout], *, plan: ServePlan,
                cancel: threading.Event | None, t_arr: float, gates: list[GateRead] | None = None) -> dict:
        """The engine call under the slot's S1 section. Budget 0: one `answer_parsed` for the whole request,
        the section held shared (the upstream call sequence, unchanged); under a deadline its questions are timed. A
        deliberate request: one question at a time, the section held exclusively for each, released between
        questions so waiting readers get in; under a deadline each question with 2+ options is chunked in its share
        and every question is timed from `t_arr`; under early exit without a deadline each is chunked every K, untimed.
        A request that asked for early exit and was refused it has `converge` stamped with why.

        Early exit's gate (`plan.gate`): before the first question thinks, one gate read (appended to `gates`); a skip
        serves every question as the same chunked trace to the same cap without early exit (no probes), and each
        deliberate question carries the read's `converge_gate`."""
        if not plan.deliberate:
            before = len(readouts)
            out = self._in_section(slot, lambda: self.engine.answer_parsed(
                req, id_slot=slot, readouts=readouts, model_name=self.model_name), exclusive=False)
            if plan.deadline_ms:
                # stamped once answer_parsed has returned, so the section wait and the gaps between questions count;
                # the questions ran in order, so each ended its successors' readout times before the last one did
                end = 1000.0 * (time.monotonic() - t_arr)
                later = 0.0
                for i in range(len(readouts) - 1, before - 1, -1):
                    qr = readouts[i]
                    readouts[i] = dataclasses.replace(qr, deadline_ms=plan.deadline_ms, elapsed_ms=end - later,
                                                      deadline_fallback=plan.fallback_reason)
                    later += qr.ms
            self._stamp_converge(readouts, before, plan.converge_status)
            return out
        answers: dict = {}
        tokens = 0
        request_end = 1000.0 * t_arr + plan.deadline_ms if plan.deadline_ms else None

        def clock() -> float:
            return 1000.0 * time.monotonic()

        gate: dict | None = None
        thinking = sum(1 for q in req.questions if len(q.labels) > 1)
        if plan.gate and thinking:                  # only a request that will think
            wait = None                             # under a deadline: the first question's own bound, T_q - c_0
            if request_end is not None:
                wait = max(0.0, (share_end(clock(), request_end, thinking) - clock() - self.estimators.c0()) / 1000.0)
            g = self._gate(req, slot, cancel, wait_s=wait)
            if gates is not None:
                gates.append(g)
            gate = g.fields()
            if g.skip:
                plan = dataclasses.replace(plan, converge=None, gate=False)

        for i, q in enumerate(req.questions):
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client left during the deliberate request")
            one = dataclasses.replace(req, questions=(q,))
            before = len(readouts)
            if plan.chunked and len(q.labels) > 1:
                share = None
                if request_end is not None:
                    remaining = sum(1 for x in req.questions[i:] if len(x.labels) > 1)
                    share = share_end(clock(), request_end, remaining)
                out = self._deadline_question(one, slot, readouts, plan=plan, share_end_ms=share,
                                              request_end_ms=request_end, clock=clock, cancel=cancel)
            elif plan.chunked:                      # a single-option question: the shortcut, no upstream call
                out = self.engine.answer_parsed(one, id_slot=slot, readouts=readouts, model_name=self.model_name)
            else:
                out = self._in_section(slot, lambda one=one: self.engine.answer_parsed(
                    one, id_slot=slot, readouts=readouts, model_name=self.model_name, think_budget=plan.budget,
                    think_T=plan.think_T, think_calibrated=plan.calibrated, cancel=cancel), exclusive=True,
                    cancel=cancel)
            if plan.chunked:
                for j in range(before, len(readouts)):
                    qr = readouts[j]
                    if plan.deadline_ms:
                        qr = dataclasses.replace(qr, deadline_ms=plan.deadline_ms,
                                                 elapsed_ms=1000.0 * (time.monotonic() - t_arr))
                    if gate is not None and (qr.think is not None or qr.think_close == "budget0"):
                        qr = dataclasses.replace(qr, converge_gate=gate)
                    readouts[j] = qr
                    if qr.calibrated is False:
                        self.stats.bump_think("uncalibrated")
            self._stamp_converge(readouts, before, plan.converge_status)
            answers.update(out["answers"])
            tokens += out["usage"]["input_tokens"]
        return {"model": self.model_name, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}

    def _gate(self, req: Request, slot: int | None, cancel: threading.Event | None, *,
              wait_s: float | None = None) -> GateRead:
        """Early exit's gate: `GATE_QUESTION` on the request's own state (its context too), read at budget 0 through
        `Engine.answer` (the budget-0 readout path) on `slot`, the slot the request thinks on, under the section held
        shared as a budget-0 readout holds it, and with NO position-bias offsets: the 0.5 rule is on the model's own
        P(yes). Counted in `think.converge.gate`, never in the request's readouts. A yes (P(yes) >= GATE_SKIP_P) skips
        early exit. A read that fails (an upstream error, a refusal, no probabilities) skips it too and says why, in at
        most GATE_ERROR_MAX characters: the gate never fails the request. A client that left still ends it
        (`Cancelled`).

        `wait_s` (under a deadline): the section wait is bounded by the first question's T_q - c_0, the bound its own
        wait has, never --s1-timeout. On expiry the read is skipped (`GATE_BUSY`, no section timeout counted) and the
        question's own bounded wait and reroute take over.

        The read leaves the state's prefix in the slot's cache without `Estimators.note_sent` (no budget-0 readout
        notes what it sent), so the next chunk 0 usually evaluates fewer tokens than `uncached()` predicts (more, by
        the question's tail, when the slot's last noted chunk 0 was this same state and question); enough such chunk 0s
        turn the LCP evidence off and `uncached()` falls back to the whole prompt, which over-estimates the prefill."""
        t0 = time.perf_counter()
        section = self.router.section(slot)
        bound = self.s1_timeout if wait_s is None else min(wait_s, self.s1_timeout)
        if section is not None and not section.acquire(exclusive=False, timeout=bound, cancel=cancel):
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client left while waiting for the slot")
            error = GATE_BUSY if bound < self.s1_timeout else str(self._section_timeout())
            return self._gate_read(None, t0, error)
        try:
            r = self.engine.answer(req.state, GATE_QUESTION, id_slot=slot, context=req.context, position_bias=False)
            p = gate_p(r.probs)
            error = None if p is not None else "the gate read has no probability for yes"
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001 -- the gate fails safe: early exit is skipped, the request is served
            p, error = None, str(e) or type(e).__name__
        finally:
            if section is not None:
                section.release(exclusive=False)
        return self._gate_read(p, t0, error)

    def _gate_read(self, p: float | None, t0: float, error: str | None) -> GateRead:
        g = GateRead(p, 1000 * (time.perf_counter() - t0), gate_skips(p),
                     None if error is None else error[:GATE_ERROR_MAX])
        self.stats.add_gate(g)
        return g

    def _deadline_question(self, one: Request, slot: int | None, readouts: list[QuestionReadout], *, plan: ServePlan,
                           share_end_ms: float | None, request_end_ms: float | None, clock: Callable[[], float],
                           cancel: threading.Event | None) -> dict:
        """One deliberate question under a deadline: the exclusive section is waited for at most until
        T_q - c_0 (the time a budget-0 readout still needs). When that wait expires and another reserved slot holds no
        deliberate question, the question is read at budget 0 there (`route_s1(deliberate=False)`, flagged `budget0`
        and counted `rerouted`); otherwise it keeps waiting on its own slot (up to --s1-timeout in all) and is answered
        late, by then at budget 0 (its share has run out). Early exit without a deadline (`share_end_ms` None) has no
        share to bound the wait: the section is the budget path's."""
        def work() -> dict:
            conv = None
            if plan.converge is not None and self.converge_rules is not None:
                c_ms, _ = self.estimators.probe_cost()
                conv = ConvergeRun(plan.converge.z, self.converge_rules.K, plan.converge.n, plan.converge.y,
                                   c_ms or 0.0)
            # the reserve is read when the question starts: its window may have learned while it waited
            reserve = self.estimators.reserve()[0] if share_end_ms is not None else 0.0
            dl = DeadlineRun(plan.deadline_ms, plan.cap, plan.budget > 0, share_end_ms, request_end_ms, reserve,
                             self.estimators, self._temperature_rule(plan), clock, conv)
            return self.engine.answer_parsed(one, id_slot=slot, readouts=readouts, model_name=self.model_name,
                                             cancel=cancel, deadline=dl)

        if share_end_ms is None:
            return self._in_section(slot, work, exclusive=True, cancel=cancel)
        section = self.router.section(slot)
        if section is None:
            return work()
        wait = max(0.0, (share_end_ms - clock() - self.estimators.c0()) / 1000.0)
        other: list[int] = []

        def at_bound() -> bool:                 # once, at T_q - c_0, the ticket kept in its place
            if cancel is not None and cancel.is_set():
                return False
            processing, as_of = self._processing()
            got = self.router.route_quiet(processing, as_of=as_of, exclude=slot)
            if got is not None:
                other.append(got)
            return got is not None

        if not section.acquire(exclusive=True, timeout=self.s1_timeout, cancel=cancel,
                               check_after=min(wait, max(0.0, self.s1_timeout - 0.05)), on_check=at_bound):
            if other:
                return self._rerouted(one, readouts, other[0], plan)
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client left while waiting for the slot")
            raise self._section_timeout()
        try:
            return work()
        finally:
            section.release(exclusive=True)

    def _rerouted(self, one: Request, readouts: list[QuestionReadout], other: int, plan: ServePlan) -> dict:
        """The question at budget 0 on `other`, a reserved slot that held no deliberate question and that
        `SlotRouter.route_quiet` already took (released here), for this question alone (its own slot stays the
        request's)."""
        self.stats.bump_deadline("rerouted")
        before = len(readouts)
        try:
            out = self._in_section(other, lambda: self.engine.answer_parsed(
                one, id_slot=other, readouts=readouts, model_name=self.model_name), exclusive=False)
        finally:
            self.router.release_s1(other, deliberate=False)
        for j in range(before, len(readouts)):
            readouts[j] = dataclasses.replace(
                readouts[j], think_close="budget0", chunks=0, deadline_fallback="budget0",
                **({"converge": plan.converge.z, "converge_rule": (plan.converge.n, plan.converge.y), "probes": 0}
                   if plan.converge is not None else {}))
        return out

    def _temperature_rule(self, plan: ServePlan) -> Callable[[str, int, int], tuple[float, bool]]:
        """For this request: (close, k, the cap clipped to the slot context) -> (T, calibrated). This compares
        the CLIPPED cap with the deadline point's: a context that cut the cap below the point's misses T_D."""
        def rule(close: str, k: int, cap: int) -> tuple[float, bool]:
            c = plan.converge
            if close == "converged" and c is not None:
                if c.T is not None:                             # the rule's own T, fitted on its stop readouts
                    return c.T, True
                tp = self.think_profile
                return (tp.temperature_for(k)[0] if tp is not None else 1.0), False
            return deadline_temperature(close, k, cap=cap, deadline_ms=plan.deadline_ms, profile=self.think_profile)
        return rule

    def _section_timeout(self) -> UpstreamUnavailable:
        self.stats.bump_section_timeout()
        return UpstreamUnavailable(f"the slot's S1 section stayed busy for {self.s1_timeout:g} s (--s1-timeout); a "
                                   f"deliberate readout is holding it")


# ---- POST /callo/think-probe: calibrate's internal endpoint ------------------------------

MAX_PROBE_BUDGET = MAX_BUDGET   # think.py's hard cap: one source (--think-max does not apply to the probe)
MAX_PROBE_DEADLINES = 4         # --deadlines takes up to 4
PROBE_KS = (32, 64, 128)        # the convergence probe's K


@dataclass(frozen=True)
class ProbeExtras:
    """The probe's think-deadline additions, loopback-only as the probe is:

    - `deadlines` (with `reserve_ms`): after the Bmax generation, the planner runs as a simulation on the question's
      measured terms (`deadline.simulate` on `deadline.trace_terms`), and each deadline's cut k_D is read in the same ascending
      cut list as the budgets (a natural close reuses the n_gen readout; the budget-0 fallback is read nowhere:
      calibrate uses phase A's readout);
    - `reserve_read`: the reserve pre-pass: one served-like forced read (a natural trace's at n_gen - 1, read first);
    - `converge_probe` K: a forced probe readout at K, 2K, ... < n_gen of the Bmax trace, in the same cut list;
    - `live` (deadline ms | None, probe every K | None): the serve chunk loop instead of the one generation, capped at
      the one budget asked, with a forced probe readout at every K boundary (the cost pass);
    - `gate`: early exit's gate (`GATE_QUESTION`) read on the request's state INSTEAD of its question, as serve's
      `_gate` reads it (budget 0, no position-bias offsets), at T = 1 as every probe readout: budgets [0] and no
      other extra."""
    deadlines: tuple[int, ...] = ()
    reserve_ms: float | None = None
    reserve_read: bool = False
    converge_probe: int | None = None
    live: tuple[int | None, int | None] | None = None
    gate: bool = False


class ThinkProbe:
    """`{"request": <System One body>, "budgets": [0, 256, ...], "seed": N, "sampling": {...} | null, "plan": bool}` ->
    one question read at every budget, at T = 1 (budget 0 as served, position-bias offsets included), on one slot
    (`sampling` is the FULL pinned sampler dict, `check_pinned`, or null for the engine's own, `trace_sampling`, never
    an empty dict: 422 otherwise)
    under the S1 section (exclusive once when any budget is positive), with the statuses of /v1/systemone. `plan`
    renders and tokenizes the thinking-on prompt only: no slot, no section, no readout. `--think-max` does not
    apply: calibration may explore past what the server serves, which is why the handler keeps the route on
    loopback. Every readout counts in `s1.readouts` (the guard's condition 4) and the reply says how many it made."""

    def __init__(self, engine: Engine, router: SlotRouter, stats: Stats, *, s1_timeout: float = 900.0,
                 slots_snapshot: StampedSlotsFn | None = None, fallback: bool = True):
        self.engine, self.router, self.stats = engine, router, stats
        # The service-layer frame is the service's own (`_on_slot`, `_in_section`: one code path for the slot choice,
        # the section, the statuses and the release); constructing it also refuses an s1_timeout Section.acquire
        # cannot wait on.
        self._svc = SystemOneService(engine, router, stats, model_name="probe", slots_snapshot=slots_snapshot,
                                     fallback=fallback, s1_timeout=s1_timeout)
        self._live_estimators = Estimators()       # live mode's own windows, kept out of serve's

    @property
    def s1_timeout(self) -> float:
        """The section wait: the helper service's, so the wait and the 503 that names it cannot disagree."""
        return self._svc.s1_timeout

    @staticmethod
    def parse(body: Any) -> tuple[Request, list[int], int, dict | None, bool, ProbeExtras]:
        """(request, budgets, seed, sampling or None, plan, the think-deadline extras), or InvalidRequest (422)."""
        if not isinstance(body, dict):
            raise InvalidRequest("the probe body must be an object")
        req = parse_request(body.get("request"))
        if len(req.questions) != 1:
            raise InvalidRequest("the probe takes exactly one question per request")
        if req.media:
            raise InvalidRequest("deliberate readouts are text-only: the probe takes no media")
        budgets = body.get("budgets")
        if (not isinstance(budgets, list) or not budgets
                or not all(isinstance(b, int) and not isinstance(b, bool) and 0 <= b <= MAX_PROBE_BUDGET for b in budgets)
                or budgets != sorted(budgets) or len(set(budgets)) != len(budgets)):
            raise InvalidRequest(f"budgets must be a non-empty ascending list of distinct integers from 0 to "
                                 f"{MAX_PROBE_BUDGET}")
        seed = body.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0x7FFFFFFF:
            raise InvalidRequest("seed must be an integer from 0 to 2^31 - 1")
        sampling = body.get("sampling")
        if sampling is not None:                      # the full pinned dict or nothing, never a partial one
            why = check_pinned(sampling)
            if why is not None:
                raise InvalidRequest(f"sampling {why}; pass the full pinned sampler dict (every field of "
                                     f"think.SAMPLER_FIELDS and no other key) or null for the server's")
        plan = body.get("plan", False)
        if not isinstance(plan, bool):
            raise InvalidRequest("plan must be a boolean")
        return req, budgets, seed, sampling, plan, ThinkProbe._extras(body, budgets)

    @staticmethod
    def _extras(body: dict, budgets: list[int]) -> ProbeExtras:
        """The extras need a positive budget (a generation to simulate on and read from); the gate needs budget 0
        alone."""
        deadlines = body.get("deadlines")
        reserve, reserve_read = body.get("reserve_ms"), body.get("reserve_read", False)
        converge, live = body.get("converge_probe"), body.get("live")
        gate = body.get("gate", False)
        if not isinstance(gate, bool):
            raise InvalidRequest("gate must be a boolean")
        if gate:
            if budgets != [0]:
                raise InvalidRequest("gate reads early exit's gate at budget 0: budgets must be [0]")
            if deadlines is not None or reserve is not None or reserve_read is not False or converge is not None \
                    or live is not None:
                raise InvalidRequest("gate takes no deadlines, reserve_ms, reserve_read, converge_probe or live")
            return ProbeExtras(gate=True)
        if deadlines is None and reserve is None and reserve_read is False and converge is None and live is None:
            return ProbeExtras()
        if not isinstance(reserve_read, bool):
            raise InvalidRequest("reserve_read must be a boolean")
        if not any(b > 0 for b in budgets):
            raise InvalidRequest("deadlines, reserve_read, converge_probe and live need a positive budget")
        if converge is not None and (isinstance(converge, bool) or converge not in PROBE_KS):
            raise InvalidRequest(f"converge_probe must be one of {', '.join(map(str, PROBE_KS))} (tokens)")
        if live is not None:
            return ThinkProbe._live(live, budgets, deadlines, reserve, reserve_read, converge)
        if deadlines is None:
            if reserve is not None:
                raise InvalidRequest("reserve_ms is the simulated deadlines' reserve: it needs deadlines")
            return ProbeExtras(reserve_read=reserve_read, converge_probe=converge)
        if (not isinstance(deadlines, list) or not 1 <= len(deadlines) <= MAX_PROBE_DEADLINES
                or not all(isinstance(d, int) and not isinstance(d, bool) and 1 <= d <= MAX_DEADLINE_MS for d in deadlines)
                or deadlines != sorted(set(deadlines))):
            raise InvalidRequest(f"deadlines must be 1 to {MAX_PROBE_DEADLINES} ascending distinct integers from 1 to "
                                 f"{MAX_DEADLINE_MS} (ms)")
        if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not 0 <= reserve <= MAX_DEADLINE_MS:
            raise InvalidRequest(f"reserve_ms must be a number of ms from 0 to {MAX_DEADLINE_MS} (the simulated "
                                 f"deadlines' readout reserve)")
        return ProbeExtras(tuple(deadlines), float(reserve), reserve_read, converge)

    @staticmethod
    def _live(live: Any, budgets: list[int], deadlines: Any, reserve: Any, reserve_read: bool,
              converge: Any) -> ProbeExtras:
        """`live` takes one positive budget (the trace's cap) and none of the budgets mode's extras;
        `reserve_ms` optionally fixes a live deadline's reserve (else the probe's own learned windows give it)."""
        if not isinstance(live, dict) or set(live) - {"deadline_ms", "probe_every"}:
            raise InvalidRequest('live must be {"deadline_ms": D | null, "probe_every": K | null}')
        if len(budgets) != 1 or budgets[0] < 1:
            raise InvalidRequest("live takes exactly one positive budget: the trace's cap")
        if deadlines is not None or reserve_read or converge is not None:
            raise InvalidRequest("live takes no deadlines, reserve_read or converge_probe")
        dl, every = live.get("deadline_ms"), live.get("probe_every")
        if dl is not None and (isinstance(dl, bool) or not isinstance(dl, int) or not 1 <= dl <= MAX_DEADLINE_MS):
            raise InvalidRequest(f"live.deadline_ms must be null or an integer from 1 to {MAX_DEADLINE_MS}")
        if every is not None and (isinstance(every, bool) or every not in PROBE_KS):
            raise InvalidRequest(f"live.probe_every must be null or one of {', '.join(map(str, PROBE_KS))}")
        if reserve is not None and (isinstance(reserve, bool) or not isinstance(reserve, (int, float))
                                    or not 0 <= reserve <= MAX_DEADLINE_MS):
            raise InvalidRequest(f"reserve_ms must be a number of ms from 0 to {MAX_DEADLINE_MS}")
        return ProbeExtras(reserve_ms=None if reserve is None else float(reserve), live=(dl, every))

    def respond(self, body: Any, headers: Mapping[str, str] | None = None, *, cancel: threading.Event | None = None,
                watch: Callable[[], None] | None = None, t_arr: float | None = None) -> tuple[int, dict, dict[str, str]]:
        """The handler's entry point (`SystemOneFn`), with /v1/systemone's `cancel` / `watch` (`t_arr` is accepted and
        unused: the probe has no deadline of its own). Every probe counts once in /callo/stats `think_probe`:
        `requests` when answered, `errors` otherwise."""
        code, out, extra = self._respond(body, headers, cancel=cancel, watch=watch)
        self.stats.bump_probe("requests" if code == 200 else "errors")
        return code, out, extra

    def _respond(self, body: Any, headers: Mapping[str, str] | None, *, cancel: threading.Event | None,
                 watch: Callable[[], None] | None) -> tuple[int, dict, dict[str, str]]:
        extra = {"X-Callo-Slot": "-"}
        try:
            req, budgets, seed, sampling, plan, extras = self.parse(body)
        except InvalidRequest as e:
            return 422, {"detail": str(e)}, extra
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return 500, {"detail": "internal error"}, extra
        q = req.questions[0]
        try:                                    # before any render, trace or readout
            self.engine.check_coverage(q)
        except TooManyOptions as e:
            return _status(e), {"detail": str(e)}, extra
        positive = [b for b in budgets if b > 0]
        if positive and self.engine.think is None:
            return 422, {"detail": f"this model's chat template has no thinking mode "
                                   f"({self.engine.think_reason or 'unknown'}); probe budget 0 only"}, extra
        if plan:
            try:
                n = self.engine.plan_deliberate(req.state, q, context=req.context) if self.engine.think is not None \
                    else len(self.engine.scorer.tokenize(self.engine.template.render(
                        messages(req.state, q, self.engine.variant, context=req.context)),
                        add_special=True, parse_special=True))
            except CalloError as e:
                return _status(e), {"detail": str(e)}, extra
            except TemplateError as e:                  # a render failure is a ReadoutError (500 with its detail)
                return _status(ReadoutError(str(e))), {"detail": str(e)}, extra
            except _BACKEND_DOWN as e:
                return 503, {"detail": f"llama-server unavailable: {e}"}, extra
            except Exception:  # noqa: BLE001 -- never a dropped connection
                traceback.print_exc()
                return 500, {"detail": "internal error"}, extra
            return 200, {"readouts": 0, "questions": {q.qid: {"on_tokens": n, "n_ctx": self.engine.n_ctx}}}, extra
        deliberate = bool(positive)
        if deliberate and watch is not None:
            watch()

        def work(slot: int | None, readouts: list[QuestionReadout]) -> dict:
            # the whole probe under one acquisition of the section: exclusive when any budget is positive, and only
            # then cancellable while queued (as the service's deliberate questions are)
            return self._svc._in_section(slot, lambda: self._read(req, q, budgets, seed, sampling, slot, readouts,
                                                                  cancel, extras),
                                         exclusive=deliberate, cancel=cancel if deliberate else None)

        return self._svc._on_slot(headers, extra, deliberate=deliberate, work=work, count=self.stats.bump_probe,
                                  record=lambda _ms, readouts: self.stats.add_probe(readouts))

    def _read(self, req: Request, q: Any, budgets: list[int], seed: int, sampling: dict | None, slot: int | None,
              readouts: list[QuestionReadout], cancel: threading.Event | None, extras: ProbeExtras) -> dict:
        """The readouts, on the slot, under the section. A render failure is a ReadoutError, as `answer_parsed` maps
        it, so the shared frame's statuses are serve's."""
        try:
            if extras.gate:
                return self._read_gate(req, q, slot, readouts)
            if len(q.labels) < 2:
                return self._read_single(req, q, budgets, slot, readouts)
            if extras.live is not None:
                return self._read_live(req, q, budgets[0], seed, sampling, slot, readouts, cancel, extras)
            return self._read_budgets(req, q, budgets, seed, sampling, slot, readouts, cancel, extras)
        except TemplateError as e:
            raise ReadoutError(str(e)) from e

    def _read_single(self, req: Request, q: Any, budgets: list[int], slot: int | None,
                     readouts: list[QuestionReadout]) -> dict:
        """A single-option question, as serve reads it at any budget (`answer_parsed` gates the trace on 2+ options):
        the shortcut readout, no trace; every budget gets it, and it is one readout."""
        t0 = time.perf_counter()
        r = self.engine.answer(req.state, q, id_slot=slot, context=req.context, temperature=1.0)
        ms = 1000 * (time.perf_counter() - t0)
        readouts.append(QuestionReadout(q.qid, r, ms))
        fields = _readout_fields(r)
        per = {str(b): {**fields, "ms": ms} if b == 0 else
               {**fields, "k": 0, "read_ms": ms, "p_ms": None, "read_prompt_n": None, "read_cache_n": None,
                "text_fallback": False} for b in budgets}
        return {"readouts": 1, "questions": {q.qid: {"n_ctx": self.engine.n_ctx, "on_tokens": None, "trace": None,
                                                     "content": None, "budgets": per}}}

    def _read_gate(self, req: Request, q: Any, slot: int | None, readouts: list[QuestionReadout]) -> dict:
        """`gate`: `GATE_QUESTION` on the request's state and context, one budget-0 readout as serve's `_gate` reads
        it (`Engine.answer` with no position-bias offsets), timed as the budgets mode's budget 0 is (the readout's own
        wall ms), at T = 1: with two options the 0.5 rule is the argmax at any T. The reply is keyed by the request's
        question, `gate` {p: P(yes) or null, ms, sha256 (`GATE_SHA256`), prompt_n, cache_n, and `error` when there is
        no P(yes)}; an upstream failure is the probe's usual status."""
        t0 = time.perf_counter()
        r = self.engine.answer(req.state, GATE_QUESTION, id_slot=slot, context=req.context, temperature=1.0,
                               position_bias=False)
        ms = 1000 * (time.perf_counter() - t0)
        readouts.append(QuestionReadout(GATE_QID, r, ms))
        p = gate_p(r.probs)
        gate = {"p": p, "ms": ms, "sha256": GATE_SHA256, "prompt_n": r.prompt_n, "cache_n": r.cache_n,
                **({} if p is not None else {"error": "the gate read has no probability for yes"})}
        return {"readouts": 1, "questions": {q.qid: {"n_ctx": self.engine.n_ctx, "on_tokens": None, "gate": gate}}}

    def _read_live(self, req: Request, q: Any, cap: int, seed: int, sampling: dict | None, slot: int | None,
                   readouts: list[QuestionReadout], cancel: threading.Event | None, extras: ProbeExtras) -> dict:
        """`live`: the serve chunk loop (`Engine.answer_chunked`) on this slot, a deadline running from the
        probe's own start with its estimators kept out of serve's windows, a forced probe readout at every K boundary
        and the readout at the cut, every one at T = 1 (`temperature_of` is the constant (1.0, False): calibration
        reads, as the budgets mode's). Chunk i takes think_seed(seed, qid[, i]). The reply carries the trace tokens,
        every chunk's timings, the probes and the final readout; a deadline too short to think replies `close`
        "budget0" and no readout."""
        assert extras.live is not None
        deadline, every = extras.live
        est = self._live_estimators
        t0 = 1000.0 * time.monotonic()
        reserve = extras.reserve_ms if extras.reserve_ms is not None else est.reserve()[0]
        cr = self.engine.answer_chunked(
            req.state, q, cap=cap, explicit_cap=True, seed=(str(seed), q.qid),
            sampling=sampling if sampling is not None else self.engine.trace_sampling(),
            temperature_of=lambda close, k, cap: (1.0, False),
            share_end_ms=t0 + deadline if deadline is not None else None,
            request_end_ms=t0 + deadline if deadline is not None else None, reserve_ms=reserve,
            estimators=est if deadline is not None else None, id_slot=slot, context=req.context, cancel=cancel,
            probe_every=every)
        question: dict[str, Any] = {"n_ctx": self.engine.n_ctx}
        if cr is None:
            question["live"] = {"close": "budget0", "elapsed_ms": 1000.0 * time.monotonic() - t0}
            return {"readouts": 0, "questions": {q.qid: question}}
        t = cr.trace
        for br in (*cr.probes, cr.readout):
            readouts.append(QuestionReadout(q.qid, br.readout, br.read_ms, think=t, think_budget=cr.cap or cap,
                                            think_k=br.k, read_prompt_n=br.read_prompt_n, think_T=1.0,
                                            calibrated=False, text_fallback=br.text_fallback))
        question["on_tokens"] = t.prompt_tokens
        question["live"] = {"close": cr.close, "tokens": list(t.gen_tokens), "n_gen": t.n_gen, "frame_ms": t.frame_ms,
                            "elapsed_ms": 1000.0 * time.monotonic() - t0,
                            "chunks": [_chunk_fields(c) for c in cr.chunks],
                            "probes": [_cut_fields(p) for p in cr.probes], "readout": _cut_fields(cr.readout)}
        return {"readouts": len(cr.probes) + 1, "questions": {q.qid: question}}

    def _read_budgets(self, req: Request, q: Any, budgets: list[int], seed: int, sampling: dict | None,
                      slot: int | None, readouts: list[QuestionReadout], cancel: threading.Event | None,
                      extras: ProbeExtras | None = None) -> dict:
        extras = extras or ProbeExtras()
        per: dict[str, dict] = {}
        question: dict[str, Any] = {"n_ctx": self.engine.n_ctx}
        n_readouts = 0
        if 0 in budgets:
            if cancel is not None and cancel.is_set():  # a client that left while this probe took the section
                raise Cancelled("the client left before the probe's budget-0 readout")
            t0 = time.perf_counter()
            r = self.engine.answer(req.state, q, id_slot=slot, context=req.context, temperature=1.0)
            ms = 1000 * (time.perf_counter() - t0)
            readouts.append(QuestionReadout(q.qid, r, ms))
            per["0"] = {**_readout_fields(r), "ms": ms}
            n_readouts += 1
        positive = [b for b in budgets if b > 0]
        if positive:
            sims: dict[int, Any] = {}

            def cuts(t: Any) -> list[int]:
                """Each deadline simulated on this trace's own terms; its cut is read when it thinks."""
                a, step = trace_terms(t.frame_ms, t.gen_wall_ms, t.predicted_ms, t.n_gen)
                for dl in extras.deadlines:
                    sims[dl] = simulate(a_ms=a, d_ms=step, n_gen=t.n_gen, close=t.close, deadline_ms=dl,
                                        reserve_ms=float(extras.reserve_ms or 0.0), cap=positive[-1])
                return [s.k for s in sims.values() if s.close is not None]

            def more(t: Any) -> list[int]:
                """The deadline cuts and the convergence probes at K, 2K, ... < n_gen."""
                k = extras.converge_probe
                return [*(cuts(t) if extras.deadlines else []), *(range(k, t.n_gen, k) if k else [])]

            d = self.engine.answer_deliberate(req.state, q, positive, id_slot=slot, context=req.context, seed=seed,
                                              sampling=sampling if sampling is not None else self.engine.trace_sampling(),
                                              temperature=1.0, cancel=cancel,
                                              cuts_from=more if extras.deadlines or extras.converge_probe else None,
                                              pre_cut=extras.reserve_read)
            t = d.trace
            question["on_tokens"] = t.prompt_tokens
            question["trace"] = {"tokens": list(t.gen_tokens), "n_gen": t.n_gen, "close": t.close,
                                 "truncated": t.truncated, "gen_wall_ms": t.gen_wall_ms, "gen_prompt_ms": t.gen_prompt_ms,
                                 "gen_prompt_n": t.gen_prompt_n, "gen_cache_n": t.gen_cache_n,
                                 "predicted_ms": t.predicted_ms, "frame_ms": t.frame_ms}
            question["content"] = t.content
            for b in positive:
                per[str(b)] = _cut_fields(d.readouts[b])
            seen: set[int] = set()
            for k, br in sorted(d.by_k.items()):        # one readout per distinct read (an eos trace's n_gen - 1 cut
                if id(br.readout) not in seen:          # reuses n_gen's), as Deliberate.n_readouts counts them
                    seen.add(id(br.readout))
                    b = next((b for b in positive if min(b, t.n_gen) == k), k)
                    readouts.append(QuestionReadout(q.qid, br.readout, br.read_ms, think=t, think_budget=b,
                                                    think_k=br.k, read_prompt_n=br.read_prompt_n, think_T=1.0,
                                                    calibrated=False, text_fallback=br.text_fallback))
            n_readouts += d.n_readouts
            if extras.deadlines:                        # each deadline's simulated close and its readout
                question["deadlines"] = {str(dl): {"close": s.close or "budget0", "k": s.k, "chunks": s.chunks,
                                                   "elapsed_ms": s.elapsed_ms,
                                                   **(_cut_fields(d.by_k[s.k]) if s.close is not None else {})}
                                         for dl, s in sims.items()}
            if extras.converge_probe:                   # every probe, at T = 1
                question["probes"] = [_cut_fields(d.by_k[k]) for k in range(extras.converge_probe, t.n_gen,
                                                                            extras.converge_probe)]
            if extras.reserve_read:                     # one served-like forced read
                k = t.n_gen - 1 if t.close == "natural" and t.n_gen >= 2 else t.n_gen
                br = d.by_k.get(k)
                if br is not None:
                    question["reserve_read"] = {"k": k, "read_ms": br.read_ms, "read_prompt_n": br.read_prompt_n,
                                                "add_step": t.close != "forced"}
        else:
            question["on_tokens"] = None
        question["budgets"] = per
        return {"readouts": n_readouts, "questions": {q.qid: question}}


def _chunk_fields(c: Any) -> dict:
    """A live chunk (`deadline.Chunk`) as the probe replies with it."""
    return {"n_predict": c.n_predict, "n": len(c.tokens), "stop": c.stop, "wall_ms": c.wall_ms, "prompt_ms": c.prompt_ms,
            "predicted_ms": c.predicted_ms, "prompt_n": c.prompt_n, "cache_n": c.cache_n}


def _cut_fields(br: Any) -> dict:
    """A cut readout (`BudgetReadout`) as the probe replies with it."""
    return {**_readout_fields(br.readout), "k": br.k, "read_ms": br.read_ms, "p_ms": br.p_ms,
            "read_prompt_n": br.read_prompt_n, "read_cache_n": br.read_cache_n, "text_fallback": br.text_fallback}


def _readout_fields(r: Any) -> dict:
    """A readout as the probe replies with it; a wide one (above 26 options) adds `unread_mass`,
    `split_passes` and `partial_by_letter` (a list of [letter, mass] pairs)."""
    out = {"probs": list(r.probs), "log_marginals": list(r.log_marginals), "option_mass": r.option_mass,
           "stop": r.stop, "depth": r.depth, "miss": r.topk_miss, "passes": r.passes, "bias": r.bias,
           "prompt_n": r.prompt_n, "cache_n": r.cache_n, "prompt_ms": r.prompt_ms}
    if r.wide:
        out.update({"unread_mass": r.unread_mass, "split_passes": r.split_passes,
                    "partial_by_letter": [[f, m] for f, m in r.partial_by_letter]})
    return out
