"""Think deadlines, the pure half: the chunk planner that the serve loop and calibrate's simulation share, the "too
short to think" rule, the even split of a request's time, the deadline-only cap and the readout temperature per
close, and the server-wide estimators.

A deadline trace is a chain of non-streaming `/completion` calls ("chunks"), each continuing the previous one's tokens.
Before each chunk the planner sizes it from the time left in the question's share, so the readout can still finish
before the deadline; nothing here ever sleeps: a natural close returns at once, and the estimates only shrink
the next chunk.
"""
from __future__ import annotations

import math
import os
import re
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

ALPHA = 0.5                     # the share of the time left one chunk may plan to use
N_MIN = 4                       # from chunk 1 on, a chunk this short is not worth its overhead: close
C_MAX = 256                     # the largest chunk
SIM_OVERHEAD_MS = 5.0           # ô in calibrate's simulation (single-stream, no stall allowance)
MAX_DEADLINE_MS = 600_000       # think_deadline_ms and --think-deadline range over 0..600000
CONTEXT_ROOM = 16               # the deadline-only cap leaves the close suffix and the walk this much context


@dataclass(frozen=True)
class Estimate:
    """What one chunk decision uses, in ms: chunk 0's whole prefill P̂, the per-token time d̂, the per-chunk
    overhead ô and the stall allowance G_i, of which `known_ms` is the S_known part (the residual stall a
    continuation chunk shows is measured net of it)."""
    prefill_ms: float
    step_ms: float
    overhead_ms: float
    stall_ms: float = 0.0
    known_ms: float = 0.0
    probe_ms: float = 0.0   # early exit: a due probe's cost beyond ô


@dataclass(frozen=True)
class Chunk:
    """One chunk's reply, as the planner and the estimators read it."""
    tokens: tuple[int, ...]
    stop: str                           # "word" (the close tag) | "limit" (n_predict) | "eos"
    wall_ms: float
    prompt_ms: float | None
    predicted_ms: float | None
    prompt_n: int | None
    cache_n: int | None
    n_predict: int
    content: str = ""
    known_ms: float = 0.0               # S_known when the chunk was planned


@dataclass(frozen=True)
class ChunkRun:
    """A deadline trace: its tokens, its close (`natural` | `budget` | `deadline` | `eos`, or None when thinking did not
    fit: the budget-0 fallback) and its chunks."""
    tokens: tuple[int, ...]
    close: str | None
    chunks: tuple[Chunk, ...]

    @property
    def fallback(self) -> bool:
        return self.close is None

    @property
    def content(self) -> str:
        return "".join(c.content for c in self.chunks)


Generate = Callable[[int, int, tuple[int, ...], Estimate], Chunk]    # (i, n_i, the trace so far, its estimate)
Estimator = Callable[[int, Sequence[Chunk]], Estimate]               # (i, the chunks so far) -> chunk i's estimate


def feasible(left_ms: float, e: Estimate) -> bool:
    """Thinking fits the share iff R >= P̂ + ô + G_0 + d̂, i.e. one token of thinking fits. The only
    rule that sends a question to budget 0 (no assumed useful minimum). A due probe's cost never makes one token
    infeasible: the budget-0 fallback is for time, not for probes, so `probe_ms` is not read here."""
    return left_ms >= e.prefill_ms + e.overhead_ms + e.stall_ms + e.step_ms


def chunk_size(i: int, k: int, cap: int, left_ms: float, e: Estimate) -> int:
    """n_i = min(B - k, C_max, floor(alpha * (R_i - P̂·[i=0] - ô - G_i) / d̂)). Chunk 0 is at least 1 (step 2
    passed). From chunk 1 on, a time-bound n below n_min is 0: close with `deadline` at k. The cap alone never closes
    with `deadline`: two tokens left under the cap with time to spare is a 2-token chunk."""
    room = left_ms - (e.prefill_ms if i == 0 else 0.0) - e.overhead_ms - e.stall_ms - e.probe_ms
    by_time = math.floor(ALPHA * room / max(e.step_ms, 1e-6))
    if i == 0:
        return max(1, min(cap - k, C_MAX, by_time))
    if by_time < N_MIN:
        return 0
    return min(cap - k, C_MAX, by_time)


@dataclass(frozen=True)
class ConvergeRun:
    """Early exit for one question: probe every K tokens and close `converged` at the first probe whose last n probes
    share one argmax, each with max p >= y (read at T = 1); `z` the target whose rule this is; `c_ms` a probe's cost
    (calibrate's c or the live one, 0 when unknown), of which `with_probe_cost` charges the part beyond ô when a probe
    is due."""
    z: float
    K: int
    n: int
    y: float
    c_ms: float


def with_probe_cost(e: Estimate, *, i: int, k: int, cap: int, left_ms: float | None, every: int,
                    c_ms: float) -> Estimate:
    """Chunk i's estimate with a due probe's cost in `probe_ms`: a probe is due after the chunk when the chunk the
    planner would size without it reaches the next multiple of `every` below the cap. The charge is c - ô, never
    below 0: c (calibrate's, and the live one) is the probe read plus what the forced boundary adds to the next chunk,
    which includes that chunk's per-chunk overhead ô, and ô is already charged to every chunk. `left_ms` None (no
    deadline): due after every aligned chunk below the cap (the charge then sizes nothing). Whether a probe is due is
    judged on the uncharged chunk, so the charge can shrink the chunk short of the boundary (no probe is then read) or
    below n_min (a `deadline` close): a conservative over-charge, by design. `every` < 1 is a ValueError."""
    if every < 1:
        raise ValueError(f"with_probe_cost: every must be >= 1, got {every!r}")
    to_boundary = every - k % every
    if k + to_boundary >= cap:
        return e
    if left_ms is not None and chunk_size(i, k, cap, left_ms, e) < to_boundary:
        return e
    return replace(e, probe_ms=max(0.0, c_ms - e.overhead_ms))


def run_chunks(generate: Generate, estimate: Estimator, *, cap: int, share_end_ms: float, reserve_ms: float,
               clock: Callable[[], float], check: Callable[[], None] | None = None,
               after: Callable[[tuple[int, ...]], str | None] | None = None) -> ChunkRun:
    """Steps 1-5, the ONE loop the serve path and calibrate's simulation share: feasibility, then chunks until the
    trace closes (a `word` stop is `natural`, `eos` is `eos`, k = cap is `budget`) or the next chunk would be shorter
    than n_min (`deadline`). `clock()` is the current time in ms on the clock `share_end_ms` (T_q) is on; `check()`
    runs before every chunk (the service's cancel check); `after(trace so far)` after every chunk that does not close
    the trace (the probe's live mode reads a forced readout there) and may return a close (early exit's
    `converged`), which ends the trace there. Errors from `generate` propagate: callo
    never answers from a partial trace after an error. `cap` < 1 (a prompt that leaves no context to think in) is
    infeasible."""
    e0 = estimate(0, ())
    if cap < 1 or not feasible(share_end_ms - reserve_ms - clock(), e0):
        return ChunkRun((), None, ())
    gen: list[int] = []
    chunks: list[Chunk] = []
    i = 0
    while True:
        e = e0 if i == 0 else estimate(i, chunks)
        n = chunk_size(i, len(gen), cap, share_end_ms - reserve_ms - clock(), e)
        if n == 0:
            return ChunkRun(tuple(gen), "deadline", tuple(chunks))
        if check is not None:
            check()
        c = generate(i, n, tuple(gen), e)
        chunks.append(c)
        gen.extend(c.tokens)
        if c.stop == "word":
            return ChunkRun(tuple(gen), "natural", tuple(chunks))
        if c.stop == "eos":
            return ChunkRun(tuple(gen), "eos", tuple(chunks))
        if len(gen) >= cap:
            return ChunkRun(tuple(gen), "budget", tuple(chunks))
        if after is not None:
            close = after(tuple(gen))
            if close is not None:
                return ChunkRun(tuple(gen), close, tuple(chunks))
        i += 1


@dataclass(frozen=True)
class Simulated:
    """The planner run as a simulation on one question's measured terms. `close` None is the budget-0 fallback;
    `elapsed_ms` is the simulated time from the request's arrival to the end of the trace (the readout not included)."""
    close: str | None
    k: int
    chunks: int
    elapsed_ms: float


def simulate(*, a_ms: float, d_ms: float, n_gen: int, close: str, deadline_ms: float, reserve_ms: float, cap: int,
             overhead_ms: float = SIM_OVERHEAD_MS) -> Simulated:
    """`run_chunks` on one question's terms (`trace_terms`), single-stream: d = predicted / (n_gen - 1) per token
    (the decode step), a = frame + wall - n_gen x d (chunk 0's fixed part: the frame, the prefill, the first step, HTTP
    and scheduling; P̂ in the planner), ô = `overhead_ms`, G = 0, the reserve `reserve_ms`. The trace is the
    measured one: a `natural` or `eos` trace stops when the simulation reaches its n_gen tokens, a `forced` one (cut at
    Bmax) never stops on its own. Chunk 0 costs a + n·d, each later chunk ô + n·d. Only a and d enter, so compare can
    rerun it from a profile's `rows[].t`."""
    now = [0.0]
    est = Estimate(a_ms, d_ms, overhead_ms)
    stops = close in ("natural", "eos")

    def generate(i: int, n: int, so_far: tuple[int, ...], _e: Estimate) -> Chunk:
        k = len(so_far)
        take = min(n, n_gen - k) if stops else n
        now[0] += (a_ms if i == 0 else overhead_ms) + take * d_ms
        stop = ("eos" if close == "eos" else "word") if stops and k + take >= n_gen else "limit"
        return Chunk(tuple(range(k, k + take)), stop, 0.0, None, None, None, None, n)

    run = run_chunks(generate, lambda _i, _c: est, cap=cap, share_end_ms=deadline_ms, reserve_ms=reserve_ms,
                     clock=lambda: now[0])
    return Simulated(run.close, len(run.tokens), len(run.chunks), now[0])


def trace_terms(frame_ms: float, gen_wall_ms: float, predicted_ms: float | None, n_gen: int) -> tuple[float, float]:
    """(`rows[].t`): one measured trace's simulation terms (a, d) in ms, what `simulate` takes. d is
    the decode step, predicted_ms / (n_gen - 1): llama-server's predicted_ms leaves out the first sampled token, which
    is decoded in the prompt phase (as the serve estimators read a chunk 0). a is frame + wall -
    n_gen x d: the frame, the prefill, that first step, HTTP and scheduling, so a + n_gen x d is the measured frame +
    wall; never below the frame (a server whose planted timings exceed the measured wall, as the test fake's do,
    counts the larger, `think.budget_latency`'s continuity rule). A trace of fewer than 2 tokens has no step to
    measure: d = predicted / n_gen (0 for an empty trace)."""
    predicted = float(predicted_ms or 0.0)
    d = predicted / (n_gen - 1) if n_gen >= 2 else (predicted / n_gen if n_gen else 0.0)
    return frame_ms + max(0.0, gen_wall_ms - n_gen * d), d


def share_end(now_ms: float, request_end_ms: float, remaining: int) -> float:
    """Question i's share ends at T_q = now + (T_req - now) / m, m the deliberate questions (2+ options) not
    yet answered. Time a question leaves unused passes to the next: the next share is computed when it starts."""
    return now_ms + (request_end_ms - now_ms) / max(1, remaining)


def deadline_cap(profile_budgets: Iterable[int] | None, think_max: int) -> int:
    """The cap of a deadline-only request, B*: the profile's largest calibrated budget, else --think-max,
    and never above --think-max (a deadline never raises a cap). The engine clips it to the slot context per question
    (`clip_cap`), never refusing."""
    positive = [b for b in (profile_budgets or ()) if b > 0]
    return min(max(positive), think_max) if positive else think_max


def clip_cap(cap: int, n_ctx: int | None, on_tokens: int) -> int:
    """min(B*, n_ctx - len(on_tokens) - 16); unclipped when n_ctx is unknown."""
    return cap if n_ctx is None else min(cap, n_ctx - on_tokens - CONTEXT_ROOM)


def deadline_temperature(close: str, k: int, *, cap: int, deadline_ms: int, profile: Any) -> tuple[float, bool]:
    """(the readout T, calibrated) of a deliberate readout under a deadline. `profile` is a `ThinkProfile` or
    None (then T = 1, uncalibrated, as think budgets serve without one).

    1. A deadline point for D whose cap the served cap reaches: T_D, calibrated, whatever the close.
    2. A `natural`, `eos` or `budget` close at cap C: `temperature_for(C)`.
    3. A `deadline` cut at k: `temperature_for(k)` (the nearest lower calibrated budget, else the smallest positive),
       never calibrated.
    The budget-0 fallback is not deliberate: it is read at the engine's T_0 (rule 4)."""
    if profile is None:
        return 1.0, False
    point = profile.deadline_point(deadline_ms)
    if point is not None and cap >= int(point["cap"]):
        return float(point["T"]), True
    if close == "deadline":
        return profile.temperature_for(k)[0], False
    return profile.temperature_for(cap)


def _finite(v: Any) -> bool:
    """A finite JSON number (never a bool); a 400-digit int is not one, never an OverflowError (the
    twin of `think.finite_number`, which this leaf module cannot import)."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


LATENCY_FIELDS = ("frame_ms", "fixed_ms", "prefill_ms_per_token", "decode_ms_per_token", "read_natural_ms",
                  "read_forced_ms", "on_tokens")
LATENCY_STATS = ("p50", "p95", "p99")


def check_latency(block: Any) -> tuple[dict | None, str | None]:
    """A think profile's optional profile-level `latency` block as a dict, or (None, why): each known field null
    (read_forced_ms is null without an audit) or an object holding finite non-negative p50, p95 and p99. Absent or null:
    {}. Unknown fields are kept as they are."""
    if block is None:
        return {}, None
    if not isinstance(block, Mapping):
        return None, "latency must be an object"
    for name in LATENCY_FIELDS:
        v = block.get(name)
        if v is None:
            continue
        if not isinstance(v, Mapping):
            return None, f"latency.{name} must be null or an object with p50, p95 and p99"
        for stat in LATENCY_STATS:
            x = v.get(stat)
            if not _finite(x) or x < 0:
                return None, f"latency.{name}.{stat} must be a finite non-negative number, got {x!r}"
    return dict(block), None


def check_deadline_points(points: Any) -> tuple[dict[int, dict] | None, str | None]:
    """A think profile's optional `deadlines` block as {D: entry}, or (None, why): keys integers 1..600000, each
    entry an object with a finite positive T and an integer cap >= 1. Absent or null: {}."""
    if points is None:
        return {}, None
    if not isinstance(points, Mapping):
        return None, "deadlines must be an object"
    out: dict[int, dict] = {}
    for key, e in points.items():
        if not (isinstance(key, str) and re.fullmatch(r"[1-9][0-9]{0,5}", key) and int(key) <= MAX_DEADLINE_MS):
            return None, f"deadline key {key!r} must be an integer from 1 to {MAX_DEADLINE_MS}"
        t = e.get("T") if isinstance(e, Mapping) else None
        cap = e.get("cap") if isinstance(e, Mapping) else None
        if not _finite(t) or t <= 0:
            return None, f"deadline {key} must have a finite positive T, got {t!r}"
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            return None, f"deadline {key} must have an integer cap >= 1, got {cap!r}"
        out[int(key)] = dict(e)
    return out, None


# ---- the server-wide estimators --------------------------------------------------------------------

WINDOW = 2000                   # every stats window holds 2,000
READS_WINDOW = 64               # r is learned over the last 64 readouts after a `deadline` or `budget` close
RESID_WINDOW = 64               # s95_resid over the last 64 continuation chunks
LIVE_MIN = 8                    # a learned value replaces its seed after 8 live samples
RECENT_S = 2.0                  # chunk 0's d̂ is the window median over the chunks finished in the last 2 s
STEP_CAP = 3.0                  # a question's d̂ is capped at 3 x the server window's median
LAST_STEPS = 3                  # a question's d̂ is the median of its last 3 chunks
PREFILL_MIN_N = 32              # P̂'s rate learns from chunk 0s that prefilled more than 32 tokens
EWMA = 0.2                      # the weight of P̂'s and ô's EWMAs (a design choice)
LCP_WINDOW, LCP_SHARE, LCP_SLACK = 32, 0.9, 8      # pure-attention behaviour, measured on the last 32 chunk 0s
RESTORE_PROMPT_N = 2            # a chunk after a probe that evaluated <= 2 tokens restored
DEFAULT_RESERVE_MS, RESERVE_FLOOR_MS = 100.0, 20.0
DEFAULT_STEP_MS, DEFAULT_PREFILL_MS, DEFAULT_OVERHEAD_MS, DEFAULT_C0_MS = 30.0, 2.0, 5.0, 300.0


def percentile(xs: Sequence[float], q: float) -> float:
    """Nearest-rank, as /callo/stats (proxy.percentile): the ceil(q*n)-th smallest value."""
    s = sorted(xs)
    rank = -(-round(q * 10**6) * len(s) // 10**6)
    return s[max(0, min(len(s) - 1, rank - 1))]


def _p(profile_latency: Mapping | None, key: str, q: str) -> float | None:
    """A think profile's `latency.<key>.<q>` when it is a finite positive number, else None."""
    block = (profile_latency or {}).get(key)
    v = block.get(q) if isinstance(block, Mapping) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 else None


def _lcp(a: Sequence[int], b: Sequence[int]) -> int:
    return len(os.path.commonprefix([list(a), list(b)]))


class Estimators:
    """The server-wide rolling windows every reserved slot shares, so they carry the current concurrency. Thread
    safe; `clock` is `time.monotonic` (seconds). Seeds: the think profile's `latency` block and its budget-0 p95 (for
    c_0), else the defaults; `reserve_ms` (--think-deadline-reserve) fixes r.

    - r: 1.5 x the p95 of the last 64 readouts after a `deadline` / `budget` close, + 10 ms, floor 20; before 8 of
      them the profile's `latency.read_forced_ms.p95` x 1.5 + 10, else 100.
    - d̂: per question, the median of its last 3 chunks' (prompt_ms + predicted_ms) / n (a continuation's first token is
      decoded in its prompt phase; chunk 0's prompt phase is the prefill, so it counts predicted_ms / (n - 1): a
      1-token chunk 0 measures none), capped at 3 x
      the server window's median; chunk 0 takes the window median over the chunks finished in the last 2 s, else the
      profile's `decode_ms_per_token.p50`, else 30 ms.
    - P̂: the uncached tokens x the EWMA of chunk-0 prompt_ms / prompt_n (prompt_n > 32); before 8, the profile's
      `prefill_ms_per_token.p95`, else 2 ms. The uncached tokens are the prompt's length, unless the slot's cache is
      measured to behave like pure attention (`uncached`).
    - ô: the EWMA of wall - prompt_ms - predicted_ms; before 8 chunks, 5 ms.
    - G_i = S_known + s95_resid: the prefill callo dispatched on the OTHER reserved slots and has not seen finish,
      plus the p95 of the residual stall over the last 64 continuation chunks (0 before 8 of them). Chunk 0s never
      feed the residual: their prefill is P̂'s term.
    - c_0: the p95 of the budget-0 readouts, else the profile's budget-0 lat_p95_ms, else 300 ms.
    - c (early exit's probe cost): live = p50(probe read) + p50(the next chunk's wall - n x the window median step
      - S_known) once 8 chunks followed a probe, else the profile's `converge.cost.c_ms`, else unknown. The probe windows also
      count the chunks after a probe that restored (prompt_n <= 2) for /callo/stats."""

    @classmethod
    def from_profile(cls, tp: Any, *, reserve_ms: float | None = None) -> Estimators:
        """The server-wide estimators seeded from a think profile (None: the defaults): its `latency` block and its
        budget-0 `lat_p95_ms` (c_0). The one place that parses the seed; serve and the service both use it."""
        if tp is None:
            return cls(reserve_ms=reserve_ms, probe_seed_ms=None)
        b0 = (tp.data.get("budgets") or {}).get("0")
        p95 = b0.get("lat_p95_ms") if isinstance(b0, dict) else None
        seed = (float(p95[0]) if isinstance(p95, list) and p95 and isinstance(p95[0], (int, float))
                and not isinstance(p95[0], bool) else None)
        cv = tp.data.get("converge")
        cost = cv.get("cost") if isinstance(cv, dict) else None
        c = cost.get("c_ms") if isinstance(cost, dict) else None
        probe_seed = float(c) if _finite(c) and c >= 0 else None
        return cls(latency=tp.latency, budget0_p95_ms=seed, reserve_ms=reserve_ms, probe_seed_ms=probe_seed)

    def __init__(self, *, latency: Mapping | None = None, budget0_p95_ms: float | None = None,
                 reserve_ms: float | None = None, clock: Callable[[], float] = time.monotonic,
                 probe_seed_ms: float | None = None):
        self._lock = threading.Lock()
        self._clock = clock
        self._latency = dict(latency or {})
        self._b0_seed = budget0_p95_ms
        self._reserve_flag = reserve_ms
        self._reads: deque[float] = deque(maxlen=READS_WINDOW)
        self._steps: deque[tuple[float, float]] = deque(maxlen=WINDOW)       # (finished at, ms per token)
        self._prefills: deque[float] = deque(maxlen=WINDOW)                  # chunk-0 prompt_ms / prompt_n
        self._overheads: deque[float] = deque(maxlen=WINDOW)
        self._stalls: deque[float] = deque(maxlen=WINDOW)                    # residual stalls, for the stats
        self._resid: deque[float] = deque(maxlen=RESID_WINDOW)
        self._b0: deque[float] = deque(maxlen=WINDOW)
        self._prefill_ewma: float | None = None
        self._overhead_ewma: float | None = None
        self._sent: dict[int, tuple[int, ...]] = {}
        self._lcp_ok: deque[bool] = deque(maxlen=LCP_WINDOW)
        self._dispatched: dict[int, tuple[int | None, float, float]] = {}    # token -> (slot, started, est ms)
        self._next = 0
        self._probe_seed = probe_seed_ms
        self._probe_reads: deque[float] = deque(maxlen=WINDOW)        # early exit: each probe readout's ms
        self._probe_excess: deque[float] = deque(maxlen=WINDOW)       # what a probe added to the next chunk
        self._after_n = self._after_restored = 0                      # chunks after a probe, monotonic

    # -- r and c_0 --

    def reserve(self) -> tuple[float, str]:
        """(r in ms, its source: `flag` | `learned` | `profile` | `default`)."""
        if self._reserve_flag is not None:
            return float(self._reserve_flag), "flag"
        with self._lock:
            reads = list(self._reads)
        if len(reads) >= LIVE_MIN:
            return max(RESERVE_FLOOR_MS, 1.5 * percentile(reads, 0.95) + 10.0), "learned"
        seed = _p(self._latency, "read_forced_ms", "p95")
        if seed is not None:
            return 1.5 * seed + 10.0, "profile"
        return DEFAULT_RESERVE_MS, "default"

    def add_readout(self, ms: float) -> None:
        """One readout after a `deadline` or `budget` close (r's window)."""
        with self._lock:
            self._reads.append(float(ms))

    def c0(self) -> float:
        with self._lock:
            b0 = list(self._b0)
        if len(b0) >= LIVE_MIN:
            return percentile(b0, 0.95)
        return float(self._b0_seed) if self._b0_seed is not None and self._b0_seed > 0 else DEFAULT_C0_MS

    def add_budget0(self, ms: float) -> None:
        with self._lock:
            self._b0.append(float(ms))

    # -- early exit: the probe cost --

    def add_probe_read(self, ms: float) -> None:
        with self._lock:
            self._probe_reads.append(float(ms))

    def add_after_probe(self, c: Chunk) -> None:
        """The chunk right after a probe: whether it restored (prompt_n <= 2) and what the forced boundary added to it,
        max(0, wall - n x the window median step - S_known at its start) (calibrate's second term of c; S_known is
        G_i's, already charged)."""
        n = len(c.tokens)
        with self._lock:
            med = self._window_median_locked()
            if med is not None and n:
                self._probe_excess.append(max(0.0, c.wall_ms - n * med - c.known_ms))
            if c.prompt_n is not None:
                self._after_n += 1
                self._after_restored += c.prompt_n <= RESTORE_PROMPT_N

    def probe_cost(self) -> tuple[float | None, str | None]:
        """(c in ms, its source): `live` once 8 chunks followed a probe, else the profile's measured c (`profile`), else
        (None, None)."""
        with self._lock:
            reads, excess = list(self._probe_reads), list(self._probe_excess)
        if len(excess) >= LIVE_MIN and reads:
            return percentile(reads, 0.5) + percentile(excess, 0.5), "live"
        if self._probe_seed is not None:
            return float(self._probe_seed), "profile"
        return None, None

    def converge_stats(self) -> dict:
        """/callo/stats `think.converge`'s probe fields."""
        c, source = self.probe_cost()
        with self._lock:
            reads, n, restored = list(self._probe_reads), self._after_n, self._after_restored
        return {"probe_read_ms": {"p50": percentile(reads, 0.5) if reads else None,
                                  "p95": percentile(reads, 0.95) if reads else None},
                "after_probe": {"n": n, "restored": restored, "share": restored / n if n else None},
                "c_ms": c, "c_source": source}

    # -- d̂, P̂, ô --

    def _window_median_locked(self) -> float | None:
        return statistics.median(s for _, s in self._steps) if self._steps else None

    def step_seed(self) -> float:
        """Chunk 0's d̂."""
        now = self._clock()
        with self._lock:
            recent = [s for t, s in self._steps if now - t <= RECENT_S]
        if recent:
            return statistics.median(recent)
        return _p(self._latency, "decode_ms_per_token", "p50") or DEFAULT_STEP_MS

    def step_for(self, question_steps: Sequence[float]) -> float:
        """A later chunk's d̂: the median of the question's last 3 per-token times, capped at 3 x the window median."""
        if not question_steps:
            return self.step_seed()
        d = statistics.median(question_steps[-LAST_STEPS:])
        with self._lock:
            med = self._window_median_locked()
        return min(d, STEP_CAP * med) if med is not None else d

    def prefill_rate(self) -> float:
        with self._lock:
            if self._prefill_ewma is not None and len(self._prefills) >= LIVE_MIN:
                return self._prefill_ewma
        return _p(self._latency, "prefill_ms_per_token", "p95") or DEFAULT_PREFILL_MS

    def overhead(self) -> float:
        with self._lock:
            if self._overhead_ewma is not None and len(self._overheads) >= LIVE_MIN:
                return self._overhead_ewma
        return DEFAULT_OVERHEAD_MS

    def uncached(self, slot: int | None, prompt: Sequence[int]) -> int:
        """len(prompt) - LCP(prompt, the last array callo sent on the slot), while the window shows the cache
        behaving that way (chunk-0 prompt_n within 8 of that candidate on >= 90 % of the last 32 chunk 0s, at least 8
        of them seen); else the whole prompt (hybrid and SWA models roll back to checkpoints callo cannot see)."""
        with self._lock:
            ok = list(self._lcp_ok)
            sent = self._sent.get(slot) if slot is not None else None
        if sent is None or len(ok) < LIVE_MIN or sum(ok) < LCP_SHARE * len(ok):
            return len(prompt)
        return len(prompt) - _lcp(prompt, sent)

    def note_sent(self, slot: int | None, tokens: Sequence[int]) -> None:
        """The token array callo just sent on a slot (a chunk, a readout's first pass): what the slot's cache holds."""
        if slot is not None:
            with self._lock:
                self._sent[slot] = tuple(tokens)

    # -- G: S_known and the residual stall --

    def dispatch_start(self, slot: int | None, est_ms: float) -> int:
        """A prefill callo dispatched on `slot` (a chunk 0, a budget-0 readout's first pass), estimated at `est_ms`;
        returns its token for `dispatch_end`."""
        with self._lock:
            self._next += 1
            self._dispatched[self._next] = (slot, self._clock(), float(est_ms))
            return self._next

    def dispatch_end(self, token: int) -> None:
        with self._lock:
            self._dispatched.pop(token, None)

    def known(self, slot: int | None) -> float:
        """S_known for a chunk on `slot`: the remaining estimated prefill of every dispatch on another slot."""
        now = self._clock()
        with self._lock:
            return sum(max(0.0, est - 1000.0 * (now - t0)) for s, t0, est in self._dispatched.values() if s != slot)

    def resid95(self) -> float:
        with self._lock:
            resid = list(self._resid)
        return percentile(resid, 0.95) if len(resid) >= LIVE_MIN else 0.0

    def estimate(self, *, i: int, slot: int | None, question_steps: Sequence[float], uncached: int) -> Estimate:
        """Chunk i's estimate: chunk 0 pays P̂ = `uncached` x the prefill rate and takes the seed d̂."""
        known = self.known(slot)
        stall = known + self.resid95()
        if i == 0:
            return Estimate(uncached * self.prefill_rate(), self.step_seed(), self.overhead(), stall, known)
        return Estimate(0.0, self.step_for(question_steps), self.overhead(), stall, known)

    def add_chunk(self, i: int, c: Chunk, *, after_probe: bool = False) -> float | None:
        """Learn from one chunk; returns its per-token time (None for an empty chunk, and for a chunk 0 of one token or
        a one-token chunk after a probe whose restore missed).
        llama-server's `predicted_ms` leaves out the first sampled token, whose logits come from the prompt phase: a
        continuation's step is (prompt_ms + predicted_ms) / n (its 1-token prompt phase is that first step), chunk 0's
        is predicted_ms / (n - 1) (its prompt phase is the prefill), so a 1-token chunk 0 measures no step. A
        continuation chunk also feeds the residual stall max(0, wall - ô - n x window median - S_known at its
        start). `after_probe=True` (early exit: pass it for the chunk right after a probe, and feed the chunk to
        `add_after_probe`) keeps the chunk out of the residual stall and ô (and their stats windows): what the probe
        added to it is the probe cost's term, charged as `probe_ms`, never again as G_i or ô. Its step still feeds d̂'s
        window, which the probe excess is measured against: (prompt_ms + predicted_ms) / n when it restored (prompt_n
        <= 2), else predicted_ms / (n - 1) as chunk 0's (a missed restore re-prefilled the trace, which is no step)."""
        n = len(c.tokens)
        if n == 0:
            return None
        pm, dm = float(c.prompt_ms or 0.0), float(c.predicted_ms or 0.0)
        prefilled = i == 0 or (after_probe and not (c.prompt_n is not None and c.prompt_n <= RESTORE_PROMPT_N))
        step = (dm / (n - 1) if n >= 2 else None) if prefilled else (pm + dm) / n
        now = self._clock()
        with self._lock:
            if i > 0 and not after_probe:     # the residual, against the window median step (the chunk's own step if the window is empty)
                med = self._window_median_locked()
                ov = self._overhead_ewma if self._overhead_ewma is not None and len(self._overheads) >= LIVE_MIN \
                    else DEFAULT_OVERHEAD_MS
                r = max(0.0, c.wall_ms - ov - n * (med if med is not None else float(step or 0.0)) - c.known_ms)
                self._resid.append(r)
                self._stalls.append(r)
            if not after_probe:
                o = max(0.0, c.wall_ms - pm - dm)
                self._overheads.append(o)
                self._overhead_ewma = o if self._overhead_ewma is None else \
                    (1 - EWMA) * self._overhead_ewma + EWMA * o
            if i == 0 and c.prompt_n is not None and c.prompt_n > PREFILL_MIN_N and pm > 0:
                rate = pm / c.prompt_n
                self._prefills.append(rate)
                self._prefill_ewma = rate if self._prefill_ewma is None else \
                    (1 - EWMA) * self._prefill_ewma + EWMA * rate
            if step is not None:
                self._steps.append((now, step))
        return step

    def add_chunk0_prompt(self, slot: int | None, prompt: Sequence[int], prompt_n: int | None) -> None:
        """The pure-attention evidence: did chunk 0's prompt_n match len(prompt) - LCP(prompt, the slot's last
        array) within 8?"""
        if slot is None or prompt_n is None:
            return
        with self._lock:
            sent = self._sent.get(slot)
            if sent is not None:
                self._lcp_ok.append(abs(prompt_n - (len(prompt) - _lcp(prompt, sent))) <= LCP_SLACK)

    def stats(self) -> dict:
        """/callo/stats `think.deadline`'s estimator fields: p50 / p95 of each window, null when empty."""
        def pq(xs: Sequence[float]) -> dict:
            return {"p50": percentile(xs, 0.5) if xs else None, "p95": percentile(xs, 0.95) if xs else None}
        with self._lock:
            steps, pre, ov, st = [s for _, s in self._steps], list(self._prefills), list(self._overheads), list(self._stalls)
        return {"decode_ms_per_token": pq(steps), "prefill_ms_per_token": pq(pre), "chunk_overhead_ms": pq(ov),
                "stall_ms": pq(st)}
