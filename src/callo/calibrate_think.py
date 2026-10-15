"""`callo calibrate --think-budgets` for one server: measure the accuracy / latency trade-off
of letting the model think for B tokens before the System One readout, on the user's labelled data, and write the
think profile `callo serve --think-profile` loads.

The client of `callo serve`'s internal `POST /callo/think-probe`: every question is one single-question
request on the `callo-calibrate` lane; budget 0 is the served single-pass readout at T = 1 (position-bias offsets
included), the positive budgets come from ONE generation at the largest budget, cut for the smaller ones. The
phases: preflight (plan mode: the prompt fits the slot context with the largest budget), warm-up (discarded), phase
A (budget 0), with --converge-probe the gate reads (early exit's gate question at budget 0, as serve reads it), phase B
(the positive budgets), the cut audit (real generations at the smaller budgets on questions
the largest-budget trace was cut at), the drift check (regenerations at Bmax against the warm phase-B decode speed),
then the fit and the profile. The server-state guard of `--bias` runs before the first request, every 25 questions,
at each phase boundary and after the last; concurrent traffic would inflate latency, which is half of what is
measured. Every row is flushed to `<name>.raw.jsonl` as it arrives, so `--resume` continues an interrupted run, and
the profile is a pure function of the raw file and `--data`: a complete raw file refits with no probe request
(`refit_source`), and with no server at all when it is down (`offline_name`).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import random
import shlex
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from . import __version__, converge, guard
from .calibrate import GRID, flat_fit, limit_note, request_from_record, row_from_answer
from .calibrate_bias import BACKOFF as BACKOFF
from .calibrate_bias import RETRIES as RETRIES
from .calibrate_bias import (
    Refusal,
    RunFailed,
    _http,
    _retrying,
    parse_slot,
    read_stats,
    read_stats_mid_run,
    reply_reason,
    silence_stdout,
)
from .deadline import MAX_DEADLINE_MS, trace_terms
from .deadline_map import mapping, mapping_lines, point_cap
from .engine import DEFAULT_TOP_K
from .errors import InvalidInput
from .host import therm_limit
from .marginalize import DEFAULT_DIVERGENCE_MASS, DEFAULT_MAX_DEPTH
from .profile import resume_provenance
from .systemone import MAX_LETTERS, Question, answer_body, parse_request
from .think import (
    EE_KINDS,
    LATENCY_AXES,
    MAX_BUDGET,
    METRICS_VERSION,
    PROFILE_FORMAT,
    QUANTILES,
    STAT_FIELDS,
    Limit,
    Paired,
    Point,
    Preference,
    Sampling,
    accuracy,
    budget_latency,
    crossfit,
    ece_at,
    ee_key,
    finite_number,
    fit_T,
    frontier,
    json_safe,
    latency_stats,
    mark_approx,
    merge_identical,
    nearest_rank,
    nll_at,
    on_frontier,
    paired_bootstrap,
    parse_sampling,
    pick,
    point_latencies,
    point_stats,
    preference_from_args,
    resolve_sampling,
    tail_counts,
    think_seed,
)

RAW_FORMAT = 1
DEFAULT_BUDGETS = (0, 256, 512, 1024)
MAX_BUDGETS = 8
DEFAULT_WARMUP, DEFAULT_AUDIT, DEFAULT_SEED = 3, 6, 0
CHECK_EVERY = 25                     # questions between two guard checks
ETA_AFTER = 5                        # measured questions before the live ETA
DRIFT_N, DRIFT_MIN_GEN = 3, 64       # the drift check's questions and the trace length they need
AUDIT_MIN_CUTS = 3                   # fewer cut questions than this: `audit n/a`
LATENCY_ERR_MAX = 0.10               # a budget is `verified` when its median |relative error| is at most this
DRIFT_RATIO = 0.05                   # |ratio - 1| above this, or any thermal reading below 100, flags the drift
# RETRIES = 3 and BACKOFF = (1, 4, 16) s are calibrate_bias's, re-exported above, as is the retry loop (`_retrying`)
RETRY_STATUSES = (503, 504)
REPREFILL_PROMPT_N = 8
ROW_FIELDS = ("correct", "nll1", "pmax", "lat_ms", "k", "close")
PHASES = ("warmup", "A", "B", "audit", "drift", "cost", "gate")     # `cost`, the probe-cost pass; `gate`, early
#                                                                   exit's gate read (with --converge-probe)
MAX_DEADLINES = 4                    # --deadlines takes up to 4
CONVERGE_KS = converge.PROBE_KS      # --converge-probe's K
DEFAULT_CONVERGE_K, DEFAULT_CONVERGE_COST = 64, 6
DEFAULT_RESERVE_MS = 100.0           # r_cal with neither the flag nor a usable warm-up read
DEADLINE_CLOSES = ("natural", "budget", "deadline", "eos", "budget0")   # a deadline point's closes
# an older callo serve ignores the probe keys it does not know, so a multi-option reply without
# them is a failed run naming the fix, never a run that quietly has no deadline points
NO_DEADLINE_PROBES = "unusable reply: the server does not support --deadlines probes; upgrade callo serve"
NO_CONVERGE_PROBES = "unusable reply: the server does not support --converge-probe probes; upgrade callo serve"
NO_GATE_PROBES = "unusable reply: the server does not support the early-exit gate's probe; upgrade callo serve"
# why build_profile made no point of a measured deadline (`profile["deadline_skipped"]`)
DEADLINE_SKIPPED = "no question was read at the deadline (each fell back to budget 0 or has one option); no point"


# ---- the run-wide settings and the budgets ------------------------------------------------------------

def parse_budgets(text: str | None) -> list[int]:
    """`--think-budgets`: up to 8 distinct integers 0..32768, comma-separated; no value is 0,256,512,1024; 0 is added
    when missing (every comparison is against it). Raises InvalidInput (the CLI's exit 2)."""
    if text is None or text.strip() == "":
        return list(DEFAULT_BUDGETS)
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not (part.isascii() and part.isdecimal()):     # ASCII digits only: "²" or "١٢" are no budgets
            raise InvalidInput(f"--think-budgets: {part!r} is not a non-negative integer")
        b = int(part)
        if b > MAX_BUDGET:
            raise InvalidInput(f"--think-budgets: {b} is above the hard cap {MAX_BUDGET}")
        if b in out:
            raise InvalidInput(f"--think-budgets: {b} is listed twice")
        out.append(b)
    if 0 not in out:
        out.append(0)
    if len(out) > MAX_BUDGETS:
        raise InvalidInput(f"--think-budgets: at most {MAX_BUDGETS} budgets (0 included), got {len(out)}")
    return sorted(out)


def parse_deadlines(text: str | None, *, flag: str = "--deadlines") -> tuple[int, ...]:
    """`--deadlines D1,...` (and compare's `--deadline`): up to 4 distinct integers 1..600000 (ms),
    comma-separated, sorted; None or empty is none. Raises InvalidInput (the CLI's exit 2) naming `flag`."""
    if text is None or text.strip() == "":
        return ()
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not (part.isascii() and part.isdecimal()) or not 1 <= int(part) <= MAX_DEADLINE_MS:
            raise InvalidInput(f"{flag}: {part!r} is not a deadline in ms from 1 to {MAX_DEADLINE_MS}")
        if int(part) in out:
            raise InvalidInput(f"{flag}: {int(part)} is listed twice")
        out.append(int(part))
    if len(out) > MAX_DEADLINES:
        raise InvalidInput(f"{flag}: at most {MAX_DEADLINES} deadlines, got {len(out)}")
    return tuple(sorted(out))


@dataclass(frozen=True)
class ThinkSettings:
    """What a run measures with, run-wide: the same for every model of a menu."""
    data: str                        # the --data path
    data_name: str
    data_sha256: str
    budgets: tuple[int, ...]         # ascending, 0 included
    seed: int = DEFAULT_SEED
    warmup: int = DEFAULT_WARMUP
    audit: int = DEFAULT_AUDIT
    cache: bool = True               # --cache, on by default since 0.3.0 (--no-cache: off)
    sampling: str = "model"          # the --think-sampling policy text
    latency: str = "mean"            # the pick statistic
    max_latency: float | None = None
    min_accuracy: float | None = None
    verbose: bool = False
    allow_busy: bool = False
    limits: tuple[Limit, ...] = ()   # every --limit; --max-latency joins them in `preference`
    price: float = 0.0               # --price R, pp/s
    deadlines: tuple[int, ...] = ()  # the deadline points, in ms
    reserve_ms: int | None = None    # --think-deadline-reserve fixes r_cal (else the warm-up's pre-pass)
    converge_probe: int | None = None    # K, a probe readout every K tokens of each Bmax trace
    converge_cost: int = DEFAULT_CONVERGE_COST   # the questions the probe-cost pass regenerates live

    @property
    def preference(self) -> Preference:
        """The pick's preference (InvalidInput when out of range): never part of `comparable`, so changing it
        never stops a resume, and `callo compare` re-picks from the profile under any other."""
        return Preference(self.latency, self.price, self.limits, self.min_accuracy, None, self.max_latency)

    def __post_init__(self) -> None:
        """What `bmax` / `positive` and the run assume, and parse_budgets guarantees: a settings built elsewhere (the
        menu run) is held to it too. InvalidInput (exit 2)."""
        b = tuple(self.budgets)
        if len(b) > MAX_BUDGETS:
            raise InvalidInput(f"--think-budgets: at most {MAX_BUDGETS} budgets (0 included), got {len(b)}")
        bad = [x for x in b if type(x) is not int or not 0 <= x <= MAX_BUDGET]     # `is`: a bool is no budget
        if bad:
            raise InvalidInput(f"--think-budgets: each budget is an integer from 0 to {MAX_BUDGET}, got {bad[0]!r}")
        if any(x >= y for x, y in zip(b, b[1:], strict=False)):
            raise InvalidInput(f"--think-budgets: the budgets must be strictly ascending, got {list(b)}")
        if 0 not in b:
            raise InvalidInput(f"--think-budgets: the budgets must include 0, got {list(b)}")

    @property
    def bmax(self) -> int:
        return max(self.budgets)

    @property
    def positive(self) -> tuple[int, ...]:
        return tuple(b for b in self.budgets if b > 0)

    def comparable(self) -> dict:
        """The run-wide part of the raw header and the profile's `comparable`: what --resume must match. The
        think-deadline keys only when set, so a raw file from before them still resumes and adding one
        to a run is a header mismatch."""
        out = {"data": {"name": self.data_name, "sha256": self.data_sha256}, "budgets": list(self.budgets),
               "seed": self.seed, "warmup": self.warmup, "audit": self.audit, "cache": self.cache,
               "sampling_policy": self.sampling}
        if self.deadlines:
            out["deadlines"] = list(self.deadlines)
            if self.reserve_ms is not None:
                out["deadline_reserve_ms"] = self.reserve_ms
        if self.converge_probe:
            out["converge_probe"] = self.converge_probe
            out["converge_cost"] = self.converge_cost
        return out


def data_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- the data --------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    """One labelled text question: a single-question request carrying its record's state and context."""
    item_id: str                     # `_meta.id` when present, else `line<N>`
    qid: str
    cluster: str                     # `_meta.group_id`, else `_meta.pair_id`, else the item id
    question: Question
    label: int                       # the gold option's index in `question.keys` order
    body: dict                       # the /v1/systemone request with this one question
    kind: str                        # "plain" | "context"
    source: str | None = None        # `_meta.source`, else the question's `src`, else None: the bootstrap's strata
    # wide-choice: the position in, and the size of, the loaded set (`load_data`), so `shuffled` gives
    # every subset (a lower-coverage arm's) the full set's order; None for an Item built elsewhere
    pos: int | None = None
    of: int | None = None

    @property
    def uid(self) -> str:
        return f"{self.item_id}/{self.qid}"


def load_data(path: str | Path) -> tuple[list[Item], Counter]:
    """Every labelled text question of a labelled System One JSONL, in file order, and what was skipped (`records`,
    `variant`, `media`, `unlabelled`). A malformed line, an unreadable file, a duplicated `_meta.id` or a cluster
    whose questions come from two sources is InvalidInput (exit 1)."""
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except OSError as e:
        raise InvalidInput(f"--data {path}: {e}") from None
    except UnicodeDecodeError as e:
        raise InvalidInput(f"--data {Path(path).name}: not UTF-8 ({e})") from None
    items: list[Item] = []
    skipped: Counter = Counter()
    seen: set[str] = set()
    for n, line in enumerate(text.split("\n"), 1):     # "\n" only: a JSON string may hold a raw U+2028
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as e:
            hint = ("; its lines end in a bare CR (old Mac line endings): callo splits lines on LF, so convert the file "
                    "to LF or CRLF" if "\r" in line.rstrip("\r") else "")
            raise InvalidInput(f"--data line {n}: not valid JSON ({e}){hint}") from None
        if not isinstance(rec, dict):
            raise InvalidInput(f"--data line {n}: not a JSON object")
        skipped["records"] += 1
        meta = rec.get("_meta") if isinstance(rec.get("_meta"), dict) else {}
        if meta.get("variant", "clean") != "clean":
            skipped["variant"] += 1
            continue
        try:
            full = request_from_record(rec)
            req = parse_request(full)
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            raise InvalidInput(f"--data line {n}: not a System One request ({type(e).__name__}: {e})") from None
        if req.media:
            skipped["media"] += 1
            continue
        item_id = str(meta["id"]) if meta.get("id") is not None else f"line{n}"
        if item_id in seen:
            raise InvalidInput(f"--data line {n}: _meta.id {item_id!r} is duplicated; question ids must be unique")
        seen.add(item_id)
        cluster = str(meta.get("group_id") or meta.get("pair_id") or item_id)
        for q in req.questions:
            raw_q = rec["questions"][q.qid]
            if not isinstance(raw_q, dict) or raw_q.get("label") is None:     # null is "no label", as missing is
                skipped["unlabelled"] += 1
                continue
            try:
                label = row_from_answer(raw_q, answer_body(q, [1.0 / len(q.keys)] * len(q.keys)))[1]
            except InvalidInput:            # a typo'd label is an error, never "unlabelled"
                options = ["false", "true"] if q.type == "noul" else list(q.keys)
                raise InvalidInput(f"--data line {n} ({item_id}/{q.qid}): label {raw_q['label']!r} is not one of its "
                                   f"options: {', '.join(options)}") from None
            body = {k: v for k, v in full.items() if k != "questions"}
            body["questions"] = {q.qid: full["questions"][q.qid]}
            source = meta.get("source") or raw_q.get("src")
            items.append(Item(item_id, q.qid, cluster, q, label, body, "context" if req.context else "plain",
                              str(source) if source is not None else None))
    if any(it.source is not None for it in items):      # the bootstrap resamples clusters within sources
        of: dict[str, str] = {}
        for it in items:
            src = it.source or "unknown"
            if of.setdefault(it.cluster, src) != src:
                raise InvalidInput(f"--data: cluster {it.cluster} spans sources {of[it.cluster]} and {src}: a group "
                                   f"(_meta.group_id / pair_id) must come from one source (_meta.source)")
    return [dataclasses.replace(it, pos=i, of=len(items)) for i, it in enumerate(items)], skipped


def coverage_filter(items: Sequence[Item], stats: Mapping[str, Any]) -> tuple[list[Item], int, int]:
    """(the items this server can read, how many were dropped, the option limit). The limit is
    max(26, what the server reads): `/callo/stats.wide_readable` (Callo 0.3.1 on: labels past the coverage are read in
    two passes), else `wide_coverage` (a 0.3.0 server, which refuses past it); a server with neither (or null) is read
    as 26. The same rule reads a raw file's header. A dropped question is counted as `too_many_options` and never
    probed, so the probe's 422 never aborts a run."""
    limit = max(MAX_LETTERS, _readable(stats))
    keep = [it for it in items if len(it.question.labels) <= limit]
    return keep, len(items) - len(keep), limit


def _is_count(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _readable(stats: Mapping[str, Any]) -> int:
    """How many two-letter labels the server reads: `wide_readable`, else `wide_coverage`, else 26."""
    for key in ("wide_readable", "wide_coverage"):
        if _is_count(stats.get(key)):
            return stats[key]
    return MAX_LETTERS


def coverage_skip_line(dropped: int, limit: int, stats: Mapping[str, Any]) -> str:
    """The log line for the questions `coverage_filter` dropped."""
    readable, cov = stats.get("wide_readable"), stats.get("wide_coverage")
    if _is_count(readable):
        why = (f"this model's tokenizer can read {readable} of Callo's two-letter labels (the next one needs a letter "
               f"that is not a single token)")
    elif _is_count(cov):
        why = f"this model's tokenizer covers {cov} of callo's two-letter labels as single tokens"
    else:
        why = "the server reports no wide_coverage (read as 26)"
    return f"skipped {dropped} question(s) with more than {limit} options (too_many_options): {why}"


def probed_only(items: Sequence[Item], raw: RawThink) -> tuple[list[Item], int]:
    """The refit with no server: the items `coverage_filter` kept when the raw file was measured, without
    /callo/stats. A raw header that records `wide_readable` or `wide_coverage` decides it exactly; an older one falls back
    to the rows: a question above 26 options of which the raw file holds no row at all was dropped by that run's
    coverage (a complete run reads phase A for every item it kept). (items, dropped)."""
    if _is_count(raw.header.get("wide_readable")) or _is_count(raw.header.get("wide_coverage")):
        keep, dropped, _ = coverage_filter(items, raw.header)
        return keep, dropped
    seen = {r["q"] for r in raw.rows}
    keep = [it for it in items if len(it.question.labels) <= MAX_LETTERS or it.uid in seen]
    return keep, len(items) - len(keep)


def shuffled(items: Sequence[Item], seed: int) -> list[Item]:
    """A seeded order, identical for every model on the same data and seed. Items from one `load_data`
    are placed by the permutation of the WHOLE loaded set (random.shuffle's permutation depends only on the length), so
    a subset, such as the questions a lower-coverage arm keeps, is the full set's order with the others left out
    the full set comes out exactly as before."""
    rnd = random.Random(f"{seed}:think-order")
    sizes = {it.of for it in items}
    if items and None not in sizes and len(sizes) == 1:
        perm = list(range(next(iter(sizes))))       # type: ignore[arg-type]
        rnd.shuffle(perm)
        at = {it.pos: it for it in items}
        return [at[p] for p in perm if p in at]
    out = list(items)
    rnd.shuffle(out)
    return out


# ---- the probe client --------------------------------------------------------------------------------

def _reason(reply: Any) -> str:
    """A non-200 reply's reason: callo's `detail` when it gives one (the probe's 422 names the pin it refuses), else
    the body (`calibrate_bias.reply_reason`, which the retry loop's message uses too)."""
    return reply_reason(reply)


@dataclass(frozen=True)
class ProbeReply:
    body: dict
    slot: int | None                 # X-Callo-Slot as an int (0 for "-" with no reserved slots)
    readouts: int                    # the readouts the probe made (the guard's condition 4)
    retries: int                     # connection errors that were retried: each may have been answered upstream


class ProbeClient:
    """`POST /callo/think-probe` on the `callo-calibrate` lane, with the retry policy: a 503, a 504
    or a connection error is retried 3 times (1, 4, 16 s); any other non-200 is RunFailed naming the question."""

    def __init__(self, base: str, *, s1_slots: int, sleep: Callable[[float], None] = time.sleep,
                 timeout: float = 3600.0):
        self.base, self.s1_slots, self.sleep, self.timeout = base.rstrip("/"), s1_slots, sleep, timeout

    def _post(self, body: dict, what: str) -> tuple[int, dict[str, str], Any, int]:
        return _retrying(lambda: _http("POST", f"{self.base}/callo/think-probe", body,
                                       {"X-Callo-Conversation": guard.LANE}, timeout=self.timeout),
                         failing=what, sleep=self.sleep, statuses=RETRY_STATUSES)

    def plan(self, item: Item) -> tuple[int, int | None]:
        """Plan mode: (the thinking-on prompt's tokens, the slot n_ctx); no readout, no slot taken."""
        status, _, reply, _ = self._post({"request": item.body, "budgets": [0], "plan": True}, f"{item.uid} (plan)")
        if status != 200 or not isinstance(reply, dict):
            raise RunFailed(f"{item.uid} (plan): HTTP {status}: {_reason(reply)}")
        q = _question(reply, item)
        n, n_ctx = q.get("on_tokens") if q else None, q.get("n_ctx") if q else None
        if type(n) is not int or not (n_ctx is None or type(n_ctx) is int):
            raise RunFailed(f"{item.uid} (plan): unusable reply (no int on_tokens / n_ctx): {str(reply)[:300]}")
        return n, n_ctx

    def read(self, item: Item, budgets: Sequence[int], seed: int, sampling: dict | None,
             extras: Mapping[str, Any] | None = None) -> ProbeReply:
        """One probe; `extras` are the think-deadline additions (`deadlines` with `reserve_ms`,
        `reserve_read`), sent only when given."""
        status, headers, reply, dropped = self._post(
            {"request": item.body, "budgets": list(budgets), "seed": seed, "sampling": sampling, **(extras or {})},
            item.uid)
        if status != 200 or not isinstance(reply, dict):
            raise RunFailed(f"{item.uid}: HTTP {status}: {_reason(reply)}")
        n = reply.get("readouts")
        if type(n) is not int or _question(reply, item) is None:
            raise RunFailed(f"{item.uid}: unusable reply (no int readouts or no {item.qid!r} question): "
                            f"{str(reply)[:300]}")
        slot = parse_slot(headers.get("x-callo-slot"), self.s1_slots)
        # the server's own count, never len(budgets): an eos trace probed at n_gen - 1 and n_gen reads once for both,
        # and a single-option question is one readout at any budget with `trace: null`
        return ProbeReply(reply, slot, n, dropped)


def _question(reply: dict, item: Item) -> dict | None:
    """The reply's block for the item's question, or None when a 200 does not carry it as an object."""
    qs = reply.get("questions")
    q = qs.get(item.qid) if isinstance(qs, dict) else None
    return q if isinstance(q, dict) else None


# ---- the raw file and --resume ------------------------------------------------------------------------

@dataclass
class RawThink:
    header: dict
    rows: list[dict]                 # the phase rows, in file order
    checks: list[dict]               # the guard's check records
    therm: list[dict]                # {"at": phase boundary, "limit": int | None}
    partial: bool                    # the file ends in an incomplete line
    reserve: dict | None = None      # r_cal as the run fixed it, {"ms", "source", "reads"} (`reserve_record`)


def raw_header(stats: dict, settings: ThinkSettings, sampling: Sampling) -> dict:
    """The raw file's header: what --resume must match, plus the host (a warning only). It also records, at the
    run's start, everything else the fit reads from `/callo/stats` (`HEADER_FIT_KEYS`: the thinking mode, the model's
    sizes, the user delimiter), so a complete raw file refits with no server at all and a resume
    against a server whose thinking mode changed is refused like any other header difference."""
    router = stats.get("router") or {}
    think = stats.get("think") or {}
    return {"model_key": stats.get("model_key"), "provenance": stats.get("provenance"),
            "thinking": {"capable": bool(think.get("capable")), "reason": think.get("reason"),
                         "close": think.get("close"), "synthesized_open": think.get("synthesized_open")},
            "model": {"bytes": stats.get("model_bytes"), "mmproj_bytes": stats.get("mmproj_bytes")},
            "user_delimiter": stats.get("user_delimiter"),
            "wide_coverage": stats.get("wide_coverage"),      # what coverage_filter kept, for the refit
            "wide_readable": stats.get("wide_readable"),
            "server": {"kv": stats.get("kv"), "s1_slots": len(router.get("reserved") or []), "n_ctx": stats.get("n_ctx"),
                       "total_slots": stats.get("total_slots"), "cache_prompt": (stats.get("cache_prompt") or {}).get("plain"),
                       "position_bias_sha256": (stats.get("position_bias") or {}).get("sha256"),
                       "upstream_local": stats.get("upstream_local")},     # None when callo does not say
            "comparable": {**settings.comparable(), "sampling": sampling.as_dict()},
            "host": stats.get("host"), "instance": stats.get("instance"), "model_file": stats.get("model_file")}


_HEADER_IGNORED = ("callo_think_raw", "instance")   # read_raw checks the format; instance is a per-startup uuid
HEADER_FIT_KEYS = ("thinking", "model", "user_delimiter")   # recorded in every raw file: an older one lacks
                                                            # them, and the fit then reads them from /callo/stats
_HEADER_WARN = ("host",)                              # its fingerprint only warns; provenance.build_info too


def cache_resume_hint(recorded: Any, cache: bool) -> str:
    """The fix a resume refused on `comparable.cache` alone needs (--cache went on by default in 0.3.0, so every
    earlier run is cache off and a resume without a flag differs), as a "; ..." tail, or "" when the recorded value
    is this run's (or not a bool)."""
    if not isinstance(recorded, bool) or recorded == cache:
        return ""
    if recorded:
        return "; that run was measured with --cache on: drop --no-cache to resume it"
    return ("; that run was measured with --cache off (the default before callo 0.3): pass --no-cache to resume a "
            "cache-off run")


def header_diff(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """(the keys whose difference refuses a resume, the keys that only warn): ANY difference refuses, a key
    added to the header later included, but the host fingerprint and provenance's `build_info`, which warn, and
    `_HEADER_IGNORED`. A None fingerprint on either side is "unknown host", a warning and never a match."""
    refuse, warn = [], []
    keys = [k for k in (*old, *(k for k in new if k not in old)) if k not in _HEADER_IGNORED + _HEADER_WARN]
    for k in keys:
        if k == "provenance":
            op, np_ = dict(old.get(k) or {}), dict(new.get(k) or {})
            if op.pop("build_info", None) != np_.pop("build_info", None):
                warn.append("build_info")
            op, np_ = resume_provenance(op, np_)          # a recorded file without wide_contract
            if op != np_:
                refuse.append(k)
        elif k in ("wide_coverage", "wide_readable"):     # absent (or null) when recorded means compatible
            if old.get(k) is not None and old.get(k) != new.get(k):
                refuse.append(k)
        elif old.get(k) != new.get(k):
            refuse.append(k)
    fo, fn = (old.get("host") or {}).get("fingerprint"), (new.get("host") or {}).get("fingerprint")
    if fo is None or fn is None:
        warn.append("unknown host")
    elif fo != fn:
        warn.append("host")
    return refuse, warn


def read_raw(path: str | Path) -> RawThink:
    """The one reader of the raw format (--resume and the offline refit). An incomplete last line is left out and
    reported as `partial`; a file without the header or with an unreadable line is InvalidInput."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise InvalidInput(f"{path} is not a raw file (not UTF-8: {e})") from None
    except OSError as e:
        raise InvalidInput(f"cannot read the raw file {path} ({e})") from None
    end = text.rfind("\n") + 1
    lines = text[:end].split("\n")
    try:
        head = json.loads(lines[0]) if lines else None
    except ValueError:
        head = None
    if not isinstance(head, dict) or head.get("callo_think_raw") != RAW_FORMAT:
        raise InvalidInput(f"{path} has no callo --think-budgets raw header")
    raw = RawThink(head, [], [], [], end != len(text))
    for i, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            raise InvalidInput(f"{path} line {i} is not JSON") from None
        if not isinstance(d, dict):
            raise InvalidInput(f"{path} line {i} is not a raw record")
        if "check" in d:
            raw.checks.append(d["check"])
        elif "therm" in d:
            raw.therm.append(d["therm"])
        elif isinstance(d.get("reserve"), dict):     # the first one fixes r_cal for the run
            raw.reserve = raw.reserve or d["reserve"]
        elif d.get("phase") in PHASES and isinstance(d.get("q"), str):
            raw.rows.append(d)
        else:
            raise InvalidInput(f"{path} line {i} is not a raw record (no phase, check or therm)")
    return raw


_NUM = (int, float)


def _probs_ok(x: Any) -> bool:
    return isinstance(x, list) and bool(x) and all(isinstance(v, _NUM) for v in x)


def _row_problem(d: Mapping[str, Any]) -> str | None:
    """What a phase row lacks that the fit reads, or None. Warm-up and drift rows are read
    with defaults, so only the A, B and audit rows are checked. `read_raw` stays lenient (other readers, the bench's,
    take partial rows); the run checks every row it writes and every row it resumes from or fits (`check_rows`)."""
    phase, t = d["phase"], d.get("trace")
    if phase == "A" and not (_probs_ok(d.get("probs")) and isinstance(d.get("ms"), _NUM)):
        return "a phase-A row needs probs and ms"
    if phase == "B":
        if t is not None and not (isinstance(t, dict) and type(t.get("n_gen")) is int
                                  and isinstance(t.get("gen_wall_ms"), _NUM) and isinstance(t.get("tokens"), list)):
            return "a phase-B trace needs tokens, n_gen and gen_wall_ms"
        reads = d.get("budgets")
        if not isinstance(reads, dict) or not all(
                isinstance(r, dict) and _probs_ok(r.get("probs")) and isinstance(r.get("read_ms"), _NUM)
                and (t is None or type(r.get("k")) is int) for r in reads.values()):
            return "a phase-B row needs budgets, each with probs, read_ms and k"
        dls = d.get("deadlines")                         # each deadline's close, k, elapsed and readout
        if dls is not None and not (isinstance(dls, dict) and all(
                isinstance(r, dict) and r.get("close") in DEADLINE_CLOSES and type(r.get("k")) is int
                and isinstance(r.get("elapsed_ms"), _NUM)
                and (r["close"] == "budget0" or (_probs_ok(r.get("probs")) and isinstance(r.get("read_ms"), _NUM)))
                for r in dls.values())):
            return "a phase-B row's deadlines need close, k, elapsed_ms and, unless budget0, probs and read_ms"
        probes = d.get("probes")                         # each probe's k and probs
        if probes is not None and not (isinstance(probes, list) and all(
                isinstance(p, dict) and type(p.get("k")) is int and _probs_ok(p.get("probs")) for p in probes)):
            return "a phase-B row's probes need k and probs"
    if phase == "cost":                                  # the live probe-cost pass
        lv = d.get("live")
        if not (isinstance(lv, dict) and isinstance(lv.get("chunks"), list) and isinstance(lv.get("probes"), list)
                and all(isinstance(c, dict) and isinstance(c.get("wall_ms"), _NUM) and type(c.get("n")) is int
                        for c in lv["chunks"])
                and all(isinstance(p, dict) and isinstance(p.get("read_ms"), _NUM) for p in lv["probes"])):
            return "a cost row needs its live chunks (wall_ms, n) and probes (read_ms)"
    if phase == "gate" and not (isinstance(d.get("ms"), _NUM) and (d.get("p_yes") is None
                                                                    or isinstance(d.get("p_yes"), _NUM))):
        return "a gate row needs ms and p_yes (a number, or null for a failed read)"
    if phase == "audit" and not (type(d.get("budget")) is int and _probs_ok(d.get("probs"))
                                 and isinstance(d.get("lat_ms"), _NUM) and isinstance(t, dict)
                                 and isinstance(t.get("tokens"), list)):
        return "an audit row needs budget, probs, lat_ms and a trace with tokens"
    return None


def check_rows(raw: RawThink, path: str | Path) -> None:
    """InvalidInput naming the first row the fit could not read (a valid-JSON row cut short by hand or by another
    tool), never a KeyError traceback in the fit."""
    for n, r in enumerate(raw.rows, 1):
        if (why := _row_problem(r)) is not None:
            raise InvalidInput(f"{path}: row {n} ({r['phase']} {r['q']}) is incomplete: {why}")


def cells(rows: Sequence[dict]) -> set[tuple[str, str, int | None]]:
    """The (phase, question, budget) cells a raw file holds: audit rows carry their budget, the others None."""
    return {(r["phase"], r["q"], r.get("budget") if r["phase"] == "audit" else None) for r in rows}


def _write(f: IO[str], obj: dict) -> None:
    f.write(json.dumps(obj, separators=(",", ":")) + "\n")
    f.flush()


def _open_raw(path: Path, header: dict, *, resume: bool, log: Callable[[str], None]) -> tuple[IO[str], RawThink]:
    """Create the raw file with its header, or (`--resume`) append to the existing one after checking its header and
    dropping a hard kill's partial last line. An existing file without --resume, or a header that differs, is a
    Refusal (exit 2)."""
    if path.exists() and not resume:
        raise Refusal(f"refusing to calibrate: {path} already exists. Pass --resume to continue that run, or remove it")
    if path.exists():
        try:
            raw = read_raw(path)
        except InvalidInput as e:
            raise Refusal(f"refusing to resume: {e}; remove the file to start over") from None
        missing = [k for k in HEADER_FIT_KEYS if k not in raw.header]
        if missing:                         # an older raw file: compare what it has; the fit asks the server for the rest
            header = {k: v for k, v in header.items() if k not in missing}
        refuse, warn = header_diff(raw.header, header)
        if refuse:
            hint = cache_resume_hint((raw.header.get("comparable") or {}).get("cache"),
                                     bool((header.get("comparable") or {}).get("cache")))
            raise Refusal(f"refusing to resume: {path} was written by a different run ({', '.join(refuse)} differ from "
                          f"this one), and rows from two runs cannot be mixed. Remove the file to start over{hint}")
        try:
            check_rows(raw, path)
        except InvalidInput as e:
            raise Refusal(f"refusing to resume: {e}; remove the file to start over") from None
        for k in warn:
            log("WARNING: --resume: unknown host: cannot tell whether the raw file was measured on this machine; its "
                "latencies may not be comparable" if k == "unknown host" else
                f"WARNING: --resume: {k} differs from the raw file's; its latencies may not be comparable")
        if raw.partial:
            with path.open("r+b") as fh:
                fh.truncate(fh.read().rfind(b"\n") + 1)
        return path.open("a", encoding="utf-8"), raw
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("x", encoding="utf-8")
    _write(f, {"callo_think_raw": RAW_FORMAT, **header})
    return f, RawThink({"callo_think_raw": RAW_FORMAT, **header}, [], [], [], False)


def reserve_record(settings: ThinkSettings, reads: Sequence[float | None]) -> dict:
    """r_cal and where it came from: --think-deadline-reserve (`flag`), else 1.5 x the largest pre-pass
    read + 10 ms over warm-up questions 2..W (`pre-pass`: question 1 is cold, shader compile and page-in, so its read
    is recorded but never used), else 100 ms (`default`). `reads` holds one entry per warm-up question, in
    order: its served-like read, already one decode step longer when its first pass was 3 tokens, or None for a
    single-option question, which has none (so question 1 is skipped by its index)."""
    every = [None if x is None else float(x) for x in reads]
    if settings.reserve_ms is not None:
        return {"ms": float(settings.reserve_ms), "source": "flag", "reads": every}
    used = reserve_reads(every)
    if used:
        return {"ms": 1.5 * max(used) + 10.0, "source": "pre-pass", "reads": every}
    return {"ms": DEFAULT_RESERVE_MS, "source": "default", "reads": every}


def reserve_reads(reads: Sequence[float | None]) -> list[float]:
    """The pre-pass reads r_cal and the `latency` block use: warm-up questions 2..W (question 1 is cold), by index,
    without the single-option questions' placeholders (None)."""
    return [float(x) for x in reads[1:] if x is not None]


# ---- the fit and the profile: a pure function of the raw file and the data ------------

@dataclass(frozen=True)
class Measure:
    """One question at one budget, as the fit sees it."""
    probs: tuple[float, ...]
    correct: bool
    nll1: float
    pmax: float
    lat_ms: float
    k: int
    close: str | None               # None at budget 0
    miss: int
    option_mass: float
    read_prompt_n: int | None
    text_fallback: bool
    reprefill_ms: float             # the calibration-only re-prefill dropped from lat_ms


def _row_fields(probs: Sequence[float], label: int) -> tuple[bool, float, float]:
    p = [float(x) for x in probs]
    return max(range(len(p)), key=p.__getitem__) == label, -math.log(max(p[label], 1e-12)), max(p)


def _latest(rows: Sequence[dict], phase: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        if r.get("phase") == phase:
            out[r["q"]] = r
    return out


def _phase_b(raw: RawThink) -> dict[str, dict]:
    """The latest phase-B row per question that is a latency source. A row whose probe also read budget 0 (a mixed
    `[0, B]` probe) is not one: its budget-0 readout warmed the slot with the question before the generation, which a
    served cold question never gets. Phase B probes the positive budgets only; only the warm-up mixes."""
    return _latest([r for r in raw.rows if not (r.get("phase") == "B" and "0" in (r.get("budgets") or {}))], "B")


def _traces(rows: Sequence[dict]) -> list[dict]:
    """The rows' generations; a single-option question has none (`trace: null`)."""
    return [r["trace"] for r in rows if isinstance(r.get("trace"), dict)]


def outcomes(items: Sequence[Item], raw: RawThink, budgets: Sequence[int]) -> dict[str, dict[int, Measure]]:
    """Every measured (question, budget) outcome from the phase A and B rows (warm-up, audit and drift rows are never
    measures, nor is a mixed `[0, B]` probe's row, `_phase_b`). Budget 0 is phase A's readout; a budget B > 0 is the
    cut k_B = min(B, n_gen) of the phase-B trace, with the latency and a `forced` close when the cut is inside the
    trace. A single-option question has no trace (`trace: null`): its readout is the same at every budget, with
    k 0, no close and its own `read_ms` as the latency, as serve reads it."""
    a_rows, b_rows = _latest(raw.rows, "A"), _phase_b(raw)
    out: dict[str, dict[int, Measure]] = {}
    for it in items:
        per: dict[int, Measure] = {}
        a = a_rows.get(it.uid)
        if a is not None and 0 in budgets:
            correct, nll1, pmax = _row_fields(a["probs"], it.label)
            per[0] = Measure(tuple(a["probs"]), correct, nll1, pmax, float(a["ms"]), 0, None, int(a.get("miss") or 0),
                             float(a.get("option_mass") or 0.0), a.get("prompt_n"), False, 0.0)
        b = b_rows.get(it.uid)
        if b is not None and b.get("trace") is None:
            for bud in budgets:
                r = b["budgets"].get(str(bud))
                if bud == 0 or r is None:
                    continue
                correct, nll1, pmax = _row_fields(r["probs"], it.label)
                per[bud] = Measure(tuple(r["probs"]), correct, nll1, pmax, float(r["read_ms"]), 0, None,
                                   int(r.get("miss") or 0), float(r.get("option_mass") or 0.0), r.get("read_prompt_n"),
                                   bool(r.get("text_fallback")), 0.0)
        elif b is not None:
            t = b["trace"]
            n_gen = int(t["n_gen"])
            reads = b["budgets"]
            at_ngen = next((r for r in reads.values() if int(r["k"]) >= n_gen), None)
            # a complete row always holds the readout at k >= n_gen (Bmax); without it (a hand-cut row) p_ms(n_gen)
            # counts as 0, so the whole p_ms(k) is taken as re-prefill, off by the cache-hit readout's own p_ms
            p_ngen = at_ngen.get("p_ms") if at_ngen is not None else None
            for bud in budgets:
                r = reads.get(str(bud))
                if bud == 0 or r is None:
                    continue
                k = int(r["k"])
                lat = budget_latency(float(t.get("frame_ms") or 0.0), float(t["gen_wall_ms"]), t.get("predicted_ms"),
                                     n_gen, k, float(r["read_ms"]), r.get("p_ms"), p_ngen)
                extra = max(0.0, float(r.get("p_ms") or 0.0) - float(p_ngen or 0.0)) if k < n_gen else 0.0
                correct, nll1, pmax = _row_fields(r["probs"], it.label)
                per[bud] = Measure(tuple(r["probs"]), correct, nll1, pmax, lat, k,
                                   "forced" if k < n_gen else str(t["close"]), int(r.get("miss") or 0),
                                   float(r.get("option_mass") or 0.0), r.get("read_prompt_n"),
                                   bool(r.get("text_fallback")), extra)
        if per:
            out[it.uid] = per
    return out


def deadline_outcomes(items: Sequence[Item], raw: RawThink, deadline: int,
                      outs: Mapping[str, Mapping[int, Measure]]) -> dict[str, Measure]:
    """Every question's outcome at a deadline point, from its phase-B row's simulated close (the probe ran
    `deadline.simulate` on the trace's own terms). Its latency is on the budgets' clock (whatever the
    chunk-0 asymmetry): `think.budget_latency` at the cut k_D, frame + fixed + decode(k_D) + read_served(k_D), exactly
    what `rows[].b[B]` holds for a budget cutting the trace there, so a deadline point and a budget point compare on
    one in-callo, single-generation clock (the simulated elapsed, chunk overhead included, stays in the raw row). A
    `budget0` fallback is phase A's readout, after the frame the infeasible plan was made on; a single-option question
    (no trace) is its budget-0 readout with no close, as serve answers it under any deadline."""
    b_rows = _phase_b(raw)
    out: dict[str, Measure] = {}
    for it in items:
        per, b = outs.get(it.uid) or {}, b_rows.get(it.uid)
        if b is None:
            continue
        t = b.get("trace")
        if not isinstance(t, dict):                      # a single-option question
            if 0 in per:
                out[it.uid] = per[0]
            continue
        e = (b.get("deadlines") or {}).get(str(deadline))
        if e is None:
            continue
        if e["close"] == "budget0":
            if 0 in per:
                m = per[0]
                out[it.uid] = dataclasses.replace(m, lat_ms=float(t.get("frame_ms") or 0.0) + m.lat_ms, close="budget0")
            continue
        n_gen, k = int(t["n_gen"]), int(e["k"])
        at_ngen = next((r for r in b["budgets"].values() if int(r["k"]) >= n_gen), None)
        p_ngen = at_ngen.get("p_ms") if at_ngen is not None else None
        lat = budget_latency(float(t.get("frame_ms") or 0.0), float(t["gen_wall_ms"]), t.get("predicted_ms"), n_gen, k,
                             float(e["read_ms"]), e.get("p_ms"), p_ngen)
        extra = max(0.0, float(e.get("p_ms") or 0.0) - float(p_ngen or 0.0)) if k < n_gen else 0.0
        correct, nll1, pmax = _row_fields(e["probs"], it.label)
        out[it.uid] = Measure(tuple(e["probs"]), correct, nll1, pmax, lat, k, str(e["close"]), int(e.get("miss") or 0),
                              float(e.get("option_mass") or 0.0), e.get("read_prompt_n"), bool(e.get("text_fallback")),
                              extra)
    return out


def _pq(xs: Sequence[float]) -> dict | None:
    """{p50, p95, p99} (nearest rank), or None for no values."""
    return {"p50": percentile(xs, 0.5), "p95": percentile(xs, 0.95), "p99": percentile(xs, 0.99)} if xs else None


def latency_block(raw: RawThink) -> dict:
    """The profile-level `latency` block (each {p50, p95, p99}, or null without values) from the phase-B
    generations: the frame; fixed = wall - predicted; the prefill per uncached token (prompt_n > 32); the decode step,
    predicted / (n_gen - 1) as `deadline.trace_terms` reads it (llama-server times the first token in the prompt phase);
    the prompt length; the readout after a natural close (at n_gen); and `read_forced_ms`, the served-like forced read
    (the audit's forced generations and the reserve pre-pass's used reads), null without either. Serve seeds its
    estimators from it."""
    b_rows = list(_phase_b(raw).values())
    traces = _traces(b_rows)
    natural = []
    for row in b_rows:
        t = row.get("trace")
        if isinstance(t, dict) and t.get("close") == "natural":
            at = next((r for r in row["budgets"].values() if int(r["k"]) >= int(t["n_gen"])), None)
            if at is not None:
                natural.append(float(at["read_ms"]))
    forced = [float(r["read_ms"]) for r in raw.rows if r.get("phase") == "audit" and isinstance(r.get("read_ms"), _NUM)
              and (r.get("trace") or {}).get("close") == "forced"]
    forced += reserve_reads((raw.reserve or {}).get("reads") or [])
    return {"frame_ms": _pq([float(t.get("frame_ms") or 0.0) for t in traces]),
            "fixed_ms": _pq([max(0.0, float(t["gen_wall_ms"]) - float(t.get("predicted_ms") or 0.0)) for t in traces]),
            "prefill_ms_per_token": _pq([float(t["gen_prompt_ms"]) / int(t["gen_prompt_n"]) for t in traces
                                         if t.get("gen_prompt_ms") and (t.get("gen_prompt_n") or 0) > 32]),
            "decode_ms_per_token": _pq([trace_terms(0.0, 0.0, t.get("predicted_ms"), int(t["n_gen"]))[1] for t in traces
                                        if t.get("predicted_ms") and int(t.get("n_gen") or 0) >= 2]),
            "read_natural_ms": _pq(natural), "read_forced_ms": _pq(forced),
            "on_tokens": _pq([float(t["on_tokens"]) for t in traces if isinstance(t.get("on_tokens"), int)])}


def row_terms(row: Mapping[str, Any] | None) -> list | None:
    """`rows[].t`: [n_gen, a_ms, d_ms], `deadline.trace_terms` of the question's phase-B trace (what the probe
    simulated its deadlines on, and what `deadline_map` re-simulates); None without a trace."""
    t = (row or {}).get("trace")
    if not isinstance(t, dict) or t.get("n_gen") is None:
        return None
    a, d = trace_terms(float(t.get("frame_ms") or 0.0), float(t["gen_wall_ms"]), t.get("predicted_ms"), int(t["n_gen"]))
    return [int(t["n_gen"]), a, d]


def probe_cost(raw: RawThink, decode_ms: float | None) -> dict:
    """The probe cost, from the live cost pass: c = p50(probe read_ms) + p50(max(0, the next chunk's wall_ms - n x
    d)), d the run's median decode step (`latency.decode_ms_per_token.p50`); c in decode steps; `restore_share`, the
    continuations after a probe that evaluated at most 2 tokens (a checkpoint restored, not a re-prefill), measured,
    never assumed. Probe i follows chunk i, so the chunk after it is i + 1. c is null without cost rows or a decode
    step."""
    reads, excess, restored = [], [], []
    for r in raw.rows:
        if r.get("phase") != "cost" or not isinstance(r.get("live"), dict):
            continue
        chunks, probes = r["live"].get("chunks") or [], r["live"].get("probes") or []
        for i, p in enumerate(probes):
            reads.append(float(p["read_ms"]))
            if i + 1 < len(chunks):
                nxt = chunks[i + 1]
                if decode_ms:
                    excess.append(max(0.0, float(nxt["wall_ms"]) - int(nxt["n"]) * decode_ms))
                if nxt.get("prompt_n") is not None:             # measured, never assumed
                    restored.append(int(nxt["prompt_n"]) <= 2)
    read = percentile(reads, 0.5) if reads else None
    c = read + percentile(excess, 0.5) if read is not None and excess and decode_ms else None
    return {"c_ms": c, "c_steps": c / decode_ms if c is not None and decode_ms else None, "read_ms": read,
            "restore_share": sum(restored) / len(restored) if restored else None, "n": len(reads)}


@dataclass(frozen=True)
class PointRow:
    """One question's (correct, lat_ms) at a point no `Measure` describes: an early-exit rule's (`by_kind["ee"]`)."""
    correct: bool
    lat_ms: float


@dataclass(frozen=True)
class Convergence:
    """`convergence_block`'s result: the profile's `converge` block, its top-level `ee` block, each
    question's cross-fitted rows by z ({uid: {z: PointRow}}) and the early-exit points by `ee_key(z)`; and the same
    three for the gated points (`ee_gated`, `grows`, `gpoints`: none without the gate's reads)."""
    block: dict
    ee: dict[str, dict]
    rows: dict[str, dict[float, PointRow]]
    points: dict[str, Point]
    gated: dict[str, dict] = dataclasses.field(default_factory=dict)
    grows: dict[str, dict[float, PointRow]] = dataclasses.field(default_factory=dict)
    gpoints: dict[str, Point] = dataclasses.field(default_factory=dict)


def _qtraces(items: Sequence[Item], raw: RawThink, outs: Mapping[str, Mapping[int, Measure]], bmax: int,
             sources: Mapping[str, str] | None) -> list[converge.QTrace]:
    """Every question with a Bmax outcome as the convergence fit reads it: its probes below n_gen, F (the readout at
    n_gen), `deadline.trace_terms`' a and d, and the Bmax latency as its end latency; a question without a trace (one
    option) has no probe and never stops."""
    b_rows = _phase_b(raw)
    out = []
    for it in items:
        m = (outs.get(it.uid) or {}).get(bmax)
        row = b_rows.get(it.uid)
        if m is None or row is None:
            continue
        src = sources.get(it.uid) if sources is not None else None
        t = row.get("trace")
        if not isinstance(t, dict):
            out.append(converge.QTrace(it.uid, it.cluster, it.label, 0, m.probs, (), 0.0, 0.0, m.lat_ms, src))
            continue
        n_gen = int(t["n_gen"])
        final = next((r for r in row["budgets"].values() if int(r["k"]) >= n_gen), None)
        a, d = trace_terms(float(t.get("frame_ms") or 0.0), float(t["gen_wall_ms"]), t.get("predicted_ms"), n_gen)
        probes = tuple(converge.Probe(int(p["k"]), tuple(p["probs"])) for p in row.get("probes") or []
                       if int(p["k"]) < n_gen)
        out.append(converge.QTrace(it.uid, it.cluster, it.label, n_gen,
                                   tuple(final["probs"]) if final is not None else m.probs, probes, a, d, m.lat_ms, src))
    return out


def convergence_block(items: Sequence[Item], raw: RawThink, settings: ThinkSettings,
                      outs: Mapping[str, Mapping[int, Measure]], douts: Mapping[int, Mapping[str, Measure]], *,
                      name: str) -> Convergence:
    """From the phase-B probes and the cost pass: the flip statistics, the cell table in sample (with the
    recorded monotone fit), the fixed sequence and the cross-fitted rule per z, the probe cost, each rule's early-exit
    point and its gain verdict against this model's budget and deadline points with every early-exit point on the
    frontier, and the rho map; plus the top-level `ee` entries (`acc`, `n`, `think_mean`, the five
    statistics, its cell and K) and each question's cross-fitted rows. Without a measured probe cost there is no
    early-exit point (no latency for it); the rules and the rho map (which sets c = rho x d itself) still are."""
    k, seed, bmax = int(settings.converge_probe or 0), settings.seed, settings.bmax
    sources = _sources(items)
    qs = _qtraces(items, raw, outs, bmax, sources)
    strata = [q.source or "unknown" for q in qs] if sources is not None else None
    lat = latency_block(raw)
    decode_ms = (lat["decode_ms_per_token"] or {}).get("p50")
    cost = probe_cost(raw, decode_ms)
    block: dict[str, Any] = {"K": k, "n": len(qs), "probes_per_question": sum(len(q.probes) for q in qs) / len(qs)
                             if qs else None, "flips": converge.flip_stats(qs, k), "cost": cost, "decode_ms": decode_ms}
    if not qs:
        return Convergence({**block, "cells": [], "sequence": {}, "rules": {}, "gain": {}, "rho": None,
                            "rho_max_per_token": None}, {}, {}, {})
    rho = cost["c_steps"]
    cells = {c: converge.cell_stats(qs, c, rho=rho, seed=seed, strata=strata) for c in converge.sequence_order()}
    mono = converge.monotone_fit(cells)
    sequences = {z: converge.fixed_sequence(cells, z, len(qs)) for z in converge.TARGETS}
    rules = {z: converge.rule_for(qs, cells, sequences[z], z, rho if rho is not None else 0.0, seed, strata)
             for z in converge.TARGETS}
    others = [converge.point_over(name, qs, [(float(outs[q.uid][b].correct), outs[q.uid][b].lat_ms) for q in qs],
                                  kind="budget", value=b, seed=seed, strata=strata)
              for b in settings.budgets if all(b in outs[q.uid] for q in qs)]
    others += [converge.point_over(name, qs, [(float(per[q.uid].correct), per[q.uid].lat_ms) for q in qs],
                                   kind="deadline", value=dl, seed=seed, strata=strata, cap=bmax)
               for dl, per in douts.items() if all(q.uid in per for q in qs)]
    read_ms = cost["read_ms"] if cost["read_ms"] is not None else (lat["read_forced_ms"] or {}).get("p50") or 0.0
    ees, verdicts = ({}, {}) if cost["c_ms"] is None else converge.gains(
        name, qs, rules, others, c_ms=cost["c_ms"], read_ms=read_ms, seed=seed, strata=strata, cap=bmax)
    rho_block = converge.rho_map(name, qs, cells, sequences, others, d_ms=decode_ms, read_ms=read_ms, seed=seed,
                                 strata=strata, cap=bmax) if decode_ms else None
    ee: dict[str, dict] = {}
    rows: dict[str, dict[float, PointRow]] = {}
    for key, e in ees.items():
        rule = rules[float(key)]
        assert rule is not None
        ee[key] = {"acc": e.point.acc, "n": e.point.n, "think_mean": e.point.think_mean, **e.point.stats(),
                   "cell": list(rule.cell), "K": k, "cap": bmax}        # it runs to the Bmax cap
        for q, c, x in zip(qs, e.correct, e.lat, strict=True):
            rows.setdefault(q.uid, {})[float(key)] = PointRow(bool(c), x)
    block.update({"cells": [{**c, "conv_mono": mono[cell]} for cell, c in cells.items()],
                  "sequence": {ee_key(z): seq for z, seq in sequences.items()},
                  "rules": {ee_key(z): (r.summary if r is not None else None) for z, r in rules.items()},
                  "gain": verdicts, "rho": rho_block,
                  "rho_max_per_token": rho_block["rho_max"] / k if rho_block and rho_block["rho_max"] and k else None})
    gate, reads = gate_block(items, raw, qs)
    if gate is None:                                    # no gate read: a profile as before the gate
        return Convergence(block, ee, rows, {key: e.point for key, e in ees.items()})
    block["gate"] = gate
    gated, grows, gpoints = {}, {}, {}
    for key, e in (ees.items() if reads is not None else ()):
        rule = rules[float(key)]
        assert rule is not None
        g = gated_point(name, qs, rule, e, reads, outs, bmax=bmax, seed=seed, strata=strata)
        gated[key] = {"acc": g.point.acc, "n": g.point.n, "think_mean": g.point.think_mean, "forced": g.point.forced,
                      **g.point.stats(), "dacc_cap": g.dacc_cap, "cell": list(rule.cell), "K": k, "cap": bmax,
                      "gate": {"flagged": gate["flagged"], "n": gate["n"], "read_ms_p50": gate["read_ms"]["p50"],
                               "read_ms_p95": gate["read_ms"]["p95"]}}
        for q, c, x in zip(qs, g.correct, g.lat, strict=True):
            grows.setdefault(q.uid, {})[float(key)] = PointRow(bool(c), x)
        gpoints[key] = g.point
    return Convergence(block, ee, rows, {key: e.point for key, e in ees.items()}, gated, grows, gpoints)


@dataclass(frozen=True)
class GateRow:
    """One question's gate read from the raw file: P(yes) (None: the read failed), its ms and the gate's sha256."""
    p: float | None
    ms: float
    sha256: str | None


def gate_block(items: Sequence[Item], raw: RawThink, qs: Sequence[converge.QTrace]) -> tuple[dict | None,
                                                                                          dict[str, GateRow] | None]:
    """The profile's `converge.gate` from the gate rows of the convergence questions with two or more options (serve
    reads no gate for the others), and those reads by uid when they make gated points: every such question read, all
    with one gate. (None, None) without a gate row (a raw file from before the gate). The block: the gate's `sha256`
    (null when the reads come from two gates), `skip_p` (`converge.GATE_SKIP_P`), the reads `n`, the `flagged` share
    (P(yes) >= skip_p, or a failed read: serve's `gate_skips`), the failed reads (`errors`), the read ms p50 / p95,
    and `why` there is no gated point when there is none."""
    latest = _latest(raw.rows, "gate")
    if not latest:
        return None, None
    multi = {it.uid for it in items if len(it.question.keys) >= 2}
    need = [q.uid for q in qs if q.uid in multi]
    reads = {u: GateRow(latest[u].get("p_yes"), float(latest[u]["ms"]), latest[u].get("sha256"))
             for u in need if u in latest}
    shas = {r.sha256 for r in reads.values()}
    ms = [r.ms for r in reads.values()]
    block: dict[str, Any] = {
        "sha256": next(iter(shas)) if len(shas) == 1 else None, "skip_p": converge.GATE_SKIP_P, "n": len(reads),
        "flagged": sum(converge.gate_skips(r.p) for r in reads.values()) / len(reads) if reads else None,
        "errors": sum(r.p is None for r in reads.values()),
        "read_ms": {"p50": percentile(ms, 0.5) if ms else None, "p95": percentile(ms, 0.95) if ms else None}}
    if len(reads) < len(need):
        block["why"] = f"{len(need) - len(reads)} of {len(need)} questions have no gate read: no gated point"
        return block, None
    if len(shas) > 1:
        block["why"] = "the gate reads come from two gates (a resume under another callo): no gated point"
        return block, None
    return block, reads


@dataclass(frozen=True)
class Gated:
    """A gated early-exit point (`kind="eeg"`) with its per-question (correct, lat_ms) in the convergence questions'
    order and its cross-fitted Δacc against the cap."""
    point: Point
    correct: tuple[float, ...]
    lat: tuple[float, ...]
    dacc_cap: list[float]


def gated_point(name: str, qs: Sequence[converge.QTrace], rule: converge.Rule, ee: converge.Scored,
                reads: Mapping[str, GateRow], outs: Mapping[str, Mapping[int, Measure]], *, bmax: int, seed: int,
                strata: Sequence[str] | None) -> Gated:
    """`<model>@ee<z>+g`, serve's gate replayed offline on calibration's own readings: a question the gate flags
    (`converge.gate_skips`) takes the cap's reading (the budget point at the early-exit point's cap: its answer, its
    latency, its k and close: no probe is read), the others the early-exit reading (`ee`'s cross-fitted row and the
    rule's outcome); every question with a gate read adds its read ms. Accuracy and the five statistics as an early-exit
    point's (`converge.point_over`, `kind="eeg"`), the think tokens the answer was read at, the forced share (the
    answer read at a cut of a trace that hit the cap) and the Δacc against the cap (F, the reading at n_gen), paired
    over the same questions with the same clusters, seed and strata as the rule's cross-fitted `dacc`."""
    rows, ks, forced = [], [], []
    for i, q in enumerate(qs):
        g, cap, o = reads.get(q.uid), outs[q.uid][bmax], rule.outcomes[q.uid]
        if g is not None and converge.gate_skips(g.p):
            c, lat, k, f = float(cap.correct), cap.lat_ms, cap.k, cap.close == "forced"
        else:
            c, lat, k, f = ee.correct[i], ee.lat[i], o.k, not o.stopped and cap.close == "forced"
        rows.append((c, lat + (g.ms if g is not None else 0.0)))
        ks.append(k)
        forced.append(f)
    s = converge.point_over(name, qs, rows, kind="eeg", value=rule.z, seed=seed, strata=strata, cap=bmax,
                            think_mean=sum(ks) / len(ks))
    final = [1.0 if q.final_argmax == q.label else 0.0 for q in qs]
    dacc = paired_bootstrap(final, list(s.correct), [q.cluster for q in qs], seed=seed, strata=strata)
    return Gated(s.point.replace(forced=sum(forced) / len(forced)), s.correct, s.lat, [float(x) for x in dacc])


def _sources(items: Sequence[Item]) -> dict[str, str] | None:
    """Each question's stratum for the bootstrap (clusters are resampled within sources when the
    data carries a source), decided once for the data: None, plain clusters, when no
    item has a source. An item without one is "unknown". `load_data` refuses a cluster that spans two sources; items
    built by hand that do so get plain clusters (no stratification can keep such a cluster whole)."""
    if all(it.source is None for it in items):
        return None
    out: dict[str, str] = {}
    of: dict[str, str] = {}
    for it in items:
        src = it.source or "unknown"
        if of.setdefault(it.cluster, src) != src:
            return None
        out[it.uid] = src
    return out


REUSE_CACHE_SHARE = 0.9              # gen_cache_n at or above this share of on_tokens: the question's state was restored
PREFILL_REUSED_MAX = 0.10            # above this share of reused generations a budget's latency cannot be `verified`


def prefill_reused(trace: dict, *, same_record: bool = False) -> bool | None:
    """The generation reused a cached state of its OWN question, which a served cold
    question never gets: `gen_prompt_n <= REPREFILL_PROMPT_N`, or `gen_cache_n >= 0.9 x on_tokens` (llama-server's host
    prompt cache restored phase A's state). None when the row carries neither number.

    `gen_cache_n` is the slot's longest common prefix (the generation sends `cache_prompt: true`), whatever put it
    there: after a question of the SAME record (same state / context) it legitimately covers all but the question
    text, which a served multi-question request on that slot gets too. So the 0.9 arm judges only a generation whose
    predecessor came from another record (`same_record` False); after a same-record one it
    is unjudged (None). The `<= 8` arm judges regardless: only the question's own thinking-off state leaves so little."""
    pn, cn, on = trace.get("gen_prompt_n"), trace.get("gen_cache_n"), trace.get("on_tokens")
    if pn is not None and int(pn) <= REPREFILL_PROMPT_N:
        return True
    cache_arm = None if cn is None or not on or same_record else int(cn) >= REUSE_CACHE_SHARE * int(on)
    if cache_arm is None:
        return None if pn is None else False
    return cache_arm


def _record_of(uid: str, items_by_uid: Mapping[str, Item]) -> str:
    it = items_by_uid.get(uid)
    return it.item_id if it is not None else uid.rsplit("/", 1)[0]


def _same_record(raw: RawThink, items: Sequence[Item]) -> set[int]:
    """The generation rows (`id()`) whose previous row in file order, any phase, asked a question of the same record:
    their slot held that record's shared prefix (`prefill_reused`'s `same_record`). Every row is a predecessor, a
    readout without a trace (phase A, the warm-up, a single-option question) included: it left its record's prefix
    on the slot too, so the first phase-B generation is judged against the last phase-A readout. A resumed session's
    first generation follows its own warm-up."""
    by_uid = {it.uid: it for it in items}
    out: set[int] = set()
    prev: str | None = None
    for r in raw.rows:
        rec = _record_of(r["q"], by_uid)
        if rec == prev and isinstance(r.get("trace"), dict):
            out.add(id(r))
        prev = rec
    return out


def _reuse_share(rows: Sequence[dict], same_record: set[int]) -> tuple[float | None, int, int]:
    """(the share of the rows' generations that reused their question's state, how many did, how many could be
    judged); a row without a trace (single-option) is not a generation."""
    flags = [prefill_reused(r["trace"], same_record=id(r) in same_record) for r in rows if isinstance(r.get("trace"), dict)]
    known = [f for f in flags if f is not None]
    return (sum(known) / len(known) if known else None), sum(known), len(known)


def _audit_summary(items: Sequence[Item], raw: RawThink, outs: dict[str, dict[int, Measure]], budgets: Sequence[int],
                   settings: ThinkSettings, same_record: set[int]) -> dict:
    """Per positive budget below Bmax, how many questions were cut there, how many audited, trace identity,
    argmax agreement and |Δp| on the identical-trace questions (m of them), the latency model's |relative error|
    against the real generation, and the share of generations, phase B and audit together, that reused their
    question's cached state: above PREFILL_REUSED_MAX the budget is `unverified` with `latency_reason`
    `prefill_reused`; an error above LATENCY_ERR_MAX is `unverified` with `lat_err`. The reuse share is decided
    independently of the audit: it flags an `unaudited` or `audit n/a` budget too. Phase B's one generation is EVERY
    positive budget's latency source, so phase B's share alone above the line flags every entry (pooling it with
    clean audit rows would only dilute it) and the profile, a run whose only positive budget is Bmax (no entries)
    included; `prefill_reused` wins over `lat_err`. `n` counts the audit rows the verdict rests on: those with a
    phase-B measure to compare against. `n_cut` is the size of the pool the audit draws from (`audit_pool`, the
    run's own function), so a budget whose pool holds fewer than AUDIT_MIN_CUTS questions is `audit n/a`, never an
    `unverified` resting on nothing."""
    b_rows = _phase_b(raw)
    order = shuffled(items, settings.seed)
    positive = [b for b in budgets if b > 0]
    if not positive:
        return {"latency_model": "unaudited", "latency_reason": None, "budgets": {}}
    bmax = max(positive)
    labels = {it.uid: it.label for it in items}
    b_share, b_reused, b_known = _reuse_share([r for q, r in b_rows.items() if q in labels], same_record)
    b_flagged = b_share is not None and b_share > PREFILL_REUSED_MAX
    per: dict[str, dict] = {}
    for b in positive:
        if b == bmax:
            continue
        cut = [it.uid for it in audit_pool(order, b_rows, settings.warmup, b) if b in outs.get(it.uid, {})]
        # the latest audit row per question: a hand-merged raw file with two rows for one (question, budget) never
        # counts it twice (the run itself never writes a done cell again)
        audited = list(_latest([r for r in raw.rows if r.get("phase") == "audit" and r.get("budget") == b
                                and r["q"] in labels and isinstance(r.get("trace"), dict)], "audit").values())
        a_share, a_reused, a_known = _reuse_share(audited, same_record)
        combined = (b_reused + a_reused) / (b_known + a_known) if b_known + a_known else 0.0
        reuse_ok = combined <= PREFILL_REUSED_MAX and not b_flagged
        compared = [r for r in audited if isinstance((b_rows.get(r["q"]) or {}).get("trace"), dict)
                    and b in outs.get(r["q"], {})]
        entry: dict[str, Any] = {"n_cut": len(cut), "n": len(compared)}
        if settings.audit == 0:
            verdict = "unaudited"
        elif len(cut) < AUDIT_MIN_CUTS:
            verdict = "audit n/a"
        else:
            identical = agree = 0
            errs, dps = [], []
            for r in compared:
                full, o = b_rows[r["q"]], outs[r["q"]][b]
                k = o.k
                same_trace = list(r["trace"]["tokens"]) == list(full["trace"]["tokens"])[:k]
                identical += same_trace
                if same_trace:                                      # agreement and |Δp| on the identical-trace questions
                    agree += max(range(len(r["probs"])), key=r["probs"].__getitem__) == \
                        max(range(len(o.probs)), key=o.probs.__getitem__)
                    dps.append(max(abs(float(x) - float(y)) for x, y in zip(r["probs"], o.probs, strict=True)))
                real = float(r["lat_ms"])
                if real > 0:
                    errs.append(abs(o.lat_ms - real) / real)
            entry.update({"trace_identical": identical, "argmax_agree": agree,
                          "dp_median": statistics.median(dps) if dps else None, "dp_max": max(dps) if dps else None,
                          "lat_err_median": statistics.median(errs) if errs else None,
                          "lat_err_max": max(errs) if errs else None})
            err_ok = len(errs) >= AUDIT_MIN_CUTS and statistics.median(errs) <= LATENCY_ERR_MAX
            verdict = "verified" if err_ok else "unverified"
        reason = None
        if not reuse_ok:                                      # with or without an audit verdict
            verdict, reason = "unverified", "prefill_reused"
        elif verdict == "unverified":
            reason = "lat_err"
        entry.update({"latency_model": verdict, "prefill_reused": a_share, "latency_reason": reason})
        per[str(b)] = entry
    models = [e["latency_model"] for e in per.values()]
    overall = "unverified" if "unverified" in models or b_flagged else "verified" if "verified" in models else \
        "audit n/a" if "audit n/a" in models else "unaudited"
    reasons = [e["latency_reason"] for e in per.values()]
    reason = "prefill_reused" if b_flagged or "prefill_reused" in reasons else \
        "lat_err" if overall == "unverified" else None
    return {"latency_model": overall, "latency_reason": reason, "budgets": per}


def _drift_summary(raw: RawThink) -> dict:
    """The median over the drift questions of end ms/token over phase-B ms/token (null without positive
    budgets), the thermal readings (null: "no thermal data", never evidence; each labelled with the session that took
    it when the raw file says, so a resumed run's second `start` is not mistaken for a repeat), and the flag."""
    b_rows = _phase_b(raw)
    ratios = []
    for r in raw.rows:
        if r.get("phase") != "drift":
            continue
        full = b_rows.get(r["q"])
        if full is None or not isinstance(r.get("trace"), dict) or not isinstance(full.get("trace"), dict):
            continue
        a, b = r["trace"], full["trace"]
        if a.get("predicted_ms") and b.get("predicted_ms") and a.get("n_gen") and b.get("n_gen"):
            ratios.append((a["predicted_ms"] / a["n_gen"]) / (b["predicted_ms"] / b["n_gen"]))
    ratio = statistics.median(ratios) if ratios else None
    therm = [t.get("limit") for t in raw.therm]
    readings = [t for t in therm if type(t) is int]                   # `is`: a bool is no reading
    flagged = (ratio is not None and abs(ratio - 1) > DRIFT_RATIO) or any(t < 100 for t in readings)
    readings_out = [{"at": t.get("at"), "limit": t.get("limit"),
                     **({"session": t["session"]} if t.get("session") is not None else {})} for t in raw.therm]
    return {"n": len(ratios), "ratio": ratio, "therm": readings_out or None, "flagged": bool(flagged)}


def budget_marks(points: Sequence[Point], dacc: Callable[[Point, Point], tuple[float, float, float]], *, latency: str,
                 paired: bool) -> dict[str, dict]:
    """The stored per-budget marks (`frontier`, `dominated_by`, `approx`, `dacc_prev`, `dacc_best`) by label, computed
    over the BUDGET points only (a profile's budget block is the same whether or not deadline or
    early-exit points exist; the displayed frontier and the pick take every kind). Δacc CIs between neighbouring
    frontier points and of every frontier point against the most accurate budget point; without paired evidence
    (`paired` false) the intervals are None, never invented."""
    ranked = frontier([p for p in points if p.kind == "budget"], latency=latency)
    front = mark_approx(on_frontier(ranked), dacc)
    most = min(front, key=lambda p: (-p.acc, p.latency(latency)[0])) if front else None
    out: dict[str, dict] = {}
    for p in ranked:
        on_front = p.dominated_by is None
        i = next((n for n, f in enumerate(front) if f.label == p.label), None)
        out[p.label] = {"frontier": on_front, "dominated_by": p.dominated_by,
                        "approx": front[i].approx if i is not None else False,
                        "dacc_prev": list(dacc(front[i - 1], p)) if i and paired else None,
                        "dacc_best": list(dacc(most, p)) if paired and on_front and most is not None
                        and most.label != p.label else None}
    return out


def build_profile(items: Sequence[Item], raw: RawThink, settings: ThinkSettings, *, sampling: Sampling, stats: dict,
                  name: str, started: str | None = None, ended: str | None = None,
                  skipped: Mapping[str, int] | None = None,
                  extra_kinds: Mapping[str, Mapping[str, Mapping[Any, Any]]] | None = None) -> dict:
    """The think profile from the raw file and the data: the per-budget metrics, the latency model, the points,
    frontier, `≈` marks and pick, the audit and drift verdicts, and the per-question rows compare pairs on. Pure:
    every number comes from `raw`, `items` and `skipped` (`load_data`'s Counter: the record total and what was
    filtered out go into `comparable.data`). `provenance` and `environment.build_info` are the raw header's, the
    run's first session: a `--resume` under another build (a warning, `header_diff`) measured most rows under the
    header's.

    Point sources: the budgets and, when the run measured them, the deadline points and the
    early-exit rules: each is a kind of `by_kind`, {kind: {uid: {value: an outcome with .correct and
    .lat_ms}}}, computed BEFORE the fixed row set and the pick, and each gets its `Point`s. `extra_kinds` adds the
    outcomes of other kinds (a test hook: they shape `fixed`, so the pick's pairing, and build no point); a kind the
    run produces itself ("budget", and "deadline" / "ee" when measured) is a ValueError, never silently replaced. A
    deadline no question was read at (each fell back to budget 0 or has one option) is no point and no `rows[].d` key:
    it is listed in `deadline_skipped` with why.

    The intervals rest on two row sets, never invented: the budget block (`budgets[B].dacc` / `dnll`, `dacc_prev`,
    `dacc_best`, the `≈` marks) on the questions measured at every budget (`bfixed`), absent or null when there are
    none; the pick's `dacc_best` / `dscore_best` and each deadline point's `dacc` on the questions measured at every
    point of every kind (`fixed`), null when there are none."""
    budgets = list(settings.budgets)
    b_rows_all = _phase_b(raw)
    stats = {**stats, **_stats_fields(stats), **header_stats(raw.header)}     # the run's own record first
    think = stats.get("think") or {}
    capable = bool(think.get("capable"))
    measured = [b for b in budgets if b == 0 or capable]
    outs = outcomes(items, raw, measured)
    by_uid = {it.uid: it for it in items}
    uids = [it.uid for it in items if it.uid in outs]
    # the deadline points' outcomes, from the probe's simulated cuts, before the fixed row set
    douts = {dl: deadline_outcomes(items, raw, dl, outs) for dl in (settings.deadlines if capable else ())}
    # a deadline no question was read at (every one fell back to budget 0, or had one option)
    # has no T_D to fit, so it is no point (serve reads it by the deadline temperature rules); the profile says why
    no_point = {str(dl): DEADLINE_SKIPPED for dl, dout in douts.items()
               if any(u in outs for u in dout) and not any(o.close not in (None, "budget0")
                                                          for u, o in dout.items() if u in outs)}
    douts = {dl: dout for dl, dout in douts.items() if str(dl) not in no_point}
    own: dict[str, dict[str, dict[Any, Any]]] = {}
    if douts:
        own["deadline"] = _by_uid(douts)
    # the early-exit rules' cross-fitted outcomes, before the fixed row set too
    cv = convergence_block(items, raw, settings, outs, douts, name=name) if settings.converge_probe and capable else None
    if cv is not None and cv.rows:
        own["ee"] = cv.rows
    if cv is not None and cv.grows:                                   # the gated points' rows, beside them
        own["eeg"] = cv.grows
    clash = sorted(set(extra_kinds or {}) & {"budget", *own})
    if clash:
        raise ValueError(f"extra_kinds adds point sources beside the run's own: {', '.join(clash)} is the run's")
    # every point source by kind, {kind: {uid: {value: an outcome with .correct and .lat_ms}}}
    by_kind: dict[str, Mapping[str, Mapping[Any, Any]]] = {"budget": outs, **own, **(extra_kinds or {})}
    # the pairing contract (think._bootstrap_weights): every paired interval, of every kind, resamples
    # ONE row set in ONE order, the questions measured at every point the run holds, stratified by source
    present = {kind: sorted({v for u in uids for v in per.get(u, {})}) for kind, per in by_kind.items()}
    fixed = [u for u in uids if all(v in by_kind[k].get(u, {}) for k, vs in present.items() for v in vs)]
    # the profile's budget block (budgets[B].dacc / dnll, dacc_prev, dacc_best, approx, the marks) rests on a BUDGET-ONLY
    # fixed set, the questions every budget holds: a deadline or early-exit point measured on fewer questions can never
    # shrink it or flip a ≈ (with the stored marks over budget points only); the all-kinds `fixed` serves the
    # pick and the displayed pairs
    bfixed = [u for u in uids if all(v in by_kind["budget"].get(u, {}) for v in present["budget"])]
    sources = _sources(items)                     # decided once for the data

    def strata_of(us: Sequence[str]) -> list[str] | None:
        return [sources[u] for u in us] if sources is not None else None

    fixed_cl, fixed_st = [by_uid[u].cluster for u in fixed], strata_of(fixed)
    bfixed_cl, bfixed_st = [by_uid[u].cluster for u in bfixed], strata_of(bfixed)
    same_record = _same_record(raw, items)
    entries: dict[str, dict] = {}
    points: list[Point] = []
    model_bytes = stats.get("model_bytes")
    gib = (model_bytes + (stats.get("mmproj_bytes") or 0)) / 2**30 if isinstance(model_bytes, int) else None
    for b in measured:
        qs = [u for u in uids if b in outs[u]]
        if not qs:
            continue
        rows = [([*outs[u][b].probs], by_uid[u].label) for u in qs]
        clusters = [by_uid[u].cluster for u in qs]
        strata = strata_of(qs)
        T = fit_T(rows)
        nll_T, ece_T = crossfit(rows, clusters, settings.seed)
        lats = [outs[u][b].lat_ms for u in qs]
        e: dict[str, Any] = {"n": len(qs), "acc": accuracy(rows), "nll": nll_at(rows, 1.0), "nll_T": nll_T,
                             "ece": ece_at(rows, 1.0), "ece_T": ece_T, "T": T,
                             **({"T_flat": True} if flat_fit(rows) else {}),   # T a tie-break, not a fit
                             **latency_stats(lats, clusters, seed=settings.seed, strata=strata),
                             "option_mass": sum(outs[u][b].option_mass for u in qs) / len(qs),
                             "low_mass": sum(outs[u][b].option_mass < 0.5 for u in qs)}
        if b > 0:
            if bfixed and 0 in present["budget"]:           # vs budget 0, over the budget-only fixed row set
                e["dacc"] = list(paired_bootstrap([float(outs[u][0].correct) for u in bfixed],
                                                  [float(outs[u][b].correct) for u in bfixed], bfixed_cl,
                                                  seed=settings.seed, strata=bfixed_st))
                e["dnll"] = list(paired_bootstrap([outs[u][0].nll1 for u in bfixed], [outs[u][b].nll1 for u in bfixed],
                                                  bfixed_cl, seed=settings.seed, strata=bfixed_st))
            ks = [outs[u][b].k for u in qs]
            e.update({"think_mean": sum(ks) / len(ks), "think_p95": percentile(ks, 0.95),
                      "forced": sum(outs[u][b].close == "forced" for u in qs) / len(qs),
                      "eos": sum(outs[u][b].close == "eos" for u in qs) / len(qs),
                      "readout_reprefill": sum((outs[u][b].read_prompt_n or 0) > REPREFILL_PROMPT_N for u in qs),
                      "text_fallback": sum(outs[u][b].text_fallback for u in qs),
                      # the share of this budget's generations that reused their question's cached state
                      "prefill_reused": _reuse_share([b_rows_all[u] for u in qs if u in b_rows_all], same_record)[0]})
        entries[str(b)] = e
        points.append(Point(name=name, kind="budget", budget=b, acc=e["acc"], n=len(qs), **point_latencies(e),
                            think_mean=e.get("think_mean"), forced=e.get("forced"), sampler=sampling.trace, gib=gib,
                            capable=capable))
    deadline_entries: dict[str, dict] = {}
    for dl, dout in douts.items():                                      # the deadline points
        qs = [u for u in uids if u in dout]
        if not qs:
            continue
        e = _deadline_entry(qs, dout, outs, b_rows_all, dl, fixed, fixed_cl, fixed_st, by_uid, strata_of, settings,
                            raw, has0=0 in present["budget"])
        deadline_entries[str(dl)] = e
        points.append(Point(name=name, kind="deadline", value=dl, cap=settings.bmax, acc=e["acc"], n=e["n"],
                            **point_latencies(e), think_mean=e["think_mean"], sampler=sampling.trace, gib=gib,
                            capable=capable))
    for p in ((*cv.points.values(), *cv.gpoints.values()) if cv is not None else ()):   # listed, never picked
        points.append(p.replace(sampler=sampling.trace, gib=gib, capable=capable))      # (opt-in), plain and gated

    def same_rows(x: Point, y: Point) -> bool:
        a, b = x.budgets[-1] if x.budgets else x.budget, y.budget
        qs = [u for u in uids if a in outs[u] and b in outs[u]]
        return bool(qs) and all(outs[u][a].probs == outs[u][b].probs and outs[u][a].k == outs[u][b].k for u in qs)

    def pair_rows(p: Point) -> tuple[list[float], list[float]]:
        """A point's (correct, lat_ms) over the fixed row set, keyed on its kind and value (a merged budget point
        reads its smallest budget's rows)."""
        per = by_kind[p.kind]
        return [float(per[u][p.value].correct) for u in fixed], [float(per[u][p.value].lat_ms) for u in fixed]

    # no paired evidence (no fixed rows): every Δ is (0, 0, 0), "within noise" for the marks and the pick, and never
    # recorded (below)
    paired = Paired(pair_rows, tuple(fixed_cl), settings.seed, tuple(fixed_st) if fixed_st is not None else None)
    merged = merge_identical(points, same_rows)
    chosen = pick(merged, paired=paired, pref=settings.preference,     # every point: the filters come first
                  skipped={name: {int(d) for d in no_point}},           # never hint a deadline it skipped
                  unservable=converge.unservable(merged, cv.block if cv is not None else None))
    bpaired = Paired(lambda p: ([float(by_kind["budget"][u][p.value].correct) for u in bfixed],
                                [float(by_kind["budget"][u][p.value].lat_ms) for u in bfixed]),
                     tuple(bfixed_cl), settings.seed, tuple(bfixed_st) if bfixed_st is not None else None)
    marks = budget_marks(merged, bpaired.dacc, latency=settings.latency, paired=bool(bfixed))
    for p in merged:
        if p.label in marks:                                         # budget points: their entries' marks
            for b in p.budgets or (p.budget,):
                entries[str(b)].update({"point": p.label, **marks[p.label]})
    audit = _audit_summary(items, raw, outs, measured, settings, same_record)
    drift = _drift_summary(raw)
    overhead = sum(o.reprefill_ms for per in outs.values() for o in per.values()) / 1000
    decode = [1000 * t["n_gen"] / t["predicted_ms"] for t in _traces(list(b_rows_all.values())) if t.get("predicted_ms")]
    h = raw.header
    server = h.get("server") or {}
    prov = h.get("provenance") if isinstance(h.get("provenance"), dict) else stats.get("provenance")
    pick_block = chosen.as_dict()                        # the preference it used included
    if not fixed:
        pick_block["dacc_best"] = pick_block["dscore_best"] = None
    profile = {
        "callo_think_profile": PROFILE_FORMAT, "metrics": METRICS_VERSION,
        "model_key": stats.get("model_key"), "model_name": name,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "comparable": {**settings.comparable(), "sampling": sampling.as_dict(),
                       "data": {**settings.comparable()["data"], "questions": len(items),
                                "records": (skipped or {}).get("records"),
                                "skipped": {k: v for k, v in sorted((skipped or {}).items()) if k != "records"},
                                "strata": "source" if sources is not None else None},
                       "readout": {"variant": "main", "top_k": DEFAULT_TOP_K, "max_depth": DEFAULT_MAX_DEPTH,
                                   "divergence_mass": DEFAULT_DIVERGENCE_MASS, "callo_version": __version__,
                                   "cache_prompt": settings.cache, "user_delimiter": stats.get("user_delimiter"),
                                   # present only when true (profile.provenance): --cache readouts send the delimiter
                                   **({"cache_delimiter": True} if (prov or {}).get("cache_delimiter") else {}),
                                   # the served wide contract, when the server publishes one
                                   **({"wide_contract": (prov or {})["wide_contract"]}
                                      if (prov or {}).get("wide_contract") is not None else {}),
                                   "prefix": "tokens"}},
        "environment": {"host": h.get("host") or stats.get("host"), "build_info": (prov or {}).get("build_info"),
                        "model": {"file": stats.get("model_file"), "bytes": model_bytes, "mmproj": (prov or {}).get("mmproj"),
                                  "mmproj_bytes": stats.get("mmproj_bytes")},
                        "server": {"kv": stats.get("kv"), "n_ctx": stats.get("n_ctx"), "total_slots": stats.get("total_slots"),
                                   "s1_slots": stats.get("s1_slots"), "slot": _phase_a_slot(raw),
                                   "server_args": (prov or {}).get("server_args"),
                                   # llama-server on callo's box? From /callo/stats (None: unknown, never true)
                                   "upstream_local": server.get("upstream_local") if server.get("upstream_local") is not None
                                   else stats.get("upstream_local"),
                                   "cache_ram": server.get("cache_ram"),     # the menu stamps 0 on a launch
                                   "gpu_busy": server.get("gpu_busy", False),
                                   "position_bias_sha256": stats.get("position_bias_sha256"),
                                   "launched_by": server.get("launched_by", "url")},
                        "run": {"started": started, "ended": ended,
                                "decode_tok_s_median": statistics.median(decode) if decode else None}},
        "provenance": prov,
        "thinking": {"capable": capable, "reason": think.get("reason"), "close": think.get("close"),
                     "synthesized_open": think.get("synthesized_open")},
        "server_state": {"state": "busy" if any(c.get("failed") for c in raw.checks) else "clean", "checks": list(raw.checks)},
        "audit": audit, "drift": drift, "budgets": entries,
        "pick": pick_block,
        "calibration_overhead_s": overhead, "row_fields": list(ROW_FIELDS),
        # each row's stratum, the one its bootstrap used (null: plain clusters), so compare re-runs the same
        # stratified pairing from the profile alone; rows stay in item order
        "rows": [{"q": u, "cluster": by_uid[u].cluster, "source": sources[u] if sources is not None else None,
                  "label": by_uid[u].label, "n": len(by_uid[u].question.keys),
                  "b": {str(b): [o.correct, o.nll1, o.pmax, o.lat_ms, o.k, o.close] for b, o in sorted(outs[u].items())},
                  "t": row_terms(b_rows_all.get(u)),                     # what deadline_map re-simulates
                  **({"d": {str(dl): [o.correct, o.nll1, o.pmax, o.lat_ms, o.k, o.close]
                            for dl, per in douts.items() if (o := per.get(u)) is not None}} if douts else {}),
                  **({"e": {ee_key(z): [o.correct, o.lat_ms] for z, o in sorted(cv.rows[u].items())}}
                     if cv is not None and u in cv.rows else {}),            # [correct, lat_ms] per rule
                  **({"g": {ee_key(z): [o.correct, o.lat_ms] for z, o in sorted(cv.grows[u].items())}}
                     if cv is not None and u in cv.grows else {})}           # the same, behind the gate
                 for u in uids]}
    profile["latency"] = latency_block(raw)
    if douts:
        profile["deadlines"] = deadline_entries
    if no_point:
        profile["deadline_skipped"] = no_point
    if cv is not None:                                                   # `ee` beside `deadlines`
        profile["converge"], profile["ee"] = cv.block, cv.ee
        if cv.gated:                                                     # `ee_gated` beside `ee`
            profile["ee_gated"] = cv.gated
    return profile


def _by_uid(by_value: Mapping[Any, Mapping[str, Any]]) -> dict[str, dict[Any, Any]]:
    """{value: {uid: outcome}} as `by_kind` holds a kind, {uid: {value: outcome}}."""
    out: dict[str, dict[Any, Any]] = {}
    for v, per in by_value.items():
        for u, o in per.items():
            out.setdefault(u, {})[v] = o
    return out


def _deadline_entry(qs: Sequence[str], dout: Mapping[str, Measure], outs: Mapping[str, Mapping[int, Measure]],
                    b_rows: Mapping[str, dict], deadline: int, fixed: Sequence[str], fixed_cl: Sequence[str],
                    fixed_st: Sequence[str] | None, by_uid: Mapping[str, Item],
                    strata_of: Callable[[Sequence[str]], list[str] | None], settings: ThinkSettings, raw: RawThink, *,
                    has0: bool) -> dict:
    """One deadline point: the cap it was simulated under (Bmax) and r_cal; the
    metrics a budget's entry has, its paired Δacc against budget 0 over the all-kinds fixed row set; the five
    latency statistics; the think tokens; the share of each close over every question (n: a single-option question,
    which has no trace and no close, counts `natural`, as serve answers it at once under any deadline and as compare's
    mapping counts it); the cut k's
    percentiles over the `deadline` closes; and the mean simulated chunks of the questions that thought (served
    latency adds about (chunks - 1) x ô to the in-callo lat_ms, the chunk overhead).

    T_D, nll and ece are fitted on the rows read at T_D alone: a budget-0 fallback is read at
    T_0 and a single-option question (no close) at budget 0, so neither is evidence about T_D; acc, n
    and the latencies count every question, as the point serves them. `build_profile` makes no point of a deadline
    with no row read at T_D, so there is always one to fit."""
    rows = [([*dout[u].probs], by_uid[u].label) for u in qs]
    clusters = [by_uid[u].cluster for u in qs]
    at_d = [u for u in qs if dout[u].close not in (None, "budget0")]
    fit_rows = [([*dout[u].probs], by_uid[u].label) for u in at_d]
    T = fit_T(fit_rows)
    nll_T, ece_T = crossfit(fit_rows, [by_uid[u].cluster for u in at_d], settings.seed)
    ks = [dout[u].k for u in qs]
    closes = Counter(dout[u].close or "natural" for u in qs)          # over every question (finding 3)
    cut = [dout[u].k for u in qs if dout[u].close == "deadline"]
    chunks = [int(((b_rows.get(u) or {}).get("deadlines") or {}).get(str(deadline), {}).get("chunks") or 0)
              for u in at_d]
    return {"cap": settings.bmax, "reserve_ms": (raw.reserve or {}).get("ms"), "n": len(qs), "acc": accuracy(rows),
            "dacc": list(paired_bootstrap([float(outs[u][0].correct) for u in fixed],
                                          [float(dout[u].correct) for u in fixed], fixed_cl, seed=settings.seed,
                                          strata=fixed_st)) if fixed and has0 else None,
            "T": T, **({"T_flat": True} if flat_fit(fit_rows) else {}),                 # T a tie-break, not a fit
            "nll": nll_at(fit_rows, 1.0), "nll_T": nll_T, "ece": ece_at(fit_rows, 1.0), "ece_T": ece_T,
            **latency_stats([dout[u].lat_ms for u in qs], clusters, seed=settings.seed, strata=strata_of(qs)),
            "think_mean": sum(ks) / len(ks), "think_p95": percentile(ks, 0.95),
            "closes": {c: (closes[c] / len(qs) if qs else 0.0) for c in DEADLINE_CLOSES},
            "k_p5": percentile(cut, 0.05) if cut else None, "k_p50": percentile(cut, 0.5) if cut else None,
            "k_p95": percentile(cut, 0.95) if cut else None,
            "chunks_mean": sum(chunks) / len(chunks) if chunks else None}


def header_stats(header: Mapping[str, Any]) -> dict:
    """The `/callo/stats` fields `build_profile` reads, as the raw header recorded them when the run started, in the
    shape of /callo/stats, plus the flat `s1_slots` and `position_bias_sha256` it reads too: only the fields the header
    holds, so an older raw file (no `HEADER_FIT_KEYS`) leaves those to the server's stats."""
    out: dict[str, Any] = {k: header[k] for k in ("model_key", "model_file", "provenance", "user_delimiter")
                           if k in header}
    if header.get("host") is not None:                 # None: callo could not say; the live stats may
        out["host"] = header["host"]
    srv = header.get("server")
    if isinstance(srv, dict):
        out.update({k: srv[k] for k in ("kv", "n_ctx", "total_slots", "s1_slots", "position_bias_sha256") if k in srv})
        if srv.get("upstream_local") is not None:
            out["upstream_local"] = srv["upstream_local"]
    if isinstance(header.get("thinking"), dict):
        out["think"] = dict(header["thinking"])
    if isinstance(header.get("model"), dict):
        out.update({"model_bytes": header["model"].get("bytes"), "mmproj_bytes": header["model"].get("mmproj_bytes")})
    return out


def _stats_fields(stats: Mapping[str, Any]) -> dict:
    """The flat `s1_slots` / `position_bias_sha256` from a live /callo/stats, for a header that does not hold them."""
    return {"s1_slots": len((stats.get("router") or {}).get("reserved") or []),
            "position_bias_sha256": (stats.get("position_bias") or {}).get("sha256")}


def _phase_a_slot(raw: RawThink) -> int | None:
    """The one slot the phase-A rows were read on, or None when there are none or they disagree."""
    slots = {int(r["slot"]) for r in raw.rows if r.get("phase") == "A" and r.get("slot") is not None}
    return slots.pop() if len(slots) == 1 else None


def pick_note(pick_block: Mapping[str, Any]) -> str | None:
    """The line the summary prints under the pick when `max_gib` excluded points of unknown size (`Pick.unsized`);
    None when there are none. `callo calibrate` has no --max-memory, so its list is always empty:
    the plumbing is shared with compare."""
    unsized = list(pick_block.get("unsized") or [])
    return f"size unknown: excluded by --max-memory: {', '.join(unsized)}" if unsized else None


def serve_note(pick_block: Mapping[str, Any]) -> str | None:
    """The line under the pick naming the early-exit points the serve check excluded (`converge.unservable`: `callo
    serve` cannot run them as measured, so there is nothing to opt into), each with why; None when there are none."""
    head = converge.UNSERVED + ": "
    excluded = pick_block.get("excluded")
    gone = [f"{label} ({why.removeprefix(head)})" for label, why in (excluded or {}).items()
            if isinstance(why, str) and why.startswith(head)] if isinstance(excluded, Mapping) else []
    return ("early exit `callo serve` cannot run as measured: " + "; ".join(gone)) if gone else None


EE_MARK = "opt-in (early exit)"           # an early-exit row's mark in the tables: listed, never picked


def opt_in_label(label: str, ee_labels: Collection[str]) -> str:
    """A dominating point's label in a mark, `m@ee.9 (opt-in)` when it is an early-exit point (the pick never takes
    it, so the pick can be the row it dominates)."""
    return f"{label} (opt-in)" if label in ee_labels else label


def opt_in_line(profile: Mapping[str, Any], name: str, *, latency: str = "mean") -> str | None:
    """The one line naming the early exit a user may opt into (the pick never takes it): the profile's servable
    early-exit point (`converge.unservable` names none of it) with the highest z, its saving on this data (the rule's
    cross-fitted tokens, net of probe cost when that was measured, and the `latency` statistic against the budget
    point at its cap) and the flag; when the profile also has that z's gated point (`m@ee<z>+g`, the gate measured),
    the gated variant beside it, with its own saving against the same budget point and `--think-converge-gate`. None
    without a servable early-exit point. `name` is the model's name in the labels (compare's, which can differ from
    `model_name`)."""
    if not profile.get("ee"):
        return None
    from .compare import points_of  # compare imports this module

    pts = points_of(dict(profile), name)
    gone = converge.unservable(pts, profile.get("converge"))
    ee = [p for p in pts if p.kind == "ee" and p.label not in gone]
    if not ee:
        return None
    p = max(ee, key=lambda q: float(q.value or 0.0))
    z = float(p.value or 0.0)
    budgets = [q for q in pts if q.kind == "budget"]
    ref = next((q for q in budgets if p.cap in (q.budgets or (q.budget,))), None) or \
        max(budgets, key=lambda q: max(q.budgets or (q.budget or 0,)), default=None)
    rule = ((profile.get("converge") or {}).get("rules") or {}).get(ee_key(z)) or {}
    saves = []
    if finite_number(rule.get("net_saving")) and rule["net_saving"] > 0:
        saves.append(f"~{rule['net_saving']:.0f} tokens")
    ms = ref.latency(latency)[0] - p.latency(latency)[0] if ref is not None else None
    if ms is not None and ms > 0:
        saves.append(f"~{ms:.0f} ms ({latency})")
    saving = f" saves {' / '.join(saves)}" if saves else ""
    g = next((q for q in pts if q.kind == "eeg" and q.value == p.value and q.label not in gone), None)
    gated = gate_flag = ""
    if g is not None:
        gms = ref.latency(latency)[0] - g.latency(latency)[0] if ref is not None else None
        gated = f"; gated ({g.label})" + (f" saves ~{gms:.0f} ms ({latency})" if gms is not None and gms > 0 else "")
        gate_flag = " (add --think-converge-gate for the gated one)"
    return (f"early exit (opt-in, not picked): {p.label}{saving} at z {_z(ee_key(z))} on this data{gated}; it holds "
            f"only on questions like these: serve with --think-converge {z:g}{gate_flag} to opt in")


def percentile(xs: Sequence[float], q: float) -> float:
    """Nearest-rank, as /callo/stats (proxy.percentile, re-stated here so the fit has no proxy import)."""
    s = sorted(xs)
    rank = -(-round(q * 10**6) * len(s) // 10**6)
    return s[max(0, min(len(s) - 1, rank - 1))]


# ---- the run: preconditions, the guard, preflight, warm-up, phases A and B, the audit, the drift ---

@dataclass(frozen=True)
class Outcome:
    """What `calibrate_one` ended with: `done` (the profile's path), `failed` (a request that failed for good, an
    unusable reply, bad data: the CLI's exit 1) or `refused` (a precondition, the guard, a slot change, a raw file that
    cannot be resumed: exit 2). The raw file keeps what was read either way."""
    status: str
    reason: str | None
    profile: Path | None


def check_preconditions(stats: dict, base: str, settings: ThinkSettings) -> None:
    """A callo with think budgets, a model_key, callo's engine on, and (--url mode) the run's --cache equal to
    the server's. Refusal (exit 2)."""
    if "think" not in stats:
        raise Refusal(f"refusing to calibrate: {base}/callo/stats has no think block, so this callo predates think "
                      f"budgets; upgrade callo and restart `callo serve`")
    if not stats.get("model_key"):
        why = stats.get("model_key_error") or "no reason given"
        raise Refusal(f"refusing to calibrate: callo serve could not compute the served model's model_key ({why}), "
                      f"and a profile is keyed by it")
    if stats.get("temperature") is None:
        raise Refusal("refusing to calibrate: callo's engine is off: the upstream is a native decision model, which "
                      "has no deliberate readout")
    plain = (stats.get("cache_prompt") or {}).get("plain")
    if plain is not None and bool(plain) != settings.cache:
        raise Refusal(f"refusing to calibrate: the server's --cache is {'on' if plain else 'off'} and this run's is "
                      f"{'on' if settings.cache else 'off'}; budget 0's latency depends on it. "
                      f"{'Drop --no-cache (--cache is on by default)' if plain else 'Pass --no-cache'}, or restart "
                      f"`callo serve` to match")


def think_max_note(think: Mapping[str, Any], settings: ThinkSettings) -> str | None:
    """The warning for budgets above the server's `--think-max` (`/callo/stats.think.max`), or None. Never a refusal:
    the probe ignores `--think-max` on purpose (calibration may explore past what the server serves), but a
    profile whose pick the server would refuse with a 422 should say how to serve it."""
    served = think.get("max")
    if type(served) is not int:
        return None
    above = [b for b in settings.positive if b > served]
    if not above:
        return None
    one = len(above) == 1
    return (f"WARNING: budget{'' if one else 's'} {', '.join(str(b) for b in above)} {'is' if one else 'are'} above "
            f"this server's --think-max {served}: {'it is' if one else 'they are'} measured, but serving "
            f"{'it' if one else 'them'} needs `callo serve --think-max {max(above)}` or more")


def not_capable_line(think: dict) -> str:
    """What calibrate prints for a model it measures at budget 0 only."""
    reason = think.get("reason") or "unknown"
    if reason == "unsupported_think_format":
        return "THINKS IN A FORMAT CALLO CANNOT READ: measured budget 0 only"
    return f"NO THINKING MODE ({reason}): measured budget 0 only"


class _Guard:
    """The server-state guard around a run: check 0 before the first request, then every CHECK_EVERY questions,
    at each phase boundary and after the last. `own` is the readouts the probe reported since the last check and
    `slack` the retried requests' possible extra readouts."""

    def __init__(self, base: str, stats: dict, raw: RawThink, kinds: Sequence[str], *, allow_busy: bool,
                 log: Callable[[str], None], sleep: Callable[[float], None]):
        self.base, self.start, self.raw, self.kinds = base, stats, raw, list(kinds)
        self.allow_busy, self.log, self.sleep = allow_busy, log, sleep
        self.slot: int | None = guard.predicted_slot(stats)
        self.mark: guard.Mark | None = None
        self.own = self.slack = 0
        self.session = 1 + max((c.get("session", 0) for c in raw.checks), default=-1)
        self.f: IO[str] | None = None

    def check(self, current: dict | None = None) -> None:
        current = current if current is not None else read_stats_mid_run(self.base, sleep=self.sleep)
        if current.get("instance") != self.start.get("instance"):
            raise Refusal("refusing to continue: callo restarted during calibration (/callo/stats instance changed); "
                          "rerun with --resume: the new session starts with a fresh check")
        v = guard.evaluate_state(current, slot=self.slot, kinds=self.kinds, prev=self.mark, own=self.own,
                                 slack=self.slack)
        rec = guard.record(len(self.raw.checks), self.session, len(self.raw.rows), v)
        self.mark, self.own, self.slack = v.mark, 0, 0
        self.raw.checks.append(rec)
        if self.f is not None:
            _write(self.f, {"check": rec})
        if v.failed:
            if not self.allow_busy:
                raise Refusal(guard.message(v, stats=current, kinds=self.kinds, url=self.base))
            self.log(f"WARNING: server-state check {rec['index']} failed ({', '.join(v.failed)}); --allow-busy: "
                     f"carrying on, and the profile will be marked busy")


def _trace_row(q: dict) -> dict | None:
    """The generation as the raw file keeps it; `on_tokens` (the prompt's length) is what `prefill_reused` compares
    `gen_cache_n` against. None for a single-option question, which the probe reads without a trace
    (`trace: null`): the fit reads it at every budget as its budget-0 answer."""
    if not isinstance(q.get("trace"), dict):
        return None
    t = dict(q["trace"])
    return {"tokens": list(t["tokens"]), "n_gen": t["n_gen"], "close": t["close"], "truncated": t.get("truncated"),
            "gen_wall_ms": t["gen_wall_ms"], "gen_prompt_ms": t.get("gen_prompt_ms"), "gen_prompt_n": t.get("gen_prompt_n"),
            "gen_cache_n": t.get("gen_cache_n"), "predicted_ms": t.get("predicted_ms"), "frame_ms": t.get("frame_ms"),
            "on_tokens": q.get("on_tokens")}


@contextlib.contextmanager
def _reply_shape(uid: str) -> Iterator[None]:
    """A 200 probe reply whose question block lacks what the row is built from (`budgets.<B>.probs`, `trace.n_gen`,
    ...) is RunFailed naming the question (exit 1), never a KeyError traceback; nothing is
    written for it, since the row is built before it is written."""
    try:
        yield
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
        raise RunFailed(f"{uid}: unusable reply ({type(e).__name__}: {e})") from None


def _n_gen(row: Mapping[str, Any]) -> int | None:
    """A phase-B row's trace length, or None for a single-option question's row (`trace: null`)."""
    t = row.get("trace")
    return int(t["n_gen"]) if isinstance(t, dict) else None


def _measured_items(order: Sequence[Item], b_rows: Mapping[str, dict], warmup: int) -> list[Item]:
    """The questions the audit and the drift draw from: those with a phase-B trace, in run order, the warm-up's
    questions last-resort only (they were read first, cold, in the first session)."""
    traced = [it for it in order if _n_gen(b_rows.get(it.uid) or {}) is not None]
    warm = {it.uid for it in order[:warmup]}
    return [it for it in traced if it.uid not in warm] or traced


def audit_pool(order: Sequence[Item], b_rows: Mapping[str, dict], warmup: int, budget: int) -> list[Item]:
    """The audit pool at `budget`, the ONE definition the run's audit and the profile's `n_cut` share: the measured,
    non-warm-up questions (`_measured_items`) whose Bmax trace was cut there (n_gen > budget), in run order. `order`
    is `shuffled(items, seed)` and `b_rows` the latest phase-B row per question."""
    return [it for it in _measured_items(order, b_rows, warmup) if (_n_gen(b_rows[it.uid]) or 0) > budget]


def _audit_plan(order: Sequence[Item], b_rows: Mapping[str, dict], settings: ThinkSettings,
                positive: Sequence[int]) -> list[tuple[int, list[Item]]]:
    """Per positive budget below Bmax, the questions to regenerate there: the first `--audit` of its
    `audit_pool`. A budget whose pool is smaller than AUDIT_MIN_CUTS is `audit n/a` and regenerates nothing."""
    if settings.audit <= 0:
        return []
    out = []
    for b in positive[:-1]:
        pool = audit_pool(order, b_rows, settings.warmup, b)
        if len(pool) >= AUDIT_MIN_CUTS:
            out.append((b, pool[: settings.audit]))
    return out


def _drift_plan(order: Sequence[Item], b_rows: Mapping[str, dict], settings: ThinkSettings) -> list[Item]:
    """The first DRIFT_N measured questions whose trace is at least DRIFT_MIN_GEN tokens."""
    items = _measured_items(order, b_rows, settings.warmup)
    return [it for it in items if (_n_gen(b_rows[it.uid]) or 0) >= DRIFT_MIN_GEN][:DRIFT_N]


def _cost_plan(order: Sequence[Item], b_rows: Mapping[str, dict], settings: ThinkSettings) -> list[Item]:
    """The first --converge-cost measured questions whose trace is at least 3K tokens."""
    k = settings.converge_probe
    if not k or settings.converge_cost <= 0:
        return []
    items = _measured_items(order, b_rows, settings.warmup)
    return [it for it in items if (_n_gen(b_rows[it.uid]) or 0) >= 3 * k][: settings.converge_cost]


def _gate_plan(order: Sequence[Item], settings: ThinkSettings, positive: Sequence[int]) -> list[Item]:
    """The questions early exit's gate is read on: with --converge-probe on a model that thinks, every question with
    two or more options (serve reads no gate for a request with nothing to think about), in run order."""
    if not (settings.converge_probe and positive):
        return []
    return [it for it in order if len(it.question.keys) >= 2]


def _pending(order: Sequence[Item], raw: RawThink, settings: ThinkSettings, positive: Sequence[int], *,
             gate: bool = True) -> bool:
    """Whether a session has anything left to read in `raw`: a phase-A or phase-B cell, or, every one
    of those in, a cell of the audit or drift plans, or a gate read (`gate`; False asks about the rest alone)."""
    done = cells(raw.rows)
    if any(("A", it.uid, None) not in done for it in order):
        return True
    if gate and any(("gate", it.uid, None) not in done for it in _gate_plan(order, settings, positive)):
        return True
    if not positive:
        return False
    if any(("B", it.uid, None) not in done for it in order):
        return True
    b_rows = _latest(raw.rows, "B")
    return any(("audit", it.uid, b) not in done for b, pool in _audit_plan(order, b_rows, settings, positive)
               for it in pool) or \
        any(("drift", it.uid, None) not in done for it in _drift_plan(order, b_rows, settings)) or \
        any(("cost", it.uid, None) not in done for it in _cost_plan(order, b_rows, settings))


def _readable_raw(path: Path) -> RawThink | None:
    """The raw file at `path`, or None when there is none or it cannot be read (`_open_raw` then says why)."""
    if not path.exists():
        return None
    try:
        return read_raw(path)
    except InvalidInput:
        return None


def refit_source(path: Path, items: Sequence[Item], settings: ThinkSettings, *,
                 gate: bool = True) -> tuple[RawThink, Sampling] | None:
    """A complete raw file this run refits with no server at all ("only refits, with no requests"): its header
    records everything the fit reads (`HEADER_FIT_KEYS`), its run-wide settings are this run's
    (`comparable`), and no cell is left to read. Returns the raw file and its recorded sampler, or None: no file, an
    unreadable or older one, other settings, or something to read; the run then goes through the server, which
    refuses or continues as for any resume. `gate` False leaves the gate reads out of "something to read": a
    --converge-probe raw file from before the gate, refitted when no server answers (it then has no gated point)."""
    src = _recorded_source(path, items, settings)
    if src is None:
        return None
    raw, order, positive = src
    smp = raw.header["comparable"].get("sampling")
    try:
        sampling = Sampling(str(smp["policy"]), dict(smp["pinned"]), dict(smp["origin"]), dict(smp["gguf"]),
                            smp.get("gguf_error"))
    except (KeyError, TypeError, ValueError):
        return None
    if _pending(order, raw, settings, positive, gate=gate):
        return None
    return raw, sampling


def _recorded_source(path: Path, items: Sequence[Item],
                     settings: ThinkSettings) -> tuple[RawThink, list[Item], list[int]] | None:
    """The raw file at `path` as read with no server: (the raw file, its run order, its positive budgets), or None
    when there is none, it is unreadable or older (no `HEADER_FIT_KEYS`), or it was measured with other run-wide
    settings (`comparable`, the sampler aside)."""
    raw = _readable_raw(path)
    if raw is None or any(k not in raw.header for k in HEADER_FIT_KEYS):
        return None
    items = probed_only(items, raw)[0]              # wide-choice: what that run's coverage kept
    comp = raw.header.get("comparable")
    if not isinstance(comp, dict) or {k: v for k, v in comp.items() if k != "sampling"} != settings.comparable():
        return None
    thinking = raw.header.get("thinking")
    positive = list(settings.positive) if isinstance(thinking, dict) and thinking.get("capable") else []
    return raw, shuffled(items, settings.seed), positive


GATE_READS = "gate"


def reads_left(path: Path, settings: ThinkSettings) -> str | None:
    """What the raw file of a model a menu run recorded as done has left to read under these settings (`_pending`, as
    `calibrate_one` decides it): `GATE_READS` when only gate reads are (a --converge-probe raw file from before the
    gate, whose --resume reads the gate alone), "reads" for anything else, None for nothing: no file, one that cannot
    be read with no server, or one measured with other settings (the menu then skips the model as before)."""
    try:
        items, _ = load_data(settings.data)
    except InvalidInput:
        return None
    src = _recorded_source(Path(path), items, settings)
    if src is None:
        return None
    raw, order, positive = src
    if not _pending(order, raw, settings, positive):
        return None
    return "reads" if _pending(order, raw, settings, positive, gate=False) else GATE_READS


def refit_candidates(out_dir: Path, settings: ThinkSettings) -> list[str]:
    """The `<name>`s of the raw files in `out_dir` that `refit_source` accepts for these settings (complete, same
    run-wide settings, gate reads aside): what a `--resume` can refit when the server does not answer. Empty on bad
    data."""
    try:
        items, _ = load_data(settings.data)
    except InvalidInput:
        return []
    return [p.name[: -len(".raw.jsonl")] for p in sorted(Path(out_dir).glob("*.raw.jsonl"))
            if refit_source(p, items, settings, gate=False) is not None]


def offline_name(out_dir: Path, settings: ThinkSettings, url: str, why: str) -> str:
    """`--resume` with the server down: the one complete raw file in `--out` for
    these settings is refit with no server; none is the usual "cannot read" refusal, and several is a refusal naming
    them, since callo cannot tell which model the absent server serves."""
    names = refit_candidates(out_dir, settings)
    if len(names) == 1:
        return names[0]
    if not names:
        flipped = refit_candidates(out_dir, dataclasses.replace(settings, cache=not settings.cache))
        hint = cache_resume_hint(not settings.cache, settings.cache) if flipped else ""
        raise Refusal(f"refusing to calibrate: cannot read {url}/callo/stats ({why}){hint}")
    raise Refusal(f"refusing to calibrate: {url} does not answer ({why}), and {out_dir} holds several complete raw "
                  f"files for these settings ({', '.join(f'{n}.raw.jsonl' for n in names)}), so callo cannot tell which "
                  f"model to refit: start the server that serves it (its model name picks the file), or point --out at "
                  f"a directory holding only that model's raw file")


def calibrate_one(settings: ThinkSettings, *, url: str, name: str, out_dir: Path, resume: bool,
                  log: Callable[[str], None], sleep: Callable[[float], None] = time.sleep,
                  clock: Callable[[], float] = time.monotonic,
                  therm: Callable[[], int | None] = therm_limit) -> Outcome:
    """One model, one running `callo serve` at `url`: the whole procedure, then the profile at
    `out_dir/<name>.think.json`. Never raises for a refusal or a failure: the Outcome says which."""
    try:
        return Outcome("done", None, _run(settings, url=url, name=name, out_dir=Path(out_dir), resume=resume, log=log,
                                          sleep=sleep, clock=clock, therm=therm))
    except Refusal as e:
        log(f"REFUSED: {e}")
        return Outcome("refused", str(e), None)
    except (RunFailed, InvalidInput) as e:
        log(f"FAILED: {e}")
        return Outcome("failed", str(e), None)
    except BrokenPipeError:                         # the console went away: the CLI's own exit, never "cannot write"
        raise
    except OSError as e:                            # the raw file, the profile, --out itself
        why = f"cannot write in --out {out_dir} ({e})"
        if Path(out_dir).exists() and not Path(out_dir).is_dir():
            why += "; with --think-budgets, --out is a DIRECTORY"
        log(f"FAILED: {why}")
        return Outcome("failed", why, None)


def _run(settings: ThinkSettings, *, url: str, name: str, out_dir: Path, resume: bool, log: Callable[[str], None],
         sleep: Callable[[float], None], clock: Callable[[], float], therm: Callable[[], int | None]) -> Path:
    base = url.rstrip("/")
    items, skipped = load_data(settings.data)
    if not items:
        raise RunFailed(f"--data {settings.data_name}: no labelled text questions (skipped: "
                        f"{', '.join(f'{k} {v}' for k, v in sorted(skipped.items())) or 'nothing'})")
    raw_path = out_dir / f"{name}.raw.jsonl"
    # a --resume on a complete raw file only refits, with no request at all (no /callo/stats, no preflight):
    # the header holds what the fit reads, so the server may be gone
    if resume and (refit := refit_source(raw_path, items, settings)) is not None:
        prior, sampling = refit
        log(f"refit: {raw_path} is complete; fitting it again with no request to the server")
        items, dropped = probed_only(items, prior)
        if dropped:
            skipped["too_many_options"] += dropped
        return _write_profile(items, prior, settings, sampling=sampling, stats={}, name=name, out_dir=out_dir,
                              skipped=skipped)
    prior = _readable_raw(raw_path) if resume else None
    if prior is not None and (missing := [k for k in HEADER_FIT_KEYS if k not in prior.header]):
        log(f"NOTE: {raw_path.name} was written before callo recorded {', '.join(missing)} in the raw header; the fit "
            f"reads them from {base}/callo/stats, so this resume needs the server")
    try:
        stats = read_stats(base)
    except (OSError, ValueError) as e:
        # a --converge-probe raw file from before the gate, complete but for its gate reads, still refits with no
        # server: it then has no gated point, and a --resume with the server up adds the reads
        if resume and (refit := refit_source(raw_path, items, settings, gate=False)) is not None:
            prior, sampling = refit
            log(f"refit: {raw_path} has no gate read and {base} does not answer ({e}); fitting it with no gated "
                f"early-exit point (a --resume with the server up adds the gate reads alone)")
            items, dropped = probed_only(items, prior)
            if dropped:
                skipped["too_many_options"] += dropped
            return _write_profile(items, prior, settings, sampling=sampling, stats={}, name=name, out_dir=out_dir,
                                  skipped=skipped)
        raise Refusal(f"refusing to calibrate: cannot read {base}/callo/stats ({e})") from None
    check_preconditions(stats, base, settings)
    items, dropped, limit = coverage_filter(items, stats)          # before any probe
    if dropped:
        skipped["too_many_options"] += dropped
        log(coverage_skip_line(dropped, limit, stats))
        if not items:
            raise RunFailed(f"--data {settings.data_name}: no labelled text question this model can read (skipped: "
                            f"{', '.join(f'{k} {v}' for k, v in sorted(skipped.items()) if k != 'records')})")
    think = stats["think"]
    capable = bool(think.get("capable"))
    positive = list(settings.positive) if capable else []
    if not capable:
        log(not_capable_line(think))
    elif (note := think_max_note(think, settings)) is not None:
        log(note)
    # the served GGUF's sampling header, and why it is empty when callo could not read it
    sampling = resolve_sampling(parse_sampling(settings.sampling), think.get("gguf_sampling") or {},
                                gguf_error=think.get("gguf_error"))
    s1_slots = len((stats.get("router") or {}).get("reserved") or [])
    header = raw_header(stats, settings, sampling)
    order = shuffled(items, settings.seed)
    client = ProbeClient(base, s1_slots=s1_slots, sleep=sleep)
    # preflight, in plan mode (no slot, no section, no readout), BEFORE the raw file exists: a refusal
    # here leaves nothing behind, so the rerun with a larger --ctx needs no --resume and no clean-up. A
    # resume with nothing left to read (an older raw file's refit) skips it: it would only cost requests
    if prior is not None and not _pending(order, prior, settings, positive, gate=False):
        pass                                # gate reads alone left: budget 0, nothing to fit in the context
    elif positive:
        worst = 0
        for it in order:
            n, n_ctx = client.plan(it)
            worst = max(worst, n)
            if n_ctx is not None and n + settings.bmax + 16 > n_ctx:
                raise Refusal(f"refusing to calibrate: question {it.uid} needs {n + settings.bmax + 16} tokens at "
                              f"budget {settings.bmax} ({n} prompt tokens), the slot context is {n_ctx}; raise --ctx "
                              f"or lower the budgets")
        log(f"plan: {len(order)} questions x budgets {list(settings.budgets)}; worst case {len(order) * settings.bmax} "
            f"generated tokens plus {min(settings.warmup, len(order)) * settings.bmax} for the warm-up; the longest "
            f"prompt is {worst} tokens; time: unknown until measured")
    else:
        log(f"plan: {len(order)} questions at budget 0 only")
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh = not raw_path.exists()
    f, raw = _open_raw(raw_path, header, resume=resume, log=log)
    try:
        done = cells(raw.rows)
        kinds = sorted({it.kind for it in items})
        g = _Guard(base, stats, raw, kinds, allow_busy=settings.allow_busy, log=log, sleep=sleep)
        g.f = f
        slot: int | None = None
        for r in raw.rows:
            if r.get("slot") is not None:
                slot = int(r["slot"])

        def put(row: dict) -> None:
            if (why := _row_problem(row)) is not None:      # never write a row read_raw would refuse
                raise RunFailed(f"{row['q']}: unusable reply ({why})")
            raw.rows.append(row)
            _write(f, row)

        def therm_at(at: str) -> None:
            entry = {"at": at, "limit": therm(), "session": g.session}
            raw.therm.append(entry)
            _write(f, {"therm": entry})

        def read(it: Item, budgets: Sequence[int], extras: Mapping[str, Any] | None = None) -> dict:
            nonlocal slot
            ask = (it, budgets, think_seed(str(settings.seed), it.item_id, it.qid), sampling.pinned)
            reply = client.read(*ask) if extras is None else client.read(*ask, extras)   # only when asked
            g.own += reply.readouts
            # a retried request may have been answered twice: its budgets' reads, the trace's own, each
            # deadline's cut and the reserve read, and a convergence probe at every K below n_gen <= max
            # budget, or the cost pass's at every K below its cut, any of which may be a distinct read (the cost
            # pass's 2 + (bmax-1)//K >= ceil(bmax/K))
            ex = extras or {}
            every = ex.get("converge_probe") or (ex.get("live") or {}).get("probe_every")
            g.slack += reply.retries * (len(budgets) + 1 + len(ex.get("deadlines") or ()) + bool(ex.get("reserve_read"))
                                        + (max(0, max(budgets) - 1) // every if every else 0))
            if reply.retries:
                # a dropped connection is a restart's signature: the retry may have been answered by a NEW callo
                # (cold decode, empty caches). Check now, so such a row is refused instead of written
                g.check()
            if reply.slot is None:
                raise Refusal(f"refusing to continue: {it.uid} came back without a usable X-Callo-Slot")
            if slot is None:
                slot = reply.slot
            elif reply.slot != slot:
                raise Refusal(f"refusing to continue: calibrate's slot changed from {slot} to {reply.slot} at {it.uid} "
                              f"(a lane rebind or the S1 fallback moved it, and readouts from two slots' states cannot "
                              f"be mixed). Restart `callo serve` or remove the raw file")
            return reply.body["questions"][it.qid]

        todo_a = [it for it in order if ("A", it.uid, None) not in done]
        todo_g = [it for it in _gate_plan(order, settings, positive) if ("gate", it.uid, None) not in done]
        todo_b = [it for it in order if ("B", it.uid, None) not in done] if positive else []
        pending = _pending(order, raw, settings, positive)        # the audit and drift cells too
        # a session with only gate reads left (a --converge-probe raw file from before the gate) thinks nothing: its
        # warm-up reads budget 0 alone
        thinks = _pending(order, raw, settings, positive, gate=False)
        if not pending:
            log(f"refit: {raw_path} is complete; fitting it again")
        # a session with nothing left to measure (a --resume on a complete raw file: the settings may
        # change the fit, never the measurement) makes no request, takes no thermal reading and runs no guard check,
        # so the refit cannot flip `drift.flagged` or `server_state`
        if pending:
            try:
                g.check(stats)
            except Refusal:
                if fresh:               # refused before any request: nothing a rerun would need --resume for
                    f.close()
                    raw_path.unlink(missing_ok=True)
                raise
            therm_at("start")
            all_budgets = [0, *positive]
            warm_budgets = all_budgets if thinks else [0]
            measured, todo, t0 = 0, 0, clock()

            def phase(todo_items: Sequence[Item]) -> None:
                nonlocal measured, todo, t0
                measured, todo, t0 = 0, len(todo_items), clock()

            def tick(phase_label: str) -> None:
                nonlocal measured
                measured += 1
                if measured == ETA_AFTER:
                    per = (clock() - t0) / measured
                    log(f"  measured {per:.2f} s per question in {phase_label}: about {per * (todo - measured) / 60:.1f} "
                        f"min for the {todo - measured} left in this phase")
                if measured % CHECK_EVERY == 0:
                    g.check()

            # step 5: warm-up, discarded. Every SESSION that makes a request warms up first: a
            # resumed run's first reads would otherwise be cold (shader compile, page-in), and the audit's latency error
            # and the drift ratio (against the WARM phase-B speed) are the reads a cold start skews most. Re-fitting a
            # complete raw file makes no request.
            # with --deadlines and no --think-deadline-reserve, the warm-up of the session that has no
            # r_cal yet is also the reserve pre-pass, one served-like forced read per question; r_cal is then fixed for
            # the run (a `reserve` line in the raw file), whatever a later session's warm-up reads
            pre_pass = bool(settings.deadlines and positive) and settings.reserve_ms is None and raw.reserve is None \
                and thinks
            reserve_reads_ms: list[float | None] = []                # one per warm-up question (None: no read)
            for it in order[: settings.warmup]:
                q = read(it, warm_budgets, {"reserve_read": True} if pre_pass else None)
                with _reply_shape(it.uid):
                    put({"phase": "warmup", "q": it.uid, "slot": slot, "session": g.session,
                         "budgets": list(warm_budgets), "ms": q["budgets"]["0"]["ms"]})
                    rr = q.get("reserve_read")
                    if pre_pass and len(it.question.keys) < 2:      # single-option: no trace, no read; question 1
                        reserve_reads_ms.append(None)                # stays the skipped one by its index
                    elif pre_pass:
                        if not isinstance(rr, dict):
                            raise RunFailed(f"{it.uid}: {NO_DEADLINE_PROBES}")
                        t = q["trace"]                               # a 3-token first pass is one step short of 4
                        step = trace_terms(0.0, 0.0, t.get("predicted_ms"), int(t["n_gen"]))[1]
                        reserve_reads_ms.append(float(rr["read_ms"]) + (step if rr.get("add_step") else 0.0))
            if settings.deadlines and positive and raw.reserve is None and thinks:
                raw.reserve = reserve_record(settings, reserve_reads_ms)
                _write(f, {"reserve": raw.reserve})
                log(f"deadlines: readout reserve r_cal {raw.reserve['ms']:.1f} ms ({raw.reserve['source']})")
            g.check()
            phase(todo_a)
            for it in todo_a:                                                  # step 6: phase A
                q = read(it, [0])
                with _reply_shape(it.uid):
                    b0 = q["budgets"]["0"]
                    put({"phase": "A", "q": it.uid, "slot": slot, "probs": b0["probs"], "ms": b0["ms"],
                         "miss": b0["miss"], "option_mass": b0["option_mass"], "prompt_n": b0.get("prompt_n"),
                         "stop": b0.get("stop")})
                tick("phase A")
            therm_at("A")
            g.check()
            if todo_g:                                                         # early exit's gate, at budget 0
                phase(todo_g)
                for it in todo_g:
                    q = read(it, [0], {"gate": True})
                    if not isinstance(q.get("gate"), dict):                    # an older serve drops the key
                        raise RunFailed(f"{it.uid}: {NO_GATE_PROBES}")
                    with _reply_shape(it.uid):
                        gr = q["gate"]
                        put({"phase": "gate", "q": it.uid, "slot": slot, "p_yes": gr["p"], "ms": gr["ms"],
                             "sha256": gr.get("sha256"), "prompt_n": gr.get("prompt_n"),
                             "cache_n": gr.get("cache_n"),         # the gate's prefix reuse, measurable
                             **({"error": gr["error"]} if gr.get("error") is not None else {})})
                    tick("the gate reads")
                g.check()
            if positive:                                                       # step 7: phase B
                phase(todo_b)
                b_extras: dict[str, Any] | None = None
                if settings.deadlines and raw.reserve is not None:             # the simulated cuts
                    b_extras = {"deadlines": list(settings.deadlines), "reserve_ms": raw.reserve["ms"]}
                if settings.converge_probe:                                    # the probes ride along
                    b_extras = {**(b_extras or {}), "converge_probe": settings.converge_probe}
                for it in todo_b:
                    q = read(it, positive, b_extras)
                    with _reply_shape(it.uid):
                        row = {"phase": "B", "q": it.uid, "slot": slot, "trace": _trace_row(q),
                               "budgets": {k: v for k, v in q["budgets"].items() if k != "0"}}
                        if b_extras is not None and "deadlines" in b_extras and len(it.question.keys) >= 2:
                            if not isinstance(q.get("deadlines"), dict):
                                raise RunFailed(f"{it.uid}: {NO_DEADLINE_PROBES}")
                            row["deadlines"] = q["deadlines"]
                        if settings.converge_probe and len(it.question.keys) >= 2:
                            if not isinstance(q.get("probes"), list):      # an older serve drops the key
                                raise RunFailed(f"{it.uid}: {NO_CONVERGE_PROBES}")
                            row["probes"] = [{"k": p["k"], "probs": p["probs"], "read_ms": p["read_ms"],
                                              "read_prompt_n": p.get("read_prompt_n")} for p in q["probes"]]
                        put(row)
                    tick("phase B")
                therm_at("B")
                g.check()
                b_rows = _latest(raw.rows, "B")
                audit_plan = _audit_plan(order, b_rows, settings, positive)
                for b, pool in audit_plan:                                     # step 8: the cut audit
                    for it in pool:
                        if ("audit", it.uid, b) in done:
                            continue
                        q = read(it, [b])
                        with _reply_shape(it.uid):
                            r = q["budgets"][str(b)]
                            t = q["trace"]
                            # the real generation at b: k >= n_gen, so the latency is frame + gen_wall + read_ms
                            put({"phase": "audit", "q": it.uid, "budget": b, "slot": slot, "trace": _trace_row(q),
                                 "probs": r["probs"], "k": r["k"], "read_ms": r["read_ms"],
                                 "lat_ms": budget_latency(float(t.get("frame_ms") or 0.0), float(t["gen_wall_ms"]),
                                                          t.get("predicted_ms"), int(t["n_gen"]), int(t["n_gen"]),
                                                          float(r["read_ms"]), None, None)})
                if settings.audit > 0:
                    g.check()
                for it in _drift_plan(order, b_rows, settings):                # step 9: the drift check
                    if ("drift", it.uid, None) in done:
                        continue
                    q = read(it, [settings.bmax])
                    with _reply_shape(it.uid):
                        put({"phase": "drift", "q": it.uid, "slot": slot, "trace": _trace_row(q)})
                cost_plan = _cost_plan(order, b_rows, settings)
                for it in cost_plan:                                           # the probe-cost pass
                    if ("cost", it.uid, None) in done:
                        continue
                    q = read(it, [settings.bmax], {"live": {"deadline_ms": None, "probe_every": settings.converge_probe}})
                    if not isinstance(q.get("live"), dict):                    # an older serve drops the key
                        raise RunFailed(f"{it.uid}: {NO_CONVERGE_PROBES}")
                    with _reply_shape(it.uid):
                        lv = q["live"]
                        put({"phase": "cost", "q": it.uid, "slot": slot,
                             "live": {"close": lv["close"], "n_gen": lv.get("n_gen"),
                                      "chunks": [{"n": c["n"], "wall_ms": c["wall_ms"], "prompt_n": c.get("prompt_n"),
                                                  "prompt_ms": c.get("prompt_ms"), "predicted_ms": c.get("predicted_ms")}
                                                 for c in lv.get("chunks") or []],
                                      "probes": [{"k": p["k"], "read_ms": p["read_ms"],
                                                  "read_prompt_n": p.get("read_prompt_n")}
                                                 for p in lv.get("probes") or []]}})
                if cost_plan:
                    g.check()
                therm_at("end")
            g.check()
        # the run started at its first session's check 0, whatever session fits it
        started = (raw.checks[0].get("at") if raw.checks else None) or \
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    finally:
        with contextlib.suppress(OSError):
            f.close()
    return _write_profile(items, read_raw(raw_path), settings, sampling=sampling, stats=stats, name=name,
                          out_dir=out_dir, skipped=skipped, started=started)


def _write_profile(items: Sequence[Item], raw: RawThink, settings: ThinkSettings, *, sampling: Sampling, stats: dict,
                   name: str, out_dir: Path, skipped: Mapping[str, int], started: str | None = None) -> Path:
    """Fit `raw` and write `out_dir/<name>.think.json` as strict JSON. `started` defaults to the run's
    first check (the run started at its first session's check 0, whatever session fits it)."""
    check_rows(raw, out_dir / f"{name}.raw.jsonl")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if started is None:
        started = (raw.checks[0].get("at") if raw.checks else None) or now
    profile = build_profile(items, raw, settings, sampling=sampling, stats=stats, name=name, started=started, ended=now,
                            skipped=skipped)
    path = out_dir / f"{name}.think.json"
    tmp = path.with_name(path.name + ".tmp")        # a Ctrl-C or a full disk never truncates a good profile (minor 5)
    try:
        tmp.write_text(json.dumps(json_safe(profile), indent=1, allow_nan=False) + "\n")     # strict JSON
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return path


# ---- the summary, --dry-run and the CLI entry ------------------------------

DEFAULT_OUT_DIR = "callo-think"
SERVE_START_TIMEOUT_S = 600.0            # an in-process `callo serve --upstream` attaches to a loaded model: seconds
SMALL_N = 100                            # below this, frontier neighbours are mostly within noise
SERVED_HIGHER = ("  with --cache-ram 0 for calibration, served latency can come out slightly higher than calibrated on a "
                 "server that keeps the host cache (llama-server's default): each question switch there also pays a "
                 "state save, a few ms, more on big dense models")


def _s(ms: float | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.2f}s"


def _num(v: float | None, spec: str, suffix: str = "", scale: float = 1.0) -> str:
    return "-" if v is None else f"{format(scale * v, spec)}{suffix}"


def _ci(v: Sequence[float] | None, *, pct: bool = False) -> str:
    if not v:
        return "-"
    p, lo, hi = v
    if pct:
        return f"{100 * p:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return f"{_s(p)} [{_s(lo)}, {_s(hi)}]"


def profile_name(stats: dict) -> str:
    """`<name>` is `/callo/stats.model_file` without `.gguf` in the attach modes (or `model` without one)."""
    file = stats.get("model_file")
    return Path(file).stem if isinstance(file, str) and file else "model"


def summary_lines(profile: dict, *, verbose: bool = False, path: Path | None = None) -> list[str]:
    """A header line, one row per point, the audit and drift verdicts, the pick line with the most
    accurate point beside it, and the `serve with:` line. `--verbose` adds NLL@1, NLL@T, ECE10@T and T. Under the
    prefill-reuse line, when calibration ran (or is told to run) with llama-server's host cache off, the note that a
    server keeping it pays a per-question state save calibration did not; just before the pick line, the points whose
    T sits on the fit grid's first or last point (`calibrate.limit_note`); under the pick line, the
    points `--max-memory` excluded for an unknown size (`pick_note`), the early-exit points `callo serve` cannot run
    (`serve_note`) and the early exit to opt into (`opt_in_line`: listed, never picked)."""
    env = profile.get("environment") or {}
    model, host = env.get("model") or {}, env.get("host") or {}
    comp = profile.get("comparable") or {}
    data, sampling = comp.get("data") or {}, comp.get("sampling") or {}
    pinned, origin = sampling.get("pinned") or {}, sampling.get("origin") or {}
    gib = None
    if isinstance(model.get("bytes"), int):
        gib = (model["bytes"] + (model.get("mmproj_bytes") or 0)) / 2**30
    fields = ("temperature", "top_p", "top_k", "min_p")
    sampler = "/".join(str(pinned.get(k, "-")) for k in fields)
    origins = ", ".join(f"{k}={origin.get(k, 'callo')}" for k in fields)
    name = profile.get("model_name")
    server = env.get("server") or {}
    lines = [f"{name}  {model.get('file') or '?'}" + (f"  {gib:.1f} GiB" if gib is not None else "") +
             f"  model_key {str(profile.get('model_key') or '?')[:16]}  data {data.get('name')} "
             f"({data.get('questions')} questions)  host {host.get('chip') or host.get('machine') or '?'}  "
             f"latency: single-stream on this host, in-callo wall time",
             f"sampler {sampler} ({sampling.get('policy')}; {origins})  server state "
             f"{(profile.get('server_state') or {}).get('state')}"
             + ("" if server.get("server_args") is not None else
                "; server args not recorded (attach mode: -ctk/-ctv, MTP and draft settings unknown)")]
    thinking = profile.get("thinking") or {}
    if not thinking.get("capable"):
        lines.append(not_capable_line(thinking))
    # the five statistics, the interval on the pick statistic's column (the axis the frontier and the pick used),
    # the others plain
    axis = (profile.get("pick") or {}).get("latency")
    axis = axis if axis in LATENCY_AXES else "mean"
    head = f"{'point':<24} {'acc':>6} {'Δacc vs 0 [95% CI]':>28}"
    head += "".join(f" {st + ' [95% CI]':>24}" if st == axis else f" {st:>8}" for st in STAT_FIELDS)
    head += f" {'think':>6} {'forced':>7}"
    if verbose:
        head += f" {'NLL@1':>7} {'NLL@T':>7} {'ECE10@T':>8} {'T':>5}"
    lines.append(head)
    budgets = profile.get("budgets") or {}
    seen: set[str] = set()
    tails: list[int] = []
    fitted: list[tuple[str, Any]] = []             # (label, T) of every row with a real fit, for the search-limit note
    # with deadline or early-exit points, every row's mark is compare's one frontier over every kind
    # (display only: the stored marks stay budget-only), so the summary agrees with `callo compare`. The pick leaves
    # early exit out (opt-in), so its rows and a row an early-exit point dominates say so: the pick can be such a row
    shown: dict[str, Any] = {}
    if profile.get("deadlines") or profile.get("ee"):
        from . import compare  # compare imports this module
        shown = compare.frontier_marks(profile, str(name), latency=axis)
    opt_in = {label for label, q in shown.items() if q.kind in EE_KINDS}

    def marked(label: str, e: Mapping[str, Any]) -> str:
        q = shown.get(label)
        if q is not None:
            return "≈" if q.approx else ("" if q.dominated_by is None else
                                         f"dominated by {opt_in_label(q.dominated_by, opt_in)}")
        return "≈" if e.get("approx") else ("" if e.get("frontier") else f"dominated by {e.get('dominated_by')}")

    for b in sorted(budgets, key=int):
        e = budgets[b]
        label = e.get("point") or f"{name}@{b}"
        if label in seen:
            continue
        seen.add(label)
        mark = marked(label, e)
        if not e.get("T_flat"):
            fitted.append((label, e.get("T")))
        row = f"{label:<24} {e['acc']:6.3f} {_ci(e.get('dacc'), pct=True):>28}"
        row += "".join(f" {cell:>24}" if st == axis else f" {cell:>8}"     # an older profile's absent statistics are
                       for st, cell in stat_cells(point_stats(profile, e, "b", b), int(e["n"]), axis).items())   # its rows'
        row += f" {_num(e.get('think_mean'), '.0f'):>6} {_num(e.get('forced'), '.0f', '%', 100):>7}"
        if verbose:                      # `_num`: a non-finite metric is null in the strict-JSON profile
            row += (f" {_num(e.get('nll'), '7.3f')} {_num(e.get('nll_T'), '7.3f')} {_num(e.get('ece_T'), '8.3f')} "
                    f"{_num(e.get('T'), '5.2f')}")
        lines.append(row + (f"  {mark}" if mark else ""))
        if int(e["n"]) not in tails:
            tails.append(int(e["n"]))
    # the deadline and early-exit points follow the budgets, marked on the one
    # frontier over every kind (the profile's stored marks compare the budgets among themselves)
    others = [("deadline", f"{name}@{d}ms", e, "d", d) for d, e in sorted((profile.get("deadlines") or {}).items(),
                                                                          key=lambda kv: int(kv[0]))]
    others += [("ee", f"{name}@ee{z.removeprefix('0')}", e, "e", z)
               for z, e in sorted((profile.get("ee") or {}).items(), key=lambda kv: float(kv[0]))]
    others += [("eeg", f"{name}@ee{z.removeprefix('0')}+g", e, "g", z)        # each behind early exit's gate
               for z, e in sorted((profile.get("ee_gated") or {}).items(), key=lambda kv: float(kv[0]))]
    for kind, label, e, block, key in others:
        if not e.get("T_flat"):
            fitted.append((label, e.get("T")))
        row = f"{label:<24} {e['acc']:6.3f} {_ci(e.get('dacc'), pct=True):>28}"
        row += "".join(f" {cell:>24}" if st == axis else f" {cell:>8}"
                       for st, cell in stat_cells(point_stats(profile, e, block, key), int(e["n"]), axis).items())
        row += f" {_num(e.get('think_mean'), '.0f'):>6} {_num(e.get('forced'), '.0f', '%', 100):>7}"   # a gated one's
        if verbose:
            row += (f" {_num(e.get('nll'), '7.3f')} {_num(e.get('nll_T'), '7.3f')} {_num(e.get('ece_T'), '8.3f')} "
                    f"{_num(e.get('T'), '5.2f')}")
        mark = marked(label, e)
        mark = "; ".join(x for x in (mark, EE_MARK if kind in EE_KINDS else "") if x)
        lines.append(row + (f"  {mark}" if mark else ""))
        if int(e["n"]) not in tails:
            tails.append(int(e["n"]))
    lines += [f"  deadline {d} ms: {why}" for d, why in sorted((profile.get("deadline_skipped") or {}).items(),
                                                               key=lambda kv: int(kv[0]))]
    if others:
        lines.append("  marks: one frontier over budgets, deadline points and early exit, as `callo compare` marks it "
                     "(the profile's stored marks cover the budgets alone); the pick never takes early exit (opt-in), "
                     "so it can be a row an early-exit point dominates")
    lines += [tails_line(t) for t in tails]
    n = data.get("questions")
    if isinstance(n, int) and n < SMALL_N and len(seen) > 1:
        lines.append(f"  {n} questions: below about {SMALL_N}, neighbouring frontier points will mostly be ≈; "
                     f"rerunning with another --seed estimates the trace variance")
    where = path if path is not None else Path(f"{name}.think.json")
    for d in sorted(profile.get("deadlines") or {}, key=int):                                 # the block
        m = mapping(profile, int(d), seed=int(comp.get("seed") or 0))
        lines += mapping_lines(str(name), m, serve_command(profile, None, where, deadline_ms=int(d),
                                                            cap=point_cap(profile, int(d))))
    audit = profile.get("audit") or {}
    per = audit.get("budgets") or {}
    detail = ", ".join(f"B={b}: {a.get('trace_identical')}/{a.get('n')} traces identical, latency error median "
                       f"{_num(a.get('lat_err_median'), '.1f', '%', 100)}"
                       for b, a in sorted(per.items(), key=lambda kv: int(kv[0])) if a.get("n"))
    lines.append(f"audit: latency model {audit.get('latency_model')}"
                 + (f", {audit.get('latency_reason')}" if audit.get("latency_reason") else "")
                 + (f" ({detail})" if detail else ""))
    reuse = {b: e.get("prefill_reused") for b, e in budgets.items() if int(b) > 0 and e.get("prefill_reused") is not None}
    remedy = False
    if reuse:                                                            # the host prompt cache
        shares = ", ".join(f"B={b}: {100 * reuse[b]:.0f}%" for b in sorted(reuse, key=int))
        line = f"generation prefill reused from the question's own cached state: {shares}"
        if audit.get("latency_reason") == "prefill_reused" or any(v > PREFILL_REUSED_MAX for v in reuse.values()):
            line += (" -- llama-server's host prompt cache flattered phase B's prefill; restart llama-server with "
                     "--cache-ram 0 for calibration, or accept the flattered prefill")
            remedy = True
        lines.append(line)
    if remedy or server.get("cache_ram") == 0:
        lines.append(SERVED_HIGHER)
    drift = profile.get("drift") or {}
    therm = drift.get("therm") or []
    readings = ", ".join(f"{t.get('at')} {t.get('limit') if t.get('limit') is not None else 'n/a'}" for t in therm
                         ) if any(t.get("limit") is not None for t in therm) else ""   # all null: "no thermal data"
    ratio = drift.get("ratio")
    lines.append(f"drift: {'FLAGGED' if drift.get('flagged') else 'none'}; end/phase-B ms per token "
                 f"{_num(ratio, '.2f') if ratio is not None else 'n/a'} over {drift.get('n') or 0} regenerations"
                 + (f"; thermal limit {readings}" if readings else "; no thermal data"))
    overhead = profile.get("calibration_overhead_s")
    if overhead:
        lines.append(f"calibration-only re-prefill subtracted: {overhead:.1f} s in total")
    lines += converge_lines(profile)
    for edge in (GRID[-1], GRID[0]):         # a T on the fit grid's edge: still overconfident at 16, sharper wanted at 0.25
        at = [label for label, t in fitted if t == edge]
        if at:
            lines.append(str(limit_note(edge, at=" at " + ", ".join(at))))
    pick = profile.get("pick") or {}
    note = pick_note(pick)
    lines.append(pick_line(pick))
    n_max = max((int(e["n"]) for e in budgets.values() if isinstance(e.get("n"), int)), default=0)
    lines += [f"  {line}" for line in limit_lines(_block_preference(pick), n_max)]   # what each limit rests on
    if note:
        lines.append(f"  {note}")
    if (unserved := serve_note(pick)) is not None:
        lines.append(f"  {unserved}")
    if (opt_in := opt_in_line(profile, str(name), latency=axis)) is not None:
        lines.append(f"  {opt_in}")
    if pick.get("label"):
        merged = pick.get("budgets") or []
        if len(merged) > 1:                                              # the largest truncates least
            lines.append(f"  budgets {min(merged)}-{max(merged)} read the same thing on every question; any serves, "
                         f"{max(merged)} truncates the fewest unseen questions")
        lines.append(serve_line(profile, path if path is not None else Path(f"{name}.think.json")))
    return lines


def _z(z: str) -> str:
    return z.removeprefix("0") if z.startswith("0.") else z


# converge.gain()'s two ways an on-front early-exit point gains, each against the model's budget and deadline points
EE_GAIN_WORDS = {"speed": "faster than every budget/deadline point as accurate",
                 "accuracy": "more accurate than every budget/deadline point as fast"}


def converge_lines(profile: Mapping[str, Any]) -> list[str]:
    """The early-exit block: the probe cost on this box (ms, decode steps incl. chunk overhead, restore
    share), one line per z (its cross-fitted net saving, accuracy and Δacc against the end answer, p50, and whether it
    gains on the model's front), and the break-even line with K and the overhead per think token; [] for a profile
    without the `converge` block (no --converge-probe)."""
    cv = profile.get("converge")
    if not isinstance(cv, dict):
        return []
    cost, k = cv.get("cost") or {}, cv.get("K")
    if cost.get("c_ms") is not None:
        restore = cost.get("restore_share")
        head = (f"early exit (K {k}, probe {cost['c_ms']:.0f} ms = {cost['c_steps']:.1f} decode steps incl. chunk "
                f"overhead" + (f", restore {100 * restore:.0f} %" if restore is not None else "") + "):")
    else:                                   # say which reason applies
        why = ("no cost pass (--converge-cost 0)" if (profile.get("comparable") or {}).get("converge_cost") == 0
               else "the cost pass read no probe" if not cost.get("n")
               else "no phase-B decode step to price the chunk overhead" if cv.get("decode_ms") is None
               else "no chunk followed a probe")
        head = f"early exit (K {k}, probe cost unmeasured: {why}):"
    # unmeasured, the rules are chosen at rho 0: the saving does not net out any probe cost
    saving = "net" if cost.get("c_ms") is not None else "gross (cost not measured)"
    out = [head]
    rules, ee, gains = cv.get("rules") or {}, profile.get("ee") or {}, cv.get("gain") or {}
    for z in sorted(rules, key=float):
        r = rules[z]
        if r is None:
            out.append(f"  z {_z(z)}: no rule reaches it (no cell passed the fixed sequence on {cv.get('n')} questions)")
            continue
        g = gains.get(z) or {}
        # each gain says what it beats (the budget and deadline points), not the Δ's full-thinking reference: an
        # "accuracy gain" beside a negative Δ read as a bug
        how = [EE_GAIN_WORDS[w] for w in ("speed", "accuracy")
               if (g.get("p50") or {}).get(w) or (g.get("p95") or {}).get(w)]
        verdict = (f"ON your front ({'; '.join(how)})" if g.get("gains") else
                   "not on your front" if g else "front not tested (probe cost unmeasured)")
        p50 = (ee.get(z) or {}).get("lat_p50_ms")
        out.append(f"  z {_z(z)}: saves {r['net_saving']:.0f} tokens/question {saving}, acc {r['acc']:.2f} "
                   f"(Δ vs full think {_ci(r.get('dacc'), pct=True)})" + (f", p50 {_s(p50[0])}" if p50 else "")
                   + f" -> {verdict}")
    out += gate_lines(profile)
    rho = cv.get("rho") or {}
    here = f"here {cost['c_steps']:.1f}" if cost.get("c_steps") is not None else "here unmeasured"
    # "pays while" only up to rho_prefix (every grid rho from the cheapest up gains); an older
    # profile without the field keeps the rho_max line
    prefix, top = rho.get("rho_prefix"), max(map(float, rho["by_rho"])) if rho.get("by_rho") else None
    if "rho_prefix" in rho and prefix is not None and prefix == top:
        out.append(f"  break-even at K {k}: pays at every probe cost tested, >= {top:g} decode steps ({here}; "
                   f">= {top / k:.2f} steps per think token)")
    elif "rho_prefix" in rho and prefix is not None:
        also = f"; some z also gains at {rho['rho_max']:g}" if (rho.get("rho_max") or 0) > prefix else ""
        out.append(f"  break-even at K {k}: pays while a probe costs <= {prefix:g} decode steps ({here}; "
                   f"{prefix / k:.2f} steps per think token){also}")
    elif "rho_prefix" in rho and rho.get("rho_max") is not None:
        at = ", ".join(f"{float(r):g}" for r in sorted(rho["by_rho"], key=float) if rho["by_rho"][r])
        out.append(f"  break-even at K {k}: gains only at {at} decode steps, not from the cheapest probe cost up "
                   f"({here})")
    elif rho.get("rho_max") is not None:
        out.append(f"  break-even at K {k}: pays while a probe costs <= {rho['rho_max']:g} decode steps ({here}; "
                   f"{rho['rho_max'] / k:.2f} steps per think token)")
    elif rho:
        out.append(f"  not on your front at any z for any probe cost down to {min(map(float, rho['by_rho'])):g} decode "
                   f"steps ({here})")
    return out


def gate_lines(profile: Mapping[str, Any]) -> list[str]:
    """Early exit's gate, when calibrate measured it: what it flagged and its read cost, then one line per gated point
    (its accuracy, Δacc against full thinking, p50), or why there is none; [] without the gate's reads."""
    gate = (profile.get("converge") or {}).get("gate")
    if not isinstance(gate, Mapping):
        return []
    rm = gate.get("read_ms") or {}
    sha = gate.get("sha256")
    out = [f"  gate (experimental, sha256 {sha[:12] if isinstance(sha, str) else 'mixed'}): flags "
           f"{_num(gate.get('flagged'), '.0f', ' %', 100)} of {gate.get('n')} questions (no early exit for them), read "
           f"p50 {_num(rm.get('p50'), '.0f', ' ms')}, p95 {_num(rm.get('p95'), '.0f', ' ms')}"
           + (f"; {gate['errors']} failed reads" if gate.get("errors") else "")]
    if gate.get("why"):
        out.append(f"  gate: {gate['why']}")
    for z, e in sorted((profile.get("ee_gated") or {}).items(), key=lambda kv: float(kv[0])):
        p50 = e.get("lat_p50_ms")
        out.append(f"  z {_z(z)}+g: acc {e['acc']:.2f} (Δ vs full think {_ci(e.get('dacc_cap'), pct=True)})"
                   + (f", p50 {_s(p50[0])}" if p50 else ""))
    return out


def stat_cells(stats: Mapping[str, Any], n: int, axis: str) -> dict[str, str]:
    """A point's five table cells: the pick statistic's with its interval, the others plain; when p99's
    nearest rank is n (below 100 questions) p99 is the maximum, so its cell reads `= max` rather than a second
    number."""
    out = {}
    for st, field in STAT_FIELDS.items():
        v = stats.get(field)
        if st == "max":
            out[st] = _s(v)
        elif st == "p99" and nearest_rank(n, QUANTILES["p99"]) == n:
            out[st] = "= max"
        else:
            out[st] = _ci(v) if st == axis else _s((v or [None])[0])
    return out


def tails_line(n: int) -> str:
    """What the tail statistics rest on, `tails (n 376): p95 from 19 questions, p99 from 4, max from 1`."""
    t = tail_counts(n)
    p99 = "p99 = max" if nearest_rank(n, QUANTILES["p99"]) == n else f"p99 from {t['p99']}"
    return f"tails (n {n}): p95 from {t['p95']} question{'s' if t['p95'] != 1 else ''}, {p99}, max from 1"


def _pp(v: Sequence[float]) -> str:
    p, lo, hi = v
    return f"{p:+.2f}pp [{lo:+.2f}, {hi:+.2f}]"


def pick_line(pick: Mapping[str, Any], *, bold: bool = False) -> str:
    """The pick line from a `pick` block, old or new: `pick: tiel@512 (fastest within noise of the best score,
    --price 5pp/s on mean); best tiel@1024 84.6 pp`, then the most accurate point when it is not M, the pick's Δacc
    (and at R > 0 its Δscore) against M; without a pick, `pick: none (<reason>)`. The preference is named when it is
    not the default, in canonical, pasteable form. `bold` is report.md's markdown. A block that picked an early-exit
    point (written while the pick could take one) says it is no longer picked, as its `serve with: none` line does."""
    label = pick.get("label")
    star = "**" if bold else ""
    if not label:
        if pick.get("reason"):
            return f"{star}pick: none{star} ({pick['reason']})"
        return f"{star}pick: none{star} ({pick.get('rule')}); best {pick.get('best') or 'none'}"
    if pick.get("kind") == "ee":            # a block written while early exit could be picked: see `serve_line`
        return f"{star}pick: {label}{star} (early exit, no longer picked: re-pick with `callo compare`)"
    text = _block_preference(pick).text()
    best, score = pick.get("best"), pick.get("score") or {}
    line = f"{star}pick: {label}{star} ({pick.get('rule')}{', ' + text if text else ''}); best {best}"
    if isinstance(score.get("best"), (int, float)):
        line += f" {score['best']:.1f} pp"
    if pick.get("most_accurate") and pick["most_accurate"] != best:
        line += f"; most accurate {pick['most_accurate']}"
    if pick.get("dacc_best") and label != best:
        line += f"; Δacc vs it {_ci(pick['dacc_best'], pct=True)}"
    if pick.get("dscore_best") and label != best:
        line += f", Δscore {_pp(pick['dscore_best'])}"
    return line


def _block_preference(pick: Mapping[str, Any]) -> Preference:
    """The preference a `pick` block recorded (an old block: its own latency, max_latency_s, min_accuracy); the default
    for a hand-edited block the validator refuses, so a summary line never fails on it."""
    try:
        return Preference.from_dict(pick.get("preference"), legacy=pick)
    except (InvalidInput, KeyError, TypeError, ValueError):
        return Preference()


def limit_lines(pref: Preference, n: int | Sequence[int]) -> list[str]:
    """What each active limit rests on (`--limit p99=6s: p99 from 4 of 376`): a limit is a deterministic
    filter on a point estimate and needs no interval, but a tail limit judges very few questions. `n` is one question
    count (calibrate: one profile) or every profile's (compare); each distinct count is listed, largest first, so a
    limit over profiles of different sizes says what it rests on in each (`p99 from 4 of 376, 2 of 120`)."""
    ns = sorted({k for k in ([n] if isinstance(n, int) else n) if k > 0}, reverse=True)
    if not ns:
        return []

    def at(stat: str, k: int) -> str:
        return f"{k if stat == 'mean' else tail_counts(k)[stat]} of {k}"
    return [f"{lim.canonical}: {lim.stat} from {', '.join(at(lim.stat, k) for k in ns)}" for lim in pref.limits]


def serve_line(profile: Mapping[str, Any], where: Path | str) -> str:
    """`serve with:`: `serve_command` for the profile's own pick (a deadline pick serves `--think-deadline D`). A
    pick block that names an early-exit point (written while the pick could take one) serves nothing: early exit is
    opt-in, so the line never carries `--think-converge`, and never falls through to `--think-budget None`."""
    pick = profile.get("pick") or {}
    if pick.get("kind") == "ee":
        return ("serve with: none (this profile's pick is an early-exit point; early exit is opt-in and never picked "
                "now: re-pick with `callo compare`)")
    if pick.get("kind") == "deadline":
        return "serve with: " + serve_command(profile, None, where, deadline_ms=pick.get("deadline_ms"),
                                              cap=pick.get("cap"))
    return "serve with: " + serve_command(profile, pick.get("budget"), where)


def serve_command(profile: Mapping[str, Any], budget: Any, where: Path | str, *, deadline_ms: int | None = None,
                  cap: int | None = None) -> str:
    """`callo serve ...` for `budget` of the profile at `where`, runnable as printed (the one source of truth for
    calibrate's summary and compare's report): `--model FILE` when callo launched the measured server (its args are
    recorded), else `--upstream URL` (an attach mode: the llama-server's URL, which the profile does not record);
    `--think-max` when the budget is above serve's default (serve refuses a `--think-budget` above it); `--no-cache` when
    the run measured without the cache (budget 0's latency depends on it; on is serve's default since 0.3.0). A
    menu or `--model` launch records more: the GGUF's absolute path (`environment.model.path`, used instead of the
    file name) and the launch shape
    (`environment.server.launch`), repeated as `--ctx`, `--parallel`, `--no-kvu`, `--s1-slots`, `--user-delimiter=S`,
    `--mmproj`, `--position-bias=P` (the budget-0 readout was debiased by it), `--llama-server=BIN` when it is not the
    default `llama-server` (another build, another latency) and, after `--`, the entry's llama-server args, so the
    served readout and latency are the measured ones. Every value is shell-quoted. A deadline pick (`deadline_ms`)
    serves `--think-deadline D` instead of `--think-budget`, with `--think-max` raised to the cap D was
    calibrated under when that is above serve's default (a deadline's cap is never above `--think-max`). Never
    `--think-converge`: early exit is opt-in, never picked."""
    from .cli import DEFAULT_THINK_MAX  # cli imports this module
    from .menu import DEFAULT_LLAMA_SERVER  # menu imports this module

    env = profile.get("environment") or {}
    model = env.get("model") if isinstance(env.get("model"), dict) else {}
    server = env.get("server") if isinstance(env.get("server"), dict) else {}
    launch = server.get("launch") if isinstance(server.get("launch"), dict) else None
    q = shlex.quote
    if server.get("server_args") is not None:
        path = model.get("path") if isinstance(model.get("path"), str) and model.get("path") else None
        source = f"--model {q(str(path or model.get('file') or 'MODEL.gguf'))}"
    else:
        source = "--upstream URL"
    shape, tail = "", ""
    if launch is not None:
        if _whole(launch.get("ctx")):
            shape += f" --ctx {launch['ctx']}"
        if _whole(launch.get("parallel")):
            shape += f" --parallel {launch['parallel']}"
        if launch.get("no_kvu") is True:
            shape += " --no-kvu"
        if _whole(launch.get("s1_slots")):
            shape += f" --s1-slots {launch['s1_slots']}"
        if isinstance(launch.get("user_delimiter"), str) and launch["user_delimiter"]:
            shape += " " + q(f"--user-delimiter={launch['user_delimiter']}")
        if isinstance(model.get("mmproj_path"), str) and model["mmproj_path"]:
            shape += f" --mmproj {q(model['mmproj_path'])}"
        if isinstance(launch.get("position_bias"), str) and launch["position_bias"]:
            shape += " " + q(f"--position-bias={launch['position_bias']}")
        llama = launch.get("llama_server")
        if isinstance(llama, str) and llama and llama != DEFAULT_LLAMA_SERVER:
            shape += " " + q(f"--llama-server={llama}")
        args = launch.get("args")
        if isinstance(args, list) and args and all(isinstance(a, str) for a in args):
            tail = " -- " + " ".join(q(a) for a in args)
    extra = ""
    top = budget if budget is not None else cap if deadline_ms is not None else None   # the cap a request runs under
    if isinstance(top, int) and top > DEFAULT_THINK_MAX:
        extra += f" --think-max {top}"
    if (profile.get("comparable") or {}).get("cache") is False:
        extra += " --no-cache"
    think = f"--think-budget {budget}" if deadline_ms is None else \
        (f"--think-budget {budget} " if budget else "") + f"--think-deadline {deadline_ms}"   # both limits
    return f"callo serve {source}{shape} --think-profile {q(str(where))} {think}{extra}{tail}"


def _whole(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def dry_run_lines(settings: ThinkSettings, items: Sequence[Item], skipped: Mapping[str, int]) -> list[str]:
    """For one server: the questions after filtering, the budgets and the worst case; no requests are made.
    The time estimate from an earlier profile's rates (`--rates-from`) comes with the menu run."""
    skip = ", ".join(f"{k} {v}" for k, v in sorted(skipped.items()) if k != "records") or "nothing"
    positive = list(settings.positive)
    worst = len(items) * settings.bmax
    audit = settings.audit * sum(positive[:-1])
    warm = min(settings.warmup, len(items)) * settings.bmax if positive else 0     # the warm-up generates at Bmax too
    return [f"dry run: {len(items)} labelled text questions from {skipped.get('records', 0)} records in "
            f"{settings.data_name} (skipped: {skip})",
            f"budgets: {list(settings.budgets)}; warm-up {settings.warmup}, audit {settings.audit} per cut budget, "
            f"seed {settings.seed}, sampling {settings.sampling}, --cache {'on' if settings.cache else 'off'}",
            f"worst case: {len(items)} x {settings.bmax} = {worst} generated tokens, plus up to {audit} for the audit, "
            f"{DRIFT_N * settings.bmax} for the drift check and {warm} for the warm-up",
            *([f"probe-cost pass: up to {settings.converge_cost} x {settings.bmax} = "
               f"{settings.converge_cost * settings.bmax} generated tokens more"]
              if settings.converge_probe and settings.converge_cost > 0 and positive else []),
            *([f"early-exit gate: {gates} budget-0 reads, one per question with two or more options"]
              if (gates := sum(len(it.question.keys) >= 2 for it in items)) and settings.converge_probe and positive
              else []),
            "time: unknown until measured (the live ETA prints after 5 questions)"]


SHARED_WITH_BIAS = (("--resume", "resume"), ("--seed", "seed"), ("--allow-busy", "allow_busy"))   # cli names them
THINK_ONLY = (("--upstream", "upstream"), ("--warmup", "warmup"), ("--audit", "audit"), ("--cache", "cache"),
              ("--think-sampling", "think_sampling"), ("--latency", "latency"), ("--limit", "limit"),
              ("--price", "price"), ("--max-latency", "max_latency"), ("--min-accuracy", "min_accuracy"),
              ("--verbose", "verbose"), ("--dry-run", "dry_run"), ("--deadlines", "deadlines"),
              ("--think-deadline-reserve", "think_deadline_reserve"), ("--converge-probe", "converge_probe"),
              ("--converge-cost", "converge_cost"))
BIAS_ONLY_IN_THINK = (("--raw", "raw"), ("--min-questions", "min_questions"), ("--max-per-n", "max_per_n"))


TRI_STATE = {"cache": "--no-cache"}  # cli.OnOff with a None default: False is the given `--no-...` form


def given(args: argparse.Namespace, flags: Sequence[tuple[str, str]]) -> list[str]:
    """The flags of `flags` the user gave (`is not`, never `not in (None, False)`: `--seed 0` equals False). A
    `TRI_STATE` flag (`--cache` / `--no-cache`, default None) is given when not None, named as typed."""
    out = []
    for flag, attr in flags:
        v = getattr(args, attr, None)
        if attr in TRI_STATE and v is False:
            out.append(TRI_STATE[attr])
        elif v is not None and v is not False:
            out.append(flag)
    return out


def url_given(args: argparse.Namespace) -> bool:
    """`--url` was given (then it is a source, exclusive with `--upstream`, `--model` and `--menu`): recorded by
    the parser's `_Given` action even when the value equals the default, else (a Namespace built elsewhere) a value
    other than the default."""
    from .cli import DEFAULT_PORT  # cli imports this module

    url = getattr(args, "url", None)
    return bool(getattr(args, "url_given", False)) or (
        isinstance(url, str) and url.rstrip("/") != f"http://127.0.0.1:{DEFAULT_PORT}")


def think_usage_error(args: argparse.Namespace) -> str | None:
    """The usage errors every `--think-budgets` form shares (exit 2), checked first: `--bias` is exclusive, `--bias`'s
    own flags are refused, `--data` is required. The message without the `callo: ` prefix, or None."""
    if args.bias:
        return "--bias and --think-budgets are mutually exclusive"
    if bias_only := given(args, BIAS_ONLY_IN_THINK):
        return f"{', '.join(bias_only)} apply only with --bias"
    if args.data is None:
        return ("--think-budgets needs --data (labelled JSONL): the accuracy at each budget is measured against the "
                "labels")
    return None


HEALTH_PRECHECK_S = 5.0                  # `--upstream`: one GET /health before the in-process serve's 600 s wait


def precheck_upstream(upstream: str, *, timeout: float = HEALTH_PRECHECK_S) -> None:
    """`--upstream URL`: a URL `callo serve --upstream` would refuse is refused first with serve's own
    reason (`cli.upstream_error`: https, a path, credentials, no `http://`), then nothing
    listening (connection refused) or an unresolvable host is an immediate Refusal, instead of `wait_health`'s silent
    600 s. Anything that answers at all (200, a 503 while loading, a timeout) proceeds to the in-process serve, which
    waits for the model as before."""
    from . import cli  # cli imports this module

    if (why := cli.upstream_error(upstream)) is not None:
        raise Refusal(f"refusing to calibrate: {why}")
    try:
        urllib.request.urlopen(f"{upstream.rstrip('/')}/health", timeout=timeout).close()  # noqa: S310 -- local server
    except urllib.error.HTTPError:
        return                                                           # 503 "loading model": wait_health's case
    except urllib.error.URLError as e:
        if isinstance(e.reason, (ConnectionRefusedError, socket.gaierror)):
            raise Refusal(f"refusing to calibrate: nothing answers at {upstream} ({e.reason}); start llama-server "
                          f"first, or pass the URL of a running `callo serve` with --url") from None
    except ValueError as e:                                              # http.client.InvalidURL past urlsplit
        raise Refusal(f"refusing to calibrate: --upstream {upstream!r} is not a valid URL ({e}); expected "
                      f"http://host:port") from None
    except OSError:
        return                                                           # a slow answer: not a dead upstream


@contextlib.contextmanager
def served_in_process(upstream: str, *, cache: bool, log: Callable[[str], None]) -> Iterator[str]:
    """`--upstream URL`: a `callo serve --upstream URL --port 0 --think-budget 0` with the run's `--cache` or
    `--no-cache` (always explicit) in this process, for the run's duration. Its refusals (no thinking mode is NOT
    one: that is measured at budget 0) surface as a Refusal."""
    from . import cli  # cli imports this module

    precheck_upstream(upstream)
    argv = ["serve", "--upstream", upstream, "--port", "0", "--think-budget", "0", "--cache" if cache else "--no-cache"]
    args_list, extra = cli.split_passthrough(argv)
    args = cli.build_parser().parse_args(args_list)
    args.extra = extra
    stop, ready = threading.Event(), threading.Event()
    port: list[int] = []
    rc: list[int] = []

    def run() -> None:
        try:
            rc.append(cli.serve(args, ready=lambda p: (port.append(p), ready.set()), stop=stop))
        finally:
            ready.set()

    thread = threading.Thread(target=run, name="callo-calibrate-serve", daemon=True)
    thread.start()
    try:
        ready.wait(timeout=SERVE_START_TIMEOUT_S)
        if not port:
            raise Refusal(f"refusing to calibrate: the in-process `callo serve --upstream {upstream}` did not start"
                          + (f" (exit {rc[0]}; its log says why)" if rc else " within the timeout"))
        log(f"callo serve attached to {upstream} on port {port[0]} (in-process, --think-budget 0)")
        yield f"http://127.0.0.1:{port[0]}"
    finally:
        stop.set()
        thread.join(timeout=60)


def settings_from_args(args: argparse.Namespace, *, warn: Callable[[str], None] | None = None) -> ThinkSettings:
    """The run's settings, validated (InvalidInput: exit 2). `warn` receives the one-line warnings a legal but
    odd setting earns (`--audit 1` or `2` can never verify the latency model)."""
    budgets = parse_budgets(args.think_budgets)
    warmup = DEFAULT_WARMUP if args.warmup is None else args.warmup
    audit = DEFAULT_AUDIT if args.audit is None else args.audit
    seed = DEFAULT_SEED if args.seed is None else args.seed
    if warmup < 0 or audit < 0:
        raise InvalidInput(f"--warmup and --audit must be non-negative, got {warmup} and {audit}")
    if 0 < audit < AUDIT_MIN_CUTS and warn is not None:
        warn(f"WARNING: --audit {audit} cannot verify the latency model: a verdict needs at least {AUDIT_MIN_CUTS} "
             f"audited cut questions per budget, so the audit will read `unverified`; use --audit {AUDIT_MIN_CUTS} or "
             f"more, or --audit 0 to skip it")
    if seed < 0:
        raise InvalidInput(f"--seed must be at least 0, got {seed}")
    pref = preference_from_args(args)                                     # the validator compare shares
    sampling = args.think_sampling or "model"
    parse_sampling(sampling)                                               # its own InvalidInput names the fault
    deadlines = parse_deadlines(getattr(args, "deadlines", None))
    reserve = getattr(args, "think_deadline_reserve", None)
    if reserve is not None and not deadlines:
        raise InvalidInput("--think-deadline-reserve fixes the deadline points' readout reserve: it needs --deadlines")
    if reserve is not None and not 0 <= reserve <= MAX_DEADLINE_MS:
        raise InvalidInput(f"--think-deadline-reserve must be from 0 to {MAX_DEADLINE_MS} ms, got {reserve}")
    probe_k, cost_n = getattr(args, "converge_probe", None), getattr(args, "converge_cost", None)
    if probe_k is not None and probe_k not in CONVERGE_KS:
        raise InvalidInput(f"--converge-probe must be one of {', '.join(map(str, CONVERGE_KS))}, got {probe_k}")
    if cost_n is not None and probe_k is None:
        raise InvalidInput("--converge-cost sizes --converge-probe's cost pass: it needs --converge-probe")
    if cost_n is not None and cost_n < 0:
        raise InvalidInput(f"--converge-cost must be at least 0, got {cost_n}")
    data = Path(args.data)
    try:
        sha = data_sha256(data)
    except OSError as e:
        raise InvalidInput(f"--data {args.data}: {e}") from None
    return ThinkSettings(str(data), data.name, sha, tuple(budgets), seed=seed, warmup=warmup, audit=audit,
                         cache=getattr(args, "cache", None) is not False,   # None (no flag): on, the default
                         sampling=sampling, latency=pref.latency,
                         max_latency=args.max_latency, min_accuracy=args.min_accuracy, verbose=bool(args.verbose),
                         allow_busy=bool(args.allow_busy),
                         limits=tuple(lim for lim in pref.limits if lim.source == "--limit"), price=pref.price,
                         deadlines=deadlines, reserve_ms=reserve, converge_probe=probe_k,
                         converge_cost=DEFAULT_CONVERGE_COST if cost_n is None else cost_n)


def run(args: argparse.Namespace, *, out: IO[str] | None = None, err: IO[str] | None = None) -> int:
    """`callo calibrate --think-budgets`: 0 when the profile is written, 1 for a failed request, bad data, no
    labelled text questions or an `--out` that cannot be written, 2 for a usage error or a refusal, 130 on Ctrl-C (the
    raw file keeps what was read; `--resume` continues). The summary goes to stdout and the progress log to stderr.
    A console that went away (BrokenPipeError) exits 1 without a traceback, as `--bias` does."""
    out, err = out or sys.stdout, err or sys.stderr
    try:
        return _cli(args, out, err)
    except BrokenPipeError:
        return silence_stdout(out)


def _cli(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    def log(line: str) -> None:
        print(line, file=err, flush=True)

    if args.bias or given(args, BIAS_ONLY_IN_THINK):
        print(f"callo: {think_usage_error(args)}", file=err)
        return 2
    if args.upstream is not None and url_given(args):                   # given explicitly, even as the default
        print("callo: --url and --upstream are mutually exclusive: choose one (--url measures a running `callo serve`, "
              "--upstream a bare llama-server through an in-process one)", file=err)
        return 2
    if (usage := think_usage_error(args)) is not None:                  # --data, after the source check as before
        print(f"callo: {usage}", file=err)
        return 2
    try:
        settings = settings_from_args(args, warn=log)
    except InvalidInput as e:
        print(f"callo: {e}", file=err)
        return 2
    out_dir = Path(args.out) if args.out else Path(DEFAULT_OUT_DIR)
    if args.dry_run:
        try:
            items, skipped = load_data(settings.data)
        except InvalidInput as e:
            print(f"callo: {e}", file=err)
            return 1
        if not items:
            print(f"callo: --data {settings.data_name}: no labelled text questions (skipped: "
                  f"{', '.join(f'{k} {v}' for k, v in sorted(skipped.items())) or 'nothing'})", file=err)
            return 1
        for line in dry_run_lines(settings, items, skipped):
            print(line, file=out)
        print(f"out: {out_dir}/<name>.think.json, <name>.raw.jsonl, report.md and report.json; no requests were made",
              file=out)
        return 0
    try:
        with contextlib.ExitStack() as stack:
            # <name> comes from the server whenever it answers: one GET /callo/stats, no readout
            # (--out may hold several models' raw files). Only a server that does not answer falls back
            # to --out (`offline_name`): a complete raw file refits with no server at all
            name: str | None = None
            if args.upstream is not None:                   # `--upstream ""` too: upstream_error names the fault
                url = args.upstream.rstrip("/")
                if args.resume:
                    try:
                        precheck_upstream(args.upstream)
                    except Refusal:
                        from .cli import upstream_error  # cli imports this module
                        if upstream_error(args.upstream) is not None:
                            raise                           # not a URL serve takes: refused as before
                        name = offline_name(out_dir, settings, url, "nothing answers")
                        log(f"callo: nothing answers at {url}; --resume: refitting {name}.raw.jsonl from {out_dir} "
                            f"with no server")
                if name is None:
                    url = stack.enter_context(served_in_process(args.upstream, cache=settings.cache, log=log))
            else:
                url = args.url.rstrip("/")
            if name is None:
                try:
                    stats = read_stats(url)
                    name = profile_name(stats)
                except (OSError, ValueError) as e:
                    if not args.resume or args.upstream is not None:
                        print(f"callo: refusing to calibrate: cannot read {url}/callo/stats ({e})", file=err)
                        return 2
                    name = offline_name(out_dir, settings, url, str(e))
                    log(f"callo: {url}/callo/stats does not answer ({e}); --resume: refitting {name}.raw.jsonl from "
                        f"{out_dir} with no server")
            outcome = calibrate_one(settings, url=url, name=name, out_dir=out_dir, resume=bool(args.resume), log=log)
    except Refusal as e:
        print(f"callo: {e}", file=err)
        return 2
    except KeyboardInterrupt:
        print(f"callo: interrupted; {out_dir} keeps every readout so far, `--resume` continues the run", file=err)
        return 130
    if outcome.status == "refused":
        print(f"callo: {outcome.reason}", file=err)
        return 2
    if outcome.status != "done" or outcome.profile is None:
        print(f"callo: {outcome.reason}", file=err)
        return 1
    profile = json.loads(outcome.profile.read_text())
    for line in summary_lines(profile, verbose=settings.verbose, path=outcome.profile):
        print(line, file=out)
    if (out_dir / "run.json").exists():          # a menu's DIR: its joint report is not replaced by a single profile's
        log(f"report: {out_dir} holds a menu run; its report.md is kept (callo compare {out_dir}/*.think.json for a "
            f"joint view)")
        return 0
    from . import compare  # compare imports this module

    try:
        md, js = compare.write_report([profile], out_dir, pref=settings.preference, verbose=settings.verbose,
                                      sources=[str(outcome.profile)])      # a single profile is a valid input
    except OSError as e:
        print(f"callo: {out_dir}: cannot write report.md / report.json ({e.strerror or e})", file=err)
        return 1
    log(f"report: {md} and {js}")
    return 0
