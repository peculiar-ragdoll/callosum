"""`callo calibrate --bias`: measure the position prior against a running `callo serve`.

`collect` is the client. Every (question, rotation) is one single-question POST /v1/systemone on the
`callo-calibrate` lane (X-Callo-Conversation), so one slot serves the whole run; the run aborts (exit 2) if that slot
ever changes (a lane rebind, an S1 fallback). Cells go in a seeded shuffled order, so a question's rotations are
spread over the run. Every readout is appended to --raw as it arrives and flushed, so --resume continues an
interrupted run and `bench/bias_refit.py` can refit offline. The server-state guard (`guard.py`) reads /callo/stats
before the first readout, every 50 readouts and after the last. `build_profile` and `run` (below) fit and write.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import json
import math
import os
import random
import string
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from . import __version__, guard
from .bias import (
    BUILTIN_SET,
    BuiltinSetError,
    Item,
    Readings,
    SetError,
    builtin_set_bytes,
    builtin_set_sha256,
    cap_per_n,
    evaluate,
    fit_n,
    offsets_from_prior,
    read_set,
)
from .engine import READOUT_KINDS
from .errors import CalloError, InvalidInput
from .profile import ESTIMATOR, FORMAT, resume_provenance
from .significance import DRAWS, bootstrap_intervals, decide
from .think import PROFILE_T_FIX, think_default_refusal

RAW_FORMAT = 1
HEADER_KEYS = ("model_key", "provenance", "server", "set", "seed", "max_per_n")   # what --resume must match
CHECK_FIELDS = (("index", int), ("session", int), ("readouts", int), ("failed", list))  # what a check row needs
READOUT_FIELDS = (("q", str), ("n", int), ("s", int), ("probs", list), ("miss", int), ("slot", int))  # a readout's
RETRIES = 3
BACKOFF = (1.0, 4.0, 16.0)
CHECK_EVERY = 50
ETA_AFTER = 20
DEFAULT_MIN_QUESTIONS, DEFAULT_MAX_PER_N, DEFAULT_SEED = 24, 48, 0     # 48: the built-in diagnostic's cap per n
NO_CAP = 1_000_000                  # --max-per-n with --data: every eligible question of the user's file
DIAGNOSTIC_RAW = "callo-bias-diagnostic.raw.jsonl"   # the diagnostic's default --raw: never a --data run's default
OUT_NEEDS_DATA = ("--out needs --data with your own questions: a position-bias profile is fitted on them, "
                  "and the built-in set, under any file name, is a diagnostic that writes no profile; omit --out to "
                  "run the built-in diagnostic")
_DOWN = (OSError, http.client.HTTPException)        # URLError, refused, reset, timeout, a truncated body


class Refusal(CalloError):
    """calibrate refuses to start or to continue: exit 2."""


class RunFailed(CalloError):
    """A request failed for good, or a reply was unusable: exit 1. --raw keeps what was read."""


def _http(method: str, url: str, body: dict | None = None, headers: dict | None = None,
          timeout: float = 900.0) -> tuple[int, dict[str, str], Any]:
    """(status, lower-cased reply headers, parsed JSON or text). HTTP errors are returned; network errors raise."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 -- callo's own URL
            status, hdrs, raw = r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        status, hdrs, raw = e.code, e.headers, e.read()
    try:
        parsed: Any = json.loads(raw)
    except ValueError:
        parsed = raw.decode(errors="replace")
    return status, {k.lower(): v for k, v in (hdrs or {}).items()}, parsed


def reply_reason(reply: Any) -> str:
    """A non-200 reply's reason: callo's `detail` when it gives one (a 503's reason, the think probe's 422 naming the
    pin it refuses), else the body, cut to 300 characters."""
    if isinstance(reply, dict) and isinstance(reply.get("detail"), str):
        return reply["detail"][:600]
    return str(reply)[:300]


def read_stats(base: str) -> dict:
    status, _, body = _http("GET", f"{base}/callo/stats", timeout=60)
    if status != 200 or not isinstance(body, dict):
        raise OSError(f"GET {base}/callo/stats returned {status}")
    return body


def _retrying(request: Callable[[], tuple[int, dict[str, str], Any]], *, failing: str, sleep: Callable[[float], None],
              statuses: Collection[int] = (503,)) -> tuple[int, dict[str, str], Any, int]:
    """The one retry policy: a 503 (any of `statuses`: --think-budgets adds 504) or a
    connection error is retried 3 times with backoff 1, 4, 16 s; any other status is returned as (status, headers,
    body, the connection errors that were retried). Still failing: RunFailed("<failing>: still failing after 3
    retries (<the last error>)")."""
    last, dropped = "", 0
    for attempt in range(RETRIES + 1):
        try:
            status, headers, body = request()
        except _DOWN as e:
            last = f"{type(e).__name__}: {e}"
            dropped += 1                    # the last attempt raises below, so every drop is followed by a retry
        else:
            if status not in statuses:
                return status, headers, body, dropped
            last = f"HTTP {status}: {reply_reason(body)}"      # callo's detail, not the dict
        if attempt < RETRIES:
            sleep(BACKOFF[attempt])
    raise RunFailed(f"{failing}: still failing after {RETRIES} retries ({last})")


def read_stats_mid_run(base: str, *, sleep: Callable[[float], None] = time.sleep) -> dict:
    """/callo/stats during a run: a 503 or a connection error is retried like a readout (3 times, backoff 1, 4, 16 s),
    so one transient failure does not end hours of reading. Anything else, or still failing, is RunFailed."""
    status, _, body, _ = _retrying(lambda: _http("GET", f"{base}/callo/stats", timeout=60),
                                   failing=f"cannot read {base}/callo/stats mid-run", sleep=sleep)
    if status == 200 and isinstance(body, dict):
        return body
    raise RunFailed(f"cannot read {base}/callo/stats mid-run (HTTP {status}: {str(body)[:300]})")


def check_preconditions(stats: dict, base: str) -> None:
    """In order: a callo of 0.3 or later, callo's engine on, no thinking by default, T = 1, no
    profile loaded. The guard comes next."""
    if "model_key" not in stats:
        raise Refusal(f"refusing to calibrate: {base}/callo/stats has no model_key, so this callo is older than v0.3; "
                      f"--bias needs v0.3 on the server side too. Upgrade callo and restart `callo serve`")
    if not stats["model_key"]:
        why = stats.get("model_key_error") or "no reason given"
        raise Refusal(f"refusing to calibrate: callo serve could not compute the served model's model_key ({why}), "
                      f"and a profile is keyed by it. See callo serve's startup warning, fix it, and restart")
    t = stats.get("temperature")
    if t is None:
        raise Refusal("refusing to calibrate: callo's engine is off: the upstream is a native decision model; the "
                      "position-bias profile applies to callo's zero-training readout")
    refusal = think_default_refusal(stats)           # before T (as calibrate): one source
    if refusal is not None:
        raise Refusal(f"refusing to calibrate: {refusal}")
    if t != 1.0:
        fix = PROFILE_T_FIX if stats.get("temperature_source") == "profile" else \
            "restart `callo serve` without --temperature"
        raise Refusal(f"refusing to calibrate: the server's S1 temperature is {t!r}, not 1.0. The prior is measured on "
                      f"readouts served at T = 1: {fix}")
    if stats.get("position_bias") is not None:
        raise Refusal("refusing to calibrate: the server already applies a position-bias profile, and the prior must be "
                      "measured on raw readouts. Restart `callo serve` without --position-bias")


def parse_slot(value: str | None, s1_slots: int) -> int | None:
    """X-Callo-Slot as an int. With no reserved slots (-np 1) callo answers "-" and the only slot is 0."""
    if value is None or value == "-":
        return 0 if s1_slots == 0 and value == "-" else None
    try:
        return int(value)
    except ValueError:
        return None


def post_readout(base: str, body: dict, *, what: str,
                 sleep: Callable[[float], None] = time.sleep) -> tuple[int, dict[str, str], Any, int]:
    """One readout. A 503 or a connection error is retried 3 times with backoff 1, 4, 16 s; any other
    status is returned for the caller to judge, with the number of retries that followed a connection error: callo
    may have answered (and counted) those requests before the connection dropped."""
    return _retrying(lambda: _http("POST", f"{base}/v1/systemone", body, {"X-Callo-Conversation": guard.LANE}),
                     failing=what, sleep=sleep)


def _parse_reply(it: Item, resp: Any, headers: dict[str, str], what: str) -> tuple[tuple[float, ...], int, dict]:
    keys = string.ascii_lowercase[: it.n]
    try:
        probs_by_key = resp["answers"][it.qid]["probabilities"]
        probs = tuple(float(probs_by_key[k]) for k in keys)
        entry = next(e for e in json.loads(headers["x-callo-timings"]) if e.get("q") == it.qid)
        miss = entry["miss"]
    except (KeyError, TypeError, ValueError, OverflowError, StopIteration) as e:   # OverflowError: float(10**400)
        raise RunFailed(f"{what}: a 200 reply with no usable answer ({type(e).__name__}: {e})") from None
    if (not isinstance(miss, int) or isinstance(miss, bool)
            or not all(math.isfinite(p) and p >= 0 for p in probs) or not any(probs)):
        raise RunFailed(f"{what}: a 200 reply with no usable answer (miss {miss!r}, probabilities {probs})")
    return probs, miss, entry


@dataclass
class Raw:
    """A --raw file as `read_raw` parses it."""
    header: dict
    readings: dict[tuple[str, int], tuple[tuple[float, ...], int]]
    checks: list[dict]          # the check records, without their "type"
    slot: int | None            # the slot of the last readout; None when there is none
    partial: bool               # the file ends in an incomplete line (a hard kill mid-write, or a run still writing)


def read_raw(path: str | Path) -> Raw:
    """Parse a `callo calibrate --bias` raw file: the one reader of the format (`--resume` and the bench's offline
    refit). Reads only: an incomplete last line is left out and reported as `partial`. Raises InvalidInput for a file
    that is not UTF-8, has no raw header or has a line that is not a well-formed readout or check record."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise InvalidInput(f"{path} is not a raw file (not UTF-8: {e})") from None
    end = text.rfind("\n") + 1
    rows = text[:end].split("\n")         # "\n" only: splitlines() would also cut at a raw U+2028 in a string
    try:
        head = json.loads(rows[0]) if rows else None
    except ValueError:
        head = None
    if not isinstance(head, dict) or head.get("callo_bias_raw") != RAW_FORMAT:
        raise InvalidInput(f"{path} has no callo --bias raw header")
    raw = Raw(head, {}, [], None, end != len(text))
    for i, row in enumerate(rows[1:], 2):
        if not row.strip():
            continue
        try:
            d = json.loads(row)
        except ValueError:
            raise InvalidInput(f"{path} line {i} is not JSON") from None
        try:
            if d.get("type") == "readout":
                bad = [k for k, kind in READOUT_FIELDS if type(d.get(k)) is not kind]   # `is`: a bool is no int
                if bad:
                    raise TypeError(f"a readout record without a valid {', '.join(bad)}")
                if len(d["probs"]) != d["n"] or not all(type(p) in (int, float) and math.isfinite(p) and p >= 0
                                                        for p in d["probs"]):
                    raise TypeError(f"a readout record whose probs are not {d['n']} finite non-negative numbers")
                if not any(d["probs"]):
                    raise TypeError("a readout record whose probs are all zero")
                raw.readings[(d["q"], d["s"])] = (tuple(d["probs"]), d["miss"])
                raw.slot = d["slot"]
            elif d.get("type") == "check":
                bad = [k for k, kind in CHECK_FIELDS if type(d.get(k)) is not kind]     # `is`: a bool is no int
                if bad:
                    raise TypeError(f"a check record without a valid {', '.join(bad)}")
                raw.checks.append({k: v for k, v in d.items() if k != "type"})
        except (KeyError, TypeError, AttributeError, OverflowError) as e:    # OverflowError: isfinite(10**400)
            raise InvalidInput(f"{path} line {i} is not a raw record ({type(e).__name__}: {e})") from None
    return raw


@dataclass
class _RawState:
    readings: dict[tuple[str, int], tuple[tuple[float, ...], int]]
    checks: list[dict]
    slot: int | None            # the slot of the last readout already in the file
    session: int                # this session's index
    fresh: bool = False         # no file yet: it is created once check 0 passes
    partial: bool = False       # a hard kill's incomplete last line, dropped once check 0 passes


def _write(f: IO[str], obj: dict) -> None:
    f.write(json.dumps(obj, separators=(",", ":")) + "\n")
    f.flush()


def _raw_failed(path: Path, e: OSError) -> RunFailed:
    """The raw file's own I/O failed (a full disk, a path under a regular file): exit 1 naming --raw. Only the raw
    file's call sites convert OSError this way, so a closed console (BrokenPipeError) is never blamed on --raw."""
    return RunFailed(f"--raw {path}: {e}")


def _read_raw(path: Path, header: dict) -> _RawState:
    try:
        raw = read_raw(path)
    except InvalidInput as e:
        raise Refusal(f"refusing to resume: {e}; remove the file to start over") from None
    diff = [k for k in HEADER_KEYS if _differs(raw.header, header, k)]
    if diff:
        raise Refusal(f"refusing to resume: {path} was written by a different run ({', '.join(diff)} differ from this "
                      f"one), and readouts from two runs cannot be mixed. Remove the file to start over"
                      f"{_serve_cache_hint(raw.header, header) if 'provenance' in diff else ''}")
    session = 1 + max((c["session"] for c in raw.checks), default=-1)
    return _RawState(raw.readings, raw.checks, raw.slot, session, partial=raw.partial)


def _serve_cache_hint(old: dict, new: dict) -> str:
    """The `callo serve` restart a resume refused on provenance needs when `cache_prompt` moved (--cache went on by
    default in 0.3.0, so every earlier raw file was read cache off), as a "; ..." tail, or ""
    when it did not move (or is not a bool on either side)."""
    old_c, new_c = ((h.get("provenance") if isinstance(h.get("provenance"), dict) else {}).get("cache_prompt")
                    for h in (old, new))
    if not isinstance(old_c, bool) or not isinstance(new_c, bool) or old_c == new_c:
        return ""
    return (f"; that run read through a `callo serve` with the cache {'on' if old_c else 'off'}: restart it "
            f"{'without --no-cache' if old_c else 'with --no-cache'} to resume (--cache is on by default since "
            f"callo 0.3)")


def ragged_reading(items: Sequence[Item], readings: Readings) -> str | None:
    """What is wrong with a raw file's readout whose probabilities are not its question's n (a hand edit that kept the
    row's own n and probs in step), or None. fit_n needs every question's rows to be n wide."""
    width = {it.uid: it.n for it in items}
    for (uid, s), (probs, _) in readings.items():
        if uid in width and len(probs) != width[uid]:
            return (f"its readout of {uid} (rotation {s}) has {len(probs)} probabilities, but the question has "
                    f"{width[uid]}")
    return None


def _differs(recorded: dict, current: dict, key: str) -> bool:
    """Whether --resume refuses on `key`: any difference, but a recorded provenance without `wide_contract` (a raw file
    written before it) ignores that field on both sides (`profile.resume_provenance`, one-directional)."""
    old, new = _comparable(recorded, key), _comparable(current, key)
    if key == "provenance":
        old, new = resume_provenance(old, new)
    return old != new


def _comparable(header: dict, key: str) -> Any:
    """A header field as --resume compares it: `server` without the predicted `slot`, which another lane can move
    between sessions (each readout's slot is enforced instead)."""
    value = header.get(key)
    if key == "server" and isinstance(value, dict):
        return {k: v for k, v in value.items() if k != "slot"}
    return value


def _prepare_raw(path: Path, header: dict, *, resume: bool) -> _RawState:
    """Validate --raw and read a resumed run. Writes nothing: the file is created, or appended to (after dropping a
    hard kill's partial last line), only once this session's check 0 passed."""
    if path.exists() and not resume:
        raise Refusal(f"refusing to calibrate: {path} already exists (--raw). Pass --resume to continue that run, or "
                      f"remove the file")
    if resume and path.exists():
        return _read_raw(path, header)
    return _RawState({}, [], None, 0, fresh=True)


def _open_raw(path: Path, header: dict, state: _RawState) -> IO[str]:
    if not state.fresh:
        if state.partial:
            # Cut back to the last newline in place, never a rewrite: one that crashed or met a full disk half way
            # would lose every readout.
            with path.open("r+b") as fh:
                fh.truncate(fh.read().rfind(b"\n") + 1)
        return path.open("a", encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise Refusal(f"refusing to calibrate: {path} appeared while calibrate started (--raw); is another calibrate "
                      f"running?") from None
    _write(f, {"callo_bias_raw": RAW_FORMAT, **header})
    return f


def _identity(stats: dict) -> dict:
    """What must not change within a session: the callo process (`instance`: a restart resets the counters conditions
    4 and 5 read), the model and prompt contract, and what shapes a readout or picks its slot."""
    return {"instance": stats.get("instance"), "model_key": stats.get("model_key"),
            "provenance": stats.get("provenance"), "temperature": stats.get("temperature"),
            "position_bias": stats.get("position_bias"), "kv": stats.get("kv"),
            "s1_slots": len((stats.get("router") or {}).get("reserved") or [])}


def _same_server(stats: dict, start: dict) -> None:
    now, then = _identity(stats), _identity(start)
    if now["instance"] != then["instance"]:
        raise Refusal("refusing to continue: callo restarted during calibration (/callo/stats instance changed), so "
                      "the guard lost track of what ran on the server since the last check. Rerun with --resume: "
                      "the new session starts with a fresh check")
    changed = [k for k in now if now[k] != then[k]]
    if changed:
        raise Refusal(f"refusing to continue: the server changed during calibration ({', '.join(changed)} differ from "
                      f"when this session started). Readouts from two configurations cannot be mixed; restore the "
                      f"server and rerun with --resume, or remove the raw file to start over")


def _plan(items: Sequence[Item], done: int, set_name: str, set_sha: str, seed: int, out: IO[str]) -> None:
    per_n = Counter(it.n for it in items)
    print(f"callo calibrate --bias: set {set_name} (sha256 {set_sha[:12]}), seed {seed}", file=out)
    for n in sorted(per_n):
        print(f"  n={n}: {per_n[n]} questions, {per_n[n] * n} readouts", file=out)
    print(f"  {sum(it.n for it in items)} readouts in all; {done} already in --raw", file=out)


@dataclass
class Collected:
    readings: dict[tuple[str, int], tuple[tuple[float, ...], int]]
    checks: list[dict]
    server: dict            # {"kv", "temperature", "s1_slots", "slot"}: the slot every readout came from
    stats: dict             # /callo/stats as this session began
    header: dict


def collect(base: str, items: Sequence[Item], *, raw_path: Path, seed: int, resume: bool, allow_busy: bool,
            set_name: str, set_sha: str, max_per_n: int = 48, out: IO[str] = sys.stderr,
            sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
            on_open: Callable[[], None] | None = None) -> Collected:
    """Read every rotation of every item. Raises Refusal (exit 2) or RunFailed (exit 1; the raw file's
    own I/O errors too, as "--raw <path>: ..."). `on_open` is called once this session has opened the raw file."""
    base = base.rstrip("/")
    try:
        stats = read_stats(base)
    except (*_DOWN, ValueError) as e:
        raise Refusal(f"refusing to calibrate: cannot read {base}/callo/stats ({e})") from None
    check_preconditions(stats, base)
    kinds = sorted({it.kind for it in items})
    s1_slots = len((stats.get("router") or {}).get("reserved") or [])
    slot = guard.predicted_slot(stats)
    server = {"kv": stats.get("kv"), "temperature": 1.0, "s1_slots": s1_slots, "slot": slot}
    header = {"model_key": stats["model_key"], "provenance": stats.get("provenance"), "server": dict(server),
              "set": {"name": set_name, "sha256": set_sha}, "seed": seed, "max_per_n": max_per_n,
              "instance": stats.get("instance")}            # recorded, never compared: a resume may follow a restart
    path = Path(raw_path)
    try:
        state = _prepare_raw(path, header, resume=resume)
    except OSError as e:
        raise _raw_failed(path, e) from None
    if resume and state.fresh:                  # the bench's "same command resumes" relies on this starting fresh
        print(f"--resume: no raw file at {path}; starting a fresh run", file=out)
    readings, checks = state.readings, state.checks
    if (ragged := ragged_reading(items, readings)) is not None:
        raise Refusal(f"refusing to resume: {path}: {ragged}; remove the file to start over")
    if state.slot is not None:
        slot = state.slot
    f: IO[str] | None = None
    mark: guard.Mark | None = None
    own = slack = 0

    def put(obj: dict) -> None:
        assert f is not None
        try:
            _write(f, obj)
        except OSError as e:
            raise _raw_failed(path, e) from None

    def check(current: dict | None = None) -> None:
        nonlocal f, mark, own, slack
        if current is None:
            current = read_stats_mid_run(base, sleep=sleep)
        _same_server(current, stats)
        v = guard.evaluate_state(current, slot=slot, kinds=kinds, prev=mark, own=own, slack=slack)
        rec = guard.record(len(checks), state.session, len(readings), v)
        mark, own, slack = v.mark, 0, 0
        refused = guard.message(v, stats=current, kinds=kinds, url=base) if v.failed and not allow_busy else None
        if f is None:
            if refused is not None:             # check 0 refused: this session read nothing, so it writes nothing
                kept = (f"nothing was written to {path}" if state.fresh
                        else f"{path} is unchanged ({len(readings)} readouts)")
                raise Refusal(f"{refused}\n{kept}; rerun the same command once the server is idle")
            try:
                f = _open_raw(path, header, state)
            except OSError as e:
                raise _raw_failed(path, e) from None
            if on_open is not None:
                on_open()
        checks.append(rec)
        put({"type": "check", **rec})
        if refused is not None:
            raise Refusal(f"{refused}\n{path} keeps its {len(readings)} readouts and this failed check: --resume "
                          f"continues the run, but the profile will then be marked busy; remove the file to start "
                          f"clean")
        if v.failed:
            print(f"WARNING: server-state check {rec['index']} failed ({', '.join(v.failed)}); --allow-busy: "
                  f"carrying on, and the profile will be marked busy", file=out)

    try:
        cells = [(it, s) for it in items for s in range(it.n)]
        random.Random(f"{seed}:order").shuffle(cells)
        todo = [(it, s) for it, s in cells if (it.uid, s) not in readings]
        _plan(items, len(cells) - len(todo), set_name, set_sha, seed, out)
        check(stats)
        assert f is not None                    # check 0 passed (or --allow-busy), so the file is open
        t0, done = clock(), 0
        for it, s in todo:
            what = f"{it.record_id} (question {it.qid}, rotation {s})"
            status, headers, resp, dropped = post_readout(base, it.body(s), what=what, sleep=sleep)
            slack += dropped
            if status != 200:
                raise RunFailed(f"{what}: HTTP {status}: {str(resp)[:300]}")
            probs, miss, entry = _parse_reply(it, resp, headers, what)
            got = parse_slot(headers.get("x-callo-slot"), s1_slots)
            if got is None:
                raise Refusal(f"refusing to continue: {what} came back without a usable X-Callo-Slot "
                              f"({headers.get('x-callo-slot')!r})")
            if slot is None:
                slot = got
            elif got != slot:
                raise Refusal(f"refusing to continue: calibrate's slot changed from {slot} to {got} at {what} (a lane "
                              f"rebind or the S1 fallback moved it, and readouts from two slots' states cannot be "
                              f"mixed). Make sure nothing else sends X-Callo-Conversation: {guard.LANE}, then "
                              f"restart `callo serve` (lanes reset) or remove the raw file; --resume alone would "
                              f"meet the same slot again")
            readings[(it.uid, s)] = (probs, miss)
            own += 1
            done += 1
            put({"type": "readout", "q": it.uid, "cluster": it.cluster, "n": it.n, "s": s, "probs": list(probs),
                 "miss": miss, "slot": slot, "timings": entry, "check": len(checks) - 1})
            if done == ETA_AFTER:
                per = (clock() - t0) / done
                left = len(todo) - done
                print(f"  measured {1000 * per:.0f} ms per readout: about {per * left / 60:.1f} min for the {left} "
                      f"left", file=out)
            if done % CHECK_EVERY == 0:
                check()
        if checks[-1]["readouts"] != len(readings):
            check()
    except BaseException:
        # A Refusal, RunFailed or Ctrl-C in flight is what the user must see: --raw's close error never replaces it.
        if f is not None:
            with contextlib.suppress(OSError):
                f.close()
        raise
    if f is not None:
        try:
            f.close()
        except OSError as e:
            raise _raw_failed(path, e) from None
    server["slot"] = slot
    return Collected(readings, checks, server, stats, header)


# ---- the fit, the profile and the CLI entry ----------------------------------------

def build_profile(items: Sequence[Item], collected: Collected, *, set_name: str, set_sha: str, seed: int,
                  skipped: Counter, capped: int, min_questions: int, draws: int = DRAWS,
                  builtin: bool = False) -> tuple[dict, dict]:
    """(the profile, what the summary needs). `prior` is the only stored correction; the display intervals are not
    stored. An n with no question read is in neither `entries` nor `dropped`. `builtin`: the set is
    the built-in one by sha256 (a copy under another name included), recorded as `set.builtin`, which `load_profile`
    refuses."""
    fits = fit_n(items, collected.readings)
    decisions = decide({n: (f.rows, list(f.clusters)) for n, f in fits.items()}, min_questions=min_questions,
                       draws=draws, seed=seed)
    intervals = {n: bootstrap_intervals(f.rows, list(f.clusters), seed=seed)
                 for n, f in fits.items() if f.questions >= 2}
    entries, dropped, offsets = {}, {}, {}
    for n, f in fits.items():
        d = decisions[n]
        if d.kept:
            entries[str(n)] = {"prior": list(f.prior), "questions": f.questions, "floored": f.floored,
                               "bias": d.bias, "p_holm": d.p_holm}
            offsets[n] = offsets_from_prior(f.prior)
        else:
            dropped[str(n)] = d.reason
    sources = Counter((it.n, it.source) for it in items)
    stats = collected.stats
    profile = {
        "callo_position_bias": FORMAT, "estimator": ESTIMATOR, "model_key": stats["model_key"],
        "model_name": stats.get("model_file"),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "callo_version": __version__,
        "provenance": stats.get("provenance"), "server": collected.server,
        "server_state": {"state": "busy" if any(c["failed"] for c in collected.checks) else "clean",
                         "checks": collected.checks},
        "set": {"name": set_name, "sha256": set_sha, "builtin": builtin, "seed": seed, "questions": len(items),
                "kinds": sorted({it.kind for it in items}),
                "sources_per_n": {str(n): {src: k for (m, src), k in sorted(sources.items()) if m == n}
                                  for n in sorted({it.n for it in items})},
                "skipped": dict(sorted(skipped.items())), "capped": capped},
        "entries": entries, "dropped": dropped,
        "eval": evaluate(items, collected.readings, offsets)}
    return profile, {"fits": fits, "decisions": decisions, "intervals": intervals}


def _cells(ps: Sequence[float]) -> str:
    return "  ".join(f"{p:.3f}" for p in ps)


def _or(words: Iterable[str], conj: str) -> str:
    """"a", "a and b", "a, b and c" (with `conj` "and" or "or")."""
    w = list(words)
    return w[0] if len(w) == 1 else f"{', '.join(w[:-1])} {conj} {w[-1]}"


def summary_lines(profile: dict, fit: dict) -> list[str]:
    """The human summary: per n kept or dropped with the reason, prior_n with display intervals, meanP_n
    raw and debiased, the floored questions, and the evaluation. On the built-in set (`set.builtin`) it is the
    diagnostic's whole output: an n is SIGNIFICANT rather than KEPT, only raw figures are shown (there is no
    correction to show "debiased" figures for), and nothing is corrected."""
    s, ev = profile["set"], profile["eval"]
    state = profile["server_state"]["state"]
    diagnostic = bool(s.get("builtin"))
    what = "position-bias DIAGNOSTIC (built-in set, no profile written)" if diagnostic else "position-bias profile"
    lines = [f"{what}: {profile.get('model_name') or profile['model_key']}, set {s['name']} "
             f"({s['questions']} questions, kinds {', '.join(s['kinds'])}), server_state {state}"]
    if state == "busy":                         # the diagnostic writes no profile, so nothing for serve to load
        lines.append("WARNING: server_state busy: a server-state check failed and --allow-busy carried on; other "
                     + ("slots' state may be part of these figures" if diagnostic else
                        "slots' state may be part of this prior, and `callo serve` will warn when it loads it"))
    for n, f in fit["fits"].items():
        d = fit["decisions"][n]
        kept = "SIGNIFICANT" if diagnostic else "KEPT"
        lines.append(f"n={n}: {kept if d.kept else 'dropped: ' + str(d.reason)}"
                     + (f" (bias {d.bias:.3f}, p_holm {d.p_holm:.2g})" if d.kept else ""))
        iv = fit["intervals"].get(n)
        cells = [f"{p:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if iv else "")
                 for p, (lo, hi) in zip(f.prior, iv or [(0.0, 0.0)] * n, strict=True)]
        lines.append("  prior_n  " + "  ".join(cells))
        k = str(n)
        if diagnostic:
            lines.append(f"  meanP_n  {_cells(ev['raw']['meanP'][k])}")
            lines.append(f"  {f.questions} questions read, {f.floored} of them with a floored option "
                         f"({f.floored / f.questions:.0%})")
            lines.append(f"  rotation consistency {ev['raw']['consistency'][k]:.3f}")
            continue
        lines.append(f"  meanP_n  {_cells(ev['raw']['meanP'][k])} raw -> {_cells(ev['debiased']['meanP'][k])} debiased")
        lines.append(f"  {f.questions} questions fitted, {f.floored} of them with a floored option "
                     f"({f.floored / f.questions:.0%}; fitted all the same)")
        lines.append(f"  rotation consistency {ev['raw']['consistency'][k]:.3f} raw -> "
                     f"{ev['debiased']['consistency'][k]:.3f} debiased")
    lines.append("intervals: 95% cluster bootstrap, per position, not simultaneous (display only)")
    if "accuracy" in ev["raw"] and diagnostic:
        r = ev["raw"]
        lines.append(f"labelled, T = 1, every rotation ({r['rows']} rows): accuracy {r['accuracy']:.3f}, NLL "
                     f"{r['nll']:.4f}, ECE10 {r['ece10']:.4f}")
    elif "accuracy" in ev["raw"]:
        r, d = ev["raw"], ev["debiased"]
        lines.append(f"labelled, T = 1, every rotation ({r['rows']} rows): accuracy {r['accuracy']:.3f} -> "
                     f"{d['accuracy']:.3f}, NLL {r['nll']:.4f} -> {d['nll']:.4f}, ECE10 {r['ece10']:.4f} -> "
                     f"{d['ece10']:.4f} (in-sample: the questions the profile was fitted on)")
    if diagnostic:
        found = (f"a shared position bias on the built-in set at n = {', '.join(profile['entries'])}"
                 if profile["entries"] else
                 "no shared position bias detected on the built-in set (not proof of none: biases of opposite sign "
                 "in different questions cancel)")
        lines.append(f"DIAGNOSTIC ONLY, no profile written: {found}. A correction measured on these questions does "
                     f"not transfer to yours, and this summary is not a profile; to correct your "
                     f"readouts, measure your own: callo calibrate --bias --data YOUR.jsonl --out bias.json")
    elif not profile["entries"]:
        lines.append("NOTHING WAS KEPT: every n was dropped, so this profile corrects nothing; serving with it is the "
                     "same as serving without")
    else:
        lines.append(f"corrects {', '.join(s['kinds'])} readouts with n = {', '.join(profile['entries'])}; refit any "
                     f"--temperature afterwards (a new profile invalidates a fitted T)")
    covered = [k for k in READOUT_KINDS if k in s["kinds"]]      # a readout's kind is exclusive (media_kind): a
    left_out = [k for k in READOUT_KINDS if k not in s["kinds"]]  # file of plain records corrects no context readout
    if left_out and not diagnostic:
        lines.append(f"note: this profile corrects {_or(covered, 'and')} readouts only, not {_or(left_out, 'or')} "
                     f"readouts; include such records in --data to have them corrected")
    if s["skipped"]:
        lines.append("skipped: " + ", ".join(f"{k} {v}" for k, v in s["skipped"].items()))
    return lines


def _raw_readouts(path: Path) -> int:
    """The readouts --raw keeps, for the resume hint: through read_raw, the one reader of the format (a file that is not
    UTF-8 is InvalidInput there, so it keeps 0)."""
    try:
        return len(read_raw(path).readings)
    except (OSError, InvalidInput):
        return 0


def run(args: argparse.Namespace, *, out: IO[str] | None = None, err: IO[str] | None = None) -> int:
    """`callo calibrate --bias`: 0 when the profile is written (even when every n is dropped) or the built-in
    diagnostic finished (it writes no profile), 1 for a failed request, a malformed --data line or no eligible
    questions, 2 for a refusal, 130 on Ctrl-C. A console that went away (BrokenPipeError on stdout or
    stderr) exits 1 without a traceback."""
    out, err = out or sys.stdout, err or sys.stderr
    try:
        return _run(args, out, err)
    except BrokenPipeError:
        return silence_stdout(out)


def silence_stdout(out: IO[str]) -> int:
    """After a BrokenPipeError on the console: point stdout at /dev/null so the interpreter's exit flush does not
    complain about it either, and return the exit code 1 (`--bias` and `--think-budgets` alike)."""
    if out is sys.stdout:
        try:
            fd = sys.stdout.fileno()            # first: a stream without a descriptor (a test's) opens nothing
            null = os.open(os.devnull, os.O_WRONLY)
        except (OSError, ValueError):
            return 1
        try:
            os.dup2(null, fd)
        except OSError:
            pass
        finally:
            os.close(null)                      # fd now points at devnull on its own
    return 1


def _run(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    seed = DEFAULT_SEED if args.seed is None else args.seed
    min_q = DEFAULT_MIN_QUESTIONS if args.min_questions is None else args.min_questions
    if min_q < 2 or (args.max_per_n is not None and args.max_per_n < 1):     # 0 questions would
        print("callo: --min-questions must be at least 2 and --max-per-n at least 1", file=err)   # reach the test
        return 2
    if seed < 0:                               # numpy's default_rng refuses it, but only at the fit
        print(f"callo: --seed must be at least 0, got {seed}", file=err)
        return 2
    raw_path: Path | None = Path(args.raw) if args.raw else None      # the default depends on the set
    summary_to = out if args.out else err
    opened = False                             # this run's collect opened --raw: only then does the hint apply

    def raw_opened() -> None:
        nonlocal opened
        opened = True

    try:
        if args.data is None:
            if args.out:                       # refused before the built-in set is read (or fetched)
                raise Refusal(OUT_NEEDS_DATA)
            try:                               # the only path that may fetch the built-in set
                set_name, raw = BUILTIN_SET, builtin_set_bytes(err=err)
            except BuiltinSetError as e:
                raise RunFailed(str(e)) from None
        else:
            set_name = Path(args.data).name
            try:
                raw = Path(args.data).read_bytes()
            except OSError as e:
                raise RunFailed(f"--data {args.data}: {e}") from None
        set_sha = hashlib.sha256(raw).hexdigest()
        diagnostic = set_sha == builtin_set_sha256()     # the built-in set, under any name, is a diagnostic
        if diagnostic and args.out:
            raise Refusal(OUT_NEEDS_DATA)
        max_n = (DEFAULT_MAX_PER_N if diagnostic else NO_CAP) if args.max_per_n is None else args.max_per_n
        if raw_path is None:
            raw_path = (Path(DIAGNOSTIC_RAW) if diagnostic else
                        Path(args.out).with_suffix(".raw.jsonl") if args.out else Path("callo-bias.raw.jsonl"))
        if diagnostic:
            summary_to = out
        try:
            scan = read_set(raw.decode("utf-8"))
        except UnicodeDecodeError as e:
            raise RunFailed(f"--data {set_name}: not UTF-8 ({e})") from None
        except SetError as e:
            raise RunFailed(f"--data {set_name} {e}") from None
        items, capped = cap_per_n(scan.items, max_n, seed)
        if not items:
            skipped = ", ".join(f"{k} {v}" for k, v in sorted(scan.skipped.items())) or "nothing"
            raise RunFailed(f"no eligible questions in {set_name} (skipped: {skipped}); --bias needs canonical choice "
                            f"or noul questions with 2 to 26 options")
        if diagnostic:
            print("the built-in set is a diagnostic: it measures, and writes no profile (fit one with --data "
                  "YOUR.jsonl --out bias.json)", file=summary_to)
        collected = collect(args.url, items, raw_path=raw_path, seed=seed, resume=args.resume,
                            allow_busy=args.allow_busy, set_name=set_name, set_sha=set_sha, max_per_n=max_n,
                            out=summary_to, on_open=raw_opened)
        profile, fit = build_profile(items, collected, set_name=set_name, set_sha=set_sha, seed=seed,
                                     skipped=scan.skipped, capped=capped, min_questions=min_q,
                                     builtin=diagnostic)
        lines = summary_lines(profile, fit)
        if diagnostic:
            print("\n".join(lines), file=out)
            return 0
        text = json.dumps(profile, indent=1) + "\n"
        if args.out:
            try:
                Path(args.out).write_text(text)
            except OSError as e:
                print(f"callo: cannot write --out {args.out}: {e}. --raw {raw_path} keeps every readout: to refit "
                      f"without reading again, rerun with --resume, a writable --out and the same raw file (pass "
                      f"--raw {raw_path} explicitly if --out changes: the default --raw follows --out)", file=err)
                return 1
            print("\n".join([*lines, f"profile written to {args.out}"]), file=out)
        else:
            out.write(text)
            out.flush()
            print("\n".join(lines), file=err)
        return 0
    except Refusal as e:
        print(f"callo: {e}", file=err)
        return 2
    except RunFailed as e:
        kept = _raw_readouts(raw_path) if opened and raw_path else 0
        hint = f"; --raw {raw_path} keeps {kept} readouts: rerun with --resume" if kept else ""
        print(f"callo: {e}{hint}", file=err)
        return 1
    except KeyboardInterrupt:                   # also while writing --out or the summary
        # The hint only for a raw file this run opened, or a --resume run's own (never one left by another run), and
        # only when it keeps something, as on the RunFailed path.
        kept = _raw_readouts(raw_path) if (opened or args.resume) and raw_path else 0
        hint = f"; --raw {raw_path} keeps {kept} readouts: rerun the same command with --resume" if kept else ""
        print(f"callo: interrupted{hint}", file=err)
        return 130
