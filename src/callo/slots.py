"""Slot routing: reserved slots for classification, llama-server-style LCP/LRU routing for agents.

llama-server fills each batch slot by slot in id order, so the reserved classification slots are the LOWEST
ids. Agent routing mirrors llama-server's own slot choice (longest common prefix >= 10%, then LRU) so the
agent's prompt cache keeps working even though every request now carries an explicit `id_slot`. callo never
sends agent traffic to a busy slot while another non-reserved slot is idle; when all are busy it picks the
least loaded and accepts upstream deferral (counted).

Lanes: with 2 or more reserved slots, each distinct `X-Callo-Conversation` value on
`/v1/systemone` is bound to ONE reserved slot on first sight (the lowest-id slot no lane holds, else the
least-recently-used lane's slot), so three kinds of System One traffic with three system prefixes each keep their own
cached checkpoint. When the lane's slot is busy with callo's own request, the new request queues on that same
slot. When the slot is held by something callo did not dispatch, the S1 fallback applies and the request may
land on another idle slot, including one bound to a different lane.

Think budgets: `route_s1(deliberate=True)` sends a deliberate request (a trace, then its
readout) to the lowest idle reserved slot without a lane, and while any reserved slot holds a deliberate question a
fast request without a lane also prefers an idle reserved slot, so a 12 s generation does not queue readouts behind
it. Each reserved slot also has an S1 SECTION (`Section`): a reader/writer section with a FIFO ticket queue.
Budget-0 questions take it shared (they interleave pass by pass, exactly as in with no exclusive ticket ever
queued, every shared ticket is granted at once), a deliberate question takes it exclusive from its generation to its
last readout pass, so nothing runs on the slot between the two. Waits are bounded (`--s1-timeout`) and a deliberate
waiter leaves the queue when its client has gone. The section is taken only by the service layer, after `route_s1`
returned and the router lock was released; a request holds at most one section at a time.

Everything here is the router's OWN accounting. What llama-server is really doing is a `/slots` snapshot the
proxy supplies, already fetched, as the pure `processing` predicate when choosing an S1 slot, with the time the
snapshot was fetched (`as_of`): a slot callo released after that is past whatever the snapshot shows for it.

Media in the routing text: `canonical_text` replaces every media payload with a fixed-size
`sha256:<hex>` digest, in a COPY of the body (the bytes forwarded upstream keep the real payload), so a 20 MB video
costs the router lock O(1) instead of a 20 MB `commonprefix` per idle slot. llama-server matches media chunks by their
content hash, so equal payloads still mean an equal prefix. Its 10% LCP-vs-LRU threshold counts a media chunk at its
full size, though, so the digests are WEIGHTED: each carries its payload's length, the threshold's denominator is
`len(text) + Σ weights` and a digest lying wholly inside the common prefix adds its weight to the prefix, in
O(#media). Without media this is the threshold exactly; with media it scores each payload all-or-nothing, as
llama-server's hash-keyed media chunks do (it never counts the shared leading characters of two different payloads,
which llama-server never reuses).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import InvalidInput

log = logging.getLogger("callo")

# Endpoints whose body `id_slot` reaches the task (server-context.cpp:4463; responses via server-chat.cpp:15-16).
PINNABLE = frozenset({"/completion", "/completions", "/v1/completions", "/chat/completions",
                      "/v1/chat/completions", "/infill", "/v1/responses", "/responses"})
HINT_HEADERS = ("x-callo-conversation", "x-conversation-id")
DRAIN_S = 2.0                       # a cancelled deliberate slot counts busy this long at most
DEFAULT_ASR_HOME_IDLE = 600.0       # seconds a bound hint must be unused before its slot can be the callo:asr home
MAX_HINTS = 10_000                  # the conversation-hint bindings kept; the LRU one not on a busy slot goes
# the content-part types whose payload fields are digested (llama.cpp's chat parts, /responses'
# input_image, and the two types llama.cpp refuses, which still must not cost the router their size).
MEDIA_PART_TYPES = frozenset({"image_url", "input_image", "input_audio", "input_video", "video_url", "audio_url",
                              "video"})
PAYLOAD_FIELDS = ("url", "data", "image_url")
DIGEST_MIN = 256                                    # a payload string longer than this is digested
DIGEST_CHARS = len("sha256:") + 64                  # the length of one digest string in the routing text
Digests = Sequence[tuple[int, int]]                 # (char offset in the routing text, weight = payload length)


@dataclass(frozen=True)
class Decision:
    slot: int | None            # None: forward unpinned
    how: str                    # client | hint | lcp | lru | least_loaded | unpinned | idle_hint (a sticky home)
    rerouted: bool = False      # the client asked for a reserved slot and was moved
    deferred: bool = False      # every non-reserved slot was busy: upstream will defer this task


def _dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _digest(payload: str, found: list[tuple[str, int]]) -> str:
    # surrogatepass: a lone surrogate ("\ud800") is valid JSON, and a strict encode would raise
    d = "sha256:" + hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()
    found.append((d, len(payload)))
    return d


def _payloads(v: dict, found: list[tuple[str, int]]) -> dict:
    """One media part (or its inner object) with each long payload field digested: a new dict, or `v` unchanged."""
    out = {k: _digest(x, found) if k in PAYLOAD_FIELDS and isinstance(x, str) and len(x) > DIGEST_MIN
           else _payloads(x, found) if isinstance(x, dict) else x for k, x in v.items()}
    return out if any(out[k] is not v[k] for k in v) else v


def _digest_media(v: Any, found: list[tuple[str, int]]) -> Any:
    """A copy of `v` with every media part's payload digested. Copy on write: untouched sub-objects are
    shared and `v` itself is never changed, because the proxy forwards it. Walks in json.dumps order."""
    if isinstance(v, list):
        out = [_digest_media(x, found) for x in v]
        return out if any(a is not b for a, b in zip(out, v, strict=True)) else v
    if not isinstance(v, dict):
        return v
    t = v.get("type")
    if isinstance(t, str) and t in MEDIA_PART_TYPES:    # a JSON-schema "type": [...] is no media part
        return _payloads(v, found)
    out = {k: _digest_media(x, found) for k, x in v.items()}
    return out if any(out[k] is not v[k] for k in v) else v


def _digest_prompt(p: Any, found: list[tuple[str, int]]) -> Any:
    """/completion's object prompt `{"prompt_string", "multimodal_data": [base64, ...]}` (and arrays of them): every
    multimodal_data entry digested, in a copy."""
    if isinstance(p, list):
        out = [_digest_prompt(x, found) for x in p]
        return out if any(a is not b for a, b in zip(out, p, strict=True)) else p
    if isinstance(p, dict) and isinstance(p.get("multimodal_data"), list):
        return {**p, "multimodal_data": [_digest(x, found) if isinstance(x, str) else x for x in p["multimodal_data"]]}
    return p


def _located(text: str, found: list[tuple[str, int]]) -> list[tuple[int, int]]:
    out, at = [], 0
    for d, weight in found:                         # in json.dumps order, so each is found after the previous one
        i = text.find(d, at)
        # cannot happen (digests are ASCII, written verbatim); never -1
        if i < 0:
            continue
        out.append((i, weight))
        at = i + len(d)
    return out


def canonical_text(path: str, body: dict) -> tuple[str, list[tuple[int, int]]]:
    """The text llama-server would compare prefixes on (messages, instructions+input, or the prompt), with every media
    payload replaced by its digest, and the digests' (offset, weight) list. Never changes `body`."""
    found: list[tuple[str, int]] = []
    if path.endswith("/responses"):
        text = _dumps([body.get("instructions"), _digest_media(body.get("input"), found)])
    elif path.endswith("/infill"):
        text = _dumps([body.get("input_prefix"), body.get("input_suffix")])
    elif "messages" in body:
        text = _dumps(_digest_media(body["messages"], found))
    else:
        p = body.get("prompt")
        if isinstance(p, str):
            text = p
        elif isinstance(p, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in p):
            text = " ".join(str(x) for x in p)
        else:
            text = "" if p is None else _dumps(_digest_prompt(p, found))
    return text, _located(text, found)


def conversation_hint(headers: Mapping[str, str], body: dict) -> str | None:
    """X-Callo-Conversation, then X-Conversation-Id (the WebUI's), then OpenAI `user`, then `metadata.user_id`."""
    lower = {k.lower(): v for k, v in headers.items()}
    for h in HINT_HEADERS:
        v = lower.get(h)
        if v:
            return v
    user = body.get("user")
    if isinstance(user, str) and user:
        return user
    meta = body.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("user_id"), str) and meta["user_id"]:
        return meta["user_id"]
    return None


class Section:
    """One reserved slot's S1 section: FIFO tickets, readers together, one writer.

    A ticket may enter when no writer holds the section and, for an exclusive ticket, no reader holds it and no
    ticket is ahead of it in arrival order; for a shared ticket, when every ticket ahead of it is shared. So
    consecutive readers enter together, a reader queued behind a waiting writer waits for it, and a writer never
    starves behind readers that keep arriving. `waits` counts the tickets that had to wait at all."""

    def __init__(self):
        self._cv = threading.Condition()
        self._queue: list[tuple[int, bool]] = []        # (ticket, exclusive), in arrival order
        self._next = 0
        self._readers = 0
        self._writer = False
        self.waits = 0

    def _can_enter(self, ticket: int, exclusive: bool) -> bool:
        if self._writer:
            return False
        ahead = [x for t, x in self._queue if t < ticket]
        if exclusive:
            return self._readers == 0 and not ahead
        return not any(ahead)

    def acquire(self, *, exclusive: bool, timeout: float, cancel: threading.Event | None = None,
                check_after: float | None = None, on_check: Callable[[], bool] | None = None) -> bool:
        """True once the section is held (release it with the same `exclusive`); False when `timeout` seconds passed
        or `cancel` was set first (polled every 100 ms), in which case nothing is held.

        `check_after` / `on_check`: once, `check_after` seconds in and still queued, `on_check()` is called
        with the ticket KEPT in its place. True abandons the wait (False is returned, as for a timeout); False keeps
        waiting for the rest of `timeout`, in the same FIFO place and counted as one wait. `on_check` runs with the
        section's lock RELEASED (the ticket stays queued; the lock is retaken and `_can_enter` re-tested after it), so
        it may take any other lock. Lock order, the rule the code follows: no path takes a section's lock while
        holding the router's (`SlotRouter.snapshot` reads the sections outside its lock), and nothing holds a
        section's lock while calling out."""
        with self._cv:
            ticket = self._next
            self._next += 1
            self._queue.append((ticket, exclusive))
            t_in = time.monotonic()
            deadline = t_in + timeout
            check_at = None if (check_after is None or on_check is None) else t_in + check_after
            waited = False
            try:
                while not self._can_enter(ticket, exclusive):
                    waited = True
                    now = time.monotonic()
                    left = deadline - now
                    if cancel is not None and cancel.is_set():
                        return False
                    if left <= 0 and check_at is None:
                        return False
                    if check_at is not None and now >= check_at:
                        check_at = None
                        self._cv.release()
                        try:
                            abandon = on_check()
                        finally:
                            self._cv.acquire()
                        if abandon:
                            return False
                        continue
                    if left <= 0:                  # (unreachable with check_at None, handled above)
                        return False
                    if check_at is not None:
                        left = min(left, check_at - now)
                    self._cv.wait(min(0.1, left) if cancel is not None else left)
                if exclusive:
                    self._writer = True
                else:
                    self._readers += 1
                return True
            finally:
                self._queue.remove((ticket, exclusive))
                if waited:
                    self.waits += 1
                self._cv.notify_all()

    def release(self, *, exclusive: bool) -> None:
        with self._cv:
            if exclusive:
                self._writer = False
            else:
                self._readers = max(0, self._readers - 1)
            self._cv.notify_all()

    def snapshot(self) -> dict:
        with self._cv:
            return {"readers": self._readers, "writer": self._writer, "queued": len(self._queue), "waits": self.waits}


def asr_home_idle_problem(seconds: Any) -> str | None:
    """`callo serve --asr-home-idle`'s usage error, or None: a finite number of seconds, 0 or more."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        return f"--asr-home-idle must be a finite number of seconds, 0 or more, got {seconds!r}"
    return None


class SlotRouter:
    def __init__(self, total_slots: int, s1_slots: int = 1, *, agent_pinning: bool = True,
                 lcp_threshold: float = 0.10, asr_home_idle: float = DEFAULT_ASR_HOME_IDLE,
                 max_hints: int = MAX_HINTS, clock: Callable[[], float] = time.monotonic):
        if total_slots < 1:
            raise ValueError(f"total_slots must be >= 1, got {total_slots}")
        if s1_slots < 1:
            raise InvalidInput(f"--s1-slots must be >= 1, got {s1_slots}")
        if total_slots >= 2 and s1_slots >= total_slots:
            raise InvalidInput(f"--s1-slots {s1_slots} must be below the upstream's total_slots {total_slots}")
        if problem := asr_home_idle_problem(asr_home_idle):
            raise InvalidInput(problem)
        self.total_slots = total_slots
        self.pinning = total_slots >= 2 and agent_pinning
        self.reserved = range(s1_slots) if total_slots >= 2 else range(0)
        self.agent = list(range(len(self.reserved), total_slots)) if total_slots >= 2 else []
        self.lcp_threshold = lcp_threshold
        self.counters: Counter[str] = Counter()
        self._inflight = dict.fromkeys(range(total_slots), 0)
        self._text = dict.fromkeys(self.agent, "")
        self._last_used = dict.fromkeys(self.agent, 0)      # a monotonic tick, not wall time: deterministic LRU
        self._tick = 0
        self._conv: dict[str, int] = {}
        # each binding's latest use (`clock`, monotonic), least recently used first; capped at `max_hints`
        self._conv_used: OrderedDict[str, float] = OrderedDict()
        self._bound: Counter[int] = Counter()               # bindings per slot: eviction's O(slots) early exit (m1)
        self.asr_home_idle = asr_home_idle
        self.max_hints = max_hints
        self._clock = clock
        self._sticky: set[str] = set()                      # hints routed sticky: callo:asr
        self._homed: set[str] = set()                       # sticky hints that have had a home
        self._rehomed: Counter[str] = Counter()             # sticky hint -> homes chosen after the first
        self._released_at: dict[int, float] = {}            # time.monotonic() of each slot's latest release
        self._freed_at: dict[int, float] = {}               # `clock` at each slot's latest release
        self._slot_hint: dict[int, str | None] = {}         # the bound hint of each slot's latest request
        self._s1_next = 0
        self._lanes: dict[str, int] = {}                    # /v1/systemone lane -> its reserved slot
        self._lane_used: dict[str, int] = {}                # lane -> router tick of its latest request (LRU)
        self._lane_rebind_warned = False                    # the once-per-process thrash warning
        self._sections = {s: Section() for s in self.reserved}      # one S1 section per slot
        self._deliberate = dict.fromkeys(self.reserved, 0)          # deliberate questions in flight per slot
        self._draining: dict[int, float] = {}                       # slot -> time.monotonic() of its cancel
        self._lock = threading.Lock()

    def section(self, slot: int | None) -> Section | None:
        """The S1 section of a reserved slot; None for an unreserved slot (-np 1) or None."""
        return self._sections.get(slot) if slot is not None else None

    # ---- client id_slot ---------------------------------------------------------------------------

    def normalise_id_slot(self, v: Any) -> int | None:
        """`v % total_slots` for an int v >= 0, matching get_slot_by_id (server-context.cpp:1542-1544);
        v < 0, bools, floats, strings and null are 'absent'."""
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            return None
        return v % self.total_slots

    # ---- agent routing ----------------------------------------------------------------------------

    def choose_agent(self, text: str, *, hint: str | None = None, client_slot: Any = None,
                     digests: Digests = ()) -> Decision:
        with self._lock:
            return self._choose_agent_locked(text, hint, client_slot, digests)

    def route_agent(self, text: str, *, hint: str | None = None, client_slot: Any = None,
                    digests: Digests = (), sticky: bool = False) -> Decision:
        """choose_agent + acquire in ONE critical section: concurrent requests can never both be handed the
        same idle slot. The caller must `release(d.slot)` when `d.slot` is not None. `digests` is
        `canonical_text`'s second value.

        `sticky` (the proxy's synthetic `callo:asr` hint only): a hint that already has a home
        keeps it. When the home is busy this request takes the normal pick, but the hint is not rebound to that slot, so
        the next request goes home again. A sticky hint with no home (the first request, or after an agent claimed the
        home: `_acquire_locked`) chooses one with `_choose_home_locked`: an idle agent slot no hint is bound to; when
        every such slot is busy it stays homeless for this request, which takes the normal pick (a busy
        home is never waited for). False, every other caller, rebinds the hint to the slot it was given, as always."""
        with self._lock:
            homeless = sticky and hint is not None and hint not in self._conv
            bind = hint
            if homeless and self.pinning and client_slot is None:
                d, homed = self._choose_home_locked(text, hint, digests)
                bind = hint if homed else None
            else:
                d = self._choose_agent_locked(text, hint, client_slot, digests)
                if sticky and hint is not None and not homeless:
                    bind = None                    # a sticky hint keeps its home, busy or not
            if d.slot is not None:
                if sticky and hint is not None:
                    self._sticky.add(hint)
                    if bind is not None:           # a new home: counted when it is not the first
                        if hint in self._homed:
                            self._rehomed[hint] += 1
                        self._homed.add(hint)
                self._acquire_locked(d.slot, text, bind)
                if bind is None and hint is not None and hint in self._conv:
                    self._touch_locked(hint)       # a sticky hint used from its home (or overflowing) is in use
                    if self._conv[hint] == d.slot:
                        self._slot_hint[d.slot] = hint
            return d

    def _choose_home_locked(self, text: str, hint: str | None, digests: Digests) -> tuple[Decision, bool]:
        """The home of a sticky hint is an agent slot that
        no hint in use holds, so a transcription stream does not settle on a slot an agent conversation has claimed.
        The candidates are the slots no hint is bound to, then the slots whose every bound hint has gone unused for
        at least `asr_home_idle` seconds, counted from the later of the hints' latest use and the slot's latest release
        (a long turn that just ended is not idle); the bindings are kept, so a returning agent still gets its slot
        back by its hint and that re-homes the sticky hint (`_acquire_locked`). The LRU idle hint-free slot wins,
        else the idle slot whose hints were used least recently ("idle_hint"). When every candidate is busy the
        request takes the normal pick (`_choose_agent_locked`: LRU idle, else least-loaded) and the hint stays
        homeless (False), exactly as a busy home sends one request elsewhere: a busy home is never waited for. Only
        when there is no candidate (every agent slot held by a hint in use) is the home the normal pick (True; the
        empty routing text never wins LCP). Such a home lasts until that slot's agent hint is next routed to it,
        which unbinds the sticky hint again."""
        holders: dict[int, list[str]] = {}
        for h, slot in self._conv.items():
            holders.setdefault(slot, []).append(h)
        now = self._clock()

        def last_use(slot: int) -> float:       # idle counts from the END of the slot's latest request
            return max(max(self._conv_used[h] for h in holders[slot]), self._freed_at.get(slot, float("-inf")))

        unbound = [s for s in self.agent if s not in holders]
        stale = [s for s in self.agent if s in holders and now - last_use(s) >= self.asr_home_idle]
        if not unbound and not stale:
            return self._choose_agent_locked(text, hint, None, digests), True
        idle = [s for s in unbound if self._inflight[s] == 0]
        if idle:
            slot = min(idle, key=lambda s: (self._last_used[s], s))
            self.counters["lru"] += 1
            return Decision(slot, "lru"), True
        idle = [s for s in stale if self._inflight[s] == 0]
        if idle:
            slot = min(idle, key=lambda s: (last_use(s), s))
            self.counters["idle_hint"] += 1
            return Decision(slot, "idle_hint"), True
        return self._choose_agent_locked(text, hint, None, digests), False

    def _lcp_enough(self, text: str, n: int, digests: Digests) -> bool:
        """llama-server's 10% LCP-vs-LRU test, with each digest weighted by its payload's length: a
        digest wholly inside the common prefix `text[:n]` counts its weight towards it; every digest counts towards
        the whole. With no digests this is the `n >= 10% of len(text)`, exactly; with media each payload counts
        all-or-nothing, as llama-server's hash-keyed media chunks do."""
        inside = sum(w for at, w in digests if at + DIGEST_CHARS <= n)
        return n + inside >= self.lcp_threshold * (len(text) + sum(w for _, w in digests))

    def _choose_agent_locked(self, text: str, hint: str | None, client_slot: Any, digests: Digests = ()) -> Decision:
        if not self.pinning:
            self.counters["unpinned"] += 1
            return Decision(None, "unpinned")
        rerouted = False
        if client_slot is not None:
            slot = self.normalise_id_slot(client_slot)
            if slot is None:
                self.counters["id_slot_ignored"] += 1
            elif slot in self.reserved:
                self.counters["rerouted"] += 1
                rerouted = True
            else:
                self.counters["client"] += 1
                return Decision(slot, "client")
        idle = [s for s in self.agent if self._inflight[s] == 0]
        if hint is not None:
            known = self._conv.get(hint)
            if known is not None and known in idle:
                self.counters["hint"] += 1
                return Decision(known, "hint", rerouted)
        if idle:
            best, best_len = None, 0
            for s in idle:
                n = len(os.path.commonprefix([self._text[s], text]))
                if n > best_len:
                    best, best_len = s, n
            if best is not None and text and self._lcp_enough(text, best_len, digests):
                self.counters["lcp"] += 1
                return Decision(best, "lcp", rerouted)
            lru = min(idle, key=lambda s: (self._last_used[s], s))
            self.counters["lru"] += 1
            return Decision(lru, "lru", rerouted)
        least = min(self.agent, key=lambda s: (self._inflight[s], self._last_used[s], s))
        self.counters["deferred"] += 1
        self.counters["least_loaded"] += 1
        return Decision(least, "least_loaded", rerouted, deferred=True)

    def acquire(self, slot: int, text: str, hint: str | None = None) -> None:
        with self._lock:
            self._acquire_locked(slot, text, hint)

    def _acquire_locked(self, slot: int, text: str, hint: str | None) -> None:
        self._inflight[slot] += 1
        self._tick += 1
        if slot in self._text:
            self._text[slot] = text
            self._last_used[slot] = self._tick
        self._slot_hint[slot] = hint
        if hint is not None:
            if hint not in self._sticky:          # an agent claiming a sticky home unbinds it
                for home in [h for h in self._sticky if self._conv.get(h) == slot]:
                    self._unbind_locked(home)
            if hint in self._conv:
                self._bound[self._conv[hint]] -= 1
            self._conv[hint] = slot
            self._bound[slot] += 1
            self._touch_locked(hint)
            self._evict_locked(keep=hint, acquiring=slot)

    def _touch_locked(self, hint: str) -> None:
        self._conv_used[hint] = self._clock()
        self._conv_used.move_to_end(hint)

    def _evict_locked(self, *, keep: str | None = None, acquiring: int | None = None) -> None:
        """The hint cap: while there are more than `max_hints` bindings, drop the least recently used one
        whose slot has no request in flight, counted as `hint_evicted`. The request being acquired (on `acquiring`)
        does not count, so the hints it displaces can go; `keep`, the binding just made, never does. When every other
        binding's slot is busy the table stays over the cap until a release leaves a slot that hosts a binding with
        nothing in flight (`release`). While no such slot exists it returns before scanning the
        bindings, so a saturated server pays O(slots) per request, not O(bindings)."""
        while len(self._conv) > self.max_hints:
            pinned = self._conv.get(keep) if keep is not None else None
            if not any(n - (s == pinned) > 0 and self._inflight[s] - (s == acquiring) == 0
                       for s, n in self._bound.items()):
                return
            for h in self._conv_used:
                slot = self._conv[h]
                if h != keep and self._inflight[slot] - (slot == acquiring) == 0:
                    self._unbind_locked(h)
                    self.counters["hint_evicted"] += 1
                    break
            else:
                return

    def _unbind_locked(self, hint: str) -> None:
        self._bound[self._conv.pop(hint)] -= 1
        del self._conv_used[hint]

    def release(self, slot: int) -> None:
        with self._lock:
            self._inflight[slot] = max(0, self._inflight[slot] - 1)
            self._released_at[slot] = time.monotonic()
            self._freed_at[slot] = self._clock()            # the slot's latest request ended now ...
            hint = self._slot_hint.get(slot)
            if hint is not None and self._conv.get(hint) == slot:
                self._touch_locked(hint)                    # ... and so did its bound hint's
            self._evict_locked()                            # a binding held over the cap by a busy slot can go

    def inflight(self, slot: int) -> int:
        with self._lock:
            return self._inflight[slot]

    # ---- classification ---------------------------------------------------------------------------

    def choose_s1(self, processing: Callable[[int], bool] | None = None, *,
                  as_of: float | None = None, lane: str | None = None) -> tuple[int | None, bool]:
        """The next reserved slot (round robin per request), or, when it holds a task callo did not send, the
        lowest-id idle slot (the S1 fallback). Returns (slot, fell_back).

        `processing(slot)` says whether llama-server reports the slot busy, from a `/slots` snapshot the CALLER
        fetched before this call. It runs under the router lock, so it must be pure: no I/O and no call back into
        the router (the lock is not reentrant). A slot is FOREIGN iff it is processing and the router's own
        in-flight count for it is 0; that count is read here, under the same lock.

        `as_of` is the `time.monotonic()` at which that snapshot's GET /slots was sent. For a slot callo released
        at or after `as_of`, the snapshot can only be showing callo's own finished task, so its `processing` is
        ignored: without this, a snapshot cached up to 250 ms (or fetched while a concurrent classification was
        finishing) turns callo's OWN just-released S1 into a "foreign" task and the fallback evicts an agent
        slot's KV. None: trust `processing` as is.

        `lane`: the request's X-Callo-Conversation. With 2+ reserved slots it picks the lane's own slot
        instead of the round robin; the fallback below still applies."""
        with self._lock:
            return self._choose_s1_locked(processing, as_of, lane)

    def route_s1(self, processing: Callable[[int], bool] | None = None, *,
                 as_of: float | None = None, lane: str | None = None,
                 deliberate: bool = False) -> tuple[int | None, bool]:
        """choose_s1 + acquire_s1 in ONE critical section (same purity rule for `processing`). The caller must
        `release_s1(slot, deliberate=...)` with the same flag when the slot is not None. `deliberate`: the request
        will generate a trace, so it takes the lowest idle reserved slot without a lane, else the
        least busy slot holding no deliberate question; a fast request avoids the slots that are thinking the same
        way, falling back to the round robin only when every reserved slot thinks."""
        with self._lock:
            slot, fell_back = self._choose_s1_locked(processing, as_of, lane, deliberate)
            if slot is not None:
                self._acquire_s1_locked(slot, deliberate)
            return slot, fell_back

    def _lane_slot_locked(self, lane: str) -> int:
        slot = self._lanes.get(lane)
        if slot is None:
            held = set(self._lanes.values())
            free = [s for s in self.reserved if s not in held]
            if free:
                slot = free[0]
            else:
                victim = min(self._lanes, key=self._lane_used.__getitem__)
                slot = self._lanes.pop(victim)
                del self._lane_used[victim]
                self.counters["s1_lane_rebound"] += 1
                if not self._lane_rebind_warned:
                    self._lane_rebind_warned = True
                    k = len(self.reserved)
                    log.warning(
                        "more distinct X-Callo-Conversation values than --s1-slots (%d): lanes now evict each other "
                        "on every request and no lane's system-turn cache stays warm (worse than plain round robin). "
                        "At most %d conversations stay cache-warm; keep concurrent conversations at or below %d, or "
                        "drop the header. (logged once)", k, k, k)
            self._lanes[lane] = slot
        self._tick += 1
        self._lane_used[lane] = self._tick
        return slot

    def mark_draining(self, slot: int | None) -> None:
        """A deliberate question on `slot` was cancelled; its upstream task outlives the cancel by up to
        ~1 s, until llama-server polls the connection, so `route_s1` treats the slot as busy until a `/slots` snapshot
        taken after the cancel shows it idle, or DRAIN_S pass. Counted as `s1_draining`."""
        if slot is None or slot not in self._deliberate:
            return
        with self._lock:
            self._draining[slot] = time.monotonic()
            self.counters["s1_draining"] += 1

    def route_quiet(self, processing: Callable[[int], bool] | None = None, *, as_of: float | None = None,
                    exclude: int | None = None) -> int | None:
        """Choose AND take (not deliberate; release with `release_s1(slot)`) a reserved slot other than
        `exclude` that holds no deliberate question, in one critical section, so a deliberate request cannot take the
        last quiet slot in between. `processing` / `as_of` are `route_s1`'s: a draining slot (after `_drained_locked`)
        and a slot llama-server shows busy with a task callo did not send are not eligible. Fewest in flight, unbound
        lane first. None: no such slot."""
        with self._lock:
            self._drained_locked(processing, as_of)

            def foreign(s: int) -> bool:
                return (processing is not None and self._inflight[s] == 0 and processing(s)
                        and (as_of is None or as_of > self._released_at.get(s, float("-inf"))))

            quiet = [s for s in self.reserved if s != exclude and self._deliberate[s] == 0
                     and s not in self._draining and not foreign(s)]
            if not quiet:
                return None
            bound = set(self._lanes.values())
            slot = min(quiet, key=lambda s: (self._inflight[s], s in self._draining, s in bound, s))
            self._acquire_s1_locked(slot, False)
            return slot

    def _drained_locked(self, processing: Callable[[int], bool] | None, as_of: float | None) -> None:
        now = time.monotonic()
        for s, t in list(self._draining.items()):
            if now - t >= DRAIN_S or (processing is not None and as_of is not None and as_of > t and not processing(s)):
                del self._draining[s]

    def _choose_s1_locked(self, processing: Callable[[int], bool] | None,
                          as_of: float | None, lane: str | None = None,
                          deliberate: bool = False) -> tuple[int | None, bool]:
        if not self.reserved:
            return None, False
        self._drained_locked(processing, as_of)
        if lane is not None and len(self.reserved) >= 2:
            slot = self._lane_slot_locked(lane)
            if self._inflight[slot] > 0:
                self.counters["s1_lane_busy"] += 1    # queued upstream on its own slot, never moved
        else:
            # a draining slot (a cancelled deliberate question's task still upstream) is not idle
            idle = [s for s in self.reserved if self._inflight[s] == 0 and s not in self._draining]
            bound = set(self._lanes.values())
            quiet = [s for s in self.reserved if self._deliberate[s] == 0]   # no deliberate question in flight
            idle_unbound = [s for s in idle if s not in bound]
            if deliberate and idle_unbound:
                slot = idle_unbound[0]                                  # an idle, unbound slot
                self.counters["s1_deliberate_idle"] += 1
            elif deliberate and quiet:                                  # else the least busy quiet slot,
                slot = min(quiet, key=lambda s: (self._inflight[s], s in self._draining, s in bound, s))  # unbound
                self.counters["s1_deliberate_quiet"] += 1
                if slot in bound:                   # the trace evicts that lane's warm turn: visible, not silent
                    self.counters["s1_deliberate_lane_slot"] += 1
            elif not deliberate and len(quiet) < len(self.reserved) and quiet:
                # while a slot thinks, a fast readout never queues behind it: an idle quiet slot (unbound
                # first, so a lane's warm turn survives), else the least busy quiet slot; the round robin only when
                # every reserved slot holds a deliberate question.
                idle_quiet = [s for s in idle if s in quiet]
                unbound = [s for s in idle_quiet if s not in bound]
                slot = (unbound or idle_quiet or [min(quiet, key=lambda s: (self._inflight[s], s in self._draining, s))])[0]
                self.counters["s1_avoided_deliberate"] += 1
            else:
                rr = [s for s in self.reserved if s not in self._draining] or list(self.reserved)
                slot = rr[self._s1_next % len(rr)]
                self._s1_next += 1
        if processing is None:
            return slot, False

        def busy(s: int) -> bool:   # llama-server's view, unless callo released the slot since the snapshot
            return processing(s) and (as_of is None or as_of > self._released_at.get(s, float("-inf")))

        if self._inflight[slot] > 0 or not busy(slot):
            return slot, False
        self.counters["s1_fallback"] += 1
        for s in range(self.total_slots):
            if s != slot and self._inflight[s] == 0 and not busy(s):
                return s, True
        return slot, True

    def acquire_s1(self, slot: int, deliberate: bool = False) -> None:
        with self._lock:
            self._acquire_s1_locked(slot, deliberate)

    def _acquire_s1_locked(self, slot: int, deliberate: bool = False) -> None:
        self._inflight[slot] += 1
        if deliberate and slot in self._deliberate:
            self._deliberate[slot] += 1
        if slot in self._text:
            self._text[slot] = ""       # the S1 prompt evicts this agent slot's KV: LCP must not route here

    def release_s1(self, slot: int, deliberate: bool = False) -> None:
        with self._lock:                    # one critical section: route_s1 never sees the deliberate count dropped
            if deliberate and slot in self._deliberate:     # while the slot still counts in flight
                self._deliberate[slot] = max(0, self._deliberate[slot] - 1)
            self._inflight[slot] = max(0, self._inflight[slot] - 1)
            self._released_at[slot] = time.monotonic()

    def snapshot(self) -> dict:
        with self._lock:
            snap = {"total_slots": self.total_slots, "reserved": list(self.reserved), "pinning": self.pinning,
                    "inflight": {str(k): v for k, v in self._inflight.items()},
                    "text_chars": {str(k): len(v) for k, v in self._text.items()},
                    "conversations": sum(h not in self._sticky for h in self._conv), "counters": dict(self.counters),
                    "sticky": {h: {"home": self._conv.get(h), "rehomed": self._rehomed[h]} for h in sorted(self._sticky)},
                    "lanes": dict(self._lanes), "s1_lane_busy": self.counters["s1_lane_busy"],
                    "deliberate": {str(k): v for k, v in self._deliberate.items()},
                    "draining": sorted(self._draining)}
        # outside the router's lock: a section's lock is never taken under it (see Section.acquire)
        snap["sections"] = {str(k): s.snapshot() for k, s in self._sections.items()}
        return snap
