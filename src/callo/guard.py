"""The server-state guard of `callo calibrate --bias`. Pure: it judges the /callo/stats snapshots calibrate fetched,
before the first readout, every 50 readouts and after the last.

A readout can move with what OTHER slots hold (docs/guide/limitations.md, "Readouts and unified KV": up to 0.20 when
another slot held 16 tokens, 0.34 at 300, on TielCoder with -kvu). A slot that holds tokens for the whole run shifts
every readout the same way, and the profile would encode that as position bias. So the guard is about STATE. A check
fails when:

1. `slots` is null: the state cannot be observed (--no-slots, or a /slots fetch error);
2. a slot other than calibrate's is processing;
3. a slot other than calibrate's holds tokens (`n_prompt_tokens` > 0, absent = 0), checked under unified KV and in
   attach mode (treated as unified), and under per-slot KV too when the set holds ANY kind other than "plain"
   (context, image, audio, video, mixed), because per-slot bit-exactness was measured on plain text only;
4. the monotonic `s1.readouts` grew by more than calibrate's own readouts since the last check: foreign S1 traffic
   (`s1.n` is a 2,000-wide window and cannot be used);
5. a `routes` key outside EXEMPT_ROUTES grew since the last check: traffic through callo that started and finished
   between two checks. Proxy Stats keys `routes` by PATH only (query stripped, no method), so the exempt set is a set
   of paths: read-only routes that cannot put tokens in a slot or run a batch (`/health`, `/props`, `/slots` (the GET
   listing), `/metrics`, `/models`, `/tokenize`, ...), `/callo/stats`, and `/v1/systemone` (condition 4 counts it).
   `/slots/{id}` (the POST erase/save/restore actions) changes slot state and is NOT exempt.

Calibrate never erases a slot itself; the refusal names the fixes that work for the detected configuration.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

LANE = "callo-calibrate"
# Condition 5 ignores these paths (Stats.routes is keyed by path, not method): read-only routes that cannot put tokens
# in a slot or run a batch, callo's own /callo/stats, and /v1/systemone, which condition 4 counts by readouts instead.
# /slots/{id} (POST erase/save/restore) changes slot state and is deliberately absent.
EXEMPT_ROUTES = frozenset({"/health", "/v1/health", "/props", "/slots", "/metrics", "/models", "/v1/models",
                           "/tokenize", "/detokenize", "/apply-template", "/callo/stats", "/v1/systemone",
                           "/callo/think-probe"})   # its readouts are counted by condition 4
CONDITIONS = ("slots_unobservable", "foreign_processing", "foreign_tokens", "foreign_s1", "foreign_routes")
# The refusal's last line, for machines (a script can read why a run refused from calibrate's stderr).
FAILED_LINE = re.compile(r"^\(failed conditions: ([a-z0-9_, ]+)\)$", re.M)


@dataclass(frozen=True)
class Mark:
    """What a check saw of the monotonic counters, for the next check's deltas."""
    readouts: int
    routes: dict[str, int]


@dataclass(frozen=True)
class Verdict:
    failed: tuple[str, ...]          # the CONDITIONS that failed, in order
    held: dict[int, int]             # slot -> n_prompt_tokens, for every slot but calibrate's that holds tokens
    processing: tuple[int, ...]      # slots but calibrate's that are processing
    foreign_readouts: int            # s1.readouts growth beyond calibrate's own (0 at the first check)
    grew: tuple[str, ...]            # routes other than calibrate's that grew
    mark: Mark
    slack: int = 0                   # extra readouts condition 4 tolerated: calibrate's retried requests


def tokens_matter(kv: str | None, kinds: Iterable[str]) -> bool:
    """Condition 3 applies under unified KV, in attach mode (kv None), and under per-slot KV whenever the set holds any
    kind other than "plain" (per-slot bit-exactness was measured on plain text only)."""
    return kv != "per_slot" or any(k != "plain" for k in kinds)


def predicted_slot(stats: dict) -> int | None:
    """The slot the `callo-calibrate` lane will use, as far as /callo/stats tells: 0 with `-np 1` (no reserved
    slots, X-Callo-Slot "-"); the lane's slot when it is already bound; the one reserved slot with --s1-slots 1;
    else the lowest reserved slot no lane holds. None when every reserved slot is held by another lane (callo would
    rebind the least-recently-used one, which it does not publish): the first readout then tells."""
    router = stats.get("router") or {}
    reserved = list(router.get("reserved") or [])
    lanes = router.get("lanes") or {}
    if not reserved:
        return 0
    if LANE in lanes:
        return int(lanes[LANE])
    if len(reserved) == 1:
        return int(reserved[0])
    held = {int(s) for s in lanes.values()}
    free = [s for s in reserved if s not in held]
    return int(free[0]) if free else None


def evaluate_state(stats: dict, *, slot: int | None, kinds: Iterable[str], prev: Mark | None, own: int,
                   slack: int = 0) -> Verdict:
    """One check. `slot` is calibrate's slot (None: unknown, so every slot counts as another's); `prev` is the
    previous check's Mark (None for a session's first check: conditions 4 and 5 need a delta); `own` is how many
    readouts calibrate itself got since then; `slack` is how many of its requests it retried after a connection
    error, each of which callo may have answered and counted (condition 4 tolerates that many, and records it)."""
    kinds = list(kinds)
    failed: list[str] = []
    held: dict[int, int] = {}
    busy: list[int] = []
    snapshot = stats.get("slots")
    if not isinstance(snapshot, list):
        failed.append("slots_unobservable")
    else:
        for s in snapshot:
            if not isinstance(s, dict) or not isinstance(s.get("id"), int) or s["id"] == slot:
                continue
            if s.get("is_processing"):
                busy.append(s["id"])
            n = s.get("n_prompt_tokens")
            if isinstance(n, int) and not isinstance(n, bool) and n > 0:
                held[s["id"]] = n
        if busy:
            failed.append("foreign_processing")
        if held and tokens_matter(stats.get("kv"), kinds):
            failed.append("foreign_tokens")
    s1 = stats.get("s1") or {}
    mark = Mark(int(s1.get("readouts") or 0), {str(k): int(v) for k, v in (stats.get("routes") or {}).items()})
    foreign, grew = 0, ()
    if prev is not None:
        foreign = max(0, mark.readouts - prev.readouts - own - slack)
        if foreign:
            failed.append("foreign_s1")
        grew = tuple(sorted(k for k, v in mark.routes.items() if k not in EXEMPT_ROUTES and v > prev.routes.get(k, 0)))
        if grew:
            failed.append("foreign_routes")
    return Verdict(tuple(failed), dict(sorted(held.items())), tuple(sorted(busy)), foreign, grew, mark, slack)


def record(index: int, session: int, readouts: int, v: Verdict, *, at: str | None = None) -> dict:
    """One `server_state.checks` entry of the profile: when, after how many readouts, what failed, and which
    slots held how many tokens. Appended across --resume sessions."""
    return {"index": index, "session": session,
            "at": at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "readouts": readouts,
            "failed": list(v.failed), "held": {str(k): n for k, n in v.held.items()},
            "processing": list(v.processing), "foreign_readouts": v.foreign_readouts, "routes_grew": list(v.grew),
            "slack": v.slack}


def _slots(ids: Iterable[int]) -> str:
    ids = sorted(ids)
    return ("slot " if len(ids) == 1 else "slots ") + ", ".join(str(s) for s in ids)


def message(v: Verdict, *, stats: dict, kinds: Iterable[str], url: str) -> str:
    """The refusal (exit 2). It names what was seen, then only the fixes that work here."""
    kinds = list(kinds)
    kv = stats.get("kv")
    probe = stats.get("probe") or {}
    residue = {int(s) for s in probe.get("residue") or []}
    lines = ["refusing to calibrate: the server is not idle, and its state would be measured as position bias:"]
    if "slots_unobservable" in v.failed:
        lines.append("- callo cannot observe the llama-server slots (GET /slots failed, or llama-server runs with "
                     "--no-slots)")
    if "foreign_processing" in v.failed:
        lines.append(f"- {_slots(v.processing)} {'is' if len(v.processing) == 1 else 'are'} processing a task")
    if "foreign_tokens" in v.failed:
        layout = {"unified": "unified KV (-kvu)", None: "attach mode (treated as unified KV)"}.get(
            kv, "per-slot KV with a non-plain set")
        shown = ", ".join(f"slot {s} ({n} tokens)" for s, n in v.held.items())
        lines.append(f"- {shown} hold{'s' if len(v.held) == 1 else ''} tokens; under {layout} that shifts every "
                     f"readout")
        if residue & set(v.held):
            lines.append(f"  {_slots(residue & set(v.held))} hold callo's own startup-probe tokens: callo serve could "
                         f"not erase them (probe.erase {probe.get('erase')!r})")
    if "foreign_s1" in v.failed:
        lines.append(f"- {v.foreign_readouts} System One readouts arrived that calibrate did not send")
    if "foreign_routes" in v.failed:
        lines.append(f"- traffic passed through callo on {', '.join(v.grew)}")
    fixes = []
    tokens = "foreign_tokens" in v.failed
    probe_held = residue & set(v.held) if tokens else set()
    if tokens and probe.get("erase") == "ok":
        fixes.append("erase the slots: " + "; ".join(f"curl -X POST '{url}/slots/{s}?action=erase'" for s in v.held))
    if probe_held and kv is None:               # attach mode: llama-server answered the erase 501
        fixes.append("restart llama-server with --slot-save-path DIR, then `callo serve`: its startup probe then "
                     "erases its own tokens")
    elif probe_held:                            # launch mode already passed the flag: see serve's residue warning
        fixes.append("restart `callo serve` (its startup probe erases its own tokens; its residue warning says why the "
                     "erase failed: a passthrough flag that overrode --slot-save-path, or the erase itself)")
    # Agent traffic is a fix only for what agent traffic caused: callo's own probe residue does not clear with it.
    if set(v.failed) & {"foreign_processing", "foreign_s1", "foreign_routes"} or (tokens and set(v.held) - residue):
        fixes.append("calibrate on a server with no agent traffic (nothing else talking to callo or llama-server)")
    if tokens and kv == "unified" and all(k == "plain" for k in kinds):
        fixes.append("launch with --no-kvu (plain-text sets only): per-slot KV made text readouts independent of other "
                     "slots")
    if "slots_unobservable" in v.failed:
        fixes.append("let callo read GET /slots (start llama-server without --no-slots)")
    fixes.append("or pass --allow-busy to record the event and carry on (the profile is then marked busy)")
    return "\n".join([*lines, "fixes:", *(f"- {f}" for f in fixes), f"(failed conditions: {', '.join(v.failed)})"])
