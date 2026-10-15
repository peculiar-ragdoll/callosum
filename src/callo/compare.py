"""`callo compare`: several think profiles on one joint efficient frontier over (model, budget),
offline. A menu run (`menu.py`) ends by calling the same functions on its profiles, and a single profile is a valid
input (that is how a one-server run gets its `report.md` / `report.json`).

Refusals (exit 2): different data (`comparable.data.sha256`), a different metrics version, different readout-engine
settings (`variant`, `top_k`, `max_depth`, `divergence_mass`), or question id sets that differ. Everything
else that can make points incomparable is a loud FLAG above the table (hosts, an unknown host, builds, cache, a busy
or non-local server, an unverified latency model, drift, the same model under two samplers, seeds, bias profiles,
warm-up, server args, bootstrap strata), and the report goes on. Paired CIs are over the shared question ids: the
items are identical across profiles, so "A@1024 is better than B@1024" is a measured claim, not two overlapping
intervals; they are stratified by source when every profile was (`comparable.data.strata`).
"""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibrate_think import (
    DEADLINE_CLOSES,
    EE_MARK,
    PREFILL_REUSED_MAX,
    SMALL_N,
    _ci,
    _pp,
    _s,
    converge_lines,
    limit_lines,
    opt_in_label,
    opt_in_line,
    parse_deadlines,
    pick_line,
    pick_note,
    serve_command,
    serve_note,
    stat_cells,
    tails_line,
)
from .converge import unservable
from .deadline_map import bmax_of, mapping, mapping_lines, point_cap
from .errors import InvalidInput
from .think import (
    CLOSES,
    EE_KINDS,
    METRICS_VERSION,
    PROFILE_FORMAT,
    STAT_FIELDS,
    Paired,
    Pick,
    Point,
    Preference,
    _num_text,
    ee_key,
    finite_number,
    frontier,
    mark_approx,
    on_frontier,
    pick,
    point_latencies,
    point_stats,
    preference_from_args,
    tail_counts,
)

REPORT_FORMAT = 1
READOUT_KEYS = ("variant", "top_k", "max_depth", "divergence_mass")
TRACE_FIELDS = ("temperature", "top_p", "top_k", "min_p")
LATENCY_LINE = "latency: single-stream, in-callo wall time, on {host}"
OWN_SAMPLER = ("Each model thinks with its own sampler (shown per point). Differing samplers are part of each model's "
               "point, not noise.")


def load_profile(path: str | Path) -> dict:
    """A think profile from disk; anything else (a bias profile, a raw file, not JSON) is InvalidInput."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise InvalidInput(f"{path}: cannot read it ({e.strerror or e})") from None
    except ValueError as e:
        raise InvalidInput(f"{path}: not JSON ({e})") from None
    if not isinstance(doc, dict) or doc.get("callo_think_profile") != PROFILE_FORMAT:
        raise InvalidInput(f"{path}: not a think profile (callo_think_profile {PROFILE_FORMAT} from "
                           f"`callo calibrate --think-budgets`)")
    why = malformed(doc)
    if why is not None:
        raise InvalidInput(f"{path}: malformed think profile: {why}")
    return doc


def _int(v: Any) -> bool:
    """A budget key as calibrate writes it: `str(int(b)) == b` (`" 0"` or `"00"` would miss `budgets[str(int(b))]`)."""
    try:
        return str(int(v)) == v
    except (TypeError, ValueError):
        return False


# the sections compare reads with `.get`: each is a JSON object or absent (null)
SECTIONS = ("environment", "environment.model", "environment.server", "environment.host", "environment.run",
            "thinking", "drift", "audit", "server_state", "comparable", "comparable.data", "comparable.sampling",
            "comparable.sampling.pinned", "comparable.sampling.origin", "comparable.readout",
            "latency", "latency.read_forced_ms")                  # what `deadline_map` reads the reserve from


def malformed(doc: dict) -> str | None:
    """What makes a think profile unreadable by compare (a hand-edited or truncated file), naming the field, or None:
    compare refuses it in one line naming the file rather than failing with a traceback. Structure (the rows
    `correct_at` reads, the budgets `points_of` reads), value types (finite numbers
    for accuracy and latencies, finite-or-null `think_mean` / `forced`) and the sections read with `.get`."""
    for path in SECTIONS:
        v: Any = doc
        for k in path.split("."):
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None and not isinstance(v, dict):
            return f"{path} is not a JSON object"
    model = (doc.get("environment") or {}).get("model") or {}
    for k in ("bytes", "mmproj_bytes"):
        if model.get(k) is not None and (not isinstance(model[k], int) or isinstance(model[k], bool)):
            return f"environment.model.{k} is not an integer or null"
    therm = (doc.get("drift") or {}).get("therm")
    if therm is not None and (not isinstance(therm, list) or not all(isinstance(t, dict) for t in therm)):
        return "drift.therm is not a list of JSON objects"
    seed = (doc.get("comparable") or {}).get("seed")      # the bootstrap's seed: calibrate refuses --seed below 0
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        return "comparable.seed is not an integer >= 0"
    fingerprint = ((doc.get("environment") or {}).get("host") or {}).get("fingerprint")   # "sha256:...", or None
    if fingerprint is not None and not isinstance(fingerprint, str):
        return "environment.host.fingerprint is not a string or null"
    fields = doc.get("row_fields") or ["correct"]
    if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
        return "row_fields is not a list of field names"
    if "correct" not in fields:
        return 'row_fields has no "correct"'
    at = fields.index("correct")
    typed = {f: fields.index(f) for f in ("lat_ms", "k", "close") if f in fields}   # what `--deadline` reads
    rows = doc.get("rows") or []
    if not isinstance(rows, list):
        return "rows is not a list"
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not isinstance(r.get("q"), str):
            return f"row {i} has no question id (q)"
        per = r.get("b") or {}
        if not isinstance(per, dict):
            return f"row {i} ({r['q']}): b is not a map of budgets"
        for b, v in per.items():
            if not _int(b):
                return f"row {i} ({r['q']}): budget {b!r} is not an integer"
            if not isinstance(v, list) or len(v) <= at:
                return f"row {i} ({r['q']}): budget {b} has no correct value"
            if (why := _row_value_fault(v, typed, CLOSES)) is not None:
                return f"row {i} ({r['q']}): budget {b} {why}"
        deadlines = r.get("d") or {}                     # a deadline point's rows, in row_fields order
        if not isinstance(deadlines, dict):
            return f"row {i} ({r['q']}): d is not a map of deadlines"
        for d, v in deadlines.items():
            if not _deadline(d):
                return f"row {i} ({r['q']}): deadline {d!r} is not an integer from 1"
            if not isinstance(v, list) or len(v) <= at:
                return f"row {i} ({r['q']}): deadline {d} has no correct value"
            if (why := _row_value_fault(v, typed, DEADLINE_CLOSES)) is not None:
                return f"row {i} ({r['q']}): deadline {d} {why}"
        t = r.get("t")                                   # [n_gen, a_ms, d_ms], what `deadline_map` simulates
        if t is not None and not (isinstance(t, list) and len(t) == 3 and isinstance(t[0], int)
                                  and not isinstance(t[0], bool) and t[0] >= 0 and finite_number(t[1])
                                  and finite_number(t[2]) and t[2] >= 0):
            return f"row {i} ({r['q']}): t is not [n_gen, a_ms, d_ms]"
        rules = r.get("e") or {}                         # an early-exit rule's rows: the fixed pair [correct, lat_ms]
        if not isinstance(rules, dict):
            return f"row {i} ({r['q']}): e is not a map of early-exit rules"
        for z, v in rules.items():
            if not _rule(z):
                return f"row {i} ({r['q']}): early-exit rule {z!r} is not a z in (0, 1)"
            if not isinstance(v, list) or len(v) != 2 or not finite_number(v[1]):
                return f"row {i} ({r['q']}): early-exit rule {z} is not [correct, lat_ms]"
        gated = r.get("g") or {}                         # the same rule behind early exit's gate
        if not isinstance(gated, dict):
            return f"row {i} ({r['q']}): g is not a map of gated early-exit rules"
        for z, v in gated.items():
            if not _rule(z):
                return f"row {i} ({r['q']}): gated early-exit rule {z!r} is not a z in (0, 1)"
            if not isinstance(v, list) or len(v) != 2 or not finite_number(v[1]):
                return f"row {i} ({r['q']}): gated early-exit rule {z} is not [correct, lat_ms]"
    budgets = doc.get("budgets") or {}
    if not isinstance(budgets, dict):
        return "budgets is not a JSON object"
    lat_at = fields.index("lat_ms") if "lat_ms" in fields else None
    for b, e in budgets.items():
        if not _int(b):
            return f"budget {b!r} is not an integer"
        if not isinstance(e, dict):
            return f"budgets.{b} is not a JSON object"
        why = _point_fault(f"budgets.{b}", e) or _stats_fault(f"budgets.{b}", e, rows, "b", b, lat_at)
        if why is not None:
            return why
        if e.get("point") is not None and not isinstance(e["point"], str):
            return f"budgets.{b}.point is not a string"
    for block, ok, what, letter in (("deadlines", _deadline, "an integer from 1", "d"), ("ee", _rule, "a z in (0, 1)", "e"),
                                    ("ee_gated", _rule, "a z in (0, 1)", "g")):
        points = doc.get(block)
        if points is None:
            continue
        if not isinstance(points, dict):
            return f"{block} is not a JSON object"
        for k, e in points.items():
            if not ok(k):
                return f"{block} key {k!r} is not {what}"
            if not isinstance(e, dict):
                return f"{block}.{k} is not a JSON object"
            why = _point_fault(f"{block}.{k}", e) or _stats_fault(f"{block}.{k}", e, rows, letter, k, lat_at)
            if why is not None:
                return why
            if block == "deadlines" and (why := _deadline_fault(f"deadlines.{k}", e)) is not None:
                return why
            cap = e.get("cap")                     # an early-exit point's Bmax cap: absent in older profiles
            if block != "deadlines" and cap is not None and (not isinstance(cap, int) or isinstance(cap, bool)
                                                             or cap < 1):
                return f"{block}.{k}.cap is not an integer >= 1"
    return None


def _row_value_fault(v: list, typed: Mapping[str, int], closes: Sequence[str]) -> str | None:
    """A b / d row's values `--deadline` reads, wherever `row_fields` declares them: `k` an
    integer >= 0, `close` null or one of `closes`, `lat_ms` a finite number >= 0; or None."""
    def val(f: str) -> Any:
        return v[typed[f]] if typed[f] < len(v) else ...

    if "k" in typed and (not isinstance(k := val("k"), int) or isinstance(k, bool) or k < 0):
        return "k is not an integer >= 0"
    if "close" in typed and val("close") is not None and val("close") not in closes:
        return f"close is not null or one of {', '.join(closes)}"
    if "lat_ms" in typed and not (finite_number(lat := val("lat_ms")) and lat >= 0):
        return "lat_ms is not a finite number >= 0"
    return None


def _deadline_fault(where: str, e: dict) -> str | None:
    """A deadline point's own fields, beside `_point_fault`'s: `cap` an integer >= 1; and, each validated
    when present (calibrate always writes them; a hand-made point may lack them, and its reader says so): `reserve_ms`
    a finite number >= 0 (the reserve the cuts were simulated with),
    `closes` an object of the five closes' shares, each finite in [0, 1], and `k_p50` null or a finite number >= 0."""
    cap = e.get("cap")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        return f"{where}.cap is not an integer >= 1"
    r = e.get("reserve_ms")
    if r is not None and not (finite_number(r) and r >= 0):
        return f"{where}.reserve_ms is not a finite number >= 0"
    closes = e.get("closes")
    if closes is not None and not (isinstance(closes, dict) and set(closes) <= set(DEADLINE_CLOSES) and all(
            finite_number(v) and 0 <= v <= 1 for v in closes.values())):
        return f"{where}.closes is not an object of close shares ({', '.join(DEADLINE_CLOSES)}) from 0 to 1"
    k = e.get("k_p50")
    if k is not None and not (finite_number(k) and k >= 0):
        return f"{where}.k_p50 is not null or a finite number >= 0"
    return None


def _deadline(v: Any) -> bool:
    """A deadline key as calibrate writes it: `str(int(D)) == D`, D >= 1 (ms)."""
    return _int(v) and int(v) >= 1


def _rule(v: Any) -> bool:
    """An early-exit rule key as calibrate writes it (`think.ee_key`): z in (0, 1), its shortest decimal."""
    try:
        return isinstance(v, str) and 0 < float(v) < 1 and ee_key(float(v)) == v
    except ValueError:
        return False


def _three(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 3 and all(finite_number(x) for x in v)


def _point_fault(where: str, e: dict) -> str | None:
    """What a stored point (`budgets[B]`, `deadlines[D]`, `ee[z]`) lacks of what `points_of` reads, or None."""
    if not finite_number(e.get("acc")):
        return f"{where}.acc is not a finite number"
    for k in ("lat_p50_ms", "lat_p95_ms"):
        if not _three(e.get(k)):
            return f"{where}.{k} is not three finite numbers (point, lo, hi)"
    if not isinstance(e.get("n"), int) or isinstance(e.get("n"), bool):
        return f"{where}.n is not an integer"
    for k in ("think_mean", "forced"):
        if e.get(k) is not None and not finite_number(e[k]):
            return f"{where}.{k} is not a finite number or null"
    return None


def _stats_fault(where: str, e: dict, rows: list, block: str, key: str, lat_at: int | None) -> str | None:
    """`lat_mean_ms` / `lat_p99_ms`, when present, are three finite numbers and `lat_max_ms` one; when one is
    absent (a profile from before pick preferences) every row holding the point must carry a finite lat_ms (at
    `row_fields.index("lat_ms")`, or index 1 of an early-exit pair, plain `e` or gated `g`) to recompute it from."""
    for k in ("lat_mean_ms", "lat_p99_ms"):
        if e.get(k) is not None and not _three(e[k]):
            return f"{where}.{k} is not three finite numbers (point, lo, hi)"
    if e.get("lat_max_ms") is not None and not finite_number(e["lat_max_ms"]):
        return f"{where}.lat_max_ms is not a finite number"
    absent = [k for k in ("lat_mean_ms", "lat_p99_ms", "lat_max_ms") if e.get(k) is None]
    if not absent:
        return None
    at = 1 if block in ("e", "g") else lat_at
    held = [r[block][key] for r in rows if key in (r.get(block) or {})]
    if at is None or not held or not all(len(v) > at and finite_number(v[at]) for v in held):
        return f"{where}: {absent[0]} absent and rows carry no lat_ms"
    return None


def _env(p: dict, *keys: str) -> Any:
    v: Any = p.get("environment") or {}
    for k in keys:
        v = (v or {}).get(k) if isinstance(v, dict) else None
    return v


def _comp(p: dict, *keys: str) -> Any:
    v: Any = p.get("comparable") or {}
    for k in keys:
        v = (v or {}).get(k) if isinstance(v, dict) else None
    return v


def display_names(profiles: Sequence[dict]) -> list[str]:
    """The names points carry: `model_name`, with `#2`, `#3` when two profiles share one (the same model under two
    samplers, the intended join of Run 1 and Run 3)."""
    counts: dict[str, int] = {}
    out = []
    for p in profiles:
        n = str(p.get("model_name") or "model")
        counts[n] = counts.get(n, 0) + 1
        out.append(n if counts[n] == 1 else f"{n}#{counts[n]}")
    return out


def question_ids(p: dict) -> set[str]:
    return {r["q"] for r in p.get("rows") or [] if isinstance(r, dict) and "q" in r}


def refusals(profiles: Sequence[dict]) -> list[str]:
    """The exit-2 conditions, each naming the profiles that differ from the first."""
    names = display_names(profiles)
    first = profiles[0]
    out = []
    for i, p in enumerate(profiles[1:], 1):
        if _comp(p, "data", "sha256") != _comp(first, "data", "sha256"):
            out.append(f"{names[i]} was measured on different data ({_comp(p, 'data', 'name')}, sha256 "
                       f"{str(_comp(p, 'data', 'sha256'))[:12]}) than {names[0]} ({_comp(first, 'data', 'name')}, "
                       f"{str(_comp(first, 'data', 'sha256'))[:12]}); points on different questions do not compare")
        if p.get("metrics") != first.get("metrics"):
            out.append(f"{names[i]} has metrics version {p.get('metrics')!r}, {names[0]} {first.get('metrics')!r} "
                       f"(this callo computes {METRICS_VERSION}); recalibrate the older one")
        diff = [k for k in READOUT_KEYS if _comp(p, "readout", k) != _comp(first, "readout", k)]
        if diff:
            out.append(f"{names[i]} was read with different readout-engine settings than {names[0]} ({', '.join(diff)})")
        if question_ids(p) != question_ids(first):
            out.append(f"{names[i]} and {names[0]} hold different question id sets ({len(question_ids(p))} vs "
                       f"{len(question_ids(first))} ids)")
    return out


def _strata(p: dict) -> str | None:
    """The profile's bootstrap strata ("source" when its rows carry their source, else None)."""
    return "source" if _comp(p, "data", "strata") == "source" else None


def _held(p: dict) -> set[tuple[str, Any]]:
    """Every (kind, value) the profile holds a stored point for: its budgets, deadlines and early-exit rules, plain
    and gated."""
    return ({("budget", int(b)) for b in (p.get("budgets") or {})} | {("deadline", int(d)) for d in p.get("deadlines") or {}}
            | {("ee", float(z)) for z in p.get("ee") or {}} | {("eeg", float(z)) for z in p.get("ee_gated") or {}})


def pairing(profiles: Sequence[dict]) -> tuple[list[str], list[str], list[str] | None, list[str]]:
    """The ONE row set every paired interval resamples (`think._bootstrap_weights`' contract, as calibrate_think's
    `fixed`): the question ids measured at every point of every profile (every budget, every deadline in `rows[].d`
    and every early-exit rule in `rows[].e`), in the first profile's row order, with their clusters (a row
    without one is its own) and their strata, plus the flags about them. Strata are the first profile's row sources
    when EVERY profile was stratified by source, else None: profiles that disagree, a row without a source, or a
    cluster spanning two sources fall back to plain cluster resampling, flagged, so no ValueError ever escapes the
    bootstrap."""
    rows = [rows_at(p) for p in profiles]
    need = [_held(p) | {k for per in r.values() for k in per} for p, r in zip(profiles, rows, strict=True)]
    first = [r for r in profiles[0].get("rows") or [] if isinstance(r, dict) and "q" in r] if profiles else []
    kept = [r for r in first if all(r["q"] in rs and bs <= rs[r["q"]].keys() for rs, bs in zip(rows, need, strict=True))]
    fixed = [r["q"] for r in kept]
    clusters = [str(r["cluster"]) if r.get("cluster") is not None else str(r["q"]) for r in kept]
    out: list[str] = []
    strata: list[str] | None = None
    kinds = {_strata(p) for p in profiles}
    if len(kinds) > 1:
        out.append("different bootstrap strata (source vs none): paired intervals are unstratified")
    elif kinds == {"source"}:
        sources = [r.get("source") for r in kept]
        of: dict[str, Any] = {}
        if any(not isinstance(x, str) for x in sources):
            out.append("rows without a source: paired intervals are unstratified")
        elif any(of.setdefault(c, x) != x for c, x in zip(clusters, sources, strict=True)):
            out.append("a cluster spans two sources: paired intervals are unstratified")
        else:
            strata = [str(x) for x in sources]
    if not fixed:
        out.append("no question was measured at every point of every model: paired intervals are empty")
    return fixed, clusters, strata, out


def policy_kind(p: dict) -> str:
    policy = str(_comp(p, "sampling", "policy") or "model")
    return policy if policy in ("model", "fixed") else "spec"


def sampler_text(p: dict, *, mark_callo: bool = True) -> str:
    """`temperature/top_p/top_k/min_p`, a `*` on values callo filled in (not the GGUF's, not the user's)."""
    pinned, origin = _comp(p, "sampling", "pinned") or {}, _comp(p, "sampling", "origin") or {}
    parts = []
    for k in TRACE_FIELDS:
        v = pinned.get(k, "-")
        text = f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
        parts.append(text + ("*" if mark_callo and origin.get(k, "callo") == "callo" else ""))
    return "/".join(parts)


def sampler_statement(profiles: Sequence[dict]) -> str:
    """The header sentence about samplers."""
    kinds = {policy_kind(p) for p in profiles}
    if kinds == {"model"}:
        return OWN_SAMPLER
    values = {sampler_text(p, mark_callo=False) for p in profiles}
    if len(kinds) == 1 and len(values) == 1:
        return f"Every model thinks with the same sampler, {values.pop()}."
    names = display_names(profiles)
    return ("policies differ: " + ", ".join(f"{n} {policy_kind(p)}" for n, p in zip(names, profiles, strict=True))
            + ". Each point shows its sampler.")


def flags(profiles: Sequence[dict], pref: Preference | None = None) -> list[str]:
    """The loud flags: printed above the table and kept in `report.json`; compare goes on. Under a price or a
    non-default pick statistic the different-machines flag gains the line: the pick then prices latency."""
    names = display_names(profiles)
    out: list[str] = []

    def differs(getter, what: str, show=lambda v: str(v), *, present_only: bool = False) -> bool:
        """`present_only`: compare only the profiles that carry the value (a field older profiles lack)."""
        pairs = [(n, getter(p)) for n, p in zip(names, profiles, strict=True)]
        if present_only:
            pairs = [(n, v) for n, v in pairs if v is not None]
        if len({json.dumps(v, sort_keys=True, default=str) for _, v in pairs}) > 1:
            out.append(f"{what}: " + ", ".join(f"{n} {show(v)}" for n, v in pairs))
            return True
        return False

    # the host fingerprint is None for an unknown host: shown as "unknown", never counted as a match
    if differs(lambda p: _env(p, "host", "fingerprint"), "latencies from different machines are not comparable",
               lambda v: "unknown" if v is None else str(v)[:19]) \
            and pref is not None and (pref.price > 0 or pref.latency != "mean"):
        out.append("the pick prices latency across hosts")
    unknown = [n for n, p in zip(names, profiles, strict=True) if _env(p, "host", "fingerprint") is None]
    if unknown and len(profiles) > 1:
        out.append("latencies from an unknown host cannot be checked as same-machine: " + ", ".join(unknown))
    differs(lambda p: _env(p, "build_info"), "different llama-server builds")
    differs(lambda p: _comp(p, "readout", "callo_version"), "different callo versions")
    differs(lambda p: _comp(p, "readout", "cache_prompt"), "different --cache (budget 0's latency depends on it)")
    if len({json.dumps(_comp(p, "readout", "cache_prompt")) for p in profiles}) == 1:   # else the --cache flag says it
        differs(lambda p: _comp(p, "readout", "cache_delimiter"), "different --cache readouts (whether they send the "
                "user-turn delimiter)",
                lambda v: "delimited" if v else "undelimited")
    differs(lambda p: _comp(p, "readout", "user_delimiter"), "different user delimiters")
    # a profile written before the field carries none, and never counts as different
    differs(lambda p: _comp(p, "readout", "wide_contract"), "different wide-choice readout contracts",
            lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")), present_only=True)
    differs(lambda p: _comp(p, "seed"), "different seeds")
    # a profile without gated points carries no gate, and never counts as different
    differs(_gate_sha, "gated early exit measured with different gates", lambda v: str(v)[:12], present_only=True)
    differs(lambda p: _comp(p, "warmup"), "different warm-ups")
    differs(lambda p: _env(p, "server", "position_bias_sha256"), "different position-bias profiles (budget 0 only)",
            lambda v: "none" if v is None else str(v)[:19])
    differs(lambda p: _env(p, "server", "server_args"), "different server args (KV types, speculative/MTP, LoRA)",
            lambda v: "none" if not v else " ".join(map(str, v)) if isinstance(v, list) else str(v))
    differs(lambda p: _env(p, "server", "cache_ram"), "different host prompt-cache settings (llama-server --cache-ram; "
            "null: unknown, an attach mode)", lambda v: "unknown" if v is None else str(v))
    out.extend(pairing(profiles)[3])
    for n, p in zip(names, profiles, strict=True):
        if _env(p, "server", "upstream_local") is False:
            out.append(f"{n}: the upstream was not local (latencies include the network)")
        if _env(p, "server", "gpu_busy"):
            out.append(f"{n}: another model server was running during calibration (gpu_busy)")
        if (p.get("server_state") or {}).get("state") == "busy":
            out.append(f"{n}: server state busy (a failed server-state check was recorded under --allow-busy)")
        audit = p.get("audit") or {}
        if audit.get("latency_reason") == "prefill_reused" or _reused_budgets(p):
            out.append(f"{n}: phase-B generations reused a cached state of their own question (llama-server's host "
                       f"prompt cache; prefill_reused {_reused_text(p)}): its latencies are flattered; recalibrate with "
                       f"--cache-ram 0 on the llama-server")
        elif audit.get("latency_model") == "unverified":
            out.append(f"{n}: latency model unverified (the audit's error exceeded 10 %): its cut latencies are estimates")
        if (p.get("drift") or {}).get("flagged"):
            out.append(f"{n}: drift flagged (decode speed or thermal limit changed during the run)")
    by_key: dict[str, list[int]] = {}
    for i, p in enumerate(profiles):
        if p.get("model_key"):
            by_key.setdefault(str(p["model_key"]), []).append(i)
    for idx in by_key.values():
        samplers = {sampler_text(profiles[i], mark_callo=False) for i in idx}
        if len(samplers) > 1:
            where = " and ".join(f"{sampler_text(profiles[i], mark_callo=False)} in {names[i]}" for i in idx)
            out.append(f"the same model ({str(profiles[idx[0]].get('model_key'))[:19]}) sampled at {where}: its points "
                       f"differ by sampler, not by model")
    if len(profiles) > 1 and len({policy_kind(p) for p in profiles}) > 1:
        out.append(sampler_statement(profiles))
    return out


def _gate_sha(p: dict) -> Any:
    """The gate a profile's gated points were measured with (`converge.gate.sha256`), None without gated points."""
    cv = p.get("converge")
    gate = cv.get("gate") if isinstance(cv, dict) and p.get("ee_gated") else None
    return gate.get("sha256") if isinstance(gate, dict) else None


def _reused_budgets(p: dict) -> dict[str, float]:
    """{budget: prefill_reused share} of the budgets above PREFILL_REUSED_MAX."""
    out = {}
    for b, entry in (p.get("budgets") or {}).items():
        share = entry.get("prefill_reused") if isinstance(entry, dict) else None
        if isinstance(share, (int, float)) and not isinstance(share, bool) and share > PREFILL_REUSED_MAX:
            out[b] = float(share)
    return out


def _reused_text(p: dict) -> str:
    return ", ".join(f"{b}: {100 * s:.0f} %" for b, s in sorted(_reused_budgets(p).items(), key=lambda kv: int(kv[0]))) or "flagged"


def notes(profiles: Sequence[dict]) -> list[str]:
    """The notes (not warnings): budget-0-only models, missing thermal data, and a profile measured on
    fewer than SMALL_N questions (its intervals are wide, so neighbouring frontier points will mostly be ≈)."""
    out = []
    for n, p in zip(display_names(profiles), profiles, strict=True):
        questions = _comp(p, "data", "questions")
        if isinstance(questions, int) and not isinstance(questions, bool) and questions < SMALL_N:
            out.append(f"{n}: {questions} questions (< {SMALL_N}): its intervals are wide")
        thinking = p.get("thinking") or {}
        if not thinking.get("capable"):
            why = ("thinks in a format callo cannot read" if thinking.get("reason") == "unsupported_think_format"
                   else "no thinking mode")
            out.append(f"{n}: budget 0 only: {why}")
        therm = (p.get("drift") or {}).get("therm")
        if not therm or any(t.get("limit") is None for t in therm):
            out.append(f"{n}: no thermal data")
    return out


def points_of(p: dict, name: str) -> list[Point]:
    """Every point of the profile, built with keyword arguments: each budget (the profile's own `B≡B'`
    merges kept, one point per `point` label, the smallest budget's latency as in `merge_identical`), each deadline
    point of `deadlines`, each early-exit point of `ee` and each gated one of `ee_gated` (`<model>@ee<z>+g`; an older
    profile has none). The five latency statistics are the stored ones, an absent one recomputed from the rows
    (`think.point_stats`)."""
    budgets = p.get("budgets") or {}
    sampler = sampler_text(p)
    model = _env(p, "model") or {}
    gib = (model["bytes"] + (model.get("mmproj_bytes") or 0)) / 2**30 if isinstance(model.get("bytes"), int) else None
    capable = bool((p.get("thinking") or {}).get("capable"))
    common = {"name": name, "sampler": sampler, "gib": gib, "capable": capable}
    by_label: dict[str, list[int]] = {}
    for b in sorted(budgets, key=int):
        by_label.setdefault(budgets[b].get("point") or f"{p.get('model_name')}@{b}", []).append(int(b))
    out = []
    for bs in by_label.values():
        e = budgets[str(bs[0])]
        out.append(Point(kind="budget", budget=bs[0], budgets=tuple(bs) if len(bs) > 1 else (), acc=float(e["acc"]),
                         n=int(e["n"]), **point_latencies(point_stats(p, e, "b", str(bs[0]))),
                         think_mean=e.get("think_mean"), forced=e.get("forced"), **common))
    for d, e in sorted((p.get("deadlines") or {}).items(), key=lambda kv: int(kv[0])):
        out.append(Point(kind="deadline", value=int(d), cap=int(e["cap"]), acc=float(e["acc"]), n=int(e["n"]),
                         **point_latencies(point_stats(p, e, "d", d)), think_mean=e.get("think_mean"), **common))
    # an early-exit point's Bmax cap; an entry from before `cap` was recorded ran under the largest calibrated budget
    largest = max((int(b) for b in budgets if int(b) > 0), default=None)
    for z, e in sorted((p.get("ee") or {}).items(), key=lambda kv: float(kv[0])):
        cap = e.get("cap") if e.get("cap") is not None else largest
        out.append(Point(kind="ee", value=float(z), cap=cap, acc=float(e["acc"]), n=int(e["n"]),
                         **point_latencies(point_stats(p, e, "e", z)), think_mean=e.get("think_mean"), **common))
    for z, e in sorted((p.get("ee_gated") or {}).items(), key=lambda kv: float(kv[0])):
        cap = e.get("cap") if e.get("cap") is not None else largest
        out.append(Point(kind="eeg", value=float(z), cap=cap, acc=float(e["acc"]), n=int(e["n"]),
                         **point_latencies(point_stats(p, e, "g", z)), think_mean=e.get("think_mean"),
                         forced=e.get("forced"), **common))
    return out


def rows_at(p: dict) -> dict[str, dict[tuple[str, Any], tuple[float, float | None]]]:
    """The per-question getter's source: {question uid: {(kind, value): (1.0 if correct else 0.0, lat_ms)}}
    from the profile's rows, every kind: `b` and `d` read in `row_fields` order (lat_ms None when the rows carry
    none), `e` and `g` (gated) the fixed pair [correct, lat_ms]."""
    fields = list(p.get("row_fields") or ["correct"])
    i = fields.index("correct")
    j = fields.index("lat_ms") if "lat_ms" in fields else None

    def lat(v: list) -> float | None:
        return float(v[j]) if j is not None and len(v) > j and finite_number(v[j]) else None

    out: dict[str, dict[tuple[str, Any], tuple[float, float | None]]] = {}
    for r in p.get("rows") or []:
        per: dict[tuple[str, Any], tuple[float, float | None]] = {}
        per.update({("budget", int(b)): (float(bool(v[i])), lat(v)) for b, v in (r.get("b") or {}).items()})
        per.update({("deadline", int(d)): (float(bool(v[i])), lat(v)) for d, v in (r.get("d") or {}).items()})
        per.update({("ee", float(z)): (float(bool(v[0])), float(v[1])) for z, v in (r.get("e") or {}).items()})
        per.update({("eeg", float(z)): (float(bool(v[0])), float(v[1])) for z, v in (r.get("g") or {}).items()})
        out[r["q"]] = per
    return out


def correct_at(p: dict) -> dict[str, dict[int, float]]:
    """{question uid: {budget: 1.0 if correct else 0.0}}: `rows_at`'s budget correctness (the think bench reads it)."""
    return {u: {v: c for (k, v), (c, _) in per.items() if k == "budget"} for u, per in rows_at(p).items()}


@dataclass(frozen=True)
class Joint:
    """The joint frontier: every point ranked, the frontier with `≈` marks, the pick (made over EVERY point with the
    filters applied first, `pick`), its paired Δacc against M (`dacc_best`), the paired Δacc intervals
    (`pairs`, with a Δscore at R > 0), the preference it picked under, and what the header says."""
    profiles: tuple[dict, ...]
    names: tuple[str, ...]
    latency: str
    ranked: tuple[Point, ...]
    front: tuple[Point, ...]
    chosen: Pick
    pairs: tuple[dict, ...]
    filters: dict
    dacc_best: tuple[float, float, float] | None = None    # `chosen.dacc_best`: the pick minus M (point − M)
    fixed: tuple[str, ...] = ()                             # the one row set every pair resamples (`pairing`)
    pref: Preference = Preference()

    def by_label(self, label: str) -> Point | None:
        return next((p for p in self.ranked if p.label == label), None)


def _paired(rows: Mapping[str, Mapping[str, Mapping[tuple[str, Any], tuple[float, float | None]]]], fixed: Sequence[str],
            clusters: Sequence[str], strata: Sequence[str] | None, seed: int) -> Paired:
    """The paired intervals over `pairing`'s ONE row set: each point's per-question (correct, lat_ms) from `rows_at`,
    by its model name."""
    def point_rows(x: Point) -> tuple[list[float], list[float]]:
        r, key = rows[x.name], (x.kind, x.value)          # a merged point's value is its smallest budget
        return [r[u][key][0] for u in fixed], [r[u][key][1] or 0.0 for u in fixed]

    return Paired(point_rows, tuple(clusters), seed, tuple(strata) if strata is not None else None)


def frontier_marks(p: dict, name: str, *, latency: str = "mean") -> dict[str, Point]:
    """One profile's points of every kind (`points_of`: budgets,
    deadline points, early exit) on compare's one frontier, the `≈` marks over compare's row set (`pairing`), by
    label: the display marks calibrate's summary prints, so it agrees with `callo compare` and with the pick, which
    can pick a deadline point. The profile's stored marks are not touched: they stay budget-only."""
    fixed, clusters, strata, _ = pairing([p])
    paired = _paired({name: rows_at(p)}, fixed, clusters, strata, int(_comp(p, "seed") or 0))
    ranked = frontier(points_of(p, name), latency=latency)
    front = {q.label: q for q in mark_approx(on_frontier(ranked), paired.dacc)}
    return {q.label: front.get(q.label, q) for q in ranked}


def joint(profiles: Sequence[dict], *, pref: Preference | None = None) -> Joint:
    """With the pick preference: the points of every profile (budgets, deadlines, early-exit rules) on one frontier
    (`think.frontier`, `mark_approx`, `pick`), with paired cluster bootstraps over ONE row set (`pairing`) for the `≈`
    marks, the pick and the reported pairs. The seed is the first profile's (a differing seed is flagged, not refused).
    `dacc_best` is the pick's own (`Pick.dacc_best`, point − M). Under a positive price every point's
    rows must carry lat_ms (the per-question scores need it)."""
    pref = pref if pref is not None else Preference()
    if not profiles:
        raise InvalidInput("compare needs at least one think profile")
    bad = refusals(profiles)
    if bad:
        raise InvalidInput("refusing to compare: " + "; ".join(bad))
    names = display_names(profiles)
    seed = int(_comp(profiles[0], "seed") or 0)
    rows = {n: rows_at(p) for n, p in zip(names, profiles, strict=True)}
    # ONE row set in ONE order for every pair (the pairing contract): the questions measured at every point of
    # every profile, stratified by source when every profile was
    fixed, clusters, strata, _ = pairing(profiles)
    points = [pt for n, p in zip(names, profiles, strict=True) for pt in points_of(p, n)]
    if pref.price > 0:
        for pt in points:
            if any(rows[pt.name][u][(pt.kind, pt.value)][1] is None for u in fixed):
                raise InvalidInput(f"--price needs every question's latency: {pt.label}'s rows carry no lat_ms")

    paired = _paired(rows, fixed, clusters, strata, seed)
    ranked = frontier(points, latency=pref.latency)
    front = mark_approx(on_frontier(ranked), paired.dacc)
    # the pick takes EVERY point: the serve check (each profile's convergence block against its early-exit points),
    # early exit (opt-in, never picked), the limits and `--max-memory` apply before its own frontier, so a point that
    # fails one can never eliminate one that passes
    skipped = {n: {int(d) for d in p["deadline_skipped"] if _deadline(d)}    # never hint a skipped one again
               for n, p in zip(names, profiles, strict=True) if isinstance(p.get("deadline_skipped"), dict)}
    unserved = {label: why for n, p in zip(names, profiles, strict=True)
                for label, why in unservable([pt for pt in points if pt.name == n], p.get("converge")).items()}
    chosen = pick(ranked, paired=paired, pref=pref, skipped=skipped, unservable=unserved)
    best = chosen.best or (max(front, key=lambda q: (q.acc, -q.latency(pref.latency)[0])) if front else None)
    pairs: list[dict] = []
    seen: set[frozenset[str]] = set()

    # every pair's `dacc` is b − a (`paired_bootstrap`'s mean(b − a)). Frontier neighbours are (faster, slower): the
    # slower point minus its faster neighbour, as the profile's `dacc_prev`. The rest are (M, point): the point minus
    # M, as `Pick`'s `dacc_best` (at R = 0 M is the most accurate, the profile's `dacc_best`). A pair is recorded once,
    # in whichever orientation reached it first (a point that neighbours M keeps its neighbour orientation). At R > 0
    # each pair also carries its Δscore, descriptive.
    def add(a: Point, b: Point, why: str) -> None:
        key = frozenset((a.label, b.label))
        if a.label == b.label or key in seen:
            return
        seen.add(key)
        pair = {"a": a.label, "b": b.label, "n": len(fixed), "dacc": list(paired.dacc(a, b)), "why": why}
        if pref.price > 0:
            pair["dscore"] = list(paired.dscore(a, b, stat=pref.latency, price=pref.price))
        pairs.append(pair)

    for i in range(1, len(front)):
        add(front[i - 1], front[i], "frontier neighbours")
    if best is not None:
        for q in ranked:
            add(best, q, "against the best" if pref.price > 0 else "against the most accurate")
    return Joint(tuple(profiles), tuple(names), pref.latency, tuple(ranked), tuple(front), chosen, tuple(pairs),
                 {**pref.as_dict(), "max_latency_s": pref.max_latency_s}, chosen.dacc_best, tuple(fixed), pref)


# ---- the report: report.md with the table, the pairs, the pick and the ASCII scatter; report.json; the CLI ----

SCATTER_W, SCATTER_H = 56, 13          # odd: the middle row is exactly (hi + lo) / 2
MEMORY_LABEL = "file size: a lower bound on memory"


def _cell(v: Any, spec: str) -> str:
    """A metric cell: `-` for a null (a non-finite metric is null in the strict-JSON profile) or a missing one."""
    return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"


def _mark(p: Point, ee_labels: Collection[str] = ()) -> str:
    """`F`, `F≈` or `· by <point>` (`<point> (opt-in)` for an early-exit dominator), then `; opt-in (early exit)`
    on an early-exit row, plain or gated: the pick never takes early exit, so it can be a row an early-exit point
    dominates."""
    if p.dominated_by is not None:
        mark = f"· by {opt_in_label(p.dominated_by, ee_labels)}"
    else:
        mark = "F≈" if p.approx else "F"
    return f"{mark}; {EE_MARK}" if p.kind in EE_KINDS else mark


def _entry(j: Joint, p: Point) -> dict:
    """The profile's stored entry of a point: `budgets[B]`, `deadlines[D]`, `ee[z]` or `ee_gated[z]`."""
    prof = j.profiles[j.names.index(p.name)]
    if p.kind == "deadline":
        return (prof.get("deadlines") or {}).get(str(int(p.value or 0))) or {}
    if p.kind in EE_KINDS:
        return (prof.get("ee" if p.kind == "ee" else "ee_gated") or {}).get(ee_key(float(p.value or 0.0))) or {}
    return (prof.get("budgets") or {}).get(str(p.budget)) or {}


def table_lines(j: Joint, *, verbose: bool = False) -> list[str]:
    """One markdown row per point, sorted by latency: `model@B` (`model@Dms`, `model@ee<z>`, `model@ee<z>+g`), acc, the
    five latency statistics (the pick statistic's with its interval, p99 `= max` below 100 questions), think mean,
    forced %, sampler (`*` on callo's values), model GiB, the mark (`_mark`: `F`, `F≈`, `· by <point>`, early exit
    opt-in); `--verbose` adds NLL@T, ECE10@T, T. Then one tails line per question count."""
    head = ["point", "acc", *(f"{st} [95% CI]" if st == j.latency else st for st in STAT_FIELDS), "think", "forced",
            "sampler", "GiB", "mark"]
    if verbose:
        head += ["NLL@T", "ECE10@T", "T"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    front = {p.label: p for p in j.front}
    ee_labels = {p.label for p in j.ranked if p.kind in EE_KINDS}
    for p in j.ranked:
        q = front.get(p.label, p)
        row = [q.label, f"{q.acc:.3f}", *stat_cells(q.stats(), q.n, j.latency).values(),
               "-" if q.think_mean is None else f"{q.think_mean:.0f}",
               "-" if q.forced is None else f"{100 * q.forced:.0f}%", q.sampler or "-",
               "-" if q.gib is None else f"{q.gib:.1f}", _mark(q, ee_labels)]
        if verbose:
            e = _entry(j, q)
            row += [_cell(e.get("nll_T"), ".3f"), _cell(e.get("ece_T"), ".3f"), _cell(e.get("T"), ".2f")]
        lines.append("| " + " | ".join(row) + " |")
    return lines + ["", *(tails_line(n) for n in dict.fromkeys(p.n for p in j.ranked))]


def scatter_lines(points: Sequence[Point], *, latency: str = "mean", width: int = SCATTER_W,
                  height: int = SCATTER_H) -> list[str]:
    """An ASCII scatter of accuracy against latency on a log scale, so the frontier is visible in a terminal.
    Points are lettered in latency order; `*` marks a cell holding more than one point; the legend follows, an
    early-exit point's entry, plain or gated, marked `(opt-in)` (listed, never picked)."""
    if height < 3:
        raise ValueError(f"scatter height must be at least 3 (top, middle and bottom labels), got {height}")
    if not points:
        return ["(no points)"]
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pts = sorted(points, key=lambda p: (p.latency(latency)[0], p.label))
    xs = [math.log10(max(p.latency(latency)[0], 1.0)) for p in pts]
    x0, x1 = min(xs), max(xs)
    lo = math.floor(min(p.acc for p in pts) * 10) / 10
    hi = math.ceil(max(p.acc for p in pts) * 10) / 10
    if hi - lo < 0.1:
        hi = min(1.0, lo + 0.1) if lo < 1.0 else 1.0
        lo = hi - 0.1
    grid = [[" "] * width for _ in range(height)]
    legend = []
    for i, (p, x) in enumerate(zip(pts, xs, strict=True)):
        col = 0 if x1 == x0 else round((x - x0) / (x1 - x0) * (width - 1))
        row = round((hi - p.acc) / (hi - lo) * (height - 1))
        row = min(height - 1, max(0, row))
        letter = letters[i] if i < len(letters) else "+"
        grid[row][col] = letter if grid[row][col] == " " else "*"
        legend.append(f"{letter} {p.label}" + (" (opt-in)" if p.kind in EE_KINDS else ""))
    out = []
    for r, cells in enumerate(grid):
        label = ""                       # a labelled row's own value: a point at exactly it is drawn on that row
        if r in (0, (height - 1) // 2, height - 1):
            label = f"{hi - r * (hi - lo) / (height - 1):.2f}"
        out.append(f"{label:>5} |{''.join(cells)}".rstrip())
    out.append("      +" + "-" * width)
    left, right = _s(10 ** x0), _s(10 ** x1)
    out.append(f"       {left}{' ' * max(1, width - len(left) - len(right))}{right}   (acc vs {latency} latency, log scale)")
    out += ["  ".join(legend[k:k + 4]) for k in range(0, len(legend), 4)]
    if any("*" in row for row in grid):
        out.append("* more than one point in that cell")
    return out


def serve_line(j: Joint, sources: Sequence[str] | None = None) -> str | None:
    """The pick's `callo serve ...` command (calibrate's own `serve_command`: `--model FILE` or `--upstream URL`,
    `--think-max`, `--no-cache`) for the PICK's profile and the pick's budget (a deadline pick: `--think-deadline
    D`; the pick is never early exit), or None without a pick."""
    p = j.chosen.point
    if p is None:
        return None
    i = j.names.index(p.name)
    prof = j.profiles[i]
    where = sources[i] if sources is not None and i < len(sources) and sources[i] else f"{prof.get('model_name')}.think.json"
    d = j.chosen.as_dict()
    if d["kind"] == "deadline":
        return serve_command(prof, None, where, deadline_ms=d["deadline_ms"], cap=d["cap"])
    return serve_command(prof, d["budget"], where)


def header_lines(j: Joint) -> list[str]:
    """The latency line (the host: its chip, "N different hosts", or "unknown host" when any fingerprint is None,
    which is never a match) and the sampler sentence."""
    first = j.profiles[0]
    host = _env(first, "host") or {}
    data = _comp(first, "data") or {}
    hosts = [_env(p, "host", "fingerprint") for p in j.profiles]
    if any(h is None for h in hosts):
        where = "unknown host"
    elif len(set(hosts)) == 1:
        where = host.get("chip") or host.get("machine") or "?"
    else:
        where = f"{len(set(hosts))} different hosts"
    return [LATENCY_LINE.format(host=where) + f" ({len(j.profiles)} profile{'s' if len(j.profiles) != 1 else ''}, "
            f"{data.get('questions')} questions, data {data.get('name')})", sampler_statement(list(j.profiles))]


def pair_lines(j: Joint) -> list[str]:
    """One line per pair; at R > 0 its Δscore, which on p95 also says how few questions the quantile rests on
    (`(p95 from 19 of 376)`)."""
    def tail(x: dict) -> str:
        return f" (p95 from {tail_counts(x['n'])['p95']} of {x['n']})" if j.latency == "p95" and x["n"] else ""

    return [f"- {x['a']} → {x['b']}: {_ci(x['dacc'], pct=True)}"
            + (f", Δscore {_pp(x['dscore'])}{tail(x)}" if x.get("dscore") is not None else "")
            + f" (n={x['n']}, {x['why']})" for x in j.pairs]


def repick_lines(j: Joint) -> list[str]:
    """Compare re-picks and never writes back; when a profile's own pick used another statistic, one line
    says so (`the profile picked tiel@1024 on p50; this pick is on mean`)."""
    out = []
    for n, p in zip(j.names, j.profiles, strict=True):
        old = p.get("pick") if isinstance(p.get("pick"), dict) else {}
        if old.get("latency") is not None and old["latency"] != j.latency:
            out.append((f"{n}: " if len(j.profiles) > 1 else "") + f"the profile picked {old.get('label') or 'none'} on "
                       f"{old['latency']}; this pick is on {j.latency}")
    return out


RUNS_HEAD = ["model", "launched", "-np", "kv", "ctx", "cache-ram", "server args", "started", "ended", "decode tok/s"]


def _run_line(r: Mapping[str, Any]) -> str:
    return f"- {r.get('name')}: {r.get('status')}" + (f" ({r.get('reason')})" if r.get("reason") else "")


UNFINISHED = ("pending", "running")


def _run_sections(runs: Sequence[Mapping[str, Any]] | None) -> list[str]:
    """The run's records without a profile: failed and skipped models with their reasons, then the ones an
    interrupted run left pending or running, under their own heading."""
    out: list[str] = []
    ended = [r for r in runs or [] if r.get("status") not in ("done", *UNFINISHED)]
    open_ = [r for r in runs or [] if r.get("status") in UNFINISHED]
    if ended:
        out += ["", "Failed and skipped models:", *(_run_line(r) for r in ended)]
    if open_:
        out += ["", "Not finished (interrupted run):", *(_run_line(r) for r in open_)]
    return out


def runs_lines(j: Joint, runs: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    """One row per profile with its launch shape and timing (a box that heats up over a menu run shows
    here), then the run's own records that have no profile: the failed and skipped models with their reasons."""
    out = ["| " + " | ".join(RUNS_HEAD) + " |", "|" + "---|" * len(RUNS_HEAD)]
    for n, p in zip(j.names, j.profiles, strict=True):
        args = _env(p, "server", "server_args")
        cache_ram = _env(p, "server", "cache_ram")
        decode = _env(p, "run", "decode_tok_s_median")
        out.append("| " + " | ".join([
            n, str(_env(p, "server", "launched_by") or "?"), str(_env(p, "server", "total_slots") or "?"),
            str(_env(p, "server", "kv") or "?"), str(_env(p, "server", "n_ctx") or "?"),
            "unknown" if cache_ram is None else str(cache_ram),
            "none" if not args else (" ".join(map(str, args)) if isinstance(args, list) else str(args)),
            str(_env(p, "run", "started") or "?"), str(_env(p, "run", "ended") or "?"),
            "-" if not isinstance(decode, (int, float)) else f"{decode:.0f}"]) + " |")
    return out + _run_sections(runs)


NO_THINKING = "no calibrated thinking budget (budget 0 only): a deadline changes nothing calibrate measured"


def deadline_mappings(j: Joint, deadlines: Sequence[int], *, budget: int | None = None,
                      reserve_ms: float | None = None) -> list[dict]:
    """One mapping per (profile, deadline), in profile order, each simulated with its profile's seed. A
    budget-0-only profile gets a `skipped` entry instead (a deadline changes nothing calibrate measured); a profile whose
    rows carry no lat_ms, k or close cannot be mapped (InvalidInput, exit 2)."""
    out: list[dict] = []
    for n, p in zip(j.names, j.profiles, strict=True):
        if not bmax_of(p):
            out += [{"model": n, "deadline_ms": d, "budget": budget, "skipped": NO_THINKING} for d in deadlines]
            continue
        if deadlines and {"lat_ms", "k", "close"} - set(p.get("row_fields") or ["correct"]):
            raise InvalidInput(f"--deadline needs each row's lat_ms, k and close (row_fields): {n}'s rows carry none")
        out += [{"model": n, **mapping(p, d, budget=budget, reserve_ms=reserve_ms, seed=int(_comp(p, "seed") or 0))}
                for d in deadlines]
    return out


def deadline_lines(j: Joint, mappings: Sequence[Mapping[str, Any]], sources: Sequence[str] | None = None) -> list[str]:
    """The block per mapping, each with its `callo serve ... --think-deadline D` command (and `--think-budget B`
    for a combined setting)."""
    out: list[str] = []
    for m in mappings:
        i = j.names.index(m["model"])
        prof = j.profiles[i]
        where = sources[i] if sources is not None and i < len(sources) and sources[i] else \
            f"{prof.get('model_name')}.think.json"
        if m.get("skipped"):
            out += [f"{m['model']}  deadline {m['deadline_ms']} ms: {m['skipped']}", ""]
            continue
        serve = serve_command(prof, m["budget"], where, deadline_ms=m["deadline_ms"],
                              cap=point_cap(prof, m["deadline_ms"]))
        out += mapping_lines(m["model"], m, serve)
        skipped = prof.get("deadline_skipped")                        # calibrate measured D and made no point: why
        why = skipped.get(str(m["deadline_ms"])) if isinstance(skipped, dict) else None
        out += ([f"  calibrate: {why}"] if why else []) + [""]
    return out


def early_exit(j: Joint) -> dict:
    """Per profile with a `converge` block: K, the rules, the gain verdicts, the rho map and the opt-in line
    (`opt_in_line`; null without a servable early-exit point); with the gate measured, its block (`gate`) and the
    gated points' entries (`gated`), each only when the profile has it."""
    out = {}
    for n, p in zip(j.names, j.profiles, strict=True):
        cv = p.get("converge")
        if not isinstance(cv, dict):
            continue
        out[n] = {**{k: cv.get(k) for k in ("K", "cost", "rules", "gain", "rho", "rho_max_per_token")},
                  "opt_in": opt_in_line(p, n, latency=j.latency)}
        if cv.get("gate") is not None:
            out[n]["gate"] = cv["gate"]
        if p.get("ee_gated"):
            out[n]["gated"] = p["ee_gated"]
    return out


def render_md(j: Joint, *, verbose: bool = False, sources: Sequence[str] | None = None,
              runs: Sequence[Mapping[str, Any]] | None = None, mappings: Sequence[Mapping[str, Any]] = ()) -> str:
    """`report.md`: the header line, the flags, the table, the preference, the paired intervals (with a
    Δscore at R > 0), calibrate's pick line (`pick_line`: the preference, M and its score, the pick's Δacc against M)
    or the no-pick reason, the points `--max-memory` excluded for an unknown size (`pick_note`), the early-exit
    points `callo serve` cannot run as measured (`serve_note`), one opt-in line per profile with a servable early-exit
    point (`opt_in_line`: early exit is listed, never picked), the re-pick line, the `callo serve` line, the ASCII
    scatter, and the Runs section (launch shape, timing, the failed and skipped models)."""
    profiles = list(j.profiles)
    fl, nt = flags(profiles, j.pref), notes(profiles)
    out = ["# callo compare: the joint think frontier", "", *header_lines(j), ""]
    if fl:
        out += ["**FLAGS** (the points may not be comparable):", *(f"- {f}" for f in fl), ""]
    if nt:
        out += ["Notes:", *(f"- {n}" for n in nt), ""]
    out += table_lines(j, verbose=verbose)
    text = j.pref.text()
    rests = limit_lines(j.pref, [p.n for p in j.ranked])      # what each limit rests on, at every profile's n
    out += ["", "Marks: F frontier, F≈ frontier but within noise of its faster neighbour, · dominated. Pick statistic "
            f"{j.latency}" + (f"; preference: {text}" if text else "") + "".join(f"; {r}" for r in rests)
            + (f" (--max-memory compares the {MEMORY_LABEL})" if j.pref.max_gib is not None else "") + ".", ""]
    if j.pairs:
        head = "Paired Δacc (b − a) over the shared questions [95 % cluster-bootstrap CI]"
        if j.pref.price > 0:
            head += f", and Δscore at --price {_num_text(j.pref.price)}pp/s on {j.latency} (pp, descriptive)"
        out += [head + ":", *pair_lines(j), ""]
    pk = j.chosen.as_dict()
    note = pick_note(pk)
    out.append(pick_line(pk, bold=True))
    if note:
        out.append(note)
    if (unserved := serve_note(pk)) is not None:
        out.append(unserved)
    out += [line for n, p in zip(j.names, profiles, strict=True)
            if (line := opt_in_line(p, n, latency=j.latency)) is not None]
    out += repick_lines(j)
    if j.chosen.point is not None:
        out.append(f"serve with: `{serve_line(j, sources)}`")
    out += ["", "```", *scatter_lines(list(j.ranked), latency=j.latency), "```", ""]
    if mappings:                                                          # compare --deadline
        out += ["## Deadlines", "", "```", *deadline_lines(j, mappings, sources), "```", ""]
    early = [(n, converge_lines(p)) for n, p in zip(j.names, j.profiles, strict=True)
             if isinstance(p.get("converge"), dict)]
    if early:                                                             # the gain verdicts, per model
        out += ["## Early exit", ""]
        for n, lines in early:
            out += [f"{n}:", "```", *lines, "```", ""]
    out += ["## Runs", "", *runs_lines(j, runs), ""]
    return "\n".join(out)


DACC_SIGN = ("point − reference: pairs[].dacc (and dscore) is b − a; pick.dacc_best (and dscore_best) is pick − best (M, "
             "the best-scoring point: the most accurate at --price 0); a profile's budgets[B].dacc_best is against its "
             "most accurate point")


def report_dict(j: Joint, *, sources: Sequence[str] | None = None, runs: Sequence[Mapping[str, Any]] | None = None,
                mappings: Sequence[Mapping[str, Any]] = ()) -> dict:
    """`report.json`: the same as `report.md`, for machines (`runs`: the menu's per-model records)."""
    profiles = list(j.profiles)
    front = {p.label: p for p in j.front}
    return {
        "callo_compare": REPORT_FORMAT, "dacc": DACC_SIGN, "metrics": profiles[0].get("metrics"),
        "header": header_lines(j),
        "latency": j.latency, "filters": dict(j.filters), "flags": flags(profiles, j.pref), "notes": notes(profiles),
        "profiles": [{"name": n, "path": (sources[i] if sources is not None and i < len(sources) else None),
                      "model_key": p.get("model_key"), "model_file": (_env(p, "model") or {}).get("file"),
                      "created": p.get("created"), "policy": _comp(p, "sampling", "policy"), "sampler": sampler_text(p),
                      "host_fingerprint": _env(p, "host", "fingerprint"), "launched_by": _env(p, "server", "launched_by"),
                      "server_args": _env(p, "server", "server_args"), "kv": _env(p, "server", "kv"),
                      "n_ctx": _env(p, "server", "n_ctx"), "s1_slots": _env(p, "server", "s1_slots"),
                      "total_slots": _env(p, "server", "total_slots"), "cache_ram": _env(p, "server", "cache_ram"),
                      "started": _env(p, "run", "started"), "ended": _env(p, "run", "ended"),
                      "decode_tok_s_median": _env(p, "run", "decode_tok_s_median"),
                      "capable": bool((p.get("thinking") or {}).get("capable")),
                      "latency_model": (p.get("audit") or {}).get("latency_model"),
                      "latency_reason": (p.get("audit") or {}).get("latency_reason"),
                      "prefill_reused": {b: (e.get("prefill_reused") if isinstance(e, dict) else None)
                                         for b, e in (p.get("budgets") or {}).items() if int(b) > 0},
                      "drift_flagged": bool((p.get("drift") or {}).get("flagged"))}
                     for i, (n, p) in enumerate(zip(j.names, profiles, strict=True))],
        "runs": [dict(r) for r in runs or []],
        "points": [{"label": p.label, "model": p.name, "kind": p.kind, "value": p.value, "budget": p.budget,
                    "budgets": list(p.budgets or (p.budget,)) if p.kind == "budget" else [], "cap": p.cap,
                    "acc": p.acc, **p.stats(), "tail_n": p.tail_n, "n": p.n,
                    "think_mean": p.think_mean, "forced": p.forced, "sampler": p.sampler, "gib": p.gib,
                    "capable": p.capable, "frontier": p.dominated_by is None, "dominated_by": p.dominated_by,
                    "approx": front[p.label].approx if p.label in front else False}
                   for p in j.ranked],
        "pairs": [dict(x) for x in j.pairs], "repick": repick_lines(j),
        "pick": {**j.chosen.as_dict(), "serve": serve_line(j, sources),
                 "dacc_best": list(j.dacc_best) if j.dacc_best is not None else None},
        "deadlines": [dict(m) for m in mappings], "early_exit": early_exit(j)}


def write_report(profiles: Sequence[dict], out_dir: str | Path, *, pref: Preference | None = None, verbose: bool = False,
                 sources: Sequence[str] | None = None, runs: Sequence[Mapping[str, Any]] | None = None) -> tuple[Path, Path]:
    """`DIR/report.md` and `DIR/report.json` from the profiles under the whole preference (a menu run's last step,
    with its `runs` records; a one-server run's too)."""
    j = joint(profiles, pref=pref)
    return write_joint(j, out_dir, verbose=verbose, sources=sources, runs=runs)


def write_joint(j: Joint, out_dir: str | Path, *, verbose: bool = False, sources: Sequence[str] | None = None,
                runs: Sequence[Mapping[str, Any]] | None = None,
                mappings: Sequence[Mapping[str, Any]] = ()) -> tuple[Path, Path]:
    """`write_report` for a `Joint` already computed (`callo compare --out`: one bootstrap pass for stdout and files)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md, js = out / "report.md", out / "report.json"
    md.write_text(render_md(j, verbose=verbose, sources=sources, runs=runs, mappings=mappings), encoding="utf-8")
    js.write_text(json.dumps(report_dict(j, sources=sources, runs=runs, mappings=mappings), indent=1) + "\n",
                  encoding="utf-8")
    return md, js


def write_stub_report(out_dir: str | Path, *, runs: Sequence[Mapping[str, Any]], reason: str | None = None) -> Path:
    """A menu run that produced no profile still names its failed and skipped models with their reasons. With
    `reason` (a joint report compare refused, or a done profile that does not load), that reason is the report's
    first line and `report.json`'s `error`, so no earlier run's report survives in DIR."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = out / "report.md"
    first = f"No joint report: {reason}" if reason else "No profile was written: nothing to compare."
    lines = ["# callo compare: the joint think frontier", "", first, "", "## Runs", *_run_sections(runs), ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    doc = {"callo_compare": REPORT_FORMAT, "profiles": [], "points": [], "pairs": [], "pick": None,
           "runs": [dict(r) for r in runs]}
    if reason:
        doc["error"] = reason
    (out / "report.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return md


def run(args: Any, *, out: Any = None, err: Any = None) -> int:
    """`callo compare P1.think.json P2.think.json ...`: 0 with the report printed (markdown, or `report.json` with
    `--json`) and, with `--out DIR`, written; 2 for an unreadable or malformed profile, a refusal, a usage error or an
    `--out` that cannot be written."""
    out, err = out or sys.stdout, err or sys.stderr
    try:
        pref = preference_from_args(args)              # the validator calibrate shares, before any file is read
        deadlines = parse_deadlines(getattr(args, "deadline", None), flag="--deadline")   # MS[,...]
        budget, reserve = getattr(args, "budget", None), getattr(args, "reserve", None)
        if (budget is not None or reserve is not None) and not deadlines:
            raise InvalidInput("--budget and --reserve shape a --deadline setting: they need --deadline")
        if budget is not None and budget < 1:
            raise InvalidInput(f"--budget must be at least 1, got {budget}")
        if reserve is not None and not (math.isfinite(reserve) and reserve >= 0):
            raise InvalidInput(f"--reserve is in ms and must be a finite number >= 0, got {reserve!r}")
        profiles = [load_profile(p) for p in args.profiles]
        j = joint(profiles, pref=pref)
        mappings = deadline_mappings(j, deadlines, budget=budget, reserve_ms=reserve)
    except InvalidInput as e:
        print(f"callo: {e}", file=err)
        return 2
    sources = [str(p) for p in args.profiles]
    if args.out:
        try:
            md, js = write_joint(j, args.out, verbose=args.verbose, sources=sources, mappings=mappings)
        except OSError as e:
            print(f"callo: --out {args.out}: cannot write the report there ({e.strerror or e})", file=err)
            return 2
        print(f"report: {md} and {js}", file=err)
    if args.json:
        print(json.dumps(report_dict(j, sources=sources, mappings=mappings), indent=1), file=out)
    else:
        print(render_md(j, verbose=args.verbose, sources=sources, mappings=mappings), file=out, end="")
    return 0
