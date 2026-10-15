"""callo's companion figure: decisions while an agent works, from EXISTING results.

    python brand/concurrency.py build --src <bench>/.../plate/concurrency.json --root <bench>
    python brand/concurrency.py draw [--png]                 # -> brand/concurrency-{dark,light}.svg

`build` takes the benchmark's `harness/callo_plate/extract_concurrency.py` output, re-reads every number from its
source file and JSON pointer under `--root` (a number that does not round-trip is refused, exit 2), quotes the README's
limit on the S1 fallback, composes the fine print and writes the committed snapshot `brand/data/concurrency.json`,
which records paths relative to `--root` (a file outside it by its name) and callo's builds by role
(`data/builds.json`); `draw` needs only the snapshot.

One row per condition: the decision's p50 (dot), p95 (tail) and max (tick) on a log ms axis, and beside it the
agent's throughput. The pathological `s1_fallback` case (a foreign prefill inside the reserved slot) is fine print
only, with its numbers, for every series: never plotted, never omitted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

SNAPSHOT_FORMAT = 1
README_LIMIT = ("When a reserved slot is busy with a request Callo did not send, Callo moves the classification to "
               "the lowest-id idle slot (the fallback). That helps against a foreign decode, not a foreign prefill")


class Refusal(Exception):
    pass


def get(doc, ptr: str):
    v = doc
    for raw in ptr.lstrip("/").split("/") if ptr else []:
        k = raw.replace("~1", "/").replace("~0", "~")
        v = v[int(k)] if isinstance(v, list) else v[k]
    return v


def numbers(node, out=None, path=""):
    """Every {"value", "src", "ptr"} number in the document, with where it sits."""
    out = [] if out is None else out
    if isinstance(node, dict):
        if {"value", "src", "ptr"} <= node.keys():
            out.append((path, node))
        else:
            for k, v in node.items():
                numbers(v, out, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            key = v.get("id") or v.get("name") if isinstance(v, dict) else None
            numbers(v, out, f"{path}.{key or i}")
    return out


def verify(doc: dict, root: Path) -> int:
    """Re-read every number from its source; a computed one ("via") must at least name an existing source."""
    cache: dict[str, object] = {}
    n = 0
    for where, v in numbers(doc):
        p = Path(v["src"]) if Path(v["src"]).is_absolute() else root / v["src"]
        if not p.is_file():
            raise Refusal(f"{where}: source {v['src']} not found under {root}")
        if str(p) not in cache:
            cache[str(p)] = json.loads(p.read_text(encoding="utf-8"))
        if "via" in v:
            get(cache[str(p)], v["ptr"])
            continue
        try:
            got = get(cache[str(p)], v["ptr"])
        except (KeyError, IndexError, ValueError, TypeError):
            raise Refusal(f"{where}: {v['src']}#{v['ptr']} does not exist") from None
        if got != v["value"]:
            raise Refusal(f"{where}: {v['src']}#{v['ptr']} is {got!r}, the figure data says {v['value']!r}")
        n += 1
    return n


def readme_limit(readme: Path = REPO / "README.md") -> str:
    """The README's own bullet on the S1 fallback, as the release ships it (quoted, never paraphrased): from
    `README_LIMIT` (which may wrap across lines) to the end of its bullet, without the bullet's bold lead."""
    text = readme.read_text(encoding="utf-8")
    m = re.search(r"\s+".join(map(re.escape, README_LIMIT.split())), text)
    if m is None:
        raise Refusal(f"README.md no longer states {README_LIMIT!r}: re-check the limit before quoting it")
    bullet = re.split(r"\n(?:- |\n)", text[m.start():], maxsplit=1)[0]
    return re.sub(r"\s+", " ", bullet.replace("**", "")).strip()


def _s(v: dict) -> float:
    return v["value"] / 1000


def fallbacks(doc: dict) -> list[dict]:
    """Every measured S1-fallback case, in order: each plotted series' own, then each model measured for that case
    only (`fallback_only`, never plotted, so it carries its own provenance)."""
    out = [{"short": _short(s), "fb": s["fallback"], "own": None, "date": s.get("date")}
           for s in doc["series"] if s.get("fallback")]
    out += [{"short": e["label"], "fb": e["fallback"], "own": e} for e in doc.get("fallback_only") or []]
    return out


def fallback_sentence(e: dict) -> str:
    """One model's fallback case in full, for FIGURES.md."""
    fb, own = e["fb"], e["own"]
    prov = (f" ({own['model']} on `callo` {own['callo_version']}, {build_name(own['callo_commit'])}, --cache "
            f"{own['cache']}, llama.cpp {own['build']}, {own['date']})" if own else
            f" (the plotted run, {e['date']})" if e.get("date") else "")
    n = ""
    if fb.get("n_on") and fb.get("n_off"):
        on, off = round(fb["n_on"]["value"]), round(fb["n_off"]["value"])
        moved = round(fb["moved_on"]["value"]) if fb.get("moved_on") else None
        n = (f"; {on} decisions with it" + (", all moved off the reserved slot," if moved == on else
                                            f", {moved} moved off the reserved slot," if moved is not None else "")
             + f" and {off} without")
    tok = f"; a {fb['prompt_tokens']['value']:,.0f}-token prompt" if fb.get("prompt_tokens") else ""
    return (f"{e['short']}{prov}: p50 {_s(fb['on_p50']):.1f} s with the S1 fallback, {_s(fb['off_p50']):.1f} s "
            f"without (the prefill took {fb['prefill_on_s']['value']:.1f} s median with it, "
            f"{fb['prefill_off_s']['value']:.1f} s without{n}{tok}).")


DROPPED = ("solo-shim",)        # in the data (and FIGURES.md's floor), not on the figure


def _short(series: dict) -> str:
    return series["label"].split(",")[0]


def _rows(series: dict) -> dict:
    return {r["id"]: r for r in series["rows"]}


def subtitle(doc: dict) -> str:
    """The figure's second message, from the main series' own prefill rows (a plotted Tiel series when there is one)."""
    s = next((x for x in doc["series"] if x["name"] == "tiel"), doc["series"][0])
    r = _rows(s)
    parts = [f"{_s(r[k]['p50']):.1f} s{' p50' if i == 0 else ''} at {k.split('-')[1]}"
             for i, k in enumerate(("prefill-32k", "prefill-128k")) if k in r]
    return f"A long agent prefill still costs seconds: {', '.join(parts)} ({_short(s)})."


def fine_print(doc: dict, limit: str | None = None) -> list[dict]:
    """The figure's caption, three lines: the measurement, where the solos come from, and the case it does not
    plot (a foreign prefill inside the reserved slot), with its numbers. The rest is in FIGURES.md (`made`)."""
    out = []
    for s in doc["series"]:
        out.append({"lead": f"{Path(s['model']).stem}, llama.cpp {s['build']}, {s['date']}.",
                    "text": f"{s.get('harness', 'callo bench/acceptance.py')}; {s.get('launch') or 'release launch shape'}."})
    solos = [r["note"] for s in doc["series"] for r in s["rows"] if r["id"] not in DROPPED and r.get("note")]
    out.append({"lead": None, "text": f"Solos from another session ({'; '.join(solos)})."})
    fbs = fallbacks(doc)
    if fbs:
        lo, hi = (f(_s(e["fb"]["on_p50"]) for e in fbs) for f in (min, max))
        span = f"{lo:.0f} s" if f"{lo:.0f}" == f"{hi:.0f}" else f"{lo:.0f}–{hi:.0f} s"
        parts = []
        for i, e in enumerate(fbs):
            f, own = e["fb"], e["own"]
            who = f"{e['short']} on Callo {own['callo_version']}, {own['date']}" if own else e["short"]
            parts.append(f"{who}: p50 {_s(f['on_p50']):.1f} s with{' the S1 fallback' if i == 0 else ''}, "
                         f"{_s(f['off_p50']):.1f} s without")
        out.append({"lead": None, "text": f"A foreign prefill in the reserved slot itself still blocks: {span}, not "
                                          f"plotted ({'; '.join(parts)})."})
    return out


def made(doc: dict, limit: str) -> list[dict]:
    """How the figure was made, for FIGURES.md: the fallback case in full, the README's limit, the run-to-run
    floor, the context row's provenance and the gaps."""
    out = []
    fbs = fallbacks(doc)
    if fbs:
        out.append({"lead": "Not plotted: the S1 fallback under a foreign prefill.",
                    "text": "A foreign 32k prefill in the reserved slot itself. "
                            + " ".join(fallback_sentence(e) for e in fbs)
                            + " The fallback helps against a foreign decode, not a foreign prefill."})
    out.append({"lead": "README limit.", "text": f"“{limit}”"})
    f = doc["floor"]
    shim = next((r for s in doc["series"] for r in s["rows"] if r["id"] == "solo-shim"), None)
    shim_txt = (f" Its solo was p50 {shim['p50']['value']:.0f} / p95 {shim['p95']['value']:.0f} ms (n "
                f"{shim['n']['value']:.0f}), repeat {shim['repeat']['p50']['value']:.0f} / "
                f"{shim['repeat']['p95']['value']:.0f} ms." if shim else "")
    out.append({"lead": "Run-to-run floor.", "text": f"none through Callo; the nearest is {f['label']}: solo vs "
                f"solo-repeat |Δp| max {f['repeat_dp_max']['value']:.1f}, p50 {f['solo_p50']['value']:.0f} vs "
                f"{f['repeat_p50']['value']:.0f} ms. Contended there (no reserved slot), |Δp| max "
                f"{f['decode_dp_max']['value']:.4f} under decode, {f['prefill32k_dp_max']['value']:.4f} under 32k "
                f"prefill.{shim_txt}"})
    for c in doc.get("context") or []:
        out.append({"lead": "Context row.", "text": f"{c['model']} with {' '.join(Path(x).name if '/' in x else x for x in c['launch_flags'])}, "
                    f"`callo` v0.2, {c['date']}: a different model and session; vision ticks at 2 Hz, p50 "
                    f"{c['p50']['value']:.0f} ms, {c['overlap_p50']['value']:.0f} ms on the {c['overlap_n']['value']:.0f} "
                    f"ticks overlapping an agent prompt phase."})
    out.append({"lead": "Gaps, named and not filled.", "text": " ".join(f"({i}) {g}" for i, g in enumerate(doc["gaps"], 1))})
    files = doc.get("source_files") or []
    if files:
        out.append({"lead": "Sources.", "text": "; ".join(f"{f['role']}: `{f['name']}` (sha256 {f['sha256'][:16]}…)"
                                                          for f in files) + "."})
    return out


def _commit(commit: str) -> dict | None:
    """A callo commit as frontier.py checks it (sha, version at it, commit time, on HEAD's history)."""
    from frontier import commit_info
    return commit_info(commit)


def build_name(commit: str) -> str:
    """A callo commit as the figure names it: its build's role (data/builds.json), never the commit, which is of
    callo's development history and not published."""
    import figkit
    b = figkit.build_of(commit)
    if b is None:
        raise Refusal(f"callo commit {commit} is not in brand/data/builds.json: the figure names a build by its role")
    return b["name"]


def check_pins(doc: dict) -> None:
    """A fallback-only model's pinned callo commit: known here, on HEAD's history, at the version its summary
    records, and not later than its fallback cycles started."""
    for e in doc.get("fallback_only") or []:
        c = e.get("callo_commit")
        info = _commit(c) if c else None
        if info is None:
            raise Refusal(f"fallback_only.{e['name']}: callo commit {c!r} is not a commit of this repository")
        if not info["on_head"]:
            raise Refusal(f"fallback_only.{e['name']}: {c} is not on HEAD's history")
        if info["version"] != e.get("callo_version"):
            raise Refusal(f"fallback_only.{e['name']}: {c} is callo {info['version']}, the run says "
                          f"{e.get('callo_version')}")
        t = dt.datetime.fromisoformat(info["time"].replace("Z", "+00:00"))
        ran = dt.datetime.fromtimestamp(e["started"]["value"], dt.timezone.utc)
        if t > ran:
            raise Refusal(f"fallback_only.{e['name']}: {c} ({info['time']}) is later than its fallback cycles "
                          f"started ({ran.isoformat()})")
        e["callo_commit"] = info["short"]


def source_files(doc: dict, root: Path) -> list[dict]:
    """Every source, named by its role, file name and sha256 (no checkout paths in public text)."""
    roles = doc.get("source_roles") or {}
    out = []
    for src in doc.get("sources") or []:
        p = Path(src) if Path(src).is_absolute() else root / src
        out.append({"path": src, "role": roles.get(src, "source"), "name": p.name,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return out


def build(src: Path, root: Path, *, draft: bool = True) -> dict:
    import figkit
    doc = json.loads(Path(src).read_text(encoding="utf-8"))
    if doc.get("plate_concurrency") != 1:
        raise Refusal(f"{src}: not an extract_concurrency.py output (plate_concurrency 1)")
    n = verify(doc, root)
    check_pins(doc)
    doc["source_files"] = source_files(doc, root)
    for c in doc.get("context") or []:                     # a launch flag's path: as the snapshot records a path
        c["launch_flags"] = [figkit.public_path(x, root) if Path(x).is_absolute() else x for x in c["launch_flags"]]
    limit = readme_limit()
    snap = {**doc, "concurrency_figure": SNAPSHOT_FORMAT, "draft": draft, "verified_numbers": n,
            "source_file": figkit.public_path(os.path.abspath(src), root), "readme_limit": limit,
            "subtitle": subtitle(doc), "fine_print": fine_print(doc), "made": made(doc, limit),
            "builds": figkit.builds_named(e.get("callo_commit") for e in doc.get("fallback_only") or [])}
    if os.path.abspath(root) in json.dumps(snap, ensure_ascii=False):
        raise Refusal("the snapshot would record where the bench checkout sits; every path is relative to --root")
    return snap


# ---- drawing -----------------------------------------------------------------------------------------------------

W = 1280
SIDE = 64
AX = (452, 840)                  # the log ms axis, x range
AGENT_X = 884                    # the agent-throughput column
ROW_H = 74


def _f(v: float) -> str:
    return repr(float(v))


def _val(key: str, v: dict) -> str:
    return f'data-key="{key}" data-value="{_f(v["value"])}" data-src="{v["src"]}#{v["ptr"]}"'


def draw(snap: dict, *, theme: str = "dark") -> str:
    import figkit as fk

    t = fk.THEMES[theme]
    body: list[str] = []
    if snap.get("draft"):
        bst = fk.Style(size=11, weight=600, fill=t["sub"], tracking=1.4)
        btxt = "DRAFT · existing results only"
        body.append(f'<rect x="{SIDE}" y="40" width="{fk.num(fk.advance(btxt, bst) + 20)}" height="22" rx="2" '
                    f'fill="none" stroke="{t["sub"]}" stroke-opacity="0.6"/>')
        body.append(fk.text(btxt, SIDE + 10, 55.5, bst, role="badge")[0])
    tsans = fk.Style(size=34, weight=600, fill=t["ink"], width=96)
    tserif = fk.Style(size=38, weight=500, fill=t["ink"], face="serif")
    body.append(fk.paragraph([("Decisions ", tsans), ("stay fast while the agent decodes", tserif)], SIDE, 108,
                             W - 2 * SIDE, leading=42, role="title").svg)
    sub = fk.paragraph([(snap.get("subtitle") or "", fk.Style(size=16, fill=t["sub"]))], SIDE, 140, W - 2 * SIDE,
                       leading=23, role="subtitle")
    body.append(sub.svg)

    rows = [(s, r) for s in snap["series"] for r in s["rows"] if r["id"] not in DROPPED]
    ctx = snap.get("context") or []
    allv = [r[k]["value"] for _, r in rows for k in ("p50", "p95", "max")] + \
           [c[k]["value"] for c in ctx for k in ("p50", "overlap_p50")]
    d0, d1 = fk.log_bounds(min(allv) / 1.1, max(allv))          # the axis ends at the 1-2-5 value above the max
    xs = fk.LogScale(d0, d1, *AX)
    top = math.ceil(sub.box.y1 + 76)
    multi = len(snap["series"]) > 1
    n_rows = len(rows) + (len(snap["series"]) if multi else 0)
    ctx_top = top + n_rows * ROW_H + 40
    bottom = ctx_top + len(ctx) * ROW_H
    head_st = fk.Style(size=12, weight=600, fill=t["sub"], tracking=1.0)
    body.append(fk.text("CONDITION", SIDE, top - 22, head_st)[0])
    lx = AX[0]
    body.append(fk.text("DECISION LATENCY", lx, top - 22, head_st)[0])
    lx += fk.advance("DECISION LATENCY", head_st) + 22
    ly = top - 26
    # the legend draws the marks themselves (the brand faces have no ● or ┤)
    body.append(f'<circle cx="{fk.num(lx + 4)}" cy="{ly}" r="4" fill="{t["accent"]}"/>')
    body.append(fk.text("p50", lx + 13, top - 22, head_st)[0])
    lx += 13 + fk.advance("p50", head_st) + 16
    body.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{ly}" y2="{ly}" stroke="{t["accent"]}" stroke-opacity="0.6" '
                f'stroke-width="1.6"/><line x1="{lx + 18}" x2="{lx + 18}" y1="{ly - 4}" y2="{ly + 4}" '
                f'stroke="{t["accent"]}" stroke-opacity="0.6" stroke-width="1.6"/>')
    body.append(fk.text("p95", lx + 25, top - 22, head_st)[0])
    lx += 25 + fk.advance("p95", head_st) + 16
    body.append(f'<line x1="{lx}" x2="{lx}" y1="{ly - 6}" y2="{ly + 6}" stroke="{t["sub"]}" stroke-width="1"/>')
    body.append(fk.text("max", lx + 7, top - 22, head_st)[0])
    body.append(fk.text("AGENT THROUGHPUT", AGENT_X, top - 22, head_st)[0])
    for v in fk.log_ticks(d0, d1):
        x = xs(v)
        body.append(f'<line x1="{fk.num(x)}" x2="{fk.num(x)}" y1="{top - 8}" y2="{bottom}" stroke="{t["grid"]}"/>')
        body.append(fk.text(fk.fmt_ms(v), x, bottom + 22, fk.Style(size=13, fill=t["sub"]), anchor="middle",
                            role="tick")[0])

    lab_st = fk.Style(size=15, weight=600, fill=t["ink"])
    note_st = fk.Style(size=13, fill=t["sub"])
    num_st = fk.Style(size=13, fill=t["sub"])

    def row(y: float, key: str, r: dict, *, color: str, solo: bool) -> None:
        cy = y + ROW_H / 2
        body.append(f'<line x1="{SIDE}" x2="{W - SIDE}" y1="{fk.num(y + ROW_H)}" y2="{fk.num(y + ROW_H)}" '
                    f'stroke="{t["grid"]}"/>')
        body.append(fk.text(r["label"], SIDE, cy - 3, lab_st, role="row-label", attrs={"data-row": key})[0])
        note = f"n {r['n']['value']:.0f}"
        if r.get("mean"):
            note += f" · mean {r['mean']['value']:.0f} ms"
        if solo:
            note += f" · another session, {r.get('date') or ''}".rstrip(", ")
        body.append(fk.text(note, SIDE, cy + 16, note_st)[0])
        x50, x95, xmx = xs(r["p50"]["value"]), xs(r["p95"]["value"]), xs(r["max"]["value"])
        body.append(f'<line data-role="tail" {_val(key + ".p95", r["p95"])} x1="{_f(x50)}" x2="{_f(x95)}" '
                    f'y1="{_f(cy)}" y2="{_f(cy)}" stroke="{color}" stroke-opacity="0.6" stroke-width="1.6"/>')
        body.append(f'<line x1="{_f(x95)}" x2="{_f(x95)}" y1="{_f(cy - 5)}" y2="{_f(cy + 5)}" stroke="{color}" '
                    f'stroke-opacity="0.6" stroke-width="1.6"/>')
        body.append(f'<line data-role="max" {_val(key + ".max", r["max"])} x1="{_f(xmx)}" x2="{_f(xmx)}" '
                    f'y1="{_f(cy - 8)}" y2="{_f(cy + 8)}" stroke="{t["sub"]}" stroke-opacity="0.8" stroke-width="1"/>')
        fill = t["ground"] if solo else color
        body.append(f'<circle data-role="p50" {_val(key + ".p50", r["p50"])} cx="{_f(x50)}" cy="{_f(cy)}" r="5.5" '
                    f'fill="{fill}" stroke="{color}" stroke-width="1.8"/>')
        body.append(fk.text(fk.fmt_ms_value(r["p50"]["value"]), x50, cy - 12, num_st.with_(fill=t["ink"]),
                            anchor="middle", role="num")[0])
        body.append(fk.text(fk.fmt_ms_value(r["p95"]["value"]), x95, cy + 22, num_st, anchor="middle", role="num")[0])
        if xmx - x95 > 40:
            body.append(fk.text(fk.fmt_ms_value(r["max"]["value"]), xmx, cy + 22, num_st, anchor="middle",
                                role="num")[0])
        ag = r.get("agent")
        if ag:
            big = fk.Style(size=18, weight=600, fill=t["ink"])
            if ag["kind"] == "decode":
                txt = f"{ag['solo']['value']:.1f} → {ag['with']['value']:.1f} {ag['unit']}"
                rate = 1000 / r["p50"]["value"]
                subt = f"per decode stream: solo → while decisions run back-to-back (~{rate:.0f}/s)"
                body.append(f'<g data-role="agent" {_val(key + ".agent.solo", ag["solo"])}/>')
                body.append(f'<g data-role="agent" {_val(key + ".agent.with", ag["with"])}/>')
            else:
                txt = f"{ag['with']['value']:.0f} {ag['unit']}"
                subt = f"agent prefill of {ag['prompt_tokens']['value']:,.0f} tokens; no solo baseline"
                body.append(f'<g data-role="agent" {_val(key + ".agent.with", ag["with"])}/>')
            body.append(fk.text(txt, AGENT_X, cy - 4, big, role="agent-text")[0])
            body.append(fk.paragraph([(subt, note_st)], AGENT_X, cy + 15, W - SIDE - AGENT_X, leading=16,
                                     role="agent-note").svg)

    y = top
    for s in snap["series"]:
        if multi:
            body.append(fk.text(s["label"], SIDE, y + 40, head_st.with_(fill=t["ink"]), role="series")[0])
            y += ROW_H
        for r in s["rows"]:
            if r["id"] in DROPPED:
                continue
            solo = r["id"].startswith("solo")
            row(y, f"{s['name']}.{r['id']}", r, color=t["ink"] if solo else t["accent"], solo=solo)
            y += ROW_H
    body.append(f'<line x1="{SIDE}" x2="{W - SIDE}" y1="{top}" y2="{top}" stroke="{t["axis"]}"/>')
    # context: another model and session, set apart, in its own marks (diamond p50, triangle overlap)
    y = ctx_top
    for c in ctx:
        cy = y + ROW_H / 2
        key = f"context.{c['id']}"
        body.append(f'<line x1="{SIDE}" x2="{W - SIDE}" y1="{fk.num(y)}" y2="{fk.num(y)}" stroke="{t["axis"]}" '
                    f'stroke-dasharray="2 4"/>')
        body.append(fk.text(c["label"], SIDE, cy - 3, lab_st.with_(fill=t["sub"]), role="row-label",
                            attrs={"data-row": key})[0])
        body.append(fk.text(f"n {c['n']['value']:.0f} · {Path(c['model']).stem}, Callo v0.2, {c['date']}", SIDE,
                            cy + 16, note_st)[0])
        xa, xb = xs(c["p50"]["value"]), xs(c["overlap_p50"]["value"])
        r = 6.2
        body.append(f'<line x1="{_f(xa)}" x2="{_f(xb)}" y1="{_f(cy)}" y2="{_f(cy)}" stroke="{t["sub"]}" '
                    f'stroke-opacity="0.6" stroke-dasharray="2 3"/>')
        body.append(f'<path data-role="p50" {_val(key + ".p50", c["p50"])} d="M{_f(xa)} {_f(cy - r)} L{_f(xa + r)} {_f(cy)} '
                    f'L{_f(xa)} {_f(cy + r)} L{_f(xa - r)} {_f(cy)} Z" fill="{t["sub"]}"/>')
        body.append(f'<path data-role="overlap" {_val(key + ".overlap_p50", c["overlap_p50"])} '
                    f'd="M{_f(xb)} {_f(cy - r)} L{_f(xb + r * 0.95)} {_f(cy + r * 0.62)} L{_f(xb - r * 0.95)} '
                    f'{_f(cy + r * 0.62)} Z" fill="{t["sub"]}"/>')
        body.append(fk.text(fk.fmt_ms_value(c["p50"]["value"]), xa, cy - 12, num_st.with_(fill=t["ink"]),
                            anchor="middle", role="num")[0])
        body.append(fk.text(f"{fk.fmt_ms_value(c['overlap_p50']['value'])} on the {c['overlap_n']['value']:.0f} ticks "
                            f"overlapping an agent prompt", xb + 12, cy + 4.5, num_st, role="num")[0])
        y += ROW_H
    body.append(fk.text("decision latency, log scale  →", sum(AX) / 2, bottom + 48, fk.Style(size=13.5, fill=t["sub"]),
                        anchor="middle", role="axis-title")[0])

    # caption: three lines at 14 px
    yy = bottom + 104
    lead_st = fk.Style(size=14, weight=600, fill=t["ink"])
    fp_st = fk.Style(size=14, fill=t["sub"])
    for f in snap["fine_print"]:
        runs = [(f["lead"] + " ", lead_st), (f["text"], fp_st)] if f["lead"] else [(f["text"], fp_st)]
        pr = fk.paragraph(runs, SIDE, yy, W - 2 * SIDE, leading=20, role="fine-print")
        body.append(pr.svg)
        yy += pr.lines * 20 + 6
    body.append(f'<line x1="{SIDE}" x2="{W - SIDE}" y1="{bottom + 72}" y2="{bottom + 72}" stroke="{t["grid"]}"/>')
    H = math.ceil(yy + 22)
    bg = fk.ground(W, H, t, glow_at=(W / 2, bottom, W * 0.4, 240))
    return fk.svg_doc(W, H, bg + "".join(body), title="Decisions stay fast while the agent decodes", theme=theme)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="concurrency.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True, help="extract_concurrency.py's concurrency.json")
    b.add_argument("--root", required=True, help="the benchmark's root, which its src paths are relative to")
    b.add_argument("--final", action="store_true", help="not a draft")
    b.add_argument("--out", default=str(HERE / "data" / "concurrency.json"))
    d = sub.add_parser("draw")
    d.add_argument("--data", default=str(HERE / "data" / "concurrency.json"))
    d.add_argument("--out-dir", default=str(HERE))
    d.add_argument("--png", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            snap = build(Path(a.src), Path(a.root), draft=not a.final)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(snap, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {a.out}: {snap['verified_numbers']} numbers verified against their sources")
            return 0
        snap = json.loads(Path(a.data).read_text(encoding="utf-8"))
        for theme in ("dark", "light"):
            out = Path(a.out_dir) / f"concurrency-{theme}.svg"
            out.write_text(draw(snap, theme=theme), encoding="utf-8")
            print(f"wrote {out}")
            if a.png:
                import figkit
                png = out.with_name(out.stem + "@2x.png")
                figkit.render_png(str(out), str(png), 2.0)
                print(f"wrote {png}")
        return 0
    except Refusal as e:
        print(f"concurrency: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
