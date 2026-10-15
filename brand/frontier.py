"""callo's release plate: accuracy against per-decision latency, callo's think budgets beside trained decision heads.

    python brand/frontier.py build --spec brand/data/frontier.release.spec.json --root <results>   # -> data/frontier.json
    python brand/frontier.py draw [--latency mean|p50|p95] [--png]                   # -> brand/frontier-{dark,light}.svg

Two layers. The DATA layer (`build`) reads callo think profiles and per-question rows files and computes everything
with callo's own code: `compare.load_profile`, `compare.refusals`, `compare.pairing`, `compare.points_of`,
`compare.rows_at`, `compare.joint` (the pick), `think.frontier` (each model's own frontier), `think.latency_stats`
and `think.paired_bootstrap` (one seed, one row set, one source stratification, as `callo compare`). Its output, the
snapshot `brand/data/frontier.json`, is committed, so the DRAWING layer (`draw`) re-bakes the README assets without
the bench results. The snapshot records every source path relative to the build's `--root` (the results directory the
spec's paths are relative to) and names callo's builds by role (`data/builds.json`), never by commit.

The plate plots thinking-budget points only. A profile's other points (early exit, with or without its gate; deadline
points) are opt-in and never picked by callo: they are left off the chart and listed in FIGURES.md, and the spec's
`early_exit` block may state one measured early-exit point per model in the fine print.

The spec (JSON) names the arms; every arm is optional, and an arm whose file does not exist yet is listed as pending:

    {"data": "think/data/test-s1.jsonl", "root": "/path/to/results", "ref": "clef-flash", "latency": "mean",
     "callo_cache": false, "callo_run": "run1", "draft": true,
     "arms": [{"name": "tiel", "label": "...", "kind": "callo", "profile": "think/run1/tiel.think.json",
               "runtime": "..."},
              {"name": "clef-flash", "label": "...", "kind": "rows", "rows": "plate/clef-flash-2026-10-02",
               "runtime": "...", "marker": "diamond", "timer": "client_remote", "note": "..."}]}

Optional release keys: `callo_commits` ({arm: {"profile": commit, "measured": commit, "step": "refit"}}, the callo
commits a profile was written and measured at, checked against this repository: the commit exists on HEAD's history,
its version is the profile's `callo_version`, and it is not later than the profile it is said to have written; the
plate names each by its build's role in `data/builds.json`);
`early_exit` ({"z", "gate", "budget", "arms": {arm: profile path or null for the arm's own}}); `readout_check`
({"arm", "before": profile, "fix": commit}: one model measured on both sides of a readout fix). An arm may carry
`label_side` (start, end or along: where its line's label goes when there is room there) and `gib_note` (its clause in
the plate's GiB line).

A rows arm is a directory holding `rows.jsonl` (`{q, cluster, source, correct, lat_ms, probs, ...}` per question; an
error row carries `status`/`error` and no `probs`) and `meta.json` (the data's sha256 as `data.sha256` or
`data_sha256`; `gib`, `date`/`utc_start`, `build`, `timer`, each overridable in the spec arm; the spec's `timer` is
one of the TIMERS keys, meta.json's free-text timer is kept in the snapshot as provenance). Refusals exit 2.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))          # callo's own statistics and frontier code, never a copy
sys.path.insert(0, str(HERE))

from callo import compare, think  # noqa: E402
from callo.errors import InvalidInput  # noqa: E402

SNAPSHOT_FORMAT = 1
STATS = think.LATENCY_AXES                      # mean, p50, p95: the pick statistics
TIMERS = {
    "callo": "in-Callo frame + read wall time",
    "client_http": ("client wall per request (clef_run.py: a fresh loopback connection, the round trip and JSON parse, "
                    "one at a time)"),
    "client_remote": ("client wall per request (kev.benchmark RemotePredictor: new connection, round trip, "
                      "json.loads) against the release code (transformers, MPS)"),
    "kev_inprocess": "in-process model forward only (kev.benchmark's LocalPredictor, MPS); tokenisation outside the timer",
}
MARKERS = ("circle", "diamond", "diamond-hollow", "square", "square-hollow", "triangle", "triangle-hollow")
GIB_LINE = "GiB = file size (GGUF; for native, safetensors shards + head, as shipped)."
GIB_TEXT = "File size (GGUF; for native, safetensors shards and head, as shipped)"      # the plate's GiB line


class Refusal(Exception):
    """The plate refuses to draw something that does not compare: exit 2 with the reason."""


# ---- the spec ----------------------------------------------------------------------------------------------------

def load_spec(path: str | Path, *, root: str | Path | None = None) -> dict:
    """The spec with every path resolved: against `root` (the argument, `--root`), else the spec's own `root`, else
    the spec's directory."""
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    base = Path(os.path.abspath(root or spec.get("root") or path.parent))
    spec = dict(spec)
    spec["_spec_path"] = str(path)
    spec["_root"] = str(base)

    def res(p: str) -> str:
        return str(p if os.path.isabs(p) else base / p)

    spec["data"] = res(spec["data"])
    arms = []
    for a in spec.get("arms") or []:
        a = dict(a)
        for k in ("profile", "rows"):
            if a.get(k):
                a[k] = res(a[k])
        arms.append(a)
    spec["arms"] = arms
    q4 = spec.get("q4_cost")
    if isinstance(q4, dict) and q4.get("bf16_rows"):
        spec["q4_cost"] = {**q4, "bf16_rows": res(q4["bf16_rows"]), **({"vs": res(q4["vs"])} if q4.get("vs") else {})}
    v = spec.get("validation")
    if isinstance(v, dict) and v.get("path"):
        spec["validation"] = {**v, "path": res(v["path"])}
    ee = spec.get("early_exit")
    if isinstance(ee, dict):
        spec["early_exit"] = {**ee, "arms": {k: ({**p, "path": res(p["path"])} if isinstance(p, dict) else
                                                 res(p) if p else None) for k, p in (ee.get("arms") or {}).items()}}
    rc = spec.get("readout_check")
    if isinstance(rc, dict) and rc.get("before"):
        spec["readout_check"] = {**rc, "before": res(rc["before"])}
    return spec


def _check_arms(arms: list[dict]) -> None:
    seen: set[str] = set()
    for a in arms:
        name = a.get("name")
        if not isinstance(name, str) or not name:
            raise Refusal(f"an arm has no name: {a}")
        if name in seen:
            raise Refusal(f"two arms are named {name!r}: arm names key the plate's series")
        seen.add(name)
        if a.get("kind") not in ("callo", "rows"):
            raise Refusal(f"arm {name}: kind must be callo or rows, got {a.get('kind')!r}")
        if not a.get(a["kind"] == "callo" and "profile" or "rows"):
            raise Refusal(f"arm {name}: a {a['kind']} arm needs its {'profile' if a['kind'] == 'callo' else 'rows'} path")
        if a.get("marker") is not None and a["marker"] not in MARKERS:
            raise Refusal(f"arm {name}: marker must be one of {', '.join(MARKERS)}")
        if not a.get("label"):
            raise Refusal(f"arm {name}: no label")


def _present(a: dict) -> bool:
    if a["kind"] == "callo":
        return Path(a["profile"]).is_file()
    return (Path(a["rows"]) / "rows.jsonl").is_file()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- data --------------------------------------------------------------------------------------------------------

def read_data(path: str) -> list[dict]:
    """The test set's questions in file order, keyed as callo's rows are: q = `_meta.id` + "/" + question key."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = r.get("_meta") or {}
            for k in r["questions"]:
                out.append({"q": f"{m['id']}/{k}", "cluster": str(m.get("group_id") or m["id"]),
                            "source": m.get("source")})
    return out


def read_rows(arm: dict, data_sha: str) -> tuple[dict, list[dict]]:
    d = Path(arm["rows"])
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    rows = []
    with open(d / "rows.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    sha = (meta.get("data") or {}).get("sha256") if isinstance(meta.get("data"), dict) else None
    sha = sha or meta.get("data_sha256")
    if sha is None:
        raise Refusal(f"arm {arm['name']}: {meta_path} carries no data sha256 (data.sha256 or data_sha256)")
    if sha != data_sha:
        raise Refusal(f"arm {arm['name']} was measured on different data (sha256 {sha[:12]}) than the plate "
                      f"({data_sha[:12]}); points on different questions do not compare")
    return meta, rows


def error_row(r: dict) -> bool:
    return bool(r.get("error")) or r.get("status") not in (None, 200) or not isinstance(r.get("probs"), dict)


def _cache_text(cache: bool, run: str | None) -> str:
    return f"Callo arms: {run + ', ' if run else ''}--cache {'on (default)' if cache else 'off (pre-release default)'}"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src"], capture_output=True,
                               text=True, check=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _date(s: Any) -> str | None:
    return str(s)[:10] if s else None


def _ts(s: Any) -> dt.datetime | None:
    """An ISO time (a trailing Z included, which Python 3.10's fromisoformat refuses), as an aware datetime."""
    if not s:
        return None
    t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _git(*args: str) -> str | None:
    """git's stdout in this repository, None when git fails (no git, an unknown commit, not an ancestor)."""
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def commit_info(commit: str) -> dict | None:
    """A callo commit as the plate names it: its full and short sha, callo's version at it (pyproject.toml), its
    commit time and whether HEAD's history holds it; None when this checkout does not know it."""
    sha = _git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if not sha:
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"', _git("show", f"{sha}:pyproject.toml") or "", re.M)
    return {"sha": sha, "short": sha[:7], "version": m.group(1) if m else None,
            "time": _git("log", "-1", "--format=%cI", sha),
            "on_head": _git("merge-base", "--is-ancestor", sha, "HEAD") is not None}


def is_ancestor(a: str, b: str) -> bool:
    return _git("merge-base", "--is-ancestor", a, b) is not None


def build_name(commit: str) -> str:
    """A callo commit as the plate names it: its build's role ("the readout-fix build", data/builds.json), never the
    commit, which is of callo's development history and not published."""
    import figkit
    b = figkit.build_of(commit)
    if b is None:
        raise Refusal(f"callo commit {commit} is not in brand/data/builds.json: the plate names a build by its role")
    return b["name"]


def _file_names(text: str) -> str:
    """Each absolute path in a text reduced to its file name."""
    return re.sub(r"/(?:[^\s/,;]+/)+([^\s/,;]+)", r"\1", text)


def _pinned(arm: str, pin: Any, p: dict) -> dict:
    """An arm's pinned callo commits, checked against its profile: each commit is known, on HEAD's history and at the
    profile's callo version; the profile commit is not later than the profile's `created`, the measured one not later
    than the run's start, and the measured one is the profile commit or an ancestor of it."""
    if isinstance(pin, str):
        pin = {"profile": pin}
    if not isinstance(pin, dict) or not isinstance(pin.get("profile"), str):
        raise Refusal(f"callo_commits.{arm}: a commit, or {{profile, measured, step}}")
    version = ((p.get("comparable") or {}).get("readout") or {}).get("callo_version")
    run = (p.get("environment") or {}).get("run") or {}
    out: dict[str, Any] = {"step": pin.get("step")}
    for key, when, what in (("profile", p.get("created"), "the profile was written"),
                            ("measured", run.get("started"), "its run started")):
        c = pin.get(key)
        if c is None:
            continue
        info = commit_info(c)
        if info is None:
            raise Refusal(f"callo_commits.{arm}.{key}: {c} is not a commit of this repository")
        if not info["on_head"]:
            raise Refusal(f"callo_commits.{arm}.{key}: {c} is not on HEAD's history")
        if version and info["version"] != version:
            raise Refusal(f"callo_commits.{arm}.{key}: {c} is callo {info['version']}, the profile says {version}")
        if when and _ts(info["time"]) > _ts(when):
            raise Refusal(f"callo_commits.{arm}.{key}: {c} ({info['time']}) is later than when {what} ({when})")
        out[key] = info["short"]
    if out.get("measured") and not is_ancestor(pin["measured"], pin["profile"]):
        raise Refusal(f"callo_commits.{arm}: the measured commit {pin['measured']} is not an ancestor of the profile "
                      f"commit {pin['profile']}")
    out.setdefault("measured", out["profile"])
    return out


def build(spec: dict) -> dict:
    """The snapshot: every point with its accuracy, n, five latency statistics, paired Δacc vs the reference, own-model
    frontier membership per pick statistic, GiB; callo's pick per statistic; flags; provenance; the fine print."""
    import figkit
    arms = [dict(a) for a in spec.get("arms") or []]
    _check_arms(arms)
    data_path = spec["data"]
    if not Path(data_path).is_file():
        raise Refusal(f"no data file at {data_path}")
    data_sha = sha256_file(data_path)
    questions = read_data(data_path)
    present = [a for a in arms if _present(a)]
    pending = [a for a in arms if not _present(a)]
    if not present:
        raise Refusal("no arm has its file yet: nothing to plot")
    callo_arms = [a for a in present if a["kind"] == "callo"]
    rows_arms = [a for a in present if a["kind"] == "rows"]

    profiles = []
    for a in callo_arms:
        try:
            p = compare.load_profile(a["profile"])
        except InvalidInput as e:
            raise Refusal(str(e)) from None
        if (p.get("comparable") or {}).get("data", {}).get("sha256") != data_sha:
            raise Refusal(f"arm {a['name']} was measured on different data than {Path(data_path).name} "
                          f"({str((p.get('comparable') or {}).get('data', {}).get('sha256'))[:12]} vs {data_sha[:12]})")
        if "callo_cache" not in spec:
            raise Refusal("the spec must state callo_cache (true/false): the plate labels the callo arms' --cache")
        cache = (p.get("comparable") or {}).get("cache")
        if cache is not bool(spec["callo_cache"]):
            raise Refusal(f"arm {a['name']}: its profile was calibrated with --cache {'on' if cache else 'off'}, but "
                          f"the plate says {_cache_text(bool(spec['callo_cache']), spec.get('callo_run'))!r}")
        profiles.append({**p, "model_name": a["name"]})        # points and labels carry the arm's name
    if profiles:
        bad = compare.refusals(profiles)
        if bad:
            raise Refusal("refusing to compare: " + "; ".join(bad))
        fixed, clusters, strata, pair_flags = compare.pairing(profiles)
        seed = int((profiles[0].get("comparable") or {}).get("seed") or 0)
    else:
        fixed = [q["q"] for q in questions]
        clusters = [q["cluster"] for q in questions]
        of: dict[str, Any] = {}
        whole = all(of.setdefault(c, q["source"]) == q["source"] for c, q in zip(clusters, questions, strict=True))
        strata = [str(q["source"]) for q in questions] if whole and all(isinstance(q["source"], str) for q in questions) else None
        pair_flags = [] if strata is not None else ["rows without a usable source: paired intervals are unstratified"]
        seed = int(spec.get("seed", 0))
    if not fixed:
        raise Refusal("no question was measured at every point of every callo arm")

    # rows arms: every fixed question, no error rows, one row per question
    rows_by: dict[str, dict[str, dict]] = {}
    metas: dict[str, dict] = {}
    for a in rows_arms:
        meta, rows = read_rows(a, data_sha)
        errs = [r for r in rows if error_row(r)]
        if errs:
            raise Refusal(f"arm {a['name']}: {len(errs)} error row(s) (first {errs[0].get('q')}: status "
                          f"{errs[0].get('status')}, {errs[0].get('error')!r}); the plate refuses an arm with error rows")
        by: dict[str, dict] = {}
        for r in rows:
            if r.get("q") in by:
                raise Refusal(f"arm {a['name']}: question {r.get('q')} appears twice")
            by[r["q"]] = r
        missing = [q for q in fixed if q not in by]
        if missing:
            raise Refusal(f"arm {a['name']}: {len(missing)} of the {len(fixed)} paired questions are missing "
                          f"(first {missing[0]})")
        rows_by[a["name"]] = by
        metas[a["name"]] = meta

    # points: thinking budgets only; a profile's opt-in points (early exit, deadlines) are listed, never plotted
    points: list[tuple[dict, think.Point]] = []            # (arm, point)
    callo_rows = {}
    opt_in: dict[str, list[str]] = {}
    for a, p in zip(callo_arms, profiles, strict=True):
        callo_rows[a["name"]] = compare.rows_at(p)
        for pt in compare.points_of(p, a["name"]):
            if pt.kind == "budget":
                points.append((a, pt))
            else:
                opt_in.setdefault(a["name"], []).append(pt.label)
    for a in rows_arms:
        by = rows_by[a["name"]]
        meta = metas[a["name"]]
        lats = [float(by[q]["lat_ms"]) for q in fixed]
        stats = think.latency_stats(lats, clusters, seed=seed, strata=strata)
        correct = [float(bool(by[q]["correct"])) for q in fixed]
        gib = a.get("gib", meta.get("gib"))
        pt = think.Point(kind="budget", budget=0, name=a["name"], acc=sum(correct) / len(correct), n=len(fixed),
                         **think.point_latencies(stats), gib=float(gib) if gib is not None else None, capable=False)
        points.append((a, pt))

    def correct_of(a: dict, pt: think.Point) -> list[float]:
        if a["kind"] == "callo":
            r = callo_rows[a["name"]]
            return [r[q][(pt.kind, pt.value)][0] for q in fixed]
        return [float(bool(rows_by[a["name"]][q]["correct"])) for q in fixed]

    def label_of(a: dict, pt: think.Point) -> str:
        return pt.label if a["kind"] == "callo" else a["name"]

    # each model's own frontier, per pick statistic
    dominated: dict[str, dict[str, str | None]] = {}
    for a in present:
        own = [pt for b, pt in points if b is a]
        for stat in STATS:
            for f in think.frontier(own, latency=stat):
                dominated.setdefault(label_of(a, f), {})[stat] = (f.dominated_by if a["kind"] == "callo" else None)

    # callo's pick, per statistic, over the callo arms (compare.joint: the same pick `callo compare` prints)
    picks: dict[str, str | None] = {}
    flags: list[str] = []
    notes: list[str] = []
    if profiles:
        for stat in STATS:
            j = compare.joint(profiles, pref=think.Preference(latency=stat))
            chosen = j.chosen.point
            if chosen is not None and chosen.kind != "budget":
                raise Refusal(f"callo compare picks {chosen.label} on {stat}, not a thinking budget: the plate plots "
                              "budgets only")
            picks[stat] = chosen.label if chosen is not None else None
        flags = compare.flags(profiles, think.Preference(latency=spec.get("latency", "mean")))
        notes = compare.notes(profiles)
    else:
        flags = list(pair_flags)

    # Δ vs the reference (paired over `fixed`)
    ref_name = spec.get("ref")
    ref = next(((a, pt) for a, pt in points if a["name"] == ref_name), None)
    ref_correct = correct_of(*ref) if ref is not None else None

    out_points = []
    for a, pt in points:
        dacc = None
        if ref_correct is not None and not (ref is not None and a is ref[0]):
            dacc = list(think.paired_bootstrap(ref_correct, correct_of(a, pt), clusters, seed=seed, strata=strata))
        lab = label_of(a, pt)
        out_points.append({
            "label": lab, "arm": a["name"], "kind": pt.kind, "budget": pt.budget,
            "value": pt.value, "acc": pt.acc, "n": pt.n,
            "lat": {"mean": list(pt.lat_mean), "p50": list(pt.lat_p50), "p95": list(pt.lat_p95),
                    "p99": list(pt.lat_p99), "max": pt.lat_max},
            "dacc": dacc, "frontier": {s: dominated[lab][s] is None for s in STATS},
            "dominated_by": dict(dominated[lab]), "gib": pt.gib, "think_mean": pt.think_mean,
            "pick": [s for s in STATS if picks.get(s) == lab],
        })

    def pub(p: str | None) -> str | None:                  # a path as the snapshot records it (relative to --root)
        return figkit.public_path(p, spec["_root"])

    # arms: what the labels and the fine print say about each
    out_arms = []
    for a in arms:
        rec = {k: a.get(k) for k in ("name", "label", "short", "kind", "runtime", "marker", "note", "label_side",
                                     "gib_note")}
        rec["on_plate"] = a.get("plate", True) is not False      # False: measured and checked, reported in FIGURES.md
        rec["short"] = short(a)
        rec["timer"] = "callo" if a["kind"] == "callo" else a.get("timer")
        rec["status"] = "present" if a in present else "pending"
        rec["path"] = pub(a.get("profile") or a.get("rows"))
        if a in present and a["kind"] == "callo":
            p = profiles[callo_arms.index(a)]
            env = p.get("environment") or {}
            run = env.get("run") or {}
            pin = (spec.get("callo_commits") or {}).get(a["name"])
            rec.update(date=_date(run.get("started")), date_end=_date(run.get("ended")), build=env.get("build_info"),
                       gib=next(pt.gib for b, pt in points if b is a),
                       callo_version=((p.get("comparable") or {}).get("readout") or {}).get("callo_version"),
                       cache=(p.get("comparable") or {}).get("cache"),
                       host=(env.get("host") or {}).get("chip"),
                       memory_gib=round(((env.get("host") or {}).get("memory_bytes") or 0) / 2**30),
                       capable=bool((p.get("thinking") or {}).get("capable")),
                       budgets=[pt.budget for b, pt in points if b is a],
                       created=p.get("created"), run={"started": run.get("started"), "ended": run.get("ended")},
                       model_file=(env.get("model") or {}).get("file"),
                       commits=_pinned(a["name"], pin, p) if pin is not None else None,
                       opt_in=opt_in.get(a["name"], []),
                       files=[{"path": pub(a["profile"]), "sha256": sha256_file(a["profile"])}])
        elif a in present:
            meta = metas[a["name"]]
            sess = meta.get("sessions") if isinstance(meta.get("sessions"), list) else []
            rec.update(date=a.get("date") or _date(meta.get("date") or meta.get("utc_start") or meta.get("started")
                                                   or (sess[0].get("utc_start") if sess and isinstance(sess[0], dict)
                                                       else None)),
                       build=a.get("build") or meta.get("build") or meta.get("build_info"),
                       gib=next(pt.gib for b, pt in points if b is a),
                       timer=a.get("timer") or meta.get("timer"),
                       files=[{"path": pub(str(Path(a["rows"]) / f)), "sha256": sha256_file(Path(a["rows"]) / f)}
                              for f in ("rows.jsonl", "meta.json") if (Path(a["rows"]) / f).is_file()],
                       meta={k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool)) or v is None})
            if rec["date"] is None:
                raise Refusal(f"arm {a['name']}: no measurement date (put date in the spec arm or its meta.json)")
            if rec["gib"] is None:
                raise Refusal(f"arm {a['name']}: GiB unknown (put gib in its meta.json or the spec arm)")
            if rec["timer"] not in TIMERS or rec["timer"] == "callo":
                raise Refusal(f"arm {a['name']}: timer must be one of client_http, client_remote, kev_inprocess "
                              f"(meta.json or the spec arm), got {rec['timer']!r}")
        out_arms.append(rec)

    # the "what Q4 costs" line: UD-Q4 vs llama.cpp BF16 on the same runtime, paired over `fixed`
    q4 = None
    q4spec = spec.get("q4_cost")
    if isinstance(q4spec, dict) and q4spec.get("arm") in rows_by and Path(q4spec.get("bf16_rows", "")).is_dir():
        bf_meta, bf_rows = read_rows({"name": "q4_cost.bf16_rows", "rows": q4spec["bf16_rows"]}, data_sha)
        if any(error_row(r) for r in bf_rows):
            raise Refusal("q4_cost.bf16_rows holds error rows")
        bf = {r["q"]: r for r in bf_rows}
        if any(q not in bf for q in fixed):
            raise Refusal("q4_cost.bf16_rows misses paired questions")
        base = [float(bool(bf[q]["correct"])) for q in fixed]
        quant = [float(bool(rows_by[q4spec["arm"]][q]["correct"])) for q in fixed]
        median_dp = q4spec.get("median_dp")
        if median_dp is None and q4spec.get("vs") and Path(q4spec["vs"]).is_file():
            median_dp = json.loads(Path(q4spec["vs"]).read_text(encoding="utf-8"))["gguf_vs_native"]["dp"]["median"]
        q4 = {"arm": q4spec["arm"], "dacc": list(think.paired_bootstrap(base, quant, clusters, seed=seed, strata=strata)),
              "median_dp": median_dp, "bf16_rows": pub(q4spec["bf16_rows"]),
              "label": q4spec.get("label", "Clef-Flash UD-Q4 vs llama.cpp BF16 (same runtime)")}

    unknown = sorted(set(spec.get("callo_commits") or {}) - {a["name"] for a in callo_arms})
    if unknown:
        raise Refusal(f"callo_commits names {', '.join(unknown)}: not a present callo arm")
    by_name = {a["name"]: p for a, p in zip(callo_arms, profiles, strict=True)}
    shorts = {a["name"]: short(a) for a in arms}

    snap = {
        "plate_frontier": SNAPSHOT_FORMAT,
        "title": spec.get("title"), "subtitle": spec.get("subtitle"),
        "draft": bool(spec.get("draft")), "latency": spec.get("latency", "mean"),
        "data": {"path": pub(data_path), "sha256": data_sha, "name": Path(data_path).stem, "n": len(fixed),
                 "sources": _source_counts(fixed, questions, spec.get("source_order"))},
        "seed": seed, "strata": "source" if strata is not None else None,
        "ref": ({"name": ref_name, "label": spec.get("ref_label") or short(ref[0]), "acc": ref[1].acc} if ref is not None
                else {"name": ref_name, "label": None, "acc": None}),
        "callo_cache": spec.get("callo_cache"), "callo_run": spec.get("callo_run"),
        "arms": out_arms, "points": out_points, "picks": picks,
        "flags": [_file_names(f) for f in flags], "notes": notes, "validation": _validation(spec.get("validation"), pub),
        "q4_cost": q4,
        "generator": {"callo_commit": _git_commit(), "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "spec": figkit.public_path(os.path.abspath(spec["_spec_path"]), REPO)},
        "pending": [a["name"] for a in pending],
        "early_exit": _early_exit(spec.get("early_exit"), by_name, {a["name"]: a["profile"] for a in callo_arms}, shorts,
                                  data_sha, pub),
        "readout_check": _readout_check(spec.get("readout_check"), by_name, shorts, data_sha, pub),
    }
    rc = snap["readout_check"]
    if rc is not None:                          # the measured commits the fix is not in: their arms predate it
        rc["predates"] = sorted({a["commits"]["measured"] for a in out_arms if a.get("commits")
                                 and not is_ancestor(rc["fix"], a["commits"]["measured"])})
    snap["story"] = {s: story(snap, spec, s) for s in STATS}
    snap["fine_print"] = {s: fine_print(snap, s) for s in STATS}
    snap["made"] = made(snap)
    snap["builds"] = figkit.builds_named(
        [(a.get("commits") or {}).get(k) for a in out_arms for k in ("measured", "profile")]
        + [x.get("commit") for x in ((snap.get("early_exit") or {}).get("points") or [])]
        + ([rc["fix"], *rc["predates"]] if rc else []))
    if spec["_root"] in json.dumps(snap, ensure_ascii=False):
        raise Refusal("the snapshot would record where the results sit; every path is relative to --root")
    return snap


def _validation(v: Any, pub: Any = str) -> dict | None:
    """The same-runtime validation: inline {agree, n, median_dp}, or {path, label} to validate.py's JSON."""
    if not isinstance(v, dict):
        return None
    if v.get("path"):
        if not Path(v["path"]).is_file():
            return None
        doc = json.loads(Path(v["path"]).read_text(encoding="utf-8"))
        bar = doc["g1_bar"]
        return {"agree": bar["agree"], "n": bar["n"], "median_dp": bar["median_dp"], "pass": bar["pass"],
                "min_agree": bar["bar"]["min_agree"], "max_median_dp": bar["bar"]["max_median_dp"],
                "label": v.get("label"), "path": pub(v["path"]), "sha256": sha256_file(v["path"])}
    return dict(v) if v.get("agree") is not None else None


def _same_model(name: str, what: str, other: dict, plotted: dict, data_sha: str) -> None:
    """`other` (another profile of an arm's model) must be the plotted profile's model, data and --cache."""
    if other.get("model_key") != plotted.get("model_key"):
        raise Refusal(f"{what} for {name} is another model ({str(other.get('model_key'))[:19]} vs "
                      f"{str(plotted.get('model_key'))[:19]})")
    if ((other.get("comparable") or {}).get("data") or {}).get("sha256") != data_sha:
        raise Refusal(f"{what} for {name} was measured on different data than the plate")
    if (other.get("comparable") or {}).get("cache") != (plotted.get("comparable") or {}).get("cache"):
        raise Refusal(f"{what} for {name} ran with another --cache than its plotted profile")


def _early_exit(cfg: Any, by_name: dict[str, dict], paths: dict[str, str], shorts: dict[str, str],
                data_sha: str, pub: Any = str) -> dict | None:
    """The fine print's early-exit line: per named arm, the measured early-exit point (z, with or without the gate,
    at the think budget it caps) and the plain budget point of the same calibration, read from a profile of the same
    model, data and --cache (the arm's own when its path is null). Opt-in and never picked: never plotted."""
    if not isinstance(cfg, dict):
        return None
    z, gate, budget = float(cfg["z"]), bool(cfg.get("gate")), int(cfg["budget"])
    block = "ee_gated" if gate else "ee"
    out = []
    for name, ent in (cfg.get("arms") or {}).items():
        if name not in by_name:
            raise Refusal(f"early_exit names {name}: not a present callo arm")
        path, pin = (ent.get("path"), ent.get("commit")) if isinstance(ent, dict) else (ent, None)
        own = path is None or Path(path).resolve() == Path(paths[name]).resolve()
        try:
            p = by_name[name] if path is None else compare.load_profile(path)
        except InvalidInput as e:
            raise Refusal(str(e)) from None
        _same_model(name, "early_exit's profile", p, by_name[name], data_sha)
        e = next((v for k, v in (p.get(block) or {}).items() if float(k) == z), None)
        if e is None:
            raise Refusal(f"early_exit: {name}'s profile has no {block} point at z {z}")
        if e.get("cap") is not None and int(e["cap"]) != budget:
            raise Refusal(f"early_exit: {name}'s z {z} point caps at {e['cap']}, the spec says budget {budget}")
        plain = (p.get("budgets") or {}).get(str(budget))
        plotted = (by_name[name].get("budgets") or {}).get(str(budget))
        if plain is None or plotted is None:
            raise Refusal(f"early_exit: {name}'s profiles have no budget {budget}")
        if not own and pin is None:
            raise Refusal(f"early_exit: {name}'s separate calibration needs its callo commit (arms.{name}.commit)")
        out.append({"arm": name, "short": shorts[name], "own": own, "created": p.get("created"),
                    "commit": _pinned(name, pin, p)["profile"] if pin else None,
                    "table": {"acc": float(plotted["acc"]), "p50": float(plotted["lat_p50_ms"][0])},
                    "path": pub(path or paths[name]), "sha256": sha256_file(path or paths[name]),
                    "acc": float(e["acc"]), "n": int(e["n"]), "p50": float(e["lat_p50_ms"][0]),
                    "mean": float(e["lat_mean_ms"][0]), "think_mean": e.get("think_mean"),
                    "gate_share": (e.get("gate") or {}).get("flagged"),
                    "plain": {"acc": float(plain["acc"]), "p50": float(plain["lat_p50_ms"][0]),
                              "mean": float(plain["lat_mean_ms"][0])}})
    return {"z": z, "gate": gate, "budget": budget, "points": out}


def _readout_check(cfg: Any, by_name: dict[str, dict], shorts: dict[str, str], data_sha: str,
                   pub: Any = str) -> dict | None:
    """One model measured on both sides of a readout fix: the plotted profile (after) against `before`, per budget the
    accuracy change and whether the fitted T is the same."""
    if not isinstance(cfg, dict):
        return None
    name = cfg["arm"]
    if name not in by_name:
        raise Refusal(f"readout_check names {name}: not a present callo arm")
    try:
        before = compare.load_profile(cfg["before"])
    except InvalidInput as e:
        raise Refusal(str(e)) from None
    after = by_name[name]
    _same_model(name, "readout_check's before profile", before, after, data_sha)
    if commit_info(cfg["fix"]) is None:
        raise Refusal(f"readout_check.fix: {cfg['fix']} is not a commit of this repository")
    rows = []
    for b in sorted(after.get("budgets") or {}, key=int):
        x, y = (before.get("budgets") or {}).get(b), after["budgets"][b]
        if x is None:
            raise Refusal(f"readout_check: {name}'s before profile has no budget {b}")
        rows.append({"budget": int(b), "before": float(x["acc"]), "after": float(y["acc"]),
                     "same_T": x.get("T") == y.get("T")})
    return {"arm": name, "short": shorts[name], "fix": commit_info(cfg["fix"])["short"], "before": pub(cfg["before"]),
            "sha256": sha256_file(cfg["before"]), "budgets": rows,
            "max_move": max(abs(r["after"] - r["before"]) for r in rows)}


def _source_counts(fixed: list[str], questions: list[dict], order: list[str] | None) -> list[list]:
    src = {q["q"]: q["source"] for q in questions}
    counts: dict[str, int] = {}
    for q in fixed:
        counts[str(src.get(q))] = counts.get(str(src.get(q)), 0) + 1
    keys = [k for k in (order or []) if k in counts] + [k for k in counts if k not in (order or [])]
    return [[k, counts[k]] for k in keys]


def short(a: dict) -> str:
    """An arm's short name for the fine print: its `short`, else its label up to the first " · "."""
    return a.get("short") or a["label"].split(" · ")[0]


def _pp(x: float) -> str:
    return f"{100 * x:+.1f}".replace("-", "−")


def secs(ms: float) -> str:
    """A latency in the plate's prose: one decimal of a second from 1 s, whole ms below."""
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def pick_definition(stat: str) -> str:
    return (f"Callo's pick: the fastest point statistically tied with the most accurate (callo compare default, "
            f"on {stat}).")


BUDGET_REPLAY = ("Budgets are read from one generation per item, cut at each budget: an item that stops thinking early "
                 "reads the same at larger budgets, so p50 and p95 can stack.")


def story(snap: dict, spec: dict, stat: str) -> dict | None:
    """The plate's headline numbers on `stat` (spec `story`: {"model", "think": [B, B'], "vs"}): the model's no-think
    point against the reference (paired Δ and CI, the latency ratio) and where a short think puts it against the
    reference (its paired Δs, one decimal). None without the points it needs; every number is the snapshot's own."""
    cfg = spec.get("story")
    if not cfg or snap["ref"]["acc"] is None:
        return None
    pts = {p["label"]: p for p in snap["points"]}
    zero, ref = pts.get(f"{cfg['model']}@0"), pts.get(snap["ref"]["name"])
    span = [pts.get(f"{cfg['model']}@{b}") for b in cfg["think"]]
    if zero is None or ref is None or None in span or zero["dacc"] is None or any(p["dacc"] is None for p in span):
        return None
    arm = next(a for a in snap["arms"] if a["name"] == cfg["model"])
    ref_arm = next(a for a in snap["arms"] if a["name"] == snap["ref"]["name"])
    d, lo, hi = zero["dacc"]
    ratio = zero["lat"][stat][0] / ref["lat"][stat][0]
    tokens = [p["think_mean"] for p in span if p["think_mean"] is not None]
    tok = round(sum(tokens) / len(tokens), -1) if tokens else None
    gains = sorted(p["dacc"][0] for p in span)
    lats = sorted(p["lat"][stat][0] for p in span)
    ties = lo <= 0 <= hi
    ref_name = cfg.get("ref_short") or short(ref_arm)
    rel = (f"matches {ref_name} ({_pp(d)} pp, CI includes 0)" if ties
           else f"is {_pp(d)} pp from {ref_name} (CI {_pp(lo)} to {_pp(hi)} pp)")
    lat_txt = (f"{lats[0] / 1000:.1f}–{lats[-1] / 1000:.1f}\u00a0s" if round(lats[0], -2) != round(lats[-1], -2)
               else secs(lats[0]).replace(" ", "\u00a0"))           # a no-break space: the unit stays with its number
    # the span's points against the reference (not against no think), at one decimal: "+6.6 to +8.2 pp above"
    span_pp = f"{_pp(gains[0])} to {_pp(gains[-1])} pp" if _pp(gains[0]) != _pp(gains[-1]) else f"{_pp(gains[0])} pp"
    side = "above" if gains[0] >= 0 else "below" if gains[-1] <= 0 else "against"
    sub = (f"{short(arm)} with no thinking {rel} at {ratio:.2f}× its {stat} latency, {cfg['vs']}; "
           + (f"~{tok:.0f} thinking tokens put it " if tok else "a short think puts it ")
           + f"{span_pp} {side} {ref_name} at {lat_txt} {stat}.")
    return {"subtitle": sub, "zero": zero["label"], "ref": ref["label"],
            "span": [p["label"] for p in span], "ratio": ratio, "dacc": zero["dacc"], "ties": ties,
            "engines": _engines(snap, cfg, stat, zero, ref, ref_arm, arm)}


def _engines(snap: dict, cfg: dict, stat: str, zero: dict, ref: dict, ref_arm: dict, arm: dict) -> str | None:
    """When the reference model also ran on its release runtime (`story.release`): how much the engine alone moves
    its latency, so a latency ratio is never read across engines."""
    rel = next((p for p in snap["points"] if p["arm"] == cfg.get("release")), None)
    if rel is None:
        return None
    rel_arm = next(a for a in snap["arms"] if a["name"] == cfg["release"])
    k = rel["lat"][stat][0] / ref["lat"][stat][0]
    return (f"{short(rel_arm)}: {secs(rel['lat'][stat][0])} {stat}, {k:.1f}× the llama.cpp run's "
            f"{secs(ref['lat'][stat][0])}; {short(arm)} without thinking reads {zero['lat'][stat][0] / rel['lat'][stat][0]:.2f}× "
            f"against it, a gap of engines, not of models.")


def fine_print(snap: dict, stat: str) -> list[dict]:
    """The plate's fine print on `stat`, in order: each entry {"lead", "text"} is one line at the plate's width.
    Built here (data layer) so the tests can hold the SVG to it. The plate states GiB and the builds in a line each;
    each arm's runtime details, its run times and files, and the seed are in FIGURES.md (`made`)."""
    arms = [a for a in snap["arms"] if a["status"] == "present" and a.get("on_plate", True)]
    out: list[dict] = []
    host = next((f"{a['host'].removeprefix('Apple ')} {a['memory_gib']} GB" for a in arms if a.get("host")),
                "M2 Ultra 64 GB")
    srcs = ", ".join(f"{k} {n}" for k, n in snap["data"]["sources"])
    spans: dict[str, list[str]] = {}
    for a in arms:
        group = "Callo" if a["kind"] == "callo" else "Clef" if a["timer"] == "client_http" else short(a)
        spans.setdefault(group, []).extend(d for d in (a.get("date"), a.get("date_end")) if d)
    when = "; ".join(f"{g} {min(ds)}" + (f" to {max(ds)}" if max(ds) != min(ds) else "")
                     for g, ds in sorted(spans.items(), key=lambda kv: (max(kv[1] or [""]), min(kv[1] or [""])),
                                         reverse=True) if ds)
    out.append({"lead": "Data.", "text": f"n = {snap['data']['n']} {snap['data']['name']} ({srcs}); idle {host}, "
                                         f"one at a time; {when}."})
    kinds: list[str] = []
    for a in arms:
        if a["timer"] not in kinds:
            kinds.append(a["timer"])
    for k in kinds:
        names = [a for a in arms if a["timer"] == k]
        if k == "callo":
            out.append({"lead": "Timers.", "text": f"{_cache_text(bool(snap['callo_cache']), snap['callo_run'])}; "
                                                   f"{TIMERS['callo']}."})
        elif k == "client_http":
            days = sorted({a.get("date") for a in names if a.get("date")})
            out.append({"lead": f"Clef arms ({', '.join(days)}).",
                        "text": TIMERS[k][0].upper() + TIMERS[k][1:] + "."})
        else:
            for a in names:
                out.append({"lead": f"{short(a)} ({a.get('date')}).", "text": TIMERS[k][0].upper() + TIMERS[k][1:] + "."})
    if snap["ref"]["acc"] is not None:
        strat = "stratified by source" if snap["strata"] == "source" else "unstratified"
        out.append({"lead": "Whiskers.", "text": f"Paired 95 % cluster-bootstrap CIs vs {snap['ref']['label']}, {strat}; "
                                                 f"dashed line = its accuracy; tails run to each point's "
                                                 f"{TAIL_OF[stat]}."})
    else:
        out.append({"lead": "Whiskers.", "text": f"No reference arm ({snap['ref']['name']}) yet: no paired CIs."})
    if snap["picks"].get(stat):
        out.append({"lead": None, "text": pick_definition(stat)})
    if any(a["kind"] == "callo" and a.get("capable") for a in arms):
        out.append({"lead": None, "text": BUDGET_REPLAY})
    gnotes = [a["gib_note"] for a in arms if a.get("gib_note")]
    out.append({"lead": "GiB.", "text": GIB_TEXT + ("; " + "; ".join(gnotes) if gnotes else "") + "."})
    ee = early_exit_text(snap)
    if ee:
        out.append({"lead": "Early exit.", "text": ee})
    v = snap.get("validation")
    if isinstance(v, dict) and v.get("agree") is not None:
        out.append({"lead": "Validation.", "text": validation_text(v)})
    eng = ((snap.get("story") or {}).get(stat) or {}).get("engines")
    if eng:
        out.append({"lead": "Engines.", "text": eng})
    out.append({"lead": "Builds.", "text": builds_text(snap)})
    q4 = snap.get("q4_cost")
    pend = [a.get("short") or a["label"] for a in snap["arms"] if a["status"] == "pending"]
    if not (isinstance(v, dict) and v.get("agree") is not None):
        pend.append("the same-runtime Clef-Flash validation")
    if not q4:
        pend.append("what Q4 costs")
    if pend and snap["draft"]:
        out.append({"lead": "Pending.", "text": "; ".join(pend) + "."})
    return out


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _zt(z: float) -> str:
    """z .95 (the docs' spelling)."""
    return "z " + f"{z:.2f}".lstrip("0")


def early_exit_text(snap: dict) -> str | None:
    """The one fine-print line on early exit: opt-in, experimental, never picked, so never plotted; each named model's
    measured point (accuracy, p50) beside the plain budget of the same calibration, and, when that calibration is a
    separate run, the plotted (table's) plain budget's accuracy too."""
    ee = snap.get("early_exit")
    if not ee or not ee["points"]:
        return None
    b = ee["budget"]
    parts = []
    for p in ee["points"]:
        head = f"{p['short']} {100 * p['acc']:.1f} % at {secs(p['p50'])} p50"
        plain = f"{100 * p['plain']['acc']:.1f} % at {secs(p['plain']['p50'])} p50"
        if p["own"]:
            parts.append(f"{head} (its plain think {b}: {plain})")
        else:   # the table shows means, so only the accuracy is set beside it
            tab = p.get("table") or p["plain"]
            parts.append(f"{head}, from a separate calibration whose plain think {b} reads {plain} (the table's run: "
                         f"{100 * tab['acc']:.1f} %)")
    how = f"{_zt(ee['z'])}{' with the gate' if ee['gate'] else ''}"
    return f"Opt-in and experimental in Callo, never picked, so not plotted. At think {b}, {how}: " + "; ".join(parts) + "."


def builds_text(snap: dict) -> str:
    """The plate's Builds line: the callo version, the builds each callo arm was measured and its profile written on
    (grouped; by role, never by commit), the readout fix the earlier measurements predate and what it moved, and the
    llama.cpp builds."""
    arms = [a for a in snap["arms"] if a["status"] == "present" and a.get("on_plate", True)]
    callo = [a for a in arms if a["kind"] == "callo"]
    versions = sorted({a["callo_version"] for a in callo if a.get("callo_version")})
    groups: dict[tuple, list[str]] = {}
    for a in callo:
        c = a.get("commits")
        if c:
            groups.setdefault((c["measured"], c["profile"], c.get("step")), []).append(short(a))
    b = build_name
    clauses = [f"{_and(ns)} on {b(p)}" if m == p else f"{_and(ns)} measured on {b(m)}, {step or 'profile'} on {b(p)}"
               for (m, p, step), ns in groups.items()]
    clauses += [f"{x['short']}'s early-exit calibration on {b(x['commit'])}"
                for x in ((snap.get("early_exit") or {}).get("points") or []) if x.get("commit")]
    txt = (f"Callo {', '.join(versions)}" if versions else "Callo") + (" development builds: " + "; ".join(clauses)
                                                                       if clauses else "")
    builds: dict[str, list[str]] = {}
    for a in arms:
        if a.get("build"):
            k = "Callo" if a["kind"] == "callo" else "Clef" if a["timer"] == "client_http" else short(a)
            if k not in builds.setdefault(a["build"], []):
                builds[a["build"]].append(k)
    if builds:
        txt += "; llama.cpp " + _and([f"{b} ({', '.join(ks)})" for b, ks in builds.items()])
    rc = snap.get("readout_check")
    if rc and rc.get("predates"):
        before = _and([b(c) for c in rc["predates"]])
        txt += (f". {before[:1].upper() + before[1:]} predate{'s' if len(rc['predates']) == 1 else ''} the readout fix; "
                f"{rc['short']}, measured on both sides of it, moved by at most {100 * rc['max_move']:.1f} pp")
    return txt + "."


def validation_text(v: dict) -> str:
    bar = (f" (pre-registered bar {v['min_agree']}/{v['n']} and {v['max_median_dp']:.3f}: "
           f"{'pass' if v['pass'] else 'FAIL'})" if "min_agree" in v else "")
    return (f"{v.get('label') or 'llama.cpp vs native Clef-Flash'}: {v['agree']}/{v['n']} argmax agreement, median "
            f"|Δp| {v['median_dp']:.4f}{bar}.")


def q4_text(q4: dict) -> str:
    d, lo, hi = q4["dacc"]
    dp = f", median |Δp| {q4['median_dp']:.4f}" if q4.get("median_dp") is not None else ""
    return f"{q4['label']}: Δacc {_pp(d)} pp [{_pp(lo)}, {_pp(hi)}]{dp}."


def named(path: str | None, sha: str | None = None) -> str:
    """A results file as FIGURES.md names it: its file name and sha256 (the bench checkout's layout is not public)."""
    return f"`{Path(str(path)).name}`" + (f" (sha256 {sha[:16]}…)" if sha else "")


def _arm_made(snap: dict, a: dict) -> str:
    """One callo arm's provenance for FIGURES.md: model file, budgets, callo version and commits, --cache, llama.cpp
    build, run and profile times, the opt-in points left off the plate."""
    c = a.get("commits") or {}
    if c and c["measured"] != c["profile"]:
        at = f", measured on {build_name(c['measured'])}, {c.get('step') or 'profile'} on {build_name(c['profile'])}"
    elif c:
        at = f", on {build_name(c['profile'])}"
    else:
        at = " commit not pinned"
    run = a.get("run") or {}
    opt = (f"; opt-in points in the profile, not plotted: {', '.join(a['opt_in'])}" if a.get("opt_in") else "")
    return (f"`{a.get('model_file')}`; budgets {', '.join(map(str, a.get('budgets') or []))}; `callo` "
            f"{a.get('callo_version')}{at}; --cache {'on (default)' if a.get('cache') else 'off'}; llama.cpp "
            f"{a.get('build')}; run {run.get('started')} to {run.get('ended')}, profile written {a.get('created')}"
            f"{opt}.")


def made(snap: dict) -> list[dict]:
    """How the plate was made, for FIGURES.md (not the plate): builds, GiB, each arm's runtime note and provenance,
    the early-exit points and the readout check in full, seed, commit, callo compare's flags."""
    arms = [a for a in snap["arms"] if a["status"] == "present"]
    builds: dict[str, list[str]] = {}
    for a in arms:
        if a.get("build"):
            builds.setdefault(a["build"], []).append(short(a))
    versions = sorted({a.get("callo_version") for a in arms if a.get("callo_version")})
    out = [{"lead": "Builds.", "text": "; ".join([f"`callo` {', '.join(versions)} profiles"] if versions else [])
            + "".join(f"; llama.cpp {b} for {', '.join(ns)}" for b, ns in builds.items()) + "."},
           {"lead": "GiB.", "text": GIB_LINE}]
    for a in arms:
        if a["kind"] == "callo":
            out.append({"lead": f"{a['label']}.", "text": _arm_made(snap, a)})
    for a in arms:
        if a.get("note"):
            out.append({"lead": None, "text": a["note"]})
    ee = snap.get("early_exit")
    for p in (ee or {}).get("points") or []:
        src = ("its plotted profile" if p["own"] else
               f"its own early-exit calibration, {named(p['path'], p['sha256'])}, written {p['created']}")
        gate = (f"; the gate sent {100 * p['gate_share']:.0f} % of questions to full thinking"
                if p.get("gate_share") is not None else "")
        out.append({"lead": f"Early exit, {p['short']} (not plotted).",
                    "text": f"think {ee['budget']}, {_zt(ee['z'])}{' with the gate' if ee['gate'] else ''}, from {src}: "
                            f"accuracy {100 * p['acc']:.1f} %, p50 {secs(p['p50'])}, mean {secs(p['mean'])}{gate}; "
                            f"plain think {ee['budget']} there: {100 * p['plain']['acc']:.1f} %, p50 "
                            f"{secs(p['plain']['p50'])}, mean {secs(p['plain']['mean'])}. Opt-in and experimental; "
                            "Callo never picks it."})
    rc = snap.get("readout_check")
    if rc:
        steps = ", ".join(f"{100 * r['before']:.1f} → {100 * r['after']:.1f} % at {r['budget']}" for r in rc["budgets"])
        same = all(r["same_T"] for r in rc["budgets"])
        out.append({"lead": "Readout fix.",
                    "text": f"{rc['short']} measured before it, {named(rc['before'], rc['sha256'])}, and after it (plotted) "
                            f"on {build_name(rc['fix'])}: accuracy {steps}; temperature fits "
                            f"{'identical at every budget' if same else 'differ'}."
                            + (f" Arms measured on {_and([build_name(c) for c in rc['predates']])} predate it."
                               if rc.get("predates") else "")})
    for a in arms:
        if a["kind"] != "callo":
            t = (a.get("meta") or {}).get("timer")
            if t:
                out.append({"lead": f"{short(a)} timer (rows meta).", "text": t})
    for a in arms:
        if not a.get("on_plate", True):
            p = next(q for q in snap["points"] if q["arm"] == a["name"])
            dacc = (f"; vs {snap['ref']['label']} Δacc {_pp(p['dacc'][0])} pp [{_pp(p['dacc'][1])}, {_pp(p['dacc'][2])}]"
                    if p["dacc"] is not None else "")
            out.append({"lead": f"{a['label']} (measured, not drawn).",
                        "text": f"accuracy {100 * p['acc']:.1f} %, mean {secs(p['lat']['mean'][0])}, p95 "
                                f"{secs(p['lat']['p95'][0])}, {a['gib']:.1f} GiB, {a.get('runtime')}, {a.get('date')}"
                                f"{dacc}."})
    if snap.get("q4_cost"):
        out.append({"lead": "What Q4 costs a Clef model.", "text": q4_text(snap["q4_cost"])})
    if snap.get("validation"):
        out.append({"lead": "Validation.", "text": validation_text(snap["validation"])
                    + (f" Source: {named(snap['validation']['path'], snap['validation'].get('sha256'))}."
                       if snap["validation"].get("path") else "")})
    out.append({"lead": "Statistics.", "text": f"seed {snap['seed']}; one row set ({snap['data']['n']} questions) for every "
                "paired interval, as `callo compare`; per-model frontiers by think.frontier over each model's thinking "
                "budgets; the pick by compare.joint."})
    if snap["flags"]:
        text = _file_names("; ".join(f.replace("\n", "\\n") for f in snap["flags"]))     # a local path: its file name
        out.append({"lead": "`callo compare` flags.", "text": text + "."})
    return out


# ---- drawing -----------------------------------------------------------------------------------------------------

W = 1280                       # the plate's width (the brand plate's)
SIDE = 64                      # side margin
PLOT_X = (112, W - SIDE)       # the plot box's x range; its top follows the header
PLOT_H = 620
ROW_H = 26                     # the table's row pitch
TAIL_OF = {"mean": "p95", "p50": "p95", "p95": "p99"}


def _f(v: float) -> str:
    return repr(float(v))


def _title_runs(title: str, st_sans, st_serif) -> list:
    """`*...*` marks the Newsreader italic part of a title (the brand's second voice)."""
    return [(part, st_serif if i % 2 else st_sans) for i, part in enumerate(title.split("*")) if part]


def _budget_text(p: dict) -> str:
    import figkit
    if p["kind"] == "budget":
        return "no think" if not p["budget"] else f"think {p['budget']}"
    if p["kind"] == "deadline":
        return f"deadline {figkit.fmt_ms(p['value'])}"
    return f"early exit {p['value']}"


def _box_attr(b) -> str:
    return ",".join(_f(v) for v in (b.x0, b.y0, b.x1, b.y1))


def styles(arms: list[dict], by_arm: dict[str, list[dict]], pick_arm: str | None, t: dict) -> dict[str, dict]:
    """Series styles: Signal (the brand's one accent) for the callo model holding callo's pick; the other callo models
    in the secondary colour, thin, solid then dashed so they do not compete (a one-point model draws no line and takes
    no dash); the trained heads and Clef in the type colour, single markers (filled, or hollow for a second runtime)."""
    out: dict[str, dict] = {}
    others = iter([(1.5, None), (1.3, "6 3"), (1.3, "2 3"), (1.3, "8 3 2 3"), (1.3, "1 3")])
    for a in arms:
        m = a.get("marker")
        if a["kind"] == "callo" and a["name"] == pick_arm:
            out[a["name"]] = {"color": t["accent"], "w": 2.4, "dash": None, "marker": m or "circle",
                              "fill": t["accent"], "name": t["accent"]}
        elif a["kind"] == "callo":
            w, dash = next(others, (1.3, "1 3")) if len(by_arm[a["name"]]) > 1 else (1.3, None)
            out[a["name"]] = {"color": t["sub"], "w": w, "dash": dash, "marker": m or "circle", "fill": t["ground"],
                              "name": t["ink"]}
        else:
            m = m or "diamond"
            out[a["name"]] = {"color": t["ref"], "w": 1.6, "dash": None, "marker": m,
                              "fill": t["ground"] if m.endswith("hollow") else t["ref"], "name": t["ref"]}
    return out


def _legend(fk, t: dict, right: float, y: float, stat: str, ref_label: str | None) -> str:
    """The key, right-aligned above the plot, in the marks themselves (the brand faces have no ● or ┤): the point's
    statistic, its tail, the whisker, and what the numbers at the callo points are."""
    st = fk.Style(size=13, fill=t["sub"])
    items = [(stat, "dot"), (TAIL_OF[stat], "tail")]
    if ref_label:
        items.append((f"paired 95 % CI vs {ref_label}", "whisker"))
    items.append(("think budget, tokens", "budget"))
    num_st = fk.Style(size=12.5, fill=t["sub"])
    glyph = {"dot": 12, "tail": 22, "whisker": 10, "budget": fk.advance("256", num_st)}
    widths = [fk.advance(s, st) for s, _ in items]
    total = sum(glyph[k] + 6 + w for (_, k), w in zip(items, widths, strict=True)) + 18 * (len(items) - 1)
    x = right - total
    out = [f'<g data-role="legend" data-text={fk.esc(" / ".join(s for s, _ in items))}>']
    cy = y - 4.5
    c = t["sub"]
    for (s, k), w in zip(items, widths, strict=True):
        if k == "dot":
            out.append(f'<circle cx="{_f(x + 5)}" cy="{_f(cy)}" r="4" fill="{c}"/>')
        elif k == "tail":
            out.append(f'<line x1="{_f(x)}" x2="{_f(x + 20)}" y1="{_f(cy)}" y2="{_f(cy)}" stroke="{c}" stroke-width="1"/>'
                       f'<line x1="{_f(x + 20)}" x2="{_f(x + 20)}" y1="{_f(cy - 3.5)}" y2="{_f(cy + 3.5)}" stroke="{c}"/>')
        elif k == "whisker":
            out.append(f'<line x1="{_f(x + 5)}" x2="{_f(x + 5)}" y1="{_f(cy - 7)}" y2="{_f(cy + 7)}" stroke="{c}" '
                       f'stroke-width="1.2"/><line x1="{_f(x + 2)}" x2="{_f(x + 8)}" y1="{_f(cy - 7)}" y2="{_f(cy - 7)}" '
                       f'stroke="{c}" stroke-width="1.2"/><line x1="{_f(x + 2)}" x2="{_f(x + 8)}" y1="{_f(cy + 7)}" '
                       f'y2="{_f(cy + 7)}" stroke="{c}" stroke-width="1.2"/>')
        else:
            out.append(fk.text_paths("256", x, y, num_st))
        x += glyph[k] + 6
        out.append(fk.text_paths(s, x, y, st))
        x += w + 18
    out.append("</g>")
    return "".join(out)


def _table(fk, t: dict, snap: dict, arms: list[dict], by_arm: dict[str, list[dict]], sty: dict[str, dict], stat: str,
           pick: str | None, top: float) -> tuple[str, float]:
    """Every plotted point as numbers: a row per model, a column per think budget, each cell the accuracy and the
    latency on the plate's statistic (a trained head or Clef answers in one pass: its no-think column); the GiB."""
    head_st = fk.Style(size=12, weight=600, fill=t["sub"], tracking=1.0)
    lab_st = fk.Style(size=14, weight=500, fill=t["ink"])
    tag_st = fk.Style(size=13, fill=t["sub"])
    num_st = fk.Style(size=14, weight=600, fill=t["ink"])
    lat_st = fk.Style(size=14, fill=t["sub"])
    callo = [a for a in arms if a["kind"] == "callo"]
    budgets = sorted({p["budget"] for a in callo for p in by_arm[a["name"]]}) or [0]
    x_gib = 560
    col_w = (W - SIDE - x_gib) / len(budgets)
    cols = {b: x_gib + col_w * (i + 1) for i, b in enumerate(budgets)}
    out = ['<g data-role="table">']
    out.append(fk.text("MODEL", SIDE + 22, top, head_st, role="table-head")[0])
    out.append(fk.text("GIB", x_gib, top, head_st, anchor="end", role="table-head")[0])
    for b, x in cols.items():
        out.append(fk.text("NO THINK" if not b else f"THINK {b}", x, top, head_st, anchor="end", role="table-head")[0])
    y = top + 12
    out.append(f'<line x1="{SIDE}" x2="{W - SIDE}" y1="{_f(y)}" y2="{_f(y)}" stroke="{t["axis"]}"/>')
    ref_name = (snap.get("ref") or {}).get("name")
    for a in [*callo, *(a for a in arms if a["kind"] != "callo")]:
        st = sty[a["name"]]
        yb = y + ROW_H - 9
        mk, _ = fk.marker(st["marker"], SIDE + 7, yb - 5, st["fill"], st["color"], 1.5, scale=0.85)
        out.append(f'<g data-role="table-row" data-arm={fk.esc(a["name"])}>{mk}')
        out.append(fk.text(a["label"], SIDE + 22, yb, lab_st, role="table-label")[0])
        tag = " · ".join(x for x in (a.get("runtime"), "reference" if a["name"] == ref_name else None) if x)
        if tag:
            out.append(fk.text(tag, SIDE + 22 + fk.advance(a["label"], lab_st) + 10, yb, tag_st, role="table-tag")[0])
        out.append(fk.text(f"{a['gib']:.1f}", x_gib, yb, lat_st, anchor="end", role="table-cell")[0])
        own = {p["budget"] if a["kind"] == "callo" else 0: p for p in by_arm[a["name"]]}
        for b, x in cols.items():
            p = own.get(b)
            if p is None:
                out.append(fk.text("–", x, yb, lat_st, anchor="end", role="table-empty")[0])
                continue
            lat = secs(p["lat"][stat][0])
            w_lat = fk.advance(lat, lat_st)
            acc = f"{100 * p['acc']:.1f} %"
            ast = num_st.with_(fill=t["accent"]) if p["label"] == pick else num_st
            out.append(f'<g data-role="table-cell" data-label={fk.esc(p["label"])} data-acc="{_f(p["acc"])}" '
                       f'data-lat-ms="{_f(p["lat"][stat][0])}">'
                       f'{fk.text_paths(acc, x - w_lat - 10 - fk.advance(acc, ast), yb, ast)}'
                       f'{fk.text_paths(lat, x - w_lat, yb, lat_st)}</g>')
        out.append("</g>")
        y += ROW_H
        out.append(f'<line x1="{SIDE}" x2="{W - SIDE}" y1="{_f(y)}" y2="{_f(y)}" stroke="{t["grid"]}"/>')
    out.append("</g>")
    return "".join(out), y


def draw(snap: dict, *, theme: str = "dark", stat: str | None = None) -> str:
    """The static plate: x = the pick statistic's latency (log), with a tail to p95 (p99 on the p95 plate); y = accuracy
    with the paired CI vs the reference as a whisker; each callo model's own frontier as a line through its thinking
    budgets; trained heads and Clef as single markers; direct labels, callo's pick called out; a table of every point;
    the fine print. Every mark carries its data."""
    import figkit as fk

    stat = stat or snap.get("latency") or "mean"
    tail = TAIL_OF[stat]
    t = fk.THEMES[theme]
    arms = [a for a in snap["arms"] if a["status"] == "present" and a.get("on_plate", True)]
    pts = [p for p in snap["points"] if p["arm"] in {a["name"] for a in arms}]
    by_arm = {a["name"]: [p for p in pts if p["arm"] == a["name"]] for a in arms}

    body: list[str] = []
    # ---- header: badge, title, subtitle (the story's, on this statistic, when there is one)
    title = snap.get("title") or "Callo against trained decision heads"
    if snap.get("draft"):
        built = (snap.get("generator") or {}).get("built", "")[:10]
        bst = fk.Style(size=11, weight=600, fill=t["sub"], tracking=1.4)
        btxt = f"DRAFT · {built}"
        w = fk.advance(btxt, bst)
        body.append(f'<rect x="{SIDE}" y="40" width="{fk.num(w + 20)}" height="22" rx="2" fill="none" '
                    f'stroke="{t["sub"]}" stroke-opacity="0.6"/>')
        body.append(fk.text(btxt, SIDE + 10, 55.5, bst, role="badge")[0])
    tsans = fk.Style(size=34, weight=600, fill=t["ink"], width=96)
    tserif = fk.Style(size=38, weight=500, fill=t["ink"], face="serif")
    head = fk.paragraph(_title_runs(title, tsans, tserif), SIDE, 108, W - 2 * SIDE, leading=42, role="title")
    body.append(head.svg)
    story_ = (snap.get("story") or {}).get(stat)
    subtitle = story_["subtitle"] if story_ else snap.get("subtitle")
    y_head = head.box.y1
    if subtitle:
        sub = fk.paragraph([(subtitle, fk.Style(size=16, fill=t["sub"]))], SIDE, y_head + 30, W - 2 * SIDE,
                           leading=23, role="subtitle")
        body.append(sub.svg)
        y_head = sub.box.y1
    x0, x1 = PLOT_X
    y0 = math.ceil(y_head + 70)
    y1 = y0 + PLOT_H
    plot = fk.Box(x0, y0, x1, y1)

    pick = snap["picks"].get(stat)
    pick_arm = next((p["arm"] for p in pts if p["label"] == pick), None)
    callo_arms = [a for a in arms if a["kind"] == "callo"]
    if pick_arm is None and callo_arms:
        pick_arm = callo_arms[0]["name"]
    style = styles(arms, by_arm, pick_arm, t)

    # scales: log latency over the 1-2-5 values around every point and tail; accuracy over every point and whisker
    def tail_ms(p: dict) -> float:
        return p["lat"][tail][0]

    lat_lo = min(p["lat"][stat][0] for p in pts)
    lat_hi = max(max(p["lat"][stat][0], tail_ms(p)) for p in pts)
    d0, d1 = fk.log_bounds(lat_lo / 1.15, lat_hi * 1.08)
    xs = fk.LogScale(d0, d1, x0 + 8, x1 - 8)

    def ends(p: dict) -> tuple[float, float]:
        if p["dacc"] is None:
            return p["acc"], p["acc"]
        d, lo, hi = p["dacc"]
        return p["acc"] + (lo - d), p["acc"] + (hi - d)

    a_lo = min(min(ends(p)) for p in pts)
    a_hi = max(max(ends(p)) for p in pts)
    span = max(a_hi - a_lo, 0.04)
    step = 0.02 if span <= 0.16 else 0.04 if span <= 0.32 else 0.1
    g_lo = math.floor((a_lo - 0.008) / step) * step
    g_hi = math.ceil((a_hi + 0.03) / step) * step       # headroom for the labels above the top points
    if a_hi <= 1.0:
        g_hi = min(g_hi, 1.0)                            # no axis above 100 %
    ys = fk.LinScale(g_lo, g_hi, y1 - 6, y0 + 6)

    def X(p: dict) -> float:
        return xs(p["lat"][stat][0])

    def Y(p: dict) -> float:
        return ys(p["acc"])

    # ---- grid, axes, key
    tick_st = fk.Style(size=13, fill=t["sub"])
    for v in fk.log_ticks(d0, d1):
        x = xs(v)
        body.append(f'<line x1="{fk.num(x)}" x2="{fk.num(x)}" y1="{y0}" y2="{y1}" stroke="{t["grid"]}"/>')
        body.append(fk.text(fk.fmt_ms(v), x, y1 + 24, tick_st, anchor="middle", role="tick")[0])
    k = 0
    while g_lo + k * step <= g_hi + 1e-9:
        a = g_lo + k * step
        y = ys(a)
        body.append(f'<line x1="{x0}" x2="{x1}" y1="{fk.num(y)}" y2="{fk.num(y)}" stroke="{t["grid"]}"/>')
        body.append(fk.text(f"{100 * a:.0f} %", x0 - 12, y + 4.5, tick_st, anchor="end", role="tick")[0])
        k += 1
    body.append(f'<line x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}" stroke="{t["axis"]}"/>')
    ax_st = fk.Style(size=13.5, fill=t["sub"])
    body.append(fk.text(f"latency per decision, {stat}  (log scale)  →", (x0 + x1) / 2, y1 + 52, ax_st,
                        anchor="middle", role="axis-title")[0])
    body.append(fk.text("accuracy  ↑", SIDE, y0 - 16, ax_st, role="axis-title")[0])
    ref = snap.get("ref") or {}
    body.append(_legend(fk, t, x1, y0 - 16, stat, ref.get("label") if ref.get("acc") is not None else None))

    placer = fk.Placer(bounds=plot.pad(-2))
    marks: list[str] = []
    lines: list[str] = []
    labels: list[str] = []

    def segment(ax_: float, ay_: float, bx_: float, by_: float, pad: float = 2.0) -> None:
        """A line as placer obstacles: small boxes along it (a diagonal is not one big box)."""
        n = max(1, int(math.hypot(bx_ - ax_, by_ - ay_) / 6))
        for i in range(n + 1):
            x, y = ax_ + (bx_ - ax_) * i / n, ay_ + (by_ - ay_) * i / n
            placer.lines.append(fk.Box(x - pad, y - pad, x + pad, y + pad))

    # ---- reference line
    if ref.get("acc") is not None:
        y = ys(ref["acc"])
        lines.append(f'<line data-role="ref" data-acc="{_f(ref["acc"])}" x1="{x0}" x2="{x1}" y1="{_f(y)}" y2="{_f(y)}" '
                     f'stroke="{t["ref"]}" stroke-opacity="0.45" stroke-width="1" stroke-dasharray="2 5"/>')
        placer.lines.append(fk.Box(x0, y - 2, x1, y + 2))

    # ---- tails (one per arm and tail end: equal ends, as merged budgets have, draw once), whiskers; both are thin and
    # faint, so a label may cross one on its ground halo (`fit`), never a marker, a line or another label
    thin: list = []
    shared: dict[str, str] = {}
    seen: dict[tuple[str, float], str] = {}
    for p in sorted(pts, key=lambda p: -p["acc"]):
        key = (p["arm"], round(tail_ms(p), 6))
        if key in seen:
            shared[p["label"]] = seen[key]
        else:
            seen[key] = p["label"]
    for p in pts:
        st = style[p["arm"]]
        x, y = X(p), Y(p)
        op = 0.2 if not p["frontier"][stat] else 0.42
        if p["label"] not in shared:
            xt = xs(tail_ms(p))
            lines.append(f'<line data-role="tail" data-label={fk.esc(p["label"])} data-stat="{tail}" '
                         f'data-ms="{_f(tail_ms(p))}" x1="{_f(x)}" x2="{_f(xt)}" y1="{_f(y)}" y2="{_f(y)}" '
                         f'stroke="{st["color"]}" stroke-opacity="{op}" stroke-width="1"/>')
            lines.append(f'<line x1="{_f(xt)}" x2="{_f(xt)}" y1="{_f(y - 3.5)}" y2="{_f(y + 3.5)}" stroke="{st["color"]}" '
                         f'stroke-opacity="{op}" stroke-width="1"/>')
            thin.append(fk.Box(x, y - 2, xt, y + 2))
            placer.lines.append(thin[-1])
        if p["dacc"] is not None:
            lo, hi = ends(p)
            ylo, yhi = ys(lo), ys(hi)
            lines.append(f'<line data-role="whisker" data-label={fk.esc(p["label"])} data-lo="{_f(lo)}" data-hi="{_f(hi)}" '
                         f'x1="{_f(x)}" x2="{_f(x)}" y1="{_f(ylo)}" y2="{_f(yhi)}" stroke="{st["color"]}" '
                         f'stroke-opacity="{op + 0.16}" stroke-width="1.2"/>')
            for yy in (ylo, yhi):
                lines.append(f'<line x1="{_f(x - 3)}" x2="{_f(x + 3)}" y1="{_f(yy)}" y2="{_f(yy)}" stroke="{st["color"]}" '
                             f'stroke-opacity="{op + 0.16}" stroke-width="1.2"/>')
            thin.append(fk.Box(x - 3, yhi, x + 3, ylo))
            placer.lines.append(thin[-1])

    # ---- each callo model's own frontier: a line through its frontier points in latency order (the accent on top)
    own_line: dict[str, tuple[int, int]] = {}
    fronts: dict[str, list[dict]] = {}
    for a in sorted(callo_arms, key=lambda a: a["name"] == pick_arm):
        st = style[a["name"]]
        front = sorted((p for p in by_arm[a["name"]] if p["frontier"][stat]), key=lambda p: (p["lat"][stat][0], -p["acc"]))
        fronts[a["name"]] = front
        if len(front) < 2:
            continue
        coords = [(X(p), Y(p)) for p in front]
        n0 = len(placer.lines)
        for (ax_, ay_), (bx_, by_) in zip(coords, coords[1:], strict=False):
            segment(ax_, ay_, bx_, by_, pad=1.5)
        own_line[a["name"]] = (n0, len(placer.lines))
        dash = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ""
        lines.append(f'<polyline data-role="frontier" data-arm={fk.esc(a["name"])} '
                     f'data-labels={fk.esc(",".join(p["label"] for p in front))} '
                     f'points="{" ".join(f"{_f(x)},{_f(y)}" for x, y in coords)}" fill="none" stroke="{st["color"]}" '
                     f'stroke-width="{st["w"]}" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        # a line that runs through another series' marker breaks well clear of it, so it reads as passing under the
        # marker, not as a vertex of the line
        for q in pts:
            if q["arm"] == a["name"]:
                continue
            _, r = fk.marker(style[q["arm"]]["marker"], 0, 0, "none", "none", 1, big=q["arm"] == pick_arm)
            qx, qy = X(q), Y(q)
            for (ax_, ay_), (bx_, by_) in zip(coords, coords[1:], strict=False):
                dx, dy = bx_ - ax_, by_ - ay_
                seg = math.hypot(dx, dy)
                if not seg:
                    continue
                u = max(0.0, min(1.0, ((qx - ax_) * dx + (qy - ay_) * dy) / seg ** 2))
                cx, cy = ax_ + u * dx, ay_ + u * dy
                if math.hypot(qx - cx, qy - cy) <= r + 1.5:
                    g = r + 6.0
                    ux, uy = dx / seg, dy / seg
                    lines.append(f'<line data-role="line-gap" data-arm={fk.esc(a["name"])} data-at={fk.esc(q["label"])} '
                                 f'x1="{_f(cx - ux * g)}" y1="{_f(cy - uy * g)}" x2="{_f(cx + ux * g)}" '
                                 f'y2="{_f(cy + uy * g)}" stroke="{t["ground"]}" stroke-width="{_f(st["w"] + 3.5)}"/>')

    # ---- points: every ground halo first (a line visibly breaks at a marker), then every marker, dominated ones
    # faint; the pick ringed
    halos = ['<g data-role="halos">']
    for p in sorted(pts, key=lambda p: (p["frontier"][stat], p["arm"] == pick_arm)):
        st = style[p["arm"]]
        x, y = X(p), Y(p)
        faint = not p["frontier"][stat]
        big = p["arm"] == pick_arm
        halo, _ = fk.marker(st["marker"], x, y, t["ground"], t["ground"], 1.7 + 4.5, big=big)
        shape, r = fk.marker(st["marker"], x, y, t["ground"] if faint else st["fill"], st["color"], 1.7, big=big)
        halos.append(halo)
        placer.marks.append(fk.Box(x - r - 2, y - r - 2, x + r + 2, y + r + 2))
        dacc = ",".join(_f(v) for v in p["dacc"]) if p["dacc"] is not None else ""
        sh = f' data-tail-shared={fk.esc(shared[p["label"]])}' if p["label"] in shared else ""
        op = ' opacity="0.45"' if faint else ""
        marks.append(f'<g data-role="point" data-label={fk.esc(p["label"])} data-arm={fk.esc(p["arm"])} '
                     f'data-acc="{_f(p["acc"])}" data-lat-ms="{_f(p["lat"][stat][0])}" data-dacc="{dacc}" '
                     f'data-frontier="{1 if p["frontier"][stat] else 0}" data-x="{_f(x)}" data-y="{_f(y)}"{sh}{op}>'
                     f'{shape}</g>')
    marks.insert(0, "".join(halos) + "</g>")
    pick_pt = next((q for q in pts if q["label"] == pick), None)
    if pick_pt is not None:
        px, py = X(pick_pt), Y(pick_pt)
        marks.append(f'<circle data-role="pick" data-label={fk.esc(pick)} cx="{_f(px)}" cy="{_f(py)}" r="11" '
                     f'fill="none" stroke="{t["accent"]}" stroke-opacity="0.6" stroke-width="1.2"/>')
        placer.marks.append(fk.Box(px - 12, py - 12, px + 12, py + 12))

    # ---- label styles (README width: nothing under 12.5 px)
    name_st = fk.Style(size=14.5, weight=600)
    pst = fk.Style(size=12.5, fill=t["sub"])

    def leader(box, ax_: float, ay_: float, color: str, *, min_len: float = 16) -> None:
        cx = min(max(ax_, box.x0), box.x1)
        cy = min(max(ay_, box.y0), box.y1)
        dist = math.hypot(cx - ax_, cy - ay_)
        if dist > min_len:
            k_ = 9 / dist
            sx, sy = ax_ + (cx - ax_) * k_, ay_ + (cy - ay_) * k_
            labels.append(f'<line x1="{_f(sx)}" y1="{_f(sy)}" x2="{_f(cx)}" y2="{_f(cy)}" stroke="{color}" '
                          f'stroke-opacity="0.45" stroke-width="0.8"/>')
            segment(sx, sy, cx, cy, pad=1.5)

    thin_ids = {id(b) for b in thin}

    def fit(box, margin: float, others: list) -> float:
        """How well a label box sits: inf when it leaves the plot or comes within `margin` px of a label, a marker or a
        line in `others` (more room left and right: two labels on one baseline must not touch); otherwise the area it
        covers of thin tails and whiskers, which its ground halo knocks out."""
        if not box.inside(placer.bounds):
            return math.inf
        wide_ = fk.Box(box.x0 - margin - 20, box.y0 - margin, box.x1 + margin + 20, box.y1 + margin)
        if any(wide_.overlap(o) for o in placer.labels) or any(wide_.overlap(o) for o in placer.marks):
            return math.inf
        soft = 0.0
        near = box.pad(min(margin, 4.0))
        for o in others:
            if id(o) in thin_ids:
                soft += box.overlap(o)
            elif near.overlap(o):
                return math.inf
        return soft

    marker_at = [(X(q), Y(q), q["label"], q["arm"]) for q in pts]

    def owned(box, label: str, slack: float = 2.0, *, own_line: bool = True) -> bool:
        """The box is nearer its own marker than any other: a label never reads as a neighbour's. With
        own_line=False only other series' markers count (its own line's next budget, a few pixels away, may be as
        near: the line orders them)."""
        def dist(x: float, y: float) -> float:
            return math.hypot(max(box.x0 - x, 0, x - box.x1), max(box.y0 - y, 0, y - box.y1))
        mine, arm = next((dist(x, y), a) for x, y, lab, a in marker_at if lab == label)
        return all(mine + slack < dist(x, y) for x, y, lab, a in marker_at if lab != label and (own_line or a != arm))

    def halo(svg: str) -> str:
        """Text on a ground halo: the same outlines stroked in the ground colour underneath, so a thin line that
        passes under a label breaks around it."""
        return re.sub(r'fill="[^"]+"', f'fill="{t["ground"]}" stroke="{t["ground"]}" stroke-width="4" '
                                       f'stroke-linejoin="round"', svg) + svg

    # ---- callo's pick: a two-line callout above-left with its own numbers on this statistic (placed first)
    if pick_pt is not None:
        l1 = f"Callo's pick · {_budget_text(pick_pt)} · {100 * pick_pt['acc']:.1f} %"
        second = "p95" if stat != "p95" else "mean"
        l2 = f"{secs(pick_pt['lat'][stat][0])} {stat} · {second} {secs(pick_pt['lat'][second][0])}"
        c1, c2 = fk.Style(size=14, weight=600, fill=t["accent"]), fk.Style(size=13, fill=t["sub"])
        w = max(fk.advance(l1, c1), fk.advance(l2, c2))
        box = placer.place(w, 36, px, py, gaps=(14, 24, 36, 52, 72, 96, 120), prefer=("nw", "n", "w"),
                           scan=220, quadrant="nw")
        leader(box, px, py, t["accent"], min_len=12)
        labels.append(f'<g data-role="pick-label" data-label={fk.esc(pick)} data-text={fk.esc(l1 + " / " + l2)} '
                      f'data-box="{_box_attr(box)}">{halo(fk.text_paths(l1, box.x0, box.y0 + 14, c1))}'
                      f'{halo(fk.text_paths(l2, box.x0, box.y0 + 32, c2))}</g>')

    # ---- a callo model's label (its name: runtime and GiB are in the table) hugs its own line, just above or below
    # it, else sits just past its last point, with clear ground all round; off its line, a leader says whose it is
    def line_label(a: dict) -> None:
        st = style[a["name"]]
        own = fronts[a["name"]] or by_arm[a["name"]]
        coords = sorted(((X(p), Y(p)) for p in own), key=lambda c: c[0])
        line1 = a["label"]
        n1 = name_st.with_(fill=st["name"])
        w_ = fk.advance(line1, n1)
        h_, gap = 18.0, 6.0
        i0, i1 = own_line.get(a["name"], (0, 0))
        others = placer.lines[:i0] + placer.lines[i1:]

        def y_span(bx0: float, bx1: float) -> tuple[float, float] | None:
            """The line's lowest and highest y over [bx0, bx1] (None where the box leaves the line's x range)."""
            if bx0 < coords[0][0] - 1 or bx1 > coords[-1][0] + 1:
                return None
            ysv = [y for x, y in coords if bx0 <= x <= bx1]
            for (ax_, ay_), (bx_, by_) in zip(coords, coords[1:], strict=False):
                for xq in (bx0, bx1):
                    if ax_ <= xq <= bx_ and bx_ > ax_:
                        ysv.append(ay_ + (by_ - ay_) * (xq - ax_) / (bx_ - ax_))
            return (min(ysv), max(ysv)) if ysv else None

        best, best_c, side = None, math.inf, "free"
        want = a.get("label_side")                    # the spec may art-direct the side: others then cost more

        def pen(sd: str) -> float:
            return 0.0 if want is None or want == sd or (want == "along" and sd in ("above", "below")) else 400.0

        if len(coords) == 1:                          # one point, no line: beside its marker, to the left first
            ax_, ay_ = coords[0]
            best = placer.place(w_, h_, ax_, ay_, gaps=(8, 12, 18, 26, 40),
                                prefer=("w", "sw", "nw", "s", "n", "e", "se", "ne"), scan=120)
            leader(best, ax_, ay_, st["color"], min_len=22)
            labels.append(f'<g data-role="arm-label" data-arm={fk.esc(a["name"])} data-side="marker" '
                          f'data-text={fk.esc(line1)} data-box="{_box_attr(best)}">'
                          f'{halo(fk.text_paths(line1, best.x0, best.y0 + 14, n1))}</g>')
            return
        if len(coords) > 1:
            lx0, lx1 = coords[0][0], coords[-1][0]
            for margin in (8.0, 3.0):
                for k_ in range(81):
                    bx0 = lx0 + (lx1 - lx0 - w_) * k_ / 80      # a line shorter than its label: the label overhangs
                    ysp = y_span(max(bx0, lx0), min(bx0 + w_, lx1))
                    if ysp is None:
                        continue
                    for sd in ("above", "below"):
                        top = ysp[0] - gap - h_ if sd == "above" else ysp[1] + gap
                        box = fk.Box(bx0, top, bx0 + w_, top + h_)
                        soft = fit(box, margin, placer.lines[i0:i1] + others)
                        if soft == math.inf:
                            continue
                        c = (20 + abs(k_ / 80 - 0.5) * 60 + (0 if sd == "above" else 8) + (0 if margin > 5 else 30)
                             + 0.06 * soft + pen(sd))
                        if c < best_c:
                            best, best_c, side = box, c, sd
                if best is not None:
                    break
        # or before its first point, level with it (the line starts there)
        sx_, sy_ = coords[0]
        for dx in range(10, 130, 4):
            for dy in range(-60, 42, 2):
                box = fk.Box(sx_ - dx - w_, sy_ + dy - h_ / 2, sx_ - dx, sy_ + dy + h_ / 2)
                soft = fit(box, 4.0, others)
                if soft == math.inf:
                    continue
                c = dx + 1.4 * abs(dy) + 30 + 0.06 * soft + pen("start")
                if c < best_c:
                    best, best_c, side = box, c, "start"
        # or past its last point, a little above it, or (at the plot's right edge) over it, ending past it (the
        # nearest wins)
        ex, ey = coords[-1]
        for dx in range(-int(w_) + 24, 260, 4):
            for dy in range(-64, 30, 2):
                box = fk.Box(ex + dx, ey + dy - h_ / 2, ex + dx + w_, ey + dy + h_ / 2)
                if dx < 10 and box.y1 > ey - 12:
                    continue
                soft = fit(box, 4.0, placer.lines[i0:i1] + others)
                if soft == math.inf:
                    continue
                c = abs(dx) + abs(dy + 8) + (12 if dx < 10 else 0) + 0.06 * soft + pen("end")
                if c < best_c:
                    best, best_c, side = box, c, "end"
        if best is None:                              # nowhere beside its line: the clear spot nearest the line
            def to_line(b) -> float:
                d = math.inf
                for (ax_, ay_), (bx_, by_) in zip(coords, coords[1:], strict=False):
                    for k_ in range(21):
                        px_, py_ = ax_ + (bx_ - ax_) * k_ / 20, ay_ + (by_ - ay_) * k_ / 20
                        d = min(d, math.hypot(max(b.x0 - px_, 0, px_ - b.x1), max(b.y0 - py_, 0, py_ - b.y1)))
                return d

            cx_, cy_ = (coords[0][0] + coords[-1][0]) / 2, (coords[0][1] + coords[-1][1]) / 2
            for i in range(-90, 91, 2):
                for j in range(-50, 51, 2):
                    box = fk.Box(cx_ + 4 * i - w_ / 2, cy_ + 4 * j - h_ / 2, cx_ + 4 * i + w_ / 2, cy_ + 4 * j + h_ / 2)
                    soft = fit(box, 3.0, placer.lines[i0:i1] + others)
                    if soft == math.inf:
                        continue
                    c = to_line(box) + 0.06 * soft + 60
                    if c < best_c:
                        best, best_c, side = box, c, "free"
        if best is None:
            ax_, ay_ = coords[-1]
            best = placer.place(w_, h_, ax_, ay_, gaps=(10, 16, 26, 40, 60, 90),
                                prefer=("ne", "e", "se", "n", "nw", "s", "sw", "w"), scan=240)
            side = "free"
        else:
            placer.labels.append(best)
        if side in ("start", "end", "free"):
            ax_, ay_ = min(coords, key=lambda c: math.hypot(min(max(c[0], best.x0), best.x1) - c[0],
                                                            min(max(c[1], best.y0), best.y1) - c[1]))
            leader(best, ax_, ay_, st["color"], min_len=26)
        labels.append(f'<g data-role="arm-label" data-arm={fk.esc(a["name"])} data-side="{side}" '
                      f'data-text={fk.esc(line1)} data-box="{_box_attr(best)}">'
                      f'{halo(fk.text_paths(line1, best.x0, best.y0 + 14, n1))}</g>')

    # the pick's model and the story's model label their lines first (they carry the plate); then the single markers,
    # from the reference outward; then the other lines (a line label can slide along its line)
    story_arm = next((p["arm"] for p in pts if story_ and p["label"] == story_.get("zero")), None)
    first = sorted((a for a in callo_arms if a["name"] in (pick_arm, story_arm)), key=lambda a: a["name"] != pick_arm)
    for a in first:
        line_label(a)

    # ---- a trained head's or Clef's label: its name beside its marker (runtime and GiB in the table), the nearest
    # spot that touches no label, marker or line (it may cross a thin whisker or tail on its halo)
    def marker_label(a: dict) -> None:
        st = style[a["name"]]
        p = by_arm[a["name"]][0]
        ax_, ay_ = X(p), Y(p)
        n1 = name_st.with_(fill=st["name"])
        forms = [[a["label"]]] + ([a["label"].split(" · ", 1)] if " · " in a["label"] else [])
        best, best_c, rows = None, math.inf, forms[0]
        for strict in (True, False):
            for k_, form in enumerate(forms):
                w_, h_ = max(fk.advance(r_, n1) for r_ in form), 18.0 + 17.0 * (len(form) - 1)
                for i in range(-36, 37):
                    for j in range(-16, 17):
                        bx0, by0 = ax_ + 5 * i - w_ / 2, ay_ + 5 * j - h_ / 2
                        box = fk.Box(bx0, by0, bx0 + w_, by0 + h_)
                        if box.overlap(fk.Box(ax_ - 9, ay_ - 9, ax_ + 9, ay_ + 9)):
                            continue
                        soft = fit(box, 3.0, placer.lines)
                        if soft == math.inf or (strict and not owned(box, p["label"])):
                            continue
                        d = math.hypot(max(box.x0 - ax_, 0, ax_ - box.x1), max(box.y0 - ay_, 0, ay_ - box.y1))
                        c = d + 0.2 * soft + (0 if box.y1 <= ay_ else 3) + 8 * k_    # one line, above, on a tie
                        if c < best_c:
                            best, best_c, rows = box, c, form
            if best is not None:
                break
        if best is None:
            best = placer.place(fk.advance(a["label"], n1), 18.0, ax_, ay_, gaps=(8, 12, 18, 26, 40, 60),
                                prefer=("ne", "nw", "se", "sw", "e", "w", "n", "s"), scan=160)
        else:
            placer.labels.append(best)
        leader(best, ax_, ay_, st["color"], min_len=14)
        labels.append(f'<g data-role="arm-label" data-arm={fk.esc(a["name"])} data-text={fk.esc(a["label"])} '
                      f'data-box="{_box_attr(best)}">'
                      + "".join(halo(fk.text_paths(r_, best.x0, best.y0 + 14 + 17 * i_, n1)) for i_, r_ in enumerate(rows))
                      + "</g>")

    ref_name = ref.get("name")
    for a in sorted((a for a in arms if a["kind"] != "callo"), key=lambda a: a["name"] != ref_name):
        marker_label(a)

    for a in (a for a in callo_arms if a not in first):
        line_label(a)

    # ---- the reference line's label: on one line or two, just above or below the dashed line, at its clearest
    # stretch (on nothing, with clear ground), the left end preferred; else the least-covered stretch
    if ref.get("acc") is not None:
        rs = fk.Style(size=13, fill=t["sub"])
        name, val = ref["label"], f"accuracy {100 * ref['acc']:.1f} %"
        yref = ys(ref["acc"])
        found, found_c = None, math.inf
        for n_, rows in enumerate(([f"{name} {val}"], [name, val])):
            w = max(fk.advance(r, rs) for r in rows)
            h = 17 * len(rows)
            for c in range(int(x0 + 8), int(x1 - w - 8), 4):
                for above, top in ((True, yref - 7 - h), (False, yref + 7)):
                    box = fk.Box(c, top, c + w, top + h)
                    soft = fit(box, 5.0, placer.lines)
                    cost = 0.06 * soft + 0.03 * (c - x0) + 25 * n_ + (0 if above else 6)
                    if cost < found_c:
                        found, found_c = (rows, box), cost
        if found is None:
            rows = [f"{name}, {val}"]
            w = fk.advance(rows[0], rs)
            c = min(range(int(x0 + 10), int(x1 - w - 8), 4),
                    key=lambda c: placer.cost(fk.Box(c - 8, yref - 20, c + w + 8, yref - 2)) + 0.02 * (c - x0))
            found = (rows, fk.Box(c, yref - 24, c + w, yref - 7))
        rows, box = found
        placer.labels.append(box)
        labels.append(f'<g data-role="ref-label" data-text={fk.esc(" ".join(rows))} data-box="{_box_attr(box)}">'
                      + "".join(halo(fk.text_paths(r, box.x0, box.y0 + 13 + 17 * i, rs)) for i, r in enumerate(rows))
                      + "</g>")

    # ---- each other callo point's think budget, small, beside it: the nearest spot that is nearer its own marker than
    # any other and keeps clear of every other label (two numbers on one baseline never read as one)
    def budget_box(p: dict, w_: float):
        px_, py_ = X(p), Y(p)
        h_ = 14.0
        dirs = ("se", "e", "ne", "sw", "s", "n", "w", "nw")
        for strict in (2, 1, 0):          # owned against every marker, then against other series', then anywhere
            best, best_c = None, math.inf
            for gap in (4, 7, 10, 14, 19, 25, 32):
                for di_, d in enumerate(dirs):
                    dx = {"e": 1, "w": -1, "n": 0, "s": 0, "ne": 1, "nw": -1, "se": 1, "sw": -1}[d]
                    dy = {"e": 0, "w": 0, "n": -1, "s": 1, "ne": -1, "nw": -1, "se": 1, "sw": 1}[d]
                    bx0 = px_ + dx * gap + (0 if dx >= 0 else -w_) - (w_ / 2 if dx == 0 else 0)
                    by0 = py_ + dy * gap * 0.8 - (h_ if dy < 0 else 0 if dy > 0 else h_ / 2)
                    box = fk.Box(bx0, by0, bx0 + w_, by0 + h_)
                    if box.overlap(fk.Box(px_ - 6, py_ - 6, px_ + 6, py_ + 6)):
                        continue
                    soft = fit(box, 1.0, placer.lines)
                    if soft == math.inf or (strict and not owned(box, p["label"], 3.0, own_line=strict == 2)):
                        continue
                    c = 2.5 * gap + 0.8 * di_ + 0.5 * soft         # a number on a tail or whisker reads as a tick
                    if c < best_c:
                        best, best_c = box, c
            if best is not None:
                placer.labels.append(best)
                return best
        return placer.place(w_, h_, px_, py_, gaps=(5, 9, 14, 22, 32),
                            prefer=("se", "e", "ne", "sw", "s", "n", "w", "nw"), scan=70)

    for p in sorted((p for p in pts if p["arm"] in {a["name"] for a in callo_arms} and len(by_arm[p["arm"]]) > 1
                     and p["label"] != pick), key=lambda p: (p["arm"] != pick_arm, p["lat"][stat][0])):
        txt = str(p["budget"]) if p["kind"] == "budget" else _budget_text(p)
        box = budget_box(p, fk.advance(txt, pst))
        leader(box, X(p), Y(p), style[p["arm"]]["color"], min_len=18)
        labels.append(f'<g data-role="point-label" data-label={fk.esc(p["label"])} data-text={fk.esc(txt)} '
                      f'data-box="{_box_attr(box)}">{halo(fk.text_paths(txt, box.x0, box.y0 + 11, pst))}</g>')

    body.extend(lines)
    body.extend(marks)
    body.extend(labels)

    # ---- the table: every plotted point's accuracy and latency, by model and think budget
    sec = y1 + 108
    body.append(fk.text("BY BUDGET", SIDE, sec, fk.Style(size=12, weight=600, fill=t["sub"], tracking=1.0),
                        role="section")[0])
    body.append(fk.text(f"accuracy and {stat} latency per decision; a trained head or Clef answers in one pass, without "
                        "thinking", SIDE + 92, sec, fk.Style(size=13, fill=t["sub"]), role="section-note")[0])
    tb_svg, ty1 = _table(fk, t, snap, arms, by_arm, style, stat, pick, sec + 40)
    body.append(tb_svg)

    # ---- fine print: one column, one entry per paragraph, 14 px (legible at README width)
    rule_y = ty1 + 30
    body.append(f'<line x1="{SIDE}" x2="{W - SIDE}" y1="{rule_y}" y2="{rule_y}" stroke="{t["grid"]}"/>')
    lead_st = fk.Style(size=14, weight=600, fill=t["ink"])
    fp_st = fk.Style(size=14, fill=t["sub"])
    yy = rule_y + 32
    for f in snap["fine_print"][stat] if isinstance(snap["fine_print"], dict) else snap["fine_print"]:
        runs = [(f["lead"] + " ", lead_st), (f["text"], fp_st)] if f["lead"] else [(f["text"], fp_st)]
        pr = fk.paragraph(runs, SIDE, yy, W - 2 * SIDE, leading=20, role="fine-print")
        body.append(pr.svg)
        yy += pr.lines * 20 + 7
    H = math.ceil(yy + 18)

    bg = fk.ground(W, H, t, glow_at=((x0 + x1) / 2, y1 + 10, (x1 - x0) * 0.62, 280))
    return fk.svg_doc(W, H, bg + "".join(body), title=title.replace("*", ""), theme=theme,
                      extra_attrs={"data-plot-box": ",".join(_f(v) for v in (x0, y0, x1, y1)), "data-latency": stat})


# ---- CLI ---------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="frontier.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="spec -> snapshot (callo's statistics)")
    b.add_argument("--spec", required=True)
    b.add_argument("--root")
    b.add_argument("--out", default=str(HERE / "data" / "frontier.json"))
    d = sub.add_parser("draw", help="snapshot -> SVG (both themes), optionally PNG")
    d.add_argument("--data", default=str(HERE / "data" / "frontier.json"))
    d.add_argument("--latency", choices=STATS)
    d.add_argument("--out-dir", default=str(HERE))
    d.add_argument("--png", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            snap = build(load_spec(args.spec, root=args.root))
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(snap, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {args.out}: {len(snap['points'])} points, {len(snap['pending'])} pending arm(s)")
            return 0
        snap = json.loads(Path(args.data).read_text(encoding="utf-8"))
        stat = args.latency or snap.get("latency") or "mean"
        suffix = "" if stat == "mean" else f"-{stat}"
        for theme in ("dark", "light"):
            svg = draw(snap, theme=theme, stat=stat)
            out = Path(args.out_dir) / f"frontier-{theme}{suffix}.svg"
            out.write_text(svg, encoding="utf-8")
            print(f"wrote {out}")
            if args.png:
                import figkit
                png = out.with_suffix(".png").with_name(out.stem + "@2x.png")
                figkit.render_png(str(out), str(png), 2.0)
                print(f"wrote {png}")
        return 0
    except Refusal as e:
        print(f"frontier: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
