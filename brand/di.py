"""callo's Decision Index plate (the headline release figure): the public-subset index against
the mean request completion time, callo's thinking budgets as two-point lines beside dedicated decision models (Clef,
d1) as single markers; a line's other arms at its think budget (a plain think beside an early-exit point, a gated
variant) are rows of the plate's table, never drawn. The README's chart is the same plot alone, under one title.

    python brand/di.py build --spec brand/data/di.release.spec.json --root <bench>   # -> brand/data/di.json
    python brand/di.py draw [--png]         # -> brand/di-{dark,light}.svg + brand/di-chart-{dark,light}.svg (@2x.png)

Two layers, as frontier.py. The DATA layer (`build`) reads the benchmark's DI outputs under `--root` and writes the
committed snapshot `brand/data/di.json`, which records every source path relative to that root (a file outside it,
such as a model, by its file name) and names callo's builds by role (`data/builds.json`); the DRAWING layer (`draw`)
bakes the plate and the chart from that snapshot alone.

`build` takes every number from the bench outputs, never computes a statistic of its own:

- `harness/callo_plate/di_bootstrap.py`'s output, one per scoring treatment of the > 26-option rows (`treatments`;
  `wide_options` is the switch that picks the plotted one): per arm the kit's index with its CI, the 5 area skills
  with CIs, the paired Δindex vs the reference (Clef-Flash BF16, the whiskers) and vs the second reference (Clef 27B,
  the table only), the paired Δskill per area, and the request completion time (the kit's client) over the rows
  answered by every arm;
- `di_union.py`'s union file: the rows not answered by every arm, by reason, and every error row; an arm with an
  error row whose cause is neither in the union file nor in the spec's `explain` rules is refused (exit 2), as is an
  arm with pending rows;
- each arm's `di_meta.json` (model, sha256, think profile, budget, early exit and its /callo/stats counters, session
  dates), its `server.log` (callo's llama.cpp build) and its callo checkout (the commit), each checked against the
  spec's labels;
- `paired`: more di_bootstrap.py outputs with another reference (an early-exit point vs its plain think), refused
  unless they share the plotted bootstrap's draws, kit and derived results;
- `patches`: the re-read record (the bench's di_patch.py `apply.json`) of an arm whose results were re-read after a
  readout fix, refused unless the bootstrap scored exactly the results it wrote;
- the sample's `sample.json` (the > 26-option rows) and the Clef prompt-token counts (the over-cap rows);
- a single-marker series' own prompt-token counts (`own_prompt_tokens`, a model that reads one question per prompt,
  such as d1): checked against its run's model, sample and context, and read for its prompts on the over-cap rows.

A single-marker series (kind `clef`: a dedicated decision model on llama.cpp's native decision endpoint) names its
`family` (default Clef), which the Builds line and the run dates group it under.

The derived results each bootstrap scored are re-hashed: a snapshot is never built from a stale bootstrap. Refusals
exit 2 with the reason.
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
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

SNAPSHOT_FORMAT = 1
MIN_RESAMPLES = 1000                     # >= 1,000 paired resamples for a plate number
TREATMENTS = {"kit": "(a) kit-faithful", "all-unsupported": "(b) unanswerable for every arm"}
# the budgets each callo model was run at on DI; the others are named in the fine print, never plotted
RULED_NOT_RUN = {"tiel": (512, 1024), "sharp-minicpm": (256, 512)}
CALLO_MAX_OPTIONS = 255                  # callo's wide-choice readout (v0.3) answers up to 255 options
MARKERS = ("circle", "diamond", "diamond-hollow", "square", "square-hollow", "triangle", "triangle-hollow")
COMPASS = ("n", "ne", "e", "se", "s", "sw", "w", "nw")
CHANCE_TEXT = ("chance levels are the index's own fixed values: per row for the accuracy, nDCG@10 and BFCL benchmarks, "
               "fixed for the F1 benchmarks and the benchmarks added in 0.2")
ANSWERING_TEXT = ("Callo answers a request's questions one at a time, reusing the state prefix (thinking per question "
                  "at a budget above 0); Clef answers all of a row's questions in one joint pass; requests in the "
                  "sampler's round-robin order, one at a time.")


class Refusal(Exception):
    """The plate refuses to draw something that does not compare: exit 2 with the reason."""


# ---- formatting (one place, so the plate, the table and FIGURES.md print a number the same way) ------------------

def f1(x: float) -> str:
    """One decimal, rounded half up (the kit's two-decimal values: 16.45 -> 16.5), with a true minus."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)).replace("-", "−")


def sf1(x: float) -> str:
    """Signed, one decimal: +3.3, −16.5, 0.0."""
    s = f1(x)
    return s if s.startswith("−") or s == "0.0" else "+" + s


def ci1(lo: float, hi: float, *, signed: bool = False) -> str:
    f = sf1 if signed else f1
    return f"[{f(lo)}, {f(hi)}]"


def skill(x: float) -> float:
    """An area skill (0-1, the kit's) on the index's 0-100 scale."""
    return 100.0 * x


def secs(ms: float) -> str:
    """A latency in prose and the table: one decimal of a second from 1 s, whole ms below."""
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def budget_text(b: int | None) -> str:
    return "no think" if not b else f"think {b}"


def _thousands(n: int) -> str:
    return f"{n:,}"


# ---- the spec ----------------------------------------------------------------------------------------------------

PATH_KEYS = ("union", "sample", "prompt_tokens", "prompt_tokens_rows", "di_dir", "area_labels_from")
# a single-marker series' own files: its prompt-token counts (d1), and for a model that cannot answer some rows the
# others answer (Laya's window), its own bootstrap, the union build that keeps those rows its own, the bootstrap
# without the sensitivity's benchmark and its window's counts
OWN_KEYS = ("own_bootstrap", "own_union", "own_sensitivity", "own_window")
SERIES_PATH_KEYS = ("own_prompt_tokens", *OWN_KEYS)


def load_spec(path: str | Path, *, root: str | Path | None = None) -> dict:
    """The spec with every path resolved against `root` (the argument, `--root`), else the spec's own `root`, else its
    dir."""
    path = Path(path)
    spec = dict(json.loads(path.read_text(encoding="utf-8")))
    base = Path(os.path.abspath(root or spec.get("root") or path.parent))
    spec["_spec_path"] = str(path)
    spec["_root"] = str(base)

    def res(p: str | None) -> str | None:
        return None if p is None else str(p if os.path.isabs(p) else base / p)

    for k in PATH_KEYS:
        if spec.get(k):
            spec[k] = res(spec[k])
    spec["treatments"] = {t: res(p) for t, p in (spec.get("treatments") or {}).items()}
    spec["paired"] = [{**c, "path": res(c.get("path"))} for c in spec.get("paired") or []]
    spec["patches"] = {a: res(p) for a, p in (spec.get("patches") or {}).items()}
    if isinstance(spec.get("sensitivity"), dict):
        spec["sensitivity"] = {**spec["sensitivity"], "path": res(spec["sensitivity"].get("path"))}
    spec["series"] = [{**s, **{k: res(s[k]) for k in SERIES_PATH_KEYS if s.get(k)}} if isinstance(s, dict) else s
                      for s in spec.get("series") or []]
    return spec


def _check_series(spec: dict) -> list[dict]:
    out, seen_names, seen_arms = [], set(), set()
    for s in spec.get("series") or []:
        s = dict(s)
        name = s.get("name")
        if not isinstance(name, str) or not name:
            raise Refusal(f"a series has no name: {s}")
        if name in seen_names:
            raise Refusal(f"two series are named {name!r}")
        seen_names.add(name)
        if s.get("kind") not in ("callo", "clef"):
            raise Refusal(f"series {name}: kind must be callo or clef, got {s.get('kind')!r}")
        if not s.get("label"):
            raise Refusal(f"series {name}: no label")
        if s.get("marker") is not None and s["marker"] not in MARKERS:
            raise Refusal(f"series {name}: marker must be one of {', '.join(MARKERS)}")
        if s.get("plate", True) not in (True, False):
            raise Refusal(f"series {name}: plate is true (drawn) or false (data-only: di.json and FIGURES.md)")
        if s.get("label_side") not in (None, "above", "below", "end"):
            raise Refusal(f"series {name}: label_side is above, below or end (past its think point)")
        if s.get("point_labels") not in (None, "above", "below"):
            raise Refusal(f"series {name}: point_labels is above or below (where its budget labels prefer to sit)")
        at = s.get("point_label_at") or {}
        if not isinstance(at, dict) or any(v not in COMPASS for v in at.values()):
            raise Refusal(f"series {name}: point_label_at is {{arm: one of {', '.join(COMPASS)}}} (the side one budget "
                          "label tries first)")
        if s["kind"] == "callo":
            budgets = {int(b): a for b, a in (s.get("budgets") or {}).items()}
            if len(budgets) != 2 or 0 not in budgets:
                raise Refusal(f"series {name}: a callo line is two points, budget 0 and one think budget; got "
                              f"{sorted(budgets)}")
            not_run = sorted(set(s.get("not_run") or []) | set(RULED_NOT_RUN.get(name, ())))
            bad = sorted(set(budgets) & set(not_run))
            if bad:
                raise Refusal(f"series {name}: budget(s) {bad} were not run on DI; they are never plotted")
            # variants: more arms at the line's think budget that differ from its point in early exit (a plain think
            # beside an early-exit point, a gated one): rows of the table, never drawn
            variants = {str(a): int(b) for a, b in (s.get("variants") or {}).items()}
            for a, b in variants.items():
                if b == 0 or b not in budgets:
                    raise Refusal(f"series {name}: variant {a} is at budget {b}, not the line's think budget "
                                  f"{max(budgets)}")
            ee = {str(a): dict(c) if isinstance(c, dict) else c for a, c in (s.get("early_exit") or {}).items()}
            for a, c in ee.items():
                if a not in {*budgets.values(), *variants}:
                    raise Refusal(f"series {name}: early_exit names {a}, which is not one of its arms")
                if a == budgets[0]:
                    raise Refusal(f"series {name}: {a} is its budget-0 point; there is no thinking to exit early from")
                if (not isinstance(c, dict) or not isinstance(c.get("z"), (int, float)) or not 0 < c["z"] < 1
                        or c.get("gate") not in (True, False)):
                    raise Refusal(f"series {name}: early_exit {a} is {{\"z\": 0 < z < 1, \"gate\": true | false}}")
            same = [a for a, b in variants.items() if ee.get(a) == ee.get(budgets[b])]
            if same:
                raise Refusal(f"series {name}: variant(s) {same} run the same early exit as the line's point; a "
                              "variant must say what it varies (an early_exit entry on it or on the point)")
            s["budgets"] = dict(sorted(budgets.items()))
            s["not_run"] = not_run
            s["variants"] = variants
            s["early_exit"] = ee
            arms = list(budgets.values()) + list(variants)
        else:
            if not s.get("arm"):
                raise Refusal(f"series {name}: a clef series names its arm")
            arms = [s["arm"]]
        if s["kind"] == "callo" and (s.get("family") or s.get("own_prompt_tokens") or s.get("answering")):
            raise Refusal(f"series {name}: family, answering and own_prompt_tokens are for a single-marker series")
        for k in ("family", "answering", "cut_text"):
            if s.get(k) is not None and (not isinstance(s[k], str) or not s[k]):
                raise Refusal(f"series {name}: {k} is a non-empty string")
        if s.get("own_prompt_tokens") and not s.get("answering"):
            raise Refusal(f"series {name}: own prompt-token counts come with the words for how it answers (answering)")
        if any(s.get(k) for k in OWN_KEYS):
            if s["kind"] == "callo":
                raise Refusal(f"series {name}: an own bootstrap is for a single-marker series")
            if not (s.get("own_bootstrap") and s.get("own_union") and s.get("own_window")):
                raise Refusal(f"series {name}: an own bootstrap comes with its own union and its window's counts "
                              "(own_bootstrap, own_union, own_window)")
        stray = sorted(set(s.get("point_label_at") or {}) - set(arms))
        if stray:
            raise Refusal(f"series {name}: point_label_at names {stray}, not its arms")
        for a in arms:
            if a in seen_arms:
                raise Refusal(f"arm {a} appears in two places")
            seen_arms.add(a)
        out.append(s)
    if not out:
        raise Refusal("the spec names no series")
    if sum(1 for s in out if s["kind"] == "callo" and s.get("accent")) > 1:
        raise Refusal("one callo series at most carries the accent (Signal is used once)")
    return out


def series_arms(s: dict) -> list[tuple[int | None, str]]:
    """A series' line arms (a callo line's two points; a Clef series' one), budgets ascending."""
    return [(b, a) for b, a in s["budgets"].items()] if s["kind"] == "callo" else [(None, s["arm"])]


def series_points(s: dict) -> list[tuple[int | None, str, str | None]]:
    """Every measured arm of a series in the table's order: (budget, arm, the line arm it is a variant of, or None);
    budgets ascending, each budget's arms plain first, then with early exit, then with early exit + gate (the line's
    own point among them wherever its early exit puts it)."""
    ee = s.get("early_exit") or {}

    def level(arm: str) -> tuple[bool, bool]:
        c = ee.get(arm)
        return (c is not None, bool(c and c.get("gate")))

    out: list[tuple[int | None, str, str | None]] = []
    for b, a in series_arms(s):
        here = [(b, a, None)] + [(b, v, a) for v, vb in (s.get("variants") or {}).items() if vb == b]
        out += sorted(here, key=lambda x: level(x[1]))
    return out


def ee_text(ee: dict | None) -> str:
    """An early-exit point's words: "early exit", "early exit + gate"."""
    return "" if not ee else "early exit" + (" + gate" if ee.get("gate") else "")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pub(p: str | None, spec: dict) -> str | None:
    """A path as the snapshot records it: relative to the build's root, else its file name (figkit.public_path)."""
    import figkit
    return figkit.public_path(p, spec["_root"])


def _pub_cmd(cmd: str | None, spec: dict) -> str | None:
    """A recorded command with each absolute path in it as the snapshot records a path."""
    return None if cmd is None else " ".join(_pub(t, spec) if os.path.isabs(t) else t for t in str(cmd).split(" "))


def build_of(commit: str | None) -> dict:
    """The build a callo commit is (data/builds.json): the plate names it by its role, never by its commit."""
    import figkit
    b = figkit.build_of(commit)
    if b is None:
        raise Refusal(f"callo commit {commit} is not in brand/data/builds.json: the plate names a build by its role")
    return b


def _load(path: str | None, what: str) -> dict:
    if not path or not Path(path).is_file():
        raise Refusal(f"no {what} at {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---- the data layer ----------------------------------------------------------------------------------------------

def _check_bootstrap(doc: dict, spec: dict, treatment: str, path: str) -> dict:
    if doc.get("edition") != "0.3":
        raise Refusal(f"{path}: edition {doc.get('edition')!r}, the plate is Decision Index 0.3")
    if int(doc.get("n") or 0) < MIN_RESAMPLES:
        raise Refusal(f"{path}: {doc.get('n')} resamples; a plate number needs >= {MIN_RESAMPLES}")
    if doc.get("warnings"):
        raise Refusal(f"{path}: the bootstrap carries warnings: {'; '.join(doc['warnings'])}")
    if doc.get("reference") != spec["reference"] or doc.get("second_reference") != spec["second_reference"]:
        raise Refusal(f"{path}: references {doc.get('reference')}/{doc.get('second_reference')}, the spec says "
                      f"{spec['reference']}/{spec['second_reference']}")
    for arm, a in doc["arms"].items():
        if a["index"] != a["kit_index"]:
            raise Refusal(f"{path}: arm {arm}: the bootstrap's index {a['index']} is not the kit's {a['kit_index']}")
        if a["latency_ms"]["n_rows"] != doc["latency_rows"]:
            raise Refusal(f"{path}: arm {arm}: latency over {a['latency_ms']['n_rows']} rows, not the "
                          f"{doc['latency_rows']} answered by every arm")
    # the derived results it scored must be the files on disk now (a stale bootstrap is refused)
    root = Path(spec["_root"])
    for arm, inp in doc["inputs"].items():
        for key in ("derived", "results"):
            p = Path(inp[key]) if os.path.isabs(inp[key]) else root / inp[key]
            if not p.is_file():
                raise Refusal(f"{path}: arm {arm}: {inp[key]} is missing")
            if sha256_file(p) != inp[f"{key}_sha256"]:
                raise Refusal(f"{path}: arm {arm}: {inp[key]} changed since the bootstrap read it (sha256); re-run "
                              f"di_bootstrap.py ({treatment})")
    return doc


def _explain(rules: list[dict], e: dict) -> str | None:
    """di_union.py's own rule shape: the first rule whose arm, run id and error substring all match."""
    for r in rules:
        if r.get("arm", "*") not in ("*", e["arm"]):
            continue
        if "run_id" in r and r["run_id"] != e["run_id"]:
            continue
        if "error_contains" in r and r["error_contains"] not in str(e.get("error", "")):
            continue
        return r["cause"]
    return None


def _session(meta: dict, arm: str) -> dict:
    """The measuring session: the last one that ran the sample to its final counts."""
    done = [s for s in meta.get("sessions") or [] if s.get("final_counts")]
    if not done:
        raise Refusal(f"arm {arm}: di_meta.json has no session with final counts")
    return done[-1]


def _flag(cmd: list[str], *names: str) -> str | None:
    for n in names:
        if n in cmd:
            i = cmd.index(n)
            return cmd[i + 1] if i + 1 < len(cmd) else None
    return None


def _checkout_commit(exe: str) -> str | None:
    """The commit of the callo checkout a run's `callo` executable came from (…/<checkout>/.venv/bin/callo)."""
    try:
        checkout = Path(exe).resolve().parents[2]
    except IndexError:
        return None
    if not (checkout / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--short=7", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--", "src"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out + ("-dirty" if dirty else "")


def _server_build(arm_dir: Path) -> str | None:
    log = arm_dir / "server.log"
    if not log.is_file():
        return None
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"build_info=(\S+)", line)
            if m:
                return m.group(1)
    return None


def _early_exit(sess: dict, cmd: list[str], want: dict | None, arm: str) -> dict | None:
    """What the session says about early exit, checked against the spec: an early-exit arm was served with
    --think-converge Z (and --think-converge-gate iff gated) and di_meta recorded the same; any other callo arm was
    served without either. Its counters are /callo/stats think.converge at the end of the session less the snapshot
    right after the warm-up (`think_converge_after_warm`), so they count the sample's requests only; the read-time
    percentiles cannot be split that way and are the session's (the warm-up's few reads included)."""
    z = _flag(cmd, "--think-converge")
    gated = "--think-converge-gate" in cmd
    if not want:
        if (z is not None and float(z) > 0) or gated or float(sess.get("converge") or 0) > 0 or sess.get("gate"):
            raise Refusal(f"arm {arm}: served with early exit (--think-converge {z}, gate {gated}); the spec plots it "
                          "as a plain point")
        return None
    if z is None or float(z) != want["z"] or gated != want["gate"]:
        raise Refusal(f"arm {arm}: served with --think-converge {z}, gate {gated}; the spec says z {want['z']}, gate "
                      f"{want['gate']}")
    if sess.get("converge") != want["z"] or bool(sess.get("gate")) != want["gate"]:
        raise Refusal(f"arm {arm}: di_meta.json records converge {sess.get('converge')}, gate {sess.get('gate')}; the "
                      f"spec says z {want['z']}, gate {want['gate']}")
    check = sess.get("converge_check") or {}
    if check.get("pass") is not True:
        raise Refusal(f"arm {arm}: the server's early-exit check after launch did not pass ({check.get('why')})")
    tc, warm = sess.get("think_converge"), sess.get("think_converge_after_warm")
    if not tc or not warm:
        raise Refusal(f"arm {arm}: di_meta.json has no think_converge block (or no warm-up snapshot to subtract)")
    g, gw = tc.get("gate") or {}, warm.get("gate") or {}

    def less(k: str, a: dict = tc, b: dict = warm) -> int:
        return int(a.get(k) or 0) - int(b.get(k) or 0)

    out = {"z": want["z"], "gate": want["gate"], "default_z": tc.get("default_z"), "K": tc.get("K"),
           "requests": less("requests"), "questions": less("questions"), "converged": less("converged"),
           "probes": less("probes"), "tokens_saved": less("tokens_saved"),
           "warmup_requests": int(warm.get("requests") or 0) + int(gw.get("skipped") or 0)}
    if want["gate"]:
        out["gate_reads"] = {"asked": less("asked", g, gw), "skipped": less("skipped", g, gw),
                             "errors": less("errors", g, gw), "read_ms_session": dict(g.get("read_ms") or {})}
        if out["gate_reads"]["asked"] <= 0:
            raise Refusal(f"arm {arm}: the gate is on but read no request")
    return out


def _provenance(s: dict, budget: int | None, arm: str, spec: dict) -> dict:
    """What di_meta.json (and the run's server log and callo checkout) say about the arm, checked against the spec."""
    arm_dir = Path(spec["di_dir"]) / arm
    meta = _load(str(arm_dir / "di_meta.json"), f"di_meta.json for arm {arm}")
    if meta.get("arm") != arm:
        raise Refusal(f"{arm_dir}/di_meta.json is for arm {meta.get('arm')!r}")
    sess = _session(meta, arm)
    cmd = [str(c) for c in sess.get("cmd") or []]
    model = sess.get("model") or _flag(cmd, "-m", "--model")
    if s.get("model_file") and Path(str(model)).name != s["model_file"]:
        raise Refusal(f"arm {arm}: ran {Path(str(model)).name}, the spec labels it {s['model_file']}")
    rec: dict[str, Any] = {
        "dir": _pub(str(arm_dir), spec), "utc_start": sess.get("utc_start"), "utc_end": sess.get("utc_end"),
        "date": str(sess.get("utc_start"))[:10], "model": Path(str(model)).name, "model_sha256": sess.get("model_sha256"),
        "sample_sha256": sess.get("sample_sha256"), "kit_commit": sess.get("kit_commit"),
        "final_counts": sess.get("final_counts"), "accept_cap_loss": sess.get("accept_cap_loss"),
        "sessions": len(meta.get("sessions") or []), "files": [{"path": _pub(str(arm_dir / "di_meta.json"), spec),
                                                                 "sha256": sha256_file(arm_dir / "di_meta.json")}],
    }
    if s["kind"] == "callo":
        b = _flag(cmd, "--think-budget")
        if b is None or int(b) != budget:
            raise Refusal(f"arm {arm}: served with --think-budget {b}, the spec plots it as budget {budget}")
        rec["early_exit"] = _early_exit(sess, cmd, (s.get("early_exit") or {}).get(arm), arm)
        rec["server_args"] = [_pub(a, spec) if os.path.isabs(str(a)) else a for a in sess.get("server_args") or []]
        if sess.get("chat_template"):
            rec["chat_template"] = {"path": _pub(sess["chat_template"], spec), "sha256": sess.get("chat_template_sha256")}
        found = _checkout_commit(cmd[0]) if cmd else None
        pinned = (spec.get("callo_commits") or {}).get(arm)
        if found and pinned and not found.startswith(pinned) and not pinned.startswith(found):
            raise Refusal(f"arm {arm}: its callo checkout is at {found}, the spec pins {pinned}")
        commit = pinned or found
        if not commit:
            raise Refusal(f"arm {arm}: no callo commit (no checkout behind {cmd[:1]} and none in callo_commits)")
        build = _server_build(arm_dir) or s.get("build")
        if not build:
            raise Refusal(f"arm {arm}: no llama.cpp build (server.log build_info or the series' build)")
        rec.update(think_budget=int(b), think_profile=_pub(sess.get("think_profile"), spec),
                   think_profile_sha256=sess.get("think_profile_sha256"), callo_commit=commit,
                   cache="off" if "--no-cache" in cmd else "on (default)", build=build,
                   server={k: _flag(cmd, f"--{k}") for k in ("ctx", "parallel", "s1-slots")})
    else:
        if not s.get("build"):
            raise Refusal(f"series {s['name']}: a clef series states its llama.cpp build")
        # a run that recorded its server's --version: the commit it names is the spec's build (prefix either way)
        ver = " ".join(map(str, sess.get("server_version") or []))
        m = re.search(r"commit ([0-9a-f]{7,40})", ver)
        built = str(s["build"]).split()[-1]
        if m and not (built.startswith(m.group(1)) or m.group(1).startswith(built)):
            raise Refusal(f"arm {arm}: its llama-server reports commit {m.group(1)}, the spec labels build {s['build']}")
        rec.update(build=s["build"], server={"ctx": _flag(cmd, "-c"), "batch": _flag(cmd, "-b"),
                                             "ubatch": _flag(cmd, "-ub"), "parallel": _flag(cmd, "-np")})
    return rec


def family(s: dict) -> str:
    """The name the Builds line and the run dates group a series under: Callo, else its `family` (default Clef)."""
    return "Callo" if s["kind"] == "callo" else s.get("family") or "Clef"


def _own_tokens(s: dict, prov: dict, spec: dict, unanswered: dict, sample_sha: str) -> dict | None:
    """A single-marker series' own prompt-token counts (`own_prompt_tokens`: a model that reads one question per
    prompt, e.g. d1, counted offline by its server's own code): made from the model, sample and context its run used,
    and needing no pre-marks of its own. Returns its prompts on the over-cap rows (the longest per row, min and max)
    and the longest prompt it was sent, for the fine print."""
    path = s.get("own_prompt_tokens")
    if not path:
        return None
    doc = _load(path, f"series {s['name']}'s own prompt-token counts")
    if doc.get("arm") != s["arm"]:
        raise Refusal(f"{path}: counts for arm {doc.get('arm')!r}, not {s['arm']}")
    if prov.get("model_sha256") and doc.get("gguf_sha256") != prov["model_sha256"]:
        raise Refusal(f"{path}: counted on another model (gguf sha256) than arm {s['arm']} ran")
    if doc.get("sample_sha256_uncompressed") != sample_sha:
        raise Refusal(f"{path}: counted on another sample")
    if str(doc.get("n_ctx")) != str(prov["server"].get("ctx")):
        raise Refusal(f"{path}: counted for a {doc.get('n_ctx')}-token context, arm {s['arm']} ran with "
                      f"{prov['server'].get('ctx')}")
    if doc.get("own_premarks_needed"):
        raise Refusal(f"{path}: arm {s['arm']} needs pre-marks of its own; the plate's answered set would change")
    pre = unanswered["over_cap"]["run_ids"]
    per_row = doc.get("premarked_rows_max_question") or {}
    if set(per_row) != set(pre):
        raise Refusal(f"{path}: its pre-marked rows are not the plate's {len(pre)} over-cap rows")
    sent = doc.get("sent_max_question") or {}
    return {"path": _pub(path, spec), "sha256": sha256_file(path), "n_ctx": int(doc["n_ctx"]),
            "over_cap_question_max": [min(per_row.values()), max(per_row.values())],
            "sent_rows": doc.get("rows_sent"), "sent_questions": doc.get("questions_sent"),
            "sent_question_max": sent.get("max")}


PAIRED_FIELDS = ("index", "kit_index", "index_ci", "areas", "areas_ci", "delta_vs_reference",
                 "delta_vs_second_reference", "area_delta_vs_reference")
SAME_DRAWS = ("edition", "n", "seed", "kit_commit", "suite_sha256", "groups_per_benchmark", "rows", "reference",
              "second_reference")


def _same_draws(doc: dict, base: dict, path: str, arm: str, what: str) -> None:
    """An arm's own bootstrap pairs with the plate's: the same edition, draws (n, seed, groups), kit, suite and
    references, and every other arm in it scored from the plate's results to the plate's numbers exactly (the
    references' index, CIs and Δs; latency is over another row set, so it is not compared)."""
    for k in SAME_DRAWS:
        if doc.get(k) != base.get(k):
            raise Refusal(f"{path}: {k} {str(doc.get(k))[:40]!r} does not pair with the plate's {what} bootstrap")
    if arm not in doc["arms"] or arm in base["arms"]:
        raise Refusal(f"{path}: arm {arm} must be in its own bootstrap and not in the plate's {what} bootstrap")
    for other, a in doc["arms"].items():
        if other == arm:
            continue
        if other not in base["arms"] or doc["inputs"][other] != base["inputs"][other]:
            raise Refusal(f"{path}: arm {other} was not scored from the plate's {what} bootstrap's results")
        if {k: a[k] for k in PAIRED_FIELDS} != {k: base["arms"][other][k] for k in PAIRED_FIELDS}:
            raise Refusal(f"{path}: arm {other}'s numbers differ from the plate's {what} bootstrap: other draws")


def _own_boot(s: dict, spec: dict, boot: dict, union: dict, sample: dict) -> dict:
    """A single-marker series scored in its own bootstrap (`own_bootstrap`): a model that cannot answer some rows the
    others answer (Laya: rows whose question and options alone pass its window) counts those rows wrong for itself
    only (the bench's di_union.py --arm-only), so the plate's answered set and every other arm's numbers stay as
    they are. Its own union (`own_union`) must be the plate's union with the arm added: the same rows not answered by
    every arm, every other arm's derived results byte-identical, the arm's own misses listed as arm-only and nothing
    pending or in error. Its bootstrap pairs with the plate's (the same draws; the references' numbers equal). Its
    latency is over the rows it answered. Its window's counts (`own_window`, the bench's per-row token file) give the
    rows cut to fit and must name the same misses."""
    arm, path = s["arm"], s["own_bootstrap"]
    doc = _check_bootstrap(_load(path, f"series {s['name']}'s own bootstrap"), spec, "kit", path)
    _same_draws(doc, boot, path, arm, "kit")
    upath = s["own_union"]
    u = _load(upath, f"series {s['name']}'s own union")
    if (doc.get("union") or {}).get("sha256") != sha256_file(upath):
        raise Refusal(f"{path}: not made from the union {upath} (sha256)")
    if (set(u["union"]) != set(union["union"]) or u["sample_rows"] != union["sample_rows"]
            or u.get("sample_sha256_uncompressed") != union.get("sample_sha256_uncompressed")
            or set(u["arms"]) != set(union["arms"]) | {arm}):
        raise Refusal(f"{upath}: not the plate's union with arm {arm} added; the plate's answered set would change")
    for a in union["arms"]:
        if u["derived_sha256"].get(a) != union["derived_sha256"][a]:
            raise Refusal(f"{upath}: arm {a}'s derived results differ from the plate's union's")
    if u["derived_sha256"].get(arm) != doc["inputs"][arm]["derived_sha256"]:
        raise Refusal(f"{path}: arm {arm}'s derived results are not the ones {upath} wrote")
    if set(u.get("arm_only") or {}) - {arm}:
        raise Refusal(f"{upath}: arm-only rows of another arm")
    if u["pending"].get(arm) or u["unexplained_errors"].get(arm) or [e for e in u.get("errors") or []
                                                                       if e["arm"] == arm]:
        raise Refusal(f"{upath}: arm {arm} has pending or error rows")
    own = (u.get("arm_only") or {}).get(arm) or {"rows": 0, "per_benchmark": {}}
    if doc["latency_rows"] != u["sample_rows"] - u["union_rows"] - own["rows"]:
        raise Refusal(f"{path}: latency over {doc['latency_rows']} rows, not the {u['sample_rows']} sampled less the "
                      f"{u['union_rows']} not answered by every arm and its own {own['rows']}")
    wpath = s["own_window"]
    w = _load(wpath, f"series {s['name']}'s window counts")
    if w.get("arm") != arm or w.get("sample_sha256_uncompressed") != union.get("sample_sha256_uncompressed"):
        raise Refusal(f"{wpath}: counts for arm {w.get('arm')!r} or another sample")
    if len(w.get("own_premarks") or {}) != own["rows"]:
        raise Refusal(f"{wpath}: {len(w.get('own_premarks') or {})} rows over the window, the union counts "
                      f"{own['rows']} arm-only rows")
    per = sample["per_benchmark"]
    return {"arm": arm, "entry": doc["arms"][arm], "inputs": doc["inputs"], "doc": doc,
            "record": {"path": _pub(path, spec), "sha256": sha256_file(path), "union": _pub(upath, spec),
                       "union_sha256": sha256_file(upath), "window": _pub(wpath, spec),
                       "window_sha256": sha256_file(wpath), "window_tokens": int(w["max_len"]),
                       "truncation": w.get("truncation"), "cut": int(w["rows_truncated_sent"]),
                       "misses": own["rows"],
                       "misses_by": [[per[str(c)]["dataset"], n] for c, n in sorted(
                           own["per_benchmark"].items(), key=lambda kv: (-kv[1], int(kv[0])))],
                       "latency_rows": doc["latency_rows"], "answered_by_others": u["sample_rows"] - u["union_rows"]},
            "_paths": [path, upath, wpath]}


def _own_sensitivity(s: dict, own: dict, doc: dict, path: str) -> dict:
    """The arm's own bootstrap without the sensitivity's benchmark: the same pairing with the plate's sensitivity
    bootstrap as its own bootstrap has with the plate's."""
    arm, spath = s["arm"], s.get("own_sensitivity")
    if not spath:
        raise Refusal(f"series {s['name']}: the plate has a sensitivity footnote; its own bootstrap needs one too "
                      "(own_sensitivity)")
    sd = _load(spath, f"series {s['name']}'s own sensitivity bootstrap")
    if sd.get("warnings") or sd.get("dropped") != doc.get("dropped"):
        raise Refusal(f"{spath}: warnings, or another dropped benchmark than {path}")
    _same_draws(sd, doc, spath, arm, "sensitivity")
    if sd["inputs"] != own["inputs"] or sd.get("union") != own["doc"].get("union"):
        raise Refusal(f"{spath}: not scored from the results its own bootstrap scored")
    a = sd["arms"][arm]
    if a["index"] != a["kit_index"]:
        raise Refusal(f"{spath}: arm {arm}: the bootstrap's index {a['index']} is not the kit's {a['kit_index']}")
    return a


def _areas(boot: dict, spec: dict) -> list[dict]:
    ids = list(next(iter(boot["arms"].values()))["areas"])
    labels: dict[str, str] = {}
    weights: dict[str, float] = {}
    src = spec.get("area_labels_from")
    if src:
        doc = _load(src, "the kit's index.json for the area labels")
        who = spec.get("area_labels_arm") or spec["reference"]       # the arm the kit scored into that file
        if who not in boot["arms"]:
            raise Refusal(f"area_labels_arm {who} is not in the bootstrap")
        ref = boot["arms"][who]
        if doc.get("index") != ref["index"]:
            raise Refusal(f"{src}: index {doc.get('index')} is not {who}'s {ref['index']}: another scoring")
        labels = {a["id"]: a["label"] for a in doc.get("areas") or []}
        for m in re.finditer(r"\b(\w+) (\d+(?:\.\d+)?)%", doc.get("note") or ""):
            if m.group(1) in ids:
                weights[m.group(1)] = float(m.group(2))
        if set(labels) != set(ids):
            raise Refusal(f"{src}: areas {sorted(labels)} are not the bootstrap's {sorted(ids)}")
    return [{"id": i, "label": labels.get(i) or i.capitalize(), "weight": weights.get(i)} for i in ids]


def point_label(s: dict, budget: int | None, arm: str) -> str:
    """A point's name in the table and FIGURES.md: "Tiel-Coder · think 256", "… · think 256, early exit + gate"."""
    if s["kind"] != "callo":
        return s["label"]
    ee = ee_text((s.get("early_exit") or {}).get(arm))
    return f"{s.get('short') or s['label']} · {budget_text(budget)}" + (f", {ee}" if ee else "")


def _point(s: dict, budget: int | None, arm: str, a: dict, spec: dict) -> dict:
    ref, second = spec["reference"], spec["second_reference"]

    def d3(x: dict) -> list[float]:
        return [x["point"], *x["ci"]]

    variant_of = next((v for b, a_, v in series_points(s) if a_ == arm), None)
    return {
        "arm": arm, "series": s["name"], "kind": s["kind"], "budget": budget, "variant_of": variant_of,
        "early_exit": (s.get("early_exit") or {}).get(arm),
        "label": point_label(s, budget, arm),
        "index": a["index"], "index_ci": list(a["index_ci"]),
        "dref": None if arm == ref else d3(a["delta_vs_reference"]),
        "dsecond": None if arm == second else d3(a["delta_vs_second_reference"]),
        "areas": dict(a["areas"]), "areas_ci": {k: list(v) for k, v in a["areas_ci"].items()},
        "area_dref": None if arm == ref else {k: d3(v) for k, v in a["area_delta_vs_reference"].items()},
        "lat": {"mean": a["latency_ms"]["mean"], "p95": a["latency_ms"]["p95"],
                "mean_ci": list(a["latency_ms"]["mean_ci"]), "n": a["latency_ms"]["n_rows"]},
    }


def _wide(sample: dict, switch: str, computed: dict[str, bool], limit: int) -> dict:
    """The > 26-option rows and the treatment switch. Treatment (b) counts the rows callo cannot answer as unanswerable
    for every arm; with no row over callo's option limit that set is empty and (b) equals (a): moot."""
    tot = sample["totals"]
    if f"rows_gt{limit}" not in tot:
        raise Refusal(f"sample.json counts no rows over {limit} options (callo_max_options)")
    over = tot[f"rows_gt{limit}"]
    benches = [[b["dataset"], b["rows_gt26"]] for b in sample["per_benchmark"].values() if b.get("rows_gt26")]
    return {"rows_gt26": tot["rows_gt26"], "questions_gt26": tot["questions_gt26"], "rows_gt255": tot["rows_gt255"],
            "max_options": sample.get("max_options_overall"), "benchmarks": benches, "switch": switch,
            "label": TREATMENTS[switch], "computed": computed, "callo_max_options": limit, "rows_over_callo": over,
            "moot": over == 0, "moot_text": f"moot: no row exceeds Callo's {limit}-option limit, so it equals (a)"}


def _unanswered(union: dict, boot: dict, plotted: list[str], spec: dict, sample: dict) -> dict:
    if set(union["arms"]) != set(boot["inputs"]):
        raise Refusal(f"the union ({sorted(union['arms'])}) and the bootstrap ({sorted(boot['inputs'])}) were made "
                      "from different arms")
    if union["sample_rows"] != boot["rows"] or boot["latency_rows"] != union["sample_rows"] - union["union_rows"]:
        raise Refusal(f"the latency set ({boot['latency_rows']} rows) is not the {union['sample_rows']} sampled rows "
                      f"less the {union['union_rows']} not answered by every arm")
    for arm in plotted:
        if union["pending"].get(arm):
            raise Refusal(f"arm {arm}: {union['pending'][arm]} row(s) still pending: the run is not complete")
    errors, unexplained = [], []
    for e in union.get("errors") or []:
        cause = e.get("cause") or _explain(spec.get("explain") or [], e)
        rec = {"arm": e["arm"], "run_id": e["run_id"], "error": e.get("error"), "cause": cause,
               "row": (union["union"].get(e["run_id"]) or {}).get("group_id") or e["run_id"],
               "catalog_id": e.get("catalog_id")}
        errors.append(rec)
        if cause is None and e["arm"] in plotted:
            unexplained.append(rec)
    if unexplained:
        e = unexplained[0]
        raise Refusal(f"arm {e['arm']}: {len(unexplained)} error row(s) with no known cause (first {e['run_id']}: "
                      f"{str(e['error'])[:90]!r}); the plate refuses an arm with unexplained errors (add a rule to "
                      "the spec's explain)")
    pre = sorted(rid for rid, u in union["union"].items() if set(u["arms"].values()) == {"pre-marked"}
                 and set(u["arms"]) == set(union["arms"]))
    if len(pre) != union["union_by_reason"].get("pre-marked", 0):
        raise Refusal("the union's pre-marked rows are not pre-marked for every arm")
    cats = sorted({union["union"][r]["catalog_id"] for r in pre})
    over: dict[str, Any] = {"n": len(pre), "run_ids": pre, "catalog_ids": cats,
                            "benchmarks": [sample["per_benchmark"][str(c)]["dataset"] for c in cats],
                            "groups": len({union["union"][r]["group_id"] for r in pre}),
                            "of_groups": sum(sample["per_benchmark"][str(c)]["groups"] for c in cats)}
    if spec.get("prompt_tokens"):
        pt = _load(spec["prompt_tokens"], "the Clef prompt-token counts")
        if set(pt["over_cap_run_ids"]) != set(pre):
            raise Refusal("the pre-marked rows are not the rows over the Clef prompt cap")
        over["cap"] = pt["cap_tokens"]
        if spec.get("prompt_tokens_rows"):
            tok = _load(spec["prompt_tokens_rows"], "the per-row Clef prompt tokens")["tokens"]
            over["tokens"] = [min(tok[r] for r in pre), max(tok[r] for r in pre)]
    return {"sample_rows": union["sample_rows"], "union_rows": union["union_rows"],
            "by_reason": dict(union["union_by_reason"]), "latency_rows": boot["latency_rows"], "over_cap": over,
            "errors": errors}


def build(spec: dict) -> dict:
    """The snapshot: every plotted number with its source, provenance per arm, the fine print, FIGURES.md's items."""
    import figkit
    series = _check_series(spec)
    for k in ("reference", "second_reference"):
        if not spec.get(k):
            raise Refusal(f"the spec names no {k}")
    switch = spec.get("wide_options")
    if switch not in TREATMENTS:
        raise Refusal(f"wide_options must be one of {', '.join(TREATMENTS)}, got {switch!r}")
    tpaths = spec.get("treatments") or {}
    unknown = set(tpaths) - set(TREATMENTS)
    if unknown:
        raise Refusal(f"unknown treatment(s) {sorted(unknown)}")
    if not tpaths.get(switch):
        raise Refusal(f"wide_options picks {switch}, but the spec gives no bootstrap file for it")
    boots = {t: _check_bootstrap(_load(p, f"the {t} bootstrap"), spec, t, p) for t, p in tpaths.items() if p}
    boot = boots[switch]
    for t, b in boots.items():
        if set(b["arms"]) != set(boot["arms"]) or b["seed"] != boot["seed"] or b["kit_commit"] != boot["kit_commit"]:
            raise Refusal(f"treatment {t}'s bootstrap does not pair with {switch}'s (arms, seed or kit)")

    union = _load(spec.get("union"), "the union file")
    sample = _load(spec.get("sample"), "sample.json")
    own_boot = {s["arm"]: _own_boot(s, spec, boot, union, sample) for s in series if s.get("own_bootstrap")}
    if own_boot and len(boots) > 1:
        raise Refusal(f"arm(s) {sorted(own_boot)} have their own bootstrap for treatment {switch} only; the spec "
                      f"computes {sorted(boots)}")
    arms_all = {**boot["arms"], **{a: o["entry"] for a, o in own_boot.items()}}    # the plate's arms, then their own

    present, pending = [], []
    for s in series:
        have = [a in arms_all for _, a, _ in series_points(s)]
        if all(have):
            present.append(s)
        elif not any(have):
            pending.append(s)
        else:
            raise Refusal(f"series {s['name']}: only some of its arms are in the bootstrap; a callo line needs both "
                          "points, and its variants come with it")
    names = {a for s in present for _, a in series_arms(s)}
    for k in ("reference", "second_reference"):
        if spec[k] not in names:
            raise Refusal(f"the {k} {spec[k]} is not a plotted arm")
    for k in ("reference", "second_reference"):
        if next(s for s in present if spec[k] in s.get("budgets", {}).values() or s.get("arm") == spec[k]).get(
                "plate", True) is False:
            raise Refusal(f"the {k} {spec[k]} is data-only; the plate draws every reference it names")
    plotted = [a for s in present for _, a, _ in series_points(s)]     # measured: the union and the table in FIGURES
    extra = sorted(set(boot["arms"]) - set(plotted))
    if extra and spec.get("every_arm", False):
        raise Refusal(f"the bootstrap has arm(s) {extra} that no series names")

    unanswered = _unanswered(union, boot, plotted, spec, sample)
    sample_sha = union.get("sample_sha256_uncompressed")
    if sample.get("sample_sha256_uncompressed") != sample_sha:
        raise Refusal("sample.json and the union describe different samples")

    points = []
    for s in present:
        for b, arm, _ in series_points(s):
            pt = _point(s, b, arm, arms_all[arm], spec)
            prov = _provenance(s, b, arm, spec)
            if prov["sample_sha256"] != sample_sha:
                raise Refusal(f"arm {arm} ran another sample ({str(prov['sample_sha256'])[:12]})")
            if prov["kit_commit"] != boot["kit_commit"]:
                raise Refusal(f"arm {arm} ran kit {prov['kit_commit']}, the bootstrap scored with {boot['kit_commit']}")
            prov["reread"] = _reread(arm, own_boot[arm]["doc"] if arm in own_boot else boot, spec, prov)
            if arm in own_boot:
                ob = prov["own_bootstrap"] = own_boot[arm]["record"]
                if str(ob["window_tokens"]) != str(prov["server"].get("ctx")):
                    raise Refusal(f"arm {arm}: its window counts are for {ob['window_tokens']} tokens, it ran with "
                                  f"{prov['server'].get('ctx')}")
            if prov["reread"]:
                prov["files"].append({"path": prov["reread"]["path"], "sha256": prov["reread"]["sha256"]})
            own = _own_tokens(s, prov, spec, unanswered, sample_sha)
            if own:
                prov["own_tokens"] = own
                prov["files"].append({"path": own["path"], "sha256": own["sha256"]})
            pt["provenance"] = prov
            if pt["early_exit"]:
                pt["early_exit"] = prov["early_exit"]
            pt["on_plate"] = s.get("plate", True) is not False
            points.append(pt)
    unused = sorted(set(spec.get("patches") or {}) - {p["arm"] for p in points})
    if unused:
        raise Refusal(f"the spec names a re-read record for arm(s) {unused} that the plate does not measure")

    by_arm = {p["arm"]: p for p in points}
    paired = _paired(spec, boot, by_arm)
    sens = _sensitivity(spec, boot, by_arm, own_boot,
                        ser_of={s["arm"]: s for s in series if s.get("own_bootstrap")})
    ref, second = by_arm[spec["reference"]], by_arm[spec["second_reference"]]
    ser_of = {s["name"]: s for s in series}

    def short(name: str) -> str:
        s = ser_of[name]
        return s.get("short") or s["label"]

    computed = {t: t in boots for t in TREATMENTS}
    wide = _wide(sample, switch, computed, int(spec.get("callo_max_options", CALLO_MAX_OPTIONS)))
    snap: dict[str, Any] = {
        "plate_di": SNAPSHOT_FORMAT,
        "preliminary": bool(spec.get("preliminary")),
        "title": spec.get("title") or "The Decision Index, *public subset*",
        "edition": boot["edition"], "kit_commit": boot["kit_commit"],
        "sample": {"n": union["sample_rows"], "seed": boot["seed"], "sha256": sample_sha,
                   "command": _pub_cmd(sample.get("command"), spec), "benchmarks": len(boot["groups_per_benchmark"]),
                   "groups": [min(boot["groups_per_benchmark"].values()), max(boot["groups_per_benchmark"].values())],
                   "groups_total": sum(boot["groups_per_benchmark"].values())},
        "bootstrap": {"n": boot["n"], "seed": boot["seed"], "level": boot["level"], "method": boot["method"],
                      "generated": boot["generated_utc"]},
        "reference": {"arm": ref["arm"], "label": spec.get("reference_label") or short(ref["series"]),
                      "index": ref["index"], "areas": dict(ref["areas"])},
        "second_reference": {"arm": second["arm"], "label": spec.get("second_reference_label") or short(second["series"]),
                             "index": second["index"]},
        "areas": _areas(boot, spec),
        "series": [{**{k: s.get(k) for k in ("name", "label", "short", "kind", "marker", "runtime", "accent",
                                             "not_run", "note", "label_side", "point_labels", "point_label_at")},
                    **{k: s[k] for k in ("family", "answering", "cut_text") if s.get(k)},
                    "on_plate": s.get("plate", True) is not False,
                    "budgets": sorted(s["budgets"]) if s["kind"] == "callo" else None,
                    "arms": [a for _, a in series_arms(s)],
                    "variants": sorted(s.get("variants") or {}, key=[a for _, a, _ in series_points(s)].index),
                    "rows": [a for _, a, _ in series_points(s)],
                    "status": "present" if s in present else "pending",
                    "pending_note": s.get("pending_note")} for s in series],
        "y_range": _y_range(spec),
        "points": points,
        "paired": paired,
        "unanswered": unanswered,
        "wide": wide,
        "treatments": {t: ({"label": TREATMENTS[t], "computed": True, "path": _pub(tpaths[t], spec),
                            "sha256": sha256_file(tpaths[t]),
                            "arms": {arm: _point(ser_of[by_arm[arm]["series"]], by_arm[arm]["budget"], arm,
                                                 (arms_all if t == switch else boots[t]["arms"])[arm], spec)
                                     for arm in plotted}}
                           if t in boots else {"label": TREATMENTS[t], "computed": False,
                                               "moot": t == "all-unsupported" and wide["moot"],
                                               "why": (_cap(wide["moot_text"]) if t == "all-unsupported" and wide["moot"]
                                                       else (spec.get("notes") or {}).get(f"treatment_{t}"))})
                       for t in TREATMENTS},
        "notes": dict(spec.get("notes") or {}), "hardware": spec.get("hardware"),
        "story_cfg": spec.get("story"),
        "generator": {"callo_commit": _git_commit(), "spec": figkit.public_path(os.path.abspath(spec["_spec_path"]), REPO),
                      "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "sensitivity": sens,
        "inputs": [{"role": r, "path": _pub(p, spec), "sha256": sha256_file(p)} for r, p in
                   [*((f"bootstrap, treatment {TREATMENTS[t].split(' ', 1)[0]}", tpaths[t]) for t in boots),
                    *((f"paired vs {by_arm[c['reference']]['label']}", c["_path"]) for c in paired),
                    *([(f"bootstrap without {sens['drop']}", sens["_path"])] if sens else []),
                    *((f"{by_arm[a]['label']}: {r}", p_) for a, o in own_boot.items() for r, p_ in
                      zip(("own bootstrap", "own union", "window counts"), o["_paths"], strict=True)),
                    *((f"{by_arm[a]['label']}: own bootstrap without {sens['drop']}", p_)
                      for a, p_ in (sens or {}).get("_own_paths", {}).items()),
                    ("union", spec["union"]), ("sample", spec["sample"]),
                    *((INPUT_ROLES[k], spec[k]) for k in INPUT_ROLES if spec.get(k))]],
        "pending": [s["name"] for s in pending],
    }
    if sens:
        sens.pop("_own_paths", None)
    for c in [*paired, *([sens] if sens else [])]:
        c.pop("_path")                          # the absolute path, for the inputs' sha256 only
    snap["story"] = story(snap)
    snap["fine_print"] = fine_print(snap)
    snap["made"] = made(snap)
    snap["builds"] = figkit.builds_named([p["provenance"].get("callo_commit") for p in points]
                                         + [(p["provenance"].get("reread") or {}).get("callo_commit") for p in points])
    if spec["_root"] in json.dumps(snap, ensure_ascii=False):
        raise Refusal("the snapshot would record where the bench checkout sits; every path is relative to --root")
    return snap


def _y_range(spec: dict) -> list[float] | None:
    """The index axis the spec fixes (`y_range`: [lo, hi], within 0-100, multiples of 10), else None (fitted)."""
    r = spec.get("y_range")
    if r is None:
        return None
    if (not isinstance(r, list) or len(r) != 2 or not 0 <= r[0] < r[1] <= 100
            or any(float(v) % 10 for v in r)):
        raise Refusal(f"y_range is [lo, hi] within 0-100 in tens, got {r!r}")
    return [float(r[0]), float(r[1])]


def _reread(arm: str, boot: dict, spec: dict, prov: dict) -> dict | None:
    """An arm whose results were re-read after a readout fix (the bench's di_patch.py): the patched results the
    bootstrap scored must be the ones its apply record wrote, from the run this arm's provenance names. An arm scored
    from patched results with no record is refused."""
    inp = boot["inputs"][arm]
    path = (spec.get("patches") or {}).get(arm)
    if not path:
        if ".patched." in Path(inp["results"]).name:
            raise Refusal(f"arm {arm}: the bootstrap scored re-read results ({inp['results']}); the spec names no "
                          "re-read record (patches)")
        return None
    doc = _load(path, f"the re-read record for arm {arm}")
    if doc.get("arm") != arm:
        raise Refusal(f"{path}: a re-read record for arm {doc.get('arm')!r}")
    if doc.get("patched_sha256") != inp["results_sha256"]:
        raise Refusal(f"arm {arm}: the bootstrap scored {inp['results']}, not the results {path} wrote (sha256)")
    meta = Path(path).with_name("patch_meta.json")                # the re-read run's own record, beside it
    meta = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
    if meta and (meta.get("arm") != arm or meta.get("original_results_sha256") != doc.get("original_sha256")):
        raise Refusal(f"{path}: its patch_meta.json is for another arm or another run")
    orig = str(meta.get("original_callo_commit") or doc.get("original_callo_commit") or "")
    if prov.get("callo_commit") and not orig.startswith(str(prov["callo_commit"]).removesuffix("-dirty")):
        raise Refusal(f"arm {arm}: the re-read record patched a run at callo {orig[:7]}, the arm ran {prov['callo_commit']}")
    c = doc.get("control_summary") or {}
    if doc.get("not_ok_after_patch") or c.get("status_match") != c.get("n") or not c.get("n"):
        raise Refusal(f"arm {arm}: the re-read's controls did not match the run ({c.get('status_match')} of "
                      f"{c.get('n')} in status) or a re-read row is not ok")
    return {"path": _pub(path, spec), "sha256": sha256_file(path), "callo_commit": str(doc["callo_commit"])[:7],
            "walked": doc["walked"], "control": c["n"], "status_match": c["status_match"],
            "bit_identical": c["bit_identical"], "nondeterministic": bool(doc.get("nondeterministic")),
            "argmax_changed_rows": doc.get("argmax_changed_rows")}


def _paired(spec: dict, boot: dict, by_arm: dict) -> list[dict]:
    """The paired comparisons beside the plate's own (spec `paired`: [{"path", "arms"}]): a di_bootstrap.py output
    with another reference, on the same draws (seed), kit and derived results as the plotted bootstrap, so its Δ
    pairs with the plotted numbers. Each listed arm's Δ vs that bootstrap's reference, and its mean request time
    as a share of the reference's."""
    out = []
    for c in spec.get("paired") or []:
        path = c["path"]
        doc = _load(path, "a paired bootstrap")
        if doc.get("edition") != boot["edition"] or int(doc.get("n") or 0) < MIN_RESAMPLES or doc.get("warnings"):
            raise Refusal(f"{path}: edition {doc.get('edition')}, {doc.get('n')} resamples, warnings "
                          f"{doc.get('warnings')}: not a plate comparison")
        if doc.get("seed") != boot["seed"] or doc.get("kit_commit") != boot["kit_commit"]:
            raise Refusal(f"{path}: seed {doc.get('seed')} / kit {str(doc.get('kit_commit'))[:8]} do not pair with the "
                          "plotted bootstrap")
        for arm, inp in doc["inputs"].items():
            mine = boot["inputs"].get(arm)
            if mine is None or mine["derived_sha256"] != inp["derived_sha256"] or doc["arms"][arm]["index"] != \
                    boot["arms"][arm]["index"] or doc["arms"][arm]["latency_ms"]["mean"] != \
                    boot["arms"][arm]["latency_ms"]["mean"]:
                raise Refusal(f"{path}: arm {arm} was not scored from the plotted bootstrap's results")
        ref = doc["reference"]
        if ref not in by_arm:
            raise Refusal(f"{path}: its reference {ref} is not a measured arm")
        rows = []
        for arm in c.get("arms") or []:
            if arm not in doc["arms"] or arm not in by_arm or arm == ref:
                raise Refusal(f"{path}: no comparison of arm {arm} vs {ref}")
            d = doc["arms"][arm]["delta_vs_reference"]
            lat, lat_ref = doc["arms"][arm]["latency_ms"]["mean"], doc["arms"][ref]["latency_ms"]["mean"]
            rows.append({"arm": arm, "vs": ref, "d": [d["point"], *d["ci"]], "lat_ratio": lat / lat_ref,
                         "label": by_arm[arm]["label"], "vs_label": by_arm[ref]["label"]})
        out.append({"path": _pub(path, spec), "_path": path, "sha256": sha256_file(path), "reference": ref,
                    "comparisons": rows})
    return out


INPUT_ROLES = {"prompt_tokens": "Clef prompt tokens", "prompt_tokens_rows": "Clef prompt tokens per row",
               "area_labels_from": "area labels"}


def _sensitivity(spec: dict, boot: dict, by_arm: dict, own: dict | None = None,
                 ser_of: dict | None = None) -> dict | None:
    """The index without one benchmark (spec `sensitivity`: {"path", "drop"}): a di_bootstrap.py output with
    --drop-benchmark, the kit's own retirement (the scored edition retires it; the kit's aggregate re-weights the area),
    on the plotted bootstrap's draws (seed), kit, references and results. Per measured arm the index and the paired Δ
    with and without; the shift range; whether the order of the arms holds; which paired CIs change whether they
    include 0. Latency stays the plotted bootstrap's (the reduced edition's latency rows differ). A footnote, never the
    headline."""
    cfg = spec.get("sensitivity")
    if not isinstance(cfg, dict):
        return None
    path = cfg.get("path")
    doc = _load(path, "the sensitivity bootstrap")
    if doc.get("edition") != boot["edition"] or int(doc.get("n") or 0) < MIN_RESAMPLES or doc.get("warnings"):
        raise Refusal(f"{path}: edition {doc.get('edition')}, {doc.get('n')} resamples, warnings "
                      f"{doc.get('warnings')}: not a plate comparison")
    for k in ("seed", "kit_commit", "reference", "second_reference", "n"):
        if doc.get(k) != boot.get(k):
            raise Refusal(f"{path}: {k} {doc.get(k)!r} does not pair with the plotted bootstrap's {boot.get(k)!r}")
    if doc.get("inputs") != boot["inputs"] or set(doc["arms"]) != set(boot["arms"]):
        raise Refusal(f"{path}: not scored from the plotted bootstrap's arms and results")
    dropped = doc.get("dropped") or {}
    benches = dropped.get("benchmarks") or []
    if [b.get("dataset") for b in benches] != [cfg.get("drop")]:
        raise Refusal(f"{path}: drops {[b.get('dataset') for b in benches]}, the spec says {cfg.get('drop')!r}")
    for arm, a in doc["arms"].items():
        if a["index"] != a["kit_index"]:
            raise Refusal(f"{path}: arm {arm}: the bootstrap's index {a['index']} is not the kit's {a['kit_index']}")
    refs = (boot["reference"], boot["second_reference"])
    with_, without = dict(boot["arms"]), dict(doc["arms"])
    own_paths = {}
    for arm, o in (own or {}).items():            # an arm in its own bootstrap: its own without the benchmark
        s = (ser_of or {})[arm]
        with_[arm] = o["entry"]
        without[arm] = _own_sensitivity(s, o, doc, path)
        own_paths[arm] = s["own_sensitivity"]
    arms: dict[str, dict] = {}
    flips: list[dict] = []
    for arm in by_arm:
        w, wo = with_[arm], without[arm]
        rec = {"with": w["index"], "without": wo["index"], "without_ci": wo["index_ci"]}
        for key, d, ref in (("dref", "delta_vs_reference", refs[0]), ("dsecond", "delta_vs_second_reference", refs[1])):
            if arm == ref:
                rec[key] = rec[key + "_without"] = None
                continue
            rec[key] = [w[d]["point"], *w[d]["ci"]]
            rec[key + "_without"] = [wo[d]["point"], *wo[d]["ci"]]
            if (rec[key][1] <= 0 <= rec[key][2]) != (rec[key + "_without"][1] <= 0 <= rec[key + "_without"][2]):
                flips.append({"arm": arm, "vs": ref})
        arms[arm] = rec
    order = sorted(arms, key=lambda a: (-arms[a]["with"], a))
    order_without = sorted(arms, key=lambda a: (-arms[a]["without"], a))
    shifts = [arms[a]["without"] - arms[a]["with"] for a in arms]
    b0 = benches[0]
    return {"drop": cfg["drop"], "catalog_id": b0.get("catalog_id"), "area": b0.get("area"),
            "weights_before": b0.get("area_weights_before"), "weights_after": b0.get("area_weights_after"),
            "area_weights": dropped.get("area_weights"), "scored_edition": dropped.get("scored_edition"),
            "rows": doc.get("rows"), "n": doc["n"], "seed": doc["seed"], "path": _pub(path, spec), "_path": path,
            "sha256": sha256_file(path), "_own_paths": own_paths,
            "own": {a: {"path": _pub(p_, spec), "sha256": sha256_file(p_)} for a, p_ in own_paths.items()},
            "arms": arms, "shift": [min(shifts), max(shifts)], "order_same": order == order_without,
            "order": order, "order_without": order_without, "flips": flips,
            "pairs": sum(1 for a in arms.values() for k in ("dref", "dsecond") if a[k] is not None)}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "brand/di.py", "brand/figkit.py"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + ("-dirty" if dirty else "")          # the generator's own files (di.py reads no callo code)
    except (OSError, subprocess.CalledProcessError):
        return None


# ---- the words (data layer, so the tests hold the SVG to them) ---------------------------------------------------

def _cap(s: str) -> str:
    """A sentence start (the product's name included: "Callo" in prose, `callo` only as the command)."""
    return s[:1].upper() + s[1:]


def _series(snap: dict, name: str) -> dict:
    return next(s for s in snap["series"] if s["name"] == name)


def _short(snap: dict, name: str) -> str:
    s = _series(snap, name)
    return s.get("short") or s["label"]


def _rel(d: float, lo: float, hi: float, ref: str | None) -> str:
    """A paired difference in words: "16.5 points under Clef-Flash BF16 (CI 12.8–20.4)", or, without a name, the short
    form "5.9 under (CI 2.3–9.6)"; a CI across 0 reads "level with"."""
    if lo <= 0 <= hi:
        return f"level with {ref or 'it'} ({sf1(d)}, CI {sf1(lo)} to {sf1(hi)})"
    side = "under" if d < 0 else "over"
    a, b = sorted((abs(lo), abs(hi)))
    return (f"{f1(abs(d))} points {side} {ref}" if ref else f"{f1(abs(d))} {side}") + f" (CI {f1(a)}–{f1(b)})"


def story(snap: dict) -> dict | None:
    """The subtitle (spec `story`: {"series", "think"}): the model's no-think point against the reference, and what its
    line's think point buys (with its opt-in early exit, when the line's point has one), with the mean request time
    ratios. Every number is the snapshot's own."""
    cfg = snap.get("story_cfg")
    if not cfg:
        return None
    pts = [p for p in snap["points"] if p["series"] == cfg["series"] and not p.get("variant_of")]   # the line's own
    zero = next((p for p in pts if p["budget"] == 0), None)
    think = next((p for p in pts if p["budget"] == cfg["think"]), None)
    ref = next(p for p in snap["points"] if p["arm"] == snap["reference"]["arm"])
    if zero is None or think is None:
        return None
    name, rname = _short(snap, cfg["series"]), snap["reference"]["label"]
    r0 = zero["lat"]["mean"] / ref["lat"]["mean"]
    rb = think["lat"]["mean"] / ref["lat"]["mean"]
    d0, db = zero["dref"], think["dref"]
    how = ee_text(think.get("early_exit"))
    sub = (f"{name} with no thinking scores {f1(zero['index'])}, {_rel(*d0, rname)}, at {r0:.2f}× its mean request "
           f"time; a {cfg['think']}-token think per question" + (f" with opt-in {how}" if how else "")
           + f" lifts it to {f1(think['index'])}, {_rel(*db, None)}, at {rb:.1f}× the time.")
    closed = (d0[0] - db[0]) / d0[0] if d0[0] < 0 else None
    return {"subtitle": sub, "zero": zero["arm"], "think": think["arm"], "ratio_zero": r0, "ratio_think": rb,
            "dref_zero": d0, "dref_think": db, "gap_closed": closed}


def fine_print(snap: dict) -> list[dict]:
    """The plate's fine print, in order: each entry {"lead", "text"} is one paragraph at the plate's width."""
    out: list[dict] = []
    notes = snap.get("notes") or {}
    smp, un, wide = snap["sample"], snap["unanswered"], snap["wide"]
    if snap.get("preliminary") and notes.get("preliminary"):
        out.append({"lead": "Preliminary.", "text": _cap(notes["preliminary"]) + "."})
    out.append({"lead": "Data.", "text": (
        f"Decision Index {snap['edition']} public subset: the {smp['benchmarks']} counted benchmarks"
        + (f" ({notes['subset']})" if notes.get("subset") else "")
        + f", kit {snap['kit_commit'][:8]} and its unmodified scorer, suite sample --n {smp['n']} --seed {smp['seed']} "
        f"(sha256 {smp['sha256'][:12]}…), {smp['groups'][0]}–{smp['groups'][1]} linked groups per benchmark; not the "
        "board's Full score.")})
    out.append({"lead": "Index.", "text": "The Decision Index's balanced skill: chance-corrected benchmark scores averaged "
                                          f"in five fixed-weight areas (the panels); {CHANCE_TEXT}."})
    oc = un["over_cap"]
    own = [_short(snap, p["series"]) for p in snap["points"]
           if p["provenance"].get("own_bootstrap") and p.get("on_plate", True)]
    unans = (f"Rows over {_thousands(oc.get('cap', 16384))} Clef prompt tokens (n = {oc['n']}) and any row not "
             f"answered by every arm count as wrong for every arm (the index's no-truncation rule; all {oc['n']} are "
             f"{', '.join(oc['benchmarks'])}, {oc['groups']} of its {oc['of_groups']} groups"
             + (f", {_thousands(oc['tokens'][0])}–{_thousands(oc['tokens'][1])} tokens" if oc.get("tokens") else "")
             + ", length in the 32-document schema)"
             + (f"; {notes['cap_release']}" if notes.get("cap_release") else "") + ".")
    for e in un["errors"]:
        arm = next((p for p in snap["points"] if p["arm"] == e["arm"]), None)
        who = arm["label"] if arm else e["arm"]
        unans += f" {e['row']} counts as unanswered for every arm: {e['cause']} ({who})."
    if own:
        unans += f" The exception: rows that only {_and(own)} cannot answer count as wrong for {_and(own)} alone (below)."
    out.append({"lead": "Unanswered.", "text": unans})
    for p in snap["points"]:
        if p["provenance"].get("own_tokens") and p.get("on_plate", True):
            out.append({"lead": f"{_short(snap, p['series'])}.", "text": own_tokens_text(snap, p)})
    for p in snap["points"]:
        if p["provenance"].get("own_bootstrap") and p.get("on_plate", True):
            out.append({"lead": f"{_short(snap, p['series'])}.", "text": own_window_text(snap, p)})
    sens = sensitivity_text(snap)
    if sens:
        out.append({"lead": f"Without {snap['sensitivity']['drop']}.", "text": sens})
    other = "all-unsupported" if wide["switch"] == "kit" else "kit"
    plotted = ("treatment (a), kit-faithful" if wide["switch"] == "kit" else
               "treatment (b), counted as unanswerable for every arm, so those questions score 0 for all arms (not "
               "excluded with renormalised weights)" + (", here equal to (a)" if wide.get("moot") else ""))
    tag, words = TREATMENTS[other].split(" ", 1)
    other_txt = (f"{tag}, {words}, is in FIGURES.md" if wide["computed"][other]
                 else f"{tag}, {words}, is {wide['moot_text']}" if other == "all-unsupported" and wide.get("moot")
                 else f"{tag}, {words}, is not computed for this bake")
    over = ", none over 255 options" if not wide["rows_gt255"] else f", {wide['rows_gt255']} over 255 options"
    out.append({"lead": "> 26 options.", "text": (
        f"{wide['rows_gt26']} rows ({', '.join(f'{n} {k}' for n, k in wide['benchmarks'])}){over}"
        + (f"; {notes['wide_choice']}" if notes.get("wide_choice") else "")
        + f". Plotted: {plotted}; {other_txt}.")})
    out.append({"lead": "Answering.", "text": _cap(ANSWERING_TEXT)})
    ee = _early_exit_text(snap)
    if ee:
        out.append({"lead": "Early exit.", "text": ee})
    own_lat = [f"{_short(snap, p['series'])}'s over the {_thousands(p['lat']['n'])} rows it answered"
               for p in snap["points"] if p["provenance"].get("own_bootstrap") and p.get("on_plate", True)]
    out.append({"lead": "Latency.", "text": (
        "Request completion time, measured by one HTTP client that is the same for every arm, over the "
        f"{_thousands(un['latency_rows'])} rows answered by every arm" + "".join(f" ({x})" for x in own_lat)
        + ": points at the mean, tails to the p95.")})
    if snap.get("preliminary") and notes.get("latency"):
        out.append({"lead": "Timing.", "text": _cap(notes["latency"]) + "."})
    elif notes.get("timing"):
        out.append({"lead": "Timing.", "text": _cap(notes["timing"]) + "."})
    bs = snap["bootstrap"]
    out.append({"lead": "Whiskers.", "text": (
        f"Paired {bs['level'] * 100:.0f} % percentile bootstrap CIs of the difference vs {snap['reference']['label']} "
        f"({_thousands(bs['n'])} resamples of linked groups within each benchmark, the same draws for every arm, the "
        f"index's own scorer in the loop, seed {bs['seed']}). The difference vs "
        f"{snap['second_reference']['label']} is in the table only.")})
    rr = _reread_text(snap)
    if rr:
        out.append({"lead": "Re-read.", "text": rr})
    if notes.get("run_to_run"):
        out.append({"lead": "Run to run.", "text": _run_to_run_text(snap, notes["run_to_run"])})
    if notes.get("board"):
        out.append({"lead": "Clef cross-check.", "text": _cap(notes["board"]) + "."})
    out.append({"lead": "Builds.", "text": _builds_text(snap)})
    pend = [s.get("pending_note") or f"{s['label']} at budgets {' and '.join(map(str, s['budgets'] or []))}"
            for s in snap["series"] if s["status"] == "pending"]
    if pend:
        out.append({"lead": "Pending.", "text": "; ".join(pend) + "."})
    return out


def own_tokens_text(snap: dict, p: dict) -> str:
    """A model that reads one question per prompt (d1), on the over-cap rows: its own prompts there would fit its
    context, but the rows count as wrong for every arm all the same (one answered set for every arm)."""
    own, oc = p["provenance"]["own_tokens"], snap["unanswered"]["over_cap"]
    return (f"{_cap(_series(snap, p['series'])['answering'])}, so its prompts on the {oc['n']} "
            f"{_and(oc['benchmarks'])} rows ({_thousands(own['over_cap_question_max'][1])} tokens at most) would fit "
            f"its {_thousands(own['n_ctx'])}-token context; they still count as wrong for every arm.")


def own_window_text(snap: dict, p: dict) -> str:
    """A model with a short window (Laya): the rows cut to fit it, and the rows that do not fit at all, which count as
    wrong for it only (its own bootstrap on the plate's draws), so no other arm's number moves."""
    ob, short = p["provenance"]["own_bootstrap"], _short(snap, p["series"])
    cut = _series(snap, p["series"]).get("cut_text")
    return (f"{_thousands(ob['cut'])} of the {_thousands(ob['answered_by_others'])} rows the other arms answer were "
            f"cut to its {_thousands(ob['window_tokens'])}-token window" + (f" ({cut})" if cut else "")
            + f"; {ob['misses']} rows that do not fit at all ("
            + _and([f"{n} {k}" for k, n in ob["misses_by"]])
            + f": the question and its options alone pass {_thousands(ob['window_tokens'])} tokens) count as wrong "
            f"for {short} only, scored on the same draws, so no other arm's number moves.")


def _one_line(lead: str, text: str) -> bool:
    """Whether a fine-print entry sets on one line at the plate's width (the drawing's lead and text styles)."""
    import figkit as fk
    return (fk.advance(lead + " ", fk.Style(size=14, weight=600)) + fk.advance(text, fk.Style(size=14))
            <= W - 2 * SIDE)


def sensitivity_text(snap: dict) -> str | None:
    """The plate's footnote on the index without one benchmark: how far every arm moves, whether the order and the
    paired CIs' verdict on 0 hold, the reference's index without it and, when the line still fits on one line, each
    plotted callo line's indexes without it."""
    s = snap.get("sensitivity")
    if not s:
        return None
    lo, hi = s["shift"]
    order = "the order is unchanged" if s["order_same"] else "the order of the arms changes"
    n = len(s["flips"])
    flips = ("no paired CI changes whether it includes 0" if not n else
             f"{n} of {s['pairs']} paired CIs change whether they include 0")
    base = f"Every arm {sf1(lo)} to {sf1(hi)} (the index's own re-weighting, the same draws); {order} and {flips}"
    ref = snap["reference"]
    tail = f"; {ref['label']} {f1(s['arms'][ref['arm']]['without'])}"
    lines = []
    for ser in snap["series"]:
        if ser["kind"] != "callo" or ser["status"] != "present" or not ser.get("on_plate", True):
            continue
        vals = [f1(s["arms"][a]["without"]) for a in ser["arms"] if a in s["arms"]]
        if vals:
            lines.append(f"{ser.get('short') or ser['label']} {' and '.join(vals)}")
    lead = f"Without {s['drop']}."
    full = base + tail + (", " + ", ".join(lines) if lines else "") + "."
    for cand in (full, base + tail + "."):
        if _one_line(lead, cand):
            return cand
    return base + tail + "."


def _prose(label: str) -> str:
    """A point's label in running text: "Tiel-Coder · think 256" -> "Tiel-Coder think 256"."""
    return label.replace(" · ", " ")


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _zt(z: float) -> str:
    """z .95 (the docs' spelling)."""
    return "z " + f"{z:.2f}".lstrip("0")


def _lat_change(ratio: float, *, short: bool = False) -> str:
    """A mean request time ratio in words: "30 % less mean request time" ("30 % less"), "3.2× the mean request time"."""
    if ratio < 1:
        return f"{100 * (1 - ratio):.0f} % less" + ("" if short else " mean request time")
    return f"{ratio:.1f}×" + (" the time" if short else " the mean request time")


def _variant_name(v: dict, point: dict) -> str:
    """How the plate names a line's other arm at its think budget, beside the line's point: "plain think 256", "the
    gate variant" (the point's early exit with the gate), "think 256 with early exit"."""
    ve, pe, b = v.get("early_exit"), point.get("early_exit"), budget_text(v["budget"])
    if not ve:
        return f"plain {b}"
    if pe and ve["z"] == pe["z"] and ve.get("gate") and not pe.get("gate"):
        return "the gate variant"
    return f"{b} with {ee_text(ve)}"


def _early_exit_text(snap: dict) -> str | None:
    """What early exit is, in plain words; a line whose think point exits early, against its plain think on the same
    draws (with the line's other arms at that budget, which are table rows, not drawn); the gate's share and read cost
    on each gated line point; the other paired comparisons. Every number is the snapshot's own."""
    pts = [p for p in snap["points"] if p.get("on_plate", True)]
    by = {p["arm"]: p for p in pts}
    ee_pts = [p for p in pts if p.get("early_exit")]
    if not ee_pts:
        return None
    notes = snap.get("notes") or {}
    zs = sorted({p["early_exit"]["z"] for p in ee_pts})
    txt = (f"Opt-in in Callo (--think-converge, {_and([_zt(z) for z in zs])}): Callo stops a think once the answer has "
           "settled instead of using the whole budget.")
    ablations, others = [], []
    for c in snap.get("paired") or []:
        ref = by.get(c["reference"])
        rows = [(r, by[r["arm"]]) for r in c["comparisons"] if r["arm"] in by]
        if ref is None or not rows:
            continue
        line = [(r, p) for r, p in rows if not p.get("variant_of")]
        if (ref.get("variant_of") and len(line) == 1 and line[0][1].get("early_exit")
                and all(p["series"] == ref["series"] and p["budget"] == ref["budget"] for _, p in rows)):
            ablations.append((ref, line[0], [(r, p) for r, p in rows if p is not line[0][1]]))
        else:
            rows = [(r, p) for r, p in rows if p.get("early_exit")]
            if rows:
                others.append((ref, rows))
    for ref, (r, p), rest in ablations:
        b = budget_text(p["budget"])
        txt += (f" {_short(snap, p['series'])}'s line ends at {b} with {ee_text(p['early_exit'])}: against "
                f"{_variant_name(ref, p)} on the same draws, index {sf1(r['d'][0])} "
                f"{ci1(r['d'][1], r['d'][2], signed=True)} at {_lat_change(r['lat_ratio'])}")
        for r2, p2 in rest:
            txt += (f"; {_variant_name(p2, p)} {sf1(r2['d'][0])} {ci1(r2['d'][1], r2['d'][2], signed=True)} at "
                    f"{_lat_change(r2['lat_ratio'], short=True)}")
        hidden = [_variant_name(ref, p)] + [_variant_name(p2, p) for _, p2 in rest]
        txt += f". {_cap(_and(hidden))} {'are' if len(hidden) > 1 else 'is'} in the table, not drawn."
    gated = [p for p in ee_pts if p["early_exit"].get("gate")]
    if gated:
        txt += (" The gate first asks the model one zero-think question and sends multi-step rule problems to full "
                "thinking")
        shares = []
        for p in gated:
            if p.get("variant_of"):
                continue                           # a variant's share is in FIGURES.md; the plate names the line's
            g = p["early_exit"]["gate_reads"]
            s = _short(snap, p["series"])
            shares.append(f"on {s} it sent {_thousands(g['skipped'])} of {_thousands(g['asked'])} requests "
                          f"({100 * g['skipped'] / g['asked']:.0f} %), and its read took about "
                          f"{secs(g['read_ms_session']['p50'])} at the median"
                          + (f" {notes['gate_read']}" if notes.get("gate_read") else ""))
        txt += ("; " + "; ".join(shares) if shares else "") + "."
    for ref, rows in others:
        same = all(p["series"] == ref["series"] for _, p in rows)
        parts, less, more = [], [], []
        for r, p in rows:
            if p["budget"] == ref["budget"]:
                name = ee_text(p["early_exit"])
            elif same:
                name = f"{budget_text(p['budget'])} with {ee_text(p['early_exit'])}"
            else:
                name = _prose(p["label"])
            parts.append(f"{name} {sf1(r['d'][0])} {ci1(r['d'][1], r['d'][2], signed=True)}")
            (less if r["lat_ratio"] < 1 else more).append(r["lat_ratio"])
        when = []
        if less:
            when.append(f"at {_and([f'{100 * (1 - x):.0f} %' for x in less])} less mean request time")
        if more:
            when.append(f"at {_and([f'{x:.1f}×' for x in more])} the mean request time")
        txt += f" Paired vs {_prose(ref['label'])}: {_and(parts)}, {' and '.join(when)}."
    return txt


def _by_series(snap: dict, pts: list[dict]) -> str:
    """Points named by model: "Tiel-Coder (no think, think 256) and Sharp-MiniCPM5-2B think 1024"."""
    names = []
    for s in snap["series"]:
        own = [p for p in pts if p["series"] == s["name"]]
        if not own:
            continue
        tail = [budget_text(p["budget"]) + (f", {ee_text(p['early_exit'])}" if p.get("early_exit") else "")
                for p in own] if s["kind"] == "callo" else []
        short = s.get("short") or s["label"]
        names.append(short + (f" {tail[0]}" if len(tail) == 1 else f" ({', '.join(tail)})" if tail else ""))
    return _and(names)


def _reread_text(snap: dict) -> str | None:
    pts = [p for p in snap["points"] if p["provenance"].get("reread") and p.get("on_plate", True)]
    if not pts:
        return None
    notes = snap.get("notes") or {}
    rr = [p["provenance"]["reread"] for p in pts]
    builds = [b["name"] for b in _builds_in_order({r["callo_commit"] for r in rr})]
    ctl = sorted({r["control"] for r in rr})
    exact = [p for p in pts if p["provenance"]["reread"]["bit_identical"] == p["provenance"]["reread"]["control"]]
    exact_series = sorted({p["series"] for p in exact}, key=[s["name"] for s in snap["series"]].index)
    return (f"{_by_series(snap, pts)} ran before a readout fix"
            + (f" ({notes['reread_fix']})" if notes.get("reread_fix") else "")
            + f"; their {_and([_thousands(r['walked']) for r in rr])} rows with an affected readout were re-read on "
            f"{_and(builds)}, and {_and([str(n) for n in ctl])} unaffected rows per arm re-run as controls "
            "matched in status"
            + (f", bit for bit on {_and([_short(snap, s) for s in exact_series])}" if exact_series else "") + ".")


def _run_to_run_text(snap: dict, note: str) -> str:
    loose = [p for p in snap["points"] if (p["provenance"].get("reread") or {}).get("nondeterministic")
             and p.get("on_plate", True)]
    txt = _cap(note)
    for s in snap["series"]:
        own = [p["provenance"]["reread"] for p in loose if p["series"] == s["name"]]
        if own:
            n = sorted({r["control"] for r in own})
            txt += (f" ({s.get('short') or s['label']}'s re-read controls: "
                    f"{_and([str(r['bit_identical']) for r in own])} of {_and([str(x) for x in n])} rows bit for bit)")
    return txt + "."


def _builds_text(snap: dict) -> str:
    drawn = [p for p in snap["points"] if p.get("on_plate", True)]
    callo = [p for p in drawn if p["kind"] == "callo"]
    by_commit: dict[str, list[str]] = {}
    ee_commits = {p["provenance"]["callo_commit"] for p in callo if p.get("early_exit")}
    ee_alone = (len(ee_commits) == 1 and not [p for p in callo if not p.get("early_exit")
                                              and p["provenance"]["callo_commit"] in ee_commits])
    for s in snap["series"]:
        own = [p for p in callo if p["series"] == s["name"]]
        commits = {p["provenance"]["callo_commit"] for p in own}
        for c in sorted(commits, key=[p["provenance"]["callo_commit"] for p in callo].index):
            here = [p for p in own if p["provenance"]["callo_commit"] == c]
            if ee_alone and c in ee_commits:
                who = "the early-exit points"
            elif len(commits) == 1:          # a model whose points all ran at one commit is named once
                who = s.get("short") or s["label"]
            else:
                who = (s.get("short") or s["label"]) + " " + _and([budget_text(p["budget"]) for p in here])
            if who not in by_commit.setdefault(c, []):
                by_commit[c].append(who)
    first = {c: min(str(p["provenance"]["utc_start"]) for p in callo if p["provenance"]["callo_commit"] == c)
             for c in by_commit}
    by_commit = dict(sorted(by_commit.items(), key=lambda kv: first[kv[0]]))     # in the order they ran
    parts = []
    named = {c: build_of(c) for c in by_commit}                      # by role, never by commit
    versions = ", ".join(sorted({b["version"] for b in named.values()}))
    if len(by_commit) == 1:
        parts.append(f"Callo {versions} ({named[next(iter(by_commit))]['name']})")
    elif by_commit:
        parts.append(f"Callo {versions} development builds: "
                     + _and([f"{named[c]['name']} ({'; '.join(ls)})" for c, ls in by_commit.items()]))
    notes = snap.get("notes") or {}
    if len(by_commit) > 1 and notes.get("builds"):
        parts.append(notes["builds"])
    caches = {p["provenance"]["cache"] for p in callo}
    if caches:
        parts.append("--cache " + "/".join(sorted(caches)))
    builds: dict[str, list[str]] = {}
    for p in drawn:
        builds.setdefault(p["provenance"]["build"], [])
        k = family(_series(snap, p["series"]))
        if k not in builds[p["provenance"]["build"]]:
            builds[p["provenance"]["build"]].append(k)
    parts.append("llama.cpp " + _and([f"{b} ({', '.join(ks)})" for b, ks in builds.items()]))
    for s in snap["series"]:
        if s["kind"] == "callo" and s["status"] == "present" and s.get("not_run"):
            parts.append(f"{s.get('short') or s['label']} budgets {' and '.join(map(str, s['not_run']))} not run on DI "
                         f"(only {' and '.join(map(str, s['budgets']))})")
    dates: dict[str, set[str]] = {}
    for p in drawn:
        k = family(_series(snap, p["series"]))
        dates.setdefault(k, set()).update({str(p["provenance"]["utc_start"])[:10], str(p["provenance"]["utc_end"])[:10]})
    when = ", ".join(f"{k} {min(v)}" + (f" to {max(v)}" if max(v) != min(v) else "") for k, v in dates.items())
    if snap.get("hardware"):
        parts.append(snap["hardware"])
    parts.append(f"run {when} (UTC)")
    return "; ".join(parts) + "."


def _builds_in_order(commits: set[str]) -> list[dict]:
    """The builds `commits` are, in the order they were made (data/builds.json)."""
    import figkit
    have = {build_of(c)["commit"] for c in commits}
    return [b for b in figkit.builds() if b["commit"] in have]


def _dtxt(d: list[float] | None) -> str:
    return "—" if d is None else f"{sf1(d[0])} {ci1(d[1], d[2], signed=True)}"


def named(path: str, sha: str | None = None) -> str:
    """A results file as FIGURES.md names it: its file name and sha256 (the bench checkout's layout is not public)."""
    return f"`{Path(str(path)).name}`" + (f" (sha256 {sha[:16]}…)" if sha else "")


def _local_names(cmd: str | None) -> str:
    """A recorded command with each path in it reduced to its file name."""
    return " ".join(Path(t).name if "/" in t else t for t in str(cmd).split(" "))


def _wtxt(w: dict | None) -> str:
    """An area's in-benchmark weights, equal weights grouped: "4, 5 and 36: 0.1818; 37, 56 and 61: 0.1515"."""
    groups: dict[float, list[str]] = {}
    for k, v in (w or {}).items():
        groups.setdefault(round(float(v), 4), []).append(str(k))
    return "; ".join(f"{_and(ks)}: {v:.4f}" for v, ks in sorted(groups.items(), key=lambda kv: -kv[0]))


def _sensitivity_made(snap: dict) -> str:
    s = snap["sensitivity"]
    labels = {p["arm"]: p["label"] for p in snap["points"]}
    lo, hi = s["shift"]
    refs = f"vs {snap['reference']['label']} and vs {snap['second_reference']['label']}"
    flips = (f"none of the {s['pairs']} paired CIs ({refs}) changes whether it includes 0" if not s["flips"] else
             f"{len(s['flips'])} of the {s['pairs']} paired CIs change whether they include 0: "
             + "; ".join(f"{labels[x['arm']]} vs {labels[x['vs']]}" for x in s["flips"]))
    aw = (s.get("area_weights") or {}).get(s.get("area"))
    return (f"{s['drop']} (catalog {s['catalog_id']}) dropped by the Decision Index kit's own retirement: the scored "
            f"edition is {s['scored_edition']}, and its aggregate re-weights the {s['area']} area's benchmarks from "
            f"{_wtxt(s['weights_before'])} to {_wtxt(s['weights_after'])}; the area weights are unchanged"
            + (f" ({s['area']} {100 * aw:.2f} %)" if aw is not None else "")
            + f". The plotted bootstrap's {_thousands(s['n'])} draws (seed {s['seed']}) and results, "
            f"{_thousands(s['rows'])} rows; latency stays the plotted bootstrap's. Every arm {sf1(lo)} to {sf1(hi)} "
            f"({lo:+.2f} to {hi:+.2f}); the order of all {len(s['arms'])} arms is "
            + ("unchanged" if s["order_same"] else "changed: " + ", ".join(labels[a] for a in s["order_without"]))
            + f"; {flips}. Per arm in the table below; source {named(s['path'], s['sha256'])}.")


def made(snap: dict) -> list[dict]:
    """How the plate was made, for FIGURES.md: provenance per arm, the union, the explained errors, the inputs."""
    out: list[dict] = []
    smp, bs, un = snap["sample"], snap["bootstrap"], snap["unanswered"]
    out.append({"lead": "Sample.", "text": f"{_local_names(smp['command'])} (sha256 of the uncompressed rows "
                f"{smp['sha256']}); "
                f"{smp['n']} rows, {smp['groups_total']} linked groups in {smp['benchmarks']} benchmarks."})
    out.append({"lead": "Bootstrap.", "text": f"{bs['method']}; {bs['n']} resamples, seed {bs['seed']}, generated "
                f"{bs['generated']}."})
    w = [a for a in snap["areas"] if a.get("weight") is not None]
    if w:
        out.append({"lead": "Areas.", "text": "the index's fixed weights: " + ", ".join(f"{a['label']} {a['weight']:g} %"
                                                                                       for a in w) + "."})
    out.append({"lead": "Unanswered rows.", "text": (
        f"{un['union_rows']} of {un['sample_rows']} sampled rows are not answered by every arm ("
        + ", ".join(f"{n} {k}" for k, n in un["by_reason"].items()) + f"); latency over the other "
        f"{un['latency_rows']}. Pre-marked: {', '.join(un['over_cap']['run_ids'])}.")})
    for e in un["errors"]:
        out.append({"lead": f"Error row {e['run_id']} ({e['arm']}).", "text": f"{' '.join(str(e['error']).split())} Cause: {e['cause']}."})
    notes = snap.get("notes") or {}
    sens = snap.get("sensitivity")
    if sens:
        out.append({"lead": f"Sensitivity without {sens['drop']}.", "text": _sensitivity_made(snap)})
    elif notes.get("sensitivity"):
        out.append({"lead": "Sensitivity without BRIGHT.", "text": notes["sensitivity"]})
    if notes.get("latency_figures"):
        out.append({"lead": "Clef timings.", "text": _cap(notes["latency_figures"]) + "."})
    ref = snap["reference"]["label"]
    for p in snap["points"]:
        if not p.get("on_plate", True):            # measured and checked, reported here, not drawn
            d = p["dref"]
            out.append({"lead": f"{p['label']} (measured, not drawn).", "text": (
                f"index {p['index']:.2f} {ci1(*p['index_ci'])}, {_rel(*d, ref) if d else 'the reference'}; mean "
                f"request time {secs(p['lat']['mean'])}, p95 {secs(p['lat']['p95'])}.")})
    for p in snap["points"]:
        pv = p["provenance"]
        # the real file name, as a code span (a path, never a display name)
        bits = [f"`{Path(pv['model']).name}`" + (f" (sha256 {pv['model_sha256'][:16]}…)" if pv.get("model_sha256") else "")]
        if p["kind"] == "callo":
            b = build_of(pv["callo_commit"])
            bits.append(f"`callo` {b['version']}, {b['name']}, --think-budget {pv['think_budget']}, --cache {pv['cache']}, "
                        f"think profile {Path(str(pv['think_profile'])).name} (sha256 "
                        f"{str(pv['think_profile_sha256'])[:16]}…)")
            ee = pv.get("early_exit")
            if ee:
                bits.append(f"early exit --think-converge {ee['z']}" + (" --think-converge-gate" if ee["gate"] else "")
                            + f": {_thousands(ee['questions'])} questions on {_thousands(ee['requests'])} requests "
                            f"thought with a rule, {_thousands(ee['converged'])} converged, "
                            f"{_thousands(ee['probes'])} probes (the counters less the warm-up's "
                            f"{ee['warmup_requests']} requests)")
                g = ee.get("gate_reads")
                if g:
                    bits.append(f"gate: {_thousands(g['asked'])} reads, {_thousands(g['skipped'])} sent to full "
                                f"thinking ({100 * g['skipped'] / g['asked']:.1f} %), {g['errors']} errors; read "
                                f"p50 {secs(g['read_ms_session']['p50'])}, p95 {secs(g['read_ms_session']['p95'])} "
                                "over the session (warm-up reads included)")
            rr = pv.get("reread")
            if rr:
                bits.append(f"re-read after the readout fix: {rr['walked']} rows on {build_of(rr['callo_commit'])['name']}; "
                            f"controls {rr['status_match']} of {rr['control']} in status, {rr['bit_identical']} bit for "
                            f"bit; {named(rr['path'], rr['sha256'])}")
            if pv.get("server_args"):
                args = [Path(a).name if "/" in a else a for a in pv["server_args"]]
                bits.append("llama-server args " + " ".join(args))
        bits.append(f"llama.cpp {pv['build']}; server " + ", ".join(f"{k} {v}" for k, v in pv["server"].items() if v))
        ob = pv.get("own_bootstrap")
        if ob:
            bits.append(f"its own bootstrap {named(ob['path'], ob['sha256'])} on the plotted bootstrap's draws (the "
                        f"references' numbers equal), from its own union {named(ob['union'], ob['union_sha256'])}: the "
                        f"plate's {snap['unanswered']['union_rows']} rows not answered by every arm, every other arm's "
                        f"results byte-identical, and its own {ob['misses']} arm-only rows ("
                        + _and([f"{n} {k}" for k, n in ob["misses_by"]]) + f"); its window counts "
                        f"{named(ob['window'], ob['window_sha256'])}: {_thousands(ob['window_tokens'])} tokens, "
                        f"{_thousands(ob['cut'])} rows cut" + (f" ({ob['truncation']})" if ob.get("truncation") else "")
                        + f", latency over its {_thousands(ob['latency_rows'])} answered rows")
        own = pv.get("own_tokens")
        if own:
            lo, hi = own["over_cap_question_max"]
            bits.append(f"its own prompt-token counts {named(own['path'], own['sha256'])} (the server's own code, "
                        f"offline): {_thousands(own['sent_questions'])} questions on {_thousands(own['sent_rows'])} "
                        f"rows, one prompt each, the longest {_thousands(own['sent_question_max'])} tokens; on the "
                        f"{len(snap['unanswered']['over_cap']['run_ids'])} pre-marked rows the longest prompt per row "
                        f"is {_thousands(lo)}–{_thousands(hi)} tokens")
        bits.append(f"run {pv['utc_start']} to {pv['utc_end']}" + (f" (session {pv['sessions']} of {pv['sessions']})"
                                                                   if pv["sessions"] > 1 else ""))
        bits.append(f"mean request time {secs(p['lat']['mean'])} [{secs(p['lat']['mean_ci'][0])}, "
                    f"{secs(p['lat']['mean_ci'][1])}]")
        out.append({"lead": f"{p['label']}.", "text": "; ".join(bits) + "."})
    for c in snap.get("paired") or []:
        ref_label = next(p["label"] for p in snap["points"] if p["arm"] == c["reference"])
        out.append({"lead": f"Paired vs {ref_label}.", "text": (
            f"{named(c['path'], c['sha256'])}, the same draws, kit and results as the plotted bootstrap: "
            + "; ".join(f"{r['label']} {_dtxt(r['d'])}, mean request time ×{r['lat_ratio']:.3f}"
                        for r in c["comparisons"]) + ".")})
    for s in snap["series"]:
        if s.get("note"):
            out.append({"lead": None, "text": s["note"]})
    return out


# ---- drawing -----------------------------------------------------------------------------------------------------

W = 1280                       # the plate's width (the brand plate's)
SIDE = 64
PLOT_X = (112, W - SIDE)       # the main plot's x range; the area panels and the table share it
PLOT_H = 580
PANEL_H = 132
PANEL_GAP = 26
ROW_H = 27


def _f(v: float) -> str:
    return repr(float(v))


def _title_runs(title: str, st_sans, st_serif) -> list:
    """`*...*` marks the Newsreader italic part of a title (the brand's second voice)."""
    return [(part, st_serif if i % 2 else st_sans) for i, part in enumerate(title.split("*")) if part]


def _box_attr(b) -> str:
    return ",".join(_f(v) for v in (b.x0, b.y0, b.x1, b.y1))


def _ptr(arm: str, path: str) -> str:
    """A point's field in the snapshot, addressed by arm: /points/<arm>/<path>."""
    return f"/points/{arm}/{path}"


def _key(ptr: str, v: float) -> str:
    """A drawn number's address in the snapshot (round-tripped by the tests)."""
    return f'data-key="{ptr}" data-value="{_f(v)}"'


def refuse_unexplained(snap: dict) -> None:
    """The plot step's own check: an arm with an error row whose cause is unknown is never drawn."""
    drawn = {p["arm"] for p in snap["points"]}
    bad = [e for e in (snap.get("unanswered") or {}).get("errors") or [] if not e.get("cause") and e["arm"] in drawn]
    if bad:
        raise Refusal(f"arm {bad[0]['arm']}: {len(bad)} unexplained error row(s) (first {bad[0]['run_id']}); not drawn")


def styles(snap: dict, t: dict) -> dict[str, dict]:
    """Series styles: Signal (the one accent) for the accent callo series; the other callo series in the secondary
    colour, thin, solid then dashed (a third line fits); Clef in the type colour, diamonds (BF16 filled, Q4 hollow)."""
    out: dict[str, dict] = {}
    others = iter([(1.4, None), (1.4, "6 3"), (1.4, "2 3"), (1.4, "8 3 2 3")])
    for s in snap["series"]:
        if s["kind"] == "callo" and s.get("accent"):
            out[s["name"]] = {"color": t["accent"], "w": 2.4, "dash": None, "marker": s.get("marker") or "circle",
                              "fill": t["accent"], "name": t["accent"]}
        elif s["kind"] == "callo":
            w, dash = next(others, (1.4, "1 3"))
            out[s["name"]] = {"color": t["sub"], "w": w, "dash": dash, "marker": s.get("marker") or "circle",
                              "fill": t["ground"], "name": t["ink"]}
        else:
            m = s.get("marker") or "diamond"
            out[s["name"]] = {"color": t["ref"], "w": 1.6, "dash": None, "marker": m,
                              "fill": t["ground"] if m.endswith("hollow") else t["ref"], "name": t["ink"]}
    return out


def _mark(fk, t: dict, st: dict, x: float, y: float, *, big: bool = False, scale: float = 1.0, sw: float = 1.7,
          halo_w: float = 4.5) -> tuple[str, str, float]:
    """A point's mark in its series' marker: (ground halo, shape, radius)."""
    halo, _ = fk.marker(st["marker"], x, y, t["ground"], t["ground"], sw + halo_w, big=big, scale=scale)
    shape, r = fk.marker(st["marker"], x, y, st["fill"], st["color"], sw, big=big, scale=scale)
    return halo, shape, r


def _drawn(snap: dict) -> list[dict]:
    """The points the plot draws: each plotted series' line points and single markers. A line's other arms at its
    think budget (`variant_of`) are rows of the plate's table, never drawn; a data-only arm is in FIGURES.md only."""
    return [p for p in snap["points"] if p.get("on_plate", True) and not p.get("variant_of")]


X_TITLE = "Mean request completion time (log scale)"
Y_TITLE = "Decision Index, public subset  ↑"
CHART_TITLE = "Decision Index"
CHART_Y_TITLE = "index, public subset  ↑"
CHART_TYPE = 1.15                # the README's chart sets its type a size up: it is read at ~880 px wide


def _plot(fk, t: dict, snap: dict, sty: dict, plot, *, y_title: str, ts: float = 1.0) -> list[str]:
    """The index against the mean request completion time (log) with p95 tails and paired-CI whiskers vs the
    reference: the y title and the key above the plot, the grid, the callo lines, the marks and every label. The plate
    and the README's chart draw it the same way; `ts` scales its type (the chart, read smaller, sets it larger)."""
    pts = _drawn(snap)
    by_arm = {p["arm"]: p for p in pts}
    present = [s for s in snap["series"] if s["status"] == "present" and s.get("on_plate", True)]
    ref = snap["reference"]
    x0, y0, x1, y1 = plot.x0, plot.y0, plot.x1, plot.y1
    body: list[str] = []

    # ---- scales: log mean time over the 1-2-5 values around every mean and p95; the index over every whisker
    lat_lo = min(p["lat"]["mean"] for p in pts)
    lat_hi = max(p["lat"]["p95"] for p in pts)
    d0, d1 = fk.log_bounds(lat_lo / 1.15, lat_hi)
    xs = fk.LogScale(d0, d1, x0 + 8, x1 - 8)

    def ends(p: dict) -> tuple[float, float]:
        if p["dref"] is None:
            return p["index"], p["index"]
        d, lo, hi = p["dref"]
        return p["index"] + (lo - d), p["index"] + (hi - d)

    a_lo = min(min(ends(p)) for p in pts)
    a_hi = max(max(ends(p)) for p in pts)
    if snap.get("y_range"):                   # the spec fixes the axis (release: 0-70); nothing may fall off it
        g_lo, g_hi = snap["y_range"]
        step = 10.0
        if a_lo < g_lo or a_hi > g_hi:
            raise Refusal(f"y_range {g_lo:g}-{g_hi:g} clips a point or whisker ({a_lo:.1f}-{a_hi:.1f})")
    else:
        step = 10.0 if a_hi - a_lo > 24 else 5.0
        g_lo = max(0.0, math.floor((a_lo - 2) / step) * step)
        g_hi = min(100.0, math.ceil((a_hi + 3) / step) * step)
    ys = fk.LinScale(g_lo, g_hi, y1 - 6, y0 + 6)

    def X(p: dict) -> float:
        return xs(p["lat"]["mean"])

    def Y(p: dict) -> float:
        return ys(p["index"])

    # ---- grid, axes, key
    tick_st = fk.Style(size=13 * ts, fill=t["sub"])
    for v in fk.log_ticks(d0, d1):
        x = xs(v)
        body.append(f'<line x1="{fk.num(x)}" x2="{fk.num(x)}" y1="{y0}" y2="{y1}" stroke="{t["grid"]}"/>')
        body.append(fk.text(fk.fmt_ms(v), x, y1 + 11 + 13 * ts, tick_st, anchor="middle", role="tick")[0])
    k = 0
    while g_lo + k * step <= g_hi + 1e-9:
        a = g_lo + k * step
        y = ys(a)
        body.append(f'<line x1="{x0}" x2="{x1}" y1="{fk.num(y)}" y2="{fk.num(y)}" stroke="{t["grid"]}"/>')
        body.append(fk.text(f"{a:.0f}", x0 - 12, y + 4.5 * ts, tick_st, anchor="end", role="tick")[0])
        k += 1
    body.append(f'<line x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}" stroke="{t["axis"]}"/>')
    ax_st = fk.Style(size=13.5 * ts, fill=t["sub"])
    body.append(fk.text(X_TITLE, (x0 + x1) / 2, y1 + 52 * ts, ax_st, anchor="middle", role="axis-title")[0])
    body.append(fk.text(y_title, SIDE, y0 - 16, ax_st, role="axis-title")[0])
    body.append(_legend(fk, t, x1, y0 - 16, ref["label"], k=ts))

    placer = fk.Placer(bounds=plot.pad(-2))
    lines: list[str] = []
    marks: list[str] = []
    labels: list[str] = []

    owner: list[str] = []          # whose each placer.lines box is (a point's arm, a line's series), in step with it

    def block(box, who: str) -> None:
        placer.lines.append(box)
        owner.append(who)

    segs: list[tuple[float, float, float, float, str]] = []      # every line and leader drawn, as itself

    def segment(ax_: float, ay_: float, bx_: float, by_: float, pad: float = 2.0, *, who: str = "") -> None:
        segs.append((ax_, ay_, bx_, by_, who))
        n = max(1, int(math.hypot(bx_ - ax_, by_ - ay_) / 6))
        for i in range(n + 1):
            x, y = ax_ + (bx_ - ax_) * i / n, ay_ + (by_ - ay_) * i / n
            block(fk.Box(x - pad, y - pad, x + pad, y + pad), who)

    # ---- tails and whiskers
    for p in pts:
        st = sty[p["series"]]
        x, y = X(p), Y(p)
        xt = xs(p["lat"]["p95"])
        lines.append(f'<line data-role="tail" data-arm={fk.esc(p["arm"])} {_key(_ptr(p["arm"], "lat/p95"), p["lat"]["p95"])} '
                     f'x1="{_f(x)}" x2="{_f(xt)}" y1="{_f(y)}" y2="{_f(y)}" stroke="{st["color"]}" stroke-opacity="0.45" '
                     f'stroke-width="1"/>')
        lines.append(f'<line x1="{_f(xt)}" x2="{_f(xt)}" y1="{_f(y - 3.5)}" y2="{_f(y + 3.5)}" stroke="{st["color"]}" '
                     f'stroke-opacity="0.45" stroke-width="1"/>')
        block(fk.Box(x, y - 2, xt, y + 2), p["arm"])
        block(fk.Box(xt - 1, y - 3.5, xt + 1, y + 3.5), p["arm"])           # the p95 tick
        if p["dref"] is not None:
            lo, hi = ends(p)
            ylo, yhi = ys(lo), ys(hi)
            lines.append(f'<line data-role="whisker" data-arm={fk.esc(p["arm"])} data-vs={fk.esc(ref["arm"])} '
                         f'data-dref="{",".join(_f(v) for v in p["dref"])}" data-lo="{_f(lo)}" data-hi="{_f(hi)}" '
                         f'x1="{_f(x)}" x2="{_f(x)}" y1="{_f(ylo)}" y2="{_f(yhi)}" stroke="{st["color"]}" '
                         f'stroke-opacity="0.62" stroke-width="1.2"/>')
            for yy in (ylo, yhi):
                lines.append(f'<line x1="{_f(x - 3)}" x2="{_f(x + 3)}" y1="{_f(yy)}" y2="{_f(yy)}" stroke="{st["color"]}" '
                             f'stroke-opacity="0.62" stroke-width="1.2"/>')
            block(fk.Box(x - 3, yhi, x + 3, ylo), p["arm"])

    # ---- the callo lines: no think -> think
    own_line: dict[str, tuple[int, int]] = {}             # each line's obstacle boxes (its label may hug them)
    seg_of: dict[str, tuple[float, float, float, float]] = {}
    accent_line: list[str] = []
    for s in sorted(present, key=lambda s: bool(s.get("accent"))):     # the accent line last, on top
        if s["kind"] != "callo":
            continue
        st = sty[s["name"]]
        a, b = (by_arm[a_] for a_ in s["arms"])
        n0 = len(placer.lines)
        segment(X(a), Y(a), X(b), Y(b), pad=1.5, who=s["name"])
        own_line[s["name"]] = (n0, len(placer.lines))
        seg_of[s["name"]] = (X(a), Y(a), X(b), Y(b))
        dash = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ""
        (accent_line if s.get("accent") else lines).append(
            f'<line data-role="callo-line" data-series={fk.esc(s["name"])} '
            f'data-arms={fk.esc(",".join(s["arms"]))} x1="{_f(X(a))}" y1="{_f(Y(a))}" x2="{_f(X(b))}" '
            f'y2="{_f(Y(b))}" stroke="{st["color"]}" stroke-width="{st["w"]}" stroke-linecap="round"{dash}/>')

    # ---- points: every ground halo first (lines visibly break at a marker), then every marker, so a marker never
    # cuts a neighbour's outline; the reference drawn last (on top of its coincident Q4). The other callo lines' marks
    # sit under the accent line, so a point of theirs that lies against it (Flash-Next's no think by Tiel-Coder's think
    # point) never cuts it: the accent line runs whole to its own think point
    accent = {s["name"] for s in present if s.get("accent")}
    order = sorted(pts, key=lambda p: (p["arm"] == ref["arm"], p["kind"] == "callo"))
    layers: dict[bool, tuple[list[str], list[str]]] = {False: ([], []), True: ([], [])}
    for p in order:
        st = sty[p["series"]]
        x, y = X(p), Y(p)
        halo, shape, r = _mark(fk, t, st, x, y, big=st["color"] == t["accent"])
        under = bool(accent) and p["kind"] == "callo" and p["series"] not in accent
        layers[under][0].append(halo)
        placer.marks.append(fk.Box(x - r - 2, y - r - 2, x + r + 2, y + r + 2))
        dref = ",".join(_f(v) for v in p["dref"]) if p["dref"] is not None else ""
        layers[under][1].append(
            f'<g data-role="point" data-arm={fk.esc(p["arm"])} data-series={fk.esc(p["series"])} '
            f'data-early-exit="{ee_text(p.get("early_exit"))}" '
            f'data-budget="{"" if p["budget"] is None else p["budget"]}" data-index="{_f(p["index"])}" '
            f'data-lat-mean="{_f(p["lat"]["mean"])}" data-lat-p95="{_f(p["lat"]["p95"])}" data-dref="{dref}" '
            f'data-x="{_f(x)}" data-y="{_f(y)}">{shape}</g>')
    for under in (True, False):
        hs, shapes = layers[under]
        if hs:
            marks.append('<g data-role="halos">' + "".join(hs) + "</g>")
            marks += shapes
        if under:
            marks += accent_line

    # ---- labels: the reference line's, then each series' (name + runtime) and each callo point's budget and index
    name_st = fk.Style(size=14.5 * ts, weight=600)
    sub_st = fk.Style(size=13 * ts, fill=t["sub"])
    pst = fk.Style(size=13 * ts, fill=t["sub"])

    def leader(box, ax_: float, ay_: float, color: str, *, min_len: float = 16, who: str = "") -> None:
        cx = min(max(ax_, box.x0), box.x1)
        cy = min(max(ay_, box.y0), box.y1)
        dist = math.hypot(cx - ax_, cy - ay_)
        if dist > min_len:
            k_ = 9 / dist
            sx, sy = ax_ + (cx - ax_) * k_, ay_ + (cy - ay_) * k_
            labels.append(f'<line data-role="leader" data-for={fk.esc(who)} x1="{_f(sx)}" y1="{_f(sy)}" x2="{_f(cx)}" '
                          f'y2="{_f(cy)}" stroke="{color}" stroke-opacity="0.45" stroke-width="0.8"/>')
            segment(sx, sy, cx, cy, pad=1.5, who=who)

    free_foot: dict[str, tuple[float, float]] = {}

    def place_along(s: dict) -> tuple[fk.Box, str]:
        """A callo line's label hugs the line itself, horizontal: just above it over its middle stretch (`label_side`
        "above", the accent line's default) or just under it towards its think point ("below"), with a margin of
        clear ground to every whisker, tail, marker, other line and placed label. The nearest clear spot to the
        preferred stretch wins; with none on the preferred side the other side is tried, then the generic placer.
        Returns the box and the side it took ("above", "below" or "free")."""
        a_, b_ = sorted((by_arm[x] for x in s["arms"]), key=X)
        ax_, ay_, bx_, by_ = X(a_), Y(a_), X(b_), Y(b_)
        n1 = name_st.with_(fill=sty[s["name"]]["name"])
        w_ = max(fk.advance(s["label"], n1), fk.advance(s.get("runtime") or "Callo", sub_st))
        h_, gap, margin = 38.0 * ts, 6.0, 9.0
        first = s.get("label_side") or ("above" if s.get("accent") else "below")
        i0, i1 = own_line[s["name"]]
        others = placer.lines[:i0] + placer.lines[i1:]

        def on_line(x: float) -> float:
            u = 0.0 if bx_ == ax_ else min(max((x - ax_) / (bx_ - ax_), 0.0), 1.0)
            return ay_ + (by_ - ay_) * u

        def clear_spot(side: str, margin: float) -> fk.Box | None:
            want = 0.5 if side == "above" else 0.8
            best, best_c = None, math.inf
            for k in range(61):
                u = k / 60
                cx = ax_ + (bx_ - ax_) * u
                for shift in (0.5, 0.35, 0.65):
                    bx0 = cx - shift * w_
                    ya, yb = on_line(bx0), on_line(bx0 + w_)
                    top = min(ya, yb) - gap - h_ if side == "above" else max(ya, yb) + gap
                    box = fk.Box(bx0, top, bx0 + w_, top + h_)
                    if placer.cost(box) > 0:          # outside the plot, or on something (its own line included)
                        continue
                    wide_ = box.pad(margin)           # clear ground all round, but for its own line
                    if (any(wide_.overlap(o) for o in placer.labels) or any(wide_.overlap(o) for o in placer.marks)
                            or any(wide_.overlap(o) for o in others)):
                        continue
                    c = 160 * abs(u - want) + 40 * abs(shift - 0.5)
                    if c < best_c:
                        best, best_c = box, c
            return best

        if first == "end":
            # a line-end label: just past the think point, its box a little above the point's height (clear of the
            # point's own tail), the nearest clear spot to the right winning; a leader only when it had to move off
            best, best_c = None, math.inf
            for dx in range(8, 240, 4):
                for dy in range(-70, 24, 2):
                    box = fk.Box(bx_ + dx, by_ + dy - h_, bx_ + dx + w_, by_ + dy)
                    if placer.cost(box) > 0:
                        continue
                    wide_ = box.pad(4)
                    if (any(wide_.overlap(o) for o in placer.labels) or any(wide_.overlap(o) for o in placer.marks)
                            or any(wide_.overlap(o) for o in others)):
                        continue
                    c = dx + 1.2 * abs(dy + 4)
                    if c < best_c:
                        best, best_c = box, c
            if best is not None:
                placer.labels.append(best)
                free_foot[s["name"]] = (bx_, by_)
                return best, "end"
            first = "above"
        for m in (margin, 4.0):                   # generous clear ground first, then a tighter margin
            for side in (first, "below" if first == "above" else "above"):
                box = clear_spot(side, m)
                if box is not None:
                    placer.labels.append(box)
                    return box, side

        # no room along the line: the clear spot (4 px all round) nearest the line itself, else the generic placer
        def to_line(b: fk.Box) -> tuple[float, float, float]:
            """The distance from the box to the line, and the line point nearest it (where its leader starts)."""
            best_ = (math.inf, ax_, ay_)
            for k in range(41):
                px, py = ax_ + (bx_ - ax_) * k / 40, ay_ + (by_ - ay_) * k / 40
                d = math.hypot(max(b.x0 - px, 0, px - b.x1), max(b.y0 - py, 0, py - b.y1))
                if d < best_[0] - 1e-9 or (abs(d - best_[0]) < 1e-9 and abs(k - 20) < abs(best_[1] - 20)):
                    best_ = (d, px, py)
            return best_

        mx, my = (ax_ + bx_) / 2, (ay_ + by_) / 2
        best, best_c, foot = None, math.inf, (mx, my)
        for i in range(-60, 61):
            for j in range(-40, 41):
                box = fk.Box(mx + 8 * i - w_ / 2, my + 6 * j - h_ / 2, mx + 8 * i + w_ / 2, my + 6 * j + h_ / 2)
                if placer.cost(box.pad(4)) > 0:
                    continue
                d, fx_, fy_ = to_line(box)
                c = d + 0.05 * abs(8 * i)
                if c < best_c:
                    best, best_c, foot = box, c, (fx_, fy_)
        if best is not None:
            placer.labels.append(best)
            free_foot[s["name"]] = foot
            return best, "free"
        return placer.place(w_, h_, bx_, by_, gaps=(10, 16, 26, 40, 60), scan=240), "free"

    def place_series(s: dict) -> None:
        st = sty[s["name"]]
        own = [by_arm[a] for a in s["arms"]]
        anchor = max(own, key=lambda p: p["index"])
        attrs = ""
        if s["kind"] == "callo":
            line1, line2 = s["label"], s.get("runtime") or "Callo"
            n1 = name_st.with_(fill=st["name"])
            box, side = place_along(s)
            if side in ("free", "end"):          # off its line: a leader from the line says whose it is
                a_, b_ = (by_arm[x] for x in s["arms"])
                leader(box, *free_foot.get(s["name"], ((X(a_) + X(b_)) / 2, (Y(a_) + Y(b_)) / 2)), st["color"],
                       min_len=40 if side == "free" else 28, who=s["name"])   # a label this near needs none
        else:
            line1, line2 = f"{s['label']} · {f1(anchor['index'])}", s.get("runtime") or ""
            n1 = name_st.with_(fill=t["ink"])
            w_ = max(fk.advance(line1, n1), fk.advance(line2, sub_st))
            box = placer.place(w_, 38 * ts, X(anchor), Y(anchor), gaps=(10, 16, 26, 40, 60, 90),
                               prefer=("nw", "ne", "n", "w", "e", "sw", "se", "s"), scan=240)
            leader(box, X(anchor), Y(anchor), st["color"], who=s["name"])
            attrs = " " + _key(_ptr(anchor["arm"], "index"), anchor["index"])
            side = "free"
        labels.append(f'<g data-role="series-label" data-series={fk.esc(s["name"])} data-side="{side}" '
                      f'data-text={fk.esc(line1 + " / " + line2)} data-box="{_box_attr(box)}"{attrs}>'
                      f'{fk.text_paths(line1, box.x0, box.y0 + 14 * ts, n1)}'
                      f'{fk.text_paths(line2, box.x0, box.y0 + 32 * ts, sub_st)}'
                      f'</g>')

    def place_group(group: list[dict]) -> None:
        """Single markers that sit on one another (Clef-Flash BF16 and its Q4) share one label: the family name,
        then a row per marker (its own mark drawn inline, the rest of its label and its index), then the runtime."""
        gp = sorted((by_arm[s["arms"][0]] for s in group), key=lambda p: -p["index"])
        ser = {s["name"]: s for s in group}
        heads = {ser[p["series"]]["label"].split(" · ", 1)[0] for p in gp}
        family = heads.pop() if len(heads) == 1 else None
        rows = [(p, (ser[p["series"]]["label"].split(" · ", 1)[1] if family and " · " in ser[p["series"]]["label"]
                     else ser[p["series"]]["label"]) + f" · {f1(p['index'])}") for p in gp]
        runtimes = {ser[p["series"]].get("runtime") for p in gp}
        runtime = runtimes.pop() if len(runtimes) == 1 else None
        n1 = name_st.with_(fill=t["ink"])
        row_st = fk.Style(size=13.5 * ts, weight=500, fill=t["ink"])
        w_ = max([fk.advance(family or "", n1)] + [16 * ts + fk.advance(r, row_st) for _, r in rows]
                 + [fk.advance(runtime or "", sub_st)])
        h_ = 18 * ts * (len(rows) + (1 if family else 0) + (1 if runtime else 0)) + 2
        cx = sum(X(p) for p in gp) / len(gp)
        cy = sum(Y(p) for p in gp) / len(gp)
        box = placer.place(w_, h_, cx, cy, gaps=(12, 18, 28, 40, 60, 90), prefer=("nw", "w", "sw", "n", "ne", "e"),
                           scan=240)
        leader(box, cx, cy, t["ref"], who=",".join(p["series"] for p in gp))
        out = [f'<g data-role="series-label" data-series={fk.esc(",".join(p["series"] for p in gp))} '
               f'data-text={fk.esc(" / ".join([family or ""] + [r for _, r in rows] + [runtime or ""]))} '
               f'data-box="{_box_attr(box)}">']
        yb = box.y0 + 14 * ts
        if family:
            out.append(fk.text_paths(family, box.x0, yb, n1))
            yb += 18 * ts
        for p, r in rows:
            st = sty[p["series"]]
            mk, _ = fk.marker(st["marker"], box.x0 + 5, yb - 4.5 * ts, st["fill"], st["color"], 1.4, scale=0.72)
            out.append(f'<g {_key(_ptr(p["arm"], "index"), p["index"])} data-arm={fk.esc(p["arm"])}>{mk}'
                       f'{fk.text_paths(r, box.x0 + 16 * ts, yb, row_st)}</g>')
            yb += 18 * ts
        if runtime:
            out.append(fk.text_paths(runtime, box.x0, yb, sub_st))
        out.append("</g>")
        labels.append("".join(out))

    # the accent series first, then the other callo lines, then Clef from the reference outward; single markers that
    # sit within a marker's width of each other are labelled together
    singles = [s for s in present if s["kind"] != "callo"]
    groups: list[list[dict]] = []
    for s in sorted(singles, key=lambda s: s["arms"][0] != ref["arm"]):
        p = by_arm[s["arms"][0]]
        home = next((g for g in groups if any(math.hypot(X(p) - X(by_arm[o["arms"][0]]),
                                                         Y(p) - Y(by_arm[o["arms"][0]])) < 18 for o in g)), None)
        if home is None:
            groups.append([s])
        else:
            home.append(s)

    marker_at = [(X(q), Y(q), q["arm"]) for q in pts]
    point_boxes: list = []
    whisker_of = {q["arm"]: (ys(ends(q)[1]), ys(ends(q)[0])) for q in pts if q["dref"] is not None}  # (top, bottom)
    DIRS8 = ("e", "ne", "n", "nw", "w", "sw", "s", "se")

    def _near(box, x: float, y: float) -> tuple[float, float, float]:
        """The box's point nearest (x, y), and the distance."""
        cx, cy = min(max(x, box.x0), box.x1), min(max(y, box.y0), box.y1)
        return cx, cy, math.hypot(cx - x, cy - y)

    def _cross(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        """Whether two segments cross (or come within 1.5 px of each other's ends: a touch reads as a cross)."""
        def side(px, py, qx, qy, rx, ry):
            return (qx - px) * (ry - py) - (qy - py) * (rx - px)

        def gap(px, py, s_):
            x1_, y1_, x2_, y2_ = s_
            dx, dy = x2_ - x1_, y2_ - y1_
            u = max(0.0, min(1.0, ((px - x1_) * dx + (py - y1_) * dy) / ((dx * dx + dy * dy) or 1)))
            return math.hypot(px - x1_ - u * dx, py - y1_ - u * dy)
        d1, d2 = side(*b[:2], *b[2:], *a[:2]), side(*b[:2], *b[2:], *a[2:])
        d3, d4 = side(*a[:2], *a[2:], *b[:2]), side(*a[:2], *a[2:], *b[2:])
        return (d1 * d2 < 0 and d3 * d4 < 0) or min(gap(*a[:2], b), gap(*a[2:], b), gap(*b[:2], a),
                                                    gap(*b[2:], a)) < 1.5

    def through(ax_: float, ay_: float, bx_: float, by_: float, *, skip: tuple[str, ...] = ()) -> bool:
        """Whether the segment passes through a line, tail, whisker or leader (any but `skip`'s) or a marker."""
        if any(w not in skip and _cross((ax_, ay_, bx_, by_), sg[:4]) for sg in segs for w in (sg[4],)):
            return True
        n = max(2, int(math.hypot(bx_ - ax_, by_ - ay_) / 2))
        marks_ = [o for o in placer.marks if not (o.x0 <= ax_ <= o.x1 and o.y0 <= ay_ <= o.y1)]   # not its own
        for i in range(n + 1):
            x, y = ax_ + (bx_ - ax_) * i / n, ay_ + (by_ - ay_) * i / n
            if any(o.x0 <= x <= o.x1 and o.y0 <= y <= o.y1 and w not in skip for o, w in zip(placer.lines, owner,
                                                                                             strict=True)):
                return True
            if any(o.x0 <= x <= o.x1 and o.y0 <= y <= o.y1 for o in marks_):
                return True
        return False

    def apart(box) -> bool:
        """Two labels whose baselines sit within a few px of each other keep a wide gap, so they never read as one
        string."""
        for o in point_boxes:
            if abs(o.y0 - box.y0) < 9 and max(o.x0 - box.x1, box.x0 - o.x1) < 30:
                return False
        return True

    def point_box(p: dict, w_: float, h_: float, *, strict_only: bool = False, dry: bool = False):
        """The point's label box: the nearest clear spot (3 px of ground all round, no shared baseline with another
        budget label) that reads as this point's own. A label straight above or below the point, past its whisker,
        is joined to it by the whisker and needs no leader; one next to the point must be nearer its marker than any
        other, with no other series' line, tail or whisker between them; one further off gets a leader, which must
        cross nothing and end at a box no other marker sits near. The series' `point_labels` (above or below) orders
        the directions, and `point_label_at` ({arm: compass point}) puts one point's label on that side first. The
        generic placer when nothing qualifies."""
        px_, py_ = X(p), Y(p)
        ser = _series(snap, p["series"])
        dirs = (["ne", "n", "nw", "e", "w", "se", "s", "sw"] if ser.get("point_labels") == "above" else
                ["se", "e", "ne", "s", "w", "sw", "n", "nw"])
        at = (ser.get("point_label_at") or {}).get(p["arm"])
        pen = {d: 4 * i + (0 if at in (None, d) else 60) for i, d in enumerate(dirs)}
        wt = whisker_of.get(p["arm"])
        cands = []
        for i in range(-int((w_ + 72) / 3), 25):
            for j in range(-int((h_ + 72) / 3), 23):
                box = fk.Box(px_ + 3 * i, py_ + 3 * j, px_ + 3 * i + w_, py_ + 3 * j + h_)
                cx, cy, d = _near(box, px_, py_)
                if d < 6 or d > 72:
                    continue
                straddle = (wt is not None and abs((box.x0 + box.x1) / 2 - px_) <= w_ / 4
                            and (box.y1 <= wt[0] or box.y0 >= wt[1]))       # centred past its whisker's end
                eff = (d + (wt[0] - box.y1 if box.y1 <= wt[0] else box.y0 - wt[1])) / 2 if straddle else d
                sector = DIRS8[int(((math.degrees(math.atan2(py_ - cy, cx - px_)) + 22.5) % 360) // 45)]
                cands.append((2.5 * eff + pen[sector], box, cx, cy, d, straddle))
        cands.sort(key=lambda c: c[0])

        def reads_as_own(box, cx: float, cy: float, d: float, straddle: bool) -> bool:
            if straddle:                                  # nothing else between its whisker's end and the box
                end, edge = (wt[0] - 1.5, box.y1) if box.y1 <= wt[0] else (wt[1] + 1.5, box.y0)
                return not through(px_, end, px_, edge)
            others = [_near(box, x, y)[2] for x, y, a in marker_at if a != p["arm"]]
            if through(px_, py_, cx, cy, skip=(p["arm"], p["series"])):     # another series' line between them
                return False
            if d <= 16:                                   # beside it, no leader
                return all(d + 3 < o for o in others)
            k_ = 9 / d                                    # a leader from 9 px out, through nothing at all
            return all(o >= 18 for o in others) and not through(px_ + (cx - px_) * k_, py_ + (cy - py_) * k_,
                                                                  cx, cy)

        for strict in ((True,) if strict_only else (True, False)):
            for _, box, cx, cy, d, straddle in cands:
                if box.overlap(fk.Box(px_ - 7, py_ - 7, px_ + 7, py_ + 7)) or placer.cost(box.pad(3)) > 0:
                    continue
                if strict and not (apart(box) and reads_as_own(box, cx, cy, d, straddle)):
                    continue
                if not dry:
                    placer.labels.append(box)
                return box, straddle, _
        if strict_only:
            return None
        return placer.place(w_, h_, px_, py_, gaps=(7, 12, 20, 32, 48), prefer=dirs, scan=110), False, math.inf

    def place_point(p: dict) -> None:
        """A callo point's budget, its early exit when it has one, and its index, on one line: "think 1024 · 41.0",
        "think 256 · early exit · 49.0". An early-exit label with no clean spot for its one line sets in two
        ("think 256 · 51.7" over "early exit + gate"), narrower, where the one line would not fit."""
        how = ee_text(p.get("early_exit"))
        txt = budget_text(p["budget"]) + (f" · {how}" if how else "") + f" · {f1(p['index'])}"
        rows = [txt]
        found = point_box(p, fk.advance(txt, pst), 16 * ts, strict_only=bool(how), dry=bool(how))
        if how:                                 # the two-line form when the one line has no clean spot near
            two = [f"{budget_text(p['budget'])} · {f1(p['index'])}", how]
            w2 = max(fk.advance(r, pst) for r in two)
            alt = point_box(p, w2, 31 * ts, strict_only=True, dry=True)
            if alt is not None and (found is None or alt[2] + 20 < found[2]):
                rows, found = two, point_box(p, w2, 31 * ts, strict_only=True)
            elif found is not None:
                found = point_box(p, fk.advance(txt, pst), 16 * ts, strict_only=True)
            else:
                found = point_box(p, fk.advance(txt, pst), 16 * ts)
        box, straddle, _ = found
        point_boxes.append(box)
        if not straddle:
            leader(box, X(p), Y(p), sty[p["series"]]["color"], who=p["arm"])
        labels.append(f'<g data-role="point-label" data-arm={fk.esc(p["arm"])} data-text={fk.esc(" / ".join(rows))} '
                      f'{_key(_ptr(p["arm"], "index"), p["index"])} data-box="{_box_attr(box)}">'
                      + "".join(fk.text_paths(r, box.x0, box.y0 + (12 + 15 * i) * ts, pst)
                                for i, r in enumerate(rows))
                      + '</g>')

    # order: the budget labels of a line that asks where they sit (`point_labels`, or `point_label_at` for one point)
    # first, so a neighbour's series label makes room for them, but not its think point when the line is labelled at
    # that end (unless that point names its side); then the series
    # labels (the accent first), Clef's, and the remaining budget labels: that think point first (its side is the
    # one it asked for), then the accent's, by latency
    callo_pts = sorted((p for p in pts if p["kind"] == "callo"),
                       key=lambda p: (not _series(snap, p["series"]).get("point_labels"),
                                      not sty[p["series"]]["color"] == t["accent"], p["lat"]["mean"]))

    def early(p: dict) -> bool:
        s = _series(snap, p["series"])
        return ((bool(s.get("point_labels")) and not (s.get("label_side") == "end" and p["budget"]))
                or p["arm"] in (s.get("point_label_at") or {}))

    for p in (p for p in callo_pts if early(p)):
        place_point(p)
    for s in sorted((s for s in present if s["kind"] == "callo"), key=lambda s: not s.get("accent")):
        place_series(s)
    for g in groups:
        if len(g) == 1:
            place_series(g[0])
        else:
            place_group(g)
    for p in (p for p in callo_pts if not early(p)):
        place_point(p)

    body.extend(lines)
    body.extend(marks)
    body.extend(labels)
    return body


def draw(snap: dict, *, theme: str = "dark") -> str:
    """The plate: header; the index against the mean request completion time (log) with p95 tails and paired-CI
    whiskers vs the reference; the five area skills as small multiples; the table (every measured arm on the plate,
    with the second reference's Δ); the fine print."""
    import figkit as fk

    refuse_unexplained(snap)
    t = fk.THEMES[theme]
    sty = styles(snap, t)
    ref = snap["reference"]
    body: list[str] = []

    # ---- header
    if snap.get("preliminary"):
        built = (snap.get("generator") or {}).get("built", "")[:10]
        bst = fk.Style(size=11, weight=600, fill=t["sub"], tracking=1.4)
        btxt = f"PRELIMINARY · {built}"
        w = fk.advance(btxt, bst)
        body.append(f'<rect x="{SIDE}" y="40" width="{fk.num(w + 20)}" height="22" rx="2" fill="none" '
                    f'stroke="{t["sub"]}" stroke-opacity="0.6"/>')
        body.append(fk.text(btxt, SIDE + 10, 55.5, bst, role="badge")[0])
    tsans = fk.Style(size=34, weight=600, fill=t["ink"], width=96)
    tserif = fk.Style(size=38, weight=500, fill=t["ink"], face="serif")
    head = fk.paragraph(_title_runs(snap["title"], tsans, tserif), SIDE, 108, W - 2 * SIDE, leading=42, role="title")
    body.append(head.svg)
    y_head = head.box.y1
    st_ = snap.get("story")
    if st_:
        sub = fk.paragraph([(st_["subtitle"], fk.Style(size=16, fill=t["sub"]))], SIDE, y_head + 30, W - 2 * SIDE,
                           leading=23, role="subtitle")
        body.append(sub.svg)
        y_head = sub.box.y1
    x0, x1 = PLOT_X
    y0 = math.ceil(y_head + 70)
    y1 = y0 + PLOT_H
    body += _plot(fk, t, snap, sty, fk.Box(x0, y0, x1, y1), y_title=Y_TITLE)

    # ---- the area skills: small multiples, one panel per area, points in the table's order
    ay = y1 + 104
    body.append(fk.text("BY AREA", SIDE, ay, fk.Style(size=12, weight=600, fill=t["sub"], tracking=1.0),
                        role="section")[0])
    body.append(fk.text("skill per area, chance-corrected, 0–100, in the table's order; whiskers as above",
                        SIDE + 78, ay, fk.Style(size=13, fill=t["sub"]), role="section-note")[0])
    panels_svg, py1 = _panels(fk, t, snap, sty, ay + 30)
    body.append(panels_svg)

    # ---- the table
    tb_svg, ty1 = _table(fk, t, snap, sty, py1 + 52)
    body.append(tb_svg)

    # ---- fine print: one column, one entry per paragraph, 14 px (legible at README width)
    rule_y = ty1 + 30
    body.append(f'<line x1="{SIDE}" x2="{W - SIDE}" y1="{rule_y}" y2="{rule_y}" stroke="{t["grid"]}"/>')
    lead_st = fk.Style(size=14, weight=600, fill=t["ink"])
    fp_st = fk.Style(size=14, fill=t["sub"])
    yy = rule_y + 32
    for f in snap["fine_print"]:
        runs = [(f["lead"] + " ", lead_st), (f["text"], fp_st)] if f["lead"] else [(f["text"], fp_st)]
        pr = fk.paragraph(runs, SIDE, yy, W - 2 * SIDE, leading=20, role="fine-print")
        body.append(pr.svg)
        yy += pr.lines * 20 + 7
    H = math.ceil(yy + 18)

    bg = fk.ground(W, H, t, glow_at=((x0 + x1) / 2, y1 + 10, (x1 - x0) * 0.62, 300))
    title = snap["title"].replace("*", "")
    return fk.svg_doc(W, H, bg + "".join(body), title=title, theme=theme,
                      extra_attrs={"data-plot-box": ",".join(_f(v) for v in (x0, y0, x1, y1)),
                                   "data-treatment": snap["wide"]["switch"],
                                   "data-treatment-label": snap["wide"]["label"],
                                   "data-reference": ref["arm"]})


def draw_chart(snap: dict, *, theme: str = "dark") -> str:
    """The README's chart: the plate's plot alone under one title (`Decision Index`), with its key; no subtitle, no
    area panels, no table, no fine print. The same snapshot, scales and labels as the plate."""
    import figkit as fk

    refuse_unexplained(snap)
    t = fk.THEMES[theme]
    sty = styles(snap, t)
    head = fk.text(CHART_TITLE, SIDE, 88, fk.Style(size=38, weight=600, fill=t["ink"], width=96), role="title")
    x0, x1 = PLOT_X
    y0 = math.ceil(head[1].y1 + 66)
    y1 = y0 + PLOT_H
    body = [head[0], *_plot(fk, t, snap, sty, fk.Box(x0, y0, x1, y1), y_title=CHART_Y_TITLE, ts=CHART_TYPE)]
    H = math.ceil(y1 + 52 * CHART_TYPE + 34)
    bg = fk.ground(W, H, t, glow_at=((x0 + x1) / 2, y1 + 10, (x1 - x0) * 0.62, 300))
    return fk.svg_doc(W, H, bg + "".join(body), title=CHART_TITLE, theme=theme,
                      extra_attrs={"data-plot-box": ",".join(_f(v) for v in (x0, y0, x1, y1)),
                                   "data-treatment": snap["wide"]["switch"],
                                   "data-treatment-label": snap["wide"]["label"],
                                   "data-reference": snap["reference"]["arm"]})


def _legend(fk, t: dict, right: float, y: float, ref_label: str, *, k: float = 1.0) -> str:
    """The key, right-aligned above the plot: the marks themselves (the brand faces have no ● or ┤)."""
    st = fk.Style(size=13 * k, fill=t["sub"])
    items = [("mean", "dot"), ("p95", "tail"), (f"paired 95 % CI vs {ref_label}", "whisker")]
    widths = [fk.advance(s, st) for s, _ in items]
    glyph = {"dot": 12, "tail": 22, "whisker": 10}
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
        else:
            out.append(f'<line x1="{_f(x + 5)}" x2="{_f(x + 5)}" y1="{_f(cy - 7)}" y2="{_f(cy + 7)}" stroke="{c}" '
                       f'stroke-width="1.2"/><line x1="{_f(x + 2)}" x2="{_f(x + 8)}" y1="{_f(cy - 7)}" y2="{_f(cy - 7)}" '
                       f'stroke="{c}" stroke-width="1.2"/><line x1="{_f(x + 2)}" x2="{_f(x + 8)}" y1="{_f(cy + 7)}" '
                       f'y2="{_f(cy + 7)}" stroke="{c}" stroke-width="1.2"/>')
        x += glyph[k] + 6
        out.append(fk.text_paths(s, x, y, st))
        x += w + 18
    out.append("</g>")
    return "".join(out)


def _order(snap: dict) -> list[dict]:
    """The plate's points in the table's order: each present, plotted series in the spec's order, budgets ascending,
    a budget's arms plain first, then with early exit (a line's other arms at its think budget included)."""
    by_arm = {p["arm"]: p for p in snap["points"]}
    return [by_arm[a] for s in snap["series"] if s["status"] == "present" and s.get("on_plate", True)
            for a in (s.get("rows") or s["arms"])]


def _panels(fk, t: dict, snap: dict, sty: dict, top: float) -> tuple[str, float]:
    """Five small multiples: per area, each drawn point's skill (0-100) at its slot, a callo line's two points joined,
    whiskers = the paired CI of the area difference vs the reference. The points are the plot's own, in the table's
    order; a line's other arms at its think budget are table rows only."""
    x0, x1 = PLOT_X
    areas = snap["areas"]
    n = len(areas)
    pw = (x1 - x0 - (n - 1) * PANEL_GAP) / n
    drawn = {p["arm"] for p in _drawn(snap)}
    order = [p for p in _order(snap) if p["arm"] in drawn]
    # slots: a callo model's two budgets side by side, a gap between models
    slots, pos = {}, 0.0
    prev = None
    for p in order:
        if prev is not None:
            pos += 1.0 if p["series"] == prev else 1.7
        slots[p["arm"]] = pos
        prev = p["series"]
    span = pos or 1.0
    out = []
    title_st = fk.Style(size=14, weight=600, fill=t["ink"])
    wt_st = fk.Style(size=13, fill=t["sub"])
    tick_st = fk.Style(size=13, fill=t["sub"])
    ptop = top + 44
    pbot = ptop + PANEL_H
    ys = fk.LinScale(0, 100, pbot, ptop + 4)          # 0 sits on the panel's axis line
    ref_arm = snap["reference"]["arm"]
    for i, area in enumerate(areas):
        px0 = x0 + i * (pw + PANEL_GAP)
        px1 = px0 + pw
        aid = area["id"]
        out.append(f'<g data-role="panel" data-area="{aid}" data-box="{_f(px0)},{_f(ptop)},{_f(px1)},{_f(pbot)}">')
        out.append(fk.text(area["label"], px0, top + 14, title_st, role="panel-title")[0])
        if area.get("weight") is not None:
            out.append(fk.text(f"{area['weight']:g} % of the index", px0, top + 32, wt_st, role="panel-weight",
                               attrs={"data-key": f"/areas/{i}/weight", "data-value": _f(area["weight"])})[0])
        for v in (0, 25, 50, 75, 100):
            y = ys(v)
            if v:
                out.append(f'<line x1="{_f(px0)}" x2="{_f(px1)}" y1="{_f(y)}" y2="{_f(y)}" stroke="{t["grid"]}"/>')
            if i == 0 and v in (0, 50, 100):
                out.append(fk.text(f"{v}", x0 - 12, y + 4.5, tick_st, anchor="end", role="tick")[0])

        def sx(arm: str, px0=px0) -> float:
            return px0 + 12 + slots[arm] / span * (pw - 24)

        # whiskers, the callo joins, then the marks
        for p in order:
            st = sty[p["series"]]
            if p["area_dref"] is None:
                continue
            d, lo, hi = p["area_dref"][aid]
            v = p["areas"][aid]
            ylo, yhi = ys(skill(v + lo - d)), ys(skill(v + hi - d))
            x = sx(p["arm"])
            out.append(f'<line data-role="panel-whisker" data-area="{aid}" data-arm={fk.esc(p["arm"])} '
                       f'data-dref="{",".join(_f(u) for u in p["area_dref"][aid])}" x1="{_f(x)}" x2="{_f(x)}" '
                       f'y1="{_f(ylo)}" y2="{_f(yhi)}" stroke="{st["color"]}" stroke-opacity="0.62" stroke-width="1.1"/>')
            for yy in (ylo, yhi):
                out.append(f'<line x1="{_f(x - 2.5)}" x2="{_f(x + 2.5)}" y1="{_f(yy)}" y2="{_f(yy)}" '
                           f'stroke="{st["color"]}" stroke-opacity="0.62" stroke-width="1.1"/>')
        for s in snap["series"]:
            if s["status"] != "present" or s["kind"] != "callo" or not s.get("on_plate", True):
                continue
            st = sty[s["name"]]
            a, b = (next(p for p in order if p["arm"] == a_) for a_ in s["arms"])
            dash = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ""
            out.append(f'<line data-role="panel-line" data-area="{aid}" data-series={fk.esc(s["name"])} '
                       f'x1="{_f(sx(a["arm"]))}" y1="{_f(ys(skill(a["areas"][aid])))}" x2="{_f(sx(b["arm"]))}" '
                       f'y2="{_f(ys(skill(b["areas"][aid])))}" stroke="{st["color"]}" stroke-width="{st["w"]}" '
                       f'stroke-linecap="round"{dash}/>')
        out.append(f'<line x1="{_f(px0)}" x2="{_f(px1)}" y1="{_f(pbot)}" y2="{_f(pbot)}" stroke="{t["axis"]}"/>')
        halos, shapes = ['<g data-role="halos">'], []
        for p in sorted(order, key=lambda p: p["arm"] == ref_arm):
            st = sty[p["series"]]
            x, y = sx(p["arm"]), ys(skill(p["areas"][aid]))
            halo, shape, _ = _mark(fk, t, st, x, y, scale=0.78, sw=1.5, halo_w=3.5)
            halos.append(halo)
            shapes.append(f'<g data-role="panel-point" data-area="{aid}" data-arm={fk.esc(p["arm"])} '
                          f'{_key(_ptr(p["arm"], "areas/" + aid), p["areas"][aid])} data-x="{_f(x)}" '
                          f'data-y="{_f(y)}">{shape}</g>')
        out.append("".join(halos) + "</g>" + "".join(shapes))
        out.append("</g>")
    return "".join(out), pbot


TABLE_COLS = (("INDEX", 452), ("95 % CI", 568), ("Δ VS {ref}", 800), ("Δ VS {second}", 1000), ("MEAN", 1108),
              ("P95", W - SIDE))


def _table(fk, t: dict, snap: dict, sty: dict, top: float) -> tuple[str, float]:
    """One row per arm on the plate: the index with its CI, the paired Δ vs the reference and vs the second reference
    (the only place the second one appears), and the mean and p95 request time. Every number carries its snapshot key.
    A drawn point's row starts with its mark; a row with no mark (a line's other arm at its think budget) is not
    drawn, and its name is set in the secondary colour."""
    head_st = fk.Style(size=12, weight=600, fill=t["sub"], tracking=1.0)
    lab_st = fk.Style(size=14, weight=500, fill=t["ink"])
    num_st = fk.Style(size=14, weight=600, fill=t["ink"])
    ci_st = fk.Style(size=14, fill=t["sub"])
    ref, second = snap["reference"], snap["second_reference"]
    out = ['<g data-role="table">']
    out.append(fk.text("ARM", SIDE + 22, top, head_st, role="table-head")[0])
    heads = [(h.format(ref=ref["label"].upper(), second=second["label"].upper()), x) for h, x in TABLE_COLS]
    for h, x in heads:
        out.append(fk.text(h, x, top, head_st, anchor="end", role="table-head")[0])
    y = top + 12
    out.append(f'<line x1="{SIDE}" x2="{W - SIDE}" y1="{_f(y)}" y2="{_f(y)}" stroke="{t["axis"]}"/>')
    (_, x_idx), (_, x_ci), (_, x_dr), (_, x_ds), (_, x_mean), (_, x_p95) = heads

    def cell(s: str, x: float, yb: float, st, key: str | None, val: float | None, *, anchor: str = "end") -> str:
        attrs = {"data-key": key, "data-value": _f(val)} if key else None
        return fk.text(s, x, yb, st, anchor=anchor, role="table-cell", attrs=attrs)[0]

    def delta(d: list[float] | None, x: float, yb: float, ptr: str, self_text: str) -> str:
        if d is None:
            return cell(self_text, x, yb, ci_st, None, None)
        ci = ci1(d[1], d[2], signed=True)
        w_ci = fk.advance(" " + ci, ci_st)
        return (cell(sf1(d[0]), x - w_ci, yb, num_st, ptr + "/0", d[0])
                + fk.text(" " + ci, x - w_ci, yb, ci_st, role="table-ci",
                          attrs={"data-key": ptr, "data-lo": _f(d[1]), "data-hi": _f(d[2])})[0])

    drawn = {p["arm"] for p in _drawn(snap)}
    for p in _order(snap):
        st = sty[p["series"]]
        yb = y + ROW_H - 9
        cy = yb - 5
        a = p["arm"]
        on = a in drawn
        halo, shape, _ = _mark(fk, t, st, SIDE + 7, cy, scale=0.85, sw=1.5, halo_w=2.5) if on else ("", "", 0.0)
        out.append(f'<g data-role="table-row" data-arm={fk.esc(a)} data-drawn="{str(on).lower()}">{halo}{shape}')
        out.append(fk.text(p["label"], SIDE + 22, yb, lab_st if on else lab_st.with_(fill=t["sub"]),
                           role="table-label")[0])
        out.append(cell(f1(p["index"]), x_idx, yb, num_st, f"/points/{a}/index", p["index"]))
        ci = p["index_ci"]
        out.append(fk.text(ci1(ci[0], ci[1]), x_ci, yb, ci_st, anchor="end", role="table-ci",
                           attrs={"data-key": f"/points/{a}/index_ci", "data-lo": _f(ci[0]), "data-hi": _f(ci[1])})[0])
        out.append(delta(p["dref"], x_dr, yb, f"/points/{a}/dref", "reference"))
        out.append(delta(p["dsecond"], x_ds, yb, f"/points/{a}/dsecond", "reference"))
        out.append(cell(secs(p["lat"]["mean"]), x_mean, yb, num_st.with_(weight=500), f"/points/{a}/lat/mean",
                        p["lat"]["mean"]))
        out.append(cell(secs(p["lat"]["p95"]), x_p95, yb, ci_st, f"/points/{a}/lat/p95", p["lat"]["p95"]))
        out.append("</g>")
        y += ROW_H
        out.append(f'<line x1="{SIDE}" x2="{W - SIDE}" y1="{_f(y)}" y2="{_f(y)}" stroke="{t["grid"]}"/>')
    out.append("</g>")
    return "".join(out), y


# ---- CLI ---------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="di.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="spec -> snapshot (the bench's DI outputs, checked)")
    b.add_argument("--spec", required=True)
    b.add_argument("--root")
    b.add_argument("--out", default=str(HERE / "data" / "di.json"))
    d = sub.add_parser("draw", help="snapshot -> the plate and the README's chart, SVG (both themes), optionally PNG")
    d.add_argument("--data", default=str(HERE / "data" / "di.json"))
    d.add_argument("--out-dir", default=str(HERE))
    d.add_argument("--png", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            snap = build(load_spec(args.spec, root=args.root))
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(snap, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {args.out}: {len(snap['points'])} points, {len(snap['pending'])} pending series")
            return 0
        snap = json.loads(Path(args.data).read_text(encoding="utf-8"))
        for stem, fn in (("di", draw), ("di-chart", draw_chart)):
            for theme in ("dark", "light"):
                out = Path(args.out_dir) / f"{stem}-{theme}.svg"
                out.write_text(fn(snap, theme=theme), encoding="utf-8")
                print(f"wrote {out}")
                if args.png:
                    import figkit
                    png = out.with_name(out.stem + "@2x.png")
                    figkit.render_png(str(out), str(png), 2.0)
                    print(f"wrote {png}")
        return 0
    except Refusal as e:
        print(f"di: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
