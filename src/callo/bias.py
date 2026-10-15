"""The position-bias profile, question side: which questions a prior can be
measured on and applied to, and the rotations `callo calibrate --bias` reads.

A question is CANONICAL when its labels, as `systemone._question` assigns them, are exactly A, B, C, ... in rendered
order: letter and position then coincide, and one prior indexed by position covers both. Rotation s of a
canonical question is a `choice` with keys a, b, c, ... whose key k describes option (k + s) mod n. A single-letter
key that keeps its letter renders only its description (`systemone._question`), and `render(str)` is the identity, so
position k renders exactly the original text of option (k + s) mod n, `noul` included.

The estimator side is the probability-space prior (it replaced PriDe, whose log-space prior
measured where the unchosen options' leftover mass goes, not decisions): a question's row r_q[k] is its mean served
probability at position k over its n rotations; prior_n = the mean of the rows over every question read under all n
rotations, floored ones included, floored at PRIOR_FLOOR and renormalised; offset_n = log prior_n, centred, derived
and never stored.
"""
from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import math
import os
import random
import re
import string
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np

from .calibrate import ece, nll
from .errors import CalloError, InvalidInput
from .systemone import MAX_LETTERS, InvalidRequest, Question, parse_request
from .upstream import default_cache_dir

SKIP_REASONS = ("score", "noncanonical", "single_option", "bare_letter", "cross_reference", "variant",
                "too_many_options")
MIN_OPTIONS, MAX_OPTIONS = 2, MAX_LETTERS
# The record keys a rotation's request does not carry: everything else (state, model, images, context, and any key a
# later version adds, such as audio/videos) is forwarded verbatim.
NOT_FORWARDED = frozenset({"questions", "label", "src", "_meta"})
# cross-reference patterns, verbatim. "above"/"below" on their own are content ("above the boiling point").
_XREF = (re.compile(r"(?i)\b(all|none|both|neither) of (the )?(above|below|these|those)\b"),
         re.compile(r"(?i)\b(options?|choices?|answers?) [A-Z]\b"),
         re.compile(r"\b[A-Z] (and|or|&) [A-Z]\b"))
_XREF_SHORT = re.compile(r"(?i)\b(both|neither)\b.*\b[A-Z]\b")   # "Both A and C": only on texts of <= 12 characters
_XREF_SHORT_MAX = 12


class SetError(InvalidInput):
    """A calibration-set line that is not a usable record: the CLI prints it (with its line number) and exits 1."""


def is_canonical(q: Question) -> bool:
    return q.labels == tuple(string.ascii_uppercase[: len(q.labels)])


def is_bare_letter(text: str) -> bool:
    """A single letter, or empty: rotated, an empty description renders as its letter key (`systemone._question`),
    so neither survives the rotation-0 contract. A one-character digit ("3") rotates verbatim and stays."""
    return not text or (len(text) == 1 and text.isascii() and text.isalpha())


def is_cross_reference(text: str) -> bool:
    return (any(p.search(text) for p in _XREF)
            or (len(text) <= _XREF_SHORT_MAX and _XREF_SHORT.search(text) is not None))


def exclusion(texts: Sequence[str]) -> str | None:
    """The reason the WHOLE question is excluded, whether the offending option is the gold or a distractor."""
    if any(is_bare_letter(t) for t in texts):
        return "bare_letter"
    if any(is_cross_reference(t) for t in texts):
        return "cross_reference"
    return None


def skip_reason(q: Question) -> str | None:
    """None when `q` is eligible. `too_many_options` and `variant` are decided before parsing."""
    if q.type == "score":
        return "score"
    if len(q.labels) < MIN_OPTIONS:
        return "single_option"
    if not is_canonical(q):
        return "noncanonical"
    return exclusion(q.texts)


def rotation_criteria(texts: Sequence[str], s: int) -> dict[str, str]:
    n = len(texts)
    return {string.ascii_lowercase[k]: texts[(k + s) % n] for k in range(n)}


def rotated_label(label: int, s: int, n: int) -> int:
    """Where original option `label` sits in rotation s (position k holds option (k + s) mod n)."""
    return (label - s) % n


@dataclass(frozen=True)
class Item:
    """One eligible question: what calibrate reads under every rotation."""
    uid: str                    # f"L{line}/{qid}": unique in the set, stable across runs on the same file (--resume)
    record_id: str              # _meta.id, else "line N": what error messages name
    qid: str
    cluster: str                # _meta.group_id, else _meta.pair_id, else the uid (the bootstrap clusters)
    source: str                 # _meta.source, else the question's src, else "unknown": --max-per-n strata
    kind: str                   # Request.kind: plain | context | image | audio | video | mixed, the readout path
    n: int
    texts: tuple[str, ...]      # the rendered option texts, Question.texts
    instructions: str           # rendered; render(str) is the identity, so re-sending it changes nothing
    label: int | None           # the gold option's index in the original order; None when unlabelled
    base: dict = field(compare=False)   # every top-level record key except NOT_FORWARDED, verbatim

    def body(self, s: int) -> dict:
        """The single-question System One request of rotation s; rotation 0 renders the original."""
        return {**self.base, "questions": {self.qid: {"type": "choice", "instructions": self.instructions,
                                                      "criteria": rotation_criteria(self.texts, s)}}}


@dataclass
class SetScan:
    items: list[Item] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)


def _label_index(raw: dict, q: Question) -> int | None:
    if "label" not in raw:
        return None
    label = str(raw["label"])
    if q.type == "noul":
        return {"false": 0, "true": 1}.get(label.lower())
    return q.keys.index(label) if label in q.keys else None


def scan_record(record: dict, line: int, scan: SetScan) -> None:
    """Add one labelled record's questions to `scan`, counting every skipped question by reason. Raises
    InvalidRequest (or KeyError/TypeError/AttributeError) for a record that is not a System One request."""
    questions = record["questions"]
    meta = record.get("_meta") if isinstance(record.get("_meta"), dict) else {}
    if meta.get("variant", "clean") != "clean":            # the same filter as temperature mode (cli.py)
        scan.skipped["variant"] += len(questions)
        return
    keep: dict[str, Any] = {}
    for qid, raw in questions.items():
        qtype, crit = (raw.get("type"), raw.get("criteria")) if isinstance(raw, dict) else (None, None)
        # Only well-formed criteria can exceed bias.MAX_OPTIONS (26, the position-bias table's own cap; unrelated to a
        # "wide" two-letter-label question): a noul ignores extra keys, and a wrongly shaped one is a SetError.
        sized = (qtype == "choice" and isinstance(crit, dict)) or (qtype == "score" and isinstance(crit, list))
        if sized and len(crit) > MAX_OPTIONS:
            scan.skipped["too_many_options"] += 1          # parse_request would refuse the whole record
        else:
            keep[qid] = raw
    if not keep:
        return
    base = {k: v for k, v in record.items() if k not in NOT_FORWARDED}
    if "state" not in base:
        raise InvalidRequest("record has no state")
    req = parse_request({**base, "questions": {qid: {k: v for k, v in raw.items() if k not in ("label", "src")}
                                               for qid, raw in keep.items()}})
    kind = req.kind                         # one definition of a readout's kind
    for q in req.questions:
        raw = keep[q.qid]
        reason = skip_reason(q)
        if reason is not None:
            scan.skipped[reason] += 1
            continue
        uid = f"L{line}/{q.qid}"
        scan.items.append(Item(
            uid=uid, record_id=str(meta.get("id") or f"line {line}"), qid=q.qid,
            cluster=str(meta.get("group_id") or meta.get("pair_id") or uid),
            source=str(meta.get("source") or raw.get("src") or "unknown"), kind=kind, n=len(q.labels),
            texts=q.texts, instructions=q.instructions, label=_label_index(raw, q), base=base))


def read_set(text: str) -> SetScan:
    """Every eligible question of a JSONL set. A line that is not JSON, not an object, or not a System One request
    raises SetError naming its line (exit 1 in both calibrate modes)."""
    scan = SetScan()
    for i, line in enumerate(text.split("\n"), 1):     # "\n" only: a JSON string may hold a raw U+2028
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as e:
            raise SetError(f"line {i}: not valid JSON ({e})") from None
        if not isinstance(record, dict):
            raise SetError(f"line {i}: not a JSON object")
        try:
            scan_record(record, i, scan)
        except (InvalidRequest, KeyError, TypeError, AttributeError) as e:
            raise SetError(f"line {i}: not a System One request ({type(e).__name__}: {e})") from None
    return scan


def cap_per_n(items: Sequence[Item], max_per_n: int, seed: int) -> tuple[list[Item], int]:
    """--max-per-n: per n, a seeded shuffle within each source, then a round-robin over the sources in sorted
    order, so a file sorted by source cannot fill an n from its first source alone. Returns (the kept items in
    file order, how many were capped)."""
    by_n: dict[int, dict[str, list[Item]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_n[it.n][it.source].append(it)
    keep: set[str] = set()
    for n, groups in by_n.items():
        queues = []
        for src in sorted(groups):
            group = list(groups[src])
            random.Random(f"{seed}:{n}:{src}").shuffle(group)
            queues.append(group)
        picked = 0
        while picked < max_per_n and any(queues):
            for queue in queues:
                if queue and picked < max_per_n:
                    keep.add(queue.pop(0).uid)
                    picked += 1
    kept = [it for it in items if it.uid in keep]
    return kept, len(items) - len(kept)


# ---- the estimator and the in-sample evaluation ---------------------------------------------

# (uid, rotation) -> (served probabilities at T = 1 in position order, topk_miss)
Readings = Mapping[tuple[str, int], tuple[Sequence[float], int]]
_TINY = 1e-300


# each position's mean is floored at this BEFORE the renormalisation, so every log is finite. The renormalised
# prior can sit just under it (rows sum to 1, so the sum is then below 1 + (n-1)·PRIOR_FLOOR): a floored position ends
# at PRIOR_FLOOR / sum >= PRIOR_FLOOR / (1 + (n-1)·PRIOR_FLOOR), 9.99998e-7 at n = 3, never 0. Inactive on real
# readouts.
PRIOR_FLOOR = 1e-6


def question_row(probs: Sequence[Sequence[float]]) -> np.ndarray:
    """One question's row (rows = its n rotations): r_q[k] = (1/n) Σ_s p(q, s, k), the mean served probability
    at position k. With no position effect every position sees every option once, so the row is exactly uniform: the
    significance test's null."""
    return np.asarray(probs, dtype=float).mean(axis=0)


def prior_from_rows(rows: np.ndarray) -> tuple[float, ...]:
    """prior_n: the mean of the question rows, floored at PRIOR_FLOOR and renormalised to sum to 1."""
    p = np.maximum(np.asarray(rows, dtype=float).mean(axis=0), PRIOR_FLOOR)
    return tuple(float(x) for x in p / p.sum())


def offsets_from_prior(prior: Sequence[float]) -> tuple[float, ...]:
    """offset_n[k] = log prior_n[k] - (1/n) Σ_j log prior_n[j]."""
    lp = [math.log(p) for p in prior]
    c = math.fsum(lp) / len(lp)
    return tuple(v - c for v in lp)


def debias(probs: Sequence[float], offsets: Sequence[float] | None) -> list[float]:
    """softmax(log p - offset). For p served at T = 1 this is exactly what serving with the profile gives:
    log p_k = lm_k - logsumexp(lm), and the constant cancels in the softmax."""
    if offsets is None:
        return [float(x) for x in probs]
    z = [math.log(max(float(p), _TINY)) - o for p, o in zip(probs, offsets, strict=True)]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    s = math.fsum(e)
    return [x / s for x in e]


def complete_cells(it: Item, readings: Readings) -> list[tuple[Sequence[float], int]] | None:
    """The question's readouts under all n rotations, in rotation order, or None when one is missing."""
    cells = [readings.get((it.uid, s)) for s in range(it.n)]
    return None if any(c is None for c in cells) else cells  # type: ignore[return-value]


@dataclass(frozen=True, eq=False)    # identity equality and hash: `rows` is an ndarray
class NFit:
    n: int
    rows: np.ndarray                    # (questions, n): each question's row r_q
    clusters: tuple[str, ...]           # the cluster of each row
    prior: tuple[float, ...]            # prior_n: the mean of `rows`, floored at PRIOR_FLOOR and renormalised
    questions: int                      # questions read under all n rotations: every one is fitted
    floored: int                        # of those, the ones with a floored option in some rotation (counted, fitted)


def fit_n(items: Sequence[Item], readings: Readings) -> dict[int, NFit]:
    """The estimator per option count. Only questions read under every rotation count; an n with none is
    omitted, so it appears in neither the profile's `entries` nor its `dropped`. A floored option is served at
    an upper bound of at most the 100th-ranked token's probability, so in probability space it moves a row by at most
    that much: floored questions are fitted, and counted."""
    by_n: dict[int, list[Item]] = defaultdict(list)
    for it in items:
        by_n[it.n].append(it)
    out: dict[int, NFit] = {}
    for n in sorted(by_n):
        rows: list[np.ndarray] = []
        clusters: list[str] = []
        floored = 0
        for it in by_n[n]:
            cells = complete_cells(it, readings)
            if cells is None:
                continue
            rows.append(question_row([c[0] for c in cells]))
            clusters.append(it.cluster)
            floored += any(c[1] > 0 for c in cells)
        if not rows:
            continue
        arr = np.array(rows, dtype=float).reshape(len(rows), n)
        out[n] = NFit(n, arr, tuple(clusters), prior_from_rows(arr), len(rows), floored)
    return out


def _content_argmax(probs: Sequence[float], s: int) -> int:
    """The ORIGINAL option the readout of rotation s picks: position k holds option (k + s) mod n."""
    k = max(range(len(probs)), key=probs.__getitem__)
    return (k + s) % len(probs)


def consistent(cells: Sequence[tuple[Sequence[float], int]], offsets: Sequence[float] | None) -> bool:
    """True when the question's argmax CONTENT is the same under all its rotations (readouts debiased by `offsets`)."""
    return len({_content_argmax(debias(p, offsets), s) for s, (p, _) in enumerate(cells)}) == 1


def labelled_rows(items: Sequence[Item], readings: Readings,
                  offsets_by_n: Mapping[int, Sequence[float]]) -> list[tuple[list[float], int]]:
    """(probabilities, label) for every rotation of every labelled question, debiased where n has offsets."""
    rows = []
    for it in items:
        cells = complete_cells(it, readings)
        if cells is None or it.label is None:
            continue
        for s, (p, _) in enumerate(cells):
            rows.append((debias(p, offsets_by_n.get(it.n)), rotated_label(it.label, s, it.n)))
    return rows


def evaluate(items: Sequence[Item], readings: Readings, offsets_by_n: Mapping[int, Sequence[float]]) -> dict:
    """In-sample: per n, rotation consistency (the fraction of questions whose argmax CONTENT is the same under
    all n rotations) and meanP_n, raw vs debiased; on a labelled set also accuracy, NLL and ECE10 at T = 1 over every
    rotation (labels remapped). An n without offsets reads the same raw and debiased."""
    by_n: dict[int, list[Item]] = defaultdict(list)
    for it in items:
        by_n[it.n].append(it)
    out: dict[str, dict] = {}
    for name, offs in (("raw", {}), ("debiased", dict(offsets_by_n))):
        block: dict[str, Any] = {"consistency": {}, "meanP": {}}
        for n in sorted(by_n):
            done = [c for it in by_n[n] if (c := complete_cells(it, readings)) is not None]
            if not done:
                continue
            block["consistency"][str(n)] = sum(consistent(c, offs.get(n)) for c in done) / len(done)
            ps = np.array([debias(p, offs.get(n)) for c in done for p, _ in c])
            block["meanP"][str(n)] = [float(x) for x in ps.mean(axis=0)]
        rows = labelled_rows(items, readings, offs)
        if rows:
            hits = sum(max(range(len(p)), key=p.__getitem__) == y for p, y in rows)
            block.update(rows=len(rows), accuracy=hits / len(rows), nll=nll(rows, 1.0), ece10=ece(rows, 1.0))
        out[name] = block
    return out


# ---- the built-in set (fetched on first use, never shipped in the package) ------------------------------------------

BUILTIN_SET = "builtin-v1"
# The wheel ships code only: builtin-v1 lives in the HF dataset below, pinned by commit and sha256, and is fetched on
# the first calibration that reads it (the repo keeps the source of truth at bench/data/bias_set.jsonl).
BUILTIN_SET_REPO = "peculiar-ragdoll/callo-bias-set"
# The dataset commit that holds bias_set.jsonl + card; `make release-check` fetches it before a publish.
BUILTIN_SET_REVISION = "05abe05d908f34325b0cf1e0ab45f420ab0e285e"
BUILTIN_SET_FILE = "bias_set.jsonl"
BUILTIN_SET_SHA256 = "f73ff674470f75993076e4e00cc306c88fa1f28bde1e85b49dcf03738a9811a1"
BUILTIN_SET_SIZE = 246712                 # bytes: the download reads at most one more
BUILTIN_SET_ENV = "CALLO_BIAS_SET"
FETCH_TIMEOUT = 30.0


class BuiltinSetError(CalloError):
    """The built-in set could not be read, fetched or verified: the CLI prints the message and exits 1."""


def builtin_set_url() -> str:
    return f"https://huggingface.co/datasets/{BUILTIN_SET_REPO}/resolve/{BUILTIN_SET_REVISION}/{BUILTIN_SET_FILE}"


def builtin_set_cache_path() -> Path:
    """`<callo cache dir>/bias_set/builtin-v1-<sha256[:12]>.jsonl`, under `upstream.default_cache_dir()`."""
    return default_cache_dir() / "bias_set" / f"{BUILTIN_SET}-{BUILTIN_SET_SHA256[:12]}.jsonl"


def builtin_set_sha256() -> str:
    """The built-in set's sha256: what makes a set built-in, whatever file name it is passed under. A
    pinned constant: it never reads a file or the network."""
    return BUILTIN_SET_SHA256


def _offline_route() -> str:
    return (f"download {builtin_set_url()} yourself and point {BUILTIN_SET_ENV} at it "
            f"({BUILTIN_SET_ENV}=/path/to/{BUILTIN_SET_FILE}), or calibrate on your own questions with "
            f"--data YOUR.jsonl")


def _read_verified(path: Path) -> bytes | None:
    """The file's bytes when its sha256 is the pinned one, None when it is not; OSError when it cannot be read."""
    data = path.read_bytes()
    return data if hashlib.sha256(data).hexdigest() == BUILTIN_SET_SHA256 else None


def _download(url: str) -> bytes:
    """At most BUILTIN_SET_SIZE + 1 bytes: one more than the set shows a longer body without reading all of it."""
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as r:  # noqa: S310 -- the pinned https URL above
        return r.read(BUILTIN_SET_SIZE + 1)


def _cache(path: Path, data: bytes, err: IO[str]) -> None:
    """Write `data` to `path` atomically: a temp file in the same directory, then `os.replace`. Best effort: an
    unwritable cache costs a download next time, not this run."""
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        tmp = None
    except OSError as e:
        print(f"callo: cannot cache the built-in bias set at {path} ({e}); it will be fetched again next time",
              file=err)
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def builtin_set_bytes(*, err: IO[str] | None = None) -> bytes:
    """The built-in set `builtin-v1` (built once by bench/build_bias_set.py from MMLU validation, MIT, and Banking77
    train, CC-BY-4.0; see NOTICE), verified against BUILTIN_SET_SHA256, from the first of:

    a. the file `CALLO_BIAS_SET` names (the offline route);
    b. callo's cache, `builtin_set_cache_path()` (a cached file that fails verification is deleted and fetched again,
       once);
    c. a download of `builtin_set_url()`, cached atomically.

    Raises BuiltinSetError (a CalloError: the CLI prints one line, exit 1) with the URL and the offline route. Only a
    path that needs the set's CONTENT calls this; identity checks use `builtin_set_sha256()`. The download line and
    a cache-write note go to `err` (default sys.stderr: calibrate passes its own)."""
    err = err or sys.stderr
    if env := os.environ.get(BUILTIN_SET_ENV):
        path = Path(env).expanduser()
        try:
            data = path.read_bytes()
        except OSError as e:
            raise BuiltinSetError(f"{BUILTIN_SET_ENV}={env}: cannot read the built-in bias set there ({e})") from None
        got = hashlib.sha256(data).hexdigest()
        if got != BUILTIN_SET_SHA256:
            raise BuiltinSetError(f"{BUILTIN_SET_ENV}={path}: not the built-in bias set {BUILTIN_SET}: sha256 {got}, "
                                  f"expected {BUILTIN_SET_SHA256}; download {builtin_set_url()}") from None
        return data
    cache = builtin_set_cache_path()
    with contextlib.suppress(OSError):
        if (cached := _read_verified(cache)) is not None:
            return cached
        cache.unlink()                          # it exists and fails verification: fetch it again, once
    url = builtin_set_url()
    print(f"callo: fetching the built-in bias set ({round(BUILTIN_SET_SIZE / 1000)} KB) from {url}", file=err)
    try:
        data = _download(url)
    except (OSError, http.client.HTTPException, ValueError) as e:
        raise BuiltinSetError(f"cannot fetch the built-in bias set {BUILTIN_SET} from {url} ({e}); "
                              f"{_offline_route()}") from None
    if len(data) > BUILTIN_SET_SIZE:
        raise BuiltinSetError(f"the response from {url} is longer than the built-in bias set's {BUILTIN_SET_SIZE} "
                              f"bytes; refusing it: {_offline_route()}")
    got = hashlib.sha256(data).hexdigest()
    if got != BUILTIN_SET_SHA256:
        raise BuiltinSetError(f"the built-in bias set fetched from {url} has sha256 {got}, expected "
                              f"{BUILTIN_SET_SHA256}; refusing it: {_offline_route()}")
    _cache(cache, data, err)
    return data
