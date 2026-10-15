"""Build callo's built-in position-bias set `builtin-v1` and its held-out remainder, once.

    uv run --with pyarrow==25.0.1 python -m bench.build_bias_set --cache <a scratch directory>

Downloads two pinned, permissively licensed sources (network), checks their sha256, and writes:

- `bench/data/bias_set.jsonl`: 336 questions, 48 per n = 2..8 (1,680 readouts). n = 2, 3, 4: 24 from MMLU
  validation and 24 from Banking77 train each; n = 5..8: 48 from Banking77 only, a single domain.
- `bench/data/bias_heldout_v1.jsonl`: the held-out remainder for the acceptance bench, 48 per (source, n).
- `bench/data/bias_set_manifest.json`: sources, revisions, file sha256s, the salt, counts and the outputs' sha256.

Nothing from an upstream TEST split: cais/mmlu `validation` (MIT) and Banking77 `train.csv` (CC-BY-4.0; its `test.csv`
is an upstream test split). Every row lands in one pool by a hash of its normalised text, so duplicate texts cannot
straddle the built-in set and the held-out file; each draws from its own pool in a seeded hash order. A row any of
whose options calibration excludes (`callo.bias`: a bare letter, a cross-reference) is not used. Deterministic: rerunning gives
byte-identical files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import string
import sys
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from callo.bias import exclusion

ROOT = Path(__file__).resolve().parents[1]
SALT = "bias-v1"
SET_NAME, HELDOUT_NAME = "builtin-v1", "heldout-v1"
MMLU = {"repo": "cais/mmlu", "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "file": "all/validation-00000-of-00001.parquet",
        "sha256": "66cdf0b090ccb657d18d13cd81e31fdc55c3467da9642ffb178653268a97c8ef", "licence": "MIT"}
MMLU_URL = f"https://huggingface.co/datasets/{MMLU['repo']}/resolve/{MMLU['revision']}/{MMLU['file']}"
B77 = {"repo": "PolyAI-LDN/task-specific-datasets", "revision": "57ec275d8078af65b7731c2a98be812d844a6d6b",
       "files": {"banking_data/train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
                 "banking_data/categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63"},
       "licence": "CC-BY-4.0"}
B77_URL = f"https://raw.githubusercontent.com/{B77['repo']}/{B77['revision']}/{{file}}"
MMLU_INSTRUCTIONS = "Which option correctly answers the question?"        # the transfer set's own MMLU wording
B77_INSTRUCTIONS = "Which intent does this customer message express?"
BUILTIN_COUNTS = {"mmlu": {2: 24, 3: 24, 4: 24}, "banking77": {2: 24, 3: 24, 4: 24, 5: 48, 6: 48, 7: 48, 8: 48}}
HELDOUT_COUNTS = {"mmlu": {n: 48 for n in (2, 3, 4)}, "banking77": {n: 48 for n in range(2, 9)}}
OUT_SET = ROOT / "bench" / "data" / "bias_set.jsonl"     # uploaded as the HF dataset callo fetches
OUT_HELDOUT = ROOT / "bench" / "data" / "bias_heldout_v1.jsonl"
OUT_MANIFEST = ROOT / "bench" / "data" / "bias_set_manifest.json"


def _h(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in (SALT, *parts)).encode()).digest()[:8], "big")


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def pool(source: str, text: str) -> str:
    """'builtin' or 'heldout', by a hash of the normalised text: a duplicate text always lands in the same pool."""
    return "heldout" if _h("pool", source, norm(text)) % 2 else "builtin"


def order(source: str, row: int) -> int:
    return _h("order", source, row)


def _mmlu_text(row: dict) -> str:
    return f"{row['subject']}\n{row['question']}"


def mmlu_usable(row: dict) -> bool:
    choices = row.get("choices")
    if not isinstance(choices, list) or len(choices) != 4 or not all(isinstance(c, str) and c.strip() for c in choices):
        return False
    return len({c.strip() for c in choices}) == 4 and exclusion(choices) is None


def mmlu_record(row_index: int, row: dict, n: int, set_name: str) -> dict:
    """n = 4 as-is; n = 2, 3: the gold and the first n - 1 distractors, in file order."""
    choices, gold = list(row["choices"]), int(row["answer"])
    keep = sorted([gold, *[i for i in range(len(choices)) if i != gold][: n - 1]])
    letters = string.ascii_lowercase
    return {"state": {"subject": row["subject"].replace("_", " "), "question": row["question"]},
            "questions": {"answer": {"type": "choice", "instructions": MMLU_INSTRUCTIONS,
                                     "criteria": {letters[j]: choices[i] for j, i in enumerate(keep)},
                                     "label": letters[keep.index(gold)], "src": "mmlu"}},
            "_meta": {"id": f"{set_name}/mmlu/validation/{row_index}/n{n}", "source": "mmlu", "repo": MMLU["repo"],
                      "revision": MMLU["revision"], "split": "validation", "row": row_index, "n": n,
                      "variant": "clean"}}


def lead(a: str, b: str) -> int:
    """How many leading `_`-separated tokens two intent names share (card_* with card_*: 1; top_up_* pairs: 2),
    case-insensitively (Banking77 capitalises some names: `Refund_not_showing_up`)."""
    n = 0
    for x, y in zip(a.lower().split("_"), b.lower().split("_"), strict=False):
        if x != y:
            break
        n += 1
    return n


def jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split("_")), set(b.lower().split("_"))
    return len(ta & tb) / len(ta | tb)


def distractors(gold: str, intents: Sequence[str], k: int, rng: random.Random) -> list[str]:
    """The k most confusable other intents: shared prefix first, then token overlap, then a seeded tiebreak."""
    tie = {x: rng.random() for x in sorted(intents)}
    others = [x for x in sorted(intents) if x != gold]
    others.sort(key=lambda x: (-lead(gold, x), -jaccard(gold, x), tie[x]))
    return others[:k]


def banking_record(row_index: int, text: str, gold: str, intents: Sequence[str], n: int, set_name: str) -> dict:
    rng = random.Random(f"{SALT}:banking77:{row_index}:{n}")
    options = [gold, *distractors(gold, intents, n - 1, rng)]
    rng.shuffle(options)
    return {"state": text,
            "questions": {"intent": {"type": "choice", "instructions": B77_INSTRUCTIONS,
                                     "criteria": {o.replace("_", " "): None for o in options},
                                     "label": gold.replace("_", " "), "src": "banking77"}},
            "_meta": {"id": f"{set_name}/banking77/train/{row_index}/n{n}", "source": "banking77", "repo": B77["repo"],
                      "revision": B77["revision"], "split": "train", "file": "banking_data/train.csv",
                      "row": row_index, "n": n, "variant": "clean"}}


def _draw(rows: Sequence[tuple[int, Any]], counts: dict[int, int], make: Callable[[int, Any, int], dict]) -> list[dict]:
    """Rows in their given order fill n = 2 to its count, then n = 3, ...: each row is used at most once."""
    it = iter(rows)
    out = []
    for n in sorted(counts):
        for _ in range(counts[n]):
            try:
                idx, row = next(it)
            except StopIteration:
                raise RuntimeError(f"the pool ran out of rows at n = {n}") from None
            out.append(make(idx, row, n))
    return out


def build(mmlu_rows: Sequence[dict], banking_rows: Sequence[tuple[str, str]],
          intents: Sequence[str]) -> tuple[list[dict], list[dict]]:
    """(the built-in set, the held-out remainder), each sorted by (n, source, row)."""
    mmlu = [(i, r) for i, r in enumerate(mmlu_rows) if mmlu_usable(r)]
    seen: set[str] = set()
    banking: list[tuple[int, tuple[str, str]]] = []
    for i, (text, cat) in enumerate(banking_rows):
        if not text.strip() or cat not in intents or norm(text) in seen:
            continue
        seen.add(norm(text))
        banking.append((i, (text, cat)))
    out = {}
    for name, counts, set_name in (("builtin", BUILTIN_COUNTS, SET_NAME), ("heldout", HELDOUT_COUNTS, HELDOUT_NAME)):
        m = sorted((x for x in mmlu if pool("mmlu", _mmlu_text(x[1])) == name), key=lambda x: order("mmlu", x[0]))
        k = sorted((x for x in banking if pool("banking77", x[1][0]) == name), key=lambda x: order("banking77", x[0]))
        recs = (_draw(m, counts["mmlu"], lambda i, r, n, s=set_name: mmlu_record(i, r, n, s))
                + _draw(k, counts["banking77"],
                        lambda i, r, n, s=set_name: banking_record(i, r[0], r[1], intents, n, s)))
        out[name] = sorted(recs, key=lambda r: (r["_meta"]["n"], r["_meta"]["source"], r["_meta"]["row"]))
    return out["builtin"], out["heldout"]


def fetch(url: str, sha256: str, cache: Path) -> bytes:
    """The file at `url`, cached in `cache`, refused unless its sha256 is the pinned one."""
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / hashlib.sha256(url.encode()).hexdigest()[:16]
    if not dest.is_file():
        with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 -- pinned public dataset URLs
            dest.write_bytes(r.read())
    data = dest.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise SystemExit(f"{url}: sha256 {got}, expected {sha256}; refusing to build from a different file")
    return data


def _jsonl(records: Sequence[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="where the downloaded source files are kept")
    a = ap.parse_args(argv)
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 -- a builder-only dependency, never a callo one
    except ImportError:
        print("needs pyarrow: run `uv run --with pyarrow==25.0.1 python -m bench.build_bias_set --cache DIR`",
              file=sys.stderr)
        return 2
    cache = Path(a.cache)
    mmlu_rows = pq.read_table(io.BytesIO(fetch(MMLU_URL, MMLU["sha256"], cache))).to_pylist()
    files = {f: fetch(B77_URL.format(file=f), sha, cache) for f, sha in B77["files"].items()}
    banking_rows = [(r["text"], r["category"])
                    for r in csv.DictReader(io.StringIO(files["banking_data/train.csv"].decode("utf-8")))]
    intents = json.loads(files["banking_data/categories.json"])
    builtin, heldout = build(mmlu_rows, banking_rows, intents)
    outputs = {OUT_SET: _jsonl(builtin), OUT_HELDOUT: _jsonl(heldout)}
    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {"salt": SALT, "set": SET_NAME, "heldout": HELDOUT_NAME,
                "sources": {"mmlu": {**MMLU, "url": MMLU_URL},
                            "banking77": {**B77, "urls": [B77_URL.format(file=f) for f in B77["files"]]}},
                "counts": {name: {f"{s}/{n}": k for (s, n), k in sorted(Counter(
                    (r["_meta"]["source"], r["_meta"]["n"]) for r in recs).items())}
                    for name, recs in (("builtin", builtin), ("heldout", heldout))},
                "outputs": {str(p.relative_to(ROOT)): hashlib.sha256(d).hexdigest() for p, d in outputs.items()}}
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"builtin-v1: {len(builtin)} questions, {sum(r['_meta']['n'] for r in builtin)} readouts -> {OUT_SET}")
    print(f"heldout-v1: {len(heldout)} questions -> {OUT_HELDOUT}")
    print(f"manifest -> {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
