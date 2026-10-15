"""The wide-choice validation items: 20 BANKING77 and 20 CLINC150 utterances from the TRAIN
splits (never a test split, never the Decision Index subset), each read against its dataset's full intent list (77
and 151 options) and as a 10-option control. Run once, on the CPU, with the HF cache outside ~/.cache (the default
cache is never written):

    HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HOME=/path/to/hf-home \\
        HF_XET_CACHE=/path/to/hf-home/xet \\
        uv run --frozen --with pyarrow==25.0.1 --with huggingface_hub python -m bench.wide_items

Sources, pinned by revision (`REVISIONS`, resolved on the first fetch, 2026-10-10) and by the sha256 of the file read:
- BANKING77 (`PolyAI/banking77`, CC-BY-4.0), train. That repository is a loading script with no parquet conversion
  (`refs/convert/parquet` does not exist: the Hub stopped converting script datasets), so the parquet is
  `legacy-datasets/banking77`, the Hub's own parquet copy of it. Checked once on 2026-10-10: its 10,003 (text,
  intent) rows equal, in order, the script's source CSV (`PolyAI-LDN/task-specific-datasets` @ 57ec275,
  `banking_data/train.csv`, the file `bench/build_bias_set.py` pins), and its ClassLabel names equal the script's.
- CLINC150 (`clinc_oos`, now `clinc/clinc_oos`, config `plus`: 150 intents and `oos`, CC-BY-3.0), train.

Each source: seed 0 picks 20 distinct intents, then one utterance of each (a non-blank one, in file order). Each item
carries the dataset's full label list in ClassLabel order, and a control: the gold plus 9 distractors drawn with seed
1, shuffled with seed 2 (one generator per seed and source, used in item order). The file's first row is `_meta`
(sources, revisions, sha256s, licences, attribution, seeds); then the items, BANKING77 first. Deterministic: the same
parquet files give a byte-identical file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "data" / "wide_items.jsonl"
N_ITEMS, N_CONTROL = 20, 10
SAMPLE_SEED, DISTRACTOR_SEED, SHUFFLE_SEED = 0, 1, 2
# the commit each source is read at (resolved on the first fetch, 2026-10-10); later runs fetch exactly these
REVISIONS = {"banking77": "f54121560de48f2852f90be299010d1d6dc612ec",
             "clinc150": "155b9c710419136e17307b80d0a13e68cd46b4ec"}
SOURCES: dict[str, dict[str, Any]] = {
    "banking77": {
        "dataset": "PolyAI/banking77", "repo": "legacy-datasets/banking77",
        "file": "data/train-00000-of-00001.parquet",
        "sha256": "3c648a31689f4ab3acbd4f4f4d120944bb521cf6ba57da77aa87c57e8979be81",
        "label_column": "label", "n_labels": 77, "license": "CC-BY-4.0",
        "attribution": ("BANKING77: Inigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson and Ivan Vulic "
                        "(PolyAI), \"Efficient Intent Detection with Dual Sentence Encoders\", NLP4ConvAI 2020, "
                        "https://github.com/PolyAI-LDN/task-specific-datasets. Licensed CC BY 4.0 "
                        "(https://creativecommons.org/licenses/by/4.0/). Used: train utterances and intent names, "
                        "unmodified; sampled.")},
    "clinc150": {
        "dataset": "clinc_oos (config plus)", "repo": "clinc/clinc_oos",
        "file": "plus/train-00000-of-00001.parquet",
        "sha256": "30188119cf9f86fc9db27e1c22442d091cb5cb0913c9496f945fe11e7a02a28f",
        "label_column": "intent", "n_labels": 151, "license": "CC-BY-3.0",
        "attribution": ("CLINC150: Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, "
                        "Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang and "
                        "Jason Mars, \"An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction\", "
                        "EMNLP-IJCNLP 2019, https://github.com/clinc/oos-eval. Licensed CC BY 3.0 "
                        "(https://creativecommons.org/licenses/by/3.0/). Used: train utterances and intent names "
                        "(150 intents and oos), unmodified; sampled.")},
}


def hf_cache_problem(env: Mapping[str, str]) -> str | None:
    """Why this environment would write the HF cache to the boot volume, or None: HF_HOME and HF_XET_CACHE must both
    be set, and neither may lie under ~/.cache (the boot volume's default cache)."""
    boot = (Path.home() / ".cache").resolve()
    for var in ("HF_HOME", "HF_XET_CACHE"):
        v = env.get(var)
        if not v:
            return f"{var} is not set: set HF_HOME and HF_XET_CACHE to a cache directory outside {boot}"
        if Path(v).expanduser().resolve().is_relative_to(boot):
            return f"{var}={v} is the default cache (under {boot}): point it at a cache directory outside it"
    return None


def label_names(schema_metadata: Mapping[bytes, bytes] | None, column: str) -> list[str]:
    """The ClassLabel names of `column`, from the parquet's `huggingface` schema metadata (dataset order)."""
    try:
        feat = json.loads((schema_metadata or {})[b"huggingface"])["info"]["features"][column]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"no huggingface feature metadata for {column!r}: {e}") from None
    names = feat.get("names") if isinstance(feat, dict) else None
    if feat.get("_type") != "ClassLabel" or not isinstance(names, list) or not names:
        raise ValueError(f"{column!r} is not a ClassLabel with names")
    return [str(n) for n in names]


def sample_items(source: str, rows: Sequence[tuple[str, int]], names: Sequence[str]) -> list[dict]:
    """`rows`: (text, label index) in file order. 20 distinct intents (seed 0), then one non-blank utterance of each
    (the same generator, in intent order); each item with every label name and its 10-option control."""
    src = SOURCES[source]
    by_label: dict[int, list[tuple[int, str]]] = {}
    for i, (text, label) in enumerate(rows):
        if isinstance(text, str) and text.strip():
            by_label.setdefault(int(label), []).append((i, text))
    rng = random.Random(SAMPLE_SEED)
    intents = rng.sample(range(len(names)), N_ITEMS)
    missing = [names[li] for li in intents if li not in by_label]
    if missing:
        raise ValueError(f"{source}: no usable train utterance for intent(s) {missing}")
    picks = [(li, rng.choice(by_label[li])) for li in intents]
    distract, shuffle = random.Random(DISTRACTOR_SEED), random.Random(SHUFFLE_SEED)
    items = []
    for li, (row, text) in picks:
        gold = names[li]
        control = [gold, *distract.sample([n for n in names if n != gold], N_CONTROL - 1)]
        shuffle.shuffle(control)
        items.append({"id": f"{source}/train/{row}", "source": source, "split": "train",
                      "revision": REVISIONS[source], "license": src["license"], "text": text, "gold": gold,
                      "labels": list(names), "control": control})
    return items


def meta_row() -> dict:
    return {"_meta": {"built_by": "bench/wide_items.py", "spec": "docs/specs/2026-10-09-callo-v0.3-wide-choice-design.md §8",
                      "seeds": {"sample": SAMPLE_SEED, "distractors": DISTRACTOR_SEED, "shuffle": SHUFFLE_SEED},
                      "sources": {s: {"dataset": v["dataset"], "repo": v["repo"], "file": v["file"], "split": "train",
                                      "revision": REVISIONS[s], "sha256": v["sha256"], "license": v["license"],
                                      "attribution": v["attribution"]} for s, v in SOURCES.items()}}}


def jsonl(records: Sequence[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


def cached_path(source: str, hf_home: str | Path) -> Path:
    """Where huggingface_hub keeps the pinned file under `hf_home`."""
    s = SOURCES[source]
    return (Path(hf_home) / "hub" / f"datasets--{s['repo'].replace('/', '--')}" / "snapshots" / REVISIONS[source]
            / s["file"])


def _check_sha(source: str, path: Path) -> None:
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != SOURCES[source]["sha256"]:
        raise SystemExit(f"{path}: sha256 {got}, expected {SOURCES[source]['sha256']}; refusing to build from a "
                         f"different file")


def build_from_files(paths: Mapping[str, Path]) -> bytes:
    """The whole file from the pinned parquet files (each sha256-checked)."""
    import pyarrow.parquet as pq  # noqa: PLC0415 -- a builder-only dependency, never a callo one
    items: list[dict] = []
    for source in SOURCES:
        _check_sha(source, paths[source])
        table = pq.read_table(paths[source])
        col = SOURCES[source]["label_column"]
        names = label_names(table.schema.metadata, col)
        if len(names) != SOURCES[source]["n_labels"]:
            raise SystemExit(f"{source}: {len(names)} labels, expected {SOURCES[source]['n_labels']}")
        rows = list(zip(table.column("text").to_pylist(), table.column(col).to_pylist(), strict=True))
        items += sample_items(source, rows, names)
    return jsonl([meta_row(), *items])


def _download(source: str) -> Path:
    """The pinned file through huggingface_hub (HF_HOME / HF_XET_CACHE; HF_TOKEN if set, the datasets are public)."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 -- a builder-only dependency
    s = SOURCES[source]
    return Path(hf_hub_download(s["repo"], s["file"], repo_type="dataset", revision=REVISIONS[source]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wide_items", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    problem = hf_cache_problem(os.environ)
    if problem is not None:
        print(f"refusing to start: {problem}", file=sys.stderr)
        return 2
    paths = {s: _download(s) for s in SOURCES}
    for s, p in paths.items():
        _check_sha(s, p)
    data = build_from_files(paths)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    n = data.count(b"\n") - 1
    print(f"{n} items -> {a.out} (sha256 {hashlib.sha256(data).hexdigest()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
