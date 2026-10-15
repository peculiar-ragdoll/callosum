"""Refit a position-bias profile offline from a `callo calibrate --bias` raw file (a bench script,
not a CLI mode).

    uv run python -m bench.bias_refit RAW.jsonl [--data FILE] [--min-questions 24] [--max-per-n N] --out PROFILE.json

The raw header names the set by sha256: pass the same --data (or none for the built-in set); another set refuses.
--max-per-n defaults to the one the raw header records. `created`, `model_name` (not in the raw header) and the
guard checks' times are the only fields that can differ from the profile calibrate wrote. The file is parsed by
`callo.calibrate_bias.read_raw`, the one reader of the format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from callo.bias import builtin_set_bytes, builtin_set_sha256, cap_per_n, read_set
from callo.calibrate_bias import Collected, build_profile, ragged_reading, read_raw
from callo.errors import InvalidInput

REPO_SET = Path(__file__).resolve().parent / "data" / "bias_set.jsonl"


def builtin_bytes() -> bytes:
    """The built-in set for the bench: the repo's checked-in copy (the file uploaded as the HF dataset), verified
    against callo's pinned sha256, so a bench run never fetches; callo's own resolution (CALLO_BIAS_SET, the cache, a
    download) only when the repo copy is absent."""
    if REPO_SET.is_file():
        data = REPO_SET.read_bytes()
        if hashlib.sha256(data).hexdigest() != builtin_set_sha256():
            raise SystemExit(f"{REPO_SET}: sha256 {hashlib.sha256(data).hexdigest()}, not the pinned built-in set "
                             f"{builtin_set_sha256()}")
        return data
    return builtin_set_bytes()


def refit(raw_path: str | Path, data: str | Path | None = None, *, min_questions: int = 24,
          max_per_n: int | None = None) -> dict:
    """Raises SystemExit with a message for every input it cannot refit (the CLI's bounds, a header seed numpy would
    refuse at the fit, a raw file read_raw refuses, a readout that is not its question's width)."""
    if min_questions < 2 or (max_per_n is not None and max_per_n < 1):
        # the CLI's bounds: below 2 questions the permutation test divides by zero
        raise SystemExit("--min-questions must be at least 2 and --max-per-n at least 1")
    try:
        raw = read_raw(raw_path)
    except InvalidInput as e:
        raise SystemExit(str(e)) from None
    header = raw.header
    seed = header.get("seed")
    if type(seed) is not int or seed < 0:      # `is`: a bool is no int; numpy's default_rng refuses a negative seed
        raise SystemExit(f"{raw_path}: the header's seed {seed!r} is not a non-negative integer, so it cannot be "
                         f"refitted")
    text = builtin_bytes() if data is None else Path(data).read_bytes()
    sha = hashlib.sha256(text).hexdigest()
    if sha != header["set"]["sha256"]:
        raise SystemExit(f"{raw_path} was measured on set sha256 {header['set']['sha256']}, not {sha}: pass the same "
                         f"--data")
    scan = read_set(text.decode("utf-8"))
    cap = header.get("max_per_n", 48) if max_per_n is None else max_per_n
    items, capped = cap_per_n(scan.items, cap, seed)
    if (ragged := ragged_reading(items, raw.readings)) is not None:
        raise SystemExit(f"{raw_path}: {ragged}")
    server = {**header["server"], "slot": header["server"]["slot"] if raw.slot is None else raw.slot}
    stats = {"model_key": header["model_key"], "model_file": None, "provenance": header["provenance"]}
    profile, _ = build_profile(items, Collected(raw.readings, raw.checks, server, stats, header),
                               set_name=header["set"]["name"], set_sha=sha, seed=seed,
                               skipped=Counter(scan.skipped), capped=capped, min_questions=min_questions,
                               builtin=sha == builtin_set_sha256())
    return profile


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="refit a position-bias profile from a calibrate --bias raw file")
    ap.add_argument("raw")
    ap.add_argument("--data", default=None)
    ap.add_argument("--min-questions", type=int, default=24)
    ap.add_argument("--max-per-n", type=int, default=None, help="default: the raw header's")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    profile = refit(a.raw, a.data, min_questions=a.min_questions, max_per_n=a.max_per_n)
    Path(a.out).write_text(json.dumps(profile, indent=1) + "\n")
    print(f"entries {sorted(profile['entries'])}, dropped {profile['dropped']} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
