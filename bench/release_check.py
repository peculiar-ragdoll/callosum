"""`make release-check`: the gate to run by hand before publishing a release (network; never part of `make check`).

    uv run python -m bench.release_check

Fails while `callo.bias.BUILTIN_SET_REVISION` is still the all-zeros placeholder. Otherwise it fetches
`builtin_set_url()` the way a user's first `callo calibrate --bias` does, into a throwaway cache (HOME and
XDG_CACHE_HOME pointed at a temp dir, CALLO_BIAS_SET ignored), and checks the cached file's sha256 against
BUILTIN_SET_SHA256. The user's own environment and cache are restored and never touched.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

from callo import bias

PLACEHOLDER = "0" * 40
ENV = ("HOME", "XDG_CACHE_HOME", bias.BUILTIN_SET_ENV)


def main() -> int:
    if bias.BUILTIN_SET_REVISION == PLACEHOLDER:
        print("release-check FAILED: callo.bias.BUILTIN_SET_REVISION is still the all-zeros placeholder; set it to "
              f"the commit of the {bias.BUILTIN_SET_REPO} upload before publishing", file=sys.stderr)
        return 1
    saved = {k: os.environ.get(k) for k in ENV}
    try:
        with tempfile.TemporaryDirectory(prefix="callo-release-check-") as tmp:
            os.environ["HOME"], os.environ["XDG_CACHE_HOME"] = tmp, str(Path(tmp) / ".cache")
            os.environ.pop(bias.BUILTIN_SET_ENV, None)
            try:
                data = bias.builtin_set_bytes(err=sys.stderr)
            except bias.BuiltinSetError as e:
                print(f"release-check FAILED: {e}", file=sys.stderr)
                return 1
            cached = bias.builtin_set_cache_path()
            got = hashlib.sha256(cached.read_bytes()).hexdigest() if cached.is_file() else None
            if got != bias.BUILTIN_SET_SHA256 or hashlib.sha256(data).hexdigest() != bias.BUILTIN_SET_SHA256:
                print(f"release-check FAILED: the cached copy {cached} has sha256 {got}, expected "
                      f"{bias.BUILTIN_SET_SHA256}", file=sys.stderr)
                return 1
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print(f"release-check OK: {bias.builtin_set_url()} ({len(data)} bytes) has sha256 {bias.BUILTIN_SET_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
