"""The `callo` console script and `python -m callo`: refuse Windows, then hand over to `callo.cli`.

Callo is POSIX-only (process groups, SIGKILL, `os.kill(pid, 0)` liveness probes, which on Windows terminate the
process instead of probing it). This module imports nothing but `sys` before the check, so on Windows the user gets
one plain line and exit 2 instead of an import traceback; `callo.cli` and everything POSIX-only is imported only past
it. On Linux and macOS it is a pass-through: `callo.cli.main` sees the same arguments and prints the same bytes.
`import callo` as a library stays importable everywhere (the package `__init__` is trivial)."""
from __future__ import annotations

import sys

WINDOWS_REFUSAL = "Callo runs on Linux and macOS. Windows support is coming; for now, run it under WSL2."


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        print(WINDOWS_REFUSAL, file=sys.stderr)
        return 2
    from .cli import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
