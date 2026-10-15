"""The host fingerprint `callo serve` publishes and a think profile records, and the macOS
thermal reader the drift check uses. Gathered once, best-effort: nothing here fails startup."""
from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
from collections.abc import Callable
from typing import Any

Run = Callable[..., Any]        # subprocess.run's signature
_THERM = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")


def _out(run: Run, argv: list[str], timeout: float = 2.0) -> str | None:
    try:
        p = run(argv, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    out = getattr(p, "stdout", None)
    return out.strip() if isinstance(out, str) and getattr(p, "returncode", 1) == 0 and out.strip() else None


def _memory_bytes(system: str, run: Run) -> int | None:
    if system == "Darwin":
        v = _out(run, ["sysctl", "-n", "hw.memsize"])
        return int(v) if v and v.isdigit() else None
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def _chip(system: str, run: Run, cpuinfo: str = "/proc/cpuinfo") -> str | None:
    if system == "Darwin":
        return _out(run, ["sysctl", "-n", "machdep.cpu.brand_string"])
    if system == "Linux":
        try:
            with open(cpuinfo, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name") and line.partition(":")[2].strip():
                        return line.partition(":")[2].strip()
        except OSError:
            return None
    return None


def host_info(*, run: Run = subprocess.run, system: str | None = None, cpuinfo: str = "/proc/cpuinfo") -> dict:
    """`{"os", "machine", "chip", "memory_bytes", "gpu", "fingerprint"}`. `os` is "macOS-<version>-<machine>" on
    macOS, else "<system>-<release>-<machine>". `gpu` is nvidia-smi's name and total memory on Linux, else null. The
    fingerprint is sha256(chip|memory_bytes|gpu): `compare` uses it to tell "same box". It is None when neither chip
    nor gpu is known (RAM alone does not identify a machine, and every unprobeable box would otherwise share one
    hash): consumers read None as "unknown host", never as a match. `run` and `system` are injectable for tests."""
    system = system or platform.system()
    chip = _chip(system, run, cpuinfo)
    memory = _memory_bytes(system, run)
    gpu = _out(run, ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]) if system == "Linux" \
        else None
    fp = None if chip is None and gpu is None else \
        "sha256:" + hashlib.sha256(f"{chip}|{memory}|{gpu}".encode()).hexdigest()
    machine = platform.machine()
    mac = platform.mac_ver()[0] if system == "Darwin" else ""
    os_name = f"macOS-{mac}-{machine}" if mac else f"{system}-{platform.release()}-{machine}"
    return {"os": os_name, "machine": machine, "chip": chip, "memory_bytes": memory, "gpu": gpu, "fingerprint": fp}


def therm_limit(*, run: Run = subprocess.run, system: str | None = None) -> int | None:
    """macOS `pmset -g therm`'s CPU_Speed_Limit (100 = no throttling), or None: not macOS, no pmset, or a machine that
    prints the key only after a thermal event (null is "no thermal data", never evidence)."""
    if (system or platform.system()) != "Darwin":
        return None
    out = _out(run, ["pmset", "-g", "therm"], timeout=5.0)
    m = _THERM.search(out or "")
    return int(m.group(1)) if m else None
