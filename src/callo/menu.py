"""Model menus: `callo calibrate --think-budgets` over several GGUFs, one at a time, on the same
data, budgets, seed and sampling policy, then the joint report (`compare.py`).

This module is the driver around `calibrate_think.calibrate_one`: the sources and their names, `MENU.json` and
its keys, the shared launch flags and the `args` policy; the GPU rule and the memory precheck;
`callo serve` in-process per model with its own log file, the pid-file SIGTERM of a loading child, settle/cooldown;
`run.json`, failure isolation and `--resume`; `--dry-run` with `--rates-from`.

The `args` policy: llama-server flags that constrain the trace and cannot be overridden per request (`--grammar`,
`--json-schema` and their file forms) are refused, exit 2; sampler flags are allowed with a warning that the pinned
sampler overrides them; model and speed configuration is allowed and recorded by `callo serve` in
`provenance.server_args`, which the report shows.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from . import compare
from .calibrate_bias import Refusal, silence_stdout
from .calibrate_think import (
    DEFAULT_OUT_DIR,
    DRIFT_N,
    GATE_READS,
    SERVE_START_TIMEOUT_S,
    Outcome,
    ThinkSettings,
    cache_resume_hint,
    calibrate_one,
    dry_run_lines,
    load_data,
    reads_left,
    settings_from_args,
    think_usage_error,
    url_given,
)
from .errors import CalloError, InvalidInput
from .host import host_info, therm_limit
from .modelkey import gguf_bytes, load_model_key
from .think import PROFILE_FORMAT, finite_positive, parse_sampling
from .upstream import find_orphans

NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
MENU_KEYS = ("name", "model", "mmproj", "ctx", "parallel", "no_kvu", "s1_slots", "llama_server", "args",
             "user_delimiter", "think_sampling", "position_bias")
DEFAULT_PARALLEL, DEFAULT_S1_SLOTS, DEFAULT_LLAMA_SERVER = 4, 1, "llama-server"
REFUSED_ARGS = ("--grammar", "--grammar-file", "-j", "--json-schema", "--json-schema-file")
SAMPLER_ARGS = ("--temp", "--top-k", "--top-p", "--min-p", "--top-nsigma", "--typical", "--presence-penalty",
                "--frequency-penalty", "--samplers", "--sampling-seq", "--min-keep", "--ignore-eos", "-l",
                "--logit-bias", "--reasoning-budget", "--reasoning-budget-message")
SAMPLER_PREFIXES = ("--repeat-", "--dry-", "--xtc-", "--mirostat", "--dynatemp-", "--adaptive-")
CACHE_RAM_OFF = ("--cache-ram", "0")      # llama-server's host prompt cache off for every calibration server


@dataclass(frozen=True)
class Launch:
    """The shared launch flags: what every `--model` runs with, and a menu entry's defaults."""
    ctx: int | None = None
    parallel: int = DEFAULT_PARALLEL
    no_kvu: bool = False
    s1_slots: int = DEFAULT_S1_SLOTS
    llama_server: str = DEFAULT_LLAMA_SERVER
    user_delimiter: str | None = None
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Entry:
    """One model of the menu, with every launch setting resolved (the entry's own, else the shared flags)."""
    name: str
    model: str
    mmproj: str | None = None
    ctx: int | None = None
    parallel: int = DEFAULT_PARALLEL
    no_kvu: bool = False
    s1_slots: int = DEFAULT_S1_SLOTS
    llama_server: str = DEFAULT_LLAMA_SERVER
    args: tuple[str, ...] = ()
    user_delimiter: str | None = None
    think_sampling: str | None = None
    position_bias: str | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["args"] = list(self.args)
        return d

    def serve_argv(self, *, cache: bool, cache_dir: Path) -> list[str]:
        """The `callo serve` command line this entry runs in-process: launch mode, port 0 on 127.0.0.1,
        `--think-budget 0`, no `--temperature`, the run's `--cache` or `--no-cache` (always explicit, so the server matches
        the run whatever serve's default), the entry's flags, then after `--` llama-server's
        `--cache-ram 0` (the host prompt cache off, so phase B's prefill is a served cold question's;
        recorded in the profile's `server_args` and stamped as `environment.server.cache_ram: 0`) and the entry's
        `args` (last wins, so an explicit `--cache-ram` in `args` is honoured and visible in `server_args`).

        The string-valued flags are written `--flag=VALUE`: a value that starts with `-` (a `user_delimiter` of `-x`)
        or is `--` itself then parses as that flag's value, never as another flag or the llama-server separator."""
        argv = ["serve", f"--model={self.model}", "--port", "0", "--think-budget", "0", "--parallel", str(self.parallel),
                "--s1-slots", str(self.s1_slots), f"--llama-server={self.llama_server}", f"--cache-dir={cache_dir}"]
        if self.ctx is not None:
            argv += ["--ctx", str(self.ctx)]
        if self.no_kvu:
            argv.append("--no-kvu")
        if self.mmproj:
            argv.append(f"--mmproj={self.mmproj}")
        if self.user_delimiter:
            argv.append(f"--user-delimiter={self.user_delimiter}")
        if self.position_bias:
            argv.append(f"--position-bias={self.position_bias}")
        argv.append("--cache" if cache else "--no-cache")      # explicit: serve's default is on since 0.3.0
        argv += ["--", *CACHE_RAM_OFF, *self.args]
        return argv


def _flag(a: str) -> str:
    return a.split("=", 1)[0]


def check_args(args: Sequence[str], *, where: str = "args") -> list[str]:
    """The `args` policy. Returns the sampler flags found (allowed: the caller warns that the pinned
    sampler overrides them); a trace-constraining flag is InvalidInput (exit 2)."""
    refused = [a for a in args if _flag(a) in REFUSED_ARGS]
    if refused:
        raise InvalidInput(f"{where}: {', '.join(refused)} constrain the trace and cannot be overridden per request; "
                           f"callo pins the sampler itself, so remove them")
    return [a for a in args if _flag(a) in SAMPLER_ARGS or _flag(a).startswith(SAMPLER_PREFIXES)]


def _looks_like_flag(a: str) -> bool:
    return len(a) > 1 and a[0] == "-" and (a[1].isalpha() or a[1] == "-")


def sampler_warning(entry: Entry) -> str | None:
    """The warning for the sampler flags among an entry's `args`, each shown with its value (`--temp 0.7`, as
    `--dry-multiplier=0.8` already is): the token after a space-separated flag, unless that token is itself a flag."""
    found = check_args(entry.args, where=f"{entry.name}: args")
    if not found:
        return None
    shown, args = [], entry.args
    for i, a in enumerate(args):
        if a not in found:
            continue
        has_value = "=" not in a and i + 1 < len(args) and not _looks_like_flag(args[i + 1])
        shown.append(f"{a} {args[i + 1]}" if has_value else a)
    return (f"WARNING: {entry.name}: args {' '.join(shown)} are sampler flags; callo's pinned samplers override "
            f"them on every generation and every readout pass, so they do not change this model's measurement")


def _check(ok: bool, where: str, key: str, what: str, value: Any) -> None:
    if not ok:
        raise InvalidInput(f"{where}: {key} must be {what}, got {value!r}")


# callo serve's launch rules (cli.py `--s1-slots must be >= 1`, `kv_flag_error`, slots.SlotRouter), shared by a menu
# entry (`entry_from_dict`) and the shared launch flags (`launch_from_args`)
def _positive_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def _ctx_ok(ctx: Any) -> bool:
    return ctx is None or _positive_int(ctx)


def _s1_slots_ok(s1_slots: Any, parallel: int) -> bool:
    """Below parallel only when parallel > 1: callo serve allows `--s1-slots 1 --parallel 1`."""
    return _positive_int(s1_slots) and (parallel == 1 or s1_slots < parallel)


def entry_from_dict(d: Any, launch: Launch, *, where: str) -> Entry:
    """One `MENU.json` entry: the known keys only, `name` unique-by-caller and `[A-Za-z0-9._-]{1,64}`, `model`
    required; the shared launch flags fill what the entry leaves out.

    `null` has two meanings by key, on purpose (a MENU.json may write `"mmproj": null`): on the nullable keys
    (`mmproj`, `ctx`, `user_delimiter`, `position_bias`) it means "none for this model", overriding a shared `--ctx` /
    `--user-delimiter`; the keys that always carry a value (`name`, `model`, `parallel`, `no_kvu`, `s1_slots`,
    `llama_server`, `args`, `think_sampling`) refuse it. Leaving any key out inherits the shared value."""
    if not isinstance(d, dict):
        raise InvalidInput(f"{where}: each model is an object with at least name and model")
    unknown = sorted(set(d) - set(MENU_KEYS))
    if unknown:
        raise InvalidInput(f"{where}: unknown key(s) {', '.join(unknown)}; known keys: {', '.join(MENU_KEYS)}")
    name = d.get("name")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name in (".", ".."):
        raise InvalidInput(f"{where}: name must match [A-Za-z0-9._-]{{1,64}} and not be . or .. (it names files and "
                           f"directories), got {name!r}")
    where = f"{where} ({name})"
    model = d.get("model")
    _check(isinstance(model, str) and bool(model), where, "model", "a GGUF path", model)
    ctx = d.get("ctx", launch.ctx)                       # nullable: an explicit null is "no ctx for this model"
    _check(_ctx_ok(ctx), where, "ctx", "a positive integer or null", ctx)
    parallel = d.get("parallel", launch.parallel)
    _check(_positive_int(parallel), where, "parallel", "an integer >= 1", parallel)
    no_kvu = d.get("no_kvu", launch.no_kvu)
    _check(isinstance(no_kvu, bool), where, "no_kvu", "true or false", no_kvu)
    s1_slots = d.get("s1_slots", launch.s1_slots)
    _check(_s1_slots_ok(s1_slots, parallel), where, "s1_slots",
           f"an integer >= 1 and below parallel ({parallel}) when parallel > 1, as for callo serve", s1_slots)
    llama_server = d.get("llama_server", launch.llama_server)
    _check(isinstance(llama_server, str) and bool(llama_server), where, "llama_server", "an executable path",
           llama_server)
    args = d.get("args", list(launch.args))
    _check(isinstance(args, list) and all(isinstance(a, str) for a in args), where, "args", "a list of strings", args)
    for key in ("mmproj", "user_delimiter", "position_bias"):          # nullable: null is "none for this model"
        v = d.get(key)
        _check(v is None or (isinstance(v, str) and bool(v)), where, key, "a non-empty string or null", v)
    if "think_sampling" in d:                                          # always carries a value when given
        _check(isinstance(d["think_sampling"], str) and bool(d["think_sampling"]), where, "think_sampling",
               "a non-empty string", d["think_sampling"])
    user_delimiter = d.get("user_delimiter", launch.user_delimiter)
    if no_kvu and ctx is None:
        raise InvalidInput(f"{where}: no_kvu needs ctx (the per-slot context to reserve), as `callo serve --no-kvu` does")
    if d.get("think_sampling") is not None:
        try:
            parse_sampling(d["think_sampling"])
        except InvalidInput as e:
            raise InvalidInput(f"{where}: think_sampling: {e}") from None
    check_args(args, where=f"{where}: args")
    return Entry(name, model, d.get("mmproj"), ctx, parallel, no_kvu, s1_slots, llama_server, tuple(args),
                 user_delimiter, d.get("think_sampling"), d.get("position_bias"))


def load_menu(path: str | Path, launch: Launch) -> list[Entry]:
    """`--menu MENU.json`: `{"models": [entry, ...]}`, at least one entry, names unique. JSON, not TOML: callo
    needs Python >= 3.10, which has no `tomllib`."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise InvalidInput(f"--menu {path}: cannot read it ({e.strerror or e})") from None
    except ValueError as e:
        raise InvalidInput(f"--menu {path}: not JSON ({e})") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), list) or not doc["models"]:
        raise InvalidInput(f"--menu {path}: expected an object with a non-empty \"models\" list")
    extra = sorted(set(doc) - {"models"})
    if extra:
        raise InvalidInput(f"--menu {path}: unknown top-level key(s) {', '.join(extra)}; only \"models\" is read")
    entries = [entry_from_dict(d, launch, where=f"--menu {path}: models[{i}]") for i, d in enumerate(doc["models"])]
    seen: dict[str, int] = {}
    for i, e in enumerate(entries):
        if e.name in seen:
            raise InvalidInput(f"--menu {path}: models[{seen[e.name]}] and models[{i}] share the name {e.name!r}; names "
                               f"are the profile files' names and must be unique")
        seen[e.name] = i
    return entries


def _stem_name(path: str) -> str:
    """The GGUF's stem by one rule on every Python: the name without its last `.suffix` (a dot that is neither the
    first nor the last character). `Path.stem` changed in 3.14 (`..gguf` keeps its whole name there, `.` before)."""
    name = Path(path).name
    i = name.rfind(".")
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", (name[:i] if 0 < i < len(name) - 1 else name) or "model")[:64]
    return "_" * len(stem) if stem in (".", "..") else stem


def entries_from_models(models: Sequence[str], launch: Launch) -> list[Entry]:
    """`--model GGUF` (repeatable) with the shared launch flags: `<name>` is the GGUF stem (a character the
    name grammar refuses becomes `_`, and `.` / `..` become underscores), a name already taken gets `-2`, `-3`, ...
    (the first free one, so a generated name never collides with another stem; still at most 64 characters)."""
    if not models:
        raise InvalidInput("--model needs at least one GGUF")
    check_args(launch.args, where="arguments after --")
    used: set[str] = set()
    out = []
    for m in models:
        name = stem = _stem_name(m)
        n = 1
        while name in used:
            n += 1
            name = f"{stem[:64 - len(str(n)) - 1]}-{n}"
        used.add(name)
        out.append(Entry(name, m, None, launch.ctx, launch.parallel, launch.no_kvu, launch.s1_slots, launch.llama_server,
                         tuple(launch.args), launch.user_delimiter, None, None))
    return out


def menu_sha256(entries: Sequence[Entry]) -> str:
    """`run.json`'s `menu_sha256`: over the resolved entries, so an appended candidate or a changed entry changes
    it (a resume records the new value; only the run-wide settings must match)."""
    canon = json.dumps([e.as_dict() for e in entries], sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


# ---- the GPU rule and the memory precheck -----------------------------------------------------

# The GPU rule (a port of bench/lib.gpu_class: src cannot import bench): llama.cpp's programs ALWAYS hold the GPU;
# oMLX, MLX and ollama's servers hold it only while they hold a MODEL (resident >= RESIDENT_MIN_BYTES): the oMLX
# menu-bar app (about 90 MB) and an idle `ollama serve` (about 70 MB) are commonly left running on a Mac, and a
# plain name match would refuse every run (so a server counts only while it holds a model:
# `ollama ps` empty / oMLX with no loaded model is idle). A llama.cpp name reached through a Python interpreter is a
# script (the test suite's fake llama-server), never a GPU server.
LLAMA_PROGRAMS = frozenset({"llama-server", "llama-cli", "llama-imatrix", "llama-perplexity", "llama-bench",
                            "llama-quantize", "llama-mtmd-cli"})
RESIDENT_ROOTS = frozenset({"omlx", "mlx_lm", "mlx_vlm"})   # the program name, lowercased, up to its first "." or "-":
#                                                            omlx, oMLX, omlx-cli, mlx_lm.server, `-m mlx_lm server`
OLLAMA_SERVERS = ("serve", "runner")                        # `ollama serve` and its model-holding `ollama runner`s
RESIDENT_MIN_BYTES = 2 * 1024**3
_PY_OPTS_WITH_ARG = frozenset({"-X", "-W", "-Q"})           # interpreter options that take a separate argument
_SHELLS = frozenset({"sh", "bash", "zsh", "dash"})
_FOOTPRINT = re.compile(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB)\b")
_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
ResidentFn = Callable[[int], "int | None"]
DEFAULT_HEADROOM_GIB = 2.0
MACOS_GPU_SHARE = 0.75                                     # Metal's default working-set limit, as a share of memory
_MIB = re.compile(r"(\d+)\s*MiB")


PS_ARGV = ("ps", "-axo", "pid=,args=")
PS_TIMEOUT_S = 30
PS_FAILED = "could not list processes"
GPU_REMEDY = "stop any model server, or pass --allow-gpu-busy"


def ps_listing(run: Callable[..., Any] = subprocess.run) -> tuple[str | None, str | None]:
    """(`ps -axo pid=,args=`'s text, None), or (None, why) when ps is missing, errors, times out or lists nothing: the
    GPU rule then fails closed, as the bench's does (`bench/lib.py` lets ps's exception propagate). Never `comm=`
    before another column: macOS cuts it to 16 characters, so `/opt/homebrew/opt/ollama/bin/ollama` reads
    `/opt/homebrew/op`."""
    try:
        p = run(list(PS_ARGV), capture_output=True, text=True, timeout=PS_TIMEOUT_S, check=False)
    except FileNotFoundError:
        return None, "ps not found"
    except subprocess.TimeoutExpired:
        return None, f"timed out after {PS_TIMEOUT_S} s"
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e) or type(e).__name__
    code, out = getattr(p, "returncode", 0), getattr(p, "stdout", None)
    if code:
        err = getattr(p, "stderr", None)
        first = err.strip().splitlines()[0] if isinstance(err, str) and err.strip() else ""
        return None, f"exit {code}" + (f": {first}" if first else "")
    if not isinstance(out, str) or not out.strip():
        return None, "it listed no processes"
    return out, None


def _is_python(argv: Sequence[str]) -> bool:
    """argv runs a Python interpreter: its executable's basename starts with `python`, any case."""
    return bool(argv) and argv[0].rsplit("/", 1)[-1].lower().startswith("python")


def _program(argv: Sequence[str]) -> tuple[str, list[str]]:
    """(the program a command line runs, its own arguments): the executable's basename; under a Python interpreter,
    after its own options, the `-m` module or the script's basename; under a shell the script's basename. File
    arguments (`tail -f omlx.log`, `vim llama-server.log`) never count. A program path with spaces is not recognised
    (`ps args=` cannot tell them from the argument separators; the bench shares the limit). A script run by a shell
    given options (`bash -x /opt/homebrew/bin/omlx start`) is not recognised either (the bench's limit too)."""
    if not argv:
        return "", []
    exe = argv[0].rsplit("/", 1)[-1]
    rest = list(argv[1:])
    if _is_python(argv):
        i = 0
        while i < len(rest) and rest[i].startswith("-"):
            if rest[i] == "-m":
                return (rest[i + 1], rest[i + 2:]) if i + 1 < len(rest) else ("", [])
            if rest[i] in ("-c", "-"):
                return "", []
            i += 2 if rest[i] in _PY_OPTS_WITH_ARG else 1
        return (rest[i].rsplit("/", 1)[-1], rest[i + 1:]) if i < len(rest) else ("", [])
    if exe in _SHELLS and rest and not rest[0].startswith("-"):
        return rest[0].rsplit("/", 1)[-1], rest[1:]
    return exe, rest


def gpu_class(argv: Sequence[str]) -> str | None:
    """"always" for a llama.cpp program (a native binary, by its basename, any path), "resident" for oMLX / MLX /
    an ollama server (they count only while resident >= RESIDENT_MIN_BYTES), None for anything else."""
    name, args = _program(argv)
    if name.lower() in LLAMA_PROGRAMS:
        return None if _is_python(argv) else "always"
    root = re.split(r"[.\-]", name.lower(), maxsplit=1)[0]
    if root in RESIDENT_ROOTS:
        return "resident"
    if root == "ollama" and args[:1] and args[0] in OLLAMA_SERVERS:
        return "resident"
    return None


def footprint_bytes(text: str, pid: int) -> int | None:
    """The `Footprint:` on pid's own header line of `footprint -p PID` (`oMLX [5922]: 64-bit    Footprint: 135 MB
    (16384 bytes per page)`), never a summary line."""
    for line in text.splitlines():
        if f"[{pid}]:" in line:
            m = _FOOTPRINT.search(line)
            if m:
                return int(float(m.group(1)) * _UNITS[m.group(2)])
    return None


def process_resident_bytes(pid: int, run: Callable[..., Any] = subprocess.run) -> int | None:
    """macOS `footprint -p PID` (its own line), else RSS from `ps -o rss= -p PID`; None when neither answers (the
    process is gone, or both tools failed): unknown counts as holding a model."""
    try:
        out = run(["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=60, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    b = footprint_bytes(out if isinstance(out, str) else "", pid)
    if b is not None:
        return b
    try:
        rss = run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rss = rss.strip() if isinstance(rss, str) else ""
    return int(rss) * 1024 if rss.isdigit() else None


def model_servers(ps_text: str, *, exclude: Collection[int] = (), resident_bytes: ResidentFn | None = None,
                  min_resident: int = RESIDENT_MIN_BYTES) -> list[tuple[int, str]]:
    """(pid, command line) of every process from `ps -axo pid=,args=` that holds the GPU: every llama.cpp
    program, and every oMLX / MLX / ollama-server process whose residency (`resident_bytes(pid)`, by default
    `process_resident_bytes`) is >= `min_resident` or cannot be measured. Detection only: the driver never signals a
    process it did not start. `exclude`: the driver's own pid and its own launched child."""
    measure = resident_bytes or process_resident_bytes
    out = []
    for line in ps_text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid in exclude or pid == os.getpid():
            continue
        kind = gpu_class(cmd.split())
        if kind == "always":
            out.append((pid, cmd[:120]))
        elif kind == "resident":
            b = measure(pid)
            if b is None:
                out.append((pid, cmd[:120] + " [resident: unknown, counted as holding a model]"))
            elif b >= min_resident:
                out.append((pid, cmd[:120] + f" [resident {b / 1024**3:.1f} GiB]"))
    return out


def orphan_pid_file(cache_dir: Path | None, pid: int, port: int | None) -> str:
    """The pid file `upstream.find_orphans` read `pid` from: `llama-server-<port>.pid`, or the
    `llama-server-starting-<callo pid>.pid` holding `pid` (read only, never removed here)."""
    name = f"llama-server-{port}.pid" if port is not None else "llama-server-starting-<callo pid>.pid"
    if cache_dir is None:
        return f"{name} in callo's cache dir"
    if port is None:
        for f in sorted(Path(cache_dir).glob("llama-server-starting-*.pid")):
            try:
                if f.read_text().strip() == str(pid):
                    return str(f)
            except OSError:
                continue
    return str(Path(cache_dir) / name)


def ps_failed_reason(ps_error: str | None) -> str:
    return f"{PS_FAILED} (ps failed: {ps_error or 'unknown error'})"


def gpu_findings(ps_text: str | None, orphans: Sequence[tuple[int, int | None]], *, exclude: Collection[int] = (),
                 resident_bytes: ResidentFn | None = None, ps_error: str | None = None,
                 cache_dir: Path | None = None) -> tuple[list[str], str | None]:
    """Step 1's verdict before its wording: (one line per other model server (the ps scan) and per orphan from
    `upstream.find_orphans`, the "cannot tell" reason when ps failed (`ps_text` None), else None); the caller adds the
    remedy (`GPU_REMEDY`): the rule fails closed. An orphan's line names its pid file: a recycled pid (an unrelated
    process reusing a dead callo child's) refuses too, and deleting that file is the fix."""
    cannot_tell = None
    if ps_text is None:
        cannot_tell = ps_failed_reason(ps_error)
        found: list[tuple[int, str]] = []
    else:
        found = model_servers(ps_text, exclude=exclude, resident_bytes=resident_bytes)
    lines = [f"pid {pid}: {cmd}" for pid, cmd in found]
    seen = {pid for pid, _ in found}
    lines += [f"pid {pid}: a llama-server left by an earlier callo"
              + (f" (port {port})" if port is not None else " (still loading)")
              + f"; if pid {pid} is not that llama-server, delete {orphan_pid_file(cache_dir, pid, port)}"
              for pid, port in orphans if pid not in seen and pid not in exclude]
    return lines, cannot_tell


def scan(*, cache_dir: Path, run: Callable[..., Any] = subprocess.run, exclude: Collection[int] = (),
         resident_bytes: ResidentFn | None = None) -> list[str]:
    """The GPU rule's lines over this box's `ps` listing (through `run`) and `cache_dir`'s orphans, the "cannot tell"
    line first when ps failed. It only reads the process
    list; `upstream.find_orphans` may remove stale pid files an earlier callo left in `cache_dir` (dead or
    unparseable pids), and nothing else is written."""
    servers, cannot_tell = scan_findings(cache_dir=cache_dir, run=run, exclude=exclude, resident_bytes=resident_bytes)
    return ([f"{cannot_tell}: {GPU_REMEDY}"] if cannot_tell else []) + servers


def scan_findings(*, cache_dir: Path, run: Callable[..., Any] = subprocess.run, exclude: Collection[int] = (),
                  resident_bytes: ResidentFn | None = None) -> tuple[list[str], str | None]:
    """`scan`'s verdict before its wording (`gpu_findings`), for the dry run, which words the remedy itself."""
    text, why = ps_listing(run)
    return gpu_findings(text, find_orphans(cache_dir), exclude=exclude, resident_bytes=resident_bytes, ps_error=why,
                        cache_dir=cache_dir)


def gpu_check_failed_reason(e: BaseException) -> str:
    """A scan that raised (not a failed ps, which `gpu_findings` reports): "cannot tell", so the rule fails closed."""
    return f"GPU check failed ({type(e).__name__}: {e}): cannot tell whether another model server is running"


def gpu_check_failed(e: BaseException) -> str:
    return f"{gpu_check_failed_reason(e)}; {GPU_REMEDY}"


def gpu_busy_message(found: Sequence[str]) -> str:
    failed = [x for x in found if x.startswith(PS_FAILED)]
    servers = [x for x in found if not x.startswith(PS_FAILED)]
    tail = " to measure anyway (every profile is then marked gpu_busy, which compare flags)"
    if failed:
        return f"refusing to calibrate: {failed[0]}{tail}" + (f"; also found: {'; '.join(servers)}" if servers else "")
    return ("refusing to calibrate: another model server is running (" + "; ".join(servers) + "). Stop them (an idle "
            f"ollama serve or oMLX that holds no model is not counted), or pass --allow-gpu-busy{tail}")


def memory_budget(host: Mapping[str, Any] | None, *, gpu_mem_gib: float | None = None, system: str | None = None) -> int | None:
    """`--gpu-mem GiB` when given; else 75 % of memory on macOS, the nvidia-smi total on Linux with a
    GPU (summed over every GPU, one CSV line each: llama.cpp splits a model across them), else the memory; None when
    unknown (an unknown host included: then nothing is skipped)."""
    if gpu_mem_gib is not None:
        return int(gpu_mem_gib * 2**30)
    host = host or {}                                      # an unknown host: no fingerprint, no memory
    system = system or platform.system()
    mem = host.get("memory_bytes")
    if system == "Darwin":
        return int(mem * MACOS_GPU_SHARE) if isinstance(mem, int) else None
    gpu = host.get("gpu")
    if system == "Linux" and isinstance(gpu, str):
        mibs = _MIB.findall(gpu)
        if mibs:
            return sum(int(x) for x in mibs) * 2**20
    return mem if isinstance(mem, int) else None


def file_bytes(path: str | None) -> int | None:
    if not path:
        return 0
    try:
        return os.stat(path).st_size
    except OSError:
        return None


def model_bytes(path: str) -> int | None:
    """A GGUF's size for the precheck and the dry run: every shard of a split GGUF summed (`modelkey.gguf_bytes`, the
    size `callo serve` reports), None when the file cannot be read (the launch reports it). A missing shard of a split
    GGUF is InvalidInput naming it: the precheck fails that model, the dry run says it would fail."""
    try:
        return gguf_bytes(path)
    except OSError:
        return None


def precheck(entry: Entry, budget: int | None, headroom_gib: float) -> str | None:
    """The skip reason `needs ~X GiB, budget Y GiB (--gpu-mem to override)`, or None. A missing file is
    not a skip (the launch reports it as this model's failure), and an unknown budget never skips. A split GGUF counts
    every shard; one with a shard missing is InvalidInput (the caller fails the model). A lower bound: KV and compute
    buffers are not counted."""
    model, mmproj = model_bytes(entry.model), file_bytes(entry.mmproj)
    if model is None or mmproj is None or budget is None:
        return None
    need = model + mmproj + int(headroom_gib * 2**30)
    if need > budget:
        return f"needs ~{need / 2**30:.1f} GiB, budget {budget / 2**30:.1f} GiB (--gpu-mem to override)"
    return None


# ---- run.json -------------------------------------------------------------------------------------------

RUN_FORMAT = 1
STATUSES = ("pending", "running", "done", "failed", "skipped")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunFile:
    """`DIR/run.json`: the menu's sha256, the run-wide settings (what `--resume` must match) and one record per model
    (`status`, `reason`, `started`, `ended`, `profile`), rewritten atomically after every transition."""

    def __init__(self, path: Path, data: dict):
        self.path, self.data = Path(path), data

    @classmethod
    def new(cls, path: Path, settings: ThinkSettings, entries: Sequence[Entry]) -> RunFile:
        data = {"callo_think_run": RUN_FORMAT, "menu_sha256": menu_sha256(entries), "settings": settings.comparable(),
                "created": _now(), "order": [e.name for e in entries], "dropped": [],
                "models": {e.name: {"status": "pending", "reason": None, "started": None, "ended": None, "profile": None}
                           for e in entries}}
        return cls(path, data)

    @classmethod
    def load(cls, path: Path) -> RunFile:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as e:
            raise InvalidInput(f"{path}: cannot read it ({e.strerror or e})") from None
        except ValueError as e:
            raise InvalidInput(f"{path}: not JSON ({e})") from None
        if (not isinstance(data, dict) or data.get("callo_think_run") != RUN_FORMAT
                or not isinstance(data.get("models"), dict) or not isinstance(data.get("settings"), dict)):
            raise InvalidInput(f"{path} is not a callo think run file (callo_think_run {RUN_FORMAT})")
        why = cls._invalid(data)
        if why:
            raise InvalidInput(f"{path}: {why} (a hand-edited or partial run.json; fix it or start a new --out DIR)")
        return cls(path, data)

    @staticmethod
    def _invalid(data: dict) -> str | None:
        models, order, dropped = data["models"], data.get("order"), data.get("dropped", [])
        if (not isinstance(order, list) or not all(isinstance(n, str) for n in order) or len(set(order)) != len(order)
                or set(order) != set(models)):
            return "`order` is not a list of the names in `models`"
        if (not isinstance(dropped, list) or not all(isinstance(n, str) for n in dropped)
                or not set(dropped) <= set(order)):
            return "`dropped` is not a list of names in `order`"
        for name in order:
            rec = models[name]
            if not isinstance(rec, dict) or rec.get("status") not in STATUSES:
                return f"model {name!r} has no valid status (one of {', '.join(STATUSES)})"
        return None

    def settings_diff(self, settings: ThinkSettings) -> list[str]:
        """The run-wide settings that differ from the recorded ones (a resume refuses on any)."""
        old, new = self.data["settings"], settings.comparable()
        return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))

    def merge(self, entries: Sequence[Entry], *, log: Callable[[str], None] | None = None) -> list[str]:
        """`--resume` with the (possibly extended) menu: new names are `pending` (the appended candidate); the
        order is the menu's, then any recorded model the menu no longer names. Such a model stays in run.json as
        history (`dropped`) but is left out of `done_profiles`, `summary_rows` and `summary`, so the report covers
        only this menu; naming it again restores it. Returns the new names."""
        log = log or (lambda m: print(m, file=sys.stderr))
        added = []
        for e in entries:
            if e.name not in self.data["models"]:
                self.data["models"][e.name] = {"status": "pending", "reason": None, "started": None, "ended": None,
                                               "profile": None}
                added.append(e.name)
        names = [e.name for e in entries]
        dropped = [n for n in self.data["order"] if n not in names]
        for n in dropped:
            log(f"not in this menu: {n} (kept in run.json, left out of the report)")
        self.data["order"], self.data["dropped"] = names + dropped, dropped
        self.data["menu_sha256"] = menu_sha256(entries)
        return added

    def names(self) -> list[str]:
        """The models of this menu, in its order: `order` without the `dropped` ones."""
        dropped = set(self.data.get("dropped", []))
        return [n for n in self.data["order"] if n not in dropped]

    def record(self, name: str) -> dict:
        return self.data["models"][name]

    def profile_path(self, name: str) -> Path | None:
        """The model's recorded profile, absolute: as written (`set` stores it absolute), or, for a relative path an
        older run.json wrote (relative to that run's cwd), its file name in DIR (run.json's own directory), where a
        menu's profiles always are."""
        p = (self.data["models"].get(name) or {}).get("profile")
        if not isinstance(p, str) or not p:
            return None
        return Path(p) if Path(p).is_absolute() else Path(os.path.abspath(self.path.parent)) / Path(p).name

    def set(self, name: str, status: str, *, reason: str | None = None, profile: Path | str | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        rec = self.data["models"][name]
        rec["status"], rec["reason"] = status, reason
        if status == "running":
            rec["started"], rec["ended"] = _now(), None
        elif status in ("done", "failed", "skipped"):
            rec["ended"] = _now()
        if profile is not None:
            rec["profile"] = os.path.abspath(profile)   # absolute: a --resume from another cwd still finds it
        self.write()

    def write(self) -> None:
        """Atomic and durable: a pid-unique temp file in the same directory, fsynced, then `os.replace`d over run.json
        (a crash or a power loss leaves the old file or the new one); a failed write removes its temp file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps(self.data, indent=1) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def done_profiles(self) -> list[Path]:
        out = []
        for name in self.names():
            rec = self.data["models"].get(name) or {}
            path = self.profile_path(name)
            if rec.get("status") == "done" and path is not None and path.exists():
                out.append(path)
        return out

    def summary_rows(self) -> list[dict]:
        """One record per model, in menu order, for the report's "Runs" section (the failed and skipped models
        with their reasons)."""
        return [{"name": name, **{k: self.data["models"][name].get(k) for k in ("status", "reason", "started", "ended")},
                 "profile": str(p) if (p := self.profile_path(name)) is not None else None}
                for name in self.names()]

    def summary(self) -> list[str]:
        """One line per model for the terminal and the report's tail."""
        out = []
        for name in self.names():
            rec = self.data["models"][name]
            out.append(f"{name}: {rec['status']}" + (f" ({rec['reason']})" if rec.get("reason") else ""))
        return out


# ---- --dry-run: the precheck verdicts, the worst case and the estimate from an earlier profile -------------

def profile_rates(profile: Any) -> tuple[float, float] | None:
    """(decode tok/s, budget-0 readout s) from a think profile's `environment.run.decode_tok_s_median` and
    `budgets["0"].lat_p50_ms[0]`; None on any other shape (a hand-edited or foreign file), never an exception."""
    def get(d: Any, k: str) -> Any:
        return d.get(k) if isinstance(d, dict) else None
    decode = get(get(get(profile, "environment"), "run"), "decode_tok_s_median")
    lat = get(get(get(profile, "budgets"), "0"), "lat_p50_ms")
    if not (finite_positive(decode) and isinstance(lat, list) and lat and finite_positive(lat[0])):
        return None
    return float(decode), float(lat[0]) / 1000


def rates_profile(model_key: str | None, dirs: Sequence[Path]) -> tuple[dict, Path] | None:
    """A think profile (`callo_think_profile` PROFILE_FORMAT) with this `model_key` and usable rates, for the time
    estimate: DIR first, then each `--rates-from` in turn; within the first directory that has one, the newest
    `created`."""
    if not model_key:
        return None
    for d in dirs:
        best: tuple[str, dict, Path] | None = None
        for f in sorted(Path(d).glob("*.think.json")):
            try:
                p = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (not isinstance(p, dict) or p.get("callo_think_profile") != PROFILE_FORMAT
                    or p.get("model_key") != model_key or profile_rates(p) is None):
                continue
            created = p.get("created") if isinstance(p.get("created"), str) else ""
            if best is None or created > best[0]:
                best = (created, p, f)
        if best is not None:
            return best[1], best[2]
    return None


def _estimate(rates: tuple[float, float], settings: ThinkSettings, n_items: int) -> float:
    decode, read_s = rates
    positive = list(settings.positive)
    gen_s = settings.bmax / decode
    per_q = read_s + (gen_s + len(positive) * read_s if positive else 0.0)
    audit = sum(settings.audit * (b / decode + read_s) for b in positive[:-1])
    drift = DRIFT_N * gen_s if positive else 0.0
    return (n_items + min(settings.warmup, n_items)) * per_q + audit + drift


def estimate_s(profile: Any, settings: ThinkSettings, n_items: int) -> float | None:
    """The run's time from an earlier profile's measured decode rate and budget-0 readout time: each question
    reads once at budget 0 (phase A), generates Bmax tokens once and reads once per positive budget (phase B); the
    audit regenerates `--audit` questions per cut budget and the drift check 3 traces at Bmax; the warm-up questions
    (at most `n_items`, as the worst case counts them) cost a question each. None without usable rates, never an
    exception."""
    rates = profile_rates(profile)
    if not rates:
        return None
    est = _estimate(rates, settings, n_items)
    return est if math.isfinite(est) else None


def _hms(s: float) -> str:
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def dry_run_model_lines(entry: Entry, settings: ThinkSettings, n_items: int, *, budget: int | None, headroom_gib: float,
                        rates_dirs: Sequence[Path], cache_dir: Path | None) -> list[str]:
    """Per model: the file size and the precheck verdict, the worst case, and the estimate only when a profile
    with the same `model_key` exists in DIR or `--rates-from DIR` (otherwise "unknown until measured")."""
    try:
        size: int | None = model_bytes(entry.model)
        shard_error = None
    except InvalidInput as e:
        size, shard_error = None, str(e)
    mm = file_bytes(entry.mmproj) if entry.mmproj else 0
    if shard_error is not None:
        head = f"{entry.name}: would fail at the precheck: {shard_error}"
    elif size is None:
        head = f"{entry.name}: {entry.model}: NOT FOUND (the launch would fail; the run goes on to the next model)"
    elif mm is None:
        head = f"{entry.name}: {entry.model} ({size / 2**30:.1f} GiB); would fail: mmproj not found: {entry.mmproj}"
    else:
        gib = (size + (mm or 0)) / 2**30
        skip = precheck(entry, budget, headroom_gib)
        verdict = f"would be skipped: {skip}" if skip else (
            f"fits the budget of {budget / 2**30:.1f} GiB" if budget is not None else "memory budget unknown: not prechecked")
        head = f"{entry.name}: {entry.model} ({gib:.1f} GiB" + (" with mmproj" if entry.mmproj else "") + f"); {verdict}"
    positive = list(settings.positive)
    worst = n_items * settings.bmax
    audit = settings.audit * sum(positive[:-1])
    warm = min(settings.warmup, n_items) * settings.bmax if positive else 0     # as calibrate's dry run: Bmax too
    lines = [head, f"  worst case: {n_items} x {settings.bmax} = {worst} generated tokens, plus up to {audit} for the audit, "
                   f"{DRIFT_N * settings.bmax} for the drift check and {warm} for the warm-up"]
    key = None
    if size is not None and shard_error is None:
        try:
            key = load_model_key(entry.model, cache_dir)
        except (InvalidInput, OSError, ValueError):
            key = None
    found = rates_profile(key, rates_dirs)
    rates = profile_rates(found[0]) if found else None
    est = estimate_s(found[0], settings, n_items) if found else None       # None on a non-finite estimate, too
    if found and rates and est is not None:
        lines.append(f"  time: ~{_hms(est)} from {found[1]} (decode {rates[0]:.0f} "
                     f"tok/s, budget-0 readout {rates[1]:.2f} s)")
    else:
        lines.append("  time: unknown until measured (no profile with this model_key in the out dir or --rates-from; the "
                     "live ETA prints after 5 questions)")
    return lines


# ---- `callo serve` in-process per model: logs, signals, settle ----------------------------

DEFAULT_SETTLE_S = 30.0
SETTLE_MAX_S = 86400.0                  # `--settle` at most a day: far larger never ends (`left -= 1.0` stalls above 2**53)
COOLDOWN_MAX_S = 600.0                  # `--cooldown therm`: at most 10 min for CPU_Speed_Limit to reach 100
COOLDOWN_POLL_S = 5.0
LOG_FORMAT = "%(asctime)s callo %(levelname)s %(message)s"


class MenuFailure(CalloError):
    """One model's `callo serve` did not start: that model is `failed`, the run goes on."""


class _ThreadFilter(logging.Filter):
    """Pass only the records logged from `thread` (`invert`: only the others). everything serve logs, from ANY
    of its threads (the worker, `callo-child-watch`'s "llama-server exited", the request handlers), goes to the
    model's own file; only the driver's thread reaches the terminal, so the two never interleave."""

    def __init__(self, thread: threading.Thread, *, invert: bool = False):
        super().__init__()
        self.thread, self.invert = thread, invert

    def filter(self, record: logging.LogRecord) -> bool:
        same = record.thread == self.thread.ident
        return (not same) if self.invert else same


_ATTEMPT_RE = re.compile(r"---- \S+ callo serve \S+ ----")


def attempt_separator(name: str) -> str:
    """The line that opens each attempt in `<name>.serve.log` (appended to across attempts and `--resume`)."""
    return f"---- {_now()} callo serve {name} ----"


def last_error_line(log_path: Path, limit: int = 400) -> str:
    """serve's last ERROR record (its refusal or the load failure, with the continuation lines a multi-line message
    such as llama-server's log tail adds, joined by ` | `), else the log's last line, for `failed: <reason>`. Only the
    LAST attempt counts: the lines after the last `attempt_separator`, so an earlier attempt's error never shows."""
    try:
        lines = [ln.rstrip() for ln in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    except OSError:
        return f"no log at {log_path}"
    if not lines:
        return f"empty log {log_path}"
    seps = [i for i, ln in enumerate(lines) if _ATTEMPT_RE.fullmatch(ln)]
    if seps:
        lines = lines[seps[-1] + 1:]
        if not lines:
            return f"nothing logged by this attempt ({log_path})"
    starts = [i for i, ln in enumerate(lines) if " ERROR " in ln]
    if not starts:
        return lines[-1]
    record = lines[starts[-1]:]
    record[0] = record[0].split(" ERROR ", 1)[1]
    text = " | ".join(record)
    return text if len(text) <= limit else text[:limit - 3] + "..."


@dataclass
class Served:
    url: str
    port: int
    log_path: Path
    thread: threading.Thread
    child: int | None = None                # the llama-server pid serve launched, from OUR starting pid file only


CHILD_GRACE_S = 10.0
SERVE_JOIN_S = 60.0                     # how long the cleanup waits for the serve thread before stopping its child
START_POLL_S = 0.02                     # how often the starting pid file is read while serve starts


@contextlib.contextmanager
def served_model(entry: Entry, *, cache: bool, cache_dir: Path, log_path: Path, stop: threading.Event,
                 timeout: float = SERVE_START_TIMEOUT_S, kill: Callable[[int, int], Any] = os.killpg,
                 log: Callable[[str], None] | None = None) -> Iterator[Served]:
    """`callo serve` for `entry` in a worker thread (port 0, `--think-budget 0`, the run's `--cache`, the
    entry's flags, llama-server's `--cache-ram 0`), its log in `log_path` through thread-filtered handlers on the
    `callo` logger (propagation off for the duration: every record not from the driver's thread goes to the file, the
    driver's own still reach stderr); each attempt opens with an `attempt_separator` line.

    The launched llama-server's pid (`Served.child`) is read ONLY from `llama-server-starting-<our pid>.pid`, on every
    pass while serve starts, once more before the cleanup join and, when the serve thread outlives the join, once more
    after it (a child launched during the join); no other pid file is ever read for a pid to signal,
    so nothing this process did not launch is ever signalled. Before the launch an existing starting file of ours is
    removed when its pid is dead, and is a MenuFailure when it is alive (an earlier child of this run survived).

    A start that fails is a MenuFailure with serve's last error line. On exit the serve is stopped and joined; serve's
    own `finally` stops (and reaps) a child it launched. Only when the serve thread outlives the join (a load that has
    not returned) is the remembered child stopped here: `stop_child` SIGTERMs its process group, waits, SIGKILLs and
    waits again; once the thread has returned, a live pid may be a reused one and is left alone. The
    logger is restored even when a second Ctrl-C (StopHandler's KeyboardInterrupt) lands during that cleanup; the
    interrupt then propagates. A serve thread that outlives the join keeps logging after the restore: its records then
    reach the terminal, or the next model's log (whose file takes every non-driver thread's records); accepted, as
    such a thread is rare and the separator marks where each attempt starts."""
    from . import cli  # cli imports this module

    args_list, extra = cli.split_passthrough(entry.serve_argv(cache=cache, cache_dir=cache_dir))
    said = io.StringIO()
    try:
        with contextlib.redirect_stderr(said):            # argparse prints its usage and error, then exits
            args = cli.build_parser().parse_args(args_list)
    except SystemExit:
        lines = [ln for ln in said.getvalue().splitlines() if ln.strip()]
        why = lines[-1] if lines else "callo serve refused its arguments"
        raise MenuFailure(f"callo serve did not accept this entry's flags: {why}") from None
    args.extra = extra
    kv_error = cli.kv_flag_error(args)
    if kv_error:
        raise MenuFailure(kv_error)
    starting = Path(cache_dir) / f"llama-server-starting-{os.getpid()}.pid"
    stale = loading_child_pid(cache_dir)
    if stale is not None and alive(stale):                # the pid may be reused by an unrelated process: never "kill it"
        raise MenuFailure(f"{starting} names pid {stale}, which is alive: if it is a llama-server of an earlier callo "
                          f"run, stop it; otherwise delete {starting}")
    starting.unlink(missing_ok=True)                  # a dead (or unreadable) leftover of ours: never read as this child
    ready = threading.Event()
    port: list[int] = []
    rc: list[int] = []

    def run() -> None:
        try:
            rc.append(cli.serve(args, ready=lambda p: (port.append(p), ready.set()), stop=stop))
        finally:
            ready.set()

    thread = threading.Thread(target=run, name=f"callo-menu-{entry.name}", daemon=True)
    driver = threading.current_thread()
    logger = logging.getLogger("callo")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(attempt_separator(entry.name) + "\n")
    to_file = logging.FileHandler(log_path, encoding="utf-8")
    to_file.setFormatter(logging.Formatter(LOG_FORMAT))
    to_file.addFilter(_ThreadFilter(driver, invert=True))       # every thread of serve's, not only the worker
    to_terminal = logging.StreamHandler(sys.stderr)
    to_terminal.setFormatter(logging.Formatter(LOG_FORMAT))
    to_terminal.addFilter(_ThreadFilter(driver))
    say = log or (lambda line: print(line, file=sys.stderr, flush=True))
    child: int | None = None

    def remember() -> None:
        nonlocal child
        pid = loading_child_pid(cache_dir)
        if pid is not None:
            child = pid

    was_propagate, was_level = logger.propagate, logger.level
    logger.addHandler(to_file)
    logger.addHandler(to_terminal)
    logger.propagate = False
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    try:
        thread.start()
        deadline = time.monotonic() + timeout
        while True:                                   # remember the loading child's pid while its pid file exists
            remember()
            if ready.wait(START_POLL_S) or time.monotonic() >= deadline:
                break
        remember()
        if not port:
            if rc:
                raise MenuFailure(f"callo serve did not start (exit {rc[0]}): {last_error_line(log_path)}")
            raise MenuFailure(f"callo serve did not start within {timeout:.0f} s (its log: {log_path})")
        yield Served(f"http://127.0.0.1:{port[0]}", port[0], log_path, thread, child)
    finally:
        try:
            stop.set()
            remember()
            thread.join(timeout=SERVE_JOIN_S)
            if thread.is_alive():
                remember()                            # a child launched during the join: still only OUR starting file
                if child is not None:
                    what = stop_child(child, kill=kill, grace=CHILD_GRACE_S)
                    say(f"{entry.name}: the serve thread is still running and llama-server pid {child} {what}")
                else:
                    say(f"{entry.name}: the serve thread is still running; it had launched no llama-server")
        finally:                                      # a KeyboardInterrupt above must not leak the handlers
            logger.removeHandler(to_file)
            logger.removeHandler(to_terminal)
            to_file.close()
            logger.propagate = was_propagate
            logger.setLevel(was_level)


def loading_child_pid(cache_dir: Path) -> int | None:
    """The llama-server a `callo serve` of THIS process is still loading: `llama-server-starting-<our pid>.pid` (the
    launcher renames it to `llama-server-<port>.pid` once the server listens). None when there is none, or
    when the file does not hold a pid above 1 (`killpg(0)` would signal our own group, a negative pid every process)."""
    try:
        pid = int((Path(cache_dir) / f"llama-server-starting-{os.getpid()}.pid").read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 1 else None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_child(cache_dir: Path, *, kill: Callable[[int, int], Any] = os.killpg) -> int | None:
    """SIGTERM the process group of a llama-server this process is still LOADING (it runs in its own session, so the
    group is the child's pid): a serve blocked in `proc.start()` cannot stop it before the load returns, while a
    serving child is stopped by serve's own `finally` once `stop` is set. The pid, or None when nothing is loading."""
    pid = loading_child_pid(cache_dir)
    if pid is None:
        return None
    with contextlib.suppress(ProcessLookupError, PermissionError):
        kill(pid, signal.SIGTERM)
    return pid


def stop_child(pid: int | None, *, kill: Callable[[int, int], Any] = os.killpg, grace: float = CHILD_GRACE_S,
               sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
               is_alive: Callable[[int], bool] = alive) -> str:
    """Stop a launched llama-server that serve did not: SIGTERM its process group, wait `grace`, SIGKILL, wait again.
    Returns what happened ("had exited", "exited on SIGTERM", "exited on SIGKILL", "is still alive after SIGKILL")."""
    if pid is None or not is_alive(pid):
        return "had exited"
    for sig, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            kill(pid, sig)
        if wait_exit(pid, timeout=grace, sleep=sleep, clock=clock, is_alive=is_alive):
            return f"exited on {name}"
    return "is still alive after SIGKILL"


def _never() -> bool:
    return False


def wait_exit(pid: int | None, *, timeout: float, sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic, is_alive: Callable[[int], bool] = alive,
              should_stop: Callable[[], bool] = _never) -> bool:
    """True once `pid` has exited (or there is none), False at the timeout or as soon as `should_stop()` is (the
    driver's first Ctrl-C: the post-model wait is cut short as the settle is)."""
    if pid is None:
        return True
    deadline = clock() + timeout
    while is_alive(pid):
        if should_stop() or clock() >= deadline:
            return False
        sleep(0.2)
    return True


SETTLE_SLICE_S = 1.0                    # the settle and the cooldown sleep in slices this long, checking `should_stop`
SETTLE_CUT = "settle cut short: stopping"


def _sleep_sliced(seconds: float, *, sleep: Callable[[float], None], should_stop: Callable[[], bool]) -> bool:
    """Sleep `seconds` in slices of at most SETTLE_SLICE_S; True as soon as `should_stop()` is (before each slice)."""
    left = seconds
    while left > 0:
        if should_stop():
            return True
        step = min(SETTLE_SLICE_S, left)
        sleep(step)
        left -= step
    return should_stop()


def settle(*, seconds: float, cooldown: str | None, log: Callable[[str], None], sleep: Callable[[float], None] = time.sleep,
           therm: Callable[[], int | None] = therm_limit, clock: Callable[[], float] = time.monotonic,
           should_stop: Callable[[], bool] = _never) -> None:
    """`--settle` seconds between models, or `--cooldown therm`: wait until `pmset -g therm` reports
    CPU_Speed_Limit = 100, at most 10 min; without thermal data it settles for `seconds` instead. Either wait sleeps in
    slices of at most 1 s and ends early, logging "settle cut short: stopping", once `should_stop()` is true (the
    driver's first Ctrl-C: `time.sleep` resumes after a signal handler, PEP 475)."""
    if should_stop():
        log(SETTLE_CUT)
        return
    if cooldown == "therm":
        deadline = clock() + COOLDOWN_MAX_S
        while True:
            limit = therm()
            if limit is None:
                log(f"cooldown therm: no thermal data (pmset -g therm has no CPU_Speed_Limit); settling {seconds:g} s instead")
                break
            if limit >= 100:
                log(f"cooldown therm: CPU_Speed_Limit = {limit}, the next model starts from an idle GPU")
                return
            if clock() >= deadline:
                log(f"cooldown therm: CPU_Speed_Limit still {limit} after {COOLDOWN_MAX_S / 60:.0f} min; going on")
                return
            log(f"cooldown therm: CPU_Speed_Limit = {limit}, waiting")
            if _sleep_sliced(COOLDOWN_POLL_S, sleep=sleep, should_stop=should_stop):
                log(SETTLE_CUT)
                return
    if seconds > 0:
        log(f"settling {seconds:g} s before the next model")
        if _sleep_sliced(seconds, sleep=sleep, should_stop=should_stop):
            log(SETTLE_CUT)


class StopHandler:
    """SIGINT/SIGTERM in the driver's main thread. The first signal interrupts the current model at
    once and stops the run (STOPPING): it sets the run's `interrupted` event and the current model's `stop` (serve then
    stops its llama-server itself, calibrate fails under it, and the driver records the model `failed: interrupted;
    --resume continues from the raw file`; no further model runs) and SIGTERMs the process group of
    a llama-server still LOADING, read from the launcher's pid file, because a serve blocked in `proc.start()` sees
    `stop` only after the load (up to 600 s). A second signal escalates: SIGKILL to the child's group (the loading one,
    else `child`, the serving one the driver registered) and KeyboardInterrupt, so a user is never left without a way
    out. A launched llama-server is always stopped."""

    STOPPING = "stopping now: the current model is interrupted (--resume continues it) and no further model runs"

    def __init__(self, cache_dir: Path, *, kill: Callable[[int, int], Any] = os.killpg, log: Callable[[str], None] = print):
        self.cache_dir, self.kill, self.log = Path(cache_dir), kill, log
        self.interrupted = threading.Event()
        self.current: threading.Event | None = None
        self.child: int | None = None                       # the serving child, set by the driver per model
        self.previous: dict[int, Any] = {}
        self.signals: list[int] = []

    def __call__(self, signum: int, _frame: Any) -> None:
        self.signals.append(signum)
        if self.interrupted.is_set():                       # the second signal: escalate and leave
            pid = loading_child_pid(self.cache_dir) or self.child
            if pid is not None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    self.kill(pid, signal.SIGKILL)
            self.log(f"signal {signum} again: " + (f"SIGKILL to llama-server pid {pid}; " if pid else "") + "exiting now")
            raise KeyboardInterrupt
        self.interrupted.set()
        if self.current is not None:
            self.current.set()
        pid = terminate_child(self.cache_dir, kill=self.kill)
        self.log(f"signal {signum}: {self.STOPPING}" + (f"; SIGTERM to the loading llama-server pid {pid} (serve "
                                                         f"cannot stop it before the load returns)" if pid else ""))

    def install(self) -> bool:
        if threading.current_thread() is not threading.main_thread():
            return False
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.signal(sig, self)
        return True

    def restore(self) -> None:
        for sig, prev in self.previous.items():
            signal.signal(sig, prev)
        self.previous.clear()


# ---- the driver (failure isolation and --resume) -------------------------------------------

CACHE_DIR_NAME = ".callo-cache"           # DIR/.callo-cache: the launched llama-servers' pid files, logs and slot dir
CHILD_EXIT_WAIT_S = 60.0
CACHE_RAM_FLAGS = ("--cache-ram", "-cram")  # llama-server's long and short spellings


@dataclass(frozen=True)
class MenuOptions:
    out_dir: Path
    resume: bool = False
    allow_gpu_busy: bool = False
    headroom_gib: float = DEFAULT_HEADROOM_GIB
    gpu_mem_gib: float | None = None
    settle_s: float = DEFAULT_SETTLE_S
    cooldown: str | None = None
    rates_from: Path | None = None
    launched_by: str = "menu"             # "model" under --model (environment.server.launched_by)


def cache_ram_of(argv: Sequence[str]) -> int | None:
    """The host prompt-cache setting a launch's llama-server ran with: the LAST `--cache-ram` / `-cram` value among
    the llama-server arguments after `--` (`Entry.serve_argv` puts CACHE_RAM_OFF first and the entry's `args` after it,
    and llama-server's last flag wins); None when there is none or its value is not an integer (unknown)."""
    if "--" not in argv:
        return None
    rest = list(argv)[list(argv).index("--") + 1:]
    value: str | None = None
    for i, a in enumerate(rest):
        flag, eq, v = a.partition("=")
        if flag in CACHE_RAM_FLAGS:
            value = v if eq else (rest[i + 1] if i + 1 < len(rest) else None)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def launch_record(entry: Entry) -> dict:
    """`environment.server.launch`: the launch shape the entry's server was measured with, which
    `calibrate_think.serve_command` repeats so the served latency matches the measured one: `ctx`, `parallel`,
    `no_kvu`, `s1_slots`, `user_delimiter`, the entry's own llama-server `args` (not the calibration-only
    `--cache-ram 0`), its `position_bias` file (absolute: the budget-0 readout was debiased by it) and its
    `llama_server` binary (another build measures another latency)."""
    return {"ctx": entry.ctx, "parallel": entry.parallel, "no_kvu": entry.no_kvu, "s1_slots": entry.s1_slots,
            "user_delimiter": entry.user_delimiter, "args": list(entry.args),
            "position_bias": os.path.abspath(entry.position_bias) if entry.position_bias else None,
            "llama_server": entry.llama_server}


def stamp_profile(path: Path, *, launched_by: str, gpu_busy: bool, cache_ram: int | None = 0,
                  entry: Entry | None = None) -> dict:
    """A launch-mode profile records how it was launched, whether another model server was running and
    the host prompt-cache setting its llama-server ran with (`cache_ram` 0: `--cache-ram 0`; an explicit
    `--cache-ram N` in the entry's `args` wins; the one-server attach modes leave it null). With the `entry` it ran,
    also the absolute paths of the GGUF and the mmproj it launched (`environment.model.path`, `mmproj_path`) and its
    launch shape (`environment.server.launch`, `launch_record`), so the report's serve line is runnable as printed
    from any directory. Written after `calibrate_one` (the raw header, which --resume compares,
    stays what the one-server run writes)."""
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    env = p.setdefault("environment", {})
    server = env.setdefault("server", {})
    server["launched_by"], server["gpu_busy"], server["cache_ram"] = launched_by, gpu_busy, cache_ram
    if entry is not None:
        model = env.get("model") if isinstance(env.get("model"), dict) else {}
        env["model"] = {**model, "path": os.path.abspath(entry.model),
                        "mmproj_path": os.path.abspath(entry.mmproj) if entry.mmproj else None}
        server["launch"] = launch_record(entry)
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text(json.dumps(p, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return p


def valid_profile(path: str | None) -> bool:
    if not path or not Path(path).exists():
        return False
    try:
        compare.load_profile(path)
    except InvalidInput:
        return False
    return True


class RunJsonUnwritable(Exception):
    """run.json could not be written (an ENOSPC, a removed DIR): the run cannot record its progress, so it stops;
    `--resume` continues from what the last successful write and the raw files hold. Never a model's own failure."""


class Driver:
    """One menu run: the GPU rule, then per model the precheck, the in-process serve, `calibrate_one`, the profile
    stamp, the stop and the settle, with `run.json` rewritten at every transition. `launcher`, `scan_fn`, `host`,
    `sleep`, `clock` and `therm` are seams for the tests; `launcher` and `scan_fn` default to this module's
    `served_model` and `scan`, looked up at call time (so a test or the CLI's tests can patch `menu.scan` or
    `menu.ps_listing`). The driver iterates the CURRENT menu's entries: a model `--resume` finds in run.json but
    not in the menu is history and never runs (RunFile.merge)."""

    INTERRUPTED = "interrupted; --resume continues from the raw file"

    def __init__(self, settings: ThinkSettings, entries: Sequence[Entry], opts: MenuOptions, *,
                 log: Callable[[str], None], launcher: Callable[..., Any] | None = None,
                 scan_fn: Callable[..., list[str]] | None = None, host: Mapping[str, Any] | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 therm: Callable[[], int | None] = therm_limit):
        self.settings, self.entries, self.opts, self.log = settings, list(entries), opts, log
        self.launcher, self.scan_fn, self.host = launcher, scan_fn, host
        self.sleep, self.clock, self.therm = sleep, clock, therm
        self.cache_dir = Path(opts.out_dir) / CACHE_DIR_NAME
        self.handler = StopHandler(self.cache_dir, log=log)
        self.run_file: RunFile | None = None
        self.report_md: str | None = None
        self.report_refused: str | None = None  # why the joint report was refused (the stub report says it; exit 1)
        self.gpu_busy = False
        self.stage = ""                       # the per-model step in progress, named in a failure's serve-log line
        # the done models `todo` re-entered for their reads left: name -> (what is left, the done record as it was,
        # its profile's bytes as they were)
        self.reentered: dict[str, tuple[str, dict, bytes]] = {}

    def _scan(self, exclude: Collection[int] = ()) -> list[str]:
        return (self.scan_fn or scan)(cache_dir=self.cache_dir, exclude=exclude)

    @staticmethod
    def _write(rf: RunFile) -> None:
        try:
            rf.write()
        except OSError as e:
            raise RunJsonUnwritable(f"{rf.path}: {e}") from e

    @staticmethod
    def _set(rf: RunFile, name: str, status: str, **kw: Any) -> None:
        """`rf.set`, with a failed write raised as RunJsonUnwritable: the run stops, no model handler swallows it."""
        try:
            rf.set(name, status, **kw)
        except OSError as e:
            raise RunJsonUnwritable(f"{rf.path}: {e}") from e

    def _fail(self, entry: Entry, rf: RunFile, reason: str, *, log_path: Path | None = None,
              profile: Path | None = None) -> None:
        """Record `entry` failed, with the traceback of the exception being handled in its serve log when
        `log_path` is given. A model the user interrupted is recorded as interrupted, without a traceback: the
        failure is the stop's consequence (the serve stopped under calibrate), not a fault."""
        if self.handler.interrupted.is_set():
            if self._keep_done(entry, rf, self.INTERRUPTED):
                return
            self._set(rf, entry.name, "failed", reason=self.INTERRUPTED, profile=profile)
            self.log(f"{entry.name}: {self.INTERRUPTED}")
            return
        where = ""
        if log_path is not None:
            with contextlib.suppress(OSError), log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n{entry.name}: {self.stage} failed: {reason}\n{traceback.format_exc()}")
            where = f" (traceback in {log_path})"
        if self._keep_done(entry, rf, reason + where):
            return
        self._set(rf, entry.name, "failed", reason=reason, profile=profile)
        self.log(f"{entry.name}: failed: {reason}{where}")

    def _rewritten(self, name: str, path: Path) -> bool:
        """Whether a re-entered model's profile is no longer the one `todo` found (`calibrate_one` wrote a new one)."""
        prior = self.reentered.get(name)
        try:
            return prior is not None and path.read_bytes() != prior[2]
        except OSError:
            return True

    def _keep_done(self, entry: Entry, rf: RunFile, why: str) -> bool:
        """A model `todo` re-entered from `done` for its reads left whose re-entry did not finish (a launch, a
        precheck, a scan, a failed or refused calibrate, a Ctrl-C, a stamp): its done record comes back as it was, with
        the reason, so the model stays in the report. Before the reads, the profile is untouched and the next --resume
        tries them again (they are still left). After them (a profile `calibrate_one` rewrote but that could not be
        stamped), the profile it had is put back, atomically, and the reason says so. True when it did; False for any
        other model, or when the profile it had cannot be put back or is no longer valid (the model then fails)."""
        prior = self.reentered.get(entry.name)
        path = rf.profile_path(entry.name)
        if prior is None or path is None:
            return False
        what, rec, snapshot = prior
        restored = self._rewritten(entry.name, path)
        if restored:
            tmp = path.with_name(path.name + ".restore.tmp")
            try:
                tmp.write_bytes(snapshot)
                os.replace(tmp, path)
            except OSError as e:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                self.log(f"WARNING: {entry.name}: cannot put back the profile it had in {path} ({e})")
                return False
        if not valid_profile(str(path)):
            return False
        del self.reentered[entry.name]
        gate = what == GATE_READS
        if restored:
            why = "interrupted" if why == self.INTERRUPTED else why     # the raw file is complete: nothing to continue
            note = (f"{'gate reads' if gate else 'reads left'} done, but {why}; the profile they wrote was not stamped, "
                    f"so the one before them is kept{' (no gated points)' if gate else ''}")
        else:
            note = (f"gate reads failed: {why}; profile without gated points kept" if gate
                    else f"reads left failed: {why}; profile kept")
        rf.data["models"][entry.name] = {**rec, "reason": note}
        self._write(rf)
        self.log(f"{entry.name}: done: {note}")
        return True

    # -- setup --

    def open_run(self) -> RunFile:
        out, path = Path(self.opts.out_dir), Path(self.opts.out_dir) / "run.json"
        if out.exists() and any(out.iterdir()) and not self.opts.resume:
            raise Refusal(f"refusing to calibrate: {out} is not empty. Pass --resume to continue the run recorded in "
                          f"{path} (an appended menu entry runs alone), or choose another --out")
        if self.opts.resume and path.exists():
            try:
                rf = RunFile.load(path)
            except InvalidInput as e:
                raise Refusal(f"refusing to resume: {e}") from None
            diff = rf.settings_diff(self.settings)
            if diff:
                hint = cache_resume_hint(rf.data["settings"].get("cache"), self.settings.cache) if "cache" in diff else ""
                raise Refusal(f"refusing to resume: the run-wide settings differ from {path}'s ({', '.join(diff)}); points "
                              f"are comparable only when every model reads the same questions in the same way. Use the "
                              f"recorded settings, or another --out{hint}")
            added = rf.merge(self.entries, log=self.log)
            if added:
                self.log(f"resume: new menu entries {', '.join(added)} are pending; the report is regenerated over every "
                         f"profile")
        else:
            if self.opts.resume:
                self.log(f"resume: no {path}; starting a new run")
            rf = RunFile.new(path, self.settings, self.entries)
        self.run_file = rf
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RunJsonUnwritable(f"{out}: {e}") from e
        self._write(rf)
        return rf

    def first_gpu_check(self) -> None:
        """Before the first model, failing closed: a scan that RAISES (a pid file that
        vanished mid-read, a bug) is "cannot tell", a Refusal (exit 2) before run.json exists, unless
        --allow-gpu-busy, which goes on with every profile marked gpu_busy, as for a failed ps."""
        try:
            self.gpu_check(first=True)
        except Refusal:
            raise
        except Exception as e:  # noqa: BLE001 -- any scan fault refuses, never a traceback
            why = gpu_check_failed(e)
            if not self.opts.allow_gpu_busy:
                raise Refusal(f"refusing to calibrate: {why}") from e
            self.gpu_busy = True
            self.log(f"WARNING: {why}; --allow-gpu-busy: every profile is marked gpu_busy")

    def gpu_check(self, *, first: bool, exclude: Collection[int] = ()) -> list[str] | None:
        """Before each model: the lines naming other model servers (a failed `ps` is one too: the rule
        fails closed); a Refusal before the first model without --allow-gpu-busy (and before `open_run` writes
        anything, so a retry needs no --resume); later models fail individually (the caller decides). `exclude`: the
        driver's own launched child."""
        found = self._scan(exclude)
        if found and not self.opts.allow_gpu_busy:
            if first:
                raise Refusal(gpu_busy_message(found))
            return found
        if found:
            self.gpu_busy = True
            self.log("WARNING: another model server is running (" + "; ".join(found) + "); --allow-gpu-busy: every "
                     "profile is marked gpu_busy")
        return None

    # -- one model --

    def pre_scan(self, entry: Entry, rf: RunFile) -> bool:
        """Before a later model, isolated like the model itself: another model server, or a scan that
        raised (a pid file that vanished mid-read), fails THIS model. True when it may run."""
        self.stage = "pre-scan"
        try:
            found = self.gpu_check(first=False)
        except Exception as e:  # noqa: BLE001 -- this model fails, the run goes on
            self._fail(entry, rf, f"{type(e).__name__}: {e}", log_path=Path(self.opts.out_dir) / f"{entry.name}.serve.log")
            return False
        if found:
            self._fail(entry, rf, "another model server is running (" + "; ".join(found) + ")")
            return False
        return True

    def run_model(self, entry: Entry, rf: RunFile, budget: int | None) -> None:
        """One model, isolated: ANY exception, from the launch to the profile stamp (an ENOSPC, an
        unexpected /callo/stats shape, a bug), fails THIS model with `<Type>: <message>` and its traceback in the
        model's serve log, naming the stage; the run goes on. Only a run.json that cannot be written stops the run
        (RunJsonUnwritable). The registered child (`StopHandler.child`) is cleared as soon as the serve has stopped,
        before the post-scan, and again on every other path, so a second signal never SIGKILLs a child serve already
        reaped (only the loading child, read from our own starting file, or the one still serving)."""
        try:
            skip = precheck(entry, budget, self.opts.headroom_gib)
        except InvalidInput as e:                            # a split GGUF with a shard missing
            self._fail(entry, rf, f"precheck: {e}")
            return
        if skip:
            if self._keep_done(entry, rf, f"skipped: {skip}"):
                return
            self._set(rf, entry.name, "skipped", reason=skip)
            self.log(f"{entry.name}: skipped: {skip}")
            return
        warning = sampler_warning(entry)
        if warning:
            self.log(warning)
        self._set(rf, entry.name, "running")
        settings = self.entry_settings(entry)
        stop = threading.Event()
        self.handler.current = stop
        log_path = Path(self.opts.out_dir) / f"{entry.name}.serve.log"
        argv = entry.serve_argv(cache=self.settings.cache, cache_dir=self.cache_dir)
        pid: int | None = None
        self.log(f"{entry.name}: launching {entry.model} (callo serve in-process; its log: {log_path})")
        try:
            self.stage = "launch"
            with (self.launcher or served_model)(entry, cache=self.settings.cache, cache_dir=self.cache_dir,
                                                 log_path=log_path, stop=stop, log=self.log) as served:
                pid = self.handler.child = served.child          # only from our own starting file; may be None
                self.stage = "calibrate"
                outcome = calibrate_one(settings, url=served.url, name=entry.name, out_dir=Path(self.opts.out_dir),
                                        resume=True, log=self.log, sleep=self.sleep, clock=self.clock, therm=self.therm)
                self.stage = "serve stop"
            self.handler.child = None        # serve has stopped (and reaped) it: never SIGKILL that pid from here on
            self._finish(entry, rf, outcome, pid, cache_ram_of(argv))
        except RunJsonUnwritable:
            raise
        except MenuFailure as e:
            self._fail(entry, rf, str(e))
        except KeyboardInterrupt:                    # StopHandler's second signal: out, after serve's cleanup ran
            self.handler.interrupted.set()
            self._fail(entry, rf, self.INTERRUPTED)
        except Exception as e:  # noqa: BLE001 -- one model's failure ends THAT model, the run goes on
            self._fail(entry, rf, f"{type(e).__name__}: {e}", log_path=log_path)
        finally:
            self.handler.current = None
            self.handler.child = None
        stopping = self.handler.interrupted.is_set
        if not wait_exit(pid, timeout=CHILD_EXIT_WAIT_S, sleep=self.sleep, clock=self.clock, should_stop=stopping):
            if stopping():
                self.log(f"{entry.name}: not waiting for llama-server pid {pid} to exit: stopping")
            else:
                self.log(f"WARNING: {entry.name}: llama-server pid {pid} is still alive after {CHILD_EXIT_WAIT_S:.0f} s")

    def _finish(self, entry: Entry, rf: RunFile, outcome: Outcome, pid: int | None, cache_ram: int | None) -> None:
        """After the serve stopped: a refused or failed calibrate is this model's failure; else the post-scan (our
        own child excluded) and the stamp (`launched_by`, `gpu_busy`, `cache_ram`), then `done`."""
        if outcome.status != "done" or outcome.profile is None:
            self._fail(entry, rf, f"{entry.name}: {outcome.reason}")
            return
        self.stage = "post-scan"
        appeared = self._scan({pid} if pid else ())
        if appeared and not self.opts.allow_gpu_busy:
            why = "another model server appeared (" + "; ".join(appeared) + ")"
            prior = self.reentered.get(entry.name)
            if prior is not None and self._rewritten(entry.name, Path(outcome.profile)):
                # a re-entered model's reads are in and its profile rewritten: kept, stamped as measured beside
                # another model server, rather than put back
                self.stage = "stamp"
                stamp_profile(outcome.profile, launched_by=self.opts.launched_by, gpu_busy=True, cache_ram=cache_ram,
                              entry=entry)
                del self.reentered[entry.name]
                what = "gate reads" if prior[0] == GATE_READS else "reads left"
                points = "with gated points " if prior[0] == GATE_READS else ""
                note = f"{what} done, but {why}; profile {points}kept, marked gpu_busy"
                self._set(rf, entry.name, "done", reason=note, profile=outcome.profile)
                self.log(f"{entry.name}: done: {note}")
                return
            self._fail(entry, rf, why, profile=outcome.profile)
            return
        self.stage = "stamp"
        stamp_profile(outcome.profile, launched_by=self.opts.launched_by, gpu_busy=self.gpu_busy or bool(appeared),
                      cache_ram=cache_ram, entry=entry)
        self.reentered.pop(entry.name, None)      # stamped: a later failure must not put the old profile back
        self._set(rf, entry.name, "done", profile=outcome.profile)
        self.log(f"{entry.name}: done: {outcome.profile}")

    # -- the run --

    def entry_settings(self, entry: Entry) -> ThinkSettings:
        """The run's settings with the entry's own `think_sampling`."""
        return (dataclasses.replace(self.settings, sampling=entry.think_sampling) if entry.think_sampling
                else self.settings)

    def todo(self, rf: RunFile) -> list[Entry]:
        """The models to run: all but those done with a valid profile and a raw file with nothing left to read under
        these settings. A done model whose raw file has reads left (a --converge-probe raw file from before the gate)
        runs again, and `calibrate_one`'s resume reads only what is missing."""
        out = []
        for e in self.entries:
            rec, path = rf.record(e.name), rf.profile_path(e.name)
            if rec["status"] == "done" and path is not None and valid_profile(str(path)):
                raw = Path(self.opts.out_dir) / f"{e.name}.raw.jsonl"
                try:
                    left = reads_left(raw, self.entry_settings(e))
                    snapshot = path.read_bytes()            # restored if a re-entry rewrites it but cannot stamp it
                except Exception as ex:  # noqa: BLE001 -- cannot tell: the done model stays done
                    self.log(f"WARNING: {e.name}: cannot tell whether {raw} has reads left, or read {path} "
                             f"({type(ex).__name__}: {ex}); taken as nothing left")
                    left = None
                if left is None:
                    self.log(f"{e.name}: done already ({path}); skipped")
                    continue
                self.reentered[e.name] = (left, dict(rec), snapshot)
                if left == GATE_READS:
                    self.log(f"{e.name}: done, but {raw} has gate reads left (a --converge-probe raw file from before "
                             f"the gate); resuming it for those alone")
                else:
                    self.log(f"{e.name}: done, but {raw} has reads left under these settings; resuming it")
            out.append(e)
        return out

    def run(self) -> int:
        """0 when every model is done or skipped by the precheck, 1 when at least one failed, run.json could not be
        written (the run stops there), the report failed unexpectedly (the models ran; the report is missing) or the
        joint report was refused (the stub report names why), 130
        when interrupted; a Refusal (exit 2) propagates before any model runs. The report and the summary are written
        in every case from the profiles that exist; their own write failures are only logged. The signal
        handlers are restored before the report, so a Ctrl-C during it is a KeyboardInterrupt out of run() (the
        CLI exits 130)."""
        self.first_gpu_check()                      # before open_run writes run.json: a refused retry needs no --resume
        installed = False
        unwritable: RunJsonUnwritable | None = None
        report_failed = False
        try:
            rf = self.open_run()
            budget = memory_budget(self.host if self.host is not None else host_info(), gpu_mem_gib=self.opts.gpu_mem_gib)
            self.log(f"memory budget: {'unknown' if budget is None else f'{budget / 2**30:.1f} GiB'}; headroom "
                     f"{self.opts.headroom_gib:g} GiB per model")
            todo = self.todo(rf)
            installed = self.handler.install()
            for i, entry in enumerate(todo):
                if self.handler.interrupted.is_set():
                    break
                if i > 0 and not self.pre_scan(entry, rf):
                    continue
                self.run_model(entry, rf, budget)
                if self.handler.interrupted.is_set():
                    break
                if i + 1 < len(todo):
                    settle(seconds=self.opts.settle_s, cooldown=self.opts.cooldown, log=self.log, sleep=self.sleep,
                           therm=self.therm, clock=self.clock, should_stop=self.handler.interrupted.is_set)
        except KeyboardInterrupt:                   # a second signal between models (the settle, the child's exit wait)
            self.handler.interrupted.set()
        except RunJsonUnwritable as e:
            unwritable = e
            self.log(f"cannot write run.json ({e}): stopping; --resume continues")
        finally:
            if installed:
                self.handler.restore()
        rf = self.run_file
        if rf is not None:
            try:
                self.write_report(rf)
            except OSError as e:
                self.log(f"WARNING: could not write the report: {e}")
            except Exception as e:  # noqa: BLE001 -- the models ran; a report bug must not hide their records
                report_failed = True
                self.log(f"report failed: {type(e).__name__}: {e}")
            try:
                for line in rf.summary():
                    self.log(line)
            except OSError as e:
                with contextlib.suppress(OSError):
                    self.log(f"WARNING: could not print the summary: {e}")
        if unwritable is not None:
            return 1
        if self.handler.interrupted.is_set():
            self.log(f"interrupted; {rf.path if rf else 'run.json'} and the raw files keep what was measured, `--resume` "
                     f"continues the run")
            return 130
        statuses = {rf.record(n)["status"] for n in rf.names()} if rf is not None else set()
        return 1 if report_failed or self.report_refused or statuses & {"failed", "running", "pending"} else 0

    def write_report(self, rf: RunFile) -> None:
        """The joint report over the done profiles, with the run's own rows (launch shape, timing, the
        failed and skipped models with their reasons); a stub report when no profile exists at all, and when compare
        refuses the joint report or a done profile does not load (its first line names why; the run exits 1). The only
        report.md in DIR: `calibrate_one` writes none (a one-server run's report is the CLI's, and it keeps a menu
        DIR's)."""
        paths = rf.done_profiles()
        runs = rf.summary_rows()
        if not paths:
            md = compare.write_stub_report(self.opts.out_dir, runs=runs)
            self.report_md = md.read_text(encoding="utf-8")
            self.log(f"no profile was written; the report names the failed and skipped models: {md}")
            return
        try:
            profiles = [compare.load_profile(p) for p in paths]
            md, js = compare.write_report(profiles, self.opts.out_dir, pref=self.settings.preference,
                                          verbose=self.settings.verbose, sources=[str(p) for p in paths], runs=runs)
        except InvalidInput as e:              # compare refused (another data file, a METRICS_VERSION bump) or a done
            self.report_refused = str(e)       # profile does not load: say so in the stub, never keep a stale report
            md = compare.write_stub_report(self.opts.out_dir, runs=runs, reason=str(e))
            self.report_md = md.read_text(encoding="utf-8")
            self.log(f"no joint report: {e}; {md} names the reason and the runs")
            return
        self.report_md = md.read_text(encoding="utf-8")
        self.log(f"report: {md} and {js}")


# ---- the CLI entry (launch sources, --dry-run) ------------------------------------------------------------

MENU_ONLY = (("--model", "model"), ("--menu", "menu"), ("--ctx", "ctx"), ("--parallel", "parallel"), ("--no-kvu", "no_kvu"),
             ("--s1-slots", "s1_slots"), ("--llama-server", "llama_server"), ("--user-delimiter", "user_delimiter"),
             ("--allow-gpu-busy", "allow_gpu_busy"), ("--mem-headroom", "mem_headroom"), ("--gpu-mem", "gpu_mem"),
             ("--settle", "settle"), ("--cooldown", "cooldown"), ("--rates-from", "rates_from"))
LAUNCH_ONLY = MENU_ONLY[2:]                      # meaningful only with --model or --menu


def _flag_check(ok: bool, flag: str, what: str, value: Any) -> None:
    if not ok:
        raise InvalidInput(f"{flag} must be {what}, got {value!r}")


def launch_from_args(args: Any) -> Launch:
    """The shared launch flags as a Launch, validated with callo serve's rules (the same predicates as a
    menu entry's): `--ctx` > 0, `--parallel` >= 1, `--s1-slots` >= 1 and below `--parallel` when it is above 1,
    `--no-kvu` only with `--ctx`, a non-empty `--user-delimiter` / `--llama-server`, and the `args` policy on the
    arguments after `--` (InvalidInput: exit 2)."""
    launch = Launch(ctx=args.ctx, parallel=DEFAULT_PARALLEL if args.parallel is None else args.parallel,
                    no_kvu=bool(args.no_kvu), s1_slots=DEFAULT_S1_SLOTS if args.s1_slots is None else args.s1_slots,
                    llama_server=DEFAULT_LLAMA_SERVER if args.llama_server is None else args.llama_server,
                    user_delimiter=args.user_delimiter, args=tuple(getattr(args, "extra", []) or []))
    _flag_check(_ctx_ok(launch.ctx), "--ctx", "a positive integer", launch.ctx)
    _flag_check(_positive_int(launch.parallel), "--parallel", "an integer >= 1", launch.parallel)
    _flag_check(_s1_slots_ok(launch.s1_slots, launch.parallel), "--s1-slots",
                f"an integer >= 1 and below --parallel ({launch.parallel}) when --parallel > 1, as for callo serve",
                launch.s1_slots)
    if launch.no_kvu and launch.ctx is None:
        raise InvalidInput("--no-kvu needs --ctx: the per-slot context to reserve, as `callo serve --no-kvu` does")
    if launch.user_delimiter is not None and not launch.user_delimiter:
        raise InvalidInput("--user-delimiter must not be empty; omit it to derive it from the chat template")
    if not launch.llama_server:
        raise InvalidInput(f"--llama-server must not be empty; omit it for {DEFAULT_LLAMA_SERVER}")
    check_args(launch.args, where="arguments after --")
    return launch


def entries_from_args(args: Any) -> tuple[list[Entry], str]:
    """(the menu, `launched_by`): `--model` (repeatable) or `--menu MENU.json`, exclusive with each other, with an
    explicitly given `--url` and with `--upstream` (the sources are exclusive). Validates the shared launch flags
    and every entry (InvalidInput: exit 2)."""
    if args.model and args.menu:
        raise InvalidInput("--model and --menu are mutually exclusive sources (one menu per run)")
    if url_given(args):
        raise InvalidInput("--url and the launch sources (--model, --menu) are mutually exclusive: --url measures a "
                           "running `callo serve`, --model / --menu launch their own")
    if args.upstream is not None:
        raise InvalidInput("--upstream and the launch sources (--model, --menu) are mutually exclusive")
    launch = launch_from_args(args)
    if args.menu:
        return load_menu(args.menu, launch), "menu"
    return entries_from_models(args.model, launch), "model"


def options_from_args(args: Any, out_dir: Path, launched_by: str) -> MenuOptions:
    headroom = DEFAULT_HEADROOM_GIB if args.mem_headroom is None else args.mem_headroom
    settle_s = DEFAULT_SETTLE_S if args.settle is None else args.settle
    # finite first: argparse's float takes inf, nan and 1e400, which overflow int(GiB * 2**30) or never end a settle;
    # the GiB values must stay finite in bytes too (1e300 GiB is a finite float, but inf bytes)
    if not (math.isfinite(headroom) and math.isfinite(headroom * 2**30) and headroom >= 0):
        raise InvalidInput(f"--mem-headroom is in GiB and must be a finite number at least 0 (finite in bytes too), got "
                           f"{headroom!r}")
    if args.gpu_mem is not None and not (math.isfinite(args.gpu_mem) and math.isfinite(args.gpu_mem * 2**30)
                                         and args.gpu_mem > 0):
        raise InvalidInput(f"--gpu-mem is in GiB and must be a finite positive number (finite in bytes too), got "
                           f"{args.gpu_mem!r}")
    if not (math.isfinite(settle_s) and 0 <= settle_s <= SETTLE_MAX_S):
        raise InvalidInput(f"--settle is in seconds and must be a finite number from 0 to {SETTLE_MAX_S:g} (a day), got "
                           f"{settle_s!r}")
    return MenuOptions(out_dir, resume=bool(args.resume), allow_gpu_busy=bool(args.allow_gpu_busy), headroom_gib=headroom,
                       gpu_mem_gib=args.gpu_mem, settle_s=settle_s, cooldown=args.cooldown,
                       rates_from=Path(args.rates_from) if args.rates_from else None, launched_by=launched_by)


def run(args: Any, *, out: IO[str] | None = None, err: IO[str] | None = None) -> int:
    """`callo calibrate --think-budgets --model GGUF ... | --menu MENU.json`: 0 when every model is done or skipped
    by the precheck, 1 when at least one failed (or the report failed), 2 for a usage error or a refusal before any
    model ran, 130 when interrupted, a second Ctrl-C during the report included (run.json and the raw files keep
    what was measured; `--resume` continues). The report goes to stdout (and `DIR/report.md`), the progress to
    stderr. A console that went away (BrokenPipeError) exits 1 without a traceback, as the one-server run does."""
    out, err = out or sys.stdout, err or sys.stderr
    try:
        return _cli(args, out, err)
    except BrokenPipeError:
        return silence_stdout(out)


def _cli(args: Any, out: IO[str], err: IO[str]) -> int:
    def log(line: str) -> None:
        print(line, file=err, flush=True)

    if (usage := think_usage_error(args)) is not None:          # the checks every --think-budgets form shares
        print(f"callo: {usage}", file=err)
        return 2
    out_dir = Path(args.out) if args.out else Path(DEFAULT_OUT_DIR)
    try:
        settings = settings_from_args(args, warn=log)
        entries, launched_by = entries_from_args(args)
        opts = options_from_args(args, out_dir, launched_by)
    except InvalidInput as e:
        print(f"callo: {e}", file=err)
        return 2
    if args.dry_run:
        return dry_run(settings, entries, opts, out=out, err=err)
    problem = data_problem(settings)               # unusable data refuses before the GPU check and any launch
    if problem is not None:
        print(f"callo: {problem}", file=err)
        return 2
    driver = Driver(settings, entries, opts, log=log)            # run_model prints each entry's sampler warning once
    try:
        rc = driver.run()
    except Refusal as e:
        print(f"callo: {e}", file=err)
        return 2
    except KeyboardInterrupt:                                    # a second Ctrl-C during the report write
        print(f"callo: interrupted; {out_dir} keeps every readout so far, `--resume` continues the run", file=err)
        return 130
    if driver.report_md:
        print(driver.report_md, file=out, end="")
    return rc


def data_problem(settings: ThinkSettings) -> str | None:
    """Why `--data` cannot calibrate any model (a bad line, no labelled text question), else None: read once before
    the GPU check and the first launch, so bad data is a refusal (exit 2) in milliseconds, never N model loads that
    each fail."""
    try:
        items, skipped = load_data(settings.data)
    except InvalidInput as e:
        return str(e)
    if not items:
        return (f"--data {settings.data_name}: no labelled text questions (skipped: "
                f"{', '.join(f'{k} {v}' for k, v in sorted(skipped.items())) or 'nothing'})")
    return None


def dry_run(settings: ThinkSettings, entries: Sequence[Entry], opts: MenuOptions, *, out: IO[str], err: IO[str]) -> int:
    """The questions, the budgets, the GPU-rule scan, the memory budget and per model its size, precheck verdict,
    worst case and (from an earlier profile) time estimate. Nothing is launched and nothing is written; exit 2 when
    the run would refuse: unusable `--data`, or the scan (another model server, a ps that failed or a scan that
    raised: the rule fails closed)."""
    problem = data_problem(settings)               # the run refuses bad data (exit 2), so its dry run says so
    if problem is not None:
        print(f"callo: {problem}", file=err)
        return 2
    items, skipped = load_data(settings.data)
    for line in dry_run_lines(settings, items, skipped)[:3]:           # the time line comes per model below
        print(line, file=out)
    try:
        servers, what = scan_findings(cache_dir=Path(opts.out_dir) / CACHE_DIR_NAME)
    except Exception as e:  # noqa: BLE001 -- as the run's first check: cannot tell, so it would refuse
        servers, what = [], gpu_check_failed_reason(e)
    if what is None and servers:
        what = "another model server is running (" + "; ".join(servers) + ")"
    rc = 0
    if what:
        print(f"GPU rule: {what}: the run would "
              + ("go on under --allow-gpu-busy, every profile marked gpu_busy" if opts.allow_gpu_busy else
                 f"refuse (exit 2); {GPU_REMEDY}"), file=out)
        rc = 0 if opts.allow_gpu_busy else 2
    else:
        print("GPU rule: no other model server is running", file=out)
    budget = memory_budget(host_info(), gpu_mem_gib=opts.gpu_mem_gib)
    print(f"memory budget: {'unknown (nothing is prechecked)' if budget is None else f'{budget / 2**30:.1f} GiB'}; "
          f"headroom {opts.headroom_gib:g} GiB per model; settle {opts.settle_s:g} s"
          + (" or cooldown therm" if opts.cooldown else "") + " between models", file=out)
    dirs = [Path(opts.out_dir)] + ([opts.rates_from] if opts.rates_from else [])
    for e in entries:
        warning = sampler_warning(e)
        if warning:
            print(warning, file=out)
        for line in dry_run_model_lines(e, settings, len(items), budget=budget, headroom_gib=opts.headroom_gib,
                                        rates_dirs=dirs, cache_dir=None):
            print(line, file=out)
    print(f"out: {opts.out_dir}/<name>.think.json, <name>.raw.jsonl, <name>.serve.log, run.json, report.md and "
          f"report.json; no requests were made and nothing was launched", file=out)
    return rc
