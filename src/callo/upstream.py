"""Attach to a llama-server: read `/props`, probe idle-slot clearing, decide whether agent pinning
is safe, and keep a `/slots` snapshot that is at most 250 ms old. Launch mode (the child process) is added by
`LlamaServerProcess` below.

Why the probe exists: with the default `--cache-idle-slots`, every task dispatch saves idle slots to the host
cache and, under unified KV, CLEARS them. A pinned request skips the LRU restore path, so an agent pinned to a
slot would re-prefill its whole context after any interleaved classification. That cannot be read from `/props`
(`cache_idle_slots`, `kv_unified`, `n_batch` are not exposed), so it is measured: two pinned 16-token
completions, then `GET /slots`; if slot N-1 lost its `n_prompt_tokens`, the upstream clears idle slots.

after measuring, the probe ERASES the two slots it filled (`POST /slots/{id}?action=erase`),
so its 16 tokens are not left in a non-reserved slot, where `callo calibrate --bias` would see them as server state.
Erase needs llama-server's `--slot-save-path`: launch mode passes it, and an attached server without it keeps the
residue, which is recorded (`probe.residue`) and named by the calibration guard.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import CalloError
from .scorer import HttpPost

HttpGet = Callable[[str], "tuple[int, Any]"]
MIN_BUILD = 9890
_BUILD_RE = re.compile(r"^b(\d{3,6})\b")
_LISTENING_RE = re.compile(r"listening on https?://[^\s\"']+:(\d+)")
PROBE_TEXT = ("the quick brown fox jumps over the lazy dog while callo probes whether this server clears idle "
              "slots on every task dispatch and so destroys pinned caches")


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "callo"


def make_http_get(*, api_key: str | None = None, timeout: float = 30.0) -> HttpGet:
    """GET returning (status, parsed JSON or None). HTTP errors are returned, not raised; network errors raise."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def get(url: str) -> tuple[int, Any]:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- local server
                raw, status = resp.read(), resp.status
        except urllib.error.HTTPError as e:
            raw, status = e.read(), e.code
        try:
            return status, json.loads(raw)
        except ValueError:
            return status, None

    return get


@dataclass(frozen=True)
class UpstreamInfo:
    base_url: str
    total_slots: int
    model_path: str | None
    n_ctx: int | None
    chat_template: str | None
    build_info: str | None
    endpoint_slots: bool
    props: dict


def discover(base_url: str, *, http_get: HttpGet) -> UpstreamInfo:
    base = base_url.rstrip("/")
    status, props = http_get(f"{base}/props")
    if status == 401:
        raise CalloError(f"GET {base}/props returned 401: the upstream wants an API key; pass --upstream-api-key "
                         f"or set LLAMA_API_KEY")
    if status != 200 or not isinstance(props, dict):
        raise CalloError(f"GET {base}/props returned {status}; is this a llama-server?")
    if props.get("role") == "router":
        raise CalloError("the upstream is a llama-server in router (multi-model) mode; callo attaches to a "
                         "single-model server only")
    slots = props.get("total_slots")
    if not isinstance(slots, int):
        raise CalloError(f"/props has no integer total_slots (got {slots!r}); refusing to attach")
    dgs = props.get("default_generation_settings")
    n_ctx = dgs.get("n_ctx") if isinstance(dgs, dict) else None
    return UpstreamInfo(base, slots, props.get("model_path"), n_ctx if isinstance(n_ctx, int) else None,
                        props.get("chat_template"), props.get("build_info"),
                        bool(props.get("endpoint_slots", False)), props)


def missing_props(info: UpstreamInfo) -> list[str]:
    return [k for k, v in (("model_path", info.model_path), ("default_generation_settings.n_ctx", info.n_ctx),
                           ("chat_template", info.chat_template), ("build_info", info.build_info)) if v is None]


def version_warning(build_info: str | None) -> str | None:
    """`/props.build_info` is `"b" + build number + "-" + commit` (e.g. `b9890-74976e1ae`; `b1-926862e` on a shallow
    clone). Warn only for a build number of 3+ digits below 9890; anything else is not comparable."""
    if not build_info:
        return None
    m = _BUILD_RE.match(build_info)
    if m and int(m.group(1)) < MIN_BUILD:
        return f"upstream build {build_info} is older than b{MIN_BUILD}; the slot and /props behaviour callo relies on was verified from b{MIN_BUILD} up"
    return None


@dataclass(frozen=True)
class ProbeResult:
    clears_idle: bool | None       # None: inconclusive (no /slots, or a single slot)
    detail: str
    n_prompt_tokens_last: int | None
    residue: tuple[int, ...] = ()  # slots the probe filled and could not erase
    erase: str | None = None       # "ok" | "unsupported" (no --slot-save-path) | "failed"; None: not attempted

    def as_dict(self) -> dict:
        return {"clears_idle": self.clears_idle, "detail": self.detail, "n_prompt_tokens_last": self.n_prompt_tokens_last,
                "residue": list(self.residue), "erase": self.erase}


def erase_slots(base_url: str, slots: Sequence[int], *, http_post: HttpPost) -> tuple[str, tuple[int, ...]]:
    """`POST /slots/{id}?action=erase` for each slot: ("ok", ()) when all were erased; ("unsupported", slots) when
    llama-server answered 501 for every one (it was started without --slot-save-path, server-context.cpp:4992);
    ("failed", the slots left) otherwise."""
    base = base_url.rstrip("/")
    left: list[int] = []
    unsupported = 0
    for s in slots:
        try:
            http_post(f"{base}/slots/{s}?action=erase", {})
        except urllib.error.HTTPError as e:
            left.append(s)
            unsupported += e.code == 501
        except (OSError, ValueError):
            left.append(s)
    if not left:
        return "ok", ()
    return ("unsupported" if unsupported == len(slots) else "failed"), tuple(left)


def probe_idle_clear(base_url: str, total_slots: int, *, http_post: HttpPost, http_get: HttpGet,
                     erase: bool = True) -> ProbeResult:
    """The idle-clear probe. Uses `n_prompt_tokens` (KV-held tokens), never the prompt text (which survives
    a clear) nor `n_prompt_tokens_processed` (which master resets to 0 on release). Then it erases the slots it
    actually filled (a probe completion that succeeded), unless `erase` is False, so an attached server's live KV in
    a slot the probe never wrote is left alone; what could not be erased is `residue`. A probe completion that timed
    out client-side may still have filled its slot on the server, yet it is neither erased nor reported as residue:
    conservative by design (callo cannot tell it from an attached server's own KV, and calibrate's guard still sees
    the tokens as another slot's, refusing wherever they shift a readout)."""
    base = base_url.rstrip("/")
    if total_slots < 2:
        return ProbeResult(None, "single slot: nothing to probe", None)
    result, filled = _probe(base, total_slots, http_post=http_post, http_get=http_get)
    if not erase or not filled:
        return result
    status, left = erase_slots(base, filled, http_post=http_post)
    return replace(result, erase=status, residue=left)


def _probe(base: str, total_slots: int, *, http_post: HttpPost,
           http_get: HttpGet) -> tuple[ProbeResult, tuple[int, ...]]:
    """(the result, the slots a probe completion was sent to successfully)."""
    filled: list[int] = []
    try:
        ids = http_post(f"{base}/tokenize", {"content": PROBE_TEXT, "add_special": False}).get("tokens", [])[:16]
        if len(ids) < 16:
            return ProbeResult(None, f"/tokenize returned {len(ids)} tokens for the probe text; expected 16", None), ()
        for slot in (total_slots - 1, 0):
            http_post(f"{base}/completion", {"prompt": ids, "n_predict": 1, "id_slot": slot, "cache_prompt": True})
            filled.append(slot)
        status, slots = http_get(f"{base}/slots")
    except (OSError, ValueError) as e:  # URLError/HTTPError/TimeoutError are OSError; bad JSON is ValueError
        return ProbeResult(None, f"probe request failed ({type(e).__name__}: {e}); cannot observe idle clearing",
                           None), tuple(filled)
    done = tuple(filled)
    if status != 200 or not isinstance(slots, list):
        return ProbeResult(None, f"GET /slots returned {status}: cannot observe KV-held tokens (is --slots on?)",
                           None), done
    last = next((s for s in slots if s.get("id") == total_slots - 1), None)
    n = last.get("n_prompt_tokens") if isinstance(last, dict) else None
    if not isinstance(n, int):
        return ProbeResult(None, f"slot {total_slots - 1} has no n_prompt_tokens in /slots", None), done
    if n == 0:
        return ProbeResult(True, f"slot {total_slots - 1} dropped to 0 KV-held tokens after a dispatch to slot 0: "
                                 f"the upstream clears idle slots", 0), done
    return ProbeResult(False, f"slot {total_slots - 1} kept {n} KV-held tokens across a dispatch to slot 0", n), done


def decide_agent_pinning(info: UpstreamInfo, probe: ProbeResult) -> tuple[bool, list[str]]:
    """Agent pinning is on only when it is safe. S1 pinning is unaffected (the router handles < 2 slots)."""
    notes: list[str] = []
    if info.total_slots < 2:
        notes.append(f"total_slots={info.total_slots}: all pinning is off (need at least 2 slots)")
        return False, notes
    if not info.endpoint_slots:
        notes.append("/props.endpoint_slots is false: cannot verify idle-slot clearing, assuming the unsafe case; "
                     "agent pinning OFF. Start llama-server with --slots to enable it")
        return False, notes
    if probe.clears_idle is True:
        notes.append("ERROR: the upstream clears idle slots on every dispatch, which would wipe a pinned agent's "
                     "KV after each classification. Agent pinning is DISABLED. Restart llama-server with "
                     "--no-cache-idle-slots and -kvu (callo's launch mode does this)")
        return False, notes
    if probe.clears_idle is None:
        notes.append(f"idle-clear probe inconclusive ({probe.detail}); assuming the unsafe case: agent pinning OFF")
        return False, notes
    return True, notes


class SlotsCache:
    """`GET /slots`, re-fetched only when the last snapshot is older than `max_age` (250 ms)."""

    def __init__(self, base_url: str, *, http_get: HttpGet, max_age: float = 0.25):
        self._url = f"{base_url.rstrip('/')}/slots"
        self._get, self._max_age = http_get, max_age
        self._cur: tuple[float, list[dict] | None] = (0.0, None)    # (time the GET was sent, snapshot): one store
        self.last_error: str | None = None

    def get(self) -> list[dict] | None:
        return self.stamped()[0]

    def stamped(self) -> tuple[list[dict] | None, float]:
        """(snapshot, `time.monotonic()` when its GET /slots was sent). Whatever the snapshot shows is at least that
        recent, which is what the S1 fallback needs to tell a stale "busy" from a real one (`SlotRouter.choose_s1`)."""
        now = time.monotonic()
        at, snap = self._cur
        if snap is not None and now - at < self._max_age:
            return snap, at
        try:
            status, body = self._get(self._url)
        except OSError as e:
            self.last_error, self._cur = f"{type(e).__name__}: {e}", (0.0, None)
            return None, now
        if status != 200 or not isinstance(body, list):
            self.last_error, self._cur = f"GET /slots returned {status}", (0.0, None)
            return None, now
        self.last_error, self._cur = None, (now, body)
        return body, now


def wait_health(base_url: str, *, timeout: float = 600.0, interval: float = 0.25, http_get: HttpGet,
                alive: Callable[[], bool] | None = None) -> None:
    """Poll `/health` until 200. `alive` (launch mode: the child's liveness) turns a process that died after
    printing `listening on` into a prompt `CalloError` instead of a wait for the whole `timeout`."""
    url = f"{base_url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout
    last = "no response yet"
    while time.monotonic() < deadline:
        if alive is not None and not alive():
            raise CalloError(f"the upstream process exited before {url} was ok (last: {last})")
        try:
            status, body = http_get(url)
            if status == 200:
                return
            last = f"{status} {body}"
        except OSError as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(interval)
    raise CalloError(f"upstream /health not ok after {timeout:.0f}s: {last}")


def resolve_gguf(info: UpstreamInfo, override: str | None) -> Path:
    candidate = Path(override) if override else (Path(info.model_path) if info.model_path else None)
    if candidate is None or not candidate.is_file():
        raise CalloError(f"no readable GGUF: /props.model_path={info.model_path!r}, --gguf={override!r}. callo reads "
                         f"the vocabulary from the served GGUF; when the upstream is remote pass --gguf PATH to a "
                         f"local copy of the same file")
    return candidate


# ---- launch mode --------------------------------------------------------------------------

LAUNCH_FLAGS = ["-kvu", "--no-cache-idle-slots", "-b", "512", "-ub", "512", "--jinja"]


def build_argv(llama_server: str, model: str, *, ctx: int | None, parallel: int, extra: Sequence[str],
               mmproj: str | None = None, unified_kv: bool = True, slot_save_path: str | None = None) -> list[str]:
    """`llama-server -m X --port 0 [-c ctx] -np parallel [-kvu] --no-cache-idle-slots -b 512 -ub 512 --jinja
    [--slot-save-path DIR] [--mmproj M] [extra]`. `--slot-save-path` lets the startup probe erase the two
    slots it fills (those whose probe completion succeeded: one that timed out client-side is neither erased nor
    reported as residue, see probe_idle_clear); the directory must exist. callo's flags come first so a passthrough
    flag wins (llama-server parses last-wins). `-c` is only passed when `--ctx` was given: the default is
    llama-server's own, never assumed. `--mmproj` sits after callo's fixed flags and before the
    passthrough. `unified_kv=False` (`--no-kvu`) drops `-kvu`; `ctx` is passed through as the `-c` value, so the
    caller multiplies a per-slot context by `parallel` first: without `-kvu`, llama-server gives each slot `-c / -np`."""
    argv = [llama_server, "-m", model, "--port", "0"]
    if ctx is not None:
        argv += ["-c", str(ctx)]
    argv += ["-np", str(parallel), *(f for f in LAUNCH_FLAGS if unified_kv or f != "-kvu")]
    if slot_save_path is not None:
        argv += ["--slot-save-path", slot_save_path]
    if mmproj is not None:
        argv += ["--mmproj", mmproj]
    return argv + list(extra)


def check_passthrough_args(extra: Sequence[str]) -> None:
    """`--port 0` fails with more than one TCP host (server-http.cpp:134), so a comma-separated --host is refused."""
    for i, a in enumerate(extra):
        value = None
        if a == "--host" and i + 1 < len(extra):
            value = extra[i + 1]
        elif a.startswith("--host="):
            value = a.split("=", 1)[1]
        if value is not None and "," in value:
            raise CalloError(f"passthrough --host {value!r} names several addresses; --port 0 (which launch mode "
                             f"needs to avoid a port race) supports one TCP host. Use attach mode for multi-host")


def parse_listening(line: str) -> int | None:
    """The port from llama-server's `listening on http://HOST:PORT` line (server.cpp:519), matched as a
    substring so `--log-jsonl` output parses too."""
    m = _LISTENING_RE.search(line)
    return int(m.group(1)) if m else None


class LlamaServerProcess:
    def __init__(self, argv: Sequence[str], *, cache_dir: Path, startup_timeout: float = 600.0):
        self.argv = list(argv)
        self.cache_dir = Path(cache_dir)
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.log_file: Path = self.cache_dir / "llama-server-starting.log"
        self.pid_file: Path = self.cache_dir / "llama-server-starting.pid"
        self.returncode: int | None = None
        self._stopping = False
        self._starting_pid_file: Path | None = None

    def start(self) -> int:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.cache_dir / f"llama-server-starting-{os.getpid()}.log"
        with self.log_file.open("wb") as log:   # closed here even when Popen raises (a missing binary)
            self.proc = subprocess.Popen(self.argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
        # The pid file exists from the moment the child does: model load can take minutes, and a callo SIGKILLed in
        # that window must not leave an invisible llama-server behind. `_bind_files` renames it once the port is known.
        self._starting_pid_file = self.pid_file = self.cache_dir / f"llama-server-starting-{os.getpid()}.pid"
        self.pid_file.write_text(f"{self.proc.pid}\n")
        deadline = time.monotonic() + self.startup_timeout
        seen = 0
        while time.monotonic() < deadline:
            with self.log_file.open("rb") as f:
                f.seek(seen)
                data = f.read()
            complete = data[: data.rfind(b"\n") + 1]    # a line still being written must not parse (`:543` of `:54321`)
            seen += len(complete)
            for line in complete.decode("utf-8", errors="replace").splitlines():
                port = parse_listening(line)
                if port is not None:
                    self.port = port
                    self._bind_files(port)
                    return port
            rc = self.proc.poll()
            if rc is not None:
                self.returncode = rc
                self._remove_pid_files()
                tail = self.log_file.read_text(errors="replace")[-2000:]
                raise CalloError(f"llama-server exited with {rc} before listening. Log tail ({self.log_file}):\n{tail}")
            time.sleep(0.1)
        self.stop()
        raise CalloError(f"llama-server did not print 'listening on' within {self.startup_timeout:.0f}s "
                         f"(log: {self.log_file})")

    def _bind_files(self, port: int) -> None:
        final_log = self.cache_dir / f"llama-server-{port}.log"
        self.log_file.replace(final_log)
        self.log_file = final_log
        final_pid = self.cache_dir / f"llama-server-{port}.pid"
        self.pid_file.replace(final_pid)                 # atomic: there is never a window with no pid file
        self.pid_file = final_pid
        self._starting_pid_file = None

    def _remove_pid_files(self) -> None:
        self.pid_file.unlink(missing_ok=True)
        if self._starting_pid_file is not None:
            self._starting_pid_file.unlink(missing_ok=True)

    def _killpg(self, sig: int) -> None:
        """Signal the whole group. The child is its session leader (pgid == pid), and the group outlives the
        leader, so this is sent even when the leader has already exited: grandchildren must not survive."""
        try:
            os.killpg(self.proc.pid, sig)  # type: ignore[union-attr]
        except (ProcessLookupError, PermissionError):    # empty group (ESRCH) / a group of only zombies (macOS EPERM)
            pass

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace: float = 10.0) -> None:
        if self.proc is None:
            return
        self._stopping = True
        self._killpg(signal.SIGTERM)
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self._killpg(signal.SIGKILL)
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                logging.getLogger(__name__).warning("llama-server pid %d did not exit after SIGKILL", self.proc.pid)
        self.returncode = self.proc.poll()
        self._remove_pid_files()

    def watch(self, on_exit: Callable[[int], None]) -> threading.Thread:
        """A daemon thread that calls `on_exit(returncode)` once, when the child dies on its own: a deliberate
        `stop()` is not a death, so it stays silent."""
        def run() -> None:
            while self.proc is not None and self.proc.poll() is None:
                time.sleep(0.25)
            if self.proc is not None:
                self.returncode = self.proc.returncode
                if not self._stopping:
                    on_exit(self.proc.returncode)

        t = threading.Thread(target=run, daemon=True, name="callo-child-watch")
        t.start()
        return t


def find_orphans(cache_dir: Path) -> list[tuple[int, int | None]]:
    """(pid, port) for every `llama-server-<port>.pid` whose pid is still alive: a child left by a SIGKILLed
    callo. A `llama-server-starting-<callo pid>.pid` (the child was still loading its model) reports port None.
    Stale files (dead or unparseable pid) are removed."""
    out: list[tuple[int, int | None]] = []
    if not Path(cache_dir).is_dir():
        return out
    for f in sorted(Path(cache_dir).glob("llama-server-*.pid")):
        try:
            pid = int(f.read_text().strip())
            port = None if f.stem.startswith("llama-server-starting-") else int(f.stem.rsplit("-", 1)[1])
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError):
            f.unlink(missing_ok=True)
            continue
        except PermissionError:
            pass   # the pid exists but belongs to another user: still an orphan worth reporting
        out.append((pid, port))
    return out
