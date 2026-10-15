"""callo v0.3 position-bias acceptance bench: the position-bias profile, live, on the GPU: the pre-registered
confirmatory run of the pre-registration document `PREREG`.

    uv run --frozen python -m bench.acceptance_v03_bias --expect-head C \\
        --out DIR [--models tiel_kvu,qwen9b]

Run it only on an otherwise idle GPU, in the background. It
refuses to start while any model server runs or a volume has under 20 GiB free, starts one llama-server at a time,
and a watchdog stops it cleanly (exit 3) if another model server appears or a disk fills during the run.

It also refuses to start unless the fit and eval files and the pre-registration are the pinned ones (`pin_problems`),
and unless the code that runs is the clean pre-registered commit C, imported from this checkout (`code_problems`).
Per model, in the pre-registration's order: launch callo (launch mode), fit on public-pool-v6 calibration (the user's own data)
with `callo calibrate --bias`; on TielCoder -kvu the fit IS gate B0 (a fresh default launch, no --allow-busy),
followed by B4 (a resident 300-token prompt, a continuous decode, the negative control), then B3 (the upstream call
sequence and the CPU cost); then read public-pool-v6 test under every rotation with the same command, and judge B1, B2
and B5 offline (`bench/bias_analysis.py`). A fit that does not exit 0 stops that model early: there is nothing to
judge the held-out read against. Gates judged before an interruption (an error, the watchdog, a stop signal) are kept
in summary.json, and a rerun never measures them again. A rerun of a model that did not complete re-derives B1, B2 and
B5 with `bias_analysis.analyse`, which is a pure function of the frozen raw files, fit.json and seed 0, so their
values cannot change; B0, B3 and B4, which measure the live server, are carried over once judged.
B4's three probe runs write their small profiles and raw files under b4_probes/, never beside fit.json.
summary.json ends with the pre-registered verdict (`bias_analysis.prereg_verdict`); the exit code is 0 only when every
gate run passed and that verdict is neither ERROR nor FALSIFIED (a required model left out of --models is ERROR: 1).
`bench/report_v03_bias.py` renders docs/bench/v0.3-bias.md from summary.json.
"""
from __future__ import annotations

import _thread
import argparse
import contextlib
import hashlib
import http.client
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import callo
from bench import bias_analysis as ba
from bench import lib
from bench.acceptance import MASTER, Callo
from bench.acceptance import MODEL as QWEN9B
from bench.acceptance_v02 import TIEL, _install_stop_handlers, s2_stream_payload
from callo import cli
from callo.engine import Engine, QuestionReadout, SurfaceSets, load_gguf_meta
from callo.guard import FAILED_LINE
from callo.profile import load_profile
from callo.prompt import TemplateClient
from callo.scorer import LlamaCppForwardScorer, make_http_post
from callo.upstream import default_cache_dir

ROOT = Path(__file__).resolve().parents[1]
POOL = lib.env_path(lib.DATA_ENV, "public-pool-v6")
FIT_DATA = f"{POOL}/calibration.jsonl" if POOL else None   # the pre-registered fit set: the user's own data
EVAL_DATA = f"{POOL}/test.jsonl" if POOL else None         # the pre-registered held-out set B1, B2 and B5 judge
# the files are pinned by sha256, and the bench refuses to start on any other (pin_problems)
DATA_SHA256 = {FIT_DATA: "bb43a590d115bc559b56dc8a40b441b490d4bc87d99cbe344c66c3a44dcc1f51",
               EVAL_DATA: "ea63dfe433d05ce9548399ac770a87c6b427f47662bd68121f0091ac9119493a"}
PREREG = ROOT / "docs" / "specs" / "2026-10-04-callo-v0.3-bias-prereg.md"
# what must be committed, unmodified, in the checkout that runs
TRACKED = ("src", "bench", "pyproject.toml", "uv.lock", "docs/specs/2026-10-04-callo-v0.3-bias-prereg.md")
FINAL = ("CONFIRMED", "FALSIFIED", "INCONCLUSIVE")   # a summary.json with one of these is never re-run
CTX = 32768
# No tiel_nokvu: its 7,520 readouts were bit-identical to tiel_kvu's (one sequence on an idle reserved slot
# runs the same computation under either KV layout), so it was a replicate, not a third model.
MODELS = {"tiel_kvu": (TIEL, [], CTX), "qwen9b": (QWEN9B, [], CTX)}
FILL_TOKENS = 300
SNAPSHOT_S = 0.5                     # callo's /slots snapshot can be 250 ms old (SlotsCache): wait after a state change
SETUP_TRIES = 20                     # B4: how many snapshots to wait for a load (or its removal) to show
DECODE_WAIT_S = 120.0                # B4(b): how long the continuous decode gets to produce its first token
WATCH_JOIN_S = 60.0                  # main: how long to wait for a watchdog tick in progress when the run ends
EVERYTHING = "1000000"               # --max-per-n for the held-out reads: every question
# the boot container (results, logs) and the models' root when set; plus --out
DISK_PATHS = ("/", *filter(None, [lib.env_path(lib.MODELS_ENV)]))
MIN_FREE_GIB = 20.0
WATCH_S = 30.0
log = logging.getLogger("bench")
_OWN: set[int] = set()               # pids of the llama-server(s) this bench launched: the watchdog ignores them
_CACHES: set[Path] = set()           # cache dirs of this bench's launches: their pid files name a server still loading
_WATCH: Watchdog | None = None
_PROGRESS: Path | None = None
_DONE: list[str] = []
_PARTIAL: dict[str, dict] = {}       # the model in flight's result so far: summary.json keeps it on an interruption


class Aborted(Exception):
    """The watchdog stopped the run (a foreign model server, or a disk under MIN_FREE_GIB)."""


class B4Setup(RuntimeError):
    """B4's load could not be created, removed or seen, so B4 cannot be judged."""


# ---- safety: disk, watchdog, progress ----------------------------------------------------------------------

def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin_problems(pins: dict[str, str] | None = None, prereg: Path | None = None) -> list[str]:
    """Why the run would not be the pre-registered one: a fit or eval file that is missing or not the pinned bytes, or
    no pre-registration document. Empty when everything matches."""
    pins = DATA_SHA256 if pins is None else pins
    prereg = PREREG if prereg is None else prereg
    out = []
    for path, want in pins.items():
        try:
            got = _sha256(path)
        except OSError as e:
            out.append(f"{path}: cannot read it ({type(e).__name__}: {e})")
            continue
        if got != want:
            out.append(f"{path} has sha256 {got}, not the pre-registered {want}")
    if not prereg.is_file():
        out.append(f"{prereg} is missing: the run must follow its pre-registration")
    return out


def pinned_commit(prereg: Path | None = None) -> str | None:
    """The commit C appended to the pre-registration (`Pinned commit C: <40-hex sha> (...)`); None while
    the line is absent or the file cannot be read."""
    try:
        text = (PREREG if prereg is None else prereg).read_text()
    except OSError:
        return None
    found = re.findall(r"Pinned commit C:\W*([0-9a-f]{40})\b", text)
    return found[-1] if found else None


def _sources() -> dict[str, str]:
    """Where the code that runs was imported from: the callo package and the bench package."""
    return {"callo": str(Path(callo.__file__).resolve().parent), "bench": str(Path(ba.__file__).resolve().parent)}


def code_state(root: Path | None = None, expect: str | None = None, sources: dict[str, str] | None = None) -> dict:
    """HEAD, `git status --porcelain` (lines intact), the porcelain lines under TRACKED, `expected`: --expect-head
    resolved to a full sha (a short sha is fine; None when it names no commit here), of the checkout that runs, its
    `root`, and `sources`: where callo and the bench were imported from. `error` when it is not a git
    checkout."""
    root = ROOT if root is None else root
    sources = _sources() if sources is None else sources

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=True).stdout.rstrip("\n")
    try:
        state = {"head": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
                 "dirty": git("status", "--porcelain", "--", *TRACKED), "expected": None,
                 "root": str(Path(root).resolve()), "sources": sources}
    except (OSError, subprocess.CalledProcessError) as e:
        return {"head": None, "status": None, "dirty": None, "expected": None, "root": str(root), "sources": sources,
                "error": f"{type(e).__name__}: {e}"}
    if expect:
        try:
            state["expected"] = git("rev-parse", "--verify", "--quiet", f"{expect}^{{commit}}")
        except subprocess.CalledProcessError:
            pass
    return state


def code_problems(state: dict, expect_head: str, pinned: str | None = None) -> list[str]:
    """Why the code under test would not be the pre-registered commit: not a git
    checkout; uncommitted changes under TRACKED; an --expect-head that is not a 7-40 hex sha (HEAD, @
    or a branch would always match), names no commit, or is not HEAD; callo or the bench imported from outside this
    checkout (another venv); or, once the prereg carries its `Pinned commit C:` line (`pinned`), another commit."""
    if state.get("error"):
        return [f"cannot read the checkout's git state ({state['error']}): run from a git worktree"]
    out = []
    if state["dirty"]:
        out.append(f"uncommitted changes under {', '.join(TRACKED)} ({state['dirty']!r}): run from a clean detached "
                   f"worktree at the pre-registered commit")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", expect_head):
        out.append(f"--expect-head {expect_head} is not a commit sha (7 to 40 hex digits): name the pre-registered "
                   f"commit C itself, not a ref")
    elif state.get("expected") is None:
        out.append(f"--expect-head {expect_head} does not name a commit in this checkout")
    elif state["head"] != state["expected"]:
        out.append(f"HEAD is {state['head']}, not --expect-head {expect_head} ({state['expected']})")
    elif pinned is not None and state["expected"] != pinned:
        out.append(f"--expect-head {expect_head} ({state['expected']}) is not the prereg's pinned commit C {pinned}")
    root = Path(state.get("root") or ".")
    sources = state.get("sources") or {}
    for name, sub in (("callo", "src"), ("bench", "bench")):
        where = sources.get(name)
        if where is None or not Path(where).resolve().is_relative_to((root / sub).resolve()):
            out.append(f"{name} is imported from {where}, not from {root / sub}: run with this checkout's own "
                       f"environment (`uv run --frozen` from it)")
    return out


def model_complete(name: str, m: dict | None) -> bool:
    """A model is complete, and kept as it is on a rerun, only without an `error` and with every gate
    REQUIRED of it judged: PASS or FAIL, or `pass: None` without an `error` for B1 (nothing tested) and B5 (nothing
    corrected). Anything else (a B0 infrastructure ERROR, a failed fit, a failed eval read) is run again."""
    if not m or "error" in m:
        return False
    for g in ba.REQUIRED.get(name, ()):
        v = m.get(g)
        if not isinstance(v, dict) or v.get("error"):
            return False
        if v.get("pass") is None and g not in ba.VACUOUS_OK:
            return False
    return True


def disk_problems(paths: Sequence[str], min_gib: float, *,
                  free: Callable[[str], int] = lambda p: shutil.disk_usage(p).free) -> list[str]:
    out = []
    for p in paths:
        try:
            gib = free(p) / 1024**3
        except OSError as e:
            out.append(f"{p}: cannot read its free space ({type(e).__name__}: {e}; not mounted?)")
            continue
        if gib < min_gib:
            out.append(f"{p} has {gib:.1f} GiB free (< {min_gib:g} GiB)")
    return out


def _own_pids() -> set[int]:
    """The llama-servers this bench started: launched ones, plus any still loading its model (a launch blocks until
    the server is ready, which can take minutes, and the pid file its launch wrote is the only trace until then)."""
    own = set(_OWN)
    for cache in list(_CACHES):
        for _ in range(2):              # a listed file that vanished was renamed to its port-named file: glob again
            vanished = False
            for f in sorted(cache.glob("llama-server-*.pid")) if cache.is_dir() else ():
                try:                    # read-only: a file still being written is skipped, never deleted
                    pid = int(f.read_text().strip())
                except FileNotFoundError:
                    vanished = True
                    continue
                except (OSError, ValueError):
                    continue
                try:
                    os.kill(pid, 0)     # liveness only: signal 0 is never delivered
                except PermissionError:
                    pass                # it exists, under another user
                except OSError:
                    continue            # no such process
                own.add(pid)
            if not vanished:
                break
    return own


PS_FAILURES_MAX = 3      # the watchdog's consecutive failed ps listings before it stops the run (fails closed)


class Watchdog(threading.Thread):
    """Every `interval` s: another model server (anything `lib.running_gpu_servers` reports that this bench did not
    launch), or a volume under `min_free_gib`, sets `reason` and calls `interrupt` once. A ps that
    fails (`running_gpu_servers` raises: it fails closed) is logged and retried next tick; PS_FAILURES_MAX failures in
    a row interrupt the run as a foreign server does ("cannot tell"), so the watchdog never dies silently."""

    def __init__(self, paths: Sequence[str], *, interrupt: Callable[[], Any],
                 own: Callable[[], set[int]] = _own_pids, interval: float | None = None,
                 min_free_gib: float | None = None):
        super().__init__(daemon=True, name="bench-watchdog")
        self.paths, self.interrupt, self.own = list(paths), interrupt, own
        self.interval = WATCH_S if interval is None else interval
        self.min_free_gib = MIN_FREE_GIB if min_free_gib is None else min_free_gib
        self.reason: str | None = None
        self._halt = threading.Event()

    def run(self) -> None:
        failures = 0
        while not self._halt.wait(self.interval):
            reason = None
            try:
                servers = lib.running_gpu_servers()  # before own(): a server that became ours in between stays ours
            except Exception as e:  # noqa: BLE001 -- a failed ps is "cannot tell": retried, never the thread's end
                failures += 1
                log.warning("watchdog: cannot tell whether a model server is running (%s/%s): %s", failures,
                            PS_FAILURES_MAX, e)
                if failures < PS_FAILURES_MAX:
                    continue
                reason = f"cannot tell whether a model server is running: ps failed {failures} times in a row ({e})"
            else:
                failures = 0
                mine = self.own()
                foreign = [(pid, cmd) for pid, cmd in servers if pid not in mine]
                if foreign:
                    reason = "another model server appeared during the run: " + "; ".join(
                        f"pid {pid} ({cmd})" for pid, cmd in foreign)
                elif problems := disk_problems(self.paths, self.min_free_gib):
                    reason = "a volume ran low on space during the run: " + "; ".join(problems)
            if reason is not None:
                if self._halt.is_set():     # the run ended during this tick: nothing left to stop
                    return
                self.reason = reason
                log.error("watchdog: %s; stopping the bench", self.reason)
                self.interrupt()
                return

    def stop(self) -> None:
        self._halt.set()


def _progress(model: str | None, step: str) -> None:
    """OUT/progress.json for whoever polls the run: written atomically at every step."""
    if _PROGRESS is None:
        return
    tmp = _PROGRESS.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pid": os.getpid(), "model": model, "step": step, "done": list(_DONE),
                               "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
    tmp.replace(_PROGRESS)


def _check_watch() -> None:
    if _WATCH is not None and _WATCH.reason is not None:
        raise Aborted(_WATCH.reason)


# ---- live helpers ------------------------------------------------------------------------------------------

class _Tee(io.TextIOBase):
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def calibrate(c: Any, out_dir: Path, name: str, data: str | Path | None, *extra: str,
              fresh: bool = False) -> tuple[int, dict | None, Path, str]:
    """`callo calibrate --bias` against callo `c`, in process: (rc, profile, raw path, stderr). `fresh` deletes this
    run's own raw file first (a B4 probe must refuse on the guard, not on an old --raw); otherwise a rerun of the
    bench resumes. The stderr is captured (B4 reads why a run refused) and passed through to the log."""
    prof, raw = out_dir / f"{name}.json", out_dir / f"{name}.raw.jsonl"
    if fresh:
        raw.unlink(missing_ok=True)
    argv = ["calibrate", "--bias", "--url", c.url, "--out", str(prof), "--raw", str(raw),
            *(["--data", str(data)] if data is not None else []), *extra]
    if raw.exists() and not fresh:
        argv.append("--resume")
    captured = io.StringIO()
    with contextlib.redirect_stderr(_Tee(captured, sys.stderr)):
        rc = cli.main(argv)
    _check_watch()
    return rc, (json.loads(prof.read_text()) if rc == 0 and prof.exists() else None), raw, captured.getvalue()


def failed_conditions(stderr: str) -> list[str]:
    found = list(FAILED_LINE.finditer(stderr))
    return found[-1].group(1).split(", ") if found else []


def read_heldout(c: Any, d: Path, run: str, data: str | Path | None) -> dict:
    """One held-out read of every rotation; a non-zero exit is resumed once."""
    rc, _, raw, _ = calibrate(c, d, run, data, "--max-per-n", EVERYTHING)
    attempts = 1
    if rc != 0:
        log.warning("%s exited %d: resuming it once", run, rc)
        rc, _, raw, _ = calibrate(c, d, run, data, "--max-per-n", EVERYTHING)
        attempts = 2
    return {"rc": rc, "readouts": ba.raw_readouts(raw), "attempts": attempts}


def b3_call_sequence(upstream: str, gguf: str, profile: str | Path, bodies: Sequence[dict]) -> dict:
    """B3's call half: the same bodies through callo's engine without and with the profile, recording every upstream
    call (endpoint and exact payload). PASS needs the two sequences identical and at least one readout applied."""
    meta = load_gguf_meta(gguf, default_cache_dir())
    pb = load_profile(profile)
    logs: dict[str, list] = {}
    applied = 0
    for name, bias in (("without", None), ("with", pb)):
        calls: list = []
        real = make_http_post()

        def post(url: str, payload: Any, calls: list = calls, real: Any = real) -> dict:
            calls.append((url.rsplit("/", 1)[-1], json.dumps(payload, sort_keys=True)))
            return real(url, payload)

        eng = Engine(LlamaCppForwardScorer(upstream, http_post=post), TemplateClient(upstream, http_post=post),
                     SurfaceSets(meta.tokens, meta.tok_model), position_bias=bias)
        for body in bodies:
            readouts: list[QuestionReadout] = []
            eng.answer_request(body, id_slot=0, readouts=readouts)
            applied += sum(r.readout.bias == "applied" for r in readouts)
        logs[name] = calls
    identical = logs["without"] == logs["with"]
    return {"calls": len(logs["without"]), "identical": identical, "applied": applied,
            "pass_calls": identical and applied > 0}


def start_decode(c: Any) -> Callable[[], Any]:
    """B4(b): a continuous decode through callo (agent pinning puts it on a non-reserved slot). A seam."""
    s2 = lib.ChatStream(f"{c.url}/v1/chat/completions", s2_stream_payload, headers={"X-Callo-Conversation": "b4"})
    s2.start()
    if not s2.first_delta.wait(DECODE_WAIT_S):
        s2.stop()
        raise B4Setup(f"B4 setup: the continuous decode never produced a token in {DECODE_WAIT_S:g} s")
    return s2.stop


def _post_ok(url: str, payload: dict, what: str) -> None:
    try:
        status, _, body = lib.post(url, payload)
    except (OSError, http.client.HTTPException, ValueError) as e:  # connection errors, timeouts, a non-JSON body
        raise B4Setup(f"B4 setup: {what} ({url}) failed: {type(e).__name__}: {e}") from e
    if status != 200:
        raise B4Setup(f"B4 setup: {what} ({url}) returned HTTP {status}: {str(body)[:200]}")


def _fill(c: Any, slot: int, n_tokens: int) -> None:
    """Park `n_tokens` dummy tokens in `slot` on the upstream (the README diagnostic's s2 state)."""
    _post_ok(f"{c.upstream}/completion", {"prompt": list(range(1000, 1000 + n_tokens)), "n_predict": 1,
                                          "id_slot": slot, "cache_prompt": True}, f"filling slot {slot}")


def _erase_all(c: Any) -> None:
    for s in range(c.parallel):
        _post_ok(f"{c.upstream}/slots/{s}?action=erase", {}, f"erasing slot {s}")


def _erase_best_effort(c: Any) -> list[str]:
    """Erase every slot of the bench's own launched server, trying each one whatever the others did; the failures."""
    failed = []
    for s in range(c.parallel):
        try:
            _post_ok(f"{c.upstream}/slots/{s}?action=erase", {}, f"erasing slot {s}")
        except B4Setup as e:
            failed.append(str(e))
    return failed


def _seen(c: Any, ok: Callable[[list[dict], set[int]], bool], what: str) -> None:
    """Wait until callo's /callo/stats (what the guard reads) shows `what`; B4Setup if it never does."""
    for _ in range(SETUP_TRIES):
        time.sleep(SNAPSHOT_S)
        try:
            stats = c.stats()
        except (OSError, http.client.HTTPException, ValueError) as e:
            raise B4Setup(f"B4 setup: reading /callo/stats while waiting for {what} failed: "
                          f"{type(e).__name__}: {e}") from e
        slots = [s for s in stats.get("slots") or [] if isinstance(s, dict)]
        if ok(slots, set((stats.get("router") or {}).get("reserved") or [])):
            return
    raise B4Setup(f"B4 setup: {what} never showed in /callo/stats slots ({slots})")


def _b4(c: Any, d: Path, out: dict) -> None:
    small = ("--max-per-n", "4")
    probes = d / "b4_probes"            # 4-question probe "profiles", never mistaken for fit.json
    probes.mkdir(parents=True, exist_ok=True)
    last = c.parallel - 1
    _fill(c, last, FILL_TOKENS)
    _seen(c, lambda slots, reserved: any(s.get("id") == last and last not in reserved
                                         and (s.get("n_prompt_tokens") or 0) > 0 for s in slots),
          f"{FILL_TOKENS} tokens resident in non-reserved slot {last}")
    rc_a, _, _, err_a = calibrate(c, probes, "b4a", FIT_DATA, *small, fresh=True)
    _erase_all(c)
    stop = start_decode(c)
    try:
        _seen(c, lambda slots, reserved: any(s.get("is_processing") and s.get("id") not in reserved for s in slots),
              "a decode in a non-reserved slot")
        rc_b, _, _, err_b = calibrate(c, probes, "b4b", FIT_DATA, *small, fresh=True)
    finally:
        stop()
    _erase_all(c)
    _seen(c, lambda slots, _: not any(s.get("is_processing") or (s.get("n_prompt_tokens") or 0) > 0 for s in slots),
          "every slot idle and empty")
    rc_c, _, _, _ = calibrate(c, probes, "b4c", FIT_DATA, *small, fresh=True)
    a_failed, b_failed = failed_conditions(err_a), failed_conditions(err_b)
    ok = (rc_a == 2 and "foreign_tokens" in a_failed and rc_b == 2
          and bool({"foreign_processing", "foreign_tokens"} & set(b_failed)) and rc_c == 0)
    out["b4"] = {"a_rc": rc_a, "a_failed": a_failed, "b_rc": rc_b, "b_failed": b_failed, "c_rc": rc_c, "pass": ok}


def judged(gate: Any) -> bool:
    """A gate with a PASS or FAIL: never re-run (the pre-registration's rule)."""
    return isinstance(gate, dict) and gate.get("pass") in (True, False)


def b0_verdict(rc: int, prof: dict | None, stderr: str) -> dict:
    """B0: PASS on exit 0 with a clean profile; FAIL when the guard refused the idle default
    launch (exit 2 with the guard's failed-conditions line) or the profile is busy; anything else (a request that
    failed for good, a precondition or usage refusal, no profile) is ERROR: infrastructure, not the guard."""
    state = prof["server_state"]["state"] if prof else None
    failed = failed_conditions(stderr)
    if rc == 0 and state == "clean":
        return {"rc": rc, "state": state, "pass": True}
    if (rc == 2 and failed) or (rc == 0 and state == "busy"):
        return {"rc": rc, "state": state, "failed": failed, "pass": False}
    return {"rc": rc, "state": state, "pass": None,
            "error": f"the fit exited {rc} without a guard refusal (state {state}): infrastructure, not B0"}


def b0_b4(c: Any, d: Path, prev: dict | None = None) -> dict:
    """On ONE fresh default TielCoder -kvu launch: B0 (the idle fit), B4(a) under a resident 300-token prompt, B4(b)
    under a continuous decode, B4(c) the negative control with the load removed and every slot erased. Each load is
    checked in /callo/stats before calibrate runs, and each refusal must be for the load's own reason. A
    gate `prev` already judged is kept, not re-run. A judged B0 whose fit exited 0 but whose
    fit.json is gone is fitted again (`refit`): B0 stays as judged, and the refit's exit decides, like
    a fit's, whether the rest of the model runs. The same command resumes from fit.raw.jsonl when it is still there."""
    prev = prev or {}
    if judged(prev.get("b0")):
        out: dict[str, Any] = {"b0": prev["b0"]}
        if prev["b0"].get("rc") == 0 and not (d / "fit.json").exists():
            rc = calibrate(c, d, "fit", FIT_DATA)[0]
            out["refit"] = {"rc": rc, "note": "fit.json was missing: fitted again; B0 stays as judged"}
    else:
        rc, prof, _, err = calibrate(c, d, "fit", FIT_DATA)
        out = {"b0": b0_verdict(rc, prof, err)}
    if judged(prev.get("b4")):
        out["b4"] = prev["b4"]
        return out
    try:
        _b4(c, d, out)
    except B4Setup as e:
        # A half-done setup can leave its load behind (a filled slot), which would make the held-out reads that
        # follow refuse on foreign_tokens: erase the slots of OUR launched server (c.upstream), best effort.
        left = _erase_best_effort(c)
        out["b4"] = {"pass": None, "error": str(e) + (f"; erasing the slots afterwards failed: {'; '.join(left)}"
                                                      if left else ""), "erase_after_error": left or "ok"}
    return out


def _launch(name: str, out: Path, *, model: str, flags: list[str], ctx: int) -> Any:
    """The one place a model server starts (a seam: the smoke test swaps in callo on a fake upstream)."""
    return Callo(out / f"callo_{name}", model=model, ctx=ctx, flags=flags)


def synthetic_profile(fit: dict, ns: Sequence[int], path: Path) -> Path:
    """When the fit kept no n, B3 still needs a profile to apply. This one has the fit's model_key and
    provenance and an entry for every eval n (prior 0.6 at position A, the rest shared, so no n has all-zero
    offsets), so the serving path is exercised whatever the fit found. Written to `path`; B3 only, never served."""
    prof = {"callo_position_bias": 2, "estimator": "meanP", "model_key": fit.get("model_key"),
            "model_name": fit.get("model_name"), "provenance": fit.get("provenance") or {},
            "server_state": {"state": "clean", "checks": []},
            "set": {"name": "b3-synthetic", "builtin": False,
                    "kinds": (fit.get("set") or {}).get("kinds") or ["plain"]},
            "entries": {str(n): {"prior": [0.6] + [0.4 / (n - 1)] * (n - 1)} for n in ns}, "dropped": {}}
    path.write_text(json.dumps(prof, indent=1) + "\n")
    return path


def _sets() -> dict[str, str | Path | None]:
    return {"fit": FIT_DATA, "eval": EVAL_DATA}


def run_model(name: str, out: Path, prev: dict | None = None) -> dict:
    """One model's launch, fit, B0/B4 then B3 (TielCoder), eval read and judgement, in the pre-registration's order. `prev`:
    this model's partial result from an earlier session; its judged gates are kept. An exception after the
    launch is recorded as `error` next to every gate already judged, never instead of them; the watchdog's stop and a
    stop signal still propagate, and `_PARTIAL[name]` (this very result, filled in place) lets main keep the gates
    judged so far."""
    model, flags, ctx = MODELS[name]
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    busy = lib.running_gpu_servers()
    if busy:
        raise RuntimeError(f"a model server is already running, refusing to start another: {busy}")
    _progress(name, "launching")
    cache = d / f"callo_{name}" / "cache"     # where Callo's launch writes its llama-server pid file
    _CACHES.add(cache)
    try:
        c = _launch(name, d, model=model, flags=flags, ctx=ctx)
        if getattr(c, "pid", None) is not None:
            _OWN.add(c.pid)             # before the cache entry goes: never a moment where the server looks foreign
    finally:
        _CACHES.discard(cache)
    res: dict[str, Any] = {"model": model, "flags": flags, "build_info": getattr(c, "build_info", None)}
    prev = prev or {}
    _PARTIAL[name] = res
    reads: dict[str, dict] = {}
    try:
        try:
            res["kv"] = c.stats().get("kv")
            _progress(name, "fit")
            if name == "tiel_kvu":
                res.update(b0_b4(c, d, prev))
                fit_rc = (res.get("refit") or res["b0"])["rc"]
            else:
                fit_rc = res["fit_rc"] = calibrate(c, d, "fit", FIT_DATA)[0]
            if fit_rc != 0:             # nothing to judge held-out reads against; save the hour
                res["aborted"] = f"the fit exited {fit_rc}: held-out reads and B3 skipped"
            else:
                if name == "tiel_kvu" and judged(prev.get("b3")):
                    res["b3"] = prev["b3"]
                elif name == "tiel_kvu":
                    _progress(name, "B3")
                    fit = json.loads((d / "fit.json").read_text())
                    items = ba.items_of(EVAL_DATA)
                    entries = fit.get("entries") or {str(n): {} for n in sorted({it.n for it in items})}
                    profile = d / "fit.json" if fit.get("entries") else synthetic_profile(
                        fit, [int(n) for n in entries], d / "b3_synthetic.json")
                    calls = b3_call_sequence(c.upstream, model, profile, ba.b3_bodies(items, entries))
                    cpu = ba.b3_cpu()
                    res["b3"] = {**calls, **cpu, "synthetic": not fit.get("entries"),
                                 "pass": calls["pass_calls"] and cpu["pass"]}
                _progress(name, "eval read")
                reads["eval"] = read_heldout(c, d, "eval", EVAL_DATA)
                res["reads"] = reads
        except Exception as e:  # noqa: BLE001 -- keep every gate already judged
            if isinstance(e, Aborted) or (_WATCH is not None and _WATCH.reason is not None):
                raise
            log.exception("model %s failed after its launch", name)
            res["error"] = f"{type(e).__name__}: {e}"
            return res
    finally:
        c.finish()
        _OWN.discard(getattr(c, "pid", None))
    _progress(name, "analysing")
    try:
        res.update(ba.analyse(d, _sets(), reads))
    except Exception as e:  # noqa: BLE001 -- the gates above stay
        log.exception("model %s: analysis failed", name)
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def _stopped_by(e: BaseException) -> str:
    """What ended the run, for summary.json: the stop handlers raise SystemExit(128 + signum) (acceptance_v02), Ctrl-C
    without them is KeyboardInterrupt; any other SystemExit (argparse inside cli.main) is not a stop signal."""
    if isinstance(e, SystemExit):
        code = e.code if isinstance(e.code, int) else None
        if code is not None and code - 128 in (signal.SIGTERM, signal.SIGINT):
            return f"a stop signal ({signal.Signals(code - 128).name})"
        return f"an exit (SystemExit {e.code}, e.g. an argparse usage error inside cli.main), not a stop signal"
    return f"a stop signal ({type(e).__name__})"


def _keep_partial(summary: dict, why: str) -> None:
    """An interruption mid-model must not lose the gates judged so far (a judged gate is never re-run). The model in flight keeps its earlier result overlaid with this session's, and an `error` naming
    the interruption, so it is not complete and a rerun completes it with these gates as `prev`."""
    for name, res in list(_PARTIAL.items()):
        before = {k: v for k, v in (summary["models"].get(name) or {}).items() if k != "error"}
        summary["models"][name] = {**before, **res, "error": f"interrupted: {why}"}
    _PARTIAL.clear()


def main(argv: list[str] | None = None) -> int:
    global _WATCH, _PROGRESS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--min-free-gb", type=float, default=30.0, help="free RAM (vm_stat) needed to start")
    ap.add_argument("--expect-head", required=True,
                    help="the pre-registered commit C as a sha (7 to 40 hex digits; not HEAD or a branch): refuse to "
                         "run any other HEAD")
    ap.add_argument("--override-final", default=None, metavar="REASON",
                    help="run again although summary.json holds a final verdict; REASON and the previous verdict are "
                         "recorded in it (a deviation). Complete models are still kept as they are and judged gates "
                         "are never re-run, so on a complete summary this only re-applies prereg_verdict to the "
                         "recorded gates: to re-measure a gate, remove it (or its model) from summary.json first")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    names = [m for m in a.models.split(",") if m]
    if not names or any(m not in MODELS for m in names):
        print(f"unknown model(s) in {names}; choose from {','.join(MODELS)}")
        return 2
    unset = lib.unset_paths(("", FIT_DATA, lib.DATA_ENV), *(("", MODELS[m][0], lib.MODELS_ENV) for m in names),
                            ("", MASTER, lib.SERVER_ENV))
    if unset:
        print(unset)
        return 2
    busy = lib.running_gpu_servers()
    if busy:
        print(f"refusing to start: a model server is already running (one model server at a time; wait for it or ask "
              f"its owner): {busy}")
        return 2
    free = lib.free_memory_bytes()
    if a.min_free_gb > 0 and (free is None or free < a.min_free_gb * 1024**3):
        print(f"refusing to start: {free} bytes free (vm_stat) < --min-free-gb {a.min_free_gb}")
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    disk_paths = [*DISK_PATHS, str(out)]
    code = code_state(expect=a.expect_head)
    problems = (disk_problems(disk_paths, MIN_FREE_GIB) + pin_problems()
                + code_problems(code, a.expect_head, pinned_commit()))
    if problems:
        print("refusing to start: " + "; ".join(problems))
        return 2
    path = out / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"models": {}}
    final = (summary.get("prereg") or {}).get("verdict")
    if final in FINAL and not a.override_final:
        print(f"refusing to start: {path} already holds the final pre-registered verdict {final}; a rerun would "
              f"re-roll it. Pass --override-final REASON to run anyway: the reason is recorded")
        return 2
    if final in FINAL:
        summary.setdefault("overrides", []).append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                    "previous_verdict": final, "reason": a.override_final})
    previous = _install_stop_handlers()
    pid_file = out / "run.pid"
    summary.pop("aborted", None)
    summary.setdefault("sessions", []).append({"code": code, "expect_head": a.expect_head})
    summary.update(llama_server=MASTER, started=summary.get("started") or datetime.now(timezone.utc).isoformat(
        timespec="seconds"), free_memory_at_start=free, prereg={"sha256": _sha256(PREREG)}, code=code,
        data={"fit": {"path": str(FIT_DATA), "sha256": _sha256(FIT_DATA)},
              "eval": {"path": str(EVAL_DATA), "sha256": _sha256(EVAL_DATA)}})
    # A real SIGTERM interrupts a blocking socket read; interrupt_main is the fallback off the main thread, where no
    # stop handler is installed and SIGTERM would kill the process outright.
    interrupt = (lambda: os.kill(os.getpid(), signal.SIGTERM)) if previous else _thread.interrupt_main
    watch = _WATCH = Watchdog(disk_paths, interrupt=interrupt)
    _PROGRESS, _DONE[:] = out / "progress.json", []
    _PARTIAL.clear()
    try:
        pid_file.write_text(f"{os.getpid()}\n")
        try:
            watch.start()
            for name in names:
                prev = summary["models"].get(name)
                if model_complete(name, prev):
                    log.info("model %s: already judged in an earlier session, kept as it is", name)
                    _DONE.append(name)
                    continue
                log.info("model %s", name)
                try:
                    summary["models"][name] = run_model(name, out, prev)
                except Aborted:
                    raise
                except Exception as e:  # noqa: BLE001 -- one model's failure must not lose the others
                    if watch.reason is not None:
                        raise Aborted(watch.reason) from e
                    log.exception("model %s failed", name)
                    summary["models"][name] = {**(prev or {}), "error": f"{type(e).__name__}: {e}"}
                _PARTIAL.pop(name, None)
                _DONE.append(name)
                path.write_text(json.dumps(summary, indent=1, default=str))
            summary["g0_frames"] = {"pass": None, "note": "not run: no labelled image set with at least 24 canonical "
                                                          "questions per n exists, so no image-kind profile"}
            summary["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            summary["prereg"].update(ba.prereg_verdict(summary["models"]))
            path.write_text(json.dumps(summary, indent=1, default=str))
            _progress(None, "finished")
        finally:
            # Inside the try that maps a trip to exit 3: a trip on the last tick (its SIGTERM lands here) still
            # becomes ABORTED, and once join returns no tick can signal after the handlers are restored.
            watch.stop()
            watch.join(WATCH_JOIN_S)
    except (Aborted, SystemExit, KeyboardInterrupt) as e:
        _keep_partial(summary, watch.reason or _stopped_by(e))
        if watch.reason is None:
            path.write_text(json.dumps(summary, indent=1, default=str))
            raise                       # a person's Ctrl-C or kill: the stop handlers' usual exit
        summary["aborted"] = watch.reason
        path.write_text(json.dumps(summary, indent=1, default=str))
        _progress(None, "aborted")
        print(f"ABORTED: {watch.reason}. Every server this bench started is stopped. Rerun the same command once the "
              f"GPU and the disks are free: each calibrate run resumes from its raw file.")
        return 3
    finally:
        _WATCH, _PROGRESS = None, None
        pid_file.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    keys = ("b0", "b1", "b2", "b3", "b4", "b5")
    verdicts = {n: m["error"] if "error" in m else
                {k: "ERROR" if (m.get(k) or {}).get("error") else (m.get(k) or {}).get("pass") for k in keys}
                for n, m in summary["models"].items()}
    print(json.dumps(verdicts, indent=1))
    verdict = summary["prereg"].get("verdict")
    reasons = summary["prereg"].get("reasons") or []
    print(f"pre-registered verdict: {verdict}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    left_out = [m for m in ba.REQUIRED if m not in names]
    if left_out:
        print(f"required by the prereg but not in --models: {', '.join(left_out)}; the verdict cannot be judged "
              f"without them (exit 1)")
    failed = any(isinstance(v, str) or any(x is False or x == "ERROR" for x in v.values()) for v in verdicts.values())
    return 1 if failed or verdict in ("ERROR", "FALSIFIED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
