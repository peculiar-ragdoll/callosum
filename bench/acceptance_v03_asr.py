"""The v0.3 transcription acceptance bench: gate T1, live on Gemma 4 E4B.

One launch: `callo serve --model GEMMA --mmproj MMPROJ --no-kvu --ctx 8192 -- --cache-ram 0 --log-prompts-dir DIR`.
The 20 `say` utterances run sequentially on an idle server, each once through callo's translated
`/v1/audio/transcriptions` and once straight to the child llama-server's own endpoint, at temperature 0 with
`max_tokens` set. Every slot is erased before EVERY request (`POST /slots/N?action=erase`; launch mode passes
`--slot-save-path`).

T1 needs all of:
- the rendered prompts byte-identical, 20/20 (one new `--log-prompts-dir` file per request);
- `usage.input_tokens` equal, 20/20;
- `cached_tokens == 0` in both arms, 20/20 (the erase worked);
- `text` identical on >= 19/20, each mismatch reported with its WER;
- every translated `X-Callo-Slot` a non-reserved slot, 20/20.

Missing evidence is ERROR, never PASS. That covers:
- an utterance without a non-empty 200 transcript from both arms;
- an arm whose mean WER against the spoken sentences exceeds `WER_SANITY` (both arms agreeing on nonsense or on
  silence is not a pass);
- a broken condition: 20 translations, nothing passed through, no `/v1/systemone` readout during the run.

WER and the latency of both arms are reported descriptively.

Live-run hygiene, as the think bench does it:
- The run's pid is in `OUT/run.pid` for the whole run. A `run.pid` naming a live process refuses the start (exit 2),
  and the file is removed at exit only while it still names this process.
- Stop the live run ONLY with `kill -TERM $(cat OUT/run.pid)`. The stop handlers turn SIGTERM into `SystemExit`, so
  `finally` runs `Callo.finish()`, which stops callo and reaps its llama-server.
- `summary.json` records:
  - the commit and whether the tree is dirty;
  - `llama-server --version`;
  - the sha256 of the set's manifest, hashed from the very bytes that were parsed (each WAV is checked against it at
    load and again before it is posted).
  A missing commit or a failed `llama-server --version` is a recorded warning, in summary.json and the doc; the run
  goes on, as the think bench's does.
- SIGHUP stops the run too (this bench only: a closed terminal must not leave the server running), unless it is
  inherited as ignored: under `nohup` (with `.venv/bin/python` and no `uv` wrapper) hangups stay
  ignored. The first line on stderr says which (`hangups: ignored (nohup)` or `hangups: stop the run`), and
  summary.json records it as `run.sighup`. Whichever stop signal comes first, the other two are ignored from then on,
  so none can cut the cleanup short. A stop signal that lands as the launch returns still reaps its llama-server
  (`reap_cache_dir`, which signals a pid only while its command line names llama-server).
- A launch that fails ("callo did not start": Callo has reaped its own child) is ERROR with a summary (exit 1).
- A transport, erase or changed-WAV error mid-run ends it as ERROR (exit 1): summary.json and the doc name the item
  and the error, the rows so far are kept, and the server is stopped as on every other path.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import statistics
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import callo
from bench import lib
from bench.acceptance import MASTER, Callo, reap_cache_dir
from bench.acceptance_v02 import _stop_signal
from bench.acceptance_v03_think import _llama_server_version, _pid_in, code_state, pid_owner
from callo.upstream import find_orphans

GEMMA = lib.env_path(lib.MODELS_ENV, "unsloth", "Gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-UD-Q4_K_XL.gguf")
GEMMA_MMPROJ = lib.env_path(lib.MODELS_ENV, "unsloth", "Gemma-4-E4B-it-GGUF", "mmproj-BF16.gguf")
N_ITEMS = 20
TEXT_MIN = 19
WER_SANITY = 0.5                # an arm's mean WER against the spoken sentences above this: the model is not transcribing
CTX = 8192                      # per slot (--no-kvu); -np 4
MAX_TOKENS = "128"
FIELDS = {"temperature": "0", "max_tokens": MAX_TOKENS}
BOUNDARY = "callo-t1-0123456789abcdef"
ARMS = ("callo", "native")
CHECKS = (("prompts", "prompts byte-identical (`--log-prompts-dir`)", f"{N_ITEMS}/{N_ITEMS}"),
          ("input_tokens", "`usage.input_tokens` equal", f"{N_ITEMS}/{N_ITEMS}"),
          ("cached_tokens_zero", "`cached_tokens == 0` in both arms", f"{N_ITEMS}/{N_ITEMS}"),
          ("text", "`text` identical", f">= {TEXT_MIN}/{N_ITEMS}"),
          ("pinned", "`X-Callo-Slot` a non-reserved slot", f"{N_ITEMS}/{N_ITEMS}"))
_WORD = re.compile(r"[a-z0-9']+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
BENCH_MODULE = "acceptance_v03_asr"
NO_READOUTS = "/callo/stats has no s1.readouts: the /v1/systemone condition cannot be checked"
_pid_args = lib.pid_args                                # the seam the tests stub: `ps -o args= -p PID`


class ItemError(RuntimeError):
    """An utterance could not be measured: an erase that failed, or a WAV that changed during the run."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _nonempty_str(x: object) -> bool:
    return isinstance(x, str) and bool(x.strip())


def load_manifest(d: Path) -> tuple[dict, str]:
    """The set's manifest (bench/build_asr_set.py) and the sha256 of the very bytes parsed (read once), validated:
    an object listing N_ITEMS objects, each with a non-empty string `id` (unique) and `text`, a plain `file` name and a
    64-hex `sha256`; every WAV present and equal to its sha256. Anything else is a ValueError (main's exit 2)."""
    raw = (d / "manifest.json").read_bytes()
    m = json.loads(raw)
    if not isinstance(m, dict):
        raise ValueError(f"{d / 'manifest.json'} must be a JSON object")
    items = m.get("items")
    if not isinstance(items, list) or len(items) != N_ITEMS:
        raise ValueError(f"{d / 'manifest.json'} must list {N_ITEMS} items")
    for n, it in enumerate(items, 1):
        if not isinstance(it, dict):
            raise ValueError(f"item {n} is not an object")
        if not _nonempty_str(it.get("id")):
            raise ValueError(f"item {n}: `id` must be a non-empty string")
        where = f"item {n} ({it['id']})"
        if not _nonempty_str(it.get("text")):
            raise ValueError(f"{where}: `text` must be a non-empty string")
        if not (isinstance(it.get("sha256"), str) and _SHA256.fullmatch(it["sha256"])):
            raise ValueError(f"{where}: `sha256` must be 64 lower-case hex digits")
        f = it.get("file")
        if not (isinstance(f, str) and f and Path(f).name == f and f not in (".", "..")):
            raise ValueError(f"{where}: `file` must be a plain file name in the set's directory")
    dup = sorted({it["id"] for it in items if [j["id"] for j in items].count(it["id"]) > 1})
    if dup:
        raise ValueError(f"duplicate ids: {', '.join(dup)}")
    for it in items:
        f = d / it["file"]
        if not f.is_file():
            raise ValueError(f"{f} is missing")
        got = _sha(f.read_bytes())
        if got != it["sha256"]:
            raise ValueError(f"{f.name}: sha256 {got} does not match the manifest's {it['sha256']}")
    return m, _sha(raw)


def encode_form(fields: dict[str, str], audio: bytes, *, filename: str = "clip.wav",
                boundary: str = BOUNDARY) -> tuple[str, bytes]:
    """A CRLF multipart/form-data body as `curl -F file=@clip.wav -F k=v` sends it: the file part first."""
    b = boundary.encode()
    parts = [f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: audio/wav'
             f"\r\n\r\n".encode() + audio]
    parts += [f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode() + v.encode() for k, v in fields.items()]
    body = b"".join(b"--" + b + b"\r\n" + p + b"\r\n" for p in parts) + b"--" + b + b"--\r\n"
    return f"multipart/form-data; boundary={boundary}", body


def wer(ref: str, hyp: str) -> float:
    """Word error rate: word-level edit distance over the reference length (lower case, punctuation dropped)."""
    r, h = _WORD.findall(ref.lower()), _WORD.findall(hyp.lower())
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / max(1, len(r))


def post_form(url: str, content_type: str, body: bytes, *, timeout: float = 600) -> dict:
    """One multipart POST: status, the reply headers (lower case), the JSON body (None when not JSON), the wall ms."""
    u = urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    t0 = time.perf_counter()
    try:
        conn.request("POST", u.path, body=body, headers={"Content-Type": content_type})
        r = conn.getresponse()
        raw, status = r.read(), r.status
        headers = {k.lower(): v for k, v in r.getheaders()}
    finally:
        conn.close()
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = None
    return {"status": status, "headers": headers, "json": obj, "ms": 1000 * (time.perf_counter() - t0)}


def erase_all(upstream: str, n_slots: int) -> None:
    """Every slot of the child llama-server erased before every request."""
    for s in range(n_slots):
        status, _, body = lib.post(f"{upstream}/slots/{s}?action=erase", {})
        if status != 200:
            raise ItemError(f"erasing slot {s} on {upstream} returned HTTP {status}: {str(body)[:200]}")


def new_prompt(before: set[str], d: Path) -> tuple[str | None, int]:
    """The sha256 of the one prompt file `--log-prompts-dir` gained since `before`, and how many it gained (None
    unless exactly one: llama-server writes it at the start of the request, server-context.cpp:4475-4483)."""
    new = sorted(set(os.listdir(d)) - before)
    return (_sha((d / new[0]).read_bytes()) if len(new) == 1 else None), len(new)


def run_item(c: Any, item: dict, set_dir: Path, prompt_dir: Path, n_slots: int) -> list[dict]:
    """One utterance through both arms, callo first, each after erasing every slot. The WAV is checked against its
    manifest sha256 again before it is posted."""
    audio = (set_dir / item["file"]).read_bytes()
    if _sha(audio) != item["sha256"]:
        raise ItemError(f"{item['file']} changed during the run (sha256 no longer matches the manifest)")
    ctype, body = encode_form(FIELDS, audio)
    out = []
    for arm, url in (("callo", f"{c.url}/v1/audio/transcriptions"), ("native", f"{c.upstream}/v1/audio/transcriptions")):
        erase_all(c.upstream, n_slots)
        before = set(os.listdir(prompt_dir))
        r = post_form(url, ctype, body)
        digest, n_new = new_prompt(before, prompt_dir)
        j = r["json"] if isinstance(r["json"], dict) else {}
        usage = j.get("usage") if isinstance(j.get("usage"), dict) else {}
        details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
        text = j.get("text") if isinstance(j.get("text"), str) else None
        out.append({"id": item["id"], "arm": arm, "status": r["status"], "text": text,
                    "wer": wer(item["text"], text) if text is not None else None,
                    "input_tokens": usage.get("input_tokens"), "cached_tokens": details.get("cached_tokens"),
                    "slot": r["headers"].get("x-callo-slot"), "ms": round(r["ms"], 1),
                    "prompt_sha256": digest, "prompt_files": n_new})
    return out


def _evidence_missing(i: str, arm: str, r: dict | None) -> list[str]:
    """Why one arm's row of one utterance is no evidence: no non-empty 200 transcript, or a
    measurement the checks need that the instrument did not deliver (a FAIL would blame the translation for it)."""
    if r is None or r["status"] != 200 or not isinstance(r["text"], str) or not r["text"].strip():
        return [f"{i}: the {arm} arm has no non-empty 200 transcript"]
    why = []
    if r["prompt_files"] == 0:
        why.append(f"{i}: the {arm} arm logged no prompt file (--log-prompts-dir)")
    if r["input_tokens"] is None:
        why.append(f"{i}: the {arm} arm's reply has no usage.input_tokens")
    if r["cached_tokens"] is None:
        why.append(f"{i}: the {arm} arm's reply has no usage.input_tokens_details.cached_tokens")
    return why


def t1(rows: Sequence[dict], reserved: Sequence[int]) -> dict:
    """Gate T1. ERROR when the evidence is incomplete or empty: an utterance without a non-empty 200 transcript from
    both arms (`"" == ""` is no evidence), an arm that logged no prompt file or replied without the token counts,
    or an arm whose mean WER against the spoken text exceeds `WER_SANITY` (the checks are still reported). Else PASS
    iff every check passes."""
    by = {(r["id"], r["arm"]): r for r in rows}
    ids = sorted({r["id"] for r in rows})
    why = [] if len(ids) == N_ITEMS else [f"{len(ids)} utterances, expected {N_ITEMS}"]
    why += [w for i in ids for arm in ARMS for w in _evidence_missing(i, arm, by.get((i, arm)))]
    if why:
        return {"verdict": "ERROR", "why": why}
    pairs = [(by[(i, "callo")], by[(i, "native")]) for i in ids]
    n = {"prompts": sum(a["prompt_files"] == b["prompt_files"] == 1 and a["prompt_sha256"] == b["prompt_sha256"]
                        for a, b in pairs),
         "input_tokens": sum(a["input_tokens"] is not None and a["input_tokens"] == b["input_tokens"] for a, b in pairs),
         "cached_tokens_zero": sum(a["cached_tokens"] == 0 and b["cached_tokens"] == 0 for a, b in pairs),
         "text": sum(a["text"] == b["text"] for a, b in pairs),
         "pinned": sum(str(a["slot"]).isdigit() and int(a["slot"]) not in reserved for a, _ in pairs)}
    checks = {k: {"n": v, "pass": v >= (TEXT_MIN if k == "text" else N_ITEMS)} for k, v in n.items()}
    checks["text"]["mismatches"] = [{"id": a["id"], "callo": a["text"], "native": b["text"], "wer_callo": a["wer"],
                                     "wer_native": b["wer"], "wer_between": wer(b["text"], a["text"])}
                                    for a, b in pairs if a["text"] != b["text"]]
    insane = [f"the {arm} arm's mean WER against the spoken text is {m:.2f} > {WER_SANITY}: the model is not "
              f"transcribing what was said" for arm in ARMS
              if (m := statistics.fmean(by[(i, arm)]["wer"] for i in ids)) > WER_SANITY]
    if insane:
        return {"verdict": "ERROR", "why": insane, "checks": checks}
    return {"verdict": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL", "checks": checks}


def descriptive(rows: Sequence[dict]) -> dict:
    """Per arm: mean WER against the reference text, p50 and max latency (ms)."""
    out = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        wers, ms = [r["wer"] for r in rs if r["wer"] is not None], [r["ms"] for r in rs]
        out[arm] = {"wer_mean": sum(wers) / len(wers) if wers else None,
                    "ms_p50": statistics.median(ms) if ms else None, "ms_max": max(ms) if ms else None}
    return out


def preconditions(stats: dict) -> list[str]:
    """Why the launched server cannot run T1, or []: `/callo/stats.transcription` must say it
    translates. Its reason covers no audio, agent pinning off and a failed render check."""
    t = stats.get("transcription")
    if not isinstance(t, dict):
        return ["/callo/stats has no transcription block (a callo older than this bench?)"]
    return [] if t.get("translate") is True else [f"translation is off: {t.get('reason')}"]


def _readouts(stats: dict) -> int | None:
    s1 = stats.get("s1")
    n = s1.get("readouts") if isinstance(s1, dict) else None
    return n if isinstance(n, int) and not isinstance(n, bool) else None


def conditions_of(before: dict, after: dict) -> dict:
    """The run's conditions, from /callo/stats before the first and after the last request: every count is a delta
    over the run. A missing `s1.readouts` is a problem with a null delta (main: ERROR)."""
    counters, was = after.get("counters", {}), before.get("counters", {})
    translated = counters.get("transcription_translated", 0) - was.get("transcription_translated", 0)
    unrouted = {k: v - was.get(k, 0) for k, v in counters.items()
                if k.startswith("transcription_unrouted_") and v != was.get(k, 0)}
    r0, r1 = _readouts(before), _readouts(after)
    delta = None if r0 is None or r1 is None else r1 - r0
    problems = ([f"{translated} translated transcriptions, expected {N_ITEMS}"] if translated != N_ITEMS else []) + \
               ([f"transcriptions passed through: {unrouted}"] if unrouted else []) + \
               ([NO_READOUTS] if delta is None else [f"/v1/systemone readouts during the run: +{delta}"] if delta else [])
    return {"translated": translated, "unrouted": unrouted, "s1_readouts_delta": delta, "ok": not problems,
            "problems": problems}


def _f(x: float | None, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def _cell(s: object) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_md(s: dict) -> str:
    """`docs/bench/v0.3-asr.md` from summary.json (a run that stopped on an error has `None` conditions numbers)."""
    run, g, cond = s["run"], s["t1"], s["conditions"]
    tree = "dirty" if run["dirty"] else "clean" if run["dirty"] is False else "unknown"
    lines = ["# callo v0.3: pinned `/v1/audio/transcriptions` (gate T1)", "",
             f"Run {run['started']} to {run['finished']}: callo {run['callo']} (`{run['commit']}` ({tree})), "
             f"llama.cpp `{run['build_info']}` (`{_cell(run['llama_server_version'])}`), `{Path(run['model']).name}` + "
             f"`{Path(run['mmproj']).name}`. Launch: `-np {run['n_slots']}`, reserved slots {run['reserved']}, per-slot "
             f"KV (`--no-kvu --ctx {CTX}`), llama-server `--cache-ram 0 --log-prompts-dir`. Every slot was erased before "
             f"every request, at temperature 0 with `max_tokens` {MAX_TOKENS}, sequentially on an idle server. "
             f"Utterances: {N_ITEMS} ({_cell(run['synth'])}), manifest sha256 `{run['manifest_sha256']}`. The LFM2 "
             f"preset branch is unit-tested only: Gemma uses the default preset.", ""]
    if s.get("warnings"):
        lines += ["Warnings:", ""] + [f"- {_cell(w)}" for w in s["warnings"]] + [""]
    lines += [f"**T1: {g['verdict']}**", ""]
    lines += [f"- {_cell(w)}" for w in g.get("why", [])] + ([""] if g.get("why") else [])
    if "checks" in g:
        lines += ["| check | result | pass when | verdict |", "|---|---|---|---|"]
        for key, label, when in CHECKS:
            c = g["checks"][key]
            lines.append(f"| {label} | {c['n']}/{N_ITEMS} | {when} | {'PASS' if c['pass'] else 'FAIL'} |")
        lines.append("")
        mism = g["checks"]["text"]["mismatches"]
        if mism:
            lines += ["## Text mismatches", "", "| id | callo | native | WER callo / native / between |",
                      "|---|---|---|---|"]
            lines += [f"| {m['id']} | {_cell(m['callo'])} | {_cell(m['native'])} | {_f(m['wer_callo'], '.3f')} / "
                      f"{_f(m['wer_native'], '.3f')} / {_f(m['wer_between'], '.3f')} |" for m in mism] + [""]
    lines += ["## Conditions", "", f"- translated: {_f(cond['translated'], 'd')}/{N_ITEMS}; passed through: "
              f"{_cell(cond['unrouted']) if cond['unrouted'] else 'none'}; `/v1/systemone` readouts during the run: "
              f"{_f(cond['s1_readouts_delta'], 'd')}", ""]
    lines += [f"- {_cell(p)}" for p in cond["problems"] if cond["translated"] is None] + \
             ([""] if cond["translated"] is None else [])
    lines += ["## Descriptive", "", f"| arm | mean WER vs the spoken text (sanity bound {WER_SANITY}) | p50 ms | max ms |",
              "|---|---|---|---|"]
    lines += [f"| {arm} | {_f(d['wer_mean'], '.3f')} | {_f(d['ms_p50'], '.0f')} | {_f(d['ms_max'], '.0f')} |"
              for arm, d in s["descriptive"].items()]
    return "\n".join(lines) + "\n"


def is_bench_run(pid: int) -> bool:
    """The live pid in run.pid is another run of THIS bench (`lib.is_bench_run`, the think bench's run.pid rule)."""
    return lib.is_bench_run(pid, BENCH_MODULE, pid_args=_pid_args)


def _stop_run(signum: int, frame: object) -> None:
    """The one stop handler, for SIGTERM, SIGINT and SIGHUP (a closed terminal): acceptance_v02's `_stop_signal`, which
    raises SystemExit and ignores TERM and INT from then on, after ignoring SIGHUP too: whichever comes
    first, no second signal can cut the cleanup short."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    _stop_signal(signum, frame)


def hangups() -> str:
    """What a SIGHUP will do to this run, from the disposition this process inherited: under `nohup`
    it is ignored, else it stops the run. main prints it at startup and records it in summary.json."""
    return "ignored (nohup)" if signal.getsignal(signal.SIGHUP) is signal.SIG_IGN else "stop the run"


def _install_stop_handlers_with_hup() -> dict[int, Any]:
    """This bench's stop handlers (the shared `acceptance_v02._install_stop_handlers` is unchanged): `_stop_run` for
    TERM and INT, and for SIGHUP unless it is inherited as ignored (a `nohup` run keeps ignoring
    hangups). The previous handlers, to restore; empty off the main thread, where none can be set."""
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous = {sig: signal.signal(sig, _stop_run) for sig in (signal.SIGTERM, signal.SIGINT)}
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        previous[signal.SIGHUP] = signal.signal(signal.SIGHUP, _stop_run)
    return previous


def provenance_warnings(code: dict, version: str) -> list[str]:
    """Weak provenance, recorded and not fatal (the think bench's parity): no commit, or no `llama-server --version`."""
    out = []
    if not code.get("commit"):
        out.append(f"the commit is unknown ({code.get('error') or 'git gave none'}): the code this run measured is not "
                   f"pinned")
    if not version.strip() or version.startswith("unavailable:"):
        out.append(f"`llama-server --version` failed: {version.strip() or 'no output'}")
    return out


def _cache_dir(out: Path) -> Path:
    """Where `_launch`'s Callo writes its llama-server pid files (`Callo.cache_dir`)."""
    return out / "callo" / "cache"


def _launch(out: Path, *, model: str, mmproj: str, llama_server: str, prompt_dir: Path) -> Any:
    """The one place the model server starts (a seam: the smoke test swaps in callo attached to a fake upstream)."""
    return Callo(out / "callo", llama_server=llama_server, ctx=CTX, parallel=4, model=model,
                 flags=["--mmproj", mmproj, "--no-kvu", "--", "--cache-ram", "0", "--log-prompts-dir", str(prompt_dir)])


def main(argv: list[str] | None = None) -> int:
    """Refusals first (exit 2: a bad set, a busy or unknowable GPU, a live run.pid, a used prompt dir), then the run
    under the stop handlers with run.pid holding this process's pid. A stop signal ends in SystemExit after `finally`
    stopped the server and cleared run.pid."""
    ap = argparse.ArgumentParser(prog="acceptance_v03_asr", description="Gate T1: pinned /v1/audio/transcriptions. "
                                 "GPU: one launch of Gemma 4 E4B. Stop it ONLY with kill -TERM $(cat OUT/run.pid).")
    ap.add_argument("--set", type=Path, required=True, help="the utterance set (bench/build_asr_set.py)")
    ap.add_argument("--out", type=Path, required=True, help="raw results: run.pid, rows.jsonl, summary.json, prompts/")
    ap.add_argument("--doc", type=Path, default=Path("docs/bench/v0.3-asr.md"))
    ap.add_argument("--model", default=GEMMA)
    ap.add_argument("--mmproj", default=GEMMA_MMPROJ)
    ap.add_argument("--llama-server", default=MASTER)
    a = ap.parse_args(argv)
    unset = lib.unset_paths(("--model", a.model, lib.MODELS_ENV), ("--mmproj", a.mmproj, lib.MODELS_ENV),
                            ("--llama-server", a.llama_server, lib.SERVER_ENV))
    if unset:
        print(unset, file=sys.stderr)
        return 2
    try:
        manifest, manifest_sha256 = load_manifest(a.set)
    except (OSError, ValueError) as e:
        print(f"refusing to start: the set {a.set}: {e}", file=sys.stderr)
        return 2
    try:
        busy = lib.running_gpu_servers()
    except RuntimeError as e:                        # a ps that fails is a refusal, never a traceback
        print(f"refusing to start: {e}: cannot tell whether a model server holds the GPU", file=sys.stderr)
        return 2
    if busy:
        print(f"refusing to start: a model server holds the GPU (one at a time; wait for the window): {busy}",
              file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    pid_file = a.out / "run.pid"
    owner = pid_owner(pid_file)
    if owner is not None:
        if is_bench_run(owner):
            print(f"refusing to start: another run of this bench (pid {owner}) owns this --out ({pid_file}); stop it "
                  f"through its run.pid or wait for it to finish", file=sys.stderr)
        else:                                        # never suggest signalling a pid that may be anything
            print(f"refusing to start: run.pid names pid {owner}, which is not a bench run: check it and remove "
                  f"run.pid by hand", file=sys.stderr)
        return 2
    prompt_dir = a.out / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    if any(prompt_dir.iterdir()):
        print(f"refusing to start: {prompt_dir} is not empty (one run per --out)", file=sys.stderr)
        return 2
    sighup = hangups()                               # before the handlers change it
    print(f"hangups: {sighup}", file=sys.stderr)
    previous = _install_stop_handlers_with_hup()     # SIGTERM/INT/HUP -> SystemExit, so every `finally` below runs
    me = os.getpid()
    try:
        pid_file.write_text(f"{me}\n")              # a dead pid's stale file is replaced
        return _run(a, manifest, manifest_sha256, prompt_dir, sighup)
    finally:
        if _pid_in(pid_file) == me:                  # never remove a file another run wrote over ours
            pid_file.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


# An utterance (or the closing stats) that could not be measured: never BaseException, so a stop signal's SystemExit
# and a bug's traceback still propagate.
RUN_ERRORS = (ItemError, OSError, http.client.HTTPException, json.JSONDecodeError)


class _LaunchFailed(Exception):
    """`_launch` raised RuntimeError ("callo did not start"; Callo has reaped its own child): caught around the launch
    only, so a RuntimeError from a bug in the run itself still propagates."""


def _run(a: argparse.Namespace, manifest: dict, manifest_sha256: str, prompt_dir: Path, sighup: str) -> int:
    items = manifest["items"]
    code = code_state()                              # the commit and the dirty flag
    version = _llama_server_version(a.llama_server)  # `llama-server --version` (a seam the tests stub)
    warnings = provenance_warnings(code, version)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    preexisting = {pid for pid, _ in find_orphans(_cache_dir(a.out))}
    c: Any = None
    rows: list[dict] = []
    before: dict = {}
    after: dict | None = None
    stopped: str | None = None
    at = "the launch"
    try:
        try:
            c = _launch(a.out, model=a.model, mmproj=a.mmproj, llama_server=a.llama_server, prompt_dir=prompt_dir)
        except RuntimeError as e:
            raise _LaunchFailed(f"{type(e).__name__}: {e}") from e
        at = "the opening /callo/stats"
        before = c.stats()
        problems = preconditions(before)
        if problems:
            print("refusing to run: " + "; ".join(problems), file=sys.stderr)
            return 2
        for item in items:
            at = item["id"]
            rows += run_item(c, item, a.set, prompt_dir, before["total_slots"])
            (a.out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        at = "the closing /callo/stats"
        after = c.stats()
    except _LaunchFailed as e:
        stopped = f"{at}: {e}"
    except RUN_ERRORS as e:                          # an ERROR summary naming the item, never a traceback
        stopped = f"{at}: {type(e).__name__}: {e}"
        print(f"stopped: {stopped}", file=sys.stderr)
    finally:
        if c is not None:
            c.finish()                               # on a stop signal too: it stops callo and reaps llama-server
        else:                                        # a signal as `_launch` returned lost the Callo
            reap_cache_dir(_cache_dir(a.out), preexisting)
    router = before.get("router")
    reserved = router.get("reserved") if isinstance(router, dict) else None
    if stopped is not None:
        reserved = reserved if isinstance(reserved, list) else []      # not measured: nothing is judged against it
        conditions = {"translated": None, "unrouted": {}, "s1_readouts_delta": None, "ok": False,
                      "problems": [f"not measured: the run stopped at {at}"]}
        gate = {"verdict": "ERROR", "why": [stopped, f"{len(rows) // len(ARMS)} of {N_ITEMS} utterances ran before it"]}
    else:
        assert after is not None
        conditions = conditions_of(before, after)
        gate = t1(rows, reserved) if isinstance(reserved, list) else {"verdict": "ERROR", "why": [
            "/callo/stats has no router.reserved list: a reserved slot cannot be told from an agent slot"]}
        if (gate["verdict"] == "PASS" and not conditions["ok"]) or conditions["s1_readouts_delta"] is None:
            gate = {**gate, "verdict": "ERROR", "why": gate.get("why", []) + conditions["problems"]}   # no readouts
    summary = {"run": {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "model": a.model, "mmproj": a.mmproj, "llama_server": a.llama_server,
                       "llama_server_version": version, "build_info": getattr(c, "build_info", None),
                       "callo": callo.__version__, "commit": code.get("commit"), "dirty": code.get("dirty"),
                       "n_slots": before.get("total_slots"), "reserved": reserved,
                       "kv": before.get("kv"), "set": str(a.set), "manifest_sha256": manifest_sha256, "sighup": sighup,
                       "synth": {k: v for k, v in manifest.items() if k != "items"}},
               "warnings": warnings, "conditions": conditions, "t1": gate, "descriptive": descriptive(rows)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    a.doc.parent.mkdir(parents=True, exist_ok=True)
    a.doc.write_text(render_md(summary))
    print(f"T1: {gate['verdict']} ({a.doc})")
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
