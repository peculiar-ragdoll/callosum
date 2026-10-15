"""`callo serve | classify | calibrate | version`."""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import signal
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import __version__, calibrate_bias, calibrate_think, compare, menu, transcribe
from .av import audio_self_test, video_self_test
from .calibrate import (
    GRID_TEXT,
    ece,
    fit_temperature,
    flat_fit,
    limit_note,
    nll,
    request_from_record,
    row_from_answer,
)
from .converge import ServeRules, serve_rules
from .deadline import MAX_DEADLINE_MS, Estimators, deadline_cap
from .engine import (
    SELF_TEST_BODY,
    Engine,
    GgufMeta,
    SurfaceSets,
    load_gguf_meta,
    make_engine,
    self_test,
    think_self_test,
)
from .errors import VIDEO_OFF_DETAIL, CalloError, InvalidInput
from .host import host_info
from .media import detect_media
from .modelkey import gguf_bytes, load_model_key
from .profile import PositionBias, check_identity, check_profile, load_profile, provenance
from .prompt import TemplateClient, messages, user_delimiter
from .proxy import (
    DEFAULT_MAX_ROUTED_BODY,
    ProxyConfig,
    RequestLog,
    Stats,
    SystemOneService,
    ThinkProbe,
    _is_loopback,
    make_server,
    s1_timeout_problem,
    set_config,
)
from .scorer import LlamaCppForwardScorer, make_http_post
from .slots import DEFAULT_ASR_HOME_IDLE, SlotRouter, asr_home_idle_problem
from .systemone import MAX_LETTERS, MAX_MEDIA, WIDE_LABELS, parse_request
from .think import (
    MAX_BUDGET,
    PROFILE_T_FIX,
    GGUFSampling,
    SamplingPolicy,
    ThinkProfile,
    check_think_identity,
    check_think_profile,
    frame_from_template,
    load_think_profile,
    parse_sampling,
    read_gguf_sampling,
    resolve_sampling,
    think_default_refusal,
    unknown_props_keys,
    unpinnable,
)
from .upstream import (
    LlamaServerProcess,
    ProbeResult,
    SlotsCache,
    build_argv,
    check_passthrough_args,
    decide_agent_pinning,
    default_cache_dir,
    discover,
    find_orphans,
    make_http_get,
    missing_props,
    probe_idle_clear,
    resolve_gguf,
    version_warning,
    wait_health,
)
from .vision import SELF_TEST_IMAGE_URL, vision_self_test

DEFAULT_PORT = 8180
DEFAULT_THINK_MAX = 4096
GRID_STEP = 1.024               # one step of calibrate's log grid (16^(1/120) = 1.0234): the --temperature / T_0 mismatch tolerance
HOST_INFO_WAIT = 5.0            # seconds startup waits for host.host_info() before going on without it
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
               ".gif": "image/gif", ".bmp": "image/bmp"}
# fixed tables, never `mimetypes` (platform tables differ, e.g. for .mkv). llama.cpp sniffs the bytes anyway;
# the MIME type only has to be in the right family.
AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}
VIDEO_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
               ".avi": "video/x-msvideo"}
CLASSIFY_MAX_BODY = DEFAULT_MAX_ROUTED_BODY     # /v1/systemone answers 413 above this (64 MiB)
LOOPBACK = ("127.0.0.1", "localhost", "::1")
log = logging.getLogger("callo")


def _int_flag(flag: str, value: object) -> int:
    """An integer flag parsed in phase 1, so `--think-budget abc` is a refusal (exit 1) like the
    other phase-1 faults, not argparse's usage error."""
    try:
        return int(str(value).strip(), 10)
    except ValueError:
        raise CalloError(f"{flag} must be an integer, got {value!r}") from None


def _converge_flag(value: object) -> float:
    """--think-converge Z in phase 1: a number from 0 (off) to below 1, else a refusal (exit 1)."""
    try:
        z = float(str(value).strip())
    except ValueError:
        z = float("nan")
    if not 0 <= z < 1:                  # nan and inf fail the comparison
        raise CalloError(f"--think-converge must be a number from 0 (off) to below 1, got {value!r}")
    return z if z != 0 else 0.0         # -0 is off, published as 0.0


def _non_empty(omit: str) -> Callable[[str], str]:
    """An argparse `type` refusing an explicitly empty value at parse time (exit 2): `--user-delimiter ""` is a
    mistake, not a request to derive, and `--mmproj ""` is a mistake, not a request for no projector."""
    def check(value: str) -> str:
        if not value:
            raise argparse.ArgumentTypeError(f"must not be empty; {omit}")
        return value
    return check


class OnOff(argparse.BooleanOptionalAction):
    """`--x` / `--no-x` without Python 3.10-3.12's appended " (default: ...)": the help says the default in words
    (3.13 dropped the suffix)."""

    def __init__(self, option_strings: Sequence[str], dest: str, **kw: Any) -> None:
        help_text = kw.get("help")
        super().__init__(option_strings, dest, **kw)
        self.help = help_text


class _Given(argparse.Action):
    """Store the value and record that the flag was GIVEN (`<dest>_given`), whatever its value: `--url` given with
    its default's value is still a second source beside `--model` / `--menu`."""

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any,
                 option_string: str | None = None) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_given", True)


def add_preference_args(p: argparse.ArgumentParser, *, prefix: str = "") -> None:
    """The pick preference flags calibrate (`prefix` "--think-budgets: ") and compare share, validated by one
    function, `think.preference_from_args`. Every default is None, so calibrate can tell which flags were given; every
    example is in the canonical, shell-safe STAT=X form."""
    p.add_argument("--latency", default=None, metavar="mean|p50|p95",
                   help=f"{prefix}the pick statistic: the frontier axis, the order \"fastest\" follows and what --price "
                        f"charges (default mean; p99 and max are limits only)")
    p.add_argument("--limit", action="append", default=None, metavar="STAT=X",
                   help=f"{prefix}an upper bound on a latency statistic, repeatable: STAT is mean, p50, p95, p99 or "
                        f"max, X seconds (6, 6s) or milliseconds (6000ms), for example --limit p99=6s (the quoted "
                        f"'p99<=6s' is an alias)")
    p.add_argument("--price", default=None, metavar="R",
                   help=f"{prefix}what one more second per question costs, in accuracy points: R pp/s (1, 1pp/s or "
                        f"0.5pp/s; default 0). The target is the point maximising 100*acc - R*latency on the pick "
                        f"statistic, and the pick is the fastest point within noise of it; a positive --price excludes "
                        f"--min-accuracy")
    p.add_argument("--max-latency", type=float, default=None, metavar="S",
                   help=f"{prefix}alias for --limit STAT=S on the pick statistic (--latency)")
    p.add_argument("--min-accuracy", type=float, default=None, metavar="A",
                   help=f"{prefix}pick the fastest point at or above accuracy A (a fraction)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="callo", description="System One classification from the model you already serve")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="the front door: llama-server proxy with slot routing + POST /v1/systemone")
    mode = s.add_mutually_exclusive_group(required=True)
    mode.add_argument("--upstream", help="attach to a running llama-server, e.g. http://127.0.0.1:8199")
    mode.add_argument("--model", help="launch llama-server on this GGUF (launch mode; see --llama-server)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--s1-slots", type=int, default=1, help="reserved classification slots 0..k-1 (k < total_slots)")
    s.add_argument("--temperature", type=float, default=None,
                   help="applied to the log-marginals at budget 0 (see `callo calibrate`); default: the think profile's "
                        "T_0 when --think-profile is given, else 1.0")
    s.add_argument("--cache", action=OnOff, default=True,
                   help="on by default: cache_prompt: true, with the user-turn delimiter, on plain classification "
                        "passes (media and context passes are always cached), so a hybrid model reuses the system "
                        "turn from the user-turn checkpoint (measured on a 35B-A3B MoE: -24%% wall per readout, a "
                        "warm readout identical to a cold one). --no-cache turns it off (cache_prompt: false: every pass re-encodes the prompt)")
    s.add_argument("--s1-timeout", type=float, default=900.0, help="seconds per upstream call for classification")
    s.add_argument("--asr-home-idle", type=float, default=DEFAULT_ASR_HOME_IDLE, metavar="SECONDS",
                   help="a hint-less transcription may make its home on an agent slot whose conversation hint has been "
                        "unused this long (0: any hint not in use); default %(default)s")
    s.add_argument("--s1-fallback", choices=("on", "off"), default="on",
                   help="for benchmarking (leave it on in production): 'off' keeps classification on its reserved slot even when a foreign task holds it")
    s.add_argument("--gguf", help="the served GGUF, when /props.model_path is not readable from here")
    s.add_argument("--upstream-api-key", default=None, help="key for callo's own upstream calls (default: $LLAMA_API_KEY)")
    s.add_argument("--ctx", type=int, default=None, help="launch mode: -c for llama-server (default: llama-server's own)")
    s.add_argument("--parallel", type=int, default=4, help="launch mode: -np")
    s.add_argument("--no-kvu", action="store_true",
                   help="launch mode: omit -kvu so each slot has its own KV; --ctx is then the per-slot context "
                        "(-c is --ctx x --parallel). Removes the cross-slot readout dependence measured on text "
                        "prompts (see README caveats)")
    s.add_argument("--llama-server", default="llama-server", help="launch mode: the binary to run")
    s.add_argument("--mmproj", default=None, type=_non_empty("omit --mmproj to launch without a projector"),
                   help="launch mode: the multimodal projector GGUF, vision and/or audio (llama-server --mmproj)")
    s.add_argument("--user-delimiter", default=None,
                   type=_non_empty("omit --user-delimiter to derive it from the chat template"),
                   help="pin the user-turn delimiter instead of deriving it from the chat template (still validated)")
    s.add_argument("--position-bias", default=None, metavar="PROFILE",
                   type=_non_empty("omit --position-bias to serve raw readouts"),
                   help="debias eligible readouts with a profile from `callo calibrate --bias` (off by default)")
    s.add_argument("--think-budget", default="0", metavar="N",
                   help="think budgets: the default budget of every text request that sets no think_budget "
                        "(default 0: off)")
    s.add_argument("--think-max", default=str(DEFAULT_THINK_MAX), metavar="N",
                   help=f"the largest think_budget a request may ask for (default {DEFAULT_THINK_MAX}; at least "
                        f"--think-budget; the hard cap is {MAX_BUDGET})")
    s.add_argument("--think-profile", default=None, metavar="PROFILE",
                   type=_non_empty("omit --think-profile to serve deliberate readouts at T = 1"),
                   help="a think profile from `callo calibrate --think-budgets`: its sampling, its T_B for budgets > 0 "
                        "and, without --temperature, its T_0 for budget 0")
    s.add_argument("--think-sampling", default=None, metavar="model|fixed|SPEC",
                   type=_non_empty("omit --think-sampling for each GGUF's own sampler"),
                   help="the trace sampler without --think-profile: model (default: the GGUF's general.sampling.*, "
                        "callo's 0.6/0.95/20/0 per missing field), fixed (callo's for every field) or "
                        "temp=..,top_p=..,top_k=..,min_p=..; a profile's sampling is what was calibrated, so the two "
                        "do not combine")
    s.add_argument("--allow-think-probe", action="store_true",
                   help="answer POST /callo/think-probe (calibrate's internal endpoint) from non-loopback peers too")
    s.add_argument("--think-deadline", default="0", metavar="MS",
                   help=f"think deadlines: the deadline of every text request that sets no think_deadline_ms "
                        f"(default 0: off; 0..{MAX_DEADLINE_MS}). It applies to EVERY client: a request without "
                        f"fields then thinks under it; think_budget 0 opts a request out")
    s.add_argument("--think-deadline-reserve", default=None, metavar="MS",
                   help="fix the readout reserve a deadline trace keeps for its answer (default: learned from the "
                        "served readouts, seeded from the think profile)")
    s.add_argument("--think-converge", default="0", metavar="Z",
                   help="early exit: stop thinking once the answer has settled, at the think profile's rule for target "
                        "Z (default 0: off; 0 <= Z < 1; .9, .95 and .99 are the calibrated targets). It applies to EVERY "
                        "text request that sets no think_converge; needs --think-profile from `callo calibrate "
                        "--converge-probe`; think_budget 0 opts a request out")
    s.add_argument("--think-converge-gate", action="store_true",
                   help="early exit's gate (experimental): before a request that early-exits thinks, ask the model one "
                        "budget-0 question (would it need to apply several stated rules step by step?); a yes thinks "
                        "to the cap without early exit. It applies to every request that sets no "
                        "think_converge_gate; think_converge_gate false opts a request out")
    s.add_argument("--request-log", default=None, metavar="FILE",
                   type=_non_empty("omit --request-log to log nothing"),
                   help="append one JSON line per /v1/systemone request (its body's sha256, timings, slot, limits and "
                        "answers; never its text) after the reply is written. Covers only requests callo serves "
                        "(not a native decision model's)")
    s.add_argument("--cache-dir", default=None, help=argparse.SUPPRESS)
    s.epilog = "Launch mode: arguments after `--` are passed to llama-server after callo's own flags (last wins)."

    c = sub.add_parser("classify", help="a thin client of POST /v1/systemone: state on stdin")
    c.add_argument("--url", default=f"http://127.0.0.1:{DEFAULT_PORT}")
    c.add_argument("--options", required=True, help="comma-separated option names")
    c.add_argument("--instructions", default="")
    c.add_argument("--json", action="store_true")
    cap = f"at most {MAX_MEDIA} media files in all (--image, --audio and --video together)"
    c.add_argument("--image", action="append", default=[], metavar="PATH",
                   help=f"an image ({', '.join(IMAGE_TYPES)}), sent as a data URL; repeatable; {cap}")
    c.add_argument("--audio", action="append", default=[], metavar="PATH",
                   help=f"an audio clip ({', '.join(AUDIO_TYPES)}), sent as a data URL; repeatable; "
                        f"{cap}")
    c.add_argument("--video", action="append", default=[], metavar="PATH",
                   help=f"a video ({', '.join(VIDEO_TYPES)}), sent as a data URL; repeatable; {cap}")
    c.add_argument("--context", default=None, help="text for the cached system turn (a callo extension)")

    k = sub.add_parser("calibrate", help="fit one temperature by NLL on labelled System One requests (jsonl), or with "
                                         "--bias measure the position-bias profile; the server must run at "
                                         "--temperature 1.0 (checked via /callo/stats)")
    k.add_argument("--url", default=f"http://127.0.0.1:{DEFAULT_PORT}", action=_Given,
                   help="the running `callo serve` to measure (default %(default)s); --think-budgets: one source, "
                        "exclusive with --upstream, --model and --menu (given explicitly, even with the default value)")
    k.add_argument("--data", default=None,
                   help="labelled jsonl: state, questions{..., label}, _meta. Required for the temperature fit; "
                        "with --bias, your own questions (labels optional): a profile is fitted only on --data, and "
                        "without it --bias reads the built-in set builtin-v1 as a diagnostic and writes no profile")
    k.add_argument("--out",
                   type=_non_empty("omit --out to write no file (with --bias --data the profile then goes to "
                                   "stdout; --bias alone runs the built-in diagnostic)"),
                   help="temperature: write {temperature, n, nll/ece before/after} as JSON (summary on stdout); "
                        "--bias (needs --data): the profile path (without it the profile goes to stdout and the "
                        "summary to stderr)")
    k.add_argument("--bias", action="store_true",
                   help="measure the position-bias profile (label-free: the mean served probability per position "
                        "over every rotation)")
    k.add_argument("--raw", default=None, type=_non_empty("omit --raw for the default path"),
                   help="--bias: every readout as it arrives (default: --out with .raw.jsonl, or "
                        "./callo-bias.raw.jsonl; ./callo-bias-diagnostic.raw.jsonl for the built-in diagnostic)")
    k.add_argument("--resume", action="store_true",
                   help="--bias: continue the run recorded in --raw; --think-budgets: continue the run in --out (one "
                        "server: its raw file; a menu: run.json, where an appended entry runs alone)")
    k.add_argument("--min-questions", type=int, default=None,
                   help="--bias: questions an n needs, read under every rotation, to be tested (default 24)")
    k.add_argument("--max-per-n", type=int, default=None,
                   help="--bias: questions per n, source-stratified (default: every eligible question of --data; 48 "
                        "on the built-in diagnostic)")
    k.add_argument("--seed", type=int, default=None,
                   help="--bias: order, cap and test seed (default 0); --think-budgets: the item order, trace seeds, "
                        "bootstrap and cross-fit folds (default 0)")
    k.add_argument("--allow-busy", action="store_true",
                   help="--bias and --think-budgets: record a failed server-state check and carry on (the profile is "
                        "marked busy)")

    k.add_argument("--think-budgets", nargs="?", const=",".join(map(str, calibrate_think.DEFAULT_BUDGETS)),
                   default=None, metavar="B,B,...",
                   help=f"measure the deliberate readout at these think budgets and write the think profile "
                        f"`callo serve --think-profile` loads; no value means {','.join(map(str, calibrate_think.DEFAULT_BUDGETS))}; "
                        f"up to {calibrate_think.MAX_BUDGETS} distinct budgets 0..{MAX_BUDGET}, 0 is added when missing; "
                        f"exclusive with --bias; shares --url (or the launch sources --model / --menu), --data, "
                        f"--resume, --seed and --allow-busy, and --out is then a DIRECTORY (default "
                        f"./{calibrate_think.DEFAULT_OUT_DIR})")
    k.add_argument("--upstream", default=None, metavar="URL",
                   help="--think-budgets: measure a bare llama-server through an in-process `callo serve --upstream URL "
                        "--think-budget 0` instead of --url")
    k.add_argument("--deadlines", default=None, metavar="MS,MS,...",
                   help="--think-budgets: up to 4 deadline points (ms): each question's largest-budget trace is cut "
                        "where `callo serve --think-deadline MS` would cut it, single-stream, and read there")
    k.add_argument("--think-deadline-reserve", type=int, default=None, metavar="MS",
                   help="--deadlines: the readout reserve the cuts are simulated with (default: measured on the "
                        "warm-up questions)")
    k.add_argument("--converge-probe", nargs="?", type=int, const=calibrate_think.DEFAULT_CONVERGE_K, default=None,
                   metavar="K", help=f"--think-budgets: read a probe every K tokens of each largest-budget trace (32, "
                                     f"64 or 128; no value means {calibrate_think.DEFAULT_CONVERGE_K}) and fit the "
                                     f"early-exit rules, which `callo serve --think-converge` serves")
    k.add_argument("--converge-cost", type=int, default=None, metavar="N",
                   help=f"--converge-probe: questions regenerated live to measure a probe's serve-time cost (default "
                        f"{calibrate_think.DEFAULT_CONVERGE_COST}; 0 skips it)")
    k.add_argument("--warmup", type=int, default=None, metavar="N",
                   help=f"--think-budgets: questions read at every budget first and discarded (default "
                        f"{calibrate_think.DEFAULT_WARMUP})")
    k.add_argument("--audit", type=int, default=None, metavar="N",
                   help=f"--think-budgets: cut questions regenerated per audited budget to verify the latency model "
                        f"(default {calibrate_think.DEFAULT_AUDIT}; 0 skips the audit)")
    k.add_argument("--cache", action=OnOff, default=None,
                   help="--think-budgets: the run's --cache (on unless --no-cache, as for `callo serve`); in --url "
                        "mode it must equal the server's")
    k.add_argument("--think-sampling", default=None, metavar="model|fixed|SPEC",
                   help="--think-budgets: the trace sampler (default model: the GGUF's general.sampling.* over callo's "
                        "0.6/0.95/20/0 per missing field; fixed: callo's for every field; SPEC: "
                        "temp=..,top_p=..,top_k=..,min_p=..)")
    add_preference_args(k, prefix="--think-budgets: ")
    k.add_argument("--verbose", action="store_true", help="--think-budgets: add NLL, ECE and T to the summary")
    k.add_argument("--dry-run", action="store_true",
                   help="--think-budgets: print the filtered questions, the budgets and the worst case; no requests "
                        "(launch modes: also the GPU-rule scan, the memory precheck and a time estimate)")
    k.add_argument("--model", action="append", default=None, metavar="GGUF",
                   help="--think-budgets: launch mode, one model at a time with the shared launch flags below; "
                        "repeatable (the name is the GGUF stem, a duplicate gets -2, -3)")
    k.add_argument("--menu", default=None, metavar="MENU.json",
                   help="--think-budgets: launch mode with per-model settings: {\"models\": [{\"name\", \"model\", "
                        "\"mmproj\", \"ctx\", \"parallel\", \"no_kvu\", \"s1_slots\", \"llama_server\", \"args\", "
                        "\"user_delimiter\", \"think_sampling\", \"position_bias\"}, ...]}; the shared launch flags "
                        "are an entry's defaults")
    k.add_argument("--ctx", type=int, default=None, help="launch modes: -c for llama-server (default: llama-server's own)")
    k.add_argument("--parallel", type=int, default=None, help=f"launch modes: -np (default {menu.DEFAULT_PARALLEL})")
    k.add_argument("--no-kvu", action="store_true",
                   help="launch modes: per-slot KV, as `callo serve --no-kvu` (needs --ctx: the per-slot context)")
    k.add_argument("--s1-slots", type=int, default=None,
                   help=f"launch modes: reserved classification slots (default {menu.DEFAULT_S1_SLOTS})")
    k.add_argument("--llama-server", default=None, help=f"launch modes: the binary to run (default {menu.DEFAULT_LLAMA_SERVER})")
    k.add_argument("--user-delimiter", default=None,
                   type=_non_empty("omit --user-delimiter to derive it from the chat template"),
                   help="launch modes: pin the user-turn delimiter instead of deriving it from the chat template")
    k.add_argument("--allow-gpu-busy", action="store_true",
                   help="launch modes: run although another model server is running (every profile is marked gpu_busy, "
                        "which compare flags)")
    k.add_argument("--mem-headroom", type=float, default=None, metavar="GIB",
                   help=f"launch modes: the memory precheck's headroom over the file sizes (default "
                        f"{menu.DEFAULT_HEADROOM_GIB:g})")
    k.add_argument("--gpu-mem", type=float, default=None, metavar="GIB",
                   help="launch modes: the memory budget (default: 75%% of memory on macOS, nvidia-smi's total on Linux "
                        "with a GPU, else the memory)")
    k.add_argument("--settle", type=float, default=None, metavar="S",
                   help=f"launch modes: seconds between models once the llama-server has exited (default "
                        f"{menu.DEFAULT_SETTLE_S:g}, at most {menu.SETTLE_MAX_S:g})")
    k.add_argument("--cooldown", choices=("therm",), default=None,
                   help="launch modes: instead of --settle, wait until `pmset -g therm` reports CPU_Speed_Limit = 100 "
                        "(macOS; at most 10 min)")
    k.add_argument("--rates-from", default=None, metavar="DIR",
                   help="--dry-run in launch modes: time estimates from the profiles in DIR (matched by model_key), "
                        "besides --out's")
    k.epilog = ("Launch modes (--model, --menu): arguments after `--` are passed to llama-server and recorded; sampler "
                "flags are overridden by the pinned trace sampler (a warning); --grammar and --json-schema are refused.")

    m = sub.add_parser("compare", help="think budgets: several think profiles on one joint frontier over "
                                       "(model, budget); offline, no server calls")
    m.add_argument("profiles", nargs="+", metavar="PROFILE.think.json",
                   help="think profiles from `callo calibrate --think-budgets` (one is valid; same data required)")
    add_preference_args(m)
    m.add_argument("--max-memory", type=float, default=None, metavar="GIB",
                   help="pick only among models whose file size (GGUF plus mmproj: a lower bound on memory) is at most "
                        "GIB")
    m.add_argument("--out", default=None, metavar="DIR", help="also write DIR/report.md and DIR/report.json")
    m.add_argument("--json", action="store_true", help="print report.json to stdout instead of the markdown report")
    m.add_argument("--deadline", default=None, metavar="MS[,MS...]",
                   help="think deadlines: per profile, what `callo serve --think-deadline MS` would do: the "
                        "closes, the cut, the static budgets that meet it and the accuracy (exact for a measured "
                        "deadline point)")
    m.add_argument("--budget", type=int, default=None, metavar="B",
                   help="--deadline: combined with a token cap B (whichever binds first closes the trace)")
    m.add_argument("--reserve", type=float, default=None, metavar="MS",
                   help="--deadline: re-simulate with this readout reserve instead of the profile's")
    m.add_argument("--verbose", action="store_true", help="add NLL@T, ECE10@T and T to the table")

    sub.add_parser("version")
    return ap


def kv_flag_error(args: argparse.Namespace) -> str | None:
    """The usage error for `--no-kvu`, or None. Raised as exit 2 by `main` (and as a `CalloError` by `serve` for an
    in-process caller that skipped `main`)."""
    if not args.no_kvu:
        return None
    if not args.model:
        return ("--no-kvu is a launch mode flag (--model): attach mode does not start llama-server, so start it "
                "without -kvu yourself")
    if args.ctx is None:
        return "--no-kvu needs --ctx: the per-slot context to reserve"
    if args.ctx < 1:
        return f"--no-kvu needs --ctx >= 1: the per-slot context to reserve, got {args.ctx}"
    return None


def s1_timeout_error(args: argparse.Namespace) -> str | None:
    """The usage error for `--s1-timeout`, or None: the S1 section wait is bounded by it, and a
    wait needs a finite number of seconds above 0 (`proxy.s1_timeout_problem`, the service's own check). Raised as exit
    2 by `main` (and as a `CalloError` by `serve` for an in-process caller that skipped `main`)."""
    return s1_timeout_problem(args.s1_timeout)


def upstream_is_local(url: str) -> bool:
    """Whether `--upstream` (or the launched server's URL) names this box: a loopback name (LOOPBACK: "localhost"
    included, any case) or a loopback address (127.*, ::1). `/callo/stats.upstream_local`: the host block
    describes llama-server's machine only when this is true. No DNS lookup: a name that merely resolves to loopback
    counts as remote, the safe direction (the host block is then marked as callo's box only)."""
    host = (urlsplit(url).hostname or "").lower()
    return host in LOOPBACK or _is_loopback(host)


def upstream_error(url: str) -> str | None:
    """Why `--upstream` is not one callo can talk to, or None. callo's proxy and scorer open plain
    `http.client.HTTPConnection`s to the URL's host and port and send absolute paths (`/props`, `/completion`, ...), so
    the URL must be `http://host[:port]` with nothing after it: an https URL would get plain HTTP on its TLS port, and
    a path prefix (a reverse proxy's `/llama`) would be dropped from every call."""
    try:
        u = urlsplit(url)
        _ = u.port                      # raises ValueError on a bad port
    except ValueError as e:
        return f"--upstream {url!r} is not a valid URL ({e}); expected http://host:port"
    if u.scheme != "http":
        why = ("callo speaks plain HTTP to llama-server; https is not supported" if u.scheme == "https" else
               "it has no http:// scheme" if "://" not in url else f"the scheme {u.scheme!r} is not http")
        return f"--upstream {url!r}: {why}; expected http://host:port"
    if not u.hostname:
        return f"--upstream {url!r} has no host; expected http://host:port"
    if u.path not in ("", "/") or u.query or u.fragment:
        return (f"--upstream {url!r} has a path, query or fragment: callo sends llama-server's own paths (/props, "
                f"/completion, ...) to host:port and would drop it; expected http://host:port")
    if u.username is not None or u.password is not None:
        return f"--upstream {url!r} carries credentials, which callo would not send; use --upstream-api-key"
    return None


class _HostInfo:
    """`host.host_info()` on a daemon thread, started before the model loads, so a hardware query that hangs
    (nvidia-smi in D-state outlives its own timeout) can never hold startup: `attach` waits at most `timeout`, and a late
    result still lands in `stats.host` when it arrives."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._info: dict | None = None
        self._stats: Stats | None = None
        self.done = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="callo-host").start()

    def _run(self) -> None:
        try:
            info: dict | None = host_info()
        except Exception as e:  # noqa: BLE001 -- best-effort: the host block stays null
            log.warning("cannot gather host info (%s: %s): /callo/stats.host stays null", type(e).__name__, e)
            info = None
        with self._lock:
            self._info = info
            if self._stats is not None and info is not None:
                self._stats.host = info
        self.done.set()

    def attach(self, stats: Stats, timeout: float) -> dict | None:
        """The host info once gathered (waiting at most `timeout`), else None; either way `stats.host` gets it."""
        if not self.done.wait(timeout):
            log.warning("host info not gathered after %g s (a hardware query is slow): startup goes on; "
                        "/callo/stats.host fills in when it arrives, and a think profile's host check reads unknown",
                        timeout)
        with self._lock:
            self._stats = stats
            if self._info is not None:
                stats.host = self._info
            return self._info


def split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    """Everything after the first `--` is for llama-server."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return list(argv), []


def build_engine(upstream: str, meta: GgufMeta, *, api_key: str | None, temperature: float, cache: bool,
                 timeout: float, vision: bool = False, media_marker: str | None = None,
                 user_delimiter: str | None = None, model_name: str = "callo",
                 position_bias: PositionBias | None = None, audio: bool = False, video: bool = False,
                 sets: SurfaceSets | None = None) -> Engine:
    return make_engine(upstream, meta, api_key=api_key, temperature=temperature, cache=cache, timeout=timeout,
                       vision=vision, media_marker=media_marker, user_delimiter=user_delimiter, model_name=model_name,
                       position_bias=position_bias, audio=audio, video=video, sets=sets)


# Plan "Failed to load" means ffprobe/ffmpeg could not be run, or ffprobe could not read the clip. An ffmpeg
# that runs but cannot decode yields zero frames and a 200, which the self-test reports as a failure (video stays on).
VIDEO_OFF = ("video self-test: llama-server could not load the built-in AVI ({}): ffprobe or ffmpeg is missing from "
             "its PATH (or --video-ffmpeg-dir), or ffprobe could not read the clip. Video is OFF: video requests get "
             "501")


# the transcription render check's /apply-template calls, each bounded (two pairs, two calls each)
TRANSCRIPTION_SETUP_TIMEOUT = 30.0


def wide_note(coverage: int, readable: int) -> str:
    """The tail of serve's `wide-choice:` startup line: empty when every readable label is a single token (a
    full-coverage vocabulary's line is unchanged), the two-pass reading past the coverage, or the refusal when no
    wide question can be read."""
    if readable <= MAX_LETTERS:
        return " (wide questions are refused)"
    if readable > coverage:
        return f" (beyond {coverage}, a label that is not a single token is read in two passes)"
    return ""


def _log_lines(lines: list[str]) -> None:
    for line in lines:
        (log.warning if line.startswith("WARNING") else log.info)(line)


class _StartupAborted(BaseException):
    """SIGINT/SIGTERM before startup completes. A BaseException, so no `except Exception` on the startup path
    (self-test, probe) can swallow it on its way to `serve`'s `finally`, which stops a launched llama-server."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def residue_warning(probe: ProbeResult, *, launch: bool) -> str:
    """The startup-probe residue warning. It names only fixes that work: launch mode already
    passed --slot-save-path, so there a residue means a passthrough flag overrode it (501) or the erase itself failed;
    attach mode without the flag needs llama-server restarted with it."""
    head = (f"startup-probe residue: slot(s) {list(probe.residue)} still hold the probe's 16 tokens (erase "
            f"{probe.erase}); `callo calibrate --bias` refuses on such a server under unified KV. ")
    if probe.erase == "unsupported" and launch:
        return head + ("llama-server answered 501 to the erase although callo launched it with --slot-save-path, so "
                       "a passthrough flag after -- overrode it: remove that flag (or pass an existing directory) and "
                       "restart callo")
    if probe.erase == "unsupported":
        return head + "Restart llama-server with --slot-save-path DIR (an existing directory) so callo can erase them"
    return head + ("The erase itself failed (see the llama-server log); restart " +
                   ("callo, whose startup probe erases its slots again" if launch else
                    "llama-server (a fresh server holds no tokens)"))


def serve(args: argparse.Namespace, *, ready: Callable[[int], None] | None = None,
          stop: threading.Event | None = None, listening: Callable[[int], None] | None = None) -> int:
    """Run until `stop` is set, SIGINT/SIGTERM arrives (main thread only) or a launched llama-server dies.
    Returns 0 on a deliberate stop, 1 on a `CalloError`, 3 when the child died.

    callo binds, listens and serves FIRST (a busy port fails before anything launches; `listening(port)` is
    called then) and, like llama-server while it loads, answers every request, /health included, with
    `503 {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}` until startup
    (launch, discovery, probe, self-test) is done; then the real config is swapped in and `ready(port)` is called.

    The signal handlers go in FIRST: in launch mode `proc.start()` can block for up to 600 s while the model
    loads, and SIGTERM's default action would kill callo without running `finally`, orphaning a llama-server
    holding tens of GB. Until startup completes, a signal aborts it (raises `_StartupAborted`), the `finally`
    stops the child, and `serve` returns 0 for SIGINT and SIGTERM alike: both are deliberate stops. After
    startup, a signal only sets `stop`. The previous handlers are restored on the way out."""
    stop = stop if stop is not None else threading.Event()
    starting = True

    def on_signal(signum: int, _frame: object) -> None:
        nonlocal starting
        if starting:
            starting = False          # abort once: a later signal (e.g. during cleanup) must not interrupt proc.stop()
            raise _StartupAborted(signum)
        stop.set()

    previous: dict[int, Any] = {}
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    api_key = args.upstream_api_key or os.environ.get("LLAMA_API_KEY")
    http_get = make_http_get(api_key=api_key)
    extra = list(getattr(args, "extra", []) or [])
    proc: LlamaServerProcess | None = None
    request_log: RequestLog | None = None
    kv: str | None = None
    srv = None
    http: threading.Thread | None = None
    serving = False
    child_rc: list[int] = []
    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, on_signal)
        # Everything that can be refused is refused HERE, before a model load that can take minutes.
        if args.temperature is not None and not args.temperature > 0:     # also catches nan
            raise CalloError(f"--temperature must be positive, got {args.temperature!r}")
        if args.s1_slots < 1:
            raise CalloError(f"--s1-slots must be >= 1, got {args.s1_slots}")
        s1_error = s1_timeout_error(args)
        if s1_error:
            raise CalloError(s1_error)
        asr_home_idle = getattr(args, "asr_home_idle", DEFAULT_ASR_HOME_IDLE)
        if idle_error := asr_home_idle_problem(asr_home_idle):
            raise CalloError(idle_error)
        if args.upstream is not None and (up_error := upstream_error(args.upstream)):
            raise CalloError(up_error)
        # the flags (strings, so a non-integer is this refusal, exit 1, not argparse's
        # usage error), then the profile file (InvalidInput is a CalloError: exit 1)
        think_budget = _int_flag("--think-budget", getattr(args, "think_budget", "0"))
        think_max = _int_flag("--think-max", getattr(args, "think_max", str(DEFAULT_THINK_MAX)))
        if think_budget < 0 or think_max < 0:
            raise CalloError(f"--think-budget and --think-max must be non-negative, got {think_budget} and {think_max}")
        if think_max > MAX_BUDGET:
            raise CalloError(f"--think-max {think_max} is above the hard cap {MAX_BUDGET}")
        if think_budget > think_max:
            raise CalloError(f"--think-budget {think_budget} is above --think-max {think_max}")
        # the deadline flags, then the request log (an unwritable file exits 1 before any load)
        think_deadline = _int_flag("--think-deadline", getattr(args, "think_deadline", "0"))
        if not 0 <= think_deadline <= MAX_DEADLINE_MS:
            raise CalloError(f"--think-deadline must be from 0 to {MAX_DEADLINE_MS} ms, got {think_deadline}")
        reserve_flag = getattr(args, "think_deadline_reserve", None)
        deadline_reserve = None if reserve_flag is None else _int_flag("--think-deadline-reserve", reserve_flag)
        if deadline_reserve is not None and not 0 <= deadline_reserve <= MAX_DEADLINE_MS:
            raise CalloError(f"--think-deadline-reserve must be from 0 to {MAX_DEADLINE_MS} ms, got {deadline_reserve}")
        if getattr(args, "request_log", None):
            try:
                request_log = RequestLog(args.request_log)
            except OSError as e:
                raise CalloError(f"--request-log {args.request_log}: cannot open it for appending ({e})") from e
        tp: ThinkProfile | None = load_think_profile(args.think_profile) if getattr(args, "think_profile", None) else None
        think_converge = _converge_flag(getattr(args, "think_converge", "0"))
        # the profile's rules, parsed once: the startup line, /callo/stats and the service share this object
        converge_rules, converge_why = serve_rules(tp.data.get("converge")) if tp is not None else (None, None)
        if think_converge > 0:
            if tp is None:
                raise CalloError(f"--think-converge {think_converge:g} needs --think-profile: a think profile with a "
                                 f"convergence block, from `callo calibrate --think-budgets ... --converge-probe`")
            if converge_rules is None:
                raise CalloError(f"--think-converge {think_converge:g}: the think profile {tp.source} cannot serve "
                                 f"early exit: {converge_why}; recalibrate with `callo calibrate --think-budgets ... "
                                 f"--converge-probe` and pass that profile")
        policy_text = getattr(args, "think_sampling", None)
        if policy_text is not None and tp is not None:      # serve offers no override of a profile's sampling
            raise CalloError(f"--think-sampling {policy_text} with --think-profile: the profile's sampling "
                             f"({tp.sampling.policy}) is part of what was calibrated; omit --think-sampling")
        policy: SamplingPolicy = parse_sampling(policy_text if policy_text is not None else "model")   # InvalidInput
        # the budget-0 temperature and where it came from
        if args.temperature is not None:
            temperature, temperature_source = float(args.temperature), "flag"
        elif tp is not None and tp.t0 is not None:
            temperature, temperature_source = tp.t0, "profile"
        else:
            temperature, temperature_source = 1.0, "default"
        kv_error = kv_flag_error(args)
        if kv_error:
            raise CalloError(kv_error)
        if args.mmproj and not args.model:
            raise CalloError("--mmproj is a launch mode flag (--model); start an attached llama-server with --mmproj "
                             "yourself")
        # a bad profile file costs nothing, not a model load. InvalidInput is a CalloError: exit 1.
        pb = load_profile(args.position_bias) if args.position_bias else None
        # off the critical path: started after every phase-1 refusal (a refused start spawns no hardware query),
        # before the model loads, and waited on for at most HOST_INFO_WAIT
        host = _HostInfo()
        try:
            srv = make_server(None, host=args.host, port=args.port)   # loading 503s until set_config below
        except OSError as e:
            raise CalloError(f"cannot listen on {args.host}:{args.port} (--host/--port): {e}") from e
        http = threading.Thread(target=srv.serve_forever, daemon=True, name="callo-http")
        http.start()
        serving = True
        bound = srv.server_address[1]
        log.info("callo %s listening on http://%s:%d; answering 503 Loading model until startup completes",
                 __version__, args.host, bound)
        if args.host not in LOOPBACK:
            log.warning("binding %s: callo has no auth; anyone who can reach this port can use the model", args.host)
        if listening is not None:
            listening(bound)
        if args.model:
            check_passthrough_args(extra)
            for pid, orphan_port in find_orphans(cache_dir):
                where = f"port {orphan_port}" if orphan_port is not None else "still loading"
                log.warning("orphan llama-server pid %d (%s) from a previous callo is still running; "
                            "`kill %d` if it is not wanted", pid, where, pid)
            if args.no_kvu:
                kv, c_total = "per_slot", args.ctx * args.parallel
                log.info("kv: per-slot (no -kvu), -c %d = %d per slot x %d; text readouts independent of other "
                         "slots (see README caveats)",
                         c_total, args.ctx, args.parallel)
            else:
                kv, c_total = "unified", args.ctx
                log.info("kv: unified (-kvu): readouts on MoE models can shift with other slots' contents; "
                         "--no-kvu removes that on text (see README caveats)")
            slot_dir = cache_dir / "slots"      # lets the probe erase its slots; llama-server needs it to exist
            try:
                slot_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise CalloError(f"cannot create {slot_dir} for --slot-save-path: {e}") from e
            proc = LlamaServerProcess(build_argv(args.llama_server, args.model, ctx=c_total, parallel=args.parallel,
                                                 extra=extra, mmproj=args.mmproj, unified_kv=not args.no_kvu,
                                                 slot_save_path=str(slot_dir)),
                                      cache_dir=cache_dir)
            try:
                port = proc.start()
            except OSError as e:        # no such binary, not executable, unwritable cache dir
                raise CalloError(f"could not launch {args.llama_server!r}: {e}") from e
            upstream = f"http://127.0.0.1:{port}"
            log.info("launched llama-server pid %d on %s (log %s)", proc.proc.pid, upstream, proc.log_file)  # type: ignore[union-attr]
        else:
            upstream = args.upstream.rstrip("/")
        wait_health(upstream, http_get=http_get, alive=proc.alive if proc is not None else None)
        try:
            info = discover(upstream, http_get=http_get)
        except OSError as e:            # the upstream went away between /health and /props
            raise CalloError(f"GET {upstream}/props failed: {type(e).__name__}: {e}") from e
        if info.total_slots < 1:
            raise CalloError(f"/props.total_slots is {info.total_slots}; refusing to attach")
        log.info("upstream %s: total_slots=%d n_ctx=%s build_info=%s", upstream, info.total_slots, info.n_ctx,
                 info.build_info)
        for k in missing_props(info):
            log.warning("/props has no %s", k)
        warn = version_warning(info.build_info)
        if warn:
            log.warning(warn)
        vi = detect_media(info.props)                   # vision, audio and video
        media_on = vi.vision or vi.audio or vi.video
        log.info("vision: %s%s%s", "on" if vi.vision else "off" if media_on else f"off ({vi.note})",
                 f", media marker {vi.media_marker}" if vi.media_marker else "",
                 f", mmproj {args.mmproj}" if args.mmproj else "")
        log.info("audio: %s, video: %s", "on" if vi.audio else "off", "on" if vi.video else "off")
        if args.mmproj and not media_on:                # an audio-only projector is a projector too
            raise CalloError("--mmproj was given but the upstream reports no media modality (vision, audio "
                             f"or video): {vi.note}")
        gguf = resolve_gguf(info, args.gguf)
        meta = load_gguf_meta(gguf, cache_dir)
        model_key_error = None
        try:
            model_key: str | None = load_model_key(gguf, cache_dir)     # published in /callo/stats
        except (CalloError, OSError, ValueError, TypeError) as e:
            if pb is not None:                  # a profile needs the key: exit 1
                raise CalloError(f"--position-bias: cannot compute the served model's model_key: {e}") from e
            model_key, model_key_error = None, str(e)       # a server without a profile must not gain a new way to fail
            log.warning("cannot compute model_key (%s): /callo/stats publishes null, and `callo calibrate --bias` "
                        "refuses against this server", e)
        if pb is not None:
            # phase 2, refusing half: another model or a native upstream exits 1 HERE, before the idle-clear
            # probe and the delimiter derivation touch the upstream's slots. The warnings come after the delimiter.
            assert model_key is not None        # the except above re-raised when a profile is loaded
            check_identity(pb, model_key=model_key, native=meta.decision_type is not None)
        native = meta.decision_type is not None
        if native and (think_budget > 0 or tp is not None):     # native has no readout
            raise CalloError("--think-budget / --think-profile: the upstream is a native decision model; callo's "
                             "engine is off, so there is no deliberate readout to serve")
        if native and request_log is not None:
            log.warning("--request-log only covers /v1/systemone requests callo serves; this upstream is a native "
                        "decision model, so nothing will be logged")
        if native and think_deadline > 0:
            raise CalloError("--think-deadline: the upstream is a native decision model; callo's engine is off, so "
                             "there is no deliberate readout to serve")
        if native and think_converge > 0:
            raise CalloError("--think-converge: the upstream is a native decision model; callo's engine is off, so "
                             "there is no deliberate readout to serve")
        think_converge_gate = bool(getattr(args, "think_converge_gate", False))
        if native and think_converge_gate:
            raise CalloError("--think-converge-gate: the upstream is a native decision model; callo's engine is off, "
                             "so there is no early exit to gate")
        if tp is not None:
            if model_key is None:
                raise CalloError("--think-profile: cannot compute the served model's model_key, which the profile is "
                                 "keyed by")
            check_think_identity(tp, model_key=model_key, native=native, capable=True)   # the frame is checked below
        if info.total_slots >= 2 and info.endpoint_slots:
            probe = probe_idle_clear(upstream, info.total_slots, http_post=make_http_post(api_key=api_key, timeout=120),
                                     http_get=http_get)
        else:
            probe = ProbeResult(None, "skipped: fewer than 2 slots or no /slots endpoint", None)
        log.info("idle-clear probe: %s", probe.detail)
        if probe.residue:
            log.warning("%s", residue_warning(probe, launch=proc is not None))
        pinning, notes = decide_agent_pinning(info, probe)
        for n in notes:
            (log.error if n.startswith("ERROR") else log.warning)(n)
        router = SlotRouter(info.total_slots, args.s1_slots, agent_pinning=pinning, asr_home_idle=asr_home_idle)
        slots_cache = SlotsCache(upstream, http_get=http_get) if info.endpoint_slots else None
        slots_fn = slots_cache.get if slots_cache is not None else None
        stats = Stats()
        stats.probe = probe.as_dict()
        stats.kv = kv
        stats.vision, stats.media_marker = vi.vision, vi.media_marker
        stats.audio, stats.video = vi.audio, vi.video
        stats.model_key, stats.model_key_error, stats.model_file = model_key, model_key_error, gguf.name
        host_block = host.attach(stats, HOST_INFO_WAIT) or {}
        host_fingerprint, host_chip = host_block.get("fingerprint"), host_block.get("chip")
        stats.upstream_local = upstream_is_local(upstream)   # qualifies the host block on every upstream, native too
        stats.n_ctx, stats.total_slots = info.n_ctx, info.total_slots
        # /v1/audio/transcriptions is translated when /props reports audio (whatever callo's
        # engine and its self-tests say, native decision models included); the startup render check decides.
        # It runs on the ready upstream with /props.media_marker, and a short timeout: a wedged upstream must not hold
        # serve's startup for --s1-timeout
        transcription = None
        if vi.audio:
            assert vi.media_marker is not None          # detect_media reports audio only with a media marker
            transcription = transcribe.setup(upstream, info.chat_template, vi.media_marker,
                                             http_post=make_http_post(api_key=api_key, timeout=TRANSCRIPTION_SETUP_TIMEOUT))
            stats.asr_preset = transcription.preset
        # the router's pinning, which is what the handler gates on
        stats.transcription = transcribe.status(transcription, pinning=router.pinning)
        _log_lines(transcribe.startup_lines(transcription, pinning=router.pinning))
        try:
            stats.model_bytes = gguf_bytes(gguf)
            stats.mmproj_bytes = Path(args.mmproj).stat().st_size if args.mmproj and args.model else None
        except (OSError, CalloError) as e:
            log.warning("cannot size the model files (%s): model_bytes stays null", e)
        model_name = info.props.get("model_alias") or gguf.stem
        systemone = None
        think_probe = None
        if meta.decision_type is not None:
            log.info("GGUF carries %s.decision.type=%s: a native decision model; /v1/systemone is forwarded to it "
                     "(callo refuses audio and video with 422, and any think_budget > 0) and callo's engine is off (so "
                     "no user-turn delimiter is derived)", meta.architecture, meta.decision_type)
            stats.think = {"capable": False, "reason": "native decision model: callo's engine is off", "close": None,
                           "synthesized_open": None, "default_budget": 0, "max": think_max, "profile": None,
                           "sampling": None, "gguf_sampling": None, "gguf_error": None}
            if policy_text is not None:
                log.warning("--think-sampling %s is ignored: a native decision model has no deliberate readout",
                            policy_text)
        else:
            # step 2, on every upstream with callo's engine on: context requests need it even without vision
            post = make_http_post(api_key=api_key, timeout=args.s1_timeout)
            delimiter, lines = user_delimiter(
                TemplateClient(upstream, http_post=post), vision=vi.vision, media_marker=vi.media_marker,
                image_url=SELF_TEST_IMAGE_URL, pinned=args.user_delimiter,
                tokenize=LlamaCppForwardScorer(upstream, http_post=post).tokenize)
            _log_lines(lines)
            stats.user_delimiter = delimiter
            stats.cache_prompt = {"plain": args.cache, "image": True, "context": True, "audio": True, "video": True,
                                  "mixed": True}
            # the surface sets first: the provenance's wide contract says whether labels are read in two passes
            sets = SurfaceSets(meta.tokens, meta.tok_model)
            stats.provenance = provenance(chat_template=info.chat_template, build_info=info.build_info,
                                          mmproj=args.mmproj, user_delimiter=delimiter, cache_prompt=args.cache,
                                          server_args=list(extra) if args.model else None,
                                          wide_two_pass=sets.wide_readable() > sets.wide_coverage())
            if pb is not None:                          # the provenance/busy/empty warnings
                assert model_key is not None
                _log_lines(check_profile(pb, model_key=model_key, native=False, provenance=stats.provenance))
                stats.position_bias = pb.info()
                log.info("position bias: %s (sha256 %s), entries n=%s, kinds %s: the self-tests run with it applied",
                         pb.source, pb.sha256[:12], sorted(pb.priors), sorted(pb.kinds))
            engine = build_engine(upstream, meta, api_key=api_key, temperature=temperature, cache=args.cache,
                                  timeout=args.s1_timeout, vision=vi.vision, media_marker=vi.media_marker,
                                  user_delimiter=delimiter, model_name=model_name, position_bias=pb, audio=vi.audio,
                                  video=vi.video, sets=sets)
            stats.temperature = temperature             # reported in /callo/stats; `callo calibrate` checks it
            stats.temperature_source = temperature_source
            if temperature_source == "flag" and tp is not None and tp.t0 is not None and \
                    not 1 / GRID_STEP <= temperature / tp.t0 <= GRID_STEP:
                log.warning("--temperature %s differs from the think profile's T_0 %s by more than one grid step; "
                            "budget 0 is served at the flag's value", temperature, tp.t0)
            s1_slot = router.reserved[0] if router.reserved else None
            # the frame, the unpinnable check, the trace sampler
            self_q = parse_request(SELF_TEST_BODY).questions[0]
            frame, reason = frame_from_template(
                lambda thinking: engine.template.render(messages(SELF_TEST_BODY["state"], self_q, engine.variant),
                                                        thinking=thinking), engine.scorer.tokenize)
            dgs = info.props.get("default_generation_settings")
            params = dgs.get("params") if isinstance(dgs, dict) and isinstance(dgs.get("params"), dict) else {}
            bad_key = unpinnable(params)
            if frame is not None and bad_key is not None:
                frame, reason = None, f"sampler_unpinnable: {bad_key}"
                log.warning("llama-server's default_generation_settings.params has %r, which a request cannot "
                            "override: deliberate readouts are off (budget 0 only)", bad_key)
            unknown = unknown_props_keys(params)
            if unknown:                             # visible even when allowed: a new knob may not be neutral at 0
                log.info("think: /props has sampler keys callo does not know, at neutral values (allowed): %s",
                         ", ".join(unknown))
            engine.think, engine.think_reason, engine.n_ctx = frame, reason, info.n_ctx
            try:
                gguf_sampling = read_gguf_sampling(gguf)
            except (CalloError, OSError, ValueError, TypeError) as e:
                # recorded like gguf-py's own parse failure: stats.think.gguf_error, and a profile's sampling.gguf_error
                gguf_sampling = GGUFSampling(error=f"the GGUF metadata could not be read ({type(e).__name__}: {e})")
                log.warning("cannot read general.sampling.* from the GGUF (%s): callo's trace sampler applies", e)
            # the profile's sampling, else --think-sampling's policy (default model: this GGUF's own)
            sampling = tp.sampling if tp is not None else resolve_sampling(policy, gguf_sampling)
            engine.think_sampling = sampling.pinned
            if frame is None:
                log.info("think: this model is served and calibrated at budget 0 only (%s)", reason)
                if policy_text is not None and think_budget == 0:
                    log.warning("--think-sampling %s is ignored: this model is served at budget 0 only (%s)",
                                policy_text, reason)
                if think_budget > 0:
                    raise CalloError(f"--think-budget {think_budget}: this model's chat template has no thinking mode "
                                     f"callo can read ({reason}); this model can be served and calibrated at budget "
                                     f"0 only")
                if think_deadline > 0:
                    raise CalloError(f"--think-deadline {think_deadline}: this model's chat template has no thinking "
                                     f"mode callo can read ({reason}); this model can be served at budget 0 only")
                if think_converge > 0:
                    raise CalloError(f"--think-converge {think_converge:g}: this model's chat template has no thinking "
                                     f"mode callo can read ({reason}); this model can be served at budget 0 only")
            elif think_deadline > 0 and not frame.tag_single:
                raise CalloError(f"--think-deadline {think_deadline}: think deadlines need a single-token close tag; "
                                 f"this model's is {len(frame.tag_tokens)} tokens")
            elif think_converge > 0 and not frame.tag_single:
                raise CalloError(f"--think-converge {think_converge:g}: early exit needs a single-token close tag; "
                                 f"this model's is {len(frame.tag_tokens)} tokens")
            else:
                log.info("think: capable (close %r, synthesized opening %s); trace sampler %s (%s)", frame.close,
                         frame.synthesized_open, sampling.trace,
                         ", ".join(f"{k} {v}" for k, v in sampling.origin.items()))
            if tp is not None:
                assert model_key is not None
                check_think_identity(tp, model_key=model_key, native=False, capable=frame is not None)
                _log_lines(check_think_profile(
                    tp, provenance=stats.provenance, host_fingerprint=host_fingerprint,
                    build_info=info.build_info, position_bias_sha256=pb.sha256 if pb is not None else None,
                    temperature_from_profile=temperature_source == "profile", default_budget=think_budget,
                    host_chip=host_chip))
            elif think_budget > 0:
                log.warning("--think-budget %d without --think-profile: deliberate readouts are uncalibrated and "
                            "served at T = 1 (calibrate with `callo calibrate --think-budgets`)", think_budget)
            stats.think = {"capable": frame is not None, "reason": reason, "close": frame.close if frame else None,
                           "synthesized_open": frame.synthesized_open if frame else None,
                           "default_budget": think_budget, "max": think_max,
                           "profile": tp.info(host_fingerprint) if tp is not None else None,
                           "sampling": dict(sampling.pinned),
                           "gguf_sampling": dict(gguf_sampling),       # the served GGUF's header, never the profile's
                           # why gguf_sampling is empty when the header could not be read (None: it was read), so an
                           # unparseable GGUF never looks like one without general.sampling.* keys
                           "gguf_error": getattr(gguf_sampling, "error", None)}
            stats.think_budget_keys = ({think_budget} - {0}) | (set(tp.budgets) - {0} if tp is not None else set())
            # The self-tests, in the order (vision, then text) with audio and video appended, each
            # only for a modality the upstream reports. /callo/stats.self_test keeps each verdict.
            if vi.vision:
                vision = vision_self_test(engine, id_slot=s1_slot)
                _log_lines(vision.lines())
                stats.self_test["vision"] = vision.ok
            text = self_test(engine, id_slot=s1_slot)
            _log_lines(text.lines())
            stats.self_test["text"] = text.ok
            if vi.audio:                                # an audio load_failed only warns: audio stays on
                audio = audio_self_test(engine, id_slot=s1_slot)
                _log_lines(audio.lines())
                stats.self_test["audio"] = audio.ok
            if vi.video:
                video = video_self_test(engine, id_slot=s1_slot)
                _log_lines(video.lines())
                stats.self_test["video"] = video.ok
                if video.load_failed:                   # the ONE outcome that turns video off
                    engine.video = stats.video = False  # before SystemOneService exists: nothing flaps
                    # the 501 says why
                    engine.video_off_reason = VIDEO_OFF_DETAIL
                    # no video request is ever sent, so no cache_prompt either
                    stats.cache_prompt["video"] = None
                    log.error(VIDEO_OFF.format(video.error))
            if frame is not None:                       # the think self-test (warns, never refuses)
                think = think_self_test(engine, id_slot=s1_slot, sampling=engine.think_sampling)
                _log_lines(think.lines())
                stats.self_test["think"] = think.ok
            # the coverage, and the wide / partial / continuation sets built once now (never on a vocabulary
            # reading 26 or fewer labels, which only logs: its wide questions are the 422)
            cov = engine.sets.warm_wide()
            readable = engine.sets.wide_readable()
            stats.wide_coverage, stats.wide_readable = cov, readable
            log.info("wide-choice: %d of %d two-letter labels are single tokens; questions up to %d options%s", cov,
                     len(WIDE_LABELS), max(readable, MAX_LETTERS), wide_note(cov, readable))
            # the server-wide estimators, seeded from the profile (its latency block, its budget-0 p95)
            estimators = Estimators.from_profile(tp, reserve_ms=deadline_reserve)
            stats.deadline_default_ms = think_deadline
            if think_deadline > 0:
                _log_lines([deadline_startup_line(think_deadline, tp, think_max, estimators)])
            stats.converge_static = {"default_z": think_converge,
                                     **(converge_rules.as_dict() if converge_rules is not None
                                        else {"K": None, "rules": None})}
            if tp is not None and converge_rules is None and tp.data.get("converge") is not None:
                # only at Z = 0 (a positive Z refused above); a profile without the block is an older one, not a fault
                log.info("NOTE: the think profile's convergence block cannot serve (%s): requests that ask for "
                         "think_converge will report unreachable", converge_why)
            if think_converge > 0:
                assert converge_rules is not None and tp is not None
                _log_lines(converge_startup_lines(think_converge, converge_rules, tp, think_max, think_budget,
                                                  estimators))
            stats.converge_gate_default = think_converge_gate
            if think_converge_gate:
                _log_lines([gate_startup_line(converge_rules, chunkable=frame is not None and frame.tag_single)])
            systemone = SystemOneService(engine, router, stats, model_name=model_name,
                                         slots_snapshot=slots_cache.stamped if slots_cache is not None else None,
                                         fallback=args.s1_fallback == "on", think_default=think_budget,
                                         think_max=think_max, think_profile=tp, s1_timeout=args.s1_timeout,
                                         think_deadline=think_deadline, estimators=estimators,
                                         think_converge=think_converge, converge_rules=converge_rules,
                                         think_converge_gate=think_converge_gate)
            think_probe = ThinkProbe(engine, router, stats, s1_timeout=args.s1_timeout,
                                     slots_snapshot=slots_cache.stamped if slots_cache is not None else None,
                                     fallback=args.s1_fallback == "on")
        # --s1-timeout is the CLASSIFICATION timeout only: agent passthrough reads are never capped
        # (ProxyConfig bounds just the connect).
        cfg = ProxyConfig(upstream=upstream, router=router, systemone=systemone.respond if systemone else None,
                          slots_snapshot=slots_fn, stats=stats, systemone_cancel=True,
                          think_probe=think_probe.respond if think_probe is not None else None,
                          allow_think_probe=bool(getattr(args, "allow_think_probe", False)),
                          transcription=transcription, request_log=request_log)
        if proc is not None:
            def died(rc: int) -> None:
                child_rc.append(rc)
                log.error("llama-server exited with %s; shutting down", rc)
                stop.set()
            proc.watch(died)
        set_config(srv, cfg)            # atomic swap: from here on requests are proxied
        starting = False                # from here on a signal only sets `stop`
        log.info("callo %s serving on http://%s:%d -> %s (reserved slots %s, agent pinning %s, model %s)",
                 __version__, args.host, bound, upstream, list(router.reserved), "on" if pinning else "off", model_name)
        if ready is not None:
            ready(bound)
        while not stop.wait(0.5):
            pass
        return 3 if child_rc else 0
    except _StartupAborted as e:
        log.warning("%s during startup: stopping", signal.Signals(e.signum).name)
        return 0
    except CalloError as e:
        starting = False                # first: a signal while this logs must only set `stop`, not raise
        log.error("%s", e)
        return 1
    finally:
        starting = False
        if srv is not None:
            # shutdown() blocks until serve_forever exits, so only call it once that thread exists
            if serving or (http is not None and http.is_alive()):
                srv.shutdown()
            srv.server_close()
        if proc is not None:
            proc.stop()
        if request_log is not None:
            request_log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler if handler is not None else signal.SIG_DFL)


def deadline_startup_line(deadline_ms: int, tp: ThinkProfile | None, think_max: int, estimators: Estimators) -> str:
    """The one line naming --think-deadline's reach, its deadline-only cap and the reserve."""
    cap = deadline_cap(tp.budgets if tp is not None else None, think_max)
    largest = max((b for b in (tp.budgets if tp is not None else ()) if b > 0), default=0)
    source = "the profile's largest budget" if 0 < largest <= think_max else "--think-max"
    r, where = estimators.reserve()
    reserve = (f"reserve {r:g} ms (--think-deadline-reserve)" if where == "flag" else
               f"reserve learned (seed {r:g} ms from {'the profile' if where == 'profile' else 'the default'})")
    return (f"think deadline {deadline_ms} ms (all clients; think_budget 0 opts out): cap {cap} ({source}), "
            f"{reserve}")


def converge_startup_lines(z: float, rules: ServeRules, tp: ThinkProfile, think_max: int, think_budget: int,
                           estimators: Estimators) -> list[str]:
    """--think-converge's lines: its reach, the rule it serves (n, y, the target), K, T, the probe cost and the cap a
    request without fields runs under (--think-budget when set, else B*); a WARNING when no rule reaches z (early exit
    off), or when that cap (--think-budget, or --think-max) is below the budget the rule was fitted at (a stop is then
    measured against longer thinking than served)."""
    rule = rules.for_z(z)
    if rule is None:
        have = ", ".join(f"{t:g}" for t, r in sorted(rules.rules.items()) if r is not None) or "none"
        return [f"WARNING: --think-converge {z:g}: the think profile has no early-exit rule that reaches it (targets "
                f"with a rule: {have}); early exit is off for these requests, and X-Callo-Timings says "
                f"\"converge\": \"unreachable\""]
    cap = think_budget if think_budget > 0 else deadline_cap(tp.budgets, think_max)
    bound = f"--think-budget {think_budget}" if think_budget > 0 else f"--think-max {think_max}"
    largest = max((b for b in tp.budgets if b > 0), default=0)
    c, source = estimators.probe_cost()
    t = f"T {rule.T:.2f}" if rule.T is not None else "T from the budgets (the rule has none)"
    probe = f"probe {c:.0f} ms ({source})" if c is not None else "probe cost unmeasured"
    lines = [f"think converge {z:g} (all clients; think_budget 0 opts out): rule n {rule.n}, y {rule.y:g} (target "
             f"{rule.z:g}), K {rules.K}, {t}, {probe}; cap {cap}"]
    if 0 < cap < largest:
        lines.append(f"WARNING: --think-converge: the rule was fitted on {largest}-token traces but {bound} caps "
                     f"them at {cap}; a stop is then measured against longer thinking than served")
    return lines


def gate_startup_line(rules: ServeRules | None, *, chunkable: bool = True) -> str:
    """--think-converge-gate's line: the gate is on and what it does, or a WARNING when no early exit can run here
    (no think profile with a usable convergence block, or a model that cannot chunk: no thinking mode, or a close tag
    of several tokens), so the gate never acts."""
    if rules is None:
        return ("WARNING: --think-converge-gate: early exit cannot run on this server (no think profile with a usable "
                "convergence block), so the gate never acts")
    if not chunkable:
        return ("WARNING: --think-converge-gate: early exit cannot run on this model (no thinking mode callo can read, "
                "or a close tag of several tokens), so the gate never acts")
    return ("think converge gate on (experimental; think_converge_gate false opts a request out): before an early-exit "
            "request thinks, one budget-0 question asks whether it needs several stated rules applied step by step, "
            "and a yes thinks to the cap without early exit")


def _post_json(url: str, body: dict, timeout: float = 900.0) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 -- local server
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw.decode(errors="replace")


def _media_fields(args: argparse.Namespace) -> dict[str, list[str]] | str:
    """`--image` / `--audio` / `--video` as the request's `images` / `audio` / `videos` data URLs (only the fields
    given), or the usage error to print (exit 2). Checked before any file is read: the count (MAX_MEDIA in all), each
    extension against its fixed table, and the encoded size against the 64 MiB body cap."""
    groups = (("images", "--image", args.image, IMAGE_TYPES), ("audio", "--audio", args.audio, AUDIO_TYPES),
              ("videos", "--video", args.video, VIDEO_TYPES))
    count = sum(len(paths) for _, _, paths, _ in groups)
    if count > MAX_MEDIA:
        return f"at most {MAX_MEDIA} media files (--image, --audio and --video together), got {count}"
    encoded = 0
    for _, flag, paths, table in groups:
        for p in paths:
            mime = table.get(Path(p).suffix.lower())
            if mime is None:
                return f"{flag} {p}: unsupported extension {Path(p).suffix or '(none)'}; use one of {', '.join(table)}"
            try:
                encoded += len(f"data:{mime};base64,") + 4 * -(-Path(p).stat().st_size // 3)
            except OSError as e:
                return f"{flag} {p}: {e}"
    if encoded > CLASSIFY_MAX_BODY:
        return (f"the media would encode to {encoded} bytes, over the {CLASSIFY_MAX_BODY}-byte (64 MiB) body limit of "
                f"/v1/systemone; send fewer or smaller files")
    fields: dict[str, list[str]] = {}
    for name, flag, paths, table in groups:
        for p in paths:
            try:
                data = Path(p).read_bytes()
            except OSError as e:
                return f"{flag} {p}: {e}"
            fields.setdefault(name, []).append(
                f"data:{table[Path(p).suffix.lower()]};base64,{base64.b64encode(data).decode()}")
    return fields


def classify(args: argparse.Namespace) -> int:
    opts = [o.strip() for o in args.options.split(",") if o.strip()]
    if not opts:
        print("callo: --options must name at least one option", file=sys.stderr)
        return 2
    media = _media_fields(args)
    if isinstance(media, str):
        print(f"callo: {media}", file=sys.stderr)
        return 2
    body: dict[str, Any] = {"state": sys.stdin.read(),
                            "questions": {"q": {"type": "choice", "instructions": args.instructions,
                                                "criteria": {o: None for o in opts}}}}
    body.update(media)
    if args.context is not None:
        body["context"] = args.context
    size = len(json.dumps(body).encode())
    # the media fitted, but with the state, context and options it is over
    if size > CLASSIFY_MAX_BODY:
        print(f"callo: the request body would be {size} bytes, over the {CLASSIFY_MAX_BODY}-byte (64 MiB) body limit "
              f"of /v1/systemone; send fewer or smaller media files, or a shorter state (stdin) or --context",
              file=sys.stderr)
        return 2
    try:
        status, resp = _post_json(f"{args.url.rstrip('/')}/v1/systemone", body)
    except OSError as e:
        print(f"callo: {e}", file=sys.stderr)
        return 1
    if status != 200 or not isinstance(resp, dict):
        print(f"callo: HTTP {status}: {resp}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(resp))
        return 0
    a = resp["answers"]["q"]
    for k, p in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"{p:7.4f}  {k}")
    print(f"choice: {a['choice']}  confidence: {a['confidence']:.3f}")
    return 0


_BIAS_ONLY = (("--raw", "raw"), ("--resume", "resume"), ("--min-questions", "min_questions"),
              ("--max-per-n", "max_per_n"), ("--seed", "seed"), ("--allow-busy", "allow_busy"))


def _passthrough(args: argparse.Namespace) -> list[str]:
    """Arguments after `--` are llama-server's, so a launch-only "flag" of calibrate's."""
    return ["arguments after --"] if getattr(args, "extra", None) else []


def calibrate_cmd(args: argparse.Namespace) -> int:
    if args.think_budgets is not None:                                  # (exclusive with --bias)
        if args.model or args.menu:
            return menu.run(args)                                       # the launch modes (sources exclusive)
        if launch_only := calibrate_think.given(args, menu.LAUNCH_ONLY) + _passthrough(args):
            print(f"callo: {', '.join(launch_only)} apply only with --model or --menu", file=sys.stderr)
            return 2
        return calibrate_think.run(args)
    if think_only := calibrate_think.given(args, calibrate_think.THINK_ONLY + menu.MENU_ONLY) + _passthrough(args):
        print(f"callo: {', '.join(think_only)} apply only with --think-budgets", file=sys.stderr)
        return 2
    if args.bias:
        return calibrate_bias.run(args)
    # `is not`, never `not in (None, False)`: an explicit 0 (`--seed 0`) equals False.
    given = [flag for flag, attr in _BIAS_ONLY if getattr(args, attr) is not None and getattr(args, attr) is not False]
    if given:
        shared = {flag for flag, _ in calibrate_think.SHARED_WITH_BIAS}       # --resume, --seed, --allow-busy
        parts = [f"{', '.join(flags)} apply only with {modes}" for flags, modes in
                 (([f for f in given if f not in shared], "--bias"),
                  ([f for f in given if f in shared], "--bias or --think-budgets")) if flags]
        print(f"callo: {'; '.join(parts)}", file=sys.stderr)
        return 2
    if args.data is None:
        print("callo: calibrate needs --data (labelled JSONL) to fit a temperature; `callo calibrate --bias` measures "
              "the position bias and can use the built-in set", file=sys.stderr)
        return 2
    base = args.url.rstrip("/")
    url = f"{base}/v1/systemone"
    # The fit needs probabilities served at T = 1. A server already at T != 1 would silently yield a
    # RELATIVE temperature, so refuse unless /callo/stats confirms 1.0.
    try:
        with urllib.request.urlopen(f"{base}/callo/stats", timeout=60) as r:  # noqa: S310 -- local server
            stats = json.loads(r.read())
            temperature = stats.get("temperature")
    except (OSError, ValueError, AttributeError) as e:
        print(f"callo: refusing to calibrate: cannot read the server's S1 temperature from {base}/callo/stats ({e})",
              file=sys.stderr)
        return 2
    if temperature is None:
        print("callo: refusing to calibrate: callo's engine is off: the upstream is a native decision model; "
              "calibrate applies to callo's zero-training readout", file=sys.stderr)
        return 2
    # before T: serve with --think-budget and a think profile reports the profile's T_0, and the
    # "not 1.0" refusal would point at --temperature, which cannot help
    refusal = think_default_refusal(stats)
    if refusal is not None:
        print(f"callo: refusing to calibrate: {refusal}", file=sys.stderr)
        return 2
    if temperature != 1.0:
        fix = PROFILE_T_FIX if stats.get("temperature_source") == "profile" else \
            "restart `callo serve` without --temperature (or with --temperature 1.0)"
        print(f"callo: refusing to calibrate: the server's S1 temperature is {temperature!r}, not 1.0. The fit needs "
              f"probabilities served at T = 1: {fix}", file=sys.stderr)
        return 2
    try:                                        # worded as --bias mode words it
        text = Path(args.data).read_bytes().decode("utf-8")
    except OSError as e:
        print(f"callo: --data {args.data}: {e}", file=sys.stderr)
        return 1
    except UnicodeDecodeError as e:
        print(f"callo: --data {Path(args.data).name}: not UTF-8 ({e})", file=sys.stderr)
        return 1
    rows = []
    skipped_records = skipped_answers = 0
    for number, line in enumerate(text.split("\n"), 1):   # "\n" only: a JSON string may hold a raw U+2028
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as e:
            print(f"callo: --data line {number}: not valid JSON ({e})", file=sys.stderr)
            return 1
        if not isinstance(rec, dict):
            print(f"callo: --data line {number}: not a JSON object", file=sys.stderr)
            return 1
        meta = rec.get("_meta", {})
        if (meta.get("variant", "clean") if isinstance(meta, dict) else "clean") != "clean":
            continue                            # first, as in a non-clean record is never built
        try:
            req = request_from_record(rec)
        except (KeyError, TypeError, AttributeError) as e:
            # in both modes. The structure only: a request the server refuses (a 422, say
            # too many options) is still skipped and counted.
            print(f"callo: --data line {number}: not a System One request ({type(e).__name__}: {e})",
                  file=sys.stderr)
            return 1
        try:
            status, resp = _post_json(url, req)
        except OSError as e:
            print(f"callo: {e}", file=sys.stderr)
            return 1
        rid = meta.get("id") if isinstance(meta, dict) else None        # a non-dict _meta: no id, as for the variant
        if status != 200 or not isinstance(resp, dict):
            print(f"callo: skipping {rid}: HTTP {status}", file=sys.stderr)
            skipped_records += 1
            continue
        answers = resp.get("answers")
        if not isinstance(answers, dict):
            print(f"callo: skipping {rid}: a 200 reply with no answers", file=sys.stderr)
            skipped_records += 1
            continue
        for qid, q in rec["questions"].items():
            if "label" in q and qid in answers:
                try:
                    rows.append(row_from_answer(q, answers[qid]))
                except InvalidInput as e:
                    print(f"callo: skipping {rid} question {qid}: {e}", file=sys.stderr)
                    skipped_answers += 1
    if skipped_records or skipped_answers:
        print(f"callo: skipped {skipped_records} record(s) and {skipped_answers} answer(s)", file=sys.stderr)
    if not rows:
        print("callo: no labelled rows to fit on", file=sys.stderr)
        return 1
    t = fit_temperature(rows)
    bias = stats.get("position_bias")
    bias_sha = bias.get("sha256") if isinstance(bias, dict) else None
    result = {"temperature": t, "n": len(rows), "nll_before": nll(rows, 1.0), "nll_after": nll(rows, t),
              "ece_before": ece(rows, 1.0), "ece_after": ece(rows, t), "grid": GRID_TEXT,
              "limit_note": limit_note(t, flat=flat_fit(rows)), "skipped_records": skipped_records,
              "skipped_answers": skipped_answers,
              "position_bias_sha256": bias_sha}
    print(f"n={len(rows)} T={t:.4f}")
    print(f"NLL  {result['nll_before']:.4f} -> {result['nll_after']:.4f}")
    print(f"ECE  {result['ece_before']:.4f} -> {result['ece_after']:.4f}")
    if result["limit_note"]:
        print(result["limit_note"])
    if bias_sha:
        print(f"fitted on readouts debiased by the position-bias profile sha256 {bias_sha[:12]}: a refitted profile "
              f"invalidates this T")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args_list, extra = split_passthrough(argv)
    parser = build_parser()
    args = parser.parse_args(args_list)
    if args.cmd == "serve" and (usage := kv_flag_error(args) or s1_timeout_error(args)
                                or asr_home_idle_problem(args.asr_home_idle)):
        parser.error(usage)
    args.extra = extra
    logging.basicConfig(level=logging.INFO, format="%(asctime)s callo %(levelname)s %(message)s", stream=sys.stderr)
    if args.cmd == "version":
        print(f"callo {__version__}")
        return 0
    if args.cmd == "serve":
        return serve(args)
    if args.cmd == "classify":
        return classify(args)
    if args.cmd == "compare":
        return compare.run(args)
    return calibrate_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
