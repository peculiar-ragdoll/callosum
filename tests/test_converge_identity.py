"""Early exit is opt-in. A request without `think_converge`, on a server without `--think-converge`, behaves byte for
byte as it did before early exit existed: the same upstream calls (the raw bytes callo sent), reply bytes, response
headers (X-Callo-Timings as its raw string, keys in order), request-log line (its raw text) and /callo/stats (minus the
keys early exit adds); only named wall-clock values are masked. The think probe's live and budgets modes, the 422s of
the existing fields, `callo serve`'s parsed defaults and the deadline planner's pure functions (feasibility, chunk
sizing, the estimators and calibrate's simulation, where the time term DOES bind) are unchanged too.

The goldens (tests/data/identity_golden.json, tests/data/identity_planner_golden.json) were recorded on the code
before early exit; CALLO_UPDATE_GOLDEN=1 rewrites them, which is never the fix for a failure here. Every input is
frozen in this file (bodies, image, profile, the live server's construction), so an edit to another test module
cannot move the golden's setup."""
import dataclasses
import json
import os
import re
import string
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from fake_upstream import FakeUpstream

from callo.cli import build_parser
from callo.deadline import Chunk, Estimate, Estimators, chunk_size, feasible, share_end, simulate
from callo.engine import Engine, SurfaceSets
from callo.prompt import TemplateClient
from callo.proxy import ProxyConfig, RequestLog, Stats, SystemOneService, ThinkProbe, make_server
from callo.scorer import LlamaCppForwardScorer, make_http_post
from callo.slots import SlotRouter
from callo.systemone import native_think_refusal
from callo.think import frame_from_template, load_think_profile, parse_sampling, resolve_sampling

GOLDEN = Path(__file__).parent / "data" / "identity_golden.json"
PLANNER_GOLDEN = Path(__file__).parent / "data" / "identity_planner_golden.json"

# -- the frozen inputs (copied from test_think_serve / test_think / test_deadline_serve on the day of recording) --

TOKENS = list(string.ascii_uppercase) + ["Ġ" + c for c in string.ascii_uppercase] + ["**", "Ġthe", "x"]
DELIM = "<|im_start|>user"
BODY = {"state": "the state", "questions": {"q": {"type": "choice", "instructions": "pick",
                                                  "criteria": {"a": "one", "b": "two", "c": "three"}}}}
TWO = {**BODY, "questions": {**BODY["questions"], "r": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}}
PNG = ("data:image/png;base64,"
       "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAD0lEQVR4AQEEAPv/AAAAAAAEAAFlScNgAAAAAElFTkSuQmCC")
IMG = {**BODY, "images": [PNG]}
PINNED = {"dynatemp_range": 0.0, "dynatemp_exponent": 1.0, "top_n_sigma": -1.0, "xtc_probability": 0.0,
          "xtc_threshold": 0.1, "typical_p": 1.0, "repeat_last_n": 64, "repeat_penalty": 1.0, "presence_penalty": 0.0,
          "frequency_penalty": 0.0, "dry_multiplier": 0.0, "dry_base": 1.75, "dry_allowed_length": 2,
          "dry_penalty_last_n": 64, "dry_sequence_breakers": ["\n", ":", '"', "*"], "mirostat": 0,
          "mirostat_tau": 5.0, "mirostat_eta": 0.1, "adaptive_target": -1.0, "adaptive_decay": 0.9, "min_keep": 0,
          "samplers": ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
          "logit_bias": [], "ignore_eos": False, "n_probs": 0, "post_sampling_probs": False, "backend_sampling": False,
          "reasoning_budget_tokens": -1, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
PROFILE = {"callo_think_profile": 1, "metrics": "callo-think-metrics-1", "model_key": "sha256:" + "5a" * 32,
           "model_name": "tiel",
           "comparable": {"sampling": {"policy": "model", "pinned": PINNED,
                                       "origin": {"temperature": "gguf", "top_p": "gguf", "top_k": "gguf",
                                                  "min_p": "callo"},
                                       "gguf": {"temp": 1.0, "top_k": 20, "top_p": 0.95}},
                          "data": {"sha256": "d" * 64}},
           "environment": {"host": {"fingerprint": "sha256:" + "h" * 64}, "build_info": "b1-926862e",
                           "server": {"position_bias_sha256": None}},
           "provenance": {"prompt_sha256": "p" * 64, "chat_template_sha256": "t" * 64, "build_info": "b1-926862e",
                          "mmproj": None, "user_delimiter": "<|im_start|>user", "cache_prompt": False,
                          "server_args": None},
           "thinking": {"capable": True, "reason": None}, "server_state": {"state": "clean"},
           "audit": {"latency_model": "verified"}, "drift": {"flagged": False},
           "budgets": {"0": {"T": 1.62}, "256": {"T": 1.1}, "1024": {"T": 0.9}}, "pick": {"budget": 256}}
RESERVE_MS = 30.0
THINK_MAX = 4096

# -- the matrix --

D = 600_000     # the time term never binds at this deadline: every chunk is min(cap - k, C_MAX), so the served chunk
#                 sizes are wall-clock free; time-bound sizing is pinned by the planner golden (and binding_deadline)
BLOCK = {"K": 32, "cost": {"c_ms": 5.0}, "rules": {"0.9": {"cell": [1, 0.6], "T": 2.0}, "0.95": None, "0.99": None}}
PROFILES = {"plain": {}, "with_block": {"converge": BLOCK}}
SERVERS = {"defaults": {}, "deadline": {"think_deadline": D}, "budget": {"think_default": 256}}
BODIES = {"plain": BODY, "budget": {**BODY, "think_budget": 256}, "deadline": {**BODY, "think_deadline_ms": D},
          "both": {**BODY, "think_budget": 256, "think_deadline_ms": D},
          "budget0_deadline": {**BODY, "think_budget": 0, "think_deadline_ms": D},
          "budget_no_deadline": {**BODY, "think_budget": 256, "think_deadline_ms": 0},
          "two_deadline": {**TWO, "think_deadline_ms": D},
          "fallback": {**BODY, "think_deadline_ms": 1}}                    # too short to think: budget0
INVALID = {"budget_negative": {**BODY, "think_budget": -1}, "budget_string": {**BODY, "think_budget": "256"},
           "deadline_negative": {**BODY, "think_deadline_ms": -1},
           "deadline_string": {**BODY, "think_deadline_ms": "100"}}

# -- what is masked: wall-clock VALUES of named fields only (the key, its place and the separators stay compared) --

NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
TIMING_WALL = re.compile(rf'"(ms|gen_ms|elapsed_ms)":{NUM}')     # prompt_ms is the upstream's report: compared
LOG_WALL = re.compile(rf'"(t_arr_unix_ms|t_written_unix_ms|served_ms|elapsed_ms)":{NUM}')
HEADERS_SKIPPED = {"date", "server"}             # the clock and the version banner: not request behaviour
ADDED_STATS = {("think", "converged"), ("think", "deadline", "closes", "converged"), ("think", "converge"),
               ("s1", "wide_uncovered_readouts"), ("wide_readable",)}    # the last two: 0.3.1's two-pass reading
STATS_VOLATILE = {("instance",), ("uptime_s",)}  # dropped (key and value): the process's identity and age
STATS_WALL = {("s1", "p50_ms"), ("s1", "p95_ms"), ("s1", "image_p50_ms"), ("s1", "image_p95_ms"),
              ("think", "gen_p50_ms"), ("think", "gen_p95_ms"), ("think", "deadline", "chunk_overhead_ms"),
              ("think", "deadline", "stall_ms"), ("think", "deadline", "elapsed_ms"),
              ("think", "deadline", "over_ms")}  # a float at or under these paths is a wall clock: key kept, value null


def _mask(pattern: re.Pattern, text: str) -> str:
    return pattern.sub(lambda m: f'"{m.group(1)}":#', text)


def _stats_shape(st, path=()):
    """Every key path and leaf of /callo/stats, minus the keys early exit adds and the process identity; a float at a
    wall-clock path keeps its key with the value null, every other float is compared (to 9 decimals)."""
    if isinstance(st, dict):
        return [[k, _stats_shape(v, (*path, k))] for k, v in st.items()
                if (*path, k) not in ADDED_STATS and (*path, k) not in STATS_VOLATILE]
    if isinstance(st, list):
        return [_stats_shape(v, path) for v in st]
    if isinstance(st, float):
        return None if any(path[:n] in STATS_WALL for n in range(1, len(path) + 1)) else round(st, 9)
    return st


def _scrub(x):
    """A probe reply minus every wall clock (`ms` and every `*_ms` key), key order kept."""
    if isinstance(x, dict):
        return [[k, _scrub(v)] for k, v in x.items() if not (k == "ms" or k.endswith("_ms"))]
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    return x


# -- the live server, frozen --

def _engine(up) -> Engine:
    post = make_http_post(timeout=10)
    e = Engine(LlamaCppForwardScorer(up.url, http_post=post), TemplateClient(up.url, http_post=post),
               SurfaceSets(TOKENS, "gpt2"), user_delimiter=DELIM, temperature=1.0, vision=up.vision,
               media_marker=up.props.get("media_marker") if up.vision else None, s1_timeout=5.0)
    e.think, e.think_reason = frame_from_template(
        lambda thinking: e.template.render([{"role": "user", "content": "x"}], thinking=thinking), e.scorer.tokenize)
    e.n_ctx = up.props["default_generation_settings"]["n_ctx"]
    e.think_sampling = resolve_sampling(parse_sampling("fixed")).pinned
    return e


def _live(up, *, profile, think_deadline=0, think_default=0, request_log=None):
    """A real proxy + service + engine against the fake; returns (server, router, stats, service)."""
    engine = _engine(up)
    router, stats = SlotRouter(4, 1), Stats()
    stats.think = {"capable": engine.think is not None, "reason": engine.think_reason, "default_budget": think_default,
                   "max": THINK_MAX, "profile": None}
    svc = SystemOneService(engine, router, stats, model_name="fake-9b", think_default=think_default,
                           think_max=THINK_MAX, think_profile=profile, s1_timeout=5.0, think_deadline=think_deadline,
                           estimators=Estimators(reserve_ms=RESERVE_MS), slots_snapshot=None)
    srv = make_server(ProxyConfig(upstream=up.url, router=router, stats=stats, systemone=svc.respond,
                                  systemone_cancel=True, request_log=request_log), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, router, stats, svc


def _post_raw(srv, body: dict) -> tuple[int, list, bytes]:
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/v1/systemone",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, list(r.headers.items()), r.read()
    except urllib.error.HTTPError as e:
        return e.code, list(e.headers.items()), e.read()


def _stats(srv) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/callo/stats", timeout=10) as r:
        return json.loads(r.read())


def _stop(srv, up) -> None:
    """Both servers at once: each shutdown waits out its own 0.5 s serve_forever poll, so in turn they cost a second."""
    upstream = threading.Thread(target=up.stop)
    upstream.start()
    try:
        srv.shutdown()
    finally:
        upstream.join()


def _logged(path: Path) -> list[str]:
    """The request-log lines as written (raw text), wall-clock values masked."""
    deadline = time.monotonic() + 5
    while not path.read_text() and time.monotonic() < deadline:
        time.sleep(0.02)
    return [_mask(LOG_WALL, line) for line in path.read_text(encoding="utf-8").splitlines()]


def _upstream(up) -> list:
    """Every upstream call as callo sent it: path and raw body bytes (decoded, not re-dumped)."""
    return [[r.path, r.body.decode("utf-8")] for r in up.requests]


def _one(tmp_path: Path, name: str, body: dict, *, up_kw: dict | None = None, prof: dict | None = None,
         shape_only: bool = False, **server) -> dict:
    up_kw = dict(up_kw or {})
    multi = up_kw.pop("multi_tag", False)            # not a constructor argument: set before _live() builds the engine
    real_time = up_kw.pop("real_time", False)        # every think token SLEPT 10 ms: the 300-token trace takes 3 s
    up = FakeUpstream(think="preopened", think_len=300, **up_kw).start()
    up.think_multi_tag = multi
    up.think_real_time, up.think_ms_per_token = real_time, 10.0
    path = tmp_path / f"{name.replace('/', '-')}.jsonl"
    log = RequestLog(str(path))
    srv, *_ = _live(up, profile=load_think_profile({**PROFILE, **(prof or {})}), request_log=log, **server)
    try:
        status, headers, reply = _post_raw(srv, body)
        shape = _stats_shape(_stats(srv))
    finally:
        _stop(srv, up)
    try:
        logged = _logged(path)
    finally:
        log.close()
    timings = dict(headers).get("X-Callo-Timings")
    if shape_only:
        entries = json.loads(timings or "[]")
        return {"status": status, "timing_keys": [list(e) for e in entries],
                "closes": [e.get("think_close") for e in entries]}
    return {"status": status, "reply": reply.decode("utf-8"),
            "headers": [[k, _mask(TIMING_WALL, v) if k == "X-Callo-Timings" else v] for k, v in headers
                        if k.lower() not in HEADERS_SKIPPED],
            "upstream": _upstream(up), "log": logged, "stats": shape}


def _probe(body: dict) -> dict:
    up = FakeUpstream(think="preopened", think_len=300).start()
    srv, router, st, svc = _live(up, profile=load_think_profile(PROFILE))
    try:
        code, out, _ = ThinkProbe(svc.engine, router, st, s1_timeout=5.0).respond(body)
    finally:
        _stop(srv, up)
    return {"status": code, "reply": _scrub(out), "upstream": _upstream(up)}


def _serve_defaults() -> list:
    """`callo serve`'s parsed defaults (every option but think_converge and think_converge_gate, which early exit
    adds), in order."""
    ns = vars(build_parser().parse_args(["serve", "--upstream", "http://127.0.0.1:8199"]))
    return [[k, v] for k, v in ns.items() if k not in ("think_converge", "think_converge_gate")]


def _record(tmp_path: Path) -> dict:
    out = {}
    for pname, prof in PROFILES.items():
        for sname, server in SERVERS.items():
            for bname, body in BODIES.items():
                key = f"{pname}/{sname}/{bname}"
                out[key] = _one(tmp_path, key, body, prof=prof, **server)
    out["media/deadline"] = _one(tmp_path, "media", {**IMG, "think_deadline_ms": D}, up_kw={"vision": True})
    out["not_chunkable/deadline"] = _one(tmp_path, "multi", {**BODY, "think_deadline_ms": D},
                                         up_kw={"multi_tag": True})
    # binds deterministically: the trace needs 3 s of slept decode against a 300 ms deadline (10x), on any machine
    out["binding_deadline"] = _one(tmp_path, "binding", {**BODY, "think_deadline_ms": 300}, up_kw={"real_time": True},
                                   shape_only=True)
    for name, body in INVALID.items():
        out[f"invalid/{name}"] = _one(tmp_path, f"invalid-{name}", body)
    out["probe/live"] = _probe({"request": BODY, "budgets": [256], "seed": 0,
                                "live": {"deadline_ms": None, "probe_every": 32}})
    out["probe/budgets_converge"] = _probe({"request": BODY, "budgets": [0, 256], "seed": 0, "converge_probe": 32})
    out["native"] = [native_think_refusal(b) for b in BODIES.values()]
    out["cli/serve_defaults"] = _serve_defaults()
    return out


_FLOAT = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")


def _older_python(x):
    """Python 3.12 made sum() of floats compensated, so on 3.10/3.11 a confidence can differ from the golden (recorded
    on 3.12+) in its last digits; there the golden is compared at 12 significant digits."""
    if isinstance(x, dict):
        return {k: _older_python(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_older_python(v) for v in x]
    if isinstance(x, float):
        return float(f"{x:.12g}")
    if isinstance(x, str):
        return _FLOAT.sub(lambda m: repr(float(f"{float(m.group()):.12g}")), x)
    return x


def _check(path: Path, got) -> None:
    if os.environ.get("CALLO_UPDATE_GOLDEN") == "1":
        path.write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
    want = json.loads(path.read_text(encoding="utf-8"))
    if sys.version_info < (3, 12):
        got, want = _older_python(got), _older_python(want)
    assert got == want


def test_requests_without_think_converge_are_byte_identical_to_before_early_exit(tmp_path):
    _check(GOLDEN, _record(tmp_path))


def test_serve_starts_with_early_exit_off():
    ns = build_parser().parse_args(["serve", "--upstream", "http://127.0.0.1:8199"])
    assert float(getattr(ns, "think_converge", 0) or 0) == 0         # before early exit: no such option
    assert getattr(ns, "think_converge_gate", False) is False         # nor its gate


# -- the planner's pure functions where the time term binds --

def _r(x):
    return round(x, 9) if isinstance(x, float) else x


def _est(e: Estimate) -> list:
    """An estimate's fields in order; a field early exit adds (`probe_ms`) is left out only while it is 0."""
    return [[k, _r(v)] for k, v in dataclasses.asdict(e).items() if not (k == "probe_ms" and v == 0)]


def _estimator_states() -> dict:
    """Estimators in deterministic states: the defaults, a profile seed, and learned windows (a fake clock)."""
    now = [100.0]
    clock = lambda: now[0]                                                   # noqa: E731
    seeded = Estimators(latency={"read_forced_ms": {"p95": 40.0}, "decode_ms_per_token": {"p50": 12.5},
                                 "prefill_ms_per_token": {"p95": 0.8}}, budget0_p95_ms=150.0, clock=clock)
    learned = Estimators(clock=clock)
    stalled = Estimators(clock=clock)
    for est in (learned, stalled):
        prompt = tuple(range(400))
        for j in range(10):
            est.note_sent(0, prompt)
            est.add_chunk0_prompt(0, prompt + (j,), 1)
            est.add_chunk(0, Chunk(tuple(range(64)), "limit", 64 * 9.0 + 60.0 + 3 + j, 50.0 + j, 63 * 9.0, 400 + j,
                                   0, 64))
            est.add_chunk(1, Chunk(tuple(range(128)), "limit", 128 * (9.5 + j / 10) + 4 + j, 9.5, 127 * 9.5, 1, 400,
                                   128))
            est.add_readout(20.0 + 3 * j)
            est.add_budget0(120.0 + 7 * j)
            now[0] += 0.1
    for j in range(10):                                  # continuation chunks far slower than the window: a residual
        stalled.add_chunk(2, Chunk(tuple(range(32)), "limit", 32 * 9.5 + 80.0 + 5 * j, 9.5, 31 * 9.5, 1, 600, 32))
    stalled.dispatch_start(1, 250.0)                     # a prefill in flight on another slot: S_known
    now[0] += 0.05
    return {"defaults": Estimators(clock=clock), "seeded": seeded, "learned": learned, "stalled": stalled}


def _planner() -> dict:
    out = {}
    lefts = (5.0, 20.0, 45.0, 80.0, 150.0, 400.0, 1000.0, 3000.0, 20000.0)
    caps = ((20, 0), (256, 0), (1024, 0), (1024, 250), (1024, 1000), (4096, 3000))
    for name, est in _estimator_states().items():
        state = {"reserve": list(est.reserve()), "c0": _r(est.c0()), "step_seed": _r(est.step_seed()),
                 "prefill_rate": _r(est.prefill_rate()), "overhead": _r(est.overhead()), "resid95": _r(est.resid95()),
                 "known": [_r(est.known(s)) for s in (0, 1, None)],
                 "uncached": [est.uncached(0, tuple(range(400)) + tuple(range(9))), est.uncached(1, tuple(range(50))),
                              est.uncached(None, tuple(range(70)))],
                 "stats": {k: {q: _r(v) for q, v in d.items()} for k, d in est.stats().items()}, "plans": []}
        for i in (0, 1, 2):
            for steps in ((), (9.0, 11.0, 30.0), (40.0, 41.0, 39.0)):
                for uncached in (40, 400, 4000):
                    if i > 0 and uncached != 40:
                        continue                                 # a continuation's estimate does not read uncached
                    e = est.estimate(i=i, slot=0, question_steps=steps, uncached=uncached)
                    state["plans"].append({"i": i, "steps": list(steps), "uncached": uncached, "estimate": _est(e),
                                           "feasible": [feasible(left, e) for left in lefts],
                                           "n": [[chunk_size(i, k, cap, left, e) for left in lefts]
                                                 for cap, k in caps]})
        out[f"estimators/{name}"] = state
    sims = []
    for a_ms, d_ms in ((50.0, 10.0), (400.0, 30.0)):
        for n_gen, close in ((300, "natural"), (300, "eos"), (5000, "forced")):
            for deadline_ms in (100.0, 500.0, 2000.0, 8000.0, 60000.0):
                for cap in (256, 1024, 4096):
                    s = simulate(a_ms=a_ms, d_ms=d_ms, n_gen=n_gen, close=close, deadline_ms=deadline_ms,
                                 reserve_ms=RESERVE_MS, cap=cap)
                    sims.append([a_ms, d_ms, n_gen, close, deadline_ms, cap,
                                 [[k, _r(v)] for k, v in dataclasses.asdict(s).items()]])
    out["simulate"] = sims
    out["share_end"] = [_r(share_end(now, end, m)) for now in (0.0, 120.0) for end in (500.0, 9000.0) for m in (0, 1, 3)]
    return out


def test_planner_without_converge_is_identical_to_before_early_exit():
    _check(PLANNER_GOLDEN, _planner())
