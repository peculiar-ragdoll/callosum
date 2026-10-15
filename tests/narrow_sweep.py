"""The <=26-option byte-identity sweep: one deterministic run of callo's narrow
readout paths, recorded as data. `tests/data/narrow_sweep.json` holds this module's `snapshot()` taken on the
commit before the wide-choice change, in the development history (CAPTURED_ON); `tests/test_wide_identity_sweep.py`
takes it again on the current code and compares. This module only uses names both commits have, and drops the four
wide-only `Readout` fields after checking their defaults.

Sections (each a dict of named cases):
- `engine`: plain, context and image readouts (n in {2, 3, 10, 26}; named-key and letter-key choices, scores, noul with
  and without criteria) through `Engine.answer_parsed` on `LlamaCppForwardScorer`, whose `http_post` is a recording
  llama-server stand-in (`RecordingUpstream`): every reply is a pseudo-random top-k dump keyed by the prompt, so the
  walk steps, floors and stops differently per case; `/tokenize` greedy-matches a synthetic vocabulary, so the media
  path's retok check sometimes passes and sometimes stops. Each case's engine settings (cache, delimiter, a position-
  bias profile, max_depth, temperature, slot, a second question) come from a hash of its name. Recorded: every
  upstream call (endpoint and exact JSON), every `Readout` field (floats as `float.hex`), the X-Callo-Timings header
  (`ms` pinned) and the answer body.
- `deliberate`: budget > 0 readouts (`answer_parsed(think_budget=)`, as serve calls it) and a multi-budget
  `answer_deliberate` (the think-probe's shape, through `proxy._readout_fields`) on the thinking `FakeUpstream`, over
  HTTP: every `/completion`, `/tokenize` and `/apply-template` body as sent, every `Readout` field, the header (wall
  times pinned) and the answer body.
- `walks`: `divergence_walk` with `partial=()` (when the parameter exists) on every fixture of
  `tests/wide_walk_fixtures.py` plus the guard, swamped and refusal fixtures of `tests/test_marginalize.py` (results or
  messages).
- `profiles`: `check_profile` / `check_think_profile` lines for legacy and current profiles against the CURRENT
  commit's `provenance()` (which carries `wide_contract` after the change), across every warning cause.
- `prompt_sha256`.

No `random`: a sha256 counter (`Rng`) drives every choice, so the fixture does not depend on the Python version.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import platform
import string
import sys
from pathlib import Path

CAPTURED_ON = "ce89e73"
FIXTURE = Path(__file__).parent / "data" / "narrow_sweep.json"
NS = (2, 3, 10, 26)
WIDE_DEFAULTS = {"unread_mass": 0.0, "split_passes": 0, "wide": False, "partial_by_letter": (), "uncovered": 0}


class Rng:
    """A sha256 counter: deterministic on every Python (the `random` module's algorithms are not a contract)."""

    def __init__(self, key: str):
        self.key, self.i = key, 0

    def u(self) -> float:
        self.i += 1
        h = hashlib.sha256(f"{self.key}/{self.i}".encode()).digest()
        return int.from_bytes(h[:7], "big") / float(1 << 56)

    def below(self, n: int) -> int:
        return min(int(self.u() * n), n - 1)

    def choice(self, seq):
        return seq[self.below(len(seq))]

    def sample(self, seq, k: int) -> list:
        pool, out = list(seq), []
        for _ in range(min(k, len(pool))):
            out.append(pool.pop(self.below(len(pool))))
        return out


def hexify(o):
    """Floats as `float.hex` (bit-exact, inf and nan included); tuples as lists."""
    if isinstance(o, float):
        return o.hex()
    if isinstance(o, dict):
        return {str(k): hexify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [hexify(v) for v in o]
    return o


def unpinned(text: str) -> str:
    """A readout pass's body (`n_predict` 1 with `n_probs`) without the neutral sampler pin every readout pass carries
    since the EOG fix (`callo.scorer.readout_pin()`; on ce89e73, which has none, the body as it is). Every readout body
    MUST carry the pin with its exact values, and its raw bytes must be callo's serialisation (`json.dumps`): either
    failing is an AssertionError, so a pass that drops or changes the pin, or a serialisation change, fails the sweep.
    The rest is returned re-serialised from the parsed JSON, key order kept, for the byte comparison with ce89e73."""
    try:
        from callo.scorer import readout_pin  # noqa: PLC0415
    except ImportError:
        return text
    obj = json.loads(text) if text else None
    if not (isinstance(obj, dict) and obj.get("n_predict") == 1 and obj.get("n_probs")):
        return text
    if json.dumps(obj) != text:
        raise AssertionError(f"a readout body is not callo's json.dumps serialisation: {text[:200]}")
    pin = readout_pin()
    missing = object()
    got = {k: obj.get(k, missing) for k in pin}
    if got != pin:
        raise AssertionError(f"a readout body does not carry the exact readout pin: {got} != {pin}")
    return json.dumps({k: v for k, v in obj.items() if k not in pin})


def body_record(endpoint: str, text: str) -> dict:
    """An upstream call, byte for byte (sha256 and length of the exact body, a readout pass's `unpinned`), with its
    top-level keys and n_probs in the clear so a failure says what moved."""
    text = unpinned(text)
    obj = json.loads(text) if text else None
    return {"ep": endpoint, "len": len(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "keys": list(obj) if isinstance(obj, dict) else None,
            "n_probs": obj.get("n_probs") if isinstance(obj, dict) else None}


def readout_record(r) -> dict:
    """Every pre-change `Readout` field; the wide-only fields and `reasks` (no pass here is withheld), when present,
    must be at their defaults."""
    d = dataclasses.asdict(r)
    if d.pop("reasks", 0) != 0:
        raise AssertionError("a narrow sweep readout was re-asked")
    extra = {k: d.pop(k) for k in WIDE_DEFAULTS if k in d}
    if extra and extra != {k: WIDE_DEFAULTS[k] for k in extra}:
        raise AssertionError(f"a narrow readout's wide fields are not at their defaults: {extra}")
    return hexify(d)


# ---- the engine leg: plain, context, image ----------------------------------------------------------------------

L = string.ascii_uppercase
VOCAB = (list(L) + ["Ġ" + c for c in L] + list(string.ascii_lowercase) + ["Ġ" + c for c in string.ascii_lowercase]
         + ["**", "(", "A.", "B)", "Ċ", "Ġthe", "ĠThe", "Answer", ":", "Ġ(", "Ġ**", "<|im_end|>", "x", "y", "Ċ**"]
         + [a + b for a in "AB" for b in L] + ["Ġ" + a + b for a in "AB" for b in L])
_TEXT = [t.replace("Ġ", " ").replace("Ċ", "\n") for t in VOCAB]
_ORDER = sorted(range(len(_TEXT)), key=lambda i: (-len(_TEXT[i]), i))
# `**` twice: appended after the prompt's closing newline it re-tokenizes as `Ċ**`, the media path's retok stop
_SPECIAL = [VOCAB.index(t) for t in ("**", "**", "Ġthe", "Ċ", "Ġ**", "(")]


def _tok(text: str) -> list[int]:
    out, i = [], 0
    while i < len(text):
        for j in _ORDER:
            if text.startswith(_TEXT[j], i):
                out.append(j)
                i += len(_TEXT[j])
                break
        else:
            out.append(1000 + ord(text[i]))
            i += 1
    return out


def _detok(ids) -> str:
    return "".join(_TEXT[i] if 0 <= i < len(_TEXT) else chr(i - 1000) for i in ids)


def _dump(key: str, n_opts: int) -> dict[int, float]:
    """A top-k dump: 35% a dominant answer-neutral prefix token (the walk steps), else a spread over the letters
    and fillers (or the whole vocabulary), sometimes summing to less than 1."""
    rng = Rng(key)
    mode = rng.u()
    if mode < 0.35:
        t = rng.choice(_SPECIAL)
        p = 0.86 + 0.13 * rng.u()
        rest = rng.sample([i for i in range(len(VOCAB)) if i != t], 1 + rng.below(12))
        w = [rng.u() + 1e-3 for _ in rest]
        s = math.fsum(w)
        d = {t: p, **{r: (1 - p) * x / s for r, x in zip(rest, w, strict=True)}}
    else:
        pool = list(range(len(VOCAB)))
        if mode < 0.8:
            m = max(n_opts, 2)
            pool = ([VOCAB.index(c) for c in L[:m]] + [VOCAB.index("Ġ" + c) for c in L[:m]]
                    + [VOCAB.index(c.lower()) for c in L[:m]] + _SPECIAL)
        sel = rng.sample(pool, 1 + rng.below(25))
        w = [(rng.u() + 1e-3) ** rng.choice([1, 3, 10]) for _ in sel]
        s = math.fsum(w) * rng.choice([1.0, 1.0, 1.0, 1.0000001, 1.3])
        d = {t: x / s for t, x in zip(sel, w, strict=True)}
    return {k: math.log(v) for k, v in d.items() if v > 0}


class RecordingUpstream:
    """`http_post` for `LlamaCppForwardScorer`: a deterministic llama-server stand-in that records every call."""

    def __init__(self):
        self.log: list[dict] = []
        self.n_opts = 2

    def __call__(self, url, payload, *a, **kw):
        ep = url.rsplit("/", 1)[-1]
        self.log.append(body_record(ep, json.dumps(payload)))
        if ep == "tokenize":
            return {"tokens": _tok(payload["content"])}
        if ep == "detokenize":
            return {"content": _detok(payload["tokens"])}
        if ep == "completion":
            key = json.dumps(payload["prompt"], sort_keys=True)
            top = sorted(_dump(key, self.n_opts).items(), key=lambda kv: (-kv[1], kv[0]))[:payload["n_probs"]]
            h = int(hashlib.sha256(key.encode()).hexdigest()[:6], 16)
            return {"completion_probabilities": [{"top_logprobs": [{"id": i, "logprob": v} for i, v in top]}],
                    "tokens_evaluated": 10 + h % 50,
                    "timings": {"prompt_n": h % 7, "cache_n": h % 11, "prompt_ms": (h % 1000) / 10}}
        raise AssertionError(f"unexpected upstream call {url}")


def _question(typ: str, n: int, rng: Rng) -> dict:
    if typ == "named":
        return {"type": "choice", "instructions": "Pick one.",
                "criteria": {f"k{j}": rng.choice([None, f"desc {j}"]) for j in range(n)}}
    if typ == "letters":
        return {"type": "choice", "criteria": {k: rng.choice([None, "desc " + k])
                                               for k in rng.sample(string.ascii_lowercase, n)}}
    if typ == "score":
        return {"type": "score", "instructions": "Rate it.", "criteria": [f"level {j}" for j in range(n)]}
    if typ == "noul":
        return {"type": "noul", "instructions": "Is it spam?"}
    return {"type": "noul", "instructions": "Is it spam?", "criteria": {"spam": "unsolicited bulk mail"}}


def engine_cases() -> dict[str, tuple[str, str, int, int]]:
    out = {}
    for path in ("plain", "context", "image"):
        for typ in ("named", "letters", "score"):
            for n in NS:
                for rep in (0, 1):
                    out[f"{path}_{typ}_{n}_{rep}"] = (path, typ, n, rep)
        for typ in ("noul", "noulcrit"):
            for rep in (0, 1):
                out[f"{path}_{typ}_2_{rep}"] = (path, typ, 2, rep)
    return out


def _engine_case(name: str, path: str, typ: str, n: int, up: RecordingUpstream, sets, pb) -> dict:
    from test_engine import DELIM, MARKER, URL, ChatMLTemplate  # noqa: PLC0415

    from callo.engine import Engine  # noqa: PLC0415
    from callo.errors import CalloError  # noqa: PLC0415
    from callo.proxy import timings_header  # noqa: PLC0415
    from callo.scorer import LlamaCppForwardScorer  # noqa: PLC0415
    from callo.systemone import parse_request  # noqa: PLC0415

    rng = Rng(name)
    cache = rng.u() < 0.5
    eng = Engine(LlamaCppForwardScorer("http://u", http_post=up, cache_prompt=cache), ChatMLTemplate(), sets,
                 vision=True, media_marker=MARKER, user_delimiter=DELIM if rng.u() < 0.7 else None,
                 position_bias=pb if rng.u() < 0.4 else None,
                 max_depth=rng.choice([0, 1, 3, 4]) if rng.u() < 0.3 else 3,
                 temperature=rng.choice([1.0, 1.0, 0.7, 1.6]))
    body: dict = {"state": rng.choice(["s", "the input text", "a ** b"]), "questions": {"q1": _question(typ, n, rng)}}
    if rng.u() < 0.3:
        body["questions"]["q2"] = {"type": "choice", "criteria": {"x": None, "y": None, "z": None}}
    if path == "context":
        body["context"] = "some context ** text"
    if path == "image":
        body["images"] = [URL]
    id_slot = rng.choice([None, 0, 2])
    up.log, up.n_opts = [], n
    try:
        req = parse_request(body)
        readouts: list = []
        ans = eng.answer_parsed(req, readouts=readouts, id_slot=id_slot)
        hdr = timings_header([dataclasses.replace(q, ms=1.0) for q in readouts])
        return {"answer": hexify(ans), "readouts": [readout_record(q.readout) for q in readouts], "header": hdr,
                "http": up.log}
    except (CalloError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}", "http": up.log}


def engine_leg() -> dict:
    from test_profile import profile_dict  # noqa: PLC0415

    from callo.engine import SurfaceSets  # noqa: PLC0415
    from callo.profile import load_profile  # noqa: PLC0415

    sets = SurfaceSets(VOCAB, "gpt2")
    pb = load_profile(profile_dict({3: [0.5, 0.3, 0.2], 10: [0.1] * 10, 26: [1 / 26] * 26},
                                   kinds=("plain", "context")))
    up = RecordingUpstream()
    return {name: _engine_case(name, *spec[:3], up, sets, pb) for name, spec in engine_cases().items()}


# ---- the deliberate leg ------------------------------------------------------------------------------------------

def deliberate_cases() -> dict[str, tuple[str, int, str]]:
    out = {}
    for ctx in ("plain", "context"):
        for typ in ("named", "score"):
            for n in NS:
                out[f"{ctx}_{typ}_{n}"] = (typ, n, ctx)
        out[f"{ctx}_noul_2"] = ("noul", 2, ctx)
    return out


def _pinned(qr):
    think = qr.think if qr.think is None else dataclasses.replace(qr.think, gen_wall_ms=1.0, frame_ms=1.0)
    return dataclasses.replace(qr, ms=1.0, think=think)


def _deliberate_case(name: str, typ: str, n: int, ctx: str, started: list) -> dict:
    from fake_upstream import FakeUpstream  # noqa: PLC0415
    from test_think_engine import engine_for  # noqa: PLC0415

    from callo.proxy import _readout_fields, timings_header  # noqa: PLC0415
    from callo.systemone import parse_request  # noqa: PLC0415

    rng = Rng("deliberate/" + name)
    u = FakeUpstream(think="preopened", think_len=rng.choice([3, 17, 70])).start()
    started.append(u)                       # one fresh fake per case; stopped together (each stop waits ~0.5 s)
    e = engine_for(u)
    body: dict = {"state": rng.choice(["the state", "x"]), "questions": {"q": _question(typ, n, rng)}}
    if ctx == "context":
        body["context"] = "ctx"
    req = parse_request(body)
    budget = rng.choice([4, 64])
    u.requests.clear()
    readouts: list = []
    ans = e.answer_parsed(req, readouts=readouts, id_slot=0, think_budget=budget, think_T=1.3,
                          think_calibrated=False)
    rec = {"budget": budget, "answer": hexify(ans), "readouts": [readout_record(q.readout) for q in readouts],
           "header": timings_header([_pinned(q) for q in readouts])}
    q = req.questions[0]
    d = e.answer_deliberate(req.state, q, [1, 8, 64], id_slot=0, context=req.context, seed=7,
                            sampling=e.trace_sampling())
    rec["probe"] = {str(b): [d.readouts[b].k, d.readouts[b].text_fallback, hexify(_readout_fields(
        d.readouts[b].readout))] for b in sorted(d.readouts)}
    rec["http"] = [body_record(x.path, (x.body or b"").decode("utf-8")) for x in u.requests
                   if x.path in ("/completion", "/tokenize", "/apply-template")]
    return rec


def deliberate_leg() -> dict:
    import threading  # noqa: PLC0415
    started: list = []
    try:
        return {name: _deliberate_case(name, *spec, started) for name, spec in deliberate_cases().items()}
    finally:
        stops = [threading.Thread(target=u.stop) for u in started]
        for t in stops:
            t.start()
        for t in stops:
            t.join()


# ---- walks -------------------------------------------------------------------------------------------------------

def _walk_record(divergence_walk, sets, fn, kw) -> dict:
    if "partial" in inspect.signature(divergence_walk).parameters:
        kw = {**kw, "partial": ()}
    try:
        r = divergence_walk(sets, fn, **kw)
    except Exception as e:  # noqa: BLE001  the message is the frozen output
        return {"error": f"{type(e).__name__}: {e}"}
    return {"prefix": list(r.prefix), "option_logprobs": {k: v.hex() for k, v in r.option_logprobs.items()},
            "option_mass": r.option_mass.hex(), "depth": r.depth, "stop": r.stop,
            "token_logprobs": {str(k): v.hex() for k, v in sorted(r.token_logprobs.items())}}


def _f32(x: float) -> float:
    import struct  # noqa: PLC0415
    return struct.unpack("f", struct.pack("f", x))[0]


def walks_leg() -> dict:
    from wide_walk_fixtures import CASES, NEAR_SETS, WALK_SETS, _near_certain  # noqa: PLC0415

    from callo.marginalize import divergence_walk  # noqa: PLC0415

    small = _f32(0.999 * 2.0 ** -24)
    swamped = {1: 0.0, **{tid: _f32(math.log(small)) for tid in range(2, 81)}}
    cd = {"C": frozenset({10}), "D": frozenset({20})}
    extra = {
        "guard_negative_depth": (WALK_SETS, lambda p: {10: 0.0}, {"max_depth": -1}),
        "guard_dominance_below_divergence": (WALK_SETS, lambda p: {10: 0.0}, {"dominance": 0.05,
                                                                               "divergence_mass": 0.5}),
        "guard_dominance_0": (WALK_SETS, lambda p: {10: 0.0}, {"dominance": 0.0}),
        "guard_dominance_1.5": (WALK_SETS, lambda p: {10: 0.0}, {"dominance": 1.5}),
        "guard_divergence_0": (WALK_SETS, lambda p: {10: 0.0}, {"divergence_mass": 0.0}),
        "guard_divergence_1.5": (WALK_SETS, lambda p: {10: 0.0}, {"divergence_mass": 1.5}),
        "guard_dominance_nan": (WALK_SETS, lambda p: {10: 0.0}, {"dominance": float("nan")}),
        "guard_empty": (WALK_SETS, lambda p: {}, {}),
        "guard_nan": (WALK_SETS, lambda p: {10: float("nan"), 20: -1.0}, {}),
        "guard_positive": (WALK_SETS, lambda p: {10: 2.0, 20: -1.0}, {}),
        "above_zero_cd": (cd, lambda p: {10: 5e-7, 20: -30.0}, {}),
        "swamped": (NEAR_SETS, lambda p: swamped, {}),
        "refused_1e-3": (NEAR_SETS, lambda p: _near_certain(1e-3), {}),
        "refused_1e-4": (NEAR_SETS, lambda p: _near_certain(1e-4), {}),
        "refused_two_certain": (NEAR_SETS, lambda p: {1: 0.0, 41: 0.0}, {}),
        "refused_mixed": (NEAR_SETS, lambda p: {1: math.log(0.999), 41: math.log(0.998)}, {}),
    }
    out = {name: _walk_record(divergence_walk, *spec) for name, spec in CASES.items()}
    out.update({name: _walk_record(divergence_walk, *spec) for name, spec in extra.items()})
    return out


# ---- profile guards ----------------------------------------------------------------------------------------------

_LEGACY = {"prompt_sha256": "8b718d40420bda901afbd7e74a8e4aa2ee24b7415a18e23e248208f5f3d79f8f",
           "chat_template_sha256": hashlib.sha256(b"{{ messages }}").hexdigest(), "build_info": "b1-926862e",
           "mmproj": None, "user_delimiter": None, "cache_prompt": False, "server_args": None}
_LEGACY_PROVS = {
    "legacy": _LEGACY,
    "legacy_cached": {**_LEGACY, "user_delimiter": "<|im_start|>user", "cache_prompt": True},
    "legacy_cached_delim": {**_LEGACY, "user_delimiter": "<|im_start|>user", "cache_prompt": True,
                            "cache_delimiter": True},
    "legacy_build": {**_LEGACY, "build_info": "b2", "server_args": ["-ctk", "q8_0"]},
    "legacy_extra_field": {**_LEGACY, "future_field": "x"},
}


def _nows() -> dict:
    from callo.profile import provenance  # noqa: PLC0415
    base = {"chat_template": "{{ messages }}", "build_info": "b1-926862e", "mmproj": None, "user_delimiter": None,
            "cache_prompt": False}
    return {"same": provenance(**base),
            "build": provenance(**{**base, "build_info": "b9999-x"}),
            "cache": provenance(**{**base, "cache_prompt": True}),
            "cache_delim": provenance(**{**base, "cache_prompt": True, "user_delimiter": "<|im_start|>user"}),
            "template": provenance(**{**base, "chat_template": "{{ other }}"}),
            "mmproj": provenance(**{**base, "mmproj": "/m/mmproj-BF16.gguf"}),
            "args": provenance(**base, server_args=("-ctk", "q8_0")),
            "none": None}


def profiles_leg() -> dict:
    import test_think  # noqa: PLC0415
    from test_profile import KEY, profile_dict  # noqa: PLC0415

    from callo.errors import CalloError  # noqa: PLC0415
    from callo.profile import check_profile, load_profile  # noqa: PLC0415
    from callo.think import check_think_profile, load_think_profile  # noqa: PLC0415

    nows = _nows()
    out: dict = {}
    for now_name, now in nows.items():
        provs = {**_LEGACY_PROVS, "current": now if now is not None else _LEGACY}
        for pname, prov in provs.items():
            for variant, kw in (("clean", {}), ("busy", {"state": "busy"}), ("empty", {"entries": {}})):
                pb = load_profile(profile_dict(prov=prov, **kw))
                try:
                    lines = check_profile(pb, model_key=KEY, native=False, provenance=now)
                except CalloError as e:
                    lines = [f"CalloError: {e}"]
                out[f"bias/{pname}/{variant}/{now_name}"] = lines
            think_variants = {
                "plain": {},
                "busy": {"server_state": {"state": "busy"}, "audit": {"latency_model": "unverified"},
                         "drift": {"flagged": True}, "thinking": {"capable": False}},
                "chip_t0only": {"environment": {"host": {"fingerprint": None, "chip": "Apple M3 Ultra"},
                                                "build_info": "b1-926862e", "server": {"position_bias_sha256": "x"}},
                                "budgets": {"0": {"T": 1.4}}},
            }
            for variant, over in think_variants.items():
                tp = load_think_profile(test_think.profile_dict(provenance=dict(prov), **over))
                for env_name, env in (
                        ("same", {"host_fingerprint": "sha256:" + "h" * 64, "build_info": "b1-926862e",
                                  "position_bias_sha256": None, "temperature_from_profile": True,
                                  "default_budget": 256}),
                        ("moved", {"host_fingerprint": "sha256:" + "x" * 64, "build_info": "b2",
                                   "position_bias_sha256": "abc", "temperature_from_profile": True,
                                   "default_budget": 512, "host_chip": "Apple M4 Max"}),
                        ("unknown", {"host_fingerprint": None, "build_info": "b1-926862e",
                                     "position_bias_sha256": None, "temperature_from_profile": False,
                                     "default_budget": 0})):
                    out[f"think/{pname}/{variant}/{env_name}/{now_name}"] = check_think_profile(
                        tp, provenance=now, **env)
    return out


def pack_lines(cases: dict[str, list[str]]) -> dict:
    """The profile lines as a table of distinct lines and, per case, their indices (the fixture stays small)."""
    table: list[str] = []
    at: dict[str, int] = {}
    for lines in cases.values():
        for ln in lines:
            if ln not in at:
                at[ln] = len(table)
                table.append(ln)
    return {"lines": table, "cases": {k: [at[ln] for ln in v] for k, v in cases.items()}}


def unpack_lines(packed: dict) -> dict[str, list[str]]:
    return {k: [packed["lines"][i] for i in v] for k, v in packed["cases"].items()}


def snapshot() -> dict:
    from callo.profile import prompt_sha256  # noqa: PLC0415
    return {"prompt_sha256": prompt_sha256(), "engine": engine_leg(), "deliberate": deliberate_leg(),
            "walks": walks_leg(), "profiles": pack_lines(profiles_leg())}


if __name__ == "__main__":
    import callo
    snap = snapshot()
    snap["_meta"] = {"captured_on": CAPTURED_ON, "callo": callo.__file__.split("/src/")[0].rsplit("/", 1)[-1],
                     "platform": f"{platform.system()}-{platform.machine()}", "python": platform.python_version()}
    Path(sys.argv[1]).write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(callo.__file__, {k: len(v) for k, v in snap.items() if isinstance(v, dict)})
