"""Think budgets, the pure half: the thinking frame read off the chat
template, the pinned trace sampler, seeds, the token-built readout prefix and the think profile `callo serve
--think-profile` loads.

A model is CAPABLE of a deliberate readout when its template, rendered with `enable_thinking` on and off, shows a
`<...think...>` block callo can open and close: the close tag is the last such tag in the thinking-off render,
the close suffix is "\\n" plus everything from that tag on (`"\\n</think>\\n\\n"` on Qwen3.5, Sharp, MiniCPM5 and
Qwen3), and the opening is either already in the thinking-on render (pre-opened) or synthesised from the empty block
the thinking-off render adds (Qwen3 dense and the original Qwen3 MoE). Everything else is a reason the model
is served and calibrated at budget 0 only; `unsupported_think_format` names a template that thinks in a
format callo cannot read (Gemma 4's channel tokens), never "cannot think".

The trace sampler is PINNED: every generation sends every sampler field llama-server serialises, so no
server flag can move a trace. Under the default policy `model` the four trace fields (and the other
`general.sampling.*` keys a GGUF carries) come from the GGUF, with callo's 0.6 / 0.95 / 20 / 0 filling the rest;
`fixed` is callo's values for every model, and a `temp=...,top_p=...` spec is the user's.

The measurement half (`callo calibrate --think-budgets` and `callo compare` build on it): the per-budget
latency model from one generation at Bmax, the metrics (accuracy, NLL, ECE10, T_B, the 2-fold cross-fit), the paired
cluster bootstrap and the latency percentiles' intervals, and the frontier over (model, budget) points with the `≈`
mark and the default pick, "the fastest point within noise of the best".
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import random
import re
import struct
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np

from . import ggufkv
from .calibrate import Row, ece, fit_temperature, nll
from .deadline import check_deadline_points, check_latency
from .errors import CalloError, InvalidInput
from .modelkey import KEY_RE

log = logging.getLogger("callo")
MAX_BUDGET = 32768
METRICS_VERSION = "callo-think-metrics-1"
PROFILE_FORMAT = 1
CLOSES = ("natural", "forced", "eos")
_CLOSE_TAG = re.compile(r"</[^<>]*think[^<>]*>")
_PRE_OPENED = re.compile(r"<[^<>/]*think[^<>]*>\s*$")
_EMPTY_BLOCK = re.compile(r"\s*(<[^<>/]*think[^<>]*>)\s*</[^<>]*think[^<>]*>\s*", re.S)
Tokenize = Callable[..., Sequence[int]]        # tokenize(text, *, add_special=..., parse_special=...) -> ids


# ---- the frame -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ThinkFrame:
    """What a deliberate generation needs from the template, derived ONCE at startup and kept on the engine."""
    tag: str                            # the close tag: the generation's stop string
    close: str                          # the close suffix: "\n" + the thinking-off render from the tag on
    post: str                           # the close suffix after the tag ("\n\n")
    opening: str                        # "" when the template pre-opens the block, else the synthesised open tag + "\n"
    close_tokens: tuple[int, ...]       # tok(close), parse_special, no BOS
    post_tokens: tuple[int, ...]        # tok(post)
    tag_tokens: tuple[int, ...]         # tok(tag): one token on every model seen

    @property
    def synthesized_open(self) -> bool:
        return bool(self.opening)

    @property
    def tag_single(self) -> bool:
        return len(self.tag_tokens) == 1

    def info(self) -> dict:
        return {"close": self.close, "synthesized_open": self.synthesized_open, "tag": self.tag,
                "tag_single": self.tag_single}


def detect_frame(on: str, off: str, tokenize: Tokenize) -> tuple[ThinkFrame | None, str | None]:
    """On the two renders of one message list: (the frame, None) or (None, the reason). `on` is the render with
    `enable_thinking` true, `off` with it false."""
    closes = list(_CLOSE_TAG.finditer(off))
    if not closes:
        return None, "no_think_block" if on == off else "unsupported_think_format"
    m = closes[-1]
    tag, close, post = m.group(0), "\n" + off[m.start():], off[m.end():]
    if _PRE_OPENED.search(on):
        opening = ""
    else:
        block = _EMPTY_BLOCK.fullmatch(off[len(on):]) if off.startswith(on) else None
        if block is None:
            return None, "not_preopened"
        opening = block.group(1) + "\n"
    try:
        def tok(text: str) -> tuple[int, ...]:
            return tuple(int(t) for t in tokenize(text, add_special=False, parse_special=True))
        return ThinkFrame(tag, close, post, opening, tok(close), tok(post), tok(tag)), None
    except Exception as e:  # noqa: BLE001 -- a tokenizer failure at startup is a reason, never a crash
        return None, f"render_failed: tokenize: {type(e).__name__}: {e}"


def frame_from_template(render: Callable[[bool], str], tokenize: Tokenize) -> tuple[ThinkFrame | None, str | None]:
    """`detect_frame` over `render(thinking)`, with a render failure as the reason `render_failed: <error>`."""
    try:
        on, off = render(True), render(False)
    except Exception as e:  # noqa: BLE001
        return None, f"render_failed: {type(e).__name__}: {e}"
    return detect_frame(on, off, tokenize)


def check_on_prompt(on: str, frame: ThinkFrame) -> str:
    """The generation prompt of one deliberate question: the thinking-on render, plus the synthesised
    opening when the frame has one. Raises ValueError when the render does not end where the frame says."""
    if frame.synthesized_open:
        if _PRE_OPENED.search(on):
            raise ValueError("the thinking-on prompt pre-opens the think block, but the frame synthesises the opening")
        return on + frame.opening
    if not _PRE_OPENED.search(on):
        raise ValueError(f"the thinking-on prompt does not end inside an opened think block (...{on[-60:]!r})")
    return on


# ---- the pinned sampler ----------------------------------------------------------------------------

CALLO_SAMPLING: dict[str, Any] = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
TRACE_FIELDS = ("temperature", "top_p", "top_k", "min_p")
# llama.cpp's default sampler chain (common/common.h:263-273 @ 4ebdf2c).
DEFAULT_SAMPLERS = ("penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature")
# The neutral pins: every other sampler field llama-server serialises (server-task.cpp:95-143 @ 4ebdf2c), at the
# value that makes it inert. `dry_penalty_last_n` 64 against llama.cpp's -1 is inert because dry_multiplier is 0.
PINNED_SAMPLER: dict[str, Any] = {
    "dynatemp_range": 0.0, "dynatemp_exponent": 1.0, "top_n_sigma": -1.0, "xtc_probability": 0.0,
    "xtc_threshold": 0.1, "typical_p": 1.0, "repeat_last_n": 64, "repeat_penalty": 1.0, "presence_penalty": 0.0,
    "frequency_penalty": 0.0, "dry_multiplier": 0.0, "dry_base": 1.75, "dry_allowed_length": 2,
    "dry_penalty_last_n": 64, "dry_sequence_breakers": ["\n", ":", "\"", "*"], "mirostat": 0, "mirostat_tau": 5.0,
    "mirostat_eta": 0.1, "adaptive_target": -1.0, "adaptive_decay": 0.9, "min_keep": 0,
    "samplers": list(DEFAULT_SAMPLERS), "logit_bias": [], "ignore_eos": False, "n_probs": 0,
    "post_sampling_probs": False, "backend_sampling": False, "reasoning_budget_tokens": -1}
SAMPLER_FIELDS = frozenset((*TRACE_FIELDS, *PINNED_SAMPLER))
# /props default_generation_settings.params keys that cannot move a trace: callo sets them per request
# (seed, n_predict, stop, ...) or they are rendering / timing / speculative options.
INERT_PROPS = frozenset({"seed", "n_predict", "max_tokens", "n_keep", "n_discard", "stream", "stop", "chat_format",
                         "reasoning_format", "reasoning_in_content", "generation_prompt", "thinking_forced_open",
                         "timings_per_token", "preserved_tokens", "grammar_lazy", "grammar_triggers", "lora",
                         "id_slot", "cache_prompt", "return_tokens", "return_progress", "response_fields",
                         "t_max_predict_ms", "t_max_prompt_ms", "n_indent", "n_cmpl", "message_delimiters"})
_NEUTRAL = (None, False, 0, 0.0, "", [], {})
# `general.sampling.<key>` -> the request field it sets, as common/common.cpp:1106-1164 maps them into llama.cpp's
# own defaults; `sequence` is the sampler chain (';'-separated sampler names).
GGUF_SAMPLING = {"temp": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
                 "xtc_probability": "xtc_probability", "xtc_threshold": "xtc_threshold",
                 "penalty_last_n": "repeat_last_n", "penalty_repeat": "repeat_penalty", "mirostat": "mirostat",
                 "mirostat_tau": "mirostat_tau", "mirostat_eta": "mirostat_eta", "sequence": "samplers"}
# common_sampler_types_from_names (common/sampling.cpp:959-1002 @ 4ebdf2c): the canonical names, their kebab-case and
# no-underscore aliases, and three misc aliases; matched after an ASCII lower-casing, an unknown name dropped.
_CANONICAL = ("dry", "top_k", "top_p", "top_n_sigma", "typ_p", "min_p", "temperature", "xtc", "infill", "penalties",
              "adaptive_p")
_SAMPLER_NAMES = {**{a: c for c in _CANONICAL if "_" in c for a in (c.replace("_", "-"), c.replace("_", ""))},
                  "nucleus": "top_p", "temp": "temperature", "typ": "typ_p", **{c: c for c in _CANONICAL}}
_INT_FIELDS = ("top_k", "repeat_last_n", "mirostat")
# The C parsers llama.cpp reads the header's text form with: strtol base 10 and strtof (decimal, hex, inf, nan), each
# taking the longest valid prefix after leading whitespace.
_STRTOL = re.compile(r"[ \t\n\v\f\r]*([+-]?[0-9]+)")
_STRTOF = re.compile(r"[ \t\n\v\f\r]*([+-]?(?:0x(?:[0-9a-f]+\.?[0-9a-f]*|\.[0-9a-f]+)(?:p[+-]?[0-9]+)?"
                     r"|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?|inf(?:inity)?|nan(?:\([0-9a-z_]*\))?))", re.I)


def unpinnable(params: Mapping[str, Any]) -> str | None:
    """The first key of `/props default_generation_settings.params` a request cannot override, or None. A
    non-empty server-wide `grammar` (a request's empty grammar is ignored, server-schema.cpp:270-272), or a key on
    neither the pinned nor the inert list with a non-neutral value."""
    grammar = params.get("grammar")
    if grammar:
        return "grammar"
    for key in sorted(params):
        if key == "grammar" or key in SAMPLER_FIELDS or key in INERT_PROPS or key.startswith("speculative"):
            continue
        if params[key] not in _NEUTRAL:
            return key
    return None


def unknown_props_keys(params: Mapping[str, Any]) -> list[str]:
    """The `/props default_generation_settings.params` keys on neither the pinned nor the inert list (sorted): a new
    llama.cpp knob `unpinnable` let through because its value was neutral. serve logs them at startup so a knob whose
    neutral value is not one of _NEUTRAL (top_n_sigma's -1, say) is visible."""
    return sorted(k for k in params if k != "grammar" and k not in SAMPLER_FIELDS and k not in INERT_PROPS
                  and not k.startswith("speculative"))


def check_pinned(pinned: Any) -> str | None:
    """Why `pinned` is not a full pinned sampler dict, or None when it is: an object with EVERY field of
    SAMPLER_FIELDS and no other key, each value of its pin's type (`top_k` and the other int pins an int, a bool pin a
    bool, a list pin a list), every number finite, every value in `_PIN_RANGES` (what llama-server takes from a
    request as given, never clamped or refused, plus the two pins a trace that closes needs), and every pin in
    `_NEUTRAL_PINS` at its neutral value: those samplers no policy sets (no GGUF key, no spec field), so a value that
    turns one on is never a policy's. Shared by `load_think_profile` (phase 1, exit 1) and the probe's request check
    (422: a partial or extended dict would silently unpin the trace or override the generation's own keys).
    What it does not check: the fields a GGUF may set (the trace fields, xtc, the repetition penalty, mirostat and the
    `samplers` chain, whose order a GGUF's `sequence` may change) are taken at any value llama-server takes as given.
    `resolve_sampling` skips a GGUF value this refuses, so every dict it resolves passes."""
    if not isinstance(pinned, Mapping):
        return "must be an object"
    extra = sorted(set(pinned) - SAMPLER_FIELDS)
    if extra:
        return f"has {extra[0]!r}, which is not a sampler field"
    for f in sorted(SAMPLER_FIELDS):
        if f not in pinned:
            return f"lacks {f}"
        why = _pin_fault(f, pinned[f])
        if why is not None:
            return why
        neutral = _NEUTRAL_PINS.get(f)
        if neutral is not None and not neutral[0](pinned[f]):
            return f"{f} must be {neutral[1]} (no sampling policy sets it), got {pinned[f]!r}"
    return None


def _pin_fault(f: str, v: Any) -> str | None:
    """Why `v` cannot be the pinned value of sampler field `f`, or None. The type is the pin's own (callo's value)."""
    pin = CALLO_SAMPLING.get(f, PINNED_SAMPLER.get(f))
    if isinstance(pin, bool):
        if type(v) is not bool:
            return f"{f} must be a boolean, got {v!r}"
    elif isinstance(pin, int):
        if type(v) is not int:
            return f"{f} must be an int, got {v!r}"
    elif isinstance(pin, float):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return f"{f} must be a number, got {v!r}"
        if not math.isfinite(v):
            return f"{f} must be finite, got {v!r}"
    elif f == "logit_bias":                     # llama-server takes an array of pairs or an object of them
        if not isinstance(v, (list, Mapping)):
            return f"{f} must be a list or an object, got {v!r}"
    elif isinstance(pin, list) and not isinstance(v, list):
        return f"{f} must be a list, got {v!r}"
    rule = _PIN_RANGES.get(f)
    if rule is not None and not rule[0](v):
        return f"{f} must be {rule[1]}, got {v!r}"
    return None


@dataclass(frozen=True)
class SamplingPolicy:
    kind: str                           # "model" | "fixed" | "spec"
    values: dict[str, Any]              # kind "spec": the user's trace fields

    @property
    def label(self) -> str:
        if self.kind != "spec":
            return self.kind
        return ",".join(f"{'temp' if k == 'temperature' else k}={v}" for k, v in self.values.items())


# The pinned dict's ranges (`check_pinned`, and `--think-sampling`'s four trace fields through `_pin_fault`): exactly
# what llama-server takes from a request AS GIVEN
# (tools/server/server-schema.cpp @ 4ebdf2c). A hard limit there is a 400 and a soft limit a silent clamp; either way
# the value is not the one the server would use, so the pin cannot record it. temperature [0, inf) (0 is greedy, as
# llama.cpp reads it); top_p, min_p, xtc_probability, xtc_threshold [0, 1]; mirostat 0 / 1 / 2 (soft: 7 would run as 2,
# and llama.cpp's own sampler asserts on it); adaptive_target <= 1; repeat_last_n, dry_allowed_length,
# dry_penalty_last_n, min_keep [0, INT32_MAX] and adaptive_decay [0, 0.99] (hard); dry_base >= 1 (below, the server
# substitutes its default); dry_sequence_breakers a non-empty list of strings (else a 400); samplers names
# common_sampler_types_from_names knows (an unknown one is dropped silently). One exception: top_k is any int32, as a
# spec takes it; the server clamps a negative top_k to 0, which disables top-k exactly as <= 0 does. Two pins the
# trace depends on, whatever the server accepts: `ignore_eos` false (a trace that ignores EOS never closes on its
# own) and `reasoning_budget_tokens` -1 (the server's own budget would cut the trace; callo's budget is n_predict).
_INT32_MAX = 2**31 - 1
_PIN_RANGES: dict[str, tuple[Callable[[Any], bool], str]] = {
    "temperature": (lambda v: v >= 0, ">= 0 (0: greedy)"),
    "top_p": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "top_k": (lambda v: -(2**31) <= v < 2**31, "a 32-bit integer (<= 0 disables top-k)"),
    "min_p": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "xtc_probability": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "xtc_threshold": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "mirostat": (lambda v: v in (0, 1, 2), "0, 1 or 2"),
    "adaptive_target": (lambda v: v <= 1, "<= 1 (negative: disabled)"),
    "adaptive_decay": (lambda v: 0 <= v <= 0.99, "in [0, 0.99]"),
    "repeat_last_n": (lambda v: 0 <= v <= _INT32_MAX, "in [0, 2^31 - 1]"),
    "dry_allowed_length": (lambda v: 0 <= v <= _INT32_MAX, "in [0, 2^31 - 1]"),
    "dry_penalty_last_n": (lambda v: 0 <= v <= _INT32_MAX, "in [0, 2^31 - 1]"),
    "min_keep": (lambda v: 0 <= v <= _INT32_MAX, "in [0, 2^31 - 1]"),
    "dry_base": (lambda v: v >= 1, ">= 1"),
    "reasoning_budget_tokens": (lambda v: v == -1, "-1"),
    "ignore_eos": (lambda v: v is False, "false"),
    "samplers": (lambda v: all(isinstance(n, str) and n.encode().lower().decode() in _SAMPLER_NAMES for n in v),
                 f"llama.cpp's samplers ({', '.join(_CANONICAL)}, or their aliases)"),
    "dry_sequence_breakers": (lambda v: bool(v) and all(isinstance(x, str) for x in v), "a non-empty list of strings"),
}
# The pins no sampling policy sets (neither a GGUF's general.sampling.* nor a spec), at a value that leaves their
# sampler off: any other value turns a sampler on that no policy asked for (`check_pinned`).
_NEUTRAL_PINS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "logit_bias": (lambda v: len(v) == 0, "empty"),
    "dynatemp_range": (lambda v: v == 0, "0"),
    "typical_p": (lambda v: v == 1, "1"),
    "presence_penalty": (lambda v: v == 0, "0"),
    "frequency_penalty": (lambda v: v == 0, "0"),
    "dry_multiplier": (lambda v: v == 0, "0"),
    "top_n_sigma": (lambda v: v <= 0, "<= 0 (off)"),
}


def parse_sampling(text: str) -> SamplingPolicy:
    """`--think-sampling`: `model`, `fixed`, or `temp=...,top_p=...,top_k=...,min_p=...` (any subset, each once),
    each value in the range `check_pinned` takes (temperature >= 0, 0 greedy; top_p and min_p in [0, 1]; top_k an
    int32, <= 0 off), so a spec and the pinned dict it resolves to agree. Raises InvalidInput."""
    if text in ("model", "fixed"):
        return SamplingPolicy(text, {})
    values: dict[str, Any] = {}
    for part in text.split(","):
        key, sep, raw = part.strip().partition("=")
        key = "temperature" if key == "temp" else key
        if not sep or key not in TRACE_FIELDS or key in values:
            raise InvalidInput(f"--think-sampling: expected model, fixed, or temp=..,top_p=..,top_k=..,min_p=.. "
                               f"(each at most once), got {text!r}")
        try:
            value: Any = int(raw) if key == "top_k" else float(raw)
        except ValueError:
            raise InvalidInput(f"--think-sampling: {key} must be a number, got {raw!r}") from None
        why = _pin_fault(key, value)
        if why is not None:
            raise InvalidInput(f"--think-sampling: {why}")
        values[key] = value
    return SamplingPolicy("spec", values)


@dataclass(frozen=True)
class Sampling:
    """The resolved trace sampler: the full pinned dict, where each trace field came from, the GGUF's own keys, and why
    the GGUF's keys are missing when gguf-py could not parse its header."""
    policy: str
    pinned: dict[str, Any]
    origin: dict[str, str]              # trace field -> "gguf" | "callo" | "user"
    gguf: dict[str, Any]                # the GGUF's general.sampling.* values, short keys
    gguf_error: str | None = None       # set when the GGUF metadata could not be read (callo's sampler applies)

    @property
    def trace(self) -> str:
        p = self.pinned
        return f"{p['temperature']:g}/{p['top_p']:g}/{p['top_k']}/{p['min_p']:g}"

    def as_dict(self) -> dict:
        d = {"policy": self.policy, "pinned": dict(self.pinned), "origin": dict(self.origin), "gguf": dict(self.gguf)}
        if self.gguf_error is not None:
            d["gguf_error"] = self.gguf_error
        return d


class GGUFSampling(dict):
    """`read_gguf_sampling`'s result: the `general.sampling.*` values by short key, and `.error`, set (with the dict
    empty) when gguf-py could not parse the header."""

    def __init__(self, values: Mapping[str, Any] = (), error: str | None = None):
        super().__init__(values)
        self.error = error


def _llama_str(value: Any, size: int) -> str:
    """The text llama.cpp reads a header value through (gguf_kv_to_str, src/llama-impl.cpp:152-204), cut to the
    `size`-byte buffer llama_model_meta_val_str copies it into: std::to_string ("%f" for a float), "true"/"false"
    for a bool, an array as its bracketed list."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        text = f"{value:f}"                         # std::to_string: "%f"
    elif isinstance(value, (list, tuple)):
        text = "[" + ", ".join('"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"' if isinstance(v, str)
                               else _llama_str(v, 1 << 20) for v in value) + "]"
    else:
        text = str(value)
    return text.encode()[:size - 1].decode(errors="ignore")


def _float32_shortest(x: float) -> float:
    """The float32 nearest x, as the shortest decimal that round-trips through float32 (0.949999988... -> 0.95)."""
    f = struct.unpack("f", struct.pack("f", x))[0]
    for digits in range(1, 10):
        short = float(f"{f:.{digits}g}")
        if struct.unpack("f", struct.pack("f", short))[0] == f:
            return short
    return f


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def stof(t: str) -> float:
    """The float32 `strtof` gives for decimal text whose magnitude is at most FLT_MAX (a transcription `temperature`
    that passed `numbers_ok`; a `general.sampling.*` string in `_llama_float`), as a Python float. The decimal is
    rounded to float32 ONCE, to nearest, ties to even, as `strtof` does: `float()` then float32 would round twice and
    can land one float step off near a midpoint. For a
    transcription it means llama-server parses back exactly upstream's value (not byte equality: upstream's JSON holds
    that float widened to a double, and callo's the shortest repr of the same double)."""
    d = Decimal(t)
    if d == 0:
        return float(t)                                           # 0.0 or -0.0, as stof
    x, a = abs(float(t)), d.copy_abs()
    f = f32(x)                                                   # within one float32 step of the exact value
    (bits,) = struct.unpack("<I", struct.pack("<f", f))
    if Decimal(f) != a:
        g = struct.unpack("<f", struct.pack("<I", bits + 1 if Decimal(f) < a else bits - 1))[0]
        if not math.isinf(g):                                     # past FLT_MAX: numbers_ok bounds a far below the
            mid = Decimal((f + g) / 2)                            # midpoint, so f stands; (f+g)/2 is exact in a double
            if a != mid and (a > mid) == (g > f):
                f = g
            elif a == mid and bits % 2:                           # a tie goes to the even significand
                f = g
    return math.copysign(f, d)


def _llama_float(value: Any) -> float:
    """strtof(to_string(value)) as common.cpp's get_float reads it, as the shortest float32 decimal. ValueError when
    strtof finds no number, or the value is not finite (llama.cpp would take a nan or inf; a request cannot carry
    one, so callo's pin stays)."""
    m = _STRTOF.match(_llama_str(value, 128))
    if m is None:
        raise ValueError(f"not a number: {value!r}")
    text = m.group(1).lower()
    hexa = "0x" in text
    x = float.fromhex(text) if hexa else float(text.split("(")[0])
    if not math.isfinite(x) or abs(x) > 3.4028234663852886e38:
        raise ValueError(f"not finite: {value!r}")
    try:
        return _float32_shortest(x if hexa else stof(text))     # decimal text: rounded to float32 once
    except InvalidOperation as e:   # an exponent beyond Decimal's range: the key is skipped and callo's pin
        # stays, deliberately unlike llama.cpp, whose strtof would read it as 0.0
        raise ValueError(f"not a number callo can round: {value!r}") from e


def _coerce(field: str, value: Any) -> Any:
    """One `general.sampling.*` value as llama.cpp's common_init_sampler_from_model reads it (common.cpp:1106-1164):
    the sequence split on ';' into sampler names (all unknown: an empty chain; an empty value: the key is skipped),
    an int through strtol (truncating, a bool's "true" refused) wrapped to int32, `repeat_last_n` then clamped to 0
    as llama_sampler_init_penalties does (a request's must be >= 0), a float through strtof. ValueError: the key is
    skipped and callo's pin stays."""
    if field == "samplers":
        text = _llama_str(value, 512)
        if not text:
            raise ValueError("an empty sequence")
        chain = [_SAMPLER_NAMES[n] for n in (part.encode().lower().decode() for part in text.split(";"))
                 if n in _SAMPLER_NAMES]
        if not chain:
            log.warning("the GGUF's general.sampling.sequence %r names no sampler llama.cpp knows: an empty sampler "
                        "chain, as llama.cpp would use (traces sample the raw distribution, an effective temperature "
                        "of 1)", text)
        return chain
    if field in _INT_FIELDS:
        m = _STRTOL.match(_llama_str(value, 64))
        if m is None:
            raise ValueError(f"not an integer: {value!r}")
        v = max(-(2**63), min(2**63 - 1, int(m.group(1))))       # strtol saturates at LONG_MIN/MAX (LP64) ...
        v = (v + 2**31) % 2**32 - 2**31                            # ... and the int32_t assignment wraps
        # llama-sampler.cpp:3209 clamps penalty_last_n to 0 (penalties off), and llama-server refuses a request's
        # repeat_last_n below 0 (server-schema.cpp:126-128): the clamp, never an omitted (unpinned) field.
        return max(v, 0) if field == "repeat_last_n" else v
    return _llama_float(value)


def resolve_sampling(policy: SamplingPolicy, gguf: Mapping[str, Any] | None = None, *,
                     gguf_error: str | None = None) -> Sampling:
    """The full pinned request dict under `policy`. `gguf` is `read_gguf_sampling`'s dict, whose
    `.error` is recorded; `gguf_error` gives that error for a plain dict (`/callo/stats.think.gguf_sampling` with
    `think.gguf_error`), and wins when both are given."""
    gguf_error = gguf_error if gguf_error is not None else getattr(gguf, "error", None)
    gguf = dict(gguf or {})
    pinned: dict[str, Any] = {k: list(v) if isinstance(v, list) else v
                              for k, v in {**PINNED_SAMPLER, **CALLO_SAMPLING}.items()}
    origin = dict.fromkeys(TRACE_FIELDS, "callo")
    if policy.kind == "spec":
        for k, v in policy.values.items():
            pinned[k], origin[k] = v, "user"
    elif policy.kind == "model":
        for short, field in GGUF_SAMPLING.items():
            if short in gguf:
                try:
                    value = _coerce(field, gguf[short])
                except (TypeError, ValueError, OverflowError):
                    continue                            # a value llama.cpp would not read: callo's pin stays
                why = _pin_fault(field, value)
                if why is not None:                     # check_pinned would refuse it: callo's pin stays
                    log.warning("the GGUF's general.sampling.%s (%r): %s; callo's pin %r stays", short, gguf[short],
                                why, pinned[field])
                    continue
                pinned[field] = value
                if field in origin:
                    origin[field] = "gguf"
    return Sampling(policy.label, pinned, origin, gguf, gguf_error)


def read_gguf_sampling(path: str | Path) -> GGUFSampling:
    """The GGUF's `general.sampling.*` keys (header only; `temp`, `top_p`, ...), or {} when it has none. A float is
    recorded as the value llama.cpp ends up with (strtof of "%f"), as its shortest float32 decimal. InvalidInput when
    the file cannot be opened or is not a GGUF; when it is one gguf-py cannot parse (a malformed header), a warning
    and an empty result whose `.error` says so: callo's sampler applies to every field."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
    except OSError as e:
        raise InvalidInput(f"{path} is not a readable GGUF: {type(e).__name__}: {e}") from e
    if magic != b"GGUF":
        raise InvalidInput(f"{path} is not a readable GGUF: no GGUF magic")
    try:
        r = ggufkv.open_kv(path)
    except Exception as e:  # noqa: BLE001 -- gguf-py raises assorted errors on a header it cannot parse
        error = f"the GGUF metadata could not be read (gguf-py: {type(e).__name__}: {e})"
        log.warning("cannot read general.sampling.* from %s: %s; callo's trace sampler applies to every field",
                    path, error)
        return GGUFSampling(error=error)
    out = GGUFSampling()
    for name, field in r.fields.items():
        if name.startswith("general.sampling."):
            v = field.contents()
            v = v.item() if hasattr(v, "item") else v
            if isinstance(v, float) and not isinstance(v, bool):
                try:
                    v = _llama_float(v)
                except ValueError:
                    pass                                # nan / inf: recorded as stored, never pinned
            out[name[len("general.sampling."):]] = v
    return out


# ---- seeds and the readout prefix --------------------------------------------------

def think_seed(*parts: str) -> int:
    """A uint31: the first 4 bytes of sha256(parts joined by NUL), masked. calibrate: (seed, item_id, qid); serve:
    (request.digest, qid)."""
    h = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def readout_prefix(on_tokens: Sequence[int], gen_tokens: Sequence[int], close: str, k: int,
                   frame: ThinkFrame) -> tuple[int, ...] | None:
    """The token prefix the answer slot is read from, for the cut at k, or None when the text fallback
    applies (a natural close whose last token is not the single tag token). Nothing is stripped: the prefix extends
    the slot's cache exactly (2 new tokens after a natural close, 1 + len(close_tokens) after a forced one)."""
    if close not in CLOSES:
        raise ValueError(f"unknown close {close!r}")
    n_gen = len(gen_tokens)
    if k < 0 or k > n_gen:
        raise ValueError(f"cut {k} outside 0..{n_gen}")
    if k < n_gen or close == "forced":
        return (*on_tokens, *gen_tokens[:k], *frame.close_tokens)
    if close == "eos":
        return (*on_tokens, *gen_tokens[:-1], *frame.close_tokens)
    if frame.tag_single and n_gen and gen_tokens[-1] == frame.tag_tokens[0]:
        return (*on_tokens, *gen_tokens, *frame.post_tokens)
    return None


def context_need(prompt_tokens: int, budget: int) -> int:
    """The slot context a deliberate question needs: the prompt, the budget, and 16 for the close suffix, the
    walk's depth and one."""
    return prompt_tokens + budget + 16


# ---- the think profile ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ThinkProfile:
    source: str
    sha256: str
    data: dict
    model_key: str
    model_name: str | None
    budgets: dict[int, dict]            # budget -> its entry (with "T")
    sampling: Sampling
    capable: bool
    state: str | None                   # server_state.state
    latency_model: str | None
    drift_flagged: bool
    host_fingerprint: str | None
    build_info: str | None
    chat_template_sha256: str | None
    position_bias_sha256: str | None
    provenance: dict
    pick: int | None
    host_chip: str | None = None        # environment.host.chip: the warnings name the machine, not its hash
    # think deadlines: the deadline points by D in ms, and the profile-level `latency` block (the estimators' seeds)
    deadlines: dict[int, dict] = dataclasses.field(default_factory=dict)
    latency: dict = dataclasses.field(default_factory=dict)

    def deadline_point(self, deadline_ms: int) -> dict | None:
        """Rule 1: the profile's deadline point for exactly this D, or None."""
        return self.deadlines.get(deadline_ms)

    def temperature_for(self, budget: int) -> tuple[float, bool]:
        """(the T to serve at `budget`, whether it was calibrated). A budget the profile lacks takes
        the T of the largest calibrated budget below it, or the smallest positive one when it is below all of them."""
        if budget in self.budgets:
            return float(self.budgets[budget]["T"]), True
        lower = [b for b in self.budgets if 0 < b <= budget]
        positive = [b for b in self.budgets if b > 0]
        if lower:
            return float(self.budgets[max(lower)]["T"]), False
        if positive:
            return float(self.budgets[min(positive)]["T"]), False
        return 1.0, False

    @property
    def t0(self) -> float | None:
        return float(self.budgets[0]["T"]) if 0 in self.budgets else None

    def info(self, host_fingerprint: str | None = None) -> dict:
        """The `/callo/stats.think.profile` block; `host_match` compares the profile's host fingerprint with
        the served host's (None when either is unknown)."""
        match = None if host_fingerprint is None or self.host_fingerprint is None else \
            self.host_fingerprint == host_fingerprint
        return {"path": self.source, "sha256": self.sha256, "budgets": sorted(self.budgets),
                "model_key": self.model_key, "state": self.state, "latency_model": self.latency_model,
                "host_match": match}


def json_safe(obj: Any) -> Any:
    """`obj` with every non-finite float stringified ("nan", "inf", "-inf"), so `json.dumps(..., allow_nan=False)`
    emits strict JSON: the raw GGUF dict a profile records (`comparable.sampling.gguf`) can hold nan / inf, which
    are recorded but never pinned. The profile writer dumps through this; the loader accepts both forms."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    if isinstance(obj, Mapping):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def finite_number(v: Any) -> bool:
    """A finite JSON number: an int or a float, never a bool; an integer too large for a float (a 400-digit one) is
    not, never an OverflowError. The one predicate compare, menu and the benches share."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def finite_positive(v: Any) -> bool:
    """`finite_number` and above 0."""
    return finite_number(v) and v > 0


def load_think_profile(src: str | Path | Mapping) -> ThinkProfile:
    """(InvalidInput, exit 1, before anything starts): readable JSON, format 1, a known metrics version,
    a well-formed model_key, integer budget keys 0..32768 with "0", every T finite and positive, the pinned sampler
    complete and well-formed (`check_pinned`), a known sampling policy. A dict source is hashed as the strict JSON of
    its `json_safe` form (so a recorded nan / inf neither crashes nor changes the hash a written file would get)."""
    if isinstance(src, Mapping):
        data: Any = dict(src)
        source = "<dict>"
        try:                    # as profile.load_profile: mixed-type keys, a cycle or deep nesting is not JSON
            raw = json.dumps(json_safe(data), sort_keys=True, separators=(",", ":"), allow_nan=False,
                             default=str).encode()
        except (TypeError, ValueError, RecursionError) as e:
            raise InvalidInput(f"--think-profile {source}: not JSON-serialisable ({type(e).__name__}: {e})") from None
    else:
        source = str(src)
        try:
            raw = Path(src).read_bytes()
        except OSError as e:
            raise InvalidInput(f"--think-profile {source}: cannot read it ({e})") from None
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise InvalidInput(f"--think-profile {source}: not valid JSON ({e})") from None

    def bad(why: str) -> InvalidInput:
        return InvalidInput(f"--think-profile {source}: {why}")

    if not isinstance(data, dict):
        raise bad("not a JSON object")
    if data.get("callo_think_profile") != PROFILE_FORMAT or type(data.get("callo_think_profile")) is not int:
        raise bad(f"callo_think_profile must be {PROFILE_FORMAT}, got {data.get('callo_think_profile')!r}")
    if data.get("metrics") != METRICS_VERSION:
        raise bad(f"metrics must be {METRICS_VERSION!r}, got {data.get('metrics')!r}")
    key = data.get("model_key")
    if not isinstance(key, str) or not KEY_RE.fullmatch(key):
        raise bad(f"model_key must be 'sha256:' and 64 hex digits, got {key!r}")
    entries = data.get("budgets")
    if not isinstance(entries, dict) or "0" not in entries:
        raise bad("budgets must be an object with a \"0\" entry")
    budgets: dict[int, dict] = {}
    for k, entry in entries.items():
        if not (isinstance(k, str) and re.fullmatch(r"0|[1-9][0-9]*", k) and int(k) <= MAX_BUDGET):
            raise bad(f"budget key {k!r} must be an integer from 0 to {MAX_BUDGET}")
        if not isinstance(entry, dict) or not finite_positive(entry.get("T")):
            raise bad(f"budget {k} must be an object whose T is finite and positive, got {entry!r}")
        budgets[int(k)] = entry
    comp = data.get("comparable") if isinstance(data.get("comparable"), dict) else {}
    samp = comp.get("sampling") if isinstance(comp.get("sampling"), dict) else {}
    pinned = samp.get("pinned")
    why = check_pinned(pinned)
    if why is not None:
        raise bad(f"comparable.sampling.pinned {why}")
    policy = samp.get("policy")
    if not isinstance(policy, str) or not _is_policy(policy):
        raise bad(f"comparable.sampling.policy must be model, fixed or a spec, got {policy!r}")
    gguf_error = samp.get("gguf_error")
    if gguf_error is not None and not isinstance(gguf_error, str):
        raise bad(f"comparable.sampling.gguf_error must be a string, got {gguf_error!r}")
    origin = samp.get("origin") if isinstance(samp.get("origin"), dict) else {}
    gguf = samp.get("gguf") if isinstance(samp.get("gguf"), dict) else {}
    sampling = Sampling(policy, dict(pinned), dict(origin), dict(gguf), gguf_error)
    thinking = data.get("thinking") if isinstance(data.get("thinking"), dict) else {}
    if not isinstance(thinking.get("capable", True), bool):
        raise bad(f"thinking.capable must be true or false, got {thinking.get('capable')!r}")
    env = data.get("environment") if isinstance(data.get("environment"), dict) else {}
    host = env.get("host") if isinstance(env.get("host"), dict) else {}
    server = env.get("server") if isinstance(env.get("server"), dict) else {}
    state = (data.get("server_state") or {}).get("state") if isinstance(data.get("server_state"), dict) else None
    audit = data.get("audit") if isinstance(data.get("audit"), dict) else {}
    drift = data.get("drift") if isinstance(data.get("drift"), dict) else {}
    prov = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
    pick = (data.get("pick") or {}).get("budget") if isinstance(data.get("pick"), dict) else None
    deadlines, why = check_deadline_points(data.get("deadlines"))     # optional, validated when present
    if deadlines is None:
        raise bad(why or "deadlines is malformed")
    latency, why = check_latency(data.get("latency"))
    if latency is None:
        raise bad(why or "latency is malformed")
    return ThinkProfile(source, hashlib.sha256(raw).hexdigest(), data, key,
                        data.get("model_name") if isinstance(data.get("model_name"), str) else None,
                        dict(sorted(budgets.items())), sampling, bool(thinking.get("capable", True)),
                        state if isinstance(state, str) else None,
                        audit.get("latency_model") if isinstance(audit.get("latency_model"), str) else None,
                        bool(drift.get("flagged")), host.get("fingerprint") if isinstance(host.get("fingerprint"), str)
                        else None, env.get("build_info") if isinstance(env.get("build_info"), str) else None,
                        prov.get("chat_template_sha256") if isinstance(prov.get("chat_template_sha256"), str) else None,
                        server.get("position_bias_sha256") if isinstance(server.get("position_bias_sha256"), str)
                        else None, prov, pick if isinstance(pick, int) and not isinstance(pick, bool) else None,
                        host.get("chip") if isinstance(host.get("chip"), str) else None, deadlines, latency)


def _is_policy(text: str) -> bool:
    """A recorded sampling policy: model, fixed, or a spec `parse_sampling` takes (its label, e.g. temp=0.6,top_k=20)."""
    if text in ("model", "fixed"):
        return True
    try:
        return parse_sampling(text).kind == "spec"
    except InvalidInput:
        return False


def check_think_identity(tp: ThinkProfile, *, model_key: str, native: bool, capable: bool) -> None:
    """Phase 2, the refusing half (CalloError, exit 1): a native upstream, another model, or a thinking profile on
    a model that is not capable."""
    if native:
        raise CalloError(f"--think-profile {tp.source}: the upstream is a native decision model; callo's engine is off, "
                         f"so there is no deliberate readout to serve")
    if tp.model_key != model_key:
        raise CalloError(f"--think-profile {tp.source}: this profile was measured on {tp.model_name or 'another model'} "
                         f"({tp.model_key}), not on the served model ({model_key}). Measure this model with "
                         f"`callo calibrate --think-budgets`")
    if tp.capable and not capable:
        raise CalloError(f"--think-profile {tp.source}: the profile has thinking budgets, but the served model's "
                         f"template has no thinking mode callo can read; recalibrate this model")


def check_think_profile(tp: ThinkProfile, *, provenance: Mapping | None, host_fingerprint: str | None,
                        build_info: str | None, position_bias_sha256: str | None, temperature_from_profile: bool,
                        default_budget: int, host_chip: str | None = None) -> list[str]:
    """Phase 2, the warnings (each line starts "WARNING:"): a provenance field that differs (profile.py's
    PROVENANCE_FIELDS, as the bias guard reads them; build_info once, in the host/build line), template drift, another
    host or build (named "<chip>, <build>"), an unknown host (a None fingerprint on either side), a busy / unverified /
    drifted profile, a budget-0-only profile, an uncalibrated default budget (with the rule it is served under), and
    a served position-bias profile that is not the one T_0 was fitted on."""
    # profile imports engine, which imports this module: a module-level import would be circular
    from .profile import PROVENANCE_FIELDS, wide_contract_warning  # noqa: PLC0415

    lines = []
    prov = provenance or {}
    for f in PROVENANCE_FIELDS:
        if f == "build_info":                   # the host/build line below names it
            continue
        then, now = tp.provenance.get(f), prov.get(f)
        if then != now:
            if f == "chat_template_sha256":
                lines.append("WARNING: --think-profile: the chat template differs from the profile's, so the thinking "
                             "prompt may differ: recalibrate")
            else:
                lines.append(f"WARNING: --think-profile: {f} differs from the profile's (profile {then!r}, now "
                             f"{now!r}); the temperatures may not transfer")
    wide = wide_contract_warning(tp.provenance.get("wide_contract"), prov.get("wide_contract"), "--think-profile")
    if wide is not None:                        # only when both carry it (a legacy profile is silent)
        lines.append(wide)
    unknown_host = tp.host_fingerprint is None or host_fingerprint is None     # None is "unknown", never a match
    if unknown_host:
        lines.append("WARNING: --think-profile: unknown host: cannot tell whether the profile was measured on this "
                     "machine")
    other_host = not unknown_host and tp.host_fingerprint != host_fingerprint
    other_build = tp.build_info != build_info
    if other_host or other_build:
        what = "another host and build" if other_host and other_build else "another host" if other_host else \
            "another build"
        lines.append(f"WARNING: --think-profile: its latencies were measured on {what} (profile: "
                     f"{tp.host_chip or 'unknown chip'}, {tp.build_info or 'unknown build'}; now: "
                     f"{host_chip or 'unknown chip'}, {build_info or 'unknown build'})")
    if tp.state == "busy":
        lines.append("WARNING: --think-profile: the profile was measured on a busy server (calibrate --allow-busy)")
    if tp.latency_model == "unverified":
        lines.append("WARNING: --think-profile: the profile's latency model is unverified (its cut audit failed)")
    if tp.drift_flagged:
        lines.append("WARNING: --think-profile: the profile's run drifted (decode speed or thermal data): its latencies "
                     "are less comparable")
    if not tp.capable:
        lines.append("WARNING: --think-profile: budget 0 only: the profile supplies T_0")
    if default_budget > 0 and default_budget not in tp.budgets:
        positive = sorted(b for b in tp.budgets if b > 0)
        lower = [b for b in positive if b <= default_budget]
        rule = (f"budget {max(lower)}'s temperature (the nearest lower)" if lower else
                f"budget {positive[0]}'s temperature (the smallest calibrated; none is lower)" if positive else
                "T = 1 (no positive budget is calibrated)")
        lines.append(f"WARNING: --think-budget {default_budget} is not in the profile (calibrated budgets: "
                     f"{', '.join(map(str, positive)) or 'none'}); it is served at {rule} and marked calibrated: false")
    if temperature_from_profile and tp.position_bias_sha256 != position_bias_sha256:
        lines.append(f"WARNING: T_0 was fitted on readouts debiased by {tp.position_bias_sha256 or 'no profile'}, and "
                     f"this server serves {position_bias_sha256 or 'no profile'}; recalibrate, or pass --temperature")
    return lines


# ---- calibrate's refusals against a serve that thinks ----------------------------------------------

PROFILE_T_FIX = ("restart `callo serve` without --think-profile (its T_0 sets the budget-0 temperature), or with "
                 "--temperature 1.0")


def think_default_refusal(stats: Mapping) -> str | None:
    """Why the temperature fit and `calibrate --bias` refuse this server's `/callo/stats`, or None: a server that
    thinks by default (`think.default_budget` > 0, `think.deadline.default_ms` > 0, or `think.converge.default_z` > 0)
    would give deliberate readouts on /v1/systemone. A `think` block, `think.default_budget` or `think.converge` callo
    cannot read is a refusal too (exit 2, never a traceback); a server with no `think` block (older than think
    budgets), or with no `think.converge` (older than early exit), thinks by no default on what it lacks."""
    think = stats.get("think")
    if think is None:
        return None
    budget = think.get("default_budget", 0) if isinstance(think, Mapping) else think
    if not isinstance(think, Mapping) or isinstance(budget, bool) or not isinstance(budget, int):
        return (f"cannot tell whether the server thinks by default: /callo/stats.think.default_budget is {budget!r}, "
                f"not an integer")
    if budget > 0:
        return (f"the server thinks by default (--think-budget {budget}), so /v1/systemone would give deliberate "
                f"readouts; restart without --think-budget. Temperatures for think budgets come from `callo calibrate "
                f"--think-budgets`")
    deadline = (think.get("deadline") or {}).get("default_ms", 0) if isinstance(think.get("deadline"), Mapping) else 0
    if isinstance(deadline, (int, float)) and not isinstance(deadline, bool) and deadline > 0:
        return (f"the server thinks by default under a deadline (--think-deadline {deadline}), so /v1/systemone would "
                f"give deliberate readouts; restart without --think-deadline")
    if "converge" not in think:
        return None
    cv = think["converge"]
    if not isinstance(cv, Mapping):
        return (f"cannot tell whether the server thinks by default with early exit: /callo/stats.think.converge is "
                f"{cv!r}, not an object")
    z = cv.get("default_z", 0)
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z):
        return (f"cannot tell whether the server thinks by default with early exit: "
                f"/callo/stats.think.converge.default_z is {z!r}, not a finite number")
    if z > 0:
        return (f"the server thinks by default with early exit (--think-converge {z:g}), so /v1/systemone would give "
                f"deliberate readouts; restart without --think-converge")
    return None


# ---- the latency model and the metrics ----------------------------------------------------------

RESAMPLES = 2000
LATENCY_AXES = ("mean", "p50", "p95")      # the pick statistic: frontier axis, "fastest", what --price charges
# pick preferences: the five latency statistics every point carries, their profile fields, and the quantiles
LIMIT_STATS = ("mean", "p50", "p95", "p99", "max")
STAT_FIELDS = {"mean": "lat_mean_ms", "p50": "lat_p50_ms", "p95": "lat_p95_ms", "p99": "lat_p99_ms",
               "max": "lat_max_ms"}
QUANTILES = {"p50": 0.5, "p95": 0.95, "p99": 0.99}


def budget_latency(frame_ms: float, gen_wall_ms: float, predicted_ms: float | None, n_gen: int, k: int,
                   read_ms_k: float, p_ms_k: float | None, p_ms_ngen: float | None) -> float:
    """`lat_B` in ms, from the one generation at Bmax. Exact when the cut is the whole trace (k >= n_gen):
    frame + gen_wall + read_ms. Otherwise frame + fixed + decode(k) + read_served(k), where fixed = gen_wall -
    predicted (prefill, HTTP, scheduling), decode(k) = predicted * k / n_gen (linear in tokens) and read_served(k) =
    read_ms(k) - max(0, p_ms(k) - p_ms(n_gen)) drops the calibration-only re-prefill, never below 0. A server whose
    reported `predicted_ms` exceeds the measured wall (the test fake plants timings) counts the larger of the two in
    both branches, so the model is continuous at k = n_gen."""
    predicted = float(predicted_ms or 0.0)
    if n_gen <= 0 or k >= n_gen:
        return frame_ms + max(gen_wall_ms, predicted) + read_ms_k
    fixed = max(0.0, gen_wall_ms - predicted)
    decode = predicted * k / n_gen
    extra = max(0.0, float(p_ms_k or 0.0) - float(p_ms_ngen or 0.0))
    return frame_ms + fixed + decode + max(0.0, read_ms_k - extra)


def accuracy(rows: Sequence[Row]) -> float:
    """The argmax equals the label, a tie going to the first index."""
    if not rows:
        raise InvalidInput("no rows")
    return sum(max(range(len(p)), key=p.__getitem__) == y for p, y in rows) / len(rows)


def nll_at(rows: Sequence[Row], temperature: float) -> float:
    return nll(rows, temperature)


def ece_at(rows: Sequence[Row], temperature: float) -> float:
    return ece(rows, temperature)


def fit_T(rows: Sequence[Row]) -> float:
    """`calibrate.fit_temperature`: the 181-point log grid on 0.25..16."""
    return fit_temperature(rows)


def crossfit(rows: Sequence[Row], clusters: Sequence[str], seed: int) -> tuple[float, float]:
    """NLL and ECE10 at T_B, 2-fold cross-fitted by cluster: the clusters are split into two seeded halves,
    T is fitted on one and scored on the other, and the two out-of-fold scores are averaged weighted by their halves'
    row counts, so the cross-fitted NLL is a per-row mean on the same scale as the in-sample NLL at T = 1 (ECE10
    likewise). With fewer than two clusters the in-sample numbers are returned."""
    if len(rows) != len(clusters):
        raise ValueError("rows and clusters differ in length")
    names = sorted(set(clusters))
    random.Random(f"{seed}:crossfit").shuffle(names)
    half = set(names[: len(names) // 2])
    a = [r for r, c in zip(rows, clusters, strict=True) if c in half]
    b = [r for r, c in zip(rows, clusters, strict=True) if c not in half]
    if not a or not b:
        t = fit_T(rows)
        return nll_at(rows, t), ece_at(rows, t)
    ta, tb = fit_T(a), fit_T(b)
    na, nb, n = len(a), len(b), len(a) + len(b)
    return (nb * nll_at(b, ta) + na * nll_at(a, tb)) / n, (nb * ece_at(b, ta) + na * ece_at(a, tb)) / n


def _bootstrap_weights(clusters: Sequence[str], strata: Sequence[str] | None, seed: int,
                       resamples: int) -> tuple[Any, Any]:
    """Each row's cluster index (clusters numbered in order of first appearance) and a (resamples, clusters) matrix of
    cluster multiplicities: each resample draws whole clusters with replacement WITHIN each stratum (source), so every
    resample keeps each source's cluster count; with no strata, or one, it is
    plain cluster resampling. The draw depends only on `seed` and the first-appearance order of the clusters and
    strata, so PAIRING across budgets holds only when the caller passes one fixed row set in one fixed order to every
    call (every budget's rows and the latency rows over the same questions, in the same order)."""
    ids: dict[str, int] = {}
    idx = np.array([ids.setdefault(c, len(ids)) for c in clusters], dtype=np.intp)
    if strata is None:
        groups = [list(range(len(ids)))]
    else:
        if len(strata) != len(clusters):
            raise ValueError("strata and clusters must be of one length")
        of: dict[str, str] = {}
        for c, s in zip(clusters, strata, strict=True):
            prev = of.setdefault(c, s)
            if prev != s:
                raise ValueError(f"cluster {c!r} spans sources {prev!r} and {s!r}")
        by: dict[str, list[int]] = {}                  # strata in first-appearance order, clusters ascending in each
        for c, i in ids.items():
            by.setdefault(of[c], []).append(i)
        groups = list(by.values())
    rng = np.random.default_rng(seed)
    w = np.zeros((resamples, len(ids)))
    for g in groups:
        w[:, g] = rng.multinomial(len(g), np.full(len(g), 1.0 / len(g)), size=resamples)
    return idx, w


def paired_bootstrap(a: Sequence[float], b: Sequence[float], clusters: Sequence[str], *, seed: int,
                     resamples: int = RESAMPLES,
                     strata: Sequence[str] | None = None) -> tuple[float, float, float]:
    """mean(b - a) with a 95 % percentile interval from a cluster bootstrap (whole clusters resampled with
    replacement, within each of `strata`, the rows' sources, when given), paired by index.
    Deterministic in `seed`; see `_bootstrap_weights` for the fixed-row-set contract pairing needs."""
    if not (len(a) == len(b) == len(clusters)) or not a:
        raise ValueError("a, b and clusters must be non-empty and of one length")
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return bootstrap_mean(d, clusters, seed=seed, resamples=resamples, strata=strata)


def bootstrap_mean(values: Sequence[float], clusters: Sequence[str], *, seed: int, resamples: int = RESAMPLES,
                   strata: Sequence[str] | None = None) -> tuple[float, float, float]:
    """The mean of `values` with a 95 % percentile interval from the cluster bootstrap: each resample's
    estimate is the ratio Σ w_c·sum_c / Σ w_c·count_c over the cluster multiplicities w of `_bootstrap_weights`, so
    the same seed, clusters and strata draw the same resamples as every paired Δacc (`paired_bootstrap` is this
    function on the per-question differences)."""
    if len(values) == 0 or len(values) != len(clusters):
        raise ValueError("values and clusters must be non-empty and of one length")
    d = np.asarray(values, dtype=float)
    idx, w = _bootstrap_weights(clusters, strata, seed, resamples)
    c = w.shape[1]
    sums = np.zeros(c)
    np.add.at(sums, idx, d)
    counts = np.bincount(idx, minlength=c).astype(float)
    est = (w @ sums) / (w @ counts)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def nearest_rank(n: int, q: float) -> int:
    """The nearest rank of the q-quantile of n values, max(1, ceil(q·n)), in integer arithmetic on q to 1e-6 (as
    `/callo/stats`): 188, 358 and 373 for p50, p95 and p99 of 376."""
    return max(1, -(-round(q * 10**6) * n // 10**6))


def tail_counts(n: int) -> dict[str, int]:
    """What a statistic "rests on": how many of n questions sit at or above each statistic's rank, {"p50", "p95",
    "p99", "max"} (189, 19, 4 and 1 at n = 376). Below 100 questions p99's rank is n, so p99 rests on one question,
    as max does."""
    return {**{k: n - nearest_rank(n, q) + 1 for k, q in QUANTILES.items()}, "max": 1}


def _walk(xs: Any, idx: Any, w: Any, q: float) -> Any:
    """The w-weighted nearest-rank q-quantile of `xs` on every resample: `idx` is each value's cluster and `w` the
    (resamples, clusters) multiplicities of `_bootstrap_weights`; each resample walks the sorted values' cumulative
    weight to rank ceil(q·total). Shared by `bootstrap_percentiles` and the joint Δscore."""
    order = np.argsort(xs, kind="stable")
    xs_sorted = xs[order]
    per_value = w[:, idx[order]]                                   # (resamples, n): each value's multiplicity
    cum = np.cumsum(per_value, axis=1)
    total = cum[:, -1]
    rank = np.ceil(round(q * 10**6) * total / 10**6)
    rank = np.maximum(rank, 1.0)
    pos = (cum < rank[:, None]).sum(axis=1)
    pos = np.minimum(pos, len(xs) - 1)
    return xs_sorted[pos]


def bootstrap_percentiles(values: Sequence[float], clusters: Sequence[str], q: float, *, seed: int,
                          resamples: int = RESAMPLES,
                          strata: Sequence[str] | None = None) -> tuple[float, float, float]:
    """The nearest-rank q-percentile of `values` with a 95 % cluster-bootstrap interval, from the same
    resampling scheme as `paired_bootstrap` (the same seed, clusters and strata draw the same cluster
    multiplicities)."""
    if not values or len(values) != len(clusters):
        raise ValueError("values and clusters must be non-empty and of one length")
    xs = np.asarray(values, dtype=float)
    idx, w = _bootstrap_weights(clusters, strata, seed, resamples)
    lo, hi = np.percentile(_walk(xs, idx, w, q), [2.5, 97.5])
    return float(np.sort(xs, kind="stable")[nearest_rank(len(xs), q) - 1]), float(lo), float(hi)


def latency_stats(values: Sequence[float], clusters: Sequence[str], *, seed: int,
                  strata: Sequence[str] | None = None) -> dict[str, Any]:
    """A point's five latency statistics as its profile fields, over its own questions in row order:
    `lat_mean_ms`, `lat_p50_ms`, `lat_p95_ms`, `lat_p99_ms` ([point, lo, hi], one seed, clusters and strata, so every
    interval comes from the same resamples) and `lat_max_ms` (a number, no interval)."""
    out: dict[str, Any] = {"lat_mean_ms": list(bootstrap_mean(values, clusters, seed=seed, strata=strata))}
    for stat, q in QUANTILES.items():
        out[STAT_FIELDS[stat]] = list(bootstrap_percentiles(values, clusters, q, seed=seed, strata=strata))
    out["lat_max_ms"] = float(max(values))
    return out


def row_latencies(profile: Mapping[str, Any], block: str, key: str) -> tuple[list[float], list[str], list[str] | None]:
    """The (lat_ms, cluster, stratum) of every profile row holding `rows[][block][key]`, in row order: what calibrate
    computed that point's statistics over. `block` is "b" (budgets) or "d" (deadlines), read at
    `row_fields.index("lat_ms")`, or "e" / "g" (early-exit rules, plain or gated, a fixed `[correct, lat_ms]` pair:
    index 1). A row's cluster
    defaults to its question id; the strata are the rows' sources when `comparable.data.strata` is "source", every
    one is a string and no cluster spans two (a hand-edited profile), else None."""
    fields = list(profile.get("row_fields") or ["correct"])
    at = 1 if block in ("e", "g") else fields.index("lat_ms")
    held = [r for r in profile.get("rows") or [] if key in (r.get(block) or {})]
    lats = [float(r[block][key][at]) for r in held]
    clusters = [str(r["cluster"]) if r.get("cluster") is not None else str(r["q"]) for r in held]
    data = (profile.get("comparable") or {}).get("data") or {}
    sources = [r.get("source") for r in held]
    of: dict[str, Any] = {}
    whole = all(of.setdefault(c, x) == x for c, x in zip(clusters, sources, strict=True))
    if data.get("strata") != "source" or not all(isinstance(x, str) for x in sources) or not whole:
        return lats, clusters, None             # calibrate's `_sources`: no stratification can keep such a cluster whole
    return lats, clusters, [str(x) for x in sources]


def point_stats(profile: Mapping[str, Any], entry: Mapping[str, Any], block: str, key: str) -> dict[str, Any]:
    """The five latency fields of one stored point (`budgets[B]`, `deadlines[D]`, `ee[z]`, `ee_gated[z]`): each field
    stored in `entry` wins; an absent one is recomputed from the profile's rows (`row_latencies`) with the profile's
    seed, as calibrate computed it, so a recomputed value equals the one calibrate would have stored, bit for bit."""
    out = {f: entry.get(f) for f in STAT_FIELDS.values()}
    if all(v is not None for v in out.values()):
        return out
    lats, clusters, strata = row_latencies(profile, block, key)
    seed = int(((profile.get("comparable") or {}).get("seed")) or 0)
    fresh = latency_stats(lats, clusters, seed=seed, strata=strata)
    return {f: fresh[f] if v is None else v for f, v in out.items()}


# ---- points, dominance, the frontier and the pick -------------------------------------------------------

POINT_KINDS = ("budget", "deadline", "ee", "eeg")
EE_KINDS = ("ee", "eeg")        # early exit, plain and gated (`eeg`, `<model>@ee<z>+g`): listed, never picked


def ee_key(z: float) -> str:
    """An early-exit rule's key in a profile's `ee` block and `rows[].e`: z as its shortest decimal, "0.95"."""
    return repr(float(z))


@dataclass(frozen=True, kw_only=True)
class Point:
    """One point of a frontier, always built with keyword arguments: a think budget (`kind` "budget",
    `value` B), a deadline (`kind` "deadline", `value` D in ms, `cap` the cap D was simulated under) or an early-exit
    rule (`kind` "ee", `value` z) of one model, or that rule behind early exit's gate (`kind` "eeg", `value` z). The five latency statistics are in ms over the point's own `n`
    questions: `lat_mean`, `lat_p50`, `lat_p95` and `lat_p99` (point, lo, hi), `lat_max` a number. `budget` is
    the smallest merged budget of a budget point and None for the other kinds; `budgets` holds every budget merged
    into it (`merge_identical`); `dominated_by` and `approx` are set by `frontier` / `mark_approx` on copies."""
    name: str
    acc: float
    n: int
    lat_mean: tuple[float, float, float]
    lat_p50: tuple[float, float, float]
    lat_p95: tuple[float, float, float]
    lat_p99: tuple[float, float, float]
    lat_max: float
    kind: str = "budget"
    value: float | None = None
    budget: int | None = None
    cap: int | None = None
    think_mean: float | None = None
    forced: float | None = None
    budgets: tuple[int, ...] = ()
    sampler: str | None = None
    gib: float | None = None
    capable: bool = True
    dominated_by: str | None = None
    approx: bool = False

    def __post_init__(self) -> None:
        if self.kind not in POINT_KINDS:
            raise ValueError(f"a point's kind is one of {POINT_KINDS}, got {self.kind!r}")
        if (self.kind == "budget") != (self.budget is not None):
            raise ValueError(f"a budget point has a budget and no other kind does: {self.kind} with {self.budget!r}")
        if self.kind == "budget":
            if self.value is not None and self.value != self.budget:
                raise ValueError(f"a budget point's value is its budget: {self.value!r} against {self.budget!r}")
            object.__setattr__(self, "value", self.budget)
        if self.value is None:
            raise ValueError(f"a {self.kind} point needs its value")

    def replace(self, **changes: Any) -> Point:
        """A copy with `changes`. A budget point's `value` is its budget, so changing `budget` re-derives it;
        `dataclasses.replace(p, budget=X)` carries the old value along and raises rather than leave it stale."""
        if self.kind == "budget" and "budget" in changes:
            changes.setdefault("value", changes["budget"])
        return dataclasses.replace(self, **changes)

    @property
    def label(self) -> str:
        if self.kind == "deadline":
            return f"{self.name}@{int(self.value or 0)}ms"
        if self.kind in EE_KINDS:
            z = ee_key(float(self.value or 0.0)).removeprefix("0")
            return f"{self.name}@ee{z}" + ("+g" if self.kind == "eeg" else "")
        bs = self.budgets or (self.budget,)
        return f"{self.name}@" + "≡".join(str(b) for b in bs)

    @property
    def tail_n(self) -> dict[str, int]:
        """How many of the point's questions each tail statistic rests on (`tail_counts`)."""
        return tail_counts(self.n)

    def latency(self, axis: str = "mean") -> tuple[float, float, float]:
        """The (point, lo, hi) of a pick statistic: mean, p50 or p95 (p99 and max are limits only)."""
        if axis not in LATENCY_AXES:
            raise ValueError(f"latency axis must be one of {LATENCY_AXES}, got {axis!r}")
        return {"mean": self.lat_mean, "p50": self.lat_p50, "p95": self.lat_p95}[axis]

    def stat(self, name: str) -> float:
        """The point estimate of any of the five statistics, in ms (what a --limit filters on)."""
        if name not in LIMIT_STATS:
            raise ValueError(f"a latency statistic is one of {LIMIT_STATS}, got {name!r}")
        return self.lat_max if name == "max" else getattr(self, f"lat_{name}")[0]

    def stats(self) -> dict[str, Any]:
        """The five statistics as a profile stores them (`lat_mean_ms` ... `lat_max_ms`)."""
        return {"lat_mean_ms": list(self.lat_mean), "lat_p50_ms": list(self.lat_p50), "lat_p95_ms": list(self.lat_p95),
                "lat_p99_ms": list(self.lat_p99), "lat_max_ms": self.lat_max}


def point_latencies(stats: Mapping[str, Any]) -> dict[str, Any]:
    """`Point`'s five latency keyword arguments from a profile entry's (or `latency_stats`') five fields:
    `Point(name=..., acc=..., n=..., **point_latencies(entry), ...)`."""
    return {"lat_mean": tuple(stats["lat_mean_ms"]), "lat_p50": tuple(stats["lat_p50_ms"]),
            "lat_p95": tuple(stats["lat_p95_ms"]), "lat_p99": tuple(stats["lat_p99_ms"]),
            "lat_max": float(stats["lat_max_ms"])}


def merge_identical(points: Sequence[Point], same: Callable[[Point, Point], bool]) -> list[Point]:
    """Budgets of ONE model whose rows are identical (every question closed before both, so `same` is true) merge
    into one point labelled `B≡B'`, keeping the smallest budget's latency and `.budget`. Points of different
    models never merge, nor do budgets that are not consecutive, and only budget points merge: a deadline or
    early-exit point is passed through as it is, even when its rows equal a budget's. `same(prev, p)`
    receives the ALREADY-MERGED point first: its `.budget` is the run's first budget and `.budgets[-1]` its last, so a
    predicate should compare `p` with `prev.budgets[-1]`."""
    out: list[Point] = []
    for p in sorted((p for p in points if p.kind == "budget"), key=lambda x: (x.name, x.budget)):
        prev = out[-1] if out else None
        if prev is not None and prev.name == p.name and same(prev, p):
            out[-1] = dataclasses.replace(prev, budgets=(*(prev.budgets or (prev.budget,)), p.budget))
        else:
            out.append(dataclasses.replace(p, budgets=p.budgets or (p.budget,)))
    return out + [p for p in points if p.kind != "budget"]


def _overlap(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return a[1] <= b[2] and b[1] <= a[2]


def frontier(points: Sequence[Point], *, latency: str = "mean") -> list[Point]:
    """Every point, sorted by latency, with `dominated_by` set to the label of a dominating point or None.
    A is dominated by B when acc_B >= acc_A and lat_B <= lat_A with at least one strict, EXCEPT when acc_B == acc_A
    and the two latency intervals overlap (a difference on latency alone, within noise: both stay)."""
    out = []
    for a in points:
        dominators = []
        for b in points:
            if b is a or b.label == a.label:   # compare disambiguates duplicate names (`alpha#2`)
                continue
            la, lb = a.latency(latency), b.latency(latency)
            if b.acc >= a.acc and lb[0] <= la[0] and (b.acc > a.acc or lb[0] < la[0]):
                if b.acc == a.acc and _overlap(la, lb):
                    continue
                dominators.append(b)
        best = min(dominators, key=lambda b: (-b.acc, b.latency(latency)[0], b.label), default=None)
        out.append(dataclasses.replace(a, dominated_by=best.label if best is not None else None))
    return sorted(out, key=lambda p: (p.latency(latency)[0], -p.acc, p.label))


def on_frontier(points: Sequence[Point]) -> list[Point]:
    return [p for p in points if p.dominated_by is None]


DaccCI = Callable[[Point, Point], tuple[float, float, float]]   # (point b - point a, lo, hi) of the paired Δacc


def mark_approx(front: Sequence[Point], dacc_ci: DaccCI) -> list[Point]:
    """The `≈` mark: a frontier point whose paired Δacc CI against its faster neighbour includes 0. `front`
    is `on_frontier(frontier(...))`, sorted by latency."""
    out = []
    for i, p in enumerate(front):
        approx = False
        if i > 0:
            _, lo, hi = dacc_ci(front[i - 1], p)
            approx = lo <= 0 <= hi
        out.append(dataclasses.replace(p, approx=approx))
    return out


# ---- the preference: limits, the price, and the one validator calibrate and compare share ----------

def _num_text(x: float) -> str:
    """The shortest decimal that reads back as x, without a trailing ".0" (2.0 -> "2", 0.35 -> "0.35")."""
    text = repr(float(x))
    return text[:-2] if text.endswith(".0") else text


def limit_text(le_ms: float) -> str:
    """A limit's X in canonical form: seconds when they read back as exactly `le_ms` ("6s", "0.3s"), else
    milliseconds ("2001ms"), so a printed limit, pasted back, is the same limit."""
    s = _num_text(le_ms / 1000)
    return f"{s}s" if float(s) * 1000 == le_ms else f"{_num_text(le_ms)}ms"


@dataclass(frozen=True)
class Limit:
    """One upper bound on a latency statistic's point estimate: `--limit STAT=X`, or `--max-latency S` on the
    pick statistic (`source` "--max-latency", so the alias stays visible)."""
    stat: str
    le_ms: float
    source: str = "--limit"

    @property
    def canonical(self) -> str:
        """The shell-safe form every printed line uses: `--limit p99=6s` (never `<`, which a shell takes as a
        redirection)."""
        return f"--limit {self.stat}={limit_text(self.le_ms)}"

    def as_dict(self) -> dict:
        return {"stat": self.stat, "le_ms": self.le_ms, "from": self.source}


LIMIT_USAGE = "--limit takes STAT=X, for example --limit p99=5s"
LIMIT_UPPER = "--limit takes STAT=X (an upper bound), for example --limit p99=5s"
_ALIAS = re.compile(r"\s*([A-Za-z0-9]+)\s*<=(.*)")


def parse_limit(text: str) -> Limit:
    """`--limit STAT=X` (canonical) or the quoted alias `'STAT<=X'`; X in seconds (`5`, `5s`) or milliseconds
    (`5000ms`). `<` and `>` are looked for before the split on `=`, so `'p99>=2'` gets the STAT=X message, not an
    unknown statistic `p99>`. InvalidInput (exit 2) with the messages."""
    if "<" in text or ">" in text:
        m = _ALIAS.fullmatch(text)
        if m is None or "<" in m.group(2) or ">" in m.group(2):
            raise InvalidInput(LIMIT_UPPER)
        stat, raw = m.group(1), m.group(2)
    else:
        stat, sep, raw = text.partition("=")
        if not sep:
            raise InvalidInput(LIMIT_USAGE)
        stat = stat.strip()
        if not stat:
            raise InvalidInput(LIMIT_USAGE)
    if stat not in LIMIT_STATS:
        raise InvalidInput(f"--limit: unknown statistic {stat} ({', '.join(LIMIT_STATS)})")
    raw = raw.strip()
    unit, num = ("ms", raw[:-2]) if raw.endswith("ms") else ("s", raw[:-1]) if raw.endswith("s") else ("s", raw)
    try:
        x = float(num)
    except ValueError:
        raise InvalidInput(f"--limit {stat}: X is seconds (5, 5s) or milliseconds (5000ms)") from None
    le_ms = x * 1000 if unit == "s" else x          # converted first: 1e306s overflows to inf here
    if not math.isfinite(le_ms) or le_ms <= 0:
        raise InvalidInput(f"--limit {stat}: X must be a positive finite number")
    return Limit(stat, le_ms)


LATENCY_USAGE = "--latency must be mean, p50 or p95 (p99 and max are limits only: --limit p99=S)"
PRICE_USAGE = "--price is in pp/s: 1, 1pp/s or 0.5pp/s"
PRICE_RANGE = "--price must be a finite number >= 0"
PRICE_MAX = 1e6
PRICE_TOO_BIG = "--price is at most 1000000 pp/s"


def parse_price(text: str) -> float:
    """`--price R`: pp/s, a bare number or with the `pp/s` suffix; R >= 0. InvalidInput (exit 2)."""
    text = text.strip()
    num = (text[:-4] if text.endswith("pp/s") else text).strip()
    try:
        r = float(num)
    except ValueError:
        raise InvalidInput(PRICE_USAGE) from None
    if not math.isfinite(r) or r < 0:
        raise InvalidInput(PRICE_RANGE)
    if r > PRICE_MAX:
        raise InvalidInput(PRICE_TOO_BIG)
    return r + 0.0                                   # -0 reads as 0.0


@dataclass(frozen=True)
class Preference:
    """The user's preference, applied after the fact by `pick`: the pick statistic (`latency`), the price R
    in pp/s, the limits (every `--limit`, and `--max-latency S` as a limit on the pick statistic), `--min-accuracy`
    and compare's `--max-memory` (`max_gib`). Construction is the validator calibrate and compare share: anything
    out of range is InvalidInput (exit 2) with the message."""
    latency: str = "mean"
    price: float = 0.0
    limits: tuple[Limit, ...] = ()
    min_accuracy: float | None = None
    max_gib: float | None = None
    max_latency_s: float | None = None

    def __post_init__(self) -> None:
        if self.latency not in LATENCY_AXES:
            raise InvalidInput(LATENCY_USAGE)
        if isinstance(self.price, bool) or not isinstance(self.price, (int, float)) or not math.isfinite(self.price) \
                or self.price < 0:
            raise InvalidInput(PRICE_RANGE)
        if self.price > PRICE_MAX:
            raise InvalidInput(PRICE_TOO_BIG)
        if self.price > 0 and self.min_accuracy is not None:                     # the test is R > 0
            raise InvalidInput("--min-accuracy and a positive --price are two objectives; give one")
        if self.max_latency_s is not None and not (self.max_latency_s > 0 and math.isfinite(1000 * self.max_latency_s)):
            raise InvalidInput(f"--max-latency is in seconds and must be positive, got {self.max_latency_s!r}")
        if self.min_accuracy is not None and not 0 <= self.min_accuracy <= 1:
            raise InvalidInput(f"--min-accuracy is a fraction from 0 to 1, got {self.min_accuracy!r}")
        if self.max_gib is not None and not (self.max_gib > 0 and math.isfinite(self.max_gib)):
            raise InvalidInput(f"--max-memory is in GiB and must be positive, got {self.max_gib!r}")
        limits = tuple(self.limits)
        if self.max_latency_s is not None and not any(lim.source == "--max-latency" for lim in limits):
            limits += (Limit(self.latency, 1000 * self.max_latency_s, "--max-latency"),)
        object.__setattr__(self, "limits", limits)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None, *, legacy: Mapping[str, Any] | None = None) -> Preference:
        """The preference a `pick` block or a report recorded (`as_dict`); an old block without one (`legacy`, the pick
        block itself) gives its `latency`, `max_latency_s` and `min_accuracy`."""
        try:
            if isinstance(d, Mapping):
                limits = []
                for x in d.get("limits") or ():
                    lim = Limit(str(x["stat"]), float(x["le_ms"]), str(x.get("from") or "--limit"))
                    if lim.stat not in LIMIT_STATS or not (math.isfinite(lim.le_ms) and lim.le_ms > 0):
                        raise ValueError(f"limit {lim.stat}={lim.le_ms!r}")
                    limits.append(lim)
                alias = next((lim.le_ms / 1000 for lim in limits if lim.source == "--max-latency"), None)
                return cls(str(d.get("latency") or "mean"), float(d.get("price_pp_per_s") or 0.0), tuple(limits),
                           d.get("min_accuracy"), d.get("max_memory_gib"), alias)
            old = legacy or {}
            return cls(str(old.get("latency") or "p50"), 0.0, (), old.get("min_accuracy"), None,
                       old.get("max_latency_s"))
        except InvalidInput:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise InvalidInput(f"a recorded preference is malformed ({type(e).__name__}: {e})") from None

    @property
    def is_default(self) -> bool:
        return self == Preference()

    def as_dict(self) -> dict:
        return {"latency": self.latency, "price_pp_per_s": self.price, "limits": [lim.as_dict() for lim in self.limits],
                "min_accuracy": self.min_accuracy, "max_memory_gib": self.max_gib}

    def text(self) -> str:
        """The non-default parts in canonical, pasteable form for the pick line: `--price 5pp/s on mean`
        (or `--latency p50` without a price), each limit, `--min-accuracy A`, `--max-memory G`; "" for the default."""
        parts = []
        if self.price > 0:
            parts.append(f"--price {_num_text(self.price)}pp/s on {self.latency}")
        elif self.latency != "mean":
            parts.append(f"--latency {self.latency}")
        parts += [lim.canonical for lim in self.limits]
        if self.min_accuracy is not None:
            parts.append(f"--min-accuracy {_num_text(self.min_accuracy)}")
        if self.max_gib is not None:
            parts.append(f"--max-memory {_num_text(self.max_gib)}")
        return ", ".join(parts)


def preference_from_args(args: Any) -> Preference:
    """The preference from calibrate's or compare's flags (`--latency`, `--limit`, `--price`, `--max-latency`,
    `--min-accuracy`, compare's `--max-memory`), validated: InvalidInput (exit 2)."""
    latency = getattr(args, "latency", None) or "mean"
    if latency not in LATENCY_AXES:                     # first, as the table lists it
        raise InvalidInput(LATENCY_USAGE)
    limits = tuple(parse_limit(t) for t in getattr(args, "limit", None) or ())
    price = parse_price(args.price) if getattr(args, "price", None) is not None else 0.0
    return Preference(latency, price, limits, getattr(args, "min_accuracy", None), getattr(args, "max_memory", None),
                      getattr(args, "max_latency", None))


# ---- noise: Δacc over one fixed row set, and the reported Δscore ---------------------------------------

def score_draws(ca: Sequence[float], la: Sequence[float], cb: Sequence[float], lb: Sequence[float],
                clusters: Sequence[str], *, stat: str, price: float, seed: int, strata: Sequence[str] | None = None,
                resamples: int = RESAMPLES) -> tuple[float, Any]:
    """The joint bootstrap of Δscore(b − a) in pp, s = 100·acc − R·lat/1000 on `stat`: (the point estimate, the
    resamples' Δ*). Every resample r takes the cluster multiplicities of `_bootstrap_weights` (one seed, clusters and
    strata, so the draws are every Δacc's): Δacc* is paired_bootstrap's ratio, Q* the w-weighted nearest-rank quantile
    (`_walk`) of each point's latencies, or for "mean" the same ratio form. The point estimate is
    100·mean(c_b − c_a) − R·(Q_b − Q_a)/1000 with the unweighted quantile (or mean) over the fixed rows."""
    if not (len(ca) == len(la) == len(cb) == len(lb) == len(clusters)) or not len(ca):
        raise ValueError("score_draws needs non-empty correct / latency rows and clusters of one length")
    if stat not in LATENCY_AXES:
        raise ValueError(f"score_draws: stat must be one of {LATENCY_AXES}, got {stat!r}")
    a_c, a_l, b_c, b_l = (np.asarray(x, dtype=float) for x in (ca, la, cb, lb))
    idx, w = _bootstrap_weights(clusters, strata, seed, resamples)
    c = w.shape[1]
    counts = w @ np.bincount(idx, minlength=c).astype(float)

    def ratio(x: Any) -> Any:
        sums = np.zeros(c)
        np.add.at(sums, idx, x)
        return (w @ sums) / counts

    if stat == "mean":
        dlat, point_lat = ratio(b_l) - ratio(a_l), float(b_l.mean() - a_l.mean())
    else:
        q = QUANTILES[stat]
        r = nearest_rank(len(a_l), q) - 1
        dlat = _walk(b_l, idx, w, q) - _walk(a_l, idx, w, q)
        point_lat = float(np.sort(b_l, kind="stable")[r] - np.sort(a_l, kind="stable")[r])
    draws = 100 * ratio(b_c - a_c) - price * dlat / 1000
    return 100 * float((b_c - a_c).mean()) - price * point_lat / 1000, draws


def joint_dscore(ca: Sequence[float], la: Sequence[float], cb: Sequence[float], lb: Sequence[float],
                 clusters: Sequence[str], *, stat: str, price: float, seed: int, strata: Sequence[str] | None = None,
                 resamples: int = RESAMPLES) -> tuple[float, float, float]:
    """`score_draws` as (point, lo, hi): the 2.5 / 97.5 percentiles of Δ*."""
    point, draws = score_draws(ca, la, cb, lb, clusters, stat=stat, price=price, seed=seed, strata=strata,
                               resamples=resamples)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


PointRows = Callable[[Point], tuple[Sequence[float], Sequence[float]]]


@dataclass(frozen=True)
class Paired:
    """Every paired interval over ONE fixed row set (the pairing contract): `rows(p)` is a point's
    per-question (correct as 1.0 / 0.0, lat_ms) over the fixed questions, in their one order, keyed on the point's
    kind and value (a merged budget point reads its smallest budget's rows); `clusters` and `strata` are those
    questions'. With no fixed questions every Δ is (0, 0, 0): "within noise", which the callers flag."""
    rows: PointRows
    clusters: tuple[str, ...]
    seed: int
    strata: tuple[str, ...] | None = None
    resamples: int = RESAMPLES

    def dacc(self, a: Point, b: Point) -> tuple[float, float, float]:
        """The paired Δacc of b − a (the noise test; the pick's, unchanged)."""
        if not self.clusters:
            return (0.0, 0.0, 0.0)
        return paired_bootstrap(self.rows(a)[0], self.rows(b)[0], self.clusters, seed=self.seed,
                                resamples=self.resamples, strata=self.strata)

    def dscore(self, a: Point, b: Point, *, stat: str, price: float) -> tuple[float, float, float]:
        """The reported Δscore of b − a in pp: at R = 0 literally 100 × `dacc`; on mean `paired_bootstrap` of the
        per-question scores s_q = 100·correct_q − R·lat_q/1000; on p50 / p95 the joint bootstrap."""
        if stat not in LATENCY_AXES:
            raise ValueError(f"latency axis must be one of {LATENCY_AXES}, got {stat!r}")
        if not self.clusters:
            return (0.0, 0.0, 0.0)
        if price == 0:
            p, lo, hi = self.dacc(a, b)
            return 100 * p, 100 * lo, 100 * hi
        (ca, la), (cb, lb) = self.rows(a), self.rows(b)
        if stat == "mean":
            sa = [100 * c - price * x / 1000 for c, x in zip(ca, la, strict=True)]
            sb = [100 * c - price * x / 1000 for c, x in zip(cb, lb, strict=True)]
            return paired_bootstrap(sa, sb, self.clusters, seed=self.seed, resamples=self.resamples,
                                    strata=self.strata)
        return joint_dscore(ca, la, cb, lb, self.clusters, stat=stat, price=price, seed=self.seed,
                            strata=self.strata, resamples=self.resamples)


# ---- the pick ------------------------------------------------------------

DEADLINES_HINT = ("to add a point that meets it: a new calibrate run with --deadlines {ms} (a run without --deadlines "
                  "cannot resume with them)")


@dataclass(frozen=True)
class Pick:
    """`budget` in `as_dict` is the LARGEST budget merged into the picked point: merged budgets read the same thing,
    and the largest truncates the fewest unseen questions. `best` is M, the best-scoring frontier
    point (at R = 0 the most accurate), and `most_accurate` the most accurate one."""
    point: Point | None
    rule: str                  # "fastest within noise of the best" | "... of the best score" | "min-accuracy" | "no pick"
    best: Point | None         # M: the best-scoring frontier point after the filters
    dacc_best: tuple[float, float, float] | None = None   # the picked point's paired Δacc vs M (point − M, lo, hi)
    front: list[Point] = dataclasses.field(default_factory=list)   # the filtered, marked frontier it chose from
    unsized: list[str] = dataclasses.field(default_factory=list)   # labels `max_gib` excluded for an unknown size
    most_accurate: Point | None = None
    score: dict | None = None                                      # R > 0: {"pick", "best"} in pp
    dscore_best: tuple[float, float, float] | None = None          # R > 0: the pick's Δscore vs M (pp)
    excluded: dict[str, str] = dataclasses.field(default_factory=dict)   # label -> the first filter that removed it
    reason: str | None = None                                      # why there is no pick
    preference: Preference = dataclasses.field(default_factory=Preference)
    # always None now: kept so the block keeps its keys. An OLD block (callo 0.3 development) can carry the early-exit
    # point a run admitting them would pick, or would aim at (its M); early exit is opt-in and nothing reads them
    would_pick: str | None = None
    would_best: str | None = None

    def as_dict(self) -> dict:
        """The block. A deadline pick has `budget` null, `budgets` [], `deadline_ms` D and `cap`. `z` is null: the
        pick never takes an early-exit point (a block written while it could has `kind` "ee", `z` and `cap`)."""
        p, pref = self.point, self.preference
        budgets = list(p.budgets or (p.budget,)) if p is not None and p.kind == "budget" else []
        deadline = p is not None and p.kind == "deadline"
        ee = p is not None and p.kind == "ee"
        return {"kind": p.kind if p is not None else None, "budget": max(budgets) if budgets else None,
                "budgets": budgets, "deadline_ms": int(p.value or 0) if deadline else None,
                "cap": p.cap if deadline or ee else None, "z": float(p.value) if ee and p.value is not None else None,
                "model": p.name if p else None, "label": p.label if p else None, "rule": self.rule,
                "best": self.best.label if self.best else None,
                "most_accurate": self.most_accurate.label if self.most_accurate else None,
                "dacc_best": list(self.dacc_best) if self.dacc_best is not None else None,
                "dscore_best": list(self.dscore_best) if self.dscore_best is not None else None,
                "score": dict(self.score) if self.score is not None else None,
                "preference": pref.as_dict(), "excluded": dict(self.excluded), "reason": self.reason,
                "unsized": list(self.unsized), "would_pick": self.would_pick, "would_best": self.would_best,
                "latency": pref.latency, "max_latency_s": pref.max_latency_s, "min_accuracy": pref.min_accuracy}


Filter = tuple[str, Callable[[Point], str | None], Callable[[Sequence[Point]], str]]

# the pick's exclusion reason for every early-exit point serve can run: early exit is opt-in. On held-out questions its
# rule held on question types like the calibration data and failed on reasoning problems of a kind it had not seen
EE_OPT_IN = "early exit: opt-in, never picked"


def _filters(pref: Preference, unservable: Mapping[str, str] | None = None) -> list[Filter]:
    """Step 1, in order: the serve check (`unservable`: an early-exit point `callo serve` cannot run as measured, with
    its reason), early exit (every other early-exit point, plain or gated: opt-in, never picked), each limit on its statistic's point
    estimate, then `--max-memory` (an unknown size is excluded). Each is (its name in a no-pick reason, a test giving
    why it excludes a point or None, the nearest miss among the points it excluded)."""
    out: list[Filter] = []
    if unservable:
        gone = dict(unservable)
        out.append(("the serve check", lambda p: gone.get(p.label), lambda _: ""))
    out.append(("early exit", lambda p: EE_OPT_IN if p.kind in EE_KINDS else None, lambda _: ""))
    for lim in pref.limits:
        def over(p: Point, lim: Limit = lim) -> str | None:
            v = p.stat(lim.stat)
            return None if v <= lim.le_ms else f"{lim.stat} {v / 1000:.2f} s > {_num_text(lim.le_ms / 1000)} s"

        def near(gone: Sequence[Point], lim: Limit = lim) -> str:
            p = min(gone, key=lambda x: (x.stat(lim.stat), x.label))
            tail = f", from {p.tail_n[lim.stat]} of {p.n}" if lim.stat in QUANTILES else ""
            return f"nearest {p.label} at {lim.stat} {p.stat(lim.stat) / 1000:.2f} s{tail}"
        out.append((lim.canonical, over, near))
    if pref.max_gib is not None:
        g = pref.max_gib

        def memory(p: Point) -> str | None:
            if p.gib is None:
                return "size unknown (--max-memory)"
            return None if p.gib <= g else f"{p.gib:.1f} GiB > {_num_text(g)} GiB"

        def smallest(gone: Sequence[Point]) -> str:
            sized = [p for p in gone if p.gib is not None]
            if not sized:
                return "every size unknown"
            p = min(sized, key=lambda x: (x.gib or 0.0, x.label))
            return f"nearest {p.label} at {p.gib:.1f} GiB"
        out.append((f"--max-memory {_num_text(g)}", memory, smallest))
    return out


def _frac(x: float) -> str:
    """.832, .9: an accuracy as the reasons print it."""
    text = _num_text(round(x, 3))
    return text[1:] if text.startswith("0.") else text


def _deadline_hint(points: Sequence[Point], by: Mapping[str, int], pref: Preference,
                   skipped: Mapping[str, Collection[int]] | None = None) -> str | None:
    """The `--deadlines` hint, when for some model (per model, in compare) the profile holds no deadline
    point, its budget 0 passes every step-1 filter, and some max or p99 limit fails for one of its positive budgets.
    The value is the tightest max / p99 limit, floored to whole ms (what `--deadlines` takes). The caller checks there
    is no pick. `skipped`: per model, the deadlines its run measured and made no point of (every
    question fell back, `deadline_skipped`); a model with one at or above the value is no case for the hint, since a
    deadline no longer than a skipped one would fall back as well."""
    tails = [lim for lim in pref.limits if lim.stat in ("max", "p99")]
    if not tails:
        return None
    ms = int(min(lim.le_ms for lim in tails))
    for name in dict.fromkeys(p.name for p in points):
        mine = [p for p in points if p.name == name]
        zero = next((p for p in mine if p.kind == "budget" and 0 in (p.budgets or (p.budget,))), None)
        if any(p.kind == "deadline" for p in mine) or zero is None or zero.label in by:
            continue
        if any(d >= ms for d in (skipped or {}).get(name, ())):
            continue
        if any(p.kind == "budget" and (p.budget or 0) > 0 and p.stat(lim.stat) > lim.le_ms for p in mine for lim in tails):
            return DEADLINES_HINT.format(ms=ms)
    return None


def _no_pick_reason(points: Sequence[Point], by: Mapping[str, int], filters: Sequence[Filter],
                    missed: str | None, pref: Preference, *,
                    skipped: Mapping[str, Collection[int]] | None = None) -> str:
    """Each filter that removed points, how many and the nearest miss, then `--min-accuracy`'s miss, then the
    `--deadlines` hint; every limit in its canonical, pasteable form. Early-exit points, which the pick never takes,
    are left out of the count and of the fragments."""
    if not points:
        return "no points"
    shown = [p for p in points if p.kind not in EE_KINDS]
    if not shown:
        return "no point to pick: early exit is opt-in, never picked"
    n = len(shown)
    frags = []
    for i, (name, _, near) in enumerate(filters):
        gone = [p for p in shown if by.get(p.label) == i]
        if not gone:
            continue
        miss = near(gone)
        if len(gone) == n:
            frags.append(f"no point meets {name}: all {n} excluded" + (f"; {miss}" if miss else ""))
        else:
            frags.append(f"{name} excluded {len(gone)} of {n}" + (f", {miss}" if miss else ""))
    if missed:
        frags.append(missed)
    hint = _deadline_hint(shown, by, pref, skipped)
    if hint:
        frags.append(hint)
    return "; ".join(frags) or "no point qualifies"


def _choose(points: Sequence[Point], paired: Paired, pref: Preference, *,
            skipped: Mapping[str, Collection[int]] | None = None,
            unservable: Mapping[str, str] | None = None) -> Pick:
    stat = pref.latency
    filters = _filters(pref, unservable)
    by: dict[str, int] = {}
    excluded: dict[str, str] = {}
    cands = []
    for p in points:
        for i, (_, test, _) in enumerate(filters):
            why = test(p)
            if why is not None:
                by[p.label], excluded[p.label] = i, why
                break
        else:
            cands.append(p)
    unsized = [p.label for p in points if pref.max_gib is not None and p.gib is None and p.kind not in EE_KINDS]
    front = mark_approx(on_frontier(frontier(cands, latency=stat)), paired.dacc)   # sorted by latency
    if not front:
        return Pick(None, "no pick", None, None, [], unsized, excluded=excluded, preference=pref,
                    reason=_no_pick_reason(points, by, filters, None, pref, skipped=skipped))

    def score(p: Point) -> float:                       # pp: s = 100·acc − R·lat/1000 on the pick statistic
        return 100 * p.acc - pref.price * p.latency(stat)[0] / 1000

    most = min(front, key=lambda p: (-p.acc, p.latency(stat)[0], p.label))
    best = min(front, key=lambda p: (-score(p), p.latency(stat)[0], p.label))       # M; at R = 0 the most accurate

    def vs_best(p: Point) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0) if p is best else paired.dacc(best, p)          # point − M

    if pref.min_accuracy is not None:
        ok = [p for p in front if p.acc >= pref.min_accuracy]
        if not ok:
            missed = f"--min-accuracy {_frac(pref.min_accuracy)}: {most.label} is {_frac(most.acc)}"
            return Pick(None, "no pick", best, None, front, unsized, most_accurate=most, excluded=excluded,
                        preference=pref, reason=_no_pick_reason(points, by, filters, missed, pref, skipped=skipped))
        chosen, rule, d = ok[0], "min-accuracy", vs_best(ok[0])
    else:
        for chosen in front:            # M is in `front`, so the loop always breaks, at M at the latest
            d = vs_best(chosen)
            if chosen is best or d[1] <= 0 <= d[2]:
                break
        rule = "fastest within noise of the best" + (" score" if pref.price > 0 else "")
    scored = dscore = None
    if pref.price > 0:
        scored = {"pick": score(chosen), "best": score(best)}
        dscore = (0.0, 0.0, 0.0) if chosen is best else paired.dscore(best, chosen, stat=stat, price=pref.price)
    return Pick(chosen, rule, best, d, front, unsized, most_accurate=most, score=scored, dscore_best=dscore,
                excluded=excluded, preference=pref)


def pick(points: Sequence[Point], *, paired: Paired, pref: Preference | None = None,
         skipped: Mapping[str, Collection[int]] | None = None, unservable: Mapping[str, str] | None = None) -> Pick:
    """With the preference, over EVERY annotated point:

    1. the filters (`_filters`: the serve check, early exit, each limit, `--max-memory`) apply to the points FIRST, so
       a point that fails one can never eliminate one that passes;
    2. the frontier, its `≈` marks and the pick are recomputed on what is left, on the pick statistic;
    3. with `--min-accuracy` the pick is the fastest frontier point at or above it; else M is the best-scoring
       frontier point (s = 100·acc − R·lat/1000; a tie goes to the faster, then the label; at R = 0 the most accurate)
       and the pick is the fastest frontier point whose paired Δacc against M has a CI including 0;
    4. no point qualifies: "no pick", with a reason.

    Early-exit points, plain and gated (`EE_KINDS`), are listed, never picked (`EE_OPT_IN`: early exit is opt-in,
    served only by an explicit `callo serve --think-converge Z`, with `--think-converge-gate` for a gated one), so the pick, M and the serve line are always a budget or deadline point.
    `unservable` ({label: why}, `converge.unservable`) names the early-exit points serve cannot run as measured; the
    serve check excludes them first, with that reason. Under `max_gib` a point of UNKNOWN size is excluded and named in
    `unsized` (an early-exit point never is); without it unknown sizes stay in. `skipped`: per model, the deadlines
    its profile lists in `deadline_skipped`, which the `--deadlines` hint never suggests again."""
    pref = pref if pref is not None else Preference()
    return _choose(points, paired, pref, skipped=skipped, unservable=unservable)
