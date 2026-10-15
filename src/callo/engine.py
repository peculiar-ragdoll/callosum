"""One question -> one forward pass (plus a rare shared-prefix walk, and above 26 options at most 2 split passes)
-> P(option).

The marginalisation is the vendored marginaliser (plus the float32 option-mass tolerance; the served mass is exact and the
served log-marginals are clamped to <= 0, a change of float noise only): equivalence sets built mechanically from the
vocabulary and SUMMED, and `divergence_walk` stepping over answer-neutral prefixes such as `**` or a newline.
What this module adds is the readout a classifier needs:

- **A bound for top-k misses.** An option with no surface token in the top-k dump reads `-inf`; its true mass
  is at most the k-th logprob, so it gets exactly that (an upper bound), and the count is reported.
- **Renormalisation over the options**, with an optional temperature on the log-marginals.
- **Diagnostics, never corrections.** `option_mass`, `stop`, `depth` and the raw top tokens ride along.

The one correction is opt-in. With a position-bias profile (`--position-bias`), an eligible readout subtracts
the profile's centred log prior from the log-marginals before the temperature; `log_marginals` are not debiased;
clamped to <= 0.
"""
from __future__ import annotations

import dataclasses
import hashlib
import http.client
import json
import logging
import math
import os
import socket
import string
import threading
import time
import urllib.error
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import ggufkv
from .bias import is_canonical
from .converge import Probe, converged
from .deadline import (
    Chunk,
    ChunkRun,
    ConvergeRun,
    Estimate,
    Estimators,
    clip_cap,
    run_chunks,
    with_probe_cost,
)
from .errors import (
    CalloError,
    Cancelled,
    InvalidInput,
    MediaLoadFailed,
    MediaRejected,
    MediaTooLarge,
    NoAudio,
    NoVideo,
    NoVision,
    ReadoutError,
    ThinkTooLarge,
    TooManyOptions,
    UpstreamRejected,
    UpstreamUnavailable,
)
from .marginalize import (
    DEFAULT_DIVERGENCE_MASS,
    DEFAULT_DOMINANCE,
    DEFAULT_MAX_DEPTH,
    WIDE_MAX_SPLIT_PASSES,
    WIDE_SPLIT_READ,
    SplitResult,
    TokenLogprobs,
    WalkResult,
    _lookup,
    _marginals_and_terms,
    check_option_sets,
    clamp_marginals,
    continuation_sets,
    decode_token,
    divergence_walk,
    logsumexp,
    marginal_option_logprobs,
    option_mass,
    option_surface_sets,
    partial_sets,
    read_vocab_from_gguf,
    reasks_of,
    split_attribution,
)
from .media import MEDIA_KINDS, detect_media
from .modelkey import load_model_key
from .prompt import (
    TemplateClient,
    TemplateError,
    answer_slot_check,
    check_answer_slot,
    messages,
    user_content,
)
from .prompt import user_delimiter as _user_delimiter
from .scorer import (
    DEFAULT_TIMEOUT,
    ForwardScorer,
    GenResult,
    LlamaCppForwardScorer,
    MMResult,
    make_dist_fn,
    make_http_post,
)
from .systemone import (
    MAX_LETTERS,
    TOO_MANY_OPTIONS,
    WIDE_LABELS,
    InvalidRequest,
    MediaItem,
    Question,
    Request,
    answer_body,
    is_wide,
    media_kind,
    parse_request,
)
from .think import (
    SamplingPolicy,
    ThinkFrame,
    check_on_prompt,
    context_need,
    readout_prefix,
    resolve_sampling,
    think_seed,
)
from .upstream import default_cache_dir, discover, make_http_get
from .vision import SELF_TEST_IMAGE_URL

if TYPE_CHECKING:                       # profile.py imports READOUT_KINDS from here: no module-level import back
    from .profile import PositionBias

DEFAULT_TOP_K = 100
WIDE_TOP_K = 1000               # n_probs of every pass of a wide question: max(engine.top_k, WIDE_TOP_K)
NEG_INF = float("-inf")
log = logging.getLogger("callo")
# A transient upstream failure, mapped to 503 (never 500). `http.client.HTTPException` covers IncompleteRead (the
# upstream dies mid-body); `json.JSONDecodeError` (a ValueError) covers a truncated/invalid upstream response body —
# on the engine path json.loads only ever parses the upstream, so it cannot be a client fault. HTTPError is a
# URLError but is caught ahead of _BACKEND_DOWN in answer_parsed, so it is classified first: `classify_media_error`
# for a media request, else 503.
_BACKEND_DOWN = (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout,
                 http.client.HTTPException, json.JSONDecodeError)
_BAD_DATA_URL = ("Invalid uri-encoded base64 value", "uri must be base64 encoded")   # std::invalid_argument -> 400
# /apply-template's (and chat parsing's) refusal of a part type the upstream lacks (server-common.cpp:1155-1185): 501.
_NOT_SUPPORTED = (("image input is not supported", NoVision), ("audio input is not supported", NoAudio),
                  ("video input is not supported", NoVideo))
_MEDIA_NOUN = {"image": "an image", "audio": "an audio clip", "video": "a video", "mixed": "a media item"}
# The prompt (media tokens included) does not fit the slot: ERROR_TYPE_EXCEED_CONTEXT_SIZE, a 400 whose type is
# "exceed_context_size_error" (server-common.cpp:68-70; the two messages at server-context.cpp:3343-3357 @ 1537a0a).
_CONTEXT_TYPE = "exceed_context_size_error"
_CONTEXT_TEXT = ("exceeds the available context size", "is larger than the max context size")
TOO_LARGE_DETAIL = ("the request with its media exceeds the context size; lower --video-fps / --image-max-tokens or use "
                    "a shorter clip")


def _upstream_error(e: urllib.error.HTTPError) -> tuple[str, str | None]:
    """llama-server's `{"error": {"message": ..., "type": ...}}`: (the message, else the raw body, else the reason;
    the error type or None). Reads the body, which can be read only once."""
    try:
        raw = e.read() or b""
    except Exception:  # noqa: BLE001 -- a broken error body must not hide the error itself
        raw = b""
    try:
        err = json.loads(raw).get("error")
        msg = err.get("message") if isinstance(err, dict) else None
        etype = err.get("type") if isinstance(err, dict) else None
    except (ValueError, AttributeError):
        msg, etype = None, None
    text = msg if isinstance(msg, str) and msg else (raw.decode(errors="replace").strip()[:500] or str(e.reason))
    return text, etype if isinstance(etype, str) else None


def _upstream_message(e: urllib.error.HTTPError) -> str:
    """llama-server's `{"error": {"message": ...}}`, else the raw body, else the reason."""
    return _upstream_error(e)[0]


def _request_texts(req: Request) -> list[str]:
    return [req.state, req.context, *(t for q in req.questions for t in (q.instructions, *q.texts))]


def classify_media_error(e: urllib.error.HTTPError, *, kind: str = "mixed") -> CalloError:
    """For an upstream HTTP error on a request that carries MEDIA (`kind` is its
    `Request.kind`, which only words the 422): by message first, then by status. Reads the error body, which can be
    read only once, so call it once per error. The statuses llama.cpp actually sends (@ 1537a0a) are noted; the
    mapping keys on the message, so it holds whatever the status.

    - "Failed to load image or audio file": 422 (`MediaLoadFailed`). /completion sends it as a 400: tokenization runs
      inside handle_completions_impl's try, which answers ERROR_TYPE_INVALID_REQUEST (server-common.cpp:937-943,
      server-context.cpp:4493, 4560-4562).
    - the context exceeded (`exceed_context_size_error`, a 400): 413 (`MediaTooLarge`), a deterministic fault of the
      request (a long clip, a high --video-fps), not a 500.
    - "image/audio/video input is not supported" (/apply-template, a std::runtime_error, so a 500 through ex_wrapper):
      501 (`NoVision` / `NoAudio` / `NoVideo`): this upstream can never answer the request. A 501 is still a 5xx, which
      Kev's RemotePredictor retries like a 503 (it drops 4xx, retries 5xx); the status names the configuration fault.
    - a bad data URL (`_BAD_DATA_URL`, 400): 422. Defensive: /apply-template sees only placeholders and
      /completion's multimodal_data is raw base64, so callo's own checks make it unreachable.
    - any other 4xx (e.g. /completion's 400 "Failed to tokenize prompt"): 500 (`UpstreamRejected`); any other 5xx
      (e.g. /apply-template's 500 "Invalid base64 value"): 503 (`UpstreamUnavailable`)."""
    msg, etype = _upstream_error(e)
    if "Failed to load image" in msg:
        return MediaLoadFailed(f"the upstream could not decode {_MEDIA_NOUN.get(kind, 'a media item')}: {msg}")
    if etype == _CONTEXT_TYPE or any(s in msg for s in _CONTEXT_TEXT):
        return MediaTooLarge(f"{TOO_LARGE_DETAIL} (upstream: {msg})")
    for text, refused in _NOT_SUPPORTED:
        if text in msg:
            return refused()
    if e.code == 400 and any(s in msg for s in _BAD_DATA_URL):
        return MediaRejected(msg)
    if 400 <= e.code < 500:
        return UpstreamRejected(f"upstream HTTP {e.code}: {msg}")
    return UpstreamUnavailable(f"upstream HTTP {e.code}: {msg}")


def classify_image_error(e: urllib.error.HTTPError) -> CalloError:
    """The name: `classify_media_error` for an image request."""
    return classify_media_error(e, kind="image")


# Every Readout.kind: plain, context, image, audio, video and mixed.
# Defined once: a position-bias profile's set.kinds is validated against it (profile.py).
READOUT_KINDS = ("plain", "context", "image", "audio", "video", "mixed")


@dataclass(frozen=True)
class Readout:
    probs: tuple[float, ...]
    log_marginals: tuple[float, ...]
    option_mass: float
    stop: str
    depth: int
    topk_miss: int
    prompt_tokens: int
    top_tokens: tuple[tuple[int, float], ...] = ()   # (token id, logprob) of the 5 likeliest raw next tokens
    kind: str = "plain"                 # READOUT_KINDS: which path answered
    cache_n: int | None = None          # first upstream pass, cached paths only: tokens reused from the slot
    prompt_n: int | None = None         # first upstream pass, cached paths only: tokens evaluated
    prompt_ms: float | None = None      # first upstream pass, cached paths only: server-side prompt time
    bias: str | None = None             # applied | no_entry | noncanonical | score | single_option |
                                        # kind_not_calibrated; None when no position-bias profile is loaded;
                                        # "deliberate" on every post-think readout (never debiased)
    passes: int = 0                     # upstream scoring calls actually made (0: the
                                        # single-option shortcut; a retok stop before posting is not a pass);
                                        # wide-choice: the split passes included
    reasks: int = 0                     # extra requests for passes whose probabilities llama-server withheld (an
                                        # incomplete UTF-8 pick, re-asked with it banned); not in `passes`
    # wide-choice, at their defaults for 26 or fewer options: the first-letter mass at the slot no split
    # pass attributed (an upper bound, never spread over the labels), the split passes, whether the question was
    # wide, the first-letter mass per first letter of the question's labels, and how many of its labels are not
    # single tokens in this vocabulary (read in two passes; 0 for a question within `wide_coverage`)
    unread_mass: float = 0.0
    split_passes: int = 0
    wide: bool = False
    partial_by_letter: tuple[tuple[str, float], ...] = ()
    uncovered: int = 0


@dataclass(frozen=True)
class Trace:
    """One deliberate generation: the trace tokens and what it cost."""
    gen_tokens: tuple[int, ...]         # every sampled token, the close tag included
    n_gen: int                          # len(gen_tokens) (tokens_predicted when the two agree)
    close: str                          # natural | forced | eos
    truncated: bool
    gen_wall_ms: float
    gen_prompt_ms: float | None
    gen_prompt_n: int | None
    gen_cache_n: int | None
    predicted_ms: float | None
    frame_ms: float                     # the thinking-on render plus its tokenization
    content: str                        # the generated text (diagnostics only)
    prompt_tokens: int                  # len(on_tokens)
    # think deadlines: a chunked trace sums gen_wall_ms and predicted_ms over its chunks; gen_prompt_*
    # are chunk 0's and cont_prompt_* the later chunks' sums; a single generation is one chunk with no continuation
    chunks: int = 1
    cont_prompt_n: int | None = None
    cont_prompt_ms: float | None = None
    cont_reprefill: int = 0             # continuation chunks that evaluated more than one token (cache evicted);
                                        # not the chunks after an early-exit probe: after_probe counts those


@dataclass(frozen=True)
class BudgetReadout:
    """The readout at one budget: the cut k_B = min(B, n_gen), read at the token-built prefix."""
    readout: Readout
    k: int
    read_ms: float
    p_ms: float | None                  # the readout's first-pass prompt_ms
    read_prompt_n: int | None
    read_cache_n: int | None
    text_fallback: bool


@dataclass(frozen=True)
class Deliberate:
    trace: Trace
    readouts: dict[int, BudgetReadout]  # budget -> its readout; budgets >= n_gen share one (the same object)
    n_readouts: int                     # distinct readouts made
    by_k: dict[int, BudgetReadout] = dataclasses.field(default_factory=dict)   # every cut read, by k


@dataclass(frozen=True)
class ChunkedReadout:
    """A question read after a chunked trace: the trace (`close` in calibration's words: natural, forced or
    eos), its serve-facing close (`natural` | `budget` | `deadline` | `eos`), the readout at k with its temperature,
    the chunks, and in the probe's live mode (and under early exit) the forced readout at every probe boundary.
    `reused`: the answer is a probe re-tempered, not a read after the close (early exit's `converged`, or a
    `deadline` cut on a probe boundary): its read is already in the probe window, and the post-close read cost 0."""
    trace: Trace
    close: str
    readout: BudgetReadout
    temperature: float
    calibrated: bool
    chunks: tuple[Chunk, ...]
    probes: tuple[BudgetReadout, ...] = ()
    cap: int | None = None              # the cap the trace ran under, clipped to the slot context
    reused: bool = False                # the answer is a re-tempered probe (early exit), no read after the close


@dataclass(frozen=True)
class QuestionReadout:
    """One answered question of a /v1/systemone request: its id, its readout and its wall time. With a think budget:
    the trace and the budget it was read at, with the served temperature and whether it was calibrated, for
    X-Callo-Timings and the stats."""
    qid: str
    readout: Readout
    ms: float
    think: Trace | None = None
    think_budget: int = 0
    think_k: int = 0                    # k_B: the think tokens the readout saw
    read_prompt_n: int | None = None
    think_T: float | None = None
    calibrated: bool | None = None
    text_fallback: bool = False
    # think deadlines: the serve-facing close (`natural` | `budget` | `deadline` | `eos`, or `budget0`
    # for the fallback), the deadline it was served under, its time from the request's arrival to this readout's end
    # (set by the service), its chunks, and `deadline_fallback` (`budget0` when thinking did not fit |
    # `not_chunkable` | `media`)
    think_close: str | None = None
    deadline_ms: int | None = None
    elapsed_ms: float | None = None
    chunks: int | None = None
    deadline_fallback: str | None = None
    # early exit: the target whose rule served (a float), or why early exit was off for this question (`unreachable` |
    # `not_chunkable` | `media`); the rule's (n, y) and the probes read, when it ran
    converge: float | str | None = None
    converge_rule: tuple[int, float] | None = None
    probes: int | None = None
    # early exit's gate (experimental), on each deliberate question of a request it was asked for: P(yes), the read's
    # ms, whether it skipped early exit, and why the read failed (a failed read skips)
    converge_gate: dict | None = None


@dataclass(frozen=True)
class DeadlineRun:
    """One question's deadline, as the service builds it: the trace cap (`explicit_cap`: a
    budget's, with the budget path's 413; else B*, clipped), the share's and the request's ends (None: early exit
    without a deadline) and the reserve in ms on `clock`, the server-wide estimators, the temperature rule and early
    exit's run (None: off)."""
    deadline_ms: int
    cap: int
    explicit_cap: bool
    share_end_ms: float | None
    request_end_ms: float | None
    reserve_ms: float
    estimators: Estimators | None
    temperature_of: Callable[[str, int, int], tuple[float, bool]]
    clock: Callable[[], float]
    converge: ConvergeRun | None = None


class _RetokStop(Exception):
    """A walk step whose text does not re-tokenize to the same ids: read out where the walk stood."""


_NO_MEDIA = {"image": NoVision, "audio": NoAudio, "video": NoVideo}
_SUPPORT_FLAG = {"image": "vision", "audio": "audio", "video": "video"}


def _image_items(images: Sequence[str], image_urls: Sequence[str]) -> tuple[MediaItem, ...]:
    """The `images=` / `image_urls=` as image `MediaItem`s, paired by position (mtmd maps the i-th marker to the
    i-th multimodal_data entry, so a count mismatch is refused)."""
    if len(images) != len(image_urls):
        raise ReadoutError(f"{len(images)} image payload(s) but {len(image_urls)} image URL(s)")
    return tuple(MediaItem("image", p, u) for p, u in zip(images, image_urls, strict=True))


def _top5(token_logprobs: TokenLogprobs) -> tuple[tuple[int, float], ...]:
    items = token_logprobs.items() if isinstance(token_logprobs, Mapping) else enumerate(token_logprobs)
    return tuple(sorted(((int(t), float(v)) for t, v in items), key=lambda kv: -kv[1])[:5])


class SurfaceSets:
    """Option surface sets over ONE vocabulary, memoised per label SET. `get` returns a FRESH dict in the
    requested order, so a caller can neither mutate the cache nor see another caller's order.

    Wide-choice: the two-letter label sets are built ONCE, for `WIDE_LABELS[:wide_readable()]`, and every wide
    question takes a slice; the partial (first-letter) and continuation sets over the 26 letters are built once each.
    A label that is not one token both bare and after a space is UNCOVERED: it is read in two passes (its first
    letter, then the second letter after it), and its own set may be empty (Spark: BK, HQ; llama-spm: 28 labels), so
    the table is checked with invariant (a) waived. On a vocabulary that covers every label the table, its checks and
    every readout are what they were before uncovered labels were read."""

    def __init__(self, tokens: Sequence[str], tok_model: str):
        self._tokens, self._model = list(tokens), tok_model
        self._cache: dict[tuple[str, ...], dict[str, frozenset[int]]] = {}
        self._lock = threading.Lock()
        self._coverage: int | None = None
        self._covered: frozenset[str] = frozenset()     # the WIDE_LABELS one token both bare and after a space
        self._readable: int | None = None
        self._wide: dict[str, frozenset[int]] | None = None
        self._partial: dict[str, frozenset[int]] | None = None
        self._cont: dict[str, frozenset[int]] | None = None

    @classmethod
    def from_gguf(cls, path: str | Path) -> SurfaceSets:
        tokens, tok_model = read_vocab_from_gguf(path)
        return cls(tokens, tok_model)

    def token_str(self, tid: int) -> str:
        return self._tokens[tid] if 0 <= tid < len(self._tokens) else f"<id {tid}>"

    def wide_coverage(self) -> int:
        """How many leading `WIDE_LABELS` this vocabulary carries as ONE token both bare and after a space:
        one pass over the decoded vocabulary, cached (with which labels anywhere in the table are covered)."""
        with self._lock:
            if self._coverage is None:
                decoded = {decode_token(t, self._model) for t in self._tokens}
                covered = frozenset(lab for lab in WIDE_LABELS if lab in decoded and " " + lab in decoded)
                n = 0
                for lab in WIDE_LABELS:
                    if lab not in covered:
                        break
                    n += 1
                self._coverage, self._covered = n, covered
            return self._coverage

    def uncovered(self, labels: Sequence[str]) -> tuple[str, ...]:
        """The labels of a wide question that are not one token both bare and after a space, in the order given:
        empty for every question of `wide_coverage()` or fewer options."""
        self.wide_coverage()
        return tuple(lab for lab in labels if lab not in self._covered)

    def split_letters(self, labels: Sequence[str]) -> tuple[str, ...]:
        """The first letters that lead at least one uncovered label of the question, sorted: each adds one split pass to
        the question's cap (`Engine._read_slot`)."""
        return tuple(sorted({lab[0] for lab in self.uncovered(labels)}))

    def wide_readable(self) -> int:
        """How many leading `WIDE_LABELS` Callo can read on this vocabulary: a covered label always; an uncovered one
        when its first letter starts a token (`partial`) and its second letter is a token of its own
        (`continuations`), so it reads in two passes. Every vocabulary checked so far reads all 255; a vocabulary
        that covers all 255 builds no letter set here. Cached."""
        cov = self.wide_coverage()
        with self._lock:
            if self._readable is not None:
                return self._readable
        if cov == len(WIDE_LABELS):
            n = cov
        else:
            part, cont = self.partial(string.ascii_uppercase), self.continuations()
            n = 0
            for lab in WIDE_LABELS:
                if lab not in self._covered and not (part[lab[0]] and cont[lab[1]]):
                    break
                n += 1
        with self._lock:
            self._readable = n
        return n

    def wide(self, n: int) -> dict[str, frozenset[int]]:
        """The surface sets of `WIDE_LABELS[:n]`, in table order (an uncovered label's set may be empty). The caller
        has checked `26 < n <= wide_readable()` (`Engine.check_coverage`); the sets of every readable label are
        built and checked once, under the lock, invariant (a) waived only when some label is uncovered. Any other n is
        a ValueError (a bug of the caller), raised explicitly so `python -O` cannot strip it."""
        cov, readable = self.wide_coverage(), self.wide_readable()
        if not MAX_LETTERS < n <= readable:
            raise ValueError(f"wide({n}) on a vocabulary reading {readable} two-letter labels: a wide question "
                             f"takes {MAX_LETTERS + 1} to {readable} (Engine.check_coverage refuses the rest first)")
        with self._lock:
            if self._wide is None:
                built = option_surface_sets(self._tokens, self._model, WIDE_LABELS[:readable])
                check_option_sets(built, tokens=self._tokens, allow_empty=readable > cov)
                self._wide = built
            hit = self._wide
        return {lab: hit[lab] for lab in WIDE_LABELS[:n]}

    def partial(self, firsts: Sequence[str]) -> dict[str, frozenset[int]]:
        """{first letter: the tokens that START a two-letter label with it} (`marginalize.partial_sets`), for `firsts`
        in the order given. Built once over the 26 letters (a letter with no such token simply has an empty set). The
        letters' sets are disjoint by construction (a token normalises to ONE letter), so nothing is checked here; the
        invariant that can matter, partial tokens disjoint from every wide label surface, is checked once in
        `warm_wide` (and on every readout by `divergence_walk`)."""
        with self._lock:
            if self._partial is None:
                self._partial = partial_sets(self._tokens, self._model, string.ascii_uppercase)
            hit = self._partial
        return {f: hit[f] for f in firsts}

    def continuations(self) -> dict[str, frozenset[int]]:
        """{letter: the tokens that CONTINUE a two-letter label with it} (`marginalize.continuation_sets`), A-Z, built
        once. A fresh dict each call."""
        with self._lock:
            if self._cont is None:
                self._cont = continuation_sets(self._tokens, self._model, string.ascii_uppercase)
            return dict(self._cont)

    def warm_wide(self) -> int:
        """`callo serve`'s startup warm-up: the coverage, the readable count and, when that exceeds 26, the wide,
        partial and continuation sets. A vocabulary reading 26 or fewer builds nothing wide and never raises. Once
        built, the partial (first-letter) tokens are checked disjoint from every label's surfaces: a token in both
        would be counted as a label AND as first-letter mass. Returns the coverage."""
        cov, readable = self.wide_coverage(), self.wide_readable()
        if readable > MAX_LETTERS:
            wide = self.wide(readable)
            part = frozenset().union(*self.partial(string.ascii_uppercase).values())
            if part:                                # check_option_sets refuses an empty set; no letter tokens: no hazard
                check_option_sets({**wide, "\0partial": part}, tokens=self._tokens, allow_empty=readable > cov)
            self.continuations()
        return cov

    def get(self, labels: tuple[str, ...]) -> dict[str, frozenset[int]]:
        if len(labels) > MAX_LETTERS and tuple(labels) == WIDE_LABELS[:len(labels)]:
            return self.wide(len(labels))           # wide: one build serves every n, no per-n scan
        key = tuple(sorted(labels))
        with self._lock:
            hit = self._cache.get(key)
            if hit is None:
                hit = option_surface_sets(self._tokens, self._model, labels)
                check_option_sets(hit, tokens=self._tokens)
                self._cache[key] = hit
        return {lab: hit[lab] for lab in labels}


def bounded_readout(option_logprobs: Mapping[str, float], token_logprobs: Mapping[int, float] | Sequence[float],
                    labels: Sequence[str], temperature: float = 1.0, offsets: Sequence[float] | None = None
                    ) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """(probs over `labels`, the bounded log-marginals, how many options were bounded). `offsets`: the
    position-bias profile's centred log prior, subtracted from every option, floored ones included, BEFORE the
    temperature: z_k = (lm_k - offsets[k]) / T. The returned log-marginals are not debiased; clamped to <= 0."""
    vals = token_logprobs.values() if isinstance(token_logprobs, Mapping) else token_logprobs
    floor = min(vals)
    lms, miss = [], 0
    for lab in labels:
        v = option_logprobs[lab]
        if v == float("-inf"):
            v, miss = floor, miss + 1
        lms.append(v)
    if offsets is None:
        z = [v / temperature for v in lms]          # the exact expression: no profile, bit-identical readouts
    else:
        if len(offsets) != len(lms):
            raise ValueError(f"{len(offsets)} offsets for {len(lms)} options")
        z = [(v - o) / temperature for v, o in zip(lms, offsets, strict=True)]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    s = math.fsum(e)
    return tuple(x / s for x in e), tuple(lms), miss


def wide_bounded_readout(direct: Mapping[str, float], credit: Mapping[str, float],
                         token_logprobs: Mapping[int, float] | Sequence[float], labels: Sequence[str],
                         temperature: float, no_token: Collection[str] = ()) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """lm = logaddexp(direct if finite else floor, credit): the direct part is floored first, then the split credit is
    added; topk_miss counts only labels with neither a direct value nor any credit. A credit below the
    floor still adds to it (real mass on top of the bound). With no credit anywhere this is `bounded_readout` without
    offsets, bit for bit: a wide question is never debiased.

    `no_token`: labels with no surface token at all in this vocabulary (read only in two passes). Their direct mass is
    exactly 0, not a token below the top-k cut, so a credited one reads its credit alone, P(first) x P(second | first);
    one with no credit is floored and counted like any label with nothing read. Empty for every question within
    `wide_coverage`, which leaves the arithmetic as it was."""
    vals = token_logprobs.values() if isinstance(token_logprobs, Mapping) else token_logprobs
    floor = min(vals)
    lms, miss = [], 0
    for lab in labels:
        d, c = direct[lab], credit[lab]
        if d == NEG_INF and c == NEG_INF:
            miss += 1
        if c != NEG_INF and lab in no_token:
            lms.append(min(c, 0.0))
            continue
        base = d if d != NEG_INF else floor
        lms.append(base if c == NEG_INF else min(logsumexp([base, c]), 0.0))
    z = [v / temperature for v in lms]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    s = math.fsum(e)
    return tuple(x / s for x in e), tuple(lms), miss


def retemper(r: Readout, temperature: float) -> Readout:
    """`r` at another temperature: probs = softmax(log_marginals / T), the walk's own arithmetic on the same
    log-marginals (a deliberate readout carries no position-bias offsets), so a T = 1 probe is the readout at T with no
    second upstream pass."""
    z = [v / temperature for v in r.log_marginals]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    s = math.fsum(e)
    return dataclasses.replace(r, probs=tuple(x / s for x in e))


def _wide_fields(sr: SplitResult | None, uncovered: int = 0) -> dict[str, Any]:
    """The Readout fields a wide readout sets ({} for a narrow one: the defaults, so its Readout is unchanged).
    `uncovered`: the question's labels that are not single tokens (`SurfaceSets.uncovered`)."""
    if sr is None:
        return {}
    return {"unread_mass": sr.unread_mass, "split_passes": sr.split_passes, "wide": True,
            "partial_by_letter": sr.partial_by_letter, "uncovered": uncovered}


class Engine:
    def __init__(self, scorer: ForwardScorer, template: TemplateClient, sets: SurfaceSets, *,
                 top_k: int = DEFAULT_TOP_K, temperature: float = 1.0, variant: str = "main",
                 max_depth: int = DEFAULT_MAX_DEPTH, divergence_mass: float = DEFAULT_DIVERGENCE_MASS,
                 vision: bool = False, media_marker: str | None = None, user_delimiter: str | None = None,
                 model_name: str = "callo", position_bias: PositionBias | None = None, audio: bool = False,
                 video: bool = False, s1_timeout: float = DEFAULT_TIMEOUT):
        if not temperature > 0:
            raise ValueError(f"temperature must be positive, got {temperature!r}")
        # the frame (None: not capable), the per-slot context, the pinned trace sampler and the
        # per-call ceiling the generation timeout is built on (--s1-timeout + 0.1 s per budget token)
        self.think: ThinkFrame | None = None
        self.think_reason: str | None = None          # why `think` is None, for the 422 and the stats
        self.n_ctx: int | None = None
        self.think_sampling: dict | None = None
        self._default_sampling_warned = False       # trace_sampling's one-time fallback warning
        self._count_mismatch_warned = False         # answer_deliberate's one-time tokens_predicted warning
        self.s1_timeout = s1_timeout
        self.scorer, self.template, self.sets = scorer, template, sets
        self.top_k, self.temperature, self.variant = top_k, temperature, variant
        self.max_depth, self.divergence_mass = max_depth, divergence_mass
        self.vision, self.media_marker, self.user_delimiter = vision, media_marker, user_delimiter
        self.audio, self.video = audio, video      # serve turns video off when its self-test cannot load
        # the 501 detail once serve has turned video off (else the default)
        self.video_off_reason: str | None = None
        self.model_name = model_name
        self.position_bias = position_bias
        # think deadlines: the server-wide estimators; set by serve, so a budget-0 readout's first pass (a
        # prefill other slots' chunks may stall behind) is recorded for S_known. None: nothing is recorded
        self.estimators: Estimators | None = None
        # wide-choice split attribution, settable for the forced-split diagnostic (0, None, 1e-3)
        self.split_read_above = WIDE_SPLIT_READ
        self.split_max_passes: int | None = WIDE_MAX_SPLIT_PASSES
        self.split_min_partial = 0.0

    def supports(self, kind: str) -> bool:
        """Whether the upstream takes media of `kind`: its flag, and a media marker to check prompts by."""
        return bool(getattr(self, _SUPPORT_FLAG[kind]) and self.media_marker)

    def refuse_unsupported(self, media: Sequence[MediaItem]) -> None:
        """501 (`NoVision` / `NoAudio` / `NoVideo`) for the first kind, in contract order, the upstream lacks."""
        for kind in MEDIA_KINDS:
            if any(m.kind == kind for m in media) and not self.supports(kind):
                if kind == "video" and self.video_off_reason is not None:
                    raise NoVideo(self.video_off_reason)
                raise _NO_MEDIA[kind]()

    def bias_for(self, q: Question, kind: str) -> tuple[tuple[float, ...] | None, str | None]:
        """The offsets to subtract for this readout and its `Readout.bias` category; (None, None) without a
        profile. Eligible: `choice`/`noul`, canonical, the readout's kind among the profile's `set.kinds`, and an
        entry for n. A profile never adds an upstream call: this is a lookup."""
        pb = self.position_bias
        if pb is None:
            return None, None
        if len(q.labels) == 1:
            return None, "single_option"
        if q.type == "score":
            return None, "score"
        if is_wide(q):                          # no position-bias correction above 26 options
            return None, "wide"
        if not is_canonical(q):
            return None, "noncanonical"
        if kind not in pb.kinds:
            return None, "kind_not_calibrated"
        offsets = pb.offsets.get(len(q.labels))
        return (offsets, "applied") if offsets is not None else (None, "no_entry")

    def answer(self, state: str, q: Question, *, id_slot: int | None = None, context: str = "",
               media: Sequence[MediaItem] = (), images: Sequence[str] = (), image_urls: Sequence[str] = (),
               temperature: float | None = None, position_bias: bool = True) -> Readout:
        """One question. `media` are the request's items in contract order (`Request.media`). `images`
        (base64 payloads) and `image_urls` (the same images as data URLs) are the older keywords, kept as deprecated
        aliases for image-only callers (`vision.py`-era code, `local_engine` users). A request with neither media nor
        `context` takes exactly the plain text path below. The aliases are paired lazily:
        after the single-option shortcut and the 501, and `image_urls=` without `images=` takes the text
        path. `temperature`: the T of this readout; None is the engine's, so every served
        readout is unchanged (the probe reads budget 0 at T = 1 on a server at any --temperature). `position_bias`
        False reads the model's own probabilities with no position-bias offsets and `bias` None (early exit's gate,
        whose 0.5 rule is defined on them; its readout is never served or counted)."""
        if media and (images or image_urls):
            raise ValueError("pass media or images/image_urls, not both")
        items = tuple(media)
        if items:
            kind = media_kind(items, context)
        else:                                   # the kind for the aliases
            kind = "image" if images else "context" if context else "plain"
        offsets, bias = self.bias_for(q, kind) if position_bias else (None, None)
        if len(q.labels) == 1:
            return Readout(probs=(1.0,), log_marginals=(0.0,), option_mass=1.0, stop="single_option", depth=0,
                           topk_miss=0, prompt_tokens=0, kind=kind, bias=bias, passes=0)
        if not items and images:
            # NoVision before the pairing
            self.refuse_unsupported((MediaItem("image", images[0], None),))
            items = _image_items(images, image_urls)
        self.check_coverage(q)                  # the 422 before any render, tokenize or pass
        if items:
            return self._answer_media(state, q, id_slot, context, items, kind=kind, offsets=offsets, bias=bias,
                                      temperature=temperature)
        if context:
            return self._answer_context(state, q, id_slot, context, offsets=offsets, bias=bias,
                                        temperature=temperature)
        prompt = self.template.render(messages(state, q, self.variant))
        prefix = tuple(self.scorer.tokenize(prompt, add_special=True))
        dist = self._first_pass_recorded(make_dist_fn(self.scorer, top_k=self._k(q), id_slot=id_slot,
                                                      delimiter=self.user_delimiter), id_slot, len(prefix))
        calls = reasks = 0

        def counted(cur: tuple[int, ...]) -> TokenLogprobs:
            nonlocal calls, reasks
            out = dist(cur)
            calls += 1
            reasks += reasks_of(out)
            return out

        sets = self._sets(q)
        walk, sr = self._read_slot(q, sets, counted, prefix)
        T = self.temperature if temperature is None else temperature
        if sr is None:
            probs, lms, miss = bounded_readout(walk.option_logprobs, walk.token_logprobs, q.labels, T, offsets)
            mass = walk.option_mass
        else:
            probs, lms, miss = wide_bounded_readout(walk.option_logprobs, sr.credit, walk.token_logprobs, q.labels, T,
                                                    self._no_token(q, sets))
            mass = sr.option_mass
        items_lp = walk.token_logprobs.items() if isinstance(walk.token_logprobs, Mapping) else enumerate(walk.token_logprobs)
        top5 = tuple(sorted(((int(t), float(v)) for t, v in items_lp), key=lambda kv: -kv[1])[:5])
        return Readout(probs=probs, log_marginals=lms, option_mass=mass, stop=walk.stop, depth=walk.depth,
                       topk_miss=miss, prompt_tokens=len(prefix), top_tokens=top5, bias=bias, passes=calls,
                       reasks=reasks, **_wide_fields(sr, self._uncovered(q)))

    # ---- the /v1/systemone body, shared by the proxy and local_engine --------------------

    def answer_parsed(self, req: Request, *, id_slot: int | None = None,
                      readouts: list[QuestionReadout] | None = None, model_name: str | None = None,
                      think_budget: int = 0, think_T: float | None = None, think_calibrated: bool = True,
                      cancel: threading.Event | None = None, deadline: DeadlineRun | None = None) -> dict:
        """Every question of one request on `id_slot`, as the /v1/systemone body. Deliberate failures are
        `CalloError`s (`classify_media_error` and the branches below are the status mapping); a bug propagates
        unchanged. `readouts` receives one `QuestionReadout` per question for callers that want the diagnostics
        (stats, the X-Callo-Timings header, the demo's logs).

        Think budgets: with `think_budget` > 0 every question with 2+ options is read after a trace of at
        most that many tokens (`answer_deliberate`) at `think_T` (None: this engine's temperature), seeded by the
        request's seed digest (its digest without `think_converge`) and the question id; `think_calibrated` only rides along into the QuestionReadout. `cancel`
        is checked between questions: a client that left ends the request with `Cancelled`. An engine with no frame
        (`think` None: a model that cannot think, or `local_engine`, which detects none) refuses a positive budget with
        InvalidRequest (422), naming `think_reason`.

        `usage` keeps the shape: `input_tokens` sums each question's readout prompt (a deliberate question's
        includes its k_B trace tokens, which the readout reads), and `output_tokens` stays 0 because callo returns no
        generated text; the trace tokens are reported in X-Callo-Timings (`think_tokens`) and /callo/stats instead. A
        single-option question takes the shortcut (no trace) at any budget, so a request of only such questions
        generates nothing and counts in no `think` counter.

        Think deadlines: with `deadline` every question with 2+ options is read after a chunked trace
        (`answer_chunked`) under it, or, when thinking does not fit its share, at budget 0 (the `budget0` fallback, at
        this engine's temperature). A read after a `deadline` or `budget` close feeds the reserve's window (a reused
        early-exit probe does not: nothing was read after the close)."""
        if think_budget > 0 and req.media:              # the service refuses first; local_engine lands here
            raise MediaRejected("deliberate readouts are text-only in this version")
        if think_budget > 0 and self.think is None:     # the service refuses first (think_plan); local_engine lands here
            raise InvalidRequest(f"think_budget {think_budget}: "
                                 f"{self.think_reason or 'this model has no thinking mode callo can read'}; this "
                                 f"engine serves budget 0 only")
        self.check_request(req)
        answers, tokens = {}, 0
        for q in req.questions:
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client left during the deliberate request")
            t0 = time.perf_counter()
            extra: dict = {}
            try:
                if deadline is not None and len(q.labels) > 1:
                    r, extra = self._answer_deadline(req, q, deadline, id_slot=id_slot, cancel=cancel)
                elif think_budget > 0 and len(q.labels) > 1:
                    d = self.answer_deliberate(req.state, q, [think_budget], id_slot=id_slot, context=req.context,
                                               seed=think_seed(req.seed_digest, q.qid), sampling=self.trace_sampling(),
                                               temperature=think_T, cancel=cancel)
                    b = d.readouts[think_budget]
                    r = b.readout
                    extra = {"think": d.trace, "think_budget": think_budget, "think_k": b.k,
                             "read_prompt_n": b.read_prompt_n, "think_T": think_T if think_T is not None else self.temperature,
                             "calibrated": think_calibrated, "text_fallback": b.text_fallback}
                else:
                    r = self.answer(req.state, q, id_slot=id_slot, context=req.context, media=req.media)
            except urllib.error.HTTPError as e:     # before _BACKEND_DOWN: HTTPError is a URLError
                if req.media:                       # every request that carries media, any kind
                    raise classify_media_error(e, kind=req.kind) from e
                raise UpstreamUnavailable(f"llama-server unavailable: {e}") from e
            except _BACKEND_DOWN as e:
                raise UpstreamUnavailable(f"llama-server unavailable: {e}") from e
            except TemplateError as e:
                raise ReadoutError(str(e)) from e
            if readouts is not None:
                readouts.append(QuestionReadout(q.qid, r, 1000 * (time.perf_counter() - t0), **extra))
            answers[q.qid] = answer_body(q, r.probs)
            tokens += r.prompt_tokens
        return {"model": model_name or self.model_name, "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0}}

    def _answer_deadline(self, req: Request, q: Question, dl: DeadlineRun, *, id_slot: int | None,
                         cancel: threading.Event | None) -> tuple[Readout, dict]:
        """One chunked question, under a deadline and/or early exit: (its readout, its QuestionReadout fields)."""
        cr = self.answer_chunked(req.state, q, cap=dl.cap, explicit_cap=dl.explicit_cap, seed=(req.seed_digest, q.qid),
                                 sampling=self.trace_sampling(), temperature_of=dl.temperature_of,
                                 share_end_ms=dl.share_end_ms, request_end_ms=dl.request_end_ms,
                                 reserve_ms=dl.reserve_ms, estimators=dl.estimators, id_slot=id_slot,
                                 context=req.context, cancel=cancel, clock=dl.clock, converge=dl.converge)
        conv0 = {"converge": dl.converge.z, "converge_rule": (dl.converge.n, dl.converge.y), "probes": 0} \
            if dl.converge is not None else {}
        if cr is None:                                          # too short to think: budget 0, as served today
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client left before the budget-0 readout")
            r = self.answer(req.state, q, id_slot=id_slot, context=req.context)
            return r, {"think_close": "budget0", "deadline_ms": dl.deadline_ms if dl.deadline_ms > 0 else None,
                       "chunks": 0, "deadline_fallback": "budget0", **conv0}
        # r's window takes reads after a deadline trace's close; a reused probe (early exit) read nothing after it
        if cr.close in ("deadline", "budget") and not cr.reused and dl.estimators is not None and dl.deadline_ms > 0:
            dl.estimators.add_readout(cr.readout.read_ms)
        extra = {"think": cr.trace, "think_budget": cr.cap, "think_k": cr.readout.k,
                 "read_prompt_n": cr.readout.read_prompt_n, "think_T": cr.temperature, "calibrated": cr.calibrated,
                 "text_fallback": cr.readout.text_fallback, "think_close": cr.close,
                 "deadline_ms": dl.deadline_ms if dl.deadline_ms > 0 else None, "chunks": cr.trace.chunks}
        if dl.converge is not None:
            extra.update(converge=dl.converge.z, converge_rule=(dl.converge.n, dl.converge.y), probes=len(cr.probes))
        return cr.readout.readout, extra

    def check_request(self, req: Request) -> None:
        """The checks that need no upstream call: a media kind the upstream lacks (`NoVision` / `NoAudio` / `NoVideo`,
        501, the first missing kind in contract order), and a media request whose text holds the upstream's media
        marker (`MediaRejected`, 422: mtmd would see one more media slot than items, a deterministic fault of the
        request, not a 500 to retry). The proxy runs this before it picks a slot, so a rejected request never binds a
        lane. It runs ahead of `answer`, so it also covers a single-label media request, which `answer` would
        otherwise settle through the single-option shortcut."""
        self.refuse_unsupported(req.media)
        if req.media and self.media_marker and any(self.media_marker in t for t in _request_texts(req)):
            raise MediaRejected("the request's text contains the upstream's media marker (GET /props media_marker); "
                                "remove it")
        for q in req.questions:                     # after the media checks (a 501 wins), pre-slot
            self.check_coverage(q)

    def check_coverage(self, q: Question) -> None:
        """`TooManyOptions` (422) for a wide question with more options than this vocabulary's `wide_readable`
        (labels past `wide_coverage` are read in two passes; only a label whose letters are not tokens is refused); a
        no-op for 26 or fewer options, and for a wide question within `wide_coverage`, which never asks for more."""
        if is_wide(q):
            m = len(q.labels)
            if m > self.sets.wide_coverage():
                readable = self.sets.wide_readable()
                if m > readable:
                    raise TooManyOptions(TOO_MANY_OPTIONS.format(qid=q.qid, n=readable, m=m))

    def _no_token(self, q: Question, sets: Mapping[str, frozenset[int]]) -> frozenset[str]:
        """A wide question's labels with no surface token at all in its `sets` (none within `wide_coverage`)."""
        if len(q.labels) <= self.sets.wide_coverage():
            return frozenset()
        return frozenset(lab for lab in q.labels if not sets[lab])

    def _uncovered(self, q: Question) -> int:
        """How many of a wide question's labels are not single tokens (0 within `wide_coverage`, and for a narrow
        question, whose Readout never carries it)."""
        if not is_wide(q) or len(q.labels) <= self.sets.wide_coverage():
            return 0
        return len(self.sets.uncovered(q.labels))

    def _k(self, q: Question) -> int:
        """The top-k (`n_probs`) of every pass of this question's readout: WIDE_TOP_K for a wide one, so a floored
        label is bounded by 1/1000; the engine's otherwise (narrow payloads unchanged)."""
        return max(self.top_k, WIDE_TOP_K) if is_wide(q) else self.top_k

    def _sets(self, q: Question) -> dict[str, frozenset[int]]:
        """The question's surface sets, after the coverage check (a wide question past `wide_coverage` is the 422)."""
        self.check_coverage(q)
        return self.sets.get(q.labels)

    def _read_slot(self, q: Question, sets: Mapping[str, frozenset[int]],
                   dist_fn: Callable[[tuple[int, ...]], TokenLogprobs], prefix: tuple[int, ...],
                   *, skip: type[BaseException] | tuple[type[BaseException], ...] = ()
                   ) -> tuple[WalkResult, SplitResult | None]:
        """The one slot reader of the plain path and `_walk` (context, media, deliberate), on the caller's `sets`
        (`_sets(q)`, fetched once per readout): today's walk for 26 or fewer options (no split result); for a wide
        question, the walk with the question's first-letter tokens as `partial` (they stop it, never count as option
        mass), then split attribution at the slot. `skip`: exceptions a split pass raises before posting (the
        media path's _RetokStop), which leave its candidate unread.

        A question past `wide_coverage` has uncovered labels (not single tokens: their own sets may be empty, so the
        walk waives invariant (a)), whose mass at the slot is first-letter mass. Every first letter leading one of
        them adds one split pass to the cap (`split_max_passes` + those letters; None stays uncapped) and is reserved
        one (`split_attribution`'s `reserve`): the greedy expansion reads by mass, but never spends the last passes on
        other tokens while such a letter with real mass is unread. Within the coverage all three are what they were:
        no letter adds or reserves a pass and the walk checks every set."""
        dominance = max(DEFAULT_DOMINANCE, self.divergence_mass)
        if not is_wide(q):
            return divergence_walk(sets, dist_fn, prefix=prefix, max_depth=self.max_depth,
                                   divergence_mass=self.divergence_mass, dominance=dominance), None
        partial = self.sets.partial(tuple(sorted({lab[0] for lab in q.labels})))
        split = self.sets.split_letters(q.labels) if len(q.labels) > self.sets.wide_coverage() else ()
        max_passes = self.split_max_passes
        if split and max_passes is not None:
            max_passes += len(split)
        w = divergence_walk(sets, dist_fn, prefix=prefix, max_depth=self.max_depth,
                            divergence_mass=self.divergence_mass, dominance=dominance,
                            partial=frozenset().union(*partial.values()), allow_empty=bool(split))
        return w, split_attribution(w, q.labels, partial, self.sets.continuations(), dist_fn, skip=skip,
                                    read_above=self.split_read_above, max_passes=max_passes,
                                    min_partial=self.split_min_partial, reserve=split)

    def _retok_split(self, q: Question, sets: Mapping[str, frozenset[int]], cur: tuple[int, ...],
                     token_lps: TokenLogprobs, option_lps: dict[str, float], mass: float, n_terms: int) -> SplitResult:
        """A wide question's retok readout: the WalkResult at the last scored position, built by hand from the
        caller's marginals (`option_lps` clamped, `mass`, and the `n_terms` they summed) plus the first-letter mass
        there (which the split guard checks), then split attribution with no dist_fn: nothing is expanded, every
        first-letter token there is unread."""
        partial = self.sets.partial(tuple(sorted({lab[0] for lab in q.labels})))
        part = frozenset().union(*partial.values())
        w = WalkResult(prefix=cur, token_logprobs=token_lps, option_logprobs=option_lps, option_mass=mass,
                       depth=len(cur), stop="retok", sets=sets,
                       partial_mass=min(math.exp(logsumexp([_lookup(token_lps, t) for t in sorted(part)])), 1.0),
                       n_terms=n_terms)
        return split_attribution(w, q.labels, partial, self.sets.continuations(), None)

    def answer_request(self, body: dict, *, id_slot: int | None = None,
                       readouts: list[QuestionReadout] | None = None) -> dict:
        """Exactly the /v1/systemone response body. Raises `InvalidRequest` (422) or a `CalloError`. A
        blocking call bounded by the scorer's timeout: nothing can abort the upstream readout once it is sent. A body's
        `think_budget` is honoured on an engine whose `think` frame is set, uncalibrated and at
        this engine's temperature (marked `calibrated: false` in the readouts); `local_engine` detects no frame, so a
        positive budget there is an InvalidRequest: use `callo serve` for deliberate readouts."""
        req = parse_request(body)
        return self.answer_parsed(req, id_slot=id_slot, readouts=readouts, think_budget=req.think_budget or 0,
                                  think_calibrated=False)

    # ---- media (image, audio, video, mixed) and context requests --------------------------

    def render_prompt(self, state: str, q: Question, *, context: str = "", media: Sequence[MediaItem] = (),
                      image_urls: Sequence[str] = ()) -> str:
        """The exact prompt string a context or media request sends. Media parts are rendered as fixed
        placeholders: the template is media-blind, so the string is byte-identical to one rendered with the
        real payloads, which go only to /completion. The server's media marker must occur exactly once per item, all
        of them before the text part, or the readout would score a prompt that is not the one asked for.
        `image_urls` is the form (image data URLs), kept for its callers."""
        if media and image_urls:
            raise ValueError("pass media or image_urls, not both")
        items = tuple(media) or tuple(MediaItem("image", "", u) for u in image_urls)
        prompt = self.template.render(messages(state, q, self.variant, context=context, media=items, placeholder=True))
        if items:
            marker = self.media_marker or ""
            n = prompt.count(marker) if marker else 0
            at = prompt.rfind(user_content(state, q))
            if not marker or n != len(items) or at < 0 or prompt.rfind(marker) > at:
                noun = media_kind(items, "")
                noun = "media" if noun == "mixed" else noun
                raise ReadoutError(f"chat template did not render the {noun} parts: {n} media marker(s) for "
                                   f"{len(items)} item(s), all required before the text part")
        return prompt

    def _walk(self, q: Question, dist_fn: Callable[[tuple[int, ...]], TokenLogprobs], prefix: tuple[int, ...],
              passes: list[tuple[tuple[int, ...], MMResult]], *, kind: str, prompt_tokens: int | None,
              offsets: Sequence[float] | None = None, bias: str | None = None,
              temperature: float | None = None) -> Readout:
        """`temperature`: the T of this readout; None is the engine's, so every budget-0 path is
        unchanged. Split passes skip a `_RetokStop` (only the media dist_fn raises it, before posting): that candidate
        stays unread and is no pass. A retok stop of the walk itself reads out at the last scored position."""
        sets = self._sets(q)
        sr: SplitResult | None = None
        try:
            w, sr = self._read_slot(q, sets, dist_fn, prefix, skip=_RetokStop)
            option_lps, token_lps, mass, stop, depth = w.option_logprobs, w.token_logprobs, w.option_mass, w.stop, w.depth
        except _RetokStop:
            cur, last = passes[-1]          # the last position that WAS scored; a retok needs at least one pass
            token_lps = last.logprobs
            # the walk's own readout: the mass checked (then clamped to 1), each log-marginal clamped to <= 0
            uncovered = self._uncovered(q) > 0      # an uncovered label's set may be empty: it reads -inf here
            mass, stop, depth = option_mass(sets, token_lps, allow_empty=uncovered), "retok", len(cur)
            if is_wide(q):                  # the marginals once, with their term count for the split's slack
                raw, n_terms = _marginals_and_terms(sets, token_lps, allow_empty=uncovered)
                option_lps = clamp_marginals(raw)
                sr = self._retok_split(q, sets, cur, token_lps, option_lps, mass, n_terms)
            else:
                option_lps = clamp_marginals(marginal_option_logprobs(sets, token_lps))
        T = self.temperature if temperature is None else temperature
        if sr is None:
            probs, lms, miss = bounded_readout(option_lps, token_lps, q.labels, T, offsets)
        else:
            probs, lms, miss = wide_bounded_readout(option_lps, sr.credit, token_lps, q.labels, T, self._no_token(q, sets))
            mass = sr.option_mass
        first = passes[0][1]
        n = prompt_tokens if prompt_tokens is not None else (first.tokens_evaluated or 0)
        return Readout(probs=probs, log_marginals=lms, option_mass=mass, stop=stop, depth=depth, topk_miss=miss,
                       prompt_tokens=n, top_tokens=_top5(token_lps), kind=kind, cache_n=first.cache_n,
                       prompt_n=first.prompt_n, prompt_ms=first.prompt_ms, bias=bias, passes=len(passes),
                       reasks=sum(reasks_of(p.logprobs) for _, p in passes), **_wide_fields(sr, self._uncovered(q)))

    def _answer_media(self, state: str, q: Question, id_slot: int | None, context: str, media: tuple[MediaItem, ...],
                      *, kind: str, offsets: Sequence[float] | None = None, bias: str | None = None,
                      temperature: float | None = None) -> Readout:
        """For every media kind: `prompt_string` must be a string, so the walk appends DETOKENIZED
        text to the rendered prompt and resends the same payloads, in contract order (every pass after the first hits
        the cache; a video is decoded again by ffmpeg on every pass, which `passes` makes visible). Before each
        appended pass the text after the last media marker must re-tokenize to itself plus exactly the appended ids
        (mtmd tokenizes that tail on its own, whatever the media expanded into); otherwise, or on U+FFFD, the walk
        stops (`retok`) and reads out at the last scored position. The walk text is never escaped: /detokenize
        renders control tokens as text and parse_special=true reads them back."""
        self.refuse_unsupported(media)
        marker = self.media_marker
        # unreachable: refuse_unsupported requires the marker for every kind
        if marker is None:
            raise ReadoutError("the upstream reports no media marker")
        payloads = tuple(m.payload for m in media)  # multimodal_data: the i-th marker maps to the i-th entry
        base = self.render_prompt(state, q, context=context, media=media)
        tail = base[base.rfind(marker) + len(marker):]
        tail_ids: list[int] = []
        passes: list[tuple[tuple[int, ...], MMResult]] = []
        k = self._k(q)

        def dist_fn(cur: tuple[int, ...]) -> TokenLogprobs:
            text = base
            if cur:
                appended = self.scorer.detokenize(cur)
                if "�" in appended:
                    raise _RetokStop
                if not tail_ids:
                    tail_ids.extend(self.scorer.tokenize(tail, add_special=False, parse_special=True))
                if self.scorer.tokenize(tail + appended, add_special=False, parse_special=True) != tail_ids + list(cur):
                    raise _RetokStop
                text = base + appended
            r = self.scorer.next_logprobs_mm(text, payloads, top_k=k, id_slot=id_slot,
                                             delimiter=self.user_delimiter)
            passes.append((cur, r))
            return r.logprobs

        return self._walk(q, dist_fn, (), passes, kind=kind, prompt_tokens=None, offsets=offsets, bias=bias,
                          temperature=temperature)

    def _answer_context(self, state: str, q: Question, id_slot: int | None, context: str, *,
                        offsets: Sequence[float] | None = None, bias: str | None = None,
                        temperature: float | None = None) -> Readout:
        """A text request with a `context`: the token-array walk, but cached and delimited, so the system
        turn holding the context is reused across requests."""
        prompt = self.render_prompt(state, q, context=context)
        prefix = tuple(self.scorer.tokenize(prompt, add_special=True))
        passes: list[tuple[tuple[int, ...], MMResult]] = []
        k = self._k(q)

        def dist_fn(cur: tuple[int, ...]) -> TokenLogprobs:
            r = self.scorer.next_logprobs_cached(cur, top_k=k, id_slot=id_slot, delimiter=self.user_delimiter)
            passes.append((cur, r))
            return r.logprobs

        return self._walk(q, self._first_pass_recorded(dist_fn, id_slot, len(prefix)), prefix, passes, kind="context",
                          prompt_tokens=len(prefix), offsets=offsets, bias=bias, temperature=temperature)

    def _first_pass_recorded(self, dist_fn: Callable[[tuple[int, ...]], TokenLogprobs], id_slot: int | None,
                             n_tokens: int) -> Callable[[tuple[int, ...]], TokenLogprobs]:
        """`dist_fn` with its FIRST pass, a prefill of `n_tokens`, recorded in the estimators' dispatch
        record while it runs, so a chunk on another reserved slot counts it in S_known. Unchanged without estimators:
        the upstream calls are the same either way."""
        est = self.estimators
        if est is None or id_slot is None:
            return dist_fn
        first = [True]

        def recorded(cur: tuple[int, ...]) -> TokenLogprobs:
            if not first[0]:
                return dist_fn(cur)
            first[0] = False
            token = est.dispatch_start(id_slot, n_tokens * est.prefill_rate())
            try:
                return dist_fn(cur)
            finally:
                est.dispatch_end(token)
        return recorded

    # ---- think budgets: the deliberate readout ---------------------------------------

    def _on_prompt(self, state: str, q: Question, context: str) -> tuple[str, tuple[int, ...], float]:
        """(the thinking-on prompt with any synthesised opening, its tokens, the ms it took)."""
        if self.think is None:
            raise ReadoutError("this model's chat template has no thinking mode callo can read")
        t0 = time.perf_counter()
        rendered = self.template.render(messages(state, q, self.variant, context=context), thinking=True)
        try:
            on = check_on_prompt(rendered, self.think)
        except ValueError as e:
            raise ReadoutError(f"the chat template of {self.model_name} (its thinking-on prompt) lost its frame: "
                               f"{e}") from None
        tokens = tuple(self.scorer.tokenize(on, add_special=True, parse_special=True))
        return on, tokens, 1000 * (time.perf_counter() - t0)

    def trace_sampling(self) -> dict[str, Any]:
        """The pinned trace sampler a request runs with: `think_sampling` as serve resolved it, else think.py's
        default policy (`model`, with no GGUF keys: callo's 0.6 / 0.95 / 20 / 0.0 and the neutral pins). Never an
        empty dict, which would leave every sampler field to llama-server's flags. A capable engine (`think` set) that
        falls back says so once: serve resolves `think_sampling` per GGUF, so the fallback there means a
        caller skipped that and the model thinks with callo's sampler instead of its own."""
        if self.think_sampling:
            return self.think_sampling
        if self.think is not None and not self._default_sampling_warned:
            self._default_sampling_warned = True
            log.warning("think: no trace sampler was resolved for %s (Engine.think_sampling unset): deliberate "
                        "readouts use callo's default sampler (%s), not the GGUF's general.sampling.*",
                        self.model_name, "/".join(str(default_trace_sampling()[k]) for k in
                                                  ("temperature", "top_p", "top_k", "min_p")))
        return default_trace_sampling()

    def plan_deliberate(self, state: str, q: Question, *, context: str = "") -> int:
        """Plan mode: the length of the question's thinking-on prompt in tokens, with no generation."""
        return len(self._on_prompt(state, q, context)[1])

    def answer_deliberate(self, state: str, q: Question, budgets: Sequence[int], *, id_slot: int | None = None,
                          context: str = "", seed: int, sampling: Mapping[str, Any], temperature: float | None = None,
                          timeout: float | None = None, cancel: threading.Event | None = None,
                          cuts_from: Callable[[Trace], Iterable[int]] | None = None,
                          pre_cut: bool = False) -> Deliberate:
        """One generation at max(budgets), then the answer slot read at n_gen first and at each smaller cut in
        ascending order. `budgets` is ascending and positive (serve passes one, calibrate's probe several).
        `temperature` None is the engine's; `timeout` None is --s1-timeout + 0.1 s x max(budgets). The context
        check runs before generating; a truncated trace is ThinkTooLarge too.

        Think deadlines, calibrate's probe only: `cuts_from(trace)` names more cuts (the deadline
        points' k_D, the convergence probes), read with the budgets' in ONE ascending list, so a cut shared with a
        budget is read once; `Deliberate.by_k` holds every read by k. `pre_cut` (the reserve pre-pass): a natural
        trace is first read at n_gen - 1 with a forced close, a prefix that extends the slot's cache exactly (it holds
        on + gen[:-1]), before its n_gen readout."""
        budgets = [int(b) for b in budgets]
        if not budgets or budgets != sorted(budgets) or budgets[0] < 1 or len(set(budgets)) != len(budgets):
            raise ValueError(f"budgets must be ascending, distinct and positive, got {budgets}")
        self.check_coverage(q)                      # never generate a trace for a refused question
        if cancel is not None and cancel.is_set():
            raise Cancelled("the client left before the deliberate generation")
        bmax = budgets[-1]
        on, on_tokens, frame_ms = self._on_prompt(state, q, context)
        if self.n_ctx is not None and context_need(len(on_tokens), bmax) > self.n_ctx:
            raise ThinkTooLarge(f"the prompt ({len(on_tokens)} tokens) plus a {bmax}-token budget exceeds the slot "
                                f"context ({self.n_ctx}); lower think_budget or raise --ctx")
        frame = self.think
        assert frame is not None
        gen: GenResult = self.scorer.generate(
            on_tokens, n_predict=bmax, stop=[frame.tag], sampling=sampling, seed=seed, id_slot=id_slot,
            delimiter=self.user_delimiter, timeout=self.s1_timeout + 0.1 * bmax if timeout is None else timeout,
            cancel=cancel)
        if gen.truncated:
            raise ThinkTooLarge(f"llama-server truncated the trace at {gen.tokens_predicted} tokens: the prompt "
                                f"({len(on_tokens)} tokens) plus the budget does not fit the slot context; lower "
                                f"think_budget or raise --ctx")
        n_gen = len(gen.tokens)                     # the cut arithmetic counts the tokens callo was given ...
        if gen.tokens_predicted != n_gen:           # ... and says so (once: a server quirk repeats on every trace)
            (log.debug if self._count_mismatch_warned else log.warning)(
                "deliberate generation: %d tokens returned but tokens_predicted %d; the readout uses %d%s", n_gen,
                gen.tokens_predicted, n_gen, "" if self._count_mismatch_warned else " (logged once; later ones at debug)")
            self._count_mismatch_warned = True
        trace = Trace(gen.tokens, n_gen, gen.close, gen.truncated, gen.wall_ms, gen.prompt_ms, gen.prompt_n,
                      gen.cache_n, gen.predicted_ms, frame_ms, gen.content, len(on_tokens))
        kind = "context" if context else "plain"
        extra = {int(k) for k in (cuts_from(trace) if cuts_from is not None else ()) if 1 <= int(k) <= n_gen}
        cuts = sorted({min(b, n_gen) for b in budgets} | extra)
        ks = [n_gen, *(k for k in cuts if k < n_gen)]                       # n_gen first, then the cuts ascending
        if pre_cut and gen.close == "natural" and n_gen >= 2:               # the served-like forced read first
            ks = [n_gen - 1, *(k for k in ks if k != n_gen - 1)]
        by_k: dict[int, BudgetReadout] = {}
        for k in ks:
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client left between deliberate readouts")
            if gen.close == "eos" and k == n_gen - 1 and n_gen in by_k:
                # an eos trace is read at n_gen with its EOG replaced by the close suffix: the very prefix of the cut
                # at n_gen - 1, so that readout is reused (with its own k), not read twice
                by_k[k] = dataclasses.replace(by_k[n_gen], k=k)
                continue
            by_k[k] = self._deliberate_readout(on, on_tokens, gen, k, q, id_slot, kind=kind, temperature=temperature,
                                               cancel=cancel)
        return Deliberate(trace, {b: by_k[min(b, n_gen)] for b in budgets},
                          len({id(b.readout) for b in by_k.values()}), by_k)

    def _deliberate_readout(self, on: str, on_tokens: tuple[int, ...], gen: GenResult, k: int, q: Question,
                            id_slot: int | None, *, kind: str, temperature: float | None,
                            cancel: threading.Event | None = None) -> BudgetReadout:
        frame = self.think
        assert frame is not None
        t0 = time.perf_counter()
        prefix = readout_prefix(on_tokens, gen.tokens, gen.close, k, frame)
        fallback = prefix is None
        if prefix is None:                                                  # the documented text fallback
            prefix = tuple(self.scorer.tokenize(on + gen.content.strip() + frame.close, add_special=True,
                                                parse_special=True))
        passes: list[tuple[tuple[int, ...], MMResult]] = []
        top_k = self._k(q)

        def dist_fn(cur: tuple[int, ...]) -> TokenLogprobs:
            if cancel is not None and cancel.is_set():                      # between readout passes too
                raise Cancelled("the client left between deliberate readout passes")
            r = self.scorer.next_logprobs_cached(cur, top_k=top_k, id_slot=id_slot, delimiter=self.user_delimiter)
            passes.append((cur, r))
            return r.logprobs

        r = self._walk(q, dist_fn, prefix, passes, kind=kind, prompt_tokens=len(prefix), bias="deliberate",
                       temperature=temperature)
        first = passes[0][1]
        return BudgetReadout(r, k, 1000 * (time.perf_counter() - t0), first.prompt_ms, first.prompt_n, first.cache_n,
                             fallback)

    # ---- think deadlines: the chunked trace ----------------------------------------------------

    def answer_chunked(self, state: str, q: Question, *, cap: int, explicit_cap: bool, seed: tuple[str, ...],
                       sampling: Mapping[str, Any], temperature_of: Callable[[str, int, int], tuple[float, bool]],
                       share_end_ms: float | None = None, request_end_ms: float | None = None,
                       reserve_ms: float = 0.0, estimators: Estimators | None = None, id_slot: int | None = None,
                       context: str = "", cancel: threading.Event | None = None, probe_every: int | None = None,
                       clock: Callable[[], float] | None = None,
                       converge: ConvergeRun | None = None) -> ChunkedReadout | None:
        """The trace as a chain of `/completion` calls on `on_tokens + gen` so far, each sized by the planner
        (`deadline.run_chunks`), then ONE readout at the cut k, except that early exit (`converge`) reuses the stop
        probe, or a probe on the deadline cut, re-tempered as the answer with no read after the close (`reused`). None:
        thinking does not fit the share, so the caller reads the question at budget 0 (nothing was sent upstream).

        - `cap`: an explicit budget (`explicit_cap`: the context check, ThinkTooLarge) or the deadline-only B*,
          clipped here to the slot context (`clip_cap`: never refused; nothing left to think in is the fallback).
        - `seed`: (digest, qid): chunk 0 uses think_seed(digest, qid), chunk i think_seed(digest, qid, str(i)).
        - `share_end_ms` (T_q), `request_end_ms` (T_req) and `clock()` (ms, default the monotonic clock) are on one clock;
          `share_end_ms` None runs without a deadline (the probe's live mode): chunks of `probe_every` or C_MAX.
        - `estimators`: the server-wide windows: each chunk's estimate comes from them and each chunk feeds
          them; chunk 0's prefill is recorded for the other slots' S_known while it runs. None (the probe's live mode
          without a deadline): no estimate is needed.
        - `temperature_of(close, k, cap)`: the rule for the readout, (T, calibrated), given the clipped cap.
        - `probe_every` K: chunk ends are aligned to multiples of K and a forced readout is read at every boundary the
          trace passes without closing (the cost pass); the final readout is still the one at the cut.
        - `converge` (early exit): probe every K tokens (chunk ends aligned as in the live mode), close `converged` at
          the first probe the rule accepts and answer with that probe re-tempered to `temperature_of('converged', k,
          cap)`. Under a deadline a due probe's cost is charged like ô (`with_probe_cost`), and a probe on the cut is
          the answer's read. It excludes `probe_every`.
        - The per-chunk timeout is min(--s1-timeout + 0.1 s x n_i, T_req - now + 5 s); `cancel` is checked before
          every chunk and polled during it. Upstream errors map as on the budget path: nothing is read from a
          partial trace."""
        self.check_coverage(q)                      # never generate a trace for a refused question
        if converge is not None and probe_every:
            raise ValueError("answer_chunked takes probe_every (the probe's live mode) or converge, not both")
        every = converge.K if converge is not None else probe_every
        if cancel is not None and cancel.is_set():
            raise Cancelled("the client left before the deliberate generation")
        clock = clock or (lambda: 1000.0 * time.monotonic())
        on, on_tokens, frame_ms = self._on_prompt(state, q, context)
        frame = self.think
        assert frame is not None
        if explicit_cap:
            if self.n_ctx is not None and context_need(len(on_tokens), cap) > self.n_ctx:
                raise ThinkTooLarge(f"the prompt ({len(on_tokens)} tokens) plus a {cap}-token budget exceeds the slot "
                                    f"context ({self.n_ctx}); lower think_budget or raise --ctx")
        else:
            cap = clip_cap(cap, self.n_ctx, len(on_tokens))
        steps: list[float] = []

        def estimate(i: int, so_far: Sequence[Chunk]) -> Estimate:
            if estimators is None:
                e = Estimate(0.0, 1e-3, 0.0)
            else:
                e = estimators.estimate(i=i, slot=id_slot, question_steps=steps,
                                        uncached=estimators.uncached(id_slot, on_tokens) if i == 0 else 0)
            if converge is None:
                return e
            left = None if share_end_ms is None else share_end_ms - reserve_ms - clock()
            return with_probe_cost(e, i=i, k=sum(len(c.tokens) for c in so_far), cap=cap, left_ms=left,
                                   every=converge.K, c_ms=converge.c_ms)

        probed = [False]                            # early exit: the chunk about to be generated follows a probe
        after: set[int] = set()                     # early exit: the chunks that followed a probe (after_probe's)

        def generate(i: int, n: int, so_far: tuple[int, ...], e: Estimate) -> Chunk:
            if every:
                n = min(n, every - len(so_far) % every)
            prompt = (*on_tokens, *so_far)
            limit = self.s1_timeout + 0.1 * n
            if request_end_ms is not None:
                limit = min(limit, (request_end_ms - clock()) / 1000.0 + 5.0)
            token = estimators.dispatch_start(id_slot, e.prefill_ms) if estimators is not None and i == 0 else None
            try:
                g = self.scorer.generate(prompt, n_predict=n, stop=[frame.tag], sampling=sampling,
                                         seed=think_seed(*seed) if i == 0 else think_seed(*seed, str(i)),
                                         id_slot=id_slot, delimiter=self.user_delimiter, timeout=max(limit, 0.001),
                                         cancel=cancel)
            finally:
                if token is not None:
                    estimators.dispatch_end(token)
            if g.truncated:
                raise ThinkTooLarge(f"llama-server truncated the trace at {len(so_far) + len(g.tokens)} tokens: the "
                                    f"prompt ({len(on_tokens)} tokens) plus the trace does not fit the slot context; "
                                    f"lower think_budget or raise --ctx")
            c = Chunk(g.tokens, g.stop_type, g.wall_ms, g.prompt_ms, g.predicted_ms, g.prompt_n, g.cache_n, n,
                      g.content, e.known_ms)
            after_probe = converge is not None and probed[0]
            if after_probe:
                after.add(i)
            if estimators is not None:
                if i == 0:      # BEFORE note_sent: it records this chunk 0's array, and the LCP is against the old one
                    estimators.add_chunk0_prompt(id_slot, prompt, g.prompt_n)
                estimators.note_sent(id_slot, prompt)
                if after_probe:     # BEFORE add_chunk: the chunk's own step stays out of the excess baseline
                    estimators.add_after_probe(c)
                step = estimators.add_chunk(i, c, after_probe=after_probe)
                if step is not None:
                    steps.append(step)
            probed[0] = False
            return c

        probes: list[BudgetReadout] = []
        kind = "context" if context else "plain"

        def check() -> None:
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client left between deliberate chunks")

        def probe(so_far: tuple[int, ...]) -> str | None:
            assert every
            if len(so_far) % every:             # a deadline-sized chunk ended off a K boundary: no probe there
                return None
            fake = GenResult(so_far, "", "limit", len(so_far), False, 0.0, None, None, None, None)
            b = self._deliberate_readout(on, on_tokens, fake, len(so_far), q, id_slot, kind=kind, temperature=1.0,
                                         cancel=cancel)
            probes.append(b)
            if converge is None:
                return None
            if estimators is not None:
                estimators.add_probe_read(b.read_ms)
            probed[0] = True
            seen = [Probe(p.k, p.readout.probs) for p in probes]
            return "converged" if converged(seen, converge.n, converge.y) else None

        run: ChunkRun = run_chunks(
            generate, estimate, cap=cap,
            share_end_ms=share_end_ms if share_end_ms is not None else clock() + 1e9, reserve_ms=reserve_ms,
            clock=clock, check=check, after=probe if every else None)
        if run.fallback:
            return None
        chunks = run.chunks
        stop = {"natural": "word", "eos": "eos"}.get(run.close or "", "limit")
        cont = chunks[1:]
        gen = GenResult(run.tokens, run.content, stop, len(run.tokens), False, sum(c.wall_ms for c in chunks),
                        chunks[0].prompt_ms, sum(float(c.predicted_ms or 0.0) for c in chunks), chunks[0].prompt_n,
                        chunks[0].cache_n)
        trace = Trace(gen.tokens, len(gen.tokens), gen.close, False, gen.wall_ms, gen.prompt_ms, gen.prompt_n,
                      gen.cache_n, gen.predicted_ms, frame_ms, gen.content, len(on_tokens), len(chunks),
                      sum(int(c.prompt_n or 0) for c in cont) if cont else None,
                      sum(float(c.prompt_ms or 0.0) for c in cont) if cont else None,
                      sum(1 for i, c in enumerate(cont, 1) if i not in after and (c.prompt_n or 0) > 1))
        T, calibrated = temperature_of(run.close or "", len(gen.tokens), cap)     # the clipped cap
        if cancel is not None and cancel.is_set():
            raise Cancelled("the client left before the deliberate readout")
        last = probes[-1] if probes else None
        reused = converge is not None and run.close in ("converged", "deadline") and last is not None \
            and last.k == len(gen.tokens)
        if reused:
            assert last is not None
            # the stop probe, or a probe on the deadline cut: its prefix (a forced cut) is the answer's, read at T = 1.
            # Never after a natural or eos close: their prefix is not a forced cut's, whatever k.
            b = dataclasses.replace(last, readout=retemper(last.readout, T))
        else:
            b = self._deliberate_readout(on, on_tokens, gen, len(gen.tokens), q, id_slot, kind=kind, temperature=T,
                                         cancel=cancel)
        return ChunkedReadout(trace, run.close or "", b, T, calibrated, chunks, tuple(probes), cap, reused)


def default_trace_sampling() -> dict[str, Any]:
    """think.py's default policy (`model`) resolved without a GGUF: the full pinned dict at callo's trace values."""
    return resolve_sampling(SamplingPolicy("model", {})).pinned


# ---- GGUF header: vocabulary (cached) and decision-type detection --------------------------------------

@dataclass(frozen=True)
class GgufMeta:
    tokens: list[str]
    tok_model: str
    architecture: str
    decision_type: str | None   # `<arch>.decision.type` when present: a trained decision model


def read_gguf_meta(path: str | Path) -> GgufMeta:
    """Header only: the key-value store, no tensor view (a tensor type gguf-py does not know never stops it)."""
    try:
        reader = ggufkv.open_kv(path)
    except Exception as e:  # noqa: BLE001 -- gguf raises assorted ValueError/struct errors on garbage
        raise InvalidInput(f"{path} is not a readable GGUF: {type(e).__name__}: {e}") from e
    tokens_field = reader.get_field("tokenizer.ggml.tokens")
    model_field = reader.get_field("tokenizer.ggml.model")
    arch_field = reader.get_field("general.architecture")
    if tokens_field is None or model_field is None or arch_field is None:
        raise InvalidInput(f"{path} has no tokenizer vocabulary in its header (tokenizer.ggml.tokens / "
                           f"tokenizer.ggml.model / general.architecture); an mmproj or a split shard rather than "
                           f"the model?")
    arch = str(arch_field.contents())
    decision = reader.get_field(f"{arch}.decision.type")
    return GgufMeta([str(t) for t in tokens_field.contents()], str(model_field.contents()), arch,
                    str(decision.contents()) if decision is not None else None)


def vocab_cache_key(path: str | Path) -> str:
    st = Path(path).stat()
    return hashlib.sha256(f"{Path(path).resolve()}\0{st.st_size}\0{st.st_mtime_ns}".encode()).hexdigest()


def _read_vocab_cache(cached: Path) -> GgufMeta | None:
    """The cached meta, or None for a miss. A corrupt, truncated or old-schema file is a miss too (it is rewritten
    from the GGUF), never a startup crash."""
    try:
        d = json.loads(cached.read_bytes())
        tokens, tok_model, arch, decision = d["tokens"], d["tok_model"], d["architecture"], d["decision_type"]
    except (OSError, ValueError, KeyError, TypeError):   # missing file, bad JSON/UTF-8, missing key, not a dict
        return None
    if not (isinstance(tokens, list) and all(isinstance(t, str) for t in tokens) and isinstance(tok_model, str)
            and isinstance(arch, str) and (decision is None or isinstance(decision, str))):
        return None
    return GgufMeta(tokens, tok_model, arch, decision)


def load_gguf_meta(path: str | Path, cache_dir: Path | None) -> GgufMeta:
    """`read_gguf_meta` through the JSON cache at `cache_dir/vocab-<sha256(path,size,mtime)>.json`."""
    if cache_dir is None:
        return read_gguf_meta(path)
    cached = Path(cache_dir) / f"vocab-{vocab_cache_key(path)}.json"
    meta = _read_vocab_cache(cached)
    if meta is not None:
        return meta
    meta = read_gguf_meta(path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")   # pid-unique: concurrent first starts never share it
    try:
        tmp.write_text(json.dumps({"tokens": meta.tokens, "tok_model": meta.tok_model,
                                   "architecture": meta.architecture, "decision_type": meta.decision_type}))
        tmp.replace(cached)                                       # os.replace: atomic, readers see old or new
    finally:
        tmp.unlink(missing_ok=True)
    return meta


# ---- startup self-test --------------------------------------------------------------------

SELF_TEST_BODY = {"state": "2 + 2 = ?",
                  "questions": {"q": {"type": "choice", "instructions": "Which is the correct result?",
                                      "criteria": {"3": None, "4": None}}}}


@dataclass(frozen=True)
class SelfTestResult:
    ok: bool
    answer: str | None
    option_mass: float | None
    top_tokens: tuple[tuple[str, float], ...]
    answer_slot_ok: bool | None
    answer_slot_detail: str
    error: str | None

    def lines(self) -> list[str]:
        out = []
        if self.error:
            out.append(f"WARNING: self-test could not run: {self.error}")
        else:
            verdict = "ok" if self.ok else "FAILED"
            out.append(f"self-test {verdict}: 2 + 2 = ? -> answer={self.answer} option mass={self.option_mass:.3f}")
            if self.option_mass is not None and self.option_mass < 0.5:
                out.append("WARNING: option mass below 0.5: the scoring position may not be the answer slot")
            if not self.ok:
                shown = ", ".join(f"{tok!r} {lp:.2f}" for tok, lp in self.top_tokens)
                out.append(f"WARNING: top-5 raw next tokens at the answer slot: {shown}")
        if self.answer_slot_ok is False:
            out.append(f"WARNING: answer slot check: {self.answer_slot_detail}")
        elif self.answer_slot_ok:
            out.append(f"answer slot check: {self.answer_slot_detail}")
        return out


@dataclass(frozen=True)
class ThinkSelfTestResult:
    """The think self-test: SELF_TEST_BODY read once at budget 64, then the post-think
    answer-slot check against the template's render of the trace as `reasoning_content`."""
    ok: bool
    answer: str | None
    option_mass: float | None
    close: str | None
    think_tokens: int | None
    slot_ok: bool | None
    slot_detail: str
    error: str | None
    top_tokens: tuple[tuple[str, float], ...] = ()     # the 5 likeliest raw next tokens at the post-think answer slot

    def lines(self) -> list[str]:
        if self.error:
            return [f"WARNING: think self-test could not run: {self.error}"]
        out = [f"think self-test {'ok' if self.ok else 'FAILED'}: 2 + 2 = ? at budget {THINK_SELF_TEST_BUDGET} -> "
               f"answer={self.answer} option mass={self.option_mass:.3f} close={self.close} "
               f"think tokens={self.think_tokens}"]
        if self.option_mass is not None and self.option_mass < 0.5:
            out.append("WARNING: option mass below 0.5 after the trace: the post-think scoring position may not be "
                       "the answer slot")
        if not self.ok:                                 # as the self-test: a FAILED verdict always warns
            shown = ", ".join(f"{tok!r} {lp:.2f}" for tok, lp in self.top_tokens)
            out.append(f"WARNING: top-5 raw next tokens at the post-think answer slot: {shown or 'none'} (deliberate "
                       f"readouts on this model may be unreliable)")
        if self.slot_ok is False:
            out.append(f"WARNING: post-think answer slot check: {self.slot_detail}")
        elif self.slot_ok:
            out.append(f"post-think answer slot check: {self.slot_detail}")
        return out


THINK_SELF_TEST_BUDGET = 64


def think_self_test(engine: Engine, *, id_slot: int | None = None, sampling: Mapping[str, Any] | None = None
                    ) -> ThinkSelfTestResult:
    """One deliberate readout of the built-in item with the resolved sampling, and the slot check. Warns,
    never refuses."""
    req = parse_request(SELF_TEST_BODY)
    q = req.questions[0]
    try:
        d = engine.answer_deliberate(req.state, q, [THINK_SELF_TEST_BUDGET], id_slot=id_slot,
                                     seed=think_seed("self-test", q.qid), sampling=sampling or engine.trace_sampling())
    except Exception as e:  # noqa: BLE001 -- the self-test reports, it does not raise
        return ThinkSelfTestResult(False, None, None, None, None, None, "", f"{type(e).__name__}: {e}")
    r = d.readouts[THINK_SELF_TEST_BUDGET].readout
    answer = q.keys[max(range(len(r.probs)), key=r.probs.__getitem__)]
    slot_ok: bool | None = None
    slot_detail = ""
    frame = engine.think
    try:
        assert frame is not None
        msgs = messages(req.state, q, engine.variant)
        on = check_on_prompt(engine.template.render(msgs, thinking=True), frame)
        text = on + d.trace.content.strip() + frame.close
        rendered = engine.template.render(msgs + [{"role": "assistant", "content": q.labels[0],
                                                   "reasoning_content": d.trace.content.strip()}], thinking=True)
        slot_ok, slot_detail = check_answer_slot(text, rendered, q.labels[0])
    except Exception as e:  # noqa: BLE001
        slot_detail = f"could not run: {type(e).__name__}: {e}"
    ok = answer == "4" and r.option_mass >= 0.5
    return ThinkSelfTestResult(ok, answer, r.option_mass, d.trace.close, d.readouts[THINK_SELF_TEST_BUDGET].k,
                               slot_ok, slot_detail, None,
                               tuple((engine.sets.token_str(tid), lp) for tid, lp in r.top_tokens))


def self_test(engine: Engine, *, id_slot: int | None = None) -> SelfTestResult:
    """Classify one trivial built-in item and run the answer-slot check. Warns loudly, never refuses to start."""
    req = parse_request(SELF_TEST_BODY)
    q = req.questions[0]
    slot_ok: bool | None = None
    slot_detail = ""
    try:
        slot_ok, slot_detail = answer_slot_check(engine.template, messages(req.state, q, engine.variant))
    except Exception as e:  # noqa: BLE001 -- the self-test reports, it does not raise
        slot_detail = f"could not run: {type(e).__name__}: {e}"
    try:
        r = engine.answer(req.state, q, id_slot=id_slot)
    except Exception as e:  # noqa: BLE001
        return SelfTestResult(False, None, None, (), slot_ok, slot_detail, f"{type(e).__name__}: {e}")
    answer = q.keys[max(range(len(r.probs)), key=r.probs.__getitem__)]
    top = tuple((engine.sets.token_str(tid), lp) for tid, lp in r.top_tokens)
    ok = answer == "4" and r.option_mass >= 0.5
    return SelfTestResult(ok, answer, r.option_mass, top, slot_ok, slot_detail, None)


# ---- building an engine, and the in-process entry point for the demo's plain arm ----------

def make_engine(upstream: str, meta: GgufMeta, *, api_key: str | None = None, temperature: float = 1.0,
                cache: bool = False, timeout: float = DEFAULT_TIMEOUT, vision: bool = False,
                media_marker: str | None = None, user_delimiter: str | None = None, model_name: str = "callo",
                position_bias: PositionBias | None = None, audio: bool = False, video: bool = False,
                sets: SurfaceSets | None = None) -> Engine:
    """The engine `callo serve` runs. `cache` is `--cache`, which governs only requests with neither media nor
    context; `timeout` is `--s1-timeout`, the ceiling of every blocking upstream call. `sets`: the caller's
    `SurfaceSets` over `meta`'s vocabulary (built here when None)."""
    post = make_http_post(api_key=api_key, timeout=timeout)
    return Engine(LlamaCppForwardScorer(upstream, cache_prompt=cache, http_post=post, api_key=api_key),
                  TemplateClient(upstream, http_post=post),
                  sets if sets is not None else SurfaceSets(meta.tokens, meta.tok_model),
                  temperature=temperature, vision=vision, media_marker=media_marker, user_delimiter=user_delimiter,
                  model_name=model_name, position_bias=position_bias, audio=audio, video=video, s1_timeout=timeout)


def local_engine(upstream: str, *, gguf: str | Path, temperature: float = 1.0, api_key: str | None = None,
                 user_delimiter: str | None = None, position_bias: str | Path | Mapping | None = None) -> Engine:
    """callo's readout without callo's proxy, for a client that talks to a plain llama-server: the
    startup steps against `upstream` (media detection, `detect_media`: vision, audio and video; then the user-turn
    delimiter, derived or `user_delimiter`). No slot probe, no routing, no self-test, so a video flag is trusted as
    /props reports it (`callo serve`'s video self-test is what turns it off without ffmpeg). Use
    `answer_request(body, id_slot=None)` for unpinned requests.

    `position_bias`: a profile path or dict. Phase 1 runs before any upstream call; phase 2 after the GGUF
    is read: a profile measured on another model, or a native decision model, raises CalloError; provenance drift
    warns on the `callo` logger."""
    from .profile import check_profile, load_profile, provenance  # noqa: PLC0415 -- profile imports engine
    pb = load_profile(position_bias) if position_bias is not None else None
    base = upstream.rstrip("/")
    info = discover(base, http_get=make_http_get(api_key=api_key))
    vi = detect_media(info.props)
    log.info("local engine on %s: vision %s, audio %s, video %s%s", base, *("on" if f else "off" for f in
             (vi.vision, vi.audio, vi.video)), "" if vi.vision or vi.audio or vi.video else f" ({vi.note})")
    meta = load_gguf_meta(gguf, default_cache_dir())
    if pb is not None and meta.decision_type is not None:
        check_profile(pb, model_key="", native=True, provenance=None)        # raises: there is no callo readout
    post = make_http_post(api_key=api_key)
    delimiter, lines = _user_delimiter(TemplateClient(base, http_post=post), vision=vi.vision,
                                       media_marker=vi.media_marker, image_url=SELF_TEST_IMAGE_URL,
                                       pinned=user_delimiter, tokenize=LlamaCppForwardScorer(base, http_post=post).tokenize)
    for line in lines:
        (log.warning if line.startswith("WARNING") else log.info)(line)
    sets = SurfaceSets(meta.tokens, meta.tok_model)
    if pb is not None:
        prov = provenance(chat_template=info.chat_template, build_info=info.build_info, mmproj=None,
                          user_delimiter=delimiter, cache_prompt=False,
                          wide_two_pass=sets.wide_readable() > sets.wide_coverage())
        for line in check_profile(pb, model_key=load_model_key(gguf, default_cache_dir()), native=False,
                                  provenance=prov):
            log.warning(line)
    model_name = info.props.get("model_alias") or Path(gguf).stem
    engine = make_engine(base, meta, api_key=api_key, temperature=temperature, vision=vi.vision,
                         media_marker=vi.media_marker, user_delimiter=delimiter, model_name=model_name,
                         position_bias=pb, audio=vi.audio, video=vi.video, sets=sets)
    # no frame is detected here (that, the sampler and the think self-test are `callo serve`'s startup), so a
    # positive think_budget is refused by name instead of reading as "this model cannot think"
    engine.think_reason = "local_engine serves budget 0 only (use callo serve for deliberate readouts)"
    return engine
