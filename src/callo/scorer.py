"""The concrete `ForwardScorer`: HTTP to a live llama.cpp server.

Two facts, VERIFIED against a running llama-server rather than assumed, drive the implementation:

- Next-token logprobs are at `completion_probabilities[0].top_logprobs`, each `{id, token, logprob}`, in NATURAL
  LOG space and <= 0 (a top-k dump, so the mass is <= 1 up to float32 rounding, exactly what `marginalize` expects).
- Those logprobs are RAW pre-sampling softmax. The request must NOT set `post_sampling_probs`: that flag returns
  a DEGENERATE post-sampling distribution (`top_probs = 1.0` on the argmax).

`cache_prompt` is OFF by default here (the class keeps the payload unless asked): reusing a prior prompt's KV
perturbs the returned logprobs at the ~1e-2 level depending on the server's cache history. `callo serve --cache` (on
by default; `--no-cache` turns it off) asks for it, and then the plain pass also sends the
user-turn delimiter, so a hybrid model checkpoints at the user turn and reuses the system head (without it llama.cpp
checkpoints only near the end of the prompt, and a prompt that diverges earlier re-prefills whole). With `--cache`
off the plain payload is the uncached one (`cache_prompt: false`).

Image requests and `context` requests use their own passes, which ALWAYS send `cache_prompt: true` and the
user-turn delimiter: without the cache every walk pass re-encodes the frame, and a hybrid model can
reuse the system turn only through the checkpoint the delimiter creates. Image requests go to native `/completion`
with an object prompt, never `/v1/completions`, which rejects object prompts once an mmproj is loaded.
"""
from __future__ import annotations

import http.client
import json
import logging
import socket
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from .errors import (
    Cancelled,
    InvalidInput,
    ReadoutError,
    ThinkTimeout,
    ThinkTooLarge,
    UpstreamRejected,
    UpstreamUnavailable,
)
from .marginalize import TopLogprobs
from .think import SAMPLER_FIELDS

# A long-context teacher-forced prefill can take minutes; this only extends the ceiling (--s1-timeout 900 s).
DEFAULT_TIMEOUT = 900.0
# The deliberate generation's CONNECT bound (as the proxy's ProxyConfig.connect_timeout): a
# blackholed upstream is "connection refused" (503) after this, not a 504 after the whole generation timeout.
GEN_CONNECT_TIMEOUT = 10.0

HttpPost = Callable[[str, Mapping], dict]
log = logging.getLogger("callo")


def make_http_post(*, api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> HttpPost:
    """A JSON POST with an optional `Authorization: Bearer` for an upstream started with `--api-key`."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def post(url: str, payload: Mapping) -> dict:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- local server
            return json.loads(resp.read())

    return post


urllib_post: HttpPost = make_http_post()


@dataclass(frozen=True)
class MMResult:
    """One pass on the cached path and what llama-server reported about it."""
    logprobs: dict[int, float]
    tokens_evaluated: int | None     # the whole prompt, image positions included: Readout.prompt_tokens for images
    cache_n: int | None              # timings.cache_n: tokens reused from the slot's cache
    prompt_n: int | None             # timings.prompt_n: tokens evaluated by this request
    prompt_ms: float | None          # timings.prompt_ms: server-side prompt time


@dataclass(frozen=True)
class GenResult:
    """One deliberate generation and what llama-server reported about it."""
    tokens: tuple[int, ...]          # every sampled token, the close tag included (pushed before the stop search)
    content: str                     # the generated text, the stop string dropped (diagnostics only)
    stop_type: str                   # "word" (the tag), "limit" (the budget), "eos", or whatever the server said
    tokens_predicted: int
    truncated: bool                  # the context filled before the budget: never a close
    wall_ms: float                   # the /completion call's wall time, inside callo
    prompt_ms: float | None          # timings.prompt_ms
    predicted_ms: float | None       # timings.predicted_ms
    prompt_n: int | None             # timings.prompt_n
    cache_n: int | None              # timings.cache_n

    @property
    def close(self) -> str:
        """Natural | forced | eos; `truncated` is checked by the caller first. `generate` returns a "limit" stop
        only when the trace reached n_predict (the budget): one short of it (a server time limit such as
        `t_max_predict_ms`) and any stop_type callo does not know are refused there, so "forced" always means the budget
        closed the trace."""
        return {"word": "natural", "limit": "forced", "eos": "eos"}.get(self.stop_type, "forced")


class _CancelWatch(threading.Thread):
    """Polls `cancel` while a generation is in flight; when it is set, shuts the upstream socket down, which makes
    llama-server abort the task (its non-streaming handler polls the connection every second)."""

    def __init__(self, cancel: threading.Event, conn: http.client.HTTPConnection):
        super().__init__(daemon=True)
        self.cancel, self.conn, self._stop = cancel, conn, threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            if self.cancel.wait(0.1):
                sock = self.conn.sock
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                return

    def stop(self) -> None:
        self._stop.set()


_CONTEXT_TYPE = "exceed_context_size_error"
_CONTEXT_TEXT = ("exceeds the available context size", "is larger than the max context size")


def _gen_error(status: int, raw: bytes) -> Exception:
    """400 exceed_context_size_error -> ThinkTooLarge (413); another 4xx -> UpstreamRejected (500); a 5xx ->
    UpstreamUnavailable (503)."""
    try:
        err = json.loads(raw).get("error")
        msg = err.get("message") if isinstance(err, dict) else None
        etype = err.get("type") if isinstance(err, dict) else None
    except (ValueError, AttributeError):
        msg, etype = None, None
    text = msg if isinstance(msg, str) and msg else raw.decode(errors="replace").strip()[:500]
    if status == 400 and (etype == _CONTEXT_TYPE or any(s in text for s in _CONTEXT_TEXT)):
        return ThinkTooLarge(f"the prompt plus the think budget exceeds the slot context (upstream: {text}); lower "
                             f"think_budget or raise --ctx")
    if 400 <= status < 500:
        return UpstreamRejected(f"upstream HTTP {status} on the deliberate generation: {text}")
    return UpstreamUnavailable(f"upstream HTTP {status} on the deliberate generation: {text}")


def _top_logprobs(resp: Mapping, *, greedy: bool = True, reasks: int = 0) -> TopLogprobs:
    """The pass's top-k dump. `greedy`: the reply's pick is the pass's raw argmax (a temperature-0 pass on the pinned
    neutral chain, not re-asked with a ban), so a `stop_type` "eos" names the end-of-generation token the walk must
    never step past (`TopLogprobs.eog_id`, the sampled `completion_probabilities[0].id`)."""
    cp = resp.get("completion_probabilities")
    if not cp:
        raise InvalidInput(
            "completion endpoint returned no non-empty 'completion_probabilities'; n_probs may "
            f"have been dropped or the request rejected: {list(resp)[:10]}")
    top = cp[0].get("top_logprobs")
    if top is None:
        raise InvalidInput(
            f"completion_probabilities[0] has no 'top_logprobs': {list(cp[0])[:8]} "
            "(is post_sampling_probs set? that yields linear 'top_probs' instead)")
    eog = _int(cp[0].get("id")) if greedy and resp.get("stop_type") == "eos" else None
    return TopLogprobs(((int(e["id"]), float(e["logprob"])) for e in top), eog_id=eog, reasks=reasks)


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _mm_result(resp: Mapping, probs: Mapping | None = None, *, greedy: bool = True, reasks: int = 0) -> MMResult:
    """`resp` is the pass's reply (its timings and prompt size); `probs` the reply holding its probabilities, when a
    withheld pass was re-asked (`LlamaCppForwardScorer._read`)."""
    t = resp.get("timings")
    t = t if isinstance(t, dict) else {}
    ms = t.get("prompt_ms")
    return MMResult(_top_logprobs(resp if probs is None else probs, greedy=greedy, reasks=reasks),
                    _int(resp.get("tokens_evaluated")), _int(t.get("cache_n")), _int(t.get("prompt_n")),
                    float(ms) if isinstance(ms, (int, float)) and not isinstance(ms, bool) else None)


# Every readout pass pins a NEUTRAL sampler chain (llama.cpp @ 4ebdf2c). A request starts from the server's own
# sampling defaults (server-schema.cpp eval_llama_cmpl_schema), so `-l` biases, `--ignore-eos` (which acts only through
# EOG bans in the BASE logit_bias, common.cpp:1314-1319 and server-schema.cpp:480; the eos stop itself is
# unconditional, server-context.cpp:2007), penalties, DRY, XTC, dynatemp or mirostat would move the greedy pick off the
# raw argmax. The probabilities are the raw logits whatever the chain
# (populate_token_probs -> get_token_probabilities); only the pick, and so `stop_type`, depends on it. With this pin
# the chain is logit_bias (empty: the handler clears the base list) -> temperature 0 (keeps only the max logit) ->
# dist: the pick is the raw argmax. Not pinnable per request: a server-wide --grammar (the "grammar" handler cannot
# clear it; `callo serve` refuses it in launch mode) and the vocabulary's own suppress tokens; the walk checks the
# pick against the argmax for those. `return_tokens` names a withheld pick (below). Frozen: `readout_pin()` is the
# JSON-ready copy each payload carries.
READOUT_PIN: Mapping[str, Any] = MappingProxyType({
    "samplers": ("temperature",), "logit_bias": (), "ignore_eos": False, "dynatemp_range": 0.0, "mirostat": 0,
    "reasoning_budget_tokens": -1, "return_tokens": True})


def readout_pin() -> dict[str, Any]:
    """`READOUT_PIN` as a fresh request fragment: its tuples as JSON lists."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in READOUT_PIN.items()}

# llama-server withholds a greedy pick that ends in an INCOMPLETE UTF-8 sequence (a byte piece: process_token never
# add_token's it), and the pass's completion_probabilities with it: a reply with stop_type "limit" and none. The
# probabilities are the raw logits, which logit_bias does not touch (it acts in the sampler), so the pass is asked again
# with that pick (its `tokens`) banned, and so on while the next pick is withheld too. Each ban is another byte-piece
# token COMPETING at this one position (not another byte of one character); 4 cover any plausible run of them, and
# past them, or when the server ignores a ban or names no pick, the pass is a ReadoutError.
HELD_BACK_BANS = 4


class ForwardScorer(Protocol):
    def tokenize(self, text: str, *, add_special: bool = ..., parse_special: bool | None = ...) -> list[int]: ...
    def next_logprobs(self, tokens: Sequence[int], *, top_k: int, id_slot: int | None = ...,
                      delimiter: str | None = ...) -> Mapping[int, float]: ...
    # the image and context readouts (Engine._answer_media / _answer_context) call these through this type.
    def detokenize(self, tokens: Sequence[int]) -> str: ...
    def next_logprobs_mm(self, prompt_string: str, images: Sequence[str], *, top_k: int, id_slot: int | None = ...,
                         delimiter: str | None = ...) -> MMResult: ...
    def next_logprobs_cached(self, tokens: Sequence[int], *, top_k: int, id_slot: int | None = ...,
                             delimiter: str | None = ...) -> MMResult: ...
    # the deliberate generation (Engine.answer_deliberate) calls this through this type.
    def generate(self, tokens: Sequence[int], *, n_predict: int, stop: Sequence[str], sampling: Mapping[str, Any],
                 seed: int, id_slot: int | None = ..., delimiter: str | None = ..., timeout: float = ...,
                 cancel: threading.Event | None = ...) -> GenResult: ...


class LlamaCppForwardScorer:
    """A `ForwardScorer` backed by a llama.cpp server's `/tokenize` and `/completion` endpoints."""

    def __init__(self, base_url: str, *, temperature: float = 0.0, cache_prompt: bool = False,
                 http_post: HttpPost = urllib_post, api_key: str | None = None):
        self._base = base_url.rstrip("/")
        self._temperature = temperature
        self._cache_prompt = cache_prompt
        self._post = http_post
        self._api_key = api_key           # `generate` opens its own connection, so it needs the key itself
        self._greedy = temperature <= 0   # a readout's stop_type "eos" names its ARGMAX only when it sampled greedily

    def _url(self, endpoint: str) -> str:
        return f"{self._base}/{endpoint}"

    def _read(self, payload: dict) -> tuple[Mapping, Mapping, int]:
        """One readout pass: (its reply, the reply holding its probabilities, the re-asks that took). The two replies
        are one unless llama-server withheld the probabilities (`HELD_BACK_BANS`); a reply missing them for another
        reason (no stop_type "limit") is returned as it is, for `_top_logprobs` to refuse."""
        url = self._url("completion")
        first = resp = self._post(url, payload)
        if first.get("completion_probabilities") or first.get("stop_type") != "limit":
            return first, first, 0
        banned: list[int] = []
        while True:
            toks = resp.get("tokens")
            tok = _int(toks[0]) if isinstance(toks, list) and toks else None
            if tok is None:
                raise ReadoutError("llama-server withheld a readout pass's probabilities and named no token (no "
                                   f"'tokens' under return_tokens), so callo cannot ban it and ask again; banned so "
                                   f"far: {banned}")
            if tok in banned:
                raise ReadoutError(f"llama-server withheld a readout pass's probabilities and picked {tok} again "
                                   f"although it was banned (logit_bias {banned} ignored)")
            if len(banned) == HELD_BACK_BANS:
                raise ReadoutError(f"llama-server withheld a readout pass's probabilities: its greedy pick was an "
                                   f"incomplete UTF-8 piece with {banned} banned in turn, and then a new withheld "
                                   f"pick {tok}")
            banned.append(tok)
            log.debug("readout pass: llama-server withheld the probabilities of incomplete UTF-8 pick %d; asking "
                      "again with %s banned", tok, banned)
            resp = self._post(url, {**payload, "logit_bias": [[t, False] for t in banned]})
            if resp.get("completion_probabilities"):
                return first, resp, len(banned)
            if resp.get("stop_type") != "limit":
                return first, resp, len(banned)          # not withheld: _top_logprobs refuses it as it is

    def tokenize(self, text: str, *, add_special: bool = True, parse_special: bool | None = None) -> list[int]:
        payload: dict = {"content": text, "add_special": add_special}
        if parse_special is not None:      # the server's default is true (server-context.cpp post_tokenize)
            payload["parse_special"] = parse_special
        resp = self._post(self._url("tokenize"), payload)
        tokens = resp.get("tokens")
        if tokens is None:
            raise InvalidInput(f"tokenize endpoint returned no 'tokens': {list(resp)[:8]}")
        # `with_pieces` is deliberately NOT set, so entries are bare ints, not {id, piece} objects.
        return [int(t) for t in tokens]

    def next_logprobs(self, tokens: Sequence[int], *, top_k: int, id_slot: int | None = None,
                      delimiter: str | None = None) -> TopLogprobs:
        """The plain pass. `cache_prompt` is `--cache`; `delimiter` (the engine's validated user-turn delimiter)
        is sent only under `--cache`, so with the cache off the payload is the uncached one, and a None delimiter
        (none derived) leaves the cached payload as it was before the delimiter was sent."""
        _, probs, reasks = self._read(self._pass_payload(
            list(tokens), top_k=top_k, id_slot=id_slot, cache=self._cache_prompt,
            delimiter=delimiter if self._cache_prompt else None))
        return _top_logprobs(probs, greedy=self._greedy and not reasks, reasks=reasks)

    def detokenize(self, tokens: Sequence[int]) -> str:
        """`/detokenize` on the WHOLE sequence, so multi-byte pieces join. Control tokens come back as their text
        (server-context.cpp:5319-5330, `special = true`)."""
        resp = self._post(self._url("detokenize"), {"tokens": [int(t) for t in tokens]})
        content = resp.get("content")
        if not isinstance(content, str):
            raise InvalidInput(f"detokenize endpoint returned no 'content': {list(resp)[:8]}")
        return content

    def _pass_payload(self, prompt: Any, *, top_k: int, id_slot: int | None, delimiter: str | None,
                      cache: bool = True) -> dict:
        """One readout pass's `/completion` body, for every pass kind: the plain pass (`cache` = `--cache`) and the
        image / context / deliberate-readout passes (always cached). Key order is fixed. post_sampling_probs is
        intentionally absent: the default gives raw softmax logprobs. `READOUT_PIN` makes the pass's pick the raw
        argmax whatever the server's sampling flags."""
        payload: dict = {"prompt": prompt, "n_predict": 1, "n_probs": top_k, "temperature": self._temperature,
                         "cache_prompt": cache, **readout_pin()}
        if id_slot is not None:
            payload["id_slot"] = id_slot  # llama-server runs the task on that slot (server-context.cpp:4463)
        if delimiter:
            payload["message_delimiters"] = [{"role": "user", "delimiter": delimiter}]   # server-context.cpp:4498
        return payload

    def next_logprobs_mm(self, prompt_string: str, images: Sequence[str], *, top_k: int, id_slot: int | None = None,
                         delimiter: str | None = None) -> MMResult:
        """An image request: native `/completion` with an object prompt (server-common.cpp:979-1018). `images` are
        base64 payloads WITHOUT the `data:` prefix, which is what `multimodal_data` takes."""
        prompt = {"prompt_string": prompt_string, "multimodal_data": list(images)}
        first, probs, reasks = self._read(self._pass_payload(prompt, top_k=top_k, id_slot=id_slot, delimiter=delimiter))
        return _mm_result(first, probs, greedy=self._greedy and not reasks, reasks=reasks)

    def next_logprobs_cached(self, tokens: Sequence[int], *, top_k: int, id_slot: int | None = None,
                             delimiter: str | None = None) -> MMResult:
        """A text request that carries `context`: token arrays like the plain pass, but always cached and delimited,
        so the system turn holding the context is reused (spans are found on the tokenized prompt)."""
        first, probs, reasks = self._read(self._pass_payload(list(tokens), top_k=top_k, id_slot=id_slot,
                                                             delimiter=delimiter))
        return _mm_result(first, probs, greedy=self._greedy and not reasks, reasks=reasks)

    def generate(self, tokens: Sequence[int], *, n_predict: int, stop: Sequence[str], sampling: Mapping[str, Any],
                 seed: int, id_slot: int | None = None, delimiter: str | None = None, timeout: float = DEFAULT_TIMEOUT,
                 cancel: threading.Event | None = None) -> GenResult:
        """The deliberate generation: `/completion` on the token prompt with
        `n_predict`, the close tag as the stop string and a preserved token, `return_tokens`, the cache, the user-turn
        delimiter, the slot, the seed and EVERY field of the pinned `sampling` (spread FIRST and filtered to
        SAMPLER_FIELDS, so a sampling dict can never override the prompt, n_predict, stop or seed). Over
        `http.client`, not urllib, so a `cancel` event (polled by a watcher thread) can shut the socket down and
        abort the task upstream; a `cancel` already set sends nothing. The connect is bounded by GEN_CONNECT_TIMEOUT
        (an unreachable upstream is a 503), the reply by `timeout`. Raises
        ThinkTooLarge (413), UpstreamRejected (500, also for a `limit` stop short of n_predict, which only a server
        time limit gives: not the budget's trace), UpstreamUnavailable (503), ThinkTimeout (504), Cancelled, or
        ReadoutError for a 200 without `content` or `tokens`, a token that is not an integer, or a stop_type callo
        cannot read."""
        payload: dict[str, Any] = {**{k: v for k, v in sampling.items() if k in SAMPLER_FIELDS},
                                   "prompt": [int(t) for t in tokens], "n_predict": int(n_predict),
                                   "stop": list(stop), "preserved_tokens": list(stop), "return_tokens": True,
                                   "cache_prompt": True, "seed": int(seed)}
        if id_slot is not None:
            payload["id_slot"] = id_slot
        if delimiter:
            payload["message_delimiters"] = [{"role": "user", "delimiter": delimiter}]
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if cancel is not None and cancel.is_set():          # nothing sent: no task to start and abort upstream
            raise Cancelled("the client left before the deliberate generation")
        u = urlsplit(self._base)
        conn = http.client.HTTPConnection(u.hostname or "127.0.0.1", u.port, timeout=min(GEN_CONNECT_TIMEOUT, timeout))
        watch = None
        t0 = time.perf_counter()
        try:
            try:
                conn.connect()
            except OSError as e:                    # refused, unreachable, or the connect bound (a TimeoutError)
                raise UpstreamUnavailable(f"llama-server unavailable for the deliberate generation: {e}") from e
            try:
                conn.sock.settimeout(timeout)       # the connect was bounded; the reply waits for the generation
                if cancel is not None:
                    watch = _CancelWatch(cancel, conn)
                    watch.start()
                conn.request("POST", "/completion", body=json.dumps(payload).encode(), headers=headers)
                resp = conn.getresponse()
                status, raw = resp.status, resp.read()
            except TimeoutError as e:               # socket.timeout: the read timeout
                if cancel is not None and cancel.is_set():
                    raise Cancelled("the client left during the deliberate generation") from e
                raise ThinkTimeout(f"the deliberate generation exceeded its {timeout:g} s timeout") from e
            except (OSError, http.client.HTTPException) as e:
                if cancel is not None and cancel.is_set():
                    raise Cancelled("the client left during the deliberate generation") from e
                raise UpstreamUnavailable(f"llama-server unavailable during the deliberate generation: {e}") from e
        finally:
            if watch is not None:
                watch.stop()
            conn.close()
        wall_ms = 1000 * (time.perf_counter() - t0)
        if cancel is not None and cancel.is_set():
            raise Cancelled("the client left during the deliberate generation")
        if status != 200:
            raise _gen_error(status, raw)
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise UpstreamUnavailable(f"llama-server sent an unreadable generation reply: {e}") from e
        if not isinstance(data, dict):
            raise ReadoutError(f"the deliberate generation reply is not an object: {type(data).__name__}")
        content, toks = data.get("content"), data.get("tokens")
        if not isinstance(content, str) or not isinstance(toks, list):
            raise ReadoutError(f"the deliberate generation reply has no content or tokens: {list(data)[:8]}")
        t = data.get("timings") if isinstance(data.get("timings"), dict) else {}

        def ms(key: str) -> float | None:
            v = t.get(key)
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        bad = [x for x in toks if _int(x) is None]
        if bad:
            raise ReadoutError(f"the deliberate generation reply has a token that is not an integer: {bad[0]!r}")
        ids = tuple(int(x) for x in toks)
        n = _int(data.get("tokens_predicted"))
        stop_type, truncated = str(data.get("stop_type") or ""), bool(data.get("truncated"))
        if not truncated:                           # truncated is the caller's 413, checked first
            if stop_type not in ("word", "limit", "eos"):
                raise ReadoutError(f"the deliberate generation stopped with stop_type {stop_type!r}, which callo "
                                   f"cannot read as a close")
            if stop_type == "limit" and len(ids) < int(n_predict):
                raise UpstreamRejected(f"llama-server stopped the deliberate generation at {len(ids)} of {n_predict} "
                                       f"tokens with stop_type limit (a server time limit such as t_max_predict_ms?): "
                                       f"the trace is not the budget's")
        return GenResult(ids, content, stop_type, n if n is not None else len(ids), truncated,
                         wall_ms, ms("prompt_ms"), ms("predicted_ms"), _int(t.get("prompt_n")), _int(t.get("cache_n")))


def make_dist_fn(scorer: ForwardScorer, *, top_k: int, id_slot: int | None = None, delimiter: str | None = None):
    """Adapt the seam to `divergence_walk`'s `dist_fn(prefix) -> token logprobs` contract. The walk hands back
    the FULL prefix each time it appends a shared token: one teacher-forced forward pass per call, at most
    `max_depth + 1` per question. `id_slot` rides on every one of them (every engine request pinned), and
    so does `delimiter`, which the scorer sends only under `--cache`."""
    def dist_fn(prefix: tuple[int, ...]) -> Mapping[int, float]:
        return scorer.next_logprobs(prefix, top_k=top_k, id_slot=id_slot, delimiter=delimiter)
    return dist_fn
