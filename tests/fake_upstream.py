"""A stdlib llama-server stand-in for the proxy, discovery, systemone and CLI tests.

Records every request, serves `/health`, `/props`, `/slots`, `/tokenize`, `/detokenize`, `/apply-template`,
`/completion` and SSE chat completions with canned bodies, and echoes anything else back as JSON so a test can see
exactly what arrived. `routes[path]` overrides a path; a handler can `wait_or_abort(seconds)` to notice the proxy
dropping the connection (which is how llama-server's own abort is triggered).

v0.3: `bias=PlantedBias(...)` answers every classification prompt from its OWN option lines: each option's logit is
a content score of its text plus log prior_n[position], with an optional top-k cut that floors
options. `/tokenize` then prefixes the ids of a prompt holding an `Options:` block with 1_000_000 + an index, so
`/completion` (which sees only ids) can find the text it was asked about. Planted mode is token-array only: an object
(multimodal) prompt, or an options block that does not match the prompt's reply line, is a 500 that says so.
`slot_save=True` stands for llama-server started with `--slot-save-path`: `POST /slots/{id}?action=erase`
empties the slot and is recorded in `erased`; without it the fake answers llama-server's own 501.

v0.2: `vision=True` reports `modalities.vision` and accepts image parts (`/apply-template` turns each into
`MEDIA_MARKER`) and object prompts on `/completion`; without it each answers llama.cpp's own error (a 500 on
`/apply-template`; on `/completion` a 400 since v0.3, where v0.2's fake answered 500). `timings` and
`completion_error` script the `/completion` reply. An object prompt carrying the vision self-test's mirrored control
image gets the logprobs of ids 0 and 1 swapped: the fake "sees" the image, so the self-test (main "right", control
"left") can pass; `image_blind=True` turns that off, so every image gets the same answer and the self-test fails.

v0.3 media: `audio=True` / `video=True` report those modalities and accept `input_audio` /
`input_video` / `video_url` parts on `/apply-template`. Like llama.cpp (@ 1537a0a), `/apply-template` checks
each media part in order: its modality, else a 500 "<kind> input is not supported" (a std::runtime_error);
then its payload through `handle_media_error` (the inner object under the key equal to the type, `data` before
`url`): a 400 for a malformed data URL, a 500 "Invalid base64 value" for an empty or undecodable one. On
`/completion`, a `multimodal_data` entry whose bytes are audio (the mtmd sniff) or a video container (RIFF
AVI, ISO `ftyp`, EBML) needs that modality, else a 400 "Failed to load image or audio file" (tokenization
errors are invalid requests there), as does an entry that is not strict base64 or decodes to nothing;
`video_fails=True` answers that for every video too, as an upstream without ffmpeg/ffprobe does, and
`audio_fails=True` for every audio clip. Then, as mtmd_tokenize, a `prompt_string` whose media-marker count
differs from the entry count is a 400 "Failed to tokenize prompt". The statuses are llama.cpp's own,
so the tests exercise what a real server sends. The fake "sees" the audio self-test's tone and
the video self-test's red clip as it sees the vision control image (ids 0 and 1 swapped), so every self-test
can pass; `image_blind=True` blinds it to every medium.

v0.3 think budgets: `think="preopened"` (or another `THINK_TEMPLATES` key) turns the thinking
mode on. `/apply-template` then honours `chat_template_kwargs.enable_thinking` with that template, renders a
trailing assistant message as a prefill (`<think>\\n` + reasoning_content + `\\n</think>\\n\\n` + content, S1's slot
check), and `/tokenize` keeps `</think>`, `\\n\\n` and `\\n` as their own fixed ids (THINK_TAG_ID, THINK_PARA_ID,
THINK_NL_ID) so the close suffix `"\\n</think>\\n\\n"` is 3 tokens and the post-tag 1, as on a real tokenizer; a
prompt with an `Options:` block gets the `_TEXT_ID` prefix as in planted mode, so `/completion` can find its text.
A token-array `/completion` that asks for no probabilities (`n_probs` absent or 0, as callo's generate pins it) is
a GENERATION, whatever its `n_predict`: a deterministic trace from (seed, prompt text) over the ids
THINK_WORD_BASE.., with a planted natural length `think_len` (an int, or a callable of the prompt text) after which
it emits the tag (`stop_type` "word" when `"</think>"` is in `stop`; without it the answer turn runs on to an EOG),
`limit` at `n_predict`, `think_eos=True` for an end-of-generation token instead of the tag, `think_multi_tag=True`
for a two-token tag (the engine's text fallback); `tokens` (the tag included, only with `return_tokens`),
`tokens_predicted`, `tokens_evaluated`, `tokens_cached`, `truncated` once prompt + generated reaches the props `n_ctx`
(llama-server with context shift off, even when that token is the close), and `timings` with `prompt_n` / `cache_n`
from the slot's CACHED TOKEN ARRAY (the longest common prefix; none with `cache_prompt` false; `think_hybrid=True`
re-prefills from the first trace token when a prompt diverges before the cache end, an approximation of a checkpoint
rollback, which really lands on an n - 4 checkpoint) and a planted `predicted_ms` of `think_ms_per_token` per token
(`think_slow_after`: tokens past it cost double, a non-linear decode for the audit). A prompt that fills `n_ctx` gets
llama-server's 400 `exceed_context_size_error` (readouts too). `think_delay_s` sleeps per generated token, with the
slot `is_processing`, and aborts, recorded in `aborted`, when the proxy closes its socket (the slot keeps what it
decoded). `think_gen_hook` (a callable) sees each generation's body before it runs, so a test can change the planted
speeds mid-run; `think_gen_prompt_n` plants a host prompt-cache hit: every generation reports that `prompt_n` and
`cache_n = len(prompt) - prompt_n` (readouts keep the real LCP).
A readout after a trace (n_predict 1, trace ids in the prompt) boosts the PLANTED ANSWER (`planted_answer(texts)`)
by `planted_gain(k)` for the k trace tokens in the prompt, so a calibrate run recovers an accuracy-vs-budget curve.

Think deadlines: a generation whose prompt already holds trace tokens is a CONTINUATION: its planted natural
length counts them, so a chunked trace closes at the same length a single generation does, and the slot-cache model
evaluates one token (the cache holds every token but the last). Each continuation records a near-end checkpoint
(`checkpoints`), and one whose slot cache ran past the prompt (a probe's close tokens) is a restore (`restores`).
`think_real_time=True` makes every think-mode task take the time its timings report: `think_task_ms` per task, the
prompt at `think_prompt_ms_per_token`, each token at `think_ms_per_token` (or `think_ms_at(seconds since start)`, a
rate schedule), plus `stall_next(ms)` (the next decode step of any generation) and, with `think_prefill_stall`, a
prompt of more than 32 tokens on one slot stalls every other generating slot's next step by that prefill's time;
steps are paced to absolute due times, so unplanned host delay is caught up while planted stalls are not; the reported `prompt_ms` / `predicted_ms` are the measured sleeps, split as llama-server splits them: the first
sampled token's step in `prompt_ms`, the other n - 1 in `predicted_ms`. `gen_log` records every generation's slot,
`n_predict`, seed, `prompt_n` and arrival time, and `arrivals` every think-mode `/completion`.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import random
import re
import select
import socket
import string
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from callo.av import RED_AVI, TONE_WAV
from callo.media import sniff_audio
from callo.vision import CONTROL_PNG

QWEN_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
MEDIA_MARKER = "<__media_fake__>"
CONTROL_B64 = base64.b64encode(CONTROL_PNG).decode()
# The payloads the fake "sees" (ids 0 and 1 swapped): the vision control, the audio tone, the red video (v0.3).
SWAPPED = frozenset({CONTROL_B64, base64.b64encode(TONE_WAV).decode(), base64.b64encode(RED_AVI).decode()})
MEDIA_PARTS = {"image_url": "image", "input_audio": "audio", "input_video": "video", "video_url": "video"}
# v0.3 think budgets: (the thinking-on tail, the thinking-off tail) of each template the fake can play.
THINK_TEMPLATES = {
    "preopened": ("<|im_start|>assistant\n<think>\n", QWEN_TAIL),                     # Qwen3.5, Sharp, MiniCPM5
    "qwen3": ("<|im_start|>assistant\n", QWEN_TAIL),                                   # the synthesised opening
    "no_think": ("<|im_start|>assistant\n", "<|im_start|>assistant\n"),                # no thinking mode at all
    "underivable": ("<|im_start|>assistant\n", "<|im_start|>bot\n<think>\n\n</think>\n\n"),   # off does not extend on
    "channel": ("<|im_start|>assistant\n<|channel>thought\n", "<|im_start|>assistant\n"),     # Gemma-like: no block
}
THINK_TAG_ID, THINK_TAG2_ID, THINK_PARA_ID, THINK_NL_ID, THINK_EOG_ID = 2001, 2004, 2002, 2003, 2
THINK_WORD_BASE = 5000                     # trace token ids: THINK_WORD_BASE + 0..99
THINK_TAG_TEXT = "</think>"
_THINK_PIECE = re.compile(r"\n\n|\n|</think>|\S+")


def planted_answer(texts: Sequence[str]) -> int:
    """The option a thinking fake converges on for this question: a hash of its option texts, except that the fake
    knows 2 + 2 = 4 (callo's built-in self-test item), as it "sees" the vision self-test's control image."""
    if "4" in texts:
        return list(texts).index("4")
    return int(hashlib.sha256("|".join(texts).encode()).hexdigest()[:8], 16) % len(texts)


def planted_gain(k: int) -> float:
    """The logits the planted answer gains after k trace tokens: linear to 300 tokens, then flat."""
    return 0.03 * min(k, 300)


def think_tokenize(content: str, *, multi_tag: bool = False) -> list[int]:
    """The thinking fake's tokenizer: words get positional ids from 100 (as the plain fake), while `</think>`,
    `\\n\\n` and `\\n` keep fixed ids, so callo's token forms behave as on a real tokenizer. `multi_tag` makes the
    tag two tokens (a template whose close tag is not one token: the engine's text fallback)."""
    ids, i = [], 0
    for piece in _THINK_PIECE.findall(content):
        if piece == "</think>":
            ids.extend([THINK_TAG_ID, THINK_TAG2_ID] if multi_tag else [THINK_TAG_ID])
        elif piece == "\n\n":
            ids.append(THINK_PARA_ID)
        elif piece == "\n":
            ids.append(THINK_NL_ID)
        else:
            ids.append(100 + i)
            i += 1
    return ids


def is_trace_token(t: int) -> bool:
    return THINK_WORD_BASE <= t < THINK_WORD_BASE + 100


def context_error(n_prompt: int, n_ctx: int) -> dict:
    """llama-server's 400 for a prompt that does not fit the slot (server-context.cpp, `send_error` with
    ERROR_TYPE_EXCEED_CONTEXT_SIZE; server-task.cpp adds n_prompt_tokens and n_ctx)."""
    return {"error": {"code": 400, "message": f"request ({n_prompt} tokens) exceeds the available context size "
                                              f"({n_ctx} tokens), try increasing it",
                      "type": "exceed_context_size_error", "n_prompt_tokens": n_prompt, "n_ctx": n_ctx}}


def _lcp(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _media_kind(p) -> str | None:
    """The kind of a media content part, or None (also for a non-string "type")."""
    t = p.get("type") if isinstance(p, dict) else None
    return MEDIA_PARTS.get(t) if isinstance(t, str) else None


def _is_video(raw: bytes) -> bool:
    """A video container the fake recognises: RIFF AVI, ISO base media (`ftyp` at 4), EBML (Matroska / WebM)."""
    return (raw[:4] == b"RIFF" and raw[8:12] == b"AVI ") or raw[4:8] == b"ftyp" or raw[:4] == b"\x1a\x45\xdf\xa3"


def chat_part_payload(p: dict) -> str:
    """The string llama.cpp hands `handle_media` for one media content part (server-common.cpp:1151-1201 @ 1537a0a):
    the inner object under the key equal to the type; for image_url its `url`, otherwise `data`, falling back to `url`
    when `data` is missing, null or not a string (`json_value`); "" when there is none."""
    t = p.get("type")
    inner = p.get(t) if isinstance(t, str) else None
    inner = inner if isinstance(inner, dict) else {}
    v = inner.get("url") if t == "image_url" else inner["data"] if isinstance(inner.get("data"), str) else inner.get("url")
    return v if isinstance(v, str) else ""


_B64_RUN = re.compile(r"[A-Za-z0-9+/]*")


def handle_media_error(url: str) -> tuple[int, str] | None:
    """llama.cpp's `handle_media` (server-common.cpp:1079-1146 @ 1537a0a) as /apply-template runs it: the (status,
    message) it fails with for this part value, or None when it accepts it. `std::invalid_argument` is a 400 and
    `std::runtime_error` a 500. http(s):// and file:// are not modelled: callo never sends them.
    llama.cpp tests the `http` prefix first, so a raw base64 value that starts with "http" is fetched as a
    URL there, not decoded. Raw base64 goes through llama.cpp's own `base64_decode` (server-common.cpp:228), which
    decodes only the leading run of [A-Za-z0-9+/] (stopping at '=' or any other byte) into run // 4 * 3 +
    max(run % 4 - 1, 0) bytes: non-empty, so accepted, exactly when that run is 2 characters or more ("AA", "AAAAA"
    and "AA!!" pass, "A" and "!AAAA" are "Invalid base64 value"). Python's lenient b64decode differs both ways."""
    if url.startswith("data:"):
        parts = url.split(",")
        if len(parts) != 2:
            return 400, "Invalid uri-encoded base64 value"
        if not parts[0].startswith(("data:image/", "data:video/", "data:audio/")):
            return 400, f"Invalid uri format: {parts[0]}"
        if not parts[0].endswith("base64"):
            return 400, "uri must be base64 encoded"
        return None
    return None if len(_B64_RUN.match(url).group()) >= 2 else (500, "Invalid base64 value")


def _content(content) -> str:
    """llama.cpp's content-part join: media parts become the media marker; text parts are joined with a newline,
    with no newline next to a marker (common/chat.cpp concat_content_parts)."""
    if isinstance(content, str):
        return content
    out, last_marker = "", False
    for p in content:
        if _media_kind(p) is not None:
            out, last_marker = out + MEDIA_MARKER, True
        else:
            out, last_marker = out + ("\n" if out and not last_marker else "") + p.get("text", ""), False
    return out


OPTION_MASS = 0.97                 # the letters' share of a planted dump; three fillers carry the rest
_FILLERS = ({"id": 54, "token": "x", "logprob": math.log(0.015)}, {"id": 53, "token": "Ġthe", "logprob": math.log(0.01)},
            {"id": 52, "token": "**", "logprob": math.log(0.005)})
_TEXT_ID = 1_000_000
_OPTION = re.compile(r"^([A-Z])\. (.*)$")
_REPLY = re.compile(r"^Reply with one letter: ([A-Z](?:, [A-Z])*)\.", re.MULTILINE)


def parse_options(prompt: str) -> list[str] | None:
    """The option texts of the LAST `Options:` block of a rendered prompt, in order; None without one. The letters
    parsed must be the ones the prompt's `Reply with one letter: ...` line lists (callo's prompt.user_content writes
    both): an option text with a newline would end the block early and plant over the wrong n, so a mismatch raises
    ValueError instead."""
    at = prompt.rfind("Options:\n")
    if at < 0:
        return None
    letters, texts = [], []
    for line in prompt[at + len("Options:\n"):].split("\n"):
        m = _OPTION.match(line)
        if m is None:
            break
        letters.append(m.group(1))
        texts.append(m.group(2))
    reply = _REPLY.search(prompt, at)
    if reply is None:
        raise ValueError("planted mode: the Options: block has no 'Reply with one letter' line after it")
    listed = reply.group(1).split(", ")
    if letters != listed:
        raise ValueError(f"planted mode: parsed {len(letters)} option(s) {letters} but the prompt lists "
                         f"{len(listed)} {listed}")
    return texts or None


@dataclass
class PlantedBias:
    """logit_k = score(text_k) + log prior_n[k]. `score` is a deterministic hash of the text in [-scale, scale]
    unless `content` overrides it, so a question's options keep their scores under every rotation: PriDe's
    factorisation, exactly. A fit measures these readouts' mean probability per position, which points
    the way `priors` does but sits closer to uniform; a uniform `priors` gives exactly uniform question rows."""
    priors: dict[int, list[float]] = field(default_factory=dict)
    scale: float = 1.5
    top_k: int | None = None
    content: dict[str, float] = field(default_factory=dict)

    def score(self, text: str) -> float:
        if text in self.content:
            return self.content[text]
        u = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return (2 * u - 1) * self.scale

    def logits(self, texts: Sequence[str]) -> list[float]:
        n = len(texts)
        prior = self.priors.get(n) or [1.0 / n] * n
        return [self.score(t) + math.log(p) for t, p in zip(texts, prior, strict=True)]

    def probs(self, texts: Sequence[str]) -> list[float]:
        z = self.logits(texts)
        top = max(z)
        e = [math.exp(v - top) for v in z]
        return [x / math.fsum(e) for x in e]

    def dump(self, texts: Sequence[str], *, gain: tuple[int, float] | None = None) -> list[dict]:
        """The top-logprob reply; `gain=(k, g)` adds g to option k's logit first (the think fake's planted answer)."""
        z = self.logits(texts)
        if gain is not None:
            z[gain[0]] += gain[1]
        top = max(z)
        lse = top + math.log(math.fsum(math.exp(v - top) for v in z))
        out = [{"id": k, "token": string.ascii_uppercase[k], "logprob": v - lse + math.log(OPTION_MASS)}
               for k, v in enumerate(z)]
        out += [dict(f) for f in _FILLERS]
        out.sort(key=lambda e: -e["logprob"])
        return out[: self.top_k] if self.top_k else out


def _error(code: int, message: str) -> dict:
    return {"error": {"code": code, "message": message,
                      "type": "server_error" if code >= 500 else "invalid_request_error"}}


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def json(self):
        try:
            return json.loads(self.body)
        except ValueError:
            return None


class FakeUpstream:
    def __init__(self, *, total_slots: int = 4, clears_idle: bool = False, api_key: str | None = None,
                 endpoint_slots: bool = True, vision: bool = False, image_blind: bool = False,
                 slot_save: bool = False, bias: PlantedBias | None = None, audio: bool = False, video: bool = False,
                 video_fails: bool = False, audio_fails: bool = False, think: str | None = None,
                 think_len: int | Callable[[str], int] | None = None, port: int = 0):
        self.requests: list[Recorded] = []
        self.slot_save, self.erased = slot_save, []
        if think is not None and think not in THINK_TEMPLATES:
            raise ValueError(f"think must be one of {sorted(THINK_TEMPLATES)}, got {think!r}")
        self.think = think
        self.think_len = think_len if think_len is not None else (
            lambda text: 20 + int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 680)
        self.think_eos = self.think_multi_tag = self.think_hybrid = False
        self.think_ms_per_token, self.think_prompt_ms_per_token, self.think_slow_after = 10.0, 0.5, None
        self.think_delay_s = 0.0
        self.think_gen_hook: Callable[[dict], None] | None = None     # called with each generation's body (tests)
        self.think_gen_prompt_n: int | None = None    # planted: every generation reports this prompt_n (a host-cache hit)
        self.cache: dict[int, list[int]] = {}         # think mode: each slot's cached token array
        # think deadlines: real time, a rate schedule, stalls, and the records
        self.think_real_time = False
        self.think_ms_at: Callable[[float], float] | None = None
        self.think_task_ms = 0.0
        self.think_prefill_stall = False
        self.checkpoints: list[tuple[int, int]] = []          # (slot, position): one per continuation
        self.restores: list[tuple[int, int]] = []             # (slot, prompt length): a continuation past a probe
        self.gen_log: list[dict] = []                         # slot, n_predict, seed, prompt_n, t per generation
        self.arrivals: list[tuple[str, object, float]] = []   # ("generation" | "readout", slot, time.monotonic())
        self._stall_next = 0.0
        self._stalls: dict[object, float] = {}
        self._generating: set[object] = set()
        self._t0 = time.monotonic()
        self.bias = bias if bias is not None else (PlantedBias() if think else None)
        self._texts, self._text_ids = [], {}
        self.aborted: list[tuple[str, float]] = []
        self.total_slots, self.clears_idle, self.api_key, self.vision = total_slots, clears_idle, api_key, vision
        self.image_blind = image_blind
        self.audio, self.video, self.video_fails, self.audio_fails = audio, video, video_fails, audio_fails
        self.props = {"default_generation_settings": {"n_ctx": 32768, "params": {}}, "total_slots": total_slots,
                      "model_alias": "fake-9b", "model_path": "/nonexistent/fake.gguf", "endpoint_slots": endpoint_slots,
                      "chat_template": "{{ messages }}", "build_info": "b1-926862e",
                      "modalities": {"vision": vision, "video": video, "audio": audio}, "media_marker": MEDIA_MARKER}
        self.timings: dict = {"prompt_n": 3, "cache_n": 0, "prompt_ms": 1.5}
        # llama-server's readout reply (n_predict 1), as process_token builds it from the greedy PICK: a pick in
        # `eog_ids` (llama_vocab_is_eog) stops with stop_type "eos", empty content and its probabilities, always (the
        # eos stop is unconditional, server-context.cpp:2007); a pick in `partial_utf8_ids` (a byte piece that is not
        # complete UTF-8) is never add_token'd, so the reply has NO completion_probabilities (content U+FFFD,
        # stop_type "limit"); any other pick is the canned reply with stop_type "limit". `completion_probabilities[0]
        # .id` is the pick, and `tokens` holds it under return_tokens. The pick is the argmax of the RAW dump after the
        # sampler chain, which starts from the server's defaults as llama-server's does: the base logit_bias
        # (`base_logit_bias`, an `-l` bias, plus every EOG banned when `base_ignore_eos`, as --ignore-eos does at
        # startup, common.cpp:1314) unless the request sends `logit_bias` (which replaces it); a request
        # `ignore_eos: true` bans every EOG on top (server-schema.cpp:480); and a penalties sampler that excludes
        # `penalized_ids` from the pick while the chain (llama.cpp's default, or the request's `samplers`) has
        # "penalties". `ignores_pin` makes the server ignore those three request fields. The reported dump is always
        # the raw one. `readout_top(prompt, top)` plants a position's dump (None keeps it).
        self.eog_ids: set[int] = set()
        self.partial_utf8_ids: set[int] = set()
        self.base_logit_bias: list = []
        self.base_ignore_eos = False
        self.penalized_ids: set[int] = set()
        self.ignores_pin = False
        self.readout_top: Callable[[list, list[dict]], list[dict] | None] | None = None
        self.completion_error: tuple[int, str] | None = None
        # Like the real /slots (server-context.cpp), a slot that has never run a task has no `n_prompt_tokens`.
        self.slots = [{"id": i, "n_ctx": 32768, "is_processing": False}
                      for i in range(total_slots)]
        # Vocabulary ids follow tests/test_engine-style tiny GGUFs: 0..25 = A..Z, 26..51 = " A".." Z".
        # Logprobs sum to < 1 (0.82 + 0.08 + 0.05 + 0.02 + 0.01): a top-k dump never carries the whole mass.
        self.top_logprobs = [{"id": 1, "token": "B", "logprob": -0.2}, {"id": 0, "token": "A", "logprob": -2.5},
                             {"id": 27, "token": "ĠB", "logprob": -3.0}, {"id": 52, "token": "**", "logprob": -4.0},
                             {"id": 53, "token": "Ġthe", "logprob": -5.0}]
        self.routes: dict[str, Callable] = {}
        self.stream_chunks: list[bytes] = [b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n', b": ping\n\n",
                                           b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n', b"data: [DONE]\n\n"]
        self.stream_gap_s = 0.0
        self.lock = threading.Lock()
        self._srv = ThreadingHTTPServer(("127.0.0.1", port), self._handler())    # port 0: any (the menu fake binds one)
        self._srv.daemon_threads = True
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._srv.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeUpstream:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    def stall_next(self, ms: float) -> None:
        """Real time: the next decode step of any generation sleeps `ms` more (a one-step stall)."""
        with self.lock:
            self._stall_next += ms

    def set_slot(self, slot: int, **fields) -> None:
        """Script what `/slots` reports for one slot, e.g. `set_slot(2, n_prompt_tokens=300, is_processing=True)`."""
        with self.lock:
            self.slots[slot].update(fields)

    def last(self, path: str) -> Recorded:
        with self.lock:
            return next(r for r in reversed(self.requests) if r.path.split("?")[0] == path)

    def _handler(self):
        up = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _reply(self, code: int, body, headers: dict | None = None, chunked: bool = False) -> None:
                data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(data), data) if chunked else data)
                except OSError:
                    with up.lock:
                        up.aborted.append((self.path, time.monotonic()))

            def wait_or_abort(self, seconds: float) -> bool:
                """Block up to `seconds`; True (and recorded) if the proxy closed its connection first."""
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    r, _, _ = select.select([self.connection], [], [], max(0.0, min(0.02, deadline - time.monotonic())))
                    if r:
                        try:
                            data = self.connection.recv(1, socket.MSG_PEEK)
                        except OSError:
                            data = b""
                        if data == b"":
                            with up.lock:
                                up.aborted.append((self.path, time.monotonic()))
                            return True
                return False

            def _record(self) -> Recorded:
                n = int(self.headers.get("Content-Length") or 0)
                rec = Recorded(self.command, self.path, {k.lower(): v for k, v in self.headers.items()},
                               self.rfile.read(n) if n else b"")
                with up.lock:
                    up.requests.append(rec)
                return rec

            def _dispatch(self) -> None:
                rec = self._record()
                path = self.path.split("?", 1)[0]
                if up.api_key and path != "/health" and rec.headers.get("authorization") != f"Bearer {up.api_key}":
                    return self._reply(401, {"error": {"message": "Invalid API Key", "type": "authentication_error"}})
                route = up.routes.get(path)
                if route is not None:
                    return route(up, rec, self)
                if path == "/health":
                    return self._reply(200, {"status": "ok"})
                if path == "/props":
                    return self._reply(200, up.props)
                if path == "/slots":
                    if not up.props.get("endpoint_slots", True):
                        return self._reply(501, {"error": {"message": "This server does not support slots endpoint."}})
                    with up.lock:
                        return self._reply(200, up.slots)
                if path.startswith("/slots/") and self.command == "POST":
                    return self._slot_action(path)
                if path == "/tokenize":
                    content = str((rec.json or {}).get("content", ""))
                    ids = (think_tokenize(content, multi_tag=up.think_multi_tag) if up.think
                           else list(range(100, 100 + len(content.split()))))
                    if up.bias is not None and "Options:\n" in content:
                        with up.lock:                      # one id per distinct text: the same prompt, the same ids
                            if content not in up._text_ids:
                                up._text_ids[content] = len(up._texts)
                                up._texts.append(content)
                            ids = [_TEXT_ID + up._text_ids[content], *ids]
                    return self._reply(200, {"tokens": ids})
                if path == "/detokenize":
                    return self._reply(200, {"content": " ".join(f"t{t}" for t in rec.json.get("tokens", []))})
                if path == "/apply-template" and up.think:
                    return self._apply_template_thinking(rec.json)
                if path == "/apply-template":
                    msgs = rec.json["messages"]
                    allowed = {"image": up.vision, "audio": up.audio, "video": up.video}
                    for p in (p for m in msgs if isinstance(m["content"], list) for p in m["content"]):
                        kind = _media_kind(p)
                        if kind is None:
                            continue
                        if not allowed[kind]:              # server-common.cpp:1155, 1168, 1183: runtime_error, 500
                            return self._reply(500, _error(500, f"{kind} input is not supported - hint: if this is "
                                                                f"unexpected, you may need to provide the mmproj"))
                        failed = handle_media_error(chat_part_payload(p))
                        if failed is not None:
                            return self._reply(failed[0], _error(*failed))
                    prompt = "".join(f"<|im_start|>{m['role']}\n{_content(m['content'])}<|im_end|>\n"
                                     for m in msgs) + QWEN_TAIL
                    return self._reply(200, {"prompt": prompt})
                if path == "/completion":
                    return self._completion(rec)
                if path in ("/v1/chat/completions", "/chat/completions") and (rec.json or {}).get("stream"):
                    return self._stream()
                return self._reply(200, {"echo": rec.json, "raw_len": len(rec.body), "headers": rec.headers,
                                         "path": self.path, "method": self.command})

            def _slot_action(self, path: str) -> None:
                """server-context.cpp post_slots: 501 without --slot-save-path; erase empties the slot's prompt."""
                if not up.slot_save:
                    return self._reply(501, {"error": {"code": 501, "type": "not_supported_error", "message":
                                                       "This server does not support slots action. Start it with "
                                                       "`--slot-save-path`"}})
                action = self.path.split("action=", 1)[1] if "action=" in self.path else ""
                try:
                    slot = int(path.rsplit("/", 1)[1])
                except ValueError:
                    return self._reply(400, _error(400, "Invalid slot ID"))
                if action != "erase" or not 0 <= slot < up.total_slots:
                    return self._reply(400, _error(400, "Invalid action" if action != "erase" else "Invalid slot ID"))
                with up.lock:
                    n = up.slots[slot].get("n_prompt_tokens", 0)
                    up.slots[slot]["n_prompt_tokens"] = 0
                    up.erased.append(slot)
                return self._reply(200, {"id_slot": slot, "n_erased": n})

            def _apply_template_thinking(self, body: dict) -> None:
                """Think mode: honour enable_thinking with the chosen template, and render a trailing
                assistant message as llama-server does (a prefill: no <|im_end|>, no generation prompt)."""
                msgs = body["messages"]
                thinking = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking"))
                on_tail, off_tail = THINK_TEMPLATES[up.think]
                tail = msgs[-1] if msgs and msgs[-1].get("role") == "assistant" else None
                history = msgs[:-1] if tail is not None else msgs
                prompt = "".join(f"<|im_start|>{m['role']}\n{_content(m['content'])}<|im_end|>\n" for m in history)
                if tail is None:
                    return self._reply(200, {"prompt": prompt + (on_tail if thinking else off_tail)})
                reasoning = tail.get("reasoning_content")
                block = f"<think>\n{str(reasoning).strip()}\n</think>\n\n" if isinstance(reasoning, str) else ""
                return self._reply(200, {"prompt": prompt + "<|im_start|>assistant\n" + block + _content(tail["content"])})

            def _think_timings(self, prompt: list[int], slot: int | None, generated: int, *,
                               cache_prompt: bool = True) -> dict:
                """prompt_n / cache_n from the slot's cached token array (llama-server never reuses the whole prompt:
                at least one token is evaluated; with `cache_prompt` false it reuses nothing, n_past = 0), then the
                planted times."""
                cached = up.cache.get(slot, []) if slot is not None and cache_prompt else []
                lcp = max(0, min(_lcp(cached, prompt), len(prompt) - 1))
                if up.think_hybrid and lcp < len(cached) and lcp < len(prompt) - 1:
                    first_trace = next((i for i, t in enumerate(prompt) if is_trace_token(t)), lcp)
                    lcp = min(lcp, first_trace)                           # a checkpoint rollback: re-prefill the trace
                prompt_n = len(prompt) - lcp
                slow = up.think_slow_after
                return {"prompt_n": prompt_n, "cache_n": lcp, "prompt_ms": up.think_prompt_ms_per_token * prompt_n,
                        "predicted_n": generated,
                        "predicted_ms": sum(up.think_ms_per_token * (2 if slow is not None and i >= slow else 1)
                                            for i in range(generated))}

            def _prefill(self, slot: object, prompt_n: int) -> float:
                """Real time: a task's overhead and its prompt's prefill, slept; with `think_prefill_stall` a prefill of
                more than 32 tokens stalls every other generating slot's next step by its time. Returns the prompt ms."""
                time.sleep(up.think_task_ms / 1000)
                ms = prompt_n * up.think_prompt_ms_per_token
                if up.think_prefill_stall and prompt_n > 32:
                    with up.lock:
                        for other in up._generating:
                            if other != slot:
                                up._stalls[other] = up._stalls.get(other, 0.0) + ms
                t = time.monotonic()
                time.sleep(ms / 1000)
                return 1000 * (time.monotonic() - t)

            def _claim_slot(self, slot: object, n_tokens: int) -> None:
                """The `/slots` bookkeeping of a completion on `slot` (and `clears_idle`'s wipe of the others)."""
                with up.lock:
                    if isinstance(slot, int) and 0 <= slot < up.total_slots:
                        if up.clears_idle:
                            for s in up.slots:
                                if s["id"] != slot and "n_prompt_tokens" in s:
                                    s["n_prompt_tokens"] = 0
                        up.slots[slot]["n_prompt_tokens"] = n_tokens

            def _generate(self, body: dict, prompt: list[int]) -> None:
                """Think mode: a deliberate generation (a completion that asks for no probabilities), deterministic in
                (seed, prompt text). It closes with a "word" stop only when the tag's text is in `stop`; without it the
                answer turn runs on to an EOG. `tokens` only with `return_tokens`. As llama-server with context shift
                off, the i-th sampled token ends the generation as truncated once prompt + i reaches n_ctx, even when
                it is the close. The cache after it holds the prompt and every generated token but the last (never
                decoded); an abort keeps what was decoded before it."""
                n = int(body.get("n_predict", 1))
                n_ctx = int(up.props["default_generation_settings"].get("n_ctx") or 32768)
                text = up._texts[prompt[0] - _TEXT_ID] if prompt and prompt[0] >= _TEXT_ID else ""
                natural_len = up.think_len(text) if callable(up.think_len) else int(up.think_len)
                rng = random.Random(f"{body.get('seed')}:{hashlib.sha256(text.encode()).hexdigest()}")
                words = [THINK_WORD_BASE + rng.randrange(100) for _ in range(max(natural_len, n) + 2)]
                tag = [THINK_TAG_ID, THINK_TAG2_ID] if up.think_multi_tag else [THINK_TAG_ID]
                stops_at_tag = THINK_TAG_TEXT in (body.get("stop") or [])
                close = ([THINK_EOG_ID] if up.think_eos else tag if stops_at_tag
                         else [*tag, THINK_PARA_ID, THINK_EOG_ID])
                already = sum(1 for t in prompt if is_trace_token(t))      # a continuation's words so far
                natural = words[:max(0, natural_len - already)] + close
                tokens = natural[:n]
                stop_type = ("limit" if len(tokens) < len(natural)
                             else "eos" if tokens[-1] == THINK_EOG_ID else "word")
                room = n_ctx - len(prompt)
                truncated = len(tokens) >= room
                if truncated:
                    tokens = tokens[:room]
                    stop_type = "eos" if tokens[-1] == THINK_EOG_ID else "limit"
                if up.think_gen_hook is not None:
                    up.think_gen_hook(body)                             # a test may change the planted speeds here
                slot = body.get("id_slot")
                cached = list(up.cache.get(slot, [])) if isinstance(slot, int) else []
                timings = self._think_timings(prompt, slot, len(tokens), cache_prompt=bool(body.get("cache_prompt", True)))
                if up.think_gen_prompt_n is not None:                   # planted: the question's state was restored
                    timings = {**timings, "prompt_n": up.think_gen_prompt_n,
                               "cache_n": max(0, len(prompt) - up.think_gen_prompt_n),
                               "prompt_ms": up.think_prompt_ms_per_token * up.think_gen_prompt_n}
                with up.lock:
                    now = time.monotonic()
                    up.arrivals.append(("generation", slot, now))
                    up.gen_log.append({"slot": slot, "n_predict": n, "seed": body.get("seed"),
                                       "prompt_n": timings["prompt_n"], "t": now})
                    if already and timings["prompt_n"] <= 4 and isinstance(slot, int):     # a continuation
                        up.checkpoints.append((slot, len(prompt) - 1))
                        if _lcp(cached, prompt) < len(cached):      # the cache ran past the prompt: a probe's tokens
                            up.restores.append((slot, len(prompt)))
                self._claim_slot(slot, len(prompt) + n)
                own = isinstance(slot, int) and 0 <= slot < up.total_slots
                if own:
                    with up.lock:
                        up.slots[slot]["is_processing"] = True
                try:
                    if up.think_real_time:                              # the timings are what was slept
                        prompt_ms = self._prefill(slot, timings["prompt_n"])
                        with up.lock:
                            up._generating.add(slot)
                        t = time.monotonic()
                        first = t                   # as llama-server: the first sampled token is the prompt phase's
                        due = t                     # steps are paced to absolute due times: a timer's oversleep
                        #                             (~2 ms a wait on macOS) must not accumulate over a long chunk
                        try:
                            for i in range(len(tokens)):
                                ms = up.think_ms_at(time.monotonic() - up._t0) if up.think_ms_at else \
                                    up.think_ms_per_token
                                with up.lock:
                                    ms += up._stall_next + up._stalls.pop(slot, 0.0)
                                    up._stall_next = 0.0
                                due += ms / 1000
                                if self.wait_or_abort(max(0.0, due - time.monotonic())):
                                    if isinstance(slot, int):
                                        up.cache[slot] = prompt + tokens[: max(i - 1, 0)]
                                    return None
                                if i == 0:
                                    first = time.monotonic()
                        finally:
                            with up.lock:
                                up._generating.discard(slot)
                                up._stalls.pop(slot, None)      # a stall planted in the last step is not the next task's
                        # prompt_ms covers the prefill and the first token's step; predicted_ms the other n - 1 steps
                        timings = {**timings, "prompt_ms": prompt_ms + 1000 * (first - t),
                                   "predicted_ms": 1000 * (time.monotonic() - first)}
                    elif up.think_delay_s:
                        for i in range(len(tokens)):
                            if self.wait_or_abort(up.think_delay_s):  # the proxy closed the socket: aborted
                                if isinstance(slot, int):
                                    up.cache[slot] = prompt + tokens[: max(i - 1, 0)]
                                return None
                finally:
                    if own:
                        with up.lock:
                            up.slots[slot]["is_processing"] = False
                if isinstance(slot, int):
                    up.cache[slot] = prompt + tokens[:-1]
                content = " ".join(f"w{t - THINK_WORD_BASE}" for t in tokens if is_trace_token(t))
                if stop_type != "word":
                    after = {THINK_TAG_ID: "</thi" if up.think_multi_tag else THINK_TAG_TEXT, THINK_TAG2_ID: "nk>",
                             THINK_PARA_ID: "\n\n", THINK_NL_ID: "\n"}
                    content += "".join(after.get(t, "") for t in tokens if not is_trace_token(t))
                return self._reply(200, {"content": content,
                                         "tokens": tokens if body.get("return_tokens") else [],
                                         "tokens_predicted": len(tokens), "tokens_evaluated": len(prompt),
                                         "tokens_cached": len(prompt) + len(tokens) - 1, "stop_type": stop_type,
                                         "truncated": truncated, "timings": timings})

            def _completion(self, rec: Recorded) -> None:
                body = rec.json or {}
                if up.completion_error is not None:
                    code, message = up.completion_error
                    return self._reply(code, _error(code, message))
                prompt = body.get("prompt", [])
                if up.think and isinstance(prompt, list):
                    n_ctx = int(up.props["default_generation_settings"].get("n_ctx") or 32768)
                    if len(prompt) >= n_ctx:                                    # any task, readout or generation
                        return self._reply(400, context_error(len(prompt), n_ctx))
                    if not body.get("n_probs"):        # a generation asks for no probabilities (callo pins n_probs 0)
                        return self._generate(body, prompt)
                if isinstance(prompt, dict) and "multimodal_data" in prompt and not (up.vision or up.audio or up.video):
                    return self._reply(400, _error(400, "Multimodal data provided, but model does not support "
                                                        "multimodal requests."))          # llama.cpp's 400 (I-3)
                entries = prompt.get("multimodal_data", []) if isinstance(prompt, dict) else []
                # process_mtmd_prompt: every file loads before tokenizing
                for entry in entries:
                    try:
                        raw = base64.b64decode(entry, validate=True)
                    except (binascii.Error, ValueError, TypeError):
                        raw = b""                       # llama.cpp decodes nothing usable: no bitmap either
                    if (not raw or (sniff_audio(raw) and (not up.audio or up.audio_fails))
                            or (_is_video(raw) and (not up.video or up.video_fails))):
                        return self._reply(400, _error(400, "Failed to load image or audio file"))
                # mtmd_tokenize != 0; planted mode refuses every object prompt with its own 500 below
                if (entries and up.bias is None
                        and str(prompt.get("prompt_string", "")).count(MEDIA_MARKER) != len(entries)):
                    return self._reply(400, _error(400, "Failed to tokenize prompt"))
                slot = body.get("id_slot")
                self._claim_slot(slot, (len(prompt) if isinstance(prompt, list) else len(str(prompt).split())) + 1)
                if isinstance(prompt, dict):
                    n = len(str(prompt.get("prompt_string", "")).split()) + 16 * len(prompt.get("multimodal_data", []))
                else:
                    n = len(prompt) if isinstance(prompt, list) else len(str(prompt).split())
                top = up.top_logprobs
                if up.bias is not None and isinstance(prompt, dict) and not up.think:   # think mode: canned
                    return self._reply(500, _error(500, "fake upstream: planted mode is token-array only; an object "
                                                        "(multimodal) prompt would get canned logprobs"))
                if (up.bias is not None and isinstance(prompt, list) and prompt and isinstance(prompt[0], int)
                        and prompt[0] >= _TEXT_ID):
                    try:
                        texts = parse_options(up._texts[prompt[0] - _TEXT_ID])
                    except ValueError as e:
                        return self._reply(500, _error(500, f"fake upstream: {e}"))
                    if texts:
                        k = sum(1 for t in prompt if is_trace_token(t))     # think mode: trace tokens in the prompt
                        gain = (planted_answer(texts), planted_gain(k)) if up.think and k else None
                        top = up.bias.dump(texts, gain=gain)
                if not up.image_blind and isinstance(prompt, dict) and SWAPPED & set(prompt.get("multimodal_data", [])):
                    lp = {t["id"]: t["logprob"] for t in top}
                    top = [{**t, "logprob": lp.get(1 - t["id"], t["logprob"])} if t["id"] in (0, 1) else t for t in top]
                timings = dict(up.timings)
                if up.think and isinstance(prompt, list):
                    slot = body.get("id_slot")
                    timings = self._think_timings(prompt, slot, 0, cache_prompt=bool(body.get("cache_prompt", True)))
                    with up.lock:
                        up.arrivals.append(("readout", slot, time.monotonic()))
                    if up.think_real_time:
                        timings = {**timings, "prompt_ms": self._prefill(slot, timings["prompt_n"])}
                    if isinstance(slot, int):
                        up.cache[slot] = list(prompt)
                if up.readout_top is not None:
                    planted = up.readout_top(prompt, top)
                    top = top if planted is None else planted
                def asked(key, base):
                    return base if up.ignores_pin or key not in body else body[key]
                chain = asked("samplers", ["penalties", "temperature"])
                eog_bans = [[t, False] for t in sorted(up.eog_ids)]
                bias = asked("logit_bias", [*up.base_logit_bias, *(eog_bans if up.base_ignore_eos else [])])
                if asked("ignore_eos", False) is True:
                    bias = [*bias, *eog_bans]
                banned = {int(b[0]) for b in bias if b[1] is False}
                banned |= up.penalized_ids if "penalties" in chain else set()
                pick = [t for t in top if t["id"] not in banned]
                argmax = max(pick, key=lambda t: t["logprob"])["id"] if pick else None
                toks = [argmax] if body.get("return_tokens") and argmax is not None else []
                if argmax in up.partial_utf8_ids:
                    return self._reply(200, {"content": "\ufffd", "stop_type": "limit", "tokens": toks,
                                             "tokens_evaluated": n, "timings": timings})
                cp = [{"id": argmax, "top_logprobs": top}]
                if argmax in up.eog_ids:
                    return self._reply(200, {"content": "", "tokens": toks, "stop_type": "eos",
                                             "completion_probabilities": cp, "tokens_evaluated": n,
                                             "timings": timings})
                self._reply(200, {"content": "B", "tokens": toks, "stop_type": "limit", "completion_probabilities": cp,
                                  "tokens_evaluated": n, "timings": timings})

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    for c in up.stream_chunks:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(c), c))
                        self.wfile.flush()
                        if up.stream_gap_s:
                            time.sleep(up.stream_gap_s)
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    with up.lock:
                        up.aborted.append((self.path, time.monotonic()))

            do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = _dispatch

        return H
