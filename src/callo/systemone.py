"""Parse and validate a TypeSafe System One request; build the answer bodies.

Mirrors Kev's pydantic models (`kev/api.py` @ 6605a5a5) in stdlib: a request Kev rejects is rejected here
(`InvalidRequest` -> HTTP 422). Kev allows up to 255 options.

Labels are letters A-Z up to 26 options. Above that (27..255), every option takes a
fixed two-letter label from `WIDE_LABELS`: a label that is ONE token bare and after a space on the vocabularies the
table was built from (checked per model at startup: `wide_coverage`; a label that is not, on a shorter vocabulary, is
read in two passes, first letter then second letter). Two-letter labels need partial-label accounting, because the
marginaliser only collects tokens whose WHOLE decode normalises to the label: the token `A` (the first half of `AB`) is
invisible to it, and `divergence_walk`'s proof that it cannot step past the answer assumes a dominant non-option token
is answer-neutral, which a label's own first letter is not. So a wide readout passes the first-letter tokens to the
walk as `partial` (they stop it, never count as option mass), credits them to their labels by split passes (at most
two, plus one per first letter that leads a label the vocabulary does not carry as one token), and reports what it
could not read as `unread_mass`. Narrow questions are untouched.

`digest` is `kev.suite.record_digest` of the body minus `model`, which is exactly what Kev's runner writes as
`request_sha256` in predictions.jsonl, so the shim's diagnostics join the runner's rows without guessing.
`seed_digest` is the same hash without `think_converge` and `think_converge_gate` too: it seeds think traces, so early
exit and its gate do not reseed them. An early-exit trace is drawn K-chunked, so it matches a one-shot budget trace only within its first chunk.

callo mirrors llama.cpp's native image extension (`server-decision.cpp:202-262`): a top-level `images`
array of data URLs, and `image_url` parts inside a `state` that is a list of chat messages or `{"messages": [...]}`.
The parts are taken OUT of the state before it is rendered, so no base64 ever reaches the prompt text. `context`
is a callo extension: text for the cached system turn.

Audio and video are callo extensions: top-level `audio` / `videos` arrays of data URLs,
and `input_audio`, `input_video` and `video_url` parts in a messages state, shaped as llama.cpp reads them
(server-common.cpp:1151-1201: the inner object sits under the key equal to the part type, and its `data` wins over
its `url`). Every item becomes a `MediaItem` in contract order: `images`, then `audio`, then `videos`, then the state's
parts in document order, at most `MAX_MEDIA` in all. The bytes must agree with the declared kind (`media.bytes_error`).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .deadline import MAX_DEADLINE_MS
from .media import bytes_error, sniff_audio
from .render import choice_confidence, option_text, question_keys, render, score_confidence
from .think import MAX_BUDGET

MAX_OPTIONS = 255
MAX_LETTERS = 26
MAX_MEDIA = 8                # images, audio and videos together; native's 8-image cap is implied
MAX_THINK_BUDGET = MAX_BUDGET    # the hard cap of a request's think_budget: think.py's
THINK_BUDGET_ERROR = f"think_budget must be an integer from 0 to {MAX_THINK_BUDGET}"
THINK_DEADLINE_ERROR = f"think_deadline_ms must be an integer from 0 to {MAX_DEADLINE_MS}"
THINK_CONVERGE_ERROR = "think_converge must be a number from 0 (off) to below 1"
THINK_CONVERGE_GATE_ERROR = "think_converge_gate must be true, false or null"
SEED_FREE = ("think_converge", "think_converge_gate")    # the fields a trace's seed ignores: early exit and its gate
MAX_IMAGES = MAX_MEDIA       # the name (DECISION_MAX_IMAGES, server-decision.cpp:202), kept for importers
QUESTION_TYPES = ("noul", "choice", "score")
_B64 = ";base64,"
_FIELDS = (("images", "image"), ("audio", "audio"), ("videos", "video"))     # top-level field -> kind, contract order
_PARTS = {"input_audio": "audio", "input_video": "video", "video_url": "video"}   # state part type -> kind
_ANY_FAMILY = ("data:image/", "data:video/", "data:audio/")    # handle_media's whitelist for a part's data URL
_NOUN = {"image": "an image", "audio": "an audio clip", "video": "a video"}
# Content-part types llama.cpp itself refuses ("unsupported content[].type"): a 422, never a silent flatten of a
# multi-MB data URL into prompt text.
_UNSUPPORTED_MEDIA = frozenset({"audio_url", "video"})
TOO_MANY_MEDIA = f"too many media items, the maximum is {MAX_MEDIA} (images, audio and videos together)"

# The wide-choice label table, built ONCE offline and never recomputed at runtime: take AA..ZZ in order;
# keep each pair that is a single token both bare and space-prefixed in the qwen35 (Tiel / Qwen3.5 / Qwen3.8), gemma4
# (Gemma-4-E4B) and MiniCPM5-2B vocabularies; drop the two-letter English words (AD AH AI ... US WE: a word brings a
# lexical prior a letter does not); keep the first 255. Option i of a question with more than 26 options is labelled
# WIDE_LABELS[i]; a model's `wide_coverage` is how many leading entries its own vocabulary carries, and the
# engine reads the rest in two passes.
WIDE_LABELS: tuple[str, ...] = tuple("""
AA AB AC AE AF AG AJ AK AL AO AP AR AU AV AW AZ BA BB BC BD BF BG BH BI BJ BK BL BM BN BO BP BR BS BT BU BV BW CA CB
CC CD CE CF CG CH CI CK CL CM CN CO CP CR CS CT CU CV CW CX CY DA DB DC DD DE DF DG DH DI DJ DK DL DM DN DP DR DS DT
DU DV DW DX DY EA EB EC ED EE EF EG EH EI EL EM EN EO EP EQ ER ES ET EU EV EW EX FA FB FC FD FE FF FG FH FI FK FL FM
FN FO FP FR FS FT FU FW FX FY GA GB GC GD GE GF GG GH GI GL GM GN GP GR GS GT GU GV GW HB HC HD HF HG HH HK HL HM HN
HO HP HQ HR HS HT HU HV HW HY IA IB IC ID IE IG IH II IK IL IM IO IP IQ IR IU IV IX JA JB JC JD JE JJ JK JM JO JP JR
JS JT JU KA KB KC KD KE KF KG KH KI KK KL KM KN KO KP KR KS KT KV KW KY LA LB LC LD LE LF LG LI LL LM LN LO LP LR LS
LT LU LV LY MA MB MC MD MF MG MH MI MJ MK ML MM MN MO MP MR MS
""".split())
WIDE_LABELS_SHA256 = "039130c93c582e36143ed58c31fd68b5de1c18db83c3817ff97ca51ecf3ab50f"   # sha256(",".join(...))
# the 422 for a wide question past what the vocabulary can read at all: a label whose first or second letter is not a
# token (no vocabulary checked so far; each reads all 255, those past `wide_coverage` in two passes)
TOO_MANY_OPTIONS = ("{qid}: this model's tokenizer can read {n} of Callo's two-letter labels (the next one needs a "
                    "letter that is not a single token); the question has {m} options")


class InvalidRequest(ValueError):
    """The request itself is unusable: HTTP 422."""


@dataclass(frozen=True)
class Question:
    qid: str
    type: str
    instructions: str
    keys: tuple[str, ...]
    labels: tuple[str, ...]
    texts: tuple[str, ...]
    legend: dict[str, str] | None = None


@dataclass(frozen=True)
class MediaItem:
    """One image, audio clip or video of a request."""
    kind: str                   # "image" | "audio" | "video"
    payload: str                # raw base64 (the text after a data URL's first ";base64,"): one multimodal_data entry
    url: str | None = None      # the data URL as sent; None for raw base64 in an input_audio / input_video part
    fmt: str | None = None      # audio only: the sniffed container, "wav" | "mp3" | "flac"; None for images and videos


def media_kind(media: Sequence[MediaItem], context: str) -> str:
    """`Readout.kind` of a request: its one media kind, "mixed" for more than one kind, else
    "context" or "plain". Defined once: the engine, its single-option shortcut, the proxy's stats and calibrate's set
    scan all use it."""
    kinds = {m.kind for m in media}
    if len(kinds) > 1:
        return "mixed"
    if kinds:
        return next(iter(kinds))
    return "context" if context else "plain"


@dataclass(frozen=True)
class Request:
    state: str
    model: str
    questions: tuple[Question, ...]
    digest: str
    media: tuple[MediaItem, ...] = ()   # contract order: images, audio, videos, then the state's parts
    context: str = ""                   # "" when absent, null or empty: the request then has no context
    think_budget: int | None = None     # the request's budget; None takes the server default
    think_deadline_ms: int | None = None    # the request's deadline in ms (0: none); None takes the default
    think_converge: float | None = None     # early exit's target z (0: off); None takes the server default
    seed_digest: str = ""   # the digest without `SEED_FREE`, which seeds think traces; "" (unset): `digest`
    think_converge_gate: bool | None = None     # early exit's gate (experimental); None takes the server default

    def __post_init__(self) -> None:
        if not self.seed_digest:
            object.__setattr__(self, "seed_digest", self.digest)

    @property
    def kind(self) -> str:
        return media_kind(self.media, self.context)

    @property
    def images(self) -> tuple[str, ...]:
        """Kept for older callers of `images` / `image_urls`: the image items' base64 payloads, in order."""
        return tuple(m.payload for m in self.media if m.kind == "image")

    @property
    def image_urls(self) -> tuple[str, ...]:
        """Kept for older callers of `images` / `image_urls`: the image items' data URLs (every image arrives as a
        data URL)."""
        return tuple(m.url or "" for m in self.media if m.kind == "image")


def is_wide(q: Question) -> bool:
    """A wide question: more than 26 options, so two-letter labels from `WIDE_LABELS`."""
    return len(q.labels) > MAX_LETTERS


def _is_letter(key: str) -> bool:
    return len(key) == 1 and key.isascii() and key.isalpha()


def assign_choice_labels(keys: Sequence[str]) -> list[str]:
    """Letter keys keep their own (uppercased) letter wherever they sit; the rest take the next unused
    letters in order. Two passes, so a later letter key is never stolen by an earlier non-letter key."""
    labels: list[str | None] = [None] * len(keys)
    used: set[str] = set()
    for i, k in enumerate(keys):
        if _is_letter(k) and k.upper() not in used:
            labels[i] = k.upper()
            used.add(k.upper())
    free = iter(c for c in string.ascii_uppercase if c not in used)
    return [lab if lab is not None else next(free) for lab in labels]


def _question(qid: str, q: Any) -> Question:
    if not isinstance(q, dict) or q.get("type") not in QUESTION_TYPES:
        raise InvalidRequest(f"{qid}: type must be one of {QUESTION_TYPES}")
    qtype, crit = q["type"], q.get("criteria")
    instructions = render(q.get("instructions"))
    if qtype == "noul":
        if crit is not None and not isinstance(crit, dict):
            raise InvalidRequest(f"{qid}: noul criteria must be an object or absent")
        c = crit or {}
        return Question(qid, qtype, instructions, ("false", "true"), ("A", "B"),
                        (option_text("no", c.get("false")), option_text("yes", c.get("true"))))
    if qtype == "choice":
        if not isinstance(crit, dict) or not 1 <= len(crit) <= MAX_OPTIONS:
            raise InvalidRequest(f"{qid}: choice criteria must be an object with 1..{MAX_OPTIONS} options")
    elif not isinstance(crit, list) or not 1 <= len(crit) <= MAX_OPTIONS:
        raise InvalidRequest(f"{qid}: score criteria must be a list of 1..{MAX_OPTIONS} levels")
    keys = tuple(question_keys(qtype, crit))
    if len(crit) > MAX_LETTERS:         # wide: the table's first n labels; no key keeps its own spelling
        labels = WIDE_LABELS[:len(crit)]
        if qtype == "choice":
            return Question(qid, qtype, instructions, keys, labels, tuple(option_text(k, crit[k]) for k in keys))
        texts = tuple(render(x) for x in crit)
        return Question(qid, qtype, instructions, keys, labels, texts, dict(zip(keys, texts, strict=True)))
    if qtype == "choice":
        labels = tuple(assign_choice_labels(keys))
        texts = tuple((render(crit[k]) if crit[k] not in (None, "") else k)
                      if _is_letter(k) and lab == k.upper() else option_text(k, crit[k])
                      for k, lab in zip(keys, labels, strict=True))
        return Question(qid, qtype, instructions, keys, labels, texts)
    texts = tuple(render(x) for x in crit)
    return Question(qid, qtype, instructions, keys, tuple(string.ascii_uppercase[: len(keys)]), texts,
                    dict(zip(keys, texts, strict=True)))


def _item(kind: str, payload: str, url: str | None, where: str) -> MediaItem:
    """Decode `payload` (raw base64) and check its bytes against `kind`: 422 on failure."""
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise InvalidRequest(f"{where}: invalid base64 ({e})") from None
    if not raw:
        raise InvalidRequest(f"{where}: the {kind if kind != 'audio' else 'audio clip'} is empty")
    why = bytes_error(kind, raw)
    if why is not None:
        raise InvalidRequest(f"{where}: {why}")
    return MediaItem(kind, payload, url, sniff_audio(raw) if kind == "audio" else None)


def _data_url(url: Any, where: str, kind: str, families: tuple[str, ...]) -> MediaItem:
    """One data URL, or 422. A superset of llama.cpp's own checks (`handle_media`), so a URL that passes here never
    makes the upstream answer 'Invalid uri-encoded base64 value'."""
    if not isinstance(url, str):
        raise InvalidRequest(f"{where}: {_NOUN[kind]} must be a string data URL")
    if not url.startswith(families):
        shown = families[0] if len(families) == 1 else ", ".join(families[:-1]) + " or " + families[-1]
        raise InvalidRequest(f"{where}: {_NOUN[kind]} must be a data URL starting with {shown}")
    _, sep, payload = url.partition(_B64)
    if not sep:
        raise InvalidRequest(f"{where}: the data URL has no ;base64,")
    return _item(kind, payload, url, where)


def _part_value(value: Any, where: str, kind: str) -> MediaItem:
    """An input_audio / input_video / video_url `data` or `url`, dispatched as `handle_media` does: `data:` is a
    data URL (any of llama.cpp's three MIME families, as llama.cpp allows), `http` or `file://`
    is a URL callo never fetches (422), anything else is raw base64."""
    if not isinstance(value, str):
        raise InvalidRequest(f"{where}: the media data or url must be a string")
    if value.startswith("data:"):
        return _data_url(value, where, kind, _ANY_FAMILY)
    if value.startswith(("http", "file://")):
        raise InvalidRequest(f"{where}: http(s):// and file:// media are not supported; send a data URL or raw base64")
    return _item(kind, value, None, where)


def _add(media: list[MediaItem], item: MediaItem) -> None:
    if len(media) >= MAX_MEDIA:
        raise InvalidRequest(TOO_MANY_MEDIA)
    media.append(item)


def _state_part(part: dict, where: str, media: list[MediaItem]) -> bool:
    """Take one state content part into `media` when it is a media part; False when it stays in the state."""
    t = part.get("type")
    if not isinstance(t, str):              # a JSON-schema "type": [...] or {...} is no media part: it stays
        return False
    if t in _UNSUPPORTED_MEDIA:
        raise InvalidRequest(f"{where}: {t} parts are not supported (llama.cpp refuses them); use input_audio or "
                             f"input_video")
    if t == "image_url":
        url = part.get("image_url")
        # llama.cpp reads the object under "image_url"; never flatten it as text
        if url is None:
            raise InvalidRequest(f"{where}: image_url has no image_url object")
        if isinstance(url, dict):
            if "url" not in url:
                raise InvalidRequest(f"{where}: image_url has no url")
            url = url["url"]
        if not isinstance(url, str):
            raise InvalidRequest(f"{where}: image_url url must be a string")
        _add(media, _data_url(url, where, "image", ("data:image/",)))
        return True
    if t in _PARTS:
        inner = part.get(t)
        if not isinstance(inner, dict):          # llama.cpp reads the object under the key equal to the type
            raise InvalidRequest(f"{where}: a {t} part needs a {t!r} object (the key equal to its type)")
        value = inner["data"] if inner.get("data") is not None else inner.get("url")     # data wins over url
        if value is None:
            raise InvalidRequest(f"{where}: {t} has no data or url")
        _add(media, _part_value(value, where, _PARTS[t]))
        return True
    return False


def _take_state_media(state: Any, media: list[MediaItem]) -> Any:
    """Native `parse_state`, extended to audio and video: when `state` is a list of messages or
    `{"messages": [...]}`, move every media content part into `media` (document order) and return the state without
    them. Anything else is returned unchanged."""
    wrapped = isinstance(state, dict) and "messages" in state
    msgs = state["messages"] if wrapped else state
    if not isinstance(msgs, list):
        return state
    out = []
    for mi, msg in enumerate(msgs):
        if not (isinstance(msg, dict) and isinstance(msg.get("content"), list)):
            out.append(msg)
            continue
        content = []
        for pi, part in enumerate(msg["content"]):
            if not (isinstance(part, dict) and _state_part(part, f"state message {mi} part {pi}", media)):
                content.append(part)
        out.append({**msg, "content": content})
    return {**state, "messages": out} if wrapped else out


def parse_request(body: Any) -> Request:
    if not isinstance(body, dict) or "state" not in body:
        raise InvalidRequest("request must be an object with a state")
    model = body.get("model", "kev-latest")
    if not isinstance(model, str):
        raise InvalidRequest("model must be a string")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise InvalidRequest("at least one question is required")
    context = body.get("context")
    if context is None:
        context = ""
    elif not isinstance(context, str):
        raise InvalidRequest("context must be a string")
    think_budget = body.get("think_budget")
    if think_budget is not None and (isinstance(think_budget, bool) or not isinstance(think_budget, int)
                                     or not 0 <= think_budget <= MAX_THINK_BUDGET):
        raise InvalidRequest(THINK_BUDGET_ERROR)
    think_deadline_ms = body.get("think_deadline_ms")
    if think_deadline_ms is not None and not _deadline_ok(think_deadline_ms):
        raise InvalidRequest(THINK_DEADLINE_ERROR)
    think_converge = body.get("think_converge")
    if think_converge is not None and not _converge_ok(think_converge):
        raise InvalidRequest(THINK_CONVERGE_ERROR)
    gate = body.get("think_converge_gate")
    if gate is not None and not isinstance(gate, bool):
        raise InvalidRequest(THINK_CONVERGE_GATE_ERROR)
    media: list[MediaItem] = []
    for name, kind in _FIELDS:
        urls = body.get(name)
        if urls is None:
            continue
        if not isinstance(urls, list):
            raise InvalidRequest(f'"{name}" must be an array of data URLs')
        if len(urls) > MAX_MEDIA:
            raise InvalidRequest(TOO_MANY_MEDIA)
        for i, url in enumerate(urls):
            _add(media, _data_url(url, f"{name}[{i}]", kind, (f"data:{kind}/",)))
    state = _take_state_media(body["state"], media)
    unsigned = {k: v for k, v in body.items() if k != "model"}
    digest = _digest(unsigned)
    # Early exit stops a trace, it does not reseed it: the seeds ignore think_converge, and its gate's
    # think_converge_gate. An early-exit trace is drawn K-chunked (chunk i reseeded), so it is the chunked live-mode
    # trace of the same body cut at the stop, and matches a one-shot budget trace only within its first chunk.
    seed_digest = _digest({k: v for k, v in unsigned.items() if k not in SEED_FREE}) \
        if any(k in unsigned for k in SEED_FREE) else digest
    return Request(render(state), model, tuple(_question(str(k), v) for k, v in qs.items()), digest,
                   media=tuple(media), context=context, think_budget=think_budget,
                   think_deadline_ms=think_deadline_ms,
                   think_converge=None if think_converge is None else float(think_converge) or 0.0,  # -0.0: 0.0
                   seed_digest=seed_digest, think_converge_gate=gate)


def _digest(unsigned: dict) -> str:
    return hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _deadline_ok(v: Any) -> bool:
    """An integer 0..600000; a bool is not one."""
    return not isinstance(v, bool) and isinstance(v, int) and 0 <= v <= MAX_DEADLINE_MS


def _converge_ok(v: Any) -> bool:
    """A number from 0 to below 1; a bool is not one, and nan and inf fail the comparison."""
    return not isinstance(v, bool) and isinstance(v, (int, float)) and 0 <= v < 1


def answer_body(q: Question, probs: Sequence[float]) -> dict:
    """The System One answer for one question. Probabilities are NOT rounded."""
    p = [float(x) for x in probs]
    if q.type == "noul":
        return {"type": "noul", "noul": p[1]}
    if q.type == "choice":
        best = max(range(len(p)), key=p.__getitem__)
        return {"type": "choice", "choice": q.keys[best], "confidence": choice_confidence(p),
                "probabilities": dict(zip(q.keys, p, strict=True))}
    return {"type": "score", "score": sum(i * pi for i, pi in enumerate(p)), "legend": q.legend,
            "probabilities": dict(zip(q.keys, p, strict=True)), "confidence": score_confidence(p)}


# Part types a NATIVE decision model would leave in its prompt as text (server-decision.cpp:244-256 @ 1537a0a).
NATIVE_REFUSED_PARTS = frozenset({"input_audio", "input_video", "video_url", "audio_url", "video"})


def native_media_refusal(body: Any) -> str | None:
    """Why a /v1/systemone body must not reach a NATIVE decision model, or None. Native takes
    images only: it ignores top-level `audio` / `videos`, so the model would answer without the media, and it renders
    audio and video state parts into its prompt as text. Refused: a non-empty `audio` / `videos` list (or any other
    non-null value), or one of `NATIVE_REFUSED_PARTS` in a messages-shaped state."""
    if not isinstance(body, dict):
        return None
    for name in ("audio", "videos"):
        v = body.get(name)
        if v is not None and v != []:
            return f'the upstream is a native decision model, which takes images only: "{name}" is not supported'
    state = body.get("state")
    msgs = state.get("messages") if isinstance(state, dict) else state
    for mi, msg in enumerate(msgs if isinstance(msgs, list) else []):
        content = msg.get("content") if isinstance(msg, dict) else None
        for pi, part in enumerate(content if isinstance(content, list) else []):
            t = part.get("type") if isinstance(part, dict) else None
            if isinstance(t, str) and t in NATIVE_REFUSED_PARTS:     # a list or dict "type" is no media part
                return (f"the upstream is a native decision model, which takes images only: state message {mi} part "
                        f"{pi} is {t}")
    return None


def native_think_refusal(body: Any) -> str | None:
    """Why a /v1/systemone body must not reach a NATIVE decision model because of its `think_budget`, its
    `think_deadline_ms`, its `think_converge` or its `think_converge_gate`, or None: native has no deliberate readout and
    would silently ignore any of them, so a positive value (a gate of true) is refused (422). A null, absent or zero
    value (a gate of false) passes; a malformed one is refused with the field's own message."""
    if not isinstance(body, dict):
        return None
    v = body.get("think_budget")
    if v is not None:
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= MAX_THINK_BUDGET:
            return THINK_BUDGET_ERROR
        if v > 0:
            return ("the upstream is a native decision model, which has no deliberate readout: think_budget must be 0 "
                    "or absent")
    d = body.get("think_deadline_ms")
    if d is not None:
        if not _deadline_ok(d):
            return THINK_DEADLINE_ERROR
        if d > 0:
            return ("the upstream is a native decision model, which has no deliberate readout: think_deadline_ms must "
                    "be 0 or absent")
    z = body.get("think_converge")
    if z is not None:
        if not _converge_ok(z):
            return THINK_CONVERGE_ERROR
        if z > 0:
            return ("the upstream is a native decision model, which has no deliberate readout: think_converge must be 0 "
                    "or absent")
    g = body.get("think_converge_gate")
    if g is not None:
        if not isinstance(g, bool):
            return THINK_CONVERGE_GATE_ERROR
        if g:
            return ("the upstream is a native decision model, which has no deliberate readout: think_converge_gate "
                    "must be false or absent")
    return None
