"""Pinned `/v1/audio/transcriptions`.

llama-server reads multipart form fields as strings, so a form `id_slot` never pins a transcription. callo translates a
well-formed request into the `/v1/chat/completions` request llama-server builds itself, the proxy pins that like any
agent chat, and the chat reply is reshaped into the transcription shape. Upstream, at llama.cpp 4ebdf2c:

- the conversion: `convert_transcriptions_to_chatcmpl` (`tools/server/server-chat.cpp:651-706`); the preset:
  `common_chat_get_asr_prompt` (`common/chat.cpp:731-742`), LFM2 by `is_lfm2_template` (`common/parsers/lfm2.cpp:5-8`);
  the reply: `to_json_oaicompat_asr` (`tools/server/server-task.cpp:715-728`);
- the multipart parser: cpp-httplib's `FormDataParser` (`vendor/cpp-httplib/httplib.cpp:4538-4716`), the boundary from
  `parse_multipart_boundary` (`:4362-4374`) over `extract_media_type` (`:2823-2850`), header lines by `parse_header`
  (`:1155-1193`; the field rules `:5462-5517`), Content-Disposition by `parse_content_disposition` (`:4743-4768`) and
  `parse_disposition_params` (`:4376-4392`, over `split_unquoted` `:1369-1398`, `divide_param_pair` `:1403-1414`,
  `trim_double_quotes_copy` `:1311-1316`); a part is a file iff its filename is non-empty (`:8791-8797`), and the fields
  become JSON strings (`tools/server/server-http.cpp:700-726`).

`split_form` mirrors that parser and is STRICTER wherever being exact would cost more than passing through: anything it
cannot frame exactly is None, and the request goes to llama-server untouched (`not_multipart`), so callo never
translates a body llama-server would refuse or read differently.
"""
from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .av import encode_wav
from .media import sniff_audio
from .prompt import TemplateClient
from .scorer import HttpPost
from .think import stof

BOUNDARY_MAX = 70            # parse_multipart_boundary (httplib.cpp:4372-4373), RFC 2046 5.1.1
HEADER_MAX_LENGTH = 8192     # CPPHTTPLIB_HEADER_MAX_LENGTH (httplib.h:114): a part header line, CRLF excluded
HEADER_MAX_COUNT = 100       # CPPHTTPLIB_HEADER_MAX_COUNT (httplib.h:118): header lines per part
PARTS_MAX = 1024             # CPPHTTPLIB_MULTIPART_FORM_DATA_FILE_MAX_COUNT (httplib.h:126; the check at :8786)
_TOKEN = re.compile(rb"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")       # fields::is_token (httplib.cpp:5464-5476)
# fields::is_field_content (:5480-5511): empty, or visible (or obs-text) at both ends with SP / HTAB allowed between
_FIELD_VALUE = re.compile(rb"(?:[\x21-\x7e\x80-\xff](?:[\x20\t\x21-\x7e\x80-\xff]*[\x21-\x7e\x80-\xff])?)?")
_DISPOSITION = b"content-disposition:"
_CONTENT_TYPE = b"Content-Type:"
_FORM_DATA = b"form-data;"

TRANSCRIPTION_PATHS = frozenset({"/v1/audio/transcriptions", "/audio/transcriptions"})   # server.cpp:271-272
# the rows in order: `transcription_unrouted_<reason>` in /callo/stats.counters (`unrouted_counter`), each absent
# until first counted, as every counter there
REASONS = ("no_audio", "no_pinning", "render", "chunked", "large", "encoded", "not_multipart", "file", "fields",
           "response_format", "stream", "numbers")
UPSTREAM_MAX_BODY = 100 * 1024 * 1024    # CPPHTTPLIB_PAYLOAD_MAX_LENGTH (httplib.h:130); llama-server never raises it
ID_SLOT_ROOM = 32                        # >= len('{"id_slot":N,') - 1 for any slot id: the size check's margin
ASR_HINT = "callo:asr"                   # the routing hint of a transcription whose client sends none
# request headers dropped besides hop-by-hop; then Content-Type: application/json is set
REQUEST_DROP = frozenset({"host", "content-length", "expect", "content-type", "content-encoding", "transfer-encoding",
                          "accept-encoding"})
REPLY_DROP = frozenset({"server", "date", "content-length", "content-type"})   # the 200 reply's, besides hop-by-hop
FLT_MAX = 3.4028234663852886e38          # std::stof throws out_of_range beyond these
FLT_MIN = 1.1754943508222875e-38
ULONG_MAX = 2**64 - 1                    # std::stoul on LP64
DEFAULT_PRESET = ("", "Transcribe audio to text")       # (system, user): common/chat.cpp:731-742
LFM2_PRESET = ("Perform ASR.", "")
_TEMPERATURE = re.compile(r"-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?")
_MAX_TOKENS = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class Part:
    """One part as cpp-httplib maps it: `name` and `filename` from Content-Disposition (raw bytes; `filename == b""`
    makes it a form field), and the content, a view into the request body (no copy)."""
    name: bytes
    filename: bytes
    content: memoryview


def _split_unquoted(s: bytes, sep: bytes) -> list[bytes]:
    """`split_unquoted` (httplib.cpp:1369-1398): split on `sep` outside double quotes (a quote toggles; there are no
    escapes), trim spaces and tabs from each piece, drop the empty pieces."""
    out: list[bytes] = []
    start, quoted = 0, False
    for i in range(len(s)):
        c = s[i:i + 1]
        if c == b'"':
            quoted = not quoted
        elif c == sep and not quoted:
            piece = s[start:i].strip(b" \t")
            if piece:
                out.append(piece)
            start = i + 1
    piece = s[start:].strip(b" \t")
    if piece:
        out.append(piece)
    return out


def _unquote(v: bytes) -> bytes:
    """`trim_double_quotes_copy` (httplib.cpp:1311-1316)."""
    return v[1:-1] if len(v) >= 2 and v[:1] == b'"' and v[-1:] == b'"' else v


def _params(s: bytes, *, unquote_keys: bool) -> dict[bytes, list[bytes]]:
    """`key=value` parameters: split at the first '=' (`divide_param_pair`, :1403-1414), both sides trimmed, the value
    unquoted; the key too in Content-Disposition (`parse_disposition_params`) but not in Content-Type
    (`extract_media_type`). Every value of a repeated key is kept, so the caller can refuse a conflict."""
    out: dict[bytes, list[bytes]] = {}
    for piece in _split_unquoted(s, b";"):
        key, _, val = piece.partition(b"=")
        key = key.strip(b" \t")
        if key:
            out.setdefault(_unquote(key) if unquote_keys else key, []).append(_unquote(val.strip(b" \t")))
    return out


def form_boundary(content_type: str) -> bytes | None:
    """The boundary of a `multipart/form-data` Content-Type, exactly as cpp-httplib takes it, or None: the media type
    must be exactly `multipart/form-data` once trimmed (`is_multipart_form_data`, :6821-6824), and the boundary
    non-empty and at most 70 bytes (`parse_multipart_boundary`). Before that, httplib refuses the whole request (400)
    if the header itself is bad (`read_headers`, :3472-3484): its line, CRLF included, longer than 8192 bytes, or a
    value that is not field content (an obs-fold, a control byte). Stricter: a `boundary` repeated with different
    values is None."""
    try:
        raw = content_type.encode("latin-1")      # http.server decodes header bytes as latin-1: these are the wire bytes
    except UnicodeEncodeError:
        return None
    # the shortest wire line that carries this value is `Content-Type:` value CRLF
    if len(_CONTENT_TYPE) + len(raw) + 2 > HEADER_MAX_LENGTH or not _FIELD_VALUE.fullmatch(raw.strip(b" \t")):
        return None
    media, _, rest = raw.partition(b";")
    if media.strip(b" \t") != b"multipart/form-data":
        return None
    values = set(_params(rest, unquote_keys=False).get(b"boundary", []))
    if len(values) != 1:
        return None
    (b,) = values
    return b if 0 < len(b) <= BOUNDARY_MAX else None


def _header(line: bytes) -> tuple[bytes, bytes] | None:
    """`parse_header` (httplib.cpp:1155-1193): a token name, ':', then the value without its surrounding spaces and
    tabs, which must be field content."""
    name, colon, value = line.rstrip(b" \t").partition(b":")
    if not colon or not _TOKEN.fullmatch(name):
        return None
    value = value.lstrip(b" \t")
    return (name, value) if _FIELD_VALUE.fullmatch(value) else None


def _disposition(line: bytes) -> bytes | None:
    """The parameter text of a `Content-Disposition: form-data; ...` line (`parse_content_disposition`, :4743-4768: both
    prefixes case-insensitive, spaces and tabs skipped after each), or None."""
    if line[:len(_DISPOSITION)].lower() != _DISPOSITION:
        return None
    rest = line[len(_DISPOSITION):].lstrip(b" \t")
    if rest[:len(_FORM_DATA)].lower() != _FORM_DATA:
        return None
    return rest[len(_FORM_DATA):].lstrip(b" \t")


def _part(lines: list[bytes], content: memoryview) -> Part | None:
    """One part from its header lines (state 2, :4581-4665). httplib refuses a header it cannot parse and a
    Content-Disposition without `name`. Stricter: exactly one form-data Content-Disposition, one `name`, at most one
    `filename`, no `filename*`, and no Content-Transfer-Encoding but `binary` (httplib never decodes one, and the
    translation passes it through)."""
    name: bytes | None = None
    filename = b""
    dispositions = 0
    for line in lines:
        header = _header(line)
        if header is None:
            return None
        params = _disposition(line)
        if params is not None:
            dispositions += 1
            p = _params(params, unquote_keys=True)
            names, filenames = set(p.get(b"name", [])), set(p.get(b"filename", []))
            if len(names) != 1 or len(filenames) > 1 or b"filename*" in p:
                return None
            name = names.pop()
            filename = filenames.pop() if filenames else b""
        elif header[0].lower() == b"content-transfer-encoding" and header[1].lower() != b"binary":
            return None
    if dispositions != 1 or name is None:
        return None
    return Part(name, filename, content)


def split_form(content_type: str, body: bytes) -> list[Part] | None:
    """The parts of a multipart/form-data body, framed as cpp-httplib's FormDataParser frames it, or None:

    - the first `--B` CRLF starts the first part, and any preamble before it is skipped (state 0, :4562-4579);
    - a part's header lines end in CRLF, at most 100 of at most 8192 bytes each, up to an empty line (state 2);
    - its content runs to the next CRLF `--B` (state 3, :4666-4688), which must be followed by CRLF (another part) or
      `--` (the close delimiter, after which the epilogue is ignored) and by nothing else, so no transport padding
      (state 4, :4689-4709);
    - a body that ends before the close delimiter is None (only the close sets `is_valid_`).
    """
    boundary = form_boundary(content_type)
    if boundary is None:
        return None
    first = body.find(b"--" + boundary + b"\r\n")
    if first < 0:
        return None
    view = memoryview(body)
    delimiter = b"\r\n--" + boundary
    pos = first + len(boundary) + 4
    parts: list[Part] = []
    while len(parts) < PARTS_MAX:
        lines: list[bytes] = []
        while True:
            eol = body.find(b"\r\n", pos, pos + HEADER_MAX_LENGTH + 2)    # a line of at most 8192 bytes, CRLF excluded
            if eol < 0:
                return None
            if eol == pos:
                pos += 2
                break
            if len(lines) == HEADER_MAX_COUNT:
                return None
            lines.append(body[pos:eol])
            pos = eol + 2
        end = body.find(delimiter, pos)
        if end < 0:
            return None
        part = _part(lines, view[pos:end])
        if part is None:
            return None
        parts.append(part)
        after = end + len(delimiter)
        if body[after:after + 2] == b"--":
            return parts
        if body[after:after + 2] != b"\r\n":
            return None
        pos = after + 2
    return None


# ---- the translation ----------------------------------------------------------------------------

def is_lfm2(chat_template: str | None) -> bool:
    """`is_lfm2_template` (common/parsers/lfm2.cpp:5-8 @ 4ebdf2c) over `/props.chat_template`, the default template's
    source. A new upstream ASR preset needs a callo update."""
    return bool(chat_template) and "<|tool_list_start|>" in chat_template and "<|tool_list_end|>" in chat_template


def preset(lfm2: bool) -> tuple[str, str]:
    """(system, user) of `common_chat_get_asr_prompt` (common/chat.cpp:731-742)."""
    return LFM2_PRESET if lfm2 else DEFAULT_PRESET


def numbers_ok(fields: Mapping[str, str]) -> bool:
    """`temperature` and `max_tokens` exactly where llama-server's `std::stof` / `std::stoul` and Python
    agree. Leading zeros are fine ("007" is 7 to both). A value `stoul` throws on (above 2^64-1) is refused here, so
    llama-server answers it; this also keeps Python's int() digit limit (4300 digits on 3.11+) from ever raising.
    callo assumes LP64 (`unsigned long` is 64 bits, as on Linux and macOS; not Windows' LLP64).

    `temperature` is judged on the decimal's exact value, not on its nearest double: 0 only when every mantissa digit is
    0 (`1e-400` is a double 0.0, but `strtof` underflows on it), and a nonzero value below 2^-126 (FLT_MIN) is refused
    whatever the libc's tininess rule, even when its double rounds up to FLT_MIN. An exponent beyond `Decimal`'s range
    is refused too. The upper bound stays on the double (`|float(t)| <= FLT_MAX`), stricter than `strtof`'s overflow."""
    t = fields.get("temperature")
    if t is not None:
        if not _TEMPERATURE.fullmatch(t) or abs(float(t)) > FLT_MAX:
            return False
        try:
            x = Decimal(t).copy_abs()             # exact: abs() would round to the context
        except InvalidOperation:
            return False
        if x != 0 and x < Decimal(FLT_MIN):
            return False
    m = fields.get("max_tokens")
    if m is not None:
        if not _MAX_TOKENS.fullmatch(m):
            return False
        digits = m.lstrip("0") or "0"
        if len(digits) > 20 or int(digits) > ULONG_MAX:
            return False
    return True


def translate(fields: Mapping[str, str], audio: bytes | memoryview, fmt: str, *, lfm2: bool) -> dict[str, Any]:
    """The chat body `convert_transcriptions_to_chatcmpl` builds (server-chat.cpp:651-706), as content parts: `prompt`,
    else the preset's user text, then ` (language: X)`, then the audio LAST (the jinja path renders
    `[text, input_audio]` as `text + marker`; the startup render check verifies that on the live template). Every
    form field is copied verbatim as a string, then `messages` and `stream` are set and `max_tokens` / `temperature`
    typed. A form `id_slot` is left out: the proxy stamps the routed int."""
    system, user = preset(lfm2)
    text = fields.get("prompt") or user
    if fields.get("language"):
        text += f" (language: {fields['language']})"
    part = {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": fmt}}
    content = ([{"type": "text", "text": text}] if text else []) + [part]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body: dict[str, Any] = {k: v for k, v in fields.items() if k != "id_slot"}
    body.update(messages=messages, stream=False)
    if "max_tokens" in fields:
        body["max_tokens"] = int(fields["max_tokens"].lstrip("0") or "0")
    if "temperature" in fields:
        body["temperature"] = stof(fields["temperature"])
    return body


def unrouted_counter(reason: str) -> str:
    """`transcription_unrouted_<reason>` for one of `REASONS` (a ValueError for anything else: a typo in a caller must
    not invent a counter docs/guide/transcription.md does not list)."""
    if reason not in REASONS:
        raise ValueError(f"{reason!r} is not a transcription passthrough reason ({', '.join(REASONS)})")
    return f"transcription_unrouted_{reason}"


@dataclass(frozen=True)
class Translation:
    """A translatable request: the chat body (without `id_slot`) and the form fields, verbatim (the router reads the
    `user` field as a conversation hint)."""
    body: dict[str, Any]
    fields: dict[str, str]


def plan(content_types: Sequence[str], body: bytes, *, lfm2: bool) -> Translation | str:
    """The body rows, in order, after the proxy's pre-body rows: the translation, or the reason the request passes
    through (`not_multipart`, `file`, `fields`, `response_format`, `stream`, `numbers`). `content_types` is every
    Content-Type header the request carried (exactly one is required)."""
    parts = split_form(content_types[0], body) if len(content_types) == 1 else None
    if parts is None:
        return "not_multipart"
    named = [p for p in parts if p.name == b"file"]
    if len(named) != 1 or not named[0].filename:                  # the `file` row needs a filename
        return "file"
    audio = named[0].content
    fmt = sniff_audio(bytes(audio[:16]))
    if fmt is None:
        return "file"
    fields: dict[str, str] = {}
    for p in parts:
        if p is named[0]:
            continue
        if p.filename:                                            # another file part
            return "fields"
        try:
            k, v = p.name.decode("utf-8"), bytes(p.content).decode("utf-8")
        except UnicodeDecodeError:                                # llama-server's JSON dump would throw
            return "fields"
        if k in fields:                                           # a repeated key becomes an array upstream
            return "fields"
        fields[k] = v
    if "\x00" in fields.get("language", ""):    # upstream's `%s` of language.c_str() stops at a NUL (server-chat.cpp:678)
        return "fields"
    if fields.get("response_format", "json") != "json":
        return "response_format"
    if fields.get("stream", "false") != "false":
        return "stream"
    if not numbers_ok(fields):
        return "numbers"
    return Translation(translate(fields, audio, fmt, lfm2=lfm2), fields)


def reshape(payload: bytes) -> dict | None:
    """The transcription reply from a chat completion's bytes (`to_json_oaicompat_asr`, server-task.cpp:715-728),
    or None when `choices[0].message.content` is not a string (a bad reply, 502). `""` is a transcript. Missing
    usage numbers are null and a missing `cached_tokens` is 0.

    A known divergence the bytes cannot show: when the parsed message is entirely empty (whitespace-only output, a bare
    empty think block), the chat reply falls back to the raw generated text (server-task.cpp:414-422) where the native
    endpoint's `oaicompat_msg.content` is `""`. callo relays the chat reply's text."""
    try:
        obj = json.loads(payload)
        text = obj["choices"][0]["message"]["content"]
    except (ValueError, RecursionError, KeyError, IndexError, TypeError):
        return None
    if not isinstance(text, str):
        return None
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    return {"type": "transcript.text.done", "text": text,
            "usage": {"type": "tokens", "input_tokens": usage.get("prompt_tokens"),
                      "output_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens"),
                      "input_tokens_details": {"cached_tokens": details.get("cached_tokens", 0)}}}


# ---- the startup decision -----------------------------------------------------------------------------

PLACEHOLDER_WAV = encode_wav([])   # the 44-byte header: /apply-template decodes the base64 but never the audio
RENDER_TEXT = "T "                 # a trailing space, so a template that trims parts (StepFun) shows as a difference


@dataclass(frozen=True)
class Transcription:
    """What `callo serve` decided at startup for an upstream that reports audio: the preset, and why translation is
    off for this process (None: on)."""
    lfm2: bool
    off: str | None = None

    @property
    def preset(self) -> str:
        """`/callo/stats.asr_preset`."""
        return "lfm2" if self.lfm2 else "default"


def render_pairs(marker: str, *, lfm2: bool) -> list[tuple[list[dict], list[dict]]]:
    """(callo's content parts, upstream's string) for the two shapes a translation takes: a text part plus the audio,
    and the audio alone (LFM2 without a prompt or language), each behind the preset's system message."""
    system, _ = preset(lfm2)
    head = [{"role": "system", "content": system}] if system else []
    audio = {"type": "input_audio",
             "input_audio": {"data": base64.b64encode(PLACEHOLDER_WAV).decode("ascii"), "format": "wav"}}
    return [(head + [{"role": "user", "content": [{"type": "text", "text": RENDER_TEXT}, audio]}],
             head + [{"role": "user", "content": RENDER_TEXT + marker}]),
            (head + [{"role": "user", "content": [audio]}], head + [{"role": "user", "content": marker}])]


def render_check(render: Callable[[list[dict]], str], marker: str, *, lfm2: bool) -> str | None:
    """None when every pair renders byte-identically (translation on), else why not (translation off for the
    process: llama-server's `--no-jinja` path joins parts with a newline, StepFun templates trim each part). A render
    that fails is a reason too, and so is a parts render without the marker (an `/apply-template` that
    ignores `messages` would otherwise render both forms alike). Never raises."""
    for form, (parts, string) in zip(("text + audio", "audio alone"), render_pairs(marker, lfm2=lfm2), strict=True):
        try:
            a, b = render(parts), render(string)
        except Exception as e:  # noqa: BLE001 -- the check decides, it never stops a startup
            return f"the render check could not render the {form} form: {type(e).__name__}: {e}"
        if not marker or marker not in a:     # a non-empty render that holds the marker, before equality counts
            return (f"the {form} form renders without the media marker {marker!r} (an /apply-template that ignores "
                    f"the messages, or a wrong marker): parts end {a[-60:]!r}")
        if a != b:
            return (f"the chat template renders the {form} form differently from llama-server's own string (a "
                    f"--no-jinja server, or a template that trims content parts): parts end {a[-60:]!r}, the string "
                    f"ends {b[-60:]!r}")
    return None


def setup(upstream: str, chat_template: str | None, marker: str, *, http_post: HttpPost) -> Transcription:
    """The startup decision for an upstream that reports audio: the preset from `/props.chat_template` and the
    render check through its `/apply-template`. `http_post` carries callo's own `--upstream-api-key`, the only call
    that key is used for here: translated requests carry the client's headers."""
    lfm2 = is_lfm2(chat_template)
    return Transcription(lfm2, render_check(TemplateClient(upstream, http_post=http_post).render, marker, lfm2=lfm2))


def status(t: Transcription | None, *, pinning: bool) -> dict:
    """`/callo/stats.transcription` (shaped like `think.capable` / `reason`): whether this process translates,
    and if not why, as `<reason>: <detail>`. The reasons are tested in the proxy handler's own order (`no_audio`, then
    `no_pinning`, then `render`), so `<reason>` names the counter every transcription increments
    (`transcription_unrouted_<reason>`)."""
    if t is None:
        why: str | None = "no_audio: the upstream reports no audio"
    elif not pinning:
        why = "no_pinning: agent pinning is off" + (f" (and the render check failed: {t.off})" if t.off else "")
    elif t.off is not None:
        why = f"render: {t.off}"
    else:
        why = None
    return {"translate": why is None, "reason": why}


def startup_lines(t: Transcription | None, *, pinning: bool) -> list[str]:
    """`callo serve`'s one startup line about transcriptions, with `status`'s reason and the counter it names (no
    audio too). A failed render check is a WARNING; no audio and agent pinning off are configuration, logged as
    info."""
    st = status(t, pinning=pinning)
    if st["translate"]:
        assert t is not None
        return [f"transcription: /v1/audio/transcriptions is translated to /v1/chat/completions and pinned to an agent "
                f"slot (ASR preset {t.preset})"]
    reason, _, detail = st["reason"].partition(": ")
    line = (f"transcription: translation is OFF for this process; every /v1/audio/transcriptions passes through "
            f"unpinned (counted as transcription_unrouted_{reason}): {detail}")
    return [f"WARNING: {line}" if reason == "render" else line]
