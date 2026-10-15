"""Question -> chat messages -> the model's OWN chat template, thinking disabled.

The template is applied by llama-server (`POST /apply-template`, `chat_template_kwargs.enable_thinking=false`)
rather than reconstructed here, so nothing in this package is model-specific. The answer slot is checked
mechanically at startup (`answer_slot_check`): render the messages, render them again with an assistant turn
`"A"`, and require the answer to follow the generation prompt, up to the template's own empty-reasoning block.
That replaces an earlier Qwen-specific `expected_tail` rule, which is kept only as an optional pin.

The main prompt is FIXED: it is part of the public contract below. The other variants are dev-only fragility
diagnostics.

The rendered System One prompt is a PUBLIC CONTRACT: the `SYSTEM_MAIN` text, the
`CONTEXT_JOINER` that appends a request's `context` to it, and `user_content`'s layout (an empty state drops the
`Input:` block). Changing any of them is a breaking change. `demo/prompts/system_main.txt` is a pinned copy.

A question with more than 26 options has the same layout with two-letter labels
(`AA. <option>`, from `systemone.WIDE_LABELS`), and its last line is `WIDE_REPLY`, listing every label of the question:
`Reply with one two-letter label: AA, AB, ....` `SYSTEM_MAIN` is unchanged (the reply line overrides its "letter"). The
table's sha256 and a 30-option golden render pin that contract; up to 26 options the prompt is byte-identical.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from .scorer import HttpPost, urllib_post
from .systemone import MediaItem, Question, is_wide, parse_request

SYSTEM_MAIN = ("You are a classifier. Read the input and the question, pick the best option, and reply with "
               "that option's letter only: no words, no punctuation, no explanation.")
VARIANTS: dict[str, str | None] = {
    "main": SYSTEM_MAIN,
    "para1": "Classify the input. Respond with exactly one letter, the letter of the correct option, and nothing else.",
    "para2": "Answer the multiple-choice question about the text below. Your entire reply must be a single option letter.",
    "nosys": None,
    "oneshot": SYSTEM_MAIN,
}
_ONESHOT_USER = ("Input:\nThe package arrived two weeks late and the box was crushed.\n\n"
                 "Question: What is the sentiment of this review?\n\n"
                 "Options:\nA. positive\nB. negative\nC. neutral\n\n"
                 "Reply with one letter: A, B, C.")
_ONESHOT_ANSWER = "B"
WIDE_REPLY = "Reply with one two-letter label: {labels}."          # labels joined by ", "
CONTEXT_JOINER = "\n\nContext:\n"     # public contract: SYSTEM_MAIN + CONTEXT_JOINER + escaped context
# /apply-template is media-blind (every media part becomes the same marker text, common/chat.cpp:194-215) and
# never decodes media (it only base64-decodes the payload, server-common.cpp:1079-1146), so the S1 prompt is rendered
# with tiny fixed placeholders and the real payloads go only to /completion. A 1x1 black PNG (vision.encode_png):
PLACEHOLDER_IMAGE_URL = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAD0lEQVR4AQEEAPv/AAAAAAAEAA"
                         "FlScNgAAAAAElFTkSuQmCC")
PLACEHOLDER_PAYLOAD = "AAAA"        # raw base64 of three zero bytes: non-empty, so never "Invalid base64 value"

# server-decision.cpp:342-346: `<|name|>` written in user text must not be parsed as a control token.
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
# A template's empty reasoning block, e.g. `<think>\n\n</think>\n\n`, which some templates render in the
# generation prompt but drop when the same assistant turn is re-rendered as history.
_EMPTY_REASONING = re.compile(r"\s*<[^<>]*think[^<>]*>\s*<[^<>]*/[^<>]*think[^<>]*>\s*")


def escape_special(text: str) -> str:
    """`<|x|>` -> `<¦x¦>` (U+00A6 BROKEN BAR). The prompt is tokenised with parse_special, so a literal
    `<|im_start|>` inside a user's state would otherwise become a real control token."""
    return _SPECIAL.sub("<¦\\1¦>", text)


def user_content(state: str, q: Question) -> str:
    parts = []
    if state:
        parts.append(f"Input:\n{escape_special(state)}")
    if q.instructions:
        parts.append(f"Question: {escape_special(q.instructions)}")
    parts.append("Options:\n" + "\n".join(f"{lab}. {escape_special(text)}"
                                          for lab, text in zip(q.labels, q.texts, strict=True)))
    if is_wide(q):
        parts.append(WIDE_REPLY.format(labels=", ".join(q.labels)))
    else:
        parts.append(f"Reply with one letter: {', '.join(q.labels)}.")
    return "\n\n".join(parts)


def system_text(variant: str = "main", context: str = "") -> str | None:
    """The system message: the variant's text, then `CONTEXT_JOINER` and the escaped `context` when there is one.
    The context sits in the system turn, BEFORE the images, so the prompt cache reuses it across requests."""
    system = VARIANTS[variant]
    if not context:
        return system
    if system is None:
        raise ValueError(f"variant {variant!r} has no system message to carry a context")
    return system + CONTEXT_JOINER + escape_special(context)


def media_part(item: MediaItem, *, placeholder: bool = False) -> dict[str, Any]:
    """The chat content part of one media item, shaped as llama.cpp reads it (server-common.cpp:1151-1201). With
    `placeholder` the payload is a fixed stand-in: the 1x1 PNG for an image, `"AAAA"` for audio (with its
    sniffed format, which llama.cpp ignores) and video. The part TYPE is still the item's: llama.cpp's allow_image /
    allow_audio / allow_video gate on it."""
    if item.kind == "image":
        url = PLACEHOLDER_IMAGE_URL if placeholder else item.url or f"data:image/png;base64,{item.payload}"
        return {"type": "image_url", "image_url": {"url": url}}
    data = PLACEHOLDER_PAYLOAD if placeholder else item.url or item.payload
    if item.kind == "audio":
        return {"type": "input_audio", "input_audio": {"data": data, "format": item.fmt or "wav"}}
    if item.kind == "video":
        return {"type": "input_video", "input_video": {"data": data}}
    raise ValueError(f"unknown media kind {item.kind!r}")


def messages(state: str, q: Question, variant: str = "main", *, context: str = "",
             image_urls: Sequence[str] = (), media: Sequence[MediaItem] = (),
             placeholder: bool = False) -> list[dict[str, Any]]:
    """The plain message list when there is neither a context nor media. With media the user turn is one part per
    item, in order, then ONE text part with the state, question and options: the only layout found reliable in
    measurement. `/apply-template` replaces each media part with the server's media marker. `image_urls` is the
    older form (image parts from data URLs); `media` takes `MediaItem`s, rendered with real payloads or, with
    `placeholder`, with the placeholder stand-ins. Pass one of the two."""
    if image_urls and media:
        raise ValueError("pass image_urls or media, not both")
    system = system_text(variant, context)
    msgs: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    if variant == "oneshot":
        msgs += [{"role": "user", "content": _ONESHOT_USER}, {"role": "assistant", "content": _ONESHOT_ANSWER}]
    text = user_content(state, q)
    parts: list[dict[str, Any]] = ([{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
                                   or [media_part(m, placeholder=placeholder) for m in media])
    if not parts:
        return msgs + [{"role": "user", "content": text}]
    return msgs + [{"role": "user", "content": [*parts, {"type": "text", "text": text}]}]


class TemplateError(RuntimeError):
    """The template endpoint returned no prompt, or a prompt not ending at the expected answer slot."""


class TemplateClient:
    def __init__(self, base_url: str, *, expected_tail: str | None = None, http_post: HttpPost = urllib_post):
        self._url = f"{base_url.rstrip('/')}/apply-template"
        self._tail = expected_tail
        self._post = http_post

    def render(self, msgs: list[dict[str, Any]], *, thinking: bool = False) -> str:
        """The template applied by llama-server. `thinking` renders with
        `enable_thinking` true: the generation prompt a deliberate trace starts from. False is today's request."""
        resp = self._post(self._url, {"messages": msgs, "chat_template_kwargs": {"enable_thinking": thinking}})
        prompt = resp.get("prompt")
        if not isinstance(prompt, str):
            raise TemplateError(f"apply-template returned no prompt: {list(resp)[:8]}")
        if self._tail is not None and not prompt.endswith(self._tail):
            raise TemplateError(f"rendered prompt does not end with the expected tail {self._tail!r}: "
                                f"...{prompt[-60:]!r}")
        return prompt


def check_answer_slot(prompt: str, with_answer: str, answer: str = "A") -> tuple[bool, str]:
    """The answer-slot check: `prompt` must be a prefix of `with_answer` with `answer` immediately after it, allowing
    the template to have dropped its own empty-reasoning block when re-rendering the turn as history."""
    n = 0
    for a, b in zip(prompt, with_answer, strict=False):   # the answered render is longer by design
        if a != b:
            break
        n += 1
    tail = prompt[n:]
    if tail and not _EMPTY_REASONING.fullmatch(tail):
        return False, f"rendered prompt diverges from the answered one before its end: tail {tail!r}"
    rest = with_answer[n:]
    if not rest.startswith(answer) or rest[len(answer):][:1].isalnum():   # not "ASSISTANT: A"
        return False, f"answer {answer!r} does not follow the generation prompt: got {rest[:40]!r}"
    note = " (template drops its empty reasoning block in history)" if tail else ""
    return True, "answer slot confirmed" + note


def answer_slot_check(template: TemplateClient, msgs: list[dict[str, Any]], answer: str = "A") -> tuple[bool, str]:
    """Render `msgs`, then `msgs` plus an assistant turn `answer` plus a dummy user turn, and compare.
    A trailing assistant message would be a PREFILL for llama-server's /apply-template (continue_final_message,
    no generation prompt), not history; the trailing user turn makes the template itself render the answer
    as an assistant history turn."""
    first = template.render(msgs)
    second = template.render(msgs + [{"role": "assistant", "content": answer}, {"role": "user", "content": "x"}])
    return check_answer_slot(first, second, answer)


# ---- the user-turn delimiter ---------------------------------------------------------
#
# Hybrid (SSM) models roll back only to a context checkpoint, and llama-server takes one at a user-message start
# only when the request carries message spans; `/completion` gets spans only from `message_delimiters`. The
# delimiter is derived from the model's own template, so nothing here is model-specific.

NO_DELIMITER_WARNING = ("no user-turn delimiter: hybrid models will re-prefill the system prefix on every image "
                        "request (measured: about +180 ms per image request)")
_S, _U1, _A, _U2, _C, _Q = "⟦S⟧", "⟦U1⟧", "⟦A⟧", "⟦U2⟧", "⟦C⟧", "⟦Q⟧"


def common_suffix(a: str, b: str) -> str:
    n = 0
    while n < min(len(a), len(b)) and a[len(a) - 1 - n] == b[len(b) - 1 - n]:
        n += 1
    return a[len(a) - n:]


def _between(text: str, start: str, end: str) -> str | None:
    i = text.find(start)
    j = text.find(end, i + len(start)) if i >= 0 else -1
    return text[i + len(start):j] if i >= 0 and j >= 0 else None


def derive_user_delimiter(template: TemplateClient) -> str | None:
    """Steps 1-3: `a` = what the template writes between a system turn and a user turn, `b` = between an
    assistant turn and a user turn; the delimiter is their longest common suffix, trailing whitespace stripped
    (`<|im_end|>\\n<|im_start|>user` on ChatML)."""
    first = template.render([{"role": "system", "content": _S}, {"role": "user", "content": _U1}])
    second = template.render([{"role": "system", "content": _S}, {"role": "user", "content": _U1},
                              {"role": "assistant", "content": _A}, {"role": "user", "content": _U2}])
    a, b = _between(first, _S, _U1), _between(second, _A, _U2)
    if a is None or b is None:
        return None
    return common_suffix(a, b).rstrip() or None


def check_user_delimiter(prompt: str, delimiter: str, *, after: str, before: str) -> tuple[bool, str]:
    """On a rendered prompt: exactly one occurrence, after the system turn's text and before the first
    media marker (or, text-only, before the user content)."""
    n = prompt.count(delimiter)
    if n != 1:
        return False, f"{delimiter!r} occurs {n} times in the rendered S1 prompt; it must occur exactly once"
    lo, hi = prompt.find(after), prompt.find(before)
    if lo < 0 or hi < 0:
        return False, f"the rendered S1 prompt lost its system or user text ({after!r} or {before!r} not found)"
    at = prompt.index(delimiter)
    if not (lo + len(after) <= at and at + len(delimiter) <= hi):
        return False, f"{delimiter!r} does not sit between the system turn and the user content"
    return True, "ok"


def _occurrences(seq: list[int], sub: list[int]) -> int:
    return sum(1 for i in range(len(seq) - len(sub) + 1) if seq[i:i + len(sub)] == sub) if sub else 0


def user_delimiter(template: TemplateClient, *, vision: bool, media_marker: str | None, image_url: str | None,
                   pinned: str | None = None, tokenize: Callable[..., list[int]] | None = None
                   ) -> tuple[str | None, list[str]]:
    """The whole user-turn delimiter procedure, run once at startup on every upstream: derive (or take `pinned`,
    which skips the derivation), then validate on a rendered S1 prompt with a context: a one-image prompt when the
    upstream has vision, a text-only one otherwise. llama-server matches the delimiter TOKEN-wise (common/chat.cpp
    `split`), so with `tokenize` the delimiter's tokens must also occur exactly once in the tokenized prompt (up to
    the first media marker: mtmd tokenizes each text chunk on its own). A derived value that fails validation is
    dropped; a pinned one is still used. Returns (delimiter or None, log lines); never raises."""
    lines: list[str] = []
    how = "pinned (--user-delimiter)" if pinned else "derived from the chat template"
    delimiter = pinned
    if delimiter is None:
        try:
            delimiter = derive_user_delimiter(template)
        except Exception as e:  # noqa: BLE001 -- startup diagnostics warn, they never stop callo
            lines.append(f"WARNING: could not derive the user-turn delimiter: {type(e).__name__}: {e}")
            delimiter = None
    if not delimiter:
        lines.append(f"WARNING: {NO_DELIMITER_WARNING}")
        return None, lines
    q = parse_request({"state": "", "questions": {"q": {"type": "choice", "instructions": _Q,
                                                        "criteria": {"a": None, "b": None}}}}).questions[0]
    with_image = vision and media_marker is not None and image_url is not None
    kind = "one-image" if with_image else "text-only context"
    try:
        prompt = template.render(messages("", q, context=_C, image_urls=(image_url,) if with_image else ()))
    except Exception as e:  # noqa: BLE001
        prompt, ok, detail = None, False, f"could not render the validation prompt (/apply-template): {type(e).__name__}: {e}"
    if prompt is not None:
        ok, detail = check_user_delimiter(prompt, delimiter, after=_C,
                                          before=media_marker if with_image else user_content("", q))
        if ok and tokenize is not None:
            head = prompt[:prompt.find(media_marker)] if with_image else prompt
            try:
                n = _occurrences(tokenize(head, add_special=True, parse_special=True),
                                 tokenize(delimiter, add_special=False, parse_special=True))
                if n != 1:
                    ok, detail = False, (f"its tokens occur {n} times in the tokenized S1 prompt (llama-server "
                                         f"matches tokens, not text)")
            except Exception as e:  # noqa: BLE001
                ok, detail = False, f"could not tokenize the validation prompt (/tokenize): {type(e).__name__}: {e}"
    if ok:
        lines.append(f"user delimiter {delimiter!r}: {how}, validated on a {kind} S1 prompt")
        return delimiter, lines
    lines.append(f"WARNING: user delimiter {delimiter!r} ({how}) failed validation on a {kind} S1 prompt: {detail}")
    if pinned:
        lines.append(f"user delimiter {delimiter!r}: pinned, used anyway")
        return delimiter, lines
    lines.append(f"WARNING: {NO_DELIMITER_WARNING}")
    return None, lines
