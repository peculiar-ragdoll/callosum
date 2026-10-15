"""The user-turn delimiter that lets a hybrid model reuse the cached system turn. Derived from the model's
own template at startup, validated on a rendered S1 prompt, or pinned with --user-delimiter."""
from callo.prompt import (
    NO_DELIMITER_WARNING,
    TemplateClient,
    check_user_delimiter,
    common_suffix,
    derive_user_delimiter,
    user_delimiter,
)

MARKER = "<__media_test__>"
URL = "data:image/png;base64,iVBORw0KGgo="


def _parts(content):
    """llama.cpp's content-part join: image parts become the marker, text parts are joined with a newline, and no
    newline is added next to a marker (common/chat.cpp concat_content_parts)."""
    if isinstance(content, str):
        return content
    out, last_marker = "", False
    for p in content:
        if p["type"] == "image_url":
            out += MARKER
            last_marker = True
        else:
            out += ("\n" if out and not last_marker else "") + p["text"]
            last_marker = False
    return out


class Tmpl(TemplateClient):
    """A TemplateClient over a pure render function, recording every message list it renders."""

    def __init__(self, render_fn):
        super().__init__("http://x", http_post=lambda url, payload: {"prompt": render_fn(payload["messages"])})
        self.seen = []
        self._fn = render_fn

    def render(self, msgs):
        self.seen.append(msgs)
        return super().render(msgs)


def chatml(msgs):
    return "".join(f"<|im_start|>{m['role']}\n{_parts(m['content'])}<|im_end|>\n" for m in msgs) + \
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def turns(msgs):
    """A `<start_of_turn>` template with a real system turn."""
    role = {"assistant": "model"}
    return "".join(f"<start_of_turn>{role.get(m['role'], m['role'])}\n{_parts(m['content'])}<end_of_turn>\n"
                   for m in msgs) + "<start_of_turn>model\n"


def folded(msgs):
    """Gemma-style: the system text is folded into the first user turn, so no separator precedes the user content."""
    sys = [m for m in msgs if m["role"] == "system"]
    rest = [m for m in msgs if m["role"] != "system"]
    out = ""
    for i, m in enumerate(rest):
        text = _parts(m["content"])
        if i == 0 and sys:
            text = sys[0]["content"] + "\n\n" + text
        out += f"<start_of_turn>{m['role']}\n{text}<end_of_turn>\n"
    return out + "<start_of_turn>model\n"


def test_common_suffix():
    assert common_suffix("x<|im_end|>\n<|im_start|>user\n", "<|im_end|>\n<|im_start|>user\n") == \
        "<|im_end|>\n<|im_start|>user\n"
    assert common_suffix("abc", "xyz") == "" and common_suffix("", "a") == ""


def test_derives_the_chatml_delimiter_from_two_renders_with_thinking_off():
    t = Tmpl(chatml)
    assert derive_user_delimiter(t) == "<|im_end|>\n<|im_start|>user"
    assert [[m["role"] for m in msgs] for msgs in t.seen] == [["system", "user"],
                                                              ["system", "user", "assistant", "user"]]


def test_derives_a_start_of_turn_delimiter():
    assert derive_user_delimiter(Tmpl(turns)) == "<end_of_turn>\n<start_of_turn>user"


def test_a_template_that_folds_system_into_user_has_no_delimiter_and_warns():
    assert derive_user_delimiter(Tmpl(folded)) is None
    d, lines = user_delimiter(Tmpl(folded), vision=False, media_marker=None, image_url=None)
    assert d is None and lines == [f"WARNING: {NO_DELIMITER_WARNING}"]
    assert NO_DELIMITER_WARNING == ("no user-turn delimiter: hybrid models will re-prefill the system prefix on every "
                                    "image request (measured: about +180 ms per image request)")


def test_a_template_that_drops_the_system_turn_has_no_delimiter():
    assert derive_user_delimiter(Tmpl(lambda msgs: chatml([m for m in msgs if m["role"] != "system"]))) is None


def test_check_requires_exactly_once_between_the_system_turn_and_the_marker():
    p = "<|im_start|>system\nS⟦C⟧<|im_end|>\n<|im_start|>user\n<M>Question"
    assert check_user_delimiter(p, "<|im_end|>\n<|im_start|>user", after="⟦C⟧", before="<M>")[0]
    ok, detail = check_user_delimiter(p, "<|im_start|>", after="⟦C⟧", before="<M>")
    assert not ok and "2 times" in detail
    ok, detail = check_user_delimiter(p, "user\n<M>Question", after="⟦C⟧", before="<M>")
    assert not ok and "between" in detail
    ok, detail = check_user_delimiter(p, "<|im_end|>", after="⟦C⟧", before="<NOPE>")
    assert not ok and "not found" in detail


def test_vision_validation_renders_a_one_image_prompt_with_a_context():
    t = Tmpl(chatml)
    d, lines = user_delimiter(t, vision=True, media_marker=MARKER, image_url=URL)
    assert d == "<|im_end|>\n<|im_start|>user"
    probe = t.seen[-1]
    assert probe[0]["role"] == "system" and "Context:" in probe[0]["content"]
    assert probe[1]["content"][0] == {"type": "image_url", "image_url": {"url": URL}}
    assert any("validated" in line and "one-image" in line for line in lines)
    assert not any(line.startswith("WARNING") for line in lines)


def test_text_only_validation_runs_when_vision_is_off():
    t = Tmpl(chatml)
    d, lines = user_delimiter(t, vision=False, media_marker=None, image_url=None)
    assert d == "<|im_end|>\n<|im_start|>user"
    probe = t.seen[-1]
    assert isinstance(probe[1]["content"], str) and "Context:" in probe[0]["content"]   # no image part anywhere
    assert any("text-only" in line for line in lines)


def test_a_derived_delimiter_that_fails_validation_is_dropped_with_a_warning():
    def twice(msgs):                                    # the system text itself contains the user-turn opener
        return chatml(msgs).replace("Context:", "<|im_end|>\n<|im_start|>user Context:")

    d, lines = user_delimiter(Tmpl(twice), vision=True, media_marker=MARKER, image_url=URL)
    assert d is None
    assert any(line.startswith("WARNING") and "failed validation" in line for line in lines)
    assert any(NO_DELIMITER_WARNING in line for line in lines)


def test_a_pinned_delimiter_skips_derivation_and_is_used_even_when_validation_warns():
    t = Tmpl(chatml)
    d, lines = user_delimiter(t, vision=True, media_marker=MARKER, image_url=URL, pinned="<|im_start|>user")
    assert d == "<|im_start|>user" and len(t.seen) == 1          # only the validation render, no derivation
    assert any("pinned" in line for line in lines) and not any(line.startswith("WARNING") for line in lines)
    t2 = Tmpl(chatml)
    d, lines = user_delimiter(t2, vision=True, media_marker=MARKER, image_url=URL, pinned="<|im_start|>")
    assert d == "<|im_start|>"                                   # occurs 3 times: warned, still used
    assert any(line.startswith("WARNING") and "failed validation" in line for line in lines)


def _toy_tokenize(merge_user_newline: bool):
    """Control tokens are one id each; with `merge_user_newline`, `user\\n` is one piece (a SentencePiece-style
    tokenizer); every other character is its own id."""
    pieces = {"<|im_start|>": 1, "<|im_end|>": 2, **({"user\n": 3} if merge_user_newline else {})}

    def tokenize(text, *, add_special=True, parse_special=True):
        out, i = [], 0
        while i < len(text):
            hit = next((p for p in pieces if text.startswith(p, i)), None)
            out.append(pieces[hit] if hit else 1000 + ord(text[i]))
            i += len(hit) if hit else 1
        return out
    return tokenize


def test_the_delimiter_is_also_validated_token_wise():
    d, lines = user_delimiter(Tmpl(chatml), vision=True, media_marker=MARKER, image_url=URL,
                              tokenize=_toy_tokenize(False))
    assert d == "<|im_end|>\n<|im_start|>user" and not any(line.startswith("WARNING") for line in lines)
    d, lines = user_delimiter(Tmpl(chatml), vision=True, media_marker=MARKER, image_url=URL,
                              tokenize=_toy_tokenize(True))       # `user\n` merges: the tokens never match
    assert d is None and any("tokens occur 0 times" in line for line in lines)

    def dead_tokenize(text, **kw):
        raise ConnectionRefusedError("nope")

    d, lines = user_delimiter(Tmpl(chatml), vision=False, media_marker=None, image_url=None, tokenize=dead_tokenize)
    assert d is None and any("could not tokenize" in line and "/tokenize" in line for line in lines)


def test_a_dead_template_endpoint_is_a_warning_not_a_crash():
    def boom(url, payload):
        raise ConnectionRefusedError("nope")

    d, lines = user_delimiter(TemplateClient("http://x", http_post=boom), vision=False, media_marker=None,
                              image_url=None)
    assert d is None and any("ConnectionRefusedError" in line for line in lines)
    d, lines = user_delimiter(TemplateClient("http://x", http_post=boom), vision=False, media_marker=None,
                              image_url=None, pinned="<|im_start|>user")
    assert d == "<|im_start|>user" and any(line.startswith("WARNING") and "/apply-template" in line for line in lines)
