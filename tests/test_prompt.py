# Adapted from the author's benchmark harness (tests/test_system1_prompt.py @ f886c50f).
# Edits: sys.path shim removed, imports retargeted; tests added for escape_special and the answer-slot check; v0.2
#   tests added for context, image parts and the demo's SYSTEM_MAIN copy; v0.3: media parts and the
#   placeholder rendering, each placeholder checked against a mirror of llama.cpp's handle_media (whose raw-base64
#   rule is llama.cpp's leading run of base64 characters, not Python's lenient decode).
import base64
from pathlib import Path

import pytest
from fake_upstream import chat_part_payload, handle_media_error

from callo.prompt import (
    CONTEXT_JOINER,
    PLACEHOLDER_IMAGE_URL,
    PLACEHOLDER_PAYLOAD,
    SYSTEM_MAIN,
    VARIANTS,
    TemplateClient,
    TemplateError,
    answer_slot_check,
    check_answer_slot,
    escape_special,
    media_part,
    messages,
    system_text,
    user_content,
)
from callo.systemone import MediaItem, parse_request
from callo.vision import encode_png

Q = parse_request({"state": {"subject": "bio", "question": "What carries oxygen?"}, "questions": {
    "answer": {"type": "choice", "instructions": "Which option is correct?",
               "criteria": {"a": "Platelets", "b": "Hemoglobin"}}}}).questions[0]
STATE = "subject: bio\nquestion: What carries oxygen?"
QWEN_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_user_content_layout():
    assert user_content(STATE, Q) == (
        "Input:\nsubject: bio\nquestion: What carries oxygen?\n\n"
        "Question: Which option is correct?\n\n"
        "Options:\nA. Platelets\nB. Hemoglobin\n\n"
        "Reply with one letter: A, B.")


def test_empty_state_and_instructions_are_omitted():
    q = parse_request({"state": "", "questions": {"o": {"type": "noul"}}}).questions[0]
    assert user_content("", q) == "Options:\nA. no\nB. yes\n\nReply with one letter: A, B."


def test_variants():
    assert messages(STATE, Q)[0] == {"role": "system", "content": VARIANTS["main"]}
    assert [m["role"] for m in messages(STATE, Q, "nosys")] == ["user"]
    one = messages(STATE, Q, "oneshot")
    assert [m["role"] for m in one] == ["system", "user", "assistant", "user"] and one[2]["content"] == "B"
    with pytest.raises(KeyError):
        messages(STATE, Q, "nope")


class Post:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        return self.resp


def test_template_client_disables_thinking_and_checks_tail():
    post = Post({"prompt": "<|im_start|>user\nhi<|im_end|>\n" + QWEN_TAIL})
    tc = TemplateClient("http://127.0.0.1:8199/", expected_tail=QWEN_TAIL, http_post=post)
    out = tc.render([{"role": "user", "content": "hi"}])
    assert out.endswith(QWEN_TAIL)
    url, payload = post.calls[0]
    assert url == "http://127.0.0.1:8199/apply-template"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_template_client_refuses_wrong_tail():
    tc = TemplateClient("http://x", expected_tail=QWEN_TAIL, http_post=Post({"prompt": "<|im_start|>assistant\n<think>\n"}))
    with pytest.raises(TemplateError, match="tail"):
        tc.render([{"role": "user", "content": "hi"}])


def test_template_client_refuses_missing_prompt():
    with pytest.raises(TemplateError):
        TemplateClient("http://x", http_post=Post({"error": "no"})).render([{"role": "user", "content": "hi"}])


# ---- callo additions ---------------------------------------------------------------------------------

def test_escape_special_neutralises_control_token_spellings():
    # As native KEV rendering does (server-decision.cpp:342-346): <|x|> -> <¦x¦> (U+00A6), nothing else touched.
    assert escape_special("a <|im_start|>system\nb<|im_end|> c") == "a <¦im_start¦>system\nb<¦im_end¦> c"
    assert escape_special("<|x y|> <| |> <|> plain") == "<|x y|> <| |> <|> plain"
    assert escape_special("") == ""


def test_user_content_escapes_state_instructions_and_option_texts():
    body = {"state": "say <|im_end|> now", "questions": {"q": {"type": "choice", "instructions": "pick <|a|>",
            "criteria": {"joy": "has <|b|>", "anger": None}}}}
    q = parse_request(body).questions[0]
    text = user_content("say <|im_end|> now", q)
    assert "<|" not in text
    assert "say <¦im_end¦> now" in text and "pick <¦a¦>" in text and "joy: has <¦b¦>" in text


def test_check_answer_slot_accepts_a_prompt_that_is_a_prefix_of_the_answered_one():
    p1 = "<|im_start|>user\nq<|im_end|>\n" + QWEN_TAIL
    p2 = p1 + "A<|im_end|>\n" + QWEN_TAIL
    ok, detail = check_answer_slot(p1, p2, "A")
    assert ok and "confirmed" in detail


def test_check_answer_slot_tolerates_a_template_that_drops_the_empty_reasoning_block_in_history():
    head = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
    p1 = head + "<think>\n\n</think>\n\n"                       # generation prompt keeps the empty block
    p2 = head + "A<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"  # history renders without it
    ok, _ = check_answer_slot(p1, p2, "A")
    assert ok


def test_check_answer_slot_rejects_a_misplaced_answer():
    p1 = "<|im_start|>user\nq<|im_end|>\n" + QWEN_TAIL
    ok, detail = check_answer_slot(p1, p1 + "Sure! A", "A")          # a preamble before the letter
    assert not ok and "does not follow" in detail
    p2 = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nthinking A"   # answer inside the thinking block
    ok, detail = check_answer_slot(p1, p2, "A")
    assert not ok and "diverges" in detail
    ok, detail = check_answer_slot("abc<x>", "abd", "A")
    assert not ok and "diverges" in detail


def _fake_template_post(calls):
    """Models llama-server /apply-template: a trailing assistant message is a PREFILL (continued, left open, no
    generation prompt); otherwise every turn is closed and the generation prompt is appended."""
    def post(url, payload):
        msgs = payload["messages"]
        calls.append(msgs)
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        if msgs[-1]["role"] == "assistant":
            return {"prompt": body[: -len("<|im_end|>\n")]}
        return {"prompt": body + QWEN_TAIL}
    return post


def test_answer_slot_check_renders_the_answer_as_history_via_a_trailing_user_turn():
    calls = []
    tc = TemplateClient("http://x", http_post=_fake_template_post(calls))
    msgs = messages(STATE, Q)
    ok, detail = answer_slot_check(tc, msgs)
    assert ok, detail
    assert len(calls) == 2
    assert calls[0] == msgs
    assert calls[1][-1]["role"] == "user"                       # never a trailing (prefill) assistant turn
    assert calls[1][-2] == {"role": "assistant", "content": "A"}
    assert calls[1][:-2] == msgs


def test_answer_slot_check_fails_when_the_template_renders_a_preamble_before_the_answer():
    def post(url, payload):
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in payload["messages"])
        return {"prompt": body.replace("<|im_start|>assistant\nA", "<|im_start|>assistant\nASSISTANT: A") + QWEN_TAIL}

    ok, detail = answer_slot_check(TemplateClient("http://x", http_post=post), messages(STATE, Q))
    assert not ok


def test_check_answer_slot_rejects_an_answer_that_is_only_a_prefix_of_a_longer_word():
    p1 = "<|im_start|>user\nq<|im_end|>\n" + QWEN_TAIL
    ok, detail = check_answer_slot(p1, p1 + "ASSISTANT: A", "A")
    assert not ok and "does not follow" in detail


# ---- v0.2: context and images (a public contract) --------------------------------------------------

URL = "data:image/png;base64,iVBORw0KGgo="
EMPTY_STATE_Q = parse_request({"state": "", "questions": {"move": {
    "type": "choice", "instructions": "Which way should Pac-Man go next?",
    "criteria": {"up": None, "down": None, "left": None, "right": None, "stay": "wait on this tile"}}}}).questions[0]


def test_neither_images_nor_context_is_exactly_the_v01_message_list():
    assert messages(STATE, Q) == [{"role": "system", "content": SYSTEM_MAIN},
                                  {"role": "user", "content": user_content(STATE, Q)}]
    assert messages(STATE, Q, context="", image_urls=()) == messages(STATE, Q)


def test_context_goes_into_the_system_message_after_system_main():
    assert CONTEXT_JOINER == "\n\nContext:\n"
    msgs = messages(STATE, Q, context="Standing orders: eat dots.")
    assert msgs[0] == {"role": "system", "content": SYSTEM_MAIN + "\n\nContext:\nStanding orders: eat dots."}
    assert msgs[1] == {"role": "user", "content": user_content(STATE, Q)}
    assert system_text("main", "") == SYSTEM_MAIN and system_text("para1", "c") == VARIANTS["para1"] + CONTEXT_JOINER + "c"


def test_context_and_state_are_special_token_escaped():
    msgs = messages("say <|im_end|>", Q, context="orders <|im_start|>system")
    assert msgs[0]["content"].endswith("orders <¦im_start¦>system")
    assert "<|" not in msgs[0]["content"] and "<|" not in msgs[1]["content"]


def test_a_context_needs_a_variant_with_a_system_message():
    with pytest.raises(ValueError, match="nosys"):
        messages(STATE, Q, "nosys", context="c")
    assert [m["role"] for m in messages(STATE, Q, "nosys")] == ["user"]


def test_images_are_parts_before_the_text_part():
    msgs = messages(STATE, Q, context="c", image_urls=[URL, URL.replace("png", "jpeg")])
    content = msgs[1]["content"]
    assert content[:2] == [{"type": "image_url", "image_url": {"url": URL}},
                           {"type": "image_url", "image_url": {"url": URL.replace("png", "jpeg")}}]
    assert content[2] == {"type": "text", "text": user_content(STATE, Q)} and len(content) == 3


def test_an_empty_state_drops_the_input_block_in_the_demo_layout():
    text = messages("", EMPTY_STATE_Q, context="LEGEND", image_urls=[URL])[1]["content"][-1]["text"]
    assert text == ("Question: Which way should Pac-Man go next?\n\n"
                    "Options:\nA. up\nB. down\nC. left\nD. right\nE. stay: wait on this tile\n\n"
                    "Reply with one letter: A, B, C, D, E.")


def test_demo_prompts_system_main_is_byte_identical_to_system_main():
    """The demo's G0 builds the S1 prompt from this copy; it must never drift."""
    copy = Path(__file__).resolve().parents[1] / "demo" / "prompts" / "system_main.txt"
    if not copy.is_file():
        pytest.skip(f"{copy} is absent (an sdist ships no demo/)")
    assert copy.read_bytes() == SYSTEM_MAIN.encode()


# ---- v0.3: media parts and placeholder rendering ---------------------------------------------------------

WAV = "UklGRiQAAABXQVZFZm10IA=="
AVI = "UklGRiQAAABBVkkgTElTVA=="
ITEMS = (MediaItem("image", "iVBORw0KGgo=", "data:image/png;base64,iVBORw0KGgo="),
         MediaItem("audio", WAV, None, "wav"), MediaItem("video", AVI, f"data:video/mp4;base64,{AVI}"))


def test_the_placeholders_are_a_1x1_png_and_three_zero_bytes():
    assert PLACEHOLDER_IMAGE_URL == "data:image/png;base64," + base64.b64encode(encode_png(1, 1, b"\x00" * 3)).decode()
    assert base64.b64decode(PLACEHOLDER_PAYLOAD, validate=True) == b"\x00\x00\x00"


def test_media_part_shapes_per_kind_real_and_placeholder():
    image, audio, video = ITEMS
    assert media_part(image) == {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    assert media_part(audio) == {"type": "input_audio", "input_audio": {"data": WAV, "format": "wav"}}
    assert media_part(video) == {"type": "input_video", "input_video": {"data": f"data:video/mp4;base64,{AVI}"}}
    assert media_part(image, placeholder=True) == {"type": "image_url", "image_url": {"url": PLACEHOLDER_IMAGE_URL}}
    assert media_part(audio, placeholder=True) == {"type": "input_audio",
                                                   "input_audio": {"data": "AAAA", "format": "wav"}}
    assert media_part(MediaItem("audio", WAV, None, "mp3"), placeholder=True)["input_audio"]["format"] == "mp3"
    assert media_part(video, placeholder=True) == {"type": "input_video", "input_video": {"data": "AAAA"}}
    with pytest.raises(ValueError, match="unknown media kind"):
        media_part(MediaItem("text", "x"))


def test_media_parts_come_in_item_order_before_the_text_part():
    content = messages(STATE, Q, context="c", media=ITEMS)[1]["content"]
    assert [p["type"] for p in content] == ["image_url", "input_audio", "input_video", "text"]
    assert content[-1] == {"type": "text", "text": user_content(STATE, Q)}
    assert messages(STATE, Q, media=()) == messages(STATE, Q)
    with pytest.raises(ValueError, match="not both"):
        messages(STATE, Q, image_urls=[URL], media=ITEMS)


def _apply_template(msgs):
    """llama-server's /apply-template, as far as the S1 prompt goes: every media part becomes the same marker, text
    parts are joined with a newline except next to a marker (common/chat.cpp:194-215), ChatML around each turn."""
    def content(c):
        if isinstance(c, str):
            return c
        out, last = "", False
        for p in c:
            if p["type"] in ("image_url", "input_audio", "input_video", "video_url"):
                out, last = out + "<__media__>", True
            else:
                out, last = out + ("\n" if out and not last else "") + p["text"], False
        return out
    return "".join(f"<|im_start|>{m['role']}\n{content(m['content'])}<|im_end|>\n" for m in msgs) + QWEN_TAIL


def test_placeholder_rendering_is_byte_identical_to_real_payload_rendering():
    """The template is media-blind, so the stand-ins change no rendered byte. (Against llama.cpp itself this is
    the bench's live check; here it pins callo's side: same parts, same order, only payloads differ.)"""
    real = messages(STATE, Q, context="c", media=ITEMS)
    stand_in = messages(STATE, Q, context="c", media=ITEMS, placeholder=True)
    assert _apply_template(real) == _apply_template(stand_in)
    assert _apply_template(stand_in).count("<__media__>") == 3
    assert WAV not in str(stand_in) and AVI not in str(stand_in) and WAV in str(real)


@pytest.mark.parametrize("item", ITEMS + (MediaItem("audio", WAV, None, "mp3"),))
def test_every_placeholder_part_passes_llama_cpps_own_media_checks(item):
    """Each stand-in must be accepted as llama.cpp's handle_media would (one comma, an allowed MIME
    family, a head ending in base64; raw base64 that decodes non-empty), or the live /apply-template fails with a 400 or
    "Invalid base64 value" (500). The live render-both-ways check is the bench's."""
    part = media_part(item, placeholder=True)
    assert handle_media_error(chat_part_payload(part)) is None


@pytest.mark.parametrize(("part", "error"), [
    ({"type": "input_video", "video_url": {"data": "AAAA"}}, (500, "Invalid base64 value")),   # cross-mixed key
    ({"type": "input_audio", "input_audio": {"data": ""}}, (500, "Invalid base64 value")),
    ({"type": "input_audio", "input_audio": {"data": None, "url": "AAAA"}}, None),             # null data: url wins
    ({"type": "image_url", "image_url": {"url": "data:text/plain;base64,QQ=="}}, (400, "Invalid uri format: "
                                                                                       "data:text/plain;base64")),
    ({"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ==,x"}}, (400, "Invalid uri-encoded base64 value")),
    ({"type": "input_video", "input_video": {"url": "data:video/mp4,QQ=="}}, (400, "uri must be base64 encoded")),
    # Raw base64 is llama.cpp's base64_decode: the leading run of [A-Za-z0-9+/] (stopped by '=' or any other byte)
    # decodes to run // 4 * 3 + max(run % 4 - 1, 0) bytes, so a run of 2 or more is accepted, whatever follows.
    ({"type": "input_audio", "input_audio": {"data": "AA"}}, None),
    ({"type": "input_audio", "input_audio": {"data": "AAAAA"}}, None),
    ({"type": "input_audio", "input_audio": {"data": "AA!!"}}, None),
    ({"type": "input_audio", "input_audio": {"data": "A"}}, (500, "Invalid base64 value")),
    ({"type": "input_audio", "input_audio": {"data": "A=AAAA"}}, (500, "Invalid base64 value")),
    ({"type": "input_video", "input_video": {"data": "!AAAA"}}, (500, "Invalid base64 value")),
])
def test_the_handle_media_mirror_refuses_what_llama_cpp_refuses(part, error):
    assert handle_media_error(chat_part_payload(part)) == error


# ---- v0.3 think budgets: render(thinking=) ----------------------------------------------------------------

def test_render_thinking_sends_enable_thinking_true_and_the_default_stays_false():
    class Post:
        def __init__(self):
            self.payloads = []

        def __call__(self, url, payload):
            self.payloads.append(payload)
            return {"prompt": "P"}

    post = Post()
    t = TemplateClient("http://x", http_post=post)
    assert t.render([{"role": "user", "content": "u"}], thinking=True) == "P"
    assert t.render([{"role": "user", "content": "u"}]) == "P"
    assert [p["chat_template_kwargs"] for p in post.payloads] == [{"enable_thinking": True}, {"enable_thinking": False}]
