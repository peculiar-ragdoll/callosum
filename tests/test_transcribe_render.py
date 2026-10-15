"""The startup render check of pinned /v1/audio/transcriptions: callo's
content parts must render through the upstream's /apply-template exactly as llama-server's own string form, or
translation is off for the process. Unit tests of the decision, plus `setup` through the fake upstream."""
import base64

import pytest
from fake_upstream import MEDIA_MARKER, FakeUpstream

from callo.scorer import make_http_post
from callo.transcribe import (
    PLACEHOLDER_WAV,
    REASONS,
    RENDER_TEXT,
    Transcription,
    render_check,
    render_pairs,
    setup,
    startup_lines,
    status,
    unrouted_counter,
)

MARK = "<__media__>"


def renderer(join):
    """A fake /apply-template: `join(parts)` renders a list content; a string content renders as it is."""
    calls = []

    def render(msgs):
        calls.append(msgs)
        return "".join(f"<{m['role']}>" + (m["content"] if isinstance(m["content"], str) else join(m["content"]))
                       for m in msgs)

    return render, calls


def jinja(parts):       # common/chat.cpp:194-220: no separator around a media marker
    return "".join(MARK if p["type"] == "input_audio" else p["text"] for p in parts)


def legacy(parts):      # common/chat.cpp:1375-1395 (--no-jinja): every part joined with "\n"
    return "\n".join(MARK if p["type"] == "input_audio" else p["text"] for p in parts)


def trimmed(parts):     # StepFun's trim_all_content on each part (common/chat.cpp:1066-1078)
    return "".join(MARK if p["type"] == "input_audio" else p["text"].strip() for p in parts)


def test_the_jinja_join_turns_translation_on():
    render, calls = renderer(jinja)
    assert render_check(render, MARK, lfm2=False) is None
    assert len(calls) == 4                     # a text part plus audio, and audio alone, each in both forms
    assert calls[0][-1]["content"][0] == {"type": "text", "text": RENDER_TEXT} and calls[1][-1]["content"] == "T " + MARK
    assert [p["type"] for p in calls[2][-1]["content"]] == ["input_audio"] and calls[3][-1]["content"] == MARK
    audio = calls[0][-1]["content"][1]["input_audio"]
    assert base64.b64decode(audio["data"]) == PLACEHOLDER_WAV and len(PLACEHOLDER_WAV) == 44 and audio["format"] == "wav"


@pytest.mark.parametrize("join", [legacy, trimmed], ids=["no_jinja", "stepfun_trim"])
def test_a_template_that_joins_or_trims_parts_turns_translation_off(join):
    render, _ = renderer(join)
    why = render_check(render, MARK, lfm2=False)
    assert why is not None and "text + audio" in why and "differently" in why


def no_jinja(parts):
    """common_chat_templates_apply_legacy (common/chat.cpp:1381-1394 @ 4ebdf2c), line for line: the content starts
    empty (a parts message has none), and every part after the first is preceded by "\\n", a media marker included."""
    content = ""
    for p in parts:
        if content:
            content += "\n"
        content += MARK if p["type"] == "input_audio" else p["text"]
    return content


@pytest.mark.parametrize("lfm2", [False, True], ids=["default", "lfm2"])
def test_a_no_jinja_server_turns_translation_off(lfm2):
    # upstream's own transcription prompt is the string "T <marker>"; callo's parts render as
    # "T \n<marker>" on a --no-jinja server, so the chat callo sends would differ from llama-server's. The audio-alone
    # pair renders the same there; the text + audio pair is the one that must catch it, with both renders quoted.
    render, _ = renderer(no_jinja)
    (_, (alone_parts, alone_string)) = render_pairs(MARK, lfm2=lfm2)
    assert render(alone_parts) == render(alone_string)
    why = render_check(render, MARK, lfm2=lfm2)
    assert why is not None and "the text + audio form differently" in why
    parts_tail, string_tail = repr("T \n" + MARK)[1:], repr("T " + MARK)[1:]     # each render's end, as quoted
    assert "parts end '<" in why and parts_tail + ", the string ends" in why and why.endswith(string_tail)


def test_an_audio_alone_difference_is_caught_too():
    render, _ = renderer(lambda parts: jinja(parts) if len(parts) > 1 else "\n" + MARK)
    assert "audio alone" in render_check(render, MARK, lfm2=True)


def test_the_lfm2_preset_renders_its_system_message_in_both_forms():
    render, calls = renderer(jinja)
    assert render_check(render, MARK, lfm2=True) is None
    assert all(c[0] == {"role": "system", "content": "Perform ASR."} for c in calls)
    render, calls = renderer(jinja)
    render_check(render, MARK, lfm2=False)
    assert all(c[0]["role"] == "user" for c in calls)          # the default preset has no system message


def test_a_render_that_fails_turns_translation_off_and_never_raises():
    def boom(msgs):
        raise OSError("connection refused")
    why = render_check(boom, MARK, lfm2=False)
    assert why.startswith("the render check could not render") and "OSError: connection refused" in why


@pytest.mark.parametrize("prompt", ["<|im_start|>assistant\n", ""], ids=["ignores_messages", "empty"])
def test_a_render_without_the_media_marker_turns_translation_off(prompt):
    # an /apply-template that ignores `messages` renders both forms alike; equality alone would
    # turn translation on. The parts render must be non-empty and hold the marker before equality counts.
    def constant(msgs):
        return prompt
    why = render_check(constant, MARK, lfm2=False)
    assert why is not None and "text + audio" in why and "media marker" in why and repr(MARK) in why
    render, _ = renderer(jinja)
    assert "without the media marker ''" in render_check(render, "", lfm2=False)    # an empty marker proves nothing


def test_a_render_that_drops_the_marker_only_for_audio_alone_is_caught():
    render, _ = renderer(jinja)

    def drops_alone(msgs):   # the audio-alone pair's two forms render alike ("<user>"), both without the marker
        out = render(msgs)
        return out if RENDER_TEXT in out else out.replace(MARK, "")

    assert render_check(drops_alone, MARK, lfm2=False).startswith("the audio alone form renders without")


def test_setup_through_an_apply_template_that_ignores_the_messages():
    up = FakeUpstream(audio=True).start()
    up.routes["/apply-template"] = lambda u, rec, h: h._reply(200, {"prompt": "<|im_start|>assistant\n"})
    try:
        off = setup(up.url, None, MEDIA_MARKER, http_post=make_http_post()).off
        assert off is not None and "media marker" in off
    finally:
        up.stop()


@pytest.mark.parametrize("reply", [
    lambda h: h._reply(200, b"<html>not json</html>", {"Content-Type": "text/html"}),
    lambda h: h._reply(200, {"result": "no prompt here"}),
    lambda h: h._reply(200, {"prompt": 42}),
    lambda h: h._reply(200, ["a", "list"]),
], ids=["not_json", "no_prompt", "prompt_not_a_string", "not_an_object"])
def test_setup_through_a_malformed_apply_template_reply_turns_translation_off(reply):
    # a 200 whose body is not the {"prompt": str} object fails the check and never raises.
    up = FakeUpstream(audio=True).start()
    up.routes["/apply-template"] = lambda u, rec, h: reply(h)
    try:
        off = setup(up.url, None, MEDIA_MARKER, http_post=make_http_post()).off
        assert off is not None and off.startswith("the render check could not render the text + audio form: ")
    finally:
        up.stop()


def test_setup_through_the_upstreams_apply_template():
    up = FakeUpstream(audio=True).start()
    try:
        t = setup(up.url, up.props["chat_template"], MEDIA_MARKER, http_post=make_http_post())
        assert t == Transcription(lfm2=False, off=None) and t.preset == "default"
        t = setup(up.url, "<|tool_list_start|>{{ messages }}<|tool_list_end|>", MEDIA_MARKER, http_post=make_http_post())
        assert t == Transcription(lfm2=True, off=None) and t.preset == "lfm2"
    finally:
        up.stop()


def test_setup_authenticates_with_callos_own_key():
    up = FakeUpstream(audio=True, api_key="k").start()
    try:
        assert setup(up.url, None, MEDIA_MARKER, http_post=make_http_post(api_key="k")).off is None
        off = setup(up.url, None, MEDIA_MARKER, http_post=make_http_post()).off
        assert off is not None and "401" in off
    finally:
        up.stop()


def test_status_names_the_counter_the_handler_increments():
    # /callo/stats.transcription. The handler tests no_audio, then no_pinning, then render.
    assert status(Transcription(False), pinning=True) == {"translate": True, "reason": None}
    assert status(None, pinning=True) == {"translate": False, "reason": "no_audio: the upstream reports no audio"}
    assert status(None, pinning=False)["reason"].startswith("no_audio: ")
    assert status(Transcription(False, "x"), pinning=True) == {"translate": False, "reason": "render: x"}
    both = status(Transcription(False, "x"), pinning=False)
    assert both["reason"].startswith("no_pinning: agent pinning is off") and "render check failed: x" in both["reason"]


def test_startup_lines_say_what_happens_to_a_transcription():
    assert startup_lines(None, pinning=True) == [                       # INFO: a text model
        "transcription: translation is OFF for this process; every /v1/audio/transcriptions passes through unpinned (counted as transcription_unrouted_no_audio): the upstream reports no audio"]
    (off,) = startup_lines(Transcription(False, "x"), pinning=True)
    assert off.startswith("WARNING: transcription: translation is OFF") and "transcription_unrouted_render" in off
    assert off.endswith(": x")
    (both,) = startup_lines(Transcription(False, "x"), pinning=False)     # the counter that really rises
    assert "transcription_unrouted_no_pinning" in both and "transcription_unrouted_render" not in both
    (on,) = startup_lines(Transcription(True), pinning=True)
    assert "translated to /v1/chat/completions" in on and "ASR preset lfm2" in on


def test_unrouted_counter_names_a_reason_row_and_refuses_anything_else():
    assert [unrouted_counter(r) for r in REASONS[:2]] == ["transcription_unrouted_no_audio",
                                                          "transcription_unrouted_no_pinning"]
    with pytest.raises(ValueError, match="not a"):
        unrouted_counter("no_audo")
