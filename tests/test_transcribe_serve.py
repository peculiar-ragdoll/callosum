"""`callo serve` and pinned /v1/audio/transcriptions: the preset from
/props.chat_template, the startup render check on every upstream with audio (native decision models included),
asr_preset, the startup line, and callo's own key for the check."""
import http.client
import json
import logging

import pytest
from fake_upstream import MEDIA_MARKER, QWEN_TAIL, FakeUpstream
from helpers import Running, form_field, form_file, get_json, multipart, tiny_gguf

from callo.av import TONE_WAV
from callo.cli import main

CT = "multipart/form-data; boundary=XyZ"
BODY = multipart(form_file(TONE_WAV), form_field("language", "en"))
REPLY = {"choices": [{"message": {"content": "hello world"}}],
         "usage": {"completion_tokens": 3, "prompt_tokens": 120, "total_tokens": 123,
                   "prompt_tokens_details": {"cached_tokens": 0}}}


def _legacy_apply_template(u, rec, h):
    """llama-server's --no-jinja path: every content part joined with "\\n" (common/chat.cpp:1375-1395)."""
    def content(c):
        return c if isinstance(c, str) else "\n".join(
            p.get("text", "") if p.get("type") == "text" else MEDIA_MARKER for p in c)
    h._reply(200, {"prompt": "".join(f"<|im_start|>{m['role']}\n{content(m['content'])}<|im_end|>\n"
                                     for m in rec.json["messages"]) + QWEN_TAIL})


def _serve(tmp_path, caplog, *flags, decision=None, template=None, apply_template=None, **fake):
    up = FakeUpstream(**fake).start()
    up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf", decision=decision))
    if template is not None:
        up.props["chat_template"] = template
    if apply_template is not None:
        up.routes["/apply-template"] = apply_template
    up.routes["/v1/chat/completions"] = lambda u, rec, h: h._reply(200, REPLY)
    up.routes["/v1/audio/transcriptions"] = lambda u, rec, h: h._reply(200, {"native": True})
    with caplog.at_level(logging.INFO, logger="callo"):
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"), *flags])
    assert r.port, "callo serve did not start"
    return up, r


def post(url, headers=None):
    host, port = url.removeprefix("http://").split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=10)
    c.request("POST", "/v1/audio/transcriptions", body=BODY, headers={"Content-Type": CT, **(headers or {})})
    r = c.getresponse()
    out = r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    c.close()
    return out


def test_serve_translates_on_an_audio_upstream_and_publishes_the_preset(tmp_path, caplog):
    up, r = _serve(tmp_path, caplog, audio=True)
    try:
        status, headers, data = post(r.url)
        assert status == 200 and json.loads(data)["text"] == "hello world"
        assert headers["x-callo-slot"] not in {"-", "0"}                 # an agent slot, never the reserved one
        assert up.last("/v1/chat/completions").json["messages"][0]["content"][0] == {
            "type": "text", "text": "Transcribe audio to text (language: en)"}
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["asr_preset"] == "default" and stats["counters"]["transcription_translated"] == 1
        assert stats["transcription"] == {"translate": True, "reason": None,
                                          "asr_home": int(headers["x-callo-slot"]), "asr_home_idle_s": 600.0}
    finally:
        assert r.finish() == 0
        up.stop()
    assert "transcription: /v1/audio/transcriptions is translated to /v1/chat/completions" in caplog.text


def test_no_audio_means_no_preset_and_a_passthrough(tmp_path, caplog):
    up, r = _serve(tmp_path, caplog)
    try:
        status, headers, data = post(r.url)
        assert (status, json.loads(data), headers["x-callo-slot"]) == (200, {"native": True}, "-")
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["asr_preset"] is None and stats["counters"]["transcription_unrouted_no_audio"] == 1
        assert stats["transcription"] == {"translate": False, "reason": "no_audio: the upstream reports no audio",
                                          "asr_home": None, "asr_home_idle_s": 600.0}
    finally:
        assert r.finish() == 0
        up.stop()
    assert "(counted as transcription_unrouted_no_audio): the upstream reports no audio" in caplog.text


def test_an_lfm2_template_selects_the_lfm2_preset(tmp_path, caplog):
    up, r = _serve(tmp_path, caplog, audio=True, template="{# <|tool_list_start|> <|tool_list_end|> #}{{ messages }}")
    try:
        assert post(r.url)[0] == 200
        msgs = up.last("/v1/chat/completions").json["messages"]
        assert msgs[0] == {"role": "system", "content": "Perform ASR."}
        assert msgs[1]["content"][0] == {"type": "text", "text": " (language: en)"}
        assert get_json(f"{r.url}/callo/stats")[2]["asr_preset"] == "lfm2"
    finally:
        assert r.finish() == 0
        up.stop()


def test_a_template_path_that_joins_parts_turns_translation_off_with_a_warning(tmp_path, caplog):
    up, r = _serve(tmp_path, caplog, audio=True, apply_template=_legacy_apply_template)
    try:
        status, headers, data = post(r.url)
        assert (status, json.loads(data), headers["x-callo-slot"]) == (200, {"native": True}, "-")
        assert up.last("/v1/audio/transcriptions").body == BODY
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["asr_preset"] == "default" and stats["counters"]["transcription_unrouted_render"] == 1
        assert stats["transcription"]["translate"] is False and stats["transcription"]["reason"].startswith("render: ")
    finally:
        assert r.finish() == 0
        up.stop()
    warnings = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any(m.startswith("WARNING: transcription: translation is OFF") and "transcription_unrouted_render" in m
               and "text + audio" in m for m in warnings)


def test_a_native_decision_model_translates_too(tmp_path, caplog):
    up, r = _serve(tmp_path, caplog, decision="kev", audio=True)          # callo's engine is off
    try:
        assert post(r.url)[0] == 200
        assert get_json(f"{r.url}/callo/stats")[2]["asr_preset"] == "default"
    finally:
        assert r.finish() == 0
        up.stop()


def test_the_render_check_uses_callos_key_and_a_keyless_client_gets_the_upstreams_401(tmp_path, caplog):
    up, r = _serve(tmp_path, caplog, "--upstream-api-key", "k", audio=True, api_key="k")
    try:
        status, headers, _ = post(r.url)
        assert status == 401 and headers["x-callo-slot"] not in {"-", "0"}   # translated, refused upstream, relayed
        assert "authorization" not in up.last("/v1/chat/completions").headers
        assert post(r.url, {"Authorization": "Bearer k"})[0] == 200
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["asr_preset"] == "default" and "transcription_unrouted_render" not in stats["counters"]
    finally:
        assert r.finish() == 0
        up.stop()


def test_a_wedged_render_check_times_out_and_turns_translation_off(tmp_path, caplog, monkeypatch):
    """The render check's calls carry a short timeout of their own, not --s1-timeout, so an
    upstream whose /apply-template hangs cannot hold serve's startup; the check fails closed (translation off)."""
    import time

    from callo import cli
    monkeypatch.setattr(cli, "TRANSCRIPTION_SETUP_TIMEOUT", 0.2)

    def wedged(u, rec, h):
        time.sleep(6)
        h._reply(200, {"prompt": ""})

    t0 = time.monotonic()
    up, r = _serve(tmp_path, caplog, decision="kev", audio=True, apply_template=wedged)   # native: only setup renders
    try:
        assert time.monotonic() - t0 < 5          # a margin for a slow runner; --s1-timeout would wait out the 6 s
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["asr_preset"] == "default" and stats["transcription"]["translate"] is False
        assert stats["transcription"]["reason"].startswith("render: the render check could not render the text + audio")
    finally:
        assert r.finish() == 0
        up.stop()


def test_asr_home_idle_reaches_the_router_and_the_stats(tmp_path, caplog):
    up, r = _serve(tmp_path, caplog, "--asr-home-idle", "0", audio=True)
    try:
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["transcription"]["asr_home_idle_s"] == 0.0
    finally:
        assert r.finish() == 0
        up.stop()


@pytest.mark.parametrize("value", ["-1", "inf", "nan"])
def test_a_bad_asr_home_idle_is_a_usage_error_from_main_and_a_refusal_in_serve(tmp_path, caplog, value):
    up = FakeUpstream(audio=True).start()
    try:
        with pytest.raises(SystemExit) as e:
            main(["serve", "--upstream", up.url, "--asr-home-idle", value])
        assert e.value.code == 2
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        with caplog.at_level(logging.INFO, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"),
                         "--asr-home-idle", value])
        assert r.finish() == 1 and r.port is None and not up.requests
        assert "--asr-home-idle must be a finite number of seconds, 0 or more" in caplog.text
    finally:
        up.stop()


def test_status_and_the_startup_line_follow_the_routers_pinning(tmp_path, caplog, monkeypatch):
    """The handler gates on `router.pinning`; /callo/stats.transcription and the startup line read the same value, even
    when cli's own decision says otherwise (here: pinning "on" for a 1-slot upstream, which the router refuses)."""
    from callo import cli
    monkeypatch.setattr(cli, "decide_agent_pinning", lambda info, probe: (True, []))
    up, r = _serve(tmp_path, caplog, audio=True, total_slots=1)
    try:
        status, headers, data = post(r.url)
        assert (status, json.loads(data)) == (200, {"native": True})
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["router"]["pinning"] is False and stats["counters"]["transcription_unrouted_no_pinning"] == 1
        assert stats["transcription"]["translate"] is False
        assert stats["transcription"]["reason"].startswith("no_pinning: ")
    finally:
        assert r.finish() == 0
        up.stop()
    assert "(counted as transcription_unrouted_no_pinning)" in caplog.text
