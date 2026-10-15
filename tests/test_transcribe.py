"""Pinned /v1/audio/transcriptions: the gate rows, the chat
body llama-server's own conversion builds (tools/server/server-chat.cpp:651-706 @ 4ebdf2c), and the reply."""
import base64
import json
import struct
from decimal import Decimal

import pytest
from helpers import form_field, form_file, multipart

from callo.av import TONE_WAV
from callo.transcribe import (
    ASR_HINT,
    FLT_MAX,
    FLT_MIN,
    REASONS,
    TRANSCRIPTION_PATHS,
    ULONG_MAX,
    Translation,
    is_lfm2,
    numbers_ok,
    plan,
    reshape,
)

CT = "multipart/form-data; boundary=XyZ"
AUDIO = {"type": "input_audio", "input_audio": {"data": base64.b64encode(TONE_WAV).decode(), "format": "wav"}}
WAV = form_file(TONE_WAV)


def translated(*fields: bytes, lfm2: bool = False, data: bytes = TONE_WAV) -> Translation:
    t = plan([CT], multipart(form_file(data), *fields), lfm2=lfm2)
    assert isinstance(t, Translation), t
    return t


def test_the_routes_and_the_reasons_are_the_spec_rows():
    assert TRANSCRIPTION_PATHS == {"/v1/audio/transcriptions", "/audio/transcriptions"}   # server.cpp:271-272
    assert REASONS == ("no_audio", "no_pinning", "render", "chunked", "large", "encoded", "not_multipart", "file",
                       "fields", "response_format", "stream", "numbers")
    assert ASR_HINT == "callo:asr"


def test_the_default_preset_is_upstreams_user_text_then_the_audio():
    t = translated()
    assert t.body == {"messages": [{"role": "user", "content": [{"type": "text", "text": "Transcribe audio to text"},
                                                                AUDIO]}], "stream": False}
    assert t.fields == {}


@pytest.mark.parametrize("fields, text", [
    ({"prompt": "Names: Ada, Linus."}, "Names: Ada, Linus."),
    ({"prompt": ""}, "Transcribe audio to text"),          # an empty prompt is the preset's (server-chat.cpp:672-674)
    ({"language": "de"}, "Transcribe audio to text (language: de)"),
    ({"prompt": "P", "language": "fr"}, "P (language: fr)"),
    ({"language": ""}, "Transcribe audio to text"),
])
def test_prompt_then_language_then_the_audio_last(fields, text):
    t = translated(*[form_field(k, v) for k, v in fields.items()])
    assert t.body["messages"] == [{"role": "user", "content": [{"type": "text", "text": text}, AUDIO]}]


def test_a_nul_in_prompt_is_kept_as_upstreams_std_string_keeps_it():
    # server-chat.cpp:667-680 @ 4ebdf2c: `prompt` is a std::string (= and +=), never a C string; only `language` is
    t = translated(form_field("prompt", b"a\x00b"))
    assert t.body["messages"][0]["content"][0] == {"type": "text", "text": "a\x00b"}


def test_the_lfm2_preset_is_a_system_message_and_no_text_part():
    assert translated(lfm2=True).body["messages"] == [{"role": "system", "content": "Perform ASR."},
                                                     {"role": "user", "content": [AUDIO]}]
    assert translated(form_field("language", "en"), lfm2=True).body["messages"][1]["content"] == [
        {"type": "text", "text": " (language: en)"}, AUDIO]                 # upstream's own leading space
    assert translated(form_field("prompt", "P"), lfm2=True).body["messages"][1]["content"] == [
        {"type": "text", "text": "P"}, AUDIO]


def test_every_field_is_copied_verbatim_as_a_string_and_three_are_typed():
    t = translated(form_field("model", "whisper-1"), form_field("user", "alice"), form_field("top_p", "0.9"),
                   form_field("seed", "1"), form_field("response_format", "json"), form_field("stream", "false"),
                   form_field("max_tokens", "007"), form_field("temperature", "0"),
                   form_field("messages", "ignored"), form_field("id_slot", "3"))
    body = t.body
    assert {k: body[k] for k in ("model", "user", "top_p", "seed", "response_format")} == {
        "model": "whisper-1", "user": "alice", "top_p": "0.9", "seed": "1", "response_format": "json"}
    assert body["stream"] is False and body["max_tokens"] == 7 and type(body["max_tokens"]) is int
    assert body["temperature"] == 0.0 and type(body["temperature"]) is float
    assert body["messages"][0]["role"] == "user"                       # a form `messages` is overwritten
    assert "id_slot" not in body                                       # the proxy stamps the routed int
    assert t.fields["id_slot"] == "3" and t.fields["user"] == "alice"


@pytest.mark.parametrize("data, fmt", [(TONE_WAV, "wav"), (b"ID3" + b"\x00" * 64, "mp3"),
                                       (b"\xff\xfb\x90\x00" + b"\x00" * 64, "mp3"), (b"fLaC" + b"\x00" * 64, "flac")])
def test_the_format_is_the_sniffed_one(data, fmt):
    assert translated(data=data).body["messages"][0]["content"][1]["input_audio"]["format"] == fmt


def test_curl_and_httpx_requests_translate():
    curl = multipart(form_file(TONE_WAV, ctype="audio/x-wav"), form_field("model", "whisper-1"),
                     boundary=b"------------------------d74496d66958873e")
    t = plan(["multipart/form-data; boundary=------------------------d74496d66958873e"], curl, lfm2=False)
    assert isinstance(t, Translation) and t.body["model"] == "whisper-1"
    httpx = multipart(form_field("model", "whisper-1"), form_file(TONE_WAV, filename="Aufnahme März 1.wav"),
                      boundary=b"9b2c1f0e8d7a6b5c4d3e2f1a0b9c8d7e")
    t = plan(['multipart/form-data; boundary="9b2c1f0e8d7a6b5c4d3e2f1a0b9c8d7e"'], httpx, lfm2=False)
    assert isinstance(t, Translation) and t.body["messages"][0]["content"][1] == AUDIO


GATE = [
    ("no_content_type", [], multipart(WAV), "not_multipart"),
    ("two_content_types", [CT, CT], multipart(WAV), "not_multipart"),
    ("json_content_type", ["application/json"], multipart(WAV), "not_multipart"),
    ("lf_body", [CT], multipart(WAV).replace(b"\r\n", b"\n"), "not_multipart"),
    ("no_file", [CT], multipart(form_field("prompt", "x")), "file"),
    ("file_as_field", [CT], multipart(form_field("file", TONE_WAV)), "file"),          # curl -F 'file=<a.wav'
    ("empty_filename", [CT], multipart(form_file(TONE_WAV, filename="")), "file"),
    ("two_files", [CT], multipart(WAV, WAV), "file"),
    ("empty_file", [CT], multipart(form_file(b"")), "file"),
    ("png_file", [CT], multipart(form_file(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)), "file"),
    ("adts_file", [CT], multipart(form_file(b"\xff\xf1" + b"\x00" * 32)), "file"),       # media.py's ADTS deviation
    ("other_name", [CT], multipart(form_file(TONE_WAV, name="audio")), "file"),
    ("second_file_part", [CT], multipart(WAV, form_file(TONE_WAV, name="extra")), "fields"),
    ("repeated_field", [CT], multipart(WAV, form_field("a", "1"), form_field("a", "2")), "fields"),
    ("bad_utf8_value", [CT], multipart(WAV, form_field("a", b"\xff")), "fields"),
    ("bad_utf8_name", [CT], multipart(WAV, b'Content-Disposition: form-data; name="\xff"\r\n\r\n1'), "fields"),
    ("nul_in_language", [CT], multipart(WAV, form_field("language", b"de\x00x")), "fields"),   # %s of c_str()
    ("text_format", [CT], multipart(WAV, form_field("response_format", "text")), "response_format"),
    ("verbose_json", [CT], multipart(WAV, form_field("response_format", "verbose_json")), "response_format"),
    ("upper_json", [CT], multipart(WAV, form_field("response_format", "JSON")), "response_format"),
    ("stream_true", [CT], multipart(WAV, form_field("stream", "true")), "stream"),
    ("stream_True", [CT], multipart(WAV, form_field("stream", "True")), "stream"),
    ("stream_empty", [CT], multipart(WAV, form_field("stream", "")), "stream"),
    ("temperature_1e40", [CT], multipart(WAV, form_field("temperature", "1e40")), "numbers"),
    ("max_tokens_newline", [CT], multipart(WAV, form_field("max_tokens", "12\n")), "numbers"),
    ("max_tokens_5000_digits", [CT], multipart(WAV, form_field("max_tokens", "9" * 5000)), "numbers"),
]


@pytest.mark.parametrize("cts, body, why", [g[1:] for g in GATE], ids=[g[0] for g in GATE])
def test_gate_rows(cts, body, why):
    assert plan(cts, body, lfm2=False) == why
    assert why in REASONS


def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


FLT_MIN_EXACT = str(Decimal(FLT_MIN))                                  # 2^-126, all 88 significant digits


@pytest.mark.parametrize("t", ["0", "-0", "0.0", "0e-400", "0." + "0" * 400, "1", "0.7", "-1.5", "2e0", "1E-3",
                               "3.4028234663852886e38", "-3.4028234663852886e38", FLT_MIN_EXACT, "1.17549436e-38"])
def test_temperatures_std_stof_reads_the_same_way_translate(t):
    assert numbers_ok({"temperature": t})
    got = translated(form_field("temperature", t)).body["temperature"]
    assert type(got) is float and got == f32(float(t)) and f32(got) == got   # the float32 stof gives (short inputs)
    assert got == 0 or FLT_MIN <= abs(got) <= FLT_MAX


@pytest.mark.parametrize("t", ["1_0", "1e40", "1e-50", "1e-38", "3.5e38", "nan", "inf", " 1", "1 ", "1.", ".5", "0x1",
                               "+1", "１", "",
                               "1e-400", "0." + "0" * 400 + "1",        # underflow to 0.0 in the double: stof ERANGE
                               "1.1754943508222875e-38",                # below 2^-126, though its double IS FLT_MIN
                               "1e-99999999999999999999", "0e-99999999999999999999"])   # beyond Decimal: refused
def test_temperatures_outside_stof_agreement_pass_through(t):
    assert not numbers_ok({"temperature": t})


@pytest.mark.parametrize("t, want, double_rounding_differs", [
    # 1 + 2^-24 is the midpoint of float32 1.0 and 1 + 2^-23; this is a hair above it, but its nearest double IS the
    # midpoint, so decimal -> double -> float32 ties to even (1.0) where strtof rounds once, up to 1 + 2^-23
    ("1.000000059604644776257986738", 1 + 2**-23, True),
    ("-1.000000059604644776257986738", -(1 + 2**-23), True),
    ("1.000000059604644775390625", 1.0, False),                         # the exact midpoint: ties to even
    ("1.000000059604644775" + "3" * 20, 1.0, False),                    # a hair below it
])
def test_temperature_is_the_float32_stof_gives_not_a_double_rounding(t, want, double_rounding_differs):
    assert numbers_ok({"temperature": t})
    assert (f32(float(t)) != want) is double_rounding_differs
    assert translated(form_field("temperature", t)).body["temperature"] == want


def test_max_tokens_follows_stoul_and_never_raises():
    for ok in ("0", "128", str(ULONG_MAX), "0" * 5000 + "7"):
        assert numbers_ok({"max_tokens": ok}), ok[:30]
    for bad in ("12\n", "-1", "1.0", "", "1e3", " 1", str(ULONG_MAX + 1), "9" * 5000, "１"):
        assert not numbers_ok({"max_tokens": bad}), bad[:30]
    assert translated(form_field("max_tokens", "0" * 5000 + "7")).body["max_tokens"] == 7


def test_lfm2_is_upstreams_two_substring_test():
    # common/parsers/lfm2.cpp:5-8 @ 4ebdf2c: both markers anywhere in the default template's source
    assert is_lfm2("{{ '<|tool_list_start|>' }} ... {{ '<|tool_list_end|>' }}")
    assert not is_lfm2("<|tool_list_start|> only") and not is_lfm2("") and not is_lfm2(None)


REPLY = {"choices": [{"message": {"role": "assistant", "content": "hello world", "reasoning_content": "hm"}}],
         "usage": {"completion_tokens": 3, "prompt_tokens": 120, "total_tokens": 123,
                   "prompt_tokens_details": {"cached_tokens": 5}}, "timings": {"prompt_n": 120}}


def test_the_reply_is_upstreams_transcription_shape():
    # to_json_oaicompat_asr (server-task.cpp:715-728) from the chat usage (:365-372); reasoning dropped
    assert reshape(json.dumps(REPLY).encode()) == {
        "type": "transcript.text.done", "text": "hello world",
        "usage": {"type": "tokens", "input_tokens": 120, "output_tokens": 3, "total_tokens": 123,
                  "input_tokens_details": {"cached_tokens": 5}}}


def test_cached_tokens_absent_is_0_and_an_empty_transcript_is_a_transcript():
    r = {"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 0, "prompt_tokens": 9,
                                                              "total_tokens": 9}}
    out = reshape(json.dumps(r).encode())
    assert out["text"] == "" and out["usage"]["input_tokens_details"] == {"cached_tokens": 0}
    bare = reshape(json.dumps({"choices": [{"message": {"content": "t"}}]}).encode())
    assert bare["usage"] == {"type": "tokens", "input_tokens": None, "output_tokens": None, "total_tokens": None,
                             "input_tokens_details": {"cached_tokens": 0}}


@pytest.mark.parametrize("raw", [b"not json", b"[]", b"{}", b'{"choices": []}', b'{"choices": ["x"]}',
                                 b'{"choices": [{"message": {}}]}', b'{"choices": [{"message": {"content": null}}]}',
                                 b'{"choices": [{"message": {"content": ["parts"]}}]}'])
def test_a_reply_without_a_content_string_is_bad(raw):
    assert reshape(raw) is None
