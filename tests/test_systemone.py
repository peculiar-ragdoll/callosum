# Adapted from the author's benchmark harness (tests/test_system1_schema.py @ f886c50f).
# Edits: sys.path shim removed, import retargeted; v0.2 tests added for images, state image parts and context; v0.3:
#   audio, videos and their state parts, MediaItem / Request.media / Request.kind, MAX_MEDIA,
#   native_media_refusal; the v0.2 "video and audio parts are not supported" test is rewritten (input_audio and
#   video_url are now taken out, only audio_url / video are refused) and the 8-image test reads the MAX_MEDIA
#   message; an image_url part without its image_url object is now a 422, so the "left alone" test uses an
#   "image" part. v0.3 wide-choice: the 27-option invalid case is valid now (two-letter labels); 256 stays invalid.
import base64
import hashlib
import json

import pytest

from callo.systemone import (
    MAX_MEDIA,
    THINK_DEADLINE_ERROR,
    InvalidRequest,
    MediaItem,
    answer_body,
    assign_choice_labels,
    media_kind,
    native_media_refusal,
    native_think_refusal,
    parse_request,
)

MMLU = {"state": {"subject": "security studies", "question": "Which did not emerge?"},
        "questions": {"answer": {"type": "choice", "instructions": "Which option correctly answers the question?",
                                 "criteria": {"a": "Private to government.", "b": "Increase.", "c": "Decrease.", "d": "Growth."}}}}


def test_letter_keys_label_themselves_and_show_description_only():
    q = parse_request(MMLU).questions[0]
    assert q.labels == ("A", "B", "C", "D") and q.keys == ("a", "b", "c", "d")
    assert q.texts[2] == "Decrease."


def test_permuted_letter_keys_keep_their_letter():
    assert assign_choice_labels(["c", "a", "d", "b"]) == ["C", "A", "D", "B"]


def test_non_letter_keys_take_next_unused_letters():
    assert assign_choice_labels(["a", "b", "c", "d", "none_of_these"]) == ["A", "B", "C", "D", "E"]
    assert assign_choice_labels(["joy", "anger", "fear"]) == ["A", "B", "C"]
    assert assign_choice_labels(["none_of_these", "b", "a"]) == ["C", "B", "A"]


def test_case_colliding_letter_keys_do_not_share_a_label():
    labels = assign_choice_labels(["a", "A"])
    assert len(set(labels)) == 2 and labels[0] == "A"


def test_non_letter_option_text_keeps_name_and_description():
    body = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"joy": None, "anger": "rage"}}}}
    q = parse_request(body).questions[0]
    assert q.texts == ("joy", "anger: rage")


def test_noul_uses_kev_order_no_then_yes_with_rubric():
    body = {"state": "you idiot", "questions": {"o": {"type": "noul", "instructions": "Is it offensive?",
            "criteria": {"true": "Contains insults", "false": "Does not"}}}}
    q = parse_request(body).questions[0]
    assert q.keys == ("false", "true") and q.labels == ("A", "B")
    assert q.texts == ("no: Does not", "yes: Contains insults")


def test_noul_without_criteria():
    q = parse_request({"state": "x", "questions": {"o": {"type": "noul"}}}).questions[0]
    assert q.texts == ("no", "yes") and q.instructions == ""


def test_score_levels_are_lettered_in_order_with_legend():
    body = {"state": "x", "questions": {"s": {"type": "score", "criteria": ["low", "mid", "high"]}}}
    q = parse_request(body).questions[0]
    assert q.keys == ("0", "1", "2") and q.labels == ("A", "B", "C")
    assert q.legend == {"0": "low", "1": "mid", "2": "high"}


def test_digest_matches_kev_record_digest_of_body_without_model():
    body = {**MMLU, "model": "kev-latest"}
    want = hashlib.sha256(json.dumps(MMLU, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert parse_request(body).digest == want


@pytest.mark.parametrize("body", [
    [], {"questions": {"q": {"type": "noul"}}},                                    # not a dict / no state
    {"state": "x", "questions": {}}, {"state": "x"},                                # no questions
    {"state": "x", "questions": {"q": {"type": "rank"}}},                           # unknown type
    {"state": "x", "questions": {"q": {"type": "choice", "criteria": {}}}},         # empty choice
    {"state": "x", "questions": {"q": {"type": "choice", "criteria": ["a"]}}},      # choice criteria not a dict
    {"state": "x", "questions": {"q": {"type": "score", "criteria": {}}}},          # score criteria not a list
    {"state": "x", "questions": {"q": {"type": "noul", "criteria": ["x"]}}},        # noul criteria not a dict
    {"state": "x", "model": 3, "questions": {"q": {"type": "noul"}}},               # model not a string
    {"state": "x", "questions": {"q": {"type": "choice", "criteria": {f"k{i}": None for i in range(256)}}}},  # > 255
    {"state": "x", "questions": {"q": {"type": "score", "criteria": ["x"] * 256}}},                          # > 255
])
def test_invalid_requests_raise(body):
    with pytest.raises(InvalidRequest):
        parse_request(body)


def test_27_options_are_valid_and_256_name_the_maximum():
    """v0.3 wide choice: the 27-option case moved out of the invalid list (27..255 options read with two-letter
    labels); above MAX_OPTIONS (255) is still a 422 naming the maximum."""
    q = parse_request({"state": "x", "questions": {"q": {"type": "choice",
                                                         "criteria": {f"k{i}": None for i in range(27)}}}}).questions[0]
    assert len(q.labels) == 27
    with pytest.raises(InvalidRequest, match="1..255 options"):
        parse_request({"state": "x", "questions": {"q": {"type": "choice",
                                                         "criteria": {f"k{i}": None for i in range(256)}}}})


def test_answer_bodies():
    q = parse_request(MMLU).questions[0]
    a = answer_body(q, [0.1, 0.2, 0.6, 0.1])
    assert a["type"] == "choice" and a["choice"] == "c" and a["probabilities"]["c"] == 0.6
    assert abs(a["confidence"] - (0.6 - 0.25) / 0.75) < 1e-12
    n = parse_request({"state": "x", "questions": {"o": {"type": "noul"}}}).questions[0]
    assert answer_body(n, [0.3, 0.7]) == {"type": "noul", "noul": 0.7}
    s = parse_request({"state": "x", "questions": {"s": {"type": "score", "criteria": ["l", "m", "h"]}}}).questions[0]
    b = answer_body(s, [0.0, 0.5, 0.5])
    assert b["score"] == 1.5 and b["legend"] == {"0": "l", "1": "m", "2": "h"} and b["probabilities"] == {"0": 0.0, "1": 0.5, "2": 0.5}


# ---- v0.2: images, image_url parts in state, context ----------------------------------------------------

PNG = "iVBORw0KGgo="                       # 8 bytes (the PNG signature): valid base64, which is all parsing checks
URL = f"data:image/png;base64,{PNG}"
Q1 = {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}


def _req(**kw):
    return parse_request({"state": "s", "questions": Q1, **kw})


def test_images_are_kept_as_payloads_and_verbatim_urls():
    jpeg = "data:image/jpeg;base64,/9j/"
    r = _req(images=[URL, jpeg])
    assert r.images == (PNG, "/9j/") and r.image_urls == (URL, jpeg) and r.context == ""


def test_images_null_and_absent_mean_no_images():
    assert _req(images=None).images == () and _req().images == () and _req().image_urls == ()


def test_the_payload_is_the_text_after_the_first_base64_marker():
    r = _req(images=["data:image/svg+xml;charset=utf-8;base64,QUJD"])
    assert r.images == ("QUJD",)


@pytest.mark.parametrize(("images", "match"), [
    ("data:image/png;base64,QUJD", "array"),                     # not an array
    ({"0": URL}, "array"),
    ([3], r"images\[0\].*string"),                               # an element that is not a string
    ([URL, None], r"images\[1\].*string"),
    (["http://example.com/a.png"], r"images\[0\].*data:image/"),  # http(s) URLs are out of scope
    (["data:text/plain;base64,QUJD"], r"images\[0\].*data:image/"),
    (["data:image/png,QUJD"], r"images\[0\].*;base64,"),         # no ;base64,
    (["data:image/png;base64,QU JD"], r"images\[0\].*base64"),   # invalid base64 (validate=True)
    (["data:image/png;base64,QUJ"], r"images\[0\].*base64"),     # bad padding
    (["data:image/png;base64,"], r"images\[0\].*empty"),         # decodes to zero bytes
])
def test_invalid_images_are_rejected_with_the_index_and_why(images, match):
    with pytest.raises(InvalidRequest, match=match):
        _req(images=images)


def _msgs(*parts):
    return [{"role": "user", "content": [{"type": "text", "text": "look"}, *parts]}]


def test_state_image_parts_are_taken_out_before_rendering_unwrapped_and_wrapped():
    part_obj = {"type": "image_url", "image_url": {"url": URL}}
    part_str = {"type": "image_url", "image_url": "data:image/gif;base64,R0lG"}   # native also allows a bare string
    for state in (_msgs(part_obj, part_str), {"messages": _msgs(part_obj, part_str), "note": "n"}):
        r = parse_request({"state": state, "questions": Q1})
        assert r.images == (PNG, "R0lG") and r.image_urls == (URL, "data:image/gif;base64,R0lG")
        assert "base64" not in r.state and "image_url" not in r.state and "look" in r.state
    wrapped = parse_request({"state": {"messages": _msgs(part_obj), "note": "n"}, "questions": Q1})
    assert "note: n" in wrapped.state                                # the wrapper's other keys are kept


@pytest.mark.parametrize("part", [
    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}},
    {"type": "video", "video": {"url": "data:video/mp4;base64,AAAA"}},
])
def test_audio_url_and_video_state_parts_are_rejected_with_422_not_rendered(part):
    # v0.2's rule, rewritten for v0.3: llama.cpp refuses these two part types, so callo does too, with
    # a 422 rather than a silent multi-MB data URL in the prompt. input_audio / video_url are taken out (below).
    with pytest.raises(InvalidRequest, match="parts are not supported"):
        parse_request({"state": _msgs(part), "questions": Q1})
    with pytest.raises(InvalidRequest, match="parts are not supported"):   # wrapped form too
        parse_request({"state": {"messages": _msgs(part)}, "questions": Q1})


def test_images_come_first_then_state_parts_in_document_order():
    state = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}}]},
             {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,Qg=="}}]}]
    r = parse_request({"state": state, "images": [URL], "questions": Q1})
    assert r.images == (PNG, "QQ==", "Qg==")


def test_non_image_parts_and_non_list_contents_are_left_alone():
    state = [{"role": "system", "content": "plain text"}, "loose string",
             # not a part llama.cpp reads
             {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "t"}]}]
    r = parse_request({"state": state, "questions": Q1})
    assert r.images == () and "loose string" in r.state and "type: image" in r.state


@pytest.mark.parametrize(("part", "match"), [
    ({"type": "image_url", "image_url": {}}, r"message 0 part 1.*no url"),
    ({"type": "image_url", "image_url": {"url": 7}}, r"message 0 part 1.*string"),
    ({"type": "image_url", "image_url": 7}, r"message 0 part 1.*string"),
    ({"type": "image_url", "image_url": {"url": "data:image/png;base64,!!"}}, r"message 0 part 1.*base64"),
])
def test_a_malformed_state_image_part_is_rejected_with_message_and_part_index(part, match):
    with pytest.raises(InvalidRequest, match=match):
        parse_request({"state": _msgs(part), "questions": Q1})


def test_the_eight_image_limit_counts_both_sources():
    eight = [URL] * 8
    assert len(_req(images=eight).images) == 8
    with pytest.raises(InvalidRequest, match="too many media items, the maximum is 8"):
        _req(images=[URL] * 9)
    part = {"type": "image_url", "image_url": {"url": URL}}
    with pytest.raises(InvalidRequest, match="too many media items, the maximum is 8"):
        parse_request({"state": _msgs(part), "images": eight, "questions": Q1})


def test_context_is_kept_and_type_checked():
    assert _req(context="orders: go left").context == "orders: go left"
    assert _req(context=None).context == "" and _req(context="").context == ""
    for bad in (3, ["x"], {"a": 1}, True):
        with pytest.raises(InvalidRequest, match="context must be a string"):
            _req(context=bad)


def test_digest_covers_images_and_context():
    base = _req().digest
    assert _req(images=[URL]).digest != base and _req(context="c").digest != base
    body = {"state": "s", "questions": Q1, "images": [URL], "context": "c"}
    want = hashlib.sha256(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert parse_request({**body, "model": "m"}).digest == want


def test_a_v01_request_parses_exactly_as_before():
    r = parse_request(MMLU)
    assert (r.images, r.image_urls, r.context) == ((), (), "")
    assert r.state == "subject: security studies\nquestion: Which did not emerge?"


# ---- v0.3: audio and video ----------------------------------------------------------------------------------

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


WAV = _b64(b"RIFF\x24\x00\x00\x00WAVEfmt ")          # 16 bytes: enough for the sniff, which is all parsing checks
MP3 = _b64(b"\xff\xfb\x90\x00" + b"\x00" * 12)
FLAC = _b64(b"fLaC" + b"\x00" * 12)
AVI = _b64(b"RIFF\x24\x00\x00\x00AVI LIST")
MP4 = _b64(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4)
WAV_URL, AVI_URL = f"data:audio/wav;base64,{WAV}", f"data:video/x-msvideo;base64,{AVI}"
ADTS = _b64(b"\xff\xf1" + b"\x00" * 14)         # AAC: refused as audio, and mtmd would still decode it


def _audio_part(**inner):
    return {"type": "input_audio", "input_audio": inner}


def test_every_source_lands_in_media_in_contract_order():
    """images, then audio, then videos, then the state's parts in document order: the order of the rendered media
    parts and of multimodal_data."""
    mp3_url = f"data:audio/mpeg;base64,{MP3}"
    state = _msgs({"type": "input_video", "input_video": {"data": MP4}}, _audio_part(data=mp3_url),
                  {"type": "image_url", "image_url": {"url": URL}})
    r = parse_request({"state": state, "images": [URL], "audio": [WAV_URL], "videos": [AVI_URL], "questions": Q1})
    assert r.media == (MediaItem("image", PNG, URL), MediaItem("audio", WAV, WAV_URL, "wav"),
                       MediaItem("video", AVI, AVI_URL), MediaItem("video", MP4, None),
                       MediaItem("audio", MP3, mp3_url, "mp3"), MediaItem("image", PNG, URL))
    assert r.kind == "mixed" and r.images == (PNG, PNG) and r.image_urls == (URL, URL)
    assert "base64" not in r.state and "input_" not in r.state and "look" in r.state


@pytest.mark.parametrize(("part", "item"), [
    (_audio_part(data=WAV, format="mp3"), MediaItem("audio", WAV, None, "wav")),      # raw base64; format is ignored
    (_audio_part(data=WAV_URL), MediaItem("audio", WAV, WAV_URL, "wav")),             # a data URL in data
    (_audio_part(url=f"data:audio/flac;base64,{FLAC}"),
     MediaItem("audio", FLAC, f"data:audio/flac;base64,{FLAC}", "flac")),
    (_audio_part(data=FLAC, url=WAV_URL), MediaItem("audio", FLAC, None, "flac")),    # data wins over url
    (_audio_part(data=None, url=WAV_URL), MediaItem("audio", WAV, WAV_URL, "wav")),   # a null data does not
    ({"type": "video_url", "video_url": {"url": AVI_URL}}, MediaItem("video", AVI, AVI_URL)),
    ({"type": "input_video", "input_video": {"data": AVI, "url": "ignored"}}, MediaItem("video", AVI, None)),
    (_audio_part(data=f"data:video/mp4;base64,{WAV}"), MediaItem("audio", WAV, f"data:video/mp4;base64,{WAV}", "wav")),
])
def test_state_part_shapes_follow_llama_cpp(part, item):
    """The inner object under the key equal to the type; data before url; raw base64 or a data URL in either; a data
    URL's MIME family need not match the part type, as in llama.cpp; the format field is ignored."""
    assert parse_request({"state": _msgs(part), "questions": Q1}).media == (item,)


def test_input_video_is_taken_out_not_flattened_into_the_prompt():
    """Regression: v0.2 had no rule for input_video, so its base64 was rendered into the prompt text."""
    r = parse_request({"state": _msgs({"type": "input_video", "input_video": {"data": AVI}}), "questions": Q1})
    assert r.kind == "video" and AVI not in r.state and "input_video" not in r.state


@pytest.mark.parametrize("part", [
    # the common malformed shape: url beside the type
    {"type": "image_url", "url": URL},
    {"type": "image_url"},
    {"type": "image_url", "image_url": None},
])
def test_an_image_url_part_without_an_image_url_object_is_a_422_not_flattened(part):
    """Like the audio/video parts: an image_url part lacking its `image_url` object is refused, not rendered
    into the prompt as text (llama.cpp refuses it too: 'Invalid base64 value')."""
    with pytest.raises(InvalidRequest, match=r"part 1: image_url has no image_url object"):
        parse_request({"state": _msgs(part), "questions": Q1})


@pytest.mark.parametrize(("part", "match"), [
    ({"type": "input_video", "video_url": {"data": AVI}}, r"part 1: a input_video part needs a 'input_video' object"),
    ({"type": "video_url", "input_video": {"data": AVI}}, r"part 1: a video_url part needs a 'video_url' object"),
    ({"type": "input_audio", "input_audio": WAV}, r"part 1: a input_audio part needs"),
    (_audio_part(format="wav"), r"part 1: input_audio has no data or url"),
    (_audio_part(data=7), r"part 1: the media data or url must be a string"),
    (_audio_part(url="https://example.com/a.wav"), r"part 1: http\(s\):// and file:// media are not supported"),
    (_audio_part(data="file:///tmp/a.wav"), r"part 1: http\(s\):// and file:// media are not supported"),
    (_audio_part(data="data:text/plain;base64,QUJD"),
     r"part 1: an audio clip must be a data URL starting with data:image/, data:video/ or data:audio/"),
    (_audio_part(data="!!notbase64"), r"part 1: invalid base64"),
    (_audio_part(data="QUJD"), r"part 1: unsupported audio format; use WAV, MP3 or FLAC"),
    ({"type": "input_video", "input_video": {"data": WAV}}, r"part 1: the bytes are an image/audio file, not a video"),
    ({"type": "input_video", "input_video": {"data": _b64(b"\x89PNG\r\n\x1a\n" + bytes(8))}}, r"not a video"),
    ({"type": "input_video", "input_video": {"data": ADTS}}, r"part 1: the bytes are an image/audio file, not a video"),
])
def test_a_malformed_audio_or_video_part_is_a_422_naming_it(part, match):
    with pytest.raises(InvalidRequest, match=match):
        parse_request({"state": _msgs(part), "questions": Q1})


@pytest.mark.parametrize(("body", "match"), [
    ({"audio": WAV_URL}, '"audio" must be an array'),
    ({"videos": {"0": AVI_URL}}, '"videos" must be an array'),
    ({"audio": [3]}, r"audio\[0\]: an audio clip must be a string data URL"),
    ({"audio": [AVI_URL]}, r"audio\[0\]: an audio clip must be a data URL starting with data:audio/"),
    ({"videos": [WAV_URL]}, r"videos\[0\]: a video must be a data URL starting with data:video/"),
    ({"audio": ["data:audio/wav,QUJD"]}, r"audio\[0\]: the data URL has no ;base64,"),
    ({"audio": ["data:audio/wav;base64,"]}, r"audio\[0\]: the audio clip is empty"),
    ({"videos": ["data:video/mp4;base64,"]}, r"videos\[0\]: the video is empty"),
    ({"audio": [f"data:audio/ogg;base64,{_b64(b'OggS' + bytes(12))}"]}, r"audio\[0\]: unsupported audio format"),
    ({"audio": [f"data:audio/aac;base64,{_b64(bytes([0xFF, 0xF1]) + bytes(14))}"]}, "unsupported audio format"),
    ({"videos": [f"data:video/mp4;base64,{_b64(b'GIF89a' + bytes(10))}"]}, r"videos\[0\]: the bytes are an image"),
    ({"videos": [f"data:video/mp4;base64,{MP3}"]}, r"videos\[0\]: the bytes are an image/audio file, not a video"),
    ({"images": [f"data:image/png;base64,{WAV}"]}, r"images\[0\]: the bytes are an audio file, not an image"),
    ({"videos": [f"data:video/mp4;base64,{ADTS}"]}, r"videos\[0\]: the bytes are an image/audio file, not a video"),
    ({"images": [f"data:image/png;base64,{ADTS}"]}, r"images\[0\]: the bytes are an audio file, not an image"),
])
def test_top_level_audio_and_videos_are_checked_like_images_and_by_their_bytes(body, match):
    with pytest.raises(InvalidRequest, match=match):
        _req(**body)


def test_null_or_empty_audio_and_videos_mean_none():
    for extra in ({"audio": None, "videos": None}, {"audio": [], "videos": []}, {}):
        r = _req(**extra)
        assert r.media == () and r.kind == "plain"


def test_max_media_counts_every_kind_and_source():
    assert MAX_MEDIA == 8
    eight = _req(images=[URL] * 3, audio=[WAV_URL] * 3, videos=[AVI_URL] * 2)
    assert len(eight.media) == 8 and eight.kind == "mixed"
    with pytest.raises(InvalidRequest, match="too many media items, the maximum is 8"):
        _req(images=[URL] * 3, audio=[WAV_URL] * 3, videos=[AVI_URL] * 3)
    with pytest.raises(InvalidRequest, match="too many media items"):
        parse_request({"state": _msgs(_audio_part(data=WAV)), "videos": [AVI_URL] * 8, "questions": Q1})
    with pytest.raises(InvalidRequest, match="too many media items"):
        _req(audio=[WAV_URL] * 9)


@pytest.mark.parametrize(("body", "kind"), [
    ({}, "plain"), ({"context": "c"}, "context"), ({"images": [URL], "context": "c"}, "image"),
    ({"audio": [WAV_URL]}, "audio"), ({"videos": [AVI_URL, AVI_URL]}, "video"),
    ({"images": [URL], "audio": [WAV_URL]}, "mixed"),
])
def test_request_kind_is_defined_once(body, kind):
    r = _req(**body)
    assert r.kind == kind == media_kind(r.media, r.context)


def test_the_compatibility_properties_read_only_the_image_items():
    r = _req(audio=[WAV_URL], images=[URL])
    assert r.images == (PNG,) and r.image_urls == (URL,) and [m.kind for m in r.media] == ["image", "audio"]
    assert _req(audio=[WAV_URL]).images == () and _req(audio=[WAV_URL]).image_urls == ()


def test_digest_covers_audio_and_videos():
    base = _req().digest
    assert _req(audio=[WAV_URL]).digest != base and _req(videos=[AVI_URL]).digest != _req(audio=[WAV_URL]).digest


@pytest.mark.parametrize(("body", "refused"), [
    ({"state": "s", "audio": [WAV_URL]}, '"audio" is not supported'),
    ({"state": "s", "videos": "x"}, '"videos" is not supported'),             # any non-list, non-null value
    ({"state": _msgs(_audio_part(data=WAV))}, "state message 0 part 1 is input_audio"),
    ({"state": {"messages": _msgs({"type": "video_url", "video_url": {"url": AVI_URL}})}}, "part 1 is video_url"),
    ({"state": _msgs({"type": "input_video", "input_video": {}})}, "is input_video"),
    ({"state": _msgs({"type": "audio_url"})}, "is audio_url"),
    ({"state": _msgs({"type": "video"})}, "is video"),
    ({"state": "s", "audio": [], "videos": None}, None),                       # empty means none
    ({"state": _msgs({"type": "image_url", "image_url": {"url": URL}}), "images": [URL]}, None),   # native's own
    ({"state": [{"role": "user", "content": "text"}, "loose", {"content": [7, "x"]}]}, None),
    ({"state": _msgs({"type": ["input_audio"]}, {"type": {"input_video": 1}}, {"type": None})}, None),   # not str
    (["not", "a", "dict"], None),
])
def test_native_media_refusal(body, refused):
    got = native_media_refusal(body)
    if refused is None:
        assert got is None
    else:
        assert got.startswith("the upstream is a native decision model, which takes images only: ") and refused in got



@pytest.mark.parametrize("weird", [["input_audio", "null"], {"type": "string"}, None, 7])
def test_a_part_whose_type_is_not_a_string_stays_in_the_state_as_in_v02(weird):
    """A JSON-schema fragment such as {"type": ["string", "null"]} inside a content part must not
    crash the set lookup (a TypeError and a 500); it is no media part, so it stays in the state and is rendered."""
    part = {"type": weird, "note": "kept"}
    for state in (_msgs(part), {"messages": _msgs(part)}):
        r = parse_request({"state": state, "questions": Q1})
        assert r.media == () and "note: kept" in r.state


# ---- v0.3 think budgets: the think_budget field --------------------------------------------------------

NOUL = {"state": "s", "questions": {"q": {"type": "noul"}}}


def test_think_budget_is_parsed_and_absent_or_null_means_the_server_default():
    assert parse_request({**NOUL, "think_budget": 256}).think_budget == 256
    assert parse_request({**NOUL, "think_budget": 0}).think_budget == 0
    assert parse_request(NOUL).think_budget is None
    assert parse_request({**NOUL, "think_budget": None}).think_budget is None


@pytest.mark.parametrize("bad", [True, False, 1.5, -1, 32769, "256", [256]])
def test_a_bad_think_budget_is_a_422_naming_the_range(bad):
    with pytest.raises(InvalidRequest, match="think_budget must be an integer from 0 to 32768"):
        parse_request({**NOUL, "think_budget": bad})


def test_the_think_budget_is_part_of_the_digest():
    assert parse_request({**NOUL, "think_budget": 256}).digest != parse_request(NOUL).digest
    assert parse_request({**NOUL, "think_budget": 256}).digest != parse_request({**NOUL, "think_budget": 512}).digest


def test_native_think_refusal_refuses_only_a_positive_budget():
    assert native_think_refusal({**NOUL, "think_budget": 256}) is not None
    assert "native decision model" in native_think_refusal({**NOUL, "think_budget": 1})
    assert native_think_refusal({**NOUL, "think_budget": 0}) is None
    assert native_think_refusal(NOUL) is None and native_think_refusal({**NOUL, "think_budget": None}) is None
    assert native_think_refusal({**NOUL, "think_budget": "x"}) == "think_budget must be an integer from 0 to 32768"
    assert native_think_refusal("not a dict") is None


def test_native_think_refusal_covers_think_deadline_ms():
    for ok in (0, None):
        assert native_think_refusal({**NOUL, "think_deadline_ms": ok}) is None
    assert native_think_refusal(NOUL) is None
    assert native_think_refusal({**NOUL, "think_deadline_ms": 1500}).endswith("think_deadline_ms must be 0 or absent")
    for bad in (True, -1, 600_001, 1.5, "100"):
        assert native_think_refusal({**NOUL, "think_deadline_ms": bad}) == THINK_DEADLINE_ERROR, bad
    # think_budget is checked first
    assert native_think_refusal({**NOUL, "think_budget": "x", "think_deadline_ms": 1500}) == "think_budget must be an integer from 0 to 32768"
    assert "think_budget must be 0" in native_think_refusal({**NOUL, "think_budget": 5, "think_deadline_ms": 1500})
