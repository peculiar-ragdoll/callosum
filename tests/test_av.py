"""v0.3 media fixtures and self-tests: the stdlib WAV and AVI self-test fixtures, pinned, read back, and
decoded by ffmpeg exactly as mtmd runs it; the audio and video self-tests."""
import base64
import dataclasses
import hashlib
import io
import json
import math
import shutil
import struct
import subprocess
import urllib.error
import wave

import pytest

from callo.av import (
    AUDIO_SELF_TEST_BODY,
    AUDIO_SELF_TEST_CONTROL_BODY,
    BLUE_AVI,
    BLUE_AVI_SHA256,
    BLUE_URL,
    RED_AVI,
    RED_AVI_SHA256,
    RED_URL,
    SILENCE_URL,
    SILENCE_WAV,
    SILENCE_WAV_SHA256,
    TONE_AMPLITUDE,
    TONE_URL,
    TONE_WAV,
    TONE_WAV_SHA256,
    VIDEO_SELF_TEST_BODY,
    VIDEO_SELF_TEST_CONTROL_BODY,
    audio_self_test,
    encode_avi,
    media_self_test,
    solid_frame,
    video_self_test,
)
from callo.engine import Readout
from callo.media import bytes_error, sniff_audio
from callo.systemone import MediaItem, parse_request


def test_the_fixture_bytes_are_pinned():
    assert hashlib.sha256(TONE_WAV).hexdigest() == TONE_WAV_SHA256 == \
        "dfe74acd4e98ae4134657aadb2f7dc4cf26c34ed5a4d5fb2e4ae817f1a018b4d"
    assert hashlib.sha256(SILENCE_WAV).hexdigest() == SILENCE_WAV_SHA256 == \
        "f751fd7094be088dc785125e4a1a0222a17c2bd20469aa31728ca5e515ed5809"
    assert hashlib.sha256(RED_AVI).hexdigest() == RED_AVI_SHA256 == \
        "4db262ff45ca20770180fde2370806bd8d4c17405ec1c414c5a14161a320739f"
    assert hashlib.sha256(BLUE_AVI).hexdigest() == BLUE_AVI_SHA256 == \
        "8f39e15b816392d0bd1481043454531b44ef425750f83c821ffa2a32a4a034cc"
    for url, data, mime in ((TONE_URL, TONE_WAV, "audio/wav"), (SILENCE_URL, SILENCE_WAV, "audio/wav"),
                            (RED_URL, RED_AVI, "video/x-msvideo"), (BLUE_URL, BLUE_AVI, "video/x-msvideo")):
        assert url == f"data:{mime};base64,{base64.b64encode(data).decode()}"


@pytest.mark.parametrize(("data", "peak"), [(TONE_WAV, TONE_AMPLITUDE), (SILENCE_WAV, 0)])
def test_the_wavs_round_trip_through_wave(data, peak):
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 64000)
        frames = w.readframes(64000)
    samples = struct.unpack("<64000h", frames)
    assert max(abs(s) for s in samples) == peak and len(data) == 44 + 128_000
    assert sniff_audio(data) == "wav"


def test_the_tone_is_440_hz_at_minus_6_dbfs_and_platform_stable():
    assert TONE_AMPLITUDE == round(32767 * 10 ** (-6 / 20))
    frames = wave.open(io.BytesIO(TONE_WAV)).readframes(400)
    samples = struct.unpack("<400h", frames)
    assert samples[0] == 0 and samples[1] == round(TONE_AMPLITUDE * math.sin(2 * math.pi * 440 / 16000))
    # 440 Hz at 16 kHz repeats every 400 samples (11 cycles); no phase rounds near a half, so a last-ulp difference in
    # a platform's libm sin cannot flip a sample and break the pinned sha256.
    margin = min(abs((TONE_AMPLITUDE * math.sin(2 * math.pi * 440 * n / 16000)) % 1 - 0.5) for n in range(400))
    assert margin > 0.005                                   # measured: 0.00506


def _riff(data: bytes, at: int = 0) -> list[tuple[bytes, bytes]]:
    """(fourcc, payload) of each chunk in data[at:], a stdlib reader that follows the sizes and pad bytes."""
    out = []
    while at < len(data):
        fourcc, size = data[at:at + 4], struct.unpack("<I", data[at + 4:at + 8])[0]
        out.append((fourcc, data[at + 8:at + 8 + size]))
        at += 8 + size + size % 2
    return out


@pytest.mark.parametrize(("data", "bgr"), [(RED_AVI, b"\x00\x00\xff"), (BLUE_AVI, b"\xff\x00\x00")])
def test_the_avi_riff_structure(data, bgr):
    [(riff, body)] = _riff(data)
    assert riff == b"RIFF" and body[:4] == b"AVI " and len(data) == 8 + len(body)
    top = _riff(body, 4)
    assert [c for c, _ in top] == [b"LIST", b"LIST", b"idx1"]
    hdrl, movi, idx1 = top[0][1], top[1][1], top[2][1]
    assert hdrl[:4] == b"hdrl" and movi[:4] == b"movi"
    (avih_cc, avih), (strl_cc, strl) = _riff(hdrl, 4)
    usec, _, _, flags, frames, _, streams, _, width, height = struct.unpack("<10I", avih[:40])
    assert (avih_cc, usec, flags, frames, streams, width, height) == (b"avih", 250000, 0x10, 8, 1, 64, 64)
    (strh_cc, strh), (strf_cc, strf) = _riff(strl, 4)
    assert strl_cc == b"LIST" and strh_cc == b"strh" and strf_cc == b"strf"
    assert strh[:8] == b"vidsDIB " and struct.unpack("<II", strh[20:28]) == (1, 4)       # scale 1, rate 4: 4 fps
    assert struct.unpack("<I", strh[32:36])[0] == 8                                      # dwLength: 8 frames
    bi_size, bi_w, bi_h, planes, bits, compression = struct.unpack("<IiiHHI", strf[:20])
    assert (bi_size, bi_w, bi_h, planes, bits, compression) == (40, 64, 64, 1, 24, 0)   # BI_RGB, bottom-up (h > 0)
    chunks = _riff(movi, 4)
    assert [c for c, _ in chunks] == [b"00db"] * 8 and all(len(f) == 64 * 64 * 3 for _, f in chunks)
    assert all(f == bgr * (64 * 64) for _, f in chunks)                                 # every pixel B, G, R
    assert len(idx1) == 8 * 16 and idx1[:4] == b"00db"


def test_the_avi_is_neither_audio_nor_a_still_image_so_mtmd_takes_its_video_path():
    for data in (RED_AVI, BLUE_AVI):
        assert sniff_audio(data) is None and bytes_error("video", data) is None


def test_solid_frame_writes_bgr_and_encode_avi_checks_the_frame_size():
    assert solid_frame((255, 0, 0), width=2, height=1) == b"\x00\x00\xff\x00\x00\xff\x00\x00"   # padded to 8 bytes
    with pytest.raises(ValueError, match="12288 bytes"):
        encode_avi([b"\x00" * 10])


FFPROBE = ["ffprobe", "-v", "quiet", "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
           "-select_streams", "v:0", "-of", "default=noprint_wrappers=1", "pipe:0"]
FFMPEG = ["ffmpeg", "-nostdin", "-read_ahead_limit", "-1", "-i", "cache:pipe:0", "-vf", "fps=4.000000", "-f",
          "rawvideo", "-pix_fmt", "rgb24", "pipe:1", "-loglevel", "error"]


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg and ffprobe")
@pytest.mark.parametrize(("data", "rgb"), [(RED_AVI, (255, 0, 0)), (BLUE_AVI, (0, 0, 255))])
def test_mtmds_own_ffprobe_and_ffmpeg_command_lines_decode_8_frames_of_the_right_colour(data, rgb):
    """The exact command lines of mtmd/mtmd-helper.cpp:604-735 @ 1537a0a (a buffer input: pipe:0, cache:pipe:0,
    --video-fps 4). A naive RGB writer would decode red as blue and swap the self-test's main and control."""
    probe = subprocess.run(FFPROBE, input=data, capture_output=True, timeout=60, check=True).stdout.decode()
    fields = dict(line.split("=", 1) for line in probe.split())
    assert (fields["width"], fields["height"], fields["r_frame_rate"], fields["nb_frames"]) == ("64", "64", "4/1", "8")
    assert float(fields["duration"]) == 2.0
    raw = subprocess.run(FFMPEG, input=data, capture_output=True, timeout=60, check=True).stdout
    assert len(raw) == 8 * 64 * 64 * 3                                                   # exactly 8 frames
    assert tuple(raw[:3]) == rgb and raw == bytes(rgb) * (8 * 64 * 64)


@pytest.mark.parametrize(("body", "kind", "url"), [
    (AUDIO_SELF_TEST_BODY, "audio", TONE_URL), (AUDIO_SELF_TEST_CONTROL_BODY, "audio", SILENCE_URL),
    (VIDEO_SELF_TEST_BODY, "video", RED_URL), (VIDEO_SELF_TEST_CONTROL_BODY, "video", BLUE_URL),
])
def test_the_self_test_bodies_parse_to_one_item_of_their_kind(body, kind, url):
    req = parse_request(body)
    payload = url.split(";base64,", 1)[1]
    assert req.media == (MediaItem(kind, payload, url, "wav" if kind == "audio" else None),) and req.kind == kind
    assert req.state == "" and len(req.questions) == 1


def test_the_self_test_questions():
    a, v = AUDIO_SELF_TEST_BODY["questions"]["q"], VIDEO_SELF_TEST_BODY["questions"]["q"]
    assert a["instructions"] == "Is there a sound in this audio clip, or is it silent?"
    assert a["criteria"] == {"sound": None, "silent": None}
    assert v["instructions"] == "What colour fills the video?" and v["criteria"] == {"red": None, "blue": None}
    assert AUDIO_SELF_TEST_CONTROL_BODY["questions"] == AUDIO_SELF_TEST_BODY["questions"]
    assert VIDEO_SELF_TEST_CONTROL_BODY["questions"] == VIDEO_SELF_TEST_BODY["questions"]


# ---- the audio and video self-tests ------------------------------------------------------------------------

TONE_B64, SILENCE_B64 = TONE_URL.split(",", 1)[1], SILENCE_URL.split(",", 1)[1]
RED_B64, BLUE_B64 = RED_URL.split(",", 1)[1], BLUE_URL.split(",", 1)[1]
HEARS = {TONE_B64: (0.97, 0.03), SILENCE_B64: (0.02, 0.98), RED_B64: (0.96, 0.04), BLUE_B64: (0.05, 0.95)}


def _http_error(code, message):
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    return urllib.error.HTTPError("http://up/completion", code, "err", {}, io.BytesIO(body))


# (cache_n, prompt_n) per call: the first pass, the audio repeat, then the control, all different, so a test
# can tell whose numbers a result reports.
CACHE = {1: (0, 50), 2: (46, 4), 3: (7, 43)}


class Eng:
    """probs=None answers each fixture correctly (option A: sound / red; B: silent / blue); a tuple answers
    always. `masses` gives the option mass per call (1-based), else `mass` for every call."""

    def __init__(self, probs=None, mass=0.99, exc=None, exc_on=None, masses=None):
        self.probs, self.mass, self.exc, self.exc_on, self.calls = probs, mass, exc, exc_on, []
        self.masses = masses or {}

    def answer(self, state, q, *, id_slot=None, context="", media=()):
        self.calls.append((q.keys, id_slot, tuple(media)))
        n = len(self.calls)
        if self.exc and (self.exc_on is None or n == self.exc_on):
            raise self.exc
        probs = self.probs or HEARS[media[0].payload]
        cache_n, prompt_n = CACHE[n]
        return Readout(probs=probs, log_marginals=(-0.1, -3.0), option_mass=self.masses.get(n, self.mass),
                       stop="diverged", depth=0, topk_miss=0, prompt_tokens=50, kind=media[0].kind,
                       cache_n=cache_n, prompt_n=prompt_n, prompt_ms=9.0, passes=1)


def test_the_audio_self_test_passes_with_a_repeat_and_reports_its_cache_numbers():
    e = Eng()
    r = audio_self_test(e, id_slot=0)
    assert r.ok and (r.answer, r.control_answer) == ("sound", "silent") and (r.cache_n, r.prompt_n) == (46, 4)
    assert [c[2][0].payload for c in e.calls] == [TONE_B64, TONE_B64, SILENCE_B64] and {c[1] for c in e.calls} == {0}
    assert e.calls[0][0] == ("sound", "silent") and e.calls[0][2][0].kind == "audio"
    lines = r.lines()
    assert lines[0].startswith("audio self-test ok: 440 Hz tone -> answer=sound") and "cache_n=46" in lines[0]
    assert not any(line.startswith("WARNING") for line in lines) and not r.load_failed


def test_the_video_self_test_passes_without_a_repeat():
    """The video self-test has no identical repeat (each pass is a fresh ffmpeg decode)."""
    e = Eng()
    r = video_self_test(e, id_slot=0)
    assert r.ok and (r.answer, r.control_answer) == ("red", "blue") and (r.cache_n, r.prompt_n) == (None, None)
    assert [c[2][0].payload for c in e.calls] == [RED_B64, BLUE_B64] and e.calls[0][0] == ("red", "blue")
    assert r.lines()[0].startswith("video self-test ok: red clip -> answer=red") and "cache_n" not in r.lines()[0]


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_a_media_blind_model_fails_on_the_control_and_a_low_mass_fails_too(kind):
    blind = media_self_test(Eng(probs=(0.9, 0.1)), kind)          # always option A: right on the main, wrong on control
    assert not blind.ok and blind.answer == blind.control_answer and not blind.load_failed
    assert any("same answer" in line for line in blind.lines()) and blind.lines()[0].startswith(f"{kind} self-test FAILED")
    low = media_self_test(Eng(mass=0.3), kind)
    assert not low.ok and any("option mass below 0.5" in line for line in low.lines())


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_a_low_mass_on_only_one_side_fails_too(kind):
    """Each answer needs mass >= 0.5: a low main OR a low control fails, though both answers are right."""
    # audio sends main, repeat, control; video main, control
    control = 3 if kind == "audio" else 2
    for masses in ({1: 0.3}, {control: 0.3}):
        r = media_self_test(Eng(masses=masses), kind)
        want = ("sound", "silent") if kind == "audio" else ("red", "blue")
        assert not r.ok and (r.answer, r.control_answer) == want
        assert any("option mass below 0.5" in line for line in r.lines())


def test_a_wrong_answer_on_the_main_clip_warns_with_what_was_expected():
    r = audio_self_test(Eng(probs=(0.1, 0.9)))
    assert not r.ok and any("answered 'silent' for the 440 Hz tone, expected 'sound'" in line for line in r.lines())


@pytest.mark.parametrize(("exc", "load_failed", "named"), [
    (_http_error(500, "Failed to load image or audio file"), True, "MediaLoadFailed: the upstream could not decode"),
    (_http_error(400, "Failed to load image or audio file"), True, "MediaLoadFailed"),
    (_http_error(503, "Loading model"), False, "UpstreamUnavailable"),
    (_http_error(500, "video input is not supported - hint"), False, "NoVideo"),
    (TimeoutError("timed out"), False, "TimeoutError: timed out"),
    (ConnectionRefusedError("refused"), False, "ConnectionRefusedError"),
])
def test_only_failed_to_load_sets_load_failed(exc, load_failed, named):
    """The one outcome that turns video off is the upstream failing to LOAD the known-good clip;
    a timeout, a 503 or anything else is a warning."""
    r = video_self_test(Eng(exc=exc))
    assert not r.ok and r.load_failed is load_failed and named in r.error
    assert r.lines() == [f"WARNING: video self-test could not run: {r.error}"]


@pytest.mark.parametrize("exc_on", [2, 3])
def test_an_error_on_the_repeat_or_the_control_is_reported_not_raised(exc_on):
    r = audio_self_test(Eng(exc=_http_error(500, "Failed to load image or audio file"), exc_on=exc_on))
    assert not r.ok and r.load_failed and r.answer is None


def test_a_malformed_readout_is_an_error_result():
    class Nan(Eng):
        def answer(self, *a, **kw):
            return dataclasses.replace(super().answer(*a, **kw), probs=(float("nan"), 0.5))

    r = video_self_test(Nan())
    assert not r.ok and "NaN" in r.error and not r.load_failed
