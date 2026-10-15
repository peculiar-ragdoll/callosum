"""Audio and video: the built-in self-test fixtures, built with the stdlib, and the self-tests.

The bytes are pinned by sha256 (tests): llama.cpp keys media chunks by the sha256 of the file bytes
(`mtmd/mtmd-helper.cpp:376-379` @ 1537a0a), so stable bytes also mean a stable cache, and a builder change cannot slip
through unnoticed.

- **Audio:** WAV, 16 kHz mono 16-bit PCM, 4.0 s (Whisper-family encoders pad to 30 s windows, so a 1 s tone
  would be about 3% of the only window). The main clip is a 440 Hz tone at -6 dBFS (amplitude 16422 of 32767); the
  control is digital silence. The header is written by hand, not by `wave`, so no stdlib version can change it.
- **Video:** an uncompressed RIFF AVI (`vids` / `DIB `, BI_RGB 24-bit), 64x64, 8 frames at 4 fps (2 s). **A BI_RGB
  pixel is stored B, G, R, and the rows run bottom-up**: `solid_frame` takes an RGB colour and writes BGR, so the
  red clip really decodes red through mtmd's ffmpeg command line (an ffmpeg-gated test checks it). The main clip is
  solid red, the control solid blue.

The self-tests mirror the vision self-test: the main clip, then (audio only) an identical repeat to
show the cache, then a control that needs the opposite answer. Pass: both answers correct and both option masses at
least 0.5. They warn and never raise. One outcome is special: the upstream answering the known-good
fixture with "Failed to load image or audio file" (`load_failed`); for video that means ffprobe or ffmpeg is missing
(llama-server's PATH or --video-ffmpeg-dir) or ffprobe cannot read the clip, and serve turns video off. An ffmpeg that
starts but cannot decode yields no frames and no error (mtmd-helper.cpp's lazy reader reports a plain end of stream):
the model then answers without the video, which shows here as a wrong answer or low mass. That, a timeout or a 503
only warn.
"""
from __future__ import annotations

import base64
import math
import struct
import time
import urllib.error
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .engine import classify_media_error
from .errors import MediaLoadFailed
from .systemone import parse_request
from .vision import self_test_answer

WAV_RATE = 16000
WAV_SECONDS = 4
TONE_HZ = 440
TONE_AMPLITUDE = 16422          # round(32767 * 10 ** (-6 / 20)): -6.0 dBFS
AVI_SIZE = 64
AVI_FRAMES = 8
AVI_FPS = 4


def _chunk(fourcc: bytes, data: bytes) -> bytes:
    """A RIFF chunk: fourcc, little-endian size, data, and a pad byte when the size is odd."""
    return fourcc + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) % 2 else b"")


def _list(kind: bytes, body: bytes) -> bytes:
    return _chunk(b"LIST", kind + body)


def encode_wav(samples: Sequence[int], *, rate: int = WAV_RATE) -> bytes:
    """Mono 16-bit PCM WAV: the canonical 44-byte header, then the samples, little-endian."""
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)        # PCM, mono, rate, byte rate, align, bits
    return _chunk(b"RIFF", b"WAVE" + _chunk(b"fmt ", fmt) + _chunk(b"data", struct.pack(f"<{len(samples)}h", *samples)))


def tone_samples(*, hz: int = TONE_HZ, amplitude: int = TONE_AMPLITUDE, rate: int = WAV_RATE,
                 seconds: int = WAV_SECONDS) -> list[int]:
    """round(amplitude * sin(2 pi hz n / rate)). At 440 Hz / 16 kHz there are 400 distinct phases, and none rounds
    within 0.005 of a half (a test checks), so a last-ulp difference in a platform's sin cannot change a sample."""
    return [round(amplitude * math.sin(2 * math.pi * hz * n / rate)) for n in range(rate * seconds)]


def solid_frame(rgb: tuple[int, int, int], *, width: int = AVI_SIZE, height: int = AVI_SIZE) -> bytes:
    """One BI_RGB 24-bit frame of a single colour: B, G, R per pixel, rows padded to 4 bytes (bottom-up order does not
    matter for a solid frame)."""
    r, g, b = rgb
    row = bytes((b, g, r)) * width
    return (row + b"\x00" * (-len(row) % 4)) * height


def encode_avi(frames: Sequence[bytes], *, width: int = AVI_SIZE, height: int = AVI_SIZE, fps: int = AVI_FPS) -> bytes:
    """An uncompressed AVI: hdrl (avih, one vids/DIB stream: strh, strf = BITMAPINFOHEADER with a positive height,
    i.e. bottom-up), movi with one `00db` chunk per frame, and an idx1 index."""
    size = ((width * 3 + 3) & ~3) * height
    if any(len(f) != size for f in frames):
        raise ValueError(f"every frame must be {size} bytes for {width}x{height} BI_RGB 24-bit")
    n = len(frames)
    avih = struct.pack("<14I", 1_000_000 // fps, size * fps, 0, 0x10, n, 0, 1, size, width, height, 0, 0, 0, 0)
    strh = b"vids" + b"DIB " + struct.pack("<IHHIIIIIIIIhhhh", 0, 0, 0, 0, 1, fps, 0, n, size, 0xFFFFFFFF, 0,
                                           0, 0, width, height)
    strf = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, size, 0, 0, 0, 0)
    hdrl = _list(b"hdrl", _chunk(b"avih", avih) + _list(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf)))
    movi, idx, offset = b"", b"", 4                 # idx1 offsets count from the 'movi' list type
    for f in frames:
        ck = _chunk(b"00db", f)
        idx += b"00db" + struct.pack("<III", 0x10, offset, len(f))   # AVIIF_KEYFRAME
        movi += ck
        offset += len(ck)
    return _chunk(b"RIFF", b"AVI " + hdrl + _list(b"movi", movi) + _chunk(b"idx1", idx))


def _data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


TONE_WAV = encode_wav(tone_samples())
TONE_WAV_SHA256 = "dfe74acd4e98ae4134657aadb2f7dc4cf26c34ed5a4d5fb2e4ae817f1a018b4d"
SILENCE_WAV = encode_wav([0] * (WAV_RATE * WAV_SECONDS))
SILENCE_WAV_SHA256 = "f751fd7094be088dc785125e4a1a0222a17c2bd20469aa31728ca5e515ed5809"
RED_AVI = encode_avi([solid_frame((255, 0, 0))] * AVI_FRAMES)
RED_AVI_SHA256 = "4db262ff45ca20770180fde2370806bd8d4c17405ec1c414c5a14161a320739f"
BLUE_AVI = encode_avi([solid_frame((0, 0, 255))] * AVI_FRAMES)
BLUE_AVI_SHA256 = "8f39e15b816392d0bd1481043454531b44ef425750f83c821ffa2a32a4a034cc"

TONE_URL, SILENCE_URL = _data_url("audio/wav", TONE_WAV), _data_url("audio/wav", SILENCE_WAV)
RED_URL, BLUE_URL = _data_url("video/x-msvideo", RED_AVI), _data_url("video/x-msvideo", BLUE_AVI)

_AUDIO_Q = {"q": {"type": "choice", "instructions": "Is there a sound in this audio clip, or is it silent?",
                  "criteria": {"sound": None, "silent": None}}}
_VIDEO_Q = {"q": {"type": "choice", "instructions": "What colour fills the video?",
                  "criteria": {"red": None, "blue": None}}}
AUDIO_SELF_TEST_BODY = {"state": "", "audio": [TONE_URL], "questions": _AUDIO_Q}             # expected "sound"
AUDIO_SELF_TEST_CONTROL_BODY = {"state": "", "audio": [SILENCE_URL], "questions": _AUDIO_Q}  # expected "silent"
VIDEO_SELF_TEST_BODY = {"state": "", "videos": [RED_URL], "questions": _VIDEO_Q}             # expected "red"
VIDEO_SELF_TEST_CONTROL_BODY = {"state": "", "videos": [BLUE_URL], "questions": _VIDEO_Q}    # expected "blue"


# kind -> (main body, control body, expected main answer, expected control answer, what the main / control are,
#          whether the main request is repeated to show the cache)
_SELF_TESTS: dict[str, tuple[dict, dict, str, str, str, str, bool]] = {
    "audio": (AUDIO_SELF_TEST_BODY, AUDIO_SELF_TEST_CONTROL_BODY, "sound", "silent", "440 Hz tone", "silence", True),
    "video": (VIDEO_SELF_TEST_BODY, VIDEO_SELF_TEST_CONTROL_BODY, "red", "blue", "red clip", "blue clip", False),
}


@dataclass(frozen=True)
class MediaSelfTestResult:
    kind: str                           # "audio" | "video"
    ok: bool
    answer: str | None                  # for the main clip
    option_mass: float | None
    latency_ms: float | None
    control_answer: str | None
    control_option_mass: float | None
    cache_n: int | None = None          # of the identical repeat (audio only; None for video)
    prompt_n: int | None = None
    error: str | None = None
    load_failed: bool = False           # the upstream could not load the known-good fixture

    def lines(self) -> list[str]:
        _, _, want, want_control, main, control, _ = _SELF_TESTS[self.kind]
        if self.error:
            return [f"WARNING: {self.kind} self-test could not run: {self.error}"]
        verdict = "ok" if self.ok else "FAILED"
        repeat = (f"; identical repeat cache_n={self.cache_n} prompt_n={self.prompt_n}"
                  if self.kind == "audio" else "")
        out = [f"{self.kind} self-test {verdict}: {main} -> answer={self.answer} option mass={self.option_mass:.3f} "
               f"latency={self.latency_ms:.0f} ms; {control} control -> answer={self.control_answer} option mass="
               f"{self.control_option_mass:.3f}{repeat}"]
        if self.answer != want:
            out.append(f"WARNING: the {self.kind} self-test answered {self.answer!r} for the {main}, expected "
                       f"{want!r}: {self.kind} input may not reach the model")
        if self.control_answer != want_control:
            out.append(f"WARNING: the {self.kind} self-test control ({control}) answered {self.control_answer!r}, "
                       f"expected {want_control!r}: {self.kind} input may not reach the model")
        if self.answer == self.control_answer:
            out.append(f"WARNING: the model gave the same answer for the {main} and the {control}: it may be ignoring "
                       f"the {self.kind}")
        if (self.option_mass is not None and self.option_mass < 0.5) or (
                self.control_option_mass is not None and self.control_option_mass < 0.5):
            out.append(f"WARNING: {self.kind} self-test option mass below 0.5: the scoring position may not be the "
                       f"answer slot")
        return out


def media_self_test(engine: Any, kind: str, *, id_slot: int | None = None) -> MediaSelfTestResult:
    """The `kind` ("audio" or "video") self-test through `engine.answer(media=)`. Never raises. An upstream HTTP
    error is classified ONCE through `classify_media_error` (its body can be read only once): "Failed to load
    image or audio file" sets `load_failed`, which only the caller acts on (serve turns video off)."""
    body, control_body, want, want_control, _, _, repeat = _SELF_TESTS[kind]
    try:
        req, control = parse_request(body), parse_request(control_body)
        t0 = time.perf_counter()
        answer, mass, _ = self_test_answer(engine, req, id_slot)
        latency = 1000 * (time.perf_counter() - t0)
        again = self_test_answer(engine, req, id_slot)[2] if repeat else None
        c_answer, c_mass, _ = self_test_answer(engine, control, id_slot)
    except urllib.error.HTTPError as e:     # before Exception: HTTPError carries the upstream's message
        err = classify_media_error(e, kind=kind)
        return MediaSelfTestResult(kind, False, None, None, None, None, None, error=f"{type(err).__name__}: {err}",
                                   load_failed=isinstance(err, MediaLoadFailed))
    except Exception as e:  # noqa: BLE001 -- the self-test reports, it does not raise
        return MediaSelfTestResult(kind, False, None, None, None, None, None, error=f"{type(e).__name__}: {e}")
    ok = answer == want and c_answer == want_control and mass >= 0.5 and c_mass >= 0.5
    return MediaSelfTestResult(kind, ok, answer, mass, latency, c_answer, c_mass,
                               again.cache_n if again is not None else None,
                               again.prompt_n if again is not None else None)


def audio_self_test(engine: Any, *, id_slot: int | None = None) -> MediaSelfTestResult:
    return media_self_test(engine, "audio", id_slot=id_slot)


def video_self_test(engine: Any, *, id_slot: int | None = None) -> MediaSelfTestResult:
    return media_self_test(engine, "video", id_slot=id_slot)
