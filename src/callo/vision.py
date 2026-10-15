"""Vision: the built-in self-test image and the vision self-test. (detection from
`/props` is `detect_media` in `callo.media`, which replaced the `detect_vision`; the self-test reads
through `Engine.answer(media=)`, and `self_test_answer` is shared with the audio and video self-tests in
`callo.av`.)

The self-test image is built with the stdlib. Its deflate stream uses STORED blocks, so the bytes are the same on
every zlib build (only `crc32` and `adler32` come from zlib) and a test can pin their sha256. llama.cpp caches image
chunks by the sha256 of the image bytes (mtmd-helper.cpp:374-382), so stable bytes also mean a stable cache.

The self-test asks the same question about two images: the main one (white on the right, expected "right") and its
mirror, the control (white on the left, expected "left"). One image alone is fooled by an image-blind model with a
bias toward the later option; a blind model gives the same answer for both, so it cannot pass both. The self-test
passes only when both answers are correct and both option masses are at least 0.5.
"""
from __future__ import annotations

import base64
import math
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Any

from .systemone import parse_request


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _stored_zlib(raw: bytes) -> bytes:
    """A zlib stream of deflate STORED blocks (at most 65,535 bytes each): no compressor, so no build variance."""
    out = bytearray(b"\x78\x01")
    blocks = [raw[i:i + 65535] for i in range(0, len(raw), 65535)] or [b""]
    for i, block in enumerate(blocks):
        out += bytes([1 if i == len(blocks) - 1 else 0]) + struct.pack("<HH", len(block), len(block) ^ 0xFFFF) + block
    return bytes(out + struct.pack(">I", zlib.adler32(raw) & 0xFFFFFFFF))


def encode_png(width: int, height: int, rgb: bytes) -> bytes:
    """An 8-bit RGB PNG of `rgb` (row-major, 3 bytes per pixel), every scanline with filter 0."""
    stride = width * 3
    if len(rgb) != stride * height:
        raise ValueError(f"rgb has {len(rgb)} bytes; {width}x{height} RGB needs {stride * height}")
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", _stored_zlib(raw)) + _chunk(b"IEND", b""))


SELF_TEST_PNG = encode_png(128, 128, (b"\x00\x00\x00" * 64 + b"\xff\xff\xff" * 64) * 128)
SELF_TEST_PNG_SHA256 = "84e6d66a25cd99090fc41a34bc76cc4eee0afe0b3ae24545d1a52a25f7d08577"
SELF_TEST_IMAGE_URL = "data:image/png;base64," + base64.b64encode(SELF_TEST_PNG).decode()
# The control: the mirrored image (white on the left, black on the right); the expected answer is "left".
CONTROL_PNG = encode_png(128, 128, (b"\xff\xff\xff" * 64 + b"\x00\x00\x00" * 64) * 128)
CONTROL_PNG_SHA256 = "586f324daaea8baa6164b65cf85542e000768cf605fce015e17c65a6af879e42"
CONTROL_IMAGE_URL = "data:image/png;base64," + base64.b64encode(CONTROL_PNG).decode()
_QUESTION = {"q": {"type": "choice", "instructions": "Which half of the image is white?",
                   "criteria": {"left": None, "right": None}}}
VISION_SELF_TEST_BODY: dict[str, Any] = {"state": "", "images": [SELF_TEST_IMAGE_URL], "questions": _QUESTION}
VISION_SELF_TEST_CONTROL_BODY: dict[str, Any] = {"state": "", "images": [CONTROL_IMAGE_URL], "questions": _QUESTION}


@dataclass(frozen=True)
class VisionSelfTestResult:
    ok: bool
    answer: str | None         # for the main image (expected "right")
    option_mass: float | None
    latency_ms: float | None
    cache_n: int | None        # of the second, identical request
    prompt_n: int | None
    error: str | None
    control_answer: str | None = None       # for the mirrored control image (expected "left")
    control_option_mass: float | None = None

    def lines(self) -> list[str]:
        if self.error:
            return [f"WARNING: vision self-test could not run: {self.error}"]
        verdict = "ok" if self.ok else "FAILED"
        out = [f"vision self-test {verdict}: which half of the image is white? -> answer={self.answer} option mass="
               f"{self.option_mass:.3f} latency={self.latency_ms:.0f} ms; mirrored control image -> "
               f"answer={self.control_answer} option mass={self.control_option_mass:.3f}; identical repeat "
               f"cache_n={self.cache_n} prompt_n={self.prompt_n} (1 on pure attention; a hybrid model restores its "
               f"checkpoint 4 tokens before the end, so 4)"]
        if self.answer != "right":
            out.append(f"WARNING: the vision self-test answered {self.answer!r}, expected 'right': image input may "
                       f"not reach the model")
        if self.control_answer != "left":
            out.append(f"WARNING: the vision self-test control (mirrored image) answered {self.control_answer!r}, "
                       f"expected 'left': image input may not reach the model")
        if self.answer == self.control_answer:
            out.append("WARNING: the model gave the same answer for the image and its mirror: it may be ignoring "
                       "the image")
        if (self.option_mass is not None and self.option_mass < 0.5) or (
                self.control_option_mass is not None and self.control_option_mass < 0.5):
            out.append("WARNING: vision self-test option mass below 0.5: the scoring position may not be the answer "
                       "slot")
        return out


def self_test_answer(engine: Any, req: Any, id_slot: int | None) -> tuple[str, float, Any]:
    """One engine call for a self-test (vision here, audio and video in `callo.av`); (the argmax option key, its
    option mass, the Readout). Raises ValueError on a malformed readout."""
    q = req.questions[0]
    r = engine.answer(req.state, q, id_slot=id_slot, media=req.media)
    if not r.probs or len(r.probs) != len(q.keys):
        raise ValueError(f"readout has {len(r.probs)} probabilities for {len(q.keys)} options")
    if any(math.isnan(p) for p in r.probs):
        raise ValueError("readout probabilities contain NaN")
    if math.isnan(r.option_mass):
        raise ValueError("option mass is NaN")
    return q.keys[max(range(len(r.probs)), key=r.probs.__getitem__)], r.option_mass, r


def vision_self_test(engine: Any, *, id_slot: int | None = None) -> VisionSelfTestResult:
    """Ask which half of the built-in image is white (expect "right"), again identically (the repeat shows
    the cache), then ask about its mirror (expect "left"). Passes only when both answers are right and both option
    masses are at least 0.5. Warns loudly, never raises: callo still starts."""
    try:
        req = parse_request(VISION_SELF_TEST_BODY)
        control = parse_request(VISION_SELF_TEST_CONTROL_BODY)
        t0 = time.perf_counter()
        answer, mass, _ = self_test_answer(engine, req, id_slot)
        latency = 1000 * (time.perf_counter() - t0)
        _, _, again = self_test_answer(engine, req, id_slot)
        c_answer, c_mass, _ = self_test_answer(engine, control, id_slot)
        ok = answer == "right" and c_answer == "left" and mass >= 0.5 and c_mass >= 0.5
        return VisionSelfTestResult(ok, answer, mass, latency, again.cache_n, again.prompt_n, None, c_answer, c_mass)
    except Exception as e:  # noqa: BLE001 -- the self-test reports, it does not raise
        return VisionSelfTestResult(False, None, None, None, None, None, f"{type(e).__name__}: {e}")
