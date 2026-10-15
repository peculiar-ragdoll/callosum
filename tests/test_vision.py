"""The built-in two-tone PNG (bytes pinned) and the vision self-test. (v0.3: the self-test calls
Engine.answer with media=, so the fake engine takes media; detection from /props is detect_media, tested in
test_media.py, and v0.2's detect_vision with its test is removed.)"""
import base64
import dataclasses
import hashlib
import struct
import zlib

import pytest

from callo.engine import Readout
from callo.vision import (
    CONTROL_IMAGE_URL,
    CONTROL_PNG,
    CONTROL_PNG_SHA256,
    SELF_TEST_IMAGE_URL,
    SELF_TEST_PNG,
    SELF_TEST_PNG_SHA256,
    VISION_SELF_TEST_BODY,
    VISION_SELF_TEST_CONTROL_BODY,
    encode_png,
    vision_self_test,
)


def _decode(png: bytes) -> tuple[int, int, bytes]:
    """(width, height, raw scanlines) of an 8-bit RGB PNG, with every CRC checked."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    i, chunks = 8, {}
    while i < len(png):
        n = struct.unpack(">I", png[i:i + 4])[0]
        kind, data = png[i + 4:i + 8], png[i + 8:i + 8 + n]
        assert struct.unpack(">I", png[i + 8 + n:i + 12 + n])[0] == zlib.crc32(kind + data) & 0xFFFFFFFF
        chunks[kind] = chunks.get(kind, b"") + data
        i += 12 + n
    w, h, depth, color, *_ = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
    assert (depth, color) == (8, 2) and b"IEND" in chunks
    return w, h, zlib.decompress(chunks[b"IDAT"])


def test_the_self_test_png_bytes_are_pinned():
    assert hashlib.sha256(SELF_TEST_PNG).hexdigest() == SELF_TEST_PNG_SHA256 == \
        "84e6d66a25cd99090fc41a34bc76cc4eee0afe0b3ae24545d1a52a25f7d08577"
    assert SELF_TEST_IMAGE_URL == "data:image/png;base64," + base64.b64encode(SELF_TEST_PNG).decode()


def test_the_self_test_png_is_128_square_black_left_white_right():
    w, h, raw = _decode(SELF_TEST_PNG)
    assert (w, h) == (128, 128) and len(raw) == 128 * (1 + 128 * 3)
    row = raw[:1 + 128 * 3]
    assert row[0] == 0 and row[1:1 + 64 * 3] == b"\x00" * 192 and row[1 + 64 * 3:] == b"\xff" * 192
    assert raw == row * 128


def test_encode_png_handles_more_than_one_stored_block_and_checks_its_input():
    rgb = bytes(range(256)) * (200 * 150 * 3 // 256) + bytes(200 * 150 * 3 % 256)
    w, h, raw = _decode(encode_png(200, 150, rgb))       # 90,150 raw bytes: two stored blocks
    assert (w, h) == (200, 150) and b"".join(raw[y * 601 + 1:(y + 1) * 601] for y in range(150)) == rgb
    with pytest.raises(ValueError):
        encode_png(2, 2, b"\x00" * 11)


def test_the_self_test_question():
    q = VISION_SELF_TEST_BODY["questions"]["q"]
    assert q["instructions"] == "Which half of the image is white?" and q["criteria"] == {"left": None, "right": None}
    assert VISION_SELF_TEST_BODY["images"] == [SELF_TEST_IMAGE_URL] and VISION_SELF_TEST_BODY["state"] == ""


class Eng:
    """probs=None answers correctly per image (white right -> 'right', mirrored -> 'left'); a tuple answers it always."""

    def __init__(self, probs=None, mass=0.99, exc=None, exc_on=None):
        self.probs, self.mass, self.exc, self.exc_on, self.calls = probs, mass, exc, exc_on, []

    def answer(self, state, q, *, id_slot=None, context="", media=()):
        images, image_urls = tuple(m.payload for m in media), tuple(m.url for m in media)   # v0.3: media=
        self.calls.append((state, q.keys, id_slot, images, image_urls))
        n = len(self.calls)
        if self.exc and (self.exc_on is None or n == self.exc_on):
            raise self.exc
        probs = self.probs
        if probs is None:
            probs = (0.02, 0.98) if tuple(image_urls) == (SELF_TEST_IMAGE_URL,) else (0.97, 0.03)
        return Readout(probs, (-3.9, -0.02), self.mass, "diverged", 0, 0, 120, kind="image",
                       cache_n=0 if n == 1 else 118, prompt_n=120 if n == 1 else 2, prompt_ms=50.0)


def test_the_control_png_is_the_pinned_mirror_image():
    assert hashlib.sha256(CONTROL_PNG).hexdigest() == CONTROL_PNG_SHA256 == \
        "586f324daaea8baa6164b65cf85542e000768cf605fce015e17c65a6af879e42"
    assert CONTROL_IMAGE_URL == "data:image/png;base64," + base64.b64encode(CONTROL_PNG).decode()
    w, h, raw = _decode(CONTROL_PNG)
    row = raw[:1 + 128 * 3]
    assert (w, h) == (128, 128) and row[0] == 0 and row[1:1 + 64 * 3] == b"\xff" * 192 and row[1 + 64 * 3:] == b"\x00" * 192
    assert raw == row * 128 and CONTROL_PNG != SELF_TEST_PNG
    q = VISION_SELF_TEST_CONTROL_BODY["questions"]["q"]
    assert q == VISION_SELF_TEST_BODY["questions"]["q"] and VISION_SELF_TEST_CONTROL_BODY["images"] == [CONTROL_IMAGE_URL]


def test_vision_self_test_passes_and_reports_the_repeat_cache_numbers():
    e = Eng()
    r = vision_self_test(e, id_slot=0)
    assert r.ok and r.answer == "right" and r.option_mass == 0.99 and (r.cache_n, r.prompt_n) == (118, 2)
    assert r.control_answer == "left" and r.control_option_mass == 0.99
    assert r.latency_ms is not None and r.error is None
    assert len(e.calls) == 3 and e.calls[0] == e.calls[1] and e.calls[2] != e.calls[0]
    state, keys, slot, images, urls = e.calls[0]
    assert state == "" and keys == ("left", "right") and slot == 0 and urls == (SELF_TEST_IMAGE_URL,)
    assert images == (base64.b64encode(SELF_TEST_PNG).decode(),)
    assert e.calls[2][2:] == (0, (base64.b64encode(CONTROL_PNG).decode(),), (CONTROL_IMAGE_URL,))
    lines = r.lines()
    assert lines[0].startswith("vision self-test ok") and "answer=right" in lines[0] and "cache_n=118" in lines[0]
    assert "answer=left" in lines[0]
    assert not any(line.startswith("WARNING") for line in lines)


def test_a_wrong_answer_or_low_mass_warns_loudly_but_does_not_raise():
    r = vision_self_test(Eng(probs=(0.9, 0.1)))      # always 'left': wrong on the main image
    assert not r.ok and r.answer == "left"
    assert any(line.startswith("WARNING") and "expected 'right'" in line for line in r.lines())
    r = vision_self_test(Eng(mass=0.3))
    assert not r.ok and r.answer == "right"
    assert any(line.startswith("WARNING") and "option mass" in line for line in r.lines())


def test_an_image_blind_model_that_always_answers_right_fails_on_the_control():
    for probs in [(0.02, 0.98), (0.4, 0.6)]:
        r = vision_self_test(Eng(probs=probs))
        assert not r.ok and r.answer == "right" and r.control_answer == "right"
        lines = r.lines()
        assert lines[0].startswith("vision self-test FAILED") and "answer=right" in lines[0]
        assert any(line.startswith("WARNING") and "expected 'left'" in line for line in lines)
        assert any(line.startswith("WARNING") and "same answer" in line for line in lines)


def test_a_low_mass_on_the_control_alone_fails():
    class Low(Eng):
        def answer(self, state, q, **kw):
            r = super().answer(state, q, **kw)
            return r if len(self.calls) < 3 else dataclasses.replace(r, option_mass=0.2)
    r = vision_self_test(Low())
    assert not r.ok and r.answer == "right" and r.control_answer == "left" and r.control_option_mass == 0.2
    assert any(line.startswith("WARNING") and "option mass" in line for line in r.lines())


def test_a_failing_upstream_is_reported_not_raised():
    r = vision_self_test(Eng(exc=ConnectionRefusedError("nope")))
    assert not r.ok and "ConnectionRefusedError" in r.error
    assert r.lines() == ["WARNING: vision self-test could not run: ConnectionRefusedError: nope"]


@pytest.mark.parametrize("exc_on", [2, 3])
def test_an_exception_on_the_repeat_or_control_call_is_reported_not_raised(exc_on):
    r = vision_self_test(Eng(exc=TimeoutError("slow"), exc_on=exc_on))
    assert not r.ok and r.error == "TimeoutError: slow"
    assert r.lines() == ["WARNING: vision self-test could not run: TimeoutError: slow"]


@pytest.mark.parametrize(("probs", "mass", "msg"), [
    ((), 0.9, "readout has 0 probabilities for 2 options"),
    ((1.0,), 0.9, "readout has 1 probabilities for 2 options"),
    ((0.2, 0.3, 0.5), 0.9, "readout has 3 probabilities for 2 options"),
    ((0.2, float("nan")), 0.9, "NaN"),
    ((0.02, 0.98), float("nan"), "option mass is NaN"),
])
def test_a_malformed_readout_is_an_error_result_never_a_raise(probs, mass, msg):
    r = vision_self_test(Eng(probs=probs, mass=mass))
    assert not r.ok and r.answer is None and msg in r.error
    assert r.lines()[0].startswith("WARNING: vision self-test could not run: ValueError: ") and msg in r.lines()[0]
