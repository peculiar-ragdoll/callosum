"""The README's "How it works" illustration (brand/how_it_works.py): it tells the story, in the brand palette, as
outlines, and the committed SVGs and PNGs are what the generator draws."""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import pytest

import figkit
import how_it_works as HIW

BRAND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THEMES = ("light", "dark")
NS = "{http://www.w3.org/2000/svg}"


def _root(theme: str) -> ET.Element:
    return ET.fromstring(HIW.draw(theme))


def _texts(theme: str) -> list[str]:
    return [g.get("data-text") for g in _root(theme).iter(f"{NS}g") if g.get("data-text") is not None]


@pytest.mark.parametrize("theme", THEMES)
def test_the_committed_svg_is_what_the_generator_draws(theme):
    with open(os.path.join(BRAND, f"how-it-works-{theme}.svg"), encoding="utf-8") as f:
        assert f.read() == HIW.draw(theme), "re-run brand/build.py (and tools/export.py pngs)"


@pytest.mark.parametrize("theme", THEMES)
def test_the_png_is_the_svg_at_2x_on_a_transparent_ground(theme):
    from PIL import Image
    im = Image.open(os.path.join(BRAND, f"how-it-works-{theme}.png"))
    assert im.size == (round(HIW.W * 2), round(HIW.H * 2))
    assert 1800 <= im.size[0] <= 2000
    assert im.mode == "RGBA"
    assert im.getpixel((0, 0))[3] == 0 and im.getpixel((im.size[0] - 1, im.size[1] - 1))[3] == 0


def test_a_landscape_that_reads_in_a_readme():
    assert 2.2 <= HIW.W / HIW.H <= 2.8


@pytest.mark.parametrize("theme", THEMES)
def test_it_tells_the_story(theme):
    said = _texts(theme)
    # both kinds of traffic enter callo serve
    for s in ("System One", "POST /v1/systemone", "Agent", "callo serve"):
        assert s in said
    # the proxy passes chat through; the engine gives one readout of the options
    assert "proxy" in said and "passes chat through" in said and "unchanged" in said
    assert "engine" in said
    for name, p in HIW.READOUT:
        assert name in said and f"{p:.2f}" in said
    assert abs(sum(p for _, p in HIW.READOUT) - 1.0) < 1e-9
    # one llama-server, one model, its slots split: the reserved low slot is the private lane, the rest chat
    for s in ("llama-server", "one model", "private lane", "slot 0, reserved", "chat", "other slots", "long prefill"):
        assert s in said
    # the measured note (README "Why": 39.7 s queued behind a 32k-token prefill; through callo, p50 2.15 s)
    assert HIW.NOTE_STOCK == "39.7 s queued" and HIW.NOTE_STOCK in said
    assert HIW.NOTE_CALLO == "2.15 s" and HIW.NOTE_CALLO in said
    assert HIW.NOTE_SUB in said and "p50" in HIW.NOTE_SUB and "32k-token prefill" in HIW.NOTE_SUB
    assert not any("slots 1" in s for s in said), "the chat slots are 'other slots', not a fixed count"


@pytest.mark.parametrize("theme", THEMES)
def test_every_colour_is_from_the_brand_palette(theme):
    svg = HIW.draw(theme)
    palette = {figkit.INK, figkit.BONE, figkit.SIGNAL, figkit.NIGHT, figkit.DUSK, figkit.TILE, figkit.MIST,
               figkit.SLATE}
    assert {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", svg)} <= palette
    rgb = {tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in palette}
    for r, g, b in re.findall(r"rgba\((\d+),(\d+),(\d+),", svg):
        assert (int(r), int(g), int(b)) in rgb


def test_small_text_on_the_light_file_is_slate_never_mist():
    for g in _root("light").iter(f"{NS}g"):
        if g.get("data-text") is not None:
            assert figkit.MIST not in ET.tostring(g).decode()


@pytest.mark.parametrize("theme", THEMES)
def test_every_glyph_is_an_outline(theme):
    svg = HIW.draw(theme)
    assert "<text" not in svg and "font-family" not in svg


def test_the_two_files_share_no_ids_so_one_page_can_inline_both():
    ids = [set(re.findall(r'id="([^"]+)"', HIW.draw(t))) for t in THEMES]
    assert ids[0] and ids[1] and not ids[0] & ids[1]
