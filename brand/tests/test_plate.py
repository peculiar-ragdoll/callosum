"""The hero plate (brand/build.py `plate_svgs`, brand/tools/plate.py): its tagline says what the user chose, in two
centred lines that sit with the wordmark and clear the crown light's glow, and the committed SVGs are what the build
draws."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import pytest

import build
import plate as P

BRAND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = "{http://www.w3.org/2000/svg}"
LINES = ("Bridging quick decisions and deep work", "on the model you already serve")


@pytest.fixture(scope="module")
def plates():
    return build.plate_svgs()


def _tagline(svg: str) -> ET.Element:
    groups = [g for g in ET.fromstring(svg).iter(f"{NS}g") if g.get("data-text") is not None]
    assert len(groups) == 1, "the plate carries exactly one text group: the tagline"
    return groups[0]


def test_the_committed_plates_are_what_the_build_draws(plates):
    for name, svg in zip(("plate.svg", "plate-animated.svg"), plates, strict=True):
        with open(os.path.join(BRAND, name), encoding="utf-8") as f:
            assert f.read() == svg, f"{name}: re-run brand/build.py, then tools/export.py pngs and video"


def test_the_tagline_is_the_two_chosen_lines_in_mist(plates):
    assert build.PLATE_TAGLINE == LINES
    for svg in plates:
        g = _tagline(svg)
        assert g.get("data-text") == " ".join(LINES)
        assert g.get("fill") == build.MIST
        assert len(g.findall(f"{NS}path")) == 2, "one outlined path per line"


def _line_boxes():
    """Ink boxes of the two lines as the plate sets them (y down, plate units)."""
    lines = build.plate_tagline()
    assert [text for text, *_ in lines] == list(build.PLATE_TAGLINE)
    out = []
    for _, run, x, y in lines:
        x0, y0, x1, y1 = run.bbox()
        out.append((x0 + x, y0 + y, x1 + x, y1 + y))
    return out


def test_each_line_is_centred_on_the_plate():
    for x0, _, x1, _ in _line_boxes():
        assert abs((x0 + x1) / 2 - P.W / 2) < 0.01


def test_the_lines_read_as_one_block_with_the_wordmark_and_clear_the_glow():
    (l1x0, l1y0, l1x1, l1y1), (l2x0, l2y0, l2x1, l2y1) = _line_boxes()
    assert build.PLATE_TAGLINE_BASELINES[1] - build.PLATE_TAGLINE_BASELINES[0] == 22    # the leading the user saw
    wordmark_gap = l1y0 - P.WORD_Y                    # the wordmark has no descenders: its ink ends on the baseline
    assert 16 <= wordmark_gap <= 28, wordmark_gap
    pts, _ = P.arch_geometry(build.MARK_SIZES["display"])
    ux, uy, r = P.crown_track(pts)
    halo_top = uy - r - P.HALO_R                      # the light's highest point in the loop is the crown's top
    assert halo_top - l2y1 >= 16, halo_top - l2y1
    assert l1y1 < l2y0, "the lines do not touch"
