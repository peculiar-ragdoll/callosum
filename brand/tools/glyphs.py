"""Shape text with HarfBuzz and return outlined SVG paths (no font needed at render time).

shape(font_file, text, size, axes={...}, tracking=0) -> Run
A Run holds per-glyph path data (y-down, baseline at y=0), pen positions, ink bounds,
and the total advance.  Variable fonts are instanced with fontTools before shaping so
kerning matches the chosen axis values.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from functools import lru_cache

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

FONT_DIR = os.environ.get("CALLO_FONT_DIR", os.path.join(os.path.dirname(__file__), "..", "fonts"))


@lru_cache(maxsize=None)
def _instance(fname: str, axes: tuple):
    f = TTFont(fname if os.path.isabs(fname) else os.path.join(FONT_DIR, fname))
    if "fvar" in f:
        tags = {a.axisTag for a in f["fvar"].axes}
        loc = {k: v for k, v in axes if k in tags}
        f = instantiateVariableFont(f, loc, inplace=False, optimize=False)
    buf = io.BytesIO()
    f.save(buf)
    return f, buf.getvalue()


@dataclass
class G:
    name: str
    char: str
    x: float
    d: str
    adv: float
    bbox: tuple | None  # x0, y0, x1, y1 (y-down)


@dataclass
class Run:
    glyphs: list = field(default_factory=list)
    advance: float = 0.0
    upem: int = 1000
    scale: float = 1.0
    font: object = None

    def d(self, dx=0.0, dy=0.0):
        return " ".join(translate(g.d, dx, dy) for g in self.glyphs)

    def bbox(self):
        bs = [g.bbox for g in self.glyphs if g.bbox]
        return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))

    def metric(self, key):
        os2 = self.font["OS/2"]
        return {"cap": os2.sCapHeight, "x": os2.sxHeight}[key] * self.scale


def glyph_path(font, gname, scale, x=0.0, y=0.0):
    gs = font.getGlyphSet()
    pen = SVGPathPen(gs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
    tp = TransformPen(pen, (scale, 0, 0, -scale, x, y))
    gs[gname].draw(tp)
    bp = BoundsPen(gs)
    gs[gname].draw(bp)
    bb = None
    if bp.bounds:
        x0, y0, x1, y1 = bp.bounds
        bb = (x + x0 * scale, y - y1 * scale, x + x1 * scale, y - y0 * scale)
    return pen.getCommands(), bb


def shape(fname, text, size, axes=None, tracking=0.0, features=None):
    axes = dict(axes or {})
    font, blob = _instance(fname, tuple(sorted(axes.items())))
    upem = font["head"].unitsPerEm
    scale = size / upem
    face = hb.Face(hb.Blob(blob))
    hf = hb.Font(face)
    hf.scale = (upem, upem)
    b = hb.Buffer()
    b.add_str(text)
    b.guess_segment_properties()
    hb.shape(hf, b, features or {"kern": True, "liga": True})
    order = font.getGlyphOrder()
    run = Run(upem=upem, scale=scale, font=font)
    x = 0.0
    for info, pos in zip(b.glyph_infos, b.glyph_positions):
        gname = order[info.codepoint]
        d, bb = glyph_path(font, gname, scale, x + pos.x_offset * scale, -pos.y_offset * scale)
        adv = pos.x_advance * scale
        run.glyphs.append(G(gname, text[info.cluster], x, d, adv, bb))
        x += adv + tracking
    run.advance = x - tracking
    return run


def translate(d, dx, dy):
    """Translate absolute SVGPathPen output (M L C Q Z with absolute coords)."""
    if not dx and not dy:
        return d
    import re
    out, toks = [], re.findall(r"[MLCQZHV]|-?\d*\.?\d+(?:e-?\d+)?", d)
    cmd, i, k = None, 0, 0
    for t in toks:
        if t.isalpha():
            cmd, k = t, 0
            out.append(t)
            continue
        v = float(t)
        if cmd == "H":
            v += dx
        elif cmd == "V":
            v += dy
        else:
            v += dx if k % 2 == 0 else dy
            k += 1
        out.append(f"{v:.2f}".rstrip("0").rstrip("."))
    return " ".join(out)
