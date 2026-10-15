"""Build every Callosum brand asset from code.

    python brand/build.py            # writes SVGs + preview.html into brand/
    python brand/tools/export.py     # renders PNGs, checks the loop seam, dumps frames for MP4/WebP

Nothing here depends on an installed font at view time: every glyph is shaped with HarfBuzz and
written as outlines. See README.md for the ideas behind each choice.
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools"))
os.environ.setdefault("CALLO_FONT_DIR", os.path.join(HERE, "fonts"))

from geom import bbox, c_grotesk, clip_halfplane, to_path  # noqa: E402
from glyphs import shape  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------------------------
INK = "#14161D"      # mark + type on light grounds
BONE = "#EEEAE1"     # mark + type on dark grounds
SIGNAL = "#FF4F3A"   # the quick light; the only accent
NIGHT = "#090B11"    # plate sky, top
DUSK = "#151B2B"     # plate sky at the horizon
DEEP = "#06070B"     # plate water
MIST = "#8D95A6"     # secondary text on dark
SLATE = "#5B6272"    # secondary text on light
TILE = "#0E1322"     # favicon / app tile

SANS = "InstrumentSans.ttf"
SERIF_IT = "Newsreader-Italic.ttf"
SANS_AXES = {"wght": 700, "wdth": 92}
SERIF_AXES = {"wght": 500, "opsz": 72}

# ---------------------------------------------------------------------------------------------
# Mark: the bridge. A reverse-contrast C, turned on its side and cut at its waist, exactly as it
# stands in the plate: two heavy piers, a hairline crown.
# ---------------------------------------------------------------------------------------------
# The bridge is generated from a C (an outer superellipse with a counter pushed toward the spine)
# rotated so the spine becomes the crown, then cut at the C's centre line, the plate's waterline.
# Optical sizes: the crown is a hairline at display sizes and is thickened for small renderings so
# it never drops out; the two smallest sizes also stand a little taller on longer piers, because a
# 2.2:1 arch is only 5 px tall at 16 px.
MARK_SIZES = {
    "display": dict(a=99, b=108, n=2.2, ia=73, ib=70, ini=2.05, dx=-20, m=31),
    "text":    dict(a=99, b=108, n=2.2, ia=72, ib=69, ini=2.05, dx=-17, m=31),
    "small":   dict(a=99, b=108, n=2.25, ia=67, ib=60, ini=2.05, dx=-18, m=30),
    "favicon": dict(a=99, b=108, n=2.25, ia=62, ib=55, ini=2.05, dx=-12, m=32),
}
MARK_LEGS = {"display": 0.0, "text": 0.0, "small": 0.14, "favicon": 0.30}  # pier extension, x arch height


def arch_raw(size="display"):
    """The bridge in its own units: springing line at y = 0, crown up (negative y)."""
    pts = c_grotesk(cx=0, cy=0, **MARK_SIZES[size])
    pts = [(-y, x) for x, y in pts]                      # spine -> crown
    pts = clip_halfplane(pts, lambda x, y: -y)           # cut at the waterline
    h = -min(y for _, y in pts)
    leg = MARK_LEGS[size] * h
    if leg:
        pts = [(x, leg if y > -1e-6 else y) for x, y in pts]
    return pts


def mark_pts(size="display", box=256.0, pad=0.0):
    """The bridge fitted to a square box: full width between the pads, optically centred
    (a hair low, since the eye reads the heavy piers as the arch's weight)."""
    pts = arch_raw(size)
    x0, y0, x1, y1 = bbox(pts)
    s = (box - 2 * pad) / (x1 - x0)
    h = (y1 - y0) * s
    oy = (box - h) / 2 + h * 0.04
    return [((x - x0) * s + pad, (y - y0) * s + oy) for x, y in pts]


def mark_path(size="display", box=256.0, pad=0.0):
    return to_path(mark_pts(size, box, pad))


# ---------------------------------------------------------------------------------------------
# Wordmark: "Callo" upright grotesk + "sum" serif italic. One word, two minds.
# ---------------------------------------------------------------------------------------------
def _xh(fname, axes):
    r = shape(fname, "x", 1000, axes)
    return -r.glyphs[0].bbox[1]


def wordmark(size=100.0, join=-0.004):
    """Return (d_callo, d_sum, advance, cap, xheight) with the baseline at y=0, x starting at 0.
    The italic is scaled so its x-height matches the grotesk's exactly; `join` nudges the italic s
    toward the o (in em) so the two halves read as one word."""
    a = shape(SANS, "Callo", size, SANS_AXES)
    k = _xh(SANS, SANS_AXES) / _xh(SERIF_IT, SERIF_AXES)
    b = shape(SERIF_IT, "sum", size * k, SERIF_AXES)
    xb = a.advance + join * size
    cap = a.metric("cap")
    xh = _xh(SANS, SANS_AXES) * size / 1000
    bx = b.bbox()
    return a.d(), b.d(xb, 0), xb + bx[2], cap, xh, a, b


# ---------------------------------------------------------------------------------------------
# Lockup
# ---------------------------------------------------------------------------------------------
LOCK = dict(size=100.0, mark_h=1.04, gap=0.38, clear=0.5)  # in cap heights


def _lockup_parts(size):
    from glyphs import translate
    dc, ds, wadv, cap, xh, a, b = wordmark(size)
    return dc, ds, wadv, cap, a, b, translate


def lockup_svg(fg, title="Callosum"):
    """Horizontal lockup. The bridge stands on the wordmark's baseline, piers down, like the
    letters; its crown overshoots the cap height by 4 %, the way round letters overshoot, so the
    hairline doesn't look short. Gap 0.38 cap: the pier's straight inner face reads as a stem, so
    it is spaced like a word space tightened, not like the old C's open mouth."""
    size = LOCK["size"]
    dc, ds, wadv, cap, a, b, translate = _lockup_parts(size)
    clear = cap * LOCK["clear"]
    pts = arch_raw("text")
    x0, y0, x1, y1 = bbox(pts)
    s = cap * LOCK["mark_h"] / (y1 - y0)
    aw = (x1 - x0) * s
    gap = cap * LOCK["gap"]
    lb = a.bbox()[0]
    top = -cap * LOCK["mark_h"]
    bot = b.bbox()[3]
    W = clear + aw + gap + wadv - lb + clear
    H = clear + (bot - top) + clear
    ox, oy = clear, clear - top            # baseline at y = oy
    mark_d = to_path([(ox + (x - x0) * s, oy + (y - y1) * s) for x, y in pts])
    wx = ox + aw + gap - lb
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.1f} {H:.1f}" width="{W:.0f}" height="{H:.0f}" '
            f'role="img" aria-label="{title}"><title>{title}</title><g fill="{fg}"><path d="{mark_d}"/>'
            f'<path d="{translate(dc, wx, oy)}"/><path d="{translate(ds, wx, oy)}"/></g></svg>')


def lockup_stacked_svg(fg, title="Callosum"):
    """Stacked lockup: the bridge centred over the word, about as wide as 'Callo' (0.44 of the
    word), standing 0.5 cap above the cap height."""
    size = LOCK["size"]
    dc, ds, wadv, cap, a, b, translate = _lockup_parts(size)
    clear = cap * LOCK["clear"]
    pts = arch_raw("text")
    x0, y0, x1, y1 = bbox(pts)
    lb = a.bbox()[0]
    word_w = wadv - lb
    aw = word_w * 0.44
    s = aw / (x1 - x0)
    ah = (y1 - y0) * s
    gap = cap * 0.5
    W = clear + word_w + clear
    H = clear + ah + gap + cap + b.bbox()[3] + clear
    ax = clear + (word_w - aw) / 2
    mark_d = to_path([(ax + (x - x0) * s, clear + (y - y0) * s) for x, y in pts])
    base = clear + ah + gap + cap
    wx = clear - lb
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.1f} {H:.1f}" width="{W:.0f}" height="{H:.0f}" '
            f'role="img" aria-label="{title}"><title>{title}</title><g fill="{fg}"><path d="{mark_d}"/>'
            f'<path d="{translate(dc, wx, base)}"/><path d="{translate(ds, wx, base)}"/></g></svg>')


def mark_svg(fg, size="display", box=256, pad=16, bg=None, rx=0, light=False):
    """Square mark. With light=True the crown carries the plate's resting light: a short Signal
    segment at the top of the span, as in plate.svg at t = 0 (display size only; below 64 px the
    crown is too thin to hold a colour, so small renderings stay one colour)."""
    pts = mark_pts(size, box, pad)
    d = to_path(pts)
    bgel = f'<rect width="{box}" height="{box}" rx="{rx}" fill="{bg}"/>' if bg else ""
    lit = ""
    if light:
        x0, y0, x1, y1 = bbox(pts)
        cw = (x1 - x0) * 0.075
        lit = (f'<clipPath id="span"><path d="{d}"/></clipPath>'
               f'<rect x="{box / 2 - cw / 2:.2f}" y="0" width="{cw:.2f}" height="{box / 2:.2f}" fill="{SIGNAL}" clip-path="url(#span)"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {box} {box}" width="{box}" height="{box}" '
            f'role="img" aria-label="Callosum"><title>Callosum</title>{bgel}<path d="{d}" fill="{fg}"/>{lit}</svg>')


def write(name, text):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(text)


# The plate's tagline: two centred lines under the wordmark (baseline 124). They sit as one block with it and clear the
# glow of the light at the top of the crown (tests/test_plate.py).
PLATE_TAGLINE = ("Bridging quick decisions and deep work", "on the model you already serve")
PLATE_TAGLINE_BASELINES = (158, 180)


def plate_tagline():
    """[(text, run, x, baseline)] per tagline line, each line centred on its ink."""
    import plate as P
    out = []
    for text, y in zip(PLATE_TAGLINE, PLATE_TAGLINE_BASELINES, strict=True):
        run = shape(SANS, text, 15.5, {"wght": 480, "wdth": 100}, tracking=0.35)
        x0, _, x1, _ = run.bbox()
        out.append((text, run, P.W / 2 - (x0 + x1) / 2, y))
    return out


def plate_svgs():
    import plate as P
    dc, ds, wadv, cap, xh, a, b = wordmark(72)
    word = (dc, ds, wadv, cap)
    caption = (f'<g data-text="{" ".join(PLATE_TAGLINE)}" fill="{MIST}">'
               + "".join(f'<path d="{run.d(x, y)}"/>' for _, run, x, y in plate_tagline()) + "</g>")
    return (P.scene(MARK_SIZES["display"], word, animated=False, caption=caption),
            P.scene(MARK_SIZES["display"], word, animated=True, caption=caption))


def main():
    st, an = plate_svgs()
    write("plate.svg", st)
    write("plate-animated.svg", an)
    write("lockup-light.svg", lockup_svg(INK))
    write("lockup-dark.svg", lockup_svg(BONE))
    write("lockup-stacked-light.svg", lockup_stacked_svg(INK))
    write("lockup-stacked-dark.svg", lockup_stacked_svg(BONE))
    write("mark-light.svg", mark_svg(INK, pad=20))
    write("mark-dark.svg", mark_svg(BONE, pad=20))
    # favicons: a night tile so one file works on light and dark browser chrome
    write("favicon.svg", mark_svg(BONE, size="small", box=256, pad=40, bg=TILE, rx=58))
    write("favicon-16.svg", mark_svg(BONE, size="favicon", box=256, pad=26, bg=TILE, rx=52))
    # the README's "How it works" illustration (how_it_works.py), transparent, one file per GitHub theme
    import how_it_works
    for v in ("light", "dark"):
        write(f"how-it-works-{v}.svg", how_it_works.draw(v))
    write("preview.html", preview_html())
    print("ok")




# ---------------------------------------------------------------------------------------------
# Preview page (self-contained: every SVG inlined; the animated plate as a data-URI <img> so it
# animates exactly as it will inside a GitHub README)
# ---------------------------------------------------------------------------------------------
def preview_html():
    import base64
    import re

    def rd(n):
        with open(os.path.join(HERE, n)) as f:
            return f.read()

    def sized(svg, h=None, w=None):
        attr = f'height="{h}"' if h else f'width="{w}"'
        return re.sub(r'width="[\d.]+" height="[\d.]+"', attr, svg, count=1)

    anim = "data:image/svg+xml;base64," + base64.b64encode(rd("plate-animated.svg").encode()).decode()
    static = "data:image/svg+xml;base64," + base64.b64encode(rd("plate.svg").encode()).decode()
    ll, ld = rd("lockup-light.svg"), rd("lockup-dark.svg")
    ml, md = rd("mark-light.svg"), rd("mark-dark.svg")
    fav = rd("favicon.svg")
    fav16 = rd("favicon-16.svg")
    hiw_l, hiw_d = rd("how-it-works-light.svg"), rd("how-it-works-dark.svg")

    def strip(svg, bg, fg_note):
        return (f'<div class="strip" style="background:{bg};color:{fg_note}">'
                + "".join(f'<div class="cell">{sized(svg, h=h)}<span>{h}px</span></div>' for h in (96, 48, 32, 24)) + "</div>")

    def marks(fg, bg, fg_note):
        # each size uses its own optical size, as the exported files do
        opt = {256: ("display", False), 64: ("text", False), 32: ("small", False), 16: ("favicon", False)}
        return (f'<div class="strip" style="background:{bg};color:{fg_note}">'
                + "".join(f'<div class="cell">{sized(mark_svg(fg, size=o, pad=20, light=lt).replace("id=" + chr(34) + "span", "id=" + chr(34) + "span" + str(s) + fg[1:]).replace("#span)", "#span" + str(s) + fg[1:] + ")"), h=s)}<span>{s}</span></div>'
                          for s, (o, lt) in opt.items())
                + "</div>")

    sw = [("Ink", INK, "mark and type on light"), ("Bone", BONE, "mark and type on dark"),
          ("Signal", SIGNAL, "the quick light; the only accent"), ("Night", NIGHT, "plate sky"),
          ("Dusk", "#2A3559", "plate horizon"), ("Tile", TILE, "favicon tile"), ("Mist", MIST, "secondary on dark"),
          ("Slate", SLATE, "secondary on light")]
    swatches = "".join(f'<div class="sw"><i style="background:{c}"></i><b>{n}</b><code>{c}</code><small>{d}</small></div>'
                       for n, c, d in sw)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Callosum brand</title>
<link rel="icon" href="data:image/svg+xml;base64,{base64.b64encode(fav.encode()).decode()}">
<style>
:root{{--paper:#F4F1EA;--ink:{INK};--slate:{SLATE}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:48px 24px 96px}}
h2{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--slate);margin:64px 0 16px;font-weight:600}}
p.lede{{max-width:720px;color:#3a3f4b;margin:16px 0 0}}
.plate{{width:100%;display:block;border-radius:14px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .pair figure{{margin:0}} figcaption{{font-size:13px;color:var(--slate);margin-top:6px}}
.strip{{display:flex;align-items:flex-end;gap:40px;flex-wrap:wrap;padding:36px 32px;border-radius:14px;margin-bottom:12px}}
.cell{{display:flex;flex-direction:column;gap:10px;align-items:flex-start}} .cell span{{font-size:11px;opacity:.6}}
.sws{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px}}
.sw{{background:#fff;border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:2px;font-size:13px}}
.sw i{{display:block;height:64px;border-radius:8px;margin-bottom:8px;border:1px solid #0001}} .sw small{{color:var(--slate)}}
.fav{{display:flex;gap:24px;align-items:center}} .fav img{{image-rendering:pixelated}}
video{{width:100%;border-radius:14px;display:block}}
@media (max-width:760px){{.pair{{grid-template-columns:1fr}}}}
</style></head><body><main>
{sized(ll, h=64)}
<p class="lede">A bridge over slow water. The quick light crosses the crown while the slow tide moves underneath:
System&nbsp;One judgements passing over System&nbsp;Two work, inside one model. The mark is that bridge.</p>

<h2>Animated plate (SVG, CSS only)</h2>
<img class="plate" src="{anim}" alt="Callosum animated plate">

<h2>Static plate and video renders</h2>
<div class="pair"><figure><img class="plate" src="{static}" alt="Callosum static plate"><figcaption>plate.svg / plate@2x.png: the animation at rest</figcaption></figure>
<figure><video src="plate-loop.mp4" autoplay muted loop playsinline></video><figcaption>plate-loop.mp4 (also plate-loop.webp)</figcaption></figure></div>

<h2>Lockup</h2>
{strip(ll, "#ffffff", "#000")}
{strip(ld, "#0d1117", "#fff")}

<h2>Mark</h2>
{marks(INK, "#ffffff", "#000")}
{marks(BONE, "#0d1117", "#fff")}

<h2>Stacked lockup</h2>
<div class="pair"><div class="strip" style="background:#fff;justify-content:center">{sized(rd("lockup-stacked-light.svg"), h=150)}</div>
<div class="strip" style="background:#0d1117;justify-content:center">{sized(rd("lockup-stacked-dark.svg"), h=150)}</div></div>

<h2>Favicon</h2>
<div class="strip" style="background:#fff;color:#000"><div class="fav">{sized(fav, h=32)}{sized(fav16, h=16)}
<img src="favicon-32.png" width="64" height="64" alt="favicon 32 at 2x"><img src="favicon-16.png" width="64" height="64" alt="favicon 16 at 4x"></div></div>
<div class="strip" style="background:#0d1117;color:#fff"><div class="fav">{sized(fav, h=32)}{sized(fav16, h=16)}</div></div>

<h2>How it works (README illustration)</h2>
<div class="strip" style="background:#ffffff;display:block">{sized(hiw_l, w="100%")}</div>
<div class="strip" style="background:#0d1117;display:block">{sized(hiw_d, w="100%")}</div>

<h2>Palette</h2>
<div class="sws">{swatches}</div>
</main></body></html>"""


if __name__ == "__main__":
    main()
