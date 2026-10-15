"""callo's "How it works" illustration for the README: one front door, one model, a private lane.

    python brand/how_it_works.py [--png] [--out DIR]   # -> brand/how-it-works-{light,dark}.svg (+ .png at 2x)

`build.py` writes the two SVGs with every other asset and `tools/export.py pngs` renders the PNGs (2x, transparent:
the light file is for GitHub's #ffffff page, the dark one for #0d1117).

The story, left to right: the agent's chat traffic and a quick System One question both enter `callo serve`. The
proxy passes chat through unchanged; the engine turns the question into one readout of option probabilities. Both go
to ONE llama-server running ONE model, whose slots are split: the lowest slot is the classification's private lane,
the rest carry chat. llama-server fills each batch slot by slot in id order, so the private lane is served first and
a quick question never queues behind the agent's long prefill (the measured note, as README "Why" quotes it: a
classification behind a 32k-token prefill waited 39.7 s on stock llama-server; through callo, p50 2.15 s over 100
requests).

The drawing borrows the plate's vocabulary: chat is the slow tide (rows of ripple dashes), the System One question is
the quick coral light (Signal, the only accent), and `callo serve` carries the bridge mark.

Every text group carries `data-text` (figkit) so tests read what the figure says from the SVG.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import figkit as K  # noqa: E402
from figkit import BONE, INK, MIST, SIGNAL, SLATE, Style, num, text  # noqa: E402

W, H = 920.0, 344.0
TITLE = "How Callo works: one front door, one model, a private lane for quick questions"
NOTE_STOCK = "39.7 s queued"
NOTE_CALLO = "2.15 s"
NOTE_SUB = "stock llama-server \u2192 Callo\u2019s private lane, p50 behind a 32k-token prefill"
READOUT = (("positive", 0.82), ("negative", 0.11), ("neutral", 0.07))

# Colours: the brand palette only (README.md "Palette"); faint fills are the secondary text colour at low alpha.
THEMES: dict[str, dict[str, str]] = {
    "light": {"ink": INK, "sub": SLATE, "accent": SIGNAL,
              "line": "rgba(91,98,114,0.62)", "hair": "rgba(91,98,114,0.30)", "panel": "rgba(91,98,114,0.045)",
              "lane": "rgba(91,98,114,0.075)", "tide": "rgba(91,98,114,0.55)", "tide2": "rgba(91,98,114,0.30)",
              "bar": "rgba(91,98,114,0.42)", "lane_hot": "rgba(255,79,58,0.09)", "glow": "0.30"},
    "dark": {"ink": BONE, "sub": MIST, "accent": SIGNAL,
             "line": "rgba(141,149,166,0.62)", "hair": "rgba(141,149,166,0.28)", "panel": "rgba(141,149,166,0.05)",
             "lane": "rgba(141,149,166,0.085)", "tide": "rgba(141,149,166,0.62)", "tide2": "rgba(141,149,166,0.32)",
             "bar": "rgba(141,149,166,0.48)", "lane_hot": "rgba(255,79,58,0.11)", "glow": "0.45"},
}


def _r(x: float) -> str:
    return num(x)


def rrect(x: float, y: float, w: float, h: float, rx: float, *, fill: str = "none", stroke: str | None = None,
          sw: float = 1.0) -> str:
    s = f' stroke="{stroke}" stroke-width="{_r(sw)}"' if stroke else ""
    return f'<rect x="{_r(x)}" y="{_r(y)}" width="{_r(w)}" height="{_r(h)}" rx="{_r(rx)}" fill="{fill}"{s}/>'


def arrowhead(x: float, y: float, color: str, size: float = 6.0) -> str:
    """A small open chevron pointing right with its tip at (x, y)."""
    return (f'<path d="M{_r(x - size)} {_r(y - size * 0.8)} L{_r(x)} {_r(y)} L{_r(x - size)} {_r(y + size * 0.8)}" '
            f'fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>')


def line(d: str, color: str, sw: float = 1.6) -> str:
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{_r(sw)}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>')


def tide(x0: float, x1: float, y: float, color: str, color2: str, seed: int, fill: float) -> str:
    """One lane of slow water: ripple dashes like the plate's, filling `fill` of the lane from the left."""
    rnd = random.Random(seed)
    out = []
    end = x0 + (x1 - x0) * fill
    for row, dy in enumerate((-4.0, 0.0, 4.0)):
        x = x0 + rnd.uniform(0, 6)
        while x < end - 4:
            w = rnd.uniform(10, 34)
            w = min(w, end - x)
            c = color if row == 1 else color2
            out.append(f'<path d="M{_r(x)} {_r(y + dy)} H{_r(x + w)}" stroke="{c}" stroke-width="1.6" '
                       f'stroke-linecap="round"/>')
            x += w + rnd.uniform(4, 9)
    return "".join(out)


def bridge(x: float, y_base: float, h: float, color: str) -> tuple[str, float]:
    """The mark standing on a baseline like a letter, `h` tall, one colour: returns (path, width). At about 13 px it
    takes the `small` optical size (heavier crown, longer piers), as the 32 px favicon does."""
    from build import arch_raw
    from geom import bbox, to_path
    pts = arch_raw("small")
    x0, y0, x1, y1 = bbox(pts)
    s = h / (y1 - y0)
    d = to_path([(x + (px - x0) * s, y_base + (py - y1) * s) for px, py in pts])
    return f'<path d="{d}" fill="{color}"/>', (x1 - x0) * s


def light(cx: float, cy: float, t: dict, *, w: float = 22.0, h: float = 8.0, tail: float = 0.0, gid: str = "lt") -> str:
    """The quick light: a short Signal pill with a soft glow and, moving, a comet tail pointing the way it came (as
    on the plate's crown)."""
    defs = (f'<radialGradient id="{gid}-g" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="{SIGNAL}" '
            f'stop-opacity="{t["glow"]}"/><stop offset="1" stop-color="{SIGNAL}" stop-opacity="0"/></radialGradient>')
    body = f'<ellipse cx="{_r(cx)}" cy="{_r(cy)}" rx="{_r(w * 1.3)}" ry="{_r(h * 2.2)}" fill="url(#{gid}-g)"/>'
    if tail:
        defs += (f'<linearGradient id="{gid}-t" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{SIGNAL}" '
                 f'stop-opacity="0"/><stop offset="1" stop-color="{SIGNAL}" stop-opacity="0.55"/></linearGradient>')
        th = h * 0.5
        body += rrect(cx - w / 2 - tail, cy - th / 2, tail + w / 2, th, th / 2, fill=f"url(#{gid}-t)")
    body += rrect(cx - w / 2, cy - h / 2, w, h, h / 2, fill=SIGNAL)
    return f"<defs>{defs}</defs>{body}"


def draw(theme: str = "light") -> str:
    t = THEMES[theme]
    ink, sub, acc = t["ink"], t["sub"], t["accent"]
    title_st = Style(size=17, weight=650, fill=ink)
    head_st = Style(size=15, weight=620, fill=ink)
    sub_st = Style(size=13, weight=440, fill=sub)
    small_st = Style(size=12, weight=470, fill=sub)
    italic = Style(size=15, weight=500, fill=sub, face="serif")
    lane_name = Style(size=15, weight=500, fill=ink, face="serif")
    out: list[str] = []

    top, bot = 28.0, 316.0
    y_s1 = 118.0                     # the System One row: engine, slot 0
    lane_gap = 30.0
    y_chat = 254.0                   # the chat row: proxy, the other slots (centre lane)

    # ---- clients ---------------------------------------------------------------------------------------------------
    cx0 = 24.0
    tx = cx0 + 26
    out.append(light(cx0 + 8, y_s1 - 6, t, w=15, h=7, gid=f"in-{theme}"))
    out.append(text("System One", tx, y_s1 - 1, head_st, role="client")[0])
    out.append(text("POST /v1/systemone", tx, y_s1 + 18, sub_st, role="client-sub")[0])
    for i, (dy, ww) in enumerate(((-10, 17), (-6, 12), (-2, 17))):
        c = t["tide"] if i == 1 else t["tide2"]
        out.append(f'<path d="M{_r(cx0)} {_r(y_chat + dy)} H{_r(cx0 + ww)}" stroke="{c}" stroke-width="1.6" '
                   f'stroke-linecap="round"/>')
    out.append(text("Agent", tx, y_chat - 1, head_st, role="client")[0])
    out.append(text("chat, long prompts", tx, y_chat + 18, sub_st, role="client-sub")[0])
    client_right = tx + max(K.advance("POST /v1/systemone", sub_st), K.advance("chat, long prompts", sub_st))

    # ---- callo serve: the front door ---------------------------------------------------------------------------------
    gx0, gx1 = 216.0, 404.0
    pad = 18.0
    out.append(rrect(gx0, top, gx1 - gx0, bot - top, 14, fill=t["panel"], stroke=t["hair"], sw=1.2))
    mark, mw = bridge(gx0 + pad, top + 34, 13.0, ink)
    out.append(mark)
    out.append(text("callo serve", gx0 + pad + mw + 8, top + 34, title_st, role="node")[0])
    ex = gx0 + pad
    # engine: the question becomes one readout of the options
    out.append(text("engine", ex, y_s1 + 5, head_st, role="row")[0])
    ry = y_s1 + 31
    lab_w = max(K.advance(n, small_st) for n, _ in READOUT)
    val_w = K.advance("0.00", small_st)
    bar_x = ex + lab_w + 9
    bar_max = (gx1 - pad) - val_w - 8 - bar_x
    for i, (name, p) in enumerate(READOUT):
        yy = ry + i * 16.5
        st = small_st.with_(fill=ink if i == 0 else sub, weight=600 if i == 0 else 470)
        out.append(text(name, ex, yy, st, role="readout-label")[0])
        out.append(rrect(bar_x, yy - 7.5, bar_max, 5, 2.5, fill=t["lane"]))
        out.append(rrect(bar_x, yy - 7.5, max(bar_max * p, 5), 5, 2.5, fill=acc if i == 0 else t["bar"]))
        out.append(text(f"{p:.2f}", gx1 - pad, yy, st, anchor="end", role="readout-value")[0])
    # proxy: chat runs straight through
    out.append(text("proxy", ex, y_chat - 13, head_st, role="row")[0])
    out.append(text("passes chat through", ex, y_chat + 24, small_st, role="row-sub")[0])
    out.append(text("unchanged", ex, y_chat + 40, small_st, role="row-sub")[0])

    # ---- llama-server: one model, its slots split --------------------------------------------------------------------
    lx0, lx1 = 452.0, 896.0
    out.append(rrect(lx0, top, lx1 - lx0, bot - top, 14, fill=t["panel"], stroke=t["hair"], sw=1.2))
    out.append(text("llama-server", lx0 + 20, top + 34, title_st, role="node")[0])
    tw = K.advance("llama-server", title_st)
    out.append(text("one model", lx0 + 20 + tw + 9, top + 34, italic, role="node-sub")[0])
    lane_x0, lane_x1, lane_h = lx0 + 20, lx1 - 20, 20.0

    def lane_head(name: str, note: str, y: float) -> None:
        out.append(text(name, lane_x0 + 2, y, lane_name, role="lane-name")[0])
        out.append(text(note, lane_x1 - 2, y, small_st, anchor="end", role="lane-slots")[0])

    # slot 0: the private lane, the quick light on it
    lane_head("private lane", "slot 0, reserved", y_s1 - 19)
    out.append(rrect(lane_x0, y_s1 - lane_h / 2, lane_x1 - lane_x0, lane_h, lane_h / 2, fill=t["lane_hot"]))
    out.append(light(lane_x0 + 168, y_s1, t, tail=110, gid=f"lane-{theme}"))
    # the measured note, under the private lane
    st_a = Style(size=13.5, weight=500, fill=sub)
    st_b = Style(size=13.5, weight=700, fill=ink)
    ny = y_s1 + 34
    pen = lane_x0 + 2
    for s_, st_ in ((NOTE_STOCK, st_a), ("  \u2192  ", st_a), (NOTE_CALLO, st_b)):
        out.append(text(s_, pen, ny, st_, role="note")[0])
        pen += K.advance(s_, st_)
    out.append(text(NOTE_SUB, lane_x0 + 2, ny + 17, small_st, role="note-sub")[0])
    # the other slots: chat, the slow tide (three lanes drawn; the count is whatever the server has)
    lane_head("chat", "other slots", y_chat - lane_gap - 19)
    for i, fill in enumerate((0.74, 0.40, 0.0)):
        yy = y_chat + (i - 1) * lane_gap
        out.append(rrect(lane_x0, yy - lane_h / 2, lane_x1 - lane_x0, lane_h, lane_h / 2, fill=t["lane"]))
        if fill:
            out.append(tide(lane_x0 + 10, lane_x1 - 10, yy, t["tide"], t["tide2"], seed=3 + i, fill=fill))
    out.append(text("long prefill", lane_x1 - 12, y_chat - lane_gap + 4.5,
                    Style(size=13, weight=500, fill=sub, face="serif"), anchor="end", role="prefill")[0])

    # ---- connectors --------------------------------------------------------------------------------------------------
    a0 = client_right + 14
    tip = lane_x0 - 5
    out.append(line(f"M{_r(a0)} {_r(y_s1)} H{_r(gx0 - 5)}", acc))
    out.append(arrowhead(gx0 - 5, y_s1, acc))
    out.append(line(f"M{_r(gx1)} {_r(y_s1)} H{_r(tip)}", acc))
    out.append(arrowhead(tip, y_s1, acc))
    # chat: one unbroken line from the agent through the proxy, then routed to a chat slot
    out.append(line(f"M{_r(a0)} {_r(y_chat)} H{_r(gx1 + 8)}", t["line"]))
    fx = gx1 + 8
    for dy in (-lane_gap, 0, lane_gap):
        yy = y_chat + dy
        mid = (fx + tip) / 2
        d = (f"M{_r(fx)} {_r(y_chat)} C{_r(mid)} {_r(y_chat)} {_r(mid)} {_r(yy)} {_r(tip)} {_r(yy)}" if dy
             else f"M{_r(fx)} {_r(y_chat)} H{_r(tip)}")
        out.append(line(d, t["line"]))
        out.append(arrowhead(tip, yy, t["line"]))

    body = "".join(out)
    return K.svg_doc(W, H, body, title=TITLE, theme=theme)


def write_all(out_dir: str = HERE, png: bool = False) -> list[str]:
    paths = []
    for theme in ("light", "dark"):
        p = os.path.join(out_dir, f"how-it-works-{theme}.svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(draw(theme))
        paths.append(p)
        if png:
            q = p[:-4] + ".png"
            K.render_png(p, q, 2.0)
            paths.append(q)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="how_it_works.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--png", action="store_true", help="also render the PNGs at 2x")
    ap.add_argument("--out", default=HERE, help="output directory (default: brand/)")
    a = ap.parse_args(argv)
    for p in write_all(a.out, a.png):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
