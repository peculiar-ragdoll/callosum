"""Shared drawing kit for callo's data figures (the release plate `frontier.py`, the companion `concurrency.py`, the
Decision Index plate `di.py`).

Palette and themes from the brand guide (README.md), HarfBuzz outline text (`tools/glyphs.py`: nothing depends on an
installed font), linear and log scales, a greedy label placer, the SVG document and the Playwright render
(`tools/render.py`); and the provenance helpers the snapshots share: callo's builds by role (`data/builds.json`) and
paths relative to the build's `--root`.

Every text group carries `data-text` (the string it draws) so the figures' tests read the information a figure
conveys from the SVG itself, without OCR.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from xml.sax.saxutils import quoteattr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools"))
os.environ.setdefault("CALLO_FONT_DIR", os.path.join(HERE, "fonts"))

from glyphs import shape  # noqa: E402

# ---- palette (README.md "Palette") ------------------------------------------------------------------------------
INK = "#14161D"
BONE = "#EEEAE1"
SIGNAL = "#FF4F3A"
NIGHT = "#090B11"
DUSK = "#2A3559"
TILE = "#0E1322"
MIST = "#8D95A6"
SLATE = "#5B6272"

SANS = "InstrumentSans.ttf"
SERIF_IT = "Newsreader-Italic.ttf"

# One dark (the plate's night) and one light theme, every colour from the brand palette (README.md "Palette"):
# `accent` is Signal, the one accent, which the figures give callo's own picked series only; `ref` draws the reference
# series (trained decision heads) in the type colour; `sub` is the secondary text colour (Slate on Bone: Mist fails AA
# there); grid and axis lines are Mist / Slate at low opacity; the glow is the plate's Dusk horizon.
THEMES: dict[str, dict[str, str]] = {
    "dark": {"ground": NIGHT, "glow": DUSK, "ink": BONE, "sub": MIST, "accent": SIGNAL, "ref": BONE,
             "grid": "rgba(141,149,166,0.13)", "axis": "rgba(141,149,166,0.42)", "glow_opacity": "0.55"},
    "light": {"ground": BONE, "glow": DUSK, "ink": INK, "sub": SLATE, "accent": SIGNAL, "ref": INK,
              "grid": "rgba(91,98,114,0.15)", "axis": "rgba(91,98,114,0.45)", "glow_opacity": "0.07"},
}


def esc(s: str) -> str:
    return quoteattr(str(s))


def num(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ---- text --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Style:
    size: float = 13.0
    weight: int = 400
    fill: str = BONE
    face: str = "sans"            # "sans" (Instrument Sans) | "serif" (Newsreader Italic)
    width: int = 100              # Instrument Sans wdth axis
    tracking: float = 0.0

    def axes(self) -> dict:
        if self.face == "serif":
            return {"wght": max(200, min(800, self.weight)), "opsz": 72 if self.size >= 24 else 18}
        return {"wght": max(400, min(700, self.weight)), "wdth": self.width}

    @property
    def font(self) -> str:
        return SERIF_IT if self.face == "serif" else SANS

    def with_(self, **kw) -> Style:
        return Style(**{**self.__dict__, **kw})


# Characters the brand faces lack, drawn as outlines here so the figures can say them (Δ: Instrument Sans has no
# Greek). Each is (advance in em, builder(x, y, size, weight) -> path d).
def _delta(x: float, y: float, size: float, weight: int) -> str:
    cap = 0.72 * size
    w = 0.64 * size
    t = size * (0.075 + 0.09 * (weight - 400) / 300)
    a, b, c = (x, y), (x + w, y), (x + w / 2, y - cap)
    # the inner triangle: the outer one scaled about its incentre to inradius r - t (an even stroke on every side)
    la, lb, lc = math.dist(b, c), math.dist(a, c), math.dist(a, b)
    per = la + lb + lc
    ix = (la * a[0] + lb * b[0] + lc * c[0]) / per
    iy = (la * a[1] + lb * b[1] + lc * c[1]) / per
    area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2
    r = 2 * area / per
    k = (r - t) / r
    inner = [(ix + (px - ix) * k, iy + (py - iy) * k) for px, py in (a, c, b)]   # reversed winding
    outer = " ".join(f"{'M' if i == 0 else 'L'}{num(px)} {num(py)}" for i, (px, py) in enumerate((a, b, c))) + " Z"
    hole = " ".join(f"{'M' if i == 0 else 'L'}{num(px)} {num(py)}" for i, (px, py) in enumerate(inner)) + " Z"
    return outer + " " + hole


SYNTH = {"Δ": (0.72, _delta)}


def _segments(s: str) -> list[str]:
    out, cur = [], ""
    for ch in s:
        if ch in SYNTH:
            if cur:
                out.append(cur)
            out.append(ch)
            cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def advance(s: str, st: Style) -> float:
    w = 0.0
    for seg in _segments(s):
        if seg in SYNTH:
            w += SYNTH[seg][0] * st.size
        else:
            w += shape(st.font, seg, st.size, st.axes(), tracking=st.tracking).advance
    return w


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def overlap(self, o: Box) -> float:
        w = min(self.x1, o.x1) - max(self.x0, o.x0)
        h = min(self.y1, o.y1) - max(self.y0, o.y0)
        return w * h if w > 0 and h > 0 else 0.0

    def inside(self, o: Box) -> bool:
        return self.x0 >= o.x0 - 1e-6 and self.x1 <= o.x1 + 1e-6 and self.y0 >= o.y0 - 1e-6 and self.y1 <= o.y1 + 1e-6

    def pad(self, p: float) -> Box:
        return Box(self.x0 - p, self.y0 - p, self.x1 + p, self.y1 + p)

    @property
    def w(self) -> float:
        return self.x1 - self.x0


def line_box(x: float, y: float, w: float, st: Style) -> Box:
    """The layout box of one line of text set at baseline y (cap-height top, descender bottom)."""
    return Box(x, y - 0.76 * st.size, x + w, y + 0.24 * st.size)


def text_paths(s: str, x: float, y: float, st: Style) -> str:
    """Outlined text at baseline (x, y), anchored at its start: one <path> per run of the same fill."""
    parts = []
    pen = x
    for seg in _segments(s):
        if seg in SYNTH:
            em, build = SYNTH[seg]
            gap = 0.04 * st.size
            parts.append(build(pen + gap, y, st.size, st.weight))
            pen += em * st.size
        else:
            run = shape(st.font, seg, st.size, st.axes(), tracking=st.tracking)
            d = run.d(pen, y)
            if d.strip():
                parts.append(d)
            pen += run.advance
    return f'<path d="{" ".join(parts)}" fill="{st.fill}" fill-rule="nonzero"/>' if parts else ""


def text(s: str, x: float, y: float, st: Style, *, anchor: str = "start", role: str | None = None,
         attrs: dict | None = None) -> tuple[str, Box]:
    """One line of outlined text: (svg, layout box). `anchor` is start | middle | end."""
    w = advance(s, st)
    x0 = x - (w if anchor == "end" else w / 2 if anchor == "middle" else 0)
    extra = "".join(f" {k}={esc(v)}" for k, v in (attrs or {}).items())
    r = f' data-role="{role}"' if role else ""
    return f'<g data-text={esc(s)}{r}{extra}>{text_paths(s, x0, y, st)}</g>', line_box(x0, y, w, st)


@dataclass
class Para:
    svg: str
    box: Box
    lines: int


def paragraph(runs: Sequence[tuple[str, Style]], x: float, y: float, width: float, *, leading: float,
              role: str | None = None, attrs: dict | None = None) -> Para:
    """Rich text wrapped to `width`: `runs` are (text, style) pieces set one after another; lines break at spaces.
    The group's `data-text` is the whole paragraph."""
    words: list[tuple[str, Style]] = []
    for s, st in runs:
        for i, wd in enumerate(s.split(" ")):
            if i > 0:
                words.append((" ", st))
            if wd:
                words.append((wd, st))
    lines: list[list[tuple[str, Style]]] = [[]]
    used = 0.0
    for wd, st in words:
        w = advance(wd, st)
        if wd == " ":
            if lines[-1]:
                lines[-1].append((wd, st))
                used += w
            continue
        if used + w > width and lines[-1]:
            while lines[-1] and lines[-1][-1][0] == " ":
                lines[-1].pop()
            lines.append([])
            used = 0.0
        lines[-1].append((wd, st))
        used += w
    out = []
    yy = y
    right = x
    for ln in lines:
        pen = x
        # merge consecutive pieces of one style into one shaped run (kerning within a run)
        merged: list[list] = []
        for wd, st in ln:
            if merged and merged[-1][1] == st:
                merged[-1][0] += wd
            else:
                merged.append([wd, st])
        for s, st in merged:
            out.append(text_paths(s, pen, yy, st))
            pen += advance(s, st)
        right = max(right, pen)
        yy += leading
    full = "".join(s for s, _ in runs)
    extra = "".join(f" {k}={esc(v)}" for k, v in (attrs or {}).items())
    r = f' data-role="{role}"' if role else ""
    size = max(st.size for _, st in runs) if runs else 12
    box = Box(x, y - 0.76 * size, right, yy - leading + 0.24 * size)
    return Para(f'<g data-text={esc(full)}{r}{extra}>{"".join(out)}</g>', box, len(lines))


# ---- scales ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LinScale:
    d0: float
    d1: float
    r0: float
    r1: float

    def __call__(self, v: float) -> float:
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


@dataclass(frozen=True)
class LogScale:
    d0: float
    d1: float
    r0: float
    r1: float

    def __call__(self, v: float) -> float:
        a, b = math.log10(self.d0), math.log10(self.d1)
        return self.r0 + (math.log10(v) - a) / (b - a) * (self.r1 - self.r0)


def log_ticks(lo: float, hi: float, steps: Sequence[int] = (1, 2, 5)) -> list[float]:
    out = []
    e = math.floor(math.log10(lo))
    while 10**e <= hi:
        for m in steps:
            v = m * 10**e
            if lo <= v <= hi:
                out.append(float(v))
        e += 1
    return out


def log_bounds(lo: float, hi: float, steps: Sequence[int] = (1, 2, 5)) -> tuple[float, float]:
    """The 1-2-5 values just outside [lo, hi]."""
    grid = sorted(m * 10.0**e for e in range(-1, 8) for m in steps)
    a = max(v for v in grid if v <= lo)
    b = min(v for v in grid if v >= hi)
    return a, b


def fmt_ms(v: float) -> str:
    if v >= 1000:
        s = v / 1000
        return f"{s:.0f} s" if abs(s - round(s)) < 1e-9 or s >= 10 else f"{s:.1f} s"
    return f"{v:.0f} ms"


def fmt_ms_value(v: float) -> str:
    """A measured latency, one decimal of a second above 1 s."""
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


# ---- label placement ---------------------------------------------------------------------------------------------

@dataclass
class Placer:
    """Greedy direct labels: each label tries candidate offsets around its anchor and takes the one that overlaps
    the least (placed labels weigh most, then marks, then lines), stays inside `bounds`, and sits nearest."""
    bounds: Box
    labels: list[Box] = field(default_factory=list)
    marks: list[Box] = field(default_factory=list)
    lines: list[Box] = field(default_factory=list)

    def leader_cost(self, b: Box, ax: float, ay: float) -> float:
        """What a leader from (ax, ay) to the box's nearest point would cross: placed labels weigh most."""
        cx, cy = min(max(ax, b.x0), b.x1), min(max(ay, b.y0), b.y1)
        dist = math.hypot(cx - ax, cy - ay)
        if dist <= 16:
            return 0.0
        n = int(dist / 5)
        c = 0.0
        for i in range(2, n):
            x, y = ax + (cx - ax) * i / n, ay + (cy - ay) * i / n
            c += sum(300.0 for o in self.labels if o.x0 <= x <= o.x1 and o.y0 <= y <= o.y1)
            c += sum(40.0 for o in self.marks if o.x0 <= x <= o.x1 and o.y0 <= y <= o.y1)
        return c

    def cost(self, b: Box) -> float:
        if not b.inside(self.bounds):
            return math.inf
        return (1000 * sum(b.overlap(o) for o in self.labels) + 60 * sum(b.overlap(o) for o in self.marks)
                + 9 * sum(b.overlap(o) for o in self.lines))

    def place(self, w: float, h: float, ax: float, ay: float, *, gaps: Iterable[float] = (8, 14, 24, 38, 56, 80),
              prefer: Sequence[str] = ("e", "ne", "se", "n", "s", "w", "nw", "sw"),
              extra: Sequence[tuple[float, float, float]] = (), scan: float = 0.0,
              quadrant: str | None = None) -> Box:
        """A w x h box near (ax, ay). Directions are compass points; `prefer` orders ties (and costs a little).
        `extra` adds candidate top-left corners (x, y, penalty) a caller knows are good (below a whisker, say)."""
        best, best_cost = None, math.inf
        if scan:
            # a grid of positions within `scan` px (a crowded neighbourhood leaves no clear compass point); distance
            # costs, so the nearest clear spot wins; `quadrant` ("nw", "ne", ...) keeps the box on that side
            step = 10.0
            n = int(scan / step)
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    bx, by = ax + i * step - w / 2, ay + j * step - h / 2
                    if quadrant and (("w" in quadrant and bx + w > ax) or ("e" in quadrant and bx < ax)
                                     or ("n" in quadrant and by + h > ay) or ("s" in quadrant and by < ay)):
                        continue
                    b = Box(bx, by, bx + w, by + h)
                    if b.overlap(Box(ax - 6, ay - 6, ax + 6, ay + 6)):
                        continue
                    d = math.hypot(max(b.x0 - ax, 0, ax - b.x1), max(b.y0 - ay, 0, ay - b.y1))
                    c = self.cost(b) + 2.5 * d + 12
                    if c < best_cost:
                        c += self.leader_cost(b, ax, ay)
                    if c < best_cost:
                        best, best_cost = b, c
        for ex, ey, pen in extra:
            b = Box(ex, ey, ex + w, ey + h)
            c = self.cost(b) + pen
            if c < best_cost:
                best, best_cost = b, c
        for g in gaps:
            for di, d in enumerate(prefer):
                dx = {"e": 1, "w": -1, "n": 0, "s": 0, "ne": 1, "nw": -1, "se": 1, "sw": -1}[d]
                dy = {"e": 0, "w": 0, "n": -1, "s": 1, "ne": -1, "nw": -1, "se": 1, "sw": 1}[d]
                cx = ax + dx * g + (0 if dx >= 0 else -w) - (w / 2 if dx == 0 else 0)
                cy = ay + dy * g * 0.8 - (h if dy < 0 else 0 if dy > 0 else h / 2)
                b = Box(cx, cy, cx + w, cy + h)
                c = self.cost(b) + 2.5 * g + 0.8 * di       # far labels need long leaders: distance costs
                if c < best_cost:
                    c += self.leader_cost(b, ax, ay)
                if c < best_cost:
                    best, best_cost = b, c
        if best is None:                          # nowhere inside: clamp into the bounds at the anchor
            x = min(max(ax, self.bounds.x0), self.bounds.x1 - w)
            y = min(max(ay - h / 2, self.bounds.y0), self.bounds.y1 - h)
            best = Box(x, y, x + w, y + h)
        self.labels.append(best)
        return best


# ---- markers -----------------------------------------------------------------------------------------------------

def marker(m: str, x: float, y: float, fill: str, stroke: str, sw: float, *, big: bool = False,
           scale: float = 1.0) -> tuple[str, float]:
    """One series marker (frontier.py's shapes: circle, diamond, triangle, square; `-hollow` only names the fill the
    caller passes) and its radius. `scale` shrinks it for small multiples and table keys."""
    if m == "circle":
        r = (4.8 if big else 4.2) * scale
        return f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(r)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>', r
    if m.startswith("diamond"):
        r = 6.4 * scale
        return (f'<path d="M{num(x)} {num(y - r)} L{num(x + r)} {num(y)} L{num(x)} {num(y + r)} L{num(x - r)} {num(y)} Z" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>', r)
    if m.startswith("triangle"):
        r = 6.0 * scale
        return (f'<path d="M{num(x)} {num(y - r)} L{num(x + r * 0.95)} {num(y + r * 0.62)} L{num(x - r * 0.95)} '
                f'{num(y + r * 0.62)} Z" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>', r)
    r = 4.8 * scale
    return (f'<rect x="{num(x - r)}" y="{num(y - r)}" width="{num(2 * r)}" height="{num(2 * r)}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>', r)


# ---- document + render -------------------------------------------------------------------------------------------

def svg_doc(w: float, h: float, body: str, *, title: str, theme: str, extra_attrs: dict | None = None) -> str:
    extra = "".join(f" {k}={esc(v)}" for k, v in (extra_attrs or {}).items())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {num(w)} {num(h)}" width="{num(w)}" '
            f'height="{num(h)}" role="img" aria-label={esc(title)} data-theme="{theme}"{extra}>'
            f"<title>{_xml(title)}</title>{body}</svg>")


def _xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ground(w: float, h: float, t: dict, *, glow_at: tuple[float, float, float, float] | None = None) -> str:
    """The plate's ground: a flat night (or bone) with a soft horizon glow under the chart, as in the brand plate."""
    out = f'<rect width="{num(w)}" height="{num(h)}" fill="{t["ground"]}"/>'
    if glow_at:
        cx, cy, rx, ry = glow_at
        out += (f'<defs><radialGradient id="glow" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="{t["glow"]}" '
                f'stop-opacity="{t["glow_opacity"]}"/><stop offset="1" stop-color="{t["glow"]}" stop-opacity="0"/>'
                f'</radialGradient></defs><ellipse cx="{num(cx)}" cy="{num(cy)}" rx="{num(rx)}" ry="{num(ry)}" '
                f'fill="url(#glow)"/>')
    return out


def render_png(svg_path: str, png_path: str, scale: float = 2.0) -> None:
    from render import render
    render(svg_path, png_path, scale=scale)


# ---- provenance: callo's builds by role, paths without the checkout ----------------------------------------------

BUILDS_JSON = os.path.join(HERE, "data", "builds.json")


def builds() -> list[dict]:
    """The callo builds the figures were measured on, in the order they were made: [{"commit", "version", "name",
    "what"}] from data/builds.json. The figures name a build by its role ("the readout-fix build"), never by its
    commit, which is of callo's development history and not published."""
    with open(BUILDS_JSON, encoding="utf-8") as f:
        return list(json.load(f)["builds"])


def build_of(commit: str | None) -> dict | None:
    """The build a short or full callo commit is, or None when data/builds.json does not name it."""
    c = str(commit or "")
    if len(c) < 7:
        return None
    return next((b for b in builds() if c.startswith(b["commit"]) or b["commit"].startswith(c)), None)


def builds_named(commits: Iterable[str | None]) -> list[dict]:
    """The builds among `commits`, as a snapshot records them ({"commit", "name"}), in the order they were made."""
    have = {b["commit"] for b in (build_of(c) for c in commits) if b}
    return [{"commit": b["commit"], "name": b["name"]} for b in builds() if b["commit"] in have]


def public_path(p: str | None, root: str | os.PathLike) -> str | None:
    """A path as a snapshot records it: relative to the build's `--root` when it lies under it, else its file name
    (a model file, a chat template); a relative path is kept. Where the checkout sits is never recorded."""
    if p is None:
        return None
    if not os.path.isabs(str(p)):
        return str(p)
    a, r = os.path.abspath(str(p)), os.path.abspath(str(root))
    return os.path.relpath(a, r).replace(os.sep, "/") if a.startswith(r.rstrip(os.sep) + os.sep) else os.path.basename(a)
