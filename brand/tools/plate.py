"""The Callosum plate: a bridge over slow water.

The bridge is the mark turned on its side and stood in the water up to its waist: two heavy
piers, a thin crown, seen as a silhouette against the last light on the horizon. The water
below is a field of ripples that reflect that light; the bridge's reflection is where the
ripples go dark. The ripples drift on a slow tide (one swell per loop). Quick lights cross the
crown four times per loop, alternating direction, and glint in the water as they pass.

The static plate and the animated plate come from the same scene description. Every animated
element's resting attributes are exactly its t = 0 keyframe, so the animated SVG at rest (and
under prefers-reduced-motion, where animation is switched off) is pixel-identical to the static
one, and t = 0 equals t = LOOP by construction.
"""
from __future__ import annotations

import math
import random

from geom import bbox, c_grotesk, to_path
from glyphs import translate

W, H = 1280, 640
HY = 446            # horizon
LOOP = 16.0         # seconds: the tide's period
CROSSINGS = 4       # quick lights per loop
CROSS_DUR = 1.4     # seconds per crossing
ARCH_W = 470        # outer width of the bridge
THETA = 15.0        # half-angle of the visible crossing on the crown track
HALO_R = 34         # the glow around each light; the tagline clears it where the light peaks, on the crown
WORD_Y = 124        # the wordmark's baseline

BONE = "#EEEAE1"
SIGNAL = "#FF4F3A"


def _f(v, p=2):
    s = f"{v:.{p}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def arch_geometry(mark_params):
    pts = c_grotesk(cx=0, cy=0, **mark_params)
    pts = [(-y, x) for x, y in pts]            # +90 deg: spine -> crown, mouth under water
    x0, y0, x1, y1 = bbox(pts)
    s = ARCH_W / (x1 - x0)
    return [(W / 2 + x * s, HY + y * s) for x, y in pts], s


def crown_track(pts):
    """Circle through the crown's centreline at x = centre and centre +- 120: the lights' track."""
    cols = {}
    for x, y in pts:
        if y < HY:
            cols.setdefault(round(x), []).append(y)

    def mid_at(x):
        ys = sorted(sum((cols.get(round(x) + d, []) for d in range(-2, 3)), []))
        return (ys[0] + ys[1]) / 2

    xa, xb, xc = W / 2 - 120, W / 2, W / 2 + 120
    ya, yb, yc = mid_at(xa), mid_at(xb), mid_at(xc)
    d = 2 * (xa * (yb - yc) + xb * (yc - ya) + xc * (ya - yb))
    ux = ((xa**2 + ya**2) * (yb - yc) + (xb**2 + yb**2) * (yc - ya) + (xc**2 + yc**2) * (ya - yb)) / d
    uy = ((xa**2 + ya**2) * (xc - xb) + (xb**2 + yb**2) * (xa - xc) + (xc**2 + yc**2) * (xb - xa)) / d
    return ux, uy, math.hypot(xb - ux, yb - uy)


def _ease(u):
    """Gathers speed onto the crown, fastest at the top, settles off the other side."""
    return 0.5 - 0.5 * math.cos(math.pi * u)


def light_keyframes(k, n=24):
    """[(percent, angle_deg, opacity)] for light k. Its crossing is centred on t_k = k*LOOP/N;
    between crossings it waits, hidden, at the end it will enter from. Light 0 is mid-crossing
    (on the crown, fully lit) at t = 0, which is the resting frame."""
    tk = k * LOOP / CROSSINGS
    sign = 1 if k % 2 == 0 else -1
    entry, exit_ = -sign * THETA, sign * THETA
    ts, te = tk - CROSS_DUR / 2, tk + CROSS_DUR / 2
    pts = []
    for i in range(n + 1):
        u = i / n
        t = (ts + u * CROSS_DUR) % LOOP
        pts.append((t, entry + (exit_ - entry) * _ease(u), max(0.0, min(1.0, u / 0.22, (1 - u) / 0.22))))
    eps = 0.002
    pts.append(((te % LOOP) + eps, entry, 0.0))
    pts.append(((ts % LOOP) - eps, entry, 0.0))
    pts = sorted((t if t >= 0 else t + LOOP, a, o) for t, a, o in pts)
    if k == 0:
        first = next(p for p in pts if abs(p[0]) < 1e-9)
        pts.append((LOOP, first[1], first[2]))
    else:
        pts = [(0.0, entry, 0.0)] + pts + [(LOOP, entry, 0.0)]
    out = []
    for t, a, o in pts:
        pct = round(t / LOOP * 100, 4)
        if not (out and abs(out[-1][0] - pct) < 1e-6):
            out.append((pct, a, o))
    return out


def _swell(phase, amp, steps=24):
    return [(i / steps * 100, amp * math.sin(2 * math.pi * i / steps + phase)) for i in range(steps + 1)]


def scene(mark_params, word, animated, caption=None):
    """word = (d_callo, d_sum, width, cap) for the wordmark at plate size."""
    pts, s = arch_geometry(mark_params)
    arch_d = to_path(pts, 2)
    ux, uy, r = crown_track(pts)
    css, defs, body = [], [], []

    defs.append(
        f'<linearGradient id="sky" x1="0" y1="0" x2="0" y2="{HY}" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0" stop-color="#05070C"/><stop offset=".45" stop-color="#0A0F1C"/>'
        f'<stop offset=".78" stop-color="#17203A"/><stop offset=".95" stop-color="#2A3559"/><stop offset="1" stop-color="#3A3F62"/></linearGradient>'
        f'<radialGradient id="glow" cx="{W/2}" cy="{HY}" r="1" gradientUnits="userSpaceOnUse" '
        f'gradientTransform="translate({W/2} {HY}) scale(640 250) translate({-W/2} {-HY})">'
        f'<stop offset="0" stop-color="#7C8DBA" stop-opacity=".62"/><stop offset=".45" stop-color="#4A5A88" stop-opacity=".28"/>'
        f'<stop offset="1" stop-color="#2C3A5E" stop-opacity="0"/></radialGradient>'
        f'<linearGradient id="sea" x1="0" y1="{HY}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0" stop-color="#0B1020"/><stop offset="1" stop-color="#04050A"/></linearGradient>'
        f'<radialGradient id="shine" cx="{W/2}" cy="{HY}" r="1" gradientUnits="userSpaceOnUse" '
        f'gradientTransform="translate({W/2} {HY}) scale(600 210) translate({-W/2} {-HY})">'
        f'<stop offset="0" stop-color="#D3DCF4"/><stop offset=".4" stop-color="#8393BE" stop-opacity=".8"/>'
        f'<stop offset="1" stop-color="#3A4870" stop-opacity=".12"/></radialGradient>'
        f'<clipPath id="above"><rect width="{W}" height="{HY}"/></clipPath>'
        f'<clipPath id="archclip"><path d="{arch_d}"/></clipPath>'
        f'<linearGradient id="tailr" x1="0" x2="1"><stop offset="0" stop-color="{SIGNAL}" stop-opacity="0"/>'
        f'<stop offset=".7" stop-color="{SIGNAL}"/><stop offset=".87" stop-color="#FFC6B8"/><stop offset=".95" stop-color="{SIGNAL}"/><stop offset="1" stop-color="{SIGNAL}" stop-opacity=".2"/></linearGradient>'
        f'<linearGradient id="taill" x1="1" x2="0"><stop offset="0" stop-color="{SIGNAL}" stop-opacity="0"/>'
        f'<stop offset=".7" stop-color="{SIGNAL}"/><stop offset=".87" stop-color="#FFC6B8"/><stop offset=".95" stop-color="{SIGNAL}"/><stop offset="1" stop-color="{SIGNAL}" stop-opacity=".2"/></linearGradient>'
        f'<linearGradient id="glit" x1="0" y1="{HY}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0" stop-color="{SIGNAL}"/><stop offset=".55" stop-color="{SIGNAL}" stop-opacity=".45"/>'
        f'<stop offset="1" stop-color="{SIGNAL}" stop-opacity="0"/></linearGradient>'
        f'<radialGradient id="halo"><stop offset="0" stop-color="{SIGNAL}" stop-opacity=".55"/>'
        f'<stop offset="1" stop-color="{SIGNAL}" stop-opacity="0"/></radialGradient>'
    )
    body.append(f'<rect width="{W}" height="{HY}" fill="url(#sky)"/>')
    gcls = ' class="br"' if animated else ""
    body.append(f'<rect{gcls} width="{W}" height="{HY}" fill="url(#glow)"/>')
    if animated:
        # the horizon breathes once per loop, in step with the tide
        css.append("@keyframes br{0%,100%{opacity:1}50%{opacity:.82}}.br{animation:br " + _f(LOOP) + "s ease-in-out infinite}")
    body.append(f'<rect y="{HY}" width="{W}" height="{H - HY}" fill="url(#sea)"/>')

    rnd = random.Random(7)
    rows = []
    y = HY + 2.5
    i = 0
    while y < H + 6:
        depth = (y - HY) / (H - HY)
        th = (1.0 + 2.0 * depth) * rnd.uniform(0.75, 1.15)
        gap = (2.2 + 11.0 * depth ** 1.35) * rnd.uniform(0.7, 1.3)
        dash, total = [], 0.0
        while total < W + 600:
            a = rnd.uniform(10, 110) * (1 + 1.4 * depth) * (0.35 if rnd.random() < 0.22 else 1)
            b = rnd.uniform(5, 26) * (1 + 3.5 * depth) * (2.5 if rnd.random() < 0.12 else 1)
            dash += [a, b]
            total += a + b
        amp = 3.0 + 22.0 * depth
        phase = -3.2 * math.pi * depth + rnd.uniform(-0.3, 0.3)
        op = 0.9 * (1 - depth) ** 1.15 + 0.06
        rows.append((y + th / 2, th, dash, amp, phase, op, i))
        y += th + gap
        i += 1

    def ripple_lines(stroke, cls_prefix):
        out = []
        for yy, th, dash, amp, phase, op, j in rows:
            x0 = amp * math.sin(phase)
            cls = f' class="rp {cls_prefix}{j}"' if animated else ""
            out.append(
                f'<line{cls} x1="-300" x2="{W + 300}" y1="{_f(yy)}" y2="{_f(yy)}" stroke="{stroke}" '
                f'stroke-opacity="{_f(op, 3)}" stroke-width="{_f(th)}" '
                f'stroke-dasharray="{" ".join(_f(v, 1) for v in dash)}" style="transform:translate({_f(x0, 3)}px,0)"/>')
        return "".join(out)

    if animated:
        for yy, th, dash, amp, phase, op, j in rows:
            kf = " ".join(f"{_f(p, 3)}%{{transform:translate({_f(v, 3)}px,0)}}" for p, v in _swell(phase, amp))
            css.append(f"@keyframes w{j}{{{kf}}}.w{j},.m{j}{{animation:w{j} {LOOP}s linear infinite}}")

    defs.append(
        f'<mask id="noarch" maskUnits="userSpaceOnUse" x="0" y="{HY}" width="{W}" height="{H - HY}">'
        f'<rect y="{HY}" width="{W}" height="{H - HY}" fill="#fff"/>'
        f'<g transform="matrix(1 0 0 -1 0 {2 * HY})"><path d="{arch_d}" fill="#000" clip-path="url(#above)"/></g></mask>')
    body.append(f'<g mask="url(#noarch)">{ripple_lines("url(#shine)", "w")}</g>')

    def lights(prefix, halo):
        out = []
        for k in range(CROSSINGS):
            kf = light_keyframes(k)
            _, a0, o0 = kf[0]
            cls = f' class="lt {prefix}{k}"' if animated else ""
            cx, cy = ux, uy - r
            grad = "tailr" if k % 2 == 0 else "taill"
            inner = (f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{HALO_R}" fill="url(#halo)"/>' if halo else
                     f'<rect x="{_f(cx - 46) if k % 2 == 0 else _f(cx - 8)}" y="{_f(cy - 30)}" width="54" height="60" fill="url(#{grad})"/>')
            out.append(f'<g{cls} style="transform:rotate({_f(a0, 3)}deg);transform-origin:{_f(ux)}px {_f(uy)}px;'
                       f'transform-box:view-box;opacity:{_f(o0, 3)}">{inner}</g>')
            if animated:
                steps = " ".join(f"{_f(p, 4)}%{{transform:rotate({_f(a, 3)}deg);opacity:{_f(o, 3)}}}" for p, a, o in kf)
                css.append(f"@keyframes {prefix}{k}{{{steps}}}.{prefix}{k}{{animation:{prefix}{k} {LOOP}s linear infinite;"
                           f"transform-origin:{_f(ux)}px {_f(uy)}px;transform-box:view-box}}")
        return "".join(out)

    body.append(f'<g clip-path="url(#above)">{lights("h", True)}<path d="{arch_d}" fill="#04060A" stroke="#9AA8CC" stroke-opacity=".16" stroke-width="1.2"/>'
                f'<g clip-path="url(#archclip)">{lights("l", False)}</g></g>')
    defs.append(f'<mask id="ripmask" maskUnits="userSpaceOnUse" x="0" y="{HY}" width="{W}" height="{H - HY}">'
                f'{ripple_lines("#fff", "m")}</mask>')
    def glitter():
        out = []
        for k in range(CROSSINGS):
            kf = light_keyframes(k)
            _, a0, o0 = kf[0]
            cls = f' class="lt g{k}"' if animated else ""
            x0 = r * math.sin(math.radians(a0))
            out.append(f'<g{cls} style="transform:translate({_f(x0, 3)}px,0);opacity:{_f(o0, 3)}">'
                       f'<rect x="{_f(ux - 7)}" y="{HY}" width="14" height="{H - HY}" fill="url(#glit)"/>'
                       f'<rect x="{_f(ux - 22)}" y="{HY}" width="44" height="{(H - HY) * .45}" fill="url(#glit)" opacity=".35"/></g>')
            if animated:
                steps = " ".join(f"{_f(p, 4)}%{{transform:translate({_f(r * math.sin(math.radians(a)), 3)}px,0);opacity:{_f(o, 3)}}}"
                                 for p, a, o in kf)
                css.append(f"@keyframes g{k}{{{steps}}}.g{k}{{animation:g{k} {LOOP}s linear infinite}}")
        return "".join(out)

    body.append(f'<g mask="url(#ripmask)">{glitter()}</g>')
    body.append(f'<rect y="{HY - .5}" width="{W}" height="1" fill="#9AA8CC" fill-opacity=".22"/>')

    d_callo, d_sum, width, cap = word
    wx, wy = W / 2 - width / 2, WORD_Y
    body.append(f'<g fill="{BONE}"><path d="{translate(d_callo, wx, wy)}"/><path d="{translate(d_sum, wx, wy)}"/></g>')
    if caption:
        body.append(caption)

    style = ""
    if animated:
        style = ("<style>" + "".join(css) +
                 "@media (prefers-reduced-motion: reduce){.rp,.lt,.br{animation:none!important}}</style>")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="Callosum: a bridge over slow water"><title>Callosum</title>{style}'
            f'<defs>{"".join(defs)}</defs>{"".join(body)}</svg>')
