"""Geometry for the Callosum mark and the plate's bridge.

The mark is a C drawn with *reverse contrast*: heavy where it meets its two terminals,
a hairline at the spine. The plate's bridge uses the same profile, turned on its side.
"""
from __future__ import annotations

import math


def clip_halfplane(poly, keep):
    """Sutherland-Hodgman clip of a polygon against a half-plane given as keep(x, y) -> signed
    value (>= 0 kept). The intersection is found by bisection, so any boundary works."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        fa, fb = keep(*a), keep(*b)
        if fa >= 0:
            out.append(a)
        if (fa >= 0) != (fb >= 0):
            t = fa / (fa - fb)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def c_outline(R=78.0, wmax=64.0, wmin=11.0, p=2.2, mouth=96.0, spine=180.0, cut=None,
              asym=0.0, cx=128.0, cy=128.0, n=900, rot=0.0):
    """Reverse-contrast C. Width runs from wmin at the spine to wmax at the terminals
    (power p). `mouth` is the angular opening; `cut` (x, in local units right of centre) trims
    both terminals with a vertical line so the mouth has flat, upright faces. `asym` > 0 makes
    the lower arm heavier than the upper (the splenium is the bulkier end of the callosum)."""
    a0 = spine + 180 - mouth / 2   # top terminal (screen degrees, y down)
    a1 = spine - 180 + mouth / 2   # bottom terminal
    over = 14.0 if cut is not None else 0.0  # overshoot so the vertical cut has material
    a0 += over; a1 -= over
    outer, inner = [], []
    for i in range(n + 1):
        a = a0 - i * (a0 - a1) / n
        d = (a - spine) / (a0 - spine) if a >= spine else (spine - a) / (spine - a1)
        d = min(max(d, 0.0), 1.0)
        wt = wmax * (1 - asym) if a >= spine else wmax * (1 + asym)
        w = wmin + (wt - wmin) * d ** p
        ar = math.radians(a)
        outer.append((math.cos(ar) * (R + w / 2), math.sin(ar) * (R + w / 2)))
        inner.append((math.cos(ar) * (R - w / 2), math.sin(ar) * (R - w / 2)))
    poly = outer + inner[::-1]
    if cut is not None:
        poly = clip_halfplane(poly, lambda x, y: cut - x)
    cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    return [(cx + x * cr - y * sr, cy + x * sr + y * cr) for x, y in poly]


def to_path(pts, prec=2):
    f = lambda v: f"{v:.{prec}f}".rstrip("0").rstrip(".")
    return "M" + " L".join(f"{f(x)} {f(y)}" for x, y in pts) + " Z"


def bbox(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def c_mark(R=80.0, wmax=66.0, wmin=12.0, p=2.2, mouth=100.0, cut_angle=0.0, asym=0.0, cx=128.0,
           cy=128.0, n=900):
    """The mark: a reverse-contrast C whose two terminals are trimmed by lines through the mouth.
    cut_angle = 0 gives horizontal faces (top terminal cut by a horizontal line from below, bottom
    terminal from above); positive angles tilt the faces toward the radial direction."""
    over = 30.0
    pts = c_outline(R=R, wmax=wmax, wmin=wmin, p=p, mouth=mouth - 2 * over, asym=asym, cx=0, cy=0, n=n)
    # terminal centre lines at +/- mouth/2 from the +x axis
    h = math.radians(mouth / 2)
    ca = math.radians(cut_angle)
    # top terminal: keep points "above" a line through the point at angle -mouth/2 on the centre circle
    tx, ty = R * math.cos(-h), R * math.sin(-h)
    # face direction: horizontal rotated by cut_angle toward radial
    dx, dy = math.cos(-ca), math.sin(-ca)
    top_keep = lambda x, y: -((x - tx) * dy - (y - ty) * dx) if x > 0 else 1.0
    bx, by = R * math.cos(h), R * math.sin(h)
    dx2, dy2 = math.cos(ca), math.sin(ca)
    bot_keep = lambda x, y: ((x - bx) * dy2 - (y - by) * dx2) if x > 0 else 1.0
    pts = clip_halfplane(pts, lambda x, y: (-(y - ty) + (x - tx) * math.tan(ca)) if (x > 0 and y < 0) else 1.0)
    pts = clip_halfplane(pts, lambda x, y: ((y - by) + (x - bx) * math.tan(ca)) if (x > 0 and y > 0) else 1.0)
    return [(cx + x, cy + y) for x, y in pts]


def _se(a, b, n, th):
    c, s = math.cos(th), math.sin(th)
    return (a * math.copysign(abs(c) ** (2 / n), c), b * math.copysign(abs(s) ** (2 / n), s))


def _se_angle_at_y(a, b, n, y, right=True):
    """Angle on the superellipse (right half) where its y equals the given value (y down)."""
    lo, hi = (-math.pi / 2, math.pi / 2)
    for _ in range(80):
        mid = (lo + hi) / 2
        if _se(a, b, n, mid)[1] < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def c_grotesk(a=100.0, b=110.0, n=2.4, ia=62.0, ib=58.0, ini=2.2, dx=-22.0, m=34.0, cx=128.0, cy=128.0,
              steps=600):
    """Reverse-contrast C with grotesk proportions: outer superellipse (a, b, n), counter superellipse
    (ia, ib, ini) shifted horizontally by dx (negative = toward the spine, which thins it), and a mouth
    cut by two horizontal lines at y = -m and y = +m (flat terminal faces, like a grotesk C)."""
    # outer: from top terminal (y=-m, right) counterclockwise (through top, left, bottom) to y=+m right
    t0 = _se_angle_at_y(a, b, n, -m)
    t1 = _se_angle_at_y(a, b, n, m)
    outer = []
    # angles: t0 (negative, upper right) decreasing through -pi/2 (top), -pi (left), to -2pi + t1
    end = -2 * math.pi + t1
    for i in range(steps + 1):
        th = t0 + (end - t0) * i / steps
        outer.append(_se(a, b, n, th))
    u0 = _se_angle_at_y(ia, ib, ini, -m)
    u1 = _se_angle_at_y(ia, ib, ini, m)
    inner = []
    uend = -2 * math.pi + u1
    for i in range(steps + 1):
        th = u0 + (uend - u0) * i / steps
        x, y = _se(ia, ib, ini, th)
        inner.append((x + dx, y))
    poly = outer + inner[::-1]
    return [(cx + x, cy + y) for x, y in poly]
