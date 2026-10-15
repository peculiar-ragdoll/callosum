"""Render every raster from the SVG masters, verify the loop, and encode the loop videos.

    python brand/tools/export.py pngs     # lockups 1x/2x, marks 512, favicons, plate@2x, how-it-works 2x
    python brand/tools/export.py check    # rest frame == static plate; t=0 == t=LOOP; reduced motion
    python brand/tools/export.py video    # frames -> plate-loop.mp4 + plate-loop.webp (frames deleted)

Large temporaries go to $CALLO_TMP (default: callosum-brand in the system temp directory).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
BRAND = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, BRAND)
TMP = os.environ.get("CALLO_TMP") or os.path.join(tempfile.gettempdir(), "callosum-brand")

from render import SEEK_JS, browser, svg_page  # noqa: E402

LOOP = 16.0


def p(*a):
    return os.path.join(BRAND, *a)


def shot_svg(b, svg_path, out, scale=1.0, bg=None, t=None, reduced=False, size=None):
    text = open(svg_path).read()
    ctx = b.new_context(device_scale_factor=scale, viewport={"width": 800, "height": 600},
                        reduced_motion="reduce" if reduced else "no-preference")
    page = ctx.new_page()
    page.set_content(svg_page(text, bg), wait_until="load")
    w, h = size or page.evaluate("(()=>{const r=document.querySelector('svg').getBoundingClientRect();return [r.width,r.height]})()")
    page.set_viewport_size({"width": int(round(w)), "height": int(round(h))})
    if t is not None:
        page.evaluate(SEEK_JS, t)
    page.wait_for_timeout(60)
    page.screenshot(path=out, omit_background=bg is None, full_page=False,
                    clip={"x": 0, "y": 0, "width": int(round(w)), "height": int(round(h))})
    ctx.close()


def supersample(b, svg_path, out, px, bg=None, factor=8):
    os.makedirs(TMP, exist_ok=True)
    tmp = os.path.join(TMP, "ss.png")
    shot_svg(b, svg_path, tmp, scale=px * factor / 256, bg=bg)
    im = Image.open(tmp).convert("RGBA")
    im.resize((px, px), Image.LANCZOS).save(out)


def pngs():
    with browser() as b:
        for v in ("light", "dark"):
            shot_svg(b, p(f"lockup-stacked-{v}.svg"), p(f"lockup-stacked-{v}@2x.png"), 2)
            shot_svg(b, p(f"lockup-{v}.svg"), p(f"lockup-{v}.png"), 1)
            shot_svg(b, p(f"lockup-{v}.svg"), p(f"lockup-{v}@2x.png"), 2)
            shot_svg(b, p(f"mark-{v}.svg"), p(f"mark-{v}-512.png"), 2)
            shot_svg(b, p(f"how-it-works-{v}.svg"), p(f"how-it-works-{v}.png"), 2)
        supersample(b, p("favicon.svg"), p("favicon-32.png"), 32)
        supersample(b, p("favicon-16.svg"), p("favicon-16.png"), 16)
        shot_svg(b, p("plate.svg"), p("plate@2x.png"), 2)
    print("pngs ok")


def diff(a, b):
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    d = ImageChops.difference(ia, ib)
    return d.getbbox(), max(max(px) for px in d.getdata()) if d.getbbox() else 0


def check():
    os.makedirs(TMP, exist_ok=True)
    o = lambda n: os.path.join(TMP, n)
    with browser() as b:
        shot_svg(b, p("plate.svg"), o("static.png"))
        shot_svg(b, p("plate-animated.svg"), o("t0.png"), t=0.0)
        shot_svg(b, p("plate-animated.svg"), o("tL.png"), t=LOOP)
        shot_svg(b, p("plate-animated.svg"), o("reduced.png"), reduced=True)
        shot_svg(b, p("plate-animated.svg"), o("t2loop.png"), t=2 * LOOP + 0.0)
    for name, a, c in [("static vs t=0", "static.png", "t0.png"), ("t=0 vs t=LOOP", "t0.png", "tL.png"),
                       ("static vs reduced-motion", "static.png", "reduced.png"),
                       ("t=0 vs t=2*LOOP", "t0.png", "t2loop.png")]:
        bb, mx = diff(o(a), o(c))
        print(f"{name:28s} differing bbox={bb} max channel diff={mx}")


def video(fps=30, webp_fps=15, webp_w=960):
    frames = os.path.join(TMP, "frames")
    shutil.rmtree(frames, ignore_errors=True)
    os.makedirs(frames)
    n = int(LOOP * fps)
    text = open(p("plate-animated.svg")).read()
    with browser() as b:
        ctx = b.new_context(device_scale_factor=1, viewport={"width": 1280, "height": 640})
        page = ctx.new_page()
        page.set_content(svg_page(text, "#000"), wait_until="load")
        for i in range(n):
            page.evaluate(SEEK_JS, i / fps)
            page.screenshot(path=os.path.join(frames, f"f{i:04d}.png"), clip={"x": 0, "y": 0, "width": 1280, "height": 640})
        ctx.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frames, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", "-tune", "film",
                    "-movflags", "+faststart", p("plate-loop.mp4")], check=True)
    # animated WebP: every (fps/webp_fps)th frame, downscaled
    step = fps // webp_fps
    small = os.path.join(TMP, "frames_small")
    shutil.rmtree(small, ignore_errors=True)
    os.makedirs(small)
    names = []
    for i in range(0, n, step):
        im = Image.open(os.path.join(frames, f"f{i:04d}.png")).convert("RGB")
        im = im.resize((webp_w, webp_w // 2), Image.LANCZOS)
        nm = os.path.join(small, f"s{i:04d}.png")
        im.save(nm)
        names.append(nm)
    dur = str(int(round(1000 / webp_fps)))
    subprocess.run(["img2webp", "-loop", "0", "-lossy", "-q", "72", "-m", "6", "-d", dur, *names,
                    "-o", p("plate-loop.webp")], check=True, capture_output=True)
    shutil.rmtree(frames, ignore_errors=True)
    shutil.rmtree(small, ignore_errors=True)
    for f in ("plate-loop.mp4", "plate-loop.webp"):
        print(f, os.path.getsize(p(f)) // 1024, "KB")


if __name__ == "__main__":
    {"pngs": pngs, "check": check, "video": video}[sys.argv[1]]()
