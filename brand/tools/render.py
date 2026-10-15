"""Render SVG/HTML to PNG with headless Chromium (Playwright).

render(src, out, scale=2, t=None, bg=None, reduced=False)
  t: seek every CSS animation to t seconds (Web Animations API) before capture.
"""
from __future__ import annotations

import pathlib
from contextlib import contextmanager

from playwright.sync_api import sync_playwright

SEEK_JS = "t => { for (const a of document.getAnimations()) { a.pause(); a.currentTime = t * 1000; } }"


@contextmanager
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        try:
            yield b
        finally:
            b.close()


def svg_page(svg_text, bg=None):
    return ("<!doctype html><html><head><meta charset=utf-8><style>html,body{margin:0;padding:0;"
            f"background:{bg or 'transparent'}}}svg{{display:block}}</style></head><body>{svg_text}</body></html>")


def render(src, out, scale=2, t=None, bg=None, reduced=False, b=None, size=None):
    src = pathlib.Path(src).resolve()

    def go(br):
        ctx = br.new_context(device_scale_factor=scale, viewport={"width": 800, "height": 600},
                             reduced_motion="reduce" if reduced else "no-preference")
        page = ctx.new_page()
        if src.suffix == ".svg":
            page.set_content(svg_page(src.read_text(), bg), wait_until="load")
            w, h = size or page.evaluate("(()=>{const r=document.querySelector('svg').getBoundingClientRect();return [r.width,r.height]})()")
            page.set_viewport_size({"width": int(round(w)), "height": int(round(h))})
        else:
            page.goto(src.as_uri(), wait_until="load")
            if size:
                page.set_viewport_size({"width": size[0], "height": size[1]})
        if t is not None:
            page.wait_for_timeout(30)
            page.evaluate(SEEK_JS, t)
        page.wait_for_timeout(80)
        page.screenshot(path=str(out), omit_background=bg is None, full_page=True)
        ctx.close()

    if b is None:
        with browser() as br:
            go(br)
    else:
        go(b)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=2); ap.add_argument("--t", type=float)
    ap.add_argument("--bg"); ap.add_argument("--reduced", action="store_true")
    a = ap.parse_args()
    render(a.src, a.out, a.scale, a.t, a.bg, a.reduced)
