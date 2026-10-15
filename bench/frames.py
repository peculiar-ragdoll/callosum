"""Synthetic 11 x 11 maze frames for the v0.2 bench, stdlib only (`callo.vision.encode_png`).

The spike's own 36 readout frames are committed as files under `bench/frames/spike/` and used as they are. These
generated frames feed the paced rows instead, which need FRESH bytes on every tick: llama.cpp caches an image chunk by
the sha256 of its bytes, so a repeated frame would hit the cache and measure nothing. `salt` removes a different set
of dots for every value, so every salt gives different bytes; the same arguments always give the same bytes.
"""
from __future__ import annotations

import functools
from pathlib import Path

from callo.vision import encode_png

GRID = 11
SPIKE_DIR = Path(__file__).resolve().parent / "frames" / "spike"
DIRS = ("left", "right", "up", "down")
_BLOCKS = {(2, 2), (2, 3), (8, 2), (8, 3), (2, 7), (2, 8), (8, 7), (8, 8), (4, 2), (6, 2), (4, 8), (6, 8)}
WALLS = frozenset({(i, 0) for i in range(GRID)} | {(i, GRID - 1) for i in range(GRID)} | {(0, i) for i in range(GRID)}
                  | {(GRID - 1, i) for i in range(GRID)} | _BLOCKS)
BG, WALL, DOT, PAC, GHOST, EYE = (0, 0, 0), (33, 33, 222), (255, 184, 151), (255, 255, 0), (255, 0, 0), (255, 255, 255)
_OFFSET = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
PAC_TILE = (5, 5)


def ghost_tile(ghost_dir: str, ghost_dist: int) -> tuple[int, int]:
    dx, dy = _OFFSET[ghost_dir]
    return PAC_TILE[0] + dx * ghost_dist, PAC_TILE[1] + dy * ghost_dist


@functools.lru_cache(maxsize=64)
def _tile(tile: int, kind: str) -> tuple[bytes, ...]:
    """The `tile` pixel rows of one tile: wall, pac, ghost, dot or empty. Integer geometry, no anti-aliasing."""
    rows = []
    r = tile * 0.42
    for py in range(tile):
        row = bytearray()
        for px in range(tile):
            dx, dy = px + 0.5 - tile / 2, py + 0.5 - tile / 2
            c = BG
            if kind == "wall":
                c = WALL
            elif kind == "pac" and dx * dx + dy * dy <= r * r:
                c = PAC
            elif kind == "ghost":
                if any((dx - ex) ** 2 + (dy + 0.15 * r) ** 2 <= (0.22 * r) ** 2 for ex in (-0.35 * r, 0.35 * r)):
                    c = EYE
                elif (dy < 0 and dx * dx + dy * dy <= r * r) or (0 <= dy <= r and abs(dx) <= r):
                    c = GHOST
            elif kind == "dot" and dx * dx + dy * dy <= (tile * 0.1) ** 2:
                c = DOT
            row.extend(c)
        rows.append(bytes(row))
    return tuple(rows)


def maze_frame(size: int = 224, ghost_dir: str = "left", ghost_dist: int = 3, salt: int = 0) -> bytes:
    """A PNG: Pac-Man (yellow disc) in the centre tile, a red ghost `ghost_dist` tiles away in `ghost_dir`, walls,
    and dots on the other open tiles except those bit i of `salt` removes (i = the dot's index mod 60)."""
    ghost = ghost_tile(ghost_dir, ghost_dist)
    open_tiles = [(x, y) for y in range(GRID) for x in range(GRID) if (x, y) not in WALLS and (x, y) not in
                  (PAC_TILE, ghost)]
    dots = {t for i, t in enumerate(open_tiles) if not (salt >> (i % 60)) & 1}
    tile = size // GRID
    pad = (size - tile * GRID) // 2
    right = size - pad - tile * GRID
    bg = bytes(BG)
    rgb = bytearray(bg * size * pad)
    for ty in range(GRID):
        kinds = ["wall" if (tx, ty) in WALLS else "pac" if (tx, ty) == PAC_TILE else "ghost" if (tx, ty) == ghost
                 else "dot" if (tx, ty) in dots else "empty" for tx in range(GRID)]
        tiles = [_tile(tile, k) for k in kinds]
        for py in range(tile):
            rgb += bg * pad + b"".join(t[py] for t in tiles) + bg * right
    rgb += bg * size * (size - pad - tile * GRID)
    return encode_png(size, size, bytes(rgb))


def fresh_frames(n: int, *, size: int = 224, salt0: int = 0) -> list[bytes]:
    """`n` frames with distinct bytes: the ghost cycles through the four directions and 2-4 tiles."""
    return [maze_frame(size, DIRS[i % 4], 2 + i % 3, salt=salt0 + i + 1) for i in range(n)]


def spike_frames() -> list[tuple[str, int, int, Path]]:
    """(gold direction, ghost distance, size, path) for the spike's 36 maze frames, `maze_<dir>_d<dist>_<size>.png`."""
    out = []
    for p in sorted(SPIKE_DIR.glob("maze_*_d*_*.png")):
        _, d, dist, size = p.stem.split("_")
        out.append((d, int(dist[1:]), int(size), p))
    return out
