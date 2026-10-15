"""callo v0.2 acceptance bench. Every System One request goes through callo, which runs in launch mode on
the master build, in this process, one llama-server at a time.

    uv run python -m bench.acceptance_v02 --out DIR \\
        --g0-records <G0 first-decision JSONL> [--rows a,b,c]

The G0 records come from the game-client demo's gate harness, which is not part of this repository.

It refuses to start while any model server is running (llama.cpp, oMLX or MLX: `bench.lib.gpu_server_pids`): the
GPU is serialised with the demo's gates, and the paced rows would measure the other server's load. `bench/report_v02.py` renders `docs/bench/v0.2.md` from the
`summary.json` written under --out. A row that measured something other than what it claims carries
`invalid: <reason>` and is never reported as PASS.

SIGTERM or SIGINT to the pid in `run.pid` stops the bench cleanly: every `finally` stops its callo and llama-server
(callo's own handlers only work on the main thread, and here callo runs in a worker thread), then `run.pid` is removed.
Each row in the merged `summary.json` carries its own `finished_at` (UTC), `build_info` and `hz_seconds`.
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import logging
import os
import signal
import statistics
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench import lib
from bench.acceptance import CORPUS, MASTER, Callo, Ctx, row_drop_in, row_selftest
from bench.acceptance import MODEL as QWEN9B
from bench.acceptance import SUITE as V01_SUITE
from bench.frames import fresh_frames, maze_frame, spike_frames
from bench.kv_prompts import KV_ITEMS
from callo.engine import Engine, local_engine
from callo.systemone import parse_request

TIEL = lib.env_path(lib.MODELS_ENV, "peculiar-ragdoll", "Tiel-Coder-35B-A3B-GGUF", "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf")
TIEL_MMPROJ = lib.env_path(lib.MODELS_ENV, "peculiar-ragdoll", "Tiel-Coder-35B-A3B-GGUF", "mmproj-BF16.gguf")
QWEN9B_MMPROJ = lib.env_path(lib.MODELS_ENV, "local", "system1", "mmproj-F16.gguf")
CTX = 32768
G0_DELIMITER = "<|im_start|>user"   # G0's pinned user-turn delimiter (parity diagnostic): g0_parity compares like-for-like
KV_FILL_TOKENS = 300                # tokens parked in another slot for the kv_sensitivity s2 state (diagnostic s2)
S1_MEDIAN_MAX_MS = 411.0         # 1.25 x the spike's 2 Hz median of 329 ms (bench2_tiel_q4km.json)
S1_MIN_HZ = 1.95
S2_MIN_TOK_S = 27.0              # 0.75 x the spike's 35.8 tok/s
S2_CYCLE_S = 11.0                # the no-instinct System 2 cycle
INSTINCT_EVERY_S = 10.0
log = logging.getLogger("bench")

# ---- the demo's request shapes, frozen here so the bench never depends on the demo's code -----------

LEGEND = ("The image is an 11 by 11 tile view centred on Pac-Man, the yellow circle in the middle. Up is the top of the "
          "image, at the moment your choice is taken. Pac-Man has just reached the tile shown, facing the way it came. "
          "Blue squares are walls. Small peach dots are food; large peach dots are power pellets. Ghosts are red, "
          "pink, cyan and orange. A white ghost is frightened: touching it eats it. Touching any other ghost ends "
          "the game.")
DEFAULT_ORDERS = "Eat nearby dots. Move away from any coloured ghost within three tiles. Chase white ghosts."
TICK_CONTEXT = LEGEND + "\n\nStanding orders from System 2:\n" + DEFAULT_ORDERS
TICK_QUESTION = {"move": {"type": "choice", "instructions": "Which way should Pac-Man go next?",
                          "criteria": {"up": None, "down": None, "left": None, "right": None,
                                       "stay": "wait on this tile"}}}
# The spike's standing orders (mm_spike/accuracy_standing.py), about 180 tokens.
STANDING = ("Standing orders from the strategist: avoid the red ghost at all costs; prefer corridors with dots; "
            "when a ghost is within three tiles, move directly away from it. ") * 4
# The spike's own question text (accuracy_standing.py), so the 36/36 gate is calibrated on the prompt it came from.
READOUT_QUESTION = {"dir": {"type": "choice", "instructions": "The yellow circle is Pac-Man and the red shape is a "
                            "ghost. In which direction is the ghost, relative to Pac-Man?",
                            "criteria": {"left": None, "right": None, "up": None, "down": None}}}
MAZE = "\n".join([
    "r00 #####################", "r01 #o........#........o#", "r02 #.##.####.#.####.##.#", "r03 #...................#",
    "r04 #.##.#.#######.#.##.#", "r05 #....#....#....#....#", "r06 ####.####.#.####.####", "r07 ####.#.........#.####",
    "r08 ####.#.###-###.#.####", "r09 ####.#.#GGGGG#.#.####", "r10 ####.#.#######.#.####", "r11 ####.#.........#.####",
    "r12 ####.#.#######.#.####", "r13 #.........#.........#", "r14 #.##.####.#.####.##.#", "r15 #o.#.............#.o#",
    "r16 ##.#.#.#######.#.#.##", "r17 #....#....#....#....#", "r18 #.#######.#.#######.#", "r19 #...................#",
    "r20 #####################"])
LEGEND_TEXT = ("You answer questions about a Pac-Man-like maze from text alone; there is no image. The maze is below: 21 "
               "by 21 tiles, coordinates (column, row) from (0,0) at the top left, one line per row. # is a wall, . a "
               "dot, o a power pellet, - the ghost-house door and G the ghost house. There are no wrap-around "
               "tunnels.\n\n" + MAZE + "\n\nA ghost never reverses except when its mode changes, and at a junction it "
               "picks one of the open exits. 'Its next junction' means the next junction it enters: a tile with three "
               "or more open exits.")
LEGEND_BOARD = ("The image is the whole maze. Pac-Man is the yellow circle. Blue squares are walls. Small peach dots "
                "are food; large peach dots are power pellets. Ghosts are red, pink, cyan and orange; a white ghost "
                "is frightened.")
FOUR = {"up": None, "down": None, "left": None, "right": None}
TEXT_INSTINCT_Q = {"q": {"type": "choice", "criteria": FOUR,
                         "instructions": "When the red ghost leaves its next junction, which way will it go?"}}
VIEW_INSTINCT_Q = {"q": {"type": "choice", "criteria": FOUR,
                         "instructions": "Where is the red ghost relative to Pac-Man?"}}
S2_PROMPT = ("Write a detailed, step-by-step strategy guide for a maze game in which a player collects dots while "
             "four ghosts chase them. Cover corridors, junctions, power pellets and timing.")


def positions(i: int) -> str:
    """A use-A state: positions only."""
    return (f"Pac-Man: ({4 + i % 12},15), heading left.\nRed ghost: ({1 + (i * 7) % 18},7), heading right, chase.\n"
            "Pink ghost: (10,9), in the house.\nTime to the next mode switch: 9 s.\n"
            "Remaining pellets: (1,1), (19,1), (1,15), (19,15).")


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def tick_body(png: bytes, context: str = TICK_CONTEXT) -> dict:
    return {"state": "", "images": [data_url(png)], "context": context, "questions": TICK_QUESTION}


def text_instinct_body(i: int) -> dict:
    return {"state": positions(i), "context": LEGEND_TEXT, "questions": TEXT_INSTINCT_Q}


def view_instinct_body(png: bytes) -> dict:
    return {"state": "", "images": [data_url(png)], "context": LEGEND_BOARD, "questions": VIEW_INSTINCT_Q}


def s2_stream_payload() -> dict:
    return {"model": "x", "messages": [{"role": "user", "content": S2_PROMPT}], "stream": True, "max_tokens": 4096,
            "ignore_eos": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False}}


def s2_cycle_payload_fn(text: str):
    """The demo's per-cycle shape: a fresh 224² board, then about 430 text tokens."""
    salts = itertools.count(50_000)

    def payload() -> dict:
        board = data_url(maze_frame(224, "up", 2, salt=next(salts)))
        return {**s2_stream_payload(), "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": board}}, {"type": "text", "text": text}]}]}
    return payload


# ---- context, requests -------------------------------------------------------------------------------

class V02Ctx:
    def __init__(self, out: Path, *, g0_records: Path | None = None, hz_seconds: float = 60.0, gguf: str = TIEL,
                 instinct_every: float = INSTINCT_EVERY_S):
        self.out, self.g0_records, self.hz_seconds, self.gguf = out, g0_records, hz_seconds, gguf
        self.instinct_every = instinct_every
        self.build_info: str | None = None          # of the instance the current row runs on (row provenance)
        self.flags: list[str] | None = None          # callo launch flags of that instance (row provenance, I-3)
        self._engines: dict[str, Engine] = {}

    def save(self, row: str, name: str, obj: Any) -> None:
        d = self.out / row
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(json.dumps(obj, indent=1, default=str))

    def engine(self, c: Any) -> Engine:
        """callo's own engine code in process, against the upstream callo launched (no proxy): the exact prompt
        string callo renders, and the delimiter it derived."""
        if c.upstream not in self._engines:
            self._engines[c.upstream] = local_engine(c.upstream, gguf=self.gguf)
        return self._engines[c.upstream]


def s1_post(c: Any, body: dict, lane: str | None) -> dict:
    headers = {"X-Callo-Conversation": lane} if lane else {}
    st, h, resp = lib.post(f"{c.url}/v1/systemone", body, headers=headers, timeout=600)
    ok = st == 200 and isinstance(resp, dict)
    return {"ok": ok, "status": st, "slot": h.get("x-callo-slot"),
            "timings": json.loads(h.get("x-callo-timings") or "[]"),
            "answers": resp.get("answers") if ok else None, "error": None if ok else str(resp)[:300]}


def first_pass(row: dict) -> dict:
    """The first question's X-Callo-Timings entry; a capped header ends in `{"truncated": k}`, which is not one."""
    return next((t for t in row.get("timings") or [] if set(t) != {"truncated"}), {})


def probs(row: dict, qid: str) -> dict[str, float]:
    return row["answers"][qid]["probabilities"]


def max_dp(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(a[k] - b[k]) for k in a)


def system_turn_tokens(c: Any, ctx: V02Ctx, body: dict) -> int | None:
    """Tokens of the rendered prompt before the user-turn delimiter: what a restored system-turn checkpoint covers."""
    eng = ctx.engine(c)
    req = parse_request(body)
    prompt = eng.render_prompt(req.state, req.questions[0], context=req.context, image_urls=req.image_urls)
    if not eng.user_delimiter or eng.user_delimiter not in prompt:
        return None
    prefix = prompt[:prompt.index(eng.user_delimiter)]
    return len(lib.post(f"{c.url}/tokenize", {"content": prefix, "add_special": True})[2]["tokens"])


def _invalid(reasons: list[str | None]) -> str | None:
    return "; ".join(r for r in reasons if r) or None


def _errors(rows: list[dict]) -> str | None:
    bad = [r for r in rows if not r["ok"]]
    return f"{len(bad)} request(s) failed, first: HTTP {bad[0]['status']} {bad[0]['error']}" if bad else None


def erase_slots(c: Any, n: int) -> None:
    """Erase every slot 0..n-1 on the upstream (needs `--slot-save-path`). With `--cache-ram 0` on the launch this
    leaves the KV truly empty: no RAM prompt cache survives to restore a prefix (parity diagnostic)."""
    for s in range(n):
        lib.post(f"{c.upstream}/slots/{s}?action=erase", {})


def fill_slot(c: Any, slot: int, n_tokens: int) -> None:
    """Park `n_tokens` dummy tokens in `slot` so the unified KV holds content in another slot while a readout runs on
    slot 0 (the diagnostic's s2 state). The ids never matter: only the cell layout does (diagnostic)."""
    lib.post(f"{c.upstream}/completion",
             {"prompt": list(range(1000, 1000 + n_tokens)), "n_predict": 1, "id_slot": slot, "cache_prompt": True})


# ---- rows on the main TielCoder instance (--mmproj, --s1-slots 2) ----------------------------------------

def row_selftests(c: Any, ctx: V02Ctx) -> dict:
    text = c.log_text()
    lines = [ln for ln in c.lines if any(k in ln for k in ("self-test", "answer slot", "vision", "delimiter"))]
    ok = ("self-test ok: 2 + 2" in text and "answer slot check: answer slot confirmed" in text
          and "vision self-test ok" in text)
    ctx.save("selftests", "log.json", lines)
    return {"pass": ok, "lines": lines}


def row_readout(c: Any, ctx: V02Ctx) -> dict:
    frames = spike_frames()
    example = None
    if frames:
        req = parse_request({"state": "", "images": [data_url(frames[0][3].read_bytes())], "context": STANDING,
                             "questions": READOUT_QUESTION})
        example = ctx.engine(c).render_prompt(req.state, req.questions[0], context=req.context,
                                              image_urls=req.image_urls)
    rows = []
    for gold, dist, size, path in frames:
        r = s1_post(c, {"state": "", "images": [data_url(path.read_bytes())], "context": STANDING,
                        "questions": READOUT_QUESTION}, "aux")
        p = probs(r, "dir") if r["ok"] else None
        rows.append({"file": path.name, "size": size, "dist": dist, "gold": gold, "ok": r["ok"],
                     "choice": max(p, key=p.get) if p else None, "p_gold": p[gold] if p else None,
                     "timings": r["timings"], "error": r["error"]})
    ok = [r for r in rows if r["ok"]]
    correct = sum(r["choice"] == r["gold"] for r in ok)
    mean = statistics.fmean(r["p_gold"] for r in ok) if ok else None
    sizes = {str(s): {"correct": sum(r["choice"] == r["gold"] for r in ok if r["size"] == s),
                      "n": sum(1 for r in rows if r["size"] == s),
                      "mean_p_gold": statistics.fmean([r["p_gold"] for r in ok if r["size"] == s] or [0.0]),
                      "min_p_gold": min([r["p_gold"] for r in ok if r["size"] == s] or [0.0])}
             for s in sorted({r["size"] for r in rows})}
    out = {"n": len(rows), "correct": correct, "mean_p_gold": mean, "min_p_gold": min((r["p_gold"] for r in ok),
                                                                                       default=None),
           "per_size": sizes, "rows": rows, "rendered_prompt_example": example,
           "invalid": _invalid([None if len(frames) == 36 else f"{len(frames)} spike frames found, expected 36",
                                _errors(rows)])}
    out["pass"] = out["invalid"] is None and correct == 36 and mean is not None and mean >= 0.95
    ctx.save("readout", "summary.json", out)
    return out


def _mm(r: Any) -> dict:
    return {"tokens_evaluated": r.tokens_evaluated, "cache_n": r.cache_n, "prompt_n": r.prompt_n,
            "prompt_ms": r.prompt_ms}


def _round_trip_token(eng: Engine, prompt: str, logprobs: dict[int, float]) -> tuple[int, str] | None:
    """The likeliest of the top 5 next tokens whose detokenized text re-tokenizes to exactly itself after the prompt's
    tail: the walk's own round-trip check, so pass 2 below is a pass the walk could really send."""
    marker = eng.media_marker or ""
    tail = prompt[prompt.rfind(marker) + len(marker):] if marker else prompt
    base = eng.scorer.tokenize(tail, add_special=False, parse_special=True)
    for tid, _ in sorted(logprobs.items(), key=lambda kv: -kv[1])[:5]:
        text = eng.scorer.detokenize([tid])
        if "\ufffd" not in text and eng.scorer.tokenize(tail + text, add_special=False,
                                                        parse_special=True) == base + [tid]:
            return tid, text
    return None


def row_walk_reuse(c: Any, ctx: V02Ctx) -> dict:
    """The "walk pass reuse" row in the walk's real shape: pass 1 is an image request
    A, pass 2 is A plus the detokenized text of A's likeliest round-tripping token (what `_answer_media`'s dist_fn
    sends), on the `aux` lane's slot, through callo's own scorer. PASS iff pass 2 reuses all of A (`cache_n ==
    A.tokens_evaluated`), evaluates one token (`prompt_n == 1`) and takes <= 40 ms server-side. An identical re-send
    is reported ungated: `prompt_n` 1 on pure attention, 4 on hybrids (the checkpoint 4 tokens before the end)."""
    eng = ctx.engine(c)
    req = parse_request(tick_body(maze_frame(224, "left", 3, salt=7)))
    prompt = eng.render_prompt(req.state, req.questions[0], context=req.context, image_urls=req.image_urls)
    slot = c.stats()["router"]["lanes"].get("aux")

    def send(text: str):
        return eng.scorer.next_logprobs_mm(text, req.images, top_k=eng.top_k, id_slot=slot,
                                           delimiter=eng.user_delimiter)

    a = send(prompt)
    step = _round_trip_token(eng, prompt, a.logprobs)
    b = send(prompt + step[1]) if step else None
    again = send(prompt)
    out = {"slot": slot, "a": _mm(a), "step_token": list(step) if step else None, "pass2": _mm(b) if b else None,
           "identical_resend": _mm(again),
           "identical_resend_expectation": "prompt_n 1 on pure attention, 4 on hybrids (ungated)",
           "invalid": _invalid([None if slot is not None else "lane aux is not bound to a slot (/callo/stats)",
                                None if step else "none of A's top 5 tokens round-trips after the prompt's tail"])}
    out["pass"] = (out["invalid"] is None and b is not None and b.cache_n is not None
                   and b.cache_n == a.tokens_evaluated and b.prompt_n == 1
                   and b.prompt_ms is not None and b.prompt_ms <= 40)
    ctx.save("walk_reuse", "summary.json", out)
    return out


def row_prefix_reuse(c: Any, ctx: V02Ctx) -> dict:
    bodies = [tick_body(f) for f in fresh_frames(3, salt0=20_000)]
    ticks = [s1_post(c, b, "aux") for b in bodies]
    n_sys = system_turn_tokens(c, ctx, bodies[0])
    tick2 = first_pass(ticks[1])
    out = {"system_turn_tokens": n_sys, "ticks": [first_pass(t) for t in ticks],
           "invalid": _invalid([_errors(ticks), None if n_sys else "no user-turn delimiter in the rendered prompt"])}
    out["pass"] = out["invalid"] is None and (tick2.get("cache_n") or 0) >= n_sys
    ctx.save("prefix_reuse", "summary.json", out)
    return out


def row_context_reuse(c: Any, ctx: V02Ctx) -> dict:
    """Text only, same ~600-token context, different states, one lane: B cold, then A (its cache_n is the check),
    then B warm; |dp| between B cold and B warm is the bit-instability the cache costs."""
    b_body, a_body = text_instinct_body(1), text_instinct_body(2)
    b_cold, a_warm, b_warm = s1_post(c, b_body, "aux"), s1_post(c, a_body, "aux"), s1_post(c, b_body, "aux")
    n_sys = system_turn_tokens(c, ctx, a_body)
    context_tokens = len(lib.post(f"{c.url}/tokenize", {"content": LEGEND_TEXT})[2]["tokens"])
    rows = [b_cold, a_warm, b_warm]
    out = {"context_tokens": context_tokens, "system_turn_tokens": n_sys,
           "passes": {"b_cold": first_pass(b_cold), "a_warm": first_pass(a_warm), "b_warm": first_pass(b_warm)},
           "invalid": _invalid([_errors(rows), None if n_sys else "no user-turn delimiter in the rendered prompt"])}
    out["max_dp_cold_vs_warm"] = max_dp(probs(b_cold, "q"), probs(b_warm, "q")) if out["invalid"] is None else None
    out["pass"] = out["invalid"] is None and (first_pass(a_warm).get("cache_n") or 0) >= n_sys
    ctx.save("context_reuse", "summary.json", out)
    return out


def row_cache_drift(c: Any, ctx: V02Ctx) -> dict:
    """The same frame truly cold, then warm (after 20 other frames on the same lane): max |dp| over the options. The
    cold pass is made truly cold by erasing the aux lane's slot first; with the main instance's `--cache-ram 0` no RAM
    prompt cache survives to restore a prefix, so `cache_n` on the cold pass is 0 (the parity diagnostic found the
    earlier bench's "cold" pass reused 349 cached tokens). If the cold pass is not cold the row is INVALID, never
    PASS: it measured warm-vs-warm."""
    frame = maze_frame(224, "right", 2, salt=30_000)
    aux_slot = c.stats()["router"]["lanes"].get("aux")
    if aux_slot is not None:
        erase_slots(c, aux_slot + 1)                               # clears slot 0..aux so the cold pass starts empty
    cold = s1_post(c, tick_body(frame), "aux")
    others = [s1_post(c, tick_body(f), "aux") for f in fresh_frames(20, salt0=31_000)]
    warm = s1_post(c, tick_body(frame), "aux")
    cold_cache_n = first_pass(cold).get("cache_n")
    out = {"cold": first_pass(cold), "warm": first_pass(warm), "cold_was_cold": cold_cache_n == 0,
           "invalid": _invalid([_errors([cold, warm, *others]),
                                None if aux_slot is not None else "lane aux is not bound to a slot (/callo/stats)",
                                None if cold_cache_n == 0 else f"cold pass was not cold: cache_n {cold_cache_n} "
                                "(need --cache-ram 0 and the slot erased); measured warm vs warm"])}
    out["max_dp"] = max_dp(probs(cold, "move"), probs(warm, "move")) if out["invalid"] is None else None
    out["pass"] = None
    ctx.save("cache_drift", "summary.json", out)
    return out


def _paced_s1(c: Any, ctx: V02Ctx, lane: str, salt0: int) -> list[dict]:
    frames = fresh_frames(int(ctx.hz_seconds * 2) + 8, salt0=salt0)
    return lib.paced_ticks(lambda k: s1_post(c, tick_body(frames[k % len(frames)]), lane), hz=2.0,
                           duration_s=ctx.hz_seconds)


def _with_s2(c: Any, ctx: V02Ctx, s2: lib.ChatStream, lane: str, salt0: int, sample_memory: bool = False):
    mem = lib.MemorySampler(c.pid, interval=5.0) if sample_memory and c.pid else None
    if mem is not None:
        mem.start()
    s2.start()
    try:
        if not s2.first_delta.wait(timeout=120):
            raise RuntimeError("System 2's stream produced no token within 120 s")
        ticks = _paced_s1(c, ctx, lane, salt0)
    finally:
        s2.stop()
        if mem is not None:
            mem.stop()
            mem.join(timeout=30)
    return ticks, (mem.rows if mem is not None else [])


def _tick_summary(ticks: list[dict], ctx: V02Ctx) -> dict:
    ok = [t for t in ticks if t["ok"]]
    lat = [t["latency_ms"] for t in ok]
    return {"n": len(ticks), "ok": len(ok), "median_ms": statistics.median(lat) if lat else None,
            "p90_ms": lib.percentile(lat, 0.9) if lat else None, "achieved_hz": len(ok) / ctx.hz_seconds,
            "slots": dict(Counter(str(t["slot"]) for t in ok))}


def row_hz2_stream(c: Any, ctx: V02Ctx) -> dict:
    s2 = lib.ChatStream(f"{c.url}/v1/chat/completions", s2_stream_payload, headers={"X-Callo-Conversation": "s2"})
    ticks, mem = _with_s2(c, ctx, s2, "s1", 40_000, sample_memory=True)
    s = _tick_summary(ticks, ctx)
    w0, w1 = ticks[0]["t_start"], ticks[-1]["t_end"]
    s2_slots = sorted({str(r["slot"]) for r in s2.requests})
    out = {"s1": s, "s2_tok_s": lib.tokens_in_window(s2.deltas, w0, w1), "s2_slots": s2_slots,
           "s2_requests": len(s2.requests), "footprint_max": max((m["footprint"] for m in mem if m.get("footprint")),
                                                                 default=None),
           "invalid": _invalid([_errors(ticks), None if set(s["slots"]) == {"0"} else f"S1 ran on slots {s['slots']}",
                                None if set(s2_slots) <= {"2", "3"} else f"S2 ran on slots {s2_slots}",
                                s2.problems()])}
    out["pass"] = (out["invalid"] is None and s["median_ms"] is not None and s["median_ms"] <= S1_MEDIAN_MAX_MS
                   and s["achieved_hz"] >= S1_MIN_HZ and (out["s2_tok_s"] or 0) >= S2_MIN_TOK_S)
    for name, obj in (("ticks.json", ticks), ("s2_requests.json", s2.requests), ("memory.json", mem)):
        ctx.save("hz2_stream", name, obj)
    ctx.save("hz2_stream", "summary.json", out)
    return out


def row_hz2_prefill(c: Any, ctx: V02Ctx) -> dict:
    text, measured = lib.build_long_text(
        lambda t: len(lib.post(f"{c.url}/tokenize", {"content": t})[2]["tokens"]), 430, CORPUS)
    s2 = lib.ChatStream(f"{c.url}/v1/chat/completions", s2_cycle_payload_fn(text),
                        headers={"X-Callo-Conversation": "s2"}, restart_every=S2_CYCLE_S)
    ticks, _ = _with_s2(c, ctx, s2, "s1", 45_000)
    phases = [(r["t_send"], r["t_first"]) for r in s2.requests if r["t_first"]]
    hit, rest = lib.split_by_overlap([t for t in ticks if t["ok"]], phases)

    def med(rows: list[dict]) -> float | None:
        return statistics.median([r["latency_ms"] for r in rows]) if rows else None

    out = {"s1": _tick_summary(ticks, ctx), "s2_text_tokens": measured, "s2_cycles": len(s2.requests),
           "s2_prompt_phases": len(phases), "ticks_overlapping_s2_prompt": len(hit), "ticks_other": len(rest),
           "median_ms_overlapping_s2_prompt": med(hit), "median_ms_other": med(rest),
           "max_ms_overlapping_s2_prompt": max((r["latency_ms"] for r in hit), default=None),
           "invalid": _invalid([_errors(ticks), None if hit else "no tick overlapped a System 2 prompt phase",
                                s2.problems()]),
           "pass": None}
    ctx.save("hz2_prefill", "ticks.json", ticks)
    ctx.save("hz2_prefill", "s2_requests.json", s2.requests)
    ctx.save("hz2_prefill", "summary.json", out)
    return out


MAIN_ROWS = ["selftests", "readout", "walk_reuse", "prefix_reuse", "context_reuse", "cache_drift",
             "hz2_stream", "hz2_prefill"]
_MAIN = {"selftests": row_selftests, "readout": row_readout, "walk_reuse": row_walk_reuse,
         "prefix_reuse": row_prefix_reuse, "context_reuse": row_context_reuse, "cache_drift": row_cache_drift,
         "hz2_stream": row_hz2_stream, "hz2_prefill": row_hz2_prefill}


def bind_lanes(c: Any, lanes: list[str]) -> list[dict]:
    """One warm-up request per lane, in order, so lane i binds to reserved slot i (first sight). Each lane
    is warmed with its own system prefix: the text legend, the board legend, or the tick context."""
    warm = []
    for i, lane in enumerate(lanes):
        png = maze_frame(224, "down", 2, salt=60_000 + i)
        body = (text_instinct_body(0) if lane == "instinct_text" else view_instinct_body(png)
                if lane == "instinct_view" else tick_body(png))
        warm.append(s1_post(c, body, lane))
    return warm


# ---- rows on their own instances -------------------------------------------------------------------------

def _launch(name: str, out: Path, *, model: str, flags: list[str]) -> Any:
    """The one place a model server starts (a seam: the smoke test swaps in callo on a fake upstream)."""
    return Callo(out / f"callo_{name}", model=model, ctx=CTX, flags=flags)


def _instance(name: str, out: Path, *, model: str, flags: list[str], ctx: V02Ctx | None = None) -> Any:
    if ctx is not None:
        ctx.build_info, ctx.flags = None, None
    busy = lib.running_gpu_servers()
    if busy:
        raise RuntimeError(f"a model server is already running, refusing to start another: {busy}")
    c = _launch(name, out, model=model, flags=flags)
    if ctx is not None:
        ctx.build_info, ctx.flags = getattr(c, "build_info", None), list(flags)
    return c


def row_demo_slots(ctx: V02Ctx, out_dir: Path) -> dict:
    """The "2 Hz while System 2 streams" row at --s1-slots 3, System 2 on slot 3, all three lanes warmed in demo
    order, a text instinct and a view instinct alternating every 10 s."""
    c = _instance("demo_slots", out_dir, model=TIEL, flags=["--mmproj", TIEL_MMPROJ, "--s1-slots", "3"], ctx=ctx)
    instincts: list[dict] = []
    try:
        warm = bind_lanes(c, ["s1", "instinct_text", "instinct_view"])
        boards = itertools.count(70_000)
        stop = threading.Event()

        def instinct_loop() -> None:
            for i in itertools.count():
                if stop.wait(ctx.instinct_every):
                    return
                lane = "instinct_text" if i % 2 == 0 else "instinct_view"
                body = text_instinct_body(i + 3) if lane == "instinct_text" else view_instinct_body(
                    maze_frame(224, "up", 3, salt=next(boards)))
                t0 = time.time()
                r = s1_post(c, body, lane)
                instincts.append({"lane": lane, "t_start": t0, "t_end": time.time(), "ok": r["ok"], "slot": r["slot"],
                                  "first_pass": first_pass(r), "error": r["error"]})

        th = threading.Thread(target=instinct_loop, daemon=True)
        s2 = lib.ChatStream(f"{c.url}/v1/chat/completions", s2_stream_payload, headers={"X-Callo-Conversation": "s2"})
        th.start()
        try:
            ticks, _ = _with_s2(c, ctx, s2, "s1", 80_000)
        finally:
            stop.set()
            th.join(timeout=120)
        stats = c.stats()
    finally:
        c.finish()
    s = _tick_summary(ticks, ctx)
    w0, w1 = ticks[0]["t_start"], ticks[-1]["t_end"]
    lanes = stats["router"]["lanes"]
    s2_slots = sorted({str(r["slot"]) for r in s2.requests})
    out = {"lanes": lanes, "s1": s, "s2_tok_s": lib.tokens_in_window(s2.deltas, w0, w1), "s2_slots": s2_slots,
           "warm": [first_pass(w) for w in warm], "instincts": instincts,
           "instinct_cache_n": {ln: [i["first_pass"].get("cache_n") for i in instincts if i["lane"] == ln]
                                for ln in ("instinct_text", "instinct_view")},
           "s1_lane_busy": stats["router"].get("s1_lane_busy"),
           "invalid": _invalid([
               None if lanes == {"s1": 0, "instinct_text": 1, "instinct_view": 2} else f"lanes bound as {lanes}",
               None if s2_slots == ["3"] else f"System 2 ran on slots {s2_slots}", _errors(ticks),
               _errors(instincts), None if instincts else "no instinct request ran", s2.problems()]),
           "pass": None}
    ctx.save("demo_slots", "ticks.json", ticks)
    ctx.save("demo_slots", "summary.json", out)
    return out


def row_no_vision(ctx: V02Ctx, out_dir: Path) -> dict:
    c = _instance("no_vision", out_dir, model=TIEL, flags=[], ctx=ctx)
    try:
        body = tick_body(maze_frame(224, "left", 2, salt=90_000))
        with_images = s1_post(c, body, None)
        without = s1_post(c, {k: v for k, v in body.items() if k != "images"}, None)
        vision = c.stats()["vision"]
    finally:
        c.finish()
    out = {"vision": vision, "with_images_status": with_images["status"], "with_images_detail": with_images["error"],
           "without_images_status": without["status"],
           "pass": with_images["status"] == 501 and without["status"] == 200 and vision is False}
    ctx.save("no_vision", "summary.json", out)
    return out


def row_text_regression(ctx: V02Ctx, out_dir: Path) -> dict:
    """v0.1's "drop-in content" and "self-test + answer-slot check" rows, rerun on Qwen3.5-9B Q8_0 without an mmproj."""
    c = _instance("text_9b", out_dir, model=QWEN9B, flags=[], ctx=ctx)
    try:
        v01 = Ctx(out_dir, V01_SUITE, 1)
        drop_in = row_drop_in(c, v01, name="text_regression_drop_in")
        selftest = row_selftest(c, v01, name="text_regression_selftest")
    finally:
        c.finish()
    out = {"drop_in": drop_in, "selftest": selftest, "pass": bool(drop_in.get("pass") and selftest.get("pass"))}
    ctx.save("text_regression", "summary.json", out)
    return out


def row_selftest_9b_vision(ctx: V02Ctx, out_dir: Path) -> dict:
    c = _instance("vision_9b", out_dir, model=QWEN9B, flags=["--mmproj", QWEN9B_MMPROJ], ctx=ctx)
    try:
        lines = [ln for ln in c.lines if "vision" in ln or "self-test" in ln or "delimiter" in ln]
    finally:
        c.finish()
    out = {"lines": lines, "vision_self_test_ok": any("vision self-test ok" in ln for ln in lines), "pass": None}
    ctx.save("selftest_9b_vision", "summary.json", out)
    return out


def _slot_dir(out_dir: Path, *parts: str) -> str:
    """A fresh `--slot-save-path` directory (llama-server's `/slots?action=erase` needs one), with a trailing slash."""
    d = out_dir.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return str(d) + "/"


def row_g0_parity(ctx: V02Ctx, out_dir: Path) -> dict:
    """The "G0 parity" row, re-specified after the parity diagnostic as a DECISION-equivalence test on
    its own instance, so nothing else touches the KV: TielCoder + mmproj, G0's pinned delimiter `<|im_start|>user`,
    `--cache-ram 0`, `--s1-slots 1` (every /v1/systemone request routes to the one reserved slot 0). Before each
    record all four slots are erased, so each reads cold on slot 0 with every other slot empty — exactly G0's state.
    PASS iff, on 106/106, the prompt is byte-identical, the argmax agrees, and max |Δp| < 1e-3. Not bit-exact: G0
    records `letter_mass` and callo serves `bounded_readout`, two formulas that differ by ~1e-5 by construction
    (diagnostic test 3); 1e-3 is a readout-method tolerance far below any decision threshold. The observed max |Δp| is
    reported alongside the gate."""
    if ctx.g0_records is None or not Path(ctx.g0_records).is_file():
        return {"pass": False, "error": f"no G0 records at {ctx.g0_records}: they come from the game-client demo's "
                                       "gate harness, which is not part of this repository"}
    records = [json.loads(ln) for ln in Path(ctx.g0_records).read_text().splitlines() if ln.strip()]
    bad = lib.check_g0_records(records)
    if bad:
        return {"pass": False, "invalid": f"{ctx.g0_records}: {bad} (the G0 record contract)"}
    c = _instance("g0_parity", out_dir, model=TIEL,
                  flags=["--mmproj", TIEL_MMPROJ, "--s1-slots", "1", "--user-delimiter", G0_DELIMITER,
                         "--", "--cache-ram", "0", "--slot-save-path", _slot_dir(out_dir, "g0_parity", "slots")],
                  ctx=ctx)
    try:
        eng = ctx.engine(c)

        def render(body: dict) -> str:
            req = parse_request(body)
            return eng.render_prompt(req.state, req.questions[0], context=req.context, image_urls=req.image_urls)

        slots_seen: set[str | None] = set()

        def answer(body: dict) -> dict[str, float]:
            erase_slots(c, c.parallel)                             # cold: slot 0 empty, every other slot empty
            r = s1_post(c, body, None)                             # --s1-slots 1 routes to the reserved slot 0
            if not r["ok"]:
                raise RuntimeError(f"HTTP {r['status']}: {r['error']}")
            slots_seen.add(r["slot"])
            return probs(r, next(iter(body["questions"])))

        out = lib.g0_parity(records, render=render, answer=answer, marker=eng.media_marker or "")
        out["slots_seen"] = sorted(str(s) for s in slots_seen)     # must be ["0"]; any other slot explains a flip
    finally:
        c.finish()
    ctx.save("g0_parity", "summary.json", out)
    return {k: v for k, v in out.items() if k != "rows"}


def row_kv_sensitivity(ctx: V02Ctx, out_dir: Path) -> dict:
    """Ungated. The server-state sensitivity the parity diagnostic measured, as its own row: 40 fixed text prompts
    read in state s0 (every other slot empty) and s2 (slot 3 holds ~300 tokens), on two instances in sequence — (a)
    the default launch (`-kvu`) and (b) `--no-kvu` with the same per-slot context (`-c = ctx x parallel`, so both
    instances run `-c 32768 -np 4`, as the diagnostic did). Each readout is taken on slot 0 through callo's own engine
    with all slots erased first (`--cache-ram 0`, so no prompt-cache carry-over). Reports median/max |Δp|, median/max
    top-2 log-odds shift and argmax agreement for (a) and (b); (b) is expected bit-exact on text."""
    def measure(c: Any) -> list[tuple[dict, dict]]:
        eng = ctx.engine(c)
        pairs: list[tuple[dict, dict]] = []
        for body in KV_ITEMS:
            qid = next(iter(body["questions"]))
            erase_slots(c, c.parallel)                             # s0: all slots empty
            s0 = eng.answer_request(body, id_slot=0)["answers"][qid]["probabilities"]
            erase_slots(c, c.parallel)
            fill_slot(c, c.parallel - 1, KV_FILL_TOKENS)           # s2: the last slot holds ~300 tokens
            s2 = eng.answer_request(body, id_slot=0)["answers"][qid]["probabilities"]
            pairs.append((s0, s2))
        return pairs

    configs = [("kvu", ["--mmproj", TIEL_MMPROJ, "--s1-slots", "1",
                        "--", "--cache-ram", "0", "--slot-save-path", _slot_dir(out_dir, "kv_sensitivity", "kvu")]),
               ("no_kvu", ["--mmproj", TIEL_MMPROJ, "--s1-slots", "1", "--no-kvu", "--ctx", str(CTX // 4),
                           "--", "--cache-ram", "0",
                           "--slot-save-path", _slot_dir(out_dir, "kv_sensitivity", "no_kvu")])]
    results = {}
    for name, flags in configs:                                   # one instance at a time, in sequence
        c = _instance(f"kv_sensitivity_{name}", out_dir, model=TIEL, flags=flags, ctx=ctx)
        try:
            results[name] = lib.kv_sensitivity_summary(measure(c))
        finally:
            c.finish()
    out = {"kvu": results["kvu"], "no_kvu": results["no_kvu"], "n_prompts": len(KV_ITEMS),
           "no_kvu_expected_bit_exact": True, "pass": None,
           "invalid": _invalid([None if len(KV_ITEMS) == 40 else f"{len(KV_ITEMS)} kv prompts, expected 40"])}
    ctx.save("kv_sensitivity", "summary.json", out)
    return out


STANDALONE = {"g0_parity": row_g0_parity, "kv_sensitivity": row_kv_sensitivity, "demo_slots": row_demo_slots,
              "no_vision": row_no_vision, "text_regression": row_text_regression,
              "selftest_9b_vision": row_selftest_9b_vision}
ROWS = [*MAIN_ROWS, *STANDALONE]


def lane_binding_problem(lanes: list[str], warm: list[dict]) -> str | None:
    """None when warm-up i succeeded on reserved slot i, as `bind_lanes` intends."""
    bad = [f"{lane}: HTTP {w['status']} on slot {w['slot']}" for i, (lane, w) in enumerate(zip(lanes, warm, strict=True))
           if not w["ok"] or w["slot"] != str(i)]
    return f"lane binding failed ({'; '.join(bad)})" if bad else None


def run_main_rows(c: Any, ctx: V02Ctx, names: list[str], record) -> None:
    """The rows that share one instance, after binding lane s1 to slot 0 and aux to slot 1. If a lane did not bind
    to its slot, every row but `selftests` is invalid: it ran on slots other than the ones it reports."""
    lanes = ["s1", "aux"]
    bad_lanes = lane_binding_problem(lanes, bind_lanes(c, lanes))
    if bad_lanes:
        log.warning(bad_lanes)
    for name in names:
        log.info("row %s", name)
        try:
            res = _MAIN[name](c, ctx)
        except Exception as e:  # noqa: BLE001 -- one row's failure must not lose the others
            log.exception("row %s failed", name)
            res = {"pass": False, "error": f"{type(e).__name__}: {e}"}
        if bad_lanes and name != "selftests":
            res = {**res, "invalid": _invalid([bad_lanes, res.get("invalid")])}
        record(name, res)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(ROWS))
    ap.add_argument("--g0-records", default=None,
                    help="the G0 first-decision JSONL (106 lines) from the game-client demo's gate harness, which is "
                         "not part of this repository")
    ap.add_argument("--hz-seconds", type=float, default=60.0, help="length of each paced 2 Hz row")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rows = [r for r in a.rows.split(",") if r]
    if not rows or any(r not in ROWS for r in rows):
        print(f"unknown row(s) in {rows}; choose from {','.join(ROWS)}")
        return 2
    unset = lib.unset_paths(("", TIEL, lib.MODELS_ENV), ("", MASTER, lib.SERVER_ENV),
                            *([("", V01_SUITE, lib.DATA_ENV)] if "text_regression" in rows else []))
    if unset:
        print(unset)
        return 2
    busy = lib.running_gpu_servers()
    if busy:
        print("refusing to start: a model server is already running (the GPU is serialised with the demo gates G0/P0; "
              f"wait for it or ask its owner): {busy}")
        return 2
    free = lib.free_memory_bytes()
    if a.min_free_gb > 0 and (free is None or free < a.min_free_gb * 1024**3):
        print(f"refusing to start: {free} bytes free (vm_stat) < --min-free-gb {a.min_free_gb}")
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    previous = _install_stop_handlers()
    pid_file = out / "run.pid"
    try:
        pid_file.write_text(f"{os.getpid()}\n")              # the pid to watch (or SIGTERM) while it runs
        return _run(a, rows, out, free)
    finally:
        pid_file.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _stop_signal(signum: int, _frame: object) -> None:
    """SIGTERM/SIGINT raise SystemExit, so every `finally` stops its callo and llama-server. One shot: a second
    signal is ignored, so it cannot cut that cleanup short (SIGKILL still works)."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    log.warning("signal %d: stopping, every running server is stopped first", signum)
    raise SystemExit(128 + signum)


def _install_stop_handlers() -> dict[int, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    return {sig: signal.signal(sig, _stop_signal) for sig in (signal.SIGTERM, signal.SIGINT)}


def _run(a: argparse.Namespace, rows: list[str], out: Path, free: int | None) -> int:
    ctx = V02Ctx(out, g0_records=Path(a.g0_records) if a.g0_records else None, hz_seconds=a.hz_seconds)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"rows": {}}
    summary.update(model=TIEL, mmproj=TIEL_MMPROJ, llama_server=MASTER, ctx=CTX, free_memory_at_start=free,
                   started=summary.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    ran: dict[str, dict] = {}

    def record(name: str, res: dict) -> None:
        res = {**res, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "build_info": ctx.build_info,
               "hz_seconds": ctx.hz_seconds, "launch_flags": ctx.flags}
        ran[name] = summary["rows"][name] = res
        summary_path.write_text(json.dumps(summary, indent=1, default=str))

    main_rows = [r for r in rows if r in MAIN_ROWS]
    if main_rows:
        c = None
        try:
            c = _instance("main", out, model=TIEL,
                          flags=["--mmproj", TIEL_MMPROJ, "--s1-slots", "2",
                                 "--", "--cache-ram", "0", "--slot-save-path", _slot_dir(out, "main", "slots")],
                          ctx=ctx)
            summary["build_info"] = c.build_info
            run_main_rows(c, ctx, main_rows, record)
        except Exception as e:  # noqa: BLE001 -- the main callo did not start: its rows fail, the others still run
            log.exception("main instance failed")
            for name in main_rows:
                if name not in ran:
                    record(name, {"pass": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            if c is not None:
                summary["callo_main_log"] = c.lines
                c.finish()
    for name, fn in STANDALONE.items():
        if name in rows:
            log.info("row %s", name)
            try:
                res = fn(ctx, out)
            except Exception as e:  # noqa: BLE001
                log.exception("row %s failed", name)
                res = {"pass": False, "error": f"{type(e).__name__}: {e}"}
            record(name, res)
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary_path.write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: ("INVALID" if v.get("invalid") else v.get("pass")) for k, v in ran.items()}, indent=1))
    return 1 if any(v.get("pass") is False or v.get("error") or v.get("invalid") for v in ran.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
