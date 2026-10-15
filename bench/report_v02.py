"""Render docs/bench/v0.2.md from the v0.2 acceptance run's summary.json. Numbers come from the file, never by hand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.report import _dig, _verdict


def _ms(v) -> str:
    return "n/a" if v is None else f"{v:.0f} ms"


def _f(v, fmt: str = ".3f") -> str:
    return "n/a" if v is None else format(v, fmt)


def _sci(v) -> str:
    return "n/a" if v is None else f"{v:.2e}"


def _gib(v) -> str:
    return "n/a" if v is None else f"{v / 1024**3:.1f} GiB"


def _flags(flags) -> str:
    """The row's callo launch flags, absolute paths abbreviated to their last two components so the table stays
    readable. None (an older summary.json that never stamped them) is `n/a`; an empty list is the base launch."""
    if flags is None:
        return "n/a"
    if not flags:
        return "base launch only"

    def short(a: str) -> str:
        if a.startswith("/"):
            return "…/" + "/".join(Path(a.rstrip("/")).parts[-2:]) + ("/" if a.endswith("/") else "")
        return a

    return "`" + " ".join(short(a) for a in flags) + "`"


def render(s: dict) -> str:
    r = s["rows"]

    def g(name, *keys):
        return _dig(r.get(name, {}), keys)

    sizes = g("readout", "per_size") or {}
    per_size = "; ".join(f"{k}²: {v['correct']}/{v['n']}, mean {v['mean_p_gold']:.3f}, min {v['min_p_gold']:.3f}"
                         for k, v in sizes.items()) or "n/a"

    def kv(cfg):
        d = g("kv_sensitivity", cfg) or {}
        return (f"median \\|Δp\\| {_f(d.get('median_dp'), '.4f')}, max {_f(d.get('max_dp'), '.4f')}, top-2 log-odds "
                f"median {_f(d.get('median_logodds_shift'))}/max {_f(d.get('max_logodds_shift'))} nats, argmax "
                f"{d.get('argmax_agree')}/{d.get('n')}")

    g0_max, g0_tol = g("g0_parity", "max_dp"), g("g0_parity", "dp_tol")
    g0_margin = (f", about {g0_tol / g0_max:.0f}x tighter than the {_sci(g0_tol)} gate"
                 if g0_max and g0_tol else "")

    lines = [
        "# callo v0.2 acceptance bench",
        "",
        f"Run started {s.get('started')}, finished {s.get('finished')}. TielCoder `{s.get('model')}` with mmproj "
        f"`{s.get('mmproj')}`, callo in launch mode (`--ctx {s.get('ctx')} --parallel 4`, `--s1-slots 2` unless a row "
        f"says otherwise) on llama.cpp `{s.get('build_info')}`. One llama-server at a time; every System One request "
        "went through callo.",
        "",
        "| check | pass condition | result | verdict |",
        "|---|---|---|---|",
        f"| vision readout in the demo layout | the spike's 36 maze frames (12 each at 224², 336², 448²), ~180 tokens of "
        f"standing orders as `context`: 36/36 and mean P(gold) >= 0.95 | {g('readout', 'correct')}/{g('readout', 'n')}, "
        f"mean P(gold) {_f(g('readout', 'mean_p_gold'))}, min {_f(g('readout', 'min_p_gold'))}; {per_size} "
        f"| {_verdict(r.get('readout'))} |",
        f"| 2 Hz while System 2 streams | S1 224² on lane `s1` (slot 0), 2 Hz; S2 streaming on slot 2 or 3: S1 median "
        f"<= 411 ms, >= 1.95 Hz, S2 >= 27 tok/s | S1 median {_ms(g('hz2_stream', 's1', 'median_ms'))}, p90 "
        f"{_ms(g('hz2_stream', 's1', 'p90_ms'))}, {_f(g('hz2_stream', 's1', 'achieved_hz'), '.2f')} Hz, S1 slots "
        f"{g('hz2_stream', 's1', 'slots')}; S2 {_f(g('hz2_stream', 's2_tok_s'), '.1f')} tok/s on slots "
        f"{g('hz2_stream', 's2_slots')} | {_verdict(r.get('hz2_stream'))} |",
        f"| 2 Hz while System 2 prefills | the same, S2 restarting every 11 s with a 224² board + ~430 tokens: ticks "
        f"overlapping an S2 prompt phase vs the others, reported | overlapping: {g('hz2_prefill', 'ticks_overlapping_s2_prompt')} "
        f"ticks, median {_ms(g('hz2_prefill', 'median_ms_overlapping_s2_prompt'))}, max "
        f"{_ms(g('hz2_prefill', 'max_ms_overlapping_s2_prompt'))}; others: {g('hz2_prefill', 'ticks_other')} ticks, "
        f"median {_ms(g('hz2_prefill', 'median_ms_other'))}; {g('hz2_prefill', 's2_prompt_phases')} S2 prompt phases "
        f"| {_verdict(r.get('hz2_prefill'))} |",
        f"| G0 parity (decision-equivalence) | callo on G0's own server state (pinned delimiter `<|im_start|>user`, "
        f"slot 0, every other slot empty, `--cache-ram 0`): 106 G0 first-decision frames, prompt byte-identical (media "
        f"marker normalised), argmax agrees on 106/106 and max \\|Δp\\| < 1e-3 (a readout-method tolerance, see below) "
        f"| prompts equal {g('g0_parity', 'prompt_equal')}/{g('g0_parity', 'n')}, argmax agrees "
        f"{g('g0_parity', 'agree')}/{g('g0_parity', 'n')}, max \\|Δp\\| {_sci(g('g0_parity', 'max_dp'))} on pair "
        f"{g('g0_parity', 'max_dp_pair')} (gate {_sci(g('g0_parity', 'dp_tol'))}) | {_verdict(r.get('g0_parity'))} |",
        f"| KV-layout sensitivity (text) | 40 text prompts read in s0 (every other slot empty) vs s2 (another slot "
        f"holds ~300 tokens), default `-kvu` vs `--no-kvu` (expected bit-exact on text); measured | -kvu: {kv('kvu')}; "
        f"--no-kvu: {kv('no_kvu')} | {_verdict(r.get('kv_sensitivity'))} |",
        f"| walk pass reuse | image request A, then A plus its top token's detokenized text (a real walk pass), one "
        f"slot: pass 2 `cache_n` == A's tokens_evaluated, `prompt_n` == 1, `prompt_ms` <= 40; an identical re-send "
        f"reported (1 on pure attention, 4 on hybrids) | A {g('walk_reuse', 'a', 'tokens_evaluated')} tokens; pass 2: "
        f"cache_n {g('walk_reuse', 'pass2', 'cache_n')}, prompt_n {g('walk_reuse', 'pass2', 'prompt_n')}, prompt_ms "
        f"{_f(g('walk_reuse', 'pass2', 'prompt_ms'), '.1f')}; identical re-send: prompt_n "
        f"{g('walk_reuse', 'identical_resend', 'prompt_n')}, prompt_ms "
        f"{_f(g('walk_reuse', 'identical_resend', 'prompt_ms'), '.1f')} | {_verdict(r.get('walk_reuse'))} |",
        f"| system-prefix reuse | fresh frame each tick: tick 2's `cache_n` >= the rendered system turn's tokens | "
        f"system turn {g('prefix_reuse', 'system_turn_tokens')} tokens; cache_n per tick "
        f"{[t.get('cache_n') for t in g('prefix_reuse', 'ticks') or []]} | {_verdict(r.get('prefix_reuse'))} |",
        f"| `context`-prefix reuse, text only | same ~600-token context, different states, one lane: the second's "
        f"`cache_n` >= the system turn's tokens; max \\|Δp\\| cold vs warm reported | context "
        f"{g('context_reuse', 'context_tokens')} tokens, system turn {g('context_reuse', 'system_turn_tokens')}; "
        f"cache_n {g('context_reuse', 'passes', 'a_warm', 'cache_n')}; max \\|Δp\\| "
        f"{_f(g('context_reuse', 'max_dp_cold_vs_warm'), '.4f')} | {_verdict(r.get('context_reuse'))} |",
        f"| demo slot layout | `--s1-slots 3`, S2 on slot 3, lanes warmed in demo order, a text and a view instinct "
        f"alternating every 10 s: reported | lanes {g('demo_slots', 'lanes')}; S1 median "
        f"{_ms(g('demo_slots', 's1', 'median_ms'))}, {_f(g('demo_slots', 's1', 'achieved_hz'), '.2f')} Hz; S2 "
        f"{_f(g('demo_slots', 's2_tok_s'), '.1f')} tok/s on {g('demo_slots', 's2_slots')}; instinct cache_n "
        f"{g('demo_slots', 'instinct_cache_n')} | {_verdict(r.get('demo_slots'))} |",
        f"| cache drift | the same frame cold and warm (after 20 other frames): max \\|Δp\\| reported | max \\|Δp\\| "
        f"{_f(g('cache_drift', 'max_dp'), '.4f')} (cold cache_n {g('cache_drift', 'cold', 'cache_n')}"
        f"{'' if g('cache_drift', 'cold_was_cold') is not False else ', so NOT cold'}, warm "
        f"{g('cache_drift', 'warm', 'cache_n')}) | {_verdict(r.get('cache_drift'))} |",
        f"| no vision | launched without `--mmproj`: an image request gets 501, the same body without images 200 | "
        f"{g('no_vision', 'with_images_status')} / {g('no_vision', 'without_images_status')} "
        f"| {_verdict(r.get('no_vision'))} |",
        f"| text-path regression | Qwen3.5-9B Q8_0, no mmproj: v0.1's \"drop-in content\" and \"self-test + "
        f"answer-slot check\" | drop-in {g('text_regression', 'drop_in', 'pass')}, self-test "
        f"{g('text_regression', 'selftest', 'pass')} | {_verdict(r.get('text_regression'))} |",
        f"| self-tests | text and vision self-tests pass on TielCoder | {' / '.join(g('selftests', 'lines') or [])} "
        f"| {_verdict(r.get('selftests'))} |",
        f"| vision self-test on Qwen3.5-9B + mmproj-F16 | reported | "
        f"{' / '.join(g('selftest_9b_vision', 'lines') or [])} | {_verdict(r.get('selftest_9b_vision'))} |",
        f"| footprint | macOS `footprint` max over the 2 Hz row, reported | "
        f"{_gib(g('hz2_stream', 'footprint_max'))} | {'n/a' if g('hz2_stream', 'footprint_max') is None else 'measured'} |",
        "",
        "## Why G0 parity is a decision-equivalence check, not a bit-exact one",
        "",
        "G0 records the demo's `letter_mass` readout (normalise over the letter tokens) and callo serves the full "
        "`bounded_readout` (the marginalisation). Those are two different formulas computed from the same "
        "logprobs, so they differ by ~1e-5 by construction (parity diagnostic test 3 = 1.4e-5) even on identical "
        "server state; it is not a bug and no state change removes it. The gate therefore checks that callo renders "
        "the same prompt and reaches the same System One decision as the reference readout: argmax agreement 106/106 "
        "(any single flip fails) and max \\|Δp\\| < 1e-3, a readout-method tolerance far below any realistic decision "
        f"threshold. The observed max here is {_sci(g0_max)}{g0_margin}.",
        "",
        "## Row provenance",
        "",
        "Each row as it stands in the merged summary.json: when it finished, on which llama.cpp build, with which "
        "paced-row length, and the callo launch flags of the instance it ran on (so the one header above never "
        "stands in for a flag a row did not run under).",
        "",
        "| row | finished (UTC) | llama.cpp build | hz_seconds | callo launch flags |",
        "|---|---|---|---|---|",
        *[f"| {name} | {row.get('finished_at', 'n/a')} | {row.get('build_info') or 'n/a'} | "
          f"{row.get('hz_seconds', 'n/a')} | {_flags(row.get('launch_flags'))} |" for name, row in r.items()],
        "",
        "## Errors",
        "",
        *([f"- {name}: {row['error']}" for name, row in r.items() if row.get("error")] or ["none"]),
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Path(a.out).write_text(render(json.loads(Path(a.summary).read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
