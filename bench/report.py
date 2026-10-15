"""Render docs/bench/v0.1.md from the acceptance run's summary.json. Numbers come from the file, never by hand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ms(v) -> str:
    return "n/a" if v is None else f"{v / 1000:.2f} s"


def _s(v) -> str:
    return "n/a" if v is None else f"{v:.2f} s"


def _tps(v) -> str:
    return "n/a" if v is None else f"{v:.0f} tok/s"


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _gib(v) -> str:
    return "n/a" if v is None else f"{v / 1024**3:.1f} GiB"


def _dp(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _kv_events(events: list[dict] | None) -> str:
    """Per request: status, slot, prompt_n / cache_n and the prefill time, so a re-prefill after a purge shows."""
    out = []
    for e in events or []:
        t = e.get("timings") or {}
        pre = _ms(t.get("prompt_ms")) if t.get("prompt_ms") is not None else "n/a"
        out.append(f"{e.get('conv')}: HTTP {e.get('status')} slot {e.get('slot')}, prompt_n {t.get('prompt_n')} "
                   f"cache_n {t.get('cache_n')} (prefill {pre})")
    return "; ".join(out) or "n/a"


def _verdict(row: dict | None) -> str:
    """NOT RUN for a missing row and INVALID for one that measured something else: neither is ever PASS/measured."""
    if not row:
        return "NOT RUN"
    if row.get("invalid"):
        return "INVALID: " + str(row["invalid"]).replace("|", "/")
    p = row.get("pass")
    return "PASS" if p is True else "FAIL" if p is False else "measured"


def render(s: dict) -> str:
    r = s["rows"]
    g = lambda name, *keys: _dig(r.get(name, {}), keys)  # noqa: E731
    lines = [
        "# callo v0.1 acceptance bench",
        "",
        f"Run started {s.get('started')}, finished {s.get('finished')}. Model `{s.get('model')}`, callo in launch mode "
        f"(`--ctx 262144 --parallel 4`, so `-np 4 -kvu --no-cache-idle-slots -b 512 -ub 512 --jinja`) on llama.cpp "
        f"`{s.get('master_build_info')}`; the b9890 row on `{g('brew', 'build_info')}`. Every request went through callo.",
        "",
        "| check | pass condition | result | verdict |",
        "|---|---|---|---|",
        f"| drop-in content | identical content / tool_calls / finish_reason via callo and direct (slot 1, seed 42, T 0, cache off), streaming deltas concatenated | plain {g('drop_in', 'plain', 'equal')}, tools {g('drop_in', 'tools', 'equal')}, stream {g('drop_in', 'stream', 'equal')} | {_verdict(r.get('drop_in'))} |",
        f"| agent cache preserved | turns 2/3: `cache_n` >= 0.9 x previous context and `prompt_n` <= 0.2 x it, with a classification between turns, one slot throughout | turns {json.dumps([(t['prompt_n'], t.get('cache_n')) for t in g('agent_cache', 'turns') or []])} | {_verdict(r.get('agent_cache'))} |",
        f"| /v1/responses pinned, /v1/messages not | X-Callo-Slot in 1..3 and that slot holds tokens in /slots; /v1/messages gets `-` | responses slot {g('endpoints', 'responses_slot')} ({g('endpoints', 'responses_slot_held_tokens')} tokens), messages {g('endpoints', 'messages_slot')} | {_verdict(r.get('endpoints'))} |",
        f"| S1 under 32k prefill | {g('s1_prefill_32k', 'pass_condition')}; valid run (>= 90% overlap, prefill proven by prompt_n/cache_n) | p50 under prefill {_ms(g('s1_prefill_32k', 's1_overlap', 'p50_ms'))} (n {g('s1_prefill_32k', 's1_overlap', 'n')}), all-request p50 {_ms(g('s1_prefill_32k', 's1', 'p50_ms'))}, p95 {_ms(g('s1_prefill_32k', 's1', 'p95_ms'))}, S1 ok {g('s1_prefill_32k', 's1', 'n')} / errors {g('s1_prefill_32k', 's1', 'errors')}, overlap {_pct(g('s1_prefill_32k', 'validity', 'overlap_frac'))}, prefill {_tps(g('s1_prefill_32k', 'background', 'prefill_tok_s_mean'))}, footprint max {_gib(g('s1_prefill_32k', 'memory', 'footprint_max'))} | {_verdict(r.get('s1_prefill_32k'))} |",
        f"| S1 under 128k prefill | measured and reported (the earlier study never ran b512 at 128k) | p50 under prefill {_ms(g('s1_prefill_128k', 's1_overlap', 'p50_ms'))} (n {g('s1_prefill_128k', 's1_overlap', 'n')}), p95 {_ms(g('s1_prefill_128k', 's1', 'p95_ms'))}, S1 errors {g('s1_prefill_128k', 's1', 'errors')}, overlap {_pct(g('s1_prefill_128k', 'validity', 'overlap_frac'))}, prefill {_tps(g('s1_prefill_128k', 'background', 'prefill_tok_s_mean'))}, footprint max {_gib(g('s1_prefill_128k', 'memory', 'footprint_max'))} | {_verdict(r.get('s1_prefill_128k'))} |",
        f"| S1 under decode | measured; agent decode hit with 5 overlapping requests in 5 distinct slots (own run, `--parallel 6`), tok/s from requests entirely inside the S1 window | p50 {_ms(g('s1_decode', 's1_overlap', 'p50_ms'))}, S1 n {g('s1_decode', 's1', 'n')}, decode per stream {_tps(g('s1_decode', 'solo_decode_tok_s'))} -> {_tps(g('s1_decode', 'background', 'decode_tok_s_mean_inside_s1'))} over {g('s1_decode', 'background', 'n_inside_s1')} requests (hit {_pct(g('s1_decode', 'decode_hit_frac'))}), slots {g('s1_decode', 'slots_with_s1')} | {_verdict(r.get('s1_decode'))} |",
        f"| S1 fallback | a /v1/messages 32k prefill forced into reserved slot 0 (agent slots held by pinned requests at dispatch); S1 latency while it is in flight, with and without fallback | on: p50 {_ms(g('s1_fallback', 'p50_ms_on'))}, p95 {_ms(g('s1_fallback', 'on', 'p95_ms'))}, n {g('s1_fallback', 'on', 'n_samples')}, slots {g('s1_fallback', 'on', 'sample_slots')}, foreign_in_reserved {g('s1_fallback', 'on', 'foreign_in_reserved')}, s1_fallback {g('s1_fallback', 'on', 's1_fallback_count')}; off: p50 {_ms(g('s1_fallback', 'p50_ms_off'))}, p95 {_ms(g('s1_fallback', 'off', 'p95_ms'))}, n {g('s1_fallback', 'off', 'n_samples')}, slots {g('s1_fallback', 'off', 'sample_slots')}, foreign_in_reserved {g('s1_fallback', 'off', 'foreign_in_reserved')}, s1_fallback {g('s1_fallback', 'off', 's1_fallback_count')} (callo does not read /slots with fallback off; bench-observed foreign {g('s1_fallback', 'off', 'foreign_observed')}); foreign prefill median {_s(g('s1_fallback', 'off', 'foreign_prefill_s_median'))} | {_verdict(r.get('s1_fallback'))} |",
        f"| --cache multi-question | 5-question request vs 5 singles: latency and max abs dp, --no-cache vs --cache | --no-cache {_s(g('cache_multi', 'default', 'multi_s_median'))} vs {_s(g('cache_multi', 'default', 'five_single_s_median'))}; --cache {_s(g('cache_multi', 'cache', 'multi_s_median'))} vs {_s(g('cache_multi', 'cache', 'five_single_s_median'))}; max dp multi-vs-single {_dp(g('cache_multi', 'max_dp_multi_vs_single_default'))} (--no-cache) / {_dp(g('cache_multi', 'max_dp_multi_vs_single_cache'))} (--cache), cache-vs-off {_dp(g('cache_multi', 'max_dp_cache_vs_default_multi'))} | {_verdict(r.get('cache_multi'))} |",
        f"| KV capacity | two 140k conversations on -c 262144 (own fresh callo): observe purge / batch halving / 500 'Context size has been exceeded' | {g('kv_capacity', 'per_conversation_tokens')} tokens per conversation, resident before {g('kv_capacity', 'resident_before')} tokens; {_kv_events(g('kv_capacity', 'events'))} | {_verdict(r.get('kv_capacity'))} |",
        f"| b9890 compatibility | launch mode on Homebrew llama-server: drop-in + S1 short prompts + self-test | drop-in {g('brew', 'drop_in', 'pass')}, S1 ok {g('brew', 's1_short', 'n_ok')}/10, self-test {g('brew', 'selftest', 'pass')} | {_verdict(r.get('brew'))} |",
        f"| self-test + answer-slot check | both pass on Qwen3.5-9B | {' / '.join(g('selftest', 'lines') or [])} | {_verdict(r.get('selftest'))} |",
        "",
        "## KV capacity: what llama-server did",
        "",
        "```",
        *(g("kv_capacity", "server_log_lines") or ["(no matching server log lines)"]),
        "```",
        "",
        "The default stays uncapped unified KV. On master, `--kv-unified-per-slot ⌊c/np⌋` is the option that cannot "
        "overflow, at the cost of capping each conversation at c/np (pass it after `--` in launch mode).",
        "",
        "## Errors",
        "",
        *([f"- {name}: {row['error']}" for name, row in r.items() if row.get("error")] or ["none"]),
        "",
    ]
    return "\n".join(lines)


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Path(a.out).write_text(render(json.loads(Path(a.summary).read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
