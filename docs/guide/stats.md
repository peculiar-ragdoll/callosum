# `/callo/stats`

`GET /callo/stats` returns one JSON object describing the running server. This page lists the top-level fields; the
fields that belong to one feature are described on that feature's page and linked from here. The counters in
`counters` and the router's counters are absent until they are first counted; the `think` and `think_probe` counters
are always present.

```
curl -s localhost:8180/callo/stats
```

## Core fields

- `uptime_s`, `routes`, `instance` (a random id per `callo serve` start).
- `counters`: routing events such as `unrouted_body`, `unrouted_large`, `id_slot_rerouted` and `unrouted_refused` (a
  body too deeply nested to forward without its reserved `id_slot`, refused with 400), plus `upstream_errors`,
  `native_media_refused`, `native_unscanned`, `native_think_refused`, `think_probe_refused` and the `transcription_*`
  counters ([Transcription](transcription.md#what-is-translated)).
- `router`: the slot router's own counters (including `s1_fallback`, `hint_evicted`, `idle_hint`), `lanes`,
  `s1_lane_busy`, `conversations`, `sticky`, and the think-budget fields (see
  [Think budgets](think-budgets.md#observability)).
- `s1`: p50/p95 latency and the low-option-mass count; `readouts` (a monotonic count of question readouts; `s1.n` is
  only the 2,000-wide latency window); `topk_miss` (a monotonic count of options served at the top-k bound); the image
  and `context` request counts, image latency p50/p95, mean `cache_n` / `prompt_n` per kind, `retok_stops`,
  `eog_stops` and `reasks` ([Diagnostics](api.md#diagnostics)); per-kind audio and video keys
  ([Images, audio and video](media.md#callostats));
  `wide_readouts`, `wide_split_passes`, `wide_unread_over` and `wide_uncovered_readouts`
  ([Questions above 26 options](api.md#questions-above-26-options)); `section_waits`, `section_timeouts` and
  `draining` ([Think budgets](think-budgets.md#observability)).
- `probe`: the startup probe's result, with `residue` (the slots the startup probe filled and could not erase) and
  `erase` (`ok`, `unsupported`, `failed`, or `null` when not attempted); see
  [Required llama-server configuration](serving.md#required-llama-server-configuration).
- `slots`: a fresh `/slots` snapshot.
- `kv`: `"unified"`, `"per_slot"`, or `null` when attached ([Serving](serving.md#per-slot-kv---no-kvu)).
- `temperature` and `temperature_source` (`flag` | `profile` | `default`).

## Model and provenance

- `model_key` (see [Model identity](position-bias.md#model-identity); `null`, with the reason in `model_key_error`,
  when it cannot be computed) and `model_file` (the GGUF's file name).
- `provenance`: what a position-bias profile records. `prompt_sha256`, the prompt contract's hash;
  `chat_template_sha256`; `build_info`, the llama.cpp build; `mmproj`, the projector's file name; `user_delimiter`;
  `cache_prompt`, whether `--cache` is on; `cache_delimiter`, present and `true` only when `--cache` readouts send the
  user-turn delimiter (`--cache` on and a delimiter in use; a profile measured without it lacks it, so a `--cache` one
  warns); and `server_args`, launch mode's llama-server flags after `--`, `null` when attached. The whole block is
  `null` for a native decision model.
- `user_delimiter`, `media_marker`, `cache_prompt` (per kind: plain, image, context, audio, video, mixed); see
  [Caching](api.md#caching-and-the-user-turn-delimiter).
- `host`, `model_bytes`, `mmproj_bytes`, `n_ctx`, `total_slots`, `upstream_local`: see
  [Think budgets](think-budgets.md#observability).
- `wide_coverage`: how many of Callo's two-letter labels this vocabulary carries as single tokens, and
  `wide_readable`: how many it reads, those past the coverage in two passes
  ([Questions above 26 options](api.md#questions-above-26-options)).

## Features

- `vision`, `audio`, `video` and `self_test`: see [Images, audio and video](media.md#callostats).
- `position_bias`: `null`, or the loaded profile's `profile`, `sha256`, `entries`, `kinds` and `state`, with an
  `applied` count and `skipped` counts by reason: `no_entry`, `noncanonical`, `score`, `single_option`,
  `kind_not_calibrated`, `deliberate`, and `wide` for a question above 26 options. See
  [Position bias](position-bias.md).
- `think`, `think.deadline`, `think.converge` and `think_probe`: see [Think budgets](think-budgets.md#observability),
  [Observability under a deadline](think-budgets.md#observability-under-a-deadline) and [Observability with early
  exit](think-budgets.md#observability-with-early-exit).
- `think.converge.gate` (experimental): `default`, `asked`, `skipped`, `errors` and `read_ms` (`p50`, `p95`): see
  [Early-exit gate](think-budgets.md#early-exit-gate-experimental).
- `asr_preset` and `transcription`: see [Transcription](transcription.md#is-it-on).
