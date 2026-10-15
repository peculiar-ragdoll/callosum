# Serving: launch mode, attach mode and the proxy

`callo serve` runs in one of two modes:

- **Launch mode** (`--model M.gguf`): Callo starts llama-server for you, with the configuration below.
- **Attach mode** (`--upstream http://127.0.0.1:8199`): Callo fronts a llama-server you already run, which must be
  started with the [required configuration](#required-llama-server-configuration).

Callo listens on `http://127.0.0.1:8180` by default (not 8080, llama-server's own default). Point the agent at it.

Contents: [the proxy](#the-proxy) · [required llama-server configuration](#required-llama-server-configuration) ·
[launch mode](#launch-mode) · [`--no-kvu`](#per-slot-kv---no-kvu) · [startup](#startup) ·
[`callo serve` flags](#callo-serve-flags)

## The proxy

Every request is relayed to the upstream (SSE streaming included). Only `POST` bodies on the pinnable endpoints are
touched: `/completion`, `/completions`, `/v1/completions`, `/chat/completions`, `/v1/chat/completions`, `/infill`,
`/v1/responses` and `/responses`. A JSON-object body there is re-serialised with `id_slot` set to the chosen slot, the
only key that changes (`Content-Length` is recomputed; a client's own `id_slot` is honoured unless it names a reserved
slot, in which case it is re-routed). When agent pinning is off the body is forwarded as it came, except that a client
`id_slot` naming a reserved slot is removed. Every other body passes through byte-for-byte: other endpoints, non-JSON
or `Content-Encoding`d bodies, and bodies over 64 MiB. The slot is chosen like llama-server chooses itself (longest
common prefix ≥ 10%, then least-recently-used), so the agent's prompt cache keeps working. Responses carry
`X-Callo-Slot`.

One more body is touched: while translation is on, a well-formed `POST /v1/audio/transcriptions` (or
`/audio/transcriptions`) is rewritten into a `/v1/chat/completions` (or `/chat/completions`) request and pinned (see
[Transcription](transcription.md)). Chat bodies with media are routed on digests of their payloads (see
[Chat completions with media](media.md#chat-completions-with-media)).

Classification goes to the reserved slots `0..k-1` (`--s1-slots k`, default 1) and agent traffic to the others. When
a reserved slot holds traffic Callo did not send, the S1 fallback moves classification to the lowest-id idle slot (see
[Limitations](limitations.md#the-s1-fallback-helps-against-a-foreign-decode-not-a-foreign-prefill) for what it cannot
fix).

## Required llama-server configuration

Pinning agent traffic is only safe with a specific KV configuration. Launch mode sets it; in attach mode you must
start llama-server with:

```
mkdir -p ~/.cache/callo/slots
llama-server -m X.gguf -np 4 -kvu --no-cache-idle-slots -b 512 -ub 512 --jinja --slots \
  --slot-save-path ~/.cache/callo/slots
```

- `--no-cache-idle-slots`: the default saves idle slots to the host cache and, with unified KV, **clears**
  them on every task dispatch. A pinned request skips the restore path, so an agent pinned to a slot would
  re-prefill its whole context after any interleaved classification. Callo probes for this at startup (two
  pinned completions of a 16-token prompt with `n_predict` 1, on the last slot and on slot 0, then `GET /slots`) and,
  if clearing is detected, logs an error and **disables agent pinning** (classification stays pinned).
- `-kvu`: an explicit `-np N` turns unified KV off, giving each slot `n_ctx/N`; a 32k prefill then fails. The
  cost of `-kvu` is that, on MoE models, a readout can shift with what the other slots hold (see
  [Readouts and unified KV](limitations.md#readouts-and-unified-kv)); if you need readouts that do not, start
  llama-server without `-kvu` and with `-c` set to the per-slot context times `-np`.
- `-b 512 -ub 512`: smaller batches let classification get a turn sooner.
- `--slots`: without it Callo cannot run the probe and keeps agent pinning off.
- `--slot-save-path DIR` (the directory must exist): lets the startup probe erase the two slots it fills
  (`POST /slots/{id}?action=erase`; llama-server answers 501 without the flag). Without it the probe's 16 tokens stay
  in the slots it used (the last slot and slot 0), `/callo/stats` lists them under `probe.residue` with `probe.erase`
  `"unsupported"`, Callo logs a warning, and `callo calibrate --bias` refuses on that server: the last slot holds
  tokens, and Callo treats an attached server as unified KV because it cannot see its flags.
  The flag also enables llama-server's slot save and restore actions (`POST /slots/{id}?action=save|restore`), which
  Callo passes through like any other request: any client that can reach Callo can then write a slot's whole KV
  cache into `DIR`, which can be several GB per save at long contexts. Put `DIR` on a volume with room to spare;
  launch mode uses `~/.cache/callo/slots`, usually the boot volume, and `-- --slot-save-path DIR` overrides it (your
  flags come last and win; Callo creates only its own default directory).
- `--api-key`: pass the key to Callo with `--upstream-api-key` or `LLAMA_API_KEY`; client `Authorization`
  headers pass through untouched.

## Launch mode

Launch mode runs `llama-server -m X.gguf --port 0 [-c N] -np 4 -kvu --no-cache-idle-slots -b 512 -ub 512 --jinja
--slot-save-path ~/.cache/callo/slots [--mmproj M.gguf]` plus anything after `--`; your flags come last and win. The
child's pid and log are under `~/.cache/callo/`. The startup probe erases the two slots it used, so a fresh server
starts with empty agent slots. In attach mode the probe erases them too when llama-server has `--slot-save-path`; it
only erases slots it filled itself. The residue warning names the fix for your setup: in attach mode, a 501 means
restart llama-server with `--slot-save-path DIR`, and anything else means the erase itself failed (restart
llama-server: a fresh server holds no tokens); in launch mode, where Callo already passed the flag, a 501 means a
passthrough flag after `--` overrode it, and anything else means the erase itself failed (restart Callo).

### Per-slot KV (`--no-kvu`)

`--no-kvu` (launch mode only) omits `-kvu`, so every slot has its own KV; on text prompts that removed the dependence
of a readout on what other slots hold (measured conditions and limits are under
[Readouts and unified KV](limitations.md#readouts-and-unified-kv)). `--ctx N` then means the context **per slot**:
Callo passes `-c N*parallel` (`--ctx 32768 --parallel 4` launches `-c 131072`), and it exits 2 if `--ctx` is missing
or below 1. KV memory scales with `-c`, so `--no-kvu` reserves `parallel` times the KV of one slot, whether or not the
other slots are used. A conversation can use at most `N` tokens, since a slot cannot borrow from its neighbours. At
startup Callo logs which mode it is in, and `/callo/stats` reports it as `"kv": "unified"` or `"per_slot"` (`null`
when attached, where Callo cannot know). In attach mode `--no-kvu` exits 2: Callo does not start llama-server there,
so start it without `-kvu` yourself.

## Startup

Callo listens on its own port (default 8180) **before** it launches the model, so a busy port fails at once instead
of after a model load that can take minutes. Until startup completes (launch, discovery, probe, self-test), every
request, `/health` included, gets llama-server's own loading reply, `503
{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`, with `X-Callo-Slot: -`. Clients and
health checks that already wait on llama-server's 503 need no change.

Media detection from `/props` (vision, audio and video) runs right after discovery, before the GGUF vocabulary load
and the idle-clear probe; if `--mmproj` was given and the upstream reports no media modality (vision, audio or video),
Callo stops the child and exits 1 there. After the probe come the delimiter, a vision self-test when vision is on,
then the text self-test. None of the last three runs for a native decision model. The vision self-test makes three
calls: a built-in 128×128 two-tone PNG ("Which half of the image is white?", expected `right`), the same request
again (its `cache_n` / `prompt_n` show the cache), and the horizontally mirrored image as a control (expected
`left`). It passes only when both images are answered correctly with an option mass of at least 0.5 each, so an
image-blind model that always gives the same answer fails it. A failing self-test warns loudly and Callo still
starts, with vision on. An audio and a video self-test follow the text one, for each modality the upstream reports
(see [Images, audio and video](media.md#startup-self-tests)); on a model that can think, the think self-test runs
too (see [Think budgets](think-budgets.md#startup-cost)).

Exit codes are in [Exit codes](exit-codes.md).

## `callo serve` flags

| flag | default | meaning |
|---|---|---|
| `--model GGUF` | | launch mode: start llama-server on this GGUF (exclusive with `--upstream`) |
| `--upstream URL` | | attach mode: a running llama-server, plain `http://host[:port]` |
| `--host`, `--port` | `127.0.0.1`, `8180` | where Callo listens; a non-loopback `--host` prints a warning |
| `--s1-slots k` | 1 | reserved classification slots `0..k-1` (k below the total slots) |
| `--temperature T` | the think profile's T_0, else 1.0 | applied to the log-marginals at budget 0 (see [Calibration](calibration.md)) |
| `--cache` / `--no-cache` | on | prompt cache with the user-turn delimiter on plain classification passes (see [Caching](api.md#caching-and-the-user-turn-delimiter)) |
| `--s1-timeout S` | 900 | seconds per upstream call for classification |
| `--asr-home-idle S` | 600 | idle time after which a hint-less transcription may home on a bound agent slot (see [Transcription](transcription.md#routing)) |
| `--s1-fallback on\|off` | on | for testing: `off` keeps classification on its reserved slot even when a foreign task holds it |
| `--gguf PATH` | | the served GGUF, when `/props.model_path` is not readable from Callo's side |
| `--upstream-api-key KEY` | `$LLAMA_API_KEY` | key for Callo's own upstream calls |
| `--ctx N` | llama-server's own | launch mode: `-c` (per slot under `--no-kvu`) |
| `--parallel N` | 4 | launch mode: `-np` |
| `--no-kvu` | off | launch mode: per-slot KV (see [above](#per-slot-kv---no-kvu)) |
| `--llama-server BIN` | `llama-server` | launch mode: the binary to run |
| `--mmproj GGUF` | | launch mode: the multimodal projector, vision and/or audio |
| `--user-delimiter STR` | derived from the template | pin the user-turn delimiter (still validated) |
| `--position-bias PROFILE` | off | debias eligible readouts (see [Position bias](position-bias.md)) |
| `--think-budget N` | 0 | default think budget (see [Think budgets](think-budgets.md)) |
| `--think-max N` | 4096 | largest budget a request may ask for; hard cap 32768 |
| `--think-profile PROFILE` | | a think profile from `callo calibrate --think-budgets` |
| `--think-sampling model\|fixed\|SPEC` | `model` | the trace sampler without a profile |
| `--allow-think-probe` | off | answer `POST /callo/think-probe` from non-loopback peers too |
| `--think-deadline MS` | 0 | default think deadline, for every client (see [Think deadlines](think-budgets.md#think-deadlines)) |
| `--think-deadline-reserve MS` | learned | fix the readout reserve of a deadline trace |
| `--think-converge Z` | 0 | default early-exit target, for every client (see [Early exit](think-budgets.md#early-exit)) |
| `--think-converge-gate` | off | experimental: gate early exit with one budget-0 question before thinking, for every client (see [Early-exit gate](think-budgets.md#early-exit-gate-experimental)) |
| `--request-log FILE` | | one JSON line per `/v1/systemone` request (see [The request log](think-budgets.md#the-request-log)) |
| `-- ARGS` | | launch mode: passed to llama-server after Callo's own flags (last wins) |
