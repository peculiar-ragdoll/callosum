# Limitations

The [README](../../README.md#limitations) lists the key caveats; this page has every one, with the measurements behind
it.

## Routing and the proxy

### Not pinned

`/v1/messages` (Anthropic API, e.g. Claude Code), embeddings and rerank, the child completions of `n > 1` (the parent
can be pinned), and `/v1/audio/transcriptions` requests that are streaming, use a non-`json` format, exceed 64 MiB,
or that Callo cannot parse cleanly (counted in `/callo/stats`; see [Transcription](transcription.md)). Every
transcription also passes through unpinned when translation is off for the whole process: no audio, agent pinning
off, or a failed render check. The startup line (a WARNING for the render check) and `/callo/stats.transcription` say
which. llama-server's Anthropic converter copies only these sampling keys (`temperature`, `top_p`, `top_k`, `stream`,
`chat_template_kwargs`; `tools/server/server-chat.cpp:593`), and `id_slot` is not among the keys it maps; the others
never read it, so those requests may land on a reserved slot. The S1 fallback then moves classification to the
lowest-id idle slot, agent slots included. Converting Anthropic → OpenAI in Callo is out of scope.

### The S1 fallback helps against a foreign decode, not a foreign prefill

When a reserved slot is busy with a request Callo did not send, Callo moves the classification to the
lowest-id idle slot (the fallback). llama-server fills each batch in slot-id order, so with a foreign 32k prefill in slot 0 a classification moved to
slot 1 starves behind it just as one left on slot 0 waits for it: measured p50 39.2 s with fallback vs 41.0 s without
(the prefill took 42.1 s).

### Classification shares the GPU

Under continuous classification load (one request at a time, ~3/s), 5 concurrent agent decode streams dropped from
30.7 to 9.1 tok/s each (−70.5%) while classification p50 stayed at 0.32 s.

### HTTP/1.0 towards clients

No keep-alive; every response is close-delimited, so a client that half-closes its socket (`SHUT_WR`) after sending
the request gets no response. Chunked request bodies get 411. Requests carrying `X-Conversation-Id` (resumable
streams) are not cancelled on disconnect; use `DELETE /v1/stream`.

### Cancelling a non-streaming request can lag

Callo shuts the upstream socket as soon as a client disconnects, but llama-server notices a closed NON-streaming
request only after 1 s with no result traffic anywhere on the server (`server-queue.cpp:450-470` at 4ebdf2c restarts
that wait on every result), so on a busy server the abort can be delayed for as long as the traffic lasts. Streaming
requests are cancelled on their next token.

### KV capacity

With the default uncapped unified KV (`-kvu --no-cache-idle-slots`), two long conversations whose KV together exceeds
`-c` evict each other. Measured with two 141.8k-token conversations on `-c 262144`: B's prefill made llama-server
purge idle slots lowest id first, without saving (slot 0 with 93 tokens, then conversation A's slot 1 with 141,812
tokens), and retry the batch smaller. A's next turn then purged B's slot 2 (141,812 tokens) and re-prefilled all
141,806 tokens (`cache_n` 0, 300 s). Every request still returned 200; nothing hit "Context size has been exceeded".
On master, `--kv-unified-per-slot ⌊c/np⌋` (pass it after `--` in launch mode) cannot overflow, at the cost of capping
each conversation at c/np.

### No auth, no TLS, no Windows

Binding a non-loopback `--host` prints a warning. `--slot-save-path` exposes llama-server's slot save and restore
actions to every client of Callo (see [Required llama-server configuration](serving.md#required-llama-server-configuration)).

Callo runs on Linux and macOS. On Windows every `callo` command prints one line saying so and exits 2. Windows
support is coming; for now, run Callo under WSL2.

### Docker containers need an init

Run the container with `docker run --init` (or another init that reaps orphans, such as tini). If Callo is killed,
its llama-server is re-parented to the container's PID 1; without such an init, once that llama-server exits it stays
a zombie, which Callo's liveness check still reports as alive. Every later `callo serve` then warns about it, its pid
file is never removed, and the GPU check before a menu run refuses until you delete that pid file.

## Readouts

### Readouts and unified KV

On the MoE model tested (Tiel-Coder 35B-A3B), a readout taken while other slots held tokens differed from the same
readout on an idle server by up to 0.34 in probability (the maximum over 40 prompts, not a bound); the dense hybrid
tested (Dirk 27B) showed only floating-point noise, at most 0.001. Without `-kvu` (`--no-kvu`), text readouts were
bit-exact whatever the other slots held, under the conditions below. Readouts with the prompt cache warm, and image
readouts, were not measured under `--no-kvu`.

Measured on 40 fixed text classification prompts (four families of 10, 4 to 5 options each), read through Callo's
engine against a llama-server started with `-c 32768 -np 4` (8192 per slot in the no-`-kvu` run; Callo's own
`--no-kvu` reserves the full `--ctx` per slot instead) plus `--cache-ram 0`, so no prompt-cache carry-over, and with
all slots erased before each readout. Each cell compares the readout against the same prompt with every other slot
empty: median / max |Δp| over options, then argmax agreement out of 40. "0" is bit-exact.

| model | KV | another slot holds 16 tokens | another slot holds 300 tokens |
|---|---|---|---|
| Tiel-Coder 35B-A3B (MoE) | `-kvu` | 0.0006 / 0.202 / 40 | 0.0003 / 0.336 / 39 |
| Tiel-Coder 35B-A3B (MoE) | `--no-kvu` | 0 / 0 / 40 | 0 / 0 / 40 |
| Dirk 27B (dense hybrid) | `-kvu` | 0.0000 / 0.0011 / 40 | 0.0000 / 0.0008 / 40 |
| Dirk 27B (dense hybrid) | `--no-kvu` | 0 / 0 / 40 | 0 / 0 / 40 |

Running the readout in slot 1 instead of slot 0, or with a different user-turn delimiter, was bit-exact in all four
configurations. Most prompts are confident, so |Δp| understates the shift on them; on the top-2 log-odds the `-kvu`
shift on Tiel-Coder was 0.17 to 0.23 nats at the median and up to 1.7 nats at the maximum, against 0.002 and 0.013
nats on Dirk. The cause was not isolated; expert-routing changes are a plausible one for the MoE. Image readouts were
measured only on Tiel-Coder with `-kvu` and the prompt cache off, over 17 image pairs, where they moved by up to 0.04
in probability with the slot index and up to 0.23 with the delimiter's batch split; whether `--no-kvu` removes that
was not tested, so pin the same delimiter and slot when comparing image readouts.

Default launch mode keeps `-kvu` because without it each slot gets `n_ctx/np` and a long agent prefill fails. To
remove the cross-slot dependence measured above, launch with `--no-kvu --ctx <per-slot context>`. In attach mode,
start llama-server without `-kvu` and with `-c` set to the per-slot context times `-np`.

### Prompt cache for classification

On by default (`cache_prompt: true` with the user-turn delimiter; −24 % per readout on a 35B-A3B MoE, a warm readout
identical to a cold one; see [Caching](api.md#caching-and-the-user-turn-delimiter)). `--no-cache` restores the
bit-stable `cache_prompt: false`. On hybrid (Gated-DeltaNet) models the reuse begins at the user turn: without the
delimiter llama.cpp checkpoints only near the end of the prompt, so a prompt that differs earlier is re-prefilled
whole; with it, what precedes the user turn is reused.

### Special-token escaping covers only `<|name|>` spellings

Callo rewrites `<|im_start|>`-style text in the state, instructions and options to `<¦im_start¦>` so it cannot become
a control token, which matches llama.cpp's native decision endpoint; other model-specific special-token spellings are
not escaped.

### Images, audio and video

Data URLs (and, in audio and video state parts, raw base64) only: no `http(s)://`, no `file://`. Media encode runs in
llama-server's one server loop and blocks every slot while it runs, so a large image, a long clip or any video delays
every other request, agents included. Media and `context` requests always use the prompt cache, so they are not
bit-stable. A hybrid model needs the user-turn delimiter to reuse the system turn. `context`, `audio`, `videos`, lanes
and `X-Callo-Timings` are Callo extensions, not System One. More in
[Images, audio and video](media.md#cost-and-limits).

### Native decision models

If the GGUF carries `<arch>.decision.type` (OpenJev, Lev, Kev, Laya), llama.cpp serves `/v1/systemone` itself. Callo
forwards every request there unchanged, except that it answers 422 to audio or video (native takes images only; see
[Images, audio and video](media.md)) and sends nothing upstream. Native errors are 400 `{"error": …}`; Callo's are
422 `{"detail": …}`. Native requests ignore `id_slot`, and launch mode's `-b 512` can make native requests with large
option lists fail with 400.

## Calibration

### Position-bias profile

Position bias depends on the questions. In Callo's measurements, a prior measured on the built-in set did not carry
over to other questions, and PriDe's log-space prior, the first estimator tried, made held-out readouts worse; the
mean-probability prior measured on questions from the served distribution passed when the same readouts were
re-scored offline. So a profile is fitted only on your `--data`, and it is only as good as the match between that
file and what you serve. It corrects only the readout kinds that file contained: a file of plain records (no
`context`, no media) corrects plain readouts only, not `context`, image, audio, video or mixed readouts. One prior per
option count averages your sources: if they lean opposite ways at the same n, the prior corrects their average only.
The correction is deliberately partial: dividing by the mean probability per position pulls the readouts toward
position-free without forcing them to it, and the gains measured offline were small (rotation consistency up by 0 to
1.3 points, accuracy within ±0.3 points). The estimator was chosen after seeing those readouts, so that was not a
validation; a confirmatory run on fresh questions, judged against pass criteria fixed before it ran, then confirmed
it (see [Measurements](measurements.md#position-bias)). Non-canonical questions are not corrected: under Callo's
lettering, letter and position coincide only on canonical questions, and correcting by letter on the others is
future work.

`model_key` samples the first and last 64 KiB of each tensor, so an edit confined to the interior of a large tensor
would not change it, and it assumes the trailing `nextn_predict_layers` blocks are the MTP head (true for
`qwen35`/`qwen35moe`). In attach mode Callo cannot see llama-server's flags, so a profile does not notice a changed
`-ctk`/`-ctv`, `--lora`, `--override-kv` or `--rope-*` there; in launch mode the flags after `--` are recorded and
compared.

A GGUF that uses a quantization type newer than your gguf-py still serves and calibrates: Callo reads its header
without that type's size. Once an upgraded gguf-py knows the type, the file's `model_key` changes, and think and
position-bias profiles measured on it must be recalibrated.

### Think budgets

Deliberate readouts are text only, never position-debiased, and one sample per question (no self-consistency). The
budget is the server default or the request's: no automatic routing or per-question adaptation. Only
`<...think...>`-block thinking formats are supported; a model that thinks in another format (Gemma 4's channel tokens)
is served at budget 0 only. With `--s1-slots 1`, fast readouts wait behind a deliberate generation in FIFO order,
bounded by `--s1-timeout`; with long budgets that bound can turn into 503s for fast traffic (use `--s1-slots 2` or
more). Under `-kvu` a resident trace shifts concurrent readouts on other slots on MoE models, as any other slot's tokens
do. Without a think profile (`callo calibrate --think-budgets` writes one) deliberate readouts are uncalibrated (T = 1).
Whether a cut trace equals a shorter generation under speculative/MTP decoding depends on how llama.cpp's verification
consumes the sampler's RNG; it is measured per model by calibrate's audit, not assumed. [Early
exit](think-budgets.md#early-exit) stops when the last n probes share one answer, each with probability at least y. That
rule is fitted so that, on data like the calibration data, at least a share z of its stops agree with the model's answer
after its longest calibrated thinking; that is agreement, not correctness, and on reasoning problems of a kind the
calibration data lacked it cost accuracy on held-out questions, which is why it is opt-in (see [Why it is
opt-in](think-budgets.md#why-it-is-opt-in)).

### Menus and `callo compare`

The memory precheck is a lower bound (file sizes plus a headroom; KV and compute buffers are not counted), so a long
`--ctx` on a dense model can pass and then fail to load, which costs the load time and is recorded as that model's
failure. The GPU rule sees llama.cpp programs by name, oMLX / MLX / ollama servers only while they hold a model
(resident memory), and Callo's own orphans, not other GPU load; a program path with spaces, or a server script run by
a shell given options, is not recognised. Latency comparability is only as good as the box's stability: `--settle` /
`--cooldown` and the drift check detect and reduce thermal drift but nothing corrects for it, and profiles from
different hosts are flagged, never adjusted. With fewer than about 100 questions most frontier neighbours will be
`F≈`; a one-server run's summary and the Notes of every compare report (a menu run's and a one-server run's
`report.md` included) say so, and that is the honest answer. Compare refuses across different data, metrics versions
or readout-engine settings and only flags everything else.
