# Think budgets and deadlines

A single-pass readout answers from the model's first token. A **deliberate** readout lets the model think first: Callo
renders the same System One prompt with the chat template's thinking mode on, generates up to **B** tokens of
reasoning (stopping at the template's own closing think tag), closes the block by force if the budget runs out, and
then reads the answer slot after `</think>` with the same marginaliser. A **budget** caps the trace in tokens; a
[**deadline**](#think-deadlines) caps it in time; [**early exit**](#early-exit) stops it once the answer has settled.

On the two models first measured (a 376-question development set, separate from the test set in
[Measurements](measurements.md#think-budgets): nested boolean-rule items, date-lateness scoring items,
and MMLU / QNLI / PAWS controls; single-stream), thinking first closed most of the single-pass readout's gap on the
rule-logic items, at a latency cost that depends heavily on the model: at a 256-token budget Tiel-Coder 35B-A3B gained
25 pp on the rule-logic (composition) items at a p50 of 2.8 s per question on those items, against 0.29 s
single-pass, while Qwen3.5-9B, still mid-thought at the cut, gained nothing there at 256 (4.2 s) and 18.8 pp at 1024
(16.6 s). The right budget depends on the model, the data and your latency tolerance, so Callo is built to measure it
rather than guess: `callo calibrate --think-budgets` (see
[Calibrating think budgets](calibration.md#calibrating-think-budgets)) writes the **think profile** below.

## Think budgets

```
callo serve --model M.gguf --think-budget 256 --think-profile think/M.think.json
curl -s localhost:8180/v1/systemone -d '{"state": "...", "questions": {...}, "think_budget": 1024}'
```

### Flags and the request field

- `--think-budget N` (default 0: off) is the budget of every text request that sets no `think_budget`. The request
  field `think_budget` (an integer 0 to 32768, or `null` for the server default; `0` means no thinking even when the
  server default is on) overrides it per request and is part of the request digest. `--think-max N` (default 4096,
  at most 32768) caps what a request may ask for (422 above it); it must be at least `--think-budget`.
- `--think-profile P` loads a think profile (written by `callo calibrate --think-budgets`): it supplies the trace
  sampler and the readout temperature per budget (T_B). Budget 0 then takes the profile's T_0 **unless
  `--temperature` is given**, which wins (Callo warns when the two differ by more than one grid step).
  A budget the profile lacks is still served, at the temperature of the largest calibrated budget below it (the
  smallest positive one when it is below all of them; T = 1 when the profile has none), marked `"calibrated": false`
  in `X-Callo-Timings` and counted as `think.uncalibrated`; when `--think-budget` is such a budget, a startup warning
  names the calibrated ones and the rule it is served under. Without a profile, deliberate readouts are served at
  T = 1 (never at `--temperature`, which was fitted thinking-off), uncalibrated: marked `"calibrated": false` and
  counted as `think.uncalibrated`, with a startup warning when `--think-budget` > 0. Loading a profile warns, and still
  serves, when a provenance field (the chat template, say), the host or the llama.cpp build differs from the
  profile's (naming each side's chip and build), when either host is unknown (an unknown host is never a match), when
  the profile was measured busy, has an unverified latency model or drifted, when it is budget-0 only, and when
  budget 0 takes its T_0 and that was fitted under another `--position-bias` profile; it refuses (exit 1) a profile
  measured on another model.

### Which models can think

At startup Callo renders its self-test prompt with thinking on and off and reads the frame off the template: the
closing tag, what the template writes after it, and whether the thinking-on prompt pre-opens the block (Qwen3.5,
Sharp, MiniCPM5) or Callo must open it (Qwen3 dense and the original Qwen3 MoE). A model without a `<...think...>`
block Callo can read is served at budget 0 only; `/callo/stats.think.reason` says why: `no_think_block` (no thinking
mode at all), `unsupported_think_format` (it thinks, but not in a block Callo can read: Gemma 4's channel tokens),
`not_preopened`, `render_failed: ...`, or `sampler_unpinnable: <key>` (llama-server was started with a server-wide
`--grammar`, or `/props` shows a sampler setting Callo does not know at a non-neutral value, which a request cannot
override). `--think-budget N > 0` on such a model exits 1, and so does a profile with thinking budgets; a request
asking for a budget gets 422 (and `--think-sampling` there is ignored, with a warning). A native decision model has
no deliberate readout (`think.reason` says so): `--think-budget` > 0 and `--think-profile` exit 1, `--think-sampling`
is ignored with a warning, and a positive or malformed `think_budget` is a 422, sent nowhere and counted as
`counters.native_think_refused`.

### The trace sampler is pinned

Every generation sends every sampler field llama-server knows, so no server flag moves a trace. By default
(`--think-sampling model`) each model thinks with its GGUF's own `general.sampling.*` values, with Callo's
0.6 / 0.95 / 20 / 0 (temperature, top_p, top_k, min_p) filling only the fields the GGUF does not set. A GGUF `temp` of
0 is honoured (greedy traces). A GGUF value a request could not carry keeps Callo's pin: one llama-server would clamp
or refuse, with a warning; one that is not a finite number (nan, inf, text llama.cpp cannot parse), silently, as
llama.cpp's own reader skips it. A GGUF whose header Callo cannot read falls back to Callo's sampler, with a warning.
`--think-sampling fixed` uses Callo's values for every model, and `--think-sampling temp=..,top_p=..,top_k=..,min_p=..`
(any subset, the rest Callo's; temp at least 0, 0 being greedy; top_p and min_p in [0, 1]; top_k any 32-bit integer,
0 or less turning it off: the ranges llama-server takes as given) sets them yourself. `--think-sampling` is for
serving without a profile only: a profile carries the sampler it was calibrated with, and the two together exit 1.
`/callo/stats.think.sampling` shows the full dict, `gguf_sampling` the served GGUF's keys and `gguf_error` why they
are missing when the header could not be read (`null` when it was read).

### Deliberate readouts are never position-debiased

`Readout.bias` is `deliberate`, a skipped reason in `/callo/stats.position_bias`: the bias profile was measured
thinking-off, and a post-think distribution is a different readout. Budget-0 readouts are debiased as usual.

### One trace per question, seeded by the request

Each question with two or more options gets its own generation at the request's budget (a single-option question is
answered without one and counts in no `think` counter). The response's `usage` keeps its shape: `input_tokens`
includes the trace tokens each readout read, and `output_tokens` stays 0 because no generated text is returned (the
trace length is `think_tokens` in `X-Callo-Timings`). The seed derives from the request body's JSON content
(everything but `model`) and the question id: a retry of the same body draws with the same seed, and bodies whose
content differs are independent draws, including ones that only state the budget differently (`think_budget` absent,
`null`, or the server default written out) or ask for another budget.

### Text only

A media request that asks for a budget gets 422 (`deliberate readouts are text-only in this version`); one that only
inherits the server default is served at budget 0 and counted as `think.skipped_media`.

### Statuses

- 422: a bad `think_budget` (type, range, over `--think-max`), a budget with media, a budget on a model that cannot
  think or on a native upstream.
- 413: the prompt plus the budget (plus 16 tokens of headroom) does not fit the slot context (`n_ctx`), or
  llama-server refused or truncated the trace for its context: lower the budget or raise `--ctx`.
- 504: the generation exceeded its timeout, `--s1-timeout` + 0.1 s per budget token (a 5xx, so Kev retries it; the
  status names the cause).
- 503: upstream down (unreachable within 10 s) or failing mid-generation, or the slot's S1 section stayed busy for
  `--s1-timeout`.
- 500: another upstream 4xx on the generation, a trace llama-server stopped short of the budget for a reason other
  than its context (a server time limit such as `t_max_predict_ms`), or a readout Callo could not make.
- No reply: the client disconnected during a deliberate request (see
  [cancellation](#cancellation-and-client-timeouts); logged at INFO).

### Slots and queueing

A deliberate request takes the lowest idle reserved slot not bound to a lane, else the least busy reserved slot that
is not thinking. While a slot thinks, fast readouts without a lane go to a reserved slot that is not thinking (an idle
one first, else the least busy), never behind the trace; only when every reserved slot thinks does the round robin
apply. A lane (`X-Callo-Conversation`) keeps its own slot either way. Each reserved slot has an **S1 section**:
budget-0 requests take it shared (they interleave pass by pass), a deliberate question takes it exclusively from its
generation to its last readout pass, so nothing lands on the slot between the two (which would cost a re-prefill of
the whole trace on a hybrid model). Waits are FIFO and bounded by `--s1-timeout` (then 503, counted as
`s1.section_timeouts`). With `--s1-slots 1` every readout waits behind a deliberate generation: **mixing deliberate
and fast traffic needs `--s1-slots 2` or more**, and either lanes or this default preference. Under `-kvu` a resident
trace is "another slot's contents" for concurrent readouts on the other slots (see
[Readouts and unified KV](limitations.md#readouts-and-unified-kv)); on a MoE, `--no-kvu` is what removes that.

### Cancellation and client timeouts

Callo watches the client socket for the whole deliberate request; a client that disconnects, while queued or while
the model thinks, aborts the generation upstream (llama-server notices within about a second when it is otherwise
idle; see [Cancelling a non-streaming request can lag](limitations.md#cancelling-a-non-streaming-request-can-lag)),
frees the slot and the section, and is counted as `think.cancelled`. Budget-0 requests keep
[no cancellation](api.md#no-cancellation). Clients with a per-call timeout (Kev's `RemotePredictor`) should set it to
at least the profile's `lat_p95_ms` times the questions per request: a client that times out cancels the generation,
and its retry, with the same seed, repeats the same latency.

### Observability

`X-Callo-Timings` entries for a question read at a budget > 0 gain, after `passes` (after `split` for a question above
26 options): `think_budget`, `think_tokens` (k_B, the trace tokens the readout read: on a natural close the generated
closing tag included, on an eos close the end token Callo replaced by the closing tag; a forced close's appended tag
is not counted), `think_close` (`natural` | `budget` | `eos`; `deadline` under a deadline; `converged` with
[early exit](#early-exit)), `gen_ms`, `gen_prompt_n`, `read_prompt_n`, `temperature` and `calibrated`; budget-0
entries are unchanged, except that every entry timed under a request deadline gains `think_deadline_ms`, `elapsed_ms`
and `deadline_met`, and one whose deadline's thinking fell back carries `deadline_fallback`: `budget0` (no room to
think), `not_chunkable` (the model cannot chunk) or `media` (the request has media); an explicit `think_budget: 0`
under a deadline carries no `deadline_fallback` (see [Think deadlines](#think-deadlines)). An early-exit request with
no deadline and no room to think carries `deadline_fallback: budget0` too (see [Early exit](#early-exit)).

`/callo/stats.think` reports `capable`, `reason`, `close`, `synthesized_open`, `default_budget`, `max`, the loaded
`profile` (`path`, `sha256`, `budgets`, `model_key`, `state`, `latency_model`, `host_match`), `sampling`,
`gguf_sampling`, `gguf_error`, the counters (`requests`, `questions`, `natural`, `budget`, `eos`, `converged`,
`budget0`, `cancelled`, `timeouts`, `skipped_media`, `uncalibrated`, `too_large`, `readout_reprefill`, `text_fallback`;
`requests` counts deliberate requests and `questions` their readouts; `skipped_media` and `uncalibrated` count
requests as planned, so one that then fails still counts), `questions_by_budget` (the default and the profile's
positive budgets, the rest under `other`), `gen_p50_ms`, `gen_p95_ms` and `think_tokens_mean`. A deliberate
question's whole time, generation included, is in `s1.p50_ms` / `s1.p95_ms`; deliberate readouts stay out of the
per-kind cache means. `think.profile.host_match` says whether a loaded profile was measured on this host (`null` when
either host is unknown).

`/callo/stats` also has `host` (`os`, `machine`, `chip`, `memory_bytes`, `gpu` and a `fingerprint`, `null` when
neither the chip nor a GPU can be identified: an unknown host, never a match), `temperature_source` (`flag` |
`profile` | `default`), `model_bytes`, `mmproj_bytes` (launch mode), `n_ctx`, `total_slots`, `upstream_local` (the
upstream's host is a loopback name or address, `localhost` included, so `host` describes llama-server's box too;
published on a native upstream as well), `s1.section_waits` / `s1.section_timeouts` (probe traffic included),
`think_probe` (below), `self_test.think` (the startup think self-test's verdict, `null` when it did not run: the
built-in item read once at budget 64, then the post-think answer-slot check; it warns, never refuses), and under
`router`, the per-slot `deliberate` and `sections` and the counters `s1_deliberate_idle`, `s1_deliberate_quiet` (of
which `s1_deliberate_lane_slot`: the quiet slot taken was bound to a lane, whose warm turn the trace evicts; an
unbound slot wins a tie) and `s1_avoided_deliberate`.

### The think-probe endpoint

`POST /callo/think-probe` is calibrate's internal endpoint (one question read at several budgets at T = 1, under the
S1 section, exclusively when any budget is positive). It takes budgets up to 32768 whatever `--think-max` says, so it
answers loopback peers only unless `--allow-think-probe` (403 otherwise, counted as `counters.think_probe_refused`),
and 404 when Callo's engine is off. Its traffic is counted in its own `/callo/stats.think_probe` block (`requests`,
`readouts`, `errors`, `cancelled`, `timeouts`, `too_large`) and in `s1.readouts`, and kept out of `think` and the
other `s1` windows.

### Calibration checks

`callo calibrate` (the temperature fit) and `callo calibrate --bias` refuse (exit 2) a server that thinks by default
(`--think-budget`, `--think-deadline` or `--think-converge` > 0), before they check its temperature, and one whose
`/callo/stats.think` they cannot read. A server that serves budget 0 at a profile's T_0 reports that T in
`/callo/stats.temperature`, so the T = 1 check refuses it too, and says to restart without `--think-profile` (or with
`--temperature 1.0`).
`callo calibrate --think-budgets` checks neither: its probe reads every budget it asks for at T = 1, whatever the
server's default budget and temperature.

### Startup cost

Without these flags serving is unchanged, except that startup reads the GGUF's `general.sampling.*`, gathers the host
block (waiting at most 5 s for it), makes two extra `/apply-template` calls and three small `/tokenize` calls to read
the frame and, on a model that can think, runs the think self-test: one 64-token generation on the reserved slot and
its readout, then the answer-slot check's two renders (it warns, never refuses).

## Think deadlines

A budget caps tokens; a **deadline** caps time. `--think-deadline MS` (default 0: off; 0 to 600000) is the deadline of
every text request that sets no `think_deadline_ms`, and the request field `think_deadline_ms` (an integer 0 to
600000, `null` for the server default, `0` for none; part of the request digest; anything else is a 422) overrides
it. Under a deadline Callo generates the trace in **chunks** on the same slot (each chunk continues the previous
one's KV, so a continuation prefills one token) and sizes each chunk from learned estimates so the readout still
fits: the trace ends at the first of the close tag, the cap and the **deadline cut**, and the readout follows at once.

```
callo serve --model M.gguf --think-profile think/M.think.json --think-deadline 1800
curl -s localhost:8180/v1/systemone -d '{"state": "...", "questions": {...}, "think_deadline_ms": 1500}'
```

Each limit resolves on its own (the request's field wins; an explicit `0` turns that limit off), and whichever binds
first closes the trace:

| budget | deadline | what runs | `think_close` |
|---|---|---|---|
| 0 (not explicit) | 0 | the thinking-off readout (the single-pass calls) | none |
| B > 0 | 0 | one generation of up to B tokens (the budget path above, unchanged) | `natural`, `budget`, `eos` |
| 0 (not explicit) | D > 0 | chunked, capped at B*: the profile's largest budget (else `--think-max`), never above `--think-max`, clipped to the context | `natural`, `budget`, `deadline`, `eos`, or the `budget0` fallback |
| B > 0 | D > 0 | chunked, capped at B | `natural`, `budget`, `deadline`, `eos`, or the `budget0` fallback |
| explicit 0 | D > 0 | the thinking-off readout, timed | none (`elapsed_ms`, `deadline_met` reported) |

### A deadline never raises a cap

B* never exceeds `--think-max`: it is the smaller of the profile's largest calibrated budget and `--think-max`
(`--think-max` alone without a profile), and a request's own `think_budget` is still a 422 above `--think-max`. A
deadline-only request whose B* does not fit the slot context is clipped to it, never refused; an explicit budget
keeps its 413. The startup line names the cap and which bound set it: `think deadline 1800 ms (all clients;
think_budget 0 opts out): cap 1024 (the profile's largest budget), reserve learned (seed 70 ms from the profile)`.

### No padding

Callo never waits for time to pass: a natural close returns at once and the next question starts at once. The reserve
and the estimates only shrink the next chunk; the only waits are real ones (the S1 section wait and upstream
queueing), and both count against the deadline. A late answer is still sent (lateness is reported, never a 504); only
a chunk still running 5 s past the deadline times out, as a budget's generation does.

### The reserve

The time kept back for the readout is learned per server: 1.5 x the p95 of the last 64 readouts after a `deadline` or
`budget` close, plus 10 ms (at least 20 ms); until 8 of them exist it is 1.5 x the profile's `latency.read_forced_ms`
p95 plus 10 ms, else 100 ms. `--think-deadline-reserve MS` fixes it (tests and pilots). The estimates also hold a
stall allowance for prefills Callo has already dispatched to other slots; a prefill that starts mid-chunk is what can
still make a cut late.

### Neighbouring prefills and `-b`

A neighbour's prefill can delay a chunk by up to one prefill batch; launch mode's `-b 512 -ub 512` keeps that bounded,
and `-b 256 -ub 256` halves it at some prefill throughput.

### When thinking does not fit

The question answers at budget 0, flagged `deadline_fallback: "budget0"` and counted as the `budget0` close. A request
with several deliberate questions splits what is left evenly, question by question, with no minimum: a late question
can fall back while the first one thought.

### The serve default reaches every client

Requests that send no fields think, up to D: a client's own timeout must exceed D plus its margin. A client that wants
no thinking sends `"think_budget": 0`; one that wants a budget without the deadline sends `"think_budget": B,
"think_deadline_ms": 0` (with only `think_budget` it is chunked under the default). Deliberate requests hold reserved
slots exclusively, so fast traffic on the same server needs `--s1-slots` at least the deliberate concurrency + 1, or
lanes. Temperature and bias calibration refuse a server with a deadline default, as they refuse a default budget. The
startup log prints one line naming this reach (above).

### Which models

Deadlines need a thinking model whose close tag is one token: `--think-deadline` > 0 on a native decision model, a
model that cannot think, or a multi-token tag is a startup refusal (exit 1); a request deadline on such a server (a
native one excepted: there it is a 422, as a budget is) is served at budget 0, flagged
`deadline_fallback: "not_chunkable"` and counted `deadline.not_chunkable`. A media request is served at budget 0 and
timed, flagged `deadline_fallback: "media"` (one that asks for a budget is still the budget path's 422).

### Wide questions share the deadline estimator

A deadline question waits for its slot's S1 section only until its share's end minus `Estimators.c0`, the time a
budget-0 readout is assumed to need: the window's p95 once 8 readouts exist, else the profile's budget-0 p95, else
300 ms. The window holds the server's budget-0 readouts, wide (above 26 options) and narrow alike, so wide readouts
raise it for every deadline request on the server: see [the wide-choice latency note](api.md#wide-latency), and serve
wide traffic from a separate server when narrow deadlines matter.

### Temperature

With a profile **deadline point** for D (`callo calibrate --deadlines`) and a cap at least the point's, every readout
under D takes that point's T, `calibrated: true`. Otherwise a natural or `budget` close at cap C takes T_C, a
`deadline` cut at k takes the nearest lower calibrated budget's T with `calibrated: false`, and the fallback takes
T_0. `calibrated` describes the readout, not the setting; T never changes the argmax.

### Observability under a deadline

Under a deadline each `X-Callo-Timings` entry gains, after the think fields (when it thought): `think_deadline_ms`
(the resolved D), `elapsed_ms` (from the request's arrival to this question's readout), `deadline_met`
(`elapsed_ms` <= D), `chunks` (the generations its trace took, 0 on a `budget0` fallback; absent when it was not
planned to think) and, only when the deadline's thinking fell back, `deadline_fallback`:
`budget0` | `not_chunkable` | `media` (no room to think; a model that cannot chunk; a media request). A fallback
entry has no think fields of its own; an explicit `think_budget: 0` under a deadline is timed but carries no
`deadline_fallback`. `think_close` is `natural` | `budget` | `deadline` | `eos` (`budget` is the profile's `forced`),
or `converged` with early exit.

`/callo/stats.think` counts `natural`, `budget`, `eos`, `converged` and `budget0`, and `/callo/stats.think.deadline`
holds `default_ms`, the reserve and its source (`learned`, `profile`, `default` or `flag`), the estimates (decode,
prefill, chunk overhead, stall: p50/p95), `requests`, `questions`, `questions_by_count`, `closes` (with `deadline` and
`converged`), `met`, `missed`, `rerouted`, `not_chunkable`, `cont_reprefill` (continuation chunks that evaluated more
than one token; a chunk after an early-exit probe counts in `converge.after_probe` instead), `elapsed_ms`, `over_ms`
(elapsed - D over the `deadline` and `budget` closes, with their count), `think_tokens` by question count and
`chunks_mean`; `s1` gains `draining`.

### The request log

`--request-log FILE` appends one JSON line per `/v1/systemone` request after the reply's last byte (or at the cancel):
`seq`, `body_sha256` (of the raw body bytes), `t_arr_unix_ms`, `t_written_unix_ms`, `served_ms`, `status` (or
`"cancelled"`), `slot`, `deadline_ms`, `budget`, `converge` (only when the request asked for
[early exit](#observability-with-early-exit)) and per question `qid`, `answer`, `think_close`, `k`, `elapsed_ms`,
`prompt_tokens`. It never holds request text. A client can join its own timings to Callo's on the hash. It covers
only requests Callo serves (not a native decision model's).

### Deadline traces are not reproducible by seed alone

The per-chunk seeds are deterministic (chunk i draws with `think_seed(digest, qid, i)`, chunk 0 with a budget trace's
`think_seed(digest, qid)`), but where the chunks end depends on timing, so a retry can cut its chunks elsewhere and
draw a different trace; a budget-only trace (one generation) still is reproducible.

### Before serving, ask the profile

`callo compare P.think.json --deadline 1500,2000 [--budget B] [--reserve MS]` prints, per deadline, the closes the
planner would make, the cut k, the static `--think-budget` that meets D on 95 % and 99 % of questions, and the
accuracy: exact when the profile has a deadline point at D (and B, if given, is a calibrated budget), else bracketed;
`--reserve` re-simulates the token side with another readout reserve. The figures are single-stream and in-callo: add
your client's hop and any queueing. A profile written without these fields is mapped approximately, and says so. See
[`callo compare`](compare.md).

### Choosing a deadline for a client with a hard timeout

Run a pilot with `--think-deadline D --request-log FILE` and join the client's own timings to the log on
`body_sha256`. Pick the largest D whose tail latency stays under the client's abort, with enough calls behind the
tail: a p99.5 needs a pilot of at least 1,200 calls.

## Early exit

A budget or a deadline says how long a trace may run; **early exit** stops it sooner, once the answer has settled.
Callo reads the answer every K tokens of the trace (a **probe**: a forced close and one readout) and stops at the first
probe where the rule holds: the last n probes agree on one option, each with probability at least y. Calibration fits
the rule per model, so you choose a convergence target z, not n and y. Early exit is opt-in: calibrate and `callo
compare` list it and never pick it (see [Why it is opt-in](#why-it-is-opt-in)).

```
callo calibrate --think-budgets 0,256,512,1024 --converge-probe 32 --model M.gguf --data labelled.jsonl --out think/
callo serve --model M.gguf --think-profile think/M.think.json --think-converge 0.95
curl -s localhost:8180/v1/systemone -d '{"state": "...", "questions": {...}, "think_converge": 0.95}'
```

### Calibrating it

`callo calibrate --converge-probe K` (K 32, 64 or 128; default 64) reads the answer every K tokens of each
largest-budget trace. For each target z (.9, .95 and .99) it fits the rule that saves the most thinking net of probe
cost while the share of stopped answers equal to the answer after full thinking stays at least z (a one-sided 95 % lower
bound, tested in a fixed order from the strictest rule). It measures what a probe costs on this machine and reports
whether stopping early would put the model on your accuracy/latency front, net of probe cost. A short thinker needs a
small K (two or more probes per trace). K is fixed at calibration, and serving uses it as is.

The early-exit points are listed beside the budget and deadline points, with their savings, in `calibrate`'s summary
and `callo compare`'s report, and never picked: the `serve with:` line never carries `--think-converge`. When a profile
has an early-exit point `callo serve` can run, one line under the pick names it, with the flag that opts in:

```
early exit (opt-in, not picked): M@ee.95 saves ~47 tokens / ~500 ms (mean) at z .95 on this data; it holds only on questions like these: serve with --think-converge 0.95 to opt in
```

A point `callo serve` could not run as measured (no rule reaches its z, or the profile's convergence block is one
serve refuses) gets a line under the pick naming it and why instead.

### Why it is opt-in

The rule is fitted on the calibration data, and on held-out questions it held only on question types like that data.
With Tiel-Coder-35B-A3B Q4_K_XL and the rule at z .95, measured on one greedy trace per question, the rule stopped on
217 of 240 MMLU, PAWS and QNLI questions and agreed with the full-thinking answer on 214 of 217 stops. In a sampled
served run, on 96 logic-composition questions built from rules absent from the calibration data (reasoning problems of
a kind it had not seen), accuracy was 74 % with early exit against 100 % with full thinking; overall, accuracy moved
−6.0 points [−9.8, −2.1] (95 % interval), while the median latency on those held-out questions fell by about 1 s. The
opt-in line's saving is another figure: it is measured on your calibration data, at your pick statistic. Calibrate on
data that looks like your traffic, and opt in only if it does.

### Serving it

`--think-converge Z` (default 0: off; 0 <= Z < 1) is the target of every text request that sets no `think_converge`. The
request field `think_converge` overrides it: a number from 0 to below 1, `null` for the server default, `0` for off. It
is part of the request digest but not of the trace's seed, so adding it does not change the seeds the trace draws with;
the trace is still the K-chunked one, so past its first chunk it is not the one-shot trace a budget alone generates.
Anything else is a 422. Both need a `--think-profile` written with `--converge-probe`. `--think-converge` without one,
or with a profile that has no convergence fit, is a startup refusal (exit 1).

Z picks a rule: a Z between two targets takes the next higher target's rule (.92 takes .95's), and a Z above .99 has
none. A Z at or below .9 takes the .9 rule. That is conservative: the .9 rule is stricter than a lower Z needs, so it
stops less often than a rule fitted for that Z would. Some targets have no rule: no candidate rule's agreement passed
the test for that z, often because the calibration data held too few confident stops to show it (.99 rarely has one).
Then early exit is off for those requests: they are served as if they had not asked, and `X-Callo-Timings` says
`"converge": "unreachable"`. The server warns at startup when `--think-converge` itself has no rule, and a request's
`think_converge` on a server whose profile has no convergence fit is `unreachable` too.

Early exit is a third limit beside the budget and the deadline, and whichever binds first closes the trace:

| budget | deadline | early exit | what runs | `think_close` |
|---|---|---|---|---|
| 0 (not explicit) | 0 | Z | chunked every K tokens, capped at B* | `converged`, `natural`, `budget`, `eos`, or the `budget0` fallback (no context left to think in) |
| B > 0 | 0 | Z | chunked every K tokens, capped at B | `converged`, `natural`, `budget`, `eos` |
| 0 (not explicit) | D > 0 | Z | chunked under the deadline, chunks ending on multiples of K, capped at B* | `converged`, `natural`, `budget`, `deadline`, `eos`, or the `budget0` fallback |
| B > 0 | D > 0 | Z | chunked under the deadline, chunks ending on multiples of K, capped at B | `converged`, `natural`, `budget`, `deadline`, `eos`, or the `budget0` fallback |
| explicit 0 | any | Z | the thinking-off readout | none |

B* is the deadline's cap: the profile's largest calibrated budget, never above `--think-max`. The rule was fitted
against that budget's answers. A smaller cap (an explicit budget, or `--think-max` below it) stops on a rule measured
against longer thinking than it serves, and the startup log warns when `--think-max` or `--think-budget` does this.

### Probes and their cost

Each probe is a forced close and one readout on the slot that holds the trace: one token plus the close's suffix, four
new prompt tokens on most models. The next chunk then continues the trace, usually evaluating one or two tokens: on
hybrid and sliding-window models it restores the checkpoint left at the probe, and `after_probe` (below) shows the share
of chunks that did. Under a deadline the planner charges a probe's cost to the chunk that ends on it, as it charges the
per-chunk overhead, and a probe that falls on the deadline cut is the answer's readout. The cost is the profile's
measured one until 8 chunks have followed a served probe. After that it is the live p50 probe readout plus the p50 time
a probe adds to the next chunk.

### Temperature

A `converged` readout is served at the rule's own T, fitted at calibration on the readouts where the rule stopped
(`calibrated: true`). The stopped answers are the confident ones, so a budget's T would misstate them. A question that
never stops keeps the temperature rules above. A rule without a fitted T serves the nearest lower calibrated budget's
T with `calibrated: false`.

### Reach, opt-outs and models

`--think-converge Z` changes every client of the server, as `--think-deadline` does: requests that send no fields
think, up to B*, and stop at the rule. A client that wants no thinking sends `"think_budget": 0`. One that wants a
budget without early exit sends `"think_budget": B, "think_converge": 0` (with only `think_budget` it is chunked every
K tokens). Temperature and bias calibration refuse a server with an early-exit default.

Early exit needs a thinking model whose close tag is one token. `--think-converge` > 0 on a native decision model, on a
model that cannot think, or with a multi-token tag is a startup refusal. A request's `think_converge` on such a server
is served without early exit and flagged `"converge": "not_chunkable"`; on a native upstream it is a 422 instead. A
media request is served at budget 0 and flagged `"converge": "media"`, unless it sets an explicit `think_budget` above
0, which is a 422 (deliberate readouts are text-only).

### Observability with early exit

An entry served with early exit gains, after the think and deadline fields: `converge` (the target whose rule served:
0.9, 0.95 or 0.99), `converge_rule` (`[n, y]`) and `probes` (the probe readouts made). `think_close` is `converged`
when the rule stopped the trace, and `think_tokens` is then the stop position. An entry that asked for early exit and
did not get it carries only `converge`: `unreachable` | `not_chunkable` | `media`. One without a deadline whose cap
left no context to think in is read at budget 0 and carries `deadline_fallback: "budget0"` (before `converge`), as under
a deadline. The request log gains a `converge` key after `budget`, only for a request that asked for early exit: the
target whose rule served, or why it was off.

`/callo/stats.think` counts `converged` beside the other closes, and so does `think.deadline.closes`.
`/callo/stats.think.converge` holds:

- `default_z`, `K` and the served `rules` (n, y and T per target);
- `requests`, `questions`, `converged`, `unreachable`, `not_chunkable` and `media`;
- `probes` and `probes_per_question`;
- `stop_k` (p50 and p95 of the stop positions);
- `tokens_saved` (the cap minus the stop position, summed over converged questions) and `tokens_saved_mean`, its mean
  per converged question. Both are upper bounds: the rest of a stopped trace is never generated;
- `probe_read_ms` (p50, p95) and `after_probe` (the chunks that followed a probe, how many restored, their share);
- `c_ms` and `c_source` (`live` or `profile`; both `null` while no probe cost is known: the profile measured none and no
  live one exists yet).

`requests` and `questions` count what was served with a rule, a question that fell back to budget 0 included (with 0
probes), and `probes_per_question` divides by them. `unreachable`, `not_chunkable` and `media` count each question
that asked for early exit and could think but was served without it; a single-option question never counts.

### Early-exit gate (experimental)

The gate keeps early exit off the requests it serves worst: questions where the model has to apply stated rules step
by step, and is confidently wrong early before it corrects itself later in the trace. Before such a request starts
thinking, Callo asks the model one question at budget 0, on the request's own state and on the slot the request will
think on (so the prompt prefix is shared), with the options yes and no:

> Before answering, would you need to apply several stated rules or conditions step by step to reach the answer?

- **Yes** (P(yes) of at least 0.5): the request is served without early exit. It thinks to the same cap early exit
  would have used, with no probes, and closes `natural`, `budget` or `eos` as usual. With two options, 0.5 is the
  argmax, so the threshold does not move with the readout temperature.
- **No**: early exit runs exactly as above.
- **A failed read** (an upstream error, a refusal, no probabilities): early exit is skipped for that request, and the
  error is recorded. The gate never fails a request.

The read is one budget-0 readout per request, paid whatever the answer: on Tiel-Coder-35B-A3B about 150 ms in
calibration and a 304 ms median when served (its `ms` is the step's wall time). It
reads the model's own probabilities: a position-bias profile is not applied to it. Under a deadline, the read waits
for the slot no longer than the first question would; when the slot stays busy past that, early exit is skipped
(`"error": "the slot stayed busy past the deadline share"`) and the question waits and reroutes as usual.

`--think-converge-gate` turns the gate on for every request that sets no `think_converge_gate`. The request field
`think_converge_gate` overrides it: `true`, `false`, or `null` for the server default; anything else is a 422. Like
`think_converge`, it is part of the request digest but not of the trace's seed. A native upstream refuses `true`.

The gate acts only when early exit runs for the request, that is, when a rule was chosen for its Z. With early exit
off, refused (`not_chunkable`, `media`) or `unreachable`, or with an explicit `"think_budget": 0`, the gate setting is
ignored and no gate read happens. A request whose questions all have a single option has nothing to think about and
is not gated either.

What you see:

- **X-Callo-Timings**: every deliberate entry of a gated request ends with `converge_gate`: `{"p": P(yes), "ms": the
  read's wall time, "skip": true | false}`, plus `"error": "<reason>"` when the read failed (`p` is then `null`). An
  entry the gate skipped carries no `converge`, `converge_rule` or `probes`.
- **The request log**: a `converge_gate` key after `converge`, with the same object.
- **`/callo/stats.think.converge.gate`**: `default` (the server's `--think-converge-gate`), `asked` (gate reads),
  `skipped` (requests served without early exit because of the gate, failed reads included), `errors` (failed reads),
  and `read_ms` (p50 and p95 of the reads that answered). A skipped request ran no rule, so it is not counted in
  `think.converge.requests` or `questions`.
- **The startup log** says when the gate is on. It warns that the gate never acts when early exit cannot run on the
  server: no think profile with a usable convergence block, or a model that cannot chunk (no thinking mode, or a
  close tag of several tokens). A request whose `think_converge` has no rule (`unreachable`) is not gated either.

Calibrate measures it: with `--converge-probe`, `callo calibrate` reads the gate on every calibration question and
lists a gated point `M@ee<z>+g` beside each early-exit point (see [The early-exit gate](calibration.md#the-early-exit-gate)):
the questions the gate flags take the reading at the cap, the others the early-exit reading, each plus its read's
milliseconds. Like early exit, a gated point is listed and never picked; the opt-in line then adds
`--think-converge-gate` for the gated one. A profile calibrated before the gate has no gated point, and `calibrate
--resume` adds the gate reads to it without thinking again.

Measured on the same held-out questions as above (Tiel-Coder-35B-A3B Q4_K_XL, Z .95, the model's own sampler), with
the gate on: accuracy 89.6 % against 89.9 % with full thinking (−0.3 points [−2.4, +1.8]), and a median latency of
1,057 ms against 1,594 ms (537 ms saved [334, 746]). The gate sent 36 % of requests to full thinking. On the
logic-composition questions, accuracy was 96.9 %, against 74 % with ungated early exit and 100 % with full thinking.
Calibrate's gated point predicted a smaller median saving on its own calibration questions (322 ms) than this mix
delivered.

The caveat: the question was chosen after seeing which kind of question early exit fails on, and it has been
validated on one model (Tiel-Coder-35B-A3B) and one question mix. Treat the gate as experimental, and measure it on
traffic like yours before you rely on it.

### What it does not promise

Convergence is agreement with the model's own answer after its longest calibrated thinking, not correctness: accuracy
can move either way, and calibrate prints the change beside each rule. The rule holds on data like the calibration
data; a different mix of questions can stop more or less often, and on question types the calibration data lacks it can
cost accuracy (see [Why it is opt-in](#why-it-is-opt-in)).
