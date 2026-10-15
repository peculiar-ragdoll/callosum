# Calibration

Callo calibrates three things, each measured on a running server and written to a file `callo serve` loads:

- **Temperature** (`callo calibrate`): one scalar that makes the served probabilities honest.
- **Position bias** (`callo calibrate --bias`): see [Position bias](position-bias.md).
- **Think budgets and deadlines** (`callo calibrate --think-budgets`): accuracy and latency per budget, the readout
  temperature per budget, and a pick, written as a think profile. A **menu** runs several models in one go, and
  [`callo compare`](compare.md) puts profiles on one frontier.

Exit codes for every mode are in [Exit codes](exit-codes.md).

Contents: [temperature](#temperature) · [calibrating think budgets](#calibrating-think-budgets) ·
[menus](#menus-several-models-in-one-run)

## Temperature

Served at T = 1 the readout is over-confident. Fit one temperature on labelled data and pass it:

```
callo calibrate --url http://127.0.0.1:8180 --data labelled.jsonl --out T.json
callo serve ... --temperature 1.91
```

T = 1.91 is the value fitted for Qwen3.5-9B Q8_0 on the development split of a classification set (see
[Measurements](measurements.md#accuracy-and-calibration)); it is an example, not a default. The fit minimises NLL
over a 181-point log grid on 0.25..16: Kev's procedure, with the grid extended from 4 to 16 (the step is
unchanged, so a T inside 0.25..4 fits exactly as before). A T on either edge is reported by `callo calibrate`
and the think-budget summary: at 16 the best T may lie beyond the grid (e.g. "T is at the search limit (16): the
probabilities are still overconfident after calibration"); at 0.25 the fit wants sharper probabilities than the grid
allows, common on a small or perfectly separated set. It needs probabilities served at T = 1, so
`callo calibrate` reads `temperature` from `/callo/stats` and refuses (exit 2) unless the server runs at
`--temperature 1.0` (the default without a think profile), and it first refuses a server that thinks by default (see
[Think budgets](think-budgets.md#calibration-checks)). It also refuses against a native decision model: there Callo's
engine is off and there is nothing of Callo's to calibrate. Without `--bias`, `--data` is required (exit 2 without
it), and the summary goes to stdout.

With a position-bias profile, debias first and fit the temperature on the debiased readouts (see
[Position bias](position-bias.md)).

## Calibrating think budgets

`callo calibrate --think-budgets` measures one running server's deliberate readout (see
[Think budgets](think-budgets.md)) at several budgets on your labelled data and writes the **think profile** that
`callo serve --think-profile` loads:

```
callo calibrate --think-budgets 0,256,512,1024 --url http://127.0.0.1:8180 --data labelled.jsonl --out callo-think
callo calibrate --think-budgets --upstream http://127.0.0.1:8199 --data labelled.jsonl      # a bare llama-server
```

```
callo calibrate --think-budgets [B,B,...] [--url URL | --upstream URL] --data FILE.jsonl [--out DIR] [--resume]
[--seed 0] [--allow-busy] [--warmup 3] [--audit 6] [--no-cache] [--think-sampling model|fixed|SPEC]
[--deadlines MS[,MS...] [--think-deadline-reserve MS]] [--converge-probe [K]] [--converge-cost N]
[--latency mean|p50|p95] [--limit STAT=X ...] [--price R] [--max-latency S] [--min-accuracy A] [--verbose]
[--dry-run]
```

### `--think-budgets`

Selects the mode (exclusive with `--bias`: exit 2). Without a value it means `0,256,512,1024`; it takes up to 8
distinct budgets (0 included), each 0 to 32768, and adds 0 when missing because every comparison is against it. A
budget above the server's `--think-max` is measured all the same (calibration's probe ignores `--think-max`), with a
warning naming the `callo serve --think-max` that would serve it. It shares `--url`, `--data`, `--out`, `--resume`,
`--seed` and `--allow-busy` with the other modes; the flags from `--upstream` to `--dry-run` and the launch flags of
[menus](#menus-several-models-in-one-run) (`--model` to `--rates-from`) are its own (exit 2 without it; a launch flag
without `--model` or `--menu` is exit 2 too), and `--bias`'s `--raw`, `--min-questions` and `--max-per-n` exit 2 with
it.

### `--data` is required and labelled

Exit 2 when it is missing or cannot be read. It is the temperature fit's JSONL, one System One request per line with
a per-question `label` and an optional `_meta` (`id`, `group_id` or `pair_id`, `source`, `variant`). Each question is
read on its own. Records with media, records whose `_meta.variant` is not `clean`, unlabelled questions (no `label`,
or `"label": null`), and questions above what the server reads (`wide_readable`, or `wide_coverage` on a server
from before Callo 0.3.1; `too_many_options`) are skipped and counted.
A malformed line, a file that is not UTF-8, a label that is not one of its question's options, a duplicated
`_meta.id` and a group (`_meta.group_id`, else `pair_id`) whose questions come from two `_meta.source`s exit 1.
Every interval resamples whole groups, within their source when the data has sources.

### One generation per question

Each question's trace is generated once at the largest budget and cut to every smaller one, so the budgets share one
trace; the readout after each cut is real. A single-option question needs no trace: it is read once, counts at every
budget with that readout (as serve answers it) and is never audited.

Latency at a cut is modelled (frame + fixed cost + decode time linear in tokens + the readout, minus the
calibration-only re-prefill) and the model is **audited**: at each budget below the largest that cut at least 3
questions, `--audit N` (default 6) of them are regenerated at that budget and the measured latency compared. A budget
is `verified` when the median relative error over at least 3 audited questions is at most 10 %, else `unverified`
(`latency_reason: lat_err`); one that cut fewer than 3 questions is `audit n/a`, and `--audit 0` leaves every budget
`unaudited`. `--audit 1` or `2` can never verify, and a warning says so. `--warmup N` (default 3) questions are read
at every budget first and discarded. A **drift check** regenerates 3 traces of at least 64 tokens at the largest
budget at the end and compares their decode speed with phase B's; it flags a ratio more than 5 % away from 1, or a
`pmset -g therm` CPU speed limit below 100, read at the start, after each phase and at the end (macOS; `null`, "no
thermal data", elsewhere).

### `--url` and `--upstream`

**`--url`** (default `http://127.0.0.1:8180`) measures a running `callo serve`: a Callo server with think budgets, a
`model_key` and Callo's engine on (not a native decision model), whose `--cache` matches the run's (each is exit 2
otherwise). The run's `--cache` is on unless `--no-cache` is given, as for `callo serve`, so a server started with
`--no-cache` needs `calibrate --no-cache` (the refusal says so). A model without a thinking mode Callo can read is
measured at budget 0 only, and the summary says so.

**`--upstream URL`** attaches an in-process `callo serve --upstream URL --port 0 --think-budget 0` (with the run's
`--cache` or `--no-cache`; nothing else is forwarded) to a bare llama-server for the run; `--url` and `--upstream`
together exit 2. A URL `callo serve --upstream` would refuse (no `http://`, https, a path, credentials) is an immediate
exit 2 with serve's reason; then a `GET /health` with a 5 s timeout: nothing listening or an unresolvable host is an
immediate exit 2, while a server that answers at all, a 503 while the model loads included, gets the in-process
serve's usual wait. Two things to know in that mode: the in-process serve's startup probe fills two of that
llama-server's slots (the last and slot 0) and erases them when it has `--slot-save-path`, wiping any agent's prompt
cache there, as starting `callo serve --upstream` yourself does; and no `--position-bias` can be loaded, so budget 0
is read undebiased and its T_0 is fitted on undebiased readouts (serving the profile later with `--position-bias` and
without `--temperature` triggers serve's bias-sha mismatch warning). A run that needs `--position-bias`, `--ctx` or
llama-server flags starts `callo serve` itself and uses `--url`. When the measured `callo serve` attached to its
llama-server rather than launching it (always with `--upstream`), `environment.server.server_args` is null:
llama-server's `-ctk/-ctv`, MTP and draft settings are not recorded, so nothing can compare them later; the summary
says so.

### The host prompt cache

llama-server keeps a host-side prompt cache (`--cache-ram`, 8 GiB by default) that can hand phase B the state phase A
left for the same question, so the generation prefills a few tokens instead of the prompt and the latency comes out
flattered: something a served cold question never gets. Every generation's `gen_prompt_n` / `gen_cache_n` is
recorded; the profile's per-budget `prefill_reused` is the share of generations that reused their own question's
state (`gen_prompt_n` at most 8, or `gen_cache_n` at least 90 % of the prompt after a question of another record: a
question of the same record legitimately shares its state and context), and above 10 % the latency model is
`unverified` (`latency_reason: prefill_reused`, which wins over `lat_err`) and the summary names the remedy: restart
llama-server with `--cache-ram 0` for the calibration, or accept the flattered prefill. The other way round is small:
calibrated with the host cache off, a server that keeps it pays a state save on each question switch that calibration
did not, a few ms, more on big dense models, so served latency can come out slightly higher than calibrated; the
summary says so whenever it names the remedy. In these attach modes calibrate cannot set the flag, and
`environment.server.cache_ram` is null (unknown).

### The guard

The guard of `--bias` applies (see [Position bias](position-bias.md#a-busy-server-refuses)): `/callo/stats` is checked
before the first request, every 25 questions, at each phase boundary and after the last; the server must be idle, and
foreign slot activity during the run is a refusal unless `--allow-busy`, which records it and marks the profile
`busy`. A moved slot or a Callo restart mid-run exits 2 (`--resume` continues). A 503, a 504 or a connection error is
retried 3 times (1, 4, 16 s), and a retried request is followed by a check at once; any other failure exits 1 with
the raw file intact. Before anything is written, a preflight checks that every question's thinking-on prompt plus the
largest budget plus 16 tokens fits the slot context, and exits 2 naming the question otherwise (raise `--ctx` or lower
the budgets).

### Output and `--resume`

`--out DIR` (default `./callo-think`) gets `<name>.think.json` (the profile, strict JSON) and `<name>.raw.jsonl`
(every readout as it arrives); `<name>` is the served model file without `.gguf` (`model` when Callo does not report
one). `--out DIR` also gets `report.md` and `report.json`: the single-profile form of [`callo compare`](compare.md)
(a DIR that holds a menu run keeps the menu's report).

**`--resume`** continues an interrupted run from the raw file when its header matches: the model (`model_key`,
`provenance` but the llama.cpp build, the file name, its sizes, its thinking mode, the user delimiter), the server (KV
layout, slot counts, `n_ctx`, `--cache`, the position-bias profile, `upstream_local`), the data (name and sha256), the
budgets, `--seed`, `--warmup`, `--audit`, `--cache` and the sampler. Any difference exits 2; another host, an unknown
host or another llama.cpp build only warns. A raw file measured with the cache off resumes only with `--no-cache`, as
the refusal says. A resumed session that still has something to read warms up again and starts with a fresh check; a
complete raw file is only re-fitted, with no probe request: no question is read again, no thermal reading or check is
taken, and nothing is asked of the server beyond the one `GET /callo/stats` that names the model (the raw header
records what the fit needs); the measurement and its verdicts stand, and only the pick preference (`--latency`,
`--limit`, `--price`, `--max-latency`, `--min-accuracy`) and `--verbose` can change what comes out. The server may
even be stopped: `--resume` then refits the one complete raw file in `--out` for these settings, and exits 2 naming
them when there are several (start the server so its model picks the file, or point `--out` at a directory holding
only that model's). A raw file written before Callo recorded those header fields is refitted through the server
instead, with a note saying so. Without `--resume` an existing raw file exits 2; a run refused at its first check
leaves no raw file behind.

### The summary

The summary goes to stdout; the progress log goes to stderr. It has a header (model file, size, `model_key`, data, host,
the sampler and where each field came from), then one row per point: accuracy, Δacc vs budget 0 with a 95 %
cluster-bootstrap interval, the five latency statistics (mean, p50, p95, p99 and max; the pick statistic's with its
interval, p99 shown as `= max` below 100 questions, where its rank is the last question), mean trace length, the
forced-close share, and the frontier marks (`≈`: within noise of the faster frontier point before it; `dominated by X`,
with `(opt-in)` after an early-exit X; `opt-in (early exit)` on an early-exit row, plain `m@ee.95` or gated
`m@ee.95+g`, which the pick never takes), then a
tails line saying how many questions each tail rests on (`tails (n 376): p95 from 19 questions, p99 from 4, max from
1`). Budgets that read the same thing on every question merge into one point (`m@64≡256`). Below about 100 questions a
line says that neighbouring frontier points will mostly be `≈`, and that rerunning with another `--seed` estimates the
trace variance.

Then the audit, prefill-reuse and drift lines, the pick (see [Pick preferences](compare.md#pick-preferences); or
`pick: none (<reason>)` naming each filter, how many points it removed and the nearest miss), the target M it was
measured against and the pick's Δacc against M, and `serve with: callo serve ... --think-profile ... --think-budget B`,
B being the largest budget of a merged pick (it truncates the fewest unseen questions; a deadline pick serves
`--think-deadline D`; early exit is never picked): `--model FILE` when Callo launched the measured
llama-server, `--upstream URL` (that llama-server's) in the attach modes, plus `--think-max B` above serve's default
4096 and `--no-cache` when the run measured without the cache. A profile from a menu or `--model` run names the GGUF by
its absolute path and repeats the launch shape it was measured with (`--ctx`, `--parallel`, `--no-kvu`, `--s1-slots`,
`--user-delimiter`, `--mmproj`, `--position-bias`, `--llama-server` when it is not the default `llama-server`, and the
entry's llama-server arguments after `--`; not the calibration-only `--cache-ram 0`), so the line runs from any
directory and serves the measured readout at the measured latency. `--verbose` adds NLL@1, NLL@T, ECE10@T and T.

### `--dry-run`, `--think-sampling` and `--seed`

**`--dry-run`** reads `--data` only: it prints the filtered questions, the budgets and the worst-case token count
(with the audit's, the drift check's and the warm-up's upper bounds) and contacts no server.

**`--think-sampling`** (default `model`: the served GGUF's `general.sampling.*`, Callo's 0.6 / 0.95 / 20 / 0 for the
fields it lacks, a GGUF `temp` of 0 giving greedy traces; `fixed`: Callo's for every field; or
`temp=..,top_p=..,top_k=..,min_p=..`) is the trace sampler, recorded in the profile, which `callo serve` then pins.
**`--seed`** (default 0) fixes the question order, the trace seeds, the bootstrap and the cross-fit folds.

### Exit codes

0 with a profile written; 1 for a failed request, an unusable reply or bad data (see `--data`; also a file with no
labelled text question, or a raw row cut short; in the launch modes, `--model` and `--menu`, bad data is a refusal
instead, exit 2, their dry run included), or an `--out` that cannot be written (a file rather than a directory, a full
disk: the previous profile is kept whole); 2 for a usage error, a missing or unreadable `--data` included, or a
refusal; 130 on Ctrl-C (the raw file keeps what was read; `--resume` continues).

### Deadlines

`--deadlines MS[,MS...]` (at most 4) simulates, after each question's largest-budget generation, where
`callo serve --think-deadline MS` would cut its trace (single-stream, on the trace's own measured prefill and decode
times, with the readout reserve measured in a short pre-pass on the warm-up questions, or
`--think-deadline-reserve MS`) and reads the answer there: the profile gains a **deadline point** per MS (accuracy,
latency, the five closes, cut k, the fitted T, the reserve), each row its `d` readouts, and every run its `latency`
block and `rows[].t` terms, which `callo compare --deadline` maps from. A deadline at which no question was read
(every one fell back to budget 0 or had one option) is no point and is listed, with why, in `deadline_skipped`. A
deadline point joins the pick beside the budgets (a deadline pick serves
`--think-deadline D`; see [Think deadlines](think-budgets.md#think-deadlines)). Adding `--deadlines` or
`--converge-probe` to a run with `--resume` is a header mismatch.

### Convergence and early exit

`--converge-probe [K]` (32, 64 or 128; default 64) reads the answer every K tokens of each largest-budget trace and fits
early-exit rules (stop once the answer has held for n probes at confidence y), cross-fitted on clusters, with
fractional-Beta lower bounds and a fixed testing sequence; `--converge-cost N` (default 6, 0 to skip) regenerates N
questions live to measure what a probe costs at serve time. The profile's `converge` block and the summary report the
flip statistics, the rules, the early-exit points and whether they gain on the model's frontier of budget, deadline and
early-exit points, and over which probe costs: the gain is re-tested at probe costs of 0.5 to 32 decode steps, and early
exit "pays while a probe costs <= ρ" only up to the largest ρ at which it gains at every cheaper cost tested
(`rho_prefix`; `rho_max` is the largest ρ at which it gains at all). The early-exit points are listed beside the
budget and deadline points, with their savings, and never picked: early exit is opt-in, because its rule holds only on
questions like the calibration data (see [Why it is opt-in](think-budgets.md#why-it-is-opt-in)), so the `serve with:`
line never carries `--think-converge`. When the profile has an early-exit point `callo serve` can run, one line under
the pick names the one with the highest z, its saving against the budget it was capped at, and the flag that opts in:

```
early exit (opt-in, not picked): M@ee.95 saves ~47 tokens / ~500 ms (mean) at z .95 on this data; it holds only on questions like these: serve with --think-converge 0.95 to opt in
```

An early-exit point `callo serve` could not run as measured (no rule reaches its z, or the profile's convergence block
is one serve refuses) gets a line under the pick naming it and why instead.

### The early-exit gate

With `--converge-probe`, calibrate also reads the [early-exit gate](think-budgets.md#early-exit-gate-experimental)
(experimental) once per question with two or more options: the same built-in question serve asks, read the way serve
reads it (budget 0, the model's own probabilities with no position-bias offsets), between phase A and the thinking,
and timed like the phase-A readouts, so its milliseconds compare with the other points'. Each read is a `gate` row of
the raw file (`p_yes`, `ms` and the gate's `sha256`), and `--resume` continues the reads like any other phase. A raw
file written with `--converge-probe` before calibrate read the gate gets only the gate reads on `--resume`: nothing is
thought or read again (its warm-up reads budget 0 alone), and the profile is rebuilt. With one server and none
answering, such a file is refitted as before, with no gated point. In a menu run this holds for a model `run.json`
records as done too. A single gate read that fails is no failure of the run: its row records `p_yes` null and the
question counts as flagged, as serve counts it. When the re-entry itself does not finish (its server does not start,
the precheck skips it, another model server is found before it, calibrate refuses or fails, a Ctrl-C), the model
stays done with its profile without gated points, untouched, and a later `--resume` tries the gate reads again. When
another model server appears only after the gate reads, the new profile is kept with its gated points and marked
`gpu_busy`. When the new profile cannot be stamped, the one before is put back and the reason says so; the raw file
keeps the gate reads, so a later `--resume` finds nothing left to read and keeps that profile.

Beside each early-exit point `M@ee<z>` the profile then has a **gated point** `M@ee<z>+g` (`ee_gated`, and `g` in
each row), computed offline with serve's rule:

- a question the gate flags (P(yes) of at least 0.5, or a failed read) takes the reading of the budget at the
  early-exit point's cap: its answer, latency, thinking tokens and close, with no probe;
- every other question takes its early-exit reading;
- every question the gate was read on adds that read's milliseconds to its latency.

Its accuracy, the five latency statistics and the thinking tokens are computed as the early-exit point's; it also has
its own forced-close share (the questions answered at a cut of a trace that hit the cap), which an early-exit point
does not carry; `dacc_cap` is its cross-fitted Δacc against the budget at the cap, as the rule's own `dacc` is;
`gate` holds the flagged share and the read's p50 and p95. The `converge.gate` block names the gate the points were
measured with (its `sha256`), the 0.5 rule, the reads, the flagged share, the failed reads and the read cost; when a
question has no gate read, or the reads come from two gates, it says why and there is no gated point. Gated points are
listed and marked like the early-exit points, and never picked. When both variants of the opted-in z can be served,
the opt-in line names both:

```
early exit (opt-in, not picked): M@ee.95 saves ~47 tokens / ~500 ms (mean) at z .95 on this data; gated (M@ee.95+g) saves ~N ms (mean); it holds only on questions like these: serve with --think-converge 0.95 (add --think-converge-gate for the gated one) to opt in
```

## Menus: several models in one run

calibrate can take a **menu** of models and run them one after another on the same data, budgets, seed, warm-up,
audit and `--cache`, each through its own in-process `callo serve`, then report the **joint frontier** over every
(model, budget) point:

```
callo calibrate --think-budgets --data labelled.jsonl --model A.gguf --model B.gguf --ctx 65536 --out think/
callo calibrate --think-budgets --data labelled.jsonl --menu menu.json --out think/
callo compare think/A.think.json think/B.think.json other-run/C.think.json --out joint/
```

```
callo calibrate --think-budgets [B,B,...] --data FILE.jsonl (--model GGUF ... | --menu MENU.json) [--out DIR]
[--ctx N] [--parallel 4] [--no-kvu] [--s1-slots 1] [--llama-server BIN] [--user-delimiter S] [--allow-gpu-busy]
[--mem-headroom 2] [--gpu-mem GiB] [--settle 30] [--cooldown therm] [--rates-from DIR] [--resume] [--dry-run]
[-- llama-server args]
```

with the run-wide flags of the one-server form (`--seed`, `--warmup`, `--audit`, `--no-cache`, `--think-sampling`,
`--allow-busy`, `--latency`, `--limit`, `--price`, `--max-latency`, `--min-accuracy`, `--verbose`).

### Sources

Exactly one; two are exit 2: `--url` (a running `callo serve`; given explicitly, even with its default value, it
counts as a source), `--upstream` (a bare llama-server), `--model GGUF` (repeatable: launch mode with the **shared
launch flags** `--ctx`, `--parallel`, `--no-kvu`, `--s1-slots`, `--llama-server`, `--user-delimiter` and the
arguments after `--`; `<name>` is the GGUF stem, a character outside `[A-Za-z0-9._-]` becoming `_`, and a duplicate
gets `-2`, `-3`), or `--menu MENU.json` (launch mode with per-model settings; `<name>` is the entry's `name`). The
shared launch flags follow `callo serve`'s rules (exit 2 otherwise): `--ctx` positive, `--parallel` at least 1,
`--s1-slots` at least 1 and below `--parallel` when that is above 1, `--no-kvu` only with `--ctx`.

A menu is JSON:

```json
{"models": [{"name": "tiel-q4", "model": "/v/Tiel.gguf", "ctx": 65536},
            {"name": "qwen9b", "model": "/v/Qwen.gguf", "parallel": 4, "args": ["-b", "512"]},
            {"name": "big", "model": "/v/70B.gguf", "no_kvu": true, "ctx": 32768, "think_sampling": "fixed"}]}
```

The keys are `name` (unique, `[A-Za-z0-9._-]{1,64}`), `model`, `mmproj`, `ctx`, `parallel`, `no_kvu`, `s1_slots`,
`llama_server`, `args`, `user_delimiter`, `think_sampling`, `position_bias` (an unknown key is exit 2), and the shared
launch flags are the defaults an entry leaves out; a `null` `mmproj`, `ctx`, `user_delimiter` or `position_bias` means
"none for this model". Budgets, data, seed, warm-up, audit and `--cache` are run-wide on purpose: points are
comparable only when every model reads the same questions in the same way. `think_sampling` per entry overrides the
run's `--think-sampling` for that model; when policies differ the report says so.

### `args` policy

Flags that constrain the trace and cannot be overridden per request (`--grammar`, `--grammar-file`,
`-j`/`--json-schema`, `--json-schema-file`) are refused (exit 2). Sampler flags (`--temp`, `--top-k`, `--repeat-*`,
`--dry-*`, `--mirostat*`, `-l`, `--ignore-eos`, ...) are allowed with a warning: the pinned sampler overrides them on
every generation, and every readout pass pins a neutral chain (its pick is the raw argmax), so they change nothing.
Model and speed configuration (`-ctk`/`-ctv`, `--chat-template*`, `--lora*`, `--override-kv`, speculative and MTP
flags) is allowed and recorded in the profile's `server_args`, which compare shows.

### The GPU rule

Before the first model, no other model server may hold the GPU: any llama.cpp program (`llama-server`, `llama-cli`,
`llama-bench`, ... by the executable's basename, whatever its path), an oMLX, MLX (`mlx_lm.server`, also as a console
script or `-m mlx_lm server`) or ollama server (`ollama serve`, `ollama runner`) that holds a model (resident memory of
2 GiB or more, or unmeasurable), or an orphan left by an earlier Callo (the refusal names its pid file: delete it when
that pid is not the llama-server). An idle `ollama serve` or the oMLX menu-bar app holding no model is not counted,
and a Python script named `llama-server` (the test suite's fake) never is. The rule **fails closed**: when `ps`
cannot run, times out or lists nothing, or the scan itself fails, Callo cannot tell and refuses as if a server were
running. The scan only reads the process list (it may remove stale pid files an earlier Callo left in
`DIR/.callo-cache`, whose pid is dead or unreadable). The refusal (exit 2, before run.json or any profile is written)
lists the pids; `--allow-gpu-busy` runs anyway and marks every profile `gpu_busy`, which compare flags. The scan
repeats before each later model and after each model (the driver's own llama-server excluded): a server that appears
marks that model `failed` (`gpu_busy` under `--allow-gpu-busy`), and a scan that raises fails that model either way.

### The memory precheck

It skips a model, never the run: `need = model + mmproj + --mem-headroom` (default 2 GiB; a split GGUF,
`...-00001-of-0000N.gguf`, counts every shard, and a missing shard fails that model, naming it) against a budget of
75 % of memory on macOS (Metal's default working set), nvidia-smi's total on Linux with a GPU (summed over the GPUs),
else the memory; `--gpu-mem GiB` sets the budget on every OS. It is a lower bound (KV and compute buffers are not
counted): a model that passes and still fails to load is a per-model failure, as is a GGUF that does not exist.

### Each model's run

Each model runs as `callo serve --model ... --think-budget 0 --port 0` in a worker thread, with the run's `--cache` or
`--no-cache` (always explicit; a menu has no cache key, so the run's flag, on by default, decides), the entry's flags
and llama-server's `--cache-ram 0`: the host prompt cache is off for every calibration server, because phase A would
otherwise leave each question's thinking-off state in that cache and phase B's generation would prefill a couple of
tokens instead of the prompt, flattering every latency above budget 0 (a served cold question never gets that). An
explicit `--cache-ram` after `--` or in an entry's `args` comes later and wins; the profile records the value the
server ran with in `server_args` and as `environment.server.cache_ram` (0 by default; null in the attach modes, where
calibrate cannot know it). Its log goes to `DIR/<name>.serve.log`, never to the terminal (every thread of serve's,
the child watcher included; each attempt opens with a separator line).

A failed model's `reason` in `run.json` says what happened, for example: serve's last error line in that log when the
serve exited before listening, "did not start within N s (its log: ...)" when it never answered, "callo serve did not
accept this entry's flags: ..." for a value serve refuses, a calibrate refusal or failure in its own words
(`<name>: <reason>`), "another model server is running" or "another model server appeared (pid ...)", "interrupted;
--resume continues from the raw file" after a Ctrl-C, or, for anything unexpected, the exception's `Type: message`
(its traceback in the serve log).

Ctrl-C or SIGTERM interrupts the current model at once: serve stops its llama-server (a llama-server still loading is
SIGTERMed, since serve cannot stop it before its load returns), the model is recorded `failed: interrupted; --resume
continues from the raw file`, and the run stops there, with no further model. A second Ctrl-C SIGKILLs that
llama-server and exits. Callo only ever signals a llama-server it launched in this run: the pid is read from its own
starting pid file in `DIR/.callo-cache`, never from another file, and a llama-server whose serve thread does not
return within 60 s of the stop is SIGTERMed, then SIGKILLed. After each model the driver waits for the llama-server to
exit (at most 60 s), then `--settle` seconds (default 30, at most 86400), or with `--cooldown therm` until `pmset -g
therm` reports `CPU_Speed_Limit = 100` (macOS, at most 10 min): the next model starts from an idle GPU. A first Ctrl-C
cuts the wait and the settle short. Menu order is yours; the drift check in every profile and the start/end times and
decode rate in the report are the defence against a box that heats up over the run.

### `run.json`, failure isolation and `--resume`

`DIR/run.json` records the menu's sha256, the run-wide settings and per model `status`
(`pending|running|done|failed|skipped`), `reason`, `started`, `ended`, `profile`, rewritten atomically (and fsynced)
at every transition. One model's failure (a load failure, a preflight refusal, a request that fails after its
retries, a guard refusal without `--allow-busy`, a slot change, any unexpected error, whose traceback goes to its
serve log) ends that model only; its raw file is kept and the next model runs. A `run.json` that cannot be written (a
full disk) stops the run: the report and the summary are still attempted, exit 1, and `--resume` continues from the
last write. An existing non-empty `DIR` without `--resume` is exit 2. `--resume` re-reads `run.json`: it refuses
(exit 2) when the run-wide settings differ or the file is not a valid run file, skips a `done` model with a valid
profile (`<name>: done already (PROFILE); skipped`), resumes a `failed`, `running` or `pending` model from its raw file,
and re-prechecks a `skipped` one. A `done` model whose raw file still has reads left under these settings (a
`--converge-probe` raw file from before the gate) is resumed too: it reads only those, then its profile, `run.json`
and the report are rebuilt. If the re-entry does not finish (a launch, a precheck skip, another model server, a
refused or failed calibrate, a Ctrl-C), the model stays `done` with the profile it had (the reason says why) and in
the report, and the next `--resume` tries again; [the early-exit gate](#the-early-exit-gate) says what happens when
another model server appears only after the reads, or the new profile cannot be stamped.

**Adding a candidate later** is the intended workflow: append it to `MENU.json` and `--resume`; only it runs, and the
report is regenerated over every profile. A model the resumed menu no longer names stays in `run.json` as history
(`dropped`), is never run, and is left out of the report; naming it again restores it.

Exit codes: 0 when every model is done or skipped by the precheck, 1 when at least one failed (or the report itself
failed or was refused), 2 for a usage error or a refusal before any model ran, 130 when interrupted (a Ctrl-C during
the report included). Before the GPU check and the first model, `--data` is read once: a bad line or no labelled text
question is exit 2, and nothing is launched. The report goes to stdout and `DIR/report.md`, and is written in every
case from the profiles that exist (with no profile at all, a stub that names the failed and skipped models; when
compare refuses the joint report or a done profile does not load, a stub whose first line says why, and exit 1).
`run.json` records each profile by its absolute path, so `--resume` works from any directory.

### A menu's `--dry-run`

`--dry-run` launches nothing, writes no run.json, profile or report, and needs no GPU (its scan only reads the process
list, as the run's does, and may remove stale pid files): the questions after filtering, the budgets, the GPU-rule
scan result (exit 2 when the run would refuse, a `ps` that fails or unusable `--data` included), the memory budget,
and per model its file size and precheck verdict, the worst case in generated tokens and a time estimate **only**
when a profile with the same `model_key` exists in `DIR` or `--rates-from DIR` (its measured decode rate and budget-0
readout time); otherwise "unknown until measured", and the live ETA prints after 5 questions.
