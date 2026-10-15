# `callo compare`

`callo compare` puts think profiles (from [`callo calibrate --think-budgets`](calibration.md#calibrating-think-budgets))
on one accuracy/latency frontier and picks a configuration by your preference. It makes no server calls. A menu run
ends by calling it on its profiles; it also joins profiles from separate runs, days or boxes, and a single profile is
a valid input.

```
callo compare think/A.think.json think/B.think.json --latency mean --price 1 --out joint/
```

```
callo compare P1.think.json P2.think.json ... [--latency mean|p50|p95] [--limit STAT=X ...] [--price R]
[--max-latency S] [--min-accuracy A] [--max-memory GiB] [--deadline MS[,MS...] [--budget B] [--reserve MS]] [--out DIR]
[--json] [--verbose]
```

`--deadline` adds, per profile and deadline, the mapping block of
[Think deadlines](think-budgets.md#before-serving-ask-the-profile); `--budget B` combines it with a token cap
(whichever binds first) and `--reserve MS` re-simulates with another readout reserve.

## Refusals (exit 2)

Different data (`comparable.data.sha256`), a different metrics version, different readout-engine settings
(`variant`, `top_k`, `max_depth`, `divergence_mass`), or question id sets that differ; an unreadable or malformed
profile, a bad filter value and an `--out` that cannot be written are exit 2 too.

## Loud flags

Shown above the table and kept in `report.json`; compare goes on: different hosts ("latencies from different machines
are not comparable", an unknown host never counting as a match), builds or `callo` versions; a different `--cache`, user
delimiter, or whether `--cache` readouts send the delimiter (`cache_delimiter`); a non-local upstream; `gpu_busy` or a
`busy` server state; an unverified latency model (phase-B generations that reused a cached state of their own question
through llama-server's host prompt cache, `prefill_reused`, with the remedy; else the audit's error) or flagged drift;
different host prompt-cache settings (`cache_ram`, unknown in the attach modes); the same model (`model_key`) under
two samplers; different seeds, warm-ups, position-bias profiles, server args or bootstrap strata. Samplers that differ
**between** models are expected and never flagged: the table's sampler column shows them, a `*` on values Callo filled
in rather than the GGUF. Notes: a budget-0-only model ("no thinking mode", or "thinks in a format callo cannot
read"), missing thermal data, and a profile measured on fewer than 100 questions ("its intervals are wide").

## The table

One row per point sorted by the pick statistic: `model@B` (merged `B≡B'` when two budgets read identically; a deadline
point is `model@Dms`, an early-exit point `model@ee<z>` and its gated point `model@ee<z>+g`), acc, mean, p50, p95, p99 and max (the pick statistic's with
its interval), mean trace length, forced %, sampler, model GiB (`bytes + mmproj_bytes`) and a mark: `F` frontier, `F≈`
frontier but within noise of its faster neighbour, `·` dominated (by which point; `(opt-in)` when that is an early-exit
point, which the pick never takes), with `opt-in (early exit)` on an early-exit row and `(opt-in)` after an early-exit
point in the scatter's legend; then the tails line. A profile written before mean, p99 and max were recorded gets them
recomputed from its rows, exactly as calibrate computes them.

**Paired Δacc intervals** over the shared question ids follow, for neighbouring frontier points and for every point
against the target M (with a Δscore at a positive `--price`): the items are identical across models, so "A@1024 beats
B@1024" is a measured claim, not two overlapping intervals. Then the **pick** (see
[Pick preferences](#pick-preferences); `--max-memory` filters on file size, a lower bound on memory) with M, its
paired Δacc against M, and the `callo serve --model ... --think-profile ... --think-budget B` line for the pick, an
ASCII scatter (accuracy against log latency) so the frontier is visible in a terminal, and a Runs section: each
profile's launch shape (`-np`, KV, context, `cache-ram`, server args), start and end times and decode rate, then a
menu run's failed and skipped models with their reasons and the ones an interrupted run left unfinished (a run with no
profile at all still writes that). `--out DIR` writes `report.md` and `report.json`; `--json` prints the JSON instead
of the markdown.

## Pick preferences

calibrate and compare share the flags and their checks.

- **The pick statistic** `--latency mean|p50|p95` (default **mean**: the time spent per question, its tail weighted by
  how often it happens) orders the frontier and is what a price charges; p99 and max are limits only. `--latency p50`
  picks on the median instead, and `--max-latency` limits whichever statistic is chosen. The pick line names the best
  point (M), not "the most accurate".
- **Limits.** `--limit STAT=X` (repeatable; STAT mean, p50, p95, p99 or max; X in seconds, `6` or `6s`, or `6000ms`)
  drops every point whose statistic is above X before the frontier is built, so a point that fails a limit can never
  eliminate one that passes; the quoted `'p99<=6s'` is an alias, and every line Callo prints uses the shell-safe
  `STAT=X` form. `--max-latency S` is `--limit STAT=S` on the pick statistic.
- **Price.** `--price R` (pp/s: `1`, `1pp/s`) says what one more second per question is worth in accuracy points: the
  target M is the frontier point with the best 100·acc − R·latency (at the default `--price 0`, the most accurate),
  and the pick is the fastest frontier point whose paired Δacc against M has a 95 % CI including 0. A positive price
  excludes `--min-accuracy` (exit 2); `--price 0 --min-accuracy A` is the default rule plus the floor. Raising the
  price never moves M to a slower point and the pick is never slower than M, but the pick itself is not monotone in R
  ("within noise" is not transitive), so the pick line names M and the pick's Δacc against it.
- **Re-picking.** compare re-picks any profile under any preference in seconds and never writes back; when the
  profile's own pick used another statistic it says so (`the profile picked tiel@1024 on p50; this pick is on mean`).
- **Early exit.** Early-exit points (`model@ee<z>`) are listed in the table, the scatter and `report.json` with their
  savings, and never picked, under any preference: early exit is opt-in, because its rule holds only on questions like
  the calibration data (see [Why it is opt-in](think-budgets.md#why-it-is-opt-in)). For each profile with an early-exit
  point `callo serve` can run, one line under the pick names the one with the highest z, its saving against the budget
  it was capped at, and the flag that opts in (`callo serve --think-converge Z`; in `report.json`,
  `early_exit.<model>.opt_in`):

  ```
  early exit (opt-in, not picked): M@ee.95 saves ~47 tokens / ~500 ms (mean) at z .95 on this data; it holds only on questions like these: serve with --think-converge 0.95 to opt in
  ```

  A profile calibrated with the [early-exit gate](calibration.md#the-early-exit-gate) measured also has a gated
  point beside each early-exit point (`model@ee<z>+g`: the questions the gate flags take the reading at the cap, the
  others the early-exit one, each plus its gate read's milliseconds). Gated points are listed in the table, the
  scatter and `report.json` (kind `eeg`; `early_exit.<model>.gate` and `.gated` hold the gate's block and the gated
  entries), marked like the early-exit points and never picked; the opt-in line names both variants
  (`gated (M@ee.95+g) saves ...`, and `add --think-converge-gate for the gated one`). A profile without gate reads
  (calibrated before the gate) simply has no gated point, and profiles whose gated points were measured with different
  gates are flagged.

  An early-exit point `callo serve` could not run as measured (no rule reaches its z, or the profile's convergence
  block is one serve refuses) gets a line naming it and why instead; its gated point goes with it. compare never reads a profile's stored pick to
  choose: it re-picks from the profile's points, so a profile whose stored pick is an early-exit point gets a budget
  or deadline pick here.
- **Limits never simulate a deadline.** When no point meets a `max` or `p99` limit but budget 0 does, the reason
  suggests a new calibrate run with `--deadlines <ms>`, since a limit only filters points.

## What is not calibrated

The budget per question (no routing or per-question adaptation; the budget is the server default or the request's),
media (text only), more than one sample per question (no self-consistency), the trace sampler itself (pinned from the
GGUF, `fixed`, or your own setting; never searched), position bias at budgets above 0, the equivalence of a cut trace
with a shorter generation under speculative/MTP decoding (measured by the audit per model, not assumed), the
temperature of a budget the profile lacks (the nearest lower calibrated one is used), and latency on any other box
(single-stream, in-callo wall time, on the calibrating host).
