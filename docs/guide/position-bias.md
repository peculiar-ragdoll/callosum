# Position bias

A letter readout carries a prior over answer *positions* that has nothing to do with the content: some models put
more probability on option C, whatever the options say. `callo calibrate --bias --data YOUR.jsonl` measures that
prior on your own questions: it reads each question under every cyclic rotation of its options (the rotation design
of PriDe, Zheng et al., "Large Language Models Are Not Robust Multiple Choice Selectors", ICLR 2024), so every option
visits every position, and takes the mean served probability at each position. It needs no labels. `callo serve
--position-bias` then divides that prior out of every eligible readout: one subtraction per option in log space, no
extra upstream call. The prior is measured on your questions because position bias depends on the questions: a
correction measured on one set of questions did not carry over to others in Callo's measurements (see
[Limitations](limitations.md#position-bias-profile)).

Debiasing is off by default: without `--position-bias` readouts are not corrected. The startup probe erases the 16
tokens it leaves in the slots it used (when llama-server has `--slot-save-path`, which launch mode passes); under
`-kvu` that residue is the kind of load [Readouts and unified KV](limitations.md#readouts-and-unified-kv) measured
shifting readouts by up to 0.20.

```
callo serve --model M.gguf                                   # 1. serve at T = 1, no profile
callo calibrate --bias --data mine.jsonl --out bias.json     # 2. measure on your own questions
callo serve --model M.gguf --position-bias bias.json         # 3. restart with the profile
callo calibrate --data labelled.jsonl --out T.json           # 4. fit T on the DEBIASED readouts
callo serve --model M.gguf --position-bias bias.json --temperature <T from step 4>   # 5. serve with both
```

Debias first, then the temperature: the served probabilities are softmax((log-marginal − offset) / T), the prior is
measured at T = 1, and `--temperature` must be refitted after any new profile (the temperature JSON records the
profile's sha256 as `position_bias_sha256`, but `--temperature` is a bare number, so nothing enforces this). At about
0.3 s per idle readout on Qwen3.5-9B, step 2 costs one readout per option of every eligible question in your file
(500 four-option questions: 2,000 readouts, about 10 min); it prints the plan, then a measured ETA after 20 readouts.
`callo calibrate --bias` without `--data` reads the built-in set as a quick diagnostic and writes no profile.

```
callo calibrate --bias [--url URL] [--data FILE.jsonl] [--out PROFILE.json] [--raw RAW.jsonl] [--resume]
[--min-questions 24] [--max-per-n N] [--seed 0] [--allow-busy]
```

## Preconditions (exit 2)

The server must be a Callo server with a `model_key`, serving at T = 1 with no profile loaded, not thinking by default
(`--think-budget` 0), and Callo's engine on (not a native decision model).

## What gets corrected

Only `choice` and `noul` questions whose labels are exactly A, B, C… in order (*canonical*: every `noul`, and for
example a `choice` none of whose keys is a single letter, or one with keys `a, b, c` in that order); only option
counts the profile has an entry for; and only the readout kinds (`plain`, `context`, `image`, `audio`, `video`,
`mixed`) your `--data` contained. A readout has one kind: `plain` carries no `context` and no media, `context` carries
`context` and no media, `image`, `audio` or `video` carries that one media kind (with or without `context`), and
`mixed` carries more than one. So a file of plain records (no `context`, no media) corrects plain readouts only,
**not `context`, image, audio, video or mixed readouts**; include such records to have them corrected (the summary's
note names the kinds a profile leaves out). `score` questions, single-option questions and non-canonical ones (keys
`b, x, go, a` render `B, X, C, A`) are never corrected, and neither are questions above 26 options or deliberate
readouts. `/callo/stats.position_bias` counts what was applied and what was skipped, by reason (see
[`/callo/stats`](stats.md)).

## Your data (`--data`)

One System One request per line, as you would send it to `/v1/systemone`:
`{"state": ..., "questions": {"q1": {"type": "choice", "instructions": ..., "criteria": {...}}}}`, plus any
`images`, `context`, `audio` or `videos` the request carries, and optionally `_meta` (`id`, `source`, `group_id` or
`pair_id`, `variant`) and a per-question `label` (with labels the summary adds an in-sample accuracy, NLL and ECE
before and after; the fit never uses them). Use questions like the ones you will serve: the same instructions and
option sets, the same kinds. Every top-level key of a record except `questions`, `label`, `src` and `_meta` is sent
with each rotation as is. calibrate skips, and counts, the questions it cannot use: `score`, single-option and
non-canonical questions, questions with an option that is a bare letter or refers to other options ("all of the
above", "option B"), records whose `_meta.variant` is not `clean`, and questions with more than 26 options. An
option count needs at least `--min-questions` (24) questions to be tested at all, and the measured prior is only as
good as the number behind it: aim for 100 or more per option count you serve. Every eligible question is read by
default; `--max-per-n N` caps each n, sampling evenly across `_meta.source`.

## The built-in set `builtin-v1`

A diagnostic: `callo calibrate --bias` without `--data` reads it (48 questions per option count n = 2…8, 336
questions, 1,680 readouts), prints which n show a shared position bias on generic questions, and writes no profile
(`--out` without `--data` exits 2). Built-in means the set's bytes: a copy passed as `--data` is the same diagnostic.
Without `--raw` its readouts go to `./callo-bias-diagnostic.raw.jsonl`, so they never block a later `--data` run's
default raw file, and its summary shows raw figures only. "No shared position bias detected" is not proof of none:
biases of opposite sign in different questions cancel. n = 2, 3, 4 mix MMLU validation and Banking77 train half and
half; **n = 5…8 are Banking77 only, one domain** (customer-banking intents). Nothing comes from an upstream test
split. Data licences: MMLU is MIT, Banking77 is CC-BY-4.0 (modified: distractors sampled, intent names reformatted);
see `NOTICE`.

**Where it comes from.** The wheel ships code only (licence `Apache-2.0`), so the set is not in it. It is published
as the Hugging Face dataset
[`peculiar-ragdoll/callo-bias-set`](https://huggingface.co/datasets/peculiar-ragdoll/callo-bias-set) (the same file
as `bench/data/bias_set.jsonl` in this repository), pinned in Callo by commit and sha256. The first `callo calibrate
--bias` without `--data` downloads it (about 247 KB; one stderr line says so), checks the sha256 and caches it in
`~/.cache/callo/bias_set/`; later runs read the cache. Nothing else ever touches the network for it: `callo serve`
with a profile, `--data` runs, `--help` and imports never fetch, and "is this the built-in set" is a comparison
with the pinned sha256. **Offline:** download the file once (the URL is in the error message), then
`CALLO_BIAS_SET=/path/to/bias_set.jsonl callo calibrate --bias ...`; Callo refuses a file whose sha256 differs. A
failed download exits 1 with the URL and this route.

## What is kept

For each n, a cyclic-shift permutation test (2,000 draws, one shift per cluster: `_meta.group_id`, else
`_meta.pair_id`, else the question) with Holm across the n tested (family-wise α = 0.05); an n needs at least
`--min-questions` (24) questions read under every rotation to be tested. A dropped n is not corrected, with the reason
in the profile's `dropped`. The test runs on each question's mean probability per position over its rotations; a
rejection means the questions share a position preference, which is what one per-n prior can correct. A question with
an option outside the top-100 dump in some rotation is fitted all the same (that option is served at an upper bound,
and in probability space it can move the mean by no more than that bound); the summary counts such questions per n.
The printed 95% intervals are per position, not simultaneous, and are not stored.

## Profile format 2

A profile records `"callo_position_bias": 2` and `"estimator": "meanP"`. `callo serve` refuses (exit 1) a format-1
profile, which an earlier estimator (PriDe's log-space prior) wrote: measure again with `--data`. It also refuses a
profile marked as fitted on the built-in set.

## A busy server refuses

Exit 2. A readout can move with what other slots hold (see
[Readouts and unified KV](limitations.md#readouts-and-unified-kv)), and the profile would learn that as bias.
calibrate reads `/callo/stats` before the first readout, every 50 readouts and after the last, and refuses when:
`/slots` cannot be read (`--no-slots`, or a fetch error); another slot is processing; another slot holds tokens (under
unified KV, in attach mode, or under `--no-kvu` with a set holding any non-plain kind); System One readouts arrive
that it did not send; or traffic through Callo reaches any other route that is not read-only (chat and completion
endpoints, `POST /slots/{id}` save, restore and erase actions). The read-only routes `/health`, `/v1/health`,
`/props`, `GET /slots`, `/metrics`, `/models`, `/v1/models`, `/tokenize`, `/detokenize`, `/apply-template` and
`/callo/stats` are ignored, so a health-check poller never makes calibrate refuse.

The message names the fixes that work for your setup: erase the named slots (offered when Callo's own probe could
erase, that is with `--slot-save-path`), stop the agent traffic, or launch with `--no-kvu` (plain-text sets under
unified KV). When the only tokens held are Callo's own startup-probe residue, stopping agent traffic cannot clear
them, so the message names the restart instead: in attach mode, restart llama-server with `--slot-save-path DIR` and
then `callo serve`; in launch mode, restart `callo serve`. `--allow-busy` records the event and carries on, and the
profile is marked `busy` (`serve` warns when it loads one). calibrate never erases a slot itself. Every request
carries `X-Callo-Conversation: callo-calibrate`, so one slot serves the whole run; the run stops with exit 2 if that
slot changes, if Callo restarts (`/callo/stats` `instance` changes), or if the model, its `provenance`, the
temperature, a loaded profile, the KV layout or the reserved S1 slot count changes.

## `--raw` and `--resume`

Every readout is written to `--raw` as it arrives (default: `--out` with its extension replaced by `.raw.jsonl`, or
`./callo-bias.raw.jsonl` without `--out`, or `./callo-bias-diagnostic.raw.jsonl` for the built-in diagnostic). The
file is created only once the first server-state check passes, so a run refused before its first readout writes
nothing, and fixing the server and rerunning the same command starts clean (with `--resume` too: it then prints
`no raw file at <path>; starting a fresh run`). An existing `--raw` without `--resume` exits 2; `--resume` reads only
the missing rotations and refuses when anything in the raw header differs (`model_key`, `provenance`, the server's KV
layout, temperature and reserved slot count, the set's name and sha256, `--seed`, `--max-per-n`); a Callo restart
between sessions is fine, and each session starts with a fresh check. A raw file read through a `callo serve` with the
cache off differs in `provenance` from a default serve, and the refusal names the fix: restart serve with
`--no-cache`. A refusal after readouts keeps them and the failed check: `--resume` continues, but the profile is then
marked `busy`; remove the file to start clean. A 503 or a connection error is retried 3 times (1, 4, 16 s); any other
failure exits 1 with the raw file intact. A run that read everything but could not write `--out` refits without
reading again: rerun with `--resume`, a writable `--out` and the same raw file (pass `--raw` explicitly if `--out`
changes, since the default `--raw` follows `--out`). `bench/bias_refit.py` (in a source checkout) refits a profile
offline from a raw file.

## Output

With `--out` the profile goes to that file and the summary to stdout; without it the profile JSON goes to stdout and
the summary to stderr. The built-in diagnostic writes no profile and prints its summary to stdout. The summary gives,
per n, kept or dropped with the reason (the diagnostic says significant rather than kept), `prior_n` with its display
intervals, `meanP_n` raw and debiased, how many questions had a floored option (fitted all the same) and the rotation
consistency raw and debiased; the diagnostic's figures are raw only. Exit codes are in
[Exit codes](exit-codes.md#callo-calibrate---bias).

## Model identity

A profile carries `model_key`, a sha256 over the GGUF metadata (except `general.*`, `quantize.*`, `split.*` and the
chat template) and over the name, shape, type and the first and last 64 KiB of every tensor that reaches the readout
(the whole tensor up to 128 KiB). It leaves out the MTP head: it assumes the last `<arch>.nextn_predict_layers` blocks
are the MTP head, which holds for `qwen35`/`qwen35moe` GGUFs, so on an architecture that puts its nextn layers
elsewhere the key would hash the wrong blocks. Re-grafting or re-typing an MTP head, re-embedding the chat template,
and splitting or merging a GGUF keep the key; serving another model with the profile exits 1, refused as soon as the
key is computed, before the startup probe touches any slot. The profile's `provenance` records what can move a
readout without changing the model (the prompt contract's hash, the chat template's hash, the llama.cpp build, the
projector, the user-turn delimiter, `--cache`, whether `--cache` readouts send the delimiter (`cache_delimiter`), and
launch mode's passthrough flags as `server_args`); when any of those differs at startup `serve` only warns, one line
per field. A profile measured with the cache off (`--no-cache`) reads as it was measured only under `--no-cache`
(otherwise `serve` warns that `cache_prompt` differs).

## In process

`callo.engine.local_engine(upstream, gguf=..., position_bias=PATH)` applies a profile in process, with the same
checks. local_engine reads with the cache off (the uncached payload), so a profile measured on a default
`callo serve` (cache on) warns that `cache_prompt` differs; measure with `callo serve --no-cache` to match it.
