# Measurements

What was measured, under which conditions, and what came out. Everything ran on one Apple-silicon machine, one model
at a time. Runs through Callo used llama.cpp master `4ebdf2c` (build 50), with `callo serve` in launch mode (Callo
starts llama-server itself; the think-budget calibrations ran through
`callo calibrate --think-budgets --menu MENU.json`, which launches the same way); the plain llama-server baselines
used the Homebrew build b9890; the Clef GGUFs in the Decision Index and think-budget comparisons ran on llama.cpp's
native Clef path, build b11379 (`1537a0a8`). Intervals are paired 95 % bootstrap intervals (cluster-resampled for
think budgets and position bias; resampled by linked group within each benchmark for the Decision Index).

Contents: [Decision Index](#decision-index) · [accuracy and calibration](#accuracy-and-calibration) ·
[the private lane](#the-private-lane) ·
[drop-in behaviour](#drop-in-behaviour) · [images](#images) · [prompt cache](#prompt-cache) ·
[think budgets](#think-budgets) · [position bias](#position-bias) · [transcription](#transcription) ·
[readouts and other slots](#readouts-and-other-slots)

## Decision Index

<picture><source media="(prefers-color-scheme: dark)" srcset="../../brand/di-dark@2x.png"><img src="../../brand/di-light@2x.png" width="100%" alt="Chart: Decision Index 0.3 public subset against mean request completion time on a log scale. Clef 27B Q4_K_XL leads at 58.0 and Clef-Flash 9B BF16 scores 54.9 at 2.4 s; Liquid AI's d1-3B Q4_K_M scores 46.1 at 1.5 s. Tiel-Coder 35B-A3B rises from 38.5 with no thinking (2.2 s) to 49.0 with a 256-token think and opt-in early exit (9.7 s); Qwen3.8-Flash-Next from 47.6 to 51.7 with early exit and the gate; Sharp-MiniCPM5-2B from 22.5 to 41.0 at a 1024-token think. Below, the five area skills and a table of every arm, including Tiel-Coder's plain think 256 and its early exit with the gate, which are not drawn."></picture>

**Setup.** The Decision Index 0.3 public subset, scored by the Decision Index kit's own, unmodified scorer: 1,480 rows
sampled (seed 20260919) from the 37 counted benchmarks, 33–34 linked groups each, in five fixed-weight areas
(Knowledge & Reasoning 25.8 %, Language Understanding 25.8 %, Retrieval & Classification 20 %, Tools & Automation
18.3 %, Arts & Human Taste 10 %). The index is the kit's balanced skill on its 0–100 scale: chance-corrected benchmark
scores averaged within the areas. Callo's arms ran through `callo serve --model` (launch mode: 65,536-token context, 4
slots, 1 reserved) with the prompt cache on (the default), each with a think profile for its readout temperatures.
Qwen3.8-Flash-Next's llama-server also ran with `-b 1024 -ub 1024 -fa on -ctk q8_0 -ctv q8_0` and the Sharp chat
template (v22.5.0; "Sharp" in a model's name marks our chat template build). Callo answers a request's questions one
at a time on the shared state, thinking per question at a budget above 0. The Clef arms answer all of a row's
questions in one joint pass. One request at a time on an Apple M2 Ultra (64 GB). Time is the request completion time
measured by the kit's client, the same for every arm, over the 1,467 rows every arm answered. Intervals come from
1,000 paired resamples of linked groups within each benchmark, the same draws for every arm, with the kit's scorer in
the loop; differences are in index points.

| arm | index [95 % CI] | Δ vs Clef-Flash BF16 [95 % CI] | mean request time | p95 |
|---|---|---|---|---|
| Tiel-Coder 35B-A3B Q4_K_XL, no think | 38.5 [34.9, 41.9] | −16.4 [−20.4, −12.7] | 2.2 s | 10.9 s |
| Tiel-Coder, think 256 | 48.8 [45.4, 52.3] | −6.0 [−9.8, −2.4] | 13.9 s | 76.5 s |
| Tiel-Coder, think 256, early exit (opt-in) | 49.0 [45.6, 52.3] | −5.8 [−9.5, −2.6] | 9.7 s | 50.5 s |
| Tiel-Coder, think 256, early exit + gate (opt-in) | 48.1 [44.7, 51.4] | −6.8 [−10.7, −3.3] | 11.4 s | 54.3 s |
| Sharp-MiniCPM5-2B Q6_K_XL, no think | 22.5 [19.9, 26.0] | −32.3 [−36.0, −28.0] | 780 ms | 3.3 s |
| Sharp-MiniCPM5-2B, think 1024 | 41.0 [37.8, 43.9] | −13.8 [−17.8, −10.5] | 18.7 s | 94.8 s |
| Qwen3.8-Flash-Next (Sharp) IQ3_XXS, no think | 47.6 [44.1, 50.7] | −7.3 [−11.5, −3.9] | 8.8 s | 35.3 s |
| Qwen3.8-Flash-Next, think 256, early exit + gate (opt-in) | 51.7 [48.2, 55.1] | −3.1 [−6.9, +0.6] | 28.5 s | 126.6 s |
| Clef 27B Q4_K_XL (our quant) | 58.0 [54.8, 61.0] | +3.2 [−0.2, +6.0] | 8.5 s | 36.6 s |
| Clef-Flash 9B BF16 | 54.9 [52.1, 57.7] | reference | 2.4 s | 10.3 s |
| d1-3B Q4_K_M (Liquid AI) | 46.1 [43.1, 49.3] | −8.8 [−12.1, −5.2] | 1.5 s | 6.2 s |

- **Thinking narrows the gap.** With no thinking Tiel-Coder scored 16.4 points under Clef-Flash BF16, at 0.93× its
  mean request time. The chart's Tiel-Coder line ends at a 256-token think per question with opt-in early exit, which
  stops the think once the answer has settled instead of using the whole budget: 49.0, 5.8 points under (CI 2.6–9.5),
  at 4.1× the time. The plain 256-token think, without early exit, scored 48.8, 6.0 points under (CI 2.4–9.8), at 5.8×
  the time. Sharp-MiniCPM5-2B, a 2 GiB model, went from 22.5 to 41.0 at a 1024-token think.
- **Qwen3.8-Flash-Next** came closest: thinking 256 tokens with opt-in early exit and the experimental gate, it scored
  3.1 points under Clef-Flash BF16, an interval that includes 0, and 4.2 points [+1.7, +6.8] above its own no-think
  point, at 3.2× the mean request time. No plain think 256 was run on the index, so this opt-in arm is its only
  reasoning arm there.
- **Clef 27B** leads, 3.2 points above Clef-Flash BF16 (interval includes 0). It ran beside CPU-heavy work for its first
  1.5 hours and was not re-timed, so its request time may read high. Clef-Flash 9B Q4_K_XL (our quant), measured but
  not drawn, scored 55.6 [52.8, 58.4], not distinguishable from BF16 (+0.7 [−0.4, +1.6]), at 2.4 s (its last half hour also ran
  beside CPU-heavy work).
- **d1-3B**, Liquid AI's 3B decision model (their Q4_K_M GGUF on llama.cpp's native decision path, commit `23deec02`),
  answers each question in its own prompt. It scored 46.1, 8.8 points under Clef-Flash BF16 (CI 5.2–12.1), at 1.5 s
  mean request time (0.62× Clef-Flash BF16's).
- **Early exit on Tiel-Coder (opt-in, experimental).** On the same draws as plain think 256, early exit at z .95 (the
  chart's point) changed the index by +0.2 [−2.0, +2.3] at 30 % less mean request time, and early exit with the gate by
  −0.8 [−2.9, +1.5] at 18 % less; the gate sent 579 of 1,471 requests (39 %) to full thinking. Neither difference is
  distinguishable from 0. Plain think 256 and the gated arm are in the table, not drawn. Callo never picks early exit;
  see [Early exit](think-budgets.md#early-exit).
- **Without BRIGHT.** 13 BRIGHT rows (17,112 to 32,805 tokens) are longer than Clef's 16,384-token prompt limit and
  count as wrong for every arm, the kit's no-truncation rule. Dropping BRIGHT with the kit's own re-weighting (the same
  draws) raises every arm by 0.8 to 2.1 points (Clef-Flash BF16 to 56.7); the order of the arms is unchanged and no
  paired interval changes whether it includes 0.
- **What this is, and is not.** A 1,480-row sample of the index's public benchmarks, scored by the kit, with every arm
  run on the same machine and client. It is not the board's Full score. The 157 rows with more than 26 options are answered by every arm
  (Callo's wide-choice readout and Clef both take up to 255; none has more).
- Tiel-Coder and Qwen3.8-Flash-Next are MoE models, and on the unified KV cache their runs are not bit-reproducible.
  Every input, build and run window: [How these figures were made](../../brand/FIGURES.md#the-decision-index-plate-dipy).

## Accuracy and calibration

**Setup.** Qwen3.5-9B Q8_0 on plain llama-server (b9890), read the way Callo reads it (no training), against Clef-Flash, a dedicated
System One classifier on the same backbone, and Kev-9B. The held-out test set has 656 questions: 480 from six public
classification benchmarks (emotion classification, MMLU, PAWS, QNLI, SciQ and offensive-tweet detection; 80 each),
96 synthetic nested boolean-rule questions, and 80 synthetic contrastive questions (40 yes/no, 40 three-level
"how late is this report" date questions). The temperature was fitted on a separate development split.

| | Callo readout | Clef-Flash | Kev-9B |
|---|---|---|---|
| six public benchmarks (n 480), accuracy | 0.827 | 0.823 | 0.833 |
| all 656 questions, accuracy | 0.797 | 0.852 | 0.854 |
| nested boolean rules (n 96), accuracy | 0.604 | 0.917 | 0.896 |

- **On the six public benchmarks there is no detectable accuracy difference:** +0.4 pp against Clef-Flash
  [−2.7, +3.5], −0.6 pp against Kev [−3.5, +2.1]. Gaps larger than about 3.5 pp are excluded. This is accuracy only:
  the Brier score there is worse than Kev's (+0.047 [+0.011, +0.083]).
- **Overall the readout is 5.5 pp behind Clef-Flash** [−8.5, −2.4], and the gap sits in the two rule-reasoning sets:
  −31.3 pp on the nested boolean rules and −10.0 pp on the contrastive set, all of it on the date questions. Both are
  multi-step reasoning packed into one forward pass; [think budgets](think-budgets.md) close most of it.
- **Calibration.** Served at T = 1 the readout is over-confident (ECE 0.080). One temperature fitted on the
  development split (T = 1.91) gives test ECE 0.035, not distinguishable from Clef-Flash's (difference −0.003
  [−0.030, +0.029]).
- **The readout is clean.** Every request stopped at a real answer slot, with no top-k misses; the smallest option
  mass on the test set was 0.9876.
- **Memory.** One llama-server of about 20 GiB serves both the agent and classification, against the agent's
  18–19 GiB plus 20–23 GiB for Clef-Flash as a second process (measured separately and summed; Clef-Flash ran in
  bf16, so part of the difference is precision).

## The private lane

Qwen3.5-9B Q8_0, a classification arriving while an agent request prefills a long prompt.

| condition | classification latency |
|---|---|
| plain llama-server, behind a 32k-token prefill | p50 39.7 s |
| plain llama-server, pinned to slot 0 with `-b 512 -ub 512`, behind a 32k-token prefill | p50 2.48 s, p95 2.96 s (agent prefill −11 %) |
| through Callo (launch mode, `-c 262144 -np 4`), under a continuous 32k-token agent prefill, 100 requests | p50 2.15 s, p95 2.85 s, 0 errors |
| through Callo, under a continuous 128k-token agent prefill, 100 requests | p50 3.29 s, p95 4.64 s, 0 errors |
| through Callo, under 5 concurrent agent decode streams | p50 0.32 s; each stream dropped from 30.7 to 9.1 tok/s (−70.5 %) |

The S1 fallback does not help when a foreign prefill holds a reserved slot: 39.2 s with the fallback, 41.0 s without
(see [Limitations](limitations.md#the-s1-fallback-helps-against-a-foreign-decode-not-a-foreign-prefill)).

## Drop-in behaviour

Same setup as the private lane. Chat, tool-call and streaming output through Callo was identical to direct. The
agent's prompt cache survived a classification between turns: turns 2 and 3 prefilled only the 31 new tokens.
`/v1/responses` was pinned and `/v1/messages` was not. A Homebrew llama.cpp build (b9890) passed the same smoke test
(launch mode, drop-in, 10 of 10 classifications, self-test).

## Images

Tiel-Coder 35B-A3B (hybrid, vision) with its BF16 projector.

- 36 maze frames (12 each at 224², 336² and 448²) with about 180 tokens of standing orders as `context`: 36 of 36
  answered correctly, mean probability on the right answer 0.999 (minimum 0.987).
- 224² frames at 2 Hz on a lane while another slot streams a generation: median 357 ms, 2.00 Hz kept, the stream at
  34.9 tok/s. Ticks that overlapped another slot's prompt prefill took a median 783 ms (5 ticks); the prefill row's
  median over all 120 ticks was 352 ms.
- The system turn was reused from the cache on every tick; the same frame read cold and warm (after 20 other frames)
  gave identical probabilities.
- Callo's readout agreed with a reference readout on 106 of 106 recorded game decisions (max |Δp| 2.95e-05).
- An image request to a server launched without `--mmproj` got 501, the same body without images 200.

## Prompt cache

Tiel-Coder 35B-A3B Q4_K_XL (a hybrid MoE), 40 real request bodies: a readout took 609 ms cached against 798 ms
with `--no-cache` (−24 %), and a warm readout equalled a cold one exactly. Details under
[Caching](api.md#caching-and-the-user-turn-delimiter).

## Think budgets

<picture><source media="(prefers-color-scheme: dark)" srcset="../../brand/frontier-dark@2x.png"><img src="../../brand/frontier-light@2x.png" width="100%" alt="Chart: accuracy against mean latency per decision on a log scale, on the same 376 test questions. Each Callo model is a line through its think budgets; Clef and Kev-9B answer in one pass and are single markers, with Clef-Flash 9B BF16 on llama.cpp (85.4 %, 286 ms) as the dashed reference. Tiel-Coder 35B-A3B reads 83.2 % at 169 ms with no thinking and 92.0 to 94.1 % at 1.7 to 1.9 s with thinking; Callo's pick is Qwen3.8-Flash-Next with no thinking, 91.0 % at 567 ms. Below, a table of accuracy and mean latency per model and budget."></picture>

**Setup.** A 376-question test set (96 nested boolean-rule, 40 date-scoring and 240 MMLU / QNLI / PAWS controls;
separate from the development set the think-budget design was first measured on), every model on every question. Each
model was calibrated with `callo calibrate --think-budgets --menu MENU.json`, which launched its `callo serve` and
llama-server (65,536-token context, 4 slots, 1 reserved) with the prompt cache on (the default); latency is
single-stream, in-callo wall time on an Apple M2 Ultra (64 GB). Clef and Kev-9B answer in one pass, without thinking:
the Clef arms are timed by a client wall per request (a fresh loopback connection, the round trip and the JSON parse),
Kev-9B in process (the model forward only, on MPS). Δ is paired against Clef-Flash 9B BF16 on llama.cpp.
Qwen3.8-Flash-Next ran with the same extra llama-server arguments and Sharp chat template as on the Decision Index.
Tiel-Coder, Dirk, Qwen3.5-9B and Gemma-4-E4B were measured before a readout fix (an end-of-turn token at the answer
slot); Sharp-MiniCPM5-2B, measured on both sides of it, moved by at most 0.5 pp. Every input, build and run window:
[How these figures were made](../../brand/FIGURES.md#the-release-plate-frontierpy).

| model | GiB | accuracy at budget 0 / 256 / 512 / 1024 | mean latency, 0 → largest budget | p50 at the largest |
|---|---|---|---|---|
| Tiel-Coder 35B-A3B Q4_K_XL | 20.8 | 0.832 / 0.920 / 0.936 / 0.941 | 169 ms → 1.9 s | 1.4 s |
| Qwen3.8-Flash-Next (Sharp) IQ3_XXS | 70.6 | 0.910 / 0.920 (budgets 0 and 256 only) | 567 ms → 5.3 s | 5.5 s |
| Dirk (Qwen3.8 27B) Q4_K_XL | 16.4 | 0.830 / 0.918 / 0.939 / 0.941 | 803 ms → 8.7 s | 7.4 s |
| Qwen3.5-9B Q8_0 | 9.1 | 0.763 / 0.774 / 0.883 / 0.928 | 203 ms → 12.3 s | 12.4 s |
| Sharp-MiniCPM5-2B Q6_K_XL | 2.0 | 0.577 / 0.710 / 0.827 / 0.870 | 67 ms → 2.7 s | 2.3 s |
| Gemma-4-E4B UD-Q4_K_XL | 4.8 | 0.747 at budget 0 only (it thinks in a format Callo cannot read) | 166 ms | 152 ms |
| Clef-Flash 9B BF16, llama.cpp (reference) | 16.9 | 0.854, one pass | 286 ms | 261 ms |
| Clef-Flash 9B, Cloudflare's release code (transformers, bf16, MPS) | 17.8 | 0.856, one pass | 574 ms | 551 ms |
| Clef 27B Q4_K_XL (our quant), llama.cpp | 16.2 | 0.918, one pass | 1.0 s | 924 ms |
| Kev-9B (trained head) | 18.1 | 0.859, one pass | 704 ms | 620 ms |

GiB is the file size; Qwen3.8-Flash-Next's counts both shards of its split GGUF (the n-gram shard is read lazily from
SSD), Kev-9B's counts its base model.

- **Without thinking, Tiel-Coder is not distinguishable from Clef-Flash; a short think passes it.** With no thinking
  Tiel-Coder is not distinguishable from Clef-Flash BF16 (−2.1 pp [−6.6, +2.4]) at 0.59× its mean latency, both on
  llama.cpp; at budgets 256 and 512 (136–153 thinking tokens on average) it is +6.6 to +8.2 pp above it at 1.7 to 1.8 s mean. Clef-Flash's release code took 574 ms,
  2.0× the llama.cpp run's 286 ms: a gap of engines, not of models.
- **Callo's pick** (`callo compare`'s default: the fastest point not distinguishable from the most accurate, on the
  mean) is Qwen3.8-Flash-Next with no thinking: 0.910, +5.6 pp [+1.3, +10.4] above Clef-Flash BF16, at 567 ms. It is a
  70.6 GiB model, run on a 64 GB machine with its n-gram shard read lazily from SSD; by default `callo compare` does
  not weigh model size (`--max-memory` filters on it). On the Decision Index the same no-think point scored 47.6, 7.3
  points under Clef-Flash BF16 [−11.5, −3.9].
- On the nested boolean-rule questions, thinking to 1024 tokens gained 30.2 pp [+18.8, +43.8] on Tiel-Coder, 34.4 pp
  [+22.9, +44.8] on Dirk, 38.5 pp [+26.0, +51.0] on Qwen3.5-9B and 39.6 pp [+27.1, +52.1] on Sharp-MiniCPM5-2B; on the
  controls the gains were 2.9 pp [−0.8, +7.1], 3.3 pp [+0.0, +6.7], 5.4 pp [+2.1, +9.2] and 17.5 pp [+10.4, +24.6].
- Thinking pays very differently by model: at a 1024 budget Tiel-Coder closes its trace early (160 tokens on
  average), Dirk after 194 and Sharp-MiniCPM5-2B after 405, while Qwen3.5-9B uses 752.
- One trace per question serves every budget's measurement: a trace cut at a smaller budget (256 or 512) matched a
  real generation at that budget token for token on every audited question (48 per model and budget, or every trace
  long enough to cut: 11 for Tiel-Coder and 9 for Dirk at 512). The readouts after it agreed on the argmax on all of
  them but one per budget on the MoE Tiel-Coder (47 of 48, 10 of 11; median |Δp| 6.5e-3 at 256), and on all of them
  on Dirk, Qwen3.5-9B and Sharp-MiniCPM5-2B (median |Δp| below 3e-5). Qwen3.8-Flash-Next, calibrated at 0 and 256
  only, has no smaller budget to cut to.
- calibrate's latency model matched the audit's regenerated traces to a median error below 1 % on every audited model
  and budget. In an earlier calibration of the same test set with the prompt cache off, a served check through
  `/v1/systemone` at budget 1024 covered the p50 only: it was within 1.2 % (Tiel-Coder) and 9 % (Qwen3.5-9B) of that
  profile's p50, over 48 questions each.
- The fitted readout temperature per budget improved NLL at every positive budget on every model that thinks (for
  example Tiel-Coder at 1024: 0.566 → 0.239 at T 3.17).
- **Early exit is not plotted:** it is opt-in and never picked. At think 256, z .95 with the gate, Qwen3.8-Flash-Next
  read 0.926 at a 2.1 s p50 against 0.920 at 5.5 s for its plain think 256. Tiel-Coder's figure comes from its own
  early-exit calibration of the same questions: 0.923 at a 1.0 s p50, against 0.923 at 1.4 s for plain think 256 in
  that calibration (0.920 in the table's run). See [Early exit](think-budgets.md#early-exit).

## Position bias

Tiel-Coder and Qwen3.5-9B Q8_0, fitted on calibration questions and judged on held-out test questions from ten public
datasets (AG News, ARC, BoolQ, CommonsenseQA, DBpedia-14, IMDB, MNLI, OpenBookQA, TREC, Yelp), with pass criteria
fixed before the run; measured with `--no-cache`. The run confirmed the estimator.

| model | held-out accuracy raw → debiased | NLL raw → debiased | rotation consistency |
|---|---|---|---|
| Tiel-Coder | 0.905 → 0.906 | 0.3443 → 0.3436 | 0.900 → 0.900 |
| Qwen3.5-9B Q8_0 | 0.886 → 0.888 | 0.3701 → 0.3682 | 0.890 → 0.892 |

For every option count with a profile entry and a detectable held-out bias, debiasing reduced the held-out position
deviation; on the rows with an entry, accuracy rose 0.2 pp and NLL fell. On Tiel-Coder only (not run on Qwen3.5-9B):
loading a profile left the upstream call sequence identical and added 0.0004 ms of CPU per readout, and the
calibration guard refused under a resident 300-token prompt and under a continuous decode, and passed once the load
was removed. The gains are small by design; see
[Limitations](limitations.md#position-bias-profile).

## Transcription

Gemma-4-E4B (UD-Q4_K_XL, `mmproj-BF16.gguf`): 20 utterances spoken by macOS `say`, each sent through Callo's
translated `/v1/audio/transcriptions` and through llama-server's own endpoint, with every slot erased before every
request, at temperature 0. Prompts were byte-identical (20 of 20), token counts equal, the text identical (20 of 20),
and every translated request landed on a non-reserved slot. Mean word error rate against the spoken text was 0.030 on
both; p50 latency 348 ms through Callo, 340 ms direct.

## Readouts and other slots

How much a readout moves with what other slots hold, under unified and per-slot KV, is measured under
[Readouts and unified KV](limitations.md#readouts-and-unified-kv).
