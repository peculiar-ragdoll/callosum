# The System One API

`POST /v1/systemone` takes a TypeSafe-style System One request and answers it from the model llama-server is already
serving. Callo renders the request Kev-style, applies the model's own chat template with thinking off, reads the
next-token distribution (`n_probs` 100) at the answer slot, sums every surface spelling of each option letter, walks
over answer-neutral prefixes, and renormalises over the options. The response has the System One shape: `model`,
`answers`, `usage`.

```
curl -s localhost:8180/v1/systemone -d '{"state": "The package arrived two weeks late.",
  "questions": {"tone": {"type": "choice", "criteria": {"positive": null, "negative": null, "neutral": null}}}}'
```

Contents: [request fields](#request-fields) · [the prompt contract](#the-prompt-contract) ·
[caching](#caching-and-the-user-turn-delimiter) · [lanes](#lanes) · [errors](#errors) ·
[diagnostics](#diagnostics) · [questions above 26 options](#questions-above-26-options) ·
[in process](#in-process) · [no cancellation](#no-cancellation)

Related pages: [images, audio and video](media.md), [think budgets and deadlines](think-budgets.md),
[position bias](position-bias.md), [`/callo/stats`](stats.md).

## Request fields

Callo mirrors llama.cpp's native image extension and adds a few fields of its own:

```json
{"state": "",
 "images": ["data:image/png;base64,iVBORw0KGgo..."],
 "context": "Standing orders: eat nearby dots; move away from any ghost within three tiles.",
 "questions": {"move": {"type": "choice", "instructions": "Which way should Pac-Man go next?",
                        "criteria": {"up": null, "down": null, "left": null, "right": null}}}}
```

- **`images`**: up to 8 data URLs (`data:image/...;base64,...`); `null` means none. `image_url` content parts inside a
  `state` that is a list of chat messages, or `{"messages": [...]}`, are taken out of the state and count towards the
  8. Images from `images` come first, then the state's parts in document order. `http(s)://` URLs are not
  supported. The 8 counts every medium (images, audio and videos together), and an image whose bytes are an audio
  file is a 422.
- **`context`** (Callo only, not part of System One): text placed in the system turn, before the images, so the
  prompt cache reuses it from one request to the next. It works on text-only requests too.
- **`audio`** and **`videos`** (Callo only): see [Images, audio and video](media.md).
- **`think_budget`**, **`think_deadline_ms`** and **`think_converge`** (Callo only): see [Think budgets and
  deadlines](think-budgets.md) and [Early exit](think-budgets.md#early-exit).
- **`think_converge_gate`** (Callo only, experimental): `true`, `false`, or `null` for the server default
  (`--think-converge-gate`); anything else is a 422. When early exit runs for the request, one budget-0 question asked
  before thinking decides whether it may act. See [Early-exit gate](think-budgets.md#early-exit-gate-experimental).
- The upstream needs vision for images: start it with `--mmproj`, or in launch mode `callo serve --model M.gguf
  --mmproj mmproj.gguf`. An image request to an upstream without vision gets 501.

## The prompt contract

**The rendered System One prompt is a public contract.** Changing it is a breaking change:

```
system:  SYSTEM_MAIN  [+ "\n\nContext:\n" + context]
user:    one image part per image, then one text part:
           Input:\n<state>                   (omitted when the state is empty)
           Question: <instructions>          (omitted when there are none)
           Options:\nA. <option>\nB. <option> ...
           Reply with one letter: A, B, ...
         above 26 options:
           Options:\nAA. <option>\nAB. <option> ...
           Reply with one two-letter label: AA, AB, ...
```

`SYSTEM_MAIN` is "You are a classifier. Read the input and the question, pick the best option, and reply with that
option's letter only: no words, no punctuation, no explanation." `demo/prompts/system_main.txt` is a pinned copy.
The text blocks are separated by a blank line. The state, the instructions, the options and the context are
special-token escaped: `<|x|>` becomes `<¦x¦>` (U+00A6 BROKEN BAR) wherever x is one or more ASCII letters, digits
or underscores; nothing else is changed. A text-only user turn is a plain string, not a list with one text part.

- **Labels.** A single-letter ASCII choice key keeps its own letter, uppercased, wherever it sits (of two keys that
  differ only in case, the first keeps it); the other keys take the next unused letters in order. Criteria
  `{"b": null, "x": "ex", "go": "move", "a": null}` render as `B. b`, `X. ex`, `C. go: move`, `A. a` and
  `Reply with one letter: B, X, C, A.` The Reply line ends with a period.
- **Above 26 options** (up to 255): every option takes a fixed two-letter label, in order, from Callo's table
  of 255 pairs (`systemone.WIDE_LABELS`: `AA AB AC AE AF ...`, each one token bare and after a space on the
  vocabularies Callo serves, two-letter English words left out). No key keeps its own spelling: every option shows
  `key` or `key: description`, score levels included. The last line lists every label:
  `Reply with one two-letter label: AA, AB, ..., BO.` The labels and the prompt are the same on every model; a
  model whose vocabulary does not carry a label as one token is read in two passes
  ([coverage](#questions-above-26-options)). Questions of up to 26 options never use this form.
- **`<option>`** is the key, or `key: description` when the criterion has a description; a single-letter key that
  keeps its letter shows only the description. `noul` questions read `A. no` / `B. yes`, each with
  `: description` when given, and `score` levels are lettered A, B, ... in order.
- **`<state>`** is a string or number as given, a boolean as Python's `True` / `False`, `null` as empty; objects and
  arrays are flattened Kev-style into `key: value` lines and `- item` lists, after the image parts are taken out. A
  messages state whose content was only images still renders its skeleton (`- role: user\n  content:\n`), so its
  `Input:` block is not omitted.

## Caching and the user-turn delimiter

Media requests and requests that carry `context` always send `cache_prompt: true` and, when Callo has a user-turn
delimiter, a `message_delimiters` entry for it, whatever `--cache` says. Without the cache every divergence-walk pass
re-encodes the image, and a hybrid model (Qwen3.5, Tiel-Coder) can reuse the system turn only from the checkpoint the
delimiter creates. `--cache` governs requests with neither media nor `context`: **on by default**, their passes send
`cache_prompt: true` and the same delimiter entry, so on hybrid models the reuse begins at the user turn; `--no-cache`
turns it off, and their passes send the uncached payload (`cache_prompt: false`).

Measured on Tiel-Coder-35B-A3B Q4_K_XL (a hybrid MoE) with 40 real request bodies: a warm readout equals a cold one
exactly (Δ logprob 0, the argmax agreeing on 42 of 42 questions), the warm first pass reused exactly the system head
(`cache_n` 212, the user-turn start) on 40 of 40, and a readout took 609 ms against 798 ms with `--no-cache` (−189 ms,
−24 %; the first pass 572 against 744 ms). The cached server reads a little differently from a `--no-cache` one, but
not because of reuse (a cold readout drifts the same): median |Δp| 0, mean 0.25 pp, max 6.4 pp on one question, the
argmax agreeing on 42 of 42; that is the split prefill's batching numerics, inside the MoE bar of a median at most
1e-2. Cached KV moves logprobs at about the 1e-2 level, so media and `context` requests, and plain ones under
`--cache`, are not bit-stable; pass `--no-cache` when you need bit-stable plain readouts.

On hybrid (Gated-DeltaNet) models the reuse begins at the user turn: without the delimiter llama.cpp checkpoints only
near the end of the prompt, so a prompt that differs earlier is re-prefilled whole; with it, what precedes the user
turn is reused.

**The user-turn delimiter** is derived at startup from the model's own chat template (`<|im_end|>\n<|im_start|>user`
on ChatML templates) and validated on a rendered prompt; `--user-delimiter STR` pins it, and a pinned value that
fails validation is still used, with a warning. When none can be derived, or a derived one fails validation, Callo
warns and uses none, and a hybrid model re-prefills the system turn on every cached request.

## Lanes

With `--s1-slots k` of 2 or more, `X-Callo-Conversation` on `/v1/systemone` binds a lane to its own reserved slot on
first sight (the lowest free one, else the least recently used lane's), so traffic with different system prefixes
keeps separate cached checkpoints. The header value is stripped; an empty or blank value names no lane. A lane whose
slot is busy with Callo's own request waits for that slot. When the slot holds traffic Callo did not send, the S1
fallback can still move the request to the lowest-id idle slot, which may be another lane's slot or an agent slot.
Use at most `--s1-slots` concurrent conversations: with more distinct `X-Callo-Conversation` values than reserved
slots, the lanes evict each other on every request and no lane's cache ever stays warm (worse than plain round robin),
and Callo logs a warning the first time that happens. Without the header, or with `k = 1`, requests round-robin over
the reserved slots.

## Errors

| Condition | Status |
|---|---|
| a malformed `images` / `audio` / `videos` entry or state media part, more than 8 media items, bytes that do not match the declared kind, an `audio_url` or `video` part, a `context` that is not a string, a media request whose text contains the upstream's media marker | 422 |
| media of a kind the upstream lacks: images without vision, audio without audio, video without video (or with video turned off at startup) | 501 |
| the upstream could not decode an item ("Failed to load image or audio file", at any status) or refused a data URL's shape | 422 |
| the upstream refused a media part type ("image/audio/video input is not supported") | 501 |
| a media request whose prompt, media tokens included, exceeds the upstream's context (`exceed_context_size_error`) | 413 `{"detail": "the request with its media exceeds the context size; lower --video-fps / --image-max-tokens or use a shorter clip (upstream: ...)"}` |
| any other upstream 4xx on a media request; a template that did not render the media parts | 500 |
| any other upstream 5xx on a media request; the upstream unreachable | 503 |
| a body over 64 MiB | 413 |
| an unexpected error inside Callo | 500 `{"detail": "internal error: ..."}` |

Requests without media keep the text-only mapping. **A 501 is a server configuration fault, not a transient one:**
the upstream needs an `--mmproj` that takes that kind (for video, also ffmpeg and ffprobe on llama-server's `PATH`;
see [Images, audio and video](media.md)). **501 means the upstream cannot serve that modality, so clients must not
retry it: retrying cannot add the missing `--mmproj`.** It is still a 5xx, so a client that drops 4xx and retries
every 5xx (Kev's `RemotePredictor`) retries it like a 503 unless it exempts 501; the 422s and the 413 for an oversized
media prompt are the request's own fault and are not retried.

Think budgets and deadlines add their own statuses: see [Think budgets](think-budgets.md#statuses).

## Diagnostics

Every 200 that Callo's engine answers on `/v1/systemone` carries `X-Callo-Timings` (a native decision-model upstream
is passed through without it), a JSON list with one object per question: `q`, `kind` (`plain`, `image`, `context`,
`audio`, `video` or `mixed`), the first upstream pass's `cache_n`, `prompt_n` and `prompt_ms`, then `tokens`, `stop`,
`ms`, `miss` (how many options were absent from the top-100 dump (top-1000 above 26 options) and served at its last
logprob, an upper bound) and `passes` (the upstream scoring calls the readout made: 1 plus the walk's steps, plus any
split passes above 26 options, 0 for a single-option question) (about 145 bytes per question; ASCII JSON, so any
question id is a legal header value). The three cache fields are `null` for the `plain` kind and for a single-option
question. The header is capped at 4096 bytes: it keeps the leading entries in request order and, when any are
dropped, ends with `{"truncated":k}`, k the number omitted; when not even the first entry fits it is
`[{"truncated":n}]`. Think budgets and deadlines add their own fields: see
[Think budgets](think-budgets.md#observability).

`Engine.answer_request(body, readouts=[])` gives an in-process caller the same numbers, uncapped. `/callo/stats`
adds, under `s1`, the image and `context` request counts, image latency p50/p95, mean `cache_n` / `prompt_n` per kind
and `retok_stops`; `eog_stops`, the readouts whose answer slot was dominated by an end-of-generation token (read
there, stop `eog`: the walk never reads past the end of the turn, and `option_mass` says how little the model meant to
answer); `reasks`, the extra requests made for readout passes whose probabilities llama-server withheld (its greedy
pick was an incomplete UTF-8 piece, banned on the re-ask), counted on readouts that completed (one that ended in a
ReadoutError is not counted); and under `router`, `lanes` and `s1_lane_busy`.

## Questions above 26 options

Up to 255 options; the labels are under [the prompt contract](#the-prompt-contract). Questions of 26 or fewer options
are read the normal way; a wide question differs in five ways.

- **Coverage.** At startup Callo counts how many leading labels of its table this model's vocabulary carries as one
  token both bare and after a space, publishes that count N as `/callo/stats.wide_coverage`, and logs
  `wide-choice: N of 255 two-letter labels are single tokens; questions up to 255 options`. The Qwen3.5 / Qwen3.8,
  Gemma 4 and MiniCPM5 vocabularies cover all 255; Qwen2 / Llama 3 cover 155; Spark covers 9. A question with more
  options than N is still read: a label that is not one token (on Spark, 47 of the 255, two of them with no token at
  all; on Qwen2 / Llama 3, 5) is read in two passes, its first letter and then its second letter after it (the
  readout below), and the log line ends `(beyond N, a label that is not a single token is read in two passes)`. Only
  a label whose letters are not tokens themselves cannot be read; no vocabulary checked so far has one.
  `/callo/stats.wide_readable` is how many labels the model reads (255 on all of them); a vocabulary that reads 26
  or fewer logs `(wide questions are refused)` instead. A question with more options than `wide_readable` is a 422
  (`this model's tokenizer can read N of Callo's two-letter labels (the next one needs a letter that is not a single
  token); the question has M options`), before any upstream call, on `/v1/systemone` and on the think-probe alike.
  Before Callo 0.3.1 every question past `wide_coverage` was that 422.
- **The readout.** Every pass asks for the top 1000 tokens (`n_probs`), so a label absent from the dump is bounded
  by 1/1000. A token that spells only a label's first letter (`A`, ` A`, `**A`) counts toward the walk's stop but
  never toward an option; when that first-letter mass exceeds 0.01, extra passes read the second letter after the
  likeliest such tokens and credit the labels exactly: P(first letter) x P(second letter | first letter). At most 2
  extra passes, plus one for each first letter that leads a label the vocabulary does not carry as one token (on
  Spark, a 77-option question has 6 such labels, led by A, B and D: at most 5). Each such letter keeps its pass:
  the passes go to the likeliest tokens first, but the last ones are held for a letter of that kind whose mass is
  above 0.01 and still unread. A label carried as one token keeps its one-token reading and adds any credit to it;
  a label with no token at all reads its credit alone. What stays unexpanded is reported as unread mass, never
  spread over the labels. A question within the coverage reads exactly as before.
- **Diagnostics.** Its `X-Callo-Timings` entry adds `unread` (the unread first-letter mass, 6 decimals) and `split`
  (the extra passes, also counted in `passes`) after `passes`; a narrow entry is unchanged. `miss` is routinely
  non-zero on a wide question (on Tiel, 0 to 53 of 77 or 151 labels sit below the top 1000). `/callo/stats.s1` counts
  `wide_readouts`, `wide_split_passes`, `wide_unread_over` (readouts with unread mass above 0.01) and
  `wide_uncovered_readouts` (readouts of a question past `wide_coverage`).
- <a id="wide-latency"></a>**Latency and deadlines.** A wide question's budget-0 readout costs about 3x a 10-option one on Tiel (537/769 ms
  p50/p95 against 185/197 ms): the option list is prefilled; split passes were not the cost. Each split pass is one
  more next-token read on the prompt already in the slot's cache (the prompt plus one token), so it costs one
  decode step and a reply with the top 1000 logprobs, not a prefill; with `--no-cache` it re-prefills the prompt like
  any walk step. On hybrid and sliding-window models a pass that diverges from the previous one (a second split pass
  after another first letter) restores a context checkpoint, or re-prefills from the last one, so the cost there
  depends on the model and is measured per model. Past the coverage a readout makes one split pass per first letter
  that holds real mass (often one: the letter of the label the model picks), up to the cap above. Against a fake
  upstream (no model, replies with the top 1000 logprobs over localhost), Callo's own side of a 77-option readout is
  about 2.3 ms for one pass and about 1.2 ms more per split pass (7.3 ms with 4 split passes); on a real model each
  pass adds its decode step on top.
  Under a request deadline these readouts share the server's one budget-0 latency window with narrow ones, and
  `Estimators.c0()`, the time a budget-0 readout is assumed to need, is that window's p95. A deliberate question under
  a deadline waits for its slot's S1 section only until its share's end minus c0, then is read at budget 0. So once
  wide readouts are more than 5 % of the last 2,000, c0 rises toward the wide p95 (on Tiel, from about 200 to about
  770 ms) for every deadline request on the server, and narrow deadline questions stop waiting, and fall back to
  budget 0, that much sooner. Callo does not scale c0 per question; keep wide and deadline-sensitive narrow traffic on
  separate servers if that matters.
- **No position-bias correction** above 26 options: such a readout reports `bias` `wide`
  (`/callo/stats.position_bias.skipped.wide`). `callo calibrate --bias` still skips those questions
  (`too_many_options`), and `callo calibrate --think-budgets` skips, before probing, the ones above what the server
  reads (`wide_readable`; on a server from before Callo 0.3.1, `wide_coverage`; 26 for a server that reports
  neither), counted as `too_many_options` in the profile's `comparable.data.skipped`. Bias and think profiles record
  the wide readout's settings as `provenance.wide_contract`; a profile without it is never warned about, and one whose
  contract differs from the server's gets a startup warning. On a model that does not cover all 255 labels the
  contract also records `"uncovered": "two_pass"`, so its profiles from Callo 0.3.0 (which skipped those questions)
  and from 0.3.1 differ: `callo compare` flags them, and `callo calibrate --resume` will not mix the two. A model that
  covers all 255 reads exactly as before, and its contract is unchanged.

## In process

`callo.engine.local_engine(upstream, gguf=...)` gives the same readout against a plain llama-server with no proxy
(media detection and the delimiter only; no slot probe, no self-tests, no routing), and
`engine.answer_request(body, id_slot=None)` returns the `/v1/systemone` body. It serves budget 0 only: a body with a
positive `think_budget` raises `InvalidRequest` ("local_engine serves budget 0 only"); deliberate readouts need
`callo serve` (see [Think budgets](think-budgets.md)). `local_engine(..., position_bias=PATH)` applies a position-bias
profile (see [Position bias](position-bias.md#in-process)).

## No cancellation

A client that gives up on `/v1/systemone` frees nothing upstream: the readout runs to completion and its slot is
released after that (a deliberate request is the exception; see
[Think budgets](think-budgets.md#cancellation-and-client-timeouts)). `--s1-timeout` bounds each upstream call, not
the readout: one readout makes several calls (`/apply-template`, `/tokenize`, up to 9 walk passes, and on the image,
audio and video paths `/detokenize` and `/tokenize` per step).

## Native decision models

If the GGUF carries `<arch>.decision.type` (OpenJev, Lev, Kev, Laya), llama.cpp serves `/v1/systemone` itself and
Callo forwards requests there; see [Limitations](limitations.md#native-decision-models) for what differs.
