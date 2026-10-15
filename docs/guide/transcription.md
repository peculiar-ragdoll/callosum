# Transcription

`POST /v1/audio/transcriptions` (and `/audio/transcriptions`) is multipart. llama-server reads form fields as strings,
so a form `id_slot` never pins one, and a transcription sent straight to llama-server lands wherever llama-server puts
it, a reserved slot included. Callo **translates** a well-formed request into the `/v1/chat/completions` request
llama-server builds itself (`convert_transcriptions_to_chatcmpl`), pins it like an agent chat, and reshapes the reply.
One resident model can then transcribe while it classifies: transcriptions are kept on one home agent slot, off
the reserved slots (when every agent slot holds a conversation in use, that home is shared with one agent; see
[Routing](#routing)).

```
curl -s localhost:8180/v1/audio/transcriptions -F file=@clip.wav -F language=en
```

Measured on Gemma-4-E4B against llama-server's own endpoint, the translated request rendered a byte-identical prompt
and returned identical text (see [Measurements](measurements.md#transcription)).

## What is translated

A non-streaming, `json`-format request (both are the defaults) whose body Callo can frame exactly as llama-server's
multipart parser does:

- one `multipart/form-data` `Content-Type` and CRLF framing;
- one `file` part with a filename and WAV, MP3 or FLAC bytes;
- every other part a uniquely named UTF-8 text field, and no NUL byte in `language` (llama-server's prompt would cut
  the language off there);
- `temperature` and `max_tokens` written as llama-server parses them;
- a `Content-Length` of at most 64 MiB, and no `Content-Encoding`.

Everything else passes through untouched and llama-server answers it. `/callo/stats.counters` counts it as
`transcription_unrouted_<reason>`: `no_audio`, `no_pinning`, `render`, `chunked` (still answered 411), `large`,
`encoded`, `not_multipart`, `file`, `fields` (a NUL in `language` included), `response_format`, `stream` or
`numbers`. `transcription_translated` counts the translated replies. Like every counter there, each is absent until
it is first counted. The one exception is a malformed or negative `Content-Length`: Callo answers it with its own 400
(`bad Content-Length`), as on passthrough, and while translation is on no counter rises for it.

## The prompt

The prompt is llama-server's own:

- `prompt` (else the preset's user text), then ` (language: X)` when `language` is set, then the audio;
- every form field copied as a string, `stream` set to `false`, `max_tokens` an integer and `temperature` a number;
- a form `messages` or `id_slot` replaced.

The preset is upstream's `common_chat_get_asr_prompt`:

- for an LFM2 chat template (its source contains both `<|tool_list_start|>` and `<|tool_list_end|>`): system
  `Perform ASR.` and no user text;
- otherwise no system message and `Transcribe audio to text`.

`/callo/stats.asr_preset` is `"default"`, `"lfm2"`, or `null` without audio. A new upstream ASR preset needs a Callo
update.

## The render check

At startup, when the upstream reports audio (native decision models included), Callo renders the translated content
parts and llama-server's own string form through `/apply-template`, once with a text part before the audio and once
with the audio alone. If they differ, translation is off for the process, a warning says why, and every transcription
passes through (`render`). That happens on a `--no-jinja` llama-server, whose legacy template path joins parts with a
newline, and with templates that trim each part. A render that fails (a 30 s timeout on each socket operation of each
call) or that lacks `/props.media_marker` turns translation off the same way. The check renders with
`chat_template_kwargs: {"enable_thinking": false}`, as Callo's other template calls do; a translated request carries
no `chat_template_kwargs`, so it renders with the server's default. Media parts are joined before the template runs,
so only a template whose content-part handling depends on `enable_thinking` could tell the two apart.

## Is it on?

`/callo/stats.transcription` is `{"translate": true, "reason": null}` while translating. Otherwise `translate` is
false and `reason` names the counter every transcription now increments, then why: `no_audio: …`, `no_pinning: …` or
`render: …`, tested in that order. The startup log line says so too (a WARNING only for `render`).

## Routing

Only agent slots, never a reserved one, and never by longest common prefix, which a transcription shares with any
conversation only through JSON scaffolding. `X-Callo-Slot` names the slot.

- With the client's conversation hint (`X-Callo-Conversation`, `X-Conversation-Id`, or the form's `user` field), as
  an agent chat: to that hint's slot while it is idle; otherwise (a new hint, or its slot busy) to the
  least-recently-used idle agent slot, else the least loaded (which can be the hint's own slot), and the hint is bound
  to the slot it got.
- Without one, by the shared hint `callo:asr`, so transcriptions stay on one agent slot instead of walking every cold
  one. Its home is an idle agent slot no conversation hint is bound to (the least-recently-used one), else an idle
  agent slot whose bound hints have all gone unused for `--asr-home-idle` seconds since the slot's last request
  ended (default 600; the one idle longest), so it does not settle on a slot an agent is using while another will
  do. The bindings are kept: an agent that returns gets its slot back by its hint, and that re-homes `callo:asr`.
  `--asr-home-idle 0` counts any bound hint not in use right now; the flag must be a finite number of seconds, 0 or
  more.
- A busy slot is never waited for. When the home is busy, that one transcription goes to the least-recently-used
  idle agent slot (else the least loaded), but `callo:asr` stays home: the next one goes back. When every candidate
  for a new home is busy, the transcription is served the same way and `callo:asr` stays homeless until the next
  one.
- Any other hint routed to the home (an agent's, or a client-hinted transcription's) re-homes it: that hint keeps the
  slot and the next hint-less transcription picks a new home.
- When every agent slot holds a hint in use (more live conversations than agent slots), the home is the normal pick
  (the least-recently-used idle agent slot, else the least loaded) and that one agent slot shares its cache with the
  transcriptions until any other hint routed to that slot re-homes it. That is still one slot, where
  least-recently-used routing would visit them all.
- Callo keeps at most 10,000 conversation-hint bindings. Past that, each new one evicts the least recently used
  binding whose slot has no other request in flight, counted in `router.counters.hint_evicted`. While every slot that
  hosts a binding is busy, nothing can go; the table drains back to the cap on the next release that leaves such a
  slot with nothing in flight. An evicted conversation is routed like a new one next time.
- `/callo/stats.transcription.asr_home` is the home slot (`null` while homeless), `asr_home_idle_s` is
  `--asr-home-idle`, and the counter `transcription_rehomed` counts the homes chosen after the first.
  `router.conversations` counts client conversation hints only; `callo:asr` is reported under `router.sticky`
  (`{"callo:asr": {"home", "rehomed"}}`), and a home on an idle hint's slot counts as `router.counters.idle_hint`.

## Auth

The client's `Authorization`, `X-Api-Key` and `Origin` are forwarded unchanged, and Callo never adds its
`--upstream-api-key`: a client llama-server would refuse is still refused, as on passthrough. Callo's own startup
render check does use `--upstream-api-key`.

## The reply

A 2xx becomes `{"type": "transcript.text.done", "text", "usage": {"type": "tokens", "input_tokens", "output_tokens",
"total_tokens", "input_tokens_details": {"cached_tokens"}}}`, built from the chat reply's `choices[0].message.content`
and `usage`. `cached_tokens` comes from `prompt_tokens_details.cached_tokens`, and is 0 when absent. Reasoning is not
part of `text`, as upstream, with one known difference: when the model's output parses to an empty message
(whitespace only, or a bare empty think block), Callo returns llama-server's raw generated text, where the native
endpoint returns `""`. The upstream's other headers (CORS included) are kept. Any non-2xx is relayed unchanged. Two
502s are Callo's own:

- a 2xx without a `choices[0].message.content` string is `{"detail": "…"}`, counted as `transcription_bad_reply`;
- an upstream that cannot be reached, or fails during the request, is
  `{"error": {"message": "…", "type": "callo_upstream"}}`, counted as `upstream_errors`.

## Limits

Streaming and other response formats are not translated, and neither is a translation whose chat JSON would exceed
llama-server's 100 MiB request cap (`large`). See [Images, audio and video](media.md#cost-and-limits) for the memory
cost.
