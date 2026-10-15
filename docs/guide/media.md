# Images, audio and video

One resident model can classify images, audio and video on `/v1/systemone` while it transcribes, captions or chats on
the other slots. Callo supports the modalities; the model decides which it takes (`/props.modalities`). Tiel-Coder's
`qwen3vl_merger` projector takes images and video (video as frames, adjacent pairs merged) but not audio; Gemma-4-E4B
with `mmproj-BF16.gguf` takes audio. Gemma 4's `gemma4` vocabulary is decoded like SentencePiece (`▁A`, `<0x41>`).

```
callo serve --model M.gguf --mmproj mmproj.gguf
callo classify --image frame.png --options up,down,left,right --instructions "Which way?" < /dev/null
callo classify --audio clip.wav --options speech,music,silence --instructions "What is in the clip?" < /dev/null
```

Contents: [images](#images) · [audio and video](#audio-and-video) · [`callo classify`](#callo-classify) ·
[native decision models](#native-decision-models) · [startup self-tests](#startup-self-tests) ·
[`/callo/stats`](#callostats) · [chat completions with media](#chat-completions-with-media) ·
[cost and limits](#cost-and-limits)

## Images

The `images` field and `image_url` state parts are described in [the API reference](api.md#request-fields): up to 8
data URLs, no `http(s)://`. The upstream needs vision (`--mmproj`); an image request to an upstream without vision
gets 501.

## Audio and video

```json
{"state": "",
 "audio":  ["data:audio/wav;base64,UklGR..."],
 "videos": ["data:video/mp4;base64,AAAAIGZ0eXA..."],
 "questions": {"q": {"type": "choice", "instructions": "Is anyone speaking?", "criteria": {"yes": null, "no": null}}}}
```

- **`audio`** and **`videos`** (Callo extensions; native System One has neither): data URLs, `data:audio/...;base64,`
  and `data:video/...;base64,`; `null` or `[]` means none. Audio must be WAV, MP3 or FLAC: Callo sniffs the bytes as
  llama.cpp does, and refuses OGG, Opus, M4A, WebM audio and ADTS AAC with 422 "unsupported audio format; use WAV,
  MP3 or FLAC". A video must not be an image or audio file. One gap: AAC behind an ID3 tag sniffs as MP3 and fails
  in llama-server, which Callo maps to 422.
- **Payloads are plain base64**, unwrapped, with no newlines or spaces: Callo validates strictly, so a clip encoded
  with line breaks (`base64` without `-w 0` on Linux, for example) gets 422 "invalid base64", although llama-server's
  own lenient decoder would have read it.
- **State parts**, in a messages-shaped `state`, shaped as llama.cpp reads them: `{"type": "input_audio",
  "input_audio": {"data": ..., "format": ...}}` (`format` is ignored; the bytes decide), `{"type": "input_video",
  "input_video": {...}}` and `{"type": "video_url", "video_url": {...}}`. The inner object sits under the key equal to
  the type, and its `data` wins over its `url`; each holds a data URL (any of `data:image/`, `data:video/`,
  `data:audio/`, as in llama.cpp) or raw base64. `http(s)://` and `file://` get 422 (Callo never fetches anything), as
  do `audio_url` and `video` parts, which llama.cpp itself refuses.
- **At most 8 media items in all**, images included. **Order is part of the prompt contract:** `images`, then
  `audio`, then `videos`, then the state's media parts in document order; that is the order of the rendered media
  parts and of the payloads llama-server receives.
- **The upstream must take the kind**, or the request gets 501 naming it. The rendered prompt is the
  [System One prompt contract](api.md#the-prompt-contract), with one part per item; Callo renders it through
  `/apply-template` with tiny fixed placeholder payloads (the template is media-blind, so the string is identical) and
  sends the real payloads only to `/completion`.
- **`Readout.kind`** and the `X-Callo-Timings` `kind` are `audio`, `video`, or `mixed` for more than one kind.
  `passes` counts the upstream scoring calls: every walk pass resends the media, and **a video is decoded again by
  ffmpeg on every pass**, even on a full cache hit.
- **The user-turn delimiter** is validated on a one-image prompt when the upstream has vision, and on a text-only
  prompt otherwise, which is also right for an audio-only model: the template renders every media part the same way.

The status codes for media requests are in [the API's error table](api.md#errors).

## `callo classify`

`callo classify` takes `--image PATH`, `--audio PATH` (`.wav`, `.mp3`, `.flac`) and `--video PATH` (`.mp4`, `.webm`,
`.mkv`, `.mov`, `.avi`), each repeatable, at most 8 files in all, and refuses (exit 2) a set that would encode to more
than 64 MiB before reading it. Any other extension also exits 2 (the MIME type comes from these fixed tables, never
from the platform's), as does a request body over 64 MiB once the state, `--context` and options are added.
`callo serve` has no media flags beyond `--mmproj`: video tuning is llama-server's own, after `--`.

## Native decision models

Native decision models take images only, so Callo answers 422 to a `/v1/systemone` body whose `audio` or `videos` is
anything but `null` or `[]`, or with an audio or video state part, and sends nothing upstream (native would answer
without the media, or read the base64 as prompt text); `counters.native_media_refused` counts those. Any other body is
forwarded unchanged; one over 64 MiB or with a `Content-Encoding` is forwarded unscanned
(`counters.native_unscanned`).

## Startup self-tests

`/props.modalities` gives `vision`, `audio` and `video`; `--mmproj` in launch mode needs any one of them, so an
audio-only projector starts. The self-tests run in this order: vision (when reported), text (always, while Callo's
engine is on), audio (when reported), video (when reported). The vision self-test is described under
[Startup](serving.md#startup).

- **Audio:** a built-in 4.0 s 16 kHz WAV, a 440 Hz tone ("Is there a sound in this audio clip, or is it silent?",
  expected `sound`), the same request again (its `cache_n` / `prompt_n` show the cache), and digital silence as the
  control (expected `silent`).
- **Video:** a built-in 64×64 uncompressed AVI, 8 frames at 4 fps, solid red ("What colour fills the video?",
  expected `red`), and solid blue as the control (expected `blue`).

Each passes only with both answers right and an option mass of at least 0.5 each; a failure warns and the modality
stays on. An audio self-test that fails for any reason, "Failed to load image or audio file" included, only warns:
audio stays on.

**One outcome turns video off:** llama-server answering the known-good AVI with "Failed to load image or audio file",
which means ffprobe or ffmpeg is missing from llama-server's `PATH` (or `--video-ffmpeg-dir`), or ffprobe cannot read
the clip; `/props.modalities.video` does not check for ffmpeg. Callo then logs an error line, `/callo/stats.video` is
`false` (and `cache_prompt.video` `null`), and video requests get 501 whose detail says video was turned off at
startup because the self-test clip failed to load (not that the build lacks video). A timeout, a 503 or a wrong
answer only warn. The tests run once, at startup.

**An ffmpeg that runs but cannot decode is not caught there:** llama.cpp then sends the model "Video:" and no frames,
with no error, so the video self-test fails as a wrong answer (or low mass) and warns while video stays on, and every
later video request on that server is answered without its frames. Read a FAILED video self-test as "check ffmpeg".

## `/callo/stats`

`/callo/stats` has `vision`, `audio` and `video`; `self_test`, `{"text", "vision", "audio", "video"}` each `true`
(passed), `false` (failed, or could not run: an upstream error or timeout), or `null` when that test did not run
because the modality is off or Callo's engine is (a verdict reads the readout's answer, which is debiased when a
position-bias profile is loaded, while its option mass is raw); under `s1`, `audio_requests`, `audio_p50_ms`,
`audio_p95_ms`, `audio_cache_n_mean`, `audio_prompt_n_mean`, the same five for `video`, and `mixed_requests`; and
`cache_prompt` reports `audio`, `video` and `mixed` (always `true`, like images, except `video`, which is `null` once
video is turned off at startup).

## Chat completions with media

Chat completions with media need no Callo change: a JSON chat body of 64 MiB or less with no `Content-Encoding` gets
`id_slot` whatever its parts. For routing, Callo replaces each media payload (a media part's `url`, `data` or string
`image_url` longer than 256 characters, and every `multimodal_data` entry of a `/completion` prompt) with a
fixed-size `sha256:` digest in a copy of the body (the upstream still receives the real payload), so a 20 MB video
costs the router lock nothing extra, and weights each digest by its payload's length in the 10% longest-common-prefix
test, so a media payload counts at its full size, as it does in llama-server's own test (which weights it by its
media tokens, not by its characters).

## Cost and limits

- **Encoding stalls every slot** while a run of media chunks encodes, agents included. Audio is cheap; video costs
  frames × the per-frame token budget.
- **No duration or frame cap**, in llama.cpp or in Callo. A few seconds of video at the default `--video-fps 4` can
  reach tens of thousands of tokens on a Qwen-VL model. Pass `--video-fps 1` and/or `--image-max-tokens N` to
  llama-server (after `--` in launch mode); the second also shrinks images. `--mtmd-batch-max-tokens` bounds the
  encode batch.
- **Body caps.** `/v1/systemone` over 64 MiB gets 413. A chat body over 64 MiB is forwarded unpinned, up to
  llama-server's own 100 MiB. For video the context usually runs out long before either cap: about 25 s of 1080p at
  4 fps is some 100 frames at up to ~2k tokens each, which already approaches a 200k-token context; such a request
  gets 413.
- **Untrusted video.** Every video Callo accepts is demuxed and decoded by ffprobe/ffmpeg inside llama-server, and
  Callo's checks on the bytes are best effort, not a container validator. Do not accept video from untrusted senders
  unless you trust that ffmpeg with it (Callo adds no exposure llama-server's own chat endpoint does not have).
- **Memory.** A `/v1/systemone` body near 64 MiB is held about four times over in Callo, plus one serialisation per
  walk pass. A translated transcription near 64 MiB costs up to about 0.5 GB transiently (the body, its base64 and the
  chat JSON).
- **Stability.** The readout caveats measured for images (slot index, delimiter batch split, `-kvu`) are assumed to
  hold for audio and video; they have not been measured.
- **No think budgets on media.** Deliberate readouts are text only (see [Think budgets](think-budgets.md#text-only)).
