# Exit codes

On Windows every `callo` command exits `2` before doing anything (see
[No auth, no TLS, no Windows](limitations.md#no-auth-no-tls-no-windows)).

## `callo serve`

- `0`: a deliberate stop (SIGINT or SIGTERM, including during startup, which also stops the launched child).
- `1`: a runtime error or an invalid flag value:
  - `--temperature <= 0`, `--s1-slots < 1`, `--mmproj` in attach mode, `--mmproj` given but the upstream reports no
    media modality (vision, audio or video), a busy or unusable `--host`/`--port`, a missing `llama-server`, a
    launched `llama-server` that died while loading or before `/health` was ok, an unreachable or unusable upstream;
  - an `--upstream` that is not plain `http://host[:port]` (https, a path, a query or credentials);
  - every `--position-bias` problem: a profile file that cannot be read or is invalid (checked before Callo listens or
    launches anything), a served model whose `model_key` cannot be computed, a profile measured on another model, or
    a native decision-model upstream;
  - think budgets: a non-integer or negative `--think-budget` or `--think-max`, `--think-max` above 32768 or below
    `--think-budget`, an invalid `--think-profile` or `--think-sampling`, or `--think-sampling` with `--think-profile`
    (all before Callo listens), and, once the model is known, `--think-budget` > 0 or `--think-profile` on a native
    decision model, `--think-budget` > 0 on a model whose template has no thinking mode Callo can read or whose
    sampler a request cannot pin (`sampler_unpinnable`: a server-wide `--grammar`, or an unknown non-neutral
    setting), or a think profile measured on another model, on a model whose `model_key` cannot be computed, or with
    thinking budgets on a model that cannot think;
  - think deadlines: `--think-deadline` > 0 on a native decision model, a model that cannot think, or a model whose
    close tag is more than one token (see [Think deadlines](think-budgets.md#which-models));
  - early exit: `--think-converge` that is not a number from 0 to below 1, or a positive one without
    `--think-profile` or with a profile that has no usable convergence fit (all before Callo listens), and, once the
    model is known, `--think-converge` > 0 on a native decision model, a model that cannot think, or a model whose
    close tag is more than one token (see [Early exit](think-budgets.md#reach-opt-outs-and-models)), and
    `--think-converge-gate` on a native decision model.
- `2`: a command-line usage error (unknown flag, neither or both of `--model`/`--upstream`, a non-integer
  `--port`, an empty `--mmproj ""`, `--user-delimiter ""`, `--position-bias ""`, `--think-profile ""` or
  `--think-sampling ""`, an `--s1-timeout` that is not a finite number above 0, an `--asr-home-idle` that is not a
  finite number of 0 or more, `--no-kvu` in attach mode or without a `--ctx` of at least 1), rejected by argument
  parsing before anything starts.
- `3`: the launched `llama-server` died **after** Callo was serving. Callo shuts down and reports it.

## `callo classify` and `callo calibrate`

`callo classify` and `callo calibrate` use `0` for success, `1` for a failed request or no usable data, and `2` for a
usage error.

`callo classify` also returns `2` for an empty `--options`, for more than 8 media files (`--image`, `--audio` and
`--video` together), an extension outside its table, a file it cannot read, and for media or a body over 64 MiB.

`callo calibrate` also returns `2` when it refuses to run (see [Calibration](calibration.md)), when `--data` is
missing without `--bias`, and when a `--bias`-only flag (`--raw`, `--resume`, `--min-questions`, `--max-per-n`,
`--seed`, `--allow-busy`) is given without `--bias` (of these, `--resume`, `--seed` and `--allow-busy` also apply with
`--think-budgets`), when a `--think-budgets`-only flag (`--upstream`, `--warmup`, `--audit`, `--cache` / `--no-cache`,
`--think-sampling`, `--latency`, `--limit`, `--price`, `--max-latency`, `--min-accuracy`, `--verbose`, `--dry-run`)
is given without `--think-budgets`, `--bias` included, and for an empty `--out` or `--raw` (`--out ""` is a mistake,
not a request for the default); it returns `1` for a missing, unreadable or non-UTF-8 `--data` and for a malformed
`--data` line (naming its line number).

`callo calibrate --think-budgets` and menu runs have their own codes: see
[Calibrating think budgets](calibration.md#exit-codes) and [menus](calibration.md#runjson-failure-isolation-and---resume).

## `callo calibrate --bias`

- `0` whenever it writes a profile, even one where every n was dropped, and when the built-in diagnostic finishes;
- `1` for a request that failed for good, an unusable reply, an unreadable or malformed `--data` (naming the line), a
  built-in set that cannot be fetched or fails its sha256 (naming the URL and `CALLO_BIAS_SET`), a set with no
  eligible question, an `--out` or `--raw` it cannot write, `/callo/stats` that cannot be read
  mid-run, or a console that went away (a closed stdout or stderr pipe: no traceback);
- `2` for a refusal: `/callo/stats` unreachable at start, the preconditions (see
  [Position bias](position-bias.md#preconditions-exit-2)), a busy server, a moved slot, Callo restarted or
  reconfigured mid-run, an existing `--raw` without `--resume`, a `--resume` whose raw file has no raw header, is not
  UTF-8 or has a malformed readout or check line, or a header that differs; or a usage error (`--min-questions` below
  2, `--max-per-n` below 1, a negative `--seed`, an empty `--out` or `--raw`, or `--out` with the built-in set);
- `130` on Ctrl-C (also while it writes `--out` or the summary), with `--raw` keeping what was read and a hint to
  rerun with `--resume` when this run opened `--raw` or was itself a `--resume` run, and the file keeps readouts.

A `1` adds the `--resume` hint only when this run opened `--raw`: a failure before any reading (a bad `--data`, no
eligible question) never points at a raw file left by another run. A `--raw` that cannot be opened or written is
still named in the error.
