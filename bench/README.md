# bench/

The acceptance benches behind Callo's release results, and the scripts that built its data files.

- `acceptance*.py`, `wide_check.py`, `check_cache_delimiters.py`: GPU benches. Each launches llama-server through
  Callo, one model server at a time, and writes raw results plus a summary; `report*.py` renders the summaries.
- `build_bias_set.py`, `wide_items.py`, `build_asr_set.py`: build the bias set and the wide-choice items under
  `bench/data/`, and the utterance set the transcription bench reads (run once, CPU only).
- `release_check.py`: `make release-check`, the check to run by hand before publishing.

The acceptance benches ship no data and no models: they run on your own copies. Their default paths come from four
environment variables; a bench refuses to start, naming the flag and the variable, when a path it needs is unset.

| variable | what it points at |
|---|---|
| `CALLO_BENCH_DATA` | a checkout of [Kev](https://github.com/jaredpalmer/kev)'s `evals/` directory (the transfer set, the public pool) |
| `CALLO_BENCH_MODELS` | the root of your GGUFs; each bench names its model files relative to it (its module constants) |
| `CALLO_BENCH_OUT` | the root for raw results; a bench with a default `--out` writes under `callo_v0.3/<bench>/` |
| `CALLO_BENCH_LLAMA_SERVER` | the llama-server binary to launch |

Flags override them where a bench has one (`--data`, `--dev`, `--suite-file`, `--model`, `--mmproj`, `--out`,
`--llama-server`); `python -m bench.<module> --help` lists each bench's. Data files are pinned by sha256 where a gate
depends on their exact bytes, so a different copy is refused rather than measured.

Each bench (or its `report_*.py`) renders a result doc into `docs/bench/` in the checkout, created on first use;
`--doc`/`--out` moves it.

The unit tests run every bench on fakes and need none of this. The few tests that read real data or models skip when
the variable is unset or the file is missing.
