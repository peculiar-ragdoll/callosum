---
license: other
license_name: mit-and-cc-by-4.0
license_link: https://huggingface.co/datasets/peculiar-ragdoll/callo-bias-set/blob/main/README.md#licences
pretty_name: callo built-in position-bias set (builtin-v1)
task_categories:
- multiple-choice
- text-classification
language:
- en
size_categories:
- n<1K
viewer: false
---

# callo-bias-set: `builtin-v1`

`bias_set.jsonl` is the built-in position-bias calibration set of
[callo (Callosum)](https://github.com/peculiar-ragdoll/callosum), named `builtin-v1`. `callo calibrate --bias` without
`--data` downloads it on first use, verifies its sha256, caches it, and reads it as a diagnostic: it measures whether a
served model shows a shared position bias on generic multiple-choice questions and writes no profile. The callo wheel
ships code only; this dataset is where the set lives.

- **File:** `bias_set.jsonl`, 246,712 bytes, 336 records (one System One request per line).
- **sha256:** `f73ff674470f75993076e4e00cc306c88fa1f28bde1e85b49dcf03738a9811a1` (callo pins this value and refuses any
  other bytes).
- **Contents:** 48 questions per option count n = 2 … 8 (1,680 readouts over all rotations). n = 2, 3, 4 take 24 MMLU
  validation and 24 BANKING77 train questions each; n = 5 … 8 are BANKING77 only, a single domain. Every question is
  labelled; nothing comes from an upstream test split.
- **Offline use:** download the file at a pinned revision and point callo at it with
  `CALLO_BIAS_SET=/path/to/bias_set.jsonl`; callo checks the sha256 above.

## How it was built

[`bench/build_bias_set.py`](https://github.com/peculiar-ragdoll/callosum/blob/v0.3.1/bench/build_bias_set.py) in
callosum `v0.3.1` builds these exact bytes, deterministically, from the two pinned sources below (rerunning it gives
byte-identical output), and writes them to `bench/data/bias_set.jsonl` with the held-out remainder
(`bench/data/bias_heldout_v1.jsonl`) and a manifest of the sources' and outputs' sha256s
(`bench/data/bias_set_manifest.json`). The script in that release has sha256
`80672372519abb0601c56383917335ee50a4ee22d06a5d07ccd9f83b8e850f16`, so the provenance does not rest on the link alone.

Each record has a `state`, one `questions` entry (`type: "choice"`, `instructions`, `criteria`, `label`) and a `_meta`
naming its source, upstream repository, revision, split and row.

## Licences

The data is derived from two third-party datasets, under their own licences: MMLU (MIT) and BANKING77 (CC-BY-4.0).
The sections below are copied from callo's `NOTICE`.

### MMLU

```
MMLU (Measuring Massive Multitask Language Understanding), validation split
Source: https://huggingface.co/datasets/cais/mmlu at revision c30699e8356da336a370243923dbaf21066bb9fe
        (all/validation-00000-of-00001.parquet); originally https://github.com/hendrycks/test
Use: questions and answer options; the 2- and 3-option questions keep the gold answer and the first distractors.

MIT License

Copyright (c) 2020 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### BANKING77

```
BANKING77, training split
Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan Vulić (PolyAI), "Efficient Intent
Detection with Dual Sentence Encoders", 2020.
Source: https://github.com/PolyAI-LDN/task-specific-datasets at commit 57ec275d8078af65b7731c2a98be812d844a6d6b
        (banking_data/train.csv, banking_data/categories.json)
Licence: Creative Commons Attribution 4.0 International (CC-BY-4.0), https://creativecommons.org/licenses/by/4.0/
modified: distractors sampled, intent names reformatted, packaged as System One requests.
```
