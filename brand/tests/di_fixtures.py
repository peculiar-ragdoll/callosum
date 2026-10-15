"""Synthetic inputs for the DI plate tests: a bench tree shaped as the benchmark's DI tools write it (di_bootstrap.py's
output per treatment, di_union.py's union, each arm's di_meta.json / results / server.log, sample.json, the Clef prompt
tokens, the kit's index.json for the area labels) and a spec naming it."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

KIT = "9eb2dbe2a358004c8782c66e40a83ac07b953fec"
SAMPLE_SHA = "0ab3526d" + "0" * 56
AREAS = ("knowledge", "language", "retrieval", "tools", "arts")
LABELS = {"knowledge": "Knowledge & Reasoning", "language": "Language Understanding",
          "retrieval": "Retrieval & Classification", "tools": "Tools & Automation", "arts": "Arts & Human Taste"}
REF, SECOND = "cflash-bf16", "clef27b"
# arm -> (index, mean ms, p95 ms, area skills)
ARMS = {
    "tiel-b0": (38.3, 2225.8, 10884.1, (0.2778, 0.4572, 0.2877, 0.5655, 0.3204)),
    "tiel-b256": (48.81, 13879.9, 76491.3, (0.4448, 0.5173, 0.3976, 0.7009, 0.3163)),
    "sharp-minicpm-b0": (22.54, 780.7, 3280.1, (0.164, 0.175, 0.2253, 0.4198, 0.1589)),
    "sharp-minicpm-b1024": (41.48, 18711.7, 94810.0, (0.3508, 0.4716, 0.3454, 0.6184, 0.1997)),
    "clef27b": (58.0, 8526.4, 36583.0, (0.5552, 0.5451, 0.5721, 0.7951, 0.3572)),
    "cflash-q4": (55.49, 2437.0, 10493.6, (0.5398, 0.4628, 0.503, 0.8245, 0.4426)),
    "cflash-bf16": (54.75, 2390.6, 10264.6, (0.5568, 0.4619, 0.4634, 0.8152, 0.424)),
}
CALLO = {"tiel-b0": 0, "tiel-b256": 256, "sharp-minicpm-b0": 0, "sharp-minicpm-b1024": 1024}
# Tiel-Coder's early-exit arms at 256 (setup(variants=True), as on the release plate: the line ends at tiel-b256-ee;
# plain tiel-b256 and the gated tiel-b256-eeg are table rows): arm -> (index, mean ms, p95 ms, areas), their early exit,
# and their paired Δ vs tiel-b256 (a second di_bootstrap.py output with tiel-b256 as its reference)
VARIANTS = {"tiel-b256-ee": (49.04, 9722.5, 50492.0, (0.4501, 0.5302, 0.4103, 0.6800, 0.3002)),
            "tiel-b256-eeg": (48.06, 11423.2, 54327.9, (0.4402, 0.5201, 0.3804, 0.6902, 0.3101))}
EE = {"tiel-b256-ee": {"z": 0.95, "gate": False}, "tiel-b256-eeg": {"z": 0.95, "gate": True}}
PAIRED_D = {"tiel-b256-ee": (0.21, -1.9603, 2.3), "tiel-b256-eeg": (-0.77, -2.9213, 1.51)}
CALLO_ROWS = ["tiel-b0", "tiel-b256", *VARIANTS]          # Tiel-Coder's table rows: plain, early exit, + gate
TABLE_ONLY = ("tiel-b256", "tiel-b256-eeg")                 # setup(variants=True): rows of the table, not drawn
# real file names keep their bake names (UD- included); only the display names drop it
MODELS = {"tiel": "Tiel-UD-Q4_K_XL.gguf", "sharp-minicpm": "Mini-UD-Q6_K_XL.gguf", "clef27b": "Clef-27B-UD-Q4_K_XL.gguf",
          "cflash-q4": "Flash-UD-Q4_K_XL.gguf", "cflash-bf16": "Flash-BF16.gguf", "d1": "d1-3B-Q4_K_M.gguf"}
# setup(d1=True): a native decision model that is not Clef (Liquid AI's d1-3B on its own llama.cpp build), a single
# marker in its own family, reading one question per prompt, with its own offline prompt-token counts
D1 = "d1"
D1_ARM = (46.07, 1491.4, 6240.9, (0.2258, 0.5307, 0.4373, 0.7992, 0.3149))
D1_SHA = "16" * 32
D1_ANSWERING = "answers each question in its own prompt (the state, then that question)"


def d1_counts(*, gguf: str = D1_SHA, n_ctx: int = 16384, sample: str = SAMPLE_SHA, own: int = 0,
              premarked: list[str] | None = None) -> dict:
    """The arm's own prompt-token counts (the bench's di_d1_tokens.py output, the fields the plate reads)."""
    rows = PREMARKED if premarked is None else premarked
    return {"arm": D1, "gguf_sha256": gguf, "sample_sha256_uncompressed": sample, "n_ctx": n_ctx,
            "own_premarks_needed": own, "rows_sent": 98, "questions_sent": 880,
            "premarked_rows_max_question": dict(zip(rows, (1198, 7044), strict=False)),
            "sent_max_question": {"max": 9393}}


def d1_series() -> dict:
    return {"name": "d1", "label": "d1-3B · Q4_K_M", "short": "d1-3B", "kind": "clef", "family": "d1", "arm": D1,
            "marker": "triangle", "runtime": "Liquid AI · llama.cpp native", "model_file": MODELS["d1"],
            "build": "23deec02", "answering": D1_ANSWERING, "own_prompt_tokens": "d1_tokens.json"}
LAT_ROWS = 97
PREMARKED = ["36:BRIGHT:a:chunk0", "36:BRIGHT:b:chunk0"]
ERROR_ROW = "25:GPQA-Diamond:GPQA-Diamond:test:99"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _arm_doc(arm: str, index: float, mean: float, p95: float, areas: tuple, shift: float = 0.0) -> dict:
    idx = round(index + shift, 2)
    ref_areas = dict(zip(AREAS, ARMS[REF][3], strict=True))
    zero = {"point": 0.0, "ci": [0.0, 0.0]}

    def dl(d: float) -> dict:
        return {"point": round(d, 4), "ci": [round(d - 3.9, 4), round(d + 3.4, 4)]}

    return {
        "index": idx, "kit_index": idx, "index_ci": [round(idx - 3.4, 4), round(idx + 3.1, 4)],
        "areas": dict(zip(AREAS, areas, strict=True)),
        "areas_ci": {a: [round(v - 0.06, 4), round(v + 0.06, 4)] for a, v in zip(AREAS, areas, strict=True)},
        "delta_vs_reference": dl(idx - ARMS[REF][0]) if arm != REF else zero,
        "delta_vs_second_reference": dl(idx - ARMS[SECOND][0]) if arm != SECOND else zero,
        "area_delta_vs_reference": {a: ({"point": round(v - ref_areas[a], 4), "ci": [round(v - ref_areas[a] - 0.07, 4),
                                                                                 round(v - ref_areas[a] + 0.06, 4)]}
                                        if arm != REF else {"point": 0.0, "ci": [0.0, 0.0]})
                                    for a, v in zip(AREAS, areas, strict=True)},
        "latency_ms": {"mean": mean, "p95": p95, "mean_ci": [round(mean * 0.92, 1), round(mean * 1.07, 1)],
                       "n_rows": LAT_ROWS},
    }


def bootstrap(root: Path, arms: dict, *, shift: float = 0.0, n: int = 1000) -> dict:
    inputs = {}
    for arm in arms:
        d = root / "di" / arm
        inputs[arm] = {"derived": f"di/{arm}/results.derived.jsonl", "derived_sha256": _sha(d / "results.derived.jsonl"),
                       "results": f"di/{arm}/results.jsonl", "results_sha256": _sha(d / "results.jsonl")}
    return {
        "generated_utc": "2026-10-11T14:46:18+00:00", "edition": "0.3", "n": n, "seed": 20260919, "warnings": [],
        "level": 0.95, "kit_commit": KIT, "reference": REF, "second_reference": SECOND, "rows": 100,
        "groups_per_benchmark": {"3": 34, "25": 33, "36": 33}, "latency_rows": LAT_ROWS,
        "method": "paired percentile bootstrap (synthetic)", "inputs": inputs,
        "arms": {arm: _arm_doc(arm, *v, shift=(shift if arm in CALLO else 0.0)) for arm, v in arms.items()},
    }


def _series_of(arm: str) -> str:
    return re.sub(r"-b\d+(-\w+)?$", "", arm)


def _converge(requests: int, questions: int, *, gated: bool, asked: int = 0, skipped: int = 0) -> dict:
    return {"default_z": 0.95, "K": 32, "requests": requests, "questions": questions, "converged": questions - 3,
            "probes": 2 * questions, "tokens_saved": 200 * questions,
            "gate": {"default": gated, "asked": asked, "skipped": skipped, "errors": 0,
                     "read_ms": {"p50": 1130.06 if gated else None, "p95": 5930.1 if gated else None}}}


def write_arm(root: Path, arm: str, *, budget: int | None = None, ee: dict | None = None) -> None:
    d = root / "di" / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.jsonl").write_text(json.dumps({"run_id": "x", "status": "ok", "arm": arm}) + "\n")
    (d / "results.derived.jsonl").write_text(json.dumps({"run_id": "x", "status": "ok", "arm": arm, "d": 1}) + "\n")
    model = f"/models/{MODELS[_series_of(arm)]}"
    if arm in CALLO or budget is not None:
        b = CALLO.get(arm, budget)
        cmd = ["/nowhere/callo-frozen/.venv/bin/callo", "serve", "--model", model, "--ctx", "65536", "--parallel", "4",
               "--s1-slots", "1", "--think-budget", str(b), "--port", "8190"]
        if ee:
            cmd += ["--think-converge", str(ee["z"])] + (["--think-converge-gate"] if ee["gate"] else [])
        (d / "server.log").write_text("2026-10-11 01:42:59 callo INFO upstream: total_slots=4 build_info=b50-test\n")
        extra = {"model": model, "model_sha256": "ab" * 32, "think_profile": "/p/tiel.think.json",
                 "think_profile_sha256": "cd" * 32}
        if ee:
            g = ee["gate"]
            extra.update(converge=ee["z"], gate=g, converge_check={"pass": True},
                         think_converge_after_warm=_converge(4 if g else 5, 5, gated=g, asked=5 if g else 0,
                                                             skipped=1 if g else 0),
                         think_converge=_converge(896 if g else 1476, 9647, gated=g, asked=1476 if g else 0,
                                                  skipped=580 if g else 0))
    else:
        cmd = ["/x/llama-server", "-m", model, "-c", "16384", "-b", "16384", "-ub", "16384", "-np", "1"]
        extra = {}
    sess = {"utc_start": "2026-10-10T12:10:05+00:00", "utc_end": "2026-10-10T15:29:11+00:00", "cmd": cmd,
            "sample_sha256": SAMPLE_SHA, "kit_commit": KIT, "accept_cap_loss": ["36"],
            "final_counts": {"unsupported": 2, "ok": 98}, **extra}
    (d / "di_meta.json").write_text(json.dumps({"arm": arm, "kind": "callo" if arm in CALLO or budget else "clef",
                                                "sessions": [sess]}))


def union(arms: list[str], *, error_cause: str | None = None) -> dict:
    u = {rid: {"catalog_id": 36, "group_id": rid.split(":")[2], "arms": dict.fromkeys(arms, "pre-marked")}
         for rid in PREMARKED}
    u[ERROR_ROW] = {"catalog_id": 25, "group_id": "GPQA-Diamond:test:99", "arms": {"sharp-minicpm-b1024": "error"}}
    return {
        "arms": {a: f"di/{a}/results.jsonl" for a in arms}, "sample_rows": 100, "union_rows": 3,
        "union_by_reason": {"pre-marked": 2, "error": 1}, "union_per_benchmark": {"36": 2, "25": 1},
        "errors": [{"arm": "sharp-minicpm-b1024", "run_id": ERROR_ROW, "catalog_id": 25,
                    "error": "Server error '500 Internal Server Error'\nFor more information", "cause": error_cause}],
        "unexplained_errors": {a: int(a == "sharp-minicpm-b1024" and error_cause is None) for a in arms},
        "pending": dict.fromkeys(arms, 0), "union": u, "sample_sha256_uncompressed": SAMPLE_SHA,
    }


def paired(root: Path, arms: dict, ref: str = "tiel-b256") -> dict:
    """A second di_bootstrap.py output on the same draws, with `ref` as its reference (the early-exit comparison)."""
    doc = bootstrap(root, arms)
    doc["reference"], doc["second_reference"] = ref, "tiel-b0"
    for arm, a in doc["arms"].items():
        d = PAIRED_D.get(arm, (0.0, 0.0, 0.0) if arm == ref else (round(a["index"] - arms[ref][0], 4), -9.0, 9.0))
        a["delta_vs_reference"] = {"point": d[0], "ci": [d[1], d[2]]}
    return doc


def setup(tmp: Path, *, arms: dict | None = None, both: bool = False, switch: str = "kit",
          explain: bool = True, extra_bootstrap_arms: dict | None = None, over_limit: int = 0,
          q4_on_plate: bool = False, variants: bool = False, d1: bool = False) -> dict:
    """A bench tree under tmp and its spec (unresolved); returns {"root", "spec_path", "spec"}.
    `over_limit`: sampled rows over callo's 255-option limit (0, as in the real sample, makes treatment (b) moot).
    `variants`: Tiel-Coder's early-exit arms at 256 too, as on the release plate: its line ends at tiel-b256-ee, and
    plain tiel-b256 and the gated tiel-b256-eeg are its variants (table rows, not drawn), with the paired bootstrap of
    both early-exit arms vs tiel-b256."""
    arms = dict(arms or ARMS)
    if variants:
        arms.update(VARIANTS)
    if d1:
        arms[D1] = D1_ARM
    every = {**arms, **(extra_bootstrap_arms or {})}
    for arm in every:
        b = 256 if arm in VARIANTS else (
            int(arm.rsplit("-b", 1)[1]) if arm.startswith(("tiel-b", "sharp-minicpm-b")) else None)
        write_arm(tmp, arm, budget=b, ee=EE.get(arm))
    if d1:                       # its run recorded the model's sha256 and its server's --version, as the bench's does
        meta_p = tmp / "di" / D1 / "di_meta.json"
        meta = json.loads(meta_p.read_text())
        meta["sessions"][0].update(model_sha256=D1_SHA, server_version=[
            "version: 0.6.0-dev (build 260, commit 23deec0)", "built with AppleClang 17.0.0.17000603 for Darwin arm64"])
        meta["sessions"][0]["utc_start"], meta["sessions"][0]["utc_end"] = ("2026-10-14T21:04:14+00:00",
                                                                            "2026-10-14T21:40:45+00:00")
        meta_p.write_text(json.dumps(meta))
        (tmp / "d1_tokens.json").write_text(json.dumps(d1_counts()))
    (tmp / "boot-a.json").write_text(json.dumps(bootstrap(tmp, every)))
    if variants:
        sub = {a: every[a] for a in ("tiel-b256", *VARIANTS, "tiel-b0")}
        (tmp / "paired.json").write_text(json.dumps(paired(tmp, sub)))
    if both:
        (tmp / "boot-b.json").write_text(json.dumps(bootstrap(tmp, every, shift=-2.0)))
    (tmp / "union.json").write_text(json.dumps(union(list(every))))
    (tmp / "sample.json").write_text(json.dumps({
        "sample_sha256_uncompressed": SAMPLE_SHA, "command": "python -m decision_index suite sample --n 100 --seed 20260919",
        "totals": {"rows": 100, "rows_gt26": 5, "questions_gt26": 5, "rows_gt255": over_limit,
                   "questions_gt255": over_limit},
        "max_options_overall": 53,
        "per_benchmark": {"3": {"dataset": "API-Bank", "groups": 34, "rows_gt26": 5},
                          "25": {"dataset": "GPQA Diamond", "groups": 33, "rows_gt26": 0},
                          "36": {"dataset": "BRIGHT", "groups": 33, "rows_gt26": 0}}}))
    (tmp / "prompt_tokens.json").write_text(json.dumps({"cap_tokens": 16384, "over_cap_run_ids": PREMARKED}))
    (tmp / "prompt_tokens_rows.json").write_text(json.dumps({"tokens": {PREMARKED[0]: 17112, PREMARKED[1]: 32805}}))
    (tmp / "index.json").write_text(json.dumps({
        "index": ARMS[REF][0], "areas": [{"id": a, "label": LABELS[a]} for a in AREAS],
        "note": "fixed weights (knowledge 25.8%, language 25.8%, retrieval 20.0%, tools 18.3%, arts 10%)."}))
    spec = {
        "root": str(tmp), "preliminary": True, "title": "Clef leads, *thinking narrows the gap*",
        "wide_options": switch, "callo_max_options": 255, "treatments": {"kit": "boot-a.json", "all-unsupported": "boot-b.json" if both else None},
        "union": "union.json", "sample": "sample.json", "prompt_tokens": "prompt_tokens.json",
        "prompt_tokens_rows": "prompt_tokens_rows.json", "di_dir": "di", "area_labels_from": "index.json",
        "reference": REF, "reference_label": "Clef-Flash BF16", "second_reference": SECOND,
        "second_reference_label": "Clef 27B", "story": {"series": "tiel", "think": 256},
        "hardware": "Apple M2 Ultra, 64 GB, one request at a time",
        "callo_commits": {a: ("9d52aaf" if a == "sharp-minicpm-b0" else "0d33846" if a in VARIANTS else "b463e5f")
                          for a in every if a in CALLO or a.startswith(("tiel-b", "sharp-minicpm-b"))},
        "paired": [{"path": "paired.json", "arms": list(VARIANTS)}] if variants else [],
        "explain": ([{"arm": "sharp-minicpm-b1024", "run_id": ERROR_ROW, "cause": "a Callo 500, root-caused and fixed"}]
                    if explain else []),
        "notes": {"preliminary": "callo arms measured before a readout fix (end-of-turn token at the answer slot, "
                                 "~0.3–1 % of readouts walked); re-measured on the release bake",
                  "latency": "Clef 27B ran beside CPU-heavy test runs (~14:10–15:40 on 2026-10-10); re-timed for the "
                             "release bake",
                  "latency_figures": "two Clef arms ran beside CPU-heavy test runs (Clef 27B ~14:10–15:40, Clef-Flash "
                                     "Q4 ~18:00–18:26 on 2026-10-10); re-timed for the release bake",
                  "subset": "iSarcasmEval track A-En", "cap_release": "Clef's release code would refuse 1 and truncate 1",
                  "wide_choice": "Callo and Clef both answer up to 255 options",
                  "board": "the 0.3 board's public index for Clef is 61.71",
                  "treatment_all-unsupported": "not computed (one treatment since 2026-10-10)"},
        "series": [
            {"name": "tiel", "label": "Tiel-Coder 35B-A3B · Q4_K_XL", "short": "Tiel-Coder", "kind": "callo",
             "accent": True, "runtime": "Callo · llama.cpp", "model_file": "Tiel-UD-Q4_K_XL.gguf", "label_side": "above",
             "budgets": {"0": "tiel-b0", "256": "tiel-b256-ee" if variants else "tiel-b256"},
             **({"variants": dict.fromkeys(TABLE_ONLY, 256), "early_exit": dict(EE),
                 "point_label_at": {"tiel-b256-ee": "nw"}} if variants else {})},
            {"name": "sharp-minicpm", "label": "Sharp-MiniCPM5-2B · Q6_K_XL", "short": "Sharp-MiniCPM5-2B",
             "kind": "callo", "runtime": "Callo · llama.cpp", "model_file": "Mini-UD-Q6_K_XL.gguf", "label_side": "below",
             "budgets": {"0": "sharp-minicpm-b0", "1024": "sharp-minicpm-b1024"}},
            {"name": "flashnext-sharp", "label": "Qwen3.8-Flash-Next (Sharp)", "kind": "callo",
             "marker": "square-hollow", "budgets": {"0": "flashnext-sharp-b0", "256": "flashnext-sharp-b256"},
             "pending_note": "Qwen3.8-Flash-Next (Sharp) at budgets 0 and 256 (release bake)"},
            {"name": "clef27b", "label": "Clef 27B · Q4_K_XL (our quant)", "short": "Clef 27B", "kind": "clef",
             "arm": "clef27b", "marker": "diamond", "runtime": "llama.cpp native Clef",
             "model_file": "Clef-27B-UD-Q4_K_XL.gguf",
             "build": "b11379 1537a0a8"},
            {"name": "cflash-q4", "label": "Clef-Flash 9B · Q4_K_XL (our quant)", "short": "Clef-Flash Q4",
             "kind": "clef", "arm": "cflash-q4", "marker": "diamond-hollow", "runtime": "llama.cpp native Clef",
             "model_file": "Flash-UD-Q4_K_XL.gguf", "build": "b11379 1537a0a8", "plate": q4_on_plate},
            {"name": "cflash-bf16", "label": "Clef-Flash 9B · BF16", "short": "Clef-Flash BF16", "kind": "clef",
             "arm": "cflash-bf16", "marker": "diamond", "runtime": "llama.cpp native Clef",
             "model_file": "Flash-BF16.gguf", "build": "b11379 1537a0a8"},
            *([d1_series()] if d1 else []),
        ],
    }
    (tmp / "spec.json").write_text(json.dumps(spec))
    return {"root": tmp, "spec_path": tmp / "spec.json", "spec": copy.deepcopy(spec)}


def rewrite_spec(s: dict, **over) -> Path:
    spec = {**s["spec"], **over}
    p = Path(s["root"]) / "spec2.json"
    p.write_text(json.dumps(spec))
    return p


DROP_WEIGHTS = ({"4": 0.181818, "5": 0.181818, "36": 0.181818, "37": 0.151515, "56": 0.151515, "61": 0.151515},
                {"4": 0.222222, "5": 0.222222, "37": 0.185185, "56": 0.185185, "61": 0.185185})


def without(doc: dict, shifts: dict[str, float], *, drop: str = "BRIGHT") -> dict:
    """A di_bootstrap.py --drop-benchmark output paired with `doc` (the same draws, kit, references and results):
    each arm's index raised by its shift, the paired Δ recomputed against the shifted references, the dropped block."""
    out = copy.deepcopy(doc)
    idx = {a: round(v["index"] + shifts.get(a, 0.0), 2) for a, v in doc["arms"].items()}
    for arm, a in out["arms"].items():
        a["index"] = a["kit_index"] = idx[arm]
        a["index_ci"] = [round(idx[arm] - 3.4, 4), round(idx[arm] + 3.1, 4)]
        for key, ref in (("delta_vs_reference", doc["reference"]), ("delta_vs_second_reference", doc["second_reference"])):
            d = 0.0 if arm == ref else round(idx[arm] - idx[ref], 4)
            a[key] = {"point": d, "ci": [0.0, 0.0] if arm == ref else [round(d - 3.9, 4), round(d + 3.4, 4)]}
    out["rows"], out["latency_rows"] = doc["rows"] - 3, doc["latency_rows"] - 2
    out["dropped"] = {"benchmarks": [{"catalog_id": 36, "dataset": drop, "pool_rows": 3, "area": "retrieval",
                                      "area_weights_before": DROP_WEIGHTS[0], "area_weights_after": DROP_WEIGHTS[1]}],
                      "scored_edition": "0.3-drop-36", "area_weights": {"retrieval": 0.200229},
                      "method": "the kit's own retirement, in process (synthetic)"}
    return out


def with_sensitivity(s: dict, shifts: dict[str, float], **kw) -> Path:
    """Write the no-BRIGHT bootstrap beside the fixture's plotted one and a spec naming it."""
    root = Path(s["root"])
    doc = json.loads((root / "boot-a.json").read_text())
    (root / "nobright.json").write_text(json.dumps(without(doc, shifts, **kw)))
    return rewrite_spec(s, sensitivity={"path": "nobright.json", "drop": "BRIGHT"})
