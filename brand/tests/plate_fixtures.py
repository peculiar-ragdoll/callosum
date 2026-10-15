"""Synthetic inputs for the plate tests: a tiny test set (8 questions, 2 sources, 4 clusters), think profiles built as
dicts with budgets 0/256, and a rows arm directory, all shaped as callo and the plate driver write them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from callo import think

N = 8
SOURCES = ["alpha"] * 4 + ["beta"] * 4
CLUSTERS = ["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"]


def write_data(d: Path) -> tuple[Path, str]:
    path = d / "test-s1.jsonl"
    with open(path, "w") as f:
        for i in range(N):
            rec = {"state": {"x": i}, "questions": {"answer": {"type": "choice", "instructions": "?",
                                                               "criteria": {"a": "A", "b": "B"}, "label": "a"}},
                   "_meta": {"id": f"r{i}", "group_id": CLUSTERS[i], "source": SOURCES[i]}}
            f.write(json.dumps(rec) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def qids() -> list[str]:
    return [f"r{i}/answer" for i in range(N)]


def profile(name: str, sha: str, correct: dict[int, list[bool]], lat: dict[int, list[float]], *, cache: bool = False,
            gib: float = 4.0, started: str = "2026-10-08T10:00:00Z", build: str = "b50-4ebdf2c") -> dict:
    budgets = {}
    for b in sorted(correct):
        stats = think.latency_stats(lat[b], CLUSTERS, seed=0, strata=SOURCES)
        budgets[str(b)] = {"n": N, "acc": sum(correct[b]) / N, **stats, "point": f"{name}@{b}",
                           **({"think_mean": float(min(b, 100))} if b else {})}
    rows = [{"q": q, "cluster": CLUSTERS[i], "source": SOURCES[i], "label": 0, "n": 2,
             "b": {str(b): [correct[b][i], 0.5, 0.6, lat[b][i], min(b, 100), "natural" if b else None] for b in correct}}
            for i, q in enumerate(qids())]
    return {
        "callo_think_profile": 1, "metrics": think.METRICS_VERSION, "model_key": "sha256:" + name * 8, "model_name": name,
        "created": started,
        "comparable": {"data": {"name": "test-s1.jsonl", "sha256": sha, "questions": N, "strata": "source"},
                       "budgets": sorted(correct), "seed": 0, "warmup": 3, "audit": 4, "cache": cache,
                       "sampling_policy": "model",
                       "sampling": {"policy": "model", "pinned": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
                                    "origin": {"temperature": "gguf", "top_p": "gguf", "top_k": "gguf", "min_p": "callo"},
                                    "gguf": {}},
                       "readout": {"variant": "main", "top_k": 100, "max_depth": 8, "divergence_mass": 0.1,
                                   "callo_version": "0.2.0", "cache_prompt": cache, "user_delimiter": "<|im_start|>user",
                                   "prefix": "tokens"}},
        "environment": {"host": {"fingerprint": "sha256:" + "h" * 64, "chip": "Apple M2 Ultra", "memory_bytes": 64 * 2**30},
                        "build_info": build,
                        "model": {"file": f"{name}.gguf", "bytes": int(gib * 2**30), "mmproj": None, "mmproj_bytes": None},
                        "server": {"server_args": ["--cache-ram", "0"], "upstream_local": True, "gpu_busy": False},
                        "run": {"started": started, "ended": started}},
        "thinking": {"capable": len(correct) > 1, "reason": None if len(correct) > 1 else "unsupported_think_format"},
        "server_state": {"state": "clean"}, "audit": {"latency_model": "verified"},
        "drift": {"flagged": False, "therm": [{"at": "A", "limit": 100}]},
        "budgets": budgets, "row_fields": ["correct", "nll1", "pmax", "lat_ms", "k", "close"], "rows": rows,
    }


ALPHA_CORRECT = {0: [True, False, True, False, True, True, False, False],
                 256: [True, True, True, False, True, True, True, False]}
ALPHA_LAT = {0: [200.0, 210, 190, 205, 220, 215, 198, 202], 256: [1500.0, 1700, 1400, 1650, 1800, 1550, 1600, 1750]}
BETA_CORRECT = {0: [False, False, True, False, True, False, False, True],
                256: [False, False, True, False, True, False, False, False]}   # less accurate and slower: dominated
BETA_LAT = {0: [150.0, 160, 155, 165, 170, 158, 162, 168], 256: [3000.0, 3100, 2900, 3050, 3200, 2950, 3020, 3080]}
REF_CORRECT = [True, True, False, True, True, False, True, False]
REF_LAT = [500.0, 560, 520, 540, 610, 505, 530, 575]


def write_profile(d: Path, name: str, doc: dict) -> Path:
    p = d / f"{name}.think.json"
    p.write_text(json.dumps(doc))
    return p


def write_rows(d: Path, sha: str | None, correct: list[bool], lat: list[float], *, drop: int | None = None,
               error_at: int | None = None, gib: float | None = 17.75, timer: str = "client_remote",
               date: str = "2026-10-02") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "rows.jsonl", "w") as f:
        for i, q in enumerate(qids()):
            if i == drop:
                continue
            r = {"q": q, "id": f"r{i}", "question": "answer", "cluster": CLUSTERS[i], "source": SOURCES[i], "type": "choice",
                 "probs": {"a": 0.9 if correct[i] else 0.1, "b": 0.1 if correct[i] else 0.9}, "label": "a",
                 "correct": correct[i], "lat_ms": lat[i], "input_tokens": 100}
            if i == error_at:
                r = {"q": q, "status": 500, "error": "prompt over the ubatch", "lat_ms": lat[i]}
            f.write(json.dumps(r) + "\n")
    meta = {"data": {"sha256": sha}, "date": date, "timer": timer, "build": None}
    if gib is not None:
        meta["gib"] = gib
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def setup(tmp: Path, *, alpha_cache: bool = False) -> dict:
    """A results root with the data, two callo profiles and a reference rows arm; returns the spec (unresolved)."""
    data, sha = write_data(tmp)
    write_profile(tmp, "alpha", profile("alpha", sha, ALPHA_CORRECT, ALPHA_LAT, cache=alpha_cache, gib=20.8))
    write_profile(tmp, "beta", profile("beta", sha, BETA_CORRECT, BETA_LAT, gib=9.1))
    write_rows(tmp / "ref", sha, REF_CORRECT, REF_LAT)
    spec = {"data": "test-s1.jsonl", "ref": "ref", "ref_label": "Ref (native)", "latency": "mean", "callo_cache": False,
            "callo_run": "run1", "draft": True, "title": "A *plate*", "subtitle": "synthetic",
            "source_order": ["beta", "alpha"],
            "story": {"model": "alpha", "think": [256], "vs": "vs Ref's release code (transformers, MPS)"},
            "arms": [
                {"name": "alpha", "label": "Alpha 35B · UD-Q4_K_XL", "short": "Alpha", "kind": "callo",
                 "profile": "alpha.think.json", "runtime": "Callo · llama.cpp"},
                {"name": "beta", "label": "Beta 9B · Q8_0", "short": "Beta", "kind": "callo", "profile": "beta.think.json",
                 "runtime": "Callo · llama.cpp"},
                {"name": "ref", "label": "Ref 9B (native)", "short": "Ref", "kind": "rows", "rows": "ref",
                 "runtime": "transformers · bf16", "marker": "diamond", "note": "Ref: the release code, bf16, MPS."},
                {"name": "later", "label": "Later 27B · UD-Q4_K_XL", "kind": "rows", "rows": "later",
                 "runtime": "llama.cpp", "marker": "diamond-hollow", "timer": "client_http"},
            ]}
    (tmp / "spec.json").write_text(json.dumps(spec))
    return {"root": tmp, "sha": sha, "spec_path": tmp / "spec.json", "spec": spec}


EE_CORRECT = [True, True, True, False, True, True, True, False]       # alpha@256's accuracy, and faster
EE_LAT = [600.0, 640, 580, 700, 650, 620, 610, 690]


def with_early_exit(doc: dict, *, z: float = 0.95, cap: int = 256, gate_share: float = 0.25) -> dict:
    """A profile carrying an early-exit point at z and its gated twin, as calibrate --converge-probe writes them: both
    opt-in, never picked, and (here) faster than the budget they cap at the same accuracy."""
    stats = think.latency_stats(EE_LAT, CLUSTERS, seed=0, strata=SOURCES)
    entry = {"n": N, "acc": sum(EE_CORRECT) / N, **stats, "think_mean": 60.0, "cap": cap}
    key = repr(float(z))
    doc = {**doc, "ee": {key: entry}, "ee_gated": {key: {**entry, "gate": {"flagged": gate_share, "n": N}}}}
    doc["rows"] = [{**r, "e": {key: [EE_CORRECT[i], EE_LAT[i]]}, "g": {key: [EE_CORRECT[i], EE_LAT[i]]}}
                   for i, r in enumerate(doc["rows"])]
    return doc
