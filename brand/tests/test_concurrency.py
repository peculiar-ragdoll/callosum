"""The companion figure (brand/concurrency.py): every number round-trips to its source file and JSON pointer; every
condition is plotted; the s1_fallback case is in the fine print with its numbers and never plotted."""
from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import concurrency as C


def _v(src: str, ptr: str, doc: dict) -> dict:
    return {"value": C.get(doc, ptr), "src": src, "ptr": ptr}


def make(tmp: Path) -> dict:
    """A concurrency.json in extract_concurrency.py's shape, its numbers pointing into a source file under tmp."""
    raw = {"rows": {
        "s1_decode": {"s1": {"n": 356, "p50_ms": 316.2, "p95_ms": 439.3, "max_ms": 862.4}, "solo_decode_tok_s": 30.74,
                      "background": {"decode_tok_s_mean_inside_s1": 9.06, "decode_tok_s_mean": 12.09}},
        "s1_prefill_32k": {"s1": {"n": 100, "p50_ms": 2152.7, "p95_ms": 2852.4, "max_ms": 3043.2},
                           "background": {"prefill_tok_s_mean": 681.1}, "prompt_tokens_measured": 33122},
        "s1_fallback": {"on": {"p50_ms": 39197.8, "foreign_prefill_s_median": 40.77},
                        "off": {"p50_ms": 41028.6, "foreign_prefill_s_median": 42.13}}},
        "tiel_fallback": {"on": {"p50_ms": 30643.6, "foreign_prefill_s_median": 31.94, "n_samples": 8,
                                 "s1_fallback_count": 8, "prompt_tokens_measured": 33205},
                          "off": {"p50_ms": 31682.5, "foreign_prefill_s_median": 32.35, "n_samples": 8}},
        "cycles": [{"foreign": {"t_start": 1791726929.33}}],
        "solo": {"n": 376, "p50": 226.5, "p95": 320.1, "max": 494.0, "mean": 233.9, "rep50": 219.4, "rep95": 332.2},
        "floor": {"dp": 0.0, "dec": 0.0009, "pre": 0.00104},
        "ctx": {"n": 120, "p50": 352.3, "ov": 782.6, "ovn": 5}}
    (tmp / "res").mkdir(exist_ok=True)
    (tmp / "res" / "s.json").write_text(json.dumps(raw))
    S = "res/s.json"

    def v(ptr):
        return _v(S, ptr, raw)

    return {
        "plate_concurrency": 1,
        "series": [{"name": "qwen9b-v0.1", "label": "Qwen3.5-9B, callo v0.1, 2026-10-02", "model": "Qwen3.5-9B-Q8_0.gguf",
                    "build": "b50-4ebdf2c", "date": "2026-10-02", "harness": "callo v0.1 bench/acceptance.py",
                    "launch": "parallel 6 for s1_decode", "src": S,
                    "rows": [
                        {"id": "solo", "label": "Solo decision (reference)",
                         "note": "a think calibration's budget 0, 2026-10-08",
                         "date": "2026-10-08",
                         "n": v("/solo/n"), "mean": v("/solo/mean"), "p50": v("/solo/p50"), "p95": v("/solo/p95"),
                         "max": {"value": 494.0, "src": S, "ptr": "/solo", "via": "callo.think.point_stats"}},
                        {"id": "solo-shim", "label": "Solo (raw llama-server, no callo; same model file)",
                         "note": "raw llama-server, without callo",
                         "n": v("/solo/n"), "p50": v("/solo/p50"), "p95": v("/solo/p95"), "max": v("/solo/max"),
                         "repeat": {"p50": v("/solo/rep50"), "p95": v("/solo/rep95")}},
                        {"id": "decode", "label": "+ 5 agent decode streams", "n": v("/rows/s1_decode/s1/n"),
                         "p50": v("/rows/s1_decode/s1/p50_ms"), "p95": v("/rows/s1_decode/s1/p95_ms"),
                         "max": v("/rows/s1_decode/s1/max_ms"),
                         "agent": {"kind": "decode", "unit": "tok/s", "solo": v("/rows/s1_decode/solo_decode_tok_s"),
                                   "with": v("/rows/s1_decode/background/decode_tok_s_mean_inside_s1"),
                                   "mean_all": v("/rows/s1_decode/background/decode_tok_s_mean")}},
                        {"id": "prefill-32k", "label": "+ agent 32k prefill, continuously re-issued",
                         "n": v("/rows/s1_prefill_32k/s1/n"), "p50": v("/rows/s1_prefill_32k/s1/p50_ms"),
                         "p95": v("/rows/s1_prefill_32k/s1/p95_ms"), "max": v("/rows/s1_prefill_32k/s1/max_ms"),
                         "agent": {"kind": "prefill", "unit": "tok/s",
                                   "with": v("/rows/s1_prefill_32k/background/prefill_tok_s_mean"),
                                   "prompt_tokens": v("/rows/s1_prefill_32k/prompt_tokens_measured")}}],
                    "fallback": {"on_p50": v("/rows/s1_fallback/on/p50_ms"), "off_p50": v("/rows/s1_fallback/off/p50_ms"),
                                 "prefill_on_s": v("/rows/s1_fallback/on/foreign_prefill_s_median"),
                                 "prefill_off_s": v("/rows/s1_fallback/off/foreign_prefill_s_median")}}],
        "context": [{"id": "tiel-2hz", "label": "Context: Tiel vision ticks at 2 Hz", "model": "Tiel.gguf",
                     "date": "2026-10-03", "launch_flags": ["--mmproj", "/x/mmproj-BF16.gguf", "--s1-slots", "2"],
                     "n": v("/ctx/n"), "p50": v("/ctx/p50"), "overlap_p50": v("/ctx/ov"), "overlap_n": v("/ctx/ovn")}],
        "floor": {"label": "an earlier raw llama-server measurement without callo, on the same model file",
                  "solo_p50": v("/solo/p50"), "repeat_p50": v("/solo/rep50"),
                  "repeat_dp_max": v("/floor/dp"), "decode_dp_max": v("/floor/dec"), "prefill32k_dp_max": v("/floor/pre")},
        "gaps": [f"gap number {i}." for i in range(1, 9)],
        "sources": [S],
        "source_roles": {S: "the test run"},
    }


def with_tiel_fallback(doc: dict, tmp: Path, commit: str = "aaaaaaa") -> dict:
    """The doc plus a model measured for the fallback case only (extract_concurrency.py --tiel-fallback)."""
    S = "res/s.json"
    raw = json.loads((tmp / S).read_text())

    def v(ptr):
        return _v(S, ptr, raw)

    doc["fallback_only"] = [{
        "name": "tiel", "label": "Tiel-Coder 35B-A3B", "model": "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
        "build": "b50-4ebdf2c", "callo_version": "0.3.0", "callo_commit": commit, "cache": "on",
        "harness": "callo bench/acceptance.py", "src": S, "started": v("/cycles/0/foreign/t_start"), "date": "2026-10-11",
        "fallback": {"on_p50": v("/tiel_fallback/on/p50_ms"), "off_p50": v("/tiel_fallback/off/p50_ms"),
                     "prefill_on_s": v("/tiel_fallback/on/foreign_prefill_s_median"),
                     "prefill_off_s": v("/tiel_fallback/off/foreign_prefill_s_median"),
                     "prompt_tokens": v("/tiel_fallback/on/prompt_tokens_measured"),
                     "n_on": v("/tiel_fallback/on/n_samples"), "n_off": v("/tiel_fallback/off/n_samples"),
                     "moved_on": v("/tiel_fallback/on/s1_fallback_count")}}]
    return doc


# commit -> what frontier.commit_info would say of it in a checkout that knows it
COMMITS = {"aaaaaaa": {"sha": "a" * 40, "short": "aaaaaaa", "version": "0.3.0", "time": "2026-10-11T00:11:16Z",
                       "on_head": True},
           "bbbbbbb": {"sha": "b" * 40, "short": "bbbbbbb", "version": "0.2.0", "time": "2026-10-01T00:00:00Z",
                       "on_head": True},
           "ccccccc": {"sha": "c" * 40, "short": "ccccccc", "version": "0.3.0", "time": "2026-10-12T00:00:00Z",
                       "on_head": True},
           "ddddddd": {"sha": "d" * 40, "short": "ddddddd", "version": "0.3.0", "time": "2026-10-10T00:00:00Z",
                       "on_head": False}}


@pytest.fixture
def commits(monkeypatch):
    import figkit
    monkeypatch.setattr(C, "_commit", COMMITS.get)
    monkeypatch.setattr(figkit, "builds", lambda: [{"commit": c, "version": v["version"], "name": f"build {c[0]}",
                                                    "what": f"what {c[0]} added"} for c, v in COMMITS.items()])


def _build(tmp_path, doc):
    (tmp_path / "c.json").write_text(json.dumps(doc))
    return C.build(tmp_path / "c.json", tmp_path)


def _doc(tmp_path, **kw):
    return with_tiel_fallback(make(tmp_path), tmp_path, **kw)


@pytest.fixture
def snap(tmp_path) -> dict:
    (tmp_path / "c.json").write_text(json.dumps(make(tmp_path)))
    return C.build(tmp_path / "c.json", tmp_path)


def _svg(snap, theme="dark"):
    return ET.fromstring(C.draw(snap, theme=theme))


def _role(root, role):
    return [e for e in root.iter() if e.get("data-role") == role]


def test_every_plotted_number_round_trips_with_its_source(snap):
    root = _svg(snap)
    drawn = {e.get("data-key"): e for e in root.iter() if e.get("data-key")}
    want = {}
    for s in snap["series"]:
        for r in s["rows"]:
            if r["id"] in C.DROPPED:
                continue
            for k in ("p50", "p95", "max"):
                want[f"{s['name']}.{r['id']}.{k}"] = r[k]
            for k, v in (r.get("agent") or {}).items():
                if k in ("solo", "with"):
                    want[f"{s['name']}.{r['id']}.agent.{k}"] = v
    for c in snap["context"]:
        want[f"context.{c['id']}.p50"] = c["p50"]
        want[f"context.{c['id']}.overlap_p50"] = c["overlap_p50"]
    assert set(want) <= set(drawn)
    for k, v in want.items():
        assert float(drawn[k].get("data-value")) == v["value"]
        assert drawn[k].get("data-src") == f"{v['src']}#{v['ptr']}"
    assert snap["verified_numbers"] > 30


def test_every_condition_is_plotted_and_the_shim_row_is_not(snap):
    root = _svg(snap)
    rows = {e.get("data-row") for e in _role(root, "row-label")}
    assert rows == {"qwen9b-v0.1.solo", "qwen9b-v0.1.decode", "qwen9b-v0.1.prefill-32k", "context.tiel-2hz"}
    assert not any("solo-shim" in (e.get("data-key") or "") for e in root.iter())


def test_the_solo_row_says_another_session_and_throughput_is_qualified(snap):
    root = _svg(snap)
    texts = [e.get("data-text") or "" for e in root.iter()]
    assert any(t == "n 376 · mean 234 ms · another session, 2026-10-08" for t in texts)
    notes = [e.get("data-text") for e in _role(root, "agent-note")]
    assert "per decode stream: solo → while decisions run back-to-back (~3/s)" in notes
    assert not any("mean 12" in (n or "") for n in notes)
    assert "30.7 → 9.1 tok/s" in [e.get("data-text") for e in _role(root, "agent-text")]


def test_the_axis_ends_at_5_s_and_the_context_row_has_its_own_marks(snap):
    root = _svg(snap)
    ticks = [e.get("data-text") for e in _role(root, "tick")]
    assert ticks[-1] == "5 s"
    ctx = [e for e in root.iter() if (e.get("data-key") or "").startswith("context.")]
    assert ctx and all(e.tag.endswith("path") for e in ctx)              # not the solo rows' hollow circle


def test_title_and_subtitle(snap):
    root = _svg(snap)
    (title,) = _role(root, "title")
    assert title.get("data-text") == "Decisions stay fast while the agent decodes"
    (sub,) = _role(root, "subtitle")
    assert sub.get("data-text") == "A long agent prefill still costs seconds: 2.2 s p50 at 32k (Qwen3.5-9B)."


def test_every_colour_is_from_the_brand_palette(snap):
    import re

    import figkit
    palette = {figkit.INK, figkit.BONE, figkit.SIGNAL, figkit.NIGHT, figkit.DUSK, figkit.TILE, figkit.MIST, figkit.SLATE}
    for theme in ("dark", "light"):
        svg = C.draw(snap, theme=theme)
        assert {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", svg)} <= palette
    # small text on the light ground is Slate, never Mist (Mist on Bone fails AA)
    light = _svg(snap, "light")
    for g in light.iter():
        if g.get("data-role") in ("num", "tick", "agent-note", "fine-print"):
            assert figkit.MIST not in ET.tostring(g).decode()


def test_the_fallback_case_is_fine_print_only_with_its_numbers(snap):
    root = _svg(snap)
    fine = " ".join(g.get("data-text") for g in _role(root, "fine-print"))
    assert ("A foreign prefill in the reserved slot itself still blocks: 39 s, not plotted (Qwen3.5-9B: p50 39.2 s "
            "with the S1 fallback, 41.0 s without).") in fine
    made = " ".join(f["lead"] + " " + f["text"] for f in snap["made"])
    assert ("Not plotted: the S1 fallback under a foreign prefill. A foreign 32k prefill in the reserved slot itself. "
            "Qwen3.5-9B (the plotted run, 2026-10-02): p50 39.2 s with the S1 fallback, 41.0 s without (the prefill "
            "took 40.8 s median with it, 42.1 s without). The fallback helps against a foreign decode, not a foreign "
            "prefill.") in made
    assert "pending" not in fine + made
    plotted = [e for e in root.iter() if e.get("data-key") or e.get("data-row")]
    assert not any("fallback" in (e.get("data-key") or e.get("data-row") or "") for e in plotted)
    assert not any("39197" in (e.get("data-value") or "") for e in plotted)


def test_a_fallback_only_model_is_fine_print_only_with_its_provenance(tmp_path, commits):
    snap = _build(tmp_path, _doc(tmp_path))
    fine = " ".join(f["text"] for f in snap["fine_print"])
    assert ("A foreign prefill in the reserved slot itself still blocks: 31–39 s, not plotted (Qwen3.5-9B: p50 39.2 s "
            "with the S1 fallback, 41.0 s without; Tiel-Coder 35B-A3B on Callo 0.3.0, 2026-10-11: p50 30.6 s with, "
            "31.7 s without).") in fine
    assert len(snap["fine_print"]) == 3
    made = " ".join(f["text"] for f in snap["made"])
    assert ("Tiel-Coder 35B-A3B (Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf on `callo` 0.3.0, build a, --cache on, llama.cpp "
            "b50-4ebdf2c, 2026-10-11): p50 30.6 s with the S1 fallback, 31.7 s without (the prefill took 31.9 s median "
            "with it, 32.4 s without; 8 decisions with it, all moved off the reserved slot, and 8 without; a "
            "33,205-token prompt). The fallback helps against a foreign decode, not a foreign prefill.") in made
    assert made.count("The fallback helps against a foreign decode") == 1
    assert "aaaaaaa" not in made and snap["builds"] == [{"commit": "aaaaaaa", "name": "build a"}]   # by role
    root = _svg(snap)
    assert not any((e.get("data-key") or e.get("data-row") or "").startswith("tiel.") for e in root.iter())
    assert not any("30643" in (e.get("data-value") or "") for e in root.iter())
    assert snap["subtitle"] == "A long agent prefill still costs seconds: 2.2 s p50 at 32k (Qwen3.5-9B)."


@pytest.mark.parametrize("commit, why", [
    ("eeeeeee", "is not a commit of this repository"), ("bbbbbbb", "is callo 0.2.0, the run says 0.3.0"),
    ("ccccccc", "is later than its fallback cycles started"), ("ddddddd", "is not on HEAD's history")])
def test_a_fallback_only_commit_pin_is_checked(tmp_path, commits, commit, why):
    with pytest.raises(C.Refusal, match=why):
        _build(tmp_path, _doc(tmp_path, commit=commit))


def test_the_snapshot_records_paths_relative_to_the_root(tmp_path, commits):
    snap = _build(tmp_path, _doc(tmp_path))
    assert str(tmp_path) not in json.dumps(snap, ensure_ascii=False)
    assert snap["source_file"] == "c.json"
    assert snap["context"][0]["launch_flags"] == ["--mmproj", "mmproj-BF16.gguf", "--s1-slots", "2"]   # outside it


def test_a_fallback_only_commit_the_builds_table_does_not_name_is_refused(tmp_path, commits, monkeypatch):
    import figkit
    monkeypatch.setattr(figkit, "builds", lambda: [])
    with pytest.raises(C.Refusal, match="aaaaaaa is not in brand/data/builds.json"):
        _build(tmp_path, _doc(tmp_path))


def test_sources_are_named_by_role_file_and_sha256(tmp_path, commits):
    import hashlib
    snap = _build(tmp_path, _doc(tmp_path))
    sha = hashlib.sha256((tmp_path / "res" / "s.json").read_bytes()).hexdigest()
    assert snap["source_files"] == [{"path": "res/s.json", "role": "the test run", "name": "s.json", "sha256": sha}]
    (src,) = [f for f in snap["made"] if f["lead"] == "Sources."]
    assert src["text"] == f"the test run: `s.json` (sha256 {sha[:16]}…)."


def test_a_plotted_tiel_series_gets_its_own_fallback_sentence(tmp_path):
    doc = make(tmp_path)
    tiel = copy.deepcopy(doc["series"][0])
    tiel.update(name="tiel", label="Tiel-Coder 35B-A3B, callo release, 2026-10-12")
    doc["series"].append(tiel)
    (tmp_path / "c.json").write_text(json.dumps(doc))
    snap = C.build(tmp_path / "c.json", tmp_path)
    fine = " ".join(f["text"] for f in snap["fine_print"])
    assert "Tiel-Coder 35B-A3B: p50 39.2 s with, 41.0 s without" in fine and "pending" not in fine
    made = " ".join(f["text"] for f in snap["made"])
    assert "Tiel-Coder 35B-A3B (the plotted run, 2026-10-02): p50 39.2 s with the S1 fallback" in made
    assert "pending" not in made
    assert snap["subtitle"].endswith("(Tiel-Coder 35B-A3B).")
    assert {e.get("data-row") for e in _role(_svg(snap), "row-label")} >= {"tiel.decode", "qwen9b-v0.1.decode"}


def test_the_caption_is_three_lines_and_the_rest_is_in_figures_md(snap):
    assert len(snap["fine_print"]) == 3
    fine = " ".join((f["lead"] or "") + " " + f["text"] for f in snap["fine_print"])
    for s in ("Qwen3.5-9B-Q8_0, llama.cpp b50-4ebdf2c, 2026-10-02. callo v0.1 bench/acceptance.py; parallel 6 for "
              "s1_decode.", "Solos from another session (a think calibration's budget 0, 2026-10-08).",
              "A foreign prefill in the reserved slot itself still blocks: 39 s"):
        assert s in fine, s
    made = " ".join((f["lead"] or "") + " " + f["text"] for f in snap["made"])
    for s in ("README limit.", C.README_LIMIT, "Run-to-run floor.", "|Δp| max 0.0, p50 226 vs 219 ms",
              "max 0.0009 under decode, 0.0010 under 32k prefill", "Its solo was p50 226 / p95 320 ms",
              "the nearest is an earlier raw llama-server measurement without callo, on the same model file",
              "Context row.", "a different model and session", "--mmproj mmproj-BF16.gguf --s1-slots 2",
              *[f"({i}) gap number {i}." for i in range(1, 9)]):
        assert s in made, s
    for s in ("README limit", "Run-to-run floor", "gap number"):
        assert s not in fine
    import figures_md
    md = figures_md.render(None, snap)
    assert C.README_LIMIT in md and "(8) gap number 8." in md
    root = _svg(snap)
    texts = [g.get("data-text") for g in _role(root, "fine-print")]
    for f in snap["fine_print"]:
        assert any(f["text"] in t for t in texts)


def test_the_readme_limit_is_quoted_from_the_readme():
    q = C.readme_limit()
    assert q.startswith(C.README_LIMIT) and "39.2 s" in q


def test_a_number_that_does_not_round_trip_is_refused(tmp_path):
    doc = make(tmp_path)
    doc["series"][0]["rows"][2]["p50"]["value"] = 999.0
    (tmp_path / "c.json").write_text(json.dumps(doc))
    with pytest.raises(C.Refusal, match="the figure data says 999.0"):
        C.build(tmp_path / "c.json", tmp_path)
    assert C.main(["build", "--src", str(tmp_path / "c.json"), "--root", str(tmp_path), "--out",
                   str(tmp_path / "o.json")]) == 2


def test_a_missing_source_is_refused(tmp_path):
    doc = make(tmp_path)
    (tmp_path / "c.json").write_text(json.dumps(doc))
    with pytest.raises(C.Refusal, match="not found"):
        C.build(tmp_path / "c.json", tmp_path / "elsewhere")


def test_both_themes(snap):
    dark, light = _svg(snap, "dark"), _svg(snap, "light")
    assert dark.get("data-theme") == "dark" and light.get("data-theme") == "light"
    assert [e.get("data-key") for e in dark.iter() if e.get("data-key")] == \
           [e.get("data-key") for e in light.iter() if e.get("data-key")]


BENCH = Path(os.environ["CALLO_BENCH_ROOT"]) if os.environ.get("CALLO_BENCH_ROOT") else None    # the benchmark's root
REAL = BENCH / "harness/results/callo_v0.3/plate/concurrency.json" if BENCH else None


@pytest.fixture(scope="module")
def real() -> dict:
    if REAL is None or not REAL.is_file():
        pytest.skip(f"no extract_concurrency.py output under {BENCH}" if BENCH else "CALLO_BENCH_ROOT (the benchmark's root) is not set")
    return C.build(REAL, BENCH)


def test_real_extract_verifies_against_its_sources(real):
    snap = real
    assert snap["verified_numbers"] >= 60
    fine = " ".join(f["text"] for f in snap["fine_print"])
    assert ("A foreign prefill in the reserved slot itself still blocks: 31–39 s, not plotted (Qwen3.5-9B: p50 39.2 s "
            "with the S1 fallback, 41.0 s without; Tiel-Coder 35B-A3B on Callo 0.3.0, 2026-10-11: p50 30.6 s with, "
            "31.7 s without).") in fine
    made = " ".join(f["text"] for f in snap["made"])
    assert "p50 39.2 s with the S1 fallback, 41.0 s without (the prefill took 40.8 s median" in made
    assert ("Tiel-Coder 35B-A3B (Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf on `callo` 0.3.0, the refit build, --cache on, llama.cpp "
            "b50-4ebdf2c, 2026-10-11): p50 30.6 s with the S1 fallback, 31.7 s without (the prefill took 31.9 s median "
            "with it, 32.4 s without; 8 decisions with it, all moved off the reserved slot, and 8 without; a "
            "33,205-token prompt).") in made
    assert snap["subtitle"] == "A long agent prefill still costs seconds: 2.2 s p50 at 32k, 3.3 s at 128k (Qwen3.5-9B)."
    assert "9d52aaf" not in made and snap["builds"] == [{"commit": "9d52aaf", "name": "the refit build"}]
    assert snap["source_file"] == "harness/results/callo_v0.3/plate/concurrency.json"


def test_real_public_text_names_no_runs_gates_shims_or_paths(real):
    root = ET.fromstring(C.draw(real))
    shown = " ".join(e.get("data-text") or "" for e in root.iter())
    import figures_md
    md = figures_md.render(None, real)
    md = md[md.index("## The companion figure"):]                  # the intro's rebuild block has a <bench> placeholder
    md = md[:md.index("\n## ", 1)] if "\n## " in md[1:] else md
    for text in (shown, md):
        for bad in ("pending", "G4", "stage-1", "shim", "run1", "callo_v0", "s1_decode", "s1_prefill", "/Users/",
                    "bench-qwen36", "harness/results"):
            assert bad not in text, bad
    assert "Sources." in md and "`summary.json` (sha256 1d9543b9ccf7d64c…)" in md


def test_the_committed_snapshot_is_the_real_build(real):
    committed = json.loads((Path(C.HERE) / "data" / "concurrency.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(real, ensure_ascii=False))
