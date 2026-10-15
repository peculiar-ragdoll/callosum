"""Pick preferences in the think bench: T10's p50 verdict unchanged, with the mean, p95 and max reported beside it
and the rule that picked its budget; the five latency statistics per budget and the recorded pick beside the mean
re-pick in the descriptive results and the doc. Profiles are synthetic or the run1 fixture; no server."""
import json

import pytest
from test_compare import A
from test_pick_golden import run1

from bench import acceptance_v03_think as v03t
from bench.report_v03_think import render

PROFILE = {"row_fields": ["correct", "nll1", "pmax", "lat_ms", "k", "close"], "pick": {"budget": 256, "latency": "p50"},
           "rows": [{"q": f"q{i}", "b": {"256": [True, 0.5, 0.6, 1000.0 + 10 * i, 256, "forced"]}} for i in range(48)]}
SERVED = {f"q{i}": 1100.0 + 10 * i for i in range(48)}


def test_t10_keeps_its_p50_verdict_and_reports_mean_p95_and_max_beside_it():
    g = v03t.t10(SERVED, PROFILE, 256)
    assert (g["served_p50_ms"], g["profile_p50_ms"], g["pass"]) == (1335.0, 1235.0, True)       # unchanged
    assert g["served_mean_ms"] == pytest.approx(1335.0) and g["profile_mean_ms"] == pytest.approx(1235.0)
    assert g["mean_ratio"] == pytest.approx(1335 / 1235)
    assert (g["served_p95_ms"], g["profile_p95_ms"], g["p95_from"]) == (1550.0, 1450.0, 3)     # rank 46 of 48
    assert (g["served_max_ms"], g["profile_max_ms"], g["pick_latency"]) == (1570.0, 1470.0, "p50")
    row = render({"gates": {"t10": {"tiel": {**g, "budget_note": "the profile's pick"}}}}).split("| T10 | ")[1]
    assert ("tiel@256 (the profile's pick, picked on p50): served p50 1335 vs profile 1235 ms over 48 questions (error "
            "0.081); beside it, not gated: mean 1335 vs 1235 ms (ratio 1.081), p95 1550 vs 1450 ms (from 3 of 48), max "
            "1570 vs 1470 ms | PASS |") in row
    old = {"n": 48, "budget": 256, "served_p50_ms": 1500.0, "profile_p50_ms": 1400.0, "rel_err": 0.071, "pass": True}
    assert "(error 0.071) | PASS |" in render({"gates": {"t10": {"tiel": old}}})              # an older summary


def test_the_descriptive_results_carry_the_five_statistics_and_the_mean_re_pick():
    d = v03t.descriptive({"alpha": A})["alpha"]
    assert set(d["lat"]) == {"0", "256"} and d["lat"]["0"]["n"] == 10
    assert d["lat"]["0"]["mean"] == pytest.approx(300.0 * 1.045) and d["lat"]["0"]["max"] == pytest.approx(327.0)
    r = d["repick"]
    assert (r["recorded"], r["recorded_on"]) == ("alpha@0", "p50") and r["mean"] is not None
    assert v03t.descriptive({"g": {"budgets": {"0": {"n": 10, "acc": 0.5}}}})["g"]["repick"] is None   # unreadable: None


def test_run1_tiels_recorded_pick_and_its_mean_re_pick():
    """On the fixture: recorded tiel@1024 on p50; the mean re-pick tiel@512, within noise of tiel@1024
    with 0 vs 2 discordant questions."""
    r = v03t.repick(run1())
    assert (r["recorded"], r["recorded_on"], r["mean"], r["mean_best"]) == ("tiel@1024", "p50", "tiel@512", "tiel@1024")
    assert [round(100 * x, 1) for x in r["mean_dacc_best"]] == [-0.5, -1.3, 0.0] and r["discordant"] == [0, 2]
    lat = v03t.latencies(run1())
    assert round(lat["1024"]["p99"] / 1000, 2) == 10.0 and round(lat["1024"]["max"] / 1000, 2) == 11.44


def test_the_doc_renders_the_latency_table_and_the_pick_table():
    s = {"descriptive": v03t.descriptive({"tiel": run1()})}
    text = render(s)
    assert "### Latency per budget: mean, p50, p95, p99, max" in text
    assert "| tiel | 1024 | 376 | 1.91s | 1.36s | 4.35s | 10.00s | 11.44s |" in text
    assert "tails (n 376): p95 from 19 questions, p99 from 4, max from 1" in text
    assert "### The pick: as recorded (p50) and the mean re-pick" in text
    assert ("| tiel | tiel@1024 (p50) | tiel@1024 | +0.0pp [+0.0, +0.0] | tiel@512 | tiel@1024 | -0.5pp [-1.3, +0.0] "
            "| 0 / 2 |") in text
    small = render({"descriptive": v03t.descriptive({"alpha": A})})
    assert "| alpha | 0 | 10 | 0.31s | 0.30s | 0.45s | = max | 0.33s |" in small     # p99's rank is n below 100


def test_the_rerender_is_read_only_on_the_results_directory(tmp_path, monkeypatch):
    """`--rerender` writes only the doc. Every file under --out is byte-identical afterwards, nothing is added,
    and the seams of a real session (the server check, `llama-server --version`, data selection, the offline gates)
    are never reached."""
    out, doc = tmp_path / "think", tmp_path / "docs" / "v0.3-think.md"
    prof = out / "run1" / "tiel.think.json"
    prof.parent.mkdir(parents=True)
    p = run1()
    prof.write_text(json.dumps(p))
    (out / "run1" / "run.json").write_text(json.dumps({
        "callo_think_run": 1, "menu_sha256": "x", "settings": {}, "created": "t", "order": ["tiel"], "dropped": [],
        "models": {"tiel": {"status": "done", "reason": None, "started": "t", "ended": "t", "profile": str(prof)}}}))
    served = {r["q"]: r["b"]["1024"][3] for r in p["rows"][:48]}
    (out / "summary.json").write_text(json.dumps({
        "gates": {"t10": {"tiel": {"n": 48, "budget": 1024, "served_p50_ms": 1.0, "profile_p50_ms": 1.0, "pass": True}}},
        "serve": {"tiel": {"served_ms": served, "budget": 1024, "budget_note": "the profile's pick"}}, "errors": {}}))
    before = {f: f.read_bytes() for f in out.rglob("*") if f.is_file()}

    def boom(*a, **k):
        raise AssertionError("a re-render must not reach this")
    for name in ("_llama_server_version", "select_records", "_offline_gates", "disk_problems", "_run"):
        monkeypatch.setattr(v03t, name, boom)
    monkeypatch.setattr(v03t, "code_state", lambda: {"commit": "abc1234def5678", "dirty": False})
    monkeypatch.setattr(v03t.lib, "running_gpu_servers", boom)
    monkeypatch.setattr(v03t.lib, "free_memory_bytes", boom)
    assert v03t.main(["--rerender", "--out", str(out), "--doc", str(doc)]) == 0
    assert {f: f.read_bytes() for f in out.rglob("*") if f.is_file()} == before
    text = doc.read_text()
    assert "tiel@1024 (the profile's pick, picked on p50): served p50 " in text and "beside it, not gated: mean " in text
    assert "| tiel | 1024 | 376 | 1.91s | 1.36s | 4.35s | 10.00s | 11.44s |" in text
    assert "| tiel | tiel@1024 (p50) |" in text
    assert v03t.main(["--rerender", "--out", str(tmp_path / "none"), "--doc", str(doc)]) == 2     # no summary.json
    assert not (tmp_path / "none").exists()


def test_the_rerendered_doc_carries_a_provenance_note_in_memory_only(tmp_path, monkeypatch):
    """The T10 extras and the new tables come from later code than the recorded run, so the doc
    says it was re-rendered by `--rerender` at the HEAD commit; the note is never saved to summary.json."""
    out, doc = tmp_path / "think", tmp_path / "v0.3-think.md"
    out.mkdir()
    raw = json.dumps({"gates": {}, "errors": {}, "provenance": {}, "data": {}})
    (out / "summary.json").write_text(raw)
    monkeypatch.setattr(v03t, "code_state", lambda: {"commit": "abc1234def5678", "dirty": True})
    assert v03t.main(["--rerender", "--out", str(out), "--doc", str(doc)]) == 0
    assert ("Re-rendered offline by `--rerender` at callo abc1234def56 with uncommitted changes; T10 (its p50 "
            "verdict and the served-vs-profile mean/p95/max), the latency table, the mean re-pick table and the "
            "descriptive tables are recomputed by that code from the recorded run's summary.json and profiles; every "
            "other gate verdict renders as the run recorded it.") in doc.read_text()
    monkeypatch.setattr(v03t, "code_state", lambda: {"commit": "abc1234def5678", "dirty": False})
    v03t.main(["--rerender", "--out", str(out), "--doc", str(doc)])
    assert "at callo abc1234def56;" in doc.read_text()
    assert (out / "summary.json").read_text() == raw


def test_a_corrupt_summary_exits_2_cleanly_and_an_unreadable_budget_renders_an_na_row(tmp_path):
    out = tmp_path / "think"
    out.mkdir()
    (out / "summary.json").write_text("{not json")
    assert v03t.main(["--rerender", "--out", str(out), "--doc", str(tmp_path / "d.md")]) == 2
    assert not (tmp_path / "d.md").exists()
    p = run1()
    p["budgets"]["256"].pop("lat_p50_ms", None)
    p["rows"][0]["b"]["256"] = []                       # point_stats cannot read this budget's rows
    assert v03t.latencies(p)["256"].get("unreadable") is True
    text = render({"descriptive": v03t.descriptive({"tiel": p})})
    assert "| tiel | 256 | " in text and "| n/a | n/a | n/a | n/a | n/a |" in text


def test_odd_shapes_are_na_rows_or_a_clean_exit_2(tmp_path):
    p = run1()
    p["budgets"]["junk"] = {"n": 5}
    p["budgets"]["77"] = "not a dict"
    lat = v03t.latencies(p)
    assert lat["junk"]["unreadable"] and lat["77"]["unreadable"] and lat["1024"]["n"] == 376
    out = tmp_path / "think"
    out.mkdir()
    for bad in ({"errors": []}, {"gates": "x"}, {"errors": None}):
        (out / "summary.json").write_text(json.dumps(bad))
        assert v03t.main(["--rerender", "--out", str(out), "--doc", str(tmp_path / "d.md")]) == 2
    assert not (tmp_path / "d.md").exists()
