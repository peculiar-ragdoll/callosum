"""The v0.3 think-budgets acceptance bench, GPU-free: the data selection, the gate functions on synthetic profiles and
raw files, the report renderer, and (appended later) a smoke run of the whole orchestration through its seams on the
fake llama-server executable and the thinking fake upstream."""
import json
from pathlib import Path

import pytest
from test_calibrate_think import SAMPLING, STATS, synthetic_raw, write_data

from bench import acceptance_v03_think as v03t
from bench.report_v03_think import GATES, render
from bench.report_v03_think import main as report_main
from callo.calibrate_think import RAW_FORMAT, ThinkSettings, build_profile, load_data, read_raw, shuffled


def _record(i: int, source: str, variant: str = "clean", qtype: str = "choice") -> str:
    q = ({"type": "score", "instructions": "how late?", "criteria": ["on time", "late", "refused"], "label": 0} if qtype == "score"
         else {"type": "noul", "instructions": "ok?", "label": True} if qtype == "noul"
         else {"type": "choice", "instructions": "pick", "criteria": {"a": "x", "b": "y"}, "label": "a"})
    return json.dumps({"state": f"state {i}", "questions": {"q": q},
                       "_meta": {"id": f"{source}/{i}", "group_id": f"{source}/{i}", "variant": variant, "source": source}})


def test_select_records_keeps_the_clean_records_of_the_sources_the_date_score_group_or_a_seeded_subset(tmp_path):
    src = tmp_path / "test.jsonl"
    lines = [_record(i, s) for i, s in enumerate(["mmlu", "qnli", "paws", "composition_holdout", "emotion", "sciq"])]
    lines += [_record(9, "mmlu", "permuted"), _record(10, "mmlu", "none_present")]
    lines += [_record(11, "contrastive", qtype="score"), _record(12, "contrastive", qtype="noul")]   # date-score vs the rest
    src.write_text("\n".join(lines) + "\n\n")
    assert v03t.select_records(src, tmp_path / "d.jsonl") == 5
    kept = [json.loads(ln)["_meta"]["source"] for ln in (tmp_path / "d.jsonl").read_text().splitlines()]
    assert kept == ["mmlu", "qnli", "paws", "composition_holdout", "contrastive"]
    assert json.loads((tmp_path / "d.jsonl").read_text().splitlines()[-1])["questions"]["q"]["type"] == "score"
    assert v03t.select_records(src, tmp_path / "all.jsonl", sources=(), score_sources=()) == 8
    assert v03t.select_records(src, tmp_path / "ds.jsonl", sources=()) == 1                           # the score group alone
    assert v03t.GROUPS["contrastive"] == "date-score" and v03t.groups_of(src)["contrastive/11"] == "date-score"
    assert v03t.groups_of(src)["mmlu/0"] == "control" and v03t.groups_of(src)["emotion/4"] == "emotion"
    assert v03t.select_records(src, tmp_path / "sub.jsonl", limit=2, seed=0) == 2
    a = (tmp_path / "sub.jsonl").read_text()
    v03t.select_records(src, tmp_path / "sub2.jsonl", limit=2, seed=0)
    assert a == (tmp_path / "sub2.jsonl").read_text() and len(a.splitlines()) == 2
    assert v03t.select_records(src, tmp_path / "none.jsonl", sources=("nope",), score_sources=()) == 0
    assert (tmp_path / "none.jsonl").read_text() == ""


def test_menu_doc_has_the_run_1_launch_shape_and_per_entry_extras():
    doc = v03t.menu_doc({"tiel": {"model": "/t.gguf"}, "tiel-fixed": {"model": "/t.gguf", "think_sampling": "fixed"}},
                        llama_server="/opt/llama-server")
    assert doc == {"models": [{"name": "tiel", "ctx": 65536, "parallel": 4, "model": "/t.gguf", "llama_server": "/opt/llama-server"},
                              {"name": "tiel-fixed", "ctx": 65536, "parallel": 4, "model": "/t.gguf", "think_sampling": "fixed",
                               "llama_server": "/opt/llama-server"}]}
    assert v03t.menu_doc({"q": {"model": "/q.gguf", "llama_server": "/own"}}, ctx=4096, llama_server="/opt")["models"][0]["llama_server"] == "/own"


AUDIT48 = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 64, 256), audit=48)    # the bench's --audit 48


def _raw_file(tmp_path, items, **kw) -> Path:
    raw, settings = synthetic_raw(items, **{"settings": AUDIT48, **kw})
    path = tmp_path / "m.raw.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [{"callo_think_raw": RAW_FORMAT, **{k: v for k, v in raw.header.items()
                                                                                        if k != "callo_think_raw"}}, *raw.rows]) + "\n")
    return path, raw, settings


def test_t1_t2_read_the_audit_rows_and_gate_only_with_enough_cuts(tmp_path, monkeypatch):
    data = write_data(tmp_path / "d.jsonl", 8)
    items, _ = load_data(data)
    raw_path, raw, settings = _raw_file(tmp_path, items)                 # 4 cut items audited at 64, 2 % latency error
    profile = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    g = v03t.t1_t2(profile, raw_path, data)
    e = g["budgets"]["64"]
    assert (e["n_cut"], e["n"], e["identical"], e["argmax_agree"]) == (4, 4, 4, 4) and e["dp_median"] == 0.0
    # a resumed audit (the same question audited twice) counts once; the profile's own |Δp| wins when recorded
    dup_path = tmp_path / "dup.raw.jsonl"
    rows = raw_path.read_text().splitlines()
    audit = next(ln for ln in rows if '"phase": "audit"' in ln)
    dup_path.write_text("\n".join([*rows, audit]) + "\n")
    assert v03t.t1_t2(profile, dup_path, data)["budgets"]["64"]["n"] == 4
    recorded = {**profile, "audit": {**profile["audit"], "budgets": {**profile["audit"]["budgets"],
                                                                     "64": {**profile["audit"]["budgets"]["64"], "dp_median": 0.002, "dp_max": 0.01}}}}
    assert v03t.t1_t2(recorded, raw_path, data)["budgets"]["64"]["dp_median"] == 0.002
    assert e["lat_err_median"] == pytest.approx(0.02 / 1.02) and e["t1"] is None and e["note"] == (
        "audit n/a: n_cut 4, the gate needs 24 cut questions (4 audited)")
    assert g["gated"] == 0 and g["t1_pass"] is None and g["t2_pass"] is None and "256" not in g["budgets"]
    monkeypatch.setattr(v03t, "T1_MIN_CUTS", 3)
    g = v03t.t1_t2(profile, raw_path, data)
    e = g["budgets"]["64"]
    assert e["t1"] is True and e["t2"] is True and g["gated"] == 1 and g["t1_pass"] is True and g["t2_pass"] is True
    assert e["p50_model_ms"] == pytest.approx(e["p50_real_ms"] * 1.02 / 1.0, rel=0.05)
    bad_path, raw2, _ = _raw_file(tmp_path, items, audit_err=0.3)       # a 30 % latency error fails T2, not T1
    g2 = v03t.t1_t2(profile, bad_path, data)
    assert g2["budgets"]["64"]["t1"] is True and g2["budgets"]["64"]["t2"] is False and g2["t2_pass"] is False
    # the dense model is descriptive only: the max |Δp| against 1e-6 and the argmax agreement, never a gate
    dense = v03t.t1_t2(profile, raw_path, data, dense=True)
    d = dense["budgets"]["64"]
    assert d["note"] == ("dense, descriptive (no gate): |Δp| max 0.00e+00 against 1e-06 (within), argmax 4/4 on 4/4 "
                         "identical traces (n_cut 4)")
    assert d["t1"] is None and d["t2"] is None and d["descriptive"] is True and dense["gated"] == 0 and dense["t1_pass"] is None
    off = {**profile, "audit": {**profile["audit"], "budgets": {"64": {**profile["audit"]["budgets"]["64"], "dp_max": 0.05}}}}
    assert "|Δp| max 5.00e-02 against 1e-06 (above)" in v03t.t1_t2(off, raw_path, data, dense=True)["budgets"]["64"]["note"]
    # an incomplete audit (n_cut gated, fewer than min(n_cut, 48) audited) is that pair's ERROR, never n/a
    first = next(ln for ln in rows if '"phase": "audit"' in ln)
    short = tmp_path / "short.raw.jsonl"
    short.write_text("\n".join(ln for ln in rows if ln != first) + "\n")
    s = v03t.t1_t2(profile, short, data)
    assert s["budgets"]["64"]["error"] == ("incomplete audit at 64: 3 audited of the 4 the run regenerates (min of n_cut 4 "
                                           "and 48)")
    assert s["budgets"]["64"]["t1"] is None and s["t1_pass"] is None and s["t2_pass"] is None and s["gated"] == 0
    # the thresholds are the bench's: a run with --audit below 48 (0 or 6) is the model's ERROR, never gated on its own
    # audit count, and a required pair it should cover reads not gated with that reason
    for audit in (0, 6):
        low = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 64, 256), audit=audit)
        low_path, low_raw, _ = _raw_file(tmp_path, items, settings=low)
        lp = build_profile(items, low_raw, low, sampling=SAMPLING, stats=STATS, name="m")
        lg = v03t.t1_t2(lp, low_path, data)
        assert lg["error"] == f"the bench requires --audit 48, the profile has {audit}" and lg["t1_pass"] is None
        assert lg["budgets"] == {} and lg["gated"] == 0
        t1, _ = v03t.t1_t2_gates({"qwen9b": lg}, required=(("qwen9b", 64, False),))
        assert t1["error"] == f"required pair qwen9b@64 not gated: the bench requires --audit 48, the profile has {audit}"
        assert "error" not in v03t.t1_t2(lp, low_path, data, dense=True)            # the dense model is descriptive only
    # Bmax is the run's (comparable.budgets), not the profile's budget keys
    trimmed = {**profile, "budgets": {k: v for k, v in profile["budgets"].items() if k != "256"}}
    assert list(v03t.t1_t2(trimmed, raw_path, data)["budgets"]) == ["64"]
    # a partial last line read_raw dropped is a note, not a failure
    partial = tmp_path / "partial.raw.jsonl"
    partial.write_text(raw_path.read_text() + '{"phase": "drift", "q": "rec-0')
    pg = v03t.t1_t2(profile, partial, data)
    assert pg["notes"] == ["partial last line dropped"] and pg["t1_pass"] is True and "error" not in pg


def test_t1_t2_count_the_cuts_from_the_audit_pool_skip_null_traces_and_error_on_bad_rows_or_an_n_cut_mismatch(tmp_path):
    """`n_cut` is the profile's own (`audit_pool`: measured, non-warm-up questions cut at B), never "forced with k ==
    B" (which counts the warm-up's cuts too); a `trace: null` row is skipped; an incomplete row or an n_cut the profile
    disagrees with is that model's ERROR, naming what is wrong."""
    data = write_data(tmp_path / "d.jsonl", 8)
    items, _ = load_data(data)
    settings = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 64, 256), audit=48, warmup=4)
    assert [it.item_id for it in shuffled(items, 0)][:4] == ["rec-7", "rec-4", "rec-6", "rec-1"]   # rec-1 (cut) warms
    raw_path, raw, _ = _raw_file(tmp_path, items, settings=settings)
    profile = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    assert profile["audit"]["budgets"]["64"]["n_cut"] == 3
    g = v03t.t1_t2(profile, raw_path, data)
    assert "error" not in g and g["budgets"]["64"]["n_cut"] == 3              # 4 are forced at k 64; one is the warm-up's
    assert v03t.t1_t2({**profile, "comparable": {}}, raw_path, data)["budgets"]["64"]["n_cut"] == 3   # the raw header's
    off = {**profile, "audit": {**profile["audit"], "budgets": {"64": {**profile["audit"]["budgets"]["64"], "n_cut": 5}}}}
    bad = v03t.t1_t2(off, raw_path, data)
    assert bad["error"] == "budget 64: the bench counts n_cut 3, the profile records 5" and bad["t1_pass"] is None
    # a single-option question's row (`trace: null`) is no cut and no comparison, never a TypeError
    rows = raw_path.read_text().splitlines()
    null = tmp_path / "null.raw.jsonl"
    probs = {"probs": [0.2, 0.7, 0.1], "read_ms": 20.0}
    null.write_text("\n".join([*rows, json.dumps({"phase": "B", "q": "rec-0/q", "trace": None, "budgets": {"64": probs, "256": probs}})])
                    + "\n")
    nprof = build_profile(items, read_raw(null), settings, sampling=SAMPLING, stats=STATS, name="m")
    e = v03t.t1_t2(nprof, null, data)["budgets"]["64"]
    assert (e["n_cut"], e["n"]) == (nprof["audit"]["budgets"]["64"]["n_cut"], nprof["audit"]["budgets"]["64"]["n"]) == (2, 3)
    # an incomplete row (an audit row without its latency) is an ERROR with check_rows' message
    cut = tmp_path / "cut.raw.jsonl"
    cut.write_text("\n".join([*rows, json.dumps({"phase": "audit", "q": "rec-0/q", "budget": 64, "probs": [1.0, 0, 0],
                                                  "trace": {"tokens": []}})]) + "\n")
    err = v03t.t1_t2(profile, cut, data)
    assert "is incomplete: an audit row needs budget, probs, lat_ms" in err["error"] and err["budgets"] == {}
    assert err["t1_pass"] is None and err["t2_pass"] is None


def _entry(t1, t2=None, *, n_cut=40, **kw) -> dict:
    return {"n_cut": n_cut, "n": n_cut, "identical": n_cut, "argmax_agree": n_cut, "dp_median": 0.0, "dp_max": 0.0,
            "lat_err_median": 0.01, "p50_real_ms": 1.0, "p50_model_ms": 1.0, "t1": t1, "t2": t1 if t2 is None else t2, **kw}


NA = {"t1": None, "t2": None, "note": "audit n/a: n_cut 10, the gate needs 24 cut questions (10 audited)"}


def _res(budgets: dict, error: str | None = None) -> dict:
    gated = [e for e in budgets.values() if e["t1"] is not None]
    out = {"budgets": budgets, "gated": len(gated), "t1_pass": v03t._all3([e["t1"] for e in gated]),
           "t2_pass": v03t._all3([e["t2"] for e in gated]), "notes": []}
    return {**out, "error": error} if error else out


def test_t1_t2_gates_judge_the_required_pairs_and_carry_every_error():
    assert v03t.T1_REQUIRED == (("qwen9b", 256, False), ("qwen9b", 512, False), ("tiel", 256, True))
    qwen = _res({"256": _entry(True), "512": _entry(True)})
    tiel = _res({"256": _entry(True), "512": _entry(True)})
    t1, t2 = v03t.t1_t2_gates({"qwen9b": qwen, "tiel": tiel})
    assert t1["pass"] is True and t2["pass"] is True and "error" not in t1 and "notes" not in t1
    assert t1["required"] == {"qwen9b@256": "PASS", "qwen9b@512": "PASS", "tiel@256": "PASS"}
    assert t1["models"]["qwen9b"]["pass"] is True and t1["models"]["qwen9b"]["budgets"] is qwen["budgets"]
    # tiel short of cuts at 256: not testable, the one-hybrid note; the gate rests on qwen9b
    short = _res({"256": {**_entry(None, n_cut=10), **NA}, "512": {**_entry(None, n_cut=8), **NA}})
    t1, _ = v03t.t1_t2_gates({"qwen9b": qwen, "tiel": short})
    assert t1["pass"] is True and t1["required"]["tiel@256"] == "not testable (n_cut 10 < 24)"
    assert t1["notes"] == ["tiel@256 not testable (n_cut 10 < 24): the cut model was verified on one hybrid only"]
    # a required qwen9b pair n/a, missing, or errored is the gate's ERROR
    na = _res({"256": _entry(True), "512": {**_entry(None, n_cut=10), **NA}})
    t1, t2 = v03t.t1_t2_gates({"qwen9b": na, "tiel": tiel})
    assert t1["pass"] is None and t1["error"] == ("required pair qwen9b@512 not gated: audit n/a: n_cut 10, the gate needs "
                                                  "24 cut questions (10 audited)") and t2["error"] == t1["error"]
    t1, _ = v03t.t1_t2_gates({"tiel": tiel})
    assert t1["error"] == ("required pair qwen9b@256 not gated: no result for the model; required pair qwen9b@512 not "
                           "gated: no result for the model") and t1["pass"] is None
    inc = _res({"256": _entry(True), "512": {**_entry(None), "error": "incomplete audit at 512: 20 audited of the 40"}})
    t1, _ = v03t.t1_t2_gates({"qwen9b": inc, "tiel": tiel})
    assert t1["error"] == "required pair qwen9b@512 not gated: incomplete audit at 512: 20 audited of the 40"
    unread = _res({}, error="x.raw.jsonl: row 3 (B a/q) is incomplete")
    t1, _ = v03t.t1_t2_gates({"qwen9b": unread, "tiel": tiel})
    assert t1["models"]["qwen9b"]["error"] == "x.raw.jsonl: row 3 (B a/q) is incomplete" and t1["pass"] is None
    assert t1["required"]["qwen9b@256"] == "not gated: x.raw.jsonl: row 3 (B a/q) is incomplete"
    # tiel gated (24+ cuts) but its audit incomplete: an ERROR, not "not testable"
    tinc = _res({"256": {**_entry(None), "error": "incomplete audit at 256: 3 audited of the 40"}})
    t1, _ = v03t.t1_t2_gates({"qwen9b": qwen, "tiel": tinc})
    assert t1["error"] == "required pair tiel@256 not gated: incomplete audit at 256: 3 audited of the 40" and "notes" not in t1
    # a model-level error on an otherwise passing model (an n_cut mismatch) keeps the gate from passing
    mism = {**_res({"256": _entry(True), "512": _entry(True)}), "error": "budget 512: the bench counts n_cut 40, the profile records 41"}
    assert v03t.t1_t2_gates({"qwen9b": mism, "tiel": tiel})[0]["pass"] is None
    # any other gated pair must pass too; T2 fails on its own
    t1, t2 = v03t.t1_t2_gates({"qwen9b": qwen, "tiel": _res({"256": _entry(True), "512": _entry(False)})})
    assert t1["pass"] is False and t2["pass"] is False
    t1, t2 = v03t.t1_t2_gates({"qwen9b": _res({"256": _entry(True, False), "512": _entry(True)}), "tiel": tiel})
    assert t1["pass"] is True and t2["pass"] is False and t2["required"]["qwen9b@256"] == "FAIL"
    # the dense model never counts: a failing or errored descriptive entry leaves the verdict alone
    minicpm = _res({"256": {**_entry(None), "descriptive": True, "note": "dense, descriptive (no gate): ..."}}, error="boom")
    t1, t2 = v03t.t1_t2_gates({"qwen9b": qwen, "tiel": tiel, "minicpm": minicpm}, dense={"minicpm"})
    assert t1["pass"] is True and "minicpm" not in t1["models"] and t1["descriptive"]["minicpm"]["error"] == "boom"
    text = render({"gates": {"t1": t1, "t2": t2}})
    assert "minicpm@256: dense, descriptive (no gate): ... | PASS |" in text and "| T2 | " in text
    assert "minicpm" not in text.split("| T2 | ")[1]
    t1, _ = v03t.t1_t2_gates({"qwen9b": na, "tiel": short})
    text = render({"gates": {"t1": t1}})
    row = text.split("| T1 | ")[1].split("\n")[0]
    assert ("| required pair qwen9b@512 not gated: audit n/a: n_cut 10, the gate needs 24 cut questions (10 audited); "
            "required: qwen9b@256 PASS, qwen9b@512 not gated, tiel@256 not testable (n_cut 10 < 24); qwen9b@256: identical") in row
    assert "qwen9b@512: audit n/a" not in row and "tiel@256: audit n/a" in row          # said once, in the gate's error
    t1, _ = v03t.t1_t2_gates({"qwen9b": _res({}, error="x.raw.jsonl: row 3 (B a/q) is incomplete"), "tiel": tiel})
    row = render({"gates": {"t1": t1}}).split("| T1 | ")[1].split("\n")[0]
    assert row.count("row 3 (B a/q) is incomplete") == 2 and "qwen9b: x.raw.jsonl" not in row   # once per required pair
    assert "the cut model was verified on one hybrid only | ERROR |" in text


def test_t3_t6_t7_t8_and_descriptive_read_the_profiles_and_run_files():
    prof = {"model_name": "g", "thinking": {"capable": False, "reason": "unsupported_think_format"},
            "budgets": {"0": {"n": 10, "acc": 0.5, "nll": 0.7, "nll_T": 0.69, "T": 1.1}}, "pick": {"label": "g@0"},
            "calibration_overhead_s": 0.0, "drift": {"ratio": 1.0, "flagged": False, "therm": [{"at": "A", "limit": 100}]},
            "audit": {"budgets": {}}, "comparable": {"sampling": {"policy": "model"}},
            "environment": {"run": {"decode_tok_s_median": 80.0}}}
    assert v03t.t3(prof) == {"budgets": {}, "pass": None}
    two = {**prof, "budgets": {**prof["budgets"], "256": {"n": 10, "acc": 0.7, "nll": 0.60, "nll_T": 0.601, "T": 1.3,
                                                          "dacc": [0.2, 0.0, 0.4], "forced": 0.5, "think_mean": 200.0,
                                                          "readout_reprefill": 0, "text_fallback": 0}}}
    assert v03t.t3(two)["budgets"]["256"]["pass"] is True and v03t.t3(two)["pass"] is True
    worse = {**two, "budgets": {**two["budgets"], "256": {**two["budgets"]["256"], "nll_T": 0.61}}}
    assert v03t.t3(worse)["pass"] is False
    report = {"notes": ["g: budget 0 only: thinks in a format callo cannot read", "g: no thermal data"],
              "points": [{"label": "g@0", "model": "g"}]}
    assert v03t.t6(prof, report, status="done", rc=0)["pass"] is True and v03t.t6(prof, None, status="done")["pass"] is False
    assert v03t.t6({**prof, "thinking": {"capable": False, "reason": "no_think_block"}}, report, status="done")["pass"] is False
    assert v03t.t6(prof, report, status="failed", rc=0)["pass"] is False and v03t.t6(prof, report, status=None)["pass"] is False
    broken = {"rc": 1, "models": {"tiel": {"status": "done", "started": "t1"},
                                  "qwen9b": {"status": "failed", "reason": "model file missing", "started": None},
                                  "gemma-e4b": {"status": "done", "started": "t2"}}}
    resumed = {"rc": 0, "models": {"tiel": {"status": "done", "started": "t1"}, "qwen9b": {"status": "done", "started": "t9"},
                                   "gemma-e4b": {"status": "done", "started": "t2"}}}
    assert v03t.t7(broken, resumed)["pass"] is True and v03t.t7(broken, resumed)["only_the_failed_model_reran"] is True
    reran = {**resumed, "models": {**resumed["models"], "tiel": {"status": "done", "started": "t8"}}}
    assert v03t.t7(broken, reran)["pass"] is False
    assert v03t.t7(broken, reran)["problems"] == ["the resume reran a model other than qwen9b"]
    nameless = {**broken, "models": {**broken["models"], "qwen9b": {"status": "failed", "reason": None, "started": None}}}
    assert v03t.t7(nameless, resumed)["problems"] == ["the broken run.json has qwen9b failed with reason None, not failed "
                                                      "with a reason"]
    lacking = {"rc": 0, "models": {"tiel": resumed["models"]["tiel"], "qwen9b": resumed["models"]["qwen9b"]}}
    assert v03t.t7(broken, lacking)["problems"] == ["the resumed run.json has no gemma-e4b", "the resume reran a model "
                                                                                             "other than qwen9b"]
    unfinished = {**resumed, "models": {**resumed["models"], "qwen9b": {"status": "failed", "started": "t9"}}}
    assert v03t.t7(broken, unfinished)["problems"] == ["the resumed run's qwen9b is failed, not done"]
    alone = {"rc": 1, "models": {"qwen9b": broken["models"]["qwen9b"]}}                # nothing else ran: no vacuous pass
    assert v03t.t7(alone, {"rc": 0, "models": {"qwen9b": resumed["models"]["qwen9b"]}})["problems"] == [
        "the broken run ran no model besides qwen9b"]
    assert v03t.t7({}, {})["pass"] is False
    refused = "callo: b was measured on different data (d.jsonl, sha256 ab)\n"
    assert v03t.t8(0, 2, refused)["pass"] is True and v03t.t8(0, 0, refused)["pass"] is False
    d = v03t.descriptive({"g": two})
    assert d["g"]["dacc"] == {"256": [0.2, 0.0, 0.4]} and d["g"]["forced"] == {"256": 0.5} and d["g"]["pick"] == "g@0"
    assert d["g"]["sampler"] == "model" and d["g"]["decode_tok_s"] == 80.0 and d["g"]["n_cut"] == {} and d["g"]["dacc_by_group"] == {}


def _rows_profile(correct: dict[str, dict[int, bool]], *, seed: int = 0) -> dict:
    """A profile with rows only: `correct[uid][budget]`; the cluster is the uid's item."""
    return {"comparable": {"seed": seed}, "budgets": {str(b): {} for b in sorted({b for v in correct.values() for b in v})},
            "row_fields": ["correct"], "rows": [{"q": u, "cluster": u.rsplit("/", 1)[0], "b": {str(b): [c] for b, c in v.items()}}
                                                for u, v in correct.items()]}


def test_descriptive_groups_and_pairs_are_paired_bootstraps_over_the_shared_questions(tmp_path):
    src = tmp_path / "test.jsonl"
    src.write_text("\n".join([_record(0, "composition_holdout"), _record(1, "composition_holdout"), _record(2, "mmlu"),
                              _record(3, "contrastive", qtype="score")]) + "\n")
    tiel = _rows_profile({"composition_holdout/0/q": {0: False, 256: True}, "composition_holdout/1/q": {0: False, 256: True},
                          "mmlu/2/q": {0: True, 256: True}, "contrastive/3/q": {0: True, 256: False}})
    by = v03t.descriptive({"tiel": tiel}, data_path=src)["tiel"]["dacc_by_group"]
    assert set(by) == {"composition", "control", "date-score"} and by["composition"]["n"] == 2 and by["control"]["n"] == 1
    assert by["composition"]["256"][0] == pytest.approx(1.0) and by["control"]["256"][0] == pytest.approx(0.0)
    assert by["date-score"]["256"][0] == pytest.approx(-1.0) and len(by["composition"]["256"]) == 3
    fixed = _rows_profile({"composition_holdout/0/q": {0: False, 256: False}, "composition_holdout/1/q": {0: False, 256: True},
                           "mmlu/2/q": {0: True, 256: True}, "contrastive/3/q": {0: True, 256: True}})
    pairs = v03t.descriptive_pairs({"tiel": tiel, "tiel-fixed": fixed}, [("tiel", "tiel-fixed"), ("tiel", "missing")])
    assert list(pairs) == ["tiel vs tiel-fixed"] and pairs["tiel vs tiel-fixed"]["n"] == 4
    assert pairs["tiel vs tiel-fixed"]["0"][0] == pytest.approx(0.0) and pairs["tiel vs tiel-fixed"]["256"][0] == pytest.approx(0.0)
    assert pairs["tiel vs tiel-fixed"]["256"][1] <= 0 <= pairs["tiel vs tiel-fixed"]["256"][2]
    # one fixed row set per group: a question without every budget's readout is out of all of them, n stated once
    three = _rows_profile({"composition_holdout/0/q": {0: False, 512: False},
                           "composition_holdout/1/q": {0: False, 256: True, 512: True}})
    by3 = v03t.dacc_by_group(three, v03t.groups_of(src))["composition"]
    assert by3["n"] == 1 and set(by3) == {"n", "256", "512"} and by3["512"][0] == pytest.approx(1.0)
    better = _rows_profile({u: {0: True, 256: True} for u in ("composition_holdout/0/q", "composition_holdout/1/q", "mmlu/2/q",
                                                              "contrastive/3/q")})
    signed = v03t.descriptive_pairs({"tiel": better, "tiel-fixed": fixed}, [("tiel", "tiel-fixed")])["tiel vs tiel-fixed"]
    assert signed["256"][0] == pytest.approx(0.25) and signed["0"][0] == pytest.approx(0.5)   # tiel − tiel-fixed


def test_t4_t5_t9_t10_judge_the_served_sessions():
    cases = {"think_budget_x": 422, "think_budget_40000": 422, "over_think_max": 422, "image_with_budget": 422,
             "gemma_budget": 422, "past_n_ctx": 413, "probe_non_loopback": 403, "ok": 200}
    full = {"disconnect_clear": True, "self_test": True, "timings_ok": True}
    g = v03t.t4(cases, **full)
    assert g["pass"] is True and g["not_testable"] == [] and "error" not in g
    assert g["cases"]["ok"] == {"expected": 200, "got": 200, "pass": True}
    assert v03t.t4({**cases, "past_n_ctx": 200}, **full)["pass"] is False
    assert v03t.t4(cases, **{**full, "disconnect_clear": False})["pass"] is False
    assert v03t.t4(cases, **{**full, "timings_ok": False})["pass"] is False               # think fields at B = 0
    # T4 never passes on part of its contract: a None case, an absent case, an unmeasured check
    part = v03t.t4({**cases, "probe_non_loopback": None}, **full)
    assert part["pass"] is None and part["error"] == "incomplete: not testable: probe_non_loopback"
    absent = v03t.t4({k: v for k, v in cases.items() if k != "gemma_budget"}, disconnect_clear=None, self_test=True,
                     timings_ok=None)
    assert absent["pass"] is None and absent["not_testable"] == ["gemma_budget", "timings_ok", "disconnect_clear"]
    both = v03t.t4({**cases, "probe_non_loopback": None, "past_n_ctx": 200}, **full)
    assert both["pass"] is False and both["error"] == "incomplete: not testable: probe_non_loopback"      # FAIL stands
    assert v03t.t5({"a": 2, "b": 2, "c": 0}, {"a": ["foreign_tokens"], "b": ["foreign_processing", "foreign_tokens"]})["pass"] is True
    assert v03t.t5({"a": 2, "b": 2, "c": 2}, {"a": ["foreign_tokens"], "b": ["foreign_processing"]})["pass"] is False
    assert v03t.t5({"a": 2, "b": 2, "c": 0}, {"a": ["foreign_processing"], "b": ["foreign_processing"]})["pass"] is False
    nothing = v03t.t9({}, {})                         # the default required sets: every expected model is missing
    assert nothing["pass"] is None and set(nothing["calibrate"]) == {"tiel", "qwen9b", "minicpm"}
    assert nothing["served"]["qwen9b"] == {"error": "no serve-path session (expected for T9's served half)", "pass": None}


def _t9_raw(path: Path, n_gen_prompt_n: int, *, fallback: bool = False) -> Path:
    """A raw file whose phase-B rows read a 300-token trace at the cuts 64 and 256 (large first passes: the re-prefill
    from the frame a hybrid model does by design) and at n_gen (`n_gen_prompt_n`), plus a 40-token trace read at n_gen
    through the shared readout of every budget."""
    rows = [{"callo_think_raw": RAW_FORMAT}]
    rd = {"probs": [0.9, 0.1], "read_ms": 20.0}                     # what check_rows needs of every readout

    def trace(n_gen: int) -> dict:
        return {"n_gen": n_gen, "gen_wall_ms": 100.0, "tokens": [7] * n_gen}

    for i in range(6):
        cut = {"64": {**rd, "k": 64, "read_prompt_n": 67, "text_fallback": False},
               "256": {**rd, "k": 256, "read_prompt_n": 259, "text_fallback": False},
               "1024": {**rd, "k": 300, "read_prompt_n": n_gen_prompt_n, "text_fallback": fallback}}
        shared = {b: {**rd, "k": 40, "read_prompt_n": 2, "text_fallback": False} for b in ("64", "256", "1024")}
        rows.append({"phase": "B", "q": f"rec-{i}/q", "trace": trace(300 if i % 2 else 40), "budgets": cut if i % 2 else shared})
    rows.append({"phase": "B", "q": "rec-1/q", "trace": trace(300),                          # a resumed row: the latest wins
                 "budgets": {"64": {**rd, "k": 64, "read_prompt_n": 67}, "256": {**rd, "k": 256, "read_prompt_n": 259},
                             "1024": {**rd, "k": 300, "read_prompt_n": n_gen_prompt_n}}})
    # a single-option question: no generation (`trace: null`), never a readout at n_gen, never a TypeError
    rows.append({"phase": "B", "q": "single/q", "trace": None, "budgets": {b: {**rd, "probs": [1.0], "read_prompt_n": 2}
                                                                          for b in ("64", "256", "1024")}})
    rows.append({"phase": "audit", "q": "rec-1/q", "budget": 64, "trace": {"n_gen": 64, "tokens": [7] * 64}, "probs": [1, 0],
                 "k": 64, "lat_ms": 1.0})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _t9(raws: dict, served: dict) -> dict:
    """t9 judged on exactly the given models (no required set): the readout rules alone."""
    return v03t.t9(raws, served, expected_raw=(), expected_served=())


def test_t9_reads_the_readout_at_n_gen_from_the_raw_rows_and_the_natural_closes_from_the_served_timings(tmp_path):
    clean = _t9_raw(tmp_path / "clean.raw.jsonl", 2)
    served = {"tiel": {"think": {"natural": 30, "forced": 18, "eos": 0, "readout_reprefill": 18, "text_fallback": 0},
                       "readouts": [*({"q": f"q{i}", "think_close": "natural", "read_prompt_n": 2} for i in range(30)),
                                    *({"q": f"f{i}", "think_close": "forced", "read_prompt_n": 259} for i in range(18))]}}
    g = _t9({"tiel": clean}, served)
    assert g["pass"] is True and g["calibrate"]["tiel"]["n"] == 6 and g["calibrate"]["tiel"]["at_n_gen_ok"] == 6
    assert g["calibrate"]["tiel"]["text_fallback"] == 0 and g["served"]["tiel"]["n"] == 30 and g["served"]["tiel"]["at_n_gen_ok"] == 30
    assert "natural closes" in g["served"]["tiel"]["source"]                 # the cuts' large first passes never count
    reprefill = _t9_raw(tmp_path / "bad.raw.jsonl", 40)                       # the readout AT n_gen re-prefilled: a real failure
    assert _t9({"tiel": reprefill}, {})["pass"] is False and _t9({"tiel": reprefill}, {})["calibrate"]["tiel"]["at_n_gen_ok"] == 3
    assert _t9({"tiel": _t9_raw(tmp_path / "fb.raw.jsonl", 2, fallback=True)}, {})["pass"] is False
    counter_only = {"tiel": {"think": {"natural": 30, "forced": 18, "eos": 0, "readout_reprefill": 18, "text_fallback": 0}, "readouts": []}}
    g2 = _t9({}, counter_only)                        # the counter is never the verdict: no natural close, ERROR
    assert g2["served"]["tiel"]["pass"] is None and g2["served"]["tiel"]["error"].startswith("not testable: ")
    assert _t9({}, {"tiel": {"think": {"natural": 30, "forced": 0, "eos": 0, "readout_reprefill": 0, "text_fallback": 0}}})["pass"] is None
    # an incomplete row (valid JSON, cut short) is that model's ERROR, with the message: never a traceback or a skip
    rows = clean.read_text().splitlines()
    broken = tmp_path / "broken.raw.jsonl"
    broken.write_text("\n".join([rows[0], json.dumps({"phase": "B", "q": "rec-9/q", "trace": None, "budgets": {"64": {"k": 0}}}),
                                 *rows[1:]]) + "\n")
    g3 = _t9({"tiel": broken, "qwen9b": clean}, {})
    assert "row 1 (B rec-9/q) is incomplete" in g3["calibrate"]["tiel"]["error"] and g3["calibrate"]["tiel"]["pass"] is None
    assert g3["calibrate"]["qwen9b"]["pass"] is True and g3["pass"] is None             # an error, never a PASS
    assert _t9({"tiel": broken, "qwen9b": reprefill}, {})["pass"] is False           # a definite FAIL stands
    partial = tmp_path / "partial.raw.jsonl"                                              # a note, not a failure
    partial.write_text(clean.read_text() + '{"phase": "B", "q": "rec-7')
    gp = _t9({"tiel": partial}, {})
    assert gp["pass"] is True and gp["calibrate"]["tiel"]["note"] == "partial last line dropped"
    assert "tiel 6/6 readouts at n_gen with prompt_n <= 4, 0 fallbacks (partial last line dropped)" in render({"gates": {"t9": gp}})
    profile = {"row_fields": ["correct", "nll1", "pmax", "lat_ms", "k", "close"],
               "rows": [{"q": f"q{i}", "b": {"256": [True, 0.5, 0.6, 1000.0 + 10 * i, 256, "forced"]}} for i in range(48)]}
    served_ms = {f"q{i}": 1100.0 + 10 * i for i in range(48)}
    g = v03t.t10(served_ms, profile, 256)
    assert g["n"] == 48 and g["served_p50_ms"] == 1335.0 and g["profile_p50_ms"] == 1235.0 and g["pass"] is True
    assert g["rel_err"] == pytest.approx(100 / 1235)
    assert v03t.t10({f"q{i}": 1500.0 for i in range(48)}, profile, 256)["pass"] is False
    # fewer than the gate's 48 shared questions is not testable (an ERROR), never a PASS on a few
    few = v03t.t10({f"q{i}": 1100.0 + 10 * i for i in range(47)}, profile, 256)
    assert few == {"n": 47, "budget": 256, "pass": None, "error": "not testable: 47 shared questions < 48"}
    assert v03t.t10({"other": 1.0}, profile, 256)["error"] == "not testable: 0 shared questions < 48"
    assert "| T10 | " in render({"gates": {"t10": {"tiel": few}}}) and (
        "tiel@256: not testable: 47 shared questions < 48 | ERROR |" in render({"gates": {"t10": {"tiel": few}}}))


def test_the_report_renders_every_gate_from_the_summary(tmp_path):
    s = {"started": "2026-10-20T10:00:00Z", "finished": "2026-10-20T13:20:00Z", "llama_server": "/opt/llama-server",
         "out": "/r/think", "data": {"path": "/d/test.jsonl", "questions": 336, "sources": ["mmlu", "qnli"], "run2_questions": 120},
         "runs": {"run1": {"rc": 0, "dir": "/r/think/run1", "run_json": {"models": {"tiel": {"status": "done"}, "qwen9b": {"status": "failed", "reason": "x"}}}}},
         "gates": {"t1": {"models": {"qwen9b": {"budgets": {"256": {"n_cut": 40, "n": 40, "identical": 40, "argmax_agree": 40,
                                                                    "dp_median": 1e-5, "dp_max": 2e-4, "lat_err_median": 0.03,
                                                                    "p50_real_ms": 2000.0, "p50_model_ms": 2040.0, "t1": True,
                                                                    "t2": True}}, "pass": True},
                                     "tiel": {"budgets": {"256": {"n_cut": 10, "n": 10, "identical": 10, "argmax_agree": 10,
                                                                  "dp_median": 0.0, "dp_max": 0.0, "lat_err_median": 0.01,
                                                                  "p50_real_ms": 1.0, "p50_model_ms": 1.0, "t1": None, "t2": None,
                                                                  "note": "audit n/a: n_cut 10, the gate needs 24 cut questions (10 audited)"}},
                                              "pass": None}},
                          "required": {"qwen9b@256": "PASS", "tiel@256": "not testable (n_cut 10 < 24)"},
                          "notes": ["tiel@256 not testable (n_cut 10 < 24): the cut model was verified on one hybrid only"],
                          "pass": True},
                   "t2": {"models": {"qwen9b": {"budgets": {"256": {"n": 40, "lat_err_median": 0.03, "p50_real_ms": 2000.0,
                                                                    "p50_model_ms": 2040.0, "t1": True, "t2": True}}, "pass": True}},
                          "required": {"qwen9b@256": "PASS"}, "pass": True},
                   "t3": {"tiel": {"budgets": {"256": {"nll_1": 0.7, "nll_T": 0.65, "T": 1.2, "pass": True}}, "pass": True}},
                   "t4": {"cases": {"ok": {"expected": 200, "got": 200, "pass": True}, "past_n_ctx": {"expected": 413, "got": 200, "pass": False}},
                          "not_testable": ["probe_non_loopback"], "disconnect_clear": True, "self_test": True, "timings_ok": True,
                          "error": "incomplete: not testable: probe_non_loopback", "pass": False},
                   "t5": {"rcs": {"a": 2, "b": 2, "c": 0}, "failed": {"a": ["foreign_tokens"], "b": ["foreign_processing"]}, "pass": True},
                   "t6": {"capable": False, "reason": "unsupported_think_format", "budget_0_only": True, "rc": 0, "report_note": True,
                          "pass": True},
                   "t7": {"error": "the rerun was skipped: --parts"}, "t8": None,
                   "t9": {"calibrate": {"tiel": {"n": 300, "at_n_gen_ok": 299, "text_fallback": 0, "source": "raw", "pass": True}},
                          "served": {"tiel": {"n": 30, "at_n_gen_ok": 30, "text_fallback": 0, "source": "X-Callo-Timings", "pass": True},
                                     "qwen9b": {"n": 48, "at_n_gen_ok": 30, "text_fallback": 0, "source": "X-Callo-Timings", "pass": False}},
                          "pass": False},
                   "t10": {"tiel": {"n": 48, "budget": 256, "served_p50_ms": 1500.0, "profile_p50_ms": 1400.0, "rel_err": 0.071, "pass": True}}},
         "descriptive": {"tiel": {"acc": {"0": 0.6, "256": 0.7}, "dacc": {"256": [0.1, 0.0, 0.2]}, "forced": {"256": 0.4},
                                  "think_mean": {"256": 180.0}, "n_cut": {"256": 40}, "pick": "tiel@256", "overhead_s": 12.5,
                                  "drift": {"ratio": 1.01, "flagged": False, "therm": [{"at": "A", "limit": 100}, {"at": "end", "limit": None}]},
                                  "sampler": "model", "decode_tok_s": 95.0,
                                  "dacc_by_group": {"composition": {"256": [0.15, 0.05, 0.25], "n": 96}, "control": {"256": [0.0, -0.05, 0.05], "n": 240}}}},
         "descriptive_pairs": {"tiel vs tiel-fixed": {"0": [0.0, -0.02, 0.02], "256": [-0.03, -0.08, 0.02], "n": 376}},
         "joint_md": "# callo compare: the joint think frontier\n\n| point |\n", "errors": {"run3": "skipped"}}
    text = render(s)
    assert text.startswith("# callo v0.3: think budgets acceptance bench\n\nRun 2026-10-20T10:00:00Z to 2026-10-20T13:20:00Z on `/opt/llama-server`.")
    assert "| run1 | 0 | tiel done, qwen9b failed (x) |" in text
    assert "| T1 | cut = generation" in text and ("| required: qwen9b@256 PASS, tiel@256 not testable (n_cut 10 < 24); "
                                                  "qwen9b@256: identical 40/40, argmax 40/40, |Δp| median 1.00e-05 max 2.00e-04; "
                                                  "tiel@256: audit n/a: n_cut 10, the gate needs 24 cut questions (10 audited); "
                                                  "tiel@256 not testable (n_cut 10 < 24): the cut model was verified on one "
                                                  "hybrid only | PASS |") in text
    assert "| T2 | the latency model" in text and ("| required: qwen9b@256 PASS; qwen9b@256: median error 0.030, p50 model 2040 "
                                                   "vs real 2000 ms | PASS |") in text
    assert "| T3 | " in text and "tiel@256: NLL@1 0.7000 -> NLL@T 0.6500 (T 1.20) | PASS |" in text
    assert ("| incomplete: not testable: probe_non_loopback; ok 200, past_n_ctx 200 (want 413); X-Callo-Timings think fields only at B > 0: True; disconnect clears slot: "
            "True; think self-test: True | FAIL |") in text and text.count("not testable: probe_non_loopback") == 1
    assert ("capable False, reason unsupported_think_format, budget 0 only True, its run None (Run 1 exit 0), report note "
            "True, on the joint report's points None | PASS |") in text
    assert "(a) exit 2 [foreign_tokens], (b) exit 2 [foreign_processing], (c) exit 0 | PASS |" in text
    assert "| T7 | " in text and "the rerun was skipped: --parts | ERROR |" in text and "| T8 | " in text and "| not run | NOT RUN |" in text
    assert ("calibrate: tiel 299/300 readouts at n_gen with prompt_n <= 4, 0 fallbacks; served: tiel 30/30, qwen9b 30/48 "
            "| FAIL |") in text
    assert "tiel@256: served p50 1500 vs profile 1400 ms over 48 questions (error 0.071) | PASS |" in text
    assert "## Descriptive results (no gate)" in text and "| tiel | model | 0: 0.600, 256: 0.700 | 256: +0.100 [+0.000, +0.200] | 256: 40% | 256: 180 | 256: 40 | tiel@256 | 12.5 s | ratio 1.01, none; therm A 100, end n/a | 95 tok/s |" in text
    assert "### Δacc(B − 0) per source group" in text and "| tiel | composition | 96 | 256: +0.150 [+0.050, +0.250] |" in text
    assert "### The same model under two samplers (paired, same questions)" in text
    assert "| tiel vs tiel-fixed | 376 | 0: +0.000 [-0.020, +0.020], 256: -0.030 [-0.080, +0.020] |" in text
    assert "## The joint frontier (`callo compare`, Runs 1 and 3)\n\n# callo compare: the joint think frontier" in text
    assert "## Errors\n\n- run3: skipped" in text and "Raw output: `/r/think`" in text
    assert len(GATES) == 10 and [g[0] for g in GATES] == [f"T{i}" for i in range(1, 11)]
    aborted = render({**s, "aborted": "disk full"})
    assert "**ABORTED:** disk full." in aborted
    (tmp_path / "summary.json").write_text(json.dumps(s))
    assert report_main(["--summary", str(tmp_path / "summary.json"), "--out", str(tmp_path / "doc" / "v0.3-think.md")]) == 0
    assert (tmp_path / "doc" / "v0.3-think.md").read_text() == text


def test_a_per_model_error_is_the_gates_error_unless_another_model_fails():
    t1 = {"models": {"tiel": {"budgets": {}, "error": "budget 256: the bench counts n_cut 30, the profile records 31",
                              "pass": None},
                     "qwen9b": {"budgets": {"256": {"n_cut": 40, "n": 40, "identical": 40, "argmax_agree": 40, "dp_median": 1e-5,
                                                    "dp_max": 2e-4, "lat_err_median": 0.03, "p50_real_ms": 2000.0,
                                                    "p50_model_ms": 2040.0, "t1": True, "t2": True}}, "pass": True}},
          "pass": None}
    t9 = {"calibrate": {"tiel": {"error": "x.raw.jsonl: row 3 (B a/q) is incomplete", "pass": None},
                        "qwen9b": {"n": 4, "at_n_gen_ok": 4, "text_fallback": 0, "source": "raw", "pass": True}},
          "served": {}, "pass": None}
    text = render({"gates": {"t1": t1, "t9": t9}})
    assert "| T1 | " in text and "tiel: budget 256: the bench counts n_cut 30, the profile records 31; qwen9b@256: identical" in text
    assert "2.00e-04 | ERROR |" in text
    assert "calibrate: tiel ERROR (x.raw.jsonl: row 3 (B a/q) is incomplete), qwen9b 4/4 readouts" in text
    assert "served: none | ERROR |" in text
    failing = {**t1, "pass": False}
    assert "2.00e-04 | FAIL |" in render({"gates": {"t1": failing}})


def test_t3_t9_t10_hold_the_gate_to_every_expected_model(tmp_path):
    """A model or a half the orchestration lost is an ERROR naming it, never a PASS on what remains."""
    assert v03t.EXPECTED_PROFILES == ("tiel", "qwen9b", "gemma-e4b", "minicpm", "tiel-fixed")
    assert v03t.T9_RAW_MODELS == ("tiel", "qwen9b", "minicpm") and v03t.SERVED_MODELS == ("tiel", "qwen9b")
    good = {"thinking": {"capable": True}, "comparable": {"budgets": [0, 256]},
            "budgets": {"0": {"nll": 0.7}, "256": {"nll": 0.6, "nll_T": 0.59, "T": 1.2}}}
    t3 = v03t.t3_gate({"tiel": good, "qwen9b": good}, expected=("tiel", "qwen9b", "minicpm"))
    assert t3["tiel"]["pass"] is True and t3["minicpm"] == {"error": "no profile (expected for T3)", "pass": None}
    text = render({"gates": {"t3": t3}})
    assert "minicpm: no profile (expected for T3) | ERROR |" in text
    assert render({"gates": {"t3": v03t.t3_gate({"tiel": good}, expected=("tiel",))}}).count("| PASS |") == 1
    assert list(v03t.t3_gate({}, expected=v03t.EXPECTED_PROFILES)) == list(v03t.EXPECTED_PROFILES)
    clean = _t9_raw(tmp_path / "clean.raw.jsonl", 2)
    served = {"think": {"natural": 30, "forced": 0, "eos": 0, "readout_reprefill": 0, "text_fallback": 0},
              "readouts": [{"q": "q", "think_close": "natural", "read_prompt_n": 2}]}
    both = v03t.t9({"tiel": clean, "qwen9b": clean}, {"tiel": served, "qwen9b": served}, expected_raw=("tiel", "qwen9b"))
    assert both["pass"] is True
    no_served = v03t.t9({"tiel": clean, "qwen9b": clean}, {}, expected_raw=("tiel", "qwen9b"))      # a whole half lost
    assert no_served["pass"] is None and no_served["served"]["tiel"]["error"].startswith("no serve-path session")
    lost = v03t.t9({"tiel": clean}, {"tiel": served, "qwen9b": served})                 # qwen9b's and minicpm's raw lost
    assert lost["pass"] is None and lost["calibrate"]["minicpm"]["error"] == "no raw file (expected for T9's calibrate half)"
    assert "minicpm ERROR (no raw file (expected for T9's calibrate half))" in render({"gates": {"t9": lost}})
    assert "| ERROR |" in render({"gates": {"t9": no_served}}).split("| T9 | ")[1].split("\n")[0]
    ok = {"n": 48, "budget": 256, "served_p50_ms": 1.0, "profile_p50_ms": 1.0, "rel_err": 0.0, "pass": True}
    t10 = v03t.t10_gate({"tiel": ok})
    assert t10["tiel"] is ok and t10["qwen9b"] == {"error": "no serve-path session result (expected for T10)", "pass": None}
    row = render({"gates": {"t10": t10}}).split("| T10 | ")[1].split("\n")[0]
    assert row.endswith("qwen9b: no serve-path session result (expected for T10) | ERROR |")
    assert v03t.t10_gate({"tiel": ok, "qwen9b": ok})["qwen9b"]["pass"] is True


@pytest.mark.parametrize("name,key", [(g[0], g[1]) for g in GATES])
def test_every_gate_entry_that_is_only_an_error_renders_its_message_as_error(name, key):
    row = render({"gates": {key: {"error": "x"}}}).split(f"| {name} | ")[1].split("\n")[0]
    assert row.endswith("| x | ERROR |")


def test_t3_never_passes_a_thinking_model_it_did_not_measure():
    """Every budget > 0 of a thinking model's run needs a finite nll and nll_T; budget-0-only is `thinking.capable`
    false, never a model name."""
    good = {"thinking": {"capable": True}, "comparable": {"budgets": [0, 256, 512]},
            "budgets": {"0": {"nll": 0.7}, "256": {"nll": 0.6, "nll_T": 0.59, "T": 1.2},
                        "512": {"nll": 0.55, "nll_T": 0.55, "T": 1.1}}}
    assert v03t.t3(good)["pass"] is True and "error" not in v03t.t3(good)
    assert v03t.t3({})["error"] == "the profile records no run budgets (comparable.budgets)" and v03t.t3({})["pass"] is None
    empty = {**good, "budgets": {}}
    assert v03t.t3(empty)["error"] == "budget 256: no entry in the profile; budget 512: no entry in the profile"
    only0 = {**good, "budgets": {"0": good["budgets"]["0"]}}                   # a capability regression: still an error
    assert v03t.t3(only0)["pass"] is None and "budget 256: no entry" in v03t.t3(only0)["error"]
    missing = {**good, "budgets": {k: v for k, v in good["budgets"].items() if k != "512"}}
    m = v03t.t3(missing)
    assert m["error"] == "budget 512: no entry in the profile" and m["pass"] is None and m["budgets"]["256"]["pass"] is True
    none_t = {**good, "budgets": {**good["budgets"], "512": {"nll": 0.55, "nll_T": None, "T": None}}}
    assert v03t.t3(none_t)["error"] == "budget 512: nll_T None, not a finite number"                 # never a TypeError
    nan = {**good, "budgets": {**good["budgets"], "256": {"nll": float("nan"), "nll_T": float("inf"), "T": 1.0}}}
    assert v03t.t3(nan)["error"] == "budget 256: nll nan and nll_T inf, not a finite number"
    failing = {**none_t, "budgets": {**none_t["budgets"], "256": {"nll": 0.6, "nll_T": 0.61, "T": 1.2}}}
    assert v03t.t3(failing)["pass"] is False                                    # a measured FAIL stands over the error
    gemma = {"model_name": "gemma-e4b", "thinking": {"capable": False, "reason": "unsupported_think_format"},
             "comparable": {"budgets": [0, 256, 512, 1024]}, "budgets": {"0": {"nll": 0.7}}}
    assert v03t.t3(gemma) == {"budgets": {}, "pass": None}                      # budget 0 alone: nothing to judge
    named = {**gemma, "model_name": "tiel", "thinking": {"capable": True}}     # the flag decides, not the name
    assert "error" in v03t.t3(named)
    gate = v03t.t3_gate({"tiel": good, "qwen9b": only0, "gemma-e4b": gemma, "minicpm": None}, expected=("tiel", "qwen9b",
                                                                                                         "gemma-e4b", "minicpm"))
    assert gate["minicpm"] == {"error": "no profile (expected for T3)", "pass": None}
    row = render({"gates": {"t3": gate}}).split("| T3 | ")[1].split("\n")[0]
    assert "qwen9b: budget 256: no entry in the profile; budget 512: no entry in the profile" in row and row.endswith("| ERROR |")


def test_t10_gate_treats_a_hollow_entry_and_t9_a_missing_path_as_the_missing_session_error(tmp_path):
    ok = {"n": 48, "budget": 256, "served_p50_ms": 1.0, "profile_p50_ms": 1.0, "rel_err": 0.0, "pass": True}
    for hollow in (None, {}, {"n": 48}, "x"):
        g = v03t.t10_gate({"tiel": ok, "qwen9b": hollow})
        assert g["qwen9b"] == {"error": "no serve-path session result (expected for T10)", "pass": None}
        assert render({"gates": {"t10": g}}).split("| T10 | ")[1].split("\n")[0].endswith("| ERROR |")
    clean = _t9_raw(tmp_path / "clean.raw.jsonl", 2)
    served = {"think": {"natural": 30, "forced": 0, "eos": 0, "readout_reprefill": 0, "text_fallback": 0}}
    g = v03t.t9({"tiel": clean, "qwen9b": None}, {"tiel": served, "qwen9b": None}, expected_raw=("tiel", "qwen9b"))
    assert g["calibrate"]["qwen9b"] == {"error": "no raw file (expected for T9's calibrate half)", "pass": None}
    assert g["served"]["qwen9b"] == {"error": "no serve-path session (expected for T9's served half)", "pass": None}
    assert g["calibrate"]["tiel"]["pass"] is True and g["pass"] is None


# ---- the orchestration through its seams: the menu runs on the fake llama-server executable, the serve-path ----
# ---- sessions on the thinking fake upstream, every gate judged, the doc rendered ----------------------------------

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

from fake_llama_think import exe_script  # noqa: E402
from fake_upstream import FakeUpstream, planted_answer  # noqa: E402
from helpers import tiny_gguf  # noqa: E402
from test_bench_v02 import AttachedCallo  # noqa: E402

from bench import lib  # noqa: E402
from bench.report_v03_think import _verdict  # noqa: E402
from callo import menu  # noqa: E402
from callo.systemone import parse_request  # noqa: E402

TESTS_DIR = str(Path(__file__).resolve().parent)
CHOICES = [["one", "two", "three"], ["red", "green", "blue", "cyan"], ["yes", "no"], ["alpha", "beta", "gamma"]]
BENIGN_PS = "    1 /sbin/launchd\n  412 /usr/sbin/cfprefsd agent\n"     # an EMPTY listing refuses (fails closed)


@pytest.fixture(autouse=True)
def _no_real_version_probe(monkeypatch):
    """The bench records `llama-server --version`: under pytest the real binary never runs, not even for its version."""
    monkeypatch.setattr(v03t, "_llama_server_version", lambda binary: f"version: 0 (fake) [{Path(binary).name}]")


def _source_file(path: Path, n: int = 8) -> Path:
    """Transfer-set-shaped records (source, variant) whose label is the thinking fake's planted answer."""
    lines = []
    for i in range(n + 2):
        crit = {f"o{j}": t for j, t in enumerate(CHOICES[i % len(CHOICES)])}
        q = parse_request({"state": f"state {i}", "questions": {"q": {"type": "choice", "instructions": f"pick {i}",
                                                                      "criteria": crit}}}).questions[0]
        source, variant = ("mmlu", "clean") if i < n else ("emotion", "clean") if i == n else ("mmlu", "permuted")
        lines.append(json.dumps({"state": f"state {i}", "questions": {"q": {"type": "choice", "instructions": f"pick {i}",
                                                                           "criteria": crit, "label": f"o{planted_answer(list(q.texts))}"}},
                                 "_meta": {"id": f"{source}/{i}", "group_id": f"g{i // 2}", "variant": variant, "source": source}}))
    path.write_text("\n".join(lines) + "\n")
    return path


def _ps_args(pid: int) -> str:
    return subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True, check=False).stdout.strip()


def _reap_own_fakes(pids_file: Path, exe: Path) -> list[int]:
    """SIGKILL only the fakes THIS test launched (their pids from FAKE_PIDS_FILE, their argv naming this test's
    executable) that are still alive; the pids it had to kill."""
    killed = []
    for pid in (int(x) for x in pids_file.read_text().split()) if pids_file.exists() else ():
        if str(exe) in _ps_args(pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
    return killed


def _row(text: str, gate: str) -> str:
    return text.split(f"| {gate} | ")[1].split("\n")[0]


def test_the_whole_bench_runs_through_its_seams_on_the_fakes(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = tmp_path / "llama-server"
    exe.write_text(exe_script(sys.executable, TESTS_DIR))
    exe.chmod(0o755)
    for k in ("FAKE_FAIL", "FAKE_THINK", "FAKE_ARGV_FILE", "FAKE_IGNORE_TERM", "FAKE_LOAD_DELAY"):
        monkeypatch.delenv(k, raising=False)
    pids_file = tmp_path / "fake-pids"
    monkeypatch.setenv("FAKE_PIDS_FILE", str(pids_file))
    ggufs = {n: tiny_gguf(tmp_path / f"{n}.gguf") for n in ("tiel", "qwen9b", "gemma-e4b", "minicpm")}
    src = _source_file(tmp_path / "test.jsonl", 8)
    ups: list[FakeUpstream] = []
    launches: list[tuple[str, list[str]]] = []

    def launch(name, out, *, model, flags, ctx):
        launches.append((name, list(flags)))
        up = FakeUpstream(think="channel" if name == "gemma-e4b" else "preopened", slot_save=True,
                          think_len=lambda text: 40 if "state 1" in text or "state 3" in text else 300).start()
        up.props["model_path"] = model
        up.props["default_generation_settings"]["n_ctx"] = ctx
        for s in up.slots:
            s["n_ctx"] = ctx
        ups.append(up)
        c = AttachedCallo(up, tmp_path / out.name / name, "--no-cache", *flags)    # as the real seam's `Callo` does
        c.parallel, c.up = 4, up
        return c

    def decode(c):
        c.up.set_slot(2, is_processing=True)
        return lambda: c.up.set_slot(2, is_processing=False)

    monkeypatch.setattr(v03t, "MODELS", {"tiel": {"model": str(ggufs["tiel"])}, "qwen9b": {"model": str(ggufs["qwen9b"])},
                                         "gemma-e4b": {"model": str(ggufs["gemma-e4b"]), "args": ["--fake-think", "channel"]}})
    monkeypatch.setattr(v03t, "RUN2_MODELS", {"minicpm": {"model": str(ggufs["minicpm"])}})
    monkeypatch.setattr(v03t, "RUN3_MODELS", {"tiel-fixed": {"model": str(ggufs["tiel"]), "think_sampling": "fixed"}})
    monkeypatch.setattr(v03t, "LLAMA_SERVER", str(exe))
    monkeypatch.setattr(v03t, "SERVE_MMPROJ", {})
    monkeypatch.setattr(v03t, "BUDGETS", "0,32,64")
    monkeypatch.setattr(v03t, "AUDIT", 2)
    monkeypatch.setattr(v03t, "CTX", 4096)
    monkeypatch.setattr(v03t, "RUN2_N", 4)
    monkeypatch.setattr(v03t, "SERVE_N", 4)
    monkeypatch.setattr(v03t, "SETTLE_S", 0.0)
    monkeypatch.setattr(v03t, "DISCONNECT_AFTER_S", 0.2)
    monkeypatch.setattr(v03t, "DISCONNECT_CLEAR_S", 0.3)
    monkeypatch.setattr(v03t, "_launch", launch)
    monkeypatch.setattr(v03t, "_lan_ip", lambda: None)
    monkeypatch.setattr(v03t, "start_decode", decode)
    monkeypatch.setattr(v03t, "DISK_PATHS", (str(tmp_path),))
    monkeypatch.setattr(v03t, "MIN_FREE_GIB", 0.0)
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(menu, "ps_listing", lambda run=None: (BENIGN_PS, None))   # the menu's GPU rule: never the real ps
    out, doc = tmp_path / "out", tmp_path / "doc" / "v0.3-think.md"
    try:
        rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(doc), "--min-free-gb", "0"])
    finally:
        for up in ups:
            up.stop()
        leftover = _reap_own_fakes(pids_file, exe)
    assert leftover == []                                                     # the menu stopped every fake it launched
    s = json.loads((out / "summary.json").read_text())
    assert rc == 0, s.get("errors")
    assert s["errors"] == {} and s["data"]["questions"] == 8 and s["data"]["run2_questions"] == 4
    data = out / "data" / "test-s1.jsonl"
    assert s["data"]["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert s["data"]["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    # the code and the binary that ran are on record
    assert len(s["code"]["commit"]) == 40 and isinstance(s["code"]["dirty"], bool)
    assert s["llama_server"] == str(exe) and s["llama_server_version"] == "version: 0 (fake) [llama-server]"
    assert s["sessions"][-1]["allow_gpu_busy"] is False and s["sessions"][-1]["parts"] == list(v03t.PARTS)
    assert {m: r["status"] for m, r in s["runs"]["run1"]["run_json"]["models"].items()} == {"tiel": "done", "qwen9b": "done",
                                                                                              "gemma-e4b": "done"}
    assert s["runs"]["run1"]["rc"] == 0 and s["runs"]["run2"]["rc"] == 0 and s["runs"]["run3"]["rc"] == 0
    g = s["gates"]
    assert g["t7"]["pass"] is True and g["t7"]["broken"]["qwen9b"] == "failed" and g["t7"]["resumed"]["qwen9b"] == "done"
    assert g["t7"]["dir"] == str(out / "t7")
    assert g["t8"]["pass"] is True and (g["t8"]["join_rc"], g["t8"]["refuse_rc"]) == (0, 2) and s["joint_md"].startswith("# callo compare")
    assert "different data" in g["t8"]["refuse_reason"]                       # compare's own reason
    assert g["t6"]["pass"] is True and g["t6"]["reason"] == "unsupported_think_format" and g["t6"]["rc"] == 0
    # T1/T2 through t1_t2_gates: 2 audited cuts and no budget 256, so the required pairs are not gated: ERROR, never PASS
    assert set(g["t1"]["models"]) == {"tiel", "qwen9b"} and set(g["t1"]["descriptive"]) == {"minicpm"}
    assert g["t1"]["pass"] is None and "required pair qwen9b@256 not gated" in g["t1"]["error"]
    assert g["t1"]["models"]["tiel"]["budgets"]["32"]["note"].startswith("audit n/a") and g["t2"]["pass"] is None
    assert set(g["t3"]) == {"tiel", "qwen9b", "gemma-e4b", "minicpm", "tiel-fixed"}
    assert all("error" in v or "budgets" in v for v in g["t3"].values())
    t4 = g["t4"]["cases"]
    assert t4["think_budget_x"]["got"] == 422 and t4["ok"]["got"] == 200 and t4["gemma_budget"]["got"] == 422
    assert t4["past_n_ctx"]["got"] == 413 and t4["think_budget_40000"]["got"] == 422 and t4["over_think_max"]["got"] == 422
    assert all(v["pass"] for v in t4.values())
    assert g["t4"]["timings_ok"] is True and g["t4"]["self_test"] is True and g["t4"]["disconnect_clear"] is True
    # no mmproj, no LAN address: two cases not testable, so T4 is incomplete (ERROR), never a PASS on part of it
    assert g["t4"]["not_testable"] == ["image_with_budget", "probe_non_loopback"] and g["t4"]["pass"] is None
    assert g["t5"]["pass"] is True, g["t5"]
    assert g["t9"]["pass"] is True and set(g["t9"]["calibrate"]) == {"tiel", "qwen9b", "minicpm"}
    assert g["t9"]["calibrate"]["qwen9b"]["text_fallback"] == 0 and g["t9"]["calibrate"]["tiel"]["n"] == 8
    assert "natural closes" in g["t9"]["served"]["tiel"]["source"] and set(g["t9"]["served"]) == {"tiel", "qwen9b"}
    assert g["t10"]["tiel"]["n"] == 4 and g["t10"]["qwen9b"]["budget"] == s["serve"]["qwen9b"]["budget"] == 64
    assert s["serve"]["qwen9b"]["budget_note"] in (
        "the profile's pick", "the profile picked 0; served at Bmax so T4 and T10 see a deliberate readout")
    # the fake plants its decode timings (predicted_ms) while the served wall time is real, so the two p50s disagree by
    # far more than 15 %: T10 can only be judged on a real model; here the gate's wiring is what is tested
    assert g["t10"]["tiel"]["pass"] is False
    # a same-URL restart of a serve session answers on the URL a client already holds, as a NEW instance
    restart = s["serve"]["qwen9b-small"]["restart"]
    assert restart["pass"] is True and restart["same_url"] is True and restart["served"] == 200 and "error" not in restart
    assert restart["old_instance"] and restart["new_instance"] and restart["new_instance"] != restart["old_instance"]
    assert [n for n, _ in launches] == ["tiel", "qwen9b", "gemma-e4b", "qwen9b-small", "qwen9b-small"]
    assert launches[-1][1][-5:] == ["--port", str(restart["port"]), "--", "--cache-ram", "0"]
    assert restart["url"].endswith(f":{restart['port']}")
    # every serve launch repeats the llama-server args its profile was measured with (the menu's --cache-ram 0) and the
    # profile's projector (none): `--think-profile` finds nothing that differs; the small launch carries no profile
    assert all(f[f.index("--") + 1:][:2] == ["--cache-ram", "0"] for _, f in launches if "--" in f) and len(
        [f for _, f in launches if "--" in f]) == 5 and not any("--mmproj" in f for _, f in launches)
    assert all("--think-profile" not in f for n, f in launches if n == "qwen9b-small")
    # the served half and T10's per-question engine ms are kept, so T9 and T10 can be recomputed offline
    assert len(s["serve"]["tiel"]["served_ms"]) == 4 and len(s["serve"]["tiel"]["served_readouts"]) == 4
    assert isinstance(s["serve"]["tiel"]["stats_think"], dict)
    # provenance per part, start-up warnings and the disks at start
    assert set(s["provenance"]) == {*v03t.PARTS, "gates"} and s["provenance"]["run1"]["commit"] == s["code"]["commit"]
    assert s["provenance"]["serve"]["llama_server_version"] == "version: 0 (fake) [llama-server]"
    assert set(s["disk_free_gib_at_start"]) == {str(out), "/"} and s["sessions"][-1]["warnings"] == s["warnings"]
    assert any(w.startswith("no LAN address: T4's 403 case") for w in s["warnings"])
    assert any(w.startswith("no mmproj configured for qwen9b (the small launch)") for w in s["warnings"])
    assert any("does not name the expected build 4ebdf2c" in w for w in s["warnings"])
    assert s["descriptive"]["tiel"]["dacc_by_group"]["control"]["n"] == 8 and "tiel vs tiel-fixed" in s["descriptive_pairs"]
    assert (out / "data" / "test-s1-t7.jsonl").read_text().count("\n") == 8
    assert not (out / "run.pid").exists()                                     # written for the live run, removed at exit
    assert set(s["descriptive"]) == {"tiel", "qwen9b", "minicpm", "tiel-fixed", "gemma-e4b"}
    assert (out / "run1" / "report.md").exists() and (out / "joint" / "report.json").exists() and (out / "t7" / "run.json").exists()
    text = doc.read_text()
    assert text.startswith("# callo v0.3: think budgets acceptance bench") and "| T6 |" in text
    for gate, verdict in (("T1", "ERROR"), ("T2", "ERROR"), ("T4", "ERROR"), ("T5", "PASS"), ("T6", "PASS"), ("T7", "PASS"),
                          ("T8", "PASS"), ("T9", "PASS"), ("T10", "FAIL")):
        assert _row(text, gate).endswith(f"| {verdict} |"), (gate, _row(text, gate))
    assert "## The joint frontier" in text and "| run1 | 0 | tiel done, qwen9b done, gemma-e4b done |" in text
    dirty = " dirty" if s["code"]["dirty"] else ""
    assert "## Provenance" in text and f"| run1 | {s['code']['commit'][:12]}{dirty} | version: 0 (fake) [llama-server] |" in text
    assert f"Data sha256: test `{s['data']['sha256']}`" in text and "## Warnings (at the last start)" in text
    assert set(s["runs"]) == {"run1", "run2", "run3"} and set(s["serve"]) == {"tiel", "qwen9b", "gemma-e4b", "qwen9b-small"}
    # rerunnable: a second main with only `t8` keeps the other parts; Run 1 resumes (its DIR is not empty)
    rc2 = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(doc), "--parts", "t8", "--min-free-gb", "0"])
    s2 = json.loads((out / "summary.json").read_text())
    assert rc2 == 0 and s2["runs"]["run1"] == s["runs"]["run1"] and s2["gates"]["t7"] == g["t7"] and s2["gates"]["t8"]["pass"] is True
    assert s2["gates"]["t4"] == g["t4"] and s2["gates"]["t5"] == g["t5"] and len(s2["sessions"]) == 2
    assert s2["gates"]["t9"] == g["t9"] and s2["gates"]["t10"] == g["t10"]  # recomputed offline from the kept records
    assert s2["provenance"]["run1"] == s["provenance"]["run1"] and s2["provenance"]["t8"]["session"] == 2
    assert not (out / "run.pid").exists() and _reap_own_fakes(pids_file, exe) == []
    # a Run 1 rerun that fails leaves no exit code: T6 is ERROR (it cannot be judged), never a FAIL on rc None
    monkeypatch.setattr(v03t, "_calibrate", lambda argv: (2, "callo: refusing to resume: boom\n"))
    rc3 = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(doc), "--parts", "run1", "--min-free-gb", "0"])
    s3 = json.loads((out / "summary.json").read_text())
    assert rc3 == 1 and "run1" not in s3["runs"] and s3["errors"]["run1"].startswith("RuntimeError: calibrate --menu run1")
    assert s3["gates"]["t6"] == {"error": "Run 1 did not finish in this session (no exit code recorded): T6 cannot be "
                                          "judged", "pass": None}
    assert _row(doc.read_text(), "T6").endswith("| ERROR |")


def _stub_calibrate(seen: list, rc_for=lambda argv: 0, err: str = ""):
    def calibrate(argv):
        seen.append(list(argv))
        return rc_for(argv), err
    return calibrate


def test_a_menu_refusal_is_the_parts_error_and_an_interrupt_aborts_the_bench(tmp_path, monkeypatch):
    src = _source_file(tmp_path / "test.jsonl", 4)
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(v03t, "DISK_PATHS", (str(tmp_path),))
    monkeypatch.setattr(v03t, "MIN_FREE_GIB", 0.0)
    monkeypatch.setattr(v03t, "RUN2_N", 2)
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    pids: list[str] = []

    def refuse(argv):
        pids.append((tmp_path / "out" / "run.pid").read_text().strip())     # run.pid holds the bench's pid all along
        return 2, "callo: refusing to calibrate: another model server is running (pid 9)\n"

    monkeypatch.setattr(v03t, "_calibrate", refuse)
    out = tmp_path / "out"
    rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(tmp_path / "doc.md"), "--parts", "run1,run2", "--min-free-gb", "0"])
    s = json.loads((out / "summary.json").read_text())
    assert rc == 1 and s["errors"]["run1"].startswith("RuntimeError: calibrate --menu run1.menu.json refused (exit 2): callo: refusing")
    assert s["errors"]["run2"].startswith("RuntimeError") and "runs" in s and s["runs"] == {}
    assert pids == [str(os.getpid())] * 2 and not (out / "run.pid").exists()
    assert {sig: signal.getsignal(sig) for sig in before} == before            # the stop handlers are restored
    # a run whose DIR holds a run.json resumes it; --allow-gpu-busy starts beside a model server, recorded and passed on
    seen: list = []
    monkeypatch.setattr(v03t, "_calibrate", _stub_calibrate(seen))
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [(9, "llama-server -m x.gguf")])
    (out / "run1").mkdir()
    (out / "run1" / "run.json").write_text("{}")
    rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(tmp_path / "doc.md"), "--parts", "run1",
                    "--min-free-gb", "0", "--allow-gpu-busy"])
    s = json.loads((out / "summary.json").read_text())
    assert rc == 0 and seen[0][-2:] == ["--allow-gpu-busy", "--resume"] and "--settle" not in seen[0]   # callo's 30 s
    assert "--no-cache" in seen[0]                         # the bench's runs are cache off (the default is on since 0.3.0)
    assert v03t.t5_probe_argv("http://127.0.0.1:9", Path("s.jsonl"), Path("o"))[-1] == "--no-cache"   # the T5 probe's too
    assert "--allow-gpu-busy" in seen[0] and s["sessions"][-1]["gpu_busy_at_start"] == [[9, "llama-server -m x.gguf"]]
    assert s["sessions"][-1]["allow_gpu_busy"] is True and "run1" not in s["errors"] and s["runs"]["run1"]["rc"] == 0
    assert s["errors"]["run2"].startswith("RuntimeError")                     # a part not rerun keeps its record
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(v03t, "_calibrate", lambda argv: (130, ""))
    out2 = tmp_path / "out2"
    rc = v03t.main(["--out", str(out2), "--data", str(src), "--doc", str(tmp_path / "doc.md"), "--parts", "run1,run2", "--min-free-gb", "0"])
    s = json.loads((out2 / "summary.json").read_text())
    assert rc == 3 and s["aborted"] == "interrupted (KeyboardInterrupt)" and s["errors"] == {}   # the rest never ran
    assert "**ABORTED:** interrupted (KeyboardInterrupt)" in (tmp_path / "doc.md").read_text()
    monkeypatch.setattr(v03t, "disk_problems", lambda paths, min_gib: ["/: 1.0 GiB free < 20.0"])
    rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(tmp_path / "doc.md"), "--parts", "run1", "--min-free-gb", "0"])
    assert rc == 2                                                             # the start-up disk check refuses first


def test_the_offline_part_t8_needs_no_model_paths_and_a_gpu_part_does(tmp_path, monkeypatch, capsys):
    src = _source_file(tmp_path / "test.jsonl", 4)
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(v03t, "DISK_PATHS", (str(tmp_path),))
    monkeypatch.setattr(v03t, "MIN_FREE_GIB", 0.0)
    monkeypatch.setattr(v03t, "_calibrate", _stub_calibrate([]))
    for table in ("MODELS", "RUN2_MODELS", "RUN3_MODELS"):                    # CALLO_BENCH_MODELS unset
        monkeypatch.setattr(v03t, table, {k: {**m, "model": None} for k, m in getattr(v03t, table).items()})
    out, doc = tmp_path / "out", tmp_path / "doc.md"
    assert v03t.main(["--out", str(out), "--data", str(src), "--doc", str(doc), "--parts", "t8", "--min-free-gb", "0"]) == 0
    assert "required path" not in capsys.readouterr().out
    gpu_out = tmp_path / "gpu"
    assert v03t.main(["--out", str(gpu_out), "--data", str(src), "--doc", str(doc), "--parts", "t8,run1",
                      "--min-free-gb", "0"]) == 2
    assert "refusing to start: a required path is not set (" in capsys.readouterr().out and not gpu_out.exists()


def test_a_part_that_did_not_run_leaves_its_gates_missing_never_pass(tmp_path, monkeypatch):
    """`--parts t8` alone on a fresh DIR: T4, T5, T9 and T10 are NOT RUN, T1-T3 and T6 ERROR (no profile), never PASS;
    and a part that fails now drops the verdict an earlier session left for it."""
    src = _source_file(tmp_path / "test.jsonl", 4)
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(v03t, "DISK_PATHS", (str(tmp_path),))
    monkeypatch.setattr(v03t, "MIN_FREE_GIB", 0.0)
    seen: list = []
    monkeypatch.setattr(v03t, "_calibrate", _stub_calibrate(seen))
    out, doc = tmp_path / "out", tmp_path / "doc.md"
    rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(doc), "--parts", "t8", "--min-free-gb", "0"])
    s = json.loads((out / "summary.json").read_text())
    assert rc == 0 and seen == []                                              # no profile: compare is not even run
    assert s["gates"]["t8"]["pass"] is None and "Run 1" in s["gates"]["t8"]["error"]
    text = doc.read_text()
    for gate in ("T4", "T5", "T9", "T10"):
        assert _row(text, gate).endswith("| NOT RUN |"), gate
    for gate in ("T1", "T2", "T3", "T6", "T8"):
        assert _row(text, gate).endswith("| ERROR |"), gate
    assert all(_verdict(g) != "PASS" for g in s["gates"].values())
    # an earlier session's T4 is dropped when the serve part runs again and cannot produce it
    s["gates"]["t4"] = {"cases": {}, "pass": True}
    (out / "summary.json").write_text(json.dumps(s))
    rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(doc), "--parts", "serve,t5", "--min-free-gb", "0"])
    s = json.loads((out / "summary.json").read_text())
    assert rc == 1 and s["errors"]["serve:tiel"] == "no Run 1 profile" and s["errors"]["serve:qwen9b"] == "no Run 1 profile"
    # the t5 part runs after serve: it keeps the reason serve wrote this session
    assert s["errors"]["t5"] == "T5 was not judged: the tiel serve-path session did not run"
    assert "- t5: T5 was not judged" in doc.read_text()
    assert s["gates"]["t4"]["pass"] is None and s["gates"]["t10"]["tiel"]["error"].startswith("no serve-path session")
    assert "t5" not in s["gates"] and _row(doc.read_text(), "T4").endswith("| ERROR |") and "serve" not in s
    # the disks are checked before every part: one that fills mid-run stops the bench before the next part starts,
    # and that part keeps the verdict an earlier session left (it never started)
    calls = iter([[], ["/: 1.0 GiB free (< 20 GiB)"]])
    monkeypatch.setattr(v03t, "disk_problems", lambda paths, min_gib: next(calls))
    rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(doc), "--parts", "t8,t7", "--min-free-gb", "0"])
    s2 = json.loads((out / "summary.json").read_text())
    assert rc == 1 and s2["errors"]["t8"] == "not started: disk: /: 1.0 GiB free (< 20 GiB)" and "t7" not in s2["errors"]
    assert s2["gates"]["t8"] == s["gates"]["t8"] and "t7" not in s2["gates"]


def test_main_refuses_while_a_model_server_runs_or_with_an_unknown_part(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [(9, "llama-server")])
    assert v03t.main(["--out", str(tmp_path / "o")]) == 2 and "refusing to start: a model server" in capsys.readouterr().out
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    assert v03t.main(["--out", str(tmp_path / "o"), "--parts", "bogus"]) == 2 and "unknown part" in capsys.readouterr().out
    assert v03t.main(["--out", str(tmp_path / "o"), "--data", str(tmp_path / "nope.jsonl"), "--min-free-gb", "0"]) == 2
    assert "no data file" in capsys.readouterr().out
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 1)                 # the data check comes first: a real file
    src = _source_file(tmp_path / "test.jsonl", 4)
    assert v03t.main(["--out", str(tmp_path / "o"), "--data", str(src), "--min-free-gb", "1"]) == 2
    assert "vm_stat" in capsys.readouterr().out
    assert not (tmp_path / "o" / "run.pid").exists()
    assert v03t._all3([True, None]) is None and v03t._all3([True, True]) is True and v03t._all3([None, False]) is False


def test_the_bench_never_binds_or_calls_another_sessions_ports(monkeypatch):
    """8080 and 8130 belong to other sessions; a launch asking for one is refused before it starts, and a session
    that came up on one is stopped and refused."""
    assert v03t.FORBIDDEN_PORTS == (8080, 8130)
    assert v03t.port_problems(["http://127.0.0.1:51234", None]) == []
    assert v03t.port_problems(["http://127.0.0.1:8130"]) == [
        "http://127.0.0.1:8130 is on port 8130, another session's server: the bench never binds or calls 8080 or 8130"]
    with pytest.raises(RuntimeError, match="port 8080"):
        v03t._start("x", Path("/nonexistent"), model="/m.gguf", flags=["--port", "8080"], ctx=4096)

    class Up:
        url, upstream, finished = "http://127.0.0.1:41000", "http://127.0.0.1:8080", False

        def finish(self):
            Up.finished = True

    monkeypatch.setattr(v03t, "_launch", lambda name, out, **kw: Up())
    with pytest.raises(RuntimeError, match="8080"):
        v03t._start("x", Path("/nonexistent"), model="/m.gguf", flags=[], ctx=4096)
    assert Up.finished is True


def test_think_fields_ok_reads_x_callo_timings_and_merge_cases_keeps_a_failure():
    zero = {"x-callo-timings": json.dumps([{"q": "a/q", "ms": 3.0}])}
    deliberate = {"x-callo-timings": json.dumps([{"q": "a/q", "ms": 3.0, "think_budget": 64, "think_tokens": 40,
                                                  "think_close": "natural", "gen_ms": 9.0, "read_prompt_n": 2}])}
    assert v03t._think_fields_ok(200, zero, positive=False) is True and v03t._think_fields_ok(200, deliberate, positive=True) is True
    assert v03t._think_fields_ok(200, deliberate, positive=False) is False and v03t._think_fields_ok(200, zero, positive=True) is False
    assert v03t._think_fields_ok(500, zero, positive=False) is None and v03t._think_fields_ok(200, {}, positive=False) is None
    assert v03t._think_fields_ok(200, {"x-callo-timings": "[{\"truncated\": 3}]"}, positive=False) is None
    cases: dict = {}
    v03t._merge_cases(cases, {"think_budget_x": 422, "image_with_budget": None, "ok": 200})
    v03t._merge_cases(cases, {"think_budget_x": 200, "image_with_budget": 422, "ok": 200})
    v03t._merge_cases(cases, {"think_budget_x": 422})
    assert cases == {"think_budget_x": 200, "image_with_budget": 422, "ok": 200}     # a failure in any session stands
    assert v03t._fresh_dir(Path("/nonexistent/t7")) == Path("/nonexistent/t7")


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StubCallo:
    """A serve-path session stand-in for `_launch`: no server at all; the port is `--port`'s when given."""
    instances = 0

    def __init__(self, flags, port):
        _StubCallo.instances += 1
        asked = [flags[i + 1] for i, f in enumerate(flags[:-1]) if f == "--port"]
        self.url = f"http://127.0.0.1:{asked[-1] if asked else port}"
        self.upstream, self.instance, self.finished = "http://127.0.0.1:1", f"i{_StubCallo.instances}", False

    def stats(self):
        return {"instance": self.instance}

    def finish(self):
        self.finished = True


def test_the_small_session_binds_the_lan_only_for_the_403_and_restarts_on_loopback_on_the_same_url(tmp_path, monkeypatch):
    """The restart reuses no `--host 0.0.0.0`; it is judged through the OLD URL (same port, a new
    instance, a 200), and a restart that comes up elsewhere is a failure with its reason."""
    data = _source_file(tmp_path / "d.jsonl", 2)
    port = _free_port()
    launches: list[list[str]] = []

    def launch(name, out, *, model, flags, ctx):
        launches.append(list(flags))
        return _StubCallo(flags, port)

    def status(url, body):
        return 403 if url.endswith("/callo/think-probe") else 413 if body.get("think_budget") == v03t.SMALL_BUDGET else 200

    images: list = []

    def post_full(url, body):                                                # T4's image case, with its own reason
        images.append(body)
        return 422, {}, {"detail": "deliberate readouts are text-only in this version"}

    monkeypatch.setattr(v03t, "_launch", launch)
    monkeypatch.setattr(v03t, "_lan_ip", lambda: "192.0.2.1")
    monkeypatch.setattr(v03t, "_status", status)
    monkeypatch.setattr(v03t, "_post_full", post_full)
    monkeypatch.setattr(v03t, "MODELS", {"qwen9b": {"model": "/q.gguf"}})
    mm = tmp_path / "q-mmproj.gguf"
    mm.write_bytes(b"GGUF")
    monkeypatch.setattr(v03t, "SERVE_MMPROJ", {"qwen9b": str(mm)})
    monkeypatch.setattr(v03t.lib, "get", lambda url: {"instance": "fresh"})
    prof = tmp_path / "q.think.json"
    prof.write_text(json.dumps({**MIN_PROFILE, "provenance": {"server_args": ["--cache-ram", "0"]}}))
    r = v03t.small_session(tmp_path / "serve", prof, data)
    assert r["cases"] == {"past_n_ctx": 413, "image_with_budget": 422, "probe_non_loopback": 403} and r["lan"] == "192.0.2.1"
    assert "case_errors" not in r and images[0]["images"] == [v03t.PNG_1x1] and images[0]["think_budget"] == 256
    # the profile's llama-server args after `--`, the projector for the image case, no --think-profile
    assert launches[0] == ["--think-budget", "0", "--mmproj", str(mm), "--host", "0.0.0.0", "--", "--cache-ram", "0"]
    assert "--host" not in launches[1] and "0.0.0.0" not in launches[1]
    assert launches[1] == ["--think-budget", "0", "--mmproj", str(mm), "--port", str(port), "--", "--cache-ram", "0"]
    assert r["restart"]["pass"] is True and r["restart"]["old_instance"] == "i1" and r["restart"]["new_instance"] == "fresh"
    # a projector that is not on disk: launched without it, only the image case drops out (ERROR with the reason);
    # the 413, the 403 and the restart are still measured
    monkeypatch.setattr(v03t, "SERVE_MMPROJ", {"qwen9b": str(tmp_path / "gone-mmproj.gguf")})
    launches.clear()
    images.clear()
    gone = v03t.small_session(tmp_path / "serve", prof, data)
    assert gone["cases"] == {"past_n_ctx": 413, "probe_non_loopback": 403} and images == []
    assert all("--mmproj" not in f for f in launches) and gone["restart"]["pass"] is True
    assert gone["case_errors"] == {"image_with_budget": f"qwen9b-small: the mmproj {tmp_path / 'gone-mmproj.gguf'} is "
                                                        f"missing: the image was not sent"}
    g4 = v03t.t4({**v03t.T4_EXPECTED, **gone["cases"], "image_with_budget": None}, disconnect_clear=True, self_test=True,
                 timings_ok=True, case_errors=gone["case_errors"])
    assert g4["pass"] is None and g4["not_testable"] == ["image_with_budget"] and "gone-mmproj.gguf is missing" in g4["error"]
    # the seam's failure path: the restart lands on another port and nothing answers on the old URL
    monkeypatch.setattr(v03t, "_launch", lambda name, out, *, model, flags, ctx: _StubCallo([], _free_port()))
    monkeypatch.setattr(v03t.lib, "get", lambda url: (_ for _ in ()).throw(ConnectionRefusedError(61, "refused")))
    monkeypatch.setattr(v03t, "_status", lambda url, body: -1)
    bad = v03t.same_url_restart("x", tmp_path, model="/q.gguf", flags=["--host", "0.0.0.0"], ctx=4096,
                                url=f"http://127.0.0.1:{port}", old_instance="i1", body={"state": "s"})
    assert bad["pass"] is False and bad["flags"] == ["--port", str(port)]
    passthrough = v03t.same_url_restart("x", tmp_path, model="/q.gguf", flags=["--host", "0.0.0.0", "--", "--host", "h"],
                                        ctx=4096, url=f"http://127.0.0.1:{port}", old_instance="i1", body={"state": "s"})
    assert passthrough["flags"] == ["--port", str(port), "--", "--host", "h"]       # llama-server's own args untouched
    assert bad["error"].startswith(f"/callo/stats on http://127.0.0.1:{port} did not answer")
    assert f"not {port}" in bad["error"] and "answered -1, not 200" in bad["error"]


def test_a_failed_restart_is_the_serve_parts_error_exit_1_and_in_the_doc(tmp_path, monkeypatch):
    src = _source_file(tmp_path / "test.jsonl", 4)
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(v03t, "DISK_PATHS", (str(tmp_path),))
    monkeypatch.setattr(v03t, "MIN_FREE_GIB", 0.0)
    monkeypatch.setattr(v03t, "_lan_ip", lambda: None)
    out, doc = tmp_path / "out", tmp_path / "doc.md"
    _run_dir(out, "run1", {m: ("done", None, str(out / "run1" / f"{m}.think.json")) for m in ("tiel", "qwen9b", "gemma-e4b")})
    session = {"cases": {"ok": 200}, "stats_think": {}, "served_readouts": [], "t10": {"n": 0, "pass": None, "error": "x"},
               "disconnect_clear": True, "self_test": True, "timings_ok": True}
    monkeypatch.setattr(v03t, "serve_session", lambda name, out, prof, data, guard=False: dict(session))
    monkeypatch.setattr(v03t, "gemma_session", lambda out, prof, data: {"cases": {"gemma_budget": 422}})
    monkeypatch.setattr(v03t, "small_session", lambda out, prof, data: {
        "cases": {"past_n_ctx": 413}, "restart": {"pass": False, "error": "the restart came up on port 2, not 1"}})
    rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(doc), "--parts", "serve", "--min-free-gb", "0"])
    s = json.loads((out / "summary.json").read_text())
    assert rc == 1 and s["errors"]["serve"] == "same-URL restart of qwen9b-small failed: the restart came up on port 2, not 1"
    assert "- serve: same-URL restart of qwen9b-small failed" in doc.read_text()


def test_run_pid_is_owned_a_live_owner_refuses_a_dead_one_is_replaced_and_only_our_own_is_removed(tmp_path, monkeypatch, capsys):
    """Nothing is signalled (a liveness check with signal 0 only): this process's own pid is a live owner,
    2**22 is above every platform's pid limit, so never live."""
    src = _source_file(tmp_path / "test.jsonl", 4)
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(v03t, "DISK_PATHS", (str(tmp_path),))
    monkeypatch.setattr(v03t, "MIN_FREE_GIB", 0.0)
    out = tmp_path / "out"
    out.mkdir()
    pid_file = out / "run.pid"
    argv = ["--out", str(out), "--data", str(src), "--doc", str(tmp_path / "doc.md"), "--parts", "t8", "--min-free-gb", "0"]
    pid_file.write_text(f"{os.getpid()}\n")
    assert v03t.main(argv) == 2 and pid_file.read_text() == f"{os.getpid()}\n" and not (out / "summary.json").exists()
    assert f"names pid {os.getpid()}, which is not a bench run" in capsys.readouterr().out   # pytest is not the bench
    pid_file.write_text(f"{2**22}\n")                                         # stale: replaced, then removed at exit
    assert v03t.pid_owner(pid_file) is None and v03t.main(argv) == 0 and not pid_file.exists()
    pid_file.write_text("garbage\n")
    assert v03t.pid_owner(pid_file) is None

    def overwrite(a):                                                         # another run took the file over meanwhile
        pid_file.write_text(f"{2**22}\n")
        return 0, ""

    monkeypatch.setattr(v03t, "_calibrate", overwrite)
    assert v03t.main([*argv[:-3], "run1", "--min-free-gb", "0"]) == 0 and pid_file.read_text() == f"{2**22}\n"


def test_t10_counts_the_first_48_measured_questions_skipping_a_non_200():
    """A transient 503 among the first 48 rows is skipped and counted, not one of the 48."""
    from types import SimpleNamespace
    rows = [{"q": f"q{i}", "b": {"64": [True, 0.5, 0.6, 1000.0]}} for i in range(49)]
    by_uid = {r["q"]: SimpleNamespace(body={"state": r["q"]}) for r in rows}
    # `ms` is the question's whole engine time, the generation (`gen_ms`) inside it: the served time is 1000, never
    # 1990 (gen_ms + ms counted the generation twice: the first live run's T10 FAIL)
    ok = {"x-callo-timings": json.dumps([{"q": "q", "ms": 1000.0, "gen_ms": 990.0, "think_close": "natural",
                                          "read_prompt_n": 2, "think_tokens": 40}])}

    def post(body):
        return (503, {}) if body["state"] == "q3" else (200, ok)

    served, readouts, skipped, stopped = v03t.served_latencies(post, rows, by_uid, 64, 48)
    assert len(served) == 48 and "q3" not in served and "q48" in served and skipped == {"503": 1} and len(readouts) == 48
    assert stopped is None and set(served.values()) == {1000.0} and readouts[0]["gen_ms"] == 990.0 and readouts[0]["ms"] == 1000.0
    t = v03t.t10(served, {"rows": rows}, 64)
    assert t["n"] == 48 and t["pass"] is True and "error" not in t
    served, _, skipped, _ = v03t.served_latencies(post, rows[:48], by_uid, 64, 48)
    assert len(served) == 47 and v03t.t10(served, {"rows": rows}, 64)["error"] == "not testable: 47 shared questions < 48"
    served, _, skipped, _ = v03t.served_latencies(lambda body: (200, {"x-callo-timings": "nope"}), rows[:2], by_uid, 64, 48)
    assert served == {} and skipped == {"no timings": 2}


def test_startup_warnings_name_what_will_end_t4_error(monkeypatch):
    monkeypatch.setattr(v03t, "_lan_ip", lambda: None)
    monkeypatch.setattr(v03t, "SERVE_MMPROJ", {"tiel": "/nonexistent/mmproj.gguf"})
    w = v03t.startup_warnings(["serve"], "version: 7000 (4ebdf2c)\nbuilt with clang", {"/": 100.0, "/o": 25.0})
    assert w == ["no LAN address: T4's 403 case (probe_non_loopback) will be not testable, so T4 can only end ERROR",
                 "no mmproj configured for qwen9b (the small launch): T4's image 422 (image_with_budget) will be not "
                 "testable, so T4 can only end ERROR",
                 "/o: 25.0 GiB free, near the 20 GiB the bench stops at between parts"]
    monkeypatch.setattr(v03t, "SERVE_MMPROJ", {"qwen9b": "/nonexistent/mmproj.gguf"})
    assert v03t.startup_warnings(["serve"], "version: 7000 (4ebdf2c)", {"/": 100.0})[1] == (
        "the mmproj for qwen9b (the small launch) is missing (/nonexistent/mmproj.gguf): T4's image 422 "
        "(image_with_budget) will be not testable, so T4 can only end ERROR")
    assert v03t.startup_warnings(["t8"], "version: 1 (abc)", {"/": None}) == [
        "llama-server --version does not name the expected build 4ebdf2c: 'version: 1 (abc)'",
        "/: unknown free, near the 20 GiB the bench stops at between parts"]


def test_the_doc_shows_each_parts_provenance_and_the_data_sha256():
    s = {"provenance": {"run1": {"commit": "a" * 40, "dirty": False, "llama_server_version": "ggml_metal_init\nversion: 7000 (4ebdf2c)\nbuilt with clang"},
                        "serve": {"commit": "b" * 40, "dirty": True, "llama_server_version": "version: 7001 (5f00000)"}},
         "data": {"sha256": "d" * 64, "source_sha256": "e" * 64, "run2_sha256": "f" * 64},
         "warnings": ["no LAN address: x"]}
    text = render(s)
    assert "## Provenance\n\n| part | callo commit | llama-server |\n|---|---|---|\n" in text
    assert f"| run1 | {'a' * 12} | version: 7000 (4ebdf2c) |" in text and f"| serve | {'b' * 12} dirty | version: 7001 (5f00000) |" in text
    assert f"**Mixed commits:** the parts ran on 2 different commits ({'a' * 12}, {'b' * 12})." in text
    assert f"Data sha256: test `{'d' * 64}` (selected from `{'e' * 64}`), Run 2 subset `{'f' * 64}`." in text
    assert "## Warnings (at the last start)\n\n- no LAN address: x" in text
    assert "Mixed commits" not in render({"provenance": {"run1": s["provenance"]["run1"], "t8": s["provenance"]["run1"]}})


# ---- what the gates may use, compare's reason, the GPU rule, run.pid, the T10 loop, Ctrl-C ---------------------------

def _bench_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(v03t, "DISK_PATHS", (str(tmp_path),))
    monkeypatch.setattr(v03t, "MIN_FREE_GIB", 0.0)


MIN_PROFILE = {"callo_think_profile": 1, "environment": {"server": {}}}


def _run_dir(out: Path, run: str, models: dict[str, tuple[str, str | None, str | None]]) -> Path:
    """`out/run` with a profile on disk for every model and a run.json recording {name: (status, reason, profile)}."""
    d = out / run
    d.mkdir(parents=True, exist_ok=True)
    for name in models:
        (d / f"{name}.think.json").write_text(json.dumps({**MIN_PROFILE, "model_name": name}))
        (d / f"{name}.raw.jsonl").write_text(json.dumps({"callo_think_raw": RAW_FORMAT}) + "\n")
    (d / "run.json").write_text(json.dumps({
        "callo_think_run": 1, "menu_sha256": "x", "settings": {}, "created": "t", "order": list(models), "dropped": [],
        "models": {n: {"status": st, "reason": why, "started": "t", "ended": "t", "profile": p}
                   for n, (st, why, p) in models.items()}}))
    return d


def test_a_profile_run_json_does_not_mark_done_never_reaches_a_gate(tmp_path, monkeypatch):
    """A profile on disk whose model run.json records failed (another server appeared), still running, or done
    under another path is the `profile:` error and missing for every gate: T3 / T8 / serve name it, nothing PASSes."""
    _bench_env(monkeypatch, tmp_path)
    src = _source_file(tmp_path / "test.jsonl", 4)
    out = tmp_path / "out"
    _run_dir(out, "run1", {"tiel": ("failed", "another model server appeared (pid 7: ollama serve)", None),
                           "qwen9b": ("running", None, None),
                           "gemma-e4b": ("done", None, str(tmp_path / "elsewhere" / "gemma-e4b.think.json"))})
    _run_dir(out, "run2", {"minicpm": ("done", None, str(out / "run2" / "minicpm.think.json"))})
    launched: list = []
    monkeypatch.setattr(v03t, "_launch", lambda *a, **k: launched.append(a) or pytest.fail("no session may launch"))
    seen: list = []
    monkeypatch.setattr(v03t, "_calibrate", _stub_calibrate(seen))
    doc = tmp_path / "doc.md"
    rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(doc), "--parts", "t8,serve", "--min-free-gb", "0"])
    s = json.loads((out / "summary.json").read_text())
    assert rc == 1 and launched == [] and seen == []                          # nothing launched, compare never run
    assert s["errors"]["profile:run1/tiel"] == "run1/run.json says tiel failed: another model server appeared (pid 7: ollama serve)"
    assert s["errors"]["profile:run1/qwen9b"] == "run1/run.json says qwen9b running"
    assert s["errors"]["profile:run1/gemma-e4b"].startswith("run1/run.json records gemma-e4b's profile as ")
    assert s["errors"]["serve:tiel"] == s["errors"]["profile:run1/tiel"] and s["errors"]["serve:qwen9b-small"] == s["errors"]["profile:run1/qwen9b"]
    assert "run1/tiel.think.json" in s["gates"]["t8"]["error"] and s["gates"]["t8"]["pass"] is None
    assert s["gates"]["t3"]["tiel"] == {"error": "no profile (expected for T3)", "pass": None} and "minicpm" in s["gates"]["t3"]
    assert s["gates"]["t6"]["pass"] is None and "run1/run.json records gemma-e4b's profile" in s["gates"]["t6"]["error"]
    assert set(s["gates"]["t9"]["calibrate"]) == {"tiel", "qwen9b", "minicpm"}             # T9: their raw files are out
    assert s["gates"]["t9"]["calibrate"]["tiel"]["error"].startswith("no raw file")
    assert all(_verdict(g) != "PASS" for g in s["gates"].values())


def test_t4_is_incomplete_when_any_thinking_session_could_not_tell(tmp_path, monkeypatch):
    """One session's None (a disconnect probe that failed, a non-200 budget request) makes that T4 check not
    testable, so T4 is ERROR, never a PASS on the other session's True."""
    out = tmp_path / "out"
    _run_dir(out, "run1", {n: ("done", None, str(out / "run1" / f"{n}.think.json")) for n in ("tiel", "qwen9b", "gemma-e4b")})
    cases = {k: v for k, v in v03t.T4_EXPECTED.items() if k not in ("gemma_budget", "past_n_ctx", "probe_non_loopback")}

    def session(name, out_dir, prof, data, *, guard=False):
        return {"cases": cases, "stats_think": {}, "served_readouts": [], "t10": {"pass": None, "error": "x"},
                "disconnect_clear": None if name == "qwen9b" else True, "self_test": True,
                "timings_ok": True if name == "qwen9b" else None, "budget_note": "the profile's pick"}

    monkeypatch.setattr(v03t, "serve_session", session)
    monkeypatch.setattr(v03t, "gemma_session", lambda *a: {"cases": {"gemma_budget": 422}})
    monkeypatch.setattr(v03t, "small_session", lambda *a: {"cases": {"past_n_ctx": 413, "probe_non_loopback": 403},
                                                           "restart": {"pass": True}})
    s: dict = {"errors": {}, "gates": {}}
    v03t._part_serve(s, out, tmp_path / "d.jsonl", ["serve"], lambda k, e: None)
    t4 = s["gates"]["t4"]
    assert t4["disconnect_clear"] is None and t4["timings_ok"] is None and t4["self_test"] is True
    assert t4["pass"] is None and t4["not_testable"] == ["timings_ok", "disconnect_clear"], t4


def test_t6_judges_gemmas_own_status_and_its_point_on_the_joint_frontier():
    """T6 needs gemma's own run.json status done (Run 1's exit is descriptive: another model's failure does not
    fail it) and gemma among the joint report's points."""
    prof = {"model_name": "g", "thinking": {"capable": False, "reason": "unsupported_think_format"}, "budgets": {"0": {}}}
    report = {"notes": ["g: budget 0 only: thinks in a format callo cannot read"], "points": [{"label": "g@0", "model": "g"}]}
    ok = v03t.t6(prof, report, status="done", rc=1)
    assert ok["pass"] is True and ok["rc"] == 1 and ok["status"] == "done" and ok["on_frontier"] is True
    assert v03t.t6(prof, report, status="failed", rc=1)["pass"] is False
    assert v03t.t6(prof, {**report, "points": [{"label": "t@0", "model": "t"}]}, status="done", rc=0)["pass"] is False
    assert "on the joint report's points True" in render({"gates": {"t6": ok}})


def test_t8_needs_the_different_data_refusal_not_any_exit_2():
    """The refusal half needs compare's own reason in stderr, so a malformed profile's exit 2 is no PASS."""
    assert v03t.t8(0, 2, "callo: b was measured on different data (x.jsonl, sha256 ab)")["pass"] is True
    bad = v03t.t8(0, 2, "callo: b.think.json: not JSON")
    assert bad["pass"] is False and bad["refuse_reason"] == "callo: b.think.json: not JSON"


def test_t9s_served_half_is_error_without_read_prompt_n_never_the_counter_proxy():
    """No read_prompt_n in X-Callo-Timings (or no natural close) is that half's ERROR, not a PASS on the
    server's readout_reprefill counter."""
    counter_only = {"tiel": {"think": {"natural": 30, "forced": 0, "eos": 0, "readout_reprefill": 0, "text_fallback": 0},
                             "readouts": [{"q": "q", "think_close": "natural", "read_prompt_n": None}]}}
    g = _t9({}, counter_only)
    assert g["served"]["tiel"] == {"error": "not testable: no read_prompt_n in X-Callo-Timings", "pass": None,
                                   "text_fallback": 0} and g["pass"] is None
    forced = {"tiel": {"think": {}, "readouts": [{"q": "q", "think_close": "forced", "read_prompt_n": 259}]}}
    assert _t9({}, forced)["served"]["tiel"]["error"] == "not testable: no natural close among 1 served readouts"
    fallback = {"tiel": {"think": {"text_fallback": 2}, "readouts": []}}
    assert _t9({}, fallback)["served"]["tiel"]["pass"] is False                # a measured fallback still FAILs


def test_t10_renders_why_the_session_served_its_budget():
    """The T10 cell says why the session served that budget (`budget_note`)."""
    e = {"n": 48, "budget": 1024, "served_p50_ms": 1500.0, "profile_p50_ms": 1400.0, "rel_err": 0.071, "pass": True,
         "budget_note": "the profile picked 0; served at Bmax so T4 and T10 see a deliberate readout"}
    assert ("tiel@1024 (the profile picked 0; served at Bmax so T4 and T10 see a deliberate readout): served p50 1500"
            in render({"gates": {"t10": {"tiel": e}}}))


def test_the_benchs_gpu_rule_is_callos_and_fails_closed_when_ps_fails(tmp_path, monkeypatch, capsys):
    """The bench classifies with callo.menu's GPU rule and reads ps through menu.ps_listing: a ps that
    exits non-zero (or lists nothing) is "cannot tell", a refusal at the start and before a part, never "no servers"."""
    assert lib.gpu_class is menu.gpu_class
    monkeypatch.setattr(menu, "ps_listing", lambda run=None: (None, "exit 1: boom"))
    with pytest.raises(RuntimeError, match=r"could not list processes \(ps failed: exit 1: boom\)"):
        lib.running_gpu_servers()
    assert v03t.main(["--out", str(tmp_path / "o")]) == 2
    assert "refusing to start: could not list processes (ps failed: exit 1: boom)" in capsys.readouterr().out
    monkeypatch.setattr(menu, "ps_listing", lambda run=None: (BENIGN_PS + "  9 /opt/llama-server -m x\n", None))
    assert [pid for pid, _ in lib.running_gpu_servers()] == [9]
    calls = iter([(BENIGN_PS, None), (None, "exit 1: boom")])               # fine at the start, then ps fails
    monkeypatch.setattr(menu, "ps_listing", lambda run=None: next(calls))
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(v03t, "DISK_PATHS", (str(tmp_path),))
    monkeypatch.setattr(v03t, "MIN_FREE_GIB", 0.0)
    src = _source_file(tmp_path / "test.jsonl", 4)
    out = tmp_path / "out"
    rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(tmp_path / "doc.md"), "--parts", "t8", "--min-free-gb", "0"])
    s = json.loads((out / "summary.json").read_text())
    assert rc == 1 and s["errors"]["t8"] == "not started: cannot tell whether a model server is running: could not list " \
                                           "processes (ps failed: exit 1: boom)"


def test_one_definition_of_the_shared_constants_and_predicates():
    """PREFILL_REUSED_MAX lives in calibrate_think; one finite-number predicate serves compare, menu and the bench."""
    from callo import calibrate_think, compare, think
    assert compare.PREFILL_REUSED_MAX is calibrate_think.PREFILL_REUSED_MAX
    assert v03t.finite_number is think.finite_number and menu.finite_positive is think.finite_positive
    assert think.finite_number(1.5) and not think.finite_number(True) and not think.finite_number(float("nan"))
    assert not think.finite_positive(10**400) and not think.finite_positive(0) and think.finite_positive(10**300)


def test_a_live_run_pid_that_is_not_a_bench_run_is_named_never_suggested_to_signal(tmp_path, monkeypatch, capsys):
    """A live pid in run.pid is "another bench run" only when its `ps -o args=` names acceptance_v03_think; this
    pytest process's own pid is not one."""
    _bench_env(monkeypatch, tmp_path)
    src = _source_file(tmp_path / "test.jsonl", 4)
    out = tmp_path / "out"
    out.mkdir()
    (out / "run.pid").write_text(f"{os.getpid()}\n")
    argv = ["--out", str(out), "--data", str(src), "--doc", str(tmp_path / "doc.md"), "--parts", "t8", "--min-free-gb", "0"]
    assert v03t.main(argv) == 2
    said = capsys.readouterr().out
    assert (f"refusing to start: {out / 'run.pid'} names pid {os.getpid()}, which is not a bench run: check it and "
            f"remove run.pid by hand") in said and "stop it" not in said
    monkeypatch.setattr(v03t, "_pid_args", lambda pid: "/usr/bin/python3 -m bench.acceptance_v03_think --out x")
    assert v03t.main(argv) == 2
    assert f"another bench run (pid {os.getpid()}) owns this --out" in capsys.readouterr().out


def test_served_latencies_stop_after_48_non_200_replies():
    """Persistent 504s stop the T10 loop after 48 skipped non-200 replies; T10 says why (ERROR)."""
    from types import SimpleNamespace
    rows = [{"q": f"q{i}", "b": {"64": [True, 0.5, 0.6, 1000.0]}} for i in range(400)]
    by_uid = {r["q"]: SimpleNamespace(body={"state": r["q"]}) for r in rows}
    sent: list = []
    served, readouts, skipped, stopped = v03t.served_latencies(lambda body: sent.append(body) or (504, {}), rows, by_uid, 64)
    assert len(sent) == 48 and served == {} and skipped == {"504": 48}
    assert stopped == "too many non-200 replies: 48 (504: 48) before 48 measured questions"
    assert v03t.t10(served, {"rows": rows}, 64, stopped=stopped) == {"n": 0, "budget": 64, "pass": None, "error": stopped}


def test_an_interrupted_rerun_recomputes_the_offline_gates_so_no_stale_pass_survives(tmp_path, monkeypatch):
    """A Ctrl-C during a Run 1 rerun still runs the offline gates (they only read files): an earlier session's T6
    PASS (and T1-T3) never renders under ABORTED."""
    _bench_env(monkeypatch, tmp_path)
    src = _source_file(tmp_path / "test.jsonl", 4)
    out = tmp_path / "out"
    out.mkdir()
    stale = {"pass": True, "capable": False, "reason": "unsupported_think_format"}
    (out / "summary.json").write_text(json.dumps({"runs": {"run1": {"rc": 0}}, "gates": {"t6": stale, "t3": {"tiel": {"pass": True}}},
                                                  "errors": {}}))
    monkeypatch.setattr(v03t, "_calibrate", lambda argv: (130, ""))
    doc = tmp_path / "doc.md"
    rc = v03t.main(["--out", str(out), "--data", str(src), "--doc", str(doc), "--parts", "run1", "--min-free-gb", "0"])
    s = json.loads((out / "summary.json").read_text())
    assert rc == 3 and s["aborted"] == "interrupted (KeyboardInterrupt)"
    assert s["gates"]["t6"]["pass"] is None and s["gates"]["t3"]["tiel"]["pass"] is None
    assert _row(doc.read_text(), "T6").endswith("| ERROR |") and _row(doc.read_text(), "T3").endswith("| ERROR |")


# ---- T1's MoE-hybrid amendment, T4's image request, T9 and T10 offline, the served closes --------------------------

def _gguf(path: Path, arch: str, *, experts: int | None = None, ssm: bool = False) -> Path:
    """A header-only GGUF with the keys the T1 amendment reads (`<arch>.expert_count`, `<arch>.ssm.*`)."""
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), arch)
    if experts is not None:
        w.add_uint32(f"{arch}.expert_count", experts)
    if ssm:
        w.add_uint32(f"{arch}.ssm.conv_kernel", 4)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_t1_judges_a_moe_hybrid_on_identity_and_argmax_from_its_gguf_metadata_and_says_the_original_verdict(tmp_path,
                                                                                                         monkeypatch):
    """The post-hoc amendment (T1_AMENDED): the class comes from the GGUF header, never the name; on a MoE
    hybrid needs identity, argmax and median |Δp| <= 1e-2 (the 1e-3 outcome still said); a dense-FFN hybrid keeps the
    original criterion."""
    moe = v03t.gguf_traits(_gguf(tmp_path / "moe.gguf", "qwen35moe", experts=256, ssm=True))
    assert moe == {"arch": "qwen35moe", "expert_count": 256, "moe": True, "hybrid": True}
    dense_ffn = v03t.gguf_traits(_gguf(tmp_path / "dense.gguf", "qwen35", ssm=True))
    assert dense_ffn == {"arch": "qwen35", "expert_count": 0, "moe": False, "hybrid": True}
    attn_moe = v03t.gguf_traits(_gguf(tmp_path / "attn.gguf", "qwen3moe", experts=128))
    assert attn_moe["moe"] is True and attn_moe["hybrid"] is False
    with pytest.raises(v03t.InvalidInput, match="not a readable GGUF"):
        v03t.gguf_traits(tmp_path / "missing.gguf")
    size = (tmp_path / "moe.gguf").stat().st_size
    assert v03t.model_traits({"environment": {"model": {"path": str(tmp_path / "moe.gguf"), "bytes": size}}}) == (moe, None)
    # the file at the path must be the one measured: another size, or none recorded, keeps the original criterion
    other = v03t.model_traits({"environment": {"model": {"path": str(tmp_path / "moe.gguf"), "bytes": size + 1}}})
    assert other == (None, f"{tmp_path / 'moe.gguf'} is {size} bytes, the profile measured a GGUF of {size + 1}: not the "
                           f"same file")
    assert v03t.model_traits({"environment": {"model": {"path": str(tmp_path / "moe.gguf")}}})[1].startswith(
        "the profile records no model size")
    assert v03t.model_traits({})[1] == "the profile records no model path (environment.model.path)"
    assert v03t.model_traits({"environment": {"model": {"path": str(tmp_path / "missing.gguf")}}})[0] is None
    # the named model "tiel" with a dense GGUF is judged as dense: the name never decides
    data = write_data(tmp_path / "d.jsonl", 8)
    items, _ = load_data(data)
    raw_path, raw, settings = _raw_file(tmp_path, items)
    profile = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="tiel")
    monkeypatch.setattr(v03t, "T1_MIN_CUTS", 3)
    rec = profile["audit"]["budgets"]["64"]
    off = {**profile, "audit": {**profile["audit"], "budgets": {"64": {**rec, "dp_median": 6.53e-3, "dp_max": 0.183}}}}
    strict = v03t.t1_t2(off, raw_path, data, traits=dense_ffn)["budgets"]["64"]
    assert strict["t1"] is False and strict["criterion"] == "original" and "amended" not in strict
    assert v03t.t1_t2(off, raw_path, data)["budgets"]["64"]["t1"] is False          # unknown class: the original
    amended = v03t.t1_t2(off, raw_path, data, traits=moe)["budgets"]["64"]
    assert amended["t1"] is True and amended["criterion"] == "moe-hybrid (amended post-hoc on 2026-10-08, a deviation from the pre-registered gate)"
    assert amended["amended"] == (
        "PASS (amended post-hoc on 2026-10-08, a deviation from the pre-registered gate; for MoE hybrids, qwen35moe.expert_count 256 with SSM layers: judged on trace "
        "identity 4/4, argmax >= m - 1 and median |Δp| <= 1e-02; original |Δp| criterion 1e-03 FAILS, median 6.53e-03; MoE bound 1e-02 met; "
        "argmax 4/4 meets >= m - 1)")
    assert v03t.t1_t2(off, raw_path, data, traits=attn_moe)["budgets"]["64"]["t1"] is False   # MoE alone: no re-prefill
    assert v03t.t1_t2(off, raw_path, data, traits=moe, dense=True)["budgets"]["64"]["t1"] is None  # dense: descriptive
    wide = {**off, "audit": {**off["audit"], "budgets": {"64": {**rec, "dp_median": 0.05, "argmax_agree": 2}}}}
    bad = v03t.t1_t2(wide, raw_path, data, traits=moe)["budgets"]["64"]
    assert bad["t1"] is False and bad["amended"].startswith("FAIL (amended post-hoc")
    assert "MoE bound 1e-02 exceeded; argmax 2/4 misses >= m - 1)" in bad["amended"]
    # each of the three is required: argmax short with the median within 1e-2 FAILs ...
    noarg = {**off, "audit": {**off["audit"], "budgets": {"64": {**rec, "dp_median": 6.53e-3, "argmax_agree": 2}}}}
    na = v03t.t1_t2(noarg, raw_path, data, traits=moe)["budgets"]["64"]
    assert na["t1"] is False and na["t1_original"] is False and "MoE bound 1e-02 met; argmax 2/4 misses >= m - 1)" in na["amended"]
    # ... and so do traces identical on fewer than n - 2 with the median within 1e-2 and argmax agreeing on them
    lines = raw_path.read_text().splitlines()
    changed, k = [], 0
    for ln in lines:
        r = json.loads(ln)
        if r.get("phase") == "audit" and k < 3:
            r["trace"]["tokens"] = [-1, *r["trace"]["tokens"][1:]] if r["trace"]["tokens"] else [-1]
            k += 1
        changed.append(json.dumps(r))
    split = tmp_path / "split.raw.jsonl"
    split.write_text("\n".join(changed) + "\n")
    one = {**off, "audit": {**off["audit"], "budgets": {"64": {**rec, "dp_median": 6.53e-3, "argmax_agree": 1}}}}
    ni = v03t.t1_t2(one, split, data, traits=moe)["budgets"]["64"]
    assert (ni["identical"], ni["n"], ni["argmax_agree"]) == (1, 4, 1) and ni["t1"] is False
    assert "judged on trace identity 1/4" in ni["amended"] and "MoE bound 1e-02 met" in ni["amended"]
    assert amended["t1_original"] is False                                    # the unamended gate's verdict, kept as data
    # the 1e-2 bound is a hard part of the amended gate: identity and argmax intact, a median above it FAILs
    loose = {**off, "audit": {**off["audit"], "budgets": {"64": {**rec, "dp_median": 0.0101}}}}
    over = v03t.t1_t2(loose, raw_path, data, traits=moe)["budgets"]["64"]
    assert over["t1"] is False and over["amended"].startswith("FAIL (amended post-hoc")
    assert "median 1.01e-02; MoE bound 1e-02 exceeded; argmax 4/4 meets >= m - 1)" in over["amended"]
    edge = {**off, "audit": {**off["audit"], "budgets": {"64": {**rec, "dp_median": 0.01}}}}
    assert v03t.t1_t2(edge, raw_path, data, traits=moe)["budgets"]["64"]["t1"] is True      # <= 1e-2 passes
    clean = v03t.t1_t2(profile, raw_path, data, traits=moe)["budgets"]["64"]
    assert clean["t1"] is True and "original |Δp| criterion 1e-03 met, median 0.00e+00" in clean["amended"]
    # the report says the verdict in full on the pair's line, and the gate's description names the amendment
    t1, _ = v03t.t1_t2_gates({"qwen9b": _res({"256": _entry(True), "512": _entry(True)}),
                              "tiel": _res({"256": {**_entry(True), "dp_median": 6.53e-3, "amended": amended["amended"]}})})
    row = _row(render({"gates": {"t1": t1}}), "T1")
    assert t1["required"]["tiel@256"] == "PASS (amended post-hoc on 2026-10-08, a deviation from the pre-registered gate)" and t1["required"]["qwen9b@256"] == "PASS"
    assert ("tiel@256: identical 40/40, argmax 40/40, |Δp| median 6.53e-03 max 0.00e+00: PASS (amended post-hoc "
            "on 2026-10-08, a deviation from the pre-registered gate; for MoE hybrids") in row and \
        "tiel@256 PASS (amended post-hoc on 2026-10-08, a deviation from the pre-registered gate)" in row and \
        row.endswith("| PASS |") and "Amended post-hoc on 2026-10-08 (a deviation from the pre-registered gate)" in row


def test_t4s_image_request_carries_its_image_and_a_reply_that_shows_it_did_not_is_error_never_a_verdict():
    """The first live run sent the image inside `state` as bare content parts, which parse_request never reads as
    media: a 200, judged a FAIL. The request now uses the top-level `images` field, and `image_case` accepts a 422 only
    for the media refusal and a 200 only when the image was read; anything else is the bench's ERROR."""
    base = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": "x", "b": "y"}}}}
    req = parse_request(v03t.image_request(base))
    assert len(req.media) == 1 and req.media[0].kind == "image" and req.think_budget == 256
    old = {**base, "state": [{"type": "text", "text": "s"}, {"type": "image_url", "image_url": {"url": v03t.PNG_1x1}}],
           "think_budget": 256}
    assert not parse_request(old).media                                       # the first run's shape carried no image
    refusal = {"detail": "deliberate readouts are text-only in this version"}
    assert v03t.image_case(422, {}, refusal) == (422, None)
    status, why = v03t.image_case(422, {}, {"detail": "think_budget must be an integer"})
    assert status is None and why.startswith("a 422 for another reason ('think_budget must be an integer')")
    as_text = {"x-callo-timings": json.dumps([{"q": "q", "kind": "text", "ms": 3.0}])}
    assert v03t.image_case(200, as_text, {})[0] is None and "carried no image" in v03t.image_case(200, as_text, {})[1]
    read = {"x-callo-timings": json.dumps([{"q": "q", "kind": "image", "ms": 3.0}])}
    assert v03t.image_case(200, read, {}) == (200, None)                       # served a media request: the feature's FAIL
    assert v03t.image_case(501, {}, {"detail": "no vision"})[0] is None and v03t.image_case(-1, {}, None)[0] is None
    cases = dict(v03t.T4_EXPECTED, image_with_budget=None)
    g = v03t.t4(cases, disconnect_clear=True, self_test=True, timings_ok=True,
                case_errors={"image_with_budget": "qwen9b-small: a 200 read as ['text']: the request carried no image"})
    assert g["pass"] is None and g["error"] == ("incomplete: not testable: image_with_budget (qwen9b-small: a 200 read as "
                                                "['text']: the request carried no image)")
    assert _row(render({"gates": {"t4": g}}), "T4").endswith("| ERROR |")
    assert v03t.t4(dict(v03t.T4_EXPECTED, image_with_budget=200), disconnect_clear=True, self_test=True,
                   timings_ok=True)["pass"] is False


def test_the_serve_session_launches_as_its_profile_was_measured_and_tries_the_image_only_with_a_projector(tmp_path,
                                                                                                         monkeypatch):
    """No `server_args differs` / `mmproj differs` warning: the profile's llama-server args after `--`, its projector
    (none in Run 1's menu, so no image case there)."""
    data = _source_file(tmp_path / "d.jsonl", 2)
    items, _ = load_data(data)
    launches: list = []
    monkeypatch.setattr(v03t, "_launch", lambda name, out, *, model, flags, ctx: launches.append(list(flags)) or _StubCallo(flags, 1))
    monkeypatch.setattr(v03t, "MODELS", {"tiel": {"model": "/t.gguf"}})
    monkeypatch.setattr(v03t, "load_profile", lambda p: json.loads(Path(p).read_text()))   # a sketch, not a full profile
    monkeypatch.setattr(v03t, "_disconnect_clears", lambda c, body: True)
    sent: list = []
    monkeypatch.setattr(v03t, "_post_full", lambda url, body: sent.append(body) or (
        422, {}, {"detail": "deliberate readouts are text-only in this version"}))
    monkeypatch.setattr(_StubCallo, "stats", lambda self: {"instance": self.instance, "self_test": {"think": True}, "think": {}},
                        raising=False)
    timing = {"x-callo-timings": json.dumps([{"q": items[0].uid, "ms": 1500.0, "gen_ms": 1400.0, "think_budget": 64,
                                              "think_tokens": 40, "think_close": "natural", "read_prompt_n": 2}])}
    monkeypatch.setattr(v03t, "_post", lambda url, body: (200, timing) if body.get("think_budget") == 64 else (
        200, {"x-callo-timings": json.dumps([{"q": items[0].uid, "ms": 1.0}])}))
    monkeypatch.setattr(v03t, "_status", lambda url, body: 422)
    rows = [{"q": it.uid, "b": {"64": [True, 0.5, 0.6, 1400.0]}} for it in items]
    prof = {**MIN_PROFILE, "budgets": {"0": {}, "64": {}}, "pick": {"budget": 64}, "rows": rows,
            "provenance": {"server_args": ["--cache-ram", "0"]}, "environment": {"server": {}, "model": {"mmproj_path": None}}}
    path = tmp_path / "tiel.think.json"
    path.write_text(json.dumps(prof))
    r = v03t.serve_session("tiel", tmp_path / "serve", path, data)
    assert launches == [["--think-profile", str(path), "--think-budget", "64", "--", "--cache-ram", "0"]]
    assert "image_with_budget" not in r["cases"] and sent == [] and set(r["served_ms"].values()) == {1500.0}
    prof["environment"]["model"]["mmproj_path"] = "/t-mmproj.gguf"            # a profile measured with a projector
    path.write_text(json.dumps(prof))
    r = v03t.serve_session("tiel", tmp_path / "serve", path, data)
    assert launches[-1] == ["--think-profile", str(path), "--think-budget", "64", "--mmproj", "/t-mmproj.gguf", "--",
                            "--cache-ram", "0"]
    assert r["cases"]["image_with_budget"] == 422 and sent[0]["images"] == [v03t.PNG_1x1]


def test_t9_is_recomputed_offline_from_the_run_json_files_as_they_are_now(tmp_path):
    """The first live run judged T9 inside the serve part, before Run 2 finished: minicpm stayed `no raw file` (ERROR)
    for good. `_offline_gates` recomputes it from the current run.json files and the kept served records; a record
    from before they were kept carries the serve part's own served verdict, with a note."""
    out = tmp_path / "out"
    _run_dir(out, "run1", {m: ("done", None, str(out / "run1" / f"{m}.think.json")) for m in ("tiel", "qwen9b")})
    _run_dir(out, "run2", {"minicpm": ("running", None, None)})
    for run, m in (("run1", "tiel"), ("run1", "qwen9b"), ("run2", "minicpm")):
        _t9_raw(out / run / f"{m}.raw.jsonl", 2)
    natural = [{"q": f"q{i}", "think_close": "natural", "read_prompt_n": 2} for i in range(5)]
    s: dict = {"errors": {}, "gates": {}, "runs": {},
               "serve": {"tiel": {"served_readouts": natural, "stats_think": {"text_fallback": 0}},
                         "qwen9b": {"budget": 1024}}}                      # qwen9b: a record from before they were kept
    s["gates"]["t9"] = {"calibrate": {"minicpm": {"error": "no raw file (expected for T9's calibrate half)", "pass": None}},
                        "served": {"qwen9b": {"n": 28, "at_n_gen_ok": 28, "text_fallback": 0, "source": "X", "pass": True}},
                        "pass": None}
    data = write_data(tmp_path / "d.jsonl", 2)
    v03t._offline_gates(s, out, data, data, lambda run, names: {})
    g = s["gates"]["t9"]
    assert g["calibrate"]["minicpm"]["error"].startswith("no raw file") and g["pass"] is None   # still running: ERROR
    _run_dir(out, "run2", {"minicpm": ("done", None, str(out / "run2" / "minicpm.think.json"))})   # it finished later
    _t9_raw(out / "run2" / "minicpm.raw.jsonl", 2)
    v03t._offline_gates(s, out, data, data, lambda run, names: {})
    g = s["gates"]["t9"]
    assert g["pass"] is True and g["calibrate"]["minicpm"]["pass"] is True and g["calibrate"]["minicpm"]["n"] == 6
    assert g["served"]["tiel"]["n"] == 5 and "note" not in g["served"]["tiel"]
    assert g["served"]["qwen9b"]["n"] == 28 and g["served"]["qwen9b"]["note"] == v03t.CARRIED
    assert "qwen9b 28/28 (the serve part's own verdict, carried" in _row(render({"gates": {"t9": g}}), "T9")
    del s["serve"]["qwen9b"]                                                   # no record at all: that half's ERROR
    v03t._offline_gates(s, out, data, data, lambda run, names: {})
    assert s["gates"]["t9"]["served"]["qwen9b"]["error"].startswith("no serve-path session") and s["gates"]["t9"]["pass"] is None
    fresh: dict = {"errors": {}, "gates": {}, "runs": {}}                    # no serve part ever judged it: NOT RUN
    v03t._offline_gates(fresh, out, data, data, lambda run, names: {})
    assert "t9" not in fresh["gates"]


def test_offline_gates_read_the_model_class_from_the_gguf_at_the_profiles_path(tmp_path, monkeypatch):
    """The wiring: `_offline_gates` hands `t1_t2` the traits of the GGUF the profile names (a header-only MoE hybrid
    here), and a profile whose GGUF is gone keeps the original criterion with the "model class unknown" note."""
    data = write_data(tmp_path / "d.jsonl", 8)
    items, _ = load_data(data)
    out = tmp_path / "out"
    (out / "run1").mkdir(parents=True)
    raw_path, raw, settings = _raw_file(tmp_path, items)
    for m in ("tiel", "qwen9b"):
        (out / "run1" / f"{m}.raw.jsonl").write_text(raw_path.read_text())
    gg = _gguf(tmp_path / "moe.gguf", "qwen35moe", experts=256, ssm=True)
    base = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    rec = base["audit"]["budgets"]["64"]
    wide = {**base, "audit": {**base["audit"], "budgets": {"64": {**rec, "dp_median": 6.53e-3}}}}

    def at(path, size):
        return {**wide, "environment": {**(wide.get("environment") or {}), "model": {"path": str(path), "bytes": size}}}

    profiles = {"tiel": at(gg, gg.stat().st_size), "qwen9b": at(tmp_path / "gone.gguf", 1)}
    monkeypatch.setattr(v03t, "T1_MIN_CUTS", 3)
    s: dict = {"errors": {}, "gates": {}, "runs": {}}
    v03t._offline_gates(s, out, data, data, lambda run, names: {m: p for m, p in profiles.items() if m in names}
                        if run == "run1" else {})
    models = s["gates"]["t1"]["models"]
    tiel, qwen = models["tiel"]["budgets"]["64"], models["qwen9b"]["budgets"]["64"]
    assert tiel["criterion"] == "moe-hybrid (amended post-hoc on 2026-10-08, a deviation from the pre-registered gate)" and tiel["t1"] is True
    assert tiel["t1_original"] is False and tiel["amended"].startswith("PASS (amended post-hoc on 2026-10-08, a deviation")
    assert qwen["criterion"] == "original" and qwen["t1"] is False and "amended" not in qwen
    assert any(n.startswith(f"model class unknown ({tmp_path / 'gone.gguf'}: cannot read it") for n in models["qwen9b"]["notes"])
    assert "notes" not in models["tiel"] or not any("model class unknown" in n for n in models["tiel"]["notes"])


def test_t10_is_recomputed_offline_from_the_kept_served_ms(tmp_path):
    """The serve record keeps `served_ms` (X-Callo-Timings' `ms` alone), its budget and why; `_offline_gates`
    recomputes T10 against the Run 1 profile as it is now. A record from before served_ms was kept keeps the serve
    part's own entry; a profile no longer done is that model's ERROR."""
    rows = [{"q": f"q{i}", "b": {"256": [True, 0.5, 0.6, 1000.0 + 10 * i, 256, "forced"]}} for i in range(48)]
    profile = {"row_fields": ["correct", "nll1", "pmax", "lat_ms", "k", "close"], "rows": rows}
    stale = {"n": 48, "budget": 256, "served_p50_ms": 2470.0, "profile_p50_ms": 1235.0, "rel_err": 1.0, "pass": False}
    s: dict = {"errors": {}, "gates": {"t10": {"tiel": stale, "qwen9b": stale}}, "runs": {},
               "serve": {"tiel": {"budget": 256, "budget_note": "the profile's pick", "t10_stopped": None,
                                  "served_ms": {f"q{i}": 1100.0 + 10 * i for i in range(48)}},
                         "qwen9b": {"budget": 256}}}                          # qwen9b: from before served_ms was kept
    t10 = v03t.t10_now(s, {"tiel": profile})
    assert t10["tiel"]["pass"] is True and t10["tiel"]["served_p50_ms"] == 1335.0 and t10["tiel"]["budget_note"] == "the profile's pick"
    assert t10["qwen9b"]["error"] == "no serve-path session result (expected for T10)"   # nothing kept, no previous
    t10 = v03t.t10_now(s, {"tiel": profile}, previous=s["gates"]["t10"])
    assert t10["qwen9b"] is stale and t10["tiel"]["pass"] is True
    gone = v03t.t10_now(s, {}, previous=s["gates"]["t10"])["tiel"]
    assert gone["pass"] is None and gone["error"].startswith("no Run 1 profile its run.json marks done")
    stopped = {**s, "serve": {"tiel": {**s["serve"]["tiel"], "t10_stopped": "too many non-200 replies: 48"}}}
    assert v03t.t10_now(stopped, {"tiel": profile})["tiel"]["error"] == "too many non-200 replies: 48"
    data = write_data(tmp_path / "d.jsonl", 2)
    v03t._offline_gates(s, tmp_path / "out", data, data, lambda run, names: {"tiel": profile} if run == "run1" else {})
    assert s["gates"]["t10"]["tiel"]["pass"] is True and s["gates"]["t10"]["qwen9b"] is stale


def test_t9_counts_a_served_budget_close_as_a_cut_like_forced():
    """serve's cut close is `budget` (it was `forced`): the served half still reads only the natural closes."""
    served = {"tiel": {"think": {"text_fallback": 0},
                       "readouts": [*({"q": f"q{i}", "think_close": "natural", "read_prompt_n": 2} for i in range(30)),
                                    *({"q": f"b{i}", "think_close": "budget", "read_prompt_n": 259} for i in range(18))]}}
    g = _t9({}, served)
    assert g["served"]["tiel"]["n"] == 30 and g["served"]["tiel"]["pass"] is True


def test_t9_served_half_reads_only_natural_and_eos_closes():
    """The allow-list: `deadline` and `budget0` closes (the latter carries no read_prompt_n) are cuts, not natural."""
    served = {"tiel": {"think": {"text_fallback": 0},
                       "readouts": [*({"q": f"q{i}", "think_close": "natural", "read_prompt_n": 2} for i in range(20)),
                                    *({"q": f"e{i}", "think_close": "eos", "read_prompt_n": 3} for i in range(10)),
                                    *({"q": f"d{i}", "think_close": "deadline", "read_prompt_n": 300} for i in range(5)),
                                    *({"q": f"z{i}", "think_close": "budget0"} for i in range(5))]}}
    g = _t9({}, served)
    assert g["served"]["tiel"]["n"] == 30 and g["served"]["tiel"]["pass"] is True
