"""Pick preferences: `Point`'s kinds and five statistics, the profile's new per-budget fields and their
recomputation from an older profile's rows, and compare's deadline and early-exit points, rows and pairing set."""
import copy
import dataclasses
import json

import pytest
from test_calibrate_think import SAMPLING, STATS, synthetic_raw, write_data
from test_compare import profile as cprofile

from callo.calibrate_think import build_profile, load_data
from callo.compare import correct_at, joint, load_profile, pairing, points_of, rows_at
from callo.errors import InvalidInput
from callo.think import (
    LATENCY_AXES,
    Point,
    ee_key,
    latency_stats,
    merge_identical,
    point_latencies,
    row_latencies,
)

STATS5 = {"lat_mean_ms": [1500.0, 1400.0, 1600.0], "lat_p50_ms": [1200.0, 1100.0, 1300.0],
          "lat_p95_ms": [3000.0, 2800.0, 3200.0], "lat_p99_ms": [5000.0, 4000.0, 6000.0], "lat_max_ms": 7000.0}


def point(kind="budget", value=None, budget=None, name="tiel", acc=0.9, **kw):
    return Point(name=name, kind=kind, value=value, budget=budget, acc=acc, n=376, **point_latencies(STATS5), **kw)


def test_a_point_is_built_with_keywords_only():
    with pytest.raises(TypeError):
        Point("tiel", 0.9, 376, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 1.0)  # type: ignore[misc]


def test_labels_values_and_kinds():
    b = point(budget=512)
    assert (b.kind, b.value, b.budget, b.label) == ("budget", 512, 512, "tiel@512")
    assert point(budget=256, budgets=(256, 512)).label == "tiel@256≡512"
    d = point(kind="deadline", value=2000, cap=1024)
    assert (d.label, d.budget, d.cap) == ("tiel@2000ms", None, 1024)
    e = point(kind="ee", value=0.95)
    assert (e.label, e.budget) == ("tiel@ee.95", None) and ee_key(0.95) == "0.95" and ee_key(0.9) == "0.9"
    for bad in ({"kind": "budget"}, {"kind": "deadline", "value": 2000, "budget": 0}, {"kind": "ee"},
                {"kind": "early", "value": 1}):
        with pytest.raises(ValueError):
            point(**bad)


def test_the_five_statistics_the_pick_axes_and_the_tail_counts():
    p = point(budget=512)
    assert LATENCY_AXES == ("mean", "p50", "p95")
    assert [p.stat(s) for s in ("mean", "p50", "p95", "p99", "max")] == [1500.0, 1200.0, 3000.0, 5000.0, 7000.0]
    assert p.latency("mean") == (1500.0, 1400.0, 1600.0) and p.latency() == p.latency("mean")
    for axis in ("p99", "max"):                                           # limits only, never an axis
        with pytest.raises(ValueError):
            p.latency(axis)
    with pytest.raises(ValueError):
        p.stat("p90")
    assert p.stats() == STATS5 and p.tail_n == {"p50": 189, "p95": 19, "p99": 4, "max": 1}


def test_merge_identical_merges_budget_points_only():
    pts = [point(budget=256), point(budget=512), point(kind="deadline", value=2000, cap=512),
           point(kind="ee", value=0.95)]
    merged = merge_identical(pts, same=lambda x, y: True)                # every row identical: still never a deadline
    assert [p.label for p in merged] == ["tiel@256≡512", "tiel@2000ms", "tiel@ee.95"]
    assert merged[0].budget == 256 and merged[0].value == 256


def _built(tmp_path, **raw_kw):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items, **raw_kw)
    return items, build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")


def test_calibrate_writes_the_five_statistics_per_budget(tmp_path):
    _, prof = _built(tmp_path)
    for b, e in prof["budgets"].items():
        lats, clusters, strata = row_latencies(prof, "b", b)
        want = latency_stats(lats, clusters, seed=0, strata=strata)
        assert {k: e[k] for k in want} == want                          # all five, from the rows calibrate wrote
        assert e["lat_max_ms"] == max(r["b"][b][3] for r in prof["rows"] if b in r["b"])


def _strip(prof: dict) -> dict:
    old = copy.deepcopy(prof)
    for e in old["budgets"].values():
        for k in ("lat_mean_ms", "lat_p99_ms", "lat_max_ms"):
            del e[k]
    return old


def _sourced(path):
    lines = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        rec["_meta"]["source"] = f"s{int(rec['_meta']['group_id'][1:]) % 2}"
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("sourced", [False, True])
def test_an_older_profile_loads_and_its_statistics_recompute_bit_for_bit(tmp_path, sourced):
    data = write_data(tmp_path / "d.jsonl", 8)
    items, _ = load_data(_sourced(data) if sourced else data)
    raw, settings = synthetic_raw(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["comparable"]["data"]["strata"] == ("source" if sourced else None)
    path = tmp_path / "old.think.json"
    path.write_text(json.dumps(_strip(prof)))
    old = load_profile(path)                                             # an older profile still loads
    stored = {p.label: p for p in points_of(prof, "fake")}
    for p in points_of(old, "fake"):
        assert p.stats() == stored[p.label].stats()                      # recomputed == stored, intervals included


def _with_deadline(prof: dict, d: str = "1500", budget: str = "64") -> dict:
    """A deadline point as calibrate writes it: `deadlines[D]` and `rows[].d[D]` (here budget 64's rows)."""
    out = copy.deepcopy(prof)
    for r in out["rows"]:
        if budget in r["b"]:
            r["d"] = {d: list(r["b"][budget])}
            if r["d"][d][5] == "forced":                     # a deadline row's cut closes `deadline`, never `forced`
                r["d"][d][5] = "deadline"
    lats, clusters, strata = row_latencies(out, "d", d)
    e = out["budgets"][budget]
    out["deadlines"] = {d: {"cap": 256, "reserve_ms": 51.0, "n": e["n"], "acc": e["acc"], "T": e["T"],
                            "think_mean": e["think_mean"], **latency_stats(lats, clusters, seed=0, strata=strata)}}
    return out


def test_a_deadline_points_statistics_recompute_from_its_rows_too(tmp_path):
    _, prof = _built(tmp_path)
    full = _with_deadline(prof)
    old = copy.deepcopy(full)
    for k in ("lat_mean_ms", "lat_p99_ms", "lat_max_ms"):
        del old["deadlines"]["1500"][k]
    path = tmp_path / "old.think.json"
    path.write_text(json.dumps(old))
    by = {p.label: p for p in points_of(load_profile(path), "fake")}
    want = {p.label: p for p in points_of(full, "fake")}
    assert by["fake@1500ms"].stats() == want["fake@1500ms"].stats()
    assert (by["fake@1500ms"].kind, by["fake@1500ms"].value, by["fake@1500ms"].cap) == ("deadline", 1500, 256)


def _with_ee(prof: dict, z: str = "0.95") -> dict:
    out = copy.deepcopy(prof)
    for i, r in enumerate(out["rows"]):
        r["e"] = {z: [i % 2 == 0, 700.0 + 10 * i]}
    lats, clusters, strata = row_latencies(out, "e", z)
    out["ee"] = {z: {"label": f"fake@ee{z[1:]}", "acc": 0.5, "n": len(lats), "think_mean": 80.0,
                     **latency_stats(lats, clusters, seed=0, strata=strata)}}
    return out


def test_compare_builds_deadline_and_early_exit_points_and_reads_their_rows(tmp_path):
    _, prof = _built(tmp_path)
    both = _with_ee(_with_deadline(prof))
    pts = points_of(both, "fake")
    assert [p.label for p in pts] == ["fake@0", "fake@64", "fake@256", "fake@1500ms", "fake@ee.95"]
    assert [p.kind for p in pts[-2:]] == ["deadline", "ee"] and pts[-1].value == 0.95 and pts[-1].acc == 0.5
    rows = rows_at(both)
    first = rows[both["rows"][0]["q"]]
    assert first[("deadline", 1500)] == first[("budget", 64)] and first[("ee", 0.95)] == (1.0, 700.0)
    assert first[("budget", 64)] == (1.0, both["rows"][0]["b"]["64"][3])          # (correct, lat_ms)
    assert correct_at(both)[both["rows"][0]["q"]] == {0: 1.0, 64: 1.0, 256: 1.0}   # the bench's budget-only view
    j = joint([both])                                                    # they join the frontier and the pairs
    assert {"fake@1500ms", "fake@ee.95"} <= {p.label for p in j.ranked} and len(j.fixed) == 8


def test_the_pairing_set_drops_a_question_missing_a_deadline_or_an_early_exit_row(tmp_path):
    _, prof = _built(tmp_path)
    both = _with_ee(_with_deadline(prof))
    assert len(pairing([both])[0]) == 8
    gap = copy.deepcopy(both)
    del gap["rows"][2]["d"]
    fixed = pairing([gap])[0]
    assert len(fixed) == 7 and gap["rows"][2]["q"] not in fixed
    gap = copy.deepcopy(both)
    del gap["rows"][5]["e"]["0.95"]
    assert len(pairing([gap])[0]) == 7


A = cprofile("alpha", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0})


@pytest.mark.parametrize("edit, says", [
    (lambda p: p["budgets"]["0"].update(lat_mean_ms=[1.0, "x", 2.0]),
     "budgets.0.lat_mean_ms is not three finite numbers"),
    (lambda p: p["budgets"]["256"].update(lat_p99_ms=[1.0, 2.0]), "budgets.256.lat_p99_ms is not three finite numbers"),
    (lambda p: p["budgets"]["0"].update(lat_max_ms=float("nan")), "budgets.0.lat_max_ms is not a finite number"),
    (lambda p: p.update(row_fields=["correct", "nll1", "pmax"]),
     "budgets.0: lat_mean_ms absent and rows carry no lat_ms"),
    (lambda p: p["rows"][4]["b"]["256"].__setitem__(3, None),         # a declared lat_ms is checked per row first
     "row 4 (rec-4/q): budget 256 lat_ms is not a finite number >= 0"),
    (lambda p: p.update(deadlines=[]), "deadlines is not a JSON object"),
    (lambda p: p.update(deadlines={"0": {}}), "deadlines key '0' is not an integer from 1"),
    (lambda p: p.update(deadlines={"2000": {"acc": 0.5}}), "deadlines.2000.lat_p50_ms is not three finite numbers"),
    (lambda p: p.update(ee={".95": {}}), "ee key '.95' is not a z in (0, 1)"),
    (lambda p: p["rows"][0].update(d={"x": []}), "row 0 (rec-0/q): deadline 'x' is not an integer from 1"),
    (lambda p: p["rows"][0].update(e={"0.95": [True]}), "row 0 (rec-0/q): early-exit rule 0.95 is not [correct, lat_ms]"),
])
def test_compare_refuses_malformed_new_fields_in_one_line(tmp_path, edit, says):
    doc = copy.deepcopy(A)
    edit(doc)
    path = tmp_path / "bad.think.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(InvalidInput, match="malformed think profile") as e:
        load_profile(path)
    assert says in str(e.value)


def test_a_deadline_entry_needs_an_integer_cap(tmp_path):
    _, prof = _built(tmp_path)
    bad = _with_deadline(prof)
    bad["deadlines"]["1500"]["cap"] = 1.5
    path = tmp_path / "bad.think.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(InvalidInput, match=r"deadlines\.1500\.cap is not an integer >= 1"):
        load_profile(path)
    good = tmp_path / "good.think.json"
    good.write_text(json.dumps(_with_ee(_with_deadline(prof))))
    assert load_profile(good)["deadlines"]["1500"]["cap"] == 256


def test_calibrate_points_are_keyed_on_kind_and_value(tmp_path):
    _, prof = _built(tmp_path, cut_items=0)                              # 64 and 256 merge
    assert prof["budgets"]["64"]["point"] == prof["budgets"]["256"]["point"] == "fake@64≡256"
    p = dataclasses.replace(points_of(prof, "fake")[1])
    assert (p.kind, p.value, p.budget, p.budgets) == ("budget", 64, 64, (64, 256))


def test_a_budget_points_value_is_always_its_budget():
    import dataclasses
    b = point(budget=256)
    r = b.replace(budget=512)
    assert (r.budget, r.value, r.label) == (512, 512, "tiel@512")
    with pytest.raises(ValueError, match="value is its budget"):     # never silently stale
        dataclasses.replace(b, budget=512)
    assert point(budget=256, value=256).value == 256
    with pytest.raises(ValueError, match="value is its budget"):
        point(budget=256, value=512)
    d = point(kind="deadline", value=2000, cap=1024).replace(value=3000)
    assert (d.value, d.budget) == (3000, None)
    assert point(kind="ee", value=0.95).replace(value=0.9).value == 0.9
