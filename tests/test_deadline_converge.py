"""Calibrate's convergence measurement: the probe's
`converge_probe` and `live` modes, the cost pass, the profile's `converge` block, the top-level `ee` block and
`rows[].e`, the early-exit points in the picker (listed, never picked: early exit is opt-in) and in compare, and the
summary's lines, the opt-in line among them."""
import dataclasses
import json

import pytest
from fake_upstream import FakeUpstream
from test_calibrate_think import SAMPLING, STATS, _len, _Served, _settings, synthetic_raw, write_data
from test_think_serve import BODY, _probe

import callo.calibrate_think as ct
import callo.cli
from callo.calibrate_bias import Refusal
from callo.calibrate_think import (
    ThinkSettings,
    _open_raw,
    build_profile,
    calibrate_one,
    converge_lines,
    dry_run_lines,
    load_data,
    probe_cost,
    raw_header,
    read_raw,
    settings_from_args,
    summary_lines,
)
from callo.compare import joint, load_profile, malformed, points_of, report_dict
from callo.compare import serve_line as compare_serve_line
from callo.errors import InvalidInput


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=100).start()
    yield u
    u.stop()


def test_the_probe_reads_a_convergence_probe_every_k_tokens_of_the_bmax_trace(up):
    probe, _, _ = _probe(up)
    code, out, _ = probe.respond({"request": BODY, "budgets": [64, 1024], "seed": 3, "converge_probe": 32}, {})
    q = out["questions"]["q"]
    assert code == 200 and q["trace"]["n_gen"] == 101
    assert [p["k"] for p in q["probes"]] == [32, 64, 96] and all(len(p["probs"]) == 3 for p in q["probes"])
    assert q["probes"][1]["probs"] == q["budgets"]["64"]["probs"]                       # one read serves both
    assert out["readouts"] == 4                                                            # 32, 64, 96 and n_gen


def test_the_live_mode_runs_the_serve_chunk_loop_with_a_forced_probe_at_every_k(up):
    probe, _, _ = _probe(up)
    code, out, _ = probe.respond({"request": BODY, "budgets": [1024], "seed": 3,
                                  "live": {"deadline_ms": None, "probe_every": 32}}, {})
    lv = out["questions"]["q"]["live"]
    assert code == 200 and lv["close"] == "natural" and lv["n_gen"] == 101 and len(lv["tokens"]) == 101
    assert [c["n_predict"] for c in lv["chunks"]] == [32, 32, 32, 32]
    assert [c["prompt_n"] for c in lv["chunks"][1:]] == [1, 1, 1]                        # each continuation restores
    assert [p["k"] for p in lv["probes"]] == [32, 64, 96] and lv["readout"]["k"] == 101 and out["readouts"] == 4
    up.think_len = 10_000
    up.think_real_time, up.think_ms_per_token = True, 2.0
    code, out, _ = probe.respond({"request": BODY, "budgets": [1024], "seed": 3,
                                  "live": {"deadline_ms": 600, "probe_every": None}, "reserve_ms": 30}, {})
    lv = out["questions"]["q"]["live"]
    assert code == 200 and lv["close"] == "deadline" and lv["elapsed_ms"] < 600 + 150 and lv["probes"] == []
    code, out, _ = probe.respond({"request": BODY, "budgets": [1024], "live": {"deadline_ms": 5, "probe_every": None},
                                  "reserve_ms": 30}, {})
    assert out["questions"]["q"]["live"]["close"] == "budget0" and out["readouts"] == 0


@pytest.mark.parametrize("body, why", [({"budgets": [64], "converge_probe": 48}, "converge_probe must be one of"),
                                       ({"budgets": [64, 128], "live": {}}, "exactly one positive budget"),
                                       ({"budgets": [64], "live": {"x": 1}}, "live must be"),
                                       ({"budgets": [64], "live": {"probe_every": 50}}, "probe_every must be"),
                                       ({"budgets": [64], "live": {"deadline_ms": 0}}, "deadline_ms must be"),
                                       ({"budgets": [64], "live": {}, "converge_probe": 32}, "live takes no")])
def test_malformed_convergence_extras_are_422(up, body, why):
    probe, _, _ = _probe(up)
    code, out, _ = probe.respond({"request": BODY, **body}, {})
    assert code == 422 and why in out["detail"]


def _with_probes_and_cost(items, k=32):
    """synthetic_raw's rows plus a probe every k tokens (confident on the label: no flips) and a cost pass."""
    settings = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 64, 256), audit=6, converge_probe=k, converge_cost=2)
    raw, _ = synthetic_raw(items, settings=settings)
    for r in raw.rows:
        if r["phase"] == "B":
            label = next(it.label for it in items if it.uid == r["q"])
            n = len(r["budgets"]["64"]["probs"])
            p = [0.01 / (n - 1)] * n
            p[label] = 0.99
            r["probes"] = [{"k": j, "probs": list(p), "read_ms": 30.0, "read_prompt_n": 4}
                           for j in range(k, r["trace"]["n_gen"], k)]
    for it in items[:2]:
        raw.rows.append({"phase": "cost", "q": it.uid, "slot": 0, "live": {
            "close": "forced", "n_gen": 256,
            "chunks": [{"n": 32, "wall_ms": 332.0 + (12.0 if i else 0.0), "prompt_n": 1 if i else 60} for i in range(8)],
            "probes": [{"k": 32 * (i + 1), "read_ms": 27.0} for i in range(7)]}})
    return raw, settings


def test_the_probe_cost_is_the_read_plus_the_next_chunk_s_excess_and_restores_are_counted(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, _ = _with_probes_and_cost(items)
    c = probe_cost(raw, 10.0)
    assert c["c_ms"] == pytest.approx(27.0 + 24.0) and c["c_steps"] == pytest.approx(5.1)
    assert c["restore_share"] == 1.0 and c["n"] == 14 and c["read_ms"] == 27.0
    assert probe_cost(raw, None)["c_ms"] is None and probe_cost(raw, None)["read_ms"] == 27.0


def test_the_converge_block_fits_rules_points_gains_and_the_rho_map(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    cv = prof["converge"]
    assert cv["K"] == 32 and cv["n"] == 80 and len(cv["cells"]) == 24 and set(cv["sequence"]) == {"0.9", "0.95", "0.99"}
    assert cv["flips"]["changed_after_confident"] == 0 and cv["cost"]["restore_share"] == 1.0
    assert all(c["conv_mono"] is None or c["conv_mono"] <= c["conv"] for c in cv["cells"])    # recorded
    rule = cv["rules"]["0.9"]
    assert rule is not None and rule["conv"] == 1.0 and rule["acc"] >= 0.9 and cv["rules"]["0.99"] is None
    assert cv["rules"]["0.95"] is None                     # 80 stops in 40 two-question clusters: 40 < 59, skipped
    assert set(cv["gain"]) == {"0.9"} and set(cv["gain"]["0.9"]) == {"p50", "p95", "gains"}
    assert set(cv["rho"]["by_rho"]) == {"0.5", "1", "2", "3", "4", "8", "16", "32"}
    assert cv["rho_max_per_token"] == pytest.approx(cv["rho"]["rho_max"] / 32)
    ee = prof["ee"]["0.9"]                                     # beside `deadlines`, five stats
    assert (ee["n"], ee["K"], ee["cell"]) == (80, 32, rule["cell"]) and isinstance(ee["lat_max_ms"], float)
    assert all(len(ee[k]) == 3 for k in ("lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms"))
    e_rows = [r["e"]["0.9"] for r in prof["rows"]]
    assert len(e_rows) == 80 and all(set(r["e"]) == set(prof["ee"]) for r in prof["rows"])
    assert sum(c for c, _ in e_rows) / 80 == pytest.approx(ee["acc"]) and max(x for _, x in e_rows) == ee["lat_max_ms"]
    json.dumps(prof, allow_nan=False)


def test_the_early_exit_points_are_listed_never_picked_and_compare_agrees(tmp_path):
    """Calibrate's own pick sees the EE points (they joined `by_kind` before the fixed row set) and leaves them
    out as opt-in, though .9 is the fastest and most accurate point here; compare reads them back as `kind="ee"` points
    and leaves them out too. A --resume refits through `build_profile`, so it re-picks the same way."""
    from callo.think import EE_OPT_IN
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    pick = prof["pick"]
    assert pick["kind"] == "budget" and pick["excluded"]["m@ee.9"] == EE_OPT_IN and pick["z"] is None
    assert not pick["best"].startswith("m@ee") and (pick["would_pick"], pick["would_best"]) == (None, None)
    assert "--think-converge" not in summary_lines(prof)[-1]
    path = tmp_path / "m.think.json"
    path.write_text(json.dumps(prof))
    loaded = load_profile(path)
    pts = {p.label: p for p in points_of(loaded, "m")}
    assert (pts["m@ee.9"].kind, pts["m@ee.9"].value) == ("ee", 0.9)
    j = joint([loaded])
    assert "m@ee.9" in {p.label for p in j.ranked} and len(j.fixed) == 80 and j.chosen.point.label == pick["label"]
    assert "--think-converge" not in compare_serve_line(j, [str(path)])


def test_a_profile_whose_stored_pick_is_early_exit_is_re_picked_without_it(tmp_path):
    """A profile written while early-exit points could be picked: compare re-picks from its points (a budget point),
    and calibrate's summary of it serves nothing rather than `--think-converge`."""
    from callo.compare import render_md
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    today = prof["pick"]["label"]
    prof["pick"] = {**prof["pick"], "kind": "ee", "label": "m@ee.9", "best": "m@ee.9", "budget": None, "budgets": [],
                    "z": 0.9, "cap": 256, "excluded": {}}
    path = _saved(tmp_path, prof)
    j = joint([load_profile(path)])
    assert j.chosen.point.label == today and j.chosen.point.kind == "budget"
    md = render_md(j, sources=[str(path)])
    serve = next(line for line in md.splitlines() if line.startswith("serve with: "))
    assert "--think-converge" not in serve and "--think-budget" in serve
    assert summary_lines(prof)[-1].startswith("serve with: none (this profile's pick is an early-exit point")


OPT_IN = ("early exit (opt-in, not picked): m@ee.9 saves ~{tok} tokens / ~{ms} ms (mean) at z .9 on this data; it holds "
          "only on questions like these: serve with --think-converge 0.9 to opt in")


def test_the_opt_in_line_names_a_servable_early_exit_point_and_nothing_else(tmp_path):
    """Exactly when the profile has an early-exit point `callo serve` can run, calibrate's summary and compare's report
    (markdown and JSON) print one line naming it with its saving on this data; no such point, no line."""
    from callo.calibrate_think import opt_in_line
    from callo.compare import render_md
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    pts = {p.label: p for p in points_of(prof, "m")}
    ms = pts["m@256"].latency("mean")[0] - pts["m@ee.9"].latency("mean")[0]
    want = OPT_IN.format(tok=f"{prof['converge']['rules']['0.9']['net_saving']:.0f}", ms=f"{ms:.0f}")
    assert ms > 0 and opt_in_line(prof, "m") == want
    summary = summary_lines(prof)
    assert f"  {want}" in summary and summary.index(f"  {want}") > next(
        i for i, line in enumerate(summary) if line.startswith("pick: "))
    path = _saved(tmp_path, prof)
    j = joint([load_profile(path)])
    assert render_md(j, sources=[str(path)]).splitlines().count(want) == 1
    assert report_dict(j)["early_exit"]["m"]["opt_in"] == want
    unserved = {**prof, "converge": {**prof["converge"], "rules": {**prof["converge"]["rules"], "0.9": None}}}
    assert opt_in_line(unserved, "m") is None                                  # no rule reaches .9: not servable
    assert not any(line.startswith("  early exit (opt-in") for line in summary_lines(unserved))
    j = joint([load_profile(_saved(tmp_path, unserved, "u.think.json"))])
    assert "early exit (opt-in" not in render_md(j) and report_dict(j)["early_exit"]["m"]["opt_in"] is None
    plain = {k: v for k, v in prof.items() if k not in ("ee", "converge")}
    assert opt_in_line(plain, "m") is None
    assert not any("opt-in" in line for line in summary_lines(plain))


def test_the_opt_in_line_claims_no_saving_that_is_not_positive(tmp_path):
    """A rule whose net saving is not positive, on a point no faster than the budget at its cap, still names the
    servable early exit and its flag, with no `saves` clause."""
    from callo.calibrate_think import opt_in_line
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    prof["converge"]["rules"]["0.9"]["net_saving"] = -3.0
    prof["ee"]["0.9"]["lat_mean_ms"] = [1e7, 1e7, 1e7]                  # slower than every budget
    assert opt_in_line(prof, "m") == ("early exit (opt-in, not picked): m@ee.9 at z .9 on this data; it holds only on "
                                      "questions like these: serve with --think-converge 0.9 to opt in")
    prof["converge"]["rules"]["0.9"]["net_saving"] = 0.0
    assert " saves " not in opt_in_line(prof, "m")
    prof["converge"]["rules"]["0.9"]["net_saving"] = 12.0                # one positive saving: only that one
    assert "m@ee.9 saves ~12 tokens at z .9" in opt_in_line(prof, "m")


def test_the_tables_mark_early_exit_opt_in_and_a_row_it_dominates_says_so(tmp_path):
    """The pick can be a row an early-exit point dominates (it never takes early exit): calibrate's summary, compare's
    table and the scatter legend mark every early-exit row `opt-in`, and a dominator that is one `(opt-in)`."""
    from callo.compare import scatter_lines, table_lines
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    picked = prof["pick"]["label"]
    summary = summary_lines(prof)
    row = next(line for line in summary if line.startswith(f"{picked} "))
    assert row.endswith("dominated by m@ee.9 (opt-in)")
    assert next(line for line in summary if line.startswith("m@ee.9 ")).endswith("opt-in (early exit)")
    assert any("the pick never takes early exit (opt-in)" in line for line in summary)
    j = joint([load_profile(_saved(tmp_path, prof))])
    rows = {r.split(" | ")[0].removeprefix("| "): r for r in table_lines(j) if r.startswith("| m@")}
    assert rows[picked].endswith("| · by m@ee.9 (opt-in) |") and rows["m@ee.9"].endswith("| F; opt-in (early exit) |")
    legend = " ".join(scatter_lines(list(j.ranked)))
    assert "m@ee.9 (opt-in)" in legend and f"{picked} (opt-in)" not in legend


def test_a_resume_re_picks_a_profile_whose_stored_pick_is_early_exit(tmp_path):
    """calibrate --resume on a complete raw file refits with no server: a profile written while early exit could be
    picked (its stored pick an early-exit point) gets a budget or deadline pick back, and a serve line without
    `--think-converge`."""
    from test_calibrate_think import DEAD
    s = _Served(tmp_path, think_len=_len)
    try:
        settings = _settings(write_data(tmp_path / "d.jsonl", 12), converge_probe=32, converge_cost=2)
        out = calibrate_one(settings, url=s.url, name="fake", out_dir=tmp_path / "think", resume=False, log=s.log.append,
                            therm=lambda: 100)
    finally:
        s.stop()
    assert out.status == "done", out.reason
    prof = json.loads(out.profile.read_text())
    label = next((f"fake@ee{z.removeprefix('0')}" for z in sorted(prof.get("ee") or {})), "fake@ee.9")
    prof["pick"] = {**prof["pick"], "kind": "ee", "label": label, "best": label, "budget": None, "budgets": [],
                    "z": 0.9, "cap": 256}                                   # as an earlier pick wrote it
    out.profile.write_text(json.dumps(prof))
    log = []
    refit = calibrate_one(settings, url=DEAD, name="fake", out_dir=tmp_path / "think", resume=True, log=log.append,
                          therm=lambda: 100)
    assert refit.status == "done", refit.reason
    assert any(line.startswith("refit: ") for line in log)
    pick = json.loads(refit.profile.read_text())["pick"]
    assert pick["kind"] in ("budget", "deadline") and pick["label"] and "@ee" not in pick["label"]
    assert "--think-converge" not in summary_lines(json.loads(refit.profile.read_text()))[-1]


def _rho_lines(rho):
    cv = {"K": 32, "cost": {"c_ms": 20.0, "c_steps": 2.0, "n": 4}, "rules": {}, "gain": {}, "rho": rho}
    return converge_lines({"converge": cv})[1:]


def test_the_break_even_line_says_what_the_rho_map_supports():
    """The line "pays while a probe costs <= X" holds only up to the prefix of grid rhos that all
    gain; every grid rho gaining is "at every probe cost tested, >= 32"; a gap names only the gaining rhos; an older
    profile (no rho_prefix) keeps the rho_max line."""
    every = {f"{r:g}": [0.9] for r in (0.5, 1, 2, 3, 4, 8, 16, 32)}
    assert _rho_lines({"by_rho": every, "rho_max": 32.0, "rho_prefix": 32.0}) == [
        "  break-even at K 32: pays at every probe cost tested, >= 32 decode steps (here 2.0; >= 1.00 steps per "
        "think token)"]
    gap = {**every, "1": [], "16": [], "32": []}
    assert _rho_lines({"by_rho": gap, "rho_max": 8.0, "rho_prefix": 0.5}) == [
        "  break-even at K 32: pays while a probe costs <= 0.5 decode steps (here 2.0; 0.02 steps per think token); "
        "some z also gains at 8"]
    upto = {**every, "16": [], "32": []}
    assert _rho_lines({"by_rho": upto, "rho_max": 8.0, "rho_prefix": 8.0}) == [
        "  break-even at K 32: pays while a probe costs <= 8 decode steps (here 2.0; 0.25 steps per think token)"]
    holes = {f"{r:g}": ([0.9] if r in (2, 4) else []) for r in (0.5, 1, 2, 3, 4, 8, 16, 32)}
    assert _rho_lines({"by_rho": holes, "rho_max": 4.0, "rho_prefix": None}) == [
        "  break-even at K 32: gains only at 2, 4 decode steps, not from the cheapest probe cost up (here 2.0)"]
    assert _rho_lines({"by_rho": upto, "rho_max": 8.0}) == [                       # an older profile: no rho_prefix
        "  break-even at K 32: pays while a probe costs <= 8 decode steps (here 2.0; 0.25 steps per think token)"]
    none = {f"{r:g}": [] for r in (0.5, 1, 2, 3, 4, 8, 16, 32)}
    assert _rho_lines({"by_rho": none, "rho_max": None, "rho_prefix": None}) == [
        "  not on your front at any z for any probe cost down to 0.5 decode steps (here 2.0)"]


def _verdict(gain):
    rule = {"net_saving": 481.0, "acc": 0.90, "dacc": [-0.032, -0.059, -0.011]}
    cv = {"K": 32, "cost": {"c_ms": 20.0, "c_steps": 2.0, "n": 4}, "rules": {"0.9": rule}, "gain": {"0.9": gain},
          "rho": {}}
    (line,) = [x for x in converge_lines({"converge": cv}) if x.startswith("  z .9:")]
    return line.split(" -> ", 1)[1]


def test_the_early_exit_verdict_names_what_each_gain_is_against():
    """The z line's Δ is against full thinking, but a gain is against the budget and deadline points: an "accuracy
    gain" beside a negative Δ read as a bug, so each gain says what it beats."""
    def g(speed, accuracy):
        axis = {"front": True, "speed": speed, "accuracy": accuracy, "gains": speed or accuracy}
        return {"p50": axis, "p95": {**axis, "speed": False, "accuracy": False, "gains": False},
                "gains": speed or accuracy}
    assert _verdict(g(True, False)) == "ON your front (faster than every budget/deadline point as accurate)"
    assert _verdict(g(False, True)) == "ON your front (more accurate than every budget/deadline point as fast)"
    assert _verdict(g(True, True)) == ("ON your front (faster than every budget/deadline point as accurate; "
                                       "more accurate than every budget/deadline point as fast)")
    assert _verdict(g(False, False)) == "not on your front"
    assert "gain)" not in _verdict(g(True, True))


def test_the_early_exit_delta_says_it_is_against_full_thinking():
    """The z line's Δ is against the full-thinking end answer while its verdict is against the budget and deadline
    points; the label says which, so the two cannot be read as one comparison."""
    rule = {"net_saving": 481.0, "acc": 0.90, "dacc": [-0.032, -0.059, -0.011]}
    cv = {"K": 32, "cost": {"c_ms": 20.0, "c_steps": 2.0, "n": 4}, "rules": {"0.9": rule}, "gain": {}, "rho": {}}
    (line,) = [x for x in converge_lines({"converge": cv}) if x.startswith("  z .9:")]
    assert "acc 0.90 (Δ vs full think -3.2pp [-5.9, -1.1])" in line and "(Δ -" not in line


def test_the_summary_prints_the_early_exit_block(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    lines = converge_lines(prof)
    assert lines[0].startswith("early exit (K 32, probe ") and "restore 100 %" in lines[0]
    assert any(line.startswith("  z .9: saves ") for line in lines) and any("z .99: no rule reaches it" in x for x in lines)
    assert any("break-even at K 32" in x or "not on your front at any z" in x for x in lines)
    assert converge_lines({}) == []
    summary = summary_lines(prof)
    assert lines[0] in summary and any(line.startswith("m@ee.9") for line in summary)
    assert "--think-converge" not in summary[-1] and "--think-budget" in summary[-1]


def test_an_early_exit_row_at_the_grid_s_top_gets_the_search_limit_note(tmp_path):
    from callo.calibrate import GRID
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    for block in ("budgets", "deadlines", "ee"):
        for e in (prof.get(block) or {}).values():
            e["T"] = 1.5
    assert not any("search limit" in line for line in summary_lines(prof))
    prof["ee"]["0.9"]["T"] = GRID[-1]
    lines = summary_lines(prof)
    pick = next(i for i, line in enumerate(lines) if line.startswith("pick: "))
    assert lines[pick - 1] == ("T is at the search limit (16) at m@ee.9: the probabilities are still overconfident "
                               "after calibration")


def test_a_full_run_with_converge_probe_writes_the_probes_the_cost_pass_and_the_block(tmp_path):
    s = _Served(tmp_path, think_len=_len)
    try:
        data = write_data(tmp_path / "d.jsonl", 12)
        settings = _settings(data, converge_probe=32, converge_cost=2)
        out = calibrate_one(settings, url=s.url, name="fake", out_dir=tmp_path / "think", resume=False, log=s.log.append,
                            therm=lambda: 100)
        assert out.status == "done", out.reason
        prof = json.loads(out.profile.read_text())
        raw = read_raw(tmp_path / "think" / "fake.raw.jsonl")
    finally:
        s.stop()
    assert prof["comparable"]["converge_probe"] == 32 and prof["comparable"]["converge_cost"] == 2
    long = next(r for r in raw.rows if r["phase"] == "B" and r["trace"]["n_gen"] == 256)
    assert [p["k"] for p in long["probes"]] == list(range(32, 256, 32))
    cost = [r for r in raw.rows if r["phase"] == "cost"]
    assert len(cost) == 2 and all(len(r["live"]["probes"]) == 7 for r in cost)
    assert all(c["prompt_n"] == 1 for r in cost for c in r["live"]["chunks"][1:])
    cv = prof["converge"]
    assert cv["cost"]["c_ms"] is not None and cv["cost"]["restore_share"] == 1.0 and cv["n"] == 12


def test_the_cli_flags_reach_the_settings(tmp_path, capsys):
    data = write_data(tmp_path / "d.jsonl", 2)
    parse = callo.cli.build_parser().parse_args
    s = settings_from_args(parse(["calibrate", "--think-budgets", "--data", str(data), "--converge-probe"]))
    assert (s.converge_probe, s.converge_cost) == (64, 6)
    s = settings_from_args(parse(["calibrate", "--think-budgets", "--data", str(data), "--converge-probe", "32",
                                  "--converge-cost", "0"]))
    assert (s.converge_probe, s.converge_cost) == (32, 0) and s.comparable()["converge_cost"] == 0
    for argv, why in ((["--converge-probe", "50"], "one of 32, 64, 128"), (["--converge-cost", "3"], "needs --converge-probe"),
                      (["--converge-probe", "32", "--converge-cost", "-1"], "at least 0")):
        with pytest.raises(InvalidInput, match=why):
            settings_from_args(parse(["calibrate", "--think-budgets", "--data", str(data), *argv]))
    assert callo.cli.main(["calibrate", "--data", str(data), "--converge-probe", "32"]) == 2
    assert "--converge-probe apply only with --think-budgets" in capsys.readouterr().err


# ---- retries, ignored probe keys, the restore share, the cap, resume settings, the summary, the dry run ------------

def test_a_retried_converge_probe_request_does_not_trip_foreign_s1(tmp_path, monkeypatch):
    """A retried request's lost first attempt still added its readouts to the server's
    count, and a phase-B request with --converge-probe reads every K below n_gen and the cost pass every K below its
    cut plus the final readout, so the guard's slack covers those probe reads too (phase B and the cost pass)."""
    real_read = ct.ProbeClient.read
    seen: dict[str, int] = {}

    def read(self, item, budgets, seed, sampling, extras=None):
        kind = "cost" if extras and "live" in extras else "B" if extras and extras.get("converge_probe") else None
        if kind is not None and kind not in seen and _len(item.body["state"]) == 300:
            first = real_read(self, item, budgets, seed, sampling, extras)    # the dropped attempt the server answered
            seen[kind] = first.readouts
            return dataclasses.replace(real_read(self, item, budgets, seed, sampling, extras), retries=1)
        return real_read(self, item, budgets, seed, sampling) if extras is None else real_read(
            self, item, budgets, seed, sampling, extras)

    monkeypatch.setattr(ct.ProbeClient, "read", read)
    s = _Served(tmp_path, think_len=_len)
    try:
        settings = _settings(write_data(tmp_path / "d.jsonl", 12), converge_probe=32, converge_cost=2)
        out = calibrate_one(settings, url=s.url, name="fake", out_dir=tmp_path / "think", resume=False, log=s.log.append,
                            therm=lambda: 100)
    finally:
        s.stop()
    assert out.status == "done", out.reason
    assert seen["B"] > len(settings.positive) + 1 and seen["cost"] > 2      # more than the old bounds, 3 and 2


@pytest.mark.parametrize("strip, cost_n", [("probes", 0), ("probes", 6), ("live", 2)])
def test_a_server_that_ignores_the_converge_probe_keys_fails_the_run_loudly(tmp_path, monkeypatch, strip, cost_n):
    """An older callo serve drops the keys it does not know, so a multi-option phase-B
    reply without `probes`, or a cost-pass reply without `live`, is a failed run naming the fix, never a run whose
    converge block reports no probe."""
    real_read = ct.ProbeClient.read

    def read(self, item, budgets, seed, sampling, extras=None):
        reply = real_read(self, item, budgets, seed, sampling) if extras is None else real_read(
            self, item, budgets, seed, sampling, extras)
        for q in reply.body["questions"].values():
            q.pop(strip, None)
        return reply

    monkeypatch.setattr(ct.ProbeClient, "read", read)
    s = _Served(tmp_path, think_len=_len)
    try:
        settings = _settings(write_data(tmp_path / "d.jsonl", 8), converge_probe=32, converge_cost=cost_n)
        out = calibrate_one(settings, url=s.url, name="fake", out_dir=tmp_path / "think", resume=False, log=s.log.append,
                            therm=lambda: 100)
    finally:
        s.stop()
    assert out.status == "failed"
    assert "unusable reply: the server does not support --converge-probe probes; upgrade callo serve" in out.reason


def test_the_restore_share_skips_continuations_without_a_prompt_n(tmp_path):
    """The restore share is measured, never assumed: a continuation whose reply carries
    no prompt_n is not counted as restored (nor as re-prefilled)."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, _ = _with_probes_and_cost(items)
    cost = [r for r in raw.rows if r["phase"] == "cost"]
    for c in cost[0]["live"]["chunks"]:
        c.pop("prompt_n")
    assert probe_cost(raw, 10.0)["restore_share"] == 1.0                 # only the second pass's 7 counted
    cost[1]["live"]["chunks"][3]["prompt_n"] = 40
    assert probe_cost(raw, 10.0)["restore_share"] == pytest.approx(6 / 7)
    for c in cost[1]["live"]["chunks"]:
        c["prompt_n"] = None
    assert probe_cost(raw, 10.0)["restore_share"] is None


def test_the_early_exit_points_carry_the_bmax_cap(tmp_path):
    """An early-exit point runs to the Bmax cap, as a deadline point does."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    outs = ct.outcomes(items, raw, settings.budgets)
    cv = ct.convergence_block(items, raw, settings, outs, {}, name="m")
    assert cv.points and all(p.kind == "ee" and p.cap == settings.bmax == 256 for p in cv.points.values())
    # the profile's `ee[z]` keeps the cap, so compare's reloaded point carries it too
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    assert prof["ee"] and all(e["cap"] == 256 for e in prof["ee"].values())
    path = tmp_path / "m.think.json"
    path.write_text(json.dumps(prof))
    loaded = [p for p in points_of(load_profile(path), "m") if p.kind == "ee"]
    assert loaded and all(p.cap == 256 for p in loaded)
    report = report_dict(joint([load_profile(path)]))                # `compare --json` printed "cap": null
    assert [x["cap"] for x in report["points"] if x["kind"] == "ee"] == [256] * len(loaded)
    bad = json.loads(path.read_text())
    bad["ee"]["0.9"]["cap"] = 0
    assert (malformed(bad) or "").startswith("ee.0.9.cap is not an integer >= 1")


def test_a_resume_with_other_convergence_settings_is_refused(tmp_path):
    """--converge-probe and --converge-cost are in the comparable header, so a resume that
    adds, removes or changes either is refused; the same settings resume."""
    base = ThinkSettings("d", "d", "d" * 64, (0, 64))
    on = dataclasses.replace(base, converge_probe=32, converge_cost=6)
    for first, then in ((base, on), (on, base), (on, dataclasses.replace(on, converge_probe=64)),
                        (on, dataclasses.replace(on, converge_cost=2))):
        path = tmp_path / "x.raw.jsonl"
        path.unlink(missing_ok=True)
        f, _ = _open_raw(path, raw_header(STATS, first, SAMPLING), resume=False, log=print)
        f.close()
        f, _ = _open_raw(path, raw_header(STATS, first, SAMPLING), resume=True, log=print)
        f.close()
        with pytest.raises(Refusal, match="comparable differ"):
            _open_raw(path, raw_header(STATS, then, SAMPLING), resume=True, log=print)


def test_the_summary_says_why_the_probe_cost_is_unmeasured_and_the_saving_is_gross(tmp_path):
    """Without a measured cost the head says which reason applies, and the rules (chosen at
    rho 0) save a GROSS number of tokens."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    assert any(" tokens/question net," in x for x in converge_lines(prof))
    cv = prof["converge"]
    no_read = {**prof, "converge": {**cv, "cost": {**cv["cost"], "c_ms": None, "c_steps": None, "read_ms": None, "n": 0}}}
    lines = converge_lines(no_read)
    assert lines[0] == "early exit (K 32, probe cost unmeasured: the cost pass read no probe):"
    assert any("tokens/question gross (cost not measured)" in x for x in lines)
    assert not any(" net," in x for x in lines)
    no_d = {**prof, "converge": {**cv, "decode_ms": None, "cost": {**cv["cost"], "c_ms": None, "c_steps": None}}}
    head = converge_lines(no_d)[0]
    assert head.startswith("early exit (K 32, probe cost unmeasured: ") and "read no probe" not in head
    assert "decode" in head
    skipped = {**no_read, "comparable": {**prof["comparable"], "converge_cost": 0}}
    assert "--converge-cost 0" in converge_lines(skipped)[0]


def test_the_dry_run_counts_the_probe_cost_pass(tmp_path):
    """The dry run's worst case includes the cost pass, up to converge_cost x Bmax."""
    data = write_data(tmp_path / "d.jsonl", 8)
    items, skipped = load_data(data)
    with_cost = dry_run_lines(_settings(data, converge_probe=32, converge_cost=2), items, skipped)
    assert any("2 x 256 = 512 generated tokens" in x and "probe-cost pass" in x for x in with_cost)
    for s in (_settings(data), _settings(data, converge_probe=32, converge_cost=0)):
        assert not any("probe-cost" in x for x in dry_run_lines(s, items, skipped))


def _saved(tmp_path, prof, name="m.think.json"):
    path = tmp_path / name
    path.write_text(json.dumps(prof))
    return path


def test_compare_leaves_out_an_early_exit_point_serve_cannot_run_and_says_so(tmp_path):
    """The profile's convergence block decides whether serve can run an early-exit point; one it
    cannot is excluded from compare's pick with its reason and a line naming it, and no `--think-converge` line is
    printed."""
    from callo.compare import render_md
    from callo.converge import UNSERVED
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    prof["converge"]["rules"]["0.9"] = None                    # no rule reaches .9 any more: serve runs it without
    path = _saved(tmp_path, prof)
    j = joint([load_profile(path)])
    assert j.chosen.point.label != "m@ee.9" and j.chosen.excluded["m@ee.9"].startswith(f"{UNSERVED}: no early-exit")
    md = render_md(j, sources=[str(path)])
    assert ("early exit `callo serve` cannot run as measured: m@ee.9 (no early-exit rule reaches z .9 (serve would "
            "run without early exit))") in md
    assert "--think-converge" not in compare_serve_line(j, [str(path)])
    del prof["converge"]
    j = joint([load_profile(_saved(tmp_path, prof, "n.think.json"))])
    assert "--think-converge refuses this profile" in j.chosen.excluded["m@ee.9"]


def test_calibrate_leaves_out_an_early_exit_point_serve_cannot_run(tmp_path, monkeypatch):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    monkeypatch.setattr(ct.converge, "serve_rules", lambda block: (None, "it has no convergence block"))
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    assert prof["pick"]["kind"] == "budget" and "--think-converge refuses" in prof["pick"]["excluded"]["m@ee.9"]
    summary = summary_lines(prof)
    assert any(line.startswith("  early exit `callo serve` cannot run as measured: m@ee.9") for line in summary)
    assert "--think-converge" not in summary[-1]


def test_an_old_early_exit_entry_without_a_cap_serves_under_the_largest_budget(tmp_path):
    """An `ee` entry from before `cap` was recorded ran under the profile's largest calibrated budget,
    so its point (in compare's table and report.json) carries that cap."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    for e in prof["ee"].values():
        del e["cap"]
    pts = {p.label: p for p in points_of(load_profile(_saved(tmp_path, prof)), "m")}
    assert pts["m@ee.9"].cap == 256
    big = json.loads(json.dumps(prof).replace('"256"', '"8192"').replace("m@256", "m@8192"))   # budget 256 -> 8192
    path = _saved(tmp_path, big, "big.think.json")
    j = joint([load_profile(path)])
    assert {p.label: p.cap for p in j.ranked}["m@ee.9"] == 8192 and j.chosen.point.kind == "budget"
    assert [x["cap"] for x in report_dict(j)["points"] if x["label"] == "m@ee.9"] == [8192]
