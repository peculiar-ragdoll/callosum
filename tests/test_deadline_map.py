"""Think deadlines' static mapping and its reports: the closes, the cut, the static budgets, the
exact and approximate accuracy, a re-simulation at another reserve, old profiles, `callo compare --deadline` with its
"Deadlines" and "Early exit" sections, calibrate's summary block, and the pick-preferences contract
(the profile's deadline and early-exit points on the joint frontier)."""
import json

import pytest
from test_compare import profile

from callo.calibrate_think import DEADLINE_SKIPPED, serve_command, summary_lines
from callo.cli import main
from callo.compare import deadline_mappings, joint, malformed, points_of, report_dict
from callo.deadline import simulate
from callo.deadline_map import mapping, mapping_lines, static_budgets

N_GEN = [40, 40, 40, 40, 1000, 1000, 1000, 1000, 1000, 1000]          # four close naturally at 40, six run to Bmax


def deadline_profile(*, with_point=True, with_terms=True, ee=False):
    """test_compare's profile at budgets 0, 256, 1024 (Bmax), with `rows[].t` (a = 300 ms, d = 10 ms/token), each row's
    Bmax close, and a deadline point at 2000 ms whose rows are the simulation's own cut."""
    p = profile("m", {0: 3, 256: 6, 1024: 9}, {0: 300.0, 256: 3000.0, 1024: 9000.0})
    r = 51.0
    for i, row in enumerate(p["rows"]):
        n_gen = N_GEN[i]
        row["b"]["1024"][4], row["b"]["1024"][5] = min(n_gen, 1024), "natural" if n_gen < 1024 else "forced"
        row["b"]["256"][4], row["b"]["256"][5] = min(n_gen, 256), "natural" if n_gen <= 256 else "forced"
        if with_terms:
            row["t"] = [n_gen, 300.0, 10.0]
        if with_point:
            s = simulate(a_ms=300.0, d_ms=10.0, n_gen=n_gen, close=row["b"]["1024"][5], deadline_ms=2000,
                         reserve_ms=r, cap=1024)
            row["d"] = {"2000": [i % 3 != 0, 0.5, 0.6, 300.0 + 10.0 * s.k + 27.0, s.k, s.close or "budget0"]}
    if with_point:
        p["deadlines"] = {"2000": {"cap": 1024, "reserve_ms": r, "n": 10, "acc": 0.6, "T": 1.1, "think_mean": 120.0,
                                   "lat_p50_ms": [1800.0, 1700.0, 1900.0], "lat_p95_ms": [2000.0, 1950.0, 2050.0],
                                   "closes": {"natural": 0.4, "deadline": 0.6, "budget": 0.0, "eos": 0.0, "budget0": 0.0},
                                   "k_p50": 165}}
    p["latency"] = {"read_forced_ms": {"p50": 26.0, "p95": 27.4, "p99": 28.0}}
    if ee:                                         # `ee` beside `deadlines`, rows[].e
        p["ee"] = {"0.9": {"acc": 0.9, "n": 10, "think_mean": 60.0, "lat_p50_ms": [700.0, 650.0, 750.0],
                           "lat_p95_ms": [900.0, 850.0, 950.0]}}
        for i, row in enumerate(p["rows"]):
            row["e"] = {"0.9": [i < 9, 700.0 + i]}
        p["converge"] = {"K": 32, "n": 10, "cost": {"c_ms": 30.0, "c_steps": 3.0, "restore_share": 1.0},
                         "gain": {"0.9": {"p50": {"front": True, "speed": True, "accuracy": False, "gains": True},
                                          "p95": {"front": True, "speed": False, "accuracy": False, "gains": False},
                                          "gains": True}},
                         "rules": {"0.9": {"net_saving": 40.0, "acc": 0.9, "dacc": [0.0, -0.1, 0.1]}}, "rho": None}
    return p


def test_the_mapping_counts_closes_from_the_simulation_and_the_accuracy_from_the_point_exactly():
    m = mapping(deadline_profile(), 2000)
    assert m["reserve_source"] == "deadline point" and m["reserve_ms"] == 51.0 and m["cap"] == 1024
    assert m["closes"] == {"natural": 0.4, "budget": 0.0, "deadline": 0.6, "budget0": 0.0}
    k = simulate(a_ms=300.0, d_ms=10.0, n_gen=1000, close="forced", deadline_ms=2000, reserve_ms=51.0, cap=1024).k
    assert m["cut_k"] == [k, k, k]
    hand = (4 + sum(1 for i in range(4, 10) if i % 3 != 0)) / 10            # natural: b[1024]; cut: the point's rows
    assert m["accuracy"]["exact"] and m["accuracy"]["acc"] == pytest.approx(hand)
    assert m["accuracy"]["ci"][0] <= hand <= m["accuracy"]["ci"][1]


def test_a_combined_setting_is_first_hit_and_exact_only_for_a_calibrated_budget():
    p = deadline_profile()
    m = mapping(p, 2000, budget=256)                                  # 256 is above every cut: the deadline binds
    assert m["closes"] == {"natural": 0.4, "budget": 0.0, "deadline": 0.6, "budget0": 0.0} and m["cap"] == 256
    m = mapping(p, 2000, budget=64)                                   # 64 binds before the deadline's cut
    assert m["closes"] == {"natural": 0.4, "budget": 0.6, "deadline": 0.0, "budget0": 0.0}
    assert not m["accuracy"]["exact"] and m["accuracy"]["unknown"] == pytest.approx(0.6)
    assert mapping(p, 2000, budget=30)["closes"]["budget"] == 1.0      # below the natural closes too


def test_a_question_without_a_trace_is_natural_at_zero_and_keeps_the_mapping_exact():
    """A single-option question (no trace: `t` null, k 0, no close) is answered at once under any deadline."""
    p = deadline_profile()
    row = p["rows"][0]
    row["t"] = None
    for b in row["b"].values():
        b[4], b[5] = 0, None
    row["d"]["2000"][4:] = [0, None]
    m = mapping(p, 2000)
    assert not m["approx_terms"] and m["accuracy"]["exact"] and m["closes"]["natural"] == 0.4
    again = mapping(p, 2000, reserve_ms=400.0)                       # re-simulated: still natural, never simulated
    assert not again["approx_terms"] and again["closes"]["natural"] == 0.4 and again["n"] == 10
    assert mapping(p, 2000, reserve_ms=1996.0)["closes"] == {"natural": 0.1, "budget": 0.0, "deadline": 0.0,
                                                             "budget0": 0.9}   # D - r < ô: only the question w/o trace


def test_another_reserve_re_simulates_and_is_approximate():
    p = deadline_profile()
    m = mapping(p, 2000, reserve_ms=400.0)
    k = simulate(a_ms=300.0, d_ms=10.0, n_gen=1000, close="forced", deadline_ms=2000, reserve_ms=400.0, cap=1024).k
    assert m["reserve_source"] == "--reserve" and m["cut_k"][1] == k and not m["accuracy"]["exact"]
    assert m["accuracy"]["lo"] == pytest.approx(0.4) and m["accuracy"]["hi"] == pytest.approx(1.0)
    tiny = mapping(p, 300)                                            # a + ô + d + r > D: no token fits
    assert tiny["closes"]["budget0"] == 1.0 and tiny["reserve_source"] == "profile"


def test_the_static_budget_is_the_largest_that_meets_d_on_p_of_questions():
    rows = [({}, n, 300.0, 10.0, "natural") for n in (40, 40, 1000, 1000)]
    assert static_budgets(rows, 2000.0, 50.0, 1024) == {"0.95": 165, "0.99": 165}    # 300 + 10 * 165 + 50 = 2000
    assert static_budgets(rows, 300.0, 50.0, 1024) == {"0.95": None, "0.99": None}


def test_an_old_profile_derives_its_terms_from_the_decode_median_and_says_so():
    m = mapping(deadline_profile(with_point=False, with_terms=False), 2000)
    assert m["approx_terms"] and m["reserve_source"] == "profile" and m["reserve_ms"] == pytest.approx(1.5 * 27.4 + 10)
    assert any("approx (profile predates deadline fields)" in line for line in mapping_lines("m", m))


def test_the_block_reads_as_the_spec_shows():
    p = deadline_profile()
    serve = serve_command(p, None, "m.think.json", deadline_ms=2000, cap=1024)
    lines = mapping_lines("m", mapping(p, 2000), serve)
    assert lines[0] == ("m  deadline 2000 ms, cap 1024 (single-stream, in-callo: add your client's hop and any "
                        "queueing)")
    assert lines[1].startswith("  closes: natural 40 %  budget 0 %  deadline 60 %  budget0 0 %   cut k p5/p50/p95 ")
    assert lines[2].startswith("  static: --think-budget ") and "(exact, reserve 51 ms, deadline point)" in lines[3]
    assert lines[4].startswith("  serve: callo serve --upstream URL --think-profile m.think.json --think-deadline 2000")
    assert serve_command(p, 256, "m.think.json", deadline_ms=2000, cap=1024).endswith(
        "--think-budget 256 --think-deadline 2000")                  # a combined setting serves both limits


def test_the_profiles_deadline_and_early_exit_points_reach_the_frontier_through_pick_preferences():
    """The pick-preferences contract for the profile shape calibrate writes: `<model>@<D>ms`
    (kind deadline, its cap) and `<model>@ee<z>` (kind ee) points, paired over one fixed row set that includes
    `rows[].d` and `rows[].e`."""
    p = deadline_profile(ee=True)
    pts = {pt.label: pt for pt in points_of(p, "m")}
    assert {"m@2000ms", "m@ee.9"} <= set(pts) and pts["m@2000ms"].kind == "deadline" and pts["m@ee.9"].kind == "ee"
    assert (pts["m@2000ms"].value, pts["m@2000ms"].cap, pts["m@ee.9"].value) == (2000, 1024, 0.9)
    j = joint([p])
    assert {x.label for x in j.ranked} >= {"m@2000ms", "m@ee.9"} and len(j.fixed) == 10


def test_compare_prints_the_deadline_blocks_and_the_early_exit_and_refuses_a_lone_budget(tmp_path, capsys):
    path = tmp_path / "m.think.json"
    path.write_text(json.dumps(deadline_profile(ee=True)))
    assert main(["compare", str(path), "--deadline", "2000,1000"]) == 0
    out = capsys.readouterr().out
    assert "## Deadlines" in out and "m  deadline 1000 ms" in out and "m  deadline 2000 ms" in out
    assert f"serve: callo serve --upstream URL --think-profile {path} --think-deadline 2000" in out
    assert "## Early exit" in out and "z .9: saves 40 tokens/question net" in out and "ON your front (faster than every budget/deadline point as accurate)" in out
    assert main(["compare", str(path), "--deadline", "2000", "--budget", "256", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [(m["deadline_ms"], m["budget"]) for m in report["deadlines"]] == [(2000, 256)]
    assert report["early_exit"]["m"]["K"] == 32 and report["early_exit"]["m"]["gain"]["0.9"]["gains"] is True
    assert main(["compare", str(path), "--budget", "256"]) == 2
    assert "need --deadline" in capsys.readouterr().err
    assert main(["compare", str(path), "--deadline", "0"]) == 2
    assert "--deadline: '0' is not a deadline" in capsys.readouterr().err
    assert main(["compare", str(path), "--deadline", "2000", "--reserve", "-1"]) == 2
    assert "--reserve is in ms" in capsys.readouterr().err


def test_calibrates_summary_prints_each_deadline_points_block():
    lines = summary_lines(deadline_profile())
    head = next(i for i, line in enumerate(lines) if line.startswith("m  deadline 2000 ms, cap 1024"))
    assert lines[head + 4] == ("  serve: callo serve --upstream URL --think-profile m.think.json --think-deadline 2000"
                               "   (static: --think-budget 164)")                    # 300 + 10 x 164 + 51 <= 2000
    assert head < next(i for i, line in enumerate(lines) if line.startswith("audit: "))      # before the audit


def _with_point_at(p, deadline, r=51.0):
    """`p` plus a measured deadline point at `deadline` whose rows are the simulation's own cuts (cap Bmax 1024)."""
    for i, row in enumerate(p["rows"]):
        n_gen, a, d = row["t"]
        s = simulate(a_ms=a, d_ms=d, n_gen=n_gen, close=row["b"]["1024"][5], deadline_ms=deadline, reserve_ms=r,
                     cap=1024)
        row.setdefault("d", {})[str(deadline)] = [i % 3 != 0, 0.5, 0.6, a + d * s.k + 27.0, s.k, s.close or "budget0"]
    p["deadlines"][str(deadline)] = {**p["deadlines"]["2000"], "reserve_ms": r}
    return p


def test_a_budget_above_bmax_runs_past_the_measured_traces_and_its_accuracy_is_approximate():
    """At --budget B > Bmax a question whose Bmax trace reached the cap before the deadline
    would think on past Bmax, where the profile holds no readout: it is re-simulated under B (its close and cut) and
    its accuracy is unknown, so the combined accuracy is a bracket, never the Bmax readout reported as exact."""
    p = deadline_profile()
    for row in p["rows"][4:]:                                         # six traces still thinking at Bmax: forced
        row["t"][0], row["b"]["1024"][4:] = 1024, [1024, "forced"]
    p = _with_point_at(p, 20000)                                      # the deadline never binds before Bmax
    assert {row["d"]["20000"][5] for row in p["rows"][4:]} == {"budget"}
    assert mapping(p, 20000)["accuracy"]["exact"] and mapping(p, 20000, budget=1024)["accuracy"]["exact"]
    m = mapping(p, 20000, budget=2048)
    k = simulate(a_ms=300.0, d_ms=10.0, n_gen=1024, close="forced", deadline_ms=20000, reserve_ms=51.0, cap=2048).k
    assert 1024 < k < 2048 and m["cut_k"] == [k, k, k]
    assert m["closes"] == {"natural": 0.4, "budget": 0.0, "deadline": 0.6, "budget0": 0.0}
    acc = m["accuracy"]
    assert not acc["exact"] and acc["unknown"] == pytest.approx(0.6) and acc["beyond_bmax"] == pytest.approx(0.6)
    assert (acc["lo"], acc["hi"]) == (pytest.approx(0.4), pytest.approx(1.0))
    line = next(x for x in mapping_lines("m", m) if x.startswith("  accuracy: "))
    assert line.startswith("  accuracy: approx [0.40, 1.00]: 60 % of questions") and "above Bmax 1024" in line
    assert mapping(p, 2000, budget=2048)["accuracy"]["exact"]         # every cut below Bmax: k_D is the cap's own


def test_the_exact_combined_accuracy_matches_a_hand_count():
    """A calibrated budget B binds some questions and the deadline D others; compare's combined accuracy takes
    each question's binding limit's own readout: natural -> rows[].b[Bmax], budget -> rows[].b[B], deadline ->
    rows[].d[D]. Each source's correct pattern is distinct, so a wrong source changes the count."""
    p = deadline_profile()
    nat_b = [True, True, False, True]                                 # rows 0-3 close naturally at 40 (b256 == b1024)
    terms = {i: (300.0, 10.0) for i in (4, 5, 6)} | {i: (100.0, 5.0) for i in (7, 8, 9)}
    d_ok = {4: True, 5: False, 6: True, 7: False, 8: False, 9: False}
    b256_ok = {4: False, 5: False, 6: False, 7: True, 8: False, 9: False}
    b1024_ok = {4: True, 5: True, 6: True, 7: False, 8: True, 9: True}
    for i, row in enumerate(p["rows"]):
        if i < 4:
            row["b"]["256"][0] = row["b"]["1024"][0] = nat_b[i]
            continue
        a, d = terms[i]
        row["t"] = [1024, a, d]
        row["b"]["1024"][4:], row["b"]["256"][4:] = [1024, "forced"], [256, "forced"]
        row["b"]["256"][0], row["b"]["1024"][0] = b256_ok[i], b1024_ok[i]
        s = simulate(a_ms=a, d_ms=d, n_gen=1024, close="forced", deadline_ms=2000, reserve_ms=51.0, cap=1024)
        assert s.close == "deadline" and (s.k < 256) == (i < 7)       # 300 + 10k: k 164; 100 + 5k: k 369
        row["d"]["2000"] = [d_ok[i], 0.5, 0.6, a + d * s.k + 27.0, s.k, "deadline"]
    hand = (sum(nat_b) + d_ok[4] + d_ok[5] + d_ok[6] + b256_ok[7] + b256_ok[8] + b256_ok[9]) / 10   # (3 + 2 + 1) / 10
    assert hand == 0.6
    j = joint([p])
    (m,) = deadline_mappings(j, (2000,), budget=256)
    assert m["closes"] == {"natural": 0.4, "budget": pytest.approx(0.3), "deadline": pytest.approx(0.3), "budget0": 0.0}
    assert m["accuracy"]["exact"] and m["accuracy"]["acc"] == pytest.approx(hand)


def test_the_summary_marks_one_frontier_over_every_kind_and_keeps_the_stored_marks(tmp_path):
    """Calibrate's summary shows display marks from compare's
    frontier over `points_of(profile)` (budgets, deadline points, early exit), so it agrees with compare and with the
    pick, which can pick a deadline point; the profile's stored marks stay budget-only."""
    p = deadline_profile()                                            # m@2000ms: acc .6 at 1.8 s; m@256: .6 at 3 s
    stored = json.loads(json.dumps(p["budgets"]))
    lines = summary_lines(p)
    row = {line.split()[0]: line for line in lines if line.startswith("m@")}
    assert row["m@256"].rstrip().endswith("dominated by m@2000ms") and p["budgets"]["256"]["frontier"]
    assert "dominated" not in row["m@2000ms"] and "dominated" not in row["m@1024"]
    marks = {x["label"]: x["dominated_by"] for x in report_dict(joint([p]))["points"]}
    assert marks["m@256"] == "m@2000ms" and marks["m@2000ms"] is None             # what compare marks
    assert p["budgets"] == stored                                     # the stored marks are left as they are
    assert any(line.startswith("  marks: one frontier over budgets, deadline points and early exit") for line in lines)
    assert not any("marks compare the budgets among themselves" in line for line in lines)


# ---- malformed rows, budget-0-only profiles, hints, serve lines, type checks ----

def _compare(tmp_path, capsys, p, *args):
    path = tmp_path / "m.think.json"
    path.write_text(json.dumps(p))
    code = main(["compare", str(path), *args])
    io = capsys.readouterr()
    return code, io.out, io.err


def test_compare_deadline_refuses_malformed_terms_and_rows_without_k_or_close_with_exit_2(tmp_path, capsys):
    """A profile `malformed` accepted made `compare --deadline` raise a traceback (TypeError, ValueError);
    the CLI contract is exit 2 with one line."""
    p = deadline_profile()
    p["rows"][0]["t"] = [40, None, 10]
    code, _, err = _compare(tmp_path, capsys, p, "--deadline", "2000")
    assert code == 2 and "row 0 (rec-0/q): t is not [n_gen, a_ms, d_ms]" in err
    for bad in ([40, 300.0], [-1, 300.0, 10.0], [40.5, 300.0, 10.0], [True, 300.0, 10.0], [40, 300.0, -1.0], "x"):
        p["rows"][0]["t"] = bad
        assert (malformed(p) or "").endswith("t is not [n_gen, a_ms, d_ms]"), bad
    p["rows"][0]["t"] = None
    assert malformed(p) is None                                       # null: a question without a trace
    q = deadline_profile()
    q["row_fields"] = ["correct", "nll1", "pmax", "lat_ms"]
    for row in q["rows"]:
        for per in (row["b"], row["d"]):
            for v in per.values():
                del v[4:]
    assert malformed(q) is None
    code, _, err = _compare(tmp_path, capsys, q, "--deadline", "2000")
    assert code == 2 and "--deadline needs each row's lat_ms, k and close (row_fields): m's rows carry none" in err


def test_the_smallest_budget_clause_names_only_cuts_below_it_not_the_runs_past_bmax():
    p = deadline_profile()
    for row in p["rows"][4:]:
        row["t"][0], row["b"]["1024"][4:] = 1024, [1024, "forced"]
    p = _with_point_at(p, 20000)
    line = next(x for x in mapping_lines("m", mapping(p, 20000, budget=2048)) if x.startswith("  accuracy: "))
    assert "smallest calibrated budget" not in line and "above Bmax 1024" in line
    line = next(x for x in mapping_lines("m", mapping(p, 2000, budget=64)) if x.startswith("  accuracy: "))
    assert "(the smallest calibrated budget is 256)" in line


def test_a_budget_0_only_profile_gets_one_line_per_deadline(tmp_path, capsys):
    p = profile("m", {0: 3}, {0: 300.0})
    code, out, _ = _compare(tmp_path, capsys, p, "--deadline", "2000")
    assert code == 0 and ("m  deadline 2000 ms: no calibrated thinking budget (budget 0 only): a deadline changes "
                          "nothing calibrate measured") in out
    assert "closes:" not in out
    code, out, _ = _compare(tmp_path, capsys, p, "--deadline", "2000", "--json")
    (m,) = json.loads(out)["deadlines"]
    assert code == 0 and m["deadline_ms"] == 2000 and m["skipped"].startswith("no calibrated thinking budget")


def test_a_mapping_with_no_terms_says_so_and_counts_the_questions_it_covers():
    p = deadline_profile(with_point=False, with_terms=False)
    p["environment"]["run"]["decode_tok_s_median"] = None           # no t and nothing to derive them from
    m = mapping(p, 2000)
    assert m["n"] == 0 and m["questions"] == 10
    lines = mapping_lines("m", m)
    assert "  no question has deadline terms (rows[].t) or a decode median to derive them" in lines
    assert not any(x.startswith("  closes:") or x.startswith("  accuracy:") for x in lines)
    q = deadline_profile()
    q["rows"][9]["t"] = None                                         # one thinking row lost its terms, no decode
    q["environment"]["run"]["decode_tok_s_median"] = None
    lines = mapping_lines("m", mapping(q, 2000))
    assert any(x.startswith("  closes: ") and x.endswith("(over 9 of 10 questions)") for x in lines)
    assert "(over " not in mapping_lines("m", mapping(deadline_profile(), 2000))[1]    # every question: no count


def test_compare_never_hints_a_deadline_the_profile_measured_and_skipped(tmp_path, capsys):
    """Budget 0 passes, the p99 limit fails every positive budget, so with no deadline point the
    no-pick reason suggests `--deadlines 2000`; once the profile lists 2000 in `deadline_skipped`, it must not."""
    p = deadline_profile(with_point=False)
    code, out, _ = _compare(tmp_path, capsys, p, "--limit", "p99=2", "--min-accuracy", "0.99")
    assert code == 0 and "a new calibrate run with --deadlines 2000" in out              # the positive control
    p["deadline_skipped"] = {"2000": DEADLINE_SKIPPED}
    code, out, _ = _compare(tmp_path, capsys, p, "--limit", "p99=2", "--min-accuracy", "0.99")
    assert code == 0 and "**pick: none**" in out and "--deadlines 2000" not in out


def test_compare_relays_calibrates_reason_for_a_skipped_deadline(tmp_path, capsys):
    p = deadline_profile()
    p["deadline_skipped"] = {"900": DEADLINE_SKIPPED}
    code, out, _ = _compare(tmp_path, capsys, p, "--deadline", "900,2000")
    assert code == 0 and f"  calibrate: {DEADLINE_SKIPPED}" in out
    block = out.split("m  deadline 2000 ms")[1]
    assert "calibrate:" not in block


def test_an_old_profiles_exact_accuracy_says_its_closes_are_approximate():
    p = deadline_profile(with_point=False, with_terms=False)
    m = mapping(p, 60000, budget=256)                                 # natural or B: every binding limit has a readout
    assert m["approx_terms"] and m["accuracy"]["exact"]
    line = next(x for x in mapping_lines("m", m) if x.startswith("  accuracy: "))
    assert "exact" not in line and "readouts measured; closes approx: profile predates deadline fields)" in line


def test_a_combined_serve_line_has_no_static_suffix_and_a_large_budget_raises_think_max(tmp_path, capsys):
    p = deadline_profile()
    code, out, _ = _compare(tmp_path, capsys, p, "--deadline", "2000", "--budget", "256")
    serve = next(x for x in out.splitlines() if x.startswith("  serve: "))
    assert code == 0 and serve.endswith("--think-budget 256 --think-deadline 2000")
    code, out, _ = _compare(tmp_path, capsys, p, "--deadline", "2000", "--budget", "5000")
    serve = next(x for x in out.splitlines() if x.startswith("  serve: "))
    assert code == 0 and "--think-budget 5000 --think-deadline 2000 --think-max 5000" in serve


def test_the_serve_lines_cap_is_the_deadline_points_else_bmax_in_compare_and_the_summary(tmp_path, capsys):
    """One cap source (nit): the deadline point's `cap`, else the profile's Bmax, in both the summary and compare."""
    p = deadline_profile()
    p["budgets"]["8192"] = {**p["budgets"]["1024"], "point": "m@8192"}
    for row in p["rows"]:
        row["b"]["8192"] = list(row["b"]["1024"])
    p["deadlines"]["2000"]["cap"] = 1024                             # the point was simulated under 1024
    summary = next(x for x in summary_lines(p) if x.startswith("  serve: ") and "--think-deadline 2000" in x)
    code, out, _ = _compare(tmp_path, capsys, p, "--deadline", "2000")
    serve = next(x for x in out.splitlines() if x.startswith("  serve: "))
    assert code == 0 and "--think-max" not in serve and "--think-max" not in summary
    del p["deadlines"]["2000"]                                        # no point: Bmax 8192 > 4096
    code, out, _ = _compare(tmp_path, capsys, p, "--deadline", "2000")
    serve = next(x for x in out.splitlines() if x.startswith("  serve: "))
    assert code == 0 and "--think-max 8192" in serve


@pytest.mark.parametrize("block, key, at, bad, why", [
    ("b", "1024", 4, None, "budget 1024 k is not an integer >= 0"),
    ("b", "256", 4, -3, "budget 256 k is not an integer >= 0"),
    ("b", "256", 4, 2.5, "budget 256 k is not an integer >= 0"),
    ("d", "2000", 4, None, "deadline 2000 k is not an integer >= 0"),
    ("b", "1024", 5, "stopped", "budget 1024 close is not null or one of natural, forced, eos"),
    ("b", "1024", 5, "deadline", "budget 1024 close is not null or one of natural, forced, eos"),
    ("d", "2000", 5, "forced", "deadline 2000 close is not null or one of natural, budget, deadline, eos, budget0"),
    ("b", "1024", 3, float("nan"), "budget 1024 lat_ms is not a finite number >= 0"),
    ("b", "0", 3, -1.0, "budget 0 lat_ms is not a finite number >= 0"),
    ("d", "2000", 3, "fast", "deadline 2000 lat_ms is not a finite number >= 0"),
])
def test_malformed_type_checks_the_k_close_and_lat_ms_deadline_reads(tmp_path, capsys, block, key, at, bad, why):
    """Every rows[].b / rows[].d value `--deadline` reads (k, close,
    lat_ms, wherever row_fields declares them, the approximate-terms path included) is type-checked, so a hand-edited
    profile is a one-line refusal (exit 2), never a traceback."""
    p = deadline_profile()
    p["rows"][3][block][key][at] = bad
    assert malformed(p) == f"row 3 (rec-3/q): {why}"
    if bad == bad and not isinstance(bad, float):                     # NaN is no JSON; the rest go through the CLI
        code, _, err = _compare(tmp_path, capsys, p, "--deadline", "2000")
        assert code == 2 and why in err


def test_the_type_checks_follow_row_fields():
    p = deadline_profile()
    assert malformed(p) is None
    p["row_fields"] = ["correct", "nll1", "pmax", "lat_ms"]           # k and close not declared: not checked
    for row in p["rows"]:
        for per in (row["b"], row["d"]):
            for v in per.values():
                v[4:] = [None, "x"]
    assert malformed(p) is None
    p["rows"][2]["b"]["256"][3] = None                                # lat_ms declared: checked
    assert malformed(p) == "row 2 (rec-2/q): budget 256 lat_ms is not a finite number >= 0"


# ---- profiles `malformed` accepted that still raised a traceback ----

@pytest.mark.parametrize("latency, why", [("x", "latency is not a JSON object"),
                                          ({"read_forced_ms": 3.0}, "latency.read_forced_ms is not a JSON object")])
def test_compare_deadline_refuses_a_latency_block_that_is_not_an_object(tmp_path, capsys, latency, why):
    p = deadline_profile()
    p["latency"] = latency
    assert malformed(p) == why
    code, _, err = _compare(tmp_path, capsys, p, "--deadline", "2000")
    assert code == 2 and why in err


@pytest.mark.parametrize("fields", [["correct", ["k"]], ["correct", {"k": 1}], ["correct", 4]])
def test_compare_deadline_refuses_row_fields_that_are_not_names(tmp_path, capsys, fields):
    p = deadline_profile()
    p["row_fields"] = fields
    assert malformed(p) == "row_fields is not a list of field names"
    code, _, err = _compare(tmp_path, capsys, p, "--deadline", "2000")
    assert code == 2 and "row_fields is not a list of field names" in err
    p["row_fields"] = ["nll1"]
    assert malformed(p) == 'row_fields has no "correct"'


@pytest.mark.parametrize("seed", ["0", 1.5, -1, True, [0]])
def test_compare_refuses_a_seed_that_is_not_an_integer_from_0(tmp_path, capsys, seed):
    p = deadline_profile()
    p["comparable"]["seed"] = seed
    assert malformed(p) == "comparable.seed is not an integer >= 0"
    code, _, err = _compare(tmp_path, capsys, p)
    assert code == 2 and "comparable.seed is not an integer >= 0" in err
    for ok in (0, 7, None):
        p["comparable"]["seed"] = ok
        assert malformed(p) is None


@pytest.mark.parametrize("fingerprint", [["sha256:x"], {"a": 1}, 5])
def test_compare_refuses_a_host_fingerprint_that_is_not_a_string(tmp_path, capsys, fingerprint):
    p = deadline_profile()
    p["environment"]["host"]["fingerprint"] = fingerprint
    assert malformed(p) == "environment.host.fingerprint is not a string or null"
    code, _, err = _compare(tmp_path, capsys, p)
    assert code == 2 and "environment.host.fingerprint is not a string or null" in err
    p["environment"]["host"]["fingerprint"] = None
    assert malformed(p) is None
