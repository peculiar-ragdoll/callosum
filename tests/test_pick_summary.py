"""Pick preferences: calibrate's pick line (the preference in canonical form, M and its
score, the most accurate point), the early-exit opt-in line, the no-pick reason, a deadline pick's `serve with:` line,
and the old pick blocks still rendering and loading."""
import dataclasses
from pathlib import Path

from test_calibrate_think import SAMPLING, STATS, synthetic_raw
from test_pick_points import _built

from callo.calibrate_think import build_profile, limit_lines, pick_line, serve_line, summary_lines
from callo.think import Preference, load_think_profile, parse_limit

NEW = {"kind": "budget", "budget": 512, "budgets": [512], "deadline_ms": None, "cap": None, "model": "tiel",
       "label": "tiel@512", "rule": "fastest within noise of the best score", "best": "tiel@1024",
       "most_accurate": "tiel@1024", "dacc_best": [-0.0053, -0.0133, 0.0], "dscore_best": [-0.1308, -0.7646, 0.3556],
       "score": {"pick": 84.48, "best": 84.61},
       "preference": {"latency": "mean", "price_pp_per_s": 5.0, "limits": [], "min_accuracy": None,
                      "max_memory_gib": None},
       "excluded": {}, "reason": None, "unsized": [], "would_pick": None, "would_best": None,
       "latency": "mean", "max_latency_s": None, "min_accuracy": None}
OLD = {"budget": 1024, "budgets": [1024], "model": "tiel", "label": "tiel@1024", "rule": "fastest within noise of the best",
       "best": "tiel@1024", "dacc_best": [0.0, 0.0, 0.0], "unsized": [], "max_latency_s": None, "min_accuracy": None,
       "latency": "p50"}


def test_the_pick_line_names_a_non_default_preference_m_and_its_score():
    assert pick_line(NEW) == ("pick: tiel@512 (fastest within noise of the best score, --price 5pp/s on mean); best "
                              "tiel@1024 84.6 pp; Δacc vs it -0.5pp [-1.3, +0.0], Δscore -0.13pp [-0.76, +0.36]")
    priced_away = {**NEW, "label": "tiel@0", "budget": 0, "budgets": [0], "best": "tiel@0", "dacc_best": [0.0] * 3,
                   "dscore_best": [0.0] * 3, "score": {"pick": 80.59, "best": 80.59},
                   "preference": {**NEW["preference"], "price_pp_per_s": 10.0}}
    assert pick_line(priced_away) == ("pick: tiel@0 (fastest within noise of the best score, --price 10pp/s on mean); "
                                      "best tiel@0 80.6 pp; most accurate tiel@1024")
    default = {**NEW, "rule": "fastest within noise of the best", "score": None, "dscore_best": None,
               "preference": {**NEW["preference"], "price_pp_per_s": 0.0}}
    assert pick_line(default) == "pick: tiel@512 (fastest within noise of the best); best tiel@1024; Δacc vs it -0.5pp [-1.3, +0.0]"
    limited = {**default, "preference": {**default["preference"],
                                         "limits": [{"stat": "p99", "le_ms": 6000.0, "from": "--limit"}]}}
    assert pick_line(limited).startswith("pick: tiel@512 (fastest within noise of the best, --limit p99=6s); ")
    assert pick_line(NEW, bold=True).startswith("**pick: tiel@512** (")


def test_an_old_pick_block_still_renders_and_loads():
    assert pick_line(OLD) == "pick: tiel@1024 (fastest within noise of the best, --latency p50); best tiel@1024"
    assert pick_line({**OLD, "label": None, "rule": "no pick", "budget": None}) == "pick: none (no pick); best tiel@1024"


def test_a_no_pick_prints_its_reason():
    none = {**NEW, "kind": None, "label": None, "budget": None, "budgets": [], "rule": "no pick", "best": None,
            "reason": "no point meets --limit p99=0.3s: all 4 excluded; nearest tiel@0 at p99 0.44 s, from 4 of 376"}
    assert pick_line(none) == ("pick: none (no point meets --limit p99=0.3s: all 4 excluded; nearest tiel@0 at p99 "
                               "0.44 s, from 4 of 376)")


EE_PICK_SERVES = ("serve with: none (this profile's pick is an early-exit point; early exit is opt-in and never "
                  "picked now: re-pick with `callo compare`)")


def test_a_stored_early_exit_pick_serves_nothing_and_loads_no_budget(tmp_path):
    """A profile written while early-exit points could be picked: its `serve with:` line never carries
    `--think-converge` (early exit is opt-in) nor falls through to a budget."""
    _, prof = _built(tmp_path)
    prof["pick"] = {**NEW, "kind": "ee", "label": "fake@ee.95", "best": "fake@ee.95", "budget": None, "budgets": [],
                    "deadline_ms": None, "cap": 1024, "z": 0.95, "model": "fake"}
    lines = summary_lines(prof, path=Path("p.think.json"))
    assert lines[-1] == EE_PICK_SERVES and load_think_profile(prof).pick is None
    assert "pick: fake@ee.95 (early exit, no longer picked: re-pick with `callo compare`)" in lines
    assert pick_line(prof["pick"], bold=True) == ("**pick: fake@ee.95** (early exit, no longer picked: re-pick with "
                                                  "`callo compare`)")
    prof["pick"]["cap"] = 8192                                    # above serve's default --think-max
    assert summary_lines(prof, path=Path("p.think.json"))[-1] == EE_PICK_SERVES


def test_an_old_block_naming_an_early_exit_point_prints_no_line_about_it(tmp_path):
    """A block from before early-exit points could be picked names the one an admitting run would pick (`would_*`);
    its budget pick is today's, so the summary says nothing about it (the opt-in line speaks for early exit)."""
    _, prof = _built(tmp_path)
    prof["pick"] = {**NEW, "label": "fake@64", "best": "fake@256", "most_accurate": "fake@256", "budget": 64,
                    "budgets": [64], "model": "fake", "would_pick": "fake@ee.95", "would_best": "fake@ee.95"}
    lines = summary_lines(prof, path=Path("p.think.json"))
    at = next(i for i, line in enumerate(lines) if line.startswith("pick: "))
    assert lines[at].startswith("pick: fake@64 (fastest within noise of the best score, --price 5pp/s on mean); best "
                                "fake@256 84.6 pp")
    assert not any("fake@ee.95" in line for line in lines[at:])
    assert lines[-1].endswith("--think-profile p.think.json --think-budget 64")
    prof["pick"] = dict(OLD, label="fake@256", budget=256, budgets=[256], best="fake@256")      # an older profile
    assert summary_lines(prof)[-1].endswith("--think-budget 256")
    assert load_think_profile(prof).pick == 256


def test_a_deadline_pick_serves_think_deadline_and_loads_no_budget(tmp_path):
    _, prof = _built(tmp_path)
    prof["pick"] = {**NEW, "kind": "deadline", "budget": None, "budgets": [], "deadline_ms": 2000, "cap": 256,
                    "label": "fake@2000ms", "best": "fake@2000ms", "model": "fake"}
    assert serve_line(prof, "p.think.json") == ("serve with: callo serve --upstream URL --think-profile p.think.json "
                                                "--think-deadline 2000")
    prof["pick"]["cap"] = 8192                                           # a cap above serve's --think-max default
    assert serve_line(prof, "p.think.json").endswith("--think-deadline 2000 --think-max 8192")
    assert load_think_profile(prof).pick is None                         # serve reads no pick, and no budget here
    assert summary_lines(prof)[-1].startswith("serve with: ") and "--think-budget" not in summary_lines(prof)[-1]


def test_calibrate_picks_under_its_preference_and_says_why_there_is_no_pick(tmp_path):
    items, prof = _built(tmp_path)
    raw, settings = synthetic_raw(items)
    limited = build_profile(items, raw, dataclasses.replace(settings, limits=(parse_limit("max=0.1s"),)),
                            sampling=SAMPLING, stats=STATS, name="fake")
    pk = limited["pick"]
    assert pk["label"] is None and pk["reason"].startswith("no point meets --limit max=0.1s: all 3 excluded; nearest fake@0")
    assert pk["preference"]["limits"] == [{"stat": "max", "le_ms": 100.0, "from": "--limit"}]
    lines = summary_lines(limited)
    assert f"pick: none ({pk['reason']})" in lines and not lines[-1].startswith("serve with")
    priced = build_profile(items, raw, dataclasses.replace(settings, price=1000.0), sampling=SAMPLING, stats=STATS,
                           name="fake")
    assert priced["pick"]["rule"] == "fastest within noise of the best score" and priced["pick"]["label"] == "fake@0"
    assert priced["pick"]["best"] == "fake@0" and priced["pick"]["most_accurate"] == "fake@256"
    # each budget's own dacc_best keeps its meaning: against the most accurate point, not against M
    assert priced["budgets"]["0"]["dacc_best"] == prof["budgets"]["0"]["dacc_best"]


def test_each_active_limit_says_what_it_rests_on():
    """A limit is a deterministic filter on a point estimate; the summary states what it rests on."""
    pref = Preference(limits=(parse_limit("p99=6s"), parse_limit("max=8s"), parse_limit("mean=2s")), max_latency_s=3.0)
    assert limit_lines(pref, 376) == ["--limit p99=6s: p99 from 4 of 376", "--limit max=8s: max from 1 of 376",
                                      "--limit mean=2s: mean from 376 of 376", "--limit mean=3s: mean from 376 of 376"]
    assert limit_lines(Preference(), 376) == [] and limit_lines(pref, 0) == []
    assert limit_lines(Preference(limits=(parse_limit("p99=6s"),)), 48) == ["--limit p99=6s: p99 from 1 of 48"]
    assert limit_lines(pref, [376, 120, 376])[:2] == ["--limit p99=6s: p99 from 4 of 376, 2 of 120",   # compare: per n
                                                    "--limit max=8s: max from 1 of 376, 1 of 120"]
    pick = {**NEW, "preference": {**NEW["preference"], "limits": [{"stat": "p99", "le_ms": 6000.0, "from": "--limit"}]}}
    assert limit_lines(Preference.from_dict(pick["preference"]), 8) == ["--limit p99=6s: p99 from 1 of 8"]


def test_the_summary_prints_the_limits_under_the_pick_line(tmp_path):
    items, _ = _built(tmp_path)
    raw, settings = synthetic_raw(items)
    prof = build_profile(items, raw, dataclasses.replace(settings, limits=(parse_limit("p95=2s"),)), sampling=SAMPLING,
                         stats=STATS, name="fake")
    lines = summary_lines(prof)
    at = next(i for i, line in enumerate(lines) if line.startswith("pick: "))
    assert lines[at + 1] == "  --limit p95=2s: p95 from 1 of 8"


def _ee_pick(prof, **over):
    prof["pick"] = {**NEW, "kind": "ee", "label": "fake@ee.95", "best": "fake@ee.95", "budget": None, "budgets": [],
                    "deadline_ms": None, "cap": 1024, "z": 0.95, "model": "fake", **over}
    return prof


def test_a_stored_early_exit_pick_never_serves_a_budget_or_think_converge(tmp_path):
    """Whatever its z or cap (a hand-edited block included), a stored early-exit pick never falls through to
    `--think-budget None` and never prints `--think-converge`, a `--model` launch included."""
    _, prof = _built(tmp_path)
    prof["environment"]["model"]["path"] = "/models/fake model.gguf"
    prof["environment"]["server"]["server_args"] = ["-ngl", "99"]
    prof["environment"]["server"]["launch"] = {"ctx": 16384, "parallel": 2, "args": ["-ngl", "99"]}
    for over in ({"z": None}, {"z": 0}, {"z": 1}, {"z": True}, {"z": "0.95"}, {"z": float("nan")}, {"cap": 8192}, {}):
        assert serve_line(_ee_pick(prof, **over), "p.think.json") == EE_PICK_SERVES, over
        assert summary_lines(prof)[-1] == EE_PICK_SERVES and "--think-converge" not in EE_PICK_SERVES
