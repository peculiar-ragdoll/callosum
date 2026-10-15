"""Pick preferences: the five statistics in calibrate's summary table and compare's table, the interval on the pick
statistic's column, `= max` below 100 questions, the tails line, and report.json's points."""
import dataclasses
import json

from test_pick_points import _built, _strip, _with_deadline, _with_ee

from callo.calibrate_think import stat_cells, summary_lines, tails_line
from callo.compare import joint, report_dict, table_lines

HEAD_MEAN = ("point                       acc           Δacc vs 0 [95% CI]            mean [95% CI]      p50      p95      "
             "p99      max  think  forced")


def test_tails_lines_say_what_each_tail_rests_on():
    assert tails_line(376) == "tails (n 376): p95 from 19 questions, p99 from 4, max from 1"
    assert tails_line(8) == "tails (n 8): p95 from 1 question, p99 = max, max from 1"      # below 100: p99 is max
    assert tails_line(100) == "tails (n 100): p95 from 6 questions, p99 from 2, max from 1"


def test_stat_cells_put_the_interval_on_the_pick_statistic_only():
    stats = {"lat_mean_ms": [1830.0, 1720.0, 1950.0], "lat_p50_ms": [1360.0, 1270.0, 1490.0],
             "lat_p95_ms": [4350.0, 3590.0, 5710.0], "lat_p99_ms": [5800.0, 5670.0, 5840.0], "lat_max_ms": 5860.0}
    assert stat_cells(stats, 376, "mean") == {"mean": "1.83s [1.72s, 1.95s]", "p50": "1.36s", "p95": "4.35s",
                                              "p99": "5.80s", "max": "5.86s"}
    assert stat_cells(stats, 376, "p95")["p95"] == "4.35s [3.59s, 5.71s]" and stat_cells(stats, 376, "p95")["mean"] == "1.83s"
    assert stat_cells(stats, 48, "mean")["p99"] == "= max"


def test_the_summary_table_shows_the_five_statistics_and_the_tails_line(tmp_path):
    _, prof = _built(tmp_path)
    prof["pick"]["latency"] = "mean"                                     # the interval follows the pick statistic
    lines = summary_lines(prof)
    assert lines[2] == HEAD_MEAN
    row = next(line for line in lines if line.startswith("fake@256"))
    e = prof["budgets"]["256"]
    mean = f"{e['lat_mean_ms'][0] / 1000:.2f}s [{e['lat_mean_ms'][1] / 1000:.2f}s, {e['lat_mean_ms'][2] / 1000:.2f}s]"
    assert mean in row and "= max" in row and f"{e['lat_max_ms'] / 1000:.2f}s" in row
    at = lines.index(row)
    assert lines[at + 1] == "tails (n 8): p95 from 1 question, p99 = max, max from 1"
    prof["pick"]["latency"] = "p50"
    assert summary_lines(prof)[2].split() == ["point", "acc", "Δacc", "vs", "0", "[95%", "CI]", "mean", "p50", "[95%",
                                              "CI]", "p95", "p99", "max", "think", "forced"]


def test_an_older_profiles_summary_recomputes_the_statistics_it_lacks(tmp_path):
    _, prof = _built(tmp_path)
    assert summary_lines(_strip(prof)) == summary_lines(prof)


def test_compares_table_has_the_five_statistics_and_one_tails_line_per_question_count(tmp_path):
    _, prof = _built(tmp_path)
    j = dataclasses.replace(joint([_with_ee(_with_deadline(prof))]), latency="mean")
    rows = table_lines(j)
    assert rows[0] == ("| point | acc | mean [95% CI] | p50 | p95 | p99 | max | think | forced | sampler | GiB | "
                       "mark |")
    by = {r.split(" | ")[0][2:]: r for r in rows[2:] if r.startswith("| ")}
    assert set(by) == {"fake@0", "fake@64", "fake@256", "fake@1500ms", "fake@ee.95"}
    e = prof["budgets"]["256"]
    assert f"| {e['lat_mean_ms'][0] / 1000:.2f}s [" in by["fake@256"] and "| = max |" in by["fake@256"]
    assert rows[-2:] == ["", "tails (n 8): p95 from 1 question, p99 = max, max from 1"]
    with_nll = _with_deadline(prof)
    with_nll["deadlines"]["1500"]["nll_T"] = 0.123                        # --verbose reads a deadline's own entry
    v = table_lines(dataclasses.replace(joint([with_nll]), latency="mean"), verbose=True)
    t = with_nll["deadlines"]["1500"]["T"]
    assert next(r for r in v if r.startswith("| fake@1500ms |")).endswith(f"| 0.123 | - | {t:.2f} |")


def test_report_json_points_carry_the_kind_the_five_statistics_and_the_tail_counts(tmp_path):
    _, prof = _built(tmp_path)
    d = report_dict(joint([_with_ee(_with_deadline(prof))]))
    by = {p["label"]: p for p in d["points"]}
    b, dl, ee = by["fake@256"], by["fake@1500ms"], by["fake@ee.95"]
    assert (b["kind"], b["value"], b["budget"], b["budgets"], b["cap"]) == ("budget", 256, 256, [256], None)
    assert (dl["kind"], dl["value"], dl["budget"], dl["budgets"], dl["cap"]) == ("deadline", 1500, None, [], 256)
    assert (ee["kind"], ee["value"], ee["budget"], ee["budgets"]) == ("ee", 0.95, None, [])
    for k in ("lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms"):
        assert b[k] == prof["budgets"]["256"][k]
    assert b["lat_max_ms"] == prof["budgets"]["256"]["lat_max_ms"]
    assert b["tail_n"] == {"p50": 5, "p95": 1, "p99": 1, "max": 1} and b["n"] == 8
    json.loads(json.dumps(d))
