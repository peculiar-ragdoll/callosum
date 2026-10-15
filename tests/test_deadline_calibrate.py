"""Calibrate's deadline points: the probe's simulated cuts and its
reserve read, `--deadlines` and r_cal, the `latency` block, `rows[].t`, the deadline points (their rows on the budgets'
clock, the five latency statistics, the closes and the cut) and how they reach the picker, compare's validation, the
summary's rows and resume compatibility."""
import copy
import dataclasses
import json

import pytest
from fake_upstream import FakeUpstream
from test_calibrate_think import SAMPLING, STATS, _len, _probs, _Served, _settings, synthetic_raw, write_data
from test_think_serve import BODY, _probe

import callo.calibrate_think as ct
import callo.cli
from callo.calibrate_bias import Refusal
from callo.calibrate_think import (
    ThinkSettings,
    _open_raw,
    build_profile,
    calibrate_one,
    deadline_outcomes,
    load_data,
    outcomes,
    parse_deadlines,
    raw_header,
    read_raw,
    reserve_record,
    settings_from_args,
    summary_lines,
)
from callo.compare import frontier_marks, load_profile, malformed, points_of
from callo.deadline import simulate, trace_terms
from callo.deadline_map import mapping
from callo.errors import InvalidInput
from callo.think import crossfit, ece_at, fit_T, load_think_profile, nll_at, parse_limit


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=10_000).start()
    yield u
    u.stop()


def _reads(up):
    return [len(r.json["prompt"]) for r in up.requests if r.path == "/completion" and r.json.get("n_probs")]


def _sim(t, deadline, reserve, cap):
    a, d = trace_terms(t["frame_ms"], t["gen_wall_ms"], t["predicted_ms"], t["n_gen"])
    return simulate(a_ms=a, d_ms=d, n_gen=t["n_gen"], close=t["close"], deadline_ms=deadline, reserve_ms=reserve,
                    cap=cap)


def test_the_probe_simulates_each_deadline_on_the_trace_and_reads_its_cut_in_the_ascending_list(up):
    probe, _, _ = _probe(up)
    code, out, _ = probe.respond({"request": BODY, "budgets": [64, 1024], "seed": 1, "deadlines": [600, 3000, 60_000],
                                  "reserve_ms": 50}, {})
    assert code == 200
    q = out["questions"]["q"]
    t = q["trace"]
    for d in (600, 3000, 60_000):
        s, e = _sim(t, d, 50.0, 1024), q["deadlines"][str(d)]
        assert (e["close"], e["k"], e["chunks"], e["elapsed_ms"]) == (s.close, s.k, s.chunks, s.elapsed_ms)
        assert len(e["probs"]) == 3 and e["read_ms"] > 0
    assert q["deadlines"]["600"]["close"] == "deadline" and 0 < q["deadlines"]["600"]["k"] < 64
    assert q["deadlines"]["60000"]["close"] == "budget" and q["deadlines"]["60000"]["k"] == 1024
    assert q["deadlines"]["60000"]["probs"] == q["budgets"]["1024"]["probs"]               # k = n_gen: one read
    reads = _reads(up)
    assert reads == [reads[0], *sorted(reads[1:])] and reads[0] == max(reads)             # n_gen first, then ascending
    assert out["readouts"] == len(reads) == len({e["k"] for e in q["deadlines"].values()} | {64, 1024})


def test_a_deadline_too_short_to_think_is_budget0_with_no_readout_and_a_natural_close_reuses_n_gen(up):
    up.think_len = 30
    probe, _, _ = _probe(up)
    code, out, _ = probe.respond({"request": BODY, "budgets": [1024], "seed": 1, "deadlines": [5, 5000],
                                  "reserve_ms": 50}, {})
    q = out["questions"]["q"]
    assert code == 200 and q["deadlines"]["5"] == {"close": "budget0", "k": 0, "chunks": 0, "elapsed_ms": 0.0}
    assert q["deadlines"]["5000"]["close"] == "natural" and q["deadlines"]["5000"]["k"] == 31 and out["readouts"] == 1


def test_the_reserve_read_is_a_forced_close_one_short_of_a_natural_trace_read_first(up):
    up.think_len = 30
    probe, _, _ = _probe(up)
    code, out, _ = probe.respond({"request": BODY, "budgets": [1024], "seed": 1, "reserve_read": True}, {})
    rr = out["questions"]["q"]["reserve_read"]
    assert code == 200 and (rr["k"], rr["add_step"], rr["read_prompt_n"]) == (30, True, 3) and out["readouts"] == 2
    assert _reads(up)[0] == out["questions"]["q"]["on_tokens"] + 30 + 3                 # on + gen[:30] + the close
    up.think_len = 10_000
    code, out, _ = probe.respond({"request": BODY, "budgets": [64], "seed": 2, "reserve_read": True}, {})
    rr = out["questions"]["q"]["reserve_read"]
    assert (rr["k"], rr["add_step"], rr["read_prompt_n"]) == (64, False, 4)                 # forced: as served


@pytest.mark.parametrize("extra, why", [({"deadlines": [600]}, "reserve_ms must be"),
                                        ({"deadlines": [0], "reserve_ms": 1}, "deadlines must be"),
                                        ({"deadlines": [9, 3], "reserve_ms": 1}, "deadlines must be"),
                                        ({"deadlines": [1, 2, 3, 4, 5], "reserve_ms": 1}, "deadlines must be"),
                                        ({"deadlines": [600], "reserve_ms": -1}, "reserve_ms must be"),
                                        ({"reserve_ms": 5}, "needs deadlines"),
                                        ({"reserve_read": "yes"}, "reserve_read must be")])
def test_the_probe_refuses_malformed_extras_with_422(up, extra, why):
    probe, _, _ = _probe(up)
    code, out, _ = probe.respond({"request": BODY, "budgets": [64], **extra}, {})
    assert code == 422 and why in out["detail"]
    code, out, _ = probe.respond({"request": BODY, "budgets": [0], "deadlines": [600], "reserve_ms": 1}, {})
    assert code == 422 and "need a positive budget" in out["detail"]


def test_parse_deadlines_and_the_settings_comparable_keys_only_when_set():
    assert parse_deadlines(None) == () and parse_deadlines("2000,1000") == (1000, 2000)
    for bad, why in (("0", "from 1"), ("600001", "from 1"), ("1,1", "twice"), ("1,2,3,4,5", "at most 4"),
                     ("x", "from 1")):
        with pytest.raises(InvalidInput, match=why):
            parse_deadlines(bad)
    base = ThinkSettings("d", "d", "d" * 64, (0, 64))
    assert "deadlines" not in base.comparable() and "deadline_reserve_ms" not in base.comparable()
    s = ThinkSettings("d", "d", "d" * 64, (0, 64), deadlines=(1000,), reserve_ms=40)
    assert s.comparable()["deadlines"] == [1000] and s.comparable()["deadline_reserve_ms"] == 40


def test_r_cal_skips_warm_up_question_1_by_index_even_when_it_recorded_no_read():
    """Question 1 is the cold one whatever it is; a single-option question 1 records no read
    (a placeholder), and the first warm read is never dropped in its place."""
    base = ThinkSettings("d", "d", "d" * 64, (0, 64), deadlines=(1000,))
    assert reserve_record(base, [None, 40.0, 30.0]) == {"ms": 70.0, "source": "pre-pass", "reads": [None, 40.0, 30.0]}
    assert reserve_record(base, [90.0, None, 30.0])["ms"] == 55.0
    assert reserve_record(base, [None])["source"] == "default"


def test_r_cal_skips_the_cold_first_warm_up_read_but_records_every_read():
    base = ThinkSettings("d", "d", "d" * 64, (0, 64), deadlines=(1000,))
    assert reserve_record(base, [90.0, 20.0, 30.0]) == {"ms": 55.0, "source": "pre-pass", "reads": [90.0, 20.0, 30.0]}
    assert reserve_record(base, [90.0]) == {"ms": 100.0, "source": "default", "reads": [90.0]}
    assert reserve_record(base, []) == {"ms": 100.0, "source": "default", "reads": []}
    flagged = ThinkSettings("d", "d", "d" * 64, (0, 64), deadlines=(1000,), reserve_ms=40)
    assert reserve_record(flagged, [20.0, 30.0]) == {"ms": 40.0, "source": "flag", "reads": [20.0, 30.0]}


def test_a_raw_file_without_deadlines_resumes_without_them_and_adding_them_refuses(tmp_path):
    plain = ThinkSettings("d", "d", "d" * 64, (0, 64))
    path = tmp_path / "x.raw.jsonl"
    f, _ = _open_raw(path, raw_header(STATS, plain, SAMPLING), resume=False, log=print)
    f.close()
    f, _ = _open_raw(path, raw_header(STATS, plain, SAMPLING), resume=True, log=print)
    f.close()
    with pytest.raises(Refusal, match="comparable differ"):
        _open_raw(path, raw_header(STATS, ThinkSettings("d", "d", "d" * 64, (0, 64), deadlines=(1000,)), SAMPLING),
                  resume=True, log=print)


def test_the_latency_block_and_rows_t_come_from_the_phase_b_traces(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    lat = prof["latency"]
    assert set(lat) == {"frame_ms", "fixed_ms", "prefill_ms_per_token", "decode_ms_per_token", "read_natural_ms",
                        "read_forced_ms", "on_tokens"}
    assert lat["read_natural_ms"]["p50"] == 24.0 and lat["read_forced_ms"] is None and lat["on_tokens"]["p50"] == 60
    assert lat["decode_ms_per_token"]["p50"] == pytest.approx(3000.0 / 299)          # predicted / (n_gen - 1)
    assert lat["decode_ms_per_token"]["p95"] == pytest.approx(400.0 / 39)
    t = raw.rows[1]["trace"]
    row = next(r for r in prof["rows"] if r["q"] == items[0].uid)
    assert row["t"] == [300, *trace_terms(t["frame_ms"], t["gen_wall_ms"], t["predicted_ms"], 300)]
    assert row["t"][1] + 300 * row["t"][2] == pytest.approx(t["frame_ms"] + t["gen_wall_ms"])   # a + n·d is the end
    assert "deadlines" not in prof and "d" not in row
    assert load_think_profile(prof).latency["decode_ms_per_token"]["p50"] > 0


def _with_deadlines(items, raw, deadline=2000):
    """synthetic_raw plus each phase-B row's simulated close at `deadline`: the cut items cut at k 64 (their 64 read),
    the natural closers natural at n_gen 40, item 2 a budget-0 fallback."""
    raw.reserve = {"ms": 51.0, "source": "pre-pass", "reads": [80.0, 27.0]}
    settings = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 64, 256), audit=6, deadlines=(deadline,))
    for r in raw.rows:
        if r["phase"] != "B":
            continue
        i = next(n for n, it in enumerate(items) if it.uid == r["q"])
        if i == 2:
            e = {"close": "budget0", "k": 0, "chunks": 0, "elapsed_ms": 0.0}
        elif r["trace"]["n_gen"] == 300:
            e = {**r["budgets"]["64"], "close": "deadline", "k": 64, "chunks": 3, "elapsed_ms": 1940.0}
        else:
            e = {**r["budgets"]["256"], "close": "natural", "k": 40, "chunks": 1, "elapsed_ms": 640.0}
        r["deadlines"] = {str(deadline): e}
    return raw, settings


def test_a_deadline_rows_latency_is_the_budgets_clock_at_its_cut(tmp_path):
    """The chunk-0 asymmetry: rows[].d[D]'s lat_ms is `think.budget_latency` at k_D, the same
    in-callo clock as rows[].b (never the simulated elapsed, which adds the chunks' overhead)."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, _ = synthetic_raw(items)
    raw, settings = _with_deadlines(items, raw)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    outs = outcomes(items, raw, [0, 64, 256])
    rows = {r["q"]: r for r in prof["rows"]}
    cut, fallback, natural = items[0].uid, items[2].uid, items[5].uid
    assert rows[cut]["d"]["2000"][3] == pytest.approx(outs[cut][64].lat_ms) == pytest.approx(880.0)
    assert rows[cut]["d"]["2000"][4:] == [64, "deadline"]
    assert rows[fallback]["d"]["2000"][3] == pytest.approx(40.0 + outs[fallback][0].lat_ms)       # frame + budget 0
    assert rows[fallback]["d"]["2000"][0] == outs[fallback][0].correct and rows[fallback]["d"]["2000"][5] == "budget0"
    assert rows[natural]["d"]["2000"][3] == pytest.approx(outs[natural][256].lat_ms)
    e = prof["deadlines"]["2000"]
    assert (e["cap"], e["reserve_ms"], e["n"]) == (256, 51.0, 8) and e["T"] > 0
    assert e["closes"] == {"natural": 4 / 8, "budget": 0.0, "deadline": 3 / 8, "eos": 0.0, "budget0": 1 / 8}
    assert (e["k_p50"], e["chunks_mean"]) == (64, (3 * 3 + 4 * 1) / 7)
    assert all(len(e[k]) == 3 for k in ("lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "dacc"))
    assert e["lat_max_ms"] == max(r["d"]["2000"][3] for r in prof["rows"])
    json.dumps(prof, allow_nan=False)


def test_t_d_is_fitted_only_on_the_rows_read_at_t_d(tmp_path):
    """Serve reads a budget-0 fallback at T_0 and a single-option question at
    budget 0, so T_D, nll and ece are fitted on the rows read at T_D (a close, not budget0) alone; neither kind of row
    moves them, while acc, n and the latencies still count every question, as the point serves them."""
    one = json.dumps({"state": "s", "questions": {"q": {"type": "choice", "instructions": "only", "criteria": {"o0": "x"},
                                                        "label": "o0"}}, "_meta": {"id": "rec-8", "group_id": "g4"}})
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8, extra=[one]))
    raw, _ = synthetic_raw(items[:8])
    raw, settings = _with_deadlines(items, raw)
    single = {"probs": [1.0], "k": 0, "read_ms": 20.0, "p_ms": None, "read_prompt_n": None, "read_cache_n": None,
              "text_fallback": False, "miss": 0, "option_mass": 1.0}
    raw.rows += [{"phase": "A", "q": items[8].uid, "slot": 0, "probs": [1.0], "ms": 21.0, "miss": 0, "option_mass": 1.0},
                 {"phase": "B", "q": items[8].uid, "trace": None, "budgets": {"64": dict(single), "256": dict(single)}}]
    dout = deadline_outcomes(items, raw, 2000, outcomes(items, raw, [0, 64, 256]))
    read_at_d = [it for it in items if dout[it.uid].close not in (None, "budget0")]
    assert len(read_at_d) == 7 and dout[items[2].uid].close == "budget0" and dout[items[8].uid].close is None
    rows = [([*dout[it.uid].probs], it.label) for it in read_at_d]
    nll_T, ece_T = crossfit(rows, [it.cluster for it in read_at_d], settings.seed)
    fitted = {"T": fit_T(rows), "nll": nll_at(rows, 1.0), "nll_T": nll_T, "ece": ece_at(rows, 1.0), "ece_T": ece_T}
    e = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")["deadlines"]["2000"]
    assert {k: e[k] for k in fitted} == pytest.approx(fitted) and e["n"] == 9
    a = next(r for r in raw.rows if r["phase"] == "A" and r["q"] == items[2].uid)     # the fallback: confidently wrong
    n2 = len(items[2].question.keys)
    a["probs"] = _probs(n2, (items[2].label + 1) % n2, 0.99)
    moved = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")["deadlines"]["2000"]
    assert {k: moved[k] for k in fitted} == pytest.approx(fitted) and moved["acc"] == pytest.approx(e["acc"] - 1 / 9)


def test_a_single_option_question_is_natural_in_the_stored_closes_and_in_the_mapping_alike(tmp_path):
    """A single-option question (no trace) is answered at once under any deadline, so the
    stored `deadlines[D].closes` (shares of every question) and compare's mapping both count it natural, and the
    mapping still does where it re-simulates a deadline too short for a single token (D - r < ô)."""
    one = json.dumps({"state": "s", "questions": {"q": {"type": "choice", "instructions": "only", "criteria": {"o0": "x"},
                                                        "label": "o0"}}, "_meta": {"id": "rec-8", "group_id": "g4"}})
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8, extra=[one]))
    raw, _ = synthetic_raw(items[:8])
    raw, settings = _with_deadlines(items, raw)
    single = {"probs": [1.0], "k": 0, "read_ms": 20.0, "p_ms": None, "read_prompt_n": None, "read_cache_n": None,
              "text_fallback": False, "miss": 0, "option_mass": 1.0}
    raw.rows += [{"phase": "A", "q": items[8].uid, "slot": 0, "probs": [1.0], "ms": 21.0, "miss": 0, "option_mass": 1.0},
                 {"phase": "B", "q": items[8].uid, "trace": None, "budgets": {"64": dict(single), "256": dict(single)}}]
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    stored = prof["deadlines"]["2000"]["closes"]
    assert stored == pytest.approx({"natural": 5 / 9, "budget": 0.0, "deadline": 3 / 9, "eos": 0.0, "budget0": 1 / 9})
    m = mapping(prof, 2000)
    assert m["closes"] == pytest.approx({"natural": stored["natural"] + stored["eos"], "budget": stored["budget"],
                                         "deadline": stored["deadline"], "budget0": stored["budget0"]})
    tiny = mapping(prof, 50, reserve_ms=48.0)                        # D - r = 2 ms: no token fits, re-simulated
    assert tiny["closes"] == pytest.approx({"natural": 1 / 9, "budget": 0.0, "deadline": 0.0, "budget0": 8 / 9})


def test_a_deadline_every_question_fell_back_on_is_no_point_and_the_profile_says_why(tmp_path):
    """With no row read at T_D there is no T_D to fit, so the deadline is no point (never one
    served as calibrated at T = 1): no `deadlines[D]`, no `rows[].d[D]`, and the profile and the summary say why; serve
    then reads that deadline by rules 2 and 3. The run's other deadlines keep their points."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = _with_deadlines(items, synthetic_raw(items)[0], deadline=9000)
    settings = dataclasses.replace(settings, deadlines=(2000, 9000))
    for r in raw.rows:                                      # at 2000 ms every question falls back
        if r["phase"] == "B":
            r["deadlines"]["2000"] = {"close": "budget0", "k": 0, "chunks": 0, "elapsed_ms": 0.0}
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    assert set(prof["deadlines"]) == {"9000"} and prof["deadlines"]["9000"]["T"] > 0
    assert prof["deadline_skipped"] == {"2000": ct.DEADLINE_SKIPPED}
    # the reason covers both ways to read no question (all fell back, or all have one option)
    assert "budget 0" in ct.DEADLINE_SKIPPED and "one option" in ct.DEADLINE_SKIPPED
    assert all(set(r["d"]) == {"9000"} for r in prof["rows"])
    assert f"  deadline 2000 ms: {ct.DEADLINE_SKIPPED}" in summary_lines(prof)
    assert malformed(prof) is None and "m@2000ms" not in {p.label for p in points_of(prof, "m")}
    path = tmp_path / "m.think.json"
    path.write_text(json.dumps(prof, allow_nan=False))
    tp = load_think_profile(path)
    assert tp.deadline_point(2000) is None and tp.deadline_point(9000) is not None
    alone = dataclasses.replace(settings, deadlines=(2000,))                  # the run's only deadline: no block at all
    only = build_profile(items, raw, alone, sampling=SAMPLING, stats=STATS, name="m")
    assert not only.get("deadlines") and only["deadline_skipped"] == {"2000": ct.DEADLINE_SKIPPED}
    assert all("d" not in r or not r["d"] for r in only["rows"]) and malformed(only) is None


def test_a_deadline_whose_only_reads_are_fallbacks_and_single_option_questions_is_no_point(tmp_path):
    """A single-option question has no close at any deadline, so with every other question falling
    back the deadline still read no question at T_D; the skip reason names both cases."""
    one = json.dumps({"state": "s", "questions": {"q": {"type": "choice", "instructions": "only", "criteria": {"o0": "x"},
                                                        "label": "o0"}}, "_meta": {"id": "rec-8", "group_id": "g4"}})
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8, extra=[one]))
    raw, _ = synthetic_raw(items[:8])
    raw, settings = _with_deadlines(items, raw, deadline=9000)
    settings = dataclasses.replace(settings, deadlines=(2000, 9000))
    single = {"probs": [1.0], "k": 0, "read_ms": 20.0, "p_ms": None, "read_prompt_n": None, "read_cache_n": None,
              "text_fallback": False, "miss": 0, "option_mass": 1.0}
    raw.rows += [{"phase": "A", "q": items[8].uid, "slot": 0, "probs": [1.0], "ms": 21.0, "miss": 0, "option_mass": 1.0},
                 {"phase": "B", "q": items[8].uid, "trace": None, "budgets": {"64": dict(single), "256": dict(single)}}]
    for r in raw.rows:
        if r["phase"] == "B" and r["trace"] is not None:
            r["deadlines"]["2000"] = {"close": "budget0", "k": 0, "chunks": 0, "elapsed_ms": 0.0}
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    assert set(prof["deadlines"]) == {"9000"} and prof["deadline_skipped"] == {"2000": ct.DEADLINE_SKIPPED}


def test_the_profiles_no_pick_reason_never_hints_a_deadline_this_run_skipped(tmp_path):
    """build_profile passes its own `deadline_skipped` to the pick, so the no-pick reason does not
    send the user to a new run with a deadline this run measured and found every question falling back at."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = _with_deadlines(items, synthetic_raw(items)[0], deadline=2000)
    for r in raw.rows:
        if r["phase"] == "B":
            r["deadlines"]["2000"] = {"close": "budget0", "k": 0, "chunks": 0, "elapsed_ms": 0.0}
    pref = {"limits": (parse_limit("p99=0.5s"),), "min_accuracy": 0.9}       # budget 0 passes it, 64 and 256 do not
    no_dl = build_profile(items, raw, dataclasses.replace(settings, deadlines=(), **pref), sampling=SAMPLING,
                          stats=STATS, name="m")
    assert no_dl["pick"]["label"] is None and "--deadlines 500" in no_dl["pick"]["reason"]  # the positive control
    prof = build_profile(items, raw, dataclasses.replace(settings, **pref), sampling=SAMPLING, stats=STATS, name="m")
    assert prof["deadline_skipped"] == {"2000": ct.DEADLINE_SKIPPED} and prof["pick"]["label"] is None
    assert "--deadlines" not in prof["pick"]["reason"]


def test_the_deadline_point_reaches_the_pick_and_compare_and_extra_kinds_cannot_shadow_it(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, _ = synthetic_raw(items)
    raw, settings = _with_deadlines(items, raw)
    tight = dataclasses.replace(settings, limits=(parse_limit("max=1ms"),))
    prof = build_profile(items, raw, tight, sampling=SAMPLING, stats=STATS, name="m")
    assert prof["pick"]["label"] is None and "m@2000ms" in prof["pick"]["excluded"]     # it reached the picker
    path = tmp_path / "m.think.json"
    path.write_text(json.dumps(prof))
    pts = {p.label: p for p in points_of(load_profile(path), "m")}
    assert (pts["m@2000ms"].kind, pts["m@2000ms"].value, pts["m@2000ms"].cap) == ("deadline", 2000, 256)
    with pytest.raises(ValueError, match="deadline is the run's"):
        build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m", extra_kinds={"deadline": {}})
    with pytest.raises(ValueError, match="budget is the run's"):
        build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m", extra_kinds={"budget": {}})


def test_compare_validates_a_deadline_points_reserve_closes_and_cut(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, _ = synthetic_raw(items)
    raw, settings = _with_deadlines(items, raw)
    prof = json.loads(json.dumps(build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")))
    assert malformed(prof) is None
    for key, bad, why in (("reserve_ms", -1, "reserve_ms is not a finite number >= 0"),
                          ("reserve_ms", "51", "reserve_ms is not a finite number >= 0"),
                          ("closes", {"natural": 2.0}, "closes is not an object of close shares"),
                          ("closes", {"forced": 0.5}, "closes is not an object of close shares"),
                          ("k_p50", -3, "k_p50 is not null or a finite number >= 0")):
        broken = copy.deepcopy(prof)
        broken["deadlines"]["2000"][key] = bad
        assert (malformed(broken) or "").startswith(f"deadlines.2000.{why}"), key
    lean = copy.deepcopy(prof)
    for key in ("reserve_ms", "closes", "k_p50"):
        del lean["deadlines"]["2000"][key]
    assert malformed(lean) is None                                  # absent: a hand-made point, validated when present


def test_the_summary_lists_the_deadline_points_under_the_budgets_and_says_whose_marks_they_are(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, _ = synthetic_raw(items)
    raw, settings = _with_deadlines(items, raw)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    lines = summary_lines(prof)
    rows = [line for line in lines if line.startswith("m@")]
    assert [r.split()[0] for r in rows][-1] == "m@2000ms" and not rows[-1].rstrip().endswith("by")
    assert any(line.startswith("  marks: one frontier over budgets, deadline points and early exit") for line in lines)
    shown = frontier_marks(prof, "m", latency=prof["pick"]["latency"])  # compare's marks, every row
    for r in rows:
        q = shown[r.split()[0]]
        assert ("dominated by" in r) == (q.dominated_by is not None) and r.rstrip().endswith("≈") == bool(q.approx)
    plain = build_profile(items, synthetic_raw(items)[0], ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 64, 256),
                                                                       audit=6), sampling=SAMPLING, stats=STATS, name="m")
    assert not any(line.startswith("  marks") for line in summary_lines(plain))


def test_a_full_run_with_deadlines_writes_r_cal_the_points_and_their_rows(tmp_path):
    s = _Served(tmp_path, think_len=_len)
    try:
        data = write_data(tmp_path / "d.jsonl", 12)
        settings = _settings(data, deadlines=(400, 5000), warmup=3)
        out = calibrate_one(settings, url=s.url, name="fake", out_dir=tmp_path / "think", resume=False, log=s.log.append,
                            therm=lambda: 100)
        assert out.status == "done", out.reason
        prof = json.loads(out.profile.read_text())
        raw = read_raw(tmp_path / "think" / "fake.raw.jsonl")
    finally:
        s.stop()
    reads = raw.reserve["reads"]
    assert raw.reserve["source"] == "pre-pass" and len(reads) == 3
    assert raw.reserve["ms"] == pytest.approx(1.5 * max(reads[1:]) + 10.0)          # question 1's cold read unused
    assert any(line.startswith("deadlines: readout reserve r_cal") for line in s.log)
    assert prof["comparable"]["deadlines"] == [400, 5000]
    for d in ("400", "5000"):
        e = prof["deadlines"][d]
        assert e["cap"] == 256 and e["reserve_ms"] == raw.reserve["ms"] and e["n"] == 12 and e["T"] > 0
        assert sum(e["closes"].values()) == pytest.approx(1.0)
    assert prof["deadlines"]["5000"]["closes"]["natural"] == pytest.approx(4 / 12)
    assert prof["deadlines"]["400"]["closes"]["natural"] < 4 / 12 + 1e-9
    for r in (r for r in raw.rows if r["phase"] == "B"):             # k_D is the planner's own simulation at r_cal
        for d, e in r["deadlines"].items():
            sim = _sim(r["trace"], int(d), raw.reserve["ms"], 256)
            assert (e["close"], e["k"]) == (sim.close or "budget0", sim.k)
    assert all(set(row["d"]) == {"400", "5000"} and len(row["t"]) == 3 for row in prof["rows"])
    tp = load_think_profile(out.profile)
    assert sorted(tp.deadlines) == [400, 5000] and tp.latency["read_forced_ms"]["p50"] > 0
    assert {"fake@400ms", "fake@5000ms"} <= {p.label for p in points_of(load_profile(out.profile), "fake")}


def test_a_resumed_deadline_run_keeps_the_first_sessions_r_cal(tmp_path):
    """`r_cal` is fixed once per run (the raw file's `reserve` line); a resumed session warms up without a second
    pre-pass, and its phase-B cuts are simulated at the recorded reserve."""
    s = _Served(tmp_path, think_len=_len)
    try:
        data = write_data(tmp_path / "d.jsonl", 8)
        settings = _settings(data, deadlines=(400,), warmup=3)
        out_dir = tmp_path / "think"
        assert calibrate_one(settings, url=s.url, name="fake", out_dir=out_dir, resume=False, log=s.log.append,
                             therm=lambda: 100).status == "done"
        raw_path = out_dir / "fake.raw.jsonl"
        first = read_raw(raw_path).reserve
        lines = raw_path.read_text().splitlines()
        cut = next(i for i, line in enumerate(lines) if '"phase":"B"' in line) + 2      # two phase-B rows in
        raw_path.write_text("\n".join(lines[:cut]) + "\n")
        again = calibrate_one(settings, url=s.url, name="fake", out_dir=out_dir, resume=True, log=s.log.append,
                              therm=lambda: 100)
        assert again.status == "done", again.reason
        text = raw_path.read_text()
    finally:
        s.stop()
    raw = read_raw(raw_path)
    assert raw.reserve == first and text.count('{"reserve":') == 1
    assert all(e["reserve_ms"] == first["ms"] for e in json.loads(again.profile.read_text())["deadlines"].values())
    for r in (r for r in raw.rows if r["phase"] == "B"):
        assert (r["deadlines"]["400"]["close"], r["deadlines"]["400"]["k"]) == (
            (_sim(r["trace"], 400, first["ms"], 256).close or "budget0"), _sim(r["trace"], 400, first["ms"], 256).k)


def test_the_cli_flags_reach_the_settings_and_are_refused_alone(tmp_path, capsys):
    data = write_data(tmp_path / "d.jsonl", 2)
    parse = callo.cli.build_parser().parse_args
    s = settings_from_args(parse(["calibrate", "--think-budgets", "--data", str(data), "--deadlines", "2000,1000",
                                  "--think-deadline-reserve", "40"]))
    assert (s.deadlines, s.reserve_ms) == ((1000, 2000), 40)
    with pytest.raises(InvalidInput, match="needs --deadlines"):
        settings_from_args(parse(["calibrate", "--think-budgets", "--data", str(data), "--think-deadline-reserve", "40"]))
    with pytest.raises(InvalidInput, match="at most 4"):
        settings_from_args(parse(["calibrate", "--think-budgets", "--data", str(data), "--deadlines", "1,2,3,4,5"]))
    assert callo.cli.main(["calibrate", "--data", str(data), "--deadlines", "1000"]) == 2
    assert "--deadlines apply only with --think-budgets" in capsys.readouterr().err


def _single_first(tmp_path, n=12):
    """write_data's n questions plus a single-option one, which the run's order puts first (the warm-up's question 1)."""
    one = json.dumps({"state": "s", "questions": {"q": {"type": "choice", "instructions": "only", "criteria": {"o0": "x"},
                                                        "label": "o0"}}, "_meta": {"id": f"rec-{n}", "group_id": "g99"}})
    return write_data(tmp_path / "d.jsonl", n, extra=[one])


def test_a_single_option_warm_up_question_1_is_the_skipped_read_not_the_first_warm_one(tmp_path, monkeypatch):
    real = ct.shuffled
    monkeypatch.setattr(ct, "shuffled", lambda items, seed: sorted(real(items, seed),
                                                                   key=lambda it: len(it.question.keys) > 1))
    s = _Served(tmp_path, think_len=_len)
    try:
        settings = _settings(_single_first(tmp_path), deadlines=(400, 5000), warmup=3)
        out = calibrate_one(settings, url=s.url, name="fake", out_dir=tmp_path / "think", resume=False, log=s.log.append,
                            therm=lambda: 100)
        assert out.status == "done", out.reason
        raw = read_raw(tmp_path / "think" / "fake.raw.jsonl")
    finally:
        s.stop()
    reads = raw.reserve["reads"]
    assert reads[0] is None and len(reads) == 3 and all(r > 0 for r in reads[1:])
    assert raw.reserve["ms"] == pytest.approx(1.5 * max(reads[1:]) + 10.0)
    # raw phase-B rows store `deadlines` (rows[].d is the profile's); the positive control keeps
    # the negative from going vacuous again
    b_rows = [r for r in raw.rows if r.get("phase") == "B"]
    assert not any("deadlines" in r for r in b_rows if r["trace"] is None)
    assert any("deadlines" in r for r in b_rows if r["trace"] is not None)


def test_a_retried_phase_b_probe_with_deadlines_does_not_trip_foreign_s1(tmp_path, monkeypatch):
    """A retried request may have been answered twice, and a phase-B probe with deadlines reads
    each k_D that is no budget's cut too, so the guard's slack covers those readouts as well."""
    real_read = ct.ProbeClient.read
    seen = {"doubled": None}

    def read(self, item, budgets, seed, sampling, extras=None):
        if extras and extras.get("deadlines") and seen["doubled"] is None and _len(item.body["state"]) == 300:
            first = real_read(self, item, budgets, seed, sampling, extras)    # the dropped attempt the server answered
            seen["doubled"] = first.readouts
            return dataclasses.replace(real_read(self, item, budgets, seed, sampling, extras), retries=1)
        return real_read(self, item, budgets, seed, sampling) if extras is None else real_read(
            self, item, budgets, seed, sampling, extras)

    monkeypatch.setattr(ct.ProbeClient, "read", read)
    s = _Served(tmp_path, think_len=_len)
    try:
        settings = _settings(write_data(tmp_path / "d.jsonl", 8), deadlines=(300, 500, 700, 900), warmup=1)
        out = calibrate_one(settings, url=s.url, name="fake", out_dir=tmp_path / "think", resume=False, log=s.log.append,
                            therm=lambda: 100)
    finally:
        s.stop()
    assert seen["doubled"] is not None and seen["doubled"] > len(settings.budgets)    # more than the old bound, 2 + 1
    assert out.status == "done", out.reason


def test_a_retried_reserve_pre_pass_read_does_not_trip_foreign_s1(tmp_path, monkeypatch):
    """The warm-up's reserve pre-pass reads a natural trace at its n_gen - 1 pre-cut as well, so a
    retried pre-pass probe may have been answered with a readout for every budget AND the pre-cut; the guard's slack
    must cover them all, or the run fails on foreign S1 traffic. A 40-token natural trace under budgets 0,16,32,64
    is the most a pre-pass reads: the trace is capped at the largest budget, so its n_gen read is always a budget's.
    Mutation-checked: dropping both the slack's `+ 1` and its `bool(ex.get("reserve_read"))` turns this red; either
    alone still covers the one pre-cut (the pre-pass reads at most len(budgets) + 1, the slack is len(budgets) + 2)."""
    real_read = ct.ProbeClient.read
    seen = {"doubled": None}

    def read(self, item, budgets, seed, sampling, extras=None):
        if extras and extras.get("reserve_read") and seen["doubled"] is None and _len(item.body["state"]) == 40:
            first = real_read(self, item, budgets, seed, sampling, extras)    # the dropped attempt the server answered
            seen["doubled"] = first.readouts
            return dataclasses.replace(real_read(self, item, budgets, seed, sampling, extras), retries=1)
        return real_read(self, item, budgets, seed, sampling) if extras is None else real_read(
            self, item, budgets, seed, sampling, extras)

    monkeypatch.setattr(ct.ProbeClient, "read", read)
    s = _Served(tmp_path, think_len=_len)
    try:
        settings = _settings(write_data(tmp_path / "d.jsonl", 8), budgets=(0, 16, 32, 64), deadlines=(400,), warmup=4)
        out = calibrate_one(settings, url=s.url, name="fake", out_dir=tmp_path / "think", resume=False, log=s.log.append,
                            therm=lambda: 100)
    finally:
        s.stop()
    assert seen["doubled"] == len(settings.budgets) + 1      # budget 0, the cuts at 16 and 32, n_gen 40 and the pre-cut
    assert out.status == "done", out.reason


@pytest.mark.parametrize("strip", ["deadlines", "reserve_read"])
def test_a_server_that_ignores_the_deadline_probe_keys_fails_the_run_loudly(tmp_path, monkeypatch, strip):
    """An older callo serve drops the keys it does not know, so a multi-option reply without
    `deadlines` (phase B) or `reserve_read` (the pre-pass) is a failed run naming the fix, never a run without points."""
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
        settings = _settings(write_data(tmp_path / "d.jsonl", 8), deadlines=(400,), warmup=2)
        out = calibrate_one(settings, url=s.url, name="fake", out_dir=tmp_path / "think", resume=False, log=s.log.append,
                            therm=lambda: 100)
    finally:
        s.stop()
    assert out.status == "failed"
    assert "unusable reply: the server does not support --deadlines probes; upgrade callo serve" in out.reason
