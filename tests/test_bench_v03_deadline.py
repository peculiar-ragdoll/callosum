"""The v0.3 think-deadline acceptance bench, GPU-free: the served-answer correctness, the session drivers through a
fake `lib.post`, the served sessions' launches (callo as shipped: cache on), every gate on synthetic evidence
(pass, fail and the missing-evidence ERROR), C1's pre-registered build rule on synthetic converge blocks, the report and
the gates-only run of `main`."""
import json
import os

import pytest
from test_calibrate_think import write_data

from bench import acceptance_v03_deadline as v03d
from bench import acceptance_v03_think as v03t
from bench.report_v03_deadline import GATES, render
from callo.calibrate_think import load_data


@pytest.fixture(autouse=True)
def _no_real_provenance(monkeypatch):
    """`main` records the session's provenance (code, llama-server --version, disk): stubbed here, so no test runs git
    or a llama-server binary; the provenance test re-stubs them with its own values."""
    monkeypatch.setattr(v03t, "code_state", lambda root: {"root": str(root), "commit": "0" * 40, "dirty": False,
                                                          "status": []})
    monkeypatch.setattr(v03t, "_llama_server_version", lambda binary: "version: 1 (4ebdf2c)")
    monkeypatch.setattr(v03t, "disk_free_gib", lambda paths: {p: 100.0 for p in paths})


def _entry(close, elapsed, d, k=100, gen_ms=1000.0, prompt_n=60, fallback=None):
    """One served X-Callo-Timings entry under a deadline. A `budget0` fallback carries no think fields (serve stamps
    only the deadline fields and `deadline_fallback` on it), as `proxy._think_fields` writes it."""
    if fallback:
        return {"q": "x", "status": 200, "correct": True,
                "entry": {"think_deadline_ms": d, "elapsed_ms": elapsed, "deadline_met": elapsed <= d,
                          "deadline_fallback": fallback, **({"chunks": 0} if fallback == "budget0" else {})}}
    e = {"think_close": close, "elapsed_ms": elapsed, "think_deadline_ms": d, "think_tokens": k, "gen_ms": gen_ms,
         "gen_prompt_n": prompt_n}
    return {"q": "x", "status": 200, "entry": e, "correct": True}


def test_the_bench_writes_only_under_its_own_results_dir_and_calibrates_as_shipped(tmp_path):
    assert v03d.OUT_DEFAULT.endswith("/callo_v0.3/deadline") and "/think" not in v03d.OUT_DEFAULT
    assert v03d.CALIB["tiel"][2][v03d.CALIB["tiel"][2].index("--converge-probe") + 1] == "32"
    assert v03d.CALIB["qwen9b"][2][v03d.CALIB["qwen9b"][2].index("--converge-probe") + 1] == "64"
    assert "temp=0" in v03d.CALIB["tiel-greedy"][2]
    for name in v03d.CALIB:                     # shipped defaults, cache on (calibrate's default): no --no-cache
        argv = v03d.calib_argv(name, tmp_path, tmp_path / "d.jsonl")
        assert "--no-cache" not in argv and "--resume" not in argv
        assert argv[argv.index("--out") + 1] == str(tmp_path / "calib" / name)
    (tmp_path / "calib" / "tiel").mkdir(parents=True)
    (tmp_path / "calib" / "tiel" / "run.json").write_text("{}")
    assert "--resume" in v03d.calib_argv("tiel", tmp_path, tmp_path / "d.jsonl")


def test_served_correct_reads_choice_and_noul_answers(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 4))
    it = items[0]
    keys = list(it.question.keys)
    probs = {k: (0.9 if i == it.label else 0.1 / (len(keys) - 1)) for i, k in enumerate(keys)}
    assert v03d.served_correct(it, {"type": "choice", "probabilities": probs}) is True
    wrong = {k: (0.0 if i == it.label else 1.0 / (len(keys) - 1)) for i, k in enumerate(keys)}
    assert v03d.served_correct(it, {"type": "choice", "probabilities": wrong}) is False
    assert v03d.served_correct(it, None) is None and v03d.served_correct(it, {"probabilities": {}}) is None


def test_serve_items_sends_the_deadline_or_budget_and_keeps_each_entry(tmp_path, monkeypatch):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 3))
    sent = []

    def post(url, body):
        sent.append((url, body))
        qid = next(iter(body["questions"]))
        keys = items[0].question.keys
        return 200, {"x-callo-timings": json.dumps([{"think_close": "deadline", "elapsed_ms": 990.0}])}, \
            {"answers": {qid: {"type": "choice", "probabilities": {k: 1.0 / len(keys) for k in keys}}}}

    monkeypatch.setattr(v03d.lib, "post", post)
    rows = v03d.serve_items("http://h", items, deadline_ms=1000)
    assert [b["think_deadline_ms"] for _, b in sent] == [1000] * 3 and all("think_budget" not in b for _, b in sent)
    assert rows[0]["entry"]["think_close"] == "deadline" and rows[0]["q"] == items[0].uid
    sent.clear()
    v03d.serve_items("http://h", items[:1], deadline_ms=0, budget=1024)          # budget only: deadline off explicitly
    assert sent[0][1]["think_budget"] == 1024 and sent[0][1]["think_deadline_ms"] == 0
    assert sent[0][0] == "http://h/v1/systemone"

    def down(url, body):
        raise ConnectionRefusedError("down")

    monkeypatch.setattr(v03d.lib, "post", down)
    (row,) = v03d.serve_items("http://h", items[:1], deadline_ms=1000)
    assert {k: row[k] for k in ("q", "status", "entry", "correct")} == {"q": items[0].uid, "status": -1, "entry": None,
                                                                          "correct": None}
    assert row["wall_ms"] >= 0.0                              # the client's own clock: G1's elapsed for a 504


def test_g3_pairs_probes_live_then_the_budget_at_the_live_cut(tmp_path, monkeypatch):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 2))
    calls = []

    def post(url, body):
        calls.append(body)
        qid = next(iter(body["request"]["questions"]))
        if "live" in body:
            close = "budget0" if len(calls) > 2 else "deadline"
            return 200, {}, {"questions": {qid: {"live": {"close": close, "n_gen": 3, "tokens": [1, 2, 3], "chunks": [],
                                                          "readout": {"probs": [0.5, 0.5]}}}}}
        return 200, {}, {"questions": {qid: {"trace": {"tokens": [1, 2, 3, 4]}, "budgets": {"3": {"probs": [0.5, 0.5]}}}}}

    monkeypatch.setattr(v03d.lib, "post", post)
    pairs = v03d.g3_pairs("http://h", items, cap=1024)
    assert calls[0]["live"] == {"deadline_ms": 1500, "probe_every": None} and calls[0]["budgets"] == [1024]
    assert calls[1]["budgets"] == [3] and "live" not in calls[1]
    assert pairs[0]["budget"]["tokens"] == [1, 2, 3] and pairs[1]["budget"] is None          # budget0: no pair


def test_g1_passes_on_tight_cuts_and_fails_on_a_late_tail_or_a_fallback_at_2000():
    ok = {d: [_entry("deadline", d + 10, d) for _ in range(70)] + [_entry("natural", 400, d)] for d in (1000, 2000)}
    g = v03d.g1(ok)
    assert g["pass"] is True and g["deadlines"]["1000"]["n_cut"] == 70 and g["deadlines"]["1000"]["max_over_ms"] == 10
    late = {d: [_entry("budget", d + (80 if i < 2 else 0), d) for i in range(70)] for d in (1000, 2000)}
    assert v03d.g1(late)["pass"] is False                                    # p99 of 70 is the 70th: the late tail
    fb = {d: [*rows, _entry(None, 3, d, fallback="budget0")] for d, rows in ok.items()}
    g = v03d.g1(fb)
    assert g["pass"] is False and g["deadlines"]["1000"]["pass"] and not g["deadlines"]["2000"]["pass"]
    assert g["deadlines"]["2000"]["fallbacks"] == 1
    few = {d: [_entry("deadline", d, d) for _ in range(59)] for d in (1000, 2000)}
    g = v03d.g1(few)
    assert g["pass"] is None and "not testable" in g["deadlines"]["1000"]["error"]
    g = v03d.g1({1000: late[1000], 2000: few[2000]})                          # a measured FAIL is never hidden by an ERROR
    assert g["pass"] is False
    assert v03d.g1({1000: ok[1000]})["pass"] is None                          # a missing session is ERROR


def _pair(same=True, agree=True, dp=1e-4, first=4, conts=(1, 1, 1)):
    live = {"tokens": [1, 2, 3], "readout": {"probs": [0.6, 0.4], "read_prompt_n": first},
            "chunks": [{"prompt_n": 60}, *({"prompt_n": c} for c in conts)]}
    b = [0.6 - dp, 0.4 + dp] if agree else [0.4, 0.6]
    return {"q": "x", "live": live, "budget": {"tokens": [1, 2, 3] if same else [1, 2, 4], "readout": {"probs": b}}}


def test_g3_counts_identical_traces_argmax_dp_and_the_prompt_n_bounds():
    assert v03d.g3([_pair() for _ in range(48)])["pass"] is True
    g = v03d.g3([_pair(same=False)] * 3 + [_pair()] * 45)
    assert g["pass"] is False and g["identical"] == 45                       # n - 3 identical
    assert v03d.g3([_pair(agree=False)] * 2 + [_pair()] * 46)["pass"] is False
    assert v03d.g3([_pair(dp=2e-3) for _ in range(48)])["pass"] is False
    assert v03d.g3([_pair(conts=(1, 1, 2))] + [_pair()] * 47)["pass"] is True  # 1 of 144 continuations: >= 99 %
    assert v03d.g3([_pair(conts=(2, 2, 1))] + [_pair()] * 47)["pass"] is False # 2 of 144: < 99 %
    assert v03d.g3([_pair(first=5)] + [_pair()] * 47)["pass"] is False        # 1 of 48 first passes: < 99 %
    assert v03d.g3([{"q": "x", "live": None, "budget": None}])["pass"] is None


def test_g4_reads_the_overhead_and_compares_think_rates():
    cut = [_entry("deadline", 1000, 1000, k=90, gen_ms=1000.0 + 60 * 0.5)]
    base = [_entry("budget", None, None, k=100, gen_ms=1000.0 + 60 * 0.5)]
    g = v03d.g4({"chunk_overhead_ms": {"p50": 6.0}}, cut, base, 0.5)
    assert g["pass"] is True and g["ratio"] == pytest.approx(0.9)
    assert v03d.g4({"chunk_overhead_ms": {"p50": 16.0}}, cut, base, 0.5)["pass"] is False
    slow = [_entry("deadline", 1000, 1000, k=80, gen_ms=1030.0)]
    assert v03d.g4({"chunk_overhead_ms": {"p50": 6.0}}, slow, base, 0.5)["pass"] is False
    assert "chunk_overhead_ms" in v03d.g4(None, cut, base, 0.5)["error"]
    assert "prefill" in v03d.g4({"chunk_overhead_ms": {"p50": 6.0}}, cut, base, None)["error"]


def test_g5_compares_the_natural_share_over_every_question_and_the_cut_k_with_the_greedy_point():
    point = {"deadlines": {"2000": {"closes": {"natural": 0.5, "eos": 0.0}, "k_p50": 100}}}
    rows = [_entry("natural", 500, 2000)] * 50 + [_entry("deadline", 2000, 2000, k=104)] * 50
    g = v03d.g5(point, rows)
    assert g["pass"] is True and g["natural_served"] == 0.5 and g["k_p50_served"] == 104 and g["n"] == 100
    assert v03d.g5(point, [_entry("natural", 500, 2000)] * 54 + [_entry("deadline", 2000, 2000)] * 46)["pass"] is False
    assert v03d.g5(point, [_entry("natural", 500, 2000)] * 50 + [_entry("deadline", 2000, 2000, k=106)] * 50)["pass"] is False
    # the point's closes are over every question: a budget0 fallback counts in the served denominator too
    g = v03d.g5(point, [_entry("natural", 500, 2000)] * 50 + [_entry("deadline", 2000, 2000)] * 49
                + [_entry(None, 3, 2000, fallback="budget0")])
    assert g["n"] == 100 and g["natural_served"] == 0.5
    assert v03d.g5({}, rows)["pass"] is None


def _g6_profile(n=10):
    return {"budgets": {"0": {}, "256": {}}, "row_fields": ["correct", "nll1", "pmax", "lat_ms", "k", "close"],
            "rows": [{"q": f"r{i}/q", "cluster": f"r{i}", "b": {"0": [True, 0, 0, 0, 0, None],
                                                                 "256": [True, 0, 0, 0, 0, "natural"]},
                      "d": {"2000": [i % 2 == 0, 0, 0, 1000.0 + 100 * i, 0, "deadline"]}} for i in range(n)],
            "deadlines": {"2000": {"lat_p50_ms": [1400.0, 1300.0, 1500.0], "lat_p95_ms": [1900.0, 1800.0, 1950.0]}}}


def test_g6_is_descriptive_pairs_by_question_and_sets_served_latency_beside_the_point_s():
    profile = _g6_profile()
    rows = [{"q": f"r{i}/q", "correct": True, "entry": {"elapsed_ms": 1100.0 + 100 * i}} for i in range(10)]
    g = v03d.g6(profile, rows)
    assert g["pass"] is None and g["descriptive"] and g["n"] == 10 and g["dacc"][0] == pytest.approx(0.5)
    lat = g["latency"]                                                        # G6: the claimed 1-4 % understatement of deadline latency, checked
    assert lat["served_p50_ms"] == 1500.0 and lat["served_p95_ms"] == 2000.0     # nearest rank over the 10 served
    assert lat["point_same_p50_ms"] == 1400.0 and lat["point_same_p95_ms"] == 1900.0
    assert lat["point_p50_ms"] == 1400.0 and lat["point_p95_ms"] == 1900.0       # the profile's own (all questions)
    assert lat["served_vs_point_p50"] == pytest.approx(1500.0 / 1400.0 - 1)
    assert lat["n"] == 10
    assert "no question" in v03d.g6(profile, [{"q": "zz", "correct": True}])["error"]


def _cv(*, gains=False, by_rho=None, rho_max=None, prefix="absent", c_ms=30.0):
    rho: dict = {"rho_max": rho_max}
    if by_rho is not None:
        rho["by_rho"] = by_rho
    if prefix != "absent":
        rho["rho_prefix"] = prefix
    return {"K": 32, "n": 376, "cost": {"c_ms": c_ms, "restore_share": 1.0}, "flips": {"changed_after_confident": 3},
            "sequence": {"0.9": [{"verdict": "pass"}]}, "rules": {"0.9": None},
            "gain": {"0.9": {"gains": gains}}, "rho": rho, "rho_max_per_token": None}


GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)


def _by_rho(gaining):
    return {f"{r:g}": ([0.9] if r in gaining else []) for r in GRID}


def test_c1_builds_on_a_gain_at_a_measured_model_s_own_probe_cost():
    g = v03d.c1({"tiel": {"converge": _cv(gains=True, by_rho=_by_rho(()), prefix=None)},
                 "qwen9b": {"converge": _cv(by_rho=_by_rho(()), prefix=None)}})
    assert g["build"] is True and g["models"]["tiel"]["gains"] is True and g["models"]["qwen9b"]["gains"] is False
    assert g["models"]["tiel"]["sequence"] == {"0.9": ["pass"]}
    assert g["rule"].startswith("build = early exit gains on >= 1 measured model")
    assert "(pre-registered on 2026-10-09, before any data)" in g["rule"]


def test_c1_builds_when_gains_hold_from_the_cheap_end_and_slow_decode_counts():
    only_slow = _cv(by_rho=_by_rho((0.5,)), rho_max=0.5, prefix=0.5)
    g = v03d.c1({"tiel": {"converge": only_slow}, "qwen9b": {"converge": _cv(by_rho=_by_rho(()), prefix=None)}})
    assert g["build"] is True                                                # rho 0.5 alone, from the cheap end: build
    m = g["models"]["tiel"]
    assert (m["rho_prefix"], m["rho_max"], m["consistent"]) == (0.5, 0.5, True)
    assert m["regimes"] == {"slow decode": "yes", "pure attention": "no", "ρ 2": "no", "ρ 3": "no"}


def test_c1_does_not_build_on_an_isolated_gain_that_misses_the_cheap_end():
    isolated = _cv(by_rho=_by_rho((2.0, 3.0, 4.0)), rho_max=4.0, prefix=None)  # rho_max >= 1, but 0.5 and 1 do not gain
    g = v03d.c1({"tiel": {"converge": isolated}, "qwen9b": {"converge": _cv(by_rho=_by_rho(()), prefix=None)}})
    assert g["build"] is False
    m = g["models"]["tiel"]
    assert m["rho_max"] == 4.0 and m["rho_prefix"] is None and m["consistent"] is False
    assert m["regimes"] == {"slow decode": "no", "pure attention": "no", "ρ 2": "isolated", "ρ 3": "isolated"}


def test_c1_derives_the_prefix_numerically_when_the_profile_predates_rho_prefix():
    """The lexical-sort trap: by_rho's keys are ':g' strings, so "16" < "2" lexically. Gains at 0.5..8 but not 16 or
    32: a lexical walk stops at "16" (prefix 1); the numeric walk reaches 8."""
    cv = _cv(by_rho=_by_rho((0.5, 1.0, 2.0, 3.0, 4.0, 8.0)))              # no rho_prefix, no rho_max stored
    m = v03d.c1({"tiel": {"converge": cv}})["models"]["tiel"]
    assert m["rho_prefix"] == 8.0 and m["rho_max"] == 8.0 and m["prefix_source"] == "derived"
    assert m["regimes"] == {"slow decode": "yes", "pure attention": "yes", "ρ 2": "yes", "ρ 3": "yes"}
    assert sorted(cv["rho"]["by_rho"]) == ["0.5", "1", "16", "2", "3", "32", "4", "8"]   # the trap is real
    cv = _cv(by_rho={"16": [0.9], "2": [], "0.5": [0.9], "1": [0.9]})         # gains at 16, not 2: prefix 1, not 16
    m = v03d.c1({"tiel": {"converge": cv}})["models"]["tiel"]
    assert m["rho_prefix"] == 1.0 and m["rho_max"] == 16.0
    stored = _cv(by_rho=_by_rho((0.5, 1.0, 2.0)), rho_max=2.0, prefix=2.0)
    assert v03d.c1({"tiel": {"converge": stored}})["models"]["tiel"]["prefix_source"] == "profile"


def test_c1_reports_missing_evidence_and_never_decides_no_on_it():
    none = _cv(by_rho=_by_rho(()), prefix=None)
    g = v03d.c1({"tiel": {"converge": none}, "qwen9b": {}})
    assert g["build"] is None and "qwen9b" in g["error"]
    g = v03d.c1({"tiel": {"converge": _cv(gains=True, by_rho=_by_rho(()), prefix=None)}, "qwen9b": {}})
    assert g["build"] is True and "qwen9b" in g["error"]                     # the OR is met: the missing model is noted
    no_cost = _cv(c_ms=None, by_rho=_by_rho(()), prefix=None)                 # --converge-cost 0: no measured gain test
    g = v03d.c1({"tiel": {"converge": no_cost}})
    assert g["models"]["tiel"]["gains"] is None and g["build"] is None
    no_rho = {**_cv(), "rho": None}                                           # no decode rate: no rho map
    g = v03d.c1({"tiel": {"converge": no_rho}})
    assert g["models"]["tiel"]["consistent"] is None and g["build"] is None
    assert g["models"]["tiel"]["regimes"] == {"slow decode": "n/a", "pure attention": "n/a", "ρ 2": "n/a", "ρ 3": "n/a"}


def test_the_c1_report_prints_rho_max_the_regime_table_and_the_rule():
    g = v03d.c1({"tiel": {"converge": _cv(by_rho=_by_rho((2.0,)), rho_max=2.0, prefix=None)},
                 "qwen9b": {"converge": _cv(by_rho=_by_rho((0.5, 1.0, 2.0)), rho_max=2.0, prefix=2.0)}})
    text = render({"updated": "t", "gates": {"c1": g}})
    assert "| qwen9b | yes | yes | yes | no |" in text and "| tiel | no | no | isolated | no |" in text
    assert "| model | ρ 0.5 (slow decode) | ρ 1 (pure attention ≈ 1–3) | ρ 2 | ρ 3 |" in text
    assert f"Build rule: {v03d.BUILD_RULE}" in text and "build the early-exit serve side" in text
    assert "| qwen9b | 32 | 376 |" in text and "| 2 | 2 |" in text            # rho_max and rho_prefix columns
    assert "| 2 | none |" in text                                             # tiel: the cheapest rho does not gain
    assert "build the early-exit serve side (qwen9b: gains at every grid ρ from 0.5 up to 2)" in text
    assert g["basis"] == ["qwen9b: gains at every grid ρ from 0.5 up to 2"]


def test_the_report_has_one_row_per_gate_and_never_passes_missing_evidence():
    s = {"updated": "t", "gates": {"g1": v03d.g1({}), "g3": v03d.g3([]),
                                   "g2": {"pass": None, "error": "the game-client agent pilot has not run (--g2)"}}}
    text = render(s)
    for name, _, _ in GATES:
        assert f"| {name} |" in text
    assert "| ERROR |" in text and "| PASS |" not in text and "not run" in text
    import re
    rows = [ln for ln in text.splitlines() if ln.startswith("| G") or ln.startswith("| C1 |")]
    assert len(rows) == len(GATES)
    assert all(len(re.findall(r"(?<!\\)\|", ln)) == 5 for ln in rows)     # 4 cells: no stray pipe (|Δp| is escaped)


def test_the_report_sets_g6_s_latency_line_and_g2_s_context_beside_their_results():
    profile = _g6_profile()
    rows = [{"q": f"r{i}/q", "correct": True, "entry": {"elapsed_ms": 1100.0 + 100 * i}} for i in range(10)]
    g2 = {"pass": True, "n_calls": 1300, "n_cut": 250, "p99_over_ms": 40.0, "p995_over_ms": 60.0, "max_over_ms": 90.0,
          "closes": {"deadline": 200, "natural": 50, "budget0": 3}, "think_tokens": {"1": {"p5": 10, "p50": 60, "p95": 90}},
          "rerouted": 2, "draining": 0}
    text = render({"updated": "t", "gates": {"g6": v03d.g6(profile, rows), "g2": g2}})
    assert "served elapsed p50/p95 1500/2000 ms vs the point's 1400/1900 ms on the same 10 questions (+7.1 %/+5.3 %)" in text
    assert "closes deadline 200, natural 50, budget0 3" in text and "rerouted 2, draining 0" in text


class _Session:
    def __init__(self, name, flags):
        self.name, self.flags, self.url, self.upstream = name, list(flags), "http://127.0.0.1:1", None
        self.finished = False

    def stats(self):
        return {"think": {"deadline": {"chunk_overhead_ms": {"p50": 5.0}, "reserve_ms": 52.0, "reserve_source": "flag"}}}

    def finish(self):
        self.finished = True
        return 0


def _write_profile(path, *, deadlines=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"budgets": {"0": {}, "256": {}, "1024": {}}, "provenance": {"server_args": ["--cache-ram", "0"]},
                                **({"deadlines": deadlines} if deadlines else {})}))
    return path


def test_run_serve_launches_callo_as_shipped_with_the_profile_s_server_args(tmp_path, monkeypatch):
    """The served sessions run with callo's shipped defaults (cache on), as calibrate did,
    so G5/G6 compare like with like; each repeats its profile's llama-server passthrough; the greedy session fixes the
    point's reserve."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 6))
    tiel = _write_profile(tmp_path / "calib" / "tiel" / "Tiel.think.json")
    greedy = _write_profile(tmp_path / "calib" / "tiel-greedy" / "Tiel.think.json", deadlines={"2000": {"reserve_ms": 51.4}})
    launches, sessions = [], []

    def launch(name, out, *, model, flags, ctx, cache=False):
        launches.append({"name": name, "flags": list(flags), "cache": cache, "model": model, "ctx": ctx})
        sessions.append(_Session(name, flags))
        return sessions[-1]

    bodies = []

    def post(url, body):
        bodies.append((url, body))
        if url.endswith("/callo/think-probe"):
            qid = next(iter(body["request"]["questions"]))
            return 200, {}, {"questions": {qid: {"live": {"close": "budget0"}}}}
        qid = next(iter(body["questions"]))
        return 200, {"x-callo-timings": json.dumps([{"think_close": "natural", "elapsed_ms": 10.0}])}, \
            {"answers": {qid: {"type": "choice", "probabilities": {k: 0.25 for k in items[0].question.keys}}}}

    monkeypatch.setattr(v03t, "_launch", launch)
    monkeypatch.setattr(v03d.lib, "post", post)
    res: dict = {}
    v03d.run_serve(tmp_path, items, res)
    assert [x["name"] for x in launches] == ["serve-tiel", "serve-greedy"]
    assert all(x["cache"] is True for x in launches)                           # cache on, as shipped
    assert launches[0]["flags"] == ["--think-profile", str(tiel), "--", "--cache-ram", "0"]
    assert launches[1]["flags"] == ["--think-profile", str(greedy), "--think-deadline-reserve", "52", "--",
                                    "--cache-ram", "0"]
    assert all(s.finished for s in sessions) and res["cache"] is True
    warm = v03d.WARMUP_N
    assert len(res["g1"]["1000"]) == len(items) and len(res["g1"]["2000"]) == len(items)
    assert len(res["g4_budget"]) == min(v03d.G3_N, len(items)) and len(res["g5"]) == len(items)
    sys1 = [b for u, b in bodies if u.endswith("/v1/systemone")]
    assert len(sys1) == 2 * warm + 2 * len(items) + min(v03d.G3_N, len(items)) + len(items)
    assert res["g5_reserve"] == {"point": 51.4, "served": 52.0, "source": "flag"}


def test_main_runs_the_gates_part_without_a_gpu_and_records_g2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(v03d.lib, "running_gpu_servers", lambda: pytest.fail("gates never look for a GPU server"))
    g2 = tmp_path / "g2.json"
    g2.write_text(json.dumps({"pass": True, "n_calls": 1300, "n_cut": 250, "p99_over_ms": 40.0, "p995_over_ms": 60.0,
                              "max_over_ms": 90.0}))
    doc = tmp_path / "docs" / "doc.md"
    assert v03d.main(["--out", str(tmp_path / "out"), "--parts", "gates", "--g2", str(g2), "--doc", str(doc)]) == 0
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert s["gates"]["g2"]["pass"] is True and s["gates"]["g1"]["pass"] is None
    assert "| G2 |" in doc.read_text() and not (tmp_path / "out" / "run.pid").exists()
    printed = capsys.readouterr().out.splitlines()
    assert "G2: PASS" in printed and "G1: ERROR" in printed and printed[-1].startswith("deadline bench: done, rc 0")
    # a rerun of the gates without --g2 keeps the recorded G2 (a not-testable one included)
    assert v03d.main(["--out", str(tmp_path / "out"), "--parts", "gates", "--doc", str(doc)]) == 0
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["gates"]["g2"]["pass"] is True


def test_main_refuses_a_live_pid_file_an_unknown_part_and_a_busy_gpu(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "run.pid").write_text(f"{os.getppid()}\n")
    monkeypatch.setattr(v03d, "pid_owner", lambda path: os.getppid())
    assert v03d.main(["--out", str(out), "--parts", "gates"]) == 2
    assert v03d.main(["--out", str(out), "--parts", "nope"]) == 2
    monkeypatch.setattr(v03d, "pid_owner", lambda path: None)
    monkeypatch.setattr(v03d.lib, "running_gpu_servers", lambda: [(123, "llama-server -m x.gguf")])
    data = write_data(tmp_path / "d.jsonl", 2)
    for parts in ("calib", "serve", "calib,serve,gates"):
        assert v03d.main(["--out", str(out), "--parts", parts, "--data", str(data)]) == 2
    assert "a model server is already running" in capsys.readouterr().out

    def fails():
        raise RuntimeError("ps failed")

    monkeypatch.setattr(v03d.lib, "running_gpu_servers", fails)
    assert v03d.main(["--out", str(out), "--parts", "serve", "--data", str(data)]) == 2
    assert "cannot tell" in capsys.readouterr().out
    monkeypatch.setattr(v03d.lib, "running_gpu_servers", lambda: [])
    assert v03d.main(["--out", str(out), "--parts", "calib", "--data", str(tmp_path / "missing.jsonl")]) == 2
    assert not (out / "summary.json").exists()


def test_the_greedy_session_serves_at_the_deadline_point_s_reserve(tmp_path):
    """G5 compares like with like: the served reserve is fixed at the point's r_cal (whole ms, rounded up)."""
    p = {"deadlines": {"2000": {"reserve_ms": 51.4}}}
    assert v03d.greedy_flags(tmp_path / "g.think.json", p) == ["--think-profile", str(tmp_path / "g.think.json"),
                                                               "--think-deadline-reserve", "52"]
    assert v03d.greedy_flags(tmp_path / "g.think.json", {}) == ["--think-profile", str(tmp_path / "g.think.json")]


def test_the_report_header_lists_the_documented_deviations_from_section_10():
    """The served sessions' warm-ups and the --cache-ram 0 passthrough are documented
    deviations from the gates' original wording, named in the doc's header."""
    text = render({"updated": "t", "parts": {"serve": {"warmup": 3}}, "gates": {}})
    assert "## Deviations from the gates' original wording" in text
    assert "3 discarded warm-up requests" in text and "`--cache-ram 0`" in text
    assert "## Deviations from the gates' original wording" in render({"updated": "t", "gates": {}})   # before the serve part too


# ---- failed requests, failed probes, measured overheads, C1 verdicts, reruns ----------------------------------------

def _failed(status, wall_ms=None):
    return {"q": "x", "status": status, "entry": None, "correct": None, **({"wall_ms": wall_ms} if wall_ms else {})}


def test_g1_counts_a_504_as_a_cut_over_d_so_a_timed_out_tail_never_passes():
    """70 tight cuts and ten 504s (ThinkTimeout) at each D must not PASS: a 504 is a cut over D
    at its measured elapsed (the client's wall time), else +inf."""
    rows = {d: [_entry("deadline", d + 10, d) for _ in range(70)] + [_failed(504)] * 10 for d in (1000, 2000)}
    g = v03d.g1(rows)
    assert g["pass"] is False and g["deadlines"]["1000"]["timeouts"] == 10 and g["deadlines"]["1000"]["n_cut"] == 80
    assert g["deadlines"]["1000"]["max_over_ms"] == float("inf")
    assert "max inf ms, budget0 0, 504 10, failed 0 | FAIL |" in render({"updated": "t", "gates": {"g1": g}})
    timed = {d: [_entry("deadline", d + 10, d) for _ in range(99)] + [_failed(504, wall_ms=d + 5100.0)] for d in (1000, 2000)}
    g = v03d.g1(timed)                       # one measured 504 of 100 cuts: p99 is the 99th, the 504 is the max
    assert g["deadlines"]["1000"]["max_over_ms"] == pytest.approx(5100.0) and g["pass"] is True


def test_g1_is_an_error_when_requests_failed_otherwise():
    rows = {d: [_entry("deadline", d + 10, d) for _ in range(70)] + [_failed(500), _failed(-1)] for d in (1000, 2000)}
    g = v03d.g1(rows)
    assert g["deadlines"]["1000"]["failed"] == 2 and g["deadlines"]["1000"]["error"] == "2 requests failed"
    from bench.report_v03_deadline import verdict
    assert g["pass"] is not False and verdict("g1", g) == "ERROR"


def test_g3_is_an_error_on_a_failed_probe_and_not_testable_below_24():
    from bench.report_v03_deadline import verdict
    g = v03d.g3([_pair() for _ in range(30)] + [{"q": "x", "live": None, "budget": None}])
    assert "1 probe" in g["error"] and verdict("g3", g) == "ERROR"            # a live probe that failed
    cut = _pair()
    g = v03d.g3([_pair() for _ in range(30)] + [{"q": "x", "live": cut["live"], "budget": None}])
    assert "1 probe" in g["error"] and verdict("g3", g) == "ERROR"            # a cut whose budget probe failed
    g = v03d.g3([_pair() for _ in range(30)] + [{"q": "x", "live": {"close": "budget0"}, "budget": None}])
    assert g["pass"] is True and "error" not in g                            # budget0: legitimately no pair
    g = v03d.g3([_pair() for _ in range(23)])
    assert g["pass"] is None and "not testable" in g["error"]
    g = v03d.g3([_pair(same=False) for _ in range(30)])
    assert g["dp_median"] is None and g["pass"] is False                     # no identical trace: no |Δp| to pass on


def test_g4_fails_a_measured_overhead_even_without_a_rate():
    g = v03d.g4({"chunk_overhead_ms": {"p50": 16.0}}, [], [], 0.5)
    assert g["pass"] is False and "deadline cuts" in g["error"]
    assert v03d.g4({"chunk_overhead_ms": {"p50": 6.0}}, [], [], 0.5)["pass"] is None


def test_g5_is_an_error_when_a_served_request_failed_and_takes_k_p50_by_nearest_rank():
    point = {"deadlines": {"2000": {"closes": {"natural": 0.5, "eos": 0.0}, "k_p50": 100}}}
    rows = [_entry("natural", 500, 2000)] * 50 + [_entry("deadline", 2000, 2000, k=104)] * 49
    g = v03d.g5(point, [*rows, _failed(500)])
    assert g["pass"] is None and "1 requests failed" in g["error"]
    g = v03d.g5(point, [_entry("natural", 500, 2000)] * 2 + [_entry("deadline", 2000, 2000, k=100),
                                                             _entry("deadline", 2000, 2000, k=104)])
    assert g["k_p50_served"] == 100                                           # nearest rank, never the 102 midpoint


def _ee_profile(cv):
    return {"converge": cv, "ee": {"0.9": {"acc": 0.91, "n": 376, "lat_p50_ms": [1100.0, 1000.0, 1200.0],
                                           "lat_p95_ms": [2100.0, 2000.0, 2300.0], "cell": [2, 0.9], "K": 32}}}


def test_c1_keeps_each_gain_verdict_whole_and_reports_one_line_per_z():
    cv = _cv(by_rho=_by_rho((0.5,)), rho_max=0.5, prefix=0.5)
    cv["gain"] = {"0.9": {"p50": {"front": True, "speed": True, "accuracy": False, "gains": True},
                          "p95": {"front": False, "speed": False, "accuracy": False, "gains": False}, "gains": True}}
    cv["rules"] = {"0.9": {"cell": [2, 0.9], "acc": 0.93, "dacc": [-0.01, -0.03, 0.01], "tokens_saved": 41.0,
                           "net_saving": 35.0, "stops": 120}, "0.95": None}
    g = v03d.c1({"tiel": _ee_profile(cv)})
    m = g["models"]["tiel"]
    assert m["gain"]["0.9"]["p50"]["speed"] is True and m["gains"] is True
    text = render({"updated": "t", "gates": {"c1": g}})
    assert ("- tiel z 0.9: rule (n 2, y 0.9), acc 0.930 (Δ vs full think -0.010 [-0.030, +0.010]), saves 41.0 tokens (35.0 net); "
            "EE point acc 0.910, p50 1100 ms, p95 2100 ms; gains: p50 yes (front, speed), p95 no") in text
    assert "- tiel z 0.95: no rule reaches z" in text


def test_c1_reads_info_whenever_it_decided():
    from bench.report_v03_deadline import verdict
    g = v03d.c1({"tiel": {"converge": _cv(gains=True, by_rho=_by_rho(()), prefix=None)}, "qwen9b": {}})
    assert g["build"] is True and g.get("error") and verdict("c1", g) == "info"
    g = v03d.c1({"tiel": {"converge": _cv(by_rho=_by_rho(()), prefix=None)}, "qwen9b": {}})
    assert g["build"] is None and verdict("c1", g) == "ERROR"


def test_a_rendered_g3_pass_row_escapes_its_pipes():
    import re
    g = v03d.g3([_pair() for _ in range(48)])
    assert g["pass"] is True
    (row,) = [ln for ln in render({"updated": "t", "gates": {"g3": g}}).splitlines() if ln.startswith("| G3 |")]
    assert row.endswith("| PASS |") and len(re.findall(r"(?<!\\)\|", row)) == 5


def test_a_gates_rerun_keeps_a_not_testable_g2(tmp_path, monkeypatch):
    g2 = tmp_path / "g2.json"
    g2.write_text(json.dumps({"pass": None, "n_calls": 900, "n_cut": 120,
                              "error": "not testable: 900 calls (< 1200): play more pilot games"}))
    out, doc = tmp_path / "out", tmp_path / "doc.md"
    assert v03d.main(["--out", str(out), "--parts", "gates", "--g2", str(g2), "--doc", str(doc)]) == 0
    assert v03d.main(["--out", str(out), "--parts", "gates", "--doc", str(doc)]) == 0
    kept = json.loads((out / "summary.json").read_text())["gates"]["g2"]
    assert kept["error"].startswith("not testable: 900 calls") and kept["source"] == str(g2.resolve())


def test_main_records_the_session_s_provenance_and_warns_on_dirty_code_or_another_build(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(v03t, "code_state", lambda root: {"root": str(root), "commit": "abcdef1234567890", "dirty": True,
                                                          "status": [" M bench/x.py"]})
    monkeypatch.setattr(v03t, "_llama_server_version", lambda binary: "version: 9999 (deadbee)\nbuilt with clang")
    monkeypatch.setattr(v03t, "disk_free_gib", lambda paths: {p: 42.0 for p in paths})
    out, doc = tmp_path / "out", tmp_path / "doc.md"
    assert v03d.main(["--out", str(out), "--parts", "gates", "--doc", str(doc)]) == 0
    session = json.loads((out / "summary.json").read_text())["sessions"][-1]
    assert session["code"]["commit"] == "abcdef1234567890" and session["llama_server"] == v03d.LLAMA_SERVER
    assert session["version"].startswith("version: 9999") and session["disk"] == {str(out): 42.0, "/": 42.0}
    assert session["started"]
    text = doc.read_text()
    assert "Run: commit abcdef123456 (dirty), llama-server version: 9999 (deadbee), disk free" in text
    assert "WARNING: the code that ran is dirty" in text and "does not name 4ebdf2c" in text
    printed = capsys.readouterr().out
    assert "WARNING: the code that ran is dirty" in printed
    monkeypatch.setattr(v03t, "code_state", lambda root: {"root": str(root), "commit": "abcdef1234567890",
                                                          "dirty": False, "status": []})
    monkeypatch.setattr(v03t, "_llama_server_version", lambda binary: "version: 8000 (4ebdf2c)")
    assert v03d.main(["--out", str(out), "--parts", "gates", "--doc", str(doc)]) == 0
    assert "WARNING" not in doc.read_text()


def test_a_gates_only_rerun_keeps_the_gpu_run_s_session_and_the_header_shows_both(tmp_path, monkeypatch):
    """A rerun of `--parts gates --g2` long after the GPU run: every run appends its session, and the header shows
    the latest session and the first GPU one (its start, build and disk), not only the last."""
    out, doc = tmp_path / "out", tmp_path / "doc.md"
    out.mkdir()
    gpu = {"started": "2026-10-11T08:00:00+00:00", "parts": ["calib", "serve", "gates"],
           "code": {"commit": "1111111111112222", "dirty": False}, "llama_server": "x",
           "version": "version: 8000 (4ebdf2c)", "disk": {"/": 55.0}, "warnings": []}
    (out / "summary.json").write_text(json.dumps({"parts": {}, "errors": {}, "sessions": [gpu]}))
    monkeypatch.setattr(v03t, "disk_free_gib", lambda paths: {p: 12.0 for p in paths})
    assert v03d.main(["--out", str(out), "--parts", "gates", "--doc", str(doc)]) == 0
    sessions = json.loads((out / "summary.json").read_text())["sessions"]
    assert len(sessions) == 2 and sessions[0] == gpu and sessions[1]["parts"] == ["gates"]
    text = doc.read_text()
    assert "Run: commit 000000000000" in text and ", / 12.0 GiB (started" in text                   # the latest
    assert "GPU run: commit 111111111111, llama-server version: 8000 (4ebdf2c), disk free / 55.0 GiB" in text
    assert "(started 2026-10-11T08:00:00+00:00; parts calib, serve, gates)" in text
    legacy = {"parts": {}, "errors": {}, "session": {**gpu, "parts": None}}       # a summary from before `sessions`
    (out / "summary.json").write_text(json.dumps(legacy))
    assert v03d.main(["--out", str(out), "--parts", "gates", "--doc", str(doc)]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert "session" not in s and len(s["sessions"]) == 2 and s["sessions"][0]["started"] == gpu["started"]
