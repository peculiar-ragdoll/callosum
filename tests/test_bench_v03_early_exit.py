"""The early-exit acceptance bench's pieces on synthetic evidence (no GPU): the calibration command lines, the
counterbalanced E3 order, E1-E3's pass, fail and ERROR, the held-out guard, the labelled run's paths, the report."""
import json
from pathlib import Path

import pytest
from test_calibrate_think import write_data

from bench import acceptance_v03_early_exit as ee
from bench import acceptance_v03_think as v03t
from bench.report_v03_early_exit import render, verdict_lines
from callo.calibrate_think import load_data

PROFILE = {"latency": {"read_forced_ms": {"p50": 27.0}},
           "budgets": {"0": {}, "1024": {"lat_p50_ms": [1400.0, 1300.0, 1500.0]}},
           "converge": {"K": 32, "rules": {"0.9": None, "0.95": {"cell": [1, 0.9], "T": 2.0, "dacc": [0.0, -0.02, 0.02]},
                                           "0.99": None}},
           "ee": {"0.95": {"lat_p50_ms": [600.0, 550.0, 650.0]}}}


@pytest.fixture(autouse=True)
def _no_real_provenance(monkeypatch):
    """`main` records the session's provenance (code, llama-server --version, disk): stubbed, so no test runs git or a
    llama-server binary."""
    monkeypatch.setattr(v03t, "code_state", lambda root: {"root": str(root), "commit": "0" * 40, "dirty": False,
                                                          "status": []})
    monkeypatch.setattr(v03t, "_llama_server_version", lambda binary: "version: 1 (4ebdf2c)")
    monkeypatch.setattr(v03t, "disk_free_gib", lambda paths: {p: 100.0 for p in paths})


def _with_k(k):
    return {**PROFILE, "converge": {**PROFILE["converge"], "K": k}}


def _row(q, *, close="converged", choice="a", correct=True, ms=600.0, converge=0.95, cluster=None):
    return {"q": q, "cluster": cluster or q, "status": 200, "entry": {"think_close": close, "converge": converge},
            "choice": choice, "correct": correct, "ms": ms}


def _live(q, probes, final):
    return {"q": q, "status": 200, "probes": [[32 * (i + 1), p] for i, p in enumerate(probes)], "final": final}


def test_calibrations_record_read_forced_and_measure_convergence_with_and_without_greedy(tmp_path):
    argv = ee.calib_argv("tiel-greedy", tmp_path, tmp_path / "test.jsonl")
    assert argv[argv.index("--converge-probe") + 1] == "32" and "temp=0" in argv
    assert argv[argv.index("--deadlines") + 1] == "2000"            # the reserve pre-pass records read_forced_ms
    assert argv[argv.index("--warmup") + 1] == "9"                  # ... on warm-up questions 2..9: 8 forced reads
    assert argv[argv.index("--data") + 1] == str(tmp_path / "test.jsonl")
    argv = ee.calib_argv("minicpm", tmp_path, tmp_path / "test.jsonl")
    assert argv[argv.index("--converge-probe") + 1] == "64" and "--think-sampling" not in argv and "--deadlines" in argv
    assert set(ee.runs()) == {"tiel", "tiel-greedy", "minicpm", "minicpm-greedy"}
    assert set(ee.runs(["minicpm"])) == {"minicpm", "minicpm-greedy"}
    assert "--resume" not in argv and "--no-cache" not in argv          # callo as shipped: calibrate's default cache
    (tmp_path / "calib" / "minicpm").mkdir(parents=True)
    (tmp_path / "calib" / "minicpm" / "run.json").write_text("{}")
    assert "--resume" in ee.calib_argv("minicpm", tmp_path, tmp_path / "test.jsonl")


def test_e3_s_two_runs_alternate_which_goes_first(monkeypatch):
    calls = []
    monkeypatch.setattr(ee, "serve_rows", lambda url, items, fields: calls.append((items[0], fields["think_converge"]))
                        or [{"q": items[0]}])
    a, b = ee.serve_pairs("u", ["i0", "i1", "i2"], {"think_converge": 0.95}, {"think_converge": 0})
    assert calls == [("i0", 0.95), ("i0", 0), ("i1", 0), ("i1", 0.95), ("i2", 0.95), ("i2", 0)]
    assert [r["q"] for r in a] == [r["q"] for r in b] == ["i0", "i1", "i2"]


def test_serve_rows_and_the_live_probe_read_the_replies(tmp_path, monkeypatch):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 2))
    it = items[0]
    keys = list(it.question.keys)
    probs = {k: (0.9 if i == it.label else 0.1 / (len(keys) - 1)) for i, k in enumerate(keys)}
    sent = []

    def post(url, body):
        sent.append((url, body))
        if url.endswith("/v1/systemone"):
            entry = {"q": it.qid, "ms": 512.5, "think_close": "converged", "converge": 0.95, "probes": 2}
            return 200, {"x-callo-timings": json.dumps([entry])}, {"answers": {it.qid: {"type": "choice",
                                                                                         "probabilities": probs}}}
        live = {"close": "natural", "probes": [{"k": 32, "probs": [0.6, 0.4]}, {"k": 64, "probs": [0.95, 0.05]}],
                "readout": {"k": 70, "probs": [0.97, 0.03]}}
        return 200, {}, {"questions": {it.qid: {"live": live}}}

    monkeypatch.setattr(ee, "_post", post)
    (row,) = ee.serve_rows("http://h", [it], {"think_converge": 0.95})
    assert sent[-1] == ("http://h/v1/systemone", {**it.body, "think_converge": 0.95})
    assert (row["status"], row["ms"], row["correct"], row["choice"], row["cluster"]) == (
        200, 512.5, True, keys[it.label], it.cluster)
    (lv,) = ee.probe_live_rows("http://h", [it], cap=1024, k=32)
    url, body = sent[-1]
    assert url == "http://h/callo/think-probe" and body["budgets"] == [1024]
    assert body["live"] == {"deadline_ms": None, "probe_every": 32} and body["request"] == it.body
    assert lv["probes"] == [[32, [0.6, 0.4]], [64, [0.95, 0.05]]] and lv["final"] == [0.97, 0.03]
    monkeypatch.setattr(ee, "_post", lambda url, body: (-1, {}, None))
    assert ee.serve_rows("http://h", [it], {})[0]["correct"] is None
    assert ee.probe_live_rows("http://h", [it], cap=1024, k=32)[0]["probes"] is None


class _Session:
    def __init__(self, url):
        self.url, self.finished = url, False

    def stats(self):
        return {"think": {"converge": {"after_probe": {"n": 40, "restored": 40}, "probe_read_ms": {"p50": 20.0}}}}

    def finish(self):
        self.finished = True


def test_serve_warms_up_without_early_exit_and_reads_the_end_at_the_largest_budget(tmp_path, monkeypatch):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 4))
    for run in ("minicpm", "minicpm-greedy"):
        (tmp_path / "calib" / run).mkdir(parents=True)
        (tmp_path / "calib" / run / "m.think.json").write_text(json.dumps(_with_k(64)))
    started, sent = [], []

    def start(name, out, *, model, ctx, cache, flags):
        started.append((name, model, cache, flags))
        return _Session(f"http://{name}")

    monkeypatch.setattr(ee, "_start", start)
    monkeypatch.setattr(ee, "gguf_traits", lambda model: {"hybrid": False})
    monkeypatch.setattr(ee, "_post", lambda url, body: sent.append((url, body)) or (-1, {}, None))
    res = {"tiel": {"kept": True}}
    ee.run_serve(tmp_path, items, res, z=0.9, models=["minicpm"])
    assert [s[0] for s in started] == ["serve-minicpm", "serve-minicpm-greedy"] and all(s[2] for s in started)
    assert started[0][3][:2] == ["--think-profile", str(tmp_path / "calib" / "minicpm" / "m.think.json")]
    assert res["tiel"] == {"kept": True}                            # a subset run keeps the other model's sessions
    own = [b for u, b in sent if u == "http://serve-minicpm/v1/systemone"]
    assert all(b["think_budget"] == 64 and b["think_converge"] == 0 for b in own[:ee.WARMUP_N])
    rest = own[ee.WARMUP_N:]
    assert [b.get("think_converge") for b in rest[:4]] == [0.9, 0, 0, 0.9]
    assert all(b["think_budget"] == 1024 for b in rest if b["think_converge"] == 0)
    assert all("think_budget" not in b for b in rest if b["think_converge"])     # early exit alone: capped at B*
    probes = [b for u, b in sent if u.endswith("/callo/think-probe")]
    assert len(probes) == len(items) and probes[0]["live"] == {"deadline_ms": None, "probe_every": 64}
    m = res["minicpm"]
    assert (m["z"], m["bmax"], m["hybrid"], len(m["ee"]), len(m["end"])) == (0.9, 1024, False, 4, 4)
    assert m["stats"]["after_probe"]["n"] == 40 and len(res["minicpm-greedy"]["live"]) == 4


def test_e1_restores_on_hybrids_and_reads_within_1_5x_the_forced_read():
    cv = {"after_probe": {"n": 100, "restored": 97}, "probe_read_ms": {"p50": 30.0}}
    g = ee.e1(cv, PROFILE, hybrid=True, forced_n=8)
    assert g["pass"] is True and g["read_forced_n"] == 8
    assert ee.e1({**cv, "after_probe": {"n": 100, "restored": 90}}, PROFILE, hybrid=True, forced_n=8)["pass"] is False
    assert ee.e1({**cv, "after_probe": {"n": 100, "restored": 90}}, PROFILE, hybrid=False, forced_n=8)["pass"] is True
    assert ee.e1({**cv, "probe_read_ms": {"p50": 41.0}}, PROFILE, hybrid=True, forced_n=8)["pass"] is False   # > 1.5x27
    assert ee.e1({**cv, "after_probe": {"n": 10, "restored": 10}}, PROFILE, hybrid=True, forced_n=8)["pass"] is None
    assert ee.e1(None, PROFILE, hybrid=True, forced_n=8)["pass"] is None
    g = ee.e1(cv, {**PROFILE, "latency": {"read_forced_ms": None}}, hybrid=True, forced_n=8)
    assert g["pass"] is None and g["error"].startswith("the profile has no latency.read_forced_ms")


def test_e1_is_error_on_a_forced_read_baseline_of_fewer_than_5_reads(tmp_path):
    cv = {"after_probe": {"n": 100, "restored": 97}, "probe_read_ms": {"p50": 30.0}}
    for n in (2, 4, None):                                           # 2: --audit 0 with calibrate's default --warmup 3
        g = ee.e1(cv, PROFILE, hybrid=True, forced_n=n)
        assert g["pass"] is None and "forced reads (>= 5 needed" in g["error"]
    assert ee.e1(cv, PROFILE, hybrid=True, forced_n=5)["pass"] is True
    d = tmp_path / "calib" / "tiel"
    d.mkdir(parents=True)
    assert ee.read_forced_n(tmp_path, "tiel") is None                # no raw file: unknown
    from callo.calibrate_think import RAW_FORMAT
    lines = [{"callo_think_raw": RAW_FORMAT}, {"reserve": {"ms": 50.0, "source": "pre-pass", "reads": [
        40.0, None, *[20.0] * 7]}},                                   # question 1 cold, one single-option question
             {"phase": "audit", "q": "a", "read_ms": 21.0, "trace": {"close": "forced"}},
             {"phase": "audit", "q": "b", "read_ms": 21.0, "trace": {"close": "natural"}}]
    (d / "m.raw.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert ee.read_forced_n(tmp_path, "tiel") == 8                   # 7 pre-pass reads + 1 forced audit read


def test_e2_reads_the_stop_and_the_end_off_one_trace_with_the_served_rule():
    agree = [_live(f"q{i}", [[0.5, 0.3, 0.2], [0.95, 0.03, 0.02]], [0.9, 0.05, 0.05]) for i in range(220)]
    g = ee.e2(agree, PROFILE)                                       # rule (1, .9): stops at the second probe
    assert g["pass"] is True and (g["n"], g["share"], g["cell"]) == (220, 1.0, [1, 0.9]) and g["ci"][1] == 1.0
    flip = [_live(f"q{i}", [[0.95, 0.03, 0.02]], [0.1, 0.8, 0.1] if i < 22 else [0.9, 0.05, 0.05]) for i in range(220)]
    assert ee.e2(flip, PROFILE)["pass"] is False                    # 198/220 = .90 < .92
    never = [_live(f"q{i}", [[0.5, 0.3, 0.2]], [0.9, 0.05, 0.05]) for i in range(220)]
    assert ee.e2(never, PROFILE)["pass"] is None                    # no stop: too few
    g = ee.e2(agree, {**PROFILE, "converge": {"K": 32, "rules": {}}})
    assert g["pass"] is None and "no rule that reaches z 0.95" in g["error"]
    g = ee.e2(agree, PROFILE, z=0.91)                               # between targets: .95's rule, as serve takes it
    assert g["cell"] == [1, 0.9] and g["bound"] == pytest.approx(0.88)


def test_e2_needs_200_live_traces_and_counts_the_dropped_probes():
    ok = [_live(f"q{i}", [[0.95, 0.03, 0.02]], [0.9, 0.05, 0.05]) for i in range(199)]
    failed = [{"q": f"x{i}", "status": 503, "probes": None, "final": None} for i in range(137)]
    g = ee.e2(ok + failed, PROFILE)
    assert g["pass"] is None and (g["traces"], g["dropped"]) == (199, 137)
    assert "199 live-mode traces of 336 questions (137 probe requests failed" in g["error"]
    g = ee.e2(ok + ok[:1] + failed, PROFILE)
    assert g["pass"] is True and (g["traces"], g["dropped"]) == (200, 137)


def test_e3_the_served_trade_against_the_profile():
    end = [_row(f"q{i}", close="budget", ms=1400.0 + i) for i in range(60)]
    rows = [_row(f"q{i}", ms=600.0 + i) for i in range(60)]
    g = ee.e3(rows, end, PROFILE)
    assert g["acc_pass"] is True and g["dacc"][0] == 0.0
    assert g["saving_ms"][0] == pytest.approx(800.0) and g["lat_pass"] is True and g["pass"] is True
    assert g["closes"] == {"converged": 60}
    slow = [_row(f"q{i}", ms=1300.0 + i) for i in range(60)]
    assert ee.e3(slow, end, PROFILE)["lat_pass"] is False
    assert ee.e3(rows, end, {**PROFILE, "ee": {}})["pass"] is None
    assert ee.e3(rows, end[:30], PROFILE)["pass"] is None                                  # < 90 % paired
    g = ee.e3(rows, end, {**PROFILE, "converge": {"K": 32, "rules": {}}})
    assert g["pass"] is None and g["error"] == "the profile has no rule that reaches z 0.95"
    off = [_row(f"q{i}", converge="unreachable", ms=60.0) for i in range(60)]
    g = ee.e3(off, end, PROFILE)                                     # served without early exit: not this gate's data
    assert g["pass"] is None and "without early exit" in g["error"]


def test_e3_takes_the_rule_serve_runs_for_z():
    end = [_row(f"q{i}", close="budget", ms=1400.0 + i) for i in range(60)]
    rows = [_row(f"q{i}", ms=600.0 + i, converge=0.92) for i in range(60)]
    g = ee.e3(rows, end, PROFILE, z=0.92)                            # between .9 and .95: serve runs .95's rule
    assert g["pass"] is True and (g["target"], g["cell"], g["profile_saving_ms"]) == (0.95, [1, 0.9], 800.0)
    g = ee.e3(rows, end, PROFILE, z=0.97)                            # .99 has no rule
    assert g["pass"] is None and "no rule that reaches z 0.97" in g["error"]


def test_gates_never_judge_e3_at_a_z_other_than_the_one_served(tmp_path):
    for run in ("tiel", "tiel-greedy"):
        (tmp_path / "calib" / run).mkdir(parents=True)
        (tmp_path / "calib" / run / "t.think.json").write_text(json.dumps(PROFILE))
    own = {"z": 0.95, "hybrid": True, "ee": [_row(f"q{i}") for i in range(60)],
           "end": [_row(f"q{i}", close="budget", ms=1400.0 + i) for i in range(60)],
           "stats": {"after_probe": {"n": 100, "restored": 99}, "probe_read_ms": {"p50": 30.0}}}
    serve = {"tiel": own, "tiel-greedy": {"live": [_live(f"q{i}", [[0.95, 0.03, 0.02]], [0.9, 0.05, 0.05])
                                                   for i in range(220)]}}
    g = ee.gates(tmp_path, serve, z=0.95, models=["tiel"])["tiel"]
    assert g["e1"]["pass"] is None and "unknown number of forced reads" in g["e1"]["error"]   # no raw file
    assert g["e2"]["pass"] is True and g["e3"]["n"] == 60
    g = ee.gates(tmp_path, serve, z=0.9)
    assert set(g) == {"tiel", "minicpm"} and g["minicpm"]["e3"]["pass"] is None
    assert g["tiel"]["e3"]["pass"] is None and "served at z 0.95" in g["tiel"]["e3"]["error"]


def test_p50_saving_is_deterministic_and_its_point_is_the_plain_difference():
    end, fast = [1000.0 + i for i in range(50)], [400.0 + i for i in range(50)]
    a = ee.p50_saving(end, fast, [f"c{i % 10}" for i in range(50)], seed=0, resamples=200)
    assert a == ee.p50_saving(end, fast, [f"c{i % 10}" for i in range(50)], seed=0, resamples=200)
    assert a[0] == pytest.approx(600.0) and a[1] <= a[0] <= a[2]


def test_p50_saving_resamples_within_each_source():
    # two sources of very different latency: a stratified draw keeps 5 clusters of each, an unstratified one does not
    end = [1000.0] * 25 + [5000.0] * 25
    fast = [400.0] * 25 + [4900.0] * 25
    clusters = [f"c{i // 5}" for i in range(50)]
    strata = ["mmlu"] * 25 + ["qnli"] * 25
    st = ee.p50_saving(end, fast, clusters, strata=strata, seed=0, resamples=400)
    assert st == ee.p50_saving(end, fast, clusters, strata=strata, seed=0, resamples=400)
    assert st[1] == st[2]                                             # every draw: 25 + 25, the same p50s
    plain = ee.p50_saving(end, fast, clusters, seed=0, resamples=400)
    assert plain[1] < plain[2]


def test_clopper_pearson_brackets_the_share():
    lo, hi = ee.clopper_pearson(36, 40)
    assert lo < 0.9 < hi and (ee.clopper_pearson(0, 10)[0], ee.clopper_pearson(10, 10)[1]) == (0.0, 1.0)
    assert ee.clopper_pearson(40, 40)[0] == pytest.approx(0.025 ** (1 / 40))


def test_a_labelled_run_writes_its_own_summary_and_report(tmp_path):
    assert ee.paths(tmp_path, None) == (tmp_path / "summary.json", tmp_path / "v0.3-early-exit.md")
    assert ee.paths(tmp_path, "z0.9") == (tmp_path / "summary-z0.9.json", tmp_path / "v0.3-early-exit-z0.9.md")


def test_the_bench_refuses_a_held_out_file_that_overlaps_the_calibration_file(tmp_path, monkeypatch):
    rec = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "x", "b": "y"}, "label": "a"}},
           "_meta": {"source": "mmlu", "text_sha256": "t1"}}
    test, dev = tmp_path / "test.jsonl", tmp_path / "dev.jsonl"
    test.write_text(json.dumps(rec) + "\n")
    dev.write_text(json.dumps(rec) + "\n")
    assert ee.overlap(test, dev) == 1
    monkeypatch.setattr(ee.lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(ee.lib, "free_memory_bytes", lambda: 1 << 40)
    assert ee.main(["--out", str(tmp_path / "o"), "--data", str(test), "--dev", str(dev), "--parts", "serve"]) == 2
    assert ee.main(["--out", str(tmp_path / "o"), "--data", str(test), "--dev", str(test), "--parts", "serve"]) == 2
    assert ee.main(["--out", str(tmp_path / "o"), "--parts", "nope"]) == 2
    assert ee.main(["--out", str(tmp_path / "o"), "--parts", "gates", "--models", "gemma"]) == 2
    assert not (tmp_path / "o" / "summary.json").exists()


def test_a_gates_only_run_needs_no_gpu_and_errors_on_missing_evidence(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ee.lib, "running_gpu_servers", lambda: pytest.fail("gates never look for a GPU server"))
    out = tmp_path / "o"
    assert ee.main(["--out", str(out), "--parts", "gates", "--models", "minicpm", "--z", "0.9", "--label", "z0.9"]) == 0
    s = json.loads((out / "summary-z0.9.json").read_text())
    assert (s["z"], s["label"], set(s["gates"])) == (0.9, "z0.9", {"minicpm"})
    assert all(g["pass"] is None for g in s["gates"]["minicpm"].values())
    assert "minicpm E1: ERROR" in capsys.readouterr().out
    assert (out / "v0.3-early-exit-z0.9.md").exists() and not (out / "summary.json").exists()
    assert not (out / "run.pid").exists()


def test_the_report_names_each_model_s_gates_and_verdicts():
    s = {"parts": {"calib": {"tiel": {"rc": 0, "tail": "done"}}}, "errors": {}, "z": 0.95, "label": None,
         "gates": {"tiel": {"e1": {"pass": True, "restored_share": 0.99, "n": 300, "read_p50_ms": 28.0,
                                   "read_forced_p50_ms": 27.0, "hybrid": True},
                            "e2": {"pass": False, "share": 0.9, "ci": [0.85, 0.94], "n": 280, "bound": 0.92,
                                   "cell": [1, 0.99]},
                            "e3": {"pass": None, "error": "the profile has no rule"}}}}
    doc = render(s)
    assert "| tiel | E1 |" in doc and "PASS" in doc and "FAIL" in doc and "ERROR" in doc and "rule (1, 0.99)" in doc
    assert any("tiel E2: FAIL" in line for line in verdict_lines(s))
    s["gates"]["tiel"]["e3"] = {"pass": True, "n": 300, "dacc": [0.0, -0.02, 0.02], "profile_dacc": 0.0,
                                "saving_ms": [800.0, 700.0, 900.0], "profile_saving_ms": 780.0,
                                "closes": {"converged": 250, "natural": 50}}
    assert "converged 250 of 300" in render(s)


def _gpu_free(monkeypatch):
    monkeypatch.setattr(ee.lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(ee.lib, "free_memory_bytes", lambda: 1 << 40)
    monkeypatch.setattr(ee, "_foreign_servers", lambda: [])
    monkeypatch.setattr(ee, "select_records", lambda src, dst: dst.parent.mkdir(parents=True, exist_ok=True)
                        or dst.write_text(open(src).read()))


def test_a_calibration_that_exits_non_zero_is_a_part_error(tmp_path, monkeypatch):
    _gpu_free(monkeypatch)
    test, dev = write_data(tmp_path / "test.jsonl", 2), write_data(tmp_path / "dev.jsonl", 2)
    calls = []

    def calibrate(argv):
        run = argv[argv.index("--out") + 1].rsplit("/", 1)[-1]
        calls.append(run)
        if run == "tiel":
            return 1, "calibrate: the guard failed\n"
        d = tmp_path / "o" / "calib" / run
        d.mkdir(parents=True, exist_ok=True)
        (d / "m.think.json").write_text(json.dumps(_with_k(ee.MODELS[run.split("-")[0]][1])))
        return 0, "done\n"

    monkeypatch.setattr(ee, "_calibrate", calibrate)
    out = tmp_path / "o"
    assert ee.main(["--out", str(out), "--data", str(test), "--dev", str(dev), "--parts", "calib"]) == 1
    s = json.loads((out / "summary.json").read_text())
    assert calls == ["tiel", "tiel-greedy", "minicpm", "minicpm-greedy"]    # the other runs still had their turn
    assert s["parts"]["calib"]["tiel"] == {"rc": 1, "tail": "calibrate: the guard failed"}
    assert "tiel (exit 1: calibrate: the guard failed)" in s["errors"]["calib"]
    calls.clear()
    monkeypatch.setattr(ee, "_calibrate", lambda argv: calls.append(argv) or (2, "refusing: a model server\n"))
    assert ee.main(["--out", str(out), "--data", str(test), "--dev", str(dev), "--parts", "calib"]) == 1
    assert len(calls) == 1                                             # a refusal ends the part at once
    assert "refused (exit 2)" in json.loads((out / "summary.json").read_text())["errors"]["calib"]


def test_serve_without_a_profile_is_a_part_error_and_the_other_runs_still_serve(tmp_path, monkeypatch):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 2))
    (tmp_path / "calib" / "minicpm-greedy").mkdir(parents=True)
    (tmp_path / "calib" / "minicpm-greedy" / "m.think.json").write_text(json.dumps(_with_k(64)))
    started = []
    monkeypatch.setattr(ee, "_start", lambda name, out, **kw: started.append(name) or _Session(f"http://{name}"))
    monkeypatch.setattr(ee, "gguf_traits", lambda model: {"hybrid": False})
    monkeypatch.setattr(ee, "_post", lambda url, body: (-1, {}, None))
    res = {}
    with pytest.raises(RuntimeError, match="minicpm: no profile"):
        ee.run_serve(tmp_path, items, res, z=0.95, models=["minicpm"])
    assert started == ["serve-minicpm-greedy"] and "error" in res["minicpm"] and "live" in res["minicpm-greedy"]


def test_a_profile_of_another_k_is_refused_by_serve_and_by_the_gates(tmp_path, monkeypatch):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 2))
    for run in ("minicpm", "minicpm-greedy"):
        (tmp_path / "calib" / run).mkdir(parents=True)
        (tmp_path / "calib" / run / "m.think.json").write_text(json.dumps(_with_k(32)))     # minicpm's K is 64
    monkeypatch.setattr(ee, "_start", lambda *a, **kw: pytest.fail("no session on a profile of another K"))
    res = {}
    with pytest.raises(RuntimeError, match="converge.K 32, not the bench's K 64"):
        ee.run_serve(tmp_path, items, res, z=0.95, models=["minicpm"])
    assert "converge.K 32" in res["minicpm"]["error"]
    g = ee.gates(tmp_path, {}, models=["minicpm"])["minicpm"]
    assert all(x["pass"] is None and "converge.K 32" in x["error"] for x in g.values())


def test_a_labelled_run_reuses_the_unlabelled_greedy_session(tmp_path, monkeypatch):
    _gpu_free(monkeypatch)
    test, dev = write_data(tmp_path / "test.jsonl", 2), write_data(tmp_path / "dev.jsonl", 2)
    out = tmp_path / "o"
    for run in ("minicpm", "minicpm-greedy"):
        (out / "calib" / run).mkdir(parents=True)
        (out / "calib" / run / "m.think.json").write_text(json.dumps(_with_k(64)))
    started = []
    monkeypatch.setattr(ee, "_start", lambda name, out, **kw: started.append(name) or _Session(f"http://{name}"))
    monkeypatch.setattr(ee, "gguf_traits", lambda model: {"hybrid": False})
    monkeypatch.setattr(ee, "_post", lambda url, body: (-1, {}, None))
    argv = ["--out", str(out), "--data", str(test), "--dev", str(dev), "--models", "minicpm", "--parts", "serve"]
    assert ee.main(argv) == 0
    assert started == ["serve-minicpm", "serve-minicpm-greedy"]
    started.clear()
    assert ee.main([*argv, "--z", "0.9", "--label", "z0.9"]) == 0
    assert started == ["serve-minicpm"]                                 # the greedy rows do not depend on z
    lab = json.loads((out / "summary-z0.9.json").read_text())["parts"]["serve"]
    main = json.loads((out / "summary.json").read_text())["parts"]["serve"]
    assert lab["minicpm-greedy"]["reused"] is True and lab["minicpm-greedy"]["live"] == main["minicpm-greedy"]["live"]
    assert lab["minicpm"]["z"] == 0.9 and main["minicpm"]["z"] == 0.95
    (out / "calib" / "minicpm-greedy" / "m.think.json").write_text(json.dumps({**_with_k(64), "x": 1}))
    started.clear()
    assert ee.main([*argv, "--z", "0.9", "--label", "z0.9"]) == 0
    assert started == ["serve-minicpm", "serve-minicpm-greedy"]         # another profile file: served again


# ---- --gate: the early-exit gate's labelled serve run (Tiel only) ----------------------------------------------------

def _gated(q, *, skip=False, correct=True, ms=900.0, source="mmlu", cluster=None, read=150.0, error=None):
    g = {"p": 0.9 if skip else 0.1, "ms": read, "skip": skip, **({"error": error} if error else {})}
    entry = {"think_close": "budget" if skip else "converged", "converge_gate": g,
             **({} if skip else {"converge": 0.95})}
    return {"q": q, "cluster": cluster or q, "source": source, "status": 200, "entry": entry, "choice": "a",
            "correct": correct, "ms": ms}


def _end(q, *, correct=True, ms=1500.0, source="mmlu"):
    return {"q": q, "cluster": q, "source": source, "status": 200, "entry": {"think_close": "budget"},
            "choice": "a", "correct": correct, "ms": ms}


def test_the_gate_run_serves_tiel_alone_gated_against_the_end_reading_and_no_greedy_session(tmp_path, monkeypatch):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 4))
    for run in ("tiel", "tiel-greedy", "tiel-gate"):
        (tmp_path / "calib" / run).mkdir(parents=True)
        (tmp_path / "calib" / run / "t.think.json").write_text(json.dumps(PROFILE))
    started, sent = [], []
    monkeypatch.setattr(ee, "_start", lambda name, out, **kw: started.append(name) or _Session(f"http://{name}"))
    monkeypatch.setattr(ee, "gguf_traits", lambda model: {"hybrid": True})
    monkeypatch.setattr(ee, "_post", lambda url, body: sent.append((url, body)) or (-1, {}, None))
    res = {}
    ee.run_serve(tmp_path, items, res, z=0.95, models=["tiel"], gate=True)
    assert started == ["serve-tiel"] and set(res) == {"tiel"} and res["tiel"]["gate"] is True
    rest = [b for u, b in sent][ee.WARMUP_N:]
    assert not any(u.endswith("/callo/think-probe") for u, _ in sent)
    assert [b.get("think_converge_gate") for b in rest[:4]] == [True, None, None, True]       # counterbalanced as E3
    assert all(b["think_converge"] == 0.95 for b in rest if b.get("think_converge_gate"))
    assert all(b["think_budget"] == 1024 and b["think_converge"] == 0 for b in rest if "think_converge_gate" not in b)
    assert len(res["tiel"]["ee"]) == len(res["tiel"]["end"]) == 4


def test_the_gate_holds_when_accuracy_is_kept_and_the_p50_falls():
    sources = ["mmlu", "qnli", "composition"]
    end = [_end(f"q{i}", source=sources[i % 3], ms=1500.0 + i) for i in range(60)]
    rows = [_gated(f"q{i}", skip=i % 3 == 2, source=sources[i % 3], ms=(1500.0 if i % 3 == 2 else 600.0) + i,
                   read=140.0 + i) for i in range(60)]
    g = ee.gate_result({"ee": rows, "end": end, "stats": {"gate": {"asked": 60}}})
    assert g["verdict"] == "holds" and g["pass"] is True and g["n"] == 60
    assert g["acc"] == {"ee": 1.0, "end": 1.0} and g["dacc"][0] == 0.0 and g["dacc"][2] >= 0
    assert g["saving_ms"][1] > 0 and g["lat"]["end"]["p50"] > g["lat"]["ee"]["p50"]
    assert g["lat"]["ee"]["mean"] == pytest.approx(sum(r["ms"] for r in rows) / 60)
    assert g["skip_share"] == pytest.approx(20 / 60) and g["gate_errors"] == 0
    assert 140.0 <= g["read_ms"]["p50"] <= g["read_ms"]["p95"] <= 199.0
    assert g["by_source"]["composition"] == {"n": 20, "ee": 1.0, "end": 1.0}
    assert g["closes"] == {"converged": 40, "budget": 20} and g["stats"] == {"asked": 60}


def test_the_gate_does_not_hold_on_an_accuracy_loss_or_no_saving_and_is_error_without_gate_evidence():
    end = [_end(f"q{i}", ms=1500.0 + i) for i in range(60)]
    wrong = [_gated(f"q{i}", correct=i % 2 == 0, ms=600.0 + i) for i in range(60)]
    g = ee.gate_result({"ee": wrong, "end": end})
    assert g["dacc"][2] < 0 and (g["verdict"], g["pass"]) == ("does not hold", False)
    slow = [_gated(f"q{i}", ms=1500.0 + i) for i in range(60)]
    g = ee.gate_result({"ee": slow, "end": end})
    assert g["saving_ms"][1] <= 0 and g["verdict"] == "does not hold"
    failed = [_gated(f"q{i}", skip=True, error="boom", ms=600.0 + i) for i in range(60)]
    g = ee.gate_result({"ee": failed, "end": end})
    assert (g["pass"], g["verdict"], g["gate_errors"]) == (None, None, 60) and "60 of 60 gate reads failed" in g["error"]
    some = [_gated(f"q{i}", skip=i < 7, error="boom" if i < 7 else None, ms=600.0 + i) for i in range(60)]
    assert ee.gate_result({"ee": some, "end": end})["pass"] is None                       # 7 of 60 > 10 %
    some = [_gated(f"q{i}", skip=i < 6, error="boom" if i < 6 else None, ms=600.0 + i) for i in range(60)]
    g = ee.gate_result({"ee": some, "end": end})
    assert g["verdict"] == "holds" and g["gate_errors"] == 6                              # 6 of 60: judged
    plain = [_row(f"q{i}") for i in range(60)]                       # served without the gate: not its evidence
    g = ee.gate_result({"ee": plain, "end": end})
    assert g["pass"] is None and g["verdict"] is None and "without the gate" in g["error"]
    assert ee.gate_result({"ee": slow, "end": end[:30]})["pass"] is None                    # < 90 % paired
    assert ee.gate_result({})["pass"] is None


def test_main_runs_the_gate_as_a_labelled_tiel_serve_and_gates_run(tmp_path, monkeypatch, capsys):
    out = tmp_path / "o"
    for bad in (["--gate", "--models", "minicpm"], ["--gate", "--label", "z0.9"], ["--gate-fresh"]):
        assert ee.main(["--out", str(out), *bad]) == 2
    assert not out.exists() or not list(out.glob("summary*"))
    seen = {}
    calibs = []
    monkeypatch.setattr(ee, "run_calib", lambda o, test, models, res, *, gate=False, fresh=False:
                        calibs.append((models, gate, fresh)) or res)

    def serve(o, dev, res, *, z, models, reuse=None, gate=False):
        seen.update(z=z, models=models, gate=gate, reuse=reuse)
        res["tiel"] = {"z": z, "gate": True, "ee": [_gated(f"q{i}", ms=600.0 + i) for i in range(60)],
                       "end": [_end(f"q{i}", ms=1500.0 + i) for i in range(60)]}
        return res

    _gpu_free(monkeypatch)
    monkeypatch.setattr(ee, "run_serve", serve)
    monkeypatch.setattr(ee, "load_data", lambda p: ([], None))
    test, dev = write_data(tmp_path / "test.jsonl", 2), write_data(tmp_path / "dev.jsonl", 2)
    assert ee.main(["--out", str(out), "--data", str(test), "--dev", str(dev), "--gate", "--parts", "serve"]) == 0
    assert "not computed" in json.loads((out / "summary-gate.json").read_text())["gate"]["error"]
    assert "tiel gate: ERROR" in capsys.readouterr().out
    assert ee.main(["--out", str(out), "--data", str(test), "--dev", str(dev), "--gate"]) == 0
    assert seen == {"z": 0.95, "models": ["tiel"], "gate": True, "reuse": None}
    assert calibs == [(["tiel"], True, False)]                       # the gated calibration runs first
    assert ee.main(["--out", str(out), "--data", str(test), "--dev", str(dev), "--gate", "--gate-fresh",
                    "--parts", "calib"]) == 0 and calibs[-1] == (["tiel"], True, True)
    s = json.loads((out / "summary-gate.json").read_text())
    assert s["label"] == "gate" and s["gate"]["verdict"] == "holds" and "gates" not in s
    assert "tiel gate: holds" in capsys.readouterr().out
    doc = (out / "v0.3-early-exit-gate.md").read_text()
    assert "early-exit gate" in doc and "holds" in doc and "| mmlu |" in doc and not (out / "summary.json").exists()


def test_the_report_renders_the_gate():
    g = {"pass": False, "verdict": "does not hold", "n": 300, "acc": {"ee": 0.9, "end": 0.92},
         "dacc": [-0.02, -0.05, -0.001], "lat": {"ee": {"p50": 900.0, "mean": 1000.0},
                                                  "end": {"p50": 1500.0, "mean": 1600.0}},
         "saving_ms": [600.0, 500.0, 700.0], "skip_share": 0.36, "gate_errors": 1,
         "read_ms": {"p50": 150.0, "p95": 210.0}, "by_source": {"mmlu": {"n": 80, "ee": 0.85, "end": 0.86}},
         "closes": {"converged": 190, "budget": 110}}
    s = {"parts": {}, "errors": {}, "z": 0.95, "label": "gate", "gate": g}
    doc = render(s)
    assert "does not hold" in doc and "36.0 %" in doc and "Δacc -0.020 [-0.050, -0.001]" in doc
    assert "| mmlu | 80 | 0.850 | 0.860 |" in doc and "read p50 150 ms, p95 210 ms" in doc
    assert verdict_lines(s) == ["tiel gate: does not hold"]
    assert verdict_lines({**s, "gate": {"pass": None, "verdict": None, "error": "x"}}) == ["tiel gate: ERROR"]


# ---- --gate's calibration: the gate measured, its own directory, the profile vs served line ---------------------------

def test_the_gated_calibration_is_tiel_s_own_sampler_run_in_its_own_directory(tmp_path):
    argv = ee.calib_argv(ee.GATE_RUN, tmp_path, tmp_path / "test.jsonl")
    assert argv[argv.index("--out") + 1] == str(tmp_path / "calib" / "tiel-gate")
    assert argv[argv.index("--converge-probe") + 1] == "32" and "--think-sampling" not in argv
    assert argv[argv.index("--model") + 1] == ee.TIEL and "--resume" not in argv
    same = ee.calib_argv("tiel", tmp_path, tmp_path / "test.jsonl")
    assert argv[:argv.index("--out")] == same[:same.index("--out")]          # the unlabelled run's settings exactly
    (tmp_path / "calib" / "tiel-gate").mkdir(parents=True)
    (tmp_path / "calib" / "tiel-gate" / "T.raw.jsonl").write_text("{}\n")
    assert "--resume" in ee.calib_argv(ee.GATE_RUN, tmp_path, tmp_path / "test.jsonl")


GATED = {**PROFILE, "converge": {**PROFILE["converge"], "gate": {"sha256": "s" * 64, "flagged": 0.3}},
         "ee_gated": {"0.95": {"lat_p50_ms": [900.0, 850.0, 950.0], "dacc_cap": [0.0, -0.01, 0.01],
                               "gate": {"flagged": 0.3}}}}


def _calibrated(tmp_path, profile, calls):
    """A `_calibrate` stand-in: records the argv and writes `profile` to the run's --out."""
    def run(argv):
        calls.append(argv)
        d = Path(argv[argv.index("--out") + 1])
        d.mkdir(parents=True, exist_ok=True)
        (d / "T.think.json").write_text(json.dumps(profile))
        return 0, "done\n"
    return run


def test_the_gated_calibration_starts_from_a_copy_of_the_unlabelled_raw_file_and_never_touches_it(tmp_path,
                                                                                                   monkeypatch):
    """The addendum's resume: the unlabelled Tiel raw file (made before the gate) is copied into calib/tiel-gate and
    resumed, so only the gate is read; the unlabelled directory is left as it was. `--gate-fresh` copies nothing."""
    own = tmp_path / "calib" / "tiel"
    own.mkdir(parents=True)
    (own / "T.raw.jsonl").write_text('{"callo_think_raw": 1}\n')
    (own / "T.think.json").write_text(json.dumps(PROFILE))
    before = {p.name: p.read_bytes() for p in own.iterdir()}
    calls = []
    monkeypatch.setattr(ee, "_calibrate", _calibrated(tmp_path, GATED, calls))
    res = ee.run_calib(tmp_path, tmp_path / "test.jsonl", ["tiel"], {}, gate=True)
    (argv,) = calls
    assert argv[argv.index("--out") + 1] == str(tmp_path / "calib" / "tiel-gate") and "--resume" in argv
    assert (tmp_path / "calib" / "tiel-gate" / "T.raw.jsonl").read_bytes() == before["T.raw.jsonl"]
    assert set(res) == {"tiel-gate"} and res["tiel-gate"]["seeded"].endswith("tiel-gate/T.raw.jsonl")
    assert {p.name: p.read_bytes() for p in own.iterdir()} == before
    ee.run_calib(tmp_path, tmp_path / "test.jsonl", ["tiel"], {}, gate=True)          # a raw file is there: no copy
    assert "seeded" not in ee.run_calib(tmp_path, tmp_path / "test.jsonl", ["tiel"], {}, gate=True)["tiel-gate"]
    fresh = tmp_path / "f"
    (fresh / "calib" / "tiel").mkdir(parents=True)
    (fresh / "calib" / "tiel" / "T.raw.jsonl").write_text('{"callo_think_raw": 1}\n')
    calls.clear()
    ee.run_calib(fresh, tmp_path / "test.jsonl", ["tiel"], {}, gate=True, fresh=True)
    assert "--resume" not in calls[0] and not list((fresh / "calib" / "tiel-gate").glob("*.raw.jsonl"))


def test_a_gated_calibration_without_the_gate_measured_fails_the_part(tmp_path, monkeypatch):
    monkeypatch.setattr(ee, "_calibrate", _calibrated(tmp_path, PROFILE, []))
    with pytest.raises(RuntimeError, match="no gate measured"):
        ee.run_calib(tmp_path, tmp_path / "test.jsonl", ["tiel"], {}, gate=True)


def test_the_gate_run_serves_the_gated_calibration_s_profile(tmp_path, monkeypatch):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 4))
    for run, prof in (("tiel", PROFILE), ("tiel-gate", GATED)):
        (tmp_path / "calib" / run).mkdir(parents=True)
        (tmp_path / "calib" / run / "t.think.json").write_text(json.dumps(prof))
    started = []
    monkeypatch.setattr(ee, "_start", lambda name, out, **kw: started.append(kw["flags"]) or _Session(f"http://{name}"))
    monkeypatch.setattr(ee, "gguf_traits", lambda model: {"hybrid": True})
    monkeypatch.setattr(ee, "_post", lambda url, body: (-1, {}, None))
    res = {}
    ee.run_serve(tmp_path, items, res, z=0.95, models=["tiel"], gate=True)
    assert started[0][:2] == ["--think-profile", str(tmp_path / "calib" / "tiel-gate" / "t.think.json")]
    assert res["tiel"]["profile"] == str(tmp_path / "calib" / "tiel-gate" / "t.think.json")
    assert ee.served_profile(tmp_path, res["tiel"]) == GATED


def test_the_profile_vs_served_line_judges_the_gated_point_as_e3():
    end = [_end(f"q{i}", ms=1500.0 + i) for i in range(60)]
    rows = [_gated(f"q{i}", ms=1000.0 + i) for i in range(60)]                # served saving 500 ms
    g = ee.gate_result({"ee": rows, "end": end}, profile=GATED)
    pv = g["profile"]
    assert g["verdict"] == "holds" and pv["pass"] is True and pv["profile_dacc"] == 0.0
    assert pv["profile_saving_ms"] == 1400.0 - 900.0 and pv["acc_pass"] and pv["lat_pass"] and pv["flagged"] == 0.3
    far = {**GATED, "ee_gated": {"0.95": {**GATED["ee_gated"]["0.95"], "lat_p50_ms": [100.0, 90.0, 110.0],
                                          "dacc_cap": [-0.2, -0.3, -0.1]}}}
    pv = ee.gate_result({"ee": rows, "end": end}, profile=far)["profile"]
    assert pv["pass"] is False and not pv["acc_pass"] and not pv["lat_pass"]
    assert ee.gate_result({"ee": rows, "end": end}, profile=far)["verdict"] == "holds"     # never moves the verdict
    assert ee.gate_result({"ee": rows, "end": end}, profile=PROFILE)["profile"]["pass"] is None   # no gated point
    assert "no gated profile" in ee.gate_result({"ee": rows, "end": end})["profile"]["error"]
    s = {"parts": {"calib": {"tiel-gate": {"rc": 0, "tail": "done", "seeded": "x"}}}, "errors": {}, "z": 0.95,
         "label": "gate", "gate": ee.gate_result({"ee": rows, "end": end}, profile=GATED)}
    doc = render(s)
    assert "Profile vs served (the gated point, as E3): **PASS**" in doc and "from a copy of the unlabelled" in doc
    assert verdict_lines(s) == ["tiel gate: holds", "tiel gate profile vs served: PASS"]


def test_gate_fresh_is_refused_while_the_gated_directory_holds_a_raw_file(tmp_path, capsys):
    out = tmp_path / "o"
    (out / "calib" / "tiel-gate").mkdir(parents=True)
    (out / "calib" / "tiel-gate" / "T.raw.jsonl").write_text("{}\n")
    assert ee.main(["--out", str(out), "--gate", "--gate-fresh"]) == 2
    said = capsys.readouterr().out
    assert "already holds a raw file (T.raw.jsonl)" in said and f"delete {out / 'calib' / 'tiel-gate'} first" in said


def test_seeding_refuses_several_raw_files_and_logs_the_one_it_copies(tmp_path, caplog):
    own = tmp_path / "calib" / "tiel"
    own.mkdir(parents=True)
    (own / "A.raw.jsonl").write_text("{}\n")
    with caplog.at_level("INFO", logger="bench"):
        copied = ee.seed_gate_calib(tmp_path)
    assert copied == tmp_path / "calib" / "tiel-gate" / "A.raw.jsonl" and str(own / "A.raw.jsonl") in caplog.text
    other = tmp_path / "o"
    (other / "calib" / "tiel").mkdir(parents=True)
    for n in ("A", "B"):
        (other / "calib" / "tiel" / f"{n}.raw.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="holds 2 raw files .A.raw.jsonl, B.raw.jsonl."):
        ee.seed_gate_calib(other)
    assert not (other / "calib" / "tiel-gate").exists()
