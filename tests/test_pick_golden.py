"""Pick preferences golden and compare's report: a fixture reproducing run1 Tiel's rows (TielCoder
UD-Q4_K_XL, the transfer set's test split, 376 questions; clusters and sources renamed in order of first appearance, latencies to
0.1 ms) picks as the reference tables say, and `callo compare` prints the pick, the re-pick and the reasons in canonical,
pasteable form. The bench's results directory is never read: the fixture is a copy of the rows."""
import copy
import json
import random
import shlex
from pathlib import Path

import pytest

from callo.cli import main
from callo.compare import flags, joint, render_md, report_dict, serve_line
from callo.errors import InvalidInput
from callo.think import Preference, bootstrap_percentiles, parse_limit

FIXTURE = Path(__file__).parent / "data" / "run1_tiel.json"


def run1(name: str = "tiel", fingerprint: str = "h" * 64, scale: float = 1.0) -> dict:
    """An old-style think profile (no mean, p99 or max stored; its pick made on p50) built from the fixture's rows,
    every latency times `scale`."""
    d = json.loads(FIXTURE.read_text())
    rows = []
    for i, (cluster, source, *vals) in enumerate(d["rows"]):
        per = {}
        for k, b in enumerate(d["budgets"]):
            correct, lat = bool(vals[2 * k]), scale * float(vals[2 * k + 1])
            per[str(b)] = [correct, 0.1, 0.9, lat, 0 if b == 0 else min(b, 160), None if b == 0 else "natural"]
        rows.append({"q": f"q{i}/q", "cluster": cluster, "source": source, "label": 0, "n": 4, "b": per})
    budgets = {}
    for b in d["budgets"]:
        held = [r for r in rows if str(b) in r["b"]]
        lats, cl, st = [r["b"][str(b)][3] for r in held], [r["cluster"] for r in held], [r["source"] for r in held]
        budgets[str(b)] = {"n": len(held), "acc": sum(r["b"][str(b)][0] for r in held) / len(held), "point": f"{name}@{b}",
                           "lat_p50_ms": list(bootstrap_percentiles(lats, cl, 0.5, seed=d["seed"], strata=st)),
                           "lat_p95_ms": list(bootstrap_percentiles(lats, cl, 0.95, seed=d["seed"], strata=st)),
                           **({"think_mean": 150.0, "forced": 0.05} if b else {})}
    fields = ("temperature", "top_p", "top_k", "min_p")
    return {"callo_think_profile": 1, "metrics": "callo-think-metrics-1", "model_key": "sha256:" + "a" * 64,
            "model_name": name, "created": "2026-10-08T22:00:00Z",
            "comparable": {"data": {"name": "test-s1.jsonl", "sha256": "d" * 64, "questions": len(rows),
                                    "strata": "source"},
                           "seed": d["seed"], "budgets": d["budgets"], "warmup": 3, "audit": 6, "cache": True,
                           "sampling": {"policy": "model", "pinned": dict(zip(fields, (1.0, 0.95, 20, 0.0), strict=True)),
                                        "origin": dict.fromkeys(fields, "gguf")},
                           "readout": {"variant": "main", "top_k": 100, "max_depth": 8, "divergence_mass": 0.1}},
            "environment": {"host": {"fingerprint": "sha256:" + fingerprint, "chip": "Apple M2 Ultra"},
                            "model": {"file": "tiel.gguf", "bytes": 20 * 2**30}},
            "thinking": {"capable": True}, "budgets": budgets,
            "pick": {"budget": 1024, "budgets": [1024], "model": name, "label": f"{name}@1024",
                     "rule": "fastest within noise of the best", "best": f"{name}@1024", "dacc_best": [0.0, 0.0, 0.0],
                     "unsized": [], "max_latency_s": None, "min_accuracy": None, "latency": "p50"},
            "row_fields": ["correct", "nll1", "pmax", "lat_ms", "k", "close"], "rows": rows}


TIEL = run1()


def pref(*limits: str, **kw) -> Preference:
    return Preference(limits=tuple(parse_limit(x) for x in limits), **kw)


def ms3(v) -> list:
    return [round(x / 1000, 2) for x in v]


def test_the_fixture_reproduces_the_reference_table():
    j = joint([TIEL])
    by = {p.label: p for p in j.ranked}
    assert [round(by[f"tiel@{b}"].acc, 3) for b in (0, 256, 512, 1024)] == [0.832, 0.920, 0.936, 0.941]
    assert ms3(by["tiel@0"].lat_mean) == [0.27, 0.26, 0.27] and ms3(by["tiel@256"].lat_mean) == [1.64, 1.58, 1.70]
    assert ms3(by["tiel@512"].lat_mean) == [1.83, 1.72, 1.95] and ms3(by["tiel@1024"].lat_mean) == [1.91, 1.76, 2.07]
    assert [round(by[f"tiel@{b}"].lat_p99[0] / 1000, 2) for b in (0, 256, 512, 1024)] == [0.44, 3.04, 5.80, 10.00]
    assert [round(by[f"tiel@{b}"].lat_max / 1000, 2) for b in (0, 256, 512, 1024)] == [0.46, 3.08, 5.86, 11.44]
    assert ms3(by["tiel@1024"].lat_p99) == [10.0, 6.54, 11.36]                    # one rank moves it by 2 s
    assert by["tiel@1024"].tail_n == {"p50": 189, "p95": 19, "p99": 4, "max": 1}


@pytest.mark.parametrize("kw, label, best", [
    ({}, "tiel@512", "tiel@1024"),                                              # the default: mean
    ({"latency": "p50"}, "tiel@1024", "tiel@1024"),                             # the original p50 rule, as recorded
    ({"price": 1.0}, "tiel@512", "tiel@1024"),
    ({"price": 5.0}, "tiel@512", "tiel@1024"),
    ({"price": 10.0}, "tiel@0", "tiel@0"),                                      # a second outweighs every gain
])
def test_the_golden_picks(kw, label, best):
    c = joint([TIEL], pref=Preference(**kw)).chosen
    assert (c.point.label, c.best.label) == (label, best)


def test_the_noise_numbers_of_section_3_and_section_1():
    c = joint([TIEL]).chosen
    assert [round(100 * x, 1) for x in c.dacc_best] == [-0.5, -1.3, 0.0]          # 512 vs 1024: within noise
    pairs = {(x["a"], x["b"]): x["dacc"] for x in joint([TIEL]).pairs}
    assert [round(100 * x, 1) for x in pairs[("tiel@1024", "tiel@256")]] == [-2.1, -3.7, -0.8]
    for r, want in ((1.0, [-0.45, -1.22, 0.07]), (5.0, [-0.13, -0.76, 0.36])):
        c = joint([TIEL], pref=Preference(price=r)).chosen
        assert [round(x, 2) for x in c.dscore_best] == want and c.rule == "fastest within noise of the best score"
    c = joint([TIEL], pref=Preference(price=5.0)).chosen
    assert round(c.score["best"], 1) == 84.6 and round(c.score["pick"], 1) == 84.5


def test_a_p99_limit_moves_the_target_but_keeps_the_pick():
    j = joint([TIEL], pref=pref("p99=6s"))
    assert (j.chosen.point.label, j.chosen.best.label) == ("tiel@512", "tiel@512")
    assert j.chosen.excluded == {"tiel@1024": "p99 10.00 s > 6 s"}
    point, lo, hi = {(x["a"], x["b"]): x["dacc"] for x in j.pairs}[("tiel@256", "tiel@512")]   # neighbours: 512 − 256
    assert [round(-100 * point, 1), round(-100 * hi, 1), round(-100 * lo, 1)] == [-1.6, -3.2, -0.3]   # 256 − M


def test_the_golden_no_pick_reasons():
    tight = joint([TIEL], pref=pref("p99=0.3s")).chosen
    assert tight.point is None and tight.reason == ("no point meets --limit p99=0.3s: all 4 excluded; nearest tiel@0 "
                                                    "at p99 0.44 s, from 4 of 376")   # budget 0 fails it: no hint
    hinted = joint([TIEL], pref=pref("max=2s", min_accuracy=0.9)).chosen
    assert hinted.point is None and hinted.reason == (
        "--limit max=2s excluded 3 of 4, nearest tiel@256 at max 3.08 s; --min-accuracy .9: tiel@0 is .832; to add a "
        "point that meets it: a new calibrate run with --deadlines 2000 (a run without --deadlines cannot resume with "
        "them)")
    md = render_md(joint([TIEL], pref=pref("max=2s", min_accuracy=0.9)))
    assert f"**pick: none** ({hinted.reason})" in md and "serve with:" not in md


def test_every_printed_limit_pastes_back_as_the_argv_that_made_it():
    for argv in (["--limit", "p99=0.3s"], ["--limit", "max=2s"], ["--limit", "p99=6s", "--limit", "mean=2.5s"]):
        p = Preference(limits=tuple(parse_limit(x) for x in argv[1::2]))
        j = joint([TIEL], pref=p)
        printed = [lim.canonical for lim in p.limits]
        assert sum((shlex.split(t) for t in printed), []) == argv
        text = render_md(j) + json.dumps(report_dict(j))
        assert all(t in text for t in printed)
        reason = j.chosen.reason or ""
        assert "<" not in reason and ">" not in reason


def test_compares_report_names_the_pick_the_re_pick_and_the_preference():
    md = render_md(joint([TIEL], pref=Preference(price=5.0)), sources=["tiel.think.json"])
    assert ("**pick: tiel@512** (fastest within noise of the best score, --price 5pp/s on mean); best tiel@1024 84.6 "
            "pp; Δacc vs it -0.5pp [-1.3, +0.0], Δscore -0.13pp [-0.76, +0.36]") in md
    assert "the profile picked tiel@1024 on p50; this pick is on mean" in md
    assert "serve with: `callo serve --upstream URL --think-profile tiel.think.json --think-budget 512`" in md
    assert "Pick statistic mean; preference: --price 5pp/s on mean." in md
    assert "Paired Δacc (b − a) over the shared questions [95 % cluster-bootstrap CI], and Δscore at --price 5pp/s" in md
    assert "- tiel@1024 → tiel@256: -2.1pp [-3.7, -0.8], Δscore -" in md and "(n=376, against the best)" in md
    same = render_md(joint([TIEL], pref=Preference(latency="p50")))
    assert "the profile picked" not in same and "**pick: tiel@1024** (fastest within noise of the best, --latency p50)" in same
    d = report_dict(joint([TIEL], pref=Preference(price=5.0)))
    assert d["repick"] == ["the profile picked tiel@1024 on p50; this pick is on mean"]
    assert d["filters"]["price_pp_per_s"] == 5.0 and d["pick"]["preference"]["price_pp_per_s"] == 5.0
    assert d["pick"]["dscore_best"] == pytest.approx([-0.1308, -0.7646, 0.3556], abs=5e-4)
    assert d["dacc"].startswith("point − reference: pairs[].dacc (and dscore) is b − a")


def test_across_two_profiles_every_pair_uses_the_one_fixed_row_set_and_a_price_flags_the_hosts():
    other = run1("qwen", fingerprint="q" * 64, scale=1.5)                # no latency ties with tiel
    shuffled = copy.deepcopy(other)
    random.Random(2).shuffle(shuffled["rows"])
    a = joint([TIEL, other], pref=Preference(price=1.0))
    b = joint([TIEL, shuffled], pref=Preference(price=1.0))
    assert a.pairs == b.pairs and a.fixed == b.fixed and all("dscore" in x for x in a.pairs)
    fl = flags([TIEL, other], Preference(price=1.0))
    at = fl.index(next(f for f in fl if f.startswith("latencies from different machines")))
    assert fl[at + 1] == "the pick prices latency across hosts"
    assert "the pick prices latency across hosts" not in flags([TIEL, other], Preference())
    assert "the pick prices latency across hosts" in flags([TIEL, other], Preference(latency="p50"))
    assert "the pick prices latency across hosts" not in flags([TIEL, run1("qwen")], Preference(price=1.0))
    md = render_md(a)
    assert "tiel: the profile picked tiel@1024 on p50; this pick is on mean" in md
    sha = copy.deepcopy(other)
    sha["comparable"]["data"]["sha256"] = "e" * 64
    with pytest.raises(InvalidInput, match="different data"):
        joint([TIEL, sha], pref=Preference(price=1.0))


def test_the_hint_is_per_model_in_compare():
    with_deadline = run1("qwen")
    lats = [r["b"]["256"][3] * 0.6 for r in with_deadline["rows"]]
    for r, lat in zip(with_deadline["rows"], lats, strict=True):
        r["d"] = {"1500": [r["b"]["256"][0], 0.1, 0.9, lat, 120, "deadline"]}
    with_deadline["deadlines"] = {"1500": {"cap": 1024, "n": 376, "acc": 0.9, "T": 1.0,
                                           "lat_p50_ms": [900.0, 800.0, 1000.0], "lat_p95_ms": [1800.0, 1700.0, 1900.0]}}
    j = joint([TIEL, with_deadline], pref=pref("max=1.5s", min_accuracy=0.95))
    assert j.chosen.point is None and j.chosen.reason.endswith(
        "to add a point that meets it: a new calibrate run with --deadlines 1500 (a run without --deadlines cannot "
        "resume with them)")                                                     # tiel qualifies; qwen has a deadline
    only_deadline = joint([with_deadline], pref=pref("max=1.5s", min_accuracy=0.95))
    assert only_deadline.chosen.point is None and "--deadlines" not in only_deadline.chosen.reason


def test_a_deadline_pick_in_compare_serves_think_deadline():
    p = run1("qwen")
    for r in p["rows"]:
        r["d"] = {"2000": [True, 0.1, 0.9, 1500.0, 120, "deadline"]}
    p["deadlines"] = {"2000": {"cap": 1024, "n": 376, "acc": 1.0, "T": 1.0, "lat_p50_ms": [1500.0] * 3,
                               "lat_p95_ms": [1500.0] * 3}}
    j = joint([p], pref=pref("max=2s"))
    assert j.chosen.point.label == "qwen@2000ms"
    assert serve_line(j, ["q.think.json"]) == "callo serve --upstream URL --think-profile q.think.json --think-deadline 2000"
    d = report_dict(j)["pick"]
    assert (d["kind"], d["budget"], d["deadline_ms"], d["cap"]) == ("deadline", None, 2000, 1024)


def test_a_price_needs_every_questions_latency(tmp_path, capsys):
    bare = copy.deepcopy(TIEL)
    bare["row_fields"] = ["correct", "nll1", "pmax"]
    for e in bare["budgets"].values():                                   # stored, so the profile itself is valid
        e.update({"lat_mean_ms": [1.0] * 3, "lat_p99_ms": [1.0] * 3, "lat_max_ms": 1.0})
    with pytest.raises(InvalidInput, match="--price needs every question's latency: tiel@0's rows carry no lat_ms"):
        joint([bare], pref=Preference(price=1.0))
    assert joint([bare]).chosen.point is not None
    path = tmp_path / "bare.think.json"
    path.write_text(json.dumps(bare))
    assert main(["compare", str(path), "--price", "1"]) == 2
    assert "rows carry no lat_ms" in capsys.readouterr().err


def test_the_report_says_what_each_limit_and_a_p95_score_rest_on():
    md = render_md(joint([TIEL], pref=pref("p99=6s")))
    assert "Pick statistic mean; preference: --limit p99=6s; --limit p99=6s: p99 from 4 of 376." in md
    j = joint([TIEL], pref=Preference(latency="p95", price=1.0))
    assert render_md(j).count("(p95 from 19 of 376) (n=376, ") == len(j.pairs) > 0
    assert "(p95 from" not in render_md(joint([TIEL], pref=Preference(price=1.0)))                     # mean: none


def test_the_delta_score_header_prints_the_price_as_the_pick_line_does():
    md = render_md(joint([TIEL], pref=Preference(price=0.1234567)), sources=["tiel.think.json"])
    assert "Δscore at --price 0.1234567pp/s on mean (pp, descriptive)" in md
