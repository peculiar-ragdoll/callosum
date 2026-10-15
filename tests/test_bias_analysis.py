"""The pure half of the v0.3 position-bias acceptance bench: the gate arithmetic on fixed readouts, the offline
judgement of a model's raw files, the report, and the offline refit."""
import hashlib
import json
from collections import Counter

import numpy as np
import pytest
from fake_callo import CalloStub
from fake_upstream import PlantedBias
from test_calibrate_bias import set_text

from bench import bias_analysis as ba
from bench.bias_refit import main as refit_main
from bench.bias_refit import refit
from bench.report_v03_bias import main as report_main
from bench.report_v03_bias import render
from callo.bias import Item, labelled_rows, offsets_from_prior, read_set
from callo.calibrate import fit_temperature, nll
from callo.cli import main
from callo.engine import Engine
from callo.errors import InvalidInput
from callo.significance import Decision

PRIOR = [0.5, 0.3, 0.2]


def _items(n, count, tag="h"):
    return [Item(uid=f"{tag}{i}", record_id="r", qid="q", cluster=f"{tag}{i}", source="syn", kind="plain", n=n,
                 texts=tuple(f"t{k}" for k in range(n)), instructions="", label=0, base={"state": ""})
            for i in range(count)]


def _planted(items, prior, seed=0, margin=0.5):
    """Readouts under PriDe's factorisation: content (option 0 ahead by `margin`) times a position prior, one shared
    prior or one per question."""
    rng = np.random.default_rng(seed)
    priors = prior if np.ndim(prior) == 2 else [prior] * len(items)
    out = {}
    for it, prior in zip(items, priors, strict=True):
        c = rng.normal(0, 1, it.n)
        c[0] = c.max() + margin
        for s in range(it.n):
            z = np.array([c[(k + s) % it.n] for k in range(it.n)]) + np.log(prior)
            p = np.exp(z - z.max())
            out[(it.uid, s)] = (tuple(p / p.sum()), 0)
    return out


def test_b1_passes_when_the_true_prior_is_removed_and_reports_small_n_as_not_tested():
    items = _items(3, 60) + _items(4, 20, tag="f")
    readings = {**_planted(items[:60], PRIOR), **_planted(items[60:], [0.25] * 4)}
    offs = {3: offsets_from_prior(PRIOR), 4: offsets_from_prior([0.4, 0.3, 0.2, 0.1])}
    g = ba.gate_b1(items, readings, offs, resamples=300)
    assert g["per_n"]["3"]["pass"] is True and g["per_n"]["3"]["signed_diff_ci"][1] < 0
    assert g["per_n"]["4"] == {"tested": False, "questions": 20, "note": "not tested held-out"}
    assert g["pass"] is True
    wrong = ba.gate_b1(items, readings, {3: offsets_from_prior([0.2, 0.3, 0.5])}, resamples=300)
    assert wrong["per_n"]["3"]["pass"] is False and wrong["pass"] is False
    assert ba.gate_b1(items[60:], readings, offs, resamples=50)["pass"] is None     # nothing tested


def _scaled(prior, power):
    p = np.asarray(prior, dtype=float) ** power
    return p / p.sum()


def test_b1_judges_the_signed_deviation_along_the_fitted_direction():
    """A partial correction passes; a wrong-direction prior, an overshoot and a correction that moves
    nothing fail. max-|.| is reported, not gated."""
    items = _items(3, 60)
    readings = _planted(items, PRIOR)

    def one(prior):
        return ba.gate_b1(items, readings, {3: offsets_from_prior(prior)}, resamples=300, draws=300)["per_n"]["3"]
    half = one(_scaled(PRIOR, 0.5))
    assert half["tested"] and half["pass"] is True and half["signed_diff_ci"][1] < 0
    assert 0 < half["signed"][1] < half["signed"][0] and half["debiased"] < half["raw"]
    wrong = one([0.2, 0.3, 0.5])                                   # points away from the held-out bias
    assert wrong["signed"][0] < 0 and wrong["signed_diff_ci"][1] < 0 and wrong["pass"] is False
    over = one(_scaled(PRIOR, 3))                                 # removes the bias and then some
    assert over["signed"][1] < -over["signed"][0] < 0 and over["pass"] is False
    still = one([1 / 3] * 3)                                      # a uniform prior moves nothing
    assert still["signed"] == [0.0, 0.0] and still["pass"] is False


def test_b1_passes_a_real_correction_near_the_noise_floor_that_max_abs_would_fold():
    """Near the floor the paired bootstrap of max-|.| folds resamples at 0, so its CI reaches past 0 for a
    correction that removed most of a detected bias; the signed statistic does not fold."""
    items = _items(2, 400)
    readings = _planted(items, np.random.default_rng(0).dirichlet([4, 3.5], size=400))
    rows, clusters = ba._question_meanp(items, readings, None, 2)
    prior = rows.mean(axis=0)
    offs = offsets_from_prior(prior)
    g = ba.gate_b1(items, readings, {2: offs}, resamples=300, draws=300)["per_n"]["2"]
    assert g["tested"] and g["pass"] is True and g["signed"][1] < 0.5 * g["signed"][0]
    deb, _ = ba._question_meanp(items, readings, offs, 2)
    lo, hi = np.percentile(ba._paired(rows, deb, clusters, ba._bias, resamples=300, seed=0), [2.5, 97.5])
    assert lo < 0 <= hi                                           # the max-|.| CI would have failed it


def _null_rows(count, n, seed=5):
    """Questions whose own position preferences differ but share no direction: no detectable held-out bias."""
    items = _items(n, count, tag="f")
    return items, _planted(items, np.random.default_rng(seed).dirichlet([30] * n, size=count))


def test_b1_does_not_gate_an_n_whose_held_out_bias_is_not_detectable():
    """Where the held-out readouts show no shared position bias there is nothing to remove, so that n is
    reported, not gated by the signed rule; Holm runs across the n with entries; a mild correction there leaves the
    harm leg clear."""
    three = _items(3, 60)
    four, r4 = _null_rows(80, 4)
    readings = {**_planted(three, PRIOR), **r4}
    offs = {3: offsets_from_prior(PRIOR), 4: offsets_from_prior([0.26, 0.25, 0.25, 0.24])}
    g = ba.gate_b1(three + four, readings, offs, resamples=100, draws=300)
    assert g["per_n"]["3"]["tested"] is True and g["per_n"]["3"]["detect_p_holm"] <= 0.05
    f = g["per_n"]["4"]
    assert f["tested"] is False and f["note"].startswith("no detectable held-out bias (p_holm")
    assert f["detect_p_holm"] > 0.05 and "signed" not in f and f["harm"] is False and f["harm_p_holm"] > 0.05
    assert g["pass"] is True and g["harm"] == [] and list(g["per_n"]) == ["3", "4"]
    alone = ba.gate_b1(four, r4, {4: offs[4]}, resamples=50, draws=300)
    assert alone["pass"] is None                                 # nothing tested and no harm: vacuous, not a pass


def test_the_harm_leg_never_fails_a_correction_that_lowered_max_abs(monkeypatch):
    """The harm family is smaller than the detection family, so an n can turn "detectable" after a
    slight improvement; harm also needs the debiased max-|.| above the raw one."""
    items = _items(3, 60)
    readings = _planted(items, PRIOR)
    calls = []

    def detect(rows, **kw):                      # raw: not detectable; debiased: "detectable"
        calls.append(sorted(rows))
        return {n: Decision(n, len(calls) > 1, 0.1, 0.01, 0.01, None) for n in rows}
    monkeypatch.setattr(ba, "held_out_detection", detect)
    g = ba.gate_b1(items, readings, {3: offsets_from_prior(PRIOR)}, resamples=50)
    three = g["per_n"]["3"]
    assert three["tested"] is False and three["harm_max_abs"][1] < three["harm_max_abs"][0]
    assert three["harm"] is False and g["harm"] == [] and g["pass"] is None and calls == [[3], [3]]


def test_b1_harm_leg_fails_a_correction_that_creates_a_bias_where_none_was_detectable():
    """The transfer-mismatch failure (a correction fitted elsewhere biasing unbiased readouts)."""
    four, r4 = _null_rows(80, 4)
    g = ba.gate_b1(four, r4, {4: offsets_from_prior([0.55, 0.25, 0.12, 0.08])}, resamples=50, draws=300)
    assert g["per_n"]["4"]["tested"] is False and g["per_n"]["4"]["harm"] is True
    assert g["per_n"]["4"]["harm_p_holm"] <= 0.05 and g["harm"] == ["4"] and g["pass"] is False
    assert g["per_n"]["4"]["harm_max_abs"][1] > g["per_n"]["4"]["harm_max_abs"][0]


def test_b1_with_no_complete_question_is_not_tested(monkeypatch):
    monkeypatch.setattr(ba, "B1_MIN_QUESTIONS", 0)
    g = ba.gate_b1(_items(3, 4), {}, {3: offsets_from_prior(PRIOR)}, resamples=50)
    assert g == {"per_n": {"3": {"tested": False, "questions": 0, "note": "not tested held-out"}}, "harm": [],
                 "pass": None}


def test_b2_compares_rotation_consistency_at_the_ci_lower_bound():
    items = _items(3, 60)
    readings = _planted(items, PRIOR)
    good = ba.gate_b2(items, readings, {3: offsets_from_prior(PRIOR)}, resamples=300)
    assert good["debiased"] >= good["raw"] and good["pass"] is True
    bad = ba.gate_b2(items, readings, {3: offsets_from_prior([0.05, 0.05, 0.9])}, resamples=300)
    assert bad["diff_ci"][0] < -ba.B2_TOLERANCE and bad["pass"] is False
    broke = [f for f in bad["flips"] if f["flip"] == "broke"]           # descriptive: which questions, how close
    assert broke and all(f["n"] == 3 and f["source"] == "syn" and 0 <= f["margin"][0] <= 1 for f in broke)


def test_b2_passes_a_lower_bound_of_exactly_minus_1_pp(monkeypatch):
    """A pre-registered lower bound of exactly -1 pp passes. As floats 0.79 - 0.80 is -0.010000000000000009, so the
    bound is compared with a 1e-12 tolerance (far below one question in 958); a bound 0.01 pp worse still fails."""
    items = _items(3, 60)
    readings = _planted(items, PRIOR)
    assert 0.79 - 0.80 < -ba.B2_TOLERANCE
    monkeypatch.setattr(ba, "_paired", lambda *a, **k: np.full(50, 0.79 - 0.80))
    assert ba.gate_b2(items, readings, {3: offsets_from_prior(PRIOR)}, resamples=50)["pass"] is True
    monkeypatch.setattr(ba, "_paired", lambda *a, **k: np.full(50, -0.0101))
    assert ba.gate_b2(items, readings, {3: offsets_from_prior(PRIOR)}, resamples=50)["pass"] is False


def test_b2_tolerates_a_loss_within_1_pp_at_the_lower_bound():
    """399 questions consistent either way and 1 that debiasing makes inconsistent: the lower bound is below 0 but
    within the 1 pp tolerance, so B2 PASSES (a `lo >= 0` gate would fail it)."""
    items = _items(3, 400)
    readings = {**_planted(items[:399], [1 / 3] * 3, margin=10), **_planted(items[399:], [1 / 3] * 3, seed=1)}
    g = ba.gate_b2(items, readings, {3: offsets_from_prior([0.2, 0.2, 0.6])}, resamples=300)
    assert g["raw"] == 1.0 and g["debiased"] == pytest.approx(399 / 400)
    assert [(f["q"], f["flip"]) for f in g["flips"]] == [(items[399].uid, "broke")]
    assert -ba.B2_TOLERANCE < g["diff_ci"][0] < 0 and g["pass"] is True


def test_b2_with_no_complete_question_is_an_error_not_a_crash():
    g = ba.gate_b2(_items(3, 4), {}, {3: offsets_from_prior(PRIOR)}, resamples=50)
    assert g == {"questions": 0, "pass": None, "error": "no held-out question was read under every rotation"}


def _lab(hits, nll, *, rows=1000, raw_hits=800, raw_nll=0.50):
    return {"T1": {"raw": {"accuracy": raw_hits / rows, "hits": raw_hits, "nll": raw_nll, "rows": rows},
                   "debiased": {"accuracy": hits / rows, "hits": hits, "nll": nll, "rows": rows}}}


def test_b5_bounds_the_accuracy_drop_and_the_nll_rise():
    ok = ba.gate_b5(_lab(796, 0.5019))
    assert ok["pass"] is True and ok["d_accuracy"] == pytest.approx(-0.004) and ok["d_nll"] == pytest.approx(0.0019)
    assert ba.gate_b5(_lab(794, 0.49))["pass"] is False                          # -0.6 pp
    assert ba.gate_b5(_lab(810, 0.503))["pass"] is False                         # NLL +0.003
    assert ba.gate_b5({"T1": {}}) == {"pass": None, "error": "no labelled held-out rows"}


def test_b5_passes_an_exactly_attained_bound_and_fails_one_hit_worse():
    """A pre-registered drop of exactly 0.5 pp passes; only < -0.5 pp falsifies. As floats, 636/800 - 640/800 is
    -0.0050000000000000044, so the accuracy leg is compared in hits (exact rationals), not in float accuracies."""
    assert 636 / 800 - 640 / 800 < -ba.B5_MAX_ACC_DROP                          # the float trap this guards against
    for rows, raw_hits, drop in ((800, 640, 4), (200, 160, 1), (1000, 800, 5), (2400, 1900, 12)):
        assert ba.gate_b5(_lab(raw_hits - drop, 0.5, rows=rows, raw_hits=raw_hits))["pass"] is True, rows
        assert ba.gate_b5(_lab(raw_hits - drop - 1, 0.5, rows=rows, raw_hits=raw_hits))["pass"] is False, rows
    assert 0.502 - 0.5 > ba.B5_MAX_NLL_RISE                                      # NLL is continuous: 1e-12 tolerance
    assert ba.gate_b5(_lab(800, 0.502))["pass"] is True
    assert ba.gate_b5(_lab(800, 0.50201))["pass"] is False


def _gates(**over):
    ok = {"pass": True}
    m = {g: dict(ok) for g in ("b0", "b2", "b3", "b4", "b5")}
    m["b1"] = {"pass": True, "per_n": {"2": {"tested": True}, "6": {"tested": False}}}
    m.update(over)
    return m


def test_the_prereg_verdict_needs_every_required_gate_and_one_tested_n():
    q = {k: v for k, v in _gates().items() if k in ("b1", "b2", "b5")}
    assert ba.prereg_verdict({"tiel_kvu": _gates(), "qwen9b": q}) == {
        "verdict": "CONFIRMED", "b1_tested": ["tiel_kvu n=2", "qwen9b n=2"], "reasons": []}
    failed = ba.prereg_verdict({"tiel_kvu": _gates(b2={"pass": False}), "qwen9b": q})
    assert failed["verdict"] == "FALSIFIED" and failed["reasons"] == ["tiel_kvu B2 FAIL"]
    vacuous = {"pass": None, "per_n": {"2": {"tested": False}}}
    assert ba.prereg_verdict({"tiel_kvu": _gates(b1=vacuous), "qwen9b": {**q, "b1": vacuous}})["verdict"] == \
        "INCONCLUSIVE"
    assert ba.prereg_verdict({"tiel_kvu": _gates(b1=vacuous), "qwen9b": q})["verdict"] == "CONFIRMED"
    kept_fail = ba.prereg_verdict({"tiel_kvu": _gates(), "qwen9b": {**q, "b1": {"pass": False}, "error": "boom"}})
    assert kept_fail["verdict"] == "FALSIFIED"                    # a FAIL measured before the model errored is kept
    nothing_corrected = {"pass": None, "note": "no n has an entry: nothing is corrected"}
    assert ba.prereg_verdict({"tiel_kvu": _gates(), "qwen9b": {**q, "b5": nothing_corrected}})["verdict"] == \
        "CONFIRMED"
    errs = ba.prereg_verdict({"tiel_kvu": _gates(b4={"pass": None, "error": "B4 setup: x"}),
                              "qwen9b": {"error": "boom"}})
    assert errs["verdict"] == "ERROR" and errs["reasons"] == ["tiel_kvu B4: B4 setup: x", "qwen9b: boom"]
    missing = ba.prereg_verdict({"tiel_kvu": {k: v for k, v in _gates().items() if k != "b3"}, "qwen9b": q})
    assert missing["verdict"] == "ERROR" and missing["reasons"] == ["tiel_kvu B3: not run"]
    both = ba.prereg_verdict({"tiel_kvu": _gates(b2={"pass": False}), "qwen9b": {"error": "boom"}})
    assert both["verdict"] == "FALSIFIED"                                         # a FAIL decides even beside an ERROR
    read_err = {"pass": None, "error": "not judged on partial held-out data: eval exited 1 with 17 readouts"}
    for gate in ("b1", "b5"):                   # an error is never the vacuous null of B1 (nothing tested) / B5
        for who in ("tiel_kvu", "qwen9b"):
            models = {"tiel_kvu": _gates(), "qwen9b": dict(q)}
            models[who] = {**models[who], gate: read_err}
            got = ba.prereg_verdict(models)
            assert got["verdict"] == "ERROR" and got["reasons"] == [f"{who} {gate.upper()}: {read_err['error']}"]
    no_b3 = _gates(b3={"pass": None, "note": "not run"})          # an unjudged B3 never supports CONFIRMED
    assert ba.prereg_verdict({"tiel_kvu": no_b3, "qwen9b": q}) == {
        "verdict": "ERROR", "b1_tested": ["tiel_kvu n=2", "qwen9b n=2"], "reasons": ["tiel_kvu B3: not run"]}


def test_b5_judges_only_the_rows_the_correction_touches(tmp_path, monkeypatch):
    """B5 pooled over every n would dilute a loss at the one corrected n with unchanged rows."""
    monkeypatch.setattr(ba, "B1_MIN_QUESTIONS", 10)
    sets = _model_dir(tmp_path)
    ev = sets["eval"]
    ev.write_text(set_text(ns=(3, 4), per=30).replace("state ", "eval "))
    items = read_set(ev.read_text()).items
    readings = {**_planted([it for it in items if it.n == 3], PRIOR),
                **_planted([it for it in items if it.n == 4], [0.25] * 4, seed=1)}
    _write_raw(tmp_path / "eval.raw.jsonl", readings, hashlib.sha256(ev.read_bytes()).hexdigest())
    out = ba.analyse(tmp_path, sets, READS)
    assert out["b5"]["rows"] == 90 and out["labelled"]["T1"]["raw"]["rows"] == 210
    assert out["labelled_corrected"]["T1"]["raw"]["rows"] == 90
    (tmp_path / "fit.json").write_text(json.dumps({"entries": {}, "dropped": {"3": "x"}}))
    assert ba.analyse(tmp_path, sets, READS)["b5"] == {"pass": None, "note": "no n has an entry: nothing is corrected"}


def test_floored_rates_and_labelled_metrics():
    items = _items(3, 40)
    idle = _planted(items, PRIOR)
    floored = dict(idle)
    floored[(items[0].uid, 1)] = (floored[(items[0].uid, 1)][0], 1)
    rates = ba.floored_rates(items, floored)
    assert rates == {"syn": {"3": {"questions": 40, "floored": 1, "rate": 0.025}}}
    lab = ba.labelled(items, idle, {3: offsets_from_prior(PRIOR)}, dev_items=items, dev_readings=idle)
    dev_rows = labelled_rows(items, idle, {})
    t = fit_temperature(dev_rows)
    assert lab["T1"]["debiased"]["accuracy"] == 1.0 and t != 1.0
    assert lab["Tdev"]["raw"]["T"] == t and lab["Tdev"]["raw"]["nll"] == nll(dev_rows, t)
    assert set(lab["T1"]["raw"]) == {"accuracy", "hits", "nll", "ece10", "rows"}
    assert lab["T1"]["debiased"]["hits"] == lab["T1"]["debiased"]["rows"] == 120     # B5 compares in hits


def test_b3_cpu_cost_is_measured_per_readout():
    r = ba.b3_cpu(reps=2000)
    assert r["added_ms"] < ba.B3_MAX_MS and r["pass"] is True and r["reps"] == 2000


def test_b3_cpu_fails_above_its_budget(monkeypatch):
    monkeypatch.setattr(ba, "B3_MAX_MS", 0.0)
    assert ba.b3_cpu(reps=200)["pass"] is False


def test_b3_cpu_times_the_serving_code_path(monkeypatch):
    """B3 times what is served: callo.engine's Engine.bias_for and bounded_readout, not a copy of their logic."""
    seen = Counter()
    bias_for, readout = Engine.bias_for, ba.bounded_readout

    def counted_bias_for(self, q, kind):
        got = bias_for(self, q, kind)
        seen[got[1]] += 1
        return got

    def counted_readout(*a):
        seen["offsets" if len(a) > 4 and a[4] is not None else "raw"] += 1
        return readout(*a)

    monkeypatch.setattr(Engine, "bias_for", counted_bias_for)
    monkeypatch.setattr(ba, "bounded_readout", counted_readout)
    ba.b3_cpu(reps=10)
    assert seen == {None: 10, "applied": 10, "raw": 10, "offsets": 10}


def _n_of(body):
    return len(next(iter(body["questions"].values()))["criteria"])


def test_b3_bodies_spread_over_the_kept_n():
    """The held-out file is sorted by n, so its first 50 rows were all n = 2, and a profile that
    dropped n = 2 would have failed B3 on `applied` with a correct serving path."""
    items = _items(2, 60) + _items(3, 60, tag="t") + _items(4, 60, tag="f")
    assert Counter(_n_of(b) for b in ba.b3_bodies(items, ["3", "4"])) == {3: 25, 4: 25}
    assert Counter(_n_of(b) for b in ba.b3_bodies(items, {"4": {}})) == {4: 50}
    assert Counter(_n_of(b) for b in ba.b3_bodies(items, [])) == {2: 17, 3: 17, 4: 17}


def _write_raw(path, readings, sha=None):
    head = {"callo_bias_raw": 1, **({"set": {"sha256": sha}} if sha else {})}
    rows = [head] + [{"type": "readout", "q": q, "n": len(p), "s": s, "probs": list(p), "miss": m, "slot": 0}
                                      for (q, s), (p, m) in readings.items()]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_load_raw_reads_through_the_one_raw_reader(tmp_path):
    """callo.calibrate_bias.read_raw is the one parser of the format: a missing file is empty, a hard kill's
    incomplete last line is left out, and a file that is not a raw file is refused, not misread."""
    assert ba.load_raw(tmp_path / "none.raw.jsonl") == ({}, {}, [])
    p = tmp_path / "x.raw.jsonl"
    _write_raw(p, _planted(_items(3, 2), PRIOR))
    p.write_text(p.read_text() + '{"type": "check", "index": 1, "session": 0, "readouts": 6, "failed": []}\n'
                 + '{"type": "readout", "q": "h0"')
    header, readings, checks = ba.load_raw(p)
    assert header == {"callo_bias_raw": 1} and len(readings) == 6 and ba.raw_readouts(p) == 6
    assert checks == [{"index": 1, "session": 0, "readouts": 6, "failed": []}]
    (tmp_path / "bad.raw.jsonl").write_text('{"not": "raw"}\n')
    with pytest.raises(InvalidInput, match="raw header"):
        ba.load_raw(tmp_path / "bad.raw.jsonl")


def _model_dir(d):
    """One model's bench directory: the fit and eval sets, their raw files with the planted prior, and the fit
    profile."""
    sets = {}
    for name, per in (("fit", 30), ("eval", 30)):
        f = d / f"{name}.jsonl"
        f.write_text(set_text(ns=(3,), per=per).replace("state ", f"{name} "))
        sets[name] = f
        _write_raw(d / f"{name}.raw.jsonl", _planted(read_set(f.read_text()).items, PRIOR),
                   hashlib.sha256(f.read_bytes()).hexdigest())
    (d / "fit.json").write_text(json.dumps({"entries": {"3": {"prior": PRIOR}}, "dropped": {"2": "x"}}))
    return sets


READS = {"eval": {"rc": 0, "readouts": 90, "attempts": 1}}


def test_analyse_judges_complete_reads_and_refuses_partial_ones(tmp_path, monkeypatch):
    """B1/B2 on partial data would silently drop unread questions, and with them maybe an n."""
    monkeypatch.setattr(ba, "B1_MIN_QUESTIONS", 10)
    sets = _model_dir(tmp_path)
    ok = ba.analyse(tmp_path, sets, READS)
    assert ok["b1"]["per_n"]["3"]["tested"] is True and ok["b1"]["pass"] is True and ok["b2"]["pass"] is True
    assert ok["b5"]["pass"] is True and ok["b5"]["rows"] == 90 and ok["labelled"]["Tdev"] == {}
    assert ok["entries"] == ["3"] and ok["dropped"] == {"2": "x"}
    assert set(ok["floored_rates"]) == {"fit", "eval"}
    failed = ba.analyse(tmp_path, sets, {"eval": {"rc": 1, "readouts": 17, "attempts": 2}})
    assert failed["b1"] == failed["b2"] == failed["b5"] == {
        "pass": None, "error": "not judged on partial held-out data: eval exited 1 with 17 readouts"}
    assert failed["b1"] is not failed["b2"] and failed["b2"] is not failed["b5"]   # no shared (aliased) dict
    assert "eval not run" in ba.analyse(tmp_path, sets, {})["b1"]["error"]
    (tmp_path / "eval.raw.jsonl").unlink()
    assert "eval left no raw file" in ba.analyse(tmp_path, sets, READS)["b2"]["error"]
    (tmp_path / "fit.raw.jsonl").unlink()                               # a fit refused at check 0 writes no raw
    assert ba.analyse(tmp_path, sets, {})["floored_rates"] == {"fit": {}}
    assert ba.raw_readouts(tmp_path / "fit.raw.jsonl") == 0


def test_analyse_refuses_raw_files_of_another_set(tmp_path, monkeypatch):
    """Readings are keyed by line-numbered uids: another set's raw file would pair readouts with the wrong questions."""
    monkeypatch.setattr(ba, "B1_MIN_QUESTIONS", 10)
    sets = _model_dir(tmp_path)
    good = ba.analyse(tmp_path, sets, READS)
    raw = tmp_path / "eval.raw.jsonl"
    raw.write_text(raw.read_text().replace(hashlib.sha256(sets["eval"].read_bytes()).hexdigest(), "f" * 64, 1))
    bad = ba.analyse(tmp_path, sets, READS)
    assert bad["b1"] == bad["b2"] == {"pass": None, "error": "not judged on another set's readouts: eval.raw.jsonl "
                                      f"was measured on set sha256 {'f' * 64}, not "
                                      f"{hashlib.sha256(sets['eval'].read_bytes()).hexdigest()}"}
    raw.write_text(raw.read_text().replace("f" * 64, hashlib.sha256(sets["eval"].read_bytes()).hexdigest()))
    fit = tmp_path / "fit.raw.jsonl"                                # another set's fit: its descriptives are skipped
    fit.write_text(fit.read_text().replace(hashlib.sha256(sets["fit"].read_bytes()).hexdigest(), "0" * 64))
    other_fit = ba.analyse(tmp_path, sets, READS)
    assert other_fit["fit_error"].startswith("fit descriptives not computed: fit.raw.jsonl was measured on set")
    assert set(other_fit["floored_rates"]) == {"eval"} and other_fit["b1"] == good["b1"]
    assert other_fit["fit_error"] in render({"models": {"m": other_fit}})


def test_analyse_with_no_complete_held_out_question_reports_instead_of_raising(tmp_path):
    sets = _model_dir(tmp_path)
    for name in ba.B12_SETS:                                        # header only: the reads exited 0 with nothing
        raw = tmp_path / f"{name}.raw.jsonl"
        raw.write_text(raw.read_text().splitlines()[0] + "\n")
    out = ba.analyse(tmp_path, sets, READS)
    assert out["b1"]["pass"] is None and out["b2"]["error"] == "no held-out question was read under every rotation"


def test_refit_reproduces_the_profile_from_its_raw_file(tmp_path):
    data = tmp_path / "set.jsonl"
    data.write_text(set_text(ns=(3,), per=30))
    stub = CalloStub(PlantedBias(priors={3: PRIOR}))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--max-per-n", "26", "--out",
                     str(tmp_path / "p.json")]) == 0
    finally:
        stub.stop()
    live = json.loads((tmp_path / "p.json").read_text())
    again = refit(tmp_path / "p.raw.jsonl", data)                   # --max-per-n 26 comes from the raw header
    for p in (live, again):
        p.pop("created")
    assert again["entries"] == live["entries"] and again["eval"] == live["eval"] and again["set"] == live["set"]
    assert live["set"]["capped"] == 4 and live["entries"]["3"]["questions"] == 26
    with pytest.raises(SystemExit, match="sha256"):
        refit(tmp_path / "p.raw.jsonl", None)                       # the built-in set is a different set


SUMMARY = {"started": "t0", "finished": "t1", "llama_server": "/x/llama-server", "models": {
    "tiel_kvu": {"model": "/m/Tiel.gguf", "flags": [], "build_info": "b50-4ebdf2c", "kv": "unified",
                 "b0": {"rc": 0, "state": "clean", "pass": True},
                 "b4": {"a_rc": 2, "a_failed": ["foreign_tokens"], "b_rc": 2, "b_failed": ["foreign_processing"],
                        "c_rc": 0, "pass": True},
                 "b3": {"identical": True, "calls": 300, "applied": 40, "added_ms": 0.004, "pass": True},
                 "reads": {"eval": {"rc": 0, "readouts": 2400, "attempts": 2}},
                 "b1": {"per_n": {"4": {"tested": True, "questions": 96, "raw": 0.08, "debiased": 0.02,
                                        "signed": [0.08, 0.02], "signed_diff_ci": [-0.07, -0.05], "pass": True},
                                  "6": {"tested": False, "questions": 80, "detect_p_holm": 0.41,
                                        "note": "no detectable held-out bias (p_holm 0.41)", "harm": False,
                                        "harm_p_holm": 0.7},
                                  "7": {"tested": False, "questions": 30, "note": "not tested held-out"}},
                        "pass": True},
                 "b5": {"d_accuracy": 0.001, "d_nll": -0.004, "rows": 3976, "pass": True},
                 "b2": {"raw": 0.71, "debiased": 0.74, "diff_ci": [0.01, 0.05], "pass": True,
                        "flips": [{"q": "L12/q", "n": 5, "source": "csqa", "flip": "broke", "margin": [0.001, 0.036]},
                                  {"q": "L40/q", "n": 2, "source": "imdb", "flip": "mended", "margin": [0.2, 0.1]}]},
                 "floored_rates": {"fit": {"mmlu": {"4": {"questions": 72, "floored": 3, "rate": 0.04}}}},
                 "labelled": {"T1": {"raw": {"accuracy": 0.8, "nll": 0.6, "ece10": 0.05, "rows": 400},
                                     "debiased": {"accuracy": 0.81, "nll": 0.58, "ece10": 0.04, "rows": 400}},
                              "Tdev": {"raw": {"accuracy": 0.8, "nll": 0.5, "ece10": 0.03, "rows": 400, "T": 1.2}}},
                 "entries": ["4"], "dropped": {"8": "not significant (p_holm 0.41)"}},
    "tiel_nokvu": {"model": "/m/Tiel.gguf", "flags": ["--no-kvu"], "build_info": "b50-4ebdf2c", "kv": "per_slot",
                   "fit_rc": 2, "aborted": "the fit exited 2: held-out reads and B3 skipped",
                   "b1": {"pass": None, "error": "not judged on partial held-out data: heldout not run"},
                   "b2": {"pass": None, "error": "not judged on partial held-out data: heldout not run"},
                   "entries": [], "dropped": None},
    "qwen9b": {"error": "RuntimeError: boom"}},
    "g0_frames": {"pass": None, "note": "not run: no labelled image set"},
    "prereg": {"sha256": "c0ffee" * 10 + "abcd", "verdict": "ERROR", "b1_tested": ["tiel_kvu n=4"],
               "reasons": ["qwen9b: RuntimeError: boom"]},
    "data": {"fit": {"path": "/v/public-pool-v6/calibration.jsonl", "sha256": "bb43a590d115" + "0" * 52},
             "eval": {"path": "/v/public-pool-v6/test.jsonl", "sha256": "ea63dfe433d0" + "0" * 52}}}


def test_the_report_renders_gates_reasons_descriptives_and_errors():
    md = render(SUMMARY)
    for text in ("B0", "B1", "B2", "B3", "B4", "B5", "PASS", "not tested held-out (30 questions)", "not run",
                 "a detectable bias and a debiased max-|.| above the raw one", "The gates are `GATES` in",
                 "the pre-registration is in the development history",
                 "no detectable held-out bias (p_holm 0.41) (80 questions); harm leg clear (p_holm 0.7)",
                 "n=4: signed +0.0800 -> +0.0200, CI [-0.0700, -0.0500]; max-|.| 0.080 -> 0.020",
                 "accuracy +0.0010, NLL -0.0040 (3976 rows)", "1 questions broke, 1 mended",
                 "- `L12/q` (n=5, csqa): broke, margin 0.001 -> 0.036",
                 "**Pre-registered verdict: ERROR**", "B1 tested at: tiel_kvu n=4",
                 "the pre-registration's sha256 `c0ffeec0ffee`, as committed in C, before its `Pinned commit C:` line "
                 "was appended", "Reasons: qwen9b: RuntimeError: boom",
                 "eval set: `/v/public-pool-v6/test.jsonl` (sha256 `ea63dfe433d0`)",
                 "held-out raw, T refitted on public-pool-v6 calibration: accuracy 0.800",
                 "RuntimeError: boom", "b50-4ebdf2c", "not significant (p_holm 0.41)", "fit mmlu: n=4 3/72 (4%)",
                 "(a) 2 [foreign_tokens], (b) 2 [foreign_processing], (c) 0", "ERROR",
                 "not judged on partial held-out data: heldout not run", "the fit exited 2",
                 "eval exit 0, 2400 readouts (resumed once)"):
        assert text in md, text
    b4_setup = {**SUMMARY["models"]["tiel_kvu"], "b4": {"pass": None, "error": "B4 setup: slot 3 never held tokens"},
                "b3": {"pass": None, "note": "not run: the fit kept no n, so nothing can be applied"}}
    md = render({"models": {"tiel_kvu": b4_setup}, "aborted": "another model server appeared: pid 7 (ollama)"})
    assert "B4 setup: slot 3 never held tokens" in md and "the fit kept no n" in md
    assert md.splitlines()[2].startswith("**ABORTED:** another model server appeared")


def test_the_report_creates_its_output_directory(tmp_path):
    """A fresh clone has no docs/bench/; the report makes it rather than failing."""
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(SUMMARY))
    out = tmp_path / "docs" / "bench" / "v0.3-bias.md"
    assert report_main([str(summary), "--out", str(out)]) == 0
    assert out.read_text() == render(SUMMARY)


# ---- bench/bias_refit.py -----------------------------------------------------------------------------------

def _measured(tmp_path):
    data = tmp_path / "set.jsonl"
    data.write_text(set_text(ns=(3,), per=24))
    stub = CalloStub(PlantedBias(priors={3: PRIOR}))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out",
                     str(tmp_path / "p.json")]) == 0
    finally:
        stub.stop()
    return tmp_path / "p.raw.jsonl", data


@pytest.mark.parametrize("seed", [-1, "0", True, None])
def test_refit_refuses_a_header_seed_that_is_not_a_non_negative_int(tmp_path, seed):
    """A negative seed must be a clear error, not numpy's traceback at the fit."""
    raw, data = _measured(tmp_path)
    lines = raw.read_text().splitlines(keepends=True)
    head = json.loads(lines[0])
    head["seed"] = seed
    raw.write_text(json.dumps(head) + "\n" + "".join(lines[1:]))
    with pytest.raises(SystemExit, match=r"seed .*not a non-negative integer"):
        refit(raw, data)


@pytest.mark.parametrize(("flag", "value"), [("--min-questions", "1"), ("--min-questions", "0"),
                                             ("--min-questions", "-3"), ("--max-per-n", "0")])
def test_refit_refuses_degenerate_question_counts(tmp_path, flag, value):
    """Below 2 questions the permutation test divides by zero; mirror the CLI's bounds."""
    raw, data = _measured(tmp_path)
    with pytest.raises(SystemExit, match="--min-questions must be at least 2 and --max-per-n at least 1"):
        refit_main([str(raw), "--data", str(data), flag, value, "--out", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


def test_refit_names_a_malformed_raw_file_without_a_traceback(tmp_path):
    raw, data = _measured(tmp_path)
    with raw.open("a") as f:
        f.write('{"type": "readout", "q": "x", "n": 2, "s": 0, "probs": "ab", "miss": 0, "slot": 0}\n')
    with pytest.raises(SystemExit, match="is not a raw record"):
        refit_main([str(raw), "--data", str(data), "--out", str(tmp_path / "x.json")])
