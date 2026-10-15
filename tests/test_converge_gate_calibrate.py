"""Calibrate measures early exit's gate (experimental): with --converge-probe it reads the built-in gate question once
per calibration question at budget 0 (the think probe's `gate` mode, as serve reads it: no position-bias offsets), keeps
each read in the raw file and resumes it like any phase; a raw file made before the gate gets ONLY the gate reads on
--resume. Beside each early-exit point `m@ee<z>` the profile gains its gated point `m@ee<z>+g`, computed offline with
serve's rule (a yes takes the cap's reading, a no the early-exit one, every question adds its read ms), with its
cross-fitted Δacc against the cap, its flagged share and read ms; gated points are listed, marked opt-in and never
picked, in calibrate's summary, compare's table, scatter and report.json alike. Profiles without gate data have no
gated point, and without --converge-probe nothing changes."""
import dataclasses
import json

import pytest
from fake_upstream import _TEXT_ID, FakeUpstream
from test_calibrate_think import SAMPLING, STATS, _len, _Served, _settings, write_data
from test_converge_gate import GATE_TEXT, says
from test_deadline_converge import _saved, _with_probes_and_cost
from test_profile import profile_dict as bias_profile
from test_think_serve import BODY, _probe

import callo.calibrate_think as ct
from callo.calibrate_think import (
    build_profile,
    calibrate_one,
    load_data,
    opt_in_line,
    read_raw,
    summary_lines,
)
from callo.compare import (
    joint,
    load_profile,
    malformed,
    points_of,
    render_md,
    report_dict,
    scatter_lines,
    table_lines,
)
from callo.converge import GATE_SHA256, GATE_SKIP_P, gate_skips
from callo.profile import load_profile as load_bias_profile
from callo.think import EE_OPT_IN, Limit, Paired, Preference, pick


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=100).start()
    yield u
    u.stop()


def _gate_reads(up):
    return [r for r in up.requests if r.path == "/completion" and (r.json or {}).get("n_probs")
            and isinstance(r.json.get("prompt"), list) and r.json["prompt"] and r.json["prompt"][0] >= _TEXT_ID
            and GATE_TEXT in up._texts[r.json["prompt"][0] - _TEXT_ID]]


# ---- the think probe's gate mode ------------------------------------------------------------------------------------

@pytest.mark.parametrize("yes", [True, False])
def test_the_probe_reads_the_gate_on_the_request_s_state_at_budget_0(up, yes):
    says(up, yes)
    probe, _, stats = _probe(up)
    code, out, _ = probe.respond({"request": BODY, "budgets": [0], "seed": 3, "gate": True}, {})
    assert code == 200 and out["readouts"] == 1, out
    g = out["questions"]["q"]["gate"]
    assert (g["p"] >= GATE_SKIP_P) is yes and g["sha256"] == GATE_SHA256 and g["ms"] > 0 and "error" not in g
    (read,) = _gate_reads(up)
    assert "the state" in up._texts[read.json["prompt"][0] - _TEXT_ID]          # the item's own state
    assert not [r for r in up.requests if r.path == "/completion" and not (r.json or {}).get("n_probs")]   # no thinking
    assert "budgets" not in out["questions"]["q"] and "trace" not in out["questions"]["q"]


def test_the_probe_s_gate_read_carries_no_position_bias_offsets(up):
    """As serve's gate: a 2-option bias entry that would flip the answer is not applied."""
    says(up, False)
    probe, _, _ = _probe(up)
    probe.engine.position_bias = load_bias_profile(bias_profile({2: [0.0001, 0.9999]}))
    code, out, _ = probe.respond({"request": BODY, "budgets": [0], "gate": True}, {})
    assert code == 200 and out["questions"]["q"]["gate"]["p"] < GATE_SKIP_P


@pytest.mark.parametrize("body, why", [({"budgets": [0], "gate": 1}, "gate must be a boolean"),
                                       ({"budgets": [0, 64], "gate": True}, "budgets must be [0]"),
                                       ({"budgets": [64], "gate": True}, "budgets must be [0]"),
                                       ({"budgets": [0], "gate": True, "reserve_read": True}, "gate takes no"),
                                       ({"budgets": [0], "gate": True, "converge_probe": 32}, "gate takes no")])
def test_a_malformed_gate_probe_is_422(up, body, why):
    probe, _, _ = _probe(up)
    code, out, _ = probe.respond({"request": BODY, **body}, {})
    assert code == 422 and why in out["detail"]


def test_gate_false_is_the_probe_as_before(up):
    probe, _, _ = _probe(up)
    a = probe.respond({"request": BODY, "budgets": [0], "seed": 3}, {})[1]
    b = probe.respond({"request": BODY, "budgets": [0], "seed": 3, "gate": False}, {})[1]
    assert set(a["questions"]["q"]) == set(b["questions"]["q"]) and "gate" not in b["questions"]["q"]
    assert a["questions"]["q"]["budgets"]["0"]["probs"] == b["questions"]["q"]["budgets"]["0"]["probs"]


def test_the_rule_is_serve_s():
    assert gate_skips(None) and gate_skips(0.5) and gate_skips(0.99) and not gate_skips(0.4999)


# ---- the run: the gate phase, its raw rows, --resume and the pre-gate raw file ------------------------------------

DEAD = "http://127.0.0.1:1"


def _calibrate(tmp_path, s, *, resume=False, url=None, n=12, **over):
    data = write_data(tmp_path / "d.jsonl", n)
    settings = _settings(data, **{"converge_probe": 32, "converge_cost": 2, **over})
    out = calibrate_one(settings, url=url or s.url, name="fake", out_dir=tmp_path / "think", resume=resume,
                        log=s.log.append, therm=lambda: 100)
    return settings, out


def _phases(raw):
    return [r["phase"] for r in raw.rows]


def _budget_reads(up):
    """The budget readouts the fake saw that are not the gate's."""
    gate = {id(r) for r in _gate_reads(up)}
    return [r for r in up.requests if r.path == "/completion" and (r.json or {}).get("n_probs") and id(r) not in gate]


def test_a_converge_probe_run_reads_the_gate_once_per_question_between_phases_a_and_b(tmp_path):
    s = _Served(tmp_path, think_len=_len)
    try:
        says(s.up, True)
        _, out = _calibrate(tmp_path, s)
        reads = _gate_reads(s.up)
    finally:
        s.stop()
    assert out.status == "done", out.reason
    raw = read_raw(tmp_path / "think" / "fake.raw.jsonl")
    rows = [r for r in raw.rows if r["phase"] == "gate"]
    assert len(rows) == len(reads) == 12 and len({r["q"] for r in rows}) == 12
    assert all(r["p_yes"] >= GATE_SKIP_P and r["ms"] > 0 and r["sha256"] == GATE_SHA256 for r in rows)
    # the gate's prefix reuse, kept as the readout reports it (null on a path that reports none)
    assert all("cache_n" in r and (r["cache_n"] is None or isinstance(r["cache_n"], int)) for r in rows)
    ph = _phases(raw)
    assert max(i for i, p in enumerate(ph) if p == "A") < ph.index("gate") and \
        max(i for i, p in enumerate(ph) if p == "gate") < ph.index("B")
    gate = json.loads(out.profile.read_text())["converge"]["gate"]
    assert (gate["sha256"], gate["skip_p"], gate["n"], gate["flagged"], gate["errors"]) == (GATE_SHA256, 0.5, 12, 1.0, 0)
    assert gate["read_ms"]["p50"] > 0 and gate["read_ms"]["p95"] >= gate["read_ms"]["p50"]


def test_without_converge_probe_there_is_no_gate_read(tmp_path):
    s = _Served(tmp_path, think_len=_len)
    try:
        _, out = _calibrate(tmp_path, s, converge_probe=None, converge_cost=6)
        reads = _gate_reads(s.up)
    finally:
        s.stop()
    assert out.status == "done" and not reads
    assert "gate" not in _phases(read_raw(tmp_path / "think" / "fake.raw.jsonl"))
    prof = json.loads(out.profile.read_text())
    assert "converge" not in prof and "ee_gated" not in prof and not any("g" in r for r in prof["rows"])


def test_an_interrupted_gate_phase_resumes_with_only_its_missing_reads(tmp_path):
    s = _Served(tmp_path, think_len=_len)
    try:
        _, first = _calibrate(tmp_path, s)
        raw_path = tmp_path / "think" / "fake.raw.jsonl"
        lines = raw_path.read_text().splitlines()
        cut = next(i for i, line in enumerate(lines) if '"phase":"gate"' in line) + 3     # three gate reads in
        raw_path.write_text("\n".join(lines[:cut]) + "\n")
        before = len(_gate_reads(s.up))
        _, again = _calibrate(tmp_path, s, resume=True)
        after = len(_gate_reads(s.up))
    finally:
        s.stop()
    assert again.status == "done", again.reason
    assert after - before == 9
    raw = read_raw(raw_path)
    assert sorted(r["q"] for r in raw.rows if r["phase"] == "gate") == sorted(f"rec-{i}/q" for i in range(12))
    assert json.loads(again.profile.read_text())["converge"]["gate"]["n"] == 12


def _pre_gate(raw_path):
    """The raw file as the code before the gate wrote it: the same header, no gate row."""
    lines = raw_path.read_text().splitlines()
    raw_path.write_text("\n".join(x for x in lines if '"phase":"gate"' not in x) + "\n")


def test_a_resume_on_a_pre_gate_raw_file_adds_only_the_gate_reads(tmp_path):
    """The addendum: a --converge-probe raw file without gate reads (the code before the gate) gets the gate reads and
    nothing else: no thinking (the warm-up reads budget 0 alone), no budget read but the warm-up's, then the profile is
    rebuilt with its gate block; every other measured number is unchanged."""
    s = _Served(tmp_path, think_len=_len)
    try:
        _, first = _calibrate(tmp_path, s)
        raw_path = tmp_path / "think" / "fake.raw.jsonl"
        full = json.loads(first.profile.read_text())
        _pre_gate(raw_path)
        gens, reads, budget = len(s.gens()), len(_gate_reads(s.up)), len(_budget_reads(s.up))
        _, again = _calibrate(tmp_path, s, resume=True)
        added_gens, added_reads = len(s.gens()) - gens, len(_gate_reads(s.up)) - reads
        added_budget = len(_budget_reads(s.up)) - budget
    finally:
        s.stop()
    assert again.status == "done", again.reason
    assert added_gens == 0 and added_reads == 12 and added_budget == 1            # the one warm-up question, at 0
    raw = read_raw(raw_path)
    warm = [r for r in raw.rows if r["phase"] == "warmup"]
    assert [r["budgets"] for r in warm] == [[0, 64, 256], [0]]
    assert _phases(raw)[-12:] == ["gate"] * 12
    prof = json.loads(again.profile.read_text())
    assert prof["converge"]["gate"]["n"] == 12
    assert prof["rows"] == full["rows"] and prof["budgets"] == full["budgets"]


def test_a_pre_gate_raw_file_refits_with_no_server_and_no_gated_point(tmp_path):
    s = _Served(tmp_path, think_len=_len)
    try:
        _, first = _calibrate(tmp_path, s)
    finally:
        s.stop()
    full = json.loads(first.profile.read_text())
    _pre_gate(tmp_path / "think" / "fake.raw.jsonl")
    s.log.clear()
    _, refit = _calibrate(tmp_path, s, resume=True, url=DEAD)
    assert refit.status == "done", refit.reason
    assert any("no gate read" in line and "--resume with the server up" in line for line in s.log)
    prof = json.loads(refit.profile.read_text())
    assert "gate" not in prof["converge"] and "ee_gated" not in prof and prof["rows"] == full["rows"]


def test_a_complete_raw_file_with_its_gate_reads_refits_with_no_server(tmp_path):
    s = _Served(tmp_path, think_len=_len)
    try:
        _, first = _calibrate(tmp_path, s)
    finally:
        s.stop()
    _, refit = _calibrate(tmp_path, s, resume=True, url=DEAD)
    assert refit.status == "done", refit.reason
    assert json.loads(refit.profile.read_text())["converge"]["gate"] == \
        json.loads(first.profile.read_text())["converge"]["gate"]


def test_a_server_that_ignores_the_gate_key_fails_the_run_loudly(tmp_path, monkeypatch):
    real_read = ct.ProbeClient.read

    def read(self, item, budgets, seed, sampling, extras=None):
        if extras is None or extras.get("gate"):          # an older serve drops the key it does not know
            return real_read(self, item, budgets, seed, sampling)
        return real_read(self, item, budgets, seed, sampling, extras)

    monkeypatch.setattr(ct.ProbeClient, "read", read)
    s = _Served(tmp_path, think_len=_len)
    try:
        _, out = _calibrate(tmp_path, s, n=8)
    finally:
        s.stop()
    assert out.status == "failed"
    assert "unusable reply: the server does not support the early-exit gate's probe; upgrade callo serve" in out.reason


# ---- the gated point: the arithmetic, its Δacc, the gate block ---------------------------------------------------

def _gated_raw(items, flag=lambda i: i % 3 == 0, *, wrong=(), sha=GATE_SHA256, read=lambda i: 150.0 + i):
    """`_with_probes_and_cost`'s raw file plus one gate read per question: P(yes) .9 where `flag(i)`, else .1, read in
    `read(i)` ms. `wrong`: question indices whose probes are confidently wrong (early exit answers them wrong)."""
    raw, settings = _with_probes_and_cost(items)
    for r in raw.rows:
        i = next(j for j, it in enumerate(items) if it.uid == r["q"])
        if r["phase"] == "B" and i in wrong:
            n = len(r["budgets"]["64"]["probs"])
            for p in r["probes"]:
                p["probs"] = [0.99 if j == (items[i].label + 1) % n else 0.01 / (n - 1) for j in range(n)]
    for i, it in enumerate(items):
        raw.rows.append({"phase": "gate", "q": it.uid, "slot": 0, "p_yes": 0.9 if flag(i) else 0.1,
                         "ms": read(i), "sha256": sha})
    return raw, settings


def _profile(tmp_path, n=80, **kw):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", n))
    raw, settings = _gated_raw(items, **kw)
    return items, raw, settings, build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")


ROW = {"correct": 0, "lat_ms": 3, "k": 4}


@pytest.mark.parametrize("flag", [lambda i: i % 3 == 0, lambda i: True, lambda i: False])
def test_the_gated_point_takes_the_cap_reading_on_a_yes_the_early_exit_one_on_a_no_and_adds_the_read(tmp_path, flag):
    items, _, _, prof = _profile(tmp_path, flag=flag)
    assert set(prof["ee"]) == {"0.9"} and set(prof["ee_gated"]) == {"0.9"}
    g = prof["ee_gated"]["0.9"]
    rows = []
    for i, r in enumerate(prof["rows"]):
        b, e, ms = r["b"]["256"], r["e"]["0.9"], 150.0 + i
        want = [b[ROW["correct"]], b[ROW["lat_ms"]] + ms] if flag(i) else [e[0], e[1] + ms]
        assert r["g"]["0.9"] == pytest.approx(want), r["q"]
        rows.append(want)
    assert g["n"] == 80 and g["acc"] == pytest.approx(sum(c for c, _ in rows) / 80)
    assert g["lat_max_ms"] == pytest.approx(max(x for _, x in rows))
    assert all(len(g[k]) == 3 for k in ("lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms"))
    assert (g["cap"], g["K"], g["cell"]) == (256, 32, prof["ee"]["0.9"]["cell"])
    share = sum(1 for i in range(80) if flag(i)) / 80
    reads = sorted(150.0 + i for i in range(80))
    assert g["gate"] == {"flagged": share, "n": 80, "read_ms_p50": ct.percentile(reads, 0.5),
                         "read_ms_p95": ct.percentile(reads, 0.95)}
    if all(flag(i) for i in range(80)):            # every question thinks to the cap: the cap's think and forced share
        assert g["think_mean"] == pytest.approx(prof["budgets"]["256"]["think_mean"])
        assert g["forced"] == pytest.approx(prof["budgets"]["256"]["forced"])
    if not any(flag(i) for i in range(80)):        # none does: early exit's own think tokens
        assert g["think_mean"] == pytest.approx(prof["ee"]["0.9"]["think_mean"])
    cv = prof["converge"]["gate"]
    assert (cv["sha256"], cv["skip_p"], cv["n"], cv["flagged"], cv["errors"]) == (GATE_SHA256, 0.5, 80, share, 0)


def test_the_gated_dacc_is_cross_fitted_against_the_cap_as_the_early_exit_one_is(tmp_path):
    """Early exit answers questions 0, 3 and 6 confidently wrong; the gate flags exactly those: the gated point is as
    accurate as the cap, so its Δacc is 0 while early exit's is negative."""
    from callo.think import paired_bootstrap
    items, _, settings, prof = _profile(tmp_path, flag=lambda i: i in (0, 3, 6), wrong=(0, 3, 6))
    rule = prof["converge"]["rules"]["0.9"]
    assert rule is not None and rule["dacc"][0] < 0
    g = prof["ee_gated"]["0.9"]
    final = [float(r["b"]["256"][0]) for r in prof["rows"]]
    gated = [float(r["g"]["0.9"][0]) for r in prof["rows"]]
    want = paired_bootstrap(final, gated, [r["cluster"] for r in prof["rows"]], seed=settings.seed)
    assert g["dacc_cap"] == pytest.approx(list(want)) and g["dacc_cap"][0] == 0.0
    assert "dacc" not in g                         # the summary's Δacc column is against budget 0


def test_a_failed_gate_read_flags_its_question_as_serve_skips_it(tmp_path):
    items, raw, settings, _ = _profile(tmp_path, flag=lambda i: False)
    for r in raw.rows:
        if r["phase"] == "gate" and r["q"] == items[1].uid:
            r["p_yes"], r["error"] = None, "the gate read has no probability for yes"
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    r = prof["rows"][1]
    assert r["g"]["0.9"] == pytest.approx([r["b"]["256"][0], r["b"]["256"][3] + 151.0])
    assert prof["converge"]["gate"]["errors"] == 1 and prof["converge"]["gate"]["flagged"] == 1 / 80


def test_missing_or_mixed_gate_reads_make_no_gated_point_and_say_why(tmp_path):
    items, raw, settings, _ = _profile(tmp_path)
    raw.rows = [r for r in raw.rows if not (r["phase"] == "gate" and r["q"] == items[5].uid)]
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    assert "ee_gated" not in prof and not any("g" in r for r in prof["rows"])
    assert "1 of 80 questions have no gate read" in prof["converge"]["gate"]["why"]
    items, raw, settings, _ = _profile(tmp_path)
    next(r for r in raw.rows if r["phase"] == "gate")["sha256"] = "0" * 64
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    assert "ee_gated" not in prof and prof["converge"]["gate"]["sha256"] is None
    assert "two gates" in prof["converge"]["gate"]["why"]


def test_a_profile_without_gate_reads_has_no_gated_point_and_no_gate_key(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    assert prof["ee"] and "ee_gated" not in prof and "gate" not in prof["converge"]
    assert not any("g" in r for r in prof["rows"]) and not any("+g" in x for x in summary_lines(prof))


# ---- listed, never picked: the pick, the marks, the opt-in line ----------------------------------------------------

def _best_gated(prof):
    """The gated point made the best on every axis: right on every question, faster than anything."""
    for r in prof["rows"]:
        r["g"]["0.9"] = [True, 1.0]
    g = prof["ee_gated"]["0.9"]
    g.update({"acc": 1.0, "lat_mean_ms": [1.0, 1.0, 1.0], "lat_p50_ms": [1.0, 1.0, 1.0], "lat_p95_ms": [1.0, 1.0, 1.0],
              "lat_p99_ms": [1.0, 1.0, 1.0], "lat_max_ms": 1.0})
    return prof


PREFS = [Preference(), Preference(latency="p50"), Preference(latency="p95"), Preference(price=50.0),
         Preference(min_accuracy=0.5), Preference(limits=(Limit("p99", 1e9),)), Preference(max_gib=100.0)]


@pytest.mark.parametrize("pref", PREFS)
def test_a_gated_point_is_never_picked_under_any_preference(tmp_path, pref):
    *_, prof = _profile(tmp_path)
    assert prof["pick"]["excluded"]["m@ee.9+g"] == EE_OPT_IN and "+g" not in str(prof["pick"]["label"])
    prof = _best_gated(prof)
    j = joint([load_profile(_saved(tmp_path, prof))], pref=pref)
    assert "m@ee.9+g" in {p.label for p in j.ranked}
    assert j.chosen.point is not None and j.chosen.point.kind == "budget"
    assert j.chosen.excluded["m@ee.9+g"] == EE_OPT_IN and j.chosen.best.kind == "budget"
    pts = points_of(prof, "m")
    one = pick(pts, paired=Paired(lambda p: ([1.0], [1.0]), ("c",), 0), pref=pref)
    assert one.point is not None and one.point.kind == "budget" and one.excluded["m@ee.9+g"] == EE_OPT_IN


def test_a_gated_point_is_marked_opt_in_in_the_summary_compare_s_table_and_the_scatter(tmp_path):
    *_, prof = _profile(tmp_path)
    summary = summary_lines(prof)
    row = next(line for line in summary if line.startswith("m@ee.9+g "))
    assert row.endswith("opt-in (early exit)")
    j = joint([load_profile(_saved(tmp_path, prof))])
    rows = {r.split(" | ")[0].removeprefix("| "): r for r in table_lines(j) if r.startswith("| m@")}
    assert rows["m@ee.9+g"].endswith("; opt-in (early exit) |")
    legend = " ".join(scatter_lines(list(j.ranked)))
    assert "m@ee.9+g (opt-in)" in legend
    prof = _best_gated(prof)                       # a row a gated point dominates says the dominator is opt-in
    j = joint([load_profile(_saved(tmp_path, prof, "b.think.json"))])
    marks = [r for r in table_lines(j) if "· by m@ee.9+g (opt-in)" in r]
    assert marks


GATED_IN = ("; gated (m@ee.9+g) saves ~{ms} ms (mean); it holds only on questions like these: serve with "
            "--think-converge 0.9 (add --think-converge-gate for the gated one) to opt in")


def test_the_opt_in_line_names_both_variants_one_or_none(tmp_path):
    items, raw, settings, prof = _profile(tmp_path, flag=lambda i: i % 9 == 0, read=lambda i: 2.0)
    pts = {p.label: p for p in points_of(prof, "m")}
    ms = pts["m@256"].latency("mean")[0] - pts["m@ee.9+g"].latency("mean")[0]
    line = opt_in_line(prof, "m")
    assert ms > 0 and line.startswith("early exit (opt-in, not picked): m@ee.9 saves ")
    assert line.endswith(GATED_IN.format(ms=f"{ms:.0f}")) and line.count("\n") == 0
    assert f"  {line}" in summary_lines(prof)
    j = joint([load_profile(_saved(tmp_path, prof))])
    assert render_md(j).splitlines().count(line) == 1 and report_dict(j)["early_exit"]["m"]["opt_in"] == line
    slow = json.loads(json.dumps(prof))            # a gated point no faster than the cap: named, no saving claimed
    slow["ee_gated"]["0.9"]["lat_mean_ms"] = [1e7, 1e7, 1e7]
    assert "; gated (m@ee.9+g); it holds only" in opt_in_line(slow, "m")
    one = {k: v for k, v in prof.items() if k != "ee_gated"}          # no gate data: the one variant, as before
    assert "gated" not in opt_in_line(one, "m") and opt_in_line(one, "m").endswith(
        "serve with --think-converge 0.9 to opt in")
    none = {**prof, "converge": {**prof["converge"], "rules": {**prof["converge"]["rules"], "0.9": None}}}
    assert opt_in_line(none, "m") is None          # serve cannot run .9: neither variant
    j = joint([load_profile(_saved(tmp_path, none, "n.think.json"))])
    assert j.chosen.excluded["m@ee.9+g"].startswith("callo serve cannot run it: no early-exit rule")


def test_the_summary_s_early_exit_block_says_what_the_gate_measured(tmp_path):
    *_, prof = _profile(tmp_path)
    lines = ct.converge_lines(prof)
    gate = next(x for x in lines if x.startswith("  gate "))
    assert f"sha256 {GATE_SHA256[:12]}" in gate and "flags 34 % of 80 questions" in gate and "read p50 " in gate
    assert any(x.startswith("  z .9+g: acc ") and "Δ vs full think" in x for x in lines)
    gone = {**prof, "converge": {**prof["converge"], "gate": {**prof["converge"]["gate"], "why": "1 of 80 questions "
                                                                "have no gate read: no gated point"}}}
    gone.pop("ee_gated")
    assert any("no gated point" in x for x in ct.converge_lines(gone))


# ---- compare: the table, report.json, old profiles ------------------------------------------------------------------

def test_compare_reads_gated_points_back_and_reports_them(tmp_path):
    *_, prof = _profile(tmp_path)
    path = _saved(tmp_path, prof)
    loaded = load_profile(path)
    pts = {p.label: p for p in points_of(loaded, "m")}
    g = pts["m@ee.9+g"]
    assert (g.kind, g.value, g.cap, g.n) == ("eeg", 0.9, 256, 80) and g.forced == prof["ee_gated"]["0.9"]["forced"]
    assert g.acc == prof["ee_gated"]["0.9"]["acc"] and g.think_mean == prof["ee_gated"]["0.9"]["think_mean"]
    j = joint([loaded])
    assert len(j.fixed) == 80
    rep = report_dict(j)
    (x,) = [x for x in rep["points"] if x["kind"] == "eeg"]
    assert (x["label"], x["value"], x["cap"]) == ("m@ee.9+g", 0.9, 256)
    assert rep["early_exit"]["m"]["gate"] == prof["converge"]["gate"]
    assert rep["early_exit"]["m"]["gated"]["0.9"]["dacc_cap"] == prof["ee_gated"]["0.9"]["dacc_cap"]
    assert "| m@ee.9+g |" in render_md(j)
    for p in [*rep["pairs"]]:                                       # pairs over one row set include the gated point
        assert p["n"] == 80


def test_compare_re_picks_an_old_profile_without_gated_points(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 80))
    raw, settings = _with_probes_and_cost(items)
    old = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="m")
    j = joint([load_profile(_saved(tmp_path, old))])
    assert not any(p.kind == "eeg" for p in j.ranked) and j.chosen.point.kind == "budget"
    rep = report_dict(j)
    assert "gate" not in rep["early_exit"]["m"] and "gated" not in rep["early_exit"]["m"]
    (tmp_path / "n").mkdir()
    *_, new = _profile(tmp_path / "n")
    assert joint([load_profile(_saved(tmp_path, new, "n.think.json"))]).chosen.point.label == j.chosen.point.label


def test_compare_flags_gated_points_measured_with_two_gates(tmp_path):
    *_, a = _profile(tmp_path)
    b = json.loads(json.dumps(a))
    b["model_name"] = "n"
    b["converge"]["gate"]["sha256"] = "0" * 64
    from callo.compare import flags
    assert any(f.startswith("gated early exit measured with different gates") for f in flags([a, b]))
    assert not any("different gates" in f for f in flags([a, json.loads(json.dumps(a))]))


@pytest.mark.parametrize("edit, why", [
    (lambda p: p["ee_gated"].update({"x": {}}), "ee_gated key 'x' is not a z in (0, 1)"),
    (lambda p: p["ee_gated"]["0.9"].update({"acc": "x"}), "ee_gated.0.9.acc is not a finite number"),
    (lambda p: p["ee_gated"]["0.9"].update({"cap": 0}), "ee_gated.0.9.cap is not an integer >= 1"),
    (lambda p: p["rows"][0]["g"].update({"0.9": [True]}), "gated early-exit rule 0.9 is not [correct, lat_ms]"),
    (lambda p: p["rows"][0].update({"g": "x"}), "g is not a map of gated early-exit rules")])
def test_a_malformed_gated_block_is_refused_by_name(tmp_path, edit, why):
    *_, prof = _profile(tmp_path)
    edit(prof)
    assert why in (malformed(prof) or "")


def test_an_old_gated_entry_without_its_stored_statistics_recomputes_them_from_its_rows(tmp_path):
    *_, prof = _profile(tmp_path)
    want = {p.label: p for p in points_of(prof, "m")}["m@ee.9+g"]
    for k in ("lat_mean_ms", "lat_p99_ms", "lat_max_ms"):
        del prof["ee_gated"]["0.9"][k]
    assert malformed(prof) is None
    got = {p.label: p for p in points_of(prof, "m")}["m@ee.9+g"]
    assert (got.lat_mean, got.lat_p99, got.lat_max) == (want.lat_mean, want.lat_p99, want.lat_max)


def test_a_resume_refit_keeps_the_gated_points_and_re_picks_without_them(tmp_path):
    """`--resume` refits through build_profile: the gated points come back, and the pick is still a budget point."""
    items, raw, settings, prof = _profile(tmp_path)
    again = build_profile(items, raw, dataclasses.replace(settings, latency="p50"), sampling=SAMPLING, stats=STATS,
                          name="m")
    assert again["ee_gated"] == prof["ee_gated"] and again["pick"]["kind"] == "budget"
    assert again["pick"]["excluded"]["m@ee.9+g"] == EE_OPT_IN


def test_the_dry_run_counts_the_gate_reads(tmp_path):
    from callo.calibrate_think import dry_run_lines
    data = write_data(tmp_path / "d.jsonl", 8)
    items, skipped = load_data(data)
    assert "early-exit gate: 8 budget-0 reads, one per question with two or more options" in \
        dry_run_lines(_settings(data, converge_probe=32), items, skipped)
    assert not any("gate" in x for x in dry_run_lines(_settings(data), items, skipped))


def test_a_gate_row_without_cache_n_still_loads_and_fits(tmp_path):
    """Rows written before `cache_n` was kept (and `_gated_raw`'s have none) read, check and fit as before."""
    items, raw, settings, _ = _profile(tmp_path)
    path = tmp_path / "x.raw.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [raw.header, *raw.rows]) + "\n")
    back = read_raw(path)
    ct.check_rows(back, path)
    assert all("cache_n" not in r for r in back.rows if r["phase"] == "gate")
    assert build_profile(items, back, settings, sampling=SAMPLING, stats=STATS, name="m")["ee_gated"]


# ---- the menu's --resume: a done model whose raw file has gate reads left -------------------------------------------

def _menu(tmp_path, s, *, resume=False, launcher=None, scan_fn=None, **over):
    """One menu run of the model `fake` against the served fake (the launcher only hands calibrate its URL), on 80
    questions: enough for an early-exit point, so a gate read makes a gated one."""
    from helpers import tiny_gguf
    from test_menu import HOST, _stub_launcher

    from callo.menu import Driver, Entry, MenuOptions
    data = tmp_path / "d.jsonl"
    if not data.exists():
        write_data(data, 80)
    settings = _settings(data, **{"converge_probe": 32, "converge_cost": 2, **over})
    entry = Entry("fake", str(tiny_gguf(tmp_path / "fake.gguf")))
    opts = MenuOptions(tmp_path / "think", resume=resume, headroom_gib=0.0, gpu_mem_gib=64.0, settle_s=0.0)
    log: list[str] = []
    rc = Driver(settings, [entry], opts, log=log.append, launcher=launcher or _stub_launcher(s.url),
                scan_fn=scan_fn or (lambda **k: []), host=HOST, therm=lambda: 100).run()
    return rc, log


def _done_pre_gate(tmp_path, s):
    """A menu run done before the gate: run.json says done, the raw file has no gate row and the profile no gated
    point (refitted from that raw file with no server)."""
    assert _menu(tmp_path, s)[0] == 0
    _pre_gate(tmp_path / "think" / "fake.raw.jsonl")
    assert _calibrate(tmp_path, s, resume=True, url=DEAD, n=80)[1].status == "done"


def test_a_menu_resume_reenters_a_done_pre_gate_model_for_its_gate_reads_alone(tmp_path):
    """A regression: run.json said done, so the menu skipped the model and the raw file kept 0 gate rows. The
    pending reads decide, as in a one-server --resume: the gate reads alone, then the profile, run.json and the report
    are rebuilt."""
    s = _Served(tmp_path, think_len=_len)
    try:
        _done_pre_gate(tmp_path, s)
        raw_path, prof_path = tmp_path / "think" / "fake.raw.jsonl", tmp_path / "think" / "fake.think.json"
        full = json.loads(prof_path.read_text())
        (tmp_path / "think" / "report.json").unlink()
        gens, reads = len(s.gens()), len(_gate_reads(s.up))
        rc, log = _menu(tmp_path, s, resume=True)
        added_gens, added_reads = len(s.gens()) - gens, len(_gate_reads(s.up)) - reads
    finally:
        s.stop()
    assert rc == 0, "\n".join(log)
    assert not any("done already" in line for line in log)
    assert any(line.startswith("fake: done, but ") and "gate" in line for line in log), "\n".join(log)
    assert added_gens == 0 and added_reads == 80
    raw = read_raw(raw_path)
    assert _phases(raw)[-80:] == ["gate"] * 80
    assert [r["budgets"] for r in raw.rows if r["phase"] == "warmup"][-1] == [0]      # the gate-only warm-up
    prof = json.loads(prof_path.read_text())
    assert prof["converge"]["gate"]["n"] == 80 and "ee_gated" not in full
    assert prof["ee_gated"] and set(prof["ee_gated"]) == set(prof["ee"]) == set(full["ee"])     # the gated points
    assert prof["budgets"] == full["budgets"] and prof["ee"] == full["ee"]
    assert prof["environment"]["server"]["launched_by"] == "menu"              # stamped again
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["fake"]["status"] == "done" and run["models"]["fake"]["profile"] == str(prof_path)
    report = json.loads((tmp_path / "think" / "report.json").read_text())
    assert any(p["label"].endswith("+g") for p in report["points"])


@pytest.mark.parametrize("over", [{}, {"converge_probe": None, "converge_cost": 6}],
                         ids=["with-gate-data", "without-converge-probe"])
def test_a_menu_resume_still_skips_a_done_model_with_nothing_left_to_read(tmp_path, over):
    s = _Served(tmp_path, think_len=_len)
    try:
        assert _menu(tmp_path, s, **over)[0] == 0
        del s.up.requests[:]
        rc, log = _menu(tmp_path, s, resume=True, **over)
        requests = list(s.up.requests)
    finally:
        s.stop()
    assert rc == 0 and not requests
    assert f"fake: done already ({tmp_path / 'think' / 'fake.think.json'}); skipped" in log


def _launch_fails(exc):
    def launcher(entry, **kw):
        raise exc
    return launcher


@pytest.mark.parametrize("how", ["launch", "precheck", "interrupt"])
def test_a_reentry_that_does_not_finish_keeps_the_done_model_its_files_and_its_report_row(tmp_path, monkeypatch, how):
    """A done model re-entered for its gate reads whose re-entry fails (its server does not start, the
    precheck skips it, a Ctrl-C: the run still stops) stays done with the record and the profile it had, in the
    report, with the reason; the raw file and the profile are byte-identical, and the next --resume tries again."""
    import callo.menu as menu
    s = _Served(tmp_path, think_len=_len)
    try:
        _done_pre_gate(tmp_path, s)
        out = tmp_path / "think"
        before = {n: (out / n).read_bytes() for n in ("fake.think.json", "fake.raw.jsonl")}
        record = json.loads((out / "run.json").read_text())["models"]["fake"]
        for n in ("report.json", "report.md"):
            (out / n).unlink()
        launcher = None
        if how == "launch":
            launcher = _launch_fails(menu.MenuFailure("llama-server did not start (planted)"))
        elif how == "interrupt":
            launcher = _launch_fails(KeyboardInterrupt())
        else:
            monkeypatch.setattr(menu, "precheck", lambda *a, **k: "does not fit (planted)")
        rc, log = _menu(tmp_path, s, resume=True, launcher=launcher)
        after = {n: (out / n).read_bytes() for n in before}
        now = json.loads((out / "run.json").read_text())["models"]["fake"]
        report_json, report_md = json.loads((out / "report.json").read_text()), (out / "report.md").read_text()
        monkeypatch.undo()
        del s.up.requests[:]
        _, log2 = _menu(tmp_path, s, resume=True)                  # the next --resume: the gate reads still left
        retried = len(_gate_reads(s.up))
    finally:
        s.stop()
    assert rc == (130 if how == "interrupt" else 0), "\n".join(log)
    assert after == before
    why = {"launch": "did not start (planted)", "precheck": "skipped: does not fit (planted)",
           "interrupt": "interrupted"}[how]
    assert now["status"] == "done" and now["reason"].startswith("gate reads failed: ") and why in now["reason"]
    assert now["reason"].endswith("; profile without gated points kept")
    assert {k: v for k, v in now.items() if k != "reason"} == {k: v for k, v in record.items() if k != "reason"}
    assert f"fake: done ({now['reason']})" in log                              # the summary names why
    assert [p["name"] for p in report_json["profiles"]] == ["fake"] and "| fake@0 |" in report_md
    assert retried == 80 and any(line.startswith("fake: done, but ") for line in log2), "\n".join(log2)


def test_a_raw_file_reads_left_cannot_read_is_taken_as_nothing_left(tmp_path):
    """Any exception deciding the reads left (a hand-damaged raw file: a trace without n_gen) keeps a
    done model done, with a warning naming it."""
    s = _Served(tmp_path, think_len=_len)
    try:
        assert _menu(tmp_path, s)[0] == 0
        raw_path = tmp_path / "think" / "fake.raw.jsonl"
        lines = []
        for line in raw_path.read_text().splitlines():
            d = json.loads(line)
            if d.get("phase") == "B" and isinstance(d.get("trace"), dict):
                del d["trace"]["n_gen"]
            lines.append(json.dumps(d))
        raw_path.write_text("\n".join(lines) + "\n")
        del s.up.requests[:]
        rc, log = _menu(tmp_path, s, resume=True)
        requests = list(s.up.requests)
    finally:
        s.stop()
    assert rc == 0 and not requests
    assert any(line.startswith(f"WARNING: fake: cannot tell whether {raw_path} has reads left, or read ")
               for line in log), "\n".join(log)
    assert f"fake: done already ({tmp_path / 'think' / 'fake.think.json'}); skipped" in log


def test_another_server_appearing_after_the_gate_reads_keeps_the_new_profile_stamped_gpu_busy(tmp_path):
    """The post-scan finds another model server after the gate reads rewrote the profile: that profile
    is kept with its gated points, stamped as the menu stamps any (a runnable serve line) and marked gpu_busy, and the
    reason says exactly what happened."""
    s = _Served(tmp_path, think_len=_len)
    calls = iter([[], ["pid 9: llama-server x"]])                  # the first GPU check, then the post-scan
    try:
        _done_pre_gate(tmp_path, s)
        rc, log = _menu(tmp_path, s, resume=True, scan_fn=lambda **k: next(calls))
    finally:
        s.stop()
    out = tmp_path / "think"
    assert rc == 0, "\n".join(log)
    rec = json.loads((out / "run.json").read_text())["models"]["fake"]
    assert rec["status"] == "done" and rec["reason"] == ("gate reads done, but another model server appeared (pid 9: "
                                                         "llama-server x); profile with gated points kept, marked gpu_busy")
    prof = json.loads((out / "fake.think.json").read_text())
    env = prof["environment"]
    assert prof["ee_gated"] and env["server"]["gpu_busy"] is True and env["server"]["launched_by"] == "menu"
    assert env["model"]["path"] == str(tmp_path / "fake.gguf") and env["server"]["launch"]["args"] == []
    assert [p["name"] for p in json.loads((out / "report.json").read_text())["profiles"]] == ["fake"]
    # the launch shape the stamp records, repeated in the serve line (the fake is an attach-mode serve, so no --model)
    assert "serve with: `callo serve --upstream URL --parallel 4 --s1-slots 1 --think-profile " in \
        (out / "report.md").read_text()


def test_a_profile_the_gate_reads_rewrote_that_cannot_be_stamped_is_put_back(tmp_path, monkeypatch):
    """The stamp fails after the gate reads rewrote the profile: the profile the model had is put back
    byte for byte (never an unstamped one left in its place), the model stays done and in the report, and the reason
    says so."""
    import callo.menu as menu
    s = _Served(tmp_path, think_len=_len)
    try:
        _done_pre_gate(tmp_path, s)
        out = tmp_path / "think"
        before = (out / "fake.think.json").read_bytes()
        record = json.loads((out / "run.json").read_text())["models"]["fake"]

        def stamp(*a, **k):
            raise OSError("planted stamp failure")
        monkeypatch.setattr(menu, "stamp_profile", stamp)
        rc, log = _menu(tmp_path, s, resume=True)
    finally:
        s.stop()
    assert rc == 0, "\n".join(log)
    assert (out / "fake.think.json").read_bytes() == before and not list(out.glob("*.tmp"))
    assert len([r for r in read_raw(out / "fake.raw.jsonl").rows if r["phase"] == "gate"]) == 80   # the reads stay
    now = json.loads((out / "run.json").read_text())["models"]["fake"]
    assert now["status"] == "done" and {k: v for k, v in now.items() if k != "reason"} == \
        {k: v for k, v in record.items() if k != "reason"}
    assert now["reason"].startswith("gate reads done, but OSError: planted stamp failure (traceback in ")
    assert now["reason"].endswith("; the profile they wrote was not stamped, so the one before them is kept (no gated "
                                  "points)")
    assert [p["name"] for p in json.loads((out / "report.json").read_text())["profiles"]] == ["fake"]
