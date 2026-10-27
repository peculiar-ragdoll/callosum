"""The Decision Index plate (brand/di.py) and the README's chart, their info contract: every number round-trips to
the snapshot (and the snapshot to the bench outputs); the treatment label matches the switch; both treatments are in
FIGURES.md; the whiskers are vs Clef-Flash BF16 and the Clef-27B Δ is in the table only; no Tiel point at 512/1024 (no
MiniCPM point at 256/512); the plot step refuses an arm with unexplained errors. Plus the preliminary marking, the
release bake's early exit (the line's early-exit point, the line's other arms as table rows, the paired comparisons,
the gate's share), the re-read records, the plain words (no "wall", no "the kit's"), the chart's contents, and the
legibility invariants (every point label reads as its own point's); the real release bake at the end."""
from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import di_fixtures as fx
import pytest

import di

# ---- helpers -----------------------------------------------------------------------------------------------------


def build(tmp: Path, **over) -> dict:
    s = fx.setup(tmp, **over)
    return di.build(di.load_spec(s["spec_path"]))


def _svg(snap: dict, theme: str = "dark") -> ET.Element:
    return ET.fromstring(di.draw(snap, theme=theme))


def _chart(snap: dict, theme: str = "dark") -> ET.Element:
    return ET.fromstring(di.draw_chart(snap, theme=theme))


def _drawn(snap: dict) -> list[dict]:
    """The points the plot draws: the plotted series' line points and single markers."""
    return [p for p in snap["points"] if p["on_plate"] and not p["variant_of"]]


def _role(root: ET.Element, role: str) -> list[ET.Element]:
    return [e for e in root.iter() if e.get("data-role") == role]


def resolve(snap: dict, ptr: str):
    """A drawn number's address: /points/<arm>/<field>/..., /reference/..., /areas/<i>/..."""
    parts = ptr.strip("/").split("/")
    if parts[0] == "points":
        node = next(p for p in snap["points"] if p["arm"] == parts[1])
        parts = parts[2:]
    else:
        node = snap
    for k in parts:
        node = node[int(k)] if isinstance(node, list) else node[k]
    return node


def _fp(snap: dict) -> str:
    return " ".join((f["lead"] or "") + " " + f["text"] for f in snap["fine_print"])


@pytest.fixture
def snap(tmp_path) -> dict:
    return build(tmp_path)


# ---- the data layer: the snapshot is the bench outputs' own numbers ------------------------------------------------

def test_the_snapshot_carries_the_bootstraps_numbers_unchanged(tmp_path):
    s = fx.setup(tmp_path)
    snap = di.build(di.load_spec(s["spec_path"]))
    boot = json.loads((tmp_path / "boot-a.json").read_text())
    assert {p["arm"] for p in snap["points"]} == set(fx.ARMS)
    for p in snap["points"]:
        a = boot["arms"][p["arm"]]
        assert p["index"] == a["index"] == a["kit_index"]
        assert p["index_ci"] == a["index_ci"]
        assert p["areas"] == a["areas"] and p["areas_ci"] == a["areas_ci"]
        assert p["lat"] == {"mean": a["latency_ms"]["mean"], "p95": a["latency_ms"]["p95"],
                            "mean_ci": a["latency_ms"]["mean_ci"], "n": a["latency_ms"]["n_rows"]}
        if p["arm"] == fx.REF:
            assert p["dref"] is None and p["area_dref"] is None
        else:
            assert p["dref"] == [a["delta_vs_reference"]["point"], *a["delta_vs_reference"]["ci"]]
            assert p["area_dref"] == {k: [v["point"], *v["ci"]] for k, v in a["area_delta_vs_reference"].items()}
        if p["arm"] == fx.SECOND:
            assert p["dsecond"] is None
        else:
            assert p["dsecond"] == [a["delta_vs_second_reference"]["point"], *a["delta_vs_second_reference"]["ci"]]
    assert snap["reference"]["arm"] == fx.REF and snap["second_reference"]["arm"] == fx.SECOND
    assert snap["unanswered"]["latency_rows"] == fx.LAT_ROWS == 100 - 3
    assert snap["unanswered"]["over_cap"]["tokens"] == [17112, 32805]
    assert [a["weight"] for a in snap["areas"]] == [25.8, 25.8, 20.0, 18.3, 10.0]


def test_every_drawn_number_round_trips_to_the_snapshot(snap, tmp_path):
    # through the committed form: the JSON file, as `draw` reads it
    (tmp_path / "di.json").write_text(json.dumps(snap, ensure_ascii=False))
    snap = json.loads((tmp_path / "di.json").read_text())
    for theme, root in [(th, f(snap, th)) for th in ("dark", "light") for f in (_svg, _chart)]:
        keyed = [e for e in root.iter() if e.get("data-key")]
        assert len(keyed) >= 2 * len(_drawn(snap)), theme          # each drawn point's tail and its label at least
        for e in keyed:
            want = resolve(snap, e.get("data-key"))
            if e.get("data-value") is not None:
                assert float(e.get("data-value")) == want, e.get("data-key")
            if e.get("data-lo") is not None:
                lo, hi = (want[1], want[2]) if len(want) == 3 else (want[0], want[1])
                assert (float(e.get("data-lo")), float(e.get("data-hi"))) == (lo, hi), e.get("data-key")
        # every drawn point is keyed on the plot (and, on the plate, in the panels); every arm on the plate in the table
        for p in _drawn(snap):
            g = next(g for g in _role(root, "point") if g.get("data-arm") == p["arm"])
            assert float(g.get("data-index")) == p["index"]
            assert float(g.get("data-lat-mean")) == p["lat"]["mean"] and float(g.get("data-lat-p95")) == p["lat"]["p95"]
        if _role(root, "table"):
            assert len(keyed) > 60
            cells = {e.get("data-key") for e in _role(root, "table-cell") if e.get("data-key")}
            for p in (p for p in snap["points"] if p["on_plate"]):
                for f in ("index", "lat/mean", "lat/p95"):
                    assert f"/points/{p['arm']}/{f}" in cells
            panel = {(e.get("data-area"), e.get("data-arm")) for e in _role(root, "panel-point")}
            assert panel == {(a["id"], p["arm"]) for a in snap["areas"] for p in _drawn(snap)}


def test_the_table_prints_each_number_with_the_shared_formatter(snap):
    root = _svg(snap)
    for e in _role(root, "table-cell"):
        k = e.get("data-key")
        if not k:
            continue
        v = resolve(snap, k)
        v = v[0] if isinstance(v, list) else v
        if k.endswith("/index"):
            assert e.get("data-text") == di.f1(v)
        elif k.endswith(("/dref/0", "/dsecond/0")):
            assert e.get("data-text") == di.sf1(v)
        else:
            assert e.get("data-text") == di.secs(v)
    assert di.f1(16.45) == "16.5" and di.sf1(-16.45) == "−16.5" and di.sf1(3.25) == "+3.3" and di.f1(54.75) == "54.8"


def test_pixels_are_monotone_in_log_latency_and_in_index(snap):
    root = _svg(snap)
    pts = [(float(g.get("data-lat-mean")), float(g.get("data-index")), float(g.get("data-x")), float(g.get("data-y")))
           for g in _role(root, "point")]
    for a in pts:
        for b in pts:
            if a[0] < b[0]:
                assert a[2] < b[2]
            if a[1] < b[1]:
                assert a[3] > b[3]


# ---- the info-contract tests -----------------------------------------------------------------------------------

def test_the_treatment_label_matches_the_switch(tmp_path):
    a = build(tmp_path / "a")
    root = _svg(a)
    assert root.get("data-treatment") == "kit" and root.get("data-treatment-label") == "(a) kit-faithful"
    assert a["wide"]["switch"] == "kit" and a["wide"]["label"] == di.TREATMENTS["kit"]
    assert ("Plotted: treatment (a), kit-faithful; (b), unanswerable for every arm, is moot: no row exceeds Callo's "
            "255-option limit, so it equals (a).") in _fp(a)
    assert a["wide"]["moot"] is True and a["treatments"]["all-unsupported"]["moot"] is True

    b = build(tmp_path / "b", both=True, switch="all-unsupported", over_limit=5)
    root = _svg(b)
    assert root.get("data-treatment") == "all-unsupported"
    assert root.get("data-treatment-label") == "(b) unanswerable for every arm"
    assert "Plotted: treatment (b), counted as unanswerable for every arm" in _fp(b)
    assert "(a), kit-faithful, is in FIGURES.md" in _fp(b)
    boot_b = json.loads((tmp_path / "b" / "boot-b.json").read_text())
    for p in b["points"]:                                     # the plotted numbers are treatment (b)'s
        assert p["index"] == boot_b["arms"][p["arm"]]["index"]
    tiel = next(p for p in b["points"] if p["arm"] == "tiel-b0")
    assert tiel["index"] == pytest.approx(fx.ARMS["tiel-b0"][0] - 2.0)


def test_the_switch_refuses_a_treatment_without_its_bootstrap(tmp_path):
    s = fx.setup(tmp_path, switch="all-unsupported")
    with pytest.raises(di.Refusal, match="no bootstrap file"):
        di.build(di.load_spec(s["spec_path"]))
    with pytest.raises(di.Refusal, match="wide_options must be"):
        di.build(di.load_spec(fx.rewrite_spec(s, wide_options="drop")))


def test_both_treatments_are_in_figures_md(tmp_path):
    import figures_md
    both = build(tmp_path / "both", both=True, over_limit=5)
    md = figures_md.render(None, None, both)
    assert "**Treatment (a) kit-faithful (plotted).**" in md and "**Treatment (b) unanswerable for every arm.**" in md
    sec = md.split("### Both treatments")[1]
    a_tab, b_tab = sec.split("**Treatment (b)")
    assert "| Tiel-Coder · no think | 38.30 " in a_tab and "| Tiel-Coder · no think | 36.30 " in b_tab

    # no row over callo's 255-option limit: (b) equals (a), and FIGURES.md says it is moot
    one = build(tmp_path / "one")
    md = figures_md.render(None, None, one)
    assert "**Treatment (a) kit-faithful (plotted).**" in md
    assert ("**Treatment (b) unanswerable for every arm.** Moot: no row exceeds Callo's 255-option limit, so it "
            "equals (a).") in md
    assert md.count("| Tiel-Coder · no think |") == 1

    # rows over the limit and no (b) bootstrap: not moot, and FIGURES.md says (b) is not computed
    over = build(tmp_path / "over", over_limit=5)
    md = figures_md.render(None, None, over)
    assert "**Treatment (b) unanswerable for every arm.** not computed (one treatment since 2026-10-10)." in md
    assert "Moot" not in md and "is not computed for this bake" in _fp(over)


def test_the_whiskers_are_vs_the_reference_and_the_second_reference_only_in_the_table(snap):
    root = _svg(snap)
    by = {p["arm"]: p for p in snap["points"]}
    whiskers = {e.get("data-arm"): e for e in _role(root, "whisker")}
    assert set(whiskers) == {a for a, p in by.items() if p["on_plate"]} - {fx.REF}
    for arm, e in whiskers.items():
        assert e.get("data-vs") == fx.REF
        d, lo, hi = by[arm]["dref"]
        assert [float(v) for v in e.get("data-dref").split(",")] == by[arm]["dref"]
        assert float(e.get("data-lo")) == pytest.approx(by[arm]["index"] + lo - d)
        assert float(e.get("data-hi")) == pytest.approx(by[arm]["index"] + hi - d)
    for e in _role(root, "panel-whisker"):
        assert [float(v) for v in e.get("data-dref").split(",")] == by[e.get("data-arm")]["area_dref"][e.get("data-area")]
    # no reference line on the chart or the panels: the whiskers carry the comparison
    assert not _role(root, "ref") and not _role(root, "ref-label") and not _role(root, "panel-ref")
    (legend,) = _role(root, "legend")
    assert legend.get("data-text") == "mean / p95 / paired 95 % CI vs Clef-Flash BF16"
    assert "dashed line" not in _fp(snap) and "dashed line" not in " ".join(
        e.get("data-text") or "" for e in root.iter())
    # the second reference's Δ: in the table, nowhere else
    second = [e for e in root.iter() if "dsecond" in (e.get("data-key") or "")]
    assert second and {e.get("data-role") for e in second} <= {"table-cell", "table-ci"}
    table = next(iter(_role(root, "table")))
    in_table = {id(e) for e in table.iter()}
    assert all(id(e) in in_table for e in second)
    values = {v for p in snap["points"] if p["dsecond"] for v in p["dsecond"]}
    for e in root.iter():
        if e.get("data-role") in ("whisker", "panel-whisker"):
            assert not values & {float(v) for v in e.get("data-dref").split(",")}
    assert "The difference vs Clef 27B is in the table only." in _fp(snap)


@pytest.mark.parametrize(("series", "budget"), [("tiel", 512), ("tiel", 1024), ("sharp-minicpm", 256),
                                                ("sharp-minicpm", 512)])
def test_a_budget_not_run_on_di_is_refused(tmp_path, series, budget):
    s = fx.setup(tmp_path)
    spec = s["spec"]
    ser = next(x for x in spec["series"] if x["name"] == series)
    zero = ser["budgets"]["0"]
    ser["budgets"] = {"0": zero, str(budget): f"{series}-b{budget}"}
    ser.pop("not_run", None)                                  # the rule holds even when the spec omits it
    with pytest.raises(di.Refusal, match="not run on DI"):
        di.build(di.load_spec(fx.rewrite_spec(s, series=spec["series"])))


def test_no_tiel_point_at_512_or_1024_even_when_the_bootstrap_has_one(tmp_path):
    extra = {"tiel-b512": fx.ARMS["tiel-b256"], "tiel-b1024": fx.ARMS["tiel-b256"],
             "sharp-minicpm-b256": fx.ARMS["sharp-minicpm-b0"], "sharp-minicpm-b512": fx.ARMS["sharp-minicpm-b0"]}
    snap = build(tmp_path, extra_bootstrap_arms=extra)
    root = _svg(snap)
    drawn = {(g.get("data-series"), g.get("data-budget")) for g in _role(root, "point")}
    assert {b for s_, b in drawn if s_ == "tiel"} == {"0", "256"}
    assert {b for s_, b in drawn if s_ == "sharp-minicpm"} == {"0", "1024"}
    assert not [e for e in root.iter() if any(x in (e.get("data-arm") or "") for x in extra)]
    text = _fp(snap)
    assert "Tiel-Coder budgets 512 and 1024 not run on DI (only 0 and 256)" in text
    assert "Sharp-MiniCPM5-2B budgets 256 and 512 not run on DI (only 0 and 1024)" in text


def test_the_plot_step_refuses_an_arm_with_unexplained_errors(tmp_path, capsys):
    s = fx.setup(tmp_path, explain=False)
    with pytest.raises(di.Refusal, match="unexplained errors"):
        di.build(di.load_spec(s["spec_path"]))
    assert di.main(["build", "--spec", str(s["spec_path"]), "--out", str(tmp_path / "o.json")]) == 2
    assert "sharp-minicpm-b1024" in capsys.readouterr().err
    # a cause in the union file itself (di_union.py's explain file) is enough
    (tmp_path / "union.json").write_text(json.dumps(fx.union(list(fx.ARMS), error_cause="known: fixed")))
    assert di.build(di.load_spec(s["spec_path"]))["unanswered"]["errors"][0]["cause"] == "known: fixed"
    # and the draw step re-checks the snapshot
    snap = build(tmp_path / "x")
    snap["unanswered"]["errors"][0]["cause"] = None
    with pytest.raises(di.Refusal, match="unexplained"):
        di.draw(snap)
    (tmp_path / "bad.json").write_text(json.dumps(snap))
    assert di.main(["draw", "--data", str(tmp_path / "bad.json"), "--out-dir", str(tmp_path)]) == 2


def test_pending_rows_are_refused(tmp_path):
    s = fx.setup(tmp_path)
    u = fx.union(list(fx.ARMS))
    u["pending"]["tiel-b256"] = 4
    (tmp_path / "union.json").write_text(json.dumps(u))
    with pytest.raises(di.Refusal, match="pending"):
        di.build(di.load_spec(s["spec_path"]))


# ---- the build's other refusals -----------------------------------------------------------------------------------

def test_a_stale_bootstrap_is_refused(tmp_path):
    s = fx.setup(tmp_path)
    (tmp_path / "di" / "tiel-b0" / "results.derived.jsonl").write_text("changed\n")
    with pytest.raises(di.Refusal, match="changed since the bootstrap read it"):
        di.build(di.load_spec(s["spec_path"]))


def test_a_bootstrap_under_1000_resamples_is_refused(tmp_path):
    s = fx.setup(tmp_path)
    (tmp_path / "boot-a.json").write_text(json.dumps(fx.bootstrap(tmp_path, fx.ARMS, n=200)))
    with pytest.raises(di.Refusal, match="1000"):
        di.build(di.load_spec(s["spec_path"]))


def test_an_arm_served_at_another_budget_or_model_is_refused(tmp_path):
    s = fx.setup(tmp_path)
    meta_p = tmp_path / "di" / "tiel-b256" / "di_meta.json"
    meta = json.loads(meta_p.read_text())
    cmd = meta["sessions"][0]["cmd"]
    cmd[cmd.index("--think-budget") + 1] = "512"
    meta_p.write_text(json.dumps(meta))
    with pytest.raises(di.Refusal, match="--think-budget 512"):
        di.build(di.load_spec(s["spec_path"]))
    s = fx.setup(tmp_path / "m")
    spec = s["spec"]
    next(x for x in spec["series"] if x["name"] == "cflash-q4")["model_file"] = "Other.gguf"
    with pytest.raises(di.Refusal, match="labels it Other.gguf"):
        di.build(di.load_spec(fx.rewrite_spec(s, series=spec["series"])))


def test_a_half_present_callo_line_is_refused_and_an_absent_one_is_pending(tmp_path):
    snap = build(tmp_path / "ok")
    assert snap["pending"] == ["flashnext-sharp"]
    assert "Pending. Qwen3.8-Flash-Next (Sharp) at budgets 0 and 256 (release bake)." in _fp(snap)
    arms = {k: v for k, v in fx.ARMS.items() if k != "tiel-b256"}
    s = fx.setup(tmp_path / "half", arms=arms)
    with pytest.raises(di.Refusal, match="only some of its arms"):
        di.build(di.load_spec(s["spec_path"]))


# ---- the preliminary marking ------------------------------------------------------------------------------------

def test_the_preliminary_bake_is_marked_and_names_its_caveats(snap):
    root = _svg(snap)
    (badge,) = _role(root, "badge")
    assert badge.get("data-text").startswith("PRELIMINARY · ")
    text = _fp(snap)
    assert ("Preliminary. Callo arms measured before a readout fix (end-of-turn token at the answer slot, ~0.3–1 % of "
            "readouts walked); re-measured on the release bake.") in text
    assert "Timing. Clef 27B ran beside CPU-heavy test runs (~14:10–15:40 on 2026-10-10); re-timed for the release bake." in text
    assert "GPQA-Diamond:test:99 counts as unanswered for every arm: a Callo 500, root-caused and fixed" in text
    assert "not the board's Full score" in text and "kit 9eb2dbe2" in text
    assert di.CHANCE_TEXT in text and di.ANSWERING_TEXT in text
    assert "rows answered by every arm" in text and "one HTTP client that is the same for every arm" in text
    assert "--cache on (default)" in text and "llama.cpp b50-test (Callo) and b11379 1537a0a8 (Clef)" in text
    assert ("Callo 0.3.0 development builds: the first build (Tiel-Coder; Sharp-MiniCPM5-2B think 1024) and the refit "
            "build (Sharp-MiniCPM5-2B no think)") in text
    assert not re.search(r"\b[0-9a-f]{7}\b", text.replace("4ebdf2c", "").replace("1537a0a8", ""))   # no callo commit


def test_a_release_bake_drops_the_preliminary_marks(tmp_path):
    s = fx.setup(tmp_path)
    snap = di.build(di.load_spec(fx.rewrite_spec(s, preliminary=False)))
    root = _svg(snap)
    assert not _role(root, "badge")
    assert "Preliminary." not in _fp(snap) and "Timing." not in _fp(snap)


def test_every_fine_print_entry_and_the_story_are_on_the_plate(snap):
    root = _svg(snap)
    texts = [g.get("data-text") for g in _role(root, "fine-print")]
    assert len(texts) == len(snap["fine_print"])
    for f, t in zip(snap["fine_print"], texts, strict=True):
        assert t == (f["lead"] + " " if f["lead"] else "") + f["text"]
    (sub,) = _role(root, "subtitle")
    assert sub.get("data-text") == snap["story"]["subtitle"]
    assert snap["story"]["subtitle"] == (
        "Tiel-Coder with no thinking scores 38.3, 16.5 points under Clef-Flash BF16 (CI 13.1–20.4), at 0.93× its mean "
        "request time; a 256-token think per question lifts it to 48.8, 5.9 under (CI 2.5–9.8), at 5.8× the time.")


# ---- the index without BRIGHT: a footnote, never the headline -------------------------------------------------------

SHIFTS = {"tiel-b0": 0.9, "tiel-b256": 1.4, "sharp-minicpm-b0": 0.83, "sharp-minicpm-b1024": 1.5, "clef27b": 1.86,
          "cflash-q4": 2.12, "cflash-bf16": 1.86}


def test_the_index_without_bright_is_a_one_line_footnote_with_its_numbers_in_figures_md(tmp_path, monkeypatch):
    import figures_md
    s = fx.setup(tmp_path)
    snap = di.build(di.load_spec(fx.with_sensitivity(s, SHIFTS)))
    sens = snap["sensitivity"]
    assert sens["shift"] == [pytest.approx(0.83), pytest.approx(2.12)]
    assert sens["order_same"] is True and sens["flips"] == [] and sens["pairs"] == 12
    assert sens["arms"]["tiel-b0"] == {
        "with": 38.3, "without": 39.2, "without_ci": [35.8, 42.3],
        "dref": [-16.45, -20.35, -13.05], "dref_without": [-17.41, -21.31, -14.01],
        "dsecond": [-19.7, -23.6, -16.3], "dsecond_without": [-20.66, -24.56, -17.26]}
    (fp,) = [f for f in snap["fine_print"] if f["lead"] == "Without BRIGHT."]
    assert fp["text"] == ("Every arm +0.8 to +2.1 (the index's own re-weighting, the same draws); the order is "
                          "unchanged and no paired CI changes whether it includes 0; Clef-Flash BF16 56.6.")
    assert di._one_line(fp["lead"], fp["text"])                # one line; the callo lines' values would not fit
    monkeypatch.setattr(di, "_one_line", lambda lead, text: True)
    assert di.sensitivity_text(snap).endswith("; Clef-Flash BF16 56.6, Tiel-Coder 39.2 and 50.2, Sharp-MiniCPM5-2B "
                                              "23.4 and 43.0.")              # given where they fit
    monkeypatch.undo()
    leads = [f["lead"] for f in snap["fine_print"]]
    assert leads.index("Without BRIGHT.") == leads.index("Unanswered.") + 1
    assert "BRIGHT" not in snap["title"] + snap["story"]["subtitle"]
    root = _svg(snap)
    assert any((t.get("data-text") or "").startswith("Without BRIGHT. Every arm +0.8") for t in _role(root, "fine-print"))
    md = figures_md.render(None, None, snap)
    assert "### Without BRIGHT" in md
    assert "| Tiel-Coder · no think | 38.30 | 39.20 [35.80, 42.30] | −16.45 [−20.35, −13.05] | −17.41 [−21.31, −14.01] |" in md
    made = " ".join((f["lead"] or "") + " " + f["text"] for f in snap["made"])
    assert ("BRIGHT (catalog 36) dropped by the Decision Index kit's own retirement: the scored edition is 0.3-drop-36, "
            "and its aggregate re-weights the retrieval area's benchmarks from 4, 5 and 36: 0.1818; 37, 56 and 61: "
            "0.1515 to 4 and 5: 0.2222; 37, 56 and 61: 0.1852") in made
    assert "the order of all 7 arms is unchanged; none of the 12 paired CIs (vs Clef-Flash BF16 and vs Clef 27B)" in made
    assert "source `nobright.json` (sha256 " in made and "pending" not in made


def test_the_footnote_says_when_the_order_or_a_ci_verdict_changes(tmp_path):
    s = fx.setup(tmp_path)
    shifts = {**SHIFTS, "cflash-q4": 6.0}                   # Q4 passes Clef 27B and clears Clef-Flash BF16's CI
    snap = di.build(di.load_spec(fx.with_sensitivity(s, shifts)))
    sens = snap["sensitivity"]
    assert sens["order_same"] is False and sens["order_without"][0] == "cflash-q4"
    assert {(f["arm"], f["vs"]) for f in sens["flips"]} == {("cflash-q4", fx.REF)}
    (fp,) = [f for f in snap["fine_print"] if f["lead"] == "Without BRIGHT."]
    assert "the order of the arms changes and 1 of 12 paired CIs change whether they include 0" in fp["text"]
    assert fp["text"].startswith("Every arm +0.8 to +6.0 ") and fp["text"].endswith("; Clef-Flash BF16 56.6.")
    made = " ".join((f["lead"] or "") + " " + f["text"] for f in snap["made"])
    assert "1 of the 12 paired CIs change whether they include 0: Clef-Flash 9B · Q4_K_XL (our quant) vs Clef-Flash" in made


@pytest.mark.parametrize("case, match", [
    ("seed", "seed 1 does not pair"), ("inputs", "not scored from the plotted bootstrap's arms and results"),
    ("drop", r"drops \['GPQA'\], the spec says 'BRIGHT'"), ("n", "999 resamples"),
])
def test_a_sensitivity_bootstrap_that_does_not_pair_is_refused(tmp_path, case, match):
    s = fx.setup(tmp_path)
    spec = fx.with_sensitivity(s, SHIFTS, drop="GPQA" if case == "drop" else "BRIGHT")
    doc = json.loads((tmp_path / "nobright.json").read_text())
    if case == "seed":
        doc["seed"] = 1
    elif case == "inputs":
        doc["inputs"]["tiel-b0"]["derived_sha256"] = "0" * 64
    elif case == "n":
        doc["n"] = 999
    (tmp_path / "nobright.json").write_text(json.dumps(doc))
    with pytest.raises(di.Refusal, match=match):
        di.build(di.load_spec(spec))


def test_figures_md_names_the_di_inputs_by_file_and_sha256_never_by_a_local_path(tmp_path):
    import figures_md
    s = fx.setup(tmp_path, variants=True)
    snap = di.build(di.load_spec(fx.with_sensitivity(s, SHIFTS)))
    md = figures_md.render(None, None, snap)
    assert str(tmp_path) not in md
    assert f"- bootstrap, treatment (a): `boot-a.json` {di.sha256_file(tmp_path / 'boot-a.json')[:16]}…" in md
    assert "- paired vs Tiel-Coder · think 256: `paired.json` " in md and "- bootstrap without BRIGHT: `nobright.json` " in md
    assert "- Tiel-Coder · no think: `di_meta.json` " in md and "Source `boot-a.json` (sha256 " in md
    assert "**Paired vs Tiel-Coder · think 256.** `paired.json` (sha256 " in md


def test_the_snapshot_records_paths_relative_to_the_root_and_a_file_outside_it_by_name(tmp_path):
    s = fx.setup(tmp_path, variants=True)
    snap = di.build(di.load_spec(fx.with_sensitivity(s, SHIFTS)))
    assert str(tmp_path) not in json.dumps(snap, ensure_ascii=False) and "root" not in snap["generator"]
    pv = next(p for p in snap["points"] if p["arm"] == "tiel-b0")["provenance"]
    assert pv["dir"] == "di/tiel-b0" and pv["files"][0]["path"] == "di/tiel-b0/di_meta.json"
    assert pv["model"] == fx.MODELS["tiel"] and pv["think_profile"] == "tiel.think.json"     # outside the root
    assert snap["treatments"]["kit"]["path"] == "boot-a.json" and snap["sensitivity"]["path"] == "nobright.json"
    assert [c["path"] for c in snap["paired"]] == ["paired.json"]
    assert {f["path"] for f in snap["inputs"]} >= {"boot-a.json", "paired.json", "nobright.json", "union.json"}
    assert snap["builds"] == [{"commit": "b463e5f", "name": "the first build"},
                              {"commit": "9d52aaf", "name": "the refit build"},
                              {"commit": "0d33846", "name": "the gate build"}]


def test_a_callo_commit_the_builds_table_does_not_name_is_refused(tmp_path):
    s = fx.setup(tmp_path)
    pins = {**s["spec"]["callo_commits"], "tiel-b0": "1234567"}
    with pytest.raises(di.Refusal, match="1234567 is not in brand/data/builds.json"):
        di.build(di.load_spec(fx.rewrite_spec(s, callo_commits=pins)))


def test_the_committed_data_records_no_local_path():
    """The snapshots and specs name each source relative to the build's --root (a file outside it by its name)."""
    for f in sorted((Path(di.HERE) / "data").glob("*.json")):
        text = f.read_text(encoding="utf-8")
        for s_ in ("/Users/", "/Volumes/", "bench-qwen36"):
            assert s_ not in text, (f.name, s_)


# ---- the data-only arm, display names ------------------------------------------------------------------------------

def test_a_data_only_arm_is_in_the_snapshot_and_figures_md_but_not_drawn(tmp_path):
    import figures_md
    snap = build(tmp_path)
    q4 = next(p for p in snap["points"] if p["arm"] == "cflash-q4")
    boot = json.loads((tmp_path / "boot-a.json").read_text())["arms"]["cflash-q4"]
    assert q4["on_plate"] is False and q4["index"] == boot["index"] and q4["dref"][0] == boot["delta_vs_reference"]["point"]
    for theme in ("dark", "light"):
        root = _svg(snap, theme)
        assert not [e for e in root.iter() if "cflash-q4" in (e.get("data-arm") or "") + (e.get("data-series") or "")
                    + (e.get("data-key") or "")]
        assert {e.get("data-arm") for e in _role(root, "table-row")} == {p["arm"] for p in snap["points"]} - {"cflash-q4"}
        clef_flash = [g for g in _role(root, "point") if g.get("data-arm").startswith("cflash")]
        assert [g.get("data-arm") for g in clef_flash] == [fx.REF]
    md = figures_md.render(None, None, snap)
    assert "| Clef-Flash 9B · Q4_K_XL (our quant) (not drawn) | 55.49 " in md
    assert "**Clef-Flash 9B · Q4_K_XL (our quant) (measured, not drawn).** index 55.49 " in md
    assert "Clef-Flash Q4 ~18:00–18:26" in md                   # the full timing note lives in FIGURES.md
    s = fx.setup(tmp_path / "r")
    ser = s["spec"]["series"]
    next(x for x in ser if x["name"] == fx.REF)["plate"] = False
    with pytest.raises(di.Refusal, match="data-only"):
        di.build(di.load_spec(fx.rewrite_spec(s, series=ser)))


def test_no_displayed_string_says_ud(snap):
    """None of these quants are Unsloth's: the plate, its fine print and FIGURES.md show Q4_K_XL / Q6_K_XL; real file
    names (provenance, FIGURES.md code spans) keep the bake's own names."""
    import figures_md
    for theme in ("dark", "light"):
        root = _svg(snap, theme)
        shown = [e.get("data-text") for e in root.iter() if e.get("data-text")] + [root.get("aria-label")]
        assert shown and not [t for t in shown if "UD-" in t]
    words = [snap["title"], snap["story"]["subtitle"], _fp(snap)] + [p["label"] for p in snap["points"]]
    assert not [w for w in words if "UD-" in w]
    md = figures_md.render(None, None, snap)
    assert "UD-" not in re.sub(r"`[^`]*`", "", md)
    assert "`Tiel-UD-Q4_K_XL.gguf`" in md                      # the real file name, as a path
    tiel = next(p for p in snap["points"] if p["arm"] == "tiel-b0")
    assert tiel["provenance"]["model"].endswith("Tiel-UD-Q4_K_XL.gguf")


# ---- the callo labels hug their own lines ------------------------------------------------------------------------

def _near(box: list[float], x1: float, y1: float, x2: float, y2: float, pad: float) -> bool:
    """Whether the segment comes within `pad` of the box."""
    n = max(2, int(math.hypot(x2 - x1, y2 - y1)))
    return any(box[0] - pad <= x1 + (x2 - x1) * i / n <= box[2] + pad and
               box[1] - pad <= y1 + (y2 - y1) * i / n <= box[3] + pad for i in range(n + 1))


def test_the_callo_labels_hug_their_lines(snap):
    """Tiel-Coder's label sits just above its own line over the middle stretch; Sharp-MiniCPM5-2B's just under its line
    towards the think point; each clear of every whisker and tail (its own included) and of every other line. (A label
    with no clear spot on either side of its line is placed freely, still clear of everything.)"""
    root = _svg(snap)
    sides = {g.get("data-series"): g.get("data-side") for g in _role(root, "series-label")}
    if "flashnext-sharp" not in sides:
        assert sides["tiel"] == "above" and sides["sharp-minicpm"] == "below"
    segs = {e.get("data-series"): [float(e.get(k)) for k in ("x1", "y1", "x2", "y2")] for e in _role(root, "callo-line")}
    obstacles = [(e.get("data-arm"), [float(e.get(k)) for k in ("x1", "y1", "x2", "y2")])
                 for e in _role(root, "whisker") + _role(root, "tail")]
    pts = {g.get("data-arm"): (float(g.get("data-x")), float(g.get("data-y"))) for g in _role(root, "point")}
    for g in _role(root, "series-label"):
        name = g.get("data-series")
        if name not in segs:
            continue
        box = [float(v) for v in g.get("data-box").split(",")]
        ax, ay, bx, by = segs[name]

        def on_line(x: float, ax=ax, ay=ay, bx=bx, by=by) -> float:
            return ay + (by - ay) * min(max((x - ax) / (bx - ax), 0.0), 1.0)

        y_left, y_right = on_line(box[0]), on_line(box[2])
        u = ((box[0] + box[2]) / 2 - ax) / (bx - ax)
        if g.get("data-side") == "above":
            assert box[3] < min(y_left, y_right) and min(y_left, y_right) - box[3] <= 12, name
            assert 0.15 <= u <= 0.75, (name, u)       # over the middle stretch (left of it when its think point's
            #                                            label takes the ground above the line's end)
        elif g.get("data-side") == "below":
            assert box[1] > max(y_left, y_right) and box[1] - max(y_left, y_right) <= 12, name
            assert u >= 0.5, (name, u)                            # towards the think point
            think = max((a for a in pts if a.startswith(name)), key=lambda a: pts[a][0])
            assert box[1] > pts[think][1]                         # under the think point too
        assert not _near(box, ax, ay, bx, by, 2), name           # never on its own line
        for other, xy in segs.items():
            if other != name:
                assert not _near(box, *xy, 4), (name, other)
        for arm, xy in obstacles:
            assert not _near(box, *xy, 4), (name, arm)            # no whisker or tail, its own included
        for arm, (x, y) in pts.items():
            assert not _near(box, x, y, x, y, 8), (name, arm)


# ---- legibility invariants ----------------------------------------------------------------------------------------

def test_every_marker_sits_on_a_ground_halo_drawn_beneath_every_marker(snap):
    """Each layer of marks lies on its own ground halos (the other callo lines' marks, then the accent line over them,
    then the accent's and Clef's marks): one halo per drawn point, every halo before its layer's first mark."""
    from figkit import THEMES
    for theme in ("dark", "light"):
        for root in (_svg(snap, theme), _chart(snap, theme)):
            halos = _role(root, "halos")
            panels = [h for h in halos if any(id(h) in {id(x) for x in g.iter()} for g in _role(root, "panel"))]
            plot = [h for h in halos if h not in panels]
            assert len(panels) == (len(snap["areas"]) if _role(root, "panel") else 0)
            for h in halos:
                assert all(c.get("fill") == THEMES[theme]["ground"] == c.get("stroke") for c in h)
            order = list(root.iter())
            assert order.index(plot[0]) < min(order.index(g) for g in _role(root, "point"))
            assert sum(len(list(h)) for h in plot) == len(_drawn(snap))


def test_nothing_plotted_falls_outside_the_plot_box_and_labels_do_not_overlap(snap):
    _check_layout(snap)


def _check_layout(snap: dict) -> None:
    for root in [f(snap, th) for th in ("dark", "light") for f in (_svg, _chart)]:
        x0, y0, x1, y1 = (float(v) for v in root.get("data-plot-box").split(","))
        for g in _role(root, "point"):
            assert x0 <= float(g.get("data-x")) <= x1 and y0 <= float(g.get("data-y")) <= y1
        for e in _role(root, "whisker") + _role(root, "tail") + _role(root, "ref") + _role(root, "callo-line"):
            assert x0 <= float(e.get("x1")) <= x1 and x0 <= float(e.get("x2")) <= x1
            assert y0 <= float(e.get("y1")) <= y1 and y0 <= float(e.get("y2")) <= y1
        boxes = [[float(v) for v in g.get("data-box").split(",")]
                 for g in _role(root, "series-label") + _role(root, "point-label")]
        for b in boxes:
            assert x0 <= b[0] and b[2] <= x1 and y0 <= b[1] and b[3] <= y1
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert min(a[2], b[2]) - max(a[0], b[0]) <= 0.5 or min(a[3], b[3]) - max(a[1], b[1]) <= 0.5


def _cross(a: tuple, b: tuple) -> bool:
    """Whether segments a and b (x1, y1, x2, y2) cross."""
    def side(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    return (side(*b[:2], *b[2:], *a[:2]) * side(*b[:2], *b[2:], *a[2:]) < 0
            and side(*a[:2], *a[2:], *b[:2]) * side(*a[:2], *a[2:], *b[2:]) < 0)


def _check_owned(root: ET.Element) -> None:
    """Every budget label reads as its own point's: straight past its whisker (the whisker joins them), beside the
    point and nearer it than any other marker, or joined by its own leader; and no other series' line runs between
    the point and its label."""
    pts = {g.get("data-arm"): (float(g.get("data-x")), float(g.get("data-y")), g.get("data-series"))
           for g in _role(root, "point")}
    lines = {e.get("data-series"): tuple(float(e.get(k)) for k in ("x1", "y1", "x2", "y2"))
             for e in _role(root, "callo-line")}
    leaders = {e.get("data-for"): (float(e.get("x1")), float(e.get("y1"))) for e in _role(root, "leader")}
    assert _role(root, "point-label")
    for g in _role(root, "point-label"):
        arm = g.get("data-arm")
        x, y, ser = pts[arm]
        b = [float(v) for v in g.get("data-box").split(",")]

        def dist(px, py, b=b):
            return math.hypot(min(max(px, b[0]), b[2]) - px, min(max(py, b[1]), b[3]) - py)
        cx, cy = min(max(x, b[0]), b[2]), min(max(y, b[1]), b[3])
        others = min(dist(ox, oy) for a, (ox, oy, _) in pts.items() if a != arm)
        if arm in leaders:                                   # its leader starts at its own point
            assert math.hypot(leaders[arm][0] - x, leaders[arm][1] - y) <= 10, arm
        else:
            assert b[0] < x < b[2] or dist(x, y) + 3 < others, arm
        for name, seg in lines.items():
            if name != ser:
                assert not _cross((x, y, cx, cy), seg), (arm, name)


def test_every_budget_label_reads_as_its_own_points(snap, ee_snap):
    for s_ in (snap, ee_snap):
        for theme in ("dark", "light"):
            _check_owned(_svg(s_, theme))
            _check_owned(_chart(s_, theme))


def test_every_plotted_series_is_labelled_and_clef_flash_bf16_reads_as_one_point(snap):
    root = _svg(snap)
    labelled = {s for g in _role(root, "series-label") for s in g.get("data-series").split(",")}
    assert labelled == {s["name"] for s in snap["series"] if s["status"] == "present" and s["on_plate"]}
    bf16 = next(g for g in _role(root, "series-label") if g.get("data-series") == fx.REF)
    assert bf16.get("data-text") == "Clef-Flash 9B · BF16 · 54.8 / llama.cpp native Clef"


def test_coincident_single_markers_share_one_label(tmp_path):
    root = _svg(build(tmp_path, q4_on_plate=True))
    shared = [g for g in _role(root, "series-label") if "," in g.get("data-series")]
    assert [sorted(g.get("data-series").split(",")) for g in shared] == [["cflash-bf16", "cflash-q4"]]


def test_a_third_callo_line_fits(tmp_path):
    arms = {**fx.ARMS, "flashnext-sharp-b0": (44.0, 1450.0, 6900.0, fx.ARMS["tiel-b0"][3]),
            "flashnext-sharp-b256": (51.5, 9800.0, 52000.0, fx.ARMS["tiel-b256"][3])}
    fx.CALLO.update({"flashnext-sharp-b0": 0, "flashnext-sharp-b256": 256})
    fx.MODELS["flashnext-sharp"] = "Next.gguf"
    try:
        snap = build(tmp_path, arms=arms)
    finally:
        fx.CALLO.pop("flashnext-sharp-b0")
        fx.CALLO.pop("flashnext-sharp-b256")
    assert snap["pending"] == []
    root = _svg(snap)
    lines = {e.get("data-series"): e for e in _role(root, "callo-line")}
    assert set(lines) == {"tiel", "sharp-minicpm", "flashnext-sharp"}
    assert lines["flashnext-sharp"].get("stroke-dasharray") and not lines["sharp-minicpm"].get("stroke-dasharray")
    assert len(_role(root, "table-row")) == 8                  # the data-only Clef-Flash Q4 is not a row
    test_the_callo_labels_hug_their_lines(snap)
    test_nothing_plotted_falls_outside_the_plot_box_and_labels_do_not_overlap(snap)


def test_a_native_decision_model_beside_clef_is_a_single_marker_of_its_own_family(tmp_path):
    """d1 (Liquid AI's decision model on its own llama.cpp build) sits beside Clef in Clef's style: one mark, no line,
    its label (name · index over its runtime), a table row, a slot in every area panel; the Builds line and the run
    dates name it by its family, never as Clef; one fine-print entry says its one-question prompts on the over-cap rows
    would fit its context, and FIGURES.md names its own counts by file and sha256."""
    import figures_md
    snap = build(tmp_path, d1=True)
    d1 = next(p for p in snap["points"] if p["arm"] == fx.D1)
    boot = json.loads((tmp_path / "boot-a.json").read_text())["arms"][fx.D1]
    assert d1["kind"] == "clef" and d1["on_plate"] and d1["label"] == "d1-3B · Q4_K_M"
    assert d1["index"] == boot["index"] == 46.07 and d1["dref"][0] == boot["delta_vs_reference"]["point"]
    assert d1["provenance"]["build"] == "23deec02"
    assert d1["provenance"]["own_tokens"]["over_cap_question_max"] == [1198, 7044]
    assert {"path": "d1_tokens.json", "sha256": d1["provenance"]["own_tokens"]["sha256"]} in d1["provenance"]["files"]
    for theme in ("dark", "light"):
        for root in (_svg(snap, theme), _chart(snap, theme)):
            (pt,) = [g for g in _role(root, "point") if g.get("data-arm") == fx.D1]
            assert pt.get("data-series") == "d1" and list(pt)[0].tag.endswith("path")       # the triangle
            assert not [e for e in _role(root, "callo-line") if e.get("data-series") == "d1"]
            (lab,) = [g for g in _role(root, "series-label") if g.get("data-series") == "d1"]
            assert lab.get("data-text") == "d1-3B · Q4_K_M · 46.1 / Liquid AI · llama.cpp native"
        root = _svg(snap, theme)
        rows = [r.get("data-arm") for r in _role(root, "table-row")]
        assert rows[-1] == fx.D1 and _role(root, "table-row")[-1].get("data-drawn") == "true"
        assert {g.get("data-area") for g in _role(root, "panel-point") if g.get("data-arm") == fx.D1} == set(fx.AREAS)
    _check_layout(snap)
    text = _fp(snap)
    assert "llama.cpp b50-test (Callo), b11379 1537a0a8 (Clef) and 23deec02 (d1)" in text
    assert "run Callo 2026-10-10, Clef 2026-10-10, d1 2026-10-14 (UTC)" in text
    leads = [f["lead"] for f in snap["fine_print"]]
    assert leads[leads.index("Unanswered.") + 1] == "d1-3B."
    assert "The exception" not in snap["fine_print"][leads.index("Unanswered.")]["text"]
    assert snap["fine_print"][leads.index("d1-3B.")]["text"] == (
        "Answers each question in its own prompt (the state, then that question), so its prompts on the 2 BRIGHT rows "
        "(7,044 tokens at most) would fit its 16,384-token context; they still count as wrong for every arm.")
    md = figures_md.render(None, None, snap)
    assert "| d1-3B · Q4_K_M | 46.07 " in md
    assert "its own prompt-token counts `d1_tokens.json` (sha256 " in md
    assert "880 questions on 98 rows, one prompt each, the longest 9,393 tokens" in md
    # without d1, nothing it adds is there (the Clef-only wording is unchanged)
    plain = _fp(build(tmp_path / "plain"))
    assert "llama.cpp b50-test (Callo) and b11379 1537a0a8 (Clef)" in plain and "d1" not in plain


@pytest.mark.parametrize("over, match", [
    ({"gguf": "ff" * 32}, "another model"),
    ({"n_ctx": 32768}, "32768-token context"),
    ({"sample": "0" * 64}, "another sample"),
    ({"own": 3}, "pre-marks of its own"),
    ({"premarked": [fx.PREMARKED[0]]}, "not the plate's 2 over-cap rows"),
])
def test_a_native_models_own_counts_must_be_its_runs(tmp_path, over, match):
    s = fx.setup(tmp_path, d1=True)
    (tmp_path / "d1_tokens.json").write_text(json.dumps(fx.d1_counts(**over)))
    with pytest.raises(di.Refusal, match=match):
        di.build(di.load_spec(s["spec_path"]))


def test_a_native_models_server_must_be_the_build_its_label_names(tmp_path):
    s = fx.setup(tmp_path, d1=True)
    ser = s["spec"]["series"]
    next(x for x in ser if x["name"] == "d1")["build"] = "b11379 1537a0a8"
    with pytest.raises(di.Refusal, match="reports commit 23deec0"):
        di.build(di.load_spec(fx.rewrite_spec(s, series=ser)))


@pytest.mark.parametrize("series, change, match", [
    ("tiel", {"family": "d1"}, "for a single-marker series"),
    ("d1", {"answering": None}, "words for how it answers"),
    ("d1", {"family": ""}, "non-empty string"),
])
def test_family_answering_and_own_counts_belong_to_a_single_marker(tmp_path, series, change, match):
    s = fx.setup(tmp_path, d1=True)
    ser = s["spec"]["series"]
    next(x for x in ser if x["name"] == series).update(change)
    with pytest.raises(di.Refusal, match=match):
        di.build(di.load_spec(fx.rewrite_spec(s, series=ser)))


def test_every_colour_is_from_the_brand_palette_and_both_themes_carry_the_same_information(snap):
    import figkit
    palette = {figkit.INK, figkit.BONE, figkit.SIGNAL, figkit.NIGHT, figkit.DUSK, figkit.TILE, figkit.MIST, figkit.SLATE}
    for theme in ("dark", "light"):
        for svg in (di.draw(snap, theme=theme), di.draw_chart(snap, theme=theme)):
            assert {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", svg)} <= palette
            assert set(re.findall(r"rgba\((\d+),(\d+),(\d+),", svg)) <= {("141", "149", "166"), ("91", "98", "114")}
    for render in (_svg, _chart):
        dark, light = render(snap, "dark"), render(snap, "light")
        for role in ("point", "fine-print", "series-label", "point-label", "whisker", "table-cell", "panel-point",
                     "callo-line", "leader"):
            key = [e.get("data-text") or e.get("data-key") or e.get("data-arm") or e.get("data-for")
                   for e in _role(dark, role)]
            assert key == [e.get("data-text") or e.get("data-key") or e.get("data-arm") or e.get("data-for")
                           for e in _role(light, role)], role
        # Signal is Tiel's alone (the one accent)
        for e in dark.iter():
            if "#FF4F3A" in (e.get("stroke") or "") + (e.get("fill") or ""):
                arm = e.get("data-arm") or e.get("data-series") or e.get("data-for") or ""
                assert arm in ("", "tiel") or arm.startswith("tiel")


def test_every_character_drawn_is_in_the_brand_faces(snap):
    from fontTools.ttLib import TTFont

    import figkit
    sans = set(TTFont(os.path.join(figkit.HERE, "fonts", figkit.SANS)).getBestCmap())
    serif = set(TTFont(os.path.join(figkit.HERE, "fonts", figkit.SERIF_IT)).getBestCmap())
    text = "".join(e.get("data-text") or "" for root in (_svg(snap), _chart(snap)) for e in root.iter())
    missing = {c for c in text if ord(c) not in sans and ord(c) not in serif and c not in figkit.SYNTH}
    assert not missing


# ---- early exit: the line's early-exit point, its other arms as table rows, the paired comparisons -----------------

@pytest.fixture
def ee_snap(tmp_path) -> dict:
    return build(tmp_path / "ee", variants=True)


def test_the_line_ends_at_its_early_exit_point_and_its_other_arms_are_table_rows(ee_snap):
    by = {p["arm"]: p for p in ee_snap["points"]}
    for arm, cfg in fx.EE.items():
        p = by[arm]
        assert p["budget"] == 256 and p["series"] == "tiel"
        assert {k: p["early_exit"][k] for k in ("z", "gate")} == cfg
        assert p["label"] == "Tiel-Coder · think 256, " + ("early exit + gate" if cfg["gate"] else "early exit")
    assert by["tiel-b256-ee"]["variant_of"] is None                        # the line's own think point
    assert by["tiel-b256"]["variant_of"] == by["tiel-b256-eeg"]["variant_of"] == "tiel-b256-ee"
    assert by["tiel-b256"]["early_exit"] is None and by["tiel-b256"]["label"] == "Tiel-Coder · think 256"
    # the counters are the sample's: the session's less the warm-up snapshot
    eeg = by["tiel-b256-eeg"]["early_exit"]
    assert eeg["requests"] == 896 - 4 and eeg["gate_reads"]["asked"] == 1476 - 5
    assert eeg["gate_reads"]["skipped"] == 580 - 1 and "gate_reads" not in by["tiel-b256-ee"]["early_exit"]
    tiel = next(s for s in ee_snap["series"] if s["name"] == "tiel")
    assert tiel["arms"] == ["tiel-b0", "tiel-b256-ee"]                     # the line is two points
    assert tiel["rows"] == fx.CALLO_ROWS                                    # plain, early exit, + gate
    assert tiel["variants"] == list(fx.TABLE_ONLY)
    (c,) = ee_snap["paired"]
    assert c["reference"] == "tiel-b256" and [r["arm"] for r in c["comparisons"]] == list(fx.VARIANTS)
    for r in c["comparisons"]:
        assert r["d"] == list(fx.PAIRED_D[r["arm"]])
        assert r["lat_ratio"] == fx.VARIANTS[r["arm"]][1] / fx.ARMS["tiel-b256"][1]
    # the subtitle follows the line: its think point, with its opt-in early exit
    assert ee_snap["story"]["think"] == "tiel-b256-ee"
    assert ee_snap["story"]["subtitle"].endswith(
        "a 256-token think per question with opt-in early exit lifts it to 49.0, 5.7 under (CI 2.3–9.6), at 4.1× "
        "the time.")


def test_the_early_exit_line_is_drawn_like_any_line_and_its_other_arms_only_in_the_table(ee_snap, tmp_path):
    import figkit
    (tmp_path / "ee.json").write_text(json.dumps(ee_snap, ensure_ascii=False))
    snap = json.loads((tmp_path / "ee.json").read_text())
    for theme in ("dark", "light"):
        accent = figkit.THEMES[theme]["accent"]
        for root in (_svg(snap, theme), _chart(snap, theme)):
            # filled markers and a solid line, as the other two-point lines; no inset, nothing hollow
            (line,) = [e for e in _role(root, "callo-line") if e.get("data-series") == "tiel"]
            assert line.get("data-arms") == "tiel-b0,tiel-b256-ee" and line.get("stroke") == accent
            assert not line.get("stroke-dasharray")
            tiel = {g.get("data-arm"): g for g in _role(root, "point") if g.get("data-series") == "tiel"}
            assert set(tiel) == {"tiel-b0", "tiel-b256-ee"}
            for g in tiel.values():
                (shape,) = list(g)
                assert shape.get("fill") == accent == shape.get("stroke")
            lab = next(g for g in _role(root, "point-label") if g.get("data-arm") == "tiel-b256-ee")
            assert lab.get("data-text") == "think 256 · early exit · 49.0"
            assert not [e for e in root.iter() if (e.get("data-role") or "").startswith("inset")]
            # the table-only arms: no mark, whisker, tail, label or panel point
            assert not [e for e in root.iter() if e.get("data-arm") in fx.TABLE_ONLY and e.get("data-role") != "table-row"]
        # the plate's table keeps every arm, the drawn ones behind their mark, the others without one
        rows = {r.get("data-arm"): r for r in _role(_svg(snap, theme), "table-row")}
        assert [a for a in rows if a.startswith("tiel")] == fx.CALLO_ROWS
        for a in fx.CALLO_ROWS:
            drawn = a not in fx.TABLE_ONLY
            assert rows[a].get("data-drawn") == str(drawn).lower()
            assert bool([c for c in rows[a].iter("{http://www.w3.org/2000/svg}circle") if c.get("stroke") == accent]) \
                == drawn, a
    _check_layout(snap)
    test_the_callo_labels_hug_their_lines(snap)


def test_the_early_exit_fine_print_teaches_it_and_gives_the_paired_result(ee_snap):
    text = _fp(ee_snap)
    assert ("Early exit. Opt-in in Callo (--think-converge, z .95): Callo stops a think once the answer has settled "
            "instead of using the whole budget.") in text
    assert ("Tiel-Coder's line ends at think 256 with early exit: against plain think 256 on the same draws, index +0.2 "
            "[−2.0, +2.3] at 30 % less mean request time; the gate variant −0.8 [−2.9, +1.5] at 18 % less. Plain think "
            "256 and the gate variant are in the table, not drawn.") in text
    assert ("The gate first asks the model one zero-think question and sends multi-step rule problems to full "
            "thinking.") in text
    assert ("Callo 0.3.0 development builds: the first build (Tiel-Coder no think and think 256; Sharp-MiniCPM5-2B "
            "think 1024)") in text
    assert "the gate build (the early-exit points)" in text


def test_a_gated_line_point_names_the_gates_share_and_read(tmp_path):
    """A line point with early exit + gate (Flash-Next's think 256 on the release plate): the plate gives the gate's
    share of the sample's requests and its read time, and the point's label says early exit + gate."""
    s = fx.setup(tmp_path, variants=True)
    spec = s["spec"]
    tiel = next(x for x in spec["series"] if x["name"] == "tiel")
    tiel["budgets"]["256"] = "tiel-b256-eeg"                 # the gated arm as the line's think point
    tiel["variants"] = {"tiel-b256-ee": 256}
    tiel.pop("point_label_at")
    spec["notes"]["gate_read"] = "on long prompts"
    snap = di.build(di.load_spec(fx.rewrite_spec(s, series=spec["series"], paired=[], every_arm=False,
                                                 notes=spec["notes"])))
    text = _fp(snap)
    assert ("on Tiel-Coder it sent 579 of 1,471 requests (39 %), and its read took about 1.1 s at the median on long "
            "prompts.") in text
    root = _svg(snap)
    lab = next(g for g in _role(root, "point-label") if g.get("data-arm") == "tiel-b256-eeg")
    assert lab.get("data-text") in ("think 256 · early exit + gate · 48.1",       # one line where it fits
                                    "think 256 · 48.1 / early exit + gate")       # else two, narrower


@pytest.mark.parametrize("case", ["no-gate-flag", "plain-with-converge", "variant-without-ee", "ee-at-budget-0",
                                  "variant-at-0", "label-side", "label-arm"])
def test_early_exit_is_checked_against_the_run(tmp_path, case):
    s = fx.setup(tmp_path, variants=True)
    spec = s["spec"]
    tiel = next(x for x in spec["series"] if x["name"] == "tiel")
    if case == "no-gate-flag":
        meta_p = tmp_path / "di" / "tiel-b256-eeg" / "di_meta.json"
        meta = json.loads(meta_p.read_text())
        meta["sessions"][0]["cmd"].remove("--think-converge-gate")
        meta_p.write_text(json.dumps(meta))
        match = "gate False; the spec says z 0.95, gate True"
    elif case == "plain-with-converge":
        meta_p = tmp_path / "di" / "tiel-b256" / "di_meta.json"
        meta = json.loads(meta_p.read_text())
        meta["sessions"][0]["cmd"] += ["--think-converge", "0.95"]
        meta_p.write_text(json.dumps(meta))
        match = "plots it as a plain point"
    elif case == "variant-without-ee":                     # the line's point and a variant both plain: what varies?
        del tiel["early_exit"]["tiel-b256-ee"]
        match = "run the same early exit as the line's point"
    elif case == "ee-at-budget-0":
        tiel["early_exit"]["tiel-b0"] = {"z": 0.95, "gate": False}
        match = "no thinking to exit early from"
    elif case == "variant-at-0":
        tiel["variants"]["tiel-b256"] = 0
        match = "not the line's think budget"
    elif case == "label-side":
        tiel["point_label_at"] = {"tiel-b256-ee": "up"}
        match = "point_label_at is"
    else:
        tiel["point_label_at"] = {"clef27b": "nw"}
        match = "point_label_at names"
    with pytest.raises(di.Refusal, match=match):
        di.build(di.load_spec(fx.rewrite_spec(s, series=spec["series"])))


# ---- the plain words, the README's chart -----------------------------------------------------------------------------

def test_no_drawn_word_says_wall_or_the_kits_and_the_x_axis_says_what_it_is(snap, ee_snap):
    import figures_md
    for s_ in (snap, ee_snap):
        for theme in ("dark", "light"):
            for root in (_svg(s_, theme), _chart(s_, theme)):
                shown = [e.get("data-text") for e in root.iter() if e.get("data-text")] + [root.get("aria-label")]
                assert not [t for t in shown if re.search(r"\bwall\b", t, re.I) or "the kit's" in t.lower()]
                axis = [e.get("data-text") for e in _role(root, "axis-title")]
                assert axis[0] == di.X_TITLE == "Mean request completion time (log scale)"
        section = figures_md.render(None, None, s_).split("## The Decision Index plate")[1].split("## Development")[0]
        assert not re.search(r"\bwall\b", section, re.I) and "the kit's" not in section


# The product's name in prose is "Callo"; a lowercase "callo" is only the command: in FIGURES.md a code span (`callo`
# 0.3.0, `callo compare`), on a plate the command's own words (callo compare).
PROSE_CALLO = re.compile(r"(?<![\w/`.-])callo(?![\w/`-])(?!\.\w)(?! (?:compare|serve|calibrate|classify)\b)")


def _prose_callo(snap: dict) -> list[str]:
    """Every drawn text (plate and chart, both themes) and every line of FIGURES.md's DI section, code spans aside,
    that names the product in lowercase."""
    import figures_md
    out = []
    for theme in ("dark", "light"):
        for root in (_svg(snap, theme), _chart(snap, theme)):
            shown = [e.get("data-text") for e in root.iter() if e.get("data-text")] + [root.get("aria-label")]
            out += [t for t in shown if PROSE_CALLO.search(t)]
    section = figures_md.render(None, None, snap).split("## The Decision Index plate")[1].split("## Development")[0]
    return out + [ln for ln in section.splitlines() if PROSE_CALLO.search(re.sub(r"`[^`]*`", "", ln))]


def test_the_product_is_callo_in_prose_and_callo_in_code(snap, ee_snap):
    for s_ in (snap, ee_snap):
        assert _prose_callo(s_) == []
    text = _fp(ee_snap)
    assert "Opt-in in Callo (--think-converge, z .95): Callo stops a think" in text
    assert "llama.cpp b50-test (Callo) and" in text and "Callo answers a request's questions" in text
    for theme in ("dark", "light"):
        labels = [g.get("data-text") for g in _role(_svg(ee_snap, theme), "series-label")]
        assert "Tiel-Coder 35B-A3B · Q4_K_XL / Callo · llama.cpp" in labels, labels
    import figures_md
    md = figures_md.render(None, None, ee_snap)
    assert "; `callo` 0.3.0, the first build, --think-budget 0," in md


def test_the_chart_is_the_plot_alone_under_one_title(ee_snap):
    for theme in ("dark", "light"):
        root = _chart(ee_snap, theme)
        assert [e.get("data-text") for e in _role(root, "title")] == ["Decision Index"]
        assert root.get("aria-label") == "Decision Index"
        (legend,) = _role(root, "legend")
        assert legend.get("data-text") == "mean / p95 / paired 95 % CI vs Clef-Flash BF16"
        assert [e.get("data-text") for e in _role(root, "axis-title")] == [
            "Mean request completion time (log scale)", "index, public subset  ↑"]
        for role in ("subtitle", "badge", "section", "section-note", "panel", "panel-point", "table", "table-row",
                     "fine-print"):
            assert not _role(root, role), role
        drawn = {p["arm"] for p in _drawn(ee_snap)}
        assert {g.get("data-arm") for g in _role(root, "point")} == drawn
        assert {e.get("data-arm") for e in _role(root, "whisker")} == drawn - {fx.REF}
        assert {e.get("data-arm") for e in _role(root, "tail")} == drawn
        labelled = {s for g in _role(root, "series-label") for s in g.get("data-series").split(",")}
        assert labelled == {s["name"] for s in ee_snap["series"] if s["status"] == "present" and s["on_plate"]}
        assert {g.get("data-arm") for g in _role(root, "point-label")} == {
            p["arm"] for p in _drawn(ee_snap) if p["kind"] == "callo"}
    # the plate's plot, point for point, relative to the plot box
    def rel(root: ET.Element) -> dict:
        x0, y0 = (float(v) for v in root.get("data-plot-box").split(",")[:2])
        return {g.get("data-arm"): (round(float(g.get("data-x")) - x0, 6), round(float(g.get("data-y")) - y0, 6))
                for g in _role(root, "point")}
    assert rel(_chart(ee_snap)) == rel(_svg(ee_snap))
    test_every_colour_is_from_the_brand_palette_and_both_themes_carry_the_same_information(ee_snap)


def test_draw_writes_the_plate_and_the_chart(ee_snap, tmp_path):
    (tmp_path / "d.json").write_text(json.dumps(ee_snap, ensure_ascii=False))
    assert di.main(["draw", "--data", str(tmp_path / "d.json"), "--out-dir", str(tmp_path)]) == 0
    for stem in ("di", "di-chart"):
        for theme in ("dark", "light"):
            assert ET.parse(tmp_path / f"{stem}-{theme}.svg").getroot().get("data-theme") == theme


@pytest.mark.parametrize("case", ["seed", "derived", "reference"])
def test_a_paired_bootstrap_must_pair_with_the_plotted_one(tmp_path, case):
    s = fx.setup(tmp_path, variants=True)
    doc = json.loads((tmp_path / "paired.json").read_text())
    if case == "seed":
        doc["seed"] += 1
        match = "do not pair"
    elif case == "derived":
        doc["inputs"]["tiel-b256-ee"]["derived_sha256"] = "0" * 64
        match = "not scored from the plotted bootstrap's results"
    else:
        doc["reference"] = "nobody"
        match = "not a measured arm"
    (tmp_path / "paired.json").write_text(json.dumps(doc))
    with pytest.raises(di.Refusal, match=match):
        di.build(di.load_spec(s["spec_path"]))


# ---- re-read records -----------------------------------------------------------------------------------------------

def _reread(tmp: Path, arm: str = "tiel-b0", *, patched_sha: str | None = None, orig: str = "b463e5fd94f4",
            status_match: int = 40, bit: int = 12) -> None:
    """Make the bootstrap score `arm` from re-read results and write di_patch.py's records for them."""
    d = tmp / "di" / arm
    (d / "results.patched.jsonl").write_text(json.dumps({"run_id": "x", "status": "ok", "patched": 1}) + "\n")
    boot = json.loads((tmp / "boot-a.json").read_text())
    boot["inputs"][arm].update(results=f"di/{arm}/results.patched.jsonl",
                               results_sha256=fx._sha(d / "results.patched.jsonl"))
    (tmp / "boot-a.json").write_text(json.dumps(boot))
    (d / "patch").mkdir(exist_ok=True)
    (d / "patch" / "apply.json").write_text(json.dumps({
        "arm": arm, "patched_sha256": patched_sha or boot["inputs"][arm]["results_sha256"], "original_sha256": "o" * 64,
        "callo_commit": "a3a88c3cf52f", "walked": 66, "nondeterministic": True, "not_ok_after_patch": [],
        "argmax_changed_rows": 6, "control_summary": {"n": 40, "status_match": status_match, "bit_identical": bit}}))
    (d / "patch" / "patch_meta.json").write_text(json.dumps({"arm": arm, "original_results_sha256": "o" * 64,
                                                              "original_callo_commit": orig}))


def test_a_reread_arm_needs_its_record_and_the_plate_says_so(tmp_path):
    s = fx.setup(tmp_path)
    _reread(tmp_path)
    with pytest.raises(di.Refusal, match="names no re-read record"):
        di.build(di.load_spec(s["spec_path"]))
    notes = {**s["spec"]["notes"], "reread_fix": "an end-of-turn token at the answer slot",
             "run_to_run": "Tiel-Coder is an MoE model; its runs are not bit-reproducible"}
    path = fx.rewrite_spec(s, patches={"tiel-b0": "di/tiel-b0/patch/apply.json"}, notes=notes)
    snap = di.build(di.load_spec(path))
    rr = next(p for p in snap["points"] if p["arm"] == "tiel-b0")["provenance"]["reread"]
    assert rr["walked"] == 66 and rr["callo_commit"] == "a3a88c3" and (rr["control"], rr["bit_identical"]) == (40, 12)
    text = _fp(snap)
    assert ("Re-read. Tiel-Coder no think ran before a readout fix (an end-of-turn token at the answer slot); their 66 "
            "rows with an affected readout were re-read on the readout-fix build, and 40 unaffected rows per arm re-run as "
            "controls matched in status.") in text
    assert ("Run to run. Tiel-Coder is an MoE model; its runs are not bit-reproducible (Tiel-Coder's re-read controls: "
            "12 of 40 rows bit for bit).") in text
    import figures_md
    md = figures_md.render(None, None, snap)
    assert "re-read after the readout fix: 66 rows on the readout-fix build; controls 40 of 40 in status, 12 bit for bit" in md
    assert "| the readout-fix build | the readout fix: " in md and "| `a3a88c3` |" in md     # the commit, once, as provenance
    assert md.count("a3a88c3") == 1


@pytest.mark.parametrize(("kw", "match"), [({"patched_sha": "f" * 64}, "not the results"),
                                           ({"orig": "9d52aaf000"}, "patched a run at callo 9d52aaf"),
                                           ({"status_match": 39}, "controls did not match")])
def test_a_reread_record_that_does_not_match_is_refused(tmp_path, kw, match):
    s = fx.setup(tmp_path)
    _reread(tmp_path, **kw)
    with pytest.raises(di.Refusal, match=match):
        di.build(di.load_spec(fx.rewrite_spec(s, patches={"tiel-b0": "di/tiel-b0/patch/apply.json"})))


# ---- the fixed index axis -------------------------------------------------------------------------------------------

def test_a_fixed_y_range_is_the_axis_and_never_clips(tmp_path):
    s = fx.setup(tmp_path)
    snap = di.build(di.load_spec(fx.rewrite_spec(s, y_range=[0, 70])))
    assert snap["y_range"] == [0.0, 70.0]
    root = _svg(snap)
    ticks = [e.get("data-text") for e in _role(root, "tick") if e.get("data-text").isdigit()]
    assert ticks[:8] == [str(v) for v in range(0, 80, 10)]
    snap = di.build(di.load_spec(fx.rewrite_spec(s, y_range=[30, 70])))
    with pytest.raises(di.Refusal, match="clips"):
        di.draw(snap)
    with pytest.raises(di.Refusal, match="y_range"):
        di.build(di.load_spec(fx.rewrite_spec(s, y_range=[0, 75])))


# ---- real data: the release bake (skipped without the bench results) ------------------------------------------------

SPEC = Path(di.HERE) / "data" / "di.release.spec.json"
BENCH = Path(os.environ["CALLO_BENCH_ROOT"]) if os.environ.get("CALLO_BENCH_ROOT") else None    # the benchmark's root
RELEASE = BENCH / "harness/results/callo_v0.3/di/bootstrap-release3.json" if BENCH else None
# the 11 arms of the 2026-10-13 bake (bootstrap-release1.json) and Liquid AI's d1-3B (2026-10-14), on the same draws
RELEASE1 = BENCH / "harness/results/callo_v0.3/di/bootstrap-release1.json" if BENCH else None
ALL_ARMS = {"tiel-b0", "tiel-b256", "tiel-b256-ee", "tiel-b256-eeg", "sharp-minicpm-b0", "sharp-minicpm-b1024",
            "flashnext-sharp-b0", "flashnext-sharp-b256-eeg", "clef27b-udq4", "clefflash-udq4", "clefflash-bf16",
            "d1-3b-q4km"}
# Laya (2026-10-26) in its own bootstrap on the same draws: its rows over its window count wrong for it only
PLATE_ARMS = ALL_ARMS | {"laya"}
LAYA_BOOT = BENCH / "harness/results/callo_v0.3/di/bootstrap-laya.json" if BENCH else None


@pytest.fixture(scope="module")
def real() -> dict:
    if RELEASE is None or not RELEASE.is_file():
        pytest.skip(f"the DI outputs are not under {BENCH}" if BENCH else "CALLO_BENCH_ROOT (the benchmark's root) is not set")
    return di.build(di.load_spec(SPEC, root=BENCH))


def test_real_release_carries_every_arms_numbers_from_the_bootstrap(real):
    boot = json.loads(RELEASE.read_text(encoding="utf-8"))
    by = {p["arm"]: p for p in real["points"]}
    assert set(by) - {"laya"} == set(boot["arms"]) == ALL_ARMS and set(by) == PLATE_ARMS
    for arm, a in boot["arms"].items():
        p = by[arm]
        assert p["index"] == a["index"] and p["index_ci"] == a["index_ci"]
        assert p["lat"] == {"mean": a["latency_ms"]["mean"], "p95": a["latency_ms"]["p95"],
                            "mean_ci": a["latency_ms"]["mean_ci"], "n": 1467}
        for k, d in (("dref", "delta_vs_reference"), ("dsecond", "delta_vs_second_reference")):
            assert p[k] is None or p[k] == [a[d]["point"], *a[d]["ci"]]
    assert by["tiel-b0"]["index"] == 38.48 and by["flashnext-sharp-b256-eeg"]["dref"] == [-3.11, -6.9105, 0.571]
    assert by["clefflash-bf16"]["dref"] is None and by["clef27b-udq4"]["dsecond"] is None
    assert real["unanswered"]["latency_rows"] == 1467 and not real["unanswered"]["errors"]
    assert real["unanswered"]["over_cap"]["n"] == 13 and real["unanswered"]["over_cap"]["benchmarks"] == ["BRIGHT"]
    assert [a for a, p in by.items() if not p["on_plate"]] == ["clefflash-udq4"]
    assert real["pending"] == [] and real["preliminary"] is False
    # provenance: the builds and the re-reads the fine print names
    commits = {a: p["provenance"].get("callo_commit") for a, p in by.items() if p["kind"] == "callo"}
    assert commits == {"tiel-b0": "b463e5f", "tiel-b256": "b463e5f", "sharp-minicpm-b1024": "b463e5f",
                       "sharp-minicpm-b0": "9d52aaf", "flashnext-sharp-b0": "6fc42a8", "tiel-b256-ee": "0d33846",
                       "tiel-b256-eeg": "0d33846", "flashnext-sharp-b256-eeg": "0d33846"}
    assert {a for a, p in by.items() if p["provenance"].get("reread")} == {"tiel-b0", "tiel-b256", "sharp-minicpm-b1024"}
    # the paired comparisons, as the bench wrote them
    comps = {r["arm"]: r for c in real["paired"] for r in c["comparisons"]}
    assert comps["tiel-b256-ee"]["d"] == [0.21, -1.9603, 2.3] and comps["tiel-b256-eeg"]["d"] == [-0.77, -2.9213, 1.51]
    assert comps["flashnext-sharp-b256-eeg"]["vs"] == "flashnext-sharp-b0"
    assert comps["flashnext-sharp-b256-eeg"]["d"] == [4.15, 1.7385, 6.8103]


def test_real_release_adds_d1_and_changes_no_other_arms_number(real):
    """Adding d1 re-ran the bootstrap on the same draws: every other arm's numbers are byte for byte the 11-arm bake's
    (bootstrap-release1.json), and d1 is a single marker of its own family on its own llama.cpp build."""
    if RELEASE1 is None or not RELEASE1.is_file():
        pytest.skip("bootstrap-release1.json is not under the benchmark's root")
    r1 = json.loads(RELEASE1.read_text(encoding="utf-8"))
    r3 = json.loads(RELEASE.read_text(encoding="utf-8"))
    assert set(r3["arms"]) - set(r1["arms"]) == {"d1-3b-q4km"}
    for arm in r1["arms"]:
        assert r3["arms"][arm] == r1["arms"][arm] and r3["inputs"][arm] == r1["inputs"][arm], arm
    for k in ("n", "seed", "reference", "second_reference", "latency_rows", "kit_commit"):
        assert r3[k] == r1[k], k
    d1 = next(p for p in real["points"] if p["arm"] == "d1-3b-q4km")
    assert d1["index"] == 46.07 and d1["index_ci"] == [43.1497, 49.3203]
    assert d1["dref"] == [-8.78, -12.0918, -5.1597] and d1["dsecond"] == [-11.93, -15.3042, -8.0092]
    assert d1["lat"] == {"mean": 1491.4, "p95": 6240.9, "mean_ci": [1384.9, 1595.2], "n": 1467}
    assert d1["on_plate"] and d1["kind"] == "clef" and d1["label"] == "d1-3B · Q4_K_M"
    pv = d1["provenance"]
    assert pv["build"] == "23deec02" and pv["model"] == "d1-3B-Q4_K_M.gguf" and pv["final_counts"] == {
        "unsupported": 13, "ok": 1467}
    assert pv["server"] == {"ctx": "16384", "batch": "16384", "ubatch": "16384", "parallel": "1"}
    assert pv["own_tokens"]["over_cap_question_max"] == [1198, 7044] and pv["own_tokens"]["n_ctx"] == 16384
    ser = next(s for s in real["series"] if s["name"] == "d1")
    assert (ser["family"], ser["marker"], ser["runtime"]) == ("d1", "triangle", "Liquid AI · llama.cpp native")
    for theme in ("dark", "light"):
        for root in (_svg(real, theme), _chart(real, theme)):
            (lab,) = [g for g in _role(root, "series-label") if g.get("data-series") == "d1"]
            assert lab.get("data-text") == "d1-3B · Q4_K_M · 46.1 / Liquid AI · llama.cpp native"


def test_real_laya_in_its_own_bootstrap_changes_no_other_arms_number(real):
    """Laya answers at most 512 tokens: 67 rows that do not fit count wrong for it only (its own union and bootstrap on
    the plate's draws). The plate's union, its 12 arms' numbers and their latency rows are unchanged; Laya's latency is
    over the 1,400 rows it answered."""
    if LAYA_BOOT is None or not LAYA_BOOT.is_file():
        pytest.skip("bootstrap-laya.json is not under the benchmark's root")
    boot = json.loads(RELEASE.read_text(encoding="utf-8"))
    own = json.loads(LAYA_BOOT.read_text(encoding="utf-8"))
    for ref in ("clefflash-bf16", "clef27b-udq4"):
        assert {k: v for k, v in own["arms"][ref].items() if k != "latency_ms"} == {
            k: v for k, v in boot["arms"][ref].items() if k != "latency_ms"}
    laya = next(p for p in real["points"] if p["arm"] == "laya")
    assert laya["index"] == 4.49 and laya["index_ci"] == [3.7295, 7.7317]
    assert laya["dref"] == [-50.36, -52.69, -45.9385] and laya["dsecond"] == [-53.51, -55.87, -48.8388]
    assert laya["lat"] == {"mean": 219.6, "p95": 1162.9, "mean_ci": [199.0, 238.2], "n": 1400}
    assert laya["on_plate"] and laya["kind"] == "clef" and laya["label"] == "Laya 421M · F16"
    pv = laya["provenance"]
    assert pv["build"] == "23deec02" and pv["model"] == "laya-7b928d8-F16.gguf"
    assert pv["final_counts"] == {"unsupported": 80, "ok": 1400} and pv["server"]["ctx"] == "512"
    ob = pv["own_bootstrap"]
    assert (ob["cut"], ob["misses"], ob["window_tokens"], ob["latency_rows"], ob["answered_by_others"]) == (
        256, 67, 512, 1400, 1467)
    assert ob["misses_by"] == [["CLINC150+OOS", 34], ["POP909-CL", 33]]
    assert real["unanswered"]["union_rows"] == 13 and real["unanswered"]["latency_rows"] == 1467
    assert real["sensitivity"]["arms"]["laya"]["without"] == 4.71
    leads = [f["lead"] for f in real["fine_print"]]
    assert leads[leads.index("d1-3B.") + 1] == "Laya."
    assert real["fine_print"][leads.index("Laya.")]["text"].startswith(
        "256 of the 1,467 rows the other arms answer were cut to its 512-token window")
    assert "67 rows that do not fit at all (34 CLINC150+OOS and 33 POP909-CL" in _fp(real)
    assert "(Laya's over the 1,400 rows it answered)" in _fp(real)
    assert real["fine_print"][leads.index("Unanswered.")]["text"].endswith(
        " The exception: rows that only Laya cannot answer count as wrong for Laya alone (below).")
    for theme in ("dark", "light"):
        for root in (_svg(real, theme), _chart(real, theme)):
            (lab,) = [g for g in _role(root, "series-label") if g.get("data-series") == "laya"]
            assert lab.get("data-text") == "Laya 421M · F16 · 4.5 / ModernBERT-large · llama.cpp native"
    _check_layout(real)


def test_real_laya_refused_on_the_plates_own_union(tmp_path):
    """An own bootstrap whose union is not the plate's union with the arm added is refused."""
    if LAYA_BOOT is None or not LAYA_BOOT.is_file():
        pytest.skip("bootstrap-laya.json is not under the benchmark's root")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for s_ in spec["series"]:
        if s_["name"] == "laya":
            s_["own_union"] = spec["union"]
    with pytest.raises(di.Refusal, match="not made from the union"):
        (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        di.build(di.load_spec(tmp_path / "spec.json", root=BENCH))


def test_real_release_axis_and_marking(real):
    assert real["y_range"] == [0.0, 70.0]
    for theme in ("dark", "light"):
        root = _svg(real, theme)
        ticks = [e.get("data-text") for e in _role(root, "tick")]
        assert [t for t in ticks if t in {str(v) for v in range(0, 80, 10)}][:8] == [str(v) for v in range(0, 80, 10)]
        assert not _role(root, "badge")
        shown = [e.get("data-text") for e in root.iter() if e.get("data-text")] + [root.get("aria-label")]
        assert not [t for t in shown if "UD-" in t or "PRELIMINARY" in t.upper()]
    words = [real["title"], real["story"]["subtitle"], _fp(real)] + [p["label"] for p in real["points"]]
    assert not [w for w in words if "UD-" in w or "preliminary" in w.lower()]


def test_real_flash_next_line_has_its_two_points(real):
    root = _svg(real)
    (line,) = [e for e in _role(root, "callo-line") if e.get("data-series") == "flashnext-sharp"]
    assert line.get("data-arms") == "flashnext-sharp-b0,flashnext-sharp-b256-eeg"
    drawn = {g.get("data-arm"): g for g in _role(root, "point") if g.get("data-series") == "flashnext-sharp"}
    assert set(drawn) == {"flashnext-sharp-b0", "flashnext-sharp-b256-eeg"}
    assert drawn["flashnext-sharp-b256-eeg"].get("data-early-exit") == "early exit + gate"
    lab = next(g for g in _role(root, "point-label") if g.get("data-arm") == "flashnext-sharp-b256-eeg")
    assert lab.get("data-text") == "think 256 · 51.7 / early exit + gate"     # one line does not fit beside it
    assert {g.get("data-arm") for g in _role(root, "point")} == PLATE_ARMS - {"clefflash-udq4", "tiel-b256",
                                                                             "tiel-b256-eeg"}


def test_real_tiel_line_ends_at_think_256_with_early_exit(real):
    """Tiel-Coder's plotted line is no think -> think 256 with early exit (z .95, no gate), drawn as the other lines;
    plain think 256 and early exit + gate are rows of the plate's table only; no inset."""
    import figkit
    for theme in ("dark", "light"):
        accent = figkit.THEMES[theme]["accent"]
        for root in (_svg(real, theme), _chart(real, theme)):
            (line,) = [e for e in _role(root, "callo-line") if e.get("data-series") == "tiel"]
            assert line.get("data-arms") == "tiel-b0,tiel-b256-ee" and line.get("stroke") == accent
            assert not line.get("stroke-dasharray")
            pts = {g.get("data-arm"): g for g in _role(root, "point") if g.get("data-series") == "tiel"}
            assert set(pts) == {"tiel-b0", "tiel-b256-ee"}
            assert all(list(g)[0].get("fill") == accent for g in pts.values())
            assert pts["tiel-b256-ee"].get("data-early-exit") == "early exit"
            lab = next(g for g in _role(root, "point-label") if g.get("data-arm") == "tiel-b256-ee")
            assert lab.get("data-text") == "think 256 · early exit · 49.0"
            assert not [e for e in root.iter() if (e.get("data-role") or "").startswith("inset")]
            assert not [e for e in root.iter() if e.get("data-arm") in ("tiel-b256", "tiel-b256-eeg")
                        and e.get("data-role") != "table-row"]
            _check_owned(root)
        rows = {r.get("data-arm"): r for r in _role(_svg(real, theme), "table-row")}
        assert [a for a in rows if a.startswith("tiel")] == ["tiel-b0", "tiel-b256", "tiel-b256-ee", "tiel-b256-eeg"]
        assert [a for a in rows if rows[a].get("data-drawn") == "false"] == ["tiel-b256", "tiel-b256-eeg"]
    _check_layout(real)


def test_real_release_story_and_fine_print(real):
    assert real["story"]["subtitle"] == (
        "Tiel-Coder with no thinking scores 38.5, 16.4 points under Clef-Flash BF16 (CI 12.7–20.4), at 0.93× its mean "
        "request time; a 256-token think per question with opt-in early exit lifts it to 49.0, 5.8 under (CI 2.6–9.5), "
        "at 4.1× the time.")
    text = _fp(real)
    for s_ in ("Decision Index 0.3 public subset: the 37 counted benchmarks (iSarcasmEval track A-En), kit 9eb2dbe2 and "
               "its unmodified scorer, suite sample --n 1480 --seed 20260919 (sha256 0ab3526d6a8e…)",
               "Rows over 16,384 Clef prompt tokens (n = 13) and any row not answered by every arm count as wrong for "
               "every arm (the index's no-truncation rule; all 13 are BRIGHT",
               "over the 1,467 rows answered by every arm",
               "(1,000 resamples of linked groups within each benchmark",
               "Early exit. Opt-in in Callo (--think-converge, z .95): Callo stops a think once the answer has settled "
               "instead of using the whole budget.",
               "Tiel-Coder's line ends at think 256 with early exit: against plain think 256 on the same draws, index "
               "+0.2 [−2.0, +2.3] at 30 % less mean request time; the gate variant −0.8 [−2.9, +1.5] at 18 % less. "
               "Plain think 256 and the gate variant are in the table, not drawn.",
               "sends multi-step rule problems to full thinking; on Flash-Next it sent 455 of 1,471 requests (31 %), and "
               "its read took about 1.1 s at the median on Flash-Next's long DI prompts.",
               "Paired vs Flash-Next no think: think 256 with early exit + gate +4.2 [+1.7, +6.8], at 3.2× the mean "
               "request time.",
               "Latency. Request completion time, measured by one HTTP client that is the same for every arm, over "
               "the 1,467 rows answered by every arm",
               "Timing. Clef 27B ran beside CPU-heavy test runs", "was not re-timed",
               "Re-read. Tiel-Coder (no think, think 256) and Sharp-MiniCPM5-2B think 1024 ran before a readout fix",
               "their 66, 43 and 147 rows with an affected readout were re-read on the readout-fix build",
               "bit for bit on Sharp-MiniCPM5-2B",
               "Run to run. Tiel-Coder and Flash-Next are MoE models", "not bit-reproducible",
               "Callo 0.3.0 development builds: the first build (Tiel-Coder no think and think 256; Sharp-MiniCPM5-2B "
               "think 1024), the refit build (Sharp-MiniCPM5-2B no think), the early-exit build (Flash-Next no think) "
               "and the gate build (the early-exit points); without early exit their behaviour is byte-identical",
               "llama.cpp b50-4ebdf2c (Callo), b11379 1537a0a8 (Clef) and 23deec02 (d1, Laya)",
               "run Callo 2026-10-10 to 2026-10-13, Clef 2026-10-10, d1 2026-10-14, Laya 2026-10-26 (UTC)",
               "d1-3B. Answers each question in its own prompt (the state, then that question), so its prompts on "
               "the 13 BRIGHT rows (7,044 tokens at most) would fit its 16,384-token context; they still count as "
               "wrong for every arm.",
               "Tiel-Coder budgets 512 and 1024 not run on DI", "Sharp-MiniCPM5-2B budgets 256 and 512 not run on DI",
               "Clef cross-check. The 0.3 board's public index for “Cloudflare clef” is 61.71 (full suite, authors' "
               "runtime); Clef's card reports 0.2.1 full-suite numbers.",
               "Without BRIGHT. Every arm +0.2 to +2.1 (the index's own re-weighting, the same draws); the order is "
               "unchanged and no paired CI changes whether it includes 0; Clef-Flash BF16 56.7.",
               "so its request time may read high."):
        assert s_ in text, s_
    for gone in ("GPQA", "Preliminary", "re-timed for the release bake", "Pending.", "pending", "§", "wall", "the kit's",
                 "inset"):
        assert gone not in text, gone
    for c in ("b463e5f", "9d52aaf", "a3a88c3", "6fc42a8", "0d33846"):            # builds by role, never by commit
        assert c not in text and c not in " ".join(f["text"] for f in real["made"]), c


def test_real_release_without_bright_and_figures_md_without_local_paths(real):
    import figures_md
    sens = real["sensitivity"]
    assert sens["shift"] == [pytest.approx(0.22), pytest.approx(2.12)] and sens["order_same"] is True   # Laya +0.22
    assert sens["flips"] == [] and sens["pairs"] == 24 and sens["scored_edition"] == "0.3-drop-36"
    assert sens["arms"]["d1-3b-q4km"]["without"] == 47.83                              # +1.76, inside the range
    assert sens["arms"]["clefflash-bf16"]["without"] == 56.71
    assert sens["arms"]["clef27b-udq4"]["dref_without"] == [3.41, -0.0312, 6.501]      # still across 0, only just
    assert "BRIGHT" not in real["title"] + real["story"]["subtitle"]
    md = figures_md.render(None, None, real)
    section = md[md.index("## The Decision Index"):]
    for s_ in ("bench-qwen36", "/Users/", "/Volumes/", "harness/results"):
        assert s_ not in section, s_
    assert "| Clef-Flash 9B · BF16 | 54.85 | 56.71 [53.88, 59.75] | reference | reference | −3.15 [−6.04, +0.16] | " \
           "−3.41 [−6.50, +0.03] |" in md
    assert "- bootstrap without BRIGHT: `bootstrap-release3-nobright.json` " in md


def test_real_chart_and_plain_words(real):
    """The README's chart from the release snapshot: one title, the key, every drawn arm, nothing else; no drawn word
    on the plate or the chart says "wall" or "the kit's"; FIGURES.md's DI prose neither (but for the bench bootstrap's
    own method line, quoted as written)."""
    import figures_md
    drawn = PLATE_ARMS - {"clefflash-udq4", "tiel-b256", "tiel-b256-eeg"}
    for theme in ("dark", "light"):
        root = _chart(real, theme)
        assert [e.get("data-text") for e in _role(root, "title")] == ["Decision Index"]
        assert {g.get("data-arm") for g in _role(root, "point")} == drawn
        for role in ("subtitle", "panel", "table", "fine-print"):
            assert not _role(root, role), role
        for r in (root, _svg(real, theme)):
            shown = [e.get("data-text") for e in r.iter() if e.get("data-text")] + [r.get("aria-label")]
            assert not [t for t in shown if re.search(r"\bwall\b", t, re.I) or "the kit's" in t.lower()]
            assert _role(r, "axis-title")[0].get("data-text") == "Mean request completion time (log scale)"
    md = figures_md.render(None, None, real)
    section = md[md.index("## The Decision Index"):md.index("## Development builds")]
    prose = "\n".join(ln for ln in section.splitlines() if not ln.startswith("- **Bootstrap.**"))
    assert not re.search(r"\bwall\b", prose, re.I) and "the kit's" not in prose
    assert "| Tiel-Coder · think 256 (table only) | 48.83 " in md
    assert "| Tiel-Coder · think 256, early exit | 49.04 " in md


def test_real_release_names_the_product_callo_in_prose(real):
    assert _prose_callo(real) == []
    runtimes = {s["name"]: s["runtime"] for s in real["series"] if s["kind"] == "callo"}
    assert runtimes == {"tiel": "Callo · llama.cpp", "sharp-minicpm": "Callo · llama.cpp",
                        "flashnext-sharp": "GSQ-RCO IQ3_XXS · Callo · llama.cpp"}


def test_the_committed_snapshot_is_the_real_build(real):
    committed = json.loads((Path(di.HERE) / "data" / "di.json").read_text(encoding="utf-8"))
    strip = ("generator", "made")
    assert {k: v for k, v in committed.items() if k not in strip} == \
           {k: v for k, v in json.loads(json.dumps(real, ensure_ascii=False)).items() if k not in strip}
