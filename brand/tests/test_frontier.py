"""The release plate (brand/frontier.py): the data layer reproduces callo's own statistics; the SVG carries the
information contract (points, frontiers, reference, labels with GiB, fine print) in both themes."""
from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import plate_fixtures as fx
import pytest

import frontier
from callo import compare, think

SVG = "{http://www.w3.org/2000/svg}"


def build(tmp: Path, **over) -> dict:
    s = fx.setup(tmp, **over)
    return frontier.build(frontier.load_spec(s["spec_path"]))


def by_label(snap: dict) -> dict[str, dict]:
    return {p["label"]: p for p in snap["points"]}


def _profiles(tmp: Path) -> list[dict]:
    return [{**compare.load_profile(tmp / f"{n}.think.json"), "model_name": n} for n in ("alpha", "beta")]


# ---- data layer --------------------------------------------------------------------------------------------------

def test_each_models_frontier_is_think_frontier_of_its_own_points(tmp_path):
    snap = build(tmp_path)
    pts = by_label(snap)
    for p in _profiles(tmp_path):
        own = compare.points_of(p, p["model_name"])
        for stat in frontier.STATS:
            want = {f.label for f in think.frontier(own, latency=stat) if f.dominated_by is None}
            got = {lab for lab, x in pts.items() if x["arm"] == p["model_name"] and x["frontier"][stat]}
            assert got == want
    assert pts["beta@256"]["frontier"]["mean"] is False                 # less accurate than beta@0 and slower
    assert pts["beta@256"]["dominated_by"]["mean"] == "beta@0"
    assert pts["alpha@256"]["frontier"]["mean"] is True


def test_the_dacc_cis_are_paired_bootstrap_over_the_fixed_rows(tmp_path):
    snap = build(tmp_path)
    profiles = _profiles(tmp_path)
    fixed, clusters, strata, _ = compare.pairing(profiles)
    assert fixed == fx.qids() and strata == fx.SOURCES
    ref = [float(c) for c in fx.REF_CORRECT]
    pts = by_label(snap)
    for p in profiles:
        rows = compare.rows_at(p)
        for pt in compare.points_of(p, p["model_name"]):
            x = [rows[q][(pt.kind, pt.value)][0] for q in fixed]
            want = list(think.paired_bootstrap(ref, x, clusters, seed=0, strata=strata))
            assert pts[pt.label]["dacc"] == pytest.approx(want, abs=0)
    assert pts["ref"]["dacc"] is None
    assert snap["ref"] == {"name": "ref", "label": "Ref (native)", "acc": sum(fx.REF_CORRECT) / fx.N}


def test_the_rows_arm_latency_is_latency_stats_over_fixed(tmp_path):
    snap = build(tmp_path)
    p = by_label(snap)["ref"]
    want = think.latency_stats(fx.REF_LAT, fx.CLUSTERS, seed=0, strata=fx.SOURCES)
    assert p["lat"]["mean"] == want["lat_mean_ms"]
    assert p["lat"]["p50"] == want["lat_p50_ms"]
    assert p["lat"]["p95"] == want["lat_p95_ms"]
    assert p["lat"]["p99"] == want["lat_p99_ms"]
    assert p["lat"]["max"] == want["lat_max_ms"]
    assert p["acc"] == sum(fx.REF_CORRECT) / fx.N and p["n"] == fx.N
    assert p["gib"] == 17.75


def test_the_callo_points_are_points_of_and_the_pick_is_compare_joint(tmp_path):
    snap = build(tmp_path)
    profiles = _profiles(tmp_path)
    pts = by_label(snap)
    for p in profiles:
        for pt in compare.points_of(p, p["model_name"]):
            got = pts[pt.label]
            assert got["acc"] == pt.acc and got["n"] == pt.n
            assert got["lat"]["mean"] == list(pt.lat_mean) and got["lat"]["max"] == pt.lat_max
            assert got["gib"] == pytest.approx(pt.gib)
    for stat in frontier.STATS:
        chosen = compare.joint(profiles, pref=think.Preference(latency=stat)).chosen.point
        assert snap["picks"][stat] == chosen.label
        assert stat in pts[chosen.label]["pick"]
    assert snap["flags"] == compare.flags(profiles, think.Preference())


def test_an_absent_arm_is_pending_not_an_error(tmp_path):
    snap = build(tmp_path)
    assert snap["pending"] == ["later"]
    assert [a["status"] for a in snap["arms"]] == ["present", "present", "present", "pending"]
    assert not any(p["arm"] == "later" for p in snap["points"])


def test_a_spec_with_only_callo_arms_builds(tmp_path):
    s = fx.setup(tmp_path)
    spec = {**s["spec"], "arms": s["spec"]["arms"][:2], "ref": None}
    (tmp_path / "only.json").write_text(json.dumps(spec))
    snap = frontier.build(frontier.load_spec(tmp_path / "only.json"))
    assert {p["arm"] for p in snap["points"]} == {"alpha", "beta"}
    assert all(p["dacc"] is None for p in snap["points"])
    assert snap["ref"]["acc"] is None


def test_a_spec_with_only_rows_arms_pairs_over_the_data_file(tmp_path):
    s = fx.setup(tmp_path)
    spec = {**s["spec"], "arms": s["spec"]["arms"][2:]}
    (tmp_path / "rows.json").write_text(json.dumps(spec))
    snap = frontier.build(frontier.load_spec(tmp_path / "rows.json"))
    assert [p["label"] for p in snap["points"]] == ["ref"]
    assert snap["strata"] == "source" and snap["data"]["n"] == fx.N


def _refuses(tmp_path, spec: dict, says: str):
    (tmp_path / "bad.json").write_text(json.dumps(spec))
    with pytest.raises(frontier.Refusal, match=says):
        frontier.build(frontier.load_spec(tmp_path / "bad.json"))


def test_refuses_a_rows_arm_missing_a_fixed_question(tmp_path):
    s = fx.setup(tmp_path)
    fx.write_rows(tmp_path / "ref", s["sha"], fx.REF_CORRECT, fx.REF_LAT, drop=3)
    _refuses(tmp_path, s["spec"], "1 of the 8 paired questions are missing")


def test_refuses_a_rows_arm_with_an_error_row(tmp_path):
    s = fx.setup(tmp_path)
    fx.write_rows(tmp_path / "ref", s["sha"], fx.REF_CORRECT, fx.REF_LAT, error_at=5)
    _refuses(tmp_path, s["spec"], "error row")


def test_refuses_a_rows_arm_on_different_data(tmp_path):
    s = fx.setup(tmp_path)
    fx.write_rows(tmp_path / "ref", "0" * 64, fx.REF_CORRECT, fx.REF_LAT)
    _refuses(tmp_path, s["spec"], "different data")


def test_refuses_a_profile_on_different_data(tmp_path):
    s = fx.setup(tmp_path)
    fx.write_profile(tmp_path, "beta", fx.profile("beta", "1" * 64, fx.BETA_CORRECT, fx.BETA_LAT))
    _refuses(tmp_path, s["spec"], "different data")


def test_refuses_a_duplicate_arm_name(tmp_path):
    s = fx.setup(tmp_path)
    arms = s["spec"]["arms"]
    _refuses(tmp_path, {**s["spec"], "arms": [*arms, {**arms[0], "label": "again"}]}, "two arms are named 'alpha'")


def test_refuses_a_profile_whose_cache_differs_from_the_label(tmp_path):
    s = fx.setup(tmp_path, alpha_cache=True)
    _refuses(tmp_path, s["spec"], "--cache on")


def test_refuses_a_rows_arm_without_gib(tmp_path):
    s = fx.setup(tmp_path)
    fx.write_rows(tmp_path / "ref", s["sha"], fx.REF_CORRECT, fx.REF_LAT, gib=None)
    _refuses(tmp_path, s["spec"], "GiB unknown")


def test_the_cli_exits_2_on_a_refusal(tmp_path, capsys):
    s = fx.setup(tmp_path)
    arms = s["spec"]["arms"]
    (tmp_path / "bad.json").write_text(json.dumps({**s["spec"], "arms": [*arms, arms[1]]}))
    assert frontier.main(["build", "--spec", str(tmp_path / "bad.json"), "--out", str(tmp_path / "o.json")]) == 2
    assert "two arms" in capsys.readouterr().err


def _fp(snap: dict, stat: str = "mean") -> str:
    return " ".join((f["lead"] or "") + " " + f["text"] for f in snap["fine_print"][stat])


def test_the_fine_print_states_n_sources_timers_cis_pick_and_pending(tmp_path):
    snap = build(tmp_path)
    text = _fp(snap)
    assert "n = 8 test-s1 (beta 4, alpha 4); idle M2 Ultra 64 GB, one at a time; Callo 2026-10-08; Ref 2026-10-02." in text
    assert "Timers. Callo arms: run1, --cache off (pre-release default); in-Callo frame + read wall time." in text
    assert "Ref (2026-10-02). Client wall per request (kev.benchmark RemotePredictor" in text
    assert "against the release code (transformers, MPS)" in text
    assert "server-side" not in text
    assert "Paired 95 % cluster-bootstrap CIs vs Ref (native), stratified by source" in text
    assert "Pending." in text and "Later 27B" in text
    assert "wasted LM head" not in text


@pytest.mark.parametrize("stat", ["mean", "p50", "p95"])
def test_the_pick_definition_and_the_budget_replay_are_on_every_variant(tmp_path, stat):
    snap = build(tmp_path)
    text = _fp(snap, stat)
    assert (f"Callo's pick: the fastest point statistically tied with the most accurate (callo compare default, "
            f"on {stat}).") in text
    assert "Budgets are read from one generation per item, cut at each budget" in text
    assert frontier.BUDGET_REPLAY in text
    assert f"tails run to each point's {frontier.TAIL_OF[stat]}" in text


def test_the_compare_flags_and_arm_notes_are_in_figures_md_and_the_plate_names_the_builds(tmp_path):
    snap = build(tmp_path)
    snap["flags"] = ["different user delimiters: a <x>\nuser, b <y>\nuser",
                     "different server args: a --chat-template-file /Volumes/x/models/t.jinja"]
    for stat in frontier.STATS:
        text = _fp(snap, stat)
        assert "`callo compare` flags" not in text and "different user delimiters" not in text
        assert frontier.GIB_LINE not in text and "Ref: the release code" not in text
        assert "Builds. Callo 0.2.0; llama.cpp b50-4ebdf2c (Callo)." in text
        assert "GiB. " + frontier.GIB_TEXT + "." in text
    made = " ".join((f["lead"] or "") + " " + f["text"] for f in frontier.made(snap))
    assert "`callo compare` flags. different user delimiters" in made
    assert "--chat-template-file t.jinja" in made and "/Volumes" not in made        # a local path: its file name
    assert "llama.cpp b50-4ebdf2c for Alpha, Beta" in made and frontier.GIB_LINE in made
    assert "Ref: the release code, bf16, MPS." in made and "seed 0" in made
    assert "Alpha 35B · UD-Q4_K_XL. `alpha.gguf`; budgets 0, 256; `callo` 0.2.0 commit not pinned" in made
    import figures_md
    md = figures_md.render(snap, None)
    assert "# How these figures were made" in md and "llama.cpp b50-4ebdf2c for Alpha, Beta" in md
    assert "| Alpha 35B · UD-Q4_K_XL · think 256 |" in md                    # every point, in a table
    assert f"- Alpha 35B · UD-Q4_K_XL: `alpha.think.json` {frontier.sha256_file(tmp_path / 'alpha.think.json')[:16]}…" in md
    assert str(tmp_path) not in md                                            # inputs by file name and sha256


def test_the_cache_on_label_is_the_release_wording(tmp_path):
    s = fx.setup(tmp_path)
    for n, prof in (("alpha", fx.profile("alpha", s["sha"], fx.ALPHA_CORRECT, fx.ALPHA_LAT, cache=True)),
                    ("beta", fx.profile("beta", s["sha"], fx.BETA_CORRECT, fx.BETA_LAT, cache=True))):
        fx.write_profile(tmp_path, n, prof)
    spec = {**s["spec"], "callo_cache": True, "callo_run": None, "draft": False}
    (tmp_path / "rel.json").write_text(json.dumps(spec))
    snap = frontier.build(frontier.load_spec(tmp_path / "rel.json"))
    text = _fp(snap)
    assert "Callo arms: --cache on (default)" in text
    assert "Pending" not in text


@pytest.mark.parametrize("stat", ["mean", "p50", "p95"])
def test_the_story_is_computed_from_the_snapshot_on_each_statistic(tmp_path, stat):
    snap = build(tmp_path)
    st = snap["story"][stat]
    pts = by_label(snap)
    zero, ref, b256 = pts["alpha@0"], pts["ref"], pts["alpha@256"]
    ratio = zero["lat"][stat][0] / ref["lat"][stat][0]
    assert st["ratio"] == ratio and st["dacc"] == zero["dacc"]
    d, lo, hi = zero["dacc"]
    assert st["ties"] == (lo <= 0 <= hi)
    assert f"at {ratio:.2f}× its {stat} latency, vs Ref's release code (transformers, MPS)" in st["subtitle"]
    assert f"~{round(b256['think_mean'], -1):.0f} thinking tokens put it {frontier._pp(b256['dacc'][0])} pp " in st["subtitle"]
    side = "above" if b256["dacc"][0] >= 0 else "below"
    assert f" pp {side} Ref at " in st["subtitle"]                  # against the reference, not against no think
    assert st["subtitle"].endswith(f"{stat}.")
    root = _svg(snap, stat=stat)
    (sub,) = _role(root, "subtitle")
    assert sub.get("data-text") == st["subtitle"]
    assert not _role(root, "annotation")                 # the subtitle tells it; the plot stays clear for the data


# ---- the SVG information contract --------------------------------------------------------------------------------

def _svg(snap: dict, theme: str = "dark", stat: str = "mean") -> ET.Element:
    return ET.fromstring(frontier.draw(snap, theme=theme, stat=stat))


def _role(root: ET.Element, role: str) -> list[ET.Element]:
    return [e for e in root.iter() if e.get("data-role") == role]


def _plot_box(root: ET.Element) -> tuple[float, ...]:
    return tuple(float(v) for v in root.get("data-plot-box").split(","))


@pytest.fixture
def snap(tmp_path) -> dict:
    return build(tmp_path)


def test_every_point_is_drawn_with_data_that_round_trips(snap):
    root = _svg(snap)
    drawn = {g.get("data-label"): g for g in _role(root, "point")}
    assert set(drawn) == {p["label"] for p in snap["points"]}
    for p in snap["points"]:
        g = drawn[p["label"]]
        assert float(g.get("data-acc")) == p["acc"]
        assert float(g.get("data-lat-ms")) == p["lat"]["mean"][0]
        assert g.get("data-arm") == p["arm"]
        if p["dacc"] is None:
            assert g.get("data-dacc") == ""
        else:
            assert [float(v) for v in g.get("data-dacc").split(",")] == p["dacc"]
        assert g.get("data-frontier") == ("1" if p["frontier"]["mean"] else "0")


@pytest.mark.parametrize("stat", ["mean", "p50", "p95"])
def test_pixels_are_monotone_in_log_latency_and_in_accuracy(snap, stat):
    root = _svg(snap, stat=stat)
    pts = [(float(g.get("data-lat-ms")), float(g.get("data-acc")), float(g.get("data-x")), float(g.get("data-y")))
           for g in _role(root, "point")]
    by_p = {p["label"]: p for p in snap["points"]}
    for g in _role(root, "point"):
        assert float(g.get("data-lat-ms")) == by_p[g.get("data-label")]["lat"][stat][0]
    for a in pts:
        for b in pts:
            if a[0] < b[0]:
                assert a[2] < b[2]
            if a[1] > b[1]:
                assert a[3] < b[3]                      # more accurate is higher (smaller y)
    # log: equal ratios are equal pixel steps
    (l1, _, x1, _), (l2, _, x2, _) = sorted(pts)[0], sorted(pts)[-1]
    k = (x2 - x1) / (math.log10(l2) - math.log10(l1))
    for lat, _, x, _ in pts:
        assert x == pytest.approx(x1 + k * (math.log10(lat) - math.log10(l1)), abs=0.05)


def test_frontier_polylines_run_in_latency_order_through_frontier_points(snap):
    root = _svg(snap)
    lines = {e.get("data-arm"): e for e in _role(root, "frontier")}
    assert set(lines) == {"alpha"}                     # beta has one frontier point: a marker, no line
    for arm, e in lines.items():
        xs = [float(p.split(",")[0]) for p in e.get("points").split()]
        assert xs == sorted(xs)
        assert e.get("data-labels").split(",") == [
            p["label"] for p in sorted(snap["points"], key=lambda p: p["lat"]["mean"][0])
            if p["arm"] == arm and p["frontier"]["mean"]]


def test_the_reference_line_sits_at_the_reference_accuracy(snap):
    root = _svg(snap)
    (ref,) = _role(root, "ref")
    assert float(ref.get("data-acc")) == snap["ref"]["acc"]
    y = float(ref.get("y1"))
    assert y == float(ref.get("y2"))
    pts = {g.get("data-label"): g for g in _role(root, "point")}
    assert y == pytest.approx(float(pts["ref"].get("data-y")), abs=0.01)


def test_every_series_is_labelled_on_the_plot_and_its_runtime_and_gib_are_in_the_table(snap):
    root = _svg(snap)
    labels = {g.get("data-arm"): g.get("data-text") for g in _role(root, "arm-label")}
    present = [a for a in snap["arms"] if a["status"] == "present"]
    assert labels == {a["name"]: a["label"] for a in present}
    rows = {g.get("data-arm"): g for g in _role(root, "table-row")}
    assert set(rows) == {a["name"] for a in present}
    for a in present:
        texts = [e.get("data-text") for e in rows[a["name"]].iter() if e.get("data-text")]
        assert a["label"] in texts and f"{a['gib']:.1f}" in texts
        assert any(a["runtime"] in t_ for t_ in texts)


def test_the_table_holds_every_plotted_point_in_its_budget_column(snap):
    root = _svg(snap)
    cells = {g.get("data-label"): g for g in _role(root, "table-cell") if g.get("data-label")}
    assert set(cells) == {p["label"] for p in snap["points"]}
    for p in snap["points"]:
        assert float(cells[p["label"]].get("data-acc")) == p["acc"]
        assert float(cells[p["label"]].get("data-lat-ms")) == p["lat"]["mean"][0]
    heads = [g.get("data-text") for g in _role(root, "table-head")]
    assert heads == ["MODEL", "GIB", "NO THINK", "THINK 256"]
    (ref_row,) = [g for g in _role(root, "table-row") if g.get("data-arm") == "ref"]
    assert any("reference" in (e.get("data-text") or "") for e in ref_row.iter())


def test_the_legend_names_the_statistic_its_tail_the_whisker_and_the_budget_numbers(snap):
    for stat in frontier.STATS:
        (leg,) = _role(_svg(snap, stat=stat), "legend")
        assert leg.get("data-text") == (f"{stat} / {frontier.TAIL_OF[stat]} / paired 95 % CI vs Ref (native) / "
                                        "think budget, tokens")


def test_each_callo_point_but_the_pick_is_labelled_with_its_budget(snap):
    root = _svg(snap)
    got = {g.get("data-label"): g.get("data-text") for g in _role(root, "point-label")}
    want = {p["label"]: str(p["budget"]) for p in snap["points"]
            if p["arm"] in ("alpha", "beta") and p["label"] != snap["picks"]["mean"]}
    assert got == want


def test_a_single_point_callo_arm_is_named_beside_its_own_marker(tmp_path):
    s = fx.setup(tmp_path)
    fx.write_profile(tmp_path, "gamma", fx.profile("gamma", s["sha"], {0: fx.BETA_CORRECT[0]}, {0: fx.BETA_LAT[0]},
                                                   gib=4.8))
    arm = {"name": "gamma", "label": "Gamma E4B · UD-Q4_K_XL", "kind": "callo", "profile": "gamma.think.json",
           "runtime": "Callo · llama.cpp", "marker": "triangle-hollow"}
    (tmp_path / "g.json").write_text(json.dumps({**s["spec"], "arms": [*s["spec"]["arms"], arm]}))
    snap = frontier.build(frontier.load_spec(tmp_path / "g.json"))
    root = _svg(snap)
    (lab,) = [g for g in _role(root, "arm-label") if g.get("data-arm") == "gamma"]
    assert lab.get("data-text") == "Gamma E4B · UD-Q4_K_XL" and lab.get("data-side") == "marker"
    assert not [g for g in _role(root, "point-label") if g.get("data-label") == "gamma@0"]
    (pt,) = [g for g in _role(root, "point") if g.get("data-label") == "gamma@0"]
    assert [c.tag.split("}")[1] for c in pt] == ["path"]                  # a triangle (its halo drawn underneath)
    assert not [e for e in _role(root, "frontier") if e.get("data-arm") == "gamma"]
    bx = [float(v) for v in lab.get("data-box").split(",")]
    assert bx[0] - 60 <= float(pt.get("data-x")) <= bx[2] + 60
    (row,) = [g for g in _role(root, "table-row") if g.get("data-arm") == "gamma"]
    assert [e.get("data-text") for e in row.iter() if e.get("data-role") == "table-empty"] == ["–"]


def test_the_whiskers_map_the_paired_ci_onto_accuracy(snap):
    root = _svg(snap)
    pts = {g.get("data-label"): g for g in _role(root, "point")}
    whiskers = {e.get("data-label"): e for e in _role(root, "whisker")}
    by_p = {p["label"]: p for p in snap["points"]}
    assert set(whiskers) == {lab for lab, p in by_p.items() if p["dacc"] is not None}
    for lab, e in whiskers.items():
        d, lo, hi = by_p[lab]["dacc"]
        acc = by_p[lab]["acc"]
        assert float(e.get("data-lo")) == pytest.approx(acc + (lo - d))
        assert float(e.get("data-hi")) == pytest.approx(acc + (hi - d))
        assert float(e.get("x1")) == pytest.approx(float(pts[lab].get("data-x")))


@pytest.mark.parametrize("stat", ["mean", "p50", "p95"])
def test_every_fine_print_string_is_on_the_plate(snap, stat):
    root = _svg(snap, stat=stat)
    texts = [g.get("data-text") for g in _role(root, "fine-print")]
    assert len(texts) == len(snap["fine_print"][stat])
    for f in snap["fine_print"][stat]:
        assert any(f["text"] in t and (f["lead"] or "") in t for t in texts), f["text"]


@pytest.mark.parametrize("stat", ["mean", "p50", "p95"])
def test_the_draft_is_marked_and_the_pick_has_a_callout_with_its_own_numbers(snap, stat):
    root = _svg(snap, stat=stat)
    assert any("DRAFT" in (g.get("data-text") or "") for g in _role(root, "badge"))
    (pick,) = _role(root, "pick")
    assert pick.get("data-label") == snap["picks"][stat]
    (call,) = _role(root, "pick-label")
    p = by_label(snap)[snap["picks"][stat]]
    second = "p95" if stat != "p95" else "mean"
    budget = f"think {p['budget']}" if p["budget"] else "no think"
    assert call.get("data-text") == (f"Callo's pick · {budget} · {100 * p['acc']:.1f} % / "
                                     f"{frontier.secs(p['lat'][stat][0])} {stat} · {second} "
                                     f"{frontier.secs(p['lat'][second][0])}")
    assert not [g for g in _role(root, "point-label") if g.get("data-label") == p["label"]]


def test_every_marker_sits_on_a_ground_halo_drawn_under_every_marker(snap):
    from figkit import THEMES
    for theme in ("dark", "light"):
        root = _svg(snap, theme=theme)
        (halos,) = _role(root, "halos")
        pts = _role(root, "point")
        assert len(list(halos)) == len(pts)
        for halo, g in zip(halos, pts, strict=True):
            (mark,) = list(g)
            assert halo.get("stroke") == THEMES[theme]["ground"] == halo.get("fill")
            assert float(halo.get("stroke-width")) > float(mark.get("stroke-width"))
            assert (halo.get("cx"), halo.get("d")) == (mark.get("cx"), mark.get("d"))


def test_equal_tail_ends_draw_one_tail(snap):
    pts = by_label(snap)
    pts["alpha@0"]["lat"]["p95"] = list(pts["alpha@256"]["lat"]["p95"])
    root = _svg(snap)
    tails = [e.get("data-label") for e in _role(root, "tail")]
    assert tails.count("alpha@0") + tails.count("alpha@256") == 1
    shared = [g for g in _role(root, "point") if g.get("data-tail-shared")]
    assert [g.get("data-label") for g in shared] == ["alpha@0"]          # the more accurate keeps the tail


def test_every_colour_is_from_the_brand_palette(snap):
    import figkit
    palette = {figkit.INK, figkit.BONE, figkit.SIGNAL, figkit.NIGHT, figkit.DUSK, figkit.TILE, figkit.MIST, figkit.SLATE}
    for theme in ("dark", "light"):
        svg = frontier.draw(snap, theme=theme)
        import re
        hexes = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", svg)}
        assert hexes <= palette, hexes - palette
        rgba = set(re.findall(r"rgba\((\d+),(\d+),(\d+),", svg))
        assert rgba <= {("141", "149", "166"), ("91", "98", "114")}            # Mist and Slate, translucent


def test_nothing_plotted_falls_outside_the_plot_box(snap):
    for theme in ("dark", "light"):
        root = _svg(snap, theme=theme)
        x0, y0, x1, y1 = _plot_box(root)
        for g in _role(root, "point"):
            assert x0 <= float(g.get("data-x")) <= x1 and y0 <= float(g.get("data-y")) <= y1
        for e in _role(root, "whisker") + _role(root, "tail") + _role(root, "ref"):
            for k in ("x1", "x2"):
                assert x0 <= float(e.get(k)) <= x1
            for k in ("y1", "y2"):
                assert y0 <= float(e.get(k)) <= y1
        for e in _role(root, "frontier"):
            for pt in e.get("points").split():
                x, y = (float(v) for v in pt.split(","))
                assert x0 <= x <= x1 and y0 <= y <= y1
        for g in _role(root, "arm-label") + _role(root, "point-label") + _role(root, "pick-label"):
            bx = [float(v) for v in g.get("data-box").split(",")]
            assert x0 <= bx[0] and bx[2] <= x1 and y0 <= bx[1] and bx[3] <= y1


def test_both_themes_draw_the_same_information(snap):
    dark, light = _svg(snap, "dark"), _svg(snap, "light")
    assert dark.get("data-theme") == "dark" and light.get("data-theme") == "light"
    for role in ("point", "fine-print", "arm-label", "whisker", "frontier", "ref"):
        assert [e.get("data-text") or e.get("data-label") or e.get("data-arm") for e in _role(dark, role)] == \
               [e.get("data-text") or e.get("data-label") or e.get("data-arm") for e in _role(light, role)]
    assert frontier.draw(snap, theme="dark") != frontier.draw(snap, theme="light")


def test_a_line_breaks_around_another_series_marker_it_runs_through(tmp_path):
    s = fx.setup(tmp_path)
    snap = frontier.build(frontier.load_spec(s["spec_path"]))
    by = by_label(snap)
    a0, a256 = by["alpha@0"], by["alpha@256"]
    mid = {k: [(a0["lat"][k][0] * a256["lat"][k][0]) ** 0.5, 0, 0] for k in ("mean", "p50", "p95", "p99")}
    snap["points"].append({**by["ref"], "label": "ref", "acc": (a0["acc"] + a256["acc"]) / 2,
                           "lat": {**mid, "max": 0.0}})                       # the reference, on alpha's line
    snap["points"] = [p for p in snap["points"] if not (p["label"] == "ref" and p is by["ref"])]
    root = _svg(snap)
    gaps = _role(root, "line-gap")
    assert [(g.get("data-arm"), g.get("data-at")) for g in gaps] == [("alpha", "ref")]
    from figkit import THEMES
    assert gaps[0].get("stroke") == THEMES["dark"]["ground"]


def _gap(box: tuple[float, ...], x: float, y: float) -> float:
    """Distance from a point to a label box (0 inside it)."""
    x0, y0, x1, y1 = box
    return math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))


def _owned(root: ET.Element) -> list[str]:
    """The budget labels nearer another series' marker than their own (none should be)."""
    pts = {g.get("data-label"): (float(g.get("data-x")), float(g.get("data-y")), g.get("data-arm"))
           for g in _role(root, "point")}
    bad = []
    for g in _role(root, "point-label"):
        box = tuple(float(v) for v in g.get("data-box").split(","))
        mx, my, arm = pts[g.get("data-label")]
        if not all(_gap(box, mx, my) < _gap(box, x, y) for x, y, a in pts.values() if a != arm):
            bad.append(g.get("data-label"))
    return bad


def test_every_budget_label_is_nearer_its_own_marker_than_another_series(snap):
    assert _owned(_svg(snap)) == []


def test_labels_do_not_overlap_each_other(snap):
    root = _svg(snap)
    boxes = [[float(v) for v in g.get("data-box").split(",")]
             for g in _role(root, "arm-label") + _role(root, "point-label") + _role(root, "pick-label")
             + _role(root, "ref-label")]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert min(a[2], b[2]) - max(a[0], b[0]) <= 0.5 or min(a[3], b[3]) - max(a[1], b[1]) <= 0.5


# ---- the release keys: budgets only, early exit in the fine print, pinned commits, the readout check ---------------

def _release(tmp: Path, **over) -> dict:
    """The fixture as a release spec: --cache on, no draft, alpha carrying early-exit points."""
    s = fx.setup(tmp)
    alpha = fx.with_early_exit(fx.profile("alpha", s["sha"], fx.ALPHA_CORRECT, fx.ALPHA_LAT, cache=True, gib=20.8))
    fx.write_profile(tmp, "alpha", alpha)
    fx.write_profile(tmp, "beta", fx.profile("beta", s["sha"], fx.BETA_CORRECT, fx.BETA_LAT, cache=True, gib=9.1))
    spec = {**s["spec"], "callo_cache": True, "callo_run": None, "draft": False, **over}
    (tmp / "rel.json").write_text(json.dumps(spec))
    return frontier.build(frontier.load_spec(tmp / "rel.json"))


def test_only_thinking_budgets_are_plotted_and_a_profiles_early_exit_points_are_listed(tmp_path):
    snap = _release(tmp_path)
    assert {p["kind"] for p in snap["points"]} == {"budget"}
    alpha = next(a for a in snap["arms"] if a["name"] == "alpha")
    assert alpha["opt_in"] == ["alpha@ee.95", "alpha@ee.95+g"] and alpha["budgets"] == [0, 256]
    # each model's frontier is over its budgets alone: the faster early-exit point does not dominate alpha@256
    assert by_label(snap)["alpha@256"]["frontier"] == {"mean": True, "p50": True, "p95": True}
    for stat in frontier.STATS:
        assert snap["picks"][stat] in by_label(snap)
    made = " ".join((f["lead"] or "") + " " + f["text"] for f in snap["made"])
    assert "opt-in points in the profile, not plotted: alpha@ee.95, alpha@ee.95+g" in made
    assert not [g for g in _role(_svg(snap), "point") if "ee" in g.get("data-label")]


def test_the_early_exit_line_states_each_named_models_point_and_its_plain_budget(tmp_path, commits):
    s = fx.setup(tmp_path)
    other = fx.with_early_exit(fx.profile("beta", s["sha"], fx.BETA_CORRECT, fx.BETA_LAT, cache=True, gib=9.1),
                               gate_share=0.5)
    (tmp_path / "ee").mkdir()
    fx.write_profile(tmp_path / "ee", "beta", other)
    snap = _release(tmp_path, early_exit={"z": 0.95, "gate": True, "budget": 256,
                                          "arms": {"alpha": None,
                                                   "beta": {"path": "ee/beta.think.json", "commit": "bbbbbbb"}}})
    ee = snap["early_exit"]
    acc = sum(fx.EE_CORRECT) / fx.N
    p50 = think.latency_stats(fx.EE_LAT, fx.CLUSTERS, seed=0, strata=fx.SOURCES)["lat_p50_ms"][0]
    assert [(p["arm"], p["own"], p["acc"], p["p50"], p["commit"]) for p in ee["points"]] == [
        ("alpha", True, acc, p50, None), ("beta", False, acc, p50, "bbbbbbb")]
    text = _fp(snap)
    a256 = by_label(snap)["alpha@256"]
    b256 = by_label(snap)["beta@256"]
    b = (f"{100 * b256['acc']:.1f} % at {frontier.secs(b256['lat']['p50'][0])} p50")
    assert (f"Early exit. Opt-in and experimental in Callo, never picked, so not plotted. At think 256, z .95 with the "
            f"gate: Alpha {100 * acc:.1f} % at {frontier.secs(p50)} p50 (its plain think 256: {100 * a256['acc']:.1f} % "
            f"at {frontier.secs(a256['lat']['p50'][0])} p50); Beta {100 * acc:.1f} % at {frontier.secs(p50)} p50, from a "
            f"separate calibration whose plain think 256 reads {b} (the table's run: {100 * b256['acc']:.1f} %).") in text
    assert "Beta's early-exit calibration on build b" in text           # the Builds line names its build
    made = " ".join((f["lead"] or "") + " " + f["text"] for f in snap["made"])
    assert "Early exit, Beta (not plotted). think 256, z .95 with the gate, from its own early-exit calibration, " \
           "`beta.think.json`" in made and "the gate sent 50 % of questions to full thinking" in made
    with pytest.raises(frontier.Refusal, match="separate calibration needs its callo commit"):
        _release(tmp_path, early_exit={"z": 0.95, "gate": True, "budget": 256, "arms": {"beta": "ee/beta.think.json"}})


def test_the_early_exit_line_refuses_another_model_or_a_missing_point(tmp_path):
    s = fx.setup(tmp_path)
    fx.write_profile(tmp_path, "gamma", fx.with_early_exit(fx.profile("gamma", s["sha"], fx.BETA_CORRECT, fx.BETA_LAT,
                                                                      cache=True)))
    with pytest.raises(frontier.Refusal, match="another model"):
        _release(tmp_path, early_exit={"z": 0.95, "gate": True, "budget": 256, "arms": {"beta": "gamma.think.json"}})
    with pytest.raises(frontier.Refusal, match="no ee_gated point at z 0.9"):
        _release(tmp_path, early_exit={"z": 0.9, "gate": True, "budget": 256, "arms": {"alpha": None}})
    with pytest.raises(frontier.Refusal, match="caps at 256, the spec says budget 512"):
        _release(tmp_path, early_exit={"z": 0.95, "gate": False, "budget": 512, "arms": {"alpha": None}})


COMMITS = {"aaaaaaa": ("0.2.0", "2026-10-07T09:00:00+00:00"), "bbbbbbb": ("0.2.0", "2026-10-07T23:00:00+00:00"),
           "ccccccc": ("0.2.0", "2026-10-08T09:30:00+00:00"), "ddddddd": ("0.1.0", "2026-10-01T09:00:00+00:00"),
           "fffffff": ("0.2.0", "2026-10-09T09:00:00+00:00")}
ORDER = ["ddddddd", "aaaaaaa", "bbbbbbb", "ccccccc", "fffffff"]          # the profiles were written 2026-10-08 10:00


@pytest.fixture
def commits(monkeypatch):
    """A stand-in history: four commits in order, their versions and commit times, and the builds table naming them
    ("build a" ...)."""
    import figkit
    monkeypatch.setattr(frontier, "commit_info", lambda c: None if c not in COMMITS else {
        "sha": c * 5, "short": c, "version": COMMITS[c][0], "time": COMMITS[c][1], "on_head": True})
    monkeypatch.setattr(frontier, "is_ancestor", lambda a, b: ORDER.index(a[:7]) <= ORDER.index(b[:7]))
    monkeypatch.setattr(figkit, "builds", lambda: [{"commit": c, "version": COMMITS[c][0], "name": f"build {c[0]}",
                                                    "what": f"what {c[0]} added"} for c in ORDER])


def test_pinned_commits_name_each_models_builds_on_the_plate(tmp_path, commits):
    snap = _release(tmp_path, callo_commits={"alpha": {"profile": "bbbbbbb", "measured": "aaaaaaa", "step": "refit"},
                                             "beta": "bbbbbbb"})
    arms = {a["name"]: a for a in snap["arms"]}
    assert arms["alpha"]["commits"] == {"step": "refit", "profile": "bbbbbbb", "measured": "aaaaaaa"}
    assert arms["beta"]["commits"] == {"step": None, "profile": "bbbbbbb", "measured": "bbbbbbb"}
    assert ("Builds. Callo 0.2.0 development builds: Alpha measured on build a, refit on build b; Beta on build b; "
            "llama.cpp b50-4ebdf2c (Callo).") in _fp(snap)
    made = " ".join((f["lead"] or "") + " " + f["text"] for f in snap["made"])
    assert "`callo` 0.2.0, measured on build a, refit on build b; --cache on (default)" in made
    assert snap["builds"] == [{"commit": "aaaaaaa", "name": "build a"}, {"commit": "bbbbbbb", "name": "build b"}]
    for text in (_fp(snap), made):
        assert "aaaaaaa" not in text and "bbbbbbb" not in text                 # by role, never by commit
    import figures_md
    md = figures_md.render(snap, None)
    assert "| build a | what a added | `aaaaaaa` |" in md and md.count("aaaaaaa") == 1


def test_a_pinned_commit_the_builds_table_does_not_name_is_refused(tmp_path, commits, monkeypatch):
    import figkit
    monkeypatch.setattr(figkit, "builds", lambda: [])
    with pytest.raises(frontier.Refusal, match="aaaaaaa is not in brand/data/builds.json"):
        _release(tmp_path, callo_commits={"alpha": "aaaaaaa"})


def test_the_snapshot_records_paths_relative_to_the_root(tmp_path):
    snap = _release(tmp_path)
    assert str(tmp_path) not in json.dumps(snap, ensure_ascii=False) and "root" not in snap["generator"]
    assert snap["data"]["path"] == "test-s1.jsonl"
    arms = {a["name"]: a for a in snap["arms"]}
    assert arms["alpha"]["path"] == "alpha.think.json" and arms["alpha"]["files"][0]["path"] == "alpha.think.json"
    assert [f["path"] for f in arms["ref"]["files"]] == ["ref/rows.jsonl", "ref/meta.json"]


@pytest.mark.parametrize("pin, says", [
    ({"alpha": "eeeeeee"}, "eeeeeee is not a commit of this repository"),
    ({"alpha": "ddddddd"}, "ddddddd is callo 0.1.0, the profile says 0.2.0"),
    ({"alpha": "fffffff"}, "is later than when the profile was written"),
    ({"alpha": {"profile": "ccccccc", "measured": "fffffff"}}, "later than when its run started"),
    ({"alpha": {"profile": "aaaaaaa", "measured": "bbbbbbb"}}, "bbbbbbb is not an ancestor of the profile commit"),
    ({"ref": "aaaaaaa"}, "callo_commits names ref: not a present callo arm"),
])
def test_a_pinned_commit_that_cannot_have_written_the_profile_is_refused(tmp_path, commits, pin, says):
    with pytest.raises(frontier.Refusal, match=says):
        _release(tmp_path, callo_commits=pin)


def test_the_readout_check_says_how_far_a_fix_moved_a_model_measured_on_both_sides(tmp_path, commits):
    s = fx.setup(tmp_path)
    worse = [False] + fx.BETA_CORRECT[256][1:]                    # one question fewer right at 256 before the fix
    (tmp_path / "before").mkdir()
    fx.write_profile(tmp_path / "before", "beta", fx.profile("beta", s["sha"], {0: fx.BETA_CORRECT[0], 256: worse},
                                                             fx.BETA_LAT, cache=True, gib=9.1))
    snap = _release(tmp_path, callo_commits={"alpha": "aaaaaaa", "beta": "ccccccc"},
                    readout_check={"arm": "beta", "before": "before/beta.think.json", "fix": "bbbbbbb"})
    rc = snap["readout_check"]
    assert rc["predates"] == ["aaaaaaa"]                           # alpha ran before the fix, beta after it
    assert rc["max_move"] == pytest.approx(abs(sum(worse) - sum(fx.BETA_CORRECT[256])) / fx.N)
    assert [r["same_T"] for r in rc["budgets"]] == [True, True]
    assert ("Build a predates the readout fix; Beta, measured on both sides of it, moved by at most "
            f"{100 * rc['max_move']:.1f} pp.") in _fp(snap)
    assert rc["before"] == "before/beta.think.json"
    made = " ".join((f["lead"] or "") + " " + f["text"] for f in snap["made"])
    assert "Readout fix. Beta measured before it, `beta.think.json`" in made and "(plotted) on build b: accuracy" in made
    assert "Arms measured on build a predate it." in made


def test_the_gib_line_carries_each_arms_note_and_no_draft_or_pending_is_shown(tmp_path):
    s = fx.setup(tmp_path)
    arms = [{**a, "gib_note": "Alpha's counts both shards"} if a["name"] == "alpha" else a for a in s["spec"]["arms"]]
    snap = _release(tmp_path, arms=[a for a in arms if a["name"] != "later"])
    assert f"GiB. {frontier.GIB_TEXT}; Alpha's counts both shards." in _fp(snap)
    root = _svg(snap)
    assert not _role(root, "badge") and "Pending" not in _fp(snap)


# ---- real data: the release spec (skipped without the bench results) --------------------------------------------

RELEASE_SPEC = Path(frontier.HERE) / "data" / "frontier.release.spec.json"
RESULTS = Path(os.environ["CALLO_PLATE_RESULTS"]) if os.environ.get("CALLO_PLATE_RESULTS") else None   # the results root


@pytest.fixture(scope="module")
def real() -> dict:
    spec = json.loads(RELEASE_SPEC.read_text())
    paths = [a.get("profile") or a["rows"] + "/rows.jsonl" for a in spec["arms"]]
    paths += [p["path"] if isinstance(p, dict) else p for p in spec["early_exit"]["arms"].values() if p]
    paths += [spec["readout_check"]["before"]]
    if RESULTS is None or not all((RESULTS / p).is_file() for p in paths):
        pytest.skip(f"real plate inputs not under {RESULTS}" if RESULTS else "CALLO_PLATE_RESULTS (the results root) is not set")
    if frontier.commit_info("9d52aaf") is None:
        pytest.skip("this checkout does not hold the release commits the spec pins")
    return frontier.build(frontier.load_spec(RELEASE_SPEC, root=RESULTS))


def test_real_release_reproduces_the_measured_numbers(real):
    pts = by_label(real)
    want = {"tiel@0": 0.832, "tiel@256": 0.920, "tiel@512": 0.936, "tiel@1024": 0.941, "dirk@1024": 0.941,
            "qwen9b@1024": 0.928, "gemma-e4b@0": 0.747, "sharp-minicpm@0": 0.577, "sharp-minicpm@1024": 0.870,
            "flashnext-sharp@0": 0.910, "flashnext-sharp@256": 0.920}
    for lab, acc in want.items():
        assert round(pts[lab]["acc"], 3) == acc, lab
        assert pts[lab]["n"] == 376
    clef = {"clef-flash-bf16": (0.854, 286), "clef-flash-release": (0.856, 574), "clef-q4": (0.918, 1007),
            "clef-flash-udq4": (0.851, 290), "clef-ggml-q4km": (0.912, 919), "kev-9b": (0.859, 704)}
    for lab, (acc, mean) in clef.items():
        assert (round(pts[lab]["acc"], 3), round(pts[lab]["lat"]["mean"][0])) == (acc, mean), lab
    assert {p["kind"] for p in real["points"]} == {"budget"}
    arms = {a["name"]: a for a in real["arms"]}
    assert arms["flashnext-sharp"]["opt_in"] == ["flashnext-sharp@ee.9", "flashnext-sharp@ee.95",
                                                 "flashnext-sharp@ee.9+g", "flashnext-sharp@ee.95+g"]
    assert all(arms[n]["cache"] is True and arms[n]["callo_version"] == "0.3.0"
               for n in ("tiel", "dirk", "qwen9b", "gemma-e4b", "sharp-minicpm", "flashnext-sharp"))
    assert arms["sharp-minicpm"]["commits"]["profile"] == "a3a88c3"
    assert arms["flashnext-sharp"]["commits"] == {"step": "gate reads and profile", "profile": "ea52406",
                                                  "measured": "6fc42a8"}
    assert "cyberdirk" not in arms and real["pending"] == [] and real["draft"] is False
    assert arms["clef-flash-udq4"]["on_plate"] is False and arms["clef-ggml-q4km"]["on_plate"] is False
    assert real["picks"] == {"mean": "flashnext-sharp@0", "p50": "flashnext-sharp@0", "p95": "flashnext-sharp@0"}
    assert real["validation"]["agree"] == 373 and real["validation"]["pass"] is True
    rc = real["readout_check"]
    assert rc["predates"] == ["b463e5f"] and round(100 * rc["max_move"], 1) == 0.5


def test_real_release_fine_print_has_the_required_strings(real):
    text = _fp(real)
    for s_ in ("n = 376 test-s1 (mmlu 80, qnli 80, paws 80, composition_holdout 96, contrastive 40); idle M2 Ultra 64 GB, "
               "one at a time; Callo 2026-10-10 to 2026-10-13; Clef 2026-10-09; Kev-9B 2026-10-02.",
               "Timers. Callo arms: --cache on (default); in-Callo frame + read wall time.",
               "Clef arms (2026-10-09). Client wall per request (clef_run.py",
               "Paired 95 % cluster-bootstrap CIs vs Clef-Flash (llama.cpp BF16), stratified by source",
               "At think 256, z .95 with the gate: Tiel-Coder 92.3 % at 1.0 s p50, from a separate calibration whose "
               "plain think 256 reads 92.3 % at 1.4 s p50 (the table's run: 92.0 %); Qwen3.8-Flash-Next 92.6 % at "
               "2.1 s p50 (its plain think 256: 92.0 % at 5.5 s p50).",
               "Qwen3.8-Flash-Next's counts both shards of its split GGUF",
               "Builds. Callo 0.3.0 development builds: Tiel-Coder, Dirk, Qwen3.5-9B and Gemma-4-E4B measured on the "
               "first build, refit on the refit build; Qwen3.8-Flash-Next measured on the early-exit build, gate reads "
               "and profile on the gate-reads build; Sharp-MiniCPM5-2B on the readout-fix build; Tiel-Coder's "
               "early-exit calibration on the gate build; llama.cpp b50-4ebdf2c (Callo) and b11379 1537a0a8 (Clef). "
               "The first build predates the readout fix; Sharp-MiniCPM5-2B, measured on both sides of it, moved by at "
               "most 0.5 pp.",
               "Validation. llama.cpp BF16 vs Clef-Flash's release code (2026-10-02 rows): 373/376 argmax agreement",
               "Engines. Clef-Flash (release code): 574 ms mean, 2.0× the llama.cpp run's 286 ms",
               frontier.pick_definition("mean"), frontier.BUDGET_REPLAY):
        assert s_ in text, s_
    for s_ in ("Pending", "DRAFT", "PRELIMINARY", "run1", "`callo compare` flags", "What Q4 costs", "bench-qwen36"):
        assert s_ not in text, s_
    made = " ".join((f["lead"] or "") + " " + f["text"] for f in real["made"])
    for c in ("b463e5f", "9d52aaf", "a3a88c3", "6fc42a8", "ea52406", "0d33846"):        # builds by role, never commit
        assert c not in text and c not in made, c
    assert [b["commit"] for b in real["builds"]] == ["b463e5f", "9d52aaf", "a3a88c3", "6fc42a8", "0d33846", "ea52406"]


def _prose_callo(snap: dict) -> list[str]:
    """Every drawn text (both themes, every statistic) that names the product in lowercase: in prose it is "Callo";
    a lowercase "callo" is only the command (callo compare)."""
    import re
    prose = re.compile(r"(?<![\w/`.-])callo(?![\w/`-])(?!\.\w)(?! (?:compare|serve|calibrate|classify)\b)")
    out = []
    for theme in ("dark", "light"):
        for stat in frontier.STATS:
            root = _svg(snap, theme, stat)
            out += [t for t in [e.get("data-text") for e in root.iter()] + [root.get("aria-label")]
                    if t and prose.search(t)]
    return out


def test_the_product_is_callo_in_every_drawn_word(snap):
    assert _prose_callo(snap) == []
    text = _fp(snap)
    assert "Callo arms: run1" in text and "(callo compare default, on mean)" in text


def test_real_release_names_the_product_callo_in_every_drawn_word(real):
    assert _prose_callo(real) == []
    assert {a["runtime"] for a in real["arms"] if a["kind"] == "callo"} == {"Callo · llama.cpp"}
    tags = [e.get("data-text") for e in _role(_svg(real), "table-tag")]
    assert tags.count("Callo · llama.cpp") == 6, tags


def test_real_release_story_and_title(real):
    st = real["story"]["mean"]["subtitle"]
    assert st == ("Tiel-Coder with no thinking matches Clef-Flash (−2.1 pp, CI includes 0) at 0.59× its mean latency, "
                  "both on llama.cpp; ~140 thinking tokens put it +6.6 to +8.2 pp above Clef-Flash at 1.7–1.8\u00a0s "
                  "mean.")
    assert real["title"] == "Tiel-Coder ties Clef-Flash without thinking, *a short think passes it*"
    assert "Trained-head" not in real["title"]


def test_real_release_draws_the_plate_arms_budgets_only_and_no_ud_but_gemmas(real):
    for theme in ("dark", "light"):
        root = _svg(real, theme=theme)
        drawn = {g.get("data-arm") for g in _role(root, "point")}
        assert drawn == {"tiel", "dirk", "qwen9b", "gemma-e4b", "sharp-minicpm", "flashnext-sharp", "clef-flash-bf16",
                         "clef-flash-release", "clef-q4", "kev-9b"}
        assert len(_role(root, "point")) == 23 and not _role(root, "badge")
        (ref,) = _role(root, "ref")
        assert float(ref.get("data-acc")) == real["ref"]["acc"]
        shown = [e.get("data-text") for e in root.iter() if e.get("data-text")]
        assert all("UD-" not in t_ or "Gemma-4-E4B · UD-Q4_K_XL" in t_ for t_ in shown), \
            [t_ for t_ in shown if "UD-" in t_ and "Gemma" not in t_]
        assert not any("bench-qwen36" in t_ or "Trained-head" in t_ for t_ in shown)


def test_real_release_lines_break_at_the_clef_markers_and_budget_labels_are_owned(real):
    root = _svg(real)
    gaps = {(g.get("data-arm"), g.get("data-at")) for g in _role(root, "line-gap")}
    assert {("tiel", "clef-flash-bf16"), ("flashnext-sharp", "clef-q4")} <= gaps
    assert _owned(root) == []


def test_the_committed_snapshot_is_the_real_build(real):
    committed = json.loads((Path(frontier.HERE) / "data" / "frontier.json").read_text(encoding="utf-8"))
    strip = ("generator", "made")                    # when and at which commit it was built
    assert {k: v for k, v in committed.items() if k not in strip} == \
           {k: v for k, v in json.loads(json.dumps(real, ensure_ascii=False)).items() if k not in strip}


def test_real_release_figures_md_names_inputs_without_the_bench_checkout(real):
    import figures_md
    md = figures_md.render(real, None)
    section = md[md.index("## The release plate"):]
    for s_ in ("bench-qwen36", "/Users/", "/Volumes/", "release-refit", "release3-minicpm", "Validation (G1)",
               "listed as pending", "every arm measured at"):
        assert s_ not in section, s_
    for s_ in ("Early exit, Tiel-Coder (not plotted).", "Readout fix.", "Arms measured on the first build predate it.",
               "`callo` 0.3.0, measured on the first build, refit on the refit build;",
               "| Qwen3.8-Flash-Next (Sharp) · IQ3_XXS · no think | 91.0 % | +5.6 pp [+1.3, +10.4] |",
               "- Tiel-Coder, early-exit calibration: `tiel.think.json`"):
        assert s_ in section, s_


def test_a_figures_only_arm_is_measured_and_reported_but_not_drawn_and_dates_come_from_sessions(tmp_path):
    s = fx.setup(tmp_path)
    d = fx.write_rows(tmp_path / "q4", s["sha"], fx.REF_CORRECT, fx.REF_LAT, timer="client_http", date=None)
    meta = json.loads((d / "meta.json").read_text())
    meta.pop("date", None)
    meta["sessions"] = [{"utc_start": "2026-10-09T21:11:16+00:00"}]
    (d / "meta.json").write_text(json.dumps(meta))
    arm = {"name": "q4", "label": "Ref 9B · UD-Q4_K_XL", "kind": "rows", "rows": "q4", "runtime": "llama.cpp",
           "marker": "diamond-hollow", "timer": "client_http", "plate": False}
    (tmp_path / "f.json").write_text(json.dumps({**s["spec"], "arms": [*s["spec"]["arms"], arm]}))
    snap = frontier.build(frontier.load_spec(tmp_path / "f.json"))
    rec = next(a for a in snap["arms"] if a["name"] == "q4")
    assert rec["on_plate"] is False and rec["date"] == "2026-10-09"
    assert "q4" in {p["arm"] for p in snap["points"]}
    root = _svg(snap)
    assert "q4" not in {g.get("data-arm") for g in _role(root, "point") + _role(root, "arm-label")}
    made = " ".join((f["lead"] or "") + " " + f["text"] for f in snap["made"])
    assert "Ref 9B · UD-Q4_K_XL (measured, not drawn). accuracy" in made
    for stat in frontier.STATS:
        assert "Clef arms" not in _fp(snap, stat)                # its timer line is not on the plate either
