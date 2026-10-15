"""`callo compare`: refusals, loud flags and notes, the joint frontier with paired CIs over the
shared question ids, the pick and its filters, the report (markdown with the ASCII scatter, JSON), the CLI."""
import copy
import json
import math
import random

import pytest

from callo.compare import (
    correct_at,
    display_names,
    flags,
    joint,
    load_profile,
    notes,
    points_of,
    refusals,
    sampler_statement,
    sampler_text,
)
from callo.errors import InvalidInput
from callo.think import Preference, paired_bootstrap

SHA = "d" * 64
N = 10


def profile(name: str, correct: dict[int, int], lat: dict[int, float], *, key: str = "k1", sha: str = SHA,
            policy: str = "model", pinned=(0.6, 0.95, 20, 0.0), origin=("gguf", "gguf", "callo", "callo"),
            fingerprint: str = "h" * 64, capable: bool = True, reason=None, merge: tuple[int, int] | None = None,
            gib: float = 4.0, **over) -> dict:
    """A think profile with the fields compare reads. `correct[b]` is how many of the N questions (the first ones) are
    right at budget b; `lat[b]` the p50 in ms (the CI is +-10 %)."""
    fields = ("temperature", "top_p", "top_k", "min_p")
    budgets: dict[str, dict] = {}
    for b in sorted(correct):
        label = f"{name}@{b}" if merge is None or b not in merge else f"{name}@{merge[0]}≡{merge[1]}"
        e = {"n": N, "acc": correct[b] / N, "lat_p50_ms": [lat[b], 0.9 * lat[b], 1.1 * lat[b]],
             "lat_p95_ms": [1.5 * lat[b], 1.4 * lat[b], 1.6 * lat[b]], "nll": 0.7, "nll_T": 0.65, "ece": 0.1, "ece_T": 0.08,
             "T": 1.2, "point": label, "frontier": True, "dominated_by": None, "approx": False, "option_mass": 0.98}
        if b > 0:
            e.update({"think_mean": 0.8 * b, "think_p95": b, "forced": 0.5, "eos": 0.0, "readout_reprefill": 0,
                      "text_fallback": 0, "dacc": [(correct[b] - correct[0]) / N, -0.1, 0.5]})
        budgets[str(b)] = e
    rows = [{"q": f"rec-{i}/q", "cluster": f"g{i // 2}", "label": 0, "n": 3,
             "b": {str(b): [i < correct[b], 0.5, 0.6, lat[b] * (1 + i / 100), min(b, 300) if b else 0,
                            "forced" if b else None] for b in correct}} for i in range(N)]
    doc = {"callo_think_profile": 1, "metrics": "callo-think-metrics-1", "model_key": "sha256:" + key * 64, "model_name": name,
           "created": "2026-10-05T10:00:00Z",
           "comparable": {"data": {"name": "d.jsonl", "sha256": sha, "questions": N}, "budgets": sorted(correct), "seed": 0,
                          "warmup": 3, "audit": 6, "cache": True, "sampling_policy": policy,
                          "sampling": {"policy": policy, "pinned": dict(zip(fields, pinned, strict=True)),
                                       "origin": dict(zip(fields, origin, strict=True)), "gguf": {}},
                          "readout": {"variant": "main", "top_k": 100, "max_depth": 8, "divergence_mass": 0.1,
                                      "callo_version": "0.2.0", "cache_prompt": False, "user_delimiter": "<|im_start|>user",
                                      "prefix": "tokens"}},
           "environment": {"host": {"fingerprint": "sha256:" + fingerprint, "chip": "Apple M3 Max", "memory_bytes": 64 * 2**30},
                           "build_info": "b1-926862e",
                           "model": {"file": f"{name}.gguf", "bytes": int(gib * 2**30), "mmproj": None, "mmproj_bytes": None},
                           "server": {"kv": "unified", "n_ctx": 65536, "total_slots": 4, "s1_slots": 1, "slot": 0,
                                      "server_args": None, "upstream_local": True, "gpu_busy": False,
                                      "position_bias_sha256": None, "launched_by": "menu"},
                           "run": {"started": "2026-10-05T10:00:00Z", "ended": "2026-10-05T10:20:00Z",
                                   "decode_tok_s_median": 100.0}},
           "thinking": {"capable": capable, "reason": reason, "close": "\n</think>\n\n", "synthesized_open": False},
           "server_state": {"state": "clean", "checks": []},
           "audit": {"latency_model": "verified", "budgets": {}},
           "drift": {"n": 3, "ratio": 1.0, "therm": [{"at": "A", "limit": 100}, {"at": "end", "limit": 100}], "flagged": False},
           "budgets": budgets, "pick": {"budget": 0, "model": name, "label": f"{name}@0", "rule": "fastest within noise of the best",
                                        "best": f"{name}@{max(correct)}", "latency": "p50"},
           "calibration_overhead_s": 1.0, "row_fields": ["correct", "nll1", "pmax", "lat_ms", "k", "close"], "rows": rows}
    for k, v in over.items():
        node, _, leaf = k.rpartition("__")
        target = doc
        for part in node.split("__") if node else []:
            target = target[part]
        target[leaf] = v
    return doc


A = profile("alpha", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0})
B = profile("beta", {0: 6, 256: 9}, {0: 400.0, 256: 2500.0}, key="k2", gib=9.0)


def test_load_profile_accepts_a_think_profile_only(tmp_path):
    p = tmp_path / "a.think.json"
    p.write_text(json.dumps(A))
    assert load_profile(p)["model_name"] == "alpha"
    (tmp_path / "bias.json").write_text(json.dumps({"callo_bias_profile": 2}))
    with pytest.raises(InvalidInput, match="not a think profile"):
        load_profile(tmp_path / "bias.json")
    (tmp_path / "x.json").write_text("{nope")
    with pytest.raises(InvalidInput, match="not JSON"):
        load_profile(tmp_path / "x.json")
    with pytest.raises(InvalidInput, match="cannot read"):
        load_profile(tmp_path / "missing.json")


def test_refusals_name_the_profile_and_the_reason():
    assert refusals([A, B]) == []
    other = profile("gamma", {0: 5}, {0: 300.0}, sha="e" * 64)
    assert refusals([A, other]) and "different data" in refusals([A, other])[0] and "gamma" in refusals([A, other])[0]
    assert "metrics version" in refusals([A, profile("g", {0: 5}, {0: 1.0}, metrics="callo-think-metrics-0")])[0]
    assert "readout-engine settings" in refusals([A, profile("g", {0: 5}, {0: 1.0}, comparable__readout__top_k=50)])[0]
    fewer = profile("g", {0: 5}, {0: 1.0})
    fewer["rows"] = fewer["rows"][:-1]
    assert "different question id sets (9 vs 10 ids)" in refusals([A, fewer])[0]
    with pytest.raises(InvalidInput, match="refusing to compare: .*different data"):
        joint([A, other])
    with pytest.raises(InvalidInput, match="at least one"):
        joint([])
    with pytest.raises(InvalidInput, match="--latency"):
        joint([A], pref=Preference(latency="p99"))


def test_display_names_sampler_text_and_the_header_statement():
    assert display_names([A, B, A, A]) == ["alpha", "beta", "alpha#2", "alpha#3"]
    assert sampler_text(A) == "0.6/0.95/20*/0*" and sampler_text(A, mark_callo=False) == "0.6/0.95/20/0"
    assert sampler_statement([A, B]).startswith("Each model thinks with its own sampler")
    fixed = [profile("a", {0: 5}, {0: 1.0}, policy="fixed", origin=("callo",) * 4),
             profile("b", {0: 5}, {0: 1.0}, policy="fixed", origin=("callo",) * 4)]
    assert sampler_statement(fixed) == "Every model thinks with the same sampler, 0.6/0.95/20/0."
    mixed = [A, profile("b", {0: 5}, {0: 1.0}, policy="fixed")]
    assert sampler_statement(mixed) == "policies differ: alpha model, b fixed. Each point shows its sampler."
    assert sampler_statement(mixed) in flags(mixed)


def test_flags_and_notes_cover_every_listed_condition():
    small = [f"{n}: {N} questions (< 100): its intervals are wide" for n in ("alpha", "beta")]   # N = 10
    assert flags([A, B]) == [] and notes([A, B]) == small
    big = [profile(n, {0: 5}, {0: 1.0}, comparable__data__questions=q) for n, q in (("p", 100), ("q", 250))]
    assert notes(big) == []                                                  # at or above the threshold: no note
    c = profile("gamma", {0: 5}, {0: 1.0}, fingerprint="x" * 64, environment__build_info="b2-ffffff",
                comparable__readout__callo_version="0.3.0", comparable__readout__cache_prompt=True,
                comparable__readout__user_delimiter="<user>", comparable__seed=7, comparable__warmup=0,
                environment__server__position_bias_sha256="sha256:" + "b" * 64,
                environment__server__server_args=["-ctk", "q8_0"], environment__server__upstream_local=False,
                environment__server__gpu_busy=True, server_state__state="busy", audit__latency_model="unverified",
                drift__flagged=True)
    f = flags([A, c])
    heads = [x.split(":")[0] for x in f]
    assert heads[:9] == ["latencies from different machines are not comparable", "different llama-server builds",
                         "different callo versions", "different --cache (budget 0's latency depends on it)",
                         "different user delimiters", "different seeds", "different warm-ups",
                         "different position-bias profiles (budget 0 only)",
                         "different server args (KV types, speculative/MTP, LoRA)"]
    assert "alpha none, gamma -ctk q8_0" in f[8]
    assert f[9:] == ["gamma: the upstream was not local (latencies include the network)",
                     "gamma: another model server was running during calibration (gpu_busy)",
                     "gamma: server state busy (a failed server-state check was recorded under --allow-busy)",
                     "gamma: latency model unverified (the audit's error exceeded 10 %): its cut latencies are estimates",
                     "gamma: drift flagged (decode speed or thermal limit changed during the run)"]
    reused = profile("delta", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0}, key="k9", audit__latency_model="unverified",
                     audit__latency_reason="prefill_reused", budgets__256__prefill_reused=0.42)
    cached = profile("eps", {0: 5}, {0: 300.0}, key="k8", environment__server__cache_ram=0)
    fr = flags([A, reused, cached])
    assert fr[0].startswith("different host prompt-cache settings (llama-server --cache-ram; null: unknown, an attach mode): alpha unknown, delta unknown, eps 0")
    assert fr[1] == ("delta: phase-B generations reused a cached state of their own question (llama-server's host prompt "
                     "cache; prefill_reused 256: 42 %): its latencies are flattered; recalibrate with --cache-ram 0 on the "
                     "llama-server")
    assert not any("latency model unverified" in x for x in fr)              # the reason replaces the generic flag
    hot = profile("alpha", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0}, pinned=(1.0, 0.95, 20, 0.0), policy="fixed")
    same = [x for x in flags([A, hot]) if x.startswith("the same model")]
    assert same == ["the same model (sha256:k1k1k1k1k1k1) sampled at 0.6/0.95/20/0 in alpha and 1/0.95/20/0 in alpha#2: "
                    "its points differ by sampler, not by model"]
    assert notes([profile("g", {0: 5}, {0: 1.0}, capable=False, reason="unsupported_think_format", drift__therm=None),
                  profile("h", {0: 5}, {0: 1.0}, capable=False, reason="no_think_block",
                          drift__therm=[{"at": "A", "limit": None}])]) == [
        "g: 10 questions (< 100): its intervals are wide", "g: budget 0 only: thinks in a format callo cannot read",
        "g: no thermal data", "h: 10 questions (< 100): its intervals are wide", "h: budget 0 only: no thinking mode",
        "h: no thermal data"]


def test_points_of_keeps_the_profiles_merges_and_correct_at_reads_the_rows():
    merged = profile("m", {0: 5, 64: 7, 256: 7}, {0: 100.0, 64: 500.0, 256: 500.0}, merge=(64, 256))
    pts = points_of(merged, "m")
    assert [p.label for p in pts] == ["m@0", "m@64≡256"] and pts[1].budgets == (64, 256) and pts[1].budget == 64
    assert pts[1].lat_p50 == (500.0, 450.0, 550.0) and pts[1].sampler == "0.6/0.95/20*/0*" and pts[1].gib == 4.0
    c = correct_at(A)
    assert len(c) == N and c["rec-0/q"] == {0: 1.0, 256: 1.0} and c["rec-7/q"] == {0: 0.0, 256: 1.0}


def test_the_joint_frontier_pairs_points_over_the_shared_questions():
    j = joint([A, B])
    assert j.names == ("alpha", "beta") and [p.label for p in j.ranked] == ["alpha@0", "beta@0", "alpha@256", "beta@256"]
    assert all(p.dominated_by is None for p in j.ranked) and [p.label for p in j.front] == [p.label for p in j.ranked]
    pairs = {(x["a"], x["b"]): x for x in j.pairs}
    # `dacc` is b − a: neighbours are (faster, slower); the rest are (M, point), the point minus the most accurate
    # (the documented sign); alpha@256 neighbours M, so that pair is recorded once, as neighbours
    assert set(pairs) == {("alpha@0", "beta@0"), ("beta@0", "alpha@256"), ("alpha@256", "beta@256"),
                          ("beta@256", "alpha@0"), ("beta@256", "beta@0")}
    assert pairs[("beta@256", "alpha@0")]["n"] == N and pairs[("beta@256", "alpha@0")]["dacc"][0] == pytest.approx(-0.4)
    assert pairs[("beta@256", "alpha@0")]["why"] == "against the most accurate"
    assert pairs[("alpha@0", "beta@0")]["dacc"][0] == pytest.approx(0.1) and pairs[("alpha@0", "beta@0")]["why"] == "frontier neighbours"
    lo, hi = pairs[("alpha@256", "beta@256")]["dacc"][1:]
    assert lo <= 0.1 <= hi and all(math.isfinite(v) for v in (lo, hi))
    assert j.chosen.best is not None and j.chosen.best.label == "beta@256" and j.chosen.point is not None
    assert j.chosen.rule == "fastest within noise of the best" and j.filters["latency"] == "mean"   # the default
    assert j.dacc_best is not None and j.dacc_best[1] <= 0 <= j.dacc_best[2]      # the pick is within noise of the best
    # dacc_best is the pick's own, point − M: the pick (beta@0, 6/10) is less accurate than M (beta@256, 9/10)
    assert j.chosen.point.label == "beta@0" and j.dacc_best == j.chosen.dacc_best
    assert j.dacc_best[0] == pytest.approx(-0.3) and j.dacc_best == pytest.approx(pairs[("beta@256", "beta@0")]["dacc"])
    # filters: memory removes beta (9 GiB), min-accuracy picks the fastest point at or above it
    small = joint([A, B], pref=Preference(max_gib=5.0))
    assert small.chosen.best is not None and small.chosen.best.label == "alpha@256"
    # the filters apply before the pick's own frontier: a model that does not fit never
    # eliminates one that does, even when it dominates it
    big = profile("big", {0: 9}, {0: 250.0}, key="k7", gib=60.0)
    fits = profile("fits", {0: 8}, {0: 260.0}, key="k6", gib=5.0)
    both = joint([big, fits])
    assert both.by_label("fits@0").dominated_by == "big@0"
    only_fits = joint([big, fits], pref=Preference(max_gib=10.0))
    assert only_fits.chosen.point is not None and only_fits.chosen.point.label == "fits@0" and only_fits.chosen.best.label == "fits@0"
    assert joint([A, B], pref=Preference(min_accuracy=0.85)).chosen.point.label == "beta@256"
    assert joint([A, B], pref=Preference(min_accuracy=0.85)).chosen.rule == "min-accuracy"
    assert joint([A, B], pref=Preference(max_latency_s=0.35)).chosen.point.label == "alpha@0"
    assert joint([A, B], pref=Preference(latency="p95", max_latency_s=0.5)).chosen.point.label == "alpha@0"
    # a dominated point: slower and no more accurate
    slow = profile("slow", {0: 4}, {0: 900.0}, key="k3")
    d = joint([A, B, slow]).by_label("slow@0")
    assert d is not None and d.dominated_by == "beta@0"
    one = joint([A])
    assert [p.label for p in one.ranked] == ["alpha@0", "alpha@256"] and one.pairs[0]["why"] == "frontier neighbours"


def sourced(p: dict, sources: list, *, strata="source") -> dict:
    """`p` with calibrate's stratification record: `comparable.data.strata` and a `source` on every row."""
    p["comparable"]["data"]["strata"] = strata
    for r, s in zip(p["rows"], sources, strict=True):
        r["source"] = s
    return p


SOURCES = ["big"] * 8 + ["small"] * 2          # clusters g0-g3 in "big", g4 in "small": two very different sizes


def test_the_paired_intervals_are_stratified_by_source_when_every_profile_was():
    """The pairing re-runs calibrate's stratified bootstrap from the profiles alone."""
    a = sourced(profile("alpha", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0}), SOURCES)
    b = sourced(profile("beta", {0: 6, 256: 9}, {0: 400.0, 256: 2500.0}, key="k2", gib=9.0), SOURCES)
    got = {(x["a"], x["b"]): x["dacc"] for x in joint([a, b]).pairs}[("alpha@256", "beta@256")]
    xs = [float(i < 8) for i in range(N)]
    ys = [float(i < 9) for i in range(N)]
    clusters = [f"g{i // 2}" for i in range(N)]
    stratified = paired_bootstrap(xs, ys, clusters, seed=0, strata=SOURCES)
    plain = paired_bootstrap(xs, ys, clusters, seed=0)
    assert stratified != plain                                     # the strata change the interval here
    assert got == pytest.approx(list(stratified))
    assert not any("bootstrap strata" in f for f in flags([a, b]))
    # no profile stratified: plain cluster resampling, as before
    plain_pairs = {(x["a"], x["b"]): x["dacc"] for x in joint([A, B]).pairs}
    assert plain_pairs[("alpha@256", "beta@256")] == pytest.approx(list(plain))


def test_profiles_that_disagree_on_strata_pair_unstratified_and_say_so():
    a = sourced(profile("alpha", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0}), SOURCES)
    b = profile("beta", {0: 6, 256: 9}, {0: 400.0, 256: 2500.0}, key="k2", gib=9.0)       # strata missing: none
    assert "different bootstrap strata (source vs none): paired intervals are unstratified" in flags([a, b])
    got = {(x["a"], x["b"]): x["dacc"] for x in joint([a, b]).pairs}[("alpha@256", "beta@256")]
    plain = paired_bootstrap([float(i < 8) for i in range(N)], [float(i < 9) for i in range(N)],
                             [f"g{i // 2}" for i in range(N)], seed=0)
    assert got == pytest.approx(list(plain))
    nulled = sourced(profile("beta", {0: 6, 256: 9}, {0: 400.0, 256: 2500.0}, key="k2"), [None] * N, strata=None)
    assert "different bootstrap strata (source vs none): paired intervals are unstratified" in flags([a, nulled])


def test_an_unknown_host_never_counts_as_the_same_machine():
    """calibrate's host fingerprint is None for an unknown host; None is never a match."""
    unknown = profile("gamma", {0: 5}, {0: 1.0}, key="k3", environment__host__fingerprint=None)
    f = flags([A, unknown])
    assert f[:2] == ["latencies from different machines are not comparable: alpha sha256:hhhhhhhhhhhh, gamma unknown",
                     "latencies from an unknown host cannot be checked as same-machine: gamma"]
    other = profile("delta", {0: 5}, {0: 1.0}, key="k4", environment__host=None)
    both = flags([unknown, other])
    assert both[0] == "latencies from an unknown host cannot be checked as same-machine: gamma, delta"
    assert not any(x.startswith("latencies from different machines") for x in both)
    assert not any("unknown host" in x for x in flags([unknown]))              # one profile: nothing to compare
    assert not any("unknown host" in x for x in flags([A, B]))


def test_every_pair_resamples_one_row_set_paired_by_question_id():
    """The pairing contract (calibrate_think's `fixed`): one row set, in the first profile's order, for every pair."""
    base = joint([A, B])
    shuffled = copy.deepcopy(B)
    random.Random(1).shuffle(shuffled["rows"])
    assert [r["q"] for r in shuffled["rows"]] != [r["q"] for r in B["rows"]]
    again = joint([A, shuffled])
    assert again.pairs == base.pairs and again.dacc_best == base.dacc_best and again.fixed == base.fixed
    gap = copy.deepcopy(B)
    del gap["rows"][3]["b"]["256"]                       # rec-3/q has no budget-256 row in beta
    holed = joint([A, gap])
    assert "rec-3/q" not in holed.fixed and len(holed.fixed) == N - 1 and len(base.fixed) == N
    assert all(x["n"] == N - 1 for x in holed.pairs) and all(x["n"] == N for x in base.pairs)
    assert not any("every point" in f for f in flags([A, gap]))
    # no question at every point of every model: empty intervals, flagged
    split = copy.deepcopy(A)
    for i, r in enumerate(split["rows"]):
        del r["b"]["0" if i % 2 else "256"]
    empty = joint([split])
    assert empty.fixed == () and all(x["n"] == 0 and x["dacc"] == [0.0, 0.0, 0.0] for x in empty.pairs)
    assert "no question was measured at every point of every model: paired intervals are empty" in flags([split])


def test_a_stratified_profile_with_bad_sources_falls_back_unstratified_and_flags():
    xs, ys = [float(i < 8) for i in range(N)], [float(i < 9) for i in range(N)]
    plain = list(paired_bootstrap(xs, ys, [f"g{i // 2}" for i in range(N)], seed=0))
    b = sourced(profile("beta", {0: 6, 256: 9}, {0: 400.0, 256: 2500.0}, key="k2", gib=9.0), SOURCES)
    missing = sourced(profile("alpha", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0}), SOURCES)
    del missing["rows"][0]["source"]
    got = {(x["a"], x["b"]): x["dacc"] for x in joint([missing, b]).pairs}[("alpha@256", "beta@256")]
    assert got == pytest.approx(plain)
    assert "rows without a source: paired intervals are unstratified" in flags([missing, b])
    spans = sourced(profile("alpha", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0}), ["big"] * 7 + ["small"] * 3)
    got = {(x["a"], x["b"]): x["dacc"] for x in joint([spans, b]).pairs}[("alpha@256", "beta@256")]   # g3 spans both
    assert got == pytest.approx(plain)
    assert "a cluster spans two sources: paired intervals are unstratified" in flags([spans, b])


# ---- the report and the CLI -------------------------------------------------------------------------------------

import callo.calibrate_think as ct  # noqa: E402
from callo.cli import main  # noqa: E402
from callo.compare import (  # noqa: E402
    render_md,
    report_dict,
    scatter_lines,
    serve_line,
    table_lines,
    write_report,
    write_stub_report,
)


def test_the_table_has_one_row_per_point_with_the_marks_and_the_sampler_column():
    slow = profile("slow", {0: 4}, {0: 900.0}, key="k3", gib=1.5)
    j = joint([A, B, slow])
    rows = table_lines(j)
    # the five statistics (a profile that predates mean, p99 and max has them recomputed from its rows), the
    # interval on the pick statistic, p99 `= max` below 100 questions, and the tails line under the table
    # the default pick statistic is mean: its column carries the interval
    assert rows[0] == "| point | acc | mean [95% CI] | p50 | p95 | p99 | max | think | forced | sampler | GiB | mark |"
    assert rows[2] == ("| alpha@0 | 0.500 | 0.31s [0.31s, 0.32s] | 0.30s | 0.45s | = max | 0.33s | - | - | "
                       "0.6/0.95/20*/0* | 4.0 | F |")
    assert rows[4] == ("| slow@0 | 0.400 | 0.94s [0.92s, 0.96s] | 0.90s | 1.35s | = max | 0.98s | - | - | "
                       "0.6/0.95/20*/0* | 1.5 | · by beta@0 |")
    assert rows[3].startswith("| beta@0 | 0.600 | 0.42s [") and rows[3].endswith("| 9.0 | F≈ |")   # within noise of alpha@0
    assert rows[5].startswith("| alpha@256 | 0.800 | 2.09s [") and rows[5].endswith("| 205 | 50% | 0.6/0.95/20*/0* | 4.0 | F≈ |")
    assert rows[6].startswith("| beta@256 | 0.900 | 2.61s [") and rows[6].endswith("| 9.0 | F≈ |")
    assert rows[7:] == ["", "tails (n 10): p95 from 1 question, p99 = max, max from 1"]
    v = table_lines(j, verbose=True)
    assert v[0].endswith("| mark | NLL@T | ECE10@T | T |") and v[2].endswith("| F | 0.650 | 0.080 | 1.20 |")
    nulled = copy.deepcopy(A)
    nulled["budgets"]["0"].update({"nll_T": None, "ece_T": None, "T": None})   # non-finite metrics are null
    assert table_lines(joint([nulled]), verbose=True)[2].endswith("| F | - | - | - |")
    p95 = table_lines(joint([A, B], pref=Preference(latency="p95")))
    assert p95[0] == "| point | acc | mean | p50 | p95 [95% CI] | p99 | max | think | forced | sampler | GiB | mark |"
    assert p95[2].startswith("| alpha@0 | 0.500 | 0.31s | 0.30s | 0.45s [0.42s, 0.48s] | = max | 0.33s |")
    near = profile("near", {0: 5, 256: 8}, {0: 310.0, 256: 2100.0}, key="k4")   # same accuracy, overlapping latency
    marks = {r.split(" | ")[0][2:]: r.rsplit(" | ", 1)[1][:-2]       # on p50, where near's intervals overlap A's
             for r in table_lines(joint([A, near], pref=Preference(latency="p50")))[2:6]}
    # identical correctness vectors: the paired Δacc is exactly 0 with a zero-width CI, so both near points are F≈
    assert marks == {"alpha@0": "F", "near@0": "F≈", "alpha@256": "F≈", "near@256": "F≈"}


def test_the_scatter_letters_points_in_latency_order_with_a_legend():
    j = joint([A, B])
    lines = scatter_lines(list(j.ranked), latency="p50")
    assert len(lines) == 13 + 2 + 1 and lines[0].startswith(" 0.90 |") and lines[12].startswith(" 0.50 |")
    assert lines[6].startswith(" 0.70 |")                                   # an odd height: the middle row is the midpoint
    assert lines[13].startswith("      +----") and "0.30s" in lines[14] and "2.50s" in lines[14] and "log scale" in lines[14]
    assert lines[15] == "a alpha@0  b beta@0  c alpha@256  d beta@256"
    grid = "".join(line[7:] for line in lines[:13])
    assert sorted(ch for ch in grid if ch.strip()) == ["a", "b", "c", "d"]
    assert lines[0].count("d") == 1 and lines[12].count("a") == 1           # beta@256 at the top, alpha@0 at the bottom
    assert all(line == line.rstrip() for line in lines)                      # no trailing spaces
    twin = profile("twin", {0: 5}, {0: 300.0}, key="k5")                    # the same cell as alpha@0
    both = scatter_lines(list(joint([A, twin]).ranked), latency="p50")
    assert "*" in both[12] and both[-1] == "* more than one point in that cell"
    assert scatter_lines([]) == ["(no points)"]
    flat = scatter_lines(list(joint([profile("f", {0: 5}, {0: 100.0})]).ranked), latency="p50")
    assert flat[0].startswith(" 0.60 |") and flat[12].startswith(" 0.50 |")


def test_the_scatters_labels_sit_on_the_rows_their_values_are_drawn_on():
    """A point at exactly a labelled value is drawn on that label's row, the middle one included,
    at the default height and at an even one (whose middle row is labelled with its own value)."""
    mid = profile("mid", {0: 5, 64: 7, 256: 9}, {0: 100.0, 64: 300.0, 256: 1000.0})
    pts = list(joint([mid]).ranked)
    for height in (13, 12, 7):
        lines = scatter_lines(pts, height=height)
        labelled = {line[:5].strip(): line for line in lines[:height] if line[:5].strip()}
        assert len(labelled) == 3
        for p, letter in zip(sorted(pts, key=lambda q: q.lat_p50[0]), "abc", strict=True):
            row = labelled.get(f"{p.acc:.2f}")
            if row is not None:
                assert letter in row[7:], (height, p.label, lines)
        assert "0.70" in labelled or height % 2 == 0
    # an even height labels its middle row with that row's own value (0.9 − 5/11 × 0.4 = 0.718): a point there is on it
    even = profile("even", {0: 5, 64: 7, 256: 9}, {0: 100.0, 64: 300.0, 256: 1000.0}, budgets__64__acc=0.72)
    lines = scatter_lines(list(joint([even]).ranked), height=12)
    assert lines[5].startswith(" 0.72 |") and "b" in lines[5][7:]


def test_render_md_and_report_json_carry_the_header_flags_table_pairs_pick_and_serve_line(tmp_path):
    hot = profile("alpha", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0}, pinned=(1.0, 0.95, 20, 0.0), policy="fixed",
                  capable=True)
    j = joint([A, B, hot], pref=Preference(max_gib=10.0))
    md = render_md(j, sources=["a.think.json", "b.think.json", "hot.think.json"])
    assert md.startswith("# callo compare: the joint think frontier\n\nlatency: single-stream, in-callo wall time, on Apple M3 "
                         "Max (3 profiles, 10 questions, data d.jsonl)\npolicies differ: alpha model, beta model, alpha#2 fixed.")
    assert "**FLAGS** (the points may not be comparable):\n- the same model (sha256:k1k1k1k1k1k1) sampled at 0.6/0.95/20/0 in alpha and 1/0.95/20/0 in alpha#2" in md
    assert "| alpha#2@256 | 0.800 |" in md
    assert ("Pick statistic mean; preference: --max-memory 10 (--max-memory compares the file size: a lower bound on "
            "memory).") in md
    assert ("Paired Δacc (b − a) over the shared questions [95 % cluster-bootstrap CI]:\n- alpha#2@0 → alpha@0: +0.0pp [+0.0, +0.0] "
            "(n=10, frontier neighbours)\n- alpha@0 → beta@0: +10.0pp [+0.0, +30.0] (n=10, frontier neighbours)\n") in md
    # the rest are (most accurate → point): the point minus M, so a less accurate point reads negative
    assert "- beta@256 → alpha@0: -40.0pp [-70.0, -10.0] (n=10, against the most accurate)" in md
    assert ("**pick: beta@0** (fastest within noise of the best, --max-memory 10); best beta@256; Δacc vs it "
            "-30.0pp [") in md
    # the fixture's profiles picked on p50: compare re-picks on mean, says so, and never writes back
    assert "alpha#2: the profile picked alpha@0 on p50; this pick is on mean" in md and A["pick"]["latency"] == "p50"
    # the profiles record no server args (attach mode): the serve line names --upstream, as calibrate's does
    assert "serve with: `callo serve --upstream URL --think-profile b.think.json --think-budget 0`" in md
    assert "## Runs\n\n| model | launched | -np | kv | ctx | cache-ram | server args | started | ended | decode tok/s |" in md
    assert "| alpha | menu | 4 | unified | 65536 | unknown | none | 2026-10-05T10:00:00Z | 2026-10-05T10:20:00Z | 100 |" in md
    runs = [{"name": "alpha", "status": "done"}, {"name": "zeta", "status": "failed", "reason": "did not start"},
            {"name": "eta", "status": "skipped", "reason": "needs ~70 GiB"}]
    with_runs = render_md(j, runs=runs)
    assert "Failed and skipped models:\n- zeta: failed (did not start)\n- eta: skipped (needs ~70 GiB)" in with_runs
    assert "Failed and skipped" not in md and "size unknown" not in md and "Not finished" not in with_runs
    unfinished = render_md(j, runs=[*runs, {"name": "iota", "status": "running"}, {"name": "kappa", "status": "pending"}])
    assert ("Failed and skipped models:\n- zeta: failed (did not start)\n- eta: skipped (needs ~70 GiB)\n\n"
            "Not finished (interrupted run):\n- iota: running\n- kappa: pending\n") in unfinished
    assert "**pick: " in md and "serve with: `callo serve --upstream URL" in md and "--think-profile " in md and "```\n 0.90 |" in md
    d = report_dict(j, sources=["a.think.json", "b.think.json", "hot.think.json"])
    assert d["callo_compare"] == 1 and d["latency"] == "mean" and d["filters"]["max_memory_gib"] == 10.0
    assert d["dacc"].startswith("point − reference: pairs[].dacc (and dscore) is b − a; pick.dacc_best (and "
                                "dscore_best) is pick − best (M, the best-scoring point")
    assert d["repick"] == [f"{n}: the profile picked {n.split('#')[0]}@0 on p50; this pick is on mean"
                           for n in ("alpha", "beta", "alpha#2")]
    assert [p["name"] for p in d["profiles"]] == ["alpha", "beta", "alpha#2"] and d["profiles"][2]["path"] == "hot.think.json"
    assert d["profiles"][2]["policy"] == "fixed" and d["profiles"][0]["sampler"] == "0.6/0.95/20*/0*"
    assert {p["label"] for p in d["points"]} == {"alpha@0", "alpha@256", "beta@0", "beta@256", "alpha#2@0", "alpha#2@256"}
    assert all({"label", "acc", "lat_p50_ms", "frontier", "approx", "dominated_by", "gib"} <= set(p) for p in d["points"])
    assert d["pairs"] == [dict(x) for x in j.pairs] and d["pick"]["serve"] == serve_line(j, ["a.think.json", "b.think.json", "hot.think.json"])
    assert d["pick"]["serve"].startswith("callo serve --upstream URL ") and d["flags"] and d["notes"] == [
        f"{n}: 10 questions (< 100): its intervals are wide" for n in ("alpha", "beta", "alpha#2")]
    assert d["pick"]["dacc_best"] == list(j.dacc_best) and d["pick"]["dacc_best"][0] == pytest.approx(-0.3)
    assert d["pick"]["unsized"] == [] and d["runs"] == [] and d["profiles"][0]["cache_ram"] is None
    assert d["profiles"][0]["total_slots"] == 4 and d["profiles"][0]["prefill_reused"] == {"256": None}
    assert report_dict(j, runs=runs)["runs"] == runs
    json.dumps(d)
    none = joint([A, B], pref=Preference(max_latency_s=0.01))
    assert none.chosen.point is None and serve_line(none) is None
    assert ("**pick: none** (no point meets --limit mean=0.01s: all 4 excluded; nearest alpha@0 at mean 0.31 s)"
            in render_md(none))
    md_path, js_path = write_report([A, B], tmp_path / "out", sources=["a", "b"])
    assert md_path == tmp_path / "out" / "report.md" and json.loads(js_path.read_text())["profiles"][1]["path"] == "b"
    assert md_path.read_text() == render_md(joint([A, B]), sources=["a", "b"])
    stub = write_stub_report(tmp_path / "stub", runs=[{"name": "x", "status": "failed", "reason": "no gguf"}])
    assert stub.read_text().startswith("# callo compare: the joint think frontier\n\nNo profile was written")
    assert "## Runs\n\nFailed and skipped models:\n- x: failed (no gguf)\n" in stub.read_text()
    halted = write_stub_report(tmp_path / "halted", runs=[{"name": "y", "status": "pending"}])
    assert "Not finished (interrupted run):\n- y: pending" in halted.read_text() and "Failed" not in halted.read_text()
    assert json.loads((tmp_path / "stub" / "report.json").read_text())["runs"] == [{"name": "x", "status": "failed", "reason": "no gguf"}]


def test_the_header_says_unknown_host_and_the_pick_names_points_of_unknown_size():
    """A None host fingerprint is never a match, so the header names no chip; `Pick.unsized` is printed under the
    pick line (both branches) with calibrate's own `pick_note`, and kept in report.json."""
    unknown = profile("gamma", {0: 5}, {0: 300.0}, key="k3", environment__host__fingerprint=None)
    assert render_md(joint([A, unknown])).split("\n")[2].startswith(
        "latency: single-stream, in-callo wall time, on unknown host (2 profiles")
    assert render_md(joint([unknown])).split("\n")[2].startswith("latency: single-stream, in-callo wall time, on unknown host")
    assert "on 2 different hosts" in render_md(joint([A, profile("h", {0: 5}, {0: 1.0}, key="k4", fingerprint="x" * 64)]))
    unsized = profile("big", {0: 9, 256: 10}, {0: 250.0, 256: 900.0}, key="k7", environment__model__bytes=None)
    j = joint([A, unsized], pref=Preference(max_gib=10.0))
    assert j.chosen.unsized == ["big@0", "big@256"] and j.chosen.point is not None
    lines = render_md(j).split("\n")
    at = next(i for i, line in enumerate(lines) if line.startswith("**pick: "))
    assert lines[at + 1] == "size unknown: excluded by --max-memory: big@0, big@256"
    assert lines[at + 1] == ct.pick_note(j.chosen.as_dict())
    assert lines[at + 2:at + 4] == ["alpha: the profile picked alpha@0 on p50; this pick is on mean",
                                    "big: the profile picked big@0 on p50; this pick is on mean"]
    assert lines[at + 4].startswith("serve with: ")
    assert report_dict(j)["pick"]["unsized"] == ["big@0", "big@256"]
    none = joint([unsized], pref=Preference(max_gib=10.0))
    lines = render_md(none).split("\n")
    at = next(i for i, line in enumerate(lines) if line.startswith("**pick: none**"))
    assert lines[at + 1] == "size unknown: excluded by --max-memory: big@0, big@256"
    assert not any(line.startswith("serve with:") for line in lines)


def test_the_serve_line_is_calibrates_own_command():
    """One source of truth. For a single profile whose pick is the joint's, compare's serve line is calibrate's
    `serve with:` line; `--model FILE` only when callo launched the server, `--think-max` above serve's default,
    `--no-cache` when measured without the cache (on by default since 0.3.0)."""
    def own(p: dict) -> dict:
        p = copy.deepcopy(p)
        p["pick"] = joint([p]).chosen.as_dict()
        return p

    for p in (own(A), own(profile("m", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0}, environment__server__server_args=["-c", "4096"])),
              own(profile("c", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0}, comparable__cache=False))):
        assert "serve with: " + serve_line(joint([p]), ["w.think.json"]) == ct.serve_line(p, "w.think.json")
    assert serve_line(joint([A]), ["w"]).startswith("callo serve --upstream URL --think-profile w --think-budget ")
    launched = profile("m", {0: 5}, {0: 300.0}, environment__server__server_args=["-c", "4096"])
    assert serve_line(joint([launched]), ["w"]) == "callo serve --model m.gguf --think-profile w --think-budget 0"
    uncached = profile("c", {0: 5}, {0: 300.0}, comparable__cache=False)
    assert serve_line(joint([uncached])) == \
        "callo serve --upstream URL --think-profile c.think.json --think-budget 0 --no-cache"
    # the PICK's budget, not the profile's own pick: a deep budget above serve's 4096 default needs --think-max
    deep = profile("d", {0: 2, 8192: 10}, {0: 300.0, 8192: 9000.0})
    j = joint([deep])
    assert j.chosen.point is not None and j.chosen.point.budget == 8192 and deep["pick"]["budget"] == 0
    assert serve_line(j, ["w"]) == "callo serve --upstream URL --think-profile w --think-budget 8192 --think-max 8192"


def test_the_serve_line_shell_quotes_the_model_file_and_the_profile_path():
    """Runnable as printed with a space in a path, in compare's line and calibrate's alike."""
    spaced = profile("m", {0: 5}, {0: 300.0}, environment__server__server_args=["-c", "4096"],
                     environment__model__file="my model.gguf")
    line = serve_line(joint([spaced]), ["/tmp/think out/m.think.json"])
    assert line == "callo serve --model 'my model.gguf' --think-profile '/tmp/think out/m.think.json' --think-budget 0"
    assert ct.serve_line(spaced, "/tmp/think out/m.think.json") == "serve with: " + line


def test_the_compare_cli_prints_markdown_or_json_writes_out_and_refuses(tmp_path, capsys):
    pa, pb = tmp_path / "a.think.json", tmp_path / "b.think.json"
    pa.write_text(json.dumps(A))
    pb.write_text(json.dumps(B))
    assert main(["compare", str(pa), str(pb)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# callo compare") and "| beta@256 |" in out and "serve with: `callo serve --upstream URL " in out
    assert f"--think-profile {pa} " in out or f"--think-profile {pb} " in out      # the pick's own profile path
    assert main(["compare", str(pa), str(pb), "--json", "--out", str(tmp_path / "rep"), "--latency", "p95", "--verbose"]) == 0
    captured = capsys.readouterr()
    d = json.loads(captured.out)
    assert d["latency"] == "p95" and (tmp_path / "rep" / "report.md").exists() and "report:" in captured.err
    assert "| NLL@T |" in (tmp_path / "rep" / "report.md").read_text()
    other = tmp_path / "c.think.json"
    other.write_text(json.dumps(profile("gamma", {0: 5}, {0: 300.0}, sha="e" * 64)))
    assert main(["compare", str(pa), str(other)]) == 2 and "refusing to compare" in capsys.readouterr().err
    assert main(["compare", str(tmp_path / "missing.json")]) == 2 and "cannot read" in capsys.readouterr().err
    for bad in (["--max-latency", "0"], ["--min-accuracy", "2"], ["--max-memory", "-1"]):
        assert main(["compare", str(pa), *bad]) == 2
        assert bad[0] in capsys.readouterr().err
    assert main(["compare", str(pa), "--max-memory", "1"]) == 0           # a single profile is a valid input
    assert "**pick: none**" in capsys.readouterr().out


def _malformed(how: str) -> dict:
    p = copy.deepcopy(A)
    if how == "no q":
        del p["rows"][2]["q"]
    elif how == "b not a dict":
        p["rows"][2]["b"] = [1, 0]
    elif how == "no correct field":
        p["row_fields"] = ["nll1", "pmax"]
    elif how == "short row":
        p["rows"][2]["b"]["0"] = []
    elif how == "bad budget":
        p["rows"][2]["b"]["x"] = p["rows"][2]["b"]["0"]
    elif how == "budget entry":
        del p["budgets"]["256"]["acc"]
    return p


@pytest.mark.parametrize("how, says", [("no q", "row 2 has no question id"), ("b not a dict", "row 2 (rec-2/q): b is not"),
                                       ("no correct field", 'row_fields has no "correct"'),
                                       ("short row", "row 2 (rec-2/q): budget 0 has no correct value"),
                                       ("bad budget", "row 2 (rec-2/q): budget 'x' is not an integer"),
                                       ("budget entry", "budgets.256.acc is not a finite number")])
def test_a_malformed_profile_is_a_one_line_refusal_naming_the_file(tmp_path, capsys, how, says):
    """A malformed row must not escape as a KeyError/ValueError traceback from `correct_at`."""
    path = tmp_path / "bad.think.json"
    path.write_text(json.dumps(_malformed(how)))
    with pytest.raises(InvalidInput, match="malformed think profile"):
        load_profile(path)
    pa = tmp_path / "a.think.json"
    pa.write_text(json.dumps(A))
    assert main(["compare", str(pa), str(path)]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith(f"callo: {path}: malformed think profile") and says in err
    assert "Traceback" not in err


def _set(doc: dict, path: str, value) -> dict:
    doc = copy.deepcopy(doc)
    *node, leaf = path.split(".")
    target = doc
    for k in node:
        target = target[k] if not isinstance(target, list) else target[int(k)]
    if isinstance(target, list):
        target[int(leaf)] = value
    else:
        target[leaf] = value
    return doc


@pytest.mark.parametrize("path, value, says", [
    ("budgets.0.lat_p50_ms", ["a", "b", "c"], "budgets.0.lat_p50_ms is not three finite numbers"),
    ("budgets.256.lat_p95_ms", [float("nan"), 1.0, 2.0], "budgets.256.lat_p95_ms is not three finite numbers"),
    ("budgets.0.lat_p50_ms", [float("inf"), 1.0, 2.0], "budgets.0.lat_p50_ms is not three finite numbers"),
    ("budgets.0.acc", "0.5", "budgets.0.acc is not a finite number"),
    ("budgets.0.acc", float("nan"), "budgets.0.acc is not a finite number"),
    ("budgets.256.think_mean", "205", "budgets.256.think_mean is not a finite number or null"),
    ("budgets.256.forced", float("inf"), "budgets.256.forced is not a finite number or null"),
    ("environment", [], "environment is not a JSON object"),
    ("environment.model", "alpha.gguf", "environment.model is not a JSON object"),
    ("environment.server", 4, "environment.server is not a JSON object"),
    ("thinking", True, "thinking is not a JSON object"),
    ("drift", "none", "drift is not a JSON object"),
    ("drift.therm.0", "A 100", "drift.therm is not a list of JSON objects"),
    ("audit", [], "audit is not a JSON object"),
    ("server_state", "clean", "server_state is not a JSON object"),
    ("environment.model.bytes", "4096", "environment.model.bytes is not an integer or null"),
    ("environment.model.bytes", True, "environment.model.bytes is not an integer or null"),
    ("environment.model.mmproj_bytes", "1", "environment.model.mmproj_bytes is not an integer or null"),
    ("environment.model.mmproj_bytes", 1.5, "environment.model.mmproj_bytes is not an integer or null"),
])
def test_a_profile_value_of_the_wrong_type_is_a_one_line_refusal_naming_the_field(tmp_path, capsys, path, value, says):
    """Value types, not only structure; every one of these escaped as a traceback before."""
    bad = tmp_path / "bad.think.json"
    bad.write_text(json.dumps(_set(A, path, value)))              # json writes NaN / Infinity, as Python's json reads them
    pa = tmp_path / "a.think.json"
    pa.write_text(json.dumps(A))
    assert main(["compare", str(pa), str(bad)]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"callo: {bad}: malformed think profile: {says}")
    assert err.count("\n") == 1 and "Traceback" not in err


@pytest.mark.parametrize("where, key, says", [
    ("budgets", " 0", "budget ' 0' is not an integer"), ("budgets", "00", "budget '00' is not an integer"),
    ("budgets", "x", "budget 'x' is not an integer"), ("rows", " 0", "row 0 (rec-0/q): budget ' 0' is not an integer"),
    ("rows", "00", "row 0 (rec-0/q): budget '00' is not an integer")])
def test_a_budget_key_must_be_written_as_calibrate_writes_it(tmp_path, capsys, where, key, says):
    """`str(int(b)) == b`; a non-canonical key used to pass and then miss `budgets[str(int(b))]`."""
    doc = copy.deepcopy(A)
    holder = doc["budgets"] if where == "budgets" else doc["rows"][0]["b"]
    holder[key] = holder.pop("0")
    bad = tmp_path / "bad.think.json"
    bad.write_text(json.dumps(doc))
    assert main(["compare", str(bad)]) == 2
    err = capsys.readouterr().err
    assert err == f"callo: {bad}: malformed think profile: {says}\n"


def test_the_scatter_needs_room_for_its_three_labels():
    with pytest.raises(ValueError, match="at least 3"):
        scatter_lines(list(joint([A]).ranked), height=2)
    assert len(scatter_lines(list(joint([A]).ranked), height=3)) == 3 + 2 + 1


def test_null_where_null_is_allowed_and_absent_sections_still_compare(tmp_path, capsys):
    ok = _set(_set(A, "budgets.256.think_mean", None), "budgets.256.forced", None)
    for k in ("thinking", "drift", "audit", "server_state"):
        ok[k] = None
    del ok["environment"]["run"]
    path = tmp_path / "ok.think.json"
    path.write_text(json.dumps(ok))
    assert main(["compare", str(path)]) == 0 and "| alpha@256 | 0.800 |" in capsys.readouterr().out


def test_an_out_that_cannot_be_written_is_a_one_line_exit_2(tmp_path, capsys):
    """`--out` naming an existing file, or a DIR that cannot be written, is exit 2 without a traceback."""
    pa = tmp_path / "a.think.json"
    pa.write_text(json.dumps(A))
    assert main(["compare", str(pa), "--out", str(pa)]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith(f"callo: --out {pa}: cannot write the report there (") and captured.err.count("\n") == 1
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert main(["compare", str(pa), "--out", str(locked / "rep")]) == 2
        err = capsys.readouterr().err
        assert err.startswith(f"callo: --out {locked / 'rep'}: cannot write the report there (") and "Traceback" not in err
    finally:
        locked.chmod(0o700)


def test_the_scatter_defaults_to_the_mean_like_the_pick():
    j = joint([A, B])
    assert scatter_lines(list(j.ranked)) == scatter_lines(list(j.ranked), latency="mean")
