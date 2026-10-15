"""Pick preferences: the one picker over budget and deadline points (early-exit points are
listed, never picked): the step-1 filters (the serve check, early exit, limits, memory), the price's target M and the
noise rule against it, the `pick` block, the no-pick reasons and the `--deadlines` hint. Synthetic points throughout; the Δacc CIs are planted (`Noise`)."""
import json
import random

import pytest

from callo.calibrate_think import budget_marks
from callo.think import (
    EE_OPT_IN,
    Limit,
    Point,
    Preference,
    frontier,
    mark_approx,
    on_frontier,
    parse_limit,
    pick,
)


def P(name, kind, value, acc, mean, *, p50=None, p95=None, p99=None, mx=None, w=0.0, n=376, **kw):
    """A point in ms: p50 / p95 / p99 / max default to mean × 0.8 / 1.6 / 2.0 / 2.4; `w` the half-width of every
    interval."""
    p50, p95 = p50 if p50 is not None else 0.8 * mean, p95 if p95 is not None else 1.6 * mean
    p99, mx = p99 if p99 is not None else 2.0 * mean, mx if mx is not None else 2.4 * mean
    return Point(name=name, kind=kind, value=value, budget=value if kind == "budget" else None, acc=acc, n=n,
                 lat_mean=(mean, mean - w, mean + w), lat_p50=(p50, p50 - w, p50 + w), lat_p95=(p95, p95 - w, p95 + w),
                 lat_p99=(p99, p99 - w, p99 + w), lat_max=mx, **kw)


def B(value, acc, mean, name="m", **kw):
    return P(name, "budget", value, acc, mean, **kw)


class Noise:
    """`think.Paired`'s interface with planted Δacc CIs: `widths[frozenset((a, b))]` is a pair's half-width (default
    `w`); Δscore is 100 × Δacc minus the priced latency difference (reported only)."""

    def __init__(self, w=0.0, widths=None):
        self.w, self.widths = w, widths or {}

    def dacc(self, a, b):
        d = b.acc - a.acc
        h = self.widths.get(frozenset((a.label, b.label)), self.w)
        return (d, d - h, d + h)

    def dscore(self, a, b, *, stat, price):
        dl = price * (b.latency(stat)[0] - a.latency(stat)[0]) / 1000
        return tuple(100 * x - dl for x in self.dacc(a, b))


# ---- R = 0 is the original pick rule; what is monotone in R and what is not ------------------------------------

def tb_pick(points, noise, latency):
    """The original rule, as earlier callo versions picked: the most accurate frontier point M, the fastest frontier point within
    noise of it."""
    front = mark_approx(on_frontier(frontier(points, latency=latency)), noise.dacc)
    best = min(front, key=lambda p: (-p.acc, p.latency(latency)[0]))
    for p in front:
        d = (0.0, 0.0, 0.0) if p is best else noise.dacc(best, p)
        if p is best or d[1] <= 0 <= d[2]:
            return p, best, d
    raise AssertionError("unreachable")


def random_set(rng):
    pts = []
    for name in ("a", "b", "c")[: rng.randint(1, 3)]:
        for v in sorted(rng.sample([0, 128, 256, 512, 1024, 2048], rng.randint(1, 5))):
            pts.append(B(v, rng.choice([0.70, 0.75, 0.80, 0.85, 0.90]), rng.uniform(200, 9000), name=name,
                         p50=rng.uniform(200, 9000), p95=rng.uniform(200, 20000), w=rng.choice([0.0, 50.0, 500.0])))
    widths = {frozenset((x.label, y.label)): rng.choice([0.01, 0.06, 0.2]) for x in pts for y in pts}
    return pts, Noise(widths=widths)


@pytest.mark.parametrize("stat", ["mean", "p50", "p95"])
def test_at_price_0_the_pick_is_exactly_tb_2_5s(stat):
    rng = random.Random(stat)
    for _ in range(150):
        pts, noise = random_set(rng)
        got = pick(pts, paired=noise, pref=Preference(latency=stat))
        want, best, d = tb_pick(pts, noise, stat)
        assert (got.point.label, got.best.label, got.dacc_best) == (want.label, best.label, d)
        assert got.rule == "fastest within noise of the best" and got.score is None and got.dscore_best is None


@pytest.mark.parametrize("stat", ["mean", "p50", "p95"])
def test_the_targets_latency_never_rises_with_the_price_and_the_pick_is_never_slower_than_it(stat):
    rng = random.Random(f"monotone {stat}")
    for _ in range(80):
        pts, noise = random_set(rng)
        last = None
        for r in (0, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000):
            c = pick(pts, paired=noise, pref=Preference(latency=stat, price=r))
            m = c.best.latency(stat)[0]
            assert c.point.latency(stat)[0] <= m
            assert last is None or m <= last
            last = m


def test_the_pick_itself_is_not_monotone_in_the_price():
    """The counterexample: A .73 at 1 s, B .77 at 2 s, C .78 at 3 s; A is within noise of C (a wide CI) but not of
    B (a narrow one). R = 0 aims at C and picks A, R = 2 aims at B and picks B, R = 5 aims at A and picks A."""
    a, b, c = B(1, 0.73, 1000), B(2, 0.77, 2000), B(3, 0.78, 3000)
    noise = Noise(w=0.0, widths={frozenset(("m@1", "m@3")): 0.08, frozenset(("m@1", "m@2")): 0.01,
                                 frozenset(("m@2", "m@3")): 0.03})
    got = [pick([a, b, c], paired=noise, pref=Preference(price=r)) for r in (0, 2, 5)]
    assert [g.point.label for g in got] == ["m@1", "m@2", "m@1"]
    assert [g.best.label for g in got] == ["m@3", "m@2", "m@1"]
    assert got[1].score == {"pick": pytest.approx(73.0), "best": pytest.approx(73.0)}


def test_under_a_price_best_is_the_target_and_most_accurate_is_named_apart():
    zero, mid, top = B(0, 0.83, 270), B(512, 0.936, 1830), B(1024, 0.941, 1910)
    c = pick([zero, mid, top], paired=Noise(w=0.006), pref=Preference(price=10.0))
    assert c.best.label == "m@0" and c.most_accurate.label == "m@1024" and c.point.label == "m@0"
    assert c.rule == "fastest within noise of the best score" and c.dacc_best == (0.0, 0.0, 0.0)
    c = pick([zero, mid, top], paired=Noise(w=0.006), pref=Preference(price=1.0))
    assert (c.best.label, c.point.label) == ("m@1024", "m@512")
    assert c.dacc_best == pytest.approx((0.936 - 0.941, 0.936 - 0.941 - 0.006, 0.936 - 0.941 + 0.006))   # pick − M
    assert c.score == {"pick": pytest.approx(93.6 - 1.83), "best": pytest.approx(94.1 - 1.91)}
    assert c.dscore_best == pytest.approx((-0.5 + 0.08, -0.5 - 0.6 + 0.08, -0.5 + 0.6 + 0.08))
    d = c.as_dict()
    assert (d["best"], d["most_accurate"], d["rule"]) == ("m@1024", "m@1024", "fastest within noise of the best score")
    assert d["preference"]["price_pp_per_s"] == 1.0 and d["score"] == c.score and d["dscore_best"] == list(c.dscore_best)


# ---- the step-1 filters ---------------------------------------------------------------------------------------

def test_a_limit_filters_on_the_point_estimate_before_the_frontier():
    fast = B(256, 0.92, 1640, p99=3040, mx=3080)
    slow = B(1024, 0.95, 1600, p99=10000, mx=11440)          # dominates `fast` on mean, but fails the p99 limit
    assert pick([fast, slow], paired=Noise()).point.label == "m@1024"
    c = pick([fast, slow], paired=Noise(), pref=Preference(limits=(parse_limit("p99=6s"),)))
    assert c.point.label == "m@256" and c.best.label == "m@256" and [p.label for p in c.front] == ["m@256"]
    assert c.excluded == {"m@1024": "p99 10.00 s > 6 s"}
    assert c.as_dict()["preference"]["limits"] == [{"stat": "p99", "le_ms": 6000.0, "from": "--limit"}]


def test_max_latency_is_a_limit_on_the_pick_statistic_and_repeated_limits_all_apply():
    pts = [B(0, 0.8, 300), B(256, 0.9, 1500), B(1024, 0.95, 4000)]
    via_alias = pick(pts, paired=Noise(), pref=Preference(latency="p50", max_latency_s=2.0))
    via_limit = pick(pts, paired=Noise(), pref=Preference(latency="p50", limits=(parse_limit("p50=2s"),)))
    assert via_alias.point.label == via_limit.point.label == "m@256"            # p50 = 0.8 × mean: 1.2 s fits
    assert via_alias.as_dict()["preference"]["limits"] == [{"stat": "p50", "le_ms": 2000.0, "from": "--max-latency"}]
    assert via_alias.as_dict()["max_latency_s"] == 2.0 and via_limit.as_dict()["max_latency_s"] is None
    two = pick(pts, paired=Noise(), pref=Preference(limits=(parse_limit("mean=5s"), parse_limit("mean=1s"))))
    assert two.point.label == "m@0" and two.excluded == {"m@256": "mean 1.50 s > 1 s", "m@1024": "mean 4.00 s > 1 s"}


def test_an_early_exit_point_is_never_picked_even_when_it_dominates():
    """Early exit is opt-in (a held-out check: its rule holds only on questions like the calibration data), so the pick
    never chooses an early-exit point, under any preference, and it never eliminates the right pick."""
    zero, top = B(0, 0.83, 270, gib=5.0), B(1024, 0.94, 1910, gib=5.0)
    ee = P("m", "ee", 0.95, 0.95, 900, cap=1024, gib=5.0)        # dominates m@1024: faster and more accurate
    alone = pick([zero, top], paired=Noise(w=0.005))
    c = pick([zero, top, ee], paired=Noise(w=0.005))
    assert (c.point.label, c.best.label, c.dacc_best) == (alone.point.label, alone.best.label, alone.dacc_best)
    assert c.excluded == {"m@ee.95": EE_OPT_IN} and ee not in c.front
    assert (c.would_pick, c.would_best, c.as_dict()["z"]) == (None, None, None)
    for pref in (Preference(price=50.0), Preference(min_accuracy=0.9), Preference(latency="p50"),
                 Preference(limits=(parse_limit("p99=3s"),)), Preference(max_latency_s=1.0), Preference(max_gib=10.0)):
        c = pick([zero, top, ee], paired=Noise(w=0.005), pref=pref)
        assert c.point is None or c.point.kind != "ee", pref
        assert c.best is None or c.best.kind != "ee", pref


def test_memory_and_unknown_sizes_are_step_1_filters():
    big, small, unknown = B(0, 0.95, 300, name="big", gib=60.0), B(0, 0.9, 320, name="small", gib=5.0), \
        B(0, 0.97, 280, name="unk")
    c = pick([big, small, unknown], paired=Noise(), pref=Preference(max_gib=10.0))
    assert c.point.label == "small@0" and c.unsized == ["unk@0"]
    assert c.excluded == {"big@0": "60.0 GiB > 10 GiB", "unk@0": "size unknown (--max-memory)"}


# ---- deadline points ---------------------------------------------------------------------------------------------

def test_a_deadline_point_competes_under_a_max_limit_and_its_pick_block_names_the_deadline():
    zero = B(0, 0.83, 270, mx=460)
    b256 = B(256, 0.92, 1640, mx=3080)
    dl = P("m", "deadline", 2000, 0.90, 1500, mx=1990, cap=1024)
    c = pick([zero, b256, dl], paired=Noise(w=0.005), pref=Preference(limits=(parse_limit("max=2s"),)))
    assert c.point.label == "m@2000ms" and c.excluded == {"m@256": "max 3.08 s > 2 s"}
    d = c.as_dict()
    assert (d["kind"], d["budget"], d["budgets"], d["deadline_ms"], d["cap"]) == ("deadline", None, [], 2000, 1024)
    assert d["label"] == "m@2000ms" and d["model"] == "m"


def _hint_case(**over):
    """The documented second example in miniature: max=2s excludes the positive budget, budget 0 passes it but misses
    --min-accuracy, and the model holds no deadline point."""
    pts = [B(0, 0.832, 270, mx=460), B(256, 0.92, 1640, mx=3080)] + over.pop("extra", [])
    pref = Preference(limits=over.pop("limits", (parse_limit("max=2s"),)), min_accuracy=over.pop("min_accuracy", 0.9))
    return pick(pts, paired=Noise(w=0.005), pref=pref)


HINT = ("to add a point that meets it: a new calibrate run with --deadlines 2000 (a run without --deadlines cannot "
        "resume with them)")


def test_the_deadlines_hint_appears_only_when_all_four_conditions_hold():
    c = _hint_case()
    assert c.point is None and c.reason == ("--limit max=2s excluded 1 of 2, nearest m@256 at max 3.08 s; "
                                            f"--min-accuracy .9: m@0 is .832; {HINT}")
    assert _hint_case(min_accuracy=0.8).reason is None                                    # (1) there is a pick
    with_deadline = _hint_case(extra=[P("m", "deadline", 1500, 0.95, 1400, mx=2600, cap=1024)])
    assert with_deadline.point is None and HINT not in with_deadline.reason                # (2) a deadline point exists
    zero_fails = _hint_case(limits=(parse_limit("max=2s"), parse_limit("mean=0.1s")))
    assert zero_fails.point is None and "--deadlines" not in zero_fails.reason            # (3) budget 0 fails a limit
    mean_only = _hint_case(limits=(parse_limit("mean=1s"),))
    assert mean_only.point is None and "--deadlines" not in mean_only.reason               # (4) no max / p99 limit


def test_the_hint_is_per_model_and_names_the_binding_limit_in_ms():
    m = [B(0, 0.832, 270, mx=460), B(256, 0.92, 1640, mx=3080, p99=3040)]
    q = [B(0, 0.80, 300, name="q", mx=500), B(256, 0.95, 1700, name="q", mx=3500),
         P("q", "deadline", 1500, 0.9, 1400, mx=2600, cap=512)]
    pref = Preference(limits=(parse_limit("p99=2.5s"), parse_limit("max=2s")), min_accuracy=0.99)
    c = pick(m + q, paired=Noise(w=0.005), pref=pref)
    # m qualifies (q holds a deadline point); the value is the tightest max / p99 limit, 2 s, although p99=2.5s is the
    # first filter to exclude m@256
    assert c.point is None and c.reason.endswith(HINT)
    first_mean = _hint_case(limits=(parse_limit("mean=1s"), parse_limit("max=2s")))
    assert first_mean.excluded["m@256"].startswith("mean ") and first_mean.reason.endswith(HINT)   # max fails it too


def test_the_hint_never_suggests_a_deadline_the_run_measured_and_skipped():
    """A deadline calibrate measured and made no point of (every question fell back) cannot be
    suggested again, nor any shorter one (a shorter deadline never thinks more); a longer skip suppresses it too, and a
    shorter skip, or another model's, does not."""
    def case(skipped):
        pts = [B(0, 0.832, 270, mx=460), B(256, 0.92, 1640, mx=3080)]
        pref = Preference(limits=(parse_limit("max=2s"),), min_accuracy=0.9)
        return pick(pts, paired=Noise(w=0.005), pref=pref, skipped=skipped)

    assert case(None).reason.endswith(HINT) and case({}).reason.endswith(HINT)
    for skipped in ({"m": {2000}}, {"m": {3000}}, {"m": {900, 2500}}):
        c = case(skipped)
        assert c.point is None and "--deadlines" not in c.reason, skipped
        assert c.reason == "--limit max=2s excluded 1 of 2, nearest m@256 at max 3.08 s; --min-accuracy .9: m@0 is .832"
    assert case({"m": {1500}}).reason.endswith(HINT) and case({"q": {2000}}).reason.endswith(HINT)


def test_a_no_pick_reason_leaves_early_exit_out_and_names_the_memory_miss():
    pts = [B(0, 0.832, 270, p99=440), B(256, 0.92, 1640, p99=3040), P("m", "ee", 0.95, 0.95, 200, p99=300)]
    c = pick(pts, paired=Noise(), pref=Preference(limits=(parse_limit("p99=0.3s"),)))
    assert c.point is None            # the ee point meets the limit but is never picked, and is not counted
    assert c.reason == "no point meets --limit p99=0.3s: all 2 excluded; nearest m@0 at p99 0.44 s, from 4 of 376"
    big = [B(0, 0.9, 300, name="big", gib=60.0), B(0, 0.8, 300, name="mid", gib=12.0), B(0, 0.7, 300, name="unk")]
    c = pick(big, paired=Noise(), pref=Preference(max_gib=10.0))
    assert c.reason == "no point meets --max-memory 10: all 3 excluded; nearest mid@0 at 12.0 GiB"
    assert pick([P("m", "ee", 0.95, 0.95, 200)], paired=Noise()).reason == \
        "no point to pick: early exit is opt-in, never picked"


def test_the_stored_budget_marks_ignore_deadline_and_early_exit_points():
    """`budgets[B]`'s marks are over budget points only, so adding a deadline point
    (even one that dominates a budget) changes none of them."""
    budgets = [B(0, 0.83, 270), B(256, 0.92, 1640), B(512, 0.936, 1830)]
    dl = P("m", "deadline", 1500, 0.95, 1000, cap=512)
    noise = Noise(w=0.01)
    alone = budget_marks(budgets, noise.dacc, latency="mean", paired=True)
    assert set(alone) == {"m@0", "m@256", "m@512"} and alone["m@512"]["frontier"] is True
    assert budget_marks([*budgets, dl], noise.dacc, latency="mean", paired=True) == alone
    assert frontier([*budgets, dl])[-1].dominated_by == "m@1500ms"                # the displayed frontier: it dominates
    assert budget_marks(budgets, noise.dacc, latency="mean", paired=False)["m@0"]["dacc_best"] is None


def test_no_pick_reasons_name_each_filter_its_count_and_the_nearest_miss():
    pts = [B(0, 0.832, 270, p99=440), B(256, 0.92, 1640, p99=3040), B(512, 0.936, 1830, p99=5800)]
    c = pick(pts, paired=Noise(), pref=Preference(limits=(parse_limit("p99=0.3s"),)))
    assert c.point is None and c.rule == "no pick" and c.best is None
    assert c.reason == "no point meets --limit p99=0.3s: all 3 excluded; nearest m@0 at p99 0.44 s, from 4 of 376"
    assert pick([], paired=Noise()).reason == "no points"


def test_the_pick_block_round_trips_through_json_with_the_old_readers_keys():
    c = pick([B(0, 0.8, 300), B(256, 0.9, 1500)], paired=Noise(w=0.2), pref=Preference(latency="p95", max_latency_s=9.0))
    d = json.loads(json.dumps(c.as_dict()))
    assert list(d) == ["kind", "budget", "budgets", "deadline_ms", "cap", "z", "model", "label", "rule", "best",
                       "most_accurate", "dacc_best", "dscore_best", "score", "preference", "excluded", "reason",
                       "unsized", "would_pick", "would_best", "latency", "max_latency_s", "min_accuracy"]
    assert d["z"] is None
    assert (d["latency"], d["max_latency_s"], d["min_accuracy"]) == ("p95", 9.0, None)
    assert d["preference"] == {"latency": "p95", "price_pp_per_s": 0.0, "min_accuracy": None, "max_memory_gib": None,
                               "limits": [{"stat": "p95", "le_ms": 9000.0, "from": "--max-latency"}]}
    assert Preference.from_dict(d["preference"]) == Preference(latency="p95", max_latency_s=9.0)
    assert Limit("p95", 9000.0, "--max-latency").canonical == "--limit p95=9s"


def test_an_early_exit_point_of_unknown_size_is_not_named_unsized():
    zero, ee = B(0, 0.9, 300, gib=5.0), P("m", "ee", 0.95, 0.95, 250)         # the ee point has no size
    c = pick([zero, ee], paired=Noise(), pref=Preference(max_gib=10.0))
    assert c.point.label == "m@0" and c.unsized == [] and c.excluded["m@ee.95"] == EE_OPT_IN


def test_an_early_exit_point_serve_cannot_run_is_excluded_first_with_its_reason():
    """An early-exit point `callo serve` cannot run as measured (the caller's `unservable`, from the
    profile's convergence block) is excluded before the limits, so the pick never names a command serve would refuse
    or run without early exit."""
    zero, top = B(0, 0.83, 270), B(1024, 0.94, 1910)
    ee = P("m", "ee", 0.95, 0.95, 900, cap=1024)                   # dominates m@1024 (see above)
    why = {"m@ee.95": "callo serve cannot run it: no early-exit rule reaches z .95 (serve would run without early "
                      "exit)"}
    c = pick([zero, top, ee], paired=Noise(w=0.005), unservable=why)
    assert c.point.label == "m@1024" and c.excluded == why
    c = pick([zero, top, ee], paired=Noise(w=0.005), pref=Preference(limits=(parse_limit("mean=0.1s"),)),
             unservable=why)
    assert c.excluded["m@ee.95"] == why["m@ee.95"]                 # the serve check comes first
    assert pick([ee], paired=Noise(), unservable=why).reason == "no point to pick: early exit is opt-in, never picked"
    assert pick([zero, ee], paired=Noise(), unservable={}).excluded == {"m@ee.95": EE_OPT_IN}    # servable: opt-in
