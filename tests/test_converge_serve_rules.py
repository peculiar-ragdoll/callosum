"""The serve side reads a think profile's `converge` block: K as is, one rule per target z (.9, .95, .99), a z between
targets taking the next higher target's rule (a z at or below .9 the .9 rule: conservative), T from the full-data
cell, an empty fit as no rule, and the stop test after each probe agreeing with calibrate's stop_index."""
import random

import pytest

from callo.converge import Probe, ServeRule, converged, serve_rules, stop_index

BLOCK = {"K": 32, "cost": {"c_ms": 29.98},
         "rules": {"0.9": {"cell": [1, 0.95], "T": 2.24, "acc": 0.9}, "0.95": {"cell": [1, 0.99], "T": 2.30},
                   "0.99": None}}


def test_a_profile_block_gives_k_the_rules_and_the_probe_cost():
    r, why = serve_rules(BLOCK)
    assert why is None and (r.K, r.c_ms) == (32, 29.98)
    assert r.rules[0.9] == ServeRule(0.9, 1, 0.95, 2.24) and r.rules[0.95] == ServeRule(0.95, 1, 0.99, 2.3)
    assert r.rules[0.99] is None
    assert r.as_dict() == {"K": 32, "rules": {"0.9": {"n": 1, "y": 0.95, "T": 2.24},
                                              "0.95": {"n": 1, "y": 0.99, "T": 2.3}, "0.99": None}}


@pytest.mark.parametrize("z, target", [(0.5, 0.9), (0.9, 0.9), (0.9000000000000001, 0.9), (0.92, 0.95),
                                       (0.95, 0.95), (0.96, None), (0.995, None)])
def test_a_z_takes_the_smallest_target_at_or_above_it(z, target):
    rule = serve_rules(BLOCK)[0].for_z(z)
    assert (rule.z if rule is not None else None) == target       # .96 -> .99, which has no rule


@pytest.mark.parametrize("block, why", [
    (None, "has no convergence block"),
    ([], "not an object"),
    ({**BLOCK, "K": 48}, "K must be one of 32, 64, 128"),
    ({**BLOCK, "K": True}, "K must be one of"),
    ({**BLOCK, "rules": {"0.9": None}}, "keyed by the targets"),
    ({**BLOCK, "rules": {**BLOCK["rules"], "0.9": {"cell": [5, 0.95], "T": 1.0}}}, "cell must be [n, y] on the grid"),
    ({**BLOCK, "rules": {**BLOCK["rules"], "0.9": {"cell": [1, 0.97], "T": 1.0}}}, "cell must be [n, y] on the grid"),
    ({**BLOCK, "rules": {**BLOCK["rules"], "0.9": {"cell": [1, 0.95], "T": 0}}}, "T must be null or finite"),
    ({**BLOCK, "cost": {"c_ms": -1}}, "cost.c_ms"),
    ({**BLOCK, "cost": [29.98]}, "cost must be an object"),
    ({**BLOCK, "cells": {"n": 1, "y": 0.95, "T": 2.5}}, "cells must be a list"),
    ({**BLOCK, "cells": 7}, "cells must be a list"),
    ({**BLOCK, "cells": ["1x0.95"]}, "cells must be objects with an integer n and a finite y"),
    ({**BLOCK, "cells": [{"n": True, "y": 0.95, "T": 2.5}]}, "integer n"),
    ({**BLOCK, "cells": [{"n": 1.0, "y": 0.95, "T": 2.5}]}, "integer n"),
    ({**BLOCK, "cells": [{"n": 1, "y": "0.95", "T": 2.5}]}, "finite y"),
    ({**BLOCK, "cells": [{"n": 1, "T": 2.5}]}, "finite y"),
])
def test_a_malformed_block_says_why(block, why):
    r, reason = serve_rules(block)
    assert r is None and why in reason


def test_a_rule_without_t_is_kept_with_t_none():
    r, _ = serve_rules({**BLOCK, "rules": {**BLOCK["rules"], "0.9": {"cell": [2, 0.9], "T": None}}})
    assert r.rules[0.9] == ServeRule(0.9, 2, 0.9, None)


def test_the_served_t_is_the_full_data_cell_s_and_the_cross_fitted_one_only_without_it():
    cells = [{"n": 1, "y": 0.95, "T": 2.5, "stops": 300}, {"n": 1, "y": 0.99, "T": None, "stops": 2}]
    r, _ = serve_rules({**BLOCK, "cells": cells})
    assert r.rules[0.9].T == 2.5                       # the full-data cell's stop readouts
    assert r.rules[0.95].T is None                     # that cell's fit had < 2 stops: no T, never the summary's
    assert serve_rules(BLOCK)[0].rules[0.9].T == 2.24  # a block without cells: the rule summary's T
    bad, why = serve_rules({**BLOCK, "cells": [{"n": 1, "y": 0.95, "T": -1}]})
    assert bad is None and "cells' T" in why


def test_an_empty_rules_object_is_no_rule_at_any_target():
    r, why = serve_rules({**BLOCK, "rules": {}})       # calibrate writes {} when no question had a trace
    assert why is None and r.rules == {0.9: None, 0.95: None, 0.99: None} and r.for_z(0.9) is None


def test_checked_after_every_probe_converged_closes_where_stop_index_stops():
    rng = random.Random(7)
    for _ in range(500):
        probes = []
        for j in range(rng.randint(1, 8)):
            p = [rng.random() for _ in range(3)]
            probes.append(Probe(32 * (j + 1), tuple(x / sum(p) for x in p)))
        n, y = rng.choice((1, 2, 3, 4)), rng.choice((0.4, 0.5, 0.6))
        first = next((i for i in range(len(probes)) if converged(probes[: i + 1], n, y)), None)
        assert first == stop_index(probes, n, y)


def test_a_null_cells_or_cost_is_absent_not_malformed():
    r, why = serve_rules({**BLOCK, "cells": None, "cost": None})
    assert why is None and r.c_ms is None and r.rules[0.9].T == 2.24
    assert serve_rules({**BLOCK, "cells": []})[0].rules[0.9].T == 2.24


def test_unservable_names_each_early_exit_point_serve_cannot_run_as_measured():
    """An early-exit point whose z has no rule (serve would run without early exit), or any early-exit
    point of a profile whose block serve refuses, with why; budget points are never named."""
    from callo.converge import UNSERVED, unservable
    from callo.think import Point

    def pt(kind, value):
        return Point(name="m", kind=kind, value=value, budget=value if kind == "budget" else None, acc=0.9, n=10,
                     lat_mean=(1.0,) * 3, lat_p50=(1.0,) * 3, lat_p95=(1.0,) * 3, lat_p99=(1.0,) * 3, lat_max=1.0)
    pts = [pt("budget", 256), pt("ee", 0.9), pt("ee", 0.95), pt("ee", 0.99)]
    assert unservable(pts, BLOCK) == {
        "m@ee.99": f"{UNSERVED}: no early-exit rule reaches z .99 (serve would run without early exit)"}
    gone = unservable(pts, None)
    assert set(gone) == {"m@ee.9", "m@ee.95", "m@ee.99"}
    assert gone["m@ee.9"] == f"{UNSERVED}: --think-converge refuses this profile (it has no convergence block)"
    assert unservable(pts[:1], None) == {}
