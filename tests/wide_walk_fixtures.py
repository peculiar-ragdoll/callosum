"""The walk fixtures of tests/test_marginalize.py, as data, for the wide-choice byte-identity pin (with
`partial=()` the walk returns what it returned before the change). `tests/data/walk_identity.json` holds the
pre-change results of these exact calls, captured by `snapshot()` below on the commit before the wide-choice
change, in the development history."""
from __future__ import annotations

import math

WALK_SETS = {"C": frozenset({10, 11}), "D": frozenset({20})}
NEAR_SETS = {"A": frozenset(range(1, 41)), "B": frozenset(range(41, 81))}
STARS, THE, ANSWER, IS = 90, 91, 92, 93


def _markdown(prefix):
    if prefix == ():
        return {STARS: math.log(0.97), 10: math.log(0.01), 20: math.log(0.02)}
    return {10: math.log(0.7), 20: math.log(0.25), STARS: math.log(0.05)}


_CHAIN = {(): THE, (THE,): ANSWER, (THE, ANSWER): IS}


def _preamble(prefix):
    nxt = _CHAIN.get(prefix)
    if nxt is not None:
        return {nxt: math.log(0.99), 10: math.log(0.005), 20: math.log(0.005)}
    return {20: math.log(0.8), 10: math.log(0.15)}


def _near_certain(excess):
    small = [1e-5 * (1 + (i % 7)) for i in range(79)]
    probs = {1: 1.0 + excess - math.fsum(small)}
    probs.update(dict(zip(range(2, 81), small, strict=True)))
    return {tid: math.log(p) for tid, p in probs.items()}


_NEAR = _near_certain(5e-7)
_FLAT = {STARS: math.log(0.2), THE: math.log(0.2), ANSWER: math.log(0.2), IS: math.log(0.2),
         10: math.log(0.01), 20: math.log(0.01)}

# name -> (sets, dist_fn, walk kwargs)
CASES = {
    "decides": (WALK_SETS, lambda p: {10: math.log(0.6), 20: math.log(0.3), STARS: math.log(0.1)}, {}),
    "markdown": (WALK_SETS, _markdown, {}),
    "preamble": (WALK_SETS, _preamble, {}),
    "adversarial_5": (WALK_SETS, lambda p: {STARS: 0.0}, {"max_depth": 5}),
    "adversarial_default": (WALK_SETS, lambda p: {STARS: 0.0}, {}),
    "flat": (WALK_SETS, lambda p: _FLAT, {}),
    "never_swallows": (WALK_SETS, lambda p: {10: math.log(0.95), 20: math.log(0.05)}, {}),
    "threads_prefix": (WALK_SETS, lambda p: {STARS: math.log(0.99), 10: math.log(0.005), 20: math.log(0.005)}
                       if len(p) < 3 else {10: math.log(0.9), 20: math.log(0.05)}, {"prefix": (THE,)}),
    "zero_depth": (WALK_SETS, lambda p: {STARS: 0.0}, {"max_depth": 0}),
    "agrees": (WALK_SETS, lambda p: {10: math.log(0.6), 20: math.log(0.3)}, {}),
    "logprob_above_zero": ({"C": frozenset({10}), "D": frozenset({20})}, lambda p: {10: 5e-7, 20: -30.0}, {}),
    "near_certain": (NEAR_SETS, lambda p: _NEAR, {}),
    "thresholds": (WALK_SETS, _markdown, {"divergence_mass": 0.5, "dominance": 0.9}),
}

# check_option_sets messages, pinned on the pre-change commit (the fast path must not change them)
OVERLAPS = {
    "crafted": ({"C": [0, 7], "D": [7, 8]}, None),
    "named": ({"C": [4, 5], "D": [5]}, ["a", "b", "c", "d", "e", "ĠC"]),
    "three_way": ({"A": [1, 2, 3, 4, 5, 6, 7], "B": [1, 2, 3, 4, 5, 6, 7, 9], "C": [9, 10]}, None),
}


def snapshot(divergence_walk, check_option_sets) -> dict:
    """The fields a pre-change `WalkResult` had, floats as `float.hex` (bit-exact), and the overlap messages."""
    walks = {}
    for name, (sets, fn, kw) in CASES.items():
        r = divergence_walk(sets, fn, **kw)
        walks[name] = {"prefix": list(r.prefix), "option_logprobs": {k: v.hex() for k, v in r.option_logprobs.items()},
                       "option_mass": r.option_mass.hex(), "depth": r.depth, "stop": r.stop}
    msgs = {}
    for name, (sets, tokens) in OVERLAPS.items():
        try:
            check_option_sets({k: frozenset(v) for k, v in sets.items()}, tokens=tokens)
        except ValueError as e:
            msgs[name] = str(e)
    return {"walks": walks, "overlap_messages": msgs}
