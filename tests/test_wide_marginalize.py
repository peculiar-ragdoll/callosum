"""callo v0.3 wide-choice, the marginaliser:
partial-label tokens in the walk, partial and continuation sets, split attribution, and the `check_option_sets` fast
path. Every expectation is hand-computed; nothing reads a model."""
import json
import math
import platform
from pathlib import Path

import pytest
from wide_walk_fixtures import CASES, OVERLAPS

from callo import marginalize
from callo.errors import InvalidInput
from callo.marginalize import (
    WIDE_MAX_SPLIT_PASSES,
    WIDE_SPLIT_READ,
    SplitResult,
    WalkResult,
    _mass_tol,
    check_option_sets,
    continuation_sets,
    divergence_walk,
    option_surface_sets,
    partial_sets,
    split_attribution,
)
from callo.systemone import WIDE_LABELS

NEG_INF = float("-inf")
SNAP = json.loads((Path(__file__).parent / "data" / "walk_identity.json").read_text(encoding="utf-8"))
L = math.log
# walk_identity.json was captured on Darwin-arm64 (ce89e73). There floats must match bit for bit; elsewhere (CI's Linux
# libm rounds exp/log differently in the last place) within the same 1e-12 rel + 1e-15 abs as test_wide_identity_sweep.
EXACT_FLOATS = (platform.system(), platform.machine()) == ("Darwin", "arm64")


def _same_float(got: float, want_hex: str) -> bool:
    want = float.fromhex(want_hex)
    if EXACT_FLOATS or math.isinf(want) or math.isinf(got):
        return got.hex() == want_hex
    return math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-15)


def _counting(fn):
    calls = []

    def wrapped(prefix):
        calls.append(tuple(prefix))
        return fn(tuple(prefix))
    wrapped.calls = calls
    return wrapped


def test_constants():
    assert WIDE_SPLIT_READ == 0.01 and WIDE_MAX_SPLIT_PASSES == 2


# ---- partial and continuation sets -------------------------------------------------------------------------------

def test_partial_sets_gpt2_end_in_the_letter():
    vocab = ["A", "ĠA", "(A", "**A", "A.", "A)", "a", "Ġa.", "AĠ"]           # "AĠ" ends in a space: out
    assert partial_sets(vocab, "gpt2", ["A"]) == {"A": frozenset({0, 1, 2, 3, 6})}


def test_partial_sets_gemma_metaspace_in_trailing_wrapper_out():
    vocab = ["▁A", "A.", "A", "B"]
    assert partial_sets(vocab, "gemma4", ["A", "B"]) == {"A": frozenset({0, 2}), "B": frozenset({3})}


def test_continuation_sets_gpt2():
    vocab = ["B", "b", "ĠB", "B.", "(B", "Ｂ", "BC"]
    assert continuation_sets(vocab, "gpt2", ["B"]) == {"B": frozenset({0, 1, 3, 5})}


def test_continuation_sets_gemma_metaspace_is_a_gap():
    vocab = ["▁B", "B", "b)"]
    assert continuation_sets(vocab, "gemma4", ["B"]) == {"B": frozenset({1, 2})}


def test_partial_and_continuation_sets_are_frozen_and_keyed_in_order():
    p = partial_sets(["B", "A"], "gpt2", ["B", "A"])
    assert list(p) == ["B", "A"] and all(isinstance(v, frozenset) for v in p.values())
    c = continuation_sets(["B", "A"], "gpt2", ["B", "A"])
    assert list(c) == ["B", "A"] and all(isinstance(v, frozenset) for v in c.values())


# ---- the toy: vocab `AA ĠAA AB ĠAB BA ĠBA A ĠA B ĠB ** x`, labels AA AB BA ------------------------------------
VOCAB = ["AA", "ĠAA", "AB", "ĠAB", "BA", "ĠBA", "A", "ĠA", "B", "ĠB", "**", "x"]
AA_, GAA, AB_, GAB, BA_, GBA, A, GA, B, GB, STARS, X = range(12)
LABELS = ("AA", "AB", "BA")
SETS = option_surface_sets(VOCAB, "gpt2", LABELS)
PART = partial_sets(VOCAB, "gpt2", ["A", "B"])
CONT = continuation_sets(VOCAB, "gpt2", ["A", "B"])
PART_IDS = frozenset().union(*PART.values())
ROOT = {GAB: L(0.5), GA: L(0.3), A: L(0.05), X: L(0.15)}


def test_toy_sets_are_what_the_math_assumes():
    assert SETS == {"AA": frozenset({AA_, GAA}), "AB": frozenset({AB_, GAB}), "BA": frozenset({BA_, GBA})}
    assert PART == {"A": frozenset({A, GA}), "B": frozenset({B, GB})}
    assert CONT == {"A": frozenset({A}), "B": frozenset({B})}


def _toy_dist(after_ga=None, after_a=None, root=None):
    after_ga = after_ga if after_ga is not None else {B: L(0.8), A: L(0.1), X: L(0.1)}
    after_a = after_a if after_a is not None else {B: L(0.5), X: L(0.5)}
    table = {(): root if root is not None else ROOT, (GA,): after_ga, (A,): after_a}

    def fn(prefix):
        return table[prefix]
    return _counting(fn)


def _walk(fn, partial=PART_IDS):
    return divergence_walk(SETS, fn, partial=partial)


# ---- the walk with `partial` -------------------------------------------------------------------------------------

def test_walk_partial_mass_stops_the_walk():
    fn = _counting(lambda p: {GA: L(0.9), X: L(0.1)})
    res = _walk(fn)
    assert res.stop == "diverged" and res.depth == 0 and res.prefix == ()
    assert res.partial_mass == pytest.approx(0.9, abs=1e-15) and res.option_mass == 0.0
    assert fn.calls == [()]


def test_walk_without_partial_steps_into_the_answer():
    """The hazard partial sets close: a dominant `ĠA` is no option surface, so a narrow-style walk appends it."""
    def fn(p):
        if p == ():
            return {GA: L(0.9), X: L(0.1)}
        assert p == (GA,)
        return {X: L(0.3), STARS: L(0.3), B: L(0.4)}
    res = _walk(_counting(fn), partial=())
    assert res.prefix == (GA,) and res.depth == 1 and res.stop == "no_shared_prefix"
    assert res.partial_mass == 0.0


def test_walk_refuses_a_partial_id_inside_an_option_set():
    with pytest.raises(InvalidInput, match="not disjoint"):
        _walk(lambda p: ROOT, partial={GA, GAA})


def test_walk_refuses_option_plus_partial_mass_above_one():
    fn = lambda p: {AA_: L(0.6), GA: L(0.6)}             # noqa: E731 -- each half legal, together 1.2
    assert _walk(fn, partial=()).option_mass == pytest.approx(0.6)
    with pytest.raises(InvalidInput, match=r"at prefix \(\) \(dist_fn\)"):
        _walk(fn)


def test_walk_n_terms_counts_finite_option_terms_at_the_stop():
    res = _walk(lambda p: {AA_: L(0.3), GAA: L(0.2), AB_: L(0.1), GA: L(0.1), X: L(0.3)})
    assert res.n_terms == 3
    stepped = _walk(lambda p: {STARS: L(0.95), AA_: L(0.01)} if p == () else
                    {AA_: L(0.3), GAA: L(0.2), AB_: L(0.1), GAB: L(0.1), BA_: L(0.1), X: L(0.2)})
    assert stepped.depth == 1 and stepped.n_terms == 5


@pytest.mark.parametrize("name", sorted(CASES))
def test_walk_without_partial_is_bit_identical_to_before(name):
    """`partial=()` (and no `partial` at all) give the pre-change WalkResult, bit for bit on the capture
    platform (within 1e-12 rel elsewhere, see EXACT_FLOATS), plus the two new fields at their no-partial values."""
    sets, fn, kw = CASES[name]
    want = SNAP["walks"][name]
    for res in (divergence_walk(sets, fn, **kw), divergence_walk(sets, fn, partial=(), **kw)):
        assert list(res.prefix) == want["prefix"] and res.depth == want["depth"] and res.stop == want["stop"]
        assert set(res.option_logprobs) == set(want["option_logprobs"])
        for k, v in res.option_logprobs.items():
            assert _same_float(v, want["option_logprobs"][k]), k
        assert _same_float(res.option_mass, want["option_mass"])
        assert res.partial_mass == 0.0
        finite = sum(marginalize._lookup(res.token_logprobs, t) != NEG_INF
                     for ids in res.sets.values() for t in ids)
        assert res.n_terms == finite
    assert divergence_walk(sets, fn, **kw) == divergence_walk(sets, fn, partial=(), **kw)


# ---- split attribution -------------------------------------------------------------------------------------------

def _split(fn=None, *, walk_fn=None, **kw):
    fn = fn if fn is not None else _toy_dist()
    walk = _walk(walk_fn if walk_fn is not None else fn)
    return split_attribution(walk, LABELS, PART, CONT, fn, **kw), fn


def _p(res: SplitResult) -> dict[str, float]:
    return {lab: math.exp(v) for lab, v in res.option_logprobs.items()}


def test_split_attribution_toy():
    res, fn = _split()
    p = _p(res)
    assert p["AB"] == pytest.approx(0.765, abs=1e-12)
    assert p["AA"] == pytest.approx(0.03, abs=1e-12)
    assert p["BA"] == 0.0 and res.option_logprobs["BA"] == NEG_INF
    assert res.option_mass == pytest.approx(0.795, abs=1e-12)
    assert res.unread_mass == pytest.approx(0.0, abs=1e-15)
    assert res.split_passes == 2 and res.expanded == (GA, A)
    assert [f for f, _ in res.partial_by_letter] == ["A", "B"]
    assert res.partial_by_letter[0][1] == pytest.approx(0.35, abs=1e-15) and res.partial_by_letter[1][1] == 0.0
    assert math.exp(res.credit["AB"]) == pytest.approx(0.3 * 0.8 + 0.05 * 0.5, abs=1e-12)
    assert math.exp(res.credit["AA"]) == pytest.approx(0.3 * 0.1, abs=1e-12) and res.credit["BA"] == NEG_INF
    assert list(res.option_logprobs) == list(LABELS) == list(res.credit)
    assert fn.calls == [(), (GA,), (A,)]


def test_split_max_passes_one_leaves_the_rest_unread():
    res, fn = _split(max_passes=1)
    assert res.split_passes == 1 and res.expanded == (GA,)
    assert res.unread_mass == pytest.approx(0.05, abs=1e-12)
    assert _p(res)["AB"] == pytest.approx(0.74, abs=1e-12)
    assert fn.calls == [(), (GA,)]


def test_split_read_above_stops_once_the_unread_is_small():
    res, _ = _split(read_above=0.06)
    assert res.split_passes == 1 and res.expanded == (GA,) and res.unread_mass == pytest.approx(0.05, abs=1e-12)


def test_split_read_above_zero_reads_everything_it_can():
    res, _ = _split(read_above=0.0, max_passes=None)
    assert res.split_passes == 2 and res.unread_mass == pytest.approx(0.0, abs=1e-15)


def test_split_min_partial_skips_small_candidates():
    res, _ = _split(min_partial=0.1)
    assert res.expanded == (GA,) and res.split_passes == 1 and res.unread_mass == pytest.approx(0.05, abs=1e-12)


def test_split_ties_on_p_expand_in_id_order():
    root = {GAB: L(0.4), GA: L(0.2), A: L(0.2), X: L(0.2)}
    res, fn = _split(_toy_dist(root=root))
    assert res.expanded == (A, GA)
    one, _ = _split(_toy_dist(root=root), max_passes=1)
    assert one.expanded == (A,)


class _Boom(Exception):
    pass


def test_split_skip_costs_an_attempt_not_a_pass():
    table = {(): ROOT, (A,): {B: L(0.5), X: L(0.5)}}

    def fn(prefix):
        if prefix == (GA,):
            raise _Boom
        return table[prefix]
    res = split_attribution(_walk(fn), LABELS, PART, CONT, fn, skip=_Boom)
    assert res.split_passes == 1 and res.expanded == (A,)
    assert res.unread_mass == pytest.approx(0.3, abs=1e-12)
    assert _p(res)["AB"] == pytest.approx(0.525, abs=1e-12)
    with pytest.raises(_Boom):                                       # not in `skip`: it propagates
        split_attribution(_walk(fn), LABELS, PART, CONT, fn)


def test_split_attempts_are_capped_when_every_candidate_raises():
    ids = list(range(100, 108))
    part = {"A": frozenset(ids), "B": frozenset()}
    root = {GAB: L(0.2), **{t: L(0.1) for t in ids}}
    calls = []

    def fn(prefix):
        calls.append(prefix)
        if prefix == ():
            return root
        raise _Boom
    walk = divergence_walk(SETS, fn, partial=frozenset(ids))
    res = split_attribution(walk, LABELS, part, CONT, fn, skip=_Boom)
    assert len(calls) == 1 + WIDE_MAX_SPLIT_PASSES + 2                # the root, then 4 attempts
    assert res.split_passes == 0 and res.expanded == ()
    assert res.unread_mass == pytest.approx(0.8, abs=1e-12)
    calls.clear()
    uncapped = split_attribution(walk, LABELS, part, CONT, fn, skip=_Boom, max_passes=None)
    assert len(calls) == 8 and uncapped.split_passes == 0             # no cap: every candidate is attempted once


def test_split_without_dist_fn_reads_nothing():
    walk = _walk(_toy_dist())
    res = split_attribution(walk, LABELS, PART, CONT, None)
    assert res.split_passes == 0 and res.expanded == ()
    assert res.unread_mass == pytest.approx(0.35, abs=1e-15)
    assert res.option_logprobs == walk.option_logprobs and res.option_mass == pytest.approx(0.5)
    assert all(v == NEG_INF for v in res.credit.values())


def test_split_refuses_a_continuation_dump_above_one():
    fn = _toy_dist(after_ga={B: L(0.8), A: L(0.8)})
    with pytest.raises(InvalidInput, match=r"after split token 7 at prefix \(\)"):
        _split(fn)


def test_split_single_label_letter():
    """`B` starts only `BA` in this question: its split pass reads continuation A alone."""
    root = {GAB: L(0.3), GB: L(0.4), X: L(0.3)}
    table = {(): root, (GB,): {A: L(0.5), X: L(0.5)}}
    fn = _counting(lambda p: table[p])
    res = split_attribution(_walk(fn), LABELS, PART, CONT, fn)
    p = _p(res)
    assert p["BA"] == pytest.approx(0.2, abs=1e-12) and p["AB"] == pytest.approx(0.3, abs=1e-12) and p["AA"] == 0.0
    assert res.split_passes == 1 and res.expanded == (GB,) and res.unread_mass == pytest.approx(0.0, abs=1e-15)
    assert res.partial_by_letter == (("A", 0.0), ("B", pytest.approx(0.4, abs=1e-15)))


def test_split_partial_by_letter_lists_only_the_question_letters():
    labels = ("AA", "AB")
    walk = divergence_walk({k: SETS[k] for k in labels}, _toy_dist(), partial=PART_IDS)
    res = split_attribution(walk, labels, PART, CONT, _toy_dist())
    assert [f for f, _ in res.partial_by_letter] == ["A"]


def _tol_case(n_terms_total: int, delta: float) -> SplitResult:
    """direct AA = AB = 0.5, the partial `ĠA` carries `delta`, and its continuation is B with p = 1: every per-pass
    check passes, and the readout's direct + credit + unread is exactly 1 + delta. The final check's slack counts
    walk.n_terms (2) + len(cands) (1) + the split pass's finite continuation terms (1) = 4 terms."""
    assert n_terms_total == 4
    d = {AA_: L(0.5), AB_: L(0.5), GA: math.log(delta)}
    walk = WalkResult(prefix=(), token_logprobs=d, option_logprobs={"AA": L(0.5), "AB": L(0.5)}, option_mass=1.0,
                      depth=0, stop="diverged", sets={"AA": SETS["AA"], "AB": SETS["AB"]}, partial_mass=delta,
                      n_terms=2)
    fn = lambda p: {B: 0.0}                                  # noqa: E731
    return split_attribution(walk, ("AA", "AB"), PART, CONT, fn, read_above=0.0)


def test_split_final_tolerance_counts_every_summed_term():
    inside = _mass_tol(4) - 5e-8                             # above _mass_tol(3): a smaller count would refuse it
    assert inside > _mass_tol(3)
    res = _tol_case(4, inside)
    assert res.option_mass == 1.0 and res.split_passes == 1
    with pytest.raises(InvalidInput, match=r"\(split readout\)"):
        _tol_case(4, _mass_tol(4) + 5e-8)


def test_split_refuses_a_walk_that_was_not_given_the_partial_set():
    """A walk run WITHOUT `partial` would have stepped over a dominant first letter (the hazard partial
    sets close), and its `partial_mass` is 0.0 although the slot dump holds partial tokens: split attribution on such
    a walk is a caller bug, refused rather than read."""
    walk = _walk(_toy_dist(), partial=())
    assert walk.prefix == () and walk.partial_mass == 0.0                     # ĠAB 0.5 stops it at the root anyway
    with pytest.raises(InvalidInput, match="split_attribution: the walk was not given the partial set"):
        split_attribution(walk, LABELS, PART, CONT, _toy_dist())
    res = split_attribution(_walk(_toy_dist()), LABELS, PART, CONT, _toy_dist())
    assert res.split_passes == 2                                              # with the partial set: read as before
    nothing = WalkResult(prefix=(), token_logprobs={AB_: L(0.9), X: L(0.1)}, option_logprobs={
        "AA": NEG_INF, "AB": L(0.9), "BA": NEG_INF}, option_mass=0.9, depth=0, stop="diverged", sets=SETS, n_terms=1)
    assert split_attribution(nothing, LABELS, PART, CONT, _toy_dist()).split_passes == 0   # no candidate: no guard


def test_split_result_is_frozen():
    res, _ = _split()
    with pytest.raises(AttributeError):
        res.unread_mass = 1.0                                # type: ignore[misc]


# ---- check_option_sets: the disjoint fast path --------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(OVERLAPS))
def test_overlap_messages_are_unchanged(name):
    sets, tokens = OVERLAPS[name]
    with pytest.raises(InvalidInput) as e:
        check_option_sets({k: frozenset(v) for k, v in sets.items()}, tokens=tokens)
    assert str(e.value) == SNAP["overlap_messages"][name]


def test_disjoint_255_labels_take_the_fast_path(monkeypatch):
    calls = []
    real = marginalize._overlaps

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(marginalize, "_overlaps", spy)
    sets = {lab: frozenset({2 * i, 2 * i + 1}) for i, lab in enumerate(WIDE_LABELS)}
    check_option_sets(sets)
    assert calls == []
    with pytest.raises(InvalidInput, match="not disjoint"):
        check_option_sets({**sets, "AA": frozenset({0, 3})})          # AA now shares 3 with AB
    assert calls == [1]
