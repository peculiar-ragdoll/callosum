# Adapted from the author's benchmark harness (tests/test_sharpquant_marginalize.py @ f886c50f).
# Edits: the two MiniCPM fixtures and the 15 tests that read the MiniCPM5-2B GGUF from disk are dropped;
# imports retargeted to callo; the harness's error class -> CalloError; v0.3 adds the gemma4 decode and option-set
# tests.
"""The answer-surface marginaliser: `P(answer = C)`, not `P(token "C")`.

Three invariants carry this component: every option's equivalence set is non-empty, the sets are
disjoint, the divergence walk terminates. None of them needs a model, which is the whole reason this
file exists before a GPU-minute is spent.

Two rules this file holds itself to, both learned the hard way on this branch:

1. **No guard that cannot fail.** Every `raises` test below is paired with a demonstration that the
   rejected input is reachable -- from the real builder on the real vocabulary where possible, from
   a crafted one only where the real vocabulary happens not to contain a violator. A guard nobody
   can trip is a comment with a stack trace attached.
2. **The real tokenizer, not a mock of one.** MiniCPM5-2B's vocabulary is gpt2 byte-level BPE and it
   does NOT carry `"\\nC"` as a single token -- a fact no hand-written fixture would have told us,
   and the entire justification for the trie walk. Fixtures test the algebra; the GGUF tests the
   claim about tokenizers.
"""
import math
import struct

import pytest

from callo.errors import CalloError, InvalidInput
from callo.marginalize import (
    DEFAULT_DIVERGENCE_MASS,
    DEFAULT_DOMINANCE,
    DEFAULT_MAX_DEPTH,
    WRAPPER_CHARS,
    TopLogprobs,
    WalkResult,
    answer_agreement,
    answer_choice,
    answer_margin,
    bytes_to_unicode,
    check_option_sets,
    decode_token,
    divergence_walk,
    eog_token,
    logsumexp,
    marginal_option_logprobs,
    normalize_surface,
    option_mass,
    option_surface_sets,
)

NEG_INF = float("-inf")

# --------------------------------------------------------------------------------------------
# The real tokenizer.
#
# Header only: `GGUFReader` memory-maps the file and we touch one metadata field, so this reads a
# vocabulary out of a 5 GB BF16 model without loading a single weight. Mirrors the gating idiom in
# tests/test_sharpquant_groups.py -- an unavailable ENVIRONMENT skips, a stale REFERENCE fails,
# because those are different facts and collapsing them silently deletes coverage.
# --------------------------------------------------------------------------------------------

# Token ids verified by hand against the file (see the report): the vocabulary is 130560 tokens.

LETTERS = list("ABCDEFGHIJ")  # MMLU-Pro's ten options


# --------------------------------------------------------------------------------------------
# 1. Byte-level decode
# --------------------------------------------------------------------------------------------

def test_bytes_to_unicode_is_a_bijection_over_all_256_bytes():
    """The gpt2 map must cover every byte and collide on none, or decode is lossy for some token
    and we would find out as a mysteriously empty equivalence set rather than as an error."""
    m = bytes_to_unicode()
    assert sorted(m) == list(range(256))
    assert len(set(m.values())) == 256
    assert all(len(v) == 1 for v in m.values())


def test_bytes_to_unicode_matches_the_published_anchors():
    m = bytes_to_unicode()
    assert m[ord(" ")] == "Ġ"   # Ġ
    assert m[ord("\n")] == "Ċ"  # Ċ
    assert m[ord("A")] == "A"        # printable ASCII is its own image
    assert m[0] == "Ā"


@pytest.mark.parametrize("raw,expect", [
    ("C", "C"),
    ("ĠC", " C"),       # ĠC
    ("ĊC", "\nC"),      # ĊC
    ("ĊĊ", "\n\n"),
    ("(C", "(C"),
    ("**C**", "**C**"),
    ("ĠĠ", "  "),
])
def test_decode_undoes_the_byte_level_mapping(raw, expect):
    assert decode_token(raw, "gpt2") == expect


def test_decode_sentencepiece_metaspace_and_byte_fallback():
    """A second tokenizer family, so the byte-level branch is a BRANCH and not the only path."""
    assert decode_token("▁C", "llama") == " C"
    assert decode_token("<0x0A>", "llama") == "\n"
    assert decode_token("C", "spm") == "C"


def test_decode_unknown_tokenizer_is_identity():
    """Documented fallback: a tokenizer family we do not model is passed through verbatim. It is
    the only honest option -- inventing a decoding would silently mis-build equivalence sets -- and
    it degrades to exactly the naive behaviour rather than to an exception."""
    assert decode_token("C", "bert") == "C"
    assert decode_token("ĠC", "rwkv") == "ĠC"


def test_decode_passes_through_codepoints_outside_the_byte_map():
    """A gpt2 vocabulary should not contain these; a mislabelled one might. Passing them through
    keeps one odd token from poisoning the whole vocabulary decode with an exception."""
    assert decode_token("中", "gpt2") == "中"
    assert decode_token("Ġ中", "gpt2") == " 中"


# --------------------------------------------------------------------------------------------
# 2. Surface normalisation
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("surface", [
    "C", " C", "  C  ", "\nC", "\n\nC", "\tC",
    "(C", "C)", "(C)", "[C]", "{C}", "<C>",
    "*C*", "**C**", "__C__", "`C`", "~~C~~",
    "C.", "C:", "C,", "C;", "C!", "C?", "-C", "C-", "/C", "#C",
    "**C**.", " **(C)** ", ".c", "c", " c ", "'C'", '"C"', "“C”",
])
def test_every_surface_form_reduces_to_the_letter(surface):
    assert normalize_surface(surface) == "C"


def test_normalisation_uppercases():
    assert normalize_surface("c") == "C"
    assert normalize_surface(" (c) ") == "C"


def test_normalisation_leaves_the_interior_alone():
    """Stripping is from the ENDS only. An interior strip would make `A-B` a surface form of `AB`
    and quietly merge distinct answers."""
    assert normalize_surface("A-B") == "A-B"
    assert normalize_surface("(A B)") == "A B"


def test_normalisation_of_pure_punctuation_is_empty():
    for s in ("", "   ", "**", "()", "...", "-", "\n"):
        assert normalize_surface(s) == ""


def test_normalisation_is_idempotent():
    """A fixed-point loop that is not idempotent is a loop that did not reach its fixed point."""
    for s in ("**(C)**", " \n[c]. ", "C", "", "A B"):
        once = normalize_surface(s)
        assert normalize_surface(once) == once


def test_wrapper_chars_contains_no_alphanumerics():
    """The strip set is the one place a typo silently eats answers: a digit in it would reduce the
    option `1` to nothing, and a letter would reduce `C` to nothing."""
    assert not any(ch.isalnum() for ch in WRAPPER_CHARS)
    assert len(set(WRAPPER_CHARS)) == len(WRAPPER_CHARS)


# --------------------------------------------------------------------------------------------
# 3. Equivalence sets
# --------------------------------------------------------------------------------------------

# A crafted byte-level vocabulary. Ids are positions in this list.
CRAFT = ["C", "ĠC", "ĊC", "(C", "c", "D", "ĠD", "**", "Ġthe", "hello", "!"]
CRAFT_MODEL = "gpt2"


def test_option_sets_on_a_crafted_vocabulary():
    sets = option_surface_sets(CRAFT, CRAFT_MODEL, ["C", "D"])
    assert sets["C"] == frozenset({0, 1, 2, 3, 4})
    assert sets["D"] == frozenset({5, 6})


def test_option_sets_are_frozen():
    """The sets are built once per (model, tokenizer) and reused by every probe forever. A mutable
    set handed out by reference is a cache one caller can edit for everybody."""
    sets = option_surface_sets(CRAFT, CRAFT_MODEL, ["C", "D"])
    assert all(isinstance(v, frozenset) for v in sets.values())


# --------------------------------------------------------------------------------------------
# 4. The guards. Each one is shown rejecting a REACHABLE input.
# --------------------------------------------------------------------------------------------

def test_check_accepts_a_good_set():
    """Positive control: the guard is not simply raising on everything."""
    check_option_sets({"C": frozenset({0, 1}), "D": frozenset({2})})


def test_check_rejects_an_empty_set_crafted():
    with pytest.raises(InvalidInput) as e:
        check_option_sets({"C": frozenset({0}), "D": frozenset()})
    assert "D" in str(e.value)


def test_check_rejects_overlap_crafted():
    with pytest.raises(InvalidInput) as e:
        check_option_sets({"C": frozenset({0, 7}), "D": frozenset({7, 8})})
    msg = str(e.value)
    assert "C" in msg and "D" in msg and "7" in msg


def test_check_names_the_offending_tokens_not_just_the_options():
    with pytest.raises(InvalidInput) as e:
        check_option_sets({"C": frozenset({4, 5}), "D": frozenset({5})}, tokens=["a", "b", "c", "d",
                                                                                "e", "ĠC"])
    assert "ĠC" in str(e.value)


def test_builder_rejects_fewer_than_two_options():
    """A one-option margin is `logP(correct) - max(<empty>)`, which is the exact
    `max() iterable argument is empty` failure objective.py's module docstring is about."""
    with pytest.raises(InvalidInput):
        option_surface_sets(CRAFT, CRAFT_MODEL, ["C"])
    with pytest.raises(InvalidInput):
        option_surface_sets(CRAFT, CRAFT_MODEL, [])


def test_builder_rejects_duplicate_labels():
    with pytest.raises(InvalidInput) as e:
        option_surface_sets(CRAFT, CRAFT_MODEL, ["C", "D", "C"])
    assert "C" in str(e.value)


def test_builder_rejects_an_option_that_normalises_to_nothing():
    """`normalize_surface("**") == ""`, and 2659 real MiniCPM tokens also normalise to empty. An
    option label that vanishes would therefore collect thousands of unrelated tokens."""
    with pytest.raises(InvalidInput) as e:
        option_surface_sets(CRAFT, CRAFT_MODEL, ["C", "**"])
    assert "**" in str(e.value)


def test_errors_are_callo_errors():
    """So the CLI's one-line-on-stderr path catches them, per errors.py."""
    with pytest.raises(CalloError):
        check_option_sets({"C": frozenset(), "D": frozenset({1})})


# --------------------------------------------------------------------------------------------
# 5. Marginalisation. All of it model-free: the distribution is an argument.
#
# One fixed toy geometry, used by most of what follows. Option C is spelled three ways, none of
# which is individually the most likely token; option D is spelled once and that single token IS
# the argmax of the raw distribution. So SUM and MAX disagree about the answer, which is the
# entire point of the component.
# --------------------------------------------------------------------------------------------
TOY_SETS = {"C": frozenset({10, 11, 12}), "D": frozenset({20}), "E": frozenset({30})}
TOY_PROBS = {10: 0.15, 11: 0.15, 12: 0.15, 20: 0.30, 30: 0.05, 99: 0.20}
TOY_LP = {k: math.log(v) for k, v in TOY_PROBS.items()}


def _close(a, b, tol=1e-12):
    return abs(a - b) <= tol


# ---- log-sum-exp ---------------------------------------------------------------------------

def test_logsumexp_matches_the_definition():
    vals = [math.log(0.1), math.log(0.2), math.log(0.3)]
    assert _close(logsumexp(vals), math.log(0.6), 1e-12)


def test_logsumexp_of_n_equal_values_is_x_plus_log_n():
    assert _close(logsumexp([-3.0] * 4), -3.0 + math.log(4), 1e-12)


def test_logsumexp_is_stable_where_the_naive_form_underflows():
    """The max-shift is load-bearing, not decoration: `exp(-1000)` is 0.0 in float64, so a naive
    sum returns `-inf` -- a confident WRONG answer that says the option carries no mass at all.
    Deep in a 130k vocabulary the tail options really do sit down here."""
    assert _close(logsumexp([-1000.0, -1000.0]), -1000.0 + math.log(2), 1e-9)
    assert sum(math.exp(v) for v in [-1000.0, -1000.0]) == 0.0  # what we are avoiding


def test_logsumexp_ignores_minus_inf_terms():
    assert _close(logsumexp([math.log(0.5), NEG_INF]), math.log(0.5), 1e-12)


def test_logsumexp_of_all_minus_inf_is_minus_inf():
    assert logsumexp([NEG_INF, NEG_INF]) == NEG_INF


def test_logsumexp_of_nothing_is_minus_inf():
    """An empty sum of probabilities is zero probability; `log 0` is `-inf`, not an exception."""
    assert logsumexp([]) == NEG_INF


# ---- the marginal --------------------------------------------------------------------------

def test_marginal_sums_the_surface_forms():
    out = marginal_option_logprobs(TOY_SETS, TOY_LP)
    assert _close(math.exp(out["C"]), 0.45, 1e-12)
    assert _close(math.exp(out["D"]), 0.30, 1e-12)
    assert _close(math.exp(out["E"]), 0.05, 1e-12)


def test_sum_not_max_changes_the_answer():
    """The load-bearing test of the whole module.

    Under SUM, option C wins (0.45 > 0.30). Under MAX over surface forms -- which is what scoring
    the single most likely spelling amounts to -- option D wins (0.30 > 0.15). Both the argmax and
    the SIGN of the margin flip. A marginaliser that took the max would be a different metric
    wearing this one's name."""
    out = marginal_option_logprobs(TOY_SETS, TOY_LP)
    assert answer_choice(TOY_SETS, TOY_LP) == "C"
    as_max = {o: max(TOY_LP[t] for t in s) for o, s in TOY_SETS.items()}
    assert max(as_max, key=as_max.get) == "D"
    assert answer_margin(TOY_SETS, TOY_LP, "C") > 0
    assert as_max["C"] - as_max["D"] < 0
    assert out["C"] > out["D"]


def test_marginal_accepts_a_dense_sequence():
    """llama.cpp hands back a dense per-token vector; a top-k dump hands back a mapping. Both."""
    dense = [NEG_INF] * 40
    for tid, p in TOY_PROBS.items():
        if tid < 40:
            dense[tid] = math.log(p)
    out = marginal_option_logprobs(TOY_SETS, dense)
    assert _close(math.exp(out["C"]), 0.45, 1e-12)


def test_marginal_treats_a_token_absent_from_a_sparse_mapping_as_zero_mass():
    """A top-k dump omits everything below the cut, and `absent` there means `below the cut`, so
    `-inf` is the right reading -- but it is a LOSSY one, and the reason `option_mass` exists as a
    separate validity diagnostic rather than being folded in silently."""
    sparse = {10: math.log(0.5)}
    out = marginal_option_logprobs(TOY_SETS, sparse)
    assert _close(math.exp(out["C"]), 0.5, 1e-12)
    assert out["D"] == NEG_INF


def test_marginal_rejects_nan():
    """`nan` defeats every ordering guard downstream -- `nan < 0` and `nan > 0` are both False --
    exactly as objective.py's weight validation documents. Caught where it enters."""
    with pytest.raises(InvalidInput) as e:
        marginal_option_logprobs(TOY_SETS, {10: float("nan"), 20: -1.0, 30: -1.0})
    assert "10" in str(e.value)


def test_marginal_rejects_a_positive_logprob():
    """A log-probability above zero is a probability above one. Refusing it is what keeps
    `option_mass` honest: silently accepting unnormalised log-SCORES would make the section-6
    validity diagnostic report a number that means nothing."""
    with pytest.raises(InvalidInput):
        marginal_option_logprobs(TOY_SETS, {10: 0.5, 20: -1.0, 30: -1.0})


def test_marginal_accepts_exactly_zero():
    """`logP = 0` is `P = 1`, which is legal and happens on a fully confident position."""
    out = marginal_option_logprobs({"C": frozenset({10}), "D": frozenset({20})}, {10: 0.0})
    assert out["C"] == 0.0


def test_marginal_rejects_an_id_past_the_end_of_a_dense_vector():
    """A dense vector shorter than the vocabulary is a real mistake -- a truncated dump, or sets
    built from a different tokenizer -- and reading it as zero mass would silently delete whichever
    options live in the tail."""
    with pytest.raises(InvalidInput) as e:
        marginal_option_logprobs(TOY_SETS, [math.log(0.5)] * 15)
    assert "20" in str(e.value)


def test_marginal_rejects_sets_that_fail_the_invariants():
    """The invariants are not advisory: a token in two sets is counted in both marginals, so the
    margin between those options compares mass against itself."""
    with pytest.raises(InvalidInput):
        marginal_option_logprobs({"C": frozenset({10}), "D": frozenset({10})}, TOY_LP)
    with pytest.raises(InvalidInput):
        marginal_option_logprobs({"C": frozenset(), "D": frozenset({10})}, TOY_LP)


# ---- the validity diagnostic --------------------------------------------------------------

def test_option_mass_is_the_share_on_the_option_set():
    """The mass is NOT a correction (the margin is invariant to it) -- it tells you
    whether the scoring position is genuinely an answer slot at all."""
    assert _close(option_mass(TOY_SETS, TOY_LP), 0.80, 1e-12)


def test_option_mass_rejects_mass_above_one():
    """Reachable, and only here: each of these logprobs is individually legal (`P = 1` twice), so
    no per-value check can catch it. Two options that are each certain is not a distribution."""
    with pytest.raises(InvalidInput) as e:
        option_mass({"C": frozenset({10}), "D": frozenset({20})}, {10: 0.0, 20: 0.0})
    assert "2.0" in str(e.value)


# ---- margin ---------------------------------------------------------------------------------

def test_answer_margin_value_and_sign():
    m = answer_margin(TOY_SETS, TOY_LP, "C")
    assert _close(m, math.log(0.45) - math.log(0.30), 1e-12)
    assert answer_margin(TOY_SETS, TOY_LP, "E") < 0


def test_answer_margin_is_against_the_best_OTHER_option():
    """Not against the runner-up overall and not against the mean: the margin is a decision
    boundary, and the boundary is whichever wrong answer is closest to winning."""
    m = answer_margin(TOY_SETS, TOY_LP, "E")
    assert _close(m, math.log(0.05) - math.log(0.45), 1e-12)


def test_answer_margin_is_invariant_to_the_total_option_mass():
    """Stated as arithmetic: shifting every logprob by a constant (an unnormalised
    distribution, or one whose mass leaked to non-option tokens) leaves the margin untouched,
    because a difference of logprobs cancels the normaliser. This is why the mass is reported as a
    diagnostic instead of being divided out."""
    shifted = {k: v - 2.5 for k, v in TOY_LP.items()}
    assert _close(answer_margin(TOY_SETS, shifted, "C"),
                  answer_margin(TOY_SETS, TOY_LP, "C"), 1e-12)
    assert not _close(option_mass(TOY_SETS, shifted), option_mass(TOY_SETS, TOY_LP), 1e-3)


def test_answer_margin_rejects_an_unknown_correct_option():
    with pytest.raises(InvalidInput) as e:
        answer_margin(TOY_SETS, TOY_LP, "Z")
    assert "Z" in str(e.value)


def test_answer_margin_rejects_a_single_option():
    """`max` over the other options, with no other options. The `max() iterable argument is empty`
    failure, named at the boundary instead of surfacing from our internals."""
    with pytest.raises(InvalidInput):
        answer_margin({"C": frozenset({10})}, TOY_LP, "C")


def test_answer_margin_refuses_to_return_nan():
    """Both sides `-inf` is `-inf - -inf` = `nan`, and a `nan` in a per-item margin silently
    poisons every mean, percentile and weight computed from it downstream. The position carried no
    option mass at all, which is a fact worth a message rather than a quiet `nan`."""
    with pytest.raises(InvalidInput) as e:
        answer_margin(TOY_SETS, {99: 0.0}, "C")
    assert "no option mass" in str(e.value).lower()


def test_answer_margin_may_legitimately_be_infinite():
    """A total loss IS representable and must stay representable: if the correct option has no mass
    but a wrong one does, `-inf` is the honest answer and clamping it would invent a floor."""
    lp = {10: NEG_INF, 11: NEG_INF, 12: NEG_INF, 20: math.log(0.9), 30: NEG_INF}
    assert answer_margin(TOY_SETS, lp, "C") == NEG_INF


# ---- agreement -------------------------------------------------------------------------------

def test_answer_choice_is_the_argmax_option():
    assert answer_choice(TOY_SETS, TOY_LP) == "C"


def test_answer_choice_breaks_ties_on_the_given_option_order():
    """Deterministic by construction. A tie broken by dict iteration accident would make an
    arm-to-arm agreement rate depend on how the option list was spelled."""
    tie = {10: math.log(0.25), 11: NEG_INF, 12: NEG_INF, 20: math.log(0.25), 30: NEG_INF}
    assert answer_choice(TOY_SETS, tie) == "C"
    reordered = {"D": TOY_SETS["D"], "C": TOY_SETS["C"], "E": TOY_SETS["E"]}
    assert answer_choice(reordered, tie) == "D"


def test_answer_choice_rejects_a_position_with_no_option_mass():
    with pytest.raises(InvalidInput):
        answer_choice(TOY_SETS, {99: 0.0})


def test_answer_agreement_against_a_reference_label():
    assert answer_agreement(TOY_SETS, TOY_LP, "C") is True
    assert answer_agreement(TOY_SETS, TOY_LP, "D") is False


def test_answer_agreement_against_a_reference_distribution():
    ref = {10: math.log(0.05), 11: math.log(0.05), 12: math.log(0.05), 20: math.log(0.6),
           30: math.log(0.05)}
    assert answer_agreement(TOY_SETS, TOY_LP, ref) is False


def test_answer_agreement_rejects_an_unknown_reference_label():
    with pytest.raises(InvalidInput):
        answer_agreement(TOY_SETS, TOY_LP, "Z")


def test_option_agreement_is_better_founded_than_token_top1():
    """The design's 'free consequence', demonstrated rather than asserted.

    Reference and quant put their single largest token on DIFFERENT spellings of the SAME letter.
    Token-level `top1_disagree` -- the metric the objective weights today -- records a
    disagreement. There is none: both answered C. Counting spelling churn as an answer change is a
    measurement of formatting, and it is weighted in our objective right now."""
    ref = {10: math.log(0.30), 11: math.log(0.20), 12: math.log(0.10), 20: math.log(0.2),
           30: math.log(0.05)}
    quant = {10: math.log(0.20), 11: math.log(0.30), 12: math.log(0.10), 20: math.log(0.2),
             30: math.log(0.05)}
    assert max(ref, key=ref.get) != max(quant, key=quant.get)          # token-level: disagree
    assert answer_agreement(TOY_SETS, quant, ref) is True              # answer-level: agree
    assert answer_choice(TOY_SETS, ref) == answer_choice(TOY_SETS, quant) == "C"


# --------------------------------------------------------------------------------------------
# 6. The trie-divergence walk. Invariant (c): it terminates.
#
# `dist_fn` is injected, so every one of these is a complete test of the walk with no model, no
# weights and no llama.cpp. An adversarial distribution is one lambda here and an unreproducible
# quirk otherwise.
# --------------------------------------------------------------------------------------------
WALK_SETS = {"C": frozenset({10, 11}), "D": frozenset({20})}
STARS, THE, ANSWER, IS = 90, 91, 92, 93


def _counting(fn, cap: int | None = None):
    """Wrap a dist_fn so a test can assert how many scoring positions the walk consumed.

    `cap` makes the stub REFUSE to be called more than that many times. Without it, a walk that
    failed to terminate would hang the whole run instead of failing it -- verified by mutation:
    deleting the depth bound turns `test_walk_terminates_on_an_adversarial_always_shared_
    distribution` into an infinite loop, and a wedged CI runner is a worse signal than a red test,
    because nobody can tell it from a slow box. With the cap, non-termination surfaces as a
    failure at the exact call that went one too far.
    """
    calls: list[tuple[int, ...]] = []

    def wrapped(prefix):
        calls.append(tuple(prefix))
        if cap is not None and len(calls) > cap:
            raise AssertionError(
                f"dist_fn called {len(calls)} times, more than the {cap} the depth bound allows: "
                f"the divergence walk did not terminate (invariant c)")
        return fn(tuple(prefix))
    wrapped.calls = calls
    return wrapped


def test_walk_stops_immediately_when_the_position_already_decides():
    seen = _counting(lambda p: {10: math.log(0.6), 20: math.log(0.3), STARS: math.log(0.1)})
    res = divergence_walk(WALK_SETS, seen)
    assert res.stop == "diverged"
    assert res.depth == 0 and res.prefix == ()
    assert res.choice() == "C"
    assert len(seen.calls) == 1


def test_walk_steps_through_a_markdown_prefix_to_the_letter():
    """`**C**` begins with `**`, which every option shares, so that position carries no answer
    information. The walk is told nothing about markdown -- it observes that the mass sits on a
    token which is no option's surface form."""
    def fn(prefix):
        if prefix == ():
            return {STARS: math.log(0.97), 10: math.log(0.01), 20: math.log(0.02)}
        if prefix == (STARS,):
            return {10: math.log(0.7), 20: math.log(0.25), STARS: math.log(0.05)}
        raise AssertionError(f"walked too far: {prefix}")

    seen = _counting(fn)
    res = divergence_walk(WALK_SETS, seen)
    assert res.prefix == (STARS,)
    assert res.depth == 1
    assert res.stop == "diverged"
    assert res.choice() == "C"
    assert res.margin("C") > 0
    assert seen.calls == [(), (STARS,)]


def test_walk_steps_through_a_multi_token_preamble():
    """`The answer is C` -- three shared tokens, found without a pattern for it."""
    chain = {(): THE, (THE,): ANSWER, (THE, ANSWER): IS}

    def fn(prefix):
        nxt = chain.get(prefix)
        if nxt is not None:
            return {nxt: math.log(0.99), 10: math.log(0.005), 20: math.log(0.005)}
        return {20: math.log(0.8), 10: math.log(0.15)}

    res = divergence_walk(WALK_SETS, _counting(fn))
    assert res.prefix == (THE, ANSWER, IS)
    assert res.depth == 3
    assert res.choice() == "D"


def test_walk_terminates_on_an_adversarial_always_shared_distribution():
    """Invariant (c). This `dist_fn` puts ALL mass on a shared token forever: there is no natural
    stopping point, so termination must be structural. Bounded depth, and the bound is asserted on
    the number of scoring positions consumed, because that is the cost a real run pays.

    The stub is capped so that a non-terminating walk FAILS here rather than hanging the run."""
    seen = _counting(lambda p: {STARS: 0.0}, cap=6)
    res = divergence_walk(WALK_SETS, seen, max_depth=5)
    assert res.stop == "max_depth"
    assert res.depth == 5
    assert res.prefix == (STARS,) * 5
    assert len(seen.calls) == 6  # depth+1 positions, never more


def test_walk_default_depth_bounds_the_same_way():
    seen = _counting(lambda p: {STARS: 0.0}, cap=DEFAULT_MAX_DEPTH + 1)
    res = divergence_walk(WALK_SETS, seen)
    assert res.depth == DEFAULT_MAX_DEPTH
    assert len(seen.calls) == DEFAULT_MAX_DEPTH + 1


def test_walk_stops_when_nothing_dominates():
    """A flat position is not a shared prefix -- appending its argmax would be a guess. Stop, and
    let the near-zero `option_mass` tell the caller this position is not an answer slot."""
    flat = {STARS: math.log(0.2), THE: math.log(0.2), ANSWER: math.log(0.2), IS: math.log(0.2),
            10: math.log(0.01), 20: math.log(0.01)}
    res = divergence_walk(WALK_SETS, _counting(lambda p: flat))
    assert res.stop == "no_shared_prefix"
    assert res.depth == 0
    assert res.option_mass < DEFAULT_DIVERGENCE_MASS


EOG = 94


def test_walk_never_appends_an_end_of_generation_token():
    """A dominant EOG is no option surface, but what follows it is outside the reply (on Sharp-MiniCPM5-2B, an
    emoji's lead byte whose reply had no probabilities: a 500). The walk scores AT it, the options' conditional, and
    never asks for the position after it."""
    dist = {EOG: math.log(0.975), 10: math.log(0.012), 20: math.log(0.011)}
    seen = _counting(lambda p: TopLogprobs(dist, eog_id=EOG), cap=1)
    res = divergence_walk(WALK_SETS, seen, prefix=(1, 2))
    assert (res.stop, res.depth, res.prefix, len(seen.calls)) == ("eog", 0, (1, 2), 1)
    assert res.option_mass == pytest.approx(0.023)
    # the same values without the upstream's EOG verdict: appended as a shared prefix, exactly as before
    plain = divergence_walk(WALK_SETS, _counting(lambda p: dist if len(p) == 2 else {10: 0.0}, cap=2), prefix=(1, 2))
    assert (plain.stop, plain.prefix) == ("diverged", (1, 2, EOG))


def test_a_dominant_eog_at_the_depth_bound_is_reported_as_eog_not_max_depth():
    dist = TopLogprobs({EOG: math.log(0.9), 10: math.log(0.01)}, eog_id=EOG)
    assert divergence_walk(WALK_SETS, lambda p: dist, max_depth=0).stop == "eog"
    assert divergence_walk(WALK_SETS, lambda p: dict(dist), max_depth=0).stop == "max_depth"


def test_an_eog_verdict_counts_only_for_the_dominant_token_and_only_where_the_walk_would_append():
    """Options with real mass still diverge; a non-dominant EOG pick is no_shared_prefix; and an EOG verdict naming
    ANOTHER token than the dominant one (a server whose sampler moved the pick) never stops the walk."""
    diverged = TopLogprobs({EOG: math.log(0.6), 10: math.log(0.3)}, eog_id=EOG)
    assert divergence_walk(WALK_SETS, lambda p: diverged).stop == "diverged"
    flat = TopLogprobs({EOG: math.log(0.4), STARS: math.log(0.35), 20: math.log(0.01)}, eog_id=EOG)
    assert divergence_walk(WALK_SETS, lambda p: flat).stop == "no_shared_prefix"
    moved = TopLogprobs({STARS: math.log(0.6), EOG: math.log(0.3), 10: math.log(0.01)}, eog_id=EOG)
    res = divergence_walk(WALK_SETS, lambda p: moved if len(p) == 0 else {10: 0.0})
    assert (res.stop, res.prefix) == ("diverged", (STARS,))
    assert eog_token(flat) == EOG and eog_token(dict(flat)) is None and eog_token(TopLogprobs(flat)) is None
    assert TopLogprobs(flat, eog_id=EOG, reasks=2) == dict(flat)       # still the plain mapping it always was


def test_walk_never_appends_a_token_that_IS_an_option_surface():
    """The property that makes an explicit membership test unnecessary rather than merely absent.

    `dominance >= divergence_mass` is enforced, so any token dominant enough to append carries at
    least `dominance` of the mass; if it were an option's surface the option mass would be at least
    that, and the divergence branch -- checked first -- would already have stopped the walk. So an
    appended token is PROVABLY not an option surface, and the walk cannot swallow the answer."""
    res = divergence_walk(WALK_SETS, _counting(lambda p: {10: math.log(0.95), 20: math.log(0.05)}))
    assert res.stop == "diverged"
    assert res.prefix == ()
    assert res.choice() == "C"


def test_walk_threads_the_prefix_into_dist_fn_exactly():
    seen = _counting(lambda p: {STARS: math.log(0.99), 10: math.log(0.005), 20: math.log(0.005)}
                     if len(p) < 3 else {10: math.log(0.9), 20: math.log(0.05)})
    res = divergence_walk(WALK_SETS, seen, prefix=(THE,))
    assert seen.calls == [(THE,), (THE, STARS), (THE, STARS, STARS)]
    assert res.prefix == (THE, STARS, STARS)
    assert res.depth == 2  # depth counts steps WALKED, not the caller's starting prefix


def test_walk_with_zero_depth_scores_in_place():
    seen = _counting(lambda p: {STARS: 0.0}, cap=1)
    res = divergence_walk(WALK_SETS, seen, max_depth=0)
    assert res.stop == "max_depth"
    assert res.depth == 0 and len(seen.calls) == 1


def test_walk_result_agrees_with_the_direct_functions():
    res = divergence_walk(WALK_SETS, _counting(lambda p: {10: math.log(0.6), 20: math.log(0.3)}))
    assert isinstance(res, WalkResult)
    assert res.choice() == answer_choice(WALK_SETS, res.token_logprobs)
    assert res.margin("C") == answer_margin(WALK_SETS, res.token_logprobs, "C")
    assert _close(res.option_mass, option_mass(WALK_SETS, res.token_logprobs), 1e-12)
    assert res.option_logprobs == marginal_option_logprobs(WALK_SETS, res.token_logprobs)


# ---- the walk's guards -----------------------------------------------------------------------

def test_walk_rejects_a_negative_depth_bound():
    with pytest.raises(InvalidInput):
        divergence_walk(WALK_SETS, lambda p: {10: 0.0}, max_depth=-1)


def test_walk_rejects_a_dominance_below_the_divergence_threshold():
    """Not a stylistic bound: it is the precondition of the no-swallowed-answer proof above. Below
    it, a token could be dominant enough to append AND be an option's own surface form."""
    with pytest.raises(InvalidInput) as e:
        divergence_walk(WALK_SETS, lambda p: {10: 0.0}, dominance=0.05, divergence_mass=0.5)
    assert "dominance" in str(e.value)


@pytest.mark.parametrize("kw", [{"dominance": 0.0}, {"dominance": 1.5},
                                {"divergence_mass": 0.0}, {"divergence_mass": 1.5},
                                {"dominance": float("nan")}])
def test_walk_rejects_thresholds_outside_the_unit_interval(kw):
    with pytest.raises(InvalidInput):
        divergence_walk(WALK_SETS, lambda p: {10: 0.0}, **kw)


def test_walk_rejects_an_empty_distribution():
    """A scorer that returned nothing has failed; reading it as `no option mass` would walk the
    full depth and report a confident `max_depth` result built on no data."""
    with pytest.raises(InvalidInput):
        divergence_walk(WALK_SETS, lambda p: {})


def test_walk_rejects_a_distribution_carrying_nan():
    with pytest.raises(InvalidInput):
        divergence_walk(WALK_SETS, lambda p: {10: float("nan"), 20: -1.0})


def test_walk_rejects_a_distribution_carrying_a_positive_logprob():
    with pytest.raises(InvalidInput):
        divergence_walk(WALK_SETS, lambda p: {10: 2.0, 20: -1.0})


def test_walk_defaults_are_ordered_so_the_proof_holds():
    """The shipped defaults must themselves satisfy the precondition, or every default-argument
    call would be relying on a property that does not hold."""
    assert 0 < DEFAULT_DIVERGENCE_MASS <= DEFAULT_DOMINANCE <= 1.0
    assert DEFAULT_MAX_DEPTH >= 1


# ---- callo v0.3: Gemma 4's `gemma4` vocabulary decodes as SentencePiece ----------

def test_decode_gemma4_metaspace_and_byte_fallback():
    """llama.cpp treats `gemma4` as SPM-style BPE (escape_whitespaces, `<0xNN>` byte tokens), so callo decodes it as
    SentencePiece: the local gemma-4-E4B vocabulary has `▁A`, `▁B` and `<0x41>` and no `" A"`."""
    assert decode_token("▁A", "gemma4") == " A"
    assert decode_token("<0x41>", "gemma4") == "A"
    assert decode_token("▁▁B", "gemma4") == "  B"


def test_gemma4_option_sets_collect_the_metaspace_and_byte_spellings():
    tokens = ["<pad>", "▁A", "A", "▁B", "B", "<0x41>", "<0x42>", "▁the", "**", "▁(A"]
    sets = option_surface_sets(tokens, "gemma4", ["A", "B"])
    assert sets == {"A": frozenset({1, 2, 5, 9}), "B": frozenset({3, 4, 6})}
    check_option_sets(sets, tokens=tokens)


# ---- the option-mass tolerance: float32 rounding is not unnormalised input --------------------------------------
# Gemma 4 E4B and MiniCPM5-2B both failed their first /v1/systemone readout (and Gemma its serve self-test) with
# "option mass is 1.000000 ... above 1": a near-certain answer whose float32 top-k probabilities summed to 1 + ~4e-7,
# refused by a fixed 1e-9 mass slack that was below even exp(_LOGPROB_TOL) - 1.

NEAR_SETS = {"A": frozenset(range(1, 41)), "B": frozenset(range(41, 81))}   # 40 surface ids per option


def _near_certain(excess):
    """{id: logprob} over NEAR_SETS whose probabilities sum to exactly 1 + excess (in float64): A near-certain on id 1,
    the rest spread over 79 small spellings -- the shape float32 rounding pushes a hair past 1."""
    small = [1e-5 * (1 + (i % 7)) for i in range(79)]
    probs = {1: 1.0 + excess - math.fsum(small)}
    probs.update(dict(zip(range(2, 81), small, strict=True)))
    assert abs(math.fsum(probs.values()) - (1.0 + excess)) < 1e-15
    return {tid: math.log(p) for tid, p in probs.items()}


def test_option_mass_accepts_a_sum_float32_rounding_puts_just_above_one():
    lps = _near_certain(5e-7)
    raw = math.fsum(math.exp(v) for v in marginal_option_logprobs(NEAR_SETS, lps).values())
    assert raw > 1.0 + 1e-9                     # what the old fixed slack refused
    assert option_mass(NEAR_SETS, lps) == 1.0   # accepted, and clamped: nothing downstream sees a mass above 1


def test_option_mass_accepts_a_single_logprob_just_above_zero():
    """DEFENSIVE, not a reproduction: llama.cpp cannot emit a positive logprob (every p = e/cum_sum <= 1). But
    `_check_logprob` already lets a single logprob sit up to 1e-6 above 0, so the mass check must not then refuse the
    exp of that same legal value."""
    lps = {10: 5e-7, 20: -30.0}
    assert option_mass({"C": frozenset({10}), "D": frozenset({20})}, lps) == 1.0
    res = divergence_walk({"C": frozenset({10}), "D": frozenset({20})}, lambda p: lps)
    assert res.stop == "diverged" and res.option_mass == 1.0 and res.choice() == "C"
    assert marginal_option_logprobs(res.sets, lps)["C"] == 5e-7                   # raw: kept for argmax and margin
    assert res.option_logprobs == {"C": 0.0, "D": -30.0}    # the readout: no log-probability above 0 on the wire


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def test_swamped_terms_push_the_mass_past_one_and_are_accepted():
    """Deterministic reproduction of the real mechanism, every logprob <= 0. In float32 the top token has e = 1 and the
    other terms e = 0.999 * 2^-24, each below half an ulp of 1, so the SEQUENTIAL float32 cum_sum never leaves exactly 1
    and p_i = e_i / 1. The option sums all of them in float64: 1 + k * 0.999 * 2^-24, a hair above 1."""
    small = _f32(0.999 * 2.0 ** -24)
    assert _f32(1.0 + small) == 1.0                              # swamped: the float32 cum_sum stays exactly 1
    lps = {1: 0.0}                                               # p = 1.0 / 1.0
    lps.update({tid: _f32(math.log(small)) for tid in range(2, 81)})   # 79 swamped terms: logprob ~ -16.6
    assert all(v <= 0.0 for v in lps.values())
    raw = math.fsum(math.exp(v) for v in marginal_option_logprobs(NEAR_SETS, lps).values())
    assert raw - 1.0 > 1e-9                                      # the parent commit's fixed 1e-9 slack refused this
    assert raw - 1.0 < 1e-5
    assert option_mass(NEAR_SETS, lps) == 1.0
    res = divergence_walk(NEAR_SETS, lambda p: lps)
    assert res.stop == "diverged" and res.option_mass == 1.0 and res.choice() == "A"
    assert all(v <= 0.0 for v in res.option_logprobs.values())


def test_the_walk_accepts_the_float32_overshoot_and_clamps_its_readout():
    """The exact failure: divergence_walk raised at the first scoring position of a near-certain answer."""
    lps = _near_certain(5e-7)
    res = divergence_walk(NEAR_SETS, lambda p: lps)
    assert res.stop == "diverged" and res.depth == 0 and res.choice() == "A"
    assert res.option_mass == 1.0
    assert all(v <= 0.0 for v in res.option_logprobs.values())
    assert res.option_logprobs == marginal_option_logprobs(NEAR_SETS, lps)   # all below 0 here: untouched
    assert res.margin("A") > 0 and math.isfinite(res.margin("A"))


def test_a_replay_of_llama_cpp_float32_softmax_is_accepted():
    """llama.cpp's raw n_probs path (server-common.cpp `get_token_probabilities` @ 4ebdf2c), replayed in float32: expf
    of (logit - max), a SEQUENTIAL float32 sum over the whole vocabulary with the top n_probs first, a float32 divide,
    then a float32 `std::log`. A near-certain option set over a 262144-token vocabulary sums a hair past 1."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    vocab, n_top = 262144, 100
    ids = rng.choice(vocab, size=80, replace=False)
    sets = {"A": frozenset(int(t) for t in ids[:40]), "B": frozenset(int(t) for t in ids[40:])}
    worst = 0.0
    for _ in range(8):
        logits = rng.normal(0.0, 3.0, vocab).astype(np.float32)
        logits[ids[:40]] = (25.0 + rng.normal(0.0, 3.0, 40)).astype(np.float32)
        order = np.argsort(-logits, kind="stable")
        cur = np.concatenate([order[:n_top], rng.permutation(order[n_top:])])
        e = np.exp(logits[cur] - logits[cur[0]]).astype(np.float32)
        p = (e / np.cumsum(e, dtype=np.float32)[-1]).astype(np.float32)
        lp = np.log(p).astype(np.float32)
        dump = {int(t): float(v) for t, v in zip(cur[:n_top], lp[:n_top], strict=True)}
        raw = math.fsum(math.exp(v) for v in marginal_option_logprobs(sets, dump).values())
        worst = max(worst, raw - 1.0)
        assert option_mass(sets, dump) == min(raw, 1.0)
        assert divergence_walk(sets, lambda p, dump=dump: dump).option_mass == min(raw, 1.0)
    # acceptance-only: whether a given draw overshoots depends on numpy's float32 exp/log, so `worst` is not asserted
    # (the deterministic reproduction is test_swamped_terms_push_the_mass_past_one_and_are_accepted)
    assert worst < 1e-5


def test_the_mass_tolerance_grows_with_the_terms_and_stays_far_below_a_second_option():
    from callo.marginalize import _LOGPROB_TOL, _MASS_TOL_PER_TERM, _mass_tol
    assert _mass_tol(0) == _LOGPROB_TOL
    assert _mass_tol(100) == pytest.approx(_LOGPROB_TOL + 100 * 2.0 ** -22) and _mass_tol(100) < 2.6e-5
    assert _MASS_TOL_PER_TERM >= 2.74 * 2.0 ** -24          # the stated per-term float32 bound, with margin
    assert math.exp(_LOGPROB_TOL) - 1.0 < _mass_tol(1)      # a single legal logprob's exp is always accepted


@pytest.mark.parametrize("excess", [1e-3, 1e-4, 1.0])
def test_genuinely_unnormalised_mass_is_still_refused(excess):
    """Mass 2 (two certain options), and anything above 1 by more than float32 rounding of the summed terms can
    explain, is still refused -- by option_mass and by the walk."""
    lps = _near_certain(excess) if excess < 1.0 else {1: 0.0, 41: 0.0}
    with pytest.raises(InvalidInput, match="above 1"):
        option_mass(NEAR_SETS, lps)
    with pytest.raises(InvalidInput, match="at prefix"):
        divergence_walk(NEAR_SETS, lambda p: lps)


def test_two_distributions_mixed_together_are_still_refused():
    """A near-certain A from one distribution and a near-certain B from another: each value legal, the sum ~2."""
    lps = {1: math.log(0.999), 41: math.log(0.998)}
    with pytest.raises(InvalidInput, match="above 1"):
        option_mass(NEAR_SETS, lps)
