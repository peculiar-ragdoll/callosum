"""Callo 0.3.1: wide questions past a vocabulary's `wide_coverage`. A label that is not one token both bare and after a
space is UNCOVERED; it is read in two passes, first letter then second letter, with the split machinery wide questions
already had (the prompt is unchanged). Each first letter that leads an uncovered label of the question adds one split
pass to the cap. Within the coverage, every readout is 0.3.0's, bit for bit (`tests/data/wide_within_coverage.json`,
captured on 5532add).

The Spark-like vocabulary covers 9 labels, like Spark: past them every third label is bare only, BK and HQ have no
token at all, and the rest are bare and spaced. The letters-only vocabulary covers none."""
import json
import math
import platform
import string
from pathlib import Path

import pytest
import wide_coverage_fixtures as fx
from fake_upstream import FakeUpstream
from test_compare import profile as compare_profile
from test_engine import DELIM, MARKER, PNG, URL, ChatMLTemplate, MMScorer
from test_think_engine import SAMPLING
from test_wide_engine import _think_engine
from test_wide_identity_sweep import _diffs
from test_wide_sets import SPARK as SPARK_GGUF
from test_wide_sets import _need
from test_wide_sets import _real as _real_gguf

from callo.cli import wide_note
from callo.compare import flags as compare_flags
from callo.engine import Engine, SurfaceSets
from callo.errors import InvalidInput, TooManyOptions
from callo.marginalize import WIDE_MAX_SPLIT_PASSES, check_option_sets, divergence_walk, option_mass
from callo.profile import WIDE_CONTRACT, provenance, wide_contract_warning
from callo.systemone import TOO_MANY_OPTIONS, WIDE_LABELS, parse_request

LETTERS = list(string.ascii_uppercase)
BASE = LETTERS + ["Ġ" + c for c in LETTERS] + ["A.", "**", "Ġthe"]
GOLDEN = json.loads((Path(__file__).parent / "data" / "wide_within_coverage.json").read_text(encoding="utf-8"))


def lp(p):
    return math.log(p)


def spark_like() -> list[str]:
    toks = list(BASE)
    for i, lab in enumerate(WIDE_LABELS):
        if lab in ("BK", "HQ"):
            continue
        toks.append(lab)
        if i < 9 or i % 3:
            toks.append("Ġ" + lab)
    return toks


SPARK = spark_like()
SID = {t: i for i, t in enumerate(SPARK)}
SPARK_SETS = SurfaceSets(SPARK, "gpt2")
LETTERS_ONLY = SurfaceSets(list(BASE), "gpt2")
A_, B_, C_, GA, GB, GC, THE, STARS = (SID[t] for t in ("A", "B", "C", "ĠA", "ĠB", "ĠC", "Ġthe", "**"))


def choice(n: int):
    return parse_request({"state": "s", "questions": {"q": {"type": "choice",
                                                            "criteria": {f"k{i}": f"d{i}" for i in range(n)}}}}
                         ).questions[0]


class Scorer:
    """`dists` maps the tokens appended after the 3 prompt tokens (a tuple) to a sparse top-k dump."""

    def __init__(self, dists):
        self.dists, self.calls = dists, []

    def tokenize(self, text, *, add_special=True, parse_special=None):
        return [7, 8, 7]

    def next_logprobs(self, tokens, *, top_k, id_slot=None, delimiter=None):
        self.calls.append((tuple(tokens[3:]), top_k))
        return self.dists[tuple(tokens[3:])]


class Template:
    def render(self, msgs):
        return "PROMPT"


def plain(dists, sets=SPARK_SETS):
    sc = Scorer(dists)
    return Engine(sc, Template(), sets), sc


def mass(lm: float) -> float:
    return math.exp(lm)


# ---- the sets --------------------------------------------------------------------------------------------------

def test_a_spark_like_vocabulary_covers_9_and_reads_255():
    s = SurfaceSets(spark_like(), "gpt2")
    assert (s.wide_coverage(), s.wide_readable(), s.warm_wide()) == (9, 255, 9)
    labels = WIDE_LABELS[:27]
    want = tuple(lab for i, lab in enumerate(labels) if i >= 9 and (i % 3 == 0 or lab == "BK"))
    assert s.uncovered(labels) == want and want[0] == "AO" and "BK" in want
    assert s.split_letters(labels) == ("A", "B") and s.uncovered(WIDE_LABELS[:9]) == ()
    w = s.wide(27)
    assert w["BK"] == frozenset() and w["AO"] == frozenset({SID["AO"]}) and w["AB"] == frozenset({SID["AB"], SID["ĠAB"]})
    assert s.uncovered(WIDE_LABELS) == tuple(lab for i, lab in enumerate(WIDE_LABELS)
                                            if i >= 9 and (i % 3 == 0 or lab in ("BK", "HQ")))


def test_a_letters_only_vocabulary_covers_none_and_reads_every_label_in_two_passes():
    s = SurfaceSets(list(BASE), "gpt2")
    assert (s.warm_wide(), s.wide_readable()) == (0, 255)
    assert s.uncovered(WIDE_LABELS) == WIDE_LABELS and all(not v for v in s.wide(255).values())
    assert s.split_letters(WIDE_LABELS) == tuple("ABCDEFGHIJKLM")


def test_a_letter_that_is_no_continuation_is_still_the_422():
    """No token continues a label with Q: the first uncovered label ending in Q is the 422 limit, with the exact
    message; a covered Q label (`EQ`, one token both ways) still reads, and the limit moves to the next Q label."""
    toks = [("<Q>" if t == "Q" else t) for t in BASE] + [t for lab in WIDE_LABELS[:9] for t in (lab, "Ġ" + lab)]
    s = SurfaceSets(toks, "gpt2")
    q_labels = [i for i, lab in enumerate(WIDE_LABELS) if lab[1] == "Q"]
    assert s.wide_readable() == q_labels[0] and WIDE_LABELS[q_labels[0]] == "EQ"
    eng = Engine(None, None, s)
    eng.check_coverage(choice(q_labels[0]))
    with pytest.raises(TooManyOptions) as e:
        eng.check_coverage(choice(q_labels[0] + 1))
    assert str(e.value) == TOO_MANY_OPTIONS.format(qid="q", n=q_labels[0], m=q_labels[0] + 1)
    s2 = SurfaceSets(toks + ["EQ", "ĠEQ"], "gpt2")
    assert s2.wide_readable() == q_labels[1]


def test_the_walk_waives_non_emptiness_only_when_asked():
    """`allow_empty`: an empty option set reads -inf and the walk stops on first-letter mass; without it, invariant
    (a) refuses the empty set exactly as before; disjointness is checked either way."""
    sets = {"AA": frozenset(), "AB": frozenset({5}), "BA": frozenset()}
    w = divergence_walk(sets, lambda p: {1: lp(0.6), 5: lp(0.1), 9: lp(0.3)}, partial={1}, allow_empty=True)
    assert w.option_logprobs["AA"] == -math.inf and w.stop == "diverged" and w.partial_mass == pytest.approx(0.6)
    with pytest.raises(InvalidInput, match="NO surface form"):
        divergence_walk(sets, lambda p: {1: lp(0.6), 9: lp(0.4)}, partial={1})
    with pytest.raises(InvalidInput, match="not disjoint"):
        check_option_sets({"AA": frozenset(), "AB": frozenset({5}), "BA": frozenset({5})}, allow_empty=True)
    assert option_mass(sets, {5: lp(0.25)}, allow_empty=True) == pytest.approx(0.25)


# ---- the readout ------------------------------------------------------------------------------------------------

def test_an_uncovered_label_reads_first_letter_times_second_letter():
    """27 options on the Spark-like vocabulary. `BK` has no token at all: P(BK) = P(ĠB) P(K | ĠB), exactly. `AO` is
    bare only and its token is not in the dump: like any label absent from the top-k, its direct part is bounded by
    the dump's last logprob (the floor, 0.1 here), and P(A) P(O | A) is added. `AB` is covered and adds its split
    credit to its direct mass."""
    eng, sc = plain({(): {A_: lp(0.4), GB: lp(0.2), SID["AB"]: lp(0.1), THE: lp(0.3)},
                     (A_,): {SID["O"]: lp(0.5), B_: lp(0.25), THE: lp(0.25)},
                     (GB,): {SID["K"]: lp(0.9), THE: lp(0.1)}})
    q = choice(27)
    r = eng.answer("s", q)
    at = {lab: i for i, lab in enumerate(q.labels)}
    assert mass(r.log_marginals[at["AO"]]) == pytest.approx(0.1 + 0.4 * 0.5, rel=1e-12)
    assert mass(r.log_marginals[at["BK"]]) == pytest.approx(0.2 * 0.9, rel=1e-12)
    assert mass(r.log_marginals[at["AB"]]) == pytest.approx(0.1 + 0.4 * 0.25, rel=1e-12)
    assert r.option_mass == pytest.approx(0.2 + 0.18 + 0.2, rel=1e-12) and r.unread_mass == 0.0
    assert r.option_mass + r.unread_mass <= 1.0
    assert (r.stop, r.depth, r.passes, r.split_passes, r.wide) == ("diverged", 0, 3, 2, True)
    assert r.uncovered == len(SPARK_SETS.uncovered(q.labels)) > 0 and r.topk_miss == 27 - 3
    assert [c[0] for c in sc.calls] == [(), (A_,), (GB,)] and all(c[1] == 1000 for c in sc.calls)
    assert sum(r.probs) == pytest.approx(1.0) and max(range(27), key=r.probs.__getitem__) == at["AO"]


def test_a_letters_only_vocabulary_reads_a_question_entirely_in_two_passes():
    """Coverage 0: every label's set is empty. A dominant first letter stops the walk (never appended), and a
    credited label's mass is P(first) x P(second | first), with no floor (it has no token to be below the cut); one
    with nothing read is floored like any other."""
    ids = {t: i for i, t in enumerate(BASE)}
    A, B, C, GB_, THE_ = ids["A"], ids["B"], ids["C"], ids["ĠB"], ids["Ġthe"]
    eng, sc = plain({(): {A: lp(0.7), GB_: lp(0.2), THE_: lp(0.1)},
                     (A,): {B: lp(0.5), C: lp(0.3), THE_: lp(0.2)},
                     (GB_,): {A: lp(1.0)}}, sets=LETTERS_ONLY)
    q = choice(30)
    r = eng.answer("s", q)
    at = {lab: i for i, lab in enumerate(q.labels)}
    assert r.stop == "diverged" and r.depth == 0 and r.split_passes == 2 and r.uncovered == 30
    assert mass(r.log_marginals[at["AB"]]) == pytest.approx(0.35, rel=1e-12)
    assert mass(r.log_marginals[at["AC"]]) == pytest.approx(0.21, rel=1e-12)
    assert mass(r.log_marginals[at["BA"]]) == pytest.approx(0.2, rel=1e-12)
    assert r.option_mass == pytest.approx(0.76, rel=1e-12) and r.unread_mass == 0.0 and r.topk_miss == 27
    assert mass(r.log_marginals[at["AA"]]) == pytest.approx(0.1)                # the floor (Ġthe), as a bound
    eng, _ = plain({(): {A: lp(0.95), THE_: lp(0.05)}, (A,): {B: lp(1.0)}}, sets=LETTERS_ONLY)
    r = eng.answer("s", q)
    assert (r.depth, r.stop) == (0, "diverged") and mass(r.log_marginals[at["AB"]]) == pytest.approx(0.95)


def test_each_first_letter_leading_an_uncovered_label_adds_a_split_pass():
    """60 options lead with A, B and C; on the Spark-like vocabulary each leads an uncovered label, so the cap is
    2 + 3 = 5: five of the six equal first-letter tokens are read (id order on ties) and the sixth stays unread. On a
    full-coverage vocabulary the same dump reads 2."""
    six = {t: lp(0.1) for t in (A_, B_, C_, GA, GB, GC)}
    dists = {(): {**six, THE: lp(0.4)}, **{(t,): {THE: 0.0} for t in six}}
    q = choice(60)
    assert SPARK_SETS.split_letters(q.labels) == ("A", "B", "C") and q.labels[-1][0] == "C"
    eng, sc = plain(dists)
    r = eng.answer("s", q)
    assert r.split_passes == WIDE_MAX_SPLIT_PASSES + 3 == 5 and r.passes == 6
    assert r.unread_mass == pytest.approx(0.1) and r.option_mass == 0.0
    assert [c[0] for c in sc.calls][1:] == [(A_,), (B_,), (C_,), (GA,), (GB,)]
    full = SurfaceSets(fx.vocab(WIDE_LABELS, 255), "gpt2")
    r = plain(dists, sets=full)[0].answer("s", q)
    assert r.split_passes == 2 and r.unread_mass == pytest.approx(0.4) and r.uncovered == 0


def test_the_reserved_passes_do_not_reorder_a_read_that_reaches_every_letter():
    """Two spellings of A outweigh B and C: the greedy order (A, ĠA, B, ĠB, C) fits the cap of 5 and already reads
    every reserved letter, so the reservation changes nothing."""
    dump = {A_: lp(0.3), GA: lp(0.25), B_: lp(0.2), GB: lp(0.12), C_: lp(0.08), THE: lp(0.05)}
    eng, sc = plain({(): dump, **{(t,): {THE: 0.0} for t in dump if t != THE}})
    r = eng.answer("s", choice(60))
    assert [c[0] for c in sc.calls][1:] == [(A_,), (GA,), (B_,), (GB,), (C_,)] and r.unread_mass == 0.0


def test_a_letter_leading_an_uncovered_label_keeps_its_pass_when_likelier_spellings_fill_the_cap():
    """255 options on the Spark-like vocabulary (13 letters lead an uncovered label: cap 15). A to G each hold two
    spellings near 0.05, I one, and the answer is HQ (no token at all) through ĠH at 0.045. Greedy by mass would spend
    all 15 passes on A-G and I and leave ĠH unread; the reserved pass reads it, and HQ comes out on top."""
    q = choice(255)
    assert len(SPARK_SETS.split_letters(q.labels)) == 13
    unc = SPARK_SETS.uncovered(q.labels)

    def spread(letter):
        labs = [lab for lab in unc if lab[0] == letter]
        return {SID[lab[1]]: lp(0.95 / len(labs)) for lab in labs} | {THE: lp(0.05)}
    slot, dists = {}, {}
    for c in "ABCDEFG":
        slot[SID[c]], slot[SID["Ġ" + c]] = lp(0.055), lp(0.05)
        dists[(SID[c],)] = dists[(SID["Ġ" + c],)] = spread(c)
    slot[SID["I"]], dists[(SID["I"],)] = lp(0.05), spread("I")
    slot[SID["ĠH"]], dists[(SID["ĠH"],)] = lp(0.045), {SID["Q"]: lp(0.95), THE: lp(0.05)}
    slot[THE] = lp(1 - math.fsum(math.exp(v) for v in slot.values()) - 1e-6)
    slot[SID["A."]] = lp(1e-6)
    dists[()] = slot
    eng, sc = plain(dists)
    r = eng.answer("s", q)
    assert r.split_passes == 15 and (SID["ĠH"],) in [c[0] for c in sc.calls]
    assert max(range(255), key=r.probs.__getitem__) == q.labels.index("HQ")
    assert mass(r.log_marginals[q.labels.index("HQ")]) == pytest.approx(0.045 * 0.95, rel=1e-12)


def test_the_uncapped_diagnostic_stays_uncapped_and_the_unread_threshold_still_stops():
    six = {t: lp(0.1) for t in (A_, B_, C_, GA, GB, GC)}
    dists = {(): {**six, THE: lp(0.4)}, **{(t,): {THE: 0.0} for t in six}}
    eng, _ = plain(dists)
    eng.split_max_passes = None
    assert eng.answer("s", choice(60)).split_passes == 6
    eng, _ = plain({(): {A_: lp(0.995), GB: lp(0.005)}, (A_,): {SID["O"]: 0.0}})
    r = eng.answer("s", choice(77))
    assert r.split_passes == 1 and r.unread_mass == pytest.approx(0.005)       # below 0.01: left unread


def test_within_the_coverage_every_readout_is_the_030_readout_bit_for_bit():
    """The pin: questions of at most `wide_coverage` options (on a 40-label and a 255-label vocabulary, 30 to 255
    options, direct, split, capped, uncapped, walked and floored readouts, and a 26-option one) read exactly what
    0.3.0 read, posted the same passes, and carry `uncovered` 0."""
    now = json.loads(json.dumps(fx.snapshot(Engine, SurfaceSets, parse_request, WIDE_LABELS)))
    exact = GOLDEN["platform"] == f"{platform.system()}-{platform.machine()}"   # else libm may round the last ulp
    assert _diffs(GOLDEN["readouts"], now, exact=exact) == []
    sets = SurfaceSets(fx.vocab(WIDE_LABELS, 40), "gpt2")
    for name, (size, n, dists, _) in fx.scenarios(WIDE_LABELS).items():
        if size == 40:
            assert Engine(fx._Scorer(dists), fx._Template(), sets).answer("s", choice(n)).uncovered == 0, name


# ---- the other paths: retok, context, deliberate -------------------------------------------------------------

def _media_engine(sc):
    return Engine(sc, ChatMLTemplate(), SPARK_SETS, vision=True, media_marker=MARKER, user_delimiter=DELIM)


def test_a_retok_stop_on_an_uncovered_question_reads_the_last_position():
    """The walk stops on retok before the slot: the readout at the last scored position waives the empty sets too,
    and its first-letter mass is unread."""
    sc = MMScorer({0: {STARS: lp(0.95), A_: lp(0.03), THE: lp(0.02)}}, detok={STARS: "\n"})
    r = _media_engine(sc).answer("s", choice(27), images=[PNG], image_urls=[URL])
    assert r.stop == "retok" and r.split_passes == 0 and r.unread_mass == pytest.approx(0.03) and r.uncovered > 0


def test_a_context_readout_splits_an_uncovered_label():
    sc = MMScorer({0: {GB: lp(0.6), THE: lp(0.4)}, 1: {SID["K"]: lp(0.5), THE: lp(0.5)}})
    q = choice(27)
    r = _media_engine(sc).answer("s", q, context="ctx")
    assert r.kind == "context" and r.split_passes == 1
    assert mass(r.log_marginals[q.labels.index("BK")]) == pytest.approx(0.3, rel=1e-12)


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=5).start()
    yield u
    u.stop()


def test_a_deliberate_readout_past_the_coverage_is_read_not_refused(up):
    up.bias = None
    up.top_logprobs = [{"id": A_, "token": "A", "logprob": lp(0.4)}, {"id": SID["AB"], "token": "AB", "logprob": lp(0.3)},
                       {"id": B_, "token": "B", "logprob": lp(0.2)}, {"id": THE, "token": "Ġthe", "logprob": lp(0.1)}]
    e = _think_engine(up, SPARK_SETS)
    d = e.answer_deliberate("s", choice(27), [64], id_slot=0, seed=1, sampling=SAMPLING)
    r = d.readouts[64].readout
    assert r.bias == "deliberate" and r.wide and r.uncovered > 0 and r.split_passes == 2 and r.unread_mass == 0.0


# ---- surfaces: the log line, the wide contract ------------------------------------------------------------------

def test_the_startup_note():
    assert wide_note(255, 255) == ""                                          # a full-coverage line is unchanged
    assert wide_note(155, 255) == " (beyond 155, a label that is not a single token is read in two passes)"
    assert wide_note(9, 255) == " (beyond 9, a label that is not a single token is read in two passes)"
    assert wide_note(9, 9) == wide_note(3, 26) == " (wide questions are refused)"


def _prov(two_pass: bool) -> dict:
    return provenance(chat_template="t", build_info="b", mmproj=None, user_delimiter=None, cache_prompt=True,
                      wide_two_pass=two_pass)


def test_the_wide_contract_marks_two_pass_reading_only_where_it_can_happen():
    assert _prov(False)["wide_contract"] == WIDE_CONTRACT                     # full coverage: 0.3.0's contract
    assert _prov(True)["wide_contract"] == {**WIDE_CONTRACT, "uncovered": "two_pass"}
    assert wide_contract_warning(WIDE_CONTRACT, _prov(False)["wide_contract"], "--think-profile") is None
    line = wide_contract_warning(WIDE_CONTRACT, _prov(True)["wide_contract"], "--think-profile")
    assert line is not None and "two_pass" in line


def test_compare_names_profiles_from_either_side_of_two_pass_reading():
    """A short-coverage model's profile from 0.3.0 (no two-pass reading) and one from 0.3.1 carry different wide
    contracts, and compare flags them; a full-coverage model's profiles from either side carry the same one."""
    what = "different wide-choice readout contracts"
    old = compare_profile("old", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0},
                          comparable__readout__wide_contract=dict(WIDE_CONTRACT))
    new = compare_profile("new", {0: 6, 256: 9}, {0: 400.0, 256: 2500.0}, key="k2",
                          comparable__readout__wide_contract=_prov(True)["wide_contract"])
    full = compare_profile("full", {0: 6, 256: 9}, {0: 400.0, 256: 2500.0}, key="k3",
                           comparable__readout__wide_contract=_prov(False)["wide_contract"])
    hit = [f for f in compare_flags([old, new]) if f.startswith(what)]
    assert len(hit) == 1 and "two_pass" in hit[0]
    assert not any(f.startswith(what) for f in compare_flags([old, full]))


def _real_sets(path):
    _need(path)
    return _real_gguf(path)


def test_the_real_spark_vocabulary_reads_bk_in_two_passes():
    """Spark's own vocabulary (gated: skipped when the GGUF is not on the box): BK has no token at all, so a 27-option
    question reads it as P(ĠB) x P(K | ĠB)."""
    spark = _real_sets(SPARK_GGUF)
    tid = {t: i for i, t in enumerate(spark._tokens)}
    q = choice(27)
    assert spark.wide_coverage() == 9 and not spark.get(q.labels)["BK"]
    eng, sc = plain({(): {tid["ĠB"]: lp(0.6), tid["Ġthe"]: lp(0.4)}, (tid["ĠB"],): {tid["K"]: lp(0.5), tid["Ġthe"]: lp(0.5)}},
                    sets=spark)
    r = eng.answer("s", q)
    assert r.split_passes == 1 and r.uncovered == len(spark.uncovered(q.labels)) > 0
    assert mass(r.log_marginals[q.labels.index("BK")]) == pytest.approx(0.3, rel=1e-12)
