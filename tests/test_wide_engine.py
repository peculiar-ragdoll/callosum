"""callo v0.3 wide-choice, the engine readout: the shared slot reader (`_read_slot`), WIDE_TOP_K on every pass, split passes in `passes`, credit
flooring (`wide_bounded_readout`), the coverage check on every path, the media retok skip, deliberate and
chunked readouts, the `wide` bias category, and score questions above 26 options.

The synthetic gpt2 vocabulary: ids 0-25 are `A`..`Z`, 26-51 `ĠA`..`ĠZ`, then `A.` (52), `**` (53), `Ġthe` (54), then
each WIDE_LABELS[i] bare (55 + 2i) and spaced (56 + 2i)."""
import base64
import json
import math
import string
from pathlib import Path

import pytest
from fake_upstream import FakeUpstream
from narrow_sweep import unpinned
from test_engine import DELIM, MARKER, PNG, URL, ChatMLTemplate, MMScorer
from test_profile import profile_dict
from test_think_engine import REQ, SAMPLING
from test_think_engine import engine_for as narrow_engine_for

from callo.engine import WIDE_TOP_K, Engine, Readout, SurfaceSets, bounded_readout, wide_bounded_readout
from callo.errors import NoVision, TooManyOptions
from callo.marginalize import WIDE_MAX_SPLIT_PASSES, WIDE_SPLIT_READ
from callo.profile import load_profile
from callo.prompt import TemplateClient
from callo.scorer import LlamaCppForwardScorer, make_http_post
from callo.systemone import WIDE_LABELS, answer_body, parse_request
from callo.think import frame_from_template

LETTERS = list(string.ascii_uppercase)
NEG_INF = float("-inf")
FROZEN = json.loads((Path(__file__).parent / "data" / "narrow_chunked_probes.json").read_text(encoding="utf-8"))


def lp(p):
    return math.log(p)


def vocab(n_labels: int = 255, *, drop_all: int | None = None) -> list[str]:
    toks = LETTERS + ["Ġ" + c for c in LETTERS] + ["A.", "**", "Ġthe"]
    for i, lab in enumerate(WIDE_LABELS[:n_labels]):
        if i != drop_all:
            toks += [lab, "Ġ" + lab]
    return toks


VOCAB = vocab()
ID = {t: i for i, t in enumerate(VOCAB)}
GA, GB, A_, B_, STARS, THE = ID["ĠA"], ID["ĠB"], ID["A"], ID["B"], ID["**"], ID["Ġthe"]
SETS = SurfaceSets(VOCAB, "gpt2")


def lab_ids(lab: str) -> tuple[int, int]:
    return ID[lab], ID["Ġ" + lab]


def choice(n: int, qid: str = "q"):
    return parse_request({"state": "s", "questions": {qid: {"type": "choice",
                                                            "criteria": {f"k{i}": f"d{i}" for i in range(n)}}}}
                         ).questions[0]


class WideScorer:
    """`dists` maps the tokens appended after the 3 prompt tokens (a tuple) to a sparse top-k dump."""

    def __init__(self, dists):
        self.dists, self.calls = dists, []

    def tokenize(self, text, *, add_special=True):
        return [7, 8, 7]

    def next_logprobs(self, tokens, *, top_k, id_slot=None, delimiter=None):
        self.calls.append((tuple(tokens), top_k, id_slot))
        return self.dists[tuple(tokens[3:])]


class FakeTemplate:
    def render(self, msgs):
        return "PROMPT"


def plain(dists, sets=SETS, **kw):
    sc = WideScorer(dists)
    return Engine(sc, FakeTemplate(), sets, **kw), sc


# ---- the plain path --------------------------------------------------------------------------------------------

def test_a_77_option_question_reads_in_one_pass_at_top_k_1000():
    bare, spaced = lab_ids("AB")
    eng, sc = plain({(): {spaced: lp(0.7), bare: lp(0.1), ID["AA"]: lp(0.05), THE: lp(0.15)}})
    r = eng.answer("s", choice(77))
    assert isinstance(r, Readout) and len(r.probs) == 77
    assert max(range(77), key=r.probs.__getitem__) == 1                       # AB is option 1
    assert r.passes == 1 and r.split_passes == 0 and r.unread_mass == 0.0 and r.wide is True
    assert r.option_mass == pytest.approx(0.85) and r.stop == "diverged" and r.depth == 0
    assert r.partial_by_letter == (("A", 0.0), ("B", 0.0), ("C", 0.0), ("D", 0.0))
    assert [c[1] for c in sc.calls] == [WIDE_TOP_K] == [1000]
    assert r.topk_miss == 75 and r.bias is None


def test_a_narrow_question_keeps_top_k_100_and_the_new_fields_at_their_defaults():
    eng, sc = plain({(): {A_: lp(0.6), B_: lp(0.3), THE: lp(0.1)}})
    r = eng.answer("s", choice(3))
    assert [c[1] for c in sc.calls] == [100]
    assert (r.unread_mass, r.split_passes, r.wide, r.partial_by_letter) == (0.0, 0, False, ())


def test_a_split_pass_credits_the_first_letter_mass_and_is_counted():
    eng, sc = plain({(): {GA: lp(0.4), ID["ĠAB"]: lp(0.3), THE: lp(0.3)},
                     (GA,): {B_: lp(0.9), THE: lp(0.1)}})
    r = eng.answer("s", choice(30))
    ab = WIDE_LABELS.index("AB")
    assert math.exp(r.log_marginals[ab]) == pytest.approx(0.3 + 0.4 * 0.9)     # direct + credit 0.36
    assert r.passes == 2 and r.split_passes == 1 and r.unread_mass == 0.0 and r.wide
    assert r.option_mass == pytest.approx(0.66)
    assert [c[0] for c in sc.calls] == [(7, 8, 7), (7, 8, 7, GA)]
    assert all(c[1] == 1000 for c in sc.calls)
    assert r.partial_by_letter == (("A", pytest.approx(0.4)), ("B", 0.0))


def test_the_walk_steps_over_a_shared_prefix_then_splits_at_the_slot():
    eng, sc = plain({(): {STARS: lp(0.95), THE: lp(0.05)},
                     (STARS,): {GA: lp(0.4), ID["ĠAB"]: lp(0.3), THE: lp(0.3)},
                     (STARS, GA): {B_: lp(0.9), THE: lp(0.1)}})
    r = eng.answer("s", choice(30))
    assert r.depth == 1 and r.passes == 3 and r.split_passes == 1
    assert sc.calls[-1][0] == (7, 8, 7, STARS, GA)                           # the split prefix includes `**`


def test_a_dominant_first_letter_stops_the_walk_instead_of_being_appended():
    """The hazard: `ĠA` at 0.9 would be stepped over by a narrow walk; a wide one reads there and splits."""
    eng, sc = plain({(): {GA: lp(0.9), THE: lp(0.1)}, (GA,): {B_: lp(0.5), A_: lp(0.2), THE: lp(0.3)}})
    r = eng.answer("s", choice(30))
    assert r.depth == 0 and r.stop == "diverged" and r.split_passes == 1
    floor = 0.1                                                               # the slot dump's min (Ġthe)
    assert math.exp(r.log_marginals[WIDE_LABELS.index("AB")]) == pytest.approx(floor + 0.45)
    assert math.exp(r.log_marginals[WIDE_LABELS.index("AA")]) == pytest.approx(floor + 0.18)
    assert r.option_mass == pytest.approx(0.45 + 0.18) and r.topk_miss == 30 - 2


def test_the_split_cap_leaves_the_rest_unread():
    dump = {GA: lp(0.3), A_: lp(0.2), GB: lp(0.2), B_: lp(0.1), ID["ĠAB"]: lp(0.15), THE: lp(0.05)}
    cont = {B_: lp(0.5), THE: lp(0.5)}
    eng, sc = plain({(): dump, (GA,): cont, (A_,): cont, (GB,): cont, (B_,): cont})
    r = eng.answer("s", choice(30))
    assert r.split_passes == WIDE_MAX_SPLIT_PASSES == 2 and r.passes == 3
    assert r.unread_mass == pytest.approx(0.3)                                # ĠB and B were never expanded
    assert [c[0][3:] for c in sc.calls] == [(), (GA,), (A_,)]


def test_the_split_settings_are_engine_attributes_for_the_diagnostic():
    dump = {GA: lp(0.3), A_: lp(0.2), GB: lp(0.2), B_: lp(0.1), ID["ĠAB"]: lp(0.15), THE: lp(0.05)}
    cont = {B_: lp(0.5), THE: lp(0.5)}
    eng, sc = plain({(): dump, (GA,): cont, (A_,): cont, (GB,): cont, (B_,): cont})
    assert (eng.split_read_above, eng.split_max_passes, eng.split_min_partial) == (WIDE_SPLIT_READ, 2, 0.0)
    eng.split_read_above, eng.split_max_passes, eng.split_min_partial = 0.0, None, 1e-3
    r = eng.answer("s", choice(30))
    assert r.split_passes == 4 and r.unread_mass == 0.0 and r.passes == 5


# ---- flooring with credit ---------------------------------------------------------------------------------

def test_a_credit_below_the_floor_adds_to_it_and_is_not_a_miss():
    labels = ("AA", "AB", "BA")
    direct = {"AA": lp(0.5), "AB": NEG_INF, "BA": NEG_INF}
    credit = {"AA": NEG_INF, "AB": NEG_INF, "BA": lp(0.0002)}
    dump = {0: lp(0.5), 1: lp(0.3), 2: lp(0.001)}                             # the slot dump's min is log 0.001
    probs, lms, miss = wide_bounded_readout(direct, credit, dump, labels, 1.0)
    assert lms[0] == lp(0.5) and lms[1] == lp(0.001) and lms[2] == pytest.approx(lp(0.0012), abs=1e-12)
    assert miss == 1                                                          # AB only: no direct value, no credit
    assert sum(probs) == pytest.approx(1.0) and probs[2] > probs[1]


def test_without_credit_the_readout_is_bounded_readout_exactly():
    labels = ("AA", "AB", "BA")
    direct = {"AA": lp(0.5), "AB": lp(0.2), "BA": NEG_INF}
    none = dict.fromkeys(labels, NEG_INF)
    dump = {0: lp(0.5), 1: lp(0.2), 2: lp(0.001)}
    for T in (1.0, 0.7, 2.5):
        assert wide_bounded_readout(direct, none, dump, labels, T) == bounded_readout(direct, dump, labels, T)


def test_credit_on_a_label_with_a_direct_value_adds_and_clamps():
    labels = ("AA", "AB")
    probs, lms, miss = wide_bounded_readout({"AA": lp(0.5), "AB": lp(0.2)}, {"AA": lp(0.5), "AB": lp(0.1)},
                                            [lp(0.5), lp(0.2)], labels, 1.0)
    assert lms[0] == 0.0 and lms[1] == pytest.approx(lp(0.3)) and miss == 0


def test_the_floored_credit_reaches_the_served_probabilities():
    """Through the engine: BA has no direct value at the slot, but a split pass credits it."""
    eng, _ = plain({(): {ID["ĠAB"]: lp(0.6), GB: lp(0.02), THE: lp(0.379), ID["AC"]: lp(0.001)},
                    (GB,): {A_: lp(0.01), THE: lp(0.99)}})
    q = choice(30)
    r = eng.answer("s", q)
    ba = WIDE_LABELS.index("BA")
    assert r.split_passes == 1 and r.log_marginals[ba] == pytest.approx(lp(0.001 + 0.02 * 0.01), abs=1e-12)
    assert r.topk_miss == 30 - 3                                              # AB, AC direct; BA credited


# ---- coverage -----------------------------------------------------------------------------------------

def _no_bare_j(toks: list[str]) -> list[str]:
    """The bare `J` respelled `<J>` (ids unchanged): no token continues a label with J, so `AJ` cannot be read in two
    passes either."""
    return [("<J>" if t == "J" else t) for t in toks]


SHORT = SurfaceSets(_no_bare_j(vocab(40, drop_all=6)), "gpt2")              # no `AJ`, no bare `J`: reads 6
MSG = ("q: this model's tokenizer can read 6 of Callo's two-letter labels (the next one needs a letter that is not a "
       "single token); the question has 27 options")


def test_a_short_vocabulary_reads_26_options_and_refuses_27_from_check_request_and_answer():
    assert SHORT.wide_coverage() == SHORT.wide_readable() == 6
    eng, sc = plain({(): {0: lp(0.6), 1: lp(0.3), 54: lp(0.1)}}, sets=SHORT)
    r = eng.answer("s", choice(26))
    assert max(range(26), key=r.probs.__getitem__) == 0 and not r.wide and sc.calls[0][1] == 100
    req = parse_request({"state": "s", "questions": {"q": {"type": "choice",
                                                           "criteria": {f"k{i}": None for i in range(27)}}}})
    with pytest.raises(TooManyOptions) as e:
        eng.check_request(req)
    assert str(e.value) == MSG
    sc.calls.clear()
    with pytest.raises(TooManyOptions) as e:
        eng.answer("s", req.questions[0])
    assert str(e.value) == MSG and sc.calls == []


def test_a_501_for_missing_media_wins_over_the_coverage_422():
    eng, sc = plain({}, sets=SHORT)
    req = parse_request({"state": "s", "images": [URL],
                         "questions": {"q": {"type": "choice", "criteria": {f"k{i}": None for i in range(27)}}}})
    with pytest.raises(NoVision):
        eng.check_request(req)
    assert sc.calls == []


def _think_engine(up, sets):
    post = make_http_post(timeout=10)
    e = Engine(LlamaCppForwardScorer(up.url, http_post=post), TemplateClient(up.url, http_post=post), sets,
               user_delimiter=DELIM, s1_timeout=5)
    e.think, reason = frame_from_template(
        lambda thinking: e.template.render([{"role": "user", "content": "x"}], thinking=thinking), e.scorer.tokenize)
    assert reason is None, reason
    e.n_ctx = up.props["default_generation_settings"]["n_ctx"]
    e.think_sampling = SAMPLING
    return e


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=5).start()
    yield u
    u.stop()


def test_deliberate_and_chunked_refuse_before_rendering_or_generating(up):
    e = _think_engine(up, SHORT)
    up.requests.clear()
    with pytest.raises(TooManyOptions, match="can read 6 of Callo.s two-letter labels"):
        e.answer_deliberate("s", choice(27), [64], id_slot=0, seed=1, sampling=SAMPLING)
    with pytest.raises(TooManyOptions, match="can read 6 of Callo.s two-letter labels"):
        e.answer_chunked("s", choice(27), cap=64, explicit_cap=True, seed=("d", "q"), sampling=SAMPLING,
                         temperature_of=lambda close, k, cap: (1.0, False), id_slot=0, probe_every=32)
    assert up.requests == []                                                 # no render, no tokenize, no generation


# ---- the media path: a split candidate that does not re-tokenize ----------------------------------------------

def _media_engine(sc, sets=SETS):
    return Engine(sc, ChatMLTemplate(), sets, vision=True, media_marker=MARKER, user_delimiter=DELIM)


def test_a_retok_failure_on_the_split_candidate_costs_no_pass_and_stays_unread():
    sc = MMScorer({0: {GA: lp(0.4), ID["ĠAB"]: lp(0.3), THE: lp(0.3)}}, detok={GA: " A"})
    r = _media_engine(sc).answer("s", choice(30), images=[PNG], image_urls=[URL])
    assert r.kind == "image" and r.wide and r.stop == "diverged"
    assert r.passes == 1 and r.split_passes == 0 and r.unread_mass == pytest.approx(0.4)
    assert len(sc.mm) == 1 and sc.detok_calls == [(GA,)]
    assert [m[2] for m in sc.mm] == [1000]


def test_retok_failures_are_capped_at_max_passes_plus_two_attempts():
    ids = [GA, A_, GB, B_, ID["ĠC"], ID["C"]]
    dump = {**{t: lp(0.1) for t in ids}, ID["ĠAB"]: lp(0.3), THE: lp(0.1)}
    sc = MMScorer({0: dump}, detok={t: " " + VOCAB[t][-1] + "?" for t in ids})
    r = _media_engine(sc).answer("s", choice(77), images=[PNG], image_urls=[URL])
    assert len(sc.detok_calls) == WIDE_MAX_SPLIT_PASSES + 2 == 4
    assert r.passes == 1 and r.split_passes == 0 and r.unread_mass == pytest.approx(0.6)


def test_a_wide_retok_stop_in_the_walk_reads_the_last_position_and_expands_nothing():
    """The walk itself stops on retok (a dominant `**` that does not round-trip): the readout is built at the last
    scored position, with its partial mass counted, and no split pass is attempted (dist_fn None)."""
    first = {STARS: lp(0.95), GA: lp(0.03), THE: lp(0.02)}
    sc = MMScorer({0: first}, detok={STARS: "\n"})
    r = _media_engine(sc).answer("s", choice(30), images=[PNG], image_urls=[URL])
    assert r.stop == "retok" and r.depth == 0 and r.wide and r.passes == 1 and r.split_passes == 0
    assert r.unread_mass == pytest.approx(0.03) and r.option_mass == 0.0
    assert r.partial_by_letter == (("A", pytest.approx(0.03)), ("B", 0.0))
    assert sc.detok_calls == [(STARS,)]


# ---- deliberate and chunked readouts ---------------------------------------------------------------------------

def _wide_dump(up):
    """Every readout pass (the slot and both split passes) gets this dump: direct AB 0.3, first letters ĠA 0.4 and
    B 0.2. Split ĠA: continuation B 0.2 -> AB += 0.08; split B: continuation B 0.2 -> BB += 0.04. The thinking fake's
    planted readout parses single-letter option lines, so it is turned off: every readout gets this canned dump."""
    up.bias = None
    up.top_logprobs = [{"id": GA, "token": "ĠA", "logprob": lp(0.4)}, {"id": ID["ĠAB"], "token": "ĠAB", "logprob": lp(0.3)},
                       {"id": B_, "token": "B", "logprob": lp(0.2)}, {"id": THE, "token": "Ġthe", "logprob": lp(0.1)}]


def _reads(up):
    return [r.json for r in up.requests if r.path == "/completion" and r.json.get("n_probs")]


def test_a_deliberate_wide_readout_splits_and_is_marked_deliberate(up):
    _wide_dump(up)
    e = _think_engine(up, SETS)
    up.requests.clear()
    d = e.answer_deliberate("s", choice(77), [64], id_slot=0, seed=1, sampling=SAMPLING)
    r = d.readouts[64].readout
    assert r.bias == "deliberate" and r.wide and r.split_passes == 2 and r.passes == 3 and r.unread_mass == 0.0
    assert math.exp(r.log_marginals[WIDE_LABELS.index("AB")]) == pytest.approx(0.3 + 0.08)
    assert math.exp(r.log_marginals[WIDE_LABELS.index("BB")]) == pytest.approx(0.1 + 0.04)   # floor (Ġthe) + credit
    reads = _reads(up)
    assert len(reads) == 3 and all(x["n_probs"] == 1000 for x in reads)
    assert reads[1]["prompt"] == reads[0]["prompt"] + [GA] and reads[2]["prompt"] == reads[0]["prompt"] + [B_]


def test_chunked_wide_probes_carry_the_split_fields_and_match_the_narrow_probe_count(up):
    up.think_len = 100
    _wide_dump(up)
    e = _think_engine(up, SETS)
    T1 = lambda close, k, cap: (1.0, False)      # noqa: E731
    r = e.answer_chunked("s", choice(77), cap=1024, explicit_cap=False, seed=("d", "q"), sampling=SAMPLING,
                         temperature_of=T1, id_slot=0, probe_every=32)
    assert [p.k for p in r.probes] == [32, 64, 96] == FROZEN["chunked_probe_ks"]
    assert all(p.readout.wide and p.readout.split_passes == 2 and p.readout.unread_mass == 0.0 for p in r.probes)
    assert r.readout.readout.split_passes == 2 and r.readout.readout.passes == 3


def test_narrow_chunked_probe_payloads_are_byte_identical_to_ce89e73():
    """The narrow probes send exactly the bytes they sent before the wide build (frozen on ce89e73, n_probs 100)."""
    u = FakeUpstream(think="preopened", think_len=100).start()
    try:
        e = narrow_engine_for(u)
        u.requests.clear()
        q = REQ.questions[0]
        r = e.answer_chunked(REQ.state, q, cap=1024, explicit_cap=False, seed=(REQ.digest, q.qid),
                             sampling=e.trace_sampling(), temperature_of=lambda close, k, cap: (1.0, False),
                             id_slot=0, probe_every=32)
        reads = [base64.b64encode(unpinned(x.body.decode()).encode()).decode() for x in u.requests
                 if x.path == "/completion" and (x.json or {}).get("n_probs")]
    finally:
        u.stop()
    assert reads == FROZEN["chunked_probe_reads"]
    assert [p.k for p in r.probes] == FROZEN["chunked_probe_ks"]
    assert [{"passes": p.readout.passes, "option_mass": p.readout.option_mass.hex(),
             "probs": [v.hex() for v in p.readout.probs]} for p in r.probes] == FROZEN["chunked_probe_fields"]
    assert all((p.readout.wide, p.readout.split_passes, p.readout.unread_mass, p.readout.partial_by_letter)
               == (False, 0, 0.0, ()) for p in r.probes)


# ---- bias and score --------------------------------------------------------------------------------------------

def test_a_wide_choice_under_a_profile_is_the_wide_category():
    pb = load_profile(profile_dict({3: [0.6, 0.3, 0.1]}))
    eng, _ = plain({(): {ID["ĠAB"]: lp(0.9), THE: lp(0.1)}}, position_bias=pb)
    q = choice(30)
    assert eng.bias_for(q, "plain") == (None, "wide")
    r = eng.answer("s", q)
    assert r.bias == "wide"
    raw = plain({(): {ID["ĠAB"]: lp(0.9), THE: lp(0.1)}})[0].answer("s", q)
    assert r.probs == raw.probs
    score = parse_request({"state": "s", "questions": {"q": {"type": "score",
                                                             "criteria": [f"l{i}" for i in range(40)]}}}).questions[0]
    assert eng.bias_for(score, "plain") == (None, "score")                    # score is decided first


def test_a_40_level_score_reads_its_expected_value_from_the_wide_probs():
    q = parse_request({"state": "s", "questions": {"q": {"type": "score",
                                                         "criteria": [f"level {i}" for i in range(40)]}}}).questions[0]
    a10, a20 = WIDE_LABELS[10], WIDE_LABELS[20]
    eng, _ = plain({(): {ID["Ġ" + a10]: lp(0.6), ID["Ġ" + a20]: lp(0.3), THE: lp(0.1)}})
    r = eng.answer("s", q)
    assert r.wide and len(r.probs) == 40
    body = answer_body(q, r.probs)
    assert body["score"] == pytest.approx(sum(i * p for i, p in enumerate(r.probs)))
    assert 10 < body["score"] < 20 and r.probs[10] > r.probs[20] > r.probs[0]


# ---- coverage refusals before any work, one set fetch per readout ---------------------------------------------

class RecordingTemplate(FakeTemplate):
    def __init__(self):
        self.calls = []

    def render(self, msgs):
        self.calls.append(msgs)
        return "PROMPT"


class RecordingScorer(WideScorer):
    def __init__(self, dists):
        super().__init__(dists)
        self.tokenized = []

    def tokenize(self, text, *, add_special=True, parse_special=None):
        self.tokenized.append(text)
        return [7, 8, 7]


@pytest.mark.parametrize("how", ["plain", "context"])
def test_answer_refuses_past_the_coverage_before_rendering_or_tokenizing(how):
    """The 422 comes before any template or scorer call, on every path of Engine.answer."""
    t, sc = RecordingTemplate(), RecordingScorer({})
    eng = Engine(sc, t, SHORT)
    with pytest.raises(TooManyOptions, match="can read 6 of Callo.s two-letter labels"):
        eng.answer("s", choice(27), context="ctx" if how == "context" else "")
    assert t.calls == [] and sc.tokenized == [] and sc.calls == []


def test_answer_refuses_an_image_request_past_the_coverage_before_any_media_call():
    sc = MMScorer({0: {}})
    eng = _media_engine(sc, SHORT)
    with pytest.raises(TooManyOptions):
        eng.answer("s", choice(27), images=[PNG], image_urls=[URL])
    assert sc.mm == [] and sc.tok_calls == [] and sc.detok_calls == []
    with pytest.raises(NoVision):                                            # the 501 still wins
        Engine(MMScorer({0: {}}), ChatMLTemplate(), SHORT).answer("s", choice(27), images=[PNG], image_urls=[URL])


def test_one_readout_fetches_its_sets_once(monkeypatch):
    """`_walk` hands its sets to `_read_slot` (one coverage check, one `get`) on every path."""
    gets = []
    real = SETS.get

    def spy(labels):
        gets.append(labels)
        return real(labels)
    monkeypatch.setattr(SETS, "get", spy)
    sc = MMScorer({0: {GA: lp(0.4), ID["ĠAB"]: lp(0.3), THE: lp(0.3)}, 1: {B_: lp(0.9), THE: lp(0.1)}})
    _media_engine(sc).answer("s", choice(30), context="ctx")
    assert len(gets) == 1
    plain({(): {ID["ĠAB"]: lp(0.9), THE: lp(0.1)}})[0].answer("s", choice(30))
    assert len(gets) == 2


def test_the_wide_retok_readout_counts_the_marginals_once(monkeypatch):
    """The retok fallback of a wide question computes the slot's marginals once and `_retok_split` returns only
    the SplitResult."""
    import callo.engine as engine_mod
    from callo.marginalize import SplitResult
    calls = []
    real = engine_mod._marginals_and_terms

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(engine_mod, "_marginals_and_terms", spy)
    sc = MMScorer({0: {STARS: lp(0.95), GA: lp(0.03), THE: lp(0.02)}}, detok={STARS: "\n"})
    eng = _media_engine(sc)
    r = eng.answer("s", choice(30), images=[PNG], image_urls=[URL])
    assert r.stop == "retok" and r.unread_mass == pytest.approx(0.03) and len(calls) == 1
    sets = SETS.get(choice(30).labels)
    dump = {STARS: lp(0.95), GA: lp(0.03), THE: lp(0.02)}
    raw, n_terms = real(sets, dump)
    sr = eng._retok_split(choice(30), sets, (), dump, {k: min(v, 0.0) for k, v in raw.items()}, 0.0, n_terms)
    assert isinstance(sr, SplitResult) and sr.unread_mass == pytest.approx(0.03)
