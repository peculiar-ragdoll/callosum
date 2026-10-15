"""callo.bias, the question side: canonical questions, rotations, exclusions, the set scan (with
--max-per-n)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from helpers import bench_path

from callo.bias import (
    PRIOR_FLOOR,
    Item,
    NFit,
    SetError,
    builtin_set_sha256,
    cap_per_n,
    debias,
    evaluate,
    exclusion,
    fit_n,
    is_canonical,
    is_cross_reference,
    offsets_from_prior,
    prior_from_rows,
    question_row,
    read_set,
    rotated_label,
    rotation_criteria,
    skip_reason,
)
from callo.calibrate import request_from_record
from callo.engine import bounded_readout
from callo.prompt import user_content
from callo.systemone import parse_request

TRANSFER_DEV = Path(bench_path("CALLO_BENCH_DATA", "v4", "transfer-v4", "development.jsonl"))   # Kev's evals/
PNG_URL = "data:image/png;base64,iVBORw0KGgo="


def _q(criteria, qtype="choice"):
    return parse_request({"state": "s", "questions": {"q": {"type": qtype, "criteria": criteria}}}).questions[0]


@pytest.mark.parametrize(("criteria", "canonical"), [
    ({"a": None, "b": None, "c": None}, True),
    ({"a": "one", "go": "two", "c": "three"}, True),           # a, go, c -> A, B, C
    ({"left": None, "right": None, "up": None}, True),         # no single-letter key
    ({"1": "x1", "2": "x2"}, True),                            # digits take the letters in order
    ({"b": None, "x": None, "go": None, "a": None}, False),    # -> B, X, C, A
    ({"c": None, "a": None, "b": None}, False),
])
def test_canonical_on_every_key_style(criteria, canonical):
    assert is_canonical(_q(criteria)) is canonical


def test_every_noul_is_canonical():
    assert is_canonical(_q(None, "noul")) and is_canonical(_q({"true": "t", "false": "f"}, "noul"))


@pytest.mark.parametrize("text", ["All of the above", "none of these", "Both of the above are true",
                                  "Neither of those", "option B", "Answers C", "A and B", "C or D", "A & E",
                                  "Both A and C", "neither B"])
def test_cross_reference_patterns_match(text):
    assert is_cross_reference(text)


@pytest.mark.parametrize("text", ["above the boiling point", "below the poverty line", "a and b",
                                  "Both are wrong here, see B", "Vitamin A", "Plan B", "the answer is clear"])
def test_cross_reference_negatives_are_content(text):
    assert not is_cross_reference(text)


def test_exclusion_drops_the_whole_question_bare_letter_first():
    assert exclusion(["B", "All of the above"]) == "bare_letter"
    assert exclusion(["red", "All of the above"]) == "cross_reference"   # a distractor is enough
    assert exclusion(["red", "green"]) is None


def test_skip_reasons_in_order():
    assert skip_reason(parse_request({"state": "s", "questions": {"q": {"type": "score", "criteria": ["lo", "hi"]}}})
                       .questions[0]) == "score"
    assert skip_reason(_q({"only": None})) == "single_option"
    assert skip_reason(_q({"b": None, "x": None, "go": None, "a": None})) == "noncanonical"
    assert skip_reason(_q({"a": None, "b": "two"})) == "bare_letter"        # key a with no description renders "a"
    assert skip_reason(_q({"a": "red", "b": "none of the above"})) == "cross_reference"
    assert skip_reason(_q({"a": "red", "b": "blue"})) is None
    assert skip_reason(_q(None, "noul")) is None


def test_rotation_moves_option_k_plus_s_to_position_k_and_remaps_the_label():
    texts = ("red", "green", "blue", "cyan")
    assert rotation_criteria(texts, 1) == {"a": "green", "b": "blue", "c": "cyan", "d": "red"}
    for s in range(4):
        q = _q(rotation_criteria(texts, s))
        assert q.texts == tuple(texts[(k + s) % 4] for k in range(4)) and is_canonical(q)
        assert q.texts[rotated_label(2, s, 4)] == "blue"


RECORDS = [
    {"state": {"subject": "anatomy", "question": "Which bone?"},
     "questions": {"answer": {"type": "choice", "instructions": "Which option correctly answers the question?",
                              "criteria": {"a": "femur", "b": "tibia", "c": "ulna"}, "label": "b", "src": "mmlu"}},
     "_meta": {"id": "r1", "source": "mmlu", "group_id": "g1"}},
    {"state": "I am still waiting on my card?",
     "questions": {"intent": {"type": "choice", "instructions": "Which intent does this customer message express?",
                              "criteria": {"card arrival": None, "card linking": None, "top up failed": None},
                              "label": "card arrival"}},
     "_meta": {"id": "r2", "source": "banking77", "pair_id": "p2"}},
    {"state": [{"role": "user", "content": "hello there"}],
     "questions": {"o": {"type": "noul", "instructions": "Is it a greeting?",
                         "criteria": {"true": "it greets", "false": "it does not"}, "label": True}}},
    {"state": "s", "questions": {"m": {"type": "choice", "criteria": {"a": "x1", "go": "x2", "c": "x3"}}}},
    {"state": "s", "context": "standing orders", "questions": {"d": {"type": "choice",
                                                                     "criteria": {"1": "one", "2": None}}}},
]


def _contract(records):
    """The rotation-0 contract: rotation 0's user turn is byte-identical to the original question's, for every eligible item."""
    scan = read_set("\n".join(json.dumps(r) for r in records))
    for it in scan.items:
        rec = records[int(it.uid[1:].split("/")[0]) - 1]
        orig = parse_request(request_from_record(rec))
        q = next(x for x in orig.questions if x.qid == it.qid)
        rot0 = parse_request(it.body(0))
        assert (rot0.state, rot0.context, rot0.images) == (orig.state, orig.context, orig.images)
        assert user_content(rot0.state, rot0.questions[0]) == user_content(orig.state, q), it.uid
        for s in range(it.n):
            assert parse_request(it.body(s)).questions[0].texts == tuple(q.texts[(k + s) % it.n] for k in range(it.n))
    return scan


def test_rotation_zero_renders_the_original_user_turn_byte_for_byte():
    scan = _contract(RECORDS)
    assert len(scan.items) == 5 and not scan.skipped


def test_rotation_zero_contract_on_transfer_v4_dev():
    if not TRANSFER_DEV.is_file():
        pytest.skip(f"{TRANSFER_DEV} is missing (CALLO_BENCH_DATA unset, or not this box)")
    records = [json.loads(line) for line in TRANSFER_DEV.read_text().splitlines() if line.strip()]
    assert _contract(records).items


def test_the_scan_keeps_clusters_sources_kinds_and_labels():
    scan = read_set("\n".join(json.dumps(r) for r in RECORDS))
    by = {it.qid: it for it in scan.items}
    assert by["answer"].cluster == "g1" and by["intent"].cluster == "p2" and by["o"].cluster == "L3/o"
    assert by["answer"].source == "mmlu" and by["intent"].source == "banking77" and by["m"].source == "unknown"
    assert by["answer"].label == 1 and by["intent"].label == 0 and by["o"].label == 1 and by["m"].label is None
    assert by["d"].kind == "context" and by["answer"].kind == "plain" and by["answer"].record_id == "r1"
    assert by["m"].record_id == "line 4" and by["intent"].n == 3 and by["o"].texts == ("no: it does not",
                                                                                       "yes: it greets")
    img = read_set(json.dumps({"state": "s", "images": [PNG_URL], "questions": {"q": {"type": "noul"}}}))
    assert img.items[0].kind == "image" and img.items[0].body(1)["images"] == [PNG_URL]


def test_every_top_level_key_but_questions_label_src_and_meta_is_forwarded():
    """v0.3 media adds top-level keys (audio, videos): a rotation's request must carry them unchanged."""
    rec = {"state": "s", "model": "m", "zz_extension": {"a": [1, 2]}, "label": "x", "src": "y", "_meta": {"id": "r"},
           "questions": {"q": {"type": "choice", "criteria": {"red": None, "blue": None}, "label": "red", "src": "z"}}}
    (it,) = read_set(json.dumps(rec)).items
    body = it.body(1)
    assert body["zz_extension"] == {"a": [1, 2]} and body["model"] == "m" and body["state"] == "s"
    assert not {"label", "src", "_meta"} & set(body)
    assert body["questions"] == {"q": {"type": "choice", "instructions": "", "criteria": {"a": "blue", "b": "red"}}}
    no_model = {k: v for k, v in rec.items() if k != "model"}
    assert it.label == 0 and "model" not in read_set(json.dumps(no_model)).items[0].body(0)   # nothing is invented


def test_the_scan_counts_every_skip_reason():
    wide = {f"k{i}": None for i in range(27)}
    lines = [
        {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"x1": None, "x2": None}}},
         "_meta": {"variant": "permuted"}},
        {"state": "s", "questions": {"w": {"type": "choice", "criteria": wide},
                                     "ok": {"type": "choice", "criteria": {"red": None, "blue": None}}}},
        {"state": "s", "questions": {"s": {"type": "score", "criteria": ["lo", "hi"]},
                                     "one": {"type": "choice", "criteria": {"only": None}},
                                     "nc": {"type": "choice", "criteria": {"b": None, "a": None}},
                                     "bl": {"type": "choice", "criteria": {"a": None, "b": "two"}},
                                     "xr": {"type": "choice", "criteria": {"a": "red", "b": "All of the above"}}}},
    ]
    scan = read_set("\n".join(json.dumps(r) for r in lines) + "\n\n")
    assert [it.qid for it in scan.items] == ["ok"]
    assert dict(scan.skipped) == {"variant": 1, "too_many_options": 1, "score": 1, "single_option": 1,
                                  "noncanonical": 1, "bare_letter": 1, "cross_reference": 1}


def test_an_empty_option_text_is_a_bare_letter():
    """Rotated, an empty description renders as its letter key, breaking the rotation-0 contract."""
    q = _q({"": None, "red": None})
    assert q.texts == ("", "red") and is_canonical(q)
    assert _q(rotation_criteria(q.texts, 1)).texts == ("red", "b")
    assert skip_reason(q) == "bare_letter"
    assert exclusion(["3", "4", "9", "6"]) is None                     # a one-character digit rotates verbatim


def _scan_one(question):
    return read_set(json.dumps({"state": "s", "questions": {"q": question}}))


def test_too_many_options_counts_only_wide_choice_and_score_criteria():
    wide = 27
    choice = _scan_one({"type": "choice", "criteria": {f"k{i}": None for i in range(wide)}})
    score = _scan_one({"type": "score", "criteria": [f"l{i}" for i in range(wide)]})
    assert dict(choice.skipped) == dict(score.skipped) == {"too_many_options": 1}
    noul = _scan_one({"type": "noul", "criteria": {f"k{i}": None for i in range(wide)}})   # extra keys are ignored
    assert "too_many_options" not in noul.skipped and [it.texts for it in noul.items] == [("no", "yes")]


@pytest.mark.parametrize("question", [
    {"type": "choice", "criteria": [f"l{i}" for i in range(27)]},
    {"type": "score", "criteria": {f"k{i}": None for i in range(27)}},
])
def test_wide_criteria_of_the_wrong_shape_are_a_set_error_not_too_many_options(question):
    with pytest.raises(SetError, match="line 1: not a System One request"):
        _scan_one(question)


@pytest.mark.parametrize(("line", "match"), [
    ("{not json", "line 2: not valid JSON"),
    ("[1, 2]", "line 2: not a JSON object"),
    ('{"questions": {"q": {"type": "noul"}}}', "line 2: not a System One request"),
    ('{"state": "s", "questions": {"q": {"type": "maybe"}}}', "line 2: not a System One request"),
])
def test_a_malformed_line_is_a_set_error_with_its_line_number(line, match):
    with pytest.raises(SetError, match=match):
        read_set(json.dumps(RECORDS[3]) + "\n" + line)


def test_cap_per_n_round_robins_over_sources_seeded_and_keeps_file_order():
    recs = [{"state": f"s{i}", "questions": {"q": {"type": "choice", "criteria": {"red": None, "blue": None}}},
             "_meta": {"source": src}} for i, src in enumerate(["a"] * 6 + ["b"] * 6 + ["c"] * 6)]
    items = read_set("\n".join(json.dumps(r) for r in recs)).items
    kept, capped = cap_per_n(items, 6, seed=0)
    assert capped == 12 and len(kept) == 6
    assert sorted(it.source for it in kept) == ["a", "a", "b", "b", "c", "c"]
    assert kept == sorted(kept, key=lambda it: int(it.uid[1:].split("/")[0]))     # file order
    assert cap_per_n(items, 6, seed=0) == (kept, 12) and cap_per_n(items, 6, seed=1)[0] != kept
    assert cap_per_n(items, 100, seed=0) == (items, 0)


# ---- the probability-space estimator, debias, evaluation --------------------------------------------------

PRIORS = {2: [0.6, 0.4], 3: [0.5, 0.3, 0.2], 4: [0.4, 0.3, 0.2, 0.1]}


def _items(n, count, *, prefix="q", labelled=True):
    return [Item(uid=f"{prefix}{n}-{i}", record_id="r", qid="q", cluster=f"{prefix}{n}-{i}", source="syn",
                 kind="plain", n=n, texts=tuple(f"opt{k}" for k in range(n)), instructions="",
                 label=0 if labelled else None, base={"state": ""}) for i in range(count)]


def _planted(items, priors, seed=0, content_sd=1.0):
    """p(q,s,k) ∝ prior[k] · exp(c_q[(k+s) mod n]): PriDe's factorisation, noise-free. Content 0 is the gold."""
    rng = np.random.default_rng(seed)
    readings, contents = {}, {}
    for it in items:
        c = rng.normal(0, content_sd, it.n)
        c[0] = c.max() + 1.0                                       # the gold (label 0) wins on content
        contents[it.uid] = c
        for s in range(it.n):
            z = np.array([c[(k + s) % it.n] for k in range(it.n)]) + np.log(priors[it.n])
            p = np.exp(z - z.max())
            readings[(it.uid, s)] = (tuple(p / p.sum()), 0)
    return readings, contents


def test_a_question_row_is_its_mean_served_probability_per_position():
    probs = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]]
    assert np.allclose(question_row(probs), [1 / 3, 1 / 3, 1 / 3], atol=1e-15)
    assert np.allclose(question_row([[0.6, 0.4], [0.5, 0.5]]), [0.55, 0.45], atol=1e-15)


def test_with_no_position_effect_every_row_is_exactly_uniform():
    """The null: every position sees every option once, so a position-free question's row is uniform."""
    items = _items(4, 20)
    readings, _ = _planted(items, {4: [0.25] * 4})
    fit = fit_n(items, readings)[4]
    assert np.allclose(fit.rows, 0.25, atol=1e-12) and np.allclose(fit.prior, 0.25, atol=1e-12)


def test_the_prior_is_the_mean_served_probability_per_position_and_points_the_planted_way():
    """prior_n is the mean over every readout of each n (every fitted question has exactly n rotations). Under
    PriDe's planted factorisation it is a shrunk version of the planted prior: same order, closer to uniform."""
    items = _items(2, 30) + _items(3, 30) + _items(4, 30)
    readings, _ = _planted(items, PRIORS)
    fits = fit_n(items, readings)
    for n, planted in PRIORS.items():
        every = [readings[(it.uid, s)][0] for it in items if it.n == n for s in range(n)]
        assert np.allclose(fits[n].prior, np.mean(every, axis=0), atol=1e-12, rtol=0), n
        assert list(np.argsort(fits[n].prior)) == list(np.argsort(planted)), n
        assert np.abs(np.array(fits[n].prior) - 1 / n).max() < np.abs(np.array(planted) - 1 / n).max(), n
        assert fits[n].questions == 30 and fits[n].floored == 0 and fits[n].rows.shape == (30, n)
        assert fits[n].clusters == tuple(it.cluster for it in items if it.n == n)


def _floor(readings, uid, n, rng):
    """Censor one question like callo's top-k bound: in each rotation the two least likely positions read a floor that
    varies with the rotation (the 100th token's logprob), and the readout is renormalised."""
    for s in range(n):
        p = np.array(readings[(uid, s)][0])
        low = np.argsort(p)[:2]
        lp = np.log(p)
        lp[low] = lp[low].max() + rng.uniform(-1.0, 1.0)
        q = np.exp(lp - lp.max())
        readings[(uid, s)] = (tuple(q / q.sum()), 2)


def test_floored_questions_are_fitted_and_counted_and_incomplete_ones_skipped():
    items = _items(3, 10)
    readings, _ = _planted(items, PRIORS)
    _floor(readings, items[0].uid, 3, np.random.default_rng(2))
    del readings[(items[1].uid, 2)]                                               # one rotation missing
    fit = fit_n(items, readings)[3]
    assert fit.questions == 9 and fit.floored == 1                                # the floored question is fitted
    every = [readings[(it.uid, s)][0] for it in [items[0], *items[2:]] for s in range(3)]
    assert np.allclose(fit.prior, np.mean(every, axis=0), atol=1e-12)
    assert fit_n(items[:1], {}) == {}                                             # nothing read: n omitted


def test_the_prior_is_floored_and_renormalised():
    rows = np.array([[0.7, 0.3, 0.0], [0.5, 0.5, 0.0]])
    prior = prior_from_rows(rows)
    assert prior[2] >= PRIOR_FLOOR * 0.99 and math.isclose(math.fsum(prior), 1.0, abs_tol=1e-12)
    assert all(math.isfinite(o) for o in offsets_from_prior(prior))
    assert prior_from_rows(np.array([[0.6, 0.4]])) == pytest.approx((0.6, 0.4), abs=1e-15)


def test_offsets_are_the_centred_log_prior_and_debias_is_serving_with_them_at_t1():
    off = offsets_from_prior([0.5, 0.3, 0.2])
    assert abs(sum(off)) < 1e-12 and math.isclose(off[0] - off[2], math.log(0.5 / 0.2), rel_tol=1e-12)
    lms = [-0.3, -1.2, -2.0]                                                       # raw log-marginals
    served = np.exp(lms) / np.exp(lms).sum()
    z = np.array(lms) - np.array(off)
    want = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    assert np.allclose(debias(served, off), want, atol=1e-12)
    assert debias([0.25, 0.75], None) == [0.25, 0.75]


def test_evaluation_consistency_meanp_and_labelled_metrics():
    items = _items(4, 40)
    readings, _ = _planted(items, {4: [0.7, 0.1, 0.1, 0.1]}, content_sd=0.2)
    offs = {4: offsets_from_prior([0.7, 0.1, 0.1, 0.1])}
    ev = evaluate(items, readings, offs)
    assert ev["debiased"]["consistency"]["4"] == 1.0 and ev["raw"]["consistency"]["4"] < 1.0
    assert ev["raw"]["meanP"]["4"][0] > 0.4 and max(abs(x - 0.25) for x in ev["debiased"]["meanP"]["4"]) < 0.2
    assert ev["debiased"]["accuracy"] == 1.0 and ev["raw"]["accuracy"] < 1.0 and ev["raw"]["rows"] == 160
    assert ev["debiased"]["nll"] < ev["raw"]["nll"] and set(ev["raw"]) >= {"ece10", "nll", "accuracy"}
    unlabelled = evaluate(_items(4, 5, labelled=False), _planted(_items(4, 5, labelled=False), PRIORS)[0], {})
    assert "accuracy" not in unlabelled["raw"] and "rows" not in unlabelled["debiased"]
    assert set(unlabelled["raw"]["consistency"]) == {"4"}


def test_the_builtin_sets_sha256_is_the_pinned_one():
    """`set.builtin` is keyed on the set's sha256, so a copy of the built-in set under any name is built-in. The
    literal is bench/data/bias_set_manifest.json's: a changed byte (a CRLF checkout, see .gitattributes) fails here."""
    assert builtin_set_sha256() == "f73ff674470f75993076e4e00cc306c88fa1f28bde1e85b49dcf03738a9811a1"


def test_nfit_compares_and_hashes_by_identity_despite_its_ndarray_field():
    """`rows` is an ndarray, so a field-wise `==` would raise "truth value of an array is ambiguous" and `hash` a
    TypeError: NFit uses identity equality instead."""
    def make():
        return NFit(n=2, rows=np.full((2, 2), 0.5), clusters=("a", "b"), prior=(0.5, 0.5), questions=2, floored=0)
    a, b = make(), make()
    assert a == a and a != b and not (a == b)
    assert len({a, b, a}) == 2 and hash(a) == hash(a)


def test_an_audio_or_mixed_record_scans_as_its_request_kind_and_forwards_its_media():
    """v0.3: Item.kind is Request.kind, so a set of audio records yields an `audio` profile kind."""
    wav = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IA=="
    audio = read_set(json.dumps({"state": "s", "audio": [wav], "questions": {"q": {"type": "noul"}}}))
    assert audio.items[0].kind == "audio" and audio.items[0].body(0)["audio"] == [wav]
    mixed = read_set(json.dumps({"state": "s", "audio": [wav], "images": [PNG_URL],
                                 "questions": {"q": {"type": "noul"}}}))
    assert mixed.items[0].kind == "mixed"


@pytest.mark.parametrize("seed", range(5))
def test_a_floored_option_moves_a_row_by_at_most_its_own_served_probability(seed):
    """fit_n's docstring: a floored option is served at the top-k floor, an UPPER bound of its true marginal,
    so serving it there instead of at its true value (anything below the floor, -inf included) moves each element of
    the question's row by at most the floored option's own served probability, averaged over the rotations."""
    rng = np.random.default_rng(seed)
    n, labels = 4, "ABCD"
    floor = -8.0
    content = rng.normal(-1.0, 1.0, n)
    content[3] = -np.inf                                            # option 3 is outside the top-k in every rotation
    true_lm = content.copy()
    true_lm[3] = floor - rng.uniform(0.0, 20.0)                     # its true marginal: somewhere below the floor
    served, true, own = [], [], []
    for s in range(n):
        order = [(k + s) % n for k in range(n)]                     # position k holds option (k + s) mod n
        top = {i: floor for i in range(100)}                        # the 100th-ranked token sets the floor
        p, _, miss = bounded_readout({labels[k]: float(content[o]) for k, o in enumerate(order)}, top, labels)
        q, _, _ = bounded_readout({labels[k]: float(true_lm[o]) for k, o in enumerate(order)}, top, labels)
        assert miss == 1
        served.append(p)
        true.append(q)
        own.append(p[order.index(3)])
    moved = np.abs(question_row(served) - question_row(true))
    assert moved.max() <= np.mean(own) + 1e-15 and np.mean(own) < 0.01


def test_read_set_splits_lines_on_newline_only():
    """A JSON string may hold a raw U+2028, U+2029 or U+0085, which
    str.splitlines() would cut the record at."""
    rec = {**RECORDS[0], "state": "line one\u2028line two\u2029three\x85four"}
    scan = read_set(json.dumps(rec, ensure_ascii=False) + "\n" + json.dumps(RECORDS[1]))
    assert [it.uid for it in scan.items] == ["L1/answer", "L2/intent"]
