"""callo v0.3 wide-choice: the fixed two-letter
label table, wide questions (27-255 options), the wide reply line, and the <=26 byte-identity pins.

`tests/data/narrow_identity.json` was captured on the PRE-change commit ce89e73 (labels A-Z only): every narrow
`Question` and `user_content` must stay byte-identical to it, and so must `prompt_sha256()`.
"""
import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from callo.errors import CalloError, TooManyOptions
from callo.profile import prompt_sha256
from callo.prompt import WIDE_REPLY, user_content
from callo.proxy import _status
from callo.render import option_text
from callo.systemone import (
    MAX_LETTERS,
    MAX_OPTIONS,
    TOO_MANY_OPTIONS,
    WIDE_LABELS,
    WIDE_LABELS_SHA256,
    InvalidRequest,
    answer_body,
    is_wide,
    parse_request,
)

NARROW = json.loads((Path(__file__).parent / "data" / "narrow_identity.json").read_text(encoding="utf-8"))
# the two-letter English words dropped from the table
DENY = set("AD AH AI AM AN AS AT AX BE BY DO GO HA HE HI IF IN IS IT ME MY NO OF OH OK ON OR OX SO TO UP US WE".split())


def _q(qd: dict, state: str = "s"):
    return parse_request({"state": state, "questions": {"q": qd}}).questions[0]


def _choice(n: int, **kw) -> dict:
    return {"type": "choice", "criteria": {f"k{i}": f"d{i}" for i in range(n)}, **kw}


def test_wide_labels_table_is_pinned():
    assert len(WIDE_LABELS) == 255 == MAX_OPTIONS
    assert len(set(WIDE_LABELS)) == 255
    assert all(re.fullmatch(r"[A-Z]{2}", lab) for lab in WIDE_LABELS)
    assert hashlib.sha256(",".join(WIDE_LABELS).encode()).hexdigest() == WIDE_LABELS_SHA256
    assert WIDE_LABELS_SHA256 == "039130c93c582e36143ed58c31fd68b5de1c18db83c3817ff97ca51ecf3ab50f"
    assert not DENY & set(WIDE_LABELS)
    assert WIDE_LABELS[:4] == ("AA", "AB", "AC", "AE") and WIDE_LABELS[-3:] == ("MP", "MR", "MS")
    assert list(WIDE_LABELS) == sorted(WIDE_LABELS)          # taken from AA..ZZ in order


@pytest.mark.parametrize("n", [27, 255])
def test_wide_choice_27_and_255_labels(n):
    crit = {"a": "first", **{f"k{i}": (f"d{i}" if i % 2 else None) for i in range(1, n)}}
    q = _q({"type": "choice", "criteria": crit})
    assert is_wide(q) and len(q.labels) == n
    assert q.labels == WIDE_LABELS[:n]
    assert q.keys == tuple(crit)
    assert q.labels[0] == "AA"                                 # the letter key "a" does NOT keep its letter
    assert q.texts == tuple(option_text(k, crit[k]) for k in crit)
    assert q.texts[0] == "a: first" and q.texts[1] == "k1: d1" and q.texts[2] == "k2"
    assert q.legend is None


def test_wide_score_levels():
    levels = [f"level {i}" for i in range(40)]
    q = _q({"type": "score", "criteria": levels})
    assert is_wide(q)
    assert q.labels == WIDE_LABELS[:40]
    assert q.keys == tuple(str(i) for i in range(40))
    assert q.texts == tuple(levels) and q.legend == dict(zip(q.keys, levels, strict=True))
    p = [0.0] * 40
    p[3], p[39] = 0.25, 0.75
    body = answer_body(q, p)
    assert body["score"] == 3 * 0.25 + 39 * 0.75 and body["legend"] == q.legend
    assert body["probabilities"] == dict(zip(q.keys, p, strict=True))


def test_is_wide_is_strictly_above_26():
    assert not is_wide(_q(_choice(MAX_LETTERS)))
    assert is_wide(_q(_choice(MAX_LETTERS + 1)))
    assert not is_wide(_q({"type": "noul"}))


def test_256_options_are_still_refused():
    with pytest.raises(InvalidRequest, match=f"1..{MAX_OPTIONS} options"):
        _q(_choice(256))
    with pytest.raises(InvalidRequest, match=f"1..{MAX_OPTIONS} levels"):
        _q({"type": "score", "criteria": ["x"] * 256})


_GOLDEN_30 = ("Input:\nthe input\n\nQuestion: Pick one.\n\nOptions:\n"
              "AA. k0: d0\nAB. k1: d1\nAC. k2: d2\nAE. k3: d3\nAF. k4: d4\nAG. k5: d5\nAJ. k6: d6\nAK. k7: d7\n"
              "AL. k8: d8\nAO. k9: d9\nAP. k10: d10\nAR. k11: d11\nAU. k12: d12\nAV. k13: d13\nAW. k14: d14\n"
              "AZ. k15: d15\nBA. k16: d16\nBB. k17: d17\nBC. k18: d18\nBD. k19: d19\nBF. k20: d20\nBG. k21: d21\n"
              "BH. k22: d22\nBI. k23: d23\nBJ. k24: d24\nBK. k25: d25\nBL. k26: d26\nBM. k27: d27\nBN. k28: d28\n"
              "BO. k29: d29\n\n"
              "Reply with one two-letter label: AA, AB, AC, AE, AF, AG, AJ, AK, AL, AO, AP, AR, AU, AV, AW, AZ, BA, BB, "
              "BC, BD, BF, BG, BH, BI, BJ, BK, BL, BM, BN, BO.")


def test_wide_prompt_golden():
    q = _q(_choice(30, instructions="Pick one."), state="the input")
    assert user_content("the input", q) == _GOLDEN_30
    assert WIDE_REPLY == "Reply with one two-letter label: {labels}."


@pytest.mark.parametrize("name", sorted(NARROW["cases"]))
def test_narrow_byte_identity(name):
    case = NARROW["cases"][name]
    state = case.get("state", "the input")              # the later cases record their own (empty included)
    q = _q(case["question"], state=state)
    assert dataclasses.asdict(q) == {k: (tuple(v) if isinstance(v, list) else v) for k, v in case["parsed"].items()}
    assert q.keys == tuple(case["parsed"]["keys"]) and q.labels == tuple(case["parsed"]["labels"])
    assert not is_wide(q)
    assert user_content(state, q) == case["user_content"]


def test_narrow_identity_fixture_covers_the_contract():
    """n in {1, 2, 3, 10, 26} for named-key and letter-key choices and scores; plus noul with and
    without criteria, mixed letter and non-letter keys, special-token escaping, and an empty state with no
    instructions. All captured on ce89e73."""
    assert {int(n.rsplit("_", 1)[1]) for n in NARROW["cases"]} == {1, 2, 3, 4, 10, 26}
    assert {n.rsplit("_", 1)[0] for n in NARROW["cases"]} == {
        "choice_named", "choice_letters", "score", "noul", "noul_criteria", "choice_mixed", "choice_special",
        "choice_bare"}
    mixed = NARROW["cases"]["choice_mixed_4"]
    assert mixed["parsed"]["keys"] == ["b", "x", "go", "a"] and mixed["parsed"]["labels"] == ["B", "X", "C", "A"]
    assert "<¦im_start¦>" in NARROW["cases"]["choice_special_3"]["user_content"]
    bare = NARROW["cases"]["choice_bare_3"]
    assert bare["state"] == "" and bare["parsed"]["instructions"] == "" and bare["user_content"].startswith("Options:")


def test_prompt_sha256_is_unchanged():
    assert prompt_sha256() == NARROW["prompt_sha256"] == (
        "8b718d40420bda901afbd7e74a8e4aa2ee24b7415a18e23e248208f5f3d79f8f")


def test_too_many_options_is_a_422_callo_error():
    e = TooManyOptions(TOO_MANY_OPTIONS.format(qid="q", n=9, m=27))
    assert isinstance(e, CalloError) and _status(e) == 422
    assert str(e) == ("q: this model's tokenizer can read 9 of Callo's two-letter labels (the next one needs a letter "
                      "that is not a single token); the question has 27 options")
