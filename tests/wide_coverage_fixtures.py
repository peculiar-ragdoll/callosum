"""Wide readouts WITHIN a vocabulary's coverage, as data, for the 0.3.1 byte-identity pin: a question of at most
`wide_coverage` options reads exactly as it did before labels past the coverage were read in two passes.
`tests/data/wide_within_coverage.json` holds what `snapshot()` returned on 0.3.0 (5532add), captured by running this
module against that commit's `src`; `test_wide_uncovered.py` compares the current code with it.

The vocabularies are gpt2-style: the 26 letters bare and spaced, `A.`, `**`, `Ġthe`, then `WIDE_LABELS[:n]` bare and
spaced. `short` covers 40 labels and carries no token for the other 215 (0.3.1 reads those in two passes, so its
table is built over all 255 with empty sets); `full` covers all 255."""
from __future__ import annotations

import math
import string

LETTERS = list(string.ascii_uppercase)


def vocab(wide_labels, n_labels: int) -> list[str]:
    toks = LETTERS + ["Ġ" + c for c in LETTERS] + ["A.", "**", "Ġthe"]
    for lab in wide_labels[:n_labels]:
        toks += [lab, "Ġ" + lab]
    return toks


def lp(p: float) -> float:
    return math.log(p)


def scenarios(wide_labels) -> dict:
    """name -> (vocabulary size in labels, question options, {appended tokens: sparse dump}, engine attributes)."""
    ids = {t: i for i, t in enumerate(vocab(wide_labels, 255))}     # a label's ids are the same in both vocabularies
    GA, GB, A_, B_, STARS, THE = ids["ĠA"], ids["ĠB"], ids["A"], ids["B"], ids["**"], ids["Ġthe"]
    cap_dump = {GA: lp(0.3), A_: lp(0.2), GB: lp(0.2), B_: lp(0.1), ids["ĠAB"]: lp(0.15), THE: lp(0.05)}
    cont = {B_: lp(0.5), THE: lp(0.5)}
    out = {}
    for name, size, n in (("short30", 40, 30), ("short40", 40, 40), ("full77", 255, 77), ("full255", 255, 255)):
        out[f"{name}_direct"] = (size, n, {(): {ids["ĠAB"]: lp(0.7), ids["AB"]: lp(0.1), ids["AA"]: lp(0.05),
                                                THE: lp(0.15)}}, {})
        out[f"{name}_split"] = (size, n, {(): {GA: lp(0.4), ids["ĠAB"]: lp(0.3), THE: lp(0.3)},
                                          (GA,): {B_: lp(0.9), THE: lp(0.1)}}, {})
        out[f"{name}_cap"] = (size, n, {(): cap_dump, (GA,): cont, (A_,): cont, (GB,): cont, (B_,): cont}, {})
        out[f"{name}_uncapped"] = (size, n, {(): cap_dump, (GA,): cont, (A_,): cont, (GB,): cont, (B_,): cont},
                                   {"split_read_above": 0.0, "split_max_passes": None, "split_min_partial": 1e-3})
        out[f"{name}_walk"] = (size, n, {(): {STARS: lp(0.95), THE: lp(0.05)},
                                         (STARS,): {GA: lp(0.4), ids["ĠAB"]: lp(0.3), THE: lp(0.3)},
                                         (STARS, GA): {B_: lp(0.9), THE: lp(0.1)}}, {})
        out[f"{name}_floor_credit"] = (size, n, {(): {ids["ĠAB"]: lp(0.6), GB: lp(0.02), THE: lp(0.379),
                                                      ids["AC"]: lp(0.001)},
                                                 (GB,): {A_: lp(0.01), THE: lp(0.99)}}, {})
    out["short_narrow26"] = (40, 26, {(): {A_: lp(0.6), B_: lp(0.3), THE: lp(0.1)}}, {})
    return out


class _Scorer:
    def __init__(self, dists):
        self.dists, self.calls = dists, []

    def tokenize(self, text, *, add_special=True):
        return [7, 8, 7]

    def next_logprobs(self, tokens, *, top_k, id_slot=None, delimiter=None):
        self.calls.append((list(tokens[3:]), top_k))
        return self.dists[tuple(tokens[3:])]


class _Template:
    def render(self, msgs):
        return "PROMPT"


def snapshot(Engine, SurfaceSets, parse_request, wide_labels) -> dict:
    """Every scenario's Readout fields (floats as `float.hex`, bit-exact) and the passes it posted."""
    sets = {size: SurfaceSets(vocab(wide_labels, size), "gpt2") for size in (40, 255)}
    out = {}
    for name, (size, n, dists, attrs) in scenarios(wide_labels).items():
        sc = _Scorer(dists)
        eng = Engine(sc, _Template(), sets[size])
        for k, v in attrs.items():
            setattr(eng, k, v)
        q = parse_request({"state": "s", "questions": {"q": {"type": "choice",
                                                             "criteria": {f"k{i}": f"d{i}" for i in range(n)}}}}
                          ).questions[0]
        r = eng.answer("s", q)
        out[name] = {"probs": [v.hex() for v in r.probs], "log_marginals": [v.hex() for v in r.log_marginals],
                     "option_mass": r.option_mass.hex(), "stop": r.stop, "depth": r.depth, "topk_miss": r.topk_miss,
                     "passes": r.passes, "split_passes": r.split_passes, "unread_mass": r.unread_mass.hex(),
                     "wide": r.wide, "partial_by_letter": [[f, m.hex()] for f, m in r.partial_by_letter],
                     "calls": sc.calls}
    return out
