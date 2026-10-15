"""Verbatim port of Kev's request rendering (jaredpalmer/kev `kev/api.py` @ 6605a5a5).

Ported, not reimplemented, because the comparison is only fair if our arm sees exactly the text Kev's
model sees: the same `key: value` flattening of a JSON state, the same `name: description` options.
Any change here must be a change upstream first. The confidence formulas are Kev's too (they mirror
TypeSafe's system-one-adapter 0.2.1), so the `confidence` field means what Kev's means; note Clef's
`confidence` is plain p_max, so that field is not comparable across arms (the runner scores p_max).
"""
from __future__ import annotations

from typing import Any


def render(v: Any, indent: int = 0) -> str:
    """Flatten str | object | array into text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list))
                     else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name: str, desc: Any) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria: Any) -> list[str]:
    """Keys a question's probabilities are reported under, in option order."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def _normalize(p: list[float]) -> list[float]:
    t = sum(p)
    return [1 / len(p)] * len(p) if t == 0 else [x / t for x in p]


def choice_confidence(p: list[float]) -> float:
    """(p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty."""
    k = len(p)
    return 1.0 if k == 1 else (max(_normalize(p)) - 1 / k) / (1 - 1 / k)


def score_confidence(p: list[float]) -> float:
    """max(0, 1 - E|level - mode| / D), D = mean |i - (L-1)/2| over the L levels."""
    n = len(p)
    if n == 1:
        return 1.0
    p = _normalize(p)
    mode = max(range(n), key=p.__getitem__)
    d = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / d)
