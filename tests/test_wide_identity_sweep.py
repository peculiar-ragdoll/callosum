"""callo v0.3 wide-choice: the <=26-option byte-identity sweep. `tests/narrow_sweep.py` runs the
narrow readout paths (plain, context, image, deliberate; choice, score, noul; n in {2, 3, 10, 26}), the walk fixtures
and the profile guards; `tests/data/narrow_sweep.json` is that run on the commit before the wide-choice change, in
the development history. Every `/completion` body, `Readout` field, X-Callo-Timings entry, answer body, walk result
and warning line must match.

Floats are compared bit-exact (`float.hex`) on the platform the fixture was captured on, under Python 3.12 or later
(3.12 made `sum()` of floats compensated, which moves the answer body's `confidence` and `score` by a few ulps on
3.10 and 3.11). Elsewhere (CI's Linux libm may also round exp/log differently in the last place) a float may differ
by FLOAT_REL relative plus FLOAT_ABS absolute, far below any change a readout bug makes, while every body, header,
string and integer stays exact."""
import functools
import json
import math
import platform
import sys

import narrow_sweep
import pytest

FROZEN = json.loads(narrow_sweep.FIXTURE.read_text(encoding="utf-8"))
EXACT = (FROZEN["_meta"]["platform"] == f"{platform.system()}-{platform.machine()}"
         and sys.version_info >= (3, 12) and tuple(map(int, FROZEN["_meta"]["python"].split(".")[:2])) >= (3, 12))
FLOAT_REL, FLOAT_ABS = 1e-12, 1e-15


@functools.cache
def _now(section: str) -> dict:
    return getattr(narrow_sweep, f"{section}_leg")()


def _float(s):
    if isinstance(s, str) and (s.startswith(("0x", "-0x")) or s in ("inf", "-inf", "nan")):
        try:
            return float.fromhex(s)
        except ValueError:
            return None
    return None


def _diffs(a, b, path="", exact=EXACT) -> list[str]:
    """Where `b` (now) differs from `a` (frozen); hex floats within FLOAT_REL / FLOAT_ABS when not `exact`."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(a)} != {sorted(b)}"]
        return [d for k in a for d in _diffs(a[k], b[k], f"{path}.{k}", exact)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in _diffs(x, y, f"{path}[{i}]", exact)]
    if a == b:
        return []
    fa, fb = _float(a), _float(b)
    if not exact and fa is not None and fb is not None and math.isfinite(fa) and math.isfinite(fb) \
            and abs(fa - fb) <= FLOAT_REL * max(abs(fa), abs(fb)) + FLOAT_ABS:
        return []
    return [f"{path}: frozen {a!r} != now {b!r}"]


def test_the_sweep_is_frozen_on_the_pre_change_commit_and_covers_the_contract():
    meta = FROZEN["_meta"]
    assert meta["captured_on"] == narrow_sweep.CAPTURED_ON == "ce89e73" and meta["callo"] != "callosum-v03"
    assert set(FROZEN["engine"]) == set(narrow_sweep.engine_cases())
    assert set(FROZEN["deliberate"]) == set(narrow_sweep.deliberate_cases())
    paths = {n.split("_")[0] for n in FROZEN["engine"]}
    assert paths == {"plain", "context", "image"}
    for path in paths:
        for typ in ("named", "letters", "score"):
            assert {int(n.split("_")[2]) for n in FROZEN["engine"] if n.startswith(f"{path}_{typ}_")} == {2, 3, 10, 26}
        assert {n for n in FROZEN["engine"] if n.startswith(f"{path}_noul")} == {
            f"{path}_{t}_2_{r}" for t in ("noul", "noulcrit") for r in (0, 1)}
    reads = [r for c in FROZEN["engine"].values() for r in c.get("readouts", [])]
    assert {r["kind"] for r in reads} == {"plain", "context", "image"}
    stops = {r["stop"] for r in reads}
    assert {"diverged", "no_shared_prefix", "max_depth", "retok"} <= stops
    assert any(r["bias"] == "applied" for r in reads) and any(r["depth"] > 0 for r in reads)
    assert any(r["topk_miss"] > 0 for r in reads)
    # every scoring pass asked for n_probs 100 (a generation asks for none); bodies are pinned by sha256 and length
    scored = [b for c in FROZEN["engine"].values() for b in c["http"] if b["ep"] == "completion"]
    scored += [b for c in FROZEN["deliberate"].values() for b in c["http"] if b["ep"] == "/completion" and b["n_probs"]]
    assert len(scored) > 200 and all(b["n_probs"] == 100 and len(b["sha256"]) == 64 for b in scored)
    assert {b["ep"] for c in FROZEN["engine"].values() for b in c["http"]} == {"completion", "tokenize", "detokenize"}
    for c in FROZEN["deliberate"].values():
        assert '"think_budget"' in c["header"] and all(r["bias"] == "deliberate" for r in c["readouts"])
    assert {"guard_empty", "swamped", "refused_two_certain", "decides", "near_certain"} <= set(FROZEN["walks"])
    lines = narrow_sweep.unpack_lines(FROZEN["profiles"])
    assert len(lines) == 576 and sum(map(bool, lines.values())) > 500
    assert not any("wide_contract" in ln for v in lines.values() for ln in v)


def test_prompt_sha256_is_unchanged():
    from callo.profile import prompt_sha256
    assert prompt_sha256() == FROZEN["prompt_sha256"]


@pytest.mark.parametrize("name", sorted(FROZEN["engine"]))
def test_plain_context_and_image_readouts_are_byte_identical(name):
    assert _diffs(FROZEN["engine"][name], _now("engine")[name]) == []


@pytest.mark.parametrize("name", sorted(FROZEN["deliberate"]))
def test_deliberate_readouts_are_byte_identical(name):
    assert _diffs(FROZEN["deliberate"][name], _now("deliberate")[name]) == []


@pytest.mark.parametrize("name", sorted(FROZEN["walks"]))
def test_the_walk_without_partial_is_unchanged(name):
    assert _diffs(FROZEN["walks"][name], _now("walks")[name]) == []


def test_profile_guards_emit_the_same_lines():
    """Legacy profiles (no `wide_contract`) and current ones, against the current `provenance()`: same lines."""
    now, then = _now("profiles"), narrow_sweep.unpack_lines(FROZEN["profiles"])
    assert set(now) == set(then)
    assert {k: (then[k], now[k]) for k in now if now[k] != then[k]} == {}


def test_the_float_slack_is_tiny_and_never_reaches_bodies():
    x = 0.1
    assert _diffs({"p": x.hex()}, {"p": math.nextafter(x, 1).hex()}, exact=False) == []
    assert _diffs({"p": x.hex()}, {"p": math.nextafter(x, 1).hex()}, exact=True) != []
    assert _diffs({"p": 0.0.hex()}, {"p": (1.5e-17).hex()}, exact=False) == []     # a 3.10 sum() residue at 0
    assert _diffs({"p": x.hex()}, {"p": (x * (1 + 1e-11)).hex()}, exact=False) != []
    assert _diffs({"p": 0.0.hex()}, {"p": (1e-14).hex()}, exact=False) != []
    assert _diffs({"b": '{"n_probs": 100}'}, {"b": '{"n_probs": 101}'}, exact=False) != []
    assert _diffs({"p": "inf"}, {"p": "-inf"}, exact=False) != []


def test_a_readout_body_without_the_exact_pin_or_in_another_serialisation_fails_the_sweep():
    from callo.scorer import readout_pin
    base = {"prompt": [1, 2], "n_predict": 1, "n_probs": 100, "temperature": 0.0, "cache_prompt": False}
    good = json.dumps({**base, **readout_pin(), "id_slot": 0})
    assert narrow_sweep.unpinned(good) == json.dumps({**base, "id_slot": 0})
    for bad in (json.dumps({**base, "id_slot": 0}),                                   # the pin dropped
                json.dumps({**base, **readout_pin(), "logit_bias": [[5, False]]}),     # a pin value changed
                json.dumps({**base, **readout_pin()}, separators=(",", ":"))):        # another serialisation
        with pytest.raises(AssertionError):
            narrow_sweep.unpinned(bad)
    assert narrow_sweep.unpinned('{"content": "x"}') == '{"content": "x"}'          # not a readout body: as it is
