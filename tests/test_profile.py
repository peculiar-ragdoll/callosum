"""callo.profile: format v2, phase 1 refusals, phase 2 key check and warnings, provenance."""
import hashlib
import json
import math

import pytest

from callo.errors import CalloError, InvalidInput
from callo.profile import PROVENANCE_FIELDS, check_profile, load_profile, prompt_sha256, provenance

KEY = "sha256:" + "ab" * 32
PROV = provenance(chat_template="{{ messages }}", build_info="b1-926862e", mmproj=None, user_delimiter=None,
                  cache_prompt=False)


def profile_dict(entries=None, *, kinds=("plain",), key=KEY, state="clean", prov=None, name="m.gguf"):
    """A minimal valid profile; later test files import this helper."""
    entries = {3: [0.5, 0.3, 0.2]} if entries is None else entries
    return {"callo_position_bias": 2, "estimator": "meanP", "model_key": key, "model_name": name,
            "created": "2026-10-03T12:00:00Z",
            "callo_version": "0.2.0", "provenance": dict(PROV if prov is None else prov),
            "server": {"kv": "unified", "temperature": 1.0, "s1_slots": 1, "slot": 0},
            "server_state": {"state": state, "checks": []},
            "set": {"name": "t", "sha256": "0" * 64, "builtin": False, "seed": 0, "questions": 24,
                    "kinds": list(kinds), "sources_per_n": {}, "skipped": {}, "capped": 0},
            "entries": {str(n): {"prior": p, "questions": 24, "floored": 0, "bias": 0.1, "p_holm": 0.01}
                        for n, p in entries.items()},
            "dropped": {}, "eval": {}}


def test_provenance_has_the_seven_fields_and_hashes_the_template():
    # cache_delimiter (v0.3 cache delimiters) is present only under --cache with a delimiter: a cache-off provenance
    # keeps the seven fields it always had, plus wide-choice's wide_contract, which is not a PROVENANCE_FIELD
    assert tuple(PROV) == (*(f for f in PROVENANCE_FIELDS if f != "cache_delimiter"), "wide_contract")
    assert PROV["chat_template_sha256"] == hashlib.sha256(b"{{ messages }}").hexdigest()
    assert PROV["prompt_sha256"] == prompt_sha256() and len(prompt_sha256()) == 64
    p = provenance(chat_template=None, build_info=None, mmproj="/m/mmproj-BF16.gguf", user_delimiter="<u>",
                   cache_prompt=True)
    assert p["chat_template_sha256"] is None and p["mmproj"] == "mmproj-BF16.gguf" and p["cache_prompt"] is True
    assert p["server_args"] is None and PROV["server_args"] is None
    assert provenance(chat_template=None, build_info=None, mmproj=None, user_delimiter=None, cache_prompt=False,
                      server_args=("-ctk", "q8_0"))["server_args"] == ["-ctk", "q8_0"]


def test_a_round_trip_derives_centred_offsets_and_hashes_the_file(tmp_path):
    f = tmp_path / "bias.json"
    f.write_text(json.dumps(profile_dict({2: [0.6, 0.4], 3: [0.5, 0.3, 0.2]})))
    pb = load_profile(f)
    assert pb.sha256 == hashlib.sha256(f.read_bytes()).hexdigest() and pb.source == str(f)
    assert pb.priors == {2: (0.6, 0.4), 3: (0.5, 0.3, 0.2)} and pb.kinds == frozenset({"plain"})
    assert pb.model_key == KEY and pb.model_name == "m.gguf" and pb.state == "clean" and pb.provenance == PROV
    off = pb.offsets[3]
    assert abs(sum(off)) < 1e-12 and math.isclose(off[0] - off[1], math.log(0.5 / 0.3), rel_tol=1e-12)
    assert pb.info() == {"profile": str(f), "sha256": pb.sha256, "entries": [2, 3], "kinds": ["plain"],
                         "state": "clean"}
    assert load_profile(profile_dict()).source == "<dict>"


def _bad(**changes):
    d = profile_dict()
    for k, v in changes.items():
        d[k] = v
    return d


def _entry(prior):
    return {"3": {"prior": prior}}


@pytest.mark.parametrize(("data", "match"), [
    (_bad(callo_position_bias=3), "callo_position_bias must be 2"),
    (_bad(callo_position_bias=True), "callo_position_bias must be 2"),
    (_bad(callo_position_bias=1), "format 1 was fitted with the PriDe estimator, withdrawn"),
    (_bad(set={"kinds": ["plain"], "builtin": True}), "fitted on the built-in set, which is a diagnostic only"),
    (_bad(model_key="sha256:xyz"), "model_key"),
    (_bad(model_key=None), "model_key"),
    (_bad(entries=[]), "entries must be an object"),
    (_bad(entries={"1": {"prior": [1.0]}}), "entry key '1'"),
    (_bad(entries={"27": {"prior": [1 / 27] * 27}}), "entry key '27'"),
    (_bad(entries={"03": {"prior": [0.5, 0.3, 0.2]}}), "entry key '03'"),
    (_bad(entries={"x": {"prior": [0.5, 0.5]}}), "entry key 'x'"),
    (_bad(entries={"\u00b2": {"prior": [0.5, 0.5]}}), "entry key '\u00b2'"),          # isdigit() but not ASCII
    (_bad(entries={"\u0663": {"prior": [0.5, 0.3, 0.2]}}), "entry key '\u0663'"),    # Arabic-Indic 3: int() == 3
    (_bad(entries={" 3": {"prior": [0.5, 0.3, 0.2]}}), "entry key ' 3'"),
    (_bad(entries={"3": [0.5, 0.3, 0.2]}), "entry 3 must be an object with a prior"),
    (_bad(entries=_entry([0.5, 0.5])), "prior of entry 3 must have 3 values"),
    (_bad(entries=_entry([0.6, 0.6, -0.2])), "finite and positive"),
    (_bad(entries=_entry([0.5, 0.5, 0.0])), "finite and positive"),
    (_bad(entries=_entry([0.5, 0.5, float("nan")])), "finite and positive"),
    (_bad(entries=_entry([True, 0.5, 0.5])), "finite and positive"),
    (_bad(entries=_entry([10 ** 400, 0.5, 0.5])), "finite and positive"),          # math.isfinite overflows
    (_bad(model_key=KEY + "\n"), "model_key"),
    (_bad(entries=_entry([0.5, 0.3, 0.19])), "sum to 1"),
    (_bad(set={"kinds": []}), "set.kinds"),
    (_bad(set={"kinds": ["plain", "holo"]}), "set.kinds"),
    (_bad(set={}), "set.kinds"),
])
def test_every_phase1_refusal(data, match):
    with pytest.raises(InvalidInput, match=match):
        load_profile(data)


@pytest.mark.parametrize("data", [_bad(extra={1, 2}), _bad(extra={1: "a", "b": "c"})])
def test_a_dict_profile_that_is_not_json_serialisable_is_invalid_input(data):
    """A set value or mixed-type keys make json.dumps raise TypeError; that is InvalidInput."""
    with pytest.raises(InvalidInput, match="--position-bias <dict>: not JSON-serialisable"):
        load_profile(data)


def test_phase1_refuses_unreadable_and_non_json_files(tmp_path):
    with pytest.raises(InvalidInput, match="cannot read"):
        load_profile(tmp_path / "missing.json")
    (tmp_path / "x.json").write_text("{nope")
    with pytest.raises(InvalidInput, match="not valid JSON"):
        load_profile(tmp_path / "x.json")
    (tmp_path / "y.json").write_text("[1]")
    with pytest.raises(InvalidInput, match="not a JSON object"):
        load_profile(tmp_path / "y.json")


def test_set_kinds_accept_every_readout_kind():
    from callo.engine import READOUT_KINDS
    assert READOUT_KINDS == ("plain", "context", "image", "audio", "video", "mixed")
    assert load_profile(profile_dict(kinds=READOUT_KINDS)).kinds == frozenset(READOUT_KINDS)


def test_a_sum_within_1e_6_is_accepted():
    assert load_profile(_bad(entries=_entry([0.5, 0.3, 0.2 + 5e-7]))).priors[3][2] == 0.2 + 5e-7


def test_phase2_refuses_another_model_and_a_native_upstream():
    pb = load_profile(profile_dict(name="Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf"))
    with pytest.raises(CalloError, match="this profile was measured on Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf"):
        check_profile(pb, model_key="sha256:" + "cd" * 32, native=False, provenance=PROV)
    with pytest.raises(CalloError, match="native decision model"):
        check_profile(pb, model_key=KEY, native=True, provenance=PROV)
    assert check_profile(pb, model_key=KEY, native=False, provenance=PROV) == []


def test_phase2_warns_once_per_differing_provenance_field_and_on_busy_and_empty():
    pb = load_profile(profile_dict(state="busy"))
    now = {**PROV, "build_info": "b9999-x", "cache_prompt": True, "server_args": ["-ctk", "q8_0"]}
    lines = check_profile(pb, model_key=KEY, native=False, provenance=now)
    assert len([ln for ln in lines if "differs" in ln]) == 3
    assert any("build_info" in ln for ln in lines) and any("cache_prompt" in ln for ln in lines)
    assert any("server_args" in ln for ln in lines)
    assert any("busy" in ln for ln in lines) and all(ln.startswith("WARNING:") for ln in lines)
    empty = load_profile(profile_dict(entries={}))
    assert any("nothing will be corrected" in ln for ln in check_profile(empty, model_key=KEY, native=False,
                                                                           provenance=PROV))
