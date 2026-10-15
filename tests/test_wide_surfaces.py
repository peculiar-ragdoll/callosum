"""callo v0.3 wide-choice, the surfaces: X-Callo-Timings `unread` / `split`, the probe's readout fields and its coverage 422 before any generation,
/callo/stats `wide_coverage` and the `wide_*` counters, `skipped.wide`, the optional `wide_contract` provenance and its
asymmetric guard (bias profile, think profile, compare), and serve's startup line."""
import json
import logging

import pytest
from fake_upstream import FakeUpstream
from helpers import Running, get_json, post_json, tiny_gguf
from test_calibrate_think import SAMPLING, STATS, synthetic_raw, write_data
from test_compare import profile as compare_profile
from test_profile import KEY
from test_profile import profile_dict as bias_profile_dict
from test_think import profile_dict as think_profile_dict
from test_wide_engine import SHORT, _think_engine, vocab

from callo.calibrate_think import build_profile, load_data
from callo.compare import flags
from callo.engine import WIDE_TOP_K, QuestionReadout, Readout
from callo.marginalize import WIDE_MAX_SPLIT_PASSES, WIDE_SPLIT_READ
from callo.profile import (
    PROVENANCE_FIELDS,
    SKIPPED_STATES,
    WIDE_CONTRACT,
    check_profile,
    load_profile,
    provenance,
    wide_contract_warning,
)
from callo.proxy import Stats, ThinkProbe, _readout_fields, timings_header
from callo.slots import SlotRouter
from callo.systemone import WIDE_LABELS_SHA256
from callo.think import check_think_profile, load_think_profile

NARROW = Readout((0.5, 0.5), (-0.7, -0.7), 0.9, "diverged", 0, 0, 12, kind="plain", passes=1)
WIDE = Readout((0.5, 0.5), (-0.7, -0.7), 0.9, "diverged", 0, 0, 12, kind="plain", passes=2,
               unread_mass=0.0123456789, split_passes=1, wide=True, partial_by_letter=(("A", 0.4), ("B", 0.0)))


# ---- X-Callo-Timings and the probe's readout fields ---------------------------------------------------------------

def test_the_timings_entry_of_a_narrow_question_is_unchanged():
    assert timings_header([QuestionReadout("q", NARROW, 3.21)]) == (
        '[{"q":"q","kind":"plain","cache_n":null,"prompt_n":null,"prompt_ms":null,"tokens":12,"stop":"diverged",'
        '"ms":3.2,"miss":0,"passes":1}]')


def test_the_timings_entry_of_a_wide_question_adds_unread_and_split_after_passes():
    assert timings_header([QuestionReadout("q", WIDE, 3.21)]) == (
        '[{"q":"q","kind":"plain","cache_n":null,"prompt_n":null,"prompt_ms":null,"tokens":12,"stop":"diverged",'
        '"ms":3.2,"miss":0,"passes":2,"unread":0.012346,"split":1}]')


def test_the_probe_readout_fields_gain_the_wide_ones_only_for_a_wide_readout():
    narrow = _readout_fields(NARROW)
    assert list(narrow) == ["probs", "log_marginals", "option_mass", "stop", "depth", "miss", "passes", "bias",
                            "prompt_n", "cache_n", "prompt_ms"]
    wide = _readout_fields(WIDE)
    assert list(wide) == [*narrow, "unread_mass", "split_passes", "partial_by_letter"]
    assert wide["unread_mass"] == 0.0123456789 and wide["split_passes"] == 1
    assert wide["partial_by_letter"] == [["A", 0.4], ["B", 0.0]]
    assert json.loads(json.dumps(wide))["partial_by_letter"] == [["A", 0.4], ["B", 0.0]]


# ---- the think-probe refuses a question past the coverage before generating ----------------------------------

@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=5).start()
    yield u
    u.stop()


@pytest.mark.parametrize("extra", [{"budgets": [64]}, {"budgets": [0, 64]}, {"budgets": [64], "plan": True}])
def test_the_probe_answers_422_before_any_render_or_generation(up, extra):
    e = _think_engine(up, SHORT)
    stats = Stats()
    stats.think = {"capable": True}
    probe = ThinkProbe(e, SlotRouter(4, 1), stats, s1_timeout=5)
    up.requests.clear()
    body = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {f"k{i}": None for i in range(27)}}}}
    code, out, _ = probe.respond({"request": body, **extra}, {})
    assert code == 422
    assert out["detail"] == ("q: this model's tokenizer can read 6 of Callo's two-letter labels (the next one needs a "
                             "letter that is not a single token); the question has 27 options")
    assert up.requests == []                                                 # nothing reached the upstream
    narrow = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {f"k{i}": None for i in range(26)}}}}
    assert probe.respond({"request": narrow, **extra}, {})[0] == 200


# ---- /callo/stats --------------------------------------------------------------------------------------------------

def test_the_stats_count_wide_readouts_their_split_passes_and_unread_above_the_threshold():
    stats = Stats()
    over = Readout((1.0,) * 2, (0.0,) * 2, 0.9, "diverged", 0, 0, 1, passes=1, unread_mass=WIDE_SPLIT_READ * 2,
                   wide=True)
    at = Readout((1.0,) * 2, (0.0,) * 2, 0.9, "diverged", 0, 0, 1, passes=3, unread_mass=WIDE_SPLIT_READ,
                 split_passes=2, wide=True)
    stats.add_request("plain", 10.0, [QuestionReadout("a", NARROW, 1.0), QuestionReadout("b", WIDE, 1.0),
                                      QuestionReadout("c", over, 1.0), QuestionReadout("d", at, 1.0)])
    snap = stats.snapshot(SlotRouter(4, 1), None)
    s1 = snap["s1"]
    assert (s1["wide_readouts"], s1["wide_split_passes"], s1["wide_unread_over"]) == (3, 3, 2)
    assert snap["wide_coverage"] is None and snap["wide_readable"] is None   # serve sets them
    assert s1["wide_uncovered_readouts"] == 0
    stats.wide_coverage, stats.wide_readable = 9, 255
    snap = stats.snapshot(SlotRouter(4, 1), None)
    assert (snap["wide_coverage"], snap["wide_readable"]) == (9, 255)
    past = Readout((1.0,) * 2, (0.0,) * 2, 0.9, "diverged", 0, 0, 1, passes=2, split_passes=1, wide=True, uncovered=3)
    stats.add_request("plain", 10.0, [QuestionReadout("e", past, 1.0)])
    s1 = stats.snapshot(SlotRouter(4, 1), None)["s1"]
    assert (s1["wide_readouts"], s1["wide_split_passes"], s1["wide_uncovered_readouts"]) == (4, 4, 1)


def test_a_fresh_stats_has_zero_wide_counters():
    s1 = Stats().snapshot(SlotRouter(4, 1), None)["s1"]
    assert (s1["wide_readouts"], s1["wide_split_passes"], s1["wide_unread_over"], s1["wide_uncovered_readouts"]) == \
        (0, 0, 0, 0)


def test_the_bias_block_counts_skipped_wide():
    assert SKIPPED_STATES[-1] == "wide" and "wide" in SKIPPED_STATES
    stats = Stats()
    stats.position_bias = {"profile": "p", "sha256": "s", "entries": [3], "kinds": ["plain"], "state": "clean"}
    r = Readout((1.0,) * 2, (0.0,) * 2, 0.9, "diverged", 0, 0, 1, bias="wide", wide=True)
    stats.add_request("plain", 1.0, [QuestionReadout("q", r, 1.0)])
    assert stats.snapshot(SlotRouter(4, 1), None)["position_bias"]["skipped"]["wide"] == 1


# ---- provenance: wide_contract and its asymmetric guard ------------------------------------------------------------

def _prov(**over):
    return {**provenance(chat_template="{{ messages }}", build_info="b1-926862e", mmproj=None, user_delimiter=None,
                         cache_prompt=False), **over}


def test_provenance_carries_the_wide_contract_outside_the_provenance_fields():
    assert WIDE_CONTRACT == {"labels_sha256": WIDE_LABELS_SHA256, "top_k": WIDE_TOP_K, "split_read": WIDE_SPLIT_READ,
                             "max_split_passes": WIDE_MAX_SPLIT_PASSES}
    p = _prov()
    assert p["wide_contract"] == WIDE_CONTRACT and p["wide_contract"] is not WIDE_CONTRACT
    assert "wide_contract" not in PROVENANCE_FIELDS


def test_wide_contract_warning_only_when_both_sides_carry_it_and_differ():
    other = {**WIDE_CONTRACT, "top_k": 500}
    assert wide_contract_warning(None, WIDE_CONTRACT, "--position-bias") is None
    assert wide_contract_warning(WIDE_CONTRACT, None, "--position-bias") is None
    assert wide_contract_warning(None, None, "--position-bias") is None
    assert wide_contract_warning(dict(WIDE_CONTRACT), WIDE_CONTRACT, "--position-bias") is None
    line = wide_contract_warning(other, WIDE_CONTRACT, "--think-profile")
    assert line is not None and line.startswith("WARNING: --think-profile: wide_contract differs")
    assert "500" in line and "1000" in line


def test_check_profile_an_old_profile_without_the_field_warns_exactly_as_before():
    old_prov = {k: v for k, v in _prov().items() if k != "wide_contract"}
    pb = load_profile(bias_profile_dict(prov=old_prov))
    assert check_profile(pb, model_key=KEY, native=False, provenance=_prov()) == []
    assert check_profile(pb, model_key=KEY, native=False, provenance=old_prov) == []
    new = load_profile(bias_profile_dict(prov=_prov()))
    assert check_profile(new, model_key=KEY, native=False, provenance=_prov()) == []
    moved = _prov(wide_contract={**WIDE_CONTRACT, "split_read": 0.05})
    lines = check_profile(new, model_key=KEY, native=False, provenance=moved)
    assert len(lines) == 1 and lines[0].startswith("WARNING: --position-bias: wide_contract differs")


def _think_lines(tp, prov):
    return check_think_profile(tp, provenance=prov, host_fingerprint="sha256:" + "h" * 64, build_info="b1-926862e",
                               position_bias_sha256=None, temperature_from_profile=False, default_budget=0)


def test_check_think_profile_an_old_profile_without_the_field_gains_no_line():
    tp = load_think_profile(think_profile_dict())
    assert "wide_contract" not in tp.provenance
    base = _think_lines(tp, tp.provenance)
    assert _think_lines(tp, {**tp.provenance, "wide_contract": dict(WIDE_CONTRACT)}) == base
    new = load_think_profile(think_profile_dict(provenance={**tp.provenance, "wide_contract": dict(WIDE_CONTRACT)}))
    assert _think_lines(new, {**tp.provenance, "wide_contract": dict(WIDE_CONTRACT)}) == base
    lines = _think_lines(new, {**tp.provenance, "wide_contract": {**WIDE_CONTRACT, "max_split_passes": 3}})
    assert [x for x in lines if x not in base] == [x for x in lines if "wide_contract" in x]
    assert len([x for x in lines if x.startswith("WARNING: --think-profile: wide_contract differs")]) == 1


def test_compare_flags_differing_wide_contracts_only_among_the_profiles_that_carry_one():
    old = compare_profile("alpha", {0: 5, 256: 8}, {0: 300.0, 256: 2000.0})
    new = compare_profile("beta", {0: 6, 256: 9}, {0: 400.0, 256: 2500.0}, key="k2",
                          comparable__readout__wide_contract=dict(WIDE_CONTRACT))
    other = compare_profile("gamma", {0: 6, 256: 9}, {0: 400.0, 256: 2500.0}, key="k3",
                            comparable__readout__wide_contract={**WIDE_CONTRACT, "top_k": 500})
    what = "different wide-choice readout contracts"
    assert not any(f.startswith(what) for f in flags([old, new]))
    assert not any(f.startswith(what) for f in flags([new, compare_profile(
        "delta", {0: 6, 256: 9}, {0: 400.0, 256: 2500.0}, key="k4", comparable__readout__wide_contract=dict(WIDE_CONTRACT))]))
    hit = [f for f in flags([old, new, other]) if f.startswith(what)]
    assert len(hit) == 1 and "beta" in hit[0] and "gamma" in hit[0] and "alpha" not in hit[0]


# ---- serve: the startup line and /callo/stats wide_coverage --------------------------------------------------------

def _wide_gguf(path, tokens):
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.mark.parametrize(("which", "cov", "line"), [
    ("full", 255, "wide-choice: 255 of 255 two-letter labels are single tokens; questions up to 255 options"),
    ("short", 9, "wide-choice: 9 of 255 two-letter labels are single tokens; questions up to 255 options (beyond 9, "
                 "a label that is not a single token is read in two passes)"),
    ("letters", 0, "wide-choice: 0 of 255 two-letter labels are single tokens; questions up to 255 options (beyond "
                   "0, a label that is not a single token is read in two passes)"),
])
def test_serve_warms_the_wide_sets_and_logs_the_coverage(tmp_path, caplog, which, cov, line):
    up = FakeUpstream().start()
    if which == "letters":
        gguf = tiny_gguf(tmp_path / "m.gguf")
    else:
        gguf = _wide_gguf(tmp_path / "m.gguf", vocab(255) if which == "full" else vocab(40, drop_all=9))
    up.props["model_path"] = str(gguf)
    try:
        with caplog.at_level(logging.INFO, logger="callo"):
            r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        try:
            assert r.port
            _, _, stats = get_json(f"{r.url}/callo/stats")
            assert (stats["wide_coverage"], stats["wide_readable"]) == (cov, 255)
            two_pass = {"uncovered": "two_pass"} if cov < 255 else {}
            assert stats["provenance"]["wide_contract"] == {**WIDE_CONTRACT, **two_pass}
            # every vocabulary here reads 40 options, past its coverage in two passes
            body = {"state": "s", "questions": {"q": {"type": "choice",
                                                      "criteria": {f"k{i}": None for i in range(40)}}}}
            status, headers, _ = post_json(f"{r.url}/v1/systemone", body)
            entry = json.loads(headers["X-Callo-Timings"])[0]
            assert status == 200 and "unread" in entry and "split" in entry
            _, _, stats = get_json(f"{r.url}/callo/stats")
            assert stats["s1"]["wide_readouts"] == 1
            assert stats["s1"]["wide_uncovered_readouts"] == (cov < 40)
        finally:
            assert r.finish() == 0
    finally:
        up.stop()
    assert [m for m in caplog.messages if m.startswith("wide-choice:")] == [line]
    order = [caplog.text.index(s) for s in ("self-test ok: 2 + 2", "wide-choice:")]
    assert order == sorted(order)                                            # after the self-test



def test_a_think_profile_records_the_served_wide_contract_in_its_readout_block(tmp_path):
    """compare reads `comparable.readout.wide_contract`: calibrate copies it from the served provenance, and a run
    against a server without it writes none (so it is never compared)."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    old = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert "wide_contract" not in old["comparable"]["readout"]
    raw.header["provenance"] = {**raw.header["provenance"], "wide_contract": dict(WIDE_CONTRACT)}
    new = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert new["comparable"]["readout"]["wide_contract"] == WIDE_CONTRACT == new["provenance"]["wide_contract"]
