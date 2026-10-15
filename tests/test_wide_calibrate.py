"""callo v0.3 wide-choice, calibration: `callo calibrate --think-budgets` drops every question above max(26, what the
server reads: `wide_readable`, else `wide_coverage` on a 0.3.0 server) before probing, as `too_many_options`, and never
aborts on one; a `--resume` refit with no server keeps working; and `callo calibrate --bias` still skips a question
above 26 options as `too_many_options`."""
import json

import pytest
from fake_upstream import FakeUpstream
from helpers import Running
from test_calibrate_think import _len, _Served, _settings, write_data
from test_wide_engine import vocab
from test_wide_surfaces import _wide_gguf

import callo.calibrate_think as ct
from callo.bias import read_set
from callo.calibrate_think import calibrate_one, coverage_filter, load_data, read_raw

WIDE_ID = "wide-1"
WIDE = json.dumps({"state": "wide state", "questions": {"q": {
    "type": "choice", "instructions": "pick", "criteria": {f"k{i}": f"d{i}" for i in range(30)}, "label": "k3"}},
    "_meta": {"id": WIDE_ID, "group_id": "gw", "variant": "clean"}})


def _data(tmp_path, n=4):
    return write_data(tmp_path / "d.jsonl", n, extra=[WIDE])


def test_load_data_keeps_a_wide_record(tmp_path):
    items, skipped = load_data(_data(tmp_path))
    assert [it.item_id for it in items][-1] == WIDE_ID and len(items[-1].question.labels) == 30
    assert "too_many_options" not in skipped and skipped["records"] == 5


@pytest.mark.parametrize(("stats", "kept", "limit"), [
    ({"wide_coverage": 255}, True, 255), ({"wide_coverage": 30}, True, 30), ({"wide_coverage": 29}, False, 29),
    ({"wide_coverage": 9}, False, 26), ({"wide_coverage": 0}, False, 26), ({"wide_coverage": None}, False, 26),
    ({}, False, 26),                                                         # a server without the field: 26
    # 0.3.1: what the server reads decides, the coverage only for a server that does not say
    ({"wide_coverage": 9, "wide_readable": 255}, True, 255), ({"wide_coverage": 0, "wide_readable": 255}, True, 255),
    ({"wide_coverage": 6, "wide_readable": 29}, False, 29), ({"wide_coverage": 9, "wide_readable": None}, False, 26),
])
def test_coverage_filter_drops_questions_above_max_26_and_the_coverage(tmp_path, stats, kept, limit):
    items, _ = load_data(_data(tmp_path))
    keep, dropped, lim = coverage_filter(items, stats)
    assert lim == limit and dropped == (0 if kept else 1)
    assert [it.uid for it in keep] == [it.uid for it in items if kept or it.item_id != WIDE_ID]


def _wide_uids(raw_path):
    return {r["q"] for r in read_raw(raw_path).rows if r["q"].startswith(WIDE_ID)}


def test_a_short_coverage_server_reads_the_wide_record_in_two_passes(tmp_path):
    """The default fake serves a letters-only vocabulary (coverage 0, every label read in two passes): the 30-option
    record is probed like any other, and the profile's wide contract says its labels were read in two passes."""
    s = _Served(tmp_path)
    s.up.bias = None                     # the planted readout parses single-letter options only: every read is canned
    try:
        out = calibrate_one(_settings(_data(tmp_path), budgets=(0, 64)), url=s.url, name="fake",
                            out_dir=tmp_path / "think", resume=False, log=lambda _: None, therm=lambda: 100)
        assert out.status == "done", out.reason
        bodies = [json.dumps(q.json) for q in s.up.requests if q.json is not None]
        assert any("wide state" in b for b in bodies)
    finally:
        s.stop()
    prof = json.loads(out.profile.read_text())
    assert prof["comparable"]["data"]["skipped"] == {} and prof["comparable"]["data"]["questions"] == 5
    assert _wide_uids(tmp_path / "think" / "fake.raw.jsonl") == {f"{WIDE_ID}/q"}
    assert prof["comparable"]["readout"]["wide_contract"] == {**WIDE_CONTRACT, "uncovered": "two_pass"}
    assert read_raw(tmp_path / "think" / "fake.raw.jsonl").header["wide_readable"] == 255


def test_a_server_that_does_not_read_past_its_coverage_skips_the_wide_record_and_the_run_completes(tmp_path,
                                                                                                  monkeypatch):
    """A 0.3.0 server (it reports `wide_coverage` 0 and no `wide_readable`, read here by hiding the field): the
    30-option record is dropped before any probe, counted as too_many_options, and the offline refit of the complete
    raw file still works."""
    real = ct.read_stats
    monkeypatch.setattr(ct, "read_stats", lambda base: {k: v for k, v in real(base).items() if k != "wide_readable"})
    s = _Served(tmp_path)
    log: list[str] = []
    try:
        out = calibrate_one(_settings(_data(tmp_path), budgets=(0, 64)), url=s.url, name="fake",
                            out_dir=tmp_path / "think", resume=False, log=log.append, therm=lambda: 100)
        assert out.status == "done", out.reason
        bodies = [json.dumps(q.json) for q in s.up.requests if q.json is not None]
        assert bodies and not any("wide state" in b for b in bodies)        # never rendered, never sent upstream
    finally:
        s.stop()
    prof = json.loads(out.profile.read_text())
    assert prof["comparable"]["data"]["skipped"] == {"too_many_options": 1}
    assert prof["comparable"]["data"]["questions"] == 4 and prof["comparable"]["data"]["records"] == 5
    assert not _wide_uids(tmp_path / "think" / "fake.raw.jsonl")
    assert any("too_many_options" in line and "26" in line for line in log)
    # --resume with the server gone: the complete raw file refits with no request, the skip recorded as before
    again = calibrate_one(_settings(_data(tmp_path), budgets=(0, 64)), url=s.url, name="fake",
                          out_dir=tmp_path / "think", resume=True, log=log.append, therm=lambda: 100)
    assert again.status == "done", again.reason
    refit = json.loads(again.profile.read_text())
    assert refit["comparable"]["data"] == prof["comparable"]["data"] and refit["budgets"] == prof["budgets"]


def test_a_full_coverage_server_probes_the_wide_record_and_one_without_the_field_skips_it(tmp_path, monkeypatch):
    up = FakeUpstream(think="preopened", think_len=_len, slot_save=True).start()
    up.bias = None                       # the planted readout parses single-letter options only: every read is canned
    up.props["model_path"] = str(_wide_gguf(tmp_path / "m.gguf", vocab(255)))
    r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
    try:
        assert r.port
        out = calibrate_one(_settings(_data(tmp_path), budgets=(0, 64)), url=r.url, name="wide",
                            out_dir=tmp_path / "a", resume=False, log=lambda _: None, therm=lambda: 100)
        assert out.status == "done", out.reason
        prof = json.loads(out.profile.read_text())
        assert prof["comparable"]["data"]["skipped"] == {} and prof["comparable"]["data"]["questions"] == 5
        assert _wide_uids(tmp_path / "a" / "wide.raw.jsonl") == {f"{WIDE_ID}/q"}
        assert prof["comparable"]["readout"]["wide_contract"]["top_k"] == 1000      # served provenance
        real = ct.read_stats
        monkeypatch.setattr(ct, "read_stats", lambda base: {k: v for k, v in real(base).items()
                                                            if k not in ("wide_coverage", "wide_readable")})
        old = calibrate_one(_settings(_data(tmp_path), budgets=(0, 64)), url=r.url, name="old",
                            out_dir=tmp_path / "b", resume=False, log=lambda _: None, therm=lambda: 100)
        assert old.status == "done", old.reason
        assert json.loads(old.profile.read_text())["comparable"]["data"]["skipped"] == {"too_many_options": 1}
        assert not _wide_uids(tmp_path / "b" / "old.raw.jsonl")
    finally:
        r.finish()
        up.stop()


def test_calibrate_bias_still_skips_a_30_option_question_as_too_many_options():
    scan = read_set(WIDE)
    assert dict(scan.skipped) == {"too_many_options": 1} and scan.items == []


# ---- resumes across coverage changes, the skip line, the shared question order -------------------------------

from callo import calibrate_bias as cb  # noqa: E402
from callo.calibrate_bias import Refusal  # noqa: E402
from callo.calibrate_think import (  # noqa: E402
    ThinkSettings,
    coverage_skip_line,
    header_diff,
    probed_only,
    raw_header,
    shuffled,
)
from callo.profile import WIDE_CONTRACT  # noqa: E402
from callo.think import parse_sampling, resolve_sampling  # noqa: E402

PROV_OLD = {"prompt_sha256": "p" * 64, "chat_template_sha256": "t" * 64, "build_info": "b1", "mmproj": None,
            "user_delimiter": None, "cache_prompt": True, "server_args": None}
PROV_NEW = {**PROV_OLD, "wide_contract": dict(WIDE_CONTRACT)}
_ABSENT = object()


def _think_header(prov, cov=_ABSENT):
    stats = {"model_key": "sha256:" + "a" * 64, "provenance": prov, "host": {"fingerprint": "sha256:h"},
             "instance": "i1", "model_file": "m.gguf", **({} if cov is _ABSENT else {"wide_coverage": cov})}
    h = raw_header(stats, ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 256)),
                   resolve_sampling(parse_sampling("fixed")))
    if cov is _ABSENT:
        h.pop("wide_coverage")                          # a raw file written before the field existed
    return h


def test_raw_header_records_the_wide_coverage():
    assert _think_header(PROV_NEW, 255)["wide_coverage"] == 255
    h = raw_header({"provenance": PROV_NEW, "wide_coverage": 9, "wide_readable": 255},
                   ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 256)), resolve_sampling(parse_sampling("fixed")))
    assert (h["wide_coverage"], h["wide_readable"]) == (9, 255)
    assert raw_header({"provenance": PROV_NEW}, ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 256)),
                      resolve_sampling(parse_sampling("fixed")))["wide_coverage"] is None


def test_think_resume_an_old_raw_file_resumes_against_a_new_server():
    """The RECORDED side lacks wide_contract / wide_coverage (written before them): ignored on both sides."""
    old, new = _think_header(PROV_OLD), _think_header(PROV_NEW, 255)
    assert header_diff(old, new) == ([], [])
    assert header_diff(_think_header(PROV_OLD, None), new) == ([], [])          # recorded null: the same
    assert header_diff(old, {**new, "provenance": {**PROV_NEW, "chat_template_sha256": "u" * 64}})[0] == ["provenance"]


def test_think_resume_a_new_raw_file_against_a_pre_wide_server_is_refused():
    new, old = _think_header(PROV_NEW, 255), _think_header(PROV_OLD)
    assert header_diff(new, old)[0] == ["provenance", "wide_coverage"]


def test_think_resume_different_wide_values_are_refused():
    new = _think_header(PROV_NEW, 255)
    other = {**new, "provenance": {**PROV_NEW, "wide_contract": {**WIDE_CONTRACT, "top_k": 500}}}
    assert header_diff(new, other)[0] == ["provenance"]
    assert header_diff(new, {**new, "wide_coverage": 9})[0] == ["wide_coverage"]
    assert header_diff(new, new) == ([], [])


def test_think_resume_across_the_two_pass_reading():
    """`wide_readable` is compared like `wide_coverage` (absent or null when recorded: compatible). A short-coverage
    0.3.0 raw file resumed against 0.3.1 is still refused, by its wide contract: 0.3.0 skipped the questions 0.3.1
    reads in two passes, so the two runs never mix silently."""
    new = {**_think_header(PROV_NEW, 9), "wide_readable": 255}
    assert header_diff(_think_header(PROV_NEW, 9), new) == ([], [])                   # recorded without the field
    assert header_diff(new, {**new, "wide_readable": 30})[0] == ["wide_readable"]
    two_pass = {**new, "provenance": {**PROV_NEW, "wide_contract": {**WIDE_CONTRACT, "uncovered": "two_pass"}}}
    assert header_diff(_think_header(PROV_NEW, 9), two_pass)[0] == ["provenance"]
    assert header_diff(two_pass, two_pass) == ([], [])


def _bias_header(prov):
    return {"model_key": "sha256:" + "a" * 64, "provenance": prov, "server": {"kv": "unified", "slot": 0},
            "set": {"name": "t", "sha256": "0" * 64}, "seed": 0, "max_per_n": 50}


def _bias_raw(tmp_path, prov):
    p = tmp_path / "bias.raw.jsonl"
    p.write_text(json.dumps({"callo_bias_raw": 1, **_bias_header(prov)}) + "\n")
    return p


def test_bias_resume_an_old_raw_file_resumes_against_a_new_server(tmp_path):
    state = cb._read_raw(_bias_raw(tmp_path, PROV_OLD), _bias_header(PROV_NEW))
    assert state.readings == {} and state.session == 0


def test_bias_resume_a_new_raw_file_against_a_pre_wide_server_is_refused(tmp_path):
    with pytest.raises(Refusal, match=r"\(provenance differ"):
        cb._read_raw(_bias_raw(tmp_path, PROV_NEW), _bias_header(PROV_OLD))


def test_bias_resume_different_wide_contracts_are_refused(tmp_path):
    other = {**PROV_NEW, "wide_contract": {**WIDE_CONTRACT, "split_read": 0.05}}
    with pytest.raises(Refusal, match=r"\(provenance differ"):
        cb._read_raw(_bias_raw(tmp_path, PROV_NEW), _bias_header(other))
    with pytest.raises(Refusal, match=r"\(provenance differ"):                 # anything else still refuses
        cb._read_raw(_bias_raw(tmp_path, PROV_OLD), _bias_header({**PROV_NEW, "build_info": "b2"}))


def test_probed_only_keeps_a_covered_wide_item_when_the_header_records_the_coverage(tmp_path):
    items, _ = load_data(_data(tmp_path))
    raw = ct.RawThink({"wide_coverage": 255}, [], [], [], False)                 # no row at all for any item
    assert probed_only(items, raw) == (items, 0)
    narrow = [it for it in items if it.item_id != WIDE_ID]
    assert probed_only(items, ct.RawThink({"wide_coverage": 9}, [], [], [], False)) == (narrow, 1)
    assert probed_only(items, ct.RawThink({}, [], [], [], False)) == (narrow, 1)               # row-presence rule
    assert probed_only(items, ct.RawThink({"wide_coverage": True}, [], [], [], False)) == (narrow, 1)
    both = ct.RawThink({"wide_coverage": 9, "wide_readable": 255}, [], [], [], False)       # 0.3.1: what it reads
    assert probed_only(items, both) == (items, 0)


def test_a_resume_with_only_the_covered_wide_item_unread_reads_it_through_the_server(tmp_path):
    """A budget-0 run on a coverage-255 server interrupted with only the wide item unread;
    the resume must go through the server and read it, not refit offline and skip it."""
    up = FakeUpstream(think="preopened", think_len=_len, slot_save=True).start()
    up.bias = None
    up.props["model_path"] = str(_wide_gguf(tmp_path / "m.gguf", vocab(255)))
    r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
    try:
        st = _settings(_data(tmp_path), budgets=(0,))
        out = calibrate_one(st, url=r.url, name="w", out_dir=tmp_path / "a", resume=False, log=lambda _: None,
                            therm=lambda: 100)
        assert out.status == "done", out.reason
        raw = tmp_path / "a" / "w.raw.jsonl"
        assert read_raw(raw).header["wide_coverage"] == 255
        full = json.loads(out.profile.read_text())["comparable"]["data"]
        lines = raw.read_text().splitlines(keepends=True)
        raw.write_text("".join(ln for ln in lines if f'"{WIDE_ID}/q"' not in ln))
        assert not _wide_uids(raw)
        n_req = len(up.requests)
        again = calibrate_one(st, url=r.url, name="w", out_dir=tmp_path / "a", resume=True, log=lambda _: None,
                              therm=lambda: 100)
        assert again.status == "done", again.reason
        data = json.loads(again.profile.read_text())["comparable"]["data"]
        assert data["skipped"] == full["skipped"] == {} and data["questions"] == 5
        assert _wide_uids(raw) == {f"{WIDE_ID}/q"} and len(up.requests) > n_req
    finally:
        r.finish()
        up.stop()


@pytest.mark.parametrize(("stats", "why"), [
    ({"wide_coverage": 9}, "this model's tokenizer covers 9 of callo's two-letter labels as single tokens"),
    ({"wide_coverage": 0}, "this model's tokenizer covers 0 of callo's two-letter labels as single tokens"),
    ({"wide_coverage": None}, "the server reports no wide_coverage (read as 26)"),
    ({}, "the server reports no wide_coverage (read as 26)"),
    ({"wide_coverage": 6, "wide_readable": 6}, "this model's tokenizer can read 6 of Callo's two-letter labels (the "
     "next one needs a letter that is not a single token)"),
])
def test_the_skip_line_names_the_coverage_or_says_there_is_none(stats, why):
    assert coverage_skip_line(1, 26, stats) == f"skipped 1 question(s) with more than 26 options (too_many_options): {why}"


def test_the_question_order_is_shared_by_arms_of_any_coverage(tmp_path):
    """A lower-coverage arm reads its questions in the full-coverage arm's order, minus the skipped ones, so the
    warm-up and audit pools agree; narrow-only data keeps exactly the pre-wide order."""
    items, _ = load_data(_data(tmp_path, n=12))
    kept, dropped, _ = coverage_filter(items, {"wide_coverage": 9})
    assert dropped == 1
    for seed in (0, 1, 7):
        assert shuffled(kept, seed) == [it for it in shuffled(items, seed) if it in kept]
    narrow, _ = load_data(write_data(tmp_path / "n.jsonl", 12))
    for seed in (0, 1, 7):
        old = list(narrow)
        __import__("random").Random(f"{seed}:think-order").shuffle(old)            # the pre-wide algorithm
        assert shuffled(narrow, seed) == old
