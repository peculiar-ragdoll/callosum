"""v0.3 cache delimiters: under `--cache` the plain (v0.1) readout sends the user-turn delimiter the image and context
passes send, so a hybrid model checkpoints at the user turn and reuses the system head. With `--no-cache` (the
default before 0.3.0; now `--cache` is on unless turned off) the plain payload is v0.1's, byte for byte, plus the
neutral readout sampler pin (`READOUT_PIN`, every readout pass since the EOG fix) right after `cache_prompt`. Fakes
only."""
import dataclasses
import json

import pytest
from fake_callo import CalloStub
from fake_upstream import MEDIA_MARKER, FakeUpstream, PlantedBias
from helpers import TOKENS, Running, get_json, post_json, tiny_gguf
from test_calibrate_bias import PRIOR3, _write_set, set_text
from test_calibrate_think import SAMPLING, STATS, synthetic_raw, write_data
from test_compare import profile as cprofile
from test_profile import KEY, profile_dict

from callo.calibrate_think import build_profile, load_data
from callo.cli import build_parser, main, split_passthrough
from callo.compare import flags
from callo.engine import Engine, SurfaceSets
from callo.menu import Entry
from callo.profile import PROVENANCE_FIELDS, check_profile, load_profile, provenance
from callo.prompt import TemplateClient
from callo.scorer import LlamaCppForwardScorer, make_dist_fn, make_http_post, readout_pin

DELIM = "<|im_end|>\n<|im_start|>user"
COMPLETION = {"completion_probabilities": [{"top_logprobs": [{"id": 1, "token": "a", "logprob": -0.1}]}]}
PNG = "iVBORw0KGgo="
IMG = f"data:image/png;base64,{PNG}"
ONE_Q = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}}
TWO_Q = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}},
                                     "r": {"type": "choice", "criteria": {"x": None, "y": None, "z": None}}}}
IMG_Q = {**ONE_Q, "images": [IMG]}


class Post:
    def __init__(self):
        self.calls = []

    def __call__(self, url, payload):
        self.calls.append(payload)
        return COMPLETION


def _v01(prompt, *, top_k, id_slot=None):
    """v0.1's plain payload, as the pre-change next_logprobs built it (key order included), plus the readout pin."""
    p = {"prompt": list(prompt), "n_predict": 1, "n_probs": top_k, "temperature": 0.0, "cache_prompt": False,
         **readout_pin()}
    if id_slot is not None:
        p["id_slot"] = id_slot
    return p


# ---- the scorer ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("id_slot", [None, 0, 3])
@pytest.mark.parametrize("delimiter", [None, DELIM])
def test_cache_off_the_plain_payload_is_v01_byte_for_byte(id_slot, delimiter):
    post = Post()
    LlamaCppForwardScorer("http://h", http_post=post).next_logprobs([5, 6, 7], top_k=100, id_slot=id_slot,
                                                                   delimiter=delimiter)
    assert json.dumps(post.calls[0]) == json.dumps(_v01([5, 6, 7], top_k=100, id_slot=id_slot))


def test_cache_on_the_plain_payload_carries_the_delimiter_as_the_cached_passes_do():
    post = Post()
    sc = LlamaCppForwardScorer("http://h", http_post=post, cache_prompt=True)
    sc.next_logprobs([5, 6, 7], top_k=100, id_slot=1, delimiter=DELIM)
    sc.next_logprobs_cached([5, 6, 7], top_k=100, id_slot=1, delimiter=DELIM)
    assert post.calls[0] == post.calls[1] == {
        "prompt": [5, 6, 7], "n_predict": 1, "n_probs": 100, "temperature": 0.0, "cache_prompt": True, **readout_pin(),
        "id_slot": 1, "message_delimiters": [{"role": "user", "delimiter": DELIM}]}


@pytest.mark.parametrize("delimiter", [None, ""])
def test_cache_on_without_a_delimiter_degrades_to_the_pre_change_cached_payload(delimiter):
    post = Post()
    LlamaCppForwardScorer("http://h", http_post=post, cache_prompt=True).next_logprobs([5], top_k=4, delimiter=delimiter)
    assert json.dumps(post.calls[0]) == json.dumps({"prompt": [5], "n_predict": 1, "n_probs": 4, "temperature": 0.0,
                                                    "cache_prompt": True, **readout_pin()})


def test_make_dist_fn_threads_the_delimiter():
    post = Post()
    make_dist_fn(LlamaCppForwardScorer("http://h", http_post=post, cache_prompt=True), top_k=8, id_slot=0,
                 delimiter=DELIM)((1, 2))
    assert post.calls[0]["message_delimiters"] == [{"role": "user", "delimiter": DELIM}]


# ---- the engine: the validated delimiter is the one the plain pass sends -------------------------------------------

def _engine(up, *, cache, delimiter, vision=False):
    post = make_http_post(timeout=10)
    return Engine(LlamaCppForwardScorer(up.url, http_post=post, cache_prompt=cache),
                  TemplateClient(up.url, http_post=post), SurfaceSets(TOKENS, "gpt2"), vision=vision,
                  media_marker=MEDIA_MARKER if vision else None, user_delimiter=delimiter)


def _readouts(up, start):
    """The /completion bodies recorded since `start`: every readout pass (n_probs set)."""
    return [r.json for r in up.requests[start:] if r.path == "/completion" and r.json.get("n_probs")]


@pytest.mark.parametrize("body", [ONE_Q, TWO_Q], ids=["one", "multi"])
def test_engine_plain_readouts_send_the_engines_delimiter_only_under_cache(body):
    up = FakeUpstream().start()
    try:
        for cache in (False, True):
            for delimiter in (None, DELIM):
                start = len(up.requests)
                _engine(up, cache=cache, delimiter=delimiter).answer_request(body, id_slot=0)
                comps = _readouts(up, start)
                assert len(comps) >= len(body["questions"])
                for c in comps:
                    assert isinstance(c["prompt"], list) and c["id_slot"] == 0 and c["cache_prompt"] is cache
                    if not cache:
                        assert json.dumps(c) == json.dumps(_v01(c["prompt"], top_k=c["n_probs"], id_slot=0))
                    elif delimiter:
                        assert c["message_delimiters"] == [{"role": "user", "delimiter": delimiter}]
                    else:                                       # none derived: the pre-change cached payload
                        assert "message_delimiters" not in c
    finally:
        up.stop()


def test_engine_image_readouts_are_unchanged_by_cache():
    up = FakeUpstream(vision=True).start()
    try:
        bodies = []
        for cache in (False, True):
            start = len(up.requests)
            _engine(up, cache=cache, delimiter=DELIM, vision=True).answer_request(IMG_Q, id_slot=0)
            bodies.append(_readouts(up, start))
        assert bodies[0] == bodies[1] and bodies[0]
        for c in bodies[0]:
            assert c["cache_prompt"] is True and c["message_delimiters"] == [{"role": "user", "delimiter": DELIM}]
            assert c["prompt"]["multimodal_data"] == [PNG]
    finally:
        up.stop()


# ---- callo serve: the startup-validated delimiter rides on plain readouts under --cache only -----------------------

@pytest.mark.parametrize("cache", [False, True], ids=["cache-off", "cache-on"])
def test_serve_plain_and_image_readouts(tmp_path, monkeypatch, cache):
    monkeypatch.setenv("HOME", str(tmp_path))
    up = FakeUpstream(vision=True).start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"),
                     "--cache" if cache else "--no-cache"])          # on by default since 0.3.0
        try:
            _, _, stats = get_json(f"{r.url}/callo/stats")
            delim = stats["user_delimiter"]
            assert delim == DELIM and stats["cache_prompt"]["plain"] is cache
            assert stats["provenance"]["cache_prompt"] is cache
            assert stats["provenance"].get("cache_delimiter") is (True if cache else None)
            for body in (ONE_Q, TWO_Q):
                start = len(up.requests)
                status, _, out = post_json(f"{r.url}/v1/systemone", body)
                assert status == 200 and set(out["answers"]) == set(body["questions"])
                comps = _readouts(up, start)
                assert len(comps) >= len(body["questions"])
                for c in comps:
                    if cache:
                        assert c["cache_prompt"] is True
                        assert c["message_delimiters"] == [{"role": "user", "delimiter": delim}]
                    else:
                        assert json.dumps(c) == json.dumps(_v01(c["prompt"], top_k=c["n_probs"],
                                                                id_slot=c["id_slot"]))
            start = len(up.requests)
            status, _, _ = post_json(f"{r.url}/v1/systemone", IMG_Q)
            assert status == 200
            for c in _readouts(up, start):
                assert c["cache_prompt"] is True and c["message_delimiters"] == [{"role": "user", "delimiter": delim}]
        finally:
            assert r.finish() == 0
    finally:
        up.stop()


# ---- calibration under --no-cache is unchanged: v0.1 plain payloads and cache_prompt False in the profile -------------

def test_calibration_serve_argv_has_no_cache_unless_the_run_asks(tmp_path):
    e = Entry(name="m", model="/m.gguf")
    off, _ = split_passthrough(e.serve_argv(cache=False, cache_dir=tmp_path))
    on, _ = split_passthrough(e.serve_argv(cache=True, cache_dir=tmp_path))
    assert build_parser().parse_args(off).cache is False and build_parser().parse_args(on).cache is True


def test_calibrate_bias_readouts_and_profile_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    up = FakeUpstream(bias=PlantedBias(priors={3: PRIOR3}), slot_save=True).start()
    try:
        up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"), "--no-cache"])
        try:
            assert get_json(f"{r.url}/callo/stats")[2]["user_delimiter"]       # a delimiter exists, and is not sent
            start = len(up.requests)
            data = _write_set(tmp_path, set_text(ns=(2, 3), per=24))
            assert main(["calibrate", "--bias", "--url", r.url, "--data", str(data), "--out",
                         str(tmp_path / "bias.json")]) == 0
        finally:
            assert r.finish() == 0
        comps = _readouts(up, start)
        assert comps
        for c in comps:
            assert json.dumps(c) == json.dumps(_v01(c["prompt"], top_k=c["n_probs"], id_slot=c.get("id_slot")))
        assert json.loads((tmp_path / "bias.json").read_text())["provenance"]["cache_prompt"] is False
    finally:
        up.stop()


# ---- provenance: `cache_delimiter`, present only under --cache with a delimiter --------------------------------------

def _prov(**kw):
    base = {"chat_template": "{{ m }}", "build_info": "b1", "mmproj": None, "user_delimiter": DELIM,
            "cache_prompt": False}
    return provenance(**{**base, **kw})


def _old(prov):
    """The same provenance as callo wrote it before cache delimiters: no `cache_delimiter` key."""
    return {k: v for k, v in prov.items() if k != "cache_delimiter"}


def test_provenance_marks_delimited_cache_readouts_only():
    assert _prov(cache_prompt=True)["cache_delimiter"] is True
    assert "cache_delimiter" not in _prov(cache_prompt=True, user_delimiter=None)
    off = _prov()
    assert "cache_delimiter" not in off and off == _old(off)          # cache off: the pre-change provenance exactly
    assert "cache_delimiter" in PROVENANCE_FIELDS


def test_an_old_cache_profile_warns_and_cache_off_profiles_old_or_new_do_not():
    now_cached, now_off = _prov(cache_prompt=True), _prov()
    old_cached = load_profile(profile_dict(prov=_old(now_cached)))
    lines = check_profile(old_cached, model_key=KEY, native=False, provenance=now_cached)
    assert lines == ["WARNING: --position-bias: cache_delimiter differs from the profile's (profile None, now True); "
                     "the prior may not transfer"]
    assert check_profile(load_profile(profile_dict(prov=now_cached)), model_key=KEY, native=False,
                         provenance=now_cached) == []
    for prof in (_old(now_off), now_off):                             # cache off, before and after the change
        assert check_profile(load_profile(profile_dict(prov=prof)), model_key=KEY, native=False,
                             provenance=now_off) == []


def test_compare_flags_an_old_cache_think_profile_and_not_cache_off_ones():
    old = cprofile("alpha", {0: 5}, {0: 300.0}, comparable__readout__cache_prompt=True)
    new = cprofile("beta", {0: 5}, {0: 300.0}, key="k2", comparable__readout__cache_prompt=True,
                   comparable__readout__cache_delimiter=True)
    f = [x for x in flags([old, new]) if x.startswith("different --cache readouts")]
    assert f == ["different --cache readouts (whether they send the user-turn delimiter): alpha undelimited, beta "
                 "delimited"]
    off_a, off_b = cprofile("alpha", {0: 5}, {0: 300.0}), cprofile("beta", {0: 5}, {0: 300.0}, key="k2")
    assert not [x for x in flags([off_a, off_b]) if "--cache" in x]
    # cache off vs a new --cache profile: one flag (the --cache one), not the before/after one as well
    mixed = [x for x in flags([off_a, new]) if "--cache" in x]
    assert mixed == ["different --cache (budget 0's latency depends on it): alpha False, beta True"]


def test_a_think_profile_records_cache_delimiter_in_comparable_readout(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    for cached in (False, True):
        raw, settings = synthetic_raw(items)
        prov = {**raw.header["provenance"], "cache_prompt": cached, **({"cache_delimiter": True} if cached else {})}
        raw.header["provenance"] = prov
        prof = build_profile(items, raw, dataclasses.replace(settings, cache=cached), sampling=SAMPLING, stats=STATS,
                             name="fake", started="2026-10-05T10:00:00Z", ended="2026-10-05T10:05:00Z")
        readout = prof["comparable"]["readout"]
        assert readout.get("cache_delimiter") is (True if cached else None) and readout["cache_prompt"] is cached
        assert ("cache_delimiter" in readout) is cached               # cache off: the pre-change readout block


def test_a_bias_resume_across_the_change_refuses(tmp_path, capsys):
    """A raw file collected by a pre-change callo under --cache (no cache_delimiter) cannot be resumed against a
    post-change one: the provenance differs, so the two runs' readouts are never mixed."""
    data = _write_set(tmp_path, set_text(ns=(3,), per=60))
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    old = {**stub.stats["provenance"], "user_delimiter": DELIM, "cache_prompt": True}
    stub.stats.update(provenance=old, user_delimiter=DELIM)
    stub.stats["cache_prompt"]["plain"] = True
    base = ["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out", str(tmp_path / "b.json")]
    try:
        stub.fail_after = (30, 400)
        assert main(base) == 1
        stub.fail_after = None
        stub.stats["provenance"] = {**old, "cache_delimiter": True}
        posts = len(stub.posts)
        capsys.readouterr()
        assert main([*base, "--resume"]) == 2
        err = capsys.readouterr().err
        assert "refusing to resume" in err and "provenance" in err and len(stub.posts) == posts
        stub.stats["provenance"] = old                              # the same callo as the raw file: resumes
        assert main([*base, "--resume"]) == 0
    finally:
        stub.stop()
