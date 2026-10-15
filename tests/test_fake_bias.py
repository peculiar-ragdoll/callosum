"""The v0.3 fakes: the planted-prior fake upstream through callo's real engine, and
the CalloStub that stands in for `callo serve` in the calibrate tests."""
import json
import math
import time
import urllib.error
import urllib.request

import pytest
from fake_callo import CalloStub, default_stats
from fake_upstream import FakeUpstream, PlantedBias, parse_options
from test_engine import _write_tiny_gguf

from callo.engine import local_engine
from callo.profile import provenance
from callo.prompt import messages
from callo.systemone import parse_request

BODY = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"red": None, "green": None, "blue": None,
                                                                        "cyan": None}}}}


def test_parse_options_reads_the_last_options_block():
    q = parse_request(BODY).questions[0]
    text = "".join(m["content"] for m in messages("s", q))
    assert parse_options(text) == ["red", "green", "blue", "cyan"] and parse_options("no options here") is None


def test_parse_options_refuses_a_block_that_does_not_match_the_reply_line():
    """An option text with a newline ends the parsed block early: planting over 1 option where the prompt asked about
    2 would silently fit the wrong n, so the fake refuses."""
    q = parse_request({"state": "s", "questions": {"q": {"type": "choice", "criteria": {
        "a": "first line\nsecond line", "b": "other"}}}}).questions[0]
    text = "".join(m["content"] for m in messages("s", q))
    with pytest.raises(ValueError, match="parsed 1 option.* lists 2"):
        parse_options(text)
    with pytest.raises(ValueError, match="no 'Reply with one letter' line"):
        parse_options("Options:\nA. x\nB. y\n")


def test_planted_mode_refuses_an_object_prompt_instead_of_answering_canned_logprobs():
    up = FakeUpstream(bias=PlantedBias(priors={2: [0.9, 0.1]}), vision=True).start()
    try:
        prompt = {"prompt_string": "Options:\nA. x\nB. y\n\nReply with one letter: A, B.",
                  "multimodal_data": ["aGk="]}
        status, _, out = _post(f"{up.url}/completion", {"prompt": prompt, "n_probs": 100})
        assert status == 500 and "planted mode is token-array only" in out["error"]["message"]
    finally:
        up.stop()


def test_the_dump_is_a_sub_normalised_top_k_list():
    pb = PlantedBias(priors={3: [0.5, 0.3, 0.2]})
    d = pb.dump(["x1", "x2", "x3"])
    assert [e["logprob"] for e in d] == sorted((e["logprob"] for e in d), reverse=True)
    assert math.isclose(sum(math.exp(e["logprob"]) for e in d), 1.0, rel_tol=1e-12)
    assert len(PlantedBias(top_k=2).dump(["x1", "x2", "x3"])) == 2
    p = pb.probs(["x1", "x2", "x3"])
    z = [pb.score(t) + math.log(w) for t, w in zip(["x1", "x2", "x3"], [0.5, 0.3, 0.2], strict=True)]
    assert p == pytest.approx([math.exp(v) / sum(math.exp(u) for u in z) for v in z], abs=1e-15)
    assert PlantedBias(content={"x1": 9.0}).score("x1") == 9.0 and abs(PlantedBias().score("x1")) <= 1.5


def test_callos_engine_reads_the_planted_probabilities_from_the_fake(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    g = tmp_path / "m.gguf"
    _write_tiny_gguf(g)
    pb = PlantedBias(priors={4: [0.4, 0.3, 0.2, 0.1]})
    up = FakeUpstream(bias=pb).start()
    try:
        eng = local_engine(up.url, gguf=g)
        q = parse_request(BODY).questions[0]
        r = eng.answer("s", q)
        assert r.probs == pytest.approx(pb.probs(list(q.texts)), abs=1e-12) and r.topk_miss == 0
        up.bias = PlantedBias(priors={4: [0.4, 0.3, 0.2, 0.1]}, top_k=2)
        assert eng.answer("s", q).topk_miss == 2                              # two options floored by the cut
    finally:
        up.stop()


def _post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, json.loads(e.read())


def test_the_callo_stub_answers_like_callo_and_counts_like_callo():
    stub = CalloStub(PlantedBias(priors={4: [0.4, 0.3, 0.2, 0.1]}, top_k=3))
    try:
        status, h, out = _post(f"{stub.url}/v1/systemone", BODY, {"X-Callo-Conversation": "callo-calibrate"})
        assert status == 200 and h["x-callo-slot"] == "0" and set(out["answers"]["q"]["probabilities"]) == {
            "red", "green", "blue", "cyan"}
        assert json.loads(h["x-callo-timings"])[0]["miss"] == 1 and stub.lanes == ["callo-calibrate"]
        with urllib.request.urlopen(f"{stub.url}/callo/stats", timeout=10) as r:
            stats = json.loads(r.read())
        assert stats["s1"]["readouts"] == 1 and stats["routes"] == {"/v1/systemone": 1, "/callo/stats": 1}
        assert stats["s1"]["n"] == 1 and stats["s1"]["topk_miss"] == 1
        stub.script = [503]
        assert _post(f"{stub.url}/v1/systemone", BODY)[0] == 503
        stub.slot_after = (1, "2")
        assert _post(f"{stub.url}/v1/systemone", BODY)[1]["x-callo-slot"] == "2"
    finally:
        stub.stop()


def test_the_callo_stub_stops_promptly():
    """serve_forever polls its shutdown flag every poll_interval: 0.5 s by default, paid by every calibrate test."""
    stub = CalloStub()
    t0 = time.monotonic()
    stub.stop()
    assert time.monotonic() - t0 < 0.25


def test_the_stub_stats_have_the_shape_callo_serves():
    stats = default_stats()
    assert set(stats["provenance"]) == set(provenance(chat_template=None, build_info=None, mmproj=None,
                                                      user_delimiter=None, cache_prompt=False))
    assert {"instance", "model_key", "model_file", "provenance", "position_bias"} <= set(stats)
    assert {"readouts", "topk_miss"} <= set(stats["s1"]) and {"residue", "erase"} <= set(stats["probe"])
