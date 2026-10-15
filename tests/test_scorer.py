# Adapted from the author's benchmark harness (tests/test_sharpquant_llamacpp_scorer.py @ f886c50f).
# Edits: imports retargeted; module docstring shortened and explanatory comments trimmed; two `sc = ...` locals inlined;
#   four tests added for id_slot, make_dist_fn and make_http_post; v0.2 tests added for the multimodal and cached
#   passes, /detokenize and parse_special; v0.3: the stub server is server_close()d (no ResourceWarning); think
#   budgets: the `generate` tests and their scripted /completion stub.
"""The concrete ForwardScorer: HTTP to a live llama.cpp server. Parsing pinned to REAL responses captured from a
running llama-server (MiniCPM5-2B-BF16, 2026-09-15). Load-bearing facts, encoded as tests:
- next-token logprobs live at completion_probabilities[0].top_logprobs as {id, token, logprob};
- they are RAW pre-sampling softmax, so the request must NOT set post_sampling_probs.
"""
import json
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event

import pytest

from callo.errors import (
    Cancelled,
    InvalidInput,
    ReadoutError,
    ThinkTimeout,
    ThinkTooLarge,
    UpstreamRejected,
    UpstreamUnavailable,
)
from callo.scorer import (
    GEN_CONNECT_TIMEOUT,
    READOUT_PIN,
    ForwardScorer,
    GenResult,
    LlamaCppForwardScorer,
    MMResult,
    make_dist_fn,
    make_http_post,
    readout_pin,
)
from callo.think import PINNED_SAMPLER, parse_sampling, resolve_sampling

TOKENIZE_RESP = {"tokens": [21742, 47]}
COMPLETION_RESP = {"completion_probabilities": [{"top_logprobs": [
    {"id": 8181, "token": " Paris", "logprob": -0.16003863513469696},
    {"id": 6918, "token": " France", "logprob": -3.8093316555023193},
    {"id": 280, "token": " a", "logprob": -3.873173475265503},
]}]}


class FakePost:
    """Records (url, payload) and returns a captured response chosen by endpoint path."""

    def __init__(self, tokenize=None, completion=None, detokenize=None):
        self.calls = []
        self._resp = {"/tokenize": tokenize, "/completion": completion, "/detokenize": detokenize}

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        for path, resp in self._resp.items():
            if url.endswith(path):
                if resp is None:
                    raise AssertionError(f"unexpected call to {path}")
                return resp
        raise AssertionError(f"no fixture for {url}")


def _scorer(**kw):
    return LlamaCppForwardScorer("http://127.0.0.1:8199", **kw)


def test_tokenize_returns_the_id_list():
    post = FakePost(tokenize=TOKENIZE_RESP)
    sc = _scorer(http_post=post)
    assert sc.tokenize("Answer:", add_special=False) == [21742, 47]
    url, payload = post.calls[0]
    assert url.endswith("/tokenize")
    assert payload["content"] == "Answer:" and payload["add_special"] is False


def test_next_logprobs_parses_top_logprobs_into_id_to_logprob():
    post = FakePost(completion=COMPLETION_RESP)
    sc = _scorer(http_post=post)
    out = sc.next_logprobs([608, 4894, 304, 6918, 357], top_k=3)
    assert out == {8181: -0.16003863513469696, 6918: -3.8093316555023193, 280: -3.873173475265503}
    assert all(v <= 0 for v in out.values())


def test_next_logprobs_request_never_enables_post_sampling_probs():
    post = FakePost(completion=COMPLETION_RESP)
    sc = _scorer(http_post=post)
    sc.next_logprobs([1, 2, 3], top_k=5)
    _, payload = post.calls[0]
    assert payload["prompt"] == [1, 2, 3]
    assert payload["n_probs"] == 5
    assert payload["n_predict"] == 1
    assert payload.get("post_sampling_probs", False) is False


def test_next_logprobs_raises_when_completion_probabilities_missing():
    post = FakePost(completion={"content": "Paris", "tokens": [8181]})
    with pytest.raises(InvalidInput):
        _scorer(http_post=post).next_logprobs([1, 2, 3], top_k=5)


def test_next_logprobs_raises_on_empty_completion_probabilities():
    post = FakePost(completion={"completion_probabilities": []})
    with pytest.raises(InvalidInput):
        _scorer(http_post=post).next_logprobs([1, 2, 3], top_k=5)


def test_base_url_trailing_slash_is_normalised():
    post = FakePost(tokenize=TOKENIZE_RESP)
    LlamaCppForwardScorer("http://127.0.0.1:8199/", http_post=post).tokenize("x")
    url, _ = post.calls[0]
    assert url == "http://127.0.0.1:8199/tokenize"


def test_next_logprobs_is_deterministic_by_default_cache_off():
    post = FakePost(completion=COMPLETION_RESP)
    _scorer(http_post=post).next_logprobs([1, 2, 3], top_k=5)
    _, payload = post.calls[0]
    assert payload["cache_prompt"] is False
    assert payload["temperature"] == 0.0


def test_cache_prompt_is_opt_in():
    post = FakePost(completion=COMPLETION_RESP)
    _scorer(http_post=post, cache_prompt=True).next_logprobs([1, 2, 3], top_k=5)
    _, payload = post.calls[0]
    assert payload["cache_prompt"] is True


def test_next_logprobs_reads_the_first_completion_position_not_the_last():
    two = {"completion_probabilities": [
        {"top_logprobs": [{"id": 1, "token": "a", "logprob": -0.1}]},
        {"top_logprobs": [{"id": 2, "token": "b", "logprob": -0.2}]}]}
    assert _scorer(http_post=FakePost(completion=two)).next_logprobs([9], top_k=1) == {1: -0.1}


def test_tokenize_defaults_to_add_special_true():
    post = FakePost(tokenize=TOKENIZE_RESP)
    _scorer(http_post=post).tokenize("x")
    _, payload = post.calls[0]
    assert payload["add_special"] is True


# ---- callo additions: id_slot, make_dist_fn, make_http_post ------------------------------------------

def test_next_logprobs_carries_id_slot_only_when_given():
    post = FakePost(completion=COMPLETION_RESP)
    sc = _scorer(http_post=post)
    sc.next_logprobs([1, 2], top_k=5)
    sc.next_logprobs([1, 2], top_k=5, id_slot=0)
    sc.next_logprobs([1, 2], top_k=5, id_slot=3)
    assert "id_slot" not in post.calls[0][1]
    assert post.calls[1][1]["id_slot"] == 0 and post.calls[2][1]["id_slot"] == 3


def test_tokenize_is_never_pinned():
    post = FakePost(tokenize=TOKENIZE_RESP)
    _scorer(http_post=post).tokenize("x")
    assert "id_slot" not in post.calls[0][1]


def test_make_dist_fn_threads_top_k_and_id_slot():
    post = FakePost(completion=COMPLETION_RESP)
    fn = make_dist_fn(_scorer(http_post=post), top_k=100, id_slot=1)
    out = fn((5, 6, 7))
    assert out == {8181: -0.16003863513469696, 6918: -3.8093316555023193, 280: -3.873173475265503}
    _, payload = post.calls[0]
    assert payload["prompt"] == [5, 6, 7] and payload["n_probs"] == 100 and payload["id_slot"] == 1


def test_make_http_post_sends_json_and_bearer_key_to_a_real_socket():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["auth"] = self.headers.get("Authorization")
            seen["ctype"] = self.headers.get("Content-Type")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/completion"
        assert make_http_post(api_key="sekrit", timeout=5)(url, {"prompt": [1]}) == {"ok": True}
        assert seen == {"auth": "Bearer sekrit", "ctype": "application/json", "body": {"prompt": [1]}}
        make_http_post(timeout=5)(url, {"prompt": [2]})
        assert seen["auth"] is None
    finally:
        srv.shutdown()
        srv.server_close()



# ---- v0.2: the multimodal pass, the cached text pass, /detokenize ---------------------------------------

MM_RESP = {**COMPLETION_RESP, "tokens_evaluated": 412,
           "timings": {"cache_n": 354, "prompt_n": 58, "prompt_ms": 15.7, "predicted_n": 1}}
DELIM = "<|im_end|>\n<|im_start|>user"


def test_next_logprobs_mm_sends_an_object_prompt_with_the_cache_on_even_when_the_scorer_default_is_off():
    post = FakePost(completion=MM_RESP)
    out = _scorer(http_post=post, cache_prompt=False).next_logprobs_mm(
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<__media__>Q", ["iVBORw0KGgo=", "R0lG"],
        top_k=100, id_slot=0, delimiter=DELIM)
    url, payload = post.calls[0]
    assert url.endswith("/completion")
    assert payload == {"prompt": {"prompt_string": "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<__media__>Q",
                                  "multimodal_data": ["iVBORw0KGgo=", "R0lG"]},
                       "n_predict": 1, "n_probs": 100, "temperature": 0.0, "cache_prompt": True, **readout_pin(),
                       "id_slot": 0, "message_delimiters": [{"role": "user", "delimiter": DELIM}]}
    assert out == MMResult({8181: -0.16003863513469696, 6918: -3.8093316555023193, 280: -3.873173475265503},
                           412, 354, 58, 15.7)


def test_next_logprobs_mm_without_a_delimiter_or_slot_sends_neither():
    post = FakePost(completion=MM_RESP)
    _scorer(http_post=post).next_logprobs_mm("p", ["QUJD"], top_k=5)
    _, payload = post.calls[0]
    assert "message_delimiters" not in payload and "id_slot" not in payload and payload["cache_prompt"] is True


def test_missing_timings_read_as_none_not_a_crash():
    post = FakePost(completion=COMPLETION_RESP)
    r = _scorer(http_post=post).next_logprobs_mm("p", ["QUJD"], top_k=5)
    assert (r.tokens_evaluated, r.cache_n, r.prompt_n, r.prompt_ms) == (None, None, None, None)
    with pytest.raises(InvalidInput):
        _scorer(http_post=FakePost(completion={"content": "x"})).next_logprobs_mm("p", ["QUJD"], top_k=5)


def test_next_logprobs_cached_sends_token_arrays_with_the_cache_and_delimiter_even_with_cache_off():
    post = FakePost(completion=MM_RESP)
    r = _scorer(http_post=post, cache_prompt=False).next_logprobs_cached([1, 2, 3], top_k=100, id_slot=1,
                                                                        delimiter=DELIM)
    _, payload = post.calls[0]
    assert payload == {"prompt": [1, 2, 3], "n_predict": 1, "n_probs": 100, "temperature": 0.0, "cache_prompt": True,
                       **readout_pin(), "id_slot": 1, "message_delimiters": [{"role": "user", "delimiter": DELIM}]}
    assert r.cache_n == 354 and r.prompt_n == 58


def test_the_plain_pass_is_the_v01_payload_plus_the_neutral_readout_pin():
    post = FakePost(completion=MM_RESP)
    _scorer(http_post=post).next_logprobs([1, 2, 3], top_k=100, id_slot=0)
    _, payload = post.calls[0]
    assert payload == {"prompt": [1, 2, 3], "n_predict": 1, "n_probs": 100, "temperature": 0.0, "cache_prompt": False,
                       **readout_pin(), "id_slot": 0}
    # the pin (llama.cpp server-schema.cpp @ 4ebdf2c): the chain is temperature 0 alone, the base logit_bias (where
    # --ignore-eos puts its EOG bans) cleared and not refilled, no dynatemp / mirostat / reasoning budget
    assert readout_pin() == {"samplers": ["temperature"], "logit_bias": [], "ignore_eos": False, "dynatemp_range": 0.0,
                             "mirostat": 0, "reasoning_budget_tokens": -1, "return_tokens": True}
    # frozen, and every payload gets its own lists
    with pytest.raises(TypeError):
        READOUT_PIN["logit_bias"] = [[1, False]]                    # type: ignore[index]
    payload["samplers"].append("penalties")
    payload["logit_bias"].append([1, False])
    _scorer(http_post=post).next_logprobs([1, 2, 3], top_k=100, id_slot=0)
    assert post.calls[1][1]["samplers"] == ["temperature"] and post.calls[1][1]["logit_bias"] == []


def _scripted(*replies):
    """A /completion stub that answers with `replies` in turn and records every payload."""
    calls, it = [], iter(replies)

    def post(url, payload):
        calls.append(payload)
        return next(it)
    return post, calls


WITHHELD = {"content": "\ufffd", "stop_type": "limit", "tokens": [13922]}
EOS_RESP = {"stop_type": "eos", "tokens": [130073], "completion_probabilities": [
    {"id": 130073, "top_logprobs": [{"id": 130073, "logprob": -0.0255}, {"id": 62, "logprob": -4.37}]}]}


def test_an_eos_reply_names_its_pick_as_the_eog_and_a_re_asked_one_never_does():
    post, _ = _scripted(EOS_RESP)
    out = _scorer(http_post=post).next_logprobs([1], top_k=5)
    assert out.eog_id == 130073 and out.reasks == 0
    post, calls = _scripted(WITHHELD, EOS_RESP)
    out = _scorer(http_post=post).next_logprobs([1], top_k=5)
    assert out.eog_id is None and out.reasks == 1 and calls[1]["logit_bias"] == [[13922, False]]
    assert {k: v for k, v in calls[1].items() if k != "logit_bias"} == {k: v for k, v in calls[0].items()
                                                                      if k != "logit_bias"}
    # a sampling pass (temperature > 0) never speaks for its argmax
    post, _ = _scripted(EOS_RESP)
    assert _scorer(http_post=post, temperature=0.7).next_logprobs([1], top_k=5).eog_id is None


def test_a_withheld_pass_that_names_no_token_is_a_readout_error():
    post, calls = _scripted({"content": "\ufffd", "stop_type": "limit"})
    with pytest.raises(ReadoutError, match="named no token"):
        _scorer(http_post=post).next_logprobs([1], top_k=5)
    assert len(calls) == 1


def test_a_ban_the_server_ignores_is_a_readout_error_not_a_loop():
    post, calls = _scripted(WITHHELD, WITHHELD)
    with pytest.raises(ReadoutError, match=r"picked 13922 again although it was banned \(logit_bias \[13922\]"):
        _scorer(http_post=post).next_logprobs([1], top_k=5)
    assert len(calls) == 2


def test_the_ban_bound_names_exactly_the_tokens_banned():
    replies = [{**WITHHELD, "tokens": [t]} for t in (11, 12, 13, 14, 15)]
    post, calls = _scripted(*replies)
    with pytest.raises(ReadoutError, match=r"with \[11, 12, 13, 14\] banned in turn, and then a new withheld pick 15"):
        _scorer(http_post=post).next_logprobs_cached([1], top_k=5)
    assert [c.get("logit_bias") for c in calls] == [[], [[11, False]], [[11, False], [12, False]],
                                                    [[11, False], [12, False], [13, False]],
                                                    [[11, False], [12, False], [13, False], [14, False]]]


def test_detokenize_sends_the_whole_sequence_once():
    post = FakePost(detokenize={"content": "**\n"})
    assert _scorer(http_post=post).detokenize((52, 198)) == "**\n"
    assert len(post.calls) == 1 and post.calls[0][1] == {"tokens": [52, 198]}
    with pytest.raises(InvalidInput):
        _scorer(http_post=FakePost(detokenize={"tokens": []})).detokenize([1])


def test_tokenize_sends_parse_special_only_when_given():
    post = FakePost(tokenize=TOKENIZE_RESP)
    sc = _scorer(http_post=post)
    sc.tokenize("x", add_special=False)
    sc.tokenize("x", add_special=False, parse_special=True)
    assert "parse_special" not in post.calls[0][1]
    assert post.calls[1][1] == {"content": "x", "add_special": False, "parse_special": True}


# ---- v0.3 think budgets: the cancellable generation ----------------------------------------------------

GEN_OK = {"content": "w1 w2", "tokens": [5001, 5002, 2001], "tokens_predicted": 3, "stop_type": "word",
          "truncated": False, "timings": {"prompt_n": 4, "cache_n": 390, "prompt_ms": 12.5, "predicted_ms": 30.0}}


class _Gen(BaseHTTPRequestHandler):
    """Scripted /completion: a status and body, an optional delay, the request recorded; records a client that left.
    `cut`: send only that many body bytes under the full Content-Length, then close (the upstream dies mid-body)."""
    status, body, delay, seen, aborted, raw_body, cut = 200, GEN_OK, 0.0, [], [], None, None

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        type(self).seen.append((self.path, dict(self.headers), json.loads(self.rfile.read(n))))
        deadline = time.monotonic() + type(self).delay
        while time.monotonic() < deadline:
            r, _, _ = select.select([self.connection], [], [], 0.02)
            if r and self.connection.recv(1, socket.MSG_PEEK) == b"":
                type(self).aborted.append(time.monotonic())
                return
        data = type(self).raw_body if type(self).raw_body is not None else json.dumps(type(self).body).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data if type(self).cut is None else data[:type(self).cut])

    def log_message(self, *a):
        pass


@pytest.fixture
def gen_server():
    _Gen.status, _Gen.body, _Gen.delay, _Gen.seen, _Gen.aborted, _Gen.raw_body = 200, GEN_OK, 0.0, [], [], None
    _Gen.cut = None
    srv = HTTPServer(("127.0.0.1", 0), _Gen)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_generate_sends_the_token_prompt_the_stop_the_full_pinned_sampler_and_the_slot(gen_server):
    sampling = resolve_sampling(parse_sampling("fixed")).pinned
    r = LlamaCppForwardScorer(gen_server, api_key="k").generate([1, 2, 3], n_predict=256, stop=["</think>"],
                                                                sampling=sampling, seed=123, id_slot=0,
                                                                delimiter="<|im_start|>user", timeout=5)
    assert isinstance(r, GenResult) and r.tokens == (5001, 5002, 2001) and r.close == "natural"
    assert (r.tokens_predicted, r.truncated, r.prompt_ms, r.predicted_ms, r.prompt_n, r.cache_n) == (3, False, 12.5, 30.0, 4, 390)
    assert r.wall_ms >= 0 and r.content == "w1 w2"
    path, headers, payload = _Gen.seen[-1]
    assert path == "/completion" and headers["Authorization"] == "Bearer k"
    assert payload["prompt"] == [1, 2, 3] and payload["n_predict"] == 256 and payload["seed"] == 123
    assert payload["stop"] == ["</think>"] and payload["preserved_tokens"] == ["</think>"]
    assert payload["return_tokens"] is True and payload["cache_prompt"] is True and payload["id_slot"] == 0
    assert payload["message_delimiters"] == [{"role": "user", "delimiter": "<|im_start|>user"}]
    for k, v in PINNED_SAMPLER.items():
        assert payload[k] == v, k
    assert payload["temperature"] == 0.6 and "post_sampling_probs" in payload


def test_generate_never_lets_the_sampling_dict_override_the_generation_s_own_keys(gen_server):
    sampling = {**resolve_sampling(parse_sampling("fixed")).pinned, "n_predict": 3, "prompt": [9], "stop": ["x"],
                "seed": 7, "id_slot": 3, "cache_prompt": False, "grammar": "root ::= [a-z]+"}
    LlamaCppForwardScorer(gen_server).generate([1, 2, 3], n_predict=256, stop=["</think>"], sampling=sampling,
                                               seed=123, id_slot=0, timeout=5)
    payload = _Gen.seen[-1][2]
    assert payload["n_predict"] == 256 and payload["prompt"] == [1, 2, 3] and payload["stop"] == ["</think>"]
    assert payload["seed"] == 123 and payload["id_slot"] == 0 and payload["cache_prompt"] is True
    assert "grammar" not in payload and payload["temperature"] == 0.6      # only SAMPLER_FIELDS pass through


def test_generate_maps_the_upstream_statuses(gen_server):
    sc = LlamaCppForwardScorer(gen_server)
    kw = {"n_predict": 8, "stop": ["</think>"], "sampling": {}, "seed": 1, "timeout": 5}
    _Gen.status, _Gen.body = 400, {"error": {"code": 400, "type": "exceed_context_size_error", "message": "too long"}}
    with pytest.raises(ThinkTooLarge, match="too long"):
        sc.generate([1], **kw)
    _Gen.status, _Gen.body = 400, {"error": {"code": 400, "type": "invalid_request_error", "message": "bad"}}
    with pytest.raises(UpstreamRejected, match="HTTP 400"):
        sc.generate([1], **kw)
    _Gen.status, _Gen.body = 503, {"error": {"code": 503, "message": "Loading model"}}
    with pytest.raises(UpstreamUnavailable, match="HTTP 503"):
        sc.generate([1], **kw)
    _Gen.status, _Gen.body = 200, {"content": "x"}                              # no tokens
    with pytest.raises(ReadoutError, match="no content or tokens"):
        sc.generate([1], **kw)
    _Gen.raw_body = b"{not json"
    with pytest.raises(UpstreamUnavailable, match="unreadable"):
        sc.generate([1], **kw)


def test_generate_times_out_as_think_timeout_and_a_dead_upstream_is_unavailable(gen_server):
    _Gen.delay = 2.0
    with pytest.raises(ThinkTimeout, match="exceeded its 0.3 s timeout"):
        LlamaCppForwardScorer(gen_server).generate([1], n_predict=8, stop=[], sampling={}, seed=1, timeout=0.3)
    with pytest.raises(UpstreamUnavailable):
        LlamaCppForwardScorer("http://127.0.0.1:1").generate([1], n_predict=8, stop=[], sampling={}, seed=1, timeout=1)


def test_generate_cancel_shuts_the_socket_and_the_upstream_sees_the_client_leave(gen_server):
    _Gen.delay = 5.0
    cancel = Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        LlamaCppForwardScorer(gen_server).generate([1], n_predict=8, stop=[], sampling={}, seed=1, timeout=10,
                                                   cancel=cancel)
    assert time.monotonic() - t0 < 3
    deadline = time.monotonic() + 2
    while not _Gen.aborted and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _Gen.aborted                                       # the server noticed the closed connection


def test_generate_is_on_the_forward_scorer_protocol():
    assert "generate" in ForwardScorer.__dict__ and callable(LlamaCppForwardScorer.generate)


def test_generate_bounds_the_connect_on_its_own_and_an_unreachable_upstream_is_unavailable(monkeypatch):
    seen = []

    def blackholed(address, timeout=None, *a, **k):
        seen.append(timeout)
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "create_connection", blackholed)
    with pytest.raises(UpstreamUnavailable, match="unavailable"):        # a 503, not ThinkTimeout's 504
        LlamaCppForwardScorer("http://10.255.255.1:8080").generate([1], n_predict=4096, stop=[], sampling={}, seed=1,
                                                                   timeout=1200.0)
    assert seen == [GEN_CONNECT_TIMEOUT]                                 # never the generation's 1200 s
    seen.clear()
    with pytest.raises(UpstreamUnavailable):
        LlamaCppForwardScorer("http://10.255.255.1:8080").generate([1], n_predict=8, stop=[], sampling={}, seed=1,
                                                                   timeout=2.0)
    assert seen == [2.0]                                                 # a shorter call timeout bounds it too


def test_generate_with_cancel_already_set_sends_nothing(gen_server):
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        LlamaCppForwardScorer(gen_server).generate([1], n_predict=8, stop=[], sampling={}, seed=1, timeout=5,
                                                   cancel=cancel)
    assert _Gen.seen == []


def test_generate_reads_the_closes_and_refuses_a_limit_short_of_n_predict_or_an_unknown_stop(gen_server):
    sc = LlamaCppForwardScorer(gen_server)
    kw = {"stop": ["</think>"], "sampling": {}, "seed": 1, "timeout": 5}
    _Gen.body = {**GEN_OK, "stop_type": "limit"}
    assert sc.generate([1], n_predict=3, **kw).close == "forced"         # the budget closed it
    with pytest.raises(UpstreamRejected, match="3 of 8 tokens"):         # a server time limit: not the budget's trace
        sc.generate([1], n_predict=8, **kw)
    _Gen.body = {**GEN_OK, "stop_type": "limit", "truncated": True}       # the context filled: the caller's 413
    assert sc.generate([1], n_predict=8, **kw).truncated
    _Gen.body = {**GEN_OK, "stop_type": "eos"}
    assert sc.generate([1], n_predict=8, **kw).close == "eos"
    _Gen.body = {**GEN_OK, "stop_type": "none"}
    with pytest.raises(ReadoutError, match="stop_type 'none'"):
        sc.generate([1], n_predict=8, **kw)


def test_generate_refuses_a_token_that_is_not_an_integer(gen_server):
    sc = LlamaCppForwardScorer(gen_server)
    for bad in ("5001", 5001.5, True, None):
        _Gen.body = {**GEN_OK, "tokens": [5001, bad, 2001]}
        with pytest.raises(ReadoutError, match="not an integer"):
            sc.generate([1], n_predict=8, stop=[], sampling={}, seed=1, timeout=5)
    _Gen.body = {**GEN_OK, "tokens_predicted": "3"}                      # a bad count falls back to len(tokens)
    assert sc.generate([1], n_predict=8, stop=[], sampling={}, seed=1, timeout=5).tokens_predicted == 3


def test_generate_maps_the_context_text_fallback_a_cut_body_and_a_non_object_reply(gen_server):
    sc = LlamaCppForwardScorer(gen_server)
    kw = {"n_predict": 8, "stop": [], "sampling": {}, "seed": 1, "timeout": 5}
    _Gen.status, _Gen.body = 400, {"error": {"code": 400, "message": "the request exceeds the available context size, "
                                                                     "try increasing it"}}       # no error type
    with pytest.raises(ThinkTooLarge, match="exceeds the available context size"):
        sc.generate([1], **kw)
    _Gen.status, _Gen.body, _Gen.cut = 200, GEN_OK, 10                   # the upstream dies mid-body
    with pytest.raises(UpstreamUnavailable):
        sc.generate([1], **kw)
    _Gen.cut, _Gen.raw_body = None, b"[1, 2]"
    with pytest.raises(ReadoutError, match="not an object"):
        sc.generate([1], **kw)
