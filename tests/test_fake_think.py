"""The fake upstream's thinking mode: the templates, the tokenizer's fixed ids, the
deterministic generation, the slot cache, the planted curve and the abort."""
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest
from fake_upstream import (
    THINK_EOG_ID,
    THINK_NL_ID,
    THINK_PARA_ID,
    THINK_TAG2_ID,
    THINK_TAG_ID,
    THINK_TEMPLATES,
    THINK_WORD_BASE,
    FakeUpstream,
    PlantedBias,
    planted_answer,
    planted_gain,
    think_tokenize,
)

from callo.prompt import messages
from callo.systemone import parse_request

Q = parse_request({"state": "the state", "questions": {"q": {"type": "choice", "instructions": "pick",
                                                              "criteria": {"a": "one", "b": "two", "c": "three"}}}}).questions[0]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=5).start()
    yield u
    u.stop()


# What callo sends (scorer.generate / next_logprobs_cached): a generation asks for its tokens and stops at the tag; a
# readout asks for the top logprobs. The fake tells them apart by n_probs, as the server's reply does.
GEN = {"n_predict": 64, "stop": ["</think>"], "return_tokens": True, "cache_prompt": True, "n_probs": 0}
READ = {"n_predict": 1, "n_probs": 10, "cache_prompt": True}


def _render(up, thinking, msgs=None):
    return post(f"{up.url}/apply-template", {"messages": msgs or messages("the state", Q),
                                             "chat_template_kwargs": {"enable_thinking": thinking}})[1]["prompt"]


def _on_tokens(up):
    return post(f"{up.url}/tokenize", {"content": _render(up, True), "add_special": True, "parse_special": True})[1]["tokens"]


def test_the_templates_honour_enable_thinking_and_the_default_fake_ignores_it():
    for name, (on_tail, off_tail) in THINK_TEMPLATES.items():
        u = FakeUpstream(think=name).start()
        try:
            assert _render(u, True).endswith(on_tail) and _render(u, False).endswith(off_tail)
        finally:
            u.stop()
    u = FakeUpstream().start()
    try:
        assert _render(u, True) == _render(u, False)
    finally:
        u.stop()
    with pytest.raises(ValueError, match="think must be one of"):
        FakeUpstream(think="nope")


def test_the_tokenizer_keeps_the_tag_and_newlines_as_fixed_ids():
    assert think_tokenize("\n</think>\n\n") == [THINK_NL_ID, THINK_TAG_ID, THINK_PARA_ID]
    assert think_tokenize("\n\n") == [THINK_PARA_ID] and think_tokenize("</think>") == [THINK_TAG_ID]
    assert think_tokenize("a b\nc") == [100, 101, THINK_NL_ID, 102]


def test_a_trailing_assistant_message_renders_as_a_prefill_with_its_reasoning(up):
    msgs = messages("the state", Q) + [{"role": "assistant", "content": "A", "reasoning_content": " w1 w2 "}]
    assert _render(up, True, msgs).endswith("<|im_start|>assistant\n<think>\nw1 w2\n</think>\n\nA")


def test_a_generation_is_deterministic_in_seed_and_prompt_and_closes_at_the_planted_length(up):
    on = _on_tokens(up)
    body = {**GEN, "prompt": on, "seed": 7, "id_slot": 0}
    _, a = post(f"{up.url}/completion", body)
    _, b = post(f"{up.url}/completion", body)
    assert a["tokens"] == b["tokens"] and a["stop_type"] == "word" and a["tokens_predicted"] == 6
    assert a["tokens"][-1] == THINK_TAG_ID and all(THINK_WORD_BASE <= t < THINK_WORD_BASE + 100 for t in a["tokens"][:-1])
    assert a["content"].split() == [f"w{t - THINK_WORD_BASE}" for t in a["tokens"][:-1]]
    assert (a["tokens_evaluated"], a["tokens_cached"]) == (len(on), len(on) + 5)    # the tag was never decoded
    assert "completion_probabilities" not in a
    _, c = post(f"{up.url}/completion", {**body, "seed": 8})
    assert c["tokens"] != a["tokens"]
    _, forced = post(f"{up.url}/completion", {**body, "n_predict": 3})
    assert forced["stop_type"] == "limit" and forced["tokens"] == a["tokens"][:3]         # a prefix of the long one
    assert a["timings"]["predicted_ms"] == 60.0
    assert a["timings"]["cache_n"] == 0 and a["timings"]["prompt_n"] == len(on)           # a cold slot


def test_a_budget_of_1_is_a_generation_and_tokens_come_only_with_return_tokens(up):
    on = _on_tokens(up)
    _, a = post(f"{up.url}/completion", {**GEN, "prompt": on, "seed": 7, "id_slot": 0})
    _, one = post(f"{up.url}/completion", {**GEN, "prompt": on, "seed": 7, "id_slot": 1, "n_predict": 1})
    assert one["tokens"] == a["tokens"][:1] and one["stop_type"] == "limit" and one["tokens_predicted"] == 1
    _, bare = post(f"{up.url}/completion", {**GEN, "prompt": on, "seed": 7, "id_slot": 2, "return_tokens": False})
    assert bare["tokens"] == [] and bare["tokens_predicted"] == 6 and bare["content"] == a["content"]


def test_without_the_tag_in_stop_the_generation_runs_past_the_tag_to_eos(up):
    on = _on_tokens(up)
    _, a = post(f"{up.url}/completion", {**GEN, "prompt": on, "seed": 7, "id_slot": 0})
    _, r = post(f"{up.url}/completion", {**GEN, "prompt": on, "seed": 7, "id_slot": 1, "stop": []})
    assert r["tokens"] == a["tokens"] + [THINK_PARA_ID, THINK_EOG_ID] and r["stop_type"] == "eos"
    assert r["content"] == a["content"] + "</think>\n\n"


def test_eos_and_multi_token_tag_modes(up):
    on = _on_tokens(up)
    body = {**GEN, "prompt": on, "seed": 1, "id_slot": 0}
    up.think_eos = True
    _, r = post(f"{up.url}/completion", body)
    assert r["stop_type"] == "eos" and r["tokens"][-1] == THINK_EOG_ID
    up.think_eos, up.think_multi_tag = False, True
    _, r = post(f"{up.url}/completion", body)
    assert r["tokens"][-2:] == [THINK_TAG_ID, THINK_TAG2_ID] and r["tokens_predicted"] == 7 and r["stop_type"] == "word"


def test_the_slot_cache_gives_2_tokens_after_a_natural_close_and_4_after_a_forced_one(up):
    on = _on_tokens(up)
    _, g = post(f"{up.url}/completion", {**GEN, "prompt": on, "seed": 1, "id_slot": 0})
    _, r = post(f"{up.url}/completion", {**READ, "prompt": on + g["tokens"] + [THINK_PARA_ID], "id_slot": 0})
    assert (r["timings"]["cache_n"], r["timings"]["prompt_n"]) == (len(on) + 5, 2)
    _, g2 = post(f"{up.url}/completion", {**GEN, "prompt": on, "n_predict": 3, "seed": 1, "id_slot": 1})
    close = [THINK_NL_ID, THINK_TAG_ID, THINK_PARA_ID]
    _, r2 = post(f"{up.url}/completion", {**READ, "prompt": on + g2["tokens"] + close, "id_slot": 1})
    assert (r2["timings"]["cache_n"], r2["timings"]["prompt_n"]) == (len(on) + 2, 4)
    # a cut at k=2 of the 6-token trace on slot 0: pure attention reuses the prefix (3 new tokens)...
    _, cut = post(f"{up.url}/completion", {**READ, "prompt": on + g["tokens"][:2] + close, "id_slot": 0})
    assert cut["timings"]["prompt_n"] == 3 and cut["timings"]["prompt_ms"] == 1.5
    # ...and a hybrid re-prefills the trace from the prompt end
    up.think_hybrid = True
    up.cache[0] = on + g["tokens"][:-1]
    _, cut = post(f"{up.url}/completion", {**READ, "prompt": on + g["tokens"][:2] + close, "id_slot": 0})
    assert cut["timings"]["cache_n"] == len(on) and cut["timings"]["prompt_n"] == 5


def test_cache_prompt_false_reuses_nothing_but_the_slot_still_holds_the_prompt(up):
    on = _on_tokens(up)
    _, g = post(f"{up.url}/completion", {**GEN, "prompt": on, "seed": 1, "id_slot": 0})
    read = on + g["tokens"] + [THINK_PARA_ID]
    _, r = post(f"{up.url}/completion", {**READ, "prompt": read, "id_slot": 0, "cache_prompt": False})
    assert (r["timings"]["cache_n"], r["timings"]["prompt_n"]) == (0, len(read))
    _, g = post(f"{up.url}/completion", {**GEN, "prompt": on, "seed": 1, "id_slot": 0, "cache_prompt": False})
    assert (g["timings"]["cache_n"], g["timings"]["prompt_n"]) == (0, len(on))
    _, r = post(f"{up.url}/completion", {**READ, "prompt": read, "id_slot": 0})
    assert (r["timings"]["cache_n"], r["timings"]["prompt_n"]) == (len(read) - 2, 2)   # the cache still followed


def test_the_planted_answer_gains_with_the_trace_tokens_in_the_prompt(up):
    on = _on_tokens(up)
    logprobs = {}
    for k in (0, 50, 300):
        trace = [THINK_WORD_BASE] * k
        _, r = post(f"{up.url}/completion", {**READ, "prompt": on + trace, "id_slot": 0})
        logprobs[k] = {e["id"]: e["logprob"] for e in r["completion_probabilities"][0]["top_logprobs"]}
    win = planted_answer(list(Q.texts))
    assert logprobs[300][win] > logprobs[50][win] > logprobs[0][win]
    assert max(logprobs[300], key=logprobs[300].get) == win
    assert planted_gain(300) == planted_gain(1000) == 9.0 and planted_gain(0) == 0.0
    texts = list(Q.texts)
    assert PlantedBias().dump(texts, gain=None) == PlantedBias().dump(texts)
    assert PlantedBias().dump(texts, gain=(win, 0.0)) == PlantedBias().dump(texts)


def test_truncation_at_n_ctx_as_the_server_does_and_the_context_400(up):
    on = _on_tokens(up)
    body = {**GEN, "prompt": on, "seed": 1, "id_slot": 0}
    up.props["default_generation_settings"]["n_ctx"] = len(on) + 3
    _, r = post(f"{up.url}/completion", body)
    assert r["truncated"] is True and len(r["tokens"]) == 3 and r["stop_type"] == "limit"
    # the 6-token close lands exactly on the edge: the server still flags it truncated, as a limit
    up.props["default_generation_settings"]["n_ctx"] = len(on) + 6
    _, r = post(f"{up.url}/completion", body)
    assert (r["truncated"], len(r["tokens"]), r["stop_type"]) == (True, 6, "limit") and r["tokens"][-1] == THINK_TAG_ID
    up.think_eos = True                                                  # ...or as eos when the edge token is an EOG
    _, r = post(f"{up.url}/completion", body)
    assert (r["truncated"], r["stop_type"], r["tokens"][-1]) == (True, "eos", THINK_EOG_ID)
    up.think_eos = False
    # one token of room past the close: a natural close, not truncated
    up.props["default_generation_settings"]["n_ctx"] = len(on) + 7
    _, r = post(f"{up.url}/completion", body)
    assert (r["truncated"], len(r["tokens"]), r["stop_type"]) == (False, 6, "word")
    up.props["default_generation_settings"]["n_ctx"] = len(on)
    for req in (body, {**READ, "prompt": on, "id_slot": 0}):            # a generation and a readout alike
        status, r = post(f"{up.url}/completion", req)
        assert status == 400 and r["error"] == {
            "code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": len(on), "n_ctx": len(on),
            "message": f"request ({len(on)} tokens) exceeds the available context size ({len(on)} tokens), "
                       "try increasing it"}


def test_a_slow_generation_aborts_when_the_client_leaves(up):
    up.think_delay_s, up.think_len = 0.05, 40
    on = _on_tokens(up)
    u = urllib.parse.urlsplit(up.url)
    s = socket.create_connection((u.hostname, u.port))
    body = json.dumps({**GEN, "prompt": on, "n_predict": 40, "seed": 1, "id_slot": 0}).encode()
    s.sendall(b"POST /completion HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(body) + body)
    time.sleep(0.2)
    assert up.slots[0]["is_processing"] is True                        # busy while it generates...
    s.close()
    deadline = time.monotonic() + 3
    while not up.aborted and time.monotonic() < deadline:
        time.sleep(0.02)
    assert up.aborted and up.aborted[0][0] == "/completion"
    deadline = time.monotonic() + 3
    while up.slots[0]["is_processing"] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert up.slots[0]["is_processing"] is False                       # ...and freed by the abort
    # the slot keeps the prompt and the tokens decoded before the abort, as a released slot does
    assert up.cache[0][: len(on)] == on and len(up.cache[0]) < len(on) + 40
    assert all(THINK_WORD_BASE <= t < THINK_WORD_BASE + 100 for t in up.cache[0][len(on):])


def test_the_generation_hook_and_a_planted_host_cache_hit_touch_generations_only(up):
    on = _on_tokens(up)
    seen = []
    up.think_gen_hook = lambda body: seen.append(body["n_predict"])
    up.think_gen_prompt_n = 2
    _, g = post(f"{up.url}/completion", {**GEN, "prompt": on, "seed": 1, "id_slot": 0})
    assert g["tokens"] and g["timings"]["prompt_n"] == 2 and g["timings"]["cache_n"] == len(on) - 2
    assert g["timings"]["prompt_ms"] == 2 * up.think_prompt_ms_per_token
    _, one = post(f"{up.url}/completion", {**GEN, "prompt": on, "n_predict": 1, "seed": 1, "id_slot": 1})
    assert one["timings"]["prompt_n"] == 2                       # a budget of 1 is a generation too (no n_probs)
    _, r = post(f"{up.url}/completion", {**READ, "prompt": on + g["tokens"] + [THINK_PARA_ID], "id_slot": 0})
    assert seen == [64, 1] and r["timings"]["prompt_n"] == 2 and r["timings"]["cache_n"] == len(on) + 5   # real LCP
    up.think_gen_hook, up.think_gen_prompt_n = None, None
    _, g = post(f"{up.url}/completion", {**GEN, "prompt": on, "seed": 1, "id_slot": 2})
    assert g["timings"]["prompt_n"] == len(on) and seen == [64, 1]
