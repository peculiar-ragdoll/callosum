"""The engine's deliberate readout against the thinking fake: one generation,
the token-built prefix, n_gen first then the cuts, shared readouts, the context check, the mapping, cancel, the
self-test, and budget 0 bit-identical to v0.2."""
import base64
import dataclasses
import logging
import math
import string
import threading
import time

import pytest
from fake_upstream import (
    THINK_NL_ID,
    THINK_PARA_ID,
    THINK_TAG_ID,
    FakeUpstream,
    planted_answer,
)

from callo.engine import Deliberate, Engine, SurfaceSets, ThinkSelfTestResult, Trace, think_self_test
from callo.errors import (
    Cancelled,
    MediaRejected,
    ReadoutError,
    ThinkTooLarge,
    UpstreamRejected,
    UpstreamUnavailable,
)
from callo.prompt import TemplateClient
from callo.scorer import LlamaCppForwardScorer, make_http_post
from callo.systemone import parse_request
from callo.think import frame_from_template, parse_sampling, resolve_sampling
from callo.vision import encode_png

TOKENS = list(string.ascii_uppercase) + ["Ġ" + c for c in string.ascii_uppercase] + ["**", "Ġthe", "x"]
DELIM = "<|im_start|>user"
SAMPLING = resolve_sampling(parse_sampling("fixed")).pinned
REQ_BODY = {"state": "the state", "questions": {"q": {"type": "choice", "instructions": "pick",
                                                       "criteria": {"a": "one", "b": "two", "c": "three"}}}}
REQ = parse_request(REQ_BODY)
Q = REQ.questions[0]


def engine_for(up, *, delimiter=DELIM, tokens=TOKENS):
    post = make_http_post(timeout=10)
    e = Engine(LlamaCppForwardScorer(up.url, http_post=post), TemplateClient(up.url, http_post=post),
               SurfaceSets(tokens, "gpt2"), user_delimiter=delimiter, s1_timeout=5)
    e.think, reason = frame_from_template(
        lambda thinking: e.template.render([{"role": "user", "content": "x"}], thinking=thinking), e.scorer.tokenize)
    assert reason is None, reason
    e.n_ctx = up.props["default_generation_settings"]["n_ctx"]
    e.think_sampling = SAMPLING
    return e


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=5).start()
    yield u
    u.stop()


def _gen_calls(up):
    """Generations as llama-server tells them apart: a /completion that asks for no probabilities."""
    return [r.json for r in up.requests if r.path == "/completion" and not r.json.get("n_probs")]


def _reads(up):
    return [r.json for r in up.requests if r.path == "/completion" and r.json.get("n_probs")]


def test_one_generation_carries_the_pinned_sampler_the_stop_the_slot_and_the_delimiter(up):
    e = engine_for(up)
    d = e.answer_deliberate("the state", Q, [256], id_slot=0, seed=11, sampling=SAMPLING)
    assert isinstance(d, Deliberate) and isinstance(d.trace, Trace)
    (g,) = _gen_calls(up)
    assert g["n_predict"] == 256 and g["stop"] == ["</think>"] and g["preserved_tokens"] == ["</think>"]
    assert g["return_tokens"] is True and g["cache_prompt"] is True and g["seed"] == 11 and g["id_slot"] == 0
    assert g["message_delimiters"] == [{"role": "user", "delimiter": DELIM}]
    assert g["reasoning_budget_tokens"] == -1 and g["temperature"] == 0.6 and g["samplers"][0] == "penalties"
    assert g["prompt"][1:] == e.scorer.tokenize(e.template.render(
        __import__("callo.prompt", fromlist=["messages"]).messages("the state", Q), thinking=True),
        add_special=True, parse_special=True)[1:]
    assert d.trace.n_gen == 6 and d.trace.close == "natural" and d.trace.gen_tokens[-1] == THINK_TAG_ID
    assert d.trace.prompt_tokens == len(g["prompt"]) and d.trace.predicted_ms == 60.0 and d.trace.frame_ms >= 0


def test_a_natural_close_reads_at_the_tag_plus_post_tag_and_a_forced_one_at_the_close_suffix(up):
    e = engine_for(up)
    d = e.answer_deliberate("the state", Q, [256], id_slot=0, seed=1, sampling=SAMPLING)
    b = d.readouts[256]
    (read,) = _reads(up)
    assert read["prompt"][-2:] == [THINK_TAG_ID, THINK_PARA_ID] and read["cache_prompt"] is True
    assert (b.k, b.read_prompt_n, b.read_cache_n, b.text_fallback) == (6, 2, len(read["prompt"]) - 2, False)
    assert b.readout.bias == "deliberate" and b.readout.kind == "plain" and b.readout.passes == 1
    up.requests.clear()
    d = e.answer_deliberate("the state", Q, [3], id_slot=0, seed=1, sampling=SAMPLING)
    (read,) = _reads(up)
    assert d.trace.close == "forced" and d.trace.n_gen == 3
    assert read["prompt"][-3:] == [THINK_NL_ID, THINK_TAG_ID, THINK_PARA_ID] and d.readouts[3].read_prompt_n == 4


def test_several_budgets_read_n_gen_first_then_the_cuts_ascending_and_share_readouts_past_n_gen(up):
    up.think_len = 40
    e = engine_for(up)
    d = e.answer_deliberate("the state", Q, [8, 16, 64, 128], id_slot=0, seed=1, sampling=SAMPLING)
    assert d.trace.n_gen == 41 and d.n_readouts == 3
    assert d.readouts[64] is d.readouts[128] and d.readouts[64].k == 41           # closed before 64: shared
    assert (d.readouts[8].k, d.readouts[16].k) == (8, 16)
    reads = _reads(up)
    ks = [sum(1 for t in r["prompt"] if 5000 <= t < 5100) for r in reads]       # trace words (n_gen 41 counts the tag)
    assert ks == [40, 8, 16]                                                       # n_gen first, then the cuts ascending
    assert reads[0]["prompt"][-2:] == [THINK_TAG_ID, THINK_PARA_ID]
    assert all(r["prompt"][-3:] == [THINK_NL_ID, THINK_TAG_ID, THINK_PARA_ID] for r in reads[1:])
    assert all(b.readout.bias == "deliberate" for b in d.readouts.values())


def test_the_planted_curve_is_recovered_the_longer_the_model_thinks(up):
    up.think_len = 400
    e = engine_for(up)
    d = e.answer_deliberate("the state", Q, [1, 64, 300], id_slot=0, seed=1, sampling=SAMPLING)
    win = planted_answer(list(Q.texts))
    p = {b: d.readouts[b].readout.probs for b in (1, 64, 300)}
    assert p[300][win] > p[64][win] > p[1][win] and max(range(3), key=p[300].__getitem__) == win


def test_the_temperature_applies_to_the_readout_and_none_is_the_engines(up):
    e = engine_for(up)
    hot = e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING, temperature=4.0)
    cold = e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING, temperature=0.5)
    same = e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
    assert max(hot.readouts[64].readout.probs) < max(same.readouts[64].readout.probs) < max(cold.readouts[64].readout.probs)
    assert hot.readouts[64].readout.log_marginals == same.readouts[64].readout.log_marginals   # raw stays raw


def test_the_context_check_refuses_before_generating_and_a_truncated_trace_after(up):
    e = engine_for(up)
    n = e.plan_deliberate("the state", Q)
    e.n_ctx = n + 64 + 15
    with pytest.raises(ThinkTooLarge, match=f"prompt \\({n} tokens\\) plus a 64-token budget exceeds the slot context"):
        e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
    assert not _gen_calls(up)
    e.n_ctx = None                                                     # unknown: the check is skipped
    up.props["default_generation_settings"]["n_ctx"] = n + 3
    with pytest.raises(ThinkTooLarge, match="truncated the trace at 3 tokens"):
        e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)


def test_the_upstream_mapping_on_the_generation(up):
    e = engine_for(up)
    up.completion_error = (400, "the request exceeds the available context size")
    with pytest.raises(ThinkTooLarge):
        e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
    up.completion_error = (400, "Failed to tokenize prompt")
    with pytest.raises(UpstreamRejected):
        e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
    up.completion_error = (503, "Loading model")
    with pytest.raises(UpstreamUnavailable):
        e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)


def test_bad_budgets_and_a_missing_frame_are_errors_before_any_call(up):
    e = engine_for(up)
    for bad in ([], [0], [64, 32], [64, 64]):
        with pytest.raises(ValueError):
            e.answer_deliberate("the state", Q, bad, id_slot=0, seed=1, sampling=SAMPLING)
    e.think = None
    with pytest.raises(ReadoutError, match="no thinking mode"):
        e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
    assert not _gen_calls(up)


def test_a_template_that_lost_its_frame_is_a_readout_error():
    u = FakeUpstream(think="preopened", think_len=5).start()
    try:
        e = engine_for(u)
        u.think = "qwen3"                                                  # the render no longer pre-opens
        with pytest.raises(ReadoutError, match="lost its frame"):
            e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
    finally:
        u.stop()


def test_cancel_before_and_during_the_generation(up):
    e = engine_for(up)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING, cancel=cancel)
    assert not _gen_calls(up)
    up.think_delay_s, up.think_len = 0.05, 100
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        e.answer_deliberate("the state", Q, [100], id_slot=0, seed=1, sampling=SAMPLING, cancel=cancel)
    assert time.monotonic() - t0 < 3 and not _reads(up)
    deadline = time.monotonic() + 2
    while not up.aborted and time.monotonic() < deadline:
        time.sleep(0.02)
    assert up.aborted


def test_the_eos_close_drops_the_end_token_and_the_text_fallback_is_counted(up):
    e = engine_for(up)
    up.think_eos = True
    d = e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
    (read,) = _reads(up)
    assert d.trace.close == "eos" and read["prompt"][-3:] == [THINK_NL_ID, THINK_TAG_ID, THINK_PARA_ID]
    assert 2 not in read["prompt"] and d.readouts[64].read_prompt_n == 3       # the EOG token was never decoded
    up.think_eos, up.think_multi_tag = False, True
    up.requests.clear()
    e2 = engine_for(up)                                                    # a two-token tag: tag_single is false
    assert not e2.think.tag_single
    d = e2.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
    assert d.readouts[64].text_fallback is True and d.trace.close == "natural"


def test_answer_parsed_dispatches_on_think_budget_and_budget_0_is_v02s_call_sequence(up):
    e = engine_for(up)
    e.think_sampling = SAMPLING
    out = e.answer_parsed(REQ, id_slot=0)
    plain = [(r.path, r.json.get("n_predict"), r.json.get("cache_prompt")) for r in up.requests]
    assert set(out["answers"]) == {"q"} and all(n in (None, 1) for _, n, _ in plain) and not _gen_calls(up)
    assert all(not c for _, n, c in plain if n == 1)                      # --cache off: v0.2's uncached passes
    up.requests.clear()
    readouts = []
    e.answer_parsed(REQ, id_slot=0, readouts=readouts, think_budget=64, think_T=1.3, think_calibrated=False)
    (qr,) = readouts
    assert isinstance(qr.think, Trace) and qr.think_budget == 64 and qr.think_k == 6 and qr.read_prompt_n == 2
    assert qr.think_T == 1.3 and qr.calibrated is False and qr.readout.bias == "deliberate"
    assert _gen_calls(up)[0]["seed"] == __import__("callo.think", fromlist=["think_seed"]).think_seed(REQ.digest, "q")
    readouts.clear()
    e.answer_parsed(REQ, id_slot=0, readouts=readouts, think_budget=64)
    assert readouts[0].think_T == 1.0 and readouts[0].calibrated is True


def test_answer_parsed_refuses_media_with_a_budget_and_answer_request_honours_the_body_s_budget(up):
    e = engine_for(up)
    png = "data:image/png;base64," + base64.b64encode(encode_png(1, 1, b"\x00\x00\x00")).decode()
    img = parse_request({**REQ_BODY, "images": [png]})
    with pytest.raises(MediaRejected, match="text-only"):
        e.answer_parsed(img, id_slot=0, think_budget=64)
    assert not _gen_calls(up)
    readouts = []
    e.answer_request({**REQ_BODY, "think_budget": 64}, id_slot=0, readouts=readouts)
    assert len(_gen_calls(up)) == 1 and readouts[0].think_budget == 64 and readouts[0].calibrated is False
    assert readouts[0].think_T == e.temperature                            # local_engine: the engine's T, no profile
    up.requests.clear()
    e.answer_request(REQ_BODY, id_slot=0)
    assert not _gen_calls(up)                                              # no field: budget 0, v0.2's path


def test_answer_parsed_skips_the_trace_for_a_single_option_question_and_checks_cancel(up):
    e = engine_for(up)
    one = parse_request({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "one"}}}})
    readouts = []
    e.answer_parsed(one, id_slot=0, readouts=readouts, think_budget=64)
    assert readouts[0].think is None and readouts[0].readout.stop == "single_option" and not _gen_calls(up)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        e.answer_parsed(REQ, id_slot=0, think_budget=64, cancel=cancel)


def test_the_think_self_test_passes_on_the_fake_and_reports_a_broken_slot_check(up):
    e = engine_for(up)
    r = think_self_test(e, id_slot=0, sampling=SAMPLING)
    assert r.ok and r.answer == "4" and r.close == "natural" and r.think_tokens == 6 and r.slot_ok
    assert r.lines()[0].startswith("think self-test ok") and "post-think answer slot check: answer slot confirmed" in r.lines()[-1]
    up.think = "qwen3"
    r = think_self_test(e, id_slot=0, sampling=SAMPLING)
    assert not r.ok and r.error and r.lines()[0].startswith("WARNING: think self-test could not run")


def test_a_qwen3_style_template_gets_the_synthesised_opening_end_to_end():
    """A template that does not pre-open the block (Qwen3 dense, the original Qwen3 MoE)."""
    u = FakeUpstream(think="qwen3", think_len=5).start()
    try:
        e = engine_for(u)
        assert e.think.synthesized_open and e.think.opening == "<think>\n"
        d = e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
        (g,) = _gen_calls(u)
        on = e.template.render(__import__("callo.prompt", fromlist=["messages"]).messages("the state", Q), thinking=True)
        assert on.endswith("<|im_start|>assistant\n") and not on.endswith("<think>\n")
        assert g["prompt"] == e.scorer.tokenize(on + "<think>\n", add_special=True, parse_special=True)
        assert d.trace.close == "natural" and d.readouts[64].read_prompt_n == 2
        (read,) = _reads(u)
        assert read["prompt"][-2:] == [THINK_TAG_ID, THINK_PARA_ID]
    finally:
        u.stop()


def test_without_a_resolved_sampler_the_request_and_the_self_test_pin_the_default_policy(up):
    """`think_sampling` unset falls back to think.py's default policy (`model`, no GGUF keys),
    never to an empty sampling dict that would leave the trace to llama-server's flags."""
    e = engine_for(up)
    e.think_sampling = None
    default = resolve_sampling(parse_sampling("model")).pinned
    e.answer_parsed(REQ, id_slot=0, think_budget=64)
    assert think_self_test(e, id_slot=0).ok
    gens = _gen_calls(up)
    assert len(gens) == 2 and all({k: g[k] for k in default} == default for g in gens)
    assert gens[0]["temperature"] == 0.6 and gens[0]["samplers"][0] == "penalties"


def test_the_think_self_test_warns_when_the_post_think_answer_slot_moves(up):
    """The readout runs, but the template's render of the trace as reasoning_content does not put the answer where
    the readout reads it: the self-test still reports, with a WARNING line for the slot."""
    e = engine_for(up)
    render = e.template.render

    def moved(msgs, **kw):
        out = render(msgs, **kw)
        return out.replace("</think>\n\n", "</think>\n") if msgs[-1]["role"] == "assistant" else out

    e.template.render = moved
    r = think_self_test(e, id_slot=0, sampling=SAMPLING)
    assert r.error is None and r.slot_ok is False and r.answer == "4"
    assert r.lines()[-1].startswith("WARNING: post-think answer slot check: ")


# ---- eos traces, cancel checks, self-test warnings, EOG answer slots -----------------------------------------

def test_an_eos_trace_reads_the_n_gen_minus_1_cut_once_it_is_the_n_gen_prefix(up):
    up.think_eos = True
    e = engine_for(up)
    d = e.answer_deliberate("the state", Q, [5, 64], id_slot=0, seed=1, sampling=SAMPLING)
    assert d.trace.close == "eos" and d.trace.n_gen == 6
    assert len(_reads(up)) == 1 and d.n_readouts == 1                      # one readout, not two identical ones
    assert (d.readouts[5].k, d.readouts[64].k) == (5, 6) and d.readouts[5].readout is d.readouts[64].readout
    up.think_eos = False
    up.requests.clear()
    d = e.answer_deliberate("the state", Q, [5, 64], id_slot=0, seed=1, sampling=SAMPLING)
    assert d.trace.close == "natural" and len(_reads(up)) == 2 and d.n_readouts == 2    # a natural n_gen - 1 cut differs


def test_cancel_is_checked_before_every_readout_pass(up):
    e = engine_for(up)
    gens = []
    generate = e.scorer.generate
    e.scorer.generate = lambda *a, **k: gens.append(generate(*a, **k)) or gens[-1]
    d = e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
    on = e.template.render(__import__("callo.prompt", fromlist=["messages"]).messages("the state", Q), thinking=True)
    up.requests.clear()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled, match="readout passes"):
        e._deliberate_readout(on, (1,) * d.trace.prompt_tokens, gens[0], 6, Q, 0, kind="plain", temperature=None,
                              cancel=cancel)
    assert not _reads(up)


def test_a_tokens_predicted_mismatch_warns_once(up, caplog):
    e = engine_for(up)
    generate = e.scorer.generate
    e.scorer.generate = lambda *a, **k: dataclasses.replace(generate(*a, **k), tokens_predicted=99)
    with caplog.at_level(logging.DEBUG, logger="callo"):
        for _ in range(3):
            e.answer_deliberate("the state", Q, [64], id_slot=0, seed=1, sampling=SAMPLING)
    hits = [r for r in caplog.records if "tokens_predicted 99" in r.getMessage()]
    assert [r.levelno for r in hits] == [logging.WARNING, logging.DEBUG, logging.DEBUG]


def test_a_think_self_test_with_the_wrong_answer_warns_with_the_top_tokens(up):
    r = think_self_test(engine_for(up), id_slot=0, sampling=SAMPLING)
    assert r.ok and r.top_tokens and not any("top-5" in line for line in r.lines())
    wrong = ThinkSelfTestResult(False, "3", 0.9, "natural", 6, True, "answer slot confirmed", None,
                                (("3", -0.1), ("4", -2.4)))
    lines = wrong.lines()
    assert lines[0].startswith("think self-test FAILED: 2 + 2 = ? at budget 64 -> answer=3")
    assert lines[1] == ("WARNING: top-5 raw next tokens at the post-think answer slot: '3' -0.10, '4' -2.40 "
                        "(deliberate readouts on this model may be unreliable)")


# The Sharp-MiniCPM5-2B GPQA question at think budget 1024 (callo 0.3.0, a deterministic HTTP 500): after the forced
# close the answer slot put 0.975 on <|im_end|> (130073, an EOG) and the options under 0.025; the walk appended the EOG
# as a "shared prefix" and scored the position after it, OUTSIDE the assistant turn, whose argmax was the lead byte of
# an emoji (13922, bytes F0 9F). llama-server never add_token's an incomplete UTF-8 piece, so that reply carried no
# completion_probabilities. The planted dumps are the measured ones, rounded.
EOG, BYTE = len(TOKENS), len(TOKENS) + 1
EOG_TOKENS = [*TOKENS, "<|im_end|>", "ðŁ"]
SLOT_TOP = [{"id": EOG, "token": "", "logprob": -0.0255}, {"id": 0, "token": "A", "logprob": -4.37},
            {"id": 1, "token": "B", "logprob": -4.42}, {"id": 2, "token": "C", "logprob": -8.82}]
PAST_TOP = [{"id": BYTE, "token": "", "logprob": -1.83}, {"id": EOG, "token": "", "logprob": -2.71},
            {"id": 0, "token": "A", "logprob": -9.0}]


@pytest.fixture
def eog_up():
    u = FakeUpstream(think="preopened", think_len=5)
    u.eog_ids, u.partial_utf8_ids = {EOG}, {BYTE}
    u.readout_top = lambda prompt, _top: PAST_TOP if prompt and prompt[-1] == EOG else SLOT_TOP
    u.start()
    yield u
    u.stop()


def _slot_probs(T):
    z = [lp / T for lp in (-4.37, -4.42, -8.82)]
    e = [math.exp(v - max(z)) for v in z]
    return [x / sum(e) for x in e]


def test_an_eog_answer_slot_is_read_there_never_past_the_end_of_the_turn(eog_up):
    e = engine_for(eog_up, tokens=EOG_TOKENS)
    d = e.answer_deliberate("the state", Q, [256], id_slot=0, seed=11, sampling=SAMPLING)
    r = d.readouts[256].readout
    (read,) = _reads(eog_up)                        # one pass: the EOG is never appended and read past
    assert read["prompt"][-1] != EOG
    assert r.stop == "eog" and r.depth == 0 and r.passes == 1
    assert r.option_mass == pytest.approx(math.exp(-4.37) + math.exp(-4.42) + math.exp(-8.82))
    assert list(r.probs) == pytest.approx(_slot_probs(e.temperature))


def test_an_eog_answer_slot_at_budget_0_and_through_answer_parsed(eog_up):
    e = engine_for(eog_up, tokens=EOG_TOKENS)
    r = e.answer("the state", Q, id_slot=0)
    assert r.stop == "eog" and r.passes == 1 and list(r.probs) == pytest.approx(_slot_probs(e.temperature))
    readouts = []
    out = e.answer_parsed(REQ, id_slot=0, readouts=readouts, think_budget=64)
    assert out["answers"]["q"]["choice"] == "a" and readouts[0].readout.stop == "eog"


# The other way a readout pass loses its probabilities: llama-server withholds a greedy token that is an incomplete
# UTF-8 piece, and its completion_probabilities with it (measured on the same server: the reply after <|im_end|>).
# Its raw logits are untouched by logit_bias, so the scorer learns the token (return_tokens) and asks again with it
# banned. Planted: the answer slot's argmax is a dominant byte piece (a neutral prefix: appended) whose banned
# re-ask greedily lands on an EOG, which must NOT read as the argmax's verdict.
HELD_TOP = [{"id": BYTE, "token": "", "logprob": math.log(0.55)}, {"id": EOG, "token": "", "logprob": math.log(0.3)},
            {"id": 0, "token": "A", "logprob": math.log(0.05)}, {"id": 1, "token": "B", "logprob": math.log(0.03)}]
AFTER_BYTE = [{"id": 1, "token": "B", "logprob": math.log(0.7)}, {"id": 0, "token": "A", "logprob": math.log(0.2)}]


def test_a_withheld_partial_utf8_argmax_is_re_asked_with_it_banned_and_never_read_as_eog(eog_up):
    eog_up.readout_top = lambda prompt, _top: AFTER_BYTE if prompt[-1] == BYTE else HELD_TOP
    e = engine_for(eog_up, tokens=EOG_TOKENS)
    r = e.answer("the state", Q, id_slot=0)
    assert (r.stop, r.depth, r.passes, r.reasks) == ("diverged", 1, 2, 1) and r.probs.index(max(r.probs)) == 1
    reads = _reads(eog_up)
    assert [x["logit_bias"] for x in reads] == [[], [[BYTE, False]], []]
    assert reads[0] == {**reads[1], "logit_bias": []}                                    # the same pass, re-asked


def test_a_pass_whose_every_greedy_pick_is_withheld_is_a_readout_error_naming_the_tokens(eog_up):
    eog_up.partial_utf8_ids = {BYTE, EOG, 0, 1, 2}
    eog_up.readout_top = lambda prompt, _top: [*HELD_TOP, {"id": 2, "token": "C", "logprob": math.log(0.01)}]
    e = engine_for(eog_up, tokens=EOG_TOKENS)
    with pytest.raises(ReadoutError, match="incomplete UTF-8"):
        e.answer("the state", Q, id_slot=0)


# The pin (scorer.READOUT_PIN) makes the readout's greedy pick the raw argmax whatever the server's sampling flags.
def test_an_ignore_eos_server_still_reports_the_eog_the_pin_clears_its_base_bans(eog_up):
    eog_up.base_ignore_eos = True                                                 # llama-server --ignore-eos
    e = engine_for(eog_up, tokens=EOG_TOKENS)
    r = e.answer("the state", Q, id_slot=0)
    assert (r.stop, r.passes) == ("eog", 1) and list(r.probs) == pytest.approx(_slot_probs(e.temperature))
    (read,) = _reads(eog_up)
    assert read["logit_bias"] == [] and read["ignore_eos"] is False and read["samplers"] == ["temperature"]


def test_a_penalty_cannot_move_the_pick_off_the_eog_or_onto_it(eog_up):
    eog_up.penalized_ids = {EOG}                                                  # e.g. --repeat-penalty on <|im_end|>
    e = engine_for(eog_up, tokens=EOG_TOKENS)
    assert e.answer("the state", Q, id_slot=0).stop == "eog"
    # the other way: the dominant token is a neutral prefix ("**", 52), a penalty on it moves the pick ONTO the EOG,
    # on a server that ignores the pin: the verdict names a token the walk would not append, so it appends "**"
    stars = [{"id": 52, "token": "**", "logprob": math.log(0.6)}, {"id": EOG, "token": "", "logprob": math.log(0.3)},
             {"id": 0, "token": "A", "logprob": math.log(0.05)}, {"id": 1, "token": "B", "logprob": math.log(0.03)}]
    eog_up.readout_top = lambda prompt, _top: AFTER_BYTE if prompt[-1] == 52 else stars
    eog_up.penalized_ids = {52}
    for ignores in (False, True):
        eog_up.ignores_pin = ignores
        eog_up.requests.clear()
        r = e.answer("the state", Q, id_slot=0)
        assert (r.stop, r.depth, r.passes) == ("diverged", 1, 2), ignores
        assert [x["prompt"][-1] for x in _reads(eog_up)][1] == 52
