"""think.py, the pure half: the frame, the pinned sampler, seeds and the readout
prefix. No upstream: the tokenizer is a stand-in that gives `</think>`, `\\n\\n` and `\\n` their own ids."""
import pytest

from callo.errors import InvalidInput
from callo.think import (
    CALLO_SAMPLING,
    DEFAULT_SAMPLERS,
    INERT_PROPS,
    PINNED_SAMPLER,
    SAMPLER_FIELDS,
    TRACE_FIELDS,
    ThinkFrame,
    check_on_prompt,
    check_pinned,
    context_need,
    detect_frame,
    frame_from_template,
    parse_sampling,
    read_gguf_sampling,
    readout_prefix,
    resolve_sampling,
    think_seed,
    unknown_props_keys,
    unpinnable,
)

PRE = "<|im_start|>user\nhi<|im_end|>\n"
ON_PRE, OFF = PRE + "<|im_start|>assistant\n<think>\n", PRE + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
ON_QWEN3 = PRE + "<|im_start|>assistant\n"
IDS = {"</think>": 2001, "\n\n": 2002, "\n": 2003}


def tok(text, *, add_special=False, parse_special=True):
    import re
    out, i = [], 0
    for piece in re.findall(r"\n\n|\n|</think>|\S+", text):
        out.append(IDS.get(piece, 100 + i))
        i += piece not in IDS
    return out


def test_a_pre_opened_template_gives_the_frame_with_its_token_forms():
    frame, reason = detect_frame(ON_PRE, OFF, tok)
    assert reason is None and isinstance(frame, ThinkFrame)
    assert (frame.tag, frame.close, frame.post, frame.opening) == ("</think>", "\n</think>\n\n", "\n\n", "")
    assert frame.close_tokens == (2003, 2001, 2002) and frame.post_tokens == (2002,) and frame.tag_tokens == (2001,)
    assert frame.tag_single and not frame.synthesized_open
    assert frame.info() == {"close": "\n</think>\n\n", "synthesized_open": False, "tag": "</think>", "tag_single": True}


def test_a_qwen3_template_synthesises_the_opening_from_the_empty_block():
    frame, reason = detect_frame(ON_QWEN3, OFF, tok)
    assert reason is None and frame.synthesized_open and frame.opening == "<think>\n"
    assert check_on_prompt(ON_QWEN3, frame) == ON_QWEN3 + "<think>\n"
    with pytest.raises(ValueError, match="pre-opens"):
        check_on_prompt(ON_PRE, frame)


def test_check_on_prompt_refuses_a_pre_opened_frame_whose_render_is_not_pre_opened():
    frame, _ = detect_frame(ON_PRE, OFF, tok)
    assert check_on_prompt(ON_PRE, frame) == ON_PRE
    with pytest.raises(ValueError, match="does not end inside an opened think block"):
        check_on_prompt(ON_QWEN3, frame)


@pytest.mark.parametrize(("on", "off", "reason"), [
    (ON_QWEN3, ON_QWEN3, "no_think_block"),                                       # on == off, no tag
    (PRE + "<|im_start|>assistant\n<|channel>thought\n", ON_QWEN3, "unsupported_think_format"),   # Gemma-like
    (ON_QWEN3, PRE + "<|im_start|>bot\n<think>\n\n</think>\n\n", "not_preopened"),     # off does not extend on
    (OFF, OFF, "not_preopened"),                                                 # a template that ignores the kwarg
])
def test_the_reasons_a_model_is_budget_0_only(on, off, reason):
    assert detect_frame(on, off, tok) == (None, reason)


def test_a_tokenizer_failure_and_a_render_failure_are_reasons_not_crashes():
    def broken(text, **kw):
        raise OSError("down")

    assert detect_frame(ON_PRE, OFF, broken)[1].startswith("render_failed: tokenize: OSError")

    def render(thinking):
        raise RuntimeError("no template")

    assert frame_from_template(render, tok) == (None, "render_failed: RuntimeError: no template")
    assert frame_from_template(lambda thinking: ON_PRE if thinking else OFF, tok)[1] is None


def test_a_multi_token_tag_is_not_tag_single():
    def tok2(text, **kw):
        return [1, 2] if text == "</think>" else tok(text)

    frame, _ = detect_frame(ON_PRE, OFF, tok2)
    assert not frame.tag_single and frame.tag_tokens == (1, 2)


# ---- the pinned sampler --------------------------------------------------------------------------------

# Every key of llama-server's /props default_generation_settings.params (server-task.cpp params_from_json_cmpl
# to_json @ 4ebdf2c), with the server's own defaults.
PROPS_PARAMS = {"n_predict": -1, "seed": 4294967295, "temperature": 0.8, "dynatemp_range": 0.0,
                "dynatemp_exponent": 1.0, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "top_n_sigma": -1.0,
                "xtc_probability": 0.0, "xtc_threshold": 0.1, "typical_p": 1.0, "repeat_last_n": 64,
                "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
                "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": -1,
                "dry_sequence_breakers": ["\n", ":", "\"", "*"], "mirostat": 0, "mirostat_tau": 5.0,
                "mirostat_eta": 0.1, "stop": [], "max_tokens": -1, "n_keep": 0, "n_discard": 0, "ignore_eos": False,
                "stream": False, "logit_bias": [], "n_probs": 0, "min_keep": 0, "grammar": "", "grammar_lazy": False,
                "grammar_triggers": [], "preserved_tokens": [], "chat_format": "Content-only",
                "reasoning_format": "auto", "reasoning_in_content": False, "thinking_forced_open": False,
                "samplers": list(DEFAULT_SAMPLERS), "speculative.n_max": 16, "speculative.n_min": 0,
                "speculative.p_min": 0.75, "timings_per_token": False, "post_sampling_probs": False, "lora": [],
                "adaptive_target": -1.0, "adaptive_decay": 0.9, "backend_sampling": False,
                "reasoning_budget_tokens": -1}


def test_every_real_props_key_is_pinned_or_inert():
    for key in PROPS_PARAMS:
        assert key == "grammar" or key in SAMPLER_FIELDS or key in INERT_PROPS or key.startswith("speculative"), key
    assert unpinnable(PROPS_PARAMS) is None
    assert unpinnable({**PROPS_PARAMS, "temperature": 2.0, "mirostat": 2, "samplers": ["temperature"]}) is None


def test_a_server_grammar_or_an_unknown_non_default_key_is_unpinnable():
    assert unpinnable({**PROPS_PARAMS, "grammar": "root ::= [a-z]+"}) == "grammar"
    assert unpinnable({**PROPS_PARAMS, "new_sampler_knob": 0.3}) == "new_sampler_knob"
    assert unpinnable({**PROPS_PARAMS, "new_sampler_knob": 0.0, "another": [], "third": None}) is None
    assert unknown_props_keys({**PROPS_PARAMS, "new_sampler_knob": 0.0, "another": []}) == ["another", "new_sampler_knob"]
    assert unknown_props_keys(PROPS_PARAMS) == []


def test_check_pinned_names_the_first_fault_and_passes_the_full_dict():
    pinned = resolve_sampling(parse_sampling("fixed")).pinned
    assert check_pinned(pinned) is None and check_pinned(dict(pinned)) is None
    assert check_pinned(None) == "must be an object" and check_pinned([]) == "must be an object"
    assert check_pinned({}) == "lacks adaptive_decay"                           # the first missing field, sorted
    assert check_pinned({**pinned, "n_predict": 3}) == "has 'n_predict', which is not a sampler field"
    assert check_pinned({**pinned, "top_k": 20.0}) == "top_k must be an int, got 20.0"
    assert check_pinned({**pinned, "temperature": True}) == "temperature must be a number, got True"
    assert check_pinned({**pinned, "top_p": float("nan")}) == "top_p must be finite, got nan"
    assert check_pinned({**pinned, "mirostat_tau": float("inf")}) == "mirostat_tau must be finite, got inf"


# llama-server's request limits at 4ebdf2c (tools/server/server-schema.cpp:89-190, :242-247, :387-389, :434-480,
# :505-512): a hard limit is a 400, a soft limit a silent clamp; either way the value is not the one the server uses.
@pytest.mark.parametrize(("field", "value", "why"), [
    ("temperature", -0.5, "temperature must be"),                                  # soft [0, inf): clamped to 0
    ("top_p", 7, "top_p must be in [0, 1]"), ("top_p", -0.1, "top_p must be in [0, 1]"),   # soft: clamped
    ("top_k", 2**31, "top_k must be a 32-bit integer"),
    ("min_p", -0.1, "min_p must be in [0, 1]"), ("min_p", 1.5, "min_p must be in [0, 1]"),
    ("xtc_probability", 2.0, "xtc_probability must be in [0, 1]"),
    ("xtc_threshold", 1.5, "xtc_threshold must be in [0, 1]"),
    ("repeat_last_n", -1, "repeat_last_n must be"), ("repeat_last_n", 64.0, "repeat_last_n must be an int"),
    ("dry_penalty_last_n", -1, "dry_penalty_last_n must be"),                     # hard [0, INT32_MAX]: a 400
    ("dry_allowed_length", -1, "dry_allowed_length must be"),
    ("dry_base", 0.5, "dry_base must be >= 1"),                                   # below 1: the server's default
    ("mirostat", 3, "mirostat must be 0, 1 or 2"), ("min_keep", -1, "min_keep must be"),
    ("adaptive_decay", 0.995, "adaptive_decay must be in [0, 0.99]"),
    ("adaptive_target", 1.5, "adaptive_target must be <= 1"),
    ("reasoning_budget_tokens", 64, "reasoning_budget_tokens must be -1"),           # the server's budget would cut
    ("ignore_eos", True, "ignore_eos must be false"),                                 # a trace that never closes
    ("ignore_eos", 0, "ignore_eos must be a boolean"), ("backend_sampling", "no", "backend_sampling must be a boolean"),
    ("samplers", "top_k;top_p", "samplers must be a list"),
    ("samplers", ["top_k", "foo"], "samplers must be llama.cpp's samplers"),         # dropped silently by the server
    ("samplers", ["top_k", 3], "samplers must be llama.cpp's samplers"),
    ("dry_sequence_breakers", [], "dry_sequence_breakers must be a non-empty list of strings"),   # a 400
    ("dry_sequence_breakers", ["\n", 1], "dry_sequence_breakers must be a non-empty list of strings"),
    ("logit_bias", "x", "logit_bias must be a list or an object"),
    # final fix wave: the pins no policy sets must stay neutral (a probe or profile turning one on is refused)
    ("logit_bias", [[1, -1.0]], "logit_bias must be empty"), ("logit_bias", {"1": -1.0}, "logit_bias must be empty"),
    ("dynatemp_range", 0.5, "dynatemp_range must be 0"), ("typical_p", 0.9, "typical_p must be 1"),
    ("presence_penalty", 0.1, "presence_penalty must be 0"), ("frequency_penalty", -0.1, "frequency_penalty must be 0"),
    ("dry_multiplier", 0.8, "dry_multiplier must be 0"), ("top_n_sigma", 1.0, "top_n_sigma must be <= 0"),
    ("repeat_penalty", "1.0", "repeat_penalty must be a number"),
    ("dry_base", float("nan"), "dry_base must be finite")])
def test_check_pinned_refuses_a_value_llama_server_would_clamp_refuse_or_that_breaks_the_trace(field, value, why):
    pinned = resolve_sampling(parse_sampling("fixed")).pinned
    got = check_pinned({**pinned, field: value})
    assert got is not None and got.startswith(why), got


@pytest.mark.parametrize(("field", "value"), [
    ("temperature", 0.0), ("temperature", 0),                                      # greedy: llama.cpp's temp 0
    ("top_k", 0), ("top_k", -1), ("top_p", 1), ("top_p", 0.0), ("min_p", 1.0), ("mirostat", 2),
    ("repeat_last_n", 0), ("dry_penalty_last_n", 0), ("dry_base", 1.0), ("samplers", []),
    ("samplers", ["temperature"]), ("samplers", ["nucleus", "TOP-K"]), ("top_n_sigma", -1.0), ("n_probs", 5),
    ("logit_bias", {}), ("adaptive_target", -1.0), ("adaptive_decay", 0.99), ("top_n_sigma", 0.0),
    ("samplers", ["temperature", "top_k"])])                                       # a GGUF sequence may reorder it
def test_check_pinned_takes_every_value_llama_server_takes_as_given(field, value):
    pinned = resolve_sampling(parse_sampling("fixed")).pinned
    assert check_pinned({**pinned, field: value}) is None


def test_a_gguf_value_check_pinned_would_refuse_is_skipped_so_every_resolved_dict_passes(caplog):
    gguf = {"temp": 0.0, "top_p": 1.5, "min_p": 0.05, "mirostat": 7, "xtc_probability": 3.0, "top_k": -1}
    s = resolve_sampling(parse_sampling("model"), gguf)
    assert s.pinned["temperature"] == 0.0 and s.origin["temperature"] == "gguf"      # greedy is honoured
    assert s.pinned["top_p"] == 0.95 and s.origin["top_p"] == "callo"               # clamped by the server: skipped
    assert s.pinned["mirostat"] == 0 and s.pinned["xtc_probability"] == 0.0
    assert s.pinned["min_p"] == 0.05 and s.pinned["top_k"] == -1                     # in range: the GGUF's
    assert check_pinned(s.pinned) is None
    assert "general.sampling.top_p" in caplog.text and "callo's pin" in caplog.text
    assert "general.sampling.temp " not in caplog.text and "general.sampling.temp (" not in caplog.text


def test_the_pinned_dict_has_every_field_and_the_default_chain():
    s = resolve_sampling(parse_sampling("fixed"))
    assert set(s.pinned) == set(SAMPLER_FIELDS) and set(TRACE_FIELDS) <= set(s.pinned)
    assert s.pinned["samplers"] == list(DEFAULT_SAMPLERS) and s.pinned["reasoning_budget_tokens"] == -1
    assert {k: s.pinned[k] for k in TRACE_FIELDS} == CALLO_SAMPLING
    assert s.origin == dict.fromkeys(TRACE_FIELDS, "callo") and s.policy == "fixed" and s.trace == "0.6/0.95/20/0"
    assert s.pinned["samplers"] is not PINNED_SAMPLER["samplers"]           # a copy: a caller cannot move the pin


def test_policy_model_takes_the_gguf_values_per_field_with_callo_filling_the_rest():
    tiel = {"temp": 1.0, "top_k": 20, "top_p": 0.95}
    s = resolve_sampling(parse_sampling("model"), tiel)
    assert s.trace == "1/0.95/20/0" and s.pinned["min_p"] == 0.0
    assert s.origin == {"temperature": "gguf", "top_p": "gguf", "top_k": "gguf", "min_p": "callo"}
    assert s.gguf == tiel and s.policy == "model"
    none = resolve_sampling(parse_sampling("model"), {})
    assert none.trace == "0.6/0.95/20/0" and set(none.origin.values()) == {"callo"}


def test_policy_model_maps_the_extra_gguf_keys_as_llama_cpp_does():
    g = {"top_k": -1, "sequence": "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature", "penalty_last_n": 128, "penalty_repeat": 1.1, "mirostat": 2,
         "mirostat_tau": 4.0, "mirostat_eta": 0.2, "xtc_probability": 0.5, "xtc_threshold": 0.2}
    p = resolve_sampling(parse_sampling("model"), g).pinned
    assert p["top_k"] == -1 and type(p["top_k"]) is int                      # <= 0 disables top-k, passed through
    assert p["samplers"] == ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc",
                             "temperature"]
    assert (p["repeat_last_n"], p["repeat_penalty"], p["mirostat"], p["mirostat_tau"], p["mirostat_eta"]) == \
        (128, 1.1, 2, 4.0, 0.2)
    assert (p["xtc_probability"], p["xtc_threshold"]) == (0.5, 0.2)
    assert resolve_sampling(parse_sampling("model"), {"sequence": "zz"}).pinned["samplers"] == []    # llama.cpp: empty
    assert resolve_sampling(parse_sampling("model"), {"temp": "hot"}).pinned["temperature"] == 0.6   # malformed: pin


def test_policy_fixed_ignores_the_gguf_and_a_spec_overrides_per_field():
    assert resolve_sampling(parse_sampling("fixed"), {"temp": 1.0, "top_k": 5}).trace == "0.6/0.95/20/0"
    s = resolve_sampling(parse_sampling("temp=0.7,top_k=-1"), {"temp": 1.0, "top_p": 0.5})
    assert s.trace == "0.7/0.95/-1/0" and s.policy == "temp=0.7,top_k=-1"
    assert s.origin == {"temperature": "user", "top_p": "callo", "top_k": "user", "min_p": "callo"}


@pytest.mark.parametrize("bad", ["", "temp", "temp=x", "temp=0.7,temp=0.8", "top_q=1", "temp=nan",
                                 "temp=-1", "temp=inf", "top_p=-0.1", "top_p=1.5", "top_p=nan", "min_p=-0.1",
                                 "min_p=1.01", "top_k=2147483648", "top_k=0.5", "foo=bar"])
def test_a_bad_sampling_spec_is_invalid_input(bad):
    with pytest.raises(InvalidInput, match="--think-sampling"):
        parse_sampling(bad)


def test_a_spec_accepts_llama_cpp_s_valid_ranges():
    s = parse_sampling("temp=2,top_p=1,top_k=0,min_p=1")
    assert s.values == {"temperature": 2.0, "top_p": 1.0, "top_k": 0, "min_p": 1.0}
    assert parse_sampling("min_p=0,top_p=0.01,top_k=-1").values == {"min_p": 0.0, "top_p": 0.01, "top_k": -1}


def test_a_spec_takes_the_ranges_check_pinned_takes_greedy_included():
    """--think-sampling and the pinned dict agree, so temp=0 (greedy) and top_p=0 load."""
    s = parse_sampling("temp=0,top_p=0")
    assert s.values == {"temperature": 0.0, "top_p": 0.0}
    assert check_pinned(resolve_sampling(s).pinned) is None
    with pytest.raises(InvalidInput, match="--think-sampling: temperature must be finite, got inf"):
        parse_sampling("temp=inf")


# ---- the GGUF's own sampling, read as llama.cpp reads it (common.cpp:1106-1164 @ 4ebdf2c) ----------------

@pytest.mark.parametrize(("sequence", "chain"), [
    ("top_k;top_p;min_p;temperature", ["top_k", "top_p", "min_p", "temperature"]),       # names, ';'-separated
    ("Top-K;nucleus;MINP;temp;typ;adaptive-p;infill", ["top_k", "top_p", "min_p", "temperature", "typ_p",
                                                      "adaptive_p", "infill"]),             # aliases, any case
    ("top_k;bogus;temperature", ["top_k", "temperature"]),                                # an unknown name is dropped
    ("top_k; top_p", ["top_k"]),                                                           # no trimming
    ("kpt", []),                                                                           # letters are not names
    ("", list(DEFAULT_SAMPLERS)),                                                          # empty: the key is skipped
])
def test_the_gguf_sequence_is_parsed_as_llama_cpp_parses_it(sequence, chain):
    assert resolve_sampling(parse_sampling("model"), {"sequence": sequence}).pinned["samplers"] == chain


def test_gguf_floats_are_the_shortest_float32_decimal_llama_cpp_ends_up_with():
    import numpy as np
    g = {"temp": float(np.float32(0.6)), "top_p": float(np.float32(0.95)), "min_p": float(np.float32(0.05))}
    assert g["top_p"] == 0.949999988079071                                               # what the header holds
    s = resolve_sampling(parse_sampling("model"), g)
    assert (s.pinned["temperature"], s.pinned["top_p"], s.pinned["min_p"]) == (0.6, 0.95, 0.05)
    assert s.pinned["top_p"] == resolve_sampling(parse_sampling("fixed")).pinned["top_p"]
    # std::to_string is "%f": a value below 6 decimals reaches llama.cpp as 0
    assert resolve_sampling(parse_sampling("model"), {"min_p": 4e-7}).pinned["min_p"] == 0.0


def test_a_gguf_penalty_last_n_below_0_pins_0_as_llama_cpp_clamps_it():
    # llama-sampler.cpp:3209 clamps it to 0 (penalties off); a request's repeat_last_n must be >= 0 (a 400 otherwise)
    assert resolve_sampling(parse_sampling("model"), {"penalty_last_n": -1}).pinned["repeat_last_n"] == 0
    assert resolve_sampling(parse_sampling("model"), {"penalty_last_n": 128}).pinned["repeat_last_n"] == 128


def test_an_empty_gguf_chain_is_kept_and_warned_about(caplog):
    with caplog.at_level("WARNING", logger="callo"):
        assert resolve_sampling(parse_sampling("model"), {"sequence": "bogus"}).pinned["samplers"] == []
    assert "empty sampler chain" in caplog.text and "'bogus'" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="callo"):
        resolve_sampling(parse_sampling("model"), {"sequence": "top_k;temperature"})
    assert caplog.text == ""


def test_gguf_values_coerce_as_strtol_and_strtof_do():
    def pinned(**g):
        return resolve_sampling(parse_sampling("model"), g).pinned

    assert pinned(top_k=20.7)["top_k"] == 20 and pinned(top_k=-0.5)["top_k"] == 0           # truncation, as strtol
    assert pinned(top_k="20.7")["top_k"] == 20 and pinned(temp="0.7abc")["temperature"] == 0.7   # a numeric prefix
    assert pinned(top_k=True)["top_k"] == 20 and pinned(temp=False)["temperature"] == 0.6    # "true": ignored
    assert pinned(temp=float("nan"))["temperature"] == 0.6 and pinned(top_p=float("inf"))["top_p"] == 0.95
    assert pinned(temp="inf")["temperature"] == 0.6 and pinned(mirostat="x")["mirostat"] == 0
    s = resolve_sampling(parse_sampling("model"), {"temp": True, "top_p": float("nan")})
    assert s.origin == dict.fromkeys(TRACE_FIELDS, "callo")


def test_a_long_decimal_string_rounds_to_float32_once_as_strtof_does():
    """1 + 2^-24 is the midpoint between float32 1.0 and its successor, and the string is a hair above it, so strtof
    rounds up. float() lands exactly on the midpoint (the hair is below double precision) and float32 then ties to even:
    1.0, one float32 step off. The shared `stof` rounds once."""
    text = "1.000000059604644775390625000000000001"
    assert resolve_sampling(parse_sampling("model"), {"temp": text}).pinned["temperature"] == 1.0000001
    assert resolve_sampling(parse_sampling("model"), {"temp": text[:-1] + "0"}).pinned["temperature"] == 1.0  # a tie


def test_a_decimal_exponent_out_of_decimals_range_is_skipped_not_raised():
    s = resolve_sampling(parse_sampling("model"), {"temp": "1e-9999999999999999999"})
    assert s.pinned["temperature"] == 0.6 and s.origin["temperature"] == "callo"          # skipped: callo's pin stays


def _write_gguf(path, **kv):
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), "llama")
    for key, (kind, value) in kv.items():
        getattr(w, f"add_{kind}")(f"general.sampling.{key}", value)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_read_gguf_sampling_reads_the_header_keys(tmp_path):
    path = _write_gguf(tmp_path / "s.gguf", temp=("float32", 1.0), top_k=("int32", 20), top_p=("float32", 0.95),
                       sequence=("string", "top_k;top_p;temperature"))
    g = read_gguf_sampling(path)
    assert g == {"temp": 1.0, "top_k": 20, "top_p": 0.95, "sequence": "top_k;top_p;temperature"}
    assert type(g["top_k"]) is int and g.error is None
    s = resolve_sampling(parse_sampling("model"), g)
    assert s.trace == "1/0.95/20/0" and s.pinned["samplers"] == ["top_k", "top_p", "temperature"]
    assert s.gguf_error is None and "gguf_error" not in s.as_dict()
    assert read_gguf_sampling(_write_gguf(tmp_path / "none.gguf")) == {}


def test_a_missing_or_garbage_file_is_invalid_input(tmp_path):
    with pytest.raises(InvalidInput, match="not a readable GGUF"):
        read_gguf_sampling(tmp_path / "nope.gguf")
    (tmp_path / "junk.gguf").write_bytes(b"not a gguf at all")
    with pytest.raises(InvalidInput, match="not a readable GGUF"):
        read_gguf_sampling(tmp_path / "junk.gguf")


def test_a_gguf_gguf_py_cannot_parse_falls_back_to_callo_s_sampler(tmp_path, monkeypatch, caplog):
    pytest.importorskip("gguf")
    path = _write_gguf(tmp_path / "gsq.gguf", temp=("float32", 1.0))

    import callo.ggufkv

    def unparseable(*a, **kw):
        raise ValueError("Unknown/unhandled field type 99")

    monkeypatch.setattr(callo.ggufkv, "open_kv", unparseable)
    with caplog.at_level("WARNING", logger="callo"):
        g = read_gguf_sampling(path)
    assert g == {} and "could not be read" in g.error and "Unknown/unhandled field type 99" in g.error
    assert "general.sampling" in caplog.text and "callo's trace sampler" in caplog.text
    s = resolve_sampling(parse_sampling("model"), g)
    assert s.trace == "0.6/0.95/20/0" and s.origin == dict.fromkeys(TRACE_FIELDS, "callo")
    assert s.gguf_error == g.error and s.as_dict()["gguf_error"] == g.error


# ---- seeds and the prefix -----------------------------------------------------------------------------

def test_seeds_are_stable_uint31_and_differ_by_part():
    s = think_seed("0", "rec-17", "q")
    assert s == think_seed("0", "rec-17", "q") == 930478869 and 0 <= s <= 0x7FFFFFFF      # golden: sha256, big-endian
    assert len({s, think_seed("1", "rec-17", "q"), think_seed("0", "rec-18", "q"), think_seed("0", "rec-17", "r")}) == 4


def test_readout_prefix_natural_forced_cut_eos_and_the_text_fallback():
    frame, _ = detect_frame(ON_PRE, OFF, tok)
    on, gen = (1, 2, 3), (5000, 5001, 5002, 2001)
    assert readout_prefix(on, gen, "natural", 4, frame) == (1, 2, 3, 5000, 5001, 5002, 2001, 2002)   # tag + post
    assert readout_prefix(on, gen, "natural", 2, frame) == (1, 2, 3, 5000, 5001, 2003, 2001, 2002)   # a cut: forced
    forced = (5000, 5001, 5002, 5003)
    assert readout_prefix(on, forced, "forced", 4, frame) == (1, 2, 3, *forced, 2003, 2001, 2002)
    assert readout_prefix(on, (5000, 5001, 2), "eos", 3, frame) == (1, 2, 3, 5000, 5001, 2003, 2001, 2002)  # EOG dropped
    assert readout_prefix(on, (5000, 5001, 5002), "natural", 3, frame) is None       # last token is not the tag
    with pytest.raises(ValueError):
        readout_prefix(on, gen, "natural", 5, frame)
    with pytest.raises(ValueError):
        readout_prefix(on, gen, "weird", 4, frame)
    with pytest.raises(ValueError, match="unknown close"):
        readout_prefix(on, gen, "weird", 2, frame)                                    # a cut checks the close too


def test_context_need_is_the_prompt_the_budget_and_16():
    assert context_need(931, 1024) == 931 + 1024 + 16


# ---- the think profile ------------------------------------------------------------------------------

import json  # noqa: E402
import math  # noqa: E402

from callo.errors import CalloError  # noqa: E402
from callo.think import (  # noqa: E402
    METRICS_VERSION,
    check_think_identity,
    check_think_profile,
    load_think_profile,
)

KEY = "sha256:" + "5a" * 32
PROV = {"prompt_sha256": "p" * 64, "chat_template_sha256": "t" * 64, "build_info": "b1-926862e", "mmproj": None,
        "user_delimiter": "<|im_start|>user", "cache_prompt": False, "server_args": None}


def profile_dict(**over):
    """A minimal valid think profile, with every budget's T and the full pinned sampler."""
    sampling = resolve_sampling(parse_sampling("model"), {"temp": 1.0, "top_k": 20, "top_p": 0.95})
    d = {"callo_think_profile": 1, "metrics": METRICS_VERSION, "model_key": KEY, "model_name": "tiel",
         "comparable": {"sampling": sampling.as_dict(), "data": {"sha256": "d" * 64}},
         "environment": {"host": {"fingerprint": "sha256:" + "h" * 64}, "build_info": "b1-926862e",
                         "server": {"position_bias_sha256": None}},
         "provenance": dict(PROV), "thinking": {"capable": True, "reason": None},
         "server_state": {"state": "clean"}, "audit": {"latency_model": "verified"}, "drift": {"flagged": False},
         "budgets": {"0": {"T": 1.62}, "256": {"T": 1.1}, "1024": {"T": 0.9}}, "pick": {"budget": 256}}
    d.update(over)
    return d


def test_a_valid_profile_loads_with_its_budgets_sampling_and_identity(tmp_path):
    p = tmp_path / "t.think.json"
    p.write_text(json.dumps(profile_dict()))
    tp = load_think_profile(p)
    assert tp.source == str(p) and len(tp.sha256) == 64 and tp.model_key == KEY and tp.model_name == "tiel"
    assert sorted(tp.budgets) == [0, 256, 1024] and tp.t0 == 1.62 and tp.capable and tp.pick == 256
    assert tp.sampling.trace == "1/0.95/20/0" and tp.sampling.policy == "model" and tp.sampling.origin["top_k"] == "gguf"
    assert tp.latency_model == "verified" and not tp.drift_flagged and tp.state == "clean"
    assert tp.host_fingerprint == "sha256:" + "h" * 64 and tp.build_info == "b1-926862e"
    assert tp.chat_template_sha256 == "t" * 64 and tp.position_bias_sha256 is None
    assert tp.info() == {"path": str(p), "sha256": tp.sha256, "budgets": [0, 256, 1024], "model_key": KEY,
                         "state": "clean", "latency_model": "verified", "host_match": None}
    assert tp.info("sha256:" + "h" * 64)["host_match"] is True and tp.info("sha256:" + "x" * 64)["host_match"] is False
    assert load_think_profile(profile_dict()).source == "<dict>"


def test_temperature_for_a_calibrated_an_uncalibrated_and_a_low_budget():
    tp = load_think_profile(profile_dict())
    assert tp.temperature_for(256) == (1.1, True) and tp.temperature_for(0) == (1.62, True)
    assert tp.temperature_for(512) == (1.1, False)             # the largest calibrated budget below it
    assert tp.temperature_for(2048) == (0.9, False)
    assert tp.temperature_for(64) == (1.1, False)              # below every positive budget: the smallest one
    only0 = load_think_profile(profile_dict(budgets={"0": {"T": 1.5}}, thinking={"capable": False}))
    assert only0.temperature_for(256) == (1.0, False) and not only0.capable and only0.t0 == 1.5


@pytest.mark.parametrize(("over", "why"), [
    ({"callo_think_profile": 2}, "callo_think_profile must be 1"),
    ({"callo_think_profile": 1.0}, "callo_think_profile must be 1"),
    ({"metrics": "callo-think-metrics-2"}, "metrics must be"),
    ({"model_key": "sha256:short"}, "model_key must be"),
    ({"budgets": {"256": {"T": 1.0}}}, 'budgets must be an object with a "0" entry'),
    ({"budgets": {"0": {"T": 1.0}, "x": {"T": 1.0}}}, "budget key 'x'"),
    ({"budgets": {"0": {"T": 1.0}, "40000": {"T": 1.0}}}, "budget key '40000'"),
    ({"budgets": {"0": {"T": 0}}}, "T is finite and positive"),
    ({"budgets": {"0": {"T": float("inf")}}}, "T is finite and positive"),
    ({"budgets": {"0": 1.0}}, "T is finite and positive"),
    ({"comparable": {"sampling": {"policy": "model", "pinned": {}}}}, "pinned lacks"),
    ({"comparable": {"sampling": {"policy": "warm", "pinned": resolve_sampling(parse_sampling("fixed")).pinned}}},
     "policy must be model, fixed or a spec"),
    ({"comparable": {"sampling": {"policy": "fixed", "pinned": {**resolve_sampling(parse_sampling("fixed")).pinned,
                                                                "top_k": 20.0}}}}, "top_k must be an int"),
    ({"comparable": {"sampling": {"policy": "fixed", "pinned": {**resolve_sampling(parse_sampling("fixed")).pinned,
                                                                "temperature": float("nan")}}}}, "must be finite"),
    ({"comparable": {"sampling": {"policy": "fixed", "pinned": {**resolve_sampling(parse_sampling("fixed")).pinned,
                                                                "n_predict": 3}}}}, "not a sampler field"),
    ({"comparable": {"sampling": {"policy": "fixed", "pinned": {**resolve_sampling(parse_sampling("fixed")).pinned,
                                                                "temperature": "0.6"}}}}, "must be a number"),
])
def test_phase_1_refusals_name_the_fault(over, why):
    with pytest.raises(InvalidInput, match=why):
        load_think_profile(profile_dict(**over))


def test_phase_1_refuses_an_unreadable_file_and_bad_json(tmp_path):
    with pytest.raises(InvalidInput, match="cannot read it"):
        load_think_profile(tmp_path / "missing.json")
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(InvalidInput, match="not valid JSON"):
        load_think_profile(p)
    p.write_text("[]")
    with pytest.raises(InvalidInput, match="not a JSON object"):
        load_think_profile(p)


def test_phase_2_identity_refusals():
    tp = load_think_profile(profile_dict())
    with pytest.raises(CalloError, match="native decision model"):
        check_think_identity(tp, model_key=KEY, native=True, capable=True)
    with pytest.raises(CalloError, match="measured on tiel"):
        check_think_identity(tp, model_key="sha256:" + "0" * 64, native=False, capable=True)
    with pytest.raises(CalloError, match="no thinking mode callo can read"):
        check_think_identity(tp, model_key=KEY, native=False, capable=False)
    check_think_identity(tp, model_key=KEY, native=False, capable=True)
    only0 = load_think_profile(profile_dict(budgets={"0": {"T": 1.5}}, thinking={"capable": False}))
    check_think_identity(only0, model_key=KEY, native=False, capable=False)         # budget-0-only: loads


def test_phase_2_warnings_each_name_their_cause():
    tp = load_think_profile(profile_dict())
    quiet = check_think_profile(tp, provenance=PROV, host_fingerprint="sha256:" + "h" * 64, build_info="b1-926862e",
                                position_bias_sha256=None, temperature_from_profile=True, default_budget=256)
    assert quiet == []
    loud = check_think_profile(tp, provenance={**PROV, "chat_template_sha256": "u" * 64, "cache_prompt": True,
                                               "build_info": "b2"},
                               host_fingerprint="sha256:" + "x" * 64, build_info="b2", position_bias_sha256="abc",
                               temperature_from_profile=True, default_budget=512, host_chip="Apple M4 Max")
    assert all(line.startswith("WARNING:") for line in loud)
    text = "\n".join(loud)
    assert "thinking prompt may differ: recalibrate" in text and "cache_prompt differs" in text
    assert ("its latencies were measured on another host and build (profile: unknown chip, b1-926862e; now: Apple M4 "
            "Max, b2)") in text and "calibrated budgets: 256, 1024" in text
    assert sum("b2" in line for line in loud) == 1                        # a build change warns once
    assert "T_0 was fitted on readouts debiased by no profile, and this server serves abc" in text
    busy = load_think_profile(profile_dict(server_state={"state": "busy"}, audit={"latency_model": "unverified"},
                                           drift={"flagged": True}, thinking={"capable": False}))
    text = "\n".join(check_think_profile(busy, provenance=PROV, host_fingerprint="sha256:" + "h" * 64,
                                          build_info="b1-926862e", position_bias_sha256=None,
                                          temperature_from_profile=False, default_budget=0))
    assert "busy server" in text and "unverified" in text and "drifted" in text and "budget 0 only" in text
    assert "T_0 was fitted" not in text                                   # --temperature given: no T_0 warning


def test_an_unknown_host_on_either_side_warns_that_it_cannot_tell_instead_of_matching():
    """A None fingerprint is "unknown host", never equal to another None."""
    unknown = "WARNING: --think-profile: unknown host: cannot tell whether the profile was measured on this machine"
    same = {"provenance": PROV, "build_info": "b1-926862e", "position_bias_sha256": None,
            "temperature_from_profile": True, "default_budget": 256}
    nofp = load_think_profile(profile_dict(environment={"host": {"fingerprint": None}, "build_info": "b1-926862e",
                                                        "server": {"position_bias_sha256": None}}))
    assert nofp.host_fingerprint is None
    assert check_think_profile(nofp, host_fingerprint=None, **same) == [unknown]          # None == None is no match
    assert check_think_profile(nofp, host_fingerprint="sha256:" + "h" * 64, **same) == [unknown]
    tp = load_think_profile(profile_dict())
    assert check_think_profile(tp, host_fingerprint=None, **same) == [unknown]
    lines = check_think_profile(tp, host_fingerprint=None, **{**same, "build_info": "b2"})
    assert lines == [unknown, "WARNING: --think-profile: its latencies were measured on another build (profile: "
                              "unknown chip, b1-926862e; now: unknown chip, b2)"]     # never "profile None"


def test_a_capable_profile_with_no_positive_budget_serves_at_t_1_and_names_no_calibrated_budgets():
    """Calibrated at budget 0 only (every positive budget failed or was not asked for), then served
    with --think-budget 256 on a capable model."""
    tp = load_think_profile(profile_dict(budgets={"0": {"T": 1.4}}))
    assert tp.capable and tp.temperature_for(256) == (1.0, False) and tp.t0 == 1.4
    lines = check_think_profile(tp, provenance=PROV, host_fingerprint="sha256:" + "h" * 64, build_info="b1-926862e",
                                position_bias_sha256=None, temperature_from_profile=True, default_budget=256)
    assert any("calibrated budgets: none" in line for line in lines)


def test_a_profile_calibrated_under_a_sampling_spec_loads_with_that_policy():
    s = resolve_sampling(parse_sampling("temp=0.7,top_k=-1"))
    tp = load_think_profile(profile_dict(comparable={"sampling": s.as_dict()}))
    assert tp.sampling.policy == "temp=0.7,top_k=-1" and tp.sampling.trace == "0.7/0.95/-1/0"


def test_a_profile_keeps_the_sampling_gguf_error_and_refuses_a_malformed_one():
    """Sampling.as_dict() adds gguf_error when the GGUF metadata could not be read."""
    from callo.think import GGUFSampling

    s = resolve_sampling(parse_sampling("model"), GGUFSampling(error="the GGUF metadata could not be read (x)"))
    tp = load_think_profile(profile_dict(comparable={"sampling": s.as_dict()}))
    assert tp.sampling.gguf_error == "the GGUF metadata could not be read (x)" and tp.sampling.gguf == {}
    assert tp.sampling.as_dict() == s.as_dict()
    assert load_think_profile(profile_dict()).sampling.gguf_error is None
    with pytest.raises(InvalidInput, match="gguf_error must be a string"):
        load_think_profile(profile_dict(comparable={"sampling": {**s.as_dict(), "gguf_error": 3}}))


def test_a_profile_whose_raw_gguf_holds_non_finite_floats_loads_from_a_dict_and_a_file(tmp_path):
    """The recorded GGUF dict can hold nan / inf (never pinned). A dict profile carrying them hashes
    without a crash and as strict JSON; a file carrying them stringified (the writer's form) or as bare NaN loads."""
    from callo.think import json_safe

    s = resolve_sampling(parse_sampling("model"), {"temp": float("nan"), "top_p": float("inf"), "top_k": 20})
    assert s.pinned["temperature"] == 0.6 and s.pinned["top_k"] == 20
    d = profile_dict(comparable={"sampling": s.as_dict()})
    tp = load_think_profile(d)
    assert math.isnan(tp.sampling.gguf["temp"]) and tp.sampling.gguf["top_p"] == float("inf")
    assert tp.sha256 == load_think_profile(json_safe(d)).sha256                 # the dict is hashed in its safe form
    safe = json_safe(d)
    assert safe["comparable"]["sampling"]["gguf"] == {"temp": "nan", "top_p": "inf", "top_k": 20}
    assert json.loads(json.dumps(safe, allow_nan=False)) == safe              # strict JSON
    assert json_safe({"a": [float("-inf"), 1.5, True, None, "x"]}) == {"a": ["-inf", 1.5, True, None, "x"]}
    p = tmp_path / "nan.think.json"
    p.write_text(json.dumps(safe, allow_nan=False))
    assert load_think_profile(p).sampling.gguf == {"temp": "nan", "top_p": "inf", "top_k": 20}
    p.write_text(json.dumps(d))                                              # Python's bare NaN / Infinity
    assert math.isnan(load_think_profile(p).sampling.gguf["temp"])


# ---- host lines, profile validation, the default-budget refusal ---------------------------------------------

def test_the_host_line_names_the_chips_the_profile_recorded():
    chip = profile_dict(environment={"host": {"fingerprint": "sha256:" + "h" * 64, "chip": "Apple M3 Ultra"},
                                     "build_info": "b1-926862e", "server": {"position_bias_sha256": None}})
    tp = load_think_profile(chip)
    assert tp.host_chip == "Apple M3 Ultra" and load_think_profile(profile_dict()).host_chip is None
    (line,) = check_think_profile(tp, provenance=PROV, host_fingerprint="sha256:" + "x" * 64, build_info="b1-926862e",
                                  position_bias_sha256=None, temperature_from_profile=True, default_budget=256,
                                  host_chip="Apple M4 Max")
    assert line == ("WARNING: --think-profile: its latencies were measured on another host (profile: Apple M3 Ultra, "
                    "b1-926862e; now: Apple M4 Max, b1-926862e)")


@pytest.mark.parametrize(("budgets", "default", "rule"), [
    ({"0": {"T": 1.6}, "256": {"T": 1.1}, "1024": {"T": 0.9}}, 512, "served at budget 256's temperature (the nearest lower)"),
    ({"0": {"T": 1.6}, "256": {"T": 1.1}, "1024": {"T": 0.9}}, 64,
     "served at budget 256's temperature (the smallest calibrated; none is lower)"),
    ({"0": {"T": 1.6}}, 64, "served at T = 1 (no positive budget is calibrated)")])
def test_the_uncalibrated_default_budget_warning_names_the_rule_it_is_served_under(budgets, default, rule):
    tp = load_think_profile(profile_dict(budgets=budgets))
    (line,) = check_think_profile(tp, provenance=PROV, host_fingerprint="sha256:" + "h" * 64, build_info="b1-926862e",
                                  position_bias_sha256=None, temperature_from_profile=True, default_budget=default)
    assert rule in line and line.endswith("and marked calibrated: false")
    assert tp.temperature_for(default)[0] == ({512: 1.1, 64: 1.1}.get(default) if len(budgets) > 1 else 1.0)


def test_only_the_bias_guard_s_provenance_fields_are_compared():
    tp = load_think_profile(profile_dict(provenance={**PROV, "future_field": "x"}))
    assert check_think_profile(tp, provenance=PROV, host_fingerprint="sha256:" + "h" * 64, build_info="b1-926862e",
                               position_bias_sha256=None, temperature_from_profile=True, default_budget=256) == []


def test_a_dict_profile_that_is_not_json_is_invalid_input_not_a_crash():
    with pytest.raises(InvalidInput, match="not JSON-serialisable"):
        load_think_profile({**profile_dict(), 0: "mixed key types"})
    cyclic = profile_dict()
    cyclic["self"] = cyclic
    with pytest.raises(InvalidInput, match="not JSON-serialisable"):
        load_think_profile(cyclic)


@pytest.mark.parametrize("policy", ["foo=bar", "temp=x", "temp=-1", "warm"])
def test_a_sampling_policy_that_is_no_policy_is_refused(policy):
    pinned = resolve_sampling(parse_sampling("fixed")).pinned
    with pytest.raises(InvalidInput, match="policy must be model, fixed or a spec"):
        load_think_profile(profile_dict(comparable={"sampling": {"policy": policy, "pinned": pinned}}))


@pytest.mark.parametrize("capable", ["false", 0, None, "yes"])
def test_thinking_capable_must_be_a_boolean(capable):
    with pytest.raises(InvalidInput, match="thinking.capable must be true or false"):
        load_think_profile(profile_dict(thinking={"capable": capable}))


def test_a_bias_profile_and_a_think_profile_never_load_as_each_other(tmp_path):
    from test_profile import profile_dict as bias_profile_dict

    from callo.profile import load_profile

    with pytest.raises(InvalidInput, match="callo_think_profile must be 1"):
        load_think_profile(bias_profile_dict())
    with pytest.raises(InvalidInput):
        load_profile(profile_dict())
    for name, doc, loader in (("b.json", bias_profile_dict(), load_think_profile), ("t.json", profile_dict(), load_profile)):
        f = tmp_path / name
        f.write_text(json.dumps(doc))
        with pytest.raises(InvalidInput):
            loader(f)


def test_resolve_sampling_records_a_published_gguf_error():
    s = resolve_sampling(parse_sampling("model"), {}, gguf_error="the GGUF metadata could not be read (x)")
    assert s.gguf_error == "the GGUF metadata could not be read (x)" and s.as_dict()["gguf_error"] == s.gguf_error
    assert resolve_sampling(parse_sampling("model"), {"temp": 1.0}).gguf_error is None


@pytest.mark.parametrize(("stats", "why"), [
    ({}, None), ({"think": None}, None), ({"think": {"default_budget": 0}}, None), ({"think": {"capable": True}}, None),
    ({"think": {"default_budget": 256}}, "the server thinks by default (--think-budget 256)"),
    ({"think": "x"}, "cannot tell whether the server thinks by default"),
    ({"think": {"default_budget": None}}, "default_budget is None, not an integer"),
    ({"think": {"default_budget": True}}, "not an integer")])
def test_think_default_refusal_reads_any_stats_without_a_traceback(stats, why):
    from callo.think import think_default_refusal

    got = think_default_refusal(stats)
    assert got is None if why is None else why in got


# ---- the latency model and the metrics -----------------------------------------------------------------------

from callo.think import (  # noqa: E402
    accuracy,
    bootstrap_percentiles,
    budget_latency,
    crossfit,
    ece_at,
    fit_T,
    nll_at,
    paired_bootstrap,
)


def test_budget_latency_is_exact_at_n_gen_and_interpolates_the_decode_for_a_cut():
    # a 400-token trace: 80 ms frame, 4800 ms wall of which 4000 ms decode (10 ms/token), the n_gen readout 25 ms
    assert budget_latency(80, 4800, 4000, 400, 400, 25, 2.0, 2.0) == 80 + 4800 + 25
    assert budget_latency(80, 4800, 4000, 400, 1024, 25, 2.0, 2.0) == 80 + 4800 + 25      # a budget past n_gen
    # the cut at 100: fixed 800 + decode 1000 + the readout, whose extra prefill (30 ms over the n_gen readout) is dropped
    assert budget_latency(80, 4800, 4000, 400, 100, 60, 32.0, 2.0) == 80 + 800 + 1000 + 30
    assert budget_latency(80, 4800, 4000, 400, 100, 60, 2.0, 2.0) == 80 + 800 + 1000 + 60    # no re-prefill: nothing dropped
    assert budget_latency(80, 4800, 4000, 400, 100, 20, 90.0, 2.0) == 80 + 800 + 1000 + 0    # never negative
    assert budget_latency(80, 4800, None, 400, 100, 20, None, None) == 80 + 4800 + 20        # no timings: the wall
    assert budget_latency(80, 4800, 4000, 0, 0, 20, None, None) == 80 + 4800 + 20
    assert budget_latency(80, 100, 4000, 400, 400, 25, 2.0, 2.0) == 80 + 4000 + 25        # planted timings > wall
    assert budget_latency(80, 100, 4000, 400, 200, 25, 2.0, 2.0) == 80 + 0 + 2000 + 25     # ...and the cut branch


ROWS = [([0.7, 0.2, 0.1], 0), ([0.2, 0.5, 0.3], 1), ([0.4, 0.4, 0.2], 1), ([0.1, 0.1, 0.8], 0)]


def test_accuracy_ties_go_to_the_first_index_and_the_nll_ece_t_wrappers_are_calibrates():
    assert accuracy(ROWS) == 0.5                      # rows 1 and 2 right? no: row 3 ties on index 0 (label 1): wrong
    assert accuracy([([0.5, 0.5], 0)]) == 1.0 and accuracy([([0.5, 0.5], 1)]) == 0.0
    from callo.calibrate import ece, fit_temperature, nll
    assert nll_at(ROWS, 1.3) == nll(ROWS, 1.3) and ece_at(ROWS, 0.8) == ece(ROWS, 0.8)
    assert fit_T(ROWS) == fit_temperature(ROWS)
    with pytest.raises(InvalidInput):
        accuracy([])


def test_crossfit_respects_clusters_and_falls_back_to_in_sample_with_one_cluster():
    rows = [([0.9, 0.1], 0), ([0.8, 0.2], 0), ([0.3, 0.7], 1), ([0.6, 0.4], 0)] * 3
    clusters = ["a", "a", "b", "b"] * 3
    nll_t, ece_t = crossfit(rows, clusters, seed=0)
    assert nll_t > 0 and 0 <= ece_t <= 1
    assert crossfit(rows, clusters, seed=0) == crossfit(rows, clusters, seed=0)
    one = crossfit(rows, ["a"] * len(rows), seed=0)
    t = fit_T(rows)
    assert one == (nll_at(rows, t), ece_at(rows, t))
    with pytest.raises(ValueError):
        crossfit(rows, ["a"], seed=0)


def test_crossfit_weights_the_two_halves_by_their_row_counts():
    # two clusters of 6 and 10 rows: whatever the shuffle, each half is one cluster, so the result is the per-row
    # mean of the out-of-fold scores, on the same scale as the in-sample NLL at T = 1
    p_rows = [([0.9, 0.1], 0)] * 4 + [([0.9, 0.1], 1)] * 2
    q_rows = [([0.6, 0.4], 0)] * 5 + [([0.3, 0.7], 1)] * 3 + [([0.55, 0.45], 1)] * 2
    rows, clusters = p_rows + q_rows, ["p"] * 6 + ["q"] * 10
    tp, tq = fit_T(p_rows), fit_T(q_rows)
    want_nll = (6 * nll_at(p_rows, tq) + 10 * nll_at(q_rows, tp)) / 16
    want_ece = (6 * ece_at(p_rows, tq) + 10 * ece_at(q_rows, tp)) / 16
    got = crossfit(rows, clusters, seed=0)
    assert got == pytest.approx((want_nll, want_ece), rel=1e-12)
    assert crossfit(rows, clusters, seed=5) == pytest.approx(got, rel=1e-12)
    unweighted = (nll_at(p_rows, tq) + nll_at(q_rows, tp)) / 2
    assert abs(got[0] - unweighted) > 1e-3
    t = fit_T(rows)                                   # out of sample: the halves fit different temperatures
    assert tp != tq and abs(got[0] - nll_at(rows, t)) > 1e-6 and abs(got[1] - ece_at(rows, t)) > 1e-6


def test_paired_bootstrap_is_deterministic_by_seed_pairs_by_index_and_resamples_clusters():
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    b = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    clusters = ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"]
    point, lo, hi = paired_bootstrap(a, b, clusters, seed=0)
    assert point == 0.25 and lo <= point <= hi and lo >= 0.0
    # cluster sums (1, 1, 0, 0) over 4 two-row clusters: the estimate is Binomial(4, 1/2) / 8, support 0..0.5;
    # a row-level bootstrap would reach 0.625 and beyond
    assert (lo, hi) == (0.0, 0.5)
    assert paired_bootstrap(a, b, clusters, seed=0) == paired_bootstrap(a, b, clusters, seed=0)
    import random as _random
    rng = _random.Random(3)
    big_a, big_b = [rng.random() for _ in range(80)], [rng.random() for _ in range(80)]
    big_c = [f"c{i // 2}" for i in range(80)]
    assert paired_bootstrap(big_a, big_b, big_c, seed=1) != paired_bootstrap(big_a, big_b, big_c, seed=0)
    same = paired_bootstrap(a, a, clusters, seed=0)
    assert same == (0.0, 0.0, 0.0)
    _, lo1, hi1 = paired_bootstrap(a, b, ["x"] * 8, seed=0)              # one cluster: nothing to resample
    assert (lo1, hi1) == (0.25, 0.25)
    with pytest.raises(ValueError):
        paired_bootstrap(a, b[:-1], clusters, seed=0)


def test_bootstrap_percentiles_are_nearest_rank_with_intervals_from_the_same_resampling():
    values = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    clusters = [f"c{i}" for i in range(10)]
    p50, lo, hi = bootstrap_percentiles(values, clusters, 0.5, seed=0)
    assert p50 == 500.0 and lo <= 500.0 <= hi and lo >= 100.0 and hi <= 1000.0
    p95 = bootstrap_percentiles(values, clusters, 0.95, seed=0)
    assert p95[0] == 1000.0 and p95[1] <= 1000.0 <= p95[2]
    assert bootstrap_percentiles(values, clusters, 0.5, seed=0) == bootstrap_percentiles(values, clusters, 0.5, seed=0)
    assert bootstrap_percentiles([7.0], ["a"], 0.95, seed=0) == (7.0, 7.0, 7.0)
    with pytest.raises(ValueError):
        bootstrap_percentiles([], [], 0.5, seed=0)


def test_bootstrap_resamples_clusters_within_strata_so_each_source_keeps_its_cluster_count():
    from callo.think import _bootstrap_weights
    # source "big" has 40 one-row clusters, source "small" 3 two-row clusters
    clusters = [f"b{i}" for i in range(40)] + ["s0", "s0", "s1", "s1", "s2", "s2"]
    strata = ["big"] * 40 + ["small"] * 6
    idx, w = _bootstrap_weights(clusters, strata, seed=0, resamples=500)
    assert w.shape == (500, 43)
    assert (w[:, :40].sum(axis=1) == 40).all() and (w[:, 40:].sum(axis=1) == 3).all()
    _, plain = _bootstrap_weights(clusters, None, seed=0, resamples=500)
    assert (plain[:, 40:].sum(axis=1) != 3).any()                     # unstratified: the source mix moves
    _, one = _bootstrap_weights(clusters, ["x"] * 46, seed=0, resamples=500)
    assert (one == plain).all()                                       # a single source is plain cluster resampling
    # the paired bootstrap: the difference is 1 on every "small" row and 0 on every "big" row, so with the mix fixed
    # every resample has the same mean
    a = [0.0] * 46
    b = [0.0] * 40 + [1.0] * 6
    point, lo, hi = paired_bootstrap(a, b, clusters, seed=0, strata=strata)
    assert point == pytest.approx(6 / 46) and lo == pytest.approx(point) and hi == pytest.approx(point)
    assert paired_bootstrap(a, b, clusters, seed=0)[2] > point + 0.01
    # the percentiles: one slow "small" row among 19 fast "big" ones; the nearest-rank p95 (rank 19 of 20) is fast
    # in every stratified resample, while plain resampling sometimes draws the slow cluster twice
    vals, cl, st = [1.0] * 19 + [1000.0], [f"b{i}" for i in range(19)] + ["s"], ["big"] * 19 + ["small"]
    assert bootstrap_percentiles(vals, cl, 0.95, seed=0, strata=st) == (1.0, 1.0, 1.0)
    assert bootstrap_percentiles(vals, cl, 0.95, seed=0)[2] == 1000.0
    with pytest.raises(ValueError, match="one length"):
        paired_bootstrap(a, b, clusters, seed=0, strata=strata[:-1])
    with pytest.raises(ValueError, match="spans sources"):
        _bootstrap_weights(["c", "c"], ["s1", "s2"], seed=0, resamples=10)


# ---- points, the frontier and the pick -----------------------------------------------------------------------

import dataclasses  # noqa: E402

from callo.think import Point, Preference, frontier, mark_approx, merge_identical, on_frontier  # noqa: E402
from callo.think import pick as pick_point  # noqa: E402


def pt(name, budget, acc, p50, p95=None, lo_hi=0.0, **kw):
    """A budget point whose mean equals its p50 (so the mean default orders points as the p50 tests expect)."""
    p95 = p50 * 1.3 if p95 is None else p95
    return Point(name=name, budget=budget, acc=acc, n=100, lat_mean=(p50, p50 - lo_hi, p50 + lo_hi),
                 lat_p50=(p50, p50 - lo_hi, p50 + lo_hi), lat_p95=(p95, p95 - lo_hi, p95 + lo_hi),
                 lat_p99=(p95, p95 - lo_hi, p95 + lo_hi), lat_max=p95, **kw)


class Noise:
    """A stand-in for `think.Paired`: `dacc` is the given Δacc function, `dscore` 100 × it minus the priced latency
    difference (enough for the picker, which reads Δscore only to report it)."""

    def __init__(self, ci):
        self.ci = ci

    def dacc(self, a, b):
        return self.ci(a, b)

    def dscore(self, a, b, *, stat, price):
        dl = price * (b.latency(stat)[0] - a.latency(stat)[0]) / 1000
        return tuple(100 * x - dl for x in self.ci(a, b))


def ci_zero(a, b):
    return (b.acc - a.acc, -0.05, 0.05)


def ci_tight(a, b):
    d = b.acc - a.acc
    return (d, d - 0.001, d + 0.001)


def test_labels_and_merge_identical_merge_only_one_models_consecutive_identical_budgets():
    a, b, c, d = pt("tiel", 256, 0.9, 1400), pt("tiel", 512, 0.9, 1400), pt("tiel", 1024, 0.9, 1400), pt("qwen", 256, 0.8, 4200)
    merged = merge_identical([a, b, c, d], same=lambda x, y: x.name == "tiel")
    assert [p.label for p in merged] == ["qwen@256", "tiel@256≡512≡1024"]
    assert merged[1].budgets == (256, 512, 1024) and merged[1].budget == 256 and merged[1].lat_p50[0] == 1400
    kept = merge_identical([a, b, c], same=lambda x, y: y.budget == 512)
    assert [p.label for p in kept] == ["tiel@256≡512", "tiel@1024"]
    assert pt("m", 0, 0.5, 250).label == "m@0"
    with pytest.raises(ValueError):
        pt("m", 0, 0.5, 250).latency("p99")


def test_frontier_marks_dominated_points_with_their_dominator_and_sorts_by_latency():
    pts = [pt("tiel", 0, 0.78, 270), pt("tiel", 256, 0.89, 1400), pt("tiel", 1024, 0.92, 1500),
           pt("qwen", 0, 0.75, 230), pt("qwen", 256, 0.78, 4200), pt("qwen", 1024, 0.88, 12700)]
    fr = frontier(pts)
    assert [p.label for p in fr] == ["qwen@0", "tiel@0", "tiel@256", "tiel@1024", "qwen@256", "qwen@1024"]
    by = {p.label: p.dominated_by for p in fr}
    assert by["qwen@0"] is None and by["tiel@0"] is None and by["tiel@256"] is None and by["tiel@1024"] is None
    assert by["qwen@256"] == "tiel@1024" and by["qwen@1024"] == "tiel@1024"   # the strongest dominator is named
    assert [p.label for p in on_frontier(fr)] == ["qwen@0", "tiel@0", "tiel@256", "tiel@1024"]
    p95 = frontier(pts, latency="p95")
    assert [p.label for p in p95][:2] == ["qwen@0", "tiel@0"]


def test_equal_accuracy_with_overlapping_latency_intervals_keeps_both_and_ties_on_a_single_point():
    a, b = pt("a", 256, 0.9, 1000, lo_hi=100), pt("b", 256, 0.9, 1050, lo_hi=100)
    fr = frontier([a, b])
    assert all(p.dominated_by is None for p in fr)                     # within noise on latency alone: both stay
    far = pt("b", 256, 0.9, 1500, lo_hi=100)
    assert {p.label: p.dominated_by for p in frontier([a, far])} == {"a@256": None, "b@256": "a@256"}
    assert frontier([a]) == [dataclasses.replace(a, dominated_by=None)]


def test_mark_approx_compares_each_frontier_point_with_its_faster_neighbour():
    fr = on_frontier(frontier([pt("m", 0, 0.75, 250), pt("m", 256, 0.78, 1400), pt("m", 1024, 0.90, 4000)]))
    marked = mark_approx(fr, ci_zero)
    assert [p.approx for p in marked] == [False, True, True]            # the first point has no faster neighbour
    tight = mark_approx(fr, ci_tight)
    assert [p.approx for p in tight] == [False, False, False]


def test_the_default_pick_is_the_fastest_within_noise_of_the_best_and_the_filters_apply():
    fr = on_frontier(frontier([pt("m", 0, 0.75, 250), pt("m", 256, 0.88, 1400), pt("m", 1024, 0.92, 4000)]))

    def ci(a, b):                       # 256 vs 1024 is within noise; 0 vs 1024 is not
        d = b.acc - a.acc
        return (d, d - 0.05, d + 0.05)

    p = pick_point(fr, paired=Noise(ci))
    assert p.point.label == "m@256" and p.best.label == "m@1024" and p.most_accurate.label == "m@1024"
    assert p.rule == "fastest within noise of the best"
    assert p.dacc_best == pytest.approx((-0.04, -0.09, 0.01)) and [q.label for q in p.front] == ["m@0", "m@256", "m@1024"]
    assert p.as_dict() == {"kind": "budget", "budget": 256, "budgets": [256], "deadline_ms": None, "cap": None,
                           "z": None, "model": "m", "label": "m@256", "rule": p.rule, "best": "m@1024", "most_accurate": "m@1024",
                           "dacc_best": pytest.approx([-0.04, -0.09, 0.01]), "dscore_best": None, "score": None,
                           "preference": {"latency": "mean", "price_pp_per_s": 0.0, "limits": [], "min_accuracy": None,
                                          "max_memory_gib": None},
                           "excluded": {}, "reason": None, "unsized": [], "would_pick": None, "would_best": None,
                           "latency": "mean", "max_latency_s": None, "min_accuracy": None}
    best = pick_point(fr, paired=Noise(ci_tight))
    assert best.point.label == "m@1024" and best.dacc_best == (0.0, 0.0, 0.0)   # nothing within noise: the best itself
    assert pick_point(fr, paired=Noise(ci), pref=Preference(max_latency_s=0.5)).point.label == "m@0"
    none = pick_point(fr, paired=Noise(ci), pref=Preference(max_latency_s=0.1))
    assert none.point is None and none.rule == "no pick" and none.best is None
    assert pick_point(fr, paired=Noise(ci), pref=Preference(min_accuracy=0.8)).point.label == "m@256"
    assert pick_point(fr, paired=Noise(ci), pref=Preference(min_accuracy=0.8)).rule == "min-accuracy"
    assert pick_point(fr, paired=Noise(ci), pref=Preference(min_accuracy=0.99)).rule == "no pick"
    heavy = [dataclasses.replace(p, gib=30.0 if p.budget else 2.0) for p in fr]
    assert pick_point(heavy, paired=Noise(ci), pref=Preference(max_gib=10.0)).point.label == "m@0"
    empty = pick_point([], paired=Noise(ci))
    assert empty.point is None and empty.rule == "no pick" and empty.reason == "no points"
    merged = dataclasses.replace(p, point=dataclasses.replace(fr[1], budgets=(256, 512, 1024)))
    assert merged.as_dict()["budget"] == 1024 and merged.as_dict()["budgets"] == [256, 512, 1024]   # the largest serves


def test_the_filters_apply_before_the_frontier_so_a_model_that_does_not_fit_never_eliminates_one_that_does():
    big, small = pt("big", 0, 0.95, 300, gib=60.0), pt("small", 0, 0.90, 320, gib=5.0)
    assert pick_point([big, small], paired=Noise(ci_tight)).point.label == "big@0"      # unfiltered: small is dominated
    p = pick_point([big, small], paired=Noise(ci_tight), pref=Preference(max_gib=10.0))
    assert p.point.label == "small@0" and p.best.label == "small@0" and p.dacc_best == (0.0, 0.0, 0.0)
    assert [q.label for q in p.front] == ["small@0"]                              # big never entered the frontier
    slow = pt("m", 1024, 0.99, 5000)
    q = pick_point([big, small, slow], paired=Noise(ci_tight), pref=Preference(max_latency_s=1.0))   # --max-latency
    assert q.point.label == "big@0" and q.best.label == "big@0" and [x.label for x in q.front] == ["big@0"]
    assert pick_point([big, small, slow], paired=Noise(ci_tight)).point.label == "m@1024"
    r = pick_point([big, small, slow], paired=Noise(ci_zero), pref=Preference(max_gib=10.0))   # small alone fits
    assert r.point.label == "small@0" and r.rule == "fastest within noise of the best"


def test_max_memory_excludes_points_of_unknown_size_and_names_them_without_it_they_stay():
    big, small = pt("big", 0, 0.95, 300), pt("small", 0, 0.90, 320, gib=5.0)        # big's size is unknown
    p = pick_point([big, small], paired=Noise(ci_tight), pref=Preference(max_gib=10.0))
    assert p.point.label == "small@0" and p.best.label == "small@0" and [q.label for q in p.front] == ["small@0"]
    assert p.unsized == ["big@0"] and p.as_dict()["unsized"] == ["big@0"]       # the caller can name it
    assert p.excluded == {"big@0": "size unknown (--max-memory)"}
    loose = pick_point([big, small], paired=Noise(ci_tight))                     # no --max-memory: unknown stays in
    assert loose.point.label == "big@0" and loose.unsized == [] and loose.as_dict()["unsized"] == []
    alone = pick_point([big], paired=Noise(ci_tight), pref=Preference(max_gib=10.0))
    assert alone.point is None and alone.unsized == ["big@0"]
    assert alone.reason == "no point meets --max-memory 10: all 1 excluded; every size unknown"


def test_pick_ties_min_accuracy_and_the_p95_axis():
    a, b = pt("a", 0, 0.9, 1000, lo_hi=100), pt("b", 0, 0.9, 1050, lo_hi=100)  # equal accuracy, overlapping: both stay
    tie = pick_point([b, a], paired=Noise(ci_tight))
    assert [q.label for q in tie.front] == ["a@0", "b@0"] and tie.best.label == "a@0" and tie.point.label == "a@0"
    fr = [pt("m", 0, 0.75, 250), pt("m", 256, 0.88, 1400, p95=3000), pt("m", 1024, 0.92, 4000)]
    at_m = pick_point(fr, paired=Noise(ci_tight), pref=Preference(min_accuracy=0.92))   # the minimum is M's accuracy
    assert at_m.point.label == "m@1024" and at_m.rule == "min-accuracy" and at_m.dacc_best == (0.0, 0.0, 0.0)
    assert pick_point(fr, paired=Noise(ci_tight), pref=Preference(max_latency_s=2.0)).point.label == "m@256"  # 1.4 s
    p95 = pick_point(fr, paired=Noise(ci_tight), pref=Preference(latency="p95", max_latency_s=2.0))   # 3 s does not
    assert p95.point.label == "m@0" and [q.label for q in p95.front] == ["m@0"]


def test_the_overlap_exception_is_for_equal_accuracy_only():
    a, b = pt("a", 256, 0.90, 1050, lo_hi=100), pt("b", 256, 0.91, 1000, lo_hi=100)  # overlapping, b more accurate
    assert {p.label: p.dominated_by for p in frontier([a, b])} == {"b@256": None, "a@256": "b@256"}
