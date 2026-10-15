# Adapted from the author's benchmark harness (tests/test_system1_engine.py @ f886c50f).
# Edits: sys.path shim removed, imports retargeted, explanatory comments trimmed; FakeScorer.next_logprobs takes id_slot;
#   one top_tokens assertion added to test_engine_reads_the_answer_slot; tests added for id_slot, token_str and the GGUF meta cache
#   (incl. corrupt/old-schema cache and pid-unique temp file); v0.2 tests added for image and context readouts
#   and for local_engine; v0.3 tests added for the position-bias profile (offsets, Readout.bias, local_engine,
#   raw probs on ineligible readouts, the image path); v0.3 media: ChatMLTemplate renders audio/video parts
#   as the marker; tests for the audio, video and mixed readouts, media=, the deprecated image aliases, per-kind 501s,
#   placeholder rendering and Readout.passes; the media error table (the v0.2 context-size row now expects the 413
#   MediaTooLarge), check_request per kind and local_engine's detect_media; think budgets: local_engine refuses a
#   positive think_budget by name.
import io
import json
import logging
import math
import os
import string
import urllib.error
from pathlib import Path

import pytest
from fake_upstream import MEDIA_MARKER, FakeUpstream
from test_profile import KEY, profile_dict

from callo.engine import (
    Engine,
    Readout,
    SurfaceSets,
    bounded_readout,
    classify_image_error,
    classify_media_error,
    load_gguf_meta,
    local_engine,
    read_gguf_meta,
    self_test,
    vocab_cache_key,
)
from callo.errors import (
    CalloError,
    ImageRejected,
    InvalidInput,
    MediaLoadFailed,
    MediaRejected,
    MediaTooLarge,
    NoAudio,
    NoVideo,
    NoVision,
    ReadoutError,
    UpstreamRejected,
    UpstreamUnavailable,
)
from callo.modelkey import load_model_key
from callo.profile import load_profile
from callo.prompt import PLACEHOLDER_IMAGE_URL, PLACEHOLDER_PAYLOAD, TemplateError, messages
from callo.scorer import MMResult
from callo.systemone import InvalidRequest, MediaItem, parse_request

# A toy byte-level vocabulary: ids 0..9. "A"/" A"/"a" collect into A; "**" is a shared prefix token.
TOKENS = ["A", "ĠA", "a", "B", "ĠB", "C", "**", "Ġthe", "x", "D"]
SETS = SurfaceSets(TOKENS, "gpt2")
CHOICE = parse_request({"state": "s", "questions": {"q": {"type": "choice",
                        "criteria": {"a": "one", "b": "two", "c": "three"}}}}).questions[0]


def lp(p):
    return math.log(p)


class FakeScorer:
    """dists maps a prefix length (after the 3 prompt tokens) to a sparse top-k dump."""

    def __init__(self, dists):
        self.dists, self.calls = dists, []

    def tokenize(self, text, *, add_special=True):
        return [7, 8, 7]

    def next_logprobs(self, tokens, *, top_k, id_slot=None, delimiter=None):
        self.calls.append((tuple(tokens), top_k, id_slot))
        return self.dists[len(tokens) - 3]


class FakeTemplate:
    def render(self, msgs):
        return "PROMPT"


def test_surface_sets_sum_spellings_and_are_memoised():
    s = SETS.get(("A", "B", "C"))
    assert s["A"] == frozenset({0, 1, 2}) and s["B"] == frozenset({3, 4}) and s["C"] == frozenset({5})
    again = SETS.get(("A", "B", "C"))
    assert again == s and again is not s


def test_surface_sets_permuted_order_hits_the_cache(monkeypatch):
    import callo.engine as eng
    calls = []
    real = eng.option_surface_sets

    def counting(tokens, model, labels):
        calls.append(tuple(labels))
        return real(tokens, model, labels)

    monkeypatch.setattr(eng, "option_surface_sets", counting)
    sets = SurfaceSets(TOKENS, "gpt2")
    first = sets.get(("A", "B", "C"))
    rotated = sets.get(("C", "A", "B"))
    assert len(calls) == 1
    assert list(rotated) == ["C", "A", "B"] and rotated == first
    rotated["C"] = frozenset()
    assert sets.get(("A", "B", "C"))["C"] == frozenset({5})
    assert len(calls) == 1


def test_surface_sets_refuse_a_label_with_no_token():
    with pytest.raises(InvalidInput, match="NO surface form"):
        SurfaceSets(TOKENS, "gpt2").get(("A", "Q"))


def test_bounded_readout_sums_to_one_and_bounds_missing_options():
    marg = {"A": lp(0.5), "B": lp(0.3), "C": float("-inf")}
    top = {0: lp(0.3), 1: lp(0.2), 3: lp(0.3), 8: lp(0.05)}
    probs, lms, miss = bounded_readout(marg, top, ("A", "B", "C"))
    assert miss == 1 and lms[2] == lp(0.05)
    assert abs(sum(probs) - 1) < 1e-12 and all(p > 0 for p in probs)
    assert abs(probs[0] - 0.5 / 0.85) < 1e-12


def test_temperature_flattens_without_changing_argmax():
    marg = {"A": lp(0.8), "B": lp(0.2)}
    p1, _, _ = bounded_readout(marg, {0: lp(0.8)}, ("A", "B"))
    p2, _, _ = bounded_readout(marg, {0: lp(0.8)}, ("A", "B"), temperature=2.0)
    assert p2[0] < p1[0] and p2[0] > p2[1]


def test_engine_reads_the_answer_slot():
    sc = FakeScorer({0: {0: lp(0.6), 1: lp(0.1), 3: lp(0.2), 5: lp(0.05), 8: lp(0.05)}})
    r = Engine(sc, FakeTemplate(), SETS).answer("s", CHOICE)
    assert r.stop == "diverged" and r.depth == 0 and r.topk_miss == 0 and r.prompt_tokens == 3
    assert abs(r.probs[0] - 0.7 / 0.95) < 1e-12
    assert sc.calls[0][1] == 100
    assert r.top_tokens[0] == (0, lp(0.6)) and len(r.top_tokens) == 5


def test_engine_walks_a_shared_prefix():
    sc = FakeScorer({0: {6: lp(0.95), 8: lp(0.05)},
                     1: {5: lp(0.9), 0: lp(0.05), 3: lp(0.05)}})
    r = Engine(sc, FakeTemplate(), SETS).answer("s", CHOICE)
    assert r.depth == 1 and r.stop == "diverged" and r.probs.index(max(r.probs)) == 2


def test_engine_reports_low_mass_without_crashing():
    sc = FakeScorer({0: {7: lp(0.46), 8: lp(0.46), 0: lp(0.04), 3: lp(0.04)}})
    r = Engine(sc, FakeTemplate(), SETS).answer("s", CHOICE)
    assert r.stop == "no_shared_prefix" and r.option_mass < 0.5 and r.topk_miss == 1


def _near_certain_dump(excess=5e-7):
    """A many-term near-certain dump over SETS' option ids (0, 1, 2 -> a; 3, 4 -> b; 5 -> c) whose probabilities
    sum to 1 + excess in float64, every logprob <= 0: the shape float32 rounding of many terms pushes past 1."""
    small = {1: 1e-5, 2: 2e-5, 3: 3e-5, 4: 1e-5, 5: 2e-5}
    probs = {0: 1.0 + excess - math.fsum(small.values()), **small}
    assert all(p <= 1.0 for p in probs.values())
    return {tid: math.log(p) for tid, p in probs.items()}


def test_a_near_certain_answer_reads_out_clamped_and_every_downstream_number_stays_a_probability():
    """The live failure (Gemma 4 E4B, MiniCPM5-2B): llama.cpp emits only logprobs <= 0, but a near-certain answer's
    many float32 terms (each p <= 1) sum a hair past 1, so the option mass is 1 + ~5e-7. It must read out, with the
    mass clamped to 1, no log-marginal above 0, probabilities in [0, 1] summing to 1, and a finite NLL and ECE."""
    from callo.calibrate import ece, nll
    dump = _near_certain_dump()
    assert all(v <= 0.0 for v in dump.values())
    assert math.fsum(math.exp(v) for v in dump.values()) - 1.0 > 1e-9      # above the parent commit's 1e-9 slack
    r = Engine(FakeScorer({0: dump}), FakeTemplate(), SETS).answer("s", CHOICE)
    assert r.stop == "diverged" and r.option_mass == 1.0 and r.probs.index(max(r.probs)) == 0
    assert all(v <= 0.0 for v in r.log_marginals)
    assert all(0.0 <= p <= 1.0 for p in r.probs) and abs(math.fsum(r.probs) - 1.0) < 1e-12
    assert 1.0 - r.option_mass == 0.0          # the "other" share: never negative
    rows = [(list(r.probs), 0), (list(r.probs), 1)]
    assert all(math.isfinite(x) and x >= 0.0 for x in (nll(rows, 1.0), ece(rows, 1.0), nll(rows, 2.0)))


def test_a_single_positive_logprob_reads_out_clamped_defensively():
    """DEFENSIVE, not a reproduction: llama.cpp cannot emit a positive logprob, but one up to 1e-6 above 0 is legal
    input to the marginaliser, so the readout still clamps it."""
    r = Engine(FakeScorer({0: {0: 5e-7, 3: lp(1e-9), 5: lp(1e-11), 8: lp(1e-12)}}), FakeTemplate(), SETS).answer("s", CHOICE)
    assert r.option_mass == 1.0 and r.log_marginals[0] == 0.0 and all(v <= 0.0 for v in r.log_marginals)


def test_the_retok_fallback_clamps_a_mass_just_above_one():
    """The `_RetokStop` fallback of `Engine._walk` re-reads the last scored position through `option_mass` and
    `clamp_marginals`: a mass slightly above 1 is accepted and clamped to 1, and the log-marginals are <= 0."""
    from callo.engine import _RetokStop
    dump = _near_certain_dump()
    assert math.fsum(math.exp(v) for v in dump.values()) > 1.0

    def dist_fn(prefix):
        raise _RetokStop

    passes = [((), MMResult(dump, 400, 38, 62, 15.5))]
    r = Engine(FakeScorer({}), FakeTemplate(), SETS)._walk(CHOICE, dist_fn, (), passes, kind="plain", prompt_tokens=400)
    assert r.stop == "retok" and r.depth == 0 and r.option_mass == 1.0
    assert all(v <= 0.0 for v in r.log_marginals) and all(0.0 <= p <= 1.0 for p in r.probs)
    assert abs(math.fsum(r.probs) - 1.0) < 1e-12


def test_single_option_needs_no_model_call():
    q = parse_request({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"only": None}}}}).questions[0]
    sc = FakeScorer({})
    r = Engine(sc, FakeTemplate(), SETS).answer("s", q)
    assert r.probs == (1.0,) and r.stop == "single_option" and sc.calls == []


def test_divergence_mass_above_default_dominance_is_accepted():
    sc = FakeScorer({0: {0: lp(0.9), 3: lp(0.1)}})
    r = Engine(sc, FakeTemplate(), SETS, divergence_mass=0.6).answer("s", CHOICE)
    assert r.stop == "diverged"


# ---- callo additions ---------------------------------------------------------------------------------

def test_engine_pins_every_forward_pass_to_the_given_slot():
    sc = FakeScorer({0: {6: lp(0.95), 8: lp(0.05)}, 1: {5: lp(0.9), 0: lp(0.05), 3: lp(0.05)}})
    Engine(sc, FakeTemplate(), SETS).answer("s", CHOICE, id_slot=0)
    assert [c[2] for c in sc.calls] == [0, 0]          # the walk's second pass is pinned too
    sc2 = FakeScorer({0: {0: lp(0.9), 3: lp(0.1)}})
    Engine(sc2, FakeTemplate(), SETS).answer("s", CHOICE)
    assert sc2.calls[0][2] is None


def test_surface_sets_token_str():
    assert SETS.token_str(1) == "ĠA" and SETS.token_str(999) == "<id 999>"


def _write_tiny_gguf(path, *, decision=None):
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(list(string.ascii_uppercase) + ["Ġ" + c for c in string.ascii_uppercase] + ["**", "Ġthe", "x"])
    if decision:
        w.add_string("qwen35.decision.type", decision)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_read_gguf_meta_reads_vocab_architecture_and_decision_type(tmp_path):
    plain, kev = tmp_path / "plain.gguf", tmp_path / "kev.gguf"
    _write_tiny_gguf(plain)
    _write_tiny_gguf(kev, decision="kev")
    m = read_gguf_meta(plain)
    assert m.architecture == "qwen35" and m.tok_model == "gpt2" and m.decision_type is None
    assert m.tokens[:2] == ["A", "B"] and len(m.tokens) == 55
    assert read_gguf_meta(kev).decision_type == "kev"


def test_read_gguf_meta_rejects_a_file_without_a_vocabulary(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUF" + b"\0" * 64)
    with pytest.raises(InvalidInput):
        read_gguf_meta(bad)


def test_load_gguf_meta_caches_by_path_size_and_mtime(tmp_path):
    g = tmp_path / "m.gguf"
    _write_tiny_gguf(g)
    cache = tmp_path / "cache"
    key = vocab_cache_key(g)
    m1 = load_gguf_meta(g, cache)
    cached = cache / f"vocab-{key}.json"
    assert cached.exists() and json.loads(cached.read_text())["tokens"][:1] == ["A"]
    cached.write_text(json.dumps({"tokens": ["Z"], "tok_model": "gpt2", "architecture": "qwen35", "decision_type": None}))
    assert load_gguf_meta(g, cache).tokens == ["Z"]            # served from the cache, GGUF not re-read
    _write_tiny_gguf(g, decision="lev")                         # size/mtime change -> new key, cache miss
    m3 = load_gguf_meta(g, cache)
    assert m3.decision_type == "lev" and m3.tokens == m1.tokens
    assert load_gguf_meta(g, None).tokens == m1.tokens        # no cache dir: plain read


@pytest.mark.parametrize("junk", [
    b"{\"tokens\": [\"A\", \"B\"",                                               # half-written / corrupt JSON
    b"\xff\xfe not utf-8",                                                         # not even text
    b"[1, 2, 3]",                                                                  # JSON, wrong shape
    json.dumps({"tokens": ["Z"], "tok_model": "gpt2", "architecture": "qwen35"}).encode(),   # old schema: key missing
    json.dumps({"tokens": "ABC", "tok_model": "gpt2", "architecture": "qwen35", "decision_type": None}).encode(),
    json.dumps({"tokens": ["Z"], "tok_model": None, "architecture": "qwen35", "decision_type": None}).encode(),
], ids=["truncated", "not-utf8", "wrong-shape", "old-schema", "tokens-not-a-list", "tok-model-not-a-str"])
def test_a_bad_vocab_cache_is_a_miss_and_is_rewritten(tmp_path, junk):
    g = tmp_path / "m.gguf"
    _write_tiny_gguf(g)
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = cache / f"vocab-{vocab_cache_key(g)}.json"
    cached.write_bytes(junk)
    m = load_gguf_meta(g, cache)                                # re-reads the GGUF instead of crashing
    assert m == read_gguf_meta(g)
    assert json.loads(cached.read_text())["tokens"] == m.tokens   # and repairs the cache
    assert load_gguf_meta(g, cache) == m


def test_the_vocab_cache_is_written_through_a_pid_unique_temp_file(tmp_path, monkeypatch):
    """Two callos starting on the same GGUF at once must not `replace` each other's half-written file into place."""
    g = tmp_path / "m.gguf"
    _write_tiny_gguf(g)
    cache = tmp_path / "cache"
    seen = []
    real_replace = Path.replace

    def spy(self, target):
        seen.append(self.name)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", spy)
    load_gguf_meta(g, cache)
    assert len(seen) == 1 and str(os.getpid()) in seen[0] and seen[0].endswith(".tmp")
    assert [p.name for p in cache.iterdir()] == [f"vocab-{vocab_cache_key(g)}.json"]   # no temp file left behind


# ---- v0.2: image and context readouts -----------------------------------------------------------------

MARKER = "<__media_x__>"
DELIM = "<|im_end|>\n<|im_start|>user"
PNG = "iVBORw0KGgo="
URL = f"data:image/png;base64,{PNG}"


MEDIA_TYPES = ("image_url", "input_audio", "input_video", "video_url")


def _content(c):
    if isinstance(c, str):
        return c
    out, last = "", False
    for p in c:
        out += MARKER if p["type"] in MEDIA_TYPES else ("\n" if out and not last else "") + p["text"]
        last = p["type"] in MEDIA_TYPES
    return out


class ChatMLTemplate:
    """Renders like llama-server's /apply-template: every media part (image, audio, video) becomes the media marker."""

    def __init__(self, drop_images=False):
        self.drop_images, self.seen = drop_images, []

    def render(self, msgs):
        self.seen.append(msgs)
        if self.drop_images:
            msgs = [{**m, "content": [p for p in m["content"] if p["type"] != "image_url"]}
                    if isinstance(m["content"], list) else m for m in msgs]
        return "".join(f"<|im_start|>{m['role']}\n{_content(m['content'])}<|im_end|>\n" for m in msgs) + \
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class MMScorer:
    """dists maps the walk depth (number of appended tokens) to a sparse top-k dump. `vocab` maps special or
    multi-character spellings to one id (matched only with parse_special for `<|...|>` spellings); any other
    character tokenizes to 1000 + ord(c). `detok` maps an id to its text."""

    def __init__(self, dists, *, detok=None, vocab=None):
        self.dists, self.detok, self.vocab = dists, detok or {}, vocab or {}
        self.mm, self.cached, self.plain, self.tok_calls, self.detok_calls = [], [], [], [], []

    def tokenize(self, text, *, add_special=True, parse_special=None):
        self.tok_calls.append((text, add_special, parse_special))
        out, i = [], 0
        while i < len(text):
            hit = next((s for s in sorted(self.vocab, key=len, reverse=True)
                        if text.startswith(s, i) and (parse_special or not s.startswith("<|"))), None)
            out.append(self.vocab[hit] if hit else 1000 + ord(text[i]))
            i += len(hit) if hit else 1
        return out

    def detokenize(self, tokens):
        self.detok_calls.append(tuple(tokens))
        return "".join(self.detok[t] for t in tokens)

    def next_logprobs(self, tokens, *, top_k, id_slot=None, delimiter=None):
        self.plain.append(tuple(tokens))
        return self.dists[0]

    def _result(self, depth, n):
        return MMResult(self.dists[depth], 400 + depth, 350 if depth else 38, n, 15.5)

    def next_logprobs_mm(self, prompt_string, images, *, top_k, id_slot=None, delimiter=None):
        self.mm.append((prompt_string, tuple(images), top_k, id_slot, delimiter))
        return self._result(len(self.mm) - 1, 62)

    def next_logprobs_cached(self, tokens, *, top_k, id_slot=None, delimiter=None):
        self.cached.append((tuple(tokens), top_k, id_slot, delimiter))
        return self._result(len(self.cached) - 1, 30)


def _vision_engine(sc, template=None, **kw):
    return Engine(sc, template or ChatMLTemplate(), SETS, vision=True, media_marker=MARKER, user_delimiter=DELIM, **kw)


DIVERGED = {0: lp(0.6), 1: lp(0.1), 3: lp(0.2), 5: lp(0.05), 8: lp(0.05)}


def test_an_image_request_sends_the_rendered_prompt_and_payloads_and_reads_tokens_evaluated():
    sc, t = MMScorer({0: DIVERGED}), ChatMLTemplate()
    r = _vision_engine(sc, t).answer("s", CHOICE, id_slot=0, context="orders", images=[PNG], image_urls=[URL])
    prompt, images, top_k, slot, delim = sc.mm[0]
    assert images == (PNG,) and top_k == 100 and slot == 0 and delim == DELIM
    assert prompt == t.render(messages("s", CHOICE, context="orders", image_urls=[URL]))
    assert prompt.index("Context:\norders") < prompt.index(MARKER) < prompt.index("Options:")
    assert r.kind == "image" and r.stop == "diverged" and r.depth == 0
    assert r.prompt_tokens == 400 and (r.cache_n, r.prompt_n, r.prompt_ms) == (38, 62, 15.5)
    assert abs(r.probs[0] - 0.7 / 0.95) < 1e-12 and sc.plain == [] and sc.cached == []


def test_the_walk_appends_text_and_resends_the_same_images_after_a_passing_round_trip():
    sc = MMScorer({0: {6: lp(0.95), 8: lp(0.05)}, 1: {5: lp(0.9), 0: lp(0.05), 3: lp(0.05)}},
                  detok={6: "**"}, vocab={"**": 6})
    r = _vision_engine(sc).answer("s", CHOICE, images=[PNG, PNG], image_urls=[URL, URL])
    assert r.stop == "diverged" and r.depth == 1 and r.probs.index(max(r.probs)) == 2
    assert sc.mm[1][0] == sc.mm[0][0] + "**" and sc.mm[1][1] == sc.mm[0][1] == (PNG, PNG)
    assert sc.detok_calls == [(6,)]
    tail = sc.mm[0][0][sc.mm[0][0].rfind(MARKER) + len(MARKER):]
    assert (tail, False, True) in sc.tok_calls and (tail + "**", False, True) in sc.tok_calls
    assert r.cache_n == 38                                       # the FIRST pass's numbers, not the walk's


def test_a_failing_round_trip_stops_with_retok_at_the_previous_position():
    first = {6: lp(0.95), 8: lp(0.05)}
    sc = MMScorer({0: first, 1: {5: lp(0.9), 0: lp(0.1)}}, detok={6: "\n"}, vocab={"**": 6})   # "\n" != token 6
    r = _vision_engine(sc).answer("s", CHOICE, images=[PNG], image_urls=[URL])
    assert r.stop == "retok" and r.depth == 0 and len(sc.mm) == 1
    assert r.option_mass == 0.0 and r.topk_miss == 3                # read out where the walk stood: no option mass


def test_a_replacement_character_stops_the_walk():
    sc = MMScorer({0: {6: lp(0.95), 8: lp(0.05)}}, detok={6: "�"}, vocab={"�": 6})
    r = _vision_engine(sc).answer("s", CHOICE, images=[PNG], image_urls=[URL])
    assert r.stop == "retok" and len(sc.mm) == 1


def test_a_control_token_step_round_trips_as_unescaped_text():
    """/detokenize renders control tokens as text; parse_special=true turns that text back into the same token, so
    the walk text must never be special-token escaped."""
    sc = MMScorer({0: {7: lp(0.95), 8: lp(0.05)}, 1: {5: lp(0.9), 0: lp(0.1)}},
                  detok={7: "<|im_end|>"}, vocab={"<|im_end|>": 7})
    r = _vision_engine(sc).answer("s", CHOICE, images=[PNG], image_urls=[URL])
    assert r.stop == "diverged" and r.depth == 1
    assert sc.mm[1][0].endswith("<|im_end|>") and "<¦im_end¦>" not in sc.mm[1][0]


def test_marker_count_or_position_mismatch_is_a_readout_error():
    sc = MMScorer({0: DIVERGED})
    with pytest.raises(ReadoutError, match="did not render the image parts"):
        _vision_engine(sc, ChatMLTemplate(drop_images=True)).answer("s", CHOICE, images=[PNG], image_urls=[URL])

    class TextFirst(ChatMLTemplate):
        def render(self, msgs):
            return super().render(msgs).replace(MARKER, "") + MARKER

    with pytest.raises(ReadoutError, match="did not render the image parts"):
        _vision_engine(sc, TextFirst()).answer("s", CHOICE, images=[PNG], image_urls=[URL])
    with pytest.raises(ReadoutError):                                # two images, one marker
        _vision_engine(sc).answer("s", CHOICE, images=[PNG, PNG], image_urls=[URL])
    assert sc.mm == []


def test_an_image_request_without_vision_is_refused_before_any_upstream_call():
    sc = MMScorer({0: DIVERGED})
    with pytest.raises(NoVision, match="--mmproj"):
        Engine(sc, ChatMLTemplate(), SETS).answer("s", CHOICE, images=[PNG], image_urls=[URL])
    assert sc.mm == [] and sc.tok_calls == []


def test_a_context_request_sends_cached_token_arrays_with_the_delimiter():
    sc, t = MMScorer({0: DIVERGED}), ChatMLTemplate()
    r = _vision_engine(sc, t).answer("s", CHOICE, id_slot=1, context="LEGEND")
    tokens, top_k, slot, delim = sc.cached[0]
    prompt = t.render(messages("s", CHOICE, context="LEGEND"))
    assert tokens == tuple(sc.tokenize(prompt)) and slot == 1 and delim == DELIM and top_k == 100
    assert r.kind == "context" and r.prompt_tokens == len(tokens) and r.cache_n == 38 and r.prompt_n == 30
    assert sc.mm == [] and sc.plain == []


def test_a_request_with_neither_takes_the_v01_path():
    sc = MMScorer({0: DIVERGED})
    r = _vision_engine(sc).answer("s", CHOICE, id_slot=0)
    assert len(sc.plain) == 1 and sc.mm == [] and sc.cached == []
    assert r.kind == "plain" and r.cache_n is None and r.prompt_n is None


def test_render_prompt_is_the_string_the_image_request_sends():
    sc, t = MMScorer({0: DIVERGED}), ChatMLTemplate()
    e = _vision_engine(sc, t)
    assert e.render_prompt("s", CHOICE, context="c", image_urls=[URL]) == \
        t.render(messages("s", CHOICE, context="c", image_urls=[URL]))
    e.answer("s", CHOICE, context="c", images=[PNG], image_urls=[URL])
    assert sc.mm[0][0] == e.render_prompt("s", CHOICE, context="c", image_urls=[URL])


# ---- v0.2: answer_request and the error contract ---------------------------------------------------------

TWO_Q = {"q": {"type": "choice", "criteria": {"a": "one", "b": "two", "c": "three"}},
         "o": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}}


def test_answer_request_returns_exactly_the_systemone_body():
    sc = MMScorer({0: DIVERGED, 1: DIVERGED})
    readouts = []
    out = _vision_engine(sc, model_name="tiel").answer_request(
        {"state": "s", "images": [URL], "questions": TWO_Q}, id_slot=0, readouts=readouts)
    assert set(out) == {"model", "answers", "usage"} and out["model"] == "tiel"
    assert out["answers"]["q"]["choice"] == "a" and set(out["answers"]) == {"q", "o"}
    assert out["usage"] == {"input_tokens": 400 + 401, "output_tokens": 0}      # tokens_evaluated, image tokens included
    assert [qr.qid for qr in readouts] == ["q", "o"] and all(qr.readout.kind == "image" for qr in readouts)
    assert all(qr.ms >= 0 for qr in readouts)
    assert [m[3] for m in sc.mm] == [0, 0]


def test_answer_request_raises_invalid_request_for_a_bad_body_and_no_vision_before_any_call():
    sc = MMScorer({0: DIVERGED})
    with pytest.raises(InvalidRequest):
        _vision_engine(sc).answer_request({"state": "s", "images": "nope", "questions": TWO_Q})
    with pytest.raises(NoVision):
        Engine(sc, ChatMLTemplate(), SETS).answer_request({"state": "s", "images": [URL], "questions": TWO_Q})
    assert sc.mm == [] and sc.tok_calls == []


def _http_error(code, message):
    body = json.dumps({"error": {"code": code, "message": message, "type": "x"}}).encode()
    return urllib.error.HTTPError("http://up/completion", code, "err", {}, io.BytesIO(body))


class Failing(MMScorer):
    def __init__(self, exc):
        super().__init__({0: DIVERGED})
        self.exc = exc

    def next_logprobs_mm(self, *a, **kw):
        raise self.exc

    def next_logprobs(self, *a, **kw):
        raise self.exc


@pytest.mark.parametrize(("exc", "kind", "detail"), [
    (_http_error(500, "Failed to load image or audio file"), ImageRejected, "the upstream could not decode an image"),
    (_http_error(400, "Failed to load image or audio file"), ImageRejected, "the upstream could not decode an image"),
    (_http_error(400, "Invalid uri-encoded base64 value"), ImageRejected, "Invalid uri-encoded base64 value"),
    (_http_error(400, "uri must be base64 encoded"), ImageRejected, "uri must be base64 encoded"),
    (_http_error(400, "request (70000 tokens) exceeds the available context size (65536 tokens), try increasing it"),
     MediaTooLarge, "the request with its media exceeds the context size"),      # v0.3: 413, not 500
    (_http_error(404, "File Not Found"), UpstreamRejected, "404"),
    (_http_error(500, "Failed to tokenize prompt"), UpstreamUnavailable, "500"),
    (_http_error(503, "Loading model"), UpstreamUnavailable, "Loading model"),
    (urllib.error.URLError("refused"), UpstreamUnavailable, "refused"),
    (ConnectionResetError("reset"), UpstreamUnavailable, "reset"),
])
def test_upstream_errors_on_an_image_request_map_by_message_then_status(exc, kind, detail):
    with pytest.raises(kind, match=detail):
        _vision_engine(Failing(exc)).answer_request({"state": "s", "images": [URL], "questions": TWO_Q})


def test_the_media_marker_inside_the_request_text_is_a_422_before_any_upstream_call():
    """/props is passed through, so a client can learn the marker; in the state it would add an image slot."""
    sc = MMScorer({0: DIVERGED})
    for body in ({"state": f"x {MARKER} y", "images": [URL], "questions": TWO_Q},
                 {"state": "", "context": MARKER, "images": [URL], "questions": TWO_Q}):
        with pytest.raises(ImageRejected, match="media marker"):
            _vision_engine(sc).answer_request(body)
    assert sc.mm == [] and sc.tok_calls == []
    _vision_engine(sc).answer_request({"state": MARKER, "questions": TWO_Q})       # text-only: harmless, answered
    assert len(sc.plain) == 2


def test_a_text_request_keeps_the_v01_mapping_every_upstream_http_error_is_unavailable():
    with pytest.raises(UpstreamUnavailable):
        _vision_engine(Failing(_http_error(400, "Failed to load image or audio file"))).answer_request(
            {"state": "s", "questions": TWO_Q})


def test_an_http_error_with_a_non_json_body_still_classifies():
    raw = urllib.error.HTTPError("u", 500, "err", {}, io.BytesIO(b"Internal Server Error"))
    with pytest.raises(UpstreamUnavailable, match="Internal Server Error"):
        _vision_engine(Failing(raw)).answer_request({"state": "s", "images": [URL], "questions": TWO_Q})


def test_a_template_failure_is_a_readout_error_and_a_bug_propagates():
    class Broken(ChatMLTemplate):
        def render(self, msgs):
            raise TemplateError("apply-template returned no prompt")

    with pytest.raises(ReadoutError, match="no prompt"):
        _vision_engine(MMScorer({0: DIVERGED}), Broken()).answer_request({"state": "s", "questions": TWO_Q})
    with pytest.raises(KeyError):
        _vision_engine(Failing(KeyError("bug"))).answer_request({"state": "s", "images": [URL], "questions": TWO_Q})


# ---- the two request-level refusals are decided before any upstream call or slot -----------------------

ONE_Q = {"q": {"type": "choice", "criteria": {"only": "the single option"}}}


def test_a_single_label_image_request_on_a_non_vision_engine_is_501_not_the_single_option_200():
    sc = MMScorer({0: DIVERGED})
    with pytest.raises(NoVision):
        Engine(sc, ChatMLTemplate(), SETS).answer_request({"state": "s", "images": [URL], "questions": ONE_Q})
    out = _vision_engine(sc).answer_request({"state": "s", "images": [URL], "questions": ONE_Q})   # vision: shortcut ok
    assert out["answers"]["q"]["choice"] == "only"
    assert sc.mm == [] and sc.tok_calls == []


def test_check_request_decides_501_and_the_marker_422_without_touching_the_scorer():
    sc = MMScorer({0: DIVERGED})
    bad_marker = parse_request({"state": f"x{MARKER}", "images": [URL], "questions": ONE_Q})
    no_vision = Engine(sc, ChatMLTemplate(), SETS, media_marker=MARKER)
    with pytest.raises(NoVision):                       # 501 outranks the marker 422 on a non-vision upstream
        no_vision.check_request(bad_marker)
    with pytest.raises(ImageRejected, match="media marker"):
        _vision_engine(sc).check_request(bad_marker)
    for req in (parse_request({"state": f"x{MARKER}", "questions": ONE_Q}),
                parse_request({"state": "s", "images": [URL], "questions": ONE_Q})):
        _vision_engine(sc).check_request(req)
    assert sc.mm == [] and sc.tok_calls == [] and sc.plain == []


# ---- v0.2: local_engine, the demo's plain-server entry point -------------------------------------------

LOCAL_BODY = {"state": "", "images": [URL], "context": "orders",
              "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}}


def test_local_engine_runs_vision_detection_and_the_delimiter_and_nothing_else(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HOME", str(tmp_path))                        # the vocab cache goes under ~/.cache/callo
    g = tmp_path / "m.gguf"
    _write_tiny_gguf(g)
    up = FakeUpstream(vision=True).start()
    try:
        with caplog.at_level(logging.INFO, logger="callo"):
            e = local_engine(up.url, gguf=g)
        assert f"local engine on {up.url}: vision on" in caplog.text and "(vision: on)" not in caplog.text
        assert e.vision is True and e.media_marker == MEDIA_MARKER and e.user_delimiter == DELIM
        assert e.model_name == "fake-9b" and e.temperature == 1.0
        assert {r.path for r in up.requests} <= {"/props", "/apply-template", "/tokenize"}   # no probe/self-test
        out = e.answer_request(LOCAL_BODY, id_slot=None)
        comp = up.last("/completion").json
        assert set(out) == {"model", "answers", "usage"} and out["answers"]["q"]["choice"] == "b"
        assert "id_slot" not in comp and comp["prompt"]["multimodal_data"] == [PNG]
        assert comp["message_delimiters"] == [{"role": "user", "delimiter": DELIM}]
    finally:
        up.stop()


def test_local_engine_takes_a_pinned_delimiter_and_a_temperature(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    g = tmp_path / "m.gguf"
    _write_tiny_gguf(g)
    up = FakeUpstream().start()
    try:
        e = local_engine(up.url, gguf=g, temperature=1.9, user_delimiter="<|im_start|>user")
        assert e.vision is False and e.user_delimiter == "<|im_start|>user" and e.temperature == 1.9
        with pytest.raises(NoVision):
            e.answer_request(LOCAL_BODY)
    finally:
        up.stop()


def test_local_engine_refuses_a_think_budget_by_name_and_serves_budget_0(tmp_path, monkeypatch):
    """local_engine detects no thinking frame, so a positive think_budget is refused as such (422),
    never as "this model has no thinking mode"; budget 0 (absent or explicit) is v0.2's readout."""
    monkeypatch.setenv("HOME", str(tmp_path))
    g = tmp_path / "m.gguf"
    _write_tiny_gguf(g)
    up = FakeUpstream(think="preopened").start()
    try:
        e = local_engine(up.url, gguf=g)
        body = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}}
        with pytest.raises(InvalidRequest, match="think_budget 64: local_engine serves budget 0 only"):
            e.answer_request({**body, "think_budget": 64})
        assert not [r for r in up.requests if r.path == "/completion"]          # refused before any upstream call
        assert set(e.answer_request({**body, "think_budget": 0})["answers"]) == {"q"}
        e.think_reason = None                                                   # any engine without a frame
        with pytest.raises(InvalidRequest, match="no thinking mode callo can read; this engine serves budget 0"):
            e.answer_request({**body, "think_budget": 64})
    finally:
        up.stop()


# ---- v0.3: the position-bias profile -------------------------------------------------------------------

PB3 = load_profile(profile_dict({3: [0.6, 0.3, 0.1], 2: [0.9, 0.1]}))


def test_the_offset_is_subtracted_before_the_temperature_floored_options_included():
    marg = {"A": lp(0.5), "B": lp(0.3), "C": float("-inf")}
    top = {0: lp(0.3), 1: lp(0.2), 3: lp(0.3), 8: lp(0.05)}
    off = (0.2, -0.1, -0.1)
    probs, lms, miss = bounded_readout(marg, top, ("A", "B", "C"), 2.0, off)
    assert miss == 1 and lms == (lp(0.5), lp(0.3), lp(0.05))                 # log-marginals stay raw
    z = [(v - o) / 2.0 for v, o in zip(lms, off, strict=True)]
    e = [math.exp(v - max(z)) for v in z]
    assert probs == pytest.approx([x / sum(e) for x in e], abs=1e-15)
    assert bounded_readout(marg, top, ("A", "B", "C"), 2.0) == bounded_readout(marg, top, ("A", "B", "C"), 2.0, None)
    with pytest.raises(ValueError):
        bounded_readout(marg, top, ("A", "B", "C"), 1.0, (0.0, 0.0))


def test_without_a_profile_the_readout_is_bit_identical_and_bias_is_none():
    dist = {0: {0: lp(0.6), 1: lp(0.1), 3: lp(0.2), 5: lp(0.05), 8: lp(0.05)}}
    a = Engine(FakeScorer(dist), FakeTemplate(), SETS).answer("s", CHOICE)
    b = Engine(FakeScorer(dist), FakeTemplate(), SETS, position_bias=None).answer("s", CHOICE)
    assert a == b and a.bias is None and abs(a.probs[0] - 0.7 / 0.95) < 1e-12


def test_an_applied_profile_debiases_and_keeps_log_marginals_raw():
    dist = {0: {0: lp(0.6), 1: lp(0.1), 3: lp(0.2), 5: lp(0.05), 8: lp(0.05)}}
    raw = Engine(FakeScorer(dist), FakeTemplate(), SETS).answer("s", CHOICE)
    r = Engine(FakeScorer(dist), FakeTemplate(), SETS, position_bias=PB3).answer("s", CHOICE)
    assert r.bias == "applied" and r.log_marginals == raw.log_marginals
    want, _, _ = bounded_readout(dict(zip("ABC", raw.log_marginals, strict=True)), {0: 0.0}, ("A", "B", "C"), 1.0,
                                 PB3.offsets[3])
    assert r.probs == pytest.approx(want, abs=1e-15) and r.probs[0] < raw.probs[0]


def test_the_upstream_call_sequence_is_identical_with_and_without_a_profile():
    dist = {0: {6: lp(0.95), 8: lp(0.05)}, 1: {5: lp(0.9), 0: lp(0.05), 3: lp(0.05)}}
    a, b = FakeScorer(dist), FakeScorer(dist)
    Engine(a, FakeTemplate(), SETS).answer("s", CHOICE, id_slot=0)
    Engine(b, FakeTemplate(), SETS, position_bias=PB3).answer("s", CHOICE, id_slot=0)
    assert a.calls == b.calls and len(a.calls) == 2                          # the walk too (B3)


@pytest.mark.parametrize(("body", "kind", "state"), [
    ({"q": {"type": "choice", "criteria": {"a": "x", "b": "y", "c": "z"}}}, "plain", "applied"),
    ({"q": {"type": "choice", "criteria": {"a": "x", "b": "y", "c": "z", "d": "w"}}}, "plain", "no_entry"),
    ({"q": {"type": "choice", "criteria": {"c": "x", "b": "y", "a": "z"}}}, "plain", "noncanonical"),
    ({"q": {"type": "score", "criteria": ["lo", "mid", "hi"]}}, "plain", "score"),
    ({"q": {"type": "choice", "criteria": {"only": None}}}, "plain", "single_option"),
    ({"q": {"type": "noul"}}, "plain", "applied"),
    ({"q": {"type": "choice", "criteria": {"a": "x", "b": "y", "c": "z"}}}, "context", "kind_not_calibrated"),
    ({"q": {"type": "choice", "criteria": {"a": "x", "b": "y", "c": "z"}}}, "image", "kind_not_calibrated"),
])
def test_bias_for_every_category(body, kind, state):
    q = parse_request({"state": "s", "questions": body}).questions[0]
    eng = Engine(FakeScorer({}), FakeTemplate(), SETS, position_bias=PB3)
    off, got = eng.bias_for(q, kind)
    assert got == state and (off is not None) == (state == "applied")
    assert Engine(FakeScorer({}), FakeTemplate(), SETS).bias_for(q, kind) == (None, None)


def test_a_context_readout_is_corrected_only_when_the_profile_measured_context():
    sc = MMScorer({0: DIVERGED})
    plain_only = _vision_engine(sc, position_bias=PB3).answer("s", CHOICE, context="orders")
    both = load_profile(profile_dict({3: [0.6, 0.3, 0.1]}, kinds=("plain", "context")))
    r = _vision_engine(MMScorer({0: DIVERGED}), position_bias=both).answer("s", CHOICE, context="orders")
    raw = _vision_engine(MMScorer({0: DIVERGED})).answer("s", CHOICE, context="orders")
    assert plain_only.bias == "kind_not_calibrated" and plain_only.probs == raw.probs      # ineligible: RAW probs
    assert r.bias == "applied" and r.probs != raw.probs and r.log_marginals == raw.log_marginals


def test_a_noncanonical_plain_question_keeps_raw_probs_under_a_profile_with_its_n():
    q = parse_request({"state": "s", "questions": {"q": {"type": "choice",
                       "criteria": {"c": "x", "b": "y", "a": "z"}}}}).questions[0]
    dist = {0: {0: lp(0.6), 1: lp(0.1), 3: lp(0.2), 5: lp(0.05), 8: lp(0.05)}}
    raw = Engine(FakeScorer(dist), FakeTemplate(), SETS).answer("s", q)
    r = Engine(FakeScorer(dist), FakeTemplate(), SETS, position_bias=PB3).answer("s", q)
    assert 3 in PB3.offsets and r.bias == "noncanonical" and r.probs == raw.probs


def test_an_image_readout_is_corrected_only_when_the_profile_measured_image():
    def read(pb):
        return _vision_engine(MMScorer({0: DIVERGED}), position_bias=pb).answer("s", CHOICE, images=[PNG],
                                                                                image_urls=[URL])
    raw, plain_only = read(None), read(PB3)
    image = load_profile(profile_dict({3: [0.6, 0.3, 0.1]}, kinds=("image",)))
    r = read(image)
    assert plain_only.bias == "kind_not_calibrated" and plain_only.probs == raw.probs
    want, _, _ = bounded_readout(dict(zip("ABC", raw.log_marginals, strict=True)), {0: 0.0}, ("A", "B", "C"), 1.0,
                                 image.offsets[3])
    assert r.kind == "image" and r.bias == "applied" and r.log_marginals == raw.log_marginals
    assert r.probs == pytest.approx(want, abs=1e-15) and r.probs[0] < raw.probs[0]


def test_the_single_option_shortcut_reports_its_category():
    q = parse_request({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"only": None}}}}).questions[0]
    r = Engine(FakeScorer({}), FakeTemplate(), SETS, position_bias=PB3).answer("s", q)
    assert r.probs == (1.0,) and r.bias == "single_option"


def test_the_self_test_goes_through_the_profile():
    """The self-test question (options "3", "4") is canonical, n = 2. A profile with a strong position-A prior turns a
    raw A ("3") into a debiased B ("4"): the self-test reads through Engine.answer like every request."""
    dist = {0: {0: lp(0.6), 3: lp(0.35), 8: lp(0.05)}}                         # A 0.6, B 0.35
    assert self_test(Engine(FakeScorer(dist), FakeTemplate(), SETS)).answer == "3"
    assert self_test(Engine(FakeScorer(dist), FakeTemplate(), SETS, position_bias=PB3)).answer == "4"


def test_local_engine_checks_the_profile_in_both_phases(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HOME", str(tmp_path))
    g = tmp_path / "m.gguf"
    _write_tiny_gguf(g)
    up = FakeUpstream().start()
    try:
        with pytest.raises(InvalidInput, match="callo_position_bias"):
            local_engine(up.url, gguf=g, position_bias={**profile_dict(), "callo_position_bias": 9})
        assert up.requests == []                                              # phase 1 before any upstream call
        with pytest.raises(CalloError, match="this profile was measured on m.gguf"):
            local_engine(up.url, gguf=g, position_bias=profile_dict(key=KEY))
        key = load_model_key(g, None)
        with caplog.at_level(logging.WARNING, logger="callo"):
            e = local_engine(up.url, gguf=g, position_bias=profile_dict(key=key))
        assert e.position_bias is not None and e.position_bias.model_key == key
        assert "build_info differs" not in caplog.text                        # the profile's provenance matches
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="callo"):
            e2 = local_engine(up.url, gguf=g, position_bias=profile_dict(key=key, prov={"build_info": "b2-x"}))
        assert e2.position_bias is not None and "WARNING: --position-bias: build_info differs" in caplog.text
        assert "prompt_sha256 differs" in caplog.text           # prov= replaced the whole block: every field differs
        out = e.answer_request({"state": "s", "questions": {"q": {"type": "choice",
                                                                  "criteria": {"a": "x", "b": "y", "c": "z"}}}})
        assert set(out["answers"]["q"]["probabilities"]) == {"a", "b", "c"}
    finally:
        up.stop()


def test_local_engine_refuses_a_profile_on_a_native_decision_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    g = tmp_path / "kev.gguf"
    _write_tiny_gguf(g, decision="kev")
    up = FakeUpstream().start()
    try:
        with pytest.raises(CalloError, match="native decision model"):
            local_engine(up.url, gguf=g, position_bias=profile_dict(key=load_model_key(g, None)))
    finally:
        up.stop()


# ---- v0.3: audio, video and mixed readouts -------------------------------------------------------------------

WAV = "UklGRiQAAABXQVZFZm10IA=="                                   # 16 bytes: RIFF....WAVEfmt
AVI = "UklGRiQAAABBVkkgTElTVA=="                                   # 16 bytes: RIFF....AVI LIST
AUDIO = MediaItem("audio", WAV, f"data:audio/wav;base64,{WAV}", "wav")
VIDEO = MediaItem("video", AVI, None)
IMAGE = MediaItem("image", PNG, URL)


def _media_engine(sc, template=None, **kw):
    flags = {"vision": True, "audio": True, "video": True, **kw}
    return Engine(sc, template or ChatMLTemplate(), SETS, media_marker=MARKER, user_delimiter=DELIM, **flags)


@pytest.mark.parametrize(("media", "kind"), [((AUDIO,), "audio"), ((VIDEO, VIDEO), "video"),
                                             ((IMAGE, AUDIO, VIDEO), "mixed")])
def test_a_media_readout_sends_placeholders_to_the_template_and_payloads_in_order_to_completion(media, kind):
    sc, t = MMScorer({0: DIVERGED}), ChatMLTemplate()
    r = _media_engine(sc, t).answer("s", CHOICE, id_slot=2, context="orders", media=media)
    prompt, payloads, _, slot, delim = sc.mm[0]
    assert payloads == tuple(m.payload for m in media) and slot == 2 and delim == DELIM     # multimodal_data order
    assert prompt.count(MARKER) == len(media) and prompt.rfind(MARKER) < prompt.index("Options:")
    assert prompt == t.render(messages("s", CHOICE, context="orders", media=media))         # real parts: same bytes
    parts = t.seen[0][1]["content"][:-1]
    assert [p["type"] for p in parts] == [{"image": "image_url", "audio": "input_audio", "video": "input_video"}[m.kind]
                                          for m in media]
    assert WAV not in str(t.seen[0]) and AVI not in str(t.seen[0])              # no payload crosses /apply-template
    assert all(p["image_url"]["url"] == PLACEHOLDER_IMAGE_URL for p in parts if p["type"] == "image_url")
    # the media table in full: audio and video carry the fixed "AAAA" stand-in, audio keeps its sniffed format
    assert all(p["input_audio"] == {"data": PLACEHOLDER_PAYLOAD, "format": "wav"}
               for p in parts if p["type"] == "input_audio")
    assert all(p["input_video"] == {"data": PLACEHOLDER_PAYLOAD} for p in parts if p["type"] == "input_video")
    assert r.kind == kind and r.passes == 1 and r.stop == "diverged" and r.prompt_tokens == 400


def test_the_media_walk_resends_the_same_payloads_and_counts_its_passes():
    sc = MMScorer({0: {6: lp(0.95), 8: lp(0.05)}, 1: {5: lp(0.9), 0: lp(0.05), 3: lp(0.05)}},
                  detok={6: "**"}, vocab={"**": 6})
    r = _media_engine(sc).answer("s", CHOICE, media=(VIDEO,))
    assert r.stop == "diverged" and r.depth == 1 and r.passes == 2 and r.kind == "video"
    assert sc.mm[1][0] == sc.mm[0][0] + "**" and sc.mm[1][1] == sc.mm[0][1] == (AVI,)


def test_passes_count_scoring_calls_on_every_path():
    walk = {0: {6: lp(0.95), 8: lp(0.05)}, 1: {5: lp(0.9), 0: lp(0.1)}}
    plain = Engine(FakeScorer({0: {6: lp(0.95), 8: lp(0.05)}, 1: DIVERGED}), FakeTemplate(), SETS).answer("s", CHOICE)
    assert plain.depth == 1 and plain.passes == 2 and plain.kind == "plain"
    context = _media_engine(MMScorer(walk)).answer("s", CHOICE, context="c")
    assert context.kind == "context" and context.passes == context.depth + 1
    retok = _media_engine(MMScorer(walk, detok={6: "\n"}, vocab={"**": 6})).answer("s", CHOICE, media=(AUDIO,))
    assert retok.stop == "retok" and retok.passes == 1                  # the stopped step never reached the upstream
    q = parse_request({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"only": None}}}}).questions[0]
    sc = MMScorer({0: DIVERGED})
    one = _media_engine(sc).answer("s", q, media=(AUDIO,))
    assert one.passes == 0 and one.kind == "audio" and sc.mm == []      # the single-option shortcut


@pytest.mark.parametrize(("flags", "media", "exc"), [
    ({"audio": False}, (AUDIO,), NoAudio), ({"video": False}, (VIDEO,), NoVideo),
    ({"vision": False}, (IMAGE,), NoVision), ({"vision": False, "video": False}, (AUDIO, VIDEO, IMAGE), NoVision),
    ({"audio": False, "video": False}, (VIDEO, AUDIO), NoAudio),        # the first missing kind in contract order
])
def test_a_kind_the_upstream_lacks_is_a_501_before_any_upstream_call(flags, media, exc):
    sc = MMScorer({0: DIVERGED})
    with pytest.raises(exc):
        _media_engine(sc, **flags).answer("s", CHOICE, media=media)
    assert sc.mm == [] and sc.tok_calls == []


def test_an_audio_only_upstream_answers_audio_and_refuses_images():
    """The v0.2 `if not self.vision` guard is per kind now."""
    sc = MMScorer({0: DIVERGED, 1: DIVERGED})
    eng = _media_engine(sc, vision=False, video=False)
    assert eng.answer("s", CHOICE, media=(AUDIO,)).kind == "audio"
    with pytest.raises(NoVision):
        eng.answer("s", CHOICE, images=[PNG], image_urls=[URL])
    assert [eng.supports(k) for k in ("image", "audio", "video")] == [False, True, False]
    assert not Engine(sc, ChatMLTemplate(), SETS, audio=True).supports("audio")     # no media marker: no media


def test_the_deprecated_image_keywords_read_exactly_as_media():
    a, b = MMScorer({0: DIVERGED}), MMScorer({0: DIVERGED})
    old = _vision_engine(a).answer("s", CHOICE, id_slot=0, images=[PNG], image_urls=[URL])
    new = _vision_engine(b).answer("s", CHOICE, id_slot=0, media=(IMAGE,))
    assert old == new and a.mm == b.mm and old.kind == "image"
    with pytest.raises(ValueError, match="not both"):
        _vision_engine(a).answer("s", CHOICE, media=(IMAGE,), images=[PNG], image_urls=[URL])


def test_the_deprecated_image_keywords_pair_lazily_as_v02_did():
    """Malformed direct-call aliases behave as in v0.2 (no Request path reaches these)."""
    one = parse_request({"state": "s",
                         "questions": {"q": {"type": "choice", "criteria": {"only": None}}}}).questions[0]
    sc = MMScorer({0: DIVERGED})
    r = _vision_engine(sc).answer("s", one, images=[PNG, PNG], image_urls=[URL])   # (a) the shortcut wins
    assert r.stop == "single_option" and r.kind == "image" and sc.mm == []
    with pytest.raises(NoVision):                                                  # (b) the 501 comes first
        Engine(sc, ChatMLTemplate(), SETS).answer("s", CHOICE, images=[PNG, PNG], image_urls=[URL])
    # a vision engine still refuses
    with pytest.raises(ReadoutError, match="2 image payload"):
        _vision_engine(sc).answer("s", CHOICE, images=[PNG, PNG], image_urls=[URL])
    assert sc.mm == []
    # (c) image_urls= alone: text path
    text = FakeScorer({0: DIVERGED})
    plain = Engine(text, FakeTemplate(), SETS).answer("s", CHOICE, image_urls=[URL])
    assert plain.kind == "plain"
    assert plain == Engine(FakeScorer({0: DIVERGED}), FakeTemplate(), SETS).answer("s", CHOICE)


def test_render_prompt_takes_media_or_v02_image_urls():
    t = ChatMLTemplate()
    e = _media_engine(MMScorer({0: DIVERGED}), t)
    assert e.render_prompt("s", CHOICE, media=(AUDIO, VIDEO)).count(MARKER) == 2
    assert e.render_prompt("s", CHOICE, image_urls=[URL]) == e.render_prompt("s", CHOICE, media=(IMAGE,))
    with pytest.raises(ValueError, match="not both"):
        e.render_prompt("s", CHOICE, media=(IMAGE,), image_urls=[URL])

    class DropsAudio(ChatMLTemplate):
        def render(self, msgs):
            return super().render([{**m, "content": [p for p in m["content"] if p["type"] != "input_audio"]}
                                   if isinstance(m["content"], list) else m for m in msgs])

    with pytest.raises(ReadoutError, match="did not render the audio parts: 0 media marker"):
        _media_engine(MMScorer({0: DIVERGED}), DropsAudio()).answer("s", CHOICE, media=(AUDIO,))
    with pytest.raises(ReadoutError, match="did not render the media parts: 1 media marker"):
        _media_engine(MMScorer({0: DIVERGED}), DropsAudio()).answer("s", CHOICE, media=(AUDIO, VIDEO))


def test_a_readout_built_with_keywords_defaults_passes_to_zero():
    r = Readout(probs=(1.0,), log_marginals=(0.0,), option_mass=1.0, stop="diverged", depth=0, topk_miss=0,
                prompt_tokens=1)
    assert r.passes == 0 and r.bias is None and r.kind == "plain"


# ---- v0.3: the media error table, check_request per kind, local_engine ---------------------------------------

AUDIO_BODY = {"state": "s", "audio": [f"data:audio/wav;base64,{WAV}"], "questions": TWO_Q}
VIDEO_BODY = {"state": "s", "videos": [f"data:video/x-msvideo;base64,{AVI}"], "questions": TWO_Q}
HINT = " - hint: if this is unexpected, you may need to provide the mmproj"


@pytest.mark.parametrize(("exc", "kind", "detail"), [
    (_http_error(500, "Failed to load image or audio file"), MediaLoadFailed,
     "the upstream could not decode an audio clip: Failed to load image or audio file"),
    (_http_error(400, "Failed to load image or audio file"), MediaLoadFailed, "could not decode an audio clip"),
    (_http_error(500, "audio input is not supported" + HINT), NoAudio, "no audio support"),
    (_http_error(500, "video input is not supported" + HINT), NoVideo, "no video support"),
    (_http_error(500, "image input is not supported" + HINT), NoVision, "no vision support"),
    (_http_error(400, "Invalid uri-encoded base64 value"), MediaRejected, "Invalid uri-encoded base64 value"),
    (_http_error(400, "request (70000 tokens) exceeds the available context size (65536 tokens), try increasing it"),
     MediaTooLarge, "exceeds the context size; lower --video-fps / --image-max-tokens or use a shorter clip"),
    (_http_error(400, "input (70000 tokens) is larger than the max context size (65536 tokens). skipping"),
     MediaTooLarge, "the request with its media exceeds the context size"),
    (_http_error(400, "Failed to tokenize prompt"), UpstreamRejected, "upstream HTTP 400"),     # llama.cpp's real 400
    (_http_error(404, "File Not Found"), UpstreamRejected, "upstream HTTP 404"),
    (_http_error(500, "Invalid base64 value"), UpstreamUnavailable, "upstream HTTP 500"),      # /apply-template's 500
    (_http_error(503, "Loading model"), UpstreamUnavailable, "Loading model"),
    (urllib.error.URLError("refused"), UpstreamUnavailable, "refused"),
])
def test_the_media_error_table_row_by_row_on_an_audio_request(exc, kind, detail):
    """Every request that carries media maps upstream errors this way, not only image requests."""
    with pytest.raises(kind, match=detail):
        _media_engine(Failing(exc)).answer_request(AUDIO_BODY)


def test_the_422_names_the_requests_kind_and_the_v02_alias_still_says_image():
    for body, noun in ((VIDEO_BODY, "a video"), ({**AUDIO_BODY, "videos": VIDEO_BODY["videos"]}, "a media item"),
                       ({"state": "s", "images": [URL], "questions": TWO_Q}, "an image")):
        with pytest.raises(MediaLoadFailed, match=f"could not decode {noun}: Failed to load"):
            _media_engine(Failing(_http_error(500, "Failed to load image or audio file"))).answer_request(body)
    assert str(classify_image_error(_http_error(500, "Failed to load image or audio file"))).startswith(
        "the upstream could not decode an image: ")
    assert isinstance(classify_media_error(_http_error(500, "Failed to load image or audio file")), MediaRejected)
    assert ImageRejected is MediaRejected and issubclass(MediaLoadFailed, MediaRejected)


def test_an_apply_template_refusal_of_a_part_type_is_a_501():
    class Refuses(ChatMLTemplate):
        def render(self, msgs):
            raise _http_error(500, "video input is not supported" + HINT)

    sc = MMScorer({0: DIVERGED})
    with pytest.raises(NoVideo):
        _media_engine(sc, Refuses()).answer_request(VIDEO_BODY)
    assert sc.mm == []


def test_check_request_refuses_each_missing_kind_and_the_marker_before_any_upstream_call():
    sc = MMScorer({0: DIVERGED})
    with pytest.raises(NoAudio):
        _media_engine(sc, audio=False).check_request(parse_request(AUDIO_BODY))
    with pytest.raises(NoVideo):
        _media_engine(sc, video=False).check_request(parse_request(VIDEO_BODY))
    with pytest.raises(MediaRejected, match="media marker"):
        _media_engine(sc).check_request(parse_request({**VIDEO_BODY, "context": f"see {MARKER}"}))
    with pytest.raises(NoAudio):                       # a single-label audio request is a 501, not the shortcut's 200
        _media_engine(sc, audio=False).answer_request({**AUDIO_BODY, "questions": ONE_Q})
    _media_engine(sc).check_request(parse_request(AUDIO_BODY))
    assert sc.mm == [] and sc.tok_calls == []


def test_local_engine_detects_audio_and_video(tmp_path, monkeypatch, caplog):
    """local_engine uses detect_media, so an audio-only upstream gives an engine that answers audio."""
    monkeypatch.setenv("HOME", str(tmp_path))
    g = tmp_path / "m.gguf"
    _write_tiny_gguf(g)
    up = FakeUpstream().start()
    up.props["modalities"] = {"vision": False, "audio": True, "video": False}
    try:
        with caplog.at_level(logging.INFO, logger="callo"):
            e = local_engine(up.url, gguf=g)
        assert (e.vision, e.audio, e.video, e.media_marker) == (False, True, False, MEDIA_MARKER)
        assert f"local engine on {up.url}: vision off, audio on, video off" in caplog.text
    finally:
        up.stop()


def test_a_context_overflow_is_recognised_by_its_error_type_too():
    """llama.cpp tags both context messages with type exceed_context_size_error (server-common.cpp:68-70); a future
    rewording of the message must not turn the 413 back into a 500."""
    body = json.dumps({"error": {"code": 400, "message": "prompt too long", "type": "exceed_context_size_error",
                                 "n_prompt_tokens": 70000, "n_ctx": 65536}}).encode()
    exc = urllib.error.HTTPError("http://up/completion", 400, "err", {}, io.BytesIO(body))
    with pytest.raises(MediaTooLarge, match=r"exceeds the context size.*\(upstream: prompt too long\)"):
        _media_engine(Failing(exc)).answer_request(VIDEO_BODY)
