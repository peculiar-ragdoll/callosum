"""callo v0.3 wide-choice, the surface caches: `SurfaceSets.wide_coverage`, `wide`, `partial`, `continuations`, `warm_wide`, and the coverage 422.

The real-vocabulary tests read GGUF headers under CALLO_BENCH_MODELS (and llama.cpp's own vocabularies beside
CALLO_BENCH_LLAMA_SERVER's checkout) and skip when a file is missing or the variable is unset, as on a CI runner. `CALLO_TEST_EXTRA_GGUF=/path/to.gguf` prints that file's coverage (how LFM gets
checked)."""
import functools
import json
import os
import string
import threading
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from fake_upstream import FakeUpstream
from helpers import bench_path

from callo import engine as engine_mod
from callo.engine import Engine, SurfaceSets
from callo.errors import InvalidInput, TooManyOptions
from callo.marginalize import check_option_sets, decode_token, normalize_surface, option_surface_sets
from callo.prompt import TemplateClient
from callo.proxy import ProxyConfig, Stats, SystemOneService, make_server
from callo.scorer import LlamaCppForwardScorer, make_http_post
from callo.slots import SlotRouter
from callo.systemone import TOO_MANY_OPTIONS, WIDE_LABELS, parse_request

LETTERS = list(string.ascii_uppercase)


def _vocab(n_labels: int, *, drop_spaced: int | None = None, drop_all: int | None = None) -> list[str]:
    """gpt2-style: the 26 letters bare and spaced, `A.`, `**`, `Ġthe`, then WIDE_LABELS[:n_labels] bare and spaced.
    `drop_spaced`: that label index has only its bare form; `drop_all`: no surface token at all."""
    toks = LETTERS + ["Ġ" + c for c in LETTERS] + ["A.", "**", "Ġthe"]
    for i, lab in enumerate(WIDE_LABELS[:n_labels]):
        if i == drop_all:
            continue
        toks.append(lab)
        if i != drop_spaced:
            toks.append("Ġ" + lab)
    return toks


def _count(monkeypatch, name):
    calls = []
    real = getattr(engine_mod, name)

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(engine_mod, name, spy)
    return calls


# ---- synthetic vocabularies --------------------------------------------------------------------------------------

def test_coverage_stops_at_the_first_missing_label():
    assert SurfaceSets(_vocab(40), "gpt2").wide_coverage() == 40
    assert SurfaceSets(_vocab(40, drop_spaced=30), "gpt2").wide_coverage() == 30
    assert SurfaceSets(_vocab(40, drop_all=3), "gpt2").wide_coverage() == 3
    assert SurfaceSets(_vocab(255), "gpt2").wide_coverage() == 255


def test_coverage_reads_sentencepiece_spaces():
    toks = LETTERS + ["▁" + c for c in LETTERS] + [t for lab in WIDE_LABELS[:30] for t in (lab, "▁" + lab)]
    assert SurfaceSets(toks, "gemma4").wide_coverage() == 30


def test_wide_equals_a_fresh_build_over_its_labels():
    s = SurfaceSets(_vocab(40), "gpt2")
    want = option_surface_sets(s._tokens, "gpt2", WIDE_LABELS[:30])
    got = s.wide(30)
    assert got == want and list(got) == list(WIDE_LABELS[:30])
    assert s.get(WIDE_LABELS[:30]) == want                         # get routes a wide tuple here


def test_one_scan_serves_every_n(monkeypatch):
    calls = _count(monkeypatch, "option_surface_sets")
    s = SurfaceSets(_vocab(40), "gpt2")
    for n in (27, 30, 40, 33):
        assert list(s.wide(n)) == list(WIDE_LABELS[:n])
    assert list(s.get(WIDE_LABELS[:35])) == list(WIDE_LABELS[:35])
    assert len(calls) == 1
    s.get(("A", "B"))                                              # narrow: today's per-label-set memo
    s.get(("B", "A"))
    assert len(calls) == 2


def test_wide_coverage_is_one_cached_pass(monkeypatch):
    calls = _count(monkeypatch, "decode_token")
    s = SurfaceSets(_vocab(30), "gpt2")
    assert s.wide_coverage() == 30
    n = len(calls)
    assert n == len(s._tokens)
    assert s.wide_coverage() == 30 and len(calls) == n


def _unreadable_past(toks: list[str], letter: str) -> list[str]:
    """`letter`'s bare token respelled `<letter>` (ids unchanged): no token continues a label with it, so an uncovered
    label ending in it cannot be read in two passes."""
    return [(f"<{letter}>" if t == letter else t) for t in toks]


def test_wide_requires_a_wide_n_within_what_it_reads():
    """An explicit raise, not an assert (which `python -O` strips): the caller (`Engine.check_coverage`) refuses first.
    A vocabulary covering 30 labels reads all 255 (the rest in two passes); one that cannot spell `AJ` (label 7) at all
    reads 6."""
    s = SurfaceSets(_vocab(30), "gpt2")
    assert (s.wide_coverage(), s.wide_readable()) == (30, 255)
    assert list(s.wide(31)) == list(WIDE_LABELS[:31]) and s.wide(31)["AE"] and not s.wide(31)[WIDE_LABELS[30]]
    with pytest.raises(ValueError, match=r"wide\(26\) on a vocabulary reading 255"):
        s.wide(26)
    with pytest.raises(ValueError, match=r"wide\(256\) on a vocabulary reading 255"):
        s.wide(256)
    short = SurfaceSets(_unreadable_past(_vocab(30, drop_all=6), "J"), "gpt2")
    assert (short.wide_coverage(), short.wide_readable()) == (6, 6)
    with pytest.raises(ValueError, match=r"wide\(27\) on a vocabulary reading 6"):
        short.wide(27)


def test_label_28_missing_gives_coverage_27_and_warm_wide_succeeds(monkeypatch):
    toks = _vocab(40, drop_all=27)
    with pytest.raises(InvalidInput, match="NO surface form"):        # the hazard: building past coverage raises
        check_option_sets(option_surface_sets(toks, "gpt2", WIDE_LABELS[:40]))
    s = SurfaceSets(toks, "gpt2")
    assert s.warm_wide() == 27
    assert list(s.wide(27)) == list(WIDE_LABELS[:27])


def test_warm_wide_builds_the_wide_caches_once(monkeypatch):
    osets = _count(monkeypatch, "option_surface_sets")
    psets = _count(monkeypatch, "partial_sets")
    csets = _count(monkeypatch, "continuation_sets")
    s = SurfaceSets(_vocab(40), "gpt2")
    assert s.warm_wide() == 40
    assert (len(osets), len(psets), len(csets)) == (1, 1, 1)
    s.wide(30), s.partial(("A", "B")), s.continuations(), s.warm_wide()
    assert (len(osets), len(psets), len(csets)) == (1, 1, 1)


def test_warm_wide_on_a_vocabulary_reading_26_or_fewer_builds_no_table(monkeypatch):
    """Coverage 9 and no way to spell `AO` (label 10) in two passes: no wide question can be read, so no wide table is
    built; the letter sets that decided it are."""
    osets = _count(monkeypatch, "option_surface_sets")
    psets = _count(monkeypatch, "partial_sets")
    csets = _count(monkeypatch, "continuation_sets")
    s = SurfaceSets(_unreadable_past(_vocab(40, drop_all=9), "O"), "gpt2")
    assert s.warm_wide() == 9 and s.wide_readable() == 9
    assert (len(osets), len(psets), len(csets)) == (0, 1, 1)


def test_warm_wide_on_a_short_coverage_vocabulary_builds_every_readable_label_once(monkeypatch):
    """Coverage 9, every letter a token: all 255 labels are readable (246 of them in two passes, their sets empty), and
    the warm-up builds the table, the letter sets and the continuations once each."""
    osets = _count(monkeypatch, "option_surface_sets")
    psets = _count(monkeypatch, "partial_sets")
    csets = _count(monkeypatch, "continuation_sets")
    s = SurfaceSets(_vocab(40, drop_all=9), "gpt2")
    assert s.warm_wide() == 9 and s.wide_readable() == 255
    assert (len(osets), len(psets), len(csets)) == (1, 1, 1)
    w = s.wide(255)
    assert sum(not v for v in w.values()) == 255 - 39 and s.uncovered(WIDE_LABELS[:12]) == (WIDE_LABELS[9],)
    s.wide(30), s.partial(("A", "B")), s.continuations(), s.warm_wide()
    assert (len(osets), len(psets), len(csets)) == (1, 1, 1)


def test_partial_needs_no_disjointness_check_and_warm_wide_checks_partial_against_wide(monkeypatch):
    """A token normalises to ONE letter, so the partial sets are disjoint by construction and `partial`
    runs no check; the invariant that matters, partial tokens disjoint from every wide label surface (else a token
    would be counted as a label AND as a first letter), is checked once, in `warm_wide`."""
    checks = _count(monkeypatch, "check_option_sets")
    s = SurfaceSets(_vocab(30), "gpt2")
    s.partial(("A", "B"))
    assert checks == []
    assert s.warm_wide() == 30
    assert len(checks) == 2                                        # the wide build's own check, then partial vs wide
    ids = {t: i for i, t in enumerate(s._tokens)}
    real = engine_mod.partial_sets

    def leaky(tokens, model, letters):
        out = real(tokens, model, letters)
        return {**out, "A": out["A"] | {ids["ĠAB"]}}               # a wide label surface in a partial set
    monkeypatch.setattr(engine_mod, "partial_sets", leaky)
    with pytest.raises(InvalidInput, match="AB"):
        SurfaceSets(_vocab(30), "gpt2").warm_wide()


def test_partial_and_continuation_sets():
    s = SurfaceSets(_vocab(30), "gpt2")
    ids = {t: i for i, t in enumerate(s._tokens)}
    part = s.partial(("A", "B"))
    assert list(part) == ["A", "B"]
    assert part["A"] == frozenset({ids["A"], ids["ĠA"]}) and part["B"] == frozenset({ids["B"], ids["ĠB"]})
    cont = s.continuations()
    assert list(cont) == LETTERS and cont["B"] == frozenset({ids["B"]})
    wide_ids = frozenset().union(*s.wide(30).values())
    assert not wide_ids & frozenset().union(*s.partial(tuple(LETTERS)).values())


def test_engine_check_coverage_reads_past_the_coverage():
    """Coverage 6 (`AJ`, label 7, is missing) with every letter a token: 27 and 255 options are read, the labels past
    the coverage in two passes."""
    eng = Engine(None, None, SurfaceSets(_vocab(40, drop_all=6), "gpt2"))
    for n in (27, 255):
        q = parse_request({"state": "s", "questions": {"q": {"type": "choice",
                                                             "criteria": {f"k{i}": None for i in range(n)}}}})
        eng.check_request(q)
        eng.check_coverage(q.questions[0])


def test_engine_check_coverage_is_the_422():
    """Reads 6 (`AJ`, label 7, is missing and no token continues a label with J): 26 options still read, 27 are
    refused with the exact message."""
    toks = _unreadable_past(_vocab(40, drop_all=6), "J")
    eng = Engine(None, None, SurfaceSets(toks, "gpt2"))
    narrow = parse_request({"state": "s", "questions": {"q": {"type": "choice",
                                                              "criteria": {f"k{i}": None for i in range(26)}}}})
    eng.check_request(narrow)
    eng.check_coverage(narrow.questions[0])
    wide = parse_request({"state": "s", "questions": {"q": {"type": "choice",
                                                            "criteria": {f"k{i}": None for i in range(27)}}}})
    msg = ("q: this model's tokenizer can read 6 of Callo's two-letter labels (the next one needs a letter that is not "
           "a single token); the question has 27 options")
    with pytest.raises(TooManyOptions) as e:
        eng.check_coverage(wide.questions[0])
    assert str(e.value) == msg == TOO_MANY_OPTIONS.format(qid="q", n=6, m=27)
    with pytest.raises(TooManyOptions, match="can read 6 of Callo.s two-letter labels"):
        eng.check_request(wide)


# ---- real vocabularies (gated) -----------------------------------------------------------------------------------

def _llama_cpp_vocab(name: str) -> str:
    """One of llama.cpp's own vocabulary GGUFs (`models/`), in the checkout whose `build/bin/llama-server`
    CALLO_BENCH_LLAMA_SERVER names."""
    server = Path(bench_path("CALLO_BENCH_LLAMA_SERVER"))
    return str((server.parents[2] if len(server.parents) > 2 else server) / "models" / name)


QWEN35 = bench_path("CALLO_BENCH_MODELS", "local", "system1", "Qwen3.5-9B-Q8_0.gguf")
GEMMA4 = bench_path("CALLO_BENCH_MODELS", "unsloth", "Gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-UD-Q4_K_XL.gguf")
MINICPM5 = bench_path("CALLO_BENCH_MODELS", "local", "MiniCPM5-2B-UD-Q6_K_XL.gguf")
QWEN2 = _llama_cpp_vocab("ggml-vocab-qwen2.gguf")
SPARK = bench_path("CALLO_BENCH_MODELS", "xhtoken", "Spark-X2.5-4B-GGUF", "Spark-X2.5-4B-BF16.gguf")
LLAMA_SPM = _llama_cpp_vocab("ggml-vocab-llama-spm.gguf")
COVERAGE = [(QWEN35, 255), (GEMMA4, 255), (MINICPM5, 255), (QWEN2, 155), (SPARK, 9), (LLAMA_SPM, 3)]
# (path, labels that are not single tokens anywhere in the table, labels with no surface token at all): every one of
# these vocabularies reads all 255 labels, the uncovered ones in two passes
UNCOVERED = [(QWEN2, 5, 0), (SPARK, 47, 2), (LLAMA_SPM, 164, 28)]


def _need(path: str) -> None:
    pytest.importorskip("gguf")
    if not os.path.exists(path):
        pytest.skip(f"{path} is not on this box")


@functools.cache
def _real(path: str) -> SurfaceSets:
    return SurfaceSets.from_gguf(path)


@pytest.mark.parametrize(("path", "coverage"), COVERAGE, ids=lambda v: Path(v).stem if isinstance(v, str) else v)
def test_real_vocabulary_coverage(path, coverage):
    _need(path)
    s = _real(path)
    assert s.wide_coverage() == coverage
    assert s.warm_wide() == coverage                               # never raises, short vocabularies included
    assert s.wide_readable() == 255                                # the rest are read in two passes


@pytest.mark.parametrize(("path", "uncovered", "empty"), UNCOVERED, ids=lambda v: Path(v).stem if isinstance(v, str) else v)
def test_real_short_vocabularies_read_every_label(path, uncovered, empty):
    """Every letter is a token bare and after a space, and a label continuation on its own, in each of them: the 422
    stays only for a vocabulary that lacks one, and none here does."""
    _need(path)
    s = _real(path)
    assert len(s.uncovered(WIDE_LABELS)) == uncovered
    assert sum(not v for v in s.wide(255).values()) == empty
    assert all(s.partial((c,))[c] for c in LETTERS) and all(s.continuations()[c] for c in LETTERS)


def test_qwen35_wide_sets():
    _need(QWEN35)
    s = _real(QWEN35)
    ids = {t: i for i, t in enumerate(s._tokens)}
    w = s.wide(255)
    assert {ids["AB"], ids["ĠAB"]} <= w["AB"]
    assert ids["B"] in s.continuations()["B"]
    part = s.partial(("A",))["A"]
    assert {ids["A"], ids["ĠA"]} <= part
    # qwen35 has no `A.` token (the design's example; every token normalising to A ends in the letter here, `ª` via
    # NFKC included), so the partial set is exactly that class. The wrapper exclusion is shown on Gemma 4 below.
    norm = {i for i, t in enumerate(s._tokens) if normalize_surface(decode_token(t, s._model)) == "A"}
    ends = {i for i in norm if unicodedata.normalize("NFKC", decode_token(s._tokens[i], s._model))[-1] in "Aa"}
    assert "A." not in ids and part == frozenset(ends) == frozenset(norm)
    assert not frozenset().union(*w.values()) & frozenset().union(*s.partial(tuple(LETTERS)).values())


def test_gemma4_partial_sets_exclude_a_trailing_wrapper():
    _need(GEMMA4)
    s = _real(GEMMA4)
    ids = {t: i for i, t in enumerate(s._tokens)}
    part = s.partial(("A",))["A"]
    assert {ids["A"], ids["▁A"]} <= part
    assert ids["<a>"] not in part and ids["</a>"] not in part          # normalise to A, but end in `>`


def test_llama_spm_warm_wide_succeeds():
    _need(LLAMA_SPM)
    assert _real(LLAMA_SPM).warm_wide() == 3


def _short_sets(which: str) -> SurfaceSets:
    """Coverage 9 either way: the real Spark vocabulary (gated), or a synthetic gpt2 one whose label 10 (`AO`) has no
    surface token at all, so Linux CI runs the proxy-level readout too."""
    if which == "spark":
        _need(SPARK)
        return _real(SPARK)
    return SurfaceSets(_vocab(40, drop_all=9), "gpt2")


@pytest.mark.parametrize("which", ["synthetic", "spark"])
def test_spark_27_options_are_read_past_the_coverage(which):
    """A coverage-9 vocabulary (Spark's, or a synthetic one) warms without raising, and a 27-option request
    through the proxy is answered (200; it was a 422 before 0.3.1), counted as a wide readout of a question with
    uncovered labels; the same body with 26 options reaches the upstream as before."""
    sets = _short_sets(which)
    assert sets.warm_wide() == 9 and sets.wide_readable() == 255
    up = FakeUpstream().start()
    post = make_http_post(timeout=10)
    eng = Engine(LlamaCppForwardScorer(up.url, http_post=post), TemplateClient(up.url, http_post=post), sets)
    router, stats = SlotRouter(4, 1), Stats()
    svc = SystemOneService(eng, router, stats, model_name="spark")
    srv = make_server(ProxyConfig(upstream=up.url, router=router, stats=stats, systemone=svc.respond), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def send(n: int) -> urllib.request.Request:
        body = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {f"k{i}": None for i in range(n)}}}}
        return urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/v1/systemone",
                                      data=json.dumps(body).encode(), method="POST",
                                      headers={"Content-Type": "application/json"})

    def completions() -> list:
        return [r for r in up.requests if r.path.split("?")[0] == "/completion"]

    try:
        with urllib.request.urlopen(send(27), timeout=10) as r:
            assert r.status == 200 and set(json.loads(r.read())["answers"]) == {"q"}
        assert completions()
        s1 = stats.snapshot(router, None)["s1"]
        assert (s1["wide_readouts"], s1["wide_uncovered_readouts"]) == (1, 1)
        n = len(completions())
        with urllib.request.urlopen(send(26), timeout=10) as r:
            assert r.status == 200 and set(json.loads(r.read())["answers"]) == {"q"}
        assert len(completions()) > n
    finally:
        srv.shutdown()
        srv.server_close()
        up.stop()


def test_extra_gguf_coverage():
    path = os.environ.get("CALLO_TEST_EXTRA_GGUF")
    if not path:
        pytest.skip("CALLO_TEST_EXTRA_GGUF is not set")
    _need(path)
    s = SurfaceSets.from_gguf(path)
    cov = s.wide_coverage()
    print(f"\n{path}: wide_coverage {cov} (tokenizer {s._model}, {len(s._tokens)} tokens)")
    assert 0 <= cov <= len(WIDE_LABELS) and s.warm_wide() == cov
