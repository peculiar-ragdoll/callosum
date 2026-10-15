"""The answer-surface marginaliser: `P(answer = C)`, never `P(token "C")`.

What a multiple-choice probe wants is

    P(answer = C | context) = SUM over surface forms s expressing C of P(s | context)

**Sum, not max.** `"C"`, `" C"`, `"\\nC"`, `"(C"` are four spellings of one event, and taking the
largest of them throws away real answer mass -- unevenly, because how a letter's mass splits across
spellings is itself a thing quantization perturbs. Summing is the correct marginalisation over how
the model chose to spell its answer.

**Equivalence sets are built mechanically, never curated.** Enumerate the vocabulary, decode each
token, normalise (strip whitespace / markdown / brackets / punctuation, uppercase), keep the ones
reducing to the option letter. Nothing here knows that MiniCPM5 exists. A hand-curated list of
surface forms is model-specific scaffolding wearing a metric's clothes: it ports to nothing, and on
the next tokenizer it silently keeps working while measuring less.

**Multi-token forms: walk, do not enumerate.** Measured on MiniCPM5-2B's 130560-token vocabulary:
`"C"`, `" C"` and `"(C"` are single tokens, and **`"\\nC"` and `"**C"` are not**. So enumeration
cannot be the whole answer even on the model we ship. `divergence_walk` handles it without being
told the patterns exist: if the mass at the scoring position sits on a token that is not any
option's surface, then whatever the answer is, that token comes next -- it is a shared prefix, it
carries no answer information, so append it and re-score.

**The known asymmetry, and why it does not bite.** A letter with more tokenizer variants collects
more summed mass. That is a per-(model, letter) constant: it offsets every item's margin by a fixed
amount and CANCELS in the arm-to-arm comparison on identical items, the same cancellation by which
the shared Q8 floor cancels in our KL deltas. It would matter only when comparing margins ACROSS
models.

No model, no GPU, no inference: every scoring entry point takes the distribution as an argument.
That is what lets its three invariants (non-empty, disjoint, a walk that terminates) be proven before a
GPU-minute is spent, and it is also the only way the walk is testable at all -- an adversarial `dist_fn` that piles all
its mass on a shared token forever is one lambda here, and an unreproducible model quirk otherwise.

A note on the guards below: the disjointness invariant is NOT tautological even though `normalize_surface` is a
function and a token therefore normalises to exactly one string. It is reachable two ways, both tested: option
LABELS that collide under normalisation (`["C", "c"]`, `["C", "(C)"]`, and `["I", "ı"]` -- Turkish dotless i
uppercases to `I`), and sets arriving from anywhere other than `option_surface_sets` (a cached JSON, a union of two
vocabularies, a hand-augmented set). `option_surface_sets` deliberately does NOT pre-screen label collisions:
screening upstream would move the failure and leave the invariant with nothing it could ever reject.
"""
from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import ggufkv
from .errors import InvalidInput

NEG_INF = float("-inf")

# A distribution is accepted as a dense per-token vector (what a patched llama.cpp hands back) or
# as a sparse {id: logprob} mapping (what a top-k dump hands back). `TokenLogprobs` is either.
TokenLogprobs = Mapping[int, float] | Sequence[float]


class TopLogprobs(dict):
    """A top-k dump `{id: logprob}` that also carries what the upstream said about the pass.

    `eog_id`: the token llama-server picked GREEDILY here and stopped on as end-of-generation (`stop_type` "eos",
    `llama_vocab_is_eog`, which knows `<|im_end|>` and friends that no GGUF field lists), or None. The pick is the raw
    argmax only because the scorer pins a neutral sampler chain on every readout pass; the walk still honours it only
    when it IS the token it would append. `reasks`: how many times the pass was asked again because llama-server
    withheld its probabilities (scorer.py `HELD_BACK_BANS`). A plain mapping or vector says neither."""
    __slots__ = ("eog_id", "reasks")

    def __init__(self, items: Mapping[int, float] | Iterable[tuple[int, float]] = (), *, eog_id: int | None = None,
                 reasks: int = 0):
        super().__init__(items)
        self.eog_id, self.reasks = eog_id, reasks


def eog_token(token_logprobs: TokenLogprobs) -> int | None:
    """The upstream's end-of-generation greedy pick at this position (`TopLogprobs.eog_id`), None when it said none."""
    return token_logprobs.eog_id if isinstance(token_logprobs, TopLogprobs) else None


def reasks_of(token_logprobs: TokenLogprobs) -> int:
    return token_logprobs.reasks if isinstance(token_logprobs, TopLogprobs) else 0

# Slack on "a logprob is at most 0". Exactly 0.0 is legal (P = 1); a float32 softmax read back as
# float64 can land a hair above it, and refusing THAT would be refusing arithmetic rather than
# refusing bad input.
_LOGPROB_TOL = 1e-6
# Slack on "the option mass is at most 1", PER SUMMED TERM, on top of one `_LOGPROB_TOL`. The mass is
# a sum over every surface token of every option (30-48 ids per option; with a top-k dump, at most
# top_k of them finite), each a float32 probability, so its slack has to grow with the number of
# terms. A fixed 1e-9 -- below exp(_LOGPROB_TOL) - 1 itself -- refused near-certain answers (Gemma 4
# E4B and MiniCPM5-2B: mass 1 + ~4e-7) for arithmetic rather than for bad input.
#
# Where the values come from (llama.cpp tools/server @ 4ebdf2c; callo reads the default RAW,
# pre-sampling path, `get_token_probabilities` in server-common.cpp, never `post_sampling_probs`):
# e_i = expf(logit_i - max) in float32; a SEQUENTIAL float32 `cum_sum` over the whole vocabulary,
# the top n_probs first and the max first of all, so it starts at exactly 1; p_i = e_i / cum_sum in
# float32; then `std::log(p_i)` in float32 (server-task.cpp `logarithm`), widened exactly to double
# for the JSON. Per summed option term, with u = 2^-24:
#   - the sum: an addition errs by at most min(e_j, u * cum_j) (cum_j is representable, so rounding
#     never moves further than the addend). Any term not summed into the mass, including option spellings
#     below the top-k cut, shrinks cum_sum by at most its own mass, which it also takes out of 1, so only option terms can push the mass past 1: <= u
#     each (cum_j <= the final cum_sum);
#   - the division: relative error u on a p <= 1: <= u;
#   - the float32 log: |d ln p| <= 1 ulp <= 2u |ln p|, so dp <= 2u p|ln p| <= 2u/e = 0.74u;
#   - expf's own error cancels: cum_sum sums the same e_i, so the dump is still a normalised
#     distribution, just over slightly-wrong e_i.
# Total <= 2.74u = 1.6e-7 per term; 2^-22 = 4u = 2.4e-7 keeps ~1.5x margin on the bound. Plus ONE
# `_LOGPROB_TOL`: a near-certain logprob up to 1e-6 above 0 is legal per value, and only one term
# can carry it (two near-certain terms are already mass 2). Replaying that float32 path in numpy
# over 130k- and 262k-token vocabularies with near-certain option sets gave at most ~1.1e-8 per term
# (4.3e-7 at 40 terms): the bound is ~20x the worst observed. At callo's top_k = 100 the slack is at
# most 1e-6 + 100 * 2.4e-7 = 2.5e-5, so two certain options (mass 2), a mass of 1.001 and every
# mixed or unnormalised distribution are still refused.
_MASS_TOL_PER_TERM = 2.0 ** -22


def _mass_tol(n_terms: int) -> float:
    """The slack on "option mass <= 1" for a sum of `n_terms` finite float32 probabilities (see above)."""
    return _LOGPROB_TOL + n_terms * _MASS_TOL_PER_TERM


# Tokenizer families whose token STRINGS are byte-level-BPE-encoded, i.e. every character is a
# stand-in for one byte under `bytes_to_unicode`. Keyed on `tokenizer.ggml.model` as GGUF spells it.
BYTE_LEVEL_MODELS = frozenset({"gpt2"})
# SentencePiece: U+2581 LOWER ONE EIGHTH BLOCK for a leading space, `<0xNN>` for byte fallback. `gemma4` is llama.cpp's
# SPM-style BPE (escape_whitespaces, `<0xNN>` byte tokens; src/llama-vocab.cpp @ 1537a0a): the same two conventions.
SENTENCEPIECE_MODELS = frozenset({"llama", "spm", "gemma4"})

_SPM_METASPACE = "▁"


# --------------------------------------------------------------------------------------------
# 1. Decode
# --------------------------------------------------------------------------------------------

def bytes_to_unicode() -> dict[int, str]:
    """The gpt2 byte-to-unicode map: all 256 bytes to 256 distinct printable codepoints.

    Reproduced rather than imported because importing it would mean depending on `transformers` (or
    on `tiktoken`) for a 20-line lookup table, in a package whose whole point is to need nothing at
    probe time. The construction is the published one: keep the printable ASCII and Latin-1 ranges
    as themselves, then map the remaining 68 bytes to U+0100 upward in order -- which is why a space
    is `Ġ` (U+0120 = 256 + 32nd-unassigned) and a newline is `Ċ`.

    The reason this matters at all: in a byte-level vocabulary the token ` C` is STORED as `ĠC`. A
    decode step that skips this leaves ` C` unrecognised, and option C quietly loses the single most
    likely spelling of its own answer -- with no error, because the equivalence set is still
    non-empty (`C` itself is in it) and still disjoint. Both invariants pass while the metric
    measures less. Hence `test_decode_real_minicpm_ids`.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


_UNICODE_TO_BYTE = {v: k for k, v in bytes_to_unicode().items()}


def decode_token(tok: str, tok_model: str) -> str:
    """One vocabulary entry's stored string -> the text it actually stands for.

    `tok_model` is GGUF's `tokenizer.ggml.model`. Three behaviours, all deliberate:

    - byte-level (`gpt2`): reverse `bytes_to_unicode`, then UTF-8 decode. A token is a byte string,
      and a multi-byte character split across two tokens is normal, so `errors="replace"` rather
      than an exception -- half a UTF-8 sequence is a legitimate token, not a corrupt file.
    - SentencePiece (`llama`, `spm`, `gemma4`): metaspace to space and `<0xNN>` byte fallback.
    - anything else: identity, and that is the honest answer. Inventing a decoding for a family we
      have not checked would build wrong equivalence sets silently; identity degrades to exactly
      the naive behaviour, which is what a non-byte-level vocabulary wants anyway.

    Characters with no byte in the map are passed through as their own UTF-8 bytes. A true gpt2
    vocabulary has none (asserted against all 130560 MiniCPM tokens), so this is a safety net, not
    a path: it keeps one mislabelled entry from turning a whole-vocabulary decode into a traceback.
    """
    if tok_model in BYTE_LEVEL_MODELS:
        out = bytearray()
        for ch in tok:
            b = _UNICODE_TO_BYTE.get(ch)
            if b is None:
                out.extend(ch.encode("utf-8"))
            else:
                out.append(b)
        return out.decode("utf-8", errors="replace")
    if tok_model in SENTENCEPIECE_MODELS:
        if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
            try:
                return bytes([int(tok[3:5], 16)]).decode("utf-8", errors="replace")
            except ValueError:
                return tok
        return tok.replace(_SPM_METASPACE, " ")
    return tok


def decode_vocab(tokens: Sequence[str], tok_model: str) -> list[str]:
    """`decode_token` over a whole vocabulary. Separate entry point because it is the one the
    equivalence-set builder uses and the one worth caching per (model, tokenizer)."""
    return [decode_token(t, tok_model) for t in tokens]


# --------------------------------------------------------------------------------------------
# 2. Normalise
# --------------------------------------------------------------------------------------------

# Characters stripped from BOTH ENDS, repeatedly, before uppercasing. Four families, and the reason
# each is here rather than a general `not isalnum()` rule: a blanket rule would also strip the
# option labels of a non-Latin-script MCQ, and it would make the set unreviewable.
#
#   whitespace   -- ` C`, `\nC`, `\tC`
#   markdown     -- `**C**`, `_C_`, `` `C` ``, `~~C~~`
#   brackets     -- `(C)`, `[C]`, `{C}`, `<C>`
#   punctuation  -- `C.`, `C:`, `C)`, `- C`, and the quote families
#
# Deliberately OVER-inclusive at the edges: `-C` and `.C` are real MiniCPM tokens and both land in
# C's set. That is the right trade. Over-inclusion adds a per-(model, letter) constant, which the
# module docstring's asymmetry argument shows cancels in the arm-to-arm comparison; UNDER-inclusion
# drops mass that genuinely expressed the answer, and nothing cancels that.
#
# No alphanumerics may ever appear here -- a digit would reduce the option `1` to nothing and a
# letter would reduce `C` to nothing, in both cases producing an empty set rather than an error.
# Asserted by `test_wrapper_chars_contains_no_alphanumerics`.
# Spelled with escapes, not literal characters, for the two invisible ones: a bare NBSP or an
# ideographic space in source is indistinguishable from a plain space to every reviewer and to
# every copy-paste, and silently losing one would silently stop stripping it.
WRAPPER_CHARS = (
    " \t\n\r\f\v\xa0\u3000"      # whitespace, incl. NBSP and ideographic space
    "*_~`"                             # markdown emphasis / code
    "()[]{}<>"                         # brackets
    ".,:;!?-/\\|#+="                   # punctuation
    "\u2013\u2014"                     # en dash, em dash
    "'\""                              # straight quotes
    "\u2018\u2019\u201c\u201d\u00ab\u00bb"   # typographic quotes and guillemets
)


def normalize_surface(s: str) -> str:
    """Reduce one decoded surface form to its bare content, uppercased.

    ` C`, `C.`, `(C)`, `**C**`, `\\nC`, `.c` all reduce to `C`.

    Stripping is from the ENDS ONLY and runs to a fixed point, because the wrappers nest: `**(C)**`
    needs the markdown off before the bracket is at an end. An interior strip would make `A-B` a
    surface form of `AB` and quietly merge two distinct answers into one event.

    NFKC first, so the full-width `Ｃ` and the compatibility forms a web-scraped option list carries
    normalise onto the same letter as `C` rather than forming a silent second equivalence class.

    Uppercasing LAST, and it is what makes `c` a surface form of option `C` -- a model that answers
    in lower case answered. It is also the single sharpest edge in this module: Unicode casing is
    not a per-script identity, and `"ı".upper() == "I"`, so an option list carrying a Turkish
    dotless i collides with option `I` on any ten-option MMLU-Pro set. That collision is exactly
    what `check_option_sets` exists to catch, and it is tested.
    """
    s = unicodedata.normalize("NFKC", s)
    prev = None
    while prev != s:
        prev = s
        s = s.strip().strip(WRAPPER_CHARS)
    return s.upper()


# --------------------------------------------------------------------------------------------
# 3. Equivalence sets, and the two invariants over them
# --------------------------------------------------------------------------------------------

def option_surface_sets(tokens: Sequence[str], tok_model: str,
                        options: Sequence[str]) -> dict[str, frozenset[int]]:
    """{option label: the token ids whose single-token decode normalises to that option}.

    Built once per (model, tokenizer, option list) and reused by every probe forever. Keyed by the
    label AS GIVEN, not by its normalised form, so the caller gets back what it asked about.

    Rejects three caller inputs and screens for NOTHING else:

    - fewer than two options. `answer_margin` is `logP(correct) - max over OTHERS`, and with one
      option that `max` is over an empty sequence: `max() iterable argument is empty`, a message
      about our internals for a fault in theirs. Same failure `objective.py`'s module docstring is
      about, caught at the boundary instead.
    - a repeated label. Two identical keys are one key, so the caller would silently get fewer
      options than it passed.
    - a label that normalises to nothing (`"**"`, `"()"`, `""`). 2659 real MiniCPM tokens also
      normalise to empty, so such a label would collect thousands of unrelated tokens -- passing
      both invariants while measuring noise.

    What it deliberately does NOT screen is labels that COLLIDE under normalisation (`["C", "c"]`).
    Those produce overlapping sets and are `check_option_sets`'s business. Screening them here
    would move the failure upstream and leave the disjointness invariant unable to reject anything
    the builder can produce -- a guard that cannot fail.
    """
    if len(options) < 2:
        raise InvalidInput(
            f"need at least 2 options to marginalise over, got {list(options)}; a one-option "
            f"margin has no 'other options' to take a max over")
    dupes = sorted({o for o in options if list(options).count(o) > 1})
    if dupes:
        raise InvalidInput(f"repeated option label(s) {dupes}; labels key the result, so a repeat "
                           f"silently drops an option")
    empty_labels = sorted(o for o in options if not normalize_surface(o))
    if empty_labels:
        raise InvalidInput(
            f"option label(s) {empty_labels} normalise to the empty string; they would match every "
            f"token that is pure punctuation or whitespace")

    # One normalised key may be claimed by SEVERAL labels -- that is precisely the `["C", "c"]`
    # collision -- so this maps key -> every label wanting it, and each colliding label collects the
    # token independently. A `{key: label}` dict comprehension here keeps only the last label and
    # hands the earlier one an EMPTY set, which trips the non-emptiness guard instead of the
    # disjointness guard: the right rejection for the wrong reason, and a message pointing the
    # caller at the wrong half of their input. Caught by
    # `test_check_rejects_overlap_from_the_REAL_builder_case_collision`.
    wanted: dict[str, list[str]] = {}
    for o in options:
        wanted.setdefault(normalize_surface(o), []).append(o)
    acc: dict[str, set[int]] = {o: set() for o in options}
    for tid, tok in enumerate(tokens):
        key = normalize_surface(decode_token(tok, tok_model))
        if not key:
            continue
        for label in wanted.get(key, ()):
            acc[label].add(tid)
    # frozenset because these are a per-model cache handed to every probe: a mutable set given out
    # by reference is a cache one caller can edit for everybody, forever, silently.
    return {o: frozenset(acc[o]) for o in options}


def check_option_sets(sets: Mapping[str, Iterable[int]],
                      tokens: Sequence[str] | None = None, *, allow_empty: bool = False) -> None:
    """Invariants (a) non-empty and (b) disjoint (the module docstring lists all three).

    `allow_empty`: invariant (a) is waived (disjointness is still checked). Only wide-choice passes it, for a question
    whose two-letter labels are not all single tokens: such a label may have no surface token at all and is read
    through its first letter and a second-letter pass (`split_attribution`), never through its own set.

    Raises `InvalidInput` naming the offending option(s) and, for an overlap, the offending token
    ids -- and their stored strings too when `tokens` is supplied, because `token 371 is in both C
    and D` is a puzzle and `token 371 'ĠC'` is a diagnosis.

    Both guards are reachable and both are tested against inputs the real builder really produces
    on the real vocabulary; see the module docstring.

    Invariant (c) -- the walk terminates -- is not checkable here because it is a property of
    `divergence_walk`, not of the sets. It is structural there (bounded depth) and tested against an
    adversarial `dist_fn`.
    """
    if len(sets) < 2:
        raise InvalidInput(f"need at least 2 option sets, got {sorted(sets)}")
    empty = [] if allow_empty else sorted(o for o, s in sets.items() if not set(s))
    if empty:
        raise InvalidInput(
            f"option(s) {empty} have NO surface form in this vocabulary (invariant a). Every "
            f"probe against them would score zero mass and every margin would be infinite. Usual "
            f"cause: option TEXT passed where option LABELS were meant, or a label in a script "
            f"this tokenizer does not carry as a standalone token.")
    frozen = {o: frozenset(s) for o, s in sets.items()}
    if sum(map(len, frozen.values())) == len(frozenset().union(*frozen.values())):
        return                                     # disjoint: the pairwise loop below only runs to name an overlap
    collisions = _overlaps(frozen, tokens)
    if collisions:
        raise InvalidInput(
            "option surface sets are not disjoint (invariant b): " + "; ".join(collisions)
            + ". A token in two sets is counted in both marginals, so the margin between those two "
              "options is measuring the same probability mass against itself. Usual cause: option "
              "labels that collide under normalisation -- case ('C'/'c'), formatting ('C'/'(C)'), "
              "or Unicode casing ('I'/'ı').")


def _overlaps(frozen: Mapping[str, frozenset[int]], tokens: Sequence[str] | None) -> list[str]:
    """Every overlapping pair, described: the pairwise loop `check_option_sets` runs only once its O(total ids) count
    has found an overlap (the loop costs 5.5 ms at 255 labels, several times per readout)."""
    labels = list(frozen)
    collisions: list[str] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = frozen[a] & frozen[b]
            if shared:
                ids = sorted(shared)[:5]
                if tokens is not None:
                    shown = ", ".join(f"{t} {tokens[t]!r}" for t in ids if t < len(tokens))
                else:
                    shown = ", ".join(str(t) for t in ids)
                more = "" if len(shared) <= 5 else f" (+{len(shared) - 5} more)"
                collisions.append(f"{a!r}/{b!r} share {len(shared)} token(s): {shown}{more}")
    return collisions


def option_sets_from_gguf(path: str | Path, options: Sequence[str],
                          *, check: bool = True) -> dict[str, frozenset[int]]:
    """Equivalence sets straight from a GGUF's header. The entry point the metric will call.

    Reads `tokenizer.ggml.tokens` and `tokenizer.ggml.model` and nothing else: `GGUFReader` memory-
    maps the file, so this pulls a vocabulary out of a 5 GB BF16 model without touching a weight.

    `gguf` is imported lazily and on purpose -- every other function in this module is pure Python
    over plain lists, and making the whole module unimportable without llama.cpp's gguf-py on the
    path would take the invariant tests down with it on any box that does not have it.
    """
    reader = ggufkv.open_kv(path)
    tokens_field = reader.get_field("tokenizer.ggml.tokens")
    model_field = reader.get_field("tokenizer.ggml.model")
    if tokens_field is None or model_field is None:
        raise InvalidInput(f"{path} has no tokenizer vocabulary in its header "
                           f"(tokenizer.ggml.tokens / tokenizer.ggml.model); an mmproj or a split "
                           f"shard rather than the model?")
    tokens = [str(t) for t in tokens_field.contents()]
    sets = option_surface_sets(tokens, model_field.contents(), options)
    if check:
        check_option_sets(sets, tokens=tokens)
    return sets


def read_vocab_from_gguf(path: str | Path) -> tuple[list[str], str]:
    """The tokenizer's raw token list and model string, straight from the GGUF KV store. This is
    what build_secret_pool needs (option_sets_from_gguf reads the same store but returns surface
    sets, not the raw vocab)."""
    reader = ggufkv.open_kv(path)
    tokens_field = reader.get_field("tokenizer.ggml.tokens")
    model_field = reader.get_field("tokenizer.ggml.model")
    if tokens_field is None or model_field is None:
        raise InvalidInput(f"{path} has no tokenizer vocabulary in its header "
                           f"(tokenizer.ggml.tokens / tokenizer.ggml.model); an mmproj or a split "
                           f"shard rather than the model?")
    tokens = [str(t) for t in tokens_field.contents()]
    tok_model = str(model_field.contents())
    return tokens, tok_model


# --------------------------------------------------------------------------------------------
# 4. Marginalisation over an INJECTED distribution
#
# Every entry point below takes the distribution as an argument. That is not a testing
# convenience bolted on afterwards -- it is what makes this component finishable before a
# GPU-minute is spent, and it is the only way the walk is testable at all: an
# adversarial scorer that concentrates on a shared token forever is one lambda in a test, and an
# unreproducible model quirk otherwise.
# --------------------------------------------------------------------------------------------

def logsumexp(values: Iterable[float]) -> float:
    """`log(sum(exp(v)))`, shifted by the max so the exponentials cannot underflow.

    The shift is load-bearing, not hygiene. Option mass deep in a 130k-token vocabulary really does
    sit near `exp(-1000)`, which is 0.0 in float64: the naive form returns `-inf` -- a CONFIDENT
    claim that the option carries no mass at all -- and `-inf` propagates into the margin as a total
    failure that never happened.

    An empty sum, and a sum of nothing but `-inf`, are both `-inf`: zero probability, `log 0`. That
    is an answer, not an error, so neither raises.

    Validation is the CALLER's (`marginal_option_logprobs` does it, naming the token id). Keeping
    this function purely numeric means the one place that knows which token a value came from is the
    one place that reports it.
    """
    vals = list(values)
    if not vals:
        return NEG_INF
    top = max(vals)
    if top == NEG_INF:
        return NEG_INF
    return top + math.log(math.fsum(math.exp(v - top) for v in vals))


def _iter_dist(token_logprobs: TokenLogprobs) -> Iterator[tuple[int, float]]:
    """(token id, logprob) over a distribution in either accepted shape."""
    if isinstance(token_logprobs, Mapping):
        return iter(token_logprobs.items())
    return iter(enumerate(token_logprobs))


def _check_logprob(value: float, tid: int) -> float:
    """One value, validated where its token id is still known.

    `nan` is refused for the reason objective.py refuses a `nan` weight: it defeats every ordering
    guard downstream -- `nan < 0` and `nan > 0` are both False -- so it does not stop, it
    propagates, and it surfaces as an empty argmax somewhere with no trace of where it entered.

    A logprob above zero is a probability above one. Refusing it is what keeps `option_mass`
    meaningful: accept unnormalised log-SCORES here and the section-6 validity diagnostic quietly
    starts reporting a number that is not a probability. `-inf` is fine and expected.
    """
    if value != value:
        raise InvalidInput(f"token {tid} has a nan logprob; nan defeats every ordering guard "
                           f"downstream instead of stopping here")
    if value > _LOGPROB_TOL:
        raise InvalidInput(f"token {tid} has logprob {value!r} > 0, i.e. probability > 1; "
                           f"pass normalised log-probabilities, not logits or log-scores")
    return value


def _lookup(token_logprobs: TokenLogprobs, tid: int) -> float:
    """This token's logprob, `-inf` if a sparse mapping does not carry it.

    Absent-means-`-inf` is right for a top-k dump (absent means "below the cut") and is why
    `option_mass` exists separately: the reading is lossy, and the mass is how a caller sees that.

    A DENSE vector is different -- an id past its end is not "below the cut", it is a truncated
    dump or sets built against a different tokenizer, and reading it as zero mass would silently
    delete whichever options live in the vocabulary's tail. So that one raises.
    """
    if isinstance(token_logprobs, Mapping):
        return _check_logprob(token_logprobs.get(tid, NEG_INF), tid)
    if tid < 0 or tid >= len(token_logprobs):
        raise InvalidInput(
            f"token id {tid} is outside the dense distribution of length "
            f"{len(token_logprobs)}; a vector shorter than the vocabulary is a truncated dump or "
            f"equivalence sets built against a different tokenizer, not an absent token")
    return _check_logprob(token_logprobs[tid], tid)


def marginal_option_logprobs(sets: Mapping[str, Iterable[int]],
                             token_logprobs: TokenLogprobs) -> dict[str, float]:
    """{option: log P(answer = option)} at ONE scoring position. The core of the component.

    SUM over each option's surface forms, in log space. Not max: `"C"`, `" C"` and `"(C"` are one
    event, and taking the largest of them both understates the mass and -- demonstrated in
    `test_sum_not_max_changes_the_answer` -- can pick a different answer outright.

    The sets are re-validated on every call. They are cheap to check (ten options, a few dozen ids)
    and the invariants are not advisory: a token in two sets is counted in BOTH marginals, so the
    margin between those two options would be comparing probability mass against itself.

    Token ids are summed in sorted order so that the floating-point result is reproducible. Set
    iteration order is an implementation detail of the hash table, and a margin that shifts in its
    last bits between processes is a margin that cannot be compared to a journalled one.

    Returned in the ORDER OF `sets`, because `answer_choice` breaks ties on that order.

    RAW (not debiased, not clamped): a near-certain option can come back a hair above 0 (float32 rounding, see
    `_MASS_TOL_PER_TERM`). It stays raw for bit-stability with journalled margins (`answer_margin` reads these
    values; clamping cannot change the argmax). The readouts (`option_mass`, `WalkResult`) clamp.
    """
    return _marginals_and_terms(sets, token_logprobs)[0]


def _marginals_and_terms(sets: Mapping[str, Iterable[int]], token_logprobs: TokenLogprobs, *,
                         allow_empty: bool = False) -> tuple[dict[str, float], int]:
    """(the raw marginals, how many FINITE logprobs were summed into them): the count sizes `_mass_tol`.
    `allow_empty` as in `check_option_sets`: an empty set reads -inf."""
    check_option_sets(sets, allow_empty=allow_empty)
    marginals, n_terms = {}, 0
    for option, ids in sets.items():
        vals = [_lookup(token_logprobs, tid) for tid in sorted(ids)]
        n_terms += sum(v != NEG_INF for v in vals)
        marginals[option] = logsumexp(vals)
    return marginals, n_terms


def _checked_mass(marginals: Mapping[str, float], n_terms: int, where: str) -> float:
    """The option mass, refused when it is above 1 by more than float32 rounding of `n_terms` terms can explain,
    and CLAMPED to 1 otherwise, so no consumer ever sees a probability above 1 or a negative `1 - mass`."""
    mass = math.fsum(math.exp(v) for v in marginals.values())
    if not mass <= 1.0 + _mass_tol(n_terms):
        raise InvalidInput(
            f"option mass is {mass:.6f}{where}, above 1 by {mass - 1.0:.3g} (float32 rounding of {n_terms} "
            f"summed terms explains at most {_mass_tol(n_terms):.3g}): the option marginals do not come from a "
            f"single normalised probability distribution. Overlapping sets are ruled out by the disjointness "
            f"invariant, so this is unnormalised input or two distributions mixed together.")
    return min(mass, 1.0)


def clamp_marginals(marginals: Mapping[str, float]) -> dict[str, float]:
    """Each log-marginal clamped to <= 0, in order. Only ever moves an accepted one by float32 rounding (the mass
    check above bounds it); `-inf` stays `-inf`. Not bit-identical to the raw marginals in one rare case: a float-noise
    positive value below 2^-53 (where the float64 mass rounds to exactly 1.0) moves by less than 1.1e-16. The mass
    and the margins (raw) stay exact."""
    return {option: min(v, 0.0) for option, v in marginals.items()}


def option_mass(sets: Mapping[str, Iterable[int]], token_logprobs: TokenLogprobs, *,
                allow_empty: bool = False) -> float:
    """Total probability on the option set at this position -- a validity diagnostic.

    NOT a correction. `answer_margin` is mathematically invariant to it (a difference of logprobs
    cancels the normaliser), which is exactly why it has to be reported separately: the mass is how
    a caller learns whether the position is genuinely an answer slot or whether the ordering among
    options is being driven by formatting priors. The prompt construction was settled by measuring this, not
    by arguing about wording.

    Mass above 1 is refused here (and in `divergence_walk`, through the same `_checked_mass`):
    each contributing logprob can be individually legal (`P = 1` twice over) so no per-value check
    can see it. Two options that are each certain is not a distribution. Mass above 1 by no more
    than float32 rounding explains (`_mass_tol`) is a near-certain answer, and is returned as 1.0. `allow_empty` as
    in `check_option_sets`.
    """
    return _checked_mass(*_marginals_and_terms(sets, token_logprobs, allow_empty=allow_empty), "")


def answer_margin(sets: Mapping[str, Iterable[int]], token_logprobs: TokenLogprobs,
                  correct: str) -> float:
    """`logP(correct option) - max over OTHER options of logP(option)`.

    Against the best WRONG option, not against the runner-up overall and not against the mean: the
    margin is a decision boundary, and the boundary is whichever wrong answer is closest to winning.

    `+-inf` is meaningful and is returned: a quant that put no mass at all on the right answer
    scores `-inf`, and clamping that would invent a floor nobody chose. `nan` is NOT returned --
    both sides `-inf` is `-inf - -inf`, and a `nan` in a per-item margin silently poisons every
    mean, percentile and calibration weight computed from it downstream.
    """
    if correct not in sets:
        raise InvalidInput(f"correct option {correct!r} is not one of {sorted(sets)}")
    marginals = marginal_option_logprobs(sets, token_logprobs)
    # `check_option_sets` inside the marginal already refused a single-option set, which is what
    # keeps this `max` off an empty sequence -- the `max() iterable argument is empty` failure
    # objective.py's module docstring is about, reported as a fault in code the caller cannot see.
    best_other = max(v for k, v in marginals.items() if k != correct)
    margin = marginals[correct] - best_other
    if margin != margin:
        raise InvalidInput(
            f"margin for {correct!r} is nan: this position carries no option mass at all "
            f"(every option is -inf), so there is no decision here to measure. Check "
            f"`option_mass` and the scoring position before scoring the item.")
    return margin


def answer_choice(sets: Mapping[str, Iterable[int]], token_logprobs: TokenLogprobs) -> str:
    """The argmax OPTION -- a free consequence of the option sets, and a better-founded top-1 than the
    token-level `top1_disagree` the objective weights today.

    Token-level top-1 counts a reference and a quant that put their largest token on two different
    SPELLINGS of the same letter as a disagreement. There is none: both answered C. That metric is
    partly measuring formatting churn, and it is weighted in our objective right now
    (`test_option_agreement_is_better_founded_than_token_top1`).

    Ties go to the first option in `sets`' order. Deterministic on purpose: a tie broken by dict
    iteration accident would make an arm-to-arm agreement RATE depend on how the option list
    happened to be spelled.
    """
    marginals = marginal_option_logprobs(sets, token_logprobs)
    best, best_lp = None, NEG_INF
    for option, lp in marginals.items():
        if lp > best_lp:           # strict: the first option wins a tie
            best, best_lp = option, lp
    if best is None:
        raise InvalidInput(
            f"no option carries any mass at this position (all {len(marginals)} marginals are "
            f"-inf), so there is no argmax option. Check the scoring position.")
    return best


def answer_agreement(sets: Mapping[str, Iterable[int]], token_logprobs: TokenLogprobs,
                     reference: str | TokenLogprobs) -> bool:
    """Does this distribution's argmax option match the reference's?

    `reference` is either the reference's already-known option label or the reference's own
    distribution at the same position. Both spellings exist because both callers exist: a screening
    pass holds the reference choices from an earlier run, and a paired pass holds two live
    distributions and should not have to reduce one of them by hand.
    """
    if isinstance(reference, str):
        if reference not in sets:
            raise InvalidInput(f"reference option {reference!r} is not one of {sorted(sets)}")
        ref_choice = reference
    else:
        ref_choice = answer_choice(sets, reference)
    return answer_choice(sets, token_logprobs) == ref_choice


# --------------------------------------------------------------------------------------------
# 5. The trie-divergence walk -- invariant (c)
# --------------------------------------------------------------------------------------------

# Deep enough for the preambles that actually occur -- `The answer is ` is four tokens, `**` plus a
# newline is two -- and shallow enough that the worst case is a bounded cost rather than a hang.
# Each step is a real forward pass in production, so this is a budget, not a formality.
DEFAULT_MAX_DEPTH = 8
# Stop as soon as the options carry this much of the mass: the position has answer information, so
# walking past it would step OVER the decision we came to measure.
DEFAULT_DIVERGENCE_MASS = 0.10
# Only append a token this dominant. Below it, the position is genuinely uncertain and appending
# its argmax would be a guess about what the model was going to say.
DEFAULT_DOMINANCE = 0.50


@dataclass(frozen=True)
class WalkResult:
    """Where the walk stopped, and everything scored there.

    Carries the distribution and the sets so a caller can ask follow-up questions (`choice()`,
    `margin()`) without re-deriving which position the walk settled on -- the position IS the
    result, and re-deriving it by hand is how two call sites end up scoring different tokens.

    `option_logprobs` is CLAMPED to <= 0, `margin()` is computed from the raw `token_logprobs`: in the degenerate
    near-certain case (float32 overshoot) the two can differ by up to ~1e-6.

    Wide-choice: `partial_mass` is the mass at the stop position on the walk's `partial` tokens (0.0
    without them); `n_terms` is how many finite option logprobs were summed there (it sizes `split_attribution`'s
    final mass slack).
    """
    prefix: tuple[int, ...]
    token_logprobs: TokenLogprobs
    option_logprobs: dict[str, float]
    option_mass: float
    depth: int
    stop: str
    sets: Mapping[str, frozenset[int]] = field(repr=False, default_factory=dict)
    partial_mass: float = 0.0
    n_terms: int = 0

    def choice(self) -> str:
        return answer_choice(self.sets, self.token_logprobs)

    def margin(self, correct: str) -> float:
        return answer_margin(self.sets, self.token_logprobs, correct)


def _validate_dist(token_logprobs: TokenLogprobs, prefix: tuple[int, ...]) -> None:
    """Every value of a distribution the walk is about to SCAN, checked once.

    `marginal_option_logprobs` validates only the values it reads, which is the right cost there
    (a few dozen ids out of 130k). The walk takes an argmax over the whole distribution, so it is
    already paying for a full scan and may as well validate all of it.

    An EMPTY distribution is refused rather than read as "no option mass": read that way, a scorer
    that simply failed would walk the full depth and return a confident `max_depth` result built on
    nothing at all.
    """
    items = list(_iter_dist(token_logprobs))
    if not items:
        raise InvalidInput(f"dist_fn returned an empty distribution at prefix {prefix}; a scorer "
                           f"that returned nothing has failed, and reading that as 'no option "
                           f"mass' would walk the full depth and report a result built on no data")
    for tid, value in items:
        _check_logprob(value, tid)


def divergence_walk(sets: Mapping[str, Iterable[int]],
                    dist_fn: Callable[[tuple[int, ...]], TokenLogprobs],
                    *,
                    prefix: Sequence[int] = (),
                    max_depth: int = DEFAULT_MAX_DEPTH,
                    divergence_mass: float = DEFAULT_DIVERGENCE_MASS,
                    dominance: float = DEFAULT_DOMINANCE,
                    partial: Iterable[int] = (),
                    allow_empty: bool = False) -> WalkResult:
    """Walk forward through tokens that carry no answer information, and score where they end.

    `"**C**"` begins with `"**"`, which every option shares, so that position says nothing about
    the answer. Rather than enumerate multi-token surface forms -- which would mean guessing which
    patterns a model uses, model by model -- observe it: if the mass at this position sits on a
    token that is no option's surface form, then whatever the answer turns out to be, that token
    comes next. It is a shared prefix. Append it and re-score. This finds `(C)`, `**C**` and
    `The answer is C` without being told those patterns exist, which is the difference between a
    component that ports to a new tokenizer and one that has to be re-taught.

    It also does real work rather than notional work: `"\\nC"` and `"**C"` are NOT single tokens in
    MiniCPM5-2B's 130560-token vocabulary, so on the model we ship, enumeration alone would miss
    the newline and bold spellings entirely.

    `dist_fn(prefix_token_ids) -> token logprobs` is INJECTED. That is what makes invariant (c)
    provable without a model.

    **Termination (invariant c).** Every iteration either returns or appends exactly one token and
    increments `depth`, and `depth >= max_depth` returns. So the walk performs at most
    `max_depth + 1` scoring positions, whatever `dist_fn` does -- including a `dist_fn` that puts
    all its mass on a shared token forever, which is tested.

    **It cannot swallow the answer.** `dominance >= divergence_mass` is enforced below. A token
    dominant enough to append therefore carries at least `dominance` of the mass; were it an
    option's own surface form, that option's marginal would be at least that much, so the
    divergence test -- evaluated FIRST -- would already have stopped the walk. An appended token is
    provably not an option surface, so no explicit membership test is needed here. That is a proof,
    not an omission: a membership test would be unreachable code, which is the "check that cannot
    fail" defect in another costume.

    **Partial labels.** A two-letter label `AB` may be spelled `ĠA` + `B`: its first token is no
    option's surface, so on its own the proof above would let a dominant `ĠA` be appended as "neutral" and the walk
    would step INTO the answer. `partial` names those first-letter tokens. Their mass counts toward the divergence
    test (`mass + partial_mass`) but never toward the option mass, and they must be disjoint from every option set
    (a token in both would be counted twice). The proof extends unchanged: an appended token carries at least
    `dominance` >= `divergence_mass`; were it an option surface OR a partial token, `mass + partial_mass` would
    already have stopped the walk. With `partial` empty the comparison is `mass + 0.0 >= divergence_mass`: today's,
    bit for bit.

    **End of generation.** A dominant token the model would END ITS TURN on (`eog_token`, the upstream's own
    `llama_vocab_is_eog` verdict on its greedy pick) is no option surface either, but it is not answer-neutral: what
    follows it is outside the reply, so the position after it says nothing about the answer (on Sharp-MiniCPM5-2B it
    was an emoji's lead byte, a reply llama-server sends without probabilities). It is never appended: the walk scores
    HERE, the options' conditional, as for `no_shared_prefix`, with `option_mass` saying how little the model meant to
    answer at all. The greedy pick is the raw argmax because the scorer pins a neutral sampler chain on the pass (no
    bias, penalty or truncation can move it); the verdict counts only when the token it names IS the dominant one, so
    a server that moved the pick anyway can make the walk miss an EOG, never invent one. It is checked before the
    depth bound, so a dominant EOG there is reported as `eog`, not `max_depth`.

    **Labels with no surface token.** `allow_empty` (wide-choice only, with `partial`) accepts option sets that are
    empty: a two-letter label that is not a single token in this vocabulary is spelled first letter + second letter,
    so its mass at this position is all on `partial` tokens, which stop the walk as above. Its marginal here is -inf
    and `split_attribution` reads it. The proof is unchanged (an empty set holds no token to append). The returned
    `sets` then hold the empty sets too, so `choice()` and `margin()` (which check invariant (a)) are not for such a
    walk.

    `stop` is one of:
      `diverged`          -- the options carry real mass here; this is the answer slot.
      `no_shared_prefix`  -- nothing dominates, so appending would be a guess. Score here, and read
                             `option_mass` to see whether the position was an answer slot at all.
      `eog`               -- the dominant token ends the turn. Score here, as for `no_shared_prefix`.
      `max_depth`         -- the bound. A run that hits this often has the wrong scoring position,
                             not a too-small bound.
    """
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise InvalidInput(f"max_depth must be a non-negative int, got {max_depth!r}")
    for name, value in (("divergence_mass", divergence_mass), ("dominance", dominance)):
        # Written as a positive test so nan fails it: every comparison against nan is False, so
        # `not (0 < nan <= 1)` is True and nan is rejected, where `if value <= 0 or value > 1`
        # would wave it through and make every later threshold test silently False.
        if not 0.0 < value <= 1.0:
            raise InvalidInput(f"{name} must be in (0, 1], got {value!r}")
    if dominance < divergence_mass:
        raise InvalidInput(
            f"dominance ({dominance}) must be at least divergence_mass ({divergence_mass}): "
            f"below it, a token could be dominant enough to append AND be an option's own surface "
            f"form, and the walk would step past the answer it was measuring")
    frozen = {o: frozenset(ids) for o, ids in sets.items()}
    check_option_sets(frozen, allow_empty=allow_empty)
    part = frozenset(partial)
    if part:   # disjoint from every option, by the same invariant (a partial token in an option set is mass counted twice)
        check_option_sets({**frozen, "\0partial": part}, allow_empty=allow_empty)

    cur = tuple(prefix)
    depth = 0
    while True:
        token_logprobs = dist_fn(cur)
        _validate_dist(token_logprobs, cur)
        raw, n_terms = _marginals_and_terms(frozen, token_logprobs, allow_empty=allow_empty)
        where = f" at prefix {cur} (dist_fn)"
        # Clamped (mass to 1, each marginal to <= 0) only once the check has accepted it as rounding.
        mass = _checked_mass(raw, n_terms, where)
        pmass = 0.0
        if part:
            pvals = [_lookup(token_logprobs, t) for t in sorted(part)]
            p_lp = logsumexp(pvals)
            _checked_mass({**raw, "\0partial": p_lp}, n_terms + sum(v != NEG_INF for v in pvals), where)
            pmass = min(math.exp(p_lp), 1.0)
        marginals = clamp_marginals(raw)

        # `done` is redefined each iteration; binding the loop state as defaults captures THIS
        # iteration's values (it is always called immediately below, before cur/depth advance).
        def done(stop: str, *, cur=cur, token_logprobs=token_logprobs,
                 marginals=marginals, mass=mass, depth=depth, pmass=pmass, n_terms=n_terms) -> WalkResult:
            return WalkResult(prefix=cur, token_logprobs=token_logprobs,
                              option_logprobs=marginals, option_mass=mass, depth=depth,
                              stop=stop, sets=frozen, partial_mass=pmass, n_terms=n_terms)

        # Divergence FIRST. A position that already carries answer information is the position we
        # came for, and walking past it would step over the decision -- see the proof above, which
        # depends on this ordering.
        if mass + pmass >= divergence_mass:        # `+ 0.0` when part is empty: today's comparison, bit for bit
            return done("diverged")
        top_id, top_lp = max(_iter_dist(token_logprobs), key=lambda kv: kv[1])
        dominant = math.exp(top_lp) >= dominance
        if dominant and eog_token(token_logprobs) == top_id:     # the turn ends here: never read past it
            return done("eog")
        if depth >= max_depth:
            return done("max_depth")
        if not dominant:
            return done("no_shared_prefix")
        cur = cur + (top_id,)
        depth += 1


# --------------------------------------------------------------------------------------------
# 6. Wide-choice: partial labels, continuations, split attribution
# --------------------------------------------------------------------------------------------

WIDE_SPLIT_READ = 0.01          # split passes run only while the unread partial mass exceeds this
WIDE_MAX_SPLIT_PASSES = 2       # attempts (retok skips included) are capped at this + 2


def partial_sets(tokens: Sequence[str], tok_model: str, letters: Sequence[str]) -> dict[str, frozenset[int]]:
    """{letter: ids of the tokens that START a two-letter label with it}: the surface normalises to the letter AND the
    NFKC decode ENDS in it ("A", " A", "(A", "**A"; never "A." or "A)", which no second letter can follow)."""
    want = {c.upper(): c for c in letters}
    acc: dict[str, set[int]] = {c: set() for c in letters}
    for tid, tok in enumerate(tokens):
        s = unicodedata.normalize("NFKC", decode_token(tok, tok_model))
        c = want.get(normalize_surface(s))
        if c is not None and s and s[-1].upper() == c.upper():
            acc[c].add(tid)
    return {c: frozenset(acc[c]) for c in letters}


def continuation_sets(tokens: Sequence[str], tok_model: str, letters: Sequence[str]) -> dict[str, frozenset[int]]:
    """{letter: ids of the tokens that CONTINUE a two-letter label with that letter}: the NFKC decode starts with the
    letter itself (no leading whitespace or wrapper; any case) and only wrappers follow ("B", "b", "B.", "Ｂ").
    `A B` is not the label AB, so a leading space disqualifies."""
    want = {c.upper(): c for c in letters}
    acc: dict[str, set[int]] = {c: set() for c in letters}
    for tid, tok in enumerate(tokens):
        s = unicodedata.normalize("NFKC", decode_token(tok, tok_model))
        if not s or s[0] in WRAPPER_CHARS or s[0].isspace():
            continue
        c = want.get(s.rstrip(WRAPPER_CHARS).upper())
        if c is not None:
            acc[c].add(tid)
    return {c: frozenset(acc[c]) for c in letters}


@dataclass(frozen=True)
class SplitResult:
    credit: dict[str, float]            # label order: log of the split credit, -inf when none
    option_logprobs: dict[str, float]   # label order: log(direct + credit), clamped <= 0, NOT floored
    option_mass: float                  # direct + credit over the labels: checked, clamped to 1
    unread_mass: float                  # partial mass at the slot no split pass attributed (an upper bound)
    split_passes: int
    expanded: tuple[int, ...]
    partial_by_letter: tuple[tuple[str, float], ...]    # (first letter, partial mass at the slot), letter order


def split_attribution(walk: WalkResult, labels: Sequence[str], partial: Mapping[str, frozenset[int]],
                      cont: Mapping[str, frozenset[int]],
                      dist_fn: Callable[[tuple[int, ...]], TokenLogprobs] | None, *,
                      read_above: float = WIDE_SPLIT_READ, max_passes: int | None = WIDE_MAX_SPLIT_PASSES,
                      min_partial: float = 0.0,
                      skip: type[BaseException] | tuple[type[BaseException], ...] = (),
                      reserve: Collection[str] = ()) -> SplitResult:
    """Credit the slot's FIRST-LETTER-ONLY mass (`partial[f]`) to the two-letter labels f starts:
        credit(fx) = SUM over expanded t in partial[f] of p(t) * SUM_{u in cont[x]} p(u | slot + t).
    Exact per expanded token; never spread: the partial mass not expanded is `unread_mass`. Expansion order: p(t)
    descending, then id. It stops once the unread mass is <= `read_above`, the next candidate is <= `min_partial`, or
    after `max_passes` passes or `max_passes + 2` attempts (None: no cap; the forced-split diagnostic). `dist_fn` None
    (a retok readout) expands nothing. `skip`: exceptions a pass raises BEFORE posting (the media path's _RetokStop):
    that candidate stays unread and costs an attempt, not a pass.

    `reserve`: first letters that each keep one pass of the cap (wide-choice: the letters leading a label that is not a
    single token, whose mass is all first-letter mass). While letters of `reserve` with a candidate above `read_above`
    are still unread and the passes left are no more than their count, a candidate of any other letter is skipped (no
    attempt), so the likeliest spelling of each reserved letter is read before the cap runs out. Empty: the expansion
    is exactly the greedy one.

    Invariant: `option_mass + unread_mass <= 1`, since SUM_x P(cont[x] | slot + t) <= 1 for every expanded t. It is
    asserted (`_checked_mass`) with the slack of every summed term: the slot's option terms (`walk.n_terms`), the
    partial terms, and each split pass's finite continuation terms, so a bug is an `InvalidInput`, never a
    silent mass above 1."""
    D, s = walk.token_logprobs, walk.prefix
    firsts = sorted({lab[0] for lab in labels})
    cands = sorted((-lp, tid, f, lp) for f in firsts for tid in partial[f]
                   if (lp := _lookup(D, tid)) != NEG_INF)
    if cands and walk.partial_mass == 0.0:
        # the slot holds first-letter mass the walk never counted: it was walked without `partial`, so it may have
        # stepped over a dominant first letter -- a caller bug, never read as if it were a slot
        raise InvalidInput("split_attribution: the walk was not given the partial set")
    by_letter = tuple((f, math.fsum(math.exp(c[3]) for c in cands if c[2] == f)) for f in firsts)
    total = math.fsum(math.exp(c[3]) for c in cands)
    add: dict[str, list[float]] = {lab: [] for lab in labels}
    read: list[float] = []
    expanded: list[int] = []
    passes = attempts = n2_all = 0
    pending = {f for _n, _t, f, lp in cands if f in reserve and math.exp(lp) > read_above}
    for _neg, tid, f, lp in cands:
        if dist_fn is None:
            break
        if max_passes is not None and (passes >= max_passes or attempts >= max_passes + 2):
            break
        if total - math.fsum(read) <= read_above or math.exp(lp) <= min_partial:
            break                                   # sorted descending: every later candidate is no larger
        if max_passes is not None and f not in pending and max_passes - passes <= len(pending):
            continue                                # the passes left are held for the reserved letters
        attempts += 1
        try:
            d2 = dist_fn(s + (tid,))
        except skip:
            continue
        passes += 1
        _validate_dist(d2, s + (tid,))
        cl, n2 = {}, 0
        for x in sorted({lab[1] for lab in labels if lab[0] == f}):
            vals = [_lookup(d2, u) for u in sorted(cont[x])]
            n2 += sum(v != NEG_INF for v in vals)
            cl[x] = logsumexp(vals)
        _checked_mass(cl, n2, f" after split token {tid} at prefix {s}")     # continuations of one position <= 1
        n2_all += n2
        for lab in labels:
            if lab[0] == f:
                add[lab].append(lp + min(cl[lab[1]], 0.0))
        read.append(math.exp(lp))
        expanded.append(tid)
        pending.discard(f)
    credit = {lab: logsumexp(add[lab]) for lab in labels}
    out = {lab: min(logsumexp([walk.option_logprobs[lab], credit[lab]]), 0.0) for lab in labels}
    unread = max(total - math.fsum(read), 0.0)
    _checked_mass({**out, "\0unread": math.log(unread) if unread > 0 else NEG_INF},
                  walk.n_terms + len(cands) + n2_all, " (split readout)")             # every summed term
    return SplitResult(credit, out, min(math.fsum(math.exp(v) for v in out.values()), 1.0), unread, passes,
                       tuple(expanded), by_letter)
