"""Each gated v0.2 bench row's PASS branch and a FAIL branch, on fixed timings and probabilities, pinning the bench's
thresholds (readout mean P(gold) 0.95 and 36/36; 2 Hz median 411 ms, 1.95 Hz, 27 tok/s; G0 104 of 106; walk reuse
cache_n / prompt_n 1 / 40 ms; system- and context-prefix cache_n). No server: the rows' I/O helpers are swapped."""
import json

import pytest

from bench import acceptance_v02 as v02
from bench import lib
from bench.frames import DIRS, maze_frame, spike_frames
from bench.report_v02 import render
from callo.scorer import MMResult


class FakeC:
    url, upstream, pid, build_info = "http://callo.invalid", "http://up.invalid", None, "fake"

    def __init__(self, lanes=None, lines=()):
        self._lanes = {"s1": 0, "aux": 1} if lanes is None else lanes
        self.lines = list(lines)

    def stats(self):
        return {"router": {"lanes": self._lanes}, "vision": True}

    def log_text(self):
        return "\n".join(self.lines)


class FakeScorer:
    def __init__(self, results):
        self.results = list(results)

    def next_logprobs_mm(self, text, images, *, top_k, id_slot=None, delimiter=None):
        return self.results.pop(0)


class FakeEng:
    media_marker, user_delimiter, top_k = "<m>", "<|im_start|>user\n", 5

    def __init__(self, results=()):
        self.scorer = FakeScorer(results)

    def render_prompt(self, state, q, *, context="", image_urls=()):
        return "SYS" + self.user_delimiter + self.media_marker + "Q"


def ok(qid, probs=None, cache_n=0, slot="1"):
    return {"ok": True, "status": 200, "slot": slot, "error": None,
            "timings": [{"q": qid, "cache_n": cache_n, "prompt_n": 5, "prompt_ms": 10.0}],
            "answers": {qid: {"probabilities": probs or {"up": 0.7, "down": 0.1, "left": 0.1, "right": 0.1}}}}


@pytest.fixture
def ctx(tmp_path):
    return v02.V02Ctx(tmp_path / "out", hz_seconds=20.0, gguf="unused")


# ---- readout: 36/36 and mean P(gold) >= 0.95 ------------------------------------------------------------

@pytest.mark.parametrize(("p_gold", "first_wrong", "expect"), [(0.96, False, True), (0.94, False, False),
                                                               (0.99, True, False)])
def test_readout_gate(ctx, monkeypatch, p_gold, first_wrong, expect):
    frames = spike_frames()
    gold_of = {v02.data_url(path.read_bytes()): gold for gold, _, _, path in frames}
    first = v02.data_url(frames[0][3].read_bytes())

    def post(c, body, lane):
        gold = gold_of[body["images"][0]]
        p = p_gold if not (first_wrong and body["images"][0] == first) else 0.1
        rest = [d for d in DIRS if d != gold]
        return ok("dir", {gold: p, rest[0]: 1 - p - 0.02, rest[1]: 0.01, rest[2]: 0.01})

    monkeypatch.setattr(v02, "s1_post", post)
    monkeypatch.setattr(ctx, "engine", lambda c: FakeEng())
    out = v02.row_readout(FakeC(), ctx)
    assert out["invalid"] is None and out["n"] == 36 and out["correct"] == (35 if first_wrong else 36)
    assert out["pass"] is expect


# ---- 2 Hz while System 2 streams: median <= 411 ms, >= 1.95 Hz, S2 >= 27 tok/s ---------------------------

def _stream_class(n_deltas, *, problems=None, slots=("2",)):
    class FakeStream:
        def __init__(self, url, payload_fn, *, headers=None, restart_every=None):
            self.requests = [{"slot": s, "error": None, "t_send": 0.0, "t_first": 0.1} for s in slots]
            self.deltas = [i * 20.0 / n_deltas for i in range(n_deltas)]     # all inside the [0, 20] s window
            self.stuck = False

        def problems(self):
            return problems
    return FakeStream


def _ticks(n, latency_ms, slot="0"):
    rows = [{"ok": True, "status": 200, "slot": slot, "error": None, "timings": [], "latency_ms": latency_ms,
             "t_start": k * 20.0 / n, "t_end": k * 20.0 / n + 0.3} for k in range(n)]
    rows[-1]["t_end"] = 20.0
    return rows


@pytest.mark.parametrize(("n_ticks", "latency", "n_deltas", "expect"), [
    (39, 411.0, 540, True),       # exactly 1.95 Hz, a 411 ms median, 27.0 tok/s: the boundary passes
    (40, 411.5, 540, False),      # median over 411 ms
    (38, 400.0, 540, False),      # 1.90 Hz
    (40, 400.0, 539, False),      # 26.95 tok/s
])
def test_hz2_stream_gate(ctx, monkeypatch, n_ticks, latency, n_deltas, expect):
    monkeypatch.setattr(lib, "ChatStream", _stream_class(n_deltas))
    monkeypatch.setattr(v02, "_with_s2", lambda c, ctx, s2, lane, salt0, sample_memory=False:
                        (_ticks(n_ticks, latency), []))
    out = v02.row_hz2_stream(FakeC(), ctx)
    assert out["invalid"] is None and out["pass"] is expect


@pytest.mark.parametrize(("kw", "reason"), [({"problems": "1 System 2 request(s) errored"}, "errored"),
                                            ({"slots": ("1",)}, "S2 ran on slots")])
def test_hz2_stream_is_invalid_when_system_2_misbehaved(ctx, monkeypatch, kw, reason):
    monkeypatch.setattr(lib, "ChatStream", _stream_class(600, **kw))
    monkeypatch.setattr(v02, "_with_s2", lambda c, ctx, s2, lane, salt0, sample_memory=False: (_ticks(40, 300.0), []))
    out = v02.row_hz2_stream(FakeC(), ctx)
    assert reason in out["invalid"] and out["pass"] is False


# ---- G0 parity: decision-equivalence (prompt + argmax 106/106, max |Δp| < 1e-3), on its own instance ------

class FakeInstance:
    """The standalone g0_parity instance, stubbed: `_instance` returns it, nothing starts a server."""
    parallel, upstream, build_info = 4, "http://up.invalid", "fake"

    def finish(self):
        pass


G0P = {"up": 0.8, "down": 0.2, "left": 0.0, "right": 0.0, "stay": 0.0}


@pytest.mark.parametrize(("rec42", "expect"), [
    ({**G0P, "up": 0.8 + 3e-5, "down": 0.2 - 3e-5}, True),       # within the 1e-3 tolerance, argmax kept: PASS
    ({**G0P, "up": 0.2, "down": 0.8}, False),                    # argmax flip: hard FAIL
    ({**G0P, "up": 0.798, "down": 0.202}, False),                # |Δp| 2e-3 over tolerance, argmax intact: FAIL
])
def test_g0_parity_gate(ctx, monkeypatch, tmp_path, rec42, expect):
    body = v02.tick_body(maze_frame(224, "up", 2, salt=1))
    recs = tmp_path / "g0.jsonl"
    recs.write_text("".join(json.dumps({"pair": i, "media_marker": "<g0m>", "prompt_string": "SYS<|im_start|>user\n"
                                        "<g0m>Q", "body": body, "probs": G0P}) + "\n" for i in range(106)))
    ctx.g0_records = recs
    answers = iter([dict(G0P)] * 42 + [rec42] + [dict(G0P)] * 63)   # pair 42 is the perturbed record
    monkeypatch.setattr(v02, "_instance", lambda name, out, *, model, flags, ctx=None: FakeInstance())
    monkeypatch.setattr(v02, "erase_slots", lambda c, n: None)
    monkeypatch.setattr(v02, "s1_post", lambda c, body, lane: ok("move", next(answers)))
    monkeypatch.setattr(ctx, "engine", lambda c: FakeEng())
    out = v02.row_g0_parity(ctx, tmp_path)
    assert out["prompt_equal"] == 106 and out["slots_seen"] == ["1"] and out["pass"] is expect
    assert out["agree"] == (105 if rec42["up"] < rec42["down"] else 106)   # only the flip drops the agreement


def test_g0_parity_invalid_on_missing_or_bad_records(ctx, tmp_path):
    ctx.g0_records = None
    assert v02.row_g0_parity(ctx, tmp_path)["pass"] is False        # no records file: never launches a server
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"pair": 0, "media_marker": "<m>"}) + "\n")   # lacks prompt_string/body/probs
    ctx.g0_records = bad
    out = v02.row_g0_parity(ctx, tmp_path)
    assert out["pass"] is False and "record 0 lacks" in out["invalid"]


# ---- kv_sensitivity: ungated, two instances, s0 vs s2 on -kvu and --no-kvu --------------------------------

def test_kv_sensitivity_runs_both_configs_and_summarises(ctx, monkeypatch, tmp_path):
    launched = []

    def instance(name, out, *, model, flags, ctx=None):
        launched.append((name, flags))
        return FakeInstance()

    # s0 confident up; s2 flips to down only on -kvu (the MoE sensitivity), bit-exact on --no-kvu
    def engine_for(name):
        up_s2 = 0.2 if "kvu" in name and "no_kvu" not in name else 0.9

        class E:
            def answer_request(self, body, *, id_slot=None):
                qid = next(iter(body["questions"]))
                p = up_s2 if E._second else 0.9
                E._second = not E._second
                return {"answers": {qid: {"probabilities": {"up": p, "down": 1 - p}}}}
        E._second = False
        return E()

    engines = {}
    monkeypatch.setattr(v02, "_instance", instance)
    monkeypatch.setattr(v02, "erase_slots", lambda c, n: None)
    monkeypatch.setattr(v02, "fill_slot", lambda c, slot, n: None)
    item = {"state": "x", "context": "c", "questions": {"q": {"type": "choice", "criteria": {"up": None,
                                                                                             "down": None}}}}
    monkeypatch.setattr(v02, "KV_ITEMS", [item] * 40)
    monkeypatch.setattr(ctx, "engine", lambda c: engines.setdefault(len(engines), engine_for(launched[-1][0])))
    out = v02.row_kv_sensitivity(ctx, tmp_path)
    assert [n for n, _ in launched] == ["kv_sensitivity_kvu", "kv_sensitivity_no_kvu"]
    assert out["no_kvu_expected_bit_exact"] is True and out["pass"] is None and out["invalid"] is None
    assert out["kvu"]["argmax_agree"] == 0 and out["kvu"]["max_dp"] == pytest.approx(0.7)       # -kvu: flipped
    assert out["no_kvu"]["argmax_agree"] == 40 and out["no_kvu"]["max_dp"] == pytest.approx(0.0)  # --no-kvu: bit-exact
    assert "--no-kvu" in launched[1][1] and "--no-kvu" not in launched[0][1]


# ---- cache_drift: the cold pass must be truly cold (cache_n 0), else INVALID, never PASS ------------------

@pytest.mark.parametrize(("cold_cache_n", "invalid_sub"), [(0, None), (349, "not cold")])
def test_cache_drift_invalid_when_cold_pass_is_not_cold(ctx, monkeypatch, cold_cache_n, invalid_sub):
    cache_ns = iter([cold_cache_n] + [0] * 21)                     # cold, then 20 others, then warm
    erased = []
    monkeypatch.setattr(v02, "erase_slots", lambda c, n: erased.append(n))
    monkeypatch.setattr(v02, "s1_post", lambda c, body, lane: ok("move", cache_n=next(cache_ns)))
    out = v02.row_cache_drift(FakeC(), ctx)
    assert erased == [2] and out["pass"] is None                   # aux is slot 1, so slots 0..1 are erased
    if invalid_sub is None:
        assert out["invalid"] is None and out["cold_was_cold"] is True and out["max_dp"] == 0.0
    else:
        assert invalid_sub in out["invalid"] and out["cold_was_cold"] is False and out["max_dp"] is None


# ---- walk pass reuse: cache_n == A.tokens_evaluated, prompt_n == 1, prompt_ms <= 40 ----------------------

@pytest.mark.parametrize(("cache_n", "prompt_n", "prompt_ms", "expect"), [
    (500, 1, 40.0, True), (500, 1, 40.5, False), (499, 2, 20.0, False), (500, 4, 20.0, False)])
def test_walk_reuse_gate(ctx, monkeypatch, cache_n, prompt_n, prompt_ms, expect):
    a = MMResult({1: -0.1}, 500, 0, 500, 300.0)
    eng = FakeEng([a, MMResult({1: -0.1}, 501, cache_n, prompt_n, prompt_ms), MMResult({1: -0.1}, 500, 496, 4, 30.0)])
    monkeypatch.setattr(ctx, "engine", lambda c: eng)
    monkeypatch.setattr(v02, "_round_trip_token", lambda eng, prompt, lp: (1, "B"))
    out = v02.row_walk_reuse(FakeC(), ctx)
    assert out["invalid"] is None and out["slot"] == 1 and out["pass"] is expect


def test_walk_reuse_is_invalid_when_the_aux_lane_is_unbound(ctx, monkeypatch):
    eng = FakeEng([MMResult({1: -0.1}, 500, 0, 500, 300.0), MMResult({}, 501, 500, 1, 10.0),
                   MMResult({}, 500, 496, 4, 30.0)])
    monkeypatch.setattr(ctx, "engine", lambda c: eng)
    monkeypatch.setattr(v02, "_round_trip_token", lambda eng, prompt, lp: (1, "B"))
    out = v02.row_walk_reuse(FakeC(lanes={"s1": 0}), ctx)
    assert "lane aux is not bound" in out["invalid"] and out["pass"] is False


def test_a_failed_lane_binding_invalidates_every_row_but_the_selftests(ctx, monkeypatch):
    monkeypatch.setattr(v02, "bind_lanes", lambda c, lanes: [ok("move", slot="0"),
                                                             {**ok("move"), "ok": False, "status": 503, "slot": None}])
    for name in ("selftests", "readout"):
        monkeypatch.setitem(v02._MAIN, name, lambda c, ctx: {"pass": True, "invalid": None})
    got = {}
    v02.run_main_rows(FakeC(), ctx, ["selftests", "readout"], got.__setitem__)
    assert got["selftests"]["invalid"] is None
    assert got["readout"]["invalid"] == "lane binding failed (aux: HTTP 503 on slot None)"


# ---- system-prefix and context-prefix reuse: cache_n >= the system turn's tokens -------------------------

@pytest.mark.parametrize(("cache_n", "expect"), [(300, True), (299, False)])
def test_prefix_reuse_gate(ctx, monkeypatch, cache_n, expect):
    sent = iter([0, cache_n, cache_n])
    monkeypatch.setattr(v02, "s1_post", lambda c, body, lane: ok("move", cache_n=next(sent)))
    monkeypatch.setattr(v02, "system_turn_tokens", lambda c, ctx, body: 300)
    out = v02.row_prefix_reuse(FakeC(), ctx)
    assert out["invalid"] is None and out["pass"] is expect


@pytest.mark.parametrize(("cache_n", "expect"), [(300, True), (299, False)])
def test_context_reuse_gate(ctx, monkeypatch, cache_n, expect):
    sent = iter([0, cache_n, 900])
    monkeypatch.setattr(v02, "s1_post", lambda c, body, lane: ok("q", cache_n=next(sent)))
    monkeypatch.setattr(v02, "system_turn_tokens", lambda c, ctx, body: 300)
    monkeypatch.setattr(lib, "post", lambda url, payload, **kw: (200, {}, {"tokens": [0] * 600}))
    out = v02.row_context_reuse(FakeC(), ctx)
    assert out["invalid"] is None and out["context_tokens"] == 600 and out["pass"] is expect


# ---- text-path regression and self-tests ----------------------------------------------------------------

@pytest.mark.parametrize(("drop_in", "selftest", "expect"), [(True, True, True), (True, False, False)])
def test_text_regression_gate(ctx, monkeypatch, tmp_path, drop_in, selftest, expect):
    class Inst:
        build_info = "fake"

        def finish(self):
            pass

    monkeypatch.setattr(v02, "_instance", lambda name, out, *, model, flags, ctx=None: Inst())
    monkeypatch.setattr(v02, "Ctx", lambda *a, **k: object())
    monkeypatch.setattr(v02, "row_drop_in", lambda c, v01, name: {"pass": drop_in})
    monkeypatch.setattr(v02, "row_selftest", lambda c, v01, name: {"pass": selftest})
    assert v02.row_text_regression(ctx, tmp_path)["pass"] is expect


@pytest.mark.parametrize(("lines", "expect"), [
    (["self-test ok: 2 + 2 = 4", "answer slot check: answer slot confirmed", "vision self-test ok: right"], True),
    (["self-test ok: 2 + 2 = 4", "answer slot check: answer slot confirmed", "vision self-test FAILED"], False)])
def test_selftests_gate(ctx, lines, expect):
    assert v02.row_selftests(FakeC(lines=lines), ctx)["pass"] is expect


# ---- report: the footprint verdict follows its value ----------------------------------------------------

@pytest.mark.parametrize(("footprint", "cell"), [(None, "| n/a | n/a |"), (3 * 1024**3, "| 3.0 GiB | measured |")])
def test_footprint_verdict_is_na_without_a_value(footprint, cell):
    md = render({"rows": {"hz2_stream": {"footprint_max": footprint}}})
    assert next(ln for ln in md.splitlines() if ln.startswith("| footprint |")).endswith(cell)
