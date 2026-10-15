"""bench/check_cache_delimiters.py: its pure half, the measurement loops and the run's verdicts, on fakes (no server,
no model)."""
import argparse
import json
import math

import pytest

from bench import check_cache_delimiters as cc

LOG_LINE = ("0.13.398.391 I slot print_timing: id  {slot} | task {task} | prompt eval time =     {ms:.2f} ms /   {n} "
            "tokens (    0.88 ms per token,  1132.67 tokens per second)\n")
BODY_A = {"state": {"maze": "A"}, "questions": {"pacman": {"type": "choice", "criteria": {"up": None, "left": None}}}}
BODY_B = {"state": {"maze": "B"}, "questions": {"pacman": {"type": "choice", "criteria": {"up": None, "left": None}},
                                                "blinky": {"type": "choice", "criteria": {"up": None, "down": None}}}}
DELIM = "<|im_end|>\n<|im_start|>user"


def _ans(choice, probs):
    return {"type": "choice", "choice": choice, "probabilities": probs}


# ---- bodies ----------------------------------------------------------------------------------------------------------

def test_load_bodies_reads_jsonl_a_json_body_and_a_json_list_in_order(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps(BODY_A) + "\n\n" + json.dumps(BODY_B) + "\n")
    (tmp_path / "b.json").write_text(json.dumps(BODY_B))
    (tmp_path / "c.json").write_text(json.dumps([BODY_A, BODY_B]))
    got = cc.load_bodies([tmp_path / "a.jsonl", tmp_path / "b.json", tmp_path / "c.json"])
    assert got == [BODY_A, BODY_B, BODY_B, BODY_A, BODY_B]


def test_load_bodies_refuses_a_game_client_recording_with_the_reason(tmp_path):
    (tmp_path / "game-01.json").write_text(json.dumps({"version": 1, "frames": [0.0175], "decisions": [], "events": []}))
    with pytest.raises(ValueError, match="game-client agent recording, which holds no request bodies"):
        cc.load_bodies([tmp_path / "game-01.json"])


@pytest.mark.parametrize("line", ['{"state": "x"}', '{"state": "x", "questions": {}}', "[1]", "not json",
                                  '{"state": "x", "context": "c", "questions": {"q": {}}}',
                                  '{"state": "x", "images": ["data:"], "questions": {"q": {}}}'])
def test_load_bodies_names_the_file_and_line_of_a_bad_body(tmp_path, line):
    (tmp_path / "x.jsonl").write_text(json.dumps(BODY_A) + "\n" + line + "\n")
    with pytest.raises(ValueError, match=r"x\.jsonl:2"):
        cc.load_bodies([tmp_path / "x.jsonl"])


def test_primer_is_the_next_body_cyclically_and_needs_two():
    assert [cc.primer_index(i, 3) for i in range(3)] == [1, 2, 0]
    with pytest.raises(ValueError):
        cc.primer_index(0, 1)


# ---- comparison ------------------------------------------------------------------------------------------------------

def test_compare_answers_reports_max_abs_dlogprob_and_argmax_agreement():
    cold = {"q": _ans("a", {"a": 0.7, "b": 0.3}), "r": _ans("x", {"x": 0.55, "y": 0.45})}
    warm = {"q": _ans("a", {"a": 0.69, "b": 0.31}), "r": _ans("y", {"x": 0.4, "y": 0.6})}
    out = cc.compare_answers(cold, warm)
    assert out["q"]["dlogprob"]["a"] == pytest.approx(abs(math.log(0.69) - math.log(0.7)))
    assert out["q"]["max_dlogprob"] == pytest.approx(abs(math.log(0.31) - math.log(0.3)))
    assert out["q"]["argmax_agree"] is True and out["r"]["argmax_agree"] is False
    assert (out["r"]["argmax_ref"], out["r"]["argmax_other"]) == ("x", "y")


def test_compare_answers_zeros_and_mismatches():
    out = cc.compare_answers({"q": _ans("a", {"a": 1.0, "b": 0.0}), "r": _ans("a", {"a": 1.0, "b": 0.0}),
                              "s": _ans("a", {"a": 1.0}), "t": _ans("a", {"a": 1.0})},
                             {"q": _ans("a", {"a": 1.0, "b": 0.0}), "r": _ans("a", {"a": 0.9, "b": 0.1}),
                              "t": _ans("a", {"b": 1.0})})
    assert out["q"]["max_dlogprob"] == 0.0                     # both floored: no difference
    assert out["r"]["dlogprob"]["b"] == math.inf
    assert out["s"] == {"error": "cannot compare: no answer"}  # missing from the other side
    assert out["t"] == {"error": "the two answers do not have the same options"}


def test_noul_answers_are_read_as_two_way_and_score_answers_get_no_argmax():
    noul = cc.compare_answers({"q": {"type": "noul", "noul": 0.8}}, {"q": {"type": "noul", "noul": 0.4}})["q"]
    assert noul["dlogprob"]["1"] == pytest.approx(abs(math.log(0.4) - math.log(0.8)))
    assert noul["argmax_agree"] is False and (noul["argmax_ref"], noul["argmax_other"]) == ("1", "0")
    score = {"type": "score", "score": 1.2, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}
    s = cc.compare_answers({"q": score}, {"q": {**score, "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4}}})["q"]
    assert s["argmax_agree"] is None and s["argmax_ref"] is None and s["max_dlogprob"] > 0
    bad = cc.compare_answers({"q": {"type": "noul"}, "r": {"type": "choice", "choice": "a"}},
                             {"q": {"type": "noul", "noul": 0.4}, "r": {"type": "choice", "choice": "a"}})
    assert bad["q"]["error"] == "cannot compare: a noul answer without a numeric `noul`"
    assert bad["r"]["error"] == "cannot compare: a choice answer without `probabilities`"


# ---- timings, the llama-server log, attribution ----------------------------------------------------------------------

def test_parse_timings_drops_truncation_and_garbage():
    h = json.dumps([{"q": "a", "passes": 1}, {"truncated": 2}])
    assert cc.parse_timings(h) == [{"q": "a", "passes": 1}]
    assert cc.parse_timings(None) == [] and cc.parse_timings("nope") == [] and cc.parse_timings('{"q": 1}') == []


def test_parse_prompt_evals_reads_real_llama_server_lines_and_filters_the_slot():
    text = (LOG_LINE.format(slot=0, task=9, ms=235.73, n=267) + "0.1 I srv  log_server_r: done\n"
            + LOG_LINE.format(slot=3, task=10, ms=96.87, n=16) + LOG_LINE.format(slot=0, task=11, ms=12.5, n=3))
    assert cc.parse_prompt_evals(text, 0) == [{"slot": 0, "task": 9, "ms": 235.73, "prompt_n": 267},
                                              {"slot": 0, "task": 11, "ms": 12.5, "prompt_n": 3}]
    assert cc.parse_prompt_evals(text, 3) == [{"slot": 3, "task": 10, "ms": 96.87, "prompt_n": 16}]
    assert cc.parse_prompt_evals(text, 1) == []                        # never another slot's lines


def test_llama_log_path_comes_from_callos_launch_line():
    log = ("upstream ready\nlaunched llama-server pid 4246 on http://127.0.0.1:59102 "
           "(log /Users/x/.cache/callo/llama-server-59102.log)\nself-test ok")
    assert cc.llama_log_path(log) == "/Users/x/.cache/callo/llama-server-59102.log"
    assert cc.llama_log_path("nothing") is None


def test_first_pass_counts_maps_each_questions_first_pass_and_prefers_callos_own_numbers():
    timings = [{"q": "a", "kind": "plain", "prompt_n": None, "cache_n": None, "tokens": 1115, "passes": 2, "ms": 700},
               {"q": "b", "kind": "plain", "prompt_n": None, "cache_n": None, "tokens": 1120, "passes": 1, "ms": 90},
               {"q": "c", "kind": "context", "prompt_n": 58, "cache_n": 354, "prompt_ms": 15.5, "tokens": 412,
                "passes": 1, "ms": 30}]
    evals = [{"prompt_n": 899, "ms": 600.0}, {"prompt_n": 1, "ms": 9.0}, {"prompt_n": 40, "ms": 30.0},
             {"prompt_n": 999, "ms": 1.0}]
    out = cc.first_pass_counts(timings, evals)
    assert out["a"] == {"prompt_n": 899, "cache_n": 216, "log_ms": 600.0, "tokens": 1115, "passes": 2,
                        "source": "log", "ms": 700}
    assert (out["b"]["prompt_n"], out["b"]["cache_n"], out["b"]["log_ms"]) == (40, 1080, 30.0)   # after a's two
    assert (out["c"]["prompt_n"], out["c"]["cache_n"], out["c"]["log_ms"], out["c"]["source"]) == (58, 354, 15.5,
                                                                                                    "timings")
    short = cc.first_pass_counts(timings[:2], evals[:1])
    assert short["b"] == {"prompt_n": None, "cache_n": None, "log_ms": None, "tokens": 1120, "passes": 1,
                          "source": None, "ms": 90}


def test_a_leading_single_option_question_takes_no_log_line_and_is_not_attributed():
    """A single-option question is answered without the upstream (passes 0, tokens 0): it must not borrow the next
    question's prompt-eval line (a seen case: prompt_n 84, cache_n -84), and attribution skips it."""
    timings = [{"q": "only", "kind": "plain", "prompt_n": None, "cache_n": None, "tokens": 0, "passes": 0, "ms": 0.1},
               {"q": "a", "kind": "plain", "prompt_n": None, "cache_n": None, "tokens": 300, "passes": 1, "ms": 80.0},
               {"q": "b", "kind": "plain", "prompt_n": None, "cache_n": None, "tokens": 310, "passes": 1, "ms": 20.0}]
    evals = [{"prompt_n": 84, "ms": 50.0}, {"prompt_n": 12, "ms": 9.0}]
    out = cc.first_pass_counts(timings, evals)
    assert out["only"] == {"prompt_n": None, "cache_n": None, "log_ms": None, "tokens": 0, "passes": 0, "source": None,
                           "ms": 0.1}
    assert (out["a"]["prompt_n"], out["a"]["cache_n"], out["a"]["source"]) == (84, 216, "log")
    assert (out["b"]["prompt_n"], out["b"]["cache_n"]) == (12, 298)
    assert cc.first_pass_counts([{**timings[0], "passes": None}], evals)["only"]["source"] is None
    row = {"user_start": {"only": 5, "a": 216}, "warm": {"counts": out}}
    assert cc.attribution(row) == {"q": "a", "cache_n": 216, "user_start": 216, "attributed": True}
    assert cc.first_question(out) == "a" and cc.first_question({"only": out["only"]}) is None
    assert cc.attribution({"warm": {"counts": {"only": out["only"]}}})["attributed"] is None


def test_token_index_and_user_starts_find_the_delimiters_first_token():
    assert cc.token_index([1, 2, 3, 2, 3], [2, 3]) == 1 and cc.token_index([1, 2], [3]) is None
    assert cc.token_index([1], []) is None
    rendered = []

    def render(msgs):
        rendered.append(msgs)
        return "SYSTEM HEAD" + DELIM + "\n" + msgs[-1]["content"]

    def tokenize(text, *, add_special=True, parse_special=None):
        return ([1] if add_special else []) + [ord(ch) for ch in text]

    starts = cc.user_starts(BODY_B, DELIM, render=render, tokenize=tokenize)
    assert starts == {"pacman": 1 + len("SYSTEM HEAD"), "blinky": 1 + len("SYSTEM HEAD")}
    assert len(rendered) == 2 and rendered[0][0]["role"] == "system"      # callo's own messages, one per question


def test_attribution_compares_the_warm_first_pass_with_the_user_turn_start():
    row = {"user_start": {"pacman": 216, "blinky": 216},
           "warm": {"counts": {"pacman": {"cache_n": 214, "passes": 1}, "blinky": {"cache_n": 1112, "passes": 1}}}}
    assert cc.attribution(row) == {"q": "pacman", "cache_n": 214, "user_start": 216, "attributed": True}
    assert cc.attribution(row, tol=1)["attributed"] is False
    assert cc.attribution({**row, "user_start": {}})["attributed"] is None
    assert cc.attribution({"warm": {"counts": {}}})["attributed"] is None


def test_port_stats_and_run_problems():
    assert cc.port_problems(["http://127.0.0.1:51234", "http://127.0.0.1:59102"]) == []
    assert len(cc.port_problems(["http://127.0.0.1:8190", "http://127.0.0.1:8080", "http://h", None])) == 4
    good = {"user_delimiter": DELIM, "cache_prompt": {"plain": True}, "provenance": {"cache_delimiter": True}}
    assert cc.stats_problems(good, cached=True) == []
    assert len(cc.stats_problems({"user_delimiter": None, "cache_prompt": {"plain": False}}, cached=True)) == 3
    assert cc.stats_problems({**good, "provenance": {}}, cached=True) == [
        "provenance.cache_delimiter is not set: this callo's plain readouts do not send the delimiter"]
    assert cc.stats_problems({"cache_prompt": {"plain": False}}, cached=False) == []
    assert cc.stats_problems(good, cached=False) == ["the OFF server has cache_prompt.plain True, not false"]
    log_row = {"body": 0, "same_slot": True, "warm": {"counts": {"q": {"source": "log"}}}}
    assert cc.run_problems([log_row]) == [] and cc.run_problems([]) == []
    moved = {**log_row, "body": 3, "same_slot": False}
    no_log = {**log_row, "warm": {"counts": {"q": {"source": None}}}}
    assert cc.run_problems([log_row, moved]) == ["bodies [3]: the cold, primer and warm readouts did not all land on "
                                                 "one slot"]
    assert cc.run_problems([no_log])[0].startswith("no warm first-pass count came from the llama-server log")


def test_wait_gone_uses_the_zombie_aware_check_of_the_acceptance_bench():
    from bench import acceptance
    assert cc.wait_gone.__kwdefaults__["alive"] is acceptance._alive


def test_wait_gone_polls_until_the_pid_is_gone_and_never_trusts_an_unknown_pid():
    seen = iter([True, True, False])
    assert cc.wait_gone(42, alive=lambda pid: next(seen), sleep=lambda s: None) is True
    assert cc.wait_gone(42, timeout=0, alive=lambda pid: True, sleep=lambda s: None) is False
    assert cc.wait_gone(None, alive=lambda pid: False) is False


# ---- the measurement loops, on a fake callo + upstream ---------------------------------------------------------------

class FakeServer:
    """`post` for callo's /v1/systemone and the upstream's slot erase, with a llama-server log that grows by one
    prompt-eval line per pass: a cold slot prefills the whole 1115-token prompt, a warm one (after another body) 899,
    so cache_n is 216, the user-turn start `_starts` reports. `slots` scripts X-Callo-Slot per readout."""

    def __init__(self, slots=None):
        self.log, self.calls, self.warm, self.task, self.reads = "", [], False, 0, 0
        self.slots = slots

    def read_log(self):
        return self.log

    def post(self, url, payload, *, timeout=None, headers=None):
        self.calls.append(url)
        if "/slots/" in url:
            self.warm = False
            return 200, {}, {"id_slot": 0, "n_erased": 0}
        slot = self.slots[self.reads] if self.slots else 0
        self.reads += 1
        entries, answers = [], {}
        for q, spec in payload["questions"].items():
            n = 1115 if not self.warm and not entries else 899 if not entries else 3
            self.task += 1
            self.log += LOG_LINE.format(slot=slot, task=self.task, ms=n * 0.6, n=n)
            entries.append({"q": q, "kind": "plain", "cache_n": None, "prompt_n": None, "tokens": 1115, "passes": 1,
                            "ms": 1.0})
            opts = list(spec["criteria"])
            p = 0.7 if not self.warm else 0.69
            answers[q] = _ans(opts[0], {opts[0]: p, opts[1]: 1 - p})
        self.warm = True
        return 200, {"x-callo-slot": str(slot), "x-callo-timings": json.dumps(entries)}, {"answers": answers}


def _starts(body):
    return {q: 216 for q in body["questions"]}


def test_measure_cached_then_off_reads_cold_primes_warm_and_the_reference():
    s = FakeServer()
    seen = []
    rows = cc.measure_cached("http://callo", "http://up", 2, [BODY_A, BODY_B], post=s.post, read_log=s.read_log,
                             starts=_starts, on_row=seen.append)
    assert seen == rows and [(r["body"], r["primer"]) for r in rows] == [(0, 1), (1, 0)]
    kinds = ["erase" if "/slots/" in u else "read" for u in s.calls]
    assert kinds == ["erase", "erase", "read", "erase", "erase", "read", "read"] * 2
    r0 = rows[0]
    assert r0["cold"]["counts"]["pacman"]["prompt_n"] == 1115 and r0["cold"]["counts"]["pacman"]["cache_n"] == 0
    assert r0["warm"]["counts"]["pacman"]["prompt_n"] == 899 and r0["warm"]["counts"]["pacman"]["cache_n"] == 216
    assert r0["warm"]["counts"]["pacman"]["log_ms"] == pytest.approx(899 * 0.6)
    assert r0["attribution"]["attributed"] is True and r0["same_slot"] and r0["compare"]["pacman"]["argmax_agree"]
    assert "user-turn start 216" in cc.progress_line(r0)
    cc.measure_off("http://callo", "http://up", 2, [BODY_A, BODY_B], rows, post=s.post, read_log=s.read_log)
    assert r0["off"]["counts"]["pacman"]["prompt_n"] == 1115 and r0["cold_vs_off"]["pacman"]["max_dlogprob"] == 0.0
    summ = cc.summarize(rows)
    want = abs(math.log(0.31) - math.log(0.3))
    assert summ["warm_vs_cold"]["questions"] == 3 and summ["warm_vs_cold"]["max_dlogprob"] == pytest.approx(want)
    assert summ["warm_vs_off"]["max_dlogprob"] == pytest.approx(want) and summ["cold_vs_off"]["max_dlogprob"] == 0.0
    assert summ["warm_vs_cold"]["argmax_agree"] == summ["warm_vs_cold"]["argmax_judged"] == 3
    assert summ["arms"]["cold"]["cache_n_mean"] == pytest.approx((0 + 0 + 1112) / 3)
    assert summ["arms"]["warm"]["first_pass_log_ms_mean"] == pytest.approx((899 + 899 + 3) * 0.6 / 3)
    assert summ["arms"]["off"]["wall_ms_mean"] is not None
    # first question only: no question 2+ restoring question 1's checkpoint in the cold mean
    assert summ["arms"]["cold"]["first_question"]["prompt_n_mean"] == 1115
    assert summ["arms"]["cold"]["first_question"]["cache_n_mean"] == 0
    assert summ["arms"]["warm"]["first_question"]["prompt_n_mean"] == 899
    assert summ["arms"]["warm"]["first_question"]["log_ms_mean"] == pytest.approx(899 * 0.6)
    assert summ["arms"]["warm"]["first_question"]["ms_mean"] == 1.0
    assert (summ["attributed"], summ["attribution_judged"]) == (2, 2)


def test_measure_stops_on_a_failed_readout_or_erase():
    s = FakeServer()

    def bad(url, payload, **kw):
        return (500, {}, {"error": "boom"}) if url.endswith("/v1/systemone") else s.post(url, payload, **kw)

    with pytest.raises(cc.RunError, match="HTTP 500"):
        cc.measure_cached("http://callo", "http://up", 1, [BODY_A, BODY_B], post=bad, read_log=s.read_log)
    with pytest.raises(cc.RunError, match="erasing slot 0"):
        cc.measure_cached("http://callo", "http://up", 1, [BODY_A, BODY_B], post=lambda *a, **k: (501, {}, "no"),
                          read_log=s.read_log)


@pytest.mark.parametrize("slot", [None, "", "-", "x1"])
def test_a_readout_without_an_integer_slot_is_an_error(slot):
    headers = {"x-callo-timings": json.dumps([{"q": "pacman", "tokens": 9, "passes": 1, "prompt_n": None}])}
    if slot is not None:
        headers["x-callo-slot"] = slot

    def post(url, payload, **kw):
        return 200, headers, {"answers": {}}

    with pytest.raises(cc.RunError, match="without an integer X-Callo-Slot"):
        cc.readout("http://callo", BODY_A, post=post, read_log=lambda: LOG_LINE.format(slot=0, task=1, ms=1.0, n=9),
                   log_wait=0)


def test_readout_waits_for_the_log_lines_of_every_pass():
    lines = iter(["", "", LOG_LINE.format(slot=0, task=1, ms=1.0, n=1115)])
    state = {"log": ""}

    def post(url, payload, **kw):
        return 200, {"x-callo-slot": "0", "x-callo-timings": json.dumps([{"q": "pacman", "tokens": 1115, "passes": 1,
                                                                            "prompt_n": None}])}, {"answers": {}}

    def sleep(_):
        state["log"] += next(lines)

    r = cc.readout("http://callo", BODY_A, post=post, read_log=lambda: state["log"], log_wait=60, sleep=sleep)
    assert r["counts"]["pacman"]["prompt_n"] == 1115


# ---- the run's verdicts, with a fake Callo -----------------------------------------------------------------------------

class FakeCallo:
    def __init__(self, name, cache, events, *, port=51001, log=True, plain=None):
        self.name, self.events = name, events
        self.url, self.upstream = f"http://127.0.0.1:{port}", f"http://127.0.0.1:{port + 1}"
        self.pid, self.build_info, self.stats_calls = 4000 + port, "b1-x", 0
        self._log = f"launched llama-server pid {self.pid} on {self.upstream} (log /tmp/ll-{name}.log)" if log else ""
        plain = cache if plain is None else plain
        self._stats = {"user_delimiter": DELIM, "cache_prompt": {"plain": plain},
                       "provenance": {"cache_prompt": plain, **({"cache_delimiter": True} if plain else {})}}
        events.append(("launch", name))

    def stats(self):
        self.stats_calls += 1
        return self._stats

    def log_text(self):
        return self._log

    def finish(self):
        self.events.append(("finish", self.name))
        return 0


def _args(tmp_path):
    return argparse.Namespace(out=tmp_path, model="/m.gguf", llama_server="llama-server", ctx=8192, parallel=2,
                              bodies=["b.jsonl"], attr_tol=4)


@pytest.fixture
def live(monkeypatch, tmp_path):
    """The run's seams: launches make FakeCallo objects (`launch_kw[name]` overrides), every request goes to one
    FakeServer, the llama-server log is that server's, the stop is confirmed unless `gone` is False."""
    env = argparse.Namespace(events=[], server=FakeServer(), launch_kw={}, gone=True, callos=[])

    def launch(a, name, *, cache):
        c = FakeCallo(name, cache, env.events, **env.launch_kw.get(name, {}))
        env.callos.append(c)
        return c

    monkeypatch.setattr(cc, "_launch", launch)
    monkeypatch.setattr(cc, "code_state", lambda: {"head": "abc"})
    monkeypatch.setattr(cc, "_starts_via", lambda upstream, delimiter: _starts)
    monkeypatch.setattr(cc, "_POST", lambda *a, **k: env.server.post(*a, **k))
    monkeypatch.setattr(cc, "_read_file", lambda path: lambda: env.server.read_log())
    monkeypatch.setattr(cc, "_wait_gone", lambda pid: env.events.append(("gone?", pid)) or env.gone)
    env.summary = lambda: json.loads((tmp_path / "summary.json").read_text())
    return env


def test_run_done_measures_both_launches_one_at_a_time(live, tmp_path):
    assert cc._run(_args(tmp_path), [BODY_A, BODY_B]) == 0
    assert live.events == [("launch", "cached"), ("finish", "cached"), ("gone?", 55001), ("launch", "off"),
                           ("finish", "off")]
    s = live.summary()
    assert s["verdict"] == "DONE" and s["summary"]["attributed"] == 2 and s["summary"]["cold_vs_off"]["questions"] == 3
    assert s["launches"]["cached"]["llama_log"] == "/tmp/ll-cached.log" and s["launches"]["off"]["stats"]
    rows = [json.loads(x) for x in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all("off" in r for r in rows)


@pytest.mark.parametrize(("kw", "msg"), [
    ({"cached": {"log": False}}, "names no llama-server log"),
    ({"cached": {"plain": False}}, "cache_prompt.plain is not true"),
    ({"off": {"plain": True}}, "the OFF server has cache_prompt.plain True"),
    ({"cached": {"port": 8189}}, "port 8190 is forbidden"),
])
def test_run_errors_on_a_launch_that_would_measure_the_wrong_thing(live, tmp_path, kw, msg):
    live.launch_kw = kw
    assert cc._run(_args(tmp_path), [BODY_A, BODY_B]) == 1
    s = live.summary()
    assert s["verdict"] == "ERROR" and any(msg in e for e in s["errors"])
    assert live.events[-1][0] == "finish"                               # the server is stopped on every path
    if "port" in kw.get("cached", {}):
        assert live.callos[0].stats_calls == 0 and live.server.calls == []   # nothing sent after the port check


def test_run_errors_when_readouts_move_slots_and_never_starts_the_off_launch(live, tmp_path):
    live.server = FakeServer(slots=[0, 0, 1, 0, 0, 0])
    assert cc._run(_args(tmp_path), [BODY_A, BODY_B]) == 1
    s = live.summary()
    assert s["verdict"] == "ERROR" and s["errors"] == ["bodies [0]: the cold, primer and warm readouts did not all "
                                                       "land on one slot"]
    assert ("launch", "off") not in live.events and s["summary"]["bodies"] == 2


def test_run_errors_when_the_first_llama_server_is_not_gone(live, tmp_path):
    live.gone = False
    assert cc._run(_args(tmp_path), [BODY_A, BODY_B]) == 1
    assert "not confirmed gone" in live.summary()["errors"][0] and ("launch", "off") not in live.events


def test_an_unexpected_exception_is_recorded_before_the_server_stops(live, tmp_path, monkeypatch):
    def boom(upstream, delimiter):
        def starts(body):
            raise KeyError("surprise")
        return starts

    monkeypatch.setattr(cc, "_starts_via", boom)
    real_finish = FakeCallo.finish

    def finish(self):                                   # summary.json already holds the ERROR when the server stops
        self.events.append(("summary at finish", json.loads((tmp_path / "summary.json").read_text())["verdict"]))
        return real_finish(self)

    monkeypatch.setattr(FakeCallo, "finish", finish)
    assert cc._run(_args(tmp_path), [BODY_A, BODY_B]) == 1
    s = live.summary()
    assert s["verdict"] == "ERROR" and s["errors"] == ["unexpected KeyError: 'surprise'"] and "KeyError" in s["traceback"]
    assert live.events[-2:] == [("summary at finish", "ERROR"), ("finish", "cached")]


def test_a_stop_signal_records_stopped_and_still_stops_the_server(live, tmp_path, monkeypatch):
    def stop(*a, **k):
        raise SystemExit(143)

    monkeypatch.setattr(cc, "_POST", stop)
    with pytest.raises(SystemExit):
        cc._run(_args(tmp_path), [BODY_A, BODY_B])
    assert live.summary()["verdict"] == "STOPPED" and live.events[-1] == ("finish", "cached")


def test_main_refuses_too_few_bodies_and_a_recording_before_touching_anything(tmp_path, capsys):
    (tmp_path / "one.jsonl").write_text(json.dumps(BODY_A) + "\n")
    (tmp_path / "game-01.json").write_text(json.dumps({"frames": [], "decisions": []}))
    base = ["--model", "/m.gguf", "--out", str(tmp_path / "out")]
    assert cc.main([*base, "--bodies", str(tmp_path / "one.jsonl")]) == 2
    assert cc.main([*base, "--bodies", str(tmp_path / "game-01.json")]) == 2
    assert "game-client agent recording" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_the_two_launches_say_their_cache_explicitly(monkeypatch, tmp_path):
    """--cache is on by default since 0.3.0: the OFF arm must ask for --no-cache, or its own guard errors."""
    seen = []
    monkeypatch.setattr(cc, "Callo", lambda *a, **k: seen.append(k) or "c")
    a = argparse.Namespace(out=tmp_path, llama_server="ls", ctx=8192, parallel=4, model="/m.gguf")
    assert cc._launch(a, "cached", cache=True) == "c" and cc._launch(a, "off", cache=False) == "c"
    assert [k["cache"] for k in seen] == [True, False]
    assert all("--cache" not in k["flags"] and "--no-cache" not in k["flags"] for k in seen)
