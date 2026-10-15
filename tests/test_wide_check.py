"""bench/wide_check.py: its pure half, the serve / in-process / re-send loops and
the run's verdicts, on fakes (no server, no model, no GPU)."""
import argparse
import json
import math
from pathlib import Path

import pytest
from test_wide_engine import B_, GA, ID, SETS, THE, FakeTemplate, WideScorer, lp

from bench import wide_check as w
from callo.engine import WIDE_TOP_K, Engine
from callo.marginalize import _mass_tol
from callo.systemone import WIDE_LABELS, is_wide, parse_request

CP_CREATED = ("0.31.207.716 I slot create_check: id  {slot} | task {task} | created context checkpoint 3 of 32 "
              "(pos_min = 0, pos_max = {n1}, n_tokens = {n}, size = 75.376 MiB)\n")
CP_RESTORED = ("0.31.300.100 I slot update_slots: id  {slot} | task {task} | restored context checkpoint (pos_min = 0, "
               "pos_max = {n1}, n_tokens = {n}, n_past = {n}, size = 75.376 MiB)\n")
EVAL = ("0.13.398.391 I slot print_timing: id  {slot} | task {task} | prompt eval time =     {ms:.2f} ms /   {n} "
        "tokens (    0.88 ms per token,  1132.67 tokens per second)\n")


def _item(n=30, gold=1, source="banking77", idx=0):
    labels = [f"k{i}" for i in range(n)]
    control = [labels[gold], *[x for x in labels if x != labels[gold]][:9]]
    return {"id": f"{source}/train/{idx}", "source": source, "split": "train", "revision": "r", "license": "CC-BY-4.0",
            "text": f"utterance {idx}", "gold": labels[gold], "labels": labels, "control": control}


# ---- items and bodies -------------------------------------------------------------------------------------------

def test_load_items_selects_each_models_items():
    meta, items = w.load_items(model="tiel")
    assert set(meta) == {"_meta"} and len(items) == 40
    assert [it["source"] for it in items] == ["banking77"] * 20 + ["clinc150"] * 20
    _, gemma = w.load_items(model="gemma")
    assert gemma == items[:10] and {len(it["labels"]) for it in gemma} == {77}


def test_load_items_refuses_a_malformed_file(tmp_path):
    rows = [json.loads(line) for line in w.ITEMS.read_text().splitlines()]
    rows[3]["control"] = rows[3]["control"][:9]
    p = tmp_path / "bad.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="malformed"):
        w.load_items(p)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows[1:]))
    with pytest.raises(ValueError, match="_meta"):
        w.load_items(p)


def test_the_wide_body_reads_every_intent_and_the_control_ten_at_budget_0():
    _, items = w.load_items()
    clinc = items[-1]
    body = w.wide_body(clinc)
    assert body["think_budget"] == 0 and body["state"] == clinc["text"]
    q = parse_request(body).questions[0]
    assert is_wide(q) and q.labels == WIDE_LABELS[:151] and list(q.keys) == clinc["labels"]
    assert body["questions"]["intent"]["instructions"] == w.INSTRUCTIONS["clinc150"]
    c = parse_request(w.control_body(clinc)).questions[0]
    assert not is_wide(c) and list(c.keys) == clinc["control"] and c.labels == tuple("ABCDEFGHIJ")
    assert w.wide_body(items[0])["questions"]["intent"]["instructions"] == w.INSTRUCTIONS["banking77"]


def test_parse_checkpoints_reads_created_and_restored_lines_of_one_slot():
    log = (CP_CREATED.format(slot=0, task=5, n1=410, n=411) + CP_RESTORED.format(slot=0, task=6, n1=410, n=411)
           + CP_CREATED.format(slot=1, task=7, n1=99, n=100)
           + "0.1 W slot create_check: id  0 | task 8 | erasing old context checkpoint (pos_min = 0, pos_max = 5, "
             "n_tokens = 6, size = 1.0 MiB)\n"
           + "0.1 I slot create_check: id  0 | task 8 | superseding context checkpoint at n_tokens = 411\n"
           + CP_CREATED.format(slot=0, task=9, n1=411, n=412))
    assert w.parse_checkpoints(log, 0) == {"created": [411, 412], "restored": [411]}
    assert w.parse_checkpoints(log, 1) == {"created": [100], "restored": []}
    assert w.parse_checkpoints(log, None)["created"] == [411, 100, 412]


def test_tv_and_the_mass_tolerance():
    assert w.tv([0.5, 0.5], [0.5, 0.5]) == 0.0 and w.tv([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert w.tv([0.6, 0.4], [0.5, 0.5]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        w.tv([1.0], [0.5, 0.5])
    assert w.mass_tolerance(3) == _mass_tol(3 * WIDE_TOP_K) and w.mass_tolerance(0) == _mass_tol(WIDE_TOP_K)


# ---- the serve readout -------------------------------------------------------------------------------------------

class FakeLog:
    def __init__(self):
        self.text = ""

    def size(self):
        return len(self.text.encode())

    def since(self, offset):
        return self.text.encode()[offset:].decode()


class FakeServe:
    """callo's /v1/systemone and the upstream's /completion and /tokenize. A wide readout makes `split` split passes
    (plus one), each logging a prompt-eval line and a created checkpoint on `slot`; the answer puts `p_gold` on the
    gold (`gold_of(body)`)."""

    def __init__(self, log, *, slot=0, split=0, unread=0.0, p_gold=0.9, wide_status=200, gold_of=None, drop=()):
        self.log, self.slot, self.split, self.unread, self.p_gold = log, slot, split, unread, p_gold
        self.wide_status, self.gold_of, self.drop, self.calls, self.task = wide_status, gold_of, set(drop), [], 0

    def post(self, url, payload, *, timeout=None, headers=None):
        self.calls.append((url, payload))
        if url.endswith("/tokenize"):
            return 200, {}, {"tokens": [1, 2, 3]}
        if url.endswith("/completion"):
            return 200, {}, {"timings": {"prompt_n": 1, "cache_n": len(payload["prompt"]) - 1}}
        crit = payload["questions"]["intent"]["criteria"]
        wide = len(crit) > 26
        if wide and self.wide_status != 200:
            return self.wide_status, {}, {"error": {"message": "too many options"}}
        passes = 1 + (self.split if wide else 0)
        n0 = 1500 if wide else 300
        for k in range(passes):
            self.task += 1
            n = n0 if k == 0 else 1
            self.log.text += CP_CREATED.format(slot=self.slot, task=self.task, n1=n0 + k - 1, n=n0 + k)
            self.log.text += EVAL.format(slot=self.slot, task=self.task, ms=n * 0.5, n=n)
        if wide and self.split:
            self.log.text += CP_RESTORED.format(slot=self.slot, task=self.task, n1=n0 - 1, n=n0)
        gold = self.gold_of(payload) if self.gold_of else next(iter(crit))
        rest = (1 - self.p_gold) / (len(crit) - 1)
        probs = {k: (self.p_gold if k == gold else rest) for k in crit}
        entry = {"q": "intent", "kind": "plain", "cache_n": None, "prompt_n": None, "prompt_ms": None,
                 "tokens": n0, "stop": "diverged", "ms": 40.0 if wide else 20.0, "miss": 3, "passes": passes}
        if wide:
            entry.update(unread=self.unread, split=self.split)
        for k in self.drop:
            entry.pop(k, None)
        return 200, {"x-callo-slot": str(self.slot), "x-callo-timings": json.dumps([entry])}, {
            "answers": {"intent": {"type": "choice", "choice": gold, "probabilities": probs}}}


def test_a_wide_serve_readout_reads_the_header_the_answer_and_its_log_segment():
    log = FakeLog()
    log.text = CP_CREATED.format(slot=0, task=0, n1=9, n=10)              # before the request: never counted
    srv = FakeServe(log, split=2, unread=0.004)
    it = _item(30, gold=0)
    r = w.serve_readout("http://c", it, "wide", post=srv.post, log=log)
    assert "error" not in r and r["arm"] == "wide" and r["n_options"] == 30 and r["slot"] == 0
    assert (r["passes"], r["split"], r["unread"], r["ms"], r["miss"]) == (3, 2, 0.004, 40.0, 3)
    assert r["correct"] and r["argmax"] == "k0" and r["p_gold"] == 0.9
    assert (r["created"], r["restored"]) == (3, 1) and r["prompt_evals"] == [1500, 1, 1]
    assert r["split_prompt_n_log"] == [1, 1]                                  # the two split passes' own prompt_n
    assert srv.calls[0][0] == "http://c/v1/systemone" and srv.calls[0][1] == w.wide_body(it)


def test_a_control_readout_is_narrow_and_a_wrong_argmax_is_not_correct():
    log = FakeLog()
    srv = FakeServe(log, gold_of=lambda body: list(body["questions"]["intent"]["criteria"])[1])
    r = w.serve_readout("http://c", _item(30, gold=0), "control", post=srv.post, log=log)
    assert r["n_options"] == 10 and r["split"] is None and r["unread"] is None and not r["correct"]
    assert r["split_prompt_n_log"] is None and "error" not in r


def test_a_refused_or_headerless_wide_readout_is_an_error_row():
    log = FakeLog()
    r = w.serve_readout("http://c", _item(), "wide", post=FakeServe(log, wide_status=422).post, log=log)
    assert r["status"] == 422 and "HTTP 422" in r["error"]
    r = w.serve_readout("http://c", _item(), "wide", post=FakeServe(log, drop=("split",)).post, log=log)
    assert "no split / unread" in r["error"]


def test_reply_sizes_post_raw_completions_at_both_top_ks():
    seen = []

    def raw(url, payload):
        seen.append(payload["n_probs"])
        return 200, b"x" * (payload["n_probs"] * 50)

    out = w.reply_sizes("http://u", post=FakeServe(FakeLog()).post, raw_post=raw)
    assert out == {"n_probs_1000": 50000, "n_probs_100": 5000} and seen == [1000, 100]


# ---- the in-process readout --------------------------------------------------------------------------------------

def _engine(dists):
    return Engine(WideScorer(dists), FakeTemplate(), SETS)


def test_inprocess_read_records_the_readout_and_its_split_prefix():
    eng = _engine({(): {GA: lp(0.4), ID["ĠAB"]: lp(0.3), THE: lp(0.3)}, (GA,): {B_: lp(0.9), THE: lp(0.1)}})
    calls = w.record_calls(eng)
    row, splits, slot = w.inprocess_read(eng, _item(30, gold=1), calls)
    assert slot["prefix"] == [7, 8, 7] and slot["cands"] == [[GA, pytest.approx(0.4)]] == row["slot_cands"]
    assert row["split_passes"] == 1 and row["passes"] == 2 and row["wide"] and row["correct"] and row["argmax"] == "k1"
    assert row["option_mass"] == pytest.approx(0.66) and row["unread_mass"] == 0.0
    assert row["split_mass"] == pytest.approx(0.4) and row["partial_by_letter"][0] == ["A", pytest.approx(0.4)]
    assert splits == [[7, 8, 7, GA]] and row["split_tails"] == [[GA]] and row["mode"] == "capped"
    assert row["mass_ok"] and row["mass_tol"] == w.mass_tolerance(2) and math.isclose(sum(row["probs"]), 1.0)
    assert calls == [(7, 8, 7), (7, 8, 7, GA)]


def test_the_forced_split_settings_apply_inside_and_are_restored():
    eng = _engine({(): {GA: lp(0.006), ID["ĠB"]: lp(0.004), ID["ĠAB"]: lp(0.9), THE: lp(0.09)},
                   (GA,): {B_: lp(0.5), THE: lp(0.5)}, (ID["ĠB"],): {ID["A"]: lp(0.5), THE: lp(0.5)}})
    calls = w.record_calls(eng)
    capped, _, slot = w.inprocess_read(eng, _item(30, gold=1), calls)
    assert slot["cands"] == [[GA, pytest.approx(0.006)], [ID["ĠB"], pytest.approx(0.004)]]
    assert capped["split_passes"] == 0 and capped["unread_mass"] == pytest.approx(0.01)   # not above WIDE_SPLIT_READ
    before = (eng.split_read_above, eng.split_max_passes, eng.split_min_partial)
    with w.forced_split(eng):
        assert (eng.split_read_above, eng.split_max_passes, eng.split_min_partial) == (0.0, None, 1e-3)
        forced, splits, _ = w.inprocess_read(eng, _item(30, gold=1), calls, mode="forced")
    assert (eng.split_read_above, eng.split_max_passes, eng.split_min_partial) == before
    assert forced["split_passes"] == 2 and forced["unread_mass"] == 0.0 and len(splits) == 2
    assert w.tv(capped["probs"], forced["probs"]) < 0.01


def test_resend_sends_split_pass_1_then_split_pass_2_to_the_readout_slot():
    """s + t1 extends the readout's cache by one token; s + t2 diverges at len(s), the split-pass-2 restore path."""
    srv = FakeServe(FakeLog(), slot=2, split=1)
    r = w.resend("http://c", "http://u", _item(), [7, 8, 7], 26, 27, post=srv.post, delimiter="<|im_start|>user")
    assert r == {"id": "banking77/train/0", "prefix_len": 3, "t1": 26, "t2": 27, "slot": 2, "readout_split": 1,
                 "prompt_n": 1, "cache_n": 3, "prompt_n_2": 1, "cache_n_2": 3}
    (u1, p1), (u2, p2) = srv.calls[-2:]
    assert u1 == u2 == "http://u/completion" and p1["prompt"] == [7, 8, 7, 26] and p2["prompt"] == [7, 8, 7, 27]
    assert p1["id_slot"] == p2["id_slot"] == 2 and p1["cache_prompt"] is True and p1["message_delimiters"] == [
        {"role": "user", "delimiter": "<|im_start|>user"}]
    bad = w.resend("http://c", "http://u", _item(), [1], 2, 3, post=FakeServe(FakeLog(), wide_status=500).post,
                   delimiter=None)
    assert "error" in bad and "prompt_n" not in bad


def test_resend_order_prefers_serve_splits_then_the_largest_split_mass_and_needs_two_candidates():
    serve = [{"id": "a", "arm": "wide", "split": 0}, {"id": "b", "arm": "wide", "split": 1},
             {"id": "c", "arm": "wide", "split": 0}]
    inproc = [{"id": i, "mode": "capped", "split_mass": m} for i, m in (("a", 0.3), ("b", 0.01), ("c", 0.2),
                                                                        ("d", 0.5), ("e", 0.4))]
    two = {"prefix": [1], "cands": [[5, 0.2], [6, 0.1]]}
    slots = {"a": two, "b": two, "c": two, "d": {"prefix": [1], "cands": [[5, 0.5]]}, "e": two}
    assert [i for i, _ in w.resend_order(serve, inproc, slots)] == ["b", "e", "a"]
    assert w.resend_order(serve, inproc, {}) == []


# ---- gates, report, verdict --------------------------------------------------------------------------------------

RESENT = {"id": "x", "prompt_n": 1, "prompt_n_2": 1}


def _rows(n=4, *, split=0, unread=0.0, mass=0.95, correct=True, control_correct=True, forced_shift=0.0):
    serve, inproc, forced = [], [], []
    for i in range(n):
        src = "banking77" if i < n // 2 else "clinc150"
        iid = f"{src}/train/{i}"
        serve.append({"id": iid, "source": src, "arm": "wide", "split": split, "unread": unread, "ms": 40.0 + i,
                      "wall_ms": 50.0, "correct": correct, "created": 1 + split, "restored": 0,
                      "split_prompt_n_log": None})
        serve.append({"id": iid, "source": src, "arm": "control", "split": None, "unread": None, "ms": 20.0,
                      "wall_ms": 25.0, "correct": control_correct, "created": 1, "restored": 0,
                      "split_prompt_n_log": None})
        probs = [0.9, 0.1]
        inproc.append({"id": iid, "source": src, "mode": "capped", "option_mass": mass, "unread_mass": unread,
                       "split_passes": split, "split_mass": 0.01 * i, "partial_by_letter": [["A", 0.01 * i]],
                       "probs": probs, "mass_plus_unread": mass + unread, "mass_ok": True})
        if i < min(w.FORCED_N, n):
            forced.append({"id": iid, "source": src, "mode": "forced", "option_mass": mass, "unread_mass": 0.0,
                           "split_passes": 2, "probs": [0.9 - forced_shift, 0.1 + forced_shift],
                           "mass_plus_unread": mass, "mass_ok": True})
    return serve, inproc, forced


def test_gates_all_pass_on_clean_rows_and_name_every_section_8_gate():
    serve, inproc, forced = _rows()
    g = w.gates(serve, inproc, forced, [RESENT], n_items=4)
    assert set(g) == {"mass_median", "unread_share", "split0_share", "split_max", "split_prompt_n", "forced_tv",
                      "mass_bound", "accuracy"}
    assert {k: v["status"] for k, v in g.items()} == dict.fromkeys(g, "PASS") and w.overall(g) == "PASS"
    assert g["split_prompt_n"]["value"] == 1 and g["accuracy"]["value"]["wide"] == {"banking77": 1.0, "clinc150": 1.0}


@pytest.mark.parametrize(("kw", "resend", "gate"), [
    ({"mass": 0.85}, None, "mass_median"),
    ({"unread": 0.02}, None, "unread_share"),
    ({"split": 1}, None, "split0_share"),
    ({"split": 3}, None, "split_max"),
    ({}, [{**RESENT, "prompt_n": 17}], "split_prompt_n"),
    ({}, [{**RESENT, "prompt_n_2": 17}], "split_prompt_n"),
    ({"forced_shift": 0.02}, None, "forced_tv"),
    ({"correct": False}, None, "accuracy"),
])
def test_each_gate_fails_on_its_own_threshold(kw, resend, gate):
    serve, inproc, forced = _rows(**kw)
    g = w.gates(serve, inproc, forced, resend or [RESENT], n_items=4)
    assert g[gate]["status"] == "FAIL" and w.overall(g) == "FAIL"


def test_accuracy_needs_the_control_margin_too_and_is_only_reported_on_gemma():
    serve, inproc, forced = _rows(n=8)
    for r in serve:
        if r["arm"] == "wide" and r["id"].endswith(("/0", "/1", "/4", "/5")):
            r["correct"] = False                                        # 0.5 per dataset, control 1.0: 0.5 < 0.75
    g = w.gates(serve, inproc, forced, [], n_items=8)
    assert g["accuracy"]["status"] == "FAIL" and g["accuracy"]["value"]["wide_pooled"] == 0.5
    g = w.gates(serve, inproc, forced, [], n_items=8, accuracy="reported")
    assert g["accuracy"]["status"] == "REPORTED" and "reported only" in g["accuracy"]["note"]


def test_gates_never_pass_on_missing_rows():
    serve, inproc, forced = _rows()
    serve[0] = {**serve[0], "error": "HTTP 500"}
    g = w.gates(serve, inproc[:3], forced[:2], [{"id": "x", "error": "boom"}], n_items=4)
    for k in ("mass_median", "unread_share", "split0_share", "split_max", "forced_tv", "mass_bound", "accuracy",
              "split_prompt_n"):
        assert g[k]["status"] == "ERROR", k
    assert w.overall(g) == "ERROR"
    assert "3 of 4 in-process" in g["mass_median"]["note"] and "HTTP 500" not in g["unread_share"]["note"]


def test_no_split_anywhere_makes_the_prompt_n_gate_not_applicable():
    serve, inproc, forced = _rows()
    g = w.gates(serve, inproc, forced, [], n_items=4)
    assert g["split_prompt_n"]["status"] == "N/A" and w.overall(g) == "PASS"


def test_a_mass_bound_violation_fails_and_names_the_item():
    serve, inproc, forced = _rows()
    forced[1] = {**forced[1], "mass_ok": False, "mass_plus_unread": 1.01}
    g = w.gates(serve, inproc, forced, [], n_items=4)
    assert g["mass_bound"]["status"] == "FAIL" and "(forced)" in g["mass_bound"]["note"]


MASS_ERROR = ("InvalidInput: option mass is 1.400000 at prefix (), above 1 by 0.4 (float32 rounding of 2 summed "
              "terms explains at most 1e-06): ...")


def test_i1_one_missing_capped_row_makes_the_mass_bound_an_error_not_a_pass():
    serve, inproc, forced = _rows()
    g = w.gates(serve, inproc[1:], forced, [RESENT], n_items=4)
    assert g["mass_bound"]["status"] == "ERROR" and "3 of 4 in-process" in g["mass_bound"]["note"]
    assert g["mass_median"]["status"] == "ERROR" and g["unread_share"]["status"] == "PASS"


def test_i1_an_engine_mass_refusal_is_a_mass_bound_fail_naming_the_item():
    serve, inproc, forced = _rows()
    inproc[2] = {"id": inproc[2]["id"], "source": "clinc150", "mode": "capped", "error": MASS_ERROR}
    g = w.gates(serve, inproc, forced, [RESENT], n_items=4)
    assert g["mass_bound"]["status"] == "FAIL" and inproc[2]["id"] in g["mass_bound"]["note"]
    assert w.overall(g) == "FAIL"
    other = [*inproc[:2], {**inproc[2], "error": "OSError: timed out"}, inproc[3]]
    assert w.gates(serve, other, forced, [RESENT], n_items=4)["mass_bound"]["status"] == "ERROR"


def test_i2_forced_reads_that_expand_nothing_make_the_tv_gate_not_applicable():
    serve, inproc, forced = _rows(forced_shift=0.003)
    forced = [{**f, "split_passes": 0} for f in forced]
    for c in inproc:
        c["split_mass"] = 4e-6
    g = w.gates(serve, inproc, forced, [RESENT], n_items=4)
    assert g["forced_tv"]["status"] == "N/A"
    assert g["forced_tv"]["note"] == "forced reads expanded nothing (max split mass 4e-06 < 1e-3)"
    assert g["forced_tv"]["value"] == pytest.approx(0.003)                   # still shown: it is re-read noise
    assert w.overall(g) == "PASS"


def test_i2_the_noise_baseline_is_reported_next_to_the_tv():
    serve, inproc, forced = _rows(forced_shift=0.004)
    baseline = [{**c, "mode": "baseline", "probs": [0.899, 0.101]} for c in inproc]
    g = w.gates(serve, inproc, forced, [RESENT], n_items=4, baseline=baseline)
    assert g["forced_tv"]["status"] == "PASS" and "noise baseline max 0.001" in g["forced_tv"]["note"]
    rep = w.report(serve, inproc, forced, [RESENT], {}, baseline=baseline)
    f0 = rep["forced"][0]
    assert f0["tv"] == pytest.approx(0.004) and f0["noise_tv"] == pytest.approx(0.001)
    assert rep["noise_tv_max"] == pytest.approx(0.001)
    assert w.report(serve, inproc, forced, [], {})["forced"][0]["noise_tv"] is None      # an older record: no baseline


def test_i3_the_prompt_n_gate_folds_in_the_log_values_and_needs_split_pass_2():
    serve, inproc, forced = _rows()
    serve[0]["split_prompt_n_log"] = [1, 20]
    g = w.gates(serve, inproc, forced, [RESENT], n_items=4)
    assert g["split_prompt_n"]["status"] == "FAIL" and g["split_prompt_n"]["value"] == 20
    serve[0]["split_prompt_n_log"] = [1, 2]
    g = w.gates(serve, inproc, forced, [RESENT], n_items=4)
    assert g["split_prompt_n"]["status"] == "PASS" and g["split_prompt_n"]["value"] == 2
    assert g["split_prompt_n"]["n"] == 4                                      # t1, t2 and the two log values
    assert g["split_prompt_n"]["sources"] == {"s+t1": 1, "s+t2": 1, "-v log": 2}
    rep = w.report(serve, inproc, forced, [RESENT], {})
    line = w.verdict_lines("tiel", g, rep, {"serve": 8, "inprocess": 4, "forced": 4, "resend": 1})[4]
    assert "max 2 over 4 (s+t1 1, s+t2 1, -v log 2; <= 16): PASS" in line
    old = w.gates(serve, inproc, forced, [{"id": "x", "prompt_n": 1}], n_items=4)
    assert old["split_prompt_n"]["status"] == "ERROR" and "--only-resend" in old["split_prompt_n"]["note"]
    serve[0]["split"] = 1
    none = w.gates(serve, inproc, forced, [], n_items=4)
    assert none["split_prompt_n"]["status"] == "ERROR"                       # a split happened and nothing was re-sent


@pytest.mark.parametrize(("good", "status"), [(38, "PASS"), (37, "FAIL")])
def test_m8_unread_share_boundary_at_38_of_40(good, status):
    serve, inproc, forced = _rows(n=40)
    wide = [r for r in serve if r["arm"] == "wide"]
    for r in wide[good:]:
        r["unread"] = 0.02
    assert w.gates(serve, inproc, forced, [RESENT], n_items=40)["unread_share"]["status"] == status


@pytest.mark.parametrize(("good", "status"), [(36, "PASS"), (35, "FAIL")])
def test_m8_split0_share_boundary_at_36_of_40(good, status):
    serve, inproc, forced = _rows(n=40)
    wide = [r for r in serve if r["arm"] == "wide"]
    for r in wide[good:]:
        r["split"] = 1
    assert w.gates(serve, inproc, forced, [RESENT], n_items=40)["split0_share"]["status"] == status


def test_m5_the_report_shows_the_in_process_unread_next_to_the_header_value():
    serve, inproc, forced = _rows(unread=0.002)
    inproc[0]["unread_mass"] = 0.003
    rep = w.report(serve, inproc, forced, [], {})
    assert rep["unread"][0] == [inproc[0]["id"], 0.002, 0.003]


def test_report_and_the_ten_line_verdict():
    serve, inproc, forced = _rows(split=0)
    inproc[3]["split_mass"] = 0.2
    g = w.gates(serve, inproc, forced, [], n_items=4)
    rep = w.report(serve, inproc, forced, [], {"n_probs_1000": 52000, "n_probs_100": 5200})
    assert rep["noise_tv_max"] is None
    assert rep["ms"]["wide"] == {"p50": 41.0, "p95": 43.0, "n": 4} and rep["ms"]["control"]["p50"] == 20.0
    assert rep["split_mass_median"] == pytest.approx(0.015) and rep["split_mass_trigger"] is False
    assert rep["checkpoints"]["wide"]["created_mean"] == 1.0 and len(rep["checkpoints"]["per_readout"]) == 8
    assert rep["partial_by_letter"]["banking77/train/1"] == [["A", 0.01]]
    lines = w.verdict_lines("tiel", g, rep, {"serve": 8, "inprocess": 4, "forced": 4, "resend": 0})
    assert len(lines) == 10 and lines[0].startswith("wide_check tiel: PASS (gates: ")
    assert "N/A" in lines[0] and "FAIL" not in lines[0]                       # N/A gates are counted, not hidden
    assert sum(int(t.split()[0]) for t in lines[0].split("gates: ")[1].split(";")[0].split(", ")) == len(g)
    assert "over n/a (s+t1 0, s+t2 0, -v log 0; <= 16): N/A" in lines[4]
    text = "\n".join(lines)
    assert "N/A (no split pass" in text and "not reached" in text and "52000" in text


def test_pct_is_the_nearest_rank():
    assert w._pct([], 0.5) is None
    assert w._pct([3.0, 1.0, 2.0, 4.0], 0.5) == 2.0 and w._pct(list(range(1, 101)), 0.95) == 95


# ---- hygiene -----------------------------------------------------------------------------------------------------

ROOT = Path(w.__file__).resolve().parents[1]
CLEAN = {"head": "abc", "dirty": "", "root": str(ROOT),
         "sources": {"callo": str(ROOT / "src" / "callo"), "bench": str(ROOT / "bench")}}


def test_code_refusals():
    assert w.code_refusals(CLEAN) == []
    assert w.code_refusals({**CLEAN, "dirty": "?? bench/.DS_Store\n?? src/.DS_Store"}) == []
    assert "uncommitted" in w.code_refusals({**CLEAN, "dirty": " M src/callo/engine.py"})[0]
    assert "uncommitted" in w.code_refusals({**CLEAN, "dirty": "?? bench/extra.py"})[0]
    assert "imported from" in w.code_refusals({**CLEAN, "sources": {"callo": "/elsewhere", "bench": CLEAN["sources"][
        "bench"]}})[0]
    assert "git worktree" in w.code_refusals({"error": "not a repo"})[0]


@pytest.fixture
def no_launch(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("nothing may launch")
    monkeypatch.setattr(w, "_launch", boom)
    monkeypatch.setattr(w, "code_state", lambda: CLEAN)
    monkeypatch.setattr(w.lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(w, "disk_problems", lambda paths, gib: [])


def test_main_refuses_a_busy_or_unknown_gpu(no_launch, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(w.lib, "running_gpu_servers", lambda: [(123, "llama-server -m x.gguf")])
    assert w.main(["--out", str(tmp_path / "o")]) == 2
    assert "holds the GPU" in capsys.readouterr().err

    def fail():
        raise RuntimeError("could not list processes")
    monkeypatch.setattr(w.lib, "running_gpu_servers", fail)
    assert w.main(["--out", str(tmp_path / "o")]) == 2
    assert "cannot tell" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_main_refuses_dirty_code_bad_items_low_disk_and_a_live_run_pid(no_launch, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(w, "code_state", lambda: {**CLEAN, "dirty": " M src/callo/engine.py"})
    assert w.main(["--out", str(tmp_path / "o")]) == 2
    assert "uncommitted" in capsys.readouterr().err
    monkeypatch.setattr(w, "code_state", lambda: CLEAN)
    (tmp_path / "bad.jsonl").write_text("{}\n")
    assert w.main(["--out", str(tmp_path / "o"), "--items", str(tmp_path / "bad.jsonl")]) == 2
    monkeypatch.setattr(w, "disk_problems", lambda paths, gib: ["/ has 3.0 GiB free (< 20 GiB)"])
    assert w.main(["--out", str(tmp_path / "o")]) == 2
    assert "GiB free" in capsys.readouterr().err
    monkeypatch.setattr(w, "disk_problems", lambda paths, gib: [])
    out = tmp_path / "o"
    out.mkdir()
    (out / "run.pid").write_text("1\n")                                     # pid 1 is alive and not this bench
    assert w.main(["--out", str(out)]) == 2
    assert "remove run.pid by hand" in capsys.readouterr().err and (out / "run.pid").read_text() == "1\n"


def test_the_launch_is_cached_pinned_and_verbose(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(w, "Callo", lambda out, **k: seen.append((out, k)) or "c")
    a = argparse.Namespace(out=tmp_path, model="tiel", llama_server="ls", ctx=8192, parallel=2)
    assert w._launch(a, "/m.gguf") == "c"
    out, k = seen[0]
    assert out == tmp_path / "tiel" and k["cache"] is True and k["model"] == "/m.gguf"
    assert k["flags"] == ["--s1-slots", "1", "--", "-v"]


# ---- the run, end to end on fakes --------------------------------------------------------------------------------

class FakeCallo:
    def __init__(self, events, *, coverage=255):
        self.events, self.url, self.upstream = events, "http://127.0.0.1:51001", "http://127.0.0.1:51002"
        self.build_info, self.coverage, self.pid = "b1-x", coverage, 4000
        events.append("launch")

    def stats(self):
        return {"wide_coverage": self.coverage, "user_delimiter": "<|im_start|>user",
                "provenance": {"wide_contract": {"top_k": WIDE_TOP_K}}}

    def log_text(self):
        return "launched llama-server pid 4000 on http://127.0.0.1:51002 (log /tmp/ll.log)"

    def finish(self):
        self.events.append("finish")
        return 0


@pytest.fixture
def live(monkeypatch, tmp_path):
    env = argparse.Namespace(events=[], log=FakeLog(), coverage=255, gpu=[])
    gold = {}

    def gold_of(body):
        return gold[body["state"]]

    _, items = w.load_items()
    for it in items:
        gold[it["text"]] = it["gold"]
    env.server = FakeServe(env.log, gold_of=gold_of)
    monkeypatch.setattr(w, "_launch", lambda a, model: FakeCallo(env.events, coverage=env.coverage))
    monkeypatch.setattr(w, "code_state", lambda: CLEAN)
    monkeypatch.setattr(w, "_POST", lambda *a, **k: env.server.post(*a, **k))
    monkeypatch.setattr(w, "_log_tail", lambda path: env.log)
    monkeypatch.setattr(w, "_raw_post", lambda url, payload: (200, b"x" * payload["n_probs"]))
    monkeypatch.setattr(w, "_engine_for", lambda upstream, gguf: _engine(            # two first-letter candidates
        {(): {ID["ĠAA"]: lp(0.95), GA: lp(0.003), ID["ĠB"]: lp(0.002), THE: lp(0.045)},
         (GA,): {ID["A"]: lp(0.9), THE: lp(0.1)}, (ID["ĠB"],): {ID["A"]: lp(0.5), THE: lp(0.5)}}))
    monkeypatch.setattr(w.lib, "running_gpu_servers", lambda: list(env.gpu))
    env.args = lambda model="tiel", **kw: argparse.Namespace(**{
        "model": model, "out": tmp_path, "items": w.ITEMS, "llama_server": "ls", "ctx": 8192, "parallel": 2,
        "allow_dirty": False, "only_resend": False, **kw})
    env.summary = lambda: json.loads((tmp_path / "summary.json").read_text())
    return env


def test_run_measures_serve_inprocess_and_forced_and_writes_the_verdict(live, tmp_path):
    meta, items = w.load_items()
    assert w._run(live.args(), items, meta) == 0
    assert live.events == ["launch", "finish"]
    m = live.summary()["models"]["tiel"]
    assert m["verdict"] == "PASS" and m["wide_coverage"] == 255 and m["llama_log"] == "/tmp/ll.log"
    assert m["counts"] == {"serve": 80, "inprocess": 40, "baseline": 5, "forced": 5, "resend": 3}
    assert m["gates"]["split_prompt_n"]["status"] == "PASS" and m["gpu_after"] == []
    assert m["gates"]["forced_tv"]["status"] == "PASS" and "noise baseline" in m["gates"]["forced_tv"]["note"]
    assert m["gates"]["accuracy"]["status"] == "PASS" and m["gates"]["mass_median"]["value"] == pytest.approx(0.95)
    assert m["report"]["reply_bytes"] == {"n_probs_1000": 1000, "n_probs_100": 100}
    assert m["report"]["checkpoints"]["wide"]["created_mean"] == 1.0 and m["items_sha256"] and m["code"] == CLEAN
    rows = [json.loads(line) for line in (tmp_path / "tiel" / "rows.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows].count("serve") == 80 and [r["kind"] for r in rows].count("forced") == 5
    verdict = (tmp_path / "verdict-tiel.txt").read_text().splitlines()
    assert len(verdict) == 10 and verdict == m["verdict_lines"]


def test_a_gemma_run_keeps_the_tiel_entry_and_reports_accuracy(live, tmp_path):
    meta, items = w.load_items()
    w._run(live.args(), items, meta)
    meta, gitems = w.load_items(model="gemma")
    assert w._run(live.args("gemma"), gitems, meta) == 0
    s = live.summary()["models"]
    assert set(s) == {"tiel", "gemma"} and s["gemma"]["counts"]["serve"] == 20
    assert s["gemma"]["gates"]["accuracy"]["status"] == "REPORTED"


def test_a_short_coverage_server_is_an_error_before_any_readout(live, tmp_path):
    live.coverage = 9
    meta, items = w.load_items()
    assert w._run(live.args(), items, meta) == 1
    m = live.summary()["models"]["tiel"]
    assert m["verdict"] == "ERROR" and "wide_coverage is 9" in m["errors"][0]
    assert live.server.calls == [] and live.events == ["launch", "finish"]


def test_a_stop_signal_records_stopped_and_still_stops_the_server(live, monkeypatch):
    def stop(*a, **k):
        raise SystemExit(143)
    monkeypatch.setattr(w, "_POST", stop)
    meta, items = w.load_items()
    with pytest.raises(SystemExit):
        w._run(live.args(), items, meta)
    assert live.summary()["models"]["tiel"]["verdict"] == "STOPPED" and live.events == ["launch", "finish"]


def test_an_unexpected_exception_is_recorded_with_its_traceback(live, monkeypatch):
    def boom(upstream, gguf):
        raise KeyError("vocab")
    monkeypatch.setattr(w, "_engine_for", boom)
    meta, items = w.load_items()
    assert w._run(live.args(), items, meta) == 1
    m = live.summary()["models"]["tiel"]
    assert m["verdict"] == "ERROR" and "KeyError" in m["errors"][0] and "Traceback" in m["traceback"]
    assert live.events == ["launch", "finish"]


def test_one_failed_readout_is_its_own_error_row_and_the_run_goes_on(live, monkeypatch):
    import urllib.error
    meta, items = w.load_items()
    first = items[0]["text"]
    inner = live.server.post

    def post(url, payload, **k):
        if url.endswith("/v1/systemone") and payload["state"] == first and len(
                payload["questions"]["intent"]["criteria"]) > 26:
            raise urllib.error.URLError("timed out")
        return inner(url, payload, **k)
    monkeypatch.setattr(w, "_POST", post)
    assert w._run(live.args(), items, meta) == 1
    m = live.summary()["models"]["tiel"]
    assert m["verdict"] == "ERROR" and m["counts"]["serve"] == 80 and m["counts"]["inprocess"] == 40
    assert m["gates"]["unread_share"]["status"] == "ERROR" and "39 of 40" in m["gates"]["unread_share"]["note"]
    assert m["gates"]["mass_median"]["status"] == "PASS"


def test_m3_gpu_after_lists_the_other_model_servers_but_not_our_own(live):
    live.gpu = [(4000, "llama-server ours"), (77, "llama-server other")]
    meta, items = w.load_items(model="gemma")
    w._run(live.args("gemma"), items, meta)
    assert live.summary()["models"]["gemma"]["gpu_after"] == [[77, "llama-server other"]]


def test_m4_a_value_error_in_one_serve_readout_is_its_row(live, monkeypatch):
    meta, items = w.load_items(model="gemma")
    inner = live.server.post

    def post(url, payload, **k):
        if url.endswith("/v1/systemone") and payload["state"] == items[3]["text"]:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return inner(url, payload, **k)
    monkeypatch.setattr(w, "_POST", post)
    assert w._run(live.args("gemma"), items, meta) == 1
    m = live.summary()["models"]["gemma"]
    assert m["counts"]["serve"] == 20 and m["gates"]["unread_share"]["status"] == "ERROR"


def test_m8_an_engine_mass_refusal_in_the_run_fails_the_mass_bound_by_name(live, monkeypatch):
    monkeypatch.setattr(w, "_engine_for", lambda upstream, gguf: _engine(
        {(): {ID["ĠAA"]: lp(0.7), ID["AA"]: lp(0.7)}}))
    meta, items = w.load_items(model="gemma")
    assert w._run(live.args("gemma"), items, meta) == 1
    m = live.summary()["models"]["gemma"]
    rows = [json.loads(line) for line in (live.args().out / "gemma" / "rows.jsonl").read_text().splitlines()]
    assert all(r["error"].startswith("InvalidInput: option mass is") for r in rows if r["kind"] == "inprocess")
    assert m["gates"]["mass_bound"]["status"] == "FAIL" and items[0]["id"] in m["gates"]["mass_bound"]["note"]
    assert m["verdict"] == "FAIL"


def _no_launch(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("nothing may launch")
    monkeypatch.setattr(w, "_launch", boom)
    monkeypatch.setattr(w.lib, "running_gpu_servers", boom)


def test_recompute_rewrites_the_summary_offline_from_the_rows(live, monkeypatch, tmp_path):
    meta, items = w.load_items()
    w._run(live.args(), items, meta)
    before = live.summary()["models"]["tiel"]
    _no_launch(monkeypatch)
    assert w.main(["--recompute", "--model", "tiel", "--out", str(tmp_path)]) == 0
    after = live.summary()["models"]["tiel"]
    assert after["recomputed"]["code"] == CLEAN["head"] and after["gates"] == before["gates"]
    assert after["report"]["reply_bytes"] == before["report"]["reply_bytes"] and after["started"] == before["started"]
    assert (tmp_path / "verdict-tiel.txt").read_text().splitlines() == after["verdict_lines"]


def test_recompute_turns_a_forced_read_that_expanded_nothing_into_not_applicable(live, monkeypatch, tmp_path):
    meta, items = w.load_items(model="gemma")
    w._run(live.args("gemma"), items, meta)
    path = tmp_path / "gemma" / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [{**r, "split_passes": 0, "probs": [*r["probs"][:-1], r["probs"][-1]]} if r["kind"] == "forced" else r
            for r in rows if r["kind"] != "baseline"]                   # the record the GPU run wrote: no baseline
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _no_launch(monkeypatch)
    assert w.main(["--recompute", "--model", "gemma", "--out", str(tmp_path)]) == 0
    g = live.summary()["models"]["gemma"]["gates"]
    assert g["forced_tv"]["status"] == "N/A" and g["forced_tv"]["note"].startswith("forced reads expanded nothing")


def test_recompute_refuses_without_rows_or_with_a_live_run(live, monkeypatch, tmp_path, capsys):
    _no_launch(monkeypatch)
    assert w.main(["--recompute", "--model", "tiel", "--out", str(tmp_path)]) == 2
    assert "rows.jsonl" in capsys.readouterr().err
    (tmp_path / "run.pid").write_text("1\n")
    assert w.main(["--recompute", "--model", "tiel", "--out", str(tmp_path)]) == 2


def test_only_resend_measures_split_pass_2_and_recomputes_the_verdict(live, monkeypatch, tmp_path):
    meta, items = w.load_items()
    w._run(live.args(), items, meta)
    path = tmp_path / "tiel" / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [{k: v for k, v in r.items() if not k.endswith("_2")} if r["kind"] == "resend" else r for r in rows]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))          # the record the GPU run wrote: t1 only
    w.recompute(tmp_path, "tiel")
    m = live.summary()["models"]["tiel"]
    assert m["gates"]["split_prompt_n"]["status"] == "ERROR" and m["verdict"] == "ERROR"
    live.events.clear()
    live.server.calls.clear()
    assert w._run(live.args(only_resend=True), items, meta) == 0
    assert live.events == ["launch", "finish"]
    assert not [u for u, _ in live.server.calls if u.endswith("/tokenize")]          # no reply-size probe again
    systemone = [p for u, p in live.server.calls if u.endswith("/v1/systemone")]
    assert len(systemone) == 3                                                        # only the re-sends' readouts
    m = live.summary()["models"]["tiel"]
    assert m["resend_supplement"]["code"] == CLEAN and m["resend_supplement"]["verdict"] == "DONE"
    assert m["gates"]["split_prompt_n"]["status"] == "PASS" and m["verdict"] == "PASS" and m["counts"]["resend"] == 3
    sup = [json.loads(line) for line in (tmp_path / "tiel" / "rows-resend.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in sup].count("resend") == 3 and all("prompt_n_2" in r for r in sup if r["kind"] == "resend")
    assert len(path.read_text().splitlines()) == len(rows)                           # rows.jsonl is never rewritten
