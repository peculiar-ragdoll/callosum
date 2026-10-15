"""`callo calibrate --think-budgets` for one server: the settings, the data, the probe client,
the raw file, the fit and the profile, the run against an in-process `callo serve` on the thinking fake, and the CLI."""
import dataclasses
import json
import math
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fake_upstream import FakeUpstream, planted_answer
from helpers import Running, get_json, tiny_gguf

import callo.calibrate_think as ct
import callo.cli
from callo.calibrate_bias import Refusal, RunFailed
from callo.calibrate_think import (
    DEFAULT_BUDGETS,
    Item,
    Outcome,
    ProbeClient,
    RawThink,
    ThinkSettings,
    _open_raw,
    _write,
    build_profile,
    calibrate_one,
    cells,
    check_preconditions,
    data_sha256,
    header_diff,
    load_data,
    not_capable_line,
    outcomes,
    parse_budgets,
    pick_note,
    prefill_reused,
    raw_header,
    read_raw,
    shuffled,
)
from callo.errors import InvalidInput
from callo.think import (
    budget_latency,
    check_pinned,
    json_safe,
    load_think_profile,
    parse_sampling,
    resolve_sampling,
)

OPTIONS = [["one", "two", "three"], ["red", "green", "blue", "cyan"], ["yes", "no"], ["alpha", "beta", "gamma"]]


def write_data(path: Path, n: int = 12, *, extra: list[str] = ()) -> Path:
    """n labelled 'choice' records whose label is the thinking fake's planted answer, two records per cluster, plus any
    extra raw lines."""
    lines = []
    for i in range(n):
        texts = OPTIONS[i % len(OPTIONS)]
        crit = {f"o{j}": t for j, t in enumerate(texts)}
        q = __import__("callo.systemone", fromlist=["parse_request"]).parse_request(
            {"state": f"state {i}", "questions": {"q": {"type": "choice", "instructions": f"pick {i}", "criteria": crit}}}
        ).questions[0]
        label = f"o{planted_answer(list(q.texts))}"
        lines.append(json.dumps({"state": f"state {i}", "questions": {"q": {"type": "choice", "instructions": f"pick {i}",
                                                                           "criteria": crit, "label": label}},
                                 "_meta": {"id": f"rec-{i}", "group_id": f"g{i // 2}", "variant": "clean"}}))
    path.write_text("\n".join([*lines, *extra]) + "\n")
    return path


def test_parse_budgets_defaults_adds_zero_and_refuses_bad_lists():
    assert parse_budgets(None) == list(DEFAULT_BUDGETS) and parse_budgets("") == [0, 256, 512, 1024]
    assert parse_budgets("1024,256") == [0, 256, 1024]
    for bad, why in (("x", "not a non-negative"), ("256,256", "twice"), ("40000", "hard cap"),
                     ("1,2,3,4,5,6,7,8", "at most 8"), ("-1", "not a non-negative"),
                     ("\u00b2", "not a non-negative"), ("\u0661\u0662", "not a non-negative")):  # ASCII digits only
        with pytest.raises(InvalidInput, match=why):
            parse_budgets(bad)


def test_load_data_builds_single_question_items_with_ids_clusters_and_labels(tmp_path):
    p = write_data(tmp_path / "d.jsonl", 4, extra=[
        json.dumps({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "x", "b": "y"}, "label": "a"}},
                    "_meta": {"id": "para", "variant": "paraphrase"}}),
        json.dumps({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}}),
        json.dumps({"state": "s", "images": ["data:image/png;base64,iVBORw0KGgo="],
                    "questions": {"q": {"type": "choice", "criteria": {"a": "x", "b": "y"}, "label": "a"}}, "_meta": {"id": "img"}}),
        json.dumps({"state": "s", "context": "ctx", "questions": {"q": {"type": "noul", "label": "true"},
                                                                 "r": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}}),
    ])
    items, skipped = load_data(p)
    assert [it.uid for it in items] == ["rec-0/q", "rec-1/q", "rec-2/q", "rec-3/q", "line8/q"]
    assert [it.cluster for it in items] == ["g0", "g0", "g1", "g1", "line8"]
    it = items[0]
    assert isinstance(it, Item) and it.label == planted_answer(list(it.question.texts)) and it.kind == "plain"
    assert it.body == {"state": "state 0", "model": "kev-latest",
                       "questions": {"q": {"type": "choice", "instructions": "pick 0", "criteria": {"o0": "one", "o1": "two", "o2": "three"}}}}
    assert items[-1].kind == "context" and items[-1].body["context"] == "ctx" and items[-1].label == 1
    assert skipped == Counter(records=8, variant=1, unlabelled=2, media=1)
    assert len(data_sha256(p)) == 64


def test_load_data_refuses_malformed_lines_duplicate_ids_and_unreadable_files(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json\n")
    with pytest.raises(InvalidInput, match="--data line 1: not valid JSON"):
        load_data(p)
    p.write_text("[1, 2]\n")
    with pytest.raises(InvalidInput, match="line 1: not a JSON object"):
        load_data(p)
    p.write_text(json.dumps({"questions": {"q": {"type": "noul"}}}) + "\n")
    with pytest.raises(InvalidInput, match="line 1: not a System One request"):
        load_data(p)
    rec = json.dumps({"state": "s", "questions": {"q": {"type": "noul", "label": "true"}}, "_meta": {"id": "dup"}})
    p.write_text(rec + "\n" + rec + "\n")
    with pytest.raises(InvalidInput, match="line 2: _meta.id 'dup' is duplicated"):
        load_data(p)
    with pytest.raises(InvalidInput, match="--data"):
        load_data(tmp_path / "missing.jsonl")
    p.write_bytes(b"\xff\xfe")
    with pytest.raises(InvalidInput, match="not UTF-8"):
        load_data(p)


def test_shuffled_is_deterministic_by_seed_and_moves_with_it(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 12))
    a, b = shuffled(items, 0), shuffled(items, 0)
    assert [x.uid for x in a] == [x.uid for x in b] and sorted(x.uid for x in a) == sorted(x.uid for x in items)
    assert [x.uid for x in shuffled(items, 1)] != [x.uid for x in a]


def test_settings_comparable_and_derived_budgets():
    s = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 256, 1024), seed=2, cache=True)
    assert s.bmax == 1024 and s.positive == (256, 1024)
    assert s.comparable() == {"data": {"name": "d.jsonl", "sha256": "d" * 64}, "budgets": [0, 256, 1024], "seed": 2,
                              "warmup": 3, "audit": 6, "cache": True, "sampling_policy": "model"}


class _Probe(BaseHTTPRequestHandler):
    """A scripted /callo/think-probe: a list of statuses to answer, then 200 with a canned reply."""
    script: list = []
    seen: list = []
    reply = {"readouts": 2, "questions": {"q": {"on_tokens": 40, "n_ctx": 4096, "budgets": {}}}}

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        type(self).seen.append((self.path, dict(self.headers), json.loads(self.rfile.read(n))))
        code = type(self).script.pop(0) if type(self).script else 200
        cut = code == "cut"                     # a 200 whose body stops short of its Content-Length: callo died mid-reply
        code = 200 if cut else code
        data = json.dumps(type(self).reply if code == 200 else {"detail": f"scripted {code}"}).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data) + (100 if cut else 0)))
        self.send_header("X-Callo-Slot", "0")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def probe_server():
    _Probe.script, _Probe.seen = [], []
    srv = HTTPServer(("127.0.0.1", 0), _Probe)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_the_probe_client_plans_reads_on_the_lane_and_retries_503_504_and_connection_errors(probe_server, tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 1))
    slept = []
    c = ProbeClient(probe_server, s1_slots=1, sleep=slept.append)
    assert c.plan(items[0]) == (40, 4096)
    path, headers, body = _Probe.seen[-1]
    assert path == "/callo/think-probe" and headers["X-Callo-Conversation"] == "callo-calibrate"
    assert body == {"request": items[0].body, "budgets": [0], "plan": True}
    _Probe.script = [503, 504]
    r = c.read(items[0], [0, 256], 7, {"temperature": 0.6})
    assert (r.slot, r.readouts, r.retries) == (0, 2, 0) and slept == [1.0, 4.0]
    assert _Probe.seen[-1][2] == {"request": items[0].body, "budgets": [0, 256], "seed": 7, "sampling": {"temperature": 0.6}}
    _Probe.script = [422]
    with pytest.raises(RunFailed, match="rec-0/q: HTTP 422"):
        c.read(items[0], [0], 1, None)
    _Probe.script = [503, 503, 503, 504]                 # the last reason is callo's detail, not the dict
    with pytest.raises(RunFailed, match=r"still failing after 3 retries \(HTTP 504: scripted 504\)$"):
        c.read(items[0], [0], 1, None)
    down = ProbeClient("http://127.0.0.1:1", s1_slots=1, sleep=slept.append)
    with pytest.raises(RunFailed, match="still failing after 3 retries"):
        down.plan(items[0])


def test_the_raw_header_its_diff_and_the_reader(tmp_path):
    settings = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 256))
    sampling = resolve_sampling(parse_sampling("fixed"))
    stats = {"model_key": "sha256:" + "a" * 64, "provenance": {"build_info": "b1", "chat_template_sha256": "t"},
             "router": {"reserved": [0]}, "kv": "unified", "n_ctx": 4096, "total_slots": 4,
             "cache_prompt": {"plain": False}, "position_bias": None, "host": {"fingerprint": "sha256:h"},
             "instance": "i1", "model_file": "m.gguf"}
    h = raw_header(stats, settings, sampling)
    assert h["server"] == {"kv": "unified", "s1_slots": 1, "n_ctx": 4096, "total_slots": 4, "cache_prompt": False,
                           "position_bias_sha256": None, "upstream_local": None}
    assert raw_header({**stats, "upstream_local": False}, settings, sampling)["server"]["upstream_local"] is False
    assert h["comparable"]["budgets"] == [0, 256] and h["comparable"]["sampling"]["policy"] == "fixed"
    assert header_diff(h, h) == ([], [])
    other = {**h, "provenance": {"build_info": "b2", "chat_template_sha256": "t"}, "host": {"fingerprint": "sha256:x"}}
    assert header_diff(h, other) == ([], ["build_info", "host"])
    assert header_diff(h, {**h, "model_key": "sha256:" + "b" * 64})[0] == ["model_key"]
    assert header_diff(h, {**h, "provenance": {"build_info": "b1", "chat_template_sha256": "u"}})[0] == ["provenance"]
    p = tmp_path / "x.raw.jsonl"
    rows = [json.dumps({"callo_think_raw": 1, **h}), json.dumps({"check": {"index": 0}}),
            json.dumps({"therm": {"at": "A", "limit": None}}), json.dumps({"phase": "A", "q": "rec-0/q"}),
            json.dumps({"phase": "audit", "q": "rec-0/q", "budget": 256}), '{"phase": "B", "q": "rec-1/q"'[:-1]]
    p.write_text("\n".join(rows))                                   # the last line is cut: partial
    raw = read_raw(p)
    assert raw.partial and len(raw.rows) == 2 and raw.checks == [{"index": 0}] and raw.therm == [{"at": "A", "limit": None}]
    assert cells(raw.rows) == {("A", "rec-0/q", None), ("audit", "rec-0/q", 256)}
    p.write_text("{}\n")
    with pytest.raises(InvalidInput, match="no callo --think-budgets raw header"):
        read_raw(p)
    p.write_text(rows[0] + "\n" + json.dumps({"phase": "Z", "q": "x"}) + "\n")
    with pytest.raises(InvalidInput, match="line 2 is not a raw record"):
        read_raw(p)


def test_the_probe_client_surfaces_a_422_detail_and_takes_the_readout_count_and_a_null_trace_as_given(probe_server,
                                                                                                         tmp_path):
    """The probe's 422 for a non-neutral pin reaches the user with its reason; `readouts` is
    the server's count (an eos trace probed at n_gen - 1 and n_gen makes one fewer readout than it has budgets), never
    len(budgets); a single-option question's `trace: null` passes through."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 1))
    c = ProbeClient(probe_server, s1_slots=1, sleep=lambda s: None)
    _Probe.script = [422]
    with pytest.raises(RunFailed, match=r"rec-0/q: HTTP 422: scripted 422$"):
        c.read(items[0], [0, 256], 1, {"temperature": 0.6, "dry_multiplier": 0.8})
    old = _Probe.reply
    try:
        _Probe.reply = {"readouts": 2, "questions": {"q": {"on_tokens": 40, "n_ctx": 4096, "trace": {"n_gen": 256},
                                                          "budgets": {"0": {}, "255": {}, "256": {}}}}}
        assert c.read(items[0], [0, 255, 256], 1, None).readouts == 2
        _Probe.reply = {"readouts": 1, "questions": {"q": {"on_tokens": None, "n_ctx": 4096, "trace": None,
                                                          "content": None, "budgets": {"0": {}, "256": {}}}}}
        r = c.read(items[0], [0, 256], 1, None)
        assert r.readouts == 1 and r.body["questions"]["q"]["trace"] is None
    finally:
        _Probe.reply = old


def test_header_diff_reads_a_none_host_fingerprint_as_an_unknown_host_never_a_match():
    """None on either side is "unknown host", a warning, never a silent match."""
    settings = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 256))
    stats = {"model_key": "sha256:" + "a" * 64, "provenance": {"build_info": "b1"}, "host": {"fingerprint": "sha256:h"}}
    h = raw_header(stats, settings, resolve_sampling(parse_sampling("fixed")))
    unknown = {**h, "host": {"fingerprint": None}}
    assert header_diff(h, unknown) == ([], ["unknown host"]) and header_diff(unknown, h) == ([], ["unknown host"])
    assert header_diff(unknown, unknown) == ([], ["unknown host"])
    assert header_diff({**h, "host": None}, {**h, "host": None}) == ([], ["unknown host"])


def test_open_raw_creates_refuses_and_resumes_by_the_header(tmp_path):
    settings = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 256))
    stats = {"model_key": "sha256:" + "a" * 64, "provenance": {"build_info": "b1"}, "host": {"fingerprint": "sha256:h"}}
    h = raw_header(stats, settings, resolve_sampling(parse_sampling("fixed")))
    p = tmp_path / "out" / "m.raw.jsonl"
    logs: list[str] = []
    f, raw = _open_raw(p, h, resume=False, log=logs.append)
    _write(f, {"phase": "A", "q": "rec-0/q", "probs": [1.0], "ms": 1.0})
    f.close()
    assert raw.rows == [] and json.loads(p.read_text().splitlines()[0]) == {"callo_think_raw": 1, **h}
    with pytest.raises(Refusal, match="already exists. Pass --resume"):
        _open_raw(p, h, resume=False, log=logs.append)
    with pytest.raises(Refusal, match=r"different run \(comparable differ"):
        _open_raw(p, raw_header(stats, ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 512)),
                                resolve_sampling(parse_sampling("fixed"))), resume=True, log=logs.append)
    with p.open("a") as fh:
        fh.write('{"phase": "B", "q": "rec-')                        # a hard kill's partial last line
    f, raw = _open_raw(p, {**h, "host": {"fingerprint": None}}, resume=True, log=logs.append)
    _write(f, {"phase": "B", "q": "rec-0/q"})
    f.close()
    assert [r["q"] for r in raw.rows] == ["rec-0/q"] and raw.partial
    again = read_raw(p)
    assert not again.partial and cells(again.rows) == {("A", "rec-0/q", None), ("B", "rec-0/q", None)}
    assert logs == ["WARNING: --resume: unknown host: cannot tell whether the raw file was measured on this machine; "
                    "its latencies may not be comparable"]
    with pytest.raises(Refusal, match=r"row 2 \(B rec-0/q\) is incomplete: a phase-B row needs budgets"):
        _open_raw(p, h, resume=True, log=logs.append)                # valid JSON, but nothing the fit can read
    p.write_text("{}\n")
    with pytest.raises(Refusal, match="refusing to resume: .* no callo --think-budgets raw header"):
        _open_raw(p, h, resume=True, log=logs.append)


# ---- the probe client's failures and the data's edge cases ---------------------------------------------------

def test_a_reply_cut_mid_body_is_retried_as_a_dropped_connection_not_a_traceback(probe_server, tmp_path):
    """`http.client.IncompleteRead` is no OSError; the probe client retries it like any connection error, and a
    reply that never arrives whole is RunFailed naming the question."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 1))
    slept = []
    c = ProbeClient(probe_server, s1_slots=1, sleep=slept.append)
    _Probe.script = ["cut"]
    r = c.read(items[0], [0, 256], 1, None)
    assert (r.readouts, r.retries) == (2, 1) and slept == [1.0]
    _Probe.script = ["cut"] * 4
    with pytest.raises(RunFailed, match=r"rec-0/q \(plan\): still failing after 3 retries \(IncompleteRead"):
        c.plan(items[0])
    assert slept == [1.0, 1.0, 4.0, 16.0]


def test_unusable_200_replies_and_unreadable_raw_files_are_clean_errors(probe_server, tmp_path):
    """A 200 the client cannot read is RunFailed naming the question (exit 1), and an unreadable raw
    file is InvalidInput, never a traceback."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 1))
    c = ProbeClient(probe_server, s1_slots=1, sleep=lambda s: None)
    old = _Probe.reply
    try:
        for bad in ({"readouts": 0}, {"questions": {"other": {}}}, {"questions": {"q": {"on_tokens": None}}},
                    {"questions": {"q": {"on_tokens": "x"}}}, {"questions": {"q": {"on_tokens": 4, "n_ctx": "big"}}}):
            _Probe.reply = bad
            with pytest.raises(RunFailed, match=r"rec-0/q \(plan\): unusable reply"):
                c.plan(items[0])
        for bad in ({"readouts": "two", "questions": {}}, {"readouts": 1}, {"readouts": 1, "questions": {"q": None}}):
            _Probe.reply = bad
            with pytest.raises(RunFailed, match=r"rec-0/q: unusable reply"):
                c.read(items[0], [0], 1, None)
    finally:
        _Probe.reply = old
    with pytest.raises(InvalidInput, match="cannot read"):
        read_raw(tmp_path / "missing.raw.jsonl")
    with pytest.raises(InvalidInput, match="cannot read"):
        read_raw(tmp_path)


def test_a_label_that_is_not_an_option_is_an_error_naming_the_record(tmp_path):
    """A typo'd label is not "unlabelled": exit 1 naming the line, the question and the options."""
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"state": "s", "questions": {"q": {"type": "noul", "label": "true"}}}) + "\n" +
                 json.dumps({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "x", "b": "y"},
                                                               "label": "zzz"}}, "_meta": {"id": "typo"}}) + "\n")
    with pytest.raises(InvalidInput, match=r"--data line 2 \(typo/q\): label 'zzz' is not one of its options: a, b"):
        load_data(p)
    p.write_text(json.dumps({"state": "s", "questions": {"q": {"type": "noul", "label": "maybe"}}}) + "\n")
    with pytest.raises(InvalidInput, match=r"--data line 1 \(line1/q\): label 'maybe' is not one of its options: "
                                           r"false, true"):
        load_data(p)


def test_a_null_label_is_unlabelled_like_a_missing_one(tmp_path):
    """JSON exporters write null for "no label", so `label: null` is skipped as unlabelled, as a
    missing key is; only a non-null label that is not an option is an error."""
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"state": "s", "questions": {"q": {"type": "noul", "label": None}}}) + "\n" +
                 json.dumps({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "x", "b": "y"},
                                                               "label": None},
                                                         "r": {"type": "noul", "label": "true"}}}) + "\n")
    items, skipped = load_data(p)
    assert [it.uid for it in items] == ["line2/r"] and skipped == Counter(records=2, unlabelled=2)


def test_jsonl_lines_split_on_newline_only(tmp_path):
    """JSON strings may hold a raw U+2028, U+2029 or U+0085; splitlines() would cut such a record."""
    odd = json.dumps({"state": "a\u2028b\u2029c\x85d", "questions": {"q": {"type": "noul", "label": "true"}}},
                     ensure_ascii=False)
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 2, extra=[odd]))
    assert [it.uid for it in items] == ["rec-0/q", "rec-1/q", "line3/q"] and items[-1].body["state"] == "a\u2028b\u2029c\x85d"
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"state": "a\u2028b", "questions": {"q": {"type": "noul", "label": "true"}}},
                            ensure_ascii=False) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(InvalidInput, match="--data line 2: not valid JSON"):
        load_data(p)
    lines = [ln.encode() for ln in write_data(tmp_path / "w.jsonl", 2).read_text().split("\n") if ln]
    p.write_bytes(b"\r\n".join([*lines, b"{not json"]) + b"\r\n")         # CRLF reads; its bad line gets no CR hint
    with pytest.raises(InvalidInput, match=r"--data line 3: not valid JSON") as e:
        load_data(p)
    assert "bare CR" not in str(e.value)
    p.write_bytes(b"\r".join(lines))                                    # lone-CR line endings are named
    with pytest.raises(InvalidInput, match="line 1: not valid JSON .*its lines end in a bare CR"):
        load_data(p)
    raw = tmp_path / "x.raw.jsonl"
    raw.write_text(json.dumps({"callo_think_raw": 1}) + "\n" +
                   json.dumps({"phase": "A", "q": "x\u2028y/q"}, ensure_ascii=False) + "\r\n", encoding="utf-8")
    r = read_raw(raw)
    assert [row["q"] for row in r.rows] == ["x\u2028y/q"] and not r.partial


def test_header_diff_refuses_any_difference_but_the_named_warn_only_and_ignored_keys():
    """ANY header difference refuses a resume except the host and the build; the
    per-startup `instance` and the format tag are ignored; a key added to the header later refuses by default."""
    settings = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 256))
    stats = {"model_key": "sha256:" + "a" * 64, "provenance": {"build_info": "b1", "chat_template_sha256": "t"},
             "host": {"fingerprint": "sha256:h", "chip": "M1"}, "instance": "i1", "model_file": "m.gguf"}
    h = raw_header(stats, settings, resolve_sampling(parse_sampling("fixed")))
    assert header_diff({"callo_think_raw": 1, **h}, {**h, "instance": "i2"}) == ([], [])
    assert header_diff(h, {**h, "host": {"fingerprint": "sha256:h", "chip": "M2"}}) == ([], [])
    assert header_diff(h, {**h, "model_file": "n.gguf"}) == (["model_file"], [])
    assert header_diff(h, {**h, "future": 1}) == (["future"], [])
    assert header_diff(h, {**h, "provenance": {**h["provenance"], "new_field": "x"}}) == (["provenance"], [])
    assert header_diff(h, {**h, "comparable": {**h["comparable"], "new_field": "x"}}) == (["comparable"], [])


def test_think_settings_checks_its_own_budgets():
    """bmax / positive assume what parse_budgets guarantees; a settings built elsewhere (the menu run)
    is held to it too."""
    for bad, why in (((256, 0), "ascending"), ((0, 256, 256), "ascending"), ((256, 512), "include 0"),
                     ((-1, 0), "from 0 to 32768"), ((0, 40000), "from 0 to 32768"), ((0, 1.5), "from 0 to 32768"),
                     ((0, True), "from 0 to 32768"), (tuple(range(9)), "at most 8"), ((), "include 0")):
        with pytest.raises(InvalidInput, match=why):
            ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, bad)
    assert ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0,)).positive == ()


# ---- the fit and the profile ----------------------------------------------------------------------------------


SAMPLING = resolve_sampling(parse_sampling("fixed"))
STATS = {"model_key": "sha256:" + "a" * 64, "model_file": "fake-9b.gguf", "model_bytes": 2 * 2**30, "mmproj_bytes": None,
         "provenance": {"prompt_sha256": "p" * 64, "chat_template_sha256": "t" * 64, "build_info": "b1-926862e",
                        "mmproj": None, "user_delimiter": "<|im_start|>user", "cache_prompt": False, "server_args": None},
         "router": {"reserved": [0]}, "kv": "unified", "n_ctx": 4096, "total_slots": 4, "cache_prompt": {"plain": False},
         "position_bias": None, "host": {"fingerprint": "sha256:" + "h" * 64, "chip": "fake"}, "instance": "i1",
         "user_delimiter": "<|im_start|>user",
         "think": {"capable": True, "reason": None, "close": "\n</think>\n\n", "synthesized_open": False}}
TRACE = {"gen_wall_ms": 3200.0, "predicted_ms": 3000.0, "frame_ms": 40.0, "gen_prompt_ms": 150.0, "gen_prompt_n": 60,
         "gen_cache_n": 0, "on_tokens": 60, "truncated": False}


def _probs(n, right, strength=0.8):
    p = [(1 - strength) / (n - 1)] * n
    p[right] = strength
    return p


def synthetic_raw(items, *, cut_items=4, settings=None, audit_err=0.02, drift_factor=1.0, therm=(100, 100),
                  gen_prompt_n=60):
    """Phase A and B rows for `items`: the first `cut_items` have a 300-token trace (cut at 64 and 256), the rest a
    40-token one (closed before 64, so 64 and 256 share a readout). Budget 0 is right on even items, 64 on all but
    item 1, 256 on all. Audit rows at 64 for the cut items with the given latency error; drift rows for 3 items.
    `gen_prompt_n` is every generation's prefill (2: the host prompt cache restored the question's state)."""
    settings = settings or ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 64, 256), audit=6)
    header = raw_header(STATS, settings, SAMPLING)
    gen = {"gen_prompt_n": gen_prompt_n, "gen_cache_n": 60 - gen_prompt_n, "on_tokens": 60}
    rows = []
    for i, it in enumerate(items):
        n = len(it.question.keys)
        wrong = (it.label + 1) % n
        rows.append({"phase": "A", "q": it.uid, "slot": 0, "probs": _probs(n, it.label if i % 2 == 0 else wrong, 0.6),
                     "ms": 250.0 + i, "miss": 0, "option_mass": 0.97, "prompt_n": 5})
        if i < cut_items:
            tokens = [5000 + (j % 7) for j in range(299)] + [2001]
            reads = {"64": {"probs": _probs(n, it.label if i != 1 else wrong), "k": 64, "read_ms": 30.0, "p_ms": 1.5 + 30.0,
                            "read_prompt_n": 67, "read_cache_n": 60, "text_fallback": False, "miss": 0, "option_mass": 0.99},
                     "256": {"probs": _probs(n, it.label, 0.95), "k": 256, "read_ms": 32.0, "p_ms": 1.5 + 96.0,
                             "read_prompt_n": 259, "read_cache_n": 60, "text_fallback": False, "miss": 0, "option_mass": 0.99},
                     "300": {"probs": _probs(n, it.label, 0.95), "k": 300, "read_ms": 25.0, "p_ms": 1.5, "read_prompt_n": 2,
                             "read_cache_n": 359, "text_fallback": False, "miss": 0, "option_mass": 0.99}}
            trace = {**TRACE, **gen, "tokens": tokens, "n_gen": 300, "close": "forced"}
        else:
            tokens = [5000 + (j % 5) for j in range(39)] + [2001]
            shared = {"probs": _probs(n, it.label, 0.9), "k": 40, "read_ms": 24.0, "p_ms": 1.0, "read_prompt_n": 2,
                      "read_cache_n": 99, "text_fallback": False, "miss": 0, "option_mass": 0.99}
            reads = {"64": dict(shared), "256": dict(shared)}
            trace = {**TRACE, **gen, "tokens": tokens, "n_gen": 40, "close": "natural", "gen_wall_ms": 600.0, "predicted_ms": 400.0}
        rows.append({"phase": "B", "q": it.uid, "trace": trace, "budgets": reads})
    for i, it in enumerate(items[:cut_items]):
        est = budget_latency(40.0, 3200.0, 3000.0, 300, 64, 30.0, 31.5, 1.5)
        rows.append({"phase": "audit", "q": it.uid, "budget": 64, "trace": {"tokens": [5000 + (j % 7) for j in range(64)],
                                                                            "n_gen": 64, "close": "forced", **gen},
                     "probs": _probs(len(it.question.keys), it.label if i != 1 else (it.label + 1) % len(it.question.keys)),
                     "lat_ms": est * (1 + audit_err)})
    for it in items[:3]:
        rows.append({"phase": "drift", "q": it.uid, "trace": {"n_gen": 300, "predicted_ms": 3000.0 * drift_factor}})
    raw = RawThink({"callo_think_raw": 1, **header}, rows, [{"index": 0, "failed": []}],
                   [{"at": a, "limit": t} for a, t in zip(("A", "end"), therm, strict=True)], False)
    return raw, settings


def test_outcomes_apply_the_latency_model_and_name_the_close(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    outs = outcomes(items, raw, [0, 64, 256])
    o = outs[items[0].uid]
    assert o[0].lat_ms == 250.0 and o[0].k == 0 and o[0].close is None and o[0].correct
    # the cut at 64: frame 40 + fixed 200 + decode 640 + (30 - 30 of re-prefill) = 880; at 256: 40 + 200 + 2560 + (32 - 96 -> 0)
    assert o[64].lat_ms == pytest.approx(880.0) and o[64].close == "forced" and o[64].k == 64
    assert o[256].lat_ms == pytest.approx(2800.0) and o[256].reprefill_ms == pytest.approx(96.0)
    assert o[64].reprefill_ms == pytest.approx(30.0)
    s = outs[items[5].uid]
    assert s[64].lat_ms == s[256].lat_ms == 40.0 + 600.0 + 24.0 and s[64].close == "natural" and s[64].k == 40
    assert not outs[items[1].uid][64].correct and outs[items[1].uid][256].correct


def test_build_profile_writes_the_6_1_format_that_serve_loads(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake",
                         started="2026-10-05T10:00:00Z", ended="2026-10-05T10:05:00Z")
    assert prof["callo_think_profile"] == 1 and prof["metrics"] == "callo-think-metrics-1"
    assert set(prof["budgets"]) == {"0", "64", "256"}
    b0, b64, b256 = prof["budgets"]["0"], prof["budgets"]["64"], prof["budgets"]["256"]
    assert (b0["n"], b0["acc"]) == (8, 0.5) and b64["acc"] == 0.875 and b256["acc"] == 1.0
    assert b64["dacc"][0] == pytest.approx(0.375) and b256["dacc"][0] == pytest.approx(0.5) and "dacc" not in b0
    assert b64["think_mean"] == (64 * 4 + 40 * 4) / 8 and b64["forced"] == 0.5 and b256["forced"] == 0.5
    assert b256["readout_reprefill"] == 4 and b64["readout_reprefill"] == 4 and b64["text_fallback"] == 0
    assert b0["lat_p50_ms"][0] == 253.0 and b0["lat_p50_ms"][1] <= 253.0 <= b0["lat_p50_ms"][2]   # nearest rank of 8
    assert len(b64["lat_p95_ms"]) == 3 and all(0 < b["T"] for b in (b0, b64, b256))
    assert b0["nll_T"] > 0 and 0 <= b0["ece_T"] <= 1 and "nll" in b64 and "ece" in b64
    assert b256["point"] == "fake@256" and b64["point"] == "fake@64" and b0["frontier"] is True
    assert prof["audit"]["budgets"]["64"] == {"n_cut": 4, "n": 4, "trace_identical": 4, "argmax_agree": 4,
                                              "dp_median": 0.0, "dp_max": 0.0,
                                              "lat_err_median": pytest.approx(0.02 / 1.02), "lat_err_max": pytest.approx(0.02 / 1.02),
                                              "latency_model": "verified", "prefill_reused": 0.0, "latency_reason": None}
    assert prof["audit"]["latency_model"] == "verified" and prof["audit"]["latency_reason"] is None
    assert b64["prefill_reused"] == 0.0 and b256["prefill_reused"] == 0.0 and "prefill_reused" not in b0
    # on mean (the default) 64 is faster than 256 (0.77 s against 1.73 s), so it stays on the frontier; on p50 both
    # read 0.66 s and 256 dominated it
    assert b64["frontier"] is True and b64["dacc_prev"] == pytest.approx(b64["dacc"]) and b64["dacc_best"][0] == -0.125
    assert b256["dacc_best"] is None and b256["dacc_prev"][0] == pytest.approx(0.125)       # its faster neighbour: 64
    assert b0["dacc_prev"] is None and b0["dacc_best"][0] == pytest.approx(-0.5)               # vs the most accurate, fake@256
    assert prof["drift"] == {"n": 3, "ratio": 1.0, "therm": [{"at": "A", "limit": 100}, {"at": "end", "limit": 100}],
                             "flagged": False}
    assert prof["calibration_overhead_s"] == pytest.approx(4 * (30.0 + 96.0) / 1000)
    assert prof["server_state"] == {"state": "clean", "checks": [{"index": 0, "failed": []}]}
    assert prof["thinking"] == {"capable": True, "reason": None, "close": "\n</think>\n\n", "synthesized_open": False}
    assert prof["comparable"]["readout"] == {"variant": "main", "top_k": 100, "max_depth": 8, "divergence_mass": 0.1,
                                             "callo_version": callo.__version__, "cache_prompt": True,     # the default
                                             "user_delimiter": "<|im_start|>user", "prefix": "tokens"}
    assert prof["comparable"]["sampling"]["policy"] == "fixed" and prof["comparable"]["data"]["questions"] == 8
    env = prof["environment"]
    assert env["model"] == {"file": "fake-9b.gguf", "bytes": 2 * 2**30, "mmproj": None, "mmproj_bytes": None}
    assert env["server"]["s1_slots"] == 1 and env["run"]["decode_tok_s_median"] == 100.0 and env["host"]["chip"] == "fake"
    assert env["server"]["slot"] == 0 and env["server"]["upstream_local"] is None and env["server"]["cache_ram"] is None
    assert env["server"]["launched_by"] == "url" and env["server"]["gpu_busy"] is False
    far = build_profile(items, raw, settings, sampling=SAMPLING, stats={**STATS, "upstream_local": False}, name="fake",
                        skipped=Counter(records=9, media=1))
    assert far["environment"]["server"]["upstream_local"] is False        # a remote llama-server: the host is callo's
    assert far["comparable"]["data"] == {"name": "d.jsonl", "sha256": "d" * 64, "questions": 8, "records": 9,
                                         "skipped": {"media": 1}, "strata": None}
    assert prof["comparable"]["data"]["records"] is None and prof["comparable"]["data"]["skipped"] == {}
    assert prof["row_fields"] == ["correct", "nll1", "pmax", "lat_ms", "k", "close"] and len(prof["rows"]) == 8
    row = prof["rows"][0]
    assert row["q"] == items[0].uid and row["cluster"] == "g0" and row["n"] == 3 and set(row["b"]) == {"0", "64", "256"}
    assert row["b"]["64"] == [True, pytest.approx(-math.log(0.8)), 0.8, pytest.approx(880.0), 64, "forced"]
    assert prof["pick"]["label"] == "fake@64" and prof["pick"]["rule"] == "fastest within noise of the best"
    assert prof["pick"]["best"] == "fake@256" and prof["pick"]["dacc_best"] == [-0.125, -0.375, 0.0]   # within noise
    assert prof["pick"]["latency"] == "mean" and prof["pick"]["preference"]["latency"] == "mean"
    tp = load_think_profile(prof)                                         # serve loads exactly this
    assert sorted(tp.budgets) == [0, 64, 256] and tp.capable and tp.temperature_for(64) == (b64["T"], True)
    assert tp.latency_model == "verified" and tp.position_bias_sha256 is None


def test_identical_budgets_merge_the_audit_and_drift_verdicts_and_a_budget_0_only_profile(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items, cut_items=0)                    # every trace closes before 64
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["64"]["point"] == prof["budgets"]["256"]["point"] == "fake@64≡256"
    assert prof["audit"]["budgets"]["64"]["latency_model"] == "audit n/a" and prof["audit"]["latency_model"] == "audit n/a"
    raw, settings = synthetic_raw(items, audit_err=0.3, drift_factor=1.2, therm=(100, 80))
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["audit"]["budgets"]["64"]["latency_model"] == "unverified" and prof["audit"]["latency_model"] == "unverified"
    assert prof["drift"]["ratio"] == pytest.approx(1.2) and prof["drift"]["flagged"] is True
    raw, settings = synthetic_raw(items, settings=ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 64, 256), audit=0))
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["audit"]["budgets"]["64"]["latency_model"] == "unaudited" and prof["audit"]["latency_model"] == "unaudited"
    raw.therm = []
    only0 = {**STATS, "think": {"capable": False, "reason": "not_preopened", "close": None, "synthesized_open": None}}
    raw.header = {**raw.header, "thinking": only0["think"]}            # the run's own record decides
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=only0, name="fake")
    assert set(prof["budgets"]) == {"0"} and prof["thinking"]["capable"] is False
    assert prof["drift"] == {"n": 3, "ratio": 1.0, "therm": None, "flagged": False}
    assert prof["audit"] == {"latency_model": "unaudited", "latency_reason": None, "budgets": {}}
    tp = load_think_profile(prof)
    assert not tp.capable and tp.temperature_for(256) == (1.0, False)


def test_the_audit_summary_reads_the_latest_audit_row_per_question_and_budget(tmp_path):
    """A raw file holding two audit rows for one (question, budget) (hand-merged; the run never writes one) counts the
    question once, at its latest row: `n`, the trace identity, the argmax agreement and |Δp| never double-count."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    first = next(r for r in raw.rows if r["phase"] == "audit" and r["q"] == items[0].uid)
    n = len(items[0].question.keys)
    raw.rows.append({**first, "probs": [0.1 if j == items[0].label else 0.9 / (n - 1) for j in range(n)]})   # disagrees
    a = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")["audit"]["budgets"]["64"]
    assert (a["n"], a["trace_identical"], a["argmax_agree"]) == (4, 4, 3) and a["dp_max"] > 0.5


def test_a_generation_that_reused_its_questions_cached_state_flags_the_latency_model(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items, gen_prompt_n=2)                  # the host prompt cache restored phase A's state
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["64"]["prefill_reused"] == 1.0 and prof["budgets"]["256"]["prefill_reused"] == 1.0
    a = prof["audit"]["budgets"]["64"]
    assert a["prefill_reused"] == 1.0 and a["latency_model"] == "unverified" and a["latency_reason"] == "prefill_reused"
    assert a["lat_err_median"] == pytest.approx(0.02 / 1.02)              # the error alone would have verified it
    assert prof["audit"]["latency_model"] == "unverified" and prof["audit"]["latency_reason"] == "prefill_reused"
    assert load_think_profile(prof).latency_model == "unverified"
    # gen_cache_n alone says the same when prompt_n is missing: 58 of 60 tokens came from the cache (phase A runs
    # before phase B, so each generation follows another record's row: readouts count as predecessors)
    raw.rows = [r for r in raw.rows if r["phase"] == "A"] + [r for r in raw.rows if r["phase"] != "A"]
    for r in raw.rows:
        if r["phase"] in ("B", "audit"):
            r["trace"].pop("gen_prompt_n")
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["64"]["prefill_reused"] == 1.0 and prof["audit"]["latency_reason"] == "prefill_reused"
    assert prefill_reused({"gen_prompt_n": 8}) is True and prefill_reused({"gen_prompt_n": 9}) is False
    assert prefill_reused({"gen_cache_n": 54, "on_tokens": 60}) is True and prefill_reused({"gen_cache_n": 53, "on_tokens": 60}) is False
    assert prefill_reused({}) is None and prefill_reused({"gen_cache_n": 5}) is None
    # a mixed run, phase B and audit together: 2 of 12 generations reused (17 %) is over the 10 % line; 1 of 12 is not
    raw, settings = synthetic_raw(items)
    raw.rows[3]["trace"]["gen_prompt_n"] = raw.rows[5]["trace"]["gen_prompt_n"] = 2   # items 1 and 2's phase-B rows
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["64"]["prefill_reused"] == 0.25 and prof["audit"]["budgets"]["64"]["latency_model"] == "unverified"
    assert prof["audit"]["budgets"]["64"]["prefill_reused"] == 0.0        # the audit rows themselves were clean
    raw.rows[3]["trace"]["gen_prompt_n"] = raw.rows[5]["trace"]["gen_prompt_n"] = 60
    raw.rows[-7]["trace"]["gen_prompt_n"] = 2                            # one audit row of 4, with 8 phase-B rows: 1/12
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["audit"]["budgets"]["64"]["prefill_reused"] == 0.25 and prof["audit"]["budgets"]["64"]["latency_model"] == "verified"


def test_a_partial_raw_file_fits_only_what_it_holds(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    raw.rows = [r for r in raw.rows if not (r["phase"] == "B" and r["q"] == items[7].uid)]
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["0"]["n"] == 8 and prof["budgets"]["64"]["n"] == 7 and len(prof["rows"]) == 8
    assert set(prof["rows"][7]["b"]) == {"0"}


# ---- the fit's strict JSON, sampling pins, strata and Pick.unsized ---------------------------------------------

def test_a_profile_round_trips_through_strict_json_with_a_parseable_policy_and_a_bool_capable(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    spec = resolve_sampling(parse_sampling("temp=0.7,top_k=20"), {"temp": float("nan"), "top_p": float("inf")})
    prof = build_profile(items, raw, settings, sampling=spec, stats=STATS, name="fake")
    with pytest.raises(ValueError):
        json.dumps(prof, allow_nan=False)             # the GGUF's nan / inf are recorded: the writer goes through json_safe
    path = tmp_path / "fake.think.json"
    path.write_text(json.dumps(json_safe(prof), allow_nan=False))             # the writer's exact form
    tp = load_think_profile(path)
    assert tp.sampling.policy == "temp=0.7,top_k=20" and tp.sampling.pinned["top_k"] == 20 and tp.capable is True
    assert tp.sampling.gguf == {"temp": "nan", "top_p": "inf"} and tp.latency_model == "verified"
    # stats with no think block (and a raw header from before it recorded one): a budget-0-only
    # profile, `capable` a bool (False), the pinned sampler still complete
    bare = {k: v for k, v in STATS.items() if k != "think"}
    raw.header = {k: v for k, v in raw.header.items() if k != "thinking"}
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=bare, name="fake")
    assert set(prof["budgets"]) == {"0"} and prof["thinking"]["capable"] is False
    assert prof["comparable"]["sampling"]["policy"] == "fixed" and check_pinned(prof["comparable"]["sampling"]["pinned"]) is None
    path.write_text(json.dumps(json_safe(prof), allow_nan=False))
    tp = load_think_profile(path)
    assert tp.capable is False and sorted(tp.budgets) == [0] and tp.sampling.pinned == SAMPLING.pinned


def test_a_mixed_probe_whose_budget_0_readout_warmed_the_cache_is_never_a_latency_source(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    b = next(r for r in raw.rows if r["phase"] == "B" and r["q"] == items[0].uid)
    b["budgets"]["0"] = {"probs": _probs(3, items[0].label), "ms": 250.0, "miss": 0, "option_mass": 0.97}
    outs = outcomes(items, raw, [0, 64, 256])
    assert set(outs[items[0].uid]) == {0} and outs[items[0].uid][0].lat_ms == 250.0     # phase A's budget 0 only
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["0"]["n"] == 8 and prof["budgets"]["64"]["n"] == 7 and set(prof["rows"][0]["b"]) == {"0"}
    # a clean phase-B row of the same question earlier in the file is still its measure (a resumed run, say)
    raw.rows.insert(raw.rows.index(b), {**b, "budgets": {k: v for k, v in b["budgets"].items() if k != "0"}})
    assert set(outcomes(items, raw, [0, 64, 256])[items[0].uid]) == {0, 64, 256}


def test_a_single_option_question_with_a_null_trace_is_read_at_every_budget(tmp_path):
    one = json.dumps({"state": "s", "questions": {"q": {"type": "choice", "instructions": "only", "criteria": {"o0": "x"},
                                                        "label": "o0"}}, "_meta": {"id": "rec-8", "group_id": "g4"}})
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8, extra=[one]))
    assert len(items) == 9 and len(items[8].question.keys) == 1
    raw, settings = synthetic_raw(items[:8])
    single = {"probs": [1.0], "k": 0, "read_ms": 20.0, "p_ms": None, "read_prompt_n": None, "read_cache_n": None,
              "text_fallback": False, "miss": 0, "option_mass": 1.0}
    raw.rows += [{"phase": "A", "q": items[8].uid, "slot": 0, "probs": [1.0], "ms": 21.0, "miss": 0, "option_mass": 1.0},
                 {"phase": "B", "q": items[8].uid, "trace": None, "budgets": {"64": dict(single), "256": dict(single)}}]
    o = outcomes(items, raw, [0, 64, 256])[items[8].uid]
    assert o[64].lat_ms == o[256].lat_ms == 20.0 and o[64].k == 0 and o[64].close is None and o[64].correct
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["64"]["n"] == 9 and prof["budgets"]["64"]["prefill_reused"] == 0.0
    assert prof["rows"][8]["b"]["64"] == [True, 0.0, 1.0, 20.0, 0, None]
    assert prof["audit"]["latency_model"] == "verified" and prof["drift"]["n"] == 3


def _with_sources(path: Path) -> Path:
    """The data file with `_meta.source` set per cluster (two sources), as the transfer set's and the bias bench's data carry it."""
    lines = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        rec["_meta"]["source"] = f"s{int(rec['_meta']['group_id'][1:]) % 2}"
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_every_paired_interval_is_stratified_by_source_over_one_fixed_row_set(tmp_path, monkeypatch):
    calls = []
    real_paired, real_stats = ct.paired_bootstrap, ct.latency_stats

    def paired(a, b, clusters, *, seed, strata=None, **kw):
        calls.append(("paired", list(clusters), None if strata is None else list(strata)))
        return real_paired(a, b, clusters, seed=seed, strata=strata, **kw)

    def stats(values, clusters, *, seed, strata=None):                   # the five statistics per budget
        calls.append(("stats", list(clusters), None if strata is None else list(strata)))
        return real_stats(values, clusters, seed=seed, strata=strata)

    monkeypatch.setattr(ct, "paired_bootstrap", paired)
    monkeypatch.setattr("callo.think.paired_bootstrap", paired)          # `Paired`'s Δacc (the marks and the pick)
    monkeypatch.setattr(ct, "latency_stats", stats)
    items, _ = load_data(_with_sources(write_data(tmp_path / "d.jsonl", 8)))
    assert [it.source for it in items] == ["s0", "s0", "s1", "s1", "s0", "s0", "s1", "s1"]
    raw, settings = synthetic_raw(items)
    raw.rows = [r for r in raw.rows if not (r["phase"] == "B" and r["q"] == items[7].uid)]   # a partial run
    build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    paired_calls = [c for c in calls if c[0] == "paired"]
    assert len(paired_calls) >= 4 and all(c[2] is not None for c in calls)
    # the pairing contract: every Δacc / ΔNLL interval, at every budget, resamples ONE row set in ONE order
    fixed = [it.cluster for it in items[:7]]
    assert all(c[1] == fixed and c[2] == [it.source for it in items[:7]] for c in paired_calls)
    calls.clear()
    items, _ = load_data(write_data(tmp_path / "plain.jsonl", 8))                # no source: plain clusters
    assert {it.source for it in items} == {None}
    raw, settings = synthetic_raw(items)
    build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert calls and all(c[2] is None for c in calls)


def test_every_profile_row_carries_the_stratum_calibrate_resampled_it_in(tmp_path):
    """Each row records its `source` (the stratum calibrate's bootstrap used, null when the data has
    none), so compare can re-run the same stratified pairing from the profile alone: a round trip through
    strict JSON reproduces calibrate's own Δacc intervals."""
    items, _ = load_data(_with_sources(write_data(tmp_path / "d.jsonl", 8)))
    raw, settings = synthetic_raw(items)
    path = tmp_path / "fake.think.json"
    path.write_text(json.dumps(json_safe(build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")),
                               allow_nan=False))
    prof = _strict(path)
    assert prof["comparable"]["data"]["strata"] == "source"
    assert [(r["q"], r["source"]) for r in prof["rows"]] == [(it.uid, it.source) for it in items]
    rows = prof["rows"]
    for b in ("64", "256"):
        again = ct.paired_bootstrap([float(r["b"]["0"][0]) for r in rows], [float(r["b"][b][0]) for r in rows],
                                    [r["cluster"] for r in rows], seed=prof["comparable"]["seed"],
                                    strata=[r["source"] for r in rows])
        assert list(again) == prof["budgets"][b]["dacc"]
    items, _ = load_data(write_data(tmp_path / "plain.jsonl", 8))                # no source: null, plain clusters
    raw, settings = synthetic_raw(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["comparable"]["data"]["strata"] is None and all(r["source"] is None for r in prof["rows"])
    some = [dataclasses.replace(it, source="s0") for it in items[:2]]      # one sourced cluster: the rest "unknown"
    prof = build_profile([*some, *items[2:]], raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert [r["source"] for r in prof["rows"]] == ["s0", "s0"] + ["unknown"] * 6


def test_the_pick_block_names_points_of_unknown_size(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["pick"]["unsized"] == [] and pick_note(prof["pick"]) is None   # calibrate has no --max-memory
    assert pick_note({"unsized": ["big@0", "big@256≡512"]}) == \
        "size unknown: excluded by --max-memory: big@0, big@256≡512"


def test_the_prefill_guard_flags_the_latency_model_without_an_audit_verdict(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    flagged = {"latency_model": "unverified", "latency_reason": "prefill_reused"}
    for kw, model in (({"settings": ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 64, 256), audit=0)}, "unaudited"),
                      ({"cut_items": 0}, "audit n/a")):
        raw, settings = synthetic_raw(items, **kw)
        prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
        assert prof["audit"]["budgets"]["64"]["latency_model"] == model and prof["audit"]["latency_model"] == model
        assert prof["audit"]["latency_reason"] is None and prof["audit"]["budgets"]["64"]["latency_reason"] is None
        assert prof["budgets"]["64"]["prefill_reused"] == 0.0 and prof["audit"]["budgets"]["64"]["prefill_reused"] == \
            (0.0 if model == "unaudited" else None)                      # n/a: no cut, so no audit generation
        raw, settings = synthetic_raw(items, gen_prompt_n=2, **kw)        # the same run, flattered by the host cache
        prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
        assert {k: prof["audit"]["budgets"]["64"][k] for k in flagged} == flagged
        assert {k: prof["audit"][k] for k in flagged} == flagged and prof["budgets"]["64"]["prefill_reused"] == 1.0
        assert load_think_profile(prof).latency_model == "unverified"
    # a single positive budget: no audit entries, so phase B's share alone flags the profile
    only = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, (0, 256))
    raw, settings = synthetic_raw(items, settings=only)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["audit"] == {"latency_model": "unaudited", "latency_reason": None, "budgets": {}}
    raw, settings = synthetic_raw(items, settings=only, gen_prompt_n=2)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["audit"] == {**flagged, "budgets": {}} and prof["budgets"]["256"]["prefill_reused"] == 1.0
    # phase B is every budget's latency source: 1 of 8 phase-B generations (12.5 %) flags the audited budget too,
    # although pooled with its 4 clean audit generations it is 1 of 12
    raw, settings = synthetic_raw(items)
    raw.rows[3]["trace"]["gen_prompt_n"] = 2
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert {k: prof["audit"]["budgets"]["64"][k] for k in flagged} == flagged and prof["audit"]["budgets"]["64"]["prefill_reused"] == 0.0
    assert {k: prof["audit"][k] for k in flagged} == flagged and prof["budgets"]["64"]["prefill_reused"] == 0.125


def write_records(path: Path, records: int, per: int) -> Path:
    """`records` records of `per` labelled questions each, sharing the record's long state: few records, many
    questions, where neighbouring questions in run order often come from one record."""
    lines = []
    for r in range(records):
        qs = {f"q{j}": {"type": "choice", "instructions": f"pick {j}", "criteria": {"o0": "a", "o1": "b", "o2": "c"},
                        "label": "o0"} for j in range(per)}
        lines.append(json.dumps({"state": f"record {r}: " + "a long shared state " * 20, "questions": qs,
                                 "_meta": {"id": f"rec-{r}"}}))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_same_record_neighbour_s_shared_prefix_is_not_a_host_cache_hit(tmp_path):
    items, _ = load_data(write_records(tmp_path / "r.jsonl", 2, 4))
    assert [it.uid for it in items][:5] == ["rec-0/q0", "rec-0/q1", "rec-0/q2", "rec-0/q3", "rec-1/q0"]
    raw, settings = synthetic_raw(items)
    raw.rows = [r for r in raw.rows if r["phase"] == "A"] + [r for r in raw.rows if r["phase"] != "A"]   # A, then B
    gens = [r for r in raw.rows if isinstance(r.get("trace"), dict) and r["phase"] in ("B", "audit", "drift")]
    prev = None
    for r in raw.rows:                 # the slot's LCP: high after a question of the same record, cold otherwise
        same = prev is not None and prev["q"].split("/")[0] == r["q"].split("/")[0]
        if isinstance(r.get("trace"), dict) and r["phase"] in ("B", "audit", "drift"):
            r["trace"].pop("gen_prompt_n", None)
            if "gen_cache_n" in r["trace"]:
                r["trace"]["gen_cache_n"] = 58 if same else 0
        prev = r
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["64"]["prefill_reused"] == 0.0 and prof["audit"]["budgets"]["64"]["prefill_reused"] == 0.0
    assert prof["audit"]["latency_model"] == "verified" and prof["audit"]["latency_reason"] is None
    # a cross-record generation that restored its own question's state is still caught: 1 of the 2 judged
    next(r for r in gens if r["phase"] == "B" and r["q"] == "rec-1/q0")["trace"]["gen_cache_n"] = 58
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["64"]["prefill_reused"] == 0.5
    assert prof["audit"]["latency_model"] == "unverified" and prof["audit"]["latency_reason"] == "prefill_reused"
    # the rule: a same-record predecessor leaves the 0.9 arm unjudged; the <= 8 arm judges regardless
    assert prefill_reused({"gen_cache_n": 58, "on_tokens": 60}, same_record=True) is None
    assert prefill_reused({"gen_cache_n": 58, "on_tokens": 60}, same_record=False) is True
    assert prefill_reused({"gen_prompt_n": 2, "gen_cache_n": 58, "on_tokens": 60}, same_record=True) is True
    assert prefill_reused({"gen_prompt_n": 30, "gen_cache_n": 270, "on_tokens": 300}, same_record=True) is False


def test_the_first_generation_s_predecessor_is_the_last_phase_a_readout(tmp_path):
    """Phase A's readouts leave their record's prefix on the slot too, so a first phase-B
    generation of the same record as the last phase-A readout is not judged by the 0.9 arm."""
    items, _ = load_data(write_records(tmp_path / "r.jsonl", 2, 4))
    raw, settings = synthetic_raw(items)
    a_rows = [r for r in raw.rows if r["phase"] == "A"]
    a_rows.append(a_rows.pop(1))                                       # rec-0/q1 is the last phase-A readout
    raw.rows = a_rows + [r for r in raw.rows if r["phase"] != "A"]
    for r in raw.rows:
        if isinstance(r.get("trace"), dict) and "gen_cache_n" in r["trace"]:
            r["trace"].pop("gen_prompt_n", None)
            r["trace"]["gen_cache_n"] = 0
    first = next(r for r in raw.rows if r["phase"] == "B")
    assert first["q"] == "rec-0/q0"
    first["trace"]["gen_cache_n"] = 58                                  # rec-0's shared state and context, legitimately
    assert id(first) in ct._same_record(raw, items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["64"]["prefill_reused"] == 0.0               # rec-1/q0, the one cross-record B row: clean


def test_a_cluster_that_spans_two_sources_is_refused_and_the_profile_names_its_scheme(tmp_path):
    path = _with_sources(write_data(tmp_path / "d.jsonl", 8))
    items, _ = load_data(path)
    raw, settings = synthetic_raw(items)
    assert build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")["comparable"]["data"]["strata"] == "source"
    plain, _ = load_data(write_data(tmp_path / "plain.jsonl", 8))
    raw, settings = synthetic_raw(plain)
    assert build_profile(plain, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")["comparable"]["data"]["strata"] is None
    lines = path.read_text().splitlines()
    rec = json.loads(lines[7])                                           # rec-7, group g3, source s1
    rec["_meta"]["source"] = "s0"
    lines[7] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(InvalidInput, match="cluster g3 spans sources s1 and s0"):
        load_data(path)


def test_the_fit_s_smaller_edges(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    # the audit's n counts the regenerations its verdict rests on: one whose phase-B row is missing is not compared
    raw, settings = synthetic_raw(items)
    raw.rows = [r for r in raw.rows if not (r["phase"] == "B" and r["q"] == items[0].uid)]
    a = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")["audit"]["budgets"]["64"]
    assert (a["n_cut"], a["n"], a["trace_identical"], a["latency_model"]) == (3, 3, 3, "verified")
    # a bool is no thermal reading
    raw, settings = synthetic_raw(items)
    raw.therm = [{"at": "A", "limit": True}]
    assert build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")["drift"]["flagged"] is False
    # no question measured at every budget: no paired interval is invented
    raw, settings = synthetic_raw(items)
    raw.rows = [r for r in raw.rows if not ((r["phase"] == "A" and r["q"] in {it.uid for it in items[:4]}) or
                                            (r["phase"] == "B" and r["q"] in {it.uid for it in items[4:]}))]
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert prof["budgets"]["0"]["n"] == 4 and prof["budgets"]["64"]["n"] == 4 and "dacc" not in prof["budgets"]["64"]
    assert all(e["dacc_prev"] is None and e["dacc_best"] is None for e in prof["budgets"].values())
    assert prof["pick"]["label"] is not None and prof["pick"]["dacc_best"] is None
    # provenance is the raw header's (the run's first session), not a resumed session's live build
    raw, settings = synthetic_raw(items)
    later = {**STATS, "provenance": {**STATS["provenance"], "build_info": "b2-newer"}}
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=later, name="fake")
    assert prof["provenance"]["build_info"] == "b1-926862e" and prof["environment"]["build_info"] == "b1-926862e"


def ladder_raw(items, budgets, *, n_gen=1100, cut_items=4):
    """Phase A, B and audit rows for ANY ladder: the first `cut_items` questions think `n_gen` tokens (cut at every
    budget below it, each audited), the rest close at 40 (one readout shared by every budget)."""
    settings = ThinkSettings("d.jsonl", "d.jsonl", "d" * 64, tuple(budgets), audit=6)
    gen = {"gen_prompt_n": 60, "gen_cache_n": 0, "on_tokens": 60}
    positive = [b for b in budgets if b > 0]
    rows = []
    for i, it in enumerate(items):
        n = len(it.question.keys)
        rows.append({"phase": "A", "q": it.uid, "slot": 0, "probs": _probs(n, it.label, 0.6), "ms": 250.0,
                     "miss": 0, "option_mass": 0.97, "prompt_n": 5})
        g = n_gen if i < cut_items else 40
        reads = {str(b): {"probs": _probs(n, it.label, 0.9), "k": min(b, g), "read_ms": 30.0,
                          "p_ms": 1.5 + (30.0 if b < g else 0.0), "read_prompt_n": 67 if b < g else 2, "read_cache_n": 60,
                          "text_fallback": False, "miss": 0, "option_mass": 0.99} for b in positive}
        trace = {**TRACE, **gen, "tokens": [5000 + j % 7 for j in range(g)], "n_gen": g,
                 "close": "forced" if g > max(positive) else "natural"}
        rows.append({"phase": "B", "q": it.uid, "trace": trace, "budgets": reads})
    for b in positive[:-1]:
        for it in items[:cut_items]:
            est = budget_latency(40.0, 3200.0, 3000.0, n_gen, b, 30.0, 31.5, 1.5)
            rows.append({"phase": "audit", "q": it.uid, "budget": b, "lat_ms": est * 1.02,
                         "probs": _probs(len(it.question.keys), it.label, 0.9),
                         "trace": {"tokens": [5000 + j % 7 for j in range(b)], "n_gen": b, "close": "forced", **gen}})
    raw = RawThink({"callo_think_raw": 1, **raw_header(STATS, settings, SAMPLING)}, rows, [{"index": 0, "failed": []}],
                   [], False)
    return raw, settings


def test_the_default_ladder_audits_two_budgets_below_bmax(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = ladder_raw(items, DEFAULT_BUDGETS)                  # 0,256,512,1024: 256 and 512 are audited
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert set(prof["budgets"]) == {"0", "256", "512", "1024"} and set(prof["audit"]["budgets"]) == {"256", "512"}
    for b in ("256", "512"):
        a = prof["audit"]["budgets"][b]
        assert (a["n_cut"], a["n"], a["trace_identical"], a["latency_model"], a["prefill_reused"]) == \
            (4, 4, 4, "verified", 0.0)
    assert prof["audit"]["latency_model"] == "verified" and load_think_profile(prof).latency_model == "verified"
    # the most audited budgets a ladder can hold (8 budgets: 6 below Bmax), flattered by the host cache
    raw, settings = ladder_raw(items, (0, 64, 128, 256, 384, 512, 768, 1024))
    for r in raw.rows:
        if r["phase"] in ("B", "audit"):
            r["trace"]["gen_prompt_n"] = 2
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert len(prof["audit"]["budgets"]) == 6
    assert all(a["latency_reason"] == "prefill_reused" for a in prof["audit"]["budgets"].values())


# ---- the run against an in-process callo serve on the thinking fake --------------------------------------------

def _len(text: str) -> int:
    """The fake's natural trace length: questions 1, 3, 10 and 11 close at 40 tokens, the others at 300."""
    return 40 if any(f"state {i}" in text for i in (1, 3, 10, 11)) else 300


def _settings(data: Path, **over) -> ThinkSettings:
    base = {"budgets": (0, 64, 256), "warmup": 1, "audit": 3}
    base.update(over)
    return ThinkSettings(str(data), data.name, data_sha256(data), **base)


class _Served:
    def __init__(self, tmp_path, think="preopened", *, think_len=_len, serve_args=(), **fake):
        self.up = FakeUpstream(think=think, think_len=think_len, slot_save=True, **fake).start()   # --slot-save-path
        self.up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
        self.r = Running(["serve", "--upstream", self.up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"),
                          *serve_args])
        assert self.r.port, "callo did not start"
        del self.up.requests[:]                                          # drop callo's own startup probe
        self.url = self.r.url
        self.log: list[str] = []

    def stop(self):
        self.r.finish()
        self.up.stop()

    def gens(self, n_predict=None):
        """The generations the fake saw: a completion that asks for no probabilities (a budget of 1 is a
        generation too, so never n_predict > 1)."""
        return [q.json for q in self.up.requests if q.path == "/completion" and not q.json.get("n_probs")
                and (n_predict is None or q.json["n_predict"] == n_predict)]


@pytest.fixture
def served(tmp_path):
    s = _Served(tmp_path)
    yield s
    s.stop()


def _strict(path: Path) -> dict:
    """The profile as strict JSON: a NaN or Infinity literal is an error, never a float."""
    def refuse(c):
        raise ValueError(f"non-strict JSON constant {c}")
    return json.loads(path.read_text(), parse_constant=refuse)


GEN_FIELDS = ("on_tokens", "gen_prompt_n", "gen_cache_n", "gen_wall_ms", "frame_ms", "predicted_ms")


def test_a_full_run_recovers_the_planted_curve_and_writes_the_profile_and_the_raw_file(tmp_path, served):
    data = write_data(tmp_path / "d.jsonl", 12)
    out = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append, therm=lambda: 100)
    assert isinstance(out, Outcome) and out.status == "done" and out.reason is None
    prof = _strict(out.profile)
    tp = load_think_profile(out.profile)
    assert tp.capable and sorted(tp.budgets) == [0, 64, 256] and tp.model_key == prof["model_key"]
    b = prof["budgets"]
    assert b["0"]["n"] == b["64"]["n"] == b["256"]["n"] == 12 and len(prof["rows"]) == 12
    assert b["0"]["acc"] == 0.0 and b["64"]["acc"] == b["256"]["acc"] == 11 / 12     # one natural closer stays wrong
    assert b["256"]["nll"] < b["64"]["nll"] < b["0"]["nll"] and b["256"]["dacc"][0] == pytest.approx(11 / 12)
    assert b["256"]["think_mean"] == (8 * 256 + 4 * 41) / 12 and b["64"]["think_mean"] == (8 * 64 + 4 * 41) / 12
    assert b["256"]["forced"] == 8 / 12 and b["64"]["forced"] == 8 / 12 and b["64"]["eos"] == 0.0
    assert b["256"]["lat_p50_ms"][0] > b["64"]["lat_p50_ms"][0] > b["0"]["lat_p50_ms"][0]
    assert prof["audit"]["budgets"]["64"]["n_cut"] == 8 and prof["audit"]["budgets"]["64"]["n"] == 3
    assert prof["audit"]["budgets"]["64"]["trace_identical"] == 3 and prof["audit"]["latency_model"] == "verified"
    assert prof["audit"]["budgets"]["64"]["argmax_agree"] == 3 and prof["audit"]["budgets"]["64"]["dp_max"] == 0.0
    assert b["64"]["prefill_reused"] == 0.0 and prof["audit"]["budgets"]["64"]["prefill_reused"] == 0.0
    assert prof["environment"]["server"]["upstream_local"] is True and prof["environment"]["server"]["slot"] == 0
    assert prof["environment"]["server"]["cache_ram"] is None and prof["environment"]["server"]["server_args"] is None
    assert all(r["trace"]["on_tokens"] > 0 for r in read_raw(tmp_path / "think" / "fake.raw.jsonl").rows
               if r["phase"] in ("B", "audit", "drift"))
    assert prof["drift"]["n"] == 3 and prof["drift"]["ratio"] == pytest.approx(1.0) and prof["drift"]["flagged"] is False
    assert [t["at"] for t in prof["drift"]["therm"]] == ["start", "A", "B", "end"]
    assert prof["server_state"]["state"] == "clean" and len(prof["server_state"]["checks"]) >= 5
    assert prof["comparable"]["data"]["skipped"] == {} and prof["comparable"]["data"]["records"] == 12
    assert prof["comparable"]["sampling"]["policy"] == "model" and prof["environment"]["model"]["file"] == "m.gguf"
    assert "gguf_error" not in prof["comparable"]["sampling"]                  # the GGUF was read
    assert prof["thinking"]["capable"] is True and prof["environment"]["run"]["decode_tok_s_median"] == 100.0
    assert prof["pick"]["rule"] == "fastest within noise of the best"
    raw = read_raw(tmp_path / "think" / "fake.raw.jsonl")
    phases = Counter(r["phase"] for r in raw.rows)
    assert phases == {"warmup": 1, "A": 12, "B": 12, "audit": 3, "drift": 3}
    assert all(len(r["trace"]["tokens"]) == r["trace"]["n_gen"] for r in raw.rows if r["phase"] in ("B", "audit", "drift"))
    assert len(served.gens(256)) == 1 + 12 + 3 and len(served.gens(64)) == 3        # warm-up, phase B, drift; audit
    assert any(line.startswith("plan: 12 questions") for line in served.log)
    assert any("measured" in line and "per question" in line for line in served.log)
    assert not any("--think-max" in line for line in served.log)              # 256 is within the server's 4096
    # every generation row keeps what prefill_reused and the latency model read, as ints
    for r in raw.rows:
        if r["phase"] in ("B", "audit", "drift"):
            assert all(type(r["trace"][k]) in (int, float) for k in GEN_FIELDS), r["trace"]
            assert type(r["trace"]["on_tokens"]) is int and r["trace"]["gen_cache_n"] == 0
            assert r["trace"]["gen_prompt_n"] == r["trace"]["on_tokens"]                  # no cache: all prefilled
    # an audit row's latency is the real generation's: frame + gen_wall + read_ms (at k >= n_gen; the fake plants
    # a predicted_ms above the measured wall, which budget_latency counts instead, think.py's continuity rule)
    for r in (r for r in raw.rows if r["phase"] == "audit"):
        t = r["trace"]
        assert r["k"] == r["budget"] == t["n_gen"] == 64 and t["predicted_ms"] == 640.0 and r["read_ms"] > 0
        assert r["lat_ms"] == pytest.approx(t["frame_ms"] + max(t["gen_wall_ms"], t["predicted_ms"]) + r["read_ms"])


def test_a_model_without_a_thinking_mode_is_measured_at_budget_0_only(tmp_path):
    s = _Served(tmp_path, think=None)
    try:
        data = write_data(tmp_path / "d.jsonl", 6)
        out = calibrate_one(_settings(data), url=s.url, name="plain", out_dir=tmp_path / "think", resume=False,
                            log=s.log.append, therm=lambda: None)
        assert out.status == "done"
        prof = json.loads(out.profile.read_text())
        assert set(prof["budgets"]) == {"0"} and prof["thinking"] == {"capable": False, "reason": "not_preopened",
                                                                       "close": None, "synthesized_open": None}
        assert prof["audit"] == {"latency_model": "unaudited", "latency_reason": None, "budgets": {}}
        assert prof["drift"]["ratio"] is None
        assert prof["drift"]["therm"] == [{"at": "start", "limit": None, "session": 0},
                                          {"at": "A", "limit": None, "session": 0}]
        assert prof["drift"]["flagged"] is False                    # no thermal data is never evidence
        assert not s.gens() and "NO THINKING MODE (not_preopened): measured budget 0 only" in s.log
        assert load_think_profile(out.profile).temperature_for(256) == (1.0, False)
        assert not_capable_line({"reason": "unsupported_think_format"}) == \
            "THINKS IN A FORMAT CALLO CANNOT READ: measured budget 0 only"
    finally:
        s.stop()


def test_preflight_refuses_a_budget_that_does_not_fit_the_slot_context_before_phase_a(tmp_path):
    up = FakeUpstream(think="preopened", think_len=_len, slot_save=True)
    up.props["default_generation_settings"]["n_ctx"] = 200
    s = _Served.__new__(_Served)
    s.up = up.start()
    up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
    s.r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
    s.url, s.log = s.r.url, []
    del up.requests[:]
    try:
        data = write_data(tmp_path / "d.jsonl", 4)
        out = calibrate_one(_settings(data), url=s.url, name="fake", out_dir=tmp_path / "think", resume=False,
                            log=s.log.append, therm=lambda: 100)
        assert out.status == "refused" and "raise --ctx or lower the budgets" in out.reason
        assert "tokens at budget 256" in out.reason and not s.gens()
        assert not (tmp_path / "think" / "fake.raw.jsonl").exists()   # refused before the raw file: no --resume needed
    finally:
        s.stop()


def test_a_budget_with_fewer_than_3_cut_questions_is_not_audited(tmp_path, served):
    data = write_data(tmp_path / "d.jsonl", 4)                          # states 1 and 3 close at 40: 2 cuts at 64,
    out = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append, therm=lambda: 100)                 # rec-0 warms up: its pool holds 1
    prof = json.loads(out.profile.read_text())
    assert prof["audit"]["budgets"]["64"] == {"n_cut": 1, "n": 0, "latency_model": "audit n/a", "prefill_reused": None,
                                              "latency_reason": None}
    assert prof["audit"]["latency_model"] == "audit n/a" and not served.gens(64)      # nothing was regenerated at 64
    assert "audit" not in Counter(r["phase"] for r in read_raw(tmp_path / "think" / "fake.raw.jsonl").rows)


def test_the_audit_counts_its_cut_questions_from_the_pool_it_draws_from_so_a_warm_up_cut_is_audit_n_a(tmp_path):
    """`n_cut` and the regenerated questions come from ONE pool (`audit_pool`: the measured,
    non-warm-up questions cut at B). Three questions are cut at 64, one of them the warm-up's: the
    pool holds 2, so nothing is regenerated and the budget reads `audit n/a`, never `unverified (lat_err)` on 0."""
    s = _Served(tmp_path, think_len=lambda text: 300 if any(f"state {i}" in text for i in (0, 1, 2)) else 40)
    try:
        data = write_data(tmp_path / "d.jsonl", 6)                     # run order rec-1, rec-4, rec-0, ...: rec-1 warms
        items, _ = load_data(data)
        order = shuffled(items, 0)
        assert order[0].item_id == "rec-1"
        out = calibrate_one(_settings(data), url=s.url, name="fake", out_dir=tmp_path / "think", resume=False,
                            log=s.log.append, therm=lambda: 100)
        assert out.status == "done", out.reason
        prof = json.loads(out.profile.read_text())
        assert prof["audit"]["budgets"]["64"] == {"n_cut": 2, "n": 0, "latency_model": "audit n/a", "prefill_reused": None,
                                                  "latency_reason": None}
        assert prof["audit"]["latency_model"] == "audit n/a" and not s.gens(64)
        b_rows = ct._latest(read_raw(tmp_path / "think" / "fake.raw.jsonl").rows, "B")
        assert [it.item_id for it in ct.audit_pool(order, b_rows, 1, 64)] == ["rec-0", "rec-2"]
    finally:
        s.stop()


def test_a_host_prompt_cache_hit_on_the_generation_is_measured_and_flags_the_latency_model(tmp_path, served):
    served.up.think_gen_prompt_n = 2                                     # every generation: the question's state restored
    data = write_data(tmp_path / "d.jsonl", 8)
    out = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append, therm=lambda: 100)
    prof = json.loads(out.profile.read_text())
    assert prof["budgets"]["64"]["prefill_reused"] == 1.0 and prof["budgets"]["256"]["prefill_reused"] == 1.0
    a = prof["audit"]["budgets"]["64"]
    assert a["n"] == 3 and a["prefill_reused"] == 1.0 and a["latency_model"] == "unverified"
    assert a["latency_reason"] == "prefill_reused" and a["lat_err_median"] < 0.1     # the error alone was fine
    assert prof["audit"]["latency_model"] == "unverified" and prof["audit"]["latency_reason"] == "prefill_reused"
    rows = read_raw(tmp_path / "think" / "fake.raw.jsonl").rows
    assert all(r["trace"]["gen_prompt_n"] == 2 and r["trace"]["gen_cache_n"] == r["trace"]["on_tokens"] - 2
               for r in rows if r["phase"] in ("B", "audit", "drift"))


def test_the_guard_refuses_a_busy_server_unless_allow_busy_marks_the_profile(tmp_path, served):
    data = write_data(tmp_path / "d.jsonl", 4)
    served.up.set_slot(2, is_processing=True, n_prompt_tokens=300)
    out = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append, therm=lambda: 100)
    assert out.status == "refused" and "foreign_processing" in out.reason and not served.gens()
    assert not (tmp_path / "think" / "fake.raw.jsonl").exists()        # refused at check 0: nothing to --resume
    out = calibrate_one(_settings(data, allow_busy=True), url=served.url, name="busy", out_dir=tmp_path / "think",
                        resume=False, log=served.log.append, therm=lambda: 100)
    assert out.status == "done"
    prof = json.loads(out.profile.read_text())
    assert prof["server_state"]["state"] == "busy" and all(c["failed"] for c in prof["server_state"]["checks"])
    assert load_think_profile(out.profile).state == "busy"
    assert any("--allow-busy: carrying on" in line for line in served.log)


def test_the_preconditions_and_the_cache_match(tmp_path, served):
    data = write_data(tmp_path / "d.jsonl", 4)
    out = calibrate_one(_settings(data, cache=False), url=served.url, name="fake", out_dir=tmp_path / "think",
                        resume=False, log=served.log.append, therm=lambda: 100)
    assert out.status == "refused" and "the server's --cache is on and this run's is off" in out.reason
    s = _settings(data)
    with pytest.raises(Refusal, match="predates think budgets"):
        check_preconditions({"model_key": "k", "temperature": 1.0}, "http://x", s)
    with pytest.raises(Refusal, match="model_key"):
        check_preconditions({"think": {}, "model_key": None, "temperature": 1.0}, "http://x", s)
    with pytest.raises(Refusal, match="engine is off"):
        check_preconditions({"think": {}, "model_key": "k", "temperature": None}, "http://x", s)
    check_preconditions({"think": {}, "model_key": "k", "temperature": 1.9, "cache_prompt": {"plain": True}}, "http://x", s)
    down = calibrate_one(s, url="http://127.0.0.1:1", name="x", out_dir=tmp_path / "think", resume=False,
                         log=served.log.append)
    assert down.status == "refused" and "cannot read" in down.reason
    empty = tmp_path / "empty.jsonl"
    empty.write_text(json.dumps({"state": "s", "questions": {"q": {"type": "noul"}}}) + "\n")
    out = calibrate_one(_settings(empty), url=served.url, name="x", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append)
    assert out.status == "failed" and "no labelled text questions" in out.reason


def _measured(prof: dict) -> tuple:
    """A profile's measurement without the latencies, which a re-read re-measures (ms jitter)."""
    budgets = {b: {k: v for k, v in e.items() if not k.startswith("lat_")} for b, e in prof["budgets"].items()}
    rows = [{**r, "b": {b: c[:3] + c[4:] for b, c in r["b"].items()},
             "t": (r.get("t") or [])[:1] + (r.get("t") or [])[2:]}                # rows[].t's a is wall time
            for r in prof["rows"]]
    audit = {b: {k: v for k, v in e.items() if not k.startswith("lat_err")} for b, e in prof["audit"]["budgets"].items()}
    return budgets, rows, prof["audit"]["latency_model"], audit


def test_resume_continues_an_interrupted_run_to_the_same_profile_and_refuses_another_run(tmp_path, served):
    data = write_data(tmp_path / "d.jsonl", 8)
    out_dir = tmp_path / "think"
    first = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=False,
                          log=served.log.append, therm=lambda: 100)
    assert first.status == "done"
    full = json.loads(first.profile.read_text())
    raw_path = out_dir / "fake.raw.jsonl"
    lines = raw_path.read_text().splitlines()
    cut = next(i for i, line in enumerate(lines) if '"phase":"B"' in line) + 3   # abort three questions into phase B
    raw_path.write_text("\n".join(lines[:cut]) + "\n" + '{"phase":"B","q":"rec-' )   # plus a hard kill's partial line
    assert calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=False,
                         log=served.log.append).status == "refused"
    gens_before = len(served.gens())
    again = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=True,
                          log=served.log.append, therm=lambda: 100)
    assert again.status == "done"
    resumed = json.loads(again.profile.read_text())
    assert _measured(resumed) == _measured(full)
    assert len(resumed["server_state"]["checks"]) > len(full["server_state"]["checks"])   # a second session's checks
    assert 0 < len(served.gens()) - gens_before < 1 + 8 + 3 + 3        # the warm-up again, then only the missing cells
    warm = [r for r in read_raw(raw_path).rows if r["phase"] == "warmup"]
    assert [r["session"] for r in warm] == [0, 1]                       # every session warms up (never measured)
    # the run started when its first session did; each session's thermal readings carry its number
    assert resumed["environment"]["run"]["started"] == full["environment"]["run"]["started"]
    assert [(t["session"], t["at"]) for t in resumed["drift"]["therm"]] == \
        [(0, "start"), (0, "A"), (1, "start"), (1, "A"), (1, "B"), (1, "end")]
    gens_after = len(served.gens())
    other = calibrate_one(_settings(data, budgets=(0, 128)), url=served.url, name="fake", out_dir=out_dir,
                          resume=True, log=served.log.append)
    assert other.status == "refused" and "written by a different run (comparable differ" in other.reason
    refit = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=True,
                          log=served.log.append, therm=lambda: 100)
    assert refit.status == "done" and json.loads(refit.profile.read_text())["rows"] == resumed["rows"]
    assert len(served.gens()) == gens_after                            # a complete raw file is only re-fitted
    # the only warnings a resume on the same box may print: the host fingerprint's (a host whose
    # fingerprint callo cannot read is "unknown host", a warning, never a match)
    warnings = [line for line in served.log if line.startswith("WARNING")]
    assert all(line.startswith(("WARNING: --resume: unknown host:", "WARNING: --resume: host differs"))
               for line in warnings), warnings


def test_the_audit_flags_a_non_linear_decode_as_unverified(tmp_path, served):
    served.up.think_slow_after = 64                                     # tokens past 64 cost double: not linear
    data = write_data(tmp_path / "d.jsonl", 8)
    out = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append, therm=lambda: 100)
    prof = json.loads(out.profile.read_text())
    a = prof["audit"]["budgets"]["64"]
    assert a["n"] == 3 and a["trace_identical"] == 3 and a["lat_err_median"] > 0.1
    assert a["latency_model"] == "unverified" and prof["audit"]["latency_model"] == "unverified"
    assert load_think_profile(out.profile).latency_model == "unverified"


def test_the_drift_check_compares_regenerations_with_the_warm_phase_b_decode_speed(tmp_path, served):
    seen = []

    def slow_down_the_drift_block(body):
        seen.append(body["n_predict"])
        if seen.count(256) > 1 + 8:                                      # after the warm-up and phase B generations
            served.up.think_ms_per_token = 15.0

    served.up.think_gen_hook = slow_down_the_drift_block
    data = write_data(tmp_path / "d.jsonl", 8)
    out = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append, therm=lambda: 100)
    prof = json.loads(out.profile.read_text())
    assert prof["drift"]["ratio"] == pytest.approx(1.5) and prof["drift"]["flagged"] is True
    assert prof["audit"]["latency_model"] == "verified"                  # the audit ran before the slowdown
    assert prof["drift"]["therm"][0] == {"at": "start", "limit": 100, "session": 0}
    assert load_think_profile(out.profile).drift_flagged
    served.up.think_gen_hook = None                                      # a uniformly slower server has ratio 1.0...
    out = calibrate_one(_settings(data), url=served.url, name="hot", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append, therm=lambda: 97)
    prof = json.loads(out.profile.read_text())
    assert prof["drift"]["ratio"] == pytest.approx(1.0) and prof["drift"]["flagged"] is True   # ...but a hot box flags
    assert prof["drift"]["therm"][0] == {"at": "start", "limit": 97, "session": 0}


# ---- GGUF sampling headers, --think-max and the default ladder in the run ---------------------------------------

def _long(text: str) -> int:
    """Questions 1, 3, 10 and 11 close at 40 tokens, the others at 1100: cut at 256, 512 AND 1024."""
    return 40 if any(f"state {i}" in text for i in (1, 3, 10, 11)) else 1100


def test_a_full_run_on_the_default_ladder_audits_256_and_512_and_fits(tmp_path):
    s = _Served(tmp_path, think_len=_long)
    try:
        data = write_data(tmp_path / "d.jsonl", 8)                     # 6 questions cut at every positive budget,
        out = calibrate_one(_settings(data, budgets=DEFAULT_BUDGETS), url=s.url, name="fake",
                            out_dir=tmp_path / "think", resume=False, log=s.log.append, therm=lambda: 100)
        assert out.status == "done", out.reason
        prof = _strict(out.profile)
        assert set(prof["budgets"]) == {"0", "256", "512", "1024"} and set(prof["audit"]["budgets"]) == {"256", "512"}
        for b in ("256", "512"):
            a = prof["audit"]["budgets"][b]
            assert (a["n_cut"], a["n"], a["trace_identical"], a["argmax_agree"], a["latency_model"],     # rec-7, the
                    a["prefill_reused"]) == (5, 3, 3, 3, "verified", 0.0)       # warm-up's, is not in the pool
        assert prof["audit"]["latency_model"] == "verified" and load_think_profile(out.profile).latency_model == "verified"
        assert prof["budgets"]["1024"]["think_mean"] == (6 * 1024 + 2 * 41) / 8
        raw = read_raw(tmp_path / "think" / "fake.raw.jsonl")
        assert Counter(r["phase"] for r in raw.rows) == {"warmup": 1, "A": 8, "B": 8, "audit": 6, "drift": 3}
        assert len(s.gens(1024)) == 1 + 8 + 3 and len(s.gens(256)) == len(s.gens(512)) == 3
    finally:
        s.stop()


def test_the_profile_records_the_served_gguf_error_and_is_written_as_strict_json(tmp_path, monkeypatch):
    def unreadable(path):
        raise ValueError("planted: no header")

    monkeypatch.setattr(callo.cli, "read_gguf_sampling", unreadable)    # serve runs in-process: its read fails
    s = _Served(tmp_path)
    try:
        data = write_data(tmp_path / "d.jsonl", 4)
        out = calibrate_one(_settings(data), url=s.url, name="fake", out_dir=tmp_path / "think", resume=False,
                            log=s.log.append, therm=lambda: 100)
        assert out.status == "done", out.reason
        prof = _strict(out.profile)
        err = "the GGUF metadata could not be read (ValueError: planted: no header)"
        assert prof["comparable"]["sampling"]["gguf_error"] == err
        assert prof["comparable"]["sampling"]["gguf"] == {} and load_think_profile(out.profile).sampling.gguf_error == err
        assert read_raw(tmp_path / "think" / "fake.raw.jsonl").header["comparable"]["sampling"]["gguf_error"] == err
        # a GGUF's raw sampling header may hold nan / inf (recorded, never pinned); the writer keeps the
        # profile strict JSON all the same, and the loader reads the stringified form back
        fit = ct.build_profile

        def with_a_nan(*a, **k):
            prof = fit(*a, **k)
            prof["comparable"]["sampling"]["gguf"] = {"temp": float("nan"), "top_p": float("inf")}
            return prof

        monkeypatch.setattr(ct, "build_profile", with_a_nan)
        out = calibrate_one(_settings(data), url=s.url, name="nan", out_dir=tmp_path / "think", resume=False,
                            log=s.log.append, therm=lambda: 100)
        assert out.status == "done", out.reason
        assert _strict(out.profile)["comparable"]["sampling"]["gguf"] == {"temp": "nan", "top_p": "inf"}
        assert load_think_profile(out.profile).sampling.policy == "model"
    finally:
        s.stop()


def test_budgets_above_the_server_s_think_max_are_measured_with_a_warning(tmp_path):
    s = _Served(tmp_path, serve_args=("--think-max", "64"))
    try:
        data = write_data(tmp_path / "d.jsonl", 4)
        out = calibrate_one(_settings(data), url=s.url, name="fake", out_dir=tmp_path / "think", resume=False,
                            log=s.log.append, therm=lambda: 100)
        assert out.status == "done"                    # calibration may explore past what the server serves...
        assert set(json.loads(out.profile.read_text())["budgets"]) == {"0", "64", "256"}
        assert ("WARNING: budget 256 is above this server's --think-max 64: it is measured, but serving it needs "
                "`callo serve --think-max 256` or more") in s.log                      # ...and says so
    finally:
        s.stop()


# ---- refits, resumes and failed writes -------------------------------------------------------------------------

def test_a_single_option_question_runs_at_every_budget_without_a_trace_and_is_never_audited_or_drifted(tmp_path,
                                                                                                         served):
    data = write_data(tmp_path / "d.jsonl", 8, extra=[json.dumps(
        {"state": "state single", "questions": {"q": {"type": "choice", "instructions": "pick",
                                                      "criteria": {"o0": "only"}, "label": "o0"}},
         "_meta": {"id": "single", "variant": "clean"}})])
    out = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append, therm=lambda: 100)
    assert out.status == "done", out.reason
    prof = _strict(out.profile)
    assert prof["budgets"]["0"]["n"] == prof["budgets"]["64"]["n"] == prof["budgets"]["256"]["n"] == 9
    row = next(r for r in prof["rows"] if r["q"] == "single/q")
    assert all(c[0] is True and c[4] == 0 and c[5] is None for c in row["b"].values())    # its answer, no thinking
    raw = read_raw(tmp_path / "think" / "fake.raw.jsonl")
    b = next(r for r in raw.rows if r["phase"] == "B" and r["q"] == "single/q")
    assert b["trace"] is None and set(b["budgets"]) == {"64", "256"}
    assert not [r for r in raw.rows if r["phase"] in ("audit", "drift") and r["q"] == "single/q"]
    assert Counter(r["phase"] for r in raw.rows) == {"warmup": 1, "A": 9, "B": 9, "audit": 3, "drift": 3}


def test_a_resume_with_only_audit_or_drift_cells_left_warms_up_first(tmp_path, served):
    data = write_data(tmp_path / "d.jsonl", 8)
    out_dir = tmp_path / "think"
    assert calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=False,
                         log=served.log.append, therm=lambda: 100).status == "done"
    raw_path = out_dir / "fake.raw.jsonl"
    lines = raw_path.read_text().splitlines()
    cut = next(i for i, line in enumerate(lines) if '"phase":"audit"' in line)       # every A and B cell is there
    raw_path.write_text("\n".join(lines[:cut]) + "\n")
    before = len(served.gens())
    again = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=True,
                          log=served.log.append, therm=lambda: 100)
    assert again.status == "done"
    rows = read_raw(raw_path).rows
    assert [r["session"] for r in rows if r["phase"] == "warmup"] == [0, 1]      # warm before the audit
    assert Counter(r["phase"] for r in rows) == {"warmup": 2, "A": 8, "B": 8, "audit": 3, "drift": 3}
    assert len(served.gens()) - before == 1 + 3 + 3                    # the warm-up, the audit, the drift block
    first_new = next(i for i, r in enumerate(rows) if r["phase"] == "warmup" and r["session"] == 1)
    assert [r["phase"] for r in rows[first_new + 1:]] == ["audit"] * 3 + ["drift"] * 3


def test_a_retried_probe_rechecks_the_server_at_once_so_a_restart_keeps_no_cold_row(tmp_path, served, monkeypatch):
    real_read, real_stats = ct.ProbeClient.read, ct.read_stats_mid_run
    state = {"reads": 0, "restart_at": 4, "restarted": False}

    def read(self, item, budgets, seed, sampling):
        reply = real_read(self, item, budgets, seed, sampling)
        state["reads"] += 1
        if state["reads"] == state["restart_at"]:          # the warm-up, then phase A's 3rd question: a connection
            state["restarted"] = state["restart_at"] == 4  # error, retried, answered (by a new callo, or the same)
            return dataclasses.replace(reply, retries=1)
        return reply

    def stats(base, **k):
        s = real_stats(base, **k)
        return {**s, "instance": "a-new-callo"} if state["restarted"] else s

    monkeypatch.setattr(ct.ProbeClient, "read", read)
    monkeypatch.setattr(ct, "read_stats_mid_run", stats)
    data = write_data(tmp_path / "d.jsonl", 4)
    out = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append, therm=lambda: 100)
    assert out.status == "refused" and "callo restarted during calibration" in out.reason
    rows = read_raw(tmp_path / "think" / "fake.raw.jsonl").rows
    assert Counter(r["phase"] for r in rows) == {"warmup": 1, "A": 2}           # the cold row was never written
    # the same server after a retry: one extra check, at once (before the row), and the run carries on
    state.update(reads=0, restart_at=5, restarted=False)
    out = calibrate_one(_settings(data), url=served.url, name="same", out_dir=tmp_path / "think", resume=False,
                        log=served.log.append, therm=lambda: 100)
    assert out.status == "done", out.reason
    checks = read_raw(tmp_path / "think" / "same.raw.jsonl").checks
    assert any(c["slack"] == 2 and c["readouts"] == 1 + 3 and not c["failed"] for c in checks)


def test_the_live_eta_counts_only_the_questions_left_to_measure(tmp_path, served):
    data = write_data(tmp_path / "d.jsonl", 8)
    out_dir = tmp_path / "think"
    assert calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=False,
                         log=served.log.append, therm=lambda: 100).status == "done"
    raw_path = out_dir / "fake.raw.jsonl"
    lines = raw_path.read_text().splitlines()
    cut = next(i for i, line in enumerate(lines) if '"phase":"A"' in line) + 1         # one phase-A question kept
    raw_path.write_text("\n".join(lines[:cut]) + "\n")
    served.log.clear()
    ticks = iter(range(1000))
    assert calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=True,
                         log=served.log.append, therm=lambda: 100, clock=lambda: float(next(ticks))).status == "done"
    eta = [line for line in served.log if "per question" in line]
    assert any("in phase A" in line and "for the 2 left in this phase" in line for line in eta), eta   # 7 to do, 5 done
    assert any("in phase B" in line and "for the 3 left in this phase" in line for line in eta), eta   # 8 to do


def test_refitting_a_complete_raw_file_on_a_hot_busy_server_changes_no_measurement(tmp_path, served):
    data = write_data(tmp_path / "d.jsonl", 8)
    out_dir = tmp_path / "think"
    first = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=False,
                          log=served.log.append, therm=lambda: 100)
    assert first.status == "done"
    full = json.loads(first.profile.read_text())
    raw_path = out_dir / "fake.raw.jsonl"
    raw_bytes, gens, calls = raw_path.read_bytes(), len(served.gens()), _upstream_calls(served)
    hot = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=True,
                        log=served.log.append, therm=lambda: 90)         # a hot box alone: no reading is taken
    assert hot.status == "done" and json.loads(hot.profile.read_text())["drift"] == full["drift"]
    assert raw_path.read_bytes() == raw_bytes
    served.up.set_slot(2, is_processing=True, n_prompt_tokens=300)      # the box is now busy...
    deadline = time.monotonic() + 5                                      # (callo's /slots snapshot is <= 250 ms old)
    while not any(x.get("is_processing") for x in get_json(f"{served.url}/callo/stats")[2]["slots"]):
        assert time.monotonic() < deadline, "callo never saw the busy slot"
        time.sleep(0.05)
    refit = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=True,
                          log=served.log.append, therm=lambda: 90)       # ...and hot
    assert refit.status == "done", refit.reason                          # a refit needs no server state
    prof = json.loads(refit.profile.read_text())
    assert raw_path.read_bytes() == raw_bytes and len(served.gens()) == gens   # no reading, no check, no request:
    assert _upstream_calls(served) == calls                                       # not even a preflight
    assert prof["drift"] == full["drift"] and prof["drift"]["flagged"] is False
    assert prof["server_state"] == full["server_state"] and prof["server_state"]["state"] == "clean"
    assert prof["environment"]["run"]["started"] == full["environment"]["run"]["started"]
    assert {k: v for k, v in prof.items() if k not in ("created", "environment")} == \
        {k: v for k, v in full.items() if k not in ("created", "environment")}
    fresh = calibrate_one(_settings(data), url=served.url, name="other", out_dir=out_dir, resume=False,
                          log=served.log.append, therm=lambda: 90)
    assert fresh.status == "refused" and "foreign_processing" in fresh.reason     # the busy box is real: a run refuses


DEAD = "http://127.0.0.1:1"                                               # nothing listens: any request is refused
PLAN_PATHS = ("/apply-template", "/tokenize", "/completion")               # what a preflight or a readout sends upstream


def _upstream_calls(s: _Served) -> int:
    return sum(q.path in PLAN_PATHS for q in s.up.requests)


def test_a_complete_raw_file_refits_with_no_request_and_no_server(tmp_path, served):
    """The raw header records what the fit reads from /callo/stats
    (`thinking`, the model's sizes, the user delimiter), so a refit of a complete raw file makes no request at all,
    and works with the server gone; it may re-pick under other --latency / --max-latency / --min-accuracy."""
    data = write_data(tmp_path / "d.jsonl", 8)
    out_dir = tmp_path / "think"
    first = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=False,
                          log=served.log.append, therm=lambda: 100)
    assert first.status == "done"
    full = json.loads(first.profile.read_text())
    header = read_raw(out_dir / "fake.raw.jsonl").header
    assert header["thinking"] == full["thinking"] and header["thinking"]["capable"] is True
    assert header["model"] == {"bytes": full["environment"]["model"]["bytes"], "mmproj_bytes": None}
    assert header["user_delimiter"] == full["comparable"]["readout"]["user_delimiter"]
    calls, log = _upstream_calls(served), []
    refit = calibrate_one(_settings(data), url=DEAD, name="fake", out_dir=out_dir, resume=True, log=log.append,
                          therm=lambda: pytest.fail("a refit takes no thermal reading"))
    assert refit.status == "done", refit.reason
    assert _upstream_calls(served) == calls and any(line.startswith("refit: ") for line in log)
    prof = json.loads(refit.profile.read_text())
    assert {k: v for k, v in prof.items() if k not in ("created", "environment")} == \
        {k: v for k, v in full.items() if k not in ("created", "environment")}
    assert {**prof["environment"], "run": None} == {**full["environment"], "run": None}      # from the header alone
    assert prof["environment"]["run"]["started"] == full["environment"]["run"]["started"]
    # half the fastest point's p95 (in s), so no point passes however fast the fake ran
    floor_ms = min(e["lat_p95_ms"][0] for e in [*full["budgets"].values(), *(full.get("deadlines") or {}).values()])
    repick = calibrate_one(_settings(data, latency="p95", max_latency=floor_ms / 2000), url=DEAD, name="fake",
                           out_dir=out_dir, resume=True, log=log.append)
    assert repick.status == "done" and json.loads(repick.profile.read_text())["pick"]["label"] is None   # re-picked
    # the CLI finds <name> in --out, starts no serve and reads no /callo/stats, in --url and --upstream mode alike
    for mode in (["--url", DEAD], ["--upstream", DEAD]):
        rc = main(["calibrate", "--think-budgets", "0,64,256", "--data", str(data), "--out", str(out_dir), "--warmup", "1",
                   "--audit", "3", "--resume", *mode])
        assert rc == 0 and _upstream_calls(served) == calls
    # other settings are no refit: the server is asked, and here it is gone
    assert calibrate_one(_settings(data, seed=1), url=DEAD, name="fake", out_dir=out_dir, resume=True,
                         log=log.append).status == "refused"


def test_an_older_raw_file_asks_the_server_for_the_fit_fields_and_a_changed_thinking_mode_refuses(tmp_path, served):
    """A raw file written before the header recorded `HEADER_FIT_KEYS` falls back to /callo/stats with a note, and
    still sends no preflight when nothing is left to read; a pending resume whose server's thinking mode differs from
    the header's is refused like any header difference."""
    data = write_data(tmp_path / "d.jsonl", 8)
    out_dir = tmp_path / "think"
    full = json.loads(calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=False,
                                    log=served.log.append, therm=lambda: 100).profile.read_text())
    raw_path = out_dir / "fake.raw.jsonl"
    lines = raw_path.read_text().splitlines()
    head = json.loads(lines[0])
    old = {k: v for k, v in head.items() if k not in ct.HEADER_FIT_KEYS}
    raw_path.write_text("\n".join([json.dumps(old), *lines[1:]]) + "\n")
    log: list[str] = []
    assert calibrate_one(_settings(data), url=DEAD, name="fake", out_dir=out_dir, resume=True,
                         log=log.append).status == "refused"
    assert any(line.startswith("NOTE: fake.raw.jsonl was written before callo recorded thinking, model, user_delimiter")
               for line in log)
    calls = _upstream_calls(served)
    again = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=True,
                          log=log.append, therm=lambda: pytest.fail("a refit takes no thermal reading"))
    assert again.status == "done" and _upstream_calls(served) == calls          # no preflight, no readout
    assert _measured(json.loads(again.profile.read_text())) == _measured(full)
    flipped = {**head, "thinking": {**head["thinking"], "reason": "something else"}}
    keep = [ln for ln in lines[1:] if '"phase":"drift"' not in ln]          # the drift block is left to read
    raw_path.write_text("\n".join([json.dumps(flipped), *keep]) + "\n")
    other = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=True,
                          log=log.append, therm=lambda: 100)
    assert other.status == "refused" and "thinking differ" in other.reason


def test_resume_names_the_run_from_the_server_and_only_a_server_down_falls_back_to_out(tmp_path, served, capsys):
    """--out holds a complete a.raw.jsonl and an interrupted m.raw.jsonl; `--resume`
    against m's serve asks it once who it is and resumes m, never refits a. With the server down, the one complete
    raw file for these settings refits; two are a refusal naming both."""
    data = write_data(tmp_path / "d.jsonl", 8)
    out_dir = tmp_path / "think"
    for name in ("a", "m"):
        assert calibrate_one(_settings(data), url=served.url, name=name, out_dir=out_dir, resume=False,
                             log=served.log.append, therm=lambda: 100).status == "done"
    m_raw = out_dir / "m.raw.jsonl"
    lines = m_raw.read_text().splitlines()
    m_raw.write_text("\n".join(lines[: next(i for i, ln in enumerate(lines) if '"phase":"B"' in ln) + 3]) + "\n")
    (out_dir / "m.think.json").unlink()
    a_profile = (out_dir / "a.think.json").read_bytes()
    flags = ["calibrate", "--think-budgets", "0,64,256", "--data", str(data), "--out", str(out_dir), "--warmup", "1",
             "--audit", "3", "--resume"]
    gens = len(served.gens())
    assert main([*flags, "--url", served.url]) == 0
    captured = capsys.readouterr()
    assert "a.raw.jsonl" not in captured.err and captured.out.startswith("m  m.gguf")
    assert Counter(r["phase"] for r in read_raw(m_raw).rows)["B"] == 8 and len(served.gens()) > gens   # m resumed
    assert (out_dir / "m.think.json").exists() and (out_dir / "a.think.json").read_bytes() == a_profile
    calls = _upstream_calls(served)
    (out_dir / "a.raw.jsonl").rename(tmp_path / "a.raw.jsonl")           # server down, one candidate: m refits
    assert main([*flags, "--url", DEAD]) == 0 and _upstream_calls(served) == calls
    captured = capsys.readouterr()
    assert "--resume: refitting m.raw.jsonl" in captured.err and captured.out.startswith("m  m.gguf")
    assert main([*flags, "--upstream", DEAD]) == 0 and _upstream_calls(served) == calls
    assert "nothing answers at http://127.0.0.1:1; --resume: refitting m.raw.jsonl" in capsys.readouterr().err
    (tmp_path / "a.raw.jsonl").rename(out_dir / "a.raw.jsonl")           # server down, two candidates: refused
    for source in (["--url", DEAD], ["--upstream", DEAD]):
        assert main([*flags, *source]) == 2
        err = capsys.readouterr().err
        assert "holds several complete raw files for these settings (a.raw.jsonl, m.raw.jsonl)" in err
        assert "start the server that serves it" in err
    assert main([*flags[:-1], "--url", DEAD]) == 2                       # no --resume: never a refit, as before
    assert "cannot read http://127.0.0.1:1/callo/stats" in capsys.readouterr().err


def test_malformed_replies_and_rows_fail_cleanly_and_a_failed_write_keeps_the_old_profile(tmp_path, served,
                                                                                            monkeypatch):
    """A 200 reply whose question block lacks a field the row is built from is RunFailed
    naming the question (exit 1) and writes no row; a valid-JSON raw row the fit cannot read is a clean failure; a
    profile write that fails (ENOSPC) leaves the previous profile whole and no temporary file behind."""
    data = write_data(tmp_path / "d.jsonl", 6)
    out_dir = tmp_path / "think"
    real_read = ct.ProbeClient.read

    def no_ms(self, item, budgets, seed, sampling):
        reply = real_read(self, item, budgets, seed, sampling)
        if list(budgets) == [0]:
            del reply.body["questions"][item.qid]["budgets"]["0"]["ms"]
        return reply

    monkeypatch.setattr(ct.ProbeClient, "read", no_ms)
    out = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=False,
                        log=served.log.append, therm=lambda: 100)
    assert out.status == "failed" and out.reason.endswith("unusable reply (KeyError: 'ms')")
    assert "A" not in {r["phase"] for r in read_raw(out_dir / "fake.raw.jsonl").rows}
    monkeypatch.setattr(ct.ProbeClient, "read", real_read)
    (out_dir / "fake.raw.jsonl").unlink()
    good = calibrate_one(_settings(data), url=served.url, name="fake", out_dir=out_dir, resume=False,
                         log=served.log.append, therm=lambda: 100)
    assert good.status == "done"
    before = good.profile.read_bytes()
    real_write = Path.write_text

    def full_disk(self, text, *a, **kw):
        real_write(self, text[:20], *a, **kw)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", full_disk)
    failed = calibrate_one(_settings(data), url=DEAD, name="fake", out_dir=out_dir, resume=True, log=served.log.append)
    monkeypatch.setattr(Path, "write_text", real_write)
    assert failed.status == "failed" and "cannot write in --out" in failed.reason and "No space left" in failed.reason
    assert good.profile.read_bytes() == before and sorted(p.name for p in out_dir.iterdir()) == \
        ["fake.raw.jsonl", "fake.think.json"]
    lines = (out_dir / "fake.raw.jsonl").read_text().splitlines()
    i = next(n for n, ln in enumerate(lines) if '"phase":"A"' in ln)
    row = json.loads(lines[i])
    del row["probs"]
    lines[i] = json.dumps(row)
    (out_dir / "fake.raw.jsonl").write_text("\n".join(lines) + "\n")
    bad = calibrate_one(_settings(data), url=DEAD, name="fake", out_dir=out_dir, resume=True, log=served.log.append)
    assert bad.status == "failed" and f"(A {row['q']}) is incomplete: a phase-A row needs probs and ms" in bad.reason


def test_an_out_that_is_a_file_exits_1_with_a_message(tmp_path, served, capsys):
    """In temperature and --bias mode --out is a file, so naming one here is a likely mistake: a
    clean exit 1 that says --out is a directory, never a FileExistsError traceback."""
    data = write_data(tmp_path / "d.jsonl", 2)
    afile = tmp_path / "T.json"
    afile.write_text("{}")
    rc = main(["calibrate", "--think-budgets", "0,64", "--url", served.url, "--data", str(data), "--out", str(afile),
               "--warmup", "0", "--audit", "0"])
    err = capsys.readouterr().err
    assert rc == 1 and "Traceback" not in err
    assert f"callo: cannot write in --out {afile}" in err and "with --think-budgets, --out is a DIRECTORY" in err


# ---- the summary, --dry-run and the CLI ------------------------------------------------------------------------

from callo.calibrate_think import dry_run_lines, settings_from_args, summary_lines  # noqa: E402
from callo.cli import build_parser, main  # noqa: E402


def test_summary_lines_name_every_point_the_verdicts_the_pick_and_the_serve_line(tmp_path):
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    lines = summary_lines(prof, path=Path("callo-think/fake.think.json"))
    assert lines[0].startswith("fake  fake-9b.gguf  2.0 GiB  model_key sha256:aaaaaaaaa  data d.jsonl (8 questions)")
    assert "latency: single-stream on this host, in-callo wall time" in lines[0]
    assert lines[1].startswith("sampler 0.6/0.95/20/0.0 (fixed; temperature=callo, top_p=callo") and "server state clean" in lines[1]
    assert lines[1].endswith("server args not recorded (attach mode: -ctk/-ctv, MTP and draft settings unknown)")
    rows = [line for line in lines if line.startswith("fake@")]
    assert [r.split()[0] for r in rows] == ["fake@0", "fake@64", "fake@256"]
    assert rows[0].split()[1] == "0.500" and "+37.5pp" in rows[1] and "+50.0pp" in rows[2]
    assert rows[2].endswith("50%  ≈") and rows[1].endswith("50%") and rows[0].endswith("-")   # on mean
    assert any(line.startswith("audit: latency model verified (B=64: 4/4 traces identical") for line in lines)
    assert "generation prefill reused from the question's own cached state: B=64: 0%, B=256: 0%" in lines
    assert any(line.startswith("drift: none; end/phase-B ms per token 1.00 over 3 regenerations; thermal limit")
               for line in lines)
    assert "pick: fake@64 (fastest within noise of the best); best fake@256; Δacc vs it -12.5pp [-37.5, +0.0]" in lines
    # attach mode (no server_args): the serve line names --upstream, the file only when callo launched it
    assert lines[-1] == "serve with: callo serve --upstream URL --think-profile callo-think/fake.think.json --think-budget 64"
    assert not any("size unknown" in line or "slightly higher" in line for line in lines)   # nothing to say on either
    verbose = summary_lines(prof, verbose=True)
    assert "NLL@T" in verbose[2] and len(verbose[3].split()) >= 12
    prof["thinking"] = {"capable": False, "reason": "not_preopened"}
    assert "NO THINKING MODE (not_preopened): measured budget 0 only" in summary_lines(prof)
    raw, settings = synthetic_raw(items, gen_prompt_n=2)                 # the host prompt cache flattered phase B
    flagged = summary_lines(build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake"))
    assert any(line.startswith("audit: latency model unverified, prefill_reused (B=64") for line in flagged)
    assert any(line.startswith("generation prefill reused from the question's own cached state: B=64: 100%, B=256: 100% -- ")
               and line.endswith("restart llama-server with --cache-ram 0 for calibration, or accept the flattered prefill")
               for line in flagged)
    raw, settings = synthetic_raw(items, cut_items=0)                    # 64 and 256 merge: the largest budget serves
    merged = summary_lines(build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake"))
    assert merged[-1].endswith("--think-budget 256") and any(line.startswith("pick: fake@64≡256 ") for line in merged)
    assert "  budgets 64-256 read the same thing on every question; any serves, 256 truncates the fewest unseen questions" in merged
    raw, settings = synthetic_raw(items)
    noisy = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    noisy["pick"]["label"], noisy["pick"]["dacc_best"] = "fake@64", [-0.125, -0.3, 0.05]
    assert any(line == "pick: fake@64 (fastest within noise of the best); best fake@256; Δacc vs it -12.5pp [-30.0, +5.0]"
               for line in summary_lines(noisy))


def test_the_summary_notes_a_point_whose_t_is_at_the_search_limit(tmp_path):
    """A fitted T on the grid's top (or bottom) point is a real signal since 0.3.0's grid reaches 16: the summary
    names the points, above the pick; an interior T says nothing."""
    from callo.calibrate import GRID
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    for e in prof["budgets"].values():
        e["T"] = 1.5
    assert not any("search limit" in line for line in summary_lines(prof))
    prof["budgets"]["0"]["T"], prof["budgets"]["256"]["T"], prof["budgets"]["64"]["T"] = GRID[-1], GRID[-1], GRID[0]
    lines = summary_lines(prof)
    high = ("T is at the search limit (16) at fake@0, fake@256: the probabilities are still overconfident after "
            "calibration")
    low = ("T is at the search limit (0.25) at fake@64: the fit wants sharper probabilities than the grid allows "
           "(common on a small or perfectly separated set)")
    pick = next(i for i, line in enumerate(lines) if line.startswith("pick: "))
    assert lines[pick - 2:pick] == [high, low]                                    # top-level lines, just before pick:
    prof["budgets"]["0"]["T"] = GRID[-2]
    assert any(line.startswith("T is at the search limit (16) at fake@256:") for line in summary_lines(prof))
    prof["budgets"]["64"]["T_flat"] = True                    # a tie on a flat loss, not a fit: no 0.25 note
    assert not any(line.startswith("T is at the search limit (0.25)") for line in summary_lines(prof))
    for r in raw.rows:                                        # budget 0 read uniform: its loss is flat in T
        if r.get("phase") == "A":
            r["probs"] = [1 / len(r["probs"])] * len(r["probs"])
    flat = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    assert flat["budgets"]["0"]["T"] == GRID[0] and flat["budgets"]["0"]["T_flat"] is True
    assert not any("T_flat" in flat["budgets"][b] for b in ("64", "256"))
    assert not any(line.startswith("T is at the search limit (0.25) at fake@0") for line in summary_lines(flat))


def test_the_summary_shows_the_interval_on_the_pick_s_axis_the_small_n_caveat_no_thermal_data_and_a_runnable_serve_line(
        tmp_path):
    """Under --latency p95 the interval column is p95's;
    below 100 questions the summary says frontier neighbours will mostly be ≈; null thermal readings print "no thermal
    data" (as the README says); the serve line adds --think-max above serve's 4096 default, --no-cache when measured
    without the cache (on by default since 0.3.0), and names --model only when callo launched the server."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    prof = build_profile(items, raw, dataclasses.replace(settings, latency="p95"), sampling=SAMPLING, stats=STATS,
                         name="fake")
    lines = summary_lines(prof)
    assert "p95 [95% CI]" in lines[2] and "p50 [95% CI]" not in lines[2]
    row = next(line for line in lines if line.startswith("fake@256"))
    p95 = prof["budgets"]["256"]["lat_p95_ms"]
    assert f"{p95[0] / 1000:.2f}s [{p95[1] / 1000:.2f}s, {p95[2] / 1000:.2f}s]" in row
    assert ("  8 questions: below about 100, neighbouring frontier points will mostly be ≈; rerunning with another "
            "--seed estimates the trace variance") in lines
    assert any(line.startswith("drift: none;") and "thermal limit A 100, end 100" in line for line in lines)
    prof["drift"]["therm"] = [{"at": "start", "limit": None}, {"at": "A", "limit": None}]
    assert any(line.startswith("drift: none;") and line.endswith("; no thermal data") for line in summary_lines(prof))
    prof["pick"]["budget"] = 8192
    prof["comparable"]["cache"] = False
    prof["environment"]["server"]["server_args"] = ["-m", "/models/fake-9b.gguf"]
    assert summary_lines(prof, path=Path("p.think.json"))[-1] == \
        "serve with: callo serve --model fake-9b.gguf --think-profile p.think.json --think-budget 8192 --think-max 8192 --no-cache"


SERVED_HIGHER = ("  with --cache-ram 0 for calibration, served latency can come out slightly higher than calibrated on a "
                 "server that keeps the host cache (llama-server's default): each question switch there also pays a "
                 "state save, a few ms, more on big dense models")


def test_the_summary_says_when_served_latency_can_exceed_the_calibrated_one_and_names_unsized_points(tmp_path):
    """Calibrated with the host cache off (`--cache-ram 0`, the remedy the flagged summary names,
    or a launch that stamps `cache_ram: 0`), serving with it on also pays a per-question state save. And
    `Pick.unsized` is printed through `pick_note` under the pick line when it names any point."""
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items, gen_prompt_n=2)
    flagged = summary_lines(build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake"))
    reuse = next(i for i, line in enumerate(flagged) if line.startswith("generation prefill reused"))
    assert flagged[reuse + 1] == SERVED_HIGHER
    raw, settings = synthetic_raw(items)
    prof = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    prof["environment"]["server"]["cache_ram"] = 0                       # a launch that ran with the host cache off
    assert SERVED_HIGHER in summary_lines(prof)
    prof["pick"]["unsized"] = ["big@0", "big@256"]
    lines = summary_lines(prof)
    at = next(i for i, line in enumerate(lines) if line.startswith("pick: "))
    assert lines[at + 1] == "  size unknown: excluded by --max-memory: big@0, big@256"
    prof["pick"].update({"label": None, "rule": "no pick", "budget": None})
    lines = summary_lines(prof)
    assert "pick: none (no pick); best fake@256" in lines              # a block without a reason (an older one)
    assert "  size unknown: excluded by --max-memory: big@0, big@256" in lines and not lines[-1].startswith("serve with")


def test_dry_run_lines_and_the_settings_from_the_parser(tmp_path):
    data = write_data(tmp_path / "d.jsonl", 6, extra=[json.dumps({"state": "s", "questions": {"q": {"type": "noul"}}})])
    args = build_parser().parse_args(["calibrate", "--think-budgets", "--data", str(data), "--warmup", "1",
                                      "--audit", "2", "--max-latency", "3", "--min-accuracy", "0.5",
                                      "--latency", "p95", "--think-sampling", "fixed", "--cache", "--verbose"])
    assert args.think_budgets == "0,256,512,1024"
    warnings = []
    s = settings_from_args(args, warn=warnings.append)
    assert s.budgets == (0, 256, 512, 1024) and (s.warmup, s.audit, s.seed) == (1, 2, 0) and s.cache and s.verbose
    assert warnings == ["WARNING: --audit 2 cannot verify the latency model: a verdict needs at least 3 audited cut "
                        "questions per budget, so the audit will read `unverified`; use --audit 3 or more, or --audit 0 "
                        "to skip it"]
    assert settings_from_args(args).audit == 2                            # no `warn`: silent
    three = build_parser().parse_args(["calibrate", "--think-budgets", "--data", str(data), "--audit", "3"])
    settings_from_args(three, warn=warnings.append)
    assert len(warnings) == 1
    assert (s.latency, s.max_latency, s.min_accuracy, s.sampling) == ("p95", 3.0, 0.5, "fixed")
    defaults = settings_from_args(three)
    assert (defaults.warmup, defaults.seed, defaults.sampling, defaults.latency, defaults.cache) == (3, 0, "model", "mean", True)
    items, skipped = load_data(data)
    lines = dry_run_lines(s, items, skipped)
    assert lines[0] == "dry run: 6 labelled text questions from 7 records in d.jsonl (skipped: unlabelled 1)"
    assert lines[1].startswith("budgets: [0, 256, 512, 1024]; warm-up 1, audit 2 per cut budget, seed 0, sampling fixed, --cache on")
    assert lines[2] == (f"worst case: 6 x 1024 = {6 * 1024} generated tokens, plus up to {2 * (256 + 512)} for the audit, "
                        f"{3 * 1024} for the drift check and {1 * 1024} for the warm-up")
    assert lines[3].startswith("time: unknown until measured")
    for bad in (["--warmup", "-1"], ["--audit", "-2"], ["--seed", "-1"], ["--max-latency", "0"], ["--min-accuracy", "1.5"],
                ["--think-sampling", "nonsense"], ["--think-budgets", "0,x"], ["--think-budgets", ",".join(map(str, range(9)))]):
        flags = ["calibrate", "--think-budgets", "--data", str(data), *bad] if bad[0] != "--think-budgets" else \
            ["calibrate", "--data", str(data), *bad]
        with pytest.raises(InvalidInput):
            settings_from_args(build_parser().parse_args(flags))


def test_the_cli_runs_against_a_served_callo_and_prints_the_summary(tmp_path, served, capsys):
    data = write_data(tmp_path / "d.jsonl", 6)
    out_dir = tmp_path / "think"
    rc = main(["calibrate", "--think-budgets", "0,32,64", "--url", served.url, "--data", str(data), "--out", str(out_dir),
               "--warmup", "1", "--audit", "3"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert (out_dir / "m.think.json").exists() and (out_dir / "m.raw.jsonl").exists()      # <name> = m.gguf's stem
    assert _strict(out_dir / "m.think.json")["callo_think_profile"] == 1   # the CLI writes through calibrate_one
    assert "plan: 6 questions x budgets [0, 32, 64]" in captured.err and "pick: m@" in captured.out
    assert f"serve with: callo serve --upstream URL --think-profile {out_dir / 'm.think.json'} --think-budget " in captured.out
    assert "audit: latency model verified (B=32: 3/3 traces identical" in captured.out and "drift: none" in captured.out
    assert main(["calibrate", "--think-budgets", "0,32,64", "--url", served.url, "--data", str(data), "--out", str(out_dir),
                 "--warmup", "1", "--audit", "3"]) == 2                                  # the raw file exists: --resume
    assert "--resume" in capsys.readouterr().err
    assert main(["calibrate", "--think-budgets", "0,32,64", "--url", served.url, "--data", str(data), "--out", str(out_dir),
                 "--warmup", "1", "--audit", "3", "--resume", "--verbose"]) == 0
    assert "NLL@T" in capsys.readouterr().out
    before = len(served.up.requests)
    rc = main(["calibrate", "--think-budgets", "0,64", "--url", served.url, "--data", str(data), "--out", str(out_dir),
               "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0 and captured.out.startswith("dry run: 6 labelled text questions") and "no requests were made" in captured.out
    assert "worst case: 6 x 64 = 384 generated tokens" in captured.out
    assert len(served.up.requests) == before                              # --dry-run: nothing reached the server


def test_the_cli_upstream_mode_attaches_an_in_process_callo_serve(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))                            # the in-process serve's ~/.cache/callo
    up = FakeUpstream(think="preopened", think_len=_len, slot_save=True).start()
    up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
    try:
        data = write_data(tmp_path / "d.jsonl", 4)
        rc = main(["calibrate", "--think-budgets", "0,64", "--upstream", up.url, "--data", str(data),
                   "--out", str(tmp_path / "think"), "--warmup", "0", "--audit", "0"])
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        assert "callo serve attached to" in captured.err and "(in-process, --think-budget 0)" in captured.err
        prof = json.loads((tmp_path / "think" / "m.think.json").read_text())
        assert sorted(prof["budgets"]) == ["0", "64"] and prof["budgets"]["64"]["n"] == 4
        assert prof["audit"] == {"latency_model": "unaudited", "latency_reason": None, "budgets": {}}
        assert prof["environment"]["server"]["upstream_local"] is True   # the in-process serve attached to loopback
        rc = main(["calibrate", "--think-budgets", "0,64", "--upstream", up.url, "--data", str(data),
                   "--out", str(tmp_path / "think2"), "--warmup", "0", "--audit", "0", "--cache"])
        assert rc == 0, capsys.readouterr().err                           # --cache configures the in-process serve
        assert json.loads((tmp_path / "think2" / "m.think.json").read_text())["comparable"]["cache"] is True
    finally:
        up.stop()
    one = FakeUpstream(total_slots=0).start()                              # serve refuses a 0-slot server at once
    try:
        assert main(["calibrate", "--think-budgets", "--upstream", one.url, "--data", str(data)]) == 2
        assert "did not start (exit 1; its log says why)" in capsys.readouterr().err
    finally:
        one.stop()
    t0 = time.monotonic()                                                  # nothing listening: a 5 s pre-check, not 600 s
    assert main(["calibrate", "--think-budgets", "--upstream", "http://127.0.0.1:1", "--data", str(data)]) == 2
    assert "nothing answers at http://127.0.0.1:1" in capsys.readouterr().err and time.monotonic() - t0 < 5
    assert main(["calibrate", "--think-budgets", "--upstream", "http://no-such-host.invalid:8199", "--data", str(data)]) == 2
    assert "nothing answers at http://no-such-host.invalid:8199" in capsys.readouterr().err
    # serve's own URL check comes first: no scheme, https, a path, "" -- a refusal naming
    # the fault, never a traceback and never "nothing answers"
    for bad, why in (("u", "it has no http:// scheme"), ("127.0.0.1:8199", "it has no http:// scheme"),
                     ("http://[::1", "is not a valid URL"), ("https://127.0.0.1:1", "https is not supported"),
                     ("http://127.0.0.1:1/llama", "has a path"), ("", "it has no http:// scheme")):
        assert main(["calibrate", "--think-budgets", "--upstream", bad, "--data", str(data)]) == 2
        err = capsys.readouterr().err
        assert f"callo: refusing to calibrate: --upstream {bad!r}" in err and why in err, err
    assert main(["calibrate", "--think-budgets", "--upstream", "http://127.0.0.1:1", "--url", "http://127.0.0.1:2",
                 "--data", str(data)]) == 2                                # one source
    assert "--url and --upstream are mutually exclusive: choose one" in capsys.readouterr().err


def test_the_upstream_precheck_lets_anything_that_answers_through(tmp_path):
    """A 503 while llama-server loads (or any HTTP status) proceeds to the in-process serve's own wait."""
    class Loading(BaseHTTPRequestHandler):
        def do_GET(self):                                                 # noqa: N802 -- http.server's name
            body = b'{"error": {"code": 503, "message": "Loading model"}}'
            self.send_response(503)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Loading)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ct.precheck_upstream(f"http://127.0.0.1:{srv.server_port}")      # a 503 answers: no Refusal
    finally:
        srv.shutdown()
        srv.server_close()
    with pytest.raises(Refusal, match="nothing answers at http://127.0.0.1:1"):
        ct.precheck_upstream("http://127.0.0.1:1")


def test_a_console_that_went_away_exits_1_without_a_traceback(tmp_path):
    """A BrokenPipeError on stdout (`callo calibrate ... | head -1`) is a quiet exit 1, as
    --bias does it."""
    class Closed:
        def write(self, text):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            pass

    data = write_data(tmp_path / "d.jsonl", 2)
    args = build_parser().parse_args(["calibrate", "--think-budgets", "--data", str(data), "--dry-run"])
    assert ct.run(args, out=Closed(), err=Closed()) == 1


def test_the_cli_usage_errors_and_refusals(tmp_path, capsys):
    data = write_data(tmp_path / "d.jsonl", 2)
    assert main(["calibrate", "--think-budgets", "--bias", "--data", str(data)]) == 2
    assert "mutually exclusive" in capsys.readouterr().err
    assert main(["calibrate", "--think-budgets"]) == 2 and "needs --data" in capsys.readouterr().err
    assert main(["calibrate", "--think-budgets", "0,x", "--data", str(data)]) == 2
    assert "--think-budgets: 'x'" in capsys.readouterr().err
    assert main(["calibrate", "--think-budgets", "--data", str(data), "--raw", "r.jsonl"]) == 2
    assert "--raw apply only with --bias" in capsys.readouterr().err
    # temperature mode: the flags --think-budgets shares with --bias say so
    assert main(["calibrate", "--url", "http://127.0.0.1:1", "--data", str(data), "--seed", "1", "--resume"]) == 2
    assert capsys.readouterr().err == "callo: --resume, --seed apply only with --bias or --think-budgets\n"
    assert main(["calibrate", "--url", "http://127.0.0.1:1", "--data", str(data), "--raw", "r", "--allow-busy"]) == 2
    assert capsys.readouterr().err == \
        "callo: --raw apply only with --bias; --allow-busy apply only with --bias or --think-budgets\n"
    for flag in (["--upstream", "u"], ["--warmup", "1"], ["--audit", "0"], ["--cache"], ["--no-cache"],
                 ["--think-sampling", "fixed"], ["--latency", "p95"], ["--max-latency", "1"], ["--min-accuracy", "0.5"], ["--verbose"], ["--dry-run"]):
        assert main(["calibrate", "--url", "http://127.0.0.1:1", "--data", str(data), *flag]) == 2
        assert "apply only with --think-budgets" in capsys.readouterr().err
        assert main(["calibrate", "--bias", "--url", "http://127.0.0.1:1", *flag]) == 2
        assert "apply only with --think-budgets" in capsys.readouterr().err
    assert main(["calibrate", "--think-budgets", "--url", "http://127.0.0.1:1", "--data", str(data)]) == 2
    assert "cannot read http://127.0.0.1:1/callo/stats" in capsys.readouterr().err
    assert main(["calibrate", "--think-budgets", "--data", str(tmp_path / "missing.jsonl"), "--dry-run"]) == 2
    assert "--data" in capsys.readouterr().err
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    assert main(["calibrate", "--think-budgets", "--data", str(bad), "--dry-run"]) == 1
    assert "line 1" in capsys.readouterr().err


def test_a_one_server_run_also_writes_report_md_and_report_json_through_compare(tmp_path, served, capsys):
    data = write_data(tmp_path / "d.jsonl", 4)
    out_dir = tmp_path / "think"
    rc = main(["calibrate", "--think-budgets", "0,32", "--url", served.url, "--data", str(data), "--out", str(out_dir),
               "--warmup", "0", "--audit", "0"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert f"report: {out_dir / 'report.md'} and {out_dir / 'report.json'}" in captured.err
    md = (out_dir / "report.md").read_text()
    # an attach-mode run (no server args recorded): the serve line names --upstream, as the summary's does
    assert md.startswith("# callo compare") and "| m@0 |" in md and "| m@32 |" in md
    assert "serve with: `callo serve --upstream URL --think-profile " in md
    js = json.loads((out_dir / "report.json").read_text())
    assert js["callo_compare"] == 1 and [p["label"] for p in js["points"]] == ["m@0", "m@32"]
    assert js["profiles"][0]["path"] == str(out_dir / "m.think.json") and js["profiles"][0]["launched_by"] == "url"
    # one source of truth end to end: the summary's serve line is the report's, same pick
    summary_serve = [line for line in captured.out.splitlines() if line.startswith("serve with: ")]
    assert summary_serve == ["serve with: " + js["pick"]["serve"]] and f"serve with: `{js['pick']['serve']}`" in md
    assert main(["calibrate", "--think-budgets", "0,32", "--data", str(data), "--dry-run"]) == 0
    assert "report.md and report.json; no requests were made" in capsys.readouterr().out


def test_a_report_that_cannot_be_written_exits_1_in_one_line_and_keeps_the_profile(tmp_path, served, capsys):
    """The report is written after the profile; an OSError there is exit 1 (an `--out` that cannot be
    written), one line, no traceback, and the profile and the summary are already out."""
    data = write_data(tmp_path / "d.jsonl", 4)
    out_dir = tmp_path / "think"
    (out_dir / "report.md").mkdir(parents=True)                 # report.md cannot be written as a file
    rc = main(["calibrate", "--think-budgets", "0,32", "--url", served.url, "--data", str(data), "--out", str(out_dir),
               "--warmup", "0", "--audit", "0"])
    captured = capsys.readouterr()
    assert rc == 1 and "Traceback" not in captured.err
    errors = [line for line in captured.err.splitlines() if line.startswith("callo: ")]
    assert len(errors) == 1 and errors[0].startswith(f"callo: {out_dir}: cannot write report.md / report.json (")
    assert _strict(out_dir / "m.think.json")["callo_think_profile"] == 1 and "serve with: " in captured.out
    assert not (out_dir / "report.json").exists() and "report: " not in captured.err


def test_a_one_server_run_into_a_menu_dir_keeps_the_menus_joint_report(tmp_path, served, capsys):
    """DIR/run.json marks a menu's DIR; its report.md / report.json are left as they are."""
    data = write_data(tmp_path / "d.jsonl", 4)
    out_dir = tmp_path / "menu"
    out_dir.mkdir()
    (out_dir / "run.json").write_text("{}")
    (out_dir / "report.md").write_text("the menu's joint report\n")
    rc = main(["calibrate", "--think-budgets", "0,32", "--url", served.url, "--data", str(data), "--out", str(out_dir),
               "--warmup", "0", "--audit", "0"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert (f"report: {out_dir} holds a menu run; its report.md is kept (callo compare {out_dir}/*.think.json for a "
            f"joint view)") in captured.err
    assert (out_dir / "report.md").read_text() == "the menu's joint report\n" and not (out_dir / "report.json").exists()
    assert (out_dir / "m.think.json").exists() and "serve with: " in captured.out


def test_a_deadline_source_on_fewer_questions_leaves_the_budget_block_byte_identical(tmp_path):
    """The budget block rests on a BUDGET-ONLY fixed set (real `Paired`, real bootstraps): a deadline source that
    covers a subset of the questions shrinks the all-kinds set the pick pairs on, never budgets[B].dacc or the marks."""
    from types import SimpleNamespace
    items, _ = load_data(write_data(tmp_path / "d.jsonl", 8))
    raw, settings = synthetic_raw(items)
    base = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake")
    deadline = {it.uid: {1000: SimpleNamespace(correct=True, lat_ms=900.0)} for it in items[:3]}
    with_d = build_profile(items, raw, settings, sampling=SAMPLING, stats=STATS, name="fake",
                           extra_kinds={"deadline": deadline})
    assert json.dumps(with_d["budgets"], sort_keys=True) == json.dumps(base["budgets"], sort_keys=True)
    assert base["budgets"]["64"]["dacc"][0] == pytest.approx(0.375)


# ---- --cache on by default: --no-cache turns it off, still --think-budgets only -----------------------------------

def test_calibrate_cache_defaults_on_and_is_still_think_only(tmp_path, capsys, builtin_set):
    """`callo calibrate --cache` is a BooleanOptionalAction with a None default: the think path resolves None to on,
    and a non-think calibrate with neither flag is not refused as if `--cache` had been given. (`builtin_set`:
    `calibrate --bias` without --data reads the built-in set, from the repo copy, never a fetch.)"""
    data = write_data(tmp_path / "d.jsonl", 2)
    base = ["calibrate", "--think-budgets", "--data", str(data)]
    assert build_parser().parse_args(base).cache is None                  # the sentinel: never "given"
    assert settings_from_args(build_parser().parse_args(base)).cache is True
    assert settings_from_args(build_parser().parse_args([*base, "--cache"])).cache is True
    assert settings_from_args(build_parser().parse_args([*base, "--no-cache"])).cache is False
    assert ThinkSettings("d", "d", "d" * 64, (0, 64)).cache is True       # the dataclass default agrees
    # no flag: a temperature or --bias calibrate is not refused as think-only (it fails later, on the dead URL)
    for argv in (["calibrate", "--url", DEAD, "--data", str(data)], ["calibrate", "--bias", "--url", DEAD]):
        main(argv)
        assert "apply only with --think-budgets" not in capsys.readouterr().err
    # an explicit --cache or --no-cache without --think-budgets is refused as before, naming the flag given
    for flag in ("--cache", "--no-cache"):
        for argv in (["calibrate", "--url", DEAD, "--data", str(data)], ["calibrate", "--bias", "--url", DEAD]):
            assert main([*argv, flag]) == 2
            assert capsys.readouterr().err == f"callo: {flag} apply only with --think-budgets\n"


def _served_in(d: Path, *serve_args: str) -> _Served:
    d.mkdir(parents=True, exist_ok=True)
    return _Served(d, serve_args=serve_args)


def _run_url(served_url: str, data: Path, out: Path, *flags: str) -> list[str]:
    return ["calibrate", "--think-budgets", "0,64", "--url", served_url, "--data", str(data), "--out", str(out),
            "--warmup", "0", "--audit", "0", *flags]


def test_url_mode_compares_the_cache_default_with_the_servers(tmp_path, capsys):
    """--url mode: the run's --cache (on unless --no-cache) must equal the attached server's. A server started with
    the new default passes with no flag; one started with --no-cache needs `calibrate --no-cache`, and the refusal
    says so."""
    data = write_data(tmp_path / "d.jsonl", 4)
    on = _served_in(tmp_path / "on")
    try:
        assert main(_run_url(on.url, data, tmp_path / "a")) == 0, capsys.readouterr().err
        assert json.loads((tmp_path / "a" / "m.think.json").read_text())["comparable"]["cache"] is True
        assert main(_run_url(on.url, data, tmp_path / "b", "--no-cache")) == 2
        err = capsys.readouterr().err
        assert "the server's --cache is on and this run's is off" in err and "Drop --no-cache" in err
    finally:
        on.stop()
    off = _served_in(tmp_path / "off", "--no-cache")
    try:
        assert main(_run_url(off.url, data, tmp_path / "c")) == 2
        err = capsys.readouterr().err
        assert "the server's --cache is off and this run's is on" in err and "Pass --no-cache" in err
        assert main(_run_url(off.url, data, tmp_path / "c", "--no-cache")) == 0, capsys.readouterr().err
        prof = json.loads((tmp_path / "c" / "m.think.json").read_text())
        assert prof["comparable"]["cache"] is False and prof["provenance"]["cache_prompt"] is False
        assert "cache_delimiter" not in prof["provenance"]                # present only when true, as before
        assert " --no-cache" in capsys.readouterr().out                    # the serve line reproduces the measurement
    finally:
        off.stop()


def test_resuming_a_cache_off_run_without_a_flag_names_no_cache(tmp_path, capsys):
    """Every run profile written before --cache became the default has `comparable.cache` false. Resumed with no cache flag (now on),
    its header differs; the refusal names the fix, online (the raw header) and offline (the refit candidates)."""
    data = write_data(tmp_path / "d.jsonl", 4)
    out = tmp_path / "think"
    off = _served_in(tmp_path / "off", "--no-cache")
    try:
        assert main(_run_url(off.url, data, out, "--no-cache")) == 0, capsys.readouterr().err
    finally:
        off.stop()
    capsys.readouterr()
    on = _served_in(tmp_path / "on")
    try:
        assert main(_run_url(on.url, data, out, "--resume")) == 2
        err = capsys.readouterr().err
        assert "refusing to resume" in err and "pass --no-cache to resume a cache-off run" in err
    finally:
        on.stop()
    assert main(_run_url(DEAD, data, out, "--resume")) == 2                # server down: no candidate for cache on
    assert "pass --no-cache to resume a cache-off run" in capsys.readouterr().err
    assert main(_run_url(DEAD, data, out, "--resume", "--no-cache")) == 0  # with the flag: the offline refit
    assert "--resume: refitting m.raw.jsonl" in capsys.readouterr().err


def test_serve_command_emits_no_cache_for_a_cache_off_profile():
    prof = {"comparable": {"cache": False}, "environment": {}}
    assert ct.serve_command(prof, 0, "p.think.json").endswith("--think-budget 0 --no-cache")
    prof["comparable"]["cache"] = True                                     # the default: nothing to add
    assert ct.serve_command(prof, 0, "p.think.json") == \
        "callo serve --upstream URL --think-profile p.think.json --think-budget 0"


def test_the_in_process_serve_gets_the_runs_cache_either_way(tmp_path, monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr(ct, "precheck_upstream", lambda upstream: None)
    real = callo.cli.split_passthrough
    monkeypatch.setattr(callo.cli, "split_passthrough", lambda argv: (seen.append(list(argv)), real(argv))[1])
    monkeypatch.setattr(callo.cli, "serve", lambda args, ready, stop: 1)   # exits at once: a Refusal, argv recorded
    for cache, flag in ((True, "--cache"), (False, "--no-cache")):
        with pytest.raises(Refusal), ct.served_in_process("http://127.0.0.1:9", cache=cache, log=lambda s: None):
            pass
        assert seen[-1][-1] == flag
