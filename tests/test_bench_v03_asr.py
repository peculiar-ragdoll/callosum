"""The transcription acceptance bench: the utterance set built with a stdlib
synth, the multipart encoder against callo's own splitter, WER, gate T1 on synthetic rows, the doc, the live-run hygiene
(stop handlers, run.pid, provenance, the set's sha checks, refusals), and `main` end to end on the fakes (no GPU, no
llama-server, no model, no `say`)."""
import base64
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from collections import Counter

import pytest
from fake_upstream import MEDIA_MARKER, FakeUpstream
from helpers import tiny_gguf
from test_bench_v02 import AttachedCallo

from bench import acceptance, build_asr_set, lib
from bench import acceptance_v03_asr as asr
from callo.av import encode_wav, tone_samples
from callo.transcribe import split_form


def tone_synth(text, voice):
    """A distinct one-second tone per sentence: the fakes map each clip back to its text by sha256."""
    hz = 100 + 20 * build_asr_set.SENTENCES.index(text)
    return encode_wav(tone_samples(hz=hz, rate=build_asr_set.RATE, seconds=1), rate=build_asr_set.RATE)


def _items(d):
    """The validated items (`load_manifest`), as main reads them."""
    return asr.load_manifest(d)[0]["items"]


def make_set(d):
    return build_asr_set.build(d, tone_synth, about={"synth": "tone"})


@pytest.fixture
def hermetic(monkeypatch):
    """No real ps, git or llama-server --version under pytest."""
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(asr, "code_state", lambda: {"commit": "c0ffee", "dirty": False})
    monkeypatch.setattr(asr, "_llama_server_version", lambda binary: "version: 1 (test)")


def test_the_set_is_20_fixed_sentences_in_the_four_say_voices(tmp_path):
    m = make_set(tmp_path)
    assert build_asr_set.VOICES == ("Samantha", "Daniel", "Karen", "Moira")
    assert len(m["items"]) == 20 == len(set(build_asr_set.SENTENCES))
    assert [it["voice"] for it in m["items"][:5]] == ["Samantha", "Daniel", "Karen", "Moira", "Samantha"]
    assert m["items"][0] == {"id": "u01", "file": "u01.wav", "text": build_asr_set.SENTENCES[0], "voice": "Samantha",
                             "sha256": hashlib.sha256((tmp_path / "u01.wav").read_bytes()).hexdigest()}
    assert json.loads((tmp_path / "manifest.json").read_text()) == m and m["synth"] == "tone" and m["rate"] == 24000
    assert [it["id"] for it in _items(tmp_path)] == [f"u{i:02d}" for i in range(1, 21)]


def test_main_builds_the_set_with_say_then_afconvert(tmp_path, monkeypatch):      # both commands stubbed
    calls = []

    def run(argv, check, timeout):
        calls.append(argv)
        assert check is True and timeout == build_asr_set.TIMEOUT       # never an unbounded wait
        if argv[0] == "afconvert":                       # the stub "converts": the WAV is the tone of the spoken text
            with open(argv[-1], "wb") as f:
                f.write(tone_synth(calls[-2][-1], calls[-2][2]))

    monkeypatch.setattr(build_asr_set.subprocess, "run", run)
    assert build_asr_set.main([str(tmp_path / "set")]) == 0
    assert len(calls) == 40
    say, conv = calls[0], calls[1]
    assert say[:3] == ["say", "-v", "Samantha"] and say[3] == "-o" and say[4].endswith("u.aiff")
    assert say[5] == build_asr_set.SENTENCES[0] and calls[2][2] == "Daniel"
    assert conv == ["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", say[4], conv[-1]]
    m = json.loads((tmp_path / "set" / "manifest.json").read_text())
    assert m["synth"] == "say" and m["voices"] == list(build_asr_set.VOICES) and "macos" in m
    assert len(_items(tmp_path / "set")) == 20


def test_is_bench_run_is_the_shared_run_pid_rule_with_this_benchs_module(monkeypatch):
    """`lib.is_bench_run` (its own cases are in test_bench_lib.py) with BENCH_MODULE and this module's `_pid_args`
    seam. The liveness check is signal 0 on this process's own pid: never delivered."""
    monkeypatch.setattr(asr, "_pid_args", lambda pid: "python -m bench.acceptance_v03_asr --set s --out o")
    assert asr.is_bench_run(os.getpid()) is True
    monkeypatch.setattr(asr, "_pid_args", lambda pid: "python -m bench.acceptance_v03_think --out o")
    assert asr.is_bench_run(os.getpid()) is False


def _rewrite_manifest(d, change):
    m = json.loads((d / "manifest.json").read_text())
    (d / "manifest.json").write_text(json.dumps(change(m)))


def _item(i, **kw):
    def change(m):
        m["items"][i].update(kw)
        return m
    return change


def _drop(i, key):
    def change(m):
        del m["items"][i][key]
        return m
    return change


@pytest.mark.parametrize("change, says", [
    (lambda m: m["items"], "must be a JSON object"),
    (lambda m: {**m, "items": {"u01": 1}}, "must list 20 items"),
    (lambda m: {**m, "items": [*m["items"][:3], "u04.wav", *m["items"][4:]]}, "item 4 is not an object"),
    (_drop(0, "id"), "item 1: `id` must be a non-empty string"),
    (_item(1, id=7), "item 2: `id` must be a non-empty string"),
    (_item(2, text=""), "item 3 (u03): `text` must be a non-empty string"),
    (_drop(2, "text"), "item 3 (u03): `text` must be a non-empty string"),
    (_item(4, text=["x"]), "item 5 (u05): `text` must be a non-empty string"),
    (_drop(5, "sha256"), "item 6 (u06): `sha256` must be 64 lower-case hex digits"),
    (_item(5, sha256="abc"), "item 6 (u06): `sha256` must be 64 lower-case hex digits"),
    (_item(6, file="../u07.wav"), "item 7 (u07): `file` must be a plain file name"),
    (_item(6, file=None), "item 7 (u07): `file` must be a plain file name"),
    (_item(7, id="u01"), "duplicate ids: u01"),
], ids=["list", "items_not_list", "item_not_dict", "no_id", "id_not_str", "empty_text", "no_text", "text_not_str",
        "no_sha", "short_sha", "file_escapes", "file_none", "duplicate_id"])
def test_a_malformed_manifest_refuses_the_set(tmp_path, change, says):
    make_set(tmp_path)
    _rewrite_manifest(tmp_path, change)
    with pytest.raises(ValueError, match=re.escape(says)):
        _items(tmp_path)


def test_main_refuses_a_malformed_manifest_before_launching(tmp_path, monkeypatch, capsys, hermetic):
    make_set(tmp_path / "set")
    _rewrite_manifest(tmp_path / "set", _drop(2, "text"))
    monkeypatch.setattr(asr, "_launch", lambda *a, **k: pytest.fail("launched"))
    assert asr.main(["--set", str(tmp_path / "set"), "--out", str(tmp_path / "o"), "--doc", str(tmp_path / "d.md")]) == 2
    assert "refusing to start: the set" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_load_manifest_hashes_the_bytes_it_parsed(tmp_path):
    make_set(tmp_path)
    manifest, sha = asr.load_manifest(tmp_path)
    assert sha == hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    assert manifest == json.loads((tmp_path / "manifest.json").read_text())


def test_a_missing_or_changed_clip_refuses_the_set(tmp_path):
    make_set(tmp_path)
    (tmp_path / "u03.wav").write_bytes(b"RIFF changed")
    with pytest.raises(ValueError, match="u03.wav: sha256 .* does not match the manifest"):
        _items(tmp_path)
    make_set(tmp_path)                                                   # whole again
    (tmp_path / "u07.wav").unlink()
    with pytest.raises(ValueError, match="u07.wav is missing"):
        _items(tmp_path)


def test_the_form_is_one_callos_splitter_reads_back():
    ctype, body = asr.encode_form({"temperature": "0", "max_tokens": "128"}, b"RIFF\r\n--x\r\n")
    assert [(p.name, p.filename, bytes(p.content)) for p in split_form(ctype, body)] == [
        (b"file", b"clip.wav", b"RIFF\r\n--x\r\n"), (b"temperature", b"", b"0"), (b"max_tokens", b"", b"128")]


def test_wer():
    assert asr.wer("The cat sat.", "the cat, sat") == 0.0
    assert asr.wer("a b c d", "a x c") == 0.5              # one substitution and one deletion over four words
    assert asr.wer("", "x") == 1.0


def rows(n=20):
    out = []
    for i in range(1, n + 1):
        for arm in asr.ARMS:
            out.append({"id": f"u{i:02d}", "arm": arm, "status": 200, "text": f"text {i}", "wer": 0.0,
                        "input_tokens": 100 + i, "cached_tokens": 0, "slot": "2" if arm == "callo" else None,
                        "ms": 10.0 * i, "prompt_sha256": f"p{i}", "prompt_files": 1})
    return out


def edit(rs, id_, arm, **kw):
    next(r for r in rs if r["id"] == id_ and r["arm"] == arm).update(kw)
    return rs


def test_t1_passes_on_matching_arms():
    g = asr.t1(rows(), reserved=[0])
    assert g["verdict"] == "PASS" and all(c["pass"] and c["n"] == 20 for c in g["checks"].values())


@pytest.mark.parametrize("arm_edit, check", [
    (("u03", "callo", {"prompt_sha256": "other"}), "prompts"),
    (("u03", "native", {"prompt_files": 2}), "prompts"),             # two prompt files: not the one request's prompt
    (("u04", "callo", {"input_tokens": 99}), "input_tokens"),
    (("u05", "native", {"cached_tokens": 7}), "cached_tokens_zero"),
    (("u06", "callo", {"slot": "0"}), "pinned"),                     # the reserved slot
    (("u06", "callo", {"slot": "-"}), "pinned"),                     # unpinned
])
def test_one_failed_check_fails_t1(arm_edit, check):
    id_, arm, kw = arm_edit
    g = asr.t1(edit(rows(), id_, arm, **kw), reserved=[0])
    assert g["verdict"] == "FAIL" and not g["checks"][check]["pass"] and g["checks"][check]["n"] == 19


def test_19_of_20_identical_texts_pass_and_18_fail_with_their_wer():
    g = asr.t1(edit(rows(), "u01", "callo", text="text one"), reserved=[0])
    assert g["verdict"] == "PASS" and g["checks"]["text"]["n"] == 19
    assert g["checks"]["text"]["mismatches"] == [{"id": "u01", "callo": "text one", "native": "text 1",
                                                  "wer_callo": 0.0, "wer_native": 0.0, "wer_between": 0.5}]
    g = asr.t1(edit(edit(rows(), "u01", "callo", text="x"), "u02", "callo", text="y"), reserved=[0])
    assert g["verdict"] == "FAIL" and len(g["checks"]["text"]["mismatches"]) == 2


def test_missing_or_empty_evidence_is_an_error_never_a_pass():
    assert asr.t1(rows(19), reserved=[0])["verdict"] == "ERROR"
    g = asr.t1(edit(rows(), "u09", "native", status=500, text=None), reserved=[0])
    assert g["verdict"] == "ERROR" and g["why"] == ["u09: the native arm has no non-empty 200 transcript"]
    empty = rows()
    for r in empty:
        r["text"] = ""                     # both arms "hear" nothing: "" == "" must not pass as identical text
    g = asr.t1(empty, reserved=[0])
    assert g["verdict"] == "ERROR" and len(g["why"]) == 40


def test_transcripts_unrelated_to_the_spoken_text_are_an_error():                 # the sanity bound
    rs = rows()
    for r in rs:
        r["wer"] = 0.9                     # both arms agree with each other, but not with what `say` said
    g = asr.t1(rs, reserved=[0])
    assert g["verdict"] == "ERROR" and "checks" in g
    assert g["why"][0].startswith("the callo arm's mean WER against the spoken text is 0.90 > 0.5")


def test_preconditions_read_the_transcription_block():
    assert asr.preconditions({"transcription": {"translate": True, "reason": None}}) == []
    assert asr.preconditions({"transcription": {"translate": False, "reason": "render: x"}}) == [
        "translation is off: render: x"]
    assert asr.preconditions({})[0].startswith("/callo/stats has no transcription block")


def test_the_doc_states_the_verdict_and_every_check():
    run = {"started": "s", "finished": "f", "build_info": "b1-x", "llama_server_version": "version: 1 (test)",
           "callo": "0.2.0", "commit": "abc", "dirty": False, "model": "/m/g.gguf", "mmproj": "/m/p.gguf", "n_slots": 4,
           "reserved": [0], "kv": "per_slot", "set": "/set", "manifest_sha256": "f" * 64, "synth": {"synth": "tone"}}
    summary = {"run": run, "conditions": {"translated": 20, "unrouted": {}, "s1_readouts_delta": 0, "ok": True,
                                          "problems": []},
               "t1": asr.t1(edit(rows(), "u01", "callo", text="text | one"), reserved=[0]),
               "descriptive": asr.descriptive(rows())}
    md = asr.render_md(summary)
    assert "**T1: PASS**" in md and "| prompts byte-identical" in md and "| 19/20 | >= 19/20 | PASS |" in md
    assert "text \\| one" in md and "| callo | 0.000 |" in md                  # a pipe in a transcript is escaped
    assert "`abc` (clean)" in md and "version: 1 (test)" in md and "f" * 64 in md
    assert "LFM2 preset branch is unit-tested only" in md


def test_main_on_the_fakes_runs_both_arms_and_passes_t1(tmp_path, monkeypatch, hermetic):
    set_dir, out, doc = tmp_path / "set", tmp_path / "out", tmp_path / "v0.3-asr.md"
    manifest = make_set(set_dir)
    said = {it["sha256"]: it["text"] for it in manifest["items"]}
    up = FakeUpstream(audio=True, slot_save=True).start()
    up.props["model_path"] = str(tiny_gguf(tmp_path / "m.gguf"))
    erased_at_launch = []

    def log_prompt(text):                     # llama-server's --log-prompts-dir: one file per completion request
        (out / "prompts" / f"{time.time_ns()}.txt").write_text(text)

    def native(u, rec, h):
        audio = next(p for p in split_form(rec.headers["content-type"], rec.body) if p.name == b"file").content
        log_prompt("Transcribe audio to text" + MEDIA_MARKER)
        h._reply(200, {"type": "transcript.text.done", "text": said[hashlib.sha256(audio).hexdigest()],
                       "usage": {"type": "tokens", "input_tokens": 50, "output_tokens": 5, "total_tokens": 55,
                                 "input_tokens_details": {"cached_tokens": 0}}})

    def chat(u, rec, h):
        content = rec.json["messages"][-1]["content"]
        log_prompt("".join(p["text"] if p["type"] == "text" else MEDIA_MARKER for p in content))
        audio = base64.b64decode(content[-1]["input_audio"]["data"])
        h._reply(200, {"choices": [{"message": {"content": said[hashlib.sha256(audio).hexdigest()]}}],
                       "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55,
                                 "prompt_tokens_details": {"cached_tokens": 0}}})

    up.routes["/v1/audio/transcriptions"] = native
    up.routes["/v1/chat/completions"] = chat
    launched = []

    def launch(out_dir, **kw):
        launched.append(kw)
        c = AttachedCallo(up, tmp_path)
        erased_at_launch.append(len(up.erased))           # callo's own startup erases are not the bench's
        return c

    monkeypatch.setattr(asr, "_launch", launch)
    try:
        rc = asr.main(["--set", str(set_dir), "--out", str(out), "--doc", str(doc)])
    finally:
        up.stop()
    summary = json.loads((out / "summary.json").read_text())
    assert rc == 0 and summary["t1"]["verdict"] == "PASS", summary["t1"]
    assert summary["conditions"] == {"translated": 20, "unrouted": {}, "s1_readouts_delta": 0, "ok": True,
                                     "problems": []}
    run = summary["run"]
    assert (run["commit"], run["dirty"], run["llama_server_version"]) == ("c0ffee", False, "version: 1 (test)")
    assert run["manifest_sha256"] == hashlib.sha256((set_dir / "manifest.json").read_bytes()).hexdigest()
    assert launched[0]["prompt_dir"] == out / "prompts"
    bench_erases = up.erased[erased_at_launch[0]:]
    assert len(bench_erases) == 2 * 20 * 4 and Counter(bench_erases) == {0: 40, 1: 40, 2: 40, 3: 40}
    assert summary["warnings"] == []
    assert len((out / "rows.jsonl").read_text().splitlines()) == 40 and "**T1: PASS**" in doc.read_text()
    assert not (out / "run.pid").exists()                                     # removed at exit


class StubCallo:
    url = upstream = "http://127.0.0.1:9"
    build_info = "stub"

    def __init__(self, finished):
        self.finished = finished

    def stats(self):
        return {"transcription": {"translate": True, "reason": None}, "total_slots": 4, "router": {"reserved": [0]},
                "s1": {"readouts": 0}, "counters": {}}

    def finish(self):
        self.finished.append(True)


def test_a_stop_signal_mid_run_still_stops_the_server_and_clears_run_pid(tmp_path, monkeypatch, hermetic):
    make_set(tmp_path / "set")
    out, finished, seen = tmp_path / "o", [], {}

    def stopped(c, item, *a):
        seen["handler"] = signal.getsignal(signal.SIGTERM)
        seen["hup"] = signal.getsignal(signal.SIGHUP)
        seen["pid"] = asr._pid_in(out / "run.pid")
        raise SystemExit(128 + signal.SIGTERM)             # what the installed stop handler raises on SIGTERM

    monkeypatch.setattr(asr, "_launch", lambda *a, **k: StubCallo(finished))
    monkeypatch.setattr(asr, "run_item", stopped)
    saved = _signals_saved()
    try:
        signal.signal(signal.SIGHUP, signal.SIG_DFL)       # a terminal's SIGHUP, even if pytest itself runs under nohup
        before = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit):
            asr.main(["--set", str(tmp_path / "set"), "--out", str(out), "--doc", str(tmp_path / "d.md")])
        assert seen == {"handler": asr._stop_run, "hup": asr._stop_run, "pid": os.getpid()}   # not uv's
        assert finished == [True] and not (out / "run.pid").exists() and signal.getsignal(signal.SIGTERM) is before
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_DFL                              # restored too
    finally:
        _restore(saved)


@pytest.mark.parametrize("args, says", [
    ("/usr/bin/python3 -m bench.acceptance_v03_asr --set s --out o", "another run of this bench (pid "),
    ("/usr/bin/vim notes.txt", "which is not a bench run: check it and remove run.pid by hand"),
    (None, "which is not a bench run: check it and remove run.pid by hand"),        # ps could not read it
], ids=["bench_run", "other_process", "unreadable"])
def test_a_live_run_pid_refuses_and_is_left_alone(tmp_path, monkeypatch, capsys, hermetic, args, says):
    make_set(tmp_path / "set")
    out = tmp_path / "o"
    out.mkdir()
    (out / "run.pid").write_text(f"{os.getpid()}\n")                      # a live pid owns this --out
    monkeypatch.setattr(asr, "_pid_args", lambda pid: args)
    monkeypatch.setattr(asr, "_launch", lambda *a, **k: pytest.fail("launched"))
    assert asr.main(["--set", str(tmp_path / "set"), "--out", str(out), "--doc", str(tmp_path / "d.md")]) == 2
    err = capsys.readouterr().err
    assert says in err and ("stop it" in err) == (args is not None and "acceptance_v03_asr" in args)
    assert (out / "run.pid").read_text() == f"{os.getpid()}\n"


def _ps_fails():
    raise RuntimeError("could not list processes: ps exited 1")


@pytest.mark.parametrize("gpu", [lambda: [(123, "llama-server -m x")], _ps_fails], ids=["busy", "ps_failed"])
def test_main_refuses_a_busy_or_unknown_gpu_before_launching(tmp_path, monkeypatch, hermetic, gpu):
    make_set(tmp_path / "set")
    monkeypatch.setattr(lib, "running_gpu_servers", gpu)
    monkeypatch.setattr(asr, "_launch", lambda *a, **k: pytest.fail("launched"))
    assert asr.main(["--set", str(tmp_path / "set"), "--out", str(tmp_path / "o"), "--doc", str(tmp_path / "d.md")]) == 2


def test_main_refuses_a_changed_set_before_launching(tmp_path, monkeypatch, hermetic):
    make_set(tmp_path / "set")
    (tmp_path / "set" / "u05.wav").write_bytes(b"RIFF changed")
    monkeypatch.setattr(asr, "_launch", lambda *a, **k: pytest.fail("launched"))
    assert asr.main(["--set", str(tmp_path / "set"), "--out", str(tmp_path / "o"), "--doc", str(tmp_path / "d.md")]) == 2


@pytest.mark.parametrize("arm_edit, says", [
    (("u03", "callo", {"prompt_files": 0, "prompt_sha256": None}),
     "u03: the callo arm logged no prompt file (--log-prompts-dir)"),
    (("u04", "native", {"input_tokens": None}), "u04: the native arm's reply has no usage.input_tokens"),
    (("u05", "callo", {"cached_tokens": None}),
     "u05: the callo arm's reply has no usage.input_tokens_details.cached_tokens"),
], ids=["no_prompt_file", "no_input_tokens", "no_cached_tokens"])
def test_missing_instrument_evidence_is_an_error_not_a_fail(arm_edit, says):
    id_, arm, kw = arm_edit
    assert asr.t1(edit(rows(), id_, arm, **kw), reserved=[0]) == {"verdict": "ERROR", "why": [says]}


def test_conditions_of_names_every_broken_condition():
    before = {"counters": {"transcription_translated": 2}, "s1": {"readouts": 5}}
    after = {"counters": {"transcription_translated": 21, "transcription_unrouted_render": 1, "other": 3},
             "s1": {"readouts": 6}}
    assert asr.conditions_of(before, after) == {
        "translated": 19, "unrouted": {"transcription_unrouted_render": 1}, "s1_readouts_delta": 1, "ok": False,
        "problems": ["19 translated transcriptions, expected 20",
                     "transcriptions passed through: {'transcription_unrouted_render': 1}",
                     "/v1/systemone readouts during the run: +1"]}


class SeqCallo(StubCallo):
    """StubCallo whose /callo/stats replies come from a list: an exception in it is raised (the closing stats failing)."""

    def __init__(self, finished, stats):
        super().__init__(finished)
        self._stats = list(stats)

    def stats(self):
        s = self._stats.pop(0)
        if isinstance(s, BaseException):
            raise s
        return s


def _stats(translated, readouts=0):
    return {"transcription": {"translate": True, "reason": None}, "total_slots": 4, "router": {"reserved": [0]},
            "s1": {"readouts": readouts}, "counters": {"transcription_translated": translated}}


def _rows_of(c, item, *a):
    return [r for r in rows() if r["id"] == item["id"]]


def _main(tmp_path):
    return asr.main(["--set", str(tmp_path / "set"), "--out", str(tmp_path / "o"), "--doc", str(tmp_path / "d.md")])


def test_a_pass_with_a_broken_condition_is_an_error(tmp_path, monkeypatch, hermetic):
    make_set(tmp_path / "set")
    sha = hashlib.sha256((tmp_path / "set" / "manifest.json").read_bytes()).hexdigest()
    finished = []

    def launch(*a, **k):                     # the manifest changes after main loaded it: the summary keeps what it ran
        _rewrite_manifest(tmp_path / "set", lambda m: {**m, "synth": "swapped"})
        return SeqCallo(finished, [_stats(0), _stats(19)])

    monkeypatch.setattr(asr, "_launch", launch)
    monkeypatch.setattr(asr, "run_item", _rows_of)
    assert _main(tmp_path) == 1 and finished == [True]
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert s["t1"]["verdict"] == "ERROR" and s["t1"]["why"] == ["19 translated transcriptions, expected 20"]
    assert all(c["pass"] for c in s["t1"]["checks"].values())               # the checks are still reported
    assert s["run"]["manifest_sha256"] == sha and s["run"]["synth"] == {"rate": 24000, "synth": "tone"}
    md = (tmp_path / "d.md").read_text()
    assert "**T1: ERROR**" in md and "- 19 translated transcriptions, expected 20" in md and "| 19/20 |" not in md


def _changes_u04(tmp_path):
    def post(url, ctype, body, **kw):
        (tmp_path / "set" / "u04.wav").write_bytes(b"RIFF changed")
        return {"status": 200, "headers": {"x-callo-slot": "1"}, "json": {"text": "t"}, "ms": 1.0}
    return post


def _refused(url, ctype, body, **kw):
    if "u03" in _refused.current:
        raise ConnectionRefusedError(61, "Connection refused")
    return {"status": 200, "headers": {"x-callo-slot": "1"}, "json": {"text": "t"}, "ms": 1.0}


@pytest.mark.parametrize("case, why", [
    ("transport", "u03: ConnectionRefusedError: [Errno 61] Connection refused"),
    ("disconnect", "u01: RemoteDisconnected: gone"),
    ("erase", "u01: ItemError: erasing slot 0 on http://127.0.0.1:9 returned HTTP 501: {'error': 'x'}"),
    ("changed_wav", "u04: ItemError: u04.wav changed during the run (sha256 no longer matches the manifest)"),
    ("closing_stats", "the closing /callo/stats: URLError: <urlopen error refused>"),
], ids=["transport", "disconnect", "erase", "changed_wav", "closing_stats"])
def test_an_error_mid_run_is_an_error_summary_and_the_server_is_stopped(tmp_path, monkeypatch, hermetic, case, why):
    """A transport, erase or changed-WAV error (or the closing stats failing) mid-run ends the run as ERROR:
    summary.json and the doc name the item and the error, rows so far are kept, finish() ran, exit 1."""
    import http.client
    import urllib.error
    make_set(tmp_path / "set")
    finished = []
    stats = [_stats(0), urllib.error.URLError("refused") if case == "closing_stats" else _stats(20)]
    monkeypatch.setattr(asr, "_launch", lambda *a, **k: SeqCallo(finished, stats))
    monkeypatch.setattr(lib, "post", lambda url, payload: (501, {}, {"error": "x"}) if case == "erase" else (200, {}, {}))
    real_run_item = asr.run_item

    def run_item(c, item, *a):
        _refused.current = item["id"]
        return real_run_item(c, item, *a)

    monkeypatch.setattr(asr, "run_item", run_item)
    if case == "transport":
        monkeypatch.setattr(asr, "post_form", _refused)
    elif case == "disconnect":
        def gone(*a, **k):
            raise http.client.RemoteDisconnected("gone")
        monkeypatch.setattr(asr, "post_form", gone)
    elif case == "changed_wav":
        monkeypatch.setattr(asr, "post_form", _changes_u04(tmp_path))
    else:
        monkeypatch.setattr(asr, "post_form", lambda *a, **k: {"status": 200, "headers": {"x-callo-slot": "1"},
                                                               "json": {"text": "t"}, "ms": 1.0})
    assert _main(tmp_path) == 1 and finished == [True] and not (tmp_path / "o" / "run.pid").exists()
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert s["t1"]["verdict"] == "ERROR" and s["t1"]["why"][0] == why
    assert s["conditions"]["ok"] is False and s["run"]["commit"] == "c0ffee"
    ran = {"transport": 2, "disconnect": 0, "erase": 0, "changed_wav": 3, "closing_stats": 20}[case]
    assert s["t1"]["why"][1] == f"{ran} of 20 utterances ran before it"
    kept = tmp_path / "o" / "rows.jsonl"                                      # the rows so far are kept
    assert (len(kept.read_text().splitlines()) if kept.exists() else 0) == 2 * ran
    md = (tmp_path / "d.md").read_text()
    assert "**T1: ERROR**" in md and f"- {why}" in md and "## Conditions" in md


def test_weak_provenance_is_a_recorded_warning_and_the_run_goes_on(tmp_path, monkeypatch):
    make_set(tmp_path / "set")
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(asr, "code_state", lambda: {"commit": None, "dirty": None, "error": "FileNotFoundError: git"})
    monkeypatch.setattr(asr, "_llama_server_version", lambda binary: "unavailable: FileNotFoundError: llama-server")
    monkeypatch.setattr(asr, "_launch", lambda *a, **k: SeqCallo([], [_stats(0), _stats(20)]))
    monkeypatch.setattr(asr, "run_item", _rows_of)
    assert _main(tmp_path) == 0
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert s["t1"]["verdict"] == "PASS" and s["warnings"] == [
        "the commit is unknown (FileNotFoundError: git): the code this run measured is not pinned",
        "`llama-server --version` failed: unavailable: FileNotFoundError: llama-server"]
    md = (tmp_path / "d.md").read_text()
    assert "Warnings:" in md and "- the commit is unknown (FileNotFoundError: git)" in md and "(unknown)" in md


def test_the_sighup_stop_handler_is_one_shot_and_raises_system_exit():
    saved = {s: signal.getsignal(s) for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(SystemExit) as e:
            asr._stop_run(signal.SIGHUP, None)
        assert e.value.code == 128 + signal.SIGHUP
        assert all(signal.getsignal(s) == signal.SIG_IGN for s in saved)   # a second signal cannot cut cleanup short
    finally:
        for s, h in saved.items():
            signal.signal(s, h)


def _signals_saved():
    return {s: signal.getsignal(s) for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)}


def _restore(saved):
    for s, h in saved.items():
        signal.signal(s, h)


def test_term_or_int_first_also_ignores_hup_so_cleanup_cannot_be_cut_short():
    saved = _signals_saved()
    try:
        signal.signal(signal.SIGHUP, asr._stop_run)                   # as installed for the run
        with pytest.raises(SystemExit) as e:
            asr._stop_run(signal.SIGTERM, None)
        assert e.value.code == 128 + signal.SIGTERM                   # what acceptance_v02's _stop_signal raises
        assert all(signal.getsignal(s) == signal.SIG_IGN for s in saved)
    finally:
        _restore(saved)


def test_nohup_keeps_ignoring_hangups(tmp_path, monkeypatch, hermetic):
    """An inherited SIG_IGN on SIGHUP (`nohup`) is left alone: unit level, then through main."""
    saved = _signals_saved()
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        previous = asr._install_stop_handlers_with_hup()
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN and signal.SIGHUP not in previous
        assert signal.getsignal(signal.SIGTERM) is asr._stop_run
        _restore(saved)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        make_set(tmp_path / "set")
        seen = []
        monkeypatch.setattr(asr, "_launch", lambda *a, **k: SeqCallo([], [_stats(0), _stats(20)]))
        monkeypatch.setattr(asr, "run_item", lambda c, item, *a: seen.append(signal.getsignal(signal.SIGHUP))
                            or _rows_of(c, item))
        assert _main(tmp_path) == 0 and set(seen) == {signal.SIG_IGN}
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN and signal.getsignal(signal.SIGTERM) is saved[
            signal.SIGTERM]
    finally:
        _restore(saved)


def test_a_success_without_router_reserved_is_an_error(tmp_path, monkeypatch, hermetic):
    make_set(tmp_path / "set")
    bare = [{k: v for k, v in _stats(n).items() if k != "router"} for n in (0, 20)]
    monkeypatch.setattr(asr, "_launch", lambda *a, **k: SeqCallo([], bare))
    monkeypatch.setattr(asr, "run_item", _rows_of)
    assert _main(tmp_path) == 1
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert s["t1"] == {"verdict": "ERROR", "why": [
        "/callo/stats has no router.reserved list: a reserved slot cannot be told from an agent slot"]}
    assert s["run"]["reserved"] is None and "**T1: ERROR**" in (tmp_path / "d.md").read_text()


def test_a_launch_that_fails_is_an_error_summary(tmp_path, monkeypatch, hermetic):
    """Callo's "callo did not start" (a RuntimeError; Callo reaped its own child) is ERROR with a summary, exit 1."""
    make_set(tmp_path / "set")

    def launch(*a, **k):
        raise RuntimeError("callo did not start (rc=1):\nllama-server: cannot open the mmproj")

    monkeypatch.setattr(asr, "_launch", launch)
    assert _main(tmp_path) == 1 and not (tmp_path / "o" / "run.pid").exists()
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert s["t1"]["verdict"] == "ERROR" and s["t1"]["why"] == [
        "the launch: RuntimeError: callo did not start (rc=1):\nllama-server: cannot open the mmproj",
        "0 of 20 utterances ran before it"]
    assert s["run"]["commit"] == "c0ffee" and s["run"]["build_info"] is None and s["run"]["reserved"] == []
    assert "- the launch: RuntimeError: callo did not start (rc=1): llama-server" in (tmp_path / "d.md").read_text()


def test_a_runtime_error_in_the_item_loop_still_propagates(tmp_path, monkeypatch, hermetic):  # (c): launch only
    make_set(tmp_path / "set")
    finished = []
    monkeypatch.setattr(asr, "_launch", lambda *a, **k: SeqCallo(finished, [_stats(0), _stats(20)]))

    def bug(*a):
        raise RuntimeError("a bug")

    monkeypatch.setattr(asr, "run_item", bug)
    with pytest.raises(RuntimeError, match="a bug"):
        _main(tmp_path)
    assert finished == [True] and not (tmp_path / "o" / "summary.json").exists()


def test_a_stop_signal_right_after_the_launch_still_reaps_its_llama_server(tmp_path, monkeypatch, hermetic):
    """The Callo never reached main (the signal landed as `_launch` returned): main's `finally` reaps every llama-server
    pid file the launch wrote in its cache dir, never one that was there before. Both children are this test's own."""
    make_set(tmp_path / "set")
    cache = tmp_path / "o" / "callo" / "cache"
    cache.mkdir(parents=True)
    old = subprocess.Popen(["sleep", "60"], start_new_session=True)
    new = []
    try:
        (cache / "llama-server-9001.pid").write_text(f"{old.pid}\n")              # before the launch: not ours

        def launch(*a, **k):
            new.append(subprocess.Popen(["sleep", "60"], start_new_session=True))
            (cache / "llama-server-9002.pid").write_text(f"{new[0].pid}\n")
            raise SystemExit(128 + signal.SIGTERM)

        monkeypatch.setattr(asr, "_launch", launch)
        monkeypatch.setattr(acceptance, "_pid_args", lambda pid: "/x/llama-server -m m.gguf")   # the args check
        with pytest.raises(SystemExit):
            _main(tmp_path)
        assert new[0].wait(timeout=20) is not None and old.poll() is None
    finally:
        for p in (old, *new):
            if p.poll() is None:
                p.kill()
                p.wait(timeout=10)


# ---- readout counts, the inherited SIGHUP, hung converters, the set's docstring ---------------------------------------

def test_conditions_of_compares_deltas_and_a_missing_readout_count_is_a_problem():
    before = {"counters": {"transcription_translated": 3, "transcription_unrouted_render": 2}, "s1": {"readouts": 5}}
    after = {"counters": {"transcription_translated": 23, "transcription_unrouted_render": 2}, "s1": {"readouts": 5}}
    assert asr.conditions_of(before, after) == {"translated": 20, "unrouted": {}, "s1_readouts_delta": 0, "ok": True,
                                                "problems": []}                   # passed through before the run
    for b, a in ((before, {**after, "s1": {}}), ({**before, "s1": None}, after), (before, {"counters": {}})):
        c = asr.conditions_of(b, a)
        assert c["s1_readouts_delta"] is None and not c["ok"]
        assert "/callo/stats has no s1.readouts: the /v1/systemone condition cannot be checked" in c["problems"]


def test_a_run_without_s1_readouts_is_an_error_not_a_traceback(tmp_path, monkeypatch, hermetic):
    make_set(tmp_path / "set")
    bare = [{k: v for k, v in _stats(n).items() if k != "s1"} for n in (0, 20)]
    monkeypatch.setattr(asr, "_launch", lambda *a, **k: SeqCallo([], bare))
    monkeypatch.setattr(asr, "run_item", _rows_of)
    assert _main(tmp_path) == 1
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert s["t1"]["verdict"] == "ERROR" and s["conditions"]["s1_readouts_delta"] is None
    assert "/callo/stats has no s1.readouts: the /v1/systemone condition cannot be checked" in s["t1"]["why"]


@pytest.mark.parametrize(("inherited", "says"), [(signal.SIG_IGN, "ignored (nohup)"), (signal.SIG_DFL, "stop the run")],
                         ids=["nohup", "terminal"])
def test_main_logs_and_records_the_inherited_sighup_disposition(tmp_path, monkeypatch, capsys, hermetic,
                                                                inherited, says):
    saved = _signals_saved()
    try:
        signal.signal(signal.SIGHUP, inherited)
        make_set(tmp_path / "set")
        monkeypatch.setattr(asr, "_launch", lambda *a, **k: SeqCallo([], [_stats(0), _stats(20)]))
        monkeypatch.setattr(asr, "run_item", _rows_of)
        assert _main(tmp_path) == 0
    finally:
        _restore(saved)
    assert f"hangups: {says}" in capsys.readouterr().err
    assert json.loads((tmp_path / "o" / "summary.json").read_text())["run"]["sighup"] == says


def test_a_say_or_afconvert_that_hangs_is_a_clear_error(tmp_path, monkeypatch, capsys):
    def run(argv, check, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(build_asr_set.subprocess, "run", run)
    assert build_asr_set.main([str(tmp_path / "set")]) == 1
    err = capsys.readouterr().err
    assert f"`say` did not finish within {build_asr_set.TIMEOUT:g} s" in err and not (tmp_path / "set" / "u01.wav").exists()


def test_the_set_docstring_does_not_call_the_pangram_a_harvard_sentence():
    assert "quick brown fox" in build_asr_set.__doc__ and build_asr_set.SENTENCES[-1].startswith("The quick brown fox")
