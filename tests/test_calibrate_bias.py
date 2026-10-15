"""callo calibrate --bias: the client against CalloStub (a stand-in for callo serve), and end to end
against callo serve on the planted-prior fake upstream."""
import hashlib
import io
import json

import numpy as np
import pytest
from fake_callo import KEY, CalloStub
from fake_upstream import FakeUpstream, PlantedBias
from helpers import Running, get_json, post_json, tiny_gguf

import callo.calibrate_bias as cb
from bench.bias_refit import refit
from callo.bias import builtin_set_bytes, debias, offsets_from_prior, read_set
from callo.calibrate_bias import Refusal, RunFailed, collect, parse_slot, post_readout
from callo.cli import main
from callo.errors import InvalidInput
from callo.guard import LANE
from callo.modelkey import load_model_key
from callo.profile import load_profile

PRIOR3 = [0.5, 0.3, 0.2]


def set_text(ns=(2, 3), per=4, *, labelled=True, sources=("syn",)):
    """`per` canonical questions for each n: multi-character keys, so the labels are A, B, ... in order."""
    lines = []
    for n in ns:
        for i in range(per):
            crit = {f"o{n}_{i}_{k}": None for k in range(n)}
            q = {"type": "choice", "instructions": "pick one", "criteria": crit}
            if labelled:
                q["label"] = f"o{n}_{i}_0"
            lines.append(json.dumps({"state": f"state {n}-{i}", "questions": {"q": q},
                                     "_meta": {"id": f"r{n}-{i}", "source": sources[i % len(sources)]}}))
    return "\n".join(lines) + "\n"


def _collect(stub, tmp_path, text=None, *, resume=False, allow_busy=False, raw=None, seed=0, out=None, sleeps=None,
             max_per_n=48):
    text = text or set_text()
    items = read_set(text).items
    sleeps = [] if sleeps is None else sleeps
    return items, collect(stub.url, items, raw_path=raw or tmp_path / "raw.jsonl", seed=seed, resume=resume,
                          allow_busy=allow_busy, set_name="t.jsonl", set_sha=hashlib.sha256(text.encode()).hexdigest(),
                          max_per_n=max_per_n, out=out or io.StringIO(), sleep=sleeps.append)


@pytest.fixture
def stub():
    s = CalloStub(PlantedBias(priors={3: PRIOR3}))
    yield s
    s.stop()


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _raw_meanp(raw_path, n):
    """prior_n from its definition, computed without callo.bias: every fitted question was read under all n
    rotations, so the mean of the question rows is the mean over every n-option readout in the raw file."""
    return list(np.mean([r["probs"] for r in _lines(raw_path)[1:] if r.get("type") == "readout" and r["n"] == n],
                        axis=0))


@pytest.mark.parametrize(("edit", "match"), [
    (lambda s: s.pop("model_key"), "older than v0.3"),
    (lambda s: s.update(model_key=None, model_key_error="cannot hash"), "could not compute .*cannot hash"),
    (lambda s: s.update(model_key=None, model_key_error=None), "could not compute .*no reason given"),
    (lambda s: s.update(temperature=1.91), "1.91, not 1.0"),
    (lambda s: s.update(temperature=None), "engine is off"),
    (lambda s: s.update(position_bias={"profile": "p.json", "sha256": "f" * 64}), "without --position-bias"),
])
def test_preconditions_refuse_before_any_readout(stub, tmp_path, edit, match):
    edit(stub.stats)
    with pytest.raises(Refusal, match=match):
        _collect(stub, tmp_path)
    assert stub.posts == [] and not (tmp_path / "raw.jsonl").exists()


def test_an_unreachable_server_is_a_refusal(tmp_path):
    items = read_set(set_text()).items
    with pytest.raises(Refusal, match="cannot read"):
        collect("http://127.0.0.1:1", items, raw_path=tmp_path / "r.jsonl", seed=0, resume=False, allow_busy=False,
                set_name="t", set_sha="0" * 64, out=io.StringIO(), sleep=lambda s: None)


def test_every_rotation_is_read_on_the_pinned_lane_and_written_to_raw(stub, tmp_path):
    items, c = _collect(stub, tmp_path)
    cells = sum(it.n for it in items)
    assert len(c.readings) == cells == len(stub.posts) == 20 and set(stub.lanes) == {LANE}
    assert c.server == {"kv": "unified", "temperature": 1.0, "s1_slots": 1, "slot": 0}
    rows = _lines(tmp_path / "raw.jsonl")
    head = rows[0]
    assert head == {"callo_bias_raw": 1, "model_key": KEY, "provenance": stub.stats["provenance"],
                    "server": {"kv": "unified", "temperature": 1.0, "s1_slots": 1, "slot": 0},
                    "set": {"name": "t.jsonl", "sha256": hashlib.sha256(set_text().encode()).hexdigest()}, "seed": 0,
                    "max_per_n": 48, "instance": "stub-1"}
    readouts = [r for r in rows if r.get("type") == "readout"]
    checks = [r for r in rows if r.get("type") == "check"]
    assert len(readouts) == 20 and [c_["index"] for c_ in checks] == [0, 1] and checks[-1]["readouts"] == 20
    r = readouts[0]
    assert set(r) == {"type", "q", "cluster", "n", "s", "probs", "miss", "slot", "timings", "check"}
    assert r["slot"] == 0 and r["check"] == 0 and r["timings"]["miss"] == r["miss"] == 0
    assert abs(sum(r["probs"]) - 1) < 1e-12 and c.checks == [{k: v for k, v in x.items() if k != "type"} for x in checks]
    order = [(r["q"], r["s"]) for r in readouts]
    assert order != sorted(order)                                         # seeded shuffle, not file order


def test_an_existing_raw_without_resume_is_refused_and_left_alone(stub, tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("precious\n")
    with pytest.raises(Refusal, match="--resume"):
        _collect(stub, tmp_path)
    assert raw.read_text() == "precious\n" and stub.posts == []


def test_resume_reads_only_the_missing_cells_and_appends_a_session(stub, tmp_path):
    stub.fail_after = (7, 400)
    with pytest.raises(RunFailed, match="HTTP 400"):
        _collect(stub, tmp_path)
    assert len([r for r in _lines(tmp_path / "raw.jsonl") if r.get("type") == "readout"]) == 7
    stub.fail_after, before = None, len(stub.posts)
    _, c = _collect(stub, tmp_path, resume=True)
    assert len(stub.posts) - before == 13 and len(c.readings) == 20
    assert {x["session"] for x in c.checks} == {0, 1}                    # the first session's checks are kept


@pytest.mark.parametrize(("change", "field"), [({"seed": 1}, "seed"), ({"text": set_text(per=5)}, "set"),
                                               ({"max_per_n": 10}, "max_per_n")])
def test_resume_refuses_any_header_difference(stub, tmp_path, change, field):
    _collect(stub, tmp_path)
    with pytest.raises(Refusal, match=field):
        _collect(stub, tmp_path, resume=True, seed=change.get("seed", 0), text=change.get("text"),
                 max_per_n=change.get("max_per_n", 48))


def test_resume_ignores_the_predicted_slot_and_keeps_the_runs_slot(stub, tmp_path):
    """Another lane took slot 0 between sessions, so the predicted slot moved; the slot is enforced per
    readout instead, and callo still serves the calibrate lane on slot 0."""
    stub.stats["router"] = {"total_slots": 4, "reserved": [0, 1], "lanes": {}}
    stub.fail_after = (7, 400)
    with pytest.raises(RunFailed):
        _collect(stub, tmp_path)
    stub.fail_after = None
    stub.stats["router"] = {"total_slots": 4, "reserved": [0, 1], "lanes": {"agent": 0}}
    _, c = _collect(stub, tmp_path, resume=True)
    assert len(c.readings) == 20 and c.server["slot"] == 0


def test_resume_refuses_a_provenance_difference(stub, tmp_path):
    _collect(stub, tmp_path)
    stub.stats["provenance"] = {**stub.stats["provenance"], "build_info": "b9999-new"}
    with pytest.raises(Refusal, match="provenance") as e:
        _collect(stub, tmp_path, resume=True)
    assert "restart it" not in str(e.value)                              # no cache hint unless cache_prompt moved


@pytest.mark.parametrize(("old", "new", "fix"), [(False, True, "restart it with --no-cache"),
                                                 (True, False, "restart it without --no-cache")])
def test_resume_across_the_cache_default_names_the_serve_flag(tmp_path, capsys, old, new, fix):
    """A raw file read through a `--no-cache` serve (every bias file before --cache became the default) resumed
    against a default serve (cache on) is refused with the flag that resumes it, not only "remove the file"."""
    data = _write_set(tmp_path, set_text(ns=(3,), per=24))
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    try:
        stub.stats["provenance"] = {**stub.stats["provenance"], "cache_prompt": old}
        stub.fail_after = (30, 400)
        argv = ["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out", str(tmp_path / "a.json")]
        assert main(argv) == 1
        stub.fail_after = None
        stub.stats["provenance"] = {**stub.stats["provenance"], "cache_prompt": new}
        capsys.readouterr()
        assert main([*argv, "--resume"]) == 2
    finally:
        stub.stop()
    err = capsys.readouterr().err
    assert "provenance differ" in err and "Remove the file to start over" in err and fix in err
    assert f"cache {'on' if old else 'off'}" in err and "--cache is on by default since callo 0.3" in err


def test_resume_drops_a_partial_last_line_left_by_a_hard_kill(stub, tmp_path):
    stub.fail_after = (5, 400)
    with pytest.raises(RunFailed):
        _collect(stub, tmp_path)
    raw = tmp_path / "raw.jsonl"
    raw.write_text(raw.read_text() + '{"type": "readout", "q": "L1/q", "s": 0, "pro')
    stub.fail_after = None
    _, c = _collect(stub, tmp_path, resume=True)
    assert len(c.readings) == 20 and all(json.loads(line) for line in raw.read_text().splitlines())


def test_a_503_is_retried_with_backoff_and_a_4xx_aborts_at_once(stub, tmp_path):
    stub.script = [503, 503]
    sleeps = []
    _, c = _collect(stub, tmp_path, sleeps=sleeps)
    assert sleeps == [1.0, 4.0] and len(c.readings) == 20
    stub.script = [503] * 4
    sleeps = []
    with pytest.raises(RunFailed, match="after 3 retries"):
        _collect(stub, tmp_path, raw=tmp_path / "b.jsonl", sleeps=sleeps)
    assert sleeps == [1.0, 4.0, 16.0]
    stub.fail_after, before = (stub.readouts, 404), len(stub.posts)
    with pytest.raises(RunFailed, match="HTTP 404"):
        _collect(stub, tmp_path, raw=tmp_path / "c.jsonl")
    assert len(stub.posts) - before == 1                                   # no retry on a 4xx


def test_connection_errors_are_retried_then_fail():
    sleeps = []
    with pytest.raises(RunFailed, match="after 3 retries"):
        post_readout("http://127.0.0.1:1", {"state": "s"}, what="r1 (question q, rotation 0)", sleep=sleeps.append)
    assert sleeps == [1.0, 4.0, 16.0]


def test_a_changed_slot_aborts_with_exit_2(stub, tmp_path):
    stub.slot_after = (3, "1")
    with pytest.raises(Refusal, match="slot changed from 0 to 1") as e:
        _collect(stub, tmp_path)
    assert "restart `callo serve` (lanes reset) or remove" in str(e.value)       # --resume alone loops


def test_an_unknown_predicted_slot_is_fixed_by_the_first_readout(stub, tmp_path):
    """Every reserved slot is held by another lane, so /callo/stats cannot say which one callo
    will rebind; the header records null and the first readout's X-Callo-Slot pins the run."""
    stub.stats["router"] = {"total_slots": 4, "reserved": [0, 1], "lanes": {"a": 0, "b": 1}}
    stub.slot = "1"
    _, c = _collect(stub, tmp_path)
    rows = _lines(tmp_path / "raw.jsonl")
    assert rows[0]["server"]["slot"] is None and c.server["slot"] == 1
    assert {r["slot"] for r in rows if r.get("type") == "readout"} == {1}


def test_a_single_slot_server_reads_on_slot_0():
    assert parse_slot("-", 0) == 0 and parse_slot("2", 1) == 2 and parse_slot("-", 1) is None
    assert parse_slot(None, 1) is None and parse_slot("x", 1) is None


def test_np1_records_s1_slots_0_and_slot_0(stub, tmp_path):
    stub.stats["router"] = {"total_slots": 1, "reserved": [], "lanes": {}}
    stub.slot = "-"
    _, c = _collect(stub, tmp_path)
    assert c.server["s1_slots"] == 0 and c.server["slot"] == 0


def test_a_busy_server_refuses_at_check_0_unless_allow_busy(stub, tmp_path):
    stub.stats["slots"][2]["n_prompt_tokens"] = 300
    with pytest.raises(Refusal, match="slot 2 \\(300 tokens\\)") as e:
        _collect(stub, tmp_path)
    assert stub.posts == [] and not (tmp_path / "raw.jsonl").exists()          # nothing written
    assert "nothing was written to" in str(e.value)
    out = io.StringIO()
    _, c = _collect(stub, tmp_path, raw=tmp_path / "busy.jsonl", allow_busy=True, out=out)
    assert len(c.readings) == 20 and c.checks[0]["failed"] == ["foreign_tokens"]
    assert c.checks[0]["held"] == {"2": 300} and "--allow-busy" in out.getvalue()


def test_a_check_0_refusal_leaves_no_trace_and_a_fixed_server_calibrates_clean(stub, tmp_path):
    """A session that read nothing must not mark the profile busy forever."""
    stub.stats["slots"][2]["n_prompt_tokens"] = 300
    with pytest.raises(Refusal):
        _collect(stub, tmp_path)
    stub.stats["slots"][2]["n_prompt_tokens"] = 0
    _, c = _collect(stub, tmp_path)                                       # no --resume needed: there is no file
    assert len(c.readings) == 20 and all(x["failed"] == [] for x in c.checks)


def test_a_check_0_refusal_on_resume_leaves_the_raw_file_unchanged(stub, tmp_path):
    stub.fail_after = (7, 400)
    with pytest.raises(RunFailed):
        _collect(stub, tmp_path)
    raw = tmp_path / "raw.jsonl"
    before = raw.read_bytes()
    stub.fail_after = None
    stub.stats["slots"][2]["n_prompt_tokens"] = 300
    with pytest.raises(Refusal) as e:
        _collect(stub, tmp_path, resume=True)
    assert raw.read_bytes() == before and f"{raw} is unchanged (7 readouts)" in str(e.value)
    stub.stats["slots"][2]["n_prompt_tokens"] = 0
    _, c = _collect(stub, tmp_path, resume=True)
    assert len(c.readings) == 20 and all(x["failed"] == [] for x in c.checks)
    assert {x["session"] for x in c.checks} == {0, 1}


def test_foreign_s1_traffic_mid_run_is_caught_at_the_next_check(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "CHECK_EVERY", 5)

    def foreign(k, s):
        if k == 3:
            s.stats["s1"]["readouts"] += 2                                # two readouts calibrate did not send

    stub.on_readout = foreign
    with pytest.raises(Refusal, match="2 System One readouts arrived") as e:
        _collect(stub, tmp_path)
    assert stub.readouts == 5
    last = _lines(tmp_path / "raw.jsonl")[-1]                             # a refusal after readouts keeps the check
    assert last["type"] == "check" and last["failed"] == ["foreign_s1"]
    assert "keeps its 5 readouts and this failed check" in str(e.value) and "marked busy" in str(e.value)


def test_a_retried_readout_that_callo_already_answered_is_not_foreign(stub, tmp_path, monkeypatch):
    """The connection dropped after callo answered, so callo counted the readout twice."""
    monkeypatch.setattr(cb, "CHECK_EVERY", 5)
    real, posts = cb._http, []

    def flaky(method, url, body=None, headers=None, timeout=900.0):
        reply = real(method, url, body, headers, timeout)
        if url.endswith("/v1/systemone"):
            posts.append(url)
            if len(posts) == 3:
                raise ConnectionResetError("reset after the reply")
        return reply

    monkeypatch.setattr(cb, "_http", flaky)
    sleeps = []
    _, c = _collect(stub, tmp_path, sleeps=sleeps)
    assert sleeps == [1.0] and stub.readouts == 21 and len(c.readings) == 20
    assert all(x["failed"] == [] for x in c.checks) and c.checks[1]["slack"] == 1


@pytest.mark.parametrize(("edit", "field"), [
    (lambda s: s.update(model_key="sha256:" + "77" * 32), "model_key"),
    (lambda s: s.update(kv="per_slot"), "kv"),
    (lambda s: s.update(router={**s["router"], "reserved": [0, 1]}), "s1_slots"),
    (lambda s: s.update(provenance={**s["provenance"], "server_args": ["-ctk", "q8_0"]}), "provenance"),
    (lambda s: s.update(temperature=0.8), "temperature"),
    (lambda s: s.update(position_bias={"profile": "p.json", "sha256": "f" * 64}), "position_bias"),
])
def test_a_server_that_changes_mid_run_is_refused(stub, tmp_path, monkeypatch, edit, field):
    """The served model or the readout's configuration changed while calibrate ran."""
    monkeypatch.setattr(cb, "CHECK_EVERY", 5)
    stub.on_readout = lambda k, s: edit(s.stats) if k == 4 else None
    with pytest.raises(Refusal, match=f"the server changed during calibration \\({field}"):
        _collect(stub, tmp_path)


def test_a_callo_restart_mid_run_is_refused_and_resume_continues(stub, tmp_path, monkeypatch):
    """A restart on the same model resets s1.readouts and routes, which would hide
    foreign traffic from conditions 4 and 5; the new instance id gives it away. A resume across it is legitimate."""
    monkeypatch.setattr(cb, "CHECK_EVERY", 5)

    def restart(k, s):
        if k == 4:
            s.stats.update(instance="stub-2", routes={}, s1={"n": 0, "readouts": 0, "topk_miss": 0})

    stub.on_readout = restart
    with pytest.raises(Refusal, match="callo restarted during calibration"):
        _collect(stub, tmp_path)
    assert stub.readouts == 5
    stub.on_readout = None
    _, c = _collect(stub, tmp_path, resume=True)
    assert len(c.readings) == 20 and all(x["failed"] == [] for x in c.checks)


def test_a_200_without_a_usable_answer_names_the_record(stub, tmp_path, monkeypatch):
    real = stub.readout
    monkeypatch.setattr(stub, "readout", lambda body: ({"answers": {}}, real(body)[1]))
    with pytest.raises(RunFailed, match=r"r[23]-\d \(question q, rotation \d\): a 200 reply with no usable answer"):
        _collect(stub, tmp_path)


def test_a_measured_eta_follows_the_first_20_readouts(stub, tmp_path):
    out = io.StringIO()
    _collect(stub, tmp_path, out=out)
    text = out.getvalue()
    assert "n=2: 4 questions, 8 readouts" in text and "20 readouts in all" in text
    assert "ms per readout" in text


def test_every_readout_forwards_the_records_keys_and_names_the_lane(stub, tmp_path):
    """Every top-level record key but questions/label/src/_meta (bias.NOT_FORWARDED) reaches callo verbatim, and every
    readout carries X-Callo-Conversation: callo-calibrate."""
    extra = {"model": "kev-x", "future_key": {"nested": [1, 2]}, "src": "top-level src", "label": "top-level label"}
    text = "".join(json.dumps({**json.loads(line), **extra}) + "\n" for line in set_text().splitlines())
    _collect(stub, tmp_path, text)
    assert len(stub.posts) == 20 and stub.lanes == [LANE] * 20
    for body in stub.posts:
        assert set(body) == {"state", "model", "future_key", "questions"}
        assert body["model"] == "kev-x" and body["future_key"] == {"nested": [1, 2]}
        assert "label" not in body["questions"]["q"]


def test_read_raw_is_the_one_parser_of_the_raw_file(stub, tmp_path):
    """The parser `--resume` uses and the bench's offline refit imports (one definition of the format)."""
    _, c = _collect(stub, tmp_path)
    raw = tmp_path / "raw.jsonl"
    r = cb.read_raw(raw)
    assert r.header == c.header | {"callo_bias_raw": 1} and r.readings == c.readings and r.checks == c.checks
    assert r.slot == 0 and r.partial is False
    whole = raw.read_text()
    raw.write_text(whole + '{"type": "readout", "q": "L1/q", "s": 0, "pro')        # a live or hard-killed run
    r2 = cb.read_raw(raw)
    assert r2.readings == c.readings and r2.partial is True
    assert raw.read_text().endswith('"pro')                                        # reading never writes
    (tmp_path / "bad.jsonl").write_text('{"not": "a header"}\n')
    with pytest.raises(ValueError, match="no callo --bias raw header"):
        cb.read_raw(tmp_path / "bad.jsonl")
    (tmp_path / "bad2.jsonl").write_text('{"callo_bias_raw": 1}\nnot json\n')
    with pytest.raises(ValueError, match="line 2 is not JSON"):
        cb.read_raw(tmp_path / "bad2.jsonl")


def test_a_check_0_refusal_on_resume_keeps_even_a_partial_last_line(stub, tmp_path):
    """Requirement: a check-0 refusal writes nothing, not even the hard kill's partial-line repair."""
    stub.fail_after = (5, 400)
    with pytest.raises(RunFailed):
        _collect(stub, tmp_path)
    raw = tmp_path / "raw.jsonl"
    raw.write_text(raw.read_text() + '{"type": "readout", "q": "L1/q", "s": 0, "pro')
    before = raw.read_bytes()
    stub.fail_after = None
    stub.stats["slots"][2]["n_prompt_tokens"] = 300
    with pytest.raises(Refusal, match="is unchanged"):
        _collect(stub, tmp_path, resume=True)
    assert raw.read_bytes() == before


def test_resume_refuses_a_file_that_is_not_a_raw_file(stub, tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"callo_bias_raw": 1}\nnot json\n')
    with pytest.raises(Refusal, match="refusing to resume: .*line 2 is not JSON; remove the file"):
        _collect(stub, tmp_path, resume=True)
    assert stub.posts == []


# ---- the CLI, the profile, the summary ----------------------------------------------------------------------

PLANTED = {4: [0.4, 0.3, 0.2, 0.1], 6: [0.3, 0.25, 0.15, 0.1, 0.1, 0.1]}


def _write_set(tmp_path, text):
    f = tmp_path / "set.jsonl"
    f.write_text(text)
    return f


def _without_volatile(profile):
    p = json.loads(json.dumps(profile))
    p.pop("created")
    p["server_state"].pop("checks")
    return p


def test_without_data_the_builtin_set_is_a_diagnostic_that_writes_no_profile(tmp_path, monkeypatch, capsys,
                                                                              builtin_set):
    """The built-in set measures (summary on stdout, --raw kept) and writes no profile; refitting its raw
    file gives a profile marked builtin, which `load_profile` refuses."""
    monkeypatch.chdir(tmp_path)
    stub = CalloStub(PlantedBias(priors=PLANTED))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url]) == 0
    finally:
        stub.stop()
    cap = capsys.readouterr()
    assert "callo_position_bias" not in cap.out and cap.err == ""             # no profile JSON anywhere
    assert "position-bias DIAGNOSTIC (built-in set, no profile written)" in cap.out
    assert "n=4: SIGNIFICANT" in cap.out and "n=6: SIGNIFICANT" in cap.out and "KEPT" not in cap.out
    assert "n=2: dropped: not significant" in cap.out and "a shared position bias on the built-in set at n = 4, 6" \
        in cap.out
    assert "--data YOUR.jsonl" in cap.out and "per position, not simultaneous" in cap.out
    assert "debiased" not in cap.out and "this summary is not a profile" in cap.out      # no hypothetical correction
    assert not (tmp_path / "callo-bias.raw.jsonl").exists()                # a --data run's default stays free
    raw = tmp_path / "callo-bias-diagnostic.raw.jsonl"
    assert _lines(raw)[0]["max_per_n"] == 48 and sum(r.get("type") == "readout" for r in _lines(raw)) == 1680
    assert len(stub.posts) == 1680
    profile = refit(raw)                                                    # the same fit, offline
    assert profile["set"]["builtin"] is True and set(profile["entries"]) == {"4", "6"}
    for n, planted in PLANTED.items():
        prior = profile["entries"][str(n)]["prior"]
        assert prior == pytest.approx(_raw_meanp(raw, n), abs=1e-9)
        assert np.argmax(prior) == np.argmax(planted)                      # the planted direction, shrunk
        assert profile["entries"][str(n)]["questions"] == 48 and profile["entries"][str(n)]["p_holm"] <= 0.05
    assert set(profile["dropped"]) == {"2", "3", "5", "7", "8"}
    assert profile["set"]["name"] == "builtin-v1" and profile["set"]["questions"] == 336
    assert profile["set"]["kinds"] == ["plain"] and profile["set"]["sources_per_n"]["5"] == {"banking77": 48}
    assert profile["set"]["sources_per_n"]["3"] == {"banking77": 24, "mmlu": 24}
    assert profile["server_state"]["state"] == "clean" and profile["model_key"] == KEY
    assert profile["callo_position_bias"] == 2 and profile["estimator"] == "meanP" and profile["server"]["slot"] == 0
    with pytest.raises(InvalidInput, match="built-in set, which is a diagnostic only"):
        load_profile(profile)


def test_a_copy_of_the_builtin_set_is_the_diagnostic_too_and_its_raw_never_blocks_a_data_run(tmp_path, monkeypatch,
                                                                                             capsys, builtin_set):
    """Built-in means the built-in set's sha256, under any file name, so `--data <copy> --out` is refused
    and `--data <copy>` is the diagnostic. Its default --raw differs from a --data run's, so the user's own run that
    follows is not refused as "already exists"."""
    monkeypatch.chdir(tmp_path)
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(builtin_set_bytes())
    mine = _write_set(tmp_path, set_text(ns=(3,), per=24))
    stub = CalloStub(PlantedBias())                                      # no position effect anywhere
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(copy), "--out", "b.json"]) == 2
        assert stub.posts == [] and "--out needs --data" in capsys.readouterr().err
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(copy), "--max-per-n", "2"]) == 0
        out = capsys.readouterr().out
        assert "position-bias DIAGNOSTIC" in out and "no shared position bias detected" in out
        assert refit(tmp_path / "callo-bias-diagnostic.raw.jsonl", copy)["set"]["builtin"] is True
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(mine)]) == 0       # the user's own run
    finally:
        stub.stop()
    assert json.loads(capsys.readouterr().out)["set"]["builtin"] is False
    assert (tmp_path / "callo-bias.raw.jsonl").is_file() and not (tmp_path / "b.json").exists()


def test_out_without_data_is_refused_before_anything_is_read(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    stub = CalloStub(PlantedBias(priors=PLANTED))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--out", str(tmp_path / "bias.json")]) == 2
    finally:
        stub.stop()
    err = capsys.readouterr().err
    assert "--out needs --data" in err and "writes no profile" in err
    assert "omit --out to run the built-in diagnostic" in err
    assert stub.posts == [] and not (tmp_path / "bias.json").exists() and not (tmp_path / "bias.raw.jsonl").exists()


def test_with_data_every_eligible_question_is_read_by_default(tmp_path, capsys):
    """The user's file is the budget, so --max-per-n defaults to no cap with --data (48 is the built-in
    diagnostic's); an explicit --max-per-n still caps."""
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    data = _write_set(tmp_path, set_text(ns=(3,), per=60))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out",
                     str(tmp_path / "all.json")]) == 0
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--max-per-n", "30", "--out",
                     str(tmp_path / "cap.json")]) == 0
    finally:
        stub.stop()
    full, capped = (json.loads((tmp_path / f"{x}.json").read_text()) for x in ("all", "cap"))
    assert full["entries"]["3"]["questions"] == 60 and full["set"]["capped"] == 0
    assert _lines(tmp_path / "all.raw.jsonl")[0]["max_per_n"] == cb.NO_CAP == 1_000_000
    assert capped["entries"]["3"]["questions"] == 30 and capped["set"]["capped"] == 30


def test_with_out_the_profile_goes_to_the_file_and_the_summary_to_stdout(tmp_path, capsys):
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    data = _write_set(tmp_path, set_text(ns=(3,), per=24))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data),
                     "--out", str(tmp_path / "bias.json")]) == 0
    finally:
        stub.stop()
    cap = capsys.readouterr()
    assert "n=3: KEPT" in cap.out and cap.err == ""                     # with --out everything human is on stdout
    profile = json.loads((tmp_path / "bias.json").read_text())
    assert profile["entries"]["3"]["prior"] == pytest.approx(_raw_meanp(tmp_path / "bias.raw.jsonl", 3), abs=1e-9)
    assert profile["set"]["builtin"] is False and load_profile(tmp_path / "bias.json").priors[3] == pytest.approx(
        tuple(profile["entries"]["3"]["prior"]))                        # a --data profile loads (format 2)
    assert (tmp_path / "bias.raw.jsonl").is_file()                      # the default --raw beside --out


def test_floored_questions_are_fitted_and_counted(tmp_path, capsys):
    """In probability space a floored option moves a row by at most its own small upper bound, so floored
    questions are fitted, and the summary and the entry count them."""
    stub = CalloStub(PlantedBias(priors={4: PLANTED[4]}, top_k=2))      # every n = 4 readout floors two options
    data = _write_set(tmp_path, set_text(ns=(4,), per=30))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out",
                     str(tmp_path / "b.json")]) == 0
    finally:
        stub.stop()
    profile = json.loads((tmp_path / "b.json").read_text())
    assert profile["entries"]["4"]["questions"] == 30 and profile["entries"]["4"]["floored"] == 30
    assert profile["entries"]["4"]["prior"] == pytest.approx(_raw_meanp(tmp_path / "b.raw.jsonl", 4), abs=1e-9)
    out = capsys.readouterr().out
    assert "30 questions fitted, 30 of them with a floored option (100%; fitted all the same)" in out
    assert "4" in profile["eval"]["raw"]["consistency"]                  # rotation consistency too


def test_every_n_dropped_still_writes_the_profile_and_says_so_loudly(tmp_path, capsys):
    stub = CalloStub(PlantedBias())                                      # no position effect anywhere
    data = _write_set(tmp_path, set_text(ns=(2, 3), per=24))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out",
                     str(tmp_path / "b.json")]) == 0
    finally:
        stub.stop()
    assert json.loads((tmp_path / "b.json").read_text())["entries"] == {}
    assert "NOTHING WAS KEPT" in capsys.readouterr().out


def test_resume_after_a_mid_run_abort_gives_the_uninterrupted_profile(tmp_path):
    data = _write_set(tmp_path, set_text(ns=(3,), per=24))
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    try:
        stub.fail_after = (30, 400)
        argv = ["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out", str(tmp_path / "a.json")]
        assert main(argv) == 1
        stub.fail_after = None
        assert main([*argv, "--resume"]) == 0
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out",
                     str(tmp_path / "b.json")]) == 0
    finally:
        stub.stop()
    a, b = (json.loads((tmp_path / f).read_text()) for f in ("a.json", "b.json"))
    assert _without_volatile(a) == _without_volatile(b)
    assert {c["session"] for c in a["server_state"]["checks"]} == {0, 1}


def test_ctrl_c_mid_run_exits_130_with_a_resume_hint_and_a_clean_raw_file(tmp_path, monkeypatch, capsys):
    data = _write_set(tmp_path, set_text(ns=(3,), per=24))
    real, calls = cb.post_readout, []

    def interrupted(*a, **kw):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(*a, **kw)

    monkeypatch.setattr(cb, "post_readout", interrupted)
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out",
                     str(tmp_path / "x.json")]) == 130
    finally:
        stub.stop()
    err = capsys.readouterr().err
    assert "interrupted" in err and "2 readouts" in err and "--resume" in err and "Traceback" not in err
    assert all(json.loads(line) for line in (tmp_path / "x.raw.jsonl").read_text().splitlines())


def test_cli_refusals_and_failures_map_to_exit_codes(tmp_path, capsys):
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    data = _write_set(tmp_path, set_text(ns=(3,), per=3))
    base = ["calibrate", "--bias", "--url", stub.url, "--data", str(data)]
    try:
        stub.stats["temperature"] = 1.9
        assert main([*base, "--out", str(tmp_path / "t.json")]) == 2
        stub.stats["temperature"] = 1.0
        stub.stats["slots"][3]["n_prompt_tokens"] = 300
        assert main([*base, "--out", str(tmp_path / "busy.json")]) == 2
        assert "refusing to calibrate" in capsys.readouterr().err
        assert main([*base, "--out", str(tmp_path / "busy2.json"), "--allow-busy"]) == 0
        assert json.loads((tmp_path / "busy2.json").read_text())["server_state"]["state"] == "busy"
        stub.stats["slots"][3]["n_prompt_tokens"] = 0
        stub.fail_after = (stub.readouts, 400)
        assert main([*base, "--out", str(tmp_path / "f.json")]) == 1
        assert "HTTP 400" in capsys.readouterr().err
    finally:
        stub.stop()


def test_no_eligible_questions_is_exit_1_naming_the_skips(tmp_path, capsys):
    data = _write_set(tmp_path, json.dumps({"state": "s", "questions": {"q": {"type": "score", "criteria": ["lo", "hi"]}}})
                      + "\n")
    assert main(["calibrate", "--bias", "--url", "http://127.0.0.1:1", "--data", str(data)]) == 1
    assert "no eligible questions" in capsys.readouterr().err


def test_a_malformed_line_is_exit_1_with_its_number_before_any_request(tmp_path, capsys):
    """Bias mode reads the whole set first; temperature mode's twin is in tests/test_cli.py."""
    data = _write_set(tmp_path, set_text(ns=(3,), per=1) + "{broken\n")
    assert main(["calibrate", "--bias", "--url", "http://127.0.0.1:1", "--data", str(data)]) == 1
    assert "--data set.jsonl line 2: not valid JSON" in capsys.readouterr().err


@pytest.mark.parametrize("flag", [["--min-questions", "1"], ["--min-questions", "0"], ["--max-per-n", "0"]])
def test_degenerate_question_counts_are_a_usage_error(flag, capsys):
    """--min-questions 0 would reach the permutation test with no questions."""
    assert main(["calibrate", "--bias", "--url", "http://127.0.0.1:1", *flag]) == 2
    assert "--min-questions must be at least 2 and --max-per-n at least 1" in capsys.readouterr().err


def test_an_unwritable_out_is_exit_1_and_resume_refits_without_reading_again(tmp_path, capsys):
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    data = _write_set(tmp_path, set_text(ns=(3,), per=24))
    (tmp_path / "afile").write_text("")
    base = ["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--raw", str(tmp_path / "r.jsonl")]
    try:
        assert main([*base, "--out", str(tmp_path / "afile" / "b.json")]) == 1
        err = capsys.readouterr().err
        assert "cannot write --out" in err and "keeps every readout" in err and "--resume" in err
        assert f"pass --raw {tmp_path / 'r.jsonl'} explicitly if --out changes" in err     # the same --raw
        before = len(stub.posts)
        assert main([*base, "--out", str(tmp_path / "b.json"), "--resume"]) == 0
        assert len(stub.posts) == before                                   # nothing read again
    finally:
        stub.stop()
    assert json.loads((tmp_path / "b.json").read_text())["entries"]["3"]["prior"] == pytest.approx(
        _raw_meanp(tmp_path / "r.jsonl", 3), abs=1e-9)


def test_the_end_to_end_run_against_callo_serve_and_the_planted_fake(tmp_path, monkeypatch):
    """callo serve (attach mode) on the planted-prior fake upstream with --slot-save-path: the probe residue is
    erased, the guard passes with no --allow-busy, the fit measures the planted readouts' meanP through callo's real
    engine, and a restarted serve applies the profile: softmax(log p - offset), the same arithmetic as `debias`."""
    monkeypatch.setenv("HOME", str(tmp_path))
    up = FakeUpstream(bias=PlantedBias(priors={3: PRIOR3}), slot_save=True).start()
    try:
        g = tiny_gguf(tmp_path / "m.gguf")
        up.props["model_path"] = str(g)
        r = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
        try:
            data = _write_set(tmp_path, set_text(ns=(2, 3), per=24))
            assert main(["calibrate", "--bias", "--url", r.url, "--data", str(data), "--out",
                         str(tmp_path / "bias.json")]) == 0
        finally:
            assert r.finish() == 0
        profile = json.loads((tmp_path / "bias.json").read_text())
        assert profile["server_state"]["state"] == "clean" and profile["model_key"] == load_model_key(g, None)
        prior = profile["entries"]["3"]["prior"]
        assert prior == pytest.approx(_raw_meanp(tmp_path / "bias.raw.jsonl", 3), abs=1e-9)
        assert np.argmax(prior) == 0 and profile["dropped"]["2"].startswith("not significant")
        r2 = Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"),
                      "--position-bias", str(tmp_path / "bias.json")])
        try:
            body = {"state": "x", "questions": {"q": {"type": "choice", "criteria": {"red": None, "green": None,
                                                                                    "blue": None}}}}
            status, _, out = post_json(f"{r2.url}/v1/systemone", body)
            served = list(out["answers"]["q"]["probabilities"].values())
            raw = PlantedBias(priors={3: PRIOR3}).probs(["red", "green", "blue"])
            assert status == 200 and served == pytest.approx(debias(raw, offsets_from_prior(prior)), abs=1e-6)
            assert served != pytest.approx(raw, abs=1e-3)
            assert get_json(f"{r2.url}/callo/stats")[2]["position_bias"]["applied"] == 1
        finally:
            assert r2.finish() == 0
    finally:
        up.stop()


# ---- in-place partial-line repair, validated check rows, retried stats reads -------------------------------

def _close_fails(monkeypatch):
    """--raw whose close() raises after really closing (a full disk at the final flush)."""
    real = cb._open_raw

    class Closing:
        def __init__(self, f):
            self.f = f

        def __getattr__(self, name):
            return getattr(self.f, name)

        def close(self):
            self.f.close()
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(cb, "_open_raw", lambda *a, **kw: Closing(real(*a, **kw)))


@pytest.mark.parametrize("failure", ["run_failed", "refusal", "ctrl_c"])
def test_a_close_error_never_replaces_the_exception_in_flight(stub, tmp_path, monkeypatch, failure):
    """The readout failure (or the refusal, or Ctrl-C) is what the user must see, not --raw's close."""
    _close_fails(monkeypatch)
    if failure == "run_failed":
        stub.fail_after, want = (3, 400), RunFailed
    elif failure == "refusal":
        stub.slot_after, want = (3, "2"), Refusal
    else:
        want = KeyboardInterrupt

        def interrupted(*a, **kw):
            raise KeyboardInterrupt

        monkeypatch.setattr(cb, "post_readout", interrupted)
    with pytest.raises(want) as e:
        _collect(stub, tmp_path)
    assert "No space left" not in str(e.value)


def test_a_close_error_after_a_complete_run_is_the_raw_error(stub, tmp_path, monkeypatch):
    _close_fails(monkeypatch)
    with pytest.raises(RunFailed, match=r"--raw .*raw\.jsonl: .*No space left on device"):
        _collect(stub, tmp_path)


def test_the_partial_line_repair_truncates_in_place_and_never_rewrites_the_file(stub, tmp_path, monkeypatch):
    """A whole-file rewrite that crashed (or met a full disk) half way would destroy hours of readouts; the repair
    only cuts the file back to its last newline, on the same inode."""
    stub.fail_after = (5, 400)
    with pytest.raises(RunFailed):
        _collect(stub, tmp_path)
    raw = tmp_path / "raw.jsonl"
    prefix = raw.read_bytes()
    with raw.open("ab") as f:
        f.write(b'{"type": "readout", "q": "L1/q", "s": 0, "pro')
    ino = raw.stat().st_ino

    def no_rewrite(*a, **kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cb.Path, "write_text", no_rewrite)
    monkeypatch.setattr(cb.Path, "write_bytes", no_rewrite)
    state = cb._read_raw(raw, json.loads(prefix.splitlines()[0]))
    cb._open_raw(raw, {}, state).close()
    assert raw.read_bytes() == prefix and raw.stat().st_ino == ino


@pytest.mark.parametrize("check", [
    {"type": "check", "index": 0, "readouts": 0, "failed": []},                                 # no session
    {"type": "check", "index": 0, "session": "0", "readouts": 0, "failed": []},                 # session not an int
    {"type": "check", "session": 0, "readouts": 0, "failed": []},                               # no index
    {"type": "check", "index": 0, "session": 0, "readouts": 0},                                 # no failed
    {"type": "check", "index": 0, "session": 0, "readouts": 0, "failed": "none"},               # failed not a list
])
def test_read_raw_validates_check_rows_and_resume_refuses_a_malformed_one(stub, tmp_path, check):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"callo_bias_raw": 1}) + "\n" + json.dumps(check) + "\n")
    with pytest.raises(ValueError, match="line 2 is not a raw record"):
        cb.read_raw(raw)
    with pytest.raises(Refusal, match="refusing to resume: .*line 2 is not a raw record.*; remove the file"):
        _collect(stub, tmp_path, resume=True)
    assert stub.posts == []


@pytest.mark.parametrize("failure", ["503", "drop"])
def test_one_transient_stats_failure_mid_run_does_not_end_the_run(stub, tmp_path, monkeypatch, failure):
    """Mid-run /callo/stats reads retry a 503 or a connection error with the readouts' backoff."""
    monkeypatch.setattr(cb, "CHECK_EVERY", 5)
    real, gets = cb._http, []

    def flaky(method, url, body=None, headers=None, timeout=900.0):
        if url.endswith("/callo/stats"):
            gets.append(url)
            if len(gets) == 2:                                            # the first mid-run check
                if failure == "drop":
                    raise ConnectionResetError("reset")
                return 503, {}, "loading"
        return real(method, url, body, headers, timeout)

    monkeypatch.setattr(cb, "_http", flaky)
    sleeps = []
    _, c = _collect(stub, tmp_path, sleeps=sleeps)
    assert sleeps == [1.0] and len(c.readings) == 20 and len(gets) == 6


def test_a_stats_endpoint_that_stays_down_mid_run_fails_after_the_retries(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "CHECK_EVERY", 5)
    real, gets = cb._http, []

    def down(method, url, body=None, headers=None, timeout=900.0):
        if url.endswith("/callo/stats"):
            gets.append(url)
            if len(gets) >= 2:
                return 503, {}, "loading"
        return real(method, url, body, headers, timeout)

    monkeypatch.setattr(cb, "_http", down)
    sleeps = []
    with pytest.raises(RunFailed, match="callo/stats mid-run.*after 3 retries"):
        _collect(stub, tmp_path, sleeps=sleeps)
    assert sleeps == [1.0, 4.0, 16.0] and stub.readouts == 5



# ---- raw-file I/O errors exit 1; the resume hint counts through read_raw ---------------------------------

def test_a_raw_path_under_a_regular_file_is_exit_1_naming_raw(tmp_path, capsys):
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    data = _write_set(tmp_path, set_text(ns=(3,), per=3))
    (tmp_path / "afile").write_text("")
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data),
                     "--raw", str(tmp_path / "afile" / "r.jsonl"), "--out", str(tmp_path / "b.json")]) == 1
    finally:
        stub.stop()
    err = capsys.readouterr().err
    assert f"callo: --raw {tmp_path / 'afile' / 'r.jsonl'}: " in err and "Traceback" not in err
    assert "--resume" not in err                                          # nothing was read, so nothing to resume


def test_a_full_disk_mid_run_is_exit_1_with_the_resume_hint(tmp_path, monkeypatch, capsys):
    data = _write_set(tmp_path, set_text(ns=(3,), per=3))
    real, readouts = cb._write, []

    def full(f, obj):
        if obj.get("type") == "readout":
            readouts.append(1)
            if len(readouts) == 5:
                raise OSError(28, "No space left on device")
        real(f, obj)

    monkeypatch.setattr(cb, "_write", full)
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out",
                     str(tmp_path / "x.json")]) == 1
    finally:
        stub.stop()
    err = capsys.readouterr().err
    assert f"callo: --raw {tmp_path / 'x.raw.jsonl'}: " in err and "No space left on device" in err
    assert "keeps 4 readouts" in err and "--resume" in err and "Traceback" not in err


def test_the_resume_hint_counts_readouts_through_read_raw(stub, tmp_path):
    """One raw reader: a partial last line is not a readout, and a file that is not a raw file
    keeps none."""
    stub.fail_after = (4, 400)
    with pytest.raises(RunFailed):
        _collect(stub, tmp_path)
    raw = tmp_path / "raw.jsonl"
    with raw.open("a") as f:
        f.write('{"type":"readout","q":"L1/q","s":0,"pro')
    assert cb._raw_readouts(raw) == 4
    (tmp_path / "other.jsonl").write_text('{"type":"readout"}\n{"type":"readout"}\n')
    assert cb._raw_readouts(tmp_path / "other.jsonl") == 0
    assert cb._raw_readouts(tmp_path / "missing.jsonl") == 0


# ---- raw-file decoding, console errors, the resume hint, the summary, interrupts ---------------------------

def test_a_raw_file_that_is_not_utf8_is_a_resume_refusal_and_keeps_no_readouts(stub, tmp_path):
    """read_raw's UnicodeDecodeError is a refusal naming the cause, and the resume hint counts 0."""
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(b'{"callo_bias_raw": 1}\n\xff\xfe not text\n')
    with pytest.raises(Refusal, match="refusing to resume: .*not UTF-8.*remove the file"):
        _collect(stub, tmp_path, resume=True)
    assert stub.posts == [] and cb._raw_readouts(raw) == 0


class _BrokenPipe(io.StringIO):
    """A console whose reader went away: every write (or only one containing `on`) raises BrokenPipeError."""

    def __init__(self, on=""):
        super().__init__()
        self.on = on

    def write(self, s):
        if self.on in s:
            raise BrokenPipeError(32, "Broken pipe")
        return super().write(s)


@pytest.mark.parametrize("on", ["", "profile written to"])
def test_a_broken_console_pipe_exits_1_quietly_and_is_not_blamed_on_raw(tmp_path, monkeypatch, capsys, on):
    """With --out the plan and the summary go to stdout; a closed stdout is not a --raw error and no
    traceback, whether it breaks at the first line (inside collect) or at the last (after the profile is written)."""
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    data = _write_set(tmp_path, set_text(ns=(3,), per=3))
    monkeypatch.setattr("sys.stdout", _BrokenPipe(on))
    try:
        rc = main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out", str(tmp_path / "b.json")])
    finally:
        stub.stop()
    err = capsys.readouterr().err
    assert rc == 1 and "--raw" not in err and "Traceback" not in err


@pytest.mark.parametrize("has_fd", [True, False])
def test_the_broken_pipe_devnull_redirect_closes_what_it_opens(tmp_path, monkeypatch, has_fd):
    """dup2 onto stdout, then close the devnull fd; and open nothing when stdout has no descriptor."""
    target = (tmp_path / "console").open("w")
    opened, closed = [], []
    real_open, real_close = cb.os.open, cb.os.close

    def track_open(*a, **kw):
        fd = real_open(*a, **kw)
        opened.append(fd)
        return fd

    def track_close(fd):
        closed.append(fd)
        return real_close(fd)

    class Console(_BrokenPipe):
        def fileno(self):
            if not has_fd:
                raise io.UnsupportedOperation("fileno")
            return target.fileno()

    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    data = _write_set(tmp_path, set_text(ns=(3,), per=3))
    monkeypatch.setattr("sys.stdout", Console())
    monkeypatch.setattr(cb.os, "open", track_open)
    monkeypatch.setattr(cb.os, "close", track_close)
    try:
        rc = main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out", str(tmp_path / "b.json")])
    finally:
        monkeypatch.undo()
        stub.stop()
        target.close()
    assert rc == 1 and len(opened) == has_fd and closed == opened


def test_the_resume_hint_needs_a_raw_file_this_run_opened(tmp_path, monkeypatch, capsys):
    """A failure before collect (no eligible questions) never points at an unrelated raw file."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "callo-bias.raw.jsonl").write_text(
        '{"callo_bias_raw": 1}\n{"type":"readout","q":"x","n":1,"s":0,"probs":[1.0],"miss":0,"slot":0}\n')
    data = _write_set(tmp_path, json.dumps({"state": "s", "questions": {"q": {"type": "score",
                                                                              "criteria": ["lo", "hi"]}}}) + "\n")
    assert main(["calibrate", "--bias", "--url", "http://127.0.0.1:1", "--data", str(data)]) == 1
    err = capsys.readouterr().err
    assert "no eligible questions" in err and "--resume" not in err and "keeps" not in err


@pytest.mark.parametrize("resume", [False, True])
def test_ctrl_c_before_the_raw_file_is_opened_hints_only_on_a_resume_run(tmp_path, monkeypatch, capsys, resume):
    """A fresh run interrupted before collect opened --raw never points at a raw file left by another
    run; a --resume run's raw file is the user's own, so it keeps the hint."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "callo-bias.raw.jsonl").write_text(
        '{"callo_bias_raw": 1}\n{"type":"readout","q":"x","n":1,"s":0,"probs":[1.0],"miss":0,"slot":0}\n')
    data = _write_set(tmp_path, set_text(ns=(3,), per=3))

    def interrupted(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(cb, "read_set", interrupted)
    argv = ["calibrate", "--bias", "--url", "http://127.0.0.1:1", "--data", str(data)]
    assert main(argv + ["--resume"] * resume) == 130
    err = capsys.readouterr().err
    assert "interrupted" in err and "Traceback" not in err
    assert ("keeps 1 readouts" in err and "--resume" in err) if resume else ("--resume" not in err and "keeps" not in err)


def test_ctrl_c_with_nothing_kept_gives_no_resume_hint(tmp_path, monkeypatch, capsys):
    """A --resume run with no raw file yet, interrupted before reading, keeps 0 readouts: no hint, as on
    the RunFailed path."""
    monkeypatch.chdir(tmp_path)
    data = _write_set(tmp_path, set_text(ns=(3,), per=3))

    def interrupted(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(cb, "read_set", interrupted)
    assert main(["calibrate", "--bias", "--url", "http://127.0.0.1:1", "--data", str(data), "--resume"]) == 130
    err = capsys.readouterr().err
    assert err.strip() == "callo: interrupted"


def test_the_summary_prints_mean_p_raw_and_debiased(tmp_path, capsys):
    """meanP_n raw next to debiased, from the profile's eval."""
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    data = _write_set(tmp_path, set_text(ns=(3,), per=24))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data),
                     "--out", str(tmp_path / "b.json")]) == 0
    finally:
        stub.stop()
    ev = json.loads((tmp_path / "b.json").read_text())["eval"]
    (line,) = [ln for ln in capsys.readouterr().out.splitlines() if ln.lstrip().startswith("meanP_n")]
    raw = "  ".join(f"{p:.3f}" for p in ev["raw"]["meanP"]["3"])
    deb = "  ".join(f"{p:.3f}" for p in ev["debiased"]["meanP"]["3"])
    assert line == f"  meanP_n  {raw} raw -> {deb} debiased"


@pytest.mark.parametrize("where", ["out", "summary"])
def test_ctrl_c_while_writing_out_or_the_summary_exits_130_with_the_hint(tmp_path, monkeypatch, capsys, where):
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    data = _write_set(tmp_path, set_text(ns=(3,), per=3))
    if where == "out":
        real = cb.Path.write_text

        def interrupted(self, *a, **kw):
            if self.name == "b.json":
                raise KeyboardInterrupt
            return real(self, *a, **kw)

        monkeypatch.setattr(cb.Path, "write_text", interrupted)
    else:
        class Interrupted(io.StringIO):
            def write(self, s):
                if "profile written to" in s:
                    raise KeyboardInterrupt
                return super().write(s)

        monkeypatch.setattr("sys.stdout", Interrupted())
    try:
        rc = main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--out", str(tmp_path / "b.json")])
    finally:
        stub.stop()
    err = capsys.readouterr().err
    assert rc == 130 and "interrupted" in err and "keeps 9 readouts" in err and "--resume" in err


def test_resume_without_a_raw_file_says_it_starts_fresh(stub, tmp_path):
    """The bench's "same command resumes" keeps working, with one notice line."""
    out = io.StringIO()
    _collect(stub, tmp_path, resume=True, out=out)
    assert f"no raw file at {tmp_path / 'raw.jsonl'}; starting a fresh run" in out.getvalue()
    assert (tmp_path / "raw.jsonl").is_file()


# ---- usage errors, malformed readout rows, unusable replies -------------------------------------------------

def test_a_negative_seed_is_a_usage_error_before_any_readout(tmp_path, capsys):
    """numpy's default_rng refuses a negative seed, so it used to crash at the fit after
    every rotation was read, and the seed (a --resume header key) made the run unrecoverable."""
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    data = _write_set(tmp_path, set_text(ns=(3,), per=3))
    try:
        assert main(["calibrate", "--bias", "--url", stub.url, "--data", str(data), "--seed", "-1",
                     "--raw", str(tmp_path / "r.jsonl")]) == 2
    finally:
        stub.stop()
    assert "callo: --seed must be at least 0, got -1" in capsys.readouterr().err
    assert stub.posts == [] and not (tmp_path / "r.jsonl").exists()


@pytest.mark.parametrize("flag", ["--raw", "--out"])
@pytest.mark.parametrize("bias", [True, False])
def test_an_empty_raw_or_out_is_a_usage_error_not_the_default(flag, bias, capsys):
    """`--raw ""` and `--out ""` are mistakes, refused at parse time like serve's flags."""
    argv = ["calibrate", *(["--bias"] if bias else ["--data", "x.jsonl"]), "--url", "http://127.0.0.1:1", flag, ""]
    with pytest.raises(SystemExit) as e:
        main(argv)
    err = capsys.readouterr().err
    assert e.value.code == 2 and "must not be empty" in err
    if flag == "--out":         # the profile goes to stdout only with --data (else: the diagnostic)
        assert "with --bias --data the profile then goes to stdout" in err


def _readout(**over):
    row = {"type": "readout", "q": "x", "cluster": "x", "n": 2, "s": 0, "probs": [0.6, 0.4], "miss": 0, "slot": 0}
    row.update(over)
    return {k: v for k, v in row.items() if v is not _DROP}


_DROP = object()


@pytest.mark.parametrize("over", [
    {"probs": "ab"},                    # not a list
    {"probs": [0.6, 0.3, 0.1]},         # length is not n
    {"probs": [0.6, "0.4"]},            # an entry that is not a number
    {"probs": [True, False]},           # a bool is no probability
    {"probs": [1.2, -0.2]},             # negative
    {"probs": [float("nan"), 0.4]},     # not finite
    {"probs": [10**400, 0]},            # an int too large for a float: OverflowError, not a traceback (v0.3 2a)
    {"probs": [0, 0.0]},                # all zero: no readout to fit (v0.3 2a)
    {"n": _DROP},                       # no n
    {"n": "2"},
    {"miss": "0"},
    {"miss": False},
    {"slot": _DROP},
    {"slot": 0.0},
    {"s": "0"},
    {"q": 1},
])
def test_read_raw_validates_readout_rows_and_resume_refuses_a_malformed_one(stub, tmp_path, over):
    """A hand-edited readout row is InvalidInput (a --resume refusal), not a later numpy
    traceback in fit_n."""
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"callo_bias_raw": 1}) + "\n" + json.dumps(_readout(**over)) + "\n")
    with pytest.raises(ValueError, match="line 2 is not a raw record"):
        cb.read_raw(raw)
    with pytest.raises(Refusal, match="refusing to resume: .*line 2 is not a raw record.*; remove the file"):
        _collect(stub, tmp_path, resume=True)
    assert stub.posts == []


def test_a_well_formed_readout_row_still_reads(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"callo_bias_raw": 1}) + "\n" + json.dumps(_readout(probs=[1, 0])) + "\n")
    assert cb.read_raw(raw).readings == {("x", 0): ((1, 0), 0)}


def test_resume_refuses_a_readout_whose_length_is_not_its_questions_n(stub, tmp_path):
    """The raw row's own n can agree with its probs and still not be the question's n (a hand edit): --resume
    refuses instead of handing fit_n a ragged row."""
    _collect(stub, tmp_path)
    three = next(it for it in read_set(set_text()).items if it.n == 3)
    raw = tmp_path / "raw.jsonl"
    with raw.open("a") as f:
        f.write(json.dumps(_readout(q=three.uid, n=2, probs=[0.5, 0.5])) + "\n")
    before = len(stub.posts)
    with pytest.raises(Refusal, match=f"refusing to resume: .*{three.uid}.* 2 probabilities, but the question has 3"):
        _collect(stub, tmp_path, resume=True)
    assert len(stub.posts) == before


# ---- unusable probabilities, the summary's notes, the --data cap, line splitting ---------------------------

@pytest.mark.parametrize("probs", [{"a": 0.0, "b": 0.0}, {"a": 10**400, "b": 0}])
def test_a_reply_with_all_zero_or_overflowing_probabilities_is_no_usable_answer(stub, tmp_path, monkeypatch, probs):
    real = stub.readout

    def bad(body):
        out, timings = real(body)
        qid = next(iter(out["answers"]))
        out["answers"][qid]["probabilities"] = {k: probs[k] for k in ("a", "b")}
        return out, timings

    monkeypatch.setattr(stub, "readout", bad)
    with pytest.raises(RunFailed, match="a 200 reply with no usable answer"):
        _collect(stub, tmp_path, set_text(ns=(2,)))


def _summary_profile(kinds, *, builtin=False):
    """The fields summary_lines reads, for a profile with no n read (entries and fits empty)."""
    return {"model_key": KEY, "model_name": "m", "entries": {}, "server_state": {"state": "idle"},
            "set": {"name": "s.jsonl", "questions": 0, "kinds": kinds, "skipped": {}, "builtin": builtin},
            "eval": {"raw": {}, "debiased": {}}}


@pytest.mark.parametrize(("kinds", "note"), [
    (["plain"], "note: this profile corrects plain readouts only, not context, image, audio, video or mixed readouts; "
                "include such records in --data to have them corrected"),
    (["context", "plain"], "note: this profile corrects plain and context readouts only, not image, audio, video or "
                           "mixed readouts; include such records in --data to have them corrected"),
    (["image"], "note: this profile corrects image readouts only, not plain, context, audio, video or mixed readouts; "
                "include such records in --data to have them corrected"),
    (sorted(("plain", "context", "image", "audio", "video", "mixed")), None),
])
def test_the_summary_names_every_readout_kind_the_profile_does_not_correct(kinds, note):
    """A readout's kind is exclusive (systemone.media_kind): a file of plain records corrects plain readouts only, and a
    file with `context` records corrects context readouts too; the note names every kind of READOUT_KINDS left out."""
    lines = [ln for ln in cb.summary_lines(_summary_profile(kinds), {"fits": {}}) if ln.startswith("note:")]
    assert lines == ([note] if note else [])
    assert not [ln for ln in cb.summary_lines(_summary_profile(kinds, builtin=True), {"fits": {}})
                if ln.startswith("note:")]                              # the diagnostic corrects nothing: no note


def test_the_busy_warning_talks_about_a_profile_only_when_there_is_one():
    """Ledger minor: the diagnostic writes no profile, so its busy WARNING must not say `callo serve` loads one."""
    busy = {**_summary_profile(["plain"]), "server_state": {"state": "busy"}}
    fitted = [ln for ln in cb.summary_lines(busy, {"fits": {}}) if ln.startswith("WARNING")]
    assert fitted == ["WARNING: server_state busy: a server-state check failed and --allow-busy carried on; other "
                      "slots' state may be part of this prior, and `callo serve` will warn when it loads it"]
    busy["set"] = {**busy["set"], "builtin": True}
    diag = [ln for ln in cb.summary_lines(busy, {"fits": {}}) if ln.startswith("WARNING")]
    assert diag == ["WARNING: server_state busy: a server-state check failed and --allow-busy carried on; other "
                    "slots' state may be part of these figures"]


def test_a_data_run_resumes_under_the_default_no_cap_and_a_raw_capped_at_48_is_refused_by_name(tmp_path, capsys):
    """With --data the default --max-per-n is NO_CAP, recorded in the raw header, so the same
    command resumes; a raw file written with --max-per-n 48 (the diagnostic's cap) is a different run, and a --data
    resume without that flag is refused naming max_per_n (no silent mix of capped and uncapped readouts)."""
    data = _write_set(tmp_path, set_text(ns=(3,), per=60))
    stub = CalloStub(PlantedBias(priors={3: PRIOR3}))
    base = ["calibrate", "--bias", "--url", stub.url, "--data", str(data)]
    try:
        stub.fail_after = (30, 400)
        assert main([*base, "--out", str(tmp_path / "all.json")]) == 1
        assert _lines(tmp_path / "all.raw.jsonl")[0]["max_per_n"] == cb.NO_CAP
        stub.fail_after = None
        assert main([*base, "--out", str(tmp_path / "all.json"), "--resume"]) == 0
        assert json.loads((tmp_path / "all.json").read_text())["entries"]["3"]["questions"] == 60
        stub.fail_after = (stub.readouts + 30, 400)
        assert main([*base, "--max-per-n", "48", "--out", str(tmp_path / "cap.json")]) == 1
        stub.fail_after, posts = None, len(stub.posts)
        capsys.readouterr()
        assert main([*base, "--out", str(tmp_path / "cap.json"), "--resume"]) == 2
        err = capsys.readouterr().err
        assert "refusing to resume" in err and "max_per_n" in err and len(stub.posts) == posts
        assert main([*base, "--max-per-n", "48", "--out", str(tmp_path / "cap.json"), "--resume"]) == 0
    finally:
        stub.stop()
    assert json.loads((tmp_path / "cap.json").read_text())["entries"]["3"]["questions"] == 48


def test_read_raw_splits_lines_on_newline_only(tmp_path):
    """The raw readers split on "\\n" only, as the --data readers now do."""
    p = tmp_path / "raw.jsonl"
    p.write_text(json.dumps({"callo_bias_raw": 1}) + "\n" + json.dumps(
        {"type": "readout", "q": "L1/q\u2028x", "n": 2, "s": 0, "probs": [0.5, 0.5], "miss": 0, "slot": 0},
        ensure_ascii=False) + "\n", encoding="utf-8")
    assert cb.read_raw(p).readings == {("L1/q\u2028x", 0): ((0.5, 0.5), 0)}
