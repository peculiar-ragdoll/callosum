"""The live half of the v0.3 position-bias acceptance bench, GPU-free: a smoke run of the whole orchestration
against callo on the planted-prior fake upstream, the B4 reasons, the held-out read retry, the early abort, and the
safety checks (refusal to start, the watchdog, disk space)."""
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest
from fake_upstream import FakeUpstream, PlantedBias
from test_bench_v02 import AttachedCallo, _tiny_gguf
from test_calibrate_bias import set_text

from bench import acceptance_v03_bias as v03
from bench import bias_analysis as ba
from bench import lib
from bench.report_v03_bias import render
from callo.profile import load_profile

PRIOR = [0.5, 0.3, 0.2]
GIB = 1024**3
HEAD = ["--expect-head", "c0ffee0"]              # _quiet's code_state resolves it to the same "sha"


def _small_sets(tmp_path):
    fit = tmp_path / "fit.jsonl"
    fit.write_text(set_text(ns=(2, 3), per=30))
    ev = tmp_path / "eval.jsonl"
    ev.write_text(set_text(ns=(3,), per=30).replace("state ", "eval "))
    return fit, ev


def _quiet(monkeypatch, tmp_path):
    """No model server, plenty of memory and disk: what main checks before it starts."""
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(lib, "free_memory_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(v03, "DISK_PATHS", (str(tmp_path),))
    monkeypatch.setattr(v03, "MIN_FREE_GIB", 0.0)
    prereg = tmp_path / "prereg.md"
    prereg.write_text("hypotheses\n")
    monkeypatch.setattr(v03, "PREREG", prereg)
    monkeypatch.setattr(v03, "pin_problems", lambda: [])
    fit, ev = _small_sets(tmp_path)                    # never the pre-registered files on the data volume (CI has none)
    monkeypatch.setattr(v03, "FIT_DATA", str(fit))
    monkeypatch.setattr(v03, "EVAL_DATA", str(ev))
    monkeypatch.setattr(v03, "code_state", lambda expect=None: {"head": "c0ffee0", "status": "", "dirty": "",
                                                                 "expected": "c0ffee0", "root": str(v03.ROOT),
                                                                 "sources": v03._sources()})


def test_the_whole_bench_runs_through_the_launch_seam_on_the_planted_fake(tmp_path, monkeypatch):
    g = _tiny_gguf(tmp_path, monkeypatch)
    fit, ev = _small_sets(tmp_path)
    ups = []

    def launch(name, out, *, model, flags, ctx):
        up = FakeUpstream(bias=PlantedBias(priors={3: PRIOR}), slot_save=True).start()
        up.props["model_path"] = str(g)
        ups.append(up)
        c = AttachedCallo(up, tmp_path / name, *flags)
        c.parallel, c.up = 4, up
        return c

    def decode(c):
        c.up.set_slot(2, is_processing=True)
        return lambda: c.up.set_slot(2, is_processing=False)

    _quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(v03, "_launch", launch)
    monkeypatch.setattr(v03, "start_decode", decode)
    monkeypatch.setattr(v03, "MODELS", {"tiel_kvu": (str(g), [], 8192), "qwen9b": (str(g), [], 8192)})
    monkeypatch.setattr(v03, "FIT_DATA", str(fit))
    monkeypatch.setattr(v03, "EVAL_DATA", str(ev))
    monkeypatch.setattr(ba, "B1_MIN_QUESTIONS", 10)
    try:
        rc = v03.main(["--out", str(tmp_path / "out"), *HEAD])          # both models, the real REQUIRED
    finally:
        for up in ups:
            up.stop()
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    m = s["models"]["tiel_kvu"]
    q = s["models"]["qwen9b"]
    assert len(ups) == 2 and q["fit_rc"] == 0 and {k: v["rc"] for k, v in q["reads"].items()} == {"eval": 0}
    assert q["b1"]["pass"] is True and q["b2"]["pass"] is True and q["b5"]["pass"] is True
    assert not {"b0", "b3", "b4"} & set(q) and "error" not in q
    probes = tmp_path / "out" / "tiel_kvu" / "b4_probes"            # never beside fit.json
    assert sorted(f.name for f in probes.iterdir()) == ["b4c.json", "b4c.raw.jsonl"]   # (a), (b) refused: no trace
    assert not list((tmp_path / "out" / "tiel_kvu").glob("b4?.*"))
    assert m["b0"]["pass"] is True and m["b4"]["pass"] is True
    assert (m["b4"]["a_rc"], m["b4"]["b_rc"], m["b4"]["c_rc"]) == (2, 2, 0)
    assert m["b4"]["a_failed"] == ["foreign_tokens"] and "foreign_processing" in m["b4"]["b_failed"]
    assert m["b3"]["identical"] is True and m["b3"]["applied"] > 0
    assert {k: v["rc"] for k, v in m["reads"].items()} == {"eval": 0}
    assert m["b1"]["per_n"]["3"]["tested"] is True and m["b1"]["pass"] is True and m["b2"]["pass"] is True
    assert m["b5"]["pass"] is True
    assert m["entries"] == ["3"] and "why_we_refuse" not in m and rc == 0
    assert s["g0_frames"]["pass"] is None and "not run" in s["g0_frames"]["note"]
    assert s["prereg"]["verdict"] == "CONFIRMED" and s["prereg"]["b1_tested"] == ["tiel_kvu n=3", "qwen9b n=3"]
    assert s["prereg"]["sha256"] == hashlib.sha256(b"hypotheses\n").hexdigest()
    assert s["data"]["eval"] == {"path": str(ev), "sha256": hashlib.sha256(ev.read_bytes()).hexdigest()}
    assert "busy_fit_rc" not in m and not (tmp_path / "out" / "tiel_kvu" / "busy_fit.raw.jsonl").exists()
    assert s["code"]["head"] == "c0ffee0" and s["sessions"][0]["expect_head"] == "c0ffee0"
    assert m["b3"]["synthetic"] is False
    assert v03.main(["--out", str(tmp_path / "out"), *HEAD]) == 2                             # no re-roll
    progress = json.loads((tmp_path / "out" / "progress.json").read_text())
    assert progress["step"] == "finished" and progress["done"] == ["tiel_kvu", "qwen9b"]
    assert not (tmp_path / "out" / "run.pid").exists() and "B4" in render(s)


def test_the_run_refuses_unpinned_data_and_a_missing_prereg(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("x\n")
    b.write_text("y\n")
    prereg = tmp_path / "prereg.md"
    pins = {str(a): hashlib.sha256(b"x\n").hexdigest(), str(b): "0" * 64, str(tmp_path / "gone.jsonl"): "1" * 64}
    problems = v03.pin_problems(pins, prereg)
    assert len(problems) == 3 and f"{b} has sha256" in problems[0]
    assert "not the pre-registered " + "0" * 64 in problems[0]
    assert "gone.jsonl: cannot read it" in problems[1] and problems[2].endswith("must follow its pre-registration")
    prereg.write_text("h\n")
    assert v03.pin_problems({str(a): pins[str(a)]}, prereg) == []
    assert set(v03.DATA_SHA256) == {v03.FIT_DATA, v03.EVAL_DATA} and "tiel_nokvu" not in v03.MODELS


def test_main_refuses_unpinned_data(tmp_path, monkeypatch, capsys):
    _quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(v03, "pin_problems", lambda: ["/v/test.jsonl has sha256 ab, not the pre-registered cd"])
    assert v03.main(["--out", str(tmp_path / "o"), *HEAD]) == 2
    assert "refusing to start: /v/test.jsonl has sha256 ab" in capsys.readouterr().out


def test_b0_fails_only_on_a_guard_refusal_or_a_busy_profile():
    """A connection or HTTP failure after retries is infrastructure (ERROR), not a falsified B0."""
    guard = "callo: refusing to calibrate ...\n(failed conditions: foreign_tokens)\n"
    assert v03.b0_verdict(0, {"server_state": {"state": "clean"}}, "")["pass"] is True
    assert v03.b0_verdict(2, None, guard) == {"rc": 2, "state": None, "failed": ["foreign_tokens"], "pass": False}
    assert v03.b0_verdict(0, {"server_state": {"state": "busy"}}, "")["pass"] is False
    for rc, err in ((1, "callo: x: still failing after 3 retries"), (2, "callo: refusing to calibrate: T is 0.7")):
        b0 = v03.b0_verdict(rc, None, err)
        assert b0["pass"] is None and "infrastructure, not B0" in b0["error"]


def test_the_run_refuses_a_dirty_checkout_or_another_head(tmp_path):
    """The code that runs is the pre-registered commit, clean; HEAD and the status are recorded."""
    def git(*a):
        subprocess.run(["git", "-c", "user.email=x@y", "-c", "user.name=t", *a], cwd=tmp_path, check=True,
                       capture_output=True)
    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-qm", "c")
    here = {"callo": str(tmp_path / "src" / "callo"), "bench": str(tmp_path / "bench")}
    first = v03.code_state(tmp_path)["head"]
    (tmp_path / "src" / "b.py").write_text("y = 1\n")
    git("add", "-A")
    git("commit", "-qm", "d")
    clean = v03.code_state(tmp_path, expect=v03.code_state(tmp_path)["head"][:8], sources=here)   # short sha
    assert len(clean["head"]) == 40 and clean["expected"] == clean["head"] and clean["dirty"] == ""
    assert v03.code_problems(clean, clean["head"][:8]) == []
    for ref in ("HEAD", "@", "master", "c0ffe"):                    # a ref always matches; too short
        sym = v03.code_state(tmp_path, expect=ref, sources=here)
        assert "is not a commit sha (7 to 40 hex digits)" in " ".join(v03.code_problems(sym, ref))
    assert v03.code_problems(clean, clean["head"][:8], pinned=clean["head"]) == []
    assert f"is not the prereg's pinned commit C {first}" in v03.code_problems(clean, clean["head"][:8], first)[0]
    real = v03.code_state(tmp_path, expect=clean["head"])           # this process imports callo from ROOT
    assert real["sources"] == v03._sources() and real["root"] == str(tmp_path.resolve())
    assert [p for p in v03.code_problems(real, clean["head"]) if "is imported from" in p] == [
        f"callo is imported from {v03._sources()['callo']}, not from {tmp_path / 'src'}: run with this checkout's own "
        f"environment (`uv run --frozen` from it)",
        f"bench is imported from {v03._sources()['bench']}, not from {tmp_path / 'bench'}: run with this checkout's "
        f"own environment (`uv run --frozen` from it)"]
    ours = v03.code_state()                                         # this checkout imports its own callo and bench
    assert [p for p in v03.code_problems(ours, ours["head"]) if "is imported from" in p] == []
    older = v03.code_state(tmp_path, expect=first[:8])
    assert f"not --expect-head {first[:8]} ({first})" in v03.code_problems(older, first[:8])[0]
    assert "does not name a commit" in v03.code_problems(v03.code_state(tmp_path, expect="0" * 40), "0" * 40)[0]
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    (tmp_path / "notes.txt").write_text("outside TRACKED\n")
    dirty = v03.code_state(tmp_path, sources=here)
    assert dirty["dirty"] == " M src/a.py" and dirty["status"].startswith(" M src/a.py")   # intact
    assert "?? notes.txt" in dirty["status"]
    assert "uncommitted changes under src" in v03.code_problems(dirty, clean["head"][:8])[0]
    nope = v03.code_state(tmp_path / "src" / "nope")
    assert "run from a git worktree" in v03.code_problems(nope, clean["head"][:8])[0]


def test_the_pinned_commit_is_read_from_the_prereg_once_it_is_appended(tmp_path):
    prereg = tmp_path / "prereg.md"
    assert v03.pinned_commit(prereg) is None                        # unreadable
    prereg.write_text("records C by appending `Pinned commit C: <40-hex sha> (...)`\n")
    assert v03.pinned_commit(prereg) is None                        # the prereg's own description is not a pin
    prereg.write_text(prereg.read_text() + "Pinned commit C: " + "ab" * 20 + " (Tasks 1-6 and their fixes)\n")
    assert v03.pinned_commit(prereg) == "ab" * 20
    real = v03.pinned_commit()                                      # None until the pin is appended, its 40-hex C after
    assert real is None or (len(real) == 40 and set(real) <= set("0123456789abcdef"))


def test_a_final_verdict_is_never_rerolled_unless_overridden_and_recorded(tmp_path, monkeypatch, capsys):
    _quiet(monkeypatch, tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"models": {}, "prereg": {"verdict": "FALSIFIED"}}))
    assert v03.main(["--out", str(out), *HEAD]) == 2
    assert "already holds the final pre-registered verdict FALSIFIED" in capsys.readouterr().out
    monkeypatch.setattr(v03, "MODELS", {"tiel_kvu": ("m.gguf", [], 8192)})
    monkeypatch.setattr(v03, "run_model", lambda name, o, prev=None: {"b1": {"pass": True}})
    v03.main(["--out", str(out), "--override-final", "re-judged after a harness fix", *HEAD])
    s = json.loads((out / "summary.json").read_text())
    assert s["overrides"][0]["previous_verdict"] == "FALSIFIED"
    assert s["overrides"][0]["reason"] == "re-judged after a harness fix"


def test_a_rerun_keeps_judged_models_and_gates_and_completes_only_what_is_missing(tmp_path, monkeypatch):
    """A model already judged is skipped; an errored one keeps its judged gates (B0, B4, B3 are not
    re-run) and its earlier partial result survives a second failure."""
    _quiet(monkeypatch, tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    b4_fail = {"a_rc": 0, "a_failed": [], "b_rc": 2, "b_failed": ["foreign_tokens"], "c_rc": 0, "pass": False}
    done = {"b1": {"pass": True}, "b2": {"pass": True}, "b5": {"pass": True}}
    (out / "summary.json").write_text(json.dumps({"models": {
        "qwen9b": done, "tiel_kvu": {"b0": {"rc": 0, "state": "clean", "pass": True}, "b4": b4_fail,
                                     "error": "OSError: eval read failed"}}}))
    calls = []
    monkeypatch.setattr(v03, "run_model", lambda name, o, prev=None: calls.append((name, prev)) or (
        _ for _ in ()).throw(RuntimeError("again")))
    assert v03.main(["--out", str(out), *HEAD]) == 1
    s = json.loads((out / "summary.json").read_text())
    assert [n for n, _ in calls] == ["tiel_kvu"] and calls[0][1]["b4"] == b4_fail      # qwen9b frozen
    assert s["models"]["qwen9b"] == done and s["models"]["tiel_kvu"]["b4"] == b4_fail
    assert s["models"]["tiel_kvu"]["error"] == "RuntimeError: again"
    assert s["prereg"]["verdict"] == "FALSIFIED"                    # the B4 FAIL measured before the error decides
    seen = []
    monkeypatch.setattr(v03, "calibrate", lambda *a, **kw: seen.append(a[2]) or (0, None, tmp_path / "r", ""))
    (tmp_path / "fit.json").write_text("{}")
    kept = v03.b0_b4(None, tmp_path, {"b0": {"rc": 0, "state": "clean", "pass": True}, "b4": b4_fail})
    assert kept == {"b0": {"rc": 0, "state": "clean", "pass": True}, "b4": b4_fail} and seen == []


@pytest.mark.parametrize("rc", [0, 1])
def test_a_judged_b0_whose_fit_json_is_gone_is_refitted_not_rejudged(tmp_path, monkeypatch, rc):
    """B0 stays as judged, but B3 and B1/B2/B5 need fit.json, so a missing one is fitted again (the
    same command, which resumes from fit.raw.jsonl when it is still there); a refit that fails stops the model early
    like a failed fit, instead of crashing at B3."""
    b0 = {"rc": 0, "state": "clean", "pass": True}
    seen = []
    monkeypatch.setattr(v03, "calibrate", lambda *a, **kw: seen.append(a[2]) or (rc, None, tmp_path / "r", ""))
    out = v03.b0_b4(None, tmp_path, {"b0": b0, "b4": {"pass": True}})
    assert seen == ["fit"] and out["b0"] == b0 and out["b4"] == {"pass": True} and out["refit"]["rc"] == rc
    assert "B0 stays as judged" in out["refit"]["note"]
    seen.clear()                                                    # a B0 that never produced a fit.json: no refit
    assert "refit" not in v03.b0_b4(None, tmp_path, {"b0": {"rc": 2, "pass": False}, "b4": {"pass": True}})
    assert seen == []
    _quiet(monkeypatch, tmp_path)
    box = _Box(pid=None, upstream="http://up", url="http://callo", build_info="fake", snapshot={"kv": "unified"},
               finish=lambda: None)
    monkeypatch.setattr(v03, "_launch", lambda name, out, **kw: box)
    monkeypatch.setattr(v03, "MODELS", {"tiel_kvu": ("m.gguf", [], 8192)})
    monkeypatch.setattr(v03, "read_heldout", lambda *a: (_ for _ in ()).throw(AssertionError("read anyway")))
    res = v03.run_model("tiel_kvu", tmp_path, {"b0": b0, "b4": {"pass": True}})
    assert res["b0"] == b0 and res["refit"]["rc"] == rc
    if rc:
        assert "the fit exited 1" in res["aborted"] and "b3" not in res
    else:
        assert "the fit exited" not in res.get("aborted", "")


@pytest.mark.parametrize(("name", "partial"), [
    ("tiel_kvu", {"b0": {"rc": 1, "pass": None, "error": "the fit exited 1 without a guard refusal"},
                  "aborted": "the fit exited 1: held-out reads and B3 skipped"}),
    ("qwen9b", {"fit_rc": 1, "aborted": "the fit exited 1: held-out reads and B3 skipped",
                "b1": {"pass": None, "error": "not judged on partial held-out data: eval not run"}}),
    ("qwen9b", {"b1": {"pass": None, "error": "not judged on partial held-out data: eval exited 1 with 9 readouts"},
                "b2": {"pass": None, "error": "x"}, "b5": {"pass": None, "error": "x"}}),
])
def test_a_model_with_an_unjudged_gate_is_run_again(tmp_path, monkeypatch, name, partial):
    """B0 infrastructure ERROR, a failed fit and a failed eval read leave no top-level `error`, yet the
    model is not complete: a rerun must complete it, with its partial result as `prev`."""
    _quiet(monkeypatch, tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"models": {name: partial}}))
    seen = []
    monkeypatch.setattr(v03, "run_model", lambda n, o, prev=None: seen.append((n, prev)) or {"b1": {"pass": True}})
    v03.main(["--out", str(out), "--models", name, *HEAD])
    assert seen == [(name, partial)] and not v03.model_complete(name, partial)


def test_model_complete_needs_every_required_gate_judged():
    done = {"b1": {"pass": None, "per_n": {}}, "b2": {"pass": False}, "b5": {"pass": None, "note": "nothing corrected"}}
    assert v03.model_complete("qwen9b", done)
    assert not v03.model_complete("qwen9b", {**done, "error": "OSError: x"})
    assert not v03.model_complete("qwen9b", {k: v for k, v in done.items() if k != "b2"})
    assert not v03.model_complete("tiel_kvu", done)                 # B0, B3, B4 missing
    tiel = {**done, **{g: {"pass": True} for g in ("b0", "b3", "b4")}}
    assert v03.model_complete("tiel_kvu", tiel)
    assert not v03.model_complete("tiel_kvu", {**tiel, "b3": {"pass": None, "note": "not run"}})


def test_an_exception_after_the_gates_keeps_them_next_to_the_error(tmp_path, monkeypatch):
    _quiet(monkeypatch, tmp_path)
    box = _Box(pid=None, upstream="http://up", url="http://callo", build_info="fake", snapshot={"kv": "unified"},
               finish=lambda: None)
    monkeypatch.setattr(v03, "_launch", lambda name, out, **kw: box)
    monkeypatch.setattr(v03, "MODELS", {"tiel_kvu": ("m.gguf", [], 8192)})
    b4 = {"pass": False, "a_rc": 0}
    monkeypatch.setattr(v03, "b0_b4", lambda c, d, prev=None: {"b0": {"rc": 0, "state": "clean", "pass": True},
                                                                "b4": b4})
    _b3_ready(monkeypatch, tmp_path)
    monkeypatch.setattr(v03, "read_heldout", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    res = v03.run_model("tiel_kvu", tmp_path)
    assert res["b4"] == b4 and res["error"] == "OSError: disk" and "b1" not in res
    assert res["b3"]["pass"] is True                                # the pre-registered order: B3 before the eval read


def _b3_ready(monkeypatch, tmp_path):
    """What B3 needs once the fit has run: a fit.json, an eval set, and a stand-in for the upstream call sequence."""
    (tmp_path / "tiel_kvu").mkdir(parents=True, exist_ok=True)
    ev = tmp_path / "eval.jsonl"
    ev.write_text(set_text(ns=(2, 3), per=4))
    monkeypatch.setattr(v03, "EVAL_DATA", str(ev))
    (tmp_path / "tiel_kvu" / "fit.json").write_text(json.dumps({"entries": {}, "model_key": "sha256:" + "0" * 64}))
    monkeypatch.setattr(v03, "b3_call_sequence", lambda up, gguf, prof, bodies: {
        "calls": len(bodies), "identical": True, "applied": len(bodies), "pass_calls": True})


def _gates_then(monkeypatch, tmp_path, stop):
    """tiel_kvu judges B0 and B4 and B3, then `stop()` raises during the eval read."""
    box = _Box(pid=None, upstream="http://up", url="http://callo", build_info="fake", snapshot={"kv": "unified"},
               finish=lambda: None)
    monkeypatch.setattr(v03, "_launch", lambda name, out, **kw: box)
    monkeypatch.setattr(v03, "MODELS", {"tiel_kvu": ("m.gguf", [], 8192)})
    gates = {"b0": {"rc": 0, "state": "clean", "pass": True}, "b4": {"pass": True, "a_rc": 2}}
    monkeypatch.setattr(v03, "b0_b4", lambda c, d, prev=None: dict(gates))
    _b3_ready(monkeypatch, tmp_path / "o")
    monkeypatch.setattr(v03, "read_heldout", lambda *a: stop())
    return gates


def test_a_watchdog_stop_mid_model_keeps_the_judged_gates_for_the_rerun(tmp_path, monkeypatch):
    """B0/B4/B3 judged before the watchdog stopped the eval read are in the summary written on exit 3,
    and the rerun gets them as `prev` (so B4 is not measured again)."""
    _quiet(monkeypatch, tmp_path)

    def trip():
        v03._WATCH.reason = "another model server appeared during the run: pid 9"
        raise v03.Aborted(v03._WATCH.reason)

    gates = _gates_then(monkeypatch, tmp_path, trip)
    out = tmp_path / "o"
    assert v03.main(["--out", str(out), "--models", "tiel_kvu", *HEAD]) == 3
    m = json.loads((out / "summary.json").read_text())["models"]["tiel_kvu"]
    assert m["b0"] == gates["b0"] and m["b4"] == gates["b4"] and m["b3"]["pass"] is True
    assert m["error"] == "interrupted: another model server appeared during the run: pid 9"
    assert not v03.model_complete("tiel_kvu", m) and v03._PARTIAL == {}
    seen = []
    monkeypatch.setattr(v03, "run_model", lambda n, o, prev=None: seen.append(prev) or {"b1": {"pass": True}})
    v03.main(["--out", str(out), "--models", "tiel_kvu", *HEAD])
    assert seen[0]["b0"] == gates["b0"] and seen[0]["b4"] == gates["b4"] and seen[0]["b3"]["pass"] is True


def test_a_stop_signal_mid_model_keeps_the_judged_gates_and_still_stops(tmp_path, monkeypatch):
    """A person's Ctrl-C (or the SIGTERM handler's SystemExit) still ends the run, after summary.json kept the gates."""
    _quiet(monkeypatch, tmp_path)
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "summary.json").write_text(json.dumps({"models": {"tiel_kvu": {
        "b0": {"rc": 1, "pass": None, "error": "infra"}, "error": "OSError: earlier"}}}))

    def ctrl_c():
        raise KeyboardInterrupt

    gates = _gates_then(monkeypatch, tmp_path, ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        v03.main(["--out", str(tmp_path / "o"), "--models", "tiel_kvu", *HEAD])
    m = json.loads((tmp_path / "o" / "summary.json").read_text())["models"]["tiel_kvu"]
    assert m["b0"] == gates["b0"] and m["b4"] == gates["b4"]                # this session's gates win over the old
    assert m["error"] == "interrupted: a stop signal (KeyboardInterrupt)"


@pytest.mark.parametrize(("code", "why"), [
    (128 + signal.SIGTERM, "interrupted: a stop signal (SIGTERM)"),
    (2, "interrupted: an exit (SystemExit 2, e.g. an argparse usage error inside cli.main), not a stop signal")])
def test_an_exit_mid_model_is_labelled_for_what_it_was(tmp_path, monkeypatch, code, why):
    """Ledger minor: the stop handlers raise SystemExit(128 + signum); any other SystemExit (argparse inside cli.main)
    is not a stop signal, and summary.json must not call it one."""
    _quiet(monkeypatch, tmp_path)

    def leave():
        raise SystemExit(code)

    _gates_then(monkeypatch, tmp_path, leave)
    with pytest.raises(SystemExit):
        v03.main(["--out", str(tmp_path / "o"), "--models", "tiel_kvu", *HEAD])
    assert json.loads((tmp_path / "o" / "summary.json").read_text())["models"]["tiel_kvu"]["error"] == why


def test_a_required_model_left_out_of_models_exits_1_with_the_reason(tmp_path, monkeypatch, capsys):
    """Every tiel gate passes, but qwen9b never ran, so the pre-registered verdict is ERROR: the run
    exits 1 (could not be judged), not 0."""
    _quiet(monkeypatch, tmp_path)
    tested = {"per_n": {"3": {"tested": True, "pass": True}}, "pass": True}
    monkeypatch.setattr(v03, "run_model", lambda name, o, prev=None: {
        **{g: {"pass": True} for g in ("b0", "b2", "b3", "b4", "b5")}, "b1": tested})
    assert v03.main(["--out", str(tmp_path / "o"), "--models", "tiel_kvu", *HEAD]) == 1
    out = capsys.readouterr().out
    assert "pre-registered verdict: ERROR" in out and "qwen9b: not run" in out
    assert "required by the prereg but not in --models: qwen9b" in out


def test_failed_conditions_reads_the_guards_last_line():
    err = "callo: refusing to calibrate: ...\nfixes:\n- x\n(failed conditions: foreign_tokens, foreign_s1)\nmore\n"
    assert v03.failed_conditions(err) == ["foreign_tokens", "foreign_s1"]
    assert v03.failed_conditions("callo: HTTP 400") == []


class _Box(SimpleNamespace):
    """A callo stand-in for the B4 setup checks: stats() returns the scripted /callo/stats."""

    def stats(self):
        return self.snapshot


def test_b4_refuses_to_judge_when_its_load_never_shows(tmp_path, monkeypatch):
    """A fill that did not land must not let (a) pass on some other refusal."""
    monkeypatch.setattr(v03, "SNAPSHOT_S", 0.0)
    monkeypatch.setattr(lib, "post", lambda url, payload, **kw: (200, {}, {}))
    box = _Box(upstream="http://up", parallel=4, snapshot={"slots": [{"id": i} for i in range(4)],
                                                           "router": {"reserved": [0]}})
    calls = []
    monkeypatch.setattr(v03, "calibrate", lambda *a, **kw: calls.append(a) or (0, {"server_state": {"state": "clean"}},
                                                                                tmp_path / "r", ""))
    out = v03.b0_b4(box, tmp_path)
    assert out["b0"]["pass"] is True and out["b4"]["pass"] is None
    assert "B4 setup" in out["b4"]["error"] and "slot 3" in out["b4"]["error"] and len(calls) == 1


def test_b4_setup_fails_on_a_non_200_fill_or_erase(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "post", lambda url, payload, **kw: (501, {}, {"error": "no --slot-save-path"}))
    box = _Box(upstream="http://up", parallel=4, snapshot={})
    with pytest.raises(v03.B4Setup, match=r"HTTP 501") as e:
        v03._erase_all(box)
    assert "/slots/0?action=erase" in str(e.value)
    with pytest.raises(v03.B4Setup, match="filling slot 3"):
        v03._fill(box, 3, 300)


def test_a_failed_held_out_read_is_resumed_once(tmp_path, monkeypatch):
    rcs = [1, 0]
    monkeypatch.setattr(v03, "calibrate", lambda c, d, run, data, *extra, **kw: (rcs.pop(0), None,
                                                                                  d / f"{run}.raw.jsonl", ""))
    assert v03.read_heldout(None, tmp_path, "heldout", None) == {"rc": 0, "readouts": 0, "attempts": 2}
    rcs[:] = [2, 2]
    assert v03.read_heldout(None, tmp_path, "heldout", None)["rc"] == 2


def test_a_failed_fit_skips_the_held_out_reads(tmp_path, monkeypatch):
    """A refused B0 must not spend an hour on held-out reads it cannot judge."""
    _quiet(monkeypatch, tmp_path)
    box = _Box(pid=None, upstream="http://up", url="http://callo", build_info="fake", snapshot={"kv": "unified"},
               finish=lambda: None)
    monkeypatch.setattr(v03, "_launch", lambda name, out, **kw: box)
    monkeypatch.setattr(v03, "MODELS", {"tiel_kvu": ("m.gguf", [], 8192)})
    monkeypatch.setattr(v03, "b0_b4", lambda c, d, prev=None: {"b0": {"rc": 2, "state": None, "pass": False}})
    monkeypatch.setattr(v03, "read_heldout", lambda *a: (_ for _ in ()).throw(AssertionError("read anyway")))
    res = v03.run_model("tiel_kvu", tmp_path)
    assert "the fit exited 2" in res["aborted"] and "b3" not in res
    assert res["b1"]["error"].startswith("not judged on partial held-out data")


def test_main_refuses_while_a_model_server_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [(4242, "/x/llama-server -m y.gguf")])
    assert v03.main(["--out", str(tmp_path / "o"), *HEAD]) == 2
    assert "refusing to start" in capsys.readouterr().out


def test_main_refuses_on_a_full_or_missing_volume(tmp_path, monkeypatch, capsys):
    _quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(v03, "DISK_PATHS", (str(tmp_path / "not-mounted"),))
    assert v03.main(["--out", str(tmp_path / "o"), *HEAD]) == 2
    assert "not-mounted" in capsys.readouterr().out


def test_disk_problems_name_each_short_or_unreadable_volume():
    free = {"/a": 50 * GIB, "/b": 5 * GIB}

    def lookup(p):
        if p not in free:
            raise FileNotFoundError(p)
        return free[p]

    problems = v03.disk_problems(["/a", "/b", "/nope"], 20, free=lookup)
    assert problems[0] == "/b has 5.0 GiB free (< 20 GiB)" and "/nope" in problems[1] and len(problems) == 2


def test_the_watchdog_ignores_the_benchs_own_server_and_trips_on_any_other(tmp_path, monkeypatch):
    seen = [[(10, "llama-server -m a.gguf")], [(10, "llama-server -m a.gguf"), (11, "ollama runner")]]
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: seen.pop(0) if len(seen) > 1 else seen[0])
    tripped = threading.Event()
    w = v03.Watchdog([str(tmp_path)], interrupt=tripped.set, own=lambda: {10}, interval=0.01, min_free_gib=0)
    w.start()
    assert tripped.wait(5)
    assert "pid 11 (ollama runner)" in w.reason and "pid 10" not in w.reason


def test_a_foreign_server_mid_run_stops_the_bench_cleanly_with_exit_3(tmp_path, monkeypatch, capsys):
    """The watchdog interrupts the main thread; every finally runs; summary.json says why."""
    calls = []

    def servers():
        calls.append(1)
        return [] if len(calls) == 1 else [(4242, "/opt/homebrew/bin/ollama runner [resident 9.0 GiB]")]

    _quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(lib, "running_gpu_servers", servers)
    monkeypatch.setattr(v03, "WATCH_S", 0.05)
    stopped = []

    def run_model(name, out, prev=None):
        try:
            while True:
                time.sleep(0.01)
        finally:
            stopped.append(name)

    monkeypatch.setattr(v03, "run_model", run_model)
    assert v03.main(["--out", str(tmp_path / "o"), "--models", "tiel_kvu", *HEAD]) == 3
    assert stopped == ["tiel_kvu"]
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert "pid 4242" in s["aborted"] and "ollama runner" in s["aborted"]
    assert "ABORTED" in capsys.readouterr().out and not (tmp_path / "o" / "run.pid").exists()



def test_the_watchdog_counts_a_server_still_loading_its_model_as_the_benchs_own(tmp_path, monkeypatch):
    """A launch blocks until llama-server is ready (minutes for a 35B), so c.pid is not known yet: the pid file the
    launch writes into its cache dir while the model loads must keep the watchdog from calling it foreign."""
    seen = {}

    def launch(name, out, *, model, flags, ctx):
        cache = out / f"callo_{name}" / "cache"
        cache.mkdir(parents=True)
        (cache / "llama-server-starting-99.pid").write_text(f"{os.getpid()}\n")    # a live pid
        seen["own"] = v03._own_pids()
        raise RuntimeError("stop after the launch")

    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(v03, "_launch", launch)
    monkeypatch.setattr(v03, "MODELS", {"tiel_kvu": ("m.gguf", [], 8192)})
    with pytest.raises(RuntimeError, match="stop after the launch"):
        v03.run_model("tiel_kvu", tmp_path)
    assert os.getpid() in seen["own"] and os.getpid() not in v03._own_pids()


def test_the_own_pid_check_only_reads_the_pid_files(tmp_path, monkeypatch):
    """A pid file being written (empty or partial) must not be deleted, as upstream.find_orphans would."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "llama-server-starting-1.pid").write_text("")
    (cache / "llama-server-8080.pid").write_text(f"{os.getpid()}\n")
    (cache / "llama-server-8081.pid").write_text("999999999\n")                        # no such process
    monkeypatch.setattr(v03, "_CACHES", {cache})
    assert v03._own_pids() == {os.getpid()}
    assert sorted(f.name for f in cache.iterdir()) == ["llama-server-8080.pid", "llama-server-8081.pid",
                                                       "llama-server-starting-1.pid"]


def test_a_pid_file_renamed_between_glob_and_read_is_found_by_a_second_glob(tmp_path, monkeypatch):
    """The launch atomically renames llama-server-starting-*.pid to the port-named file once the server
    listens; a rename between the glob and the read must not make the loading server look foreign for a tick."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "llama-server-starting-1.pid").write_text(f"{os.getpid()}\n")
    read_text = Path.read_text

    def racing(self, *a, **kw):
        if self.name == "llama-server-starting-1.pid":
            os.replace(self, cache / "llama-server-8080.pid")                # the rename lands just before the read
        return read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", racing)
    monkeypatch.setattr(v03, "_CACHES", {cache})
    assert v03._own_pids() == {os.getpid()}


def test_the_launched_pid_is_own_before_the_launch_cache_is_forgotten(tmp_path, monkeypatch):
    seen = []

    class Caches(set):
        def discard(self, x):
            seen.append(set(v03._OWN))
            super().discard(x)

    def stats():
        raise RuntimeError("stop after the launch")

    box = _Box(pid=777, upstream="http://up", build_info="fake", stats=stats, finish=lambda: None)
    monkeypatch.setattr(v03, "_CACHES", Caches())
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    monkeypatch.setattr(v03, "_launch", lambda name, out, **kw: box)
    monkeypatch.setattr(v03, "MODELS", {"tiel_kvu": ("m.gguf", [], 8192)})
    res = v03.run_model("tiel_kvu", tmp_path)                    # recorded next to the gates, not raised
    assert res["error"] == "RuntimeError: stop after the launch"
    assert seen == [{777}] and 777 not in v03._OWN


def test_the_watchdog_lists_servers_before_it_asks_which_are_its_own(tmp_path, monkeypatch):
    """The other order lets a server that became ours between the two reads look foreign."""
    order = []
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: order.append("servers") or [])
    w = v03.Watchdog([str(tmp_path)], interrupt=lambda: None, own=lambda: order.append("own") or set(),
                     interval=0.01, min_free_gib=0)
    w.start()
    while len(order) < 2:
        time.sleep(0.01)
    w.stop()
    w.join(5)
    assert order[:2] == ["servers", "own"]


def test_a_stopped_watchdog_never_interrupts(tmp_path, monkeypatch):
    """A trip found on a tick that was already running when the run ended must not signal the process."""
    entered, release = threading.Event(), threading.Event()

    def servers():
        entered.set()
        release.wait(5)
        return [(11, "ollama runner")]

    monkeypatch.setattr(lib, "running_gpu_servers", servers)
    tripped = threading.Event()
    w = v03.Watchdog([str(tmp_path)], interrupt=tripped.set, own=set, interval=0.01, min_free_gib=0)
    w.start()
    assert entered.wait(5)
    w.stop()
    release.set()
    w.join(5)
    assert not w.is_alive() and not tripped.is_set() and w.reason is None


def test_a_trip_as_the_run_ends_still_exits_3(tmp_path, monkeypatch, capsys):
    """The worst race: the watchdog's SIGTERM lands while main is stopping it. main must still record the reason and
    return 3 (not escape with SystemExit 143), and restore the signal handlers."""
    _quiet(monkeypatch, tmp_path)
    before = signal.getsignal(signal.SIGTERM)

    class Racing(v03.Watchdog):
        def stop(self):
            super().stop()
            self.reason = "late trip"
            self.interrupt()

    monkeypatch.setattr(v03, "Watchdog", Racing)
    monkeypatch.setattr(v03, "run_model", lambda name, out, prev=None: {"b0": {"pass": True}})
    assert v03.main(["--out", str(tmp_path / "o"), "--models", "tiel_kvu", *HEAD]) == 3
    assert json.loads((tmp_path / "o" / "summary.json").read_text())["aborted"] == "late trip"
    assert "ABORTED" in capsys.readouterr().out and signal.getsignal(signal.SIGTERM) is before


def _b4_box(slot3: dict, reserved=(0,)):
    slots = [{"id": i} for i in range(3)] + [{"id": 3, **slot3}]
    return _Box(upstream="http://up", url="http://callo", parallel=4,
                snapshot={"slots": slots, "router": {"reserved": list(reserved)}})


def _b0_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(v03, "SNAPSHOT_S", 0.0)
    monkeypatch.setattr(v03, "calibrate",
                        lambda *a, **kw: (0, {"server_state": {"state": "clean"}}, tmp_path / "r", ""))


def test_b4a_load_must_sit_in_a_non_reserved_slot(tmp_path, monkeypatch):
    _b0_ok(monkeypatch, tmp_path)
    monkeypatch.setattr(lib, "post", lambda url, payload, **kw: (200, {}, {}))
    out = v03.b0_b4(_b4_box({"n_prompt_tokens": 300}, reserved=(3,)), tmp_path)
    assert out["b0"]["pass"] is True and out["b4"]["pass"] is None and "slot 3" in out["b4"]["error"]


def test_a_b4_setup_error_erases_every_slot_of_the_benchs_own_server(tmp_path, monkeypatch):
    """A half-done setup (here the fill landed but never showed) must not leave its tokens behind, or the
    held-out reads that follow would refuse on foreign_tokens."""
    _b0_ok(monkeypatch, tmp_path)
    posts = []
    monkeypatch.setattr(lib, "post", lambda url, payload, **kw: posts.append(url) or (200, {}, {}))
    out = v03.b0_b4(_b4_box({}), tmp_path)                                   # slot 3 never shows the fill
    assert out["b4"]["pass"] is None and "never showed" in out["b4"]["error"]
    assert posts == ["http://up/completion", *(f"http://up/slots/{s}?action=erase" for s in range(4))]
    assert out["b4"]["erase_after_error"] == "ok"


def test_the_erase_after_a_b4_setup_error_is_best_effort_and_says_what_failed(tmp_path, monkeypatch):
    _b0_ok(monkeypatch, tmp_path)
    posts = []

    def post(url, payload, **kw):
        posts.append(url)
        if url.endswith("/completion"):
            return 200, {}, {}
        if "/slots/1?" in url:
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        return 501, {}, {"error": "no --slot-save-path"}

    monkeypatch.setattr(lib, "post", post)
    out = v03.b0_b4(_b4_box({}), tmp_path)
    assert posts[1:] == [f"http://up/slots/{s}?action=erase" for s in range(4)]   # every slot tried
    left = out["b4"]["erase_after_error"]
    assert len(left) == 4 and "erasing slot 1" in left[1] and "Connection refused" in left[1] and "HTTP 501" in left[0]
    assert out["b4"]["error"].startswith("B4 setup:") and "erasing the slots afterwards failed" in out["b4"]["error"]


def test_a_decode_that_never_starts_is_a_b4_error_and_keeps_b0(tmp_path, monkeypatch):
    _b0_ok(monkeypatch, tmp_path)
    monkeypatch.setattr(lib, "post", lambda url, payload, **kw: (200, {}, {}))
    monkeypatch.setattr(v03, "DECODE_WAIT_S", 0.01)

    class Silent:
        def __init__(self, *a, **kw):
            self.first_delta, self.stopped = threading.Event(), False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(lib, "ChatStream", Silent)
    out = v03.b0_b4(_b4_box({"n_prompt_tokens": 300}), tmp_path)
    assert out["b0"]["pass"] is True and out["b4"]["pass"] is None
    assert "B4 setup" in out["b4"]["error"] and "never produced a token" in out["b4"]["error"]


def test_a_connection_error_in_b4_setup_is_a_b4_error_and_the_model_keeps_its_results(tmp_path, monkeypatch):
    """A B4 setup failure is B4 `error`, not FAIL, and loses none of the model's other results."""
    def refused(url, payload, **kw):
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    _b0_ok(monkeypatch, tmp_path)
    monkeypatch.setattr(lib, "post", refused)
    monkeypatch.setattr(lib, "running_gpu_servers", lambda: [])
    box = _b4_box({})
    box.pid, box.build_info, box.finish = None, "fake", lambda: None
    box.snapshot["kv"] = "unified"
    reads = []
    monkeypatch.setattr(v03, "_launch", lambda name, out, **kw: box)
    monkeypatch.setattr(v03, "MODELS", {"tiel_kvu": ("m.gguf", [], 8192)})
    monkeypatch.setattr(v03, "read_heldout", lambda c, d, run, data: reads.append(run) or {"rc": 0, "readouts": 0,
                                                                                          "attempts": 1})
    ev = tmp_path / "eval.jsonl"
    ev.write_text(set_text(ns=(2, 3), per=4))
    monkeypatch.setattr(v03, "EVAL_DATA", str(ev))
    profiles = []
    monkeypatch.setattr(v03, "b3_call_sequence", lambda up, gguf, prof, bodies: profiles.append(prof) or {
        "calls": len(bodies), "identical": True, "applied": len(bodies), "pass_calls": True})
    (tmp_path / "tiel_kvu").mkdir()
    (tmp_path / "tiel_kvu" / "fit.json").write_text(json.dumps({"entries": {}, "model_key": "sha256:" + "0" * 64}))
    monkeypatch.setattr(v03.ba, "analyse", lambda d, sets, r: {"b1": {"pass": True}})
    res = v03.run_model("tiel_kvu", tmp_path)
    assert res["b0"]["pass"] is True and res["b4"]["pass"] is None and "filling slot 3" in res["b4"]["error"]
    assert "Connection refused" in res["b4"]["error"] and reads == ["eval"]
    assert res["b1"] == {"pass": True} and res["b3"]["pass"] is True and res["b3"]["synthetic"] is True
    synthetic = load_profile(profiles[0])                          # B3 runs although the fit kept no n
    assert sorted(synthetic.priors) == [2, 3] and profiles[0].name == "b3_synthetic.json"
    with pytest.raises(v03.B4Setup, match="erasing slot 0"):
        v03._erase_all(box)


def test_the_watchdog_retries_a_failed_ps_and_fails_closed_after_three_in_a_row(tmp_path, monkeypatch):
    """A failed ps (lib.running_gpu_servers raises: it fails closed) is logged and retried, never the end of the
    watchdog thread; a success resets the count; three failures in a row interrupt the run like a foreign server."""
    calls: list[int] = []

    def servers():
        calls.append(1)
        if len(calls) in (1, 2, 4, 5, 6):                 # fail, fail, ok, then three in a row
            raise RuntimeError("could not list processes (ps failed: exit 1: boom)")
        return []

    monkeypatch.setattr(lib, "running_gpu_servers", servers)
    tripped = threading.Event()
    w = v03.Watchdog([str(tmp_path)], interrupt=tripped.set, own=set, interval=0.01, min_free_gib=0)
    w.start()
    assert tripped.wait(5)
    w.join(5)
    assert not w.is_alive() and len(calls) == 6
    assert w.reason == ("cannot tell whether a model server is running: ps failed 3 times in a row (could not list "
                        "processes (ps failed: exit 1: boom))")
