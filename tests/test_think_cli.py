"""`callo serve` with think budgets: the flags, the --temperature sentinel, the phase-1 and
phase-2 refusals, frame detection at startup, the think self-test, the stats and the probe route, against the
thinking fake in attach mode."""
import json
import logging
import urllib.error
import urllib.request

import pytest
from fake_upstream import FakeUpstream
from helpers import Running, get_json, post_json, tiny_gguf
from test_think import profile_dict

import callo.proxy as proxy_module
from callo.cli import main
from callo.modelkey import load_model_key

BODY = {"state": "the state", "questions": {"q": {"type": "choice", "instructions": "pick",
                                                  "criteria": {"a": "one", "b": "two", "c": "three"}}}}


def _profile(tmp_path, up, gguf, name="t", **over):
    """A think profile keyed to `gguf` (its real model_key, unless `over` says otherwise), written to tmp_path."""
    p = tmp_path / f"{name}.think.json"
    p.write_text(json.dumps(profile_dict(**{"model_key": load_model_key(gguf, tmp_path / "c"), **over})))
    return p, None


def _serve(tmp_path, up, *flags, think="preopened", gguf=None):
    gguf = gguf or tiny_gguf(tmp_path / "m.gguf")
    up.props["model_path"] = str(gguf)
    return Running(["serve", "--upstream", up.url, "--port", "0", "--cache-dir", str(tmp_path / "c"), *flags])


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=5).start()
    yield u
    u.stop()


def _probe_requests(up):
    return [r.json for r in up.requests if r.path == "/apply-template"]


def test_a_capable_upstream_is_detected_self_tested_and_serves_a_deliberate_readout(tmp_path, up, caplog):
    caplog.set_level(logging.INFO, logger="callo")
    r = _serve(tmp_path, up)
    try:
        assert r.port
        _, _, stats = get_json(f"{r.url}/callo/stats")
        think = stats["think"]
        assert think["capable"] is True and think["reason"] is None and think["close"] == "\n</think>\n\n"
        assert think["synthesized_open"] is False and think["default_budget"] == 0 and think["max"] == 4096
        assert think["profile"] is None and think["sampling"]["temperature"] == 0.6 and think["gguf_sampling"] == {}
        assert stats["self_test"]["think"] is True and stats["temperature"] == 1.0
        assert stats["temperature_source"] == "default" and stats["host"]["fingerprint"].startswith("sha256:")
        assert stats["model_bytes"] == (tmp_path / "m.gguf").stat().st_size and stats["mmproj_bytes"] is None
        assert stats["n_ctx"] == 32768 and stats["total_slots"] == 4 and stats["upstream_local"] is True
        kwargs = [q["chat_template_kwargs"]["enable_thinking"] for q in _probe_requests(up)]
        assert True in kwargs and False in kwargs                       # the frame: on and off renders
        assert "think: capable" in caplog.text and "think self-test ok" in caplog.text
        status, headers, out = post_json(f"{r.url}/v1/systemone", {**BODY, "think_budget": 64})
        assert status == 200 and json.loads(headers["X-Callo-Timings"])[0]["think_budget"] == 64
        status, headers, _ = post_json(f"{r.url}/v1/systemone", BODY)
        assert "think_budget" not in json.loads(headers["X-Callo-Timings"])[0]
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["think"]["requests"] == 1 and stats["think"]["uncalibrated"] == 1
    finally:
        assert r.finish() == 0


def test_a_plain_upstream_is_budget_0_only_and_serving_is_v02s(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="callo")
    up = FakeUpstream().start()
    try:
        r = _serve(tmp_path, up)
        try:
            _, _, stats = get_json(f"{r.url}/callo/stats")
            assert stats["think"]["capable"] is False and stats["think"]["reason"] == "not_preopened"
            assert stats["self_test"]["think"] is None
            assert "budget 0 only (not_preopened)" in caplog.text
            comps = [q.json for q in up.requests if q.path == "/completion"]
            assert all(c.get("n_predict", 1) == 1 for c in comps)       # no generation ever reached the upstream
            with pytest.raises(urllib.error.HTTPError) as e:
                post_json(f"{r.url}/v1/systemone", {**BODY, "think_budget": 64})
            assert e.value.code == 422
        finally:
            r.finish()
    finally:
        up.stop()


@pytest.mark.parametrize("flags", [["--think-budget", "-1"], ["--think-max", "-5"], ["--think-max", "40000"],
                                   ["--think-budget", "512", "--think-max", "256"], ["--temperature", "0"],
                                   ["--think-budget", "abc"], ["--think-max", "1e3"]])
def test_phase_1_flag_refusals_happen_before_anything_is_contacted(tmp_path, up, caplog, flags):
    r = _serve(tmp_path, up, *flags)
    assert r.finish() == 1 and r.port is None and not up.requests
    assert "ERROR" in caplog.text


def test_a_bad_profile_file_is_refused_in_phase_1(tmp_path, up, caplog):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    r = _serve(tmp_path, up, "--think-profile", str(p))
    assert r.finish() == 1 and r.port is None and not up.requests and "--think-profile" in caplog.text
    with pytest.raises(SystemExit) as e:                                   # an empty value is a usage error (exit 2)
        main(["serve", "--upstream", up.url, "--think-profile", ""])
    assert e.value.code == 2


def test_the_profile_supplies_t_0_unless_temperature_is_given_which_warns_on_a_mismatch(tmp_path, up, caplog):
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf)
    r = _serve(tmp_path, up, "--think-profile", str(p), gguf=gguf)
    try:
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["temperature"] == 1.62 and stats["temperature_source"] == "profile"
        assert stats["think"]["profile"]["budgets"] == [0, 256, 1024] and stats["think"]["sampling"]["temperature"] == 1.0
        _, headers, _ = post_json(f"{r.url}/v1/systemone", {**BODY, "think_budget": 256})
        e = json.loads(headers["X-Callo-Timings"])[0]
        assert (e["temperature"], e["calibrated"]) == (1.1, True)
    finally:
        assert r.finish() == 0
    assert "differs from the think profile's T_0" not in caplog.text
    caplog.clear()
    r = _serve(tmp_path, up, "--think-profile", str(p), "--temperature", "1.9", gguf=gguf)
    try:
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["temperature"] == 1.9 and stats["temperature_source"] == "flag"
        assert "--temperature 1.9 differs from the think profile's T_0 1.62" in caplog.text
    finally:
        r.finish()


def test_phase_2_refusals_native_another_model_and_a_thinking_profile_on_a_budget_0_only_model(tmp_path, up, caplog):
    native = tiny_gguf(tmp_path / "kev.gguf", decision="kev")
    r = _serve(tmp_path, up, "--think-budget", "64", gguf=native)
    assert r.finish() == 1 and "native decision model" in caplog.text
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf)
    caplog.clear()
    r = _serve(tmp_path, up, "--think-profile", str(p), gguf=native)
    assert r.finish() == 1 and "native decision model" in caplog.text
    other, _ = _profile(tmp_path, up, gguf, name="other", model_key="sha256:" + "0" * 64)
    caplog.clear()
    r = _serve(tmp_path, up, "--think-profile", str(other), gguf=gguf)
    assert r.finish() == 1 and "not on the served model" in caplog.text
    plain = FakeUpstream().start()
    try:
        caplog.clear()
        r = _serve(tmp_path, plain, "--think-profile", str(p), gguf=gguf)
        assert r.finish() == 1 and "no thinking mode callo can read" in caplog.text
        only0, _ = _profile(tmp_path, plain, gguf, name="only0", budgets={"0": {"T": 1.5}}, thinking={"capable": False})
        caplog.clear()
        r = _serve(tmp_path, plain, "--think-profile", str(only0), gguf=gguf)
        try:
            _, _, stats = get_json(f"{r.url}/callo/stats")
            assert stats["temperature"] == 1.5 and "budget 0 only: the profile supplies T_0" in caplog.text
        finally:
            assert r.finish() == 0
    finally:
        plain.stop()


def test_startup_warnings_for_an_uncalibrated_default_budget_and_the_position_bias_mismatch(tmp_path, up, caplog):
    gguf = tiny_gguf(tmp_path / "m.gguf")
    r = _serve(tmp_path, up, "--think-budget", "64", gguf=gguf)
    try:
        assert "--think-budget 64 without --think-profile: deliberate readouts are uncalibrated" in caplog.text
        assert get_json(f"{r.url}/callo/stats")[2]["think"]["default_budget"] == 64
    finally:
        r.finish()
    p, _ = _profile(tmp_path, up, gguf, name="drift", environment={"host": {"fingerprint": "x"}, "build_info": "b9",
                                                                   "server": {"position_bias_sha256": "abc"}})
    caplog.clear()
    r = _serve(tmp_path, up, "--think-profile", str(p), "--think-budget", "512", gguf=gguf)
    try:
        text = caplog.text
        assert "--think-budget 512 is not in the profile (calibrated budgets: 256, 1024)" in text
        assert "T_0 was fitted on readouts debiased by abc, and this server serves no profile" in text
        assert "its latencies were measured on another" in text and "b9;" in text      # "<chip>, <build>"
    finally:
        r.finish()


def test_an_unpinnable_server_grammar_makes_the_model_budget_0_only(tmp_path, up, caplog):
    up.props["default_generation_settings"]["params"] = {"grammar": "root ::= [a-z]+", "temperature": 0.8}
    r = _serve(tmp_path, up)
    try:
        _, _, stats = get_json(f"{r.url}/callo/stats")
        assert stats["think"]["capable"] is False and stats["think"]["reason"] == "sampler_unpinnable: grammar"
        assert "which a request cannot override" in caplog.text
    finally:
        r.finish()
    r = _serve(tmp_path, up, "--think-budget", "64")
    assert r.finish() == 1 and "sampler_unpinnable: grammar" in caplog.text


def test_the_gguf_s_own_sampling_is_the_default_trace_sampler(tmp_path, up):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "s.gguf"
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(__import__("helpers").TOKENS)
    w.add_float32("general.sampling.temp", 1.0)
    w.add_int32("general.sampling.top_k", 40)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    r = _serve(tmp_path, up, gguf=path)
    try:
        think = get_json(f"{r.url}/callo/stats")[2]["think"]
        assert think["gguf_sampling"] == {"temp": 1.0, "top_k": 40}
        assert (think["sampling"]["temperature"], think["sampling"]["top_k"], think["sampling"]["top_p"]) == (1.0, 40, 0.95)
        post_json(f"{r.url}/v1/systemone", {**BODY, "think_budget": 8})
        gen = [q.json for q in up.requests if q.path == "/completion" and q.json.get("n_predict", 1) == 8][0]
        assert gen["temperature"] == 1.0 and gen["top_k"] == 40
    finally:
        r.finish()


def test_the_probe_route_is_wired_and_loopback_only_unless_allowed(tmp_path, up, monkeypatch):
    r = _serve(tmp_path, up)
    try:
        status, _, out = post_json(f"{r.url}/callo/think-probe", {"request": BODY, "budgets": [0, 64], "seed": 1})
        assert status == 200 and out["readouts"] == 2 and set(out["questions"]["q"]["budgets"]) == {"0", "64"}
        monkeypatch.setattr(proxy_module, "_is_loopback", lambda addr: False)
        with pytest.raises(urllib.error.HTTPError) as e:
            post_json(f"{r.url}/callo/think-probe", {"request": BODY, "budgets": [0]})
        assert e.value.code == 403
    finally:
        r.finish()
    r = _serve(tmp_path, up, "--allow-think-probe")
    try:
        assert post_json(f"{r.url}/callo/think-probe", {"request": BODY, "budgets": [0]})[0] == 200
    finally:
        r.finish()
    native = tiny_gguf(tmp_path / "kev.gguf", decision="kev")
    monkeypatch.setattr(proxy_module, "_is_loopback", lambda addr: True)
    r = _serve(tmp_path, up, gguf=native)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post_json(f"{r.url}/callo/think-probe", {"request": BODY, "budgets": [0]})
        assert e.value.code == 404
        assert get_json(f"{r.url}/callo/stats")[2]["think"]["capable"] is False
    finally:
        r.finish()


# ---- the sampler, T_0, the startup refusals, the probe wiring, the host --------------------------------------------

import callo.cli as cli_module  # noqa: E402 -- the carried tests patch serve's collaborators
from callo.cli import upstream_error  # noqa: E402


def _sampling_gguf(path, **keys):
    """A tiny GGUF carrying `general.sampling.<key>` values (floats, or int32 for top_k)."""
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(__import__("helpers").TOKENS)
    for k, v in keys.items():
        (w.add_int32 if isinstance(v, int) else w.add_float32)(f"general.sampling.{k}", v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _gens(up, n):
    return [q.json for q in up.requests if q.path == "/completion" and q.json.get("n_predict", 1) == n
            and not q.json.get("n_probs")]


def test_the_engine_thinks_with_the_served_gguf_s_sampler_and_never_warns_of_the_fallback(tmp_path, up, caplog):
    """Serve sets Engine.think_sampling from the served GGUF (policy model), so the self-test and every
    deliberate generation pin the GGUF's values, and the engine's fallback warning never fires."""
    caplog.set_level(logging.INFO, logger="callo")
    gguf = _sampling_gguf(tmp_path / "s.gguf", temp=0.7, top_p=0.8, top_k=33, min_p=0.05)
    r = _serve(tmp_path, up, gguf=gguf)
    try:
        think = get_json(f"{r.url}/callo/stats")[2]["think"]
        pinned = think["sampling"]
        assert (pinned["temperature"], pinned["top_p"], pinned["top_k"], pinned["min_p"]) == (0.7, 0.8, 33, 0.05)
        (self_test,) = _gens(up, 64)                                     # the think self-test, at budget 64
        post_json(f"{r.url}/v1/systemone", {**BODY, "think_budget": 9})
        (served,) = _gens(up, 9)
        for g in (self_test, served):
            assert {k: g[k] for k in pinned} == pinned                   # the whole pinned dict, as resolved
        assert "trace sampler 0.7/0.8/33/0.05 (temperature gguf, top_p gguf, top_k gguf, min_p gguf)" in caplog.text
    finally:
        assert r.finish() == 0
    assert "no trace sampler was resolved" not in caplog.text


@pytest.mark.parametrize(("policy", "trace"), [("fixed", (0.6, 0.95, 20, 0.0)), ("temp=0.8,top_k=7", (0.8, 0.95, 7, 0.0)),
                                              ("model", (1.0, 0.95, 40, 0.0))])
def test_think_sampling_picks_the_policy_without_a_profile(tmp_path, up, policy, trace):
    gguf = _sampling_gguf(tmp_path / "s.gguf", temp=1.0, top_k=40)
    r = _serve(tmp_path, up, "--think-sampling", policy, gguf=gguf)
    try:
        think = get_json(f"{r.url}/callo/stats")[2]["think"]
        s = think["sampling"]
        assert (s["temperature"], s["top_p"], s["top_k"], s["min_p"]) == trace
        assert think["gguf_sampling"] == {"temp": 1.0, "top_k": 40}      # always the served GGUF's header
        post_json(f"{r.url}/v1/systemone", {**BODY, "think_budget": 8})
        (gen,) = _gens(up, 8)
        assert (gen["temperature"], gen["top_k"]) == (trace[0], trace[2])
    finally:
        assert r.finish() == 0


def test_think_sampling_is_refused_in_phase_1_when_invalid_or_given_with_a_profile(tmp_path, up, caplog):
    r = _serve(tmp_path, up, "--think-sampling", "temp=hot")
    assert r.finish() == 1 and r.port is None and not up.requests and "--think-sampling" in caplog.text
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf)
    caplog.clear()
    r = _serve(tmp_path, up, "--think-sampling", "fixed", "--think-profile", str(p), gguf=gguf)
    assert r.finish() == 1 and r.port is None and not up.requests
    assert "the profile's sampling (model) is part of what was calibrated; omit --think-sampling" in caplog.text


def test_engine_trace_sampling_warns_once_when_a_capable_engine_falls_back(up, caplog):
    from test_think_engine import engine_for
    e = engine_for(up)
    e.think_sampling = None
    caplog.set_level(logging.WARNING, logger="callo")
    first, second = e.trace_sampling(), e.trace_sampling()
    assert first == second and (first["temperature"], first["top_k"]) == (0.6, 20)
    assert caplog.text.count("no trace sampler was resolved") == 1
    caplog.clear()
    e.think = None                                       # not capable: the default is just the default, no warning
    e._default_sampling_warned = False
    e.trace_sampling()
    e.think_sampling = {"temperature": 0.5}
    assert e.trace_sampling() == {"temperature": 0.5}
    assert "no trace sampler was resolved" not in caplog.text


def _probs(r):
    return post_json(f"{r.url}/v1/systemone", BODY)[2]["answers"]["q"]["probabilities"]


def test_budget_0_is_read_at_the_profile_s_t_0_exactly_as_an_explicit_temperature_would(tmp_path, up):
    """Without --temperature the profile's T_0 IS the budget-0 temperature (the readout, not just the
    stats), and an explicit flag replaces it."""
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf)
    runs = {}
    for name, flags in (("profile", ["--think-profile", str(p)]), ("flag", ["--temperature", "1.62"]),
                        ("default", []), ("override", ["--think-profile", str(p), "--temperature", "1.0"])):
        r = _serve(tmp_path, up, *flags, gguf=gguf)
        try:
            runs[name] = (_probs(r), get_json(f"{r.url}/callo/stats")[2]["temperature_source"])
        finally:
            assert r.finish() == 0
    assert runs["profile"][0] == runs["flag"][0] and runs["profile"][0] != runs["default"][0]
    assert runs["override"][0] == runs["default"][0]                   # the explicit flag wins over T_0
    assert [runs[k][1] for k in ("profile", "flag", "default", "override")] == ["profile", "flag", "default", "flag"]


@pytest.mark.parametrize(("value", "warns"), [("1.65", False), ("1.585", False), ("1.66", True), ("1.58", True)])
def test_the_temperature_mismatch_warning_fires_beyond_one_grid_step(tmp_path, up, caplog, value, warns):
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf)
    r = _serve(tmp_path, up, "--think-profile", str(p), "--temperature", value, gguf=gguf)
    try:
        assert get_json(f"{r.url}/callo/stats")[2]["temperature"] == float(value)
    finally:
        assert r.finish() == 0
    assert ("differs from the think profile's T_0 1.62" in caplog.text) is warns


def test_a_think_default_on_a_model_that_cannot_think_is_refused_before_the_self_tests(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="callo")
    plain = FakeUpstream().start()
    try:
        r = _serve(tmp_path, plain, "--think-budget", "64")
        assert r.finish() == 1 and r.port is None
        assert "--think-budget 64: this model's chat template has no thinking mode callo can read (not_preopened)" \
            in caplog.text and "this model can be served and calibrated at budget 0 only" in caplog.text
        assert "self-test" not in caplog.text                          # refused before any self-test ran
    finally:
        plain.stop()


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan"])
def test_a_bad_s1_timeout_is_a_usage_error_from_main_and_a_phase_1_refusal_in_serve(tmp_path, up, caplog, value):
    """The service and the probe refuse an s1_timeout the S1 section cannot wait on; serve says so
    before anything starts, not after a model load."""
    with pytest.raises(SystemExit) as e:
        main(["serve", "--upstream", up.url, "--s1-timeout", value])
    assert e.value.code == 2
    r = _serve(tmp_path, up, "--s1-timeout", value)
    assert r.finish() == 1 and r.port is None and not up.requests
    assert "--s1-timeout must be a finite number of seconds above 0" in caplog.text


@pytest.mark.parametrize(("url", "why"), [
    ("127.0.0.1:8199", "no http:// scheme"), ("localhost:8199", "no http:// scheme"),
    ("https://127.0.0.1:8199", "https is not supported"), ("ftp://127.0.0.1:8199", "the scheme 'ftp' is not http"),
    ("http://127.0.0.1:8199/llama", "has a path"), ("http://127.0.0.1:8199/v1/", "has a path"),
    ("http://127.0.0.1:8199?x=1", "has a path, query or fragment"), ("http://:8199", "has no host"),
    ("http://127.0.0.1:99999", "is not a valid URL"), ("http://u:p@127.0.0.1:8199", "carries credentials")])
def test_an_upstream_url_callo_cannot_talk_to_is_refused_in_phase_1(tmp_path, caplog, url, why):
    caplog.set_level(logging.INFO, logger="callo")
    assert why in upstream_error(url)
    r = Running(["serve", "--upstream", url, "--port", "0", "--cache-dir", str(tmp_path / "c")])
    assert r.finish() == 1 and r.port is None
    assert why in caplog.text and "listening on" not in caplog.text     # refused before callo even listens


@pytest.mark.parametrize("url", ["http://127.0.0.1:8199", "http://127.0.0.1:8199/", "http://localhost", "http://[::1]:80"])
def test_a_plain_http_host_port_upstream_is_accepted(url):
    assert upstream_error(url) is None


def test_a_trailing_slash_upstream_still_serves(tmp_path, up):
    gguf = tiny_gguf(tmp_path / "m.gguf")
    up.props["model_path"] = str(gguf)
    r = Running(["serve", "--upstream", up.url + "/", "--port", "0", "--cache-dir", str(tmp_path / "c")])
    try:
        assert r.port and post_json(f"{r.url}/v1/systemone", BODY)[0] == 200
    finally:
        assert r.finish() == 0


def test_the_service_the_probe_and_the_proxy_get_their_arguments(tmp_path, up, monkeypatch):
    """Carry (Tasks 8, 9): s1_timeout, slots_snapshot and fallback reach both the service and ThinkProbe; the proxy
    passes cancel (systemone_cancel) and routes the probe."""
    seen: dict = {}
    real_svc, real_probe, real_set = cli_module.SystemOneService, cli_module.ThinkProbe, cli_module.set_config
    monkeypatch.setattr(cli_module, "SystemOneService", lambda *a, **k: (seen.setdefault("svc", k), real_svc(*a, **k))[1])
    monkeypatch.setattr(cli_module, "ThinkProbe", lambda *a, **k: (seen.setdefault("probe", k), real_probe(*a, **k))[1])
    monkeypatch.setattr(cli_module, "set_config", lambda srv, cfg: (seen.setdefault("cfg", cfg), real_set(srv, cfg))[1])
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf)
    r = _serve(tmp_path, up, "--s1-timeout", "7.5", "--s1-fallback", "off", "--think-budget", "256",
               "--think-max", "1024", "--think-profile", str(p), "--allow-think-probe", gguf=gguf)
    try:
        assert r.port
        svc, probe, cfg = seen["svc"], seen["probe"], seen["cfg"]
        assert (svc["s1_timeout"], svc["fallback"], svc["think_default"], svc["think_max"]) == (7.5, False, 256, 1024)
        assert svc["think_profile"].sha256 == get_json(f"{r.url}/callo/stats")[2]["think"]["profile"]["sha256"]
        assert (probe["s1_timeout"], probe["fallback"]) == (7.5, False)
        assert callable(svc["slots_snapshot"]) and callable(probe["slots_snapshot"])
        assert cfg.systemone_cancel is True and cfg.think_probe is not None and cfg.allow_think_probe is True
    finally:
        assert r.finish() == 0


def test_a_native_model_refuses_a_positive_think_budget_and_says_so_at_startup(tmp_path, up, caplog):
    caplog.set_level(logging.INFO, logger="callo")
    r = _serve(tmp_path, up, gguf=tiny_gguf(tmp_path / "kev.gguf", decision="kev"))
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post_json(f"{r.url}/v1/systemone", {**BODY, "think_budget": 64})
        assert e.value.code == 422
        stats = get_json(f"{r.url}/callo/stats")[2]
        assert stats["temperature"] is None and stats["think"]["capable"] is False and stats["self_test"]["think"] is None
    finally:
        assert r.finish() == 0
    line = next(rec.getMessage() for rec in caplog.records if "native decision model" in rec.getMessage())
    assert "callo refuses audio and video with 422, and any think_budget > 0" in line


def test_host_info_is_gathered_off_the_startup_path(tmp_path, up, monkeypatch, caplog):
    """A host query that hangs (nvidia-smi in D-state) holds startup for HOST_INFO_WAIT at most;
    the profile's host check then reads unknown, and stats.host fills in when the query returns."""
    import threading
    import time
    release = threading.Event()
    info = {"os": "x", "machine": "arm64", "chip": "c", "memory_bytes": 1, "gpu": None, "fingerprint": "sha256:" + "f" * 64}
    monkeypatch.setattr(cli_module, "host_info", lambda: (release.wait(30), info)[1])
    monkeypatch.setattr(cli_module, "HOST_INFO_WAIT", 0.2)
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf)
    try:
        r = _serve(tmp_path, up, "--think-profile", str(p), gguf=gguf)
        try:
            assert r.port
            stats = get_json(f"{r.url}/callo/stats")[2]
            assert stats["host"] is None and stats["think"]["profile"]["host_match"] is None
            assert "host info not gathered after 0.2 s" in caplog.text and "unknown host" in caplog.text
            release.set()
            deadline = time.monotonic() + 10
            while get_json(f"{r.url}/callo/stats")[2]["host"] is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert get_json(f"{r.url}/callo/stats")[2]["host"] == info       # the late result still lands
        finally:
            assert r.finish() == 0
    finally:
        release.set()


def test_a_host_query_that_raises_leaves_host_null_and_startup_unharmed(tmp_path, up, monkeypatch, caplog):
    def boom():
        raise RuntimeError("no sysctl")
    monkeypatch.setattr(cli_module, "host_info", boom)
    r = _serve(tmp_path, up)
    try:
        assert r.port and get_json(f"{r.url}/callo/stats")[2]["host"] is None
        assert "cannot gather host info (RuntimeError: no sysctl)" in caplog.text
    finally:
        assert r.finish() == 0


# ---- temperature calibrate and calibrate --bias refuse a server that thinks by default ---------------------

import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

from fake_callo import default_stats  # noqa: E402

from callo import calibrate_bias  # noqa: E402


class _ThinkingStats(BaseHTTPRequestHandler):
    """A callo whose /callo/stats says it thinks by default; it counts the readouts it is asked for."""
    posts = 0
    extra: dict = {}

    def do_GET(self):
        data = json.dumps({**default_stats(), "think": {"default_budget": 256}, **type(self).extra}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        type(self).posts += 1
        self.send_response(500)
        self.end_headers()

    def log_message(self, *a):
        pass


def _calibrate_against_thinking_stats(tmp_path, extra=None):
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"state": "s", "questions": {"q": {"type": "choice", "label": "a",
                                                                  "criteria": {k: None for k in "ab"}}}}) + "\n")
    _ThinkingStats.posts, _ThinkingStats.extra = 0, extra or {}
    srv = HTTPServer(("127.0.0.1", 0), _ThinkingStats)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        return main(["calibrate", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--data", str(data)])
    finally:
        srv.shutdown()
        srv.server_close()
        _ThinkingStats.extra = {}


def test_temperature_calibrate_refuses_a_server_that_thinks_by_default(tmp_path, capsys):
    rc = _calibrate_against_thinking_stats(tmp_path)
    err = capsys.readouterr().err
    assert rc == 2 and "thinks by default (--think-budget 256)" in err and "callo calibrate --think-budgets" in err
    assert _ThinkingStats.posts == 0


def test_temperature_calibrate_names_thinking_before_a_profile_s_t_0(tmp_path, capsys):
    # serve with --think-budget and a think profile reports the profile's T_0: the think refusal, which names
    # the real fix, comes first; the "not 1.0" refusal would send the user to --temperature, which cannot help
    rc = _calibrate_against_thinking_stats(tmp_path, {"temperature": 1.62})
    err = capsys.readouterr().err
    assert rc == 2 and "thinks by default (--think-budget 256)" in err and "not 1.0" not in err
    assert _ThinkingStats.posts == 0


def test_bias_calibrate_s_preconditions_refuse_a_server_that_thinks_by_default():
    stats = {**default_stats(), "think": {"default_budget": 1024}}
    with pytest.raises(calibrate_bias.Refusal, match="thinks by default \\(--think-budget 1024\\)"):
        calibrate_bias.check_preconditions(stats, "http://x")
    calibrate_bias.check_preconditions({**default_stats(), "think": {"default_budget": 0}}, "http://x")
    calibrate_bias.check_preconditions(default_stats(), "http://x")          # a callo without the think block


def test_bias_calibrate_s_think_refusal_comes_before_temperature_and_profile_and_after_engine_off():
    think = {"think": {"default_budget": 256}}
    for over in ({"temperature": 1.62}, {"position_bias": {"source": "p.json"}},
                 {"temperature": 1.62, "position_bias": {"source": "p.json"}}):
        with pytest.raises(calibrate_bias.Refusal, match="thinks by default \\(--think-budget 256\\)"):
            calibrate_bias.check_preconditions({**default_stats(), **think, **over}, "http://x")
    # the earlier refusals keep their place: an older callo, and callo's engine off
    with pytest.raises(calibrate_bias.Refusal, match="has no model_key"):
        calibrate_bias.check_preconditions({k: v for k, v in {**default_stats(), **think}.items()
                                            if k != "model_key"}, "http://x")
    with pytest.raises(calibrate_bias.Refusal, match="engine is off"):
        calibrate_bias.check_preconditions({**default_stats(), **think, "temperature": None}, "http://x")
    # without thinking, the earlier refusals are unchanged
    with pytest.raises(calibrate_bias.Refusal, match="S1 temperature is 1.62, not 1.0"):
        calibrate_bias.check_preconditions({**default_stats(), "temperature": 1.62}, "http://x")
    with pytest.raises(calibrate_bias.Refusal, match="already applies a position-bias profile"):
        calibrate_bias.check_preconditions({**default_stats(), "position_bias": {"source": "p.json"}}, "http://x")


# ---- upstream locality, GGUF sampling in the stats, T_0 from a think profile, malformed think blocks -----------

from callo.errors import InvalidInput  # noqa: E402
from callo.think import GGUFSampling  # noqa: E402


@pytest.mark.parametrize(("url", "local"), [
    ("http://localhost:8080", True), ("http://LOCALHOST:8080", True), ("http://127.0.0.1:8080", True),
    ("http://127.3.2.1:8080", True), ("http://[::1]:8080", True), ("http://10.0.0.5:8080", False),
    ("http://gpu-box:8080", False), ("http://localhost.example.com:8080", False)])
def test_upstream_is_local_takes_loopback_names_and_addresses(url, local):
    assert cli_module.upstream_is_local(url) is local


def test_an_upstream_named_localhost_publishes_upstream_local_true(tmp_path, up):
    gguf = tiny_gguf(tmp_path / "m.gguf")
    up.props["model_path"] = str(gguf)
    r = Running(["serve", "--upstream", up.url.replace("127.0.0.1", "localhost"), "--port", "0", "--cache-dir",
                 str(tmp_path / "c")])
    try:
        assert r.port and get_json(f"{r.url}/callo/stats")[2]["upstream_local"] is True
    finally:
        assert r.finish() == 0


def test_a_native_upstream_publishes_upstream_local_too(tmp_path, up):
    r = _serve(tmp_path, up, gguf=tiny_gguf(tmp_path / "kev.gguf", decision="kev"))
    try:
        stats = get_json(f"{r.url}/callo/stats")[2]
        assert stats["upstream_local"] is True and stats["think"]["gguf_error"] is None
    finally:
        assert r.finish() == 0


@pytest.mark.parametrize(("reader", "error"), [
    (lambda path: GGUFSampling(error="the GGUF metadata could not be read (gguf-py: ValueError: bad type)"),
     "the GGUF metadata could not be read (gguf-py: ValueError: bad type)"),
    (lambda path: (_ for _ in ()).throw(InvalidInput("not a GGUF")),
     "the GGUF metadata could not be read (InvalidInput: not a GGUF)"),
    (lambda path: GGUFSampling({"temp": 0.7}), None)])
def test_stats_think_publishes_why_the_gguf_s_sampling_is_missing(tmp_path, up, monkeypatch, reader, error):
    """An unparseable GGUF must not look like one without general.sampling.* keys."""
    monkeypatch.setattr(cli_module, "read_gguf_sampling", reader)
    r = _serve(tmp_path, up)
    try:
        think = get_json(f"{r.url}/callo/stats")[2]["think"]
        assert think["gguf_error"] == error and think["gguf_sampling"] == ({} if error else {"temp": 0.7})
    finally:
        assert r.finish() == 0


def test_a_gguf_holding_nan_keeps_callo_stats_strict_json(tmp_path, up, monkeypatch):
    monkeypatch.setattr(cli_module, "read_gguf_sampling", lambda path: GGUFSampling({"temp": float("nan")}))
    r = _serve(tmp_path, up)
    try:
        with urllib.request.urlopen(f"{r.url}/callo/stats", timeout=10) as resp:
            raw = resp.read().decode()
        assert "NaN" not in raw and json.loads(raw)["think"]["gguf_sampling"] == {"temp": "nan"}
    finally:
        assert r.finish() == 0


def test_a_phase_1_refusal_starts_no_host_query(tmp_path, up, monkeypatch):
    started = []
    monkeypatch.setattr(cli_module, "_HostInfo", lambda: started.append(1))
    r = _serve(tmp_path, up, "--position-bias", str(tmp_path / "missing.json"))
    assert r.finish() == 1 and not started and not up.requests


def test_think_sampling_on_a_model_that_cannot_use_it_warns_that_it_is_ignored(tmp_path, up, caplog):
    plain = FakeUpstream().start()
    try:
        r = _serve(tmp_path, plain, "--think-sampling", "fixed")
        try:
            assert r.port and "--think-sampling fixed is ignored: this model is served at budget 0 only" in caplog.text
        finally:
            assert r.finish() == 0
    finally:
        plain.stop()
    caplog.clear()
    r = _serve(tmp_path, up, "--think-sampling", "fixed", gguf=tiny_gguf(tmp_path / "kev.gguf", decision="kev"))
    try:
        assert r.port and "--think-sampling fixed is ignored: a native decision model" in caplog.text
    finally:
        assert r.finish() == 0


def test_a_t_0_from_a_think_profile_is_refused_with_the_profile_named(tmp_path, capsys):
    """Serve --think-profile at budget 0 reports the profile's T_0; the fix is the profile, not
    --temperature (which was never given)."""
    rc = _calibrate_against_thinking_stats(tmp_path, {"think": {"default_budget": 0}, "temperature": 1.62,
                                                      "temperature_source": "profile"})
    err = capsys.readouterr().err
    assert rc == 2 and "S1 temperature is 1.62, not 1.0" in err and "without --think-profile" in err
    stats = {**default_stats(), "think": {"default_budget": 0}, "temperature": 1.62, "temperature_source": "profile"}
    with pytest.raises(calibrate_bias.Refusal, match="without --think-profile"):
        calibrate_bias.check_preconditions(stats, "http://x")
    rc = _calibrate_against_thinking_stats(tmp_path, {"think": {"default_budget": 0}, "temperature": 1.62,
                                                      "temperature_source": "flag"})
    assert rc == 2 and "without --temperature" in capsys.readouterr().err        # the earlier wording otherwise


@pytest.mark.parametrize("think", ["x", {"default_budget": None}, {"default_budget": "256"}])
def test_a_malformed_think_block_is_a_refusal_not_a_traceback(tmp_path, capsys, think):
    rc = _calibrate_against_thinking_stats(tmp_path, {"think": think})
    assert rc == 2 and "cannot tell whether the server thinks by default" in capsys.readouterr().err
    assert _ThinkingStats.posts == 0
    with pytest.raises(calibrate_bias.Refusal, match="cannot tell whether the server thinks by default"):
        calibrate_bias.check_preconditions({**default_stats(), "think": think}, "http://x")
