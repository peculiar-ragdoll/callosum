"""`callo serve --think-converge Z`: the phase-1 refusals (a bad value, no profile, a profile without a usable
convergence block), the phase-2 refusals (native, no thinking mode, a multi-token tag), the warning when no rule reaches
Z, the startup line with its reach, the cap warning, and the published block."""
import logging
import math

import pytest
from fake_upstream import FakeUpstream
from helpers import get_json, tiny_gguf
from test_converge_serve import block
from test_think import profile_dict
from test_think_cli import _profile, _serve

from callo.cli import converge_startup_lines
from callo.converge import serve_rules
from callo.deadline import Estimators
from callo.think import load_think_profile, think_default_refusal


@pytest.fixture
def up():
    u = FakeUpstream(think="preopened", think_len=5).start()
    yield u
    u.stop()


@pytest.mark.parametrize("value", ["-0.1", "1", "1.5", "abc", "nan", "inf"])
def test_a_bad_value_is_a_phase_1_refusal(tmp_path, up, caplog, value):
    r = _serve(tmp_path, up, "--think-converge", value)
    assert r.finish() == 1 and r.port is None and not up.requests
    assert "--think-converge must be a number from 0 (off) to below 1" in caplog.text


def test_early_exit_needs_a_profile_with_a_usable_convergence_block(tmp_path, up, caplog):
    gguf = tiny_gguf(tmp_path / "m.gguf")
    r = _serve(tmp_path, up, "--think-converge", "0.95", gguf=gguf)
    assert r.finish() == 1 and "--think-converge 0.95 needs --think-profile" in caplog.text and not up.requests
    for over, why in (({}, "has no convergence block"), ({"converge": {**block(), "K": 48}}, "K must be one of")):
        caplog.clear()
        p, _ = _profile(tmp_path, up, gguf, **over)
        r = _serve(tmp_path, up, "--think-profile", str(p), "--think-converge", "0.95", gguf=gguf)
        assert r.finish() == 1 and why in caplog.text and "--converge-probe" in caplog.text and not up.requests


def test_phase_2_refusals_native_no_thinking_mode_and_a_multi_token_tag(tmp_path, up, caplog):
    native = tiny_gguf(tmp_path / "kev.gguf", decision="kev")
    p, _ = _profile(tmp_path, up, native, converge=block())
    r = _serve(tmp_path, up, "--think-profile", str(p), "--think-converge", "0.95", gguf=native)
    assert r.finish() == 1 and "native decision model" in caplog.text
    plain = FakeUpstream().start()
    try:
        gguf = tiny_gguf(tmp_path / "m.gguf")
        p, _ = _profile(tmp_path, plain, gguf, converge=block())
        caplog.clear()
        r = _serve(tmp_path, plain, "--think-profile", str(p), "--think-converge", "0.95", gguf=gguf)
        assert r.finish() == 1 and "no thinking mode callo can read" in caplog.text
    finally:
        plain.stop()
    up.think_multi_tag = True
    gguf = tiny_gguf(tmp_path / "m2.gguf")
    p, _ = _profile(tmp_path, up, gguf, converge=block())
    caplog.clear()
    r = _serve(tmp_path, up, "--think-profile", str(p), "--think-converge", "0.95", gguf=gguf)
    assert r.finish() == 1 and "early exit needs a single-token close tag; this model's is 2 tokens" in caplog.text


def test_serve_names_the_reach_publishes_the_block_and_warns_when_no_rule_reaches_z(tmp_path, up, caplog):
    caplog.set_level(logging.INFO, logger="callo")
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf, converge=block())
    r = _serve(tmp_path, up, "--think-profile", str(p), "--think-converge", "0.95", gguf=gguf)
    try:
        assert ("think converge 0.95 (all clients; think_budget 0 opts out): rule n 2, y 0.6 (target 0.95), K 32, "
                "T 1.50, probe 5 ms (profile); cap 1024") in caplog.text
        _, _, st = get_json(f"{r.url}/callo/stats")
        cv = st["think"]["converge"]
        assert cv["default_z"] == 0.95 and cv["K"] == 32 and cv["rules"]["0.99"] is None
        assert "--think-converge 0.95" in think_default_refusal(st)
    finally:
        assert r.finish() == 0
    caplog.clear()
    r = _serve(tmp_path, up, "--think-profile", str(p), "--think-converge", "0.99", gguf=gguf)
    try:
        assert "--think-converge 0.99: the think profile has no early-exit rule that reaches it" in caplog.text
        assert '"converge": "unreachable"' in caplog.text
    finally:
        assert r.finish() == 0


def test_the_startup_lines_warn_when_a_cap_is_below_the_calibrated_budget():
    tp = load_think_profile(profile_dict(converge=block()))           # budgets 0, 256, 1024
    rules, _ = serve_rules(block())
    est = Estimators.from_profile(tp)
    lines = converge_startup_lines(0.9, rules, tp, 512, 0, est)
    assert lines[0].endswith("cap 512") and lines[1].startswith("WARNING: --think-converge: the rule was fitted on "
                                                                 "1024-token traces but --think-max 512 caps them at 512")
    lines = converge_startup_lines(0.9, rules, tp, 4096, 256, est)     # --think-budget 256: the cap of a fieldless request
    assert lines[0].endswith("cap 256") and lines[1].startswith("WARNING: --think-converge: the rule was fitted on "
                                                                 "1024-token traces but --think-budget 256 caps them at "
                                                                 "256")
    assert len(converge_startup_lines(0.9, rules, tp, 4096, 0, est)) == 1


def test_an_empty_fit_starts_with_the_no_rule_warning(tmp_path, up, caplog):
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf, converge={**block(), "rules": {}})
    r = _serve(tmp_path, up, "--think-profile", str(p), "--think-converge", "0.9", gguf=gguf)
    try:
        assert "no early-exit rule that reaches it (targets with a rule: none)" in caplog.text
    finally:
        assert r.finish() == 0


def test_at_z_0_a_valid_block_is_published_and_a_malformed_one_starts_with_a_note_and_nulls(tmp_path, up, caplog):
    caplog.set_level(logging.INFO, logger="callo")
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf, converge=block())
    r = _serve(tmp_path, up, "--think-profile", str(p), gguf=gguf)
    try:
        _, _, st = get_json(f"{r.url}/callo/stats")
        cv = st["think"]["converge"]
        assert cv["default_z"] == 0.0 and cv["K"] == 32 and cv["rules"]["0.99"] is None
        assert cv["rules"]["0.95"] == {"n": 2, "y": 0.6, "T": 1.5}
        assert "convergence block cannot serve" not in caplog.text and "think converge" not in caplog.text
    finally:
        assert r.finish() == 0
    caplog.clear()
    p, _ = _profile(tmp_path, up, gguf, name="bad", converge={**block(), "K": 48})
    r = _serve(tmp_path, up, "--think-profile", str(p), gguf=gguf)
    try:
        assert r.port
        _, _, st = get_json(f"{r.url}/callo/stats")
        cv = st["think"]["converge"]
        assert cv["default_z"] == 0.0 and cv["K"] is None and cv["rules"] is None
        assert ("NOTE: the think profile's convergence block cannot serve (its convergence block's K must be one of "
                "32, 64, 128, got 48): requests that ask for think_converge will report unreachable") in caplog.text
    finally:
        assert r.finish() == 0


def test_a_profile_without_a_convergence_block_starts_at_z_0_with_no_note(tmp_path, up, caplog):
    caplog.set_level(logging.INFO, logger="callo")
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf)
    r = _serve(tmp_path, up, "--think-profile", str(p), gguf=gguf)
    try:
        _, _, st = get_json(f"{r.url}/callo/stats")
        assert st["think"]["converge"]["K"] is None and "convergence block cannot serve" not in caplog.text
    finally:
        assert r.finish() == 0


def test_minus_zero_is_off_and_published_as_plus_zero(tmp_path, up):
    r = _serve(tmp_path, up, "--think-converge", "-0")
    try:
        _, _, st = get_json(f"{r.url}/callo/stats")
        z = st["think"]["converge"]["default_z"]
        assert z == 0.0 and math.copysign(1.0, z) == 1.0
    finally:
        assert r.finish() == 0


def test_the_service_serves_the_rules_cli_parsed_once(tmp_path, up, monkeypatch):
    import callo.cli as cli
    import callo.proxy as proxy

    parsed, services = [], []
    real_parse, real_service = cli.serve_rules, cli.SystemOneService

    def parse(b):
        out = real_parse(b)
        parsed.append(out[0])
        return out

    def service(*a, **kw):
        services.append(real_service(*a, **kw))
        return services[-1]

    def no_second_parse(b):
        raise AssertionError("the service parsed the convergence block again")

    monkeypatch.setattr(cli, "serve_rules", parse)
    monkeypatch.setattr(cli, "SystemOneService", service)
    monkeypatch.setattr(proxy, "serve_rules", no_second_parse)
    gguf = tiny_gguf(tmp_path / "m.gguf")
    p, _ = _profile(tmp_path, up, gguf, converge=block())
    r = _serve(tmp_path, up, "--think-profile", str(p), "--think-converge", "0.95", gguf=gguf)
    try:
        assert r.port and len(parsed) == 1 and parsed[0] is not None
        assert len(services) == 1 and services[0].converge_rules is parsed[0]
    finally:
        assert r.finish() == 0
