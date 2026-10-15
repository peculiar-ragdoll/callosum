"""`think_converge`: a number from 0 (off) to below 1, null or absent for the server default; anything else is a 422.
It is part of the request digest, and a native upstream refuses a positive one."""
import math
import re
from types import SimpleNamespace

import pytest
from fake_upstream import FakeUpstream
from test_think_serve import BODY, _engine

from callo.systemone import THINK_CONVERGE_ERROR, InvalidRequest, native_think_refusal, parse_request
from callo.think import think_seed

ABSENT = object()


def test_the_error_text_is_the_documented_one():
    assert THINK_CONVERGE_ERROR == "think_converge must be a number from 0 (off) to below 1"


@pytest.mark.parametrize("bad", [-0.1, 1, 1.0, 1.5, True, False, "0.9", float("nan"), float("inf"), [0.9], {}])
def test_a_bad_value_is_422(bad):
    with pytest.raises(InvalidRequest, match=re.escape(THINK_CONVERGE_ERROR)):
        parse_request({**BODY, "think_converge": bad})


def test_null_absent_zero_and_a_probability():
    assert parse_request(BODY).think_converge is None
    assert parse_request({**BODY, "think_converge": None}).think_converge is None
    assert parse_request({**BODY, "think_converge": 0}).think_converge == 0.0       # an integer 0: off
    r = parse_request({**BODY, "think_converge": 0.95})
    assert r.think_converge == 0.95 and type(r.think_converge) is float


def test_negative_zero_is_normalised_to_zero():
    v = parse_request({**BODY, "think_converge": -0.0}).think_converge
    assert v == 0.0 and math.copysign(1.0, v) == 1.0


BODY_DIGEST = "82342c1785847349a4f8d51c9d4a6f04b090e1c4cd3e16e89249c68004ce48b3"     # recorded on the base (9500113)


def test_the_field_is_part_of_the_digest_and_its_absence_changes_nothing():
    assert parse_request(BODY).digest == BODY_DIGEST
    assert parse_request({**BODY, "think_converge": 0.95}).digest != BODY_DIGEST
    assert parse_request({**BODY, "think_converge": 0}).digest != BODY_DIGEST


def test_the_seed_digest_ignores_think_converge_so_early_exit_stops_on_a_prefix_of_the_same_trace():
    assert parse_request(BODY).seed_digest == BODY_DIGEST                      # no field: seed_digest == digest
    for z in (0.95, 0, None, -0.0):
        r = parse_request({**BODY, "think_converge": z})
        assert r.seed_digest == BODY_DIGEST
    assert parse_request({**BODY, "think_budget": 64}).seed_digest != BODY_DIGEST      # every other field still counts
    assert parse_request({**BODY, "model": "other"}).seed_digest == BODY_DIGEST        # model never did


class _Stop(Exception):
    pass


@pytest.fixture
def engine():
    u = FakeUpstream(think="preopened", think_len=5).start()
    try:
        yield _engine(u)
    finally:
        u.stop()


@pytest.mark.parametrize("z", [ABSENT, None, 0, 0.95])
def test_a_budget_trace_is_seeded_by_the_seed_digest(engine, z):
    seen = []

    def capture(*a, **kw):
        seen.append(kw["seed"])
        raise _Stop

    engine.answer_deliberate = capture
    body = BODY if z is ABSENT else {**BODY, "think_converge": z}
    with pytest.raises(_Stop):
        engine.answer_parsed(parse_request(body), think_budget=64)
    assert seen == [think_seed(BODY_DIGEST, "q")]


@pytest.mark.parametrize("z", [ABSENT, None, 0, 0.95])
def test_deadline_chunks_are_seeded_by_the_seed_digest(engine, z):
    seen = []

    def capture(*a, **kw):
        seen.append(kw["seed"])
        raise _Stop

    engine.answer_chunked = capture
    dl = SimpleNamespace(cap=64, explicit_cap=False, temperature_of=None, share_end_ms=0.0, request_end_ms=0.0,
                         reserve_ms=0.0, estimators=None, clock=None, converge=None)
    body = BODY if z is ABSENT else {**BODY, "think_converge": z}
    with pytest.raises(_Stop):
        engine.answer_parsed(parse_request(body), deadline=dl)
    assert seen == [(BODY_DIGEST, "q")]


def test_a_native_upstream_refuses_a_positive_value_and_names_a_bad_one():
    assert "think_converge must be 0 or absent" in native_think_refusal({**BODY, "think_converge": 0.9})
    assert native_think_refusal({**BODY, "think_converge": 0}) is None
    assert native_think_refusal({**BODY, "think_converge": 2}) == THINK_CONVERGE_ERROR
