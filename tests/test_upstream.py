"""Attach-mode discovery against the fake llama-server: /props, the idle-clear probe, pinning decision, /slots cache."""
import io
import time
import urllib.error
from pathlib import Path

import pytest
from fake_upstream import FakeUpstream

from callo.errors import CalloError
from callo.scorer import make_http_post
from callo.upstream import (
    PROBE_TEXT,
    ProbeResult,
    SlotsCache,
    UpstreamInfo,
    decide_agent_pinning,
    discover,
    erase_slots,
    make_http_get,
    missing_props,
    probe_idle_clear,
    resolve_gguf,
    version_warning,
    wait_health,
)


@pytest.fixture
def up():
    u = FakeUpstream().start()
    yield u
    u.stop()


def test_discover_reads_the_verified_props_fields(up):
    info = discover(up.url, http_get=make_http_get())
    assert info.total_slots == 4 and info.n_ctx == 32768 and info.model_path == "/nonexistent/fake.gguf"
    assert info.build_info == "b1-926862e" and info.endpoint_slots is True and info.chat_template
    assert missing_props(info) == []


def test_discover_logs_missing_optional_values(up):
    del up.props["model_path"]
    del up.props["build_info"]
    info = discover(up.url, http_get=make_http_get())
    assert info.model_path is None and info.build_info is None
    assert missing_props(info) == ["model_path", "build_info"]


def test_discover_refuses_router_mode_and_missing_total_slots(up):
    up.props["role"] = "router"
    with pytest.raises(CalloError, match="router"):
        discover(up.url, http_get=make_http_get())
    del up.props["role"]
    del up.props["total_slots"]
    with pytest.raises(CalloError, match="total_slots"):
        discover(up.url, http_get=make_http_get())


def test_discover_sends_the_api_key_and_reports_401_clearly():
    up = FakeUpstream(api_key="k1").start()
    try:
        with pytest.raises(CalloError, match="401"):
            discover(up.url, http_get=make_http_get())
        info = discover(up.url, http_get=make_http_get(api_key="k1"))
        assert info.total_slots == 4 and up.last("/props").headers["authorization"] == "Bearer k1"
    finally:
        up.stop()


def test_version_warning_only_for_parseable_bNNNN_below_9890():
    assert version_warning("b9000-abcdef1") is not None and "9890" in version_warning("b9000-abcdef1")
    assert version_warning("b9890-74976e1ae") is None
    assert version_warning("0.5.0-dev (build 1, commit 926862e)") is None
    assert version_warning("b1-926862e") is None  # shallow clone: a 1-digit build count is not comparable
    assert "9890" in version_warning("b9000-abc")
    assert version_warning(None) is None and version_warning("garbage") is None


def test_probe_detects_idle_slot_clearing():
    up = FakeUpstream(clears_idle=True).start()
    try:
        r = probe_idle_clear(up.url, 4, http_post=make_http_post(timeout=5), http_get=make_http_get())
        assert r.clears_idle is True and r.n_prompt_tokens_last == 0
        comps = [q for q in up.requests if q.path == "/completion"]
        assert [c.json["id_slot"] for c in comps] == [3, 0]
        assert all(len(c.json["prompt"]) == 16 and c.json["n_predict"] == 1 for c in comps)
        assert len(PROBE_TEXT.split()) >= 16
    finally:
        up.stop()


def test_probe_reports_no_clearing_when_the_slot_keeps_its_tokens(up):
    r = probe_idle_clear(up.url, 4, http_post=make_http_post(timeout=5), http_get=make_http_get())
    assert r.clears_idle is False and r.n_prompt_tokens_last == 17


def test_probe_is_inconclusive_without_the_slots_endpoint():
    up = FakeUpstream(endpoint_slots=False).start()
    try:
        r = probe_idle_clear(up.url, 4, http_post=make_http_post(timeout=5), http_get=make_http_get())
        assert r.clears_idle is None and "slots" in r.detail
    finally:
        up.stop()


def test_probe_fails_closed_when_a_request_raises():
    def boom(url, payload):
        raise urllib.error.URLError("refused")

    def unauthorized(url, payload):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, io.BytesIO(b"{}"))

    for post in (boom, unauthorized):
        r = probe_idle_clear("http://x", 4, http_post=post, http_get=make_http_get())
        assert r.clears_idle is None and r.n_prompt_tokens_last is None
    assert "401" in r.detail
    off, notes = decide_agent_pinning(_info(), r)
    assert not off and any("inconclusive" in n for n in notes)


def test_probe_is_inconclusive_when_the_slot_has_no_n_prompt_tokens(up):
    def post(url, payload):  # tokenizes for real but never runs a task, so /slots keeps the key absent
        return {"tokens": list(range(16))} if url.endswith("/tokenize") else {}

    r = probe_idle_clear(up.url, 4, http_post=post, http_get=make_http_get())
    assert r.clears_idle is None and "n_prompt_tokens" in r.detail


def _info(**kw) -> UpstreamInfo:
    base = dict(base_url="http://x", total_slots=4, model_path="/m.gguf", n_ctx=4096, chat_template="t",
                build_info="b", endpoint_slots=True, props={})
    return UpstreamInfo(**{**base, **kw})


def test_decide_agent_pinning():
    on, notes = decide_agent_pinning(_info(), ProbeResult(False, "ok", 17))
    assert on and notes == []
    off, notes = decide_agent_pinning(_info(), ProbeResult(True, "cleared", 0))
    assert not off and any("--no-cache-idle-slots" in n and "-kvu" in n for n in notes)
    off, notes = decide_agent_pinning(_info(endpoint_slots=False), ProbeResult(None, "no /slots", None))
    assert not off and any("endpoint_slots" in n or "--slots" in n for n in notes)
    off, notes = decide_agent_pinning(_info(total_slots=1), ProbeResult(None, "single slot", None))
    assert not off and any("total_slots" in n for n in notes)


def test_slots_cache_serves_a_snapshot_at_most_250ms_old(up):
    cache = SlotsCache(up.url, http_get=make_http_get(), max_age=0.25)
    a = cache.get()
    b = cache.get()
    assert a == b and len([r for r in up.requests if r.path == "/slots"]) == 1
    time.sleep(0.3)
    cache.get()
    assert len([r for r in up.requests if r.path == "/slots"]) == 2
    assert cache.last_error is None


def test_slots_cache_stamps_each_snapshot_with_when_it_was_fetched(up):
    cache = SlotsCache(up.url, http_get=make_http_get(), max_age=0.25)
    t0 = time.monotonic()
    snap, at = cache.stamped()
    assert snap == up.slots and t0 <= at <= time.monotonic()
    assert cache.stamped() == (snap, at)                  # a cached snapshot keeps its original fetch time
    time.sleep(0.3)
    assert cache.stamped()[1] > at


def test_slots_cache_returns_none_on_failure_and_remembers_why():
    cache = SlotsCache("http://127.0.0.1:1", http_get=make_http_get(timeout=1), max_age=0.25)
    assert cache.get() is None and cache.last_error
    assert cache.stamped()[0] is None


def test_wait_health_gives_up_promptly_when_the_process_is_dead():
    t0 = time.monotonic()
    with pytest.raises(CalloError, match="exited"):
        wait_health("http://127.0.0.1:1", timeout=60, interval=0.05, http_get=make_http_get(timeout=0.2),
                    alive=lambda: False)
    assert time.monotonic() - t0 < 5


def test_wait_health_returns_when_ok_and_times_out_otherwise(up):
    wait_health(up.url, timeout=2, http_get=make_http_get())
    with pytest.raises(CalloError, match="health"):
        wait_health("http://127.0.0.1:1", timeout=0.5, interval=0.1, http_get=make_http_get(timeout=0.2))


def test_resolve_gguf_prefers_override_then_model_path_and_names_the_flag(tmp_path):
    g = tmp_path / "m.gguf"
    g.write_bytes(b"GGUF")
    assert resolve_gguf(_info(model_path="/nope.gguf"), str(g)) == g
    assert resolve_gguf(_info(model_path=str(g)), None) == g
    with pytest.raises(CalloError, match="--gguf"):
        resolve_gguf(_info(model_path="/nonexistent/x.gguf"), None)
    with pytest.raises(CalloError, match="--gguf"):
        resolve_gguf(_info(model_path=None), None)
    assert isinstance(resolve_gguf(_info(model_path=str(g)), None), Path)


# ---- v0.3: the probe erases the two slots it filled -------------------------------------------------------


def test_the_probe_erases_both_slots_it_used_when_the_upstream_allows_it():
    up = FakeUpstream(slot_save=True).start()
    try:
        r = probe_idle_clear(up.url, 4, http_post=make_http_post(timeout=5), http_get=make_http_get())
        assert r.clears_idle is False and r.n_prompt_tokens_last == 17      # measured BEFORE the erase
        assert r.erase == "ok" and r.residue == () and up.erased == [3, 0]
        assert all(s.get("n_prompt_tokens", 0) == 0 for s in up.slots)
        assert r.as_dict()["residue"] == [] and r.as_dict()["erase"] == "ok"
        posts = [q.path for q in up.requests if q.method == "POST" and q.path.startswith("/slots/")]
        assert posts == ["/slots/3?action=erase", "/slots/0?action=erase"]
    finally:
        up.stop()


def test_an_upstream_without_slot_save_path_leaves_the_residue_recorded(up):
    r = probe_idle_clear(up.url, 4, http_post=make_http_post(timeout=5), http_get=make_http_get())
    assert r.clears_idle is False and r.erase == "unsupported" and r.residue == (3, 0)
    assert up.slots[3]["n_prompt_tokens"] == 17 and up.erased == []


def test_erase_slots_reports_a_partial_failure():
    def post(url, payload):
        if url.endswith("/slots/0?action=erase"):
            raise urllib.error.HTTPError(url, 500, "boom", {}, io.BytesIO(b"{}"))
        return {"id_slot": 3, "n_erased": 17}

    assert erase_slots("http://x", (3, 0), http_post=post) == ("failed", (0,))

    def down(url, payload):
        raise urllib.error.URLError("refused")

    assert erase_slots("http://x", (3, 0), http_post=down) == ("failed", (3, 0))
    assert erase_slots("http://x", (), http_post=down) == ("ok", ())


@pytest.mark.parametrize("other", [urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"{}")),
                                   urllib.error.URLError("refused"), None])
def test_a_501_mixed_with_another_outcome_is_failed_not_unsupported(other):
    """Only a 501 for EVERY slot means "no --slot-save-path"; a 501 next to another error (or a success) is a failure,
    and the residue lists exactly the slots not erased."""
    def post(url, payload):
        if url.endswith("/slots/3?action=erase"):
            raise urllib.error.HTTPError(url, 501, "not supported", {}, io.BytesIO(b"{}"))
        if other is not None:
            raise other
        return {"id_slot": 0, "n_erased": 17}

    assert erase_slots("http://x", (3, 0), http_post=post) == ("failed", (3, 0) if other is not None else (3,))


def test_the_probe_can_be_told_not_to_erase(up):
    r = probe_idle_clear(up.url, 4, http_post=make_http_post(timeout=5), http_get=make_http_get(), erase=False)
    assert r.erase is None and r.residue == () and not [q for q in up.requests if q.path.startswith("/slots/")]


def test_the_probe_erases_only_the_slots_it_filled():
    """A probe that failed half-way must not erase an attached server's live KV in the other slot."""
    calls = []

    def post(url, payload, fail_slot=None):
        calls.append(url.rsplit("/", 1)[-1])
        if url.endswith("/tokenize"):
            return {"tokens": list(range(16))}
        if url.endswith("/completion") and payload.get("id_slot") == fail_slot:
            raise urllib.error.URLError("refused")
        return {}

    r = probe_idle_clear("http://x", 4, http_post=lambda u, b: post(u, b, fail_slot=0), http_get=make_http_get())
    assert r.clears_idle is None and [c for c in calls if "erase" in c] == ["3?action=erase"]
    assert r.erase == "ok" and r.residue == ()
    calls.clear()
    short = probe_idle_clear("http://x", 4, http_post=lambda u, b: {"tokens": [1, 2]}, http_get=make_http_get())
    assert short.erase is None and short.residue == ()                      # nothing was filled, nothing erased


def test_a_single_slot_upstream_is_not_probed_and_not_erased(up):
    r = probe_idle_clear(up.url, 1, http_post=make_http_post(timeout=5), http_get=make_http_get())
    assert r.clears_idle is None and r.erase is None and up.requests == []
