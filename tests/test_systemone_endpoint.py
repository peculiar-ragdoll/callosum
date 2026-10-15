# Adapted from the author's benchmark harness (tests/test_system1_server.py @ f886c50f).
# Edits: only the handle() error-contract cases are kept (DiagLog, HTTP and with_id_slot cases dropped: /callo/stats and
#   scorer tests cover them); the two unexpected-exception tests are merged; the slot tests are new; v0.2: FakeEngine
#   subclasses Engine (the service calls Engine.answer_parsed) and tests for images, lanes, the error contract,
#   X-Callo-Timings and the v0.2 stats are added; v0.3: `miss` in X-Callo-Timings and the v0.3 stats fields, and the
#   422 HTTPError is closed (no ResourceWarning); v0.3 media: a state part with a non-string type is answered;
#   /apply-template sees the placeholder image part; the 422/501/413 status mapping for every media kind (the v0.2
#   context-size row now expects 413); `passes` in X-Callo-Timings (the three exact-entry assertions gain it), `_live`
#   takes audio= / video=, and the audio, video and mixed requests and stats through the proxy.
import base64
import dataclasses
import http.client
import json
import string
import threading
import time
import urllib.error
import urllib.request

import pytest
from fake_upstream import MEDIA_MARKER, FakeUpstream

from callo.av import RED_AVI, RED_URL, TONE_URL, TONE_WAV
from callo.engine import Engine, QuestionReadout, Readout, SurfaceSets
from callo.errors import (
    ImageRejected,
    InvalidInput,
    MediaLoadFailed,
    MediaTooLarge,
    NoAudio,
    NoVideo,
    NoVision,
    ReadoutError,
    UpstreamRejected,
    UpstreamUnavailable,
)
from callo.prompt import PLACEHOLDER_IMAGE_URL, TemplateClient
from callo.proxy import TIMINGS_MAX_BYTES, ProxyConfig, Stats, SystemOneService, make_server, timings_header
from callo.scorer import LlamaCppForwardScorer, make_http_post
from callo.slots import SlotRouter
from callo.upstream import SlotsCache

BODY = {"state": "s", "model": "kev-latest", "questions": {
    "q": {"type": "choice", "criteria": {"a": "one", "b": "two"}}, "o": {"type": "noul"}}}
# The fake upstream's vocabulary: 0..25 = A..Z, 26..51 = " A".." Z", 52 = "**", 53 = " the".
TOKENS = list(string.ascii_uppercase) + ["Ġ" + c for c in string.ascii_uppercase] + ["**", "Ġthe", "x"]


class FakeEngine(Engine):
    """The real Engine.answer_parsed around a canned answer(): no scorer, template or vocabulary is ever reached."""

    def __init__(self, exc=None):
        self.exc, self.calls = exc, []
        self.vision, self.media_marker, self.model_name = False, None, "fake"

    def answer(self, state, q, *, id_slot=None, **kw):
        self.calls.append((q.qid, id_slot))
        if self.exc:
            raise self.exc
        return Readout((0.25, 0.75), (-1.4, -0.3), 0.9, "diverged", 0, 0, 12)


def _now(snapshot):
    """A stamped /slots snapshot fn: `snapshot`, fetched just now."""
    return lambda: (snapshot, time.monotonic())


def _svc(engine, router=None, **kw):
    return SystemOneService(engine, router or SlotRouter(4, 1), Stats(), model_name="qwen35-9b-s1", **kw)


def test_ok_response_shape_matches_kev_remote_predictor():
    code, resp, slot = _svc(FakeEngine())(BODY)
    assert code == 200 and resp["model"] == "qwen35-9b-s1" and slot == "0"
    assert resp["answers"]["q"]["probabilities"] == {"a": 0.25, "b": 0.75}
    assert resp["answers"]["o"] == {"type": "noul", "noul": 0.75}
    assert resp["usage"] == {"input_tokens": 24, "output_tokens": 0}


def test_invalid_request_is_422():
    code, resp, _ = _svc(FakeEngine())({"state": "x", "questions": {}})
    assert code == 422 and "detail" in resp
    code, resp, _ = _svc(FakeEngine())(["not", "an", "object"])
    assert code == 422


def test_backend_down_is_503_not_4xx():
    code, _, slot = _svc(FakeEngine(urllib.error.URLError("refused")))(BODY)
    assert code == 503 and slot == "0"        # the reserved slot that was used is reported on error replies too
    code, _, _ = _svc(FakeEngine(ConnectionResetError()))(BODY)
    assert code == 503
    code, _, _ = _svc(FakeEngine(urllib.error.HTTPError("u", 500, "boom", {}, None)))(BODY)
    assert code == 503            # an upstream HTTP 500 is retryable, never a dropped record


def test_a_dropped_or_truncated_upstream_maps_to_503_not_500():
    """An http.client.HTTPException (e.g. IncompleteRead: the upstream dies mid-body) and a truncated/invalid
    upstream JSON response (json.JSONDecodeError, a ValueError) are transient upstream failures -> 503, not 500."""
    code, out, _ = _svc(FakeEngine(http.client.IncompleteRead(b"partial")))(BODY)
    assert code == 503 and out["detail"].startswith("llama-server unavailable:")
    code, out, _ = _svc(FakeEngine(json.JSONDecodeError("Expecting value", "", 0)))(BODY)
    assert code == 503 and out["detail"].startswith("llama-server unavailable:")


def test_readout_failure_is_500_with_detail():
    code, resp, _ = _svc(FakeEngine(InvalidInput("option(s) ['Z'] have NO surface form")))(BODY)
    assert code == 500 and "Z" in resp["detail"]


def test_unexpected_exception_is_500_never_422_and_prints_traceback(capsys):
    for exc in (KeyError("boom"), ValueError("zip() argument 2 is shorter"), RuntimeError("deep bug")):
        code, resp, _ = _svc(FakeEngine(exc))(BODY)
        assert code == 500 and "detail" in resp
    assert "Traceback" in capsys.readouterr().err


def test_every_question_of_a_request_stays_on_one_reserved_slot_and_requests_spread_with_k_gt_1():
    eng = FakeEngine()
    svc = _svc(eng, SlotRouter(5, 2))
    _, _, s1 = svc(BODY)
    _, _, s2 = svc(BODY)
    _, _, s3 = svc(BODY)
    assert [s1, s2, s3] == ["0", "1", "0"]
    assert eng.calls == [("q", 0), ("o", 0), ("q", 1), ("o", 1), ("q", 0), ("o", 0)]


def test_slot_accounting_is_released_after_the_request_even_on_error():
    router = SlotRouter(4, 1)
    svc = _svc(FakeEngine(RuntimeError("x")), router)
    svc(BODY)
    assert router.inflight(0) == 0


def test_s1_fallback_when_the_reserved_slot_holds_a_foreign_task():
    router = SlotRouter(4, 1)
    stats = Stats()
    snapshot = [{"id": 0, "is_processing": True, "n_prompt_tokens": 9000}, {"id": 1, "is_processing": True},
                {"id": 2, "is_processing": False}, {"id": 3, "is_processing": False}]
    eng = FakeEngine()
    svc = SystemOneService(eng, router, stats, model_name="m", slots_snapshot=_now(snapshot))
    code, _, slot = svc(BODY)
    assert code == 200 and slot == "2" and eng.calls[0][1] == 2
    assert router.counters["s1_fallback"] == 1 and stats.counters["foreign_in_reserved"] == 1


def test_s1_fallback_ignores_our_own_in_flight_tasks_on_the_reserved_slot():
    router = SlotRouter(4, 1)
    router.acquire_s1(0)                       # a classification callo itself sent is not "foreign"
    snapshot = [{"id": 0, "is_processing": True}, {"id": 1, "is_processing": False}]
    svc = SystemOneService(FakeEngine(), router, Stats(), model_name="m", slots_snapshot=_now(snapshot))
    _, _, slot = svc(BODY)
    assert slot == "0" and router.counters["s1_fallback"] == 0


def _agent_slot_with_kv(router):
    d = router.route_agent("an agent conversation " * 50)
    router.release(d.slot)
    return d.slot


def test_a_stats_poll_during_our_own_classification_is_not_read_as_foreign():
    """A /slots fetch while S1 #1 runs on slot 0 (a /callo/stats poll) caches slot 0 as busy; S1 #2
    arriving within the snapshot's 250 ms must not take that for a foreign task, fall back onto an agent slot and
    evict the agent's KV. Real SlotRouter + SlotsCache + SystemOneService, as in the original reproduction."""
    busy0, fetches = [False], []

    def http_get(url):
        fetches.append(url)
        return 200, [{"id": s, "is_processing": s == 0 and busy0[0]} for s in range(4)]

    cache = SlotsCache("http://upstream", http_get=http_get, max_age=0.1)
    router = SlotRouter(4, 1)
    agent = _agent_slot_with_kv(router)

    class PolledDuringS1(FakeEngine):
        def answer(self, state, q, *, id_slot=None, **kw):
            if len(self.calls) == 0:
                time.sleep(0.11)          # S1 #1's own pre-dispatch snapshot has expired by now
                busy0[0] = True           # llama-server: slot 0 is running our classification
                cache.get()               # ... and a /callo/stats poll snapshots exactly that
                busy0[0] = False          # the classification finishes
            return super().answer(state, q, id_slot=id_slot)

    eng = PolledDuringS1()
    svc = SystemOneService(eng, router, Stats(), model_name="m", slots_snapshot=cache.stamped)
    assert svc(BODY)[2] == "0"
    code, _, slot = svc(BODY)             # S1 #2, well inside the polled snapshot's 100 ms
    assert len(fetches) == 2              # S1 #2 really was served the stale snapshot (slot 0 busy)
    assert code == 200 and slot == "0" and router.counters["s1_fallback"] == 0
    assert router.snapshot()["text_chars"][str(agent)] > 0      # the agent's KV is still routable


def test_our_release_between_the_snapshot_and_routing_is_not_foreign():
    """The concurrent-client form of the same race: S1 #2's fresh snapshot shows S1 #1 on slot 0, then S1 #1 releases before
    S1 #2 routes. The slot is idle by callo's own accounting AND the snapshot predates that release: not foreign."""
    router = SlotRouter(4, 1)
    agent = _agent_slot_with_kv(router)
    router.acquire_s1(0)                  # S1 #1 in flight

    def snap():
        out = ([{"id": s, "is_processing": s == 0} for s in range(4)], time.monotonic())
        router.release_s1(0)              # S1 #1 finishes after the fetch, before S1 #2's route_s1
        return out

    _, _, slot = SystemOneService(FakeEngine(), router, Stats(), model_name="m", slots_snapshot=snap)(BODY)
    assert slot == "0" and router.counters["s1_fallback"] == 0
    assert router.snapshot()["text_chars"][str(agent)] > 0


def test_s1_fallback_can_be_disabled():
    router = SlotRouter(4, 1)
    snapshot = [{"id": 0, "is_processing": True}, {"id": 1, "is_processing": False}]
    svc = SystemOneService(FakeEngine(), router, Stats(), model_name="m", slots_snapshot=_now(snapshot),
                           fallback=False)
    _, _, slot = svc(BODY)
    assert slot == "0" and router.counters["s1_fallback"] == 0


def test_a_broken_slots_snapshot_fails_open_and_is_counted():
    def boom():
        raise http.client.IncompleteRead(b"par")

    router, stats, eng = SlotRouter(4, 1), Stats(), FakeEngine()
    code, _, slot = SystemOneService(eng, router, stats, model_name="m", slots_snapshot=boom)(BODY)
    assert code == 200 and slot == "0" and stats.counters["slots_snapshot_error"] == 1


def test_a_snapshot_with_non_dict_items_is_skipped():
    snapshot = ["junk", None, {"id": 0, "is_processing": True}, {"id": 1}]
    code, _, slot = SystemOneService(FakeEngine(), SlotRouter(4, 1), Stats(), model_name="m",
                                     slots_snapshot=_now(snapshot))(BODY)
    assert code == 200 and slot == "1"


def test_the_snapshot_is_fetched_outside_the_router_lock():
    router = SlotRouter(4, 1)
    out = []
    svc = SystemOneService(FakeEngine(), router, Stats(), model_name="m",
                           slots_snapshot=lambda: (router.inflight(0), ([], time.monotonic()))[1])
    t = threading.Thread(target=lambda: out.append(svc(BODY)), daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive() and out[0][0] == 200


def test_a_parser_bug_is_a_500_not_an_escape(monkeypatch, capsys):
    def bug(body):
        raise RuntimeError("parser bug")

    monkeypatch.setattr("callo.proxy.parse_request", bug)
    code, resp, _ = _svc(FakeEngine())(BODY)
    assert code == 500 and resp == {"detail": "internal error"}       # generic body (M-1), no raw exception text
    err = capsys.readouterr().err
    assert "Traceback" in err and "parser bug" in err                 # the real exception is logged server-side only


def test_stats_record_latency_and_low_mass():
    class LowMass(FakeEngine):
        def answer(self, state, q, *, id_slot=None, **kw):
            return Readout((0.5, 0.5), (-1.0, -1.0), 0.2, "no_shared_prefix", 0, 1, 5)

    stats = Stats()
    SystemOneService(LowMass(), SlotRouter(4, 1), stats, model_name="m")(BODY)
    snap = stats.snapshot(SlotRouter(4, 1), None)
    assert snap["s1"]["n"] == 2 and snap["s1"]["low_mass"] == 2


def test_unpinned_when_the_upstream_has_a_single_slot():
    eng = FakeEngine()
    _, _, slot = _svc(eng, SlotRouter(1, 1))(BODY)
    assert slot == "-" and eng.calls[0][1] is None


def test_end_to_end_through_the_proxy_against_the_fake_upstream():
    up = FakeUpstream().start()
    try:
        post = make_http_post(timeout=10)
        engine = Engine(LlamaCppForwardScorer(up.url, http_post=post), TemplateClient(up.url, http_post=post),
                        SurfaceSets(TOKENS, "gpt2"))
        router = SlotRouter(4, 1)
        stats = Stats()
        cfg = ProxyConfig(upstream=up.url, router=router, stats=stats,
                          systemone=SystemOneService(engine, router, stats, model_name="fake-9b").respond)
        srv = make_server(cfg, port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/v1/systemone",
                                         data=json.dumps(BODY).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
            with urllib.request.urlopen(req, timeout=10) as r:
                assert r.status == 200 and r.headers["X-Callo-Slot"] == "0"
                out = json.loads(r.read())
            assert out["answers"]["q"]["choice"] == "b" and out["answers"]["o"]["noul"] > 0.5
            comps = [q.json for q in up.requests if q.path == "/completion"]
            assert comps and all(c["id_slot"] == 0 and c["cache_prompt"] is False and c["n_probs"] == 100 for c in comps)
            tmpl = up.last("/apply-template").json
            assert tmpl["chat_template_kwargs"] == {"enable_thinking": False}
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(urllib.request.Request(req.full_url, data=b"not json", method="POST"), timeout=10)
            assert e.value.code == 422
            e.value.close()                     # the error response holds its connection open until closed
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        up.stop()


# ---- v0.2: images, lanes, the error contract, X-Callo-Timings, stats ---------------------------------------

PNG = "iVBORw0KGgo="
URL = f"data:image/png;base64,{PNG}"
DELIM = "<|im_end|>\n<|im_start|>user"
IMG_BODY = {"state": "", "images": [URL], "context": "orders: eat dots",
            "questions": {"move": {"type": "choice", "instructions": "Which way?", "criteria": {"a": None, "b": None}}}}


def _live(up, *, vision=True, s1_slots=1, delimiter=DELIM, audio=False, video=False):
    """A real proxy + service + engine against the fake upstream; returns (server, router, stats)."""
    post = make_http_post(timeout=10)
    engine = Engine(LlamaCppForwardScorer(up.url, http_post=post), TemplateClient(up.url, http_post=post),
                    SurfaceSets(TOKENS, "gpt2"), vision=vision,
                    media_marker=MEDIA_MARKER if vision or audio or video else None, user_delimiter=delimiter,
                    audio=audio, video=video)
    router, stats = SlotRouter(4, s1_slots), Stats()
    svc = SystemOneService(engine, router, stats, model_name="fake-9b")
    srv = make_server(ProxyConfig(upstream=up.url, router=router, stats=stats, systemone=svc.respond), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, router, stats


def _post(srv, body, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/v1/systemone",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


@pytest.fixture
def vup():
    u = FakeUpstream(vision=True).start()
    yield u
    u.stop()


def test_an_image_request_end_to_end_through_the_proxy(vup):
    vup.timings = {"prompt_n": 62, "cache_n": 354, "prompt_ms": 15.7}
    srv, _, stats = _live(vup)
    try:
        status, headers, out = _post(srv, IMG_BODY)
        assert status == 200 and headers["X-Callo-Slot"] == "0" and out["answers"]["move"]["choice"] == "b"
        comp = vup.last("/completion").json
        assert comp["prompt"]["multimodal_data"] == [PNG] and comp["cache_prompt"] is True and comp["id_slot"] == 0
        assert comp["message_delimiters"] == [{"role": "user", "delimiter": DELIM}]
        ps = comp["prompt"]["prompt_string"]
        assert ps.index("Context:\norders: eat dots") < ps.index(MEDIA_MARKER) < ps.index("Question: Which way?")
        part = vup.last("/apply-template").json["messages"][1]["content"][0]
        assert part == {"type": "image_url", "image_url": {"url": PLACEHOLDER_IMAGE_URL}}     # v0.3
        assert PNG not in vup.last("/apply-template").body.decode()
        timings = json.loads(headers["X-Callo-Timings"])
        assert timings == [{"q": "move", "kind": "image", "cache_n": 354, "prompt_n": 62, "prompt_ms": 15.7,
                            "tokens": out["usage"]["input_tokens"], "stop": "diverged",
                            "ms": timings[0]["ms"], "miss": 0, "passes": 1}]
        snap = stats.snapshot(SlotRouter(4, 1), None)["s1"]
        assert snap["image_requests"] == 1 and snap["image_cache_n_mean"] == 354 and snap["image_prompt_n_mean"] == 62
        assert snap["image_p50_ms"] is not None and snap["context_requests"] == 0 and snap["retok_stops"] == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_text_request_with_context_is_cached_and_delimited_and_counted(vup):
    srv, _, stats = _live(vup)
    try:
        status, _, _ = _post(srv, {**IMG_BODY, "images": None})
        comp = vup.last("/completion").json
        assert status == 200 and isinstance(comp["prompt"], list) and comp["cache_prompt"] is True
        assert comp["message_delimiters"] == [{"role": "user", "delimiter": DELIM}]
        s1 = stats.snapshot(SlotRouter(4, 1), None)["s1"]
        assert s1["context_requests"] == 1 and s1["image_requests"] == 0 and s1["context_cache_n_mean"] == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_empty_images_and_an_empty_or_null_context_stay_on_the_v01_path(vup):
    """A client that always sends the v0.2 keys, empty, must get v0.1's bit-stable request: no cache, no delimiter."""
    srv, _, stats = _live(vup)
    try:
        for extra in ({"images": [], "context": ""}, {"images": None, "context": None}):
            status, headers, _ = _post(srv, {**BODY, **extra})
            comp = vup.last("/completion").json
            assert status == 200 and comp["cache_prompt"] is False and "message_delimiters" not in comp
            assert isinstance(comp["prompt"], list) and json.loads(headers["X-Callo-Timings"])[0]["kind"] == "plain"
        assert stats.snapshot(SlotRouter(4, 1), None)["s1"]["context_requests"] == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_images_without_vision_are_501_and_the_same_body_without_images_is_200():
    up = FakeUpstream().start()
    srv, _, _ = _live(up, vision=False)
    try:
        status, headers, out = _post(srv, IMG_BODY)
        assert status == 501 and out == {"detail": "the upstream llama-server has no vision support; start it with "
                                                   "--mmproj"}
        assert "X-Callo-Timings" not in headers and not [r for r in up.requests if r.path == "/completion"]
        status, _, _ = _post(srv, {k: v for k, v in IMG_BODY.items() if k != "images"})
        assert status == 200
    finally:
        srv.shutdown()
        srv.server_close()
        up.stop()


@pytest.mark.parametrize(("code", "message", "want", "detail"), [
    (500, "Failed to load image or audio file", 422, "the upstream could not decode an image: Failed to load image"),
    (400, "Invalid uri-encoded base64 value", 422, "Invalid uri-encoded base64 value"),
    (400, "request (70000 tokens) exceeds the available context size (65536 tokens), try increasing it", 413,
     "the request with its media exceeds the context size"),                     # v0.3: 413
    (404, "File Not Found", 500, "upstream HTTP 404"),
    (500, "Failed to tokenize prompt", 503, "upstream HTTP 500"),
    (503, "Loading model", 503, "Loading model"),
])
def test_upstream_errors_on_an_image_request_map_per_the_contract(vup, code, message, want, detail):
    vup.completion_error = (code, message)
    srv, _, _ = _live(vup)
    try:
        status, headers, out = _post(srv, IMG_BODY)
        assert status == want and detail in out["detail"] and headers["X-Callo-Slot"] == "0"
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_501_or_a_marker_422_never_takes_a_slot_or_binds_a_lane(vup):
    up = FakeUpstream().start()
    srv, router, _ = _live(up, vision=False, s1_slots=2)
    try:
        status, headers, _ = _post(srv, IMG_BODY, {"X-Callo-Conversation": "s1"})
        assert status == 501 and headers["X-Callo-Slot"] == "-" and router.snapshot()["lanes"] == {}
    finally:
        srv.shutdown()
        srv.server_close()
        up.stop()
    srv, router, _ = _live(vup, s1_slots=2)
    try:
        status, headers, out = _post(srv, {**IMG_BODY, "state": f"see {MEDIA_MARKER}"}, {"X-Callo-Conversation": "s1"})
        assert status == 422 and "media marker" in out["detail"] and headers["X-Callo-Slot"] == "-"
        assert router.snapshot()["lanes"] == {} and not [r for r in vup.requests if r.path == "/completion"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_malformed_image_is_422_before_any_upstream_call(vup):
    srv, _, _ = _live(vup)
    try:
        status, headers, out = _post(srv, {**IMG_BODY, "images": ["data:image/png;base64,***"]})
        assert status == 422 and "images[0]" in out["detail"] and headers["X-Callo-Slot"] == "-"
        assert not [r for r in vup.requests if r.path in ("/completion", "/apply-template")]
    finally:
        srv.shutdown()
        srv.server_close()


def test_lanes_bind_on_first_sight_stick_and_show_in_the_router(vup):
    srv, router, _ = _live(vup, s1_slots=3)
    try:
        slots = [_post(srv, IMG_BODY, {"X-Callo-Conversation": lane})[1]["X-Callo-Slot"]
                 for lane in ("s1", "instinct_text", "instinct_view", "s1", "instinct_view")]
        assert slots == ["0", "1", "2", "0", "2"]
        assert _post(srv, IMG_BODY)[1]["X-Callo-Slot"] == "0"      # no header: v0.1 round robin
        assert router.snapshot()["lanes"] == {"s1": 0, "instinct_text": 1, "instinct_view": 2}
        assert [r.json["id_slot"] for r in vup.requests if r.path == "/completion"] == [0, 1, 2, 0, 2, 0]
    finally:
        srv.shutdown()
        srv.server_close()


def test_with_one_reserved_slot_the_lane_header_is_ignored(vup):
    srv, router, _ = _live(vup, s1_slots=1)
    try:
        assert [_post(srv, IMG_BODY, {"X-Callo-Conversation": c})[1]["X-Callo-Slot"] for c in ("a", "b")] == ["0", "0"]
        assert router.snapshot()["lanes"] == {}
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_busy_lane_waits_on_its_own_slot():
    router = SlotRouter(4, 2)
    router.route_s1(lane="s1")                                   # a tick still in flight on slot 0
    eng = FakeEngine()
    code, _, slot = SystemOneService(eng, router, Stats(), model_name="m")(BODY, {"x-callo-conversation": "s1"})
    assert code == 200 and slot == "0" and eng.calls[0][1] == 0 and router.counters["s1_lane_busy"] == 1


def test_the_service_maps_callo_errors_to_statuses():
    for exc, want in ((ImageRejected("bad image"), 422), (NoVision(), 501), (UpstreamUnavailable("down"), 503),
                      (UpstreamRejected("HTTP 404"), 500), (ReadoutError("markers"), 500)):
        code, out, _ = _svc(FakeEngine(exc))(BODY)
        assert code == want and out["detail"] == str(exc)


def test_the_timings_header_is_ascii_json_even_for_non_ascii_question_ids():
    r = Readout((0.5, 0.5), (-0.7, -0.7), 0.9, "retok", 1, 0, 70, kind="image", cache_n=5, prompt_n=2, prompt_ms=3.0)
    text = timings_header([QuestionReadout("ruta→norte\r\n", r, 12.34)])
    assert text.isascii() and "\r" not in text and "\n" not in text
    assert json.loads(text) == [{"q": "ruta→norte\r\n", "kind": "image", "cache_n": 5, "prompt_n": 2, "prompt_ms": 3.0,
                                 "tokens": 70, "stop": "retok", "ms": 12.3, "miss": 0, "passes": 0}]


def test_v02_stats_fields():
    stats = Stats()
    stats.vision, stats.media_marker, stats.user_delimiter = True, MEDIA_MARKER, DELIM
    stats.cache_prompt = {"plain": False, "image": True, "context": True}
    r = Readout((0.5, 0.5), (-0.7, -0.7), 0.9, "retok", 1, 0, 70, kind="image", cache_n=10, prompt_n=4, prompt_ms=3.0)
    stats.add_request("image", 300.0, [QuestionReadout("q", r, 290.0)])
    stats.add_request("image", 500.0, [QuestionReadout("q", dataclasses.replace(r, cache_n=30), 480.0)])
    snap = stats.snapshot(SlotRouter(4, 2), None)
    assert (snap["vision"], snap["media_marker"], snap["user_delimiter"]) == (True, MEDIA_MARKER, DELIM)
    assert snap["cache_prompt"] == {"plain": False, "image": True, "context": True}
    s1 = snap["s1"]
    assert s1["image_requests"] == 2 and s1["image_p50_ms"] == 300.0 and s1["image_p95_ms"] == 500.0
    assert s1["image_cache_n_mean"] == 20 and s1["image_prompt_n_mean"] == 4
    assert s1["retok_stops"] == 2 and s1["n"] == 2 and s1["context_cache_n_mean"] is None
    assert (s1["eog_stops"], s1["reasks"]) == (0, 0)
    assert snap["router"]["lanes"] == {} and snap["router"]["s1_lane_busy"] == 0
    stats.add_request("plain", 9.0, [QuestionReadout("q", dataclasses.replace(r, stop="eog", reasks=2), 8.0),
                                     QuestionReadout("q2", dataclasses.replace(r, stop="diverged", reasks=1), 8.0)])
    s1 = stats.snapshot(SlotRouter(4, 2), None)["s1"]
    assert (s1["retok_stops"], s1["eog_stops"], s1["reasks"]) == (2, 1, 3)


# ---- v0.2: the X-Callo-Timings cap, respond always answers, raw backend-down, lane strip -----------------

def _many(n):
    return {"state": "s", "questions": {f"question_{i:04d}": {"type": "choice", "criteria": {"a": None, "b": None}}
                                        for i in range(n)}}


def _qr(qid):
    r = Readout((0.5, 0.5), (-0.7, -0.7), 0.9, "diverged", 0, 0, 12, kind="plain")
    return QuestionReadout(qid, r, 1.0)


def test_the_timings_header_is_capped_and_ends_with_a_truncated_count():
    text = timings_header([_qr(f"question_{i:04d}") for i in range(600)])
    assert text.isascii() and len(text.encode()) <= TIMINGS_MAX_BYTES
    entries = json.loads(text)
    assert entries[-1].keys() == {"truncated"} and entries[-1]["truncated"] > 0
    kept = entries[:-1]
    assert [e["q"] for e in kept] == [f"question_{i:04d}" for i in range(len(kept))]   # the leading ones, in order
    assert len(kept) + entries[-1]["truncated"] == 600
    nxt = json.loads(timings_header([_qr(f"question_{len(kept):04d}")]))[0]                 # as many as fit
    longer = kept + [nxt, {"truncated": 600 - len(kept) - 1}]
    assert len(json.dumps(longer, separators=(",", ":"))) > TIMINGS_MAX_BYTES


def test_a_qid_too_long_for_the_cap_leaves_only_the_truncated_count():
    assert json.loads(timings_header([_qr("x" * 70_000)])) == [{"truncated": 1}]
    assert json.loads(timings_header([_qr("x" * 70_000), _qr("q")])) == [{"truncated": 2}]


def test_a_small_request_has_no_truncated_element():
    entries = json.loads(timings_header([_qr("a"), _qr("b")]))
    assert [e["q"] for e in entries] == ["a", "b"] and all("truncated" not in e for e in entries)


def test_a_many_question_request_through_the_proxy_reads_as_a_200():
    svc = SystemOneService(FakeEngine(), SlotRouter(4, 1), Stats(), model_name="m")
    srv = make_server(ProxyConfig(upstream="http://127.0.0.1:9", router=svc.router, stats=svc.stats,
                                  systemone=svc.respond), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        status, headers, out = _post(srv, _many(600))
        assert status == 200 and len(out["answers"]) == 600
        timings = json.loads(headers["X-Callo-Timings"])
        assert len(headers["X-Callo-Timings"]) <= TIMINGS_MAX_BYTES and len(timings) - 1 + timings[-1]["truncated"] == 600
    finally:
        srv.shutdown()
        srv.server_close()


def test_respond_answers_500_for_a_bug_outside_the_readout_and_never_leaks_a_slot(capsys):
    class BadCheck(FakeEngine):
        def check_request(self, req):
            raise TypeError("check bug")

    class BadStats(Stats):
        def add_request(self, kind, ms, readouts):
            raise RuntimeError("stats bug")

    router = SlotRouter(4, 2)
    code, out, extra = SystemOneService(BadCheck(), router, Stats(), model_name="m").respond(
        BODY, {"X-Callo-Conversation": "s1"})
    assert code == 500 and out == {"detail": "internal error"} and extra["X-Callo-Slot"] == "-"   # generic (M-1)
    assert router.snapshot()["lanes"] == {}
    code, out, extra = SystemOneService(FakeEngine(), router, BadStats(), model_name="m").respond(BODY)
    assert code == 500 and out == {"detail": "internal error"}
    assert extra["X-Callo-Slot"] == "0" and all(router.inflight(s) == 0 for s in range(4))
    err = capsys.readouterr().err                 # the real exceptions are logged server-side, never in the body
    assert "check bug" in err and "stats bug" in err
    svc = SystemOneService(BadCheck(), SlotRouter(4, 1), Stats(), model_name="m")       # and over the wire
    srv = make_server(ProxyConfig(upstream="http://127.0.0.1:9", router=svc.router, stats=svc.stats,
                                  systemone=svc.respond), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        status, _, out = _post(srv, BODY)
        assert status == 500 and out == {"detail": "internal error"}
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_forced_internal_error_yields_a_generic_500_body(capsys):
    """A 500 from an unexpected exception must not carry the raw exception text (it can leak a cache path or an
    internal field name); the body is generic and the real error is logged server-side."""
    code, out, _ = _svc(FakeEngine(OSError("/Users/secret/.cache/callo/vocab-abc.json: Permission denied")))(BODY)
    assert code == 500 and out == {"detail": "internal error"}
    err = capsys.readouterr().err
    assert "Permission denied" in err and "Permission denied" not in json.dumps(out)


def test_a_raw_backend_down_error_from_answer_parsed_is_503():
    class Unwrapped(FakeEngine):
        def answer_parsed(self, req, **kw):
            raise urllib.error.URLError("refused")

    router = SlotRouter(4, 1)
    code, out, extra = SystemOneService(Unwrapped(), router, Stats(), model_name="m").respond(BODY)
    assert code == 503 and out["detail"].startswith("llama-server unavailable:") and extra["X-Callo-Slot"] == "0"
    assert router.inflight(0) == 0 and "X-Callo-Timings" not in extra


def test_the_lane_header_value_is_stripped():
    router = SlotRouter(4, 2)
    svc = SystemOneService(FakeEngine(), router, Stats(), model_name="m")
    assert svc.respond(BODY, {"X-Callo-Conversation": " s1 "})[2]["X-Callo-Slot"] == "0"
    assert svc.respond(BODY, {"x-callo-conversation": "s1"})[2]["X-Callo-Slot"] == "0"
    assert router.snapshot()["lanes"] == {"s1": 0}
    svc.respond(BODY, {"X-Callo-Conversation": "   "})
    assert router.snapshot()["lanes"] == {"s1": 0}


# ---- v0.3: miss in X-Callo-Timings, the position-bias and readout stats ----------------------------------

def test_the_timings_header_carries_each_readouts_floored_option_count():
    r = Readout(probs=(0.6, 0.3, 0.1), log_marginals=(-0.5, -1.2, -9.0), option_mass=0.9, stop="diverged", depth=0,
                topk_miss=2, prompt_tokens=40)
    entries = json.loads(timings_header([QuestionReadout("q", r, 5.0), QuestionReadout("p", dataclasses.replace(
        r, topk_miss=0), 5.0)]))
    assert [e["miss"] for e in entries] == [2, 0]
    assert list(entries[0]) == ["q", "kind", "cache_n", "prompt_n", "prompt_ms", "tokens", "stop", "ms", "miss",
                                "passes"]


def test_the_service_reply_carries_miss_for_every_question():
    class Floored(FakeEngine):
        def answer(self, state, q, *, id_slot=None, **kw):
            return Readout(probs=(0.25, 0.75), log_marginals=(-1.4, -0.3), option_mass=0.9, stop="diverged", depth=0,
                           topk_miss=1, prompt_tokens=12)

    code, _, extra = _svc(Floored()).respond(BODY)
    assert code == 200 and [e["miss"] for e in json.loads(extra["X-Callo-Timings"])] == [1, 1]


def test_v03_stats_count_readouts_misses_and_bias_categories():
    stats = Stats()
    stats.position_bias = {"profile": "p.json", "sha256": "f" * 64, "entries": [3], "kinds": ["plain"],
                           "state": "clean"}
    stats.model_key, stats.model_file, stats.provenance = "sha256:" + "a" * 64, "m.gguf", {"build_info": "b"}
    r = Readout(probs=(0.5, 0.3, 0.2), log_marginals=(-0.7, -1.2, -1.6), option_mass=0.9, stop="diverged", depth=0,
                topk_miss=1, prompt_tokens=20, bias="applied")
    readouts = [QuestionReadout("a", r, 1.0), QuestionReadout("b", dataclasses.replace(r, bias="no_entry"), 1.0),
                QuestionReadout("c", dataclasses.replace(r, bias="kind_not_calibrated", topk_miss=0), 1.0)]
    stats.add_request("plain", 3.0, readouts)
    snap = stats.snapshot(SlotRouter(4, 1), None)
    assert snap["s1"]["readouts"] == 3 and snap["s1"]["topk_miss"] == 2
    assert snap["model_key"] == "sha256:" + "a" * 64 and snap["model_file"] == "m.gguf"
    assert snap["provenance"] == {"build_info": "b"}
    pb = snap["position_bias"]
    assert pb["profile"] == "p.json" and pb["applied"] == 1 and pb["entries"] == [3]
    assert pb["skipped"] == {"no_entry": 1, "noncanonical": 0, "score": 0, "single_option": 0,
                             "kind_not_calibrated": 1, "deliberate": 0, "wide": 0}     # wide choice: skipped.wide


def test_stats_without_a_profile_report_null_and_still_count_readouts():
    stats = Stats()
    single = Readout(probs=(1.0,), log_marginals=(0.0,), option_mass=1.0, stop="single_option", depth=0, topk_miss=0,
                     prompt_tokens=0)
    stats.add_request("plain", 1.0, [QuestionReadout("q", single, 0.1)])
    snap = stats.snapshot(SlotRouter(4, 1), None)
    assert snap["position_bias"] is None and snap["s1"]["readouts"] == 1 and snap["model_key"] is None


def test_s1_readouts_is_monotonic_past_the_2000_window():
    stats = Stats()
    r = Readout(probs=(0.5, 0.5), log_marginals=(-0.7, -0.7), option_mass=0.9, stop="diverged", depth=0, topk_miss=0,
                prompt_tokens=10)
    for _ in range(25):
        stats.add_request("plain", 1.0, [QuestionReadout(f"q{i}", r, 1.0) for i in range(100)])
    s1 = stats.snapshot(SlotRouter(4, 1), None)["s1"]
    assert s1["readouts"] == 2500 and s1["n"] == 2000                     # n is the latency window


def test_the_service_maps_the_media_errors_to_statuses():
    """v0.3: 422 for media the upstream could not load, 501 for each kind it lacks, 413 past the context."""
    for exc, want in ((MediaLoadFailed("could not decode an audio clip"), 422), (NoAudio(), 501), (NoVideo(), 501),
                      (MediaTooLarge("exceeds the context size"), 413)):
        code, out, _ = _svc(FakeEngine(exc))(BODY)
        assert code == want and out["detail"] == str(exc)


# ---- v0.3: audio, video and mixed through the proxy; passes; the per-kind stats ---------------------------------

WAV_B64 = base64.b64encode(TONE_WAV).decode()
AVI_B64 = base64.b64encode(RED_AVI).decode()
AV_Q = {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}
AUDIO_REQ = {"state": "", "audio": [TONE_URL], "questions": AV_Q}
VIDEO_REQ = {"state": "", "videos": [RED_URL], "questions": AV_Q}


@pytest.fixture
def avup():
    u = FakeUpstream(vision=True, audio=True, video=True).start()
    yield u
    u.stop()


def test_an_audio_request_end_to_end_sends_a_placeholder_part_and_the_real_clip(avup):
    srv, _, stats = _live(avup, audio=True, video=True)
    try:
        status, headers, out = _post(srv, AUDIO_REQ)
        assert status == 200 and out["answers"]["q"]["choice"] == "a"          # the fake hears the tone (swapped)
        part = avup.last("/apply-template").json["messages"][1]["content"][0]
        assert part == {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
        comp = avup.last("/completion").json
        assert comp["prompt"]["multimodal_data"] == [WAV_B64] and comp["cache_prompt"] is True
        [t] = json.loads(headers["X-Callo-Timings"])
        assert t["kind"] == "audio" and t["passes"] == 1 and list(t)[-2:] == ["miss", "passes"]
        s1 = stats.snapshot(SlotRouter(4, 1), None)["s1"]
        assert s1["audio_requests"] == 1 and s1["audio_p50_ms"] is not None and s1["audio_cache_n_mean"] == 0
        assert s1["image_requests"] == 0 and s1["video_requests"] == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_video_request_end_to_end_sends_a_placeholder_part_and_the_real_clip(avup):
    """A video-only 200 through the proxy, as the audio test above."""
    srv, _, stats = _live(avup, audio=True, video=True)
    try:
        status, headers, out = _post(srv, VIDEO_REQ)
        # the fake sees the red clip (swapped)
        assert status == 200 and out["answers"]["q"]["choice"] == "a"
        part = avup.last("/apply-template").json["messages"][1]["content"][0]
        assert part == {"type": "input_video", "input_video": {"data": "AAAA"}}
        comp = avup.last("/completion").json
        assert comp["prompt"]["multimodal_data"] == [AVI_B64] and comp["cache_prompt"] is True
        assert comp["prompt"]["prompt_string"].count(MEDIA_MARKER) == 1
        [t] = json.loads(headers["X-Callo-Timings"])
        assert t["kind"] == "video" and t["passes"] == 1
        s1 = stats.snapshot(SlotRouter(4, 1), None)["s1"]
        assert s1["video_requests"] == 1 and s1["audio_requests"] == 0 and s1["image_requests"] == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_mixed_request_keeps_the_contract_order_to_the_upstream(avup):
    srv, _, stats = _live(avup, audio=True, video=True)
    try:
        body = {"state": "", "images": [URL], "audio": [TONE_URL], "videos": [RED_URL], "questions": AV_Q}
        status, headers, _ = _post(srv, body)
        assert status == 200 and json.loads(headers["X-Callo-Timings"])[0]["kind"] == "mixed"
        assert avup.last("/completion").json["prompt"]["multimodal_data"] == [PNG, WAV_B64, AVI_B64]
        parts = avup.last("/apply-template").json["messages"][1]["content"]
        assert [p["type"] for p in parts] == ["image_url", "input_audio", "input_video", "text"]
        s1 = stats.snapshot(SlotRouter(4, 1), None)["s1"]
        assert s1["mixed_requests"] == 1 and "mixed_p50_ms" not in s1
    finally:
        srv.shutdown()
        srv.server_close()


def test_audio_and_video_on_an_upstream_without_them_are_501_before_any_upstream_call():
    up = FakeUpstream(vision=True).start()
    srv, router, _ = _live(up, s1_slots=2)
    try:
        for body, word in ((AUDIO_REQ, "audio"), (VIDEO_REQ, "video")):
            status, headers, out = _post(srv, body, {"X-Callo-Conversation": "s1"})
            assert status == 501 and f"no {word} support" in out["detail"] and headers["X-Callo-Slot"] == "-"
        assert router.snapshot()["lanes"] == {} and not [r for r in up.requests if r.path == "/completion"]
    finally:
        srv.shutdown()
        srv.server_close()
        up.stop()


def test_an_upstream_that_lost_audio_behind_callo_answers_501_not_503():
    """callo's view can diverge from the upstream's (a restart behind callo). /apply-template's own refusal
    ("audio input is not supported") is a 501, so Kev does not retry it forever."""
    up = FakeUpstream(vision=True).start()                                    # no audio upstream
    srv, _, _ = _live(up, audio=True)                                         # but callo believes there is
    try:
        status, _, out = _post(srv, AUDIO_REQ)
        assert status == 501 and "no audio support" in out["detail"]
    finally:
        srv.shutdown()
        srv.server_close()
        up.stop()


def test_a_video_the_upstream_cannot_load_is_a_422(avup):
    avup.video_fails = True                                                    # e.g. no ffmpeg on its PATH
    srv, _, _ = _live(avup, audio=True, video=True)
    try:
        status, _, out = _post(srv, VIDEO_REQ)
        assert status == 422 and out["detail"].startswith("the upstream could not decode a video: Failed to load")
    finally:
        srv.shutdown()
        srv.server_close()


def test_v03_media_stats_fields():
    stats = Stats()
    assert stats.snapshot(SlotRouter(4, 1), None)["self_test"] == {"text": None, "vision": None, "audio": None,
                                                                   "video": None, "think": None}
    stats.audio, stats.video = True, False
    stats.self_test.update(text=True, audio=False)
    r = Readout(probs=(0.5, 0.5), log_marginals=(-0.7, -0.7), option_mass=0.9, stop="diverged", depth=0, topk_miss=0,
                prompt_tokens=70, kind="audio", cache_n=10, prompt_n=4, prompt_ms=3.0, passes=1)
    stats.add_request("audio", 300.0, [QuestionReadout("q", r, 290.0)])
    stats.add_request("audio", 500.0, [QuestionReadout("q", dataclasses.replace(r, cache_n=30), 480.0)])
    stats.add_request("video", 900.0, [QuestionReadout("q", dataclasses.replace(r, kind="video", prompt_n=8), 1.0)])
    stats.add_request("mixed", 50.0, [QuestionReadout("q", dataclasses.replace(r, kind="mixed"), 1.0)])
    snap = stats.snapshot(SlotRouter(4, 1), None)
    assert (snap["audio"], snap["video"]) == (True, False)
    assert snap["self_test"] == {"text": True, "vision": None, "audio": False, "video": None, "think": None}
    s1 = snap["s1"]
    assert (s1["audio_requests"], s1["audio_p50_ms"], s1["audio_p95_ms"]) == (2, 300.0, 500.0)
    assert (s1["audio_cache_n_mean"], s1["audio_prompt_n_mean"]) == (20, 4)
    assert (s1["video_requests"], s1["video_p50_ms"], s1["video_prompt_n_mean"]) == (1, 900.0, 8)
    assert s1["mixed_requests"] == 1 and s1["image_requests"] == 0 and s1["image_p50_ms"] is None
    assert snap["s1"]["readouts"] == 4


def test_a_state_part_whose_type_is_not_a_string_is_answered_not_a_500():
    """v0.3: such a part is no media part; it stays in the state, as in v0.2's intent,
    instead of a TypeError from the set lookup that respond turned into a 500."""
    body = {"state": [{"role": "user", "content": [{"type": ["string", "null"]}, {"type": "text", "text": "hi"}]}],
            "questions": {"q": {"type": "choice", "criteria": {"a": "one", "b": "two"}}}}
    code, out, _ = _svc(FakeEngine())(body)
    assert code == 200 and out["answers"]["q"]["choice"] == "b"


def test_the_fake_upstream_answers_with_llama_cpps_own_statuses(avup):
    """/completion's tokenization failures are 400s; /apply-template refuses a part llama.cpp
    would refuse (a cross-mixed key, an empty payload) with llama.cpp's status, so a wrong placeholder cannot pass the
    integration tests and then fail live."""
    def post(path, body):
        req = urllib.request.Request(avup.url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def template(*parts):
        return post("/apply-template", {"messages": [{"role": "user", "content": [*parts, {"type": "text",
                                                                                           "text": "q"}]}]})

    assert template({"type": "input_video", "input_video": {"data": "AAAA"}})[0] == 200
    code, out = template({"type": "input_video", "video_url": {"data": "AAAA"}})               # cross-mixed
    assert (code, out["error"]["message"]) == (500, "Invalid base64 value")
    code, out = template({"type": "image_url", "image_url": {"url": "data:image/png,QQ=="}})
    assert (code, out["error"]["message"]) == (400, "uri must be base64 encoded")
    assert template({"type": ["input_audio"], "text": "schema"})[0] == 200                   # no media part (I-1)
    avup.video_fails = True
    code, out = post("/completion", {"prompt": {"prompt_string": "x", "multimodal_data": [AVI_B64]}})
    # load before tokenize
    assert (code, out["error"]["message"]) == (400, "Failed to load image or audio file")
    avup.video_fails = False
    # undecodable or empty: no bitmap, so "Failed to load"
    for bad in ("!!notbase64", ""):
        code, out = post("/completion", {"prompt": {"prompt_string": MEDIA_MARKER, "multimodal_data": [bad]}})
        assert (code, out["error"]["message"]) == (400, "Failed to load image or audio file")
    for prompt_string in ("x", MEDIA_MARKER * 2):         # mtmd: markers and bitmaps must pair one to one
        code, out = post("/completion", {"prompt": {"prompt_string": prompt_string,
                                                    "multimodal_data": [AVI_B64]}})
        assert (code, out["error"]["message"]) == (400, "Failed to tokenize prompt")
    ok = {"prompt": {"prompt_string": MEDIA_MARKER + "x", "multimodal_data": [AVI_B64]}}
    assert post("/completion", ok)[0] == 200


@pytest.mark.parametrize(("fake", "entry", "message"), [
    # no projector
    ({}, WAV_B64, "Multimodal data provided, but model does not support multimodal requests."),
    # audio bytes, no audio (the sniff)
    ({"vision": True}, WAV_B64, "Failed to load image or audio file"),
    # a video container, no video
    ({"vision": True}, AVI_B64, "Failed to load image or audio file"),
])
def test_the_fake_upstream_refuses_media_its_modalities_lack_on_completion(fake, entry, message):
    up = FakeUpstream(**fake).start()
    try:
        body = {"prompt": {"prompt_string": MEDIA_MARKER, "multimodal_data": [entry]}}
        req = urllib.request.Request(up.url + "/completion", method="POST", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        with e.value:
            assert (e.value.code, json.loads(e.value.read())["error"]["message"]) == (400, message)
    finally:
        up.stop()


# The audio cases, on the fake upstream's own /apply-template now that it marks audio and video parts (its
# "audio input is not supported" row is test_an_upstream_that_lost_audio_behind_callo_answers_501_not_503 now,
# where the real fake sends it from /apply-template as llama.cpp does).
ONE_OPTION = {"q": {"type": "choice", "criteria": {"only": "the single option"}}}


def test_an_audio_request_without_audio_is_a_501_before_any_upstream_call_through_the_proxy(vup):
    """v0.3: check_request refuses a missing kind before a slot is taken, before any upstream call, and before
    the single-option shortcut. The control (an image request on the same proxy) shows the path does reach the
    upstream when the kind is supported."""
    srv, _, _ = _live(vup, audio=False)
    try:
        status, headers, _ = _post(srv, IMG_BODY)                              # control: a supported kind
        assert status == 200 and headers["X-Callo-Slot"] == "0"
        assert [r for r in vup.requests if r.path == "/completion"]
        before = len(vup.requests)
        for questions in (IMG_BODY["questions"], ONE_OPTION):
            status, headers, out = _post(srv, {**AUDIO_REQ, "questions": questions})
            assert status == 501 and "no audio support" in out["detail"] and headers["X-Callo-Slot"] == "-"
        assert len(vup.requests) == before                                    # the 501s made no upstream call
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize(("code", "message", "want", "detail"), [
    (400, "Failed to load image or audio file", 422, "the upstream could not decode an audio clip: Failed to load"),
    (400, "request (70000 tokens) exceeds the available context size (65536 tokens), try increasing it", 413,
     "the request with its media exceeds the context size"),
    (400, "Failed to tokenize prompt", 500, "upstream HTTP 400"),
    (500, "Invalid base64 value", 503, "upstream HTTP 500"),
])
def test_upstream_errors_on_an_audio_request_map_per_the_media_table(avup, code, message, want, detail):
    """v0.3: an audio request's upstream errors use the media table, not v0.1's text mapping."""
    avup.completion_error = (code, message)
    srv, _, _ = _live(avup, audio=True, video=True)
    try:
        status, _, out = _post(srv, AUDIO_REQ)
        assert status == want and detail in out["detail"]
    finally:
        srv.shutdown()
        srv.server_close()
