"""POST /v1/audio/transcriptions through the proxy, against the fake llama-server: the translated
path, the routing, the headers, the replies, every passthrough row passing through untouched,
disconnects, a dying or missing upstream."""
import base64
import gzip
import http.client
import json
import socket
import threading
import time

import pytest
from fake_upstream import FakeUpstream
from helpers import form_field, form_file, multipart

import callo.transcribe as transcribe
from callo.av import TONE_WAV
from callo.proxy import ProxyConfig, Stats, make_server
from callo.slots import SlotRouter
from callo.transcribe import Transcription

CT = "multipart/form-data; boundary=XyZ"
BODY = multipart(form_file(TONE_WAV), form_field("temperature", "0"))
AUDIO = {"type": "input_audio", "input_audio": {"data": base64.b64encode(TONE_WAV).decode(), "format": "wav"}}
REPLY = {"choices": [{"message": {"role": "assistant", "content": "hello world"}}],
         "usage": {"completion_tokens": 3, "prompt_tokens": 120, "total_tokens": 123,
                   "prompt_tokens_details": {"cached_tokens": 0}}, "timings": {"prompt_n": 120}}
ASR = {"type": "transcript.text.done", "text": "hello world",
       "usage": {"type": "tokens", "input_tokens": 120, "output_tokens": 3, "total_tokens": 123,
                 "input_tokens_details": {"cached_tokens": 0}}}


def _chat(u, rec, h, reply=REPLY):
    h._reply(200, reply, {"Access-Control-Allow-Origin": "http://app", "X-Upstream": "1",
                          "Content-Type": "application/json; charset=utf-8"})


@pytest.fixture
def up():
    u = FakeUpstream(audio=True).start()
    u.routes["/v1/chat/completions"] = u.routes["/chat/completions"] = _chat
    yield u
    u.stop()


def _serve(cfg):
    srv = make_server(cfg, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def proxy(up):
    cfg = ProxyConfig(upstream=up.url, router=SlotRouter(4, 1), transcription=Transcription(lfm2=False))
    srv = _serve(cfg)
    yield srv, cfg
    srv.shutdown()
    srv.server_close()


def post(srv, body=BODY, headers=None, path="/v1/audio/transcriptions"):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    c.request("POST", path, body=body, headers={"Content-Type": CT, **(headers or {})})
    r = c.getresponse()
    raw_headers = r.getheaders()
    out = r.status, {k.lower(): v for k, v in raw_headers}, r.read(), raw_headers
    c.close()
    return out


def raw(srv, payload: bytes) -> bytes:
    s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10)
    s.sendall(payload)
    out = b""
    while chunk := s.recv(65536):
        out += chunk
    s.close()
    return out


def _wait(pred, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _idle(router):
    return _wait(lambda: all(router.inflight(s) == 0 for s in range(router.total_slots)))


def post_idle(srv, cfg, *args, **kwargs):
    """`post`, then wait for the slot's release: the handler releases only after the client has the whole reply, so a
    test whose next routing depends on this one must not race it."""
    out = post(srv, *args, **kwargs)
    assert _idle(cfg.router)
    return out


def _transcription_counters(cfg):
    return {k: v for k, v in cfg.stats.counters.items() if k.startswith("transcription_")}


# ---- the translated path ------------------------------------------------------------------------------------------

def test_a_well_formed_transcription_is_translated_pinned_and_reshaped(up, proxy):
    srv, cfg = proxy
    status, headers, data, raw_headers = post(srv)
    assert status == 200 and json.loads(data) == ASR
    assert headers["content-type"] == "application/json; charset=utf-8" and headers["content-length"] == str(len(data))
    assert headers["x-callo-slot"] == "1" and headers["access-control-allow-origin"] == "http://app"   # CORS
    assert headers["x-upstream"] == "1" and [k.lower() for k, _ in raw_headers].count("server") == 1
    rec = up.last("/v1/chat/completions")
    assert rec.json == {"id_slot": 1, "temperature": 0.0, "stream": False, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Transcribe audio to text"}, AUDIO]}]}
    assert type(rec.json["id_slot"]) is int and rec.headers["content-type"] == "application/json"
    assert rec.headers["content-length"] == str(len(rec.body))
    assert not [r for r in up.requests if "/audio/" in r.path]
    assert _transcription_counters(cfg) == {"transcription_translated": 1} and _idle(cfg.router)


def test_the_unprefixed_route_and_the_query_string_are_kept(up, proxy):
    srv, _ = proxy
    assert post(srv, path="/audio/transcriptions?trace=1")[0] == 200
    assert up.requests[-1].path == "/chat/completions?trace=1"


def test_the_clients_auth_and_origin_are_forwarded_and_callo_adds_no_key(up, proxy):
    srv, _ = proxy
    post(srv, headers={"Authorization": "Bearer client", "X-Api-Key": "k2", "Origin": "http://app",
                       "Accept-Encoding": "gzip", "X-Trace": "t"})
    h = up.last("/v1/chat/completions").headers
    assert (h["authorization"], h["x-api-key"], h["origin"], h["x-trace"]) == ("Bearer client", "k2", "http://app", "t")
    assert h["content-type"] == "application/json" and h["accept-encoding"] == "identity"     # not the client's gzip
    post(srv)
    assert "authorization" not in up.last("/v1/chat/completions").headers     # never callo's own key


def test_a_form_id_slot_is_ignored_and_a_reserved_slot_is_never_used(up, proxy):
    srv, cfg = proxy
    for _ in range(6):
        _, headers, _, _ = post(srv, multipart(form_file(TONE_WAV), form_field("id_slot", "0")))
        assert headers["x-callo-slot"] in {"1", "2", "3"}
        assert up.last("/v1/chat/completions").json["id_slot"] == int(headers["x-callo-slot"])
    assert cfg.router.counters["id_slot_ignored"] == 0 and cfg.router.counters["rerouted"] == 0   # client_slot=None


def test_routing_is_hint_then_lru_never_lcp(up, proxy):
    """An idle agent slot whose conversation starts with a user turn shares the transcription's JSON
    scaffolding, enough for the 10% LCP test; the transcription takes the LRU slot instead (its unknown `callo:asr`
    hint binds there)."""
    srv, cfg = proxy
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    c.request("POST", "/v1/chat/completions", body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
    assert c.getresponse().read() and c.close() is None                # the agent chat lands on slot 1
    assert _idle(cfg.router)
    _, h1, _, _ = post_idle(srv, cfg)
    assert h1["x-callo-slot"] == "2" and cfg.router.counters["lcp"] == 0
    _, h2, _, _ = post_idle(srv, cfg, multipart(form_file(TONE_WAV), form_field("user", "alice")))
    _, h3, _, _ = post_idle(srv, cfg, multipart(form_file(TONE_WAV), form_field("user", "alice")))
    assert h2["x-callo-slot"] == h3["x-callo-slot"] == "3" and cfg.router.counters["hint"] == 1


def test_hintless_transcriptions_converge_on_one_slot(up, proxy):
    """With no client hint, `callo:asr` keeps every transcription on one agent slot instead of walking the
    cold ones by LRU."""
    srv, cfg = proxy
    assert {post_idle(srv, cfg)[1]["x-callo-slot"] for _ in range(4)} == {"1"}
    assert cfg.router.counters["hint"] == 3 and cfg.router.counters["lru"] == 1
    assert cfg.router.snapshot()["sticky"] == {"callo:asr": {"home": 1, "rehomed": 0}}    # one shared hint, not one
    assert cfg.router.snapshot()["conversations"] == 0                  # per request; reported apart from conversations


def test_a_client_hint_still_wins_over_callo_asr(up, proxy):
    srv, cfg = proxy
    assert post_idle(srv, cfg)[1]["x-callo-slot"] == "1"                                 # callo:asr binds slot 1
    assert post_idle(srv, cfg, multipart(form_file(TONE_WAV), form_field("user", "alice")))[1]["x-callo-slot"] == "2"
    assert post_idle(srv, cfg, headers={"X-Callo-Conversation": "bob"})[1]["x-callo-slot"] == "3"
    assert post_idle(srv, cfg)[1]["x-callo-slot"] == "1"                                 # back on the asr slot


def test_a_busy_home_serves_elsewhere_and_the_next_transcription_goes_home(up, proxy):
    """While the `callo:asr` home is busy, that one transcription takes the normal pick (the LRU idle
    agent slot, not deferred onto the busy home), and the hint stays home: the next one goes back there."""
    srv, cfg = proxy
    assert post_idle(srv, cfg)[1]["x-callo-slot"] == "1"
    cfg.router.acquire(1, "agent")                      # an agent chat now holds the home
    try:
        assert post(srv)[1]["x-callo-slot"] == "2" and cfg.router.counters["deferred"] == 0
        assert _wait(lambda: cfg.router.inflight(2) == 0)
    finally:
        cfg.router.release(1)
    assert _idle(cfg.router)
    assert post_idle(srv, cfg)[1]["x-callo-slot"] == "1"     # home again: the hint did not move


def test_with_every_agent_slot_busy_it_queues_on_an_agent_slot_never_the_reserved_one(up, proxy):
    srv, cfg = proxy
    for s in (1, 2, 3):
        cfg.router.acquire(s, "agent")
    try:
        _, headers, _, _ = post(srv)
        assert headers["x-callo-slot"] in {"1", "2", "3"} and cfg.router.counters["deferred"] == 1
    finally:
        for s in (1, 2, 3):
            cfg.router.release(s)


def test_curls_expect_100_continue_is_answered_and_the_upload_translated(up, proxy):
    srv, _ = proxy
    s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10)
    s.sendall(b"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: x\r\nContent-Type: " + CT.encode()
              + b"\r\nExpect: 100-continue\r\nContent-Length: %d\r\n\r\n" % len(BODY))
    assert s.recv(65536).startswith(b"HTTP/1.1 100 Continue")
    s.sendall(BODY)
    out = b""
    while chunk := s.recv(65536):
        out += chunk
    s.close()
    assert out.startswith(b"HTTP/1.0 200") and b'"transcript.text.done"' in out
    assert "expect" not in up.last("/v1/chat/completions").headers


# ---- replies ------------------------------------------------------------------------------------------------------

def test_a_non_2xx_is_relayed_byte_identically(up, proxy):
    srv, cfg = proxy
    up.routes["/v1/chat/completions"] = lambda u, rec, h: h._reply(
        401, b'{"error":{"message":"Invalid API Key"}} ', {"Content-Type": "application/json", "X-Why": "auth"})
    status, headers, data, _ = post(srv)
    assert (status, data, headers["x-why"], headers["x-callo-slot"]) == (
        401, b'{"error":{"message":"Invalid API Key"}} ', "auth", "1")
    assert _transcription_counters(cfg) == {} and _idle(cfg.router)


def test_a_200_without_a_content_string_is_a_502_counted_as_a_bad_reply(up, proxy):
    srv, cfg = proxy
    up.routes["/v1/chat/completions"] = lambda u, rec, h: h._reply(200, {"choices": []})
    status, headers, data, _ = post(srv)
    assert status == 502 and "content" in json.loads(data)["detail"] and headers["x-callo-slot"] == "1"
    assert _transcription_counters(cfg) == {"transcription_bad_reply": 1}
    assert cfg.stats.counters["upstream_errors"] == 0 and _idle(cfg.router)


def test_a_non_ascii_transcript_is_utf8_with_its_byte_length(up, proxy):
    srv, _ = proxy
    reply = {**REPLY, "choices": [{"message": {"content": "日本語のテキスト, naïve"}}]}
    up.routes["/v1/chat/completions"] = lambda u, rec, h: _chat(u, rec, h, reply)
    _, headers, data, _ = post(srv)
    assert json.loads(data.decode("utf-8"))["text"] == "日本語のテキスト, naïve"
    assert "日本語".encode() in data and int(headers["content-length"]) == len(data)    # raw UTF-8


# ---- every passthrough row passes through untouched and is counted ------------------------------------------------

BODY_ROWS = {
    "not_multipart": (multipart(form_file(TONE_WAV)).replace(b"\r\n", b"\n"), {}),
    "file": (multipart(form_file(TONE_WAV, filename="")), {}),
    "fields": (multipart(form_file(TONE_WAV), form_field("a", "1"), form_field("a", "2")), {}),
    "response_format": (multipart(form_file(TONE_WAV), form_field("response_format", "text")), {}),
    "stream": (multipart(form_file(TONE_WAV), form_field("stream", "true")), {}),
    "numbers": (multipart(form_file(TONE_WAV), form_field("temperature", "1e40")), {}),
    "encoded": (gzip.compress(BODY), {"Content-Encoding": "gzip"}),
}


@pytest.mark.parametrize("why", sorted(BODY_ROWS))
def test_a_request_callo_cannot_translate_passes_through_untouched_and_is_counted(up, proxy, why):
    srv, cfg = proxy
    up.routes["/v1/audio/transcriptions"] = lambda u, rec, h: h._reply(400, {"native": True})
    body, extra = BODY_ROWS[why]
    status, headers, data, _ = post(srv, body, extra)
    rec = up.last("/v1/audio/transcriptions")
    assert (status, json.loads(data), headers["x-callo-slot"]) == (400, {"native": True}, "-")
    assert rec.body == body and rec.headers["content-type"] == CT
    assert _transcription_counters(cfg) == {f"transcription_unrouted_{why}": 1}
    assert not [r for r in up.requests if "/chat/" in r.path] and _idle(cfg.router)


def test_a_nul_in_language_passes_through_counted_as_fields(up, proxy):
    """Upstream's `%s` of language.c_str() stops at a NUL, so plan() says `fields` and the handler
    counts it there."""
    srv, cfg = proxy
    up.routes["/v1/audio/transcriptions"] = lambda u, rec, h: h._reply(200, {"native": True})
    body = multipart(form_file(TONE_WAV), form_field("language", "en\x00fr"))
    status, headers, _, _ = post(srv, body)
    assert (status, headers["x-callo-slot"]) == (200, "-") and up.last("/v1/audio/transcriptions").body == body
    assert _transcription_counters(cfg) == {"transcription_unrouted_fields": 1} and _idle(cfg.router)
    assert not [r for r in up.requests if "/chat/" in r.path]


@pytest.mark.parametrize("why, kw", [
    ("no_audio", {"transcription": None}),
    ("no_pinning", {"router": SlotRouter(4, 1, agent_pinning=False)}),
    ("render", {"transcription": Transcription(lfm2=False, off="a --no-jinja server")}),
    ("large", {"max_routed_body": 64}),
])
def test_the_pre_body_and_size_rows_pass_through_and_are_counted(up, why, kw):
    cfg = ProxyConfig(**{"upstream": up.url, "router": SlotRouter(4, 1), "transcription": Transcription(lfm2=False),
                         **kw})
    srv = _serve(cfg)
    try:
        up.routes["/v1/audio/transcriptions"] = lambda u, rec, h: h._reply(200, {"native": True})
        status, headers, data, _ = post(srv)
        assert (status, json.loads(data), headers["x-callo-slot"]) == (200, {"native": True}, "-")
        assert up.last("/v1/audio/transcriptions").body == BODY
        assert _transcription_counters(cfg) == {f"transcription_unrouted_{why}": 1}
        assert cfg.stats.counters["unrouted_large"] == (1 if why == "large" else 0)
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_chunked_transcription_still_gets_411_and_is_counted(up, proxy):
    srv, cfg = proxy
    out = raw(srv, b"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: x\r\nContent-Type: " + CT.encode()
              + b"\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n")
    assert out.startswith(b"HTTP/1.0 411") and _transcription_counters(cfg) == {"transcription_unrouted_chunked": 1}
    assert not [r for r in up.requests if r.path.startswith("/v1/")]


def test_a_translation_over_llama_servers_100_mib_cap_passes_through_as_large(up, proxy, monkeypatch):
    srv, cfg = proxy
    monkeypatch.setattr(transcribe, "UPSTREAM_MAX_BODY", 1000)          # BODY's JSON is ~171 kB
    up.routes["/v1/audio/transcriptions"] = lambda u, rec, h: h._reply(200, {"native": True})
    status, headers, _, _ = post(srv)
    assert status == 200 and headers["x-callo-slot"] == "-" and up.last("/v1/audio/transcriptions").body == BODY
    assert _transcription_counters(cfg) == {"transcription_unrouted_large": 1}
    assert cfg.stats.counters["unrouted_large"] == 1 and cfg.router.counters["lru"] == 0    # no slot was taken


# ---- disconnects and a broken upstream -----------------------------------------------------------------------------

@pytest.mark.parametrize("conversation", [None, "c1"])
def test_a_client_that_leaves_aborts_the_generation_and_frees_the_slot(up, proxy, conversation):
    """The watcher always runs, X-Conversation-Id or not; callo writes nothing and releases in `finally`. The client
    closes only once the upstream holds the request (an event, not a sleep), so a slow runner cannot close first."""
    srv, cfg = proxy
    seen = threading.Event()

    def slow(u, rec, h):
        seen.set()
        if not h.wait_or_abort(5.0):
            h._reply(200, REPLY)

    up.routes["/v1/chat/completions"] = slow
    head = b"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: x\r\nContent-Type: " + CT.encode() + b"\r\n"
    if conversation:
        head += b"X-Conversation-Id: " + conversation.encode() + b"\r\n"
    s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10)
    s.sendall(head + b"Content-Length: %d\r\n\r\n" % len(BODY) + BODY)
    assert seen.wait(10.0)
    s.close()
    assert _wait(lambda: up.aborted, timeout=10.0) and _idle(cfg.router)
    assert cfg.stats.counters["upstream_errors"] == 0 and _transcription_counters(cfg) == {}


def test_real_concurrent_transcriptions_overflow_once_go_home_and_a_disconnect_frees_its_slot(up, proxy):
    """Overlapping requests, the fake upstream held on an Event. The first holds the
    home (slot 1); the second, a raw client, takes the normal pick (slot 2, not deferred onto the busy home) and then
    leaves mid-request, which frees slot 2 while the home is still held; the first completes; the third goes home.
    Every slot ends idle and the counters add up."""
    srv, cfg = proxy
    gate = threading.Event()
    arrivals: list[int] = []
    arrived = threading.Condition()

    def held(u, rec, h):
        with arrived:
            arrivals.append(rec.json["id_slot"])
            arrived.notify_all()
        while not gate.is_set():
            if h.wait_or_abort(0.05):
                return
        _chat(u, rec, h)

    def arrived_n(n):
        with arrived:
            return arrived.wait_for(lambda: len(arrivals) >= n, timeout=10.0)

    up.routes["/v1/chat/completions"] = held
    first: list = []
    t = threading.Thread(target=lambda: first.append(post(srv)), daemon=True)
    t.start()
    try:
        assert arrived_n(1) and arrivals == [1] and cfg.router.inflight(1) == 1
        s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10)
        s.sendall(b"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: x\r\nContent-Type: " + CT.encode()
                  + b"\r\nContent-Length: %d\r\n\r\n" % len(BODY) + BODY)
        assert arrived_n(2) and arrivals == [1, 2] and cfg.router.inflight(2) == 1
        s.close()                                                   # the second client leaves mid-request
        assert _wait(lambda: up.aborted) and _wait(lambda: cfg.router.inflight(2) == 0)
        assert cfg.router.inflight(1) == 1 and not first            # the home is still held
    finally:
        gate.set()
        t.join(10.0)
    assert first and first[0][0] == 200 and first[0][1]["x-callo-slot"] == "1" and _idle(cfg.router)
    status, headers, _, _ = post_idle(srv, cfg)
    assert (status, headers["x-callo-slot"]) == (200, "1") and arrivals == [1, 2, 1]     # the third goes home
    assert _transcription_counters(cfg) == {"transcription_translated": 2}                # the leaver got nothing
    assert cfg.stats.counters["upstream_errors"] == 0
    rc = cfg.router.counters
    assert (rc["lru"], rc["hint"], rc["deferred"]) == (2, 1, 0) and cfg.router.snapshot()["conversations"] == 0
    assert cfg.router.snapshot()["sticky"]["callo:asr"]["home"] == 1
    assert all(cfg.router.inflight(slot) == 0 for slot in range(cfg.router.total_slots))


def test_an_upstream_that_dies_mid_body_is_a_502_callo_upstream(up, proxy):
    srv, cfg = proxy

    def half(u, rec, h):
        h.send_response(200)
        h.send_header("Content-Length", "1000")
        h.end_headers()
        h.wfile.write(b'{"choices": [')
        h.close_connection = True

    up.routes["/v1/chat/completions"] = half
    status, headers, data, _ = post(srv)
    assert status == 502 and json.loads(data)["error"]["type"] == "callo_upstream" and headers["x-callo-slot"] == "1"
    assert json.loads(data)["error"]["message"].startswith("upstream unreachable: ")     # passthrough's own 502
    assert cfg.stats.counters["upstream_errors"] == 1 and _transcription_counters(cfg) == {} and _idle(cfg.router)


def test_an_unreachable_upstream_is_a_502_and_frees_the_slot():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead = s.getsockname()[1]
    s.close()
    cfg = ProxyConfig(upstream=f"http://127.0.0.1:{dead}", router=SlotRouter(4, 1),
                      transcription=Transcription(lfm2=False))
    srv = _serve(cfg)
    try:
        status, _, data, _ = post(srv)
        assert status == 502 and json.loads(data)["error"]["type"] == "callo_upstream"
        assert cfg.stats.counters["upstream_errors"] == 1 and _idle(cfg.router)
    finally:
        srv.shutdown()
        srv.server_close()


def test_stats_publish_asr_preset_and_the_transcription_block():
    snap = Stats().snapshot(SlotRouter(4, 1), None)
    assert snap["asr_preset"] is None and snap["transcription"] is None       # serve sets both
    st = Stats()
    st.asr_preset, st.transcription = "lfm2", {"translate": False, "reason": "render: x"}
    snap = st.snapshot(SlotRouter(4, 1), None)
    assert snap["asr_preset"] == "lfm2" and snap["transcription"] == {"translate": False, "reason": "render: x",
                                                                      "asr_home": None, "asr_home_idle_s": 600.0}


def test_stats_show_the_asr_home_and_count_its_moves(up, proxy):
    """`transcription.asr_home` is callo:asr's slot (null while homeless), `transcription_rehomed`
    counts the homes chosen after the first (absent until first counted), and `router.conversations` counts client
    conversations only; the sticky hint is reported under `router.sticky`."""
    srv, cfg = proxy
    cfg.stats.transcription = {"translate": True, "reason": None}
    _, _, snap = get_stats(cfg)
    assert snap["transcription"]["asr_home"] is None and "transcription_rehomed" not in snap["counters"]
    assert post_idle(srv, cfg)[1]["x-callo-slot"] == "1"
    _, _, snap = get_stats(cfg)
    assert snap["transcription"]["asr_home"] == 1 and snap["router"]["conversations"] == 0
    assert snap["router"]["sticky"] == {"callo:asr": {"home": 1, "rehomed": 0}}
    cfg.router.acquire(2, "busy")
    cfg.router.acquire(3, "busy")
    assert post(srv, headers={"X-Callo-Conversation": "alice"})[1]["x-callo-slot"] == "1"   # alice claims the home
    assert _wait(lambda: cfg.router.inflight(1) == 0)
    cfg.router.release(2)
    cfg.router.release(3)
    _, _, snap = get_stats(cfg)
    assert snap["transcription"]["asr_home"] is None and snap["router"]["conversations"] == 1
    assert post_idle(srv, cfg)[1]["x-callo-slot"] == "2"
    _, _, snap = get_stats(cfg)
    assert snap["transcription"]["asr_home"] == 2 and snap["counters"]["transcription_rehomed"] == 1


def get_stats(cfg):
    return None, None, cfg.stats.snapshot(cfg.router, None)
