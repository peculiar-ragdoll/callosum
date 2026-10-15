"""The front door, against a fake llama-server. Every documented proxy case plus the concurrency
edge cases; v0.3: the native-model /v1/systemone handler and media chat routing."""
import gzip
import http.client
import json
import socket
import threading
import time

import pytest
from fake_upstream import FakeUpstream

import callo.proxy as proxy_module
from callo.proxy import HOP_BY_HOP, ProxyConfig, Stats, make_server, percentile, set_config
from callo.slots import SlotRouter

CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}


@pytest.fixture
def up():
    u = FakeUpstream().start()
    yield u
    u.stop()


def _serve(cfg):
    srv = make_server(cfg, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def proxy(up):
    cfg = ProxyConfig(upstream=up.url, router=SlotRouter(4, 1))
    srv = _serve(cfg)
    yield srv, cfg
    srv.shutdown()
    srv.server_close()


def port(srv):
    return srv.server_address[1]


def call(srv, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port(srv), timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    out = (r.status, {k.lower(): v for k, v in r.getheaders()}, data)
    c.close()
    return out


def raw(srv, payload: bytes, *, close_after: float | None = None, read_until_close: bool = True) -> bytes:
    s = socket.create_connection(("127.0.0.1", port(srv)), timeout=10)
    s.sendall(payload)
    if close_after is not None:
        time.sleep(close_after)
        s.close()
        return b""
    out = b""
    while read_until_close:
        chunk = s.recv(65536)
        if not chunk:
            break
        out += chunk
    s.close()
    return out


def _wait(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_unrouted_passthrough_is_byte_identical(up, proxy):
    srv, _ = proxy
    up.routes["/v1/embeddings"] = lambda u, rec, h: h._reply(200, b"\x00\x01 raw \xff bytes", {"Content-Type": "application/octet-stream"})
    body = b'{"input": "caf\xc3\xa9", "model": "x"}  \n'
    status, headers, data = call(srv, "POST", "/v1/embeddings", body,
                                 {"Content-Type": "application/json", "Authorization": "Bearer abc", "Connection": "keep-alive"})
    assert status == 200 and data == b"\x00\x01 raw \xff bytes" and headers["x-callo-slot"] == "-"
    rec = up.last("/v1/embeddings")
    assert rec.body == body and rec.headers["authorization"] == "Bearer abc"
    assert rec.headers["host"] == f"127.0.0.1:{up.port}" and "connection" not in rec.headers


def test_get_passthrough_and_options(up, proxy):
    srv, _ = proxy
    status, _, data = call(srv, "GET", "/props")
    assert status == 200 and json.loads(data)["total_slots"] == 4
    status, _, data = call(srv, "OPTIONS", "/v1/chat/completions", headers={"Origin": "http://x"})
    assert status == 200 and json.loads(data)["method"] == "OPTIONS"
    assert up.last("/v1/chat/completions").headers["origin"] == "http://x"


def test_routed_json_gains_exactly_id_slot(up, proxy):
    srv, _ = proxy
    status, headers, data = call(srv, "POST", "/v1/chat/completions", json.dumps(CHAT).encode(),
                                 {"Content-Type": "application/json"})
    assert status == 200 and headers["x-callo-slot"] == "1"
    rec = up.last("/v1/chat/completions")
    assert rec.json == {**CHAT, "id_slot": 1}
    assert rec.headers["content-length"] == str(len(rec.body))


def test_client_id_slot_respected_and_reserved_rerouted(up, proxy):
    srv, cfg = proxy
    _, headers, _ = call(srv, "POST", "/completion", json.dumps({"prompt": "x", "id_slot": 3}).encode())
    assert headers["x-callo-slot"] == "3" and up.last("/completion").json["id_slot"] == 3
    _, headers, _ = call(srv, "POST", "/completion", json.dumps({"prompt": "x", "id_slot": 0}).encode())
    assert headers["x-callo-slot"] != "0" and cfg.router.counters["rerouted"] == 1


def test_sse_relayed_chunk_by_chunk_with_timing_and_pings(up, proxy):
    srv, _ = proxy
    up.stream_gap_s = 0.3
    body = json.dumps({**CHAT, "stream": True}).encode()
    s = socket.create_connection(("127.0.0.1", port(srv)), timeout=10)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
    buf, events, t = b"", [], []
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
        if b"\r\n\r\n" in buf and not t:
            head, buf = buf.split(b"\r\n\r\n", 1)
            assert head.startswith(b"HTTP/1.0 200") and b"transfer-encoding" not in head.lower()
            assert b"text/event-stream" in head.lower()
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            events.append(ev + b"\n\n")
            t.append(time.monotonic())
    s.close()
    assert b"".join(events) == b"".join(up.stream_chunks)
    assert events[1] == b": ping\n\n"
    assert all(b - a >= 0.2 for a, b in zip(t, t[1:], strict=False))   # chunks arrive as the upstream sends them, not at the end


def test_upstream_error_status_is_relayed(up, proxy):
    srv, _ = proxy
    up.routes["/v1/chat/completions"] = lambda u, rec, h: h._reply(400, {"error": {"message": "bad", "code": 400}})
    status, _, data = call(srv, "POST", "/v1/chat/completions", json.dumps(CHAT).encode())
    assert status == 400 and json.loads(data)["error"]["message"] == "bad"


def test_chunked_request_body_gets_411(up, proxy):
    srv, _ = proxy
    out = raw(srv, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n")
    assert out.startswith(b"HTTP/1.0 411") and b"X-Callo-Slot: -" in out
    assert not [r for r in up.requests if r.path == "/v1/chat/completions"]


def _idle(router):
    return _wait(lambda: all(router.inflight(s) == 0 for s in range(router.total_slots)))


def test_disconnect_before_first_byte_closes_upstream_within_1s(up, proxy):
    srv, cfg = proxy
    up.routes["/v1/chat/completions"] = lambda u, rec, h: None if h.wait_or_abort(5.0) else h._reply(200, {"late": True})
    body = json.dumps(CHAT).encode()
    t0 = time.monotonic()
    raw(srv, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body),
        close_after=0.1)
    assert _wait(lambda: up.aborted, timeout=2.0)
    assert up.aborted[0][1] - (t0 + 0.1) < 1.0
    assert _idle(cfg.router)                  # the slot was released, not leaked


def test_disconnect_mid_stream_closes_upstream(up, proxy):
    # after the first byte, a failed write to the client must close the upstream too.
    srv, cfg = proxy
    up.stream_chunks = [b"data: %d\n\n" % i for i in range(40)]
    up.stream_gap_s = 0.1
    body = json.dumps({**CHAT, "stream": True}).encode()
    s = socket.create_connection(("127.0.0.1", port(srv)), timeout=10)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
    assert s.recv(65536)                      # headers + first chunk
    t_close = time.monotonic()
    s.close()
    assert _wait(lambda: up.aborted, timeout=3.0)
    assert up.aborted[0][1] - t_close < 1.0
    assert _idle(cfg.router)


def test_x_conversation_id_requests_are_not_cancelled_on_disconnect(up, proxy):
    # Resumable streams on master need DELETE /v1/stream; callo must not trip the abort for them.
    srv, _ = proxy
    up.routes["/v1/chat/completions"] = lambda u, rec, h: h._reply(200, {"done": True}) if not h.wait_or_abort(0.6) else None
    body = json.dumps(CHAT).encode()
    raw(srv, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nX-Conversation-Id: c1\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body),
        close_after=0.05)
    time.sleep(0.9)
    assert up.aborted == [] and up.last("/v1/chat/completions").headers["x-conversation-id"] == "c1"


def test_expect_100_continue_is_answered_immediately(up, proxy):
    # curl sends Expect: 100-continue for bodies > 1 KiB and waits 1 s unless answered.
    srv, _ = proxy
    body = json.dumps({**CHAT, "messages": [{"role": "user", "content": "x" * 2000}]}).encode()
    s = socket.create_connection(("127.0.0.1", port(srv)), timeout=10)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\nContent-Length: %d\r\n\r\n" % len(body))
    t0 = time.monotonic()
    first = s.recv(1024)
    assert first.startswith(b"HTTP/1.1 100 Continue\r\n\r\n") and time.monotonic() - t0 < 0.5
    s.sendall(body)
    rest = b""
    while True:
        c = s.recv(65536)
        if not c:
            break
        rest += c
    s.close()
    assert rest.startswith(b"HTTP/1.0 200")
    rec = up.last("/v1/chat/completions")
    assert rec.json["id_slot"] == 1 and len(rec.json["messages"][0]["content"]) == 2000
    assert "expect" not in rec.headers


def test_watcher_ignores_pipelined_bytes(up, proxy):
    # bytes queued on the client socket are not EOF. They must arrive AFTER the handler parsed the
    # request: sent in the same sendall, the handler's buffered rfile swallows them and the watcher never sees them.
    srv, _ = proxy
    up.routes["/v1/chat/completions"] = lambda u, rec, h: h._reply(200, {"slow": True}) if not h.wait_or_abort(0.6) else None
    body = json.dumps(CHAT).encode()
    s = socket.create_connection(("127.0.0.1", port(srv)), timeout=10)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
    assert _wait(lambda: any(r.path == "/v1/chat/completions" for r in up.requests))   # upstream is now working
    time.sleep(0.1)
    s.sendall(b"GET /health HTTP/1.0\r\n\r\n")                                    # pipelined, while pending
    out = b""
    while True:
        try:
            c = s.recv(65536)
        except ConnectionResetError:   # the proxy closes with the pipelined request unread; the reply came first
            break
        if not c:
            break
        out += c
    s.close()
    assert out.startswith(b"HTTP/1.0 200") and b'"slow": true' in out
    assert up.aborted == []


def test_content_encoding_body_on_routed_endpoint_is_unpinned(up, proxy):
    srv, cfg = proxy
    gz = gzip.compress(json.dumps(CHAT).encode())
    _, headers, _ = call(srv, "POST", "/v1/chat/completions", gz, {"Content-Encoding": "gzip"})
    assert headers["x-callo-slot"] == "-" and up.last("/v1/chat/completions").body == gz
    _, headers, _ = call(srv, "POST", "/v1/chat/completions", b"not json at all")
    assert headers["x-callo-slot"] == "-" and cfg.stats.counters["unrouted_body"] == 2


def test_large_body_streams_through_unrouted(up):
    cfg = ProxyConfig(upstream=up.url, router=SlotRouter(4, 1), max_routed_body=4096)
    srv = _serve(cfg)
    try:
        body = json.dumps({**CHAT, "messages": [{"role": "user", "content": "y" * 8000}]}).encode()
        status, headers, data = call(srv, "POST", "/v1/chat/completions", body)
        assert status == 200 and headers["x-callo-slot"] == "-"
        assert json.loads(data)["raw_len"] == len(body) and cfg.stats.counters["unrouted_large"] == 1
        assert "id_slot" not in up.last("/v1/chat/completions").json
    finally:
        srv.shutdown()
        srv.server_close()


def test_hop_by_hop_headers_are_stripped_from_the_response(up, proxy):
    srv, _ = proxy
    up.routes["/x"] = lambda u, rec, h: h._reply(200, {"ok": 1}, {"Keep-Alive": "timeout=5", "X-Custom": "1", "Upgrade": "h2c"})
    _, headers, _ = call(srv, "GET", "/x")
    assert headers["x-custom"] == "1" and not (set(headers) & HOP_BY_HOP - {"connection"})
    assert headers.get("connection", "close").lower() in ("close",)


def test_chunked_upstream_response_is_dechunked_and_delimited_by_close(up, proxy):
    srv, _ = proxy
    up.routes["/x"] = lambda u, rec, h: h._reply(200, {"ok": 1}, chunked=True)
    out = raw(srv, b"GET /x HTTP/1.1\r\nHost: x\r\n\r\n")
    head, body = out.split(b"\r\n\r\n", 1)
    assert b"transfer-encoding" not in head.lower() and b"content-length" not in head.lower()
    assert json.loads(body) == {"ok": 1}


def test_upstream_down_gives_502_and_health_503():
    cfg = ProxyConfig(upstream="http://127.0.0.1:1", router=SlotRouter(4, 1), connect_timeout=2)
    srv = _serve(cfg)
    try:
        status, headers, data = call(srv, "POST", "/v1/chat/completions", json.dumps(CHAT).encode())
        assert status == 502 and json.loads(data)["error"]["type"] == "callo_upstream"
        assert headers["x-callo-slot"] == "-" and _idle(cfg.router)
        status, _, data = call(srv, "GET", "/health")
        assert status == 503 and json.loads(data)["status"] == "error"
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize("stream", [False, True])
def test_passthrough_reads_have_no_timeout(up, stream):
    """A long generation must relay however long it takes (llama-server has no cap): only the CONNECT is bounded.
    A non-streaming reply arrives as one late read; a stream can go quiet between chunks for as long as it likes."""
    if stream:
        up.stream_gap_s = 0.6
    else:
        def slow(u, rec, h):
            time.sleep(0.8)
            h._reply(200, {"done": True})
        up.routes["/v1/chat/completions"] = slow
    cfg = ProxyConfig(upstream=up.url, router=SlotRouter(4, 1), connect_timeout=0.2)
    assert ProxyConfig(upstream=up.url, router=SlotRouter(4, 1)).connect_timeout > 0   # the connect stays bounded
    srv = _serve(cfg)
    try:
        body = json.dumps({**CHAT, "stream": stream}).encode()
        status, headers, data = call(srv, "POST", "/v1/chat/completions", body)
        assert status == 200 and headers["x-callo-slot"] == "1"
        assert (b"data: [DONE]" in data) if stream else json.loads(data) == {"done": True}
        assert cfg.stats.counters["upstream_errors"] == 0 and cfg.stats.counters["relay_aborted"] == 0
        assert up.aborted == [] and _idle(cfg.router)
    finally:
        srv.shutdown()
        srv.server_close()


def test_health_mirrors_upstream(up, proxy):
    srv, _ = proxy
    status, _, data = call(srv, "GET", "/health")
    assert status == 200 and json.loads(data) == {"status": "ok"}
    up.routes["/health"] = lambda u, rec, h: h._reply(503, {"error": {"message": "Loading model"}})
    status, _, _ = call(srv, "GET", "/health")
    assert status == 503


def test_stats_endpoint_shape(up, proxy):
    srv, cfg = proxy
    cfg.stats.add_s1(12.0, 0.9)
    cfg.stats.add_s1(30.0, 0.2)
    call(srv, "POST", "/v1/chat/completions", json.dumps(CHAT).encode())
    status, _, data = call(srv, "GET", "/callo/stats")
    s = json.loads(data)
    assert status == 200 and set(s) >= {"uptime_s", "routes", "counters", "router", "s1", "probe", "slots", "temperature"}
    assert s["routes"]["/v1/chat/completions"] == 1 and s["router"]["reserved"] == [0]
    assert {k: s["s1"][k] for k in ("n", "p50_ms", "p95_ms", "low_mass")} == \
        {"n": 2, "p50_ms": 12.0, "p95_ms": 30.0, "low_mass": 1}               # v0.2 adds image/context keys
    assert {"vision", "media_marker", "user_delimiter", "cache_prompt"} <= set(s)
    assert s["slots"] is None                       # no slots_snapshot wired in this test


def test_stats_report_the_effective_s1_temperature(up, proxy):
    srv, cfg = proxy
    assert json.loads(call(srv, "GET", "/callo/stats")[2])["temperature"] is None   # no engine wired: nothing to report
    cfg.stats.temperature = 1.91
    assert json.loads(call(srv, "GET", "/callo/stats")[2])["temperature"] == 1.91


def test_server_and_date_headers_are_not_duplicated(up, proxy):
    srv, _ = proxy
    head = raw(srv, b"GET /props HTTP/1.0\r\n\r\n").split(b"\r\n\r\n", 1)[0].lower()
    assert head.count(b"\r\nserver:") == 1 and head.count(b"\r\ndate:") == 1   # callo's own, not also the upstream's


def test_native_systemone_passes_through_when_no_engine(up, proxy):
    srv, _ = proxy
    up.routes["/v1/systemone"] = lambda u, rec, h: h._reply(501, {"error": {"message": "This model is not a decision model"}})
    status, headers, data = call(srv, "POST", "/v1/systemone", b'{"state": "x"}')
    assert status == 501 and "not a decision model" in data.decode() and headers["x-callo-slot"] == "-"


def test_percentile_is_nearest_rank():
    assert percentile([3, 1, 2], 0.5) == 2 and percentile([1, 2, 3, 4], 0.95) == 4 and percentile([7], 0.5) == 7


def test_stats_percentiles_and_low_mass():
    s = Stats()
    for ms, mass in ((5, 0.9), (50, 0.4), (10, 0.95)):
        s.add_s1(ms, mass)
    snap = s.snapshot(SlotRouter(4, 1), None)
    assert {k: snap["s1"][k] for k in ("n", "p50_ms", "p95_ms", "low_mass")} == \
        {"n": 3, "p50_ms": 10, "p95_ms": 50, "low_mass": 1}
    assert snap["s1"]["image_requests"] == 0 and snap["s1"]["image_p50_ms"] is None
    assert snap["s1"]["readouts"] == 3                  # add_s1 bumps the monotonic count too, so n <= readouts


def test_s1_n_never_exceeds_the_monotonic_readout_count():
    s = Stats()
    for i in range(2100):                               # past the 2,000-wide latency window
        s.add_s1(float(i), 0.9)
    snap = s.snapshot(SlotRouter(4, 1), None)["s1"]
    assert snap["n"] == 2000 and snap["readouts"] == 2100


def _unpinned_proxy(up):
    cfg = ProxyConfig(upstream=up.url, router=SlotRouter(4, 1, agent_pinning=False))
    return _serve(cfg), cfg


def test_unpinned_strips_client_id_slot_that_lands_on_a_reserved_slot(up):
    srv, cfg = _unpinned_proxy(up)
    try:
        for v in (0, 4):                                   # 4 % 4 == 0: the S1 slot after llama-server's own wrap
            _, headers, _ = call(srv, "POST", "/completion", json.dumps({"prompt": "x", "id_slot": v}).encode())
            assert headers["x-callo-slot"] == "-" and up.last("/completion").json == {"prompt": "x"}
        assert cfg.stats.counters["id_slot_rerouted"] == 2
    finally:
        srv.shutdown()
        srv.server_close()


def test_unpinned_leaves_other_bodies_untouched(up):
    srv, cfg = _unpinned_proxy(up)
    try:
        body = b'{"prompt": "x",  "id_slot": 2}'            # agent range: byte-identical, not re-serialised
        call(srv, "POST", "/completion", body)
        assert up.last("/completion").body == body
        body = b'{"prompt": "x", "id_slot": -1}'            # absent by normalise_id_slot
        call(srv, "POST", "/completion", body)
        assert up.last("/completion").body == body
        assert cfg.stats.counters["id_slot_rerouted"] == 0
    finally:
        srv.shutdown()
        srv.server_close()


def _post_raw(srv, head: bytes, body: bytes = b"") -> bytes:
    return raw(srv, b"POST " + head + b"\r\n\r\n" + body)


def test_negative_or_garbage_content_length_gets_400_and_acquires_nothing(up):
    cfg = ProxyConfig(upstream=up.url, router=SlotRouter(4, 1), max_routed_body=1024)
    srv = _serve(cfg)
    try:
        for cl in (b"-1", b"abc"):
            for path in (b"/v1/chat/completions", b"/v1/embeddings", b"/v1/systemone"):
                out = _post_raw(srv, path + b" HTTP/1.1\r\nHost: x\r\nContent-Length: " + cl, b"y" * 8000)
                assert out.startswith(b"HTTP/1.0 400") and b"X-Callo-Slot: -" in out
        assert up.requests == []
        assert _idle(cfg.router) and not cfg.router.counters
    finally:
        srv.shutdown()
        srv.server_close()


def test_systemone_body_over_the_cap_gets_413(up):
    cfg = ProxyConfig(upstream=up.url, router=SlotRouter(4, 1), max_routed_body=1024,
                      systemone=lambda obj, headers: (200, {"ok": True}, {"X-Callo-Slot": "0"}))
    srv = _serve(cfg)
    try:
        out = _post_raw(srv, b"/v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: 2000000", b"")
        assert out.startswith(b"HTTP/1.0 413") and b'"detail"' in out and b"X-Callo-Slot: -" in out
        status, _, _ = call(srv, "POST", "/v1/systemone", b'{"state": "x"}')
        assert status == 200
    finally:
        srv.shutdown()
        srv.server_close()


NESTED = b"[" * 100_000 + b"]" * 100_000   # ~200 KB, under the 64 MiB cap


def _force_recursionerror(monkeypatch):
    # json.loads raises RecursionError on a deeply nested body on CPython 3.10-3.13, but 3.14's parser handles that
    # depth without raising (the body just parses). I-1 is about the HANDLER never dropping the connection when the
    # parse DOES raise RecursionError, so force that branch deterministically across versions. Raise only for the
    # NESTED probe (a long run of '['), and delegate every other json.loads to the real one — `callo.proxy.json` is
    # the shared stdlib module, so a blanket patch would also break the test's own json.loads of the 422 body.
    real = json.loads

    def maybe_raise(s, *a, **k):
        body = s.encode() if isinstance(s, str) else s
        if isinstance(body, (bytes, bytearray)) and body[:10] == b"[" * 10:
            raise RecursionError("forced deep-nesting")
        return real(s, *a, **k)
    monkeypatch.setattr("callo.proxy.json.loads", maybe_raise)


def test_a_deeply_nested_systemone_body_gets_a_422_not_a_dropped_connection(up, monkeypatch):
    # I-1: a json.loads RecursionError (not a ValueError) must still produce a 422, never a dropped connection.
    _force_recursionerror(monkeypatch)
    cfg = ProxyConfig(upstream=up.url, router=SlotRouter(4, 1),
                      systemone=lambda obj, headers: (200, {"ok": True}, {"X-Callo-Slot": "0"}))
    srv = _serve(cfg)
    try:
        status, headers, data = call(srv, "POST", "/v1/systemone", NESTED, {"Content-Type": "application/json"})
        assert status == 422 and json.loads(data) == {"detail": "body is not JSON"}
        assert headers["x-callo-slot"] == "-"
        assert not [r for r in up.requests if r.path == "/v1/systemone"]   # never reached the upstream or the engine
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_deeply_nested_routed_body_passes_through_unrouted_not_dropped(up, proxy, monkeypatch):
    # I-1 on a PINNABLE route: a body whose parse raises RecursionError is forwarded byte-for-byte, as non-JSON is.
    _force_recursionerror(monkeypatch)
    srv, cfg = proxy
    up.routes["/v1/chat/completions"] = lambda u, rec, h: h._reply(200, {"ok": 1})   # body-agnostic
    status, headers, _ = call(srv, "POST", "/v1/chat/completions", NESTED, {"Content-Type": "application/json"})
    assert status == 200 and headers["x-callo-slot"] == "-"
    assert up.last("/v1/chat/completions").body == NESTED    # byte-identical: no id_slot could be added
    assert cfg.stats.counters["unrouted_body"] == 1


def test_100_continue_is_not_sent_to_http10_clients(up, proxy):
    srv, _ = proxy
    body = json.dumps(CHAT).encode()
    out = raw(srv, b"POST /v1/chat/completions HTTP/1.0\r\nExpect: 100-continue\r\nContent-Length: %d\r\n\r\n%s"
              % (len(body), body))
    assert out.startswith(b"HTTP/1.0 200") and b"100 Continue" not in out


def test_systemone_over_cap_body_is_drained_so_the_413_reaches_the_client(up):
    # M-11: a client mid-upload of an over-cap body must read the 413, not a connection reset. The body is drained
    # (bounded) before the socket closes, so the reply is not lost.
    cfg = ProxyConfig(upstream=up.url, router=SlotRouter(4, 1), max_routed_body=1024,
                      systemone=lambda obj, headers: (200, {"ok": True}, {"X-Callo-Slot": "0"}))
    srv = _serve(cfg)
    try:
        body = b"x" * 5000                        # a real over-cap upload, actually sent (not just a declared length)
        out = _post_raw(srv, b"/v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: %d" % len(body), body)
        assert out.startswith(b"HTTP/1.0 413") and b'"detail"' in out and b"X-Callo-Slot: -" in out
        assert cfg.stats.counters["unrouted_large"] == 1
    finally:
        srv.shutdown()
        srv.server_close()


LOADING = {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}   # server-http.cpp:307-326


def test_a_server_without_config_answers_every_request_with_llama_servers_loading_503(up):
    srv = _serve(None)
    try:
        for method, path, body in (("GET", "/health", None), ("GET", "/callo/stats", None),
                                   ("POST", "/v1/chat/completions", json.dumps(CHAT).encode()),
                                   ("POST", "/v1/systemone", b'{"state": "s"}'), ("GET", "/slots", None)):
            t0 = time.monotonic()
            c = http.client.HTTPConnection("127.0.0.1", port(srv), timeout=10)
            c.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
            r = c.getresponse()
            assert r.status == 503 and json.loads(r.read()) == LOADING and r.getheader("X-Callo-Slot") == "-"
            assert r.getheader("Content-Type") == "application/json; charset=utf-8"
            assert time.monotonic() - t0 < 2
            c.close()
        assert up.requests == []                                     # nothing reached the upstream while loading
        set_config(srv, ProxyConfig(upstream=up.url, router=SlotRouter(4, 1)))
        c = http.client.HTTPConnection("127.0.0.1", port(srv), timeout=10)
        c.request("GET", "/health")
        r = c.getresponse()
        assert r.status == 200 and json.loads(r.read()) == {"status": "ok"}
        c.close()
    finally:
        srv.shutdown()
        srv.server_close()


def test_stats_snapshot_has_the_v03_keys_even_before_any_request():
    snap = Stats().snapshot(SlotRouter(4, 1), None)
    assert {"instance", "model_key", "model_key_error", "model_file", "provenance", "position_bias"} <= set(snap)
    assert len(snap["instance"]) == 32 and Stats().instance != snap["instance"]       # one id per startup
    assert snap["s1"]["readouts"] == 0 and snap["s1"]["topk_miss"] == 0


# ---- v0.3: /v1/systemone for a native decision model ---------------------------------------------------------

WAV_URL = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IA=="


def _native(up):
    up.routes["/v1/systemone"] = lambda u, rec, h: h._reply(200, {"native": True})


@pytest.mark.parametrize("body", [
    {"state": "x", "audio": [WAV_URL]},
    {"state": "x", "videos": "not a list"},
    {"state": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA"}}]}]},
    {"state": {"messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "x"}}]}]}},
    {"state": [{"role": "user", "content": [{"type": "input_video", "input_video": {"data": "AAAA"}}]}]},
    {"state": [{"role": "user", "content": [{"type": "audio_url"}]}]},
    {"state": [{"role": "user", "content": [{"type": "video"}]}]},
])
def test_native_systemone_refuses_audio_and_video_with_422_and_sends_nothing(up, proxy, body):
    srv, cfg = proxy
    _native(up)
    status, headers, data = call(srv, "POST", "/v1/systemone", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    detail = json.loads(data)["detail"]
    assert status == 422 and detail.startswith("the upstream is a native decision model, which takes images only")
    assert headers["x-callo-slot"] == "-" and not [r for r in up.requests if r.path == "/v1/systemone"]
    assert cfg.stats.counters["native_media_refused"] == 1


NATIVE_NESTED = b"[" * 100_000 + b"]" * 100_000


@pytest.mark.parametrize("body", [
    b'{"state": "x", "audio": [], "videos": null, "images": ["data:image/png;base64,QUJD"]}',   # empty means none
    b'{"state":  "x",  "questions": {}}  \n',          # spacing kept: the ORIGINAL bytes go upstream
    b"not json at all \xff",                             # not JSON, not even UTF-8
    b'{"state": [{"role": "user", "content": [{"type": ["input_audio"]}, {"type": {"a": 1}}]}]}',   # no media (I-1)
    NATIVE_NESTED,                                       # json.loads raises RecursionError
])
def test_native_systemone_forwards_anything_else_as_the_original_bytes(up, proxy, body):
    srv, cfg = proxy
    _native(up)
    sent = {"Content-Type": "application/json", "Authorization": "Bearer k", "X-Callo-Conversation": "lane-a"}
    status, headers, data = call(srv, "POST", "/v1/systemone", body, sent)
    assert status == 200 and json.loads(data) == {"native": True} and headers["x-callo-slot"] == "-"
    assert up.last("/v1/systemone").body == body
    got = up.last("/v1/systemone").headers                     # the client's headers are forwarded too
    assert {k.lower(): v for k, v in sent.items()}.items() <= got.items()
    assert cfg.stats.counters["native_unscanned"] == 0 and cfg.stats.counters["native_media_refused"] == 0


def test_native_systemone_passes_an_oversized_or_encoded_body_through_unscanned_and_counts_it(up):
    cfg = ProxyConfig(upstream=up.url, router=SlotRouter(4, 1), max_routed_body=4096)
    srv = _serve(cfg)
    try:
        _native(up)
        big = json.dumps({"state": "x", "audio": [WAV_URL + "A" * 8000]}).encode()
        status, _, _ = call(srv, "POST", "/v1/systemone", big)
        assert status == 200 and up.last("/v1/systemone").body == big                    # streamed through
        gz = gzip.compress(json.dumps({"state": "x", "audio": [WAV_URL]}).encode())
        status, _, _ = call(srv, "POST", "/v1/systemone", gz, {"Content-Encoding": "gzip"})
        assert status == 200 and up.last("/v1/systemone").body == gz
        assert cfg.stats.counters["native_unscanned"] == 2 and cfg.stats.counters["native_media_refused"] == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_native_systemone_still_answers_a_chunked_body_with_411(up, proxy):
    srv, _ = proxy
    out = raw(srv, b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n")
    assert out.startswith(b"HTTP/1.0 411") and not [r for r in up.requests if r.path == "/v1/systemone"]


# ---- v0.3: media chat is pinned, routed on digests, and forwarded intact ---------------------------------------

@pytest.mark.parametrize("part", [
    {"type": "input_audio", "input_audio": {"data": "U" * 300_000, "format": "wav"}},
    {"type": "input_video", "input_video": {"data": "V" * 300_000}},
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + "W" * 300_000}},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "I" * 300_000}},
])
def test_a_media_chat_body_gets_id_slot_and_reaches_the_upstream_with_its_real_payload(up, proxy, part):
    srv, cfg = proxy
    body = {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, part]}]}
    status, headers, _ = call(srv, "POST", "/v1/chat/completions", json.dumps(body).encode(),
                              {"Content-Type": "application/json"})
    assert status == 200 and headers["x-callo-slot"] == "1"
    assert up.last("/v1/chat/completions").json == {**body, "id_slot": 1}          # no digest leaked upstream
    assert cfg.router.snapshot()["text_chars"]["1"] < 1024                          # the router holds the digest


def test_a_body_too_deep_to_walk_is_routed_unpinned_not_dropped(up, proxy):
    """A real body: on Python 3.12+ json.loads accepts 3,000 levels and the digest walk recurses past the limit; on
    3.10/3.11 json.loads refuses it. Either way it is forwarded byte-identical and unpinned."""
    srv, cfg = proxy
    up.routes["/v1/chat/completions"] = lambda u, rec, h: h._reply(200, {"ok": 1})   # body-agnostic
    body = b'{"model":"m","messages":[{"role":"user","content":' + b"[" * 3000 + b"]" * 3000 + b"}]}"
    status, headers, _ = call(srv, "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    assert status == 200 and headers["x-callo-slot"] == "-" and up.last("/v1/chat/completions").body == body
    assert cfg.stats.counters["unrouted_body"] == 1 and _idle(cfg.router)


@pytest.mark.parametrize(("client_slot", "forwarded"), [(0, None), (2, 2)])
def test_a_walk_failure_still_strips_a_reserved_client_id_slot(up, proxy, monkeypatch, client_slot, forwarded):
    """The unpinned fallback keeps v0.2's rule that an agent can never pin onto the S1 slot."""
    def deep(path, body):
        raise RecursionError("maximum recursion depth exceeded")

    srv, cfg = proxy
    monkeypatch.setattr(proxy_module, "canonical_text", deep)
    status, headers, _ = call(srv, "POST", "/v1/chat/completions", json.dumps({**CHAT, "id_slot": client_slot}).encode(),
                              {"Content-Type": "application/json"})
    assert status == 200 and headers["x-callo-slot"] == "-"
    assert up.last("/v1/chat/completions").json.get("id_slot") == forwarded
    assert cfg.stats.counters["id_slot_rerouted"] == (forwarded is None) and _idle(cfg.router)


def test_a_body_too_deep_to_serialise_after_the_strip_is_refused_not_forwarded_to_the_s1_slot(up, proxy, monkeypatch):
    class Shallow:                          # json as proxy.py sees it: re-serialising the request body recurses
        loads = staticmethod(json.loads)

        @staticmethod
        def dumps(obj, **kw):
            if isinstance(obj, dict) and "messages" in obj:
                raise RecursionError("maximum recursion depth exceeded")
            return json.dumps(obj, **kw)

    def deep(path, body):
        raise RecursionError("maximum recursion depth exceeded")

    srv, cfg = proxy
    monkeypatch.setattr(proxy_module, "canonical_text", deep)
    monkeypatch.setattr(proxy_module, "json", Shallow)
    status, headers, _ = call(srv, "POST", "/v1/chat/completions", json.dumps({**CHAT, "id_slot": 0}).encode())
    assert status == 400 and headers["x-callo-slot"] == "-"
    assert not [r for r in up.requests if r.path == "/v1/chat/completions"]
    # the refusal is counted as one; nothing it never did (forward unrouted, reroute an id_slot) is
    c = cfg.stats.counters
    assert (c["unrouted_refused"], c["unrouted_body"], c["id_slot_rerouted"]) == (1, 0, 0)


LONE = b'{"model":"m","messages":[{"role":"user","content":"half a pair: \\ud800!"}]'


@pytest.mark.parametrize("walk_fails", [False, True])
def test_a_lone_surrogate_body_is_forwarded_not_a_dropped_connection(up, proxy, monkeypatch, walk_fails):
    """The string "\\ud800" is valid JSON but cannot be UTF-8 encoded; re-serialising falls back to ASCII
    escapes instead of raising, on the pinned path and on the unpinned strip path."""
    srv, cfg = proxy
    if walk_fails:
        def deep(path, body):
            raise RecursionError("maximum recursion depth exceeded")
        monkeypatch.setattr(proxy_module, "canonical_text", deep)
    body = LONE + (b',"id_slot":0}' if walk_fails else b"}")
    status, headers, _ = call(srv, "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    assert status == 200 and headers["x-callo-slot"] == ("-" if walk_fails else "1")
    got = up.last("/v1/chat/completions")
    assert b"\\ud800" in got.body and got.json["messages"][0]["content"] == "half a pair: \ud800!"
    assert got.json.get("id_slot") == (None if walk_fails else 1)
    assert cfg.stats.counters["id_slot_rerouted"] == int(walk_fails) and _idle(cfg.router)


def test_responses_and_completion_media_bodies_are_routed_and_forwarded_unchanged(up, proxy):
    """The digests live in the router's copy only; the upstream sees the client's body plus
    id_slot."""
    srv, cfg = proxy
    responses = {"model": "m", "instructions": "be brief", "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64," + "I" * 300_000},
        {"type": "input_text", "text": "what is this?"}]}]}
    completion = {"prompt": {"prompt_string": "<__media__> describe", "multimodal_data": ["A" * 300_000]},
                  "n_predict": 1}
    for path, body in (("/v1/responses", responses), ("/completion", completion)):
        up.routes[path] = lambda u, rec, h: h._reply(200, {"ok": 1})
        status, headers, _ = call(srv, "POST", path, json.dumps(body).encode(),
                                  {"Content-Type": "application/json"})
        slot = headers["x-callo-slot"]
        assert status == 200 and slot in ("1", "2", "3")                 # pinned (the second lands by LRU)
        assert up.last(path).json == {**body, "id_slot": int(slot)}
        assert cfg.router.snapshot()["text_chars"][slot] < 1024


def test_a_chat_body_with_a_schema_typed_part_routes_as_in_v02(up, proxy):
    """A dict whose "type" is a list (a JSON-schema fragment) used to crash the digest walk's set
    lookup and drop the connection; it is no media part, so the body is pinned as v0.2 pinned it."""
    srv, _ = proxy
    body = {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "fill the form"}, {"type": "json", "json": {"type": ["string", "null"]}}]}]}
    status, headers, _ = call(srv, "POST", "/v1/chat/completions", json.dumps(body).encode(),
                              {"Content-Type": "application/json"})
    assert status == 200 and headers["x-callo-slot"] == "1"
    assert up.last("/v1/chat/completions").json == {**body, "id_slot": 1}
