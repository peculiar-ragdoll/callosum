"""Routing is pure logic over the router's own accounting: every documented routing case, no sockets."""
import copy
import hashlib
import logging
import threading
import time
from collections import OrderedDict

import pytest

from callo.errors import InvalidInput
from callo.slots import DIGEST_CHARS, PINNABLE, SlotRouter, _located, canonical_text, conversation_hint


def test_pinnable_endpoints_are_the_spec_list():
    assert PINNABLE == {"/completion", "/completions", "/v1/completions", "/chat/completions",
                        "/v1/chat/completions", "/infill", "/v1/responses", "/responses"}
    assert "/v1/messages" not in PINNABLE and "/v1/embeddings" not in PINNABLE


def test_reserved_range_is_the_lowest_ids():
    r = SlotRouter(4, 1)
    assert list(r.reserved) == [0] and r.agent == [1, 2, 3] and r.pinning
    r2 = SlotRouter(4, 2)
    assert list(r2.reserved) == [0, 1] and r2.agent == [2, 3]


def test_k_bounds():
    with pytest.raises(InvalidInput):
        SlotRouter(4, 4)
    with pytest.raises(InvalidInput):
        SlotRouter(4, 0)
    SlotRouter(4, 3)  # k < total_slots is fine


def test_fewer_than_two_slots_turns_all_pinning_off():
    r = SlotRouter(1, 1)
    assert not r.pinning and list(r.reserved) == [] and r.agent == []
    assert r.choose_agent("hello").slot is None and r.choose_agent("hello").how == "unpinned"
    assert r.choose_s1() == (None, False)


def test_probe_result_disables_agent_pinning_but_not_s1():
    r = SlotRouter(4, 1, agent_pinning=False)
    d = r.choose_agent("hello")
    assert d.slot is None and d.how == "unpinned"
    assert r.choose_s1() == (0, False)


def test_lru_fallback_on_fresh_router_is_lowest_id_first():
    r = SlotRouter(4, 1)
    d1 = r.choose_agent("alpha")
    assert (d1.slot, d1.how) == (1, "lru")
    r.acquire(1, "alpha")
    r.release(1)
    d2 = r.choose_agent("beta")
    assert (d2.slot, d2.how) == (2, "lru")              # slot 1 was used; 2 and 3 never -> 2
    r.acquire(2, "beta")
    r.release(2)
    r.acquire(3, "gamma")
    r.release(3)
    assert r.choose_agent("delta").slot == 1             # least recently used now


def test_lcp_choice_and_the_ten_percent_threshold():
    r = SlotRouter(4, 1)
    r.acquire(2, "system: you are helpful\nuser: tell me about cats")
    r.release(2)
    long_prefix = "system: you are helpful\nuser: tell me about dogs"   # shares 44 of 48 chars: 92%
    d = r.choose_agent(long_prefix)
    assert (d.slot, d.how) == (2, "lcp")
    short = "system: " + "x" * 200                                    # shares 8 of 208 chars: < 10%
    d2 = r.choose_agent(short)
    assert d2.how == "lru" and d2.slot in (1, 3)
    r.acquire(3, "s" * 90 + "z" * 910)
    r.release(3)
    d3 = r.choose_agent("s" * 90 + "q" * 810)                         # lcp 90 of 900 = exactly 10%: counts
    assert (d3.slot, d3.how) == (3, "lcp")
    d4 = r.choose_agent("s" * 89 + "q" * 811)                         # 89 of 900 < 10%: LRU instead
    assert d4.how == "lru"


def test_busy_slots_are_skipped_even_with_a_perfect_prefix():
    r = SlotRouter(4, 1)
    r.acquire(1, "conversation one")                     # in flight
    d = r.choose_agent("conversation one")
    assert d.slot != 1 and d.how == "lru"


def test_all_busy_goes_to_least_loaded_and_is_counted():
    r = SlotRouter(3, 1)
    r.acquire(1, "a")
    r.acquire(2, "b")
    r.acquire(2, "c")
    d = r.choose_agent("d")
    assert (d.slot, d.how, d.deferred) == (1, "least_loaded", True)
    assert r.counters["deferred"] == 1


def test_conversation_hint_maps_to_its_slot_when_idle():
    r = SlotRouter(4, 1)
    d = r.choose_agent("turn 1", hint="conv-A")
    r.acquire(d.slot, "turn 1", hint="conv-A")
    r.release(d.slot)
    first = d.slot
    r.acquire(first + 1 if first + 1 <= 3 else 1, "other")           # make a different slot busy
    d2 = r.choose_agent("completely different text", hint="conv-A")
    assert (d2.slot, d2.how) == (first, "hint")
    r.acquire(first, "x")                                            # the hinted slot is busy now
    d3 = r.choose_agent("completely different text", hint="conv-A")
    assert d3.slot != first and d3.how in ("lru", "least_loaded")


def test_client_id_slot_is_normalised_modulo_total_slots():
    r = SlotRouter(4, 1)
    assert r.normalise_id_slot(2) == 2 and r.normalise_id_slot(6) == 2 and r.normalise_id_slot(-1) is None
    d = r.choose_agent("x", client_slot=7)
    assert (d.slot, d.how, d.rerouted) == (3, "client", False)


def test_client_id_slot_in_the_reserved_range_is_rerouted_and_counted():
    r = SlotRouter(4, 2)
    d = r.choose_agent("x", client_slot=1)
    assert d.slot in (2, 3) and d.rerouted and d.how == "lru"
    d2 = r.choose_agent("x", client_slot=4)                 # 4 % 4 = 0: reserved
    assert d2.rerouted
    assert r.counters["rerouted"] == 2


def test_client_id_slot_of_wrong_type_is_absent():
    # "0", true, 1.0, null must neither crash nor pin.
    r = SlotRouter(4, 1)
    for bad in ("0", True, 1.0, None, [1], {"a": 1}):
        d = r.choose_agent("x", client_slot=bad)
        assert d.how == "lru" and not d.rerouted
    assert r.counters["id_slot_ignored"] == 5          # None is simply absent, not ignored


def test_s1_round_robin_across_reserved_slots_per_request():
    r = SlotRouter(5, 2)
    assert r.choose_s1() == (0, False)
    assert r.choose_s1() == (1, False)
    assert r.choose_s1() == (0, False)


def test_s1_fallback_to_lowest_idle_slot_when_reserved_slot_holds_a_foreign_task():
    r = SlotRouter(4, 1)
    r.acquire(1, "agent")                                 # slot 1 busy with our own agent traffic
    slot, fallback = r.choose_s1(processing=lambda s: s == 0)
    assert (slot, fallback) == (2, True)                  # 0 foreign, 1 ours-busy, 2 idle
    assert r.counters["s1_fallback"] == 1
    slot, fallback = r.choose_s1(processing=lambda s: True)
    assert (slot, fallback) == (0, True)                  # nothing idle: stay on the reserved slot
    slot, fallback = r.choose_s1(processing=lambda s: False)
    assert (slot, fallback) == (0, False)


def test_s1_own_inflight_task_on_the_reserved_slot_is_not_foreign():
    r = SlotRouter(4, 1)
    r.acquire_s1(0)                                       # a classification callo itself sent
    assert r.choose_s1(processing=lambda s: s == 0) == (0, False)
    assert r.counters["s1_fallback"] == 0


def test_s1_snapshot_older_than_our_release_is_not_foreign_but_a_newer_one_is():
    r = SlotRouter(4, 1)
    before = time.monotonic()
    r.acquire_s1(0)
    r.release_s1(0)                                       # our classification ended after the snapshot was fetched
    assert r.choose_s1(processing=lambda s: s == 0, as_of=before) == (0, False)
    after = time.monotonic()                              # a snapshot fetched after our release: busy is real
    assert r.choose_s1(processing=lambda s: s == 0, as_of=after) == (1, True)
    assert r.choose_s1(processing=lambda s: s == 0) == (1, True)    # no as_of: trusted as is
    r.acquire(1, "agent")
    r.release(1)                                          # a stale busy on a candidate is ignored the same way
    assert r.route_s1(processing=lambda s: s in (0, 1), as_of=after) == (1, True)


def test_s1_accounting_is_separate_from_agent_accounting():
    r = SlotRouter(4, 1)
    r.acquire_s1(0)
    assert r.inflight(0) == 1 and r.choose_agent("x").slot == 1
    r.release_s1(0)
    assert r.inflight(0) == 0


def test_snapshot_shape():
    r = SlotRouter(4, 1)
    r.acquire(1, "abc", hint="h")
    s = r.snapshot()
    assert s["total_slots"] == 4 and s["reserved"] == [0] and s["pinning"] is True
    assert s["inflight"] == {"0": 0, "1": 1, "2": 0, "3": 0} and s["text_chars"]["1"] == 3
    assert s["conversations"] == 1 and isinstance(s["counters"], dict)


def test_canonical_text_per_endpoint():
    # v0.3: canonical_text returns (text, digests); without media the text is v0.2's and there are no digests.
    assert canonical_text("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}) == \
        ('[{"role":"user","content":"hi"}]', [])
    assert canonical_text("/completion", {"prompt": "raw text"}) == ("raw text", [])
    assert canonical_text("/completion", {"prompt": [1, 2, 3]}) == ("1 2 3", [])
    assert canonical_text("/v1/responses", {"instructions": "be terse", "input": "hello"}) == ('["be terse","hello"]', [])
    assert canonical_text("/infill", {"input_prefix": "def f(", "input_suffix": "return 1"}) == \
        ('["def f(","return 1"]', [])
    assert canonical_text("/completion", {}) == ("", [])


def test_canonical_text_accepts_content_parts():
    # OpenAI multimodal content arrays must serialise, not raise.
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                                       {"type": "image_url", "image_url": {"url": "data:x"}}]}]}
    out, digests = canonical_text("/v1/chat/completions", body)
    assert isinstance(out, str) and '"text":"hi"' in out and '"data:x"' in out and digests == []   # short: kept


def test_conversation_hint_precedence():
    assert conversation_hint({"x-callo-conversation": "A", "x-conversation-id": "B"}, {"user": "C"}) == "A"
    assert conversation_hint({"x-conversation-id": "B"}, {"user": "C"}) == "B"
    assert conversation_hint({}, {"user": "C", "metadata": {"user_id": "D"}}) == "C"
    assert conversation_hint({}, {"metadata": {"user_id": "D"}}) == "D"
    assert conversation_hint({}, {"user": 5, "metadata": "nope"}) is None
    assert conversation_hint({}, {}) is None


def test_route_agent_never_hands_two_requests_the_same_idle_slot():
    r = SlotRouter(4, 1)
    r.acquire(2, "shared prefix conversation")
    r.release(2)
    d1 = r.route_agent("shared prefix conversation, turn A")
    d2 = r.route_agent("shared prefix conversation, turn B")
    assert (d1.slot, d1.how) == (2, "lcp")
    assert d2.slot in (1, 3) and not d2.deferred
    assert r.inflight(2) == 1 and r.inflight(d2.slot) == 1


def test_route_agent_unpinned_acquires_nothing():
    r = SlotRouter(4, 1, agent_pinning=False)
    assert r.route_agent("x").slot is None
    assert sum(r.snapshot()["inflight"].values()) == 0


def test_route_agent_concurrent_threads_get_distinct_slots():
    n = 6
    r = SlotRouter(n + 1, 1)
    barrier = threading.Barrier(n)
    out: list = []

    def go():
        barrier.wait()
        out.append(r.route_agent("same prompt for everyone"))

    threads = [threading.Thread(target=go) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(d.slot for d in out) == r.agent
    assert not any(d.deferred for d in out)
    assert sum(r.inflight(s) for s in r.agent) == n


def test_route_s1_acquires_and_round_robins():
    r = SlotRouter(5, 2)
    assert r.route_s1() == (0, False)
    assert r.route_s1() == (1, False)
    assert r.inflight(0) == 1 and r.inflight(1) == 1
    r.release_s1(0)
    assert r.inflight(0) == 0


def test_route_s1_concurrent_threads_account_every_acquire():
    r = SlotRouter(6, 2)
    n = 8
    barrier = threading.Barrier(n)
    out: list = []

    def go():
        barrier.wait()
        out.append(r.route_s1())

    threads = [threading.Thread(target=go) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(s for s, _ in out) == [0] * 4 + [1] * 4          # strict round robin under contention
    assert r.inflight(0) == 4 and r.inflight(1) == 4


def test_route_s1_fallback_to_an_idle_slot_acquires_that_slot():
    r = SlotRouter(4, 1)
    slot, fallback = r.route_s1(processing=lambda s: s == 0)
    assert (slot, fallback) == (1, True) and r.inflight(1) == 1 and r.inflight(0) == 0


def test_s1_fallback_acquire_clears_the_agent_slots_remembered_text():
    r = SlotRouter(4, 1)
    r.acquire(1, "system: you are helpful\nuser: tell me about cats")
    r.release(1)
    r.acquire_s1(1)                                       # S1 fell back onto agent slot 1: its KV is evicted
    r.release_s1(1)
    assert r.snapshot()["text_chars"]["1"] == 0
    d = r.choose_agent("system: you are helpful\nuser: tell me about cats")
    assert d.how == "lru"                                 # no stale LCP match on slot 1
    r.acquire(1, "system: you are helpful\nuser: tell me about cats")   # seed slot 1 again, then fall back onto it
    r.release(1)
    assert r.snapshot()["text_chars"]["1"] > 0
    slot, fallback = r.route_s1(processing=lambda s: s == 0)
    assert (slot, fallback) == (1, True)
    assert r.snapshot()["text_chars"]["1"] == 0


class _LoggedLock:
    """A real Lock that records every acquire/release, to count critical sections deterministically."""

    def __init__(self, after_first_release=None):
        self._lock = threading.Lock()
        self.events: list[str] = []
        self._hook = after_first_release          # runs once, right after the first release: a forced interleaving

    def __enter__(self):
        self._lock.acquire()
        self.events.append("acquire")
        return self

    def __exit__(self, *exc):
        self.events.append("release")
        self._lock.release()
        hook, self._hook = self._hook, None
        if hook is not None:
            hook()


def test_route_agent_is_one_critical_section():
    r = SlotRouter(4, 1)
    r._lock = _LoggedLock()
    assert r.route_agent("x").slot == 1
    assert r._lock.events == ["acquire", "release"]          # choose + acquire non-atomically would log two pairs


def test_route_s1_is_one_critical_section():
    r = SlotRouter(4, 1)
    r._lock = _LoggedLock()
    assert r.route_s1() == (0, False)
    assert r._lock.events == ["acquire", "release"]


def _rival(fn, out):
    """Run `fn` on another thread to completion; stands in for a concurrent request landing in the gap."""
    def go():
        t = threading.Thread(target=lambda: out.append(fn()))
        t.start()
        t.join(5)
        assert not t.is_alive()
    return go


def test_route_agent_rival_landing_between_choose_and_acquire_gets_another_slot():
    r = SlotRouter(4, 1)
    r.acquire(2, "shared prefix conversation")
    r.release(2)
    rival: list = []
    r._lock = _LoggedLock(_rival(lambda: r.route_agent("shared prefix conversation, turn B"), rival))
    mine = r.route_agent("shared prefix conversation, turn A")
    assert mine.slot == 2 and rival[0].slot in (1, 3)       # a non-atomic route_agent hands both slot 2


def test_route_s1_rival_landing_between_choose_and_acquire_gets_another_slot():
    r = SlotRouter(4, 1)
    rival: list = []
    r._lock = _LoggedLock(_rival(lambda: r.route_s1(processing=lambda s: s == 0), rival))
    mine = r.route_s1(processing=lambda s: s == 0)
    assert mine == (1, True) and rival == [(2, True)]       # a non-atomic route_s1 hands both slot 1


def test_route_s1_sequential_fallbacks_get_distinct_slots():
    r = SlotRouter(4, 1)
    assert r.route_s1(processing=lambda s: s == 0) == (1, True)
    assert r.route_s1(processing=lambda s: s == 0) == (2, True)     # slot 1 is ours-busy now, not idle
    assert r.counters["s1_fallback"] == 2


def test_total_slots_below_one_is_rejected():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            SlotRouter(bad, 1)


# ---- v0.2: lanes ------------------------------------------------------------------------------------

def test_lanes_bind_on_first_sight_to_the_lowest_free_reserved_slot_in_demo_order():
    r = SlotRouter(4, 3)
    assert [r.route_s1(lane=lane)[0] for lane in ("s1", "instinct_text", "instinct_view")] == [0, 1, 2]
    assert r.snapshot()["lanes"] == {"s1": 0, "instinct_text": 1, "instinct_view": 2}


def test_a_lane_is_sticky_and_ignores_the_round_robin():
    r = SlotRouter(5, 3)
    assert r.choose_s1(lane="b") == (0, False)
    assert r.choose_s1() == (0, False)                 # an unlaned request keeps v0.1's round robin
    assert r.choose_s1() == (1, False)
    assert [r.choose_s1(lane="b")[0] for _ in range(4)] == [0, 0, 0, 0]
    assert r.choose_s1(lane="c") == (1, False)


def test_a_busy_lane_queues_on_its_own_slot_and_never_takes_another_lanes_slot():
    r = SlotRouter(4, 2)
    assert r.route_s1(lane="s1") == (0, False)         # in flight, not released
    assert r.route_s1(lane="aux") == (1, False)
    assert r.route_s1(lane="s1") == (0, False)         # still slot 0: llama-server defers it until the slot frees
    assert r.counters["s1_lane_busy"] == 1 and r.snapshot()["s1_lane_busy"] == 1
    assert r.inflight(0) == 2 and r.inflight(1) == 1


def test_with_one_reserved_slot_the_header_is_ignored():
    r = SlotRouter(4, 1)
    assert r.route_s1(lane="s1") == (0, False) and r.route_s1(lane="other") == (0, False)
    assert r.snapshot()["lanes"] == {} and r.counters["s1_lane_busy"] == 0
    assert SlotRouter(1, 1).route_s1(lane="s1") == (None, False)


def test_a_new_lane_takes_the_least_recently_used_lanes_slot_once_all_are_bound():
    r = SlotRouter(4, 2)
    r.choose_s1(lane="a")
    r.choose_s1(lane="b")
    r.choose_s1(lane="a")                              # b is now the least recently used
    assert r.choose_s1(lane="c") == (1, False)
    assert r.snapshot()["lanes"] == {"a": 0, "c": 1} and r.counters["s1_lane_rebound"] == 1
    assert r.choose_s1(lane="b") == (0, False)         # a is now the LRU lane
    assert r.snapshot()["lanes"] == {"c": 1, "b": 0}


def test_more_lanes_than_reserved_slots_warns_once(caplog):
    # I-2: with more distinct conversations than reserved slots every request evicts a lane; warn once, name the cause.
    r = SlotRouter(4, 2)
    with caplog.at_level(logging.WARNING, logger="callo"):
        for lane in ("a", "b", "c", "d", "e"):         # a,b bind to 0,1; c,d,e each evict (3 rebinds)
            r.choose_s1(lane=lane)
    warns = [rec.getMessage() for rec in caplog.records
             if rec.name == "callo" and "distinct X-Callo-Conversation" in rec.getMessage()]
    assert len(warns) == 1 and r.counters["s1_lane_rebound"] == 3
    assert "--s1-slots (2)" in warns[0] and "at or below 2" in warns[0]


def test_the_s1_fallback_still_applies_on_top_of_a_lane():
    r = SlotRouter(4, 2)
    r.route_s1(lane="aux")                             # aux -> slot 0, in flight (ours)
    r.release_s1(0)
    slot, fell_back = r.route_s1(lambda s: s == 0, lane="aux")   # a FOREIGN task holds slot 0
    assert (slot, fell_back) == (1, True) and r.counters["s1_fallback"] == 1
    assert r.snapshot()["lanes"] == {"aux": 0}          # the binding itself does not move


def test_a_stale_snapshot_does_not_make_our_own_released_lane_slot_foreign():
    r = SlotRouter(4, 2)
    before = time.monotonic()
    assert r.route_s1(lane="a") == (0, False)
    r.release_s1(0)                                    # our request ended after the snapshot was fetched
    assert r.route_s1(lambda s: s == 0, as_of=before, lane="a") == (0, False)
    assert r.counters["s1_fallback"] == 0
    r.release_s1(0)
    after = time.monotonic()                           # a snapshot fetched after our release: busy is real
    assert r.route_s1(lambda s: s == 0, as_of=after, lane="a") == (1, True)
    assert r.counters["s1_fallback"] == 1


def test_a_foreign_task_on_a_lanes_slot_sends_it_to_another_lanes_idle_slot():
    r = SlotRouter(4, 2)                               # documents the v0.1 fallback applying on top of lanes
    r.route_s1(lane="a")
    r.release_s1(0)
    r.route_s1(lane="b")
    r.release_s1(1)
    assert r.snapshot()["lanes"] == {"a": 0, "b": 1}
    assert r.route_s1(lambda s: s == 0, lane="a") == (1, True)      # slot 0 is foreign-held: lands on b's slot
    assert r.snapshot()["lanes"] == {"a": 0, "b": 1}   # bindings unchanged
    assert r.inflight(0) == 0 and r.inflight(1) == 1


def test_evicting_a_lane_with_a_request_in_flight_queues_behind_it_and_releases_cleanly():
    r = SlotRouter(4, 2)
    assert r.route_s1(lane="a") == (0, False)          # a: slot 0, in flight
    assert r.route_s1(lane="b") == (1, False)
    r.release_s1(1)                                    # b finished and is idle; a (the LRU lane) is still running
    assert r.route_s1(lane="c") == (0, False)          # evicts a, queues behind a's request on slot 0
    assert r.snapshot()["lanes"] == {"b": 1, "c": 0}
    assert r.counters["s1_lane_rebound"] == 1 and r.counters["s1_lane_busy"] == 1
    assert r.inflight(0) == 2
    r.release_s1(0)                                    # the evicted lane's request finishes
    assert r.inflight(0) == 1
    r.release_s1(0)
    assert r.inflight(0) == 0 and r.inflight(1) == 0



# ---- v0.3: media payloads are digested in the routing text, and weighted ---------------------------------------

BIG = "A" * 300                     # over the 256-character digest threshold
BIG2 = "B" * 300


def _sha(s):
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


def _chat(*parts, text="look"):
    return {"messages": [{"role": "system", "content": "be terse"},
                         {"role": "user", "content": [{"type": "text", "text": text}, *parts]}]}


@pytest.mark.parametrize(("part", "payload"), [
    ({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{BIG}"}}, f"data:image/png;base64,{BIG}"),
    ({"type": "input_audio", "input_audio": {"data": BIG, "format": "wav"}}, BIG),
    ({"type": "input_audio", "input_audio": {"url": f"data:audio/wav;base64,{BIG}"}}, f"data:audio/wav;base64,{BIG}"),
    ({"type": "input_video", "input_video": {"data": BIG}}, BIG),
    ({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{BIG}"}}, f"data:video/mp4;base64,{BIG}"),
    ({"type": "input_image", "image_url": f"data:image/png;base64,{BIG}"}, f"data:image/png;base64,{BIG}"),
    ({"type": "audio_url", "audio_url": {"url": BIG}}, BIG),        # refused upstream, but never costs the router
])
def test_every_media_part_kind_is_digested_with_its_offset_and_weight(part, payload):
    path, body = "/v1/chat/completions", _chat(part)
    if part["type"] == "input_image":                                # /responses: a STRING image_url
        path, body = "/v1/responses", {"instructions": "be terse", "input": [{"role": "user", "content": [part]}]}
    text, digests = canonical_text(path, body)
    assert BIG not in text and _sha(payload) in text
    [(at, weight)] = digests
    assert text[at:at + DIGEST_CHARS] == _sha(payload) and weight == len(payload)


def test_completion_object_prompts_digest_every_multimodal_data_entry():
    one = {"prompt_string": "<__media__> what is it?", "multimodal_data": [BIG, "QUJD"]}
    text, digests = canonical_text("/completion", {"prompt": one})
    assert BIG not in text and _sha(BIG) in text and _sha("QUJD") in text and "what is it?" in text
    assert [w for _, w in digests] == [300, 4]
    text2, digests2 = canonical_text("/completion", {"prompt": [one, {**one, "multimodal_data": [BIG2]}]})
    assert [w for _, w in digests2] == [300, 4, 300] and _sha(BIG2) in text2


def test_short_payloads_and_non_media_parts_are_not_digested():
    body = _chat({"type": "input_audio", "input_audio": {"data": "QUJD"}}, text=BIG)   # a long TEXT stays text
    text, digests = canonical_text("/v1/chat/completions", body)
    assert digests == [] and BIG in text and '"QUJD"' in text


def test_canonical_text_never_changes_the_body_it_digests():
    """The proxy forwards the SAME body after routing; a digest in it would make llama-server fail with
    'Failed to load image or audio file'."""
    body = _chat({"type": "input_audio", "input_audio": {"data": BIG}},
                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{BIG2}"}})
    before = copy.deepcopy(body)
    system = body["messages"][0]
    canonical_text("/v1/chat/completions", body)
    assert body == before and body["messages"][0] is system


def test_a_20_mb_part_gives_constant_size_text_and_different_media_give_different_texts():
    big = "Q" * (20 * 1024 * 1024)
    text, digests = canonical_text("/v1/chat/completions", _chat({"type": "input_video", "input_video": {"data": big}}))
    assert len(text) < 1024 and digests[0][1] == len(big)
    other, _ = canonical_text("/v1/chat/completions",
                              _chat({"type": "input_video", "input_video": {"data": big[:-1] + "R"}}))
    same, _ = canonical_text("/v1/chat/completions", _chat({"type": "input_video", "input_video": {"data": big}}))
    assert other != text and same == text


def test_routing_a_20_mb_media_body_is_fast_under_the_lock():
    """The unit microbenchmark: v0.2 compared 20 MB per idle slot (about 0.36 s each). Now the
    digest is hashed once, outside the lock, and the compare is over a few hundred characters."""
    r = SlotRouter(5, 1)
    big = "Q" * (20 * 1024 * 1024)
    for slot, salt in ((1, "a"), (2, "b"), (3, "c")):
        r.acquire(slot, canonical_text("/v1/chat/completions",
                                       _chat({"type": "input_video", "input_video": {"data": big[:-1] + salt}}))[0])
        r.release(slot)
    t0 = time.perf_counter()
    text, digests = canonical_text("/v1/chat/completions", _chat({"type": "input_video", "input_video": {"data": big}}))
    t_digest = time.perf_counter() - t0
    # what the router compares and stores is tiny
    assert len(text) < 1024 and t_digest < 2.0
    times = []
    # the min of 5: one slow sample is no signal
    for _ in range(5):
        t0 = time.perf_counter()
        d = r.route_agent(text, digests=digests)
        times.append(time.perf_counter() - t0)
        # the stored routing text is the digested one
        assert r.snapshot()["text_chars"][str(d.slot)] == len(text)
        r.release(d.slot)
    assert min(times) < 0.05


def test_the_lcp_threshold_weights_digests_like_the_media_they_stand_for():
    """The failure case: an idle slot holds [system S, video V1, ...]; the request is [S, video V2, short
    text]. By llama-server's tokens, S is a sliver of the request, so LRU wins; the digested text alone would see S as
    well over 10% and evict the cached video conversation."""
    video = "V" * 200_000
    sys_turn = "You are a careful video assistant. " * 3

    def body(payload, text):
        return {"messages": [{"role": "system", "content": sys_turn},
                             {"role": "user", "content": [{"type": "input_video", "input_video": {"data": payload}},
                                                          {"type": "text", "text": text}]}]}

    r = SlotRouter(4, 1)
    held, _ = canonical_text("/v1/chat/completions", body(video, "what happens?"))
    r.acquire(2, held)
    r.release(2)
    text, digests = canonical_text("/v1/chat/completions", body(video[:-1] + "W", "and now?"))
    assert r.choose_agent(text).how == "lcp"                                # unweighted: the v0.2-latent mistake
    d = r.choose_agent(text, digests=digests)
    assert d.how == "lru" and d.slot != 2                                    # weighted: llama-server's choice
    same, same_digests = canonical_text("/v1/chat/completions", body(video, "and now?"))
    assert r.choose_agent(same, digests=same_digests) == r.choose_agent(same)   # the same video: lcp, slot 2
    assert r.choose_agent(same, digests=same_digests).slot == 2


def test_the_weighting_flips_lru_to_lcp_for_the_same_video_and_a_long_new_turn():
    """The other direction of the weighted threshold: the same video plus a long new text turn. By characters
    the shared prefix (system turn + one digest) is under 10% of the request, so the unweighted test picks
    LRU; by llama-server's tokens the cached video is most of it, so the weighted test keeps the slot that
    holds it."""
    video = "V" * 200_000
    sys_turn = "You are a careful video assistant. "

    def body(text):
        part = {"type": "input_video", "input_video": {"data": video}}
        return {"messages": [{"role": "system", "content": sys_turn},
                             {"role": "user", "content": [part, {"type": "text", "text": text}]}]}

    r = SlotRouter(4, 1)
    held, _ = canonical_text("/v1/chat/completions", body("what happens?"))
    r.acquire(2, held)
    r.release(2)
    text, digests = canonical_text("/v1/chat/completions", body("now describe every frame in detail. " * 200))
    unweighted = r.choose_agent(text)
    assert unweighted.how == "lru" and unweighted.slot != 2
    assert r.choose_agent(text, digests=digests) == r.choose_agent(text, digests=digests)
    d = r.choose_agent(text, digests=digests)
    assert d.how == "lcp" and d.slot == 2


def test_located_skips_a_digest_it_cannot_find_never_offset_minus_one():
    """The digests appear verbatim in the text, but a miss must never become offset
    -1."""
    d1, d2 = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    assert _located(f"x{d1}y", [(d1, 500), (d2, 900)]) == [(1, 500)]
    # a miss does not move the search start
    assert _located(f"{d2}", [(d1, 500), (d2, 900)]) == [(0, 900)]


def test_a_part_whose_type_is_not_a_string_is_no_media_part():
    """{"type": ["string", "null"]} must not raise in the set lookup."""
    body = _chat({"type": ["string", "null"], "data": BIG}, {"type": {"nested": 1}}, {"type": None, "url": BIG})
    text, digests = canonical_text("/v1/chat/completions", body)
    assert digests == [] and BIG in text


def test_a_lone_surrogate_payload_is_digested_not_a_crash():
    """The string "\\ud800" is valid JSON; the digest hashes it with surrogatepass."""
    lone = "\ud800" * 300
    text, digests = canonical_text("/v1/chat/completions", _chat({"type": "input_audio", "input_audio": {"data": lone}}))
    assert lone not in text and digests[0][1] == 300


# ---- v0.3 think budgets: deliberate routing and the S1 section ---------------------------------------------

from callo.slots import Section  # noqa: E402


def test_a_deliberate_request_takes_the_lowest_idle_unbound_reserved_slot_else_round_robin():
    r = SlotRouter(5, 2)
    assert r.route_s1(deliberate=True) == (0, False)                   # idle, unbound: the lowest
    assert r.route_s1(deliberate=True) == (1, False)                   # 0 is in flight
    third, _ = r.route_s1(deliberate=True)                             # every slot thinks: the round robin (slot 0)
    assert third == 0 and r.snapshot()["deliberate"] == {"0": 2, "1": 1}
    r.release_s1(third, deliberate=True)                               # released, so the lane part below is real
    r.release_s1(0, deliberate=True)
    r.release_s1(1, deliberate=True)
    assert r.counters["s1_deliberate_idle"] == 2 and r.snapshot()["deliberate"] == {"0": 0, "1": 0}
    r.route_s1(lane="L")                                               # binds lane L to slot 0
    r.release_s1(0)
    assert all(r.inflight(s) == 0 for s in (0, 1))                     # both idle: only the lane filter decides
    assert r.route_s1(deliberate=True) == (1, False)                   # 0 is lane-bound: skipped


def test_when_nothing_is_idle_requests_prefer_the_least_busy_slot_without_a_deliberate_question():
    r = SlotRouter(5, 2)
    assert r.route_s1(deliberate=True) == (0, False)                   # slot 0 thinks
    assert r.route_s1() == (1, False)                                  # a fast request is in flight on slot 1
    for _ in range(4):
        assert r.route_s1() == (1, False)                              # never slot 0, busy or not
    assert r.counters["s1_avoided_deliberate"] == 5
    assert r.route_s1(deliberate=True) == (1, False)                   # a second trace: the quiet slot, not slot 0
    assert r.counters["s1_deliberate_quiet"] == 1
    assert r.route_s1()[0] in (0, 1)                                   # every slot thinks: the round robin
    assert r.counters["s1_avoided_deliberate"] == 5


def test_a_fast_request_avoids_a_slot_holding_a_deliberate_question_and_a_lane_stays():
    r = SlotRouter(5, 2)
    assert r.route_s1(deliberate=True) == (0, False)
    assert r.route_s1() == (1, False)                                  # not the round robin: slot 1 is idle
    r.release_s1(1)
    assert r.route_s1() == (1, False)                                  # again, where round robin would say 0
    assert r.counters["s1_avoided_deliberate"] == 2
    r.release_s1(1)
    r.route_s1(lane="L")
    r.release_s1(r._lanes["L"])
    bound = r._lanes["L"]
    assert r.route_s1(lane="L") == (bound, False)                      # a lane request stays on its lane
    r.release_s1(0, deliberate=True)
    assert r.snapshot()["deliberate"]["0"] == 0


def test_without_deliberate_traffic_routing_is_v02s_round_robin():
    r = SlotRouter(5, 2)
    assert [r.route_s1()[0] for _ in range(3)] == [0, 1, 0]
    assert r.counters["s1_avoided_deliberate"] == 0 and r.counters["s1_deliberate_idle"] == 0
    assert r.section(0) is not None and r.section(1) is not None and r.section(None) is None
    assert SlotRouter(1, 1).section(0) is None                          # -np 1: no reserved slots


def test_shared_tickets_enter_together_and_an_exclusive_one_waits_for_readers():
    s = Section()
    assert s.acquire(exclusive=False, timeout=1) and s.acquire(exclusive=False, timeout=1)
    assert s.waits == 0 and s.snapshot()["readers"] == 2
    got = []
    t = threading.Thread(target=lambda: got.append(s.acquire(exclusive=True, timeout=5)))
    t.start()
    time.sleep(0.1)
    assert got == [] and s.snapshot()["queued"] == 1
    s.release(exclusive=False)
    s.release(exclusive=False)
    t.join(timeout=2)
    assert got == [True] and s.snapshot()["writer"] is True and s.waits == 1
    s.release(exclusive=True)


def test_fifo_a_reader_behind_a_waiting_writer_waits_for_it_and_a_timeout_returns_false():
    s = Section()
    assert s.acquire(exclusive=False, timeout=1)                        # a reader holds
    order = []

    def writer():
        order.append(("w", s.acquire(exclusive=True, timeout=5)))

    def reader():
        order.append(("r", s.acquire(exclusive=False, timeout=5)))

    tw = threading.Thread(target=writer)
    tw.start()
    time.sleep(0.1)
    tr = threading.Thread(target=reader)
    tr.start()
    time.sleep(0.1)
    assert order == []                                                  # the reader waits behind the writer
    assert s.acquire(exclusive=False, timeout=0.2) is False             # bounded: a late reader times out
    s.release(exclusive=False)
    tw.join(timeout=2)
    assert order == [("w", True)]
    s.release(exclusive=True)
    tr.join(timeout=2)
    assert order == [("w", True), ("r", True)]
    s.release(exclusive=False)
    assert s.snapshot() == {"readers": 0, "writer": False, "queued": 0, "waits": 3}


def test_a_cancelled_waiter_leaves_the_queue():
    s = Section()
    assert s.acquire(exclusive=True, timeout=1)
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    t0 = time.monotonic()
    assert s.acquire(exclusive=True, timeout=10, cancel=cancel) is False
    assert time.monotonic() - t0 < 2 and s.snapshot()["queued"] == 0
    s.release(exclusive=True)
    assert s.acquire(exclusive=True, timeout=1)


def test_many_threads_never_mix_a_writer_with_anyone_and_leave_no_ticket_behind():
    """Stress (beyond the brief): readers, writers, timeouts and cancels at once; the holder invariant always holds,
    nobody hangs, and the queue ends empty."""
    s = Section()
    held = {"r": 0, "w": 0, "bad": 0}
    lock = threading.Lock()

    def worker(i):
        exclusive = i % 4 == 0
        cancel = threading.Event() if i % 7 == 0 else None
        if cancel is not None:
            threading.Timer(0.01, cancel.set).start()
        if not s.acquire(exclusive=exclusive, timeout=0.02 if i % 5 == 0 else 5, cancel=cancel):
            return
        with lock:
            held["w" if exclusive else "r"] += 1
            held["bad"] += held["w"] > 1 or (held["w"] > 0 and held["r"] > 0)
        time.sleep(0.002)
        with lock:
            held["w" if exclusive else "r"] -= 1
        s.release(exclusive=exclusive)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(80)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads) and held["bad"] == 0
    snap = s.snapshot()
    assert (snap["readers"], snap["writer"], snap["queued"]) == (0, False, 0)


def test_a_deliberate_pick_of_a_lane_bound_slot_is_counted_and_an_unbound_one_wins_a_tie():
    """With no idle unbound slot, the least busy quiet slot can be an idle lane-bound one; the eviction of that lane's warm turn shows in s1_deliberate_lane_slot, and on a tie an unbound slot wins."""
    r = SlotRouter(5, 2)
    r.route_s1(lane="L")                                               # lane L binds slot 0 ...
    r.release_s1(0)                                                    # ... which is idle again
    r.acquire_s1(1)                                                    # slot 1 busy (unbound, no trace)
    assert r.route_s1(deliberate=True) == (0, False)                   # the idle quiet slot: lane L's
    assert r.counters["s1_deliberate_quiet"] == 1 and r.counters["s1_deliberate_lane_slot"] == 1
    r.release_s1(0, deliberate=True)
    r2 = SlotRouter(5, 2)
    r2.route_s1(lane="L")                                              # slot 0: lane L, in flight
    r2.acquire_s1(1)                                                   # slot 1: unbound, in flight: a tie on inflight
    assert r2.route_s1(deliberate=True) == (1, False)                  # the unbound slot wins the tie
    assert r2.counters["s1_deliberate_lane_slot"] == 0


def test_release_s1_takes_the_router_lock_once():
    r = SlotRouter(5, 2)
    slot, _ = r.route_s1(deliberate=True)
    taken = []

    class Counting:
        def __init__(self, lock):
            self.lock = lock

        def __enter__(self):
            taken.append(1)
            return self.lock.__enter__()

        def __exit__(self, *a):
            return self.lock.__exit__(*a)

    r._lock = Counting(r._lock)
    r.release_s1(slot, deliberate=True)
    assert len(taken) == 1 and r.inflight(slot) == 0 and r.snapshot()["deliberate"][str(slot)] == 0


def test_an_empty_routing_text_is_never_an_lcp_match_and_clears_the_slots_text():
    """Transcriptions ride on this contract: route_agent("") takes the hint, else the LRU slot, never
    LCP (`text and ...`), and the slot's routing text becomes "" (its KV now holds the ASR prompt)."""
    r = SlotRouter(4, 1)
    d = r.route_agent('[{"role":"user","content":"agent"}]')
    r.release(d.slot)
    d = r.route_agent("")
    assert (d.slot, d.how) == (2, "lru") and r.counters["lcp"] == 0
    r.release(d.slot)
    assert r.snapshot()["text_chars"]["2"] == 0
    d = r.route_agent("", hint="h")
    r.release(d.slot)
    assert r.route_agent("", hint="h").how == "hint"


ASR = "callo:asr"


def test_a_sticky_hint_keeps_its_home_when_the_home_is_busy():
    """Transcriptions: a busy home serves this request on the normal pick, but the hint stays home."""
    r = SlotRouter(4, 1)
    d = r.route_agent("", hint=ASR, sticky=True)
    assert (d.slot, d.how) == (1, "lru")                 # the first binding: the home is slot 1
    r.release(d.slot)
    r.acquire(1, "agent")                                # something else now holds the home
    d = r.route_agent("", hint=ASR, sticky=True)
    assert (d.slot, d.how) == (2, "lru") and r.counters["deferred"] == 0
    r.release(d.slot)
    r.release(1)
    d = r.route_agent("", hint=ASR, sticky=True)
    assert (d.slot, d.how) == (1, "hint")                # the next one goes home again


def test_a_client_hint_still_moves_on_fallback():
    """Every other caller keeps today's rule: a hint is rebound to whatever slot it was given."""
    for kw in ({}, {"sticky": False}):
        r = SlotRouter(4, 1)
        d = r.route_agent("x", hint="alice", **kw)
        r.release(d.slot)
        assert d.slot == 1
        r.acquire(1, "other")
        d = r.route_agent("x", hint="alice", **kw)
        r.release(d.slot)
        r.release(1)
        assert d.slot == 2
        assert r.route_agent("x", hint="alice", **kw).slot == 2          # moved with the fallback


def test_after_an_overflow_the_stream_goes_home_and_never_settles_on_an_agents_slot():
    """An overflow may land once on an agent's slot (when only it is idle); the stream that follows never does. With
    the hint moving (sticky=False), every later transcription would evict that agent's KV."""
    r = SlotRouter(4, 1)
    home = r.route_agent("", hint=ASR, sticky=True)
    r.release(home.slot)                                                  # home: slot 1
    alice = r.route_agent("alice turn", hint="alice")
    r.release(alice.slot)                                                 # alice: slot 2
    r.acquire(1, "busy home")
    r.acquire(3, "bob")                                                   # only alice's slot is idle
    d = r.route_agent("", hint=ASR, sticky=True)
    assert d.slot == alice.slot                                           # the one overflow
    r.release(d.slot)
    r.release(1)
    r.release(3)
    a = r.route_agent("alice turn 2", hint="alice")
    r.release(a.slot)
    assert a.slot == alice.slot                                           # alice's hint is hers again
    for _ in range(20):
        d = r.route_agent("", hint=ASR, sticky=True)
        r.release(d.slot)
        assert d.slot == home.slot                                        # never alice's slot
    assert r.snapshot()["text_chars"][str(alice.slot)] == len("alice turn 2")


# ---- the sticky home is a slot no agent hint holds ----------------------------------------------------------------

def test_the_first_home_avoids_a_hint_bound_agent_slot():
    """The LRU idle slot is alice's, so the home is the one slot no hint is bound to (carol's
    chat came without a hint). A busy hint-free slot is never waited for: that request takes the
    normal pick, an idle hint-bound slot, without binding callo:asr there."""
    r = SlotRouter(4, 1)
    taken = []
    for text, hint in (("alice turn", "alice"), ("bob turn", "bob"), ("carol turn", None)):
        d = r.route_agent(text, hint=hint)
        r.release(d.slot)
        taken.append(d.slot)
    assert taken == [1, 2, 3]                              # slot 1 (alice's) is now the LRU idle slot
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    assert (d.slot, d.how) == (3, "lru")
    r = SlotRouter(4, 1)
    for text, hint in (("alice turn", "alice"), ("bob turn", "bob")):
        r.release(r.route_agent(text, hint=hint).slot)
    r.acquire(3, "carol")
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    r.release(3)
    assert (d.slot, d.how, d.deferred) == (1, "lru", False) and r.counters["deferred"] == 0
    assert r.snapshot()["sticky"][ASR]["home"] is None


def test_an_agent_claiming_the_home_moves_the_next_transcription_to_a_free_slot():
    """alice's hint lands on the home (her fallback while 2 and 3 are busy), which unbinds
    callo:asr; the next transcription chooses a new hint-free home and stays there, and alice keeps slot 1."""
    r = SlotRouter(4, 1)
    home = r.route_agent("", hint=ASR, sticky=True)
    r.release(home.slot)
    assert home.slot == 1
    r.acquire(2, "busy")
    r.acquire(3, "busy")
    a = r.route_agent("alice turn", hint="alice")
    r.release(a.slot)
    assert a.slot == home.slot                             # the agent has claimed the home
    r.release(2)
    r.release(3)
    seen = set()
    for _ in range(5):
        d = r.route_agent("", hint=ASR, sticky=True)
        r.release(d.slot)
        seen.add(d.slot)
    assert seen == {2}                                     # a new, hint-free home, kept
    a = r.route_agent("alice turn 2", hint="alice")
    r.release(a.slot)
    assert (a.slot, a.how) == (1, "hint")


def test_with_every_agent_slot_hint_bound_the_home_is_the_normal_pick_until_its_agent_returns():
    """The documented fallback: no hint-free slot, so the home is the normal pick (LRU idle); it is
    kept while its agent is away, and that agent's next turn re-homes callo:asr by the normal pick again."""
    r = SlotRouter(4, 1)
    for n in ("a", "b", "c"):
        d = r.route_agent(f"{n} turn", hint=n)
        r.release(d.slot)
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    assert (d.slot, d.how) == (1, "lru") and r.snapshot()["conversations"] == 3       # callo:asr is under `sticky`
    assert r.snapshot()["sticky"][ASR]["home"] == 1
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    assert (d.slot, d.how) == (1, "hint")                  # kept while a is away
    d = r.route_agent("a turn 2", hint="a")
    r.release(d.slot)
    assert (d.slot, d.how) == (1, "hint") and r.snapshot()["sticky"][ASR]["home"] is None    # a is back: unbound
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    assert (d.slot, d.how) == (2, "lru")                   # the normal pick again: b's slot is now the LRU


def test_non_sticky_routing_ignores_the_sticky_home():
    """Client hints and hint-less agent chats route exactly as before. A client hint still moves on a
    fallback, onto the home too; LCP and LRU see the home as any other slot."""
    r = SlotRouter(4, 1)
    home = r.route_agent("", hint=ASR, sticky=True)
    r.release(home.slot)
    chat = '[{"role":"user","content":"agent one"}]'
    d = r.route_agent(chat)
    r.release(d.slot)
    assert (d.slot, d.how) == (2, "lru")
    d = r.route_agent(chat[:-1] + ',{"role":"assistant","content":"x"}]')
    r.release(d.slot)
    assert (d.slot, d.how) == (2, "lcp")
    d = r.route_agent("alice turn", hint="alice", sticky=False)
    r.release(d.slot)
    assert (d.slot, d.how) == (3, "lru")
    r.acquire(3, "busy")
    d = r.route_agent("alice turn 2", hint="alice")
    r.release(d.slot)
    r.release(3)
    assert (d.slot, d.how) == (1, "lru")                   # moved with the fallback, onto the home slot
    assert r.route_agent("alice turn 3", hint="alice").slot == 1


# ---- final fix wave: a homeless callo:asr never waits on a busy unbound slot; the home is visible ---------

def test_a_homeless_transcription_never_waits_on_a_busy_unbound_slot():
    """A busy home is never waited for: the one hint-free slot is busy and the hint-bound ones
    are idle, so this transcription is served at once on the normal pick (the LRU idle slot), and callo:asr stays
    homeless; the next one, with the hint-free slot idle again, homes there."""
    r = SlotRouter(4, 1)
    for text, hint in (("alice turn", "alice"), ("bob turn", "bob")):
        r.release(r.route_agent(text, hint=hint).slot)
    r.acquire(3, "carol")                                  # the hint-free slot is busy with a hint-less chat
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    assert (d.slot, d.how, d.deferred) == (1, "lru", False) and r.counters["deferred"] == 0
    assert r.snapshot()["sticky"] == {ASR: {"home": None, "rehomed": 0}}     # still unbound
    a = r.route_agent("alice turn 2", hint="alice")
    r.release(a.slot)
    assert (a.slot, a.how) == (1, "hint")                  # alice's binding was not disturbed
    r.release(3)
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    assert (d.slot, d.how) == (3, "lru") and r.snapshot()["sticky"] == {ASR: {"home": 3, "rehomed": 0}}


def test_the_snapshot_reports_the_sticky_home_apart_from_the_conversations():
    """`conversations` counts client conversation hints only; each sticky hint's home (or None) and how
    often it was re-homed (bound again after losing its home) are under `sticky`."""
    r = SlotRouter(4, 1)
    assert r.snapshot()["sticky"] == {} and r.snapshot()["conversations"] == 0
    r.release(r.route_agent("", hint=ASR, sticky=True).slot)
    assert r.snapshot()["sticky"] == {ASR: {"home": 1, "rehomed": 0}} and r.snapshot()["conversations"] == 0
    r.acquire(2, "busy")
    r.acquire(3, "busy")
    r.release(r.route_agent("alice turn", hint="alice").slot)      # alice claims the home: callo:asr unbound
    r.release(2)
    r.release(3)
    snap = r.snapshot()
    assert snap["sticky"] == {ASR: {"home": None, "rehomed": 0}} and snap["conversations"] == 1
    r.release(r.route_agent("", hint=ASR, sticky=True).slot)
    assert r.snapshot()["sticky"] == {ASR: {"home": 2, "rehomed": 1}}
    r.release(r.route_agent("", hint=ASR, sticky=True).slot)      # going home is not a re-home
    assert r.snapshot()["sticky"] == {ASR: {"home": 2, "rehomed": 1}}


# ---- an idle-aware callo:asr home, a hint cap ----------------------------------------------------------------------

class Clock:
    """An injected monotonic clock: `t` seconds, moved by the test."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _routed(r, text, hint, clock, at):
    clock.t = at
    d = r.route_agent(text, hint=hint)
    r.release(d.slot)
    return d


def _stale_setup(at):
    """bob (slot 1) at t=0, carol (slot 2) at t=10, dave (slot 3) at t=20, each turn ending as it starts; then a
    hint-less transcription at `at`."""
    clock = Clock()
    r = SlotRouter(4, 1, asr_home_idle=60, clock=clock)
    assert [_routed(r, f"{n} turn", n, clock, t).slot for n, t in (("bob", 0), ("carol", 10), ("dave", 20))] == [1, 2, 3]
    clock.t = at
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    return r, clock, d


def test_an_idle_bound_slot_becomes_a_home_candidate_after_the_threshold_and_not_before():
    r, _, d = _stale_setup(59)                        # nothing idle for 60 s yet: every agent slot is hint-bound
    assert (d.slot, d.how) == (1, "lru") and r.counters["idle_hint"] == 0                  # the normal pick (LRU)
    assert r.snapshot()["sticky"][ASR]["home"] == 1
    r, _, d = _stale_setup(60)                        # bob's hint has been idle 60 s: his slot is a candidate
    assert (d.slot, d.how) == (1, "idle_hint") and r.counters["idle_hint"] == 1
    assert r.snapshot()["sticky"][ASR]["home"] == 1 and r.snapshot()["conversations"] == 3   # every binding kept


def test_the_stalest_idle_bound_slot_wins_and_a_busy_one_is_never_waited_for():
    r, clock, d = _stale_setup(100)                   # all three hints idle >= 60 s: bob's (t=0) is the stalest
    assert (d.slot, d.how) == (1, "idle_hint")
    clock = Clock()
    r = SlotRouter(4, 1, asr_home_idle=60, clock=clock)
    for n, t in (("bob", 0), ("carol", 10), ("dave", 20)):
        _routed(r, f"{n} turn", n, clock, t)
    r.acquire(1, "bob's long generation")              # the stalest candidate is busy: the next stalest, idle
    clock.t = 100
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    r.release(1)
    assert (d.slot, d.how, d.deferred) == (2, "idle_hint", False)
    r = SlotRouter(3, 1, asr_home_idle=60, clock=clock)  # the one candidate busy: homeless, normal pick
    _routed(r, "bob turn", "bob", clock, 0)
    _routed(r, "carol turn", "carol", clock, 95)
    r.acquire(1, "bob's slot, in use without a hint")
    clock.t = 100
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    r.release(1)
    assert (d.slot, d.how) == (2, "lru") and r.snapshot()["sticky"][ASR]["home"] is None


def test_an_unbound_slot_is_preferred_over_an_idle_bound_one():
    clock = Clock()
    r = SlotRouter(4, 1, asr_home_idle=60, clock=clock)
    _routed(r, "bob turn", "bob", clock, 0)
    _routed(r, "carol turn", "carol", clock, 0)
    assert _routed(r, "zzz", None, clock, 1).slot == 3            # slot 3: hint-free, and the most recently used
    clock.t = 1000
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    assert (d.slot, d.how) == (3, "lru")


def test_the_returning_agent_gets_its_slot_back_and_the_transcription_rehomes():
    r, clock, d = _stale_setup(100)
    assert d.slot == 1
    for _ in range(3):                                            # home while bob is away
        clock.t = 110
        h = r.route_agent("", hint=ASR, sticky=True)
        r.release(h.slot)
        assert (h.slot, h.how) == (1, "hint")
    b = _routed(r, "bob turn 2", "bob", clock, 120)
    assert (b.slot, b.how) == (1, "hint") and r.snapshot()["sticky"][ASR] == {"home": None, "rehomed": 0}
    clock.t = 130
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    assert (d.slot, d.how) == (2, "idle_hint") and r.snapshot()["sticky"][ASR] == {"home": 2, "rehomed": 1}


def test_a_zero_threshold_makes_any_bound_hint_not_in_use_a_candidate():
    clock = Clock()
    r = SlotRouter(4, 1, asr_home_idle=0, clock=clock)
    for n in ("a", "b", "c"):
        _routed(r, f"{n} turn", n, clock, 5)
    r.acquire(1, "a in use")
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    r.release(1)
    assert (d.slot, d.how) == (2, "idle_hint")


def test_the_hint_cap_evicts_the_least_recently_used_binding_never_one_on_a_busy_slot():
    clock = Clock()
    r = SlotRouter(4, 1, max_hints=3, clock=clock)
    for n, t in (("a", 0), ("b", 1), ("c", 2)):
        _routed(r, f"{n} turn", n, clock, t)
    r.acquire(1, "a's slot is busy")                  # a is the LRU binding, but its slot has a request in flight
    _routed(r, "d turn", "d", clock, 3)               # d takes b's slot 2 (d's own request does not make
    r.release(1)                                      # b's slot busy), so b, the next LRU, goes
    assert r.counters["hint_evicted"] == 1 and r.snapshot()["conversations"] == 3
    assert _routed(r, "a turn 2", "a", clock, 4).how == "hint"                 # a and c kept their bindings
    assert _routed(r, "c turn 2", "c", clock, 5).how == "hint"
    assert _routed(r, "b turn 2", "b", clock, 6).how != "hint"                 # b's was evicted; binding it again
    assert r.counters["hint_evicted"] == 2 and r.snapshot()["conversations"] == 3   # evicts d, now the LRU


def test_the_cap_holds_with_one_agent_slot():
    r = SlotRouter(2, 1, max_hints=100)
    for i in range(3000):
        r.release(r.route_agent(f"turn {i}", hint=f"h{i}").slot)
        assert r.snapshot()["conversations"] <= 100
    assert r.counters["hint_evicted"] == 2900


def test_the_cap_holds_when_lcp_concentrates_hints_on_one_slot():
    r = SlotRouter(4, 1, max_hints=100)
    prefix = "a long shared system prompt " * 20
    for i in range(3000):
        r.release(r.route_agent(prefix + f"turn {i}", hint=f"h{i}").slot)
        assert r.snapshot()["conversations"] <= 100
    assert r.counters["lcp"] > 2000 and r.counters["hint_evicted"] == 2900


def test_the_cap_drains_on_release_once_a_busy_slot_frees():
    r = SlotRouter(2, 1, max_hints=2)
    held = [r.route_agent(f"turn {i}", hint=f"h{i}") for i in range(4)]     # all on slot 1, all in flight
    assert r.snapshot()["conversations"] == 4 and r.counters["hint_evicted"] == 0       # never evicted while busy
    for d in held:
        r.release(d.slot)
    assert r.snapshot()["conversations"] == 2 and r.counters["hint_evicted"] == 2


def test_idle_time_counts_from_the_end_of_the_hints_last_request():
    """alice's turn runs from t=0 to t=700; slots 2 and 3 hold stale hints but are busy. At t=701 (threshold 600)
    alice's slot is not an idle home (the transcription takes the normal pick, homeless); at t=1301 it is."""
    def at(t):
        clock = Clock()
        r = SlotRouter(4, 1, asr_home_idle=600, clock=clock)
        assert [r.route_agent(f"{n} turn", hint=n).slot for n in ("alice", "bob", "carol")] == [1, 2, 3]
        clock.t = 700
        r.release(1)                                  # alice's turn ends; bob and carol are still generating
        clock.t = t
        d = r.route_agent("", hint=ASR, sticky=True)
        return (d.slot, d.how), r.snapshot()["sticky"][ASR]["home"]

    assert at(701) == ((1, "lru"), None)
    assert at(1301) == ((1, "idle_hint"), 1)


def test_non_sticky_routing_is_unchanged_by_the_clock():
    """The idle-aware home changes only the sticky home: a client hint returns to its slot however long it was idle, with a zero
    threshold too, and LCP / LRU route hint-less chats as before."""
    clock = Clock()
    r = SlotRouter(4, 1, asr_home_idle=0, clock=clock)
    a = _routed(r, "alice turn", "alice", clock, 0)
    assert (a.slot, a.how) == (1, "lru")
    chat = '[{"role":"user","content":"agent one"}]'
    assert _routed(r, chat, None, clock, 1).how == "lru"
    assert _routed(r, chat[:-1] + ',{"role":"assistant","content":"x"}]', None, clock, 2).how == "lcp"
    back = _routed(r, "alice turn 2", "alice", clock, 1e9)
    assert (back.slot, back.how) == (1, "hint")
    assert r.counters["idle_hint"] == 0 and r.counters["hint_evicted"] == 0


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_the_router_refuses_a_bad_home_idle_threshold(bad):
    with pytest.raises(InvalidInput, match="--asr-home-idle"):
        SlotRouter(4, 1, asr_home_idle=bad)


class _CountingOrder(OrderedDict):
    """`_conv_used` that counts its scans: `_evict_locked` iterates it to find a victim."""
    scans = 0

    def __iter__(self):
        type(self).scans += 1
        return super().__iter__()


def test_eviction_does_not_scan_the_table_while_no_bound_slot_is_free():
    """One agent slot that always has another request in flight: no binding can go, and `_evict_locked` returns before
    scanning the bindings (no O(n) work per request under the router lock); the release that leaves the slot with
    nothing in flight drains the table back to the cap."""
    r = SlotRouter(2, 1, max_hints=2)
    r._conv_used = _CountingOrder()
    _CountingOrder.scans = 0
    r.acquire(1, "a long request without a hint")
    for i in range(50):
        r.release(r.route_agent(f"turn {i}", hint=f"h{i}").slot)
    assert r.snapshot()["conversations"] == 50 and r.counters["hint_evicted"] == 0 and _CountingOrder.scans == 0
    r.release(1)
    assert r.snapshot()["conversations"] == 2 and r.counters["hint_evicted"] == 48


def test_the_idle_hint_home_is_ordered_by_the_end_of_each_slots_last_request():
    """bob's turn runs 0 -> 50 s on slot 1; carol's (slot 2, at 10 s) and dave's (slot 3, at 12 s) end as they start.
    At 200 s all three are idle past the threshold: the stalest by release is carol's slot 2, where the routing-tick
    LRU order would pick bob's slot 1."""
    clock = Clock()
    r = SlotRouter(4, 1, asr_home_idle=60, clock=clock)
    bob = r.route_agent("bob turn", hint="bob")
    assert bob.slot == 1
    assert _routed(r, "carol turn", "carol", clock, 10).slot == 2
    assert _routed(r, "dave turn", "dave", clock, 12).slot == 3
    clock.t = 50
    r.release(bob.slot)
    clock.t = 200
    d = r.route_agent("", hint=ASR, sticky=True)
    r.release(d.slot)
    assert (d.slot, d.how) == (2, "idle_hint")


# ---- a cancelled deliberate slot drains ----------------------------------------------------------------

def test_a_draining_slot_is_busy_to_route_s1_until_a_later_snapshot_shows_it_idle():
    r = SlotRouter(4, 2)
    s, _ = r.route_s1(deliberate=True)
    assert s == 0
    r.release_s1(0, deliberate=True)
    r.mark_draining(0)
    t_cancel = time.monotonic()
    busy = {0}.__contains__                                        # the zombie task is still upstream
    for deliberate in (True, False):
        slot, fell_back = r.route_s1(busy, as_of=time.monotonic(), deliberate=deliberate)
        assert slot == 1 and fell_back is False and "s1_fallback" not in r.counters     # routed, not the foreign fallback
        r.release_s1(slot, deliberate=deliberate)
    for _ in range(3):                                             # no snapshot at all: the round robin skips it too
        slot, fell_back = r.route_s1(deliberate=False)
        assert slot == 1 and fell_back is False
        r.release_s1(slot, deliberate=False)
    slot, _ = r.route_s1(lambda s: False, as_of=t_cancel - 1.0, deliberate=True)   # a snapshot from before the cancel
    assert slot == 1
    r.release_s1(slot, deliberate=True)
    slot, _ = r.route_s1(lambda s: False, as_of=time.monotonic(), deliberate=True)  # after it, idle: drained
    assert slot == 0 and r.counters["s1_draining"] == 1 and r.snapshot()["draining"] == []


def test_a_draining_slot_clears_after_drain_s_without_a_snapshot(monkeypatch):
    import callo.slots as slots
    monkeypatch.setattr(slots, "DRAIN_S", 0.05)
    r = SlotRouter(4, 2)
    r.mark_draining(0)
    assert r.snapshot()["draining"] == [0]
    slot, _ = r.route_s1(deliberate=True)
    assert slot == 1
    r.release_s1(slot, deliberate=True)
    time.sleep(0.06)
    assert r.route_s1(deliberate=True)[0] == 0
    r.mark_draining(None)
    r.mark_draining(3)                                              # not reserved: nothing to drain
    assert r.counters["s1_draining"] == 1


def test_a_draining_slot_sorts_after_any_in_flight_count_not_tied_with_one_readout():
    r = SlotRouter(4, 2)
    r.acquire_s1(0)                                                # slot 0: one budget-0 readout in flight
    r.mark_draining(1)                                             # slot 1: draining, nothing in flight
    slot, _ = r.route_s1(deliberate=True)
    assert slot == 1                                               # in-flight count first (0 < 1), draining only after
    r.release_s1(slot, deliberate=True)
    r2 = SlotRouter(4, 2)
    r2.acquire_s1(1)                                               # slot 1: one in flight
    r2.mark_draining(0)                                            # slot 0: draining, nothing in flight
    r2.acquire_s1(0)                                               # ... and now one in flight too: a real tie
    assert r2.route_s1(deliberate=True)[0] == 1                    # draining goes after the tie, not by index


def test_route_quiet_takes_a_quiet_reserved_slot_atomically_or_returns_none():
    r = SlotRouter(4, 2)
    r.route_s1(deliberate=True)                                    # slot 0 thinks
    assert r.route_quiet(exclude=0) == 1 and r.snapshot()["inflight"]["1"] == 1
    r.release_s1(1)
    assert r.route_quiet(exclude=1) is None                        # slot 0 thinks, slot 1 is the excluded one
    r.route_s1(deliberate=True)                                    # slot 1 thinks too
    assert r.route_quiet(exclude=0) is None and r.route_quiet() is None


def test_section_check_keeps_the_ticket_in_place_and_counts_one_wait():
    from callo.slots import Section
    sec = Section()
    assert sec.acquire(exclusive=True, timeout=1)
    order: list[str] = []
    calls: list[int] = []

    def waiter(name, check_after=None):
        if sec.acquire(exclusive=True, timeout=5, check_after=check_after,
                       on_check=lambda: calls.append(1) or False):
            order.append(name)
            sec.release(exclusive=True)

    b = threading.Thread(target=waiter, args=("B", 0.05))
    b.start()
    time.sleep(0.02)
    c = threading.Thread(target=waiter, args=("C",))
    c.start()
    time.sleep(0.2)                                                # B's check fired at 50 ms and kept waiting
    assert calls == [1] and sec.snapshot()["queued"] == 2
    sec.release(exclusive=True)
    b.join()
    c.join()
    assert order == ["B", "C"] and sec.waits == 2                  # one wait each, not two for B


def test_section_check_runs_before_a_timeout_that_the_wakeup_overslept():
    """A wakeup that arrives past both the check time and the timeout still tries the reroute before it gives up
    (the s1_timeout <= check_after edge): on_check runs once, then the wait ends False either way."""
    from callo.slots import Section
    sec = Section()
    assert sec.acquire(exclusive=True, timeout=1)
    for verdict in (False, True):
        calls: list[int] = []
        assert sec.acquire(exclusive=True, timeout=0.0, check_after=0.0,
                           on_check=lambda c=calls, v=verdict: c.append(1) or v) is False
        assert calls == [1] and sec.snapshot()["queued"] == 0
    sec.release(exclusive=True)


def test_route_quiet_honours_draining_foreign_tasks_and_the_snapshot():
    r = SlotRouter(4, 3)
    r.route_s1(deliberate=True)                                    # slot 0 thinks
    r.mark_draining(1)
    t_cancel = time.monotonic()
    assert r.route_quiet(exclude=0) == 2                           # slot 1 drains: slot 2 is the only eligible one
    r.release_s1(2)
    foreign = {1, 2}.__contains__                                  # llama-server shows slots 1, 2 busy (1: still draining)
    assert r.route_quiet(foreign, as_of=time.monotonic(), exclude=0) is None
    assert r.route_quiet(lambda s: False, as_of=t_cancel - 1.0, exclude=0) == 2       # snapshot older than the cancel
    r.release_s1(2)
    assert r.route_quiet(lambda s: False, as_of=time.monotonic(), exclude=0) == 1     # a later idle snapshot has drained 1


def test_section_check_runs_with_the_section_lock_released_so_snapshot_cannot_deadlock():
    """The lock-order regression: on_check takes the router's lock, router.snapshot() reads the sections. While
    on_check is held open, snapshot() must finish (it takes the router lock, then must not wait for the section)."""
    r = SlotRouter(4, 2)
    sec = r.section(0)
    assert sec.acquire(exclusive=True, timeout=1)
    in_check, go = threading.Event(), threading.Event()
    out: dict = {}

    def check():
        in_check.set()
        go.wait(5)
        return r.route_quiet(exclude=0) is not None

    def question():
        out["q"] = sec.acquire(exclusive=True, timeout=5, check_after=0.02, on_check=check)

    def stats():
        in_check.wait(5)
        out["snap"] = r.snapshot()
        out["sec"] = sec.snapshot()

    tq, ts = threading.Thread(target=question, daemon=True), threading.Thread(target=stats, daemon=True)
    tq.start()
    ts.start()
    ts.join(2)
    stuck = ts.is_alive()
    go.set()
    tq.join(3)
    ts.join(3)
    assert not stuck and not tq.is_alive() and not ts.is_alive()
    assert out["q"] is False and out["snap"]["sections"]["0"]["queued"] == 1
    sec.release(exclusive=True)
    r.release_s1(1)
