"""callo.guard: the five guard conditions under every kv layout and set kind, the predicted slot, the records, and
messages that name only fixes that work."""
import pytest
from fake_callo import default_stats

from callo.guard import (
    EXEMPT_ROUTES,
    FAILED_LINE,
    LANE,
    Mark,
    evaluate_state,
    message,
    predicted_slot,
    record,
    tokens_matter,
)

URL = "http://127.0.0.1:8180"


def _stats(**kw):
    s = default_stats()
    s.update(kw)
    return s


def _slots(**per_slot):
    """slots 0..3; per_slot maps 's2' -> dict of fields for slot 2."""
    out = [{"id": i, "is_processing": False} for i in range(4)]
    for k, fields in per_slot.items():
        out[int(k[1:])].update(fields)
    return out


@pytest.mark.parametrize(("router", "slot"), [
    ({"total_slots": 1, "reserved": [], "lanes": {}}, 0),                      # -np 1: the only slot
    ({"total_slots": 4, "reserved": [0], "lanes": {}}, 0),
    ({"total_slots": 4, "reserved": [0, 1], "lanes": {}}, 0),
    ({"total_slots": 4, "reserved": [0, 1], "lanes": {"agent": 0}}, 1),
    ({"total_slots": 4, "reserved": [0, 1], "lanes": {LANE: 1, "agent": 0}}, 1),
    ({"total_slots": 4, "reserved": [0, 1], "lanes": {"a": 0, "b": 1}}, None),  # the LRU victim is not published
])
def test_predicted_slot(router, slot):
    assert predicted_slot(_stats(router=router)) == slot


def test_a_clean_server_passes():
    v = evaluate_state(_stats(), slot=0, kinds=["plain"], prev=None, own=0)
    assert v.failed == () and v.held == {} and v.mark == Mark(0, {})


def test_condition_1_unobservable_slots():
    assert evaluate_state(_stats(slots=None), slot=0, kinds=["plain"], prev=None, own=0).failed == ("slots_unobservable",)


def test_condition_2_another_slot_processing_but_not_calibrates_own():
    assert evaluate_state(_stats(slots=_slots(s2={"is_processing": True})), slot=0, kinds=["plain"], prev=None,
                          own=0).failed == ("foreign_processing",)
    v = evaluate_state(_stats(slots=_slots(s0={"is_processing": True})), slot=0, kinds=["plain"], prev=None, own=0)
    assert v.failed == ()


@pytest.mark.parametrize(("kv", "kinds", "fails"), [
    ("unified", ["plain"], True), (None, ["plain"], True),                       # attach mode is treated as unified
    ("per_slot", ["plain"], False),                                              # per-slot text is bit-exact
    ("per_slot", ["plain", "image"], True), ("per_slot", ["context"], True),     # measured on text only
    ("per_slot", ["plain", "audio"], True), ("per_slot", ["mixed"], True),       # any non-plain kind
])
def test_condition_3_another_slot_holding_tokens(kv, kinds, fails):
    v = evaluate_state(_stats(kv=kv, slots=_slots(s3={"n_prompt_tokens": 17}, s0={"n_prompt_tokens": 900})), slot=0,
                       kinds=kinds, prev=None, own=0)
    assert ("foreign_tokens" in v.failed) is fails and v.held == {3: 17}         # held is recorded either way
    assert tokens_matter(kv, kinds) is fails


def test_an_absent_n_prompt_tokens_counts_as_zero_and_an_unknown_slot_checks_every_slot():
    assert evaluate_state(_stats(slots=_slots()), slot=0, kinds=["plain"], prev=None, own=0).failed == ()
    v = evaluate_state(_stats(slots=_slots(s0={"n_prompt_tokens": 5})), slot=None, kinds=["plain"], prev=None, own=0)
    assert v.failed == ("foreign_tokens",) and v.held == {0: 5}


def test_condition_4_foreign_s1_readouts_since_the_last_check():
    prev = Mark(10, {"/v1/systemone": 10, "/callo/stats": 1})
    st = _stats(s1={"readouts": 65, "n": 65}, routes={"/v1/systemone": 65, "/callo/stats": 2})
    v = evaluate_state(st, slot=0, kinds=["plain"], prev=prev, own=50)
    assert v.failed == ("foreign_s1",) and v.foreign_readouts == 5
    assert evaluate_state(st, slot=0, kinds=["plain"], prev=prev, own=55).failed == ()
    assert evaluate_state(st, slot=0, kinds=["plain"], prev=None, own=0).failed == ()   # no delta at the first check


def test_condition_4_tolerates_one_extra_readout_per_retried_request():
    """A connection error after callo already answered makes calibrate retry; callo counted both."""
    prev = Mark(10, {"/v1/systemone": 10})
    st = _stats(s1={"readouts": 62, "n": 62}, routes={"/v1/systemone": 62})
    assert evaluate_state(st, slot=0, kinds=["plain"], prev=prev, own=50).failed == ("foreign_s1",)
    v = evaluate_state(st, slot=0, kinds=["plain"], prev=prev, own=50, slack=2)
    assert v.failed == () and v.slack == 2 and record(1, 0, 50, v)["slack"] == 2
    v = evaluate_state(st, slot=0, kinds=["plain"], prev=prev, own=50, slack=1)
    assert v.failed == ("foreign_s1",) and v.foreign_readouts == 1


def test_read_only_routes_do_not_refuse_but_state_changing_ones_do():
    """A /health poller or a dashboard is not server state; a chat completion or a slot action is."""
    prev = Mark(5, {"/v1/systemone": 5, "/callo/stats": 1})
    quiet = {"/v1/systemone": 10, "/callo/stats": 2, "/health": 3, "/v1/health": 1, "/props": 2, "/slots": 4,
             "/metrics": 1, "/models": 1, "/v1/models": 1, "/tokenize": 2, "/detokenize": 1, "/apply-template": 1}
    assert set(quiet) <= EXEMPT_ROUTES
    st = _stats(s1={"readouts": 10, "n": 10}, routes=quiet)
    assert evaluate_state(st, slot=0, kinds=["plain"], prev=prev, own=5).failed == ()
    for route in ("/v1/chat/completions", "/slots/3", "/completion"):
        busy = _stats(s1={"readouts": 10, "n": 10}, routes={**quiet, route: 1})
        v = evaluate_state(busy, slot=0, kinds=["plain"], prev=prev, own=5)
        assert v.failed == ("foreign_routes",) and v.grew == (route,)
        assert route in message(v, stats=busy, kinds=["plain"], url=URL)


def test_condition_5_compares_against_pre_existing_route_counters():
    """A counter that already stood at 3 before the check is not traffic since the check; only growth past it is."""
    prev = Mark(5, {"/v1/systemone": 5, "/v1/chat/completions": 3})
    same = _stats(s1={"readouts": 10, "n": 10}, routes={"/v1/systemone": 10, "/v1/chat/completions": 3})
    assert evaluate_state(same, slot=0, kinds=["plain"], prev=prev, own=5).failed == ()
    more = _stats(s1={"readouts": 10, "n": 10}, routes={"/v1/systemone": 10, "/v1/chat/completions": 4})
    v = evaluate_state(more, slot=0, kinds=["plain"], prev=prev, own=5)
    assert v.failed == ("foreign_routes",) and v.grew == ("/v1/chat/completions",)


def test_the_record_is_what_the_profile_keeps():
    v = evaluate_state(_stats(slots=_slots(s3={"n_prompt_tokens": 17})), slot=0, kinds=["plain"], prev=None, own=0)
    r = record(4, 1, 250, v, at="2026-10-03T12:00:00Z")
    assert r == {"index": 4, "session": 1, "at": "2026-10-03T12:00:00Z", "readouts": 250, "failed": ["foreign_tokens"],
                 "held": {"3": 17}, "processing": [], "foreign_readouts": 0, "routes_grew": [], "slack": 0}
    assert record(0, 0, 0, v)["at"].endswith("Z")


def _msg(stats, kinds=("plain",), slot=0):
    v = evaluate_state(stats, slot=slot, kinds=kinds, prev=None, own=0)
    assert v.failed
    return message(v, stats=stats, kinds=kinds, url=URL)


def test_messages_name_only_fixes_that_work():
    held = _slots(s3={"n_prompt_tokens": 300})
    launch = _msg(_stats(kv="unified", slots=held))                              # launch mode, erase supported
    assert launch.startswith("refusing to calibrate:") and "slot 3 (300 tokens)" in launch
    assert f"curl -X POST '{URL}/slots/3?action=erase'" in launch and "--no-kvu" in launch
    assert "no agent traffic" in launch and "--allow-busy" in launch
    attach = _msg(_stats(kv=None, slots=held, probe={"residue": [3, 0], "erase": "unsupported"}))
    assert "--no-kvu" not in attach and "curl" not in attach                     # neither works in attach mode here
    assert "startup-probe" in attach and "--slot-save-path" in attach            # the residue is named
    image = _msg(_stats(kv="per_slot", slots=held), kinds=("plain", "image"))
    assert "--no-kvu" not in image                                               # per-slot does not cover images
    video = _msg(_stats(kv="unified", slots=held), kinds=("plain", "video"))
    assert "--no-kvu" not in video and "curl" in video                           # --no-kvu: plain-only sets
    blind = _msg(_stats(slots=None))
    assert "/slots" in blind and "curl" not in blind and "--no-kvu" not in blind
    busy = _msg(_stats(slots=_slots(s2={"is_processing": True})))
    assert "slot 2" in busy and "processing" in busy and "curl" not in busy


def test_the_erase_curl_is_gated_on_probe_erase_not_on_the_kv_layout():
    """The curl works exactly when llama-server accepted the probe's erase (it has --slot-save-path), whatever kv is."""
    held = _slots(s3={"n_prompt_tokens": 300})
    unsupported = _msg(_stats(kv="unified", slots=held, probe={"residue": [], "erase": "unsupported"}))
    assert "curl" not in unsupported                                             # launch mode, but the erase is a 501
    attach_ok = _msg(_stats(kv=None, slots=held, probe={"residue": [], "erase": "ok"}))
    assert f"curl -X POST '{URL}/slots/3?action=erase'" in attach_ok              # attach mode with --slot-save-path


def test_per_slot_kv_names_the_set_non_plain_not_non_text():
    """`context` is text, so the per-slot refusal says the set is non-plain."""
    text = _msg(_stats(kv="per_slot", slots=_slots(s3={"n_prompt_tokens": 9})), kinds=("plain", "context"))
    assert "per-slot KV with a non-plain set" in text and "non-text" not in text


def test_a_bool_n_prompt_tokens_is_not_a_token_count():
    """isinstance(True, int) holds, so True would read as 1 held token."""
    v = evaluate_state(_stats(slots=_slots(s3={"n_prompt_tokens": True})), slot=0, kinds=["plain"], prev=None, own=0)
    assert v.failed == () and v.held == {}


@pytest.mark.parametrize("kv", [None, "unified"])
def test_a_residue_only_refusal_does_not_blame_agent_traffic(kv):
    """When every held slot is callo's own probe residue, stopping agent traffic cannot clear it."""
    stats = _stats(kv=kv, slots=_slots(s3={"n_prompt_tokens": 16}), probe={"residue": [3, 0], "erase": "unsupported"})
    text = _msg(stats)
    fixes = text.split("fixes:", 1)[1]
    assert "no agent traffic" not in fixes and "startup-probe" in text
    if kv is None:                                                    # attach mode: the flag is the fix
        assert "restart llama-server with --slot-save-path DIR" in fixes
    else:                                                             # launch mode already passed the flag
        assert "--slot-save-path DIR" not in fixes and "restart `callo serve`" in fixes


def test_residue_plus_another_held_slot_still_names_agent_traffic():
    stats = _stats(kv=None, slots=_slots(s3={"n_prompt_tokens": 16}, s2={"n_prompt_tokens": 300}),
                   probe={"residue": [3, 0], "erase": "unsupported"})
    fixes = _msg(stats).split("fixes:", 1)[1]
    assert "no agent traffic" in fixes and "restart llama-server with --slot-save-path DIR" in fixes


def test_the_message_ends_with_the_failed_conditions_for_machines():
    both = _stats(slots=_slots(s2={"is_processing": True, "n_prompt_tokens": 4}))
    text = _msg(both)
    assert text.splitlines()[-1] == "(failed conditions: foreign_processing, foreign_tokens)"
    assert FAILED_LINE.search(text).group(1).split(", ") == ["foreign_processing", "foreign_tokens"]
    assert FAILED_LINE.search("x\n(failed conditions: foreign_s1)\n").group(1) == "foreign_s1"    # a digit
