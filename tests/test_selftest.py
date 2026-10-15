"""The built-in `2 + 2 = ?` item and the mechanical answer-slot check, on canned /apply-template output."""
import math
import string

from callo.engine import SELF_TEST_BODY, Engine, SurfaceSets, self_test
from callo.prompt import TemplateClient

TOKENS = list(string.ascii_uppercase) + ["Ġ" + c for c in string.ascii_uppercase] + ["**", "Ġthe", "x"]
QWEN_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _template(prompt_for):
    def post(url, payload):
        return {"prompt": prompt_for(payload["messages"])}
    return TemplateClient("http://x", http_post=post)


def _chatml(msgs):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + QWEN_TAIL


class Scorer:
    def __init__(self, top):
        self.top, self.calls = top, []

    def tokenize(self, text, *, add_special=True):
        return [100, 101, 102]

    def next_logprobs(self, tokens, *, top_k, id_slot=None, delimiter=None):
        self.calls.append(id_slot)
        return self.top


def test_self_test_passes_on_a_model_that_answers_4():
    sc = Scorer({1: math.log(0.9), 0: math.log(0.05), 52: math.log(0.03)})      # B = "4"
    r = self_test(Engine(sc, _template(_chatml), SurfaceSets(TOKENS, "gpt2")), id_slot=0)
    assert r.ok and r.answer == "4" and r.option_mass > 0.9 and r.answer_slot_ok and r.error is None
    assert sc.calls == [0]
    assert SELF_TEST_BODY["questions"]["q"]["criteria"] == {"3": None, "4": None}
    assert any("answer=4" in line for line in r.lines())


def test_self_test_reports_the_wrong_answer_and_top_tokens_without_raising():
    sc = Scorer({0: math.log(0.6), 1: math.log(0.1), 53: math.log(0.2), 52: math.log(0.05)})   # A = "3"
    r = self_test(Engine(sc, _template(_chatml), SurfaceSets(TOKENS, "gpt2")))
    assert not r.ok and r.answer == "3"
    assert r.top_tokens[0] == ("A", math.log(0.6)) and r.top_tokens[1] == ("Ġthe", math.log(0.2))
    assert any("top-5" in line for line in r.lines())


def test_self_test_flags_low_option_mass():
    sc = Scorer({53: math.log(0.5), 52: math.log(0.45), 1: math.log(0.03), 0: math.log(0.01)})
    r = self_test(Engine(sc, _template(_chatml), SurfaceSets(TOKENS, "gpt2")))
    assert not r.ok and r.option_mass < 0.5 and any("option mass" in line for line in r.lines())


def test_answer_slot_check_warns_but_does_not_fail_the_self_test():
    def misplaced(msgs):          # an assistant turn in history renders inside a thinking block: a misplaced slot
        def one(m):
            if m["role"] == "assistant":
                return f"<|im_start|>assistant\n<think>\nthinking {m['content']}<|im_end|>\n"
            return f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return "".join(one(m) for m in msgs) + QWEN_TAIL

    sc = Scorer({1: math.log(0.9), 0: math.log(0.05)})
    r = self_test(Engine(sc, _template(misplaced), SurfaceSets(TOKENS, "gpt2")))
    assert r.answer == "4" and r.answer_slot_ok is False and "diverges" in r.answer_slot_detail
    assert any("WARNING" in line and "answer slot" in line for line in r.lines())


def test_self_test_survives_a_dead_template_endpoint():
    def boom(url, payload):
        raise ConnectionRefusedError("nope")

    r = self_test(Engine(Scorer({}), TemplateClient("http://x", http_post=boom), SurfaceSets(TOKENS, "gpt2")))
    assert not r.ok and r.error and "ConnectionRefusedError" in r.error
