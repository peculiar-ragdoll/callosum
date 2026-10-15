# Adapted from the author's benchmark harness (tests/test_system1_render.py @ f886c50f).
# Edits: sys.path shim removed, import retargeted.
"""The rendering port must be byte-identical to Kev's: every arm has to see the same information."""
from callo.render import choice_confidence, option_text, question_keys, render, score_confidence


def test_render_object_state_is_key_value_lines():
    state = {"subject": "security studies", "question": "Which did not emerge?"}
    assert render(state) == "subject: security studies\nquestion: Which did not emerge?"


def test_render_nested_and_list():
    assert render({"a": {"b": 1}, "c": [1, {"d": 2}]}) == "a:\n  b: 1\nc:\n  - 1\n  - d: 2"


def test_render_none_and_scalars():
    assert render(None) == "" and render(3) == "3" and render(True) == "True" and render("x") == "x"


def test_option_text():
    assert option_text("joy", None) == "joy"
    assert option_text("joy", "") == "joy"
    assert option_text("yes", "Contains insults") == "yes: Contains insults"


def test_question_keys():
    assert question_keys("noul", None) == ["false", "true"]
    assert question_keys("choice", {"b": 1, "a": 2}) == ["b", "a"]
    assert question_keys("score", ["lo", "mid", "hi"]) == ["0", "1", "2"]


def test_confidences_match_kev_reference_values():
    assert choice_confidence([0.25, 0.25, 0.25, 0.25]) == 0.0
    assert choice_confidence([1.0, 0.0]) == 1.0
    assert abs(choice_confidence([0.7, 0.2, 0.1]) - (0.7 - 1 / 3) / (2 / 3)) < 1e-12
    # L=3: D = mean |i - 1| = 2/3; p all on level 0 -> 1.0; uniform -> 0
    assert score_confidence([1.0, 0.0, 0.0]) == 1.0
    assert score_confidence([1 / 3, 1 / 3, 1 / 3]) == 0.0
    # p=[0.5,0.5,0]: mode=0, E|i-0| = 0.5, D=2/3 -> 1 - 0.75 = 0.25
    assert abs(score_confidence([0.5, 0.5, 0.0]) - 0.25) < 1e-12
