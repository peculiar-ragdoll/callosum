"""`callo calibrate` must recover a known temperature from probabilities served at T = 1."""
import math
import random

import pytest

from callo.calibrate import (
    GRID,
    GRID_HI,
    GRID_LO,
    GRID_POINTS,
    ece,
    fit_temperature,
    flat_fit,
    limit_note,
    nll,
    request_from_record,
    row_from_answer,
    temper,
)
from callo.errors import InvalidInput


def _softmax(z):
    top = max(z)
    e = [math.exp(v - top) for v in z]
    s = sum(e)
    return [x / s for x in e]


def _rows(true_t: float, n: int = 4000, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = [rng.gauss(0.0, 2.0) for _ in range(4)]
        served = _softmax(z)                                   # what callo returns at T = 1
        label = rng.choices(range(4), weights=_softmax([v / true_t for v in z]))[0]
        rows.append((served, label))
    return rows


# the grid before 0.3.0's widening, computed with the old constants exactly as calibrate.py computed it
OLD_GRID = [math.exp(math.log(0.25) + i * (math.log(4.0) - math.log(0.25)) / (121 - 1)) for i in range(121)]


def _old_fit(rows):
    losses = [nll(rows, t) for t in OLD_GRID]
    return OLD_GRID[min(range(len(OLD_GRID)), key=losses.__getitem__)]


def test_grid_is_181_log_spaced_points_on_quarter_to_sixteen_with_the_old_grid_as_its_bitwise_prefix():
    assert len(GRID) == GRID_POINTS == 181 and GRID[0] == GRID_LO == 0.25 and abs(GRID[-1] - 16.0) < 1e-12
    assert GRID_HI == GRID[-1]
    assert [t.hex() for t in GRID[:121]] == [t.hex() for t in OLD_GRID]          # bitwise, not approximately
    ratios = [b / a for a, b in zip(GRID, GRID[1:], strict=False)]
    assert max(ratios) - min(ratios) < 1e-9 and abs(GRID[60] - 1.0) < 1e-9 and GRID[120] == 4.0


def test_an_overconfident_row_set_fits_past_four_not_at_the_old_ceiling():
    rows = _rows(8.0)
    t = fit_temperature(rows)
    assert _old_fit(rows) == 4.0 and 6.5 <= t <= 10.0 and t in GRID
    assert nll(rows, t) < nll(rows, 4.0)


@pytest.mark.parametrize(("true_t", "seed"), [(0.5, 2), (1.0, 1), (2.0, 0), (3.0, 3)])
def test_fits_inside_the_old_range_are_unchanged(true_t, seed):
    rows = _rows(true_t, seed=seed)
    assert fit_temperature(rows).hex() == _old_fit(rows).hex()


def test_the_limit_note_appears_only_at_the_grid_s_edges():
    top, bottom = limit_note(GRID[-1]), limit_note(GRID[0])
    assert top == "T is at the search limit (16): the probabilities are still overconfident after calibration"
    assert bottom == ("T is at the search limit (0.25): the fit wants sharper probabilities than the grid allows "
                      "(common on a small or perfectly separated set)")
    assert limit_note(GRID[-1], at=" at tiel@0") == ("T is at the search limit (16) at tiel@0: the probabilities "
                                                    "are still overconfident after calibration")
    assert all(limit_note(t) is None for t in (GRID[1], GRID[-2], 1.0, 4.0, GRID[120]))


def test_a_flat_fit_keeps_its_tie_break_t_but_gets_no_limit_note():
    """Uniform rows score the same NLL at every T: the fit still returns GRID[0] (the first minimum, as before), but
    that is a tie, not a fit wanting sharper probabilities, so the note is suppressed."""
    flat = [([0.5, 0.5], 0), ([0.5, 0.5], 1), ([0.25] * 4, 2)]
    assert fit_temperature(flat) == GRID[0] and flat_fit(flat)
    assert limit_note(GRID[0], flat=True) is None and limit_note(GRID[-1], flat=True) is None
    separated = [([0.9, 0.1], 0), ([0.2, 0.8], 1)]
    assert fit_temperature(separated) == GRID[0] and not flat_fit(separated) and limit_note(GRID[0]) is not None
    assert not flat_fit(_rows(8.0, n=200))


def test_temper_is_identity_at_one_and_flattens_above_one():
    p = [0.7, 0.2, 0.1]
    assert all(abs(a - b) < 1e-12 for a, b in zip(temper(p, 1.0), p, strict=True))
    q = temper(p, 2.0)
    assert q[0] < p[0] and q[0] > q[1] > q[2] and abs(sum(q) - 1) < 1e-12


def test_fit_recovers_a_known_temperature_and_improves_nll_and_ece():
    rows = _rows(2.0)
    t = fit_temperature(rows)
    assert 1.85 <= t <= 2.15
    assert nll(rows, t) < nll(rows, 1.0) and ece(rows, t) < ece(rows, 1.0)
    assert t in GRID


def test_fit_recovers_one_when_already_calibrated():
    assert 0.93 <= fit_temperature(_rows(1.0, seed=1)) <= 1.07


def test_fit_rejects_no_rows():
    with pytest.raises(InvalidInput):
        fit_temperature([])


def test_row_from_answer_for_each_question_type():
    probs, y = row_from_answer({"type": "noul", "label": True}, {"type": "noul", "noul": 0.8})
    assert probs == pytest.approx([0.2, 0.8]) and y == 1                     # 1.0 - 0.8 is 0.19999999999999996
    probs, y = row_from_answer({"type": "noul", "label": False}, {"type": "noul", "noul": 0.8})
    assert probs == pytest.approx([0.2, 0.8]) and y == 0
    probs, y = row_from_answer({"type": "choice", "label": "d"},
                               {"type": "choice", "probabilities": {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}})
    assert probs == [0.1, 0.2, 0.3, 0.4] and y == 3
    probs, y = row_from_answer({"type": "score", "label": 2}, {"type": "score", "probabilities": {"0": 0.5, "1": 0.3, "2": 0.2}})
    assert y == 2


def test_request_from_record_keeps_only_the_api_fields():
    rec = {"state": {"q": "x"}, "questions": {"answer": {"type": "choice", "instructions": "i", "criteria": {"a": "A"},
                                                         "label": "a", "src": "mmlu"}}, "_meta": {"id": "m/1"}}
    assert request_from_record(rec) == {"state": {"q": "x"}, "model": "kev-latest",
                                        "questions": {"answer": {"type": "choice", "instructions": "i", "criteria": {"a": "A"}}}}


def test_request_from_record_keeps_images_and_context():
    rec = {"state": "", "images": ["data:image/png;base64,QUJD"], "context": "orders",
           "questions": {"q": {"type": "choice", "criteria": {"a": None}, "label": "a", "src": "demo"}},
           "_meta": {"id": "d/1"}}
    assert request_from_record(rec) == {"state": "", "model": "kev-latest", "images": ["data:image/png;base64,QUJD"],
                                        "context": "orders", "questions": {"q": {"type": "choice",
                                                                                  "criteria": {"a": None}}}}


def test_row_from_answer_rejects_a_label_missing_from_the_answer_or_a_malformed_answer():
    with pytest.raises(InvalidInput, match="label"):
        row_from_answer({"type": "choice", "label": "z"}, {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}})
    with pytest.raises(InvalidInput):
        row_from_answer({"type": "choice", "label": "a"}, {"type": "choice"})          # no probabilities
    with pytest.raises(InvalidInput):
        row_from_answer({"type": "noul", "label": "maybe"}, {"type": "noul", "noul": 0.5})


def test_request_from_record_keeps_audio_and_videos():
    """v0.3: labelled audio and video records calibrate exactly as they are served."""
    rec = {"state": "", "audio": ["data:audio/wav;base64,QUJD"], "videos": ["data:video/mp4;base64,QUJD"],
           "questions": {"q": {"type": "noul", "label": "true"}}}
    req = request_from_record(rec)
    assert req["audio"] == ["data:audio/wav;base64,QUJD"] and req["videos"] == ["data:video/mp4;base64,QUJD"]
