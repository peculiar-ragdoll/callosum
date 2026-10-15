"""Pick preferences: the preference (`--latency`, `--limit`, `--price`, `--max-latency`, `--min-accuracy`,
`--max-memory`), its spellings, the canonical shell-safe form every printed limit takes, and the one validator."""
import argparse
import math
import shlex

import pytest

from callo.errors import InvalidInput
from callo.think import Limit, Preference, limit_text, parse_limit, parse_price, preference_from_args


@pytest.mark.parametrize("text", ["p99=2", "p99=2s", "p99=2000ms", "p99<=2s", " p99 <= 2s", "p99=2.0s"])
def test_every_spelling_of_a_limit_is_the_same_limit(text):
    assert parse_limit(text) == Limit("p99", 2000.0)


@pytest.mark.parametrize("stat", ["mean", "p50", "p95", "p99", "max"])
def test_every_statistic_takes_a_limit(stat):
    assert parse_limit(f"{stat}=6s") == Limit(stat, 6000.0, "--limit")


@pytest.mark.parametrize("text, says", [
    ("p99", "--limit takes STAT=X, for example --limit p99=5s"),
    ("p90=2s", "--limit: unknown statistic p90 (mean, p50, p95, p99, max)"),
    ("p99<2s", "--limit takes STAT=X (an upper bound), for example --limit p99=5s"),
    ("p99>=2", "--limit takes STAT=X (an upper bound), for example --limit p99=5s"),
    ("p99>2", "--limit takes STAT=X (an upper bound), for example --limit p99=5s"),
    ("p99=2min", "--limit p99: X is seconds (5, 5s) or milliseconds (5000ms)"),
    ("p99=fast", "--limit p99: X is seconds (5, 5s) or milliseconds (5000ms)"),
    ("p99=0", "--limit p99: X must be a positive finite number"),
    ("p99=nan", "--limit p99: X must be a positive finite number"),
    ("p99=inf", "--limit p99: X must be a positive finite number"),
    ("p99=-1s", "--limit p99: X must be a positive finite number"),
    ("p99=1e306s", "--limit p99: X must be a positive finite number"),
    ("=2", "--limit takes STAT=X, for example --limit p99=5s"),
    ("p99<=2>", "--limit takes STAT=X (an upper bound), for example --limit p99=5s"),
    ("p99<=2<3", "--limit takes STAT=X (an upper bound), for example --limit p99=5s"),
])
def test_a_limit_that_does_not_parse_says_how_to_spell_it(text, says):
    with pytest.raises(InvalidInput) as e:
        parse_limit(text)
    assert str(e.value) == says


@pytest.mark.parametrize("le_ms, text", [(6000.0, "6s"), (300.0, "0.3s"), (2001.0, "2.001s"), (2500.0, "2.5s"),
                                         (1001.0, "1001ms"), (63.7, "63.7ms"), (350.0, "0.35s")])
def test_the_canonical_form_is_shell_safe_and_reads_back_as_the_same_limit(le_ms, text):
    lim = Limit("p99", le_ms)
    assert limit_text(le_ms) == text and lim.canonical == f"--limit p99={text}"
    argv = shlex.split(lim.canonical)
    assert argv == ["--limit", f"p99={text}"] and "<" not in lim.canonical and ">" not in lim.canonical
    assert parse_limit(argv[1]) == lim


@pytest.mark.parametrize("text, r", [("1", 1.0), ("1pp/s", 1.0), ("0.5pp/s", 0.5), ("0", 0.0), ("10pp/s", 10.0)])
def test_a_price_is_in_pp_per_second(text, r):
    assert parse_price(text) == r


@pytest.mark.parametrize("text, says", [("1pp/min", "--price is in pp/s: 1, 1pp/s or 0.5pp/s"),
                                        ("1%/s", "--price is in pp/s: 1, 1pp/s or 0.5pp/s"),
                                        ("x", "--price is in pp/s: 1, 1pp/s or 0.5pp/s"),
                                        ("-1", "--price must be a finite number >= 0"),
                                        ("nan", "--price must be a finite number >= 0"),
                                        ("inf", "--price must be a finite number >= 0")])
def test_a_price_that_does_not_parse_says_so(text, says):
    with pytest.raises(InvalidInput) as e:
        parse_price(text)
    assert str(e.value) == says


@pytest.mark.parametrize("kw, says", [
    ({"latency": "p99"}, "--latency must be mean, p50 or p95 (p99 and max are limits only: --limit p99=S)"),
    ({"latency": "max"}, "--latency must be mean, p50 or p95 (p99 and max are limits only: --limit p99=S)"),
    ({"price": -1.0}, "--price must be a finite number >= 0"),
    ({"price": math.inf}, "--price must be a finite number >= 0"),
    ({"price": 1.0, "min_accuracy": 0.9}, "--min-accuracy and a positive --price are two objectives; give one"),
    ({"max_latency_s": 0.0}, "--max-latency is in seconds and must be positive, got 0.0"),
    ({"min_accuracy": 1.5}, "--min-accuracy is a fraction from 0 to 1, got 1.5"),
    ({"max_latency_s": math.inf}, f"--max-latency is in seconds and must be positive, got {math.inf!r}"),
    ({"max_latency_s": 1e306}, "--max-latency is in seconds and must be positive, got 1e+306"),
    ({"max_gib": math.inf}, f"--max-memory is in GiB and must be positive, got {math.inf!r}"),
    ({"price": 1e7}, "--price is at most 1000000 pp/s"),
    ({"max_gib": -1.0}, "--max-memory is in GiB and must be positive, got -1.0"),
])
def test_the_validator_refuses_with_the_documented_messages(kw, says):
    with pytest.raises(InvalidInput) as e:
        Preference(**kw)
    assert str(e.value) == says


def test_a_zero_price_with_min_accuracy_is_the_default_rule_plus_min_accuracy():
    """The test is R > 0, not the flag's presence."""
    assert Preference(price=0.0, min_accuracy=0.9) == Preference(min_accuracy=0.9)


def test_max_latency_is_a_limit_on_the_pick_statistic_and_stays_visible():
    p = Preference(latency="p95", max_latency_s=3.0)
    assert p.limits == (Limit("p95", 3000.0, "--max-latency"),)
    assert p.as_dict()["limits"] == [{"stat": "p95", "le_ms": 3000.0, "from": "--max-latency"}]
    assert p.limits[0].le_ms == parse_limit("p95=3").le_ms               # the same as --limit p95=3
    both = Preference(limits=(parse_limit("p99=6s"), parse_limit("p99=5s")), max_latency_s=2.0)
    assert [lim.source for lim in both.limits] == ["--limit", "--limit", "--max-latency"]   # every limit applies


def test_the_record_round_trips_and_names_what_is_not_the_default():
    p = Preference(latency="mean", price=5.0, limits=(parse_limit("p99=6s"),), max_gib=40.0)
    assert p.as_dict() == {"latency": "mean", "price_pp_per_s": 5.0,
                           "limits": [{"stat": "p99", "le_ms": 6000.0, "from": "--limit"}],
                           "min_accuracy": None, "max_memory_gib": 40.0}
    assert Preference.from_dict(p.as_dict()) == p
    assert p.text() == "--price 5pp/s on mean, --limit p99=6s, --max-memory 40"
    assert Preference().text() == "" and Preference().is_default and not p.is_default
    assert Preference(latency="p50").text() == "--latency p50"
    assert Preference(min_accuracy=0.9, max_latency_s=2.0).text() == "--limit mean=2s, --min-accuracy 0.9"
    alias = Preference(max_latency_s=2.0)
    assert Preference.from_dict(alias.as_dict()) == alias
    # a pick block from before pick preferences: no `preference`, its own latency / max_latency_s / min_accuracy
    old = Preference.from_dict(None, legacy={"latency": "p50", "max_latency_s": 3.0, "min_accuracy": None})
    assert old == Preference(latency="p50", max_latency_s=3.0)


def _args(**kw) -> argparse.Namespace:
    base = {"latency": None, "limit": None, "price": None, "max_latency": None, "min_accuracy": None,
            "max_memory": None}
    return argparse.Namespace(**{**base, **kw})


def test_preference_from_args_parses_and_validates_the_flags():
    assert preference_from_args(_args()) == Preference()
    p = preference_from_args(_args(limit=["p99=6s", "max<=8s"], price="2pp/s", max_memory=10.0))
    assert p == Preference(price=2.0, limits=(Limit("p99", 6000.0), Limit("max", 8000.0)), max_gib=10.0)
    with pytest.raises(InvalidInput, match="--latency must be mean, p50 or p95"):
        preference_from_args(_args(latency="p99", limit=["nonsense"]))   # the latency first, as the flag table lists it
    with pytest.raises(InvalidInput, match="two objectives"):
        preference_from_args(_args(price="1", min_accuracy=0.8))
    assert preference_from_args(_args(price="0", min_accuracy=0.8)) == Preference(min_accuracy=0.8)
    assert preference_from_args(argparse.Namespace()) == Preference()    # a namespace without the flags: the default


def test_price_edge_cases():
    assert str(parse_price("-0")) == "0.0" and str(parse_price(" 2 pp/s ")) == "2.0"
    with pytest.raises(InvalidInput) as e:
        parse_price("1e9")
    assert str(e.value) == "--price is at most 1000000 pp/s"
    with pytest.raises(InvalidInput):
        parse_price("1e999")


@pytest.mark.parametrize("d", [
    {"min_accuracy": "0.9"}, {"limits": [{"stat": "p99"}]}, {"limits": [{"le_ms": 5.0}]},
    {"limits": [{"stat": "p90", "le_ms": 5.0}]}, {"limits": [{"stat": "p99", "le_ms": -5.0}]},
    {"limits": [{"stat": "p99", "le_ms": "fast"}]}, {"limits": [5]}, {"max_memory_gib": "40"},
    {"price_pp_per_s": "x"}, {"latency": "p99"},
])
def test_a_malformed_recorded_preference_is_refused_not_crashed(d):
    with pytest.raises(InvalidInput):
        Preference.from_dict(d)
