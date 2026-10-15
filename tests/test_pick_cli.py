"""Pick preferences: the flags calibrate and compare share, their one validator (every refusal is exit 2
with its message in both commands), the preference calibrate records, and the menu's report passing it through."""
import dataclasses
import json

import pytest
from test_compare import A
from test_menu import HOST, _opts, _stub_calibrate, _stub_launcher, _two, write_items

import callo.menu as menu
from callo.calibrate_think import ThinkSettings, settings_from_args
from callo.cli import build_parser, main
from callo.menu import Driver
from callo.think import Limit, Preference

ROWS = [
    (["--latency", "p99"], "--latency must be mean, p50 or p95 (p99 and max are limits only: --limit p99=S)"),
    (["--latency", "max"], "--latency must be mean, p50 or p95 (p99 and max are limits only: --limit p99=S)"),
    (["--limit", "p99"], "--limit takes STAT=X, for example --limit p99=5s"),
    (["--limit", "p90=2s"], "--limit: unknown statistic p90 (mean, p50, p95, p99, max)"),
    (["--limit", "p99<2s"], "--limit takes STAT=X (an upper bound), for example --limit p99=5s"),
    (["--limit", "p99>=2"], "--limit takes STAT=X (an upper bound), for example --limit p99=5s"),
    (["--limit", "p99>2"], "--limit takes STAT=X (an upper bound), for example --limit p99=5s"),
    (["--limit", "p99=2min"], "--limit p99: X is seconds (5, 5s) or milliseconds (5000ms)"),
    (["--limit", "p99=fast"], "--limit p99: X is seconds (5, 5s) or milliseconds (5000ms)"),
    (["--limit", "p99=0"], "--limit p99: X must be a positive finite number"),
    (["--limit", "p99=nan"], "--limit p99: X must be a positive finite number"),
    (["--limit", "p99=inf"], "--limit p99: X must be a positive finite number"),
    (["--price", "1pp/min"], "--price is in pp/s: 1, 1pp/s or 0.5pp/s"),
    (["--price", "1%/s"], "--price is in pp/s: 1, 1pp/s or 0.5pp/s"),
    (["--price", "x"], "--price is in pp/s: 1, 1pp/s or 0.5pp/s"),
    (["--price=-1"], "--price must be a finite number >= 0"),
    (["--price", "nan"], "--price must be a finite number >= 0"),
    (["--price", "inf"], "--price must be a finite number >= 0"),
    (["--price", "1", "--min-accuracy", "0.9"], "--min-accuracy and a positive --price are two objectives; give one"),
]


@pytest.fixture
def data(tmp_path):
    from test_calibrate_think import write_data
    return write_data(tmp_path / "d.jsonl", 4)


@pytest.mark.parametrize("flags, says", ROWS)
def test_every_validation_row_is_exit_2_with_its_message_in_calibrate(data, capsys, flags, says):
    assert main(["calibrate", "--think-budgets", "--data", str(data), "--dry-run", *flags]) == 2
    assert capsys.readouterr().err == f"callo: {says}\n"


@pytest.mark.parametrize("flags, says", ROWS)
def test_every_validation_row_is_exit_2_with_its_message_in_compare(tmp_path, capsys, flags, says):
    path = tmp_path / "a.think.json"
    path.write_text(json.dumps(A))
    assert main(["compare", str(path), *flags]) == 2
    assert capsys.readouterr().err == f"callo: {says}\n"


@pytest.mark.parametrize("flags", [["--limit", "p99=2"], ["--limit", "p99=2s"], ["--limit", "p99=2000ms"],
                                   ["--limit", "p99<=2s"], ["--price", "0", "--min-accuracy", "0.5"],
                                   ["--price", "2pp/s", "--limit", "max=9s", "--limit", "mean=3"],
                                   ["--latency", "p50"], ["--max-latency", "3"]])
def test_valid_spellings_are_accepted_by_both_commands(data, tmp_path, capsys, flags):
    assert main(["calibrate", "--think-budgets", "--data", str(data), "--dry-run", *flags]) == 0
    path = tmp_path / "a.think.json"
    path.write_text(json.dumps(A))
    assert main(["compare", str(path), *flags]) == 0
    capsys.readouterr()


def test_calibrate_records_the_preference_in_its_settings(data):
    args = build_parser().parse_args(["calibrate", "--think-budgets", "--data", str(data), "--limit", "p99<=2s",
                                      "--limit", "max=3s", "--price", "1pp/s", "--max-latency", "4"])
    s = settings_from_args(args)
    assert (s.latency, s.price, s.max_latency) == ("mean", 1.0, 4.0)
    assert s.limits == (Limit("p99", 2000.0), Limit("max", 3000.0))                 # the --limit ones
    assert s.preference == Preference(price=1.0, limits=s.limits, max_latency_s=4.0)  # --max-latency joins them
    assert s.preference.limits[-1] == Limit("mean", 4000.0, "--max-latency")
    default = settings_from_args(build_parser().parse_args(["calibrate", "--think-budgets", "--data", str(data)]))
    assert default.latency == "mean" and default.preference == Preference()
    assert ThinkSettings("d", "d", "x", (0,)).latency == "mean"
    assert "price" not in default.comparable() and "latency" not in default.comparable()   # a resume never depends on it


@pytest.mark.parametrize("flag", [["--limit", "p99=2s"], ["--price", "1"]])
def test_the_new_flags_apply_only_with_think_budgets(data, capsys, flag):
    assert main(["calibrate", "--url", "http://127.0.0.1:1", "--data", str(data), *flag]) == 2
    assert "apply only with --think-budgets" in capsys.readouterr().err


def test_the_menus_report_passes_the_whole_preference_through(tmp_path, monkeypatch):
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate({}))
    monkeypatch.setattr(menu.compare, "write_report", lambda *a, **k: seen.update(k) or (_ for _ in ()).throw(
        menu.InvalidInput("stop here")))
    settings = ThinkSettings(str(data), data.name, "x" * 64, (0, 32, 64), warmup=0, audit=2, price=2.0,
                             limits=(Limit("p99", 6000.0),), max_latency=9.0)
    Driver(settings, _two(tmp_path), _opts(tmp_path), log=lambda s: None, launcher=_stub_launcher(),
           scan_fn=lambda **k: [], host=HOST).run()
    assert seen["pref"] == settings.preference
    assert seen["pref"].limits == (Limit("p99", 6000.0), Limit("mean", 9000.0, "--max-latency"))
    assert dataclasses.replace(settings, price=0.0).preference.price == 0.0
