"""Model menus: sources and names, `MENU.json`, the `args` policy, the GPU rule and the memory
precheck, `run.json` and `--resume`, the in-process `callo serve` per model on the fake llama-server executable, the
driver and the CLI."""
import json
from pathlib import Path

import pytest

from callo.calibrate_think import settings_from_args
from callo.errors import InvalidInput
from callo.menu import (
    MENU_KEYS,
    Entry,
    Launch,
    check_args,
    entries_from_models,
    entry_from_dict,
    load_menu,
    menu_sha256,
    sampler_warning,
)

MENU = {"models": [
    {"name": "tiel-q4", "model": "/v/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf", "ctx": 65536},
    {"name": "qwen9b", "model": "/v/Qwen3.5-9B-Q8_0.gguf", "parallel": 2, "args": ["-b", "512"], "s1_slots": 1},
    {"name": "big-dense", "model": "/v/70B-Q4_K_M.gguf", "no_kvu": True, "ctx": 32768, "think_sampling": "fixed"}]}


def write_menu(path: Path, doc=MENU) -> Path:
    path.write_text(json.dumps(doc))
    return path


def test_a_menu_parses_with_the_shared_launch_flags_as_defaults(tmp_path):
    launch = Launch(ctx=8192, parallel=4, llama_server="/opt/llama-server", args=("-ctk", "q8_0"))
    tiel, qwen, big = load_menu(write_menu(tmp_path / "menu.json"), launch)
    assert tiel == Entry("tiel-q4", "/v/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf", None, 65536, 4, False, 1, "/opt/llama-server",
                         ("-ctk", "q8_0"), None, None, None)
    assert (qwen.parallel, qwen.args, qwen.ctx) == (2, ("-b", "512"), 8192)         # the entry's own args, the shared ctx
    assert (big.no_kvu, big.think_sampling, big.mmproj, big.user_delimiter) == (True, "fixed", None, None)
    assert big.serve_argv(cache=True, cache_dir=Path("/c")) == [
        "serve", "--model=/v/70B-Q4_K_M.gguf", "--port", "0", "--think-budget", "0", "--parallel", "4", "--s1-slots", "1",
        "--llama-server=/opt/llama-server", "--cache-dir=/c", "--ctx", "32768", "--no-kvu", "--cache", "--",
        "--cache-ram", "0", "-ctk", "q8_0"]                                      # the host prompt cache off
    assert qwen.serve_argv(cache=False, cache_dir=Path("/c"))[-5:] == ["--", "--cache-ram", "0", "-b", "512"]
    assert Entry("p", "/p.gguf").serve_argv(cache=False, cache_dir=Path("/c"))[-3:] == ["--", "--cache-ram", "0"]
    assert "--ctx" in qwen.serve_argv(cache=False, cache_dir=Path("/c"))
    assert tiel.as_dict()["args"] == ["-ctk", "q8_0"] and json.dumps(tiel.as_dict())      # JSON-serialisable


def test_serve_argv_says_the_runs_cache_either_way(tmp_path):
    """--cache is on by default since 0.3.0: a cache-off run's servers get an explicit --no-cache (before `--`),
    so a menu-launched server matches the run. A menu has no cache key (the run-wide flag decides, default on)."""
    e = Entry("p", "/p.gguf")
    on, off = e.serve_argv(cache=True, cache_dir=Path("/c")), e.serve_argv(cache=False, cache_dir=Path("/c"))
    assert "--cache" in on and "--no-cache" not in on and on.index("--cache") < on.index("--")
    assert "--no-cache" in off and "--cache" not in off and off.index("--no-cache") < off.index("--")
    for argv, want in ((on, True), (off, False)):
        args_list, _ = cli_mod.split_passthrough(argv)
        assert cli_mod.build_parser().parse_args(args_list).cache is want
    assert "cache" not in MENU_KEYS
    (entry,) = load_menu(write_menu(tmp_path / "m.json", {"models": [{"name": "a", "model": "/m.gguf"}]}), Launch())
    data = tmp_path / "d.jsonl"
    data.write_text("{}\n")
    args = cli_mod.build_parser().parse_args(["calibrate", "--think-budgets", "--data", str(data), "--menu", "m.json"])
    settings = settings_from_args(args)
    assert settings.cache is True and "--cache" in entry.serve_argv(cache=settings.cache, cache_dir=Path("/c"))


@pytest.mark.parametrize("doc, why", [
    ({"models": [{"name": "a", "model": "/m.gguf", "bogus": 1}]}, "unknown key"),
    ({"models": [{"name": "a b", "model": "/m.gguf"}]}, "name must match"),
    ({"models": [{"name": "a"}]}, "model must be a GGUF path"),
    ({"models": [{"name": "a", "model": "/m.gguf"}, {"name": "a", "model": "/n.gguf"}]}, "share the name"),
    ({"models": [{"name": "a", "model": "/m.gguf", "no_kvu": True}]}, "no_kvu needs ctx"),
    ({"models": [{"name": "a", "model": "/m.gguf", "s1_slots": 4}]}, r"s1_slots must be an integer >= 1 and below parallel \(4\)"),
    ({"models": [{"name": "a", "model": "/m.gguf", "parallel": 2, "s1_slots": 2}]}, "s1_slots must be an integer >= 1"),
    ({"models": [{"name": "a", "model": "/m.gguf", "s1_slots": 0}]}, "s1_slots must be an integer >= 1"),
    ({"models": [{"name": "a", "model": "/m.gguf", "parallel": 1, "s1_slots": 0}]}, "s1_slots must be an integer >= 1"),
    ({"models": [{"name": ".", "model": "/m.gguf"}]}, "name must match"),
    ({"models": [{"name": "..", "model": "/m.gguf"}]}, "name must match"),
    ({"models": [{"name": None, "model": "/m.gguf"}]}, "name must match"),           # the keys that always carry a value
    ({"models": [{"name": "a", "model": None}]}, "model must be a GGUF path, got None"),
    ({"models": [{"name": "a", "model": "/m.gguf", "parallel": None}]}, "parallel must be an integer >= 1, got None"),
    ({"models": [{"name": "a", "model": "/m.gguf", "no_kvu": None}]}, "no_kvu must be true or false, got None"),
    ({"models": [{"name": "a", "model": "/m.gguf", "s1_slots": None}]}, "s1_slots must be an integer >= 1"),
    ({"models": [{"name": "a", "model": "/m.gguf", "llama_server": None}]}, "llama_server must be an executable path, got None"),
    ({"models": [{"name": "a", "model": "/m.gguf", "args": None}]}, "args must be a list of strings, got None"),
    ({"models": [{"name": "a", "model": "/m.gguf", "think_sampling": None}]}, "think_sampling must be a non-empty string, got None"),
    ({"models": [{"name": "a", "model": "/m.gguf", "mmproj": ""}]}, "mmproj must be a non-empty string or null"),
    ({"models": [{"name": "a", "model": "/m.gguf", "parallel": 0}]}, "parallel must be an integer >= 1"),
    ({"models": [{"name": "a", "model": "/m.gguf", "ctx": True}]}, "ctx must be a positive integer"),
    ({"models": [{"name": "a", "model": "/m.gguf", "think_sampling": "bogus"}]}, "think_sampling: --think-sampling"),
    ({"models": [{"name": "a", "model": "/m.gguf", "args": "-b 512"}]}, "args must be a list of strings"),
    ({"models": [{"name": "a", "model": "/m.gguf", "args": ["--grammar", "root ::= x"]}]}, "constrain the trace"),
    ({"models": ["/m.gguf"]}, "each model is an object"),
    ({"models": []}, "non-empty"),
    ({"models": [{"name": "a", "model": "/m.gguf"}], "budgets": [0]}, "unknown top-level key"),
    ([], "expected an object"),
])
def test_menu_refusals_name_the_entry_and_the_key(tmp_path, doc, why):
    with pytest.raises(InvalidInput, match=why):
        load_menu(write_menu(tmp_path / "menu.json", doc), Launch())


def test_s1_slots_follows_callo_serve_and_null_means_none_for_the_nullable_keys(tmp_path):
    # callo serve's rule (cli.py, slots.SlotRouter): s1_slots >= 1, and below parallel only when parallel > 1
    one = entry_from_dict({"name": "a", "model": "/m.gguf", "parallel": 1}, Launch(), where="w")
    assert (one.parallel, one.s1_slots) == (1, 1)
    assert entry_from_dict({"name": "a", "model": "/m.gguf", "parallel": 4, "s1_slots": 3}, Launch(), where="w").s1_slots == 3
    # leaving a key out inherits the shared flag; null on a nullable key means "none for this model", overriding it
    shared = Launch(ctx=8192, user_delimiter="<|user|>")
    e = entry_from_dict({"name": "a", "model": "/m.gguf"}, shared, where="w")
    assert (e.ctx, e.user_delimiter) == (8192, "<|user|>")
    n = entry_from_dict({"name": "a", "model": "/m.gguf", "ctx": None, "user_delimiter": None, "mmproj": None,
                         "position_bias": None}, shared, where="w")
    assert (n.ctx, n.user_delimiter, n.mmproj, n.position_bias) == (None, None, None, None)
    with pytest.raises(InvalidInput, match="no_kvu needs ctx"):                    # null ctx is no ctx
        entry_from_dict({"name": "a", "model": "/m.gguf", "ctx": None}, Launch(ctx=8192, no_kvu=True), where="w")


# the reference MENU.json example, verbatim: the loader reads paths, it does not open them
SPEC_MENU = """{"models": [
  {"name": "tiel-q4", "model": "/models/…/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf", "ctx": 65536},
  {"name": "qwen9b", "model": "/models/…/Qwen3.5-9B-Q8_0.gguf", "parallel": 4, "args": ["-b", "512"]},
  {"name": "big-dense", "model": "/…/70B-Q4_K_M.gguf", "mmproj": null, "no_kvu": true, "ctx": 32768,
   "think_sampling": "fixed", "position_bias": null, "user_delimiter": null}]}
"""


def test_the_spec_example_menu_loads(tmp_path):
    (tmp_path / "menu.json").write_text(SPEC_MENU, encoding="utf-8")
    tiel, qwen, big = load_menu(tmp_path / "menu.json", Launch(ctx=8192, user_delimiter="<|u|>"))
    assert (tiel.name, tiel.ctx, tiel.user_delimiter) == ("tiel-q4", 65536, "<|u|>")
    assert (qwen.parallel, qwen.args, qwen.ctx) == (4, ("-b", "512"), 8192)
    assert (big.mmproj, big.position_bias, big.user_delimiter) == (None, None, None)     # null: none for this model
    assert (big.no_kvu, big.ctx, big.think_sampling, big.model) == (True, 32768, "fixed", "/…/70B-Q4_K_M.gguf")


def test_an_unreadable_or_non_json_menu_is_a_usage_error(tmp_path):
    with pytest.raises(InvalidInput, match="cannot read"):
        load_menu(tmp_path / "missing.json", Launch())
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(InvalidInput, match="not JSON"):
        load_menu(tmp_path / "bad.json", Launch())


def test_the_args_policy_refuses_trace_constraints_and_names_the_sampler_flags():
    for bad in (["--grammar-file", "g.gbnf"], ["-j", "{}"], ["--json-schema={}"], ["--json-schema-file", "s.json"]):
        with pytest.raises(InvalidInput, match="constrain the trace"):
            check_args(["-b", "512", *bad])
    found = check_args(["-ctk", "q8_0", "--temp", "0.7", "--repeat-penalty", "1.1", "--dry-multiplier=0.8", "-l", "1+2",
                        "--mirostat", "2", "--adaptive-target", "0.5", "--reasoning-budget", "100", "-np", "4"])
    assert found == ["--temp", "--repeat-penalty", "--dry-multiplier=0.8", "-l", "--mirostat", "--adaptive-target",
                     "--reasoning-budget"]
    e = entry_from_dict({"name": "x", "model": "/m.gguf", "args": ["--temp", "0.7", "-ctk", "q8_0"]}, Launch(), where="w")
    assert sampler_warning(e) == ("WARNING: x: args --temp 0.7 are sampler flags; callo's pinned samplers override them on "
                                  "every generation and every readout pass, so they do not change this model's "
                                  "measurement")
    assert sampler_warning(entry_from_dict({"name": "y", "model": "/m.gguf", "args": ["-ctk", "q8_0"]}, Launch(),
                                           where="w")) is None


def test_model_paths_become_entries_named_by_stem_with_suffixes_for_duplicates():
    launch = Launch(ctx=4096, args=("-b", "256"))
    a, b, c, d = entries_from_models(["/x/Qwen-9B.gguf", "/y/Qwen-9B.gguf", "/z/Tiel Q4 (v2).gguf", "/x/Qwen-9B.gguf"], launch)
    assert [e.name for e in (a, b, c, d)] == ["Qwen-9B", "Qwen-9B-2", "Tiel_Q4__v2_", "Qwen-9B-3"]
    assert a.ctx == 4096 and a.args == ("-b", "256") and a.llama_server == "llama-server" and a.think_sampling is None
    names = [e.name for e in entries_from_models(["/a/Q.gguf", "/b/Q.gguf", "/c/Q-2.gguf"], launch)]
    assert names == ["Q", "Q-2", "Q-2-2"] and len(set(names)) == 3       # a generated suffix never collides
    long = "L" * 70
    many = [e.name for e in entries_from_models([f"/{i}/{long}.gguf" for i in range(101)], launch)]
    assert len(set(many)) == 101 and max(map(len, many)) == 64 and many[100] == "L" * 60 + "-101"
    assert [e.name for e in entries_from_models(["/x/..gguf", "/y/...gguf"], launch)] == ["_", "__"]   # never . or ..
    with pytest.raises(InvalidInput, match="at least one"):
        entries_from_models([], launch)
    with pytest.raises(InvalidInput, match="arguments after --: --grammar"):
        entries_from_models(["/m.gguf"], Launch(args=("--grammar", "x")))


def test_menu_sha256_is_stable_and_changes_with_an_appended_candidate(tmp_path):
    entries = load_menu(write_menu(tmp_path / "menu.json"), Launch())
    again = load_menu(write_menu(tmp_path / "menu2.json"), Launch())
    assert menu_sha256(entries) == menu_sha256(again) and menu_sha256(entries).startswith("sha256:")
    more = {"models": [*MENU["models"], {"name": "new", "model": "/v/new.gguf"}]}
    assert menu_sha256(load_menu(write_menu(tmp_path / "menu3.json", more), Launch())) != menu_sha256(entries)
    assert menu_sha256(entries) != menu_sha256(load_menu(write_menu(tmp_path / "menu.json"), Launch(ctx=1024)))


# ---- the GPU rule, the memory precheck, run.json and --dry-run ----------------------------------------------------

import os  # noqa: E402

from helpers import tiny_gguf  # noqa: E402

from callo.calibrate_think import ThinkSettings, data_sha256  # noqa: E402
from callo.menu import (  # noqa: E402
    RunFile,
    dry_run_model_lines,
    estimate_s,
    footprint_bytes,
    gpu_busy_message,
    gpu_class,
    gpu_findings,
    memory_budget,
    model_servers,
    precheck,
    process_resident_bytes,
    rates_profile,
)
from callo.modelkey import load_model_key  # noqa: E402

# Real `ps -axo pid=,args=` lines from a dev box: Homebrew's ollama, the oMLX menu-bar app, a
# llama-server started by absolute path, a console-script and a `-m` mlx_lm.server, the test suite's own Python fake,
# an `ollama runner` holding a model, and file arguments that merely name a server.
PS = """\
  791 /opt/homebrew/opt/ollama/bin/ollama serve
 5922 /Applications/oMLX.app/Contents/MacOS/oMLX
90535 /opt/llama.cpp/build/bin/llama-server -m /v/CyberOrca-Q4_K_XL.gguf --port 8080
  103 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/mlx_lm.server --model x
  104 /usr/bin/python3 -m mlx_lm.server --model y
  105 /usr/bin/python3 /tmp/pytest/llama-server --port 55000
  106 vim llama-server.log
  107 /bin/zsh
  108 /opt/homebrew/opt/ollama/bin/ollama runner --model /x/blobs/sha256-abc --port 61234
  109 /usr/bin/tail -f omlx.log
  110 ./build/bin/llama-bench -m /v/m.gguf
  bogus line
"""
# residency (bytes) per pid: idle daemons, a loaded mlx server, one that cannot be measured, a runner holding a model
RESIDENT = {791: 70 * 2**20, 5922: 90 * 2**20, 103: 5 * 2**30, 104: None, 108: 3 * 2**30}


def test_gpu_class_reads_the_program_from_argv_like_the_bench():
    assert gpu_class(["/opt/llama.cpp/build/bin/llama-server", "-m", "x"]) == "always"
    assert gpu_class(["./build/bin/llama-bench", "-m", "x"]) == "always" and gpu_class(["LLAMA-SERVER"]) == "always"
    assert gpu_class(["/usr/bin/python3", "/tmp/pytest/llama-server", "--port", "1"]) is None        # a Python script
    assert gpu_class(["/Applications/oMLX.app/Contents/MacOS/oMLX"]) == "resident"
    assert gpu_class(["/opt/homebrew/opt/ollama/bin/ollama", "serve"]) == "resident"
    assert gpu_class(["ollama", "runner", "--model", "x"]) == "resident" and gpu_class(["ollama", "ps"]) is None
    assert gpu_class(["/opt/homebrew/bin/python3.13", "/opt/homebrew/bin/mlx_lm.server", "--model", "x"]) == "resident"
    assert gpu_class(["/usr/bin/python3", "-u", "-X", "dev", "-m", "mlx_lm.server"]) == "resident"
    assert gpu_class(["python3", "-m", "mlx_lm", "server"]) == "resident" and gpu_class(["sh", "/opt/homebrew/bin/omlx", "start"]) == "resident"
    assert gpu_class(["vim", "llama-server.log"]) is None and gpu_class(["tail", "-f", "omlx.log"]) is None
    assert gpu_class(["python3", "-c", "import mlx_lm.server"]) is None and gpu_class([]) is None


def test_model_servers_and_the_gpu_rule_from_real_ps_lines_with_the_residency_rule():
    measure = RESIDENT.get
    found = model_servers(PS, resident_bytes=measure)
    assert [pid for pid, _ in found] == [90535, 103, 104, 108, 110]  # not the idle daemons, the fake or the file arguments
    assert found[0][1].startswith("/opt/llama.cpp/build/bin/llama-server -m ")
    assert found[1][1].endswith(" [resident 5.0 GiB]") and found[2][1].endswith(" [resident: unknown, counted as holding a model]")
    assert found[3][1].startswith("/opt/homebrew/opt/ollama/bin/ollama runner") and found[3][1].endswith("[resident 3.0 GiB]")
    assert found[4][1] == "./build/bin/llama-bench -m /v/m.gguf"                 # every llama.cpp program, any path
    loaded = {**RESIDENT, 791: 6 * 2**30, 5922: 20 * 2**30}        # the same daemons holding models: they count
    assert [pid for pid, _ in model_servers(PS, resident_bytes=loaded.get)] == [791, 5922, 90535, 103, 104, 108, 110]
    assert [pid for pid, _ in model_servers(PS, exclude={90535, 108, 110}, resident_bytes=measure)] == [103, 104]
    assert model_servers(f"  {os.getpid()} /opt/llama-server\n", resident_bytes=measure) == []       # never itself
    lines, cannot_tell = gpu_findings(PS, [(90535, 8080), (900, None), (901, 8200)], exclude={103, 104, 108, 110, 901},
                                      resident_bytes=measure)
    assert cannot_tell is None and lines == ["pid 90535: /opt/llama.cpp/build/bin/llama-server -m "
                     "/v/CyberOrca-Q4_K_XL.gguf --port 8080",
                     "pid 900: a llama-server left by an earlier callo (still loading); if pid 900 is not that "
                     "llama-server, delete llama-server-starting-<callo pid>.pid in callo's cache dir"]
    msg = gpu_busy_message(lines[:1])
    assert msg.startswith("refusing to calibrate: another model server is running (pid 90535: ") and "--allow-gpu-busy" in msg
    assert "holds no model is not counted" in msg and gpu_findings("", [], resident_bytes=measure) == ([], None)


def test_process_resident_bytes_reads_footprint_then_rss_and_tolerates_failures():
    text = "oMLX [5922]: 64-bit    Footprint: 135 MB (16384 bytes per page)\nTotal footprint: 20 GB\n"
    assert footprint_bytes(text, 5922) == 135 * 2**20 and footprint_bytes(text, 1) is None
    assert footprint_bytes("x [7]: Footprint: 2.5 GB", 7) == int(2.5 * 2**30)

    class R:
        def __init__(self, stdout):
            self.stdout = stdout

    calls = []

    def run(argv, **kw):
        calls.append(argv[0])
        return R(text if argv[0] == "footprint" else " 4096\n")

    assert process_resident_bytes(5922, run) == 135 * 2**20 and calls == ["footprint"]
    assert process_resident_bytes(1, run) == 4096 * 1024 and calls[-2:] == ["footprint", "ps"]   # not its line: RSS

    def failing(argv, **kw):
        raise OSError("no such tool")

    assert process_resident_bytes(1, failing) is None
    assert process_resident_bytes(1, lambda argv, **kw: R("")) is None


def test_memory_budget_per_platform_and_the_precheck(tmp_path):
    host = {"memory_bytes": 64 * 2**30, "gpu": None}
    assert memory_budget(host, system="Darwin") == 48 * 2**30
    assert memory_budget({**host, "gpu": "NVIDIA H100 80GB HBM3, 81559 MiB"}, system="Linux") == 81559 * 2**20
    assert memory_budget(host, system="Linux") == 64 * 2**30 and memory_budget({}, system="Darwin") is None
    assert memory_budget(host, gpu_mem_gib=12.5, system="Darwin") == int(12.5 * 2**30)
    unknown = {"os": None, "machine": None, "chip": None, "memory_bytes": None, "gpu": None, "fingerprint": None}
    assert memory_budget(unknown, system="Darwin") is None and memory_budget(unknown, system="Linux") is None
    assert memory_budget({**unknown, "gpu": "some GPU"}, system="Linux") is None and memory_budget(None, system="Darwin") is None
    model = tiny_gguf(tmp_path / "m.gguf")
    e = Entry("m", str(model))
    assert precheck(e, 2**30, 2.0) == "needs ~2.0 GiB, budget 1.0 GiB (--gpu-mem to override)"
    assert precheck(e, 4 * 2**30, 2.0) is None and precheck(e, 2**30, 0.0) is None
    assert precheck(Entry("x", str(tmp_path / "missing.gguf")), 2**30, 2.0) is None       # the launch reports it
    assert precheck(e, None, 2.0) is None
    with_mm = Entry("mm", str(model), mmproj=str(tmp_path / "missing-mmproj.gguf"))
    assert precheck(with_mm, 2**30, 2.0) is None
    (tmp_path / "proj.gguf").write_bytes(b"x" * 2**20)
    assert precheck(Entry("mm", str(model), mmproj=str(tmp_path / "proj.gguf")), 2**30, 1.0).startswith("needs ~1.0 GiB")


def _settings(tmp_path, **over) -> ThinkSettings:
    data = tmp_path / "d.jsonl"
    if not data.exists():
        data.write_text("{}\n")
    base = {"budgets": (0, 64, 256), "warmup": 1, "audit": 3}
    base.update(over)
    return ThinkSettings(str(data), data.name, data_sha256(data), **base)


def test_run_file_transitions_resume_merge_and_the_atomic_write(tmp_path):
    settings = _settings(tmp_path)
    entries = [Entry("a", "/a.gguf"), Entry("b", "/b.gguf")]
    rf = RunFile.new(tmp_path / "run.json", settings, entries)
    assert rf.data["order"] == ["a", "b"] and rf.data["menu_sha256"] == menu_sha256(entries)
    assert not rf.path.exists()
    rf.set("a", "running")
    assert rf.path.exists() and not list(tmp_path.glob("run.json*.tmp"))
    prof = tmp_path / "a.think.json"
    prof.write_text("{}")
    rf.set("a", "done", profile=prof)
    rf.set("b", "failed", reason="llama-server exited with 7 before listening")
    again = RunFile.load(rf.path)
    assert again.record("a")["status"] == "done" and again.record("a")["started"] and again.record("a")["ended"]
    assert again.record("b") == {"status": "failed", "reason": "llama-server exited with 7 before listening",
                                 "started": None, "ended": again.record("b")["ended"], "profile": None}
    assert again.done_profiles() == [prof] and again.settings_diff(settings) == []
    assert again.settings_diff(_settings(tmp_path, budgets=(0, 128), seed=3)) == ["budgets", "seed"]
    assert again.summary() == ["a: done", "b: failed (llama-server exited with 7 before listening)"]
    added = again.merge([Entry("b", "/b.gguf"), Entry("c", "/c.gguf")])        # the menu lost a, gained c
    assert added == ["c"] and again.data["order"] == ["b", "c", "a"] and again.record("c")["status"] == "pending"
    assert again.data["menu_sha256"] == menu_sha256([Entry("b", "/b.gguf"), Entry("c", "/c.gguf")])
    with pytest.raises(ValueError):
        again.set("c", "bogus")
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(InvalidInput, match="not a callo think run file"):
        RunFile.load(tmp_path / "bad.json")
    with pytest.raises(InvalidInput, match="cannot read"):
        RunFile.load(tmp_path / "missing.json")


def test_rates_profile_estimate_and_the_dry_run_lines(tmp_path):
    model = tiny_gguf(tmp_path / "m.gguf")
    key = load_model_key(model, None)
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    prof = {"callo_think_profile": 1, "model_key": key, "environment": {"run": {"decode_tok_s_median": 100.0}},
            "budgets": {"0": {"lat_p50_ms": [250.0, 240.0, 260.0]}}}
    (earlier / "old.think.json").write_text(json.dumps(prof))
    (earlier / "junk.think.json").write_text("not json")
    found = rates_profile(key, [tmp_path / "out", earlier])
    assert found is not None and found[1] == earlier / "old.think.json" and rates_profile(None, [earlier]) is None
    assert rates_profile("sha256:" + "0" * 64, [earlier]) is None
    settings = _settings(tmp_path, budgets=(0, 64, 256), warmup=1, audit=3)
    # per question 0.25 + 2.56 + 2 x 0.25 = 3.31 s; 7 questions (6 + warm-up); audit 3 x (0.64 + 0.25); drift 3 x 2.56
    assert estimate_s(prof, settings, 6) == pytest.approx(7 * 3.31 + 3 * 0.89 + 3 * 2.56)
    assert estimate_s({"budgets": {"0": {"lat_p50_ms": [250.0]}}}, settings, 6) is None
    assert estimate_s(prof, _settings(tmp_path, budgets=(0,)), 6) == pytest.approx(7 * 0.25)
    many = _settings(tmp_path, budgets=(0, 64, 256), warmup=5, audit=3)       # the warm-up is capped at the questions
    assert estimate_s(prof, many, 2) == pytest.approx(4 * 3.31 + 3 * 0.89 + 3 * 2.56)
    capped = dry_run_model_lines(Entry("m", str(model)), many, 2, budget=None, headroom_gib=2.0, rates_dirs=[],
                                 cache_dir=None)
    assert capped[1].endswith(", 768 for the drift check and 512 for the warm-up")
    assert dry_run_model_lines(Entry("m", str(model)), _settings(tmp_path, budgets=(0,)), 2, budget=None,
                               headroom_gib=2.0, rates_dirs=[], cache_dir=None)[1].endswith("and 0 for the warm-up")
    lines = dry_run_model_lines(Entry("m", str(model)), settings, 6, budget=48 * 2**30, headroom_gib=2.0,
                                rates_dirs=[tmp_path / "out", earlier], cache_dir=None)
    assert lines[0] == f"m: {model} (0.0 GiB); fits the budget of 48.0 GiB"
    assert lines[1] == ("  worst case: 6 x 256 = 1536 generated tokens, plus up to 192 for the audit, 768 for the drift "
                        "check and 256 for the warm-up")
    assert lines[2] == f"  time: ~0m34s from {earlier / 'old.think.json'} (decode 100 tok/s, budget-0 readout 0.25 s)"
    skipped = dry_run_model_lines(Entry("m", str(model)), settings, 6, budget=2**30, headroom_gib=2.0, rates_dirs=[],
                                  cache_dir=None)
    assert skipped[0].endswith("would be skipped: needs ~2.0 GiB, budget 1.0 GiB (--gpu-mem to override)")
    assert skipped[2].startswith("  time: unknown until measured")
    missing = dry_run_model_lines(Entry("x", str(tmp_path / "no.gguf")), settings, 6, budget=None, headroom_gib=2.0,
                                  rates_dirs=[], cache_dir=None)
    assert missing[0].endswith("NOT FOUND (the launch would fail; the run goes on to the next model)")
    unknown = dry_run_model_lines(Entry("m", str(model)), settings, 6, budget=None, headroom_gib=2.0, rates_dirs=[],
                                  cache_dir=None)
    assert unknown[0].endswith("memory budget unknown: not prechecked")


# ---- ps and the GPU rule failing closed, the scan, rates profiles, run.json edge cases --------------------------

import subprocess  # noqa: E402

from callo import menu as menu_mod  # noqa: E402
from callo.menu import (  # noqa: E402
    PS_ARGV,
    _hms,
    orphan_pid_file,
    ps_listing,
    scan,
)


class _Proc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def _raising(exc):
    def run(argv, **kw):
        raise exc
    return run


def test_ps_listing_pins_the_argv_and_fails_closed_on_missing_error_timeout_and_empty():
    seen = []

    def run(argv, **kw):
        seen.append((list(argv), kw))
        return _Proc(PS)

    assert ps_listing(run) == (PS, None)
    assert seen[0][0] == ["ps", "-axo", "pid=,args="] and list(PS_ARGV) == seen[0][0]   # never comm=
    assert seen[0][1]["timeout"] == 30 and seen[0][1]["text"] is True
    assert ps_listing(_raising(FileNotFoundError(2, "No such file"))) == (None, "ps not found")
    assert ps_listing(_raising(subprocess.TimeoutExpired(["ps"], 30))) == (None, "timed out after 30 s")
    assert ps_listing(_raising(PermissionError(13, "Permission denied")))[0] is None
    assert ps_listing(lambda argv, **kw: _Proc("", 1, "ps: illegal option -- o\nusage: ps\n")) == \
        (None, "exit 1: ps: illegal option -- o")
    assert ps_listing(lambda argv, **kw: _Proc("", 2)) == (None, "exit 2")
    assert ps_listing(lambda argv, **kw: _Proc("  \n")) == (None, "it listed no processes")
    assert ps_listing(lambda argv, **kw: _Proc(None)) == (None, "it listed no processes")
    assert ps_listing(_raising(FileNotFoundError()))[0] is None


def _rule(ps_text, orphans, **kw) -> list[str]:
    """The GPU rule's lines as `scan` words them: the "cannot tell" line (with the remedy) first, then the servers."""
    servers, cannot_tell = gpu_findings(ps_text, orphans, **kw)
    return ([f"{cannot_tell}: {menu_mod.GPU_REMEDY}"] if cannot_tell else []) + servers


def test_the_gpu_rule_refuses_when_ps_cannot_tell():
    lines = _rule(None, [], ps_error="ps not found")
    assert lines == ["could not list processes (ps failed: ps not found): stop any model server, or pass "
                     "--allow-gpu-busy"]
    assert _rule(None, [])[0].startswith("could not list processes (ps failed: unknown error)")
    msg = gpu_busy_message(lines)
    assert msg == ("refusing to calibrate: could not list processes (ps failed: ps not found): stop any model server, "
                   "or pass --allow-gpu-busy to measure anyway (every profile is then marked gpu_busy, which compare "
                   "flags)")
    both = _rule(None, [(900, 8200)], ps_error="exit 1")
    assert len(both) == 2 and gpu_busy_message(both).endswith(
        "; also found: pid 900: a llama-server left by an earlier callo (port 8200); if pid 900 is not that "
        "llama-server, delete llama-server-8200.pid in callo's cache dir")


def test_scan_with_fake_runners_and_fake_orphans(tmp_path, monkeypatch):
    (tmp_path / "llama-server-starting-4242.pid").write_text("900\n")
    (tmp_path / "llama-server-starting-4343.pid").write_text("777\n")
    orphans = [(900, None), (901, 8200), (90535, 8080)]
    monkeypatch.setattr(menu_mod, "find_orphans", lambda cache_dir: orphans if cache_dir == tmp_path else [])
    runs = []

    def run(argv, **kw):
        runs.append(list(argv))
        return _Proc(PS)

    lines = scan(cache_dir=tmp_path, run=run, exclude={103, 104, 108, 110}, resident_bytes=RESIDENT.get)
    assert runs == [["ps", "-axo", "pid=,args="]]                          # one read-only listing, nothing else
    assert lines[0].startswith("pid 90535: /opt/llama.cpp/build/bin/llama-server")
    assert lines[1:] == [
        f"pid 900: a llama-server left by an earlier callo (still loading); if pid 900 is not that llama-server, "
        f"delete {tmp_path / 'llama-server-starting-4242.pid'}",
        f"pid 901: a llama-server left by an earlier callo (port 8200); if pid 901 is not that llama-server, "
        f"delete {tmp_path / 'llama-server-8200.pid'}"]                      # 90535 once: the ps line already names it
    failed = scan(cache_dir=tmp_path, run=_raising(subprocess.TimeoutExpired(["ps"], 30)), resident_bytes=RESIDENT.get)
    assert failed[0] == ("could not list processes (ps failed: timed out after 30 s): stop any model server, or pass "
                         "--allow-gpu-busy") and len(failed) == 4
    assert scan(cache_dir=tmp_path / "other", run=run, exclude={90535, 103, 104, 108, 110},
                resident_bytes=RESIDENT.get) == []
    assert orphan_pid_file(tmp_path, 555, None) == str(tmp_path / "llama-server-starting-<callo pid>.pid")


def test_model_servers_boundary_and_truncation():
    at = {791: 2 * 2**30, 5922: 2 * 2**30 - 1}
    found = model_servers(PS, exclude={90535, 103, 104, 108, 110}, resident_bytes=at.get)
    assert found == [(791, "/opt/homebrew/opt/ollama/bin/ollama serve [resident 2.0 GiB]")]   # >= 2 GiB counts
    long = "/opt/bin/llama-server " + "-x " * 100
    (pid, cmd), = model_servers(f"77 {long}\n", resident_bytes=at.get)
    assert pid == 77 and cmd == long[:120] and len(cmd) == 120


def test_memory_budget_sums_every_gpu_on_linux():
    host = {"memory_bytes": 512 * 2**30, "gpu": "NVIDIA H100 80GB HBM3, 81559 MiB\nNVIDIA H100 80GB HBM3, 81559 MiB"}
    assert memory_budget(host, system="Linux") == 2 * 81559 * 2**20


def test_dry_run_mmproj_lines(tmp_path):
    model = tiny_gguf(tmp_path / "m.gguf")
    settings = _settings(tmp_path)
    missing = dry_run_model_lines(Entry("mm", str(model), mmproj=str(tmp_path / "nope.gguf")), settings, 6,
                                  budget=48 * 2**30, headroom_gib=2.0, rates_dirs=[], cache_dir=None)
    assert missing[0] == f"mm: {model} (0.0 GiB); would fail: mmproj not found: {tmp_path / 'nope.gguf'}"
    (tmp_path / "proj.gguf").write_bytes(b"x" * 2**20)
    ok = dry_run_model_lines(Entry("mm", str(model), mmproj=str(tmp_path / "proj.gguf")), settings, 6,
                             budget=48 * 2**30, headroom_gib=2.0, rates_dirs=[], cache_dir=None)
    assert ok[0] == f"mm: {model} (0.0 GiB with mmproj); fits the budget of 48.0 GiB"


def _prof(key, created, decode=100.0, fmt=1):
    return {"callo_think_profile": fmt, "model_key": key, "created": created,
            "environment": {"run": {"decode_tok_s_median": decode}}, "budgets": {"0": {"lat_p50_ms": [250.0]}}}


def test_rates_profile_checks_the_format_and_prefers_the_newest_in_dir_first(tmp_path):
    key = "sha256:" + "a" * 64
    out, rates = tmp_path / "out", tmp_path / "rates"
    out.mkdir()
    rates.mkdir()
    (out / "a-foreign.think.json").write_text(json.dumps(_prof(key, "2026-10-09T00:00:00Z", fmt=2)))
    (out / "b-old.think.json").write_text(json.dumps(_prof(key, "2026-10-01T00:00:00Z", decode=50.0)))
    (out / "c-new.think.json").write_text(json.dumps(_prof(key, "2026-10-03T00:00:00Z", decode=80.0)))
    (out / "d-norates.think.json").write_text(json.dumps({**_prof(key, "2026-10-08T00:00:00Z"), "budgets": {}}))
    (rates / "newest.think.json").write_text(json.dumps(_prof(key, "2026-10-07T00:00:00Z")))
    p, f = rates_profile(key, [out, rates])
    assert f == out / "c-new.think.json" and p["environment"]["run"]["decode_tok_s_median"] == 80.0
    assert rates_profile(key, [tmp_path / "absent", rates])[1] == rates / "newest.think.json"
    assert rates_profile(key, [out / "nothing-here"]) is None
    assert rates_profile("sha256:" + "b" * 64, [out, rates]) is None


def test_estimate_s_returns_none_on_malformed_shapes(tmp_path):
    settings = _settings(tmp_path)
    good = _prof("k", "x")
    assert estimate_s(good, settings, 6) == pytest.approx(7 * 3.31 + 3 * 0.89 + 3 * 2.56)
    bad = [None, [], "x", {"environment": [], "budgets": {"0": {"lat_p50_ms": [250.0]}}},
           {**good, "budgets": {"0": {"lat_p50_ms": 250.0}}}, {**good, "budgets": {"0": {"lat_p50_ms": []}}},
           {**good, "budgets": {"0": {"lat_p50_ms": ["250"]}}}, {**good, "budgets": []},
           {**good, "environment": {"run": {"decode_tok_s_median": "100"}}},
           {**good, "environment": {"run": {"decode_tok_s_median": True}}},
           {**good, "environment": {"run": {"decode_tok_s_median": -1.0}}},
           {**good, "environment": {"run": {"decode_tok_s_median": float("nan")}}},
           {**good, "environment": {"run": None}}]
    assert [estimate_s(b, settings, 6) for b in bad] == [None] * len(bad)


def test_hms_minutes_and_hours():
    assert _hms(34.4) == "0m34s" and _hms(3599.6) == "1h00m" and _hms(2 * 3600 + 5 * 60 + 59) == "2h05m"


def _run_file(tmp_path):
    rf = RunFile.new(tmp_path / "run.json", _settings(tmp_path), [Entry("a", "/a.gguf"), Entry("b", "/b.gguf")])
    rf.write()
    return rf


def test_run_file_load_refuses_a_hand_edited_file(tmp_path):
    rf = _run_file(tmp_path)
    base = json.loads(rf.path.read_text())
    (tmp_path / "nj.json").write_text("{not json")
    with pytest.raises(InvalidInput, match="nj.json: not JSON"):
        RunFile.load(tmp_path / "nj.json")
    cases = {
        "no-order": {k: v for k, v in base.items() if k != "order"},
        "extra": {**base, "order": ["a", "b", "c"]},
        "short": {**base, "order": ["a"]},
        "dupe": {**base, "order": ["a", "b", "a"]},
        "not-str": {**base, "order": ["a", 2]},
        "rec": {**base, "models": {**base["models"], "b": "done"}},
        "status": {**base, "models": {**base["models"], "b": {**base["models"]["b"], "status": "finished"}}},
        "dropped": {**base, "dropped": ["zzz"]},
    }
    for name, data in cases.items():
        f = tmp_path / f"{name}-run.json"
        f.write_text(json.dumps(data))
        with pytest.raises(InvalidInput, match=f"{name}-run.json: .*hand-edited or partial run.json"):
            RunFile.load(f)
    old = {k: v for k, v in base.items() if k != "dropped"}           # `dropped` is optional
    (tmp_path / "old.json").write_text(json.dumps(old))
    assert RunFile.load(tmp_path / "old.json").summary() == ["a: pending", "b: pending"]


def test_run_file_transitions_rows_and_the_durable_write(tmp_path, monkeypatch):
    rf = _run_file(tmp_path)
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(menu_mod.os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd)))
    rf.set("a", "failed", reason="boom")
    assert synced and not list(tmp_path.glob("*.tmp"))
    first_end = rf.record("a")["ended"]
    rf.set("a", "running")                                       # a resume retries the failed model
    assert rf.record("a")["ended"] is None and rf.record("a")["started"] and first_end
    rf.set("b", "skipped", reason="needs ~9.0 GiB, budget 4.0 GiB (--gpu-mem to override)")
    assert rf.record("b")["ended"] and rf.record("b")["status"] == "skipped"
    rows = rf.summary_rows()
    assert [r["name"] for r in rows] == ["a", "b"] and rows[0]["status"] == "running" and rows[0]["ended"] is None
    assert rows[1] == {"name": "b", "status": "skipped", "reason": "needs ~9.0 GiB, budget 4.0 GiB (--gpu-mem to "
                       "override)", "started": None, "ended": rf.record("b")["ended"], "profile": None}
    seen_tmp = []

    def failing_replace(src, dst):
        seen_tmp.append(Path(src).name)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(menu_mod.os, "replace", failing_replace)
    before = rf.path.read_text()
    with pytest.raises(OSError):
        rf.set("b", "done")
    assert seen_tmp == [f"run.json.{os.getpid()}.tmp"] and not list(tmp_path.glob("*.tmp"))
    assert rf.path.read_text() == before                         # the old file survives a failed write


def test_merge_keeps_a_dropped_model_as_history_out_of_the_report(tmp_path):
    rf = _run_file(tmp_path)
    prof = tmp_path / "a.think.json"
    prof.write_text("{}")
    rf.set("a", "done", profile=prof)
    rf.set("b", "done", profile=prof)
    logged = []
    added = rf.merge([Entry("b", "/b.gguf"), Entry("c", "/c.gguf")], log=logged.append)
    assert added == ["c"] and logged == ["not in this menu: a (kept in run.json, left out of the report)"]
    assert rf.data["order"] == ["b", "c", "a"] and rf.data["dropped"] == ["a"] and rf.record("a")["status"] == "done"
    assert rf.done_profiles() == [prof] and [r["name"] for r in rf.summary_rows()] == ["b", "c"]
    assert rf.summary() == ["b: done", "c: pending"]
    rf.write()
    again = RunFile.load(rf.path)
    assert again.data["dropped"] == ["a"] and [r["name"] for r in again.summary_rows()] == ["b", "c"]
    logged.clear()
    assert again.merge([Entry("a", "/a.gguf"), Entry("b", "/b.gguf"), Entry("c", "/c.gguf")], log=logged.append) == []
    assert logged == [] and again.data["dropped"] == [] and again.done_profiles() == [prof, prof]


# ---- callo serve in-process per model on the fake llama-server executable -----------------------------------------

import contextlib  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from fake_llama_think import exe_script, natural_len  # noqa: E402
from helpers import get_json  # noqa: E402

from callo.menu import (  # noqa: E402
    MenuFailure,
    StopHandler,
    alive,
    attempt_separator,
    last_error_line,
    loading_child_pid,
    served_model,
    settle,
    stop_child,
    terminate_child,
    wait_exit,
)

TESTS_DIR = str(Path(__file__).resolve().parent)


def _ps_args(pid: int) -> str:
    out = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return out.stdout.strip()


@pytest.fixture
def fake_exe(tmp_path, monkeypatch):
    """A factory: the thinking fake as a llama-server executable, FAKE_* knobs as keyword arguments. Every fake it
    starts records its pid in FAKE_PIDS_FILE; at teardown (a failed test included) any of those still alive whose argv
    names THIS test's executable is SIGKILLed, so a test never leaves an orphan fake behind. Nothing else is touched."""
    exe = tmp_path / "llama-server"
    pids_file = tmp_path / "fake-pids"

    def make(**env) -> Path:
        exe.write_text(exe_script(sys.executable, TESTS_DIR))
        exe.chmod(0o755)
        for k in ("FAKE_FAIL", "FAKE_THINK", "FAKE_ARGV_FILE", "FAKE_IGNORE_TERM", "FAKE_LOAD_DELAY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("FAKE_PIDS_FILE", str(pids_file))
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return exe

    yield make
    pids = [int(x) for x in pids_file.read_text().split()] if pids_file.exists() else []
    for pid in pids:
        if alive(pid) and str(exe) in _ps_args(pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)


def test_natural_len_is_the_planted_trace_length_of_the_calibrate_tests():
    assert natural_len("state 1") == natural_len("state 10") == 40 and natural_len("state 2") == natural_len("state 5") == 300


def test_served_model_runs_callo_serve_in_process_and_logs_to_the_models_own_file(tmp_path, monkeypatch, capsys, fake_exe):
    monkeypatch.setenv("HOME", str(tmp_path))                      # the vocabulary cache
    # a 1 s load holds the starting pid file open for served_model's 50 ms poll: on a slow runner an instant load can be
    # renamed before the first poll, and then `child` is None by design (only our own starting file counts)
    exe = fake_exe(FAKE_ARGV_FILE=str(tmp_path / "argv.json"), FAKE_LOAD_DELAY="1")
    model = tiny_gguf(tmp_path / "m.gguf")
    entry = Entry("m", str(model), ctx=4096, llama_server=str(exe), args=("-ctk", "q8_0"))
    cache_dir, log_path = tmp_path / "cache", tmp_path / "m.serve.log"
    stop = threading.Event()
    logging.getLogger("callo").info("driver line before")
    with served_model(entry, cache=False, cache_dir=cache_dir, log_path=log_path, stop=stop) as s:
        assert s.url == f"http://127.0.0.1:{s.port}" and s.thread.name == "callo-menu-m" and s.thread.is_alive()
        pid = s.child                                                           # remembered from the starting pid file
        assert pid is not None and alive(pid) and pid != os.getpid()
        assert [int(f.read_text()) for f in cache_dir.glob("llama-server-*.pid")] == [pid]   # renamed on bind: the same
        _, _, stats = get_json(f"{s.url}/callo/stats")
        assert stats["think"]["capable"] is True and stats["model_file"] == "m.gguf" and stats["temperature"] == 1.0
        assert stats["provenance"]["server_args"] and "-ctk" in stats["provenance"]["server_args"]
        assert "--cache-ram" in stats["provenance"]["server_args"]              # the host prompt cache off
        argv = json.loads((tmp_path / "argv.json").read_text())
        assert argv[:2] == ["-m", str(model)] and "-c" in argv and argv[argv.index("-c") + 1] == "4096" and "-ctk" in argv
        assert argv[argv.index("--cache-ram") + 1] == "0" and argv.index("--cache-ram") < argv.index("-ctk")
        logging.getLogger("callo").info("driver line during")
        other = threading.Thread(target=lambda: logging.getLogger("callo").warning("from another thread of serve"))
        other.start()
        other.join()
    assert not s.thread.is_alive() and wait_exit(pid, timeout=10) and not list(cache_dir.glob("llama-server-*.pid"))
    text = log_path.read_text()
    assert re.fullmatch(r"---- \S+Z callo serve m ----", text.splitlines()[0])       # each attempt opens with one
    assert "launched llama-server pid" in text and "driver line" not in text and " callo INFO " in text
    assert "from another thread of serve" in text                              # every non-driver thread's record
    err = capsys.readouterr().err
    assert "driver line during" in err and "launched llama-server pid" not in err and "another thread" not in err
    assert logging.getLogger("callo").propagate is True


def test_a_child_that_ignores_sigterm_is_killed_when_the_serve_is_stopped(tmp_path, monkeypatch, capsys, fake_exe):
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = fake_exe(FAKE_IGNORE_TERM="1", FAKE_LOAD_DELAY="1")       # the delay: `child` is seen (see above)
    model = tiny_gguf(tmp_path / "m.gguf")
    cache_dir = tmp_path / "cache"
    logged: list[str] = []
    with served_model(Entry("m", str(model), llama_server=str(exe)), cache=False, cache_dir=cache_dir,
                      log_path=tmp_path / "m.serve.log", stop=threading.Event(), log=logged.append) as s:
        pid = s.child
        assert pid is not None and alive(pid)
    assert wait_exit(pid, timeout=5) and not alive(pid)                       # SIGTERM ignored: the SIGKILL got it
    assert logged == [] and not list(cache_dir.glob("llama-server-*.pid"))    # serve's own escalation did it, not ours


@pytest.mark.parametrize("where", ["join", "kill"])
def test_a_second_ctrl_c_during_cleanup_still_restores_the_callo_logger(tmp_path, monkeypatch, fake_exe, where):
    """A KeyboardInterrupt (StopHandler's second signal) raised while the cleanup joins the serve thread or stops
    the child must not leak the thread-filtered handlers, `propagate` or the level; the interrupt still propagates."""
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = fake_exe(FAKE_LOAD_DELAY="1")                     # the delay: `child` is seen (see above)
    model = tiny_gguf(tmp_path / "m.gguf")
    logger = logging.getLogger("callo")
    before = (list(logger.handlers), logger.propagate, logger.level)

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        with served_model(Entry("m", str(model), llama_server=str(exe)), cache=False, cache_dir=tmp_path / "cache",
                          log_path=tmp_path / "m.serve.log", stop=threading.Event(),
                          kill=interrupt, log=lambda line: None) as s:
            thread, pid = s.thread, s.child
            assert pid is not None and alive(pid)
            if where == "join":
                monkeypatch.setattr(thread, "join", interrupt)
            else:                                             # the join returns at once with the thread still running:
                monkeypatch.setattr(thread, "join", lambda timeout=None: None)        # stop_child sends the "signal"
                monkeypatch.setattr(thread, "is_alive", lambda: True)
    assert (list(logger.handlers), logger.propagate, logger.level) == before
    monkeypatch.undo()                                        # the real join: serve stops its own child (stop is set)
    thread.join(timeout=30)
    assert not thread.is_alive() and wait_exit(pid, timeout=10)


def test_stop_child_escalates_from_sigterm_to_sigkill_with_injected_kill_and_clock():
    assert stop_child(None, kill=lambda p, s: None) == "had exited"
    assert stop_child(7, kill=lambda p, s: None, is_alive=lambda p: False) == "had exited"
    sent: list[int] = []
    life = iter([True, True, False])                                           # alive at the check, alive once, then gone
    assert stop_child(7, kill=lambda p, s: sent.append(s), is_alive=lambda p: next(life), sleep=lambda s: None,
                      clock=lambda: 0.0) == "exited on SIGTERM"
    assert sent == [signal.SIGTERM]
    sent.clear()
    ticks = iter([0.0, 11.0, 0.0, 11.0])                                       # both waits time out at the grace
    assert stop_child(7, kill=lambda p, s: sent.append(s), is_alive=lambda p: True, sleep=lambda s: None,
                      clock=lambda: next(ticks), grace=10.0) == "is still alive after SIGKILL"
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    sent.clear()
    stages = iter([True, True, True, False])
    ticks = iter([0.0, 11.0, 0.0, 0.5])
    assert stop_child(7, kill=lambda p, s: sent.append(s), is_alive=lambda p: next(stages), sleep=lambda s: None,
                      clock=lambda: next(ticks), grace=10.0) == "exited on SIGKILL"

    def gone(pid, sig):
        raise ProcessLookupError                                               # the group vanished between the check and the kill

    life = iter([True, False])
    assert stop_child(7, kill=gone, is_alive=lambda p: next(life), sleep=lambda s: None, clock=lambda: 0.0) == "exited on SIGTERM"


def test_a_load_failure_or_a_missing_gguf_is_a_menu_failure_naming_the_cause(tmp_path, monkeypatch, fake_exe):
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = fake_exe(FAKE_FAIL="1")
    model = tiny_gguf(tmp_path / "m.gguf")
    log_path = tmp_path / "m.serve.log"
    with pytest.raises(MenuFailure, match=r"did not start \(exit 1\): .*exited with 7 before listening") as e:
        with served_model(Entry("m", str(model), llama_server=str(exe)), cache=False, cache_dir=tmp_path / "c",
                          log_path=log_path, stop=threading.Event()):
            pytest.fail("the serve must not start")
    assert "planted load failure" in str(e.value) and "planted load failure" in log_path.read_text()
    monkeypatch.delenv("FAKE_FAIL")
    with pytest.raises(MenuFailure, match=r"did not start \(exit 1\): .*missing\.gguf"):
        with served_model(Entry("x", str(tmp_path / "missing.gguf"), llama_server=str(exe)), cache=False,
                          cache_dir=tmp_path / "c2", log_path=tmp_path / "x.serve.log", stop=threading.Event()):
            pytest.fail("the serve must not start")
    with pytest.raises(MenuFailure, match="--no-kvu needs --ctx"):
        with served_model(Entry("k", str(model), no_kvu=True, llama_server=str(exe)), cache=False,
                          cache_dir=tmp_path / "c3", log_path=tmp_path / "k.serve.log", stop=threading.Event()):
            pytest.fail("the serve must not start")
    (tmp_path / "e.log").write_text("2026 callo INFO a\n2026 callo ERROR the reason. Log tail:\nline one\nline two\n")
    assert last_error_line(tmp_path / "e.log") == "the reason. Log tail: | line one | line two"
    assert last_error_line(tmp_path / "e.log", limit=20) == "the reason. Log t..."
    (tmp_path / "f.log").write_text("only a line\n")
    assert last_error_line(tmp_path / "f.log") == "only a line" and last_error_line(tmp_path / "none.log").startswith("no log")


def test_child_pid_terminate_child_and_wait_exit_read_the_pid_files(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    assert loading_child_pid(cache) is None and terminate_child(cache) is None and wait_exit(None, timeout=0)
    (cache / "llama-server-8199.pid").write_text("4242\n")
    (cache / f"llama-server-starting-{os.getpid()}.pid").write_text("4343\n")
    assert loading_child_pid(cache) == 4343                          # ours only: the port pid file is never read
    killed = []
    assert terminate_child(cache, kill=lambda pid, sig: killed.append((pid, sig))) == 4343
    assert killed == [(4343, signal.SIGTERM)]
    (cache / f"llama-server-starting-{os.getpid()}.pid").unlink()
    assert loading_child_pid(cache) is None
    assert terminate_child(cache, kill=lambda pid, sig: killed.append((pid, sig))) is None   # serving: serve stops it
    assert len(killed) == 1
    for bad in ("garbage", "0", "1", "-5"):                            # killpg(0) is our own group: never a child
        (cache / f"llama-server-starting-{os.getpid()}.pid").write_text(f"{bad}\n")
        assert loading_child_pid(cache) is None and terminate_child(cache, kill=lambda pid, sig: killed.append(pid)) is None
    assert len(killed) == 1
    (cache / f"llama-server-starting-{os.getpid()}.pid").unlink()

    def raising(pid, sig):
        raise ProcessLookupError

    (cache / f"llama-server-starting-{os.getpid()}.pid").write_text("77\n")
    assert terminate_child(cache, kill=raising) == 77                 # an already-gone child is fine
    (cache / f"llama-server-starting-{os.getpid()}.pid").unlink()
    clock = iter([0.0, 0.1, 0.2, 5.0, 5.1])
    assert wait_exit(77, timeout=1.0, sleep=lambda s: None, clock=lambda: next(clock), is_alive=lambda p: False)
    assert not wait_exit(77, timeout=1.0, sleep=lambda s: None, clock=lambda: next(clock), is_alive=lambda p: True)
    assert alive(os.getpid()) and not alive(2**22)


def test_settle_and_cooldown_therm(tmp_path):
    slept, logged = [], []
    settle(seconds=30.0, cooldown=None, log=logged.append, sleep=slept.append)
    assert slept == [1.0] * 30 and logged == ["settling 30 s before the next model"]     # 1 s slices
    slept.clear()
    settle(seconds=2.5, cooldown=None, log=logged.append, sleep=slept.append)
    assert slept == [1.0, 1.0, 0.5]
    slept.clear()
    settle(seconds=0.0, cooldown=None, log=logged.append, sleep=slept.append)
    assert slept == []
    readings = iter([80, 95, 100])
    slept.clear()
    logged.clear()
    settle(seconds=30.0, cooldown="therm", log=logged.append, sleep=slept.append, therm=lambda: next(readings),
           clock=lambda: 0.0)
    assert slept == [1.0] * 10 and logged[-1] == "cooldown therm: CPU_Speed_Limit = 100, the next model starts from an idle GPU"
    slept.clear()
    settle(seconds=12.0, cooldown="therm", log=logged.append, sleep=slept.append, therm=lambda: None, clock=lambda: 0.0)
    assert slept == [1.0] * 12 and "no thermal data" in logged[-2] and logged[-1] == "settling 12 s before the next model"
    ticks = iter([0.0, 700.0])
    slept.clear()
    settle(seconds=30.0, cooldown="therm", log=logged.append, sleep=slept.append, therm=lambda: 90, clock=lambda: next(ticks))
    assert slept == [] and logged[-1] == "cooldown therm: CPU_Speed_Limit still 90 after 10 min; going on"


def test_the_stop_handler_sets_the_events_and_terminates_a_loading_child(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"llama-server-starting-{os.getpid()}.pid").write_text("5151\n")
    killed, logged = [], []
    h = StopHandler(cache, kill=lambda pid, sig: killed.append((pid, sig)), log=logged.append)
    current = threading.Event()
    h.current = current
    h(signal.SIGTERM, None)
    assert h.interrupted.is_set() and current.is_set() and killed == [(5151, signal.SIGTERM)] and h.signals == [signal.SIGTERM]
    assert logged == [f"signal {signal.SIGTERM}: {StopHandler.STOPPING}; SIGTERM to the loading llama-server "
                      f"pid 5151 (serve cannot stop it before the load returns)"]
    with pytest.raises(KeyboardInterrupt):                            # a second signal escalates: SIGKILL and out
        h(signal.SIGINT, None)
    assert killed[-1] == (5151, signal.SIGKILL) and logged[-1] == f"signal {signal.SIGINT} again: SIGKILL to llama-server pid 5151; exiting now"
    h = StopHandler(cache, kill=lambda pid, sig: killed.append((pid, sig)), log=logged.append)
    (cache / f"llama-server-starting-{os.getpid()}.pid").unlink()
    h.child = 6161                                                    # a serving child, registered by the driver
    h(signal.SIGINT, None)
    assert logged[-1] == f"signal {signal.SIGINT}: {StopHandler.STOPPING}" and killed[-1] == (5151, signal.SIGKILL)
    with pytest.raises(KeyboardInterrupt):
        h(signal.SIGINT, None)
    assert killed[-1] == (6161, signal.SIGKILL)
    h = StopHandler(cache, kill=lambda pid, sig: killed.append((pid, sig)), log=logged.append)
    h(signal.SIGINT, None)
    with pytest.raises(KeyboardInterrupt):
        h(signal.SIGINT, None)
    assert logged[-1] == f"signal {signal.SIGINT} again: exiting now"
    assert h.install() is True
    try:
        assert signal.getsignal(signal.SIGTERM) is h and signal.getsignal(signal.SIGINT) is h
    finally:
        h.restore()
    assert signal.getsignal(signal.SIGTERM) is not h and h.previous == {}
    off_main: list[bool] = []
    t = threading.Thread(target=lambda: off_main.append(StopHandler(cache).install()))
    t.start()
    t.join()
    assert off_main == [False]


# ---- estimate_s on a huge JSON integer, an unhashable `dropped` -------------------------------------------

def test_estimate_s_is_none_on_a_huge_integer_never_an_overflow(tmp_path):
    settings = _settings(tmp_path)
    good = _prof("k", "x")
    huge = 10**400                                                     # json.loads gives a Python int this size
    assert estimate_s({**good, "environment": {"run": {"decode_tok_s_median": huge}}}, settings, 6) is None
    assert estimate_s({**good, "budgets": {"0": {"lat_p50_ms": [huge]}}}, settings, 6) is None
    assert estimate_s(json.loads(json.dumps({**good, "budgets": {"0": {"lat_p50_ms": [huge]}}})), settings, 6) is None
    assert estimate_s({**good, "budgets": {"0": {"lat_p50_ms": [1e300]}}}, settings, 10**300) is None   # an inf estimate


def test_run_file_load_refuses_an_unhashable_dropped_entry(tmp_path):
    rf = _run_file(tmp_path)
    base = json.loads(rf.path.read_text())
    for i, dropped in enumerate(([["a"]], [{"a": 1}], [1], "a")):
        f = tmp_path / f"d{i}-run.json"
        f.write_text(json.dumps({**base, "dropped": dropped}))
        with pytest.raises(InvalidInput, match="`dropped` is not a list of names in `order`"):
            RunFile.load(f)


# ---- only our own child is ever signalled; loading-phase paths on a real fake ------------------------------------

def _serve_thread(name: str) -> threading.Thread | None:
    return next((t for t in threading.enumerate() if t.name == f"callo-menu-{name}"), None)


def _fake_pids(tmp_path) -> list[int]:
    f = tmp_path / "fake-pids"
    return [int(x) for x in f.read_text().split()] if f.exists() else []


def test_a_foreign_pid_file_in_the_cache_dir_is_never_signalled(tmp_path, monkeypatch, fake_exe):
    """A leftover `llama-server-<port>.pid` (a crashed earlier run) or another process's starting file naming an
    ALIVE pid (this test's own) is never read as our child: an early failure, a load failure and a clean serve send
    no signal at all through the injected `kill`."""
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = fake_exe(FAKE_FAIL="1")
    model = tiny_gguf(tmp_path / "m.gguf")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "llama-server-12345.pid").write_text(f"{os.getpid()}\n")
    (cache_dir / "llama-server-starting-99999999.pid").write_text(f"{os.getpid()}\n")
    sent: list[tuple[int, int]] = []

    def record(pid, sig):
        sent.append((pid, sig))

    def serve_once(entry):
        with served_model(entry, cache=False, cache_dir=cache_dir, log_path=tmp_path / f"{entry.name}.serve.log",
                          stop=threading.Event(), kill=record, log=lambda line: None) as s:
            return s.child

    with pytest.raises(MenuFailure, match=r"did not start \(exit 1\): --s1-slots must be >= 1"):   # before any launch
        serve_once(Entry("x", str(model), s1_slots=0, llama_server=str(exe)))
    assert _fake_pids(tmp_path) == []
    with pytest.raises(MenuFailure, match="planted load failure"):             # launched, then failed
        serve_once(Entry("m", str(model), llama_server=str(exe)))
    monkeypatch.delenv("FAKE_FAIL")
    child = serve_once(Entry("ok", str(model), llama_server=str(exe)))
    assert child is not None and child != os.getpid() and wait_exit(child, timeout=10)
    assert sent == []
    assert (cache_dir / "llama-server-12345.pid").exists() and (cache_dir / "llama-server-starting-99999999.pid").exists()


def test_a_stale_starting_file_of_ours_is_removed_when_dead_and_refused_when_alive(tmp_path, monkeypatch, fake_exe):
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = fake_exe(FAKE_LOAD_DELAY="1")                     # the delay: `child` is seen (see above)
    model = tiny_gguf(tmp_path / "m.gguf")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    ours = cache_dir / f"llama-server-starting-{os.getpid()}.pid"
    sent: list[tuple[int, int]] = []
    ours.write_text(f"{os.getpid()}\n")                                        # alive: an earlier child that survived
    with pytest.raises(MenuFailure, match=rf"^{re.escape(str(ours))} names pid {os.getpid()}, which is alive: if it is "
                                          rf"a llama-server of an earlier callo run, stop it; otherwise delete "
                                          rf"{re.escape(str(ours))}$"):
        with served_model(Entry("m", str(model), llama_server=str(exe)), cache=False, cache_dir=cache_dir,
                          log_path=tmp_path / "m.serve.log", stop=threading.Event(),
                          kill=lambda p, s: sent.append((p, s))):
            pytest.fail("must not launch")
    assert sent == [] and ours.exists() and _fake_pids(tmp_path) == [] and _serve_thread("m") is None
    dead = 2**22                                                               # PID_MAX_LIMIT: never allocated
    assert not alive(dead)
    ours.write_text(f"{dead}\n")                                               # dead: a leftover, removed
    with served_model(Entry("m", str(model), llama_server=str(exe)), cache=False, cache_dir=cache_dir,
                      log_path=tmp_path / "m.serve.log", stop=threading.Event(),
                      kill=lambda p, s: sent.append((p, s))) as s:
        assert s.child is not None and s.child != dead and s.child in _fake_pids(tmp_path)
    assert sent == [] and not ours.exists() and wait_exit(s.child, timeout=10)


@pytest.mark.parametrize("ignore_term, outcome", [(False, "exited on SIGTERM"), (True, "exited on SIGKILL")])
def test_a_start_timeout_stops_the_loading_child_with_stop_child(tmp_path, monkeypatch, fake_exe, ignore_term, outcome):
    """The start timeout on a fake still loading (FAKE_LOAD_DELAY): the serve thread outlives the join, so the cleanup's
    own `stop_child` signals the remembered child with the real `os.killpg`; a fake that ignores SIGTERM gets SIGKILL."""
    monkeypatch.setenv("HOME", str(tmp_path))
    env = {"FAKE_LOAD_DELAY": "120"} | ({"FAKE_IGNORE_TERM": "1"} if ignore_term else {})
    exe = fake_exe(**env)
    model = tiny_gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(menu_mod, "SERVE_JOIN_S", 0.3)
    monkeypatch.setattr(menu_mod, "CHILD_GRACE_S", 1.5)
    logged: list[str] = []
    with pytest.raises(MenuFailure, match=r"did not start within 10 s"):   # 10 s: a slow runner starts the fake first
        with served_model(Entry("m", str(model), llama_server=str(exe)), cache=False, cache_dir=tmp_path / "cache",
                          log_path=tmp_path / "m.serve.log", stop=threading.Event(), timeout=10.0, log=logged.append):
            pytest.fail("the serve must not start")
    [pid] = _fake_pids(tmp_path)
    assert logged == [f"m: the serve thread is still running and llama-server pid {pid} {outcome}"]
    thread = _serve_thread("m")
    if thread is not None:                                                     # the launcher saw its child die
        thread.join(timeout=15)
    assert _serve_thread("m") is None and wait_exit(pid, timeout=5)


def test_the_first_signal_sigterms_a_real_loading_child(tmp_path, monkeypatch, fake_exe):
    """StopHandler's first signal while the fake is still loading: `terminate_child`'s real `os.killpg` SIGTERMs our
    own child's group; the launcher sees it die before listening, so the start is a MenuFailure naming that."""
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = fake_exe(FAKE_LOAD_DELAY="120")
    model = tiny_gguf(tmp_path / "m.gguf")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(menu_mod, "SERVE_JOIN_S", 15.0)
    logged: list[str] = []
    stop = threading.Event()
    h = StopHandler(cache_dir, log=logged.append)
    h.current = stop

    def signal_when_loading():
        deadline = time.monotonic() + 20
        # wait for the fake itself to be up (it records its pid after the interpreter starts): on a slow runner a
        # SIGTERM right after the launcher's starting file can land before that, leaving no pid to check below
        while (loading_child_pid(cache_dir) is None or not _fake_pids(tmp_path)) and time.monotonic() < deadline:
            time.sleep(0.05)
        h(signal.SIGTERM, None)

    t = threading.Thread(target=signal_when_loading)
    t.start()
    try:
        with pytest.raises(MenuFailure, match=r"did not start \(exit 1\): .*exited with -15 before listening"):
            with served_model(Entry("m", str(model), llama_server=str(exe)), cache=False, cache_dir=cache_dir,
                              log_path=tmp_path / "m.serve.log", stop=stop, timeout=30.0, log=logged.append):
                pytest.fail("the serve must not start")
    finally:
        t.join()
    [pid] = _fake_pids(tmp_path)
    assert h.interrupted.is_set() and stop.is_set() and wait_exit(pid, timeout=5)
    assert logged == [f"signal {signal.SIGTERM}: {StopHandler.STOPPING}; SIGTERM to the loading llama-server "
                      f"pid {pid} (serve cannot stop it before the load returns)"]


def test_last_error_line_reads_only_the_last_attempt(tmp_path):
    f = tmp_path / "m.serve.log"
    sep = attempt_separator("m")
    assert re.fullmatch(r"---- \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ callo serve m ----", sep)
    f.write_text(f"{sep}\n2026 callo ERROR the old reason\n{sep}\n2026 callo INFO launched llama-server pid 9\n")
    assert last_error_line(f) == "2026 callo INFO launched llama-server pid 9"
    f.write_text(f"{sep}\n2026 callo ERROR the old reason\n{sep}\n")
    assert last_error_line(f).startswith("nothing logged by this attempt")
    f.write_text(f"{sep}\n2026 callo ERROR the old reason\n{sep}\n2026 callo ERROR the new reason\n")
    assert last_error_line(f) == "the new reason"


def test_the_dry_run_estimate_is_unknown_on_an_overflowing_rate(tmp_path):
    model = tiny_gguf(tmp_path / "m.gguf")
    key = load_model_key(model, None)
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "tiny.think.json").write_text(json.dumps(_prof(key, "x", decode=1e-308)))
    lines = dry_run_model_lines(Entry("m", str(model)), _settings(tmp_path, budgets=(0, 64, 256)), 6, budget=None,
                                headroom_gib=2.0, rates_dirs=[earlier], cache_dir=None)
    assert lines[2].startswith("  time: unknown until measured")


# ---- the driver: failure isolation, run.json, --resume with an appended candidate, exit codes ---------------------

from fake_upstream import planted_answer  # noqa: E402

from callo import menu  # noqa: E402
from callo.calibrate_bias import Refusal  # noqa: E402
from callo.calibrate_think import Outcome  # noqa: E402
from callo.menu import Driver, MenuOptions, cache_ram_of, stamp_profile, valid_profile  # noqa: E402
from callo.systemone import parse_request  # noqa: E402

CHOICES = [["one", "two", "three"], ["red", "green", "blue", "cyan"], ["yes", "no"], ["alpha", "beta", "gamma"]]
DEAD_PID = 2**22                           # Linux's PID_MAX_LIMIT (pids stay below it) and above macOS's 99998: never live


def write_items(path: Path, n: int = 6) -> Path:
    """n labelled 'choice' records whose label is the thinking fake's planted answer, two per cluster."""
    lines = []
    for i in range(n):
        crit = {f"o{j}": t for j, t in enumerate(CHOICES[i % len(CHOICES)])}
        q = parse_request({"state": f"state {i}", "questions": {"q": {"type": "choice", "instructions": f"pick {i}",
                                                                      "criteria": crit}}}).questions[0]
        lines.append(json.dumps({"state": f"state {i}", "questions": {"q": {"type": "choice", "instructions": f"pick {i}",
                                                                           "criteria": crit, "label": f"o{planted_answer(list(q.texts))}"}},
                                 "_meta": {"id": f"rec-{i}", "group_id": f"g{i // 2}", "variant": "clean"}}))
    path.write_text("\n".join(lines) + "\n")
    return path


def _menu_settings(data: Path, **over) -> ThinkSettings:
    base = {"budgets": (0, 32, 64), "warmup": 0, "audit": 2}
    base.update(over)
    return ThinkSettings(str(data), data.name, data_sha256(data), **base)


def _opts(tmp_path, **over) -> MenuOptions:
    base = {"out_dir": tmp_path / "think", "headroom_gib": 0.0, "gpu_mem_gib": 64.0, "settle_s": 0.0}
    base.update(over)
    return MenuOptions(**base)


BENIGN_PS = "    1 /sbin/launchd\n  412 /usr/sbin/cfprefsd agent\n"


def _fake_listing(tmp_path, exe: Path, extra: str = ""):
    """`menu.ps_listing` for a test: benign lines, `extra`, and every fake this test launched that is still alive,
    written as a NATIVE llama-server line (`<pid> <exe> --port 0`, which the GPU rule always counts), so a driver
    that did not exclude its own child would refuse it. The box's real process list is never read."""
    def listing(run=None):
        mine = "".join(f"{pid} {exe} --port 0\n" for pid in _fake_pids(tmp_path) if alive(pid))
        return BENIGN_PS + extra + mine, None
    return listing


def test_a_menu_run_isolates_a_failed_model_writes_run_json_and_resumes_only_the_new_work(tmp_path, monkeypatch, fake_exe):
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = fake_exe()
    monkeypatch.setattr(menu, "ps_listing", _fake_listing(tmp_path, exe))     # the real scan, a fake ps
    good = tiny_gguf(tmp_path / "good.gguf")
    other = tiny_gguf(tmp_path / "other.gguf")
    data = write_items(tmp_path / "d.jsonl", 6)
    settings = _menu_settings(data)
    launch = Launch(ctx=4096, llama_server=str(exe))
    broken = [Entry("a", str(good), **{k: getattr(launch, k) for k in ("ctx", "llama_server")}),
              Entry("b", str(tmp_path / "missing.gguf"), ctx=4096, llama_server=str(exe)),
              Entry("c", str(other), ctx=4096, llama_server=str(exe), think_sampling="fixed",
                    args=("--cache-ram", "8192"))]                           # an explicit --cache-ram wins
    log: list[str] = []
    d = Driver(settings, broken, _opts(tmp_path), log=log.append)
    try:
        rc = d.run()
    finally:
        assert not [p for p in _fake_pids(tmp_path) if alive(p)]             # every launched fake was stopped
    assert rc == 1, "\n".join(log)
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["callo_think_run"] == 1 and run["order"] == ["a", "b", "c"] and run["settings"] == settings.comparable()
    assert run["models"]["a"]["status"] == "done" and run["models"]["c"]["status"] == "done"
    assert run["models"]["b"]["status"] == "failed" and "did not start (exit 1)" in run["models"]["b"]["reason"]
    assert run["models"]["a"]["profile"] == str(tmp_path / "think" / "a.think.json") and run["models"]["b"]["profile"] is None
    assert run["models"]["a"]["started"] <= run["models"]["a"]["ended"]
    a = json.loads((tmp_path / "think" / "a.think.json").read_text())
    c = json.loads((tmp_path / "think" / "c.think.json").read_text())
    assert a["environment"]["server"]["launched_by"] == "menu" and a["environment"]["server"]["gpu_busy"] is False
    assert a["environment"]["server"]["cache_ram"] == 0 and "--cache-ram" in a["environment"]["server"]["server_args"]
    assert c["environment"]["server"]["cache_ram"] == 8192                    # what the server actually ran with
    assert a["comparable"]["sampling"]["policy"] == "model" and c["comparable"]["sampling"]["policy"] == "fixed"
    assert a["model_name"] == "a" and sorted(a["budgets"]) == ["0", "32", "64"] and a["budgets"]["64"]["n"] == 6
    assert a["environment"]["server"]["server_args"] is not None and a["environment"]["server"]["n_ctx"] == 4096
    assert (tmp_path / "think" / "a.serve.log").exists() and (tmp_path / "think" / "c.serve.log").exists()
    assert (tmp_path / "think" / "b.serve.log").exists() and (tmp_path / "think" / "a.raw.jsonl").exists()
    report = json.loads((tmp_path / "think" / "report.json").read_text())
    assert [p["name"] for p in report["profiles"]] == ["a", "c"] and d.report_md is not None and "| a@0 |" in d.report_md
    assert [r["name"] for r in report["runs"]] == ["a", "b", "c"] and report["runs"][1]["status"] == "failed"
    assert "## Runs" in d.report_md and "Failed and skipped models:\n- b: failed (" in d.report_md
    assert "| a | menu | 4 | unified | 4096 | 0 | " in d.report_md             # the launch shape per profile
    assert "different host prompt-cache settings" in d.report_md and "a 0, c 8192" in d.report_md
    assert [str(p) for p in (tmp_path / "think").rglob("report.md")] == [str(tmp_path / "think" / "report.md")]
    assert log[-3:] == ["a: done", "b: failed (" + run["models"]["b"]["reason"] + ")", "c: done"]   # the summary, exact
    assert any(ln.startswith("report: ") for ln in log)
    assert not list((tmp_path / "think" / menu.CACHE_DIR_NAME).glob("llama-server-*.pid"))
    # --resume with b's path fixed and a candidate appended: only b and d run; the report covers all four
    fixed = [broken[0], Entry("b", str(good), ctx=4096, llama_server=str(exe)), broken[2],
             Entry("d", str(other), ctx=4096, llama_server=str(exe))]
    log2: list[str] = []
    d2 = Driver(settings, fixed, _opts(tmp_path, resume=True), log=log2.append)
    assert d2.run() == 0, "\n".join(log2)
    run2 = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run2["order"] == ["a", "b", "c", "d"] and {m["status"] for m in run2["models"].values()} == {"done"}
    assert run2["models"]["a"]["started"] == run["models"]["a"]["started"] and run2["models"]["c"]["ended"] == run["models"]["c"]["ended"]
    assert run2["menu_sha256"] == menu_sha256(fixed) != run["menu_sha256"]
    assert "a: done already" in "\n".join(log2) and "resume: new menu entries d are pending" in "\n".join(log2)
    assert not any(ln.startswith(("a: launching", "c: launching")) for ln in log2)
    assert [p["name"] for p in json.loads((tmp_path / "think" / "report.json").read_text())["profiles"]] == ["a", "b", "c", "d"]
    # a third resume with nothing to do only regenerates the report
    d3 = Driver(settings, fixed, _opts(tmp_path, resume=True), log=lambda s: None)
    assert d3.run() == 0 and d3.report_md is not None
    assert not [p for p in _fake_pids(tmp_path) if alive(p)]


def _stub_launcher(url: str = "http://127.0.0.1:1", child: int | None = None):
    @contextlib.contextmanager
    def launcher(entry, *, cache, cache_dir, log_path, stop, log=None):
        log_path.write_text("stub\n")
        yield menu.Served(url, 1, log_path, threading.current_thread(), child)
    return launcher


def _stub_calibrate(out: dict, status: str = "done"):
    def calibrate_one(settings, *, url, name, out_dir, resume, log, **kw):
        out[name] = {"url": url, "sampling": settings.sampling, "resume": resume}
        if status != "done":
            return Outcome(status, f"planted {status}", None)
        path = Path(out_dir) / f"{name}.think.json"
        path.write_text(json.dumps({"callo_think_profile": 1, "model_name": name, "environment": {"server": {}}}))
        return Outcome("done", None, path)
    return calibrate_one


HOST = {"memory_bytes": 64 * 2**30}


def test_refusals_before_any_model_runs(tmp_path, monkeypatch):
    data = write_items(tmp_path / "d.jsonl", 4)
    entries = [Entry("a", str(tiny_gguf(tmp_path / "a.gguf")), llama_server=str(tmp_path / "no-such-llama-server"))]
    out = tmp_path / "think"
    out.mkdir()
    (out / "stale.txt").write_text("x")
    with pytest.raises(Refusal, match="is not empty. Pass --resume"):
        Driver(_menu_settings(data), entries, _opts(tmp_path), log=lambda s: None, scan_fn=lambda **kw: []).run()
    (out / "stale.txt").unlink()
    with pytest.raises(Refusal, match="another model server is running .pid 9: llama-server x.. Stop them .*or pass --allow-gpu-busy"):
        Driver(_menu_settings(data), entries, _opts(tmp_path), log=lambda s: None,
               scan_fn=lambda **kw: ["pid 9: llama-server x"]).run()
    assert not list(out.iterdir())                                     # the GPU rule ran before run.json was written
    # ps itself failing is a refusal before the first model too (the real scan, a failing fake ps)
    monkeypatch.setattr(menu, "ps_listing", lambda run=None: (None, "exit 1: boom"))
    with pytest.raises(Refusal, match=r"could not list processes \(ps failed: exit 1: boom\)"):
        Driver(_menu_settings(data), entries, _opts(tmp_path), log=lambda s: None, host=HOST).run()
    assert not list(out.iterdir())
    (out / "run.json").write_text(json.dumps({"callo_think_run": 1, "menu_sha256": "x", "order": [], "models": {},
                                              "settings": {**_menu_settings(data).comparable(), "seed": 9}}))
    with pytest.raises(Refusal, match="settings differ from .* \\(seed\\)"):
        Driver(_menu_settings(data), entries, _opts(tmp_path, resume=True), log=lambda s: None, scan_fn=lambda **kw: []).run()
    # every run recorded before --cache became the default is cache off: resumed with no flag (now on), the refusal names the fix
    (out / "run.json").write_text(json.dumps({"callo_think_run": 1, "menu_sha256": "x", "order": [], "models": {},
                                              "settings": _menu_settings(data, cache=False).comparable()}))
    with pytest.raises(Refusal, match=r"\(cache\).*pass --no-cache to resume a cache-off run"):
        Driver(_menu_settings(data), entries, _opts(tmp_path, resume=True), log=lambda s: None, scan_fn=lambda **kw: []).run()
    (out / "run.json").write_text("{}")
    with pytest.raises(Refusal, match="refusing to resume: .*not a callo think run file"):
        Driver(_menu_settings(data), entries, _opts(tmp_path, resume=True), log=lambda s: None, scan_fn=lambda **kw: []).run()


def test_gpu_busy_allowed_precheck_skips_mid_run_servers_and_the_exit_codes(tmp_path, monkeypatch):
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    big = tmp_path / "big.gguf"
    big.write_bytes(b"\0" * (3 * 2**20))
    entries = [Entry("a", str(tiny_gguf(tmp_path / "a.gguf")), think_sampling="fixed"), Entry("big", str(big)),
               Entry("c", str(tiny_gguf(tmp_path / "c.gguf")))]
    log: list[str] = []
    d = Driver(_menu_settings(data), entries, _opts(tmp_path, allow_gpu_busy=True, headroom_gib=0.0, gpu_mem_gib=2 / 1024),
               log=log.append, launcher=_stub_launcher(), scan_fn=lambda **kw: ["pid 9: llama-server x"], host=HOST)
    assert d.run() == 0
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["big"]["status"] == "skipped" and run["models"]["big"]["reason"].startswith("needs ~0.0 GiB, budget 0.0 GiB")
    assert run["models"]["a"]["status"] == run["models"]["c"]["status"] == "done"
    assert seen["a"]["sampling"] == "fixed" and seen["c"]["sampling"] == "model" and seen["a"]["resume"] is True
    a_env = json.loads((tmp_path / "think" / "a.think.json").read_text())["environment"]
    assert a_env["server"] == {"launched_by": "menu", "gpu_busy": True, "cache_ram": 0,     # and the launch shape
                               "launch": {"ctx": None, "parallel": 4, "no_kvu": False, "s1_slots": 1,
                                          "user_delimiter": None, "args": [], "position_bias": None,
                                          "llama_server": "llama-server"}}
    assert a_env["model"] == {"path": str(tmp_path / "a.gguf"), "mmproj_path": None}
    assert any(ln.startswith("WARNING: another model server is running") for ln in log)
    # a server that appears mid-run fails the current model; a later one fails before launching; exit 1
    calls = iter([[], ["pid 7: ollama serve"], ["pid 7: ollama serve"]])   # first check; after a; before c
    d2 = Driver(_menu_settings(data), [entries[0], entries[2]], _opts(tmp_path, out_dir=tmp_path / "t2"), log=log.append,
                launcher=_stub_launcher(), scan_fn=lambda **kw: next(calls), host=HOST)
    assert d2.run() == 1
    run2 = json.loads((tmp_path / "t2" / "run.json").read_text())
    assert run2["models"]["a"]["status"] == "failed" and run2["models"]["a"]["reason"] == "another model server appeared (pid 7: ollama serve)"
    assert run2["models"]["a"]["profile"].endswith("a.think.json")           # the profile is kept, the status says why
    assert run2["models"]["c"]["status"] == "failed" and run2["models"]["c"]["reason"] == "another model server is running (pid 7: ollama serve)"
    # --model: launched_by "model"; a refused calibrate is that model's failure, named
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen, "refused"))
    d3 = Driver(_menu_settings(data), [entries[0]], _opts(tmp_path, out_dir=tmp_path / "t3", launched_by="model"),
                log=log.append, launcher=_stub_launcher(), scan_fn=lambda **kw: [], host=HOST)
    assert d3.run() == 1
    assert json.loads((tmp_path / "t3" / "run.json").read_text())["models"]["a"]["reason"] == "a: planted refused"
    assert d3.report_md is not None and "No profile was written" in d3.report_md and "- a: failed (a: planted refused)" in d3.report_md
    assert json.loads((tmp_path / "t3" / "report.json").read_text())["runs"][0]["status"] == "failed"
    restamped = stamp_profile(Path(tmp_path / "think" / "c.think.json"), launched_by="model", gpu_busy=False)
    assert {k: restamped["environment"]["server"][k] for k in ("launched_by", "gpu_busy", "cache_ram")} == {
        "launched_by": "model", "gpu_busy": False, "cache_ram": 0}
    assert valid_profile(str(tmp_path / "think" / "c.think.json")) and not valid_profile(str(tmp_path / "think" / "run.json"))
    assert not valid_profile(None)


def test_an_unexpected_exception_fails_that_model_only_with_its_traceback_in_the_serve_log(tmp_path, monkeypatch):
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    good = _stub_calibrate(seen)

    def calibrate_one(settings, *, url, name, out_dir, resume, log, **kw):
        if name == "a":
            raise OSError(28, "No space left on device")                     # ENOSPC while writing the raw file
        return good(settings, url=url, name=name, out_dir=out_dir, resume=resume, log=log, **kw)

    monkeypatch.setattr(menu, "calibrate_one", calibrate_one)
    entries = [Entry("a", str(tiny_gguf(tmp_path / "a.gguf"))), Entry("b", str(tiny_gguf(tmp_path / "b.gguf")))]
    log: list[str] = []
    d = Driver(_menu_settings(data), entries, _opts(tmp_path), log=log.append, launcher=_stub_launcher(child=DEAD_PID),
               scan_fn=lambda **kw: [], host=HOST)
    children: list = []
    real_run_model = d.run_model

    def run_model(entry, rf, budget):
        real_run_model(entry, rf, budget)
        children.append(d.handler.child)

    d.run_model = run_model
    assert d.run() == 1
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["a"]["status"] == "failed" and run["models"]["a"]["reason"] == "OSError: [Errno 28] No space left on device"
    assert run["models"]["b"]["status"] == "done" and list(seen) == ["b"]
    assert "Traceback" in (tmp_path / "think" / "a.serve.log").read_text() and "No space left" in (tmp_path / "think" / "a.serve.log").read_text()
    assert any(ln.startswith("a: failed: OSError") and "traceback in" in ln for ln in log)
    assert "a: calibrate failed: OSError: [Errno 28]" in (tmp_path / "think" / "a.serve.log").read_text()
    assert children == [None, None]              # the child is cleared on every path, a raise included


def test_an_exception_after_the_measurement_fails_that_model_only(tmp_path, monkeypatch):
    """Per-model isolation covers the post-scan and the profile stamp too (an ENOSPC rewriting the profile, a scan bug)."""
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    real_stamp = menu.stamp_profile

    def stamp(path, **kw):
        if Path(path).name == "a.think.json":
            raise OSError(28, "No space left on device")
        return real_stamp(path, **kw)

    monkeypatch.setattr(menu, "stamp_profile", stamp)
    entries = [Entry("a", str(tiny_gguf(tmp_path / "a.gguf"))), Entry("b", str(tiny_gguf(tmp_path / "b.gguf")))]
    log: list[str] = []
    d = Driver(_menu_settings(data), entries, _opts(tmp_path), log=log.append, launcher=_stub_launcher(),
               scan_fn=lambda **kw: [], host=HOST)
    assert d.run() == 1
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["a"]["status"] == "failed" and run["models"]["a"]["reason"].startswith("OSError: [Errno 28]")
    assert run["models"]["b"]["status"] == "done" and "Traceback" in (tmp_path / "think" / "a.serve.log").read_text()
    assert "a: stamp failed: OSError: [Errno 28]" in (tmp_path / "think" / "a.serve.log").read_text()


def test_an_interrupt_stops_after_the_current_model_with_exit_130(tmp_path, monkeypatch):
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    entries = [Entry("a", str(tiny_gguf(tmp_path / "a.gguf"))), Entry("b", str(tiny_gguf(tmp_path / "b.gguf")))]
    log: list[str] = []
    driver = Driver(_menu_settings(data), entries, _opts(tmp_path), log=log.append, scan_fn=lambda **kw: [], host=HOST)

    @contextlib.contextmanager
    def launcher(entry, *, cache, cache_dir, log_path, stop, log=None):
        driver.handler(signal.SIGINT, None)                                 # a Ctrl-C while the model is up
        assert stop.is_set()
        yield menu.Served("http://127.0.0.1:1", 1, log_path, threading.current_thread())

    driver.launcher = launcher
    assert driver.run() == 130
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["a"]["status"] == "done" and run["models"]["b"]["status"] == "pending" and list(seen) == ["a"]
    assert log[-1].startswith("interrupted; ") and "--resume" in log[-1]
    assert signal.getsignal(signal.SIGINT) is not driver.handler


def test_a_second_signal_with_no_known_child_sends_no_kill_and_exits_130(tmp_path, monkeypatch):
    """`Served.child` is None when serve's child was never seen in OUR starting file; the second signal then only
    raises KeyboardInterrupt (served_model's finally and serve's own stop the child), never a kill."""
    data = write_items(tmp_path / "d.jsonl", 4)
    entries = [Entry("a", str(tiny_gguf(tmp_path / "a.gguf"))), Entry("b", str(tiny_gguf(tmp_path / "b.gguf")))]
    log: list[str] = []
    driver = Driver(_menu_settings(data), entries, _opts(tmp_path), log=log.append, launcher=_stub_launcher(child=None),
                    scan_fn=lambda **kw: [], host=HOST)
    sent: list = []
    driver.handler.kill = lambda pid, sig: sent.append((pid, sig))
    raised: list[bool] = []

    def calibrate_one(settings, *, url, name, out_dir, resume, log, **kw):
        assert driver.handler.child is None
        driver.handler(signal.SIGINT, None)
        try:
            driver.handler(signal.SIGINT, None)
        except KeyboardInterrupt:
            raised.append(True)
            raise
        pytest.fail("the second signal must raise")

    monkeypatch.setattr(menu, "calibrate_one", calibrate_one)
    assert driver.run() == 130
    assert raised == [True] and sent == [] and driver.handler.child is None
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["a"]["status"] == "failed" and run["models"]["a"]["reason"].startswith("interrupted")
    assert run["models"]["b"]["status"] == "pending"
    assert f"signal {signal.SIGINT} again: exiting now" in log


def test_the_post_scan_excludes_the_drivers_own_child(tmp_path, monkeypatch):
    """The real scan over a fake ps listing that shows our child as a native llama-server (that listing line
    is all the scan sees of it): excluded, so the model is done; another llama-server appearing fails it."""
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    child = DEAD_PID
    scans: list = []
    real_scan = menu.scan

    def recording_scan(**kw):
        scans.append(set(kw.get("exclude") or ()))
        return real_scan(**kw)

    monkeypatch.setattr(menu, "scan", recording_scan)               # the default scan_fn resolves menu.scan at call time
    state = {"post": False}

    @contextlib.contextmanager
    def launcher(entry, *, cache, cache_dir, log_path, stop, log=None):
        state["post"] = True                                        # from here on ps lists our child
        yield menu.Served("http://127.0.0.1:1", 1, log_path, threading.current_thread(), child)

    def listing(run=None):
        mine = f"{child} /opt/llama.cpp/bin/llama-server -m /m.gguf --port 41234\n" if state["post"] else ""
        return BENIGN_PS + mine, None

    monkeypatch.setattr(menu, "ps_listing", listing)
    entries = [Entry("a", str(tiny_gguf(tmp_path / "a.gguf")))]
    d = Driver(_menu_settings(data), entries, _opts(tmp_path), log=lambda s: None, launcher=launcher, host=HOST)
    assert d.run() == 0
    assert json.loads((tmp_path / "think" / "run.json").read_text())["models"]["a"]["status"] == "done"
    assert scans == [set(), {child}]                                # the first check, then the post-scan with our child
    state["post"] = False
    other = "4321 /opt/llama.cpp/bin/llama-server -m /other.gguf\n"
    monkeypatch.setattr(menu, "ps_listing", lambda run=None: (BENIGN_PS + (other if state["post"] else ""), None))
    d2 = Driver(_menu_settings(data), entries, _opts(tmp_path, out_dir=tmp_path / "t2"), log=lambda s: None,
                launcher=launcher, host=HOST)
    assert d2.run() == 1
    assert json.loads((tmp_path / "t2" / "run.json").read_text())["models"]["a"]["reason"] == (
        "another model server appeared (pid 4321: /opt/llama.cpp/bin/llama-server -m /other.gguf)")


def test_a_later_ps_failure_fails_that_model_only(tmp_path, monkeypatch):
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    answers = iter([(BENIGN_PS, None), (BENIGN_PS, None), (None, "timed out after 30 s")])   # first; after a; before b
    monkeypatch.setattr(menu, "ps_listing", lambda run=None: next(answers))
    entries = [Entry("a", str(tiny_gguf(tmp_path / "a.gguf"))), Entry("b", str(tiny_gguf(tmp_path / "b.gguf")))]
    d = Driver(_menu_settings(data), entries, _opts(tmp_path), log=lambda s: None, launcher=_stub_launcher(), host=HOST)
    assert d.run() == 1
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["a"]["status"] == "done" and list(seen) == ["a"]
    assert run["models"]["b"]["status"] == "failed"
    assert run["models"]["b"]["reason"].startswith("another model server is running (could not list processes (ps failed: "
                                                   "timed out after 30 s)")


def test_resume_with_a_menu_that_drops_a_done_model_runs_only_the_current_menu(tmp_path, monkeypatch):
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    a, b, c = (Entry(n, str(tiny_gguf(tmp_path / f"{n}.gguf"))) for n in "abc")
    kw = {"launcher": _stub_launcher(), "scan_fn": lambda **k: [], "host": HOST}
    assert Driver(_menu_settings(data), [a, b], _opts(tmp_path), log=lambda s: None, **kw).run() == 0
    assert list(seen) == ["a", "b"]
    seen.clear()
    log: list[str] = []
    d = Driver(_menu_settings(data), [b, c], _opts(tmp_path, resume=True), log=log.append, **kw)
    assert d.run() == 0
    assert list(seen) == ["c"]                                            # a is history: never run again
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["order"] == ["b", "c", "a"] and run["dropped"] == ["a"] and run["models"]["a"]["status"] == "done"
    report = json.loads((tmp_path / "think" / "report.json").read_text())
    assert [r["name"] for r in report["runs"]] == ["b", "c"]
    assert "not in this menu: a (kept in run.json, left out of the report)" in log
    assert log[-2:] == ["b: done", "c: done"] and not any(ln.startswith("a: ") for ln in log)


def test_cache_ram_of_reads_the_last_cache_ram_in_the_launch_argv():
    base = Entry("m", "/m.gguf")
    assert cache_ram_of(base.serve_argv(cache=False, cache_dir=Path("/c"))) == 0
    assert cache_ram_of(Entry("m", "/m.gguf", args=("--cache-ram", "8192")).serve_argv(cache=False, cache_dir=Path("/c"))) == 8192
    assert cache_ram_of(["serve", "--", "--cache-ram", "0", "--cache-ram=4096"]) == 4096
    assert cache_ram_of(["serve", "--", "--cache-ram", "0", "-cram", "-1"]) == -1
    assert cache_ram_of(["serve", "--", "--cache-ram", "0", "--cache-ram", "lots"]) is None
    assert cache_ram_of(["serve", "--", "--cache-ram", "0", "--cache-ram"]) is None
    assert cache_ram_of(["serve", "--model", "/m.gguf"]) is None                # no llama-server args at all
    assert cache_ram_of(["serve", "--cache-ram", "5", "--"]) is None           # only after `--` is llama-server's


# ---- a child seen only after the join, signalled alone among planted foreign files --------------------------------

from callo.upstream import LlamaServerProcess  # noqa: E402


def test_a_child_launched_during_the_join_is_stopped_and_no_foreign_pid_is_signalled(tmp_path, monkeypatch, fake_exe):
    """The thread-still-alive branch, ordered by events, not delays: serve's launch waits until the cleanup's last
    pre-join read of OUR starting file (stop is set by then), so the child appears only during the join (5 s, against a
    launch of milliseconds); the cleanup reads OUR starting file once more and stop_child signals that pid alone (the
    injected kill records instead of killing), never the planted foreign pid files' (this test's own pid)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = fake_exe(FAKE_LOAD_DELAY="120")
    model = tiny_gguf(tmp_path / "m.gguf")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "llama-server-12345.pid").write_text(f"{os.getpid()}\n")
    (cache_dir / "llama-server-starting-99999999.pid").write_text(f"{os.getpid()}\n")
    monkeypatch.setattr(menu_mod, "SERVE_JOIN_S", 5.0)
    monkeypatch.setattr(menu_mod, "CHILD_GRACE_S", 0.2)
    stop = threading.Event()
    go = threading.Event()
    real_start, real_read = LlamaServerProcess.start, menu_mod.loading_child_pid

    def late_start(self):
        go.wait(timeout=60)                                             # until the pre-join read has happened
        return real_start(self)

    def read(cache_dir):
        pid = real_read(cache_dir)
        if stop.is_set():                                               # the cleanup's pre-join read: launch now
            go.set()
        return pid

    monkeypatch.setattr(LlamaServerProcess, "start", late_start)
    monkeypatch.setattr(menu_mod, "loading_child_pid", read)
    sent: list[tuple[int, int]] = []
    logged: list[str] = []
    try:
        with pytest.raises(MenuFailure, match=r"did not start within 1 s"):
            with served_model(Entry("m", str(model), llama_server=str(exe)), cache=False, cache_dir=cache_dir,
                              log_path=tmp_path / "m.serve.log", stop=stop, timeout=1.0,
                              kill=lambda p, s: sent.append((p, s)), log=logged.append):
                pytest.fail("the serve must not start")
        [pid] = _fake_pids(tmp_path)
        assert sent == [(pid, signal.SIGTERM), (pid, signal.SIGKILL)] and os.getpid() not in {p for p, _ in sent}
        assert logged == [f"m: the serve thread is still running and llama-server pid {pid} is still alive after SIGKILL"]
    finally:
        go.set()
        for pid in _fake_pids(tmp_path):                                    # only the fake this test launched
            if alive(pid):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
        thread = _serve_thread("m")
        if thread is not None:
            thread.join(timeout=15)
    assert _serve_thread("m") is None
    assert (cache_dir / "llama-server-12345.pid").exists() and (cache_dir / "llama-server-starting-99999999.pid").exists()


# ---- run.json write failures, scans that raise, menu values, Ctrl-C between models ---------------------------------

from callo import cli as cli_mod  # noqa: E402


def _two(tmp_path):
    return [Entry("a", str(tiny_gguf(tmp_path / "a.gguf"))), Entry("b", str(tiny_gguf(tmp_path / "b.gguf")))]


def test_a_run_json_that_cannot_be_written_stops_the_run_with_its_report_and_exit_1(tmp_path, monkeypatch):
    """I-1: calibrate raises, and the failure handler's own run.json write hits ENOSPC too: the run stops (b never
    runs), says so, still writes the report and the summary, and returns 1; the OSError never escapes run()."""
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}

    def calibrate_one(settings, *, url, name, out_dir, resume, log, **kw):
        seen[name] = True
        raise ValueError("an unexpected /callo/stats shape")

    monkeypatch.setattr(menu, "calibrate_one", calibrate_one)
    real_set = RunFile.set

    def failing_set(self, name, status, **kw):
        if status == "failed":
            raise OSError(28, "No space left on device")
        return real_set(self, name, status, **kw)

    monkeypatch.setattr(RunFile, "set", failing_set)
    log: list[str] = []
    d = Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path), log=log.append, launcher=_stub_launcher(),
               scan_fn=lambda **kw: [], host=HOST)
    assert d.run() == 1
    assert list(seen) == ["a"]                                               # stopped: b never ran
    assert any(ln.startswith("cannot write run.json (") and "No space left" in ln
               and ln.endswith("): stopping; --resume continues") for ln in log)
    assert d.report_md is not None and (tmp_path / "think" / "report.md").exists()
    assert log[-2:] == ["a: running", "b: pending"]                          # the summary, from the in-memory state
    on_disk = json.loads((tmp_path / "think" / "run.json").read_text())
    assert on_disk["models"]["a"]["status"] == "running"                     # the last write that succeeded
    # the report's own write failing is only logged
    monkeypatch.setattr(RunFile, "set", real_set)
    monkeypatch.setattr(menu.compare, "write_stub_report", lambda *a, **k: (_ for _ in ()).throw(OSError(28, "full")))
    log.clear()
    d2 = Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path, out_dir=tmp_path / "t2"), log=log.append,
                launcher=_stub_launcher(), scan_fn=lambda **kw: [], host=HOST)
    assert d2.run() == 1 and any(ln.startswith("WARNING: could not write the report: ") for ln in log)
    assert log[-2:] == ["a: failed (ValueError: an unexpected /callo/stats shape)",
                        "b: failed (ValueError: an unexpected /callo/stats shape)"]


def test_a_later_scan_that_raises_fails_that_model_only(tmp_path, monkeypatch):
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    entries = [*_two(tmp_path), Entry("c", str(tiny_gguf(tmp_path / "c.gguf")))]
    calls = iter([[], [], "raise", [], []])                 # first check; post a; pre b (raises); pre c; post c

    def scan_fn(**kw):
        x = next(calls)
        if x == "raise":
            raise FileNotFoundError(2, "No such file or directory", "llama-server-8080.pid")
        return x

    d = Driver(_menu_settings(data), entries, _opts(tmp_path), log=lambda s: None, launcher=_stub_launcher(),
               scan_fn=scan_fn, host=HOST)
    assert d.run() == 1
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["b"]["status"] == "failed" and run["models"]["b"]["reason"].startswith("FileNotFoundError: [Errno 2]")
    assert run["models"]["a"]["status"] == run["models"]["c"]["status"] == "done" and list(seen) == ["a", "c"]
    assert "b: pre-scan failed: FileNotFoundError" in (tmp_path / "think" / "b.serve.log").read_text()


def test_a_post_scan_that_raises_fails_that_model_only(tmp_path, monkeypatch):
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    calls = iter([[], "raise", [], []])                      # first check; post a (raises); pre b; post b

    def scan_fn(**kw):
        x = next(calls)
        if x == "raise":
            raise PermissionError(13, "Permission denied")
        return x

    d = Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path), log=lambda s: None, launcher=_stub_launcher(),
               scan_fn=scan_fn, host=HOST)
    assert d.run() == 1
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["a"]["status"] == "failed" and run["models"]["a"]["reason"] == "PermissionError: [Errno 13] Permission denied"
    assert run["models"]["b"]["status"] == "done"
    assert "a: post-scan failed: PermissionError" in (tmp_path / "think" / "a.serve.log").read_text()


def test_a_menu_value_argparse_refuses_is_a_menu_failure_and_a_leading_dash_value_parses(tmp_path, monkeypatch):
    entry = Entry("m", "/m.gguf", user_delimiter="-x", position_bias="--", mmproj="-p.gguf")
    args_list, extra = cli_mod.split_passthrough(entry.serve_argv(cache=False, cache_dir=Path("/c")))
    args = cli_mod.build_parser().parse_args(args_list)
    probe = argparse.ArgumentParser()
    probe.add_argument("--x")
    if probe.parse_args(["--x=--"]).x == "--":               # older argparse (3.10) drops a `--` even from `--x=--`
        assert args.position_bias == "--"
    assert (args.user_delimiter, args.mmproj, args.model) == ("-x", "-p.gguf", "/m.gguf")
    assert extra == ["--cache-ram", "0"]                     # a value of `--` is not the llama-server separator
    nowhere = str(tmp_path / "no-such-llama-server")          # defence in depth: never the PATH default
    bad = Entry("m", "/m.gguf", parallel="many", llama_server=nowhere)   # a value argparse cannot read
    with pytest.raises(MenuFailure, match=r"^callo serve did not accept this entry's flags: .*argument --parallel: "
                                          r"invalid int value: 'many'$"):
        with served_model(bad, cache=False, cache_dir=tmp_path / "c", log_path=tmp_path / "m.serve.log",
                          stop=threading.Event()):
            pytest.fail("must not start")
    assert _serve_thread("m") is None
    # in a run: that model fails with argparse's reason, the next one runs
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    good = Entry("b", str(tiny_gguf(tmp_path / "b.gguf")))
    d = Driver(_menu_settings(data), [Entry("a", str(tiny_gguf(tmp_path / "a.gguf")), parallel="many", llama_server=nowhere),
                                      good],
               _opts(tmp_path), log=lambda s: None, scan_fn=lambda **kw: [], host=HOST)
    d.launcher = lambda entry, **kw: (served_model(entry, **kw) if entry.name == "a" else _stub_launcher()(entry, **kw))
    assert d.run() == 1
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert "invalid int value: 'many'" in run["models"]["a"]["reason"] and run["models"]["b"]["status"] == "done"


@pytest.mark.parametrize("how", ["raise", "outcome", "menu_failure"])
def test_a_model_failing_after_a_ctrl_c_is_recorded_as_interrupted(tmp_path, monkeypatch, how):
    """The first Ctrl-C stops the serve under calibrate, which then fails on transport (an exception or a failed
    Outcome); the reason says interrupted, with no traceback in the serve log, and the run exits 130."""
    data = write_items(tmp_path / "d.jsonl", 4)
    log: list[str] = []
    driver = Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path), log=log.append, scan_fn=lambda **kw: [],
                    host=HOST)
    if how == "menu_failure":
        @contextlib.contextmanager
        def launcher(entry, *, cache, cache_dir, log_path, stop, log=None):
            log_path.write_text("stub\n")
            driver.handler(signal.SIGINT, None)                  # the loading child was SIGTERMed: serve failed
            raise MenuFailure("callo serve did not start (exit 1): llama-server exited with -15 before listening")
            yield  # pragma: no cover

        driver.launcher = launcher
    else:
        driver.launcher = _stub_launcher()

    def calibrate_one(settings, *, url, name, out_dir, resume, log, **kw):
        driver.handler(signal.SIGINT, None)
        if how == "raise":
            raise ConnectionResetError(54, "Connection reset by peer")
        return Outcome("failed", "the request failed for good", None)

    monkeypatch.setattr(menu, "calibrate_one", calibrate_one)
    assert driver.run() == 130
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["a"]["status"] == "failed" and run["models"]["a"]["reason"] == "interrupted; --resume continues from the raw file"
    assert run["models"]["b"]["status"] == "pending"
    assert "Traceback" not in (tmp_path / "think" / "a.serve.log").read_text()


def test_settle_and_the_cooldown_are_cut_short_by_should_stop():
    slept, logged = [], []
    stop_after = iter([False, False, False, True])
    settle(seconds=30.0, cooldown=None, log=logged.append, sleep=slept.append, should_stop=lambda: next(stop_after))
    assert slept == [1.0, 1.0] and logged == ["settling 30 s before the next model", "settle cut short: stopping"]
    slept.clear()
    logged.clear()
    stop_after = iter([False, False, False, True])
    settle(seconds=30.0, cooldown="therm", log=logged.append, sleep=slept.append, therm=lambda: 80, clock=lambda: 0.0,
           should_stop=lambda: next(stop_after))
    assert slept == [1.0, 1.0] and logged == ["cooldown therm: CPU_Speed_Limit = 80, waiting", "settle cut short: stopping"]
    slept.clear()
    logged.clear()
    settle(seconds=30.0, cooldown="therm", log=logged.append, sleep=slept.append, therm=lambda: 80, should_stop=lambda: True)
    assert slept == [] and logged == ["settle cut short: stopping"]


def test_a_ctrl_c_during_the_settle_stops_before_the_next_model(tmp_path, monkeypatch):
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    log: list[str] = []
    slept: list[float] = []
    driver = Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path, settle_s=30.0), log=log.append,
                    launcher=_stub_launcher(), scan_fn=lambda **kw: [], host=HOST)

    def sleep(s):
        slept.append(s)
        if len(slept) == 2:
            driver.handler(signal.SIGINT, None)                  # the first Ctrl-C, two seconds into the settle

    driver.sleep = sleep
    assert driver.run() == 130
    assert slept == [1.0, 1.0] and "settle cut short: stopping" in log and list(seen) == ["a"]
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["a"]["status"] == "done" and run["models"]["b"]["status"] == "pending"


def test_a_second_ctrl_c_between_models_returns_130_with_the_report(tmp_path, monkeypatch):
    """A KeyboardInterrupt (StopHandler's second signal) in the settle, outside any model: run() returns 130 after
    the report and the summary, and restores the signal handlers."""
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    log: list[str] = []

    def sleep(s):
        raise KeyboardInterrupt

    driver = Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path, settle_s=30.0), log=log.append,
                    launcher=_stub_launcher(), scan_fn=lambda **kw: [], host=HOST, sleep=sleep)
    assert driver.run() == 130
    run = json.loads((tmp_path / "think" / "run.json").read_text())
    assert run["models"]["a"]["status"] == "done" and run["models"]["b"]["status"] == "pending" and list(seen) == ["a"]
    assert driver.report_md is not None and log[-3:-1] == ["a: done", "b: pending"] and log[-1].startswith("interrupted; ")
    assert signal.getsignal(signal.SIGINT) is not driver.handler


# ---- the CLI: --model / --menu, the shared launch flags, --dry-run with --rates-from, the usage errors -------------

import argparse  # noqa: E402

from callo.cli import build_parser, main  # noqa: E402
from callo.menu import launch_from_args  # noqa: E402


def _calibrate_help() -> str:
    sub = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    text = " ".join(sub.choices["calibrate"].format_help().split())        # argparse wraps help at spaces and hyphens
    return re.sub(r"-\s+", "-", text)


def _quiet_scan(monkeypatch, ps: str = "") -> None:
    """`menu.ps_listing` is the seam: never the box's real process list."""
    monkeypatch.setattr(menu, "ps_listing", lambda run=None: (BENIGN_PS + ps, None))


def test_the_cli_runs_a_menu_and_resumes_it(tmp_path, monkeypatch, capsys, fake_exe):
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = fake_exe()
    monkeypatch.setattr(menu, "ps_listing", _fake_listing(tmp_path, exe))      # our own fakes listed as native servers
    model = tiny_gguf(tmp_path / "m.gguf")
    data = write_items(tmp_path / "d.jsonl", 4)
    doc = {"models": [{"name": "one", "model": str(model)}, {"name": "two", "model": str(model), "args": ["--temp", "0.9"]}]}
    menu_path = write_menu(tmp_path / "menu.json", doc)
    out = tmp_path / "think"
    argv = ["calibrate", "--think-budgets", "0,32", "--menu", str(menu_path), "--data", str(data), "--out", str(out),
            "--warmup", "0", "--audit", "0", "--settle", "0", "--mem-headroom", "0", "--gpu-mem", "64", "--ctx", "4096",
            "--llama-server", str(exe)]
    try:
        rc = main(argv)
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        assert captured.out.startswith("# callo compare") and "| one@0 |" in captured.out and "| two@32 |" in captured.out
        assert "Notes:\n- one: 4 questions (< 100): its intervals are wide\n" in captured.out   # the small-sample note in the report
        assert captured.err.count("WARNING: two: args --temp 0.9 are sampler flags") == 1     # once, by the driver
        assert "one: done" in captured.err and "two: done" in captured.err
        assert f"report: {out / 'report.md'} and {out / 'report.json'}" in captured.err
        run = json.loads((out / "run.json").read_text())
        assert {m["status"] for m in run["models"].values()} == {"done"} and run["order"] == ["one", "two"]
        one = json.loads((out / "one.think.json").read_text())
        assert one["environment"]["server"]["launched_by"] == "menu" and one["environment"]["server"]["n_ctx"] == 4096
        assert (out / "one.serve.log").exists() and (out / "two.serve.log").exists()
        assert main(argv) == 2 and "is not empty. Pass --resume" in capsys.readouterr().err
        assert main([*argv, "--resume"]) == 0
        captured = capsys.readouterr()
        assert "one: done already" in captured.err and captured.out.startswith("# callo compare")
        assert main(["calibrate", "--think-budgets", "0,32", "--model", str(model), "--data", str(data), "--out",
                     str(tmp_path / "t2"), "--warmup", "0", "--audit", "0", "--settle", "0", "--mem-headroom", "0",
                     "--gpu-mem", "64", "--llama-server", str(exe)]) == 0
        assert json.loads((tmp_path / "t2" / "m.think.json").read_text())["environment"]["server"]["launched_by"] == "model"
        assert (tmp_path / "t2" / "run.json").exists() and "| m@32 |" in capsys.readouterr().out
    finally:                                         # cleanup only: an assertion here would mask the body's failure
        left = [p for p in _fake_pids(tmp_path) if alive(p)]
        for pid in left:                                                      # only the fakes this test launched
            if str(exe) in _ps_args(pid):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
    assert not left                                                           # every launched fake was stopped


def test_the_cli_dry_run_for_launch_modes_prints_the_scan_the_precheck_and_the_estimate(tmp_path, monkeypatch, capsys):
    _quiet_scan(monkeypatch)
    model = tiny_gguf(tmp_path / "m.gguf")
    data = write_items(tmp_path / "d.jsonl", 6)
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "m.think.json").write_text(json.dumps({"callo_think_profile": 1, "model_key": load_model_key(model, None),
                                                      "environment": {"run": {"decode_tok_s_median": 50.0}},
                                                      "budgets": {"0": {"lat_p50_ms": [200.0, 190.0, 210.0]}}}))
    argv = ["calibrate", "--think-budgets", "0,64,256", "--data", str(data), "--out", str(tmp_path / "think"), "--dry-run",
            "--model", str(model), "--model", str(tmp_path / "missing.gguf"), "--gpu-mem", "48", "--mem-headroom", "2",
            "--rates-from", str(earlier), "--warmup", "1", "--audit", "3",
            "--llama-server", str(tmp_path / "no-such-llama-server")]           # defence in depth: never the PATH default
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("dry run: 6 labelled text questions from 6 records in d.jsonl")
    assert "GPU rule: no other model server is running\nmemory budget: 48.0 GiB; headroom 2 GiB per model; settle 30 s between models\n" in out
    assert f"m: {model} (0.0 GiB); fits the budget of 48.0 GiB\n  worst case: 6 x 256 = 1536 generated tokens" in out
    assert "  time: ~" in out and f"from {earlier / 'm.think.json'} (decode 50 tok/s, budget-0 readout 0.20 s)" in out
    assert "missing: " in out and "NOT FOUND (the launch would fail" in out and "time: unknown until measured" in out
    assert out.rstrip().endswith("run.json, report.md and report.json; no requests were made and nothing was launched")
    assert not (tmp_path / "think").exists()
    _quiet_scan(monkeypatch, "  101 /opt/llama-server -m x.gguf\n")                 # `ps -axo pid=,args=`
    assert main(argv) == 2
    out = capsys.readouterr().out
    assert "GPU rule: another model server is running (pid 101: /opt/llama-server -m x.gguf): the run would refuse (exit 2)" in out
    assert main([*argv, "--allow-gpu-busy", "--cooldown", "therm", "--settle", "5"]) == 0
    out = capsys.readouterr().out
    assert "would go on under --allow-gpu-busy" in out and "settle 5 s or cooldown therm between models" in out
    # ps itself failing: the rule fails closed in the dry run too, worded as the run's refusal
    monkeypatch.setattr(menu, "ps_listing", lambda run=None: (None, "exit 1: boom"))
    assert main(argv) == 2
    out = capsys.readouterr().out
    assert ("GPU rule: could not list processes (ps failed: exit 1: boom): the run would refuse (exit 2); stop any model "
            "server, or pass --allow-gpu-busy\n") in out                     # the remedy once, composed by the dry run
    # a scan that raises: the same, never a traceback
    monkeypatch.setattr(menu, "scan_findings", lambda **kw: (_ for _ in ()).throw(PermissionError(13, "Permission denied")))
    assert main(argv) == 2
    out = capsys.readouterr().out
    assert ("GPU rule: GPU check failed (PermissionError: [Errno 13] Permission denied): cannot tell whether another model "
            "server is running: the run would refuse (exit 2); stop any model server, or pass --allow-gpu-busy\n") in out
    assert main(["calibrate", "--think-budgets", "--data", str(tmp_path / "none.jsonl"), "--dry-run", "--model", str(model),
                 "--llama-server", str(tmp_path / "no-such-llama-server")]) == 2
    assert "--data" in capsys.readouterr().err


def test_the_cli_usage_errors_for_launch_modes_and_the_shared_help_texts(tmp_path, monkeypatch, capsys):
    _quiet_scan(monkeypatch)
    data = write_items(tmp_path / "d.jsonl", 2)
    model = tiny_gguf(tmp_path / "m.gguf")
    base = ["calibrate", "--think-budgets", "--data", str(data)]
    launch = ["--model", str(model), "--llama-server", str(tmp_path / "no-such-llama-server")]   # never the PATH default
    assert main([*base, *launch, "--menu", "m.json"]) == 2 and "mutually exclusive sources" in capsys.readouterr().err
    assert main([*base, *launch, "--upstream", "http://127.0.0.1:1"]) == 2
    assert "--upstream and the launch sources" in capsys.readouterr().err
    assert main([*base, *launch, "--url", "http://127.0.0.1:1"]) == 2
    assert "--url and the launch sources" in capsys.readouterr().err
    for given_url in (["--url", "http://127.0.0.1:8180"], ["--url=http://127.0.0.1:8180"]):    # explicit, = the default
        assert main([*base, "--menu", "m.json", *given_url]) == 2
        assert "--url and the launch sources" in capsys.readouterr().err
        assert main([*base, "--upstream", "http://127.0.0.1:1", *given_url]) == 2                # one source there too
        assert "--url and --upstream are mutually exclusive" in capsys.readouterr().err
    assert main([*base, *launch, "--no-kvu"]) == 2 and "--no-kvu needs --ctx" in capsys.readouterr().err
    assert main([*base, *launch, "--s1-slots", "4"]) == 2
    assert "--s1-slots must be an integer >= 1 and below --parallel (4)" in capsys.readouterr().err
    assert main([*base, *launch, "--s1-slots", "0"]) == 2 and "--s1-slots must be" in capsys.readouterr().err
    assert main([*base, *launch, "--parallel", "0"]) == 2 and "--parallel must be an integer >= 1" in capsys.readouterr().err
    assert main([*base, *launch, "--ctx", "0"]) == 2 and "--ctx must be a positive integer" in capsys.readouterr().err
    assert main([*base, *launch, "--mem-headroom", "-1"]) == 2 and "--mem-headroom" in capsys.readouterr().err
    assert main([*base, *launch, "--gpu-mem", "0"]) == 2 and "--gpu-mem" in capsys.readouterr().err
    assert main([*base, *launch, "--settle", "-1"]) == 2 and "--settle" in capsys.readouterr().err
    assert main([*base, "--menu", str(tmp_path / "missing.json")]) == 2 and "--menu" in capsys.readouterr().err
    assert main([*base, *launch, "--", "--grammar", "root ::= x"]) == 2
    assert "arguments after --: --grammar" in capsys.readouterr().err
    assert main([*base, *launch, "--raw", "r.jsonl"]) == 2 and "--raw apply only with --bias" in capsys.readouterr().err
    assert main([*base, *launch, "--bias"]) == 2 and "mutually exclusive" in capsys.readouterr().err
    for flag in (["--ctx", "4096"], ["--parallel", "2"], ["--no-kvu"], ["--s1-slots", "1"], ["--llama-server", "x"],
                 ["--user-delimiter", "u"], ["--allow-gpu-busy"], ["--mem-headroom", "1"], ["--gpu-mem", "8"],
                 ["--settle", "0"], ["--cooldown", "therm"], ["--rates-from", "d"]):
        assert main([*base, "--url", "http://127.0.0.1:1", *flag]) == 2
        assert f"{flag[0]} apply only with --model or --menu" in capsys.readouterr().err
        assert main(["calibrate", "--data", str(data), *flag]) == 2
        assert "apply only with --think-budgets" in capsys.readouterr().err
    for flag in (["--model", str(model)], ["--menu", "m.json"]):
        assert main(["calibrate", "--bias", *flag]) == 2 and "apply only with --think-budgets" in capsys.readouterr().err
    text = _calibrate_help()
    assert "--think-budgets: continue the run in --out" in text and "--bias: continue the run recorded in --raw" in text
    assert "--think-budgets: the item order, trace seeds, bootstrap and cross-fit folds" in text
    assert "--bias and --think-budgets: record a failed server-state check" in text
    assert "arguments after `--` are passed to llama-server" in text and "--grammar" in text


def _launch_args(**over) -> argparse.Namespace:
    base = {"ctx": None, "parallel": None, "no_kvu": False, "s1_slots": None, "llama_server": None,
            "user_delimiter": None, "extra": []}
    base.update(over)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("over, why", [
    ({"ctx": 0}, "--ctx must be a positive integer, got 0"),
    ({"ctx": -5}, "--ctx must be a positive integer, got -5"),
    ({"parallel": 0}, "--parallel must be an integer >= 1, got 0"),
    ({"s1_slots": 0}, r"--s1-slots must be an integer >= 1 and below --parallel \(4\) when --parallel > 1"),
    ({"s1_slots": 4}, r"--s1-slots must be an integer >= 1 and below --parallel \(4\)"),
    ({"parallel": 2, "s1_slots": 2}, r"--s1-slots must be an integer >= 1 and below --parallel \(2\)"),
    ({"parallel": 1, "s1_slots": 0}, "--s1-slots must be"),
    ({"no_kvu": True}, "--no-kvu needs --ctx"),
    ({"user_delimiter": ""}, "--user-delimiter must not be empty"),
    ({"llama_server": ""}, "--llama-server must not be empty"),
    ({"extra": ["--json-schema", "{}"]}, "arguments after --: --json-schema"),
])
def test_launch_from_args_applies_callo_serves_rules_to_the_shared_flags(over, why):
    """The Launch built from the shared flags is validated as `callo serve` (and a menu entry) would be."""
    with pytest.raises(InvalidInput, match=why):
        launch_from_args(_launch_args(**over))


def test_launch_from_args_defaults_and_the_rules_that_pass():
    assert launch_from_args(_launch_args()) == Launch()
    assert launch_from_args(_launch_args(parallel=1, s1_slots=1)).parallel == 1      # callo serve allows k = np = 1
    got = launch_from_args(_launch_args(ctx=8192, no_kvu=True, parallel=2, s1_slots=1, llama_server="/opt/ls",
                                        user_delimiter="<u>", extra=["-b", "512"]))
    assert got == Launch(8192, 2, True, 1, "/opt/ls", "<u>", ("-b", "512"))


def _menu_cli(tmp_path, monkeypatch, *extra: str) -> list[str]:
    """The argv of a two-model `--model` run whose launch and calibrate are stubs (no process is started)."""
    monkeypatch.setattr(menu, "served_model", _stub_launcher())
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate({}))
    monkeypatch.setattr(menu, "host_info", lambda: HOST)
    data = write_items(tmp_path / "d.jsonl", 4)
    return ["calibrate", "--think-budgets", "0,32", "--data", str(data), "--out", str(tmp_path / "think"), "--settle", "0",
            "--model", str(tiny_gguf(tmp_path / "a.gguf")), "--model", str(tiny_gguf(tmp_path / "b.gguf")), *extra]


def test_a_second_ctrl_c_during_the_report_write_exits_130(tmp_path, monkeypatch, capsys):
    """The signal handlers are restored before the report; a Ctrl-C there is a KeyboardInterrupt out of
    Driver.run(), which the CLI maps to 130."""
    _quiet_scan(monkeypatch)
    argv = _menu_cli(tmp_path, monkeypatch)

    def write_report(self, rf):
        raise KeyboardInterrupt

    monkeypatch.setattr(Driver, "write_report", write_report)
    assert main(argv) == 130
    err = capsys.readouterr().err
    assert "callo: interrupted; " in err and "`--resume` continues the run" in err
    assert {m["status"] for m in json.loads((tmp_path / "think" / "run.json").read_text())["models"].values()} == {"done"}


def test_a_first_scan_that_raises_is_a_refusal_unless_allow_gpu_busy(tmp_path, monkeypatch, capsys):
    """The FIRST GPU check fails closed: a scan that raises is a refusal (exit 2) before run.json
    exists; under --allow-gpu-busy the run goes on with every profile marked gpu_busy."""
    calls: list[int] = []

    def scan(**kw):
        calls.append(1)
        if len(calls) == 1:
            raise FileNotFoundError(2, "No such file or directory", "llama-server-8080.pid")
        return []

    monkeypatch.setattr(menu, "scan", scan)
    argv = _menu_cli(tmp_path, monkeypatch)
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "refusing to calibrate: GPU check failed (FileNotFoundError: [Errno 2] No such file or directory: " in err
    assert "pass --allow-gpu-busy" in err and not (tmp_path / "think").exists()
    calls.clear()
    assert main([*argv, "--allow-gpu-busy"]) == 0
    err = capsys.readouterr().err
    assert "WARNING: GPU check failed (FileNotFoundError" in err and "--allow-gpu-busy: every profile is marked gpu_busy" in err
    assert json.loads((tmp_path / "think" / "a.think.json").read_text())["environment"]["server"]["gpu_busy"] is True


def test_a_report_that_fails_unexpectedly_is_logged_and_the_exit_is_at_least_1(tmp_path, monkeypatch):
    """Any exception from compare (not only OSError / InvalidInput) is logged, never a traceback out
    of run(); the models ran, the report is missing, so the exit is 1 (130 stays 130)."""
    data = write_items(tmp_path / "d.jsonl", 4)
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate({}))
    monkeypatch.setattr(menu.compare, "write_report", lambda *a, **k: (_ for _ in ()).throw(ZeroDivisionError("bad")))
    log: list[str] = []
    d = Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path), log=log.append, launcher=_stub_launcher(),
               scan_fn=lambda **kw: [], host=HOST)
    assert d.run() == 1
    assert "report failed: ZeroDivisionError: bad" in log and log[-2:] == ["a: done", "b: done"]
    assert {m["status"] for m in json.loads((tmp_path / "think" / "run.json").read_text())["models"].values()} == {"done"}


def test_the_wait_for_the_child_to_exit_is_cut_short_by_a_ctrl_c(tmp_path, monkeypatch):
    """The post-model wait for the llama-server to exit (up to 60 s) stops at the first Ctrl-C, as the
    settle does. The child here is a pid that never exits (is_alive always True); nothing is signalled."""
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    good = _stub_calibrate(seen)
    log: list[str] = []
    slept: list[float] = []
    ticks = iter(range(10**6))                                             # an advancing clock: bounded even unfixed
    driver = Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path), log=log.append,
                    launcher=_stub_launcher(child=DEAD_PID), scan_fn=lambda **kw: [], host=HOST, sleep=slept.append,
                    clock=lambda: float(next(ticks)))
    sent: list = []
    driver.handler.kill = lambda pid, sig: sent.append((pid, sig))
    real_wait = menu.wait_exit
    monkeypatch.setattr(menu, "wait_exit", lambda pid, **kw: real_wait(pid, **{**kw, "is_alive": lambda p: True}))

    def calibrate_one(settings, *, url, name, out_dir, resume, log, **kw):
        driver.handler(signal.SIGINT, None)                                 # the first Ctrl-C while a is measured
        return good(settings, url=url, name=name, out_dir=out_dir, resume=resume, log=log, **kw)

    monkeypatch.setattr(menu, "calibrate_one", calibrate_one)
    assert driver.run() == 130
    assert slept == [] and sent == [] and list(seen) == ["a"]               # not 60 s of 0.2 s polls
    assert f"a: not waiting for llama-server pid {DEAD_PID} to exit: stopping" in log
    assert not any("is still alive after" in ln for ln in log)


def test_a_run_json_write_hitting_enospc_stops_the_run_with_the_failure_in_the_summary(tmp_path, monkeypatch):
    """The real ENOSPC path, `RunFile.write` itself failing (after the in-memory record changed):
    the run stops, the summary shows the model's failure from memory, exit 1."""
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}

    def calibrate_one(settings, *, url, name, out_dir, resume, log, **kw):
        seen[name] = True
        raise ValueError("an unexpected /callo/stats shape")

    monkeypatch.setattr(menu, "calibrate_one", calibrate_one)
    real_write = RunFile.write

    def write(self):
        if any(m["status"] == "failed" for m in self.data["models"].values()):
            raise OSError(28, "No space left on device")
        return real_write(self)

    monkeypatch.setattr(RunFile, "write", write)
    log: list[str] = []
    d = Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path), log=log.append, launcher=_stub_launcher(),
               scan_fn=lambda **kw: [], host=HOST)
    assert d.run() == 1 and list(seen) == ["a"]
    assert any(ln.startswith("cannot write run.json (") and "No space left" in ln for ln in log)
    assert log[-2:] == ["a: failed (ValueError: an unexpected /callo/stats shape)", "b: pending"]
    assert json.loads((tmp_path / "think" / "run.json").read_text())["models"]["a"]["status"] == "running"


# ---- launch numbers, arguments after the separator, the URL predicate ---------------------------------------------

from callo.calibrate_think import url_given  # noqa: E402


@pytest.mark.parametrize("flag, value", [*((f, v) for f in ("--gpu-mem", "--mem-headroom", "--settle")
                                            for v in ("inf", "nan", "-inf", "1e400", "-1")),
                                          ("--gpu-mem", "1e300"), ("--mem-headroom", "1e300")])   # inf in bytes
def test_launch_numbers_must_be_finite_and_in_range(tmp_path, monkeypatch, capsys, flag, value):
    """argparse's float takes inf / nan / 1e400; each is one line and exit 2, never an OverflowError from
    int(GiB * 2**30) or a settle that never ends."""
    _quiet_scan(monkeypatch)
    launched: list[str] = []

    def no_launch(entry, **kw):                     # even on unfixed code nothing is ever launched (no real llama-server)
        launched.append(entry.name)
        raise MenuFailure("launch refused by the test")

    monkeypatch.setattr(menu, "served_model", no_launch)
    monkeypatch.setattr(menu, "settle", lambda **kw: None)
    data = write_items(tmp_path / "d.jsonl", 2)
    argv = ["calibrate", "--think-budgets", "--data", str(data), "--model", str(tiny_gguf(tmp_path / "m.gguf")),
            "--out", str(tmp_path / "think"), "--llama-server", str(tmp_path / "no-such-llama-server"), f"{flag}={value}"]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"callo: {flag} is in ") and "must be a finite" in err and err.count("\n") == 1, err
    assert main([*argv, "--dry-run"]) == 2 and capsys.readouterr().err == err
    assert not (tmp_path / "think").exists() and launched == []


def test_arguments_after_the_separator_need_a_launch_source(tmp_path, capsys):
    """llama-server arguments with no launch source are a usage error, as the other launch-only flags."""
    data = write_items(tmp_path / "d.jsonl", 2)
    assert main(["calibrate", "--think-budgets", "--data", str(data), "--url", "http://127.0.0.1:1", "--", "-ctk", "q8_0"]) == 2
    assert capsys.readouterr().err == "callo: arguments after -- apply only with --model or --menu\n"
    assert main(["calibrate", "--think-budgets", "--data", str(data), "--ctx", "8", "--", "-b", "512"]) == 2
    assert capsys.readouterr().err == "callo: --ctx, arguments after -- apply only with --model or --menu\n"
    assert main(["calibrate", "--data", str(data), "--", "-b", "512"]) == 2
    assert capsys.readouterr().err == "callo: arguments after -- apply only with --think-budgets\n"


def test_one_explicit_url_predicate_for_every_source_check(tmp_path):
    """The parser's record, else a non-default value (a Namespace built elsewhere), in both checks."""
    default = "http://127.0.0.1:8180"
    assert not url_given(argparse.Namespace(url=default)) and url_given(argparse.Namespace(url=default, url_given=True))
    assert url_given(argparse.Namespace(url="http://127.0.0.1:9")) and not url_given(argparse.Namespace())
    ns = argparse.Namespace(model=[str(tmp_path / "m.gguf")], menu=None, url="http://127.0.0.1:9", upstream=None)
    with pytest.raises(InvalidInput, match="--url and the launch sources"):
        menu.entries_from_args(ns)


# ---- serve lines, data and split GGUFs before launch, stop wording, resume paths, the joint report ----------------

import shlex  # noqa: E402

from callo.calibrate_think import serve_command  # noqa: E402


def test_a_menu_profiles_serve_line_round_trips_through_callo_serves_parser_to_the_entrys_values(tmp_path, monkeypatch):
    """The stamp records the GGUF's absolute path, the mmproj's and the launch shape; the serve line repeats them,
    so it parses with callo serve's own parser to the entry's values and names an existing file from any cwd."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.gguf").write_bytes(b"GGUF")
    (models / "mm.gguf").write_bytes(b"GGUF")
    (models / "bias.json").write_text("{}")
    binary = "/opt/llama.cpp build/bin/llama-server"                    # a non-default build, a space in its path
    monkeypatch.chdir(models)                                             # the menu names the files relative to here
    entry = Entry("m", "m.gguf", mmproj="mm.gguf", ctx=8192, parallel=3, no_kvu=True, s1_slots=2, llama_server=binary,
                  user_delimiter="-x", args=("-ctk", "q8_0", "--cache-ram", "8192", "--spec-type", "mtp"),
                  position_bias="bias.json")
    prof = tmp_path / "out dir" / "m.think.json"
    prof.parent.mkdir()
    prof.write_text(json.dumps({"callo_think_profile": 1, "model_name": "m", "environment": {
        "model": {"file": "m.gguf", "bytes": 4}, "server": {"server_args": ["-m", "m.gguf"]}}}))
    p = stamp_profile(prof, launched_by="menu", gpu_busy=False, cache_ram=8192, entry=entry)
    assert p["environment"]["model"] == {"file": "m.gguf", "bytes": 4, "path": str(models / "m.gguf"),
                                         "mmproj_path": str(models / "mm.gguf")}
    assert p["environment"]["server"]["launch"] == {"ctx": 8192, "parallel": 3, "no_kvu": True, "s1_slots": 2,
                                                    "user_delimiter": "-x", "args": list(entry.args),
                                                    "position_bias": str(models / "bias.json"), "llama_server": binary}
    line = serve_command(json.loads(prof.read_text()), 256, prof)
    monkeypatch.chdir(tmp_path)                                           # from anywhere
    argv = shlex.split(line)
    assert argv[:2] == ["callo", "serve"]
    args_list, extra = cli_mod.split_passthrough(argv[1:])
    args = build_parser().parse_args(args_list)
    assert Path(args.model).is_file() and Path(args.model).is_absolute() and args.model == str(models / "m.gguf")
    assert (args.ctx, args.parallel, args.no_kvu, args.s1_slots) == (8192, 3, True, 2)
    assert args.user_delimiter == "-x" and args.mmproj == str(models / "mm.gguf") and extra == list(entry.args)
    assert args.think_profile == str(prof) and args.think_budget == "256" and "--cache-ram 0" not in line
    assert args.position_bias == str(models / "bias.json") and args.llama_server == binary
    # the defaults an entry leaves alone still round-trip: no --ctx, no --no-kvu, no mmproj, no args, no position
    # bias, and the default binary (found on PATH as it was when measured) is not repeated
    plain = Entry("p", str(models / "m.gguf"))
    prof.write_text(json.dumps({"callo_think_profile": 1, "environment": {"server": {"server_args": []}}}))
    stamp_profile(prof, launched_by="model", gpu_busy=False, entry=plain)
    line = serve_command(json.loads(prof.read_text()), 0, "w.think.json")
    assert line == f"callo serve --model {models / 'm.gguf'} --parallel 4 --s1-slots 1 --think-profile w.think.json --think-budget 0"
    args = build_parser().parse_args(shlex.split(line)[1:])
    assert (args.ctx, args.parallel, args.no_kvu, args.s1_slots, args.mmproj, args.user_delimiter) == (None, 4, False, 1, None, None)
    assert args.position_bias is None and args.llama_server == "llama-server"


def _no_launch(calls: list):
    @contextlib.contextmanager
    def launcher(entry, **kw):
        calls.append(entry.name)
        raise AssertionError("no model may be launched")
        yield  # pragma: no cover
    return launcher


@pytest.mark.parametrize("content, why", [
    ('{"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}, '
     '"label": "a"}}}\nnot json\n', "not valid JSON"),
    ('{"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}}}}\n',
     "no labelled text questions")])
def test_unusable_data_refuses_before_the_gpu_check_and_any_launch(tmp_path, monkeypatch, capsys, content, why):
    """`--data` is read once before the GPU check and the first model: a bad line or no labelled question is
    exit 2 with one line; nothing is scanned, launched or written."""
    launched: list[str] = []
    scanned: list[int] = []
    monkeypatch.setattr(menu, "served_model", _no_launch(launched))
    monkeypatch.setattr(menu, "scan", lambda **kw: scanned.append(1) or [])
    monkeypatch.setattr(menu, "host_info", lambda: HOST)
    data = tmp_path / "bad.jsonl"
    data.write_text(content)
    rc = main(["calibrate", "--think-budgets", "0,32", "--data", str(data), "--out", str(tmp_path / "think"),
               "--settle", "0", "--model", str(tiny_gguf(tmp_path / "a.gguf")), "--model", str(tiny_gguf(tmp_path / "b.gguf"))])
    err = capsys.readouterr().err
    assert rc == 2 and launched == [] and scanned == [] and not (tmp_path / "think").exists()
    assert err.count("\n") == 1 and err.startswith("callo: ") and why in err, err


def _shards(tmp_path, n: int = 3, mib: int = 1) -> list[Path]:
    out = []
    for i in range(1, n + 1):
        p = tmp_path / f"big-Q4_K_M-{i:05d}-of-{n:05d}.gguf"
        p.write_bytes(b"\0" * (mib * 2**20))
        out.append(p)
    return out


def test_a_split_gguf_counts_every_shard_in_the_precheck_and_the_dry_run_and_a_missing_shard_fails_it(tmp_path):
    """The precheck and the dry run size a split GGUF as the sum of its shards (as `callo serve` reports it); a
    missing shard is the model's precheck failure, naming the shard, and the dry run's `would fail`."""
    first, second, _ = _shards(tmp_path)
    entry = Entry("big", str(first))
    budget = int(2.5 * 2**20)                                             # above one shard, below three
    assert menu.model_bytes(str(first)) == 3 * 2**20
    assert precheck(entry, budget, 0.0) == "needs ~0.0 GiB, budget 0.0 GiB (--gpu-mem to override)"
    assert precheck(entry, 4 * 2**20, 0.0) is None
    settings = _settings(tmp_path)
    head = dry_run_model_lines(entry, settings, 6, budget=budget, headroom_gib=0.0, rates_dirs=[], cache_dir=None)[0]
    assert head.startswith(f"big: {first} (0.0 GiB); would be skipped: needs ~0.0 GiB"), head
    second.unlink()
    with pytest.raises(InvalidInput, match=second.name):
        precheck(entry, budget, 0.0)
    head = dry_run_model_lines(entry, settings, 6, budget=budget, headroom_gib=0.0, rates_dirs=[], cache_dir=None)[0]
    assert head.startswith("big: would fail at the precheck: ") and second.name in head
    data = write_items(tmp_path / "d.jsonl", 4)
    launched: list[str] = []
    log: list[str] = []
    d = Driver(_menu_settings(data), [entry], _opts(tmp_path), log=log.append, launcher=_no_launch(launched),
               scan_fn=lambda **kw: [], host=HOST)
    assert d.run() == 1 and launched == []
    rec = json.loads((tmp_path / "think" / "run.json").read_text())["models"]["big"]
    assert rec["status"] == "failed" and rec["reason"].startswith("precheck: ") and second.name in rec["reason"]


def test_the_stop_wording_says_the_current_model_is_interrupted_now():
    """The first signal interrupts the current model at once (recorded failed: interrupted) and the run stops."""
    assert StopHandler.STOPPING == ("stopping now: the current model is interrupted (--resume continues it) and no "
                                    "further model runs")
    root = Path(__file__).resolve().parents[1]
    guide = (root / "docs" / "guide" / "calibration.md").read_text()      # the menu reference moved out of the README
    readme = (root / "README.md").read_text()
    assert "stops after the current model" not in guide + readme and "interrupts the current model at once" in guide


def test_the_registered_child_is_cleared_before_the_post_scan(tmp_path, monkeypatch):
    """Once the serve has stopped (and reaped its child), a second signal during the post-scan must not SIGKILL
    that pid: `StopHandler.child` is already None when `_finish` scans."""
    data = write_items(tmp_path / "d.jsonl", 4)
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate({}))
    seen: list = []
    d: Driver

    def scan_fn(**kw):
        seen.append(d.handler.child)
        return []

    d = Driver(_menu_settings(data), [Entry("a", str(tiny_gguf(tmp_path / "a.gguf")))], _opts(tmp_path),
               log=lambda s: None, launcher=_stub_launcher(child=4242424), scan_fn=scan_fn, host=HOST)
    assert d.run() == 0
    assert seen == [None, None]                                           # the first check, then the post-scan


def test_run_json_keeps_profiles_absolute_and_a_resume_from_another_cwd_skips_the_done_models(tmp_path, monkeypatch):
    """`run.json` stores each profile absolute, so `--resume` from another cwd (a relative --out) finds the done
    models; an older run.json's relative path is read as its file name in DIR."""
    data = write_items(tmp_path / "d.jsonl", 4)
    seen: dict = {}
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate(seen))
    a, b = _two(tmp_path)
    kw = {"launcher": _stub_launcher(), "scan_fn": lambda **k: [], "host": HOST}
    one, two = tmp_path / "cwd1", tmp_path / "cwd2"
    one.mkdir()
    two.mkdir()
    monkeypatch.chdir(one)
    assert Driver(_menu_settings(data), [a, b], MenuOptions(Path("think"), settle_s=0.0), log=lambda s: None, **kw).run() == 0
    run = json.loads((one / "think" / "run.json").read_text())
    assert run["models"]["a"]["profile"] == str(one / "think" / "a.think.json")
    monkeypatch.chdir(two)
    seen.clear()
    log: list[str] = []
    opts = MenuOptions(Path("../cwd1/think"), resume=True, settle_s=0.0)
    assert Driver(_menu_settings(data), [a, b], opts, log=log.append, **kw).run() == 0
    assert seen == {} and "a: done already" in "\n".join(log)              # nothing relaunched
    run["models"]["a"]["profile"] = "think/a.think.json"                  # an older run.json: relative to its cwd
    run["models"]["b"]["profile"] = "b.think.json"
    (one / "think" / "run.json").write_text(json.dumps(run))
    assert Driver(_menu_settings(data), [a, b], opts, log=log.append, **kw).run() == 0
    assert seen == {}
    report = json.loads((one / "think" / "report.json").read_text())
    assert [r["profile"] for r in report["runs"]] == [str(one / "think" / "a.think.json"), str(one / "think" / "b.think.json")]


def test_a_refused_joint_report_writes_the_stub_naming_why_and_exits_1(tmp_path, monkeypatch):
    """`compare` refusing the joint report (or a done profile that does not load) is logged, the stub report names
    the reason first (no stale report.md survives), and the run exits 1."""
    data = write_items(tmp_path / "d.jsonl", 4)
    monkeypatch.setattr(menu, "calibrate_one", _stub_calibrate({}))
    out = tmp_path / "think"
    kw = {"launcher": _stub_launcher(), "scan_fn": lambda **k: [], "host": HOST}
    assert Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path), log=lambda s: None, **kw).run() == 0
    assert "No joint report" not in (out / "report.md").read_text()
    why = "profiles differ in metrics version: a 2, b 3"
    monkeypatch.setattr(menu.compare, "write_report", lambda *a, **k: (_ for _ in ()).throw(InvalidInput(why)))
    log: list[str] = []
    d = Driver(_menu_settings(data), _two(tmp_path), _opts(tmp_path, resume=True), log=log.append, **kw)
    assert d.run() == 1
    md = (out / "report.md").read_text()
    assert md.splitlines()[2] == f"No joint report: {why}" and "## Runs" in md and d.report_md == md
    assert json.loads((out / "report.json").read_text())["error"] == why
    assert any(ln.startswith(f"no joint report: {why}; ") for ln in log)


@pytest.mark.parametrize("value", ["86400.5", "1e300", "-1"])
def test_settle_is_capped_at_a_day(tmp_path, monkeypatch, capsys, value):
    """`--settle` is finite, at least 0 and at most 86400 s (exit 2 otherwise)."""
    argv = _menu_cli(tmp_path, monkeypatch)
    argv[argv.index("--settle") + 1] = value
    assert main(argv) == 2
    assert "--settle is in seconds and must be a finite number from 0 to 86400" in capsys.readouterr().err


def test_bad_data_is_exit_2_in_the_launch_modes_dry_run_and_exit_1_in_the_attach_modes(tmp_path, monkeypatch, capsys):
    """The launch-mode run refuses unusable --data (exit 2), so its dry run says the same; the one-server attach
    modes keep exit 1 for bad data, as documented. Nothing is scanned or launched either way."""
    monkeypatch.setattr(menu, "scan_findings", lambda **kw: pytest.fail("the dry run must not scan on bad data"))
    data = tmp_path / "bad.jsonl"
    data.write_text("not json\n")
    base = ["calibrate", "--think-budgets", "0,32", "--data", str(data), "--dry-run"]
    assert main([*base, "--model", str(tiny_gguf(tmp_path / "a.gguf"))]) == 2
    assert capsys.readouterr().err.startswith("callo: --data line 1: not valid JSON")
    assert main([*base, "--menu", str(write_menu(tmp_path / "m.json", {"models": [{"name": "a", "model": "/v/a.gguf"}]}))]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    assert main(base) == 1                                                # --url (the default): the attach mode's exit 1
    assert "not valid JSON" in capsys.readouterr().err
