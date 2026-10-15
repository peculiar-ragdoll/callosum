"""A fake llama-server EXECUTABLE for launch-mode tests of the think-budget menu: the thinking fake upstream
(`fake_upstream.FakeUpstream(think=...)`) bound to the `--port` llama-server would be given, printing llama-server's
`listening on` line. `tests/test_launch.py::FAKE` covers the launch plumbing (argv, pid files, log parsing); this one
answers callo's thinking-mode traffic (the think probe, the self-tests, the slot erase), so a whole
`callo calibrate --think-budgets --menu` run can execute without a GPU.

A test writes `exe_script(sys.executable, tests_dir)` to a file, makes it executable and names it in a menu entry's
`llama_server`. Environment: FAKE_ARGV_FILE (records argv as JSON), FAKE_THINK (the template: preopened by default),
FAKE_FAIL (print an error and exit 7 before listening: a planted load failure). A menu entry's `args` may carry
`--fake-think TEMPLATE` instead of the environment variable (one process per model, several templates per run).
FAKE_IGNORE_TERM makes the fake ignore SIGTERM (a llama-server that does not stop on the first signal), so the menu's
escalation to SIGKILL can be tested (it is ignored from the start, the loading phase included). FAKE_LOAD_DELAY
(seconds) sleeps before listening: a llama-server still loading its model, for the loading-phase SIGTERM, the start
timeout and the cleanup's own SIGTERM/SIGKILL on a real (fake) process. FAKE_PIDS_FILE: each fake appends its pid there as it starts, so a test can reap
exactly the fakes it launched (and nothing else) even when it fails."""
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path


def natural_len(text: str) -> int:
    """The planted natural trace length: questions 1, 3, 10 and 11 (and 1x, 3x, ...) close at 40 tokens, the others at
    300, so a 64-token budget cuts some traces and not others."""
    return 40 if any(f"state {i}" in text for i in (1, 3, 10, 11)) else 300


def exe_script(python: str, tests_dir: str) -> str:
    return (f"#!{python}\nimport sys\nsys.path.insert(0, {tests_dir!r})\nimport fake_llama_think\n"
            f"sys.exit(fake_llama_think.main(sys.argv[1:]))\n")


def main(argv: list[str]) -> int:
    if os.environ.get("FAKE_PIDS_FILE"):
        with open(os.environ["FAKE_PIDS_FILE"], "a", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n")
    if os.environ.get("FAKE_ARGV_FILE"):
        Path(os.environ["FAKE_ARGV_FILE"]).write_text(json.dumps(argv))
    if os.environ.get("FAKE_FAIL"):
        print("srv    load_model: loading model", file=sys.stderr, flush=True)
        print("error: planted load failure (FAKE_FAIL)", file=sys.stderr, flush=True)
        return 7
    if os.environ.get("FAKE_IGNORE_TERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if os.environ.get("FAKE_LOAD_DELAY"):
        print("srv    load_model: loading model", file=sys.stderr, flush=True)
        time.sleep(float(os.environ["FAKE_LOAD_DELAY"]))
    from fake_upstream import FakeUpstream  # noqa: PLC0415 -- after sys.path is set by the script

    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 0
    model = argv[argv.index("-m") + 1] if "-m" in argv else "/x.gguf"
    think = argv[argv.index("--fake-think") + 1] if "--fake-think" in argv else os.environ.get("FAKE_THINK", "preopened")
    up = FakeUpstream(think=think, think_len=natural_len, slot_save=True, port=port)
    up.props["model_path"] = model
    if "-c" in argv:
        up.props["default_generation_settings"]["n_ctx"] = int(argv[argv.index("-c") + 1])
    up.start()
    print(f"srv  listening on http://127.0.0.1:{up.port}", file=sys.stderr, flush=True)
    threading.Event().wait()                                     # until SIGTERM (the default action: exit)
    return 0
