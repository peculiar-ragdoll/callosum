"""Launch mode against a fake `llama-server` executable: argv order, --port 0 log parsing, process-group stop,
orphan pid files, child death. (v0.3: FAKE_AUDIO_ONLY makes its --mmproj an audio-only projector.)"""
import base64
import gc
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import warnings

import pytest
from helpers import STARTUP_TIMEOUT

from callo.errors import CalloError
from callo.upstream import (
    LAUNCH_FLAGS,
    LlamaServerProcess,
    build_argv,
    check_passthrough_args,
    find_orphans,
    parse_listening,
)
from callo.vision import CONTROL_PNG

FAKE = '''#!{python}
"""A fake llama-server: records argv, binds --port (0 = any), logs `listening on`, serves /health /props /slots.
Imports are kept minimal (and deferred where possible) so the interpreter reaches `listening on` as fast as it can,
even on a heavily contended CI runner; it never imports the callo project."""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

argv = sys.argv[1:]
with open(os.environ["FAKE_ARGV_FILE"], "w") as f:
    json.dump(argv, f)
AUDIO = "--mmproj" in argv and bool(os.environ.get("FAKE_AUDIO_ONLY"))     # an audio-only projector
VISION = "--mmproj" in argv and not os.environ.get("FAKE_NO_VISION") and not AUDIO
MEDIA = ("image_url", "input_audio", "input_video", "video_url")


def text(content):
    if isinstance(content, str):
        return content
    return "".join("<__media_fake__>" if p.get("type") in MEDIA else p.get("text", "") for p in content)
mode = os.environ.get("FAKE_MODE", "serve")
if mode == "die":
    sys.exit(7)
port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8080
if "--slot-save-path" in argv and not os.path.isdir(argv[argv.index("--slot-save-path") + 1]):
    print("error: not a directory: " + argv[argv.index("--slot-save-path") + 1], file=sys.stderr, flush=True)
    sys.exit(1)                                       # common/arg.cpp: --slot-save-path must be an existing directory
if os.environ.get("FAKE_CHILD_PID_FILE"):
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(os.environ["FAKE_CHILD_PID_FILE"], "w") as f:
        f.write(str(child.pid))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            body = {"status": "ok"}
        elif self.path == "/props":
            body = {"total_slots": 4, "model_path": os.environ.get("FAKE_MODEL", "/x.gguf"),
                    "default_generation_settings": {"n_ctx": 4096}, "chat_template": "t", "endpoint_slots": True,
                    "build_info": "b1-926862e", "modalities": {"vision": VISION, "video": False, "audio": AUDIO},
                    "media_marker": "<__media_fake__>"}
        else:
            body = [{"id": i, "is_processing": False} for i in range(4)]
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        code = 200
        if self.path.startswith("/slots/"):          # launch mode passes --slot-save-path: erase is supported
            out = {"id_slot": int(self.path.split("?")[0].rsplit("/", 1)[1]), "n_erased": 0}
            code = int(os.environ.get("FAKE_ERASE_STATUS", "200"))     # 501: the flag was overridden; 500: it failed
            if code != 200:
                out = {"error": {"code": code, "message": "erase refused"}}
        elif self.path == "/tokenize":
            out = {"tokens": list(range(100, 100 + len(str(body.get("content", "")).split())))}
        elif self.path == "/apply-template":
            msgs = body["messages"]
            out = {"prompt": "".join(f"<|im_start|>{m['role']}\\n{text(m['content'])}<|im_end|>\\n" for m in msgs)
                   + "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"}
        else:                                    # the mirrored self-test control image answers A ("left")
            mm = body["prompt"].get("multimodal_data", []) if isinstance(body.get("prompt"), dict) else []
            b, a = (-2.5, -0.2) if "{control_b64}" in mm else (-0.2, -2.5)
            out = {"content": "B", "completion_probabilities": [{"top_logprobs": [
                {"id": 1, "token": "B", "logprob": b}, {"id": 0, "token": "A", "logprob": a}]}],
                   "timings": {"prompt_n": 3, "cache_n": 0}}
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


srv = HTTPServer(("127.0.0.1", port), H)
time.sleep(float(os.environ.get("FAKE_STARTUP_DELAY", "0")))
print("srv    load_model: loading model", file=sys.stderr, flush=True)
if mode == "split":
    pass
elif os.environ.get("FAKE_JSONL"):
    print(json.dumps({"msg": f"srv  listening on http://127.0.0.1:{srv.server_address[1]}"}), file=sys.stderr, flush=True)
else:
    print(f"srv  listening on http://127.0.0.1:{srv.server_address[1]}", file=sys.stderr, flush=True)
if mode == "exit-after-listen":                                      # dies between "listening" and /health
    os._exit(5)
if mode == "split":                                                  # a log line arrives in two writes
    pending = f"srv  listening on http://127.0.0.1:{srv.server_address[1]}"
    sys.stderr.write(pending[:-2]); sys.stderr.flush(); time.sleep(0.8); sys.stderr.write(pending[-2:] + "\\n")
    sys.stderr.flush()
if mode == "die-later":
    import threading
    threading.Timer(float(os.environ.get("FAKE_DIE_AFTER", "1.5")), lambda: os._exit(9)).start()
srv.serve_forever()
'''.replace("{control_b64}", base64.b64encode(CONTROL_PNG).decode())


@pytest.fixture
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_text(FAKE.replace("{python}", sys.executable))
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_ARGV_FILE", str(tmp_path / "argv.json"))
    monkeypatch.delenv("FAKE_MODE", raising=False)
    monkeypatch.delenv("FAKE_CHILD_PID_FILE", raising=False)
    monkeypatch.delenv("FAKE_JSONL", raising=False)
    monkeypatch.delenv("FAKE_DIE_AFTER", raising=False)
    return exe


def _argv(tmp_path):
    return json.loads((tmp_path / "argv.json").read_text())


def test_build_argv_has_the_exact_spec_order():
    argv = build_argv("/bin/llama-server", "/m/x.gguf", ctx=32768, parallel=4, extra=["--no-mmap", "-np", "8"])
    assert argv == ["/bin/llama-server", "-m", "/m/x.gguf", "--port", "0", "-c", "32768", "-np", "4",
                    "-kvu", "--no-cache-idle-slots", "-b", "512", "-ub", "512", "--jinja", "--no-mmap", "-np", "8"]
    assert LAUNCH_FLAGS == ["-kvu", "--no-cache-idle-slots", "-b", "512", "-ub", "512", "--jinja"]
    no_ctx = build_argv("ls", "m.gguf", ctx=None, parallel=4, extra=[])
    assert "-c" not in no_ctx                           # --ctx defaults to llama-server's default, never assumed
    assert "--mmproj" not in no_ctx


def test_build_argv_puts_mmproj_after_the_fixed_flags_and_before_the_passthrough():
    argv = build_argv("/bin/llama-server", "/m/x.gguf", ctx=32768, parallel=4, extra=["--no-mmap"], mmproj="/m/mm.gguf")
    assert argv == ["/bin/llama-server", "-m", "/m/x.gguf", "--port", "0", "-c", "32768", "-np", "4",
                    "-kvu", "--no-cache-idle-slots", "-b", "512", "-ub", "512", "--jinja", "--mmproj", "/m/mm.gguf",
                    "--no-mmap"]


def test_build_argv_without_unified_kv_drops_kvu_and_scales_c_by_parallel_in_the_caller():
    argv = build_argv("/bin/llama-server", "/m/x.gguf", ctx=131072, parallel=4, extra=["--no-mmap"],
                      mmproj="/m/mm.gguf", unified_kv=False)
    assert argv == ["/bin/llama-server", "-m", "/m/x.gguf", "--port", "0", "-c", "131072", "-np", "4",
                    "--no-cache-idle-slots", "-b", "512", "-ub", "512", "--jinja", "--mmproj", "/m/mm.gguf",
                    "--no-mmap"]
    assert "-kvu" not in argv and LAUNCH_FLAGS[0] == "-kvu"       # LAUNCH_FLAGS itself is unchanged
    assert build_argv("ls", "m", ctx=None, parallel=2, extra=[], unified_kv=False).count("-c") == 0
    assert "-kvu" in build_argv("ls", "m", ctx=8192, parallel=2, extra=[])            # default stays unified
    assert "-kvu" in build_argv("ls", "m", ctx=8192, parallel=2, extra=[], unified_kv=True)


def test_check_passthrough_args_rejects_multi_host():
    with pytest.raises(CalloError, match="--host"):
        check_passthrough_args(["--host", "127.0.0.1,::1"])
    with pytest.raises(CalloError, match="--host"):
        check_passthrough_args(["--host=127.0.0.1,10.0.0.1"])
    check_passthrough_args(["--host", "127.0.0.1", "-np", "8"])


def test_parse_listening_matches_plain_and_jsonl_lines():
    assert parse_listening("srv  listening on http://127.0.0.1:54321\n") == 54321
    assert parse_listening('{"msg":"srv  listening on http://127.0.0.1:8199"}') == 8199
    assert parse_listening("srv    load_model: loading model '/x.gguf'") is None
    assert parse_listening("listening on unix:///tmp/x.sock") is None


def test_start_parses_the_bound_port_and_writes_pid_and_log_files(fake, tmp_path):
    argv = build_argv(str(fake), "/x.gguf", ctx=None, parallel=4, extra=[])
    p = LlamaServerProcess(argv, cache_dir=tmp_path / "cache", startup_timeout=STARTUP_TIMEOUT)
    try:
        port = p.start()
        assert port > 0 and p.port == port and p.alive()
        assert _argv(tmp_path)[:4] == ["-m", "/x.gguf", "--port", "0"]
        assert p.pid_file == tmp_path / "cache" / f"llama-server-{port}.pid" and p.pid_file.read_text().strip() == str(p.proc.pid)
        assert p.log_file == tmp_path / "cache" / f"llama-server-{port}.log" and "listening on" in p.log_file.read_text()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            assert json.loads(r.read())["status"] == "ok"
    finally:
        p.stop()
    assert not p.alive() and not p.pid_file.exists()


def test_start_parses_jsonl_logs(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_JSONL", "1")
    p = LlamaServerProcess(build_argv(str(fake), "/x.gguf", ctx=None, parallel=2, extra=["--log-jsonl"]),
                           cache_dir=tmp_path / "cache", startup_timeout=STARTUP_TIMEOUT)
    try:
        assert p.start() > 0
    finally:
        p.stop()


def test_stop_kills_the_whole_process_group(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(tmp_path / "child.pid"))
    p = LlamaServerProcess(build_argv(str(fake), "/x.gguf", ctx=None, parallel=4, extra=[]),
                           cache_dir=tmp_path / "cache", startup_timeout=STARTUP_TIMEOUT)
    p.start()
    child = int((tmp_path / "child.pid").read_text())
    assert os.getpgid(child) == os.getpgid(p.proc.pid) != os.getpgid(os.getpid())   # start_new_session=True
    p.stop()
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def test_child_that_dies_before_listening_raises_with_the_log_tail(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "die")
    p = LlamaServerProcess(build_argv(str(fake), "/x.gguf", ctx=None, parallel=4, extra=[]),
                           cache_dir=tmp_path / "cache", startup_timeout=STARTUP_TIMEOUT)
    with pytest.raises(CalloError, match="exited with 7"):
        p.start()
    assert not list((tmp_path / "cache").glob("*.pid"))


def test_watch_reports_child_death(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "die-later")
    p = LlamaServerProcess(build_argv(str(fake), "/x.gguf", ctx=None, parallel=4, extra=[]),
                           cache_dir=tmp_path / "cache", startup_timeout=STARTUP_TIMEOUT)
    p.start()
    seen = []
    t = p.watch(seen.append)
    t.join(timeout=STARTUP_TIMEOUT)
    assert seen == [9] and p.returncode == 9 and not p.alive()
    p.stop()                                                    # idempotent on a dead child
    p.stop()


def test_stop_still_signals_the_group_when_the_leader_already_exited(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(tmp_path / "child.pid"))
    monkeypatch.setenv("FAKE_MODE", "die-later")
    p = LlamaServerProcess(build_argv(str(fake), "/x.gguf", ctx=None, parallel=4, extra=[]),
                           cache_dir=tmp_path / "cache", startup_timeout=STARTUP_TIMEOUT)
    p.start()
    child = int((tmp_path / "child.pid").read_text())
    p.proc.wait(timeout=STARTUP_TIMEOUT)                                     # the leader is gone; its sleeping child is not
    os.kill(child, 0)
    p.stop()
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def test_a_deliberate_stop_is_not_reported_as_a_death(fake, tmp_path):
    p = LlamaServerProcess(build_argv(str(fake), "/x.gguf", ctx=None, parallel=4, extra=[]),
                           cache_dir=tmp_path / "cache", startup_timeout=STARTUP_TIMEOUT)
    p.start()
    seen = []
    t = p.watch(seen.append)
    p.stop()
    t.join(timeout=STARTUP_TIMEOUT)
    assert not t.is_alive() and seen == [] and not p.alive()


def test_pid_file_exists_while_the_model_loads_and_is_renamed_after_bind(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_STARTUP_DELAY", "1.5")
    cache = tmp_path / "cache"
    starting = cache / f"llama-server-starting-{os.getpid()}.pid"
    p = LlamaServerProcess(build_argv(str(fake), "/x.gguf", ctx=None, parallel=4, extra=[]),
                           cache_dir=cache, startup_timeout=STARTUP_TIMEOUT)
    result = []
    th = threading.Thread(target=lambda: result.append(p.start()))
    th.start()
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while not starting.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert starting.exists() and th.is_alive()              # loading: no port yet, but already discoverable
        child = int(starting.read_text())
        assert find_orphans(cache) == [(child, None)] and starting.exists()
        th.join(timeout=STARTUP_TIMEOUT)
        port = result[0]
        assert not starting.exists() and p.pid_file == cache / f"llama-server-{port}.pid"
        assert p.pid_file.read_text().strip() == str(child) == str(p.proc.pid)
    finally:
        th.join(timeout=STARTUP_TIMEOUT)
        p.stop()
    assert not list(cache.glob("*.pid"))


def test_stop_during_load_removes_the_starting_pid_file(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_STARTUP_DELAY", "30")
    cache = tmp_path / "cache"
    p = LlamaServerProcess(build_argv(str(fake), "/x.gguf", ctx=None, parallel=4, extra=[]),
                           cache_dir=cache, startup_timeout=1)
    with pytest.raises(CalloError, match="did not print"):
        p.start()
    assert not p.alive() and not list(cache.glob("*.pid"))


def test_start_ignores_a_listening_line_that_is_still_being_written(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "split")
    p = LlamaServerProcess(build_argv(str(fake), "/x.gguf", ctx=None, parallel=4, extra=[]),
                           cache_dir=tmp_path / "cache", startup_timeout=STARTUP_TIMEOUT)
    try:
        port = p.start()
        assert port > 10000                                     # not the truncated `...:543` of `...:54321`
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            assert json.loads(r.read())["status"] == "ok"
    finally:
        p.stop()


def test_find_orphans_reports_live_pids_and_removes_stale_files(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (cache / "llama-server-9999.pid").write_text(str(sleeper.pid))
        (cache / "llama-server-8888.pid").write_text(str(2**22 + 12345))          # no such pid
        (cache / "llama-server-7777.pid").write_text("not a pid")
        assert find_orphans(cache) == [(sleeper.pid, 9999)]
        assert sorted(f.name for f in cache.glob("*.pid")) == ["llama-server-9999.pid"]
    finally:
        sleeper.send_signal(signal.SIGTERM)
        sleeper.wait(timeout=5)
    assert find_orphans(cache) == [] and not list(cache.glob("*.pid"))
    assert find_orphans(tmp_path / "missing") == []


def test_find_orphans_reports_a_starting_file_with_port_none(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (cache / "llama-server-starting-4242.pid").write_text(f"{sleeper.pid}\n")    # 4242 is callo's pid, not a port
        (cache / "llama-server-starting-4243.pid").write_text(str(2**22 + 12345))     # stale
        assert find_orphans(cache) == [(sleeper.pid, None)]
        assert [f.name for f in cache.glob("*.pid")] == ["llama-server-starting-4242.pid"]
    finally:
        sleeper.send_signal(signal.SIGTERM)
        sleeper.wait(timeout=5)


# ---- v0.3: --slot-save-path ---------------------------------------------------------------------------

def test_build_argv_puts_slot_save_path_after_the_fixed_flags_and_before_mmproj():
    argv = build_argv("/bin/llama-server", "/m/x.gguf", ctx=32768, parallel=4, extra=["--no-mmap"],
                      mmproj="/m/mm.gguf", slot_save_path="/c/slots")
    assert argv == ["/bin/llama-server", "-m", "/m/x.gguf", "--port", "0", "-c", "32768", "-np", "4",
                    "-kvu", "--no-cache-idle-slots", "-b", "512", "-ub", "512", "--jinja",
                    "--slot-save-path", "/c/slots", "--mmproj", "/m/mm.gguf", "--no-mmap"]
    assert "--slot-save-path" not in build_argv("ls", "m", ctx=None, parallel=2, extra=[])     # v0.2 argv kept


def test_the_fake_refuses_a_slot_save_path_that_is_not_a_directory(fake, tmp_path):
    p = LlamaServerProcess([str(fake), "-m", "x", "--port", "0", "--slot-save-path", str(tmp_path / "nope")],
                           cache_dir=tmp_path / "c", startup_timeout=20)
    with pytest.raises(CalloError, match="not a directory"):
        p.start()


def test_a_binary_that_cannot_start_leaves_no_open_log_file(tmp_path):
    """Popen raising (a missing llama-server) must not leak the child's log file handle."""
    p = LlamaServerProcess([str(tmp_path / "missing-llama-server"), "-m", "/x.gguf"], cache_dir=tmp_path / "cache")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ResourceWarning)
        with pytest.raises(OSError):
            p.start()
        gc.collect()
    assert [str(w.message) for w in seen if issubclass(w.category, ResourceWarning)] == []
