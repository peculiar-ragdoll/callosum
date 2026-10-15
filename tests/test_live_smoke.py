"""Live smoke (the 'b9890 compatibility' smoke + the self-test row), skipped unless `-m live`.

Master is llama.cpp 4ebdf2c (`0.5.0-dev (build 50, commit 4ebdf2c)`); Homebrew is b9890."""
import json
import os
import threading
import urllib.request
from pathlib import Path

import pytest
from helpers import bench_path

from callo.cli import build_parser, serve, split_passthrough

MASTER = Path(bench_path("CALLO_BENCH_LLAMA_SERVER"))                 # llama.cpp master @ 4ebdf2c
BREW = Path("/opt/homebrew/bin/llama-server")          # b9890; used only for the b9890-compatibility case
MODEL = Path(bench_path("CALLO_BENCH_MODELS", "local", "system1", "Qwen3.5-9B-Q8_0.gguf"))
pytestmark = pytest.mark.live


class Live:
    def __init__(self, llama_server: Path, tmp: Path, extra=()):
        args, extra_args = split_passthrough(["serve", "--model", str(MODEL), "--llama-server", str(llama_server),
                                              "--port", "0", "--ctx", "32768", "--cache-dir", str(tmp), *extra])
        self.args = build_parser().parse_args(args)
        self.args.extra = extra_args
        self.stop, self.port, self.rc = threading.Event(), None, None
        ready = threading.Event()

        def run():
            self.rc = serve(self.args, ready=lambda p: (setattr(self, "port", p), ready.set()), stop=self.stop)
            ready.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        ready.wait(timeout=600)
        assert self.port, f"callo did not come up (rc={self.rc}); see {tmp}"

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=60)
        return self.rc


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status, dict(r.headers), json.loads(r.read())


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


@pytest.fixture(params=[pytest.param(MASTER, id="master-4ebdf2c"), pytest.param(BREW, id="brew-b9890")])
def live(request, tmp_path, caplog):
    exe = request.param
    if not exe.exists() or not MODEL.exists() or os.environ.get("CALLO_NO_LIVE"):
        pytest.skip(f"missing {exe} or {MODEL}")
    caplog.set_level("INFO", logger="callo")
    srv = Live(exe, tmp_path)
    yield srv, caplog
    assert srv.finish() == 0


def test_launch_probe_selftest_and_routes(live):
    srv, caplog = live
    text = "\n".join(r.getMessage() for r in caplog.get_records("setup"))   # startup logs during fixture setup
    assert "idle-clear probe: slot 3 kept" in text                      # launch flags make pinning safe
    assert "self-test ok: 2 + 2 = ? -> answer=4" in text
    assert "answer slot check: answer slot confirmed" in text
    stats = _get(f"{srv.url}/callo/stats")
    assert stats["router"]["pinning"] is True and stats["probe"]["clears_idle"] is False
    assert stats["slots"] is not None and len(stats["slots"]) == 4
    body = {"state": {"subject": "biology", "question": "What carries oxygen in the blood?"},
            "questions": {"answer": {"type": "choice", "instructions": "Which option correctly answers the question?",
                                     "criteria": {"a": "Platelets", "b": "Hemoglobin", "c": "Insulin", "d": "Bile"}}}}
    status, headers, out = _post(f"{srv.url}/v1/systemone", body)
    assert status == 200 and headers["X-Callo-Slot"] == "0" and out["answers"]["answer"]["choice"] == "b"
    assert out["answers"]["answer"]["probabilities"]["b"] > 0.8
    chat = {"model": "x", "messages": [{"role": "user", "content": "Say the single word: hello"}],
            "max_tokens": 8, "temperature": 0, "seed": 1, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}       # else 8 tokens of thinking leave content empty
    status, headers, out = _post(f"{srv.url}/v1/chat/completions", chat)
    assert status == 200 and headers["X-Callo-Slot"] in ("1", "2", "3")
    assert out["choices"][0]["message"]["content"]
    stats = _get(f"{srv.url}/callo/stats")
    assert stats["routes"]["/v1/chat/completions"] == 1 and stats["s1"]["n"] >= 1
