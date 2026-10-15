"""Helpers shared by the CLI tests: a tiny GGUF, JSON GET/POST against a running callo, and `serve` in a thread.
(Moved out of test_cli.py so test_calibrate_bias.py does not import one test module from another.)"""
import json
import os
import string
import threading
import urllib.request

import pytest

from callo.cli import build_parser, serve, split_passthrough

# Shared, env-overridable startup/readiness ceiling for the fake-llama-server subprocess tests. A contended CI runner
# (the macOS GitHub runner, where a cold Python interpreter plus the fake's imports can take many seconds) needs a
# generous value; a healthy local box reaches "listening on" in well under a second, so a high ceiling only ever
# affects the failure path. CI sets CALLO_TEST_STARTUP_TIMEOUT in .github/workflows/ci.yml.
STARTUP_TIMEOUT = float(os.environ.get("CALLO_TEST_STARTUP_TIMEOUT", "90"))


def bench_path(var: str, *parts: str) -> str:
    """A file of the user's own data or models under `$var` (one of conftest's BENCH_ENV, bench/README.md). A test
    that reads it skips when it is not there, so an unset variable (conftest's placeholder) skips it."""
    return os.path.join(os.environ[var], *parts)

TOKENS = list(string.ascii_uppercase) + ["Ġ" + c for c in string.ascii_uppercase] + ["**", "Ġthe", "x"]


def tiny_gguf(path, decision=None):
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(TOKENS)
    if decision:
        w.add_string("qwen35.decision.type", decision)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, dict(r.headers), json.loads(r.read())


def post_json(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, dict(r.headers), json.loads(r.read())


class Running:
    """`serve` in a thread, with the bound port and a stop event."""

    def __init__(self, argv):
        args, extra = split_passthrough(argv)
        self.args = build_parser().parse_args(args)
        self.args.extra = extra
        self.stop, self.port, self.rc = threading.Event(), None, None
        ready = threading.Event()

        def run():
            self.rc = serve(self.args, ready=lambda p: (setattr(self, "port", p), ready.set()), stop=self.stop)
            ready.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        ready.wait(timeout=STARTUP_TIMEOUT)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def finish(self, timeout=STARTUP_TIMEOUT):
        self.stop.set()
        self.thread.join(timeout=timeout)
        return self.rc


# ---- multipart bodies (v0.3 transcriptions) ----------------------------------------------------------------------

def form_field(name: str, value: bytes | str) -> bytes:
    """One multipart field part (its headers, the blank line, its content), as `curl -F name=value` writes it."""
    v = value.encode() if isinstance(value, str) else value
    return f'Content-Disposition: form-data; name="{name}"'.encode() + b"\r\n\r\n" + v


def form_file(data: bytes, *, name: str = "file", filename: str = "clip.wav", ctype: str | None = "audio/wav") -> bytes:
    """One multipart file part, as `curl -F file=@clip.wav` writes it (`filename` is sent as UTF-8)."""
    head = f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode()
    if ctype is not None:
        head += f"\r\nContent-Type: {ctype}".encode()
    return head + b"\r\n\r\n" + data


def multipart(*parts: bytes, boundary: bytes = b"XyZ", preamble: bytes = b"", epilogue: bytes = b"\r\n") -> bytes:
    """A CRLF multipart/form-data body: `--B` CRLF before each part, CRLF after it, then the close `--B--`."""
    out = preamble
    for p in parts:
        out += b"--" + boundary + b"\r\n" + p + b"\r\n"
    return out + b"--" + boundary + b"--" + epilogue
