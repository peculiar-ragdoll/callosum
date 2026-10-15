"""A stand-in for `callo serve` as `callo calibrate --bias` sees it (v0.3 Tasks 13-16): GET /callo/stats and
POST /v1/systemone, nothing else.

Each readout runs callo's own readout arithmetic (`bounded_readout`, `answer_body`, `timings_header`) over a
`PlantedBias` dump of the request's own option texts, so a calibrate run against the stub measures the planted
readouts' prior, with `miss` set when the dump's top-k cut floors options. `stats` is the /callo/stats
body, built by callo's own `Stats.snapshot()` so it carries every key a v0.3 callo serves; tests edit it, and
`routes`, `s1.n`, `s1.readouts` and `s1.topk_miss` advance as a real callo's do. Scripting: `script` (statuses to
answer next, e.g. [503, 503]), `fail_after` ((k, status): every POST after k readouts answers status), `slot_after`
((k, slot): X-Callo-Slot changes after k readouts), `on_readout(k, stub)` (called after readout k, to change state
mid-run).
"""
from __future__ import annotations

import copy
import json
import string
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fake_upstream import PlantedBias

from callo.engine import QuestionReadout, Readout, bounded_readout
from callo.profile import WIDE_CONTRACT
from callo.proxy import Stats, timings_header
from callo.slots import SlotRouter
from callo.systemone import InvalidRequest, answer_body, parse_request
from callo.upstream import ProbeResult

KEY = "sha256:" + "5a" * 32


def default_stats() -> dict:
    """The /callo/stats body of a launch-mode `callo serve` on a 4-slot upstream (one reserved S1 slot, no profile),
    produced by callo's own `Stats.snapshot()`, so its shape is exactly what calibrate reads from a real callo."""
    st = Stats()
    st.instance, st.temperature, st.kv = "stub-1", 1.0, "unified"
    st.vision, st.media_marker, st.user_delimiter = False, None, None
    st.cache_prompt = {"plain": False, "image": True, "context": True}
    st.model_key, st.model_key_error, st.model_file = KEY, None, "fake-9b.gguf"
    st.provenance = {"prompt_sha256": "p" * 64, "chat_template_sha256": "t" * 64, "build_info": "b1-926862e",
                     "mmproj": None, "user_delimiter": None, "cache_prompt": False, "server_args": None,
                     "wide_contract": dict(WIDE_CONTRACT)}                   # wide choice: every served provenance
    st.probe = ProbeResult(False, "stub: idle slots keep their prompt", 16, (), "ok").as_dict()
    snap = st.snapshot(SlotRouter(4, 1), [{"id": i, "n_ctx": 8192, "is_processing": False} for i in range(4)])
    snap["uptime_s"] = 0.0
    return snap


class CalloStub:
    def __init__(self, bias: PlantedBias | None = None, *, slot: str = "0"):
        self.bias = bias or PlantedBias()
        self.stats = default_stats()
        self.slot = slot
        self.script: list[int] = []
        self.fail_after: tuple[int, int] | None = None
        self.slot_after: tuple[int, str] | None = None
        self.on_readout: Callable[[int, CalloStub], None] | None = None
        self.posts: list[dict] = []
        self.lanes: list[str | None] = []
        self.readouts = 0
        self.lock = threading.Lock()
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._srv.daemon_threads = True
        threading.Thread(target=self._srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    def readout(self, body: dict) -> tuple[dict, str]:
        """(the /v1/systemone body, its X-Callo-Timings header) for a single-question request."""
        q = parse_request(body).questions[0]
        dump = self.bias.dump(list(q.texts))
        toks = {e["id"]: e["logprob"] for e in dump}
        marg = {lab: toks.get(string.ascii_uppercase.index(lab), float("-inf")) for lab in q.labels}
        probs, lms, miss = bounded_readout(marg, toks, q.labels)
        out = {"model": "fake-9b", "answers": {q.qid: answer_body(q, probs)},
               "usage": {"input_tokens": 10, "output_tokens": 0}}
        r = Readout(probs=probs, log_marginals=lms, option_mass=0.97, stop="diverged", depth=0, topk_miss=miss,
                    prompt_tokens=10)
        return out, timings_header([QuestionReadout(q.qid, r, 1.0)])

    def _handler(self):
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, obj: dict, headers: dict | None = None) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def _count(self) -> str:
                path = self.path.split("?", 1)[0]
                with stub.lock:
                    stub.stats["routes"][path] = stub.stats["routes"].get(path, 0) + 1
                return path

            def do_GET(self):
                if self._count() != "/callo/stats":
                    return self._send(404, {"detail": "not found"})
                with stub.lock:
                    snapshot = copy.deepcopy(stub.stats)
                return self._send(200, snapshot)

            def do_POST(self):
                path = self._count()
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if path != "/v1/systemone":
                    return self._send(404, {"detail": "not found"})
                with stub.lock:
                    stub.posts.append(body)
                    stub.lanes.append(self.headers.get("X-Callo-Conversation"))
                    if stub.script:
                        code = stub.script.pop(0)
                    elif stub.fail_after is not None and stub.readouts >= stub.fail_after[0]:
                        code = stub.fail_after[1]
                    else:
                        code = 200
                if code != 200:
                    return self._send(code, {"detail": f"scripted {code}"}, {"X-Callo-Slot": "-"})
                try:
                    out, timings = stub.readout(body)
                except InvalidRequest as e:
                    return self._send(422, {"detail": str(e)}, {"X-Callo-Slot": "-"})
                with stub.lock:
                    stub.readouts += 1
                    k = stub.readouts
                    s1 = stub.stats["s1"]
                    s1["n"] = min(s1["n"] + 1, 2000)                 # callo's s1.n: a 2,000-wide latency window
                    s1["readouts"] += 1
                    s1["topk_miss"] += json.loads(timings)[0]["miss"]
                    slot = stub.slot_after[1] if stub.slot_after is not None and k > stub.slot_after[0] else stub.slot
                if stub.on_readout is not None:
                    stub.on_readout(k, stub)
                return self._send(200, out, {"X-Callo-Slot": slot, "X-Callo-Timings": timings})

        return H
