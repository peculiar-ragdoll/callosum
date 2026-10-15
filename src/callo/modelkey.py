"""`model_key`: a sha256 over exactly what reaches the S1 readout, so a position-bias
profile refuses another model and survives what cannot change the readout.

Hashed, per shard in order: every GGUF KV entry in file order except `general.*`, `quantize.*`,
`tokenizer.chat_template*`, `split.*`, `<arch>.nextn_predict_layers` and the reader's `GGUF.*` header pseudo-fields;
then every non-MTP tensor in file order: its name, shape and type, and the first and last 64 KiB of its data (the whole
tensor up to 128 KiB). One definition drives both the MTP exclusion and the hashed block count: the BODY is
block_count - nextn_predict_layers blocks when 0 < nextn < block_count (llama-arch.cpp:227; a missing nextn key is 0),
and all block_count blocks otherwise, since a nextn key without a usable block_count would drop every block. Blocks
>= body are the MTP head and are left out, and `<arch>.block_count` is hashed as `body`: converting a non-MTP file to
MTP adds the head's tensors, the nextn key AND one block (TielCoder UD-Q4_K_XL: block_count 40 -> 41, same body), so
the raw count would change an identical body's key. The file size is left out. Leaving `split.*` out makes a split
file's key equal to the merged file's. The key cache's file name folds in the recipe version (`_DOMAIN`), so a
recipe change never reads a key cached under an older one.

So an MTP graft or re-type keeps the key, and a template re-embed keeps it too: template drift is the profile's
`chat_template_sha256` provenance warning (one policy). The sample does not see an edit confined to the interior
of a tensor larger than 128 KiB; that is the price of reading about 65 MiB instead of the whole file (under
0.2 s on TielCoder, plus the GGUF parse the vocabulary-cache miss already pays).

A tensor whose type the installed gguf-py does not know is sampled up to the next tensor's data
(`ggufkv.tensor_infos`). Once gguf-py knows the type its exact size is used, so the key of such a file changes with
that gguf-py upgrade.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from . import ggufkv
from .errors import InvalidInput

KEY_RE = re.compile(r"sha256:[0-9a-f]{64}")    # apply with fullmatch: `$` would accept a trailing newline
SAMPLE = 64 * 1024
_SPLIT = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")
_BLOCK = re.compile(r"^blk\.(\d+)\.")
_SKIP = ("GGUF.", "general.", "quantize.", "tokenizer.chat_template", "split.")
_DOMAIN = b"callo-model-key-v1\0"


def shard_paths(path: str | Path) -> list[Path]:
    """Every shard of a split GGUF (`<stem>-0000i-of-0000N.gguf`) in order, or `[path]` for a single file."""
    p = Path(path)
    m = _SPLIT.match(p.name)
    if m is None:
        return [p]
    n = int(m.group("n"))
    shards = [p.with_name(f"{m.group('stem')}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]
    missing = [s.name for s in shards if not s.is_file()]
    if missing:
        raise InvalidInput(f"{p} is one shard of a {n}-shard GGUF, but {', '.join(missing)} is missing; "
                           f"model_key hashes every shard")
    return shards


def gguf_bytes(path: str | Path) -> int:
    """The size of a GGUF on disk: the sum over every shard of a split one (`shard_paths`; a missing shard is
    InvalidInput naming it), else the file's own size (OSError when it cannot be read)."""
    return sum(p.stat().st_size for p in shard_paths(path))


def _count(reader, name: str, shard: Path) -> int:
    """An integer metadata count, 0 when absent; a malformed one (a string, an array) is InvalidInput."""
    field = reader.get_field(name)
    if field is None:
        return 0
    try:
        return int(field.contents())
    except (TypeError, ValueError) as e:
        raise InvalidInput(f"{shard}: {name} is not an integer ({e})") from None


def compute_model_key(path: str | Path) -> str:
    h = hashlib.sha256(_DOMAIN)
    arch, block_count, body = "", 0, 0
    for i, shard in enumerate(shard_paths(path)):
        try:
            r = ggufkv.open_kv(shard)
        except Exception as e:  # noqa: BLE001 -- gguf raises assorted ValueError/struct errors on garbage
            raise InvalidInput(f"{shard} is not a readable GGUF: {type(e).__name__}: {e}") from e
        if i == 0:
            a = r.get_field("general.architecture")
            if a is None:
                raise InvalidInput(f"{shard} has no general.architecture: not a model GGUF")
            arch = str(a.contents())
            block_count = _count(r, f"{arch}.block_count", shard)
            nextn = _count(r, f"{arch}.nextn_predict_layers", shard)
            body = block_count - nextn if 0 < nextn < block_count else block_count   # blocks >= body: the MTP head
        for name, field in r.fields.items():
            if name.startswith(_SKIP) or name == f"{arch}.nextn_predict_layers":
                continue
            h.update(b"K" + name.encode() + b"\0")
            if name == f"{arch}.block_count":
                h.update(str(body).encode())                     # the body's blocks: an MTP graft adds one
                continue
            for part in field.parts[2:]:                          # the value's type, lengths and bytes
                h.update(part.tobytes())
        for t in ggufkv.tensor_infos(r):
            m = _BLOCK.match(t.name)
            if body < block_count and m is not None and int(m.group(1)) >= body:
                continue                                          # the MTP head never runs on the S1 path
            h.update(b"T" + json.dumps([t.name, list(t.shape), t.type_id]).encode() + b"\0")
            a0, n = t.data_offset, t.n_bytes
            if n <= 2 * SAMPLE:
                h.update(bytes(r.data[a0:a0 + n]))
            else:
                h.update(bytes(r.data[a0:a0 + SAMPLE]))
                h.update(bytes(r.data[a0 + n - SAMPLE:a0 + n]))
    return "sha256:" + h.hexdigest()


def _cache_name(paths: list[Path]) -> str:
    h = hashlib.sha256(_DOMAIN)                               # the recipe version: a new recipe misses old entries
    for p in paths:
        st = p.stat()
        h.update(f"{p.resolve()}\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
    return f"modelkey-{h.hexdigest()}.json"


def load_model_key(path: str | Path, cache_dir: Path | None) -> str:
    """`compute_model_key` through a JSON cache next to the vocabulary cache, keyed like it by (path, size, mtime) of
    every shard. A corrupt or malformed entry is a miss and is rewritten, never a startup crash."""
    if cache_dir is None:
        return compute_model_key(path)
    cached = Path(cache_dir) / _cache_name(shard_paths(path))
    try:
        key = json.loads(cached.read_bytes())["model_key"]
        if isinstance(key, str) and KEY_RE.fullmatch(key):
            return key
    except (OSError, ValueError, KeyError, TypeError):
        pass
    key = compute_model_key(path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")   # pid-unique: concurrent first starts never share it
    try:
        tmp.write_text(json.dumps({"model_key": key}))
        tmp.replace(cached)
    finally:
        tmp.unlink(missing_ok=True)
    return key
