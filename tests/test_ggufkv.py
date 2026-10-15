"""A GGUF whose tensors use a type gguf-py does not know (a quantization newer than the installed gguf-py) must never
stop `callo serve` or `calibrate`: callo reads only the header's keys and, for the model key, each tensor's name, shape,
type id and a sample of its bytes."""
import hashlib
import json
import string
import struct

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from callo import ggufkv, modelkey  # noqa: E402
from callo.engine import read_gguf_meta  # noqa: E402
from callo.marginalize import option_sets_from_gguf, read_vocab_from_gguf  # noqa: E402
from callo.modelkey import compute_model_key  # noqa: E402
from callo.think import read_gguf_sampling  # noqa: E402

TOKENS = list(string.ascii_uppercase) + ["Ġ" + c for c in string.ascii_uppercase] + ["**", "Ġthe", "x"]
UNKNOWN = 42                    # no GGMLQuantizationType has it: the GSQ-RCO IQ3_XXS crash


def _write(path, *, unknown=False):
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_block_count(1)
    w.add_tokenizer_model("gpt2")
    w.add_token_list(TOKENS)
    w.add_float32("general.sampling.temp", 0.7)
    w.add_tensor("blk.0.attn_q.weight", np.linspace(0, 1, 256).astype(np.float32))
    w.add_tensor("blk.0.ffn_up.weight", np.linspace(1, 2, 512).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    if unknown:                 # the tensor-info entry: name, n_dims u32, dims u64 x n_dims, type u32, offset u64
        raw = bytearray(path.read_bytes())
        name = b"blk.0.ffn_up.weight"
        at = raw.index(name) + len(name)
        n_dims = struct.unpack_from("<I", raw, at)[0]
        type_at = at + 4 + 8 * n_dims
        assert struct.unpack_from("<I", raw, type_at)[0] == int(gguf.GGMLQuantizationType.F32)
        struct.pack_into("<I", raw, type_at, UNKNOWN)
        path.write_bytes(bytes(raw))
    return path


def _reference_key(path):
    """compute_model_key as it hashed a one-shard file through gguf-py's full GGUFReader, before the KV-only reader."""
    r = gguf.GGUFReader(str(path), "r")
    h = hashlib.sha256(modelkey._DOMAIN)
    arch = str(r.get_field("general.architecture").contents())
    for name, field in r.fields.items():
        if name.startswith(modelkey._SKIP) or name == f"{arch}.nextn_predict_layers":
            continue
        h.update(b"K" + name.encode() + b"\0")
        if name == f"{arch}.block_count":
            h.update(str(int(field.contents())).encode())
            continue
        for part in field.parts[2:]:
            h.update(part.tobytes())
    for t in r.tensors:
        h.update(b"T" + json.dumps([t.name, [int(d) for d in t.shape], int(t.tensor_type)]).encode() + b"\0")
        a0, n = t.data_offset, t.n_bytes
        if n <= 2 * modelkey.SAMPLE:
            h.update(bytes(r.data[a0:a0 + n]))
        else:
            h.update(bytes(r.data[a0:a0 + modelkey.SAMPLE]))
            h.update(bytes(r.data[a0 + n - modelkey.SAMPLE:a0 + n]))
    return "sha256:" + h.hexdigest()


def test_the_fixture_reproduces_the_crash_with_gguf_py_s_own_reader(tmp_path):
    with pytest.raises(ValueError, match="is not a valid GGMLQuantizationType"):
        gguf.GGUFReader(str(_write(tmp_path / "u.gguf", unknown=True)), "r")


def test_every_header_read_works_on_an_unknown_tensor_type(tmp_path):
    p = _write(tmp_path / "u.gguf", unknown=True)
    m = read_gguf_meta(p)
    assert (m.architecture, m.tok_model, m.decision_type, m.tokens[:2]) == ("qwen35", "gpt2", None, ["A", "B"])
    assert read_vocab_from_gguf(p) == (TOKENS, "gpt2")
    assert set(option_sets_from_gguf(p, ["A", "B"])) == {"A", "B"}
    g = read_gguf_sampling(p)
    assert g.error is None and g["temp"] == pytest.approx(0.7)
    key = compute_model_key(p)
    assert modelkey.KEY_RE.fullmatch(key) and key != compute_model_key(_write(tmp_path / "k.gguf"))


def test_tensor_infos_list_known_and_unknown_types_in_file_order(tmp_path):
    p = _write(tmp_path / "u.gguf", unknown=True)
    q, up = ggufkv.tensor_infos(ggufkv.open_kv(p))
    assert (q.name, q.shape, q.type_id, q.n_bytes) == ("blk.0.attn_q.weight", (256,), 0, 256 * 4)
    assert (up.name, up.type_id) == ("blk.0.ffn_up.weight", UNKNOWN)
    assert up.data_offset > q.data_offset and up.data_offset + up.n_bytes == p.stat().st_size   # to the file's end


def test_the_model_key_of_known_types_is_unchanged(tmp_path):
    p = _write(tmp_path / "k.gguf")
    assert compute_model_key(p) == _reference_key(p)


def test_a_gguf_py_without_the_overridden_hook_is_a_clear_refusal(tmp_path, monkeypatch):
    from callo.errors import InvalidInput

    class Renamed:                                  # a gguf-py whose reader no longer has _build_tensors
        def __init__(self, *a, **kw):
            raise AssertionError("never constructed")

    monkeypatch.setattr(gguf, "GGUFReader", Renamed)
    with pytest.raises(InvalidInput, match=r"GGUFReader\._build_tensors"):
        ggufkv.open_kv(_write(tmp_path / "k.gguf"))

    class Skips(gguf.GGUFReader):                   # one that has it but stopped calling it
        def __init__(self, path, mode="r"):
            self.fields, self.tensors, self.data = {}, [], b""

        def _build_tensors(self, start_offs, fields):  # kept, never called
            raise AssertionError("never called")

    monkeypatch.setattr(gguf, "GGUFReader", Skips)
    with pytest.raises(InvalidInput, match="no longer calls"):
        ggufkv.open_kv(_write(tmp_path / "k2.gguf"))


def _patch_tensor(path, name, *, rename=None, type_id=None):
    """Rewrite one tensor-info entry in place: its name (same length) and/or its type id."""
    raw = bytearray(path.read_bytes())
    at = raw.index(name) + len(name)
    if rename is not None:
        assert len(rename) == len(name)
        raw[at - len(name):at] = rename
    if type_id is not None:
        n_dims = struct.unpack_from("<I", raw, at)[0]
        struct.pack_into("<I", raw, at + 4 + 8 * n_dims, type_id)
    path.write_bytes(bytes(raw))
    return path


def _refused_by_every_reader(p, match):
    from callo.errors import InvalidInput

    with pytest.raises(ValueError, match=match):
        ggufkv.open_kv(p)
    with pytest.raises(InvalidInput, match="not a readable GGUF"):
        read_gguf_meta(p)
    with pytest.raises(InvalidInput, match="not a readable GGUF"):
        compute_model_key(p)


def test_a_duplicated_tensor_name_is_refused_as_gguf_py_refuses_it(tmp_path):
    p = _patch_tensor(_write(tmp_path / "d.gguf"), b"blk.0.ffn_up.weight", rename=b"blk.0.attn_q.weight")
    with pytest.raises(ValueError, match="duplicated tensor"):
        gguf.GGUFReader(str(p), "r")
    _refused_by_every_reader(p, r"duplicated tensor.*blk\.0\.attn_q\.weight")


@pytest.mark.parametrize("cut", [1, 100, 2048])
def test_a_truncated_gguf_of_known_types_is_refused(tmp_path, cut):
    p = _write(tmp_path / "t.gguf")                  # blk.0.ffn_up.weight, the last tensor, holds 2048 bytes
    p.write_bytes(p.read_bytes()[:-cut])
    with pytest.raises(ValueError):
        gguf.GGUFReader(str(p), "r")                 # gguf-py's reader refused it too
    _refused_by_every_reader(p, r"blk\.0\.ffn_up\.weight.*truncated")


@pytest.mark.parametrize("how", ["cut at its data", "offset past the end"])
def test_an_unknown_type_whose_data_starts_past_the_end_is_refused(tmp_path, how):
    p = _write(tmp_path / "u.gguf", unknown=True)
    _, up = ggufkv.tensor_infos(ggufkv.open_kv(p))
    raw = bytearray(p.read_bytes())
    if how == "cut at its data":                     # a zero-byte span: the file ends where its data would start
        raw = raw[:up.data_offset]
    else:                                            # a span that would be negative: the offset u64 after the type
        name = b"blk.0.ffn_up.weight"
        at = raw.index(name) + len(name)
        off_at = at + 4 + 8 * struct.unpack_from("<I", raw, at)[0] + 4
        struct.pack_into("<Q", raw, off_at, struct.unpack_from("<Q", raw, off_at)[0] + 4096)
    p.write_bytes(bytes(raw))
    _refused_by_every_reader(p, r"blk\.0\.ffn_up\.weight \(type 42\).*truncated")


def test_a_quantized_row_that_is_not_a_whole_number_of_blocks_is_refused(tmp_path):
    w = gguf.GGUFWriter(str(tmp_path / "q.gguf"), "qwen35")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(TOKENS)
    w.add_tensor("blk.0.attn_q.weight", np.zeros(100, dtype=np.float32))   # 400 bytes of data
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    p = _patch_tensor(tmp_path / "q.gguf", b"blk.0.attn_q.weight", type_id=int(gguf.GGMLQuantizationType.Q8_0))
    with pytest.raises(ValueError, match="not a multiple"):
        gguf.GGUFReader(str(p), "r")
    _refused_by_every_reader(p, r"blk\.0\.attn_q\.weight.*row of 100.*Q8_0 block of 32")


def test_a_real_malformed_header_reaches_the_sampling_fallback(tmp_path):
    key = b"general.sampling.temp"                   # magic, version 3, 0 tensors, 1 key whose value type is 99
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, len(key)) + key + struct.pack("<I", 99) + b"\0" * 8)
    g = read_gguf_sampling(p)
    assert g == {} and "could not be read" in g.error and "is not a valid GGUFValueType" in g.error
