"""callo.modelkey on synthetic GGUFs: what changes the key and what must not."""
import json
from pathlib import Path

import numpy as np
import pytest
from helpers import bench_path

import callo.modelkey as mk
from callo.errors import InvalidInput
from callo.modelkey import KEY_RE, compute_model_key, load_model_key, shard_paths

gguf = pytest.importorskip("gguf")
TOKENS = ["A", "B", "C", "ĠA", "ĠB", "ĠC"]
BIG = 65536                     # float32 elements = 256 KiB: larger than the two 64 KiB windows
TIEL = Path(bench_path("CALLO_BENCH_MODELS", "peculiar-ragdoll", "Tiel-Coder-35B-A3B-GGUF",
                       "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf"))
TIEL_MTP = Path(bench_path("CALLO_BENCH_MODELS", "local", "Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"))


def _body():
    return {"token_embd.weight": np.linspace(-1, 1, 4096, dtype=np.float32),
            "blk.0.attn_q.weight": np.linspace(0, 1, BIG, dtype=np.float32),
            "blk.1.ffn_down.weight": np.full(2048, 0.25, dtype=np.float32),
            "blk.2.ffn_up.weight": np.arange(1024, dtype=np.float32),
            "output.weight": np.linspace(1, 2, 4096, dtype=np.float32)}


def _mtp(dtype):
    return {"blk.3.nextn.eh_proj.weight": np.linspace(0, 3, 512).astype(dtype),
            "blk.3.attn_q.weight": np.linspace(3, 4, 512).astype(dtype)}


def _write(path, *, tensors=None, nextn=None, tokens=TOKENS, template=None, name=None, kv=None, header=True,
           block_count=True, raw_kv=None):
    """A qwen35 GGUF with a 3-block body; `nextn` adds that many MTP blocks to block_count, as a real graft does.
    `block_count`: True for that count, False for no block_count key, or an int written as is."""
    w = gguf.GGUFWriter(str(path), "qwen35")
    if header:
        if block_count is not False:
            w.add_block_count(3 + (nextn or 0) if block_count is True else block_count)
        if nextn is not None:
            w.add_uint32("qwen35.nextn_predict_layers", nextn)
        if name:
            w.add_name(name)
        w.add_tokenizer_model("gpt2")
        w.add_token_list(tokens)
        if template is not None:
            w.add_chat_template(template)
    for k, v in (kv or {}).items():
        w.add_uint16(k, v)
    for k, v in (raw_kv or {}).items():                 # a malformed value: a string or an array where a count goes
        if isinstance(v, str):
            w.add_string(k, v)
        else:
            w.add_array(k, v)
    for tname, arr in (_body() if tensors is None else tensors).items():
        w.add_tensor(tname, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_the_key_is_well_formed_and_deterministic(tmp_path):
    a = _write(tmp_path / "a.gguf")
    assert KEY_RE.fullmatch(compute_model_key(a)) and compute_model_key(a) == compute_model_key(a)


def test_a_different_middle_body_tensor_gives_a_different_key(tmp_path):
    body = _body()
    body["blk.1.ffn_down.weight"] = np.full(2048, 0.5, dtype=np.float32)
    assert compute_model_key(_write(tmp_path / "a.gguf")) != compute_model_key(_write(tmp_path / "b.gguf", tensors=body))


def test_a_retyped_body_tensor_gives_a_different_key(tmp_path):
    body = _body()
    body["blk.2.ffn_up.weight"] = body["blk.2.ffn_up.weight"].astype(np.float16)
    assert compute_model_key(_write(tmp_path / "a.gguf")) != compute_model_key(_write(tmp_path / "b.gguf", tensors=body))


def test_an_mtp_head_grafted_at_a_different_quant_type_keeps_the_key(tmp_path):
    base = compute_model_key(_write(tmp_path / "base.gguf"))
    f32 = _write(tmp_path / "f32.gguf", tensors={**_body(), **_mtp(np.float32)}, nextn=1, name="Tiel-MTP")
    f16 = _write(tmp_path / "f16.gguf", tensors={**_body(), **_mtp(np.float16)}, nextn=1)
    assert compute_model_key(f32) == compute_model_key(f16) == base     # block_count 3 -> 4 and the nextn key too


def test_a_nextn_key_without_a_usable_block_count_excludes_nothing(tmp_path):
    """With nextn > 0 but no block_count, every blk.* would count as MTP and drop
    out of the key; the key then hashes every block instead."""
    body = _body()
    body["blk.1.ffn_down.weight"] = np.full(2048, 0.5, dtype=np.float32)
    a = _write(tmp_path / "a.gguf", nextn=1, block_count=False)
    b = _write(tmp_path / "b.gguf", tensors=body, nextn=1, block_count=False)
    assert compute_model_key(a) != compute_model_key(b)


def test_a_nextn_not_below_block_count_hashes_the_full_count_and_excludes_nothing(tmp_path):
    """One body definition: the hashed block count and the MTP exclusion agree. With nextn >=
    block_count nothing is excluded, so the full block count is hashed and the nextn key is the only difference."""
    plain = compute_model_key(_write(tmp_path / "p.gguf"))                    # block_count 3, no nextn key
    assert compute_model_key(_write(tmp_path / "e.gguf", nextn=3, block_count=3)) == plain
    assert compute_model_key(_write(tmp_path / "g.gguf", nextn=5, block_count=3)) == plain
    body = _body()
    body["blk.2.ffn_up.weight"] = np.arange(1, 1025, dtype=np.float32)        # the last block still counts
    assert compute_model_key(_write(tmp_path / "x.gguf", tensors=body, nextn=3, block_count=3)) != plain


def test_the_chat_template_does_not_change_the_key_and_the_vocabulary_does(tmp_path):
    a = compute_model_key(_write(tmp_path / "a.gguf", template="{{ messages }}"))
    assert compute_model_key(_write(tmp_path / "b.gguf", template="{% for m in messages %}{{ m }}{% endfor %}")) == a
    assert compute_model_key(_write(tmp_path / "c.gguf")) == a                                 # no template at all
    assert compute_model_key(_write(tmp_path / "d.gguf", tokens=[*TOKENS[:-1], "ĠD"])) != a


def test_small_tensors_are_hashed_whole_and_a_big_tensors_interior_is_not_sampled(tmp_path):
    small = {"blk.0.w": np.arange(16, dtype=np.float32), "blk.1.w": np.arange(32, dtype=np.float32)}
    k = compute_model_key(_write(tmp_path / "s.gguf", tensors=small))
    assert KEY_RE.fullmatch(k)
    small2 = {**small, "blk.1.w": np.concatenate([np.arange(31), [99]]).astype(np.float32)}   # the last element
    assert compute_model_key(_write(tmp_path / "s2.gguf", tensors=small2)) != k
    body = _body()
    body["blk.0.attn_q.weight"] = body["blk.0.attn_q.weight"].copy()
    body["blk.0.attn_q.weight"][BIG // 2] = 7.0                     # byte 128 KiB: outside both 64 KiB windows
    assert compute_model_key(_write(tmp_path / "i.gguf", tensors=body)) == compute_model_key(_write(tmp_path / "j.gguf"))


def test_a_big_tensors_first_and_last_elements_are_sampled(tmp_path):
    """Complements the interior test: the first and last 64 KiB windows of a tensor over 128 KiB are hashed."""
    k = compute_model_key(_write(tmp_path / "base.gguf"))
    for i in (0, BIG - 1):
        body = _body()
        body["blk.0.attn_q.weight"] = body["blk.0.attn_q.weight"].copy()
        body["blk.0.attn_q.weight"][i] = 7.0
        assert compute_model_key(_write(tmp_path / f"e{i}.gguf", tensors=body)) != k


def test_a_split_gguf_hashes_every_shard_and_matches_the_merged_file(tmp_path):
    merged = compute_model_key(_write(tmp_path / "m.gguf"))
    names = list(_body())
    split = {"split.no": 0, "split.count": 2}
    first = tmp_path / "s-00001-of-00002.gguf"
    _write(first, tensors={k: _body()[k] for k in names[:3]}, kv=split)
    second = tmp_path / "s-00002-of-00002.gguf"
    _write(second, tensors={k: _body()[k] for k in names[3:]}, kv={**split, "split.no": 1}, header=False)
    assert shard_paths(first) == [first, second]
    assert compute_model_key(first) == merged
    second.unlink()
    with pytest.raises(InvalidInput, match="s-00002-of-00002.gguf is missing"):
        compute_model_key(first)


def test_a_file_that_is_not_a_gguf_is_invalid_input(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUF" + b"\0" * 64)
    with pytest.raises(InvalidInput):
        compute_model_key(bad)


@pytest.mark.parametrize("raw_kv", [{"qwen35.block_count": "three"}, {"qwen35.block_count": [3, 4]},
                                    {"qwen35.nextn_predict_layers": "one"}, {"qwen35.nextn_predict_layers": [1]}])
def test_malformed_block_count_or_nextn_metadata_is_invalid_input(tmp_path, raw_kv):
    """int() of a malformed count is InvalidInput naming the field, never a bare ValueError/TypeError."""
    g = _write(tmp_path / "m.gguf", block_count=False, raw_kv=raw_kv)
    (field,) = raw_kv
    with pytest.raises(InvalidInput, match=field):
        compute_model_key(g)


def test_a_cached_key_with_a_trailing_newline_is_a_miss(tmp_path):
    """`$` matches before a trailing newline; every KEY_RE check is a fullmatch."""
    g = _write(tmp_path / "m.gguf")
    cache = tmp_path / "cache"
    cache.mkdir()
    entry = cache / mk._cache_name(shard_paths(g))
    entry.write_text(json.dumps({"model_key": "sha256:" + "00" * 32 + "\n"}))
    assert load_model_key(g, cache) == compute_model_key(g)


def test_load_model_key_caches_and_survives_a_corrupt_cache(tmp_path, monkeypatch):
    g = _write(tmp_path / "m.gguf")
    cache = tmp_path / "cache"
    key = load_model_key(g, cache)
    (entry,) = cache.glob("modelkey-*.json")
    assert json.loads(entry.read_text()) == {"model_key": key}
    monkeypatch.setattr(mk, "compute_model_key", lambda p: pytest.fail("cache miss"))
    assert load_model_key(g, cache) == key
    monkeypatch.undo()
    entry.write_text("{not json")
    assert load_model_key(g, cache) == key and json.loads(entry.read_text()) == {"model_key": key}
    assert load_model_key(g, None) == key


def test_a_cache_entry_from_another_key_recipe_is_not_used(tmp_path, monkeypatch):
    """The cache name folds in the key-recipe version (_DOMAIN): a recipe change never reads an old cached key."""
    g = _write(tmp_path / "m.gguf")
    cache = tmp_path / "cache"
    cache.mkdir()
    stale = "sha256:" + "00" * 32
    monkeypatch.setattr(mk, "_DOMAIN", b"callo-model-key-v0\0")
    (cache / mk._cache_name(shard_paths(g))).write_text(json.dumps({"model_key": stale}))
    monkeypatch.undo()
    assert load_model_key(g, cache) == compute_model_key(g) != stale


@pytest.mark.live
def test_the_real_tielcoder_mtp_graft_keeps_the_key():
    """No GPU: reads two real GGUFs' headers and samples (about 10 s). block_count 40 vs 41, same body."""
    if not (TIEL.is_file() and TIEL_MTP.is_file()):
        pytest.skip("the TielCoder GGUFs are not under CALLO_BENCH_MODELS")
    assert compute_model_key(TIEL) == compute_model_key(TIEL_MTP)
