"""The wide-choice validation items: the builder's rules on synthetic rows, its HF cache guard,
and the committed `bench/data/wide_items.jsonl` (a `_meta` row with the attribution, then 40 items)."""
import json
import os
from pathlib import Path

import pytest

from bench import wide_items as w

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "bench" / "data" / "wide_items.jsonl"


def _names(n: int, prefix: str = "intent") -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def _rows(names: list[str], per: int = 3) -> list[tuple[str, int]]:
    """(text, label index) in a file-like order: every label `per` times, interleaved, plus an empty text."""
    rows = [(f"utterance {k} of {names[i]}", i) for k in range(per) for i in range(len(names))]
    rows.insert(5, ("   ", 0))
    return rows


# ---- the builder's rules ----------------------------------------------------------------------------------------

def test_constants_pin_revisions_files_and_licences():
    assert w.REVISIONS == {"banking77": "f54121560de48f2852f90be299010d1d6dc612ec",
                           "clinc150": "155b9c710419136e17307b80d0a13e68cd46b4ec"}
    assert w.SOURCES["banking77"]["dataset"] == "PolyAI/banking77" and w.SOURCES["banking77"]["license"] == "CC-BY-4.0"
    assert w.SOURCES["clinc150"]["repo"] == "clinc/clinc_oos" and w.SOURCES["clinc150"]["license"] == "CC-BY-3.0"
    assert w.SOURCES["clinc150"]["file"].startswith("plus/train-") and "train" in w.SOURCES["banking77"]["file"]
    assert {s["n_labels"] for s in w.SOURCES.values()} == {77, 151}
    assert all(len(s["sha256"]) == 64 for s in w.SOURCES.values())
    assert (w.N_ITEMS, w.N_CONTROL, w.SAMPLE_SEED, w.DISTRACTOR_SEED, w.SHUFFLE_SEED) == (20, 10, 0, 1, 2)


def test_items_are_20_distinct_intents_one_train_utterance_each_with_the_full_label_list():
    names = _names(77)
    items = w.sample_items("banking77", _rows(names), names)
    assert len(items) == 20 and len({it["gold"] for it in items}) == 20
    for it in items:
        assert set(it) == {"id", "source", "split", "revision", "license", "text", "gold", "labels", "control"}
        assert it["split"] == "train" and it["source"] == "banking77" and it["revision"] == w.REVISIONS["banking77"]
        assert it["license"] == "CC-BY-4.0" and it["labels"] == names            # every intent, in dataset order
        assert it["text"].startswith("utterance ") and it["text"].endswith(f"of {it['gold']}")
        row = int(it["id"].rsplit("/", 1)[1])
        assert it["id"] == f"banking77/train/{row}"
    assert len({it["id"] for it in items}) == 20


def test_controls_hold_the_gold_and_nine_distinct_distractors_shuffled():
    names = _names(151)
    items = w.sample_items("clinc150", _rows(names), names)
    for it in items:
        c = it["control"]
        assert len(c) == 10 == len(set(c)) and it["gold"] in c and set(c) <= set(names)
    assert {it["control"].index(it["gold"]) for it in items} != {0}           # shuffled: the gold is not always first


def test_the_builder_is_deterministic_and_depends_only_on_its_rows():
    names = _names(77)
    a = w.sample_items("banking77", _rows(names), names)
    assert a == w.sample_items("banking77", _rows(names), names)
    meta = w.meta_row()
    assert w.jsonl([meta, *a]) == w.jsonl([meta, *a]) and w.jsonl([meta, *a]).count(b"\n") == 21
    other = w.sample_items("banking77", _rows(names, per=4), names)
    assert [it["gold"] for it in other] == [it["gold"] for it in a]           # the intents depend on the labels only


def test_empty_texts_are_never_picked_and_an_intent_without_text_is_refused():
    names = _names(20)
    rows = [("", i) for i in range(20)] + [(f"t{i}", i) for i in range(20)]
    assert all(it["text"].strip() for it in w.sample_items("banking77", rows, names))
    with pytest.raises(ValueError, match="no usable train utterance"):
        w.sample_items("banking77", [(f"t{i}", i) for i in range(19)], names)


def test_label_names_come_from_the_parquet_class_label_metadata():
    md = {b"huggingface": json.dumps({"info": {"features": {"text": {"dtype": "string", "_type": "Value"},
                                                             "intent": {"names": ["b", "a"], "_type": "ClassLabel"}}}})}
    assert w.label_names(md, "intent") == ["b", "a"]
    with pytest.raises(ValueError, match="ClassLabel"):
        w.label_names(md, "text")


@pytest.mark.parametrize(("env", "ok"), [
    ({"HF_HOME": "/data/hf-home", "HF_XET_CACHE": "/data/hf-home/xet"}, True),
    ({}, False),
    ({"HF_HOME": "/data/hf-home"}, False),
    # "~" is expanded when the check runs, against the test's HOME (conftest `_isolated_home`), not at collection
    ({"HF_HOME": "~/.cache/huggingface", "HF_XET_CACHE": "/data/x"}, False),
    ({"HF_HOME": "/data/hf-home", "HF_XET_CACHE": "~/.cache/huggingface/xet"}, False),
])
def test_the_hf_cache_must_be_set_and_off_the_boot_cache(env, ok):
    problem = w.hf_cache_problem(env)
    assert (problem is None) == ok, problem


def test_main_refuses_without_the_vault_cache(capsys, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    assert w.main([]) == 2
    assert "HF_HOME" in capsys.readouterr().err


def test_main_builds_from_the_download_seam_and_checks_sha256(tmp_path, monkeypatch):
    """The live path's shape on fakes: a fetched file whose sha256 is not the pinned one is refused."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "hf" / "xet"))
    bad = tmp_path / "x.parquet"
    bad.write_bytes(b"not the pinned file")
    monkeypatch.setattr(w, "_download", lambda source: bad)
    with pytest.raises(SystemExit, match="sha256"):
        w.main(["--out", str(tmp_path / "items.jsonl")])
    assert not (tmp_path / "items.jsonl").exists()


# ---- the committed file -----------------------------------------------------------------------------------------

def _committed():
    return [json.loads(line) for line in ITEMS.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_committed_file_is_a_meta_row_then_40_items():
    rows = _committed()
    meta, items = rows[0], rows[1:]
    assert set(meta) == {"_meta"} and len(items) == 40
    m = meta["_meta"]
    assert m["seeds"] == {"sample": 0, "distractors": 1, "shuffle": 2}
    for src in ("banking77", "clinc150"):
        s = m["sources"][src]
        assert s["revision"] == w.REVISIONS[src] and s["sha256"] == w.SOURCES[src]["sha256"]
        assert s["license"] == w.SOURCES[src]["license"] and s["attribution"]
    assert "Casanueva" in m["sources"]["banking77"]["attribution"]
    assert "Larson" in m["sources"]["clinc150"]["attribution"]
    assert [it["source"] for it in items] == ["banking77"] * 20 + ["clinc150"] * 20
    for it in items:
        src = w.SOURCES[it["source"]]
        assert it["split"] == "train" and it["revision"] == w.REVISIONS[it["source"]]
        assert it["license"] == src["license"] and len(it["labels"]) == src["n_labels"] == len(set(it["labels"]))
        assert it["gold"] in it["labels"] and it["gold"] in it["control"] and len(set(it["control"])) == 10
        assert set(it["control"]) <= set(it["labels"]) and it["text"].strip()
    assert "oos" in items[-1]["labels"] and items[0]["labels"][0] == "activate_my_card"
    for src in ("banking77", "clinc150"):
        assert len({it["gold"] for it in items if it["source"] == src}) == 20


def test_the_committed_file_rebuilds_byte_identically_from_the_cached_parquet():
    """Gated: the pinned parquet files in the HF cache the builder used ($HF_HOME) and pyarrow (a builder-only
    dependency)."""
    pytest.importorskip("pyarrow")
    if os.environ.get("CALLO_TEST_HIDE_VOLUMES"):
        pytest.skip("volumes hidden")
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        pytest.skip("HF_HOME is not set")
    paths = {s: w.cached_path(s, hf_home) for s in w.SOURCES}
    if not all(p.is_file() for p in paths.values()):
        pytest.skip(f"the pinned parquet files are not in {hf_home}")
    assert w.build_from_files(paths) == ITEMS.read_bytes()
