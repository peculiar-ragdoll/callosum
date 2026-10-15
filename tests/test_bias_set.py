"""The built-in set builtin-v1: the builder's rules on synthetic rows, the checked-in files (the
repo's bench/data/bias_set.jsonl, the copy uploaded as the HF dataset callo fetches), the licences, and a code-only
wheel and sdist with NOTICE."""
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import pytest
from conftest import bench_importers, real_env
from helpers import bench_path
from test_bias import _contract

from bench import build_bias_set as b
from callo.bias import BUILTIN_SET, BUILTIN_SET_SHA256, read_set

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "bench" / "data" / "bias_set.jsonl"
HELDOUT = ROOT / "bench" / "data" / "bias_heldout_v1.jsonl"
TRANSFER_TEST = Path(bench_path("CALLO_BENCH_DATA", "v4", "transfer-v4", "test.jsonl"))   # Kev's evals/
INTENTS = ["card_arrival", "card_linking", "card_delivery_estimate", "top_up_failed", "top_up_reverted",
           "top_up_limits", "exchange_rate", "pin_blocked", "cash_withdrawal_charge"]


def _records(path_or_bytes):
    text = path_or_bytes.decode() if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---- the builder's rules ----------------------------------------------------------------------------------

def test_mmlu_truncation_keeps_the_gold_and_the_first_distractors_in_file_order():
    row = {"subject": "college_medicine", "question": "Q?", "choices": ["w", "x1", "gold", "z"], "answer": 2}
    rec = b.mmlu_record(7, row, 3, "builtin-v1")
    q = rec["questions"]["answer"]
    assert q["criteria"] == {"a": "w", "b": "x1", "c": "gold"} and q["label"] == "c"
    assert rec["state"] == {"subject": "college medicine", "question": "Q?"}
    assert rec["_meta"] == {"id": "builtin-v1/mmlu/validation/7/n3", "source": "mmlu", "repo": "cais/mmlu",
                            "revision": b.MMLU["revision"], "split": "validation", "row": 7, "n": 3, "variant": "clean"}
    assert b.mmlu_record(7, row, 2, "x")["questions"]["answer"]["criteria"] == {"a": "w", "b": "gold"}
    assert b.mmlu_record(7, row, 4, "x")["questions"]["answer"]["criteria"] == {"a": "w", "b": "x1", "c": "gold",
                                                                                 "d": "z"}


@pytest.mark.parametrize(("choices", "usable"), [
    (["one", "two", "three", "four"], True),
    (["one", "two", "three", "All of the above"], False),       # a cross-referencing distractor drops the row
    (["Both A and B", "two", "three", "four"], False),
    (["B", "two", "three", "four"], False),                     # a bare letter
    (["one", "two", "", "four"], False),
    (["one", "one", "three", "four"], False),
    (["one", "two", "three"], False),
])
def test_mmlu_usable(choices, usable):
    assert b.mmlu_usable({"subject": "s", "question": "q", "choices": choices, "answer": 0}) is usable


def test_banking_distractors_prefer_shared_prefixes_then_token_overlap_seeded():
    rng = random.Random(0)
    d = b.distractors("top_up_failed", INTENTS, 3, rng)
    assert set(d[:2]) == {"top_up_reverted", "top_up_limits"} and d[2].startswith(("card_", "cash_", "pin_",
                                                                                   "exchange_"))
    assert b.lead("top_up_failed", "top_up_limits") == 2 and b.lead("card_arrival", "cash_withdrawal_charge") == 0
    assert b.jaccard("card_arrival", "card_linking") == pytest.approx(1 / 3)
    assert b.distractors("card_arrival", INTENTS, 2, random.Random(1)) == b.distractors(
        "card_arrival", INTENTS, 2, random.Random(1))


def test_intent_names_are_compared_case_insensitively():
    """Banking77 has capitalised names (`Refund_not_showing_up`); `refund` must still match `Refund`."""
    assert b.jaccard("Refund_not_showing_up", "request_refund") == pytest.approx(1 / 5)
    assert b.lead("Card_arrival", "card_linking") == 1
    intents = ["Refund_not_showing_up", "request_refund", "card_arrival", "pin_blocked", "exchange_rate"]
    assert b.distractors("Refund_not_showing_up", intents, 1, random.Random(0)) == ["request_refund"]


def test_banking_record_shape_and_shuffle():
    rec = b.banking_record(17, "Where is my card?", "card_arrival", INTENTS, 4, "builtin-v1")
    q = rec["questions"]["intent"]
    assert q["instructions"] == "Which intent does this customer message express?" and len(q["criteria"]) == 4
    assert q["label"] == "card arrival" and "card arrival" in q["criteria"] and all(v is None for v in q["criteria"].values())
    assert all("_" not in k for k in q["criteria"]) and rec["state"] == "Where is my card?"
    assert rec["_meta"]["split"] == "train" and rec["_meta"]["file"] == "banking_data/train.csv"
    assert b.banking_record(17, "Where is my card?", "card_arrival", INTENTS, 4, "builtin-v1") == rec


def test_pools_split_by_normalised_text_so_duplicates_never_straddle():
    assert b.pool("banking77", "Where is my card?") == b.pool("banking77", "  where IS my   card? ")
    pools = Counter(b.pool("mmlu", f"text {i}") for i in range(2000))
    assert 900 < pools["builtin"] < 1100 and set(pools) == {"builtin", "heldout"}


def test_build_on_synthetic_sources_meets_the_counts_and_keeps_the_pools_disjoint():
    mmlu = [{"subject": "s", "question": f"q{i}", "choices": [f"a{i}", f"b{i}", f"c{i}", f"d{i}"], "answer": i % 4}
            for i in range(800)]
    banking = [(f"message number {i}", INTENTS[i % len(INTENTS)]) for i in range(2000)] + [("message number 3", "pin_blocked")]
    builtin, heldout = b.build(mmlu, banking, INTENTS)
    count = Counter((r["_meta"]["source"], r["_meta"]["n"]) for r in builtin)
    assert count == {("mmlu", 2): 24, ("mmlu", 3): 24, ("mmlu", 4): 24, ("banking77", 2): 24, ("banking77", 3): 24,
                     ("banking77", 4): 24, ("banking77", 5): 48, ("banking77", 6): 48, ("banking77", 7): 48,
                     ("banking77", 8): 48}
    assert Counter((r["_meta"]["source"], r["_meta"]["n"]) for r in heldout) == {
        **{("mmlu", n): 48 for n in (2, 3, 4)}, **{("banking77", n): 48 for n in range(2, 9)}}
    ids = {(r["_meta"]["source"], r["_meta"]["row"]) for r in builtin}
    assert len(ids) == len(builtin) and not ids & {(r["_meta"]["source"], r["_meta"]["row"]) for r in heldout}
    assert 2000 not in {r["_meta"]["row"] for r in builtin + heldout if r["_meta"]["source"] == "banking77"}
    assert b.build(mmlu, banking, INTENTS) == (builtin, heldout)                     # deterministic
    scan = read_set("\n".join(json.dumps(r) for r in builtin))
    assert len(scan.items) == 336 and not scan.skipped


# ---- the checked-in files ---------------------------------------------------------------------------------

def test_the_builtin_set_loads_and_every_question_is_eligible():
    assert BUILTIN_SET == "builtin-v1"
    scan = read_set(SET.read_text(encoding="utf-8"))
    assert not scan.skipped and len(scan.items) == 336 and sum(it.n for it in scan.items) == 1680
    assert Counter(it.n for it in scan.items) == {n: 48 for n in range(2, 9)}
    mix = Counter((it.n, it.source) for it in scan.items)
    assert all(mix[(n, "mmlu")] == 24 and mix[(n, "banking77")] == 24 for n in (2, 3, 4))
    assert all(mix[(n, "banking77")] == 48 for n in range(5, 9))
    assert all(it.label is not None and it.kind == "plain" for it in scan.items)


def test_every_row_comes_from_mmlu_validation_or_banking77_train_at_the_pinned_revisions():
    for rec in _records(SET) + _records(HELDOUT):
        m = rec["_meta"]
        assert (m["source"], m["repo"], m["revision"], m["split"]) in {
            ("mmlu", "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "validation"),
            ("banking77", "PolyAI-LDN/task-specific-datasets", "57ec275d8078af65b7731c2a98be812d844a6d6b", "train")}


def test_no_builtin_row_overlaps_the_heldout_remainder_by_identity_or_text():
    built, held = _records(SET), _records(HELDOUT)
    ident = {(r["_meta"]["source"], r["_meta"]["row"]) for r in built}
    assert not ident & {(r["_meta"]["source"], r["_meta"]["row"]) for r in held}
    def text(r):
        return b.norm(json.dumps(r["state"], sort_keys=True))

    assert not {text(r) for r in built} & {text(r) for r in held}
    assert Counter((r["_meta"]["source"], r["_meta"]["n"]) for r in held) == {
        **{("mmlu", n): 48 for n in (2, 3, 4)}, **{("banking77", n): 48 for n in range(2, 9)}}


def _content(rec):
    """One hash per question, of its CONTENT, not its labels or provenance: the whitespace-normalised, case-folded
    state, instructions and sorted option texts (an option's text is its description, or its key when it has none)."""
    state = rec["state"] if isinstance(rec["state"], str) else json.dumps(rec["state"], sort_keys=True)
    out = set()
    for q in rec["questions"].values():
        crit = q.get("criteria") or {}
        texts = ([v if isinstance(v, str) and v.strip() else k.replace("_", " ") for k, v in crit.items()]
                 if isinstance(crit, dict) else [json.dumps(c, sort_keys=True) for c in crit])   # a score's list
        options = sorted(b.norm(t) for t in texts)
        out.add(hashlib.sha256(json.dumps([b.norm(state), b.norm(q.get("instructions", "")), options]).encode())
                .hexdigest())
    return out


def _state(rec):
    """The whitespace-normalised, case-folded state alone."""
    return b.norm(rec["state"] if isinstance(rec["state"], str) else json.dumps(rec["state"], sort_keys=True))


def _leaks(built, test):
    """The ids of the built-in questions whose content, normalised state alone (a copy truncated to fewer options
    hashes differently), or upstream identity (repo, split, row) is in `test`."""
    test_ids = {(r["_meta"].get("repo"), r["_meta"].get("split"), r["_meta"].get("row")) for r in test}
    test_content = set().union(*map(_content, test))
    test_states = set(map(_state, test))
    return {r["_meta"]["id"] for r in built
            if (r["_meta"]["repo"], r["_meta"]["split"], r["_meta"]["row"]) in test_ids or _content(r) & test_content
            or _state(r) in test_states}


def test_the_leak_check_sees_content_through_new_provenance_labels_order_and_spacing():
    test = [{"state": "Where is my card?", "questions": {"intent": {"type": "choice", "instructions": "Which intent?",
                                                                   "criteria": {"card arrival": None, "pin blocked": None},
                                                                   "label": "card arrival"}},
             "_meta": {"repo": "PolyAI-LDN/task-specific-datasets", "split": "test", "row": 5}}]
    leak = {"state": "  where IS my\tcard? ", "questions": {"q": {"type": "choice", "instructions": "which intent?",
                                                                 "criteria": {"pin_blocked": None, "card_arrival": None},
                                                                 "label": "pin blocked"}},
            "_meta": {"id": "leak", "repo": "PolyAI-LDN/task-specific-datasets", "split": "train", "row": 9}}
    assert _leaks([leak], test) == {"leak"}
    leak["state"] = "Where is my new card?"
    assert _leaks([leak], test) == set()
    assert _leaks([{**leak, "_meta": {**leak["_meta"], "split": "test", "row": 5}}], test) == {"leak"}
    test[0]["questions"]["intent"]["criteria"]["top up failed"] = None                # the test question has n = 3
    two = {**leak, "state": "where is my card?"}                                     # a truncated n = 2 copy of it
    assert not _content(two) & _content(test[0]) and _leaks([two], test) == {"leak"}


def test_no_builtin_row_overlaps_transfer_v4_test():
    if not TRANSFER_TEST.is_file():
        pytest.skip(f"{TRANSFER_TEST} is missing (CALLO_BENCH_DATA unset, or not this box; the split check above "
                    "already rules out any upstream test row)")
    test = _records(TRANSFER_TEST)
    assert not _leaks(_records(SET), test)
    planted = {**test[0], "_meta": {**test[0]["_meta"], "id": "planted", "split": "validation", "row": -1}}
    assert _leaks([planted], test) == {"planted"}                     # the check can fail on a real test row
    (q,) = planted["questions"].values()
    two = {**planted, "questions": {"q": {**q, "criteria": dict(list(q["criteria"].items())[:2])}}}
    assert not _content(two) & set().union(*map(_content, test)) and _leaks([two], test) == {"planted"}  # truncated


def test_rotation_zero_contract_on_every_builtin_question():
    assert len(_contract(_records(SET)).items) == 336


def test_the_manifest_pins_the_sources_and_the_outputs():
    m = json.loads((ROOT / "bench" / "data" / "bias_set_manifest.json").read_text())
    assert m["sources"]["mmlu"]["sha256"] == b.MMLU["sha256"] and m["salt"] == b.SALT
    assert m["outputs"]["bench/data/bias_set.jsonl"] == hashlib.sha256(SET.read_bytes()).hexdigest() == BUILTIN_SET_SHA256
    assert m["outputs"]["bench/data/bias_heldout_v1.jsonl"] == hashlib.sha256(HELDOUT.read_bytes()).hexdigest()


# ---- licences ---------------------------------------------------------------------------------------------

def test_notice_carries_both_licences():
    text = (ROOT / "NOTICE").read_text()
    assert "MIT License" in text and "Copyright (c) 2020 Dan Hendrycks" in text
    assert "CC-BY-4.0" in text and "https://creativecommons.org/licenses/by/4.0/" in text
    assert "modified: distractors sampled, intent names reformatted, packaged as System One requests" in text


def test_notice_says_the_package_is_apache_and_names_where_the_data_lives():
    text = (ROOT / "NOTICE").read_text()
    opening = text.split("=" * 20)[0]
    assert "is Apache-2.0 and contains no third-party data" in opening and '"Apache-2.0".' in opening
    assert "AND MIT" not in text and "src/callo/data" not in text
    for f in ("bench/data/bias_set.jsonl", "bench/data/bias_heldout_v1.jsonl", "bench/data/wide_items.jsonl",
              "peculiar-ragdoll/callo-bias-set"):
        assert f in opening


def test_the_dataset_card_pins_the_set_and_copies_both_licence_sections_from_notice():
    card = (ROOT / "bench" / "data" / "callo-bias-set.README.md").read_text()
    front = card.split("---")[1]
    assert "license: other" in front and "license_name: mit-and-cc-by-4.0" in front and "pretty_name:" in front
    assert "task_categories:" in front and "size_categories:" in front
    assert "viewer: false" in front and "configs:" not in front
    assert re.search(r"^license_link: https://\S+$", front, re.M)
    assert BUILTIN_SET_SHA256 in card and "<REPO-COMMIT" not in card
    # the build script at the pushed commit that produced these exact bytes (it then wrote src/callo/data/...)
    assert "https://github.com/peculiar-ragdoll/callosum/blob/91fb146/bench/build_bias_set.py" in card
    assert "src/callo/data/bias_set.jsonl" in card
    sections = (ROOT / "NOTICE").read_text().split("=" * 117)
    assert sections[1].startswith("\nMMLU") and sections[2].startswith("\nBANKING77")
    for section in sections[1:3]:
        assert section.strip("\n") in card


def test_notice_covers_the_wide_items_file():
    text = (ROOT / "NOTICE").read_text()
    assert "bench/data/wide_items.jsonl" in text
    assert "CLINC150" in text and "CC-BY-3.0" in text and "https://creativecommons.org/licenses/by/3.0/" in text
    assert "155b9c710419136e17307b80d0a13e68cd46b4ec" in text  # clinc/clinc_oos revision
    assert "f54121560de48f2852f90be299010d1d6dc612ec" in text  # legacy-datasets/banking77 revision


NETWORK_ERRORS = ("failed to fetch", "failed to download", "dns error", "error sending request", "connection",
                  "network", "no solution found", "failed to resolve", "offline")


def test_the_wheel_and_the_sdist_ship_code_and_notice_but_no_data(tmp_path):
    """The package is Apache-2.0 and ships no third-party data; the built-in set is fetched on first
    use. NOTICE still ships (license-files and the sdist include list)."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    r = subprocess.run(["uv", "build", "--out-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True,
                       env=real_env())                     # uv's own cache: the user's HOME, not the test's
    if r.returncode != 0 and any(m in r.stderr.lower() for m in NETWORK_ERRORS):
        pytest.skip(f"uv build needs hatchling, which a cold uv cache must download: {r.stderr.strip()[-200:]}")
    assert r.returncode == 0, r.stderr
    (wheel,) = tmp_path.glob("*.whl")
    z = zipfile.ZipFile(wheel)
    names = z.namelist()
    assert any(n.endswith(".dist-info/licenses/NOTICE") for n in names)
    assert any(n.endswith(".dist-info/licenses/LICENSE") for n in names)
    assert "callo/bias.py" in names and not [n for n in names if n.endswith((".jsonl", ".parquet", ".csv"))]
    assert not [n for n in names if n.startswith("callo/data/")]
    (meta,) = [n for n in names if n.endswith(".dist-info/METADATA")]
    assert "License-Expression: Apache-2.0" in z.read(meta).decode().splitlines()
    (sdist,) = tmp_path.glob("*.tar.gz")
    members = tarfile.open(sdist).getnames()
    assert any(n.endswith("/NOTICE") for n in members) and any(n.endswith("src/callo/bias.py") for n in members)
    assert not [n for n in members if n.endswith((".jsonl", ".parquet", ".csv"))]
    assert not [n for n in members if "/bench/" in n or "/callo/data/" in n]


def test_bench_importers_follows_imports_through_other_test_modules(tmp_path):
    (tmp_path / "test_direct.py").write_text("from bench import lib\n")
    (tmp_path / "test_dotted.py").write_text("import bench.wide_items as w\n")
    (tmp_path / "test_through.py").write_text("from test_direct import x\n")
    (tmp_path / "helper.py").write_text("def f():\n    from bench.lib import y\n")      # function-local, not a test_
    (tmp_path / "test_via_helper.py").write_text("import helper\n")
    (tmp_path / "test_clean.py").write_text('"""from bench/data, in prose"""\nimport benchmark\nfrom callo import bias\n')
    assert bench_importers(tmp_path) == ["test_direct.py", "test_dotted.py", "test_through.py", "test_via_helper.py"]
    assert "test_cache_delimiters.py" in bench_importers(ROOT / "tests")      # bench only through test_calibrate_bias


def test_the_sdist_test_suite_collects_without_the_bench_package(tmp_path):
    """The sdist ships tests/ but not bench/ (bench/data is not code): conftest leaves out the modules that import
    bench, so the rest collect cleanly, against the sdist's own src/callo."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    r = subprocess.run(["uv", "build", "--sdist", "--offline", "--out-dir", str(tmp_path)], cwd=ROOT,
                       capture_output=True, text=True, env=real_env())
    if r.returncode != 0 and any(m in r.stderr.lower() for m in NETWORK_ERRORS):
        pytest.skip(f"uv build needs its build backend, which a cold uv cache must download: {r.stderr.strip()[-200:]}")
    assert r.returncode == 0, r.stderr
    (sdist,) = tmp_path.glob("*.tar.gz")
    with tarfile.open(sdist) as tar:
        tar.extractall(tmp_path / "x", **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    (root,) = (tmp_path / "x").iterdir()
    assert not (root / "bench").exists() and (root / "tests" / "conftest.py").is_file()
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=root,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1000:]
    assert "error" not in r.stdout.lower().splitlines()[-1]
    assert "tests/test_bias.py::" in r.stdout and "tests/test_bias_set.py" not in r.stdout


def test_the_pypi_long_description_has_only_absolute_links(tmp_path):
    """PyPI cannot resolve a path inside the repository: hatch-fancy-pypi-readme rewrites README.md's relative image
    paths (src=, srcset=) to raw.githubusercontent.com and its relative markdown links to github.com blob URLs, so the
    wheel's METADATA long description keeps no relative link (in-page `#anchors` excepted)."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    r = subprocess.run(["uv", "build", "--out-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True,
                       env=real_env())
    if r.returncode != 0 and any(m in r.stderr.lower() for m in NETWORK_ERRORS):
        pytest.skip(f"uv build needs its build backend, which a cold uv cache must download: {r.stderr.strip()[-200:]}")
    assert r.returncode == 0, r.stderr
    (wheel,) = tmp_path.glob("*.whl")
    z = zipfile.ZipFile(wheel)
    (meta,) = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
    text = z.read(meta).decode()
    head, _, body = text.partition("\n\n")
    assert "Description-Content-Type: text/markdown" in head
    assert "# Callosum: `callo`" in body and "<!-- NOTE" not in body
    assert not re.findall(r'(?:src|srcset)="(?!https?://)[^"]*"', body)
    assert not re.findall(r"\]\((?!https?://|#)[^)]*\)", body)
    assert 'src="https://raw.githubusercontent.com/peculiar-ragdoll/callosum/main/brand/plate@2x.png"' in body
    assert "](https://github.com/peculiar-ragdoll/callosum/blob/main/docs/guide/" in body
