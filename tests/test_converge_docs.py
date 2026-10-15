"""The public docs serve early exit: every flag, field, close word, X-Callo-Timings field and stats block is
documented, nothing still says it is not served, early exit is opt-in (listed, never picked, with the held-out result
that made it so), and no process artifact (a spec section, a review or gate code, the
word "stub" in the early-exit text) reaches README.md or docs/guide."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = [ROOT / "README.md", *sorted((ROOT / "docs" / "guide").glob("*.md"))]
ARTIFACT = re.compile(r"§|\bTD\b|\bTB\b|\bU-D\d|\bS-E\d|\b[EGCRK]\d{1,2}\b|measured, not served|not served yet")


def test_no_process_artifacts_in_the_public_docs():
    for path in PUBLIC:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert not ARTIFACT.search(line), f"{path.name}:{i}: {line}"


def test_the_early_exit_section_documents_the_surface():
    think = (ROOT / "docs" / "guide" / "think-budgets.md").read_text(encoding="utf-8")
    section = think[think.index("## Early exit"):]
    assert "stub" not in section.lower()
    for text in ("--think-converge Z", "\"think_converge\": 0.95", "\"think_converge\": 0", "\"think_budget\": 0",
                 "--converge-probe K", "`converged`", "`converge`", "`converge_rule`", "`probes`",
                 "`unreachable` | `not_chunkable` | `media`", "/callo/stats.think.converge", "`tokens_saved`",
                 "`calibrated: true`", "a Z between two targets takes the next higher target's rule",
                 "A Z at or below .9 takes the .9 rule. That is conservative",
                 "would put the model on your accuracy/latency front, net of probe cost"):
        assert text in " ".join(section.split()) or text in section, text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--think-converge 0.95" in readme and "think_converge" in readme
    assert "`--think-converge Z`" in (ROOT / "docs" / "guide" / "serving.md").read_text(encoding="utf-8")
    assert "think_converge" in (ROOT / "docs" / "guide" / "api.md").read_text(encoding="utf-8")
    assert "think.converge" in (ROOT / "docs" / "guide" / "stats.md").read_text(encoding="utf-8")
    assert "--think-converge" in (ROOT / "docs" / "guide" / "exit-codes.md").read_text(encoding="utf-8")
    compare = (ROOT / "docs" / "guide" / "compare.md").read_text(encoding="utf-8")
    calibration = (ROOT / "docs" / "guide" / "calibration.md").read_text(encoding="utf-8")
    assert "--think-converge Z" in compare and "### Convergence and early exit" in calibration
    for doc in (compare, calibration, section):
        flat = " ".join(doc.split())
        assert "never picked" in flat and "early exit (opt-in, not picked): M@ee.95 saves" in flat
        assert "serve with --think-converge 0.95 to opt in" in flat and "joins the pick beside the budget and" not in flat
        assert "early-exit pick" not in flat and "candidates like budget" not in flat
    limitations = (ROOT / "docs" / "guide" / "limitations.md").read_text(encoding="utf-8")
    assert "model_key" in limitations and "gguf-py" in limitations                     # the model_key note
    assert "which is why it is opt-in" in " ".join(limitations.split())
    assert "Early exit is opt-in" in readme and "never pick it" in " ".join(readme.split())


def test_the_early_exit_section_says_why_it_is_opt_in():
    think = (ROOT / "docs" / "guide" / "think-budgets.md").read_text(encoding="utf-8")
    why = " ".join(think[think.index("### Why it is opt-in"):].split("\n### ", 2)[0].split())
    for text in ("Tiel-Coder-35B-A3B Q4_K_XL", "z .95", "one greedy trace per question",
                 "217 of 240 MMLU, PAWS and QNLI questions", "214 of 217 stops", "In a sampled served run",
                 "96 logic-composition questions built from rules absent from the calibration data",
                 "74 % with early exit against 100 % with full thinking", "−6.0 points [−9.8, −2.1]",
                 "median latency on those held-out questions fell by about 1 s",
                 "measured on your calibration data, at your pick statistic",
                 "Calibrate on data that looks like your traffic, and opt in only if it does"):
        assert text in why, text
