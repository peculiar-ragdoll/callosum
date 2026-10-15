"""The T1 utterance set: 20 fixed English sentences (19 Harvard sentences, IEEE 1969,
public domain, and the "quick brown fox" pangram) spoken by macOS `say` in the four installed voices Samantha, Daniel,
Karen and Moira, converted by `afconvert` to 24 kHz mono 16-bit WAVs, with a manifest. Run once, on the CPU, before the
GPU window:

    uv run --frozen python -m bench.build_asr_set OUT

The synthesiser is a seam (`synth(text, voice) -> WAV bytes`): the tests pass a stdlib tone, so nothing here runs
`say` under pytest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

SENTENCES = (
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It is easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The small pup gnawed a hole in the sock.",
    "The fish twisted and turned on the bent hook.",
    "The quick brown fox jumps over the lazy dog.",
)
VOICES = ("Samantha", "Daniel", "Karen", "Moira")   # installed on macOS (`say -v '?'`)
RATE = 24000
TIMEOUT = 60.0                      # seconds for each `say` / `afconvert` call


def build(out: Path, synth: Callable[[str, str], bytes], *, about: dict) -> dict:
    """Write `u01.wav` .. `u20.wav` (the voices in turn) and `manifest.json`: items (id, file, text, voice, sha256), the
    rate, and `about` (the synthesiser and its version)."""
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for i, text in enumerate(SENTENCES, 1):
        voice = VOICES[(i - 1) % len(VOICES)]
        wav = synth(text, voice)
        name = f"u{i:02d}.wav"
        (out / name).write_bytes(wav)
        items.append({"id": f"u{i:02d}", "file": name, "text": text, "voice": voice,
                      "sha256": hashlib.sha256(wav).hexdigest()})
    manifest = {"items": items, "rate": RATE, **about}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def say_synth() -> tuple[Callable[[str, str], bytes], dict]:
    """macOS `say` to AIFF, then `afconvert` to 24 kHz mono 16-bit little-endian WAV (a file in a temporary directory,
    removed with it)."""
    def run(argv: list[str]) -> None:
        try:
            subprocess.run(argv, check=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"`{argv[0]}` did not finish within {TIMEOUT:g} s ({' '.join(argv[:3])} ...)") from e

    def synth(text: str, voice: str) -> bytes:
        with tempfile.TemporaryDirectory() as d:
            aiff, wav = Path(d) / "u.aiff", Path(d) / "u.wav"
            run(["say", "-v", voice, "-o", str(aiff), text])
            run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1", str(aiff), str(wav)])
            return wav.read_bytes()

    return synth, {"synth": "say", "voices": list(VOICES), "macos": platform.mac_ver()[0]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_asr_set", description="The 20-utterance T1 set.")
    ap.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    synth, about = say_synth()
    try:
        m = build(a.out, synth, about=about)
    except RuntimeError as e:
        print(f"build_asr_set: {e}", file=sys.stderr)
        return 1
    print(f"{len(m['items'])} utterances in {a.out} ({about})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
