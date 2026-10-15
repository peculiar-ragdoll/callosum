"""The position-bias profile, format version 2.

`prior` per option count is the only stored correction; the offsets are derived on load, so a hand-edited file
cannot be internally inconsistent. Loading has two phases:

- **Phase 1** (`load_profile`), with the other flag checks, BEFORE anything starts: readable, JSON, version 2 (a
  version 1 file, fitted with the withdrawn PriDe estimator, is refused with its own message), not fitted on the
  built-in set (`set.builtin`), a well-formed `model_key`, entry keys 2..26, each prior of length n, finite,
  positive and summing to 1 within 1e-6, `set.kinds` a non-empty subset of the known kinds. Any failure:
  `InvalidInput` (exit 1), and no child is spawned.
- **Phase 2** (`check_profile`), after `resolve_gguf`/`load_gguf_meta`: another model or a native decision-model
  upstream refuses (exit 1); a differing provenance field, a busy profile or empty entries only warn.

`provenance` records what can move a readout without changing the model: the public prompt contract, the chat
template actually served, the llama.cpp build, the projector, the user-turn delimiter, `--cache`,
whether `--cache` readouts send that delimiter (`cache_delimiter`, present only when true) and, in launch mode, the
passthrough llama-server flags (`-ctk`/`-ctv`, `--lora`, `--override-kv`, ...).

Wide-choice: `provenance` also writes `wide_contract` (the label table's sha256, WIDE_TOP_K and
the split settings, plus `uncovered: "two_pass"` on a model whose vocabulary does not carry every label as one token,
whose labels past the coverage Callo 0.3.1 reads in two passes where 0.3.0 refused them). It is NOT a
PROVENANCE_FIELD: its guard (`wide_contract_warning`) is asymmetric, so a profile written before it (no field) stays
valid and silent.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bias import MAX_OPTIONS, MIN_OPTIONS, offsets_from_prior
from .engine import READOUT_KINDS, WIDE_TOP_K
from .errors import CalloError, InvalidInput
from .marginalize import WIDE_MAX_SPLIT_PASSES, WIDE_SPLIT_READ
from .modelkey import KEY_RE
from .prompt import CONTEXT_JOINER, SYSTEM_MAIN, user_content
from .systemone import WIDE_LABELS_SHA256, parse_request

FORMAT = 2
ESTIMATOR = "meanP"            # recorded in every profile; FORMAT is the contract that loading checks
PROVENANCE_FIELDS = ("prompt_sha256", "chat_template_sha256", "build_info", "mmproj", "user_delimiter",
                     "cache_prompt", "cache_delimiter", "server_args")
SKIPPED_STATES = ("no_entry", "noncanonical", "score", "single_option", "kind_not_calibrated", "deliberate", "wide")
# what moves a wide readout without changing the model or the narrow prompt contract
WIDE_CONTRACT = {"labels_sha256": WIDE_LABELS_SHA256, "top_k": WIDE_TOP_K, "split_read": WIDE_SPLIT_READ,
                 "max_split_passes": WIDE_MAX_SPLIT_PASSES}
# added to the contract only where it changes a readout: a vocabulary with labels that are not single tokens, read in
# two passes (one more split pass per first letter leading one). A full-coverage model's contract is unchanged, so its
# profiles from before and after mix; a short-coverage model's differ, so they never mix silently.
WIDE_UNCOVERED = {"uncovered": "two_pass"}
_SUM_TOL = 1e-6
_ENTRY_KEY = re.compile(r"[1-9][0-9]*")         # ASCII only: str.isdigit() also accepts "²", which int() refuses


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_sha256() -> str:
    """The public prompt contract: SYSTEM_MAIN, the context joiner and the user-turn layout."""
    q = parse_request({"state": "s", "questions": {"q": {"type": "choice", "instructions": "i",
                                                         "criteria": {"x": None, "y": None}}}}).questions[0]
    return _sha("\0".join([SYSTEM_MAIN, CONTEXT_JOINER, user_content("s", q)]))


def provenance(*, chat_template: str | None, build_info: str | None, mmproj: str | None,
               user_delimiter: str | None, cache_prompt: bool, server_args: Sequence[str] | None = None,
               wide_two_pass: bool = False) -> dict:
    """`server_args`: launch mode's passthrough argv after `--` (KV cache types, LoRA, overrides move readouts too);
    None in attach mode, where callo cannot see how llama-server was started. `cache_delimiter` (present, and true,
    only when `--cache` is on AND there is a user-turn delimiter): the plain readouts send the delimiter too, so a hybrid
    model reuses KV from the user-turn checkpoint. Absent otherwise, so a cache-off provenance is unchanged and
    a `--cache` profile measured before the change reads as different (None vs True). `wide_contract` is always
    written; it is not a PROVENANCE_FIELD (see `wide_contract_warning`). `wide_two_pass`: the served vocabulary has
    labels that are not single tokens and reads them in two passes (`SurfaceSets.wide_readable` above
    `wide_coverage`), recorded as `WIDE_UNCOVERED` in the contract."""
    out = {"prompt_sha256": prompt_sha256(),
           "chat_template_sha256": _sha(chat_template) if chat_template is not None else None,
           "build_info": build_info, "mmproj": Path(mmproj).name if mmproj else None,
           "user_delimiter": user_delimiter, "cache_prompt": bool(cache_prompt),
           "server_args": list(server_args) if server_args is not None else None}
    if cache_prompt and user_delimiter:
        out["cache_delimiter"] = True
    out["wide_contract"] = {**WIDE_CONTRACT, **(WIDE_UNCOVERED if wide_two_pass else {})}
    return out


def resume_provenance(recorded: Any, current: Any) -> tuple[Any, Any]:
    """The provenance pair a calibrate `--resume` compares, relaxed ONE way: a recorded
    (raw-file) provenance without `wide_contract` was written before the field existed, so the field is ignored on
    both sides and an old raw file resumes against a new server. A recorded `wide_contract` is compared as is: against
    a server without one (whose coverage would drop wide questions the file has rows for) or a different one, the
    resume is refused."""
    if isinstance(recorded, Mapping) and isinstance(current, Mapping) and recorded.get("wide_contract") is None:
        return ({k: v for k, v in recorded.items() if k != "wide_contract"},
                {k: v for k, v in current.items() if k != "wide_contract"})
    return recorded, current


def wide_contract_warning(then: Any, now: Any, who: str) -> str | None:
    """The asymmetric guard of `wide_contract`: a warning only when BOTH sides carry it and they
    differ. Absent on either side is a legacy profile or server, so every profile written before the field stays
    silent. `who` names the flag (`--position-bias`, `--think-profile`)."""
    if then is None or now is None or then == now:
        return None
    return (f"WARNING: {who}: wide_contract differs from the profile's (profile {then!r}, now {now!r}); readouts of "
            f"questions above 26 options may not compare")


@dataclass(frozen=True)
class PositionBias:
    source: str                                 # the file path, or "<dict>"
    sha256: str                                 # of the file bytes (or of the canonical JSON of a dict)
    model_key: str
    model_name: str | None
    priors: dict[int, tuple[float, ...]]
    offsets: dict[int, tuple[float, ...]]       # derived: the centred log prior
    kinds: frozenset[str]                       # the readout kinds the set contained: only these are corrected
    state: str | None                           # server_state.state: "clean" | "busy"
    provenance: dict

    def info(self) -> dict:
        """The /callo/stats `position_bias` block, before its counters."""
        return {"profile": self.source, "sha256": self.sha256, "entries": sorted(self.priors),
                "kinds": sorted(self.kinds), "state": self.state}


def _number(v: Any) -> bool:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v) and v > 0
    except OverflowError:                       # an int beyond float range, e.g. 10**400
        return False


def load_profile(src: str | Path | Mapping) -> PositionBias:
    """Phase 1. Raises InvalidInput naming the source."""
    if isinstance(src, Mapping):
        data: Any = dict(src)
        source = "<dict>"
        try:
            raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        except (TypeError, ValueError) as e:    # a set value, mixed-type keys, a circular reference
            raise InvalidInput(f"--position-bias {source}: not JSON-serialisable ({e})") from None
    else:
        source = str(src)
        try:
            raw = Path(src).read_bytes()
        except OSError as e:
            raise InvalidInput(f"--position-bias {source}: cannot read it ({e})") from None
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise InvalidInput(f"--position-bias {source}: not valid JSON ({e})") from None

    def bad(why: str) -> InvalidInput:
        return InvalidInput(f"--position-bias {source}: {why}")

    if not isinstance(data, dict):
        raise bad("not a JSON object")
    version = data.get("callo_position_bias")
    if type(version) is int and version == 1:
        raise bad("format 1 was fitted with the PriDe estimator, withdrawn because it miscorrected "
                  "held-out readouts; measure again with `callo calibrate --bias --data YOUR.jsonl --out PROFILE`")
    if type(version) is not int or version != FORMAT:
        raise bad(f"callo_position_bias must be {FORMAT}, got {version!r}")
    if isinstance(data.get("set"), dict) and data["set"].get("builtin") is True:
        raise bad("it was fitted on the built-in set, which is a diagnostic only: a correction measured "
                  "there does not transfer; fit one on your own questions with `callo calibrate --bias --data "
                  "YOUR.jsonl --out PROFILE`")
    key = data.get("model_key")
    if not isinstance(key, str) or not KEY_RE.fullmatch(key):
        raise bad(f"model_key must be 'sha256:' and 64 hex digits, got {key!r}")
    entries = data.get("entries")
    if not isinstance(entries, dict):
        raise bad("entries must be an object")
    priors: dict[int, tuple[float, ...]] = {}
    for k, entry in entries.items():
        if not (isinstance(k, str) and _ENTRY_KEY.fullmatch(k) and MIN_OPTIONS <= int(k) <= MAX_OPTIONS):
            raise bad(f"entry key {k!r} must be an option count from {MIN_OPTIONS} to {MAX_OPTIONS}")
        n = int(k)
        prior = entry.get("prior") if isinstance(entry, dict) else None
        if not isinstance(prior, list):
            raise bad(f"entry {n} must be an object with a prior")
        if len(prior) != n:
            raise bad(f"the prior of entry {n} must have {n} values, got {len(prior)}")
        if not all(_number(p) for p in prior):
            raise bad(f"the prior of entry {n} must be finite and positive, got {prior}")
        if abs(math.fsum(prior) - 1.0) > _SUM_TOL:
            raise bad(f"the prior of entry {n} must sum to 1 within {_SUM_TOL}, got {math.fsum(prior)!r}")
        priors[n] = tuple(float(p) for p in prior)
    kinds = (data.get("set") or {}).get("kinds") if isinstance(data.get("set"), dict) else None
    if not (isinstance(kinds, list) and kinds and all(isinstance(x, str) and x in READOUT_KINDS for x in kinds)):
        raise bad(f"set.kinds must be a non-empty list of {', '.join(READOUT_KINDS)}, got {kinds!r}")
    state = (data.get("server_state") or {}).get("state") if isinstance(data.get("server_state"), dict) else None
    prov = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
    name = data.get("model_name") if isinstance(data.get("model_name"), str) else None
    return PositionBias(source, hashlib.sha256(raw).hexdigest(), key, name, dict(sorted(priors.items())),
                        {n: offsets_from_prior(p) for n, p in sorted(priors.items())}, frozenset(kinds),
                        state if isinstance(state, str) else None, prov)


def check_identity(pb: PositionBias, *, model_key: str, native: bool) -> None:
    """The refusing half of phase 2: a native decision model or another model (CalloError, exit 1). Needs only the
    GGUF, so `serve` runs it before its startup probe touches any slot."""
    if native:
        raise CalloError(f"--position-bias {pb.source}: the upstream is a native decision model; callo's engine is "
                         f"off, so there is no readout of callo's to correct")
    if pb.model_key != model_key:
        raise CalloError(f"--position-bias {pb.source}: this profile was measured on {pb.model_name or 'another model'} "
                         f"({pb.model_key}), not on the served model ({model_key}). Measure this model with "
                         f"`callo calibrate --bias`")


def check_profile(pb: PositionBias, *, model_key: str, native: bool, provenance: Mapping | None) -> list[str]:
    """Phase 2: refuse another model or a native decision model (CalloError, exit 1, `check_identity`); warn
    on everything else that may make the prior stale. Returns the warning lines, each starting "WARNING:"."""
    check_identity(pb, model_key=model_key, native=native)
    lines = []
    for f in PROVENANCE_FIELDS:
        then, now = pb.provenance.get(f), (provenance or {}).get(f)
        if then != now:
            lines.append(f"WARNING: --position-bias: {f} differs from the profile's (profile {then!r}, now {now!r}); "
                         f"the prior may not transfer")
    wide = wide_contract_warning(pb.provenance.get("wide_contract"), (provenance or {}).get("wide_contract"),
                                 "--position-bias")
    if wide is not None:
        lines.append(wide)
    if pb.state == "busy":
        lines.append("WARNING: --position-bias: the profile was measured on a busy server (server_state busy, "
                     "calibrate --allow-busy): other slots' state may be part of the prior")
    if not pb.priors:
        lines.append("WARNING: --position-bias: the profile has no entries; nothing will be corrected")
    return lines
