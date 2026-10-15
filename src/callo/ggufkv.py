"""A GGUF's key-value header, read without interpreting its tensors.

gguf-py's GGUFReader builds a typed view of every tensor while it opens a file, and a tensor type it does not know (a
quantization newer than the installed gguf-py) raises ValueError there. callo reads only header keys and, for the model
key, each tensor's name, shape, type id and a sample of its bytes, so it opens GGUFs through `open_kv`: the key-value
store and the tensor-info table parsed exactly as GGUFReader parses them, and no tensor view built.

`open_kv` overrides GGUFReader's private `_build_tensors`. A gguf-py release that renames it, or stops calling it, is a
clear InvalidInput, never a silent return to the crash. A tensor of an unknown type is sized up to the next tensor's
data, so a model key over such a file changes once an upgraded gguf-py knows the type (the profiles keyed to it must
be recalibrated).

The override keeps the checks gguf-py's tensor views made, as ValueError while the file opens: a duplicated tensor
name, a quantized row that is not a whole number of blocks, and a tensor whose data runs past the end of the file (a
truncated or partly downloaded GGUF). For an unknown type only the start of its data can be checked: a file cut
inside that tensor's data is not noticed."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InvalidInput

_CLASSES: dict[type, type] = {}
_HOOK = "_build_tensors"


@dataclass(frozen=True)
class TensorInfo:
    """One tensor-info entry, for a type gguf-py knows or not."""
    name: str
    shape: tuple[int, ...]      # as stored (gguf-py's ReaderTensor.shape)
    type_id: int                # the raw ggml type id
    data_offset: int            # absolute, in the file
    n_bytes: int                # gguf-py's n_elems x type_size // block_size; an unknown type: up to the next data


def _kv_class(base: type) -> type:
    cls = _CLASSES.get(base)
    if cls is None:
        class KVReader(base):  # type: ignore[misc, valid-type]
            def _build_tensors(self, start_offs: int, fields: list) -> None:
                self.tensor_fields, self.tensors_start, self.tensors = list(fields), int(start_offs), []
                tensor_infos(self)                      # gguf-py's checks: refuse a malformed file while it opens

        cls = _CLASSES[base] = KVReader
    return cls


def open_kv(path: str | Path) -> Any:
    """A GGUFReader over `path` whose `.fields`, `.get_field` and `.data` are gguf-py's, whose `.tensors` is empty,
    and whose raw tensor-info table is `.tensor_fields` (read it with `tensor_infos`). It raises what GGUFReader raises
    on a file it cannot parse (no magic, a bad version, a malformed key), and InvalidInput when the installed gguf-py
    no longer has, or no longer calls, the `_build_tensors` hook this reader overrides."""
    import gguf  # noqa: PLC0415 -- keep the module importable without gguf

    base = gguf.GGUFReader
    if not callable(getattr(base, _HOOK, None)):
        raise InvalidInput(f"gguf-py {getattr(gguf, '__version__', '?')} has no GGUFReader.{_HOOK}, which callo's "
                           f"header reader overrides; install a gguf-py callo supports (pyproject's range)")
    reader = _kv_class(base)(str(path), "r")
    if not hasattr(reader, "tensor_fields"):
        raise InvalidInput(f"gguf-py {getattr(gguf, '__version__', '?')} no longer calls GGUFReader.{_HOOK}, which "
                           f"callo's header reader overrides; install a gguf-py callo supports (pyproject's range)")
    return reader


def tensor_infos(reader: Any) -> list[TensorInfo]:
    """The tensor-info table in file order. A known type's n_bytes is gguf-py's own, so a model key over known types
    is unchanged; an unknown type's tensor spans up to the next tensor's data (in offset order) or the end of the file.
    ValueError, naming the tensor, on a duplicated name, a quantized row that is not a whole number of blocks, or data
    that starts (an unknown type) or ends (a known one) past the end of the file."""
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType  # noqa: PLC0415

    raw, seen = [], set()
    for f in reader.tensor_fields:
        _len, name, _ndims, dims, dtype, offset = f.parts
        name = str(bytes(name), encoding="utf-8")
        if name in seen:
            raise ValueError(f"Found duplicated tensor with name {name}")
        seen.add(name)
        raw.append((name, tuple(int(d) for d in dims), int(dtype[0]), reader.tensors_start + int(offset[0])))
    starts = sorted({at for *_, at in raw})
    size_of_file = len(reader.data)
    out = []
    for name, shape, type_id, at in raw:
        try:
            ggml_type = GGMLQuantizationType(type_id)
            block, size = GGML_QUANT_SIZES[ggml_type]
        except (ValueError, KeyError):
            if at >= size_of_file:
                raise ValueError(f"tensor {name} (type {type_id}) starts its data at {at}, past the end of the "
                                 f"{size_of_file}-byte file: truncated") from None
            later = [s for s in starts if s > at]
            n_bytes = (later[0] if later else size_of_file) - at
        else:
            if shape and shape[0] % block:
                raise ValueError(f"tensor {name}: a row of {shape[0]} is not a multiple of the {ggml_type.name} "
                                 f"block of {block}")
            n_elems = 1
            for d in shape:
                n_elems *= d
            n_bytes = n_elems * size // block
            if at + n_bytes > size_of_file:
                raise ValueError(f"tensor {name}: its {n_bytes} bytes at {at} end past the end of the "
                                 f"{size_of_file}-byte file: truncated")
        out.append(TensorInfo(name, shape, type_id, at, n_bytes))
    return out
