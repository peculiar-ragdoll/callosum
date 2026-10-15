"""Media kinds: byte sniffing that mirrors llama.cpp's mtmd, and detection from `/props`.

mtmd sniffs every `multimodal_data` entry by its BYTES, in this order (`mtmd/mtmd-helper.cpp:372-458` @ 1537a0a):
audio magic, else an stb image, else WebP through ffmpeg, else video through ffprobe/ffmpeg. A request names each
item's kind (the `images` / `audio` / `videos` field or the part type), and callo checks that the bytes agree before
any upstream call: otherwise an "image" could be read as audio, and `Readout.kind`, the per-kind stats and the 501
check would all name the wrong thing.

`sniff_audio` is `is_audio_file` (`mtmd-helper.cpp:301-316`) with one deliberate deviation: an MPEG sync word
whose layer bits are 00 is refused. Those are ADTS AAC (`FF F0`, `F1`, `F8`, `F9`), which miniaudio cannot decode, and
reserved-layer headers (`FF E0`, `E1`, `E8`, `E9`), which are never a valid MP3. The deviation narrows only what the
AUDIO direction accepts: a video or an image is refused on mtmd's unrestricted rule (`_mtmd_audio`), since mtmd would
still route those bytes to its audio decoder and fail upstream with an opaque error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MEDIA_KINDS = ("image", "audio", "video")       # the contract order of a request's media


def sniff_audio(b: bytes) -> str | None:
    """"wav" | "mp3" | "flac" | None, exactly as mtmd decides "this is audio", minus ADTS (module docstring)."""
    if len(b) < 12:
        return None
    if b[0:4] == b"RIFF" and b[8:12] == b"WAVE":
        return "wav"
    if b[0:3] == b"ID3":
        return "mp3"
    if b[0] == 0xFF and (b[1] & 0xE0) == 0xE0 and (b[1] & 0x06) != 0:   # MPEG audio; layer bits 00 = ADTS AAC, refused
        return "mp3"
    if b[0:4] == b"fLaC":
        return "flac"
    return None


def _mtmd_audio(b: bytes) -> bool:
    """mtmd's own `is_audio_file` (`mtmd-helper.cpp:301-316`), unrestricted: ADTS AAC and reserved-layer sync words
    included. What mtmd would hand to its audio decoder, whatever the request called it."""
    if len(b) < 12:
        return False
    return ((b[0:4] == b"RIFF" and b[8:12] == b"WAVE") or b[0:3] == b"ID3"
            or (b[0] == 0xFF and (b[1] & 0xE0) == 0xE0) or b[0:4] == b"fLaC")


# The still-image magics llama.cpp would take before its video path: stb_image's formats (PNG, JPEG, GIF, BMP, PNM,
# PSD, HDR, PIC) and WebP. Best effort: TGA has no magic, so a residual stb decode is possible, and it
# yields a one-frame image, which is harmless.
_STILL_IMAGES: tuple[tuple[str, bytes], ...] = (
    ("PNG", b"\x89PNG\r\n\x1a\n"), ("JPEG", b"\xff\xd8\xff"), ("GIF", b"GIF8"), ("BMP", b"BM"),
    ("PNM", b"P1"), ("PNM", b"P2"), ("PNM", b"P3"), ("PNM", b"P4"), ("PNM", b"P5"), ("PNM", b"P6"),
    ("PSD", b"8BPS"), ("HDR", b"#?RADIANCE"), ("HDR", b"#?RGBE"), ("PIC", b"\x53\x80\xf6\x34"),
)


def sniff_still_image(b: bytes) -> str | None:
    """The still-image format `b` starts with ("PNG", "WebP", ...), or None."""
    if b[0:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "WebP"
    return next((name for name, magic in _STILL_IMAGES if b.startswith(magic)), None)


def bytes_error(kind: str, raw: bytes) -> str | None:
    """Why `raw` cannot be an item of `kind`, or None. The caller prefixes where the item sits."""
    if kind == "audio":
        return None if sniff_audio(raw) else "unsupported audio format; use WAV, MP3 or FLAC"
    if kind == "video":
        return "the bytes are an image/audio file, not a video" if _mtmd_audio(raw) or sniff_still_image(raw) else None
    if kind == "image":
        return "the bytes are an audio file, not an image" if _mtmd_audio(raw) else None
    raise ValueError(f"unknown media kind {kind!r}")


@dataclass(frozen=True)
class MediaInfo:
    """What `/props` says the upstream takes. Every flag is False without a media marker: no media request
    could be checked or rendered without it."""
    vision: bool
    audio: bool
    video: bool
    media_marker: str | None
    note: str

    def supports(self, kind: str) -> bool:
        flags = {"image": self.vision, "audio": self.audio, "video": self.video}
        if kind not in flags:
            raise ValueError(f"unknown media kind {kind!r}")
        return flags[kind]


def detect_media(props: dict[str, Any]) -> MediaInfo:
    """`/props.modalities` {vision, audio, video} and `/props.media_marker` (server-context.cpp:4837-4842). `video`
    is "an MTMD_VIDEO build plus vision": it does not check that ffmpeg is present, which the video self-test does."""
    mods = props.get("modalities")
    marker = props.get("media_marker")
    marker = marker if isinstance(marker, str) and marker else None
    if not isinstance(mods, dict):
        return MediaInfo(False, False, False, None, "upstream reports no modalities")
    vision, audio, video = (mods.get(k) is True for k in ("vision", "audio", "video"))
    if not (vision or audio or video):
        return MediaInfo(False, False, False, marker, "upstream reports no media modality")
    on = ", ".join(f"{k}: {'on' if v else 'off'}" for k, v in (("vision", vision), ("audio", audio), ("video", video)))
    if marker is None:
        return MediaInfo(False, False, False, None, f"upstream reports {on} but no media_marker; media input is off")
    return MediaInfo(vision, audio, video, marker, on)
