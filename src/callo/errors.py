"""The errors callo raises DELIBERATELY.

The CLI turns a `CalloError` into one line on stderr and exit 1, and lets everything else surface as a
traceback. An exception type the package chooses to raise is a diagnosis; a `ValueError` from the middle of
the marginaliser is a bug, and a bug's stack is the useful part. `InvalidInput` also subclasses `ValueError`,
so callers that already catch `ValueError` keep catching it.
"""
from __future__ import annotations


class CalloError(Exception):
    """Base of every error this package raises on purpose."""


class InvalidInput(CalloError, ValueError):
    """Something the caller passed is not usable: a label with no surface form, a nan logprob, an unreadable GGUF."""


NO_VISION_DETAIL = "the upstream llama-server has no vision support; start it with --mmproj"
NO_AUDIO_DETAIL = "the upstream llama-server has no audio support; start it with an audio-capable --mmproj"
NO_VIDEO_DETAIL = ("the upstream llama-server has no video support; it needs a vision --mmproj on a build with video "
                   "(MTMD_VIDEO), and ffmpeg/ffprobe on its PATH")
# The upstream reports video, but serve's video self-test clip failed to load there, so callo turned video off.
VIDEO_OFF_DETAIL = ("video was turned off at startup: the upstream llama-server reports video, but callo's "
                    "video self-test clip failed to load there (ffprobe or ffmpeg missing from its PATH or "
                    "--video-ffmpeg-dir, or ffprobe could not read the clip); see callo's startup log")


class NoMedia(CalloError):
    """A media request reached an upstream that does not take that kind: HTTP 501."""


class NoVision(NoMedia):
    """An image request reached an upstream without vision: HTTP 501, as native (server-context.cpp:5432-5437)."""

    def __init__(self, detail: str = NO_VISION_DETAIL):
        super().__init__(detail)


class NoAudio(NoMedia):
    """An audio request reached an upstream without audio: HTTP 501."""

    def __init__(self, detail: str = NO_AUDIO_DETAIL):
        super().__init__(detail)


class NoVideo(NoMedia):
    """A video request reached an upstream without video, or whose video self-test could not load its clip: HTTP
    501."""

    def __init__(self, detail: str = NO_VIDEO_DETAIL):
        super().__init__(detail)


class ReadoutError(CalloError):
    """The readout could not be made, e.g. the chat template did not render the image parts: HTTP 500."""


class MediaRejected(CalloError):
    """The upstream could not decode an image, audio clip or video, or refused a data URL's shape: HTTP 422 (the
    request's fault)."""


ImageRejected = MediaRejected   # the name, kept for importers


class MediaLoadFailed(MediaRejected):
    """The upstream's "Failed to load image or audio file": still a 422, and the one outcome of the video
    self-test that turns video off."""


class MediaTooLarge(CalloError):
    """A media request whose prompt, media tokens included, exceeds the upstream's context: HTTP 413.
    Deterministic for that request, so never a status a client retries."""


class UpstreamUnavailable(CalloError):
    """The upstream is unreachable, or answered with a 5xx (any HTTP error on the text path): HTTP 503."""


class UpstreamRejected(CalloError):
    """The upstream answered an image request with another 4xx: HTTP 500, never a 4xx the caller would drop."""


class ThinkTooLarge(CalloError):
    """A deliberate question's prompt plus its think budget does not fit the slot context, or llama-server truncated
    the trace: HTTP 413, deterministic for that request."""


class ThinkTimeout(CalloError):
    """The deliberate generation exceeded its read timeout, --s1-timeout + 0.1 s per budget token: HTTP 504, a
    5xx Kev retries, whose status names the cause."""


class TooManyOptions(CalloError):
    """A question with more than 26 options and more than this model's `wide_readable` (how many leading two-letter
    labels of `systemone.WIDE_LABELS` the engine can read on its vocabulary: one token each, or two passes when each
    letter is a token): HTTP 422. A `CalloError`, not an `InvalidRequest`, because the engine raises it
    (`check_request`), where an `InvalidRequest` would map to 500; deterministic for that request and model."""


class Cancelled(CalloError):
    """The client left during a deliberate request and callo aborted the generation: the request ends with no
    reply."""
