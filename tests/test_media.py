"""v0.3 media: the byte sniff that mirrors mtmd, the per-kind byte check, /props detection."""
import pytest

from callo.media import MediaInfo, bytes_error, detect_media, sniff_audio, sniff_still_image

PAD = b"\x00" * 16          # every sniff needs 12 bytes


@pytest.mark.parametrize(("head", "fmt"), [
    (b"RIFF\x24\x00\x00\x00WAVE", "wav"),
    (b"ID3\x04\x00", "mp3"),
    (b"\xff\xfb", "mp3"), (b"\xff\xfa", "mp3"),      # MPEG-1 Layer III
    (b"\xff\xf3", "mp3"), (b"\xff\xf2", "mp3"),      # MPEG-2 Layer III
    (b"\xff\xe3", "mp3"),                            # MPEG-2.5 Layer III
    (b"fLaC", "flac"),
])
def test_sniff_audio_accepts_what_mtmd_decodes(head, fmt):
    assert sniff_audio(head + PAD) == fmt


@pytest.mark.parametrize("head", [
    b"\xff\xf0", b"\xff\xf1", b"\xff\xf8", b"\xff\xf9",   # ADTS AAC: the deliberate deviation
    b"OggS\x00\x02", b"\x00\x00\x00\x20ftypM4A ",          # OGG and M4A are not WAV, MP3 or FLAC
    b"RIFF\x24\x00\x00\x00AVI ", b"\x89PNG\r\n\x1a\n",
])
def test_sniff_audio_refuses_the_rest(head):
    assert sniff_audio(head + PAD) is None


def test_the_mpeg_rule_refuses_exactly_the_layer_00_headers():
    refused = {b for b in range(0xE0, 0x100) if sniff_audio(bytes([0xFF, b]) + PAD) is None}
    assert refused == {0xE0, 0xE1, 0xE8, 0xE9, 0xF0, 0xF1, 0xF8, 0xF9}


@pytest.mark.parametrize("b", [b"", b"RIFF\x00\x00\x00\x00WAV", b"ID3" + b"\x00" * 8, b"fLaC" + b"\x00" * 7])
def test_anything_under_12_bytes_is_not_audio(b):
    assert len(b) < 12 and sniff_audio(b) is None


STILL = [(b"\x89PNG\r\n\x1a\n", "PNG"), (b"\xff\xd8\xff\xe0", "JPEG"), (b"GIF89a", "GIF"), (b"GIF87a", "GIF"),
         (b"BM", "BMP"), (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "WebP"), *((f"P{i}\n".encode(), "PNM") for i in range(1, 7)),
         (b"8BPS", "PSD"), (b"#?RADIANCE\n", "HDR"), (b"#?RGBE\n", "HDR"), (b"\x53\x80\xf6\x34", "PIC")]
AUDIO = [b"RIFF\x24\x00\x00\x00WAVE", b"ID3\x04", b"\xff\xfb\x90", b"fLaC"]
VIDEO = [b"RIFF\x24\x00\x00\x00AVI LIST", b"\x00\x00\x00\x18ftypmp42", b"\x1a\x45\xdf\xa3\x01", b"FLV\x01"]


@pytest.mark.parametrize(("head", "name"), STILL)
def test_every_still_image_magic_is_named_and_refused_as_a_video(head, name):
    assert sniff_still_image(head + PAD) == name
    assert bytes_error("video", head + PAD) == "the bytes are an image/audio file, not a video"


@pytest.mark.parametrize("head", AUDIO)
def test_every_audio_magic_is_refused_as_a_video_and_as_an_image(head):
    assert bytes_error("video", head + PAD) == "the bytes are an image/audio file, not a video"
    assert bytes_error("image", head + PAD) == "the bytes are an audio file, not an image"
    assert bytes_error("audio", head + PAD) is None


@pytest.mark.parametrize("head", VIDEO)
def test_video_containers_pass_the_video_check_and_fail_the_audio_one(head):
    assert bytes_error("video", head + PAD) is None and sniff_still_image(head + PAD) is None
    assert bytes_error("audio", head + PAD) == "unsupported audio format; use WAV, MP3 or FLAC"


def test_images_are_unchanged_unless_they_are_audio():
    assert bytes_error("image", b"\x89PNG\r\n\x1a\n" + PAD) is None
    assert bytes_error("image", b"ABC") is None                 # v0.2 accepted any bytes; mtmd decides
    with pytest.raises(ValueError, match="unknown media kind"):
        bytes_error("text", b"x")


@pytest.mark.parametrize(("props", "want"), [
    ({"modalities": {"vision": True, "audio": True, "video": True}, "media_marker": "<m>"},
     MediaInfo(True, True, True, "<m>", "vision: on, audio: on, video: on")),
    ({"modalities": {"vision": False, "audio": True, "video": False}, "media_marker": "<m>"},   # audio-only
     MediaInfo(False, True, False, "<m>", "vision: off, audio: on, video: off")),
    ({"modalities": {"vision": True, "video": True}, "media_marker": "<m>"},
     MediaInfo(True, False, True, "<m>", "vision: on, audio: off, video: on")),
    ({"modalities": {"vision": False, "audio": False, "video": False}, "media_marker": "<m>"},
     MediaInfo(False, False, False, "<m>", "upstream reports no media modality")),
    ({"modalities": {"audio": "true"}, "media_marker": "<m>"},
     MediaInfo(False, False, False, "<m>", "upstream reports no media modality")),
    ({"modalities": {"audio": True}},
     MediaInfo(False, False, False, None, "upstream reports vision: off, audio: on, video: off but no media_marker; "
                                          "media input is off")),
    ({"modalities": "yes", "media_marker": "<m>"}, MediaInfo(False, False, False, None, "upstream reports no modalities")),
    ({}, MediaInfo(False, False, False, None, "upstream reports no modalities")),
])
def test_detect_media(props, want):
    assert detect_media(props) == want


def test_supports_maps_each_kind_to_its_flag():
    info = MediaInfo(False, True, True, "<m>", "")
    assert [info.supports(k) for k in ("image", "audio", "video")] == [False, True, True]


def test_video_and_image_refuse_everything_mtmd_routes_to_its_audio_decoder():
    """Video and image apply mtmd's UNRESTRICTED `is_audio_file` (mtmd-helper.cpp:301-316), so the
    layer-00 MPEG sync words `sniff_audio` refuses for the audio direction (ADTS AAC and the reserved layer) are still
    refused as a video or an image: mtmd would send them to miniaudio and fail with an opaque error."""
    for second in range(0xE0, 0x100):
        b = bytes([0xFF, second]) + PAD
        assert bytes_error("video", b) == "the bytes are an image/audio file, not a video"
        assert bytes_error("image", b) == "the bytes are an audio file, not an image"
    adts = b"\xff\xf1" + PAD
    assert sniff_audio(adts) is None and bytes_error("audio", adts) == "unsupported audio format; use WAV, MP3 or FLAC"
    short = b"\xff\xf1" + b"\x00" * 9                            # under 12 bytes: mtmd's rule does not fire
    assert bytes_error("video", short) is None and bytes_error("image", short) is None


def test_supports_refuses_an_unknown_kind_like_bytes_error():
    with pytest.raises(ValueError, match="unknown media kind 'text'"):
        MediaInfo(True, True, True, "<m>", "").supports("text")
