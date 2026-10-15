"""The multipart splitter of pinned /v1/audio/transcriptions: it takes
only what cpp-httplib's FormDataParser takes (vendor/cpp-httplib/httplib.cpp @ llama.cpp 4ebdf2c), and refuses
(None: the request passes through untouched) whatever it cannot frame exactly."""
import pytest
from helpers import form_field, form_file, multipart

from callo.av import TONE_WAV
from callo.transcribe import (
    BOUNDARY_MAX,
    HEADER_MAX_COUNT,
    HEADER_MAX_LENGTH,
    PARTS_MAX,
    form_boundary,
    split_form,
)

CT = "multipart/form-data; boundary=XyZ"
CURL_BOUNDARY = b"------------------------d74496d66958873e"
HTTPX_BOUNDARY = b"9b2c1f0e8d7a6b5c4d3e2f1a0b9c8d7e"
CT_VALUE_MAX = HEADER_MAX_LENGTH - len("Content-Type:") - 2    # the longest value whose wire line (CRLF too) fits


def curl_body(data: bytes) -> bytes:
    """`curl -F file=@clip.wav -F model=whisper-1 URL`: file first, curl's dash-heavy boundary, its part Content-Type."""
    return multipart(form_file(data, ctype="audio/x-wav"), form_field("model", "whisper-1"), boundary=CURL_BOUNDARY)


def httpx_body(data: bytes) -> bytes:
    """httpx (the OpenAI SDK's client): fields first, a 32-hex boundary, the filename as given (spaces, UTF-8)."""
    return multipart(form_field("model", "whisper-1"), form_field("language", "de"),
                     form_file(data, filename="Aufnahme März 1.wav", ctype="audio/x-wav"), boundary=HTTPX_BOUNDARY)


def _lines(n: int) -> bytes:
    return b"".join(b"X-H%d: v\r\n" % i for i in range(n))


def test_the_boundary_is_taken_as_cpp_httplib_takes_it():
    # parse_multipart_boundary (httplib.cpp:4362-4374) over extract_media_type (:2823-2850)
    assert form_boundary(CT) == b"XyZ"
    assert form_boundary('multipart/form-data; boundary="a b:c"') == b"a b:c"            # quotes trimmed
    assert form_boundary("multipart/form-data;boundary=XyZ ; charset=utf-8") == b"XyZ"
    assert form_boundary(" multipart/form-data ; boundary=XyZ") == b"XyZ"                # the media type is trimmed
    assert form_boundary("multipart/form-data; boundary=" + "b" * BOUNDARY_MAX) == b"b" * 70
    for bad in ("Multipart/Form-Data; boundary=XyZ",     # is_multipart_form_data compares exactly (:6821-6824)
                "multipart/form-data", "multipart/form-data; boundary=", 'multipart/form-data; boundary=""',
                "multipart/form-data; boundary=" + "b" * (BOUNDARY_MAX + 1), "multipart/mixed; boundary=XyZ",
                'multipart/form-data; "boundary"=XyZ',  # extract_media_type does not unquote KEYS
                "multipart/form-data; boundary=a; boundary=b",       # stricter: a conflicting repeat
                "multipart/form-data; boundary=☃", "application/json", "",
                # read_headers (:3472-3484) 400s the whole request on a bad header line, before multipart is reached
                "multipart/form-data; boundary=XyZ\r\n x=y",       # an obs-fold
                "multipart/form-data; boundary=Xy\x01Z",            # a control byte
                "multipart/form-data; boundary=XyZ; x=" + "v" * (CT_VALUE_MAX - 36)):   # one byte over the line limit
        assert form_boundary(bad) is None, bad
    assert form_boundary("multipart/form-data; boundary=XyZ; x=" + "v" * (CT_VALUE_MAX - 37)) == b"XyZ"   # at it


@pytest.mark.parametrize("data", [
    TONE_WAV, TONE_WAV + b"\r", TONE_WAV + b"\n", TONE_WAV + b"\r\n", b"\r\n" + TONE_WAV,
    TONE_WAV + b"--XyZ" + TONE_WAV,            # the boundary text with no CRLF before it is data (state 3: \r\n--B)
    TONE_WAV + b"\n--XyZ\r\n" + TONE_WAV,      # a bare-LF "delimiter" is data too (stdlib email split here)
    b"",
], ids=["wav", "ends_cr", "ends_lf", "ends_crlf", "starts_crlf", "boundary_text", "bare_lf_delimiter", "empty"])
def test_fields_and_binary_file_bytes_come_back_exactly(data):
    parts = split_form(CT, multipart(form_file(data), form_field("prompt", "hi there"), form_field("language", "")))
    assert [(p.name, p.filename, bytes(p.content)) for p in parts] == [
        (b"file", b"clip.wav", data), (b"prompt", b"", b"hi there"), (b"language", b"", b"")]


def test_a_preamble_and_an_epilogue_are_skipped():
    # state 0 (httplib.cpp:4562-4579) finds the first `--B` CRLF; state 5 (:4710) drops the epilogue
    body = multipart(form_field("a", "1"), preamble=b"ignored --XyZ preamble\r\n", epilogue=b"\r\n--XyZ\r\ngarbage")
    assert [(p.name, bytes(p.content)) for p in split_form(CT, body)] == [(b"a", b"1")]


def test_a_part_is_a_file_iff_its_filename_is_non_empty():
    # httplib.cpp:8791-8797: an empty filename makes the part a form FIELD (curl -F 'file=<a.wav' sends none)
    body = multipart(form_file(TONE_WAV, filename=""), form_field("file", TONE_WAV),
                     form_file(TONE_WAV, name="f2", filename="a b ä.wav"))
    assert [p.filename for p in split_form(CT, body)] == [b"", b"", "a b ä.wav".encode()]


def test_repeated_parts_all_come_back_in_wire_order():
    # llama-server makes a repeated field an array and keeps the last repeated file (server-http.cpp:705-724): the
    # consumer decides what to do with repeats, so the splitter must hand every one over, in order
    body = multipart(form_field("model", "a"), form_file(b"1"), form_field("model", "b"), form_file(b"2"))
    assert [(p.name, bytes(p.content)) for p in split_form(CT, body)] == [
        (b"model", b"a"), (b"file", b"1"), (b"model", b"b"), (b"file", b"2")]


def test_an_identical_repeated_name_is_accepted():
    # parse_disposition_params keeps one value per key; a repeat with the same value is not a conflict
    (p,) = split_form(CT, multipart(b'Content-Disposition: form-data; name="a"; name=a\r\n\r\n1'))
    assert (p.name, p.filename, bytes(p.content)) == (b"a", b"", b"1")


def test_header_names_and_form_data_are_case_insensitive_and_binary_cte_is_accepted():
    # parse_content_disposition (:4743-4768) and the Content-Type test ignore case; httplib never decodes a CTE
    part = b"content-disposition:FORM-DATA;name=file; filename=a.wav\r\nCONTENT-TRANSFER-ENCODING: Binary\r\n\r\n"
    (p,) = split_form(CT, multipart(part + TONE_WAV))
    assert (p.name, p.filename, bytes(p.content)) == (b"file", b"a.wav", TONE_WAV)


def test_curl_and_httpx_bodies_split():
    (f, m) = split_form("multipart/form-data; boundary=" + CURL_BOUNDARY.decode(), curl_body(TONE_WAV))
    assert (f.name, f.filename, bytes(f.content), m.name, bytes(m.content)) == (
        b"file", b"clip.wav", TONE_WAV, b"model", b"whisper-1")
    parts = split_form('multipart/form-data; boundary="' + HTTPX_BOUNDARY.decode() + '"', httpx_body(TONE_WAV))
    assert [(p.name, p.filename) for p in parts] == [(b"model", b""), (b"language", b""),
                                                     (b"file", "Aufnahme März 1.wav".encode())]


def test_the_limits_themselves_are_accepted():
    ok_headers = multipart(b'Content-Disposition: form-data; name="a"\r\n' + _lines(HEADER_MAX_COUNT - 1) + b"\r\n1")
    long_line = b"X-H: " + b"v" * (HEADER_MAX_LENGTH - 5)                  # exactly HEADER_MAX_LENGTH bytes
    ok_long = multipart(b'Content-Disposition: form-data; name="a"\r\n' + long_line + b"\r\n\r\n1")
    ok_parts = multipart(*[form_field(f"f{i}", "1") for i in range(PARTS_MAX)])
    assert len(split_form(CT, ok_headers)) == 1 and len(split_form(CT, ok_long)) == 1
    assert len(split_form(CT, ok_parts)) == PARTS_MAX


REFUSED = {
    "lf_framing": multipart(form_field("a", "1")).replace(b"\r\n", b"\n"),             # CRLF only (crlf_, :4544)
    "padding_first": b"--XyZ  \r\n" + form_field("a", "1") + b"\r\n--XyZ--\r\n",          # no `--B` CRLF at all
    "padding_later": (b"--XyZ\r\n" + form_field("a", "1") + b"\r\n--XyZ \r\n" + form_field("b", "2")
                      + b"\r\n--XyZ--"),                                                 # state 4 (:4689-4709)
    "no_close": b"--XyZ\r\n" + form_field("a", "1") + b"\r\n",                          # is_valid_ only on `--`
    "truncated_in_headers": b"--XyZ\r\nContent-Disposition: form-data; name=",
    "no_name": multipart(b'Content-Disposition: form-data; filename="a.wav"\r\n\r\n' + TONE_WAV),   # :4627-4633
    "filename_star": multipart(b"Content-Disposition: form-data; name=file; filename*=UTF-8''a.wav\r\n\r\n" + TONE_WAV),
    "base64_cte": multipart(b'Content-Disposition: form-data; name="a"\r\nContent-Transfer-Encoding: base64\r\n\r\naGk='),
    "two_dispositions": multipart(b'Content-Disposition: form-data; name="a"\r\n'
                                  b'Content-Disposition: form-data; name="a"\r\n\r\n1'),
    "conflicting_name": multipart(b'Content-Disposition: form-data; name="a"; name="b"\r\n\r\n1'),
    "no_disposition": multipart(b"Content-Type: text/plain\r\n\r\n1"),
    "no_headers": multipart(b"\r\n1"),
    "not_form_data": multipart(b'Content-Disposition: attachment; name="a"\r\n\r\n1'),
    "header_without_colon": multipart(b'Content-Disposition: form-data; name="a"\r\nX-Bad\r\n\r\n1'),
    "space_before_colon": multipart(b'Content-Disposition : form-data; name="a"\r\n\r\n1'),
    "control_in_value": multipart(b'Content-Disposition: form-data; name="a"\r\nX-H: a\x01b\r\n\r\n1'),
    "obs_folded_header": multipart(b'Content-Disposition: form-data; name="a"\r\nX-H: v\r\n folded: y\r\n\r\n1'),
    "form_data_without_semicolon": multipart(b"Content-Disposition: form-data\r\n\r\n1"),    # not `form-data;`
    "headerless_run": b"--XyZ\r\n" + b"v" * (1 << 20) + b"\r\n\r\n1\r\n--XyZ--\r\n",      # refused within 8 KiB
    "bare_lf_in_header": multipart(b'Content-Disposition: form-data; name="a"\nX-H: v\r\n\r\n1'),
    "too_many_headers": multipart(b'Content-Disposition: form-data; name="a"\r\n' + _lines(HEADER_MAX_COUNT) + b"\r\n1"),
    "header_too_long": multipart(b'Content-Disposition: form-data; name="a"\r\nX-H: ' + b"v" * HEADER_MAX_LENGTH
                                 + b"\r\n\r\n1"),
    "boundary_then_text": multipart(form_field("a", "1\r\n--XyZjunk")),
    "too_many_parts": multipart(*[form_field(f"f{i}", "1") for i in range(PARTS_MAX + 1)]),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_what_cpp_httplib_refuses_or_callo_will_not_frame_is_none(name):
    assert split_form(CT, REFUSED[name]) is None


def test_a_bad_content_type_is_none_whatever_the_body():
    assert split_form("application/json", multipart(form_field("a", "1"))) is None
