import builtins
import struct

import pytest

from conftest import DATA
from kunmeto.raf import read_jpeg_meta, read_raf_meta


def test_rejects_non_raf(tmp_path):
    p = tmp_path / "x.RAF"
    p.write_bytes(b"not a raf at all" * 10)
    with pytest.raises(ValueError):
        read_raf_meta(p)


def test_rejects_non_jpeg(tmp_path):
    p = tmp_path / "x.JPG"
    p.write_bytes(b"GIF89a" + b"\0" * 100)
    with pytest.raises(ValueError):
        read_jpeg_meta(p)


# --- synthetic Exif/maker-note construction (no data/ needed) -------------------------------

MAKE = b"FUJIFILM\0"
MODEL = b"GFX100S\0"
SERIAL = b"23002091\0"


def _pack_ifd(entries, offset):
    """entries: list of (tag, typ, count, payload_bytes). `offset` is this IFD's own position,
    relative to whatever base `read_ifd` will be called with. Returns a self-contained blob
    (header + inline values + next-IFD=0 + overflow values) meant to sit at that offset.
    Always little-endian: everything this module builds (main TIFF and Fuji maker note) is."""
    n = len(entries)
    header_len = 2 + 12 * n + 4
    overflow_start = offset + header_len
    entry_bytes = bytearray()
    overflow = bytearray()
    for tag, typ, count, payload in entries:
        if len(payload) <= 4:
            voff = payload + b"\0" * (4 - len(payload))
        else:
            voff = struct.pack("<I", overflow_start + len(overflow))
            overflow += payload
        entry_bytes += struct.pack("<HHI", tag, typ, count) + voff
    return struct.pack("<H", n) + bytes(entry_bytes) + struct.pack("<I", 0) + bytes(overflow)


def _fuji_makernote(fuji_entries):
    """"FUJIFILM" + LE uint32 IFD offset (relative to this blob) + the Fuji IFD at offset 12."""
    return b"FUJIFILM" + struct.pack("<I", 12) + _pack_ifd(fuji_entries, offset=12)


def _jpeg(tiff_bytes):
    return b"\xff\xd8" + b"Exif\0\0" + tiff_bytes


def _full_jpeg(fuji_entries=None, exif_ptr_override=None, mn_count_override=None):
    """IFD0 (Make/Model + 0x8769 pointer) -> Exif IFD (Serial + MakerNote) -> Fuji IFD.

    exif_ptr_override, if given, is (typ, count, payload) and replaces the 0x8769 entry so it
    no longer points at a valid Exif IFD (used to test malformed-pointer handling).
    mn_count_override, if given, replaces the MakerNote entry's declared byte count, letting it
    claim more bytes than are actually present (used to test the truncated-maker-note guard).
    """
    if fuji_entries is None:
        fuji_entries = [
            (0x1438, 4, 1, struct.pack("<I", 12345)),           # ImageCount LONG
            (0x1106, 10, 2, struct.pack("<4i", 0, 1, -5, 2)),   # PixelShiftOffset SRATIONAL[2] -> (0.0, -2.5)
        ]
    mn_blob = _fuji_makernote(fuji_entries)
    mn_count = mn_count_override if mn_count_override is not None else len(mn_blob)
    exif_entries = [
        (0xA431, 2, len(SERIAL), SERIAL),
        (0x927C, 7, mn_count, mn_blob),
    ]

    ifd0_off = 8
    n0 = 3
    header0_len = 2 + 12 * n0 + 4
    overflow0_len = len(MAKE) + len(MODEL)
    exif_ifd_off = ifd0_off + header0_len + overflow0_len

    if exif_ptr_override is not None:
        typ, count, payload = exif_ptr_override
        exif_ptr_entry = (0x8769, typ, count, payload)
    else:
        exif_ptr_entry = (0x8769, 4, 1, struct.pack("<I", exif_ifd_off))

    ifd0_entries = [(0x10F, 2, len(MAKE), MAKE), (0x110, 2, len(MODEL), MODEL), exif_ptr_entry]
    ifd0_bytes = _pack_ifd(ifd0_entries, offset=ifd0_off)
    exif_ifd_bytes = _pack_ifd(exif_entries, offset=exif_ifd_off)
    tiff = b"II" + struct.pack("<H", 42) + struct.pack("<I", ifd0_off) + ifd0_bytes + exif_ifd_bytes
    return _jpeg(tiff)


def _all_properties(m):
    """Touch every RafMeta property; must never raise."""
    return (m.model, m.make, m.serial, m.image_count, m.auto_bracketing, m.pixel_shift_shots,
            m.pixel_shift_offset, m.sequence_number, m.exposure_count, m.multi_exposure_mode)


def test_read_jpeg_meta_synthetic(tmp_path):
    p = tmp_path / "synthetic.JPG"
    p.write_bytes(_full_jpeg())
    m = read_jpeg_meta(p)
    assert m.model == "GFX100S"
    assert m.serial == "23002091"
    assert m.image_count == 12345
    assert m.pixel_shift_offset == (0.0, -2.5)
    assert m.makernote != b""


def _check_image_count_none(m):
    assert m.image_count is None


MALFORMED_CASES = {
    "bad_ifd0_offset": (
        _jpeg(b"II" + struct.pack("<H", 42) + struct.pack("<I", 999_999)),
        True,
        None,
    ),
    "truncated_tiff_header": (
        _jpeg(b"II"),
        True,
        None,
    ),
    "exif_ptr_ascii": (
        _full_jpeg(exif_ptr_override=(2, 6, b"hello\0")),
        False,
        None,
    ),
    "exif_ptr_float": (
        _full_jpeg(exif_ptr_override=(11, 1, struct.pack("<f", 1.5))),
        False,
        None,
    ),
    "exif_ptr_count0": (
        _full_jpeg(exif_ptr_override=(4, 0, b"")),
        False,
        None,
    ),
    "fuji_image_count_count0": (
        _full_jpeg(fuji_entries=[
            (0x1438, 4, 0, b""),
            (0x1106, 10, 2, struct.pack("<4i", 0, 1, -5, 2)),
        ]),
        False,
        None,
    ),
    "fuji_image_count_ascii": (
        _full_jpeg(fuji_entries=[
            (0x1438, 2, 6, b"hello\0"),
            (0x1106, 10, 2, struct.pack("<4i", 0, 1, -5, 2)),
        ]),
        False,
        None,
    ),
    "fuji_image_count_nan": (
        _full_jpeg(fuji_entries=[
            (0x1438, 11, 1, struct.pack("<f", float("nan"))),
            (0x1106, 10, 2, struct.pack("<4i", 0, 1, -5, 2)),
        ]),
        False,
        _check_image_count_none,
    ),
    "fuji_image_count_inf": (
        _full_jpeg(fuji_entries=[
            (0x1438, 11, 1, struct.pack("<f", float("inf"))),
            (0x1106, 10, 2, struct.pack("<4i", 0, 1, -5, 2)),
        ]),
        False,
        _check_image_count_none,
    ),
}


@pytest.mark.parametrize("data,should_raise,check", MALFORMED_CASES.values(), ids=list(MALFORMED_CASES))
def test_malformed_jpeg_handling(tmp_path, data, should_raise, check):
    p = tmp_path / "x.JPG"
    p.write_bytes(data)
    if should_raise:
        with pytest.raises(ValueError):
            read_jpeg_meta(p)
    else:
        m = read_jpeg_meta(p)
        _all_properties(m)  # must not raise
        if check:
            check(m)


def test_raf_truncated_header(tmp_path):
    p = tmp_path / "x.RAF"
    p.write_bytes(b"FUJIFILMCCD-RAW" + b"\0" * 45)  # magic present, only 60 bytes total
    with pytest.raises(ValueError):
        read_raf_meta(p)


def test_truncated_makernote_guard(tmp_path):
    """0x927C claims more bytes than the file actually holds: treat it as no Fuji data at all."""
    p = tmp_path / "x.JPG"
    p.write_bytes(_full_jpeg(mn_count_override=99_999))
    m = read_jpeg_meta(p)
    assert m.fuji == {}
    assert m.makernote == b""
    assert m.image_count is None


def test_image_count_masks_bit15(tmp_path):
    """ImageCount (0x1438) is a 15-bit counter; exiftool masks & 0x7fff since some writers set
    bit 15. 0x8000 | 42 must decode the same as a plain 42."""
    p = tmp_path / "x.JPG"
    p.write_bytes(_full_jpeg(fuji_entries=[
        (0x1438, 4, 1, struct.pack("<I", 0x8000 | 42)),
        (0x1106, 10, 2, struct.pack("<4i", 0, 1, -5, 2)),
    ]))
    m = read_jpeg_meta(p)
    assert m.image_count == 42


def test_raf_bounds_preview_read(tmp_path, monkeypatch):
    """read_raf_meta must not read the whole ~2.3 MB embedded preview: the Exif APP1 block sits
    at its start, so only a bounded prefix (256 KiB) should ever be requested from the file,
    even when the RAF header's declared preview length is much larger."""
    import kunmeto.raf as raf_mod

    header = bytearray(108)
    header[0:15] = b"FUJIFILMCCD-RAW"
    header[84:92] = struct.pack(">2I", 108, 2_300_000)   # jpeg_off=108, jpeg_len~2.3 MB (declared)

    jpeg = _full_jpeg()
    body = jpeg + b"\0" * (300 * 1024 - len(jpeg))       # real data available: 300 KiB
    p = tmp_path / "big.RAF"
    p.write_bytes(bytes(header) + body)

    reads = []
    real_open = builtins.open

    class Spy:
        def __init__(self, fh):
            self._fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._fh.__exit__(*a)

        def read(self, n=-1):
            reads.append(n)
            return self._fh.read(n)

        def seek(self, *a, **kw):
            return self._fh.seek(*a, **kw)

    monkeypatch.setattr(raf_mod, "open", lambda *a, **kw: Spy(real_open(*a, **kw)), raising=False)

    m = raf_mod.read_raf_meta(p)
    assert m.model == "GFX100S"
    assert m.image_count == 12345
    assert all(n is not None and 0 <= n <= 256 * 1024 for n in reads)
