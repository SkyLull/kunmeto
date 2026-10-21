"""Read metadata from a Fujifilm RAF, or a camera JPG, without exiftool.

RAF = "FUJIFILMCCD-RAW " header -> embedded JPEG (offset/length at byte 84) whose APP1 Exif block is
a TIFF: IFD0 -> ExifIFD (0x8769) -> MakerNote (0x927C). The Fuji maker note is "FUJIFILM" + LE uint32
IFD offset, and its offsets are relative to the maker note start, so the blob is self-contained. A
camera JPG (e.g. the composite output of multi-exposure mode) carries the same Exif/maker-note
structure directly in its own APP1 block.
"""
import struct
from dataclasses import dataclass, field

TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}


class ExifError(ValueError):
    """A RAF or camera JPG had the expected magic bytes but a malformed Exif/maker-note block."""


# Fuji maker note tags (verified against exiftool output on GFX100S RAF/JPG files)
FUJI_SEQUENCE_NUMBER = 0x1101
FUJI_AUTO_BRACKETING = 0x1100      # 6 = Pixel Shift
FUJI_PIXEL_SHIFT_SHOTS = 0x1105
FUJI_PIXEL_SHIFT_OFFSET = 0x1106   # SRATIONAL[2], (x, y)
FUJI_EXPOSURE_COUNT = 0x1032       # >1 only on the camera's multi-exposure composite JPG
FUJI_MULTI_EXPOSURE_MODE = 0x1037  # composite JPG only; 3 = "bright" seen, other values unverified
FUJI_IMAGE_COUNT = 0x1438          # shared by all frames of one pixel-shift set
PIXEL_SHIFT = 6
IMAGE_COUNT_MOD = 0x8000           # ImageCount is a 15-bit counter, wraps; exiftool masks & 0x7fff
                                    # (FujiFilm.pm) since bit 15 may be set by some writers


@dataclass
class Entry:
    tag: int
    type: int
    count: int
    raw: bytes          # value bytes exactly as stored, file byte order
    value: object       # decoded value(s)


@dataclass
class RafMeta:
    ifd0: dict = field(default_factory=dict)        # tag -> Entry
    exif: dict = field(default_factory=dict)        # tag -> Entry
    makernote: bytes = b""                          # raw Fuji maker note blob
    fuji: dict = field(default_factory=dict)        # maker-note tag -> Entry
    byteorder: str = "<"

    def _fuji_num(self, tag, idx=0):
        e = self.fuji.get(tag)
        v = e.value if e else None
        return v[idx] if isinstance(v, tuple) and len(v) > idx else None

    def _fuji_int(self, tag, default=0):
        v = self._fuji_num(tag)
        return v if isinstance(v, int) else default

    @property
    def model(self):
        e = self.ifd0.get(0x110)
        return e.value if e and isinstance(e.value, str) else ""

    @property
    def make(self):
        e = self.ifd0.get(0x10F)
        return e.value if e and isinstance(e.value, str) else ""

    @property
    def serial(self):
        e = self.exif.get(0xA431)
        return e.value if e and isinstance(e.value, str) else ""

    @property
    def image_count(self):
        v = self._fuji_int(FUJI_IMAGE_COUNT, None)
        return v & (IMAGE_COUNT_MOD - 1) if v is not None else None

    @property
    def auto_bracketing(self):
        return self._fuji_int(FUJI_AUTO_BRACKETING)

    @property
    def pixel_shift_shots(self):
        return self._fuji_int(FUJI_PIXEL_SHIFT_SHOTS)

    @property
    def pixel_shift_offset(self):
        """(x, y) sensor offset in pixels, or None for a normal shot."""
        x = self._fuji_num(FUJI_PIXEL_SHIFT_OFFSET, 0)
        y = self._fuji_num(FUJI_PIXEL_SHIFT_OFFSET, 1)
        return (float(x), float(y)) if x is not None and y is not None else None

    @property
    def sequence_number(self):
        return self._fuji_int(FUJI_SEQUENCE_NUMBER)

    @property
    def exposure_count(self):
        return self._fuji_int(FUJI_EXPOSURE_COUNT, 1)

    @property
    def multi_exposure_mode(self):
        v = self._fuji_num(FUJI_MULTI_EXPOSURE_MODE)
        return v if isinstance(v, int) else None


def _decode(bo, typ, count, raw):
    if typ == 2:
        return raw.split(b"\0", 1)[0].decode("latin-1")
    if typ in (1, 7):
        return raw
    if typ == 6:
        return struct.unpack(f"{count}b", raw)
    fmt = {3: "H", 4: "I", 8: "h", 9: "i", 11: "f", 12: "d", 13: "I"}.get(typ)
    if fmt:
        return struct.unpack(f"{bo}{count}{fmt}", raw)
    if typ in (5, 10):
        v = struct.unpack(f"{bo}{2 * count}{'I' if typ == 5 else 'i'}", raw)
        return tuple(v[i] / v[i + 1] if v[i + 1] else 0.0 for i in range(0, len(v), 2))
    return raw


def read_ifd(buf, off, bo, base=0):
    """Parse one IFD at buf[base + off]; value offsets are relative to base."""
    p = base + off
    (n,) = struct.unpack_from(f"{bo}H", buf, p)
    out = {}
    for i in range(n):
        tag, typ, count, voff = struct.unpack_from(f"{bo}HHI4s", buf, p + 2 + 12 * i)
        size = TYPE_SIZE.get(typ, 1) * count
        if size <= 4:
            raw = voff[:size]
        else:
            (o,) = struct.unpack(f"{bo}I", voff)
            raw = buf[base + o:base + o + size]
        out[tag] = Entry(tag, typ, count, raw, _decode(bo, typ, count, raw))
    return out


def _meta_from_jpeg(jpeg, where) -> RafMeta:
    i = jpeg.find(b"Exif\0\0")
    if i < 0:
        raise ValueError(f"{where}: no Exif block")
    tiff = jpeg[i + 6:]
    if tiff[:2] not in (b"II", b"MM"):
        raise ValueError(f"{where}: malformed Exif block")
    bo = "<" if tiff[:2] == b"II" else ">"
    try:
        (ifd0_off,) = struct.unpack_from(f"{bo}I", tiff, 4)

        meta = RafMeta(byteorder=bo)
        meta.ifd0 = read_ifd(tiff, ifd0_off, bo)

        exif_ptr = meta.ifd0.get(0x8769)
        v = exif_ptr.value if exif_ptr else None
        if isinstance(v, tuple) and v and isinstance(v[0], int):
            meta.exif = read_ifd(tiff, v[0], bo)

        mn = meta.exif.get(0x927C)
        # A truncated maker note is deliberately treated as "no Fuji data" (files without
        # ImageCount are skipped by grouping), not as an ExifError.
        if mn and mn.raw.startswith(b"FUJIFILM") and len(mn.raw) == mn.count:
            meta.makernote = mn.raw
            (mn_ifd,) = struct.unpack_from("<I", mn.raw, 8)   # Fuji maker notes are always little-endian
            meta.fuji = read_ifd(mn.raw, mn_ifd, "<")
    except (struct.error, IndexError, TypeError) as e:
        raise ExifError(f"{where}: malformed Exif ({e})") from e
    return meta


def read_raf_meta(path) -> RafMeta:
    with open(path, "rb") as fh:
        head = fh.read(108)
        if not head.startswith(b"FUJIFILMCCD-RAW"):
            raise ValueError(f"{path}: not a RAF")
        try:
            jpeg_off, jpeg_len = struct.unpack(">2I", head[84:92])
        except struct.error as e:
            raise ExifError(f"{path}: malformed RAF header ({e})") from e
        fh.seek(jpeg_off)
        # The Exif APP1 block is near the start of the embedded preview; the maker-note length
        # guard in _meta_from_jpeg already treats a truncated maker note as "no Fuji data", so
        # capping the read avoids pulling in the whole ~2.3 MB preview just to read a few KB of
        # tags.
        jpeg = fh.read(min(jpeg_len, 256 * 1024))
    return _meta_from_jpeg(jpeg, path)


def read_jpeg_meta(path) -> RafMeta:
    """Exif + Fuji maker note of a camera JPG. APP1 is <= 64 KiB and sits near the start of the
    file; we read 256 KiB for margin."""
    with open(path, "rb") as fh:
        head = fh.read(256 * 1024)
    if not head.startswith(b"\xff\xd8"):
        raise ValueError(f"{path}: not a JPEG")
    return _meta_from_jpeg(head, path)
