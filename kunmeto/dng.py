"""DNG writer for kunmeto outputs.

IFD0 = small 8-bit RGB preview (NewSubfileType 1) + DNG/colour tags; SubIFD 0 = the full image in one of:
- LinearRaw (3 samples/pixel, demosaiced): pixel-shift output, like Fuji Pixel Shift Combiner.
- CFA (Bayer mosaic, 1 sample/pixel): multi-exposure stacks, so the raw developer still demosaics.

uint16 samples use 16-bit lossless-JPEG tiles (as Fuji does). float16/float32 use Adobe Deflate
(compression 8) + floating-point predictor 3 (DNG 1.4); tifffile's plain 'deflate' (32946) is not
read by LibRaw/darktable. Classic TIFF only (DNG readers don't take BigTIFF): the payload has to
stay under the 4 GiB / 32-bit offset limit. tifffile 2026.9.20 does not itself guard that limit on
this write path -- going over it surfaces as a struct.error while packing offsets, after tiles have
already been streamed to disk, leaving a broken multi-GB file. write_dng catches that, but a
struct.error can also come from an unrelated bad tag value (e.g. a negative crop origin); only the
case where the file on disk has actually grown to the 4 GiB limit is reported as DngTooLargeError,
deleting the broken file first -- anything else is re-raised as a plain ValueError so callers (like
the pixel-shift/float16 fallback in pipeline.py) don't mistake a real bug for an oversized payload.
tifffile quirks are patched after writing.

Overflow detection (_is_overflow) relies on this exact tifffile behaviour -- streaming tiles to
disk before raising struct.error on the too-large offset, rather than checking the size up front --
which is why pyproject.toml pins tifffile>=2026.1,<2027 (verified on 2026.9.20); an earlier or
later tifffile that validates offsets before writing, or that raises a different exception, would
make DngTooLargeError undetectable here.
"""
import os
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import tifffile

# TIFF field types
BYTE, ASCII, SHORT, LONG, RATIONAL, SRATIONAL = 1, 2, 3, 4, 5, 10
D65 = 21  # EXIF LightSource code for CalibrationIlluminant
LINEAR_RAW, CFA = 34892, 32803
EXIF_IFD = 34665
EXIF_PLACEHOLDER = 34664  # tifffile refuses to write 34665; written as this, then renumbered
# Exif tags that describe the RAF's embedded JPEG or point at other IFDs: not valid for the DNG.
EXIF_SKIP = {0x927C, 0xA002, 0xA003, 0xA005, 0x9286,   # MakerNote, PixelX/YDimension, Interop, UserComment
             0x9101, 0x9102}                          # ComponentsConfiguration, CompressedBitsPerPixel


class DngTooLargeError(ValueError):
    """The written DNG would exceed (or exceeded) the classic-TIFF 4 GiB / 32-bit offset limit."""


# Slack below the hard 4 GiB / 2**32 offset limit: room for the preview IFD/tags and the Exif IFD
# appended after the raw image tiles, so a file that's merely "close" isn't misreported as overflow.
_CLASSIC_TIFF_SLACK = 2 ** 25


def _is_overflow(path):
    """True when the file already on disk is within `_CLASSIC_TIFF_SLACK` of the 4 GiB offset limit.

    Used to tell a genuine classic-TIFF overflow (tiles already streamed past the limit) apart from
    a struct.error raised for an unrelated bad tag value, which leaves a small or absent file.
    """
    p = Path(path)
    return p.exists() and p.stat().st_size >= 2 ** 32 - _CLASSIC_TIFF_SLACK


def _rationals(values, signed=False, denom=10000):
    out = []
    for v in values:
        f = Fraction(float(v)).limit_denominator(denom)
        n, d = f.numerator, f.denominator
        if not signed and n < 0:
            raise ValueError("negative value in unsigned RATIONAL")
        out += [n, d]
    return out


@dataclass
class DngMeta:
    make: str
    model: str
    color_matrix_d65: np.ndarray        # 3x3 XYZ(D65) -> camera, DNG ColorMatrix convention
    as_shot_neutral: tuple              # camera-space neutral, G = 1
    black: float                        # black level in the written image's units
    white: float | tuple                # white level in the written image's units (1 for float);
                                        # a tuple gives one value per sample (LinearRaw R, G, B)
    crop_origin: tuple = (0, 0)         # DefaultCropOrigin (x, y)
    crop_size: tuple | None = None      # DefaultCropSize (w, h); None = full image
    cfa_pattern: tuple | None = None    # CFA layout only: 2x2 row-major, 0=R 1=G 2=B
    orientation: int = 1                # TIFF Orientation (1 = normal)
    software: str = "kunmeto"
    extra_ifd0: list = field(default_factory=list)
    exif: list = field(default_factory=list)       # raf.Entry list (little-endian raw values) for the Exif IFD
    makernote: bytes = b""                         # original maker note blob -> DNGPrivateData "MakN"
    makernote_order: str = "<"


def short_entry(tag, *values):
    """A SHORT-typed IFD entry in raf.Entry shape, for adding/overriding Exif tags."""
    from .raf import Entry
    return Entry(tag, SHORT, len(values), struct.pack(f"<{len(values)}H", *values), values)


def rational_entry(tag, *values, denom=10000):
    """A RATIONAL-typed IFD entry in raf.Entry shape, for adding/overriding Exif tags with a
    recomputed value (e.g. a rescaled FocalPlaneResolution)."""
    from .raf import Entry
    packed = _rationals(values, denom=denom)
    decoded = tuple(packed[i] / packed[i + 1] if packed[i + 1] else 0.0 for i in range(0, len(packed), 2))
    return Entry(tag, RATIONAL, len(values), struct.pack(f"<{len(packed)}I", *packed), decoded)


def composite_exif(n_used, n_total):
    """EXIF 2.32 CompositeImage = 2 (general composite) + SourceImageNumberOfCompositeImage."""
    return [short_entry(0xA460, 2), short_entry(0xA461, n_total, n_used)]


def _ifd_blob(entries, start):
    """Serialize a little-endian IFD (raf.Entry list) to be placed at file offset `start`."""
    entries = sorted(entries, key=lambda e: e.tag)
    n = len(entries)
    data_off = start + 2 + 12 * n + 4
    head, data = [struct.pack("<H", n)], bytearray()
    for e in entries:
        raw = e.raw
        if len(raw) <= 4:
            head.append(struct.pack("<HHI", e.tag, e.type, e.count) + raw.ljust(4, b"\0"))
        else:
            if (data_off + len(data)) % 2:
                data += b"\0"
            head.append(struct.pack("<HHII", e.tag, e.type, e.count, data_off + len(data)))
            data += raw
    head.append(struct.pack("<I", 0))  # no next IFD
    return b"".join(head) + bytes(data)


def _makn(makernote, order):
    # DNG 1.4 spec, DNGPrivateData: "Adobe\0" + "MakN" + BE count + byte order + BE original offset + blob.
    body = (b"II" if order == "<" else b"MM") + struct.pack(">I", 0) + makernote
    return b"Adobe\0" + b"MakN" + struct.pack(">I", len(body)) + body


def _preview(img, meta, scale):
    """Crudely white-balanced, gamma-encoded 8-bit RGB thumbnail; not used for rendering."""
    if img.ndim == 2:
        h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
        step = 2 * max(1, scale // 2)
        subs = {(pi, pj): img[pi:h:step, pj:w:step] for pi, pj in ((0, 0), (0, 1), (1, 0), (1, 1))}
        rows = min(s.shape[0] for s in subs.values())
        cols = min(s.shape[1] for s in subs.values())
        rgb = np.zeros((rows, cols, 3), np.float32)
        n = np.zeros(3, np.float32)
        for k, (pi, pj) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            c = meta.cfa_pattern[k]
            rgb[..., c] += subs[(pi, pj)][:rows, :cols]
            n[c] += 1
        rgb = rgb / n
    else:
        rgb = img[::scale, ::scale].astype(np.float32)
    rgb = (rgb - meta.black) / np.asarray(meta.as_shot_neutral, np.float32)
    rgb = np.clip(rgb / max(float(np.percentile(rgb, 99.5)), 1e-6), 0, 1) ** (1 / 2.2)
    return (rgb * 255).astype(np.uint8)


def _validate_meta(meta: DngMeta):
    """Catch obviously-bad metadata up front, before anything is written to disk.

    Without this, a bad crop_origin/crop_size/black value can surface deep inside tifffile as a
    struct.error, which write_dng would otherwise be unable to tell apart from a real 4 GiB overflow.
    """
    if len(meta.crop_origin) != 2:
        raise ValueError(f"crop_origin must have exactly 2 values, got {meta.crop_origin}")
    for v in meta.crop_origin:
        if int(v) != v or v < 0:
            raise ValueError(f"crop_origin must be two non-negative ints, got {meta.crop_origin}")
    if meta.crop_size is not None:
        if len(meta.crop_size) != 2:
            raise ValueError(f"crop_size must have exactly 2 values, got {meta.crop_size}")
        for v in meta.crop_size:
            if int(v) != v or v <= 0:
                raise ValueError(f"crop_size must be two positive ints, got {meta.crop_size}")
    if not np.isfinite(meta.black) or meta.black < 0:
        raise ValueError(f"black must be a finite value >= 0, got {meta.black}")


def _white_levels(white, nsamp):
    """WhiteLevel values: a scalar repeated per sample, or one per sample."""
    if np.ndim(white) == 0:
        return (int(round(float(white))),) * nsamp
    if len(white) != nsamp:
        raise ValueError(f"white has {len(white)} values for {nsamp} samples per pixel")
    return tuple(int(round(float(w))) for w in white)


def write_dng(path, img, meta: DngMeta, tile=(96, 128), preview_scale=16):
    """Write img as a DNG: (H, W, 3) -> LinearRaw, (H, W) -> CFA (needs meta.cfa_pattern)."""
    _validate_meta(meta)
    if img.ndim == 3 and img.shape[2] == 3:
        layout, nsamp = LINEAR_RAW, 3
    elif img.ndim == 2:
        if meta.cfa_pattern is None or len(meta.cfa_pattern) != 4:
            raise ValueError("CFA image needs meta.cfa_pattern (4 entries)")
        if set(meta.cfa_pattern) != {0, 1, 2}:
            # DNG CFAPattern is R=0/G=1/B=2/C=3; LibRaw reports Fuji's second green as 3 (Cyan),
            # so callers must map that 3 -> 1 before calling write_dng.
            raise ValueError(f"cfa_pattern must cover R=0/G=1/B=2 (map LibRaw's G2=3 to 1), got {meta.cfa_pattern}")
        layout, nsamp = CFA, 1
    else:
        raise ValueError(f"expected (H, W, 3) or (H, W) image, got {img.shape}")
    if img.dtype == np.uint16:
        codec = dict(compression="jpeg", bitspersample=16,
                     compressionargs={"lossless": True, "predictor": 1, "bitspersample": 16})
        backward = (1, 1, 0, 0)
    elif img.dtype in (np.float16, np.float32):
        if float(meta.white) != 1.0:
            raise ValueError(f"float DNGs must use white=1 (DNG's float white point), got {meta.white}")
        codec = dict(compression="adobe_deflate", predictor=3)
        backward = (1, 4, 0, 0)
    else:
        raise ValueError(f"unsupported sample type {img.dtype}")
    h, w = img.shape[:2]
    crop_size = meta.crop_size or (w, h)

    ifd0 = [
        (271, ASCII, 0, meta.make, True),
        (272, ASCII, 0, meta.model, True),
        (274, SHORT, 1, meta.orientation if 1 <= meta.orientation <= 8 else 1, True),  # Orientation (1-8)
        (50706, BYTE, 4, (1, 4, 0, 0), True),                      # DNGVersion
        (50707, BYTE, 4, backward, True),                          # DNGBackwardVersion
        (50708, ASCII, 0, f"{meta.make} {meta.model}", True),      # UniqueCameraModel
        (50721, SRATIONAL, 9, _rationals(np.ravel(meta.color_matrix_d65), signed=True), True),
        (50778, SHORT, 1, D65, True),                              # CalibrationIlluminant1
        (50728, RATIONAL, 3, _rationals(meta.as_shot_neutral), True),
        *meta.extra_ifd0,
    ]
    if meta.exif:
        ifd0.append((EXIF_PLACEHOLDER, LONG, 1, 0, True))  # renumbered + pointed at the Exif IFD after writing
    if meta.makernote:
        blob = _makn(meta.makernote, meta.makernote_order)
        ifd0.append((50740, BYTE, len(blob), blob, True))  # DNGPrivateData

    if layout == CFA:
        raw_tags = [
            (33421, SHORT, 2, (2, 2), True),                       # CFARepeatPatternDim
            (33422, BYTE, 4, tuple(meta.cfa_pattern), True),       # CFAPattern
            (50710, BYTE, 3, (0, 1, 2), True),                     # CFAPlaneColor
            (50711, SHORT, 1, 1, True),                            # CFALayout: rectangular
            (50713, SHORT, 2, (2, 2), True),                       # BlackLevelRepeatDim
            (50714, RATIONAL, 4, _rationals([meta.black] * 4), True),
        ]
    else:
        raw_tags = [
            (50713, SHORT, 2, (1, 1), True),
            (50714, RATIONAL, 3, _rationals([meta.black] * 3), True),
        ]
    raw_tags += [
        (50717, LONG, nsamp, _white_levels(meta.white, nsamp), True),        # WhiteLevel
        (50719, LONG, 2, tuple(int(v) for v in meta.crop_origin), True),     # DefaultCropOrigin
        (50720, LONG, 2, tuple(int(v) for v in crop_size), True),            # DefaultCropSize
    ]

    try:
        with tifffile.TiffWriter(path, bigtiff=False) as tw:
            tw.write(_preview(img, meta, preview_scale), photometric="rgb", subfiletype=1, subifds=1,
                     software=meta.software, extratags=ifd0, metadata=None)
            tw.write(img, photometric=layout, planarconfig="contig" if nsamp == 3 else None, tile=tile,
                     subfiletype=0, extratags=raw_tags, metadata=None, software=False,
                     maxworkers=os.cpu_count(), **codec)   # tiles compressed in parallel
    except struct.error as e:
        too_big = _is_overflow(path)
        Path(path).unlink(missing_ok=True)
        if too_big:
            raise DngTooLargeError(f"{path}: DNG too large for classic TIFF (4 GiB limit)") from e
        raise ValueError(f"{path}: invalid DNG tag value ({e})") from e

    _fix_tifffile_tags(path)
    if meta.exif:
        deduped = {}
        for e in meta.exif:
            deduped[e.tag] = e  # last occurrence per tag wins, so callers can append overrides
        _append_exif_ifd(path, [e for e in deduped.values() if e.tag not in EXIF_SKIP])


write_linear_dng = write_dng  # deprecated alias predating CFA support


def _fix_tifffile_tags(path):
    """Undo two tifffile choices that DNG readers dislike.

    - LinearRaw is unknown to tifffile, so it writes 3-sample pixels as 1 colour + ExtraSamples(0, 0).
      Delete that entry from the raw IFD.
    - SubIFDs (0x14A) is written with type IFD (13). Retype it to LONG, as Adobe DNGs do.
    """
    with tifffile.TiffFile(path) as tf:
        subifds = tf.pages[0].tags[330]
        raw = tf.pages[0].pages[0]
        extra = raw.tags.get(338)
        raw_ifd, extra_entry = raw.offset, (extra.offset if extra else None)
    with open(path, "r+b") as fh:
        fh.seek(subifds.offset + 2)
        fh.write(struct.pack("<H", LONG))
        if extra_entry is not None:
            fh.seek(raw_ifd)
            (n,) = struct.unpack("<H", fh.read(2))
            table = bytearray(fh.read(12 * n + 4))       # entries + next-IFD offset
            i = (extra_entry - raw_ifd - 2) // 12
            del table[12 * i:12 * i + 12]
            fh.seek(raw_ifd)
            fh.write(struct.pack("<H", n - 1) + bytes(table) + b"\0" * 12)


def _append_exif_ifd(path, entries):
    """Append an Exif IFD at EOF and point IFD0's ExifIFD tag at it."""
    with tifffile.TiffFile(path) as tf:
        if tf.byteorder != "<":
            raise ValueError("Exif IFD patching assumes a little-endian DNG")
        tag = tf.pages[0].tags[EXIF_PLACEHOLDER]
        entry, valueoffset = tag.offset, tag.valueoffset
    too_big = False
    with open(path, "r+b") as fh:
        fh.seek(0, 2)
        start = fh.tell()
        if start % 2:
            fh.write(b"\0")
            start += 1
        blob = _ifd_blob(entries, start)
        if start + len(blob) >= 2 ** 32:
            too_big = True
        else:
            fh.write(blob)
            fh.seek(entry)
            fh.write(struct.pack("<H", EXIF_IFD))
            fh.seek(valueoffset)
            fh.write(struct.pack("<I", start))
    if too_big:
        # Unlink only after the file handle above is closed, so this is safe on Windows too.
        Path(path).unlink(missing_ok=True)
        raise DngTooLargeError(f"{path}: DNG too large for classic TIFF (4 GiB limit)")
