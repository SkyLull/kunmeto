"""End-to-end jobs: decode -> merge -> DNG."""
import dataclasses
import os
from pathlib import Path

import numpy as np

from . import __version__
from .dng import ASCII, DngMeta, DngTooLargeError, composite_exif, rational_entry, write_dng
from .frames import decode_all
from .progress import report as progress_report
from .raf import read_raf_meta
from .repair import merge_repaired, replaced_fraction
from .shift import CH, clip_ceiling, grid_for, merge_shift, normalization_gains, on_grid
from .stack import merge_stack

SOFTWARE = f"kunmeto {__version__}"
CHUNK = 1024  # rows per conversion chunk, keeps temporaries small on 400 MP images
# FocalPlaneX/YResolution, FocalPlaneResolutionUnit
FOCAL_PLANE_TAGS = (0xA20E, 0xA20F, 0xA210)
# PixelX/YDimension: the embedded JPEG's size, which FocalPlaneX/YResolution are relative to.
PIXEL_DIMENSION_TAGS = (0xA002, 0xA003)


def output_path(first, out_dir, suffix):
    first = Path(first)
    return Path(out_dir or first.parent) / f"{first.stem}.{suffix}.dng"


def levels(frame):
    """(input black, scale): scale maps the input bit depth onto 16 bits (14-bit -> x4, like Fuji)."""
    black_max = max(frame.black)
    if frame.white <= black_max:
        raise ValueError(f"white ({frame.white}) must be greater than black ({black_max})")
    bits = max(1, int(np.ceil(np.log2(frame.white + 1))))
    return float(np.mean(frame.black)), 2.0 ** (16 - bits)


def to_uint16(img, frame, clip=None):
    """Black-subtracted float image -> (uint16 image, black_out, white_out).

    clip: optional per-channel ceiling (black-subtracted units, see shift.clip_ceiling) the
    image was capped at; white_out is then a per-channel tuple at exactly that level, so a raw
    converter treats every capped sample as clipped."""
    black_in, scale = levels(frame)
    black_out = int(round(black_in * scale))
    # WhiteLevel uses the highest per-channel black (the channel with the least post-subtraction
    # headroom, i.e. the one that saturates first in output units), so a stored value at or above it
    # reliably marks a clipped highlight in every channel, not only the mean one.
    white_out = min(65535, int(round((frame.white - max(frame.black)) * scale)) + black_out)
    if clip is not None:
        white_out = tuple(min(white_out, int(np.floor(c * scale)) + black_out) for c in clip)
    out = np.empty(img.shape, np.uint16)
    for r in range(0, img.shape[0], CHUNK):
        out[r:r + CHUNK] = np.clip(np.rint(img[r:r + CHUNK] * scale + black_out), 0, 65535)
    return out, black_out, white_out


def to_float(img, frame, dtype=np.float32):
    """Black-subtracted image -> float in [0, 1] (white = 1; values above 1 are kept, e.g. 'add').

    The [0, 1] range is sized from levels()'s mean per-channel black, unlike to_uint16's
    WhiteLevel tag, which uses the max per-channel black (see the comment there). The two agree
    on the GFX100S, whose black levels are equal across channels in practice; they would give a
    (very slightly) different white point on a sensor with real per-channel black offsets.
    """
    black_in, _ = levels(frame)
    rng = np.float32(frame.white - black_in)
    out = np.empty(img.shape, dtype)
    for r in range(0, img.shape[0], CHUNK):
        out[r:r + CHUNK] = img[r:r + CHUNK] / rng
    return out


def _scalar(rm_exif, tag):
    """A positive int/float out of an Exif entry's decoded value (a 1-tuple for a scalar RATIONAL,
    SHORT, or LONG tag), or None if the tag is missing or not usable for arithmetic."""
    e = rm_exif.get(tag)
    v = e.value if e else None
    return v[0] if isinstance(v, tuple) and v and isinstance(v[0], (int, float)) and v[0] > 0 else None


def _rescale_focal_plane(exif, rm_exif, out_shape):
    """FocalPlaneX/YResolution in the RAF's Exif describe the embedded JPEG preview
    (PixelX/YDimension), not our output raster -- e.g. 913 px/cm against a 4000x3000 JPEG on a
    GFX100S, meaningless for a 17472x23296 hires output or an 8746x11662 stack. Rescale by
    out_size / PixelDimension, or drop the FocalPlane tags entirely if that isn't computable: a
    stale JPEG-based value would otherwise silently carry through to every DNG we write, whatever
    the output size (write_dng dedupes Exif entries by tag, last occurrence wins, so an override
    here replaces the RAF's original)."""
    out_h, out_w = out_shape
    px, py = (_scalar(rm_exif, t) for t in PIXEL_DIMENSION_TAGS)
    fx, fy = (_scalar(rm_exif, t) for t in FOCAL_PLANE_TAGS[:2])
    if None in (px, py, fx, fy):
        return [e for e in exif if e.tag not in FOCAL_PLANE_TAGS]
    return exif + [rational_entry(0xA20E, fx * out_w / px), rational_entry(0xA20F, fy * out_h / py)]


def base_meta(frame, black, white, n_used, n_total, out_shape, **kw):
    rm = frame.meta
    extra = []
    if 0x132 in rm.ifd0:
        extra.append((0x132, ASCII, 0, rm.ifd0[0x132].value, True))    # DateTime
    if rm.serial:
        extra.append((50735, ASCII, 0, rm.serial, True))               # CameraSerialNumber
    o = rm.ifd0.get(0x112)                                              # Orientation
    orientation = (o.value[0] if o and isinstance(o.value, tuple) and o.value
                   and isinstance(o.value[0], int) else 1)
    exif = _rescale_focal_plane(list(rm.exif.values()), rm.exif, out_shape) + composite_exif(n_used, n_total)
    return DngMeta(
        make=rm.make, model=rm.model, color_matrix_d65=frame.cam_xyz, as_shot_neutral=frame.neutral,
        black=black, white=white, software=SOFTWARE, extra_ifd0=extra, makernote=rm.makernote,
        orientation=orientation, exif=exif, **kw)   # write_dng: last tag wins


def _write(path, img, frame, use_float, n_used, n_total, clip=None, **kw):
    """Write img (black-subtracted float32) as uint16 or float DNG; float32 falls back to float16 at
    4 GiB. Writes to a '<name>.part' file next to `path` and atomically renames it onto `path` only
    on success, so a failed or interrupted write (OSError, a DNG post-write patching error, a
    KeyboardInterrupt) never leaves a partial file at `path` -- the '.part' file is always removed in
    `finally`.
    """
    path = Path(path)
    part = path.with_name(path.name + ".part")
    try:
        if not use_float:
            out, black, white = to_uint16(img, frame, clip)
            write_dng(part, out, base_meta(frame, black, white, n_used, n_total, **kw))
            fmt = "uint16"
        else:
            meta = base_meta(frame, 0, 1, n_used, n_total, **kw)   # built once, reused by both attempts
            fallback = False
            try:
                write_dng(part, to_float(img, frame, np.float32), meta)
                fmt = "float32"
            except DngTooLargeError:
                fallback = True
            if fallback:
                # Outside the except block: while it is live, an exception's traceback keeps every
                # frame on the stack alive, including write_dng's, holding its failed 4 GiB+ float32
                # array argument -- retrying inside the handler would keep that array alive for the
                # whole float16 write too.
                write_dng(part, to_float(img, frame, np.float16), meta)
                fmt = "float16"
        os.replace(part, path)
        return fmt
    finally:
        part.unlink(missing_ok=True)


def write_mask(path, det, names):
    """Repair debug output: a multi-page 8-bit TIFF, page k = frame k's replaced fraction per
    detection cell (255 = fully replaced), page description = the frame's file name."""
    import tifffile
    path = Path(path)
    part = path.with_name(path.name + ".part")
    try:
        with tifffile.TiffWriter(part) as tw:
            for w, name in zip(det.weights, names):
                tw.write(np.rint((1 - w) * 255).astype(np.uint8), description=name)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def run_shift(paths, out_path, mode="hires", normalize=True, use_float=False, jobs=None,
              repair=True, mask_path=None, progress=None):
    """repair applies to mode="hires" with >= 3 frames only (colour mode has no redundant frames
    to compare); info["repair"] is then {"replaced": {frame name: replaced fraction},
    "defective": number of defective photosites}, else None.
    mask_path: also write the repair masks there (see write_mask).

    mode="color" only needs the 4 whole-pixel-offset frames: their PixelShiftOffset is read
    from each RAF's metadata (cheap: no raw-pixel decode) before deciding which paths to decode
    at all, so the other 12 frames are never decoded."""
    paths = [Path(p) for p in paths]
    offsets = []
    for p in paths:
        off = read_raf_meta(p).pixel_shift_offset
        if off is None:
            raise ValueError(f"{p.name}: not a pixel-shift frame")
        offsets.append(off)
    n_total = len(paths)
    if mode == "color":
        keep = [i for i, (ox, oy) in enumerate(offsets) if on_grid(ox) and on_grid(oy)]
        if not keep:
            raise ValueError(f"no frames usable for mode {mode!r}")
        paths = [paths[i] for i in keep]
        offsets = [offsets[i] for i in keep]
    frames = decode_all(paths, jobs, progress)
    grid = grid_for(frames[0].meta.model)
    # every frame here is merged (colour mode kept only its 4 frames above)
    gains = normalization_gains(frames) if normalize else [np.ones((2, 2))] * len(frames)
    clip = clip_ceiling(frames, gains)
    report = None
    if repair and mode == "hires" and len(frames) >= 3:   # fewer: nothing to take a median of
        img, missing, used, det = merge_repaired(frames, offsets, grid, normalize=normalize,
                                                 progress=progress, gains=gains, cap=clip)
        names = [f.path.name for f in frames]
        report = {"replaced": dict(zip(names, (float(v) for v in replaced_fraction(det)))),
                  "defective": det.defective}
        if mask_path is not None:
            write_mask(mask_path, det, names)
        del det
    else:
        img, missing, used = merge_shift(frames, offsets, grid, mode=mode, normalize=normalize,
                                         progress=progress, gains=gains, cap=clip)
    n_used = len(used)
    first = dataclasses.replace(frames[0], raw=np.empty((0, 0), np.uint16))  # done with the raw pixels
    del frames
    h, w = img.shape[:2]
    cx, cy = (16, 12) if mode == "hires" else (8, 6)          # Fuji's default crop margins
    progress_report(progress, "write", 0, 1)
    fmt = _write(out_path, img, first, use_float, n_used, n_total, clip=clip,
                 crop_origin=(cx, cy), crop_size=(w - 2 * cx, h - 2 * cy), out_shape=(h, w))
    progress_report(progress, "write", 1, 1)
    return {"shape": img.shape, "missing": missing, "format": fmt, "repair": report}


def run_stack(paths, out_path, mode, normalize=False, use_float=False, jobs=None, progress=None):
    frames = decode_all(paths, jobs, progress)
    progress_report(progress, "merge", 0, 1)
    m, colors = merge_stack(frames, mode, normalize=normalize)
    progress_report(progress, "merge", 1, 1)
    n = len(frames)
    first = dataclasses.replace(frames[0], raw=np.empty((0, 0), np.uint16))  # done with the raw pixels
    del frames
    # 'add' always writes float, values >1 kept as-is (by design, not clipped). Tested on a
    # 9-frame set: pre-scaling to <=1 with a DNG BaselineExposure tag rendered *darker* in
    # darktable-cli than plain 'average' (mean luma 52 vs 154 for the as-is >1 version, 62 for
    # 'average'); darktable does not apply BaselineExposure, so the as-is semantics are kept.
    auto_float = mode == "add" and not use_float
    left, top, cw, ch = first.crop
    kw = dict(cfa_pattern=tuple(CH[int(c)] for c in colors.ravel()), out_shape=m.shape[:2])
    if cw > 0 and ch > 0:
        kw.update(crop_origin=(left, top), crop_size=(cw, ch))
    progress_report(progress, "write", 0, 1)
    fmt = _write(out_path, m, first, use_float or auto_float, n, n, **kw)
    progress_report(progress, "write", 1, 1)
    return {"shape": m.shape, "float": use_float or auto_float, "auto_float": auto_float, "format": fmt}
