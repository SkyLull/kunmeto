"""Decode RAFs with LibRaw (rawpy): one file per process, OpenMP pinned to 1 thread (see __init__)."""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy

from .progress import report
from .raf import RafMeta, read_raf_meta


class DecodeError(ValueError):
    """A RAF (or the file at a given path) could not be decoded; the message names the path."""


@dataclass(eq=False)  # ndarray fields make the generated __eq__ raise on comparison
class Frame:
    path: Path
    raw: np.ndarray        # full sensor readout incl. margins, uint16
    colors: np.ndarray     # 2x2 LibRaw colour index at raw[0:2, 0:2] (0 R, 1 G, 2 B, 3 G2)
    black: tuple           # black level per colour index 0..3
    white: int
    visible: tuple[int, int, int, int]  # (top, left, height, width) of the visible area within raw
    crop: tuple[int, int, int, int]     # (left, top, width, height), relative to the visible area
    cam_xyz: np.ndarray    # 3x3 XYZ(D65) -> camera, the DNG ColorMatrix
    wb: tuple              # camera white-balance multipliers (R, G, B, G2)
    meta: RafMeta | None

    def plane(self, pi, pj) -> np.ndarray:
        """Black-subtracted float32 CFA phase plane raw[pi::2, pj::2]."""
        return self.raw[pi::2, pj::2].astype(np.float32) - self.black[int(self.colors[pi, pj])]

    @property
    def neutral(self):
        if self.wb[0] == 0 or self.wb[2] == 0:
            return (1.0, 1.0, 1.0)
        return (self.wb[1] / self.wb[0], 1.0, self.wb[1] / self.wb[2])


def _is_bayer(r) -> bool:
    return (
        r.raw_pattern is not None
        and r.raw_pattern.shape == (2, 2)
        and r.num_colors == 3
        and r.raw_type == rawpy.RawType.Flat
    )


def _visible_crop(s) -> tuple[int, int, int, int]:
    """Frame.crop: (left, top, width, height), relative to the visible area, or (0, 0, 0, 0)
    ("no crop"; the pipeline skips it) if the reported crop is degenerate or falls outside the
    visible area.

    crop_left_margin/crop_top_margin are in full raw coordinates (verified against a DNG with
    ActiveArea top=2,left=4 and DefaultCropOrigin (8,8): rawpy reported crop (12,10)); this makes
    the result relative to the visible area (s.top_margin, s.left_margin) instead.
    """
    cl = max(0, s.crop_left_margin - s.left_margin)
    ct = max(0, s.crop_top_margin - s.top_margin)
    cw, ch = s.crop_width, s.crop_height
    if cw <= 0 or ch <= 0 or cl + cw > s.width or ct + ch > s.height:
        return (0, 0, 0, 0)
    return (cl, ct, cw, ch)


def decode(path) -> Frame:
    path = Path(path)
    try:
        with rawpy.imread(str(path)) as r:
            if not _is_bayer(r):
                raise DecodeError(f"{path}: not a 2x2 Bayer RAW (kunmeto supports GFX Bayer sensors)")
            s = r.sizes
            raw = r.raw_image.copy()
            colors = r.raw_colors[:2, :2].copy()
            black = tuple(int(b) for b in r.black_level_per_channel)
            white = int(r.white_level)
            visible = (s.top_margin, s.left_margin, s.height, s.width)
            crop = _visible_crop(s)
            cam_xyz = np.array(r.rgb_xyz_matrix[:3], np.float64)
            wb = tuple(float(v) for v in r.camera_whitebalance)
    except DecodeError:
        raise
    except (rawpy.LibRawError, OSError) as e:
        raise DecodeError(f"{path}: LibRaw could not decode ({e})") from e

    try:
        meta = read_raf_meta(path)
    except ValueError as e:
        raise DecodeError(f"{path}: could not read RAF metadata ({e})") from e

    return Frame(
        path=path, raw=raw, colors=colors, black=black, white=white,
        visible=visible, crop=crop, cam_xyz=cam_xyz, wb=wb, meta=meta,
    )


def gains_from_phase_means(means, names):
    """Per-item, per-CFA-phase gains that bring every item to the set's median value.

    `means` is (N, 2, 2): one black-subtracted phase-mean array per item (frame). `names` are N
    labels, used only to name the offending item in the error. Shared by shift.normalization_gains
    (central region of the raw frame) and stack.normalization_gains (visible area), which compute
    `means` differently but must apply the same threshold and raise the same way: a phase mean
    that's non-finite or below 1.0 DN is too dark to normalize reliably.
    """
    means = np.asarray(means)
    for name, m in zip(names, means):
        bad = ~np.isfinite(m) | (m < 1.0)
        if np.any(bad):
            raise ValueError(
                f"{name}: CFA phase mean {float(m[bad].flat[0]):.2f} too dark to normalize; "
                f"use normalize=False"
            )
    return list(np.median(means, axis=0) / means)


def _worker_init():
    # A spawned child inherits the parent's *global* start method name; if anything in the parent
    # ever touched the default context on Linux (py < 3.14), that name is 'fork' and rawpy's
    # fork check would warn although this process was spawned. Record the truth.
    multiprocessing.set_start_method("spawn", force=True)


def decode_all(paths, jobs=None, progress=None):
    paths = [Path(p) for p in paths]
    if jobs is not None and jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    jobs = max(1, min(jobs or os.cpu_count() or 1, len(paths)))
    out = []
    report(progress, "decode", 0, len(paths))
    if jobs == 1:
        for p in paths:
            out.append(decode(p))
            report(progress, "decode", len(out), len(paths))
        return out
    # spawn, never fork: a forked child inherits LibRaw's OpenMP state and, once a set has been
    # repaired, XLA's thread pool -- both can deadlock it (rawpy warns about the former). An
    # explicit context leaves the global start method unset, so rawpy's check in the child sees
    # no 'fork'. spawn is also the macOS default, i.e. the tested path.
    ex = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_worker_init)
    try:
        for fr in ex.map(decode, paths):      # in order; counts as each one arrives
            out.append(fr)
            report(progress, "decode", len(out), len(paths))
        return out
    except BrokenProcessPool as e:
        raise DecodeError(
            f"a decoder process crashed while decoding {len(paths)} files"
            "; rerun with -j 1 to identify the file"
        ) from e
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
