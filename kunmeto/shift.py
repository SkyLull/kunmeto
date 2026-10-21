"""Pixel-shift merge: put every frame's samples straight onto the output grid (no interpolation)."""
import math
from dataclasses import dataclass

import numpy as np

from .frames import gains_from_phase_means
from .progress import report

CH = {0: 0, 1: 1, 2: 2, 3: 1}   # LibRaw colour index (R, G, B, G2) -> RGB channel


@dataclass(frozen=True)
class Grid:
    origin: tuple   # (y, x) in raw_image coordinates that maps to output pixel (0, 0) for offset (0, 0)
    size: tuple     # (h, w) in raw pixels; the hires output is twice this


# Calibrated against FUJIFILM Pixel Shift Combiner 1.7 output: our grid == Fuji's 23296x17472 grid.
GRIDS = {"GFX100S": Grid(origin=(3.5, 1.0), size=(8736, 11648))}


def grid_for(model):
    try:
        return GRIDS[model]
    except KeyError:
        raise ValueError(f"no pixel-shift grid calibration for {model!r} "
                         f"(supported: {', '.join(GRIDS)})") from None


def phase_means(frame):
    """Mean black-subtracted value per CFA phase over the central region (2x2 array)."""
    h, w = frame.raw.shape
    y0, x0 = (h // 8) & ~1, (w // 8) & ~1
    sub = frame.raw[y0:h - y0, x0:w - x0]
    # Clipped (near-white) pixels are included in the mean; negligible skew for normal exposures.
    return np.array([[float(sub[pi::2, pj::2].mean()) - frame.black[int(frame.colors[pi, pj])]
                      for pj in (0, 1)] for pi in (0, 1)])


def normalization_gains(frames):
    """Per-frame, per-phase gains that bring every frame to the set's median brightness.

    The CFA is fixed to the sensor, so a sub-pixel scene shift barely changes these means, while
    fading light (a dusk sequence) changes them by ~1 %, which would otherwise become a lattice.
    """
    means = np.stack([phase_means(f) for f in frames])
    return gains_from_phase_means(means, [f.path.name for f in frames])


def _lattice(shape, plane_shape, y0, x0, step):
    """(target slices, plane slices) for acc[y0 + step*a, x0 + step*b] <- plane[a, b], clipped to
    `shape`; None if nothing lands inside."""
    H, W = shape
    a0 = -(y0 // step) if y0 < 0 else 0
    b0 = -(x0 // step) if x0 < 0 else 0
    a1 = min(plane_shape[0], (H - 1 - y0) // step + 1)
    b1 = min(plane_shape[1], (W - 1 - x0) // step + 1)
    if a1 <= a0 or b1 <= b0:
        return None
    ys, xs = y0 + step * a0, x0 + step * b0
    return ((slice(ys, ys + step * (a1 - a0), step), slice(xs, xs + step * (b1 - b0), step)),
            (slice(a0, a1), slice(b0, b1)))


def clip_ceiling(frames, gains):
    """(3,) float32 ceiling for the brightness-normalized samples: the highest level EVERY frame
    can record in every channel, min over frames and CFA phases of gain * (white - black). The
    same value for all three channels: some raw converters read only the first WhiteLevel value,
    and the per-channel ceilings differ by < 1 % anyway.

    Frames are normalized by gains ~1 % apart, so a clipped photosite lands at a slightly
    different level in each frame; merged, those differences form a grid in and around the
    highlights, and with DNG WhiteLevel at the gain-1 level some of them read as unclipped.
    Capping every sample at this ceiling (and writing it as WhiteLevel) makes clipped areas
    identical across frames. Raw values above LibRaw's white level (GFX: up to 16383 vs 15872)
    are capped too."""
    cap = np.full(3, np.inf)
    for f, g in zip(frames, gains):
        for pi in (0, 1):
            for pj in (0, 1):
                ci = int(f.colors[pi, pj])
                cap[CH[ci]] = min(cap[CH[ci]], g[pi, pj] * (f.white - f.black[ci]))
    return np.full(3, cap.min(), np.float32)


def _place(acc_c, cnt_c, plane, y0, x0, step, w=None):
    """acc_c[y0 + step*a, x0 + step*b] += plane[a, b] (times w[a, b] if given; cnt_c gets 1 or
    w), clipped to the accumulator bounds."""
    lat = _lattice(acc_c.shape, plane.shape, y0, x0, step)
    if lat is None:
        return
    sl, src = lat
    if w is None:
        acc_c[sl] += plane[src]
        cnt_c[sl] += 1
    else:
        acc_c[sl] += plane[src] * w[src]
        cnt_c[sl] += w[src]


def on_grid(v, tol=1e-6):
    """Whether `v` is within `tol` of an integer.

    Public: shared by colour-mode frame selection (is this frame's offset a whole pixel? --
    pipeline.run_shift uses this to decide, before decoding, which frames mode="color" needs) and
    _integral (does a computed placement land exactly on the output grid?).
    """
    return abs(v - round(v)) <= tol


def _integral(v):
    if not on_grid(v):
        raise ValueError(f"offset does not land on the output grid: {v}")
    return int(round(v))


def _check_frames(frames, offsets):
    if len(offsets) != len(frames):
        raise ValueError(f"len(offsets)={len(offsets)} does not match len(frames)={len(frames)}")
    if not frames:
        return
    shape0, colors0 = frames[0].raw.shape, frames[0].colors
    for f in frames[1:]:
        if f.raw.shape != shape0:
            raise ValueError(f"{f.path.name}: raw shape {f.raw.shape} != {shape0}")
        if not np.array_equal(f.colors, colors0):
            raise ValueError(f"{f.path.name}: colors {f.colors.tolist()} != {colors0.tolist()}")


def merge_shift(frames, offsets, grid, mode="hires", normalize=True, progress=None, gains=None,
                cap=None):
    """Merge a pixel-shift set.

    Returns (img, missing, used): img is (H, W, 3) float32 in black-subtracted raw units; missing
    is the number of (pixel, channel) samples no frame covered (0 for a complete set); uncovered
    samples are left at 0 in img; used is the sorted list of indices into `frames`/`offsets` this
    call actually placed on the grid (all of them for 'hires'; only the whole-pixel-offset ones
    for 'color' -- callers use len(used) for the DNG's SourceImageNumberOfCompositeImage instead
    of recomputing the selection themselves).
    'hires' puts all frames on the 2x grid; 'color' uses the whole-pixel-offset frames on the 1x grid.
    gains: per-frame (2, 2) normalization gains for the used frames (default: computed, or ones
    with normalize=False); cap: (3,) per-channel ceiling applied after normalization (see
    clip_ceiling; default: none).
    """
    _check_frames(frames, offsets)

    if mode == "hires":
        s, use = 2, list(range(len(frames)))
        gy, gx = grid.origin
    elif mode == "color":
        s = 1
        use = [k for k, (ox, oy) in enumerate(offsets) if on_grid(ox) and on_grid(oy)]
        # 1x pixel (y, x) = hires pixel (2y+1, 2x): the first whole-pixel row/column inside Fuji's grid
        gy, gx = math.ceil(grid.origin[0]), math.ceil(grid.origin[1])
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if not use:
        raise ValueError(f"no frames usable for mode {mode!r}")

    H, W = grid.size[0] * s, grid.size[1] * s
    if gains is None:
        gains = (normalization_gains([frames[k] for k in use]) if normalize
                 else [np.ones((2, 2))] * len(use))
    acc = np.zeros((H, W, 3), np.float32)
    cnt = np.zeros((H, W, 3), np.uint8)
    for i, (g, k) in enumerate(zip(gains, use)):
        report(progress, "merge", i, len(use))
        f, (ox, oy) = frames[k], offsets[k]
        for pi in (0, 1):
            for pj in (0, 1):
                ch = CH[int(f.colors[pi, pj])]
                y0 = _integral(s * (pi - oy - gy))
                x0 = _integral(s * (pj + ox - gx))
                plane = f.plane(pi, pj) * np.float32(g[pi, pj])
                if cap is not None:
                    np.minimum(plane, cap[ch], out=plane)
                _place(acc[..., ch], cnt[..., ch], plane, y0, x0, 2 * s)
    report(progress, "merge", len(use), len(use))
    missing = int((cnt == 0).sum())
    for c in range(3):
        np.divide(acc[..., c], cnt[..., c], out=acc[..., c], where=cnt[..., c] > 0)
    return acc, missing, use
