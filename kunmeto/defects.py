"""Defective photosites (hot / dead) in a pixel-shift set.

A defective photosite is fixed to the sensor, so it stands out from its 8 same-colour neighbours
at the same raw position in many frames, while the sensor moves the scene across it. A scene
detail can't do that across half-pixel groups: frames in different groups are 2.5 px apart, so
a detail lands on a different photosite. A small detail can stay on one photosite within a
group, where the frames are only 1 px apart. So a photosite is defective if it stands out in at
least MIN_FRAMES frames spread over at least MIN_GROUPS of the half-pixel groups.

On the GFX100S most defects show up in only 8 frames: the 2nd and 3rd shot of every group
(whole-pixel step (0, 1) or (1, 0)), thousands of photosites per set; why is unknown. A few are
hot in all 16. Their dot clusters show in Fuji's Combiner output too. Samples are dropped only
in the frames where the photosite stands out; the repair fill covers them
(repair.merge_repaired).
"""
import jax.numpy as jnp
import numpy as np

from . import kernels
from .progress import report

THRESHOLD = 50       # DN over the neighbours' extreme, at 14-bit scale (rescaled by white level)
MIN_FRAMES = 6
MIN_GROUPS = 3
_REF_RANGE = 15872 - 62   # GFX100S 14-bit white - black, the scale THRESHOLD was tuned at


def group_of(offset):
    """Half-pixel group of a PixelShiftOffset (x, y): which axes have a half-pixel part."""
    ox, oy = offset
    return (abs(ox - round(ox)) > 0.25) * 2 + (abs(oy - round(oy)) > 0.25)


def find_defects(frames, offsets, threshold=THRESHOLD, min_frames=MIN_FRAMES,
                 min_groups=MIN_GROUPS, progress=None):
    """(drop, n_sites): drop is a raw-shaped uint16 bit field, bit k set = drop frame k's sample
    at that photosite; n_sites is the number of defective photosites. Supports up to 16 frames."""
    if len(frames) > 16:
        raise ValueError("defect detection supports at most 16 frames")
    rng = min(f.white for f in frames) - max(frames[0].black)
    thr = threshold * rng / _REF_RANGE
    h, w = frames[0].raw.shape
    ph, pw = h + h % 2, w + w % 2            # phase planes need even dimensions
    strong = jnp.zeros((4, ph // 2, pw // 2), jnp.uint16)   # stands out by > thr
    weak = jnp.zeros((4, ph // 2, pw // 2), jnp.uint16)     # > thr / 2 (dropped once qualified)
    for k, f in enumerate(frames):
        raw = f.raw if (ph, pw) == (h, w) else np.pad(f.raw, ((0, ph - h), (0, pw - w)), "edge")
        strong, weak = kernels.accumulate_defect_bits(strong, weak, jnp.asarray(raw),
                                                      jnp.uint16(1 << k), np.float32(thr))
        report(progress, "defects", k + 1, len(frames))
    masks = tuple(sum(1 << k for k, off in enumerate(offsets) if group_of(off) == g)
                  for g in range(4))
    drop, n = kernels.qualify_defects(strong, weak, tuple(m for m in masks if m),
                                      min_frames, min_groups)
    return np.asarray(drop)[:h, :w], int(n)


def frame_mask(drop, k):
    """Raw-shaped bool: frame k's samples to drop."""
    return ((drop >> k) & 1).astype(bool)


def patch(plane, mask):
    """plane with the masked elements replaced by the median of their 8 neighbours (in place)."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return plane
    h, w = plane.shape
    nb = [plane[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)]
          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    plane[ys, xs] = np.median(np.stack(nb), axis=0)
    return plane
