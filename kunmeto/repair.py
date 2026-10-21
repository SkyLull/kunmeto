"""Pixel-shift defect repair: find where a frame disagrees with the rest of the set (camera shake
during the e-shutter readout, wind), take those samples out of the merge, and fill the gaps.

1. detect(): every frame's R/G/B planes are blurred and resampled onto one low-res grid (CELL
   hires px per cell, PixelShiftOffset compensated exactly). The reference is the per-cell median
   of all frames, so a minority of moving frames doesn't pollute it. z = residual / sigma with
   sigma^2 = a*S + b + (c*|grad S|)^2, fitted per set to the robust spread across frames (the
   gradient term absorbs sub-pixel misregistration at edges). Clipped areas are never flagged.
2. Weights: 1 at z <= LO, 0 at z >= HI (z dilated by DILATE cells). The least-flagged frame is the
   base frame: it regains weight wherever the other frames are mostly rejected, so an area where
   every frame moved still shows one consistent snapshot instead of a hole.
3. Defective photosites (hot / dead, defects.py) get weight 0 in the frames where they stand out.
4. merge_repaired(): weighted merge, then fill(): every sample short of full weight is topped up
   with a prediction -- colour-difference normalized convolution of the surviving samples where
   they are dense enough, else the base frame's own demosaiced rendering.

Thresholds were tuned on GFX100S sets with camera shake, with wind, and on a clean static scene
(~0.2 % of its samples replaced).
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from . import defects, kernels
from .progress import report
from .shift import CH, _check_frames, _integral, _lattice, _place, normalization_gains

CELL = 8            # hires px per detection cell
SIGMA = 2.0         # detection blur, in phase-plane samples (1 sample = 4 hires px)
LO, HI = 4.0, 8.0   # z ramp: full weight at LO, none at HI
DILATE = 3          # cells
FILL_SIGMA = 1.5    # hires px, normalized-convolution kernel of the fill
DENSE = 0.5         # local weight density above which the fill trusts the surviving samples
STRIP = 1024        # rows per fill strip
MARGIN = 16         # rows of context above/below each strip (> 3 * FILL_SIGMA, + interpolation)
EDGE_TOL = 0.25     # fraction of the fitted edge term actually tolerated (see zscores)
SAT = 0.75          # blurred level (brightest frame) above this fraction of clip = saturated;
                    # low because blurring averages clipped pixels with their neighbours


@dataclass
class Detection:
    weights: np.ndarray   # (N, h, w) float32 per-frame weight on the detection grid
    z: np.ndarray         # (N, h, w) float16 z-scores (before dilation)
    base: int             # index of the base frame
    noise: list           # per channel (a, b, c)
    defective: int = 0    # defective photosites found (see defects.py)


def _origin(f, off, pi, pj, grid):
    ox, oy = off
    gy, gx = grid.origin
    return _integral(2 * (pi - oy - gy)), _integral(2 * (pj + ox - gx))


def _phase_planes(f, off, g, grid, bad=None, cap=None):
    """{rgb channel: [(black-subtracted, gain-normalized plane, y0, x0, (pi, pj)), ...]} for one
    frame; capped at `cap` (per channel, see shift.clip_ceiling); `bad` (raw-shaped bool)
    photosites are patched from their neighbours."""
    planes = {}
    for pi in (0, 1):
        for pj in (0, 1):
            y0, x0 = _origin(f, off, pi, pj, grid)
            ch = CH[int(f.colors[pi, pj])]
            p = f.plane(pi, pj) * np.float32(g[pi, pj])
            if cap is not None:
                np.minimum(p, cap[ch], out=p)
            if bad is not None:
                defects.patch(p, bad[pi::2, pj::2])
            planes.setdefault(ch, []).append((p, y0, x0, (pi, pj)))
    return planes


def feature_maps(frames, offsets, gains, grid, progress=None, cap=None):
    """(N, 3, h, w) float32: each frame's blurred R, G, B on the common detection grid."""
    H, W = grid.size[0] * 2, grid.size[1] * 2
    shape = (H // CELL, W // CELL)
    c0 = (CELL - 1) / 2                     # first cell centre, hires px
    maps = np.zeros((len(frames), 3, *shape), np.float32)
    for k, (f, off, g) in enumerate(zip(frames, offsets, gains)):
        for ch, planes in _phase_planes(f, off, g, grid, cap=cap).items():
            for p, y0, x0, _ in planes:
                m = kernels.blur_resample(p, SIGMA, (c0 - y0) / 4, CELL / 4, shape[0],
                                          (c0 - x0) / 4, CELL / 4, shape[1])
                maps[k, ch] += np.asarray(m) / len(planes)
        report(progress, "detect", k + 1, len(frames))
    return maps


def fit_noise(maps, med, grad, ok, seed=0):
    """Per channel (a, b, c) of sigma^2 = a*S + b + c^2*G^2, fitted (IRLS) to the robust
    across-frame spread over the cells in `ok`."""
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        raise ValueError("no unsaturated area to estimate noise from")
    if idx.size > 400_000:
        idx = np.random.default_rng(seed).choice(idx, 400_000, replace=False)
    params = []
    for c in range(3):
        spread = 1.4826 * np.median(np.abs(maps[:, c].reshape(len(maps), -1)[:, idx]
                                           - med[c].ravel()[idx]), axis=0)
        y = spread.astype(np.float64) ** 2
        A = np.stack([med[c].ravel()[idx], np.ones_like(y), grad[c].ravel()[idx] ** 2], 1)
        w = np.ones_like(y)
        for _ in range(5):                  # IRLS: down-weight outliers (the moving areas)
            sol, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
            r = y - A @ sol
            s = max(1.4826 * np.median(np.abs(r)), 1e-12)
            w = 1 / np.maximum(1, np.abs(r) / (2 * s))
        a, b, c2 = np.maximum(sol, [0.0, 1e-3, 0.0])
        params.append((float(a), float(b), float(np.sqrt(c2))))
    return params


def zscores(maps, sat_level):
    """(z (N, h, w) float32, noise params): per-frame RMS-over-channels z against the median."""
    med = np.asarray(kernels.median_frames(maps))
    # near-clip cells: some frames clip and others don't, which would inflate the noise fit
    sat = gaussian_filter((maps.max(axis=(0, 1)) >= sat_level).astype(np.float32), 2) > 0.01
    grad = np.stack([np.hypot(*np.gradient(med[c])) / CELL for c in range(3)])
    params = fit_noise(maps, med, grad, ~sat)
    # The across-frame spread at edges that the fit sees is not all benign misregistration: it
    # also holds the aliasing mismatch between half-pixel groups that shows up as zipper/lattice
    # in the merge. Tolerating only EDGE_TOL of the fitted edge term flags those edges, while a
    # clean static set still keeps ~99.8 % of its samples.
    a, b, cc = (np.array([p[i] for p in params], np.float32) for i in range(3))
    # Clipped areas are no longer zeroed here: with samples capped at a common ceiling
    # (shift.clip_ceiling) they agree across frames by construction, so motion at the edge of a
    # highlight is detected like anywhere else. `sat` only keeps them out of the noise fit.
    z = np.array(kernels.zmap(maps, med, grad.astype(np.float32), a, b, EDGE_TOL * cc))
    return z, params


def frame_weights(z, lo=LO, hi=HI, dilate=DILATE):
    """(weights (N, h, w) float32, base frame index)."""
    zd = np.stack([maximum_filter(zk, dilate) for zk in z]) if dilate > 1 else z
    W = np.clip((hi - zd) / (hi - lo), 0, 1).astype(np.float32)
    base = int(np.argmin(np.median(z.reshape(len(z), -1), axis=1)))
    others = W.sum(0) - W[base]
    W[base] = np.maximum(W[base], np.clip(1 - others / 4, 0, 1))
    return W, base


def detect(frames, offsets, grid, gains, progress=None, cap=None):
    maps = feature_maps(frames, offsets, gains, grid, progress, cap)
    sat_level = SAT * (min(f.white for f in frames) - max(frames[0].black))
    report(progress, "z-scores", 0, 1)
    z, noise = zscores(maps, sat_level)
    report(progress, "z-scores", 1, 1)
    del maps
    W, base = frame_weights(z)
    return Detection(weights=W, z=z.astype(np.float16), base=base, noise=noise)


def plane_weights(Wk, y0, x0, shape):
    """Bilinear upsampling of one frame's cell weights onto a phase plane's sample positions."""
    c0 = (CELL - 1) / 2
    return np.array(kernels.resample(Wk, (y0 - c0) / CELL, 4 / CELL, shape[0],
                                     (x0 - c0) / CELL, 4 / CELL, shape[1]))   # writable copy


class BaseRender:
    """One frame demosaiced onto the hires grid, rendered strip by strip: bilinear G, then
    bilinear R-G / B-G difference planes (computed once, at the colour's own sample sites)."""

    def __init__(self, planes):
        self.G = [(jnp.asarray(p), y0, x0) for p, y0, x0, _ in planes[1]]
        self.D = {}
        for c in (0, 2):
            p, y0, x0, _ = planes[c][0]
            g = self._g(y0, 4, p.shape[0], x0, 4, p.shape[1])   # G at this colour's sites
            self.D[c] = (jnp.asarray(p) - g, y0, x0)

    def _g(self, ys, ystep, ny, xs, xstep, nx):
        return sum(kernels.resample(p, (ys - y0) / 4, ystep / 4, ny, (xs - x0) / 4, xstep / 4, nx)
                   for p, y0, x0 in self.G) / len(self.G)

    def rows(self, r0, r1, W):
        """(3, r1 - r0, W) for hires rows [r0, r1)."""
        n = r1 - r0
        G = self._g(r0, 1, n, 0, 1, W)
        out = [None, G, None]
        for c, (d, y0, x0) in self.D.items():
            out[c] = G + kernels.resample(d, (r0 - y0) / 4, 0.25, n, -x0 / 4, 0.25, W)
        return jnp.stack(out)


def fill(acc, cnt, base_planes, progress=None):
    """base_planes: the base frame's _phase_planes (see BaseRender)."""
    return _fill(acc, cnt, BaseRender(base_planes), progress)


def _fill(acc, cnt, base, progress=None):
    """Finish a weighted merge in place: acc (H, W, 3) float32 weighted sums -> the image.

    Strip by strip; a strip's result is written back only after the next strip (which reads
    MARGIN rows of its unmodified input) has been computed.
    """
    H, W = acc.shape[:2]
    pending = None
    n_strips = -(-H // STRIP)
    for i, r0 in enumerate(range(0, H, STRIP)):
        r1 = min(H, r0 + STRIP)
        a0, a1 = max(0, r0 - MARGIN), min(H, r1 + MARGIN)
        if (cnt[a0:a1] >= 1).all():
            res = acc[r0:r1] / cnt[r0:r1].astype(np.float32)
        else:
            out = kernels.fill_strip(jnp.asarray(acc[a0:a1]), jnp.asarray(cnt[a0:a1]),
                                     base.rows(a0, a1, W), fill_sigma=FILL_SIGMA, dense=DENSE)
            res = np.asarray(out)[r0 - a0:r1 - a0]
        if pending is not None:
            acc[pending[0]:pending[0] + len(pending[1])] = pending[1]
        pending = (r0, res)
        report(progress, "fill", i + 1, n_strips)
    if pending is not None:
        acc[pending[0]:] = pending[1]
    return acc


def _uncovered(frames, offsets, grid):
    """Number of (pixel, channel) samples no frame covers, whatever the weights."""
    H, W = grid.size[0] * 2, grid.size[1] * 2
    cov = np.zeros((H, W), bool)
    total = 0
    for ch in range(3):
        cov[:] = False
        for f, off in zip(frames, offsets):
            for pi in (0, 1):
                for pj in (0, 1):
                    if CH[int(f.colors[pi, pj])] != ch:
                        continue
                    y0, x0 = _origin(f, off, pi, pj, grid)
                    shape = ((f.raw.shape[0] - pi + 1) // 2, (f.raw.shape[1] - pj + 1) // 2)
                    lat = _lattice(cov.shape, shape, y0, x0, 4)
                    if lat is not None:
                        cov[lat[0]] = True
        total += int((~cov).sum())
    return total


def merge_repaired(frames, offsets, grid, normalize=True, progress=None, gains=None, cap=None):
    """Hires merge with defect repair. Returns (img, missing, used, detection); img as
    shift.merge_shift's (black-subtracted float32). `missing` counts samples no frame covers at
    all; unlike merge_shift, they are filled (from neighbours / the base frame), not left at 0."""
    _check_frames(frames, offsets)
    if len(frames) < 3:
        raise ValueError("repair needs at least 3 frames")
    if gains is None:
        gains = normalization_gains(frames) if normalize else [np.ones((2, 2))] * len(frames)
    det = detect(frames, offsets, grid, gains, progress, cap)
    drop, det.defective = defects.find_defects(frames, offsets, progress=progress)
    H, W = grid.size[0] * 2, grid.size[1] * 2
    acc = np.zeros((H, W, 3), np.float32)
    cnt = np.zeros((H, W, 3), np.float16)
    for k, (f, off, g) in enumerate(zip(frames, offsets, gains)):
        bad = defects.frame_mask(drop, k)
        for ch, planes in _phase_planes(f, off, g, grid, cap=cap).items():
            for p, y0, x0, (pi, pj) in planes:
                w = plane_weights(det.weights[k], y0, x0, p.shape)
                w[bad[pi::2, pj::2]] = 0
                _place(acc[..., ch], cnt[..., ch], p, y0, x0, 4, w)
        report(progress, "merge", k + 1, len(frames))
    missing = _uncovered(frames, offsets, grid)
    base = det.base
    fill(acc, cnt, _phase_planes(frames[base], offsets[base], gains[base], grid,
                                 bad=defects.frame_mask(drop, base), cap=cap), progress)
    if cap is not None:            # the colour-difference fill can overshoot a channel's ceiling
        for r in range(0, H, STRIP):
            np.minimum(acc[r:r + STRIP], cap, out=acc[r:r + STRIP])
    return acc, missing, list(range(len(frames))), det


def replaced_fraction(det):
    """Per frame: fraction of its sample weight taken out of the merge."""
    return 1 - det.weights.reshape(len(det.weights), -1).mean(axis=1)
