"""Multi-exposure stacking on the CFA mosaic (visible sensor area), keeping the Bayer layout."""
import numpy as np

from .frames import gains_from_phase_means

MODES = ("add", "average", "lighten", "darken")


def visible_colors(frame):
    """2x2 LibRaw colour index at the visible area's origin."""
    top, left = frame.visible[:2]
    return np.roll(frame.colors, (-top, -left), axis=(0, 1))


def mosaic(frame):
    """Black-subtracted float32 mosaic of the visible area (a single copy of the raw data)."""
    top, left, h, w = frame.visible
    m = frame.raw[top:top + h, left:left + w].astype(np.float32)
    colors = visible_colors(frame)
    for pi in (0, 1):
        for pj in (0, 1):
            m[pi::2, pj::2] -= frame.black[int(colors[pi, pj])]
    return m


def phase_means(frame):
    """Mean black-subtracted value per CFA phase over the visible area (2x2 array), computed
    directly from the raw uint16 data (strided means, no full float mosaic) for a cheap
    normalization pass.
    """
    top, left, h, w = frame.visible
    sub = frame.raw[top:top + h, left:left + w]
    colors = visible_colors(frame)
    return np.array([[float(sub[pi::2, pj::2].mean()) - frame.black[int(colors[pi, pj])]
                      for pj in (0, 1)] for pi in (0, 1)])


def normalization_gains(frames):
    """Per-frame, per-CFA-phase gains that scale every frame to the set's median phase brightness."""
    means = np.stack([phase_means(f) for f in frames])
    return gains_from_phase_means(means, [f.path.name for f in frames])


def quad_luma(m, colors, wb):
    """Per 2x2 quad: white-balanced sum of its four photosites (h//2, w//2).

    Falls back to unweighted (all-ones) weights if any of wb[0:3] (R, G, B) is non-positive, to
    avoid dividing by (or weighting toward) a bogus white balance.
    """
    if wb[0] <= 0 or wb[1] <= 0 or wb[2] <= 0:
        weight = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    else:
        weight = {0: wb[0] / wb[1], 1: 1.0, 2: wb[2] / wb[1], 3: 1.0}
    h2, w2 = m.shape[0] // 2, m.shape[1] // 2
    acc = m[0:2 * h2:2, 0:2 * w2:2] * np.float32(weight[int(colors[0, 0])])
    for pi, pj in ((0, 1), (1, 0), (1, 1)):
        acc += m[pi:2 * h2:2, pj:2 * w2:2] * np.float32(weight[int(colors[pi, pj])])
    return acc


def _scaled_mosaic(frame, gain):
    """mosaic(frame), scaled in place per CFA phase by `gain` (a 2x2 array, or None for no-op).

    mosaic() already allocates a fresh float32 array, so scaling in place (rather than
    `mosaic(f) * gain`) avoids a second full-size copy.
    """
    m = mosaic(frame)
    if gain is not None:
        for pi in (0, 1):
            for pj in (0, 1):
                g = gain[pi, pj]
                if g != 1.0:
                    m[pi::2, pj::2] *= np.float32(g)
    return m


def _apply_trailing(out, m, sel, h2, w2):
    """Decide any trailing row/column left over from an odd visible height/width (not part of any
    full 2x2 quad) by the adjacent quad's winner: the last full quad row/column of `sel`, rather
    than always keeping frame 0.
    """
    if h2 == 0 or w2 == 0:
        return
    H, W = out.shape
    if H > 2 * h2:
        row_sel = np.repeat(sel[-1:, :], 2, axis=1)
        np.copyto(out[2 * h2:2 * h2 + 1, :2 * w2], m[2 * h2:2 * h2 + 1, :2 * w2], where=row_sel)
    if W > 2 * w2:
        col_sel = np.repeat(sel[:, -1:], 2, axis=0)
        np.copyto(out[:2 * h2, 2 * w2:2 * w2 + 1], m[:2 * h2, 2 * w2:2 * w2 + 1], where=col_sel)
    if H > 2 * h2 and W > 2 * w2 and sel[-1, -1]:
        out[2 * h2, 2 * w2] = m[2 * h2, 2 * w2]


def merge_stack(frames, mode, normalize=False):
    """Stack frames; returns (float32 black-subtracted mosaic, 2x2 colour index pattern).

    add/average work per photosite. lighten/darken pick whole 2x2 quads by white-balanced luma
    (frames[0].wb): picking per photosite would mix colour channels from different frames. Any
    trailing row/column left over by an odd visible height/width follows the adjacent quad's
    decision. normalize=True (opt-in; default False) scales each frame to the set's median
    brightness per CFA phase before combining (raises ValueError if a frame is too dark to
    normalize). Saturated photosites are scaled too: a phase with gain > 1 can end up above
    white - black, and one with gain < 1 can drop below it — expected, not a bug.

    The result can contain values below 0 (below-black noise, e.g. from darken) and above
    white - black (sensor overshoot / blown highlights, or normalize gain > 1). Writing the
    result as uint16 (pipeline.to_uint16) clips these to [0, 65535]; writing it as float
    (pipeline.to_float, always used for 'add') keeps them as-is, negatives and values above
    white = 1 included.
    """
    if not frames:
        raise ValueError("no frames to stack")
    if mode not in MODES:
        raise ValueError(f"unknown stack mode {mode!r} (choose from {', '.join(MODES)})")
    colors = visible_colors(frames[0])
    shape = frames[0].visible[2:]
    for f in frames[1:]:
        if f.visible[2:] != shape or not np.array_equal(visible_colors(f), colors):
            raise ValueError(f"{f.path.name}: sensor layout differs from {frames[0].path.name}")

    gains = normalization_gains(frames) if normalize else None

    out = _scaled_mosaic(frames[0], gains[0] if gains is not None else None)
    if mode in ("add", "average"):
        for idx, f in enumerate(frames[1:], start=1):
            out += _scaled_mosaic(f, gains[idx] if gains is not None else None)
        if mode == "average":
            out /= np.float32(len(frames))
        return out, colors

    h2, w2 = out.shape[0] // 2, out.shape[1] // 2
    wb = frames[0].wb
    best = quad_luma(out, colors, wb)
    for idx, f in enumerate(frames[1:], start=1):
        m = _scaled_mosaic(f, gains[idx] if gains is not None else None)
        luma = quad_luma(m, colors, wb)
        sel = luma > best if mode == "lighten" else luma < best
        np.copyto(best, luma, where=sel)
        for pi in (0, 1):
            for pj in (0, 1):
                np.copyto(out[pi:2 * h2:2, pj:2 * w2:2], m[pi:2 * h2:2, pj:2 * w2:2], where=sel)
        _apply_trailing(out, m, sel, h2, w2)
        # Drop this frame's mosaic/luma/selection before the next iteration allocates its own —
        # otherwise both frames' full-size buffers are alive at once at the peak.
        del m, luma, sel
    return out, colors
