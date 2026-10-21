import numpy as np
import pytest

from conftest import DATA, make_frame
from kunmeto import repair
from kunmeto.repair import detect, fill, merge_repaired, _uncovered
from kunmeto.shift import CH, Grid, merge_shift, normalization_gains

OFFS = [(0, 0), (0, 1), (1, 0), (1, 1), (0, -2.5), (0, -1.5), (1, -2.5), (1, -1.5),
        (-2.5, 0), (-2.5, 1), (-1.5, 0), (-1.5, 1), (-2.5, -2.5), (-2.5, -1.5), (-1.5, -2.5), (-1.5, -1.5)]
COLORS = np.array([[0, 1], [3, 2]])
BLACK = 60
GRID = Grid(origin=(3.5, 1.0), size=(96, 128))
PATCH = np.s_[60:120, 80:170]           # hires rows/cols of the "moving" region


def scene(shift_x=0.0):
    """Textured test scene on the hires grid, (2h, 2w, 3); shift_x moves the pattern."""
    h, w = GRID.size
    y = np.arange(2 * h, dtype=np.float32)[:, None]
    x = np.arange(2 * w, dtype=np.float32)[None, :] - shift_x
    lum = 3000 + 1500 * np.sin(x / 5) * np.sin(y / 7) + 500 * np.cos((x + y) / 11)
    return np.stack([lum * 0.6, lum, lum * 0.5], -1)


def synth(scenes, seed=0, noise=True):
    """16 Bayer frames; frame k samples scenes[k] (a list) or `scenes` (one array) at OFFS[k]."""
    rng = np.random.default_rng(seed)
    h, w = GRID.size
    gy, gx = GRID.origin
    i = np.arange(h + 8)[:, None]
    j = np.arange(w + 8)[None, :]
    chan = np.vectorize(lambda c: CH[int(c)])(COLORS[i % 2, j % 2])
    frames = []
    for k, (ox, oy) in enumerate(OFFS):
        sc = scenes[k] if isinstance(scenes, list) else scenes
        Y = np.clip((2 * (i - oy - gy)).astype(int), 0, 2 * h - 1)
        X = np.clip((2 * (j + ox - gx)).astype(int), 0, 2 * w - 1)
        v = sc[Y, X, chan]
        if noise:
            v = v + rng.normal(0, np.sqrt(0.5 * v + 16))
        frames.append(make_frame(np.clip(np.rint(v) + BLACK, 0, 15872), black=BLACK))
    return frames


def moved_scenes(k_moved=4, dx=6.0):
    base, moved = scene(), scene(dx)
    out = []
    for k in range(16):
        s = base.copy()
        if k == k_moved:
            s[PATCH] = moved[PATCH]
        out.append(s)
    return out


def cells(sl):
    return np.s_[sl[0].start // repair.CELL:sl[0].stop // repair.CELL,
                 sl[1].start // repair.CELL:sl[1].stop // repair.CELL]


def _detect(frames):
    return detect(frames, OFFS, GRID, normalization_gains(frames))


def test_clean_set_keeps_nearly_every_sample():
    det = _detect(synth(scene()))
    assert det.weights.shape == (16, 2 * 96 // repair.CELL, 2 * 128 // repair.CELL)
    # outermost cells excluded: synth() repeats edge values there, unlike a real sensor margin
    assert det.weights[:, 2:-2, 2:-2].mean() > 0.999


def test_moving_patch_flagged_in_that_frame_only():
    det = _detect(synth(moved_scenes(k_moved=4)))
    inner = np.s_[PATCH[0].start + 16:PATCH[0].stop - 16, PATCH[1].start + 16:PATCH[1].stop - 16]
    w = det.weights
    assert w[4][cells(inner)].mean() < 0.1
    others = np.delete(w, [4, det.base], axis=0)
    assert others[:, cells(inner)[0], cells(inner)[1]].mean() > 0.95
    ring = 32                                         # dilation + blur reach around the patch
    near = np.s_[PATCH[0].start - ring:PATCH[0].stop + ring, PATCH[1].start - ring:PATCH[1].stop + ring]
    outside = w[4].copy()
    outside[cells(near)] = 1
    assert outside[2:-2, 2:-2].mean() > 0.99


def test_repaired_merge_matches_plain_merge_on_clean_set():
    frames = synth(scene())
    plain, _, _ = merge_shift(frames, OFFS, GRID)
    fixed, missing, used, _ = merge_repaired(frames, OFFS, GRID)
    assert missing == 0 and used == list(range(16))
    interior = np.s_[8:-8, 8:-8]
    rel = np.abs(fixed[interior] - plain[interior]).mean() / plain[interior].mean()
    assert rel < 0.002


def test_repair_reduces_error_in_moving_patch():
    truth = scene()
    frames = synth(moved_scenes(k_moved=4))
    plain, _, _ = merge_shift(frames, OFFS, GRID)
    fixed, _, _, _ = merge_repaired(frames, OFFS, GRID)
    err_plain = np.abs(plain[PATCH] - truth[PATCH]).mean()
    err_fixed = np.abs(fixed[PATCH] - truth[PATCH]).mean()
    assert err_fixed < 0.5 * err_plain


def test_all_frames_moving_falls_back_to_base_frame_without_holes():
    # every frame sees the patch at a different position: no majority, the base frame fills in
    scenes = [scene() for _ in range(16)]
    for k in range(16):
        scenes[k][PATCH] = scene(1.5 * k)[PATCH]
    frames = synth(scenes)
    fixed, _, _, det = merge_repaired(frames, OFFS, GRID)
    assert np.isfinite(fixed).all()
    p = fixed[PATCH]
    assert p.min() >= 0 and p[..., 1].max() < 6000
    # base frame tops up wherever the others are mostly rejected: no cell is ever left empty
    assert det.weights.sum(0).min() >= 1 - 1e-6


def test_fill_in_small_strips_matches_one_pass(monkeypatch):
    rng = np.random.default_rng(3)
    H, W = 200, 120
    val = 1000 + 200 * rng.random((H, W, 3)).astype(np.float32)
    cnt = np.ones((H, W, 3), np.float16)
    cnt[rng.random((H, W, 3)) < 0.2] = 0.25
    acc = val * cnt.astype(np.float32)
    frames = synth(scene(), noise=False)
    planes = repair._phase_planes(frames[0], OFFS[0], np.ones((2, 2)), GRID)
    one = fill(acc.copy(), cnt, planes)
    monkeypatch.setattr(repair, "STRIP", 24)
    strips = fill(acc.copy(), cnt, planes)
    np.testing.assert_allclose(strips, one, rtol=1e-5)


def test_uncovered_counts_like_merge_shift():
    frames = synth(scene(), noise=False)
    _, missing, _ = merge_shift(frames[1:], OFFS[1:], GRID, normalize=False)
    assert missing > 0
    assert _uncovered(frames[1:], OFFS[1:], GRID) == missing
    assert _uncovered(frames, OFFS, GRID) == 0


def test_too_few_frames_raises():
    frames = synth(scene(), noise=False)[:2]
    with pytest.raises(ValueError, match="at least 3"):
        merge_repaired(frames, OFFS[:2], GRID)
