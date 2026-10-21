import numpy as np
import pytest

from conftest import make_frame
from kunmeto.shift import CH, Grid, grid_for, merge_shift, normalization_gains

OFFS = [(0, 0), (0, 1), (1, 0), (1, 1), (0, -2.5), (0, -1.5), (1, -2.5), (1, -1.5),
        (-2.5, 0), (-2.5, 1), (-1.5, 0), (-1.5, 1), (-2.5, -2.5), (-2.5, -1.5), (-1.5, -2.5), (-1.5, -1.5)]
COLORS = np.array([[0, 1], [3, 2]])
BLACK = 60


def synth(scene, grid, gains=None):
    """Sample `scene` (2h x 2w x 3 on the output grid) into 16 Bayer frames per OFFS."""
    h, w = grid.size
    gy, gx = grid.origin
    frames = []
    for k, (ox, oy) in enumerate(OFFS):
        g = 1.0 if gains is None else gains[k]
        raw = np.zeros((h + 8, w + 8), np.uint16)
        for i in range(h + 8):
            for j in range(w + 8):
                # Pixels that land off the grid get the nearest edge value: merge_shift discards
                # them anyway, and real sensors have no zero borders to skew the brightness means.
                Y = min(max(int(2 * (i - oy - gy)), 0), 2 * h - 1)
                X = min(max(int(2 * (j + ox - gx)), 0), 2 * w - 1)
                raw[i, j] = round(scene[Y, X, CH[int(COLORS[i % 2, j % 2])]] * g) + BLACK
        frames.append(make_frame(raw, black=BLACK))
    return frames


def test_hires_reconstructs_scene_exactly():
    grid = Grid(origin=(3.5, 1.0), size=(12, 16))
    scene = np.random.default_rng(0).integers(100, 4000, (24, 32, 3)).astype(np.float32)
    img, missing, used = merge_shift(synth(scene, grid), OFFS, grid, normalize=False)
    assert missing == 0
    assert used == list(range(16))
    np.testing.assert_array_equal(img, scene)


def test_color_mode_uses_whole_pixel_frames_on_1x_grid():
    grid = Grid(origin=(3.5, 1.0), size=(12, 16))
    scene = np.random.default_rng(1).integers(100, 4000, (24, 32, 3)).astype(np.float32)
    img, missing, used = merge_shift(synth(scene, grid), OFFS, grid, mode="color", normalize=False)
    assert img.shape == (12, 16, 3) and missing == 0
    assert used == [0, 1, 2, 3]   # OFFS[:4] are the whole-pixel-offset frames
    # colour mode output pixel (y, x) = raw position (y + 4, x + 1) = hires pixel (2y + 1, 2x)
    np.testing.assert_array_equal(img, scene[1::2, 0::2])


def test_normalization_removes_frame_brightness_steps():
    grid = Grid(origin=(3.5, 1.0), size=(12, 16))
    yy = np.arange(24, dtype=np.float32)[:, None, None]
    scene = np.broadcast_to(np.array([8000, 9000, 7000], np.float32) * (1 + 0.001 * yy / 24), (24, 32, 3)).copy()
    gains = [1 + 0.01 * k for k in range(16)]                    # light fading across the set
    frames = synth(scene, grid, gains)
    raw_img, _, _ = merge_shift(frames, OFFS, grid, normalize=False)
    norm_img, _, _ = merge_shift(frames, OFFS, grid, normalize=True)
    assert np.std(raw_img / scene) > 5e-3
    assert np.std(norm_img / scene) < 1e-3


def test_grid_for_unknown_model():
    assert grid_for("GFX100S").size == (8736, 11648)
    with pytest.raises(ValueError, match="no pixel-shift grid"):
        grid_for("X-T5")


def test_normalization_guards_dark_frame():
    raw = np.full((16, 16), BLACK, np.uint16)
    frame = make_frame(raw, black=BLACK)
    with pytest.raises(ValueError, match="too dark to normalize"):
        normalization_gains([frame])


def test_incomplete_set_reports_missing():
    grid = Grid(origin=(3.5, 1.0), size=(12, 16))
    scene = np.random.default_rng(2).integers(100, 4000, (24, 32, 3)).astype(np.float32)
    frames = synth(scene, grid)
    img, missing, used = merge_shift(frames[1:], OFFS[1:], grid, normalize=False)
    assert missing > 0
    assert used == list(range(15))
    # scene values are always >= 100, so an exact 0 in the output can only be an uncovered sample.
    assert np.sum(img == 0) == missing


def test_mode_bogus_raises():
    grid = Grid(origin=(3.5, 1.0), size=(12, 16))
    scene = np.random.default_rng(3).integers(100, 4000, (24, 32, 3)).astype(np.float32)
    frames = synth(scene, grid)
    with pytest.raises(ValueError, match="unknown mode"):
        merge_shift(frames, OFFS, grid, mode="bogus")


def test_offgrid_offset_raises():
    grid = Grid(origin=(3.5, 1.0), size=(12, 16))
    scene = np.random.default_rng(4).integers(100, 4000, (24, 32, 3)).astype(np.float32)
    frames = synth(scene, grid)
    offs = list(OFFS)
    offs[0] = (0.3, 0)
    with pytest.raises(ValueError, match="does not land on the output grid"):
        merge_shift(frames, offs, grid, normalize=False)


def test_color_mode_no_integer_offsets_raises():
    grid = Grid(origin=(3.5, 1.0), size=(12, 16))
    scene = np.random.default_rng(5).integers(100, 4000, (24, 32, 3)).astype(np.float32)
    frames = synth(scene, grid)
    offs = [(0.5, 0.5)] * len(frames)
    with pytest.raises(ValueError, match="no frames usable"):
        merge_shift(frames, offs, grid, mode="color", normalize=False)


def test_offsets_length_mismatch_raises():
    grid = Grid(origin=(3.5, 1.0), size=(12, 16))
    scene = np.random.default_rng(6).integers(100, 4000, (24, 32, 3)).astype(np.float32)
    frames = synth(scene, grid)
    with pytest.raises(ValueError, match="does not match"):
        merge_shift(frames, OFFS[:-1], grid, normalize=False)


def test_mismatched_frame_shape_raises():
    grid = Grid(origin=(3.5, 1.0), size=(12, 16))
    scene = np.random.default_rng(7).integers(100, 4000, (24, 32, 3)).astype(np.float32)
    frames = synth(scene, grid)
    bad = make_frame(np.zeros((10, 10), np.uint16))
    frames2 = frames[:1] + [bad] + frames[2:]
    with pytest.raises(ValueError, match="shape"):
        merge_shift(frames2, OFFS, grid, normalize=False)


def test_clip_ceiling_is_the_lowest_normalized_clip_level():
    from kunmeto.shift import clip_ceiling
    f = make_frame(np.zeros((4, 4)), black=BLACK, white=15872)
    gains = [np.full((2, 2), 1.0), np.array([[1.0, 0.99], [1.0, 1.0]])]
    cap = clip_ceiling([f, f], gains)
    np.testing.assert_allclose(cap, [0.99 * (15872 - BLACK)] * 3, rtol=1e-6)


def test_capped_merge_makes_clipped_area_uniform():
    # light fading across the set (normalization gains ~1 % apart) over a blown highlight: capped at
    # the common ceiling, every clipped output sample is the same value -- no grid
    from kunmeto.shift import clip_ceiling
    grid = Grid(origin=(3.5, 1.0), size=(12, 16))
    scene = np.full((24, 32, 3), 20000.0, np.float32)            # above white: clips in every frame
    scene[:, :8] = 3000                                          # an unclipped strip for the gains
    frames = synth(scene, grid)
    for k, f in enumerate(frames):                               # clip, then fade the unclipped part
        r = f.raw.astype(np.float64)
        r = np.where(r >= 15872, 15872, BLACK + (r - BLACK) * (1 - 0.001 * k))
        f.raw[:] = np.rint(r).astype(np.uint16)
    gains = normalization_gains(frames)
    cap = clip_ceiling(frames, gains)
    img, _, _ = merge_shift(frames, OFFS, grid, gains=gains, cap=cap)
    clipped = img[4:-4, 20:-4]
    assert np.ptp(clipped) == 0 and clipped.max() == cap[0]
    plain, _, _ = merge_shift(frames, OFFS, grid, gains=gains)
    assert np.ptp(plain[4:-4, 20:-4]) > 10                        # uncapped: the grid
