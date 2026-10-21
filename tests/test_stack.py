import numpy as np
import pytest

from conftest import make_frame
from kunmeto.stack import merge_stack, normalization_gains, quad_luma, visible_colors

B = 60


def quad(r, g1, g2, b):
    """A 2x2 RGGB mosaic tile, raw units incl. black."""
    return np.array([[r + B, g1 + B], [g2 + B, b + B]])


def test_average_and_add():
    a = make_frame(np.tile(quad(100, 200, 200, 50), (2, 2)), black=B)
    b = make_frame(np.tile(quad(300, 400, 400, 150), (2, 2)), black=B)
    avg, colors = merge_stack([a, b], "average")
    np.testing.assert_array_equal(avg, np.tile(quad(200, 300, 300, 100), (2, 2)) - B)
    add, _ = merge_stack([a, b], "add")
    np.testing.assert_array_equal(add, np.tile(quad(400, 600, 600, 200), (2, 2)) - B)
    assert colors.tolist() == [[0, 1], [3, 2]]


def test_lighten_takes_whole_quad_by_luma():
    # Frame a has the brighter red photosite, but frame b's quad is brighter overall.
    a = make_frame(quad(100, 50, 50, 10), black=B, wb=(1.0, 1.0, 1.0, 0.0))
    b = make_frame(quad(90, 80, 80, 10), black=B, wb=(1.0, 1.0, 1.0, 0.0))
    out, _ = merge_stack([a, b], "lighten")
    np.testing.assert_array_equal(out, quad(90, 80, 80, 10) - B)
    out, _ = merge_stack([a, b], "darken")
    np.testing.assert_array_equal(out, quad(100, 50, 50, 10) - B)


def test_visible_area_and_pattern_follow_margins():
    raw = np.arange(6 * 6).reshape(6, 6) + 100
    f = make_frame(raw, black=0, visible=(1, 1, 4, 4))
    out, colors = merge_stack([f], "average")
    np.testing.assert_array_equal(out, raw[1:5, 1:5])
    assert colors.tolist() == [[2, 3], [1, 0]]          # RGGB shifted by one row and column


def test_rejects_unknown_mode_and_mismatched_frames():
    a = make_frame(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        merge_stack([a], "screen")
    with pytest.raises(ValueError):
        merge_stack([a, make_frame(np.zeros((6, 6)))], "average")


def test_visible_colors_helper():
    f = make_frame(np.zeros((4, 4)), visible=(0, 1, 4, 3))
    assert visible_colors(f).tolist() == [[1, 0], [2, 3]]


def test_empty_frame_list_raises():
    with pytest.raises(ValueError, match="no frames to stack"):
        merge_stack([], "average")


# --- code-review fixes ---------------------------------------------------------------------

def test_quad_luma_falls_back_to_unweighted_wb_guard():
    m = np.array([[10.0, 20.0], [20.0, 5.0]], dtype=np.float32)
    colors = np.array([[0, 1], [3, 2]])
    weighted = quad_luma(m, colors, wb=(2.0, 1.0, 1.5, 0.0))
    assert float(weighted[0, 0]) == pytest.approx(10 * 2.0 + 20 * 1.0 + 20 * 1.0 + 5 * 1.5)
    for bad_wb in [(0.0, 1.0, 1.5, 0.0), (2.0, 0.0, 1.5, 0.0), (2.0, 1.0, 0.0, 0.0), (-1.0, 1.0, 1.5, 0.0)]:
        unweighted = quad_luma(m, colors, wb=bad_wb)
        assert float(unweighted[0, 0]) == pytest.approx(10 + 20 + 20 + 5)


def test_lighten_wb_weighting_flips_the_winner():
    # Plain (unweighted) sums: a=130 < b=200, so b would win without WB weighting.
    # WB-weighted (R x2.0, B x1.5): a=235 > b=215, so a should win instead.
    a = make_frame(quad(100, 10, 10, 10), black=B, wb=(2.0, 1.0, 1.5, 0.0))
    b = make_frame(quad(10, 90, 90, 10), black=B, wb=(2.0, 1.0, 1.5, 0.0))
    out, _ = merge_stack([a, b], "lighten")
    np.testing.assert_array_equal(out, quad(100, 10, 10, 10) - B)


def test_darken_keeps_below_black_negative_values():
    raw_a = np.full((2, 2), 90, np.uint16)   # black=100 -> -10 per photosite
    raw_b = np.full((2, 2), 100, np.uint16)  # black=100 -> 0 per photosite
    a = make_frame(raw_a, black=100, wb=(1.0, 1.0, 1.0, 0.0))
    b = make_frame(raw_b, black=100, wb=(1.0, 1.0, 1.0, 0.0))
    out, _ = merge_stack([a, b], "darken")
    np.testing.assert_array_equal(out, np.full((2, 2), -10, np.float32))
    assert (out < 0).all()


def test_normalization_gains_are_computed_per_phase():
    a = make_frame(np.tile(quad(100, 200, 200, 50), (2, 2)), black=B)
    b = make_frame(np.tile(quad(200, 100, 600, 200), (2, 2)), black=B)
    gains = normalization_gains([a, b])
    # a_true=(100,200,200,50), b_true=(200,100,600,200); n=2, so per-phase median == elementwise mean.
    expected_gain_a = np.array([[150 / 100, 150 / 200], [400 / 200, 125 / 50]])
    expected_gain_b = np.array([[150 / 200, 150 / 100], [400 / 600, 125 / 200]])
    np.testing.assert_allclose(gains[0], expected_gain_a, rtol=1e-6)
    np.testing.assert_allclose(gains[1], expected_gain_b, rtol=1e-6)


def test_normalize_per_phase_removes_gain_differences():
    a = make_frame(np.tile(quad(100, 200, 200, 50), (2, 2)), black=B)
    b = make_frame(np.tile(quad(200, 100, 600, 200), (2, 2)), black=B)
    out, _ = merge_stack([a, b], "average", normalize=True)
    expected = np.tile(quad(150, 150, 400, 125), (2, 2)) - B
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-2)


def test_normalization_gains_guards_dark_frame():
    raw = np.full((4, 4), B, np.uint16)
    frame = make_frame(raw, black=B)
    with pytest.raises(ValueError, match="too dark to normalize"):
        normalization_gains([frame])


def test_merge_stack_normalize_guards_dark_frame():
    raw = np.full((4, 4), B, np.uint16)
    frame = make_frame(raw, black=B)
    with pytest.raises(ValueError, match="too dark to normalize"):
        merge_stack([frame], "average", normalize=True)


def test_lighten_darken_odd_height_and_width_use_adjacent_quad_for_trailing():
    # 5x3 visible area: two 2x2 quad-rows (rows 0-1, 2-3) x one quad-col (cols 0-1),
    # a trailing row (4), a trailing column (2), and a trailing corner (4, 2).
    raw_a = np.array([
        [100, 50, 1000],
        [50, 10, 3000],
        [90, 80, 5000],
        [80, 10, 7000],
        [9000, 9100, 9200],
    ], dtype=np.uint16)
    raw_b = np.array([
        [90, 80, 2000],
        [80, 10, 4000],
        [100, 50, 6000],
        [50, 10, 8000],
        [9500, 9300, 9600],
    ], dtype=np.uint16)
    a = make_frame(raw_a, black=0, wb=(1.0, 1.0, 1.0, 0.0))
    b = make_frame(raw_b, black=0, wb=(1.0, 1.0, 1.0, 0.0))

    lighten, _ = merge_stack([a, b], "lighten")
    expected_lighten = np.array([
        [90, 80, 2000],
        [80, 10, 4000],
        [90, 80, 5000],
        [80, 10, 7000],
        [9000, 9100, 9200],
    ], dtype=np.float32)
    np.testing.assert_array_equal(lighten, expected_lighten)

    darken, _ = merge_stack([a, b], "darken")
    expected_darken = np.array([
        [100, 50, 1000],
        [50, 10, 3000],
        [100, 50, 6000],
        [50, 10, 8000],
        [9500, 9300, 9600],
    ], dtype=np.float32)
    np.testing.assert_array_equal(darken, expected_darken)


def test_lighten_with_offset_visible_moves_whole_quads():
    raw_a = np.zeros((6, 6), np.uint16)
    raw_b = np.zeros((6, 6), np.uint16)
    raw_a[1:5, 1:5] = [[100, 50, 100, 50],
                        [50, 10, 50, 10],
                        [100, 50, 100, 50],
                        [50, 10, 50, 10]]
    raw_b[1:5, 1:5] = [[90, 80, 90, 80],
                        [80, 10, 80, 10],
                        [90, 80, 90, 80],
                        [80, 10, 80, 10]]
    # margins carry decoy extreme values that must never leak into the output.
    raw_a[0, :] = raw_a[:, 0] = raw_a[5, :] = raw_a[:, 5] = 60000
    raw_b[0, :] = raw_b[:, 0] = raw_b[5, :] = raw_b[:, 5] = 0
    a = make_frame(raw_a, black=0, wb=(1.0, 1.0, 1.0, 0.0), visible=(1, 1, 4, 4))
    b = make_frame(raw_b, black=0, wb=(1.0, 1.0, 1.0, 0.0), visible=(1, 1, 4, 4))
    out, colors = merge_stack([a, b], "lighten")
    expected = np.array([[90, 80, 90, 80],
                          [80, 10, 80, 10],
                          [90, 80, 90, 80],
                          [80, 10, 80, 10]], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)
    assert colors.tolist() == [[2, 3], [1, 0]]


def test_lighten_darken_side_by_side_quads_pick_different_frames():
    # 2x4 mosaic: two quad-columns (cols 0-1, cols 2-3), each won by a different frame.
    raw_a = np.array([[100, 50, 190, 180],
                       [50, 10, 180, 20]], dtype=np.uint16)
    raw_b = np.array([[90, 80, 200, 150],
                       [80, 10, 150, 20]], dtype=np.uint16)
    a = make_frame(raw_a, black=0, wb=(1.0, 1.0, 1.0, 0.0))
    b = make_frame(raw_b, black=0, wb=(1.0, 1.0, 1.0, 0.0))

    lighten, _ = merge_stack([a, b], "lighten")
    expected_lighten = np.array([[90, 80, 190, 180],
                                  [80, 10, 180, 20]], dtype=np.float32)
    np.testing.assert_array_equal(lighten, expected_lighten)

    darken, _ = merge_stack([a, b], "darken")
    expected_darken = np.array([[100, 50, 200, 150],
                                 [50, 10, 150, 20]], dtype=np.float32)
    np.testing.assert_array_equal(darken, expected_darken)
