from pathlib import Path
import hashlib
import types

import numpy as np
import pytest
import rawpy

from conftest import DATA, make_frame
from kunmeto.frames import DecodeError, _is_bayer, _visible_crop, decode, decode_all, \
    gains_from_phase_means

# Deterministic stand-ins for "junk bytes" (no os.urandom, so failures are reproducible).
_MAGIC_JUNK = b"FUJIFILMCCD-RAW " + b"\0" * 1008     # looks like a RAF, corrupt past the header
_PLAIN_JUNK = bytes(range(256)) * 4                  # no recognizable magic at all


def test_plane_is_black_subtracted_float():
    f = make_frame(np.full((4, 4), 100), black=60)
    p = f.plane(0, 1)
    assert p.dtype == np.float32 and p.shape == (2, 2) and np.all(p == 40)


def test_neutral_from_camera_wb():
    f = make_frame(np.zeros((2, 2)), wb=(2.0, 1.0, 1.25, 0.0))
    assert f.neutral == (0.5, 1.0, 0.8)


@pytest.mark.parametrize("wb", [(0.0, 1.0, 1.5, 0.0), (2.0, 1.0, 0.0, 0.0)])
def test_neutral_falls_back_to_unity_when_wb_missing(wb):
    f = make_frame(np.zeros((2, 2)), wb=wb)
    assert f.neutral == (1.0, 1.0, 1.0)


def test_decode_junk_file_raises_decode_error(tmp_path):
    bad = tmp_path / "junk.RAF"
    bad.write_bytes(_MAGIC_JUNK)
    with pytest.raises(DecodeError) as excinfo:
        decode(bad)
    assert str(bad) in str(excinfo.value)


def test_decode_all_rejects_non_positive_jobs():
    with pytest.raises(ValueError):
        decode_all([Path("x.RAF")], jobs=0)


def _sizes(**over):
    """A rawpy ImageSizes-like stub for _visible_crop; defaults describe a plausible frame."""
    base = dict(
        top_margin=2, left_margin=0, height=100, width=200,
        crop_left_margin=8, crop_top_margin=8, crop_width=190, crop_height=90,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_visible_crop_relative_to_visible_area():
    assert _visible_crop(_sizes()) == (8, 6, 190, 90)


@pytest.mark.parametrize("over", [
    {"crop_width": 0},
    {"crop_height": 0},
    {"crop_width": -1},
    {"crop_left_margin": 15, "crop_width": 190},   # cl(15) + cw(190) = 205 > width(200)
    {"crop_top_margin": 15, "crop_height": 90},    # ct(13) + ch(90) = 103 > height(100)
])
def test_visible_crop_falls_back_to_no_crop_when_degenerate_or_out_of_bounds(over):
    assert _visible_crop(_sizes(**over)) == (0, 0, 0, 0)


def _rawlike(**over):
    base = dict(raw_pattern=np.zeros((2, 2), np.uint8), num_colors=3, raw_type=rawpy.RawType.Flat)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_is_bayer_true_for_2x2_flat_3_colour():
    assert _is_bayer(_rawlike()) is True


def test_is_bayer_false_for_6x6_pattern():
    assert _is_bayer(_rawlike(raw_pattern=np.zeros((6, 6), np.uint8))) is False


def test_is_bayer_false_when_pattern_is_none():
    assert _is_bayer(_rawlike(raw_pattern=None)) is False


def test_is_bayer_false_for_4_colours():
    assert _is_bayer(_rawlike(num_colors=4)) is False


# --- gains_from_phase_means: shared normalization-gain helper (review item: was duplicated in
# shift.py and stack.py with different too-dark thresholds) --------------------------------------

def test_gains_from_phase_means_scales_to_median():
    means = np.array([[[100.0, 200.0], [200.0, 50.0]], [[200.0, 100.0], [600.0, 200.0]]])
    gains = gains_from_phase_means(means, ["a", "b"])
    expected_a = np.array([[150 / 100, 150 / 200], [400 / 200, 125 / 50]])
    expected_b = np.array([[150 / 200, 150 / 100], [400 / 600, 125 / 200]])
    np.testing.assert_allclose(gains[0], expected_a, rtol=1e-6)
    np.testing.assert_allclose(gains[1], expected_b, rtol=1e-6)


def test_gains_from_phase_means_rejects_non_finite():
    means = np.array([[[np.nan, 200.0], [200.0, 50.0]]])
    with pytest.raises(ValueError, match="too dark to normalize"):
        gains_from_phase_means(means, ["a"])


@pytest.mark.parametrize("value", [0.0, 0.5, 0.999])
def test_gains_from_phase_means_rejects_below_one_dn(value):
    # One unified threshold (< 1.0 DN): shift.py previously used < 1.0, stack.py used <= 0 --
    # a mean like 0.5 must now be rejected by both callers alike.
    means = np.array([[[value, 200.0], [200.0, 50.0]]])
    with pytest.raises(ValueError, match="too dark to normalize"):
        gains_from_phase_means(means, ["a"])


def test_gains_from_phase_means_error_names_the_offending_item():
    means = np.array([[[100.0, 200.0], [200.0, 50.0]], [[0.0, 100.0], [600.0, 200.0]]])
    with pytest.raises(ValueError, match="b: CFA phase mean"):
        gains_from_phase_means(means, ["a", "b"])


def test_decode_all_uses_spawn(monkeypatch):
    import kunmeto.frames as fr
    seen = {}

    class FakePool:
        def __init__(self, max_workers, mp_context, initializer=None):
            seen["method"] = mp_context.get_start_method()

        def map(self, fn, paths):
            return [p for p in paths]

        def shutdown(self, **kw):
            pass

    monkeypatch.setattr(fr, "ProcessPoolExecutor", FakePool)
    assert fr.decode_all(["a", "b"], jobs=2) == [Path("a"), Path("b")]
    assert seen["method"] == "spawn"
