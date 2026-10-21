"""The JAX kernels must match the scipy calls they replaced (to float32 rounding)."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from kunmeto import kernels


@pytest.mark.parametrize("sigma", [1.5, 2.0])
def test_blur_matches_gaussian_filter(sigma):
    a = np.random.default_rng(0).random((37, 53), dtype=np.float32) * 1000
    got = np.asarray(kernels.blur_resample(a, sigma, 0.0, 1.0, 37, 0.0, 1.0, 53))
    np.testing.assert_allclose(got, gaussian_filter(a, sigma), rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("ys,ystep,xs,xstep", [(-1.75, 2.0, 0.25, 2.0), (1.875, 0.5, -0.875, 0.5),
                                               (-0.75, 0.25, 0.5, 0.25)])
def test_resample_matches_map_coordinates(ys, ystep, xs, xstep):
    a = np.random.default_rng(1).random((20, 30), dtype=np.float32) * 100
    ny, nx = 40, 50                         # runs past both edges: clamping ('nearest')
    got = np.asarray(kernels.resample(a, ys, ystep, ny, xs, xstep, nx))
    yy, xx = np.meshgrid(ys + ystep * np.arange(ny), xs + xstep * np.arange(nx), indexing="ij")
    want = map_coordinates(a, [yy, xx], order=1, mode="nearest")
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-3)


def test_neighbour_excess():
    p = np.full((7, 7), 100, np.int32)
    p[2, 2] = 400
    p[4, 5] = 10
    e = np.asarray(kernels.neighbour_excess(p))
    assert e[2, 2] == 300 and e[4, 5] == 90
    assert e[2, 3] == 0                          # next to the hot one: max(100-400, 100-100)
