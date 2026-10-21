"""JIT-compiled array kernels for repair and defect detection: JAX on the CPU (XLA fuses the
element-wise work and spreads it over all cores; ~10x scipy on the blurs and the fill).

float32 throughout (JAX's default; x64 stays off). Semantics match the scipy calls they
replace: blur = scipy.ndimage.gaussian_filter (truncate 4, mode 'reflect' = numpy 'symmetric'
padding); resample = scipy.ndimage.map_coordinates(order=1, mode='nearest') on a regular grid,
done per axis. Results agree with scipy to float32 rounding.

Only regular grids are ever sampled here (phase planes, detection cells and the hires output are
all regular lattices of each other), so interpolation is separable: gather two rows, then two
columns.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

TRUNCATE = 4.0


def _taps(sigma):
    r = int(TRUNCATE * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32), r


def _blur_axis(a, sigma, axis):
    k, r = _taps(sigma)
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = jnp.pad(a, pad, mode="symmetric")
    n = a.shape[axis]
    out = k[0] * jax.lax.slice_in_dim(p, 0, n, axis=axis)
    for i in range(1, 2 * r + 1):
        out = out + k[i] * jax.lax.slice_in_dim(p, i, i + n, axis=axis)
    return out


def _blur2(a, sigma):
    """Gaussian blur over the last two axes."""
    return _blur_axis(_blur_axis(a, sigma, a.ndim - 2), sigma, a.ndim - 1)


def _lin_axis(a, start, step, n, axis):
    """Linear interpolation of `a` along `axis` at start + step * arange(n), edges clamped
    (map_coordinates order=1, mode='nearest')."""
    size = a.shape[axis]
    c = jnp.clip(start + step * jnp.arange(n, dtype=jnp.float32), 0, size - 1)
    i0 = jnp.clip(jnp.floor(c).astype(jnp.int32), 0, max(size - 2, 0))
    f = c - i0
    lo = jnp.take(a, i0, axis=axis)
    hi = jnp.take(a, jnp.minimum(i0 + 1, size - 1), axis=axis)
    shape = [1] * a.ndim
    shape[axis] = n
    return lo + (hi - lo) * f.reshape(shape)


def _resample(a, ys, ystep, ny, xs, xstep, nx):
    return _lin_axis(_lin_axis(a, ys, ystep, ny, a.ndim - 2), xs, xstep, nx, a.ndim - 1)


@partial(jax.jit, static_argnames=("sigma", "ny", "nx"))
def blur_resample(p, sigma, ys, ystep, ny, xs, xstep, nx):
    """Blur a plane, then sample it on a regular grid (detection features)."""
    return _resample(_blur2(p, sigma), ys, ystep, ny, xs, xstep, nx)


@partial(jax.jit, static_argnames=("ny", "nx"))
def resample(a, ys, ystep, ny, xs, xstep, nx):
    return _resample(a, ys, ystep, ny, xs, xstep, nx)


@jax.jit
def neighbour_excess(p):
    """How far each element stands out from its 8 neighbours (edges replicated): max(p - max,
    min - p). int32 in, int32 out."""
    q = jnp.pad(p, 1, mode="edge")
    h, w = p.shape
    views = [q[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
             for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    mx = views[0]
    mn = views[0]
    for v in views[1:]:
        mx = jnp.maximum(mx, v)
        mn = jnp.minimum(mn, v)
    return jnp.maximum(p - mx, mn - p)


def phases(raw):
    """(h, w) -> (4, h/2, w/2): the CFA phase planes raw[pi::2, pj::2], pi-major (h, w even)."""
    h, w = raw.shape
    return raw.reshape(h // 2, 2, w // 2, 2).transpose(1, 3, 0, 2).reshape(4, h // 2, w // 2)


def unphases(ph):
    """Inverse of phases()."""
    _, h2, w2 = ph.shape
    return ph.reshape(2, 2, h2, w2).transpose(2, 0, 3, 1).reshape(2 * h2, 2 * w2)


@partial(jax.jit, donate_argnums=(0, 1))
def accumulate_defect_bits(strong, weak, raw, bit, thr):
    """strong/weak ((4, h/2, w/2) per CFA phase) |= bit where the photosite stands out from its
    same-colour neighbours by > thr / > thr/2 in this frame's raw (h, w)."""
    e = jax.vmap(neighbour_excess)(phases(raw.astype(jnp.int32)))
    strong = strong | jnp.where(e > thr, bit, jnp.uint16(0))
    weak = weak | jnp.where(e > thr / 2, bit, jnp.uint16(0))
    return strong, weak


@partial(jax.jit, static_argnames=("group_masks", "min_frames", "min_groups"))
def qualify_defects(strong, weak, group_masks, min_frames, min_groups):
    """(drop bits, number of defective sites)."""
    n = jax.lax.population_count(strong)
    groups = sum(((strong & jnp.uint16(m)) != 0).astype(jnp.uint8) for m in group_masks)
    site = (n >= min_frames) & (groups >= min_groups)
    return unphases(jnp.where(site, weak, jnp.uint16(0))), site.sum()


@partial(jax.jit, static_argnames=("fill_sigma", "dense"))
def fill_strip(acc, cnt, base, fill_sigma, dense):
    """acc (h, w, 3) float32 weighted sums, cnt (h, w, 3) weights (any float), base (3, h, w)
    -> (h, w, 3) float32.

    Every sample short of full weight is topped up with a prediction: colour-difference
    normalized convolution of the surviving samples where they're dense enough (local weight
    density >= dense), else the base frame's own rendering."""
    def nconv(v, w):
        num, den = _blur2(v * w, fill_sigma), _blur2(w, fill_sigma)
        return num / jnp.maximum(den, 1e-6), den

    acc = jnp.moveaxis(acc, -1, 0)
    cnt = jnp.moveaxis(cnt, -1, 0).astype(jnp.float32)
    val = acc / jnp.maximum(cnt, 1e-6)
    wG = jnp.minimum(cnt[1], 1)
    pG, den = nconv(val[1], wG)
    t = jnp.clip(den / dense, 0, 1)
    lam = jnp.maximum(0, 1 - cnt[1])
    predG = jnp.maximum(t * pG + (1 - t) * base[1], 0)
    G = (acc[1] + lam * predG) / (cnt[1] + lam)
    out = [None, G, None]
    for c in (0, 2):
        wc = jnp.minimum(cnt[c], 1)
        pD, den = nconv(jnp.where(wc > 0, val[c] - G, 0), wc)
        t = jnp.clip(den / dense, 0, 1)
        lam = jnp.maximum(0, 1 - cnt[c])
        pred = jnp.maximum(G + t * pD + (1 - t) * (base[c] - base[1]), 0)
        out[c] = (acc[c] + lam * pred) / (cnt[c] + lam)
    return jnp.stack(out, axis=-1)


@jax.jit
def median_frames(maps):
    """Median over axis 0 (the frames)."""
    return jnp.median(maps, axis=0)


@jax.jit
def zmap(maps, med, grad, a, b, c):
    """Per-frame RMS-over-channels z: maps (N, 3, h, w), med/grad (3, h, w), a/b/c (3,) noise
    params, sigma^2 = a*S + b + (c*G)^2."""
    var = (a[:, None, None] * jnp.maximum(med, 0) + b[:, None, None]
           + (c[:, None, None] * grad) ** 2)
    return jnp.sqrt(((maps - med) ** 2 / var).mean(axis=1))
