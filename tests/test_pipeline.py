import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rawpy
import tifffile

from conftest import DATA, make_frame
from kunmeto import pipeline
from kunmeto.dng import DngTooLargeError, _rationals
from kunmeto.pipeline import levels, output_path, run_shift, run_stack, to_uint16
from kunmeto.raf import Entry
from kunmeto.shift import Grid


def stub_meta(model="GFX100S", make="FUJIFILM", serial="12345", offset=(0.0, 0.0)):
    """A minimal stand-in for RafMeta exposing just what base_meta/run_shift touch (per review: a
    SimpleNamespace is enough, base_meta only uses attribute access and dict-like ifd0/exif)."""
    return SimpleNamespace(model=model, make=make, serial=serial, ifd0={}, exif={}, makernote=b"",
                            pixel_shift_offset=offset)


def test_output_path(tmp_path):
    assert output_path(Path("/x/DSCF0121.RAF"), None, "kunmeto") == Path("/x/DSCF0121.kunmeto.dng")
    assert output_path(Path("/x/DSCF0156.RAF"), tmp_path, "kunmeto-lighten") == tmp_path / "DSCF0156.kunmeto-lighten.dng"


def test_levels_follow_bit_depth():
    assert levels(make_frame(np.zeros((2, 2)), black=62, white=15872)) == (62.0, 4.0)     # 14-bit
    assert levels(make_frame(np.zeros((2, 2)), black=248, white=65535)) == (248.0, 1.0)   # 16-bit


def test_levels_rejects_white_not_above_black():
    f = make_frame(np.zeros((2, 2)), black=100, white=100)
    with pytest.raises(ValueError, match="white"):
        levels(f)


def test_to_uint16_matches_fuji_scaling():
    f = make_frame(np.zeros((2, 2)), black=62, white=15872)
    img = np.array([[[0.0, 100.0, 15810.0]]], np.float32)
    out, black, white = to_uint16(img, f)
    assert black == 248 and white == (15872 - 62) * 4 + 248
    assert out.tolist() == [[[248, 648, 63488]]]


def test_to_uint16_white_out_uses_max_black():
    # Per-channel blacks differ (mean 61.25, max 65): WhiteLevel must use the max, not the mean.
    import dataclasses
    f = dataclasses.replace(make_frame(np.zeros((2, 2)), black=60, white=15872), black=(60, 62, 58, 65))
    img = np.zeros((1, 1, 3), np.float32)
    _, black, white = to_uint16(img, f)
    black_in, scale = levels(f)
    assert black == round(black_in * scale)
    assert white == round((15872 - 65) * scale) + black


# --- review round: atomic writes, lean float fallback ------------------------------------------

def test_write_atomic_no_partial_on_failure(tmp_path, monkeypatch):
    def boom(path, img, meta):
        Path(path).write_bytes(b"junk")  # simulate a write that got partway before failing
        raise OSError("disk gone")

    monkeypatch.setattr(pipeline, "write_dng", boom)
    frame = make_frame(np.zeros((4, 4), np.uint16), meta=stub_meta())
    out = tmp_path / "x.dng"
    with pytest.raises(OSError):
        pipeline._write(out, np.zeros((4, 4, 3), np.float32), frame, False, 1, 1, out_shape=(4, 4))
    assert not out.exists()
    assert not (tmp_path / "x.dng.part").exists()


def test_float_fallback_writes_float16_after_dng_too_large(tmp_path, monkeypatch):
    calls = []

    def fake_write_dng(path, img, meta):
        calls.append(img.dtype)
        if len(calls) == 1:
            raise DngTooLargeError("too big")
        Path(path).write_bytes(b"ok")

    monkeypatch.setattr(pipeline, "write_dng", fake_write_dng)
    frame = make_frame(np.zeros((4, 4), np.uint16), meta=stub_meta())
    out = tmp_path / "x.dng"
    fmt = pipeline._write(out, np.zeros((4, 4, 3), np.float32), frame, True, 1, 1, out_shape=(4, 4))
    assert fmt == "float16"
    assert calls == [np.float32, np.float16]
    assert out.exists()
    assert not (tmp_path / "x.dng.part").exists()


# --- review round: clear errors -----------------------------------------------------------------

def test_run_shift_rejects_non_pixel_shift_frame(tmp_path, monkeypatch):
    frame = make_frame(np.zeros((8, 8), np.uint16), meta=stub_meta(offset=None))
    monkeypatch.setattr(pipeline, "read_raf_meta", lambda p: stub_meta(offset=None))
    monkeypatch.setattr(pipeline, "decode_all", lambda paths, jobs=None, progress=None: [frame])
    with pytest.raises(ValueError, match="not a pixel-shift frame"):
        pipeline.run_shift(["a.RAF"], tmp_path / "x.dng")


# --- review round: synthetic run_shift / run_stack coverage ------------------------------------

def _shift_frames(offsets, grid, black=60, white=15872):
    """Minimal synthetic Bayer frames for merge_shift via run_shift: pixel content is irrelevant to
    these tests (they check plumbing -- n_used, crop, Exif -- not reconstructed values), so every
    frame is just a uniformly-lit array large enough for `grid`, tagged with its own offset."""
    h, w = grid.size
    frames = []
    for ox, oy in offsets:
        raw = np.full((h + 8, w + 8), 1000, np.uint16)
        frames.append(make_frame(raw, black=black, white=white, meta=stub_meta(offset=(ox, oy))))
    return frames


def _patch_shift_metadata(monkeypatch, paths, offsets):
    """Stub pipeline.read_raf_meta (review item 2: PixelShiftOffset is read from metadata before
    any decode) so these synthetic tests don't touch real files; keyed by path so a test can
    check decode_all only receives the paths run_shift decided to keep."""
    by_path = {str(p): off for p, off in zip(paths, offsets)}
    monkeypatch.setattr(pipeline, "read_raf_meta", lambda p: stub_meta(offset=by_path[str(p)]))


def test_run_shift_color_mode_n_used_counts_near_integer_offsets(tmp_path, monkeypatch):
    # Reuses shift.on_grid's tolerance (review item 7): the second offset is off by 5e-7, within
    # on_grid's 1e-6 tolerance (so merge_shift itself also treats it as on-grid and uses it), but
    # `float(x).is_integer()` would reject it and undercount n_used to 3 instead of 4.
    grid = Grid(origin=(3.5, 1.0), size=(40, 48))
    monkeypatch.setattr(pipeline, "grid_for", lambda model: grid)
    offsets = [(0.0, 0.0), (0.0, 1.0 + 5e-7), (1.0, 0.0), (1.0, 1.0)]
    paths = ["a", "b", "c", "d"]
    _patch_shift_metadata(monkeypatch, paths, offsets)
    frames = _shift_frames(offsets, grid)

    def fake_decode_all(kept, jobs=None, progress=None):
        assert [str(p) for p in kept] == paths     # all four offsets are on-grid: none skipped
        return frames

    monkeypatch.setattr(pipeline, "decode_all", fake_decode_all)
    out = tmp_path / "c.dng"
    run_shift(paths, out, mode="color", normalize=False)
    tf = tifffile.TiffFile(out)
    exif = tf.pages[0].tags[34665].value
    assert exif["SourceImageNumberCompositeImage"] == (4, 4)


def test_run_shift_color_mode_decodes_only_whole_pixel_frames(tmp_path, monkeypatch):
    """Efficiency (review item 2): mode="color" must decode only the whole-pixel-offset frames,
    never the other (half-pixel-offset) frames of the set -- decode is the expensive step."""
    grid = Grid(origin=(3.5, 1.0), size=(40, 48))
    monkeypatch.setattr(pipeline, "grid_for", lambda model: grid)
    on_grid_offsets = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    off_grid_offsets = [(0.0, -2.5), (0.0, -1.5), (1.0, -2.5), (1.0, -1.5)]
    offsets = on_grid_offsets + off_grid_offsets
    paths = [f"p{i}" for i in range(len(offsets))]
    _patch_shift_metadata(monkeypatch, paths, offsets)
    frames = _shift_frames(on_grid_offsets, grid)

    def fake_decode_all(kept, jobs=None, progress=None):
        assert [str(p) for p in kept] == paths[:4]
        return frames

    monkeypatch.setattr(pipeline, "decode_all", fake_decode_all)
    out = tmp_path / "c.dng"
    info = run_shift(paths, out, mode="color", normalize=False)
    assert info["shape"] == (40, 48, 3)
    tf = tifffile.TiffFile(out)
    exif = tf.pages[0].tags[34665].value
    assert exif["SourceImageNumberCompositeImage"] == (8, 4)   # 8 total frames, 4 used


def test_run_shift_color_mode_no_on_grid_frames_raises_before_decoding(tmp_path, monkeypatch):
    grid = Grid(origin=(3.5, 1.0), size=(40, 48))
    monkeypatch.setattr(pipeline, "grid_for", lambda model: grid)
    paths = ["a", "b"]
    _patch_shift_metadata(monkeypatch, paths, [(0.5, 0.5), (0.5, 0.5)])

    def fail_decode_all(paths, jobs=None):
        raise AssertionError("must not decode when no frame is usable for mode=color")

    monkeypatch.setattr(pipeline, "decode_all", fail_decode_all)
    with pytest.raises(ValueError, match="no frames usable"):
        run_shift(paths, tmp_path / "x.dng", mode="color", normalize=False)


def _focal_plane_meta(offset=(0.0, 0.0), px=4000, py=3000, fx=913.0, fy=913.0, unit=3):
    """A stub meta with a JPEG-derived PixelX/YDimension and FocalPlaneX/YResolution, as a real RAF
    reports them (they describe the embedded preview JPEG, not our raw output)."""
    meta = stub_meta(offset=offset)
    meta.exif = {
        0xA002: Entry(0xA002, 4, 1, struct.pack("<I", px), (px,)),
        0xA003: Entry(0xA003, 4, 1, struct.pack("<I", py), (py,)),
        0xA20E: Entry(0xA20E, 5, 1, struct.pack("<2I", int(fx * 1000), 1000), (fx,)),
        0xA20F: Entry(0xA20F, 5, 1, struct.pack("<2I", int(fy * 1000), 1000), (fy,)),
        0xA210: Entry(0xA210, 3, 1, struct.pack("<H", unit), (unit,)),
    }
    return meta


def test_run_shift_focal_plane_rescaled_to_output_size(tmp_path, monkeypatch):
    # FocalPlaneX/YResolution (913 px/cm) describe the 4000x3000 embedded JPEG, not our hires
    # output: the correct value is old * out_size / PixelDimension, not the RAF's original.
    grid = Grid(origin=(3.5, 1.0), size=(40, 48))
    monkeypatch.setattr(pipeline, "grid_for", lambda model: grid)
    meta = _focal_plane_meta()
    monkeypatch.setattr(pipeline, "read_raf_meta", lambda p: meta)
    raw = np.full((grid.size[0] + 8, grid.size[1] + 8), 1000, np.uint16)
    frame = make_frame(raw, black=60, white=15872, meta=meta)
    monkeypatch.setattr(pipeline, "decode_all", lambda paths, jobs=None, progress=None: [frame])
    out = tmp_path / "h.dng"
    run_shift(["a"], out, normalize=False)
    out_h, out_w = 2 * grid.size[0], 2 * grid.size[1]      # hires doubles both axes
    tf = tifffile.TiffFile(out)
    exif = tf.pages[0].tags[34665].value
    xn, xd = exif["FocalPlaneXResolution"]
    yn, yd = exif["FocalPlaneYResolution"]
    assert xn / xd == pytest.approx(913.0 * out_w / 4000, rel=1e-4)
    assert yn / yd == pytest.approx(913.0 * out_h / 3000, rel=1e-4)
    assert exif["FocalPlaneResolutionUnit"] == 3                          # untouched


def test_run_stack_focal_plane_rescaled_to_output_size(tmp_path, monkeypatch):
    meta = _focal_plane_meta()
    frames = [make_frame(np.full((4, 4), 1000, np.uint16), black=60, white=15872, meta=meta)
              for _ in range(2)]
    monkeypatch.setattr(pipeline, "decode_all", lambda paths, jobs=None, progress=None: frames)
    out = tmp_path / "s.dng"
    run_stack(["a", "b"], out, "average", jobs=1)
    tf = tifffile.TiffFile(out)
    exif = tf.pages[0].tags[34665].value
    xn, xd = exif["FocalPlaneXResolution"]
    yn, yd = exif["FocalPlaneYResolution"]
    assert xn / xd == pytest.approx(913.0 * 4 / 4000, rel=1e-4)
    assert yn / yd == pytest.approx(913.0 * 4 / 3000, rel=1e-4)


def test_focal_plane_tags_dropped_when_pixel_dimension_missing(tmp_path, monkeypatch):
    meta = stub_meta()
    meta.exif = {
        0xA20E: Entry(0xA20E, 5, 1, struct.pack("<2I", 913000, 1000), (913.0,)),
        0xA20F: Entry(0xA20F, 5, 1, struct.pack("<2I", 913000, 1000), (913.0,)),
        0xA210: Entry(0xA210, 3, 1, struct.pack("<H", 3), (3,)),
    }   # no PixelXDimension/PixelYDimension -> can't compute the rescale
    frames = [make_frame(np.full((4, 4), 1000, np.uint16), black=60, white=15872, meta=meta)
              for _ in range(2)]
    monkeypatch.setattr(pipeline, "decode_all", lambda paths, jobs=None, progress=None: frames)
    out = tmp_path / "s.dng"
    run_stack(["a", "b"], out, "average", jobs=1)
    tf = tifffile.TiffFile(out)
    exif = tf.pages[0].tags[34665].value
    assert "FocalPlaneXResolution" not in exif
    assert "FocalPlaneYResolution" not in exif
    assert "FocalPlaneResolutionUnit" not in exif


def test_run_stack_add_is_always_float_white_one(tmp_path, monkeypatch):
    frames = [make_frame(np.full((4, 4), 1000, np.uint16), black=60, white=15872, meta=stub_meta())
              for _ in range(2)]
    monkeypatch.setattr(pipeline, "decode_all", lambda paths, jobs=None, progress=None: frames)
    out = tmp_path / "a.dng"
    info = run_stack(["a", "b"], out, "add", jobs=1)
    assert info["float"] is True and info["auto_float"] is True
    tf = tifffile.TiffFile(out)
    raw = tf.pages[0].pages[0]
    assert raw.dtype in (np.float16, np.float32)
    assert raw.tags[50717].value == 1                          # WhiteLevel, float DNGs use white = 1


def test_run_stack_dng_tags_readback(tmp_path, monkeypatch):
    frames = [make_frame(np.full((4, 4), 1000, np.uint16), black=60, white=15872, meta=stub_meta())
              for _ in range(3)]
    monkeypatch.setattr(pipeline, "decode_all", lambda paths, jobs=None, progress=None: frames)
    out = tmp_path / "s.dng"
    run_stack(["a", "b", "c"], out, "average", jobs=1)

    black_in, scale = levels(frames[0])
    black_out = round(black_in * scale)
    white_out = min(65535, round((frames[0].white - max(frames[0].black)) * scale) + black_out)

    tf = tifffile.TiffFile(out)
    raw = tf.pages[0].pages[0]
    assert tuple(raw.tags[50714].value) == tuple(_rationals([black_out] * 4))    # BlackLevel
    assert raw.tags[50717].value == white_out                                    # WhiteLevel
    assert tuple(raw.tags[50719].value) == (0, 0)                                # DefaultCropOrigin
    assert tuple(raw.tags[50720].value) == (4, 4)                                # DefaultCropSize
    assert tuple(raw.tags[33422].value) == (0, 1, 1, 2)                          # CFAPattern (G2 -> 1)
    exif = tf.pages[0].tags[34665].value
    assert exif["SourceImageNumberCompositeImage"] == (3, 3)


def test_to_uint16_white_level_at_the_clip_ceiling():
    frame = make_frame(np.zeros((4, 4), np.uint16), black=62, white=15872)
    img = np.full((2, 2, 3), 15000.0, np.float32)
    out, black, white = to_uint16(img, frame, clip=np.full(3, 15000.0, np.float32))
    assert white == (15000 * 4 + black,) * 3
    assert (out >= white[0]).all()                 # capped samples read as clipped
