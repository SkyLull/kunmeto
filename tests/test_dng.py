import os
import struct

import numpy as np
import pytest
import rawpy
import tifffile

from kunmeto.dng import DngMeta, DngTooLargeError, _rationals, composite_exif, short_entry, write_dng

CM = np.array([[1.6212, -0.8423, -0.1583], [-0.4336, 1.2583, 0.1937], [-0.0195, 0.0726, 0.6199]])


def meta(**kw):
    base = dict(make="FUJIFILM", model="GFX100S", color_matrix_d65=CM, as_shot_neutral=(0.5, 1.0, 0.7),
                black=248, white=63484, exif=composite_exif(16, 16))
    base.update(kw)
    return DngMeta(**base)


def raw_ifd(path):
    tf = tifffile.TiffFile(path)
    return tf, tf.pages[0].pages[0]


def test_linear_raw_uint16(tmp_path):
    img = np.random.default_rng(0).integers(248, 63484, (192, 256, 3), dtype=np.uint16)
    p = tmp_path / "lin.dng"
    write_dng(p, img, meta())
    tf, raw = raw_ifd(p)
    np.testing.assert_array_equal(raw.asarray(), img)
    assert raw.photometric == 34892 and raw.compression == 7
    assert 338 not in raw.tags                                    # no bogus ExtraSamples
    assert tf.pages[0].tags[330].dtype == 4                       # SubIFDs typed LONG
    assert 34665 in tf.pages[0].tags                              # Exif IFD pointer
    with rawpy.imread(str(p)) as r:
        assert r.num_colors == 3


def test_cfa_uint16(tmp_path):
    img = np.random.default_rng(1).integers(248, 63484, (192, 256), dtype=np.uint16)
    p = tmp_path / "cfa.dng"
    write_dng(p, img, meta(cfa_pattern=(0, 1, 1, 2)))
    tf, raw = raw_ifd(p)
    np.testing.assert_array_equal(raw.asarray(), img)
    # tifffile returns BYTE-typed tag values as raw bytes, not a decoded tuple; normalize before comparing.
    assert raw.photometric == 32803 and tuple(raw.tags[33422].value) == (0, 1, 1, 2)
    assert 338 not in raw.tags                                    # no bogus ExtraSamples
    assert tf.pages[0].tags[330].dtype == 4                       # SubIFDs typed LONG
    with rawpy.imread(str(p)) as r:
        assert r.raw_pattern.tolist() == [[0, 1], [3, 2]]


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_float_linear_raw(tmp_path, dtype):
    img = np.random.default_rng(2).random((192, 256, 3)).astype(dtype)
    p = tmp_path / "f.dng"
    write_dng(p, img, meta(black=0, white=1))
    tf, raw = raw_ifd(p)
    np.testing.assert_array_equal(raw.asarray(), img)
    assert raw.compression == 8 and raw.predictor == 3           # Adobe Deflate + FP predictor
    assert 338 not in raw.tags                                   # no bogus ExtraSamples
    # tifffile returns BYTE-typed tag values as raw bytes, not a decoded tuple; normalize before comparing.
    assert tuple(tf.pages[0].tags[50707].value) == (1, 4, 0, 0)   # DNGBackwardVersion 1.4 for float
    with rawpy.imread(str(p)):
        pass


def test_float_cfa(tmp_path):
    img = np.random.default_rng(3).random((192, 256)).astype(np.float32)
    p = tmp_path / "fcfa.dng"
    write_dng(p, img, meta(black=0, white=1, cfa_pattern=(0, 1, 1, 2)))
    _, raw = raw_ifd(p)
    np.testing.assert_array_equal(raw.asarray(), img)
    assert raw.photometric == 32803
    assert raw.compression == 8 and raw.predictor == 3
    with rawpy.imread(str(p)):
        pass


def test_rejects_bad_shapes(tmp_path):
    with pytest.raises(ValueError):
        write_dng(tmp_path / "x.dng", np.zeros((4, 4, 2), np.uint16), meta())
    with pytest.raises(ValueError):
        write_dng(tmp_path / "x.dng", np.zeros((4, 4), np.uint16), meta())     # CFA without pattern


def test_rejects_bad_cfa_pattern(tmp_path):
    img = np.zeros((4, 4), np.uint16)
    with pytest.raises(ValueError):
        write_dng(tmp_path / "x.dng", img, meta(cfa_pattern=(0, 1, 3, 2)))     # 3 = LibRaw's G2, not valid DNG


def test_rejects_float_white_not_one(tmp_path):
    img = np.zeros((4, 4, 3), np.float32)
    with pytest.raises(ValueError):
        write_dng(tmp_path / "x.dng", img, meta(black=0, white=2))


def test_exif_dedupe_last_wins(tmp_path):
    img = np.zeros((4, 4, 3), np.uint16)
    p = tmp_path / "x.dng"
    exif = [short_entry(0xA460, 2), short_entry(0xA460, 99), short_entry(0xA461, 16, 16)]
    write_dng(p, img, meta(exif=exif))
    tf, _ = raw_ifd(p)
    value = tf.pages[0].tags[34665].value
    assert value["CompositeImage"] == 99                          # last occurrence of 0xA460 wins
    assert value["SourceImageNumberCompositeImage"] == (16, 16)


def test_makernote_dng_private_data(tmp_path):
    img = np.zeros((4, 4, 3), np.uint16)
    p = tmp_path / "x.dng"
    makernote = b"FUJIFILM" + b"\x00" * 20
    write_dng(p, img, meta(makernote=makernote))
    tf, _ = raw_ifd(p)
    blob = tf.pages[0].tags[50740].value
    assert blob[:10] == b"Adobe\x00MakN"
    (count,) = struct.unpack(">I", blob[10:14])
    body = blob[14:]
    assert count == len(body) == 2 + 4 + len(makernote)            # order(2) + orig offset(4) + makernote


def test_level_and_crop_tags_linear_raw(tmp_path):
    img = np.zeros((8, 8, 3), np.uint16)
    p = tmp_path / "x.dng"
    write_dng(p, img, meta(black=248, white=63484, crop_origin=(1, 2), crop_size=(6, 5)))
    _, raw = raw_ifd(p)
    assert tuple(raw.tags[50713].value) == (1, 1)                          # BlackLevelRepeatDim
    assert tuple(raw.tags[50714].value) == tuple(_rationals([248] * 3))    # BlackLevel
    assert tuple(raw.tags[50717].value) == (63484, 63484, 63484)           # WhiteLevel
    assert tuple(raw.tags[50719].value) == (1, 2)                          # DefaultCropOrigin
    assert tuple(raw.tags[50720].value) == (6, 5)                          # DefaultCropSize


def test_level_and_crop_tags_cfa(tmp_path):
    img = np.zeros((8, 8), np.uint16)
    p = tmp_path / "x.dng"
    write_dng(p, img, meta(black=248, white=63484, cfa_pattern=(0, 1, 1, 2)))
    _, raw = raw_ifd(p)
    assert tuple(raw.tags[50713].value) == (2, 2)                          # BlackLevelRepeatDim
    assert tuple(raw.tags[50714].value) == tuple(_rationals([248] * 4))    # BlackLevel
    assert raw.tags[50717].value == 63484                                 # WhiteLevel, 1 sample (scalar)
    assert tuple(raw.tags[50719].value) == (0, 0)                          # DefaultCropOrigin
    assert tuple(raw.tags[50720].value) == (8, 8)                          # DefaultCropSize


def test_too_large_raises_and_cleans_up(tmp_path, monkeypatch):
    """A struct.error that hits once the file is actually near the 4 GiB limit -> DngTooLargeError."""
    p = tmp_path / "big.dng"
    real_write = tifffile.TiffWriter.write
    calls = []

    def boom(self, *a, **kw):
        calls.append(1)
        if len(calls) == 1:
            return real_write(self, *a, **kw)  # let the small preview write succeed normally
        # Simulate the real tifffile failure mode: tiles already streamed past the offset limit.
        # os.truncate is sparse on APFS/most filesystems, so this doesn't actually use 4 GiB of disk.
        os.truncate(p, 2 ** 32)
        raise struct.error("'I' format requires 0 <= number <= 4294967295")

    monkeypatch.setattr(tifffile.TiffWriter, "write", boom)
    img = np.zeros((4, 4, 3), np.uint16)
    with pytest.raises(DngTooLargeError):
        write_dng(p, img, meta())
    assert not p.exists()


def test_struct_error_without_overflow_is_plain_value_error(tmp_path, monkeypatch):
    """A struct.error from an unrelated bad tag value (file stays small) -> plain ValueError, not DngTooLargeError."""
    p = tmp_path / "bad.dng"

    def boom(*_a, **_kw):
        raise struct.error("argument out of range")

    monkeypatch.setattr(tifffile.TiffWriter, "write", boom)
    img = np.zeros((4, 4, 3), np.uint16)
    with pytest.raises(ValueError) as exc_info:
        write_dng(p, img, meta())
    assert not isinstance(exc_info.value, DngTooLargeError)
    assert not p.exists()


def test_rejects_negative_crop_origin(tmp_path):
    img = np.zeros((4, 4, 3), np.uint16)
    with pytest.raises(ValueError) as exc_info:
        write_dng(tmp_path / "x.dng", img, meta(crop_origin=(-1, 0)))
    assert not isinstance(exc_info.value, DngTooLargeError)


def test_rejects_non_positive_crop_size(tmp_path):
    img = np.zeros((4, 4, 3), np.uint16)
    with pytest.raises(ValueError):
        write_dng(tmp_path / "x.dng", img, meta(crop_size=(0, 4)))


def test_orientation_tag_defaults_to_normal(tmp_path):
    img = np.zeros((4, 4, 3), np.uint16)
    p = tmp_path / "o.dng"
    write_dng(p, img, meta())
    tf, _ = raw_ifd(p)
    assert tf.pages[0].tags[274].value == 1


def test_orientation_tag_uses_meta_value(tmp_path):
    img = np.zeros((4, 4, 3), np.uint16)
    p = tmp_path / "o.dng"
    write_dng(p, img, meta(orientation=6))
    tf, _ = raw_ifd(p)
    assert tf.pages[0].tags[274].value == 6


@pytest.mark.parametrize("bad", [0, -1, 9, 99])
def test_orientation_tag_clamps_out_of_range(tmp_path, bad):
    img = np.zeros((4, 4, 3), np.uint16)
    p = tmp_path / "o.dng"
    write_dng(p, img, meta(orientation=bad))
    tf, _ = raw_ifd(p)
    assert tf.pages[0].tags[274].value == 1


def test_rejects_bad_black(tmp_path):
    img = np.zeros((4, 4, 3), np.uint16)
    with pytest.raises(ValueError):
        write_dng(tmp_path / "x.dng", img, meta(black=-1))
    with pytest.raises(ValueError):
        write_dng(tmp_path / "x.dng", img, meta(black=float("nan")))
