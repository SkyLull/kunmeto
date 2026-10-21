import numpy as np

from kunmeto.defects import find_defects, frame_mask, group_of, patch
from kunmeto.repair import merge_repaired
from kunmeto.shift import merge_shift
from test_repair import GRID, OFFS, scene, synth

SITE = (41, 57)          # raw position of the injected defect (away from the edges)
EIGHT = [1, 2, 5, 6, 9, 10, 13, 14]   # 0-based: the 2nd and 3rd shot of every group


def inject(frames, ks, amount=3000):
    # large: synth()'s test scene changes by up to ~600 DN between same-colour neighbours
    for k in ks:
        r = frames[k].raw
        r[SITE] = int(np.clip(int(r[SITE]) + amount, 0, 15872))
    return frames


def bits(ks):
    return sum(1 << k for k in ks)


def test_groups_of_the_16_offsets():
    groups = [group_of(o) for o in OFFS]
    assert sorted(groups) == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    assert groups[:4] == [0] * 4          # each run of 4 shots is one half-pixel group


def test_clean_set_has_no_defects():
    drop, n = find_defects(synth(scene()), OFFS)
    assert n == 0 and not drop.any()


def test_photosite_hot_in_all_frames():
    drop, n = find_defects(inject(synth(scene()), range(16)), OFFS)
    assert n == 1
    assert drop[SITE] == bits(range(16))


def test_photosite_hot_in_the_eight_frame_class_is_dropped_there_only():
    drop, n = find_defects(inject(synth(scene()), EIGHT), OFFS)
    assert n == 1
    assert drop[SITE] == bits(EIGHT)
    assert frame_mask(drop, 1)[SITE] and not frame_mask(drop, 0)[SITE]


def test_outlier_confined_to_one_group_is_not_a_defect():
    # frames 0-3 are 1 px apart: a small scene detail can sit on one photosite in all four
    drop, n = find_defects(inject(synth(scene()), range(4)), OFFS)
    assert n == 0


def test_dead_photosite():
    drop, n = find_defects(inject(synth(scene()), range(16), amount=-3000), OFFS)
    assert n == 1


def test_patch_uses_neighbour_median():
    p = np.arange(25, dtype=np.float32).reshape(5, 5)
    p[2, 2] = 1000
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    patch(p, mask)
    assert p[2, 2] == 12              # median of 6, 7, 8, 11, 13, 16, 17, 18


def test_repair_removes_hot_photosite_from_merge():
    truth = scene()
    frames = inject(synth(scene()), EIGHT)
    plain, _, _ = merge_shift(frames, OFFS, GRID)
    fixed, _, _, det = merge_repaired(frames, OFFS, GRID)
    assert det.defective == 1
    win = np.s_[2 * SITE[0] - 16:2 * SITE[0] + 16, 2 * SITE[1] - 16:2 * SITE[1] + 16]
    err_plain = np.abs(plain[win] - truth[win]).max()
    err_fixed = np.abs(fixed[win] - truth[win]).max()
    assert err_plain > 2000
    assert err_fixed < 400
