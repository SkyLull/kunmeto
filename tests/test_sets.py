import os
from pathlib import Path

import pytest

from conftest import DATA
from kunmeto import sets, stack
from kunmeto.sets import (ShotInfo, collect_files, find_shift_sets, find_stack_sets, gather,
                          mode_from_jpg, stack_set_from_rafs)


def test_modes_defined_once_in_stack_and_imported_by_sets():
    assert sets.MODES is stack.MODES


def raf(name, count, ab=0, seq=0, shots=0, folder="/a", serial="1"):
    return ShotInfo(Path(folder) / name, "raf", "GFX100S", serial, count, ab, seq, shots)


def jpg(name, count, n, mode, folder="/a", serial="1"):
    return ShotInfo(Path(folder) / name, "jpg", "GFX100S", serial, count, exposure_count=n, multi_mode=mode)


# --- shift sets -------------------------------------------------------------------------------

def test_shift_set_grouped_by_shared_image_count():
    infos = [raf(f"A{k:02d}.RAF", 100, 6, k, 4) for k in (3, 1, 4, 2)]
    infos += [raf("B.RAF", 101)]                                  # unrelated normal shot
    sets, problems = find_shift_sets(infos)
    assert problems == []
    assert len(sets) == 1
    assert [f.sequence for f in sets[0].frames] == [1, 2, 3, 4]


def test_incomplete_shift_set_reported():
    infos = [raf(f"A{k}.RAF", 100, 6, k, 4) for k in (1, 2, 4)]
    sets, problems = find_shift_sets(infos)
    assert sets == []
    assert len(problems) == 1 and "expected 1..4" in problems[0]


def test_shift_set_duplicate_sequence_reported_explicitly():
    infos = [raf(f"A{k}.RAF", 100, 6, seq, 4) for k, seq in enumerate((1, 1, 3, 4), start=1)]
    sets, problems = find_shift_sets(infos)
    assert sets == []
    assert len(problems) == 1 and "duplicate sequence numbers" in problems[0]


def test_shift_set_duplicate_and_missing_reported_together():
    """Sequences [1, 1, 4] for an expected 1..4 set have both a duplicate (1) and gaps (2, 3);
    the problem message must surface both, not just the duplicate."""
    infos = [raf(f"A{k}.RAF", 100, 6, seq, 4) for k, seq in enumerate((1, 1, 4), start=1)]
    sets, problems = find_shift_sets(infos)
    assert sets == []
    assert len(problems) == 1
    assert "duplicate sequence numbers [1]" in problems[0]
    assert "missing [2, 3]" in problems[0]


def test_shift_sets_grouped_per_folder():
    """The same pixel-shift set copied into two folders (e.g. a backup) must stay two sets, not
    merge into one with duplicate sequence numbers."""
    infos = [raf(f"A{k:02d}.RAF", 100, 6, k, 4, folder="/a") for k in (1, 2, 3, 4)]
    infos += [raf(f"A{k:02d}.RAF", 100, 6, k, 4, folder="/b") for k in (1, 2, 3, 4)]
    sets, problems = find_shift_sets(infos)
    assert problems == []
    assert len(sets) == 2
    assert {s.frames[0].path.parent for s in sets} == {Path("/a"), Path("/b")}


def test_shift_sets_grouped_per_serial():
    """Two cameras that happen to land on the same ImageCount must not be merged."""
    infos = [raf(f"A{k:02d}.RAF", 100, 6, k, 4, serial="1") for k in (1, 2, 3, 4)]
    infos += [raf(f"A{k:02d}.RAF", 100, 6, k, 4, serial="2") for k in (1, 2, 3, 4)]
    sets, problems = find_shift_sets(infos)
    assert problems == []
    assert len(sets) == 2
    assert {s.frames[0].serial for s in sets} == {"1", "2"}


def test_files_without_image_count_never_group():
    infos = [raf("A.RAF", None, 6, 1, 1), raf("B.RAF", None), raf("C.RAF", None), jpg("J.JPG", None, 2, 3)]
    assert find_shift_sets(infos) == ([], [])
    assert find_stack_sets(infos) == ([], [])


# --- stack sets --------------------------------------------------------------------------------

def test_stack_set_from_composite_jpg():
    infos = [raf(f"R{c}.RAF", c) for c in range(10, 16)] + [jpg("J.JPG", 15, 3, 3)]
    sets, problems = find_stack_sets(infos)
    assert problems == []
    assert [f.image_count for f in sets[0].frames] == [13, 14, 15]
    assert sets[0].mode == "lighten" and sets[0].warnings == [] and sets[0].mode_warnings == []
    assert sets[0].anchor.path == Path("/a/J.JPG")


def test_stack_set_missing_frames_reported():
    infos = [raf("R14.RAF", 14), raf("R15.RAF", 15), jpg("J.JPG", 15, 3, 3)]
    sets, problems = find_stack_sets(infos)
    assert sets == []
    assert "are missing" in problems[0] and "13" in problems[0]


def test_stack_only_filter_limits_to_given_jpgs():
    infos = [raf(f"R{c}.RAF", c) for c in range(1, 7)] + [jpg("J1.JPG", 3, 3, 3), jpg("J2.JPG", 6, 3, 3)]
    sets, problems = find_stack_sets(infos, only={Path("/a/J2.JPG")})
    assert [s.anchor.path for s in sets] == [Path("/a/J2.JPG")]
    assert problems == []


def test_stack_only_reports_non_composite_paths():
    infos = [raf(f"R{c}.RAF", c) for c in range(1, 4)] + [jpg("J.JPG", 3, 3, 3)]
    sets, problems = find_stack_sets(infos, only={Path("/a/J.JPG"), Path("/a/Other.JPG")})
    assert [s.anchor.path for s in sets] == [Path("/a/J.JPG")]
    assert len(problems) == 1
    assert "not a readable camera multi-exposure composite JPG" in problems[0] and "Other.JPG" in problems[0]


def test_stack_sets_scoped_to_jpg_folder():
    """A composite JPG must not pick up RAFs from a different folder (e.g. a backup copy)."""
    infos = [raf(f"R{c}.RAF", c, folder="/b") for c in range(10, 16)]
    infos += [jpg("J.JPG", 15, 3, 3, folder="/a")]
    sets, problems = find_stack_sets(infos)
    assert sets == []
    assert len(problems) == 1
    assert "are missing" in problems[0] and "/a" in problems[0]


def test_stack_set_counts_wrap_at_15_bits():
    """ImageCount wraps at 2**15; a composite whose count sits just after the wrap must still
    match the RAFs on the other side of it."""
    infos = [raf("R32767.RAF", 32767), raf("R0.RAF", 0), raf("R1.RAF", 1)]
    infos += [jpg("J.JPG", 1, 3, 3)]
    sets, problems = find_stack_sets(infos)
    assert problems == []
    assert [f.image_count for f in sets[0].frames] == [32767, 0, 1]


def test_stack_set_mode2_is_unverified():
    infos = [raf(f"R{c}.RAF", c) for c in range(1, 4)] + [jpg("J.JPG", 3, 3, 2)]
    sets, problems = find_stack_sets(infos)
    assert problems == []
    assert sets[0].mode == "average"
    assert sets[0].warnings == []
    assert any("unverified" in w for w in sets[0].mode_warnings)


def test_mode_mapping():
    assert mode_from_jpg(3) == ("lighten", [])
    mode, warnings = mode_from_jpg(2)
    assert mode == "average" and "unverified" in warnings[0]
    assert mode_from_jpg(99)[0] is None


# --- explicit RAF targets ----------------------------------------------------------------------

def test_raf_targets_become_one_modeless_set():
    s = stack_set_from_rafs([raf("B.RAF", 2), raf("A.RAF", 1)])
    assert [f.path.name for f in s.frames] == ["A.RAF", "B.RAF"] and s.mode is None
    assert s.warnings == []


def test_stack_set_from_rafs_orders_by_image_count():
    s = stack_set_from_rafs([raf("A.RAF", 5), raf("B.RAF", 3), raf("C.RAF", 4)])
    assert [f.image_count for f in s.frames] == [3, 4, 5]


def test_stack_set_from_rafs_warns_on_mixed_serials():
    s = stack_set_from_rafs([raf("Z.RAF", 5, serial="1"), raf("Y.RAF", 3, serial="2")])
    assert any("camera" in w or "serial" in w for w in s.warnings)


def test_stack_set_from_rafs_warns_on_pixel_shift_frames():
    s = stack_set_from_rafs([raf("A.RAF", 1, ab=6, shots=4, seq=1), raf("B.RAF", 2)])
    assert any("pixel-shift" in w for w in s.warnings)


def test_stack_set_from_rafs_dedupes_same_file():
    """The same file passed twice (e.g. once explicitly and once via a glob) must not be
    stacked twice."""
    a = raf("A.RAF", 1)
    dup = ShotInfo(Path("/a/./A.RAF"), "raf", "GFX100S", "1", 1)   # same file, different spelling
    s = stack_set_from_rafs([a, dup, raf("B.RAF", 2)])
    assert [f.path.name for f in s.frames] == ["A.RAF", "B.RAF"]


# --- file collection ---------------------------------------------------------------------------

def test_collect_files(tmp_path):
    for n in ("a.RAF", "b.jpg", "c.txt"):
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.RAF").write_bytes(b"x")

    files, missing = collect_files([tmp_path])
    assert [p.name for p in files] == ["a.RAF", "b.jpg"]
    assert missing == []

    files, missing = collect_files([tmp_path], recursive=True)
    assert [p.name for p in files] == ["a.RAF", "b.jpg", "d.RAF"]


def test_collect_files_explicit_missing_and_uppercase_jpeg(tmp_path):
    explicit = tmp_path / "explicit.bin"           # explicit file, included regardless of suffix
    explicit.write_bytes(b"x")
    sub = tmp_path / "folder"
    sub.mkdir()
    (sub / "X.JPEG").write_bytes(b"x")              # uppercase .JPEG recognized in folder contents
    missing_path = tmp_path / "nope.RAF"            # never created

    files, missing = collect_files([explicit, sub, missing_path])
    assert explicit.resolve() in files
    assert (sub / "X.JPEG").resolve() in files
    assert missing == [missing_path.resolve()]


def test_collect_files_dedupes_relative_and_absolute_spelling(tmp_path, monkeypatch):
    (tmp_path / "a.RAF").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    files, missing = collect_files([tmp_path, Path(".")])
    assert len(files) == 1
    assert missing == []


def test_gather_reports_unreadable_file_as_skipped(tmp_path):
    bad = tmp_path / "bad.RAF"
    bad.write_bytes(b"not a raf")
    infos, skipped = gather([bad])
    assert infos == [] and skipped == [bad]


def test_collect_files_reports_unreadable_folder_as_missing(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores permission bits")
    sub = tmp_path / "locked"
    sub.mkdir()
    (sub / "a.RAF").write_bytes(b"x")
    sub.chmod(0o000)
    try:
        files, missing = collect_files([sub])
    finally:
        sub.chmod(0o755)
    assert files == []
    assert missing == [sub.resolve()]


# --- real folders (data-marked) ------------------------------------------------------------------
