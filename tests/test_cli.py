import argparse
import os
from pathlib import Path

import pytest

from conftest import DATA
from kunmeto import pipeline
from kunmeto.cli import UsageError, _run_sets, main, prompt_mode
from kunmeto.sets import ShiftSet, ShotInfo, StackSet


def _ns(**kw):
    base = dict(output=None, force=False, dry_run=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_prompt_mode_accepts_number_or_name():
    answers = iter(["", "7", "2"])
    assert prompt_mode(lambda _: next(answers), lambda *_: None) == "average"
    assert prompt_mode(lambda _: "darken", lambda *_: None) == "darken"


def test_prompt_mode_eof_is_usage_error():
    def boom(_):
        raise EOFError

    with pytest.raises(UsageError):
        prompt_mode(boom, lambda *_: None)


def test_stack_rejects_mixed_targets(tmp_path, capsys):
    (tmp_path / "a.RAF").write_bytes(b"x")
    (tmp_path / "b.JPG").write_bytes(b"x")
    assert main(["stack", str(tmp_path / "a.RAF"), str(tmp_path / "b.JPG")]) == 2
    assert "mix" in capsys.readouterr().err


def test_missing_path_is_usage_error(tmp_path, capsys):
    assert main(["scan", str(tmp_path / "nope")]) == 2
    assert "no such file or folder" in capsys.readouterr().err


def test_missing_stack_target_is_usage_error_like_scan(tmp_path, capsys):
    assert main(["stack", str(tmp_path / "nope")]) == 2
    assert "no such file or folder" in capsys.readouterr().err


def test_stack_rejects_unreadable_raf(tmp_path, capsys):
    (tmp_path / "a.RAF").write_bytes(b"x")
    assert main(["stack", str(tmp_path / "a.RAF")]) == 2
    assert "not a readable Fujifilm RAF" in capsys.readouterr().err


def test_output_must_be_a_directory(tmp_path, capsys):
    bad = tmp_path / "notadir"
    bad.write_bytes(b"x")
    assert main(["shift", "-o", str(bad), str(tmp_path)]) == 2
    assert "not a folder" in capsys.readouterr().err


def test_jobs_must_be_positive(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(["shift", "-j", "0", str(tmp_path)])
    assert exc_info.value.code == 2


def test_keyboard_interrupt_is_reported(monkeypatch, capsys):
    def boom(a):
        raise KeyboardInterrupt

    monkeypatch.setattr("kunmeto.cli.cmd_scan", boom)
    assert main(["scan", "."]) == 130
    assert "interrupted" in capsys.readouterr().err


# --- shift output naming: color mode must not collide with hires --------------------------------

def _synthetic_shift_set(tmp_path):
    f = ShotInfo(tmp_path / "DSCF0001.RAF", "raf", "GFX100S", "1", 1, 6, 1, 16)
    return ShiftSet([f])


def test_shift_hires_uses_plain_kunmeto_suffix(tmp_path, monkeypatch, capsys):
    s = _synthetic_shift_set(tmp_path)
    monkeypatch.setattr("kunmeto.cli.find_shift_sets", lambda infos: ([s], []))
    assert main(["shift", "--dry-run", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "DSCF0001.kunmeto.dng" in out
    assert "kunmeto-color" not in out


def test_shift_color_mode_uses_distinct_suffix(tmp_path, monkeypatch, capsys):
    s = _synthetic_shift_set(tmp_path)
    monkeypatch.setattr("kunmeto.cli.find_shift_sets", lambda infos: ([s], []))
    assert main(["shift", "--dry-run", "--mode", "color", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "DSCF0001.kunmeto-color.dng" in out


# --- scan: [done] marker for stack sets too -------------------------------------------------------

def _synthetic_stack_scan_set(tmp_path, mode="lighten"):
    f = ShotInfo(tmp_path / "DSCF0001.RAF", "raf", "GFX100S", "1", 1)
    anchor = ShotInfo(tmp_path / "DSCF0001.JPG", "jpg", "GFX100S", "1", 1, exposure_count=1)
    return StackSet([f], mode=mode, anchor=anchor)


def test_scan_marks_stack_set_done_when_its_mode_output_exists(tmp_path, monkeypatch, capsys):
    s = _synthetic_stack_scan_set(tmp_path)
    monkeypatch.setattr("kunmeto.cli.find_shift_sets", lambda infos: ([], []))
    monkeypatch.setattr("kunmeto.cli.find_stack_sets", lambda infos: ([s], []))
    (tmp_path / "DSCF0001.kunmeto-lighten.dng").write_bytes(b"x")
    assert main(["scan", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "[done]" in out


def test_scan_stack_set_not_done_when_output_missing(tmp_path, monkeypatch, capsys):
    s = _synthetic_stack_scan_set(tmp_path)
    monkeypatch.setattr("kunmeto.cli.find_shift_sets", lambda infos: ([], []))
    monkeypatch.setattr("kunmeto.cli.find_stack_sets", lambda infos: ([s], []))
    assert main(["scan", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "[done]" not in out


def test_scan_stack_set_with_unknown_mode_never_marked_done(tmp_path, monkeypatch, capsys):
    s = _synthetic_stack_scan_set(tmp_path, mode=None)
    monkeypatch.setattr("kunmeto.cli.find_shift_sets", lambda infos: ([], []))
    monkeypatch.setattr("kunmeto.cli.find_stack_sets", lambda infos: ([s], []))
    assert main(["scan", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "mode?" in out
    assert "[done]" not in out


# --- shift: surface `missing` (uncovered output samples) ------------------------------------------

def test_shift_missing_samples_warns_and_fails(tmp_path, monkeypatch, capsys):
    s = _synthetic_shift_set(tmp_path)
    monkeypatch.setattr("kunmeto.cli.find_shift_sets", lambda infos: ([s], []))
    monkeypatch.setattr(pipeline, "run_shift",
                         lambda *a, **kw: {"shape": (2, 2, 3), "missing": 5, "format": "uint16"})
    code = main(["shift", "-o", str(tmp_path), str(tmp_path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "5 output samples not covered by any frame (left black)" in err


def test_shift_no_missing_samples_does_not_warn_or_fail(tmp_path, monkeypatch, capsys):
    s = _synthetic_shift_set(tmp_path)
    monkeypatch.setattr("kunmeto.cli.find_shift_sets", lambda infos: ([s], []))
    monkeypatch.setattr(pipeline, "run_shift",
                         lambda *a, **kw: {"shape": (2, 2, 3), "missing": 0, "format": "uint16"})
    code = main(["shift", "-o", str(tmp_path), str(tmp_path)])
    assert code == 0
    assert "not covered" not in capsys.readouterr().err


# --- warnings: mode_warnings vs warnings --------------------------------------------------------

def _synthetic_stack_set():
    f1 = ShotInfo(DATA.parent / "a" / "A.RAF", "raf", "GFX100S", "1", 1)
    f2 = ShotInfo(DATA.parent / "a" / "B.RAF", "raf", "GFX100S", "1", 2)
    return StackSet([f1, f2], mode="average", warnings=["general warning"],
                     mode_warnings=["mode-guess warning"])


def test_mode_override_suppresses_only_mode_warnings(monkeypatch, capsys):
    s = _synthetic_stack_set()
    monkeypatch.setattr("kunmeto.cli.resolve_stack", lambda targets, recursive: ([s], []))
    assert main(["stack", "--dry-run", "--mode", "darken", "x"]) == 0
    out = capsys.readouterr().out
    assert "general warning" in out
    assert "mode-guess warning" not in out


def test_mode_warnings_shown_without_override(monkeypatch, capsys):
    s = _synthetic_stack_set()
    monkeypatch.setattr("kunmeto.cli.resolve_stack", lambda targets, recursive: ([s], []))
    assert main(["stack", "--dry-run", "x"]) == 0
    out = capsys.readouterr().out
    assert "general warning" in out
    assert "mode-guess warning" in out


# --- announce before merging ---------------------------------------------------------------------

def test_run_sets_announces_before_merging(tmp_path, capsys):
    calls = []

    def job(out):
        calls.append(out)
        return {"format": "uint16"}

    first = tmp_path / "DSCF0001.RAF"
    first.write_bytes(b"x")
    code = _run_sets([("set A", first, "kunmeto", job)], _ns())
    assert code == 0
    assert calls == [tmp_path / "DSCF0001.kunmeto.dng"]
    out = capsys.readouterr().out
    assert "merging set A ->" in out
    assert out.index("merging") < out.index("wrote")


# --- output collisions --------------------------------------------------------------------------

def test_run_sets_reports_output_collision(tmp_path, capsys):
    folder_a, folder_b = tmp_path / "A", tmp_path / "B"
    folder_a.mkdir()
    folder_b.mkdir()
    first_a, first_b = folder_a / "DSCF0001.RAF", folder_b / "DSCF0001.RAF"
    first_a.write_bytes(b"x")
    first_b.write_bytes(b"x")
    calls = []

    def job(out):
        calls.append(out)
        return {"format": "uint16"}

    common = tmp_path / "out"
    common.mkdir()
    jobs = [("set A", first_a, "kunmeto", job), ("set B", first_b, "kunmeto", job)]
    code = _run_sets(jobs, _ns(output=str(common)))
    assert code == 1
    assert calls == []
    err = capsys.readouterr().err
    assert "2 sets would write" in err
    assert str(folder_a) in err and str(folder_b) in err
    assert "-o" in err


def test_run_sets_disambiguates_labels_across_folders(tmp_path, capsys):
    folder_a, folder_b = tmp_path / "A", tmp_path / "B"
    folder_a.mkdir()
    folder_b.mkdir()
    first_a, first_b = folder_a / "DSCF0001.RAF", folder_b / "DSCF0002.RAF"
    first_a.write_bytes(b"x")
    first_b.write_bytes(b"x")

    def job(out):
        return {"format": "uint16"}

    jobs = [("DSCF0001.RAF", first_a, "kunmeto", job), ("DSCF0002.RAF", first_b, "kunmeto", job)]
    code = _run_sets(jobs, _ns())
    assert code == 0
    out = capsys.readouterr().out
    assert "A/DSCF0001.RAF" in out
    assert "B/DSCF0002.RAF" in out


def test_run_sets_disambiguates_labels_with_same_leaf_folder_name(tmp_path, capsys):
    """Two card dumps both named e.g. "100_FUJI" under different parents must stay distinguishable:
    a bare parent.name prefix would make both labels read identically."""
    folder_a = tmp_path / "a" / "100_FUJI"
    folder_b = tmp_path / "b" / "100_FUJI"
    folder_a.mkdir(parents=True)
    folder_b.mkdir(parents=True)
    first_a, first_b = folder_a / "DSCF0001.RAF", folder_b / "DSCF0001.RAF"
    first_a.write_bytes(b"x")
    first_b.write_bytes(b"x")

    def job(out):
        return {"format": "uint16"}

    label = "DSCF0001.RAF (16 frames)"
    jobs = [(label, first_a, "kunmeto", job), (label, first_b, "kunmeto", job)]
    code = _run_sets(jobs, _ns())
    assert code == 0
    out = capsys.readouterr().out
    assert f"a{os.sep}100_FUJI{os.sep}{label}" in out
    assert f"b{os.sep}100_FUJI{os.sep}{label}" in out


def test_run_sets_creates_output_folder_once_before_loop(tmp_path, capsys, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    first1, first2 = src / "DSCF0001.RAF", src / "DSCF0002.RAF"
    first1.write_bytes(b"x")
    first2.write_bytes(b"x")
    common = tmp_path / "out"

    calls = []
    orig_mkdir = Path.mkdir

    def counting_mkdir(self, *args, **kwargs):
        calls.append(self)
        return orig_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", counting_mkdir)

    def job(out):
        return {"format": "uint16"}

    jobs = [("s1", first1, "kunmeto", job), ("s2", first2, "kunmeto", job)]
    code = _run_sets(jobs, _ns(output=str(common)))
    assert code == 0
    assert calls == [common]   # created exactly once for the shared -o folder, not once per set


def test_run_sets_does_not_create_output_folder_on_dry_run(tmp_path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    first = src / "DSCF0001.RAF"
    first.write_bytes(b"x")
    common = tmp_path / "out"

    def job(out):
        raise AssertionError("must not run on a dry run")

    code = _run_sets([("s1", first, "kunmeto", job)], _ns(output=str(common), dry_run=True))
    assert code == 0
    assert not common.exists()


# --- failures --------------------------------------------------------------------------------

def test_fail_line_includes_exception_type(tmp_path, capsys):
    def job(out):
        raise ValueError("boom")

    first = tmp_path / "DSCF0001.RAF"
    first.write_bytes(b"x")
    code = _run_sets([("set", first, "kunmeto", job)], _ns())
    assert code == 1
    err = capsys.readouterr().err
    assert "FAIL" in err and "ValueError: boom" in err
    assert "Traceback" not in err


def test_fail_prints_traceback_when_debug_env_set(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("KUNMETO_DEBUG", "1")

    def job(out):
        raise ValueError("boom")

    first = tmp_path / "DSCF0001.RAF"
    first.write_bytes(b"x")
    code = _run_sets([("set", first, "kunmeto", job)], _ns())
    assert code == 1
    err = capsys.readouterr().err
    assert "Traceback" in err


# --- folder stack with RAFs but no composite JPG -----------------------------------------------

# --- scan / shift smoke tests --------------------------------------------------------------------

# --- skip / force end-to-end ----------------------------------------------------------------------
