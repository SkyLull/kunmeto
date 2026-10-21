import io

import numpy as np

from kunmeto.progress import SHIFT_REPAIR, Bar, report
from kunmeto.repair import merge_repaired
from test_repair import GRID, OFFS, scene, synth


def test_report_none_is_a_no_op():
    report(None, "decode", 1, 16)


def test_fraction_is_cumulative_over_phases():
    bar = Bar("x", {"a": 1, "b": 3}, stream=io.StringIO())
    assert bar.fraction("a", 0, 4) == 0
    assert bar.fraction("a", 4, 4) == 0.25
    assert bar.fraction("b", 1, 2) == 0.625
    assert bar.fraction("b", 2, 2) == 1


def test_bar_draws_phase_and_percentage_and_clears():
    out = io.StringIO()
    bar = Bar("[1/3] DSCF0103", {"decode": 1, "fill": 1}, stream=out, width=80, min_interval=0)
    bar("fill", 3, 6)
    line = out.getvalue()
    assert line.startswith("\r[1/3] DSCF0103") and " 75%" in line and "fill 3/6" in line
    assert len(line) <= 80
    bar.close()
    assert out.getvalue().endswith("\r")


def test_merge_repaired_reports_every_repair_phase_in_order():
    seen = []
    merge_repaired(synth(scene()), OFFS, GRID,
                   progress=lambda phase, done, total: seen.append((phase, done, total)))
    phases = list(dict.fromkeys(p for p, _, _ in seen))
    assert phases == ["detect", "z-scores", "defects", "merge", "fill"]
    assert all(p in SHIFT_REPAIR for p in phases)
    for phase in phases:                             # each phase ends complete
        last = [s for s in seen if s[0] == phase][-1]
        assert last[1] == last[2]
