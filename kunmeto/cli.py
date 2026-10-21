"""kunmeto command line: scan / shift / stack."""
import argparse
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from . import __version__, pipeline
from .progress import SHIFT_PLAIN, SHIFT_REPAIR, STACK, Bar
from .raf import PIXEL_SHIFT
from .sets import (MODES, SUFFIXES, collect_files, find_shift_sets, find_stack_sets, gather,
                   info_from_file, stack_set_from_rafs)

EXIT_CODES_EPILOG = (
    "exit codes: 0 ok, 1 one or more sets failed or were skipped due to a problem, "
    "2 usage error, 130 interrupted (Ctrl-C)"
)


class UsageError(Exception):
    pass


def positive_int(s):
    """argparse type for -j/--jobs: rejects 0 or negative with the usual argparse usage error."""
    n = int(s)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def prompt_mode(input_fn=input, out=print):
    out("Multi-exposure mode: " + "  ".join(f"[{i}] {m}" for i, m in enumerate(MODES, 1)))
    while True:
        try:
            a = input_fn("mode> ").strip().lower()
        except EOFError:
            raise UsageError("no mode chosen")
        if a.isdigit() and 1 <= int(a) <= len(MODES):
            return MODES[int(a) - 1]
        if a in MODES:
            return a
        out(f"answer 1-{len(MODES)} or one of: {', '.join(MODES)}")


def _gather(paths, recursive):
    """ShotInfos from folders/files; a mistyped path is a usage error, unreadable files are counted."""
    files, missing = collect_files(paths, recursive)
    if missing:
        raise UsageError(f"{missing[0]}: no such file or folder")
    infos, skipped = gather(files)
    if skipped:
        print(f"({len(skipped)} file(s) skipped: not a readable Fujifilm RAF/JPG)")
    return infos


def _span(frames):
    names = [f.path.name for f in frames]
    return names[0] if len(names) == 1 else f"{names[0]} … {names[-1]}"


def _classify(targets):
    kinds = set()
    for t in map(Path, targets):
        if not t.exists():
            raise UsageError(f"{t}: no such file or folder")
        if t.is_dir():
            kinds.add("dir")
        elif t.is_file() and t.suffix.lower() in SUFFIXES:
            kinds.add(SUFFIXES[t.suffix.lower()])
        else:
            raise UsageError(f"{t}: not a folder, RAF or JPG")
    if len(kinds) > 1:
        raise UsageError(f"don't mix target kinds in one call ({', '.join(sorted(kinds))})")
    return kinds.pop()


def resolve_stack(targets, recursive):
    kind = _classify(targets)
    if kind == "raf":
        paths = [Path(t).resolve() for t in targets]
        infos = [info_from_file(t) for t in paths]
        bad = [t for t, i in zip(paths, infos) if i is None]
        if bad:
            raise UsageError(f"{bad[0]}: not a readable Fujifilm RAF")
        return [stack_set_from_rafs(infos)], []
    if kind == "dir":
        infos = _gather(targets, recursive)
        sets, problems = find_stack_sets(infos)
        if (not sets and not problems
                and any(i.kind == "raf" and i.auto_bracketing != PIXEL_SHIFT for i in infos)):
            problems = ["no camera composite JPG found; pass the RAFs themselves with --mode"]
        return sets, problems
    jpgs = [Path(t) for t in targets]
    # find_stack_sets reports every `only` JPG that doesn't become a set.
    return find_stack_sets(_gather({j.parent for j in jpgs}, False), only=jpgs)


def _validate_output(a):
    """-o must be a folder (or not exist yet, to be created later); checked up front, before any
    gathering/decoding work, so a typo is reported immediately."""
    out = getattr(a, "output", None)
    if out and Path(out).exists() and not Path(out).is_dir():
        raise UsageError(f"{out}: not a folder")


def repair_summary(rep):
    """One line from run_shift's repair report (see pipeline.run_shift)."""
    fr = rep["replaced"]
    total = sum(fr.values()) / max(len(fr), 1)
    worst = sorted(((v, k) for k, v in fr.items() if v >= 0.01), reverse=True)[:3]
    most = ", ".join(f"{k} {100 * v:.1f}%" for v, k in worst)
    return (f"repair: {100 * total:.1f}% of samples replaced" + (f" (most: {most})" if most else "")
            + f"; {rep['defective']} defective photosites masked")


def _run_sets(jobs_list, a):
    """jobs_list: (label, first_frame_path, suffix, callable(out_path[, progress])[, weights]).
    Returns exit code. With `weights` (a progress phase table, see progress.py) and stderr on a
    terminal, the job is called with progress=Bar(...) and a progress line is drawn per set.

    Output paths are computed for every set up front: N sets that would write the same path
    (e.g. same-named frames from different folders sharing one -o) are reported and all skipped,
    rather than one silently overwriting another. When the jobs span more than one folder, each
    label is prefixed with its source folder's path relative to the sets' common ancestor (not
    just the folder's own name), so two folders that happen to share a leaf name (e.g. two card
    dumps both named "100_FUJI") still read as distinct in the collision report and everywhere
    else.
    """
    if not jobs_list:
        return 0
    jobs_list = [(label, first, suffix, job, rest[0] if rest else None)
                 for label, first, suffix, job, *rest in jobs_list]
    folders = {Path(first).parent for _, first, _, _, _ in jobs_list}
    if len(folders) > 1:
        common = Path(os.path.commonpath([str(f) for f in folders]))
        jobs_list = [(f"{Path(first).parent.relative_to(common)}/{label}", first, suffix, job, w)
                     for label, first, suffix, job, w in jobs_list]
    outs = [pipeline.output_path(first, a.output, suffix) for _, first, suffix, _, _ in jobs_list]
    by_out = defaultdict(list)
    for idx, out in enumerate(outs):
        by_out[out].append(idx)

    skip_idx = set()
    failed = 0
    for out, idxs in sorted(by_out.items(), key=lambda kv: str(kv[0])):
        if len(idxs) > 1:
            folder_names = sorted({str(Path(jobs_list[i][1]).parent) for i in idxs})
            print(f"!  {len(idxs)} sets would write {out}: {' and '.join(folder_names)}; "
                  f"run them with different -o", file=sys.stderr)
            skip_idx.update(idxs)
            failed += 1

    if a.output and not a.dry_run:
        try:
            Path(a.output).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise UsageError(f"{a.output}: could not create output folder ({e})") from e

    show_bar = sys.stderr.isatty() and not getattr(a, "no_progress", False)
    for i, (label, first, suffix, job, weights) in enumerate(jobs_list):
        if i in skip_idx:
            continue
        out = outs[i]
        if out.exists() and not a.force:
            print(f"skip  {label}: {out} exists (use --force)")
            continue
        if a.dry_run:
            print(f"would write {out}  <-  {label}")
            continue
        print(f"merging {label} -> {out}", flush=True)
        t = time.time()
        bar = (Bar(f"[{i + 1}/{len(jobs_list)}] {Path(first).stem}", weights)
               if show_bar and weights else None)
        try:
            info = job(out, progress=bar) if bar else job(out)
        except Exception as e:  # keep going with the other sets
            if bar:
                bar.close()
            failed += 1
            print(f"FAIL  {label}: {type(e).__name__}: {e}", file=sys.stderr)
            if os.environ.get("KUNMETO_DEBUG"):
                traceback.print_exc(file=sys.stderr)
            continue
        finally:
            if bar:
                bar.close()        # clear the bar line before any report (also on Ctrl-C)
        print(f"wrote {out}  ({info['format']}, {time.time() - t:.0f}s)  <-  {label}")
        rep = info.get("repair")
        if rep is not None:
            print(f"   {repair_summary(rep)}")
        missing = info.get("missing", 0)
        if missing:
            how = "filled from neighbours" if rep is not None else "left black"
            print(f"!  {label}: {missing} output samples not covered by any frame ({how})",
                  file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def cmd_scan(a):
    infos = _gather(a.paths, a.recursive)
    shift, p1 = find_shift_sets(infos)
    stack, p2 = find_stack_sets(infos)
    for s in shift:
        # scan doesn't know which --mode a later `shift` will use; "kunmeto" is hires' (the
        # default) suffix, so this reports whether the default run's output already exists.
        out = pipeline.output_path(s.frames[0].path, None, "kunmeto")
        print(f"pixel shift     {_span(s.frames)}  ({len(s.frames)} frames)  -> {out.name}"
              + ("  [done]" if out.exists() else ""))
    for s in stack:
        mode = s.mode or "mode?"
        done = ""
        if s.mode:
            out = pipeline.output_path(s.frames[0].path, None, f"kunmeto-{s.mode}")
            done = "  [done]" if out.exists() else ""
        print(f"multi-exposure  {_span(s.frames)}  ({len(s.frames)} frames, {mode}, "
              f"from {s.anchor.path.name}){done}")
        for w in s.warnings + s.mode_warnings:
            print(f"   ! {w}")
    for p in p1 + p2:
        print(f"!  {p}", file=sys.stderr)
    if not (shift or stack or p1 or p2):
        print("no pixel-shift or multi-exposure sets found")
    return 0


def mask_path(out):
    """DSCF0103.kunmeto.dng -> DSCF0103.kunmeto.mask.tif"""
    return Path(out).with_suffix(".mask.tif")


def cmd_shift(a):
    _validate_output(a)
    sets, problems = find_shift_sets(_gather(a.paths, a.recursive))
    for p in problems:
        print(f"!  {p}", file=sys.stderr)
    if not sets:
        print("no pixel-shift sets found")
        return 1 if problems else 0
    repair = not a.no_repair and a.mode == "hires"
    if a.mode == "color" and not a.no_repair and not a.dry_run:
        print("note: defect repair is skipped in color mode (4 frames, nothing to compare against)")
    if a.repair_mask and not repair:
        raise UsageError("--repair-mask needs repair (hires mode, without --no-repair)")
    # hires keeps the plain "kunmeto" suffix (unchanged, so existing outputs are still found by
    # --force/skip); color gets its own suffix so it never collides with a hires output of the
    # same frames.
    suffix = "kunmeto" if a.mode == "hires" else f"kunmeto-{a.mode}"
    jobs = [(f"{_span(s.frames)} ({len(s.frames)} frames)", s.frames[0].path, suffix,
             lambda out, progress=None, s=s: pipeline.run_shift(
                 [f.path for f in s.frames], out, mode=a.mode, use_float=a.float, jobs=a.jobs,
                 repair=repair, mask_path=mask_path(out) if a.repair_mask else None,
                 progress=progress),
             SHIFT_REPAIR if repair else SHIFT_PLAIN)
            for s in sets]
    return max(_run_sets(jobs, a), 1 if problems else 0)


def cmd_stack(a):
    _validate_output(a)
    sets, problems = resolve_stack(a.targets, a.recursive)
    for p in problems:
        print(f"!  {p}", file=sys.stderr)
    jobs = []
    failed = False
    for s in sets:
        for w in s.warnings:
            print(f"   ! {w}")
        mode = a.mode
        source = "--mode"
        if mode is None:
            for w in s.mode_warnings:
                print(f"   ! {w}")
            mode = s.mode
            source = f"from {s.anchor.path.name}" if s.anchor else None
        if mode is None:
            if sys.stdin.isatty() and not a.dry_run:
                print(f"{_span(s.frames)} ({len(s.frames)} frames) has no recorded mode.")
                mode = prompt_mode()
                source = "chosen at prompt"
            else:
                print(f"!  {_span(s.frames)}: no mode recorded; pass --mode ({', '.join(MODES)})",
                      file=sys.stderr)
                failed = True
                continue
        names = ", ".join(f.path.name for f in s.frames)
        print(f"stack: {names} ({len(s.frames)} frames), mode={mode} ({source})")
        label = f"{_span(s.frames)} ({len(s.frames)} frames, {mode}, {source})"
        jobs.append((label, s.frames[0].path, f"kunmeto-{mode}",
                     lambda out, progress=None, s=s, mode=mode: pipeline.run_stack(
                         [f.path for f in s.frames], out, mode, normalize=a.normalize,
                         use_float=a.float, jobs=a.jobs, progress=progress),
                     STACK))
    if not sets and not problems:
        print("no multi-exposure sets found")
    code = _run_sets(jobs, a) if jobs else 0
    return max(code, 1 if (problems or failed) else 0)


def build_parser():
    ap = argparse.ArgumentParser(prog="kunmeto", description=__doc__, epilog=EXIT_CODES_EPILOG)
    ap.add_argument("--version", action="version", version=f"kunmeto {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, outputs=True):
        p.add_argument("-r", "--recursive", action="store_true", help="scan folders recursively")
        if outputs:
            p.add_argument("-o", "--output", metavar="DIR", help="output folder (default: next to the first frame)")
            p.add_argument("--force", action="store_true", help="overwrite existing outputs")
            p.add_argument("-j", "--jobs", type=positive_int, default=None,
                           help="parallel decode processes (default: CPU count)")
            p.add_argument("--float", action="store_true", help="write 32-bit float DNG")
            p.add_argument("--dry-run", action="store_true", help="show what would be written")
            p.add_argument("--no-progress", action="store_true",
                           help="no progress bar (it is only drawn when stderr is a terminal)")

    p = sub.add_parser("scan", help="list pixel-shift and multi-exposure sets")
    p.add_argument("paths", nargs="+", help="folders or files (RAF/JPG) to scan")
    common(p, outputs=False)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("shift", help="merge pixel-shift sets into LinearRaw DNGs")
    p.add_argument("paths", nargs="+", help="folders or files (RAF) to merge")
    common(p)
    p.add_argument("--mode", choices=("hires", "color"), default="hires",
                   help="hires: 16-shot 400 MP; color: 4-shot full-colour 100 MP")
    p.add_argument("--no-repair", action="store_true",
                   help="skip defect repair (camera shake / wind); writes the plain merge")
    p.add_argument("--repair-mask", action="store_true",
                   help="also write <name>.kunmeto.mask.tif: per frame, where samples were replaced")
    p.set_defaults(func=cmd_shift)

    p = sub.add_parser("stack", help="merge multi-exposure sets into CFA DNGs")
    p.add_argument("targets", nargs="+", help="folders, camera composite JPGs, or RAF files")
    common(p)
    p.add_argument("--mode", choices=MODES, default=None,
                   help="override / supply the blend mode (required for RAF targets unless run "
                        "in a terminal)")
    p.add_argument("--normalize", action="store_true", help="equalize frame brightness before stacking")
    p.set_defaults(func=cmd_stack)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except UsageError as e:
        print(f"kunmeto: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
