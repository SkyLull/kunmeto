"""Find pixel-shift and multi-exposure sets from RAF/JPG metadata (never from filenames)."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .raf import IMAGE_COUNT_MOD, PIXEL_SHIFT, read_jpeg_meta, read_raf_meta
from .stack import MODES  # single source of truth; sets.py only re-exports it for cli.py

# Camera JPG tag 0x1037 -> mode. Guessed from the camera menu order; only 3 has been seen.
MODE_FROM_JPG = {1: "add", 2: "average", 3: "lighten", 4: "darken"}
VERIFIED_JPG_MODES = {3}
SUFFIXES = {".raf": "raf", ".jpg": "jpg", ".jpeg": "jpg"}


@dataclass(frozen=True)
class ShotInfo:
    path: Path
    kind: str                 # "raf" | "jpg"
    model: str
    serial: str
    image_count: int | None   # None when the file lacks Fuji's ImageCount tag; such files never group
    auto_bracketing: int = 0
    sequence: int = 0
    shift_shots: int = 0
    exposure_count: int = 1
    multi_mode: int | None = None


@dataclass
class ShiftSet:
    frames: list[ShotInfo]              # sorted by sequence


@dataclass
class StackSet:
    frames: list[ShotInfo]              # in shooting order
    mode: str | None                     # None = the user must choose
    anchor: ShotInfo | None = None       # the camera composite JPG, if any
    warnings: list[str] = field(default_factory=list)       # always shown: mixed cameras, pixel-shift frames
    mode_warnings: list[str] = field(default_factory=list)  # the JPG mode-mapping guess; `--mode` suppresses these


def collect_files(paths, recursive=False) -> tuple[list[Path], list[Path]]:
    """RAF/JPG files from folders (optionally recursive) and explicit files, resolved to absolute
    paths, sorted and de-duplicated (so the same file under two spellings - relative, absolute,
    `..`, a symlink - counts once). Folder contents are filtered by suffix; a file named
    explicitly is included regardless of its suffix, so `gather` can report it as skipped if it
    turns out not to be readable. Returns (files, missing), where `missing` holds paths - folders
    or files - that don't exist."""
    files, missing = set(), []
    for p in map(Path, paths):
        if p.is_dir():
            try:
                it = p.rglob("*") if recursive else p.iterdir()
                files.update(q.resolve() for q in it if q.is_file() and q.suffix.lower() in SUFFIXES)
            except OSError:      # e.g. a folder without read/execute permission
                missing.append(p.resolve())
        elif p.is_file():
            files.add(p.resolve())
        else:
            missing.append(p.resolve())
    return sorted(files), missing


def info_from_file(path) -> ShotInfo | None:
    """ShotInfo for a Fujifilm RAF/JPG, or None if it isn't one (or has no usable Exif)."""
    path = Path(path)
    kind = SUFFIXES.get(path.suffix.lower())
    if kind is None:
        return None
    try:
        m = read_raf_meta(path) if kind == "raf" else read_jpeg_meta(path)
    except (ValueError, OSError):      # raf.ExifError is a ValueError; RafMeta properties never raise
        return None
    return ShotInfo(path, kind, m.model, m.serial, m.image_count, m.auto_bracketing,
                    m.sequence_number, m.pixel_shift_shots, m.exposure_count, m.multi_exposure_mode)


def gather(paths) -> tuple[list[ShotInfo], list[Path]]:
    """(infos, skipped): `skipped` holds paths info_from_file couldn't read, or that aren't a
    Fuji RAF/JPG at all."""
    infos, skipped = [], []
    for p in paths:
        i = info_from_file(p)
        if i is not None:
            infos.append(i)
        else:
            skipped.append(p)
    return infos, skipped


def _shift_problem(path, seqs, n):
    dupes = sorted({s for s, c in Counter(seqs).items() if c > 1})
    missing = sorted(set(range(1, n + 1)) - set(seqs))
    if dupes:
        msg = (f"pixel-shift shot at {path}: duplicate sequence numbers {dupes} "
               f"(found {seqs}, expected 1..{n})")
        if missing:
            msg += f", missing {missing}"
        return msg + "; skipped"
    return (f"pixel-shift shot at {path}: found sequence numbers {seqs}, expected 1..{n}, "
            f"missing {missing}; skipped")


def find_shift_sets(infos) -> tuple[list[ShiftSet], list[str]]:
    """Groups RAFs by (folder, model, serial, ImageCount): the camera writes one pixel-shift set
    into one folder, and different cameras or folders (e.g. a backup copy) must never merge."""
    groups = defaultdict(list)
    for i in infos:
        if i.kind == "raf" and i.auto_bracketing == PIXEL_SHIFT and i.shift_shots > 0 and i.image_count is not None:
            groups[(i.path.parent, i.model, i.serial, i.image_count)].append(i)
    sets, problems = [], []
    for frames in sorted(groups.values(), key=lambda fs: min(f.path for f in fs)):
        frames.sort(key=lambda f: (f.sequence, f.path))
        n = frames[0].shift_shots
        seqs = [f.sequence for f in frames]
        if seqs == list(range(1, n + 1)) and all(f.shift_shots == n for f in frames):
            sets.append(ShiftSet(frames))
        else:
            problems.append(_shift_problem(str(frames[0].path), seqs, n))
    return sets, problems


def mode_from_jpg(value):
    """(mode or None, warnings) for the camera JPG's 0x1037 value."""
    mode = MODE_FROM_JPG.get(value)
    if mode is None:
        return None, [f"unknown multi-exposure mode value {value} in the camera JPG; pass --mode"]
    if value in VERIFIED_JPG_MODES:
        return mode, []
    return mode, [f"camera JPG mode value {value} -> '{mode}' is an unverified guess; pass --mode to override"]


def find_stack_sets(infos, only=None) -> tuple[list[StackSet], list[str]]:
    """Sets anchored on camera composite JPGs; a composite only pairs with RAFs in its own folder
    (the camera writes one set into one folder, so this avoids backup copies colliding). ImageCount
    is a 15-bit counter that wraps, so expected counts wrap through IMAGE_COUNT_MOD too. `only`
    limits to those JPG paths (resolved for comparison); an `only` path that never became a set and
    has no problem of its own is reported as not a composite JPG."""
    rafs = {(i.path.parent, i.model, i.serial, i.image_count): i
            for i in infos if i.kind == "raf" and i.auto_bracketing != PIXEL_SHIFT and i.image_count is not None}
    only_paths = {Path(p).resolve() for p in only} if only is not None else None
    handled = set()
    sets, problems = [], []
    candidates = sorted((i for i in infos if i.kind == "jpg" and i.exposure_count > 1 and i.image_count is not None),
                        key=lambda i: i.path)
    for j in candidates:
        if only_paths is not None and j.path not in only_paths:
            continue
        handled.add(j.path)
        n = j.exposure_count
        counts = [(j.image_count - n + 1 + k) % IMAGE_COUNT_MOD for k in range(n)]
        frames = [rafs.get((j.path.parent, j.model, j.serial, c)) for c in counts]
        found = [f for f in frames if f is not None]
        if len(found) != n:
            missing_counts = [c for c, f in zip(counts, frames) if f is None]
            problems.append(
                f"{j.path}: composite of {n} exposures, but {len(missing_counts)} RAFs are missing "
                f"in {j.path.parent} (ImageCount {', '.join(map(str, missing_counts))}); skipped")
            continue
        mode, mode_warnings = mode_from_jpg(j.multi_mode)
        sets.append(StackSet(found, mode, anchor=j, mode_warnings=mode_warnings))
    if only_paths is not None:
        for p in sorted(only_paths - handled):
            problems.append(f"{p}: not a readable camera multi-exposure composite JPG")
    return sets, problems


def stack_set_from_rafs(infos) -> StackSet:
    """Explicit RAF targets: one set, frames in shooting order (by ImageCount; filenames wrap so
    they aren't a reliable order), mode left to the user. The same file passed twice (under
    different spellings) is de-duplicated. Warns if the frames span more than one camera or
    include pixel-shift shots, either of which likely means the wrong files were given."""
    by_path = {}
    for i in infos:
        by_path.setdefault(i.path.resolve(), i)
    frames = sorted(by_path.values(), key=lambda i: (i.image_count is None, i.image_count, i.path))
    warnings = []
    cameras = sorted({(f.model, f.serial) for f in frames})
    if len(cameras) > 1:
        warnings.append(f"frames come from more than one camera: {cameras}")
    if any(f.auto_bracketing == PIXEL_SHIFT for f in frames):
        warnings.append("includes pixel-shift frames, which don't belong in a multi-exposure stack")
    return StackSet(frames, None, warnings=warnings)
