"""kunmeto: RAW-level merging for Fujifilm GFX pixel-shift and multi-exposure shots."""
import os

# LibRaw's Fujifilm decoder has an OpenMP data race (LibRaw issue #845): repeated decodes of the
# same RAF can differ. Pin OpenMP to one thread before rawpy is ever imported; kunmeto
# parallelizes with one process per file instead. Forced (not setdefault): correctness, not a
# default, so an inherited/pre-set value must not win.
#
# This only takes effect if kunmeto is imported before rawpy loads its LibRaw shared library
# (LibRaw reads OMP_NUM_THREADS at load time). It also only matters for OpenMP-enabled LibRaw
# builds (Linux wheels, or source builds compiled with OpenMP) — the macOS rawpy wheel bundles a
# LibRaw built without OpenMP, so this guard is a no-op there but kept for portability.
os.environ["OMP_NUM_THREADS"] = "1"

__version__ = "0.1.0"
