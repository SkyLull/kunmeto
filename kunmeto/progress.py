"""Progress reporting: library code calls report(progress, phase, done, total); the CLI passes a
Bar, which draws one self-updating line on a terminal. progress=None (the default everywhere)
reports nothing."""
import shutil
import sys
import time


def report(progress, phase, done, total):
    if progress is not None:
        progress(phase, done, total)


# Relative duration of each phase (seconds per 16-shot set on an M-series Mac), used to turn
# phase progress into one overall percentage. Only the ratios matter.
SHIFT_REPAIR = {"decode": 7, "detect": 4.5, "z-scores": 2.5, "defects": 2.2, "merge": 8,
                "fill": 9, "write": 5}
SHIFT_PLAIN = {"decode": 7, "merge": 6, "write": 5}
STACK = {"decode": 5, "merge": 2, "write": 3}


def _clock(s):
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


class Bar:
    """`label  ▕████░░░░▏  45%  fill 7/17   0:21 < 0:25` on one terminal line (stderr)."""

    def __init__(self, label, weights, stream=None, width=None, min_interval=0.1):
        self.label = label
        self.weights = weights
        self.total_weight = sum(weights.values())
        self.stream = stream or sys.stderr
        self.width = width
        self.min_interval = min_interval
        self.start = time.monotonic()
        self.last_draw = 0.0
        self.shown = False

    def fraction(self, phase, done, total):
        before = 0.0
        for name, w in self.weights.items():
            if name == phase:
                part = done / total if total else 0.0
                return min(1.0, (before + w * min(part, 1.0)) / self.total_weight)
            before += w
        return min(1.0, before / self.total_weight)       # unknown phase: count it as done

    def __call__(self, phase, done, total):
        now = time.monotonic()
        if done < total and now - self.last_draw < self.min_interval:
            return
        self.last_draw = now
        f = self.fraction(phase, done, total)
        elapsed = now - self.start
        eta = f" < {_clock(elapsed * (1 - f) / f)}" if 0.05 < f < 1 else ""
        step = f"{phase} {done}/{total}" if total > 1 else phase
        tail = f" {100 * f:3.0f}%  {step:<14} {_clock(elapsed)}{eta}"
        cols = self.width or shutil.get_terminal_size((100, 20)).columns
        bar_w = max(10, min(30, cols - len(self.label) - len(tail) - 6))
        filled = int(round(f * bar_w))
        line = f"{self.label}  ▕{'█' * filled}{'░' * (bar_w - filled)}▏{tail}"
        self.stream.write("\r" + line[: cols - 1].ljust(cols - 1))
        self.stream.flush()
        self.shown = True

    def close(self):
        """Clear the bar's line so the next message starts at column 0."""
        if self.shown:
            cols = self.width or shutil.get_terminal_size((100, 20)).columns
            self.stream.write("\r" + " " * (cols - 1) + "\r")
            self.stream.flush()
            self.shown = False
