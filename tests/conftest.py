from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"


def make_frame(raw, colors=((0, 1), (3, 2)), black=60, white=15872, wb=(2.0, 1.0, 1.5, 0.0),
               visible=None, crop=None, meta=None, path="synthetic.RAF"):
    """A kunmeto.frames.Frame built from a synthetic uint16 mosaic (tests only)."""
    from kunmeto.frames import Frame

    raw = np.asarray(raw, np.uint16)
    h, w = raw.shape
    return Frame(
        path=Path(path), raw=raw, colors=np.array(colors), black=(black,) * 4, white=white,
        visible=visible or (0, 0, h, w), crop=crop or (0, 0, w, h),
        cam_xyz=np.array([[1.6212, -0.8423, -0.1583], [-0.4336, 1.2583, 0.1937], [-0.0195, 0.0726, 0.6199]]),
        wb=wb, meta=meta,
    )
