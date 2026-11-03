"""
Brush / Eraser tool.

A "stroke" is the path the mouse takes from button-down to button-up.
The whole stroke becomes ONE undo step in the LabelMap.

Points are (z, y, x) voxel coordinates.
mode="2d"  →  disk brush on a single slice (z is the same for every point).
mode="3d"  →  sphere brush that can span multiple slices.
"""
import numpy as np


# ── footprint (the brush stamp) ──────────────────────────────────────────────

def footprint(radius: float, shape: str = "round", mode: str = "2d") -> np.ndarray:
    """
    Returns a bool array that is the brush shape.
    2-D: shape (1, d, d)   – d = 2*radius+1
    3-D: shape (d, d, d)
    shape="round"  →  disk / sphere
    shape="square" →  full square / cube
    """
    r = int(radius)
    ax = np.arange(-r, r + 1)
    if mode == "2d":
        yy, xx = np.meshgrid(ax, ax, indexing="ij")
        if shape == "square":
            fp = np.ones(yy.shape, bool)
        else:
            fp = (yy ** 2 + xx ** 2) <= r * r
        return fp[None]       # add z axis → shape (1, d, d)
    else:
        zz, yy, xx = np.meshgrid(ax, ax, ax, indexing="ij")
        if shape == "square":
            return np.ones(zz.shape, bool)
        return (zz ** 2 + yy ** 2 + xx ** 2) <= r * r


# ── stamp helper (used while building the stroke mask) ───────────────────────

def _stamp(local: np.ndarray, center: np.ndarray, fp: np.ndarray):
    """OR the footprint `fp` into the local scratch array at `center`."""
    half = np.array([s // 2 for s in fp.shape])
    lo = center - half
    hi = lo + np.array(fp.shape)
    a = np.maximum(lo, 0)
    b = np.minimum(hi, np.array(local.shape))
    if np.any(a >= b):
        return
    dst = tuple(slice(int(x), int(y)) for x, y in zip(a, b))
    src = tuple(slice(int(x - l), int(y - l)) for x, y, l in zip(a, b, lo))
    local[dst] |= fp[src]


# ── build the stroke mask ─────────────────────────────────────────────────────

def stroke_mask(points, radius: float, shape: str = "round", mode: str = "2d"):
    """
    Build the 3-D bool mask that covers the entire stroke.
    Interpolates between consecutive points so fast mouse movement leaves no gaps.

    Returns: (mask, origin)
        mask   – 3-D bool array  (size ≤ volume)
        origin – (z0, y0, x0) where mask[0,0,0] sits in the volume
    Call lm.paint_block(mask, origin, label) to apply.
    """
    pts = np.asarray(points, dtype=int)  # shape (N, 3)
    if pts.ndim == 1:
        pts = pts[None]
    fp   = footprint(radius, shape, mode)
    half = np.array([s // 2 for s in fp.shape])

    lo = pts.min(axis=0) - half
    hi = pts.max(axis=0) + half + 1
    local = np.zeros(hi - lo, bool)

    segments = list(zip(pts[:-1], pts[1:])) if len(pts) > 1 else [(pts[0], pts[0])]
    for a, b in segments:
        n = int(np.abs(b - a).max()) + 1
        for t in np.linspace(0, 1, n):
            center = np.rint(a + t * (b - a)).astype(int) - lo
            _stamp(local, center, fp)

    return local, tuple(int(v) for v in lo)


# ── public API ───────────────────────────────────────────────────────────────

def paint_stroke(lm, points, radius: float, label: int,
                 shape: str = "round", mode: str = "2d", protect: bool = False) -> bool:
    """Apply a brush stroke to the LabelMap (one undo step)."""
    mask, origin = stroke_mask(points, radius, shape, mode)
    return lm.paint_block(mask, origin, label, protect)


def erase_stroke(lm, points, radius: float,
                 shape: str = "round", mode: str = "2d", only_label=None) -> bool:
    """Apply an eraser stroke to the LabelMap (one undo step)."""
    mask, origin = stroke_mask(points, radius, shape, mode)
    return lm.erase_block(mask, origin, only_label)
