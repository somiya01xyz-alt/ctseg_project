"""
Slice propagation – carry a segmentation from one slice to others.

Method A – interpolate_between_slices
  You segment slice z0 and slice z1, this fills the slices between them
  using shape-based interpolation (Raya & Udupa, 1990).
  Best for smooth objects that don't change too fast.

Method B – propagate_slices
  Walk slice by slice from z_start toward z_end.
  For each new slice, keep pixels that:
    (1) have similar intensity to the previous label mask, AND
    (2) are spatially near the previous mask.
  Best for tracking irregular shapes along the stack.
"""
import numpy as np
from scipy import ndimage as ndi
from .morphology import dilate
from .select import remove_small


# ── helper: signed distance transform ────────────────────────────────────────

def _signed_distance(m: np.ndarray) -> np.ndarray:
    """
    Positive inside the shape, negative outside, ~0 on the boundary.
    Used by shape-based interpolation.
    """
    if not m.any():
        return -np.full(m.shape, float(max(m.shape)))
    inside  = ndi.distance_transform_edt(m)
    outside = ndi.distance_transform_edt(~m)
    return inside - outside


# ── Method A: shape-based interpolation ──────────────────────────────────────

def interpolate_between_slices(lm, label: int, z0: int, z1: int) -> bool:
    """
    Fill slices z0+1 … z1-1 by linearly interpolating the signed distance maps
    of the segmented masks at z0 and z1.

    You MUST have label painted at z0 and z1 before calling this.
    z0 < z1 (swap them if needed).
    """
    if z0 > z1:
        z0, z1 = z1, z0
    if z1 - z0 < 2:
        return False   # nothing between them

    da = _signed_distance(lm.get_mask(label, z0))
    db = _signed_distance(lm.get_mask(label, z1))
    n  = z1 - z0 - 1

    interpolated = np.stack([
        ((1 - w) * da + w * db) > 0
        for w in (t / (n + 1) for t in range(1, n + 1))
    ])   # shape: (n, H, W)

    return lm.set_label_block(label, interpolated, (z0 + 1, 0, 0))


def _fill_small_holes(m: np.ndarray, max_hole: int) -> np.ndarray:
    """Fill enclosed holes smaller than max_hole pixels; keep large holes open."""
    holes = ndi.binary_fill_holes(m) & ~m
    lab, n = ndi.label(holes)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())
    small = sizes <= max_hole
    small[0] = False
    return m | small[lab]


# ── Method B: propagate slice by slice ───────────────────────────────────────

def propagate_slices(lm, image: np.ndarray, label: int,
                     z_start: int, z_end: int,
                     grow: int = 3, min_size: int = 20,
                     max_hole: int = 200) -> int:
    """
    Walk from z_start toward z_end (works in both directions).
    For each new slice z:
      1. Find the intensity range that covers the label in the previous slice (2nd–98th %ile).
      2. Threshold the new slice to that range.
      3. Keep only pixels within `grow` voxels of the previous mask.
      4. Fill SMALL holes (< max_hole pixels) and remove tiny blobs.
         Large holes are kept, so a ring stays a ring instead of becoming a disc.
      5. Write the result to the LabelMap.

    Returns the number of slices successfully propagated.
    """
    step = 1 if z_end >= z_start else -1
    done = 0
    for z in range(z_start + step, z_end + step, step):
        prev_mask = lm.get_mask(label, z - step)
        if not prev_mask.any():
            break
        prev_pixels = image[z - step][prev_mask]
        lo, hi = np.percentile(prev_pixels, [2, 98])
        intensity_ok  = (image[z] >= lo) & (image[z] <= hi)
        spatial_ok    = dilate(prev_mask, grow)
        new = intensity_ok & spatial_ok
        new = _fill_small_holes(new, max_hole)
        new = remove_small(new, min_size)
        lm.set_label_mask(label, new, z=z)
        done += 1
    return done
