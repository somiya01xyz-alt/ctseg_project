"""
Morphological operations for cleanup.

Pass a 2-D mask  → uses a disk   (2-D structuring element).
Pass a 3-D mask  → uses a ball   (3-D structuring element).

`apply_morphology` is the convenient function that edits a LabelMap in one undo step.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, ball


def _fp(radius: int, ndim: int):
    """Return the right structuring element for the number of dimensions."""
    return disk(radius) if ndim == 2 else ball(radius)


def dilate(mask, radius: int = 1):
    """Grow the mask outward by `radius` voxels."""
    return ndi.binary_dilation(mask, structure=_fp(radius, mask.ndim))


def erode(mask, radius: int = 1):
    """Shrink the mask inward by `radius` voxels."""
    return ndi.binary_erosion(mask, structure=_fp(radius, mask.ndim))


def opening(mask, radius: int = 1):
    """Erode then dilate. Removes thin bridges and tiny specks."""
    return ndi.binary_opening(mask, structure=_fp(radius, mask.ndim))


def closing(mask, radius: int = 1):
    """Dilate then erode. Closes small holes and gaps."""
    return ndi.binary_closing(mask, structure=_fp(radius, mask.ndim))


def fill_holes(mask, radius: int = 0):
    """Fill enclosed holes in the mask."""
    return ndi.binary_fill_holes(mask)


# ── mapping name → function (used by apply_morphology) ───────────────────────
OPS = {
    "dilate":  dilate,
    "erode":   erode,
    "open":    opening,
    "close":   closing,
    "fill":    fill_holes,
}


def apply_morphology(lm, label: int, op: str, radius: int = 1, z=None) -> bool:
    """
    Apply a morphological operation to one label in the LabelMap (one undo step).
    op    : "dilate" | "erode" | "open" | "close" | "fill"
    z     : int  → operate on one slice only
            None → operate on the full 3-D mask
    """
    if op not in OPS:
        raise ValueError(f"Unknown op {op!r}. Choose from {list(OPS)}")
    current_mask = lm.get_mask(label, z=z)
    new_mask = OPS[op](current_mask, radius)
    return lm.set_label_mask(label, new_mask, z=z)


# ── thin structures (1-2 px lines such as a braze joint or an outer skin) ────
# Intensity growing cannot find a line that is only 1-2 pixels wide. These two
# tools derive such lines from the neighbouring thick regions instead.

def _per_slice(mask, fn):
    """Apply a 2-D function to every slice of a 3-D mask (or directly to a 2-D one)."""
    if mask.ndim == 2:
        return fn(mask)
    return np.stack([fn(m) for m in mask])


def outline(mask, width: int = 1):
    """Inner boundary of `mask`, `width` pixels thick, slice by slice
    (the mask minus the mask eroded by `width`)."""
    st = ndi.generate_binary_structure(2, 1)

    def one(m):
        m = m.astype(bool)
        return m & ~ndi.binary_erosion(m, structure=st, iterations=int(width), border_value=0)
    return _per_slice(mask, one)


def interface(mask_a, mask_b, width: int = 1):
    """Thin band where region A touches region B: dilate(A) & dilate(B), slice by slice.
    E.g. a braze joint that sits between a ceramic and a metal region."""
    st = ndi.generate_binary_structure(2, 1)

    def grow(m):
        return ndi.binary_dilation(m.astype(bool), structure=st, iterations=int(width))
    return _per_slice(mask_a, grow) & _per_slice(mask_b, grow)


def outer_boundary(mask, width: int = 1, outside: bool = False):
    """Only the OUTERMOST contour of the region(s): holes and inner rings are ignored.
    The mask is hole-filled first, then its edge is taken, slice by slice.
    outside=False -> the line is the outer pixels of the region (inside its edge);
    outside=True  -> the line is just outside the region."""
    st = ndi.generate_binary_structure(2, 1)

    def one(m):
        f = ndi.binary_fill_holes(m.astype(bool))
        if outside:
            return ndi.binary_dilation(f, structure=st, iterations=int(width)) & ~f
        return f & ~ndi.binary_erosion(f, structure=st, iterations=int(width), border_value=0)
    return _per_slice(mask, one)
