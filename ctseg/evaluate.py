"""
Evaluation metrics for comparing two segmentation masks.

dice(a, b) → float in [0, 1].  1.0 = identical.  0.0 = no overlap at all.
iou(a, b)  → float in [0, 1].  Also called Jaccard index.

Both functions accept any array-like of bool or 0/1 values.
"""
import numpy as np


def dice(a, b) -> float:
    """Sørensen–Dice coefficient: 2*|A∩B| / (|A| + |B|)."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0   # both empty → perfectly agree
    return 2.0 * float((a & b).sum()) / float(s)


def iou(a, b) -> float:
    """Intersection over Union (Jaccard): |A∩B| / |A∪B|."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    u = float((a | b).sum())
    if u == 0:
        return 1.0
    return float((a & b).sum()) / u


def per_slice_dice(pred_volume: np.ndarray, gt_volume: np.ndarray) -> np.ndarray:
    """
    Compute Dice for every slice and return a 1-D array of length z.
    Useful for plotting how well the segmentation tracks across slices.
    """
    assert pred_volume.shape == gt_volume.shape, "Shapes must match"
    scores = [dice(pred_volume[z], gt_volume[z]) for z in range(pred_volume.shape[0])]
    return np.array(scores)
