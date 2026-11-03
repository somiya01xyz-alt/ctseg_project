"""
Semi-automatic selection tools.
Each function returns a bool mask – it does NOT edit the LabelMap.
You decide what to do with the result: paint it, preview it, etc.

Works on 2-D images (y, x) and 3-D volumes (z, y, x).
Seeds are tuples with the same number of elements as the image dimensions.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.segmentation import flood


# ── threshold ────────────────────────────────────────────────────────────────

def threshold_mask(image: np.ndarray, low=None, high=None) -> np.ndarray:
    """All voxels with  low <= intensity <= high."""
    m = np.ones(image.shape, bool)
    if low  is not None: m &= image >= low
    if high is not None: m &= image <= high
    return m


# ── flood fill (paint-bucket) ────────────────────────────────────────────────

def flood_fill(image: np.ndarray, seed: tuple, tolerance: float) -> np.ndarray:
    """
    Connected voxels whose value is within ±tolerance of the seed voxel's value.
    Works in 2-D and 3-D.
    """
    return flood(image, tuple(seed), tolerance=tolerance)


# ── region growing ────────────────────────────────────────────────────────────

def region_grow(image: np.ndarray, seed: tuple,
                low: float, high: float, connectivity: int = 1) -> np.ndarray:
    """
    Threshold [low, high] then keep only the connected component containing the seed.
    connectivity=1: face-connected. connectivity=2 (2-D) or 3 (3-D): corner-connected.
    """
    seed = tuple(seed)
    cand = threshold_mask(image, low, high)
    if not cand[seed]:
        return np.zeros(image.shape, bool)
    struct = ndi.generate_binary_structure(image.ndim, connectivity)
    lab, _ = ndi.label(cand, structure=struct)
    return lab == lab[seed]


def region_grow_auto(image: np.ndarray, seed: tuple,
                     k: float = 2.5, radius: int = 3,
                     connectivity: int = 1) -> np.ndarray:
    """
    Like region_grow, but the range is computed automatically from a small
    neighbourhood around the seed:  mean ± k*std.
    Good default when you just click on a material without knowing its intensity range.
    """
    seed = tuple(seed)
    sl = tuple(slice(max(s - radius, 0), s + radius + 1) for s in seed)
    patch = image[sl].astype(float)
    mu, sigma = patch.mean(), max(patch.std(), 1.0)
    return region_grow(image, seed, mu - k * sigma, mu + k * sigma, connectivity)


# ── connected-component selection ────────────────────────────────────────────

def component_at(binary: np.ndarray, seed: tuple, connectivity: int = 1) -> np.ndarray:
    """Pick the connected blob of `binary` that contains `seed`."""
    seed = tuple(seed)
    if not binary[seed]:
        return np.zeros(binary.shape, bool)
    struct = ndi.generate_binary_structure(binary.ndim, connectivity)
    lab, _ = ndi.label(binary, structure=struct)
    return lab == lab[seed]


def largest_component(binary: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Keep only the largest connected component."""
    struct = ndi.generate_binary_structure(binary.ndim, connectivity)
    lab, n = ndi.label(binary, structure=struct)
    if n == 0:
        return binary.copy()
    counts = np.bincount(lab.ravel())
    counts[0] = 0  # ignore background
    return lab == counts.argmax()


def remove_small(binary: np.ndarray, min_size: int) -> np.ndarray:
    """Delete blobs smaller than min_size voxels."""
    b = binary.astype(bool)
    try:                                   # scikit-image >= 0.26
        return remove_small_objects(b, max_size=max(min_size - 1, 0))
    except TypeError:                      # older versions
        return remove_small_objects(b, min_size=min_size)


# ── multi-Otsu: several materials in one go ──────────────────────────────────

def multi_otsu_thresholds(volume: np.ndarray, classes: int = 4, sample_step: int = 8) -> np.ndarray:
    """
    Intensity thresholds that split `volume` into `classes` groups (classes-1 numbers, ascending).
    They are computed on every `sample_step`-th voxel in each direction, so a 700-million-voxel
    CT takes well under a second. Pass sample_step=1 to use every voxel.
    """
    from skimage.filters import threshold_multiotsu
    step = max(int(sample_step), 1)
    sample = np.asarray(volume)[tuple(slice(None, None, step) for _ in range(volume.ndim))]
    return np.asarray(threshold_multiotsu(sample, classes=classes))


def apply_thresholds(volume: np.ndarray, thresholds, slab: int = 16) -> np.ndarray:
    """
    Class number of every voxel: 0 = darker than the first threshold, 1 = between the first and
    second, ... Works slab by slab along z so a huge volume does not need a big temporary array.
    Returns uint8.
    """
    volume = np.asarray(volume)
    th = np.asarray(thresholds)
    out = np.empty(volume.shape, np.uint8)
    if volume.ndim == 2:
        out[...] = np.digitize(volume, th)
        return out
    for z in range(0, volume.shape[0], slab):
        out[z:z + slab] = np.digitize(volume[z:z + slab], th)
    return out


def multi_otsu(volume: np.ndarray, classes: int = 4, sample_step: int = 8, thresholds=None):
    """
    Split a volume (or image) into `classes` intensity classes with multi-Otsu.
    Returns (labels, thresholds): `labels` is a uint8 array, 0 = darkest class (background),
    classes-1 = brightest. The thresholds come from a subsample (see multi_otsu_thresholds) and
    are then applied to the full array. Pass `thresholds` to reuse earlier ones, e.g. to segment
    one slice with the thresholds of the whole volume.
    No napari here: the caller decides what to do with the labels.
    """
    if thresholds is None:
        thresholds = multi_otsu_thresholds(volume, classes, sample_step)
    return apply_thresholds(volume, thresholds), np.asarray(thresholds)


# ── several seeds, each with its own label ───────────────────────────────────

def multi_seed_grow(image: np.ndarray, seeds, k: float = 4.0, radius: int = 8,
                    connectivity: int = 1) -> np.ndarray:
    """
    Region-grow from several seeds at once. `seeds` is a list of (label, seed_tuple); the seed
    tuple has as many numbers as `image` has dimensions. Each seed grows with the same rule as
    region_grow_auto (mean +- k*std of the neighbourhood). Returns an int array: 0 where no seed
    reached, otherwise the label of the seed that claimed the voxel.
    Where two seeds reach the same voxel, the one whose mean intensity is closest (in units of
    its own std) wins, so neighbouring materials do not overwrite each other.
    """
    out = np.zeros(image.shape, np.int32)
    best = np.full(image.shape, np.inf)
    for label, seed in seeds:
        seed = tuple(int(s) for s in seed)
        sl = tuple(slice(max(s - radius, 0), s + radius + 1) for s in seed)
        patch = image[sl].astype(float)
        mu, sigma = patch.mean(), max(patch.std(), 1.0)
        m = region_grow(image, seed, mu - k * sigma, mu + k * sigma, connectivity)
        if not m.any():
            continue
        dist = np.abs(image[m].astype(float) - mu) / sigma
        win = dist < best[m]
        idx = np.nonzero(m)
        sel = tuple(i[win] for i in idx)
        out[sel] = int(label)
        best[sel] = dist[win]
    return out
