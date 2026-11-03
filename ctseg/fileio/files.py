"""
File I/O for CT volumes and segmentation projects.

load_tiff_stack  – read a folder of .tif slices → numpy array (z, y, x)
save_project     – write labels.npy + labels.json [+ rois.json]
load_project     – read them back → (LabelMap, LabelTable, list[ROI])
export_tiff      – write the label volume as a multi-page TIFF
"""
import glob
import json
import os
import re

import numpy as np
import tifffile

from ..core.labelmap import LabelMap
from ..core.label_table import LabelTable


# ── sort filenames naturally (Assembly001 < Assembly002 < Assembly010) ────────

def _natural_key(path: str):
    return [int(t) if t.isdigit() else t
            for t in re.split(r"(\d+)", os.path.basename(path))]


# ── load a stack of TIFF slices ───────────────────────────────────────────────

def load_tiff_stack(folder: str,
                    step: int = 1,
                    downsample: int = 1,
                    channel: int = 0,
                    max_slices: int = None,
                    verbose: bool = True) -> np.ndarray:
    """
    Read every .tif file in `folder` (sorted naturally) and stack them into
    a single 3-D array of shape (z, y, x).

    Parameters
    ----------
    folder      : path to the folder containing the .tif slices
    step        : keep every n-th slice (e.g. step=2 halves the z dimension)
    downsample  : spatial downsampling factor applied with simple stride
                  (e.g. downsample=2 → half the y and x resolution)
    channel     : which colour channel to extract from RGB-saved grayscale TIFFs
                  (your Assembly data is RGB with identical channels, so 0 is fine)
    max_slices  : stop after this many slices (useful for quick tests)
    verbose     : print progress every 100 slices

    Returns
    -------
    volume : np.ndarray  shape (z, y, x)
    """
    files = sorted(glob.glob(os.path.join(folder, "*.tif")), key=_natural_key)
    files = files[::step]
    if max_slices:
        files = files[:max_slices]
    if not files:
        raise FileNotFoundError(f"No .tif files found in {folder!r}")

    # Pre-allocate once and fill slice by slice: np.stack on a list would need
    # about twice the memory at the end, which crashes laptops on big volumes.
    first = tifffile.imread(files[0])
    if first.ndim == 3:
        first = first[..., channel]
    first = first[::downsample, ::downsample]
    vol = np.empty((len(files),) + first.shape, dtype=first.dtype)
    vol[0] = first
    for i, f in enumerate(files[1:], start=1):
        im = tifffile.imread(f)
        if im.ndim == 3:          # (H, W, C) - take one channel
            im = im[..., channel]
        vol[i] = im[::downsample, ::downsample]
        if verbose and i % 100 == 0:
            print(f"   ... {i}/{len(files)} slices", flush=True)
    return vol


# ── load the ground-truth mask folders ───────────────────────────────────────

def load_mask_stack(folder: str,
                    bg_value: int = 18,
                    step: int = 1,
                    downsample: int = 1,
                    max_slices: int = None) -> np.ndarray:
    """
    Like load_tiff_stack, but binarises the result:
      pixel value > bg_value  →  True  (foreground)
      pixel value == bg_value →  False (background)

    Your Braze / Ceramic / Metal / Outer folders are CT overlay screenshots
    where the background is always intensity 18.
    """
    raw = load_tiff_stack(folder, step=step, downsample=downsample,
                          channel=0, max_slices=max_slices)
    return raw > bg_value


# ── save / load a segmentation project ───────────────────────────────────────

def save_project(folder: str, lm: LabelMap,
                 table: LabelTable = None, rois: list = None):
    """
    Write:
        <folder>/labels.npy   – the label volume (uint8)
        <folder>/labels.json  – label names and colours
        <folder>/rois.json    – ROI list (optional)
    """
    from ..tools.roi import save_rois
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "labels.npy"), lm.data)
    with open(os.path.join(folder, "labels.json"), "w") as f:
        json.dump((table or LabelTable()).to_dict(), f, indent=2)
    if rois:
        save_rois(rois, os.path.join(folder, "rois.json"))
    print(f"Project saved to {folder}/")


def load_project(folder: str):
    """
    Read a previously saved project.
    Returns: (LabelMap, LabelTable, list[ROI])
    """
    from ..tools.roi import load_rois
    data = np.load(os.path.join(folder, "labels.npy"))
    lm = LabelMap(data.shape, dtype=data.dtype)
    lm.data[...] = data
    with open(os.path.join(folder, "labels.json")) as f:
        table = LabelTable.from_dict(json.load(f))
    roi_path = os.path.join(folder, "rois.json")
    rois = load_rois(roi_path) if os.path.exists(roi_path) else []
    print(f"Project loaded from {folder}/ — shape {lm.shape}, labels {lm.labels_present()}")
    return lm, table, rois


# ── export ────────────────────────────────────────────────────────────────────

def export_tiff(lm: LabelMap, path: str):
    """Export the label volume as a multi-page TIFF (readable by ImageJ/napari/Slicer)."""
    tifffile.imwrite(path, lm.data)
    print(f"Label TIFF written to {path}")
