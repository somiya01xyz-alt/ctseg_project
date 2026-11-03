"""
Finds better settings for one-click region growing.

Why: in validate_on_real_data.py the grown Ceramic region on slice 160 had
358,529 pixels while the ground truth has about 161,000, so Dice was only 0.61
because the region LEAKED into neighbouring grey material. The seed point and
the width of the intensity range (k) decide whether it leaks.

This script tries many (k, radius) settings and several seed points per
material, on 3 slices, and prints the best settings with their Dice scores.
It only reads a few slices, so it takes about a minute.

Run:
    python -u examples/tune_region_grow.py "C:\\Users\\somiy\\B2_local" > tuning.txt 2>&1
"""
import sys, os, glob, csv
import numpy as np
from scipy import ndimage as ndi
import tifffile

from ctseg.fileio.files import _natural_key
from ctseg.evaluate import dice

DOWNSAMPLE = 2
MATERIALS = ["Ceramic", "Metal", "Braze", "Outer"]
BG = 18
K_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
RADII = [2, 4, 8]
N_SEEDS = 8


def read_slice(files, i, mask=False):
    im = tifffile.imread(files[i])
    if im.ndim == 3:
        im = im[..., 0]
    im = im[::DOWNSAMPLE, ::DOWNSAMPLE]
    return (im > BG) if mask else im


def grow_2d(img, seed, k, radius):
    """Same rule as region_grow_auto (mean +/- k*std around the seed) but in 2-D."""
    y, x = seed
    patch = img[max(y - radius, 0):y + radius + 1, max(x - radius, 0):x + radius + 1].astype(float)
    mu, sigma = patch.mean(), max(patch.std(), 1.0)
    cand = (img >= mu - k * sigma) & (img <= mu + k * sigma)
    if not cand[y, x]:
        return np.zeros(img.shape, bool)
    lab, _ = ndi.label(cand)
    return lab == lab[y, x]


def pick_seeds(gt, n):
    """Seeds well inside the ground-truth region: spread over its interior,
    far from the edges (distance transform), not just the median pixel."""
    dist = ndi.distance_transform_edt(gt)
    thresh = 0.5 * dist.max()
    ys, xs = np.nonzero(dist >= thresh)
    if len(ys) == 0:
        ys, xs = np.nonzero(gt)
    if len(ys) == 0:
        return []
    idx = np.linspace(0, len(ys) - 1, n).astype(int)
    order = np.lexsort((xs, ys))
    return [(int(ys[order[i]]), int(xs[order[i]])) for i in idx]


def main(root):
    ct_files = sorted(glob.glob(os.path.join(root, "Assembly", "*.tif")), key=_natural_key)
    nz = len(ct_files)
    zs = [nz // 2 - 10, nz // 2, nz // 2 + 10]
    print(f"{nz} slices, testing slices {zs}, downsample {DOWNSAMPLE}")
    rows = []
    for name in MATERIALS:
        gt_files = sorted(glob.glob(os.path.join(root, name, "*.tif")), key=_natural_key)
        if len(gt_files) != nz:
            print(f"{name}: {len(gt_files)} files vs {nz} CT slices - skipped")
            continue
        data = []
        for z in zs:
            img = read_slice(ct_files, z)
            gt = read_slice(gt_files, z, mask=True)
            if gt.shape != img.shape:
                print(f"{name}: mask {gt.shape} != CT {img.shape} - skipped")
                data = []
                break
            if gt.sum() > 0:
                data.append((img, gt))
        if not data:
            continue
        # baseline = what validate_on_real_data.py did: median foreground pixel, k=2, r=4
        base = []
        for img, gt in data:
            ys, xs = np.where(gt)
            seed = (int(ys[len(ys) // 2]), int(xs[len(xs) // 2]))
            base.append(dice(grow_2d(img, seed, 2.0, 4), gt))
        baseline = float(np.mean(base))
        results = []
        for k in K_VALUES:
            for r in RADII:
                scores = []
                for img, gt in data:
                    best_seed = [dice(grow_2d(img, s, k, r), gt) for s in pick_seeds(gt, N_SEEDS)]
                    scores.append(float(np.median(best_seed)))   # median over seeds = robust
                results.append((float(np.mean(scores)), k, r))
                rows.append([name, k, r, round(float(np.mean(scores)), 4)])
        results.sort(reverse=True)
        print(f"\n== {name}: baseline (old rule) Dice = {baseline:.3f}")
        for sc, k, r in results[:3]:
            print(f"   k={k:<4} radius={r:<2} -> Dice {sc:.3f}")

    out = "tuning_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["material", "k", "radius", "median_dice_over_seeds_and_slices"])
        w.writerows(rows)
    print(f"\nAll results saved to {out}")
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/set1")
