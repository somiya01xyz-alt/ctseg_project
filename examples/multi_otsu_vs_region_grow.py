"""
Multi-Otsu vs the ground-truth material folders: one Dice table for the slide.
Run:  python examples/multi_otsu_vs_region_grow.py /path/to/set1 [downsample] [classes] [z_start] [z_end]

Defaults score the SAME held-out slices as the region-growing results (160-319 at the
original slice numbering), so the two tables can sit side by side.

Multi-Otsu gives anonymous classes ordered by brightness, so for every material (Ceramic, Metal,
Braze, Outer) this tries each class and each union of neighbouring classes and reports the best
Dice. That choice looks at the ground truth, so it is an ORACLE (upper bound) for Multi-Otsu.
It needs no click at all, unlike region growing.
Thresholds are computed on the whole volume (no ground truth involved), only scoring is restricted.
"""
import itertools
import os
import sys

import numpy as np

from ctseg.evaluate import dice
from ctseg.fileio.files import load_mask_stack, load_tiff_stack
from ctseg.tools.select import multi_otsu


def main(root, downsample=2, classes=4, z0=160, z1=319):
    vol = load_tiff_stack(os.path.join(root, "Assembly"), downsample=downsample)
    labels, th = multi_otsu(vol, classes=classes, sample_step=8)
    z1 = min(z1, vol.shape[0] - 1)
    print(f"thresholds {th.tolist()}  (every 8th voxel of {vol.shape}); scored on slices {z0}-{z1}")
    print("\n| Material | Best class(es), oracle | Multi-Otsu Dice (held-out slices) |\n|---|---|---|")
    lab = labels[z0:z1 + 1]
    for name in ["Ceramic", "Metal", "Braze", "Outer"]:
        gt = load_mask_stack(os.path.join(root, name), downsample=downsample)[z0:z1 + 1]
        best = (-1, None)
        for a, b in itertools.combinations_with_replacement(range(classes), 2):
            d = dice((lab >= a) & (lab <= b), gt)
            if d > best[0]:
                best = (d, (a, b))
        a, b = best[1]
        print(f"| {name} | {a if a == b else f'{a}-{b}'} | {best[0]:.3f} |")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 2, int(a[3]) if len(a) > 3 else 4,
         int(a[4]) if len(a) > 4 else 160, int(a[5]) if len(a) > 5 else 319)
