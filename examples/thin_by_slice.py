"""
Where does the derived Braze line fail?  Scores every 10th slice and saves the worst one.

    python -u examples/thin_by_slice.py C:\\Users\\somiy\\B2_local 2

Prints, per sampled slice: pixels of Ceramic / Metal / Braze, whether Ceramic and Metal
touch, and the tolerant F1 of the derived Braze (thickness 4). Saves worst_braze_slice.png
(CT grey, true Braze red, our Braze blue).
"""
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi

from evaluate_thin import tolerant_f1          # same folder
from ctseg.tools.morphology import interface


def read(root, name, z, ds, files, thresh=True):
    a = tifffile.imread(os.path.join(root, name, files[name][z]))
    if a.ndim == 3:
        a = a[..., 0]
    a = a[::ds, ::ds]
    return a > 18 if thresh else a


def main(root, ds):
    files = {n: sorted(f for f in os.listdir(os.path.join(root, n)) if f.lower().endswith((".tif", ".tiff")))
             for n in ["Assembly", "Ceramic", "Metal", "Braze"]}
    rows = []
    for z in range(0, 320, 10):
        c, m, b = (read(root, n, z, ds, files) for n in ["Ceramic", "Metal", "Braze"])
        pred = interface(c, m, 4)
        touch = int((ndi.binary_dilation(c, iterations=2) & m).sum())
        f1 = tolerant_f1(pred, b) if b.any() else float("nan")
        rows.append((z, f1, int(c.sum()), int(m.sum()), int(b.sum()), touch))
        print(f"slice {z:3d}: Ceramic {c.sum():7d}  Metal {m.sum():7d}  Braze {b.sum():5d}  "
              f"Ceramic-Metal contact px {touch:6d}  Braze F1 {f1:.3f}", flush=True)
    ok = [r for r in rows if not np.isnan(r[1])]
    worst = min(ok, key=lambda r: r[1])
    z = worst[0]
    print(f"worst sampled slice: {z} (F1 {worst[1]:.3f})")
    ct = read(root, "Assembly", z, ds, files, thresh=False).astype(float)
    g = np.clip(ct / max(ct.max(), 1), 0, 1)
    rgb = np.stack([g, g, g], -1)
    c, m, b = (read(root, n, z, ds, files) for n in ["Ceramic", "Metal", "Braze"])
    rgb[ndi.binary_dilation(interface(c, m, 4))] = (0, 0.4, 1)
    rgb[ndi.binary_dilation(b)] = (1, 0, 0)
    from PIL import Image
    Image.fromarray((rgb * 255).astype(np.uint8)).save("worst_braze_slice.png")
    print("saved worst_braze_slice.png")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
