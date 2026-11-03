"""
Why do the derived Braze / Outer lines miss the ground truth?  Draws one slice and prints distances.

    python -u examples/inspect_thin.py C:\\Users\\somiy\\B2_local 2 [slice]   (no slice = a slice that has Braze)

Writes thin_overlay.png: CT in grey, ground-truth Braze in RED, ground-truth Outer in GREEN,
our derived Braze (thickness 4) in BLUE and derived Outer (thickness 1) in YELLOW.
Prints, for each ground-truth line, how far (px) it lies from the Ceramic / Metal edges.
"""
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi

from ctseg.fileio.files import load_mask_stack
from ctseg.tools.morphology import interface, outer_boundary, outline


def main(root, ds, z):
    names = ["Ceramic", "Metal", "Braze", "Outer"]
    files = {n: sorted(f for f in os.listdir(os.path.join(root, n)) if f.lower().endswith((".tif", ".tiff"))) for n in names}
    present = [i for i, f in enumerate(files["Braze"])
               if (tifffile.imread(os.path.join(root, "Braze", f)) > 18).any()]
    print(f"Braze ground truth is present on {len(present)} slices"
          + (f": {present[0]} .. {present[-1]}" if present else ""))
    if z < 0 and present:
        z = present[len(present) // 2]
    elif z < 0:
        z = 240
    ct_dir = os.path.join(root, "Assembly")
    ct_files = sorted(f for f in os.listdir(ct_dir) if f.lower().endswith((".tif", ".tiff")))
    ct = tifffile.imread(os.path.join(ct_dir, ct_files[z]))
    if ct.ndim == 3:
        ct = ct[..., 0]
    ct = ct[::ds, ::ds]
    m = {}
    for n in names:
        raw = tifffile.imread(os.path.join(root, n, files[n][z]))
        if raw.ndim == 3:
            raw = raw[..., 0]
        m[n] = (raw > 18)[::ds, ::ds]
    cm = m["Ceramic"] | m["Metal"]
    pb = interface(m["Ceramic"], m["Metal"], 4)
    po = outer_boundary(cm, 1)

    def dist_to(edge, gt):
        d = ndi.distance_transform_edt(~edge)
        v = d[gt]
        return (f"median {np.median(v):.1f}px, 90% within {np.percentile(v, 90):.1f}px") if v.size else "empty"
    print(f"slice {z}  pixels: " + ", ".join(f"{n} {int(m[n].sum())}" for n in names))
    for gtn in ["Braze", "Outer"]:
        for en, edge in [("edge of Ceramic", outline(m["Ceramic"], 1)), ("edge of Metal", outline(m["Metal"], 1)),
                         ("edge of Ceramic|Metal", po)]:
            print(f"  GT {gtn} -> distance to {en}: {dist_to(edge, m[gtn])}")
    g = np.clip(ct.astype(float) / max(ct.max(), 1), 0, 1)
    rgb = np.stack([g, g, g], -1)
    for mask, col in [(pb, (0, 0.4, 1)), (po, (1, 1, 0)), (m["Braze"], (1, 0, 0)), (m["Outer"], (0, 1, 0))]:
        mm = ndi.binary_dilation(mask, iterations=1)
        rgb[mm] = col
    from PIL import Image
    Image.fromarray((rgb * 255).astype(np.uint8)).save("thin_overlay.png")
    print("saved thin_overlay.png")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2,
         int(sys.argv[3]) if len(sys.argv) > 3 else -1)
