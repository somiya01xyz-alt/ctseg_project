"""
Checks whether the 'ground truth' mask images contain things that are NOT material
(for example a dashed white selection box burned into the screenshot).

For each material folder it samples every 10th slice and reports:
  - average number of 'foreground' pixels per slice  (pixel value > 18)
  - how many pixels are foreground in >= 90% of the sampled slices ("always on").
    Real material changes from slice to slice; a burned-in box does not.
  - the pixel values of those always-on pixels
It also saves overlay_<Material>.png: grey = foreground on the middle slice,
red = always-on pixels. Look at these pictures.

Run:
    python -u examples/inspect_gt_overlay.py "C:\\Users\\somiy\\B2_local" > overlay.txt 2>&1
"""
import sys, os, glob
import numpy as np
import tifffile
from ctseg.fileio.files import _natural_key

DS = 2
BG = 18


def main(root):
    for name in ["Ceramic", "Metal", "Braze", "Outer"]:
        files = sorted(glob.glob(os.path.join(root, name, "*.tif")), key=_natural_key)
        if not files:
            print(name, "- no files"); continue
        picks = files[::10]
        stack = []
        for f in picks:
            im = tifffile.imread(f)
            im = im[..., 0] if im.ndim == 3 else im
            stack.append(im[::DS, ::DS])
        stack = np.stack(stack)
        fg = stack > BG
        frac = fg.mean(0)
        always = frac >= 0.9
        per_slice = fg.reshape(len(fg), -1).sum(1)
        vals = stack[:, always] if always.any() else np.array([[0]])
        uniq, cnt = np.unique(vals, return_counts=True)
        top = [int(u) for u, c in sorted(zip(uniq, cnt), key=lambda t: -t[1])[:4]]
        print(f"{name:8s} slices_sampled={len(picks)}  avg_fg_px_per_slice={per_slice.mean():.0f}  "
              f"always_on_px={int(always.sum())}  "
              f"always_on_share_of_fg={always.sum() / max(per_slice.mean(), 1):.0%}  "
              f"always_on_values={top}")
        try:
            from PIL import Image
            mid = fg[len(fg) // 2]
            rgb = np.zeros(mid.shape + (3,), np.uint8)
            rgb[mid] = (120, 120, 120)
            rgb[always] = (255, 0, 0)
            Image.fromarray(rgb).save(f"overlay_{name}.png")
        except Exception as e:
            print("   (no picture:", e, ")")
    print("Done. Pictures: overlay_Ceramic.png, overlay_Metal.png, overlay_Braze.png, overlay_Outer.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/set1")
