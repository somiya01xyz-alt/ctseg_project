"""Usage: python inspect_data.py "C:\\path\\to\\folder_containing_Assembly_Ceramic_etc"
Prints size/type of the middle slice of each folder and saves a preview PNG. Changes nothing."""
import sys, os, glob
import numpy as np, tifffile
root = sys.argv[1]
for name in ["Assembly", "Ceramic", "Metal", "Braze", "Outer"]:
    files = sorted(glob.glob(os.path.join(root, name, "*.tif")))
    if not files:
        print(name, "-> no files"); continue
    f = files[len(files)//2]
    im = tifffile.imread(f)
    g = im[..., 0] if im.ndim == 3 else im
    vals, cnt = np.unique(g, return_counts=True)
    top = sorted(zip(cnt, vals), reverse=True)[:4]
    print(f"{name:9s} files={len(files)} example={os.path.basename(f)} shape={im.shape} "
          f"dtype={im.dtype} min={g.min()} max={g.max()} most_common_values={[int(v) for c,v in top]}")
    try:
        from PIL import Image
        Image.fromarray(g[::4, ::4].astype(np.uint8)).save(f"preview_{name}.png")
    except Exception as e:
        print("   (no preview:", e, ")")
print("Done. Preview PNGs are in the current folder.")
