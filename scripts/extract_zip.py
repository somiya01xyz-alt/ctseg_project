"""
One-time setup: unzip Assembly.zip / Assembly_1.zip into plain folders so
load_tiff_stack() / load_mask_stack() can read them directly.

Usage:
    python scripts/extract_zip.py /path/to/Assembly.zip data/set1
    python scripts/extract_zip.py /path/to/Assembly_1.zip data/set2

Afterwards you will have:
    data/set1/Assembly/*.tif   data/set1/Ceramic/*.tif   data/set1/Metal/*.tif  ...
"""
import sys
import zipfile
from pathlib import Path


def main(zip_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".tif")]
        print(f"Extracting {len(names)} .tif files from {zip_path} -> {out_dir}")
        z.extractall(out_dir, members=names)
    print("Done. Folders:", sorted(p.name for p in out_dir.iterdir() if p.is_dir()))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
