"""
Day 1 task: load your CT stack and look at it.

Usage:
    python examples/day1_quickstart.py data/set1/Assembly
"""
import sys
import matplotlib.pyplot as plt
from ctseg.fileio.files import load_tiff_stack


def main(folder):
    volume = load_tiff_stack(folder)
    print("shape (z, y, x):", volume.shape)
    print("dtype:", volume.dtype)
    print("min/max intensity:", volume.min(), volume.max())

    z = volume.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[z], cmap="gray");            axes[0].set_title(f"slice {z} (axial)")
    axes[1].imshow(volume[:, volume.shape[1] // 2, :], cmap="gray", aspect="auto")
    axes[1].set_title("coronal view")
    axes[2].imshow(volume[:, :, volume.shape[2] // 2], cmap="gray", aspect="auto")
    axes[2].set_title("sagittal view")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    out = "day1_views.png"
    plt.savefig(out, dpi=120)
    print(f"Saved a 3-view preview to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/set1/Assembly")
