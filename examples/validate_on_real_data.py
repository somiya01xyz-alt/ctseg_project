"""
Sanity-checks the whole ctseg pipeline against real IIT Roorkee CT data.
Run with:  python examples/validate_on_real_data.py /path/to/Assembly_sub/raw

This is not a unit test (it needs real files) — it's a demo + smoke test that
prints what each stage produced, so you can see the tools working on your
actual ring/ceramic/metal/braze assembly instead of toy arrays.
"""
import sys, os
import numpy as np

from ctseg.fileio.files import load_tiff_stack, load_mask_stack, save_project, load_project, export_tiff
from ctseg.core.labelmap import LabelMap
from ctseg.core.label_table import LabelTable
from ctseg.tools.select import region_grow_auto, threshold_mask, largest_component
from ctseg.tools.brush import paint_stroke
from ctseg.tools.morphology import apply_morphology
from ctseg.tools.propagate import propagate_slices, interpolate_between_slices
from ctseg.evaluate import dice, per_slice_dice


def main(root, downsample=2):
    print(f"(running at 1/{downsample} resolution to save memory)")
    print("1) Loading CT volume (Assembly) ...")
    volume = load_tiff_stack(os.path.join(root, "Assembly"), downsample=downsample)
    print(f"   shape={volume.shape} dtype={volume.dtype} "
          f"min={volume.min()} max={volume.max()}")

    print("\n2) Loading ground-truth material masks ...")
    # Fail fast (before reading thousands of files) if masks don't match the CT.
    import glob, tifffile
    for name in ["Ceramic", "Metal", "Braze", "Outer"]:
        f0 = sorted(glob.glob(os.path.join(root, name, "*.tif")))[0]
        h, w = tifffile.imread(f0).shape[:2]
        mshape = (-(-h // downsample), -(-w // downsample))
        if mshape != volume.shape[1:]:
            sys.exit(f"\nSTOP: {name} masks are {mshape} per slice but the CT is "
                     f"{volume.shape[1:]}.\nThis dataset's masks are not pixel-aligned "
                     f"with the CT (this is the case for set2/Assembly_1).\n"
                     f"Run validation on data/set1 instead.")
    gt = {}
    for name in ["Ceramic", "Metal", "Braze", "Outer"]:
        gt[name] = load_mask_stack(os.path.join(root, name), downsample=downsample)
        print(f"   {name:8s} shape={gt[name].shape} "
              f"foreground_voxels={gt[name].sum()}")

    assert gt["Ceramic"].shape == volume.shape, "mask/volume slice count mismatch"

    print("\n3) Building a LabelMap and label table ...")
    lm = LabelMap(volume.shape)
    table = LabelTable()
    LABEL = {}
    for name in ["Ceramic", "Metal", "Braze", "Outer"]:
        LABEL[name] = table.add(name)
    print(f"   labels: {table.to_dict()}")

    print("\n4) Region-growing the Ceramic ring on the middle slice ...")
    zc = volume.shape[0] // 2
    ys, xs = np.where(gt["Ceramic"][zc])
    assert len(ys) > 0, "no ceramic in the middle slice of this subset"
    seed = (zc, int(ys[len(ys) // 2]), int(xs[len(xs) // 2]))   # a real foreground pixel
    mask3d = region_grow_auto(volume, seed, k=2.0, radius=4)
    lm.paint_slice(zc, mask3d[zc], label=LABEL["Ceramic"])
    d = dice(lm.get_mask(LABEL["Ceramic"], z=zc), gt["Ceramic"][zc])
    print(f"   seed={seed}  painted_voxels={mask3d[zc].sum()}  "
          f"Dice vs ground truth on that slice = {d:.3f}")

    print("\n5) Brush: manually touching up one spot ...")
    paint_stroke(lm, [(zc, seed[1] + 5, seed[2] + 5)], radius=6, label=LABEL["Ceramic"])
    print(f"   voxels on slice {zc} after brush: {lm.get_mask(LABEL['Ceramic'], z=zc).sum()}")

    print("\n6) Morphology: closing small gaps, keeping the largest blob ...")
    apply_morphology(lm, LABEL["Ceramic"], "close", radius=2, z=zc)
    cleaned = largest_component(lm.get_mask(LABEL["Ceramic"], z=zc))
    lm.set_label_mask(LABEL["Ceramic"], cleaned, z=zc)
    print(f"   voxels after close+largest_component: {cleaned.sum()}")

    print("\n7) Undo check ...")
    before = lm.get_mask(LABEL["Ceramic"], z=zc).sum()
    lm.undo()
    after = lm.get_mask(LABEL["Ceramic"], z=zc).sum()
    print(f"   before undo={before}  after undo={after}  (should differ)")
    lm.redo()

    print("\n8) Slice propagation: copy the Ceramic mask forward 5 slices ...")
    n = propagate_slices(lm, volume, LABEL["Ceramic"], z_start=zc, z_end=min(zc + 5, volume.shape[0] - 1))
    for z in range(zc, min(zc + 6, volume.shape[0])):
        dz = dice(lm.get_mask(LABEL["Ceramic"], z=z), gt["Ceramic"][z])
        print(f"   slice {z}: Dice vs ground truth = {dz:.3f}")

    print("\n9) Shape-based interpolation: fill between two segmented keyframes ...")
    lm2 = LabelMap(volume.shape)
    lm2.paint_slice(zc, gt["Ceramic"][zc], label=1)
    lm2.paint_slice(zc + 6, gt["Ceramic"][zc + 6], label=1)
    interpolate_between_slices(lm2, 1, zc, zc + 6)
    scores = per_slice_dice(lm2.data[zc:zc + 7] == 1, gt["Ceramic"][zc:zc + 7])
    print(f"   per-slice Dice {zc}..{zc+6}: {np.round(scores, 3).tolist()}")

    print("\n10) Save / reload a project ...")
    out = "/tmp/ctseg_demo_project"
    save_project(out, lm, table)
    lm_r, table_r, _ = load_project(out)
    assert np.array_equal(lm_r.data, lm.data)
    print("   round-trip OK")

    print("\n11) Export a multi-page label TIFF ...")
    export_tiff(lm, "/tmp/ctseg_demo_labels.tif")

    print("\nAll stages completed without errors.")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    ds = int(sys.argv[2]) if len(sys.argv) > 2 else 2     # 2 = half resolution (safe default)
    main(root, ds)
