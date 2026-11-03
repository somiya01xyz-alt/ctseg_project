"""
Experiments for the report (train/test split + tool comparisons).

Usage:
    python examples/run_experiments.py data/set1 [downsample]
    (or pass a folder of cached .npy stacks with --cache DIR)

What it does
------------
A. TUNE on TRAIN slices (z 0-159): choose region-growing settings (k, radius)
   per material by mean Dice vs ground truth.
B. TEST on HELD-OUT slices (z 160-319): score the chosen settings, and compare
   them with the untuned default and with morphological clean-up.
C. 2D vs 3D region growing (accuracy and time) on a 32-slice block of test data.
D. Slice propagation vs shape interpolation: Dice as a function of distance
   from the hand-segmented key slice (test slices only).
E. Data-structure benchmark: LabelMap undo memory and brush timing.

The "user click" (seed) is simulated: the deepest interior point of the
ground-truth mask on that slice, i.e. a click well inside the material.
So the numbers measure the TOOL, not a fully automatic method.
Outputs go to results/ (results.json, results.md, figures).
"""
import sys, os, json, time
import numpy as np
from scipy import ndimage as ndi

from ctseg.core.labelmap import LabelMap
from ctseg.tools.select import region_grow_auto, largest_component
from ctseg.tools.morphology import closing
from ctseg.tools.propagate import propagate_slices, interpolate_between_slices
from ctseg.tools.brush import paint_stroke
from ctseg.evaluate import dice, iou

TRAIN = (0, 160)
TEST = (160, 320)
K_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
R_GRID = [4, 8, 12, 16]
MIN_AREA = 5000   # slices where the material covers fewer pixels are skipped (material absent)
DEFAULT = dict(k=2.0, radius=4)
MATERIALS = ["Ceramic", "Metal"]
OUT = "results"


def load(root, ds, cache=None):
    if cache:
        vol = np.load(os.path.join(cache, "Assembly.npy"))
        gt = {n: np.load(os.path.join(cache, f"{n}.npy")) for n in ["Ceramic", "Metal", "Braze", "Outer"]}
        return vol, gt
    from ctseg.fileio.files import load_tiff_stack, load_mask_stack
    vol = load_tiff_stack(os.path.join(root, "Assembly"), downsample=ds)
    gt = {n: load_mask_stack(os.path.join(root, n), downsample=ds) for n in ["Ceramic", "Metal", "Braze", "Outer"]}
    return vol, gt


def click_seed(mask2d):
    """Simulated user click: the deepest interior pixel of the material."""
    if not mask2d.any():
        return None
    d = ndi.distance_transform_edt(mask2d)
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return int(y), int(x)


def grow_slice(vol, gt2d, z, k, radius, clean=False):
    seed = click_seed(gt2d)
    if seed is None:
        return None
    pred = region_grow_auto(vol[z], seed, k=k, radius=radius)
    if clean:
        pred = largest_component(closing(pred, 2))
    return pred


def mean_dice(vol, gt, z_list, k, radius, clean=False, min_area=MIN_AREA):
    ds_ = []
    for z in z_list:
        if gt[z].sum() < min_area:
            continue
        p = grow_slice(vol, gt[z], z, k, radius, clean)
        if p is not None:
            ds_.append(dice(p, gt[z]))
    return float(np.mean(ds_)), float(np.std(ds_))


def main(root, ds=2, cache=None):
    os.makedirs(OUT, exist_ok=True)
    vol, gt = load(root, ds, cache)
    print("volume", vol.shape)
    res = {"volume_shape": list(vol.shape), "train": TRAIN, "test": TEST}

    train_z = list(range(TRAIN[0], TRAIN[1], 10))
    test_z = list(range(TEST[0], TEST[1], 5))

    # ---- A. tune on train ----
    print("\nA. Tuning on TRAIN slices", TRAIN)
    best = {}
    grid_rows = []
    for m in MATERIALS:
        scores = {}
        for k in K_GRID:
            for r in R_GRID:
                mu, _ = mean_dice(vol, gt[m], train_z, k, r)
                scores[(k, r)] = mu
                grid_rows.append((m, k, r, mu))
        (kb, rb), sb = max(scores.items(), key=lambda t: t[1])
        best[m] = dict(k=kb, radius=rb, train_dice=sb)
        print(f"   {m}: best k={kb} radius={rb} train Dice={sb:.3f}")
    res["tuned"] = best
    res["grid"] = [dict(material=a, k=b, radius=c, train_dice=d) for a, b, c, d in grid_rows]

    # ---- B. test on held-out ----
    print("\nB. Held-out TEST slices", TEST)
    rows = {}
    for m in MATERIALS:
        kb, rb = best[m]["k"], best[m]["radius"]
        tuned = mean_dice(vol, gt[m], test_z, kb, rb)
        default = mean_dice(vol, gt[m], test_z, DEFAULT["k"], DEFAULT["radius"])
        cleaned = mean_dice(vol, gt[m], test_z, kb, rb, clean=True)
        rows[m] = dict(tuned=tuned, default=default, tuned_plus_cleanup=cleaned)
        print(f"   {m}: default {default[0]:.3f}  tuned {tuned[0]:.3f}  tuned+close/largest {cleaned[0]:.3f}")
    # thin structures (documented limitation)
    for m in ["Braze", "Outer"]:
        mu, sd = mean_dice(vol, gt[m], test_z, DEFAULT["k"], DEFAULT["radius"], min_area=50)
        rows[m] = dict(default=(mu, sd))
        print(f"   {m} (thin line, default settings): {mu:.3f}")
    res["test"] = rows

    # ---- C. 2D vs 3D region growing ----
    print("\nC. 2D vs 3D region growing on z 200-231 (Ceramic)")
    z0, z1 = 200, 232
    blk = vol[z0:z1]
    g = gt["Ceramic"][z0:z1]
    kb, rb = best["Ceramic"]["k"], best["Ceramic"]["radius"]
    t = time.time()
    p2 = np.stack([region_grow_auto(blk[i], click_seed(g[i]), k=kb, radius=rb) for i in range(len(blk))])
    t2 = time.time() - t
    mid = len(blk) // 2
    t = time.time()
    p3 = region_grow_auto(blk, (mid,) + click_seed(g[mid]), k=kb, radius=rb)
    t3 = time.time() - t
    res["two_vs_three_d"] = dict(
        dice_2d_per_slice_click=dice(p2, g), time_2d_s=t2, clicks_2d=len(blk),
        dice_3d_single_click=dice(p3, g), time_3d_s=t3, clicks_3d=1)
    print("  ", res["two_vs_three_d"])

    # ---- D. propagation vs interpolation ----
    print("\nD. Propagation vs interpolation (key slice z=200)")
    zk = 200
    dist = list(range(0, 21))
    prop = {}
    interp = {}
    for m in MATERIALS:
        lm = LabelMap(vol.shape)
        lm.paint_slice(zk, gt[m][zk], 1)
        propagate_slices(lm, vol, 1, zk, zk + 20)
        prop[m] = [dice(lm.get_mask(1, z=zk + d), gt[m][zk + d]) for d in dist]
        lm2 = LabelMap(vol.shape)
        lm2.paint_slice(zk, gt[m][zk], 1)
        lm2.paint_slice(zk + 20, gt[m][zk + 20], 1)
        interpolate_between_slices(lm2, 1, zk, zk + 20)
        interp[m] = [dice(lm2.get_mask(1, z=zk + d), gt[m][zk + d]) for d in dist]
        print(f"   {m}: propagate Dice at +1/+5/+10/+20 = "
              f"{prop[m][1]:.3f}/{prop[m][5]:.3f}/{prop[m][10]:.3f}/{prop[m][20]:.3f}; "
              f"interpolate (keys at 0 and 20) min={min(interp[m][1:20]):.3f}")
    res["propagation"] = dict(distance=dist, propagate=prop, interpolate=interp)

    # ---- E. data structure benchmark ----
    print("\nE. LabelMap benchmark")
    lm = LabelMap(vol.shape)
    pts = [(240, 300 + i, 300 + 2 * i) for i in range(100)]
    t = time.time(); paint_stroke(lm, pts, radius=10, label=1); tb = time.time() - t
    entry = lm._undo[-1]
    undo_bytes = entry[1].nbytes + entry[2].nbytes
    t = time.time(); lm.undo(); tu = time.time() - t
    t = time.time(); lm.redo(); tr = time.time() - t
    pts3 = [(220 + i // 4, 300 + i, 300 + i) for i in range(40)]
    t = time.time(); paint_stroke(lm, pts3, radius=6, label=2, mode="3d"); tb3 = time.time() - t
    res["labelmap"] = dict(volume_bytes=int(lm.data.nbytes), brush2d_s=tb, undo_entry_bytes=int(undo_bytes),
                           undo_s=tu, redo_s=tr, brush3d_s=tb3,
                           full_copy_would_be_bytes=int(lm.data.nbytes))
    print("  ", res["labelmap"])

    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)
    make_figures(vol, gt, best, res)
    print("\nDone. Results in", OUT)


def make_figures(vol, gt, best, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zshow = 240
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for row, m in enumerate(MATERIALS):
        kb, rb = best[m]["k"], best[m]["radius"]
        p = grow_slice(vol, gt[m][zshow], zshow, kb, rb)
        axs[row, 0].imshow(vol[zshow], cmap="gray"); axs[row, 0].set_title(f"CT slice {zshow}")
        axs[row, 1].imshow(gt[m][zshow], cmap="gray"); axs[row, 1].set_title(f"{m}: ground truth")
        ov = np.zeros(vol[zshow].shape + (3,))
        ov[..., 1] = gt[m][zshow]; ov[..., 0] = p
        axs[row, 2].imshow(ov); axs[row, 2].set_title(f"{m}: tool (red) vs GT (green)\nDice={dice(p, gt[m][zshow]):.3f}")
        for a in axs[row]: a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_region_grow.png"), dpi=110); plt.close()

    d = res["propagation"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axs, MATERIALS):
        ax.plot(d["distance"], d["propagate"][m], "o-", label="slice-by-slice propagation (1 key slice)")
        ax.plot(d["distance"], d["interpolate"][m], "s-", label="shape interpolation (2 key slices)")
        ax.set_title(m); ax.set_xlabel("slices from key slice"); ax.set_ylabel("Dice"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    axs[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_propagation.png"), dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in MATERIALS:
        kb, rb = best[m]["k"], best[m]["radius"]
        zs = list(range(TEST[0], TEST[1], 5))
        zs = [z for z in zs if gt[m][z].sum() >= MIN_AREA]
        sc = [dice(grow_slice(vol, gt[m][z], z, kb, rb), gt[m][z]) for z in zs]
        ax.plot(zs, sc, label=m)
    ax.set_xlabel("slice (held-out test range)"); ax.set_ylabel("Dice"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_test_dice.png"), dpi=110); plt.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cache = None
    if "--cache" in sys.argv:
        cache = sys.argv[sys.argv.index("--cache") + 1]
        args = [a for a in args if a != cache]
    root = args[0] if args else "data/set1"
    ds = int(args[1]) if len(args) > 1 else 2
    main(root, ds, cache)
