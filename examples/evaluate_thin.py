"""
Thin structures (Braze, Outer): derive them from the thick regions instead of growing.

    python examples/evaluate_thin.py data/set1 [downsample]

Braze  = interface(Ceramic, Metal, t)   (a thin band where the two regions touch)
Outer  = outer_boundary(Ceramic | Metal, t) (only the outermost edge; holes/inner rings ignored)

The thickness t is chosen on TRAIN slices (0-159) and scored on TEST slices (160-319).
Two scores per structure:
  Dice          exact pixel overlap (very harsh for a 1-2 px line)
  tolerant F1   a predicted pixel counts as correct if it lies within 2 px of the
                ground-truth line, and a ground-truth pixel counts as found if a
                predicted pixel lies within 2 px of it. Fair for lines this thin.
Ceramic/Metal here are the ground-truth masks, i.e. what a user would have after
segmenting those two regions with the other tools. Writes results/thin_results.md.
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

from ctseg.evaluate import dice
from ctseg.fileio.files import load_mask_stack
from ctseg.tools.morphology import interface, outer_boundary

TRAIN, TEST = (0, 160), (160, 320)
TOL = 2
THICKNESS = [1, 2, 3, 4, 6]


def tolerant_f1(pred, gt, tol=TOL):
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    d_gt = ndi.distance_transform_edt(~gt)
    d_pr = ndi.distance_transform_edt(~pred)
    prec = float((d_gt[pred] <= tol).mean())
    rec = float((d_pr[gt] <= tol).mean())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def is_box(gt2d):
    """True if the ground-truth 'line' is a rectangle (an annotation box in the export),
    not a ring: nearly all its pixels lie on the rows/columns of its bounding box."""
    ys, xs = np.nonzero(gt2d)
    if ys.size == 0:
        return False
    on_edge = (ys == ys.min()) | (ys == ys.max()) | (xs == xs.min()) | (xs == xs.max())
    return float(on_edge.mean()) > 0.8


def score(pred, gt, rng, ring_only=False):
    zs = [z for z in range(*rng) if gt[z].any() and not (ring_only and is_box(gt[z]))]
    if not zs:
        return float("nan"), float("nan"), 0
    d = [dice(pred[z], gt[z]) for z in zs]
    f = [tolerant_f1(pred[z], gt[z]) for z in zs]
    return float(np.mean(d)), float(np.mean(f)), len(zs)


def main(root, ds):
    m = {n: load_mask_stack(os.path.join(root, n), downsample=ds)
         for n in ["Ceramic", "Metal", "Braze", "Outer"]}
    cm = m["Ceramic"] | m["Metal"]
    makers = {"Braze": lambda t: interface(m["Ceramic"], m["Metal"], t),
              "Outer (inside edge)": lambda t: outer_boundary(cm, t),
              "Outer (just outside)": lambda t: outer_boundary(cm, t, outside=True)}
    truth = {"Braze": "Braze", "Outer (inside edge)": "Outer", "Outer (just outside)": "Outer"}
    for n in ("Braze", "Outer"):
        boxes = [z for z in range(320) if m[n][z].any() and is_box(m[n][z])]
        print(f"ground-truth {n} looks like a rectangle (not a ring) on {len(boxes)} of 320 slices", flush=True)
    lines = ["# Thin structures from thick regions (held-out slices 160-319)", "",
             f"Tolerant F1 uses a {TOL}-px tolerance. Thickness chosen on slices 0-159, "
             "using only slices where the ground truth is a ring (not an annotation rectangle).", "",
             "| Structure | Thickness | Dice (all) | F1 (all) | Dice (ring slices) | F1 (ring slices) | ring slices |",
             "|---|---|---|---|---|---|---|"]
    for name, make in makers.items():
        best = None
        for t in THICKNESS:
            pred = make(t)
            tr = score(pred, m[truth[name]], TRAIN, ring_only=True)[1]
            print(f"{name} thickness {t}: train tolerant F1 (ring slices) {tr:.3f}", flush=True)
            if best is None or tr > best[0]:
                best = (tr, t, pred)
        da, fa, _ = score(best[2], m[truth[name]], TEST)
        dr, fr, nr = score(best[2], m[truth[name]], TEST, ring_only=True)
        lines.append(f"| {name} | {best[1]} | {da:.3f} | {fa:.3f} | {dr:.3f} | {fr:.3f} | {nr} |")
    os.makedirs("results", exist_ok=True)
    open("results/thin_results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/set1",
         int(sys.argv[2]) if len(sys.argv) > 2 else 2)
