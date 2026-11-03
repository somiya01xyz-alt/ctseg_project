# ctseg — Interactive CT Segmentation and ROI Creation Tools

A working starter project for the IIT Roorkee project brief
*"Interactive CT Segmentation and ROI Creation Tools."* Everything in
`ctseg/` is tested and has run successfully against your actual
`Assembly.zip` data. Treat this as scaffolding you understand and extend,
not a black box — every file is short and commented for that reason.

## 0. Install (10 minutes)

```bash
conda create -n ctseg python=3.11 -y
conda activate ctseg
cd ctseg_project
pip install -e .           # installs ctseg + numpy/scipy/scikit-image/tifffile
pip install -e ".[dev]"    # + pytest, matplotlib (for tests and the Day-1 script)
pip install -e ".[gui]"    # + napari, magicgui — only needed for Day 5 onward
```

Confirm it worked:
```bash
pytest tests/ -v
```
You should see 39 tests pass in well under a second. These use small made-up
arrays, not your CT data, so they run anywhere, instantly, and are your
safety net every time you change a file — run them often.

## 1. Get your data into plain folders

Your CT slices are inside `Assembly.zip` / `Assembly_1.zip` as
`Assembly/`, `Braze/`, `Ceramic/`, `Metal/`, `Outer/`. Unzip them once:

```bash
python scripts/extract_zip.py /path/to/Assembly.zip   data/set1
python scripts/extract_zip.py /path/to/Assembly_1.zip data/set2
```

`data/set1` (from `Assembly.zip`) is the one to start with — 320 slices,
clean 8-bit grayscale, and the material masks line up pixel-for-pixel with
the CT slices. `data/set2` (from `Assembly_1.zip`) is harder: the material
images are a different pixel size than the assembly images, so they need
cropping/registration before you can compare them directly — leave that for
later, if you need it at all.

## 2. Run the demo against your real data

```bash
python examples/day1_quickstart.py data/set1/Assembly
python examples/validate_on_real_data.py data/set1
```

The second script is the one to actually read start to finish — it runs
every tool in the package (region growing, brush, morphology, undo,
propagation, interpolation, save/load, export) on your real ring/ceramic/
metal/braze assembly and prints what happened at each step, including Dice
scores against your ground-truth material folders. Reading its output
top to bottom is the fastest way to see what the library can do.

## 3. How the code is organised

```
ctseg/
  core/
    labelmap.py      # the mask volume itself: paint, erase, undo/redo
    label_table.py   # names + colours for label numbers (1="Ceramic", ...)
  tools/
    brush.py         # paint/erase strokes, 2D or 3D
    roi.py            # rectangle/ellipse/polygon/freehand ROIs + stats
    select.py          # threshold, flood fill, region growing, components
    morphology.py      # dilate/erode/open/close/fill, edits a LabelMap
    propagate.py        # carry a mask across slices, two different methods
  fileio/
    files.py         # load TIFF stacks, load reference masks, save/load projects
  gui/
    napari_widget.py # builds the viewer: CT + Labels layer + tabbed panels
    panels.py        # tabs: Select (threshold, components), Edit (morphology,
                      # propagate, interpolate), ROI, Labels (names, project, Dice), View (3D, STL), Help
  evaluate.py         # dice / iou / per_slice_dice — compare two masks
tests/
  test_core.py        # 25 core unit tests
  test_gui_panels.py  # 14 tests of the napari panels (offscreen)
examples/
  day1_quickstart.py          # load + view a stack
  validate_on_real_data.py    # the full pipeline, explained above
scripts/
  extract_zip.py       # unzip your data into the folder layout above
```

**The one rule that keeps this reusable** (the brief explicitly asks for
this): nothing in `core/` or `tools/` imports napari. Only `gui/` does. That
is what lets the "unified CT application" mentioned in your brief reuse
these tools with a totally different interface later if it needs to.

## 4. The one idea the whole thing is built on

A segmentation is just a 3D array the same shape as your CT volume, where
each number means a material:

```python
import numpy as np
from ctseg.core.labelmap import LabelMap

lm = LabelMap(shape=(320, 1427, 1680))   # starts all zero = "unlabeled"
lm.data[20, 700, 800]                    # 0
lm.paint_slice(20, some_2d_mask, label=1)   # now those pixels on slice 20 are label 1
lm.undo()                                   # one call, undoes that paint
```

Every tool in `tools/` just computes *which pixels should change* (a bool
mask) — `select.py` finds them by intensity or growing, `morphology.py` by
shape, `propagate.py` by copying from a neighbouring slice. `LabelMap` is
the only thing that actually writes pixels, which is why undo/redo and
"don't overwrite another material" (`protect=True`) work the same way no
matter which tool you used.

## 5. Mapping to your 14-day roadmap

| Day | What to do | Where |
|---|---|---|
| 1 | Load + view your data | `examples/day1_quickstart.py` — already working |
| 2 | Write your Plan of Action | use section 3/4 above as a starting point |
| 3 | Understand `LabelMap` + save/load | `core/labelmap.py`, `fileio/files.py` |
| 4 | Brush/erase + undo | `tools/brush.py` — read `test_brush_stroke_paints_a_line` |
| 5 | Get the napari GUI running | `gui/napari_widget.py` (see §6 below — I could not test this myself) |
| 6 | ROI tools | `tools/roi.py` |
| 7 | Threshold / flood fill / region growing | `tools/select.py` |
| 8 | Morphology + connected components | `tools/morphology.py` |
| 9 | 3D brush, 3D region growing | pass `mode="3d"` to `brush.py`; `select.py` already works on 3D arrays |
| 10 | Slice propagation | `tools/propagate.py` — read step 8 and 9 of `validate_on_real_data.py` |
| 11 | Validation | `evaluate.py` — already wired into the demo script |
| 12 | Testing/polish | add cases to `tests/test_core.py` as you extend things |
| 13 | GitHub + README | this file is a starting point; add screenshots of your own runs |
| 14 | Presentation | the Dice numbers and before/after images from the demo script are good slides |

## 6. About the GUI

`ctseg/gui/napari_widget.py` opens the CT with napari's own Labels layer (paint, erase,
bucket fill, undo/redo) and adds tabs on the right: **Select** (threshold with live
preview, **multi-Otsu**, **multi-seed growing** `Ctrl+J`, flood fill `Ctrl+F`, largest/small components), **Brush** (round/square, 2D/3D, paint/erase; stamp at the cursor or tick *drag* to draw), **Edit** (dilate, erode, open, close, fill, slice
propagation, interpolation, thin structures: outline / interface), **ROI** (rectangle, ellipse, polygon, freehand, stats, JSON
save/load), **Labels** (names, colours, save/load project, Dice vs a reference folder), **View** (2D/3D switch, show only the active label, STL export of a label) and
**Help** (shortcuts). `Ctrl+G` region-grows at the cursor and `Ctrl+K` keeps the component under it.
The panels are tested offscreen (`tests/test_gui_panels.py`); the full window still needs
your own click-through on a real display:

```bash
python -m ctseg.gui.napari_widget data/set1/Assembly 2
```

Thin 1-2 px structures (Braze, Outer) are derived from the thick regions with
`outline` and `interface` (Edit tab, or `ctseg.tools.morphology`). To score them on your data:

```bash
python examples/evaluate_thin.py data/set1 2    # writes results/thin_results.md
```

## 7. Known limitations (worth putting in your Plan of Action)

- `region_grow_auto`'s automatic intensity range is estimated from a small
  neighborhood around your seed point. If that neighborhood straddles a
  material edge, the estimate breaks and it over-segments. Click well
  inside a material, or pass a smaller `radius`. (Found this while writing
  the tests — see `test_region_grow_auto_finds_a_blob_without_knowing_its_range`.)
- `propagate_slices` drifts slowly (real data, Dice 0.98 at +1 slice,
  0.94-0.96 at +20 after the ring-keeping fix); `interpolate_between_slices`
  stayed above 0.997 because it only needs the two key slices to be correct.
  Use interpolation when you can afford two keyframes; use propagation when you can't.
- `data/set2` (`Assembly_1.zip`) has mismatched image sizes between the
  assembly volume and the material masks, and the exports have UI text
  burned into the corners. Needs cropping/registration before it's usable
  the same way `data/set1` is.
- Thin 1-2 px structures (Braze, Outer) are not captured by intensity-range region growing (Dice 0.01 and 0.08).
- Ceramic and metal overlap in grey value on some slices, so ceramic growing can leak into the metal ring.

## 8. If you get stuck

Come back and paste the exact error message and which file/line it's on —
that's much faster for me to debug than a description of the symptom.

## 9. Reproduce the report results (train/test split)

```bash
pip install -r requirements.txt && pip install -e .
python scripts/make_train_test_split.py data/set1 submission_data   # makes training_data/ and testing_data/
python examples/run_experiments.py data/set1                        # writes results/results.json, results.md, figures
```
Slices 0-159 are used only to choose settings; slices 160-319 are used only for scoring. Summary tables: `results/results.md`. Plan of Action: `docs/PLAN_OF_ACTION.md`.

## 10. AI-assistance disclosure
This project was developed with an AI assistant (Claude). The author ran, tested and directed the work. See `docs/DECLARATION.md`.
