# Plan of Action - Interactive CT Segmentation and ROI Creation Tools

**Name:** Somiya   **Enrolment no.:** 25125036

## 1. Understanding
Build reusable, GUI-independent tools to create and edit masks and ROIs on 2D CT slices and 3D CT volumes, so they can later plug into a unified CT application (visualisation, quantification, AI modules).

## 2. Tools investigated
Brush/eraser (round/square, 2D disk and 3D sphere), ROI (rectangle, ellipse, polygon, freehand), threshold, flood fill, region growing (fixed range and automatic range), connected components, morphology (dilate, erode, open, close, fill holes), slice propagation, shape-based interpolation, multi-label management, undo/redo, save/load/export, Dice/IoU evaluation.

## 3. Existing software studied
3D Slicer Segment Editor (paint, erase, threshold, grow-from-seeds, fill-between-slices), napari Labels layer (paint, erase, bucket fill, undo), ITK/SimpleITK (connected-threshold and morphology filters), scikit-image/SciPy (the building blocks actually used here), Raya and Udupa 1990 (shape-based interpolation).

## 4. Paint and erase
A stroke is a list of voxel points; points are joined by interpolation so fast mouse motion leaves no gaps; a footprint (disk or sphere) is stamped along the path into a small local box; the box is applied to the label volume as ONE undo step.

## 5. ROI tools
Rectangle, ellipse, polygon, freehand, stored as their defining points (not pixels) in JSON and rasterised on demand. 2D ROIs belong to one slice. Planned: ROI-to-label helper and ROI list in the GUI.

## 6. Mask and label representation
One uint8 volume (z, y, x) where 0 is background and 1..255 are labels, plus a label table (id, name, colour, visibility). `protect=True` prevents overwriting other labels.

## 7. Morphology
Disk (2D) or ball (3D) structuring element; operations act on one label, in one slice or the whole volume, as one undo step.

## 8. 2D vs 3D focus
Both. Tools take 2D or 3D arrays; 2D editing for precision, 3D for speed (one click instead of one per slice - measured in the results).

## 9. Propagation between slices
(a) Walk slice by slice using the previous mask's intensity range inside a dilated neighbourhood; (b) shape-based interpolation of signed distance maps between two key slices.

## 10. Language and GUI
Python (NumPy, SciPy, scikit-image, tifffile) for the core; napari + magicgui for the optional viewer. C++/ITK would only be needed if profiling shows a bottleneck.

## 11. CPU/GPU
Everything is CPU and fast at this size (brush 9 ms, 3D region grow 0.25 s on 24 M voxels). GPU (CuPy/cuCIM) is worth testing only for full-resolution volumes.

## 12. Inputs and outputs
In: folder of 8-bit TIFF slices. Out: `labels.npy`, `labels.json` (label table), `rois.json`, multi-page label TIFF.

## 13. Integration into the unified application
`core/` and `tools/` never import any GUI library. The unified application imports `LabelMap` and the tool functions and supplies its own viewer.

## 14. Module structure
`ctseg/core` (LabelMap, LabelTable), `ctseg/tools` (brush, roi, select, morphology, propagate), `ctseg/fileio`, `ctseg/gui`, `ctseg/evaluate.py`, `tests/`, `examples/`, `scripts/`.

## 15. Challenges, limitations, assumptions
Thin structures (Braze, Outer) are not captured by intensity-range growing; ceramic/metal overlap in grey value causes leakage; ground truth are exported screenshots, not original label files; results come from one assembly; GUI needs wider user testing; set 2 needs registration before use.
