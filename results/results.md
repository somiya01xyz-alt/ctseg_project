# Results (data set 1, half resolution, volume 320 x 714 x 840)

Split by block: slices 0-159 = training (used only to choose settings), slices 160-319 = testing (used only to score).
Seeds ("user clicks") are simulated as the deepest interior point of the ground-truth mask. Slices with < 5,000 px of material are skipped.
Ground truth = the Ceramic/Metal/Braze/Outer exports supplied with the data.

## 1. Region growing (region_grow_auto), mean Dice (± std over slices)
| Material | Chosen k, radius | Train Dice | Test, default (k=2, r=4) | Test, tuned | Test, tuned + close + largest component |
|---|---|---|---|---|---|
| Ceramic | 5.0, 16 | 0.966 | 0.301 ± 0.248 | 0.751 ± 0.225 | 0.762 ± 0.230 |
| Metal   | 8.0, 4  | 0.924 | 0.624 ± 0.209 | 0.969 ± 0.022 | 0.971 ± 0.021 |
| Braze (thin line)  | default | - | 0.012 | - | - |
| Outer (thin line)  | default | - | 0.081 | - | - |

Ceramic shows a train-to-test gap (0.97 -> 0.75): on some slices the ceramic and the metal ring have overlapping grey values and the region leaks into the metal (see fig_region_grow.png).
Braze and Outer are 1-2 px lines; stride down-sampling and a plain intensity range cannot capture them.

## 2. 2D vs 3D region growing (Ceramic, slices 200-231)
| Approach | User clicks | Dice | Time |
|---|---|---|---|
| 2D, one click per slice | 32 | 0.888 | 1.41 s |
| 3D, one click | 1 | 0.960 | 0.25 s |

## 3. Propagation vs interpolation (key slice 200, Dice vs ground truth)
| Material | Propagation +1 / +5 / +10 / +20 | Interpolation between key slices 200 and 220 (worst slice) |
|---|---|---|
| Ceramic | 0.981 / 0.955 / 0.958 / 0.956 | 0.998 |
| Metal   | 0.981 / 0.956 / 0.941 / 0.934 | 0.997 |

## 4. Data structure (LabelMap, 192 MB volume)
Brush stroke (100 points, radius 10): 9 ms. Undo entry: 52.6 KB (0.03 % of the volume) instead of a 192 MB copy. Undo 13 us, redo 4 us.
3D brush stroke: 6 ms.
