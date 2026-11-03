# Thin structures from thick regions (held-out slices 160-319)

Tolerant F1 uses a 2-px tolerance. Thickness chosen on slices 0-159, using only slices where the ground truth is a ring (not an annotation rectangle).

| Structure | Thickness | Dice (all) | F1 (all) | Dice (ring slices) | F1 (ring slices) | ring slices |
|---|---|---|---|---|---|---|
| Braze | 4 | 0.292 | 0.466 | 0.577 | 0.919 | 79 |
| Outer (inside edge) | 1 | 0.000 | 0.741 | 0.000 | 0.877 | 125 |
| Outer (just outside) | 3 | 0.535 | 0.796 | 0.633 | 0.942 | 125 |
