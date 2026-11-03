"""LabelMap = the mask volume + undo/redo.

Think of it as a 3D array the same shape as the CT volume, in (z, y, x) order.
Each voxel holds a number: 0 = background, 1 = label 1, 2 = label 2, ...

Every edit goes through ONE function, `_write`. It saves only the small box that
changed (not the whole volume), which is what makes undo/redo cheap.
"""
import numpy as np


class LabelMap:
    def __init__(self, shape, dtype=np.uint8, max_history=50):
        self.data = np.zeros(shape, dtype=dtype)
        self.max_history = max_history
        self._undo = []
        self._redo = []

    @property
    def shape(self):
        return self.data.shape

    # ------------------------------------------------------------------ core
    def _write(self, box, new_block):
        """Replace data[box] with new_block, remembering what changed."""
        view = self.data[box]                      # a view: writing to it edits self.data
        changed = new_block != view
        if not changed.any():
            return False                           # nothing to do -> no undo entry
        idx = np.nonzero(changed)
        sub = tuple(slice(i.min(), i.max() + 1) for i in idx)   # tight box around changes
        old = view[sub].copy()
        new = new_block[sub].copy()
        view[sub] = new
        full = tuple(slice(b.start + s.start, b.start + s.stop) for b, s in zip(box, sub))
        self._undo.append((full, old, new))
        self._undo = self._undo[-self.max_history:]
        self._redo.clear()
        return True

    def _clip(self, mask, origin):
        """Place `mask` at `origin`; cut off whatever sticks out of the volume."""
        lo = [max(o, 0) for o in origin]
        hi = [min(o + s, n) for o, s, n in zip(origin, mask.shape, self.shape)]
        if any(h <= l for l, h in zip(lo, hi)):
            return None, None
        box = tuple(slice(l, h) for l, h in zip(lo, hi))
        crop = tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, origin))
        return box, mask[crop].astype(bool)

    # ------------------------------------------------------------ paint/erase
    def paint_block(self, mask, origin, label, protect=False):
        """Set voxels under `mask` (3D bool array placed at `origin`) to `label`.
        protect=True -> never overwrite voxels that already belong to another label."""
        box, m = self._clip(mask, origin)
        if box is None:
            return False
        view = self.data[box]
        sel = m & ((view == 0) | (view == label)) if protect else m
        new = view.copy()
        new[sel] = label
        return self._write(box, new)

    def erase_block(self, mask, origin, only_label=None):
        """Set voxels under `mask` to 0. only_label=k -> erase only label k."""
        box, m = self._clip(mask, origin)
        if box is None:
            return False
        view = self.data[box]
        sel = m & (view == only_label) if only_label is not None else m
        new = view.copy()
        new[sel] = 0
        return self._write(box, new)

    def paint_slice(self, z, mask2d, label, protect=False):
        return self.paint_block(np.asarray(mask2d)[None], (z, 0, 0), label, protect)

    def erase_slice(self, z, mask2d, only_label=None):
        return self.erase_block(np.asarray(mask2d)[None], (z, 0, 0), only_label)

    def paint_volume(self, mask3d, label, protect=False):
        return self.paint_block(mask3d, (0, 0, 0), label, protect)

    def erase_volume(self, mask3d, only_label=None):
        return self.erase_block(mask3d, (0, 0, 0), only_label)

    # ------------------------------------------------- replace a whole label
    def set_label_block(self, label, mask, origin, protect=True):
        """Inside the block: remove `label`, then re-add it exactly where mask is True.
        Used by morphology and slice propagation (they compute a whole new shape)."""
        box, m = self._clip(mask, origin)
        if box is None:
            return False
        view = self.data[box]
        new = view.copy()
        new[view == label] = 0
        ok = m & (new == 0) if protect else m
        new[ok] = label
        return self._write(box, new)

    def set_label_mask(self, label, mask, z=None, protect=True):
        """z=None -> `mask` is 3D (whole volume). z=int -> `mask` is 2D (one slice)."""
        if z is None:
            return self.set_label_block(label, mask, (0, 0, 0), protect)
        return self.set_label_block(label, np.asarray(mask)[None], (z, 0, 0), protect)

    # ------------------------------------------------------------- undo/redo
    def undo(self):
        if not self._undo:
            return False
        box, old, new = self._undo.pop()
        self.data[box] = old
        self._redo.append((box, old, new))
        return True

    def redo(self):
        if not self._redo:
            return False
        box, old, new = self._redo.pop()
        self.data[box] = new
        self._undo.append((box, old, new))
        return True

    # ---------------------------------------------------------------- queries
    def get_mask(self, label, z=None):
        d = self.data if z is None else self.data[z]
        return d == label

    def labels_present(self):
        return [int(v) for v in np.unique(self.data) if v != 0]
